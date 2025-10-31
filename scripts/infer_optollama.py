
#!/usr/bin/env python3
"""
Minimal, reusable inference & validation script for OptoLlama/DiT.

- Reuses the existing `validate_model` function for consistent metrics.
- Works from CLI *and* inside Spyder/Jupyter by exposing a `run_inference(...)` function.
- Handles both single-process and DDP runs (but defaults to single-process for IDE use).
- Loads checkpoints saved by your training script (supports a few common layouts).

Usage (CLI):
    python infer_optollama.py --config config_MD58.py --validsim TMM_FAST --ckpt /path/to/checkpoint.pt

Usage (Python/Spyder):
    import infer_optollama as infer
    infer.run_inference(config="config_MD58", validsim="TMM_FAST")

Outputs:
    - Prints mean accuracy/MAE.
    - Saves per-example JSON via utils.save_JSONPICKLE(...) into cfg.PATH_SAVED.
"""
from __future__ import annotations

import os
import importlib
from typing import Any, Dict, Optional, Tuple

import torch

# Project imports (local files in the repo)
import cli
from utils import init_tokenmaps, save_JSONPICKLE
from cli import parse_arguments
from utils_runner import setup_run, _is_ddp
from utils_data import make_loader, SpectraDataset
from utils_model import build_model
from utils_eval import build_tmm_context, validate_model


# ----------------------------- helpers -----------------------------
def _flex_load_state_dict(model: torch.nn.Module, state: Dict[str, Any]) -> None:
    """
    Try several common checkpoint layouts and load into (possibly DDP-wrapped) model.
    """
    from utils import core_module_crop, load_state_dict_flexible
    core = core_module_crop(model)

    # Training script stored {'model_state': ..., 'optimizer_state': ..., ...}
    if isinstance(state, dict) and "model_state" in state:
        sd = state["model_state"]
        load_state_dict_flexible(core, sd)
        return

    # utils.save_checkpoint(state={'model': ...})
    if isinstance(state, dict) and "model" in state:
        sd = state["model"]
        load_state_dict_flexible(core, sd)
        return

    # Maybe the file *is* a plain state_dict already
    if isinstance(state, dict) and any(torch.is_tensor(v) for v in state.values()):
        load_state_dict_flexible(core, state)
        return

    raise RuntimeError("Unrecognized checkpoint format; expected keys 'model_state' or 'model'.")


def _maybe_load_checkpoint(model: torch.nn.Module, ckpt_path: Optional[str], device: str) -> Tuple[torch.nn.Module, Optional[int]]:
    """
    Load a checkpoint if provided and return (model, start_epoch) where epoch may be None.
    """
    if not ckpt_path:
        return model, None

    if not os.path.exists(ckpt_path):
        print(f"⚠️  Checkpoint not found: {ckpt_path}")
        return model, None

    print(f"📦 Loading checkpoint: {ckpt_path}")
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # if it's a dict with epoch, keep it; otherwise None
    start_epoch = None
    if isinstance(blob, dict) and "epoch" in blob and isinstance(blob["epoch"], (int, float)):
        start_epoch = int(blob["epoch"])

    try:
        _flex_load_state_dict(model, blob)
    except Exception as e:
        print(f"⚠️  Failed to load model weights from checkpoint: {e}")

    # move to device (safe if already there)
    return model.to(device), start_epoch


def _build_model_for_cfg(cfg, example_spectrum, vocab_size, max_stack_depth, token_meta, device: str) -> torch.nn.Module:
    model = build_model(
        model_type=getattr(cfg, "ARCH", getattr(cfg, "OL_MODEL", "dit")),
        sample_spectrum=example_spectrum,
        vocab_size=vocab_size,
        max_stack_depth=max_stack_depth,
        d_model=int(getattr(cfg, "D_MODEL", 1024)),
        n_blocks=int(getattr(cfg, "N_BLOCKS", 6)),
        n_heads=int(getattr(cfg, "N_HEADS", 8)),
        timesteps=int(getattr(cfg, "STEPS", 500)),
        dropout=float(getattr(cfg, "DROPOUT", 0.0)),
        idx_to_token=token_meta["idx_to_token"],
        mask_idx=token_meta["msk_idx"],
        pad_idx=token_meta["pad_idx"],
        eos_idx=token_meta["eos_idx"],
        device=device,
    )
    # sampling knobs (optional)
    temp = float(getattr(cfg, "TEMPERATURE", 0.0))
    top_k = int(getattr(cfg, "TOP_K", 0))
    top_p = float(getattr(cfg, "TOP_P", 0.0))
    if hasattr(model, "set_sampling"):
        model.set_sampling(temperature=temp, top_k=top_k, top_p=top_p)
    return model


def _ensure_dirs(cfg):
    # Only create PATH_SAVED if present
    path_saved = getattr(cfg, "PATH_SAVED", None)
    if path_saved:
        os.makedirs(path_saved, exist_ok=True)


# ----------------------------- core API -----------------------------

@torch.no_grad()
def run_inference(
    config: str = "config_MD58",
    validsim: str = "TMM_FAST",
    ckpt: Optional[str] = None,
    mc_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    High-level, Spyder-friendly entry point. Returns a results dict:
        {'mean_acc': float, 'mean_mae': float|None, 'results': list}
    """
    name = os.path.splitext(os.path.basename(config))[0]
    cfg = importlib.import_module(name)

    # Set up device / (optional) DDP. Single-process path works great in IDEs.
    device, local_rank, rank, world_size = setup_run(cfg, make_dirs=False)
    is_ddp = _is_ddp()

    # Tokens
    tokens, token_to_idx, idx_to_token, EOS_TOKEN, PAD_TOKEN, MSK_TOKEN, \
        eos_idx, pad_idx, msk_idx = init_tokenmaps(cfg.PATH_DATA)
    T = {
        "tokens": tokens,
        "token_to_idx": token_to_idx,
        "idx_to_token": idx_to_token,
        "EOS_TOKEN": EOS_TOKEN,
        "PAD_TOKEN": PAD_TOKEN,
        "MSK_TOKEN": MSK_TOKEN,
        "eos_idx": eos_idx,
        "pad_idx": pad_idx,
        "msk_idx": msk_idx,
    }
    # Loader
    setattr(cfg, "BATCH_SIZE", int(getattr(cfg, "VALID_BATCH", getattr(cfg, "BATCH_SIZE", 64))))
    valid_subset_n = int(getattr(cfg, "NUM_SAMPLES_VALID", 0) or 0) or None
    valid_ds, valid_loader, _ = make_loader(cfg, split="valid", subset_n=valid_subset_n, ddp=is_ddp)
    # Build model
    example_spec = valid_ds.spectra[0] if isinstance(valid_ds, SpectraDataset) else valid_ds.dataset.spectra[0]
    vocab_size = len(T["idx_to_token"])
    max_stack = valid_ds.maximum_depth if isinstance(valid_ds, SpectraDataset) else valid_ds.dataset.maximum_depth
    model = _build_model_for_cfg(cfg, example_spec, vocab_size, max_stack, T, device=device)

    # Load weights (ckpt argument wins; else try cfg.PATH_CHKPT)
    ckpt_path = ckpt or getattr(cfg, "PATH_CHKPT", None)
    model, _ = _maybe_load_checkpoint(model, ckpt_path, device)

    # Optional TMM context
    tmm_ctx = None
    mode = (validsim or "NOSIM").upper()
    if mode == "TMM_FAST":
        try:
            tmm_ctx = build_tmm_context(cfg=cfg, idx_to_token=T["idx_to_token"], device=device)
        except Exception as e:
            print(f"⚠️  Could not initialize TMM context, falling back to NOSIM: {e}")
            mode = "NOSIM"

    # MC samples
    samples = int(mc_samples) if mc_samples is not None else int(getattr(cfg, "MC_SAMPLES", 1) or 1)

    # Validate / infer
    out = validate_model(
        model,
        valid_loader,
        mode=mode,
        eos=T["eos_idx"], pad=T["pad_idx"], msk=T["msk_idx"],
        device=device,
        idx_to_token=T["idx_to_token"],
        tmm_ctx=tmm_ctx,
        mc_samples=samples,
        rank=rank,
        world_size=world_size,
        gather=True,
    )

    # Persist per-example results on rank 0 (or single-process)
    if (not is_ddp) or rank == 0:
        _ensure_dirs(cfg)
        try:
            save_JSONPICKLE(cfg.PATH_SAVED, out.get("results", []), f"results_{getattr(cfg,'RUN_NAME','infer')}")
            print(f"💾 Saved {len(out.get('results', []))} records to {cfg.PATH_SAVED}")
        except Exception as e:
            print(f"⚠️  Could not save results JSON: {e}")

    # Summary
    mean_acc = out.get("mean_acc", None)
    mean_mae = out.get("mean_mae", None)
    if mean_acc is not None:
        print(f"✔ mean token accuracy: {100.0*float(mean_acc):.2f}%")
    if mean_mae is not None:
        print(f"✔ mean spectrum MAE: {float(mean_mae):.6f}")

    return out


# ----------------------------- CLI entry -----------------------------
if __name__ == "__main__":
    import sys
    sys.argv.extend(["--config", "config_MD58"])
    args = cli.parse_arguments()
    run_inference(
        config=args.config,
        validsim=args.validsim,
        ckpt=args.ckpt,
        mc_samples=args.mc_samples,
    )
