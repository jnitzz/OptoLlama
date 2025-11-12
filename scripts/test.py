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

import os, sys
import torch
import importlib
from typing import Any, Dict, Optional

# Project imports (local files in the repo)
import cli
from utils import init_tokenmaps, load_as_json, load_checkpoint, apply_sampling_from_sources
from inference import load_spectra_from_json_or_csv
from evaluate import validate_model
from runner import setup_run, _is_ddp
from dataset import make_loader, SpectraDataset, make_repeated_spec_loader
from model import build_model
from simulation_TMM_FAST import build_tmm_context

@torch.no_grad()
def run_inference(
    config: str = "config_MD58",
    validsim: str = "TMM_FAST",
    ckpt: Optional[str] = None,
    mc_samples: Optional[int] = None,
    target: Optional[str] = None,
    n_targets: Optional[int] = None,   # NEW
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

    # --- always build model from a dataset sample to lock dims like training ---
    setattr(cfg, "BATCH_SIZE", int(getattr(cfg, "VALID_BATCH", getattr(cfg, "BATCH_SIZE", 64))))
    valid_subset_n = int(getattr(cfg, "NUM_SAMPLES_VALID", 0) or 0) or None
    valid_ds, valid_loader, _ = make_loader(cfg, split="valid", subset_n=valid_subset_n, ddp=is_ddp)
    
    # figure out W (number of wavelength samples) from cfg or dataset
    if hasattr(cfg, "WAVELENGTHS"):
        W = len(cfg.WAVELENGTHS)
    else:
        ex = valid_ds.spectra[0] if isinstance(valid_ds, SpectraDataset) else valid_ds.dataset.spectra[0]
        W = int(ex.shape[0])  # fallback
    
    # 🔧 ALWAYS use a stub [3, W] as the example for model sizing
    example_spec_for_build = torch.zeros((3, W), dtype=torch.float32, device=device)
    
    vocab_size = len(idx_to_token)
    max_stack  = valid_ds.maximum_depth if isinstance(valid_ds, SpectraDataset) else valid_ds.dataset.maximum_depth
    
    model = build_model(
        model_type=getattr(cfg, "ARCH", getattr(cfg, "OL_MODEL", "dit")),
        sample_spectrum=example_spec_for_build,
        vocab_size=vocab_size,
        max_stack_depth=max_stack,
        d_model=int(getattr(cfg, "D_MODEL", 1024)),
        n_blocks=int(getattr(cfg, "N_BLOCKS", 6)),
        n_heads=int(getattr(cfg, "N_HEADS", 8)),
        timesteps=int(getattr(cfg, "STEPS", 500)),
        dropout=float(getattr(cfg, "DROPOUT", 0.0)),
        idx_to_token=idx_to_token,
        mask_idx=msk_idx,
        pad_idx=pad_idx,
        eos_idx=eos_idx,
        device=device,
    )
    # remove the local temp/top_k/top_p extraction and do:
    apply_sampling_from_sources(model, args=None, cfg=cfg)
    

    # Load weights (ckpt argument wins; else try cfg.PATH_CHKPT)
    ckpt_path = ckpt or getattr(cfg, "PATH_CHKPT", None)
    ckpt_path_best = ckpt_path.split(".")[0] + '_best.' + ckpt_path.split(".")[1]
    if ckpt_path_best and os.path.exists(ckpt_path_best):
        print(f"📦 Loading checkpoint: {ckpt_path_best}")
        _, _ = load_checkpoint(ckpt_path_best, model, map_location="cpu", strict=True)
    elif ckpt_path and os.path.exists(ckpt_path):
        print(f"📦 Loading checkpoint: {ckpt_path}")
        _, _ = load_checkpoint(ckpt_path, model, map_location="cpu", strict=True)
    else:
        print("No valid ckpt path set")    
    
    # MC samples
    samples = int(mc_samples) if mc_samples is not None else int(getattr(cfg, "MC_SAMPLES", 1) or 1)
    
    # How many designed targets?
    n_tar = int(n_targets) if (n_targets is not None) else int(getattr(cfg, "N_TARGETS", 1))

    
    # Build loader depending on branch
    if target is None:
        loader = valid_loader
    else:
        # Load base [3,W] spectrum with noise/smoothing DISABLED here;
        # fresh noise (if enabled) will be drawn per item by the repeated dataset.
        base_spec = load_spectra_from_json_or_csv(
            target,
            expect_shape="3xW",
            cfg=cfg,
            noise_cfg={"enabled": False},
            smooth_cfg={"enabled": False},
        ).to(device)
    
        if n_tar > 0:
            _, loader, _ = make_repeated_spec_loader(
                base_spec, n_tar, max_stack_depth=max_stack, pad_idx=pad_idx,
                wavelengths=getattr(cfg, "WAVELENGTHS"),
                cfg=cfg,
                batch_size=min(n_tar, getattr(cfg, "VALID_BATCH", 64)),
            )
        else:
            raise ValueError(f'use n_tar > 0. Got {n_tar}')
            
    # Optional TMM context
    tmm_ctx = None
    mode = (validsim or "NOSIM").upper()
    if mode == "TMM_FAST":
        try:
            tmm_ctx = build_tmm_context(cfg=cfg, idx_to_token=idx_to_token, device=device)
        except Exception as e:
            print(f"⚠️  Could not initialize TMM context, falling back to NOSIM: {e}")
            mode = "NOSIM"

    # Run validate_model — for a single item, it will still return the generated stack(s)
    out = validate_model(
        model,
        loader,
        mode=mode,
        eos=eos_idx, pad=pad_idx, msk=msk_idx,
        device=device,
        idx_to_token=idx_to_token,
        tmm_ctx=tmm_ctx,
        mc_samples=samples,
        rank=rank,
        world_size=world_size,
        gather=True,
    )

    # Persist results on rank 0 (or single-process)
    if (not is_ddp) or rank == 0:
        path_saved = getattr(cfg, "PATH_SAVED", None)
        if path_saved:
            os.makedirs(path_saved, exist_ok=True)
        try:
            base = getattr(cfg, 'RUN_NAME', 'infer')
            tag  = "target" if target else "valid"
            load_as_json(cfg.PATH_SAVED, out.get("results", []), f"results_{base}_{tag}")
            print(f"💾 Saved {len(out.get('results', []))} record(s) to {cfg.PATH_SAVED}")
        except Exception as e:
            print(f"⚠️  Could not save results JSON: {e}")

    # Summary (accuracy/MAE may be None for single target without ground-truth)
    if out.get("mean_acc") is not None:
        print(f"✔ mean token accuracy: {100.0*float(out['mean_acc']):.2f}%")
    if out.get("mean_mae") is not None:
        print(f"✔ mean spectrum MAE: {float(out['mean_mae']):.6f}")

    return out


# ----------------------------- CLI entry -----------------------------
if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", "config_MD63.py"])                             #TODO rename to better name
        # sys.argv.extend(["--config", "config_MD64.py"])                             #TODO rename to better name
    if "--target" not in sys.argv:
        sys.argv.extend(["--target",
        # "d:/Profile/a3536/Eigene Dateien/GitHub/ColorAppearanceToolbox/Diffusion/data/TF_MA2_safetensors/test_bandpass.json"])
        # "d:/Profile/a3536/Eigene Dateien/GitHub/ColorAppearanceToolbox/Diffusion/data/TF_MA2_safetensors/test_bandstop.json"])
        # "d:/Profile/a3536/Eigene Dateien/GitHub/ColorAppearanceToolbox/Diffusion/data/TF_MA2_safetensors/test_gaussian_peak.json"])
        "d:/Profile/a3536/Eigene Dateien/GitHub/ColorAppearanceToolbox/Diffusion/data/TF_MA2_safetensors/test2.json"])
        # "d:/Profile/a3536/Eigene Dateien/GitHub/ColorAppearanceToolbox/Diffusion/data/TF_MA2_safetensors/testfile.csv"])
        
    # Parse args and build final config (applies --ckpt/--mc-samples/--validsim and --set)
    args = cli.parse_arguments()
    n_targets = getattr(args, "n_targets", None)
    cfg = cli.load_config_with_overrides(args)
    
    out = run_inference(
        config=args.config,
        validsim=cfg.VALIDSIM,
        ckpt=args.ckpt,
        mc_samples=cfg.MC_SAMPLES,
        target=args.target,
        n_targets=n_targets or getattr(cfg, "N_TARGETS", 1),
    )
    from plots import plot_samples
    valkey = min([[item['mae'],i] for i, item in enumerate(out['results'])])
    plot_samples(cfg, out['results'][valkey[1]]['rat_pred_flat'], out['results'][valkey[1]]['rat_target_flat'], out['results'][valkey[1]]['stack_pred_tokens'], '', 0, cfg.MC_SAMPLES, RAT_tar_mean = None)
        
