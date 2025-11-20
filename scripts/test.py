from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

import cli
import torch
from dataset import SpectraDataset, make_loader, make_repeated_spec_loader
from evaluate import validate_model
from inference import load_spectra_from_json_or_csv
from model import build_model
from runner import _is_ddp, setup_run
from simulation_TMM_FAST import build_tmm_context
from utils import init_tokenmaps, load_checkpoint, save_as_json

# ruff: noqa: N806


@torch.no_grad()
def run_inference(
    cfg: Any,
    validsim: Optional[str] = "TMM_FAST",
    ckpt: Optional[str] = None,
    mc_samples: Optional[int] = None,
    target: Optional[str] = None,
    n_targets: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run model inference / validation for a given configuration.

    This function sets up the runtime, builds the model, optionally loads a
    checkpoint, constructs the appropriate data loader (validation split or a
    user-provided target spectrum), and finally calls :func:`evaluate.validate_model`.

    Parameters
    ----------
    cfg:
        Configuration object with attributes such as
        ``PATH_DATA``, ``WAVELENGTHS``, ``PATH_CKPT``, ``PATH_SAVED``,
        ``MODEL_KEY``, ``D_MODEL``, ``N_BLOCKS``, ``N_HEADS``,
        ``STEPS``, ``DROPOUT``, ``MC_SAMPLES``, ``N_TARGETS``,
        ``NUM_SAMPLES_VALID``, ``VALID_BATCH``, ``MISMATCH_FILL_ORDER``, etc.
    validsim:
        Name of the simulator mode used inside validation. Typical values are
        ``"TMM_FAST"`` (full optical simulation) or ``"NOSIM"`` (token metrics
        only). If ``None``, defaults to ``"TMM_FAST"``.
    ckpt:
        Optional checkpoint path overriding ``cfg.PATH_CKPT``. If provided and
        the file (or its ``*_best`` variant) exists, the model weights are
        loaded from there.
    mc_samples:
        Number of Monte-Carlo samples used in :func:`validate_model` (best-of-N
        decoding). If ``None``, falls back to ``cfg.MC_SAMPLES`` (default 1).
    target:
        Optional path to a JSON/CSV RAT file describing a *target* spectrum
        [R, A, T] over wavelength. When provided, the function builds a loader
        that repeats this base spectrum instead of using the validation set.
        When ``None``, the standard validation loader is used.
    n_targets:
        Number of repeated target samples to generate when ``target`` is given.
        If ``None``, falls back to ``cfg.N_TARGETS`` (default 1). Must be
        strictly positive when ``target`` is not ``None``.

    Returns
    -------
    Dict[str, Any]
        A dictionary with at least the following keys:

        - ``"mean_acc"`` (float): mean token accuracy over the (sub)dataset.
        - ``"mean_mae"`` (float | None): mean spectrum MAE (only for
          simulator modes that produce spectra, e.g. ``"TMM_FAST"``).
        - ``"results"`` (list[dict]): list of per-example records, including
          token stacks and RAT spectra. Present only on rank 0
          in DDP runs; in single-process runs it is always present.

    Notes
    -----
    - The function handles both single-process and DDP runs via
      :func:`runner.setup_run` and :func:`runner._is_ddp`.
    - Per-example results are saved as JSON in ``cfg.PATH_SAVED`` (if set),
      with a filename of the form ``results_{cfg.RUN_NAME}_{tag}.json`` where
      ``tag`` is ``"valid"`` or ``"target"``.
    """
    if cfg is None:
        raise ValueError("cfg must not be None; pass a loaded config object.")

    # Set up device / (optional) DDP. Single-process path works great in IDEs.
    device, local_rank, rank, world_size = setup_run(cfg, make_dirs=False)
    is_ddp = _is_ddp()

    # Tokens
    tokens, token_to_idx, idx_to_token, EOS_TOKEN, PAD_TOKEN, MSK_TOKEN, eos_idx, pad_idx, msk_idx = init_tokenmaps(cfg.PATH_DATA)

    # --- always build model from a dataset sample to lock dims like training ---
    valid_subset_n = getattr(cfg, "NUM_SAMPLES_VALID")
    valid_ds, valid_loader, _ = make_loader(cfg, split="valid", subset_n=valid_subset_n, ddp=is_ddp)

    # figure out W (number of wavelength samples) from cfg or dataset
    if hasattr(cfg, "WAVELENGTHS"):
        w = len(cfg.WAVELENGTHS)
    else:
        ex = valid_ds.spectra[0] if isinstance(valid_ds, SpectraDataset) else valid_ds.dataset.spectra[0]
        w = int(ex.shape[0])  # fallback

    # ALWAYS use a stub [3, W] as the example for model sizing
    example_spec_for_build = torch.zeros((3, w), dtype=torch.float32, device=device)

    vocab_size = len(idx_to_token)
    max_stack = valid_ds.maximum_depth if isinstance(valid_ds, SpectraDataset) else valid_ds.dataset.maximum_depth

    model = build_model(
        model_type=getattr(cfg, "MODEL_KEY"),
        sample_spectrum=example_spec_for_build,
        vocab_size=vocab_size,
        max_stack_depth=max_stack,
        d_model=getattr(cfg, "D_MODEL"),
        n_blocks=getattr(cfg, "N_BLOCKS"),
        n_heads=getattr(cfg, "N_HEADS"),
        timesteps=getattr(cfg, "STEPS"),
        dropout=getattr(cfg, "DROPOUT"),
        idx_to_token=idx_to_token,
        mask_idx=msk_idx,
        pad_idx=pad_idx,
        eos_idx=eos_idx,
        device=device,
        temperature=getattr(cfg, "TEMPERATURE"),
        top_k=getattr(cfg, "TOP_K"),
        top_p=getattr(cfg, "TOP_P"),
    )

    # Load weights (ckpt argument wins; else try cfg.PATH_CKPT)
    ckpt_path = cfg.PATH_CKPT
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        _, _ = load_checkpoint(ckpt_path, model, map_location="cpu", strict=True)
    else:
        print("No valid ckpt path set")

    # How many designed targets?
    n_tar = cfg.N_TARGETS
    if n_tar <= 0:
        print(f"use n_tar > 0. Got {n_tar}. n_tar = 1 used.")
        n_tar = 1

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
            mismatch_order=cfg.MISMATCH_FILL_ORDER,
        ).to(device)

        _, loader, _ = make_repeated_spec_loader(
            base_spec,
            n_tar,
            max_stack_depth=max_stack,
            pad_idx=pad_idx,
            wavelengths=getattr(cfg, "WAVELENGTHS"),
            cfg=cfg,
            batch_size=min(n_tar, getattr(cfg, "VALID_BATCH", 64)),
        )

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
        eos=eos_idx,
        pad=pad_idx,
        msk=msk_idx,
        device=device,
        idx_to_token=idx_to_token,
        tmm_ctx=tmm_ctx,
        mc_samples=cfg.MC_SAMPLES,
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
            base = getattr(cfg, "RUN_NAME", "infer")
            tag = "target" if target else "valid"
            save_as_json(cfg.PATH_SAVED, out.get("results", []), f"results_{base}_{tag}")
            print(f"💾 Saved {len(out.get('results', []))} record(s) to {cfg.PATH_SAVED}")
        except Exception as e:
            print(f"⚠️  Could not save results JSON: {e}")

    # Summary (accuracy/MAE may be None for single target without ground-truth)
    if out.get("mean_acc") is not None:
        print(f"✔ mean token accuracy: {100.0 * float(out['mean_acc']):.2f}%")
    if out.get("mean_mae") is not None:
        print(f"✔ mean spectrum MAE: {float(out['mean_mae']):.6f}")

    return out


# ----------------------------- CLI entry -----------------------------
if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", "config_OL_LOCAL.yaml"])

    # Parse args and build final config (applies --ckpt/--mc-samples/--validsim and --set)
    args = cli.parse_arguments()
    cfg = cli.load_config_with_overrides(args)

    out = run_inference(
        cfg=cfg,
        validsim=cfg.VALIDSIM,
        ckpt=cfg.PATH_CKPT,
        mc_samples=cfg.MC_SAMPLES,
        target=cfg.TARGET,
        n_targets=cfg.N_TARGETS,
    )
    # from evaluate import masked_mae
    # from plots import plot_samples

    # tar = torch.tensor([out["results"][0]["rat_target"]])
    # valkey = min([[masked_mae(torch.tensor([out["results"][i]["rat_pred"]]), tar), i] for i, item in enumerate(out["results"])])
    # for key in range(len(out["results"])):
    #     plot_samples(
    #         cfg,
    #         out["results"][key]["rat_pred_flat"],
    #         out["results"][key]["rat_target_flat"],
    #         out["results"][key]["stack_pred_tokens"],
    #         "",
    #         0,
    #         cfg.MC_SAMPLES,
    #         RAT_tar_mean=None,
    #     )
