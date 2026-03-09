from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Tuple

import torch

import optollama.scripts.cli as cli
from optollama.dataloader.dataset import SpectraDataset, make_loader, make_repeated_spec_loader
from optollama.model.model import build_model
from optollama.scripts.evaluate import validate_model
from optollama.scripts.inference import load_spectra_from_json_or_csv
from optollama.scripts.runner import _is_ddp, setup_run
from optollama.utils.simulation_TMM_FAST import build_tmm_context
from optollama.utils.utils import init_tokenmaps, load_checkpoint, save_as_json, wl_mask

# ruff: noqa: N806

# --- Material group helpers (keywords -> token ids) ---


def base_material_name(token: str) -> str:
    """Extract a base material name from a token string.

    Robust to common formats where thickness/parameters are appended, e.g.
    "SiO2_120", "TiN(30nm)", etc. Adjust regex if your token format differs.
    """
    return token.split("_", 1)[0]


def build_base_to_ids(token_to_idx: dict[str, int], special: set[str]) -> dict[str, list[int]]:
    """Build a base material index."""
    base_to_ids: dict[str, list[int]] = {}
    for tok, tid in token_to_idx.items():
        if tok in special:
            continue
        base = base_material_name(tok)
        base_to_ids.setdefault(base, []).append(int(tid))
    return base_to_ids


def expand_material_or_token_to_ids(items: list, token_to_idx: dict[str, int], base_to_ids: dict[str, list[int]]) -> torch.Tensor:
    """Expand filtered materials or tokens to ids.

    - int -> token id
    - str -> exact token if exists (e.g. "SiO2_120")
           -> otherwise base material (e.g. "SiO2" expands to all SiO2_*)
    """
    ids: list[int] = []
    for x in items:
        if isinstance(x, int):
            ids.append(int(x))
        elif isinstance(x, str):
            s = x.strip()
            if s in token_to_idx:
                ids.append(int(token_to_idx[s]))
            elif s in base_to_ids:
                ids.extend(base_to_ids[s])
            else:
                raise ValueError(
                    f"Unknown token/material in filter list: {s!r}. Use exact token like 'SiO2_120' or base material like 'SiO2'."
                )
        else:
            raise TypeError(f"Filter entries must be int or str, got: {type(x)}")

    if not ids:
        return torch.empty((0,), dtype=torch.long)
    return torch.unique(torch.tensor(ids, dtype=torch.long))


def build_group_token_ids(tokens: list[str], token_to_idx: dict[str, int]) -> dict[str, torch.Tensor]:
    """Build predefined material-group id sets.

    Groups:
      - metals: Ag, Al, TiN
      - semiconductors: Ge, ITO, Si, ZnO, ZnS, ZnSe
      - dielectrics: remaining non-special tokens
    """
    metals = {"Ag", "Al", "TiN"}
    semis = {"Ge", "ITO", "Si", "ZnO", "ZnS", "ZnSe"}
    special = {"<PAD>", "<MSK>", "<EOS>"}

    metal_toks: list[str] = []
    semi_toks: list[str] = []
    other_toks: list[str] = []

    for t in tokens:
        if t in special:
            continue
        b = base_material_name(t)
        if b in metals:
            metal_toks.append(t)
        elif b in semis:
            semi_toks.append(t)
        else:
            other_toks.append(t)

    def _ids(xs: list[str]) -> torch.Tensor:
        return torch.tensor([int(token_to_idx[x]) for x in xs], dtype=torch.long)

    return {
        "metals": _ids(metal_toks),
        "semiconductors": _ids(semi_toks),
        "dielectrics": _ids(other_toks),
    }


@torch.no_grad()
def run_inference(
    cfg: Any,
    ckpt: Optional[str] = None,
    mc_samples: Optional[int] = None,
    target: Optional[str] = None,
    n_targets: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[int, str], int, int, int]:
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

    special = {EOS_TOKEN, PAD_TOKEN, MSK_TOKEN}
    base_to_ids = build_base_to_ids(token_to_idx, special)

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
        sample_spectrum=example_spec_for_build,  # [3,W]
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

    # ---------------- Token/material constraints (target-spectrum relevant) ----------------
    # In the "target spectrum" branch, stacks in the dataset are dummy PADs, so dataset filtering
    # does not apply. Instead, we constrain what the model can *emit* during sampling by masking
    # logits inside OptoLlama._sample_logits.
    if bool(getattr(cfg, "TOKEN_FILTER_ENABLED", False)):
        mode = str(getattr(cfg, "TOKEN_FILTER_MODE", "exclude")).lower().strip()
        groups = list(getattr(cfg, "TOKEN_FILTER_GROUPS", []) or [])
        allow_toks = list(getattr(cfg, "TOKEN_FILTER_ALLOW_TOKENS", []) or [])
        exclude_toks = list(getattr(cfg, "TOKEN_FILTER_EXCLUDE_TOKENS", []) or [])
        allow_msk = bool(getattr(cfg, "TOKEN_FILTER_ALLOW_MSK", False))
        allow_eos_pad = bool(getattr(cfg, "TOKEN_FILTER_ALLOW_EOS_PAD", True))

        group_ids = build_group_token_ids(tokens, token_to_idx)

        # Expand groups into ids
        allow_group_ids: list[torch.Tensor] = []
        exclude_group_ids: list[torch.Tensor] = []
        for g in groups:
            gl = str(g).lower().strip()
            if gl not in group_ids:
                raise ValueError(f"Unknown TOKEN_FILTER_GROUPS entry: {g!r}. Use one of {list(group_ids.keys())}.")
            # In allow mode, groups are treated as allowlists. In exclude mode, treated as blocklists.
            (allow_group_ids if mode == "allow" else exclude_group_ids).append(group_ids[gl])

        allow_ids = torch.cat(allow_group_ids, dim=0) if allow_group_ids else torch.empty((0,), dtype=torch.long)
        exclude_ids = torch.cat(exclude_group_ids, dim=0) if exclude_group_ids else torch.empty((0,), dtype=torch.long)

        # Merge explicit token lists
        if allow_toks:
            allow_ids = torch.unique(
                torch.cat([allow_ids, expand_material_or_token_to_ids(allow_toks, token_to_idx, base_to_ids)], dim=0)
            )
        if exclude_toks:
            exclude_ids = torch.unique(
                torch.cat([exclude_ids, expand_material_or_token_to_ids(exclude_toks, token_to_idx, base_to_ids)], dim=0)
            )

        if mode == "allow" and allow_ids.numel() == 0:
            raise ValueError(
                "TOKEN_FILTER_MODE='allow' requires TOKEN_FILTER_GROUPS and/or TOKEN_FILTER_ALLOW_TOKENS to be non-empty."
            )

        # Apply to model if supported (OptoLlama)
        if hasattr(model, "set_token_constraints"):
            model.set_token_constraints(
                allow_ids=allow_ids if allow_ids.numel() > 0 else None,
                exclude_ids=exclude_ids if exclude_ids.numel() > 0 else None,
                allow_eos_pad=allow_eos_pad,
                allow_msk=allow_msk,
            )
            print(f"Token constraints enabled (mode={mode}, allow={int(allow_ids.numel())}, exclude={int(exclude_ids.numel())}).")
        else:
            print("Warning: model does not support token constraints; TOKEN_FILTER_* will be ignored.")

    # Optional: hard cap on generated solution length (target-spectrum relevant)
    # This is an inference-time decoding constraint, not a model architecture change.
    max_emit_len = getattr(cfg, "MAX_EMIT_LEN", None)
    if max_emit_len is not None:
        if hasattr(model, "set_max_emit_len"):
            model.set_max_emit_len(int(max_emit_len))
            print(f"MAX_EMIT_LEN enabled: {int(max_emit_len)}")
        else:
            print("Warning: model does not support MAX_EMIT_LEN; ignoring.")

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
        if target == "random":
            base_spec = torch.rand([3, 171], device=device)
        else:
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
    if cfg.VALIDSIM == "TMM_FAST":
        try:
            tmm_ctx = build_tmm_context(cfg=cfg, idx_to_token=idx_to_token, device=device)
        except Exception as e:
            print(f"Could not initialize TMM context, falling back to NOSIM: {e}")
            cfg.VALIDSIM = "NOSIM"

    # Run validate_model — for a single item, it will still return the generated stack(s)
    out = validate_model(
        model,
        loader,
        mode=cfg.VALIDSIM,
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
        track_step_mae=cfg.TRACK_STEP_MAE,
        roi_mask=wl_mask(cfg.WAVELENGTHS, cfg.ROI_MIN, cfg.ROI_MAX, device),
        record_all_mc=True,  # <-- enable grid recording
        record_pred_spectra=True,  # <-- include predicted spectra grid (TMM_FAST)
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
            save_as_json(cfg.PATH_SAVED, out.get("mae_grid", []).numpy().tolist(), f"results_{base}_{tag}_mae")
            save_as_json(cfg.PATH_SAVED, out.get("ids_grid", []).numpy().tolist(), f"results_{base}_{tag}_ids")
            print(f"Saved {len(out.get('results', []))} record(s) to {cfg.PATH_SAVED}")
        except Exception as e:
            print(f"Could not save results JSON: {e}")

    # Summary (accuracy/MAE may be None for single target without ground-truth)
    if out.get("mean_acc") is not None:
        print(f"mean token accuracy: {100.0 * float(out['mean_acc']):.2f}%")
    if out.get("mean_mae") is not None:
        print(f"mean spectrum MAE: {float(out['mean_mae']):.6f}")

    return out, idx_to_token, eos_idx, pad_idx, msk_idx


# ----------------------------- CLI entry -----------------------------
if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", "config_OG_LOCAL.yaml"])
        sys.argv.extend(["--config", "config_OL_LOCAL.yaml"])
        # sys.argv.extend(["--config", "./OptoLlama/scripts/config_OL_HPCZ1.yaml"])

    # Parse args and build final config (applies --ckpt/--mc-samples/--validsim and --set)
    args = cli.parse_arguments()
    cfg = cli.load_config_with_overrides(args)

    if getattr(args, "print_config", False):
        cli.print_config(cfg)
        sys.exit(0)

    out, idx_to_token, eos_idx, pad_idx, msk_idx = run_inference(
        cfg=cfg,
        ckpt=cfg.PATH_CKPT,
        mc_samples=cfg.MC_SAMPLES,
        target=getattr(cfg, "TARGET", None),
        n_targets=cfg.N_TARGETS,
    )
