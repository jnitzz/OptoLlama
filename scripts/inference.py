#!/usr/bin/env python

import os

# ruff: noqa: N806
from typing import Any

import torch

import optollama.data
import optollama.evaluation
import optollama.model
import optollama.plotting
import optollama.utils


def _target_width(cfg: dict[str, Any]) -> int:
    """Return the configured wavelength-grid width."""
    wavelengths = cfg["WAVELENGTHS"]
    return int(wavelengths.numel()) if torch.is_tensor(wavelengths) else len(wavelengths)


def _common_mae_wavelengths(cfg: dict[str, Any], device: torch.device) -> torch.Tensor | None:
    """Build the fixed comparison grid for cross-config MAE reporting."""
    if not bool(cfg.get("COMMON_MAE_ENABLED", True)):
        return None

    wl_min = int(cfg.get("COMMON_MAE_WAVELENGTH_MIN", 300))
    wl_max = int(cfg.get("COMMON_MAE_WAVELENGTH_MAX", 1700))
    wl_step = int(cfg.get("COMMON_MAE_WAVELENGTH_STEPS", 10))
    return torch.arange(wl_min, wl_max + 1, wl_step, dtype=torch.float32, device=device)


def _selection_target_mode(cfg: dict[str, Any]) -> str:
    """Return whether MC ranking uses the original or conditioned target."""
    mode = str((cfg.get("TARGET_PHYSICALIZE") or {}).get("SELECTION_TARGET", "conditioned")).lower()
    if mode not in {"conditioned", "original"}:
        raise ValueError("TARGET_PHYSICALIZE.SELECTION_TARGET must be 'conditioned' or 'original'.")
    return mode


def _load_target_spectra(target: str, cfg: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Load one configured target spectrum and optional separate score target."""
    if target == "random":
        original = torch.rand([3, _target_width(cfg)], device=device)
    else:
        original = optollama.utils.load_spectra(target, cfg).to(device)

    spectrum, info = optollama.data.physicalize_target_spectrum(original, cfg, device=device)
    if info.get("enabled"):
        selection_target = _selection_target_mode(cfg)
        nn = info.get("nn")
        if nn:
            print(
                "TARGET_PHYSICALIZE enabled: "
                f"selection_target={selection_target}, "
                f"NN id={nn['global_index']} mae={nn['mae']:.6f} "
                f"file={nn['file']}:{nn['local_index']}"
            )
        else:
            print(f"TARGET_PHYSICALIZE enabled: selection_target={selection_target}.")

        if selection_target == "original":
            return spectrum.to(device), original.to(device)

    return spectrum.to(device), None


def _make_target_loader(
    target: str,
    cfg: dict[str, Any],
    device: torch.device,
    msk_idx: int,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.DataLoader, torch.Tensor | None]:
    """Build the repeated-spectrum loader for one target spectrum."""
    spectrum, score_spectrum = _load_target_spectra(target, cfg, device)
    dataset, loader = optollama.data.RepeatedSpectrumDataset.make_loader(
        spectrum,
        cfg=cfg,
        msk_idx=msk_idx,
    )
    return dataset, loader, score_spectrum


def _example_spectrum(test_ds: torch.utils.data.Dataset) -> torch.Tensor:
    """Extract one spectrum to initialize the model input layers."""
    if isinstance(test_ds, torch.utils.data.Subset):
        return test_ds.dataset.spectra[0]
    if hasattr(test_ds, "spectra"):
        return test_ds.spectra[0]
    return test_ds.spectrum


def _apply_token_constraints(
    model: torch.nn.Module,
    cfg: dict[str, Any],
    tokens: list[str],
    token_to_idx: dict[str, int],
) -> None:
    """Apply optional material-token constraints to target-spectrum sampling."""
    if not cfg["TOKEN_FILTER_ENABLED"]:
        return

    material_groups = optollama.data.make_material_groups(tokens, token_to_idx)

    mode = cfg["TOKEN_FILTER_MODE"]
    groups = cfg["TOKEN_FILTER_GROUPS"]

    allow_group_ids, exclude_group_ids = [], []
    for group in groups:
        if group not in material_groups:
            raise ValueError(f"Unknown TOKEN_FILTER_GROUPS entry: {group!r}. Use one of {list(material_groups.keys())}.")

        # In allow mode, groups are treated as allowlists. In exclude mode, treated as blocklists.
        (allow_group_ids if mode == "allow" else exclude_group_ids).append(material_groups[group])

    allow_ids = torch.cat(allow_group_ids, dim=0) if allow_group_ids else torch.empty((0,), dtype=torch.long)
    exclude_ids = torch.cat(exclude_group_ids, dim=0) if exclude_group_ids else torch.empty((0,), dtype=torch.long)

    material_token_ids = optollama.data.make_material_token_ids(token_to_idx)

    allow_tokens = cfg["TOKEN_FILTER_ALLOW_TOKENS"]
    if allow_tokens:
        allow_ids = torch.unique(
            torch.cat([allow_ids, optollama.data.token_ids_of(allow_tokens, token_to_idx, material_token_ids)], dim=0)
        )

    exclude_tokens = cfg["TOKEN_FILTER_EXCLUDE_TOKENS"]
    if exclude_tokens:
        exclude_ids = torch.unique(
            torch.cat([exclude_ids, optollama.data.token_ids_of(exclude_tokens, token_to_idx, material_token_ids)], dim=0)
        )

    allowed_count = allow_ids.numel()
    excluded_count = exclude_ids.numel()

    if mode == "allow" and not allowed_count:
        raise ValueError(
            "TOKEN_FILTER_MODE='allow' requires TOKEN_FILTER_GROUPS and/or TOKEN_FILTER_ALLOW_TOKENS to be non-empty."
        )

    try:
        model.set_token_constraints(
            allow_ids=allow_ids if allowed_count > 0 else None,
            exclude_ids=exclude_ids if excluded_count > 0 else None,
        )
        print(f"Token constraints enabled (mode={mode}, allow={allowed_count}, exclude={excluded_count}).")
    except AttributeError:
        print("Warning: model does not support token constraints; TOKEN_FILTER_* will be ignored.")


def _save_output(cfg: dict[str, Any], test_output: dict[str, Any], rank: int) -> None:
    """Save inference output artifacts for one validation/target run."""
    if rank != 0:
        return

    os.makedirs(cfg["OUTPUT_PATH"], exist_ok=True)

    samples = len(test_output["results"])
    optollama.utils.save_as_json(cfg["SAMPLES_PATH"], test_output["results"])
    print(f"[rank 0] Saved {samples} samples -> {cfg['SAMPLES_PATH']}")

    grid = test_output.get("mae_grid", [])
    if torch.is_tensor(grid):
        optollama.utils.save_as_json(cfg["GRID_PATH"], grid.detach().cpu().numpy().tolist())

    ids = test_output.get("ids_grid", [])
    if torch.is_tensor(ids):
        optollama.utils.save_as_json(cfg["IDS_PATH"], ids.detach().cpu().numpy().tolist())

    plot_bundle_path = cfg.get("PLOT_BUNDLE_PATH")
    if plot_bundle_path:
        optollama.plotting.save_plot_bundle(
            plot_bundle_path,
            test_output,
            wavelengths=cfg["WAVELENGTHS"],
            roi_min=cfg.get("ROI_MIN"),
            roi_max=cfg.get("ROI_MAX"),
        )
        print(f"[rank 0] Saved plot bundle -> {plot_bundle_path}")

    accuracy = test_output["mean_acc"]
    mae = test_output.get("mean_mae", 0.0)

    print(f"\tmean token accuracy: {accuracy:.2f}%")
    print(f"\ttest MAE: {mae:.6f}")
    mae_common = test_output.get("mean_mae_common")
    if mae_common is not None:
        wl_min = int(cfg.get("COMMON_MAE_WAVELENGTH_MIN", 300))
        wl_max = int(cfg.get("COMMON_MAE_WAVELENGTH_MAX", 1700))
        wl_step = int(cfg.get("COMMON_MAE_WAVELENGTH_STEPS", 10))
        print(f"\tcommon-grid MAE ({wl_min}-{wl_max} nm, {wl_step} nm): {mae_common:.6f}")
    timing = test_output.get("timing")
    if timing:
        print(
            "\ttiming: "
            f"model={timing.get('model_s', 0.0):.2f}s, "
            f"tmm={timing.get('tmm_s', 0.0):.2f}s, "
            f"record={timing.get('record_s', 0.0):.2f}s, "
            f"post={timing.get('post_s', 0.0):.2f}s"
        )
        if "dedup_ratio" in timing:
            print(
                "\tdedup: "
                f"unique/input={timing['dedup_ratio']:.3f} "
                f"({int(timing.get('dedup_unique', 0))}/{int(timing.get('dedup_input', 0))})"
            )


@torch.no_grad()
def inference(cfg: dict) -> tuple[dict[str, Any], dict[int, str], int, int, int]:
    """
    Run validation or target-spectrum inference.

    If ``TARGETS`` or ``TARGET_GLOB`` is configured, the model/checkpoint are
    loaded once and each target is evaluated into ``OUTPUT_PATH/<target_name>/``.
    The legacy single ``TARGET`` and validation-dataset modes keep their
    existing output paths.
    """
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cudnn.benchmark = True

    # --- distributed computation setup ---
    device, _, rank, world_size = optollama.utils.setup_run(cfg, make_dirs=True)

    # --- tokens ---
    tokens, token_to_idx, idx_to_token, _, _, _, eos_idx, pad_idx, msk_idx = optollama.data.init_tokens(cfg["TOKENS_PATH"])

    target_entries, multi_target = optollama.utils.target_cfgs(cfg)
    first_target_loader: tuple[torch.utils.data.Dataset, torch.utils.data.DataLoader, torch.Tensor | None] | None = None

    if target_entries:
        first_spec, first_cfg = target_entries[0]
        first_target_loader = _make_target_loader(first_spec.target, first_cfg, device, msk_idx)
        test_ds = first_target_loader[0]
        if rank == 0 and multi_target:
            print(f"Multi-target inference enabled: {len(target_entries)} targets.")
    else:
        test_ds, test_loader, _ = optollama.data.SpectraDataset.make_loader(
            cfg, split="test", subset_n=cfg["NUM_SAMPLES_TEST"], ddp=False
        )

    # --- model ---
    vocab_size = len(idx_to_token)
    model = optollama.model.build_model(
        model_type=cfg["MODEL"],
        sample_spectrum=_example_spectrum(test_ds),  # [3,W] example
        vocab_size=vocab_size,
        max_stack_depth=cfg["MAX_SEQ_LEN"],
        d_model=cfg["D_MODEL"],
        n_blocks=cfg["N_BLOCKS"],
        n_heads=cfg["N_HEADS"],
        timesteps=cfg.get("DIFFUSION_STEPS", None),
        dropout=cfg["DROPOUT"],
        idx_to_token=idx_to_token,
        mask_idx=msk_idx,
        pad_idx=pad_idx,
        eos_idx=eos_idx,
        device=device,
        temperature=cfg["TEMPERATURE"],
        top_k=cfg["TOP_K"],
        top_p=cfg["TOP_P"],
    ).to(device)

    # --- checkpointing ---
    checkpoint = cfg["BEST_CHECKPOINT_PATH"]
    if checkpoint and os.path.exists(checkpoint):
        print(f"Resuming from checkpoint {checkpoint}")
        optollama.utils.load_checkpoint(checkpoint, model, map_location="cpu", strict=True)
    else:
        print(f"Checkpoint path set to {checkpoint} does not exist.")

    _apply_token_constraints(model, cfg, tokens, token_to_idx)

    # --- emittance capping ---
    max_emit_len = cfg["MAX_EMIT_LEN"]
    try:
        model.set_max_emit_len(max_emit_len)
        print(f"MAX_EMIT_LEN enabled: {max_emit_len}")
    except AttributeError:
        pass

    # --- TMM simulation ---
    tmm_ctx = (
        optollama.evaluation.simulation.TMMContext.make(cfg=cfg, idx_to_token=idx_to_token, device=device)
        if cfg["VALID_SIM"] == "TMM_FAST"
        else None
    )
    if tmm_ctx is not None and tmm_ctx.realistic_enabled:
        print(
            "REALISTIC_TMM enabled: "
            f"{len(tmm_ctx.average_thetas)} angles x {len(tmm_ctx.polarizations)} polarizations x "
            f"{tmm_ctx.jitter_realizations} jitter realizations, +/-{tmm_ctx.thickness_jitter_nm:g} nm"
        )

    # ---- validation options ----
    model.eval()
    record_all_mc = bool(cfg.get("INFERENCE_RECORD_ALL_MC", True))
    record_pred_spectra = bool(cfg.get("INFERENCE_RECORD_PRED_SPECTRA", True))
    show_progress = bool(cfg.get("INFERENCE_SHOW_PROGRESS", True))
    profile_timing = bool(cfg.get("INFERENCE_PROFILE_TIMING", False))
    deduplicate_stacks = bool(cfg.get("INFERENCE_DEDUP_STACKS", False))
    common_mae_wavelengths = _common_mae_wavelengths(cfg, device)
    if rank == 0:
        print(
            "Inference options: "
            f"record_all_mc={record_all_mc}, record_pred_spectra={record_pred_spectra}, "
            f"deduplicate_stacks={deduplicate_stacks}, profile_timing={profile_timing}"
        )
        if common_mae_wavelengths is not None:
            wl_min = int(cfg.get("COMMON_MAE_WAVELENGTH_MIN", 300))
            wl_max = int(cfg.get("COMMON_MAE_WAVELENGTH_MAX", 1700))
            wl_step = int(cfg.get("COMMON_MAE_WAVELENGTH_STEPS", 10))
            print(f"Common-grid MAE enabled: {wl_min}-{wl_max} nm, {wl_step} nm")

    def run_prediction(
        run_cfg: dict[str, Any],
        loader: torch.utils.data.DataLoader,
        score_spectrum: torch.Tensor | None,
    ) -> dict[str, Any]:
        return optollama.evaluation.model_prediction(
            model,
            loader,
            device=device,
            mode=run_cfg["VALID_SIM"],
            eos=eos_idx,
            pad=pad_idx,
            msk=msk_idx,
            idx_to_token=idx_to_token,
            tmm_ctx=tmm_ctx,
            mc_samples=run_cfg["MC_SAMPLES"],
            rank=rank,
            world_size=world_size,
            gather=True,
            track_step_mae=run_cfg["TRACK_DIFFUSION_STEPS_MAE"],
            roi_mask=optollama.data.spectra.wavelength_mask(
                run_cfg["WAVELENGTHS"],
                run_cfg["ROI_MIN"],
                run_cfg["ROI_MAX"],
                device,
            ),
            source_wavelengths=run_cfg["WAVELENGTHS"],
            common_mae_wavelengths=common_mae_wavelengths,
            score_spectrum=score_spectrum,
            record_all_mc=record_all_mc,
            record_pred_spectra=record_pred_spectra,
            show_progress=show_progress,
            profile_timing=profile_timing,
            deduplicate_stacks=deduplicate_stacks,
        )

    if target_entries:
        outputs: dict[str, Any] = {}
        for index, (spec, target_cfg) in enumerate(target_entries):
            if rank == 0 and multi_target:
                print(f"\n[{index + 1}/{len(target_entries)}] Target {spec.name}: {spec.target}")

            if index == 0 and first_target_loader is not None:
                _, loader, score_spectrum = first_target_loader
            else:
                _, loader, score_spectrum = _make_target_loader(spec.target, target_cfg, device, msk_idx)

            test_output = run_prediction(target_cfg, loader, score_spectrum)
            _save_output(target_cfg, test_output, rank)
            outputs[spec.name] = test_output

        if multi_target:
            return {"target_outputs": outputs}, idx_to_token, eos_idx, pad_idx, msk_idx
        return next(iter(outputs.values())), idx_to_token, eos_idx, pad_idx, msk_idx

    test_output = run_prediction(cfg, test_loader, None)
    _save_output(cfg, test_output, rank)
    return test_output, idx_to_token, eos_idx, pad_idx, msk_idx


if __name__ == "__main__":
    args = optollama.utils.parse_arguments()
    cfg = optollama.utils.load_config(args)

    inference(cfg)
