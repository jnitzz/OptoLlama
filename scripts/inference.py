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


@torch.no_grad()
def inference(cfg: dict) -> tuple[dict[str, Any], dict[int, str], int, int, int]:
    """
    Run model inference / validation for a given configuration.

    This function sets up the runtime, builds the model, optionally loads a
    checkpoint, constructs the appropriate data loader (validation split or a
    user-provided target spectrum), and finally calls :func:`evaluate.validate_model`.

    Args
    ----
    cfg: dict
        Configuration object

    Returns
    -------
    tuple[dict[str, Any], dict[int, str], int, int, int]
        A 5-tuple of ``(out, idx_to_token, eos_idx, pad_idx, msk_idx)``
        where ``out`` is a dictionary with at least:

        - ``"mean_acc"`` (float): mean token accuracy.
        - ``"mean_mae"`` (float or None): mean spectrum MAE (``None`` when
          not simulating).
        - ``"results"`` (list[dict]): per-example records (rank 0 only in
          DDP; always present in single-process runs).

    Notes
    -----
    Handles both single-process and DDP runs via
    :func:`runner.setup_run` and :func:`runner._is_ddp`. Per-example
    results are saved as JSON in ``cfg.PATH_SAVED`` (if set) with a
    filename of the form ``results_{cfg.RUN_NAME}_{tag}.json`` where
    ``tag`` is ``"valid"`` or ``"target"``.
    """
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cudnn.benchmark = True
    """
    Run model inference / validation for a given configuration.

    This function sets up the runtime, builds the model, optionally loads a
    checkpoint, constructs the appropriate data loader (validation split or a
    user-provided target spectrum), and finally calls :func:`evaluate.validate_model`.

    Args
    ----
    cfg: dict
        Configuration object

    Returns
    -------
    tuple[dict[str, Any], dict[int, str], int, int, int]
        A 5-tuple of ``(out, idx_to_token, eos_idx, pad_idx, msk_idx)``
        where ``out`` is a dictionary with at least:

        - ``"mean_acc"`` (float): mean token accuracy.
        - ``"mean_mae"`` (float or None): mean spectrum MAE (``None`` when
          not simulating).
        - ``"results"`` (list[dict]): per-example records (rank 0 only in
          DDP; always present in single-process runs).

    Notes
    -----
    Handles both single-process and DDP runs via
    :func:`runner.setup_run` and :func:`runner._is_ddp`. Per-example
    results are saved as JSON in ``cfg.PATH_SAVED`` (if set) with a
    filename of the form ``results_{cfg.RUN_NAME}_{tag}.json`` where
    ``tag`` is ``"valid"`` or ``"target"``.
    """

    # --- distributed computation setup ---
    device, _, rank, world_size = optollama.utils.setup_run(cfg, make_dirs=True)

    # --- tokens ---
    tokens, token_to_idx, idx_to_token, _, _, _, eos_idx, pad_idx, msk_idx = optollama.data.init_tokens(cfg["TOKENS_PATH"])

    # --- data source ---
    target = cfg.get("TARGET")

    if target is None:  # no target defined, load a validation dataset
        test_ds, test_loader, _ = optollama.data.SpectraDataset.make_loader(
            cfg, split="test", subset_n=cfg["NUM_SAMPLES_TEST"], ddp=False
        )
    else:
        if target == "random":
            spectrum = torch.rand([3, 171], device=device)
        else:
            spectrum = optollama.utils.load_spectra(target, cfg).to(device)

        test_ds, test_loader = optollama.data.RepeatedSpectrumDataset.make_loader(
            spectrum,
            cfg=cfg,
            msk_idx=msk_idx,
        )

    # --- model ---
    vocab_size = len(idx_to_token)
    if isinstance(test_ds, torch.utils.data.Subset):
        example_spectrum = test_ds.dataset.spectra[0]
    elif hasattr(test_ds, "spectra"):
        example_spectrum = test_ds.spectra[0]
    else:
        example_spectrum = test_ds.spectrum

    model = optollama.model.build_model(
        model_type=cfg["MODEL"],
        sample_spectrum=example_spectrum,  # [W,3] example
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
        start_epoch, blob = optollama.utils.load_checkpoint(checkpoint, model, map_location="cpu", strict=True)
    else:
        print(f"Checkpoint path set to {checkpoint} does not exist.")

    # ---------------- token/material constraints (target-spectrum relevant) ----------------
    # In the "target spectrum" branch, stacks in the dataset are dummy PADs, so dataset filtering
    # does not apply. Instead, we constrain what the model can *emit* during sampling by masking
    # logits during sampling.
    if cfg["TOKEN_FILTER_ENABLED"]:
        material_groups = optollama.data.make_material_groups(tokens, token_to_idx)

        # --- allow or disallow certain tokens for inference ---
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

        # Merge explicit token lists
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

        # Apply to model if supported (OptoLlama)
        try:
            model.set_token_constraints(
                allow_ids=allow_ids if allowed_count > 0 else None, exclude_ids=exclude_ids if excluded_count > 0 else None
            )
            print(f"Token constraints enabled (mode={mode}, allow={allowed_count}, exclude={excluded_count}).")
        except AttributeError:
            print("Warning: model does not support token constraints; TOKEN_FILTER_* will be ignored.")

    # --- emittance capping ---
    max_emit_len = cfg["MAX_EMIT_LEN"]
    try:
        model.set_max_emit_len(max_emit_len)
        print(f"MAX_EMIT_LEN enabled: {max_emit_len}")
    except:
        pass

    # --- TMM simulation ---
    tmm_ctx = (
        optollama.evaluation.simulation.TMMContext.make(cfg=cfg, idx_to_token=idx_to_token, device=device)
        if cfg["VALID_SIM"] == "TMM_FAST"
        else None
    )

    # ---- validation ----
    model.eval()
    test_output = optollama.evaluation.model_prediction(
        model,
        test_loader,
        device=device,
        mode=cfg["VALID_SIM"],
        eos=eos_idx,
        pad=pad_idx,
        msk=msk_idx,
        idx_to_token=idx_to_token,
        tmm_ctx=tmm_ctx,
        mc_samples=cfg["MC_SAMPLES"],
        rank=rank,
        world_size=world_size,
        gather=True,
        track_step_mae=cfg["TRACK_DIFFUSION_STEPS_MAE"],
        roi_mask=optollama.data.spectra.wavelength_mask(cfg["WAVELENGTHS"], cfg["ROI_MIN"], cfg["ROI_MAX"], device),
        record_all_mc=True,
        record_pred_spectra=True,
    )

    # --- save outputs to disk ---
    if rank == 0:
        os.makedirs(cfg["OUTPUT_PATH"], exist_ok=True)

        samples = len(test_output["results"])
        optollama.utils.save_as_json(cfg["SAMPLES_PATH"], test_output["results"])
        print(f"[rank 0] Saved {samples} samples -> {cfg['SAMPLES_PATH']}")

        grid = test_output.get("mae_grid", [])
        if torch.is_tensor(grid):
            optollama.utils.save_as_json(cfg["GRID_PATH"], grid.numpy().tolist())

        ids = test_output.get("ids_grid", [])
        if torch.is_tensor(ids):
            optollama.utils.save_as_json(cfg["IDS_PATH"], ids.numpy().tolist())

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

    return test_output, idx_to_token, eos_idx, pad_idx, msk_idx


if __name__ == "__main__":
    # parse args and build final config
    args = optollama.utils.parse_arguments()
    cfg = optollama.utils.load_config(args)

    inference(cfg)
