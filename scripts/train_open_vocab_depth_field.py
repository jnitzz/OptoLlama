from __future__ import annotations

import argparse
import math
import os
import random
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as functional
import tqdm  # type: ignore[import-untyped]

import optollama.data
import optollama.model
import optollama.utils
from scripts.train_open_layer_flow import (
    autocast_context,
    make_loader,
    nested,
    resolve_amp_dtype,
    resume_global_samples,
    scheduled_learning_rate,
    set_optimizer_lr,
    synchronized_finite,
)


def parse_args() -> argparse.Namespace:
    """Parse training and smoke-test overrides."""
    parser = argparse.ArgumentParser(description="Train continuous-time open-vocabulary depth-field diffusion.")
    parser.add_argument("--config", default="configs/depth_field_open_vocab_01.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true", help="Run one tiny real-data optimizer step.")
    return parser.parse_args()


def unwrap_model(model: torch.nn.Module) -> optollama.model.OpenVocabularyDepthFieldDiffusion | optollama.model.OpenLayerJointDFM:
    """Return the underlying model when DDP and/or torch.compile are active."""
    core = model
    while True:
        if isinstance(core, torch.nn.parallel.DistributedDataParallel):
            core = core.module
            continue
        original = getattr(core, "_orig_mod", None)
        if isinstance(original, torch.nn.Module):
            core = original
            continue
        return core  # type: ignore[return-value]


class LocalGradientDiagnostic:
    """Inspect DDP buckets before their gradients are averaged across ranks."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.names = {id(parameter): name for name, parameter in model.named_parameters()}
        self.bad_parameters: list[str] = []

    def reset(self) -> None:
        """Discard observations from the preceding backward pass."""
        self.bad_parameters.clear()


def diagnostic_allreduce(state: LocalGradientDiagnostic, bucket):
    """Preserve DDP's mean reduction while recording rank-local bad gradients."""
    if not bool(torch.isfinite(bucket.buffer()).all().item()):
        for parameter, gradient in zip(bucket.parameters(), bucket.gradients(), strict=True):
            if not bool(torch.isfinite(gradient).all().item()):
                state.bad_parameters.append(state.names.get(id(parameter), "<unnamed>"))
    world = torch.distributed.get_world_size()
    future = torch.distributed.all_reduce(bucket.buffer(), async_op=True).get_future()
    return future.then(lambda result: result.value()[0].div_(world))


def capture_rank_state(train_collator: Any, val_collator: Any) -> dict[str, Any]:
    """Capture rank-local randomness and collator positions at a batch boundary."""
    return {
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "train_collator_calls": train_collator.calls,
        "val_collator_calls": val_collator.calls,
    }


def restore_rank_state(state: dict[str, Any]) -> None:
    """Restore randomness after creating the resumed data iterator."""
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None:
        torch.cuda.set_rng_state(state["cuda_rng"])


class ModelEma:
    """Maintain a floating-point exponential moving average of model state."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        if not 0.0 < float(decay) < 1.0:
            raise ValueError("EMA decay must be in (0,1).")
        self.decay = float(decay)
        self.updates = 0
        self.shadow = {
            name: value.detach().clone()
            for name, value in unwrap_model(model).state_dict().items()
            if torch.is_floating_point(value)
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module, *, decay: float | None = None) -> None:
        """Update shadow tensors after one successful optimizer step."""
        effective_decay = self.decay if decay is None else float(decay)
        for name, value in unwrap_model(model).state_dict().items():
            if name in self.shadow:
                self.shadow[name].mul_(effective_decay).add_(value.detach(), alpha=1.0 - effective_decay)
        self.updates += 1

    def state_dict(self) -> dict[str, Any]:
        """Return checkpoint state on CPU."""
        return {
            "decay": self.decay,
            "updates": self.updates,
            "shadow": {name: value.detach().cpu() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, state: dict[str, Any], model: torch.nn.Module) -> None:
        """Restore matching shadow tensors and retain current values for new keys."""
        self.decay = float(state.get("decay", self.decay))
        self.updates = int(state.get("updates", 0))
        current = unwrap_model(model).state_dict()
        saved = state.get("shadow") or {}
        for name in self.shadow:
            if name in saved and saved[name].shape == current[name].shape:
                self.shadow[name] = saved[name].to(device=current[name].device, dtype=current[name].dtype)

    @contextmanager
    def apply(self, model: torch.nn.Module):
        """Temporarily evaluate the model with EMA parameters."""
        core = unwrap_model(model)
        backup = {name: value.detach().clone() for name, value in core.state_dict().items() if name in self.shadow}
        current = core.state_dict()
        current.update(
            {name: value.to(device=current[name].device, dtype=current[name].dtype) for name, value in self.shadow.items()}
        )
        core.load_state_dict(current, strict=True)
        try:
            yield
        finally:
            restored = core.state_dict()
            restored.update(backup)
            core.load_state_dict(restored, strict=True)


def model_config_from_mapping(
    block: dict[str, Any], spectrum_shape: tuple[int, ...]
) -> optollama.model.OpenVocabularyDepthFieldConfig:
    """Construct model metadata from the open-vocabulary config section."""
    model = nested(block, "MODEL", default={}) or {}
    bank = nested(block, "MATERIAL_BANK", default={}) or {}
    grid = nested(block, "GRID", default={}) or {}
    d_model = int(model.get("D_MODEL", 896))
    n_blocks = int(model.get("N_BLOCKS", 8))
    return optollama.model.OpenVocabularyDepthFieldConfig(
        spectrum_shape=spectrum_shape,
        depth_bins=int(round(float(grid.get("MAX_THICKNESS_NM", 10_000.0)) / float(grid.get("DZ_NM", 5.0)))),
        depth_patch_size=int(model.get("DEPTH_PATCH_SIZE", 1)),
        max_candidates=int(bank.get("MAX_CANDIDATES", 24)),
        d_model=d_model,
        n_blocks=n_blocks,
        n_heads=int(model.get("N_HEADS", 8)),
        ffn_multiplier=float(model.get("FFN_MULTIPLIER", 4.0)),
        kernel_size=int(model.get("KERNEL_SIZE", 7)),
        dropout=float(model.get("DROPOUT", 0.0)),
        conv_type=str(model.get("CONV_TYPE", "separable")),
        hybrid_dilations=tuple(int(value) for value in model.get("HYBRID_DILATIONS", ())),
        hybrid_residual_init=float(model.get("HYBRID_RESIDUAL_INIT", 1.0e-3)),
        spectrum_patch_size=int(model.get("SPECTRUM_PATCH_SIZE", 4)),
        spectrum_patch_stride=int(model.get("SPECTRUM_PATCH_STRIDE", 2)),
        spectrum_encoder_blocks=int(model.get("SPECTRUM_ENCODER_BLOCKS", 4)),
        spectrum_encoder_heads=int(model.get("SPECTRUM_ENCODER_HEADS", 8)),
        spectrum_ffn_multiplier=float(model.get("SPECTRUM_FFN_MULTIPLIER", 2.0)),
        wavelength_scale_nm=float(model.get("WAVELENGTH_SCALE_NM", 1_000.0)),
        wavelength_fourier_bands=int(model.get("WAVELENGTH_FOURIER_BANDS", 4)),
        query_spectrum=bool(model.get("QUERY_SPECTRUM", False)),
        candidate_cross_attention=bool(model.get("CANDIDATE_CROSS_ATTENTION", False)),
        fp32_attention=bool(model.get("FP32_ATTENTION", False)),
    )


def make_collator(
    cfg: dict[str, Any],
    block: dict[str, Any],
    catalog: optollama.data.MaterialCatalog,
    idx_to_token: dict[int, str],
    eos_idx: int,
    pad_idx: int,
    msk_idx: int,
    *,
    train: bool,
    seed: int,
    joint: bool = False,
) -> optollama.data.OpenLayerBatchCollator:
    """Build a full-grid or coordinate-query training/validation collator."""
    bank = nested(block, "MATERIAL_BANK", default={}) or {}
    grid = nested(block, "GRID", default={}) or {}
    optical = nested(block, "OPTICAL_CONDITION", default={}) or {}
    query = nested(block, "QUERY", default={}) or {}
    wavelengths = cfg["WAVELENGTHS"]
    coordinate_query = bool(query.get("ENABLED", False))
    query_points = int(query.get("POINTS", len(wavelengths))) if coordinate_query else len(wavelengths)
    query_mode = str(query.get("TRAIN_MODE" if train else "VAL_MODE", "mixed" if train else "full"))
    collator_type = optollama.data.OpenLayerJointDFMCollator if joint else optollama.data.OpenVocabularyDepthFieldCollator
    geometry = (
        {"min_layer_nm": float(nested(block, "THICKNESS", "MIN_NM", default=1.0)),
         "max_total_nm": float(nested(block, "THICKNESS", "MAX_TOTAL_NM", default=10_000.0))}
        if joint else {"dz_nm": float(grid.get("DZ_NM", 5.0)), "max_total_nm": float(grid.get("MAX_THICKNESS_NM", 10_000.0))}
    )
    return collator_type(
        wavelengths_nm=wavelengths,
        catalog=catalog,
        idx_to_token=idx_to_token,
        eos_idx=eos_idx,
        pad_idx=pad_idx,
        msk_idx=msk_idx,
        channels=("R", "A", "T"),
        max_layers=int(block.get("MAX_LAYERS", 100)),
        max_candidates=int(bank.get("MAX_CANDIDATES", 24)),
        min_query_points=query_points,
        max_query_points=query_points,
        query_sampling=f"coordinate_{query_mode}" if coordinate_query else "full",
        query_band_min_nm=float(query.get("MIN_BAND_NM", 100.0)),
        query_band_max_nm=float(query.get("MAX_BAND_NM", 900.0)),
        randomize_candidates=bool(bank.get("RANDOMIZE_ORDER", True) if train else False),
        random_distractors=bool(bank.get("RANDOM_DISTRACTORS", True)),
        max_random_distractors=(
            int(bank["MAX_RANDOM_DISTRACTORS"])
            if train and bank.get("MAX_RANDOM_DISTRACTORS") is not None
            else None
        ),
        full_bank_probability=float(bank.get("FULL_BANK_PROBABILITY", 0.0)) if train else 0.0,
        holdout_materials=bank.get("HOLDOUT_MATERIALS", []) if train else (),
        coverage_tolerance_nm=float(bank.get("MATERIAL_COVERAGE_TOLERANCE_NM", 100.0)),
        seed=seed,
        incidence_angle_deg=float(optical.get("ANGLE_DEG", 0.0)),
        polarization=str(optical.get("POLARIZATION", "s")),
        **geometry,
    )


def select_valid_rows(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor] | None:
    """Drop empty or held-out stacks before computing a supervised loss."""
    valid = batch["sample_mask"].to(dtype=torch.bool)
    if not bool(valid.any()):
        return None
    return {key: value[valid] for key, value in batch.items()}


def compute_joint_loss(
    model: torch.nn.Module, core: optollama.model.OpenLayerJointDFM, batch: dict[str, torch.Tensor], train_cfg: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Train count, unweighted categorical endpoints, and velocity on one noised state."""
    clean = batch["material_targets"]
    mask = batch["layer_mask"]
    s = torch.rand(clean.shape[0], device=clean.device)
    material, thickness, target_velocity, corrupted = core.training_state(batch, s)
    output = model(
        material_ids=material, thickness_state=thickness, layer_mask=mask, flow_time=s,
        spectra=batch["target_spectrum"].transpose(1, 2), wavelengths_nm=batch["wavelengths_nm"],
        candidate_nk=batch["candidate_nk"], candidate_mask=batch["candidate_mask"],
        incidence_angle_deg=batch["incidence_angle_deg"], polarization_id=batch["polarization_id"],
    )
    logits = output["material_logits"].float()
    material_loss = functional.cross_entropy(logits.flatten(0, 1), clean.flatten(), reduction="sum") / mask.sum().clamp_min(1)
    active_state = core.state_mask(mask)
    squared = (output["thickness_velocity"].float() - target_velocity).square()
    thickness_loss = squared.masked_fill(~active_state, 0).sum() / active_state.sum().clamp_min(1)
    target_counts = mask.sum(dim=1)
    count_loss = functional.cross_entropy(output["count_logits"].float(), target_counts)
    correct = logits.argmax(dim=-1) == clean
    return {
        "loss": material_loss + float(train_cfg.get("THICKNESS_LOSS_WEIGHT", 1.0)) * thickness_loss +
        float(train_cfg.get("COUNT_LOSS_WEIGHT", 1.0)) * count_loss,
        "material_loss": material_loss, "thickness_loss": thickness_loss, "count_loss": count_loss,
        "count_accuracy": (output["count_logits"].argmax(dim=-1) == target_counts).float().mean(),
        "accuracy": (correct & mask).sum() / mask.sum().clamp_min(1),
        "corrupted_accuracy": (correct & corrupted).sum() / corrupted.sum().clamp_min(1),
        "noise_probability": (1 - s).square().mean(),
        "corrupted_fraction": corrupted.sum() / mask.sum().clamp_min(1),
        "condition_dropped": logits.new_zeros(()),
        "diagnostic_timesteps": s.detach(),
        "diagnostic_noised": material.detach(),
        "diagnostic_logits": logits.detach(),
    }


def compute_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    train_cfg: dict[str, Any],
    corruption: optollama.model.DepthFieldCorruptionConfig,
) -> dict[str, torch.Tensor]:
    """Compute continuous-time denoising loss through the DDP wrapper."""
    core = unwrap_model(model)
    if isinstance(core, optollama.model.OpenLayerJointDFM):
        return compute_joint_loss(model, core, batch, train_cfg)
    clean = batch["clean_fields"].long()
    timesteps = torch.rand(clean.shape[0], device=clean.device)
    noised, corrupted = core.corrupt(
        clean,
        timesteps,
        batch["candidate_mask"],
        random_replace_prob=float(train_cfg.get("RANDOM_REPLACE_PROB", 0.10)),
        corruption_config=corruption,
    )
    spectra, dropped = optollama.model.drop_spectrum_condition(
        batch["target_spectrum"].transpose(1, 2),
        float(train_cfg.get("CONDITION_DROPOUT_PROB", 0.0)),
    )
    logits = model(
        spectra,
        noised,
        timesteps,
        wavelengths_nm=batch["wavelengths_nm"],
        candidate_nk=batch["candidate_nk"],
        candidate_mask=batch["candidate_mask"],
        incidence_angle_deg=batch["incidence_angle_deg"],
        polarization_id=batch["polarization_id"],
    )
    weights = torch.ones(core.num_materials, device=logits.device, dtype=logits.dtype)
    weights[core.void_id] = float(train_cfg.get("VOID_LOSS_WEIGHT", 0.10))
    per_bin = functional.cross_entropy(
        logits.reshape(-1, core.num_materials), clean.reshape(-1), weight=weights, reduction="none"
    ).view_as(clean)
    boundary_cfg = train_cfg.get("BOUNDARY_LOSS") or {}
    boundary = optollama.model.depth_field_boundary_mask(clean, int(boundary_cfg.get("RADIUS_BINS", 2)))
    if bool(boundary_cfg.get("ENABLED", True)):
        per_bin = per_bin * torch.where(
            boundary,
            per_bin.new_full((), float(boundary_cfg.get("WEIGHT", 2.0))),
            per_bin.new_ones(()),
        )
    loss = optollama.model.weighted_depth_field_loss(
        per_bin,
        corrupted,
        corrupted_loss_weight=float(train_cfg.get("CORRUPTED_LOSS_WEIGHT", 1.0)),
        uncorrupted_loss_weight=float(train_cfg.get("UNCORRUPTED_LOSS_WEIGHT", 0.1)),
        loss_on_corrupted_only=bool(train_cfg.get("LOSS_ON_CORRUPTED_ONLY", False)),
    )
    prediction = logits.argmax(dim=-1)
    result = {
        "loss": loss,
        "accuracy": (prediction == clean).float().mean(),
        "corrupted_accuracy": (prediction[corrupted] == clean[corrupted]).float().mean()
        if bool(corrupted.any())
        else loss.new_zeros(()),
        "noise_probability": core.noise_probability(timesteps).mean(),
        "corrupted_fraction": corrupted.float().mean(),
        "condition_dropped": dropped.float().mean(),
    }
    if isinstance(core, optollama.model.OpenVocabularyDepthFieldDFM):
        result["diagnostic_timesteps"] = timesteps.detach()
        result["diagnostic_noised"] = noised.detach()
        result["diagnostic_logits"] = logits.detach()
        with torch.no_grad():
            correct = prediction == clean
            source_groups = (
                noised == core.mask_id,
                (noised < core.void_id) & (noised != clean),
                (noised == core.void_id) & (noised != clean),
                noised == clean,
            )
            time_groups = tuple(
                ((timesteps >= index / 4) & (timesteps < (index + 1) / 4 if index < 3 else timesteps <= 1))[:, None]
                .expand_as(clean)
                for index in range(4)
            )
            result["flow_diagnostics"] = torch.stack(
                [
                    torch.stack((per_bin[mask].sum(), correct[mask].sum(), mask.sum()))
                    for mask in (*source_groups, *time_groups)
                ]
            )
    return result


def flow_diagnostic_metrics(totals: torch.Tensor) -> dict[str, float]:
    """Convert summed source-state and time-band counts to mean metrics."""
    labels = ("mask", "wrong_material", "wrong_void", "visible", "t_0_25", "t_25_50", "t_50_75", "t_75_100")
    result: dict[str, float] = {}
    for index, label in enumerate(labels):
        count = max(1.0, float(totals[index, 2].item()))
        result[f"{label}_loss"] = float(totals[index, 0].item()) / count
        result[f"{label}_accuracy"] = float(totals[index, 1].item()) / count
        result[f"{label}_bins"] = int(totals[index, 2].item())
    return result


def run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    train_cfg: dict[str, Any],
    corruption: optollama.model.DepthFieldCorruptionConfig,
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    amp_dtype: torch.dtype | None,
    epoch: int,
    epochs: int,
    max_steps: int | None,
    base_lr: float,
    lr_schedule: dict[str, Any] | None,
    global_samples_seen: int,
    ema: ModelEma | None = None,
    global_optimizer_steps: int = 0,
    log_every_steps: int = 50,
    ema_update_every_steps: int = 1,
    eval_every_steps: int | None = None,
    on_evaluation_step: Callable[[int, int, dict[str, float]], None] | None = None,
    checkpoint_every_steps: int | None = None,
    on_checkpoint_step: Callable[[int, int, int, dict[str, Any]], None] | None = None,
    resume_state: dict[str, Any] | None = None,
    gradient_diagnostic: LocalGradientDiagnostic | None = None,
    diagnostic_dir: Path | None = None,
) -> dict[str, float]:
    """Train or validate one epoch and aggregate metrics over all ranks."""
    training = optimizer is not None
    model.train(training)
    joint = isinstance(unwrap_model(model), optollama.model.OpenLayerJointDFM)
    keys = ("loss", "accuracy", "corrupted_accuracy", "noise_probability", "corrupted_fraction", "condition_dropped")
    if joint:
        keys += ("material_loss", "thickness_loss", "count_loss", "count_accuracy")
    totals = torch.zeros(len(keys) + 1, dtype=torch.float64, device=device)
    flow_totals = (
        torch.zeros((8, 3), dtype=torch.float64, device=device)
        if isinstance(unwrap_model(model), optollama.model.OpenVocabularyDepthFieldDFM)
        else None
    )
    local_rows = 0
    optimizer_steps = 0
    nonfinite_skips = 0
    nonfinite_loss_skips = 0
    nonfinite_grad_skips = 0
    consecutive_nonfinite_skips = 0
    max_nonfinite_steps = max(0, int(train_cfg.get("MAX_NONFINITE_STEPS_PER_EPOCH", 0)))
    max_consecutive_nonfinite_steps = max(0, int(train_cfg.get("MAX_CONSECUTIVE_NONFINITE_STEPS", 0)))
    start_batch = 0
    if training and resume_state:
        start_batch = int(resume_state["next_batch"])
        totals.copy_(resume_state["totals"].to(device))
        if flow_totals is not None and resume_state.get("flow_totals") is not None:
            flow_totals.copy_(resume_state["flow_totals"].to(device))
        local_rows = int(resume_state["local_rows"])
        optimizer_steps = int(resume_state["optimizer_steps"])
        nonfinite_skips = int(resume_state["nonfinite_skips"])
        nonfinite_loss_skips = int(resume_state["nonfinite_loss_skips"])
        nonfinite_grad_skips = int(resume_state["nonfinite_grad_skips"])
        consecutive_nonfinite_skips = int(resume_state["consecutive_nonfinite_skips"])
    log_every_steps = max(1, int(log_every_steps))
    ema_update_every_steps = max(1, int(ema_update_every_steps))
    eval_every_steps = None if not eval_every_steps else max(1, int(eval_every_steps))
    checkpoint_every_steps = None if not checkpoint_every_steps else max(1, int(checkpoint_every_steps))
    world = torch.distributed.get_world_size() if optollama.utils.is_ddp() else 1
    show = not optollama.utils.is_ddp() or torch.distributed.get_rank() == 0
    progress = tqdm.tqdm(
        loader,
        desc=f"Epoch {epoch + 1}/{epochs} {'joint layer DFM' if joint else 'open-vocab depth'} {'train' if training else 'val'}",
        disable=not show,
        miniters=log_every_steps,
        mininterval=5.0,
        initial=start_batch if training else 0,
        total=start_batch + len(loader) if training else len(loader),
    )
    for step, raw in enumerate(progress, start=start_batch):
        if max_steps is not None and step >= max_steps:
            break
        if training and resume_state is not None and step == start_batch:
            restore_rank_state(resume_state["rng"])
        selected = raw if joint else select_valid_rows(raw)
        if selected is None:
            continue
        batch = {key: value.to(device, non_blocking=True) for key, value in selected.items()}
        rows = int(batch["material_targets" if joint else "clean_fields"].shape[0])
        if training:
            assert optimizer is not None
            lr = scheduled_learning_rate(base_lr, lr_schedule, global_samples_seen + local_rows * world)
            set_optimizer_lr(optimizer, lr)
            optimizer.zero_grad(set_to_none=True)
        context = torch.enable_grad() if training else torch.no_grad()
        with context, autocast_context(device, amp_dtype):
            output = compute_loss(model, batch, train_cfg, corruption)
        if not training and not synchronized_finite(output["loss"]):
            raise FloatingPointError(f"Non-finite open-vocabulary depth loss at epoch={epoch + 1}, step={step}.")
        if training:
            assert optimizer is not None
            # Complete DDP backward even for a bad loss so the next forward is safe.
            if gradient_diagnostic is not None:
                gradient_diagnostic.reset()
            scaler.scale(output["loss"]).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(train_cfg.get("GRAD_CLIP", 1.0)), error_if_nonfinite=False
            )
            finite = torch.stack(
                (torch.isfinite(output["loss"].detach()).all(), torch.isfinite(norm.detach()).all())
            ).to(dtype=torch.int32)
            if optollama.utils.is_ddp():
                torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
            loss_finite, gradient_finite = (bool(value) for value in finite.tolist())
            failure = "loss" if not loss_finite else ("gradient" if not gradient_finite else None)
            if failure is not None:
                local_bad = (
                    not bool(torch.isfinite(output["loss"]).all().item())
                    if failure == "loss"
                    else not bool(torch.isfinite(norm).all().item())
                )
                if local_bad:
                    sample_ids = batch.get("sample_indices")
                    indices = sample_ids.tolist() if sample_ids is not None else []
                    rank = torch.distributed.get_rank() if optollama.utils.is_ddp() else 0
                    bad_names = gradient_diagnostic.bad_parameters if gradient_diagnostic is not None else []
                    print(
                        f"[rank{rank}] Non-finite {failure} at epoch={epoch + 1}, step={step}, "
                        f"loss={float(output['loss'].detach()):.6g}, "
                        f"grad_norm={float(norm):.6g}, local_bad_parameters={bad_names[:12]}, "
                        f"sample_indices={indices}",
                        flush=True,
                    )
                    if diagnostic_dir is not None:
                        t = output.get("diagnostic_timesteps")
                        noised = output.get("diagnostic_noised")
                        logits = output.get("diagnostic_logits")
                        core = unwrap_model(model)
                        valid_logits = None
                        if logits is not None:
                            if joint:
                                valid_logits = logits[batch["layer_mask"][:, :, None] & batch["candidate_mask"][:, None, :]]
                            else:
                                valid_materials = batch["candidate_mask"][:, None, :].expand_as(logits[..., :-1])
                                valid_logits = torch.cat((logits[..., :-1][valid_materials], logits[..., -1].reshape(-1)))
                        record = {
                            "epoch": epoch + 1,
                            "batch_index": step,
                            "rank": rank,
                            "failure": failure,
                            "loss": float(output["loss"].detach()) if loss_finite else None,
                            "grad_norm": float(norm) if gradient_finite else None,
                            "local_bad_parameters": bad_names,
                            "sample_indices": indices,
                            "candidate_counts": batch["candidate_mask"].sum(dim=1).tolist(),
                            "timestep_min_mean_max": [float(value) for value in (t.min(), t.mean(), t.max())] if t is not None else None,
                            "input_mask_fraction": float((noised == core.mask_id).float().mean()) if noised is not None else None,
                            "input_void_fraction": float((noised == core.void_id).float().mean()) if noised is not None and not joint else None,
                            "valid_logits_finite": bool(torch.isfinite(valid_logits).all()) if valid_logits is not None else None,
                            "valid_logit_abs_max": float(valid_logits.abs().max()) if valid_logits is not None and valid_logits.numel() and bool(torch.isfinite(valid_logits).all()) else None,
                        }
                        optollama.utils.save_as_json(str(diagnostic_dir / f"epoch_{epoch + 1:04d}_batch_{step:06d}_rank_{rank}.json"), record)
                optimizer.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.update()
                nonfinite_skips += 1
                consecutive_nonfinite_skips += 1
                if failure == "loss":
                    nonfinite_loss_skips += 1
                else:
                    nonfinite_grad_skips += 1
                if nonfinite_skips > max_nonfinite_steps or (
                    max_consecutive_nonfinite_steps > 0 and consecutive_nonfinite_skips > max_consecutive_nonfinite_steps
                ):
                    raise FloatingPointError(
                        f"Non-finite {failure} at epoch={epoch + 1}, step={step}; "
                        f"skipped {nonfinite_skips} steps this epoch "
                        f"({consecutive_nonfinite_skips} consecutive)."
                    )
                if show:
                    print(
                        f"Skipped non-finite {failure} step {step} in epoch {epoch + 1} "
                        f"({nonfinite_skips}/{max_nonfinite_steps} this epoch).",
                        flush=True,
                    )
                continue
            consecutive_nonfinite_skips = 0
            scaler.step(optimizer)
            scaler.update()
            local_rows += rows
            optimizer_steps += 1
            current_global_step = global_optimizer_steps + optimizer_steps
            if ema is not None and current_global_step % ema_update_every_steps == 0:
                ema.update(model, decay=ema.decay**ema_update_every_steps)
        metric_values = torch.stack([output[key].detach().to(dtype=torch.float64) for key in keys])
        totals[:-1].add_(metric_values * rows)
        totals[-1].add_(rows)
        if flow_totals is not None:
            flow_totals.add_(output["flow_diagnostics"].to(dtype=torch.float64))
        display_step = optimizer_steps if training else step + 1
        if display_step == 1 or display_step % log_every_steps == 0:
            running = {key: float((totals[index] / totals[-1]).item()) for index, key in enumerate(keys)}
            progress.set_postfix(
                loss=f"{running['loss']:.4f}",
                acc=f"{100 * running['accuracy']:.1f}%",
                corr=f"{100 * running['corrupted_accuracy']:.1f}%",
                refresh=False,
                **({key: f"{running[key]:.4f}" for key in ("material_loss", "thickness_loss", "count_loss")} if joint else {}),
            )
        if (
            training
            and eval_every_steps is not None
            and on_evaluation_step is not None
            and current_global_step % eval_every_steps == 0
        ):
            snapshot_totals = totals.clone()
            if optollama.utils.is_ddp():
                torch.distributed.all_reduce(snapshot_totals)
            snapshot = {
                key: float((snapshot_totals[index] / snapshot_totals[-1]).item()) for index, key in enumerate(keys)
            }
            snapshot["samples"] = int(snapshot_totals[-1].item())
            if flow_totals is not None:
                flow_snapshot = flow_totals.clone()
                if optollama.utils.is_ddp():
                    torch.distributed.all_reduce(flow_snapshot)
                snapshot.update(flow_diagnostic_metrics(flow_snapshot))
            on_evaluation_step(
                current_global_step,
                global_samples_seen + local_rows * world,
                snapshot,
            )
            model.train(True)
        if (
            training
            and checkpoint_every_steps is not None
            and on_checkpoint_step is not None
            and current_global_step % checkpoint_every_steps == 0
        ):
            state = {
                "next_batch": step + 1,
                "totals": totals.detach().cpu(),
                "flow_totals": flow_totals.detach().cpu() if flow_totals is not None else None,
                "local_rows": local_rows,
                "optimizer_steps": optimizer_steps,
                "nonfinite_skips": nonfinite_skips,
                "nonfinite_loss_skips": nonfinite_loss_skips,
                "nonfinite_grad_skips": nonfinite_grad_skips,
                "consecutive_nonfinite_skips": consecutive_nonfinite_skips,
            }
            on_checkpoint_step(current_global_step, global_samples_seen + local_rows * world, step + 1, state)
    if optollama.utils.is_ddp():
        torch.distributed.all_reduce(totals)
        if flow_totals is not None:
            torch.distributed.all_reduce(flow_totals)
    if totals[-1] <= 0:
        raise RuntimeError("No valid samples were processed.")
    metrics = {key: float(totals[index] / totals[-1]) for index, key in enumerate(keys)}
    if flow_totals is not None:
        metrics.update(flow_diagnostic_metrics(flow_totals))
    metrics["samples"] = int(totals[-1].item())
    metrics["samples_seen"] = local_rows * world if training else int(totals[-1].item())
    metrics["global_samples_seen"] = global_samples_seen + int(metrics["samples_seen"]) if training else global_samples_seen
    metrics["optimizer_steps"] = optimizer_steps
    metrics["global_optimizer_steps"] = global_optimizer_steps + optimizer_steps
    metrics["nonfinite_skips"] = nonfinite_skips
    metrics["nonfinite_loss_skips"] = nonfinite_loss_skips
    metrics["nonfinite_grad_skips"] = nonfinite_grad_skips
    metrics["learning_rate"] = float(optimizer.param_groups[0]["lr"]) if optimizer is not None else 0.0
    return metrics


def joint_metric_suffix(metrics: dict[str, float], enabled: bool) -> str:
    """Report the three joint losses without changing legacy log formats."""
    return "" if not enabled else ", " + ", ".join(
        f"{key}={metrics[key]:.5f}" for key in ("material_loss", "thickness_loss", "count_loss", "count_accuracy")
    )


def main() -> None:
    """Train and checkpoint an open-vocabulary depth-field model."""
    args = parse_args()
    cfg = optollama.utils.load_config_file(args.config)
    cfg["WAVELENGTHS"] = torch.arange(
        int(cfg["WAVELENGTH_MIN"]), int(cfg["WAVELENGTH_MAX"]) + 1, int(cfg["WAVELENGTH_STEPS"]), dtype=torch.float32
    )
    joint = "OPEN_LAYER_DFM_JOINT" in cfg
    block = cfg.get("OPEN_LAYER_DFM_JOINT" if joint else "OPEN_VOCAB_DEPTH_FIELD") or {}
    checkpoint_prefix = "open-layer-dfm-joint" if joint else "open-vocab-depth"
    model_metadata_key = "open_layer_dfm_joint_config" if joint else "open_vocab_depth_field_config"
    label = "Joint layer DFM" if joint else "Open-vocab depth"
    train_cfg = nested(block, "TRAIN", default={}) or {}
    if args.batch_size is not None:
        cfg["TRAIN_BATCH_SIZE"] = args.batch_size
    if args.smoke_test:
        cfg["TRAIN_BATCH_SIZE"] = 1
        cfg["TEST_BATCH_SIZE"] = 1
    if args.device is not None and int(os.getenv("SLURM_NTASKS", "1")) > 1:
        raise ValueError("--device cannot be used with distributed SLURM training.")

    setup_device, local_rank, rank, world = optollama.utils.setup_run(cfg, make_dirs=False)
    device = torch.device(args.device or setup_device)
    if device.type == "cuda":
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("highest" if joint and nested(block, "MODEL", "FP32_ATTENTION", default=True) else "high")
    tokens, token_to_idx, idx_to_token, _, _, _, eos_idx, pad_idx, msk_idx = optollama.data.init_tokens(cfg["TOKENS_PATH"])
    material_names = optollama.data.material_names_from_tokens(tokens)
    catalog = optollama.data.load_material_catalog(cfg["MATERIALS_PATH"], material_names)
    max_candidates = int(nested(block, "MATERIAL_BANK", "MAX_CANDIDATES", default=24))
    if len(catalog.names) > max_candidates and not bool(nested(block, "MATERIAL_BANK", "RANDOM_DISTRACTORS", default=True)):
        raise ValueError("The complete catalog exceeds MAX_CANDIDATES.")
    train_collator = make_collator(
        cfg, block, catalog, idx_to_token, eos_idx, pad_idx, msk_idx, train=True,
        seed=int(cfg.get("SEED", 0)) + rank * 17, joint=joint,
    )
    val_collator = make_collator(
        cfg, block, catalog, idx_to_token, eos_idx, pad_idx, msk_idx, train=False,
        seed=int(cfg.get("SEED", 0)) + 10_000 + rank * 17, joint=joint,
    )
    train_n = int(args.max_train_samples or (world * 2 if args.smoke_test else cfg["NUM_SAMPLES_TRAIN"]))
    val_n = int(args.max_val_samples or (world if args.smoke_test else cfg["NUM_SAMPLES_TEST"]))
    train_dataset, train_loader = make_loader(
        cfg, split="train", collator=train_collator, subset_n=train_n, rank=rank, world_size=world
    )
    _, val_loader = make_loader(cfg, split="test", collator=val_collator, subset_n=val_n, rank=rank, world_size=world)
    if joint:
        if len(train_loader) * int(cfg["TRAIN_BATCH_SIZE"]) * world != train_n:
            raise ValueError("Joint DFM requires all requested training rows and a sample count divisible by the global batch size.")
        if len(val_loader.dataset) * world != val_n:
            raise ValueError("Joint DFM requires all requested validation rows, evenly partitioned across ranks.")
    if int(train_cfg.get("CHECKPOINT_EVERY_STEPS", 0)) and not hasattr(train_dataset, "set_resume_batch"):
        raise ValueError("Mid-epoch checkpoints require DATA_LOADING.SHARDED_LOADING: true.")

    query_cfg = nested(block, "QUERY", default={}) or {}
    query_points = (
        int(query_cfg.get("POINTS", len(cfg["WAVELENGTHS"])))
        if query_cfg.get("ENABLED", False)
        else len(cfg["WAVELENGTHS"])
    )
    flow_config = None
    if joint:
        model_config = optollama.model.OpenLayerJointDFMConfig.from_mapping(block)
        if args.smoke_test:
            model_config = replace(model_config, d_model=32, n_blocks=2, n_heads=4, ffn_multiplier=2.0, query_encoder_blocks=1)
        for key in ("COUNT_LOSS_WEIGHT", "THICKNESS_LOSS_WEIGHT"):
            value = float(train_cfg.get(key, 1.0))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive.")
        process = "joint_layer_dfm"
        raw_model = optollama.model.OpenLayerJointDFM(model_config).to(device)
    else:
        model_config = model_config_from_mapping(block, (3, query_points))
        if model_config.query_spectrum != bool(query_cfg.get("ENABLED", False)):
            raise ValueError("MODEL.QUERY_SPECTRUM and QUERY.ENABLED must agree.")
        if args.smoke_test:
            model_config = replace(
                model_config, d_model=32, n_blocks=2, n_heads=4, ffn_multiplier=2.0, kernel_size=3,
                hybrid_dilations=(1, 2), spectrum_encoder_blocks=1, spectrum_encoder_heads=4,
            )
        process = str(nested(block, "DENOISING", "PROCESS", default="remask"))
        if process not in {"remask", "discrete_flow"}:
            raise ValueError(f"Unknown open-vocabulary depth process: {process}")
    if not joint and process == "discrete_flow":
        flow_config = optollama.model.DiscreteFlowConfig.from_dict(nested(block, "DENOISING", "FLOW", default={}))
        if float(train_cfg.get("RANDOM_REPLACE_PROB", 0.0)) != 0.0:
            raise ValueError("DFM requires TRAIN.RANDOM_REPLACE_PROB: 0.")
        if any(float(train_cfg.get(key, 1.0)) != 1.0 for key in (
            "VOID_LOSS_WEIGHT", "CORRUPTED_LOSS_WEIGHT", "UNCORRUPTED_LOSS_WEIGHT"
        )) or bool(nested(train_cfg, "BOUNDARY_LOSS", "ENABLED", default=False)) or bool(
            train_cfg.get("LOSS_ON_CORRUPTED_ONLY", False)
        ):
            raise ValueError("DFM clean-field posterior training requires unweighted cross-entropy.")
        raw_model = optollama.model.OpenVocabularyDepthFieldDFM(model_config, flow_config).to(device)
    elif not joint:
        raw_model = optollama.model.OpenVocabularyDepthFieldDiffusion(model_config).to(device)
    base_lr = float(train_cfg.get("LEARNING_RATE", 5.0e-5))
    optimizer_cfg = train_cfg.get("OPTIMIZER") or {}
    fused_optimizer = bool(optimizer_cfg.get("FUSED", True)) and device.type == "cuda"
    optimizer_kwargs = {
        "lr": base_lr,
        "weight_decay": float(train_cfg.get("WEIGHT_DECAY", 0.01)),
    }
    try:
        optimizer = torch.optim.AdamW(raw_model.parameters(), **optimizer_kwargs, fused=fused_optimizer)
    except (RuntimeError, TypeError):
        if not fused_optimizer:
            raise
        fused_optimizer = False
        optimizer = torch.optim.AdamW(raw_model.parameters(), **optimizer_kwargs)
    amp_dtype = resolve_amp_dtype(bool(train_cfg.get("AMP", True)), device, str(train_cfg.get("AMP_DTYPE", "auto")))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    lr_schedule = train_cfg.get("LR_SCHEDULE")
    corruption = optollama.model.DepthFieldCorruptionConfig.from_dict(nested(block, "DENOISING", "CORRUPTION", default={}))
    output_dir = Path(block.get("OUT_DIR") or cfg["OUTPUT_PATH"])
    if args.smoke_test:
        output_dir = output_dir / "smoke-test"
    epochs = 1 if args.smoke_test else int(args.epochs or train_cfg.get("EPOCHS", 20))
    start_epoch = 0
    history: list[dict[str, Any]] = []
    step_evaluations: list[dict[str, Any]] = []
    resume_cursor: dict[str, Any] | None = None
    rank_resume_state: dict[str, Any] | None = None
    ema_cfg = train_cfg.get("EMA") or {}
    ema = ModelEma(raw_model, float(ema_cfg.get("DECAY", 0.9998))) if bool(ema_cfg.get("ENABLED", True)) else None
    resume = args.resume or nested(cfg, "CHECKPOINT", "RESUME")
    checkpoint_extra: dict[str, Any] = {}
    if resume:
        loaded_epoch, blob = optollama.utils.load_checkpoint(
            str(resume), raw_model, optimizer=optimizer, scaler=scaler, map_location="cpu"
        )
        start_epoch = int(loaded_epoch or 0)
        checkpoint_extra = blob.get("extra") or {}
        if joint and (checkpoint_extra.get(model_metadata_key) != model_config.to_dict() or "optimizer_state" not in blob):
            raise ValueError("Joint DFM resume requires a full checkpoint with the same joint architecture.")
        saved_process = checkpoint_extra.get("denoising_process", "remask")
        saved_flow = checkpoint_extra.get("discrete_flow_config")
        if saved_process != process or (
            flow_config is not None and (
                not isinstance(saved_flow, dict)
                or optollama.model.DiscreteFlowConfig.from_dict(saved_flow) != flow_config
            )
        ):
            raise ValueError("Checkpoint denoising process or discrete-flow source differs from this config.")
        history = list(checkpoint_extra.get("history") or [])
        step_evaluations = list(checkpoint_extra.get("step_evaluations") or [])
        if ema is not None and isinstance(checkpoint_extra.get("ema"), dict):
            ema.load_state_dict(checkpoint_extra["ema"], raw_model)
        resume_cursor = checkpoint_extra.get("resume_cursor")
        if resume_cursor is not None:
            expected = {
                "world_size": world,
                "batch_size": int(cfg["TRAIN_BATCH_SIZE"]),
                "train_samples": train_n,
                "dataset_paths": [str(path) for path in train_dataset.paths],
            }
            for key, value in expected.items():
                if resume_cursor.get(key) != value:
                    raise ValueError(f"Mid-epoch resume requires unchanged {key}: {resume_cursor.get(key)!r} != {value!r}")
            start_epoch = int(resume_cursor["epoch"])
            rank_resume_state = resume_cursor["rank_states"][rank]
            train_collator.calls = int(rank_resume_state["rng"]["train_collator_calls"])
            val_collator.calls = int(rank_resume_state["rng"]["val_collator_calls"])

    model: torch.nn.Module = raw_model
    compile_cfg = train_cfg.get("COMPILE") or {}
    compile_enabled = bool(compile_cfg.get("ENABLED", False)) and device.type == "cuda" and not args.smoke_test
    if compile_enabled:
        compile_fn = getattr(torch, "compile", None)
        if compile_fn is None:
            raise RuntimeError("OPEN_VOCAB_DEPTH_FIELD.TRAIN.COMPILE is enabled, but this PyTorch has no torch.compile.")
        if bool(compile_cfg.get("SUPPRESS_ERRORS", True)):
            torch._dynamo.config.suppress_errors = True
        model = compile_fn(
            model,
            mode=str(compile_cfg.get("MODE", "default")),
            fullgraph=bool(compile_cfg.get("FULLGRAPH", False)),
            dynamic=bool(compile_cfg.get("DYNAMIC", False)),
        )
    ddp_cfg = train_cfg.get("DDP") or {}
    if optollama.utils.is_ddp():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            static_graph=bool(ddp_cfg.get("STATIC_GRAPH", True)),
            gradient_as_bucket_view=bool(ddp_cfg.get("GRADIENT_AS_BUCKET_VIEW", True)),
        )
    gradient_diagnostic = None
    if bool(train_cfg.get("DIAGNOSE_LOCAL_GRADIENTS", False)) and optollama.utils.is_ddp():
        gradient_diagnostic = LocalGradientDiagnostic(raw_model)
        model.register_comm_hook(gradient_diagnostic, diagnostic_allreduce)
    global_seen = (
        int(resume_cursor["epoch_base_global_seen"])
        if resume_cursor is not None
        else resume_global_samples(history, start_epoch, train_n)
    )
    global_steps = int(
        (resume_cursor["epoch_base_global_steps"] if resume_cursor is not None else checkpoint_extra.get("global_optimizer_steps", 0))
        or global_seen // max(1, int(cfg["TRAIN_BATCH_SIZE"]) * world)
    )
    log_every_steps = max(1, int(train_cfg.get("LOG_EVERY_STEPS", 50)))
    eval_every_steps = max(0, int(train_cfg.get("EVAL_EVERY_STEPS", 0)))
    checkpoint_every_steps = max(0, int(train_cfg.get("CHECKPOINT_EVERY_STEPS", 0)))
    ema_update_every_steps = max(1, int(ema_cfg.get("UPDATE_EVERY_STEPS", 1)))
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if optollama.utils.is_ddp():
        torch.distributed.barrier()
    diagnostic_dir = output_dir / "nonfinite_diagnostics" if bool(train_cfg.get("DIAGNOSE_LOCAL_GRADIENTS", False)) else None
    if diagnostic_dir is not None:
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        print(
            f"{label} model: parameters={sum(p.numel() for p in model.parameters()):,}, world={world}, batch/rank={cfg['TRAIN_BATCH_SIZE']}"
        )
        if joint:
            print(f"Continuous time: s~Uniform(0,1), mask probability=(1-s)^2, linear thickness flow; held-out loss rows={val_n:,}")
        else:
            print(
                f"Continuous time: t~Uniform(0,1), noise_probability=t^2; validation sampling steps={nested(block, 'EVAL', 'SAMPLING_STEPS', default=64)}"
            )
        print(f"Denoising process: {process}" + (f", source={flow_config.to_dict()}" if flow_config else ""))
        print(
            f"Optical condition: angle={nested(block, 'OPTICAL_CONDITION', 'ANGLE_DEG', default=0.0)}deg, polarization={nested(block, 'OPTICAL_CONDITION', 'POLARIZATION', default='s')}"
        )
        print(
            "Optimizations: "
            f"compile={compile_enabled}, fused_adamw={fused_optimizer}, "
            f"log_every={log_every_steps}, eval_every={eval_every_steps or 'epoch'}, "
            f"ema_every={ema_update_every_steps}"
        )
        print(
            f"Stability: amp_dtype={amp_dtype}, fp32_attention={model_config.fp32_attention}, "
            f"local_gradient_diagnostics={gradient_diagnostic is not None}, "
            f"checkpoint_every={checkpoint_every_steps or 'epoch'}"
        )
    best = min((float(item["val"]["loss"]) for item in history), default=math.inf)
    for epoch in range(start_epoch, epochs):
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        if rank_resume_state is not None:
            train_dataset.set_resume_batch(int(rank_resume_state["next_batch"]), int(cfg["TRAIN_BATCH_SIZE"]))
        max_train_steps = 1 if args.smoke_test else args.max_train_steps

        def checkpoint_metadata(optimizer_step: int, cursor: dict[str, Any] | None = None) -> dict[str, Any]:
            return {
                model_metadata_key: model_config.to_dict(),
                "denoising_process": process,
                "discrete_flow_config": flow_config.to_dict() if flow_config is not None else None,
                "material_names": list(catalog.names),
                "config_path": args.config,
                "history": history,
                "step_evaluations": step_evaluations,
                "global_optimizer_steps": optimizer_step,
                "continuous_time": True,
                "ema": ema.state_dict() if ema is not None else None,
                "resume_cursor": cursor,
            }

        def checkpoint_at_step(step: int, samples_seen: int, next_batch: int, state: dict[str, Any]) -> None:
            state["rng"] = capture_rank_state(train_collator, val_collator)
            rank_states: list[dict[str, Any] | None] = [None] * world
            if optollama.utils.is_ddp():
                torch.distributed.all_gather_object(rank_states, state)
            else:
                rank_states[0] = state
            cursor = {
                "epoch": epoch,
                "next_batch": next_batch,
                "epoch_base_global_seen": global_seen,
                "epoch_base_global_steps": global_steps,
                "world_size": world,
                "batch_size": int(cfg["TRAIN_BATCH_SIZE"]),
                "train_samples": train_n,
                "dataset_paths": [str(path) for path in train_dataset.paths],
                "rank_states": rank_states,
            }
            save_error = None
            if rank == 0:
                try:
                    optollama.utils.save_checkpoint(
                        str(output_dir / f"{checkpoint_prefix}-progress.pt"),
                        model=raw_model,
                        optimizer=optimizer,
                        scaler=scaler,
                        epoch=epoch,
                        extra=checkpoint_metadata(step, cursor),
                    )
                    print(f"Saved resumable {label.lower()} checkpoint at step {step:,}, batch {next_batch:,}.")
                except Exception as exc:
                    save_error = f"{type(exc).__name__}: {exc}"
            errors = [save_error]
            if optollama.utils.is_ddp():
                torch.distributed.broadcast_object_list(errors, src=0)
            if errors[0] is not None:
                raise RuntimeError(f"Could not save mid-epoch checkpoint: {errors[0]}")

        def evaluate_at_step(step: int, samples_seen: int, train_running: dict[str, float]) -> None:
            """Run and persist lightweight validation at a global optimizer-step boundary."""
            validate_ema = bool(ema is not None and ema_cfg.get("VALIDATE", True))
            validation_context = ema.apply(model) if validate_ema and ema is not None else nullcontext()
            with validation_context:
                interval_val = run_epoch(
                    model,
                    val_loader,
                    device,
                    train_cfg,
                    corruption,
                    optimizer=None,
                    scaler=scaler,
                    amp_dtype=amp_dtype,
                    epoch=epoch,
                    epochs=epochs,
                    max_steps=None,
                    base_lr=base_lr,
                    lr_schedule=lr_schedule,
                    global_samples_seen=samples_seen,
                    global_optimizer_steps=step,
                    log_every_steps=log_every_steps,
                )
            record = {
                "epoch": epoch + 1,
                "global_step": step,
                "global_samples_seen": samples_seen,
                "train_running": train_running,
                "val": interval_val,
            }
            step_evaluations.append(record)
            if rank == 0:
                optollama.utils.save_as_json(
                    str(output_dir / f"{checkpoint_prefix}-step-evaluations.json"), step_evaluations
                )
                print(
                    f"{label} step {step:,}: val_loss={interval_val['loss']:.6f}, "
                    f"val_acc={100 * interval_val['accuracy']:.2f}%" + joint_metric_suffix(interval_val, joint)
                )

        train = run_epoch(
            model,
            train_loader,
            device,
            train_cfg,
            corruption,
            optimizer=optimizer,
            scaler=scaler,
            amp_dtype=amp_dtype,
            epoch=epoch,
            epochs=epochs,
            max_steps=max_train_steps,
            base_lr=base_lr,
            lr_schedule=lr_schedule,
            global_samples_seen=global_seen,
            ema=ema,
            global_optimizer_steps=global_steps,
            log_every_steps=log_every_steps,
            ema_update_every_steps=ema_update_every_steps,
            eval_every_steps=eval_every_steps,
            on_evaluation_step=evaluate_at_step if eval_every_steps else None,
            checkpoint_every_steps=checkpoint_every_steps,
            on_checkpoint_step=checkpoint_at_step if checkpoint_every_steps else None,
            resume_state=rank_resume_state,
            gradient_diagnostic=gradient_diagnostic,
            diagnostic_dir=diagnostic_dir,
        )
        rank_resume_state = None
        global_seen = int(train["global_samples_seen"])
        global_steps = int(train["global_optimizer_steps"])
        validate_ema = bool(ema is not None and ema_cfg.get("VALIDATE", True))
        if step_evaluations and int(step_evaluations[-1]["global_step"]) == global_steps:
            val = dict(step_evaluations[-1]["val"])
        else:
            validation_context = ema.apply(model) if validate_ema and ema is not None else nullcontext()
            with validation_context:
                val = run_epoch(
                    model,
                    val_loader,
                    device,
                    train_cfg,
                    corruption,
                    optimizer=None,
                    scaler=scaler,
                    amp_dtype=amp_dtype,
                    epoch=epoch,
                    epochs=epochs,
                    max_steps=None,
                    base_lr=base_lr,
                    lr_schedule=lr_schedule,
                    global_samples_seen=global_seen,
                    global_optimizer_steps=global_steps,
                    log_every_steps=log_every_steps,
                )
        history.append({"epoch": epoch + 1, "train": train, "val": val})
        if rank == 0:
            extra = checkpoint_metadata(global_steps)
            optollama.utils.save_checkpoint(
                str(output_dir / f"{checkpoint_prefix}-last.pt"),
                model=raw_model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                extra=extra,
            )
            if val["loss"] < best:
                best = val["loss"]
                optollama.utils.save_checkpoint(
                    str(output_dir / f"{checkpoint_prefix}-best.pt"),
                    model=raw_model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    extra=extra,
                )
            optollama.utils.save_as_json(str(output_dir / f"{checkpoint_prefix}-history.json"), history)
            print(
                f"{label} epoch {epoch + 1}: val_loss={val['loss']:.6f}, val_acc={100 * val['accuracy']:.2f}%, best={best:.6f}"
                + joint_metric_suffix(val, joint)
            )
    if optollama.utils.is_ddp():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
