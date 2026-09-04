from __future__ import annotations

import argparse
import math
import os
from collections.abc import Sized
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import tqdm  # type: ignore[import-untyped]
from torch.utils.data import DataLoader, Subset

import optollama.data
import optollama.evaluation.simulation
import optollama.model
import optollama.utils


def parse_args() -> argparse.Namespace:
    """Parse training overrides for the open-layer MVP."""
    parser = argparse.ArgumentParser(description="Train the open-vocabulary layer-space flow model.")
    parser.add_argument("--config", default="configs/open_layer_flow_01.yaml")
    parser.add_argument("--resume", default=None, help="Optional full checkpoint to resume.")
    parser.add_argument("--device", default=None, help="Single-process device override.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-train-steps", type=int, default=None, help="Debug limit per epoch and rank.")
    return parser.parse_args()


def nested(mapping: dict[str, Any], *path: str, default: Any = None) -> Any:
    """Read a nested mapping value."""
    value: Any = mapping
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def resolve_device(value: str | None) -> torch.device:
    """Resolve a single-process device override."""
    if value:
        device = torch.device(value)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def resolve_amp_dtype(enabled: bool, device: torch.device, requested: str) -> torch.dtype | None:
    """Resolve CUDA autocast precision with BF16 preferred in auto mode."""
    if not enabled or device.type != "cuda":
        return None
    normalized = str(requested).lower()
    if normalized in {"bf16", "bfloat16"}:
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("AMP_DTYPE=bfloat16 was requested but this GPU does not support BF16.")
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized != "auto":
        raise ValueError(f"Unknown AMP_DTYPE={requested!r}.")
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def autocast_context(device: torch.device, dtype: torch.dtype | None) -> AbstractContextManager[Any]:
    """Return CUDA autocast or a no-op context."""
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move a collated tensor dictionary onto the model device."""
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def model_config_from_mapping(block: dict[str, Any]) -> optollama.model.OpenLayerFlowConfig:
    """Construct model metadata from ``OPEN_LAYER`` config."""
    model = nested(block, "MODEL", default={}) or {}
    query = nested(block, "QUERY", default={}) or {}
    process = nested(block, "DENOISING", default={}) or {}
    corruption = nested(process, "CORRUPTION", default={}) or {}
    thickness = nested(block, "THICKNESS", default={}) or {}
    channels = tuple(str(value) for value in query.get("CHANNELS", ["R", "T"]))
    return optollama.model.OpenLayerFlowConfig(
        target_channels=len(channels),
        max_layers=int(block.get("MAX_LAYERS", 100)),
        d_model=int(model.get("D_MODEL", 512)),
        n_blocks=int(model.get("N_BLOCKS", 12)),
        n_heads=int(model.get("N_HEADS", 8)),
        ffn_multiplier=float(model.get("FFN_MULTIPLIER", 4.0)),
        query_encoder_blocks=int(model.get("QUERY_ENCODER_BLOCKS", 2)),
        dropout=float(model.get("DROPOUT", 0.0)),
        adaln_zero=bool(model.get("ADALN_ZERO", False)),
        wavelength_scale_nm=float(query.get("WAVELENGTH_SCALE_NM", 1_000.0)),
        wavelength_fourier_bands=int(query.get("FOURIER_BANDS", 4)),
        material_process=str(process.get("MATERIAL_PROCESS", "monotonic")),
        material_corruption_mode=str(corruption.get("MODE", "iid")),
        material_iid_fraction=float(corruption.get("IID_FRACTION", 1.0)),
        material_span_fraction=float(corruption.get("SPAN_FRACTION", 0.0)),
        material_span_min_layers=int(corruption.get("SPAN_MIN_LAYERS", 2)),
        material_span_max_layers=int(corruption.get("SPAN_MAX_LAYERS", 8)),
        material_span_scale_with_noise=bool(corruption.get("SPAN_SCALE_WITH_NOISE", True)),
        material_random_replace_prob=float(process.get("RANDOM_REPLACE_PROB", 0.0)),
        material_random_replace_schedule=str(corruption.get("RANDOM_REPLACE_SCHEDULE", "constant")),
        material_random_replace_power=float(corruption.get("RANDOM_REPLACE_POWER", 1.0)),
        material_corrupted_loss_weight=float(process.get("CORRUPTED_LOSS_WEIGHT", 1.0)),
        material_uncorrupted_loss_weight=float(process.get("UNCORRUPTED_LOSS_WEIGHT", 0.0)),
        thickness_loss_weight=float(process.get("THICKNESS_LOSS_WEIGHT", 1.0)),
        thickness_huber_delta=float(process.get("THICKNESS_HUBER_DELTA", 0.1)),
        min_thickness_nm=float(thickness.get("MIN_NM", 5.0)),
        max_thickness_nm=float(thickness.get("MAX_LAYER_NM", thickness.get("MAX_NM", 10_000.0))),
        max_total_thickness_nm=float(thickness.get("MAX_TOTAL_NM", 10_000.0)),
    )


def make_collator(
    *,
    cfg: dict[str, Any],
    block: dict[str, Any],
    catalog: optollama.data.MaterialCatalog,
    idx_to_token: dict[int, str],
    eos_idx: int,
    pad_idx: int,
    msk_idx: int,
    train: bool,
    seed: int,
) -> optollama.data.OpenLayerBatchCollator:
    """Build the train or deterministic validation collator."""
    query = nested(block, "QUERY", default={}) or {}
    bank = nested(block, "MATERIAL_BANK", default={}) or {}
    thickness = nested(block, "THICKNESS", default={}) or {}
    synchronize_query_shapes = bool(query.get("SYNC_SHAPES_ACROSS_RANKS", True))
    query_shape_seed = int(cfg.get("SEED", 0)) + (0 if train else 10_000) if synchronize_query_shapes else None
    transform = optollama.data.ThicknessTransform(
        min_nm=float(thickness.get("MIN_NM", 5.0)),
        max_nm=float(thickness.get("MAX_LAYER_NM", thickness.get("MAX_NM", 10_000.0))),
    )
    return optollama.data.OpenLayerBatchCollator(
        wavelengths_nm=cfg["WAVELENGTHS"],
        catalog=catalog,
        idx_to_token=idx_to_token,
        eos_idx=eos_idx,
        pad_idx=pad_idx,
        msk_idx=msk_idx,
        channels=query.get("CHANNELS", ["R", "T"]),
        max_layers=int(block.get("MAX_LAYERS", 100)),
        max_candidates=int(bank.get("MAX_CANDIDATES", 15)),
        min_query_points=int(query.get("MIN_POINTS", 64)),
        max_query_points=int(query.get("MAX_POINTS", len(cfg["WAVELENGTHS"]))),
        query_sampling=str(query.get("SAMPLING", "mixed") if train else "full"),
        randomize_candidates=bool(bank.get("RANDOMIZE_ORDER", True) if train else False),
        random_distractors=bool(bank.get("RANDOM_DISTRACTORS", True)),
        holdout_materials=bank.get("HOLDOUT_MATERIALS", []) if train else (),
        merge_adjacent=bool(block.get("MERGE_ADJACENT", True)),
        thickness_transform=transform,
        coverage_tolerance_nm=float(query.get("MATERIAL_COVERAGE_TOLERANCE_NM", 0.0)),
        seed=seed,
        query_shape_seed=query_shape_seed,
    )


def make_loader(
    cfg: dict[str, Any],
    *,
    split: str,
    collator: optollama.data.OpenLayerBatchCollator,
    subset_n: int,
    rank: int,
    world_size: int,
) -> tuple[torch.utils.data.Dataset, DataLoader]:
    """Build an eager or sharded loader while preserving DDP rank partitioning."""
    train = split == "train"
    prefix = "DATA_PATH_TRAIN" if train else "DATA_PATH_TEST"
    paths = sorted(str(value) for key, value in cfg.items() if key == prefix or key.startswith(f"{prefix}_"))
    if not paths:
        raise KeyError(f"No configured {prefix} paths were found.")
    batch_size = int(cfg["TRAIN_BATCH_SIZE" if train else "TEST_BATCH_SIZE"])
    workers = int(cfg.get("NUM_WORKERS", 0))
    if bool(cfg.get("SHARDED_LOADING", False)):
        if workers and rank == 0:
            print(
                f"Sharded open-layer loading forces NUM_WORKERS=0 (configured {workers}) "
                "to prevent each worker from replaying the rank-local sample range."
            )
        sharded_dataset = optollama.data.ShardedSpectraDataset(
            paths,
            split=split,
            subset_n=subset_n,
            rank=rank,
            world_size=world_size,
            seed=int(cfg.get("SEED", 0)),
            shuffle=train,
        )
        loader = DataLoader(
            sharded_dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=train,
            collate_fn=collator,
        )
        return sharded_dataset, loader

    eager_dataset: torch.utils.data.Dataset = optollama.data.SpectraDataset(paths)
    if subset_n < len(cast(Sized, eager_dataset)):
        eager_dataset = Subset(eager_dataset, range(subset_n))
    sampler: torch.utils.data.Sampler[Any] | None = None
    if world_size > 1:
        sampler = torch.utils.data.DistributedSampler(eager_dataset, shuffle=train, drop_last=train)
    loader = DataLoader(
        eager_dataset,
        batch_size=batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
        collate_fn=collator,
    )
    return eager_dataset, loader


def reduce_totals(totals: torch.Tensor) -> torch.Tensor:
    """Sum metric accumulators over DDP ranks."""
    if optollama.utils.is_ddp():
        torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
    return totals


def synchronized_finite(value: torch.Tensor) -> bool:
    """Return whether a tensor is finite on every distributed rank."""
    finite = torch.tensor(
        int(bool(torch.isfinite(value.detach()).all().item())),
        dtype=torch.int32,
        device=value.device,
    )
    if optollama.utils.is_ddp():
        torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
    return bool(finite.item())


def reduce_max(values: torch.Tensor) -> torch.Tensor:
    """Take distributed maxima for counters synchronized per optimizer step."""
    if optollama.utils.is_ddp():
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.MAX)
    return values


def averaged_metrics(totals: torch.Tensor, metric_keys: tuple[str, ...]) -> dict[str, float]:
    """Convert sample-weighted metric totals into running averages."""
    samples = float(totals[-1].item())
    if samples <= 0:
        return {key: 0.0 for key in metric_keys}
    return {key: float(totals[idx].item() / samples) for idx, key in enumerate(metric_keys)}


def select_mc_spectral_metrics(channel_mae: torch.Tensor) -> dict[str, torch.Tensor]:
    """Select coherent best-RT and best-RAT candidates from per-MC channel errors."""
    if channel_mae.ndim != 3 or channel_mae.shape[-1] != 3:
        raise ValueError("channel_mae must have shape [samples, mc, 3] in R/A/T order.")
    rt_mae = channel_mae[..., (0, 2)].mean(dim=-1)
    rat_mae = channel_mae.mean(dim=-1)
    rows = torch.arange(channel_mae.shape[0], device=channel_mae.device)
    best_rt_indices = rt_mae.argmin(dim=1)
    best_rat_indices = rat_mae.argmin(dim=1)
    best_rat_channels = channel_mae[rows, best_rat_indices]
    return {
        "best_rt_indices": best_rt_indices,
        "best_rat_indices": best_rat_indices,
        "best_rt_mae": rt_mae[rows, best_rt_indices],
        "best_rat_mae": rat_mae[rows, best_rat_indices],
        "rat_mae_at_best_rt": rat_mae[rows, best_rt_indices],
        "rt_mae_at_best_rat": rt_mae[rows, best_rat_indices],
        "r_mae_at_best_rat": best_rat_channels[:, 0],
        "a_mae_at_best_rat": best_rat_channels[:, 1],
        "t_mae_at_best_rat": best_rat_channels[:, 2],
    }


def concatenate_validation_records(records: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate validation record chunks along their sample dimension."""
    nonempty = [record for record in records if record]
    if not nonempty:
        return {}
    keys = tuple(nonempty[0])
    if any(tuple(record) != keys for record in nonempty[1:]):
        raise ValueError("Validation record chunks do not contain the same fields.")
    return {key: np.concatenate([record[key] for record in nonempty], axis=0) for key in keys}


def save_validation_spectra(
    path: Path,
    records: dict[str, np.ndarray],
    *,
    material_names: tuple[str, ...],
    sampling_steps: int,
) -> None:
    """Save target spectra, all MC predictions, errors, and stack metadata."""
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    channel_mae = torch.from_numpy(records["mae_per_channel"])
    selected = select_mc_spectral_metrics(channel_mae)
    payload: dict[str, Any] = dict(records)
    payload.update(
        format_version=np.asarray(1, dtype=np.int64),
        channel_order=np.asarray(("R", "A", "T")),
        material_names=np.asarray(material_names),
        sampling_steps=np.asarray(sampling_steps, dtype=np.int64),
        mae_rt=channel_mae[..., (0, 2)].mean(dim=-1).cpu().numpy(),
        mae_rat=channel_mae.mean(dim=-1).cpu().numpy(),
        best_rt_indices=selected["best_rt_indices"].cpu().numpy(),
        best_rat_indices=selected["best_rat_indices"].cpu().numpy(),
    )
    np.savez_compressed(path, **payload)


def unwrap_model(model: torch.nn.Module) -> optollama.model.OpenLayerFlow:
    """Return the concrete model behind an optional DDP wrapper."""
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return cast(optollama.model.OpenLayerFlow, model.module)
    return cast(optollama.model.OpenLayerFlow, model)


def compute_training_loss(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Compute the joint objective through the DDP wrapper when present."""
    core = unwrap_model(model)
    state = core.prepare_training_state(batch)
    outputs = model(
        wavelengths_nm=batch["wavelengths_nm"],
        target_spectrum=batch["target_spectrum"],
        query_mask=batch["query_mask"],
        candidate_nk=batch["candidate_nk"],
        candidate_mask=batch["candidate_mask"],
        material_ids=state["noised_materials"],
        thickness_state=state["thickness_state"],
        layer_mask=state["layer_mask"],
        timesteps=state["timesteps"],
    )
    return core.loss_from_training_state(outputs, state)


def run_loss_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    amp_dtype: torch.dtype | None,
    grad_clip: float,
    epoch: int,
    epochs: int,
    max_steps: int | None = None,
    max_consecutive_nonfinite_steps: int = 8,
) -> dict[str, float]:
    """Train or validate the denoising objectives for one epoch."""
    train = optimizer is not None
    model.train(train)
    metric_keys = (
        "loss",
        "material_loss",
        "thickness_loss",
        "material_accuracy",
        "full_material_accuracy",
        "mean_timestep",
        "corrupted_fraction",
        "masked_fraction",
        "replaced_fraction",
    )
    totals = torch.zeros(len(metric_keys) + 1, dtype=torch.float64, device=device)
    stability = torch.zeros(4, dtype=torch.long, device=device)
    consecutive_nonfinite = 0
    show_progress = not (torch.distributed.is_initialized() and torch.distributed.get_rank() != 0)
    progress = tqdm.tqdm(
        loader,
        desc=f"Epoch {epoch + 1}/{epochs} open-layer {'train' if train else 'val'}",
        disable=not show_progress,
    )
    for step, raw_batch in enumerate(progress):
        if max_steps is not None and step >= max_steps:
            break
        batch = move_batch(raw_batch, device)
        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
        context = nullcontext() if train else torch.no_grad()
        with context, autocast_context(device, amp_dtype):
            outputs = compute_training_loss(model, batch)
            loss = outputs["loss"]
        if not synchronized_finite(loss):
            if not train:
                raise FloatingPointError(f"Non-finite open-layer validation loss at epoch={epoch + 1}, step={step}.")
            stability[0] += 1
            consecutive_nonfinite += 1
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            if show_progress:
                tqdm.tqdm.write(
                    f"Skipped non-finite open-layer forward step at epoch={epoch + 1}, step={step}; "
                    f"consecutive={consecutive_nonfinite}."
                )
            limit = int(max_consecutive_nonfinite_steps)
            if limit <= 0 or consecutive_nonfinite >= limit:
                raise FloatingPointError(
                    f"Aborting after {consecutive_nonfinite} consecutive non-finite open-layer forward steps "
                    f"at epoch={epoch + 1}, step={step}."
                )
            continue
        if train:
            assert optimizer is not None
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_limit = float(grad_clip) if grad_clip > 0.0 else float("inf")
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_limit, error_if_nonfinite=False)
            if not synchronized_finite(norm):
                stability[1] += 1
                consecutive_nonfinite += 1
                if scaler.is_enabled():
                    scale = float(scaler.get_scale())
                    scaler.update(new_scale=max(scale * 0.5, 1.0))
                    stability[2] += 1
                optimizer.zero_grad(set_to_none=True)
                if show_progress:
                    tqdm.tqdm.write(
                        f"Skipped non-finite open-layer gradient step at epoch={epoch + 1}, step={step}; "
                        f"consecutive={consecutive_nonfinite}."
                    )
                limit = int(max_consecutive_nonfinite_steps)
                if limit <= 0 or consecutive_nonfinite >= limit:
                    raise FloatingPointError(
                        f"Aborting after {consecutive_nonfinite} consecutive non-finite open-layer gradient steps "
                        f"at epoch={epoch + 1}, step={step}."
                    )
                continue
            scale_before = float(scaler.get_scale()) if scaler.is_enabled() else None
            scaler.step(optimizer)
            scaler.update()
            amp_skipped = bool(scale_before is not None and float(scaler.get_scale()) < scale_before)
            if amp_skipped:
                stability[2] += 1
            else:
                stability[3] += 1
                consecutive_nonfinite = 0
        else:
            norm = torch.zeros((), device=device)

        batch_size = int(outputs["supervised_samples"].item())
        totals += torch.tensor(
            [float(outputs[key].detach().item()) * batch_size for key in metric_keys] + [float(batch_size)],
            dtype=torch.float64,
            device=device,
        )
        running = averaged_metrics(totals, metric_keys)
        progress.set_postfix(
            loss=f"{running['loss']:.4f}",
            mat=f"{100.0 * running['material_accuracy']:.1f}%",
            full=f"{100.0 * running['full_material_accuracy']:.1f}%",
            grad=f"{float(norm):.2f}",
            nf=f"{int(stability[0].item() + stability[1].item())}",
            amp_skip=f"{int(stability[2].item())}",
        )

    totals = reduce_totals(totals)
    stability = reduce_max(stability)
    samples = float(totals[-1].item())
    if samples <= 0:
        raise RuntimeError("No supervised samples remained after applying material holdouts.")
    return averaged_metrics(totals, metric_keys) | {
        "samples": int(samples),
        "nonfinite_forward_steps": int(stability[0].item()),
        "nonfinite_gradient_steps": int(stability[1].item()),
        "amp_skipped_steps": int(stability[2].item()),
        "optimizer_steps": int(stability[3].item()),
    }


@torch.no_grad()
def validate_tmm(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    catalog: optollama.data.MaterialCatalog,
    tmm_ctx: optollama.evaluation.simulation.TMMContext,
    material_to_token_id: dict[str, int],
    eos_idx: int,
    pad_idx: int,
    msk_idx: int,
    device: torch.device,
    max_samples: int,
    mc_samples: int,
    sampling_steps: int,
    save_spectra_path: Path | None = None,
) -> dict[str, Any]:
    """Measure exact-TMM R/A/T errors and optionally retain every MC spectrum."""
    core = unwrap_model(model)
    core.eval()
    channel_mae_chunks: list[torch.Tensor] = []
    record_chunks: list[dict[str, np.ndarray]] = []
    processed = 0
    for raw_batch in loader:
        if processed >= max_samples:
            break
        valid_rows = raw_batch["sample_mask"].to(dtype=torch.bool)
        if not bool(valid_rows.any()):
            continue
        raw_batch = {key: value[valid_rows] for key, value in raw_batch.items()}
        keep = min(int(raw_batch["target_spectrum"].shape[0]), max_samples - processed)
        raw_batch = {key: value[:keep] for key, value in raw_batch.items()}
        batch = move_batch(raw_batch, device)
        layer_counts = batch["layer_mask"].sum(dim=1)
        target_rat = raw_batch["target_spectrum_rat"].permute(0, 2, 1)
        predicted_spectra: list[torch.Tensor] = []
        channel_mae: list[torch.Tensor] = []
        predicted_material_ids: list[torch.Tensor] = []
        predicted_thickness_nm: list[torch.Tensor] = []
        for _ in range(mc_samples):
            sampled = core.sample(
                wavelengths_nm=batch["wavelengths_nm"],
                target_spectrum=batch["target_spectrum"],
                query_mask=batch["query_mask"],
                candidate_nk=batch["candidate_nk"],
                candidate_mask=batch["candidate_mask"],
                layer_counts=layer_counts,
                steps=sampling_steps,
            )
            runs = optollama.data.layer_batch_to_runs(
                sampled["material_ids"],
                sampled["thickness_nm"],
                batch["candidate_global_ids"],
                catalog,
                sampled["layer_mask"],
            )
            predicted = optollama.evaluation.simulation.simulate_material_runs(
                runs,
                tmm_ctx,
                material_to_token_id=material_to_token_id,
                eos=eos_idx,
                pad=pad_idx,
                msk=msk_idx,
            )
            target_on_tmm = target_rat.to(predicted.device)
            predicted_spectra.append(predicted.to(device="cpu", dtype=torch.float32))
            channel_mae.append((predicted - target_on_tmm).abs().mean(dim=-1).to(device="cpu", dtype=torch.float32))

            sampled_ids = sampled["material_ids"]
            sampled_mask = sampled["layer_mask"]
            global_ids = batch["candidate_global_ids"].gather(1, sampled_ids.clamp_min(0))
            global_ids = torch.where(sampled_mask, global_ids, -torch.ones_like(global_ids))
            max_layers = int(batch["layer_mask"].shape[1])
            padded_ids = torch.full((keep, max_layers), -1, dtype=torch.long, device=global_ids.device)
            padded_thickness = torch.zeros((keep, max_layers), dtype=torch.float32, device=global_ids.device)
            sampled_layers = int(global_ids.shape[1])
            padded_ids[:, :sampled_layers] = global_ids
            padded_thickness[:, :sampled_layers] = sampled["thickness_nm"]
            predicted_material_ids.append(padded_ids.cpu())
            predicted_thickness_nm.append(padded_thickness.cpu())

        batch_predicted = torch.stack(predicted_spectra, dim=1)
        batch_channel_mae = torch.stack(channel_mae, dim=1)
        channel_mae_chunks.append(batch_channel_mae)

        target_local_ids = raw_batch["material_targets"]
        target_layer_mask = raw_batch["layer_mask"] & target_local_ids.ge(0)
        target_global_ids = raw_batch["candidate_global_ids"].gather(1, target_local_ids.clamp_min(0))
        target_global_ids = torch.where(target_layer_mask, target_global_ids, -torch.ones_like(target_global_ids))
        record_chunks.append(
            {
                "sample_indices": raw_batch["sample_indices"].cpu().numpy(),
                "wavelengths_nm": raw_batch["wavelengths_nm"].to(torch.float32).cpu().numpy(),
                "target_spectra": target_rat.to(torch.float32).cpu().numpy(),
                "predicted_spectra": batch_predicted.numpy(),
                "mae_per_channel": batch_channel_mae.numpy(),
                "target_material_ids": target_global_ids.cpu().numpy(),
                "target_thickness_nm": raw_batch["thickness_nm"].to(torch.float32).cpu().numpy(),
                "predicted_material_ids": torch.stack(predicted_material_ids, dim=1).numpy(),
                "predicted_thickness_nm": torch.stack(predicted_thickness_nm, dim=1).numpy(),
                "layer_counts": layer_counts.to(device="cpu", dtype=torch.long).numpy(),
            }
        )
        processed += keep

    local_channel_mae = (
        torch.cat(channel_mae_chunks, dim=0)
        if channel_mae_chunks
        else torch.empty((0, mc_samples, 3), dtype=torch.float32)
    )
    local_records = concatenate_validation_records(record_chunks)
    if optollama.utils.is_ddp():
        gathered_errors: list[Any] = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered_errors, local_channel_mae.numpy())
        local_channel_mae = torch.from_numpy(np.concatenate(gathered_errors, axis=0))
        if save_spectra_path is not None:
            gathered_records: list[Any] = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered_records, local_records)
            local_records = concatenate_validation_records(gathered_records)

    if local_channel_mae.shape[0] == 0:
        return {
            "r_mae_mean": math.nan,
            "a_mae_mean": math.nan,
            "t_mae_mean": math.nan,
            "rt_mae_mean": math.nan,
            "rat_mae_mean": math.nan,
            "samples": 0,
            "mc_samples": mc_samples,
        }

    selected = select_mc_spectral_metrics(local_channel_mae)

    def mean_median(values: torch.Tensor) -> tuple[float, float]:
        return float(values.mean().item()), float(values.median().item())

    r_mean, r_median = mean_median(selected["r_mae_at_best_rat"])
    a_mean, a_median = mean_median(selected["a_mae_at_best_rat"])
    t_mean, t_median = mean_median(selected["t_mae_at_best_rat"])
    rt_mean, rt_median = mean_median(selected["best_rt_mae"])
    rat_mean, rat_median = mean_median(selected["best_rat_mae"])
    rt_at_rat_mean, rt_at_rat_median = mean_median(selected["rt_mae_at_best_rat"])
    rat_at_rt_mean, rat_at_rt_median = mean_median(selected["rat_mae_at_best_rt"])

    rank = torch.distributed.get_rank() if optollama.utils.is_ddp() else 0
    if save_spectra_path is not None and rank == 0:
        save_validation_spectra(
            save_spectra_path,
            local_records,
            material_names=catalog.names,
            sampling_steps=sampling_steps,
        )
        print(f"Saved open-layer validation spectra -> {save_spectra_path}")

    return {
        "r_mae_mean": r_mean,
        "r_mae_median": r_median,
        "a_mae_mean": a_mean,
        "a_mae_median": a_median,
        "t_mae_mean": t_mean,
        "t_mae_median": t_median,
        "rt_mae_mean": rt_mean,
        "rt_mae_median": rt_median,
        "rat_mae_mean": rat_mean,
        "rat_mae_median": rat_median,
        "rt_mae_at_best_rat_mean": rt_at_rat_mean,
        "rt_mae_at_best_rat_median": rt_at_rat_median,
        "rat_mae_at_best_rt_mean": rat_at_rt_mean,
        "rat_mae_at_best_rt_median": rat_at_rt_median,
        "channel_metrics_selection": "best_rat",
        "samples": int(local_channel_mae.shape[0]),
        "mc_samples": mc_samples,
        "spectra_file": str(save_spectra_path) if save_spectra_path is not None else None,
    }


def main() -> None:
    """Train and checkpoint the open-layer MVP."""
    args = parse_args()
    cfg = optollama.utils.load_config_file(args.config)
    cfg["WAVELENGTHS"] = torch.arange(int(cfg["WAVELENGTH_MIN"]), int(cfg["WAVELENGTH_MAX"]) + 1, int(cfg["WAVELENGTH_STEPS"]))
    block = cfg.get("OPEN_LAYER") or {}
    train_cfg = nested(block, "TRAIN", default={}) or {}
    eval_cfg = nested(block, "EVAL", default={}) or {}
    if args.batch_size is not None:
        cfg["TRAIN_BATCH_SIZE"] = int(args.batch_size)
    if args.device is not None and int(os.getenv("SLURM_NTASKS", "1")) > 1:
        raise ValueError("--device cannot be combined with multi-process SLURM training.")

    output_dir = Path(block.get("OUT_DIR") or cfg["OUTPUT_PATH"])
    setup_device, local_rank, rank, world_size = optollama.utils.setup_run(cfg, make_dirs=False)
    device = torch.device(setup_device)
    if args.device is not None:
        device = resolve_device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)

    tokens, token_to_idx, idx_to_token, _, _, _, eos_idx, pad_idx, msk_idx = optollama.data.init_tokens(cfg["TOKENS_PATH"])
    material_names = optollama.data.material_names_from_tokens(tokens)
    catalog = optollama.data.load_material_catalog(cfg["MATERIALS_PATH"], material_names)
    train_collator = make_collator(
        cfg=cfg,
        block=block,
        catalog=catalog,
        idx_to_token=idx_to_token,
        eos_idx=eos_idx,
        pad_idx=pad_idx,
        msk_idx=msk_idx,
        train=True,
        seed=int(cfg.get("SEED", 0)) + rank * 17,
    )
    val_collator = make_collator(
        cfg=cfg,
        block=block,
        catalog=catalog,
        idx_to_token=idx_to_token,
        eos_idx=eos_idx,
        pad_idx=pad_idx,
        msk_idx=msk_idx,
        train=False,
        seed=int(cfg.get("SEED", 0)) + 10_000 + rank * 17,
    )
    train_n = int(args.max_train_samples or cfg["NUM_SAMPLES_TRAIN"])
    val_n = int(args.max_val_samples or cfg["NUM_SAMPLES_TEST"])
    train_dataset, train_loader = make_loader(
        cfg, split="train", collator=train_collator, subset_n=train_n, rank=rank, world_size=world_size
    )
    _, val_loader = make_loader(cfg, split="test", collator=val_collator, subset_n=val_n, rank=rank, world_size=world_size)

    model_config = model_config_from_mapping(block)
    model: torch.nn.Module = optollama.model.OpenLayerFlow(model_config).to(device)
    if optollama.utils.is_ddp():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
        )
    learning_rate = float(args.learning_rate or train_cfg.get("LEARNING_RATE", 1.0e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=float(train_cfg.get("WEIGHT_DECAY", 0.01)))
    amp_enabled = bool(train_cfg.get("AMP", True) and device.type == "cuda")
    amp_dtype = resolve_amp_dtype(amp_enabled, device, str(train_cfg.get("AMP_DTYPE", "auto")))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    grad_clip = float(train_cfg.get("GRAD_CLIP", 1.0))
    max_consecutive_nonfinite_steps = int(train_cfg.get("MAX_CONSECUTIVE_NONFINITE_STEPS", 8))
    epochs = int(args.epochs or train_cfg.get("EPOCHS", 10))
    start_epoch = 0
    history: list[dict[str, Any]] = []
    resume = args.resume or nested(cfg, "CHECKPOINT", "RESUME")
    if resume:
        loaded_epoch, blob = optollama.utils.load_checkpoint(
            str(resume), model, optimizer=optimizer, scaler=scaler, map_location="cpu"
        )
        start_epoch = int(loaded_epoch or 0)
        history = list(((blob.get("extra") or {}).get("history") or []))

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"Open-layer model: parameters={parameter_count:,}, train_samples={train_n:,}, "
            f"world={world_size}, batch/rank={cfg['TRAIN_BATCH_SIZE']}"
        )
        query_cfg = nested(block, "QUERY", default={}) or {}
        print(
            f"Open-layer query sampling: mode={query_cfg.get('SAMPLING', 'mixed')}, "
            f"points={query_cfg.get('MIN_POINTS', 64)}-{query_cfg.get('MAX_POINTS', len(cfg['WAVELENGTHS']))}, "
            f"sync_shapes_across_ranks={bool(query_cfg.get('SYNC_SHAPES_ACROSS_RANKS', True))}"
        )
        print(f"Open-layer decoder conditioning: adaln_zero={model_config.adaln_zero}")
        print(
            f"Open-layer material process: {model_config.material_process}, "
            f"corruption={model_config.material_corruption_mode}, "
            f"iid/span={model_config.material_iid_fraction:g}/{model_config.material_span_fraction:g}, "
            f"span_layers={model_config.material_span_min_layers}-{model_config.material_span_max_layers}, "
            f"random_replace={model_config.material_random_replace_prob:g}/"
            f"{model_config.material_random_replace_schedule}, "
            f"loss_weights={model_config.material_corrupted_loss_weight:g}/"
            f"{model_config.material_uncorrupted_loss_weight:g}"
        )
    if optollama.utils.is_ddp():
        torch.distributed.barrier()

    tmm_ctx: optollama.evaluation.simulation.TMMContext | None = None
    material_to_token_id: dict[str, int] | None = None
    if bool(eval_cfg.get("TMM_ENABLED", True)):
        depth_vocab = optollama.data.build_depth_field_vocab(tokens, token_to_idx)
        material_to_token_id = {name: depth_vocab.token_options[name][0].token_id for name in material_names}
        tmm_device = (
            device
            if str(eval_cfg.get("TMM_DEVICE", "same")).lower() in {"same", "auto"}
            else torch.device(str(eval_cfg["TMM_DEVICE"]))
        )
        tmm_ctx = optollama.evaluation.simulation.TMMContext.make(cfg, idx_to_token, tmm_device)

    best_loss = min((float(item["val"]["loss"]) for item in history), default=math.inf)
    for epoch in range(start_epoch, epochs):
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        elif hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        train_metrics = run_loss_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            amp_dtype=amp_dtype,
            grad_clip=grad_clip,
            epoch=epoch,
            epochs=epochs,
            max_steps=args.max_train_steps,
            max_consecutive_nonfinite_steps=max_consecutive_nonfinite_steps,
        )
        val_metrics = run_loss_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
            scaler=scaler,
            amp_dtype=amp_dtype,
            grad_clip=grad_clip,
            epoch=epoch,
            epochs=epochs,
        )
        tmm_metrics = None
        if tmm_ctx is not None and material_to_token_id is not None:
            tmm_metrics = validate_tmm(
                model=model,
                loader=val_loader,
                catalog=catalog,
                tmm_ctx=tmm_ctx,
                material_to_token_id=material_to_token_id,
                eos_idx=eos_idx,
                pad_idx=pad_idx,
                msk_idx=msk_idx,
                device=device,
                max_samples=int(eval_cfg.get("TMM_MAX_SAMPLES", 32)),
                mc_samples=int(eval_cfg.get("MC_SAMPLES", 4)),
                sampling_steps=int(eval_cfg.get("SAMPLING_STEPS", 32)),
                save_spectra_path=(
                    output_dir
                    / str(eval_cfg.get("SPECTRA_DIR", "validation_spectra"))
                    / f"epoch_{epoch + 1:04d}.npz"
                    if bool(eval_cfg.get("RECORD_SPECTRA", True))
                    else None
                ),
            )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics, "tmm": tmm_metrics})
        extra = {
            "open_layer_config": model_config.to_dict(),
            "channels": list(nested(block, "QUERY", "CHANNELS", default=["R", "T"])),
            "training_material_names": list(material_names),
            "holdout_material_names": list(nested(block, "MATERIAL_BANK", "HOLDOUT_MATERIALS", default=[])),
            "config_path": str(args.config),
            "history": history,
        }
        if rank == 0:
            last_path = output_dir / "open-layer-last.pt"
            optollama.utils.save_checkpoint(
                str(last_path), model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, extra=extra
            )
            if val_metrics["loss"] < best_loss:
                best_loss = float(val_metrics["loss"])
                optollama.utils.save_checkpoint(
                    str(output_dir / "open-layer-best.pt"),
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    extra=extra,
                )
            optollama.utils.save_as_json(str(output_dir / "open-layer-history.json"), history)
            tmm_note = (
                ""
                if tmm_metrics is None
                else (
                    f", tmm_rt_mae={tmm_metrics['rt_mae_mean']:.6f}, "
                    f"tmm_rat_mae={tmm_metrics['rat_mae_mean']:.6f}, "
                    f"R/A/T@bestRAT={tmm_metrics['r_mae_mean']:.6f}/"
                    f"{tmm_metrics['a_mae_mean']:.6f}/{tmm_metrics['t_mae_mean']:.6f}"
                )
            )
            print(f"Open-layer epoch {epoch + 1}: val_loss={val_metrics['loss']:.6f}{tmm_note}, best={best_loss:.6f}")

    if optollama.utils.is_ddp():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
