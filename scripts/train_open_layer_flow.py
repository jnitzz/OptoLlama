from __future__ import annotations

import argparse
import math
import os
from collections.abc import Sized
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as functional
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
        wavelength_scale_nm=float(query.get("WAVELENGTH_SCALE_NM", 1_000.0)),
        wavelength_fourier_bands=int(query.get("FOURIER_BANDS", 4)),
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
            num_workers=workers,
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


def unwrap_model(model: torch.nn.Module) -> optollama.model.OpenLayerFlow:
    """Return the concrete model behind an optional DDP wrapper."""
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return cast(optollama.model.OpenLayerFlow, model.module)
    return cast(optollama.model.OpenLayerFlow, model)


def compute_training_loss(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Compute the joint objective through the DDP wrapper when present."""
    core = unwrap_model(model)
    clean_materials = batch["material_targets"].to(dtype=torch.long)
    clean_thickness = batch["thickness_targets"].to(dtype=torch.float32)
    layer_mask = batch["layer_mask"].to(dtype=torch.bool)
    sample_mask = batch.get(
        "sample_mask",
        torch.ones(layer_mask.shape[0], device=layer_mask.device, dtype=torch.bool),
    ).to(dtype=torch.bool)
    supervised_layers = layer_mask & sample_mask[:, None]
    batch_size = clean_materials.shape[0]
    timesteps = torch.rand(batch_size, device=clean_materials.device).clamp_(1.0e-4, 1.0 - 1.0e-4)
    corrupted = (torch.rand(clean_materials.shape, device=clean_materials.device) < timesteps[:, None]) & supervised_layers
    for row in range(batch_size):
        if bool(sample_mask[row]) and not bool(corrupted[row].any()):
            active = torch.nonzero(supervised_layers[row], as_tuple=False).flatten()
            if active.numel():
                choice = active[torch.randint(active.numel(), (1,), device=active.device)]
                corrupted[row, choice] = True
    noised_materials = clean_materials.clone()
    noised_materials[corrupted | ~layer_mask] = core.MASK_MATERIAL
    thickness_noise = torch.randn_like(clean_thickness)
    thickness_state = (1.0 - timesteps[:, None]) * clean_thickness + timesteps[:, None] * thickness_noise
    target_velocity = thickness_noise - clean_thickness
    outputs = model(
        wavelengths_nm=batch["wavelengths_nm"],
        target_spectrum=batch["target_spectrum"],
        query_mask=batch["query_mask"],
        candidate_nk=batch["candidate_nk"],
        candidate_mask=batch["candidate_mask"],
        material_ids=noised_materials,
        thickness_state=thickness_state,
        layer_mask=layer_mask,
        timesteps=timesteps,
    )
    if bool(corrupted.any()):
        material_loss = functional.cross_entropy(outputs["material_logits"][corrupted], clean_materials[corrupted])
        accuracy = (outputs["material_logits"][corrupted].argmax(-1) == clean_materials[corrupted]).float().mean()
    else:
        finite_logits = torch.where(
            torch.isfinite(outputs["material_logits"]),
            outputs["material_logits"],
            torch.zeros_like(outputs["material_logits"]),
        )
        material_loss = finite_logits.sum() * 0.0
        accuracy = torch.zeros((), device=clean_materials.device)
    if bool(supervised_layers.any()):
        thickness_loss = functional.smooth_l1_loss(
            outputs["thickness_velocity"][supervised_layers],
            target_velocity[supervised_layers],
            beta=core.config.thickness_huber_delta,
        )
    else:
        thickness_loss = outputs["thickness_velocity"].sum() * 0.0
    total = material_loss + core.config.thickness_loss_weight * thickness_loss
    return {
        "loss": total,
        "material_loss": material_loss.detach(),
        "thickness_loss": thickness_loss.detach(),
        "material_accuracy": accuracy,
        "mean_timestep": timesteps.mean().detach(),
        "corrupted_fraction": corrupted.sum().float() / supervised_layers.sum().clamp_min(1),
        "supervised_samples": sample_mask.sum().detach(),
    }


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
) -> dict[str, float]:
    """Train or validate the denoising objectives for one epoch."""
    train = optimizer is not None
    model.train(train)
    totals = torch.zeros(7, dtype=torch.float64, device=device)
    progress = tqdm.tqdm(
        loader,
        desc=f"Epoch {epoch + 1}/{epochs} open-layer {'train' if train else 'val'}",
        disable=(torch.distributed.is_initialized() and torch.distributed.get_rank() != 0),
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
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"Non-finite open-layer loss at epoch={epoch + 1}, step={step}.")
        if train:
            assert optimizer is not None
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
        else:
            norm = torch.zeros((), device=device)

        batch_size = int(outputs["supervised_samples"].item())
        totals += torch.tensor(
            [
                float(loss.detach().item()) * batch_size,
                float(outputs["material_loss"].item()) * batch_size,
                float(outputs["thickness_loss"].item()) * batch_size,
                float(outputs["material_accuracy"].item()) * batch_size,
                float(outputs["mean_timestep"].item()) * batch_size,
                float(outputs["corrupted_fraction"].item()) * batch_size,
                float(batch_size),
            ],
            dtype=torch.float64,
            device=device,
        )
        progress.set_postfix(
            loss=f"{float(loss.detach().item()):.4f}",
            mat=f"{100.0 * float(outputs['material_accuracy'].item()):.1f}%",
            grad=f"{float(norm):.2f}",
        )

    totals = reduce_totals(totals)
    samples = float(totals[-1].item())
    if samples <= 0:
        raise RuntimeError("No supervised samples remained after applying material holdouts.")
    keys = ("loss", "material_loss", "thickness_loss", "material_accuracy", "mean_timestep", "corrupted_fraction")
    return {key: float(totals[idx].item() / samples) for idx, key in enumerate(keys)} | {"samples": int(samples)}


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
) -> dict[str, float]:
    """Measure best-of-MC exact-TMM R/T MAE using each target's true layer count."""
    core = unwrap_model(model)
    core.eval()
    mae_values: list[torch.Tensor] = []
    processed = 0
    for raw_batch in loader:
        if processed >= max_samples:
            break
        keep = min(int(raw_batch["target_spectrum"].shape[0]), max_samples - processed)
        raw_batch = {key: value[:keep] for key, value in raw_batch.items()}
        batch = move_batch(raw_batch, device)
        layer_counts = batch["layer_mask"].sum(dim=1)
        batch_mae: list[torch.Tensor] = []
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
            target_full = raw_batch["target_spectrum"].permute(0, 2, 1).to(predicted.device)
            batch_mae.append((predicted[:, (0, 2)] - target_full).abs().mean(dim=(1, 2)).cpu())
        mae_values.append(torch.stack(batch_mae, dim=1).min(dim=1).values)
        processed += keep
    local = torch.cat(mae_values) if mae_values else torch.empty(0)
    if optollama.utils.is_ddp():
        gathered: list[Any] = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, local.tolist())
        local = torch.tensor([value for values in gathered for value in values], dtype=torch.float32)
    if local.numel() == 0:
        return {"rt_mae_mean": math.nan, "rt_mae_median": math.nan, "samples": 0}
    return {
        "rt_mae_mean": float(local.mean().item()),
        "rt_mae_median": float(local.median().item()),
        "samples": int(local.numel()),
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
            tmm_note = "" if tmm_metrics is None else f", tmm_rt_mae={tmm_metrics['rt_mae_mean']:.6f}"
            print(f"Open-layer epoch {epoch + 1}: val_loss={val_metrics['loss']:.6f}{tmm_note}, best={best_loss:.6f}")

    if optollama.utils.is_ddp():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
