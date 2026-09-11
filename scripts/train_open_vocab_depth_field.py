from __future__ import annotations

import argparse
import math
import os
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

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


def unwrap_model(model: torch.nn.Module) -> optollama.model.OpenVocabularyDepthFieldDiffusion:
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
) -> optollama.data.OpenVocabularyDepthFieldCollator:
    """Build a deterministic full-grid validation or randomized training collator."""
    bank = nested(block, "MATERIAL_BANK", default={}) or {}
    grid = nested(block, "GRID", default={}) or {}
    optical = nested(block, "OPTICAL_CONDITION", default={}) or {}
    wavelengths = cfg["WAVELENGTHS"]
    return optollama.data.OpenVocabularyDepthFieldCollator(
        wavelengths_nm=wavelengths,
        catalog=catalog,
        idx_to_token=idx_to_token,
        eos_idx=eos_idx,
        pad_idx=pad_idx,
        msk_idx=msk_idx,
        channels=("R", "A", "T"),
        max_layers=int(block.get("MAX_LAYERS", 100)),
        max_candidates=int(bank.get("MAX_CANDIDATES", 24)),
        min_query_points=len(wavelengths),
        max_query_points=len(wavelengths),
        query_sampling="full",
        randomize_candidates=bool(bank.get("RANDOMIZE_ORDER", True) if train else False),
        random_distractors=bool(bank.get("RANDOM_DISTRACTORS", True)),
        holdout_materials=bank.get("HOLDOUT_MATERIALS", []) if train else (),
        merge_adjacent=True,
        coverage_tolerance_nm=float(bank.get("MATERIAL_COVERAGE_TOLERANCE_NM", 100.0)),
        seed=seed,
        dz_nm=float(grid.get("DZ_NM", 5.0)),
        max_total_nm=float(grid.get("MAX_THICKNESS_NM", 10_000.0)),
        incidence_angle_deg=float(optical.get("ANGLE_DEG", 0.0)),
        polarization=str(optical.get("POLARIZATION", "s")),
    )


def select_valid_rows(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor] | None:
    """Drop empty or held-out stacks before computing a supervised loss."""
    valid = batch["sample_mask"].to(dtype=torch.bool)
    if not bool(valid.any()):
        return None
    return {key: value[valid] for key, value in batch.items()}


def compute_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    train_cfg: dict[str, Any],
    corruption: optollama.model.DepthFieldCorruptionConfig,
) -> dict[str, torch.Tensor]:
    """Compute continuous-time denoising loss through the DDP wrapper."""
    core = unwrap_model(model)
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
    return {
        "loss": loss,
        "accuracy": (prediction == clean).float().mean(),
        "corrupted_accuracy": (prediction[corrupted] == clean[corrupted]).float().mean()
        if bool(corrupted.any())
        else loss.new_zeros(()),
        "noise_probability": core.noise_probability(timesteps).mean(),
        "corrupted_fraction": corrupted.float().mean(),
        "condition_dropped": dropped.float().mean(),
    }


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
) -> dict[str, float]:
    """Train or validate one epoch and aggregate metrics over all ranks."""
    training = optimizer is not None
    model.train(training)
    keys = ("loss", "accuracy", "corrupted_accuracy", "noise_probability", "corrupted_fraction", "condition_dropped")
    totals = torch.zeros(len(keys) + 1, dtype=torch.float64, device=device)
    local_rows = 0
    optimizer_steps = 0
    log_every_steps = max(1, int(log_every_steps))
    ema_update_every_steps = max(1, int(ema_update_every_steps))
    eval_every_steps = None if not eval_every_steps else max(1, int(eval_every_steps))
    world = torch.distributed.get_world_size() if optollama.utils.is_ddp() else 1
    show = not optollama.utils.is_ddp() or torch.distributed.get_rank() == 0
    progress = tqdm.tqdm(
        loader,
        desc=f"Epoch {epoch + 1}/{epochs} open-vocab depth {'train' if training else 'val'}",
        disable=not show,
        miniters=log_every_steps,
        mininterval=5.0,
    )
    for step, raw in enumerate(progress):
        if max_steps is not None and step >= max_steps:
            break
        selected = select_valid_rows(raw)
        if selected is None:
            continue
        batch = {key: value.to(device, non_blocking=True) for key, value in selected.items()}
        rows = int(batch["clean_fields"].shape[0])
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
            scaler.scale(output["loss"]).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(train_cfg.get("GRAD_CLIP", 1.0)), error_if_nonfinite=False
            )
            if not synchronized_finite(torch.stack((output["loss"].detach().float(), norm.detach().float()))):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(f"Non-finite loss or gradient at epoch={epoch + 1}, step={step}.")
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
        display_step = optimizer_steps if training else step + 1
        if display_step == 1 or display_step % log_every_steps == 0:
            running = {key: float((totals[index] / totals[-1]).item()) for index, key in enumerate(keys)}
            progress.set_postfix(
                loss=f"{running['loss']:.4f}",
                acc=f"{100 * running['accuracy']:.1f}%",
                corr=f"{100 * running['corrupted_accuracy']:.1f}%",
                refresh=False,
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
            on_evaluation_step(
                current_global_step,
                global_samples_seen + local_rows * world,
                snapshot,
            )
            model.train(True)
    if optollama.utils.is_ddp():
        torch.distributed.all_reduce(totals)
    if totals[-1] <= 0:
        raise RuntimeError("No valid samples were processed.")
    metrics = {key: float(totals[index] / totals[-1]) for index, key in enumerate(keys)}
    metrics["samples"] = int(totals[-1].item())
    metrics["samples_seen"] = local_rows * world if training else int(totals[-1].item())
    metrics["global_samples_seen"] = global_samples_seen + int(metrics["samples_seen"]) if training else global_samples_seen
    metrics["optimizer_steps"] = optimizer_steps
    metrics["global_optimizer_steps"] = global_optimizer_steps + optimizer_steps
    metrics["learning_rate"] = float(optimizer.param_groups[0]["lr"]) if optimizer is not None else 0.0
    return metrics


def main() -> None:
    """Train and checkpoint an open-vocabulary depth-field model."""
    args = parse_args()
    cfg = optollama.utils.load_config_file(args.config)
    cfg["WAVELENGTHS"] = torch.arange(
        int(cfg["WAVELENGTH_MIN"]), int(cfg["WAVELENGTH_MAX"]) + 1, int(cfg["WAVELENGTH_STEPS"]), dtype=torch.float32
    )
    block = cfg.get("OPEN_VOCAB_DEPTH_FIELD") or {}
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
        torch.set_float32_matmul_precision("high")
    tokens, token_to_idx, idx_to_token, _, _, _, eos_idx, pad_idx, msk_idx = optollama.data.init_tokens(cfg["TOKENS_PATH"])
    material_names = optollama.data.material_names_from_tokens(tokens)
    catalog = optollama.data.load_material_catalog(cfg["MATERIALS_PATH"], material_names)
    max_candidates = int(nested(block, "MATERIAL_BANK", "MAX_CANDIDATES", default=24))
    if len(catalog.names) > max_candidates and not bool(nested(block, "MATERIAL_BANK", "RANDOM_DISTRACTORS", default=True)):
        raise ValueError("The complete catalog exceeds MAX_CANDIDATES.")
    train_collator = make_collator(
        cfg, block, catalog, idx_to_token, eos_idx, pad_idx, msk_idx, train=True, seed=int(cfg.get("SEED", 0)) + rank * 17
    )
    val_collator = make_collator(
        cfg, block, catalog, idx_to_token, eos_idx, pad_idx, msk_idx, train=False, seed=int(cfg.get("SEED", 0)) + 10_000 + rank * 17
    )
    train_n = int(args.max_train_samples or (world * 2 if args.smoke_test else cfg["NUM_SAMPLES_TRAIN"]))
    val_n = int(args.max_val_samples or (world if args.smoke_test else cfg["NUM_SAMPLES_TEST"]))
    train_dataset, train_loader = make_loader(
        cfg, split="train", collator=train_collator, subset_n=train_n, rank=rank, world_size=world
    )
    _, val_loader = make_loader(cfg, split="test", collator=val_collator, subset_n=val_n, rank=rank, world_size=world)

    model_config = model_config_from_mapping(block, (3, len(cfg["WAVELENGTHS"])))
    if args.smoke_test:
        model_config = replace(
            model_config,
            d_model=32,
            n_blocks=2,
            n_heads=4,
            ffn_multiplier=2.0,
            kernel_size=3,
            hybrid_dilations=(1, 2),
            spectrum_encoder_blocks=1,
            spectrum_encoder_heads=4,
        )
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
        history = list(checkpoint_extra.get("history") or [])
        step_evaluations = list(checkpoint_extra.get("step_evaluations") or [])
        if ema is not None and isinstance(checkpoint_extra.get("ema"), dict):
            ema.load_state_dict(checkpoint_extra["ema"], raw_model)

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
    global_seen = resume_global_samples(history, start_epoch, train_n)
    global_steps = int(
        checkpoint_extra.get("global_optimizer_steps", 0)
        or global_seen // max(1, int(cfg["TRAIN_BATCH_SIZE"]) * world)
    )
    log_every_steps = max(1, int(train_cfg.get("LOG_EVERY_STEPS", 50)))
    eval_every_steps = max(0, int(train_cfg.get("EVAL_EVERY_STEPS", 0)))
    ema_update_every_steps = max(1, int(ema_cfg.get("UPDATE_EVERY_STEPS", 1)))
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"Open-vocab depth model: parameters={sum(p.numel() for p in model.parameters()):,}, world={world}, batch/rank={cfg['TRAIN_BATCH_SIZE']}"
        )
        print(
            f"Continuous time: t~Uniform(0,1), noise_probability=t^2; validation sampling steps={nested(block, 'EVAL', 'SAMPLING_STEPS', default=64)}"
        )
        print(
            f"Optical condition: angle={nested(block, 'OPTICAL_CONDITION', 'ANGLE_DEG', default=0.0)}deg, polarization={nested(block, 'OPTICAL_CONDITION', 'POLARIZATION', default='s')}"
        )
        print(
            "Optimizations: "
            f"compile={compile_enabled}, fused_adamw={fused_optimizer}, "
            f"log_every={log_every_steps}, eval_every={eval_every_steps or 'epoch'}, "
            f"ema_every={ema_update_every_steps}"
        )
    best = min((float(item["val"]["loss"]) for item in history), default=math.inf)
    for epoch in range(start_epoch, epochs):
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        max_train_steps = 1 if args.smoke_test else args.max_train_steps

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
                    str(output_dir / "open-vocab-depth-step-evaluations.json"), step_evaluations
                )
                print(
                    f"Open-vocab depth step {step:,}: val_loss={interval_val['loss']:.6f}, "
                    f"val_acc={100 * interval_val['accuracy']:.2f}%"
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
        )
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
            extra = {
                "open_vocab_depth_field_config": model_config.to_dict(),
                "material_names": list(catalog.names),
                "config_path": args.config,
                "history": history,
                "step_evaluations": step_evaluations,
                "global_optimizer_steps": global_steps,
                "continuous_time": True,
                "ema": ema.state_dict() if ema is not None else None,
            }
            optollama.utils.save_checkpoint(
                str(output_dir / "open-vocab-depth-last.pt"),
                model=raw_model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                extra=extra,
            )
            if val["loss"] < best:
                best = val["loss"]
                optollama.utils.save_checkpoint(
                    str(output_dir / "open-vocab-depth-best.pt"),
                    model=raw_model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    extra=extra,
                )
            optollama.utils.save_as_json(str(output_dir / "open-vocab-depth-history.json"), history)
            print(
                f"Open-vocab depth epoch {epoch + 1}: val_loss={val['loss']:.6f}, val_acc={100 * val['accuracy']:.2f}%, best={best:.6f}"
            )
    if optollama.utils.is_ddp():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
