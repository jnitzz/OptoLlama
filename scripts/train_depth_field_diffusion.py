#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import os
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import torch
import tqdm

import optollama.data
import optollama.evaluation.simulation
import optollama.model
import optollama.utils

# ruff: noqa: D103


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an experimental 10 nm depth-field diffusion model.")
    parser.add_argument("--config", type=str, default="configs/optollama.yaml", help="Project config YAML.")
    parser.add_argument("--out-dir", type=str, default=None, help="Checkpoint output dir.")
    parser.add_argument("--device", type=str, default=None, help='Device, e.g. "cuda", "cuda:0", or "cpu".')
    parser.add_argument("--seed", type=int, default=None, help="Random seed. Defaults to config SEED.")

    parser.add_argument("--dz-nm", type=float, default=None, help="Depth resolution in nm.")
    parser.add_argument("--max-thickness-nm", type=float, default=None, help="Maximum represented total thickness in nm.")
    parser.add_argument("--output-seq-len", type=int, default=None, help="Decoded token sequence length metadata.")

    parser.add_argument("--epochs", type=int, default=None, help="Epoch count. Defaults to config EPOCHS.")
    parser.add_argument("--batch-size", type=int, default=None, help="Train batch size. Defaults to config TRAIN_BATCH_SIZE.")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Validation batch size. Defaults to train batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers. Defaults to config NUM_WORKERS.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional train subset size.")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Optional validation subset size.")
    parser.add_argument("--no-val", action="store_true", help="Skip validation.")
    parser.add_argument(
        "--validate-every-n-train-samples",
        type=int,
        default=None,
        help="Override VALIDATE_EVERY_N_TRAIN_SAMPLES. 0 disables mid-epoch validation.",
    )
    parser.add_argument(
        "--validate-at-epoch-end",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run validation at epoch end. Defaults to VALIDATION.AT_EPOCH_END.",
    )
    parser.add_argument("--sharded-loading", action="store_true", help="Force sharded streaming dataset loading.")
    parser.add_argument("--eager-loading", action="store_true", help="Force eager in-memory dataset loading.")
    parser.add_argument(
        "--keep-overlimit-stacks",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep stacks thicker than --max-thickness-nm by clipping them. Default skips them.",
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=[
            "conv",
            "convolution",
            "cnn",
            "dilated_conv",
            "dilated-conv",
            "attention",
            "attn",
            "mha",
            "multihead_attention",
            "multi-head-attention",
            "multi_head_attention",
            "transformer",
            "optollama_depth",
            "optollama-depth",
            "optollama_depth_field",
            "optollama-depth-field",
            "opto_depth",
            "opto-depth",
            "dit_depth",
            "dit-depth",
        ],
        help=(
            "Depth-field backbone type. 'conv' uses dilated Conv1d blocks, 'attention' uses global self-attention, "
            "and 'optollama_depth' uses OptoLlama-style cross/self-attention blocks."
        ),
    )
    parser.add_argument("--d-model", type=int, default=None, help="Depth-field model channel width.")
    parser.add_argument("--n-blocks", type=int, default=None, help="Number of denoising blocks.")
    parser.add_argument("--kernel-size", type=int, default=None, help="Conv1d kernel size.")
    parser.add_argument(
        "--conv-type",
        type=str,
        default=None,
        choices=["full", "standard", "conv", "separable", "depthwise", "depthwise_separable", "depthwise-separable"],
        help="Depth-field residual convolution type. 'separable' uses depthwise + pointwise Conv1d.",
    )
    parser.add_argument("--n-heads", type=int, default=None, help="Attention heads when --model-type=attention.")
    parser.add_argument("--ffn-multiplier", type=float, default=None, help="Attention feed-forward width multiplier.")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout inside residual blocks.")
    parser.add_argument("--diffusion-steps", type=int, default=None, help="Discrete depth-field diffusion timesteps.")

    parser.add_argument("--learning-rate", type=float, default=None, help="Optimizer LR. Defaults to config LEARNING_RATE.")
    parser.add_argument("--weight-decay", type=float, default=None, help="AdamW weight decay.")
    parser.add_argument("--grad-clip", type=float, default=None, help="Gradient norm clip. <=0 disables clipping.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None, help="Use CUDA autocast/GradScaler.")

    parser.add_argument("--void-loss-weight", type=float, default=None, help="CE class weight for the void depth class.")
    parser.add_argument(
        "--random-replace-prob",
        type=float,
        default=None,
        help="Fraction of corrupted bins that are random material replacements rather than masks.",
    )
    parser.add_argument(
        "--loss-on-corrupted-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compute CE only on bins that were masked/replaced. Default supervises every bin.",
    )
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint to resume.")
    parser.add_argument("--save-every", type=int, default=1, help="Save the last checkpoint every N epochs.")
    parser.add_argument(
        "--eval-mode",
        type=str,
        default=None,
        choices=["tmm", "denoise", "both"],
        help="Validation mode. 'tmm' samples all-mask fields and scores TMM MAE; 'denoise' uses CE on corrupted targets.",
    )
    parser.add_argument("--eval-mc-samples", type=int, default=None, help="TMM validation candidates per target.")
    parser.add_argument("--eval-sampling-steps", type=int, default=None, help="TMM validation denoising steps.")
    parser.add_argument("--eval-temperature", type=float, default=None, help="TMM validation sampling temperature.")
    parser.add_argument("--eval-top-k", type=int, default=None, help="TMM validation top-k material sampling filter.")
    parser.add_argument("--eval-deterministic", action=argparse.BooleanOptionalAction, default=None, help="Use argmax sampling for TMM validation.")
    parser.add_argument(
        "--eval-remask-strategy",
        type=str,
        default=None,
        choices=["confidence", "random"],
        help="TMM validation remasking strategy: confidence reopens least-confident bins, random uses Bernoulli remasking.",
    )
    parser.add_argument("--eval-tmm-device", type=str, default=None, help='TMM validation device. "auto" uses model device.')
    parser.add_argument("--eval-tmm-batch-size", type=int, default=None, help="Decoded stacks per TMM validation chunk.")
    parser.add_argument(
        "--save-eval-samples",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save TMM validation sample JSONs. Required unless DEPTH_FIELD.EVAL.SAVE_SAMPLES is set.",
    )
    parser.add_argument(
        "--eval-samples-dir",
        type=str,
        default=None,
        help="Directory for TMM validation sample JSONs. Required when validation sample saving is enabled.",
    )
    parser.add_argument(
        "--eval-record-spectra",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Store target/predicted spectra in validation sample JSONs. Required unless DEPTH_FIELD.EVAL.RECORD_SPECTRA is set.",
    )
    parser.add_argument(
        "--eval-record-all-mc",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Store every MC validation candidate, not only the selected best. Required unless DEPTH_FIELD.EVAL.RECORD_ALL_MC is set.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        device = torch.device(device_arg)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


def resolve_tmm_device(device_arg: str | None, default_device: torch.device) -> torch.device:
    if device_arg is None or str(device_arg).lower() in {"auto", "same"}:
        return default_device
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA TMM validation was requested, but torch.cuda.is_available() is false.")
    return device


def set_loader_epoch(loader: torch.utils.data.DataLoader, epoch: int) -> None:
    dataset = getattr(loader, "dataset", None)
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)
    sampler = getattr(loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def example_spectrum(dataset: torch.utils.data.Dataset) -> torch.Tensor:
    if isinstance(dataset, torch.utils.data.Subset):
        base = dataset.dataset
        first_idx = int(dataset.indices[0])
        if hasattr(base, "spectra"):
            return base.spectra[first_idx]
        return base[first_idx][0]
    if hasattr(dataset, "example_spectrum"):
        return dataset.example_spectrum()
    if hasattr(dataset, "spectra"):
        return dataset.spectra[0]
    return next(iter(dataset))[0]


def apply_loader_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    if args.batch_size is not None:
        cfg["TRAIN_BATCH_SIZE"] = int(args.batch_size)
    if args.eval_batch_size is not None:
        cfg["TEST_BATCH_SIZE"] = int(args.eval_batch_size)
    elif args.batch_size is not None:
        cfg["TEST_BATCH_SIZE"] = int(args.batch_size)
    if args.num_workers is not None:
        cfg["NUM_WORKERS"] = int(args.num_workers)
    if args.sharded_loading and args.eager_loading:
        raise ValueError("Use only one of --sharded-loading or --eager-loading.")
    if args.sharded_loading:
        cfg["SHARDED_LOADING"] = True
    if args.eager_loading:
        cfg["SHARDED_LOADING"] = False


def depth_field_block(cfg: dict[str, Any]) -> dict[str, Any]:
    block = cfg.get("DEPTH_FIELD") or {}
    return block if isinstance(block, dict) else {}


def nested_get(block: dict[str, Any], *path: str, default: Any = None) -> Any:
    value: Any = block
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


MISSING = object()


def nested_required(block: dict[str, Any], *path: str) -> Any:
    value: Any = block
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return MISSING
        value = value[key]
    return value


def set_arg_config_required(args: argparse.Namespace, name: str, value: Any, source: str) -> None:
    if getattr(args, name) is not None:
        return
    if value is MISSING or value is None:
        flag = f"--{name.replace('_', '-')}"
        raise ValueError(f"Missing required setting {source}. Set it in the config or pass {flag}.")
    setattr(args, name, value)


def cfg_required(cfg: dict[str, Any], key: str, flag: str | None = None) -> Any:
    if key in cfg and cfg[key] is not None:
        return cfg[key]
    suffix = f" or pass {flag}" if flag else ""
    raise ValueError(f"Missing required setting {key}. Set it in the config{suffix}.")


def set_arg_default(args: argparse.Namespace, name: str, value: Any) -> None:
    if getattr(args, name) is None and value is not None:
        setattr(args, name, value)


def resume_value_to_path(value: Any, default_path: Path) -> tuple[Path | None, bool]:
    if value is None:
        return None, False
    if isinstance(value, bool):
        return (default_path, False) if value else (None, False)
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"0", "false", "no", "none", "null", "off"}:
            return None, False
        if text.lower() in {"1", "true", "yes", "on"}:
            return default_path, False
        return Path(text), True
    if isinstance(value, (int, float)):
        if value == 0:
            return None, False
        if value == 1:
            return default_path, False
    if isinstance(value, os.PathLike):
        return Path(value), True
    raise TypeError(f"Unsupported depth-field resume value {value!r} ({type(value).__name__}).")


def resolve_resume_checkpoint(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    default_path: Path,
) -> tuple[Path | None, str | None, bool]:
    if args.resume:
        return Path(args.resume), "--resume", True

    block = depth_field_block(cfg)
    candidates = (
        ("DEPTH_FIELD.RESUME_PATH", nested_get(block, "RESUME_PATH")),
        ("DEPTH_FIELD.RESUME", nested_get(block, "RESUME")),
        ("CHECKPOINT.RESUME", cfg.get("RESUME_CHECKPOINT")),
    )
    for source, value in candidates:
        path, required = resume_value_to_path(value, default_path)
        if path is not None:
            return path, source, required
    return None, None, False


def apply_depth_field_defaults(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    block = depth_field_block(cfg)
    set_arg_config_required(args, "out_dir", block.get("OUT_DIR", MISSING), "DEPTH_FIELD.OUT_DIR")
    set_arg_config_required(args, "dz_nm", block.get("DZ_NM", MISSING), "DEPTH_FIELD.DZ_NM")
    set_arg_config_required(args, "max_thickness_nm", block.get("MAX_THICKNESS_NM", MISSING), "DEPTH_FIELD.MAX_THICKNESS_NM")
    set_arg_config_required(args, "output_seq_len", block.get("OUTPUT_SEQ_LEN", MISSING), "DEPTH_FIELD.OUTPUT_SEQ_LEN")

    if args.model_type is None:
        args.model_type = nested_get(block, "MODEL", "TYPE", default=nested_get(block, "MODEL", "MODEL_TYPE", default="conv"))
    set_arg_config_required(args, "d_model", nested_required(block, "MODEL", "D_MODEL"), "DEPTH_FIELD.MODEL.D_MODEL")
    set_arg_config_required(args, "n_blocks", nested_required(block, "MODEL", "N_BLOCKS"), "DEPTH_FIELD.MODEL.N_BLOCKS")
    if args.kernel_size is None:
        args.kernel_size = nested_get(block, "MODEL", "KERNEL_SIZE", default=7)
    if args.conv_type is None:
        args.conv_type = nested_get(block, "MODEL", "CONV_TYPE", default="full")
    if args.n_heads is None:
        args.n_heads = nested_get(block, "MODEL", "N_HEADS", default=8)
    if args.ffn_multiplier is None:
        args.ffn_multiplier = nested_get(block, "MODEL", "FFN_MULTIPLIER", default=4.0)
    set_arg_config_required(args, "dropout", nested_required(block, "MODEL", "DROPOUT"), "DEPTH_FIELD.MODEL.DROPOUT")
    set_arg_config_required(args, "diffusion_steps", nested_required(block, "MODEL", "DIFFUSION_STEPS"), "DEPTH_FIELD.MODEL.DIFFUSION_STEPS")

    set_arg_config_required(args, "weight_decay", nested_required(block, "TRAIN", "WEIGHT_DECAY"), "DEPTH_FIELD.TRAIN.WEIGHT_DECAY")
    set_arg_config_required(args, "grad_clip", nested_required(block, "TRAIN", "GRAD_CLIP"), "DEPTH_FIELD.TRAIN.GRAD_CLIP")
    set_arg_config_required(args, "amp", nested_required(block, "TRAIN", "AMP"), "DEPTH_FIELD.TRAIN.AMP")
    set_arg_config_required(
        args,
        "keep_overlimit_stacks",
        nested_required(block, "TRAIN", "KEEP_OVERLIMIT_STACKS"),
        "DEPTH_FIELD.TRAIN.KEEP_OVERLIMIT_STACKS",
    )
    set_arg_config_required(args, "void_loss_weight", nested_required(block, "TRAIN", "VOID_LOSS_WEIGHT"), "DEPTH_FIELD.TRAIN.VOID_LOSS_WEIGHT")
    set_arg_config_required(
        args,
        "random_replace_prob",
        nested_required(block, "TRAIN", "RANDOM_REPLACE_PROB"),
        "DEPTH_FIELD.TRAIN.RANDOM_REPLACE_PROB",
    )
    set_arg_config_required(
        args,
        "loss_on_corrupted_only",
        nested_required(block, "TRAIN", "LOSS_ON_CORRUPTED_ONLY"),
        "DEPTH_FIELD.TRAIN.LOSS_ON_CORRUPTED_ONLY",
    )

    set_arg_config_required(args, "eval_mode", nested_required(block, "EVAL", "MODE"), "DEPTH_FIELD.EVAL.MODE")
    set_arg_config_required(args, "eval_mc_samples", nested_required(block, "EVAL", "MC_SAMPLES"), "DEPTH_FIELD.EVAL.MC_SAMPLES")
    set_arg_config_required(args, "eval_sampling_steps", nested_required(block, "EVAL", "SAMPLING_STEPS"), "DEPTH_FIELD.EVAL.SAMPLING_STEPS")
    set_arg_config_required(args, "eval_temperature", nested_required(block, "EVAL", "TEMPERATURE"), "DEPTH_FIELD.EVAL.TEMPERATURE")
    set_arg_config_required(args, "eval_top_k", nested_required(block, "EVAL", "TOP_K"), "DEPTH_FIELD.EVAL.TOP_K")
    set_arg_config_required(args, "eval_deterministic", nested_required(block, "EVAL", "DETERMINISTIC"), "DEPTH_FIELD.EVAL.DETERMINISTIC")
    set_arg_config_required(args, "eval_remask_strategy", nested_required(block, "EVAL", "REMASK_STRATEGY"), "DEPTH_FIELD.EVAL.REMASK_STRATEGY")
    set_arg_config_required(args, "eval_tmm_device", nested_required(block, "EVAL", "TMM_DEVICE"), "DEPTH_FIELD.EVAL.TMM_DEVICE")
    set_arg_config_required(args, "eval_tmm_batch_size", nested_required(block, "EVAL", "TMM_BATCH_SIZE"), "DEPTH_FIELD.EVAL.TMM_BATCH_SIZE")
    set_arg_config_required(args, "save_eval_samples", nested_required(block, "EVAL", "SAVE_SAMPLES"), "DEPTH_FIELD.EVAL.SAVE_SAMPLES")
    set_arg_default(args, "eval_samples_dir", nested_get(block, "EVAL", "SAMPLES_DIR"))
    if bool(args.save_eval_samples) and args.eval_samples_dir is None:
        raise ValueError("Missing required setting DEPTH_FIELD.EVAL.SAMPLES_DIR. Set it in the config or pass --eval-samples-dir.")
    set_arg_config_required(args, "eval_record_spectra", nested_required(block, "EVAL", "RECORD_SPECTRA"), "DEPTH_FIELD.EVAL.RECORD_SPECTRA")
    set_arg_config_required(args, "eval_record_all_mc", nested_required(block, "EVAL", "RECORD_ALL_MC"), "DEPTH_FIELD.EVAL.RECORD_ALL_MC")


def autocast_context(enabled: bool):
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def ddp_active() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1


def ddp_rank() -> int:
    return torch.distributed.get_rank() if ddp_active() else 0


def ddp_world_size() -> int:
    return torch.distributed.get_world_size() if ddp_active() else 1


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def schedule_enabled(schedule: dict[str, Any] | None) -> bool:
    if not isinstance(schedule, dict) or not schedule:
        return False
    if "ENABLED" in schedule and not bool(schedule.get("ENABLED")):
        return False
    schedule_type = str(schedule.get("TYPE", "none")).lower().replace("-", "_")
    return schedule_type not in {"", "none", "off", "false", "disabled"}


def lr_schedule_config(cfg: dict[str, Any]) -> dict[str, Any]:
    block = depth_field_block(cfg)
    train_cfg = nested_get(block, "TRAIN", default={}) or {}
    schedule = train_cfg.get("LR_SCHEDULE") if isinstance(train_cfg, dict) else None
    return schedule if isinstance(schedule, dict) else {}


def timestep_schedule_config(cfg: dict[str, Any]) -> dict[str, Any]:
    block = depth_field_block(cfg)
    train_cfg = nested_get(block, "TRAIN", default={}) or {}
    schedule = train_cfg.get("TIMESTEP_SCHEDULE") if isinstance(train_cfg, dict) else None
    return schedule if isinstance(schedule, dict) else {}


def ema_config(cfg: dict[str, Any]) -> dict[str, Any]:
    block = depth_field_block(cfg)
    train_cfg = nested_get(block, "TRAIN", default={}) or {}
    cfg_value = train_cfg.get("EMA") if isinstance(train_cfg, dict) else None
    return cfg_value if isinstance(cfg_value, dict) else {}


def scheduled_learning_rate(base_lr: float, schedule: dict[str, Any] | None, global_samples_seen: int) -> float:
    if not schedule_enabled(schedule):
        return float(base_lr)

    schedule_type = str(schedule.get("TYPE", "cosine")).lower().replace("-", "_")
    sample = max(0, int(global_samples_seen))
    max_lr = float(schedule.get("MAX_LR", schedule.get("LR", base_lr)))
    min_lr = float(schedule.get("MIN_LR", 0.0))
    start_lr = float(schedule.get("START_LR", min_lr))
    warmup_samples = max(0, int(schedule.get("WARMUP_SAMPLES", 0)))
    total_samples = max(warmup_samples + 1, int(schedule.get("TOTAL_SAMPLES", warmup_samples + 1)))

    if warmup_samples > 0 and sample < warmup_samples:
        progress = float(sample) / float(max(warmup_samples, 1))
        return start_lr + progress * (max_lr - start_lr)

    progress = float(sample - warmup_samples) / float(max(total_samples - warmup_samples, 1))
    progress = max(0.0, min(1.0, progress))
    if schedule_type == "cosine":
        return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    if schedule_type == "linear":
        return max_lr + progress * (min_lr - max_lr)
    if schedule_type in {"constant", "flat"}:
        return max_lr
    raise ValueError(f"Unknown DEPTH_FIELD.TRAIN.LR_SCHEDULE.TYPE={schedule_type!r}.")


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def timestep_min_fraction(schedule: dict[str, Any] | None, global_samples_seen: int) -> float:
    if not schedule_enabled(schedule):
        return 0.0
    schedule_type = str(schedule.get("TYPE", "high_noise_warmup")).lower().replace("-", "_")
    if schedule_type not in {"high_noise_warmup", "high_noise", "noise_warmup"}:
        raise ValueError(f"Unknown DEPTH_FIELD.TRAIN.TIMESTEP_SCHEDULE.TYPE={schedule_type!r}.")

    high_min = float(schedule.get("HIGH_NOISE_MIN_FRACTION", schedule.get("MIN_FRACTION", 0.6)))
    high_min = max(0.0, min(1.0, high_min))
    warmup_samples = max(0, int(schedule.get("HIGH_NOISE_SAMPLES", schedule.get("WARMUP_SAMPLES", 0))))
    if warmup_samples <= 0:
        return high_min
    progress = float(max(0, int(global_samples_seen))) / float(max(warmup_samples, 1))
    progress = max(0.0, min(1.0, progress))
    return high_min * (1.0 - progress)


def sample_depth_timesteps(
    *,
    timesteps: int,
    batch_size: int,
    device: torch.device,
    schedule: dict[str, Any] | None = None,
    global_samples_seen: int = 0,
) -> torch.Tensor:
    total = int(timesteps)
    if total <= 0:
        raise ValueError(f"timesteps must be positive, got {timesteps}")
    min_fraction = timestep_min_fraction(schedule, global_samples_seen)
    min_timestep = int(round(min_fraction * float(max(total - 1, 0))))
    min_timestep = max(0, min(min_timestep, total - 1))
    return torch.randint(min_timestep, total, (int(batch_size),), device=device)


class ModelEma:
    """Exponential moving average of model state dict tensors."""

    def __init__(self, model: torch.nn.Module, *, decay: float, device: torch.device | None = None) -> None:
        self.decay = float(decay)
        if not 0.0 < self.decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}.")
        self.device = device
        self.updates = 0
        self.shadow: dict[str, torch.Tensor] = {}
        self.reset(model)

    def _target_device(self, tensor: torch.Tensor) -> torch.device:
        return self.device if self.device is not None else tensor.device

    @torch.no_grad()
    def reset(self, model: torch.nn.Module) -> None:
        self.shadow = {}
        for name, tensor in unwrap_model(model).state_dict().items():
            if torch.is_floating_point(tensor):
                self.shadow[name] = tensor.detach().to(device=self._target_device(tensor), copy=True)
        self.updates = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        decay = float(self.decay)
        for name, tensor in unwrap_model(model).state_dict().items():
            if not torch.is_floating_point(tensor):
                continue
            value = tensor.detach()
            if name not in self.shadow:
                self.shadow[name] = value.to(device=self._target_device(value), copy=True)
                continue
            ema_value = self.shadow[name]
            ema_value.mul_(decay).add_(value.to(device=ema_value.device, dtype=ema_value.dtype), alpha=1.0 - decay)
        self.updates += 1

    def state_dict(self, *, cpu: bool = False) -> dict[str, Any]:
        device = torch.device("cpu") if cpu else None
        return {
            "decay": float(self.decay),
            "updates": int(self.updates),
            "shadow": {
                name: tensor.detach().to(device=device, copy=True) if device is not None else tensor.detach().clone()
                for name, tensor in self.shadow.items()
            },
        }

    @torch.no_grad()
    def load_state_dict(self, state: dict[str, Any], model: torch.nn.Module) -> None:
        shadow = state.get("shadow") if isinstance(state, dict) else None
        if not isinstance(shadow, dict):
            self.reset(model)
            return
        self.decay = float(state.get("decay", self.decay))
        self.updates = int(state.get("updates", 0))
        self.shadow = {}
        current = unwrap_model(model).state_dict()
        for name, tensor in shadow.items():
            if name in current and torch.is_floating_point(current[name]):
                self.shadow[name] = tensor.detach().to(device=self._target_device(current[name]), dtype=current[name].dtype, copy=True)

    def model_state_dict(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        current = unwrap_model(model).state_dict()
        state: dict[str, torch.Tensor] = {}
        for name, tensor in current.items():
            if name in self.shadow:
                state[name] = self.shadow[name].to(device=tensor.device, dtype=tensor.dtype)
            else:
                state[name] = tensor
        return state

    @contextmanager
    def apply_to(self, model: torch.nn.Module):
        core = unwrap_model(model)
        backup = {name: tensor.detach().clone() for name, tensor in core.state_dict().items()}
        core.load_state_dict(self.model_state_dict(model), strict=True)
        try:
            yield
        finally:
            core.load_state_dict(backup, strict=True)


def make_ema(cfg: dict[str, Any], model: torch.nn.Module) -> ModelEma | None:
    cfg_value = ema_config(cfg)
    if not bool(cfg_value.get("ENABLED", False)):
        return None
    device_value = str(cfg_value.get("DEVICE", "model")).lower()
    ema_device = torch.device("cpu") if device_value == "cpu" else None
    return ModelEma(model, decay=float(cfg_value.get("DECAY", 0.9999)), device=ema_device)


def all_reduce_sum(values: torch.Tensor) -> torch.Tensor:
    if ddp_active():
        torch.distributed.all_reduce(values)
    return values


def ddp_collective_device() -> torch.device:
    """Return the tensor device required by the active distributed backend."""
    backend = str(torch.distributed.get_backend()).lower()
    if "nccl" in backend:
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def gather_1d_tensor(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().cpu().reshape(-1)
    if not ddp_active():
        return values

    device = ddp_collective_device()
    local = values.to(device=device)
    local_count = torch.tensor([local.numel()], dtype=torch.long, device=device)
    counts = [torch.zeros_like(local_count) for _ in range(ddp_world_size())]
    torch.distributed.all_gather(counts, local_count)
    count_values = [int(count.item()) for count in counts]
    max_count = max(count_values, default=0)
    if max_count == 0:
        return torch.empty(0, dtype=values.dtype)

    padded = torch.zeros(max_count, dtype=local.dtype, device=device)
    if local.numel() > 0:
        padded[: local.numel()] = local
    gathered = [torch.empty_like(padded) for _ in range(ddp_world_size())]
    torch.distributed.all_gather(gathered, padded)

    parts = [tensor[:count].cpu() for tensor, count in zip(gathered, count_values) if count > 0]
    return torch.cat(parts, dim=0) if parts else torch.empty(0, dtype=values.dtype)



def depth_field_training_loss(
    model: torch.nn.Module,
    spectra: torch.Tensor,
    clean_fields: torch.Tensor,
    *,
    void_id: int,
    void_loss_weight: float = 0.25,
    random_replace_prob: float = 0.10,
    loss_on_corrupted_only: bool = False,
    timestep_schedule: dict[str, Any] | None = None,
    global_samples_seen: int = 0,
) -> dict[str, torch.Tensor]:
    core = unwrap_model(model)
    batch_size = int(clean_fields.size(0))
    timesteps = sample_depth_timesteps(
        timesteps=int(core.timesteps),
        batch_size=batch_size,
        device=clean_fields.device,
        schedule=timestep_schedule,
        global_samples_seen=global_samples_seen,
    )
    noised_fields, corrupted = core.corrupt(
        clean_fields,
        timesteps,
        random_replace_prob=random_replace_prob,
    )
    logits = model(spectra, noised_fields, timesteps)

    weights = torch.ones(int(core.num_materials), device=clean_fields.device, dtype=logits.dtype)
    if 0 <= int(void_id) < int(core.num_materials):
        weights[int(void_id)] = float(void_loss_weight)
    loss_per_bin = torch.nn.functional.cross_entropy(
        logits.reshape(-1, int(core.num_materials)),
        clean_fields.long().reshape(-1),
        weight=weights,
        reduction="none",
    ).view_as(clean_fields)

    if loss_on_corrupted_only:
        denom = corrupted.float().sum().clamp_min(1.0)
        loss = (loss_per_bin * corrupted.float()).sum() / denom
    else:
        loss = loss_per_bin.mean()

    return {
        "loss": loss,
        "logits": logits,
        "timesteps": timesteps,
        "noised_fields": noised_fields,
        "corrupted": corrupted,
    }


def reduced_epoch_metrics(
    *,
    counts: torch.Tensor,
    loss_sum: float,
    batches: int,
    active_nm_sum: float,
    full_count: int,
    samples: int,
    seen_samples: int,
    overlimit_seen: int,
    skipped_overlimit: int,
    dz_nm: float,
    device: torch.device,
) -> dict:
    reduced_counts = counts.detach().clone()
    totals = torch.tensor(
        [
            float(loss_sum),
            float(batches),
            float(active_nm_sum),
            float(full_count),
            float(samples),
            float(seen_samples),
            float(overlimit_seen),
            float(skipped_overlimit),
        ],
        dtype=torch.float64,
        device=device,
    )
    all_reduce_sum(reduced_counts)
    all_reduce_sum(totals)
    return counts_to_metrics(
        reduced_counts,
        loss_sum=float(totals[0].item()),
        batches=int(totals[1].item()),
        active_nm_sum=float(totals[2].item()),
        full_count=int(totals[3].item()),
        samples=int(totals[4].item()),
        seen_samples=int(totals[5].item()),
        overlimit_seen=int(totals[6].item()),
        skipped_overlimit=int(totals[7].item()),
        dz_nm=dz_nm,
    )


@torch.no_grad()
def accuracy_counts(logits: torch.Tensor, fields: torch.Tensor, void_id: int) -> torch.Tensor:
    pred = logits.argmax(dim=-1)
    correct = pred == fields
    non_void = fields != int(void_id)
    void = fields == int(void_id)
    return torch.tensor(
        [
            float(correct.sum().item()),
            float(fields.numel()),
            float((correct & non_void).sum().item()),
            float(non_void.sum().item()),
            float((correct & void).sum().item()),
            float(void.sum().item()),
        ],
        dtype=torch.float64,
        device=fields.device,
    )


def counts_to_metrics(
    counts: torch.Tensor,
    *,
    loss_sum: float,
    batches: int,
    active_nm_sum: float,
    full_count: int,
    samples: int,
    seen_samples: int,
    overlimit_seen: int,
    skipped_overlimit: int,
    dz_nm: float,
) -> dict:
    return {
        "loss": loss_sum / max(int(batches), 1),
        "acc": float(counts[0].item() / max(counts[1].item(), 1.0)),
        "mat_acc": float(counts[2].item() / max(counts[3].item(), 1.0)),
        "void_acc": float(counts[4].item() / max(counts[5].item(), 1.0)),
        "mean_active_thickness_nm": active_nm_sum / max(int(samples), 1),
        "full_depth_fraction": float(full_count / max(int(samples), 1)),
        "overlimit_fraction": float(overlimit_seen / max(int(seen_samples), 1)),
        "overlimit_skipped": int(skipped_overlimit),
        "overlimit_skip_fraction": float(skipped_overlimit / max(int(seen_samples), 1)),
        "samples_kept": int(samples),
        "samples_seen": int(seen_samples),
        "dz_nm": float(dz_nm),
    }


def metric_score(metrics: dict) -> float:
    """Return the scalar used to rank checkpoints for a validation/train metrics dict."""
    for key in ("score", "mae_mean", "loss"):
        value = metrics.get(key)
        if value is not None:
            return float(value)
    raise KeyError(f"Metrics do not contain one of score/mae_mean/loss: {sorted(metrics.keys())}")


def record_score(record: dict) -> float:
    """Return the checkpoint score for a history record."""
    return metric_score(record.get("val") or record["train"])


def score_name_for_eval_mode(eval_mode: str) -> str:
    """Return the primary score name for an evaluation mode."""
    return "tmm_mae_mean" if str(eval_mode).lower() in {"tmm", "both"} else "denoise_loss"


def comparable_record_score(record: dict, score_name: str) -> float | None:
    """Return a history record score only if it is comparable to the requested score type."""
    metrics = record.get("val") or record["train"]
    existing_score_name = metrics.get("score_name")
    if existing_score_name == score_name:
        return metric_score(metrics)
    if existing_score_name is None and score_name == "denoise_loss" and "loss" in metrics:
        return metric_score(metrics)
    return None


def mae_per_sample(pred: torch.Tensor, target: torch.Tensor, wavelengths: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
    """Return per-sample RAT MAE over the configured ROI/channels."""
    wl = wavelengths.to(device=pred.device, dtype=torch.float32)
    mask = torch.ones_like(wl, dtype=torch.bool)
    roi_min = cfg.get("ROI_MIN")
    roi_max = cfg.get("ROI_MAX")
    if roi_min is not None:
        mask &= wl >= float(roi_min)
    if roi_max is not None:
        mask &= wl <= float(roi_max)

    diff = (pred - target).abs()
    channels = cfg.get("MAE_CHANNELS")
    if channels:
        if isinstance(channels, str):
            channels = [channels]
        index = {"R": 0, "A": 1, "T": 2}
        channel_idx = [index[str(channel).upper()] for channel in channels]
        diff = diff[:, channel_idx, :]
    return diff[:, :, mask].mean(dim=(1, 2))


def simulate_decoded(
    token_ids: torch.Tensor,
    *,
    tmm_ctx: optollama.evaluation.simulation.TMMContext,
    eos_idx: int,
    pad_idx: int,
    msk_idx: int,
    tmm_batch_size: int,
) -> torch.Tensor:
    """Simulate decoded token stacks in chunks."""
    outputs = []
    for start in range(0, int(token_ids.size(0)), int(tmm_batch_size)):
        chunk = token_ids[start : start + int(tmm_batch_size)].to(tmm_ctx.wl.device)
        outputs.append(
            optollama.evaluation.simulation.simulate_token_sequence(
                chunk,
                tmm_ctx,
                eos=eos_idx,
                pad=pad_idx,
                msk=msk_idx,
            ).detach()
        )
    return torch.cat(outputs, dim=0)


def simulate_field_runs(
    fields_cpu: torch.Tensor,
    *,
    vocab: optollama.data.DepthFieldVocab,
    tmm_ctx: optollama.evaluation.simulation.TMMContext,
    material_to_token_id: dict[str, int],
    dz_nm: float,
    eos_idx: int,
    pad_idx: int,
    msk_idx: int,
    tmm_batch_size: int,
) -> torch.Tensor:
    """Simulate sampled depth fields as native material runs in chunks."""
    runs_batch = [optollama.data.depth_field_runs(field, vocab, dz_nm=dz_nm) for field in fields_cpu]
    outputs = []
    for start in range(0, len(runs_batch), int(tmm_batch_size)):
        outputs.append(
            optollama.evaluation.simulation.simulate_material_runs(
                runs_batch[start : start + int(tmm_batch_size)],
                tmm_ctx,
                material_to_token_id=material_to_token_id,
                eos=eos_idx,
                pad=pad_idx,
                msk=msk_idx,
            ).detach()
        )
    return torch.cat(outputs, dim=0)


def safe_trigger_name(trigger: str) -> str:
    """Return a filesystem-safe validation trigger label."""
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(trigger))


def stack_total_thickness_nm(tokens: list[str]) -> float:
    total = 0.0
    for token in tokens:
        parts = optollama.data.layer_token_parts(token)
        if parts is not None:
            total += float(parts[1])
    return total


def gather_validation_records(local_records: list[dict]) -> list[dict]:
    """Gather JSON-serializable validation records to rank 0."""
    if not ddp_active():
        return local_records
    gathered: list[list[dict]] = [[] for _ in range(ddp_world_size())]
    torch.distributed.all_gather_object(gathered, local_records)
    if ddp_rank() != 0:
        return []
    records: list[dict] = []
    for part in gathered:
        records.extend(part)
    return records


def validation_candidate_record(
    *,
    flat_idx: int,
    fields_cpu: torch.Tensor,
    token_ids_cpu: torch.Tensor,
    mae_values: torch.Tensor,
    pred_spectra_cpu: torch.Tensor,
    target_spectrum_cpu: torch.Tensor,
    vocab: optollama.data.DepthFieldVocab,
    idx_to_token: dict[int, str],
    eos_idx: int,
    pad_idx: int,
    dz_nm: float,
    record_spectra: bool,
) -> dict:
    tokens = optollama.data.token_stack_strings(
        token_ids_cpu[flat_idx],
        idx_to_token,
        eos_idx=eos_idx,
        pad_idx=pad_idx,
    )
    active_bins = int(optollama.data.depth_field_active_bins(fields_cpu[flat_idx], vocab.void_id).item())
    field_runs = optollama.data.depth_field_runs(fields_cpu[flat_idx], vocab, dz_nm=dz_nm)
    mae = float(mae_values[flat_idx].item())
    record = {
        "mae": mae,
        "field_mae": mae,
        "tokens": tokens,
        "material_layers": int(len(tokens)),
        "decoded_total_thickness_nm": float(stack_total_thickness_nm(tokens)),
        "field_active_thickness_nm": float(active_bins * dz_nm),
        "field_material_runs": int(len(field_runs)),
        "field_total_thickness_nm": float(sum(float(run["thickness_nm"]) for run in field_runs)),
        "field_runs": field_runs,
    }
    if record_spectra:
        record["target_spectra"] = target_spectrum_cpu.detach().cpu().tolist()
        record["pred_spectra"] = pred_spectra_cpu[flat_idx].detach().cpu().tolist()
    return record


@torch.no_grad()
def run_tmm_evaluation(
    *,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    tmm_ctx: optollama.evaluation.simulation.TMMContext,
    cfg: dict[str, Any],
    idx_to_token: dict[int, str],
    vocab: optollama.data.DepthFieldVocab,
    args: argparse.Namespace,
    eos_idx: int,
    pad_idx: int,
    msk_idx: int,
    epoch: int,
    epochs: int,
    trigger: str,
    save_samples_path: Path | None,
) -> dict:
    """Run all-mask depth-field sampling and score native material runs by TMM MAE."""
    model.eval()
    sample_model = unwrap_model(model)
    set_loader_epoch(loader, epoch)

    mc_samples = int(args.eval_mc_samples)
    if mc_samples <= 0:
        raise ValueError(f"--eval-mc-samples must be positive, got {mc_samples}")

    mae_values: list[torch.Tensor] = []
    best_layers: list[torch.Tensor] = []
    best_thickness: list[torch.Tensor] = []
    target_overlimit = 0
    samples_seen = 0
    validation_records: list[dict] = []
    material_to_token_id = optollama.data.depth_field_material_token_ids(vocab)
    show_progress = ddp_rank() == 0
    save_samples = save_samples_path is not None
    pbar = tqdm.tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs} tmm", leave=True, disable=not show_progress)

    for batch in pbar:
        spectra_cpu, stacks_cpu = batch[0], batch[1]
        batch_size = int(spectra_cpu.size(0))
        batch_start = samples_seen
        samples_seen += batch_size
        indices_cpu = (
            batch[2].detach().cpu()
            if len(batch) > 2 and torch.is_tensor(batch[2])
            else torch.arange(batch_start, batch_start + batch_size, dtype=torch.long)
        )
        target_thickness = optollama.data.token_stack_total_thickness_nm(
            stacks_cpu,
            idx_to_token,
            eos_idx=eos_idx,
            pad_idx=pad_idx,
            msk_idx=msk_idx,
        )
        target_overlimit += int((target_thickness > (float(args.max_thickness_nm) + 1.0e-6)).sum().item())

        spectra = spectra_cpu.to(device, non_blocking=True)
        spectra_rep = spectra.repeat_interleave(mc_samples, dim=0)
        fields = sample_model.sample(
            spectra_rep,
            steps=args.eval_sampling_steps,
            temperature=float(args.eval_temperature),
            top_k=int(args.eval_top_k),
            deterministic=bool(args.eval_deterministic or args.eval_temperature <= 0.0),
            remask_strategy=str(args.eval_remask_strategy),
        )
        fields_cpu = fields.detach().cpu()
        pred_spectra = simulate_field_runs(
            fields_cpu,
            vocab=vocab,
            tmm_ctx=tmm_ctx,
            material_to_token_id=material_to_token_id,
            dz_nm=float(args.dz_nm),
            eos_idx=eos_idx,
            pad_idx=pad_idx,
            msk_idx=msk_idx,
            tmm_batch_size=int(args.eval_tmm_batch_size),
        )
        target_rep = spectra_cpu.to(tmm_ctx.wl.device).repeat_interleave(mc_samples, dim=0)
        mae = mae_per_sample(pred_spectra, target_rep, cfg["WAVELENGTHS"], cfg).detach().cpu().view(batch_size, mc_samples)
        best_mae, best_idx = mae.min(dim=1)
        flat_idx = torch.arange(batch_size, dtype=torch.long) * mc_samples + best_idx
        best_runs = [optollama.data.depth_field_runs(fields_cpu[int(idx.item())], vocab, dz_nm=float(args.dz_nm)) for idx in flat_idx]
        run_layers = torch.tensor([len(runs) for runs in best_runs], dtype=torch.float32)
        run_thickness = torch.tensor(
            [sum(float(run["thickness_nm"]) for run in runs) for runs in best_runs],
            dtype=torch.float32,
        )

        mae_values.append(best_mae)
        best_layers.append(run_layers)
        best_thickness.append(run_thickness)

        if save_samples:
            output_seq_len = int(args.output_seq_len)
            token_ids_cpu = optollama.data.decode_depth_field_to_tokens(
                fields_cpu,
                vocab,
                output_seq_len=output_seq_len,
                dz_nm=float(args.dz_nm),
                eos_idx=eos_idx,
                pad_idx=pad_idx,
            )
            pred_spectra_cpu = pred_spectra.detach().cpu()
            mae_flat = mae.reshape(-1)
            for row_idx in range(batch_size):
                selected_flat = int(flat_idx[row_idx].item())
                record = {
                    "dataset_index": int(indices_cpu[row_idx].item()),
                    "best_mc": int(best_idx[row_idx].item()),
                    "mc_samples": int(mc_samples),
                    "target_tokens": optollama.data.token_stack_strings(
                        stacks_cpu[row_idx],
                        idx_to_token,
                        eos_idx=eos_idx,
                        pad_idx=pad_idx,
                    ),
                    "target_total_thickness_nm": float(target_thickness[row_idx].item()),
                }
                record.update(
                    validation_candidate_record(
                        flat_idx=selected_flat,
                        fields_cpu=fields_cpu,
                        token_ids_cpu=token_ids_cpu,
                        mae_values=mae_flat,
                        pred_spectra_cpu=pred_spectra_cpu,
                        target_spectrum_cpu=spectra_cpu[row_idx],
                        vocab=vocab,
                        idx_to_token=idx_to_token,
                        eos_idx=eos_idx,
                        pad_idx=pad_idx,
                        dz_nm=float(args.dz_nm),
                        record_spectra=bool(args.eval_record_spectra),
                    )
                )
                if bool(args.eval_record_all_mc):
                    record["all_mc"] = [
                        validation_candidate_record(
                            flat_idx=row_idx * mc_samples + candidate_idx,
                            fields_cpu=fields_cpu,
                            token_ids_cpu=token_ids_cpu,
                            mae_values=mae_flat,
                            pred_spectra_cpu=pred_spectra_cpu,
                            target_spectrum_cpu=spectra_cpu[row_idx],
                            vocab=vocab,
                            idx_to_token=idx_to_token,
                            eos_idx=eos_idx,
                            pad_idx=pad_idx,
                            dz_nm=float(args.dz_nm),
                            record_spectra=bool(args.eval_record_spectra),
                        )
                        for candidate_idx in range(mc_samples)
                    ]
                validation_records.append(record)
        if show_progress:
            all_mae = torch.cat(mae_values)
            pbar.set_postfix(
                mae=f"{all_mae.mean().item():.5f}",
                best=f"{all_mae.min().item():.5f}",
                layers=f"{torch.cat(best_layers).mean().item():.1f}",
                th=f"{torch.cat(best_thickness).mean().item():.0f}nm",
            )

    if mae_values:
        local_mae = torch.cat(mae_values)
        local_layers = torch.cat(best_layers)
        local_thickness = torch.cat(best_thickness)
    else:
        local_mae = torch.empty(0, dtype=torch.float32)
        local_layers = torch.empty(0, dtype=torch.float32)
        local_thickness = torch.empty(0, dtype=torch.float32)

    all_mae = gather_1d_tensor(local_mae)
    all_layers = gather_1d_tensor(local_layers)
    all_thickness = gather_1d_tensor(local_thickness)
    target_totals = torch.tensor([float(target_overlimit), float(samples_seen)], dtype=torch.float64, device=device)
    all_reduce_sum(target_totals)

    if all_mae.numel() == 0:
        raise RuntimeError("TMM validation produced no samples.")

    metrics = {
        "score": float(all_mae.mean().item()),
        "score_name": "tmm_mae_mean",
        "mae_mean": float(all_mae.mean().item()),
        "mae_median": float(all_mae.median().item()),
        "mae_min": float(all_mae.min().item()),
        "mae_max": float(all_mae.max().item()),
        "score_basis": "field_material_runs",
        "material_runs_mean": float(all_layers.mean().item()),
        "field_total_thickness_nm_mean": float(all_thickness.mean().item()),
        "target_overlimit_fraction": float(target_totals[0].item() / max(int(target_totals[1].item()), 1)),
        "samples_seen": int(target_totals[1].item()),
        "mc_samples": int(mc_samples),
        "sampling_steps": int(args.eval_sampling_steps or sample_model.timesteps),
        "remask_strategy": str(args.eval_remask_strategy),
        "tmm_batch_size": int(args.eval_tmm_batch_size),
    }
    if save_samples:
        all_records = gather_validation_records(validation_records)
        if ddp_rank() == 0 and save_samples_path is not None:
            all_records.sort(key=lambda item: int(item.get("dataset_index", 0)))
            os.makedirs(save_samples_path.parent, exist_ok=True)
            payload = {
                "summary": {
                    "config": str(args.config),
                    "epoch": int(epoch),
                    "epoch_1based": int(epoch + 1),
                    "trigger": str(trigger),
                    "samples": int(len(all_records)),
                    "mc_samples": int(mc_samples),
                    "score_mode": "field",
                    "rank_by": "field",
                    "remask_strategy": str(args.eval_remask_strategy),
                    "record_spectra": bool(args.eval_record_spectra),
                    "record_all_mc": bool(args.eval_record_all_mc),
                    "depth_field": {
                        "dz_nm": float(args.dz_nm),
                        "max_thickness_nm": float(args.max_thickness_nm),
                        "depth_bins": optollama.data.depth_bins_for(args.max_thickness_nm, args.dz_nm),
                        "classes": list(vocab.material_names),
                    },
                    "metrics": metrics,
                },
                "results": all_records,
            }
            optollama.utils.save_as_json(str(save_samples_path), payload)
            metrics["samples_path"] = str(save_samples_path)
            metrics["samples_saved"] = int(len(all_records))
    return metrics


def run_epoch(
    *,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    idx_to_token: dict[int, str],
    vocab: optollama.data.DepthFieldVocab,
    args: argparse.Namespace,
    eos_idx: int,
    pad_idx: int,
    msk_idx: int,
    epoch: int,
    epochs: int,
    train: bool,
    validation_callback: Callable[[int, int, dict], None] | None = None,
    validate_every_samples: int = 0,
    global_sample_offset: int = 0,
    optimizer_base_lr: float | None = None,
    lr_schedule: dict[str, Any] | None = None,
    timestep_schedule: dict[str, Any] | None = None,
    ema: ModelEma | None = None,
) -> dict:
    model.train(train)
    set_loader_epoch(loader, epoch)

    loss_sum = 0.0
    batches = 0
    samples = 0
    seen_samples = 0
    overlimit_seen = 0
    skipped_overlimit = 0
    full_count = 0
    active_nm_sum = 0.0
    counts = torch.zeros(6, dtype=torch.float64, device=device)
    desc = f"Epoch {epoch + 1}/{epochs} {'train' if train else 'val'}"
    show_progress = ddp_rank() == 0
    pbar = tqdm.tqdm(loader, desc=desc, leave=True, disable=not show_progress)
    validation_interval = int(validate_every_samples or 0)
    next_validation_sample = validation_interval if train and validation_callback is not None and validation_interval > 0 else None
    base_lr = float(optimizer_base_lr if optimizer_base_lr is not None else (optimizer.param_groups[0]["lr"] if optimizer is not None else 0.0))
    current_lr = base_lr

    for batch in pbar:
        raw_spectra_cpu, raw_stacks_cpu = batch[0], batch[1]
        spectra_cpu, stacks_cpu = raw_spectra_cpu, raw_stacks_cpu
        batch_seen_start = int(seen_samples)
        global_samples_before = int(global_sample_offset + batch_seen_start * ddp_world_size())
        true_thickness_nm = optollama.data.token_stack_total_thickness_nm(
            stacks_cpu,
            idx_to_token,
            eos_idx=eos_idx,
            pad_idx=pad_idx,
            msk_idx=msk_idx,
        )
        seen_samples += int(stacks_cpu.size(0))
        overlimit = true_thickness_nm > (float(args.max_thickness_nm) + 1.0e-6)
        overlimit_count = int(overlimit.sum().item())
        overlimit_seen += overlimit_count
        count_batch_metrics = True
        if not args.keep_overlimit_stacks:
            skipped_overlimit += overlimit_count
            keep = ~overlimit
            if not bool(keep.any()):
                if ddp_active():
                    count_batch_metrics = False
                    spectra_cpu = raw_spectra_cpu[:1]
                    stacks_cpu = raw_stacks_cpu[:1]
                else:
                    metrics = reduced_epoch_metrics(
                        counts=counts,
                        loss_sum=loss_sum,
                        batches=batches,
                        active_nm_sum=active_nm_sum,
                        full_count=full_count,
                        samples=samples,
                        seen_samples=seen_samples,
                        overlimit_seen=overlimit_seen,
                        skipped_overlimit=skipped_overlimit,
                        dz_nm=args.dz_nm,
                        device=device,
                    )
                    if show_progress:
                        pbar.set_postfix(skip=f"{metrics['overlimit_skip_fraction'] * 100.0:.1f}%")
                    if next_validation_sample is not None and int(metrics["samples_seen"]) >= next_validation_sample:
                        validation_callback(epoch, int(metrics["samples_seen"]), dict(metrics))
                        model.train(True)
                        while next_validation_sample <= int(metrics["samples_seen"]):
                            next_validation_sample += validation_interval
                    continue
            spectra_cpu = spectra_cpu[keep]
            stacks_cpu = stacks_cpu[keep]

        fields_cpu = optollama.data.rasterize_stack_to_depth_field(
            stacks_cpu,
            idx_to_token,
            vocab,
            dz_nm=args.dz_nm,
            max_thickness_nm=args.max_thickness_nm,
            eos_idx=eos_idx,
            pad_idx=pad_idx,
            msk_idx=msk_idx,
        )
        if count_batch_metrics:
            active_bins = optollama.data.depth_field_active_bins(fields_cpu, vocab.void_id)
            full_count += int((active_bins >= fields_cpu.size(1)).sum().item())
            active_nm_sum += float(active_bins.float().sum().item() * float(args.dz_nm))
            samples += int(fields_cpu.size(0))

        spectra = spectra_cpu.to(device, non_blocking=True)
        fields = fields_cpu.to(device, non_blocking=True)

        if train:
            if optimizer is None:
                raise RuntimeError("optimizer is required for training")
            current_lr = scheduled_learning_rate(base_lr, lr_schedule, global_samples_before)
            set_optimizer_lr(optimizer, current_lr)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(bool(scaler.is_enabled())):
                out = depth_field_training_loss(
                    model,
                    spectra,
                    fields,
                    void_id=vocab.void_id,
                    void_loss_weight=args.void_loss_weight,
                    random_replace_prob=args.random_replace_prob,
                    loss_on_corrupted_only=args.loss_on_corrupted_only,
                    timestep_schedule=timestep_schedule,
                    global_samples_seen=global_samples_before,
                )
            loss = out["loss"]
            if not count_batch_metrics:
                loss = loss * 0.0
            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
        else:
            with torch.no_grad(), autocast_context(bool(scaler.is_enabled())):
                out = depth_field_training_loss(
                    model,
                    spectra,
                    fields,
                    void_id=vocab.void_id,
                    void_loss_weight=args.void_loss_weight,
                    random_replace_prob=args.random_replace_prob,
                    loss_on_corrupted_only=args.loss_on_corrupted_only,
                )
            loss = out["loss"]
            if not count_batch_metrics:
                loss = loss * 0.0

        if count_batch_metrics:
            loss_sum += float(loss.detach().item())
            batches += 1
            counts += accuracy_counts(out["logits"].detach(), fields, vocab.void_id)
        metrics = reduced_epoch_metrics(
            counts=counts,
            loss_sum=loss_sum,
            batches=batches,
            active_nm_sum=active_nm_sum,
            full_count=full_count,
            samples=samples,
            seen_samples=seen_samples,
            overlimit_seen=overlimit_seen,
            skipped_overlimit=skipped_overlimit,
            dz_nm=args.dz_nm,
            device=device,
            )
        metrics["global_samples_seen"] = int(global_sample_offset + int(metrics["samples_seen"]))
        if train:
            metrics["lr"] = float(current_lr)
            metrics["timestep_min_fraction"] = float(timestep_min_fraction(timestep_schedule, global_samples_before))
        if show_progress:
            pbar.set_postfix(
                loss=f"{metrics['loss']:.4f}",
                lr=f"{metrics.get('lr', current_lr):.2e}",
                acc=f"{metrics['acc'] * 100.0:.2f}%",
                mat=f"{metrics['mat_acc'] * 100.0:.2f}%",
                void=f"{metrics['void_acc'] * 100.0:.2f}%",
                th=f"{metrics['mean_active_thickness_nm']:.0f}nm",
                skip=f"{metrics['overlimit_skip_fraction'] * 100.0:.1f}%",
                full=f"{metrics['full_depth_fraction'] * 100.0:.1f}%",
            )

        if next_validation_sample is not None and int(metrics["samples_seen"]) >= next_validation_sample:
            validation_callback(epoch, int(metrics["samples_seen"]), dict(metrics))
            model.train(True)
            while next_validation_sample <= int(metrics["samples_seen"]):
                next_validation_sample += validation_interval

    metrics = reduced_epoch_metrics(
        counts=counts,
        loss_sum=loss_sum,
        batches=batches,
        active_nm_sum=active_nm_sum,
        full_count=full_count,
        samples=samples,
        seen_samples=seen_samples,
        overlimit_seen=overlimit_seen,
        skipped_overlimit=skipped_overlimit,
        dz_nm=args.dz_nm,
        device=device,
    )
    metrics["global_samples_seen"] = int(global_sample_offset + int(metrics["samples_seen"]))
    if train:
        metrics["lr"] = float(current_lr)
        metrics["timestep_min_fraction"] = float(timestep_min_fraction(timestep_schedule, int(global_sample_offset + int(metrics["samples_seen"]))))
    if int(metrics["samples_kept"]) == 0:
        raise RuntimeError(
            "No samples remained after over-limit filtering. Increase --max-thickness-nm or use --keep-overlimit-stacks."
        )

    return metrics


def make_checkpoint_extra(
    *,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    vocab: optollama.data.DepthFieldVocab,
    model_config: optollama.model.DepthFieldModelConfig,
    history: list[dict],
    ema: ModelEma | None = None,
    include_ema_state: bool = False,
    checkpoint_weights: str = "live",
) -> dict:
    extra = {
        "depth_field": {
            "dz_nm": float(args.dz_nm),
            "max_thickness_nm": float(args.max_thickness_nm),
            "depth_bins": optollama.data.depth_bins_for(args.max_thickness_nm, args.dz_nm),
            "output_seq_len": int(args.output_seq_len),
            "vocab": vocab.to_dict(),
            "representation": "material_depth_field_with_void",
            "filter_overlimit_stacks": not bool(args.keep_overlimit_stacks),
            "eval_mode": str(args.eval_mode),
            "eval_mc_samples": int(args.eval_mc_samples),
            "eval_sampling_steps": args.eval_sampling_steps,
            "eval_temperature": float(args.eval_temperature),
            "eval_top_k": int(args.eval_top_k),
            "eval_deterministic": bool(args.eval_deterministic),
            "eval_remask_strategy": str(args.eval_remask_strategy),
            "eval_tmm_batch_size": int(args.eval_tmm_batch_size),
            "save_eval_samples": bool(args.save_eval_samples),
            "eval_samples_dir": args.eval_samples_dir,
            "eval_record_spectra": bool(args.eval_record_spectra),
            "eval_record_all_mc": bool(args.eval_record_all_mc),
        },
        "model_config": model_config.to_dict(),
        "config_path": str(args.config),
        "checkpoint_weights": str(checkpoint_weights),
        "train_schedules": {
            "lr": lr_schedule_config(cfg),
            "timestep": timestep_schedule_config(cfg),
        },
        "ema": {
            "enabled": ema is not None,
            "decay": float(ema.decay) if ema is not None else None,
            "updates": int(ema.updates) if ema is not None else 0,
        },
        "history": history,
    }
    if ema is not None and include_ema_state:
        extra["ema_state"] = ema.state_dict(cpu=True)
    return extra


def save_depth_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    history: list[dict],
    extra: dict,
    ema: ModelEma | None = None,
    use_ema_weights: bool = False,
) -> None:
    train_losses = torch.tensor([item["train"]["loss"] for item in history], dtype=torch.float32)
    val_losses = torch.tensor([record_score(item) for item in history], dtype=torch.float32)
    ctx = ema.apply_to(model) if use_ema_weights and ema is not None else nullcontext()
    with ctx:
        optollama.utils.save_checkpoint(
            str(path),
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_losses=train_losses,
            test_mae=val_losses,
            extra=extra,
        )


def main() -> None:
    args = parse_args()
    cfg = optollama.utils.load_config(args)
    apply_depth_field_defaults(cfg, args)
    apply_loader_overrides(cfg, args)

    if args.seed is not None:
        cfg["SEED"] = int(args.seed)
    device, local_rank, rank, world_size = optollama.utils.setup_run(cfg, make_dirs=False)
    if args.device is not None and world_size == 1:
        device = resolve_device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
    ddp = optollama.utils.is_ddp()
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    tokens, token_to_idx, idx_to_token, _, _, _, eos_idx, pad_idx, msk_idx = optollama.data.init_tokens(cfg["TOKENS_PATH"])
    vocab = optollama.data.build_depth_field_vocab(tokens, token_to_idx)
    depth_bins = optollama.data.depth_bins_for(args.max_thickness_nm, args.dz_nm)

    train_subset = args.max_train_samples if args.max_train_samples is not None else cfg_required(cfg, "NUM_SAMPLES_TRAIN", "--max-train-samples")
    val_subset = args.max_val_samples if args.max_val_samples is not None else cfg_required(cfg, "NUM_SAMPLES_TEST", "--max-val-samples")
    train_ds, train_loader, _ = optollama.data.SpectraDataset.make_loader(cfg, split="train", subset_n=train_subset, ddp=ddp)
    val_loader = None
    if not args.no_val:
        _, val_loader, _ = optollama.data.SpectraDataset.make_loader(cfg, split="test", subset_n=val_subset, ddp=ddp)
    eval_mode = str(args.eval_mode).lower()
    tmm_ctx = None
    if val_loader is not None and eval_mode in {"tmm", "both"}:
        tmm_device = resolve_tmm_device(args.eval_tmm_device, device)
        tmm_ctx = optollama.evaluation.simulation.TMMContext.make(cfg=cfg, idx_to_token=idx_to_token, device=tmm_device)

    model_config = optollama.model.DepthFieldModelConfig(
        spectrum_shape=tuple(int(v) for v in example_spectrum(train_ds).shape),
        num_materials=vocab.num_clean_classes,
        depth_bins=depth_bins,
        model_type=str(args.model_type),
        d_model=int(args.d_model),
        n_blocks=int(args.n_blocks),
        kernel_size=int(args.kernel_size),
        n_heads=int(args.n_heads),
        ffn_multiplier=float(args.ffn_multiplier),
        conv_type=str(args.conv_type),
        timesteps=int(args.diffusion_steps),
        dropout=float(args.dropout),
    )
    model = optollama.model.build_depth_field_model(model_config).to(device)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate if args.learning_rate is not None else cfg_required(cfg, "LEARNING_RATE", "--learning-rate")),
        weight_decay=float(args.weight_decay),
    )
    optimizer_base_lr = float(optimizer.param_groups[0]["lr"])
    lr_schedule = lr_schedule_config(cfg)
    timestep_schedule = timestep_schedule_config(cfg)
    ema = make_ema(cfg, model)

    epochs = int(args.epochs if args.epochs is not None else cfg_required(cfg, "EPOCHS", "--epochs"))
    out_dir = Path(args.out_dir)
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    if ddp:
        torch.distributed.barrier()
    best_path = out_dir / "depth-field-best.pt"
    last_path = out_dir / "depth-field-last.pt"
    ema_best_path = out_dir / "depth-field-best-ema.pt"
    ema_last_path = out_dir / "depth-field-last-ema.pt"
    history_path = out_dir / "depth-field-history.json"
    eval_samples_dir = Path(args.eval_samples_dir) if args.eval_samples_dir is not None else None

    start_epoch = 0
    history: list[dict] = []
    resume_path, resume_source, resume_required = resolve_resume_checkpoint(cfg, args, last_path)
    if resume_path is not None:
        if resume_path.exists():
            start_epoch_loaded, blob = optollama.utils.load_checkpoint(str(resume_path), model, optimizer=optimizer, map_location="cpu")
            start_epoch = int(start_epoch_loaded or 0)
            history = list(((blob.get("extra") or {}).get("history") or []))
            if ema is not None:
                ema_state = ((blob.get("extra") or {}).get("ema_state") or {})
                if ema_state:
                    ema.load_state_dict(ema_state, model)
                else:
                    ema.reset(model)
            if rank == 0:
                print(f"Resumed depth-field checkpoint {resume_path} at epoch {start_epoch} ({resume_source}).")
        elif resume_required:
            raise FileNotFoundError(f"{resume_source} checkpoint does not exist: {resume_path}")
        elif rank == 0:
            print(f"Depth-field resume is enabled via {resume_source}, but {resume_path} does not exist; starting fresh.")

    primary_score_name = score_name_for_eval_mode(eval_mode)
    comparable_scores = [score for item in history if (score := comparable_record_score(item, primary_score_name)) is not None]
    best_score = min(comparable_scores, default=float("inf"))
    validate_every_samples = (
        int(args.validate_every_n_train_samples)
        if args.validate_every_n_train_samples is not None
        else int(cfg_required(cfg, "VALIDATE_EVERY_N_TRAIN_SAMPLES", "--validate-every-n-train-samples"))
    )
    validate_at_epoch_end = (
        bool(args.validate_at_epoch_end)
        if args.validate_at_epoch_end is not None
        else bool(cfg_required(cfg, "VALIDATE_AT_EPOCH_END", "--validate-at-epoch-end"))
    )
    if val_loader is None:
        validate_every_samples = 0

    def run_validation_trigger(epoch: int, trigger: str, train_metrics: dict, samples_seen_epoch: int) -> None:
        nonlocal best_score
        if val_loader is None:
            return

        denoise_metrics = None
        tmm_metrics = None
        validate_with_ema = ema is not None and bool(ema_config(cfg).get("VALIDATE", True))
        validation_ctx = ema.apply_to(model) if validate_with_ema else nullcontext()
        with validation_ctx:
            if eval_mode in {"denoise", "both"}:
                denoise_metrics = run_epoch(
                    model=model,
                    loader=val_loader,
                    optimizer=None,
                    scaler=scaler,
                    device=device,
                    idx_to_token=idx_to_token,
                    vocab=vocab,
                    args=args,
                    eos_idx=eos_idx,
                    pad_idx=pad_idx,
                    msk_idx=msk_idx,
                    epoch=epoch,
                    epochs=epochs,
                    train=False,
                )
                denoise_metrics = {"score": float(denoise_metrics["loss"]), "score_name": "denoise_loss", **denoise_metrics}
            if eval_mode in {"tmm", "both"}:
                if tmm_ctx is None:
                    raise RuntimeError("TMM validation requested but TMM context was not initialized.")
                sample_path = (
                    eval_samples_dir / f"epoch_{epoch + 1:04d}_{safe_trigger_name(trigger)}.json"
                    if bool(args.save_eval_samples) and eval_samples_dir is not None
                    else None
                )
                tmm_metrics = run_tmm_evaluation(
                    model=model,
                    loader=val_loader,
                    device=device,
                    tmm_ctx=tmm_ctx,
                    cfg=cfg,
                    idx_to_token=idx_to_token,
                    vocab=vocab,
                    args=args,
                    eos_idx=eos_idx,
                    pad_idx=pad_idx,
                    msk_idx=msk_idx,
                    epoch=epoch,
                    epochs=epochs,
                    trigger=trigger,
                    save_samples_path=sample_path,
                )

        if eval_mode == "denoise":
            val_metrics = denoise_metrics
        elif eval_mode == "tmm":
            val_metrics = tmm_metrics
        else:
            if tmm_metrics is None or denoise_metrics is None:
                raise RuntimeError("eval_mode='both' expected both denoise and TMM metrics.")
            val_metrics = dict(tmm_metrics)
            for key, value in denoise_metrics.items():
                val_metrics[f"denoise_{key}"] = value

        if rank != 0:
            return

        record = {
            "epoch": int(epoch),
            "trigger": str(trigger),
            "samples_seen_epoch": int(samples_seen_epoch),
            "validation_weights": "ema" if validate_with_ema else "live",
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)

        checkpoint_weights = "ema" if validate_with_ema else "live"
        extra = make_checkpoint_extra(
            args=args,
            cfg=cfg,
            vocab=vocab,
            model_config=model_config,
            history=history,
            ema=ema,
            checkpoint_weights=checkpoint_weights,
        )
        score = metric_score(val_metrics)
        if score < best_score:
            best_score = score
            save_depth_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                history=history,
                extra=extra,
                ema=ema,
                use_ema_weights=validate_with_ema,
            )
            if validate_with_ema:
                save_depth_checkpoint(
                    path=ema_best_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    history=history,
                    extra=extra,
                    ema=ema,
                    use_ema_weights=True,
                )
            print(f"Saved best checkpoint -> {best_path} ({val_metrics.get('score_name', 'score')}={best_score:.6f}, trigger={trigger})")

        optollama.utils.save_as_json(str(history_path), history)

    if rank == 0:
        model_desc = f"type={model_config.model_type}"
        if model_config.model_type in {"attention", "optollama_depth"}:
            model_desc += f", heads={model_config.n_heads}, ffn={model_config.ffn_multiplier:g}"
            if model_config.model_type == "optollama_depth":
                model_desc += ", spectrum_cross_attn=true"
        else:
            model_desc += f", conv={model_config.conv_type}, kernel={model_config.kernel_size}"
        print(
            "Depth-field diffusion: "
            f"materials={vocab.num_clean_classes - 1}+void, bins={depth_bins}, dz={args.dz_nm:g}nm, "
            f"max={args.max_thickness_nm:g}nm, device={device}, amp={amp_enabled}, "
            f"ddp={ddp}, world={world_size}, {model_desc}, "
            f"eval_mc={args.eval_mc_samples}, eval_steps={args.eval_sampling_steps}, "
            f"eval_temp={args.eval_temperature:g}, eval_top_k={args.eval_top_k}, eval_remask={args.eval_remask_strategy}"
        )
        if validate_every_samples > 0:
            print(f"Mid-epoch validation enabled every {validate_every_samples} global train samples.")
        print(f"Epoch-end validation: {validate_at_epoch_end}.")
        if schedule_enabled(lr_schedule):
            print(
                "LR schedule: "
                f"type={lr_schedule.get('TYPE', 'cosine')}, "
                f"warmup={int(lr_schedule.get('WARMUP_SAMPLES', 0))}, "
                f"total={int(lr_schedule.get('TOTAL_SAMPLES', 0))}, "
                f"max={float(lr_schedule.get('MAX_LR', optimizer_base_lr)):g}, "
                f"min={float(lr_schedule.get('MIN_LR', 0.0)):g}."
            )
        if schedule_enabled(timestep_schedule):
            print(
                "Timestep schedule: "
                f"type={timestep_schedule.get('TYPE', 'high_noise_warmup')}, "
                f"warmup={int(timestep_schedule.get('WARMUP_SAMPLES', timestep_schedule.get('HIGH_NOISE_SAMPLES', 0)))}, "
                f"high_min={float(timestep_schedule.get('HIGH_NOISE_MIN_FRACTION', timestep_schedule.get('MIN_FRACTION', 0.6))):g}."
            )
        if ema is not None:
            print(f"EMA enabled: decay={ema.decay:g}, validate={bool(ema_config(cfg).get('VALIDATE', True))}.")
        if val_loader is not None:
            print(f"Validation mode: {eval_mode}.")
            if eval_mode in {"tmm", "both"}:
                if bool(args.save_eval_samples):
                    print(f"TMM validation samples will be saved to {eval_samples_dir}.")
                else:
                    print("TMM validation sample saving disabled.")

    for epoch in range(start_epoch, epochs):
        mid_validation_callback = None
        if validate_every_samples > 0 and val_loader is not None:
            def mid_validation_callback(e: int, seen: int, metrics: dict) -> None:
                run_validation_trigger(e, f"sample_{seen}", metrics, seen)

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            idx_to_token=idx_to_token,
            vocab=vocab,
            args=args,
            eos_idx=eos_idx,
            pad_idx=pad_idx,
            msk_idx=msk_idx,
            epoch=epoch,
            epochs=epochs,
            train=True,
            validation_callback=mid_validation_callback,
            validate_every_samples=validate_every_samples,
            global_sample_offset=int(epoch) * int(train_subset),
            optimizer_base_lr=optimizer_base_lr,
            lr_schedule=lr_schedule,
            timestep_schedule=timestep_schedule,
            ema=ema,
        )

        if val_loader is not None and validate_at_epoch_end:
            run_validation_trigger(
                epoch=epoch,
                trigger="epoch_end",
                train_metrics=train_metrics,
                samples_seen_epoch=int(train_metrics["samples_seen"]),
            )
        elif rank == 0:
            history.append(
                {
                    "epoch": int(epoch),
                    "trigger": "epoch_end",
                    "samples_seen_epoch": int(train_metrics["samples_seen"]),
                    "validation_weights": "none",
                    "train": train_metrics,
                }
            )

        if rank == 0:
            extra = make_checkpoint_extra(
                args=args,
                cfg=cfg,
                vocab=vocab,
                model_config=model_config,
                history=history,
                ema=ema,
                include_ema_state=ema is not None,
                checkpoint_weights="live",
            )
            if args.save_every > 0 and ((epoch + 1) % int(args.save_every) == 0 or epoch == epochs - 1):
                save_depth_checkpoint(path=last_path, model=model, optimizer=optimizer, epoch=epoch, history=history, extra=extra)
                print(f"Saved last checkpoint -> {last_path}")
                if ema is not None:
                    ema_extra = make_checkpoint_extra(
                        args=args,
                        cfg=cfg,
                        vocab=vocab,
                        model_config=model_config,
                        history=history,
                        ema=ema,
                        checkpoint_weights="ema",
                    )
                    save_depth_checkpoint(
                        path=ema_last_path,
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        history=history,
                        extra=ema_extra,
                        ema=ema,
                        use_ema_weights=True,
                    )
                    print(f"Saved EMA last checkpoint -> {ema_last_path}")

            if val_loader is None:
                score = metric_score(train_metrics)
                if score < best_score:
                    best_score = score
                    save_depth_checkpoint(path=best_path, model=model, optimizer=optimizer, epoch=epoch, history=history, extra=extra)
                    print(f"Saved best checkpoint -> {best_path} (train_loss={best_score:.6f})")

            optollama.utils.save_as_json(str(history_path), history)


if __name__ == "__main__":
    optollama.utils.stop_ddp()
    try:
        main()
    finally:
        optollama.utils.stop_ddp()
