#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
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
            "optollama_depth_windowed",
            "optollama-depth-windowed",
            "optollama_windowed_depth",
            "optollama-windowed-depth",
            "optollama_depth_patched",
            "optollama-depth-patched",
            "windowed_depth",
            "windowed-depth",
            "optollama_depth_windowed_v2",
            "optollama-depth-windowed-v2",
            "optollama_windowed_depth_v2",
            "optollama-windowed-depth-v2",
            "windowed_depth_v2",
            "windowed-depth-v2",
            "optollama_depth_windowed_v3",
            "optollama-depth-windowed-v3",
            "optollama_windowed_depth_v3",
            "optollama-windowed-depth-v3",
            "windowed_depth_v3",
            "windowed-depth-v3",
            "hybrid",
            "depth_hybrid",
            "depth-hybrid",
            "optollama_depth_hybrid",
            "optollama-depth-hybrid",
        ],
        help=(
            "Depth-field backbone type. 'conv' uses dilated Conv1d blocks, 'attention' uses global self-attention, "
            "'optollama_depth' uses OptoLlama-style cross/self-attention blocks, and "
            "'optollama_depth_windowed' adds wavelength-window spectrum conditioning, and its V2 variant adds "
            "pooled-spectrum AdaLN conditioning plus final LayerNorm. V3 instead appends the pooled spectrum as "
            "a global cross-attention token and keeps AdaLN timestep-only. 'hybrid' interleaves V3 blocks with "
            "conditioned depth-axis convolutions."
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
    parser.add_argument(
        "--hybrid-dilations",
        type=int,
        nargs="+",
        default=None,
        help="One depth-axis convolution dilation per transformer block in the hybrid model.",
    )
    parser.add_argument(
        "--hybrid-residual-init",
        type=float,
        default=None,
        help="Initial residual scale for interleaved hybrid convolution blocks.",
    )
    parser.add_argument("--n-heads", type=int, default=None, help="Attention heads when --model-type=attention.")
    parser.add_argument("--ffn-multiplier", type=float, default=None, help="Attention feed-forward width multiplier.")
    parser.add_argument("--spectrum-patch-size", type=int, default=None, help="Wavelength samples per spectrum window.")
    parser.add_argument("--spectrum-patch-stride", type=int, default=None, help="Stride between spectrum windows.")
    parser.add_argument("--spectrum-encoder-blocks", type=int, default=None, help="Self-attention blocks over spectrum windows.")
    parser.add_argument("--spectrum-encoder-heads", type=int, default=None, help="Attention heads in the spectrum encoder.")
    parser.add_argument(
        "--spectrum-ffn-multiplier",
        type=float,
        default=None,
        help="Feed-forward width multiplier in spectrum encoder blocks.",
    )
    parser.add_argument("--dropout", type=float, default=None, help="Dropout inside residual blocks.")
    parser.add_argument("--diffusion-steps", type=int, default=None, help="Discrete depth-field diffusion timesteps.")

    parser.add_argument("--learning-rate", type=float, default=None, help="Optimizer LR. Defaults to config LEARNING_RATE.")
    parser.add_argument("--weight-decay", type=float, default=None, help="AdamW weight decay.")
    parser.add_argument("--grad-clip", type=float, default=None, help="Gradient norm clip. <=0 disables clipping.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None, help="Use CUDA autocast/GradScaler.")
    parser.add_argument(
        "--amp-dtype",
        type=str,
        choices=["auto", "float16", "fp16", "bfloat16", "bf16"],
        default=None,
        help="CUDA autocast dtype. 'auto' uses BF16 only when every DDP rank supports it.",
    )
    parser.add_argument(
        "--max-consecutive-nonfinite-steps",
        type=int,
        default=None,
        help="Abort after this many consecutive non-finite forward/gradient steps. 0 aborts immediately.",
    )

    parser.add_argument("--void-loss-weight", type=float, default=None, help="CE class weight for the void depth class.")
    parser.add_argument(
        "--random-replace-prob",
        type=float,
        default=None,
        help="Maximum fraction of corrupted bins that are random material replacements rather than masks.",
    )
    parser.add_argument(
        "--random-replace-schedule",
        choices=["constant", "noise_complement", "noise-complement"],
        default=None,
        help="Schedule the replacement fraction by diffusion noise probability.",
    )
    parser.add_argument(
        "--random-replace-power",
        type=float,
        default=None,
        help="Exponent applied by the noise-complement replacement schedule.",
    )
    parser.add_argument("--corruption-mode", choices=["iid", "hybrid"], default=None, help="Depth-bin corruption layout.")
    parser.add_argument("--corruption-iid-fraction", type=float, default=None, help="Hybrid budget fraction for independent bins.")
    parser.add_argument("--corruption-span-fraction", type=float, default=None, help="Hybrid budget fraction for contiguous spans.")
    parser.add_argument("--corruption-layer-fraction", type=float, default=None, help="Hybrid budget fraction for complete material runs.")
    parser.add_argument("--corruption-span-min-bins", type=int, default=None, help="Minimum contiguous corruption span in depth bins.")
    parser.add_argument("--corruption-span-max-bins", type=int, default=None, help="Maximum contiguous corruption span in depth bins.")
    parser.add_argument(
        "--corruption-span-scale-with-noise",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Increase the allowed corruption span length with the timestep noise level.",
    )
    parser.add_argument(
        "--loss-on-corrupted-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compute CE only on bins that were masked/replaced. Default supervises every bin.",
    )
    parser.add_argument(
        "--corrupted-loss-weight",
        type=float,
        default=None,
        help="Relative CE weight for bins changed by corruption.",
    )
    parser.add_argument(
        "--uncorrupted-loss-weight",
        type=float,
        default=None,
        help="Relative CE weight for bins left unchanged by corruption.",
    )
    parser.add_argument(
        "--condition-dropout-prob",
        type=float,
        default=None,
        help="Probability of replacing a complete target spectrum with the all-zero CFG condition during training.",
    )
    parser.add_argument(
        "--boundary-loss",
        dest="boundary_loss_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Increase CE weight around material transitions.",
    )
    parser.add_argument(
        "--boundary-loss-radius-bins",
        type=int,
        default=None,
        help="Number of bins to expand on each side of transition-adjacent bins.",
    )
    parser.add_argument(
        "--boundary-loss-weight",
        type=float,
        default=None,
        help="CE multiplier for bins selected by the boundary mask.",
    )
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint to resume.")
    parser.add_argument("--init-from", type=str, default=None, help="Initialize model weights for a new fine-tuning run.")
    parser.add_argument(
        "--reset-optimizer-on-resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Restore model/epoch/history but start with fresh optimizer and AMP scaler state.",
    )
    parser.add_argument(
        "--save-validation-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Rotate a recovery checkpoint after every finite mid-training validation.",
    )
    parser.add_argument(
        "--spectral-aux",
        dest="spectral_aux_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable the frozen field-to-spectrum auxiliary training loss.",
    )
    parser.add_argument("--spectral-aux-checkpoint", type=str, default=None, help="Forward-surrogate checkpoint.")
    parser.add_argument("--spectral-aux-weight", type=float, default=None, help="Maximum spectral auxiliary loss weight.")
    parser.add_argument("--spectral-aux-every-n-steps", type=int, default=None, help="Apply spectral loss every N train steps.")
    parser.add_argument(
        "--spectral-aux-max-samples-per-rank",
        type=int,
        default=None,
        help="Maximum eligible local samples sent through the surrogate on an auxiliary step.",
    )
    parser.add_argument(
        "--solution-bank",
        nargs="+",
        default=None,
        help="Solution-bank shard files/directories used for exact-TMM-verified replay.",
    )
    parser.add_argument(
        "--solution-bank-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable solution-bank replay.",
    )
    parser.add_argument(
        "--solution-bank-replay-fraction",
        type=float,
        default=None,
        help="Fraction of each retained training batch replaced with solution-bank examples.",
    )
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
    parser.add_argument(
        "--eval-cfg-scale",
        type=float,
        default=None,
        help="Classifier-free guidance scale. 1 uses one conditional forward pass; other positive values use CFG.",
    )
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
        "--eval-fail-on-nonfinite",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fail validation if sampling has to repair any non-finite model logits.",
    )
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


def resolve_init_checkpoint(cfg: dict[str, Any], args: argparse.Namespace) -> tuple[Path | None, str | None]:
    """Resolve a weights-only initialization checkpoint for a new run."""
    if args.init_from:
        return Path(args.init_from), "--init-from"
    checkpoint_cfg = cfg.get("CHECKPOINT") if isinstance(cfg.get("CHECKPOINT"), dict) else {}
    value = checkpoint_cfg.get("INIT_FROM")
    if value is None:
        value = checkpoint_cfg.get("INIT_PATH")
    if value is None or value is False:
        return None, None
    return Path(str(value)), "CHECKPOINT.INIT_FROM"


def spectral_aux_config(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Return the normalized frozen-surrogate auxiliary objective settings."""
    block = depth_field_block(cfg)
    raw = nested_get(block, "TRAIN", "SPECTRAL_AUX", default={}) or {}
    if not isinstance(raw, dict):
        raise TypeError("DEPTH_FIELD.TRAIN.SPECTRAL_AUX must be a mapping.")
    enabled = bool(args.spectral_aux_enabled if args.spectral_aux_enabled is not None else raw.get("ENABLED", False))
    checkpoint = args.spectral_aux_checkpoint or raw.get("CHECKPOINT")
    weight = float(args.spectral_aux_weight if args.spectral_aux_weight is not None else raw.get("WEIGHT", 0.0))
    every_n_steps = int(
        args.spectral_aux_every_n_steps
        if args.spectral_aux_every_n_steps is not None
        else raw.get("EVERY_N_STEPS", 4)
    )
    max_samples = int(
        args.spectral_aux_max_samples_per_rank
        if args.spectral_aux_max_samples_per_rank is not None
        else raw.get("MAX_SAMPLES_PER_RANK", 8)
    )
    channel_names = raw.get("CHANNELS", ["R", "T"])
    if isinstance(channel_names, str):
        channel_names = [channel_names]
    channel_map = {"R": 0, "A": 1, "T": 2}
    try:
        channels = tuple(channel_map[str(name).upper()] for name in channel_names)
    except KeyError as exc:
        raise ValueError(f"Unknown SPECTRAL_AUX channel {exc.args[0]!r}; expected R, A, or T.") from exc
    config = {
        "enabled": enabled,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "weight": weight,
        "every_n_steps": every_n_steps,
        "max_samples_per_rank": max_samples,
        "max_noise_probability": float(raw.get("MAX_NOISE_PROBABILITY", 0.5)),
        "skip_dropped_conditions": bool(raw.get("SKIP_DROPPED_CONDITIONS", True)),
        "weight_ramp_samples": int(raw.get("WEIGHT_RAMP_SAMPLES", 5_000_000)),
        "start_after_samples": int(raw.get("START_AFTER_SAMPLES", 0)),
        "derivative_weight": float(raw.get("DERIVATIVE_WEIGHT", 0.25)),
        "huber_delta": float(raw.get("HUBER_DELTA", 0.02)),
        "channels": channels,
        "channel_names": [str(name).upper() for name in channel_names],
        "straight_through_temperature": float(raw.get("STRAIGHT_THROUGH_TEMPERATURE", 1.0)),
    }
    if enabled and not config["checkpoint"]:
        raise ValueError("SPECTRAL_AUX.ENABLED=true requires SPECTRAL_AUX.CHECKPOINT.")
    if weight < 0.0 or not math.isfinite(weight):
        raise ValueError(f"SPECTRAL_AUX.WEIGHT must be finite and non-negative, got {weight}.")
    if every_n_steps <= 0 or max_samples <= 0:
        raise ValueError("SPECTRAL_AUX.EVERY_N_STEPS and MAX_SAMPLES_PER_RANK must be positive.")
    if not 0.0 <= config["max_noise_probability"] <= 1.0:
        raise ValueError("SPECTRAL_AUX.MAX_NOISE_PROBABILITY must be in [0,1].")
    if config["weight_ramp_samples"] < 0 or config["start_after_samples"] < 0:
        raise ValueError("SPECTRAL_AUX sample schedule values must be non-negative.")
    if not channels:
        raise ValueError("SPECTRAL_AUX.CHANNELS must contain at least one of R, A, or T.")
    if config["derivative_weight"] < 0.0 or not math.isfinite(config["derivative_weight"]):
        raise ValueError("SPECTRAL_AUX.DERIVATIVE_WEIGHT must be finite and non-negative.")
    if config["huber_delta"] <= 0.0 or not math.isfinite(config["huber_delta"]):
        raise ValueError("SPECTRAL_AUX.HUBER_DELTA must be finite and positive.")
    if config["straight_through_temperature"] <= 0.0 or not math.isfinite(
        config["straight_through_temperature"]
    ):
        raise ValueError("SPECTRAL_AUX.STRAIGHT_THROUGH_TEMPERATURE must be finite and positive.")
    return config


def solution_bank_config(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Return normalized exact-TMM solution-bank replay settings."""
    block = depth_field_block(cfg)
    raw = nested_get(block, "TRAIN", "SOLUTION_BANK", default={}) or {}
    if not isinstance(raw, dict):
        raise TypeError("DEPTH_FIELD.TRAIN.SOLUTION_BANK must be a mapping.")
    enabled_override = getattr(args, "solution_bank_enabled", None)
    paths_override = getattr(args, "solution_bank", None)
    replay_override = getattr(args, "solution_bank_replay_fraction", None)
    paths = paths_override if paths_override is not None else raw.get("PATHS", raw.get("PATH"))
    if isinstance(paths, (str, Path)):
        paths = [str(paths)]
    config = {
        "enabled": bool(enabled_override if enabled_override is not None else raw.get("ENABLED", False)),
        "paths": [str(path) for path in (paths or [])],
        "replay_fraction": float(replay_override if replay_override is not None else raw.get("REPLAY_FRACTION", 0.0)),
        "gold_fraction": float(raw.get("GOLD_FRACTION", 2.0 / 3.0)),
        "seed": int(raw.get("SEED", cfg.get("SEED", 0) or 0)),
    }
    if not 0.0 <= config["replay_fraction"] <= 1.0:
        raise ValueError("SOLUTION_BANK.REPLAY_FRACTION must be in [0,1].")
    if not 0.0 <= config["gold_fraction"] <= 1.0:
        raise ValueError("SOLUTION_BANK.GOLD_FRACTION must be in [0,1].")
    if config["enabled"] and not config["paths"]:
        raise ValueError("SOLUTION_BANK.ENABLED=true requires SOLUTION_BANK.PATHS.")
    if config["enabled"] and config["replay_fraction"] <= 0.0:
        raise ValueError("SOLUTION_BANK.ENABLED=true requires a positive REPLAY_FRACTION.")
    return config


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
    set_arg_default(args, "hybrid_dilations", nested_get(block, "MODEL", "HYBRID_DILATIONS"))
    set_arg_default(
        args,
        "hybrid_residual_init",
        nested_get(block, "MODEL", "HYBRID_RESIDUAL_INIT", default=1.0e-3),
    )
    if args.n_heads is None:
        args.n_heads = nested_get(block, "MODEL", "N_HEADS", default=8)
    if args.ffn_multiplier is None:
        args.ffn_multiplier = nested_get(block, "MODEL", "FFN_MULTIPLIER", default=4.0)
    if args.spectrum_patch_size is None:
        args.spectrum_patch_size = nested_get(block, "MODEL", "SPECTRUM_PATCH_SIZE", default=8)
    if args.spectrum_patch_stride is None:
        args.spectrum_patch_stride = nested_get(block, "MODEL", "SPECTRUM_PATCH_STRIDE", default=4)
    if args.spectrum_encoder_blocks is None:
        args.spectrum_encoder_blocks = nested_get(block, "MODEL", "SPECTRUM_ENCODER_BLOCKS", default=2)
    if args.spectrum_encoder_heads is None:
        args.spectrum_encoder_heads = nested_get(block, "MODEL", "SPECTRUM_ENCODER_HEADS", default=args.n_heads)
    if args.spectrum_ffn_multiplier is None:
        args.spectrum_ffn_multiplier = nested_get(block, "MODEL", "SPECTRUM_FFN_MULTIPLIER", default=2.0)
    set_arg_config_required(args, "dropout", nested_required(block, "MODEL", "DROPOUT"), "DEPTH_FIELD.MODEL.DROPOUT")
    set_arg_config_required(args, "diffusion_steps", nested_required(block, "MODEL", "DIFFUSION_STEPS"), "DEPTH_FIELD.MODEL.DIFFUSION_STEPS")

    set_arg_config_required(args, "weight_decay", nested_required(block, "TRAIN", "WEIGHT_DECAY"), "DEPTH_FIELD.TRAIN.WEIGHT_DECAY")
    set_arg_config_required(args, "grad_clip", nested_required(block, "TRAIN", "GRAD_CLIP"), "DEPTH_FIELD.TRAIN.GRAD_CLIP")
    set_arg_config_required(args, "amp", nested_required(block, "TRAIN", "AMP"), "DEPTH_FIELD.TRAIN.AMP")
    set_arg_default(args, "amp_dtype", nested_get(block, "TRAIN", "AMP_DTYPE", default="float16"))
    set_arg_default(
        args,
        "max_consecutive_nonfinite_steps",
        nested_get(block, "TRAIN", "MAX_CONSECUTIVE_NONFINITE_STEPS", default=3),
    )
    checkpoint_cfg = cfg.get("CHECKPOINT") if isinstance(cfg.get("CHECKPOINT"), dict) else {}
    set_arg_default(args, "reset_optimizer_on_resume", checkpoint_cfg.get("RESET_OPTIMIZER", False))
    set_arg_default(args, "save_validation_checkpoint", checkpoint_cfg.get("SAVE_VALIDATION_CHECKPOINT", False))
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
    corruption = nested_get(block, "CORRUPTION", default={}) or {}
    set_arg_default(args, "corruption_mode", nested_get(corruption, "MODE", default="iid"))
    set_arg_default(args, "corruption_iid_fraction", nested_get(corruption, "IID_FRACTION", default=1.0))
    set_arg_default(args, "corruption_span_fraction", nested_get(corruption, "SPAN_FRACTION", default=0.0))
    set_arg_default(args, "corruption_layer_fraction", nested_get(corruption, "LAYER_FRACTION", default=0.0))
    set_arg_default(args, "corruption_span_min_bins", nested_get(corruption, "SPAN_MIN_BINS", default=4))
    set_arg_default(args, "corruption_span_max_bins", nested_get(corruption, "SPAN_MAX_BINS", default=64))
    set_arg_default(
        args,
        "corruption_span_scale_with_noise",
        nested_get(corruption, "SPAN_SCALE_WITH_NOISE", default=True),
    )
    set_arg_default(
        args,
        "random_replace_schedule",
        nested_get(corruption, "RANDOM_REPLACE_SCHEDULE", default="constant"),
    )
    set_arg_default(
        args,
        "random_replace_power",
        nested_get(corruption, "RANDOM_REPLACE_POWER", default=1.0),
    )
    set_arg_config_required(
        args,
        "loss_on_corrupted_only",
        nested_required(block, "TRAIN", "LOSS_ON_CORRUPTED_ONLY"),
        "DEPTH_FIELD.TRAIN.LOSS_ON_CORRUPTED_ONLY",
    )
    set_arg_default(
        args,
        "corrupted_loss_weight",
        nested_get(block, "TRAIN", "CORRUPTED_LOSS_WEIGHT", default=1.0),
    )
    set_arg_default(
        args,
        "uncorrupted_loss_weight",
        nested_get(block, "TRAIN", "UNCORRUPTED_LOSS_WEIGHT", default=1.0),
    )
    set_arg_default(
        args,
        "condition_dropout_prob",
        nested_get(block, "TRAIN", "CONDITION_DROPOUT_PROB", default=0.0),
    )
    boundary_loss = nested_get(block, "TRAIN", "BOUNDARY_LOSS", default={}) or {}
    set_arg_default(args, "boundary_loss_enabled", nested_get(boundary_loss, "ENABLED", default=False))
    set_arg_default(args, "boundary_loss_radius_bins", nested_get(boundary_loss, "RADIUS_BINS", default=0))
    set_arg_default(args, "boundary_loss_weight", nested_get(boundary_loss, "WEIGHT", default=1.0))

    set_arg_config_required(args, "eval_mode", nested_required(block, "EVAL", "MODE"), "DEPTH_FIELD.EVAL.MODE")
    set_arg_config_required(args, "eval_mc_samples", nested_required(block, "EVAL", "MC_SAMPLES"), "DEPTH_FIELD.EVAL.MC_SAMPLES")
    set_arg_config_required(args, "eval_sampling_steps", nested_required(block, "EVAL", "SAMPLING_STEPS"), "DEPTH_FIELD.EVAL.SAMPLING_STEPS")
    set_arg_config_required(args, "eval_temperature", nested_required(block, "EVAL", "TEMPERATURE"), "DEPTH_FIELD.EVAL.TEMPERATURE")
    set_arg_config_required(args, "eval_top_k", nested_required(block, "EVAL", "TOP_K"), "DEPTH_FIELD.EVAL.TOP_K")
    set_arg_default(args, "eval_cfg_scale", nested_get(block, "EVAL", "CFG_SCALE", default=1.0))
    set_arg_config_required(args, "eval_deterministic", nested_required(block, "EVAL", "DETERMINISTIC"), "DEPTH_FIELD.EVAL.DETERMINISTIC")
    set_arg_config_required(args, "eval_remask_strategy", nested_required(block, "EVAL", "REMASK_STRATEGY"), "DEPTH_FIELD.EVAL.REMASK_STRATEGY")
    set_arg_config_required(args, "eval_tmm_device", nested_required(block, "EVAL", "TMM_DEVICE"), "DEPTH_FIELD.EVAL.TMM_DEVICE")
    set_arg_config_required(args, "eval_tmm_batch_size", nested_required(block, "EVAL", "TMM_BATCH_SIZE"), "DEPTH_FIELD.EVAL.TMM_BATCH_SIZE")
    set_arg_default(args, "eval_fail_on_nonfinite", nested_get(block, "EVAL", "FAIL_ON_NONFINITE", default=False))
    set_arg_config_required(args, "save_eval_samples", nested_required(block, "EVAL", "SAVE_SAMPLES"), "DEPTH_FIELD.EVAL.SAVE_SAMPLES")
    set_arg_default(args, "eval_samples_dir", nested_get(block, "EVAL", "SAMPLES_DIR"))
    if bool(args.save_eval_samples) and args.eval_samples_dir is None:
        raise ValueError("Missing required setting DEPTH_FIELD.EVAL.SAMPLES_DIR. Set it in the config or pass --eval-samples-dir.")
    set_arg_config_required(args, "eval_record_spectra", nested_required(block, "EVAL", "RECORD_SPECTRA"), "DEPTH_FIELD.EVAL.RECORD_SPECTRA")
    set_arg_config_required(args, "eval_record_all_mc", nested_required(block, "EVAL", "RECORD_ALL_MC"), "DEPTH_FIELD.EVAL.RECORD_ALL_MC")


def autocast_context(device: torch.device, dtype: torch.dtype | None):
    if dtype is not None and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def normalize_amp_dtype(value: str | None) -> str:
    normalized = str(value or "float16").strip().lower()
    aliases = {"fp16": "float16", "float16": "float16", "bf16": "bfloat16", "bfloat16": "bfloat16", "auto": "auto"}
    if normalized not in aliases:
        raise ValueError(f"Unknown AMP dtype {value!r}; expected auto, float16, or bfloat16.")
    return aliases[normalized]


def resolve_amp_dtype(*, enabled: bool, device: torch.device, requested: str | None) -> torch.dtype | None:
    if not enabled or device.type != "cuda":
        return None
    normalized = normalize_amp_dtype(requested)
    if normalized == "float16":
        return torch.float16
    local_bf16 = bool(torch.cuda.is_bf16_supported())
    if normalized == "bfloat16" and not local_bf16:
        raise RuntimeError(f"CUDA device {device} does not support BF16 autocast.")
    if normalized == "auto":
        support = torch.tensor([int(local_bf16)], dtype=torch.int32, device=ddp_collective_device() if ddp_active() else device)
        if ddp_active():
            torch.distributed.all_reduce(support, op=torch.distributed.ReduceOp.MIN)
        return torch.bfloat16 if bool(support.item()) else torch.float16
    return torch.bfloat16


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


def corruption_config_from_args(args: argparse.Namespace) -> optollama.model.DepthFieldCorruptionConfig:
    """Build the shared training and random-remasking policy."""
    return optollama.model.DepthFieldCorruptionConfig(
        mode=str(args.corruption_mode),
        iid_fraction=float(args.corruption_iid_fraction),
        span_fraction=float(args.corruption_span_fraction),
        layer_fraction=float(args.corruption_layer_fraction),
        span_min_bins=int(args.corruption_span_min_bins),
        span_max_bins=int(args.corruption_span_max_bins),
        span_scale_with_noise=bool(args.corruption_span_scale_with_noise),
        random_replace_schedule=str(args.random_replace_schedule),
        random_replace_power=float(args.random_replace_power),
    )


def file_fingerprint(path: Path) -> dict[str, Any]:
    """Return a stable checkpoint provenance record without loading its tensors."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "sha256": digest.hexdigest(),
    }


def distributed_file_fingerprint(path: Path) -> dict[str, Any]:
    """Fingerprint a shared checkpoint only on the rank that saves metadata."""
    if ddp_rank() == 0:
        return file_fingerprint(path)
    stat = path.stat()
    return {"path": str(path), "size": int(stat.st_size), "sha256": None}


def load_frozen_spectral_surrogate(
    config: dict[str, Any],
    *,
    device: torch.device,
    vocab: optollama.data.DepthFieldVocab,
    depth_bins: int,
    spectrum_width: int,
    dz_nm: float,
    wavelength_min: int,
    wavelength_max: int,
    wavelength_step: int,
) -> tuple[torch.nn.Module | None, dict[str, Any] | None]:
    """Load and validate the optional training-only spectrum surrogate."""
    if not bool(config.get("enabled")):
        return None, None
    path = Path(str(config["checkpoint"]))
    if not path.is_file():
        raise FileNotFoundError(f"Spectral surrogate checkpoint does not exist: {path}")
    surrogate, extra = optollama.model.load_depth_field_spectrum_surrogate(path, device=device)
    surrogate_config = surrogate.config
    expected = {
        "num_materials": int(vocab.num_clean_classes),
        "void_id": int(vocab.void_id),
        "depth_bins": int(depth_bins),
        "spectrum_width": int(spectrum_width),
    }
    actual = {key: int(getattr(surrogate_config, key)) for key in expected}
    mismatches = [f"{key}: expected {expected[key]}, got {actual[key]}" for key in expected if expected[key] != actual[key]]
    if not math.isclose(float(surrogate_config.dz_nm), float(dz_nm), rel_tol=0.0, abs_tol=1.0e-9):
        mismatches.append(f"dz_nm: expected {dz_nm}, got {surrogate_config.dz_nm}")
    saved_vocab = ((extra.get("depth_field") or {}).get("vocab") or {}).get("material_names")
    if not isinstance(saved_vocab, list):
        mismatches.append("checkpoint is missing depth_field.vocab.material_names")
    elif list(saved_vocab) != list(vocab.material_names):
        mismatches.append("material_names/order differs from the training depth-field vocabulary")
    saved_grid = extra.get("spectral_grid") or {}
    expected_grid = {
        "wavelength_min": int(wavelength_min),
        "wavelength_max": int(wavelength_max),
        "wavelength_step": int(wavelength_step),
    }
    for key, expected_value in expected_grid.items():
        if saved_grid.get(key) is None:
            mismatches.append(f"checkpoint is missing spectral_grid.{key}")
        elif int(saved_grid[key]) != expected_value:
            mismatches.append(f"spectral_grid.{key}: expected {expected_value}, got {saved_grid[key]}")
    if mismatches:
        raise ValueError("Incompatible spectral surrogate checkpoint: " + "; ".join(mismatches))
    surrogate.eval()
    surrogate.requires_grad_(False)
    metadata = {
        **distributed_file_fingerprint(path),
        "surrogate_config": surrogate_config.to_dict(),
        "loss": extra.get("loss"),
    }
    return surrogate, metadata


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


def synchronized_finite_flags(*values: torch.Tensor) -> tuple[list[bool], list[bool]]:
    """Return local and all-rank finite flags for tensors in a fixed collective order."""
    device = ddp_collective_device() if ddp_active() else values[0].device
    local = torch.stack(
        [torch.isfinite(value.detach()).all().to(device=device, dtype=torch.int32) for value in values]
    )
    global_flags = local.clone()
    if ddp_active():
        torch.distributed.all_reduce(global_flags, op=torch.distributed.ReduceOp.MIN)
    return [bool(value) for value in local.cpu().tolist()], [bool(value) for value in global_flags.cpu().tolist()]


def global_int_sum(value: int, *, device: torch.device) -> int:
    total = torch.tensor([int(value)], dtype=torch.long, device=ddp_collective_device() if ddp_active() else device)
    if ddp_active():
        torch.distributed.all_reduce(total)
    return int(total.item())


def finite_tensor_stats(value: torch.Tensor | None) -> dict[str, Any] | None:
    if value is None:
        return None
    tensor = value.detach()
    finite = torch.isfinite(tensor)
    finite_count = int(finite.sum().item())
    result: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
        "nonfinite": int(tensor.numel() - finite_count),
    }
    if finite_count:
        finite_values = tensor[finite].float()
        result.update(
            minimum=float(finite_values.min().item()),
            maximum=float(finite_values.max().item()),
            abs_maximum=float(finite_values.abs().max().item()),
        )
    return result


def first_nonfinite_model_tensor(model: torch.nn.Module) -> str | None:
    for name, tensor in unwrap_model(model).state_dict().items():
        if torch.is_floating_point(tensor) and not bool(torch.isfinite(tensor).all().item()):
            return name
    return None


def first_nonfinite_optimizer_tensor(optimizer: torch.optim.Optimizer) -> str | None:
    for parameter_index, state in enumerate(optimizer.state.values()):
        for name, value in state.items():
            if torch.is_tensor(value) and torch.is_floating_point(value) and not bool(torch.isfinite(value).all().item()):
                return f"parameter_{parameter_index}.{name}"
    return None


def save_nonfinite_diagnostic(
    *,
    out_dir: str | Path,
    epoch: int,
    batch_index: int,
    global_samples_before: int,
    reason: str,
    spectra: torch.Tensor,
    fields: torch.Tensor,
    out: dict[str, torch.Tensor] | None,
    loss: torch.Tensor | None,
    grad_norm: torch.Tensor | None,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    batch_indices: torch.Tensor | None,
    inspect_state: bool = True,
) -> Path:
    rank = ddp_rank()
    diagnostic_dir = Path(out_dir) / "nonfinite_diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    path = diagnostic_dir / f"epoch_{epoch + 1:04d}_batch_{batch_index:06d}_rank_{rank:04d}.json"
    payload = {
        "reason": str(reason),
        "rank": int(rank),
        "epoch": int(epoch),
        "epoch_1based": int(epoch + 1),
        "batch_index": int(batch_index),
        "global_samples_before": int(global_samples_before),
        "amp_scale": float(scaler.get_scale()) if scaler.is_enabled() else None,
        "spectra": finite_tensor_stats(spectra),
        "fields": finite_tensor_stats(fields),
        "timesteps": finite_tensor_stats(out.get("timesteps") if out is not None else None),
        "logits": finite_tensor_stats(out.get("logits") if out is not None else None),
        "loss": finite_tensor_stats(loss),
        "grad_norm": finite_tensor_stats(grad_norm),
        "first_nonfinite_model_tensor": first_nonfinite_model_tensor(model) if inspect_state else None,
        "first_nonfinite_optimizer_tensor": first_nonfinite_optimizer_tensor(optimizer) if inspect_state else None,
        "dataset_indices": batch_indices.detach().cpu().reshape(-1).tolist() if batch_indices is not None else None,
    }
    optollama.utils.save_as_json(str(path), payload)
    return path


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
    corruption_config: optollama.model.DepthFieldCorruptionConfig | dict | None = None,
    loss_on_corrupted_only: bool = False,
    corrupted_loss_weight: float = 1.0,
    uncorrupted_loss_weight: float = 1.0,
    condition_dropout_prob: float = 0.0,
    boundary_loss_enabled: bool = False,
    boundary_loss_radius_bins: int = 0,
    boundary_loss_weight: float = 1.0,
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
        corruption_config=corruption_config,
    )
    conditioned_spectra, condition_dropped = optollama.model.drop_spectrum_condition(
        spectra,
        condition_dropout_prob,
    )
    logits = model(conditioned_spectra, noised_fields, timesteps)

    weights = torch.ones(int(core.num_materials), device=clean_fields.device, dtype=logits.dtype)
    if 0 <= int(void_id) < int(core.num_materials):
        weights[int(void_id)] = float(void_loss_weight)
    loss_per_bin = torch.nn.functional.cross_entropy(
        logits.reshape(-1, int(core.num_materials)),
        clean_fields.long().reshape(-1),
        weight=weights,
        reduction="none",
    ).view_as(clean_fields)

    boundary = optollama.model.depth_field_boundary_mask(clean_fields, boundary_loss_radius_bins)
    boundary_weight = float(boundary_loss_weight)
    if not math.isfinite(boundary_weight) or boundary_weight <= 0.0:
        raise ValueError(f"boundary_loss_weight must be finite and positive, got {boundary_loss_weight}")
    if boundary_loss_enabled:
        loss_per_bin = loss_per_bin * torch.where(
            boundary,
            loss_per_bin.new_full((), boundary_weight),
            loss_per_bin.new_ones(()),
        )

    loss = optollama.model.weighted_depth_field_loss(
        loss_per_bin,
        corrupted,
        corrupted_loss_weight=corrupted_loss_weight,
        uncorrupted_loss_weight=uncorrupted_loss_weight,
        loss_on_corrupted_only=loss_on_corrupted_only,
    )

    return {
        "loss": loss,
        "logits": logits,
        "timesteps": timesteps,
        "noised_fields": noised_fields,
        "corrupted": corrupted,
        "condition_dropped": condition_dropped,
        "boundary": boundary,
    }


def spectral_auxiliary_loss(
    *,
    output: dict[str, torch.Tensor],
    target_spectra: torch.Tensor,
    core_model: torch.nn.Module,
    surrogate: torch.nn.Module | None,
    config: dict[str, Any],
    global_samples_seen: int,
    batch_index: int,
) -> dict[str, Any]:
    """Return a sparse, ramped spectrum-consistency loss for eligible samples."""
    zero = output["logits"].new_zeros(())
    result: dict[str, Any] = {
        "loss": zero,
        "raw_loss": zero.detach(),
        "level_loss": zero.detach(),
        "derivative_loss": zero.detach(),
        "weight": 0.0,
        "samples": 0,
        "applied": False,
    }
    if surrogate is None or not bool(config.get("enabled")):
        return result
    if int(batch_index) % int(config["every_n_steps"]) != 0:
        return result
    start_after = int(config["start_after_samples"])
    progress_samples = max(0, int(global_samples_seen) - start_after)
    if int(global_samples_seen) < start_after:
        return result
    ramp_samples = int(config["weight_ramp_samples"])
    ramp = 1.0 if ramp_samples <= 0 else min(1.0, float(progress_samples) / float(ramp_samples))
    effective_weight = float(config["weight"]) * ramp
    if effective_weight <= 0.0:
        return result

    noise_probability = core_model.noise_probability(output["timesteps"]).detach()
    eligible = noise_probability <= float(config["max_noise_probability"])
    if bool(config["skip_dropped_conditions"]):
        eligible &= ~output["condition_dropped"].detach().bool()
    indices = torch.nonzero(eligible, as_tuple=False).flatten()[: int(config["max_samples_per_rank"])]
    if indices.numel() == 0:
        return result

    probabilities = optollama.model.straight_through_material_probabilities(
        output["logits"][indices],
        temperature=float(config["straight_through_temperature"]),
    )
    predicted_spectra = surrogate(probabilities)
    parts = optollama.model.depth_field_spectrum_loss(
        predicted_spectra,
        target_spectra[indices],
        channels=tuple(int(index) for index in config["channels"]),
        derivative_weight=float(config["derivative_weight"]),
        huber_delta=float(config["huber_delta"]),
    )
    result.update(
        loss=parts["loss"] * effective_weight,
        raw_loss=parts["loss"].detach(),
        level_loss=parts["level_loss"].detach(),
        derivative_loss=parts["derivative_loss"].detach(),
        weight=effective_weight,
        samples=int(indices.numel()),
        applied=True,
    )
    return result


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
    stability_counts: torch.Tensor | None = None,
    spectral_stats: torch.Tensor | None = None,
    replay_samples: int = 0,
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
            float(replay_samples),
        ],
        dtype=torch.float64,
        device=device,
    )
    all_reduce_sum(reduced_counts)
    all_reduce_sum(totals)
    metrics = counts_to_metrics(
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
    replay_total = int(round(float(totals[8].item())))
    metrics["solution_bank_replay_samples"] = replay_total
    metrics["solution_bank_replay_fraction"] = replay_total / max(int(totals[4].item()), 1)
    if stability_counts is not None:
        reduced_stability = stability_counts.detach().clone().to(device=device, dtype=torch.float64)
        all_reduce_sum(reduced_stability)
        divisor = float(ddp_world_size())
        averaged = reduced_stability / divisor
        metrics.update(
            nonfinite_forward_steps=int(round(float(averaged[0].item()))),
            nonfinite_gradient_steps=int(round(float(averaged[1].item()))),
            amp_skipped_steps=int(round(float(averaged[2].item()))),
            optimizer_steps=int(round(float(averaged[3].item()))),
        )
    if spectral_stats is not None:
        reduced_spectral = spectral_stats.detach().clone().to(device=device, dtype=torch.float64)
        all_reduce_sum(reduced_spectral)
        spectral_samples = max(float(reduced_spectral[4].item()), 1.0)
        metrics.update(
            spectral_aux_loss=float(reduced_spectral[0].item() / spectral_samples),
            spectral_aux_weighted_loss=float(reduced_spectral[1].item() / spectral_samples),
            spectral_aux_level_loss=float(reduced_spectral[2].item() / spectral_samples),
            spectral_aux_derivative_loss=float(reduced_spectral[3].item() / spectral_samples),
            spectral_aux_samples=int(round(float(reduced_spectral[4].item()))),
            spectral_aux_batches=int(round(float(reduced_spectral[5].item()))),
            spectral_aux_weight=float(reduced_spectral[6].item() / spectral_samples),
        )
    return metrics


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
    local_nonfinite_sampling_positions = 0
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
            guidance_scale=float(args.eval_cfg_scale),
            remask_strategy=str(args.eval_remask_strategy),
            corruption_config=args.corruption_config,
        )
        repaired_positions = int(getattr(sample_model, "_last_sampling_nonfinite_positions", 0))
        local_nonfinite_sampling_positions += repaired_positions
        if bool(args.eval_fail_on_nonfinite):
            repaired_global = global_int_sum(repaired_positions, device=device)
            if repaired_global > 0:
                raise FloatingPointError(
                    f"Validation aborted after repairing {repaired_global} non-finite sampling positions "
                    f"at epoch {epoch + 1}, trigger={trigger}."
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
    nonfinite_sampling_positions = global_int_sum(local_nonfinite_sampling_positions, device=device)

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
        "cfg_scale": float(args.eval_cfg_scale),
        "remask_strategy": str(args.eval_remask_strategy),
        "corruption": args.corruption_config.to_dict(),
        "tmm_batch_size": int(args.eval_tmm_batch_size),
        "nonfinite_sampling_positions": int(nonfinite_sampling_positions),
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
                    "cfg_scale": float(args.eval_cfg_scale),
                    "remask_strategy": str(args.eval_remask_strategy),
                    "corruption": args.corruption_config.to_dict(),
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
    amp_dtype: torch.dtype | None,
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
    spectral_aux_model: torch.nn.Module | None = None,
    solution_bank: optollama.data.DepthFieldSolutionBankReplay | None = None,
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
    stability_counts = torch.zeros(4, dtype=torch.float64, device=device)
    spectral_stats = torch.zeros(7, dtype=torch.float64, device=device)
    replay_samples = 0
    consecutive_nonfinite_steps = 0
    desc = f"Epoch {epoch + 1}/{epochs} {'train' if train else 'val'}"
    show_progress = ddp_rank() == 0
    pbar = tqdm.tqdm(loader, desc=desc, leave=True, disable=not show_progress)
    validation_interval = int(validate_every_samples or 0)
    next_validation_sample = validation_interval if train and validation_callback is not None and validation_interval > 0 else None
    base_lr = float(optimizer_base_lr if optimizer_base_lr is not None else (optimizer.param_groups[0]["lr"] if optimizer is not None else 0.0))
    current_lr = base_lr

    for batch_index, batch in enumerate(pbar):
        raw_spectra_cpu, raw_stacks_cpu = batch[0], batch[1]
        batch_indices_cpu = batch[2] if len(batch) > 2 and torch.is_tensor(batch[2]) else None
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
                        stability_counts=stability_counts,
                        spectral_stats=spectral_stats,
                        replay_samples=replay_samples,
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
            if batch_indices_cpu is not None:
                batch_indices_cpu = batch_indices_cpu[keep]

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
        if train and count_batch_metrics and solution_bank is not None:
            spectra_cpu, fields_cpu, replay_count = solution_bank.mix_batch(
                spectra_cpu,
                fields_cpu,
                epoch=epoch,
                batch_index=batch_index,
                rank=ddp_rank(),
            )
            replay_samples += int(replay_count)
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
            with autocast_context(device, amp_dtype):
                out = depth_field_training_loss(
                    model,
                    spectra,
                    fields,
                    void_id=vocab.void_id,
                    void_loss_weight=args.void_loss_weight,
                    random_replace_prob=args.random_replace_prob,
                    corruption_config=args.corruption_config,
                    loss_on_corrupted_only=args.loss_on_corrupted_only,
                    corrupted_loss_weight=args.corrupted_loss_weight,
                    uncorrupted_loss_weight=args.uncorrupted_loss_weight,
                    condition_dropout_prob=float(getattr(args, "condition_dropout_prob", 0.0)),
                    boundary_loss_enabled=bool(getattr(args, "boundary_loss_enabled", False)),
                    boundary_loss_radius_bins=int(getattr(args, "boundary_loss_radius_bins", 0)),
                    boundary_loss_weight=float(getattr(args, "boundary_loss_weight", 1.0)),
                    timestep_schedule=timestep_schedule,
                    global_samples_seen=global_samples_before,
                )
                spectral_aux = spectral_auxiliary_loss(
                    output=out,
                    target_spectra=spectra,
                    core_model=unwrap_model(model),
                    surrogate=spectral_aux_model,
                    config=getattr(args, "spectral_aux_config", {"enabled": False}),
                    global_samples_seen=global_samples_before,
                    batch_index=batch_index,
                )
                denoise_loss = out["loss"]
                loss = denoise_loss + spectral_aux["loss"]
            out["denoise_loss"] = denoise_loss.detach()
            out["spectral_aux_loss"] = spectral_aux["raw_loss"]
            out["spectral_aux_weighted_loss"] = spectral_aux["loss"].detach()
            if not count_batch_metrics:
                loss = loss * 0.0
            local_finite, global_finite = synchronized_finite_flags(spectra, out["logits"], loss)
            if not all(global_finite):
                stability_counts[0] += 1
                consecutive_nonfinite_steps += 1
                optimizer.zero_grad(set_to_none=True)
                if not all(local_finite) or ddp_rank() == 0:
                    reason_names = ["spectra", "logits", "loss"]
                    local_reasons = [name for name, is_finite in zip(reason_names, local_finite) if not is_finite]
                    reason = "nonfinite_" + "_".join(local_reasons or ["remote_rank"])
                    diagnostic_path = save_nonfinite_diagnostic(
                        out_dir=args.out_dir,
                        epoch=epoch,
                        batch_index=batch_index,
                        global_samples_before=global_samples_before,
                        reason=reason,
                        spectra=spectra,
                        fields=fields,
                        out=out,
                        loss=loss,
                        grad_norm=None,
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        batch_indices=batch_indices_cpu,
                        inspect_state=ddp_rank() == 0,
                    )
                    if show_progress:
                        tqdm.tqdm.write(f"Skipped non-finite forward step; diagnostic -> {diagnostic_path}")
                limit = int(args.max_consecutive_nonfinite_steps)
                if limit <= 0 or consecutive_nonfinite_steps >= limit:
                    raise FloatingPointError(
                        f"Aborting after {consecutive_nonfinite_steps} consecutive non-finite forward steps "
                        f"at epoch {epoch + 1}, batch {batch_index}."
                    )
                continue
            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(args.grad_clip),
                    error_if_nonfinite=False,
                )
            else:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=False)
            local_grad_finite, global_grad_finite = synchronized_finite_flags(grad_norm)
            if not all(global_grad_finite):
                stability_counts[1] += 1
                consecutive_nonfinite_steps += 1
                if not all(local_grad_finite) or ddp_rank() == 0:
                    diagnostic_path = save_nonfinite_diagnostic(
                        out_dir=args.out_dir,
                        epoch=epoch,
                        batch_index=batch_index,
                        global_samples_before=global_samples_before,
                        reason="nonfinite_gradient_norm",
                        spectra=spectra,
                        fields=fields,
                        out=out,
                        loss=loss,
                        grad_norm=grad_norm,
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        batch_indices=batch_indices_cpu,
                        inspect_state=ddp_rank() == 0,
                    )
                    if show_progress:
                        tqdm.tqdm.write(f"Skipped non-finite gradient step; diagnostic -> {diagnostic_path}")
                scale_before = float(scaler.get_scale()) if scaler.is_enabled() else None
                if scaler.is_enabled():
                    scaler.update(new_scale=max(float(scale_before or 1.0) * 0.5, 1.0))
                    stability_counts[2] += 1
                optimizer.zero_grad(set_to_none=True)
                limit = int(args.max_consecutive_nonfinite_steps)
                if limit <= 0 or consecutive_nonfinite_steps >= limit:
                    raise FloatingPointError(
                        f"Aborting after {consecutive_nonfinite_steps} consecutive non-finite gradient steps "
                        f"at epoch {epoch + 1}, batch {batch_index}."
                    )
                continue
            scale_before = float(scaler.get_scale()) if scaler.is_enabled() else None
            scaler.step(optimizer)
            scaler.update()
            amp_skipped = bool(scale_before is not None and float(scaler.get_scale()) < scale_before)
            if amp_skipped:
                stability_counts[2] += 1
            else:
                stability_counts[3] += 1
                consecutive_nonfinite_steps = 0
            if ema is not None and not amp_skipped:
                ema.update(model)
            if count_batch_metrics and bool(spectral_aux["applied"]) and not amp_skipped:
                spectral_samples = int(spectral_aux["samples"])
                spectral_stats += torch.tensor(
                    [
                        float(spectral_aux["raw_loss"].item()) * spectral_samples,
                        float(spectral_aux["loss"].detach().item()) * spectral_samples,
                        float(spectral_aux["level_loss"].item()) * spectral_samples,
                        float(spectral_aux["derivative_loss"].item()) * spectral_samples,
                        float(spectral_samples),
                        1.0,
                        float(spectral_aux["weight"]) * spectral_samples,
                    ],
                    dtype=torch.float64,
                    device=device,
                )
        else:
            with torch.no_grad(), autocast_context(device, amp_dtype):
                out = depth_field_training_loss(
                    model,
                    spectra,
                    fields,
                    void_id=vocab.void_id,
                    void_loss_weight=args.void_loss_weight,
                    random_replace_prob=args.random_replace_prob,
                    corruption_config=args.corruption_config,
                    loss_on_corrupted_only=args.loss_on_corrupted_only,
                    corrupted_loss_weight=args.corrupted_loss_weight,
                    uncorrupted_loss_weight=args.uncorrupted_loss_weight,
                    condition_dropout_prob=0.0,
                    boundary_loss_enabled=bool(getattr(args, "boundary_loss_enabled", False)),
                    boundary_loss_radius_bins=int(getattr(args, "boundary_loss_radius_bins", 0)),
                    boundary_loss_weight=float(getattr(args, "boundary_loss_weight", 1.0)),
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
            stability_counts=stability_counts,
            spectral_stats=spectral_stats,
            replay_samples=replay_samples,
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
                nf=f"{metrics.get('nonfinite_forward_steps', 0) + metrics.get('nonfinite_gradient_steps', 0)}",
                amp_skip=f"{metrics.get('amp_skipped_steps', 0)}",
                spec=f"{metrics.get('spectral_aux_loss', 0.0):.4f}",
                bank=f"{metrics.get('solution_bank_replay_fraction', 0.0) * 100.0:.1f}%",
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
        stability_counts=stability_counts,
        spectral_stats=spectral_stats,
        replay_samples=replay_samples,
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
            "eval_cfg_scale": float(args.eval_cfg_scale),
            "eval_remask_strategy": str(args.eval_remask_strategy),
            "corruption": args.corruption_config.to_dict(),
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
        "training_objective": {
            "condition_dropout_prob": float(args.condition_dropout_prob),
            "boundary_loss_enabled": bool(args.boundary_loss_enabled),
            "boundary_loss_radius_bins": int(args.boundary_loss_radius_bins),
            "boundary_loss_weight": float(args.boundary_loss_weight),
            "loss_on_corrupted_only": bool(args.loss_on_corrupted_only),
            "corrupted_loss_weight": float(args.corrupted_loss_weight),
            "uncorrupted_loss_weight": float(args.uncorrupted_loss_weight),
            "spectral_aux": {
                **dict(getattr(args, "spectral_aux_config", {})),
                "surrogate": getattr(args, "spectral_aux_metadata", None),
            },
            "solution_bank": getattr(args, "solution_bank_metadata", None),
        },
        "initialization": getattr(args, "init_from_provenance", None),
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
    scaler: torch.amp.GradScaler | None,
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
            scaler=scaler,
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
    args.corruption_config = corruption_config_from_args(args)
    args.spectral_aux_config = spectral_aux_config(cfg, args)
    args.spectral_aux_metadata = None
    args.solution_bank_config = solution_bank_config(cfg, args)
    args.solution_bank_metadata = None
    args.init_from_provenance = None

    if args.seed is not None:
        cfg["SEED"] = int(args.seed)
    device, local_rank, rank, world_size = optollama.utils.setup_run(cfg, make_dirs=False)
    if args.device is not None and world_size == 1:
        device = resolve_device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
    ddp = optollama.utils.is_ddp()
    amp_enabled = bool(args.amp and device.type == "cuda")
    amp_dtype = resolve_amp_dtype(enabled=amp_enabled, device=device, requested=args.amp_dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

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
        hybrid_dilations=tuple(int(value) for value in (args.hybrid_dilations or ())),
        hybrid_residual_init=float(args.hybrid_residual_init),
        timesteps=int(args.diffusion_steps),
        dropout=float(args.dropout),
        spectrum_patch_size=int(args.spectrum_patch_size),
        spectrum_patch_stride=int(args.spectrum_patch_stride),
        spectrum_encoder_blocks=int(args.spectrum_encoder_blocks),
        spectrum_encoder_heads=int(args.spectrum_encoder_heads),
        spectrum_ffn_multiplier=float(args.spectrum_ffn_multiplier),
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
    spectral_aux_model, args.spectral_aux_metadata = load_frozen_spectral_surrogate(
        args.spectral_aux_config,
        device=device,
        vocab=vocab,
        depth_bins=depth_bins,
        spectrum_width=int(model_config.spectrum_shape[-1]),
        dz_nm=float(args.dz_nm),
        wavelength_min=int(cfg["WAVELENGTH_MIN"]),
        wavelength_max=int(cfg["WAVELENGTH_MAX"]),
        wavelength_step=int(cfg["WAVELENGTH_STEPS"]),
    )
    solution_bank = None
    if bool(args.solution_bank_config["enabled"]):
        solution_bank = optollama.data.DepthFieldSolutionBankReplay(
            args.solution_bank_config["paths"],
            replay_fraction=float(args.solution_bank_config["replay_fraction"]),
            gold_fraction=float(args.solution_bank_config["gold_fraction"]),
            seed=int(args.solution_bank_config["seed"]),
            expected_spectrum_shape=model_config.spectrum_shape,
            expected_depth_bins=depth_bins,
            expected_material_names=vocab.material_names,
            expected_dz_nm=float(args.dz_nm),
        )
        args.solution_bank_metadata = solution_bank.summary()

    epochs = int(args.epochs if args.epochs is not None else cfg_required(cfg, "EPOCHS", "--epochs"))
    out_dir = Path(args.out_dir)
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    if ddp:
        torch.distributed.barrier()
    best_path = out_dir / "depth-field-best.pt"
    last_path = out_dir / "depth-field-last.pt"
    recovery_path = out_dir / "depth-field-recovery.pt"
    ema_best_path = out_dir / "depth-field-best-ema.pt"
    ema_last_path = out_dir / "depth-field-last-ema.pt"
    history_path = out_dir / "depth-field-history.json"
    eval_samples_dir = Path(args.eval_samples_dir) if args.eval_samples_dir is not None else None

    start_epoch = 0
    global_sample_offset_adjustment = 0
    history: list[dict] = []
    resume_path, resume_source, resume_required = resolve_resume_checkpoint(cfg, args, last_path)
    init_path, init_source = resolve_init_checkpoint(cfg, args)
    if resume_path is not None and init_path is not None:
        if args.init_from is not None:
            raise ValueError(f"Use either resume ({resume_source}) or --init-from, not both.")
        if resume_path.is_file() or resume_required:
            if rank == 0:
                print(f"Resume via {resume_source} supersedes initialization via {init_source}.")
            init_path, init_source = None, None
        else:
            if rank == 0:
                print(f"No checkpoint found for {resume_source}; using initialization via {init_source}.")
            resume_path, resume_source = None, None
    if init_path is not None:
        if not init_path.is_file():
            raise FileNotFoundError(f"{init_source} checkpoint does not exist: {init_path}")
        _, init_blob = optollama.utils.load_checkpoint(
            str(init_path),
            model,
            optimizer=None,
            scaler=None,
            map_location="cpu",
            strict=True,
        )
        nonfinite_model_tensor = first_nonfinite_model_tensor(model)
        checkpoint_finite = torch.tensor(
            [int(nonfinite_model_tensor is None)],
            dtype=torch.int32,
            device=ddp_collective_device() if ddp_active() else device,
        )
        if ddp_active():
            torch.distributed.all_reduce(checkpoint_finite, op=torch.distributed.ReduceOp.MIN)
        if not bool(checkpoint_finite.item()):
            raise FloatingPointError(
                f"Refusing to initialize from non-finite checkpoint {init_path}; model_tensor={nonfinite_model_tensor}."
            )
        source_extra = init_blob.get("extra") or {}
        source_history = list(source_extra.get("history") or [])
        source_samples = None
        if source_history:
            source_samples = (source_history[-1].get("train") or {}).get("global_samples_seen")
        args.init_from_provenance = {
            **distributed_file_fingerprint(init_path),
            "source": str(init_source),
            "source_checkpoint_weights": source_extra.get("checkpoint_weights"),
            "source_global_samples": source_samples,
            "source_config_path": source_extra.get("config_path"),
        }
        if ema is not None:
            ema.reset(model)
        if rank == 0:
            print(f"Initialized new depth-field run from {init_path} ({init_source}); optimizer/history/schedules start fresh.")
    if resume_path is not None:
        if resume_path.exists():
            reset_optimizer = bool(args.reset_optimizer_on_resume)
            start_epoch_loaded, blob = optollama.utils.load_checkpoint(
                str(resume_path),
                model,
                optimizer=None if reset_optimizer else optimizer,
                scaler=None if reset_optimizer else scaler,
                map_location="cpu",
            )
            start_epoch = int(start_epoch_loaded or 0)
            history = list(((blob.get("extra") or {}).get("history") or []))
            if history:
                resume_global_samples = (history[-1].get("train") or {}).get("global_samples_seen")
                if isinstance(resume_global_samples, (int, float)) and math.isfinite(float(resume_global_samples)):
                    global_sample_offset_adjustment = int(resume_global_samples) - start_epoch * int(train_subset)
            nonfinite_model_tensor = first_nonfinite_model_tensor(model)
            nonfinite_optimizer_tensor = None if reset_optimizer else first_nonfinite_optimizer_tensor(optimizer)
            checkpoint_finite = torch.tensor(
                [int(nonfinite_model_tensor is None and nonfinite_optimizer_tensor is None)],
                dtype=torch.int32,
                device=ddp_collective_device() if ddp_active() else device,
            )
            if ddp_active():
                torch.distributed.all_reduce(checkpoint_finite, op=torch.distributed.ReduceOp.MIN)
            if not bool(checkpoint_finite.item()):
                raise FloatingPointError(
                    f"Refusing to resume non-finite checkpoint {resume_path}; "
                    f"model_tensor={nonfinite_model_tensor}, optimizer_tensor={nonfinite_optimizer_tensor}."
                )
            if ema is not None:
                ema_state = ((blob.get("extra") or {}).get("ema_state") or {})
                if ema_state:
                    ema.load_state_dict(ema_state, model)
                else:
                    ema.reset(model)
            if rank == 0:
                print(f"Resumed depth-field checkpoint {resume_path} at epoch {start_epoch} ({resume_source}).")
                if reset_optimizer:
                    print("Reset optimizer and AMP scaler state while preserving checkpoint weights/history/epoch.")
                elif scaler.is_enabled() and not isinstance(blob.get("scaler_state"), dict):
                    print("Checkpoint has no AMP scaler state; using a fresh GradScaler state.")
                if global_sample_offset_adjustment:
                    resumed_at = start_epoch * int(train_subset) + global_sample_offset_adjustment
                    print(f"Continuing schedules from checkpoint sample position {resumed_at}.")
        elif resume_required:
            raise FileNotFoundError(f"{resume_source} checkpoint does not exist: {resume_path}")
        elif rank == 0:
            print(f"Depth-field resume is enabled via {resume_source}, but {resume_path} does not exist; starting fresh.")

    if rank == 0 and spectral_aux_model is not None:
        spectral_cfg = args.spectral_aux_config
        print(
            "Spectral auxiliary: "
            f"checkpoint={spectral_cfg['checkpoint']}, weight={spectral_cfg['weight']:g}, "
            f"every={spectral_cfg['every_n_steps']} steps, max_local={spectral_cfg['max_samples_per_rank']}, "
            f"noise<={spectral_cfg['max_noise_probability']:g}, channels={spectral_cfg['channel_names']}."
        )
    if rank == 0 and solution_bank is not None:
        bank_summary = solution_bank.summary()
        print(
            "Solution-bank replay: "
            f"samples={bank_summary['samples']}, anchors={bank_summary['anchors']}, "
            f"topologies={bank_summary['topologies']}, gold={bank_summary['gold_samples']}, "
            f"silver={bank_summary['silver_samples']}, replay={bank_summary['replay_fraction']:.1%}."
        )

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
                    amp_dtype=amp_dtype,
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
                scaler=scaler,
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
                    scaler=scaler,
                    epoch=epoch,
                    history=history,
                    extra=extra,
                    ema=ema,
                    use_ema_weights=True,
                )
            print(f"Saved best checkpoint -> {best_path} ({val_metrics.get('score_name', 'score')}={best_score:.6f}, trigger={trigger})")

        repaired_positions = int(val_metrics.get("nonfinite_sampling_positions", 0))
        if bool(args.save_validation_checkpoint) and math.isfinite(score) and repaired_positions == 0:
            save_depth_checkpoint(
                path=recovery_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                history=history,
                extra=extra,
                ema=ema,
                use_ema_weights=False,
            )
            print(f"Saved rolling recovery checkpoint -> {recovery_path} (trigger={trigger})")

        optollama.utils.save_as_json(str(history_path), history)

    if rank == 0:
        model_desc = f"type={model_config.model_type}"
        transformer_models = {
            "attention",
            "optollama_depth",
            "optollama_depth_windowed",
            "optollama_depth_windowed_v2",
            "optollama_depth_windowed_v3",
            "optollama_depth_hybrid",
        }
        if model_config.model_type in transformer_models:
            model_desc += f", heads={model_config.n_heads}, ffn={model_config.ffn_multiplier:g}"
            if model_config.model_type != "attention":
                model_desc += ", spectrum_cross_attn=true"
            if model_config.model_type in {
                "optollama_depth_windowed",
                "optollama_depth_windowed_v2",
                "optollama_depth_windowed_v3",
                "optollama_depth_hybrid",
            }:
                model_desc += (
                    f", spectrum_patch={model_config.spectrum_patch_size}"
                    f"/{model_config.spectrum_patch_stride}, spectrum_blocks={model_config.spectrum_encoder_blocks}, "
                    f"spectrum_heads={model_config.spectrum_encoder_heads}"
                )
            if model_config.model_type == "optollama_depth_windowed_v2":
                model_desc += ", pooled_spectrum_adaln=true, final_norm=true"
            elif model_config.model_type == "optollama_depth_windowed_v3":
                model_desc += ", pooled_spectrum_token=true, adaln_condition=time_only, final_norm=true"
            elif model_config.model_type == "optollama_depth_hybrid":
                receptive_field_nm = (
                    1 + 2 * (model_config.kernel_size - 1) * sum(model_config.hybrid_dilations)
                ) * float(args.dz_nm)
                model_desc += (
                    ", pooled_spectrum_token=true, adaln_condition=time_only, final_norm=true, "
                    f"interleaved_conv={model_config.conv_type}, dilations={list(model_config.hybrid_dilations)}, "
                    f"conv_rf={receptive_field_nm:g}nm"
                )
        else:
            model_desc += f", conv={model_config.conv_type}, kernel={model_config.kernel_size}"
        print(
            "Depth-field diffusion: "
            f"materials={vocab.num_clean_classes - 1}+void, bins={depth_bins}, dz={args.dz_nm:g}nm, "
            f"max={args.max_thickness_nm:g}nm, device={device}, amp={amp_enabled}, amp_dtype={amp_dtype}, "
            f"ddp={ddp}, world={world_size}, {model_desc}, "
            f"corruption={args.corruption_config.mode}, "
            f"random_replace={args.random_replace_prob:g}/{args.corruption_config.random_replace_schedule}, "
            f"loss_weights={args.corrupted_loss_weight:g}:"
            f"{0.0 if args.loss_on_corrupted_only else args.uncorrupted_loss_weight:g}, "
            f"condition_dropout={args.condition_dropout_prob:g}, "
            f"boundary_loss={args.boundary_loss_enabled}/{args.boundary_loss_radius_bins}/{args.boundary_loss_weight:g}, "
            f"eval_mc={args.eval_mc_samples}, eval_steps={args.eval_sampling_steps}, "
            f"eval_temp={args.eval_temperature:g}, eval_top_k={args.eval_top_k}, eval_cfg={args.eval_cfg_scale:g}, "
            f"eval_remask={args.eval_remask_strategy}"
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
            amp_dtype=amp_dtype,
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
            global_sample_offset=global_sample_offset_adjustment + int(epoch) * int(train_subset),
            optimizer_base_lr=optimizer_base_lr,
            lr_schedule=lr_schedule,
            timestep_schedule=timestep_schedule,
            ema=ema,
            spectral_aux_model=spectral_aux_model,
            solution_bank=solution_bank,
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
                save_depth_checkpoint(
                    path=last_path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    history=history,
                    extra=extra,
                )
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
                        scaler=scaler,
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
                    save_depth_checkpoint(
                        path=best_path,
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        epoch=epoch,
                        history=history,
                        extra=extra,
                    )
                    print(f"Saved best checkpoint -> {best_path} (train_loss={best_score:.6f})")

            optollama.utils.save_as_json(str(history_path), history)


if __name__ == "__main__":
    optollama.utils.stop_ddp()
    try:
        main()
    finally:
        optollama.utils.stop_ddp()
