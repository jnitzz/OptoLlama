#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
import tqdm

import optollama.data
import optollama.evaluation.simulation
import optollama.model
import optollama.plotting
import optollama.utils

# ruff: noqa: D103


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample and TMM-score the experimental depth-field diffusion model.")
    parser.add_argument("--config", type=str, default="configs/optollama.yaml", help="Project config YAML.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Depth-field .pt checkpoint.")
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Output JSON. Required unless config SAMPLES_PATH is set.",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to score.")
    parser.add_argument("--target", type=str, default=None, help="Optional target spectrum CSV/JSON. Defaults to config TARGET when set.")
    parser.add_argument("--target-samples", type=int, default=None, help="Repeated target spectra to evaluate. Required unless config N_TARGETS is set.")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum split spectra to process. Required unless config NUM_SAMPLES_TEST is set.")
    parser.add_argument("--batch-size", type=int, default=None, help="Target spectra per model batch. Required unless config TEST_BATCH_SIZE is set.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers. Required unless config NUM_WORKERS is set.")
    parser.add_argument("--sharded-loading", action="store_true", help="Force streaming shard loading.")
    parser.add_argument("--eager-loading", action="store_true", help="Force eager in-memory loading.")
    parser.add_argument("--device", type=str, default=None, help='Model device, e.g. "cuda", "cuda:0", or "cpu".')
    parser.add_argument("--tmm-device", type=str, default=None, help='TMM device. "auto" uses model device.')

    parser.add_argument("--mc-samples", type=int, default=None, help="Candidate fields per target spectrum.")
    parser.add_argument(
        "--mc-batch-size",
        type=int,
        default=None,
        help="Maximum MC candidates per target to sample/score at once. Set this below --mc-samples to reduce VRAM.",
    )
    parser.add_argument("--sampling-steps", type=int, default=None, help="Denoising steps. Required unless DEPTH_FIELD.EVAL.SAMPLING_STEPS is set.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature. <=0 uses argmax.")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k material sampling filter. 0 disables.")
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=None,
        help="Classifier-free guidance scale. 1 uses one conditional forward pass; other positive values use CFG.",
    )
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None, help="Use argmax instead of sampling.")
    parser.add_argument(
        "--remask-strategy",
        type=str,
        default=None,
        choices=["confidence", "random"],
        help="Denoising remask strategy: confidence reopens least-confident bins, random uses Bernoulli remasking.",
    )
    parser.add_argument("--corruption-mode", choices=["iid", "hybrid"], default=None, help="Random-remasking layout.")
    parser.add_argument("--corruption-iid-fraction", type=float, default=None, help="Hybrid budget fraction for independent bins.")
    parser.add_argument("--corruption-span-fraction", type=float, default=None, help="Hybrid budget fraction for contiguous spans.")
    parser.add_argument("--corruption-layer-fraction", type=float, default=None, help="Hybrid budget fraction for complete predicted runs.")
    parser.add_argument("--corruption-span-min-bins", type=int, default=None, help="Minimum contiguous remasking span in depth bins.")
    parser.add_argument("--corruption-span-max-bins", type=int, default=None, help="Maximum contiguous remasking span in depth bins.")
    parser.add_argument(
        "--corruption-span-scale-with-noise",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Increase the allowed remasking span length with the timestep noise level.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Sampling seed. Required unless config SEED is set.")

    parser.add_argument("--tmm-batch-size", type=int, default=None, help="Decoded stacks per TMM chunk.")
    parser.add_argument(
        "--score-mode",
        type=str,
        default=None,
        choices=["field", "decoded", "both"],
        help="Score native depth-field material runs, decoded token stacks, or both.",
    )
    parser.add_argument(
        "--rank-by",
        type=str,
        default=None,
        choices=["auto", "field", "decoded"],
        help="Candidate MAE used for MC ranking. Auto uses field for both/field mode and decoded for decoded mode.",
    )
    parser.add_argument(
        "--record-spectra",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Store target and predicted spectra arrays.",
    )
    parser.add_argument(
        "--record-all-mc",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Store every MC candidate, not only the best one.",
    )
    parser.add_argument("--plot-bundle", type=str, default=None, help="Dashboard .npz path when recording all MC. Required unless config PLOT_BUNDLE_PATH is set.")
    parser.add_argument("--no-plot-bundle", action="store_true", help="Do not save the dashboard plot bundle.")
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


def resolve_tmm_device(tmm_arg: str, default_device: torch.device) -> torch.device:
    if tmm_arg is None or str(tmm_arg).lower() in {"auto", "same"}:
        return default_device
    device = torch.device(tmm_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA TMM was requested, but CUDA is not available.")
    return device


def resolve_out_json(cfg: dict[str, Any], out_json_arg: str | None) -> str:
    if out_json_arg:
        return str(out_json_arg)
    return str(cfg_required(cfg, "SAMPLES_PATH", "--out-json"))


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


def apply_depth_field_eval_defaults(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    block = depth_field_block(cfg)
    set_arg_config_required(args, "mc_samples", nested_required(block, "EVAL", "MC_SAMPLES"), "DEPTH_FIELD.EVAL.MC_SAMPLES")
    if args.mc_batch_size is None:
        mc_batch_size = nested_get(block, "EVAL", "MC_BATCH_SIZE")
        if mc_batch_size is not None:
            args.mc_batch_size = mc_batch_size
    set_arg_config_required(args, "sampling_steps", nested_required(block, "EVAL", "SAMPLING_STEPS"), "DEPTH_FIELD.EVAL.SAMPLING_STEPS")
    set_arg_config_required(args, "temperature", nested_required(block, "EVAL", "TEMPERATURE"), "DEPTH_FIELD.EVAL.TEMPERATURE")
    set_arg_config_required(args, "top_k", nested_required(block, "EVAL", "TOP_K"), "DEPTH_FIELD.EVAL.TOP_K")
    if args.cfg_scale is None:
        args.cfg_scale = nested_get(block, "EVAL", "CFG_SCALE", default=1.0)
    set_arg_config_required(args, "deterministic", nested_required(block, "EVAL", "DETERMINISTIC"), "DEPTH_FIELD.EVAL.DETERMINISTIC")
    set_arg_config_required(args, "remask_strategy", nested_required(block, "EVAL", "REMASK_STRATEGY"), "DEPTH_FIELD.EVAL.REMASK_STRATEGY")
    corruption = nested_get(block, "CORRUPTION", default={}) or {}
    corruption_defaults = {
        "corruption_mode": nested_get(corruption, "MODE", default="iid"),
        "corruption_iid_fraction": nested_get(corruption, "IID_FRACTION", default=1.0),
        "corruption_span_fraction": nested_get(corruption, "SPAN_FRACTION", default=0.0),
        "corruption_layer_fraction": nested_get(corruption, "LAYER_FRACTION", default=0.0),
        "corruption_span_min_bins": nested_get(corruption, "SPAN_MIN_BINS", default=4),
        "corruption_span_max_bins": nested_get(corruption, "SPAN_MAX_BINS", default=64),
        "corruption_span_scale_with_noise": nested_get(corruption, "SPAN_SCALE_WITH_NOISE", default=True),
    }
    for name, value in corruption_defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    set_arg_config_required(args, "tmm_device", nested_required(block, "EVAL", "TMM_DEVICE"), "DEPTH_FIELD.EVAL.TMM_DEVICE")
    set_arg_config_required(args, "tmm_batch_size", nested_required(block, "EVAL", "TMM_BATCH_SIZE"), "DEPTH_FIELD.EVAL.TMM_BATCH_SIZE")
    set_arg_config_required(args, "score_mode", nested_required(block, "EVAL", "SCORE_MODE"), "DEPTH_FIELD.EVAL.SCORE_MODE")
    set_arg_config_required(args, "rank_by", nested_required(block, "EVAL", "RANK_BY"), "DEPTH_FIELD.EVAL.RANK_BY")
    set_arg_config_required(args, "record_spectra", nested_required(block, "EVAL", "RECORD_SPECTRA"), "DEPTH_FIELD.EVAL.RECORD_SPECTRA")
    set_arg_config_required(args, "record_all_mc", nested_required(block, "EVAL", "RECORD_ALL_MC"), "DEPTH_FIELD.EVAL.RECORD_ALL_MC")


def corruption_config_from_args(args: argparse.Namespace) -> optollama.model.DepthFieldCorruptionConfig:
    """Build the random-remasking policy shared with training."""
    return optollama.model.DepthFieldCorruptionConfig(
        mode=str(args.corruption_mode),
        iid_fraction=float(args.corruption_iid_fraction),
        span_fraction=float(args.corruption_span_fraction),
        layer_fraction=float(args.corruption_layer_fraction),
        span_min_bins=int(args.corruption_span_min_bins),
        span_max_bins=int(args.corruption_span_max_bins),
        span_scale_with_noise=bool(args.corruption_span_scale_with_noise),
    )


def resolve_plot_bundle_path(cfg: dict[str, Any], args: argparse.Namespace) -> str | None:
    if args.no_plot_bundle or not bool(args.record_all_mc):
        return None
    if args.plot_bundle:
        return str(args.plot_bundle)
    return str(cfg_required(cfg, "PLOT_BUNDLE_PATH", "--plot-bundle"))


def depth_field_metadata(cfg: dict[str, Any], depth_info: dict[str, Any], key: str, *config_path: str) -> Any:
    value = depth_info.get(key)
    if value is not None:
        return value
    config_value = nested_required(depth_field_block(cfg), *config_path)
    if config_value is MISSING or config_value is None:
        source = ".".join(("DEPTH_FIELD", *config_path))
        raise ValueError(f"Checkpoint is missing depth_field.{key}, and {source} is not set in the config.")
    return config_value


def split_paths(cfg: dict[str, Any], split: str) -> list[str]:
    prefix = "DATA_PATH_TRAIN" if split == "train" else "DATA_PATH_TEST"
    paths = sorted([str(cfg[key]) for key in cfg.keys() if key == prefix or key.startswith(f"{prefix}_")])
    if not paths:
        raise ValueError(f"No configured paths found for {split!r} split with prefix {prefix}.")
    return paths


def make_eval_loader(cfg: dict[str, Any], args: argparse.Namespace) -> torch.utils.data.DataLoader:
    if args.sharded_loading and args.eager_loading:
        raise ValueError("Use only one of --sharded-loading or --eager-loading.")

    if args.sharded_loading:
        use_shards = True
    elif args.eager_loading:
        use_shards = False
    else:
        use_shards = bool(cfg_required(cfg, "SHARDED_LOADING", "--sharded-loading/--eager-loading"))

    paths = split_paths(cfg, args.split)
    if use_shards:
        dataset = optollama.data.ShardedSpectraDataset(
            paths,
            split=args.split,
            subset_n=args.max_samples if args.max_samples is not None else cfg_required(cfg, "NUM_SAMPLES_TEST", "--max-samples"),
            rank=0,
            world_size=1,
            seed=int(args.seed if args.seed is not None else cfg_required(cfg, "SEED", "--seed")),
            shuffle=False,
        )
    else:
        dataset = optollama.data.SpectraDataset(paths)
        max_samples = int(args.max_samples if args.max_samples is not None else cfg_required(cfg, "NUM_SAMPLES_TEST", "--max-samples"))
        if max_samples < len(dataset):
            indices = optollama.data.SpectraDataset.indices_of_unique_equidistant_subset(0, len(dataset) - 1, max_samples)
            dataset = torch.utils.data.Subset(dataset, indices)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size if args.batch_size is not None else cfg_required(cfg, "TEST_BATCH_SIZE", "--batch-size")),
        shuffle=False,
        num_workers=int(args.num_workers if args.num_workers is not None else cfg_required(cfg, "NUM_WORKERS", "--num-workers")),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def configured_target(cfg: dict[str, Any], args: argparse.Namespace) -> str | None:
    target = args.target if args.target is not None else cfg.get("TARGET")
    if target is None:
        return None
    target = str(target)
    return target if target.strip() else None


def target_physicalize_enabled(cfg: dict[str, Any]) -> bool:
    return bool((cfg.get("TARGET_PHYSICALIZE") or {}).get("ENABLED", False))


def selection_target_mode(cfg: dict[str, Any]) -> str:
    if not target_physicalize_enabled(cfg):
        return "conditioned"
    block = cfg.get("TARGET_PHYSICALIZE") or {}
    if "SELECTION_TARGET" not in block or block["SELECTION_TARGET"] is None:
        raise ValueError("Missing required setting TARGET_PHYSICALIZE.SELECTION_TARGET.")
    mode = str(block["SELECTION_TARGET"]).lower()
    if mode not in {"conditioned", "original"}:
        raise ValueError("TARGET_PHYSICALIZE.SELECTION_TARGET must be 'conditioned' or 'original'.")
    return mode


def log_physicalization(info: dict[str, Any], cfg: dict[str, Any]) -> None:
    if not info.get("enabled"):
        return
    parts = [
        "TARGET_PHYSICALIZE enabled",
        f"selection_target={selection_target_mode(cfg)}",
    ]
    ae = info.get("autoencoder")
    if ae:
        parts.append(
            "AE "
            f"mae_to_input={ae['mae_to_input']:.6f} "
            f"latent_dim={ae['latent_dim']} "
            f"mode={ae['mode']}"
        )
    nn = info.get("nn")
    if nn:
        parts.append(
            f"NN id={nn['global_index']} mae={nn['mae']:.6f} "
            f"file={nn['file']}:{nn['local_index']}"
        )
        cache = nn.get("cache") or {}
        if cache:
            parts.append(f"NN cache={'hit' if cache.get('hit') else 'miss'}")
    random_info = info.get("random")
    if random_info:
        parts.append(
            "RANDOM "
            f"variants={random_info['variants']} "
            f"sigma_abs={random_info['sigma_abs']:g} "
            f"sigma_rel={random_info['sigma_rel']:g}"
        )
    print(", ".join(parts) + ".")


def load_target_spectra(
    target: str,
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    if target == "random":
        width = int(cfg["WAVELENGTHS"].numel() if torch.is_tensor(cfg["WAVELENGTHS"]) else len(cfg["WAVELENGTHS"]))
        original = torch.rand((3, width), dtype=torch.float32, device=device)
    else:
        original = optollama.utils.load_spectra(target, cfg).to(device=device, dtype=torch.float32)

    conditioned, info = optollama.data.physicalize_target_spectrum(original, cfg, device=device)
    score_spectrum = None
    if info.get("enabled"):
        if selection_target_mode(cfg) == "original":
            score_spectrum, _ = optollama.data.ensure_3w(original)
            score_spectrum = score_spectrum.to(torch.float32)

    return conditioned.detach().cpu(), None if score_spectrum is None else score_spectrum.detach().cpu(), info


def make_target_loader(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    target: str,
    msk_idx: int,
    device: torch.device,
) -> tuple[torch.utils.data.DataLoader, torch.Tensor | None, dict[str, Any]]:
    spectrum, score_spectrum, physicalize_info = load_target_spectra(target, cfg, device)
    n_targets = int(args.target_samples if args.target_samples is not None else cfg_required(cfg, "N_TARGETS", "--target-samples"))
    n_targets = max(1, n_targets)
    spectra = spectrum.unsqueeze(0).repeat(n_targets, 1, 1).contiguous()
    spectra, random_info = optollama.data.randomize_target_spectra(spectra, cfg)
    if random_info.get("enabled"):
        physicalize_info = dict(physicalize_info)
        physicalize_info["random"] = random_info
    log_physicalization(physicalize_info, cfg)
    stacks = torch.full((n_targets, int(cfg["MAX_SEQ_LEN"])), int(msk_idx), dtype=torch.long)
    indices = torch.arange(n_targets, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(spectra, stacks, indices)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(n_targets, int(args.batch_size if args.batch_size is not None else cfg_required(cfg, "TEST_BATCH_SIZE", "--batch-size"))),
        shuffle=False,
        num_workers=int(args.num_workers if args.num_workers is not None else cfg_required(cfg, "NUM_WORKERS", "--num-workers")),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return loader, score_spectrum, physicalize_info


def score_spectra_for_batch(
    spectra_cpu: torch.Tensor,
    score_spectrum_cpu: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if score_spectrum_cpu is None:
        return spectra_cpu, None

    batch_size = int(spectra_cpu.size(0))
    score = torch.as_tensor(score_spectrum_cpu, dtype=spectra_cpu.dtype)
    if score.dim() == 2:
        score = score.unsqueeze(0).expand(batch_size, -1, -1)
    elif score.dim() == 3 and score.size(0) == 1:
        score = score.expand(batch_size, -1, -1)
    elif score.dim() == 3 and score.size(0) == batch_size:
        pass
    else:
        raise ValueError(
            "score_spectrum must have shape [3,W], [1,3,W], or [B,3,W], "
            f"got {tuple(score.shape)} for batch size {batch_size}"
        )
    return score.contiguous(), spectra_cpu


def load_depth_model(checkpoint: str, device: torch.device) -> tuple[optollama.model.DepthFieldDiffusion, dict]:
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    extra = blob.get("extra") or {}
    if "model_config" not in extra:
        raise RuntimeError("Depth-field checkpoint is missing extra['model_config']; use the .pt checkpoint from training.")
    model_config = optollama.model.DepthFieldModelConfig.from_dict(extra["model_config"])
    model = optollama.model.build_depth_field_model(model_config)
    model.load_state_dict(blob["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, extra


def mae_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    wavelengths: torch.Tensor,
    cfg: dict[str, Any],
) -> torch.Tensor:
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
    diff = diff[:, :, mask]
    return diff.mean(dim=(1, 2))


def simulate_decoded(
    token_ids: torch.Tensor,
    *,
    tmm_ctx: optollama.evaluation.simulation.TMMContext,
    eos_idx: int,
    pad_idx: int,
    msk_idx: int,
    tmm_batch_size: int,
) -> torch.Tensor:
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
    outputs = []
    runs_batch = [optollama.data.depth_field_runs(field, vocab, dz_nm=dz_nm) for field in fields_cpu]
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


def stack_total_thickness_nm(tokens: list[str]) -> float:
    total = 0.0
    for token in tokens:
        parts = optollama.data.layer_token_parts(token)
        if parts is not None:
            total += float(parts[1])
    return total


def candidate_record(
    *,
    flat_idx: int,
    fields_cpu: torch.Tensor,
    token_ids_cpu: torch.Tensor,
    mae_values: torch.Tensor,
    pred_spectra_cpu: torch.Tensor,
    target_spectrum_cpu: torch.Tensor,
    conditioning_spectrum_cpu: torch.Tensor | None,
    field_mae_values: torch.Tensor | None,
    decoded_mae_values: torch.Tensor | None,
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
    record = {
        "mae": float(mae_values[flat_idx].item()),
        "tokens": tokens,
        "material_layers": int(len(tokens)),
        "decoded_total_thickness_nm": float(stack_total_thickness_nm(tokens)),
        "field_active_thickness_nm": float(active_bins * dz_nm),
        "field_material_runs": int(len(field_runs)),
        "field_total_thickness_nm": float(sum(float(run["thickness_nm"]) for run in field_runs)),
        "field_runs": field_runs,
    }
    if field_mae_values is not None:
        record["field_mae"] = float(field_mae_values[flat_idx].item())
    if decoded_mae_values is not None:
        record["decoded_mae"] = float(decoded_mae_values[flat_idx].item())
    if record_spectra:
        record["target_spectra"] = target_spectrum_cpu.detach().cpu().tolist()
        record["pred_spectra"] = pred_spectra_cpu[flat_idx].detach().cpu().tolist()
        if conditioning_spectrum_cpu is not None:
            record["rat_conditioning"] = conditioning_spectrum_cpu.detach().cpu().tolist()
    return record


def run_depth_field_inference(
    *,
    cfg: dict[str, Any],
    args: argparse.Namespace,
    out_json: str,
    target: str | None,
    loader: torch.utils.data.DataLoader,
    score_spectrum_cpu: torch.Tensor | None,
    physicalize_info: dict[str, Any],
    model: optollama.model.DepthFieldDiffusion,
    extra: dict[str, Any],
    tokens: list[str],
    token_to_idx: dict[str, int],
    idx_to_token: dict[int, str],
    eos_idx: int,
    pad_idx: int,
    msk_idx: int,
    device: torch.device,
    tmm_device: torch.device,
) -> dict[str, Any]:
    vocab = optollama.data.build_depth_field_vocab(tokens, token_to_idx)
    material_to_token_id = optollama.data.depth_field_material_token_ids(vocab)
    depth_info = extra.get("depth_field") or {}
    dz_nm = float(depth_field_metadata(cfg, depth_info, "dz_nm", "DZ_NM"))
    max_thickness_nm = float(depth_field_metadata(cfg, depth_info, "max_thickness_nm", "MAX_THICKNESS_NM"))
    output_seq_len = int(depth_field_metadata(cfg, depth_info, "output_seq_len", "OUTPUT_SEQ_LEN"))

    if vocab.num_clean_classes != model.num_materials:
        raise RuntimeError(
            f"Token config gives {vocab.num_clean_classes} depth classes, but checkpoint expects {model.num_materials}."
        )

    tmm_ctx = optollama.evaluation.simulation.TMMContext.make(cfg=cfg, idx_to_token=idx_to_token, device=tmm_device)
    results: list[dict] = []
    mae_all: list[float] = []
    selected_field_mae: list[float] = []
    selected_decoded_mae: list[float] = []
    all_mc_mae_batches: list[torch.Tensor] = []
    all_mc_ids_batches: list[torch.Tensor] = []
    all_mc_pred_batches: list[torch.Tensor] = []
    mc_samples = int(args.mc_samples)
    mc_batch_size = int(args.mc_batch_size) if args.mc_batch_size is not None else mc_samples
    if mc_samples <= 0:
        raise ValueError(f"--mc-samples must be positive, got {mc_samples}")
    if mc_batch_size <= 0:
        raise ValueError(f"--mc-batch-size must be positive when set, got {mc_batch_size}")
    mc_batch_size = min(mc_batch_size, mc_samples)
    score_mode = str(args.score_mode).lower()
    rank_by = str(args.rank_by).lower()
    if rank_by == "auto":
        rank_by = "field" if score_mode in {"field", "both"} else "decoded"
    if score_mode != "both" and rank_by != score_mode:
        raise ValueError(f"--rank-by={rank_by} requires --score-mode=both or --score-mode={rank_by}.")

    print(
        "Depth-field inference: "
        f"{'target=' + target if target is not None else 'split=' + args.split}, "
        f"mc={mc_samples}, bins={model.depth_bins}, dz={dz_nm:g}nm, "
        f"max={max_thickness_nm:g}nm, score_mode={score_mode}, rank_by={rank_by}, "
        f"remask={args.remask_strategy}, cfg={args.cfg_scale:g}, corruption={args.corruption_config.mode}, "
        f"mc_batch={mc_batch_size}, "
        f"model_device={device}, tmm_device={tmm_device}"
    )

    mc_chunks_per_batch = (mc_samples + mc_batch_size - 1) // mc_batch_size
    progress_total = len(loader) * mc_chunks_per_batch if hasattr(loader, "__len__") else None
    progress = tqdm.tqdm(
        total=progress_total,
        desc="depth-field inference",
        unit="mc-batch",
        leave=True,
    )

    for loader_batch_idx, batch in enumerate(loader, start=1):
        spectra_cpu, _, indices = batch[0], batch[1], batch[2]
        batch_size = int(spectra_cpu.size(0))
        score_spectra_cpu, conditioning_spectra_cpu = score_spectra_for_batch(spectra_cpu, score_spectrum_cpu)
        spectra = spectra_cpu.to(device, non_blocking=True)

        fields_chunks: list[torch.Tensor] = []
        token_chunks: list[torch.Tensor] = []
        mae_chunks: list[torch.Tensor] = []
        pred_spectra_chunks: list[torch.Tensor] = []
        field_mae_chunks: list[torch.Tensor] = []
        decoded_mae_chunks: list[torch.Tensor] = []

        for mc_start in range(0, mc_samples, mc_batch_size):
            chunk_n = min(mc_batch_size, mc_samples - mc_start)
            spectra_rep = spectra.repeat_interleave(chunk_n, dim=0)

            fields = model.sample(
                spectra_rep,
                steps=args.sampling_steps,
                temperature=float(args.temperature),
                top_k=int(args.top_k),
                deterministic=bool(args.deterministic or args.temperature <= 0.0),
                guidance_scale=float(args.cfg_scale),
                remask_strategy=str(args.remask_strategy),
                corruption_config=args.corruption_config,
            )
            fields_cpu_chunk = fields.detach().cpu()
            token_ids_cpu_chunk = optollama.data.decode_depth_field_to_tokens(
                fields_cpu_chunk,
                vocab,
                output_seq_len=output_seq_len,
                dz_nm=dz_nm,
                eos_idx=eos_idx,
                pad_idx=pad_idx,
            )

            target_rep = score_spectra_cpu.to(tmm_device).repeat_interleave(chunk_n, dim=0)
            field_mae_chunk = None
            decoded_mae_chunk = None
            field_spectra_cpu_chunk = None
            decoded_spectra_cpu_chunk = None

            if score_mode in {"field", "both"}:
                field_spectra = simulate_field_runs(
                    fields_cpu_chunk,
                    vocab=vocab,
                    tmm_ctx=tmm_ctx,
                    material_to_token_id=material_to_token_id,
                    dz_nm=dz_nm,
                    eos_idx=eos_idx,
                    pad_idx=pad_idx,
                    msk_idx=msk_idx,
                    tmm_batch_size=int(args.tmm_batch_size),
                )
                field_mae_chunk = mae_per_sample(field_spectra, target_rep, cfg["WAVELENGTHS"], cfg).detach().cpu()
                field_spectra_cpu_chunk = field_spectra.detach().cpu()
                field_mae_chunks.append(field_mae_chunk.reshape(batch_size, chunk_n))

            if score_mode in {"decoded", "both"}:
                decoded_spectra = simulate_decoded(
                    token_ids_cpu_chunk,
                    tmm_ctx=tmm_ctx,
                    eos_idx=eos_idx,
                    pad_idx=pad_idx,
                    msk_idx=msk_idx,
                    tmm_batch_size=int(args.tmm_batch_size),
                )
                decoded_mae_chunk = mae_per_sample(decoded_spectra, target_rep, cfg["WAVELENGTHS"], cfg).detach().cpu()
                decoded_spectra_cpu_chunk = decoded_spectra.detach().cpu()
                decoded_mae_chunks.append(decoded_mae_chunk.reshape(batch_size, chunk_n))

            if rank_by == "field":
                if field_mae_chunk is None or field_spectra_cpu_chunk is None:
                    raise RuntimeError("Field MAE requested for ranking but was not computed.")
                mae_chunk = field_mae_chunk
                pred_spectra_cpu_chunk = field_spectra_cpu_chunk
            else:
                if decoded_mae_chunk is None or decoded_spectra_cpu_chunk is None:
                    raise RuntimeError("Decoded MAE requested for ranking but was not computed.")
                mae_chunk = decoded_mae_chunk
                pred_spectra_cpu_chunk = decoded_spectra_cpu_chunk

            fields_chunks.append(fields_cpu_chunk.reshape(batch_size, chunk_n, -1))
            token_chunks.append(token_ids_cpu_chunk.reshape(batch_size, chunk_n, -1))
            mae_chunks.append(mae_chunk.reshape(batch_size, chunk_n))
            pred_spectra_chunks.append(pred_spectra_cpu_chunk.reshape(batch_size, chunk_n, *pred_spectra_cpu_chunk.shape[1:]))
            progress.update(1)
            progress.set_postfix(
                batch=f"{loader_batch_idx}/{len(loader) if hasattr(loader, '__len__') else '?'}",
                mc=f"{mc_start + chunk_n}/{mc_samples}",
            )

        fields_cpu = torch.cat(fields_chunks, dim=1).reshape(batch_size * mc_samples, -1)
        token_ids_cpu = torch.cat(token_chunks, dim=1).reshape(batch_size * mc_samples, -1)
        mae_matrix = torch.cat(mae_chunks, dim=1)
        mae = mae_matrix.reshape(batch_size * mc_samples)
        pred_spectra_cpu = torch.cat(pred_spectra_chunks, dim=1).reshape(batch_size * mc_samples, *pred_spectra_chunks[0].shape[2:])
        field_mae = torch.cat(field_mae_chunks, dim=1).reshape(batch_size * mc_samples) if field_mae_chunks else None
        decoded_mae = torch.cat(decoded_mae_chunks, dim=1).reshape(batch_size * mc_samples) if decoded_mae_chunks else None

        best_mae, best_idx = mae_matrix.min(dim=1)
        if args.record_all_mc:
            all_mc_mae_batches.append(mae_matrix.detach().cpu().to(torch.float32))
            all_mc_ids_batches.append(token_ids_cpu.reshape(batch_size, mc_samples, -1).detach().cpu().to(torch.long))
            if args.record_spectra:
                all_mc_pred_batches.append(
                    pred_spectra_cpu.reshape(batch_size, mc_samples, *pred_spectra_cpu.shape[1:]).detach().cpu().to(torch.float32)
                )

        for row_idx in range(batch_size):
            flat_idx = row_idx * mc_samples + int(best_idx[row_idx].item())
            record = {
                "dataset_index": int(indices[row_idx].item() if torch.is_tensor(indices[row_idx]) else indices[row_idx]),
                "best_mc": int(best_idx[row_idx].item()),
                "mc_samples": mc_samples,
            }
            record.update(
                candidate_record(
                    flat_idx=flat_idx,
                    fields_cpu=fields_cpu,
                    token_ids_cpu=token_ids_cpu,
                    mae_values=mae,
                    pred_spectra_cpu=pred_spectra_cpu,
                    target_spectrum_cpu=score_spectra_cpu[row_idx],
                    conditioning_spectrum_cpu=conditioning_spectra_cpu[row_idx] if conditioning_spectra_cpu is not None else None,
                    field_mae_values=field_mae,
                    decoded_mae_values=decoded_mae,
                    vocab=vocab,
                    idx_to_token=idx_to_token,
                    eos_idx=eos_idx,
                    pad_idx=pad_idx,
                    dz_nm=dz_nm,
                    record_spectra=bool(args.record_spectra),
                )
            )
            if args.record_all_mc:
                record["all_mc"] = [
                    candidate_record(
                        flat_idx=row_idx * mc_samples + candidate_idx,
                        fields_cpu=fields_cpu,
                        token_ids_cpu=token_ids_cpu,
                        mae_values=mae,
                        pred_spectra_cpu=pred_spectra_cpu,
                        target_spectrum_cpu=score_spectra_cpu[row_idx],
                        conditioning_spectrum_cpu=conditioning_spectra_cpu[row_idx] if conditioning_spectra_cpu is not None else None,
                        field_mae_values=field_mae,
                        decoded_mae_values=decoded_mae,
                        vocab=vocab,
                        idx_to_token=idx_to_token,
                        eos_idx=eos_idx,
                        pad_idx=pad_idx,
                        dz_nm=dz_nm,
                        record_spectra=bool(args.record_spectra),
                    )
                    for candidate_idx in range(mc_samples)
                ]
            results.append(record)
            mae_all.append(float(best_mae[row_idx].item()))
            if field_mae is not None:
                selected_field_mae.append(float(field_mae[flat_idx].item()))
            if decoded_mae is not None:
                selected_decoded_mae.append(float(decoded_mae[flat_idx].item()))

    progress.close()

    field_mae_tensor = torch.tensor(selected_field_mae) if selected_field_mae else None
    decoded_mae_tensor = torch.tensor(selected_decoded_mae) if selected_decoded_mae else None
    summary = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "out_json": out_json,
        "split": str(args.split),
        "target": target,
        "samples": int(len(results)),
        "mc_samples": mc_samples,
        "score_mode": score_mode,
        "rank_by": rank_by,
        "cfg_scale": float(args.cfg_scale),
        "remask_strategy": str(args.remask_strategy),
        "corruption": args.corruption_config.to_dict(),
        "mae_mean": float(torch.tensor(mae_all).mean().item()) if mae_all else None,
        "mae_median": float(torch.tensor(mae_all).median().item()) if mae_all else None,
        "mae_min": float(min(mae_all)) if mae_all else None,
        "mae_max": float(max(mae_all)) if mae_all else None,
        "field_mae_mean": float(field_mae_tensor.mean().item()) if field_mae_tensor is not None else None,
        "field_mae_median": float(field_mae_tensor.median().item()) if field_mae_tensor is not None else None,
        "decoded_mae_mean": float(decoded_mae_tensor.mean().item()) if decoded_mae_tensor is not None else None,
        "decoded_mae_median": float(decoded_mae_tensor.median().item()) if decoded_mae_tensor is not None else None,
        "depth_field": {
            "dz_nm": dz_nm,
            "max_thickness_nm": max_thickness_nm,
            "depth_bins": int(model.depth_bins),
            "classes": list(vocab.material_names),
        },
    }
    if physicalize_info.get("enabled"):
        summary["target_physicalize"] = {
            "enabled": True,
            "selection_target": selection_target_mode(cfg),
            "autoencoder": physicalize_info.get("autoencoder"),
            "nn": physicalize_info.get("nn"),
            "random": physicalize_info.get("random"),
        }
    plot_bundle_path = resolve_plot_bundle_path(cfg, args)
    if plot_bundle_path:
        bundle_output: dict[str, torch.Tensor] = {
            "mae_grid": torch.cat(all_mc_mae_batches, dim=0) if all_mc_mae_batches else torch.empty((0, mc_samples)),
            "ids_grid": torch.cat(all_mc_ids_batches, dim=0) if all_mc_ids_batches else torch.empty((0, mc_samples, 0), dtype=torch.long),
        }
        if all_mc_pred_batches:
            bundle_output["pred_spectra_grid"] = torch.cat(all_mc_pred_batches, dim=0)
        optollama.plotting.save_plot_bundle(
            plot_bundle_path,
            bundle_output,
            wavelengths=cfg["WAVELENGTHS"],
            roi_min=cfg.get("ROI_MIN"),
            roi_max=cfg.get("ROI_MAX"),
        )
        summary["plot_bundle"] = plot_bundle_path
        print(f"Saved depth-field plot bundle -> {plot_bundle_path}")
    out = {"summary": summary, "results": results}
    out_path = Path(out_json)
    os.makedirs(out_path.parent, exist_ok=True)
    optollama.utils.save_as_json(str(out_path), out)
    print(f"Saved {len(results)} depth-field samples -> {out_path}")
    if mae_all:
        print(f"MAE mean={summary['mae_mean']:.6f}, median={summary['mae_median']:.6f}, best={summary['mae_min']:.6f}")
    return out


def main() -> None:
    args = parse_args()
    cfg = optollama.utils.load_config(args)
    apply_depth_field_eval_defaults(cfg, args)
    args.corruption_config = corruption_config_from_args(args)
    sample_stamp = optollama.utils.make_run_stamp()
    seed = int(args.seed if args.seed is not None else cfg_required(cfg, "SEED", "--seed"))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = resolve_device(args.device)
    tmm_device = resolve_tmm_device(args.tmm_device, device)
    tokens, token_to_idx, idx_to_token, _, _, _, eos_idx, pad_idx, msk_idx = optollama.data.init_tokens(cfg["TOKENS_PATH"])
    model, extra = load_depth_model(args.checkpoint, device)

    explicit_target = configured_target(cfg, args) if args.target is not None else None
    if explicit_target is not None:
        target_spec = optollama.utils.TargetSpec(
            target=explicit_target,
            name=optollama.utils.safe_target_name(explicit_target),
        )
        run_cfg = optollama.utils.cfg_for_target(
            cfg,
            target_spec,
            multi_target=False,
            sample_stamp=None if args.out_json else sample_stamp,
        )
        loader, score_spectrum_cpu, physicalize_info = make_target_loader(run_cfg, args, explicit_target, msk_idx, device)
        run_depth_field_inference(
            cfg=run_cfg,
            args=args,
            out_json=resolve_out_json(run_cfg, args.out_json),
            target=explicit_target,
            loader=loader,
            score_spectrum_cpu=score_spectrum_cpu,
            physicalize_info=physicalize_info,
            model=model,
            extra=extra,
            tokens=tokens,
            token_to_idx=token_to_idx,
            idx_to_token=idx_to_token,
            eos_idx=eos_idx,
            pad_idx=pad_idx,
            msk_idx=msk_idx,
            device=device,
            tmm_device=tmm_device,
        )
        return

    target_entries, multi_target = optollama.utils.target_cfgs(cfg, sample_stamp=sample_stamp)
    if target_entries:
        if multi_target and args.out_json:
            raise ValueError(
                "--out-json cannot be used with TARGETS/TARGET_GLOB because each target writes its own samples file."
            )
        if multi_target and args.plot_bundle:
            raise ValueError(
                "--plot-bundle cannot be used with TARGETS/TARGET_GLOB because each target writes its own plot bundle."
            )
        if multi_target:
            print(f"Depth-field multi-target inference enabled: {len(target_entries)} targets.")
        for index, (spec, target_cfg) in enumerate(target_entries, start=1):
            if multi_target:
                print(f"\n[{index}/{len(target_entries)}] Target {spec.name}: {spec.target}")
            loader, score_spectrum_cpu, physicalize_info = make_target_loader(target_cfg, args, spec.target, msk_idx, device)
            run_depth_field_inference(
                cfg=target_cfg,
                args=args,
                out_json=resolve_out_json(target_cfg, args.out_json),
                target=spec.target,
                loader=loader,
                score_spectrum_cpu=score_spectrum_cpu,
                physicalize_info=physicalize_info,
                model=model,
                extra=extra,
                tokens=tokens,
                token_to_idx=token_to_idx,
                idx_to_token=idx_to_token,
                eos_idx=eos_idx,
                pad_idx=pad_idx,
                msk_idx=msk_idx,
                device=device,
                tmm_device=tmm_device,
            )
        return

    run_cfg = optollama.utils.cfg_with_timestamped_samples_path(cfg, None if args.out_json else sample_stamp)
    loader = make_eval_loader(run_cfg, args)
    run_depth_field_inference(
        cfg=run_cfg,
        args=args,
        out_json=resolve_out_json(run_cfg, args.out_json),
        target=None,
        loader=loader,
        score_spectrum_cpu=None,
        physicalize_info={"enabled": False},
        model=model,
        extra=extra,
        tokens=tokens,
        token_to_idx=token_to_idx,
        idx_to_token=idx_to_token,
        eos_idx=eos_idx,
        pad_idx=pad_idx,
        msk_idx=msk_idx,
        device=device,
        tmm_device=tmm_device,
    )


if __name__ == "__main__":
    main()
