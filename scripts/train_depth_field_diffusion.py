#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from contextlib import nullcontext
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
    parser.add_argument("--out-dir", type=str, default="data/checkpoints/depth_field_10um", help="Checkpoint output dir.")
    parser.add_argument("--device", type=str, default=None, help='Device, e.g. "cuda", "cuda:0", or "cpu".')
    parser.add_argument("--seed", type=int, default=None, help="Random seed. Defaults to config SEED.")

    parser.add_argument("--dz-nm", type=float, default=10.0, help="Depth resolution in nm.")
    parser.add_argument("--max-thickness-nm", type=float, default=10_000.0, help="Maximum represented total thickness in nm.")
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
    parser.add_argument("--sharded-loading", action="store_true", help="Force sharded streaming dataset loading.")
    parser.add_argument("--eager-loading", action="store_true", help="Force eager in-memory dataset loading.")
    parser.add_argument(
        "--keep-overlimit-stacks",
        action="store_true",
        help="Keep stacks thicker than --max-thickness-nm by clipping them. Default skips them.",
    )

    parser.add_argument("--d-model", type=int, default=192, help="Depth-field model channel width.")
    parser.add_argument("--n-blocks", type=int, default=12, help="Number of dilated Conv1d residual blocks.")
    parser.add_argument("--kernel-size", type=int, default=7, help="Conv1d kernel size.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout inside residual blocks.")
    parser.add_argument("--diffusion-steps", type=int, default=100, help="Discrete depth-field diffusion timesteps.")

    parser.add_argument("--learning-rate", type=float, default=None, help="Optimizer LR. Defaults to config LEARNING_RATE.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient norm clip. <=0 disables clipping.")
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast/GradScaler.")

    parser.add_argument("--void-loss-weight", type=float, default=0.25, help="CE class weight for the void depth class.")
    parser.add_argument(
        "--random-replace-prob",
        type=float,
        default=0.10,
        help="Fraction of corrupted bins that are random material replacements rather than masks.",
    )
    parser.add_argument(
        "--loss-on-corrupted-only",
        action="store_true",
        help="Compute CE only on bins that were masked/replaced. Default supervises every bin.",
    )
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint to resume.")
    parser.add_argument("--save-every", type=int, default=1, help="Save the last checkpoint every N epochs.")
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="tmm",
        choices=["tmm", "denoise", "both"],
        help="Validation mode. 'tmm' samples all-mask fields and scores TMM MAE; 'denoise' uses CE on corrupted targets.",
    )
    parser.add_argument("--eval-mc-samples", type=int, default=4, help="TMM validation candidates per target.")
    parser.add_argument("--eval-sampling-steps", type=int, default=None, help="TMM validation denoising steps.")
    parser.add_argument("--eval-temperature", type=float, default=1.0, help="TMM validation sampling temperature.")
    parser.add_argument("--eval-top-k", type=int, default=0, help="TMM validation top-k material sampling filter.")
    parser.add_argument("--eval-deterministic", action="store_true", help="Use argmax sampling for TMM validation.")
    parser.add_argument("--eval-tmm-device", type=str, default="auto", help='TMM validation device. "auto" uses model device.')
    parser.add_argument("--eval-tmm-batch-size", type=int, default=64, help="Decoded stacks per TMM validation chunk.")
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


def autocast_context(enabled: bool):
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


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


@torch.no_grad()
def run_tmm_evaluation(
    *,
    model: optollama.model.DepthFieldDiffusion,
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
) -> dict:
    """Run proper all-mask depth-field sampling and score decoded stacks by TMM MAE."""
    model.eval()
    set_loader_epoch(loader, epoch)

    mc_samples = int(args.eval_mc_samples)
    if mc_samples <= 0:
        raise ValueError(f"--eval-mc-samples must be positive, got {mc_samples}")

    output_seq_len = int(args.output_seq_len or cfg["MAX_SEQ_LEN"])
    mae_values: list[torch.Tensor] = []
    best_layers: list[torch.Tensor] = []
    best_thickness: list[torch.Tensor] = []
    target_overlimit = 0
    samples_seen = 0
    pbar = tqdm.tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs} tmm", leave=True)

    for batch in pbar:
        spectra_cpu, stacks_cpu = batch[0], batch[1]
        batch_size = int(spectra_cpu.size(0))
        samples_seen += batch_size
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
        fields = model.sample(
            spectra_rep,
            steps=args.eval_sampling_steps,
            temperature=float(args.eval_temperature),
            top_k=int(args.eval_top_k),
            deterministic=bool(args.eval_deterministic or args.eval_temperature <= 0.0),
        )
        token_ids_cpu = optollama.data.decode_depth_field_to_tokens(
            fields.detach().cpu(),
            vocab,
            output_seq_len=output_seq_len,
            dz_nm=args.dz_nm,
            eos_idx=eos_idx,
            pad_idx=pad_idx,
        )
        pred_spectra = simulate_decoded(
            token_ids_cpu,
            tmm_ctx=tmm_ctx,
            eos_idx=eos_idx,
            pad_idx=pad_idx,
            msk_idx=msk_idx,
            tmm_batch_size=int(args.eval_tmm_batch_size),
        )
        target_rep = spectra_cpu.to(tmm_ctx.wl.device).repeat_interleave(mc_samples, dim=0)
        mae = mae_per_sample(pred_spectra, target_rep, cfg["WAVELENGTHS"], cfg).detach().cpu().view(batch_size, mc_samples)
        best_mae, best_idx = mae.min(dim=1)
        flat_idx = torch.arange(batch_size, dtype=torch.long) * mc_samples + best_idx
        best_ids = token_ids_cpu[flat_idx]
        non_pad = best_ids != int(pad_idx)
        non_special = non_pad & (best_ids != int(eos_idx)) & (best_ids != int(msk_idx))
        decoded_thickness = optollama.data.token_stack_total_thickness_nm(
            best_ids,
            idx_to_token,
            eos_idx=eos_idx,
            pad_idx=pad_idx,
            msk_idx=msk_idx,
        )

        mae_values.append(best_mae)
        best_layers.append(non_special.sum(dim=1).float())
        best_thickness.append(decoded_thickness.float())
        all_mae = torch.cat(mae_values)
        pbar.set_postfix(
            mae=f"{all_mae.mean().item():.5f}",
            best=f"{all_mae.min().item():.5f}",
            layers=f"{torch.cat(best_layers).mean().item():.1f}",
            th=f"{torch.cat(best_thickness).mean().item():.0f}nm",
        )

    if not mae_values:
        raise RuntimeError("TMM validation produced no samples.")

    all_mae = torch.cat(mae_values)
    all_layers = torch.cat(best_layers)
    all_thickness = torch.cat(best_thickness)
    return {
        "score": float(all_mae.mean().item()),
        "score_name": "tmm_mae_mean",
        "mae_mean": float(all_mae.mean().item()),
        "mae_median": float(all_mae.median().item()),
        "mae_min": float(all_mae.min().item()),
        "mae_max": float(all_mae.max().item()),
        "material_layers_mean": float(all_layers.mean().item()),
        "decoded_total_thickness_nm_mean": float(all_thickness.mean().item()),
        "target_overlimit_fraction": float(target_overlimit / max(samples_seen, 1)),
        "samples_seen": int(samples_seen),
        "mc_samples": int(mc_samples),
        "sampling_steps": int(args.eval_sampling_steps or model.timesteps),
        "tmm_batch_size": int(args.eval_tmm_batch_size),
    }


def run_epoch(
    *,
    model: optollama.model.DepthFieldDiffusion,
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
    pbar = tqdm.tqdm(loader, desc=desc, leave=True)
    validation_interval = int(validate_every_samples or 0)
    next_validation_sample = validation_interval if train and validation_callback is not None and validation_interval > 0 else None

    for batch in pbar:
        spectra_cpu, stacks_cpu = batch[0], batch[1]
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
        if not args.keep_overlimit_stacks:
            skipped_overlimit += overlimit_count
            keep = ~overlimit
            if not bool(keep.any()):
                pbar.set_postfix(skip=f"{skipped_overlimit / max(seen_samples, 1) * 100.0:.1f}%")
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
        active_bins = optollama.data.depth_field_active_bins(fields_cpu, vocab.void_id)
        full_count += int((active_bins >= fields_cpu.size(1)).sum().item())
        active_nm_sum += float(active_bins.float().sum().item() * float(args.dz_nm))
        samples += int(fields_cpu.size(0))

        spectra = spectra_cpu.to(device, non_blocking=True)
        fields = fields_cpu.to(device, non_blocking=True)

        if train:
            if optimizer is None:
                raise RuntimeError("optimizer is required for training")
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(bool(scaler.is_enabled())):
                out = model.training_loss(
                    spectra,
                    fields,
                    void_id=vocab.void_id,
                    void_loss_weight=args.void_loss_weight,
                    random_replace_prob=args.random_replace_prob,
                    loss_on_corrupted_only=args.loss_on_corrupted_only,
                )
            loss = out["loss"]
            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad(), autocast_context(bool(scaler.is_enabled())):
                out = model.training_loss(
                    spectra,
                    fields,
                    void_id=vocab.void_id,
                    void_loss_weight=args.void_loss_weight,
                    random_replace_prob=args.random_replace_prob,
                    loss_on_corrupted_only=args.loss_on_corrupted_only,
                )
            loss = out["loss"]

        loss_sum += float(loss.detach().item())
        batches += 1
        counts += accuracy_counts(out["logits"].detach(), fields, vocab.void_id)
        metrics = counts_to_metrics(
            counts,
            loss_sum=loss_sum,
            batches=batches,
            active_nm_sum=active_nm_sum,
            full_count=full_count,
            samples=samples,
            seen_samples=seen_samples,
            overlimit_seen=overlimit_seen,
            skipped_overlimit=skipped_overlimit,
            dz_nm=args.dz_nm,
        )
        pbar.set_postfix(
            loss=f"{metrics['loss']:.4f}",
            acc=f"{metrics['acc'] * 100.0:.2f}%",
            mat=f"{metrics['mat_acc'] * 100.0:.2f}%",
            void=f"{metrics['void_acc'] * 100.0:.2f}%",
            th=f"{metrics['mean_active_thickness_nm']:.0f}nm",
            skip=f"{metrics['overlimit_skip_fraction'] * 100.0:.1f}%",
            full=f"{metrics['full_depth_fraction'] * 100.0:.1f}%",
        )

        if next_validation_sample is not None and seen_samples >= next_validation_sample:
            validation_callback(epoch, seen_samples, dict(metrics))
            model.train(True)
            while next_validation_sample <= seen_samples:
                next_validation_sample += validation_interval

    if batches == 0:
        raise RuntimeError(
            "No samples remained after over-limit filtering. Increase --max-thickness-nm or use --keep-overlimit-stacks."
        )

    return counts_to_metrics(
        counts,
        loss_sum=loss_sum,
        batches=batches,
        active_nm_sum=active_nm_sum,
        full_count=full_count,
        samples=samples,
        seen_samples=seen_samples,
        overlimit_seen=overlimit_seen,
        skipped_overlimit=skipped_overlimit,
        dz_nm=args.dz_nm,
    )


def make_checkpoint_extra(
    *,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    vocab: optollama.data.DepthFieldVocab,
    model_config: optollama.model.DepthFieldModelConfig,
    history: list[dict],
) -> dict:
    return {
        "depth_field": {
            "dz_nm": float(args.dz_nm),
            "max_thickness_nm": float(args.max_thickness_nm),
            "depth_bins": optollama.data.depth_bins_for(args.max_thickness_nm, args.dz_nm),
            "output_seq_len": int(args.output_seq_len or cfg["MAX_SEQ_LEN"]),
            "vocab": vocab.to_dict(),
            "representation": "material_depth_field_with_void",
            "filter_overlimit_stacks": not bool(args.keep_overlimit_stacks),
            "eval_mode": str(args.eval_mode),
            "eval_mc_samples": int(args.eval_mc_samples),
            "eval_sampling_steps": args.eval_sampling_steps,
            "eval_temperature": float(args.eval_temperature),
            "eval_top_k": int(args.eval_top_k),
            "eval_deterministic": bool(args.eval_deterministic),
            "eval_tmm_batch_size": int(args.eval_tmm_batch_size),
        },
        "model_config": model_config.to_dict(),
        "config_path": str(args.config),
        "history": history,
    }


def save_depth_checkpoint(
    *,
    path: Path,
    model: optollama.model.DepthFieldDiffusion,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[dict],
    extra: dict,
) -> None:
    train_losses = torch.tensor([item["train"]["loss"] for item in history], dtype=torch.float32)
    val_losses = torch.tensor([record_score(item) for item in history], dtype=torch.float32)
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
    apply_loader_overrides(cfg, args)

    seed = int(args.seed if args.seed is not None else cfg.get("SEED", 0))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    tokens, token_to_idx, idx_to_token, _, _, _, eos_idx, pad_idx, msk_idx = optollama.data.init_tokens(cfg["TOKENS_PATH"])
    vocab = optollama.data.build_depth_field_vocab(tokens, token_to_idx)
    depth_bins = optollama.data.depth_bins_for(args.max_thickness_nm, args.dz_nm)

    train_subset = args.max_train_samples if args.max_train_samples is not None else cfg.get("NUM_SAMPLES_TRAIN")
    val_subset = args.max_val_samples if args.max_val_samples is not None else cfg.get("NUM_SAMPLES_TEST")
    train_ds, train_loader, _ = optollama.data.SpectraDataset.make_loader(cfg, split="train", subset_n=train_subset, ddp=False)
    val_loader = None
    if not args.no_val:
        _, val_loader, _ = optollama.data.SpectraDataset.make_loader(cfg, split="test", subset_n=val_subset, ddp=False)
    eval_mode = str(args.eval_mode).lower()
    tmm_ctx = None
    if val_loader is not None and eval_mode in {"tmm", "both"}:
        tmm_device = resolve_tmm_device(args.eval_tmm_device, device)
        tmm_ctx = optollama.evaluation.simulation.TMMContext.make(cfg=cfg, idx_to_token=idx_to_token, device=tmm_device)

    model_config = optollama.model.DepthFieldModelConfig(
        spectrum_shape=tuple(int(v) for v in example_spectrum(train_ds).shape),
        num_materials=vocab.num_clean_classes,
        depth_bins=depth_bins,
        d_model=int(args.d_model),
        n_blocks=int(args.n_blocks),
        kernel_size=int(args.kernel_size),
        timesteps=int(args.diffusion_steps),
        dropout=float(args.dropout),
    )
    model = optollama.model.DepthFieldDiffusion(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate if args.learning_rate is not None else cfg["LEARNING_RATE"]),
        weight_decay=float(args.weight_decay),
    )

    start_epoch = 0
    history: list[dict] = []
    if args.resume:
        start_epoch_loaded, blob = optollama.utils.load_checkpoint(args.resume, model, optimizer=optimizer, map_location="cpu")
        start_epoch = int(start_epoch_loaded or 0)
        history = list(((blob.get("extra") or {}).get("history") or []))
        print(f"Resumed depth-field checkpoint {args.resume} at epoch {start_epoch}.")

    epochs = int(args.epochs if args.epochs is not None else cfg.get("EPOCHS", 20))
    out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    best_path = out_dir / "depth-field-best.pt"
    last_path = out_dir / "depth-field-last.pt"
    history_path = out_dir / "depth-field-history.json"
    primary_score_name = score_name_for_eval_mode(eval_mode)
    comparable_scores = [score for item in history if (score := comparable_record_score(item, primary_score_name)) is not None]
    best_score = min(comparable_scores, default=float("inf"))
    validate_every_samples = (
        int(args.validate_every_n_train_samples)
        if args.validate_every_n_train_samples is not None
        else int(cfg.get("VALIDATE_EVERY_N_TRAIN_SAMPLES") or 0)
    )
    if val_loader is None:
        validate_every_samples = 0

    def run_validation_trigger(epoch: int, trigger: str, train_metrics: dict, samples_seen_epoch: int) -> None:
        nonlocal best_score
        if val_loader is None:
            return

        denoise_metrics = None
        tmm_metrics = None
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

        record = {
            "epoch": int(epoch),
            "trigger": str(trigger),
            "samples_seen_epoch": int(samples_seen_epoch),
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)

        extra = make_checkpoint_extra(args=args, cfg=cfg, vocab=vocab, model_config=model_config, history=history)
        score = metric_score(val_metrics)
        if score < best_score:
            best_score = score
            save_depth_checkpoint(path=best_path, model=model, optimizer=optimizer, epoch=epoch, history=history, extra=extra)
            print(f"Saved best checkpoint -> {best_path} ({val_metrics.get('score_name', 'score')}={best_score:.6f}, trigger={trigger})")

        optollama.utils.save_as_json(str(history_path), history)

    print(
        "Depth-field diffusion: "
        f"materials={vocab.num_clean_classes - 1}+void, bins={depth_bins}, dz={args.dz_nm:g}nm, "
        f"max={args.max_thickness_nm:g}nm, device={device}, amp={amp_enabled}"
    )
    if validate_every_samples > 0:
        print(f"Mid-epoch validation enabled every {validate_every_samples} seen train samples.")
    if val_loader is not None:
        print(f"Validation mode: {eval_mode}.")

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
        )

        if val_loader is not None:
            run_validation_trigger(
                epoch=epoch,
                trigger="epoch_end",
                train_metrics=train_metrics,
                samples_seen_epoch=int(train_metrics["samples_seen"]),
            )
        else:
            history.append(
                {
                    "epoch": int(epoch),
                    "trigger": "epoch_end",
                    "samples_seen_epoch": int(train_metrics["samples_seen"]),
                    "train": train_metrics,
                }
            )

        extra = make_checkpoint_extra(args=args, cfg=cfg, vocab=vocab, model_config=model_config, history=history)
        if args.save_every > 0 and ((epoch + 1) % int(args.save_every) == 0 or epoch == epochs - 1):
            save_depth_checkpoint(path=last_path, model=model, optimizer=optimizer, epoch=epoch, history=history, extra=extra)
            print(f"Saved last checkpoint -> {last_path}")

        if val_loader is None:
            score = metric_score(train_metrics)
            if score < best_score:
                best_score = score
                save_depth_checkpoint(path=best_path, model=model, optimizer=optimizer, epoch=epoch, history=history, extra=extra)
                print(f"Saved best checkpoint -> {best_path} (train_loss={best_score:.6f})")

        optollama.utils.save_as_json(str(history_path), history)


if __name__ == "__main__":
    main()
