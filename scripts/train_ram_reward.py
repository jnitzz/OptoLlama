#!/usr/bin/env python

import argparse
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F
import tqdm

import optollama.data
import optollama.evaluation
import optollama.evaluation.simulation
import optollama.model
import optollama.utils

from optollama.evaluation.prediction import split_model_sample_output
from scripts import self_improve_lite as lite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "RAM-style reward post-training for OptoLlama. The model samples "
            "complete stacks, TMM scores them with MAE plus a small length "
            "penalty, then learns from re-noised sampled endpoints with "
            "reward-weighted denoising CE."
        )
    )
    parser.add_argument("--config", type=str, default="configs/optollama.yaml", help="Project config YAML.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Current model checkpoint. Defaults to BEST then LAST.")
    parser.add_argument(
        "--reference-checkpoint",
        type=str,
        default=None,
        help="Frozen KL-anchor checkpoint. Defaults to --checkpoint.",
    )
    parser.add_argument("--out-dir", type=str, default=None, help="RAM checkpoint/history output directory.")
    parser.add_argument(
        "--target-source",
        type=str,
        default="train",
        choices=["auto", "config", "file", "synthetic", "train", "test"],
        help="Where target spectra come from.",
    )
    parser.add_argument("--target-file", type=str, default=None, help="Target CSV/JSON for --target-source file.")
    parser.add_argument("--max-targets", type=int, default=1024, help="Target spectra processed per epoch.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. Defaults to config SEED.")
    parser.add_argument("--device", type=str, default=None, help='Execution device, e.g. "cuda", "cuda:0", or "cpu".')
    parser.add_argument("--epochs", type=int, default=1, help="RAM post-training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Target spectra per candidate sampling batch.")
    parser.add_argument("--num-candidates", type=int, default=16, help="Model samples per target.")
    parser.add_argument("--renoise-samples", type=int, default=1, help="Re-noised training views per sampled endpoint.")
    parser.add_argument("--eval-batch-size", type=int, default=512, help="TMM simulation chunk size.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Optimizer LR. Defaults to config LEARNING_RATE.")
    parser.add_argument("--length-penalty", type=float, default=1.0e-4, help="MAE-equivalent penalty per material layer.")
    parser.add_argument("--advantage-temperature", type=float, default=1.0, help="Temperature for exp(advantage / T).")
    parser.add_argument("--max-weight", type=float, default=5.0, help="Clamp for reward CE sample weights.")
    parser.add_argument("--kl-weight", type=float, default=0.05, help="KL anchor weight against the reference model.")
    parser.add_argument(
        "--supervised-weight",
        type=float,
        default=0.0,
        help="Optional supervised CE mixing weight from the configured train split.",
    )
    parser.add_argument(
        "--sampling-steps",
        "--diffusion-steps",
        dest="sampling_steps",
        type=int,
        default=None,
        help="Override diffusion denoising steps used for endpoint sampling.",
    )
    parser.add_argument("--max-seq-len", type=int, default=None, help="Optional model sequence length override.")
    parser.add_argument("--max-emit-len", type=int, default=None, help="Optional sampling EOS cap.")
    parser.add_argument("--roi-min", type=float, default=None, help="MAE ROI lower wavelength. Defaults to config ROI_MIN.")
    parser.add_argument("--roi-max", type=float, default=None, help="MAE ROI upper wavelength. Defaults to config ROI_MAX.")
    parser.add_argument("--save-every", type=int, default=1, help="Save the last checkpoint every N epochs.")
    return parser.parse_args()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def resolve_checkpoint(cfg: dict, explicit: str | None) -> str:
    candidates = [explicit, cfg.get("BEST_CHECKPOINT_PATH"), cfg.get("LAST_CHECKPOINT_PATH")]
    for item in candidates:
        if item and Path(str(item)).exists():
            return str(item)
    raise FileNotFoundError(
        "No checkpoint found. Pass --checkpoint or set BEST_CHECKPOINT_PATH/LAST_CHECKPOINT_PATH to an existing file."
    )


def build_model_from_checkpoint(
    cfg: dict,
    checkpoint: str,
    sample_spectrum: torch.Tensor,
    idx_to_token: dict[int, str],
    msk_idx: int,
    pad_idx: int,
    eos_idx: int,
    device: torch.device,
    init_seed: int,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    torch.manual_seed(int(init_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(init_seed))

    model = optollama.model.build_model(
        model_type=cfg["MODEL"],
        sample_spectrum=sample_spectrum,
        vocab_size=len(idx_to_token),
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
        spectrum_latent=cfg.get("SPECTRUM_LATENT"),
        depth_position=cfg.get("DEPTH_POSITION"),
        depth_rope=cfg.get("DEPTH_ROPE"),
        factored_output=cfg.get("FACTORED_OUTPUT"),
    ).to(device)

    report = optollama.utils.load_checkpoint_weights_for_init(
        checkpoint,
        model,
        map_location="cpu",
        strict=True,
        fallback_filter=True,
    )
    try:
        model.set_max_emit_len(cfg["MAX_EMIT_LEN"])
    except AttributeError:
        pass
    return model.eval(), report


def shard_targets_for_rank(targets: torch.Tensor, rank: int, world_size: int) -> tuple[torch.Tensor, int]:
    if world_size <= 1:
        return targets, int(targets.size(0))

    total = (targets.size(0) // world_size) * world_size
    if total <= 0:
        raise ValueError(f"Need at least world_size={world_size} targets for DDP RAM training.")
    return targets[:total][rank:total:world_size].contiguous(), int(total)


def set_loader_epoch(loader: torch.utils.data.DataLoader, epoch: int) -> None:
    dataset = getattr(loader, "dataset", None)
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)
    sampler = getattr(loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def cycle_loader(loader: torch.utils.data.DataLoader) -> Iterator:
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def sample_and_score_candidates(
    model: torch.nn.Module,
    targets: torch.Tensor,
    num_candidates: int,
    tmm_ctx: optollama.evaluation.simulation.TMMContext,
    eos: int,
    pad: int,
    msk: int,
    roi_mask: torch.Tensor | None,
    eval_batch_size: int,
    length_penalty: float,
) -> dict[str, torch.Tensor]:
    model.eval()
    b = targets.size(0)
    m = max(1, int(num_candidates))
    targets_mc = targets.unsqueeze(1).expand(b, m, *targets.shape[1:]).reshape(b * m, *targets.shape[1:])

    ids_flat, _, _ = split_model_sample_output(model(targets_mc))
    ids_flat, layers = optollama.data.reencode_stacks_for_output(
        ids_flat,
        output_seq_len=ids_flat.size(1),
        eos=eos,
        pad=pad,
        msk=msk,
    )

    mae = lite.simulate_mae_in_chunks(
        ids_flat,
        targets_mc,
        tmm_ctx=tmm_ctx,
        eos=eos,
        pad=pad,
        msk=msk,
        roi_mask=roi_mask,
        eval_batch_size=max(1, int(eval_batch_size)),
    )
    layers = layers.to(device=mae.device, dtype=torch.float32)
    score = mae + float(length_penalty) * layers

    score_grouped = score.view(b, m)
    mean = score_grouped.mean(dim=1, keepdim=True)
    std = score_grouped.std(dim=1, keepdim=True, unbiased=False).clamp_min(1.0e-6)
    advantage = ((mean - score_grouped) / std).reshape(-1)

    return {
        "targets": targets_mc,
        "stacks": ids_flat,
        "mae": mae,
        "score": score,
        "layers": layers,
        "advantage": advantage,
        "best_mae": mae.view(b, m).min(dim=1).values,
        "best_score": score_grouped.min(dim=1).values,
    }


def advantage_weights(advantage: torch.Tensor, temperature: float, max_weight: float) -> torch.Tensor:
    temperature = max(float(temperature), 1.0e-6)
    weights = torch.exp(advantage / temperature)
    if max_weight > 0:
        weights = weights.clamp(max=float(max_weight))
    return weights / weights.mean().clamp_min(1.0e-6)


def renoise_endpoints(
    core_model: torch.nn.Module,
    targets: torch.Tensor,
    stacks: torch.Tensor,
    weights: torch.Tensor,
    renoise_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    k = max(1, int(renoise_samples))
    targets_rep = targets.repeat_interleave(k, dim=0)
    stacks_rep = stacks.repeat_interleave(k, dim=0)
    weights_rep = weights.repeat_interleave(k, dim=0)

    timesteps = core_model._sample_t(stacks_rep)
    betas = core_model.noise(timesteps).reshape(-1, 1)
    mask = torch.full_like(stacks_rep, int(core_model.mask))
    flipped = torch.rand_like(stacks_rep, dtype=torch.float32) < betas
    noised = torch.where(flipped, mask, stacks_rep)
    return targets_rep, noised, stacks_rep, weights_rep, timesteps


def weighted_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    pad_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    vocab_size = logits.size(-1)
    ce = F.cross_entropy(
        logits.reshape(-1, vocab_size).float(),
        targets.reshape(-1),
        ignore_index=int(pad_idx),
        reduction="none",
    ).view_as(targets)
    valid = targets != int(pad_idx)
    per_sample = (ce * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    loss = (per_sample * weights).sum() / weights.sum().clamp_min(1.0e-6)
    return loss, per_sample.detach().mean()


def kl_anchor_loss(
    logits: torch.Tensor,
    ref_logits: torch.Tensor,
    targets: torch.Tensor,
    pad_idx: int,
) -> torch.Tensor:
    log_current = F.log_softmax(logits.float(), dim=-1)
    ref_probs = F.softmax(ref_logits.float(), dim=-1)
    token_kl = F.kl_div(log_current, ref_probs, reduction="none").sum(dim=-1)
    valid = targets != int(pad_idx)
    return (token_kl * valid).sum() / valid.sum().clamp_min(1)


def reduce_epoch_metrics(metrics: dict[str, float], device: torch.device, ddp: bool) -> dict[str, float]:
    keys = sorted(metrics.keys())
    values = torch.tensor([float(metrics[key]) for key in keys], device=device, dtype=torch.float64)
    if ddp:
        torch.distributed.all_reduce(values)
    return {key: float(value.item()) for key, value in zip(keys, values, strict=True)}


def main() -> None:
    args = parse_args()
    cfg = lite.load_config(args.config)
    if cfg["MODEL"] != "optollama":
        raise ValueError("RAM reward training currently requires MODEL: optollama.")

    if args.seed is not None:
        cfg["SEED"] = int(args.seed)
    if args.sampling_steps is not None:
        cfg["DIFFUSION_STEPS"] = int(args.sampling_steps)
    if args.max_seq_len is not None:
        cfg["MAX_SEQ_LEN"] = int(args.max_seq_len)
        if args.max_emit_len is None:
            cfg["MAX_EMIT_LEN"] = int(args.max_seq_len)
    if args.max_emit_len is not None:
        cfg["MAX_EMIT_LEN"] = int(args.max_emit_len)
    if args.device is not None:
        # setup_run owns device selection; this is used for single-process manual runs.
        if not optollama.utils.is_ddp():
            pass

    device, local_rank, rank, world_size = optollama.utils.setup_run(cfg, make_dirs=True)
    if args.device is not None and world_size == 1:
        device = torch.device(args.device)

    ddp = optollama.utils.is_ddp()
    checkpoint = resolve_checkpoint(cfg, args.checkpoint)
    reference_checkpoint = args.reference_checkpoint or checkpoint
    if not Path(reference_checkpoint).exists():
        raise FileNotFoundError(f"--reference-checkpoint does not exist: {reference_checkpoint}")

    tokens, token_to_idx, idx_to_token, _, _, _, eos_idx, pad_idx, msk_idx = optollama.data.init_tokens(cfg["TOKENS_PATH"])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(cfg["SEED"]) + 1009)
    targets_cpu, target_source = lite.load_targets(args, cfg, msk_idx=msk_idx, generator=generator)
    targets_cpu, total_targets = shard_targets_for_rank(targets_cpu, rank=rank, world_size=world_size)

    if rank == 0:
        print(f"RAM reward training from {checkpoint}")
        print(f"Reference checkpoint: {reference_checkpoint}")
        print(
            "Reward score: MAE_ROI + "
            f"{float(args.length_penalty):g} * material_layers; "
            f"targets={total_targets} source={target_source}"
        )

    tmm_ctx = optollama.evaluation.simulation.TMMContext.make(cfg, idx_to_token, device=device)
    roi_mask = lite.build_roi_mask(args, cfg, device)

    init_seed = int(cfg["SEED"]) + 31_337
    model, model_report = build_model_from_checkpoint(
        cfg,
        checkpoint=checkpoint,
        sample_spectrum=targets_cpu[0],
        idx_to_token=idx_to_token,
        msk_idx=msk_idx,
        pad_idx=pad_idx,
        eos_idx=eos_idx,
        device=device,
        init_seed=init_seed,
    )
    reference_model, reference_report = build_model_from_checkpoint(
        cfg,
        checkpoint=reference_checkpoint,
        sample_spectrum=targets_cpu[0],
        idx_to_token=idx_to_token,
        msk_idx=msk_idx,
        pad_idx=pad_idx,
        eos_idx=eos_idx,
        device=device,
        init_seed=init_seed,
    )
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad_(False)

    lite.apply_token_constraints_if_configured(model, cfg, tokens, token_to_idx)
    if rank == 0:
        print(
            "Current load: "
            f"mode={model_report['mode']}, fallback={model_report['fallback_used']}, "
            f"loaded={model_report['loaded_keys']}, skipped={len(model_report['skipped_keys'])}"
        )
        print(
            "Reference load: "
            f"mode={reference_report['mode']}, fallback={reference_report['fallback_used']}, "
            f"loaded={reference_report['loaded_keys']}, skipped={len(reference_report['skipped_keys'])}"
        )
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if torch.device(device).type == "cuda" else None,
            output_device=local_rank if torch.device(device).type == "cuda" else None,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate or cfg.get("LEARNING_RATE", 1.0e-4)))

    supervised_iter = None
    train_loader = None
    if float(args.supervised_weight) > 0.0:
        _, train_loader, _ = optollama.data.SpectraDataset.make_loader(
            cfg,
            split="train",
            subset_n=cfg.get("NUM_SAMPLES_TRAIN"),
            ddp=ddp,
        )
        supervised_iter = cycle_loader(train_loader)

    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg["OUTPUT_PATH"]) / "ram_reward"
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    run_name = str(cfg.get("RUN", "run"))
    last_checkpoint = out_dir / f"optollama-{run_name}-ram-last.pt"
    best_checkpoint = out_dir / f"optollama-{run_name}-ram-best.pt"
    history_path = out_dir / "ram_reward_history.json"

    history: list[dict[str, Any]] = []
    best_score = float("inf")
    batch_size = max(1, int(args.batch_size))
    target_count_local = int(targets_cpu.size(0))

    for epoch in range(int(args.epochs)):
        if train_loader is not None:
            set_loader_epoch(train_loader, epoch)
            supervised_iter = cycle_loader(train_loader)

        epoch_totals = {
            "batches": 0.0,
            "targets": 0.0,
            "candidates": 0.0,
            "loss_sum": 0.0,
            "reward_ce_sum": 0.0,
            "kl_sum": 0.0,
            "supervised_ce_sum": 0.0,
            "mean_mae_sum": 0.0,
            "mean_score_sum": 0.0,
            "mean_best_mae_sum": 0.0,
            "mean_best_score_sum": 0.0,
            "mean_layers_sum": 0.0,
        }

        iterator = range(0, target_count_local, batch_size)
        pbar = tqdm.tqdm(iterator, desc=f"RAM epoch {epoch + 1}/{int(args.epochs)}", disable=(rank != 0))
        for start in pbar:
            end = min(start + batch_size, target_count_local)
            targets = targets_cpu[start:end].to(device, non_blocking=True)

            sampled = sample_and_score_candidates(
                model,
                targets=targets,
                num_candidates=int(args.num_candidates),
                tmm_ctx=tmm_ctx,
                eos=eos_idx,
                pad=pad_idx,
                msk=msk_idx,
                roi_mask=roi_mask,
                eval_batch_size=int(args.eval_batch_size),
                length_penalty=float(args.length_penalty),
            )
            weights = advantage_weights(
                sampled["advantage"],
                temperature=float(args.advantage_temperature),
                max_weight=float(args.max_weight),
            )

            model.train()
            optimizer.zero_grad(set_to_none=True)
            core = unwrap_model(model)
            train_targets, noised, clean, train_weights, timesteps = renoise_endpoints(
                core,
                sampled["targets"],
                sampled["stacks"],
                weights,
                renoise_samples=int(args.renoise_samples),
            )

            logits = model(train_targets, noised, timesteps)
            reward_ce, reward_ce_unweighted = weighted_ce_loss(logits, clean, train_weights, pad_idx=pad_idx)
            total_loss = reward_ce

            kl_loss = torch.zeros((), device=device)
            if float(args.kl_weight) > 0.0:
                with torch.no_grad():
                    ref_logits = reference_model(train_targets, noised, timesteps)
                kl_loss = kl_anchor_loss(logits, ref_logits, clean, pad_idx=pad_idx)
                total_loss = total_loss + float(args.kl_weight) * kl_loss

            supervised_ce = torch.zeros((), device=device)
            if supervised_iter is not None and float(args.supervised_weight) > 0.0:
                sup_batch = next(supervised_iter)
                sup_spectra = sup_batch[0].to(device, non_blocking=True)
                sup_stacks = sup_batch[1].to(device, non_blocking=True)
                sup_logits = model(sup_spectra, sup_stacks)
                supervised_ce = F.cross_entropy(
                    sup_logits.reshape(-1, sup_logits.size(-1)).float(),
                    sup_stacks.reshape(-1),
                    ignore_index=pad_idx,
                )
                total_loss = total_loss + float(args.supervised_weight) * supervised_ce

            total_loss.backward()
            optimizer.step()

            batch_targets = float(targets.size(0))
            batch_candidates = float(sampled["stacks"].size(0))
            epoch_totals["batches"] += 1.0
            epoch_totals["targets"] += batch_targets
            epoch_totals["candidates"] += batch_candidates
            epoch_totals["loss_sum"] += float(total_loss.detach().item())
            epoch_totals["reward_ce_sum"] += float(reward_ce.detach().item())
            epoch_totals["kl_sum"] += float(kl_loss.detach().item())
            epoch_totals["supervised_ce_sum"] += float(supervised_ce.detach().item())
            epoch_totals["mean_mae_sum"] += float(sampled["mae"].mean().item())
            epoch_totals["mean_score_sum"] += float(sampled["score"].mean().item())
            epoch_totals["mean_best_mae_sum"] += float(sampled["best_mae"].mean().item())
            epoch_totals["mean_best_score_sum"] += float(sampled["best_score"].mean().item())
            epoch_totals["mean_layers_sum"] += float(sampled["layers"].mean().item())

            if rank == 0:
                pbar.set_postfix(
                    loss=f"{total_loss.detach().item():.4f}",
                    mae=f"{sampled['mae'].mean().item():.4f}",
                    best=f"{sampled['best_mae'].mean().item():.4f}",
                    layers=f"{sampled['layers'].mean().item():.1f}",
                )

        reduced = reduce_epoch_metrics(epoch_totals, device=torch.device(device), ddp=ddp)
        batches = max(reduced["batches"], 1.0)
        epoch_summary = {
            "epoch": epoch + 1,
            "targets": int(reduced["targets"]),
            "candidates": int(reduced["candidates"]),
            "loss": reduced["loss_sum"] / batches,
            "reward_ce": reduced["reward_ce_sum"] / batches,
            "kl": reduced["kl_sum"] / batches,
            "supervised_ce": reduced["supervised_ce_sum"] / batches,
            "mean_mae": reduced["mean_mae_sum"] / batches,
            "mean_score": reduced["mean_score_sum"] / batches,
            "mean_best_mae": reduced["mean_best_mae_sum"] / batches,
            "mean_best_score": reduced["mean_best_score_sum"] / batches,
            "mean_layers": reduced["mean_layers_sum"] / batches,
        }
        history.append(epoch_summary)

        if rank == 0:
            print(
                "RAM epoch "
                f"{epoch + 1}: loss={epoch_summary['loss']:.6f}, "
                f"mean MAE={epoch_summary['mean_mae']:.6f}, "
                f"best MAE={epoch_summary['mean_best_mae']:.6f}, "
                f"best score={epoch_summary['mean_best_score']:.6f}, "
                f"layers={epoch_summary['mean_layers']:.1f}"
            )
            optollama.utils.save_as_json(str(history_path), history)

            save_last = (epoch + 1) % max(1, int(args.save_every)) == 0 or epoch + 1 == int(args.epochs)
            checkpoint_paths = []
            if save_last:
                checkpoint_paths.append(str(last_checkpoint))
            if epoch_summary["mean_best_score"] < best_score:
                best_score = float(epoch_summary["mean_best_score"])
                checkpoint_paths.append(str(best_checkpoint))

            for path in checkpoint_paths:
                optollama.utils.save_checkpoint(
                    path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    train_losses=torch.tensor([item["loss"] for item in history], dtype=torch.float32),
                    test_mae=torch.tensor([item["mean_best_mae"] for item in history], dtype=torch.float32),
                    extra={
                        "ram_reward": {
                            "source_checkpoint": checkpoint,
                            "reference_checkpoint": reference_checkpoint,
                            "target_source": target_source,
                            "num_candidates": int(args.num_candidates),
                            "renoise_samples": int(args.renoise_samples),
                            "length_penalty": float(args.length_penalty),
                            "kl_weight": float(args.kl_weight),
                            "supervised_weight": float(args.supervised_weight),
                            "advantage_temperature": float(args.advantage_temperature),
                            "max_weight": float(args.max_weight),
                        }
                    },
                )


if __name__ == "__main__":
    optollama.utils.stop_ddp()
    try:
        main()
    finally:
        optollama.utils.stop_ddp()
