#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import tqdm

import optollama.data
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
    val_losses = torch.tensor(
        [item.get("val", item["train"])["loss"] for item in history],
        dtype=torch.float32,
    )
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
    best_score = min([item.get("val", item["train"])["loss"] for item in history], default=float("inf"))

    print(
        "Depth-field diffusion: "
        f"materials={vocab.num_clean_classes - 1}+void, bins={depth_bins}, dz={args.dz_nm:g}nm, "
        f"max={args.max_thickness_nm:g}nm, device={device}, amp={amp_enabled}"
    )

    for epoch in range(start_epoch, epochs):
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
        )

        val_metrics = None
        if val_loader is not None:
            val_metrics = run_epoch(
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

        record = {"epoch": int(epoch), "train": train_metrics}
        if val_metrics is not None:
            record["val"] = val_metrics
        history.append(record)

        extra = make_checkpoint_extra(args=args, cfg=cfg, vocab=vocab, model_config=model_config, history=history)
        if args.save_every > 0 and ((epoch + 1) % int(args.save_every) == 0 or epoch == epochs - 1):
            save_depth_checkpoint(path=last_path, model=model, optimizer=optimizer, epoch=epoch, history=history, extra=extra)
            print(f"Saved last checkpoint -> {last_path}")

        score = float((val_metrics or train_metrics)["loss"])
        if score < best_score:
            best_score = score
            save_depth_checkpoint(path=best_path, model=model, optimizer=optimizer, epoch=epoch, history=history, extra=extra)
            print(f"Saved best checkpoint -> {best_path} (score={best_score:.6f})")

        optollama.utils.save_as_json(str(history_path), history)


if __name__ == "__main__":
    main()
