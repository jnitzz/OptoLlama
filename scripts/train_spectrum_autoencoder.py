#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import tqdm

import optollama.data
import optollama.model
import optollama.utils


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a spectrum autoencoder on OptoLlama spectra shards.")
    parser.add_argument("--config", type=str, default="configs/optollama.yaml", help="Path to OptoLlama config.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for checkpoints and history.")
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g. cuda or cpu.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Training and validation batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers. Defaults to config value.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional cap for train spectra.")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Optional cap for validation spectra.")
    parser.add_argument("--sharded-loading", action="store_true", help="Force streaming shard loading for this trainer.")
    parser.add_argument("--eager-loading", action="store_true", help="Disable config SHARDED_LOADING for this trainer.")
    parser.add_argument("--latent-dim", type=int, default=128, help="Spectrum latent dimension.")
    parser.add_argument("--hidden-dim", type=int, default=1024, help="MLP hidden dimension.")
    parser.add_argument("--n-hidden", type=int, default=2, help="Number of hidden layers in encoder/decoder.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout in encoder/decoder MLPs.")
    parser.add_argument("--latent-bound", type=float, default=3.0, help="Tanh latent bound. Use <=0 to disable.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--denoise-sigma", type=float, default=0.01, help="Input noise std for denoising training.")
    parser.add_argument("--smoothness-weight", type=float, default=0.0, help="Penalty on decoded wavelength curvature.")
    parser.add_argument("--resume", action="store_true", help="Resume from spectrum_autoencoder-last.pt if present.")
    return parser.parse_args()


def _set_loader_epoch(loader: torch.utils.data.DataLoader, epoch: int) -> None:
    dataset = getattr(loader, "dataset", None)
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)


def _example_spectrum(dataset: torch.utils.data.Dataset) -> torch.Tensor:
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


def _loader_cfg(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.sharded_loading and args.eager_loading:
        raise ValueError("--sharded-loading and --eager-loading are mutually exclusive.")

    loader_cfg = dict(cfg)
    loader_cfg["TRAIN_BATCH_SIZE"] = int(args.batch_size)
    loader_cfg["TEST_BATCH_SIZE"] = int(args.batch_size)
    if args.num_workers is not None:
        loader_cfg["NUM_WORKERS"] = int(args.num_workers)
    if args.eager_loading:
        loader_cfg["SHARDED_LOADING"] = False
    if args.sharded_loading:
        loader_cfg["SHARDED_LOADING"] = True
    return loader_cfg


def _make_loaders(cfg: dict[str, Any], args: argparse.Namespace):
    loader_cfg = _loader_cfg(cfg, args)
    train_limit = args.max_train_samples if args.max_train_samples is not None else cfg.get("NUM_SAMPLES_TRAIN")
    val_limit = args.max_val_samples if args.max_val_samples is not None else cfg.get("NUM_SAMPLES_TEST")
    train_ds, train_loader, _ = optollama.data.SpectraDataset.make_loader(
        loader_cfg,
        split="train",
        subset_n=train_limit,
        ddp=False,
    )
    val_ds, val_loader, _ = optollama.data.SpectraDataset.make_loader(
        loader_cfg,
        split="test",
        subset_n=val_limit,
        ddp=False,
    )
    return train_ds, train_loader, val_ds, val_loader


def _augment_input(clean: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return clean
    return (clean + torch.randn_like(clean) * float(sigma)).clamp(0.0, 1.0)


def _smoothness_loss(recon: torch.Tensor) -> torch.Tensor:
    if recon.size(-1) < 3:
        return recon.new_tensor(0.0)
    d2 = recon[..., 2:] - 2.0 * recon[..., 1:-1] + recon[..., :-2]
    return d2.abs().mean()


def _run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    denoise_sigma: float,
    smoothness_weight: float,
    desc: str,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_mae = 0.0
    total_count = 0

    pbar = tqdm.tqdm(total=len(loader), desc=desc, leave=True)
    for batch in loader:
        clean = batch[0].to(device, dtype=torch.float32, non_blocking=True)
        inputs = _augment_input(clean, denoise_sigma if train else 0.0)

        if train:
            optimizer.zero_grad(set_to_none=True)

        recon, _ = model(inputs)
        mae_loss = F.l1_loss(recon, clean)
        loss = mae_loss
        if smoothness_weight > 0:
            loss = loss + float(smoothness_weight) * _smoothness_loss(recon)

        if train:
            loss.backward()
            optimizer.step()

        n = int(clean.size(0))
        total_loss += float(loss.detach().item()) * n
        total_mae += float(mae_loss.detach().item()) * n
        total_count += n
        pbar.set_postfix(loss=f"{total_loss / max(total_count, 1):.6f}", mae=f"{total_mae / max(total_count, 1):.6f}")
        pbar.update()
    pbar.close()

    return {
        "loss": total_loss / max(total_count, 1),
        "mae": total_mae / max(total_count, 1),
    }


def _checkpoint_blob(
    model: optollama.model.SpectrumAutoencoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    cfg: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "arch": model.arch_config(),
        "epoch": int(epoch),
        "train_mae": float(train_metrics["mae"]),
        "train_loss": float(train_metrics["loss"]),
        "val_mae": float(val_metrics["mae"]),
        "val_loss": float(val_metrics["loss"]),
        "wavelengths": cfg["WAVELENGTHS"].detach().cpu().tolist(),
        "args": vars(args),
    }


def train(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out_dir or Path(cfg["OUTPUT_PATH"]) / "spectrum_autoencoder")
    out_dir.mkdir(parents=True, exist_ok=True)
    last_path = out_dir / "spectrum_autoencoder-last.pt"
    best_path = out_dir / "spectrum_autoencoder-best.pt"
    history_path = out_dir / "spectrum_autoencoder-history.json"

    torch.manual_seed(int(cfg.get("SEED", 0)))
    train_ds, train_loader, _, val_loader = _make_loaders(cfg, args)
    example = _example_spectrum(train_ds)
    width = int(example.size(-1))

    model = optollama.model.SpectrumAutoencoder(
        width=width,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        n_hidden=args.n_hidden,
        dropout=args.dropout,
        latent_bound=args.latent_bound,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    start_epoch = 0
    best_val_mae = float("inf")
    history: list[dict[str, Any]] = []
    if args.resume and last_path.exists():
        blob = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(blob["model_state"], strict=True)
        optimizer.load_state_dict(blob["optimizer_state"])
        start_epoch = int(blob.get("epoch", 0))
        best_val_mae = float(blob.get("best_val_mae", blob.get("val_mae", best_val_mae)))
        if history_path.exists():
            history = optollama.utils.load_as_json(str(history_path))
        print(f"Resumed spectrum autoencoder from {last_path} at epoch {start_epoch}")

    print(
        "Training spectrum autoencoder: "
        f"device={device}, width={width}, latent={args.latent_dim}, "
        f"batch={args.batch_size}, "
        f"sharded={bool(args.sharded_loading or (cfg.get('SHARDED_LOADING', False) and not args.eager_loading))}"
    )

    for epoch in range(start_epoch, int(args.epochs)):
        _set_loader_epoch(train_loader, epoch)
        _set_loader_epoch(val_loader, epoch)

        train_metrics = _run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            denoise_sigma=float(args.denoise_sigma),
            smoothness_weight=float(args.smoothness_weight),
            desc=f"Epoch {epoch + 1}/{args.epochs} train",
        )
        with torch.no_grad():
            val_metrics = _run_epoch(
                model,
                val_loader,
                device,
                optimizer=None,
                denoise_sigma=0.0,
                smoothness_weight=0.0,
                desc=f"Epoch {epoch + 1}/{args.epochs} val",
            )

        row = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
        }
        history.append(row)
        print(
            f"Epoch {epoch + 1}: train_mae={train_metrics['mae']:.6f}, "
            f"val_mae={val_metrics['mae']:.6f}"
        )

        is_best = val_metrics["mae"] < best_val_mae
        if is_best:
            best_val_mae = float(val_metrics["mae"])
        blob = _checkpoint_blob(model, optimizer, epoch + 1, train_metrics, val_metrics, cfg, args)
        blob["best_val_mae"] = best_val_mae
        torch.save(blob, last_path)

        if is_best:
            torch.save(blob, best_path)
            print(f"Saved best spectrum autoencoder -> {best_path}")

        optollama.utils.save_as_json(str(history_path), history)


if __name__ == "__main__":
    args = parse_arguments()
    cfg = optollama.utils.load_config_file(args.config)
    wl_min = int(cfg["WAVELENGTH_MIN"])
    wl_max = int(cfg["WAVELENGTH_MAX"])
    wl_step = int(cfg["WAVELENGTH_STEPS"])
    cfg["WAVELENGTHS"] = torch.arange(wl_min, wl_max + 1, wl_step, dtype=torch.int)
    train(cfg, args)
