#!/usr/bin/env python

import argparse
import os
from pathlib import Path
from typing import Any

import torch
import tqdm

import optollama.data
import optollama.model
import optollama.utils


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    p = argparse.ArgumentParser(description="Train the learned OptoLlama world-edit scorer.")
    p.add_argument("--config", type=str, default="configs/world_model.yaml")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--train-subset", type=int, default=None)
    p.add_argument("--test-subset", type=int, default=None)
    p.add_argument("--checkpoint", type=str, default=None)
    return p.parse_args()


def load_config(path: str) -> dict:
    """
    Load config and enrich it with wavelength values.
    """
    cfg = optollama.utils.load_config_file(path)
    wl_min = int(cfg["WAVELENGTH_MIN"])
    wl_max = int(cfg["WAVELENGTH_MAX"])
    wl_step = int(cfg["WAVELENGTH_STEPS"])
    cfg["WAVELENGTHS"] = torch.arange(wl_min, wl_max + 1, wl_step, dtype=torch.int)
    return cfg


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """
    Move a transition batch to the training device.
    """
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def batch_metrics(parts: dict[str, torch.Tensor]) -> dict[str, float]:
    """
    Convert detached loss parts to plain floats.
    """
    return {key: float(value.item()) for key, value in parts.items()}


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    """
    Average metric dictionaries.
    """
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def save_world_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: dict[str, Any],
    metrics: dict[str, float],
) -> None:
    """
    Save a world-model checkpoint.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": int(epoch),
            "config": cfg,
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    """
    Train the world-edit scorer.
    """
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("SEED", 3))
    optollama.utils.set_all_seeds(seed)

    device = torch.device(args.device or cfg.get("WORLD_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))
    _, _, idx_to_token, _, _, _, eos_idx, pad_idx, msk_idx = optollama.data.init_tokens(cfg["TOKENS_PATH"])

    train_ds, train_loader, _ = optollama.data.WorldTransitionDataset.make_loader(
        cfg,
        split="train",
        subset_n=args.train_subset or cfg.get("WORLD_NUM_SAMPLES_TRAIN"),
        ddp=False,
    )
    has_test = any(key.startswith("WORLD_DATA_PATH_TEST") for key in cfg)
    test_loader = None
    if has_test:
        _, test_loader, _ = optollama.data.WorldTransitionDataset.make_loader(
            cfg,
            split="test",
            subset_n=args.test_subset or cfg.get("WORLD_NUM_SAMPLES_TEST"),
            ddp=False,
        )

    sample = train_ds[0] if not isinstance(train_ds, torch.utils.data.Subset) else train_ds.dataset[train_ds.indices[0]]
    spectra_shape = tuple(int(v) for v in sample["target_spectra"].shape)
    model = optollama.model.WorldEditScorer(
        spectra_shape=spectra_shape,
        vocab_size=len(idx_to_token),
        max_stack_depth=int(cfg.get("WORLD_OUTPUT_SEQ_LEN", cfg["MAX_SEQ_LEN"])),
        eos_idx=eos_idx,
        pad_idx=pad_idx,
        msk_idx=msk_idx,
        d_model=int(cfg.get("WORLD_D_MODEL", 256)),
        n_heads=int(cfg.get("WORLD_N_HEADS", 4)),
        stack_layers=int(cfg.get("WORLD_STACK_LAYERS", 2)),
        dropout=float(cfg.get("WORLD_DROPOUT", 0.0)),
    ).to(device)

    lr = float(args.learning_rate if args.learning_rate is not None else cfg.get("WORLD_LEARNING_RATE", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(cfg.get("WORLD_WEIGHT_DECAY", 0.0)))
    epochs = int(args.epochs if args.epochs is not None else cfg.get("WORLD_EPOCHS", 20))
    checkpoint = args.checkpoint or cfg["WORLD_CHECKPOINT_PATH"]
    last_checkpoint = cfg.get("WORLD_LAST_CHECKPOINT_PATH", str(Path(checkpoint).with_name("world-model-last.pt")))
    roi_mask = optollama.data.wavelength_mask(cfg["WAVELENGTHS"], cfg["ROI_MIN"], cfg["ROI_MAX"], device)

    best_val = float("inf")
    for epoch in range(epochs):
        model.train()
        train_rows: list[dict[str, float]] = []
        pbar = tqdm.tqdm(train_loader, desc=f"world epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = model.loss(
                batch,
                spectrum_weight=float(cfg.get("WORLD_SPECTRUM_LOSS_WEIGHT", 0.1)),
                cost_weight=float(cfg.get("WORLD_COST_LOSS_WEIGHT", 1.0)),
                delta_weight=float(cfg.get("WORLD_DELTA_LOSS_WEIGHT", 1.0)),
                roi_mask=roi_mask,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("WORLD_GRAD_CLIP", 1.0)))
            optimizer.step()
            row = batch_metrics(parts)
            train_rows.append(row)
            pbar.set_postfix(loss=f"{row['loss']:.5f}", cost=f"{row['cost_loss']:.5f}", delta=f"{row['delta_loss']:.5f}")

        metrics = {f"train_{key}": value for key, value in mean_metrics(train_rows).items()}

        if test_loader is not None:
            model.eval()
            val_rows: list[dict[str, float]] = []
            with torch.no_grad():
                for batch in test_loader:
                    batch = move_batch(batch, device)
                    _, parts = model.loss(
                        batch,
                        spectrum_weight=float(cfg.get("WORLD_SPECTRUM_LOSS_WEIGHT", 0.1)),
                        cost_weight=float(cfg.get("WORLD_COST_LOSS_WEIGHT", 1.0)),
                        delta_weight=float(cfg.get("WORLD_DELTA_LOSS_WEIGHT", 1.0)),
                        roi_mask=roi_mask,
                    )
                    val_rows.append(batch_metrics(parts))
            metrics.update({f"val_{key}": value for key, value in mean_metrics(val_rows).items()})

        save_world_checkpoint(last_checkpoint, model, optimizer, epoch, cfg, metrics)
        score = metrics.get("val_loss", metrics["train_loss"])
        if score < best_val:
            best_val = score
            save_world_checkpoint(checkpoint, model, optimizer, epoch, cfg, metrics)
            print(f"Saved best world model checkpoint -> {checkpoint} ({score:.6f})")


if __name__ == "__main__":
    main()
