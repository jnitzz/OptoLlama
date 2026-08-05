from __future__ import annotations

import argparse
import math
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import tqdm

import optollama.data
import optollama.model
import optollama.utils
from scripts.train_depth_field_diffusion import (
    autocast_context,
    cfg_required,
    ddp_active,
    ddp_rank,
    example_spectrum,
    resolve_amp_dtype,
    resolve_device,
    set_loader_epoch,
)


def parse_args() -> argparse.Namespace:
    """Parse standalone surrogate-training options."""
    parser = argparse.ArgumentParser(description="Train a depth-field-to-spectrum forward surrogate.")
    parser.add_argument("--config", default="configs/depth_field_spectrum_surrogate.yaml")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def surrogate_block(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the validated surrogate configuration block."""
    block = cfg.get("DEPTH_FIELD_SURROGATE") or {}
    if not isinstance(block, dict):
        raise TypeError("DEPTH_FIELD_SURROGATE must be a mapping.")
    return block


def nested(block: dict[str, Any], *path: str, default: Any = None) -> Any:
    """Read an optional nested configuration value."""
    value: Any = block
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def build_surrogate_config(
    cfg: dict[str, Any],
    vocab: optollama.data.DepthFieldVocab,
    spectrum_width: int,
) -> optollama.model.DepthFieldSpectrumSurrogateConfig:
    """Build the serializable model and representation contract."""
    block = surrogate_block(cfg)
    model = nested(block, "MODEL", default={}) or {}
    dz_nm = float(block.get("DZ_NM", nested(cfg.get("DEPTH_FIELD") or {}, "DZ_NM", default=5.0)))
    maximum_nm = float(block.get("MAX_THICKNESS_NM", nested(cfg.get("DEPTH_FIELD") or {}, "MAX_THICKNESS_NM", default=10_000.0)))
    return optollama.model.DepthFieldSpectrumSurrogateConfig(
        num_materials=vocab.num_clean_classes,
        void_id=vocab.void_id,
        depth_bins=optollama.data.depth_bins_for(maximum_nm, dz_nm),
        spectrum_width=int(spectrum_width),
        dz_nm=dz_nm,
        d_model=int(model.get("D_MODEL", 128)),
        conv_dilations=tuple(int(value) for value in model.get("CONV_DILATIONS", [1, 2, 4, 8, 16, 32, 64, 128, 256])),
        kernel_size=int(model.get("KERNEL_SIZE", 7)),
        depth_pool=int(model.get("DEPTH_POOL", 16)),
        decoder_blocks=int(model.get("DECODER_BLOCKS", 2)),
        decoder_heads=int(model.get("DECODER_HEADS", 8)),
        ffn_multiplier=float(model.get("FFN_MULTIPLIER", 2.0)),
        dropout=float(model.get("DROPOUT", 0.0)),
    )


def reduce_metrics(values: torch.Tensor) -> torch.Tensor:
    """Sum a metric vector over all DDP ranks."""
    values = values.detach().clone()
    if ddp_active():
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
    return values


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
    eos_idx: int,
    pad_idx: int,
    msk_idx: int,
    dz_nm: float,
    max_thickness_nm: float,
    derivative_weight: float,
    huber_delta: float,
    grad_clip: float,
    epoch: int,
    epochs: int,
) -> dict[str, float]:
    """Run one train or validation epoch for the forward surrogate."""
    train = optimizer is not None
    model.train(train)
    set_loader_epoch(loader, epoch)
    totals = torch.zeros(6, dtype=torch.float64, device=device)
    progress = tqdm.tqdm(
        loader,
        desc=f"Epoch {epoch + 1}/{epochs} surrogate {'train' if train else 'val'}",
        disable=ddp_rank() != 0,
    )

    for spectra_cpu, stacks_cpu, *_ in progress:
        fields_cpu = optollama.data.rasterize_stack_to_depth_field(
            stacks_cpu,
            idx_to_token,
            vocab,
            dz_nm=dz_nm,
            max_thickness_nm=max_thickness_nm,
            eos_idx=eos_idx,
            pad_idx=pad_idx,
            msk_idx=msk_idx,
        )
        fields = fields_cpu.to(device, non_blocking=True)
        spectra = spectra_cpu.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        context = nullcontext() if train else torch.no_grad()
        with context, autocast_context(device, amp_dtype):
            predicted = model(fields)
            parts = optollama.model.depth_field_spectrum_loss(
                predicted,
                spectra,
                channels=(0, 1, 2),
                derivative_weight=derivative_weight,
                huber_delta=huber_delta,
            )
            loss = parts["loss"]
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"Non-finite surrogate loss at epoch {epoch + 1}.")
        if train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()

        batch_size = int(fields.size(0))
        absolute = (predicted.detach().float() - spectra.float()).abs()
        totals += torch.tensor(
            [
                float(loss.detach().item()) * batch_size,
                float(parts["level_loss"].detach().item()) * batch_size,
                float(parts["derivative_loss"].detach().item()) * batch_size,
                float(absolute.mean().item()) * batch_size,
                float(absolute[:, (0, 2)].mean().item()) * batch_size,
                float(batch_size),
            ],
            dtype=torch.float64,
            device=device,
        )
        if ddp_rank() == 0:
            progress.set_postfix(loss=f"{float(loss.detach().item()):.5f}", mae=f"{float(absolute.mean().item()):.5f}")

    totals = reduce_metrics(totals)
    samples = max(float(totals[5].item()), 1.0)
    return {
        "loss": float(totals[0].item() / samples),
        "level_loss": float(totals[1].item() / samples),
        "derivative_loss": float(totals[2].item() / samples),
        "mae": float(totals[3].item() / samples),
        "rt_mae": float(totals[4].item() / samples),
        "samples": int(round(samples)),
    }


def main() -> None:
    """Train and checkpoint the field-to-spectrum surrogate."""
    args = parse_args()
    cfg = optollama.utils.load_config_file(args.config)
    block = surrogate_block(cfg)
    train_cfg = nested(block, "TRAIN", default={}) or {}
    loss_cfg = nested(block, "LOSS", default={}) or {}

    if args.batch_size is not None:
        cfg["TRAIN_BATCH_SIZE"] = int(args.batch_size)
    cfg["TEST_BATCH_SIZE"] = int(block.get("VAL_BATCH_SIZE", cfg.get("TEST_BATCH_SIZE", cfg["TRAIN_BATCH_SIZE"])))
    if args.out_dir is not None:
        block["OUT_DIR"] = args.out_dir
    out_dir = Path(block.get("OUT_DIR") or Path(cfg["OUTPUT_PATH"]) / "spectrum_surrogate")

    device, local_rank, rank, world_size = optollama.utils.setup_run(cfg, make_dirs=False)
    setup_device = device
    if args.device is not None and world_size == 1:
        device = resolve_device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
    if rank == 0 and device != setup_device:
        print(f"Surrogate effective device={device} (overriding auto-selected {setup_device}).")
    ddp = optollama.utils.is_ddp()
    amp_enabled = bool(train_cfg.get("AMP", True) and device.type == "cuda")
    amp_dtype = resolve_amp_dtype(enabled=amp_enabled, device=device, requested=str(train_cfg.get("AMP_DTYPE", "auto")))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    tokens, token_to_idx, idx_to_token, _, _, _, eos_idx, pad_idx, msk_idx = optollama.data.init_tokens(cfg["TOKENS_PATH"])
    vocab = optollama.data.build_depth_field_vocab(tokens, token_to_idx)
    train_n = args.max_train_samples if args.max_train_samples is not None else cfg_required(cfg, "NUM_SAMPLES_TRAIN")
    val_n = args.max_val_samples if args.max_val_samples is not None else cfg_required(cfg, "NUM_SAMPLES_TEST")
    train_ds, train_loader, _ = optollama.data.SpectraDataset.make_loader(cfg, "train", subset_n=train_n, ddp=ddp)
    _, val_loader, _ = optollama.data.SpectraDataset.make_loader(cfg, "test", subset_n=val_n, ddp=ddp)

    spectrum_width = int(example_spectrum(train_ds).shape[-1])
    model_config = build_surrogate_config(cfg, vocab, spectrum_width)
    model = optollama.model.DepthFieldSpectrumSurrogate(model_config).to(device)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
        )

    learning_rate = float(args.learning_rate if args.learning_rate is not None else train_cfg.get("LEARNING_RATE", 2.0e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=float(train_cfg.get("WEIGHT_DECAY", 0.01)))
    epochs = int(args.epochs if args.epochs is not None else train_cfg.get("EPOCHS", 10))
    start_epoch = 0
    history: list[dict[str, Any]] = []
    resume = args.resume or block.get("RESUME")
    if resume:
        loaded_epoch, blob = optollama.utils.load_checkpoint(
            str(resume), model, optimizer=optimizer, scaler=scaler, map_location="cpu"
        )
        start_epoch = int(loaded_epoch or 0)
        history = list(((blob.get("extra") or {}).get("history") or []))

    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    if ddp:
        torch.distributed.barrier()

    best_mae = min((float(item["val"]["mae"]) for item in history), default=math.inf)
    derivative_weight = float(loss_cfg.get("DERIVATIVE_WEIGHT", 0.25))
    huber_delta = float(loss_cfg.get("HUBER_DELTA", 0.02))
    grad_clip = float(train_cfg.get("GRAD_CLIP", 1.0))
    dz_nm = float(model_config.dz_nm)
    max_thickness_nm = float(model_config.depth_bins) * dz_nm

    for epoch in range(start_epoch, epochs):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            amp_dtype=amp_dtype,
            device=device,
            idx_to_token=idx_to_token,
            vocab=vocab,
            eos_idx=eos_idx,
            pad_idx=pad_idx,
            msk_idx=msk_idx,
            dz_nm=dz_nm,
            max_thickness_nm=max_thickness_nm,
            derivative_weight=derivative_weight,
            huber_delta=huber_delta,
            grad_clip=grad_clip,
            epoch=epoch,
            epochs=epochs,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            scaler=scaler,
            amp_dtype=amp_dtype,
            device=device,
            idx_to_token=idx_to_token,
            vocab=vocab,
            eos_idx=eos_idx,
            pad_idx=pad_idx,
            msk_idx=msk_idx,
            dz_nm=dz_nm,
            max_thickness_nm=max_thickness_nm,
            derivative_weight=derivative_weight,
            huber_delta=huber_delta,
            grad_clip=grad_clip,
            epoch=epoch,
            epochs=epochs,
        )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        extra = {
            "surrogate_config": model_config.to_dict(),
            "depth_field": {"vocab": vocab.to_dict(), "dz_nm": dz_nm, "max_thickness_nm": max_thickness_nm},
            "spectral_grid": {
                "wavelength_min": int(cfg["WAVELENGTH_MIN"]),
                "wavelength_max": int(cfg["WAVELENGTH_MAX"]),
                "wavelength_step": int(cfg["WAVELENGTH_STEPS"]),
            },
            "loss": {
                "type": "smooth_l1_normalized_huber",
                "derivative_weight": derivative_weight,
                "huber_delta": huber_delta,
            },
            "config_path": str(args.config),
            "history": history,
        }
        if rank == 0:
            optollama.utils.save_checkpoint(
                str(out_dir / "depth-field-spectrum-surrogate-last.pt"),
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                train_losses=torch.tensor([item["train"]["loss"] for item in history]),
                test_mae=torch.tensor([item["val"]["mae"] for item in history]),
                extra=extra,
            )
            if val_metrics["mae"] < best_mae:
                best_mae = float(val_metrics["mae"])
                optollama.utils.save_checkpoint(
                    str(out_dir / "depth-field-spectrum-surrogate-best.pt"),
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    train_losses=torch.tensor([item["train"]["loss"] for item in history]),
                    test_mae=torch.tensor([item["val"]["mae"] for item in history]),
                    extra=extra,
                )
            optollama.utils.save_as_json(str(out_dir / "depth-field-spectrum-surrogate-history.json"), history)
            print(f"Surrogate epoch {epoch + 1}: val_mae={val_metrics['mae']:.6f}, best={best_mae:.6f}")


if __name__ == "__main__":
    main()
