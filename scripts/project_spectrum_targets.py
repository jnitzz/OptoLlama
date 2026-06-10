#!/usr/bin/env python

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

import optollama.data
import optollama.model
import optollama.utils


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project target spectra through a trained spectrum autoencoder.")
    parser.add_argument("--config", type=str, default="configs/optollama.yaml", help="Path to OptoLlama config.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Spectrum autoencoder checkpoint.")
    parser.add_argument("--target", action="append", default=None, help="Target CSV/JSON. Can be repeated.")
    parser.add_argument("--target-glob", action="append", default=None, help="Glob for target CSV/JSON files. Can be repeated.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for CSV/PNG projections.")
    parser.add_argument("--device", type=str, default=None, help="Projection device. Defaults to cuda if available.")
    parser.add_argument("--blend", type=float, default=1.0, help="Blend with original target: 1=full AE projection.")
    parser.add_argument("--show", action="store_true", help="Show plots interactively.")
    parser.add_argument("--no-csv", action="store_true", help="Do not write projected CSV files.")
    parser.add_argument("--no-plot", action="store_true", help="Do not write projection plots.")
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    cfg = optollama.utils.load_config_file(path)
    wl_min = int(cfg["WAVELENGTH_MIN"])
    wl_max = int(cfg["WAVELENGTH_MAX"])
    wl_step = int(cfg["WAVELENGTH_STEPS"])
    cfg["WAVELENGTHS"] = torch.arange(wl_min, wl_max + 1, wl_step, dtype=torch.int)
    return cfg


def checkpoint_from_cfg(cfg: dict[str, Any], explicit: str | None) -> str:
    if explicit:
        return explicit
    block = (cfg.get("TARGET_PHYSICALIZE") or {}).get("AE_PROJECT") or {}
    checkpoint = block.get("CHECKPOINT")
    if not checkpoint:
        raise ValueError("Pass --checkpoint or set TARGET_PHYSICALIZE.AE_PROJECT.CHECKPOINT in the config.")
    return str(checkpoint)


def target_specs_from_args(cfg: dict[str, Any], args: argparse.Namespace) -> list[optollama.utils.TargetSpec]:
    targets: list[str] = []
    for pattern in args.target_glob or []:
        matches = sorted(path for path in glob.glob(pattern) if os.path.isfile(path))
        if not matches:
            raise FileNotFoundError(f"--target-glob did not match files: {pattern}")
        targets.extend(matches)
    targets.extend(args.target or [])

    if targets:
        return [
            optollama.utils.TargetSpec(target=target, name=optollama.utils.safe_target_name(target, fallback=f"target_{i}"))
            for i, target in enumerate(targets)
        ]

    specs, _ = optollama.utils.resolve_target_specs(cfg)
    if not specs:
        raise ValueError("No target configured. Use --target, --target-glob, TARGET, TARGETS, or TARGET_GLOB.")
    return specs


def save_projected_csv(path: Path, spectrum: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = spectrum.detach().cpu().numpy().T
    np.savetxt(path, arr, delimiter=",", fmt="%.8g")


def plot_projection(
    path: Path,
    wavelengths: torch.Tensor,
    original: torch.Tensor,
    projected: torch.Tensor,
    *,
    title: str,
    show: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wl = wavelengths.detach().cpu().numpy().astype(float)
    orig = original.detach().cpu().numpy()
    proj = projected.detach().cpu().numpy()
    labels = ["R", "A", "T"]
    colors = {"R": "#d62728", "A": "#2ca02c", "T": "#1f77b4"}

    fig, (ax_spec, ax_res) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for idx, label in enumerate(labels):
        ax_spec.plot(wl, orig[idx], color=colors[label], lw=1.4, label=f"Original {label}")
        ax_spec.plot(wl, proj[idx], color=colors[label], lw=1.2, ls="--", label=f"Projected {label}")
        ax_res.plot(wl, proj[idx] - orig[idx], color=colors[label], lw=1.0, label=f"d{label}")

    ax_spec.set_title(title)
    ax_spec.set_ylabel("R / A / T")
    ax_spec.grid(True, alpha=0.3)
    ax_spec.legend(fontsize=8, ncol=3)
    ax_res.axhline(0.0, color="k", lw=0.8, ls="--")
    ax_res.set_xlabel("Wavelength [nm]")
    ax_res.set_ylabel("Residual")
    ax_res.grid(True, alpha=0.3)
    ax_res.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"Saved projection plot -> {path}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_arguments()
    cfg = load_config(args.config)
    checkpoint = checkpoint_from_cfg(cfg, args.checkpoint)
    out_dir = Path(args.out_dir or Path(cfg["OUTPUT_PATH"]) / "spectrum_autoencoder" / "projected_targets")
    wavelengths = cfg["WAVELENGTHS"]

    for spec in target_specs_from_args(cfg, args):
        if spec.target == "random":
            original = torch.rand((3, wavelengths.numel()), dtype=torch.float32)
        else:
            original = optollama.utils.load_spectra(spec.target, cfg)

        projected, info = optollama.model.project_spectrum_with_autoencoder(original, checkpoint, device=args.device)
        blend = float(args.blend)
        if blend < 1.0:
            projected = (1.0 - blend) * original + blend * projected
            projected = optollama.data.redistribute_mismatch(projected, cfg.get("MISMATCH_FILL_ORDER", "R>T>A"), target_sum=1.0)

        mae = torch.mean(torch.abs(projected - original)).item()
        print(
            f"{spec.name}: AE mae_to_input={info['mae_to_input']:.6f}, "
            f"final_mae={mae:.6f}, latent_dim={info['latent_dim']}"
        )

        if not args.no_csv:
            save_projected_csv(out_dir / f"{spec.name}_ae_projected.csv", projected)
        if not args.no_plot:
            plot_projection(
                out_dir / f"{spec.name}_ae_projected.png",
                wavelengths,
                original,
                projected,
                title=f"{spec.name} AE projection",
                show=args.show,
            )


if __name__ == "__main__":
    main()
