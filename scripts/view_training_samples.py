#!/usr/bin/env python

import argparse
import math
from pathlib import Path
from typing import Optional

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import safetensors.torch
import torch

import optollama.data
import optollama.plotting
import optollama.utils


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="View random or explicit training-data stacks and their saved RAT spectra."
    )
    p.add_argument("--config", type=str, default="configs/optollama.yaml", help="Path to config YAML.")
    p.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Config split to inspect.")
    p.add_argument("--input", nargs="*", default=None, help="Optional explicit .safetensors files or directories.")
    p.add_argument("--indices", nargs="*", type=int, default=None, help="Explicit global dataset indices to view.")
    p.add_argument("--num-random", type=int, default=4, help="Number of random samples when --indices is omitted.")
    p.add_argument("--seed", type=int, default=3, help="Random seed for sample selection.")
    p.add_argument("--save-dir", type=str, default="data/output/training_sample_viewer", help="Directory for saved figures.")
    p.add_argument("--show", action="store_true", help="Show matplotlib windows interactively.")
    p.add_argument("--no-save", action="store_true", help="Do not save figures.")
    p.add_argument("--max-stack-labels", type=int, default=60, help="Maximum layer labels drawn on the stack bar.")
    return p.parse_args()


def load_config(path: str) -> dict:
    cfg = optollama.utils.load_config_file(path)
    wl_min = int(cfg["WAVELENGTH_MIN"])
    wl_max = int(cfg["WAVELENGTH_MAX"])
    wl_step = int(cfg["WAVELENGTH_STEPS"])
    cfg["WAVELENGTHS"] = torch.arange(wl_min, wl_max + 1, wl_step, dtype=torch.int)
    return cfg


def resolve_input_shards(cfg: dict, split: str, explicit_paths: Optional[list[str]]) -> list[Path]:
    if explicit_paths:
        raw_paths = explicit_paths
    else:
        prefix = "DATA_PATH_TRAIN" if split == "train" else "DATA_PATH_TEST"
        raw_paths = sorted([cfg[key] for key in cfg if key.startswith(prefix)])

    files: list[Path] = []
    for item in raw_paths:
        path = Path(item)
        if path.is_dir():
            files.extend(sorted(path.glob("*.safetensors")))
        elif path.suffix == ".safetensors":
            files.append(path)
        else:
            raise ValueError(f"Unsupported input path: {path}")

    if not files:
        raise FileNotFoundError("No .safetensors input shards found.")
    return sorted(files, key=lambda path: optollama.data.SpectraDataset.shard_sort_key(str(path.with_suffix(""))))


def load_shard_index(paths: list[Path]) -> tuple[list[dict[str, int | Path]], int]:
    entries: list[dict[str, int | Path]] = []
    offset = 0
    for path in paths:
        with safetensors.torch.safe_open(str(path), framework="pt", device="cpu") as handle:
            n = int(handle.get_tensor("spectra").size(0))
        entries.append({"path": path, "offset": offset, "count": n})
        offset += n
    return entries, offset


def select_indices(args: argparse.Namespace, total: int) -> list[int]:
    if args.indices:
        indices = [int(v) for v in args.indices]
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(args.seed))
        count = min(max(0, int(args.num_random)), total)
        indices = torch.randperm(total, generator=generator)[:count].tolist()

    invalid = [idx for idx in indices if idx < 0 or idx >= total]
    if invalid:
        raise IndexError(f"Indices out of range for dataset with {total} samples: {invalid}")
    return indices


def load_global_sample(entries: list[dict[str, int | Path]], global_idx: int) -> tuple[torch.Tensor, torch.Tensor, Path, int]:
    for entry in entries:
        offset = int(entry["offset"])
        count = int(entry["count"])
        if offset <= global_idx < offset + count:
            local_idx = global_idx - offset
            path = Path(entry["path"])
            with safetensors.torch.safe_open(str(path), framework="pt", device="cpu") as handle:
                spectrum = handle.get_slice("spectra")[local_idx].to(torch.float32)
                stack = handle.get_slice("thin_films")[local_idx].long()
            return spectrum, stack, path, local_idx
    raise IndexError(f"Global index not found: {global_idx}")


def decode_tokens(ids: torch.Tensor, idx_to_token: dict[int, str], eos: int, pad: int, msk: int) -> list[str]:
    tokens: list[str] = []
    for token_id in ids.tolist():
        token_id = int(token_id)
        if token_id == eos:
            break
        if token_id in (pad, msk):
            continue
        tokens.append(idx_to_token[token_id])
    return tokens


def token_to_layer(token: str) -> tuple[str, float] | None:
    if "_" not in token:
        return None
    material, thickness = token.rsplit("_", 1)
    try:
        return material, float(thickness)
    except ValueError:
        return None


def stack_summary(tokens: list[str]) -> tuple[int, float, list[str]]:
    total = 0.0
    materials: list[str] = []
    for token in tokens:
        layer = token_to_layer(token)
        if layer is None:
            continue
        material, thickness = layer
        total += thickness
        materials.append(material)
    return len(materials), total, materials


def print_sample_summary(global_idx: int, shard_path: Path, local_idx: int, tokens: list[str]) -> None:
    n_layers, total_nm, materials = stack_summary(tokens)
    material_counts = {material: materials.count(material) for material in sorted(set(materials))}
    print(f"\nSample {global_idx} ({shard_path.name}:{local_idx})")
    print(f"  layers: {n_layers}, total thickness: {total_nm:.0f} nm")
    print(f"  materials: {material_counts}")
    print(f"  stack: {' | '.join(tokens)}")


def build_stack_grid(tokens: list[str], max_labels: int) -> tuple[np.ndarray, float, float, list[tuple[float, str]]]:
    layers: list[tuple[str, float]] = []
    total = 0.0
    for token in tokens:
        layer = token_to_layer(token)
        if layer is not None:
            layers.append(layer)
            total += layer[1]

    max_total = max(total, 1.0)
    resolution = max(1, int(round(max_total)))
    nm_per_pixel = max_total / resolution
    grid = np.full((1, resolution), -1, dtype=int)
    labels: list[tuple[float, str]] = []

    pos_nm = 0.0
    label_stride = max(1, math.ceil(len(layers) / max(1, max_labels)))
    for i, (material, thickness) in enumerate(layers):
        mat_idx = optollama.plotting.sample_plots.MATERIAL_TO_INDEX.get(material, -1)
        pix_start = int(pos_nm / nm_per_pixel)
        pix_end = min(resolution, max(pix_start + 1, int((pos_nm + thickness) / nm_per_pixel)))
        if mat_idx >= 0:
            grid[0, pix_start:pix_end] = mat_idx
        if i % label_stride == 0:
            labels.append(((pix_start + pix_end - 1) / 2.0, material))
        pos_nm += thickness

    return grid, nm_per_pixel, max_total, labels


def plot_training_sample(
    wavelengths: np.ndarray,
    spectrum: torch.Tensor,
    tokens: list[str],
    title: str,
    max_stack_labels: int,
) -> plt.Figure:
    spec = spectrum.detach().cpu().numpy()
    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.0])
    ax_spec = fig.add_subplot(gs[0])
    ax_stack = fig.add_subplot(gs[1])

    labels = ["R", "A", "T"]
    colors = {"R": "#d62728", "A": "#2ca02c", "T": "#1f77b4"}
    for idx, channel in enumerate(labels):
        ax_spec.plot(wavelengths, spec[idx], color=colors[channel], lw=1.6, label=channel)

    ax_spec.set_title(title)
    ax_spec.set_ylabel("R / A / T")
    ax_spec.set_xlabel("Wavelength [nm]")
    ax_spec.set_ylim(-0.03, 1.03)
    ax_spec.grid(True, alpha=0.3)
    ax_spec.legend(ncol=3, loc="upper right")

    grid, nm_per_pixel, max_total, stack_labels = build_stack_grid(tokens, max_stack_labels)
    masked = np.ma.masked_less(grid, 0)
    cmap = plt.get_cmap("inferno", len(optollama.plotting.sample_plots.MATERIAL_ORDER)).copy()
    cmap.set_bad("black")

    ax_stack.imshow(
        masked,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=len(optollama.plotting.sample_plots.MATERIAL_ORDER) - 1,
    )
    ax_stack.set_yticks([])
    ticks = np.linspace(0, grid.shape[1] - 1, num=min(8, grid.shape[1]), dtype=int)
    ax_stack.set_xticks(ticks)
    ax_stack.set_xticklabels((ticks * nm_per_pixel).astype(int))
    ax_stack.set_xlabel(f"Stack thickness [nm], total {max_total:.0f} nm")

    for x, material in stack_labels:
        ax_stack.text(
            x,
            0,
            material,
            rotation=90,
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            path_effects=[pe.withStroke(linewidth=1.3, foreground="black")],
            clip_on=True,
        )

    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    _, _, idx_to_token, _, _, _, eos_idx, pad_idx, msk_idx = optollama.data.init_tokens(cfg["TOKENS_PATH"])

    shards = resolve_input_shards(cfg, args.split, args.input)
    entries, total = load_shard_index(shards)
    indices = select_indices(args, total)
    wavelengths = cfg["WAVELENGTHS"].detach().cpu().numpy().astype(float)

    save_dir = Path(args.save_dir)
    if not args.no_save:
        save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {len(shards)} shard(s), {total} samples")
    print(f"Selected indices: {indices}")

    for global_idx in indices:
        spectrum, stack_ids, shard_path, local_idx = load_global_sample(entries, global_idx)
        tokens = decode_tokens(stack_ids, idx_to_token, eos=eos_idx, pad=pad_idx, msk=msk_idx)
        print_sample_summary(global_idx, shard_path, local_idx, tokens)

        fig = plot_training_sample(
            wavelengths,
            spectrum,
            tokens,
            title=f"Training sample {global_idx} ({shard_path.name}:{local_idx})",
            max_stack_labels=int(args.max_stack_labels),
        )
        if not args.no_save:
            out_path = save_dir / f"training_sample_{global_idx}.png"
            fig.savefig(out_path, dpi=160)
            print(f"  saved: {out_path}")
        if args.show:
            plt.show()
        else:
            plt.close(fig)


if __name__ == "__main__":
    main()
