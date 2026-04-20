from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from optollama.data.spectra import wavelength_mask as wl_mask
from optollama.evaluation.metrics import masked_mae_roi

MATERIAL_ORDER = [
    "Al",
    "Ag",
    "TiN",
    "ZnSe",
    "ZnO",
    "TiO2",
    "Si3N4",
    "Ta2O5",
    "AlN",
    "ZnS",
    "SiO2",
    "Al2O3",
    "MgO",
    "MgF2",
    "HfO2",
    "EVA",
    "Si",
    "ITO",
    "Ge",
]
MATERIAL_TO_INDEX = {m: i for i, m in enumerate(MATERIAL_ORDER)}
SPECIAL_TOKENS = {"<PAD>", "<SOS>", "<EOS>", "<MSK>"}


def _to_numpy(value: Union[np.ndarray, torch.Tensor, Sequence[float]]) -> np.ndarray:
    """Convert plotting inputs to NumPy arrays."""
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def token_to_layer(token: str) -> Optional[tuple[str, float]]:
    """
    Parse a token string into a ``(material, thickness_nm)`` pair.
    """
    if token in SPECIAL_TOKENS or "_" not in token:
        return None
    material, thickness = token.split("_", 1)
    numeric = "".join(ch for ch in thickness if (ch.isdigit() or ch == "."))
    if not numeric:
        return None
    return material, float(numeric)


def build_thickness_grid(
    token_seqs: Sequence[Sequence[str]],
) -> tuple[np.ndarray, float, float]:
    """
    Convert token stacks into an absolute-thickness pixel grid.
    """
    n_rows = len(token_seqs)
    parsed: list[list[tuple[str, float]]] = []
    totals: list[float] = []

    for seq in token_seqs:
        layers: list[tuple[str, float]] = []
        total = 0.0
        for token in seq:
            layer = token_to_layer(token)
            if layer is not None:
                layers.append(layer)
                total += layer[1]
        parsed.append(layers)
        totals.append(total)

    max_total = max(totals) if any(totals) else 1.0
    resolution = max(1, int(round(max_total)))
    nm_per_pixel = max_total / resolution
    grid = np.full((n_rows, resolution), -1, dtype=int)

    for row, layers in enumerate(parsed):
        pos_nm = 0.0
        for material, thickness in layers:
            idx = MATERIAL_TO_INDEX.get(material, -1)
            pix_start = int(pos_nm / nm_per_pixel)
            pix_end = min(resolution, int((pos_nm + thickness) / nm_per_pixel))
            if idx >= 0:
                grid[row, pix_start:pix_end] = idx
            pos_nm += thickness

    return grid, nm_per_pixel, max_total


def plot_sample_comparison(
    wavelengths: Union[np.ndarray, torch.Tensor, Sequence[float]],
    target_spectrum: Union[np.ndarray, torch.Tensor],
    predicted_spectrum: Union[np.ndarray, torch.Tensor],
    predicted_tokens: Sequence[str],
    target_tokens: Sequence[str],
    sample_acc: Optional[float] = None,
    sample_mae: Optional[float] = None,
    mc_samples: Optional[int] = None,
    roi_min: Optional[float] = None,
    roi_max: Optional[float] = None,
    target_mean: Optional[Union[np.ndarray, torch.Tensor]] = None,
    nn_spectrum: Optional[Union[np.ndarray, torch.Tensor]] = None,
    nn_tokens: Optional[Sequence[str]] = None,
    nn_id: Optional[int] = None,
    nn_mae: Optional[float] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot target vs prediction spectra plus token-stack bars.
    """
    target = _to_numpy(target_spectrum)
    predicted = _to_numpy(predicted_spectrum)
    wavelengths_np = _to_numpy(wavelengths).astype(float)
    target_mean_np = _to_numpy(target_mean) if target_mean is not None else None
    nn_np = _to_numpy(nn_spectrum) if nn_spectrum is not None else None

    plot_nn = nn_np is not None and nn_tokens is not None

    roi_mask = None
    if roi_min is not None and roi_max is not None:
        roi_mask = wl_mask(wavelengths_np, roi_min, roi_max, "cpu")

    pred_mae_val = (
        float(masked_mae_roi(torch.tensor(predicted), torch.tensor(target), roi_mask).item())
        if sample_mae is None
        else float(sample_mae)
    )
    nn_mae_val = (
        float(masked_mae_roi(torch.tensor(nn_np), torch.tensor(target), roi_mask).item())
        if plot_nn and nn_mae is None
        else nn_mae
    )

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.0, 1])

    ax_spec = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax_spec)
    ax_stack = fig.add_subplot(gs[2])

    labels = ["R", "A", "T"]
    colors = {"R": "#d62728", "A": "#2ca02c", "T": "#1f77b4"}

    for idx, channel in enumerate(labels):
        color = colors[channel]
        ax_spec.plot(wavelengths_np, target[idx], color=color, label=f"Target {channel}", lw=1.5)
        ax_spec.plot(wavelengths_np, predicted[idx], color=color, ls="--", label=f"Pred {channel}", lw=1.4)
        if plot_nn:
            ax_spec.plot(wavelengths_np, nn_np[idx], color=color, ls=":", label=f"NN {channel}", lw=1.2, alpha=0.9)
        if target_mean_np is not None:
            ax_spec.plot(wavelengths_np, target_mean_np[idx], color=color, ls="-.", label=f"Target mean {channel}", lw=1.0)

    ax_spec.set_title(title or "Prediction vs Target", fontsize=11)
    ax_spec.set_ylabel("R / A / T", fontsize=10)
    ax_spec.grid(True, alpha=0.3)
    ax_spec.legend(fontsize=8, ncol=3, loc="center right")

    text_lines_left = []
    text_lines_right = []
    if mc_samples is not None:
        text_lines_left.append("MC samples:")
        text_lines_right.append(str(mc_samples))
    text_lines_left.append("MAE:")
    text_lines_right.append(f"{pred_mae_val:.4f}")
    if sample_acc is not None:
        text_lines_left.append("Accuracy:")
        text_lines_right.append(f"{float(sample_acc):.4f}")
    if plot_nn and nn_mae_val is not None:
        text_lines_left.append("NN MAE:")
        text_lines_right.append(f"{float(nn_mae_val):.4f}")
    if plot_nn and nn_id is not None:
        text_lines_left.append("NN id:")
        text_lines_right.append(str(int(nn_id)))

    ax_spec.text(0.82, 0.98, "\n".join(text_lines_left), transform=ax_spec.transAxes, fontsize=9, va="top")
    ax_spec.text(0.98, 0.98, "\n".join(text_lines_right), transform=ax_spec.transAxes, fontsize=9, va="top", ha="right")

    residual = predicted - target
    residual_nn = (nn_np - target) if plot_nn else None
    for idx, channel in enumerate(labels):
        color = colors[channel]
        ax_res.plot(wavelengths_np, residual[idx], color=color, lw=1.0, label=f"d{channel} (pred)")
        if residual_nn is not None:
            ax_res.plot(wavelengths_np, residual_nn[idx], color=color, lw=0.9, ls="--", alpha=0.8, label=f"d{channel} (NN)")

    ax_res.axhline(0.0, color="k", lw=0.8, ls="--")
    ax_res.set_xlabel("Wavelength [nm]", fontsize=10)
    ax_res.set_ylabel("Residual", fontsize=10)
    ax_res.grid(True, alpha=0.3)
    ax_res.legend(fontsize=7, ncol=3, loc="upper right")
    ax_res.xaxis.set_ticks_position("top")
    plt.setp(ax_res.get_xticklabels(), visible=False)

    token_seqs = [list(target_tokens), list(predicted_tokens)]
    row_labels = ["Target", "Prediction"]
    if plot_nn:
        token_seqs.append(list(nn_tokens))
        row_labels.append("Nearest\nneighbor")

    grid, nm_per_pixel, _ = build_thickness_grid(token_seqs)
    masked = np.ma.masked_less(grid, 0)
    cmap = plt.get_cmap("inferno", len(MATERIAL_ORDER)).copy()
    cmap.set_bad("black")

    ax_stack.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=len(MATERIAL_ORDER) - 1)
    n_rows, n_cols = grid.shape
    ax_stack.set_yticks(range(n_rows))
    ax_stack.set_yticklabels(row_labels[:n_rows])
    ticks = np.arange(0, n_cols, max(1, n_cols // 8))
    ax_stack.set_xticks(ticks)
    ax_stack.set_xticklabels((ticks * nm_per_pixel).astype(int))
    ax_stack.set_xlabel("Thickness [nm]")

    for row in range(n_rows):
        vals = grid[row]
        start = 0
        cur = vals[0]
        for col in range(1, n_cols + 1):
            nxt = vals[col] if col < n_cols else None
            if nxt != cur:
                if cur >= 0 and col > start:
                    xc = (start + col - 1) / 2
                    mat = MATERIAL_ORDER[cur]
                    ax_stack.text(
                        xc,
                        row,
                        mat,
                        rotation=90,
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white",
                        path_effects=[pe.withStroke(linewidth=1.3, foreground="black")],
                        clip_on=True,
                    )
                start = col
                cur = nxt

    plt.tight_layout()
    return fig


def plot_model_vs_nn_scatter(
    val_results: Sequence[Mapping[str, Any]],
    nn_matches: Sequence[Mapping[str, Any]],
    max_points: int = 1000,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Scatter plot comparing model MAE vs nearest-neighbor MAE.
    """
    nn_mae_by_test = {int(item["test_index"]): float(item["mae"]) for item in nn_matches}

    model_mae: list[float] = []
    nn_mae: list[float] = []

    for rec in val_results:
        if "dataset_index" not in rec or "mae" not in rec:
            continue
        idx = int(rec["dataset_index"])
        if idx not in nn_mae_by_test:
            continue
        model_mae.append(float(rec["mae"]))
        nn_mae.append(float(nn_mae_by_test[idx]))

    if not model_mae:
        raise ValueError("No overlapping indices between inference results and NN matches.")

    model_mae_np = np.asarray(model_mae, dtype=float)
    nn_mae_np = np.asarray(nn_mae, dtype=float)

    if model_mae_np.shape[0] > max_points:
        select = np.random.permutation(model_mae_np.shape[0])[:max_points]
        model_mae_np = model_mae_np[select]
        nn_mae_np = nn_mae_np[select]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.scatter(nn_mae_np, model_mae_np, s=6, alpha=0.5)

    max_lim = float(max(model_mae_np.max(), nn_mae_np.max()) * 1.05)
    ax.plot([0, max_lim], [0, max_lim], "--", linewidth=1)
    ax.set_xlim(0, max_lim)
    ax.set_ylim(0, max_lim)
    ax.set_xlabel("MAE(nearest training spectrum vs target)")
    ax.set_ylabel("MAE(model prediction vs target)")
    ax.set_title(title or f"Model vs nearest-neighbor baseline (N={len(model_mae_np)})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

