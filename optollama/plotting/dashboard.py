from __future__ import annotations

import textwrap

from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np


def wrap_tokens(tokens: Sequence[str], width: int = 60) -> str:
    """Wrap a token sequence for compact text display."""
    return "\n".join(textwrap.wrap(" ".join(tokens), width=width))


def plot_mc_dashboard(
    mae_grid: Union[np.ndarray, Sequence[Sequence[float]]],
    target_spec: Union[np.ndarray, Sequence[Sequence[Sequence[float]]]],
    pred_spec_grid: Optional[Union[np.ndarray, Sequence]] = None,
    pred_tokens_grid: Optional[Sequence[Sequence[Sequence[str]]]] = None,
    wavelengths: Optional[Union[np.ndarray, Sequence[float]]] = None,
    topk: int = 6,
    title: str = "n_targets x mc_samples dashboard",
) -> plt.Figure:
    """
    Plot a static dashboard over the ``(target_variant, mc_sample)`` grid.
    """
    mae_grid_np = np.asarray(mae_grid, dtype=float)
    n_targets, n_mc = mae_grid_np.shape

    target_spec_np = np.asarray(target_spec, dtype=float)
    if target_spec_np.shape[0] == 1 and n_targets > 1:
        target_spec_np = np.repeat(target_spec_np, n_targets, axis=0)

    pred_spec_grid_np = np.asarray(pred_spec_grid, dtype=float) if pred_spec_grid is not None else None
    x_axis = np.arange(target_spec_np.shape[-1]) if wavelengths is None else np.asarray(wavelengths, dtype=float)

    flat = mae_grid_np.reshape(-1)
    topk = min(int(topk), flat.size)
    best_flat_idx = np.argsort(flat)[:topk]
    best_cells = [(idx // n_mc, idx % n_mc) for idx in best_flat_idx]
    row_best_mc = np.argmin(mae_grid_np, axis=1)
    global_best = best_cells[0] if best_cells else (0, 0)

    fig = plt.figure(figsize=(12, 4 + 2.4 * topk))
    gs = fig.add_gridspec(
        nrows=1 + topk,
        ncols=2,
        height_ratios=[2.2] + [1.6] * topk,
        width_ratios=[3.0, 1.4],
    )

    ax_hm = fig.add_subplot(gs[0, :])
    image = ax_hm.imshow(mae_grid_np, aspect="auto")
    fig.colorbar(image, ax=ax_hm, fraction=0.03, pad=0.02)
    ax_hm.set_title(title)
    ax_hm.set_xlabel("mc_sample")
    ax_hm.set_ylabel("target_variant")
    ax_hm.scatter(row_best_mc, np.arange(n_targets), s=30, marker="o", facecolors="none", edgecolors="k", linewidths=1.0)
    ax_hm.scatter([global_best[1]], [global_best[0]], s=90, marker="*", edgecolors="k", linewidths=1.0)
    ax_hm.set_xticks(np.arange(n_mc))
    ax_hm.set_yticks(np.arange(n_targets))

    for row_idx, (target_idx, mc_idx) in enumerate(best_cells, start=1):
        ax_plot = fig.add_subplot(gs[row_idx, 0])
        ax_text = fig.add_subplot(gs[row_idx, 1])
        ax_text.axis("off")

        target_row = target_spec_np[target_idx]
        ax_plot.plot(x_axis, target_row[0], label="R_target")
        ax_plot.plot(x_axis, target_row[1], label="A_target")
        ax_plot.plot(x_axis, target_row[2], label="T_target")

        if pred_spec_grid_np is not None:
            pred_row = pred_spec_grid_np[target_idx, mc_idx]
            ax_plot.plot(x_axis, pred_row[0], label="R_pred")
            ax_plot.plot(x_axis, pred_row[1], label="A_pred")
            ax_plot.plot(x_axis, pred_row[2], label="T_pred")

        ax_plot.set_title(f"Cell (target={target_idx}, mc={mc_idx})  MAE={mae_grid_np[target_idx, mc_idx]:.6g}")
        ax_plot.set_xlabel("wavelength" if wavelengths is not None else "index")
        ax_plot.set_ylabel("value")
        ax_plot.legend(ncol=3, fontsize=8, loc="upper right")

        if pred_tokens_grid is not None:
            tokens = pred_tokens_grid[target_idx][mc_idx]
            text = wrap_tokens(tokens, width=40)
        else:
            text = "(pred_tokens_grid not provided)"
        ax_text.text(0.0, 1.0, text, va="top", ha="left", fontsize=9)

    plt.tight_layout()
    return fig


def plot_mae_trajectory(
    mae_traj: Sequence[float],
    timesteps: Optional[Sequence[float]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot MAE over denoising steps for a single example.
    """
    mae_traj_np = np.asarray(mae_traj, dtype=float)
    x_axis = np.arange(len(mae_traj_np)) if timesteps is None else np.asarray(timesteps, dtype=float)
    xlabel = "Denoising step" if timesteps is None else "Diffusion time"

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(x_axis, mae_traj_np, marker="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MAE")
    ax.grid(True, alpha=0.3)
    if title is not None:
        ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_mae_band(
    mae_trajs: Sequence[Sequence[float]],
    mode: str = "percentile",
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot an uncertainty band over multiple MAE trajectories.
    """
    arr = np.asarray(mae_trajs, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Expected mae_trajs with shape [N, steps].")

    x_axis = np.arange(arr.shape[1])
    mean = arr.mean(axis=0)

    if mode == "percentile":
        lower = np.percentile(arr, 10, axis=0)
        upper = np.percentile(arr, 90, axis=0)
        label = "10-90 percentile"
    elif mode == "minmax":
        lower = arr.min(axis=0)
        upper = arr.max(axis=0)
        label = "min-max"
    elif mode == "std":
        std = arr.std(axis=0)
        lower = mean - std
        upper = mean + std
        label = "mean +/- std"
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    ax.plot(x_axis, mean, label="mean MAE")
    ax.fill_between(x_axis, lower, upper, alpha=0.3, label=label)
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("MAE")
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig

