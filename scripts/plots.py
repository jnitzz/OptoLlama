from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch  # if not already imported
from metrics import masked_mae_roi
from utils import save_as_json, wl_mask

# ruff: noqa: N802, N803, N806

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


def token_to_layer(token: str) -> Optional[Tuple[str, float]]:
    """Separate tokens to material and thickness values."""
    if token in SPECIAL_TOKENS or "_" not in token:
        return None
    mat, th = token.split("_", 1)
    num = "".join(ch for ch in th if (ch.isdigit() or ch == "."))
    if not num:
        return None
    return mat, float(num)


def build_thickness_grid(
    token_seqs: Sequence[Sequence[str]],
) -> Tuple[np.ndarray, float, float]:
    """
    Convert stacks into an absolute-thickness grid.

    X-axis represents physical thickness in nm (or your unit).
    Resolution = pixels along the thickness axis.

    No normalization: stack lengths reflect real total thickness.
    """
    n = len(token_seqs)

    # Parse layers and totals
    parsed = []
    totals = []
    for seq in token_seqs:
        layers = []
        tot = 0.0
        for tok in seq:
            layer = token_to_layer(tok)
            if layer:
                layers.append(layer)
                tot += layer[1]
        parsed.append(layers)
        totals.append(tot)

    # Determine global maximum physical thickness so both rows align
    max_total = max(totals) if any(totals) else 1.0
    resolution = int(max_total)
    grid = np.full((n, resolution), -1, dtype=int)

    # Convert nm → pixel index with fixed scaling
    nm_per_pixel = max_total / resolution

    for row, layers in enumerate(parsed):
        pos_nm = 0.0  # physical thickness in nm
        for mat, thick in layers:
            idx = MATERIAL_TO_INDEX.get(mat, -1)
            # convert physical thickness to pixels
            pix_start = int(pos_nm / nm_per_pixel)
            pix_end = int((pos_nm + thick) / nm_per_pixel)
            pix_end = min(resolution, pix_end)
            if idx >= 0:
                grid[row, pix_start:pix_end] = idx
            pos_nm += thick
    return grid, nm_per_pixel, max_total


def plot_samples_clean_NN(
    cfg: Any,
    RAT_pred: Union[np.ndarray, torch.Tensor],  # [3, W] - model prediction
    RAT_tar: Union[np.ndarray, torch.Tensor],  # [3, W] - target
    stack_pred: list[str],
    stack_tar: list[str],
    ACC: float,
    number: int,
    RAT_nn: Union[np.ndarray, torch.Tensor],  # [3, W] - nearest neighbor spectrum
    stack_nn: list[str],  # nearest neighbor stack
    nn_global_id: int,  # global id of NN in train set
    RAT_tar_mean: Union[torch.Tensor, None] = None,
) -> None:
    """Plot target, template and model prediction, their residuals and stacks."""
    # --- to numpy ---------------------------------------------------------
    try:
        RAT_pred = RAT_pred.detach().cpu().numpy()
        RAT_tar = RAT_tar.detach().cpu().numpy()
        RAT_nn = RAT_nn.detach().cpu().numpy()
        if RAT_tar_mean is not None:
            RAT_tar_mean = RAT_tar_mean.detach().cpu().numpy()
    except AttributeError:
        pass

    def _strip(seq: Sequence[str]) -> List[str]:
        return [t for t in seq if t not in SPECIAL_TOKENS]

    stack_pred = _strip(stack_pred)
    stack_tar = _strip(stack_tar)
    stack_nn = _strip(stack_nn)

    wls = np.array(cfg.WAVELENGTHS, dtype=float)
    wl_range = wl_mask(cfg.WAVELENGTHS, cfg.ROI_MIN, cfg.ROI_MAX, "cpu")

    # 3-row layout: spectra, residuals (mini-plot), stacks
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.0, 1])

    ax_spec = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax_spec)
    ax_stack = fig.add_subplot(gs[2])

    # --- spectra ----------------------------------------------------------
    labels = ["R", "A", "T"]
    spec_colors = {
        "R": "#d62728",  # red
        "A": "#2ca02c",  # green
        "T": "#1f77b4",  # blue
    }

    for i, ch in enumerate(labels):
        color = spec_colors[ch]
        # target
        ax_spec.plot(wls, RAT_tar[i], color=color, label=f"Target {ch}", lw=1.5)
        # prediction
        ax_spec.plot(wls, RAT_pred[i], color=color, ls="--", label=f"Pred {ch}", lw=1.4)
        # nearest neighbor
        ax_spec.plot(
            wls,
            RAT_nn[i],
            color=color,
            ls=":",
            label=f"NN {ch}",
            lw=1.2,
            alpha=0.9,
        )
        if RAT_tar_mean is not None:
            ax_spec.plot(wls, RAT_tar_mean[i], color=color, ls=":", label=f"Target mean {ch}")

    # spectrum_mae = float(np.mean(np.abs(RAT_pred - RAT_tar)))
    # nn_mae = float(np.mean(np.abs(RAT_nn - RAT_tar)))
    spectrum_mae = masked_mae_roi(torch.tensor(RAT_pred), torch.tensor(RAT_tar), wl_range).item()
    nn_mae = masked_mae_roi(torch.tensor(RAT_nn), torch.tensor(RAT_tar), wl_range).item()

    ax_spec.set_title(f"Prediction vs Target [{cfg.RUN_NAME}]", fontsize=11)
    ax_spec.set_ylabel("R / A / T", fontsize=10)
    ax_spec.grid(True, alpha=0.3)
    ax_spec.legend(fontsize=8, ncol=3, loc="center right")

    # stats box (now with NN MAE)
    # text_left = "MC samples:\nMAE:\nAccuracy:\nNN MAE:\nNN id:"
    # text_right = f"{number}\n{spectrum_mae:.4f}\n{ACC:.4f}\n{nn_mae:.4f}\n{nn_global_id}"
    text_left = "MC samples:\nMAE:\nNN MAE:\nNN id:"
    text_right = f"{number}\n{spectrum_mae:.4f}\n{nn_mae:.4f}\n{nn_global_id}"
    ax_spec.text(0.83, 0.98, text_left, transform=ax_spec.transAxes, fontsize=9, va="top")
    ax_spec.text(0.98, 0.98, text_right, transform=ax_spec.transAxes, fontsize=9, va="top", ha="right")

    # --- mini-plot: residuals ---------------------------------------------
    residual = RAT_pred - RAT_tar  # [3, W] prediction residual
    residual_nn = RAT_nn - RAT_tar  # [3, W] NN residual

    for i, ch in enumerate(labels):
        color = spec_colors[ch]
        # prediction residual
        ax_res.plot(wls, residual[i], color=color, lw=1.0, label=f"Δ{ch} (pred)")
        # NN residual (dashed)
        ax_res.plot(
            wls,
            residual_nn[i],
            color=color,
            lw=0.9,
            ls="--",
            alpha=0.8,
            label=f"Δ{ch} (NN)",
        )

    ax_res.axhline(0.0, color="k", lw=0.8, ls="--")
    ax_res.set_xlabel("Wavelength [nm]", fontsize=10)
    ax_res.set_ylabel("Residual", fontsize=10)
    ax_res.grid(True, alpha=0.3)
    ax_res.legend(fontsize=7, ncol=3, loc="upper right")

    ax_res.xaxis.set_ticks_position("top")
    plt.setp(ax_res.get_xticklabels(), visible=False)

    # --- thickness-aware stack bars (ABSOLUTE SCALE) ----------------------
    token_seqs = [stack_tar, stack_pred, stack_nn]

    grid, nm_per_pixel, max_total_nm = build_thickness_grid(token_seqs)

    masked = np.ma.masked_less(grid, 0)
    cmap = plt.get_cmap("inferno", len(MATERIAL_ORDER)).copy()
    cmap.set_bad("black")

    ax_stack.imshow(
        masked,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=len(MATERIAL_ORDER) - 1,
    )

    n_rows, n_cols = grid.shape

    ax_stack.set_yticks(range(n_rows))
    ax_stack.set_yticklabels(["Target", "Prediction", "Nearest\nneighbor"][:n_rows])

    # x-axis in absolute thickness
    ticks = np.arange(0, n_cols, 250)
    tick_labels = (ticks * nm_per_pixel).astype(int)
    ax_stack.set_xticks(ticks)
    ax_stack.set_xticklabels(tick_labels)
    ax_stack.set_xlabel("Thickness [nm]")

    # material labels inside bars
    min_px = 0
    for row in range(n_rows):
        vals = grid[row]
        start = 0
        cur = vals[0]
        for col in range(1, n_cols + 1):
            nxt = vals[col] if col < n_cols else None
            if nxt != cur:
                if cur >= 0:
                    w = col - start
                    if w >= min_px:
                        xc = (start + col - 1) / 2
                        yc = row
                        mat = MATERIAL_ORDER[cur]
                        ax_stack.text(
                            xc,
                            yc,
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
    save_str = f"{cfg.PATH_SAVED}/{cfg.TARGET.split('/')[-1][:-5]}" if cfg.TARGET else f"{cfg.PATH_SAVED}/plot_sample"
    plt.savefig(f"{save_str}.pdf", dpi=600, bbox_inches="tight")
    plt.show()

    data = {
        "wls": wls.tolist(),
        "RAT_tarR": RAT_tar[0].tolist(),
        "RAT_tarA": RAT_tar[1].tolist(),
        "RAT_tarT": RAT_tar[2].tolist(),
        "RAT_predR": RAT_pred[0].tolist(),
        "RAT_predA": RAT_pred[1].tolist(),
        "RAT_predT": RAT_pred[2].tolist(),
        "RAT_nnR": RAT_nn[0].tolist(),
        "RAT_nnA": RAT_nn[1].tolist(),
        "RAT_nnT": RAT_nn[2].tolist(),
        "text_left": text_left,
        "text_right": text_right,
        "residualR": residual[0].tolist(),
        "residualA": residual[1].tolist(),
        "residualT": residual[2].tolist(),
        "residual_nnR": residual_nn[0].tolist(),
        "residual_nnA": residual_nn[1].tolist(),
        "residual_nnT": residual_nn[2].tolist(),
        "grid": grid.tolist(),
        "max_total_nm": max_total_nm,
        "token_seqs": token_seqs,
        "nn_global_id": nn_global_id,
        "nn_mae": nn_mae,
    }
    save_as_json(f"{save_str}.json", data)

    df = pd.DataFrame(
        {
            "wls": wls,
            "RAT_tarR": RAT_tar[0],
            "RAT_tarA": RAT_tar[1],
            "RAT_tarT": RAT_tar[2],
            "RAT_predR": RAT_pred[0],
            "RAT_predA": RAT_pred[1],
            "RAT_predT": RAT_pred[2],
            "RAT_nnR": RAT_nn[0],
            "RAT_nnA": RAT_nn[1],
            "RAT_nnT": RAT_nn[2],
            "residualR": residual[0],
            "residualA": residual[1],
            "residualT": residual[2],
            "residual_nnR": residual_nn[0],
            "residual_nnA": residual_nn[1],
            "residual_nnT": residual_nn[2],
        }
    )
    df.to_csv(f"{save_str}.csv", index=False)

    print(f"Saved scatter data to: {save_str}.csv")


# %%
def plot_model_vs_nn_scatter(
    val_results: Sequence[Mapping[str, Any]],
    nn_matches: Sequence[Mapping[str, Any]],
    save_path: str,
    max_points: int = 1000,
    title: Union[str, None] = None,
) -> None:
    """
    Scatter plot comparing model MAE vs nearest-neighbor MAE.

    Parameters
    ----------
    val_results : list[dict]
        Per-sample records from `validate_model`, each containing
        at least:
          - "dataset_index" : int
          - "mae"           : float  (model vs target spectrum)
    nn_matches : list[dict]
        List loaded from your nearest-neighbors JSON, entries like:
          {
            "test_index": <int>,
            "best_train_index": <int>,
            "mae": <float>   # NN vs target spectrum
          }
    max_points : int
        Max number of points to plot (random subset if more).
    title : str | None
        Optional custom plot title.
    """
    # Map: test_index -> NN MAE
    nn_mae_by_test = {int(d["test_index"]): float(d["mae"]) for d in nn_matches}

    model_mae = []
    nn_mae = []
    idxs = []

    for rec in val_results:
        if "dataset_index" not in rec or "mae" not in rec:
            continue

        idx = int(rec["dataset_index"])
        if idx not in nn_mae_by_test:
            continue

        model_mae.append(float(rec["mae"]))  # model vs target MAE (from validate_model) :contentReference[oaicite:1]{index=1}
        nn_mae.append(nn_mae_by_test[idx])  # nearest-train vs target MAE (from NN JSON)
        idxs.append(idx)

    if not model_mae:
        raise ValueError("No overlapping indices between val_results and nn_matches.")

    model_mae = np.array(model_mae, dtype=float)
    nn_mae = np.array(nn_mae, dtype=float)
    idxs = np.array(idxs)

    # Optional downsampling to ~max_points
    n = model_mae.shape[0]
    if n > max_points:
        perm = np.random.permutation(n)[:max_points]
        model_mae = model_mae[perm]
        nn_mae = nn_mae[perm]
        idxs = idxs[perm]
    else:
        perm = np.arange(0, n, 1)

    df = pd.DataFrame(
        {
            "perm_idx": perm,
            "dataset_index": idxs,
            "model_mae": model_mae,
            "nearest_neighbor_mae": nn_mae,
        }
    )
    df.to_csv(f"{save_path}/plot_model_vs_nn_{max_points}.csv", index=False)

    data = {
        "perm_idx": perm.tolist(),
        "dataset_index": idxs.tolist(),
        "model_mae": model_mae.tolist(),
        "nearest_neighbor_mae": nn_mae.tolist(),
    }
    save_as_json(f"{save_path}/plot_model_vs_nn_{max_points}.json", data)
    print(f"Saved scatter data to: {save_path}/plot_model_vs_nn_{max_points}.csv")

    # Scatter
    plt.figure(figsize=(6, 6))
    plt.scatter(nn_mae, model_mae, s=6, alpha=0.5)

    # Identity line y = x
    max_lim = float(max(model_mae.max(), nn_mae.max()) * 1.05)
    plt.plot([0, max_lim], [0, max_lim], "--", linewidth=1)
    plt.xlim(0, max_lim)
    plt.ylim(0, max_lim)

    plt.xlabel("MAE(nearest training spectrum vs target)")
    plt.ylabel("MAE(model prediction vs target)")
    plt.title(title or f"Model vs nearest-neighbor baseline (N={len(model_mae)})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/plot_model_vs_nn_{max_points}.pdf", dpi=600, bbox_inches="tight")
    plt.show()


def plot_mae_trajectory(
    mae_traj: Sequence[float],
    timesteps: Union[Sequence[float], None] = None,
    title: Union[str, None] = None,
) -> None:
    """
    Plot MAE over denoising steps for a single example.

    Parameters
    ----------
    mae_traj : Sequence[float]
        Per-step MAE, e.g. results[i]["mae_traj"] from validate_model.
    timesteps : Sequence[float], optional
        Optional x-axis values. If None, uses step indices [0..T-1].
        (You could pass the actual diffusion times if you want.)
    title : str, optional
        Optional plot title.
    """
    mae_traj = np.asarray(mae_traj, dtype=float)
    if timesteps is None:
        x = np.arange(len(mae_traj))
        x_label = "Denoising step"
    else:
        x = np.asarray(timesteps, dtype=float)
        x_label = "Diffusion time"

    plt.figure(figsize=(6, 4))
    plt.plot(x, mae_traj, marker="o")
    plt.xlabel(x_label)
    plt.ylabel("MAE")
    plt.grid(True, alpha=0.3)
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def build_thickness_grid_from_tokens(
    token_seqs: list[list[str]],
    resolution: int = 400,
) -> np.ndarray:
    """
    Convert stack to plottable thickness grid.

    [n_stacks, resolution], x proportional to *global* max thickness.
    Unused (trailing) pixels stay -1 and will be shown as black.
    """
    n_stacks = len(token_seqs)
    grid = np.full((n_stacks, resolution), -1, dtype=int)

    # first pass: parse layers and total thickness per stack
    parsed_layers: list[list[tuple[str, float]]] = []
    totals: list[float] = []
    for seq in token_seqs:
        layers: list[tuple[str, float]] = []
        for tok in seq:
            layer = token_to_layer(tok)
            if layer is not None:
                layers.append(layer)
        parsed_layers.append(layers)
        totals.append(sum(t for _, t in layers))

    max_total = max(totals) if any(t > 0 for t in totals) else 1.0

    # second pass: paint into grid, scaled by global max_total
    for row, (layers, total) in enumerate(zip(parsed_layers, totals)):
        if not layers or total <= 0:
            continue
        pos = 0
        for mat, thick in layers:
            idx = MATERIAL_TO_INDEX.get(mat, -1)
            width = int(round(thick / max_total * resolution))
            if width <= 0:
                continue
            end = min(resolution, pos + width)
            if idx >= 0:
                grid[row, pos:end] = idx
            pos = end
            if pos >= resolution:
                break
        # trailing cells stay -1 → will be drawn as black

    return grid


def plot_mae_band(
    mae_trajs: Sequence[Sequence[float]],
    save_path: str,
    mode: str = "percentile",
    title: Union[str, None] = None,
) -> None:
    """
    Plot MAE trajectory uncertainty as a shaded band from multiple samples.

    Parameters
    ----------
    mae_trajs : Sequence[Sequence[float]]
        List of trajectories. Each trajectory has shape [steps].
        Example: [ result[i]["mae_traj"] for i in all samples ]
    mode : str
        One of:
            - "percentile" -> shaded 10th–90th percentile
            - "minmax"     -> shaded min–max envelope
            - "std"        -> shaded (mean ± std)
    title : str, optional
        Optional plot title.

    Returns
    -------
    None (shows plot)
    """
    # Convert list to array [N, steps]
    arr = np.asarray(mae_trajs, dtype=float)
    assert arr.ndim == 2, "Expected array shape [N, steps]"

    steps = arr.shape[1]
    x = np.arange(steps)

    mean = arr.mean(axis=0)

    if mode == "percentile":
        lower = np.percentile(arr, 10, axis=0)
        upper = np.percentile(arr, 90, axis=0)
        label = "10–90 percentile"
    elif mode == "minmax":
        lower = arr.min(axis=0)
        upper = arr.max(axis=0)
        label = "min–max"
    elif mode == "std":
        std = arr.std(axis=0)
        lower = mean - std
        upper = mean + std
        label = "mean ± std"
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    # Plot
    plt.figure(figsize=(7, 4))
    plt.plot(x, mean, label="mean MAE")
    plt.fill_between(x, lower, upper, alpha=0.3, label=label)

    plt.xlabel("Denoising step")
    plt.ylabel("MAE")
    plt.grid(True, alpha=0.3)
    if title:
        plt.title(title)

    plt.legend()
    plt.tight_layout()
    plt.show()

    data = {
        "x": x.tolist(),
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "mean": mean.tolist(),
    }
    save_as_json(f"{save_path}/plot_mae_trajs.json", data)
    df = pd.DataFrame(
        {
            "x": x.tolist(),
            "lower": lower.tolist(),
            "upper": upper.tolist(),
            "mean": mean.tolist(),
        }
    )
    df.to_csv(f"{save_path}/plot_mae_trajs.csv", index=False)

    print(f"Saved scatter data to: {save_path}/plot_mae_trajs.csv")
