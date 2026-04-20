#!/usr/bin/env python

from __future__ import annotations

import argparse
import os

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

import optollama.data
import optollama.plotting
import optollama.utils


def train_paths_from_cfg(cfg: dict) -> list[str]:
    """Collect configured training shard paths."""
    return sorted([cfg[key] for key in cfg.keys() if key.startswith("DATA_PATH_TRAIN")])


def decode_ids(ids: torch.Tensor, idx_to_token: dict[int, str], eos_idx: int, pad_idx: int, msk_idx: int) -> list[str]:
    """Decode token ids into strings until a special token is encountered."""
    tokens: list[str] = []
    for token_id in ids.tolist():
        token_id = int(token_id)
        if token_id in (eos_idx, pad_idx, msk_idx):
            break
        tokens.append(idx_to_token[token_id])
    return tokens


def parse_arguments() -> argparse.Namespace:
    """Parse plotting CLI arguments."""
    parser = argparse.ArgumentParser(description="Plot OptoLlama inference outputs.")
    parser.add_argument("--config", type=str, default="configs/optollama.yaml", help="Path to YAML config file.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample", help="Plot one target/prediction sample.")
    sample.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    sample.add_argument("--index", type=int, default=None, help="Result index to plot. Defaults to the best-MAE sample.")
    sample.add_argument("--show", action="store_true", help="Display the figure in addition to saving it.")
    sample.add_argument("--save", type=str, default=None, help="Optional output path for the figure.")

    dashboard = subparsers.add_parser("dashboard", help="Plot the MC dashboard over all saved samples.")
    dashboard.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    dashboard.add_argument("--topk", type=int, default=4, help="Number of best grid cells to detail below the heatmap.")
    dashboard.add_argument("--show", action="store_true", help="Display the figure in addition to saving it.")
    dashboard.add_argument("--save", type=str, default=None, help="Optional output path for the figure.")

    nn_scatter = subparsers.add_parser("nn-scatter", help="Plot model-vs-nearest-neighbor MAE scatter.")
    nn_scatter.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    nn_scatter.add_argument("--nn-matches", type=str, default=None, help="Path to nearest-neighbor matches JSON.")
    nn_scatter.add_argument("--max-points", type=int, default=1000, help="Maximum number of points to plot.")
    nn_scatter.add_argument("--show", action="store_true", help="Display the figure in addition to saving it.")
    nn_scatter.add_argument("--save", type=str, default=None, help="Optional output path for the figure.")

    return parser.parse_args()


def load_config(path: str) -> dict:
    """Load config and reconstruct the wavelength tensor."""
    cfg = optollama.utils.load_config_file(path)
    wl_min = int(cfg["WAVELENGTH_MIN"])
    wl_max = int(cfg["WAVELENGTH_MAX"])
    wl_step = int(cfg["WAVELENGTH_STEPS"])
    cfg["WAVELENGTHS"] = torch.arange(wl_min, wl_max + 1, wl_step, dtype=torch.int)
    return cfg


def ensure_plot_dir(cfg: dict) -> str:
    """Create and return the default plot output directory."""
    plot_dir = os.path.join(cfg["OUTPUT_PATH"], "plots")
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir


def save_figure(fig: plt.Figure, path: str, show: bool) -> None:
    """Save the figure, optionally display it, then close it."""
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {path}")
    if show:
        plt.show()
    plt.close(fig)


def sample_command(cfg: dict, args: argparse.Namespace) -> None:
    """Render a single-sample comparison plot from inference outputs."""
    results = optollama.plotting.load_results(cfg["SAMPLES_PATH"])
    if not results:
        raise RuntimeError(f"No inference results found at {cfg['SAMPLES_PATH']}")

    bundle = None
    bundle_path = cfg.get("PLOT_BUNDLE_PATH")
    if bundle_path and os.path.exists(bundle_path):
        bundle = optollama.plotting.load_plot_bundle(bundle_path)

    sample_index = args.index if args.index is not None else optollama.plotting.select_best_result_index(results)
    if sample_index < 0 or sample_index >= len(results):
        raise IndexError(f"Sample index {sample_index} is out of range for {len(results)} results.")

    sample = results[sample_index]
    wavelengths = (
        bundle.wavelengths
        if bundle is not None and bundle.wavelengths is not None
        else cfg["WAVELENGTHS"].detach().cpu().numpy().astype(float)
    )
    nn_spectrum = None
    nn_tokens = None
    nn_id = None
    nn_mae = None

    if cfg.get("PLOT_SAMPLE_WITH_NN", False):
        from template_matching import find_best_train_for_target, load_train_sample_by_global_id

        _, _, idx_to_token, _, _, _, eos_idx, pad_idx, msk_idx = optollama.data.init_tokens(cfg["TOKENS_PATH"])
        plot_device = cfg.get("PLOT_NN_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        roi_mask = optollama.data.spectra.wavelength_mask(
            cfg["WAVELENGTHS"],
            cfg.get("ROI_MIN"),
            cfg.get("ROI_MAX"),
            plot_device,
        )
        nn_result = find_best_train_for_target(
            torch.tensor(sample["rat_target"]),
            train_paths=train_paths_from_cfg(cfg),
            train_chunk_size=int(cfg.get("PLOT_NN_CHUNK_SIZE", 4096)),
            device=plot_device,
            wl_range=roi_mask,
        )
        nn_spectrum, nn_ids, _, _ = load_train_sample_by_global_id(
            nn_result["best_global_index"],
            train_paths=train_paths_from_cfg(cfg),
        )
        nn_tokens = decode_ids(nn_ids, idx_to_token, eos_idx, pad_idx, msk_idx)
        nn_id = nn_result["best_global_index"]
        nn_mae = nn_result["best_mae"]
        print(f"NN match for sample {sample_index}: train index {nn_id}, MAE={nn_mae:.6f}")

    fig = optollama.plotting.plot_sample_comparison(
        wavelengths=wavelengths,
        target_spectrum=np.asarray(sample["rat_target"], dtype=np.float32),
        predicted_spectrum=np.asarray(sample["rat_pred"], dtype=np.float32),
        predicted_tokens=sample.get("stack_pred_tokens", []),
        target_tokens=sample.get("stack_target_tokens", []),
        sample_acc=sample.get("acc"),
        sample_mae=sample.get("mae"),
        mc_samples=cfg.get("MC_SAMPLES"),
        roi_min=cfg.get("ROI_MIN"),
        roi_max=cfg.get("ROI_MAX"),
        nn_spectrum=nn_spectrum,
        nn_tokens=nn_tokens,
        nn_id=nn_id,
        nn_mae=nn_mae,
        title=f"Sample {sample_index}",
    )

    plot_dir = ensure_plot_dir(cfg)
    save_path = args.save or os.path.join(plot_dir, f"sample_{sample_index}.pdf")
    save_figure(fig, save_path, args.show)


def dashboard_command(cfg: dict, args: argparse.Namespace) -> None:
    """Render the MC dashboard plot from the saved plot bundle."""
    bundle_path = cfg.get("PLOT_BUNDLE_PATH")
    if not bundle_path or not os.path.exists(bundle_path):
        raise FileNotFoundError(
            "Plot bundle not found. Run inference first so it can write the compressed plotting bundle."
        )

    bundle = optollama.plotting.load_plot_bundle(bundle_path)
    if bundle.mae_grid is None:
        raise RuntimeError(f"No mae_grid found in plot bundle {bundle_path}")

    results = optollama.plotting.load_results(cfg["SAMPLES_PATH"])
    target_spec = optollama.plotting.results_target_spectra(results)

    pred_tokens_grid = None
    if bundle.ids_grid is not None:
        _, _, idx_to_token, _, _, _, eos_idx, pad_idx, msk_idx = optollama.data.init_tokens(cfg["TOKENS_PATH"])
        pred_tokens_grid = optollama.plotting.build_pred_tokens_grid(
            bundle.ids_grid,
            idx_to_token,
            eos_idx,
            pad_idx,
            msk_idx,
        )

    wavelengths = (
        bundle.wavelengths
        if bundle.wavelengths is not None
        else cfg["WAVELENGTHS"].detach().cpu().numpy().astype(float)
    )

    fig = optollama.plotting.plot_mc_dashboard(
        mae_grid=bundle.mae_grid,
        target_spec=target_spec,
        pred_spec_grid=bundle.pred_spectra_grid,
        pred_tokens_grid=pred_tokens_grid,
        wavelengths=wavelengths,
        topk=args.topk,
        title=f"{cfg['MODEL']} {cfg['RUN']} dashboard",
    )

    plot_dir = ensure_plot_dir(cfg)
    save_path = args.save or os.path.join(plot_dir, "dashboard.pdf")
    save_figure(fig, save_path, args.show)


def nn_scatter_command(cfg: dict, args: argparse.Namespace) -> None:
    """Render a model-vs-nearest-neighbor scatter plot."""
    nn_matches_path = args.nn_matches or cfg.get("NN_MATCHES_PATH")
    if not nn_matches_path:
        raise ValueError("No NN matches path provided. Set --nn-matches or NN_MATCHES_PATH in the config.")
    if not os.path.exists(nn_matches_path):
        raise FileNotFoundError(f"NN matches file not found: {nn_matches_path}")

    results = optollama.plotting.load_results(cfg["SAMPLES_PATH"])
    nn_matches = optollama.utils.load_as_json(nn_matches_path)

    fig = optollama.plotting.plot_model_vs_nn_scatter(
        results,
        nn_matches,
        max_points=args.max_points,
        title=f"{cfg['MODEL']} {cfg['RUN']} vs nearest-neighbor baseline",
    )

    plot_dir = ensure_plot_dir(cfg)
    save_path = args.save or os.path.join(plot_dir, "nn_scatter.pdf")
    save_figure(fig, save_path, args.show)


def main() -> None:
    """Entry point for plotting inference outputs."""
    args = parse_arguments()
    cfg = load_config(args.config or "configs/optollama.yaml")

    if args.command == "sample":
        sample_command(cfg, args)
    elif args.command == "dashboard":
        dashboard_command(cfg, args)
    elif args.command == "nn-scatter":
        nn_scatter_command(cfg, args)
    else:
        raise ValueError(f"Unknown plotting command: {args.command}")


if __name__ == "__main__":
    main()
