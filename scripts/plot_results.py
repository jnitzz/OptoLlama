#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
import re

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

import optollama.data
import optollama.plotting
import optollama.utils


SAMPLE_STAMP_RE = re.compile(r"(?:^|-)(\d{6}-\d{4})(?:-samples)?\.json$")


def train_paths_from_cfg(cfg: dict) -> list[str]:
    """Collect configured training shard paths."""
    return sorted([cfg[key] for key in cfg.keys() if key.startswith("DATA_PATH_TRAIN")])


def plot_nn_cache_block(cfg: dict, plot_device: str | torch.device) -> dict:
    """Build cached nearest-neighbor settings for sample plots."""
    cache_path = cfg.get("PLOT_NN_CACHE_PATH")
    if cache_path is None:
        cache_path = str(Path(cfg["OUTPUT_PATH"]) / "plot_nn_cache.json")
    return {
        "SOURCE_PATHS": train_paths_from_cfg(cfg),
        "CHUNK_SIZE": int(cfg.get("PLOT_NN_CHUNK_SIZE", 4096)),
        "DEVICE": str(plot_device),
        "METRIC_ROI_MIN": cfg.get("ROI_MIN", float(cfg["WAVELENGTHS"].min())),
        "METRIC_ROI_MAX": cfg.get("ROI_MAX", float(cfg["WAVELENGTHS"].max())),
        "METRIC_CHANNELS": cfg.get("PLOT_NN_METRIC_CHANNELS", ["R", "A", "T"]),
        "CACHE_ENABLED": bool(cfg.get("PLOT_NN_CACHE_ENABLED", True)),
        "CACHE_PATH": cache_path,
    }


def load_train_ids_from_nn_match(match: dict) -> torch.Tensor:
    """Load the token sequence for a cached nearest-neighbor match."""
    from safetensors.torch import safe_open

    with safe_open(str(match["file"]), framework="pt", device="cpu") as handle:
        if "thin_films" not in handle.keys():
            raise KeyError(f"{match['file']} must contain a 'thin_films' tensor.")
        thin_films = handle.get_tensor("thin_films")
    return thin_films[int(match["local_index"])]


def decode_ids(ids: torch.Tensor, idx_to_token: dict[int, str], eos_idx: int, pad_idx: int, msk_idx: int) -> list[str]:
    """Decode token ids into strings until a special token is encountered."""
    tokens: list[str] = []
    for token_id in ids.tolist():
        token_id = int(token_id)
        if token_id in (eos_idx, pad_idx, msk_idx):
            break
        tokens.append(idx_to_token[token_id])
    return tokens


def continuous_layers_to_plot_tokens(sample: dict) -> list[str]:
    """Return token-like strings with continuous thicknesses for stack plots."""
    layers = sample.get("stack_pred_continuous") or []
    tokens: list[str] = []
    for layer in layers:
        material = layer.get("material")
        thickness = layer.get("thickness_nm")
        if material is None or thickness is None:
            continue
        tokens.append(f"{material}_{float(thickness):.3g}")
    return tokens


def field_runs_to_plot_tokens(sample: dict) -> list[str]:
    """Return token-like strings from depth-field material runs."""
    runs = sample.get("field_runs") or []
    tokens: list[str] = []
    for run in runs:
        material = run.get("material")
        thickness = run.get("thickness_nm")
        if material is None or thickness is None:
            continue
        tokens.append(f"{material}_{float(thickness):.3g}")
    return tokens


def sample_target_spectrum(sample: dict) -> np.ndarray:
    """Return target spectrum from either token-model or depth-field outputs."""
    if "rat_target" in sample:
        return np.asarray(sample["rat_target"], dtype=np.float32)
    if "target_spectra" in sample:
        return np.asarray(sample["target_spectra"], dtype=np.float32)
    raise KeyError("Sample has neither 'rat_target' nor 'target_spectra'. Re-run inference with --record-spectra.")


def sample_pred_spectrum(sample: dict) -> np.ndarray:
    """Return predicted spectrum from either token-model or depth-field outputs."""
    if "rat_pred" in sample:
        return np.asarray(sample["rat_pred"], dtype=np.float32)
    if "pred_spectra" in sample:
        return np.asarray(sample["pred_spectra"], dtype=np.float32)
    raise KeyError("Sample has neither 'rat_pred' nor 'pred_spectra'. Re-run inference with --record-spectra.")


def sample_pred_tokens(sample: dict) -> list[str]:
    """Return predicted stack tokens from any supported inference output."""
    return (
        continuous_layers_to_plot_tokens(sample)
        or field_runs_to_plot_tokens(sample)
        or list(sample.get("stack_pred_tokens", []))
        or list(sample.get("tokens", []))
    )


def load_sample_results(path: str) -> list[dict]:
    """Load old flat results or new depth-field wrapped results."""
    payload = optollama.plotting.load_results(path)
    if isinstance(payload, dict) and "results" in payload:
        results = payload["results"]
    else:
        results = payload
    if not isinstance(results, list):
        raise TypeError(f"Expected a result list or a wrapper with 'results' at {path}.")
    return results


def _latest_path(paths: list[Path]) -> Path | None:
    """Return the newest path, using filename timestamp order before mtime."""
    if not paths:
        return None
    return sorted(paths, key=lambda path: (path.name, path.stat().st_mtime_ns), reverse=True)[0]


def _resolve_samples_file(path: str) -> str:
    """Resolve explicit or timestamped sample result paths."""
    sample_path = Path(path)
    if sample_path.is_file():
        return str(sample_path)
    if sample_path.is_dir():
        candidates = list(sample_path.glob("*samples*.json"))
        latest = _latest_path(candidates)
        if latest is not None:
            return str(latest)
        raise FileNotFoundError(f"No sample JSON matching '*samples*.json' found in {sample_path}.")

    parent = sample_path.parent
    if parent.exists():
        patterns = []
        if sample_path.stem == "samples":
            patterns.append("samples-*.json")
        elif sample_path.stem.endswith("-samples"):
            prefix = sample_path.stem[: -len("-samples")]
            patterns.append(f"{prefix}-*-samples{sample_path.suffix}")
        patterns.append(sample_path.name)

        candidates: list[Path] = []
        for pattern in patterns:
            candidates.extend(path for path in parent.glob(pattern) if path.is_file())
        latest = _latest_path(candidates)
        if latest is not None:
            return str(latest)

    return str(sample_path)


def _sample_stamp(path: Path) -> str | None:
    """Return a YYMMDD-HHMM sample timestamp from a result filename."""
    match = SAMPLE_STAMP_RE.search(path.name)
    return None if match is None else match.group(1)


def _sample_plot_filename(sample_path: str, sample_index: int) -> str:
    """Build the default sample plot filename from the sample result timestamp."""
    stamp = _sample_stamp(Path(sample_path)) or optollama.utils.make_run_stamp()
    return f"sample_{sample_index}_{stamp}.pdf"


def _latest_sample_entries_by_folder(
    cfg: dict,
    target_selector: str | None,
    samples_root: str | None = None,
) -> list[tuple[optollama.utils.TargetSpec, dict]]:
    """Discover target folders and use each folder's newest timestamped samples file."""
    root = Path(samples_root) if samples_root else Path(cfg["OUTPUT_PATH"])
    if not root.is_dir():
        return []

    latest_by_name: dict[str, tuple[Path, optollama.utils.TargetSpec]] = {}
    for target_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        spec = optollama.utils.TargetSpec(target=str(target_dir), name=target_dir.name)
        if target_selector is not None and not _target_matches(spec, target_selector):
            continue
        candidates: list[Path] = []
        for sample_file in target_dir.glob("*samples*.json"):
            stamp = _sample_stamp(sample_file)
            if stamp is not None:
                candidates.append(sample_file)
        latest = _latest_path(candidates)
        if latest is not None:
            latest_by_name[spec.name] = (latest, spec)

    entries: list[tuple[optollama.utils.TargetSpec, dict]] = []
    for name in sorted(latest_by_name):
        sample_file, spec = latest_by_name[name]
        target_dir = sample_file.parent
        target_cfg = dict(cfg)
        target_cfg["TARGET_NAME"] = spec.name
        target_cfg["OUTPUT_PATH"] = str(target_dir)
        target_cfg["SAMPLES_PATH"] = str(sample_file)
        target_cfg["GRID_PATH"] = str(target_dir / "grid.json")
        target_cfg["IDS_PATH"] = str(target_dir / "ids.json")
        target_cfg["PLOT_BUNDLE_PATH"] = str(target_dir / "plot-bundle.npz")
        entries.append((spec, target_cfg))

    return entries


def samples_path(cfg: dict, args: argparse.Namespace) -> str:
    """Return the sample JSON path selected by CLI or config."""
    return _resolve_samples_file(str(args.samples_path or cfg["SAMPLES_PATH"]))


def plot_bundle_path(cfg: dict, explicit: str | None = None) -> str | None:
    """Return a plot bundle path, accepting either a file or containing folder."""
    raw_path = explicit or cfg.get("PLOT_BUNDLE_PATH")
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if path.is_dir():
        latest = _latest_path([candidate for candidate in path.glob("*plot-bundle*.npz") if candidate.is_file()])
        return None if latest is None else str(latest)
    return str(path)


def _args_for_target_artifacts(args: argparse.Namespace, spec: optollama.utils.TargetSpec) -> argparse.Namespace:
    """Resolve artifact-folder overrides for one target in a multi-target run."""
    copied = argparse.Namespace(**vars(args))
    if args.samples_path:
        samples_root = Path(args.samples_path)
        target_samples_dir = samples_root / spec.name
        if samples_root.is_dir() and target_samples_dir.is_dir():
            copied.samples_path = str(target_samples_dir)
    plot_bundle = getattr(args, "plot_bundle", None)
    if plot_bundle:
        bundle_root = Path(plot_bundle)
        target_bundle_dir = bundle_root / spec.name
        if bundle_root.is_dir() and target_bundle_dir.is_dir():
            copied.plot_bundle = str(target_bundle_dir)
    return copied


def _should_discover_target_folders(cfg: dict, args: argparse.Namespace) -> bool:
    """Return whether plotting should discover per-target result folders."""
    samples_override = getattr(args, "samples_path", None)
    if samples_override is not None:
        return Path(samples_override).is_dir()
    return bool(getattr(args, "target", None)) or optollama.utils.has_multi_target_config(cfg)


def parse_arguments() -> argparse.Namespace:
    """Parse plotting CLI arguments."""
    parser = argparse.ArgumentParser(description="Plot OptoLlama inference outputs.")
    parser.add_argument("--config", type=str, default="configs/optollama.yaml", help="Path to YAML config file.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample", help="Plot one target/prediction sample.")
    sample.add_argument("--config", type=str, default=argparse.SUPPRESS, help="Path to YAML config file.")
    sample.add_argument("--index", type=int, default=None, help="Result index to plot. Defaults to the best-MAE sample.")
    sample.add_argument("--target", type=str, default=None, help="Optional target name/stem/path to plot in multi-target mode.")
    sample.add_argument("--samples-path", type=str, default=None, help="Override config SAMPLES_PATH.")
    sample.add_argument("--no-nn", action="store_true", help="Disable optional nearest-neighbor overlay.")
    sample.add_argument("--show", action="store_true", help="Display the figure in addition to saving it.")
    sample.add_argument("--save", type=str, default=None, help="Output file for one target, output directory for multiple targets.")

    dashboard = subparsers.add_parser("dashboard", help="Plot the MC dashboard over all saved samples.")
    dashboard.add_argument("--config", type=str, default=argparse.SUPPRESS, help="Path to YAML config file.")
    dashboard.add_argument("--topk", type=int, default=4, help="Number of best grid cells to detail below the heatmap.")
    dashboard.add_argument("--mae", choices=("native", "common"), default="native", help="MAE grid to display.")
    dashboard.add_argument("--target", type=str, default=None, help="Optional target name/stem/path to plot in multi-target mode.")
    dashboard.add_argument("--samples-path", type=str, default=None, help="Override config SAMPLES_PATH.")
    dashboard.add_argument("--plot-bundle", type=str, default=None, help="Override config PLOT_BUNDLE_PATH.")
    dashboard.add_argument("--show", action="store_true", help="Display the figure in addition to saving it.")
    dashboard.add_argument("--save", type=str, default=None, help="Output file for one target, output directory for multiple targets.")

    nn_scatter = subparsers.add_parser("nn-scatter", help="Plot model-vs-nearest-neighbor MAE scatter.")
    nn_scatter.add_argument("--config", type=str, default=argparse.SUPPRESS, help="Path to YAML config file.")
    nn_scatter.add_argument("--nn-matches", type=str, default=None, help="Path to nearest-neighbor matches JSON.")
    nn_scatter.add_argument("--samples-path", type=str, default=None, help="Override config SAMPLES_PATH.")
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
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {path}")
    if show:
        plt.show()
    plt.close(fig)


def _target_matches(spec: optollama.utils.TargetSpec, query: str) -> bool:
    """Return whether a target selector matches a resolved target."""
    query_lower = query.lower()
    choices = {
        spec.name.lower(),
        Path(spec.target).stem.lower(),
        str(spec.target).lower(),
    }
    return query_lower in choices


def _plot_target_entries(
    cfg: dict,
    target_selector: str | None,
) -> tuple[list[tuple[optollama.utils.TargetSpec | None, dict]], bool]:
    """Return target-specific plotting configs when multi-target mode is configured."""
    entries, multi_target = optollama.utils.target_cfgs(cfg)
    if not entries:
        return [(None, cfg)], False
    if not multi_target:
        if target_selector is not None and not _target_matches(entries[0][0], target_selector):
            raise ValueError(f"Configured target did not match --target {target_selector!r}.")
        return [(entries[0][0], entries[0][1])], False

    if target_selector is not None:
        entries = [(spec, target_cfg) for spec, target_cfg in entries if _target_matches(spec, target_selector)]
        if not entries:
            raise ValueError(f"No configured target matched --target {target_selector!r}.")

    return entries, True


def _target_batch_plot_path(
    save_arg: str | None,
    spec: optollama.utils.TargetSpec,
    filename: str,
    target_count: int,
) -> str | None:
    """Return an explicit target-batch plot path when --save was provided."""
    if save_arg is None:
        return None

    save_path = Path(save_arg)
    if save_path.suffix:
        if target_count == 1:
            return str(save_path)
        raise ValueError("--save must be a directory when plotting multiple targets.")

    return str(save_path / f"{spec.name}_{filename}")


def _sample_one(
    cfg: dict,
    args: argparse.Namespace,
    *,
    title_prefix: str | None = None,
    save_path: str | None = None,
) -> None:
    """Render a single-sample comparison plot from inference outputs."""
    path = samples_path(cfg, args)
    results = load_sample_results(path)
    if not results:
        raise RuntimeError(f"No inference results found at {path}")

    bundle = None
    bundle_path = plot_bundle_path(cfg, getattr(args, "plot_bundle", None))
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

    if cfg.get("PLOT_SAMPLE_WITH_NN", False) and not bool(getattr(args, "no_nn", False)):
        _, _, idx_to_token, _, _, _, eos_idx, pad_idx, msk_idx = optollama.data.init_tokens(cfg["TOKENS_PATH"])
        plot_device = cfg.get("PLOT_NN_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        nn_match = optollama.data.find_nearest_training_spectrum_cached(
            torch.tensor(sample_target_spectrum(sample)),
            cfg,
            plot_nn_cache_block(cfg, plot_device),
            cfg["WAVELENGTHS"],
            device=plot_device,
        )
        nn_spectrum = nn_match["spectrum"]
        nn_ids = load_train_ids_from_nn_match(nn_match)
        nn_tokens = decode_ids(nn_ids, idx_to_token, eos_idx, pad_idx, msk_idx)
        nn_id = nn_match["global_index"]
        nn_mae = nn_match["mae"]
        cache = nn_match.get("cache") or {}
        cache_note = f", cache={'hit' if cache.get('hit') else 'miss'}" if cache else ""
        print(f"NN match for sample {sample_index}: train index {nn_id}, MAE={nn_mae:.6f}{cache_note}")

    fig = optollama.plotting.plot_sample_comparison(
        wavelengths=wavelengths,
        target_spectrum=sample_target_spectrum(sample),
        predicted_spectrum=sample_pred_spectrum(sample),
        predicted_tokens=sample_pred_tokens(sample),
        target_tokens=sample.get("stack_target_tokens", []),
        sample_acc=sample.get("acc"),
        sample_mae=sample.get("mae"),
        sample_mae_common=sample.get("mae_common"),
        mc_samples=cfg.get("MC_SAMPLES"),
        roi_min=cfg.get("ROI_MIN"),
        roi_max=cfg.get("ROI_MAX"),
        conditioning_spectrum=(
            np.asarray(sample["rat_conditioning"], dtype=np.float32)
            if "rat_conditioning" in sample
            else None
        ),
        nn_spectrum=nn_spectrum,
        nn_tokens=nn_tokens,
        nn_id=nn_id,
        nn_mae=nn_mae,
        title=f"{title_prefix} sample {sample_index}" if title_prefix else f"Sample {sample_index}",
    )

    if save_path is None and args.save is None:
        default_filename = _sample_plot_filename(path, sample_index)
        if cfg.get("TARGET_NAME"):
            save_path = os.path.join(str(cfg["OUTPUT_PATH"]), default_filename)
        else:
            save_path = os.path.join(ensure_plot_dir(cfg), default_filename)
    else:
        save_path = save_path or args.save
    save_figure(fig, save_path, args.show)


def sample_command(cfg: dict, args: argparse.Namespace) -> None:
    """Render one sample plot, or one plot per configured target."""
    entries = (
        _latest_sample_entries_by_folder(cfg, args.target, args.samples_path)
        if _should_discover_target_folders(cfg, args)
        else []
    )
    multi_target = bool(entries)
    if entries:
        print(f"Plotting latest samples per target folder: {len(entries)} targets.")
    else:
        entries, multi_target = _plot_target_entries(cfg, args.target)
    if not multi_target:
        _sample_one(entries[0][1], args)
        return

    target_count = len(entries)
    for spec, target_cfg in entries:
        save_path = _target_batch_plot_path(args.save, spec, "sample.pdf", target_count)
        target_args = _args_for_target_artifacts(args, spec)
        _sample_one(target_cfg, target_args, title_prefix=spec.name, save_path=save_path)


def _dashboard_one(
    cfg: dict,
    args: argparse.Namespace,
    *,
    title_prefix: str | None = None,
    save_path: str | None = None,
) -> None:
    """Render the MC dashboard plot from the saved plot bundle."""
    bundle_path = plot_bundle_path(cfg, args.plot_bundle)
    if not bundle_path or not os.path.exists(bundle_path):
        raise FileNotFoundError(
            "Plot bundle not found. Run inference first so it can write the compressed plotting bundle."
        )

    bundle = optollama.plotting.load_plot_bundle(bundle_path)
    mae_grid = bundle.mae_common_grid if args.mae == "common" else bundle.mae_grid
    if mae_grid is None:
        raise RuntimeError(f"No {args.mae} MAE grid found in plot bundle {bundle_path}")

    results = load_sample_results(samples_path(cfg, args))
    target_spec = optollama.plotting.results_target_spectra(results)
    conditioning_spec = optollama.plotting.results_conditioning_spectra(results)

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
        mae_grid=mae_grid,
        target_spec=target_spec,
        conditioning_spec=conditioning_spec,
        pred_spec_grid=bundle.pred_spectra_grid,
        pred_tokens_grid=pred_tokens_grid,
        wavelengths=wavelengths,
        topk=args.topk,
        title=(
            f"{cfg['MODEL']} {cfg['RUN']} {title_prefix} dashboard ({args.mae} MAE)"
            if title_prefix
            else f"{cfg['MODEL']} {cfg['RUN']} dashboard ({args.mae} MAE)"
        ),
    )

    if save_path is None and args.save is None:
        if cfg.get("TARGET_NAME"):
            save_path = os.path.join(str(cfg["OUTPUT_PATH"]), "dashboard.pdf")
        else:
            save_path = os.path.join(ensure_plot_dir(cfg), "dashboard.pdf")
    else:
        save_path = save_path or args.save
    save_figure(fig, save_path, args.show)


def dashboard_command(cfg: dict, args: argparse.Namespace) -> None:
    """Render one dashboard, or one dashboard per configured target."""
    entries = (
        _latest_sample_entries_by_folder(cfg, args.target, args.samples_path)
        if _should_discover_target_folders(cfg, args)
        else []
    )
    multi_target = bool(entries)
    if entries:
        print(f"Plotting dashboards for latest samples per target folder: {len(entries)} targets.")
    else:
        entries, multi_target = _plot_target_entries(cfg, args.target)
    if not multi_target:
        _dashboard_one(entries[0][1], args)
        return

    target_count = len(entries)
    for spec, target_cfg in entries:
        save_path = _target_batch_plot_path(args.save, spec, "dashboard.pdf", target_count)
        target_args = _args_for_target_artifacts(args, spec)
        _dashboard_one(target_cfg, target_args, title_prefix=spec.name, save_path=save_path)


def nn_scatter_command(cfg: dict, args: argparse.Namespace) -> None:
    """Render a model-vs-nearest-neighbor scatter plot."""
    nn_matches_path = args.nn_matches or cfg.get("NN_MATCHES_PATH")
    if not nn_matches_path:
        raise ValueError("No NN matches path provided. Set --nn-matches or NN_MATCHES_PATH in the config.")
    if not os.path.exists(nn_matches_path):
        raise FileNotFoundError(f"NN matches file not found: {nn_matches_path}")

    results = load_sample_results(samples_path(cfg, args))
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
