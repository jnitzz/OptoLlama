from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

import optollama.model
import optollama.utils


def parse_args() -> argparse.Namespace:
    """Parse generated-field audit options."""
    parser = argparse.ArgumentParser(
        description="Audit a depth-field spectrum surrogate against exact spectra saved during validation."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--include-all-mc", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--fail-above-rt-mae",
        type=float,
        default=None,
        help="Exit with an error when mean held-out R/T MAE exceeds this value.",
    )
    parser.add_argument("--save", default=None)
    return parser.parse_args()


def resolve_device(value: str | None) -> torch.device:
    """Resolve an explicit device or choose the fastest available device."""
    if value is not None:
        device = torch.device(value)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {value!r} requested, but CUDA is unavailable.")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def expand_sample_paths(values: list[str]) -> list[Path]:
    """Expand validation sample files and directories into JSON paths."""
    paths: list[Path] = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.json")))
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(path)
    if not paths:
        raise FileNotFoundError("No validation sample JSON files found.")
    return paths


def iter_sample_entries(blob: dict[str, Any], *, include_all_mc: bool):
    """Yield selected results and optionally every recorded MC candidate."""
    for result_index, result in enumerate(blob.get("results") or []):
        if isinstance(result, dict):
            yield result_index, None, result
            if include_all_mc:
                for mc_index, candidate in enumerate(result.get("all_mc") or []):
                    if isinstance(candidate, dict):
                        yield result_index, mc_index, candidate


def field_from_runs(
    runs: list[dict[str, Any]],
    *,
    material_to_id: dict[str, int],
    void_id: int,
    depth_bins: int,
    dz_nm: float,
) -> torch.Tensor:
    """Rasterize compact physical runs into the surrogate depth-field contract."""
    field = torch.full((depth_bins,), int(void_id), dtype=torch.long)
    cursor = 0
    for run in runs:
        material = str(run["material"])
        if material not in material_to_id:
            raise KeyError(f"Unknown material {material!r} in field_runs.")
        bins = max(0, int(round(float(run["thickness_nm"]) / float(dz_nm))))
        end = min(cursor + bins, depth_bins)
        field[cursor:end] = int(material_to_id[material])
        cursor = end
        if cursor >= depth_bins:
            break
    return field


def summarize(values: torch.Tensor) -> dict[str, float]:
    """Return robust scalar summary statistics."""
    values = values.float().cpu()
    return {
        "mean": float(values.mean().item()),
        "median": float(values.median().item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "max": float(values.max().item()),
    }


def main() -> None:
    """Compare surrogate predictions with stored exact TMM spectra."""
    args = parse_args()
    if args.batch_size <= 0 or args.max_samples <= 0:
        raise ValueError("--batch-size and --max-samples must be positive.")
    if args.fail_above_rt_mae is not None and args.fail_above_rt_mae < 0.0:
        raise ValueError("--fail-above-rt-mae must be non-negative.")

    device = resolve_device(args.device)
    model, metadata = optollama.model.load_depth_field_spectrum_surrogate(args.checkpoint, device=device)
    model.eval()
    config = model.config
    vocab = ((metadata.get("depth_field") or {}).get("vocab") or {})
    material_names = list(vocab.get("material_names") or [])
    if len(material_names) != config.num_materials:
        raise RuntimeError("Checkpoint material metadata does not match the surrogate configuration.")
    material_to_id = {name: index for index, name in enumerate(material_names)}

    sample_paths = expand_sample_paths(args.samples)
    records: list[dict[str, Any]] = []
    for path in sample_paths:
        blob = optollama.utils.load_as_json(str(path))
        summary = blob.get("summary") or {}
        classes = ((summary.get("depth_field") or {}).get("classes") or material_names)
        if list(classes) != material_names:
            raise RuntimeError(f"Material ordering in {path} does not match the surrogate checkpoint.")
        for result_index, mc_index, entry in iter_sample_entries(blob, include_all_mc=args.include_all_mc):
            runs = entry.get("field_runs")
            exact = entry.get("pred_spectra")
            if not isinstance(runs, list) or exact is None:
                continue
            exact_tensor = torch.as_tensor(exact, dtype=torch.float32)
            if tuple(exact_tensor.shape) != (3, config.spectrum_width):
                raise RuntimeError(
                    f"Unexpected pred_spectra shape {tuple(exact_tensor.shape)} in {path}; "
                    f"expected {(3, config.spectrum_width)}."
                )
            records.append(
                {
                    "file": str(path),
                    "result_index": int(result_index),
                    "mc_index": mc_index,
                    "field": field_from_runs(
                        runs,
                        material_to_id=material_to_id,
                        void_id=config.void_id,
                        depth_bins=config.depth_bins,
                        dz_nm=config.dz_nm,
                    ),
                    "exact": exact_tensor,
                }
            )
            if len(records) >= args.max_samples:
                break
        if len(records) >= args.max_samples:
            break
    if not records:
        raise RuntimeError("No entries with field_runs and pred_spectra were found.")

    predictions: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(records), args.batch_size):
            fields = torch.stack([item["field"] for item in records[start : start + args.batch_size]]).to(device)
            predictions.append(model(fields).cpu())
    predicted = torch.cat(predictions, dim=0)
    exact = torch.stack([item["exact"] for item in records])
    absolute = (predicted - exact).abs()
    per_sample = absolute.mean(dim=(1, 2))
    per_sample_rt = absolute[:, (0, 2)].mean(dim=(1, 2))
    per_sample_derivative = (predicted.diff(dim=-1) - exact.diff(dim=-1)).abs().mean(dim=(1, 2))
    conservation = (predicted.sum(dim=1) - 1.0).abs().amax(dim=1)

    report = {
        "checkpoint": str(Path(args.checkpoint)),
        "sample_files": [str(path) for path in sample_paths],
        "samples": len(records),
        "include_all_mc": bool(args.include_all_mc),
        "mae_rat": summarize(per_sample),
        "mae_rt": summarize(per_sample_rt),
        "mae_by_channel": {
            name: summarize(absolute[:, index].mean(dim=1))
            for index, name in enumerate(("R", "A", "T"))
        },
        "derivative_mae_rat": summarize(per_sample_derivative),
        "conservation_max": float(conservation.max().item()),
        "worst": [
            {
                "file": records[index]["file"],
                "result_index": records[index]["result_index"],
                "mc_index": records[index]["mc_index"],
                "mae_rat": float(per_sample[index].item()),
                "mae_rt": float(per_sample_rt[index].item()),
            }
            for index in torch.argsort(per_sample, descending=True)[: min(20, len(records))].tolist()
        ],
    }
    print(
        f"Surrogate audit: n={len(records)}, RAT MAE={report['mae_rat']['mean']:.6f}, "
        f"RT MAE={report['mae_rt']['mean']:.6f}, RAT p95={report['mae_rat']['p95']:.6f}, "
        f"conservation_max={report['conservation_max']:.3e}"
    )
    if args.save:
        optollama.utils.save_as_json(args.save, report)
        print(f"Saved surrogate audit -> {args.save}")
    if args.fail_above_rt_mae is not None and report["mae_rt"]["mean"] > float(args.fail_above_rt_mae):
        raise RuntimeError(
            f"Surrogate RT MAE {report['mae_rt']['mean']:.6f} exceeds "
            f"the required {float(args.fail_above_rt_mae):.6f}."
        )


if __name__ == "__main__":
    main()
