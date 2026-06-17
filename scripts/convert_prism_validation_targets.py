#!/usr/bin/env python

import argparse
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.ipc as ipc
import safetensors.torch
import torch
import tqdm

import optollama.data
import optollama.utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert PRISM validation Arrow files into OptoLlama target-only "
            "safetensor datasets. PRISM spectra are assumed to be [R,T] over "
            "400-1100 nm in 10 nm steps; output spectra are [R,A,T] on the "
            "configured OptoLlama wavelength grid."
        )
    )
    parser.add_argument("--config", type=str, default="configs/optollama_realistic_07.yaml", help="OptoLlama config.")
    parser.add_argument("--source", type=str, default="data/targets/prism", help="Root folder containing PRISM validation Arrow files.")
    parser.add_argument("--out-dir", type=str, default="data/prism_optollama", help="Output root for converted datasets.")
    parser.add_argument("--shard-size", type=int, default=10000, help="Rows per output safetensors shard.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional limit on PRISM validation files.")
    parser.add_argument("--max-rows-per-file", type=int, default=None, help="Optional per-file row limit for smoke tests.")
    parser.add_argument("--save-dtype", choices=["float32", "float16"], default="float32", help="Saved spectra dtype.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output dataset folders.")
    parser.add_argument(
        "--spectra-order",
        choices=["RT", "TR"],
        default="RT",
        help="Order of PRISM's two spectra channels. Default assumes first channel R, second T.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    cfg = optollama.utils.load_config_file(path)
    cfg["WAVELENGTHS"] = torch.arange(
        int(cfg["WAVELENGTH_MIN"]),
        int(cfg["WAVELENGTH_MAX"]) + 1,
        int(cfg["WAVELENGTH_STEPS"]),
        dtype=torch.float32,
    )
    return cfg


def validation_arrow_files(source: Path) -> list[Path]:
    files = sorted(source.glob("**/val/*.arrow"))
    if not files:
        raise FileNotFoundError(f"No PRISM validation Arrow files found under {source}")
    return files


def dataset_name(source: Path, arrow_file: Path) -> str:
    rel = arrow_file.parent.parent.relative_to(source)
    return "_".join(rel.parts).replace(" ", "_")


def read_arrow_table(path: Path) -> pa.Table:
    with pa.memory_map(str(path), "r") as source:
        try:
            return ipc.open_file(source).read_all()
        except Exception:
            source.seek(0)
            return ipc.open_stream(source).read_all()


def interpolate_spectra(
    spectra: torch.Tensor,
    source_wavelengths: torch.Tensor,
    target_wavelengths: torch.Tensor,
) -> torch.Tensor:
    """Linearly interpolate [N,3,Wsrc] spectra, edge-holding outside source range."""
    spectra = spectra.to(torch.float32)
    source = source_wavelengths.to(dtype=torch.float32)
    target = target_wavelengths.to(dtype=torch.float32)

    target_clamped = target.clamp(float(source[0]), float(source[-1]))
    idx_hi = torch.searchsorted(source, target_clamped).clamp(1, source.numel() - 1)
    idx_lo = idx_hi - 1
    src_lo = source.index_select(0, idx_lo)
    src_hi = source.index_select(0, idx_hi)
    denom = (src_hi - src_lo).clamp_min(torch.finfo(torch.float32).eps)
    weight = ((target_clamped - src_lo) / denom).view(1, 1, -1)

    lo = spectra.index_select(-1, idx_lo)
    hi = spectra.index_select(-1, idx_hi)
    out = lo * (1.0 - weight) + hi * weight

    exact = (target[:, None] == source[None, :]).any(dim=1)
    if exact.any():
        exact_target_idx = exact.nonzero(as_tuple=False).squeeze(1)
        exact_source_idx = torch.searchsorted(source, target.index_select(0, exact_target_idx))
        out[:, :, exact_target_idx] = spectra.index_select(-1, exact_source_idx)

    return out


def prism_spectra_to_rat(
    raw_spectra: torch.Tensor,
    spectra_order: str,
) -> torch.Tensor:
    if raw_spectra.dim() != 2 or raw_spectra.size(1) != 142:
        raise ValueError(f"Expected PRISM spectra shape [N,142], got {tuple(raw_spectra.shape)}")
    channels = raw_spectra.to(torch.float32).view(raw_spectra.size(0), 2, 71)
    if spectra_order == "RT":
        r, t = channels[:, 0], channels[:, 1]
    else:
        t, r = channels[:, 0], channels[:, 1]
    a = (1.0 - r - t).clamp(0.0, 1.0)
    return torch.stack([r.clamp(0.0, 1.0), a, t.clamp(0.0, 1.0)], dim=1)


def write_shard(out_path: Path, spectra: torch.Tensor, stacks: torch.Tensor, save_dtype: str) -> None:
    dtype = torch.float16 if save_dtype == "float16" else torch.float32
    safetensors.torch.save_file(
        {
            "spectra": spectra.to(dtype).contiguous(),
            "thin_films": stacks.long().contiguous(),
        },
        str(out_path),
    )


def convert_file(
    arrow_file: Path,
    source_root: Path,
    out_root: Path,
    cfg: dict[str, Any],
    msk_idx: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    name = dataset_name(source_root, arrow_file)
    dataset_dir = out_root / name
    if dataset_dir.exists() and any(dataset_dir.glob("*.safetensors")):
        if not args.overwrite:
            raise FileExistsError(f"Output already exists for {name}: {dataset_dir}. Use --overwrite.")
        for file in dataset_dir.glob("*.safetensors"):
            file.unlink()
    dataset_dir.mkdir(parents=True, exist_ok=True)

    table = read_arrow_table(arrow_file)
    total_rows = table.num_rows
    if args.max_rows_per_file is not None:
        total_rows = min(total_rows, int(args.max_rows_per_file))

    prism_wavelengths = torch.arange(400, 1100 + 1, 10, dtype=torch.float32)
    target_wavelengths = cfg["WAVELENGTHS"].to(torch.float32)
    stack = torch.full((int(args.shard_size), int(cfg["MAX_SEQ_LEN"])), int(msk_idx), dtype=torch.long)

    written = 0
    shard_idx = 0
    preservation_max_abs = 0.0

    for start in tqdm.tqdm(range(0, total_rows, int(args.shard_size)), desc=name, leave=False):
        rows = min(int(args.shard_size), total_rows - start)
        batch = table.slice(start, rows).select(["spectra"]).to_pylist()
        raw = torch.tensor([item["spectra"] for item in batch], dtype=torch.float32)
        prism_rat = prism_spectra_to_rat(raw, args.spectra_order)
        spectra = interpolate_spectra(prism_rat, prism_wavelengths, target_wavelengths)

        exact_target_idx = torch.nonzero(
            (target_wavelengths[:, None] == prism_wavelengths[None, :]).any(dim=1),
            as_tuple=False,
        ).squeeze(1)
        if exact_target_idx.numel():
            recovered = spectra.index_select(-1, exact_target_idx)
            source_idx = torch.searchsorted(prism_wavelengths, target_wavelengths.index_select(0, exact_target_idx))
            original = prism_rat.index_select(-1, source_idx)
            preservation_max_abs = max(preservation_max_abs, float((recovered - original).abs().max().item()))

        out_file = dataset_dir / f"{name}-{shard_idx:05d}.safetensors"
        write_shard(out_file, spectra, stack[:rows], args.save_dtype)
        written += rows
        shard_idx += 1

    return {
        "name": name,
        "source": str(arrow_file),
        "out_dir": str(dataset_dir),
        "rows": int(written),
        "shards": int(shard_idx),
        "spectra_order": args.spectra_order,
        "original_wavelength_min": 400,
        "original_wavelength_max": 1100,
        "original_wavelength_step": 10,
        "target_wavelength_min": int(target_wavelengths[0].item()),
        "target_wavelength_max": int(target_wavelengths[-1].item()),
        "target_wavelength_step": int((target_wavelengths[1] - target_wavelengths[0]).item()) if target_wavelengths.numel() > 1 else None,
        "exact_original_grid_points_preserved": bool(preservation_max_abs == 0.0),
        "max_abs_error_at_original_grid_points": preservation_max_abs,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    _, _, _, _, _, _, _, _, msk_idx = optollama.data.init_tokens(cfg["TOKENS_PATH"])

    source = Path(args.source)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    files = validation_arrow_files(source)
    if args.max_files is not None:
        files = files[: int(args.max_files)]

    manifest = {
        "config": args.config,
        "source": str(source),
        "out_dir": str(out_root),
        "channel_order": args.spectra_order,
        "format": "spectra=[R,A,T], thin_films=<MSK> placeholders",
        "comparison_roi_nm": [400, 1100],
        "datasets": [],
    }

    for arrow_file in tqdm.tqdm(files, desc="PRISM validation files"):
        manifest["datasets"].append(convert_file(arrow_file, source, out_root, cfg, msk_idx, args))

    optollama.utils.save_as_json(str(out_root / "manifest.json"), manifest)
    print(f"Converted {len(manifest['datasets'])} PRISM validation datasets -> {out_root}")
    print(f"Saved manifest -> {out_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
