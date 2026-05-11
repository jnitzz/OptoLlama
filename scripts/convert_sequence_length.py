#!/usr/bin/env python

import argparse
import os
import shutil
from pathlib import Path
from typing import Optional

import safetensors.torch
import torch
import tqdm

import optollama.data
import optollama.utils


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Convert OptoLlama safetensor shards to a wider fixed token "
            "sequence length by preserving layer tokens, EOS, and PAD."
        )
    )
    p.add_argument("--config", type=str, default="configs/optollama.yaml", help="Path to the project config YAML.")
    p.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Config split to convert.")
    p.add_argument("--input", nargs="*", default=None, help="Optional explicit .safetensors files or directories.")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory for converted shards.")
    p.add_argument("--output-seq-len", type=int, required=True, help="Target thin_films sequence width.")
    p.add_argument("--max-samples", type=int, default=None, help="Optional global cap for quick checks.")
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


def copy_tokens_file(cfg: dict, out_dir: Path) -> None:
    src = Path(cfg["TOKENS_PATH"])
    if src.exists():
        shutil.copy2(src, out_dir / src.name)


def main() -> None:
    args = parse_args()
    if args.output_seq_len <= 0:
        raise ValueError("--output-seq-len must be positive.")

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    copy_tokens_file(cfg, out_dir)

    _, _, _, _, _, _, eos_idx, pad_idx, msk_idx = optollama.data.init_tokens(cfg["TOKENS_PATH"])
    input_shards = resolve_input_shards(cfg, args.split, args.input)
    remaining = args.max_samples
    summary: list[dict[str, object]] = []

    for shard_path in tqdm.tqdm(input_shards, desc="convert-seq-len"):
        if remaining is not None and remaining <= 0:
            break

        data = safetensors.torch.load_file(str(shard_path), device="cpu")
        spectra = data["spectra"].to(torch.float32)
        stacks = data["thin_films"].long()

        if remaining is not None:
            take = min(int(remaining), spectra.size(0))
            spectra = spectra[:take]
            stacks = stacks[:take]
            remaining -= take

        converted, lengths = optollama.data.reencode_stacks_for_output(
            stacks,
            output_seq_len=int(args.output_seq_len),
            eos=eos_idx,
            pad=pad_idx,
            msk=msk_idx,
        )

        out_path = out_dir / shard_path.name
        safetensors.torch.save_file(
            {
                "spectra": spectra,
                "thin_films": converted.long(),
            },
            str(out_path),
        )
        summary.append(
            {
                "input_shard": str(shard_path),
                "output_shard": str(out_path),
                "samples": int(converted.size(0)),
                "input_seq_len": int(stacks.size(1)),
                "output_seq_len": int(converted.size(1)),
                "mean_layers": float(lengths.float().mean().item()) if lengths.numel() else 0.0,
                "max_layers": int(lengths.max().item()) if lengths.numel() else 0,
            }
        )

    summary_path = out_dir / "sequence_length_conversion_summary.json"
    optollama.utils.save_as_json(str(summary_path), summary)
    print(f"Saved conversion summary to {summary_path}")


if __name__ == "__main__":
    main()
