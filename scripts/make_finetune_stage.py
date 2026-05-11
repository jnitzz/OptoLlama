#!/usr/bin/env python

import argparse
from pathlib import Path

import safetensors.torch
import torch
import tqdm

import optollama.data
import optollama.utils


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a fixed-width fine-tuning stage dataset from short and long shard pools."
    )
    p.add_argument("--short-input", nargs="+", required=True, help="Converted short shards or directories.")
    p.add_argument("--long-input", nargs="+", required=True, help="Generated long shards or directories.")
    p.add_argument("--out-dir", type=str, required=True, help="Output stage directory.")
    p.add_argument("--num-samples", type=int, required=True, help="Total samples to write.")
    p.add_argument("--short-fraction", type=float, required=True, help="Fraction of samples drawn from short-input.")
    p.add_argument("--shard-size", type=int, default=10000, help="Samples per output shard.")
    p.add_argument("--seed", type=int, default=3, help="Shuffle seed inside each output shard.")
    p.add_argument("--prefix", type=str, default="stage", help="Output shard filename prefix.")
    return p.parse_args()


def resolve_shards(items: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in items:
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


class ShardCursor:
    """
    Sequential chunk reader over safetensor shards.
    """

    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths
        self.path_idx = 0
        self.row_idx = 0
        self.spectra: torch.Tensor | None = None
        self.stacks: torch.Tensor | None = None
        self._load_next()

    def _load_next(self) -> None:
        if self.path_idx >= len(self.paths):
            self.spectra = None
            self.stacks = None
            return

        data = safetensors.torch.load_file(str(self.paths[self.path_idx]), device="cpu")
        self.spectra = data["spectra"].to(torch.float32)
        self.stacks = data["thin_films"].long()
        self.row_idx = 0
        self.path_idx += 1

    def take(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        spectra_parts: list[torch.Tensor] = []
        stack_parts: list[torch.Tensor] = []
        remaining = int(count)

        while remaining > 0:
            if self.spectra is None or self.stacks is None:
                raise RuntimeError("Input shards exhausted before requested sample count was reached.")

            available = self.spectra.size(0) - self.row_idx
            take = min(remaining, available)
            if take > 0:
                spectra_parts.append(self.spectra[self.row_idx : self.row_idx + take])
                stack_parts.append(self.stacks[self.row_idx : self.row_idx + take])
                self.row_idx += take
                remaining -= take

            if self.spectra is not None and self.row_idx >= self.spectra.size(0):
                self._load_next()

        return torch.cat(spectra_parts, dim=0), torch.cat(stack_parts, dim=0)


def validate_args(args: argparse.Namespace) -> None:
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be positive.")
    if not 0.0 <= args.short_fraction <= 1.0:
        raise ValueError("--short-fraction must be in [0, 1].")


def main() -> None:
    args = parse_args()
    validate_args(args)

    short_paths = resolve_shards(args.short_input)
    long_paths = resolve_shards(args.long_input)
    short_cursor = ShardCursor(short_paths)
    long_cursor = ShardCursor(long_paths)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed))

    remaining = int(args.num_samples)
    shard_idx = 0
    total_short = 0
    total_long = 0
    summary: list[dict[str, object]] = []

    with tqdm.tqdm(total=args.num_samples, desc="make-finetune-stage") as pbar:
        while remaining > 0:
            shard_count = min(int(args.shard_size), remaining)
            short_count = int(round(shard_count * float(args.short_fraction)))
            long_count = shard_count - short_count

            spectra_parts: list[torch.Tensor] = []
            stack_parts: list[torch.Tensor] = []
            source_parts: list[torch.Tensor] = []

            if short_count:
                spectra_short, stacks_short = short_cursor.take(short_count)
                spectra_parts.append(spectra_short)
                stack_parts.append(stacks_short)
                source_parts.append(torch.zeros((short_count,), dtype=torch.long))
                total_short += short_count
            if long_count:
                spectra_long, stacks_long = long_cursor.take(long_count)
                spectra_parts.append(spectra_long)
                stack_parts.append(stacks_long)
                source_parts.append(torch.ones((long_count,), dtype=torch.long))
                total_long += long_count

            seq_widths = {int(part.size(1)) for part in stack_parts}
            if len(seq_widths) != 1:
                raise RuntimeError(f"All input shards must have the same thin_films width, got: {sorted(seq_widths)}")

            spectra = torch.cat(spectra_parts, dim=0)
            stacks = torch.cat(stack_parts, dim=0)
            sources = torch.cat(source_parts, dim=0)

            if spectra.size(0) != stacks.size(0):
                raise RuntimeError("Mismatched spectra/stacks while building stage shard.")

            perm = torch.randperm(spectra.size(0), generator=generator)
            spectra = spectra[perm]
            stacks = stacks[perm]
            sources = sources[perm]

            out_path = out_dir / f"{args.prefix}-{shard_idx}.safetensors"
            safetensors.torch.save_file({"spectra": spectra, "thin_films": stacks}, str(out_path))
            summary.append(
                {
                    "output_shard": str(out_path),
                    "samples": int(spectra.size(0)),
                    "short_samples": int((sources == 0).sum().item()),
                    "long_samples": int((sources == 1).sum().item()),
                    "seq_len": int(stacks.size(1)),
                }
            )

            shard_idx += 1
            remaining -= shard_count
            pbar.update(shard_count)

    summary_obj = {
        "num_samples": int(args.num_samples),
        "short_fraction": float(args.short_fraction),
        "short_samples": int(total_short),
        "long_samples": int(total_long),
        "shards": summary,
    }
    optollama.utils.save_as_json(str(out_dir / "finetune_stage_summary.json"), summary_obj)
    print(f"Saved stage dataset to {out_dir}")


if __name__ == "__main__":
    main()
