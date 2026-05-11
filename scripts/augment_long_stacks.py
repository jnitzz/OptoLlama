#!/usr/bin/env python

import argparse
import collections
import os
import shutil
from pathlib import Path
from typing import Optional

import safetensors.torch
import torch
import tqdm

import optollama.data
import optollama.evaluation
import optollama.utils


def format_length_histogram(lengths: list[int], width: int = 48) -> str:
    """
    Build a compact terminal histogram of final layer lengths.

    Args
    ----
    lengths : list[int]
        Final layer lengths.
    width : int
        Maximum bar width.

    Returns
    -------
    str
        Multi-line ASCII histogram.
    """
    if not lengths:
        return "Final length histogram: no samples"

    counts = collections.Counter(int(v) for v in lengths)
    max_count = max(counts.values())
    min_len = min(counts)
    max_len = max(counts)
    length_values = list(range(min_len, max_len + 1))

    stride = max(1, (len(length_values) + width - 1) // width)
    bins: list[tuple[int, int, int]] = []
    for start in range(min_len, max_len + 1, stride):
        end = min(start + stride - 1, max_len)
        count = sum(counts.get(length, 0) for length in range(start, end + 1))
        bins.append((start, end, count))

    lines = ["Final length histogram (frequency x sequence length):"]
    for level in range(max_count, 0, -1):
        row = "".join("#" if count >= level else " " for _, _, count in bins)
        lines.append(f"{level:4d} | {row}")

    tick_row = [" "] * len(bins)
    tick_positions = {0: str(min_len), len(bins) - 1: str(max_len)}
    for i, (start, end, _) in enumerate(bins):
        if start <= min_len or end >= max_len:
            continue
        if start % 10 == 0:
            tick_positions[i] = str(start)

    rendered_positions = set()
    for pos, label in sorted(tick_positions.items(), key=lambda item: (item[0] not in (0, len(bins) - 1), -len(item[1]))):
        start_pos = min(pos, max(0, len(bins) - len(label)))
        left_guard = max(0, start_pos - 1)
        right_guard = min(len(bins), start_pos + len(label) + 1)
        if any(tick_row[j] != " " for j in range(left_guard, right_guard)):
            continue
        for j, char in enumerate(label):
            tick_row[start_pos + j] = char
        rendered_positions.add(pos)

    axis = "".join("|" if i in rendered_positions else "." for i in range(len(bins)))
    lines.append("     +" + "-" * len(bins))
    lines.append("      " + axis)
    lines.append("      " + "".join(tick_row).rstrip())
    if stride > 1:
        lines.append(f"      bin width: {stride} layers")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for long-stack augmentation.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    p = argparse.ArgumentParser(
        description=(
            "Extend existing thin-film datasets with low-loss suffix layers and "
            "write augmented safetensor shards to a separate folder."
        )
    )
    p.add_argument("--config", type=str, default="configs/optollama.yaml", help="Path to the project config YAML.")
    p.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Which config-defined split to augment when --input is not provided.",
    )
    p.add_argument(
        "--input",
        nargs="*",
        default=None,
        help="Optional explicit .safetensors files or directories. Overrides --split.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory where augmented .safetensors shards will be written.",
    )
    p.add_argument("--seed", type=int, default=7, help="Random seed for suffix generation.")
    p.add_argument("--batch-size", type=int, default=32, help="Number of source samples processed per augmentation batch.")
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Optional global cap on the number of source stacks to augment across the selected shards. Useful for manual dry runs."
        ),
    )
    p.add_argument(
        "--samples-per-input",
        type=int,
        default=1,
        help="How many augmented variants to generate per input sample.",
    )
    p.add_argument(
        "--max-layers",
        type=int,
        default=100,
        help="Maximum number of material layers in the augmented stack.",
    )
    p.add_argument(
        "--min-layers",
        type=int,
        default=30,
        help="Minimum target number of material layers in the augmented stack.",
    )
    p.add_argument(
        "--output-seq-len",
        type=int,
        default=None,
        help="Length of the saved token sequence. Defaults to max_layers + 1 to store EOS explicitly.",
    )
    p.add_argument(
        "--num-candidates",
        type=int,
        default=8,
        help="Number of candidate suffix tokens simulated per sample and round.",
    )
    p.add_argument(
        "--proposal-rounds",
        type=int,
        default=4,
        help="How many candidate rounds to try before stopping a sample early.",
    )
    p.add_argument(
        "--min-delta-mae",
        type=float,
        default=0.002,
        help="Minimum MAE change required for an appended layer to be accepted.",
    )
    p.add_argument(
        "--lookahead-candidates",
        type=int,
        default=0,
        help="Number of one-step lookahead proposals per first-step candidate.",
    )
    p.add_argument(
        "--lookahead-topk",
        type=int,
        default=0,
        help=(
            "Only evaluate lookahead for the top-k first-step candidates per "
            "sample, ranked by immediate spectral change. Use 0 to evaluate all."
        ),
    )
    p.add_argument(
        "--lookahead-weight",
        type=float,
        default=0.35,
        help="Weight of the lookahead delta in the combined candidate score.",
    )
    p.add_argument(
        "--min-lookahead-delta",
        type=float,
        default=0.0005,
        help="Minimum best lookahead delta required when another suffix step is still needed.",
    )
    p.add_argument(
        "--metal-penalty",
        type=float,
        default=0.01,
        help="Score penalty for proposals in the predefined metals group.",
    )
    p.add_argument(
        "--absorber-penalty",
        type=float,
        default=0.005,
        help="Score penalty for proposals in the predefined metals or semiconductors groups.",
    )
    p.add_argument(
        "--absorber-load-penalty",
        type=float,
        default=0.01,
        help="Additional penalty proportional to the current absorber fraction in the stack.",
    )
    p.add_argument(
        "--allowed-groups",
        nargs="*",
        default=None,
        help=("Optional material groups allowed for suffix generation. If omitted, all non-special tokens are eligible."),
    )
    p.add_argument(
        "--exclude-materials",
        nargs="*",
        default=None,
        help="Optional base material names to exclude from suffix generation.",
    )
    p.add_argument(
        "--exclude-tokens",
        nargs="*",
        default=None,
        help="Optional exact tokens to exclude from suffix generation.",
    )
    p.add_argument(
        "--delta-roi-min",
        type=float,
        default=None,
        help="Optional lower wavelength bound for the acceptance-test MAE.",
    )
    p.add_argument(
        "--delta-roi-max",
        type=float,
        default=None,
        help="Optional upper wavelength bound for the acceptance-test MAE.",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help='Execution device, e.g. "cuda", "cuda:0", or "cpu". Defaults to CUDA when available.',
    )

    return p.parse_args()


def load_config(path: str) -> dict:
    """
    Load project config and construct the wavelength grid.

    Args
    ----
    path : str
        Config path.

    Returns
    -------
    dict
        Configuration dictionary with ``WAVELENGTHS`` populated.
    """
    cfg = optollama.utils.load_config_file(path)
    wl_min = int(cfg["WAVELENGTH_MIN"])
    wl_max = int(cfg["WAVELENGTH_MAX"])
    wl_step = int(cfg["WAVELENGTH_STEPS"])
    cfg["WAVELENGTHS"] = torch.arange(wl_min, wl_max + 1, wl_step, dtype=torch.int)

    return cfg


def resolve_input_shards(cfg: dict, split: str, explicit_paths: Optional[list[str]]) -> list[Path]:
    """
    Resolve the set of input shards to augment.

    Args
    ----
    cfg : dict
        Project config.
    split : str
        Config split key, ``"train"`` or ``"test"``.
    explicit_paths : list[str] or None
        Explicit user-provided paths overriding the config split.

    Returns
    -------
    list[pathlib.Path]
        Sorted list of input shard files.
    """
    raw_paths: list[str]
    if explicit_paths:
        raw_paths = explicit_paths
    else:
        prefix = "DATA_PATH_TRAIN" if split == "train" else "DATA_PATH_TEST"
        raw_paths = sorted([cfg[key] for key in cfg if key.startswith(prefix)])

    files: list[Path] = []
    for item in raw_paths:
        p = Path(item)
        if p.is_dir():
            files.extend(sorted(p.glob("*.safetensors")))
        elif p.suffix == ".safetensors":
            files.append(p)
        else:
            raise ValueError(f"Unsupported input path: {p}")

    if not files:
        raise FileNotFoundError("No .safetensors input shards found.")

    return sorted(files, key=lambda p: optollama.data.SpectraDataset.shard_sort_key(str(p.with_suffix(""))))


def build_roi_mask(args: argparse.Namespace, cfg: dict, device: torch.device) -> Optional[torch.Tensor]:
    """
    Build the optional ROI mask used during suffix acceptance.

    Args
    ----
    args : argparse.Namespace
        Parsed CLI arguments.
    cfg : dict
        Project config.
    device : torch.device
        Execution device.

    Returns
    -------
    torch.Tensor or None
        Boolean wavelength mask or ``None``.
    """
    if args.delta_roi_min is None or args.delta_roi_max is None:
        return None

    return optollama.data.wavelength_mask(
        cfg["WAVELENGTHS"],
        args.delta_roi_min,
        args.delta_roi_max,
        device,
    )


def copy_tokens_file(cfg: dict, out_dir: Path) -> None:
    """
    Copy the shared token vocabulary to the output folder.

    Args
    ----
    cfg : dict
        Project config.
    out_dir : pathlib.Path
        Output directory.
    """
    src = Path(cfg["TOKENS_PATH"])
    if src.exists():
        shutil.copy2(src, out_dir / src.name)


def main() -> None:
    """
    Run the long-stack augmentation pipeline.
    """
    args = parse_args()
    cfg = load_config(args.config)

    output_seq_len = int(args.output_seq_len or (args.max_layers + 1))
    if output_seq_len < args.max_layers + 1:
        raise ValueError(f"--output-seq-len must be at least max_layers + 1 (got {output_seq_len} vs {args.max_layers + 1}).")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    copy_tokens_file(cfg, out_dir)

    tokens, token_to_idx, idx_to_token, _, _, _, eos_idx, pad_idx, msk_idx = optollama.data.init_tokens(cfg["TOKENS_PATH"])
    pool = optollama.data.build_extension_pool(
        tokens,
        token_to_idx,
        allowed_groups=args.allowed_groups,
        exclude_materials=args.exclude_materials,
        exclude_tokens=args.exclude_tokens,
    )
    roi_mask = build_roi_mask(args, cfg, device)
    tmm_ctx = optollama.evaluation.simulation.TMMContext.make(cfg, idx_to_token, device=device)

    input_shards = resolve_input_shards(cfg, args.split, args.input)
    generator = torch.Generator(device="cuda" if device.type == "cuda" else "cpu")
    generator.manual_seed(int(args.seed))

    summary: list[dict[str, object]] = []
    remaining_samples = args.max_samples
    all_final_lengths: list[int] = []

    for shard_path in input_shards:
        if remaining_samples is not None and remaining_samples <= 0:
            break

        print(f"Augmenting shard: {shard_path}")
        data = safetensors.torch.load_file(str(shard_path), device="cpu")
        spectra_cpu = data["spectra"].to(torch.float32)
        stacks_cpu = data["thin_films"].long()
        n_samples = spectra_cpu.size(0)

        if spectra_cpu.size(-1) != cfg["WAVELENGTHS"].numel():
            raise ValueError(
                f"Wavelength grid mismatch for {shard_path}: spectra have W={spectra_cpu.size(-1)} "
                f"but config defines {cfg['WAVELENGTHS'].numel()} wavelengths. "
                "Check WAVELENGTH_MIN/MAX/STEPS in the config."
            )

        if remaining_samples is not None:
            take = min(int(remaining_samples), n_samples)
            spectra_cpu = spectra_cpu[:take]
            stacks_cpu = stacks_cpu[:take]
            n_samples = take
            remaining_samples -= take

        if n_samples == 0:
            continue

        aug_spectra_parts: list[torch.Tensor] = []
        aug_stack_parts: list[torch.Tensor] = []
        source_len_parts: list[torch.Tensor] = []
        target_len_parts: list[torch.Tensor] = []
        final_len_parts: list[torch.Tensor] = []
        stopped_parts: list[torch.Tensor] = []

        pbar = tqdm.tqdm(range(0, n_samples, args.batch_size), desc=shard_path.name, leave=False)
        for start in pbar:
            end = min(start + args.batch_size, n_samples)
            batch_spectra = spectra_cpu[start:end].to(device, non_blocking=True)
            batch_stacks = stacks_cpu[start:end].to(device, non_blocking=True)

            aug_spectra, aug_stacks, stats = optollama.data.augment_stack_batch(
                batch_spectra,
                batch_stacks,
                tmm_ctx=tmm_ctx,
                pool=pool,
                eos=eos_idx,
                pad=pad_idx,
                msk=msk_idx,
                max_layers=args.max_layers,
                output_seq_len=output_seq_len,
                min_layers=args.min_layers,
                samples_per_input=args.samples_per_input,
                num_candidates=args.num_candidates,
                proposal_rounds=args.proposal_rounds,
                min_delta_mae=args.min_delta_mae,
                lookahead_candidates=args.lookahead_candidates,
                lookahead_topk=args.lookahead_topk,
                lookahead_weight=args.lookahead_weight,
                min_lookahead_delta=args.min_lookahead_delta,
                metal_penalty=args.metal_penalty,
                absorber_penalty=args.absorber_penalty,
                absorber_load_penalty=args.absorber_load_penalty,
                roi_mask=roi_mask,
                generator=generator,
            )

            aug_spectra_parts.append(aug_spectra.detach().cpu())
            aug_stack_parts.append(aug_stacks.detach().cpu())
            source_len_parts.append(stats["source_lengths"])
            target_len_parts.append(stats["target_lengths"])
            final_len_parts.append(stats["final_lengths"])
            stopped_parts.append(stats["stopped_early"])

            mean_final = float(stats["final_lengths"].float().mean().item())
            pbar.set_postfix(final_layers=f"{mean_final:.1f}")

        aug_spectra_all = torch.cat(aug_spectra_parts, dim=0)
        aug_stacks_all = torch.cat(aug_stack_parts, dim=0)
        source_lengths_all = torch.cat(source_len_parts, dim=0)
        target_lengths_all = torch.cat(target_len_parts, dim=0)
        final_lengths_all = torch.cat(final_len_parts, dim=0)
        stopped_all = torch.cat(stopped_parts, dim=0)
        all_final_lengths.extend(int(v) for v in final_lengths_all.tolist())

        out_path = out_dir / shard_path.name
        safetensors.torch.save_file(
            {
                "spectra": aug_spectra_all.to(torch.float32),
                "thin_films": aug_stacks_all.long(),
            },
            str(out_path),
        )

        summary.append(
            {
                "input_shard": str(shard_path),
                "output_shard": str(out_path),
                "samples_in": int(n_samples),
                "samples_out": int(aug_stacks_all.size(0)),
                "mean_source_layers": float(source_lengths_all.float().mean().item()),
                "mean_target_layers": float(target_lengths_all.float().mean().item()),
                "mean_final_layers": float(final_lengths_all.float().mean().item()),
                "mean_layer_increment": float((final_lengths_all - source_lengths_all).float().mean().item()),
                "stopped_early_fraction": float(stopped_all.float().mean().item()),
            }
        )
        print(
            f"Saved {aug_stacks_all.size(0)} augmented samples to {out_path} "
            f"(mean final layers: {summary[-1]['mean_final_layers']:.1f}, "
            f"mean increment: {summary[-1]['mean_layer_increment']:.1f})"
        )

    optollama.utils.save_as_json(os.path.join(out_dir, "augmentation_summary.json"), summary)
    if summary:
        mean_increment = sum(float(item["mean_layer_increment"]) for item in summary) / len(summary)
        print(f"Overall mean layer increment: {mean_increment:.1f}")
        print(format_length_histogram(all_final_lengths))
    print(f"Saved augmentation summary to {out_dir / 'augmentation_summary.json'}")


if __name__ == "__main__":
    main()
