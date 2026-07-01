#!/usr/bin/env python

import argparse
import collections
import math
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


FAMILIES = ("dbr", "chirped_dbr", "cavity", "random_dielectric", "random")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate large long-stack training datasets from physically "
            "structured priors and batched CUDA TMM simulation."
        )
    )
    p.add_argument("--config", type=str, default="configs/optollama.yaml", help="Path to the project config YAML.")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory for generated safetensor shards.")
    p.add_argument("--num-samples", type=int, required=True, help="Number of accepted samples to write.")
    p.add_argument("--shard-size", type=int, default=10000, help="Accepted samples per output shard.")
    p.add_argument("--candidate-batch-size", type=int, default=256, help="Number of candidate stacks generated per batch.")
    p.add_argument("--eval-batch-size", type=int, default=512, help="Number of stacks simulated per TMM batch.")
    p.add_argument("--min-layers", type=int, default=21, help="Minimum material layers in generated stacks.")
    p.add_argument("--max-layers", type=int, default=100, help="Maximum material layers in generated stacks.")
    p.add_argument(
        "--max-total-thickness-nm",
        type=float,
        default=None,
        help="Optional maximum summed layer thickness per generated stack in nm, e.g. 10000 for 10 um.",
    )
    p.add_argument("--output-seq-len", type=int, default=None, help="Saved thin_films width. Defaults to max_layers + 1.")
    p.add_argument(
        "--families",
        nargs="*",
        default=list(FAMILIES),
        choices=FAMILIES,
        help="Stack-generation families to sample uniformly.",
    )
    p.add_argument(
        "--allowed-groups",
        nargs="*",
        default=["dielectrics"],
        help="Material groups allowed for generation. Default: dielectrics.",
    )
    p.add_argument("--exclude-materials", nargs="*", default=None, help="Optional base material names to exclude.")
    p.add_argument("--exclude-tokens", nargs="*", default=None, help="Optional exact tokens to exclude.")
    p.add_argument("--center-min", type=float, default=None, help="Minimum design center wavelength for structured families.")
    p.add_argument("--center-max", type=float, default=None, help="Maximum design center wavelength for structured families.")
    p.add_argument("--jitter-fraction", type=float, default=0.08, help="Relative thickness jitter for structured stacks.")
    p.add_argument("--roi-min", type=float, default=None, help="MAE ROI lower wavelength bound. Defaults to config ROI_MIN.")
    p.add_argument("--roi-max", type=float, default=None, help="MAE ROI upper wavelength bound. Defaults to config ROI_MAX.")
    p.add_argument(
        "--min-prefix-mae",
        type=float,
        default=0.003,
        help="Keep only if short prefixes differ from full stack by at least this ROI MAE. Use 0 to disable.",
    )
    p.add_argument("--prefix-filter-max-layers", type=int, default=20, help="Maximum short-prefix length tested.")
    p.add_argument("--prefix-filter-stride", type=int, default=5, help="Stride for prefix lengths when --prefix-lengths is omitted.")
    p.add_argument("--prefix-lengths", nargs="*", type=int, default=None, help="Explicit short-prefix lengths to test.")
    p.add_argument(
        "--max-attempts-factor",
        type=float,
        default=20.0,
        help="Stop after num_samples * factor generated candidates if prefix filtering rejects too many.",
    )
    p.add_argument("--record-examples", type=int, default=200, help="Number of accepted examples to keep in summary JSON.")
    p.add_argument("--seed", type=int, default=None, help="Random seed. Defaults to config SEED.")
    p.add_argument("--device", type=str, default=None, help='Execution device, e.g. "cuda", "cuda:0", or "cpu".')
    return p.parse_args()


def load_config(path: str) -> dict:
    cfg = optollama.utils.load_config_file(path)
    wl_min = int(cfg["WAVELENGTH_MIN"])
    wl_max = int(cfg["WAVELENGTH_MAX"])
    wl_step = int(cfg["WAVELENGTH_STEPS"])
    cfg["WAVELENGTHS"] = torch.arange(wl_min, wl_max + 1, wl_step, dtype=torch.int)
    return cfg


def copy_tokens_file(cfg: dict, out_dir: Path) -> None:
    src = Path(cfg["TOKENS_PATH"])
    if src.exists():
        shutil.copy2(src, out_dir / src.name)


def build_roi_mask(args: argparse.Namespace, cfg: dict, device: torch.device) -> torch.Tensor:
    roi_min = float(args.roi_min if args.roi_min is not None else cfg["ROI_MIN"])
    roi_max = float(args.roi_max if args.roi_max is not None else cfg["ROI_MAX"])
    return optollama.data.wavelength_mask(cfg["WAVELENGTHS"], roi_min, roi_max, device)


def build_prefix_lengths(args: argparse.Namespace) -> list[int]:
    if args.min_prefix_mae <= 0.0:
        return []
    if args.prefix_lengths:
        return sorted({int(v) for v in args.prefix_lengths if int(v) > 0})

    stride = max(1, int(args.prefix_filter_stride))
    max_prefix = max(1, int(args.prefix_filter_max_layers))
    return list(range(stride, max_prefix + 1, stride))


@torch.no_grad()
def simulate_in_chunks(
    stacks: torch.Tensor,
    tmm_ctx: optollama.evaluation.simulation.TMMContext,
    eos: int,
    pad: int,
    msk: int,
    eval_batch_size: int,
) -> torch.Tensor:
    spectra: list[torch.Tensor] = []
    for start in range(0, stacks.size(0), int(eval_batch_size)):
        end = min(start + int(eval_batch_size), stacks.size(0))
        spectra.append(
            optollama.evaluation.simulation.simulate_token_sequence(
                stacks[start:end],
                tmm_ctx,
                eos=eos,
                pad=pad,
                msk=msk,
            )
        )
    return torch.cat(spectra, dim=0)


def save_shard(out_dir: Path, shard_idx: int, spectra: torch.Tensor, stacks: torch.Tensor) -> Path:
    out_path = out_dir / f"long-stacks-{shard_idx}.safetensors"
    safetensors.torch.save_file(
        {
            "spectra": spectra.to(torch.float32).cpu(),
            "thin_films": stacks.long().cpu(),
        },
        str(out_path),
    )
    return out_path


def flush_ready_shards(
    out_dir: Path,
    shard_idx: int,
    spectra_parts: list[torch.Tensor],
    stack_parts: list[torch.Tensor],
    shard_size: int,
    written_paths: list[str],
) -> tuple[int, list[torch.Tensor], list[torch.Tensor]]:
    if not spectra_parts:
        return shard_idx, spectra_parts, stack_parts

    spectra = torch.cat(spectra_parts, dim=0)
    stacks = torch.cat(stack_parts, dim=0)

    start = 0
    while spectra.size(0) - start >= shard_size:
        end = start + shard_size
        out_path = save_shard(out_dir, shard_idx, spectra[start:end], stacks[start:end])
        written_paths.append(str(out_path))
        print(f"Saved shard {out_path} ({shard_size} samples)")
        shard_idx += 1
        start = end

    if start >= spectra.size(0):
        return shard_idx, [], []
    return shard_idx, [spectra[start:].cpu()], [stacks[start:].cpu()]


def validate_args(args: argparse.Namespace) -> None:
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be positive.")
    if args.candidate_batch_size <= 0:
        raise ValueError("--candidate-batch-size must be positive.")
    if args.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive.")
    if args.min_layers <= 0:
        raise ValueError("--min-layers must be positive.")
    if args.max_layers < args.min_layers:
        raise ValueError("--max-layers must be >= --min-layers.")
    if args.max_total_thickness_nm is not None and args.max_total_thickness_nm <= 0:
        raise ValueError("--max-total-thickness-nm must be positive when set.")
    if args.jitter_fraction < 0.0:
        raise ValueError("--jitter-fraction must be non-negative.")
    if not args.families:
        raise ValueError("--families must contain at least one family.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg["SEED"])
    optollama.utils.set_all_seeds(seed)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    output_seq_len = int(args.output_seq_len or (args.max_layers + 1))
    if output_seq_len < args.max_layers + 1:
        raise ValueError("--output-seq-len must be at least --max-layers + 1.")

    center_min = float(args.center_min if args.center_min is not None else cfg["ROI_MIN"])
    center_max = float(args.center_max if args.center_max is not None else cfg["ROI_MAX"])
    if center_max <= center_min:
        raise ValueError("Center wavelength range must have center_max > center_min.")

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
    mutation_index = optollama.data.build_token_mutation_index(tokens, pool)
    library = optollama.data.build_long_stack_library(mutation_index)
    print(
        f"Generation materials: {len(library.materials)} total, "
        f"{len(library.low_index_materials)} low-index, {len(library.high_index_materials)} high-index"
    )

    tmm_ctx = optollama.evaluation.simulation.TMMContext.make(cfg, idx_to_token, device=device)
    roi_mask = build_roi_mask(args, cfg, device)
    prefix_lengths = build_prefix_lengths(args)
    if prefix_lengths:
        print(f"Prefix filter lengths: {prefix_lengths}, min_prefix_mae={args.min_prefix_mae}")
    else:
        print("Prefix filter disabled.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 17)

    shard_idx = 0
    accepted_total = 0
    generated_total = 0
    written_paths: list[str] = []
    spectra_buffer: list[torch.Tensor] = []
    stack_buffer: list[torch.Tensor] = []
    family_generated = collections.Counter()
    family_accepted = collections.Counter()
    length_hist = collections.Counter()
    best_prefix_mae_values: list[float] = []
    examples: list[dict[str, object]] = []
    max_attempts = int(math.ceil(args.num_samples * float(args.max_attempts_factor)))

    pbar = tqdm.tqdm(total=args.num_samples, desc="long-stack-generate")
    while accepted_total < args.num_samples and generated_total < max_attempts:
        candidate_count = min(int(args.candidate_batch_size), max_attempts - generated_total)
        stacks, lengths, families = optollama.data.generate_long_stack_batch(
            batch_size=candidate_count,
            families=args.families,
            library=library,
            min_layers=int(args.min_layers),
            max_layers=int(args.max_layers),
            output_seq_len=output_seq_len,
            eos=eos_idx,
            pad=pad_idx,
            center_min_nm=center_min,
            center_max_nm=center_max,
            jitter_fraction=float(args.jitter_fraction),
            generator=generator,
            device=device,
            max_total_thickness_nm=args.max_total_thickness_nm,
        )
        generated_total += candidate_count
        family_generated.update(families)

        spectra = simulate_in_chunks(
            stacks,
            tmm_ctx=tmm_ctx,
            eos=eos_idx,
            pad=pad_idx,
            msk=msk_idx,
            eval_batch_size=int(args.eval_batch_size),
        )

        keep_mask, best_prefix_mae = optollama.data.prefix_filter_mask(
            stacks,
            spectra,
            prefix_lengths=prefix_lengths,
            min_prefix_mae=float(args.min_prefix_mae),
            tmm_ctx=tmm_ctx,
            eos=eos_idx,
            pad=pad_idx,
            msk=msk_idx,
            roi_mask=roi_mask,
            eval_batch_size=int(args.eval_batch_size),
        )

        selected = keep_mask.nonzero(as_tuple=False).squeeze(1)
        remaining = args.num_samples - accepted_total
        if selected.numel() > remaining:
            selected = selected[:remaining]

        if selected.numel():
            kept_spectra = spectra[selected].detach().cpu()
            kept_stacks = stacks[selected].detach().cpu()
            spectra_buffer.append(kept_spectra)
            stack_buffer.append(kept_stacks)

            selected_list = selected.detach().cpu().tolist()
            for row in selected_list:
                family = families[int(row)]
                length = int(lengths[int(row)].item())
                family_accepted[family] += 1
                length_hist[length] += 1
                if len(examples) < int(args.record_examples):
                    examples.append(
                        {
                            "family": family,
                            "layers": length,
                            "best_prefix_mae": float(best_prefix_mae[int(row)].item()),
                        }
                    )
            if torch.isfinite(best_prefix_mae[selected]).any():
                best_prefix_mae_values.extend(
                    float(v) for v in best_prefix_mae[selected][torch.isfinite(best_prefix_mae[selected])].detach().cpu().tolist()
                )

            accepted_total += int(selected.numel())
            pbar.update(int(selected.numel()))

        shard_idx, spectra_buffer, stack_buffer = flush_ready_shards(
            out_dir,
            shard_idx,
            spectra_buffer,
            stack_buffer,
            int(args.shard_size),
            written_paths,
        )

        acceptance = accepted_total / max(generated_total, 1)
        pbar.set_postfix(generated=generated_total, acceptance=f"{acceptance:.2f}")

    pbar.close()

    if spectra_buffer:
        spectra = torch.cat(spectra_buffer, dim=0)
        stacks = torch.cat(stack_buffer, dim=0)
        out_path = save_shard(out_dir, shard_idx, spectra, stacks)
        written_paths.append(str(out_path))
        print(f"Saved final shard {out_path} ({stacks.size(0)} samples)")

    if accepted_total < args.num_samples:
        print(
            f"Stopped after {generated_total} candidates with {accepted_total}/{args.num_samples} accepted. "
            "Lower --min-prefix-mae or increase --max-attempts-factor if this is unintended."
        )

    summary = {
        "num_samples_requested": int(args.num_samples),
        "num_samples_written": int(accepted_total),
        "generated_candidates": int(generated_total),
        "acceptance_rate": float(accepted_total / max(generated_total, 1)),
        "families": list(args.families),
        "family_generated": dict(family_generated),
        "family_accepted": dict(family_accepted),
        "length_histogram": {str(key): int(value) for key, value in sorted(length_hist.items())},
        "mean_best_prefix_mae": (
            float(sum(best_prefix_mae_values) / len(best_prefix_mae_values)) if best_prefix_mae_values else None
        ),
        "min_layers": int(args.min_layers),
        "max_layers": int(args.max_layers),
        "max_total_thickness_nm": None if args.max_total_thickness_nm is None else float(args.max_total_thickness_nm),
        "output_seq_len": int(output_seq_len),
        "center_min": float(center_min),
        "center_max": float(center_max),
        "jitter_fraction": float(args.jitter_fraction),
        "min_prefix_mae": float(args.min_prefix_mae),
        "prefix_lengths": prefix_lengths,
        "allowed_groups": args.allowed_groups,
        "exclude_materials": args.exclude_materials,
        "exclude_tokens": args.exclude_tokens,
        "shards": written_paths,
        "examples": examples,
    }
    summary_path = out_dir / "long_stack_generation_summary.json"
    optollama.utils.save_as_json(str(summary_path), summary)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
