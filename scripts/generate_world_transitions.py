#!/usr/bin/env python

import argparse
import os
from pathlib import Path
from typing import Optional

import safetensors.torch
import torch
import tqdm

import optollama.data
import optollama.evaluation
import optollama.utils


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    p = argparse.ArgumentParser(
        description=(
            "Generate local stack-edit transition data for a learned OptoLlama "
            "world-model scorer."
        )
    )
    p.add_argument("--config", type=str, default="configs/world_model.yaml")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--source-split", type=str, default=None, choices=["train", "test"])
    p.add_argument("--num-base-samples", type=int, default=None)
    p.add_argument("--num-transitions", type=int, default=None)
    p.add_argument("--shard-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--eval-batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--current-edits", type=int, default=None)
    p.add_argument("--next-edits", type=int, default=None)
    p.add_argument("--num-next-perturbations", type=int, default=None)
    p.add_argument("--include-anchor-repair", action="store_true")
    p.add_argument("--allowed-groups", nargs="*", default=None)
    p.add_argument("--exclude-materials", nargs="*", default=None)
    p.add_argument("--exclude-tokens", nargs="*", default=None)
    return p.parse_args()


def load_config(path: str) -> dict:
    """
    Load config and enrich it with the wavelength tensor.
    """
    cfg = optollama.utils.load_config_file(path)
    wl_min = int(cfg["WAVELENGTH_MIN"])
    wl_max = int(cfg["WAVELENGTH_MAX"])
    wl_step = int(cfg["WAVELENGTH_STEPS"])
    cfg["WAVELENGTHS"] = torch.arange(wl_min, wl_max + 1, wl_step, dtype=torch.int)
    return cfg


def resolve_arg(args: argparse.Namespace, cfg: dict, name: str, default: Optional[object] = None) -> object:
    """
    Use a CLI value when provided, otherwise a config value.
    """
    cli_value = getattr(args, name)
    if cli_value is not None:
        return cli_value
    cfg_key = name.upper().replace("-", "_")
    return cfg.get(cfg_key, default)


def simulate_in_chunks(
    stacks: torch.Tensor,
    tmm_ctx: optollama.evaluation.simulation.TMMContext,
    eos: int,
    pad: int,
    msk: int,
    eval_batch_size: int,
) -> torch.Tensor:
    """
    Simulate stack spectra in bounded TMM batches.
    """
    out: list[torch.Tensor] = []
    for start in range(0, stacks.size(0), eval_batch_size):
        end = min(start + eval_batch_size, stacks.size(0))
        out.append(
            optollama.evaluation.simulation.simulate_token_sequence(
                stacks[start:end],
                tmm_ctx,
                eos=eos,
                pad=pad,
                msk=msk,
            )
        )
    return torch.cat(out, dim=0)


def mae_in_chunks(
    targets: torch.Tensor,
    spectra: torch.Tensor,
    roi_mask: Optional[torch.Tensor],
    eval_batch_size: int,
) -> torch.Tensor:
    """
    Compute ROI MAE in chunks.
    """
    out: list[torch.Tensor] = []
    for start in range(0, targets.size(0), eval_batch_size):
        end = min(start + eval_batch_size, targets.size(0))
        out.append(optollama.evaluation.masked_mae_roi(targets[start:end], spectra[start:end], wl_mask=roi_mask))
    return torch.cat(out, dim=0)


def flush_shard(buffers: dict[str, list[torch.Tensor]], out_dir: Path, shard_idx: int) -> int:
    """
    Write one transition shard and clear the in-memory buffers.
    """
    if not buffers["target_spectra"]:
        return shard_idx

    payload = {key: torch.cat(values, dim=0).detach().cpu() for key, values in buffers.items()}
    path = out_dir / f"world-transitions-{shard_idx:04d}.safetensors"
    safetensors.torch.save_file(payload, str(path))
    for values in buffers.values():
        values.clear()
    print(f"Saved {payload['target_spectra'].size(0)} transitions -> {path}")
    return shard_idx + 1


def append_rows(
    buffers: dict[str, list[torch.Tensor]],
    target_spectra: torch.Tensor,
    current_stacks: torch.Tensor,
    current_spectra: torch.Tensor,
    next_stacks: torch.Tensor,
    next_spectra: torch.Tensor,
    cost_before: torch.Tensor,
    cost_after: torch.Tensor,
) -> None:
    """
    Append transition rows to shard buffers.
    """
    buffers["target_spectra"].append(target_spectra.detach().cpu().to(torch.float32))
    buffers["current_stacks"].append(current_stacks.detach().cpu().long())
    buffers["current_spectra"].append(current_spectra.detach().cpu().to(torch.float32))
    buffers["next_stacks"].append(next_stacks.detach().cpu().long())
    buffers["next_spectra"].append(next_spectra.detach().cpu().to(torch.float32))
    buffers["cost_before"].append(cost_before.detach().cpu().to(torch.float32))
    buffers["cost_after"].append(cost_after.detach().cpu().to(torch.float32))


def main() -> None:
    """
    Generate world-transition safetensor shards.
    """
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("SEED", 3))
    optollama.utils.set_all_seeds(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed + 917)

    device = torch.device(args.device or cfg.get("WORLD_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out_dir or cfg["WORLD_TRANSITION_OUTPUT_PATH"])
    out_dir.mkdir(parents=True, exist_ok=True)

    source_split = str(args.source_split or cfg.get("WORLD_SOURCE_SPLIT", "train"))
    num_base_samples = int(args.num_base_samples or cfg.get("WORLD_NUM_BASE_SAMPLES", 1024))
    num_transitions = int(args.num_transitions or cfg.get("WORLD_NUM_TRANSITIONS", 8192))
    shard_size = int(args.shard_size or cfg.get("WORLD_SHARD_SIZE", 4096))
    batch_size = int(args.batch_size or cfg.get("WORLD_GENERATE_BATCH_SIZE", 64))
    eval_batch_size = int(args.eval_batch_size or cfg.get("WORLD_EVAL_BATCH_SIZE", 512))
    current_edits = int(args.current_edits if args.current_edits is not None else cfg.get("WORLD_CURRENT_EDITS", 2))
    next_edits = int(args.next_edits if args.next_edits is not None else cfg.get("WORLD_NEXT_EDITS", 1))
    num_next = int(
        args.num_next_perturbations
        if args.num_next_perturbations is not None
        else cfg.get("WORLD_NUM_NEXT_PERTURBATIONS", 4)
    )
    include_anchor = bool(args.include_anchor_repair or cfg.get("WORLD_INCLUDE_ANCHOR_REPAIR", False))

    tokens, token_to_idx, idx_to_token, _, _, _, eos_idx, pad_idx, msk_idx = optollama.data.init_tokens(cfg["TOKENS_PATH"])
    pool = optollama.data.build_extension_pool(
        tokens,
        token_to_idx,
        allowed_groups=args.allowed_groups or cfg.get("WORLD_ALLOWED_GROUPS"),
        exclude_materials=args.exclude_materials or cfg.get("WORLD_EXCLUDE_MATERIALS"),
        exclude_tokens=args.exclude_tokens or cfg.get("WORLD_EXCLUDE_TOKENS"),
    )
    mutation_index = optollama.data.build_token_mutation_index(tokens, pool)
    tmm_ctx = optollama.evaluation.simulation.TMMContext.make(cfg, idx_to_token, device=device)
    roi_mask = optollama.data.wavelength_mask(cfg["WAVELENGTHS"], cfg["ROI_MIN"], cfg["ROI_MAX"], device)

    original_train_batch = cfg.get("TRAIN_BATCH_SIZE")
    original_test_batch = cfg.get("TEST_BATCH_SIZE")
    cfg["TRAIN_BATCH_SIZE"] = batch_size
    cfg["TEST_BATCH_SIZE"] = batch_size
    try:
        _, loader, _ = optollama.data.SpectraDataset.make_loader(
            cfg,
            split=source_split,
            subset_n=num_base_samples,
            ddp=False,
        )
    finally:
        if original_train_batch is not None:
            cfg["TRAIN_BATCH_SIZE"] = original_train_batch
        if original_test_batch is not None:
            cfg["TEST_BATCH_SIZE"] = original_test_batch

    output_seq_len = int(cfg.get("WORLD_OUTPUT_SEQ_LEN", cfg["MAX_SEQ_LEN"]))
    max_layers = int(cfg.get("WORLD_MAX_LAYERS", output_seq_len - 1))

    buffers: dict[str, list[torch.Tensor]] = {key: [] for key in optollama.data.WORLD_TRANSITION_KEYS}
    rows_in_buffer = 0
    total_rows = 0
    shard_idx = 0

    pbar = tqdm.tqdm(loader, desc="world transitions")
    for batch in pbar:
        if total_rows >= num_transitions:
            break

        targets = batch[0].to(device, non_blocking=True).to(torch.float32)
        base_stacks = batch[1].to(device, non_blocking=True).long()
        base_stacks, _ = optollama.data.reencode_stacks_for_output(
            base_stacks,
            output_seq_len=output_seq_len,
            eos=eos_idx,
            pad=pad_idx,
            msk=msk_idx,
        )

        if current_edits > 0:
            current_stacks, current_source = optollama.data.perturb_stack_candidates(
                base_stacks,
                index=mutation_index,
                eos=eos_idx,
                pad=pad_idx,
                msk=msk_idx,
                max_layers=max_layers,
                output_seq_len=output_seq_len,
                num_perturbations=1,
                generator=generator,
                edits_per_perturbation=current_edits,
            )
            current_targets = targets[current_source]
            current_base_stacks = base_stacks[current_source]
        else:
            current_stacks = base_stacks
            current_targets = targets
            current_base_stacks = base_stacks

        current_spectra = simulate_in_chunks(current_stacks, tmm_ctx, eos_idx, pad_idx, msk_idx, eval_batch_size)
        cost_before = mae_in_chunks(current_targets, current_spectra, roi_mask, eval_batch_size)

        next_stacks, source_idx = optollama.data.perturb_stack_candidates(
            current_stacks,
            index=mutation_index,
            eos=eos_idx,
            pad=pad_idx,
            msk=msk_idx,
            max_layers=max_layers,
            output_seq_len=output_seq_len,
            num_perturbations=num_next,
            generator=generator,
            edits_per_perturbation=next_edits,
        )
        next_targets = current_targets[source_idx]
        next_current_stacks = current_stacks[source_idx]
        next_current_spectra = current_spectra[source_idx]
        next_cost_before = cost_before[source_idx]
        next_spectra = simulate_in_chunks(next_stacks, tmm_ctx, eos_idx, pad_idx, msk_idx, eval_batch_size)
        cost_after = mae_in_chunks(next_targets, next_spectra, roi_mask, eval_batch_size)

        append_rows(
            buffers,
            next_targets,
            next_current_stacks,
            next_current_spectra,
            next_stacks,
            next_spectra,
            next_cost_before,
            cost_after,
        )
        rows_in_buffer += int(next_targets.size(0))
        total_rows += int(next_targets.size(0))

        if include_anchor:
            anchor_spectra = simulate_in_chunks(current_base_stacks, tmm_ctx, eos_idx, pad_idx, msk_idx, eval_batch_size)
            anchor_cost = mae_in_chunks(current_targets, anchor_spectra, roi_mask, eval_batch_size)
            append_rows(
                buffers,
                current_targets,
                current_stacks,
                current_spectra,
                current_base_stacks,
                anchor_spectra,
                cost_before,
                anchor_cost,
            )
            rows_in_buffer += int(current_targets.size(0))
            total_rows += int(current_targets.size(0))

        pbar.set_postfix(rows=total_rows, buffered=rows_in_buffer)
        if rows_in_buffer >= shard_size:
            shard_idx = flush_shard(buffers, out_dir, shard_idx)
            rows_in_buffer = 0

    if rows_in_buffer:
        shard_idx = flush_shard(buffers, out_dir, shard_idx)

    summary = {
        "source_split": source_split,
        "num_base_samples": num_base_samples,
        "num_transitions_requested": num_transitions,
        "num_transitions_written": total_rows,
        "shards": shard_idx,
        "output_seq_len": output_seq_len,
        "max_layers": max_layers,
        "current_edits": current_edits,
        "next_edits": next_edits,
        "num_next_perturbations": num_next,
        "include_anchor_repair": include_anchor,
    }
    optollama.utils.save_as_json(str(out_dir / "world_transition_summary.json"), summary)
    print(f"Saved summary to {out_dir / 'world_transition_summary.json'}")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
