#!/usr/bin/env python

import argparse
from pathlib import Path
from typing import Any

import safetensors.torch
import torch
import tqdm

import optollama.data
import optollama.evaluation
import optollama.evaluation.simulation
import optollama.utils

from scripts import self_improve_lite as lite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate CE pseudo-labels by sampling model candidates, ranking "
            "them with exact hard-token TMM MAE, and saving the best stacks."
        )
    )
    parser.add_argument("--config", type=str, default="configs/optollama.yaml", help="Project config YAML.")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory for the ranked pseudo-label dataset.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint override. Defaults to BEST_CHECKPOINT_PATH.")
    parser.add_argument(
        "--target-source",
        type=str,
        default="train",
        choices=["auto", "config", "file", "synthetic", "train", "test"],
        help="Where target spectra come from.",
    )
    parser.add_argument("--target-file", type=str, default=None, help="Target CSV/JSON for --target-source file.")
    parser.add_argument("--max-targets", type=int, default=1024, help="Maximum number of target spectra to process.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. Defaults to config SEED.")
    parser.add_argument("--device", type=str, default=None, help='Execution device, e.g. "cuda", "cuda:0", or "cpu".')
    parser.add_argument("--batch-size", type=int, default=8, help="Target spectra per model/TMM batch.")
    parser.add_argument("--num-candidates", type=int, default=64, help="Model samples per target.")
    parser.add_argument(
        "--rank-candidates",
        type=int,
        default=0,
        help="Raw-MAE candidates per target to rank. 0 means rank all sampled candidates.",
    )
    parser.add_argument("--keep-per-target", type=int, default=1, help="Number of ranked stacks saved per target.")
    parser.add_argument("--eval-batch-size", type=int, default=512, help="TMM simulation chunk size.")
    parser.add_argument("--shard-size", type=int, default=10000, help="Maximum saved samples per output shard.")
    parser.add_argument("--summary-examples", type=int, default=100, help="Number of selected examples stored in the JSON summary.")
    parser.add_argument("--include-stacks-in-summary", action="store_true", help="Store token-id stacks in summary examples.")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Optional model sequence length override.")
    parser.add_argument("--max-emit-len", type=int, default=None, help="Optional sampling EOS cap.")
    parser.add_argument("--output-seq-len", type=int, default=None, help="Saved sequence width. Defaults to config MAX_SEQ_LEN.")
    parser.add_argument("--roi-min", type=float, default=None, help="MAE ROI lower wavelength. Defaults to config ROI_MIN.")
    parser.add_argument("--roi-max", type=float, default=None, help="MAE ROI upper wavelength. Defaults to config ROI_MAX.")
    parser.add_argument("--min-layers", type=int, default=1, help="Minimum material layers accepted without fallback.")
    parser.add_argument("--max-layers", type=int, default=None, help="Maximum material layers accepted without fallback.")
    parser.add_argument("--metal-materials", nargs="*", default=["Ag", "Al", "TiN"], help="Materials counted as metals.")
    parser.add_argument("--max-metal-fraction", type=float, default=1.0, help="Hard validity cap for metal layer fraction.")
    parser.add_argument("--max-repeat-fraction", type=float, default=1.0, help="Hard validity cap for adjacent repeated-token fraction.")
    parser.add_argument("--metal-penalty", type=float, default=0.0, help="MAE-equivalent penalty multiplied by metal fraction.")
    parser.add_argument("--repeat-penalty", type=float, default=0.0, help="MAE-equivalent penalty multiplied by repeat fraction.")
    parser.add_argument("--length-penalty", type=float, default=0.0, help="MAE-equivalent penalty per layer.")
    parser.add_argument("--length-reward", type=float, default=0.0, help="MAE-equivalent reward per layer.")
    parser.add_argument(
        "--no-fallback-to-best-raw",
        action="store_true",
        help="If all candidates for a target fail validity filters, save nothing instead of the best raw-MAE candidate.",
    )
    return parser.parse_args()


def _token_material_masks(tokens: list[str], metal_materials: list[str]) -> torch.Tensor:
    metals = set(metal_materials)
    values = []
    for token in tokens:
        parts = optollama.data.layer_token_parts(token)
        values.append(parts is not None and parts[0] in metals)
    return torch.tensor(values, dtype=torch.bool)


def _max_repeat_run(stack: torch.Tensor, eos: int, pad: int, msk: int) -> int:
    best = 0
    current = 0
    previous: int | None = None
    for raw_id in stack.detach().cpu().tolist():
        token_id = int(raw_id)
        if token_id == int(eos):
            break
        if token_id in (int(pad), int(msk)):
            continue
        if token_id == previous:
            current += 1
        else:
            current = 1
            previous = token_id
        best = max(best, current)
    return best


def _candidate_stats(
    stacks: torch.Tensor,
    metal_token_mask: torch.Tensor,
    eos: int,
    pad: int,
    msk: int,
) -> dict[str, torch.Tensor]:
    is_eos = stacks == int(eos)
    before_eos = is_eos.cumsum(dim=1) == 0
    valid = before_eos & (stacks != int(pad)) & (stacks != int(msk))
    layers = valid.sum(dim=1).to(torch.float32)

    metal_mask = metal_token_mask.to(stacks.device)[stacks] & valid
    metal_fraction = metal_mask.sum(dim=1).to(torch.float32) / layers.clamp_min(1.0)

    if stacks.size(1) > 1:
        adjacent_valid = valid[:, :-1] & valid[:, 1:]
        adjacent_same = (stacks[:, :-1] == stacks[:, 1:]) & adjacent_valid
        repeat_fraction = adjacent_same.sum(dim=1).to(torch.float32) / (layers - 1.0).clamp_min(1.0)
    else:
        repeat_fraction = torch.zeros_like(layers)

    max_repeat_run = torch.tensor(
        [_max_repeat_run(row, eos=eos, pad=pad, msk=msk) for row in stacks],
        dtype=torch.float32,
        device=stacks.device,
    )

    return {
        "layers": layers,
        "metal_fraction": metal_fraction,
        "repeat_fraction": repeat_fraction,
        "max_repeat_run": max_repeat_run,
    }


def _rank_candidates(
    candidates: torch.Tensor,
    base_mae: torch.Tensor,
    target_indices: torch.Tensor,
    args: argparse.Namespace,
    metal_token_mask: torch.Tensor,
    eos: int,
    pad: int,
    msk: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    stats = _candidate_stats(candidates, metal_token_mask=metal_token_mask, eos=eos, pad=pad, msk=msk)
    score = (
        base_mae
        + float(args.metal_penalty) * stats["metal_fraction"]
        + float(args.repeat_penalty) * stats["repeat_fraction"]
        + float(args.length_penalty) * stats["layers"]
        - float(args.length_reward) * stats["layers"]
    )

    max_layers = int(args.max_layers if args.max_layers is not None else candidates.size(1) - 1)
    valid = (
        (stats["layers"] >= int(args.min_layers))
        & (stats["layers"] <= max_layers)
        & (stats["metal_fraction"] <= float(args.max_metal_fraction))
        & (stats["repeat_fraction"] <= float(args.max_repeat_fraction))
    )

    selected: list[int] = []
    fallback_count = 0
    keep = max(1, int(args.keep_per_target))
    fallback = not bool(args.no_fallback_to_best_raw)

    for target_idx in torch.unique(target_indices).tolist():
        rows = (target_indices == int(target_idx)).nonzero(as_tuple=False).squeeze(1)
        rows_valid = rows[valid[rows]]
        if rows_valid.numel():
            order = score[rows_valid].argsort()
            selected.extend(rows_valid[order[:keep]].tolist())
            continue
        if fallback:
            order = base_mae[rows].argsort()
            selected.extend(rows[order[:keep]].tolist())
            fallback_count += 1

    if not selected:
        selected_tensor = torch.empty((0,), dtype=torch.long, device=candidates.device)
    else:
        selected_tensor = torch.tensor(selected, dtype=torch.long, device=candidates.device)

    diagnostics = {
        "score": score,
        "valid": valid,
        "fallback_targets": fallback_count,
        **stats,
    }
    return selected_tensor, diagnostics


@torch.no_grad()
def _sample_rank_candidates(
    model: torch.nn.Module,
    targets: torch.Tensor,
    global_target_start: int,
    num_candidates: int,
    rank_candidates: int,
    tmm_ctx: optollama.evaluation.simulation.TMMContext,
    eos: int,
    pad: int,
    msk: int,
    roi_mask: torch.Tensor | None,
    eval_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    b = targets.size(0)
    m = max(1, int(num_candidates))
    keep = min(max(1, int(rank_candidates)), m)

    targets_mc = targets.unsqueeze(1).expand(b, m, *targets.shape[1:]).reshape(b * m, *targets.shape[1:])
    logits_or_ids, _ = model(targets_mc)
    ids_flat = logits_or_ids.argmax(dim=-1) if logits_or_ids.dim() == 3 else logits_or_ids
    mae_flat = lite.simulate_mae_in_chunks(
        ids_flat,
        targets_mc,
        tmm_ctx=tmm_ctx,
        eos=eos,
        pad=pad,
        msk=msk,
        roi_mask=roi_mask,
        eval_batch_size=max(1, int(eval_batch_size)),
    )

    ids = ids_flat.view(b, m, -1)
    mae = mae_flat.view(b, m)
    top_mae, top_idx = mae.topk(k=keep, dim=1, largest=False)
    gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, ids.size(-1))
    candidate_stacks = ids.gather(dim=1, index=gather_idx).reshape(b * keep, ids.size(-1))
    candidate_targets = targets.unsqueeze(1).expand(b, keep, *targets.shape[1:]).reshape(b * keep, *targets.shape[1:])
    target_indices = (
        torch.arange(global_target_start, global_target_start + b, device=targets.device)
        .unsqueeze(1)
        .expand(b, keep)
        .reshape(-1)
    )

    return candidate_stacks, candidate_targets, target_indices, top_mae.reshape(-1)


def _metadata_for_selection(
    selected: torch.Tensor,
    candidates: torch.Tensor,
    base_mae: torch.Tensor,
    target_indices: torch.Tensor,
    diagnostics: dict[str, Any],
    include_stacks: bool = False,
) -> list[dict[str, float | int | bool]]:
    rows = selected.detach().cpu().tolist()
    metadata: list[dict[str, float | int | bool]] = []
    for row in rows:
        item = {
            "target_index": int(target_indices[row].item()),
            "mae": float(base_mae[row].item()),
            "score": float(diagnostics["score"][row].item()),
            "valid": bool(diagnostics["valid"][row].item()),
            "layers": int(round(float(diagnostics["layers"][row].item()))),
            "metal_fraction": float(diagnostics["metal_fraction"][row].item()),
            "repeat_fraction": float(diagnostics["repeat_fraction"][row].item()),
            "max_repeat_run": int(round(float(diagnostics["max_repeat_run"][row].item()))),
        }
        if include_stacks:
            item["stack"] = candidates[row].detach().cpu().tolist()
        metadata.append(item)
    return metadata


def _buffer_size(items: list[torch.Tensor]) -> int:
    return int(sum(tensor.size(0) for tensor in items))


def _flush_shard(
    out_dir: Path,
    shard_idx: int,
    spectra_parts: list[torch.Tensor],
    stack_parts: list[torch.Tensor],
) -> tuple[int, str | None]:
    if not spectra_parts:
        return 0, None
    spectra = torch.cat(spectra_parts, dim=0).to(torch.float32)
    stacks = torch.cat(stack_parts, dim=0).long()
    out_path = out_dir / f"tmm-ranked-{shard_idx}.safetensors"
    safetensors.torch.save_file({"spectra": spectra, "thin_films": stacks}, str(out_path))
    return int(stacks.size(0)), str(out_path)


def main() -> None:
    args = parse_args()
    cfg = lite.load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg["SEED"])
    optollama.utils.set_all_seeds(seed)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cudnn.benchmark = True

    if args.checkpoint:
        cfg["BEST_CHECKPOINT_PATH"] = args.checkpoint
    if args.max_seq_len is not None:
        cfg["MAX_SEQ_LEN"] = int(args.max_seq_len)
        if args.max_emit_len is None:
            cfg["MAX_EMIT_LEN"] = int(args.max_seq_len)
    if args.max_emit_len is not None:
        cfg["MAX_EMIT_LEN"] = int(args.max_emit_len)
    output_seq_len = int(args.output_seq_len if args.output_seq_len is not None else cfg["MAX_SEQ_LEN"])

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lite.copy_tokens_file(cfg, out_dir)

    tokens, token_to_idx, idx_to_token, _, _, _, eos_idx, pad_idx, msk_idx = optollama.data.init_tokens(cfg["TOKENS_PATH"])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1009)
    targets_cpu, target_source = lite.load_targets(args, cfg, msk_idx=msk_idx, generator=generator)
    print(f"Loaded {targets_cpu.size(0)} target spectra from {target_source}.")

    roi_mask = lite.build_roi_mask(args, cfg, device)
    tmm_ctx = optollama.evaluation.simulation.TMMContext.make(cfg, idx_to_token, device=device)
    model = lite.build_model_from_config(
        cfg,
        sample_spectrum=targets_cpu[0],
        idx_to_token=idx_to_token,
        msk_idx=msk_idx,
        pad_idx=pad_idx,
        eos_idx=eos_idx,
        device=device,
    )
    lite.apply_token_constraints_if_configured(model, cfg, tokens, token_to_idx)

    metal_token_mask = _token_material_masks(tokens, list(args.metal_materials))
    rank_candidates = int(args.rank_candidates) if int(args.rank_candidates) > 0 else int(args.num_candidates)
    rank_candidates = min(rank_candidates, int(args.num_candidates))

    out_spectra: list[torch.Tensor] = []
    out_stacks: list[torch.Tensor] = []
    summary_examples: list[dict[str, float | int | bool]] = []
    saved_files: list[dict[str, int | str]] = []
    shard_idx = 0
    saved_count = 0
    mae_sum = 0.0
    layers_sum = 0.0
    fallback_targets = 0
    candidate_total = 0
    shard_size = max(1, int(args.shard_size))
    summary_limit = max(0, int(args.summary_examples))

    pbar = tqdm.tqdm(range(0, targets_cpu.size(0), int(args.batch_size)), desc="tmm-ranked")
    for start in pbar:
        end = min(start + int(args.batch_size), targets_cpu.size(0))
        targets = targets_cpu[start:end].to(device, non_blocking=True)

        candidates, candidate_targets, target_indices, base_mae = _sample_rank_candidates(
            model,
            targets,
            global_target_start=start,
            num_candidates=int(args.num_candidates),
            rank_candidates=rank_candidates,
            tmm_ctx=tmm_ctx,
            eos=eos_idx,
            pad=pad_idx,
            msk=msk_idx,
            roi_mask=roi_mask,
            eval_batch_size=int(args.eval_batch_size),
        )
        candidate_total += candidates.size(0)

        encoded_candidates, _ = optollama.data.reencode_stacks_for_output(
            candidates,
            output_seq_len=output_seq_len,
            eos=eos_idx,
            pad=pad_idx,
            msk=msk_idx,
        )
        selected, diagnostics = _rank_candidates(
            encoded_candidates,
            base_mae=base_mae,
            target_indices=target_indices,
            args=args,
            metal_token_mask=metal_token_mask,
            eos=eos_idx,
            pad=pad_idx,
            msk=msk_idx,
        )
        fallback_targets += int(diagnostics["fallback_targets"])

        if selected.numel():
            out_spectra.append(candidate_targets[selected].detach().cpu())
            out_stacks.append(encoded_candidates[selected].detach().cpu())
            batch_metadata = _metadata_for_selection(
                selected,
                encoded_candidates,
                base_mae,
                target_indices,
                diagnostics,
                include_stacks=bool(args.include_stacks_in_summary),
            )
            for item in batch_metadata:
                saved_count += 1
                mae_sum += float(item["mae"])
                layers_sum += float(item["layers"])
                if len(summary_examples) < summary_limit:
                    summary_examples.append(item)

        if _buffer_size(out_stacks) >= shard_size:
            written, path = _flush_shard(out_dir, shard_idx, out_spectra, out_stacks)
            if path is not None:
                saved_files.append({"path": path, "samples": written})
                shard_idx += 1
            out_spectra.clear()
            out_stacks.clear()

        pbar.set_postfix(saved=saved_count, candidates=candidate_total, fallback=fallback_targets)

    written, path = _flush_shard(out_dir, shard_idx, out_spectra, out_stacks)
    if path is not None:
        saved_files.append({"path": path, "samples": written})

    summary: dict[str, Any] = {
        "target_source": target_source,
        "targets": int(targets_cpu.size(0)),
        "candidate_samples_per_target": int(args.num_candidates),
        "ranked_candidates_per_target": int(rank_candidates),
        "keep_per_target": int(args.keep_per_target),
        "saved": int(saved_count),
        "saved_files": saved_files,
        "fallback_targets": int(fallback_targets),
        "output_seq_len": int(output_seq_len),
        "min_layers": int(args.min_layers),
        "max_layers": int(args.max_layers if args.max_layers is not None else output_seq_len - 1),
        "max_metal_fraction": float(args.max_metal_fraction),
        "max_repeat_fraction": float(args.max_repeat_fraction),
        "metal_penalty": float(args.metal_penalty),
        "repeat_penalty": float(args.repeat_penalty),
        "length_penalty": float(args.length_penalty),
        "length_reward": float(args.length_reward),
        "summary_examples": summary_examples,
    }

    if saved_count:
        summary["mean_mae"] = mae_sum / float(saved_count)
        summary["mean_layers"] = layers_sum / float(saved_count)
        print(
            f"Saved {saved_count} ranked pseudo-labels across {len(saved_files)} shard(s) "
            f"(mean MAE: {summary['mean_mae']:.6f}, mean layers: {summary['mean_layers']:.1f})"
        )
    else:
        print("No candidates were selected; no safetensor shard was written.")

    summary_path = out_dir / "tmm_ranked_summary.json"
    optollama.utils.save_as_json(str(summary_path), summary)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
