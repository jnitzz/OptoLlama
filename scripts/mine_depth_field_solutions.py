#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import tqdm

import optollama.data
import optollama.evaluation.simulation
import optollama.model
import optollama.utils
from scripts.inference_depth_field import load_depth_model, resolve_device, resolve_tmm_device, simulate_field_runs

# ruff: noqa: D103


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine topology-distinct depth-field alternatives and verify them with exact TMM.")
    parser.add_argument("--config", default="configs/depth_field_hybrid_24_selftrain.yaml")
    parser.add_argument("--checkpoint", default=None, help="Source depth-field checkpoint.")
    parser.add_argument("--out-dir", default=None, help="Versioned solution-bank output directory.")
    parser.add_argument("--split", choices=["train"], default="train")
    parser.add_argument("--anchor-offset", type=int, default=None)
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Anchor spectra per model batch on each rank.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mc-samples", type=int, default=None)
    parser.add_argument("--mc-batch-size", type=int, default=None)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--remask-strategy", choices=["random", "confidence"], default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tmm-device", default=None)
    parser.add_argument("--tmm-batch-size", type=int, default=None)
    parser.add_argument("--gold-max-mae", type=float, default=None)
    parser.add_argument("--silver-max-mae", type=float, default=None)
    parser.add_argument("--derivative-weight", type=float, default=None)
    parser.add_argument("--keep-gold-per-target", type=int, default=None)
    parser.add_argument("--keep-silver-per-target", type=int, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--topology-thickness-quantum-nm", type=float, default=None)
    parser.add_argument("--shard-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def nested_get(mapping: dict[str, Any], *path: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def apply_defaults(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    mining = nested_get(cfg, "SELF_TRAIN", "MINING", default={}) or {}
    eval_cfg = nested_get(cfg, "DEPTH_FIELD", "EVAL", default={}) or {}
    defaults = {
        "checkpoint": mining.get("CHECKPOINT"),
        "out_dir": mining.get("OUT_DIR"),
        "anchor_offset": mining.get("ANCHOR_OFFSET", 0),
        "max_targets": mining.get("MAX_TARGETS", 1000),
        "batch_size": mining.get("BATCH_SIZE", 1),
        "mc_samples": mining.get("MC_SAMPLES", 16),
        "mc_batch_size": mining.get("MC_BATCH_SIZE", 4),
        "sampling_steps": mining.get("SAMPLING_STEPS", eval_cfg.get("SAMPLING_STEPS", 200)),
        "temperature": mining.get("TEMPERATURE", eval_cfg.get("TEMPERATURE", 1.0)),
        "top_k": mining.get("TOP_K", eval_cfg.get("TOP_K", 0)),
        "cfg_scale": mining.get("CFG_SCALE", 1.0),
        "remask_strategy": mining.get("REMASK_STRATEGY", eval_cfg.get("REMASK_STRATEGY", "random")),
        "tmm_device": mining.get("TMM_DEVICE", eval_cfg.get("TMM_DEVICE", "auto")),
        "tmm_batch_size": mining.get("TMM_BATCH_SIZE", eval_cfg.get("TMM_BATCH_SIZE", 16)),
        "gold_max_mae": mining.get("GOLD_MAX_MAE", 0.01),
        "silver_max_mae": mining.get("SILVER_MAX_MAE", 0.02),
        "derivative_weight": mining.get("DERIVATIVE_WEIGHT", 0.25),
        "keep_gold_per_target": mining.get("KEEP_GOLD_PER_TARGET", 3),
        "keep_silver_per_target": mining.get("KEEP_SILVER_PER_TARGET", 2),
        "max_runs": mining.get("MAX_RUNS", max(1, int(cfg["MAX_SEQ_LEN"]) - 1)),
        "topology_thickness_quantum_nm": mining.get("TOPOLOGY_THICKNESS_QUANTUM_NM", 25.0),
        "shard_size": mining.get("SHARD_SIZE", 2048),
        "seed": cfg.get("SEED", 3),
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)

    required = ["checkpoint", "out_dir"]
    missing = [name for name in required if getattr(args, name) in (None, "")]
    if missing:
        raise ValueError(f"Missing required mining settings: {missing}")
    if int(args.max_targets) <= 0 or int(args.mc_samples) <= 0 or int(args.mc_batch_size) <= 0:
        raise ValueError("MAX_TARGETS, MC_SAMPLES, and MC_BATCH_SIZE must be positive.")
    if not 0.0 <= float(args.gold_max_mae) <= float(args.silver_max_mae):
        raise ValueError("Acceptance thresholds must satisfy 0 <= GOLD_MAX_MAE <= SILVER_MAX_MAE.")
    if int(args.keep_gold_per_target) < 0 or int(args.keep_silver_per_target) < 0:
        raise ValueError("Per-target retention counts must be non-negative.")


def split_paths(cfg: dict[str, Any], split: str) -> list[str]:
    prefix = "DATA_PATH_TRAIN" if split == "train" else "DATA_PATH_TEST"
    paths = sorted(str(cfg[key]) for key in cfg if key == prefix or key.startswith(f"{prefix}_"))
    if not paths:
        raise ValueError(f"No configured {split} data paths were found.")
    return paths


def select_topology_distinct_candidates(
    *,
    level_mae: torch.Tensor,
    score: torch.Tensor,
    topology_hash: torch.Tensor,
    run_count: torch.Tensor,
    active_bins: torch.Tensor,
    batch_size: int,
    mc_samples: int,
    gold_max_mae: float,
    silver_max_mae: float,
    keep_gold: int,
    keep_silver: int,
    max_runs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select low-error candidates while retaining distinct complete topologies."""
    selected: list[int] = []
    selected_tiers: list[int] = []
    for anchor_row in range(int(batch_size)):
        start = anchor_row * int(mc_samples)
        stop = start + int(mc_samples)
        rows = list(range(start, stop))
        valid_rows = [
            row
            for row in rows
            if int(active_bins[row]) > 0
            and int(run_count[row]) <= int(max_runs)
            and math.isfinite(float(level_mae[row]))
            and math.isfinite(float(score[row]))
        ]
        used_topologies: set[int] = set()
        tier_specs = (
            (optollama.data.QUALITY_GOLD, float(gold_max_mae), int(keep_gold), None),
            (optollama.data.QUALITY_SILVER, float(silver_max_mae), int(keep_silver), float(gold_max_mae)),
        )
        for tier, maximum, keep, lower_exclusive in tier_specs:
            if keep <= 0:
                continue
            eligible = [
                row
                for row in valid_rows
                if float(level_mae[row]) <= maximum and (lower_exclusive is None or float(level_mae[row]) > lower_exclusive)
            ]
            eligible.sort(key=lambda row: (float(score[row]), float(level_mae[row]), row))
            kept = 0
            for row in eligible:
                topology = int(topology_hash[row])
                if topology in used_topologies:
                    continue
                selected.append(row)
                selected_tiers.append(int(tier))
                used_topologies.add(topology)
                kept += 1
                if kept >= keep:
                    break
    return torch.tensor(selected, dtype=torch.long), torch.tensor(selected_tiers, dtype=torch.uint8)


def corruption_config(cfg: dict[str, Any]) -> optollama.model.DepthFieldCorruptionConfig:
    block = nested_get(cfg, "DEPTH_FIELD", "CORRUPTION", default={}) or {}
    return optollama.model.DepthFieldCorruptionConfig.from_dict(block)


def checkpoint_identity(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    stat = path.stat()
    return {"path": str(path), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = optollama.utils.load_config(args)
    apply_defaults(cfg, args)
    if args.seed is not None:
        cfg["SEED"] = int(args.seed)

    device, _local_rank, rank, world_size = optollama.utils.setup_run(cfg, make_dirs=False)
    if args.device is not None and world_size == 1:
        device = resolve_device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
    optollama.utils.set_all_seeds(int(args.seed) + rank * 100_003)
    tmm_device = resolve_tmm_device(args.tmm_device, device)

    out_dir = Path(args.out_dir)
    existing_bank = torch.tensor(
        [0],
        dtype=torch.int32,
        device=device if device.type == "cuda" else torch.device("cpu"),
    )
    if rank == 0:
        if out_dir.exists() and (
            (out_dir / "manifest.json").exists()
            or any(out_dir.glob("rank-*-manifest.json"))
            or any(out_dir.glob("solution-bank-rank*-shard*.safetensors"))
        ):
            existing_bank.fill_(1)
        out_dir.mkdir(parents=True, exist_ok=True)
    if optollama.utils.is_ddp():
        torch.distributed.broadcast(existing_bank, src=0)
        torch.distributed.barrier()
    if bool(existing_bank.item()):
        raise FileExistsError(f"Refusing to overwrite existing solution bank {out_dir}; use a new versioned cycle directory.")

    tokens, token_to_idx, idx_to_token, _, _, _, eos_idx, pad_idx, msk_idx = optollama.data.init_tokens(cfg["TOKENS_PATH"])
    vocab = optollama.data.build_depth_field_vocab(tokens, token_to_idx)
    model, extra = load_depth_model(str(args.checkpoint), device)
    depth_info = extra.get("depth_field") or {}
    dz_nm = float(depth_info.get("dz_nm", nested_get(cfg, "DEPTH_FIELD", "DZ_NM")))
    max_thickness_nm = float(depth_info.get("max_thickness_nm", nested_get(cfg, "DEPTH_FIELD", "MAX_THICKNESS_NM")))
    depth_bins = optollama.data.depth_bins_for(max_thickness_nm, dz_nm)
    if int(model.depth_bins) != depth_bins or int(model.num_materials) != int(vocab.num_clean_classes):
        raise RuntimeError("Checkpoint depth-field dimensions or material vocabulary do not match the mining config.")

    dataset = optollama.data.ShardedSpectraDataset(
        split_paths(cfg, args.split),
        split=args.split,
        subset_n=int(args.max_targets),
        start_index=int(args.anchor_offset),
        rank=rank,
        world_size=world_size,
        seed=int(args.seed),
        shuffle=False,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    tmm_ctx = optollama.evaluation.simulation.TMMContext.make(cfg=cfg, idx_to_token=idx_to_token, device=tmm_device)
    material_to_token_id = optollama.data.depth_field_material_token_ids(vocab)
    corruption = corruption_config(cfg)
    metadata = {
        "config": str(args.config),
        "checkpoint": checkpoint_identity(args.checkpoint),
        "material_names": list(vocab.material_names),
        "dz_nm": dz_nm,
        "max_thickness_nm": max_thickness_nm,
        "depth_bins": depth_bins,
        "spectrum_width": int(model.config.spectrum_shape[-1]),
        "rank": rank,
        "world_size": world_size,
    }
    writer = optollama.data.SolutionBankShardWriter(
        out_dir,
        rank=rank,
        shard_size=int(args.shard_size),
        metadata=metadata,
    )

    processed = 0
    accepted_gold = 0
    accepted_silver = 0
    generated = 0
    progress = tqdm.tqdm(loader, desc="mine depth-field solutions", disable=rank != 0, unit="anchor-batch")
    for spectra_cpu, _stacks_cpu, anchor_indices in progress:
        batch_size = int(spectra_cpu.size(0))
        spectra = spectra_cpu.to(device, non_blocking=True)
        field_parts: list[torch.Tensor] = []
        pred_parts: list[torch.Tensor] = []
        for mc_start in range(0, int(args.mc_samples), int(args.mc_batch_size)):
            chunk = min(int(args.mc_batch_size), int(args.mc_samples) - mc_start)
            spectra_rep = spectra.repeat_interleave(chunk, dim=0)
            sampled = model.sample(
                spectra_rep,
                steps=int(args.sampling_steps),
                temperature=float(args.temperature),
                top_k=int(args.top_k),
                deterministic=bool(float(args.temperature) <= 0.0),
                guidance_scale=float(args.cfg_scale),
                remask_strategy=str(args.remask_strategy),
                corruption_config=corruption,
            )
            compacted = optollama.data.compact_depth_fields(sampled.detach().cpu(), vocab.void_id)
            predicted = simulate_field_runs(
                compacted,
                vocab=vocab,
                tmm_ctx=tmm_ctx,
                material_to_token_id=material_to_token_id,
                dz_nm=dz_nm,
                eos_idx=eos_idx,
                pad_idx=pad_idx,
                msk_idx=msk_idx,
                tmm_batch_size=int(args.tmm_batch_size),
            )
            field_parts.append(compacted.view(batch_size, chunk, -1))
            pred_parts.append(predicted.detach().cpu().view(batch_size, chunk, *predicted.shape[1:]))

        fields = torch.cat(field_parts, dim=1).reshape(batch_size * int(args.mc_samples), depth_bins)
        pred_spectra = torch.cat(pred_parts, dim=1).reshape(batch_size * int(args.mc_samples), 3, -1)
        targets = spectra_cpu.repeat_interleave(int(args.mc_samples), dim=0)
        level_mae, derivative_mae = optollama.data.spectrum_error_metrics(
            pred_spectra,
            targets,
            wavelengths=cfg["WAVELENGTHS"],
            channels=(0, 2),
            roi_min=cfg.get("ROI_MIN"),
            roi_max=cfg.get("ROI_MAX"),
        )
        score = level_mae + float(args.derivative_weight) * derivative_mae
        topology_hash, run_count, active_bins = optollama.data.field_topology_statistics(
            fields,
            void_id=vocab.void_id,
            dz_nm=dz_nm,
            thickness_quantum_nm=float(args.topology_thickness_quantum_nm),
        )
        selected, selected_tiers = select_topology_distinct_candidates(
            level_mae=level_mae,
            score=score,
            topology_hash=topology_hash,
            run_count=run_count,
            active_bins=active_bins,
            batch_size=batch_size,
            mc_samples=int(args.mc_samples),
            gold_max_mae=float(args.gold_max_mae),
            silver_max_mae=float(args.silver_max_mae),
            keep_gold=int(args.keep_gold_per_target),
            keep_silver=int(args.keep_silver_per_target),
            max_runs=int(args.max_runs),
        )
        if selected.numel():
            anchor_rows = torch.div(selected, int(args.mc_samples), rounding_mode="floor")
            candidate_rows = selected.remainder(int(args.mc_samples))
            writer.append(
                {
                    "anchor_spectra": targets[selected].to(torch.float32),
                    "fields": fields[selected].to(torch.int16),
                    "pred_spectra": pred_spectra[selected].to(torch.float32),
                    "anchor_indices": anchor_indices[anchor_rows].to(torch.int64),
                    "candidate_indices": candidate_rows.to(torch.int32),
                    "level_mae": level_mae[selected].to(torch.float32),
                    "derivative_mae": derivative_mae[selected].to(torch.float32),
                    "score": score[selected].to(torch.float32),
                    "quality_tier": selected_tiers,
                    "topology_hash": topology_hash[selected].to(torch.int64),
                    "run_count": run_count[selected].to(torch.int32),
                    "active_bins": active_bins[selected].to(torch.int32),
                }
            )
            accepted_gold += int((selected_tiers == optollama.data.QUALITY_GOLD).sum().item())
            accepted_silver += int((selected_tiers == optollama.data.QUALITY_SILVER).sum().item())
        processed += batch_size
        generated += batch_size * int(args.mc_samples)
        if rank == 0:
            progress.set_postfix(gold=accepted_gold, silver=accepted_silver, processed=processed)
    progress.close()
    shard_paths = writer.close()

    rank_manifest = {
        "schema_version": optollama.data.SOLUTION_BANK_SCHEMA_VERSION,
        "rank": rank,
        "world_size": world_size,
        "anchors_processed": processed,
        "candidates_generated": generated,
        "accepted_gold": accepted_gold,
        "accepted_silver": accepted_silver,
        "shards": [path.name for path in shard_paths],
        "settings": {
            "seed": int(args.seed),
            "anchor_offset": int(args.anchor_offset),
            "max_targets": int(args.max_targets),
            "mc_samples": int(args.mc_samples),
            "mc_batch_size": int(args.mc_batch_size),
            "sampling_steps": int(args.sampling_steps),
            "temperature": float(args.temperature),
            "top_k": int(args.top_k),
            "cfg_scale": float(args.cfg_scale),
            "remask_strategy": str(args.remask_strategy),
            "gold_max_mae": float(args.gold_max_mae),
            "silver_max_mae": float(args.silver_max_mae),
            "derivative_weight": float(args.derivative_weight),
            "topology_thickness_quantum_nm": float(args.topology_thickness_quantum_nm),
        },
    }
    optollama.utils.save_as_json(str(out_dir / f"rank-{rank:05d}-manifest.json"), rank_manifest)
    if optollama.utils.is_ddp():
        torch.distributed.barrier()

    if rank == 0:
        rank_manifests = []
        for path in sorted(out_dir.glob("rank-*-manifest.json")):
            with path.open("r", encoding="utf-8") as handle:
                rank_manifests.append(json.load(handle))
        manifest = {
            "schema_version": optollama.data.SOLUTION_BANK_SCHEMA_VERSION,
            "config": str(args.config),
            "checkpoint": checkpoint_identity(args.checkpoint),
            "world_size": world_size,
            "anchors_processed": sum(int(item["anchors_processed"]) for item in rank_manifests),
            "candidates_generated": sum(int(item["candidates_generated"]) for item in rank_manifests),
            "accepted_gold": sum(int(item["accepted_gold"]) for item in rank_manifests),
            "accepted_silver": sum(int(item["accepted_silver"]) for item in rank_manifests),
            "shards": [shard for item in rank_manifests for shard in item["shards"]],
            "rank_manifests": [f"rank-{int(item['rank']):05d}-manifest.json" for item in rank_manifests],
            "settings": rank_manifests[0]["settings"] if rank_manifests else {},
        }
        manifest["acceptance_rate"] = (int(manifest["accepted_gold"]) + int(manifest["accepted_silver"])) / max(
            int(manifest["candidates_generated"]), 1
        )
        optollama.utils.save_as_json(str(out_dir / "manifest.json"), manifest)
        print(
            "Solution bank complete: "
            f"anchors={manifest['anchors_processed']}, generated={manifest['candidates_generated']}, "
            f"gold={manifest['accepted_gold']}, silver={manifest['accepted_silver']}, out={out_dir}"
        )

    optollama.utils.stop_ddp()


if __name__ == "__main__":
    main()
