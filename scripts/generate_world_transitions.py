#!/usr/bin/env python

import argparse
import os
from pathlib import Path
from typing import Iterator, Optional

import safetensors.torch
import torch
import tqdm

import optollama.data
import optollama.evaluation
import optollama.model
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
    p.add_argument(
        "--state-source",
        type=str,
        default=None,
        choices=["perturbed_truth", "proposal", "mixed"],
        help="How to create current states before local edit perturbations.",
    )
    p.add_argument("--proposal-mc-samples", type=int, default=None)
    p.add_argument("--proposal-keep-candidates", type=int, default=None)
    p.add_argument("--proposal-checkpoint", type=str, default=None)
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


def resolve_source_shards(cfg: dict, split: str) -> list[Path]:
    """
    Resolve configured source split paths without loading tensors into memory.
    """
    prefix = "DATA_PATH_TRAIN" if split == "train" else "DATA_PATH_TEST"
    raw_paths = sorted([cfg[key] for key in cfg if key.startswith(prefix)])
    files: list[Path] = []

    for raw_path in raw_paths:
        path = Path(raw_path)
        if path.is_dir():
            files.extend(sorted(path.glob("*.safetensors")))
        elif path.suffix == ".safetensors":
            files.append(path)

    if not files:
        raise FileNotFoundError(f"No .safetensors source shards found for split {split!r}.")

    return sorted(files, key=lambda item: optollama.data.SpectraDataset.shard_sort_key(str(item.with_suffix(""))))


def iter_source_batches(
    cfg: dict,
    split: str,
    max_samples: int,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[Iterator[tuple[torch.Tensor, torch.Tensor]], int]:
    """
    Stream source spectra/stack batches shard by shard.

    ``SpectraDataset`` concatenates all configured shards before subsetting,
    which is too memory-hungry for the length-101 data. This iterator keeps
    only one source shard and one mini-batch in memory at a time.
    """
    files = resolve_source_shards(cfg, split)
    if bool(cfg.get("WORLD_SHUFFLE_SOURCE_SHARDS", False)):
        order = torch.randperm(len(files), generator=generator).tolist()
        files = [files[int(i)] for i in order]

    target_samples = max(1, int(max_samples))
    expected_batches = (target_samples + int(batch_size) - 1) // int(batch_size)

    def _iterator():
        emitted = 0
        for shard_path in files:
            if emitted >= target_samples:
                break

            data = safetensors.torch.load_file(str(shard_path), device="cpu")
            spectra = data["spectra"].to(torch.float32)
            stacks = data["thin_films"].long()

            remaining = target_samples - emitted
            take = min(remaining, int(spectra.size(0)))
            if take <= 0:
                continue

            if bool(cfg.get("WORLD_SHUFFLE_SOURCE_ROWS", False)):
                rows = torch.randperm(int(spectra.size(0)), generator=generator)[:take]
                spectra = spectra[rows]
                stacks = stacks[rows]
            else:
                spectra = spectra[:take]
                stacks = stacks[:take]

            for start in range(0, take, int(batch_size)):
                end = min(start + int(batch_size), take)
                emitted += end - start
                yield spectra[start:end], stacks[start:end]

    return _iterator(), expected_batches


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


def apply_token_constraints_if_configured(
    model: torch.nn.Module,
    cfg: dict,
    tokens: list[str],
    token_to_idx: dict[str, int],
) -> None:
    """
    Apply the same optional inference constraints used by OptoLlama sampling.
    """
    if not cfg.get("TOKEN_FILTER_ENABLED", False):
        return

    material_groups = optollama.data.make_material_groups(tokens, token_to_idx)
    material_token_ids = optollama.data.make_material_token_ids(token_to_idx)
    mode = cfg["TOKEN_FILTER_MODE"]
    groups = cfg["TOKEN_FILTER_GROUPS"]
    allow_group_ids, exclude_group_ids = [], []

    for group in groups:
        if group not in material_groups:
            raise ValueError(f"Unknown TOKEN_FILTER_GROUPS entry: {group!r}.")
        (allow_group_ids if mode == "allow" else exclude_group_ids).append(material_groups[group])

    allow_ids = torch.cat(allow_group_ids, dim=0) if allow_group_ids else torch.empty((0,), dtype=torch.long)
    exclude_ids = torch.cat(exclude_group_ids, dim=0) if exclude_group_ids else torch.empty((0,), dtype=torch.long)

    if cfg.get("TOKEN_FILTER_ALLOW_TOKENS"):
        allow_ids = torch.unique(
            torch.cat(
                [
                    allow_ids,
                    optollama.data.token_ids_of(cfg["TOKEN_FILTER_ALLOW_TOKENS"], token_to_idx, material_token_ids),
                ],
                dim=0,
            )
        )
    if cfg.get("TOKEN_FILTER_EXCLUDE_TOKENS"):
        exclude_ids = torch.unique(
            torch.cat(
                [
                    exclude_ids,
                    optollama.data.token_ids_of(cfg["TOKEN_FILTER_EXCLUDE_TOKENS"], token_to_idx, material_token_ids),
                ],
                dim=0,
            )
        )

    model.set_token_constraints(
        allow_ids=allow_ids if allow_ids.numel() else None,
        exclude_ids=exclude_ids if exclude_ids.numel() else None,
    )


def build_proposal_model(
    cfg: dict,
    sample_spectrum: torch.Tensor,
    idx_to_token: dict[int, str],
    eos_idx: int,
    pad_idx: int,
    msk_idx: int,
    device: torch.device,
    checkpoint: str,
) -> torch.nn.Module:
    """
    Build and load OptoLlama for proposal-seeded transition generation.
    """
    model = optollama.model.build_model(
        model_type=cfg["MODEL"],
        sample_spectrum=sample_spectrum,
        vocab_size=len(idx_to_token),
        max_stack_depth=cfg["MAX_SEQ_LEN"],
        d_model=cfg["D_MODEL"],
        n_blocks=cfg["N_BLOCKS"],
        n_heads=cfg["N_HEADS"],
        timesteps=cfg.get("DIFFUSION_STEPS", None),
        dropout=cfg["DROPOUT"],
        idx_to_token=idx_to_token,
        mask_idx=msk_idx,
        pad_idx=pad_idx,
        eos_idx=eos_idx,
        device=device,
        temperature=cfg["TEMPERATURE"],
        top_k=cfg["TOP_K"],
        top_p=cfg["TOP_P"],
    ).to(device)
    optollama.utils.load_checkpoint(checkpoint, model, map_location="cpu", strict=True)
    try:
        model.set_max_emit_len(cfg.get("MAX_EMIT_LEN", cfg["MAX_SEQ_LEN"]))
    except AttributeError:
        pass
    return model.eval()


def mutation_kwargs_from_cfg(cfg: dict) -> dict:
    """
    Return mutation-proposal weights from config.
    """
    return {
        "insertion_prob": float(cfg.get("WORLD_MUTATION_INSERT_PROB", 0.25)),
        "material_prob": float(cfg.get("WORLD_MUTATION_MATERIAL_PROB", 0.25)),
        "thickness_prob": float(cfg.get("WORLD_MUTATION_THICKNESS_PROB", 0.45)),
        "delete_prob": float(cfg.get("WORLD_MUTATION_DELETE_PROB", 0.05)),
        "pair_insert_prob": float(cfg.get("WORLD_MUTATION_PAIR_INSERT_PROB", 0.10)),
        "swap_prob": float(cfg.get("WORLD_MUTATION_SWAP_PROB", 0.05)),
        "scale_prob": float(cfg.get("WORLD_MUTATION_SCALE_PROB", 0.10)),
        "thickness_deltas": cfg.get("WORLD_MUTATION_THICKNESS_DELTAS", [-50, -40, -30, -20, -10, 10, 20, 30, 40, 50]),
        "thickness_scale_factors": cfg.get("WORLD_MUTATION_SCALE_FACTORS", [0.85, 0.9, 0.95, 1.05, 1.1, 1.15]),
    }


@torch.no_grad()
def sample_proposal_states(
    model: torch.nn.Module,
    targets: torch.Tensor,
    mc_samples: int,
    keep_candidates: int,
    tmm_ctx: optollama.evaluation.simulation.TMMContext,
    eos: int,
    pad: int,
    msk: int,
    roi_mask: torch.Tensor,
    output_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample OptoLlama candidates and return top current states per target.
    """
    b = targets.size(0)
    m = max(1, int(mc_samples))
    keep = min(max(1, int(keep_candidates)), m)
    targets_mc = targets.unsqueeze(1).expand(b, m, *targets.shape[1:]).reshape(b * m, *targets.shape[1:])
    logits_or_ids, _ = model(targets_mc)
    ids_flat = logits_or_ids.argmax(dim=-1) if logits_or_ids.dim() == 3 else logits_or_ids
    ids_flat, _ = optollama.data.reencode_stacks_for_output(
        ids_flat,
        output_seq_len=output_seq_len,
        eos=eos,
        pad=pad,
        msk=msk,
    )
    spectra_flat = optollama.evaluation.simulation.simulate_token_sequence(ids_flat, tmm_ctx, eos=eos, pad=pad, msk=msk)
    mae = optollama.evaluation.masked_mae_roi(targets_mc, spectra_flat, wl_mask=roi_mask).view(b, m)
    ids = ids_flat.view(b, m, -1)
    spectra = spectra_flat.view(b, m, *spectra_flat.shape[1:])

    top_mae, top_idx = mae.topk(k=keep, dim=1, largest=False)
    gather_ids = top_idx.unsqueeze(-1).expand(-1, -1, ids.size(-1))
    gather_spec = top_idx.view(b, keep, 1, 1).expand(-1, -1, spectra.size(-2), spectra.size(-1))
    states = ids.gather(dim=1, index=gather_ids).reshape(b * keep, ids.size(-1))
    state_spectra = spectra.gather(dim=1, index=gather_spec).reshape(b * keep, *spectra.shape[2:])
    state_targets = targets.unsqueeze(1).expand(b, keep, *targets.shape[1:]).reshape(b * keep, *targets.shape[1:])
    return states, state_targets, state_spectra, top_mae.reshape(-1)


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
    state_source = str(args.state_source or cfg.get("WORLD_STATE_SOURCE", "proposal"))
    if state_source not in ("perturbed_truth", "proposal", "mixed"):
        raise ValueError("WORLD_STATE_SOURCE must be one of: perturbed_truth, proposal, mixed.")
    proposal_mc_samples = int(args.proposal_mc_samples or cfg.get("WORLD_PROPOSAL_MC_SAMPLES", cfg.get("MC_SAMPLES", 10)))
    proposal_keep_candidates = int(args.proposal_keep_candidates or cfg.get("WORLD_PROPOSAL_KEEP_CANDIDATES", 4))
    proposal_checkpoint = args.proposal_checkpoint or cfg.get("WORLD_PROPOSAL_CHECKPOINT_PATH", cfg["BEST_CHECKPOINT_PATH"])
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
    mutation_kwargs = mutation_kwargs_from_cfg(cfg)
    proposal_model = None
    if state_source in ("proposal", "mixed"):
        if not proposal_checkpoint or not os.path.exists(proposal_checkpoint):
            raise FileNotFoundError(f"Proposal checkpoint does not exist: {proposal_checkpoint}")
        proposal_model = build_proposal_model(
            cfg,
            sample_spectrum=torch.zeros((3, cfg["WAVELENGTHS"].numel()), dtype=torch.float32),
            idx_to_token=idx_to_token,
            eos_idx=eos_idx,
            pad_idx=pad_idx,
            msk_idx=msk_idx,
            device=device,
            checkpoint=proposal_checkpoint,
        )
        apply_token_constraints_if_configured(proposal_model, cfg, tokens, token_to_idx)

    output_seq_len = int(cfg.get("WORLD_OUTPUT_SEQ_LEN", cfg["MAX_SEQ_LEN"]))
    max_layers = int(cfg.get("WORLD_MAX_LAYERS", output_seq_len - 1))
    source_batches, expected_batches = iter_source_batches(
        cfg,
        split=source_split,
        max_samples=num_base_samples,
        batch_size=batch_size,
        generator=generator,
    )

    buffers: dict[str, list[torch.Tensor]] = {key: [] for key in optollama.data.WORLD_TRANSITION_KEYS}
    rows_in_buffer = 0
    total_rows = 0
    shard_idx = 0

    pbar = tqdm.tqdm(source_batches, total=expected_batches, desc="world transitions")
    for spectra_cpu, stacks_cpu in pbar:
        if total_rows >= num_transitions:
            break

        targets = spectra_cpu.to(device, non_blocking=True).to(torch.float32)
        base_stacks = stacks_cpu.to(device, non_blocking=True).long()
        base_stacks, _ = optollama.data.reencode_stacks_for_output(
            base_stacks,
            output_seq_len=output_seq_len,
            eos=eos_idx,
            pad=pad_idx,
            msk=msk_idx,
        )

        current_batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

        if state_source in ("perturbed_truth", "mixed") and current_edits > 0:
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
                **mutation_kwargs,
            )
            current_targets = targets[current_source]
            current_base_stacks = base_stacks[current_source]
            current_spectra = simulate_in_chunks(current_stacks, tmm_ctx, eos_idx, pad_idx, msk_idx, eval_batch_size)
            cost_before = mae_in_chunks(current_targets, current_spectra, roi_mask, eval_batch_size)
            current_batches.append((current_stacks, current_targets, current_spectra, cost_before, current_base_stacks))
        elif state_source in ("perturbed_truth", "mixed"):
            current_spectra = simulate_in_chunks(base_stacks, tmm_ctx, eos_idx, pad_idx, msk_idx, eval_batch_size)
            cost_before = mae_in_chunks(targets, current_spectra, roi_mask, eval_batch_size)
            current_batches.append((base_stacks, targets, current_spectra, cost_before, base_stacks))

        if state_source in ("proposal", "mixed"):
            if proposal_model is None:
                raise RuntimeError("Proposal model was not initialized.")
            current_stacks, current_targets, current_spectra, cost_before = sample_proposal_states(
                proposal_model,
                targets,
                mc_samples=proposal_mc_samples,
                keep_candidates=proposal_keep_candidates,
                tmm_ctx=tmm_ctx,
                eos=eos_idx,
                pad=pad_idx,
                msk=msk_idx,
                roi_mask=roi_mask,
                output_seq_len=output_seq_len,
            )
            current_base_stacks = current_stacks
            current_batches.append((current_stacks, current_targets, current_spectra, cost_before, current_base_stacks))

        for current_stacks, current_targets, current_spectra, cost_before, current_base_stacks in current_batches:
            if total_rows >= num_transitions:
                break
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
                **mutation_kwargs,
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

            if include_anchor and state_source in ("perturbed_truth", "mixed"):
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
        "state_source": state_source,
        "shards": shard_idx,
        "output_seq_len": output_seq_len,
        "max_layers": max_layers,
        "current_edits": current_edits,
        "next_edits": next_edits,
        "num_next_perturbations": num_next,
        "proposal_mc_samples": proposal_mc_samples,
        "proposal_keep_candidates": proposal_keep_candidates,
        "proposal_checkpoint": proposal_checkpoint,
        "include_anchor_repair": include_anchor,
    }
    optollama.utils.save_as_json(str(out_dir / "world_transition_summary.json"), summary)
    print(f"Saved summary to {out_dir / 'world_transition_summary.json'}")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
