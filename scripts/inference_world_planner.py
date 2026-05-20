#!/usr/bin/env python

import argparse
import os
from pathlib import Path
from typing import Optional

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
        description="Run OptoLlama proposals followed by learned world-model edit planning."
    )
    p.add_argument("--config", type=str, default="configs/world_model.yaml")
    p.add_argument("--target-file", type=str, default=None)
    p.add_argument("--target-source", type=str, default=None, choices=["config", "file", "test"])
    p.add_argument("--max-targets", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--opto-checkpoint", type=str, default=None)
    p.add_argument("--world-checkpoint", type=str, default=None)
    p.add_argument("--out-path", type=str, default=None)
    p.add_argument("--mc-samples", type=int, default=None)
    p.add_argument("--beam-size", type=int, default=None)
    p.add_argument("--planning-rounds", type=int, default=None)
    p.add_argument("--num-perturbations", type=int, default=None)
    p.add_argument("--verify-per-target", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def load_config(path: str) -> dict:
    """
    Load config and enrich it with wavelength values.
    """
    cfg = optollama.utils.load_config_file(path)
    wl_min = int(cfg["WAVELENGTH_MIN"])
    wl_max = int(cfg["WAVELENGTH_MAX"])
    wl_step = int(cfg["WAVELENGTH_STEPS"])
    cfg["WAVELENGTHS"] = torch.arange(wl_min, wl_max + 1, wl_step, dtype=torch.int)
    return cfg


@torch.no_grad()
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


def load_targets(
    args: argparse.Namespace,
    cfg: dict,
    msk_idx: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Load target spectra from a file, config target, or test split.
    """
    source = args.target_source or ("file" if args.target_file else "config" if cfg.get("TARGET") else "test")
    max_targets = int(args.max_targets or cfg.get("WORLD_PLANNER_MAX_TARGETS", cfg.get("N_TARGETS", 1)))

    if source == "file":
        if not args.target_file:
            raise ValueError("--target-source file requires --target-file.")
        target = optollama.utils.load_spectra(args.target_file, cfg).to(torch.float32)
        dataset = optollama.data.RepeatedSpectrumDataset(target, max_targets, cfg, msk_idx)
        return torch.stack([dataset[i][0].to(torch.float32) for i in range(len(dataset))], dim=0).to(device)

    if source == "config":
        target_path = cfg.get("TARGET")
        if not target_path or target_path == "random":
            raise ValueError("--target-source config requires a concrete TARGET path.")
        target = optollama.utils.load_spectra(str(target_path), cfg).to(torch.float32)
        dataset = optollama.data.RepeatedSpectrumDataset(target, max_targets, cfg, msk_idx)
        return torch.stack([dataset[i][0].to(torch.float32) for i in range(len(dataset))], dim=0).to(device)

    test_ds, _, _ = optollama.data.SpectraDataset.make_loader(cfg, split="test", subset_n=max_targets, ddp=False)
    spectra = test_ds.dataset.spectra[test_ds.indices] if isinstance(test_ds, torch.utils.data.Subset) else test_ds.spectra
    return spectra[:max_targets].to(device).to(torch.float32)


def build_opto_model(
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
    Build and load the existing OptoLlama proposal model.
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


def build_world_model(
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
    Build and load the learned world-edit scorer.
    """
    model = optollama.model.WorldEditScorer(
        spectra_shape=tuple(int(v) for v in sample_spectrum.shape),
        vocab_size=len(idx_to_token),
        max_stack_depth=int(cfg.get("WORLD_OUTPUT_SEQ_LEN", cfg["MAX_SEQ_LEN"])),
        eos_idx=eos_idx,
        pad_idx=pad_idx,
        msk_idx=msk_idx,
        d_model=int(cfg.get("WORLD_D_MODEL", 256)),
        n_heads=int(cfg.get("WORLD_N_HEADS", 4)),
        stack_layers=int(cfg.get("WORLD_STACK_LAYERS", 2)),
        dropout=float(cfg.get("WORLD_DROPOUT", 0.0)),
    ).to(device)
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model_state"], strict=True)
    return model.eval()


@torch.no_grad()
def sample_initial_candidates(
    model: torch.nn.Module,
    targets: torch.Tensor,
    mc_samples: int,
    beam_size: int,
    tmm_ctx: optollama.evaluation.simulation.TMMContext,
    eos: int,
    pad: int,
    msk: int,
    roi_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample initial OptoLlama candidates and keep the best per target by TMM MAE.
    """
    b = targets.size(0)
    m = max(1, int(mc_samples))
    keep = min(max(1, int(beam_size)), m)
    targets_mc = targets.unsqueeze(1).expand(b, m, *targets.shape[1:]).reshape(b * m, *targets.shape[1:])
    logits_or_ids, _ = model(targets_mc)
    ids_flat = logits_or_ids.argmax(dim=-1) if logits_or_ids.dim() == 3 else logits_or_ids
    spectra_flat = optollama.evaluation.simulation.simulate_token_sequence(ids_flat, tmm_ctx, eos=eos, pad=pad, msk=msk)
    mae = optollama.evaluation.masked_mae_roi(targets_mc, spectra_flat, wl_mask=roi_mask).view(b, m)
    ids = ids_flat.view(b, m, -1)
    spectra = spectra_flat.view(b, m, *spectra_flat.shape[1:])

    top_mae, top_idx = mae.topk(k=keep, dim=1, largest=False)
    gather_ids = top_idx.unsqueeze(-1).expand(-1, -1, ids.size(-1))
    gather_spec = top_idx.view(b, keep, 1, 1).expand(-1, -1, spectra.size(-2), spectra.size(-1))
    beam_stacks = ids.gather(dim=1, index=gather_ids).reshape(b * keep, ids.size(-1))
    beam_spectra = spectra.gather(dim=1, index=gather_spec).reshape(b * keep, *spectra.shape[2:])
    beam_targets = targets.unsqueeze(1).expand(b, keep, *targets.shape[1:]).reshape(b * keep, *targets.shape[1:])
    target_indices = torch.arange(b, device=targets.device).unsqueeze(1).expand(b, keep).reshape(-1)
    return beam_stacks, beam_targets, beam_spectra, top_mae.reshape(-1), target_indices


@torch.no_grad()
def predict_costs_in_chunks(
    world_model: torch.nn.Module,
    targets: torch.Tensor,
    current_spectra: torch.Tensor,
    current_stacks: torch.Tensor,
    next_stacks: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """
    Predict candidate costs in bounded batches.
    """
    out: list[torch.Tensor] = []
    for start in range(0, next_stacks.size(0), batch_size):
        end = min(start + batch_size, next_stacks.size(0))
        pred = world_model(
            targets[start:end],
            current_spectra[start:end],
            current_stacks[start:end],
            next_stacks[start:end],
        )
        out.append(pred["cost_after"])
    return torch.cat(out, dim=0)


def top_rows_per_target(scores: torch.Tensor, target_indices: torch.Tensor, keep: int) -> torch.Tensor:
    """
    Select the lowest-score rows per target.
    """
    selected: list[torch.Tensor] = []
    for target_idx in torch.unique(target_indices).tolist():
        rows = (target_indices == int(target_idx)).nonzero(as_tuple=False).squeeze(1)
        order = scores[rows].argsort()
        selected.append(rows[order[:keep]])
    return torch.cat(selected, dim=0) if selected else torch.empty((0,), device=scores.device, dtype=torch.long)


def stack_to_tokens(stack: torch.Tensor, idx_to_token: dict[int, str], eos: int, pad: int, msk: int) -> list[str]:
    """
    Convert a token-id stack to visible layer tokens.
    """
    out: list[str] = []
    for token_id in stack.detach().cpu().tolist():
        token_id = int(token_id)
        if token_id == eos:
            break
        if token_id in (pad, msk):
            continue
        out.append(idx_to_token[token_id])
    return out


def main() -> None:
    """
    Run world-model guided planning.
    """
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("SEED", 3))
    optollama.utils.set_all_seeds(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1701)

    device = torch.device(args.device or cfg.get("WORLD_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokens, token_to_idx, idx_to_token, _, _, _, eos_idx, pad_idx, msk_idx = optollama.data.init_tokens(cfg["TOKENS_PATH"])
    targets = load_targets(args, cfg, msk_idx=msk_idx, device=device)
    roi_mask = optollama.data.wavelength_mask(cfg["WAVELENGTHS"], cfg["ROI_MIN"], cfg["ROI_MAX"], device)
    tmm_ctx = optollama.evaluation.simulation.TMMContext.make(cfg, idx_to_token, device=device)

    opto_checkpoint = args.opto_checkpoint or cfg["BEST_CHECKPOINT_PATH"]
    world_checkpoint = args.world_checkpoint or cfg["WORLD_CHECKPOINT_PATH"]
    opto_model = build_opto_model(cfg, targets[0], idx_to_token, eos_idx, pad_idx, msk_idx, device, opto_checkpoint)
    world_model = build_world_model(cfg, targets[0], idx_to_token, eos_idx, pad_idx, msk_idx, device, world_checkpoint)

    pool = optollama.data.build_extension_pool(
        tokens,
        token_to_idx,
        allowed_groups=cfg.get("WORLD_ALLOWED_GROUPS"),
        exclude_materials=cfg.get("WORLD_EXCLUDE_MATERIALS"),
        exclude_tokens=cfg.get("WORLD_EXCLUDE_TOKENS"),
    )
    mutation_index = optollama.data.build_token_mutation_index(tokens, pool)

    mc_samples = int(args.mc_samples or cfg.get("WORLD_PLANNER_MC_SAMPLES", cfg.get("MC_SAMPLES", 10)))
    beam_size = int(args.beam_size or cfg.get("WORLD_PLANNER_BEAM_SIZE", 4))
    planning_rounds = int(args.planning_rounds or cfg.get("WORLD_PLANNER_ROUNDS", 2))
    num_perturbations = int(args.num_perturbations or cfg.get("WORLD_PLANNER_NUM_PERTURBATIONS", 64))
    verify_per_target = int(args.verify_per_target or cfg.get("WORLD_PLANNER_VERIFY_PER_TARGET", 8))
    eval_batch_size = int(cfg.get("WORLD_EVAL_BATCH_SIZE", 512))
    world_batch_size = int(cfg.get("WORLD_PLANNER_BATCH_SIZE", 1024))
    output_seq_len = int(cfg.get("WORLD_OUTPUT_SEQ_LEN", cfg["MAX_SEQ_LEN"]))
    max_layers = int(cfg.get("WORLD_MAX_LAYERS", output_seq_len - 1))

    beam_stacks, beam_targets, beam_spectra, beam_mae, target_indices = sample_initial_candidates(
        opto_model,
        targets,
        mc_samples=mc_samples,
        beam_size=beam_size,
        tmm_ctx=tmm_ctx,
        eos=eos_idx,
        pad=pad_idx,
        msk=msk_idx,
        roi_mask=roi_mask,
    )
    initial_best = beam_mae.clone()
    tmm_calls = int(targets.size(0) * mc_samples)

    for _ in tqdm.tqdm(range(planning_rounds), desc="world planner"):
        variants, source_idx = optollama.data.perturb_stack_candidates(
            beam_stacks,
            index=mutation_index,
            eos=eos_idx,
            pad=pad_idx,
            msk=msk_idx,
            max_layers=max_layers,
            output_seq_len=output_seq_len,
            num_perturbations=num_perturbations,
            generator=generator,
            edits_per_perturbation=int(cfg.get("WORLD_PLANNER_EDITS_PER_PERTURBATION", 1)),
        )
        variant_targets = beam_targets[source_idx]
        variant_current_spectra = beam_spectra[source_idx]
        variant_current_stacks = beam_stacks[source_idx]
        variant_target_indices = target_indices[source_idx]

        predicted_cost = predict_costs_in_chunks(
            world_model,
            variant_targets,
            variant_current_spectra,
            variant_current_stacks,
            variants,
            batch_size=world_batch_size,
        )
        verify_rows = top_rows_per_target(predicted_cost, variant_target_indices, keep=verify_per_target)
        if verify_rows.numel() == 0:
            continue

        verify_stacks = variants[verify_rows]
        verify_targets = variant_targets[verify_rows]
        verify_spectra = simulate_in_chunks(verify_stacks, tmm_ctx, eos_idx, pad_idx, msk_idx, eval_batch_size)
        verify_mae = optollama.evaluation.masked_mae_roi(verify_targets, verify_spectra, wl_mask=roi_mask)
        verify_target_indices = variant_target_indices[verify_rows]
        tmm_calls += int(verify_rows.numel())

        beam_stacks = torch.cat([beam_stacks, verify_stacks], dim=0)
        beam_targets = torch.cat([beam_targets, verify_targets], dim=0)
        beam_spectra = torch.cat([beam_spectra, verify_spectra], dim=0)
        beam_mae = torch.cat([beam_mae, verify_mae], dim=0)
        target_indices = torch.cat([target_indices, verify_target_indices], dim=0)
        keep_rows = top_rows_per_target(beam_mae, target_indices, keep=beam_size)
        beam_stacks = beam_stacks[keep_rows]
        beam_targets = beam_targets[keep_rows]
        beam_spectra = beam_spectra[keep_rows]
        beam_mae = beam_mae[keep_rows]
        target_indices = target_indices[keep_rows]

    best_rows = top_rows_per_target(beam_mae, target_indices, keep=1)
    records = []
    for row in best_rows.detach().cpu().tolist():
        target_idx = int(target_indices[row].item())
        initial_rows = (torch.arange(targets.size(0), device=device).repeat_interleave(min(beam_size, mc_samples)) == target_idx)
        init_mae = float(initial_best[initial_rows].min().item()) if initial_rows.any() else None
        final_mae = float(beam_mae[row].item())
        records.append(
            {
                "target_index": target_idx,
                "initial_best_mae": init_mae,
                "final_mae": final_mae,
                "improvement": None if init_mae is None else init_mae - final_mae,
                "stack_pred_tokens": stack_to_tokens(beam_stacks[row], idx_to_token, eos_idx, pad_idx, msk_idx),
            }
        )

    out = {
        "mean_final_mae": sum(item["final_mae"] for item in records) / max(len(records), 1),
        "mean_improvement": sum(item["improvement"] for item in records if item["improvement"] is not None) / max(len(records), 1),
        "tmm_calls": tmm_calls,
        "records": records,
    }
    out_path = Path(args.out_path or cfg.get("WORLD_PLANNER_OUTPUT_PATH", "data/output/world-planner-results.json"))
    os.makedirs(out_path.parent, exist_ok=True)
    optollama.utils.save_as_json(str(out_path), out)
    print(f"Saved world-planner results -> {out_path}")


if __name__ == "__main__":
    main()
