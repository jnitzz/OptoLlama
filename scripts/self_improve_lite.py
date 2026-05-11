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
import optollama.evaluation
import optollama.model
import optollama.utils


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Target-driven self-improving-lite augmentation: sample model "
            "designs, perturb them, keep only target-error improvements, and "
            "write standard OptoLlama safetensor shards."
        )
    )
    p.add_argument("--config", type=str, default="configs/optollama.yaml", help="Path to the project config YAML.")
    p.add_argument("--out-dir", type=str, required=True, help="Directory where augmented data will be written.")
    p.add_argument(
        "--target-source",
        type=str,
        default="auto",
        choices=["auto", "config", "file", "synthetic", "train", "test"],
        help="Where target spectra come from. 'auto' uses --target-file, then config TARGET, otherwise synthetic.",
    )
    p.add_argument("--target-file", type=str, default=None, help="Optional JSON/CSV target spectrum.")
    p.add_argument("--max-targets", type=int, default=32, help="Maximum number of target spectra to process.")
    p.add_argument("--seed", type=int, default=None, help="Random seed. Defaults to config SEED.")
    p.add_argument("--device", type=str, default=None, help='Execution device, e.g. "cuda", "cuda:0", or "cpu".')
    p.add_argument("--batch-size", type=int, default=8, help="Number of target spectra sampled by the model at once.")
    p.add_argument("--eval-batch-size", type=int, default=512, help="Number of candidate stacks simulated per TMM batch.")
    p.add_argument("--mc-samples", type=int, default=None, help="Model MC samples per target. Defaults to config MC_SAMPLES.")
    p.add_argument("--keep-candidates", type=int, default=4, help="Best model candidates per target to perturb.")
    p.add_argument("--num-perturbations", type=int, default=32, help="Perturbed variants generated per candidate and round.")
    p.add_argument("--mutation-rounds", type=int, default=1, help="Sequential local-search rounds per candidate.")
    p.add_argument("--edits-per-perturbation", type=int, default=1, help="Number of stochastic edits per perturbed variant.")
    p.add_argument("--min-improvement", type=float, default=0.0005, help="Minimum MAE improvement required to keep a result.")
    p.add_argument(
        "--min-layer-gain",
        type=int,
        default=0,
        help="Minimum net material-layer increase required when saving an improved result.",
    )
    p.add_argument(
        "--length-reward",
        type=float,
        default=0.0,
        help="MAE-equivalent reward per added layer during local search and final candidate ranking.",
    )
    p.add_argument(
        "--max-mae-regression",
        type=float,
        default=0.0,
        help="Maximum allowed raw MAE worsening when length reward is enabled. Default keeps raw MAE non-worsening.",
    )
    p.add_argument("--keep-improved-per-target", type=int, default=1, help="Maximum improved stacks saved per target.")
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help=(
            "Override config MAX_SEQ_LEN for model sampling and default output width. "
            "This is token length, including the EOS slot."
        ),
    )
    p.add_argument(
        "--max-emit-len",
        type=int,
        default=None,
        help=(
            "Optional inference-time EOS cap in tokens. Defaults to max_seq_len "
            "when --max-seq-len is provided, otherwise config MAX_EMIT_LEN."
        ),
    )
    p.add_argument("--max-layers", type=int, default=None, help="Maximum material layers after perturbation.")
    p.add_argument("--output-seq-len", type=int, default=None, help="Saved token sequence length. Defaults to max_layers + 1.")
    p.add_argument("--roi-min", type=float, default=None, help="MAE ROI lower wavelength bound. Defaults to config ROI_MIN.")
    p.add_argument("--roi-max", type=float, default=None, help="MAE ROI upper wavelength bound. Defaults to config ROI_MAX.")
    p.add_argument("--allowed-groups", nargs="*", default=None, help="Optional material groups allowed for inserted/mutated tokens.")
    p.add_argument("--exclude-materials", nargs="*", default=None, help="Optional base material names to exclude.")
    p.add_argument("--exclude-tokens", nargs="*", default=None, help="Optional exact tokens to exclude.")
    p.add_argument("--insert-prob", type=float, default=0.25, help="Relative probability for insertion mutations.")
    p.add_argument("--material-prob", type=float, default=0.25, help="Relative probability for material mutations.")
    p.add_argument("--thickness-prob", type=float, default=0.45, help="Relative probability for thickness mutations.")
    p.add_argument("--delete-prob", type=float, default=0.05, help="Relative probability for deletion mutations.")
    p.add_argument(
        "--thickness-deltas",
        nargs="*",
        type=int,
        default=[-50, -40, -30, -20, -10, 10, 20, 30, 40, 50],
        help="Allowed thickness perturbations in nm.",
    )
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


def resolve_split_shards(cfg: dict, split: str) -> list[Path]:
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
        raise FileNotFoundError(f"No .safetensors files found for split {split!r}.")
    return sorted(files, key=lambda item: optollama.data.SpectraDataset.shard_sort_key(str(item.with_suffix(""))))


def load_split_targets(cfg: dict, split: str, max_targets: int) -> torch.Tensor:
    targets: list[torch.Tensor] = []
    remaining = int(max_targets)
    for shard_path in resolve_split_shards(cfg, split):
        if remaining <= 0:
            break
        data = safetensors.torch.load_file(str(shard_path), device="cpu")
        spectra = data["spectra"].to(torch.float32)
        take = min(remaining, spectra.size(0))
        targets.append(spectra[:take])
        remaining -= take

    if not targets:
        raise RuntimeError(f"No target spectra loaded from split {split!r}.")
    return torch.cat(targets, dim=0)


def load_repeated_target_file(path: str, cfg: dict, msk_idx: int, max_targets: int) -> torch.Tensor:
    spectrum = optollama.utils.load_spectra(path, cfg)
    dataset = optollama.data.RepeatedSpectrumDataset(spectrum, n_targets=max_targets, cfg=cfg, msk_idx=msk_idx)
    return torch.stack([dataset[i][0].to(torch.float32) for i in range(len(dataset))], dim=0)


def resolve_target_source(args: argparse.Namespace, cfg: dict) -> str:
    if args.target_source != "auto":
        return args.target_source
    if args.target_file:
        return "file"
    if cfg.get("TARGET") and cfg.get("TARGET") != "random":
        return "config"
    return "synthetic"


def load_targets(
    args: argparse.Namespace,
    cfg: dict,
    msk_idx: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, str]:
    source = resolve_target_source(args, cfg)
    max_targets = max(1, int(args.max_targets))

    if source == "file":
        if not args.target_file:
            raise ValueError("--target-source file requires --target-file.")
        targets = load_repeated_target_file(args.target_file, cfg, msk_idx=msk_idx, max_targets=max_targets)
    elif source == "config":
        target = cfg.get("TARGET")
        if not target or target == "random":
            raise ValueError("--target-source config requires a concrete TARGET path in the config.")
        targets = load_repeated_target_file(str(target), cfg, msk_idx=msk_idx, max_targets=max_targets)
    elif source in ("train", "test"):
        targets = load_split_targets(cfg, source, max_targets=max_targets)
    elif source == "synthetic":
        roi_min = float(args.roi_min if args.roi_min is not None else cfg["ROI_MIN"])
        roi_max = float(args.roi_max if args.roi_max is not None else cfg["ROI_MAX"])
        targets = optollama.data.generate_filter_targets(
            max_targets,
            cfg["WAVELENGTHS"],
            roi_min=roi_min,
            roi_max=roi_max,
            generator=generator,
        )
    else:
        raise ValueError(f"Unsupported target source: {source}")

    if targets.size(-1) != cfg["WAVELENGTHS"].numel():
        raise ValueError(
            f"Target spectra have W={targets.size(-1)} but config defines {cfg['WAVELENGTHS'].numel()} wavelengths."
        )
    return targets.to(torch.float32), source


def build_roi_mask(args: argparse.Namespace, cfg: dict, device: torch.device) -> torch.Tensor:
    roi_min = float(args.roi_min if args.roi_min is not None else cfg["ROI_MIN"])
    roi_max = float(args.roi_max if args.roi_max is not None else cfg["ROI_MAX"])
    return optollama.data.wavelength_mask(cfg["WAVELENGTHS"], roi_min, roi_max, device)


def apply_token_constraints_if_configured(
    model: torch.nn.Module,
    cfg: dict,
    tokens: list[str],
    token_to_idx: dict[str, int],
) -> None:
    if not cfg.get("TOKEN_FILTER_ENABLED", False):
        return

    material_groups = optollama.data.make_material_groups(tokens, token_to_idx)
    mode = cfg["TOKEN_FILTER_MODE"]
    groups = cfg["TOKEN_FILTER_GROUPS"]
    allow_group_ids, exclude_group_ids = [], []

    for group in groups:
        if group not in material_groups:
            raise ValueError(f"Unknown TOKEN_FILTER_GROUPS entry: {group!r}.")
        (allow_group_ids if mode == "allow" else exclude_group_ids).append(material_groups[group])

    allow_ids = torch.cat(allow_group_ids, dim=0) if allow_group_ids else torch.empty((0,), dtype=torch.long)
    exclude_ids = torch.cat(exclude_group_ids, dim=0) if exclude_group_ids else torch.empty((0,), dtype=torch.long)
    material_token_ids = optollama.data.make_material_token_ids(token_to_idx)

    if cfg["TOKEN_FILTER_ALLOW_TOKENS"]:
        allow_ids = torch.unique(
            torch.cat(
                [
                    allow_ids,
                    optollama.data.token_ids_of(cfg["TOKEN_FILTER_ALLOW_TOKENS"], token_to_idx, material_token_ids),
                ],
                dim=0,
            )
        )

    if cfg["TOKEN_FILTER_EXCLUDE_TOKENS"]:
        exclude_ids = torch.unique(
            torch.cat(
                [
                    exclude_ids,
                    optollama.data.token_ids_of(cfg["TOKEN_FILTER_EXCLUDE_TOKENS"], token_to_idx, material_token_ids),
                ],
                dim=0,
            )
        )

    if mode == "allow" and allow_ids.numel() == 0:
        raise ValueError("TOKEN_FILTER_MODE='allow' requires at least one allowed token or group.")

    try:
        model.set_token_constraints(
            allow_ids=allow_ids if allow_ids.numel() else None,
            exclude_ids=exclude_ids if exclude_ids.numel() else None,
        )
    except AttributeError:
        print("Warning: model does not support token constraints; TOKEN_FILTER_* ignored.")


def build_model_from_config(
    cfg: dict,
    sample_spectrum: torch.Tensor,
    idx_to_token: dict[int, str],
    msk_idx: int,
    pad_idx: int,
    eos_idx: int,
    device: torch.device,
) -> torch.nn.Module:
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

    checkpoint = cfg["BEST_CHECKPOINT_PATH"]
    if not checkpoint or not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint}")
    optollama.utils.load_checkpoint(checkpoint, model, map_location="cpu", strict=True)

    try:
        model.set_max_emit_len(cfg["MAX_EMIT_LEN"])
    except AttributeError:
        pass

    return model.eval()


@torch.no_grad()
def simulate_mae_in_chunks(
    stacks: torch.Tensor,
    targets: torch.Tensor,
    tmm_ctx: optollama.evaluation.simulation.TMMContext,
    eos: int,
    pad: int,
    msk: int,
    roi_mask: Optional[torch.Tensor],
    eval_batch_size: int,
) -> torch.Tensor:
    maes: list[torch.Tensor] = []
    for start in range(0, stacks.size(0), eval_batch_size):
        end = min(start + eval_batch_size, stacks.size(0))
        pred = optollama.evaluation.simulation.simulate_token_sequence(
            stacks[start:end],
            tmm_ctx,
            eos=eos,
            pad=pad,
            msk=msk,
        )
        maes.append(optollama.evaluation.masked_mae_roi(targets[start:end], pred, wl_mask=roi_mask))
    return torch.cat(maes, dim=0)


@torch.no_grad()
def sample_model_candidates(
    model: torch.nn.Module,
    targets: torch.Tensor,
    global_target_start: int,
    mc_samples: int,
    keep_candidates: int,
    tmm_ctx: optollama.evaluation.simulation.TMMContext,
    eos: int,
    pad: int,
    msk: int,
    roi_mask: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    b = targets.size(0)
    m = max(1, int(mc_samples))
    keep = min(max(1, int(keep_candidates)), m)

    targets_mc = targets.unsqueeze(1).expand(b, m, *targets.shape[1:]).reshape(b * m, *targets.shape[1:])
    logits_or_ids, _ = model(targets_mc)
    ids_flat = logits_or_ids.argmax(dim=-1) if logits_or_ids.dim() == 3 else logits_or_ids
    pred_flat = optollama.evaluation.simulation.simulate_token_sequence(ids_flat, tmm_ctx, eos=eos, pad=pad, msk=msk)
    mae = optollama.evaluation.masked_mae_roi(targets_mc, pred_flat, wl_mask=roi_mask).view(b, m)
    ids = ids_flat.view(b, m, -1)

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


@torch.no_grad()
def improve_candidates(
    candidates: torch.Tensor,
    targets: torch.Tensor,
    base_mae: torch.Tensor,
    args: argparse.Namespace,
    mutation_index: optollama.data.TokenMutationIndex,
    tmm_ctx: optollama.evaluation.simulation.TMMContext,
    eos: int,
    pad: int,
    msk: int,
    roi_mask: Optional[torch.Tensor],
    perturb_generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    current_stacks, source_lengths = optollama.data.reencode_stacks_for_output(
        candidates,
        output_seq_len=int(args.output_seq_len),
        eos=eos,
        pad=pad,
        msk=msk,
    )
    current_mae = base_mae.clone()
    improved = torch.zeros_like(current_mae, dtype=torch.bool)

    for _ in range(max(1, int(args.mutation_rounds))):
        variants, source_idx = optollama.data.perturb_stack_candidates(
            current_stacks,
            index=mutation_index,
            eos=eos,
            pad=pad,
            msk=msk,
            max_layers=int(args.max_layers),
            output_seq_len=int(args.output_seq_len),
            num_perturbations=int(args.num_perturbations),
            generator=perturb_generator,
            edits_per_perturbation=int(args.edits_per_perturbation),
            insertion_prob=float(args.insert_prob),
            material_prob=float(args.material_prob),
            thickness_prob=float(args.thickness_prob),
            delete_prob=float(args.delete_prob),
            thickness_deltas=args.thickness_deltas,
        )
        variant_targets = targets[source_idx]
        variant_mae = simulate_mae_in_chunks(
            variants,
            variant_targets,
            tmm_ctx=tmm_ctx,
            eos=eos,
            pad=pad,
            msk=msk,
            roi_mask=roi_mask,
            eval_batch_size=int(args.eval_batch_size),
        )

        candidate_count = current_stacks.size(0)
        current_lengths = optollama.data.count_layer_tokens(current_stacks, eos=eos, pad=pad, msk=msk)
        variant_lengths = optollama.data.count_layer_tokens(variants, eos=eos, pad=pad, msk=msk)
        layer_delta = (variant_lengths - current_lengths[source_idx]).to(variant_mae.dtype)
        raw_improvement = current_mae[source_idx] - variant_mae
        adjusted_improvement = raw_improvement + float(args.length_reward) * layer_delta

        score_grid = adjusted_improvement.view(candidate_count, int(args.num_perturbations))
        best_score, best_variant_idx = score_grid.max(dim=1)
        row_offsets = torch.arange(candidate_count, device=current_stacks.device) * int(args.num_perturbations)
        best_variant_rows = row_offsets + best_variant_idx
        best_variant_mae = variant_mae[best_variant_rows]
        best_raw_improvement = current_mae - best_variant_mae
        accept = (
            (best_score >= float(args.min_improvement))
            & (best_raw_improvement >= -float(args.max_mae_regression))
        )

        if accept.any():
            current_stacks[accept] = variants[best_variant_rows[accept]]
            current_mae[accept] = best_variant_mae[accept]
            improved[accept] = True

    final_lengths = optollama.data.count_layer_tokens(current_stacks, eos=eos, pad=pad, msk=msk)
    return current_stacks, current_mae, improved, source_lengths, final_lengths


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg["SEED"])
    optollama.utils.set_all_seeds(seed)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    if int(args.min_layer_gain) < 0:
        raise ValueError("--min-layer-gain must be non-negative.")
    if float(args.length_reward) < 0.0:
        raise ValueError("--length-reward must be non-negative.")
    if float(args.max_mae_regression) < 0.0:
        raise ValueError("--max-mae-regression must be non-negative.")

    if args.max_seq_len is not None:
        if int(args.max_seq_len) <= 0:
            raise ValueError("--max-seq-len must be positive.")
        cfg["MAX_SEQ_LEN"] = int(args.max_seq_len)
        if args.max_emit_len is None:
            cfg["MAX_EMIT_LEN"] = int(args.max_seq_len)

    if args.max_emit_len is not None:
        cfg["MAX_EMIT_LEN"] = int(args.max_emit_len)

    args.max_layers = int(args.max_layers or max(1, cfg["MAX_SEQ_LEN"] - 1))
    args.output_seq_len = int(args.output_seq_len or (args.max_layers + 1))
    if args.output_seq_len < args.max_layers + 1:
        raise ValueError("--output-seq-len must be at least --max-layers + 1.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    copy_tokens_file(cfg, out_dir)

    tokens, token_to_idx, idx_to_token, _, _, _, eos_idx, pad_idx, msk_idx = optollama.data.init_tokens(cfg["TOKENS_PATH"])
    perturb_generator = torch.Generator(device="cpu")
    perturb_generator.manual_seed(seed + 1009)

    targets_cpu, target_source = load_targets(args, cfg, msk_idx=msk_idx, generator=perturb_generator)
    print(f"Loaded {targets_cpu.size(0)} target spectra from {target_source}.")

    pool = optollama.data.build_extension_pool(
        tokens,
        token_to_idx,
        allowed_groups=args.allowed_groups,
        exclude_materials=args.exclude_materials,
        exclude_tokens=args.exclude_tokens,
    )
    mutation_index = optollama.data.build_token_mutation_index(tokens, pool)
    roi_mask = build_roi_mask(args, cfg, device)
    tmm_ctx = optollama.evaluation.simulation.TMMContext.make(cfg, idx_to_token, device=device)

    model = build_model_from_config(
        cfg,
        sample_spectrum=targets_cpu[0],
        idx_to_token=idx_to_token,
        msk_idx=msk_idx,
        pad_idx=pad_idx,
        eos_idx=eos_idx,
        device=device,
    )
    apply_token_constraints_if_configured(model, cfg, tokens, token_to_idx)

    mc_samples = int(args.mc_samples if args.mc_samples is not None else cfg["MC_SAMPLES"])
    out_spectra: list[torch.Tensor] = []
    out_stacks: list[torch.Tensor] = []
    metadata: list[dict[str, float | int]] = []
    total_candidates = 0

    pbar = tqdm.tqdm(range(0, targets_cpu.size(0), int(args.batch_size)), desc="self-improve-lite")
    for start in pbar:
        end = min(start + int(args.batch_size), targets_cpu.size(0))
        targets = targets_cpu[start:end].to(device, non_blocking=True)

        candidates, candidate_targets, target_indices, base_mae = sample_model_candidates(
            model,
            targets,
            global_target_start=start,
            mc_samples=mc_samples,
            keep_candidates=int(args.keep_candidates),
            tmm_ctx=tmm_ctx,
            eos=eos_idx,
            pad=pad_idx,
            msk=msk_idx,
            roi_mask=roi_mask,
        )
        total_candidates += candidates.size(0)

        final_stacks, final_mae, improved, source_lengths, final_lengths = improve_candidates(
            candidates,
            candidate_targets,
            base_mae=base_mae,
            args=args,
            mutation_index=mutation_index,
            tmm_ctx=tmm_ctx,
            eos=eos_idx,
            pad=pad_idx,
            msk=msk_idx,
            roi_mask=roi_mask,
            perturb_generator=perturb_generator,
        )

        selected = optollama.data.select_top_improved_per_target(
            target_indices,
            base_mae,
            final_mae,
            max_per_target=int(args.keep_improved_per_target),
            min_improvement=float(args.min_improvement),
            source_lengths=source_lengths,
            final_lengths=final_lengths,
            min_layer_gain=int(args.min_layer_gain),
            length_reward=float(args.length_reward),
            max_mae_regression=float(args.max_mae_regression),
        )

        if selected.numel():
            out_spectra.append(candidate_targets[selected].detach().cpu())
            out_stacks.append(final_stacks[selected].detach().cpu())
            for row in selected.detach().cpu().tolist():
                metadata.append(
                    {
                        "target_index": int(target_indices[row].item()),
                        "base_mae": float(base_mae[row].item()),
                        "final_mae": float(final_mae[row].item()),
                        "improvement": float((base_mae[row] - final_mae[row]).item()),
                        "source_layers": int(source_lengths[row].item()),
                        "final_layers": int(final_lengths[row].item()),
                        "layer_gain": int((final_lengths[row] - source_lengths[row]).item()),
                        "adjusted_improvement": float(
                            (
                                (base_mae[row] - final_mae[row])
                                + float(args.length_reward) * (final_lengths[row] - source_lengths[row]).float()
                            ).item()
                        ),
                    }
                )

        pbar.set_postfix(improved=len(metadata), candidates=total_candidates)

    summary = {
        "target_source": target_source,
        "targets": int(targets_cpu.size(0)),
        "model_candidates": int(total_candidates),
        "improved_saved": int(len(metadata)),
        "mc_samples": int(mc_samples),
        "keep_candidates": int(args.keep_candidates),
        "num_perturbations": int(args.num_perturbations),
        "mutation_rounds": int(args.mutation_rounds),
        "min_improvement": float(args.min_improvement),
        "min_layer_gain": int(args.min_layer_gain),
        "length_reward": float(args.length_reward),
        "max_mae_regression": float(args.max_mae_regression),
        "max_seq_len": int(cfg["MAX_SEQ_LEN"]),
        "max_emit_len": int(cfg["MAX_EMIT_LEN"]) if cfg.get("MAX_EMIT_LEN") is not None else None,
        "max_layers": int(args.max_layers),
        "output_seq_len": int(args.output_seq_len),
        "examples": metadata,
    }

    if out_spectra:
        spectra_all = torch.cat(out_spectra, dim=0).to(torch.float32)
        stacks_all = torch.cat(out_stacks, dim=0).long()
        out_path = out_dir / "self-improve-lite.safetensors"
        safetensors.torch.save_file({"spectra": spectra_all, "thin_films": stacks_all}, str(out_path))
        improvements = torch.as_tensor([item["improvement"] for item in metadata], dtype=torch.float32)
        final_layers = torch.as_tensor([item["final_layers"] for item in metadata], dtype=torch.float32)
        layer_gains = torch.as_tensor([item["layer_gain"] for item in metadata], dtype=torch.float32)
        summary["mean_improvement"] = float(improvements.mean().item())
        summary["mean_final_layers"] = float(final_layers.mean().item())
        summary["mean_layer_gain"] = float(layer_gains.mean().item())
        print(
            f"Saved {stacks_all.size(0)} improved samples to {out_path} "
            f"(mean improvement: {summary['mean_improvement']:.6f}, "
            f"mean final layers: {summary['mean_final_layers']:.1f}, "
            f"mean layer gain: {summary['mean_layer_gain']:.1f})"
        )
    else:
        print("No improved candidates met --min-improvement; no safetensor shard was written.")

    summary_path = out_dir / "self_improve_lite_summary.json"
    optollama.utils.save_as_json(str(summary_path), summary)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
