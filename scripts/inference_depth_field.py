#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
import tqdm

import optollama.data
import optollama.evaluation.simulation
import optollama.model
import optollama.plotting
import optollama.utils

# ruff: noqa: D103


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample and TMM-score the experimental depth-field diffusion model.")
    parser.add_argument("--config", type=str, default="configs/optollama.yaml", help="Project config YAML.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Depth-field .pt checkpoint.")
    parser.add_argument("--out-json", type=str, default="data/output_depth_field_10um/samples.json", help="Output JSON.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to score.")
    parser.add_argument("--target", type=str, default=None, help="Optional target spectrum CSV/JSON. Defaults to config TARGET when set.")
    parser.add_argument("--target-samples", type=int, default=None, help="Repeated target spectra to evaluate. Defaults to config N_TARGETS.")
    parser.add_argument("--max-samples", type=int, default=256, help="Maximum target spectra to process.")
    parser.add_argument("--batch-size", type=int, default=16, help="Target spectra per model batch.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--sharded-loading", action="store_true", help="Force streaming shard loading.")
    parser.add_argument("--eager-loading", action="store_true", help="Force eager in-memory loading.")
    parser.add_argument("--device", type=str, default=None, help='Model device, e.g. "cuda", "cuda:0", or "cpu".')
    parser.add_argument("--tmm-device", type=str, default="auto", help='TMM device. "auto" uses model device.')

    parser.add_argument("--mc-samples", type=int, default=4, help="Candidate fields per target spectrum.")
    parser.add_argument("--sampling-steps", type=int, default=None, help="Denoising steps. Defaults to checkpoint timesteps.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature. <=0 uses argmax.")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k material sampling filter. 0 disables.")
    parser.add_argument("--deterministic", action="store_true", help="Use argmax instead of sampling.")
    parser.add_argument(
        "--remask-strategy",
        type=str,
        default="confidence",
        choices=["confidence", "random"],
        help="Denoising remask strategy: confidence reopens least-confident bins, random uses Bernoulli remasking.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Sampling seed. Defaults to config SEED.")

    parser.add_argument("--tmm-batch-size", type=int, default=64, help="Decoded stacks per TMM chunk.")
    parser.add_argument(
        "--score-mode",
        type=str,
        default="field",
        choices=["field", "decoded", "both"],
        help="Score native depth-field material runs, decoded token stacks, or both.",
    )
    parser.add_argument(
        "--rank-by",
        type=str,
        default="auto",
        choices=["auto", "field", "decoded"],
        help="Candidate MAE used for MC ranking. Auto uses field for both/field mode and decoded for decoded mode.",
    )
    parser.add_argument("--record-spectra", action="store_true", help="Store target and predicted spectra arrays.")
    parser.add_argument("--record-all-mc", action="store_true", help="Store every MC candidate, not only the best one.")
    parser.add_argument("--plot-bundle", type=str, default=None, help="Optional dashboard .npz path. Defaults to config PLOT_BUNDLE_PATH.")
    parser.add_argument("--no-plot-bundle", action="store_true", help="Do not save the dashboard plot bundle.")
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        device = torch.device(device_arg)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


def resolve_tmm_device(tmm_arg: str, default_device: torch.device) -> torch.device:
    if tmm_arg is None or str(tmm_arg).lower() in {"auto", "same"}:
        return default_device
    device = torch.device(tmm_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA TMM was requested, but CUDA is not available.")
    return device


def split_paths(cfg: dict[str, Any], split: str) -> list[str]:
    prefix = "DATA_PATH_TRAIN" if split == "train" else "DATA_PATH_TEST"
    paths = sorted([str(cfg[key]) for key in cfg.keys() if key == prefix or key.startswith(f"{prefix}_")])
    if not paths:
        raise ValueError(f"No configured paths found for {split!r} split with prefix {prefix}.")
    return paths


def make_eval_loader(cfg: dict[str, Any], args: argparse.Namespace) -> torch.utils.data.DataLoader:
    if args.sharded_loading and args.eager_loading:
        raise ValueError("Use only one of --sharded-loading or --eager-loading.")

    use_shards = bool(cfg.get("SHARDED_LOADING", False))
    if args.sharded_loading:
        use_shards = True
    if args.eager_loading:
        use_shards = False

    paths = split_paths(cfg, args.split)
    if use_shards:
        dataset = optollama.data.ShardedSpectraDataset(
            paths,
            split=args.split,
            subset_n=args.max_samples,
            rank=0,
            world_size=1,
            seed=int(cfg.get("SEED", 0)),
            shuffle=False,
        )
    else:
        dataset = optollama.data.SpectraDataset(paths)
        if args.max_samples is not None and args.max_samples < len(dataset):
            indices = optollama.data.SpectraDataset.indices_of_unique_equidistant_subset(0, len(dataset) - 1, args.max_samples)
            dataset = torch.utils.data.Subset(dataset, indices)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def configured_target(cfg: dict[str, Any], args: argparse.Namespace) -> str | None:
    target = args.target if args.target is not None else cfg.get("TARGET")
    if target is None:
        return None
    target = str(target)
    return target if target.strip() else None


def make_target_loader(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    target: str,
    msk_idx: int,
) -> torch.utils.data.DataLoader:
    if target == "random":
        width = int(cfg["WAVELENGTHS"].numel() if torch.is_tensor(cfg["WAVELENGTHS"]) else len(cfg["WAVELENGTHS"]))
        spectrum = torch.rand((3, width), dtype=torch.float32)
    else:
        spectrum = optollama.utils.load_spectra(target, cfg).to(torch.float32)
    spectrum, _ = optollama.data.ensure_3w(spectrum)
    spectrum = optollama.data.redistribute_mismatch(
        spectrum,
        str(cfg.get("MISMATCH_FILL_ORDER", "R>T>A")),
        target_sum=1.0,
    )

    n_targets = int(args.target_samples if args.target_samples is not None else cfg.get("N_TARGETS", 1))
    n_targets = max(1, n_targets)
    spectra = spectrum.unsqueeze(0).repeat(n_targets, 1, 1).contiguous()
    stacks = torch.full((n_targets, int(cfg["MAX_SEQ_LEN"])), int(msk_idx), dtype=torch.long)
    indices = torch.arange(n_targets, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(spectra, stacks, indices)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=min(n_targets, int(args.batch_size)),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def load_depth_model(checkpoint: str, device: torch.device) -> tuple[optollama.model.DepthFieldDiffusion, dict]:
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    extra = blob.get("extra") or {}
    if "model_config" not in extra:
        raise RuntimeError("Depth-field checkpoint is missing extra['model_config']; use the .pt checkpoint from training.")
    model_config = optollama.model.DepthFieldModelConfig.from_dict(extra["model_config"])
    model = optollama.model.DepthFieldDiffusion(model_config)
    model.load_state_dict(blob["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, extra


def mae_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    wavelengths: torch.Tensor,
    cfg: dict[str, Any],
) -> torch.Tensor:
    wl = wavelengths.to(device=pred.device, dtype=torch.float32)
    mask = torch.ones_like(wl, dtype=torch.bool)
    roi_min = cfg.get("ROI_MIN")
    roi_max = cfg.get("ROI_MAX")
    if roi_min is not None:
        mask &= wl >= float(roi_min)
    if roi_max is not None:
        mask &= wl <= float(roi_max)

    diff = (pred - target).abs()
    channels = cfg.get("MAE_CHANNELS")
    if channels:
        if isinstance(channels, str):
            channels = [channels]
        index = {"R": 0, "A": 1, "T": 2}
        channel_idx = [index[str(channel).upper()] for channel in channels]
        diff = diff[:, channel_idx, :]
    diff = diff[:, :, mask]
    return diff.mean(dim=(1, 2))


def simulate_decoded(
    token_ids: torch.Tensor,
    *,
    tmm_ctx: optollama.evaluation.simulation.TMMContext,
    eos_idx: int,
    pad_idx: int,
    msk_idx: int,
    tmm_batch_size: int,
) -> torch.Tensor:
    outputs = []
    for start in range(0, int(token_ids.size(0)), int(tmm_batch_size)):
        chunk = token_ids[start : start + int(tmm_batch_size)].to(tmm_ctx.wl.device)
        outputs.append(
            optollama.evaluation.simulation.simulate_token_sequence(
                chunk,
                tmm_ctx,
                eos=eos_idx,
                pad=pad_idx,
                msk=msk_idx,
            ).detach()
        )
    return torch.cat(outputs, dim=0)


def simulate_field_runs(
    fields_cpu: torch.Tensor,
    *,
    vocab: optollama.data.DepthFieldVocab,
    tmm_ctx: optollama.evaluation.simulation.TMMContext,
    material_to_token_id: dict[str, int],
    dz_nm: float,
    eos_idx: int,
    pad_idx: int,
    msk_idx: int,
    tmm_batch_size: int,
) -> torch.Tensor:
    outputs = []
    runs_batch = [optollama.data.depth_field_runs(field, vocab, dz_nm=dz_nm) for field in fields_cpu]
    for start in range(0, len(runs_batch), int(tmm_batch_size)):
        outputs.append(
            optollama.evaluation.simulation.simulate_material_runs(
                runs_batch[start : start + int(tmm_batch_size)],
                tmm_ctx,
                material_to_token_id=material_to_token_id,
                eos=eos_idx,
                pad=pad_idx,
                msk=msk_idx,
            ).detach()
        )
    return torch.cat(outputs, dim=0)


def stack_total_thickness_nm(tokens: list[str]) -> float:
    total = 0.0
    for token in tokens:
        parts = optollama.data.layer_token_parts(token)
        if parts is not None:
            total += float(parts[1])
    return total


def candidate_record(
    *,
    flat_idx: int,
    fields_cpu: torch.Tensor,
    token_ids_cpu: torch.Tensor,
    mae_values: torch.Tensor,
    pred_spectra_cpu: torch.Tensor,
    target_spectrum_cpu: torch.Tensor,
    field_mae_values: torch.Tensor | None,
    decoded_mae_values: torch.Tensor | None,
    vocab: optollama.data.DepthFieldVocab,
    idx_to_token: dict[int, str],
    eos_idx: int,
    pad_idx: int,
    dz_nm: float,
    record_spectra: bool,
) -> dict:
    tokens = optollama.data.token_stack_strings(
        token_ids_cpu[flat_idx],
        idx_to_token,
        eos_idx=eos_idx,
        pad_idx=pad_idx,
    )
    active_bins = int(optollama.data.depth_field_active_bins(fields_cpu[flat_idx], vocab.void_id).item())
    field_runs = optollama.data.depth_field_runs(fields_cpu[flat_idx], vocab, dz_nm=dz_nm)
    record = {
        "mae": float(mae_values[flat_idx].item()),
        "tokens": tokens,
        "material_layers": int(len(tokens)),
        "decoded_total_thickness_nm": float(stack_total_thickness_nm(tokens)),
        "field_active_thickness_nm": float(active_bins * dz_nm),
        "field_material_runs": int(len(field_runs)),
        "field_total_thickness_nm": float(sum(float(run["thickness_nm"]) for run in field_runs)),
        "field_runs": field_runs,
    }
    if field_mae_values is not None:
        record["field_mae"] = float(field_mae_values[flat_idx].item())
    if decoded_mae_values is not None:
        record["decoded_mae"] = float(decoded_mae_values[flat_idx].item())
    if record_spectra:
        record["target_spectra"] = target_spectrum_cpu.detach().cpu().tolist()
        record["pred_spectra"] = pred_spectra_cpu[flat_idx].detach().cpu().tolist()
    return record


def main() -> None:
    args = parse_args()
    cfg = optollama.utils.load_config(args)
    seed = int(args.seed if args.seed is not None else cfg.get("SEED", 0))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = resolve_device(args.device)
    tmm_device = resolve_tmm_device(args.tmm_device, device)

    tokens, token_to_idx, idx_to_token, _, _, _, eos_idx, pad_idx, msk_idx = optollama.data.init_tokens(cfg["TOKENS_PATH"])
    target = configured_target(cfg, args)
    loader = make_target_loader(cfg, args, target, msk_idx) if target is not None else make_eval_loader(cfg, args)
    vocab = optollama.data.build_depth_field_vocab(tokens, token_to_idx)
    material_to_token_id = optollama.data.depth_field_material_token_ids(vocab)
    model, extra = load_depth_model(args.checkpoint, device)
    depth_info = extra.get("depth_field") or {}
    dz_nm = float(depth_info.get("dz_nm", 10.0))
    max_thickness_nm = float(depth_info.get("max_thickness_nm", 10_000.0))
    output_seq_len = int(depth_info.get("output_seq_len", cfg["MAX_SEQ_LEN"]))

    if vocab.num_clean_classes != model.num_materials:
        raise RuntimeError(
            f"Token config gives {vocab.num_clean_classes} depth classes, but checkpoint expects {model.num_materials}."
        )

    tmm_ctx = optollama.evaluation.simulation.TMMContext.make(cfg=cfg, idx_to_token=idx_to_token, device=tmm_device)
    results: list[dict] = []
    mae_all: list[float] = []
    selected_field_mae: list[float] = []
    selected_decoded_mae: list[float] = []
    all_mc_mae_batches: list[torch.Tensor] = []
    all_mc_ids_batches: list[torch.Tensor] = []
    all_mc_pred_batches: list[torch.Tensor] = []
    mc_samples = int(args.mc_samples)
    score_mode = str(args.score_mode).lower()
    rank_by = str(args.rank_by).lower()
    if rank_by == "auto":
        rank_by = "field" if score_mode in {"field", "both"} else "decoded"
    if score_mode != "both" and rank_by != score_mode:
        raise ValueError(f"--rank-by={rank_by} requires --score-mode=both or --score-mode={rank_by}.")

    print(
        "Depth-field inference: "
        f"{'target=' + target if target is not None else 'split=' + args.split}, "
        f"mc={mc_samples}, bins={model.depth_bins}, dz={dz_nm:g}nm, "
        f"max={max_thickness_nm:g}nm, score_mode={score_mode}, rank_by={rank_by}, "
        f"remask={args.remask_strategy}, "
        f"model_device={device}, tmm_device={tmm_device}"
    )

    for batch in tqdm.tqdm(loader, desc="depth-field inference", leave=True):
        spectra_cpu, _, indices = batch[0], batch[1], batch[2]
        batch_size = int(spectra_cpu.size(0))
        spectra = spectra_cpu.to(device, non_blocking=True)
        spectra_rep = spectra.repeat_interleave(mc_samples, dim=0)

        fields = model.sample(
            spectra_rep,
            steps=args.sampling_steps,
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            deterministic=bool(args.deterministic or args.temperature <= 0.0),
            remask_strategy=str(args.remask_strategy),
        )
        fields_cpu = fields.detach().cpu()
        token_ids_cpu = optollama.data.decode_depth_field_to_tokens(
            fields_cpu,
            vocab,
            output_seq_len=output_seq_len,
            dz_nm=dz_nm,
            eos_idx=eos_idx,
            pad_idx=pad_idx,
        )

        target_rep = spectra_cpu.to(tmm_device).repeat_interleave(mc_samples, dim=0)
        field_mae = None
        decoded_mae = None
        field_spectra_cpu = None
        decoded_spectra_cpu = None

        if score_mode in {"field", "both"}:
            field_spectra = simulate_field_runs(
                fields_cpu,
                vocab=vocab,
                tmm_ctx=tmm_ctx,
                material_to_token_id=material_to_token_id,
                dz_nm=dz_nm,
                eos_idx=eos_idx,
                pad_idx=pad_idx,
                msk_idx=msk_idx,
                tmm_batch_size=int(args.tmm_batch_size),
            )
            field_mae = mae_per_sample(field_spectra, target_rep, cfg["WAVELENGTHS"], cfg).detach().cpu()
            field_spectra_cpu = field_spectra.detach().cpu()

        if score_mode in {"decoded", "both"}:
            decoded_spectra = simulate_decoded(
                token_ids_cpu,
                tmm_ctx=tmm_ctx,
                eos_idx=eos_idx,
                pad_idx=pad_idx,
                msk_idx=msk_idx,
                tmm_batch_size=int(args.tmm_batch_size),
            )
            decoded_mae = mae_per_sample(decoded_spectra, target_rep, cfg["WAVELENGTHS"], cfg).detach().cpu()
            decoded_spectra_cpu = decoded_spectra.detach().cpu()

        if rank_by == "field":
            if field_mae is None or field_spectra_cpu is None:
                raise RuntimeError("Field MAE requested for ranking but was not computed.")
            mae = field_mae
            pred_spectra_cpu = field_spectra_cpu
        else:
            if decoded_mae is None or decoded_spectra_cpu is None:
                raise RuntimeError("Decoded MAE requested for ranking but was not computed.")
            mae = decoded_mae
            pred_spectra_cpu = decoded_spectra_cpu

        mae_matrix = mae.view(batch_size, mc_samples)
        best_mae, best_idx = mae_matrix.min(dim=1)
        if args.record_all_mc:
            all_mc_mae_batches.append(mae_matrix.detach().cpu().to(torch.float32))
            all_mc_ids_batches.append(token_ids_cpu.reshape(batch_size, mc_samples, -1).detach().cpu().to(torch.long))
            if args.record_spectra:
                all_mc_pred_batches.append(
                    pred_spectra_cpu.reshape(batch_size, mc_samples, *pred_spectra_cpu.shape[1:]).detach().cpu().to(torch.float32)
                )

        for row_idx in range(batch_size):
            flat_idx = row_idx * mc_samples + int(best_idx[row_idx].item())
            record = {
                "dataset_index": int(indices[row_idx].item() if torch.is_tensor(indices[row_idx]) else indices[row_idx]),
                "best_mc": int(best_idx[row_idx].item()),
                "mc_samples": mc_samples,
            }
            record.update(
                candidate_record(
                    flat_idx=flat_idx,
                    fields_cpu=fields_cpu,
                    token_ids_cpu=token_ids_cpu,
                    mae_values=mae,
                    pred_spectra_cpu=pred_spectra_cpu,
                    target_spectrum_cpu=spectra_cpu[row_idx],
                    field_mae_values=field_mae,
                    decoded_mae_values=decoded_mae,
                    vocab=vocab,
                    idx_to_token=idx_to_token,
                    eos_idx=eos_idx,
                    pad_idx=pad_idx,
                    dz_nm=dz_nm,
                    record_spectra=bool(args.record_spectra),
                )
            )
            if args.record_all_mc:
                record["all_mc"] = [
                    candidate_record(
                        flat_idx=row_idx * mc_samples + candidate_idx,
                        fields_cpu=fields_cpu,
                        token_ids_cpu=token_ids_cpu,
                        mae_values=mae,
                        pred_spectra_cpu=pred_spectra_cpu,
                        target_spectrum_cpu=spectra_cpu[row_idx],
                        field_mae_values=field_mae,
                        decoded_mae_values=decoded_mae,
                        vocab=vocab,
                        idx_to_token=idx_to_token,
                        eos_idx=eos_idx,
                        pad_idx=pad_idx,
                        dz_nm=dz_nm,
                        record_spectra=bool(args.record_spectra),
                    )
                    for candidate_idx in range(mc_samples)
                ]
            results.append(record)
            mae_all.append(float(best_mae[row_idx].item()))
            if field_mae is not None:
                selected_field_mae.append(float(field_mae[flat_idx].item()))
            if decoded_mae is not None:
                selected_decoded_mae.append(float(decoded_mae[flat_idx].item()))

    field_mae_tensor = torch.tensor(selected_field_mae) if selected_field_mae else None
    decoded_mae_tensor = torch.tensor(selected_decoded_mae) if selected_decoded_mae else None
    summary = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "split": str(args.split),
        "target": target,
        "samples": int(len(results)),
        "mc_samples": mc_samples,
        "score_mode": score_mode,
        "rank_by": rank_by,
        "remask_strategy": str(args.remask_strategy),
        "mae_mean": float(torch.tensor(mae_all).mean().item()) if mae_all else None,
        "mae_median": float(torch.tensor(mae_all).median().item()) if mae_all else None,
        "mae_min": float(min(mae_all)) if mae_all else None,
        "mae_max": float(max(mae_all)) if mae_all else None,
        "field_mae_mean": float(field_mae_tensor.mean().item()) if field_mae_tensor is not None else None,
        "field_mae_median": float(field_mae_tensor.median().item()) if field_mae_tensor is not None else None,
        "decoded_mae_mean": float(decoded_mae_tensor.mean().item()) if decoded_mae_tensor is not None else None,
        "decoded_mae_median": float(decoded_mae_tensor.median().item()) if decoded_mae_tensor is not None else None,
        "depth_field": {
            "dz_nm": dz_nm,
            "max_thickness_nm": max_thickness_nm,
            "depth_bins": int(model.depth_bins),
            "classes": list(vocab.material_names),
        },
    }
    plot_bundle_path = None
    if not args.no_plot_bundle and args.record_all_mc:
        plot_bundle_path = str(args.plot_bundle or cfg.get("PLOT_BUNDLE_PATH") or "")
        if plot_bundle_path:
            bundle_output: dict[str, torch.Tensor] = {
                "mae_grid": torch.cat(all_mc_mae_batches, dim=0) if all_mc_mae_batches else torch.empty((0, mc_samples)),
                "ids_grid": torch.cat(all_mc_ids_batches, dim=0) if all_mc_ids_batches else torch.empty((0, mc_samples, 0), dtype=torch.long),
            }
            if all_mc_pred_batches:
                bundle_output["pred_spectra_grid"] = torch.cat(all_mc_pred_batches, dim=0)
            optollama.plotting.save_plot_bundle(
                plot_bundle_path,
                bundle_output,
                wavelengths=cfg["WAVELENGTHS"],
                roi_min=cfg.get("ROI_MIN"),
                roi_max=cfg.get("ROI_MAX"),
            )
            summary["plot_bundle"] = plot_bundle_path
            print(f"Saved depth-field plot bundle -> {plot_bundle_path}")
    out = {"summary": summary, "results": results}
    out_path = Path(args.out_json)
    os.makedirs(out_path.parent, exist_ok=True)
    optollama.utils.save_as_json(str(out_path), out)
    print(f"Saved {len(results)} depth-field samples -> {out_path}")
    if mae_all:
        print(f"MAE mean={summary['mae_mean']:.6f}, median={summary['mae_median']:.6f}, best={summary['mae_min']:.6f}")


if __name__ == "__main__":
    main()
