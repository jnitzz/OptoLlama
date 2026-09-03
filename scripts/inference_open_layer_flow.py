from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import torch

import optollama.data
import optollama.evaluation.simulation
import optollama.model
import optollama.plotting
import optollama.utils


def parse_args() -> argparse.Namespace:
    """Parse open-layer inference options."""
    parser = argparse.ArgumentParser(description="Sample open-vocabulary continuous layer stacks.")
    parser.add_argument("--config", default="configs/open_layer_flow_01.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--candidate-materials", default=None, help="Comma-separated material names; default uses the catalog.")
    parser.add_argument("--layer-counts", default=None, help="Comma-separated fixed layer counts to enumerate.")
    parser.add_argument("--mc-samples", type=int, default=None, help="Samples generated for each layer count.")
    parser.add_argument("--mc-batch-size", type=int, default=None)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--tmm-device", default=None)
    parser.add_argument("--save", default=None, help="Optional JSON path; defaults to the target output folder.")
    return parser.parse_args()


def nested(mapping: dict[str, Any], *path: str, default: Any = None) -> Any:
    """Read a nested mapping value."""
    value: Any = mapping
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def parse_int_list(value: str | Sequence[int] | None, default: Sequence[int]) -> list[int]:
    """Parse a comma-separated or configured integer list."""
    if value is None:
        values = list(default)
    elif isinstance(value, str):
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    else:
        values = [int(item) for item in value]
    if not values or any(item <= 0 for item in values):
        raise ValueError("Layer counts must contain positive integers.")
    return list(dict.fromkeys(values))


def resolve_device(value: str | None) -> torch.device:
    """Resolve a usable inference device."""
    if value:
        device = torch.device(value)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def repeat_condition(condition: dict[str, torch.Tensor], repeats: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Repeat a single target/material condition along its batch dimension."""
    return {key: value.repeat((repeats,) + (1,) * (value.ndim - 1)).to(device) for key, value in condition.items()}


def tmm_vocabulary_for_candidates(
    idx_to_token: dict[int, str],
    candidate_names: Sequence[str],
) -> tuple[dict[int, str], dict[str, int]]:
    """Extend the TMM token table with identity-only tokens for unseen materials."""
    extended = dict(idx_to_token)
    existing_material_ids: dict[str, int] = {}
    for token_id, token in extended.items():
        parts = optollama.data.layer_token_parts(token)
        if parts is not None:
            existing_material_ids.setdefault(parts[0], int(token_id))
    next_id = max(extended, default=-1) + 1
    material_to_token_id: dict[str, int] = {}
    for name in candidate_names:
        if name in existing_material_ids:
            material_to_token_id[name] = existing_material_ids[name]
        else:
            material_to_token_id[name] = next_id
            extended[next_id] = f"{name}_1"
            next_id += 1
    return extended, material_to_token_id


def main() -> None:
    """Generate, exactly score, and save open-layer candidates."""
    args = parse_args()
    cfg = optollama.utils.load_config_file(args.config)
    cfg["WAVELENGTHS"] = torch.arange(int(cfg["WAVELENGTH_MIN"]), int(cfg["WAVELENGTH_MAX"]) + 1, int(cfg["WAVELENGTH_STEPS"]))
    block = cfg.get("OPEN_LAYER") or {}
    eval_cfg = nested(block, "EVAL", default={}) or {}
    output_dir = Path(block.get("OUT_DIR") or cfg["OUTPUT_PATH"])
    checkpoint = Path(args.checkpoint or nested(block, "CHECKPOINT") or output_dir / "open-layer-best.pt")
    target_path = Path(args.target or cfg.get("TARGET") or "")
    if not target_path.is_file():
        raise FileNotFoundError(f"Target spectrum does not exist: {target_path}")

    device = resolve_device(args.device)
    tmm_device = resolve_device(args.tmm_device) if args.tmm_device else device
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    extra = blob.get("extra") or {}
    saved_config = extra.get("open_layer_config")
    if not isinstance(saved_config, dict):
        raise ValueError(f"Checkpoint {checkpoint} is missing extra.open_layer_config metadata.")
    model_config = optollama.model.OpenLayerFlowConfig.from_dict(saved_config)
    model = optollama.model.OpenLayerFlow(model_config).to(device).eval()
    optollama.utils.load_checkpoint(str(checkpoint), model, map_location="cpu")

    tokens, _, idx_to_token, _, _, _, eos_idx, pad_idx, msk_idx = optollama.data.init_tokens(cfg["TOKENS_PATH"])
    del tokens
    catalog = optollama.data.load_material_catalog(cfg["MATERIALS_PATH"])
    if args.candidate_materials:
        candidate_names = [value.strip() for value in args.candidate_materials.split(",") if value.strip()]
    else:
        candidate_names = list(catalog.names)
    max_candidates = int(nested(block, "MATERIAL_BANK", "MAX_CANDIDATES", default=15))
    if len(candidate_names) > max_candidates:
        raise ValueError(
            f"Candidate bank contains {len(candidate_names)} materials but MAX_CANDIDATES={max_candidates}. "
            "Pass --candidate-materials or raise the configured limit."
        )

    target_spectrum, target_wavelengths = optollama.data.load_open_layer_target(
        target_path,
        fallback_wavelengths_nm=cfg["WAVELENGTHS"],
    )
    cfg["WAVELENGTHS"] = target_wavelengths
    channels = tuple(str(value).upper() for value in extra.get("channels", ["R", "T"]))
    condition = optollama.data.make_open_layer_condition(
        target_spectrum=target_spectrum,
        wavelengths_nm=target_wavelengths,
        catalog=catalog,
        candidate_names=candidate_names,
        coverage_tolerance_nm=float(nested(block, "QUERY", "MATERIAL_COVERAGE_TOLERANCE_NM", default=0.0)),
        channels=channels,
    )
    layer_counts = parse_int_list(
        args.layer_counts,
        nested(eval_cfg, "LAYER_COUNTS", default=[1, 2, 4, 8, 12, 16, 24, 32]),
    )
    if max(layer_counts) > model_config.max_layers:
        raise ValueError(f"Layer count {max(layer_counts)} exceeds checkpoint max_layers={model_config.max_layers}.")
    mc_samples = int(args.mc_samples or eval_cfg.get("MC_SAMPLES", 10))
    mc_batch_size = int(args.mc_batch_size or eval_cfg.get("MC_BATCH_SIZE", mc_samples))
    sampling_steps = int(args.sampling_steps or eval_cfg.get("SAMPLING_STEPS", 32))
    temperature = float(args.temperature if args.temperature is not None else eval_cfg.get("TEMPERATURE", 1.0))
    if mc_samples <= 0 or mc_batch_size <= 0:
        raise ValueError("MC sample counts must be positive.")
    print(
        f"Open-layer inference: process={model_config.material_process}, "
        f"corruption={model_config.material_corruption_mode}, layers={layer_counts}, "
        f"mc={mc_samples}, steps={sampling_steps}, temperature={temperature:g}"
    )

    tmm_idx_to_token, material_to_token_id = tmm_vocabulary_for_candidates(idx_to_token, candidate_names)
    tmm_ctx: optollama.evaluation.simulation.TMMContext = optollama.evaluation.simulation.TMMContext.make(
        cfg, tmm_idx_to_token, tmm_device
    )
    all_records: list[dict[str, Any]] = []
    target_rt = target_spectrum[[0, 2]].to(tmm_device)

    with torch.no_grad():
        for layer_count in layer_counts:
            generated = 0
            while generated < mc_samples:
                current_batch = min(mc_batch_size, mc_samples - generated)
                model_condition = repeat_condition(condition, current_batch, device)
                sampled = model.sample(
                    wavelengths_nm=model_condition["wavelengths_nm"],
                    target_spectrum=model_condition["target_spectrum"],
                    query_mask=model_condition["query_mask"],
                    candidate_nk=model_condition["candidate_nk"],
                    candidate_mask=model_condition["candidate_mask"],
                    layer_counts=torch.full((current_batch,), layer_count, device=device, dtype=torch.long),
                    steps=sampling_steps,
                    temperature=temperature,
                    deterministic=args.deterministic,
                )
                runs_batch = optollama.data.layer_batch_to_runs(
                    sampled["material_ids"],
                    sampled["thickness_nm"],
                    model_condition["candidate_global_ids"],
                    catalog,
                    sampled["layer_mask"],
                )
                predicted = optollama.evaluation.simulation.simulate_material_runs(
                    runs_batch,
                    tmm_ctx,
                    material_to_token_id=material_to_token_id,
                    eos=eos_idx,
                    pad=pad_idx,
                    msk=msk_idx,
                )
                mae = (predicted[:, (0, 2)] - target_rt.unsqueeze(0)).abs().mean(dim=(1, 2))
                for row, runs in enumerate(runs_batch):
                    all_records.append(
                        {
                            "mc_index": len(all_records),
                            "layer_count_requested": int(layer_count),
                            "layer_count_merged": len(runs),
                            "mae": float(mae[row].item()),
                            "field_mae": float(mae[row].item()),
                            "tokens": [f"{run['material']}_{run['thickness_nm']:.3f}" for run in runs],
                            "field_runs": runs,
                            "decoded_total_thickness_nm": float(sum(run["thickness_nm"] for run in runs)),
                            "pred_spectra": predicted[row].detach().cpu().tolist(),
                            "candidate_materials": candidate_names,
                        }
                    )
                generated += current_batch

    all_records.sort(key=lambda item: float(item["mae"]))
    best = dict(all_records[0])
    best.update(
        {
            "target_spectra": target_spectrum.tolist(),
            "wavelengths_nm": target_wavelengths.tolist(),
            "target_path": str(target_path),
            "checkpoint": str(checkpoint),
            "material_process": model_config.material_process,
            "material_corruption_mode": model_config.material_corruption_mode,
            "all_mc": all_records,
        }
    )
    if args.save:
        save_path = Path(args.save)
    else:
        target_dir = output_dir / target_path.stem
        save_path = target_dir / f"samples-{optollama.utils.make_run_stamp()}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    optollama.utils.save_as_json(str(save_path), [best])
    optollama.plotting.save_plot_bundle(
        str(save_path.parent / "plot-bundle.npz"),
        output={},
        wavelengths=target_wavelengths,
        roi_min=float(target_wavelengths.min().item()),
        roi_max=float(target_wavelengths.max().item()),
    )
    print(
        f"Saved {len(all_records)} open-layer candidates -> {save_path}\n"
        f"Best R/T MAE={best['mae']:.6f}, requested layers={best['layer_count_requested']}, "
        f"merged layers={best['layer_count_merged']}"
    )


if __name__ == "__main__":
    main()
