from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import torch

import optollama.data
import optollama.evaluation.simulation
import optollama.model
import optollama.utils


def parse_args() -> argparse.Namespace:
    """Parse inference, condition, and material-bank overrides."""
    parser = argparse.ArgumentParser(description="Sample the open-vocabulary continuous-time depth-field model.")
    parser.add_argument("--config", default="configs/depth_field_open_vocab_01.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--target", default=None)
    parser.add_argument("--materials-dir", default=None)
    parser.add_argument(
        "--candidate-materials", default=None, help="Comma-separated material names; default is the training catalog."
    )
    parser.add_argument("--angle-deg", type=float, default=None)
    parser.add_argument("--polarization", choices=("s", "p"), default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--mc-samples", type=int, default=10)
    parser.add_argument("--mc-batch-size", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--remask-strategy", choices=("random", "confidence"), default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save", default=None)
    return parser.parse_args()


def nested(mapping: dict[str, Any], *path: str, default: Any = None) -> Any:
    """Read one value from a nested mapping."""
    value: Any = mapping
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def local_fields_to_runs(
    fields: torch.Tensor,
    candidate_names: Sequence[str],
    *,
    void_id: int,
    dz_nm: float,
) -> list[list[dict[str, float | str]]]:
    """Convert candidate-local depth labels to contiguous physical runs."""
    batches: list[list[dict[str, float | str]]] = []
    for row in fields.detach().cpu().tolist():
        runs: list[dict[str, float | str]] = []
        start = 0
        while start < len(row):
            material_id = int(row[start])
            stop = start + 1
            while stop < len(row) and int(row[stop]) == material_id:
                stop += 1
            if material_id != void_id:
                if not 0 <= material_id < len(candidate_names):
                    raise ValueError(f"Sampled invalid local material id {material_id}.")
                runs.append({"material": candidate_names[material_id], "thickness_nm": (stop - start) * dz_nm})
            start = stop
        batches.append(runs)
    return batches


def main() -> None:
    """Sample candidate-local fields and optionally score them with exact TMM."""
    args = parse_args()
    cfg = optollama.utils.load_config_file(args.config)
    block = cfg.get("OPEN_VOCAB_DEPTH_FIELD") or {}
    wavelengths = torch.arange(
        int(cfg["WAVELENGTH_MIN"]), int(cfg["WAVELENGTH_MAX"]) + 1, int(cfg["WAVELENGTH_STEPS"]), dtype=torch.float32
    )
    cfg["WAVELENGTHS"] = wavelengths
    target_path = Path(args.target or cfg["TARGET"])
    target_rat, target_wavelengths = optollama.data.load_open_layer_target(target_path, wavelengths)
    if target_wavelengths.shape != wavelengths.shape or not torch.allclose(target_wavelengths, wavelengths):
        raise ValueError("The first open-vocabulary depth-field package requires the configured fixed wavelength grid.")

    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    extra = blob.get("extra") or {}
    model_metadata = extra.get("open_vocab_depth_field_config")
    if not isinstance(model_metadata, dict):
        raise ValueError("Checkpoint has no open_vocab_depth_field_config metadata.")
    model_config = optollama.model.OpenVocabularyDepthFieldConfig.from_dict(model_metadata)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = optollama.model.OpenVocabularyDepthFieldDiffusion(model_config).to(device).eval()
    optollama.utils.load_checkpoint(args.checkpoint, model, map_location="cpu")
    ema_state = extra.get("ema") if args.weights == "ema" else None
    if isinstance(ema_state, dict) and isinstance(ema_state.get("shadow"), dict):
        current = model.state_dict()
        current.update(
            {
                name: value.to(device=current[name].device, dtype=current[name].dtype)
                for name, value in ema_state["shadow"].items()
                if name in current and value.shape == current[name].shape
            }
        )
        model.load_state_dict(current, strict=True)
        print(f"Loaded EMA weights ({int(ema_state.get('updates', 0)):,} updates).")
    elif args.weights == "ema":
        print("Checkpoint has no EMA state; using raw model weights.")

    tokens, token_to_idx, idx_to_token, _, _, _, eos_idx, pad_idx, msk_idx = optollama.data.init_tokens(cfg["TOKENS_PATH"])
    training_names = tuple(extra.get("material_names") or optollama.data.material_names_from_tokens(tokens))
    candidate_names = (
        tuple(value.strip() for value in args.candidate_materials.split(",") if value.strip())
        if args.candidate_materials
        else training_names
    )
    if not candidate_names:
        raise ValueError("At least one candidate material is required.")
    if len(candidate_names) > model_config.max_candidates:
        raise ValueError(f"Selected {len(candidate_names)} materials but checkpoint supports {model_config.max_candidates}.")
    materials_dir = args.materials_dir or cfg["MATERIALS_PATH"]
    catalog = optollama.data.load_material_catalog(materials_dir, candidate_names)
    curves = catalog.interpolate(
        wavelengths, coverage_tolerance_nm=float(nested(block, "MATERIAL_BANK", "MATERIAL_COVERAGE_TOLERANCE_NM", default=100.0))
    )
    max_candidates = model_config.max_candidates
    candidate_nk = torch.zeros(1, max_candidates, len(wavelengths), 2)
    candidate_nk[:, : len(candidate_names)] = curves
    candidate_mask = torch.zeros(1, max_candidates, dtype=torch.bool)
    candidate_mask[:, : len(candidate_names)] = True

    optical = nested(block, "OPTICAL_CONDITION", default={}) or {}
    angle = float(args.angle_deg if args.angle_deg is not None else optical.get("ANGLE_DEG", 0.0))
    polarization = str(args.polarization or optical.get("POLARIZATION", "s"))
    steps = int(args.steps or nested(block, "EVAL", "SAMPLING_STEPS", default=64))
    remask = str(args.remask_strategy or nested(block, "EVAL", "REMASK_STRATEGY", default="random"))
    corruption = optollama.model.DepthFieldCorruptionConfig.from_dict(nested(block, "DENOISING", "CORRUPTION", default={}))
    mc_batch = max(1, int(args.mc_batch_size or args.mc_samples))
    all_fields: list[torch.Tensor] = []
    all_runs: list[list[dict[str, float | str]]] = []
    dz_nm = float(nested(block, "GRID", "DZ_NM", default=5.0))
    for start in range(0, args.mc_samples, mc_batch):
        count = min(mc_batch, args.mc_samples - start)
        fields = model.sample(
            spectra=target_rat.unsqueeze(0).expand(count, -1, -1).to(device),
            wavelengths_nm=wavelengths.unsqueeze(0).expand(count, -1).to(device),
            candidate_nk=candidate_nk.expand(count, -1, -1, -1).to(device),
            candidate_mask=candidate_mask.expand(count, -1).to(device),
            incidence_angle_deg=torch.full((count,), angle, device=device),
            polarization_id=torch.full((count,), 0 if polarization == "s" else 1, dtype=torch.long, device=device),
            steps=steps,
            temperature=args.temperature,
            top_k=args.top_k,
            deterministic=args.deterministic,
            remask_strategy=remask,
            corruption_config=corruption,
        ).cpu()
        all_fields.append(fields)
        all_runs.extend(local_fields_to_runs(fields, candidate_names, void_id=model.void_id, dz_nm=dz_nm))
    fields = torch.cat(all_fields, dim=0)

    token_materials = optollama.data.material_names_from_tokens(tokens)
    can_simulate = set(candidate_names).issubset(token_materials)
    predicted_rat: torch.Tensor | None = None
    mae: torch.Tensor | None = None
    if can_simulate:
        depth_vocab = optollama.data.build_depth_field_vocab(tokens, token_to_idx)
        material_to_token_id = {name: depth_vocab.token_options[name][0].token_id for name in candidate_names}
        cfg["MATERIALS_PATH"] = materials_dir
        cfg["INCIDENCE_ANGLE"] = angle
        cfg["REALISTIC_TMM"] = {
            "ENABLED": True,
            "ANGLES": [angle],
            "ANGLE_WEIGHTS": [1.0],
            "POLARIZATIONS": [polarization],
            "JITTER_REALIZATIONS": 1,
            "THICKNESS_JITTER_NM": 0.0,
        }
        tmm_ctx = optollama.evaluation.simulation.TMMContext.make(cfg, idx_to_token, device)
        predicted_rat = optollama.evaluation.simulation.simulate_material_runs(
            all_runs,
            tmm_ctx,
            material_to_token_id=material_to_token_id,
            eos=eos_idx,
            pad=pad_idx,
            msk=msk_idx,
        ).cpu()
        mae = (predicted_rat - target_rat.unsqueeze(0)).abs().mean(dim=(1, 2))

    target_dir = Path(block.get("OUT_DIR") or cfg["OUTPUT_PATH"]) / target_path.stem
    save_path = Path(args.save) if args.save else target_dir / f"samples-{datetime.now().strftime('%y%m%d-%H%M')}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for index, runs in enumerate(all_runs):
        record: dict[str, Any] = {"mc_index": index, "field_runs": runs}
        if predicted_rat is not None and mae is not None:
            record["mae"] = float(mae[index])
            record["pred_spectra"] = predicted_rat[index].tolist()
        records.append(record)
    payload = {
        "checkpoint": str(args.checkpoint),
        "target": str(target_path),
        "continuous_time": True,
        "sampling_steps": steps,
        "angle_deg": angle,
        "polarization": polarization,
        "candidate_materials": list(candidate_names),
        "target_spectra": target_rat.tolist(),
        "samples": records,
    }
    save_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} open-vocabulary depth-field samples -> {save_path}")
    if mae is None:
        print("Exact TMM scoring skipped because at least one candidate is not represented by the current token/TMM catalog.")
    else:
        print(f"RAT MAE mean={float(mae.mean()):.6f}, best={float(mae.min()):.6f}")


if __name__ == "__main__":
    main()
