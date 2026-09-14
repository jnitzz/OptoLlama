from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import torch
import tqdm  # type: ignore[import-untyped]

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
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--target", default=None, help="Single target CSV/JSON. Defaults to TARGET from the config.")
    source.add_argument("--split", choices=("train", "test"), default=None, help="Evaluate a configured dataset split.")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum split samples; defaults to NUM_SAMPLES_TEST.")
    parser.add_argument("--batch-size", type=int, default=1, help="Target spectra per split-evaluation batch.")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
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
    parser.add_argument(
        "--record-spectra",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For split evaluation, save target and best-candidate RAT arrays.",
    )
    parser.add_argument(
        "--record-fields",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For split evaluation, save target and best-candidate material runs.",
    )
    parser.add_argument(
        "--save-every", type=int, default=10, help="Checkpoint split results after this many new targets; 0 disables."
    )
    parser.add_argument("--resume-output", action="store_true", help="Resume split evaluation from an existing --save JSON.")
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


RAT_CHANNELS = ("R", "A", "T")


def candidate_bank_names(
    candidate_global_ids: torch.Tensor,
    candidate_mask: torch.Tensor,
    catalog_names: Sequence[str],
) -> list[tuple[str, ...]]:
    """Resolve each batch row's padded local material bank to names."""
    banks: list[tuple[str, ...]] = []
    for ids, mask in zip(candidate_global_ids.detach().cpu(), candidate_mask.detach().cpu(), strict=True):
        selected: list[str] = []
        for global_id in ids[mask].tolist():
            index = int(global_id)
            if not 0 <= index < len(catalog_names):
                raise ValueError(f"Candidate global material id {index} is outside the catalog.")
            selected.append(str(catalog_names[index]))
        if not selected:
            raise ValueError("A split-evaluation sample has an empty candidate material bank.")
        banks.append(tuple(selected))
    return banks


def apply_fixed_candidate_bank(
    batch: dict[str, torch.Tensor],
    candidate_curves: torch.Tensor,
    *,
    max_candidates: int,
) -> dict[str, torch.Tensor]:
    """Remap collated fields and conditions to one catalog-ordered material bank."""
    candidate_count = int(candidate_curves.shape[0])
    if candidate_count <= 0 or candidate_count > max_candidates:
        raise ValueError(f"Fixed candidate bank has {candidate_count} materials; supported maximum is {max_candidates}.")
    old_fields = batch["clean_fields"].long()
    old_global_ids = batch["candidate_global_ids"].long()
    is_material = old_fields < max_candidates
    old_local_ids = old_fields.clamp(min=0, max=max_candidates - 1)
    remapped = old_global_ids.gather(1, old_local_ids)
    if bool((is_material & (remapped < 0)).any()):
        raise ValueError("A clean field points to padding in its collated candidate bank.")
    remapped_fields = torch.where(is_material, remapped, torch.full_like(remapped, max_candidates))

    batch_size = int(old_fields.shape[0])
    fixed_nk = candidate_curves.new_zeros((max_candidates, *candidate_curves.shape[1:]))
    fixed_nk[:candidate_count] = candidate_curves
    fixed_mask = torch.zeros(max_candidates, dtype=torch.bool)
    fixed_mask[:candidate_count] = True
    fixed_ids = torch.full((max_candidates,), -1, dtype=torch.long)
    fixed_ids[:candidate_count] = torch.arange(candidate_count)
    updated = dict(batch)
    updated["clean_fields"] = remapped_fields
    updated["candidate_nk"] = fixed_nk.unsqueeze(0).expand(batch_size, -1, -1, -1).clone()
    updated["candidate_mask"] = fixed_mask.unsqueeze(0).expand(batch_size, -1).clone()
    updated["candidate_global_ids"] = fixed_ids.unsqueeze(0).expand(batch_size, -1).clone()
    return updated


def _distribution(values: Sequence[float]) -> dict[str, float | None]:
    """Return compact scalar distribution statistics for JSON output."""
    if not values:
        return {"mean": None, "median": None, "p90": None, "min": None, "max": None}
    tensor = torch.tensor(list(values), dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "median": float(torch.quantile(tensor, 0.5)),
        "p90": float(torch.quantile(tensor, 0.9)),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
    }


def summarize_split_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate single-draw, mean-candidate, and oracle best-of-MC RAT errors."""
    summary: dict[str, Any] = {
        "num_samples": len(records),
        "rat_mae": {
            "single_draw": _distribution([float(record["single_draw_mae"]) for record in records]),
            "mean_candidate": _distribution([float(record["mean_candidate_mae"]) for record in records]),
            "best_of_mc": _distribution([float(record["best_mae"]) for record in records]),
        },
        "channel_mae_mean": {},
    }
    for metric in ("single_draw_channel_mae", "mean_candidate_channel_mae", "best_channel_mae"):
        summary["channel_mae_mean"][metric.removesuffix("_channel_mae")] = {
            channel: (
                float(torch.tensor([float(record[metric][channel]) for record in records]).mean()) if records else None
            )
            for channel in RAT_CHANNELS
        }
    return summary


def save_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write restartable evaluation state without exposing a partial JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def split_paths(cfg: dict[str, Any], split: str) -> list[str]:
    """Return configured train/test dataset paths from the flattened config."""
    prefix = "DATA_PATH_TRAIN" if split == "train" else "DATA_PATH_TEST"
    paths = sorted(str(value) for key, value in cfg.items() if key == prefix or key.startswith(f"{prefix}_"))
    if not paths:
        raise KeyError(f"No configured {prefix} paths were found.")
    return paths


def make_split_loader(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    block: dict[str, Any],
    catalog: optollama.data.MaterialCatalog,
    idx_to_token: dict[int, str],
    eos_idx: int,
    pad_idx: int,
    msk_idx: int,
) -> tuple[torch.utils.data.DataLoader, int]:
    """Build a deterministic, single-process loader for held-out RAT evaluation."""
    if args.split is None:
        raise ValueError("A dataset split is required.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.split == "train" and args.max_samples is None:
        raise ValueError("--split train requires --max-samples to avoid unintentionally evaluating the full training set.")
    default_samples = cfg["NUM_SAMPLES_TEST"] if args.split == "test" else cfg["NUM_SAMPLES_TRAIN"]
    subset_n = int(args.max_samples if args.max_samples is not None else default_samples)
    if subset_n <= 0:
        raise ValueError("--max-samples must be positive.")

    bank = nested(block, "MATERIAL_BANK", default={}) or {}
    grid = nested(block, "GRID", default={}) or {}
    optical = nested(block, "OPTICAL_CONDITION", default={}) or {}
    wavelengths = cfg["WAVELENGTHS"]
    collator = optollama.data.OpenVocabularyDepthFieldCollator(
        wavelengths_nm=wavelengths,
        catalog=catalog,
        idx_to_token=idx_to_token,
        eos_idx=eos_idx,
        pad_idx=pad_idx,
        msk_idx=msk_idx,
        channels=RAT_CHANNELS,
        max_layers=int(block.get("MAX_LAYERS", 100)),
        max_candidates=int(bank.get("MAX_CANDIDATES", 24)),
        min_query_points=len(wavelengths),
        max_query_points=len(wavelengths),
        query_sampling="full",
        randomize_candidates=False,
        random_distractors=bool(bank.get("RANDOM_DISTRACTORS", True)),
        holdout_materials=(),
        merge_adjacent=True,
        coverage_tolerance_nm=float(bank.get("MATERIAL_COVERAGE_TOLERANCE_NM", 100.0)),
        seed=int(args.seed if args.seed is not None else cfg.get("SEED", 0)) + 10_000,
        dz_nm=float(grid.get("DZ_NM", 5.0)),
        max_total_nm=float(grid.get("MAX_THICKNESS_NM", 10_000.0)),
        incidence_angle_deg=float(optical.get("ANGLE_DEG", 0.0)),
        polarization=str(optical.get("POLARIZATION", "s")),
    )
    paths = split_paths(cfg, args.split)
    workers = int(args.num_workers if args.num_workers is not None else cfg.get("NUM_WORKERS", 0))
    if bool(cfg.get("SHARDED_LOADING", False)):
        dataset: torch.utils.data.Dataset = optollama.data.ShardedSpectraDataset(
            paths,
            split=args.split,
            subset_n=subset_n,
            rank=0,
            world_size=1,
            seed=int(args.seed if args.seed is not None else cfg.get("SEED", 0)),
            shuffle=False,
        )
        workers = 0
    else:
        eager: torch.utils.data.Dataset = optollama.data.SpectraDataset(paths)
        dataset = torch.utils.data.Subset(eager, range(min(subset_n, len(eager))))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collator,
    )
    return loader, len(dataset)


def _repeat_rows(value: torch.Tensor, count: int, device: torch.device) -> torch.Tensor:
    """Repeat complete batch rows in target-major order and move them to a device."""
    return value.repeat_interleave(count, dim=0).to(device, non_blocking=True)


def _channel_mapping(values: torch.Tensor) -> dict[str, float]:
    """Convert one three-channel error tensor to a named JSON mapping."""
    return {channel: float(values[index]) for index, channel in enumerate(RAT_CHANNELS)}


@torch.inference_mode()
def evaluate_split(
    *,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    block: dict[str, Any],
    model: optollama.model.OpenVocabularyDepthFieldDiffusion,
    model_config: optollama.model.OpenVocabularyDepthFieldConfig,
    catalog: optollama.data.MaterialCatalog,
    candidate_names: Sequence[str],
    tokens: Sequence[str],
    wavelengths: torch.Tensor,
    idx_to_token: dict[int, str],
    token_to_idx: dict[str, int],
    eos_idx: int,
    pad_idx: int,
    msk_idx: int,
    device: torch.device,
    corruption: optollama.model.DepthFieldCorruptionConfig,
    angle: float,
    polarization: str,
    steps: int,
    remask: str,
    dz_nm: float,
) -> Path:
    """Generate candidate fields for a split and score their exact-TMM RAT spectra."""
    if args.split is None:
        raise ValueError("A split is required for split evaluation.")
    if args.mc_samples <= 0 or (args.mc_batch_size is not None and args.mc_batch_size <= 0):
        raise ValueError("--mc-samples and --mc-batch-size must be positive.")
    if args.save_every < 0:
        raise ValueError("--save-every cannot be negative.")
    if args.resume_output and not args.save:
        raise ValueError("--resume-output requires an explicit --save path.")
    evaluation_seed = int(args.seed if args.seed is not None else cfg.get("SEED", 0))
    mc_batch = max(1, int(args.mc_batch_size or args.mc_samples))

    loader, dataset_size = make_split_loader(args, cfg, block, catalog, idx_to_token, eos_idx, pad_idx, msk_idx)
    token_materials = set(optollama.data.material_names_from_tokens(tokens))
    unsupported = sorted(set(candidate_names) - token_materials)
    if unsupported:
        raise ValueError(f"Exact TMM split scoring cannot map these candidates to tokens: {unsupported}")
    depth_vocab = optollama.data.build_depth_field_vocab(tokens, token_to_idx)
    material_to_token_id = {name: depth_vocab.token_options[name][0].token_id for name in candidate_names}
    candidate_curves = catalog.interpolate(
        wavelengths,
        coverage_tolerance_nm=float(
            nested(block, "MATERIAL_BANK", "MATERIAL_COVERAGE_TOLERANCE_NM", default=100.0)
        ),
    )
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

    output_dir = Path(block.get("OUT_DIR") or cfg["OUTPUT_PATH"])
    save_path = (
        Path(args.save)
        if args.save
        else output_dir / f"rat-{args.split}-{datetime.now().strftime('%y%m%d-%H%M')}.json"
    )
    records: list[dict[str, Any]] = []
    invalid_sample_indices: list[int] = []
    if args.resume_output and save_path.is_file():
        previous = json.loads(save_path.read_text(encoding="utf-8"))
        if previous.get("mode") != "split" or previous.get("split") != args.split:
            raise ValueError(f"Cannot resume {save_path}: it is not a matching {args.split!r} split result.")
        if str(previous.get("checkpoint")) != str(args.checkpoint):
            raise ValueError(f"Cannot resume {save_path}: checkpoint path differs from --checkpoint.")
        expected_settings = {
            "weights": args.weights,
            "sampling_steps": steps,
            "mc_samples": int(args.mc_samples),
            "mc_batch_size": mc_batch,
            "batch_size": int(args.batch_size),
            "temperature": float(args.temperature),
            "top_k": int(args.top_k),
            "deterministic": bool(args.deterministic),
            "remask_strategy": remask,
            "angle_deg": angle,
            "polarization": polarization,
            "candidate_materials": list(candidate_names),
            "record_spectra": bool(args.record_spectra),
            "record_fields": bool(args.record_fields),
            "seed": evaluation_seed,
            "dataset_size": dataset_size,
        }
        changed = [key for key, value in expected_settings.items() if previous.get(key) != value]
        if changed:
            raise ValueError(f"Cannot resume {save_path}: evaluation settings changed: {changed}")
        records = list(previous.get("samples") or [])
        invalid_sample_indices = [int(value) for value in previous.get("invalid_sample_indices") or []]
        print(f"Resuming {save_path} with {len(records):,} completed samples.")
    completed = {int(record["sample_index"]) for record in records}
    known_invalid = set(invalid_sample_indices)

    def payload() -> dict[str, Any]:
        ordered = sorted(records, key=lambda record: int(record["sample_index"]))
        return {
            "mode": "split",
            "split": args.split,
            "checkpoint": str(args.checkpoint),
            "weights": args.weights,
            "continuous_time": True,
            "sampling_steps": steps,
            "mc_samples": int(args.mc_samples),
            "mc_batch_size": mc_batch,
            "batch_size": int(args.batch_size),
            "temperature": float(args.temperature),
            "top_k": int(args.top_k),
            "deterministic": bool(args.deterministic),
            "remask_strategy": remask,
            "seed": evaluation_seed,
            "angle_deg": angle,
            "polarization": polarization,
            "rat_channel_order": list(RAT_CHANNELS),
            "wavelengths_nm": wavelengths.tolist(),
            "candidate_materials": list(candidate_names),
            "candidate_bank_mode": "fixed catalog order",
            "record_spectra": bool(args.record_spectra),
            "record_fields": bool(args.record_fields),
            "best_of_mc_selection": "oracle exact-TMM minimum RAT MAE",
            "dataset_size": dataset_size,
            "invalid_sample_indices": sorted(known_invalid),
            "summary": summarize_split_records(ordered),
            "samples": ordered,
        }

    saved_count = len(records)
    progress = tqdm.tqdm(loader, total=math.ceil(dataset_size / int(args.batch_size)), desc=f"RAT {args.split}")
    for raw in progress:
        raw_indices = raw["sample_indices"].tolist()
        for sample_index, valid in zip(raw_indices, raw["sample_mask"].tolist(), strict=True):
            if not bool(valid):
                known_invalid.add(int(sample_index))
        keep = torch.tensor(
            [
                bool(valid) and int(sample_index) not in completed
                for sample_index, valid in zip(raw_indices, raw["sample_mask"].tolist(), strict=True)
            ],
            dtype=torch.bool,
        )
        if not bool(keep.any()):
            continue
        batch = apply_fixed_candidate_bank(
            {key: value[keep] for key, value in raw.items()},
            candidate_curves,
            max_candidates=model_config.max_candidates,
        )
        batch_size = int(batch["sample_indices"].shape[0])
        batch_seed = evaluation_seed + int(batch["sample_indices"][0]) * 1_000_003
        torch.manual_seed(batch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(batch_seed)
        banks = [tuple(candidate_names)] * batch_size
        target_rat = batch["target_spectrum_rat"].transpose(1, 2).contiguous().cpu()
        target_runs = (
            [
                local_fields_to_runs(
                    batch["clean_fields"][row : row + 1],
                    banks[row],
                    void_id=model_config.max_candidates,
                    dz_nm=dz_nm,
                )[0]
                for row in range(batch_size)
            ]
            if args.record_fields
            else []
        )
        mae_chunks: list[torch.Tensor] = []
        channel_mae_chunks: list[torch.Tensor] = []
        predicted_chunks: list[torch.Tensor] = []
        runs_by_target: list[list[list[dict[str, float | str]]]] = [[] for _ in range(batch_size)]

        for mc_start in range(0, args.mc_samples, mc_batch):
            count = min(mc_batch, args.mc_samples - mc_start)
            fields = model.sample(
                spectra=_repeat_rows(batch["target_spectrum"].transpose(1, 2), count, device),
                wavelengths_nm=_repeat_rows(batch["wavelengths_nm"], count, device),
                candidate_nk=_repeat_rows(batch["candidate_nk"], count, device),
                candidate_mask=_repeat_rows(batch["candidate_mask"], count, device),
                incidence_angle_deg=_repeat_rows(batch["incidence_angle_deg"], count, device),
                polarization_id=_repeat_rows(batch["polarization_id"], count, device),
                steps=steps,
                temperature=args.temperature,
                top_k=args.top_k,
                deterministic=args.deterministic,
                remask_strategy=remask,
                corruption_config=corruption,
            ).cpu().reshape(batch_size, count, -1)
            chunk_runs: list[list[dict[str, float | str]]] = []
            for row in range(batch_size):
                decoded = local_fields_to_runs(
                    fields[row],
                    banks[row],
                    void_id=model_config.max_candidates,
                    dz_nm=dz_nm,
                )
                runs_by_target[row].extend(decoded)
                chunk_runs.extend(decoded)
            predicted = optollama.evaluation.simulation.simulate_material_runs(
                chunk_runs,
                tmm_ctx,
                material_to_token_id=material_to_token_id,
                eos=eos_idx,
                pad=pad_idx,
                msk=msk_idx,
            ).cpu().reshape(batch_size, count, 3, -1)
            absolute_error = (predicted - target_rat.unsqueeze(1)).abs()
            mae_chunks.append(absolute_error.mean(dim=(2, 3)))
            channel_mae_chunks.append(absolute_error.mean(dim=3))
            if args.record_spectra:
                predicted_chunks.append(predicted)

        mae_grid = torch.cat(mae_chunks, dim=1)
        channel_mae_grid = torch.cat(channel_mae_chunks, dim=1)
        predicted_grid = torch.cat(predicted_chunks, dim=1) if predicted_chunks else None
        for row in range(batch_size):
            sample_index = int(batch["sample_indices"][row])
            best_index = int(mae_grid[row].argmin())
            record: dict[str, Any] = {
                "sample_index": sample_index,
                "single_draw_mae": float(mae_grid[row, 0]),
                "mean_candidate_mae": float(mae_grid[row].mean()),
                "best_mae": float(mae_grid[row, best_index]),
                "best_mc_index": best_index,
                "candidate_mae": mae_grid[row].tolist(),
                "single_draw_channel_mae": _channel_mapping(channel_mae_grid[row, 0]),
                "mean_candidate_channel_mae": _channel_mapping(channel_mae_grid[row].mean(dim=0)),
                "best_channel_mae": _channel_mapping(channel_mae_grid[row, best_index]),
                "candidate_channel_mae": [
                    _channel_mapping(channel_mae_grid[row, mc_index]) for mc_index in range(args.mc_samples)
                ],
            }
            if args.record_fields:
                record["target_field_runs"] = target_runs[row]
                record["best_field_runs"] = runs_by_target[row][best_index]
            if predicted_grid is not None:
                record["target_spectra"] = target_rat[row].tolist()
                record["best_pred_spectra"] = predicted_grid[row, best_index].tolist()
            records.append(record)
            completed.add(sample_index)
        progress.set_postfix(best=f"{summarize_split_records(records)['rat_mae']['best_of_mc']['mean']:.4f}")
        if args.save_every > 0 and len(records) - saved_count >= args.save_every:
            save_json_atomic(save_path, payload())
            saved_count = len(records)

    final_payload = payload()
    save_json_atomic(save_path, final_payload)
    if not records:
        raise RuntimeError(f"The {args.split!r} split contained no valid samples to score.")
    summary = final_payload["summary"]["rat_mae"]
    print(f"Saved {len(records):,} exact-TMM split results -> {save_path}")
    print(
        f"RAT MAE: single-draw mean={summary['single_draw']['mean']:.6f}, "
        f"candidate mean={summary['mean_candidate']['mean']:.6f}, "
        f"best-of-{args.mc_samples} mean={summary['best_of_mc']['mean']:.6f}, "
        f"median={summary['best_of_mc']['median']:.6f}, p90={summary['best_of_mc']['p90']:.6f}"
    )
    best_channels = final_payload["summary"]["channel_mae_mean"]["best"]
    print(
        "Best-of-MC channel MAE: "
        + ", ".join(f"{channel}={best_channels[channel]:.6f}" for channel in RAT_CHANNELS)
    )
    return save_path


def main() -> None:
    """Sample candidate-local fields and optionally score them with exact TMM."""
    args = parse_args()
    cfg = optollama.utils.load_config_file(args.config)
    block = cfg.get("OPEN_VOCAB_DEPTH_FIELD") or {}
    wavelengths = torch.arange(
        int(cfg["WAVELENGTH_MIN"]), int(cfg["WAVELENGTH_MAX"]) + 1, int(cfg["WAVELENGTH_STEPS"]), dtype=torch.float32
    )
    cfg["WAVELENGTHS"] = wavelengths
    seed = int(args.seed if args.seed is not None else cfg.get("SEED", 0))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
    cfg["MATERIALS_PATH"] = materials_dir
    catalog = optollama.data.load_material_catalog(materials_dir, candidate_names)

    optical = nested(block, "OPTICAL_CONDITION", default={}) or {}
    angle = float(args.angle_deg if args.angle_deg is not None else optical.get("ANGLE_DEG", 0.0))
    polarization = str(args.polarization or optical.get("POLARIZATION", "s"))
    steps = int(args.steps or nested(block, "EVAL", "SAMPLING_STEPS", default=64))
    remask = str(args.remask_strategy or nested(block, "EVAL", "REMASK_STRATEGY", default="random"))
    corruption = optollama.model.DepthFieldCorruptionConfig.from_dict(nested(block, "DENOISING", "CORRUPTION", default={}))
    dz_nm = float(nested(block, "GRID", "DZ_NM", default=5.0))
    if args.split is not None:
        evaluate_split(
            args=args,
            cfg=cfg,
            block=block,
            model=model,
            model_config=model_config,
            catalog=catalog,
            candidate_names=candidate_names,
            tokens=tokens,
            wavelengths=wavelengths,
            idx_to_token=idx_to_token,
            token_to_idx=token_to_idx,
            eos_idx=eos_idx,
            pad_idx=pad_idx,
            msk_idx=msk_idx,
            device=device,
            corruption=corruption,
            angle=angle,
            polarization=polarization,
            steps=steps,
            remask=remask,
            dz_nm=dz_nm,
        )
        return

    target_path = Path(args.target or cfg["TARGET"])
    target_rat, target_wavelengths = optollama.data.load_open_layer_target(target_path, wavelengths)
    if target_wavelengths.shape != wavelengths.shape or not torch.allclose(target_wavelengths, wavelengths):
        raise ValueError("The first open-vocabulary depth-field package requires the configured fixed wavelength grid.")
    curves = catalog.interpolate(
        wavelengths, coverage_tolerance_nm=float(nested(block, "MATERIAL_BANK", "MATERIAL_COVERAGE_TOLERANCE_NM", default=100.0))
    )
    max_candidates = model_config.max_candidates
    candidate_nk = torch.zeros(1, max_candidates, len(wavelengths), 2)
    candidate_nk[:, : len(candidate_names)] = curves
    candidate_mask = torch.zeros(1, max_candidates, dtype=torch.bool)
    candidate_mask[:, : len(candidate_names)] = True
    mc_batch = max(1, int(args.mc_batch_size or args.mc_samples))
    all_fields: list[torch.Tensor] = []
    all_runs: list[list[dict[str, float | str]]] = []
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
