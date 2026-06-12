import argparse
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy

import omegaconf
import torch


def parse_arguments() -> argparse.Namespace:
    """
    Parse namespace arguments.

    Returns
    -------
        The CLI arguments as a Namespace.
    """
    p = argparse.ArgumentParser()

    p.add_argument(
        "--config",
        type=str,
        default="configs/optollama.yaml",
        help=("Path to YAML config file (.yaml/.yml), e.g. configs/config_optollama.yaml"),
    )

    p.add_argument(
        "--results",
        type=str,
        default=None,
        help=("Path to the results file (JSON format)"),
    )

    return p.parse_args()


def _get_nested(cfg: Mapping, path: Sequence[str], default=None):
    """Return a nested config value, or ``default`` if any path segment is absent."""
    cur = cfg
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _set_if_present(flat: dict, legacy_key: str, cfg: Mapping, path: Sequence[str]) -> None:
    """Set a legacy flat key from a nested path when the path exists."""
    missing = object()
    value = _get_nested(cfg, path, missing)
    if value is not missing:
        flat[legacy_key] = value


def _set_path_list(flat: dict, prefix: str, values) -> None:
    """Set legacy DATA_PATH_* keys from a nested list/string path entry."""
    if values is None:
        return
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, Sequence):
        raise TypeError(f"{prefix} paths must be a string or sequence of strings, got {type(values).__name__}")

    for key in list(flat.keys()):
        if key == prefix or key.startswith(f"{prefix}_"):
            flat.pop(key)

    for idx, value in enumerate(values):
        if idx == 0 and prefix == "DATA_PATH_TEST":
            flat[prefix] = value
        else:
            flat[f"{prefix}_{idx}"] = value


def _flatten_realistic_dataset_aliases(flat: dict) -> None:
    """Add legacy REALISTIC_DATASET aliases for the nested readable schema."""
    block = flat.get("REALISTIC_DATASET")
    if not isinstance(block, dict):
        return

    layers = block.get("LAYERS") or {}
    if isinstance(layers, Mapping):
        aliases = {
            "MIN_LAYERS": ("MIN",),
            "MAX_LAYERS": ("MAX",),
            "OUTPUT_SEQ_LEN": ("OUTPUT_SEQ_LEN",),
        }
        for legacy_key, path in aliases.items():
            value = _get_nested(layers, path)
            if value is not None:
                block[legacy_key] = value

        family_min = layers.get("FAMILY_MIN_LAYERS") or {}
        if isinstance(family_min, Mapping):
            for family, value in family_min.items():
                block[f"{str(family).upper()}_MIN_LAYERS"] = value

    thickness = block.get("THICKNESS") or {}
    if isinstance(thickness, Mapping):
        _copy = {"THICKNESS_MIN": "MIN", "THICKNESS_MAX": "MAX", "THICKNESS_STEP": "STEP"}
        for legacy_key, nested_key in _copy.items():
            if nested_key in thickness:
                block[legacy_key] = thickness[nested_key]

    spectrum = block.get("SPECTRUM") or {}
    if isinstance(spectrum, Mapping):
        _copy = {
            "WAVELENGTH_MIN": "WAVELENGTH_MIN",
            "WAVELENGTH_MAX": "WAVELENGTH_MAX",
            "WAVELENGTH_STEP": "WAVELENGTH_STEP",
        }
        for legacy_key, nested_key in _copy.items():
            if nested_key in spectrum:
                block[legacy_key] = spectrum[nested_key]

    averaging = block.get("REALISTIC_AVERAGING") or {}
    if isinstance(averaging, Mapping):
        for key in ("ANGLES", "ANGLE_WEIGHTS", "POLARIZATIONS", "JITTER_REALIZATIONS", "THICKNESS_JITTER_NM", "SAVE_DTYPE"):
            if key in averaging:
                block[key] = averaging[key]

    structures = block.get("STRUCTURES") or {}
    if isinstance(structures, Mapping):
        for key in ("CENTER_MIN", "CENTER_MAX", "STRUCTURE_JITTER_FRACTION"):
            if key in structures:
                block[key] = structures[key]


def flatten_config_sections(cfg: dict) -> dict:
    """
    Add legacy flat keys for the nested config schema.

    Existing code still reads keys like ``cfg["D_MODEL"]`` and
    ``cfg["DATA_PATH_TRAIN_0"]``. This compatibility layer lets configs group
    those values under readable sections without refactoring all call sites at
    once. Already-flat configs pass through unchanged.
    """
    flat = deepcopy(cfg)

    mappings = {
        "MODEL": ("EXPERIMENT", "MODEL"),
        "RUN": ("EXPERIMENT", "RUN"),
        "SEED": ("EXPERIMENT", "SEED"),
        "DATA_PATH": ("PATHS", "DATA_PATH"),
        "MATERIALS_PATH": ("PATHS", "MATERIALS_PATH"),
        "TOKENS_PATH": ("PATHS", "TOKENS_PATH"),
        "OUTPUT_PATH": ("PATHS", "OUTPUT_PATH"),
        "SAMPLES_PATH": ("PATHS", "SAMPLES_PATH"),
        "GRID_PATH": ("PATHS", "GRID_PATH"),
        "IDS_PATH": ("PATHS", "IDS_PATH"),
        "PLOT_BUNDLE_PATH": ("PATHS", "PLOT_BUNDLE_PATH"),
        "BEST_CHECKPOINT_PATH": ("PATHS", "BEST_CHECKPOINT_PATH"),
        "LAST_CHECKPOINT_PATH": ("PATHS", "LAST_CHECKPOINT_PATH"),
        "INIT_CHECKPOINT_PATH": ("CHECKPOINT", "INIT_PATH"),
        "INIT_CHECKPOINT_STRICT": ("CHECKPOINT", "INIT_STRICT"),
        "INIT_CHECKPOINT_FALLBACK_FILTER": ("CHECKPOINT", "INIT_FALLBACK_FILTER"),
        "RESUME_CHECKPOINT": ("CHECKPOINT", "RESUME"),
        "NUM_WORKERS": ("DATA_LOADING", "NUM_WORKERS"),
        "SHARDED_LOADING": ("DATA_LOADING", "SHARDED_LOADING"),
        "NUM_SAMPLES_TRAIN": ("DATA_LOADING", "NUM_SAMPLES_TRAIN"),
        "NUM_SAMPLES_TEST": ("DATA_LOADING", "NUM_SAMPLES_TEST"),
        "TRAIN_BATCH_SIZE": ("DATA_LOADING", "TRAIN_BATCH_SIZE"),
        "TEST_BATCH_SIZE": ("DATA_LOADING", "TEST_BATCH_SIZE"),
        "WAVELENGTH_MIN": ("SPECTRAL_GRID", "WAVELENGTH_MIN"),
        "WAVELENGTH_MAX": ("SPECTRAL_GRID", "WAVELENGTH_MAX"),
        "WAVELENGTH_STEPS": ("SPECTRAL_GRID", "WAVELENGTH_STEPS"),
        "MAX_SEQ_LEN": ("SEQUENCE", "MAX_SEQ_LEN"),
        "MAX_EMIT_LEN": ("SEQUENCE", "MAX_EMIT_LEN"),
        "D_MODEL": ("MODEL_ARCH", "D_MODEL"),
        "N_BLOCKS": ("MODEL_ARCH", "N_BLOCKS"),
        "N_HEADS": ("MODEL_ARCH", "N_HEADS"),
        "DROPOUT": ("MODEL_ARCH", "DROPOUT"),
        "DIFFUSION_STEPS": ("MODEL_ARCH", "DIFFUSION_STEPS"),
        "DEPTH_POSITION": ("MODEL_ARCH", "DEPTH_POSITION"),
        "DEPTH_ROPE": ("MODEL_ARCH", "DEPTH_ROPE"),
        "FACTORED_OUTPUT": ("MODEL_ARCH", "FACTORED_OUTPUT"),
        "LEARNING_RATE": ("OPTIMIZATION", "LEARNING_RATE"),
        "EPOCHS": ("OPTIMIZATION", "EPOCHS"),
        "TEMPERATURE": ("SAMPLING", "TEMPERATURE"),
        "TOP_K": ("SAMPLING", "TOP_K"),
        "TOP_P": ("SAMPLING", "TOP_P"),
        "MC_SAMPLES": ("SAMPLING", "MC_SAMPLES"),
        "TRACK_DIFFUSION_STEPS_MAE": ("INFERENCE", "TRACK_DIFFUSION_STEPS_MAE"),
        "INFERENCE_RECORD_ALL_MC": ("INFERENCE", "RECORD_ALL_MC"),
        "INFERENCE_RECORD_PRED_SPECTRA": ("INFERENCE", "RECORD_PRED_SPECTRA"),
        "INFERENCE_SHOW_PROGRESS": ("INFERENCE", "SHOW_PROGRESS"),
        "INFERENCE_PROFILE_TIMING": ("INFERENCE", "PROFILE_TIMING"),
        "INFERENCE_DEDUP_STACKS": ("INFERENCE", "DEDUP_STACKS"),
        "VALIDATE_EVERY_N_TRAIN_SAMPLES": ("VALIDATION", "EVERY_N_TRAIN_SAMPLES"),
        "VALIDATE_AT_EPOCH_END": ("VALIDATION", "AT_EPOCH_END"),
        "VALID_SIM": ("EVALUATION", "VALID_SIM"),
        "TMM_DEVICE": ("EVALUATION", "TMM_DEVICE"),
        "INCIDENCE_ANGLE": ("EVALUATION", "INCIDENCE_ANGLE"),
        "ROI_MIN": ("EVALUATION", "ROI_MIN"),
        "ROI_MAX": ("EVALUATION", "ROI_MAX"),
        "MAE_CHANNELS": ("EVALUATION", "MAE_CHANNELS"),
        "COMMON_MAE_ENABLED": ("EVALUATION", "COMMON_MAE", "ENABLED"),
        "COMMON_MAE_WAVELENGTH_MIN": ("EVALUATION", "COMMON_MAE", "WAVELENGTH_MIN"),
        "COMMON_MAE_WAVELENGTH_MAX": ("EVALUATION", "COMMON_MAE", "WAVELENGTH_MAX"),
        "COMMON_MAE_WAVELENGTH_STEPS": ("EVALUATION", "COMMON_MAE", "WAVELENGTH_STEPS"),
        "TARGET": ("TARGET_SELECTION", "TARGET"),
        "TARGETS": ("TARGET_SELECTION", "TARGETS"),
        "TARGET_GLOB": ("TARGET_SELECTION", "TARGET_GLOB"),
        "N_TARGETS": ("TARGET_SELECTION", "N_TARGETS"),
        "MISMATCH_FILL_ORDER": ("TARGET_PREPROCESS", "MISMATCH_FILL_ORDER"),
        "NOISE": ("TARGET_PREPROCESS", "NOISE"),
        "SMOOTH": ("TARGET_PREPROCESS", "SMOOTH"),
        "FILL_OUTSIDE_ROI": ("TARGET_PREPROCESS", "FILL_OUTSIDE_ROI"),
        "PLOT_SAMPLE_WITH_NN": ("PLOTTING", "SAMPLE_WITH_NN"),
        "PLOT_NN_CHUNK_SIZE": ("PLOTTING", "NN_CHUNK_SIZE"),
        "PLOT_NN_DEVICE": ("PLOTTING", "NN_DEVICE"),
        "PLOT_NK": ("PLOTTING", "NK"),
        "TOKEN_FILTER_ENABLED": ("TOKEN_FILTER", "ENABLED"),
        "TOKEN_FILTER_MODE": ("TOKEN_FILTER", "MODE"),
        "TOKEN_FILTER_GROUPS": ("TOKEN_FILTER", "GROUPS"),
        "TOKEN_FILTER_EXCLUDE_TOKENS": ("TOKEN_FILTER", "EXCLUDE_TOKENS"),
        "TOKEN_FILTER_ALLOW_TOKENS": ("TOKEN_FILTER", "ALLOW_TOKENS"),
    }

    for legacy_key, path in mappings.items():
        _set_if_present(flat, legacy_key, cfg, path)

    _set_path_list(flat, "DATA_PATH_TRAIN", _get_nested(cfg, ("PATHS", "TRAIN")))
    _set_path_list(flat, "DATA_PATH_TEST", _get_nested(cfg, ("PATHS", "TEST")))
    _flatten_realistic_dataset_aliases(flat)

    return flat


def load_config_file(path: str) -> dict:
    """
    Load config from a YAML file.

    Args
    ----
        path: Path to the config file.

    Returns
    -------
        The configuration as a dictionary.
    """
    cfg = omegaconf.OmegaConf.load(path)

    return flatten_config_sections(omegaconf.OmegaConf.to_container(cfg, resolve=True))


def load_config(args: argparse.Namespace) -> dict:
    """
    Load config from YAML and enrich wavelengths.

    Args
    ----
        path: Path to the config file.

    Returns
    -------
        The configuration as a dictionary.
    """
    cfg = load_config_file(args.config)

    # --- build WAVELENGTHS if needed ---
    wl_min = int(cfg["WAVELENGTH_MIN"])
    wl_max = int(cfg["WAVELENGTH_MAX"])
    wl_step = int(cfg["WAVELENGTH_STEPS"])

    cfg["WAVELENGTHS"] = torch.arange(wl_min, wl_max + 1, wl_step, dtype=torch.int)

    return cfg
