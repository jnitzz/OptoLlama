import argparse
import ast
import json
import os
from types import SimpleNamespace
from typing import Any, Tuple

import torch

try:
    import yaml  # type: ignore[import]
except Exception:
    yaml = None


def load_config_file(config_arg: str) -> Any:
    """
    Load config from a JSON/YAML file.

    Supports:
      - *.json
      - *.yaml / *.yml

    Returns a SimpleNamespace with attributes mapped from the file.
    """
    path = os.path.abspath(config_arg)
    lower = path.lower()

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    if lower.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    elif lower.endswith(".yaml") or lower.endswith(".yml"):
        if yaml is None:
            raise ImportError("PyYAML is required to load YAML configs, but it is not installed.")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    else:
        raise ValueError(f"Config must be a .json or .yaml/.yml file (got: {config_arg})")

    if not isinstance(raw, dict):
        raise TypeError(f"Config file {path} must contain a top-level object/dict, got {type(raw).__name__}.")

    return SimpleNamespace(**raw)


def parse_kv(s: str) -> Tuple[str, Any]:
    """Parse the key value pairs provided via --set for config."""
    if "=" not in s:
        raise ValueError(f"--set expects KEY=VALUE, got: {s}")
    key, raw = s.split("=", 1)
    key = key.strip()
    raw = raw.strip()
    try:
        val: Any = ast.literal_eval(raw)
    except Exception:
        val = raw
    return key, val


def set_top_level(cfg: Any, key: str, value: Any) -> None:
    """
    Only top-level keys.

    Supports attribute-style or dict-style configs.
    - If cfg is a dict, assign cfg[key] = value
    - Else setattr(cfg, key, value)
    """
    if "." in key:
        raise ValueError(f"Nested keys are disabled. Use a top-level key (got: {key})")
    if isinstance(cfg, dict):
        cfg[key] = value
    else:
        setattr(cfg, key, value)


def parse_arguments() -> argparse.Namespace:
    """Parse namespace arguments."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--notrain",
        action="store_true",
        help="Sets if training should not be called.",
    )
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help=("Config file (.json or .yaml/.yml), e.g. configs/config_[OL/OG]_[LOCAL/HPC].yaml"),
    )
    p.add_argument(
        "--validsim",
        type=str,
        default=None,
        help="Override validation simulator (TMM_FAST or NOSIM)",
    )
    p.add_argument(
        "--path-ckpt",
        type=str,
        default=None,
        help="Override PATH_CKPT",
    )
    p.add_argument(
        "--mc-samples",
        type=int,
        default=None,
        help="Override MC_SAMPLES for Monte Carlo best-of-N",
    )
    p.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=[],
        help="Top-level override(s), e.g. --set EPOCHS=200 --set TRAIN_BATCH=128",
    )
    p.add_argument(
        "--print-config",
        action="store_true",
        default=True,
        help="Print the final config and exit",
    )
    p.add_argument(
        "--target",
        type=str,
        default=None,
        help="Path to a JSON/CSV RAT file for interactive inference.",
    )
    p.add_argument(
        "--n-targets",
        type=int,
        default=None,
        help="Override N_TARGETS from the config (number of design targets / repeated runs).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override TEMPERATURE (sampling temperature)",
    )
    p.add_argument(
        "--top-k",
        dest="top_k",
        type=int,
        default=None,
        help="Override TOP_K (top-k sampling)",
    )
    p.add_argument(
        "--top-p",
        dest="top_p",
        type=float,
        default=None,
        help="Override TOP_P (top-p / nucleus sampling)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Global default batch size (fallback for train/valid batches).",
    )
    p.add_argument(
        "--train-batch",
        type=int,
        default=None,
        help="Training batch size (overrides BATCH_SIZE for train).",
    )
    p.add_argument(
        "--valid-batch",
        type=int,
        default=None,
        help="Validation batch size (overrides BATCH_SIZE for valid).",
    )
    return p.parse_args()


def load_config_with_overrides(args: argparse.Namespace) -> Any:
    """Load config from JSON/YAML and apply CLI overrides."""
    cfg = load_config_file(args.config)

    # --- convenience flags (CLI wins) ---
    if args.mc_samples is not None:
        set_top_level(cfg, "MC_SAMPLES", args.mc_samples)
    if args.path_ckpt is not None:
        set_top_level(cfg, "PATH_CKPT", args.path_ckpt)
    if args.validsim is not None:
        set_top_level(cfg, "VALIDSIM", args.validsim)
    if args.n_targets is not None:
        set_top_level(cfg, "N_TARGETS", int(args.n_targets))
    if args.target is not None:
        set_top_level(cfg, "TARGET", str(args.target))
    if getattr(args, "notrain", False):
        set_top_level(cfg, "NOTRAIN", True)

    # --- generic top-level --set KEY=VALUE (config-level overrides) ---
    for s in args.sets:
        key, val = parse_kv(s)
        set_top_level(cfg, key, val)

    # apply CLI overrides into cfg
    if args.batch_size is not None:
        set_top_level(cfg, "BATCH_SIZE", args.batch_size)
    if args.train_batch is not None:
        set_top_level(cfg, "TRAIN_BATCH", args.train_batch)
    if args.valid_batch is not None:
        set_top_level(cfg, "VALID_BATCH", args.valid_batch)

    #  derive TRAIN_BATCH / VALID_BATCH with fallbacks
    #  precedence: explicit TRAIN/VALID -> BATCH_SIZE -> hardcoded default
    train_batch = int(getattr(cfg, "TRAIN_BATCH", getattr(cfg, "BATCH_SIZE", 256)))

    valid_batch = int(getattr(cfg, "VALID_BATCH", getattr(cfg, "BATCH_SIZE", 64)))
    set_top_level(cfg, "TRAIN_BATCH", train_batch)
    set_top_level(cfg, "VALID_BATCH", valid_batch)

    # (optional) if you want, also normalize BATCH_SIZE itself:
    if not hasattr(cfg, "BATCH_SIZE"):
        set_top_level(cfg, "BATCH_SIZE", train_batch)

    # --- fill VALIDSIM default if still missing ---
    if not hasattr(cfg, "VALIDSIM"):
        set_top_level(cfg, "VALIDSIM", "TMM_FAST")

    # --- build WAVELENGTHS if needed ---
    if not hasattr(cfg, "WAVELENGTHS"):
        if hasattr(cfg, "WAVELENGTH_MIN") and hasattr(cfg, "WAVELENGTH_MAX") and hasattr(cfg, "WAVELENGTH_STEPS"):
            wl_min = int(cfg.WAVELENGTH_MIN)
            wl_max = int(cfg.WAVELENGTH_MAX)
            wl_step = int(cfg.WAVELENGTH_STEPS)
            wl = torch.arange(wl_min, wl_max + 1, wl_step).to(int)
            setattr(cfg, "WAVELENGTHS", wl)
        else:
            raise ValueError("Config must define either WAVELENGTHS or (WAVELENGTH_MIN, WAVELENGTH_MAX, WAVELENGTH_STEPS)")

    # precedence: CLI flags > config values > hardcoded default
    temperature = args.temperature if args.temperature is not None else float(getattr(cfg, "TEMPERATURE", 0.0))
    top_k = args.top_k if args.top_k is not None else int(getattr(cfg, "TOP_K", 0))
    top_p = args.top_p if args.top_p is not None else float(getattr(cfg, "TOP_P", 0.0))

    # normalize
    temperature = float(temperature or 0.0)
    top_k = int(top_k or 0)
    top_p = float(top_p or 0.0)

    set_top_level(cfg, "TEMPERATURE", temperature)
    set_top_level(cfg, "TOP_K", top_k)
    set_top_level(cfg, "TOP_P", top_p)

    return cfg
