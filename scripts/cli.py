import argparse
import ast
import importlib
import importlib.util
import os
import sys
from typing import Any, Tuple


def import_config_module(config_arg: str) -> Any:
    """
    Return the imported module (treated as Any for flexibility).

    Import a config either from a module name (e.g. 'config_MD60')
    or from a Python file path (e.g. 'path/to/config_MD60.py').
    """
    if config_arg.endswith(".py") or os.path.sep in config_arg:
        path = os.path.abspath(config_arg)
        spec = importlib.util.spec_from_file_location("user_cfg", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load config from path: {config_arg}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["user_cfg"] = module
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        return module
    return importlib.import_module(config_arg)


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
    p.add_argument("--notrain", action="store_true", help="Sets if training should not be called.")
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config module or file (e.g., config_MD60 or path/to/config_MD60.py)",
    )
    p.add_argument(
        "--validsim",
        type=str,
        default="TMM_FAST",
        help="Override validation simulator (TMM_FAST or NOSIM)",
    )
    p.add_argument("--ckpt", type=str, default=None, help="Override PATH_CHKPT")
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
    return p.parse_args()


def load_config_with_overrides(args: argparse.Namespace) -> Any:
    """
    Return the cfg object (usually a module, but typed as Any).

    Load the config module/file specified in args.config and apply
    CLI overrides (mc_samples, ckpt, validsim, and --set KEY=VALUE).
    """
    cfg = import_config_module(args.config)

    # Convenience flags
    if args.mc_samples is not None:
        set_top_level(cfg, "MC_SAMPLES", int(args.mc_samples))
    if args.ckpt is not None:
        set_top_level(cfg, "PATH_CHKPT", args.ckpt)
    if args.validsim is not None:
        set_top_level(cfg, "VALIDSIM", args.validsim)

    # Generic top-level --set KEY=VALUE
    for s in args.sets:
        key, val = parse_kv(s)
        set_top_level(cfg, key, val)

    return cfg
