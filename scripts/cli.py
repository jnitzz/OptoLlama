# import argparse
# #TODO default im config und cli überschreiben
# def parse_arguments() -> argparse.Namespace:
#     p = argparse.ArgumentParser()
    
#     p.add_argument('--notrain', action='store_true')
#     p.add_argument("--config", type=str, required=True, help="Config module (e.g., config_MD58 or config_MD58.py)")
#     p.add_argument("--validsim", type=str, default="TMM_FAST", help="TMM_FAST or NOSIM")
#     p.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint (overrides cfg.PATH_CHKPT)")
#     p.add_argument("--mc-samples", type=int, default=None, help="Override cfg.MC_SAMPLES for Monte Carlo best-of-N")
#     return p.parse_args()

import argparse
import importlib
import importlib.util
import os
import sys
import ast
from typing import Any, Tuple

def _import_config_module(config_arg: str):
    if config_arg.endswith(".py") or os.path.sep in config_arg:
        path = os.path.abspath(config_arg)
        spec = importlib.util.spec_from_file_location("user_cfg", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load config from path: {config_arg}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["cfg"] = module
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        return module
    else:
        return importlib.import_module(config_arg)

def _parse_kv(s: str) -> Tuple[str, Any]:
    if "=" not in s:
        raise ValueError(f"--set expects KEY=VALUE, got: {s}")
    key, raw = s.split("=", 1)
    key = key.strip()
    raw = raw.strip()
    try:
        val = ast.literal_eval(raw)
    except Exception:
        val = raw
    return key, val

def _set_top_level(cfg: Any, key: str, value: Any):
    """
    Only top-level keys. Supports attribute-style or dict-style configs.
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
    p = argparse.ArgumentParser()
    p.add_argument('--notrain', action='store_true',
                   help="Sets if training should not be called.")
    p.add_argument("--config", type=str, required=True,
                   help="Config module or file (e.g., config_MD60 or path/to/config_MD60.py)")
    p.add_argument("--validsim", type=str, default="TMM_FAST",
                   help="Override validation simulator (TMM_FAST or NOSIM)")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Override PATH_CHKPT")
    p.add_argument("--mc-samples", type=int, default=None,
                   help="Override MC_SAMPLES for Monte Carlo best-of-N")
    p.add_argument("--set", dest="sets", action="append", default=[],
                   help="Top-level override(s), e.g. --set EPOCHS=200 --set TRAIN_BATCH=128")
    p.add_argument("--print-config", action="store_true",
                   help="Print the final config and exit")
    return p.parse_args()

def load_config_with_overrides(args: argparse.Namespace):
    cfg = _import_config_module(args.config)

    # Convenience flags
    if args.mc_samples is not None:
        _set_top_level(cfg, "MC_SAMPLES", int(args.mc_samples))
    if args.ckpt is not None:
        _set_top_level(cfg, "PATH_CHKPT", args.ckpt)
    if args.validsim is not None:
        _set_top_level(cfg, "VALIDSIM", args.validsim)

    # Generic top-level --set KEY=VALUE
    for s in args.sets:
        key, val = _parse_kv(s)
        _set_top_level(cfg, key, val)

    return cfg