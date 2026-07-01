from __future__ import annotations

import glob
import os
import re

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TargetSpec:
    """Resolved target entry for multi-target inference/plotting."""

    target: str
    name: str


def _as_list(value: Any) -> list[Any]:
    """Normalize optional scalar/list config values."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def safe_target_name(target: str, fallback: str = "target") -> str:
    """Return a filesystem-safe name for a target path or special target key."""
    if target == "random":
        raw = "random"
    else:
        raw = Path(str(target)).stem or fallback

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
    return safe or fallback


def _unique_names(targets: list[str]) -> list[str]:
    """Build unique, stable names for a list of target entries."""
    counts: dict[str, int] = {}
    names: list[str] = []
    for index, target in enumerate(targets):
        base = safe_target_name(target, fallback=f"target_{index}")
        count = counts.get(base, 0)
        counts[base] = count + 1
        names.append(base if count == 0 else f"{base}_{count + 1}")
    return names


def _expand_target_globs(patterns: list[Any]) -> list[str]:
    """Resolve target glob patterns to sorted file paths."""
    targets: list[str] = []
    for pattern in patterns:
        pattern = os.path.expandvars(os.path.expanduser(str(pattern)))
        if os.path.isdir(pattern):
            pattern = os.path.join(pattern, "*.csv")
        matches = [match for match in sorted(glob.glob(pattern)) if os.path.isfile(match)]
        if not matches:
            raise FileNotFoundError(f"TARGET_GLOB did not match any files: {pattern}")
        targets.extend(matches)
    return targets


def make_run_stamp(now: datetime | None = None) -> str:
    """Return the default timestamp key for inference sample artifacts."""
    return (now or datetime.now()).strftime("%y%m%d-%H%M")


def timestamped_sample_path(path: str, stamp: str) -> str:
    """Insert a run timestamp into a sample JSON filename."""
    sample_path = Path(path)
    stem = sample_path.stem
    if re.search(r"(?:^|-)\d{6}-\d{4}(?:-|$)", stem):
        return str(sample_path)
    if stem.endswith("-samples"):
        stem = f"{stem[: -len('-samples')]}-{stamp}-samples"
    else:
        stem = f"{stem}-{stamp}"
    return str(sample_path.with_name(f"{stem}{sample_path.suffix}"))


def cfg_with_timestamped_samples_path(cfg: dict[str, Any], stamp: str | None) -> dict[str, Any]:
    """Return a shallow config copy with timestamped ``SAMPLES_PATH`` when configured."""
    stamped_cfg = dict(cfg)
    if stamp and stamped_cfg.get("SAMPLES_PATH"):
        stamped_cfg["SAMPLES_PATH"] = timestamped_sample_path(str(stamped_cfg["SAMPLES_PATH"]), stamp)
    return stamped_cfg


def has_multi_target_config(cfg: dict[str, Any]) -> bool:
    """Return whether the config explicitly requests multi-target mode."""
    return bool(_as_list(cfg.get("TARGETS")) or _as_list(cfg.get("TARGET_GLOB")))


def resolve_target_specs(cfg: dict[str, Any]) -> tuple[list[TargetSpec], bool]:
    """
    Resolve configured target spectra.

    ``TARGETS`` and ``TARGET_GLOB`` enable multi-target mode and take
    precedence over ``TARGET``. Without multi-target config, this returns the
    single ``TARGET`` entry, or an empty list for validation-dataset mode.
    """
    multi_target = has_multi_target_config(cfg)
    if multi_target:
        targets = _expand_target_globs(_as_list(cfg.get("TARGET_GLOB")))
        targets.extend(str(target) for target in _as_list(cfg.get("TARGETS")))
    else:
        target = cfg.get("TARGET")
        targets = [] if target is None else [str(target)]

    names = _unique_names(targets)
    return [TargetSpec(target=target, name=name) for target, name in zip(targets, names)], multi_target


def cfg_for_target(
    cfg: dict[str, Any],
    spec: TargetSpec,
    multi_target: bool,
    sample_stamp: str | None = None,
) -> dict[str, Any]:
    """
    Return a shallow config copy for one target.

    Target runs write inference artifacts into ``OUTPUT_PATH/<target_name>/``.
    Validation-dataset runs without a target keep the configured root paths.
    """
    target_cfg = dict(cfg)
    target_cfg["TARGET"] = spec.target
    target_cfg["TARGET_NAME"] = spec.name

    output_path = os.path.join(str(cfg["OUTPUT_PATH"]), spec.name)
    target_cfg["OUTPUT_PATH"] = output_path
    target_cfg["SAMPLES_PATH"] = os.path.join(output_path, "samples.json")
    target_cfg["GRID_PATH"] = os.path.join(output_path, "grid.json")
    target_cfg["IDS_PATH"] = os.path.join(output_path, "ids.json")
    target_cfg["PLOT_BUNDLE_PATH"] = os.path.join(output_path, "plot-bundle.npz")
    return cfg_with_timestamped_samples_path(target_cfg, sample_stamp)


def target_cfgs(
    cfg: dict[str, Any],
    sample_stamp: str | None = None,
) -> tuple[list[tuple[TargetSpec, dict[str, Any]]], bool]:
    """Resolve targets and return per-target configs."""
    specs, multi_target = resolve_target_specs(cfg)
    return [(spec, cfg_for_target(cfg, spec, multi_target, sample_stamp)) for spec in specs], multi_target
