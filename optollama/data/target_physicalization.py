from __future__ import annotations

import hashlib
import json
import os
import re

from pathlib import Path
from typing import Any

import torch
from safetensors.torch import safe_open

from optollama.data.spectra import ensure_3w, redistribute_mismatch, smooth_1d, wavelength_mask


CHANNEL_TO_INDEX = {"R": 0, "A": 1, "T": 2}


def _enabled(cfg: dict[str, Any]) -> bool:
    return bool((cfg.get("TARGET_PHYSICALIZE") or {}).get("ENABLED", False))


def _dataset_paths_from_cfg(cfg: dict[str, Any], block: dict[str, Any]) -> list[str]:
    paths = block.get("SOURCE_PATHS")
    if paths:
        return [str(path) for path in paths]

    split = str(block.get("SOURCE_SPLIT", "train")).lower()
    prefix = "DATA_PATH_TEST" if split == "test" else "DATA_PATH_TRAIN"
    return sorted(str(value) for key, value in cfg.items() if str(key).startswith(prefix))


def _collect_safetensors(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            files.extend(sorted(path.glob("*.safetensors")))
        elif path.suffix.lower() == ".safetensors":
            files.append(path)
        else:
            raise ValueError(f"Unsupported NN source path: {path}. Expected directory or .safetensors file.")

    if not files:
        raise FileNotFoundError("No .safetensors files found for TARGET_PHYSICALIZE.NN_BLEND.")

    return sorted(files, key=_shard_sort_key)


def _file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _tensor_fingerprint(tensor: torch.Tensor) -> dict[str, Any]:
    cpu = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(cpu.shape)).encode("utf-8"))
    digest.update(cpu.numpy().tobytes())
    return {
        "shape": list(cpu.shape),
        "dtype": "float32",
        "sha256": digest.hexdigest(),
    }


def _cache_key(
    target: torch.Tensor,
    wavelengths: torch.Tensor,
    block: dict[str, Any],
    files: list[Path],
) -> str:
    metric_roi_min = float(block.get("METRIC_ROI_MIN", float(wavelengths.min())))
    metric_roi_max = float(block.get("METRIC_ROI_MAX", float(wavelengths.max())))
    payload = {
        "version": 1,
        "target": _tensor_fingerprint(target),
        "wavelengths": _tensor_fingerprint(wavelengths),
        "metric_roi_min": metric_roi_min,
        "metric_roi_max": metric_roi_max,
        "metric_channels": [str(channel).upper() for channel in block.get("METRIC_CHANNELS", ["R", "T"])],
        "source_files": [_file_fingerprint(path) for path in files],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _nn_cache_path(cfg: dict[str, Any], block: dict[str, Any]) -> Path:
    configured = block.get("CACHE_PATH")
    if configured:
        return Path(os.path.expandvars(os.path.expanduser(str(configured))))
    return Path(str(cfg.get("OUTPUT_PATH", "."))) / "target_physicalize_nn_cache.json"


def _load_nn_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "entries": {}}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError:
        return {"version": 1, "entries": {}}
    if not isinstance(data, dict):
        return {"version": 1, "entries": {}}
    data.setdefault("version", 1)
    if not isinstance(data.get("entries"), dict):
        data["entries"] = {}
    return data


def _save_nn_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _load_cached_match(entry: dict[str, Any], target_width: int) -> dict[str, Any] | None:
    file_path = Path(str(entry.get("file", "")))
    local_index = int(entry.get("local_index", -1))
    if local_index < 0 or not file_path.exists():
        return None

    with safe_open(str(file_path), framework="pt", device="cpu") as handle:
        if "spectra" not in handle.keys():
            return None
        spectra = handle.get_tensor("spectra")

    if spectra.dim() != 3 or spectra.size(1) != 3 or spectra.size(-1) != target_width:
        return None
    if local_index >= spectra.size(0):
        return None

    return {
        "spectrum": spectra[local_index].to(torch.float32).clone(),
        "mae": float(entry["mae"]),
        "global_index": int(entry["global_index"]),
        "file": str(file_path),
        "local_index": local_index,
    }


def _shard_sort_key(path: Path) -> tuple[str, int | float]:
    match = re.match(r"^(.*?)(\d+)$", path.stem.lower())
    if match:
        prefix, number = match.groups()
        return prefix, int(number)
    return path.stem.lower(), float("inf")


def _channel_indices(channels: list[str] | None) -> torch.Tensor:
    if not channels:
        return torch.arange(3, dtype=torch.long)

    indices = []
    for channel in channels:
        name = str(channel).upper()
        if name not in CHANNEL_TO_INDEX:
            raise ValueError(f"Unknown TARGET_PHYSICALIZE.NN_BLEND.METRIC_CHANNELS entry: {channel!r}")
        indices.append(CHANNEL_TO_INDEX[name])

    return torch.tensor(indices, dtype=torch.long)


def _nm_to_points(wavelengths: torch.Tensor, value_nm: float, minimum: int = 1) -> int:
    if wavelengths.numel() < 2:
        return minimum

    diffs = wavelengths[1:] - wavelengths[:-1]
    step_nm = float(torch.median(diffs.abs()).item())
    if step_nm <= 0:
        return minimum
    return max(minimum, int(round(float(value_nm) / step_nm)))


def _normalize_rat(spectrum: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
    return redistribute_mismatch(
        spectrum.clamp(0.0, 1.0),
        str(cfg.get("MISMATCH_FILL_ORDER", "R>T>A")),
        target_sum=1.0,
    )


def _apply_level_relaxation(spectrum: torch.Tensor, block: dict[str, Any], cfg: dict[str, Any]) -> torch.Tensor:
    level_cfg = block.get("LEVEL_RELAXATION") or {}
    if not bool(level_cfg.get("ENABLED", False)):
        return spectrum

    pass_value = float(level_cfg.get("PASS_VALUE", 0.95))
    stop_value = float(level_cfg.get("STOP_VALUE", 0.05))
    absorption_scale = float(level_cfg.get("ABS_SCALE", 1.0))

    relaxed = spectrum.clone()
    relaxed[[0, 2], :] = stop_value + (pass_value - stop_value) * relaxed[[0, 2], :]
    relaxed[1, :] = relaxed[1, :] * absorption_scale
    return _normalize_rat(relaxed, cfg)


def _apply_edge_smoothing(spectrum: torch.Tensor, wavelengths: torch.Tensor, block: dict[str, Any], cfg: dict[str, Any]) -> torch.Tensor:
    smooth_cfg = block.get("EDGE_SMOOTHING") or {}
    if not bool(smooth_cfg.get("ENABLED", False)):
        return spectrum

    sigma_nm = float(smooth_cfg.get("SIGMA_NM", 20.0))
    win_nm = float(smooth_cfg.get("WIN_NM", max(4.0 * sigma_nm, 1.0)))
    sigma_points = max(float(_nm_to_points(wavelengths, sigma_nm, minimum=1)), 1.0)
    win_points = _nm_to_points(wavelengths, win_nm, minimum=3)
    if win_points % 2 == 0:
        win_points += 1

    smoothed = smooth_1d(spectrum, method="gaussian", win=win_points, sigma=sigma_points)
    return _normalize_rat(smoothed, cfg)


def _blend_weights(wavelengths: torch.Tensor, block: dict[str, Any]) -> torch.Tensor:
    roi_min = float(block.get("ROI_MIN"))
    roi_max = float(block.get("ROI_MAX"))
    roi_blend = float(block.get("ROI_BLEND", 0.0))
    outside_blend = float(block.get("OUTSIDE_ROI_BLEND", 1.0))
    transition_blend = float(block.get("TRANSITION_BLEND", (roi_blend + outside_blend) * 0.5))
    transition_width_nm = float(block.get("TRANSITION_WIDTH_NM", 0.0))

    if transition_width_nm <= 0:
        inside = (wavelengths >= roi_min) & (wavelengths <= roi_max)
        weights = torch.full_like(wavelengths, outside_blend, dtype=torch.float32)
        weights[inside] = roi_blend
        return weights.clamp(0.0, 1.0)

    def smoothstep(x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    weights = torch.full_like(wavelengths, outside_blend, dtype=torch.float32)

    lower_outer = roi_min - transition_width_nm
    lower_inner = roi_min + transition_width_nm
    upper_inner = roi_max - transition_width_nm
    upper_outer = roi_max + transition_width_nm

    core = (wavelengths >= lower_inner) & (wavelengths <= upper_inner)
    weights[core] = roi_blend

    lower_outer_band = (wavelengths >= lower_outer) & (wavelengths < roi_min)
    if torch.any(lower_outer_band):
        x = (wavelengths[lower_outer_band] - lower_outer) / max(transition_width_nm, 1e-8)
        weights[lower_outer_band] = outside_blend + (transition_blend - outside_blend) * smoothstep(x)

    lower_inner_band = (wavelengths >= roi_min) & (wavelengths < lower_inner)
    if torch.any(lower_inner_band):
        x = (wavelengths[lower_inner_band] - roi_min) / max(transition_width_nm, 1e-8)
        weights[lower_inner_band] = transition_blend + (roi_blend - transition_blend) * smoothstep(x)

    upper_inner_band = (wavelengths > upper_inner) & (wavelengths <= roi_max)
    if torch.any(upper_inner_band):
        x = (wavelengths[upper_inner_band] - upper_inner) / max(transition_width_nm, 1e-8)
        weights[upper_inner_band] = roi_blend + (transition_blend - roi_blend) * smoothstep(x)

    upper_outer_band = (wavelengths > roi_max) & (wavelengths <= upper_outer)
    if torch.any(upper_outer_band):
        x = (wavelengths[upper_outer_band] - roi_max) / max(transition_width_nm, 1e-8)
        weights[upper_outer_band] = transition_blend + (outside_blend - transition_blend) * smoothstep(x)

    return weights.clamp(0.0, 1.0)


@torch.no_grad()
def find_nearest_training_spectrum(
    target: torch.Tensor,
    cfg: dict[str, Any],
    block: dict[str, Any],
    wavelengths: torch.Tensor,
    device: torch.device | str | None = None,
    files: list[Path] | None = None,
) -> dict[str, Any]:
    """
    Stream dataset shards and return the closest training/test spectrum.
    """
    if device is None:
        device = block.get("DEVICE")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    if files is None:
        paths = _dataset_paths_from_cfg(cfg, block)
        files = _collect_safetensors(paths)
    chunk_size = int(block.get("CHUNK_SIZE", 512000))
    channels = _channel_indices(block.get("METRIC_CHANNELS", ["R", "T"])).to(device_t)

    metric_roi_min = float(block.get("METRIC_ROI_MIN", cfg.get("ROI_MIN", float(wavelengths.min()))))
    metric_roi_max = float(block.get("METRIC_ROI_MAX", cfg.get("ROI_MAX", float(wavelengths.max()))))
    wl_mask = wavelength_mask(wavelengths, metric_roi_min, metric_roi_max, device_t)

    target_d = target.to(device_t, dtype=torch.float32)
    if wl_mask is not None:
        target_d = target_d.index_select(0, channels)[:, wl_mask]
    else:
        target_d = target_d.index_select(0, channels)

    best_mae = float("inf")
    best_global_index = -1
    best_local_index = -1
    best_file = ""
    best_spectrum: torch.Tensor | None = None
    global_offset = 0

    for shard_path in files:
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            if "spectra" not in handle.keys():
                raise KeyError(f"{shard_path} must contain a 'spectra' tensor.")
            spectra = handle.get_tensor("spectra")

        if spectra.dim() != 3 or spectra.size(1) != 3:
            raise ValueError(f"'spectra' in {shard_path} must be [N,3,W], got {tuple(spectra.shape)}")
        if spectra.size(-1) != target.size(-1):
            raise ValueError(
                f"Wavelength dimension mismatch in {shard_path}: "
                f"dataset W={spectra.size(-1)}, target W={target.size(-1)}"
            )

        for start in range(0, spectra.size(0), chunk_size):
            end = min(start + chunk_size, spectra.size(0))
            chunk = spectra[start:end].to(device_t, dtype=torch.float32, non_blocking=True)
            chunk = chunk.index_select(1, channels)
            if wl_mask is not None:
                chunk = chunk[:, :, wl_mask]

            mae = (chunk - target_d.unsqueeze(0)).abs().mean(dim=(1, 2))
            value, local = torch.min(mae, dim=0)
            value_f = float(value.item())
            if value_f < best_mae:
                local_i = start + int(local.item())
                best_mae = value_f
                best_global_index = global_offset + local_i
                best_local_index = local_i
                best_file = str(shard_path)
                best_spectrum = spectra[local_i].to(torch.float32).clone()

        global_offset += int(spectra.size(0))

    if best_spectrum is None:
        raise RuntimeError("No spectra were processed while searching for nearest neighbor.")

    return {
        "spectrum": best_spectrum,
        "mae": best_mae,
        "global_index": int(best_global_index),
        "file": best_file,
        "local_index": int(best_local_index),
    }


@torch.no_grad()
def find_nearest_training_spectrum_cached(
    target: torch.Tensor,
    cfg: dict[str, Any],
    block: dict[str, Any],
    wavelengths: torch.Tensor,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """
    Return the nearest training/test spectrum, using a persistent metadata cache when enabled.
    """
    paths = _dataset_paths_from_cfg(cfg, block)
    files = _collect_safetensors(paths)
    cache_enabled = bool(block.get("CACHE_ENABLED", True))
    cache_path = _nn_cache_path(cfg, block)
    key = _cache_key(target, wavelengths, block, files)

    if cache_enabled:
        cache = _load_nn_cache(cache_path)
        entry = cache.get("entries", {}).get(key)
        if isinstance(entry, dict):
            cached = _load_cached_match(entry, target_width=int(target.size(-1)))
            if cached is not None:
                cached["cache"] = {"hit": True, "path": str(cache_path), "key": key}
                return cached
    else:
        cache = {"version": 1, "entries": {}}

    match = find_nearest_training_spectrum(target, cfg, block, wavelengths, device=device, files=files)
    match["cache"] = {"hit": False, "path": str(cache_path), "key": key}

    if cache_enabled:
        cache.setdefault("version", 1)
        cache.setdefault("entries", {})
        cache["entries"][key] = {
            "mae": float(match["mae"]),
            "global_index": int(match["global_index"]),
            "file": str(match["file"]),
            "local_index": int(match["local_index"]),
        }
        _save_nn_cache(cache_path, cache)

    return match


def _apply_nn_blend(
    spectrum: torch.Tensor,
    cfg: dict[str, Any],
    block: dict[str, Any],
    wavelengths: torch.Tensor,
    device: torch.device | str | None,
) -> tuple[torch.Tensor, dict[str, Any] | None]:
    nn_cfg = block.get("NN_BLEND") or {}
    if not bool(nn_cfg.get("ENABLED", False)):
        return spectrum, None

    nn_cfg = dict(nn_cfg)
    nn_cfg.setdefault("ROI_MIN", cfg.get("ROI_MIN", float(wavelengths.min())))
    nn_cfg.setdefault("ROI_MAX", cfg.get("ROI_MAX", float(wavelengths.max())))

    nn_device = nn_cfg.get("DEVICE")
    if nn_device is None:
        nn_device = device
    match = find_nearest_training_spectrum_cached(spectrum, cfg, nn_cfg, wavelengths, device=nn_device)
    nn_spectrum = match["spectrum"].to(spectrum.device, dtype=spectrum.dtype)

    mode = str(nn_cfg.get("MODE", "outside_roi")).lower()
    if mode == "all":
        weight = torch.full_like(wavelengths, float(nn_cfg.get("BLEND", 0.5)), dtype=torch.float32)
    elif mode == "outside_roi":
        weight = _blend_weights(wavelengths, nn_cfg)
    else:
        raise ValueError(f"Unknown TARGET_PHYSICALIZE.NN_BLEND.MODE: {mode!r}")

    weight = weight.to(spectrum.device, dtype=spectrum.dtype).view(1, -1)
    blended = (1.0 - weight) * spectrum + weight * nn_spectrum
    return _normalize_rat(blended, cfg), match


def _apply_autoencoder_projection(
    spectrum: torch.Tensor,
    cfg: dict[str, Any],
    block: dict[str, Any],
    wavelengths: torch.Tensor,
    device: torch.device | str | None,
) -> tuple[torch.Tensor, dict[str, Any] | None]:
    ae_cfg = block.get("AE_PROJECT") or {}
    if not bool(ae_cfg.get("ENABLED", False)):
        return spectrum, None

    checkpoint = ae_cfg.get("CHECKPOINT")
    if not checkpoint:
        raise ValueError("TARGET_PHYSICALIZE.AE_PROJECT.CHECKPOINT must be set when AE_PROJECT is enabled.")

    ae_device = ae_cfg.get("DEVICE")
    if ae_device is None:
        ae_device = device

    from optollama.model.spectrum_autoencoder import project_spectrum_with_autoencoder

    projected, info = project_spectrum_with_autoencoder(spectrum, checkpoint, device=ae_device)
    projected = projected.to(spectrum.device, dtype=spectrum.dtype)

    mode = str(ae_cfg.get("MODE", "all")).lower()
    if mode == "all":
        weight = torch.full_like(wavelengths, float(ae_cfg.get("BLEND", 1.0)), dtype=torch.float32)
    elif mode == "outside_roi":
        ae_cfg = dict(ae_cfg)
        ae_cfg.setdefault("ROI_MIN", cfg.get("ROI_MIN", float(wavelengths.min())))
        ae_cfg.setdefault("ROI_MAX", cfg.get("ROI_MAX", float(wavelengths.max())))
        weight = _blend_weights(wavelengths, ae_cfg)
    else:
        raise ValueError(f"Unknown TARGET_PHYSICALIZE.AE_PROJECT.MODE: {mode!r}")

    weight = weight.to(spectrum.device, dtype=spectrum.dtype).view(1, -1)
    blended = (1.0 - weight) * spectrum + weight * projected
    info["mode"] = mode
    info["blend_mean"] = float(weight.mean().item())
    return _normalize_rat(blended, cfg), info


def physicalize_target_spectrum(
    spectrum: torch.Tensor,
    cfg: dict[str, Any],
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Adapt an ideal target spectrum toward the realistic training-data style.
    """
    spectrum, _ = ensure_3w(spectrum)
    spectrum = spectrum.to(torch.float32).clone()
    info: dict[str, Any] = {"enabled": False}

    if not _enabled(cfg):
        return spectrum, info

    block = cfg.get("TARGET_PHYSICALIZE") or {}
    wavelengths = torch.as_tensor(cfg["WAVELENGTHS"], dtype=torch.float32, device=spectrum.device)

    conditioned = _normalize_rat(spectrum, cfg)
    conditioned = _apply_level_relaxation(conditioned, block, cfg)
    conditioned = _apply_edge_smoothing(conditioned, wavelengths, block, cfg)
    conditioned, ae_info = _apply_autoencoder_projection(conditioned, cfg, block, wavelengths, device)
    conditioned, nn_info = _apply_nn_blend(conditioned, cfg, block, wavelengths, device)
    conditioned = _normalize_rat(conditioned, cfg)

    info = {
        "enabled": True,
        "autoencoder": ae_info,
        "nn": None
        if nn_info is None
        else {
            "mae": nn_info["mae"],
            "global_index": nn_info["global_index"],
            "file": nn_info["file"],
            "local_index": nn_info["local_index"],
            "cache": nn_info.get("cache"),
        },
    }
    return conditioned, info
