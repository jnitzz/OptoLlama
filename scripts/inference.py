import csv
import json
import math
from typing import Any, Dict, Literal, Optional, Union

import torch
from utils import apply_noise, apply_smoothing, ensure_3w, normalize_rat_fill_crop, redistribute_mismatch

# ruff: noqa: E731


def fwhm_to_sigma(fwhm: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
    """Convert Full Width at Half Maximum (FWHM) to sigma for Gaussian functions.

    Args:
        fwhm: Full Width at Half Maximum.

    Returns
    -------
        Sigma value.
    """
    return float(fwhm) / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def gaussian(
    x: torch.Tensor,
    center: Union[float, torch.Tensor],
    fwhm: Union[float, torch.Tensor],
    amplitude: Union[float, torch.Tensor],
) -> torch.Tensor:
    """Compute a Gaussian function.

    Args:
        x: Input tensor.
        center: Center of the Gaussian.
        fwhm: Full Width at Half Maximum.
        amplitude: Amplitude of the Gaussian.

    Returns
    -------
        Gaussian function evaluated at x.
    """
    sigma = fwhm_to_sigma(fwhm)
    return float(amplitude) * torch.exp(-0.5 * ((x - float(center)) / sigma) ** 2)


def lorentzian(
    x: torch.Tensor,
    center: Union[float, torch.Tensor],
    fwhm: Union[float, torch.Tensor],
    amplitude: Union[float, torch.Tensor],
) -> torch.Tensor:
    """Compute a Lorentzian function.

    Args:
        x: Input tensor.
        center: Center of the Lorentzian.
        fwhm: Full Width at Half Maximum.
        amplitude: Amplitude of the Lorentzian.

    Returns
    -------
        Lorentzian function evaluated at x.
    """
    gamma = float(fwhm) / 2.0
    return float(amplitude) * (gamma**2) / ((x - float(center)) ** 2 + gamma**2)


def supergaussian(
    x: torch.Tensor,
    center: Union[float, torch.Tensor],
    fwhm: Union[float, torch.Tensor],
    amplitude: Union[float, torch.Tensor],
    order: int = 4,
) -> torch.Tensor:
    """Compute a super-Gaussian function.

    Args:
        x: Input tensor.
        center: Center of the super-Gaussian.
        fwhm: Full Width at Half Maximum.
        amplitude: Amplitude of the super-Gaussian.
        order: Order of the super-Gaussian (default: 4).

    Returns
    -------
        Super-Gaussian function evaluated at x.
    """
    sigma = fwhm_to_sigma(fwhm)
    return float(amplitude) * torch.exp(-0.5 * torch.abs((x - float(center)) / sigma) ** (2 * int(order)))


def box(x: torch.Tensor, lo: float, hi: float, value: float) -> torch.Tensor:
    """Compute a box (step) function.

    Args:
        x: Input tensor.
        lo: Lower bound of the box.
        hi: Upper bound of the box.
        value: Value of the box.

    Returns
    -------
        Box function evaluated at x.
    """
    lo, hi = float(lo), float(hi)
    y = torch.zeros_like(x, dtype=torch.float32)
    mask = (x >= lo) & (x <= hi)
    y[mask] = float(value)
    return y


def linear_ramp(x: torch.Tensor, lo: float, hi: float, start: float, end: float) -> torch.Tensor:
    """Compute a linear ramp function.

    Args:
        x: Input tensor.
        lo: Lower bound of the ramp.
        hi: Upper bound of the ramp.
        start: Value at the start of the ramp.
        end: Value at the end of the ramp.

    Returns
    -------
        Linear ramp function evaluated at x.
    """
    y = torch.zeros_like(x, dtype=torch.float32)
    lo, hi = float(lo), float(hi)
    start, end = float(start), float(end)
    m = (x >= lo) & (x <= hi)
    if hi > lo:
        t = (x[m] - lo) / (hi - lo)
        y[m] = start + t * (end - start)
    return y


def apply_combine(dst: torch.Tensor, src: torch.Tensor, how: str = "overwrite") -> torch.Tensor:
    """Combine following the method defined in 'how'."""
    how = (how or "overwrite").lower()
    if how == "overwrite":
        return src
    if how == "max":
        return torch.maximum(dst, src)
    if how == "min":
        return torch.minimum(dst, src)
    if how == "add":
        return dst + src
    if how == "multiply":
        return dst * src
    raise ValueError(f"Unknown combine policy: {how}")


def apply_json_edit(wl: torch.Tensor, arr: torch.Tensor, edit: Dict, combine: str = "overwrite") -> torch.Tensor:
    """Return a new array after applying one edit to an existing target array."""
    edit = dict(edit)
    typ = edit.get("type")
    # Optional wavelength mask
    rng = edit.get("range", None)
    if rng and len(rng) == 2:
        lo, hi = float(rng[0]), float(rng[1])
        mask = (wl >= lo) & (wl <= hi)
    else:
        mask = torch.ones_like(wl, dtype=bool)

    if "at" in edit and "value" in edit and typ is None:
        # point assignment
        x = float(edit["at"])
        v = float(edit["value"])
        out = arr.clone()
        # find nearest grid index
        i = int(torch.argmin(torch.abs(wl - x)))
        # overwrite policy only makes sense at a point; treat others as overwrite here
        out[i] = v
        return out

    if "range" in edit and "value" in edit and typ is None:
        # constant range
        lo, hi = edit["range"]
        src = box(wl, lo, hi, edit["value"])
        return apply_combine(arr, src, combine)

    # procedural shapes
    tgt = torch.zeros_like(arr, dtype=float)
    if typ == "gaussian_peak":
        tgt = gaussian(wl, edit["center"], edit["fwhm"], edit["amplitude"])
    elif typ == "lorentzian_peak":
        tgt = lorentzian(wl, edit["center"], edit["fwhm"], edit["amplitude"])
    elif typ == "supergaussian_peak":
        tgt = supergaussian(wl, edit["center"], edit["fwhm"], edit["amplitude"], edit.get("order", 4))
    elif typ == "box":
        lo, hi = edit["range"]
        tgt = box(wl, lo, hi, edit["value"])
    elif typ == "linear_ramp":
        lo, hi = edit["range"]
        tgt = linear_ramp(wl, lo, hi, edit["start"], edit["end"])
    else:
        raise ValueError(f"Unsupported edit type: {typ}")

    # Apply combine, but only inside mask; outside keep original
    out = arr.clone()
    combined = apply_combine(arr, tgt, combine)
    out[mask] = combined[mask]
    return out


def load_spectra_from_json_or_csv(
    path: str,
    *,
    expect_shape: Literal["3xW", "Wx3", "chw", "hwc"] = "3xW",
    cfg: Optional[object] = None,
    wavelengths_override: Optional[torch.Tensor] = None,
    noise_cfg: Optional[Dict[str, Any]] = None,
    smooth_cfg: Optional[Dict[str, Any]] = None,
    mismatch_order: Optional[str] = None,
) -> torch.Tensor:
    """Load spectra from a JSON or CSV file, with optional noise, smoothing, and normalization.

    Args:
        path: Path to the JSON or CSV file.
        expect_shape: Output shape format ("3xW" or "Wx3").
        cfg: Configuration object (optional).
        wavelengths_override: Override wavelengths (optional).
        noise_cfg: Noise configuration (optional).
        smooth_cfg: Smoothing configuration (optional).
        mismatch_order: Order for redistributing mismatch (optional).

    Returns
    -------
        Spectra tensor in the specified shape.

    Raises
    ------
        ValueError: If wavelengths are not provided.
    """
    # --- Determine wavelengths
    wl = (
        wavelengths_override
        if wavelengths_override is not None
        else (getattr(cfg, "WAVELENGTHS", None) if cfg is not None else None)
    )
    if wl is None:
        raise ValueError("WAVELENGTHS are required (provide cfg.WAVELENGTHS or wavelengths_override).")
    wl = torch.as_tensor(wl, dtype=torch.float32)
    w = wl.shape[0]

    # --- Load noise and smoothing configs
    ncfg = noise_cfg if noise_cfg is not None else (getattr(cfg, "NOISE", None) if cfg is not None else None)
    scfg = smooth_cfg if smooth_cfg is not None else (getattr(cfg, "SMOOTH", None) if cfg is not None else None)
    morder = (
        mismatch_order
        if mismatch_order is not None
        else (getattr(cfg, "MISMATCH_FILL_ORDER", "R>A>T") if cfg is not None else "R>A>T")
    )

    # --- Load data
    if path.lower().endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)
        if "spectra" in data:
            x = torch.as_tensor(data["spectra"], dtype=torch.float32)
            x, _ = ensure_3w(x)
        else:
            # --- Build from defaults
            defaults = data.get("defaults", {})
            r0, a0, t0 = (
                float(defaults.get("R", 0.0)),
                float(defaults.get("A", 0.0)),
                float(defaults.get("T", 1.0)),
            )
            base = torch.stack(
                [
                    torch.full((w,), r0, dtype=torch.float32),
                    torch.full((w,), a0, dtype=torch.float32),
                    torch.full((w,), t0, dtype=torch.float32),
                ],
                dim=0,
            )  # shape: [3, W]
            file_combine = data.get("combine_policy") or data.get("combine") or "overwrite"
            for edit in data.get("edits", []):
                if not isinstance(edit, dict):
                    continue
                # Scalar channel edits
                rng = edit.get("range")
                mask = (wl >= float(rng[0])) & (wl <= float(rng[1])) if rng and len(rng) == 2 else torch.ones(w, dtype=bool)
                for ch_name, idx in (("R", 0), ("A", 1), ("T", 2)):
                    for op in ("", "+", "*"):
                        key = f"{op}{ch_name}"
                        if key in edit:
                            val = float(edit[key])
                            if op == "":
                                base[idx, mask] = val
                            elif op == "+":
                                base[idx, mask] += val
                            elif op == "*":
                                base[idx, mask] *= val
                # Procedural edits
                if "type" in edit and not any(ch in edit for ch in ("R", "A", "T")):
                    targets = edit.get("target", ["R"])
                    targets = [targets] if isinstance(targets, str) else targets
                    combine_policy = edit.get("combine", file_combine) or "overwrite"
                    for tname in targets:
                        tname = tname.upper()
                        if tname in ("R", "A", "T"):
                            idx = {"R": 0, "A": 1, "T": 2}[tname]
                            base[idx] = apply_json_edit(wl, base[idx], edit, combine=combine_policy)

            # Normalize
            rn, an, tn = normalize_rat_fill_crop(base[0], base[1], base[2], target=float(data.get("normalize_to", 1.0)))
            x = torch.stack([rn, an, tn], dim=0)
    else:
        # --- Load from CSV
        with open(path, "r", newline="") as f:
            rows = [[float(x) for x in row] for row in csv.reader(f) if row]
        x = torch.tensor(rows, dtype=torch.float32).T  # shape: [3, W]
        x, _ = ensure_3w(x)

    # --- Final processing
    x = apply_noise(x, ncfg, wl)
    x = apply_smoothing(x, scfg)
    x = redistribute_mismatch(x, morder, target_sum=1.0)

    # --- Return in expected shape
    if expect_shape.lower() in ("3xw", "chw"):
        return x
    elif expect_shape.lower() in ("wx3", "hwc"):
        return x.permute(1, 0).contiguous()
    else:
        return x
