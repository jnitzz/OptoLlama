import json
import csv
import math
import numpy as np
import torch
from utils import ensure_3W, apply_noise, apply_smoothing, redistribute_mismatch

def normalize_rat_fill_crop(R, A, T, target=1.0):
    """
    Non-negative RAT with T as the filler and primary crop source.

    Steps per wavelength:
      1) Clamp R,A,T >= 0
      2) If sum < target: add (target - sum) to T
      3) If sum > target: reduce T first; if still > target, reduce A; then R
         (never below 0). This guarantees sum == target afterward.

    Returns R, A, T as float arrays.
    """
    R = np.maximum(np.asarray(R, dtype=float), 0.0)
    A = np.maximum(np.asarray(A, dtype=float), 0.0)
    T = np.maximum(np.asarray(T, dtype=float), 0.0)

    total = R + A + T
    # case 1: fill deficit into T
    deficit = target - total
    need = deficit > 0
    T[need] += deficit[need]   # safe since deficit>0 only at 'need'

    # case 2: crop excess from T, then A, then R
    excess = (R + A + T) - target
    over = np.maximum(excess, 0.0)
    has_over = over > 0

    if np.any(has_over):
        # crop from T
        cut = np.minimum(T[has_over], over[has_over])
        T[has_over] -= cut
        over[has_over] -= cut
        
        # crop from A
        still = over > 0
        if np.any(still):
            cut = np.minimum(A[still], over[still])
            A[still] -= cut
            over[still] -= cut
            # By construction, over should now be 0
        
        # crop from R
        still = over > 0
        if np.any(still):
            cut = np.minimum(R[still], over[still])
            R[still] -= cut
            over[still] -= cut
        
    return R.astype(np.float32), A.astype(np.float32), T.astype(np.float32)


def fwhm_to_sigma(fwhm):
    return float(fwhm) / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def gaussian(x, center, fwhm, amplitude):
    sigma = fwhm_to_sigma(fwhm)
    return float(amplitude) * np.exp(-0.5 * ((x - float(center)) / sigma) ** 2)


def lorentzian(x, center, fwhm, amplitude):
    gamma = float(fwhm) / 2.0
    return float(amplitude) * (gamma**2) / ((x - float(center))**2 + gamma**2)


def supergaussian(x, center, fwhm, amplitude, order=4):
    # supergaussian: exp(-0.5 * ((x-c)/sigma)^(2*order))
    sigma = fwhm_to_sigma(fwhm)
    return float(amplitude) * np.exp(-0.5 * np.abs((x - float(center)) / sigma) ** (2 * int(order)))


def box(x, lo, hi, value):
    lo, hi = float(lo), float(hi)
    y = np.zeros_like(x, dtype=float)
    mask = (x >= lo) & (x <= hi)
    y[mask] = float(value)
    return y


def linear_ramp(x, lo, hi, start, end):
    lo, hi = float(lo), float(hi)
    start, end = float(start), float(end)
    y = np.zeros_like(x, dtype=float)
    m = (x >= lo) & (x <= hi)
    if hi > lo:
        t = (x[m] - lo) / (hi - lo)
        y[m] = start + t * (end - start)
    return y


def apply_combine(dst, src, how="overwrite"):
    how = (how or "overwrite").lower()
    if how == "overwrite":
        return src
    if how == "max":
        return np.maximum(dst, src)
    if how == "min":
        return np.minimum(dst, src)
    if how == "add":
        return dst + src
    if how == "multiply":
        return dst * src
    raise ValueError(f"Unknown combine policy: {how}")


def apply_json_edit(wl, arr, edit, combine="overwrite"):
    """
    Returns a new array after applying one edit to an existing target array.
    """
    edit = dict(edit)  # shallow copy
    typ = edit.get("type")
    # Optional wavelength mask
    rng = edit.get("range", None)
    if rng and len(rng) == 2:
        lo, hi = float(rng[0]), float(rng[1])
        mask = (wl >= lo) & (wl <= hi)
    else:
        mask = np.ones_like(wl, dtype=bool)
        
    if "at" in edit and "value" in edit and typ is None:
        # point assignment
        x = float(edit["at"])
        v = float(edit["value"])
        out = arr.copy()
        # find nearest grid index
        i = int(np.argmin(np.abs(wl - x)))
        # overwrite policy only makes sense at a point; treat others as overwrite here
        out[i] = v
        return out

    if "range" in edit and "value" in edit and typ is None:
        # constant range
        lo, hi = edit["range"]
        src = box(wl, lo, hi, edit["value"])
        return apply_combine(arr, src, combine)

    # procedural shapes
    tgt = np.zeros_like(arr, dtype=float)
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
    out = arr.copy()
    combined = apply_combine(arr, tgt, combine)
    out[mask] = combined[mask]
    return out


def load_spectra_from_json_or_csv(
    path: str,
    *,
    expect_shape="3xW",
    cfg=None,
    wavelengths_override=None,
    noise_cfg=None,
    smooth_cfg=None,
    mismatch_order=None,
):

    # figure out wavelengths (needed for sizing and range masks)
    wl = None
    if wavelengths_override is not None:
        wl = wavelengths_override
    elif cfg is not None and hasattr(cfg, "WAVELENGTHS"):
        wl = getattr(cfg, "WAVELENGTHS")
    if wl is None:
        raise ValueError("WAVELENGTHS are required (provide cfg.WAVELENGTHS or wavelengths_override).")
    wl = np.asarray(wl, dtype=np.float32)
    W = int(wl.shape[0])

    ncfg = noise_cfg if noise_cfg is not None else (getattr(cfg, "NOISE", None) if cfg is not None else None)
    scfg = smooth_cfg if smooth_cfg is not None else (getattr(cfg, "SMOOTH", None) if cfg is not None else None)
    morder = mismatch_order if mismatch_order is not None else (getattr(cfg, "MISMATCH_FILL_ORDER", "R>A>T") if cfg is not None else "R>A>T")

    # --- load
    if path.lower().endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)

        # Support your example format with defaults/edits/normalize_to
        # If "spectra" key exists, we fall back to the old path.
        if "spectra" in data:
            arr = np.asarray(data["spectra"], dtype=np.float32)
            x = torch.from_numpy(arr)
            x, _ = ensure_3W(x)
        else:
            # --- build from defaults
            defaults = data.get("defaults", {})
            R0 = float(defaults.get("R", 0.0))
            A0 = float(defaults.get("A", 0.0))
            T0 = float(defaults.get("T", 1.0))
            base = np.stack([np.full(W, R0, np.float32),
                             np.full(W, A0, np.float32),
                             np.full(W, T0, np.float32)], axis=0)  # [3,W]

            file_combine = (data.get("combine_policy")
                            or data.get("combine")
                            or "overwrite")
            for e in data.get("edits", []):
                if not isinstance(e, dict):
                    continue
                # Scalar channel edits (R/A/T, +R, *R etc.)
                did_scalar = False
                rng = e.get("range", None)
                if rng and len(rng) == 2:
                    lo, hi = float(rng[0]), float(rng[1])
                    mask = (wl >= lo) & (wl <= hi)
                else:
                    mask = np.ones(W, dtype=bool)
                for ch_name, idx in (("R",0), ("A",1), ("T",2)):
                    if ch_name in e:
                        base[idx, mask] = float(e[ch_name]); did_scalar = True
                    if ("+"+ch_name) in e:
                        base[idx, mask] = base[idx, mask] + float(e["+"+ch_name]); did_scalar = True
                    if ("*"+ch_name) in e:
                        base[idx, mask] = base[idx, mask] * float(e["*"+ch_name]); did_scalar = True

                # Procedural edit (type/target)
                if (not did_scalar) and ("type" in e):
                    # allow single string or list for target(s)
                    targets = e.get("target", "R")
                    if isinstance(targets, str):
                        targets = [targets]
                    combine_policy = e.get("combine", file_combine) or "overwrite"
                    for tname in targets:
                        tname = tname.upper()
                        if tname not in ("R","A","T"):
                            continue
                        idx = {"R":0,"A":1,"T":2}[tname]
                        base[idx] = apply_json_edit(wl, base[idx], e, combine=combine_policy)

            x = torch.from_numpy(base)  # [3,W]

            # Normalize using fill-crop (fill deficits into T, crop excess from T then A then R)
            target_sum = float(data.get("normalize_to", 1.0))
            Rn, An, Tn = normalize_rat_fill_crop(base[0], base[1], base[2], target=target_sum)
            x = torch.from_numpy(np.stack([Rn, An, Tn], axis=0).astype(np.float32))

    else:
        # CSV path unchanged; adapt to your CSV layout as you already had
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            rows = [[float(x) for x in r] for r in reader if len(r) > 0]
        arr = np.asarray(rows, dtype=np.float32).T  # if rows=W, cols=3
        x = torch.from_numpy(arr)
        x, _ = ensure_3W(x)

    # --- final processing: noise -> smoothing -> enforce sum to 1 with order
    x = apply_noise(x, ncfg, wl)
    x = apply_smoothing(x, scfg)
    # Keep a small final correction, but DO NOT re-create a zero-baseline fill into R
    # If you want to be extra safe, prefer filling into T by default:
    #   (set cfg.MISMATCH_FILL_ORDER = "T>A>R")
    x = redistribute_mismatch(x, morder, target_sum=1.0)
    x = x.to(torch.float32)
    # --- output shape
    if expect_shape.lower() in ("3xw", "chw"):
        return x  # [3,W]
    elif expect_shape.lower() in ("wx3", "hwc"):
        return x.permute(1, 0).contiguous()  # [W,3]
    else:
        return x