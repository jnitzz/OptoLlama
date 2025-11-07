# --- utils.py additions -------------------------------------------------------
import json, csv, os, math
import numpy as np
import torch

# ---------- core helpers ----------

def _cfg_grid_or_default(cfg=None, json_obj=None):
    """
    Decide the wavelength grid.
    Priority:
      1) JSON grid {start, stop, step}
      2) cfg.WAVELENGTHS (np-like)
      3) fallback 300..2000 step 10
    """
    if isinstance(json_obj, dict) and "grid" in json_obj:
        g = json_obj["grid"]
        wl = np.arange(float(g["start"]), float(g["stop"]) + 1e-9, float(g["step"]), dtype=float)
        return wl
    if cfg is not None and hasattr(cfg, "WAVELENGTHS"):
        wl = np.asarray(getattr(cfg, "WAVELENGTHS"), dtype=float)
        if wl.ndim != 1 or wl.size < 2:
            raise ValueError("cfg.WAVELENGTHS must be 1-D with at least 2 points")
        return wl
    return np.arange(300.0, 2000.0 + 1e-9, 10.0, dtype=float)

def _nonneg(x):
    return np.maximum(x, 0.0)

def normalize_rat(R, A, T, normalize_to=1.0, fallback=(0.0, 0.0, 1.0)):
    """
    Enforce non-negativity and normalize per wavelength so R+A+T = normalize_to.
    If the sum is 0, use fallback (defaults to T=1).
    """
    R = _nonneg(np.asarray(R, dtype=float))
    A = _nonneg(np.asarray(A, dtype=float))
    T = _nonneg(np.asarray(T, dtype=float))
    total = R + A + T
    eps = 1e-12
    mask = total > eps
    scale = np.zeros_like(total)
    scale[mask] = normalize_to / total[mask]
    R[mask] *= scale[mask]
    A[mask] *= scale[mask]
    T[mask] *= scale[mask]
    if np.any(~mask):
        R[~mask], A[~mask], T[~mask] = fallback
    return R, A, T


def normalize_rat_fillT_cropT(R, A, T, target=1.0):
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

        # crop from R
        still = over > 0
        if np.any(still):
            cut = np.minimum(R[still], over[still])
            R[still] -= cut
            over[still] -= cut
            # By construction, over should now be 0

    return R, A, T


# ---------- shapes for JSON "edits" ----------

def _fwhm_to_sigma(fwhm):
    return float(fwhm) / (2.0 * math.sqrt(2.0 * math.log(2.0)))

def _gaussian(x, center, fwhm, amplitude):
    sigma = _fwhm_to_sigma(fwhm)
    return float(amplitude) * np.exp(-0.5 * ((x - float(center)) / sigma) ** 2)

def _lorentzian(x, center, fwhm, amplitude):
    gamma = float(fwhm) / 2.0
    return float(amplitude) * (gamma**2) / ((x - float(center))**2 + gamma**2)

def _supergaussian(x, center, fwhm, amplitude, order=4):
    # supergaussian: exp(-0.5 * ((x-c)/sigma)^(2*order))
    sigma = _fwhm_to_sigma(fwhm)
    return float(amplitude) * np.exp(-0.5 * np.abs((x - float(center)) / sigma) ** (2 * int(order)))

def _box(x, lo, hi, value):
    lo, hi = float(lo), float(hi)
    y = np.zeros_like(x, dtype=float)
    mask = (x >= lo) & (x <= hi)
    y[mask] = float(value)
    return y

def _linear_ramp(x, lo, hi, start, end):
    lo, hi = float(lo), float(hi)
    start, end = float(start), float(end)
    y = np.zeros_like(x, dtype=float)
    m = (x >= lo) & (x <= hi)
    if hi > lo:
        t = (x[m] - lo) / (hi - lo)
        y[m] = start + t * (end - start)
    return y

def _apply_combine(dst, src, how="overwrite"):
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

def _apply_json_edit(wl, arr, edit, combine="overwrite"):
    """
    Returns a new array after applying one edit to an existing target array.
    """
    edit = dict(edit)  # shallow copy
    typ = edit.get("type")
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
        src = _box(wl, lo, hi, edit["value"])
        return _apply_combine(arr, src, combine)

    # procedural shapes
    tgt = np.zeros_like(arr, dtype=float)
    if typ == "gaussian_peak":
        tgt = _gaussian(wl, edit["center"], edit["fwhm"], edit["amplitude"])
    elif typ == "lorentzian_peak":
        tgt = _lorentzian(wl, edit["center"], edit["fwhm"], edit["amplitude"])
    elif typ == "supergaussian_peak":
        tgt = _supergaussian(wl, edit["center"], edit["fwhm"], edit["amplitude"], edit.get("order", 4))
    elif typ == "box":
        lo, hi = edit["range"]
        tgt = _box(wl, lo, hi, edit["value"])
    elif typ == "linear_ramp":
        lo, hi = edit["range"]
        tgt = _linear_ramp(wl, lo, hi, edit["start"], edit["end"])
    else:
        raise ValueError(f"Unsupported edit type: {typ}")

    return _apply_combine(arr, tgt, combine)

# ---------- CSV loader ----------

def _load_csv(path, wl_grid, defaults=(0.0, 1.0, 0.0)):
    """
    Load sparse numeric samples from CSV with columns: wavelength_nm,R,T,A
    Interpolate onto wl_grid; missing columns fall back to defaults.
    """
    data = {"wavelength_nm": [], "R": [], "T": [], "A": []}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = [c.strip() for c in reader.fieldnames or []]
        if "wavelength_nm" not in cols:
            raise ValueError("CSV must contain a 'wavelength_nm' column.")
        for row in reader:
            w = float(row["wavelength_nm"])
            data["wavelength_nm"].append(w)
            for ch in ("R", "T", "A"):
                v = row.get(ch, "")
                data[ch].append(float(v) if (v is not None and v.strip() != "") else np.nan)

    if len(data["wavelength_nm"]) == 0:
        raise ValueError("CSV appears empty.")

    w = np.asarray(data["wavelength_nm"], dtype=float)
    order = np.argsort(w)
    w = w[order]

    def interp_or_default(series, default_val):
        y = np.asarray(series, dtype=float)[order]
        # mask valid samples
        mask = np.isfinite(y)
        if np.count_nonzero(mask) >= 2:
            return np.interp(wl_grid, w[mask], y[mask])
        elif np.count_nonzero(mask) == 1:
            # constant extrapolation from the single known point
            return np.full_like(wl_grid, y[mask][0], dtype=float)
        else:
            return np.full_like(wl_grid, float(default_val), dtype=float)

    R = interp_or_default(data["R"], defaults[0])
    T = interp_or_default(data["T"], defaults[1])
    A = interp_or_default(data["A"], defaults[2])
    return R, A, T

# ---------- JSON loader ----------

def _load_json(path, cfg=None):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    wl = _cfg_grid_or_default(cfg, obj)

    # defaults
    d = obj.get("defaults", {}) or {}
    R = np.full_like(wl, float(d.get("R", 0.0)), dtype=float)
    T = np.full_like(wl, float(d.get("T", 1.0)), dtype=float)
    A = np.full_like(wl, float(d.get("A", 0.0)), dtype=float)

    combine = (obj.get("combine") or "overwrite").lower()
    edits = obj.get("edits", []) or []
    for ed in edits:
        target = ed.get("target", "").upper()
        if target not in ("R", "A", "T"):
            raise ValueError(f"Edit missing/invalid 'target': {ed}")
        if target == "R":
            R = _apply_json_edit(wl, R, ed, combine)
        elif target == "A":
            A = _apply_json_edit(wl, A, ed, combine)
        else:
            T = _apply_json_edit(wl, T, ed, combine)

    # normalize_to = float(obj.get("normalize_to", 1.0))
    # Enforce non-negativity and normalize
    # R, A, T = normalize_rat(R, A, T, normalize_to=normalize_to, fallback=(0.0, 0.0, normalize_to))
    R, A, T = normalize_rat_fillT_cropT(R, A, T, target=float(obj.get("normalize_to", 1.0)))
    return wl, R, A, T, obj

# ---------- public entry point ----------

def load_spectra_from_json_or_csv(
    path: str,
    cfg=None,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, np.ndarray, dict]:
    """
    Load a human-readable RAT spectrum from JSON or CSV and return:
      spec_tensor: torch.Tensor [W, 3] in (R, A, T) order
      wavelengths: np.ndarray [W] (nm)
      meta: dict with info {'source', 'normalize_to', 'grid', ...}

    JSON supports procedural 'edits'; CSV supports numeric samples.
    Both enforce non-negativity and normalize so R+A+T=1 per wavelength.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        wl, R, A, T, obj = _load_json(path, cfg=cfg)
        meta = {
            "source": "json",
            "normalize_to": float(obj.get("normalize_to", 1.0)),
            "combine": (obj.get("combine") or "overwrite"),
            "grid": obj.get("grid", None),
            "metadata": obj.get("metadata", None),
        }
    elif ext == ".csv":
        wl = _cfg_grid_or_default(cfg, json_obj=None)
        R, A, T = _load_csv(path, wl, defaults=(0.0, 1.0, 0.0))
        # R, A, T = normalize_rat(R, A, T, normalize_to=1.0, fallback=(0.0, 0.0, 1.0))
        R, A, T = normalize_rat_fillT_cropT(R, A, T, target=1.0)
        meta = {
            "source": "csv",
            "normalize_to": 1.0,
            "grid": {"start": float(wl[0]), "stop": float(wl[-1]), "step": float(np.diff(wl).mean())},
            "metadata": None,
        }
    else:
        raise ValueError(f"Unsupported file extension: {ext} (use .json or .csv)")

    spec = np.stack([R, A, T], axis=-1).T  # [W, 3] in (R, A, T)
    ten = torch.tensor(spec, dtype=dtype, device=device)  # ready for model
    return ten, wl, meta

# from utils import load_spectra_from_json_or_csv
# import config_MD60 as cfg
# spec, wavelengths, meta = load_spectra_from_json_or_csv(
#     os.path.join(cfg.PATH_DATA,"test1.json"), cfg=cfg, device="cuda"  # or "cpu"
# )
# %%

# --- single-spec dataset for interactive inference ---------------------------
import torch
from torch.utils.data import Dataset, DataLoader

class _SingleSpecDataset(Dataset):
    """
    Minimal dataset to feed a single [W,3] spectrum into validate_model.
    We provide a dummy 'stacks' tensor of length max_stack_depth filled with PAD.
    """
    def __init__(self, spec_tensor: torch.Tensor, max_stack_depth: int, pad_idx: int):
        assert spec_tensor.ndim == 2 and spec_tensor.shape[0] == 3, "spec must be [3, W] (R,A,T channels first)"
        self.spectra = [spec_tensor]  # validate_model / utils_data expect attribute 'spectra'
        self.maximum_depth = int(max_stack_depth)
        self.pad_idx = int(pad_idx)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        spectra = self.spectra[0]                                    # [W,3]
        stacks  = torch.full((self.maximum_depth,), self.pad_idx)    # dummy target; unused for NOSIM
        return spectra, stacks

def make_single_spec_loader(spec_tensor: torch.Tensor, max_stack_depth: int, pad_idx: int, *, batch_size: int = 1):
    ds = _SingleSpecDataset(spec_tensor, max_stack_depth=max_stack_depth, pad_idx=pad_idx)
    # Simple default collate (no DDP sharding needed here)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return ds, loader, None
