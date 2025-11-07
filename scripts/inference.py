# --- utils.py additions -------------------------------------------------------
import json, csv, math
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
    return R.astype(np.float32), A.astype(np.float32), T.astype(np.float32)


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


import torch.nn.functional as F

def _ensure_3W(x: torch.Tensor):
    """
    Accept [3,W], [W,3], [B,3,W], or [B,W,3]; return (x_[...,3,W], was_permuted)
    """
    if x.dim() == 2:
        if x.size(0) == 3:
            return x, False
        elif x.size(1) == 3:
            return x.permute(1, 0).contiguous(), True
    elif x.dim() == 3:
        if x.size(1) == 3:
            return x, False
        elif x.size(2) == 3:
            return x.permute(0, 2, 1).contiguous(), True
    raise ValueError(f"Expected shape [...,3,W] or [...,W,3], got {tuple(x.shape)}")
    
def _wl_mask(wavelengths, wl_min: float, wl_max: float, device):
    if wavelengths is None:
        return None
    wl = torch.as_tensor(wavelengths, dtype=torch.float32, device=device)
    return (wl >= float(wl_min)) & (wl <= float(wl_max))

def _boxcar_kernel(win: int, device):
    win = max(1, int(win))
    if win % 2 == 0: win += 1
    k = torch.ones(win, device=device, dtype=torch.float32) / float(win)
    return k

def _gauss_kernel(win: int, sigma: float, device):
    win = max(1, int(win))
    if win % 2 == 0: win += 1
    r = (win - 1) // 2
    xs = torch.arange(-r, r+1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (xs / max(1e-6, float(sigma)))**2)
    k = k / k.sum().clamp_min(1e-6)
    return k

def _smooth_1d(x_3w: torch.Tensor, method: str, win: int, sigma: float) -> torch.Tensor:
    # x_3w: [...,3,W] -> do depthwise 1D conv along last dim
    orig_dim = x_3w.dim()
    if orig_dim == 2:
        x_3w = x_3w.unsqueeze(0)  # [1,3,W]
    _, C, W = x_3w.shape
    device = x_3w.device
    if method.lower() == "gaussian":
        k = _gauss_kernel(win, sigma, device)
    else:
        k = _boxcar_kernel(win, device)
    weight = k.view(1, 1, -1).repeat(C, 1, 1)  # [C,1,K]
    x = F.pad(x_3w, (k.numel()//2, k.numel()//2), mode="reflect")
    x = F.conv1d(x, weight, groups=C)
    x = x[:, :, :W]
    return x.squeeze(0) if orig_dim == 2 else x

def _parse_order(order_str: str):
    order_str = (order_str or "R>A>T").upper()
    mapping = {'R':0, 'A':1, 'T':2}
    seq = [mapping[c.strip()] for c in order_str.split('>') if c.strip() in mapping]
    rest = [i for i in (0,1,2) if i not in seq]
    seq.extend(rest)
    return tuple(seq[:3])

def _redistribute_mismatch(x: torch.Tensor, order: str, target_sum: float = 1.0):
    """
    x: [...,3,W] with values in [0,1]. Enforce per-W sum≈target_sum
    by distributing residual in the given priority order.
    """
    x, _ = _ensure_3W(x)
    orig_dim = x.dim()
    if orig_dim == 2:
        x = x.unsqueeze(0)  # [1,3,W]
    B, C, W = x.shape
    pri = _parse_order(order)
    total = x.sum(dim=1, keepdim=True)                  # [B,1,W]
    res = float(target_sum) - total                     # +: add, -: remove

    for idx in pri:
        ch = x[:, idx:idx+1, :]
        if (res.abs() < 1e-12).all():
            break
        # add
        add_capacity = (1.0 - ch).clamp_min(0.0)
        add = torch.sign(res) * torch.minimum(res.clamp_min(0.0), add_capacity)
        ch = ch + add
        res = res - add
        # remove
        rem_capacity = ch.clamp_max(1.0)
        rem = -torch.minimum((-res).clamp_min(0.0), rem_capacity)
        ch = ch + rem
        res = res - rem
        x[:, idx:idx+1, :] = ch

    x = x.clamp_(0.0, 1.0)
    return x.squeeze(0) if orig_dim == 2 else x

def _apply_noise(x: torch.Tensor, noise_cfg: dict, wavelengths):
    if not noise_cfg or not noise_cfg.get("enabled", False):
        return x
    x, _ = _ensure_3W(x)
    orig_dim = x.dim()
    if orig_dim == 2:
        x = x.unsqueeze(0)  # [1,3,W]
    B, C, W = x.shape
    device = x.device

    sigma_abs = float(noise_cfg.get("sigma_abs", 0.0))
    sigma_rel = float(noise_cfg.get("sigma_rel", 0.0))
    per_ch = noise_cfg.get("per_channel", [1.0, 1.0, 1.0])
    per_ch = torch.tensor(per_ch, dtype=torch.float32, device=device).view(1, C, 1)

    wl_min = noise_cfg.get("wl_min", None)
    wl_max = noise_cfg.get("wl_max", None)
    mask = None
    if wl_min is not None and wl_max is not None:
        mask = _wl_mask(wavelengths, wl_min, wl_max, device) if wavelengths is not None else None

    eps = torch.randn_like(x) * (sigma_abs + sigma_rel * x)
    eps = eps * per_ch

    if mask is not None:
        m = mask.view(1, 1, W)
        x = torch.where(m, x + eps, x)
    else:
        x = x + eps

    if noise_cfg.get("clip_0_1", True):
        x = x.clamp_(0.0, 1.0)

    return x.squeeze(0) if orig_dim == 2 else x

def _apply_smoothing(x: torch.Tensor, smooth_cfg: dict):
    if not smooth_cfg or not smooth_cfg.get("enabled", False):
        return x
    x, _ = _ensure_3W(x)
    method = smooth_cfg.get("method", "gaussian")
    win    = int(smooth_cfg.get("win", 5))
    sigma  = float(smooth_cfg.get("sigma", 1.0))
    return _smooth_1d(x, method, win, sigma)
# ------------------------------------------------------------------------------


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

    # Apply combine, but only inside mask; outside keep original
    out = arr.copy()
    combined = _apply_combine(arr, tgt, combine)
    out[mask] = combined[mask]
    return out

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
    A = np.full_like(wl, float(d.get("A", 0.0)), dtype=float)
    T = np.full_like(wl, float(d.get("T", 1.0)), dtype=float)

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
    R, A, T = normalize_rat_fill_crop(R, A, T, target=float(obj.get("normalize_to", 1.0)))
    return wl, R, A, T, obj


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
    import json, csv
    import numpy as np
    import torch

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

    # pull knobs
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
            x, _ = _ensure_3W(x)
        else:
            # --- build from defaults
            defaults = data.get("defaults", {})
            R0 = float(defaults.get("R", 0.0))
            A0 = float(defaults.get("A", 0.0))
            T0 = float(defaults.get("T", 1.0))
            base = np.stack([np.full(W, R0, np.float32),
                             np.full(W, A0, np.float32),
                             np.full(W, T0, np.float32)], axis=0)  # [3,W]

            # --- apply edits (optional)
            # Supported edit fields:
            #   range: [wl_min, wl_max] (inclusive; honoured for both scalars and procedural types)
            #   absolute overrides: "R", "A", "T" (scalars)
            #   additive changes:   "+R", "+A", "+T" (scalars)
            #   multiplicative:     "*R", "*A", "*T" (scalars)
            #   procedural:         {"target":"R|A|T" or ["R","A",...], "type":"gaussian_peak|lorentzian_peak|supergaussian_peak|box|linear_ramp", ...}
            # Combine policy can be given per-edit via "combine" or as file default via top-level "combine" / "combine_policy".
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
                        base[idx] = _apply_json_edit(wl, base[idx], e, combine=combine_policy)

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
        x, _ = _ensure_3W(x)

    # --- final processing: noise -> smoothing -> enforce sum to 1 with order
    x = _apply_noise(x, ncfg, wl)
    x = _apply_smoothing(x, scfg)
    # Keep a small final correction, but DO NOT re-create a zero-baseline fill into R
    # If you want to be extra safe, prefer filling into T by default:
    #   (set cfg.MISMATCH_FILL_ORDER = "T>A>R")
    x = _redistribute_mismatch(x, morder, target_sum=1.0)
    x = x.to(torch.float32)
    # --- output shape
    if expect_shape.lower() in ("3xw", "chw"):
        return x  # [3,W]
    elif expect_shape.lower() in ("wx3", "hwc"):
        return x.permute(1, 0).contiguous()  # [W,3]
    else:
        return x


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


# --- repeated-spec dataset for N targets with fresh noise each sample ---------
class _RepeatedSpecDataset(Dataset):
    """
    Dataset that repeats a *base* [3,W] spectrum N times.
    If NOISE.enabled=True, each item draws fresh noise (and smoothing).
    """
    def __init__(
        self,
        base_spec_3w: torch.Tensor,   # [3,W] before noise/smoothing
        n_items: int,
        max_stack_depth: int,
        pad_idx: int,
        *,
        wavelengths,                  # np.ndarray [W] (used for wl masks)
        noise_cfg: dict | None,
        smooth_cfg: dict | None,
        mismatch_order: str = "R>A>T",
    ):
        assert base_spec_3w.ndim == 2 and base_spec_3w.shape[0] == 3, "base_spec must be [3,W]"
        self.base = base_spec_3w.detach().clone()     # untouched template
        self.n = int(n_items)
        self.maximum_depth = int(max_stack_depth)
        self.pad_idx = int(pad_idx)
        self.wavelengths = wavelengths
        self.noise_cfg = noise_cfg or {"enabled": False}
        self.smooth_cfg = smooth_cfg or {"enabled": False}
        self.mismatch_order = mismatch_order

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = self.base.clone()  # [3,W]
        # draw fresh noise per item (or keep identical if disabled)
        x = _apply_noise(x, self.noise_cfg, self.wavelengths)
        x = _apply_smoothing(x, self.smooth_cfg)
        x = _redistribute_mismatch(x, self.mismatch_order, target_sum=1.0)
        stacks = torch.full((self.maximum_depth,), self.pad_idx)  # dummy; eval uses TMM
        return x, stacks

def make_repeated_spec_loader(
    base_spec_3w: torch.Tensor,
    n_items: int,
    max_stack_depth: int,
    pad_idx: int,
    *,
    wavelengths,
    cfg=None,
    batch_size: int = 1,
):
    # pull knobs from cfg (if present)
    noise_cfg = getattr(cfg, "NOISE", None) if cfg is not None else None
    smooth_cfg = getattr(cfg, "SMOOTH", None) if cfg is not None else None
    mismatch_order = getattr(cfg, "MISMATCH_FILL_ORDER", "R>A>T") if cfg is not None else "R>A>T"

    ds = _RepeatedSpecDataset(
        base_spec_3w, n_items, max_stack_depth, pad_idx,
        wavelengths=wavelengths,
        noise_cfg=noise_cfg, smooth_cfg=smooth_cfg, mismatch_order=mismatch_order,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return ds, loader, None
