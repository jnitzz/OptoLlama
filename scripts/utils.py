# -*- coding: utf-8 -*-
"""
Created on Tue May  6 14:21:15 2025

@author: a3536
"""
import torch
import numpy as np
import random

import json
import jsonpickle
from typing import Any

def seed_everything(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_JSONPICKLE(PATH: str, name: str) -> Any:
    with open(f'{PATH}/{name}.json') as f:
        d = json.load(f)
    return jsonpickle.decode(d)

def save_JSONPICKLE(PATH: str, pyobj: Any, name: str) -> None:
    frozen = jsonpickle.encode(pyobj)
    with open(f"{PATH}/{name}.json", 'w') as f:
        json.dump(frozen, f)


from typing import Union, List, Dict, Optional, Tuple, Any
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

def _build_baseline(
    wavelengths: np.ndarray,
    baseline_segments: Union[float, List[Dict[str, float]]]
) -> np.ndarray:
    """
    Create a baseline curve from user-defined segments.
    If baseline_segments is a float, produces a constant baseline.
    If it’s a list of dicts with keys 'start', 'end', 'value', (and now optional 'noise'),
    it uses 'value' for the baseline.
    """
    if isinstance(baseline_segments, (int, float)):
        return np.full_like(wavelengths, baseline_segments, dtype=float)

    baseline_curve = np.zeros_like(wavelengths, dtype=float)
    for seg in baseline_segments:
        mask = (wavelengths >= seg["start"]) & (wavelengths <= seg["end"])
        baseline_curve[mask] = seg["value"]
    return baseline_curve

def _build_noise_std(
    wavelengths: np.ndarray,
    baseline_segments: Union[float, List[Dict[str, float]]]
) -> np.ndarray:
    """
    Create a per-wavelength noise-std curve from the same segments list,
    looking for a 'noise' key in each dict (defaulting to 0.0).
    """
    if isinstance(baseline_segments, (int, float)):
        return np.zeros_like(wavelengths, dtype=float)

    noise_curve = np.zeros_like(wavelengths, dtype=float)
    for seg in baseline_segments:
        mask = (wavelengths >= seg["start"]) & (wavelengths <= seg["end"])
        noise_curve[mask] = seg.get("noise", 0.0)
    return noise_curve

def generate_signal(
    start_wavelength: float,
    end_wavelength: float,
    step_size: float,
    num_samples: int,
    *,
    peaks: Optional[List[Dict[str, float]]] = None,
    baseline: Union[float, List[Dict[str, float]]] = 0.0,
    noise_std_dev: float = 0.01,
    smooth_signal: bool = False,
    smooth_sigma: float = 1.0,
    pure_signal: Optional[np.ndarray] = None,
    **legacy_kwargs: Any
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate pure, noisy and RAT signals with support for
    • multiple Gaussian peaks
    • piece-wise baselines (with per-segment noise via a 'noise' key)
    • optional smoothing before noise addition

    Returns
    -------
    wavelengths : ndarray
    signals    : ndarray  # num_samples × len(wavelengths), pure repeated
    noisy_signals : ndarray
    RAT        : ndarray
    """
    # 1) DEFINE GRID
    wavelengths = np.arange(start_wavelength, end_wavelength + step_size, step_size)

    # 2) PURE SIGNAL
    if pure_signal is not None:
        if len(pure_signal) != len(wavelengths):
            raise ValueError("Provided pure_signal must match the length of the wavelength grid.")
        pure_signal = np.clip(pure_signal, 0.0, 1.0)
        baseline_curve = _build_baseline(wavelengths, baseline)
        pure_signal = np.clip(baseline_curve + pure_signal, 0.0, 1.0)
    elif peaks is None:
        pure_signal = _build_baseline(wavelengths, baseline)
    else:
        baseline_curve = _build_baseline(wavelengths, baseline)
        sigma_factor = 1 / (2 * np.sqrt(2 * np.log(2)))
        total_peak = np.zeros_like(wavelengths, dtype=float)
        for p in peaks:
            sigma = p["fwhm"] * sigma_factor
            total_peak += p["amplitude"] * np.exp(-((wavelengths - p["anchor"]) / sigma) ** 2)
        pure_signal = np.clip(baseline_curve + total_peak, 0.0, 1.0)

    # 3) BUILD NOISE STD CURVE
    per_segment_noise = _build_noise_std(wavelengths, baseline)
    total_noise_std = noise_std_dev + per_segment_noise

    # 4) MONTE-CARLO NOISE ADDITION
    signals = np.tile(pure_signal, (num_samples, 1))
    noise = np.random.normal(loc=0.0, scale=total_noise_std, size=signals.shape)
    noisy_signals = np.clip(signals + noise, 0.0, 1.0)

    # 5) OPTIONAL SMOOTHING
    if smooth_signal:
        noisy_signals = np.array([
            gaussian_filter1d(ns, smooth_sigma)
            for ns in noisy_signals
        ])

    # 6) RAT EXTENSION
    RAT = np.concatenate(
        (noisy_signals,
         np.zeros_like(noisy_signals),
         1.0 - noisy_signals),
        axis=1
    )

    return wavelengths, signals, noisy_signals, RAT
    
from typing import Union, List, Dict, Optional, Tuple, Any
import numpy as np
from scipy.ndimage import gaussian_filter1d


from typing import Union, List, Dict, Optional, Tuple, Any
import numpy as np
from scipy.ndimage import gaussian_filter1d


def generate_signal2(
    start_wavelength: float,
    end_wavelength: float,
    step_size: float,
    num_samples: int,
    *,
    peaks: Optional[List[Dict[str, float]]] = None,
    baseline: Union[float, List[Dict[str, float]]] = 0.0,
    noise_std_dev: float = 0.01,
    smooth_signal: bool = False,
    smooth_sigma: float = 1.0,
    pure_signal: Optional[np.ndarray] = None,      # now always 1-D if provided
    **legacy_kwargs: Any
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate pure, noisy and RAT signals.

    pure_signal (optional, 1-D)
        • len == n_wl            → only R is supplied (legacy mode)
        • len == 3·n_wl          → concatenated R|A|T
        • len == 2·n_wl          → concatenated R|T   (A is inferred)

    All other behaviour is unchanged.
    """
    # ------------------------------------------------------------------ 1) λ-grid
    wl = np.arange(start_wavelength, end_wavelength + step_size, step_size)
    n_wl = len(wl)

    # -------------------------------------------- helpers for baseline & noise std
    def _baseline_curve():
        if isinstance(baseline, (int, float)):
            return np.full(n_wl, baseline, dtype=float)
        out = np.zeros(n_wl, dtype=float)
        for seg in baseline:
            m = (wl >= seg["start"]) & (wl <= seg["end"])
            out[m] = seg["value"]
        return out

    def _noise_std_curve():
        if isinstance(baseline, (int, float)):
            return np.zeros(n_wl, dtype=float)
        out = np.zeros(n_wl, dtype=float)
        for seg in baseline:
            m = (wl >= seg["start"]) & (wl <= seg["end"])
            out[m] = seg.get("noise", 0.0)
        return out + noise_std_dev

    # -------------------------------------------------------- 2) build pure RAT
    if pure_signal is not None:                           # user-supplied curve
        vec = np.asarray(pure_signal, dtype=float).ravel()   # ensure 1-D

        if len(vec) == n_wl:                                  # R only
            R = np.clip(vec, 0.0, 1.0)
            A = np.zeros_like(R)
            T = 1.0 - R

        elif len(vec) == 2 * n_wl:                            # R|T
            R, T = vec[:n_wl], vec[n_wl:]
            A = 1.0 - R - T

        elif len(vec) == 3 * n_wl:                            # R|A|T
            R, A, T = np.split(vec, 3)

        else:
            raise ValueError(
                "pure_signal must have length n_wl, 2·n_wl or 3·n_wl "
                f"(got {len(vec)} for n_wl={n_wl})."
            )

        # clip & renormalise so that R+A+T = 1
        RAT = np.clip(np.stack([R, A, T]), 0.0, 1.0)
        s = RAT.sum(axis=0, keepdims=True)
        s[s == 0.0] = 1.0
        RAT /= s
        R, A, T = RAT

    else:                                                  # build R from peaks
        sigma_fac = 1.0 / (2 * np.sqrt(2 * np.log(2)))
        R = _baseline_curve()
        if peaks:
            for p in peaks:
                sigma = p["fwhm"] * sigma_fac
                R += p["amplitude"] * np.exp(-((wl - p["anchor"]) / sigma) ** 2)
        R = np.clip(R, 0.0, 1.0)
        A = np.zeros_like(R)
        T = 1.0 - R

    # --------------------------------------------- 3) Monte-Carlo noise & smooth
    noise_sigma = _noise_std_curve()

    def _noisy(base):
        y = np.tile(base, (num_samples, 1))
        y += np.random.normal(0.0, noise_sigma, y.shape)
        y = np.clip(y, 0.0, 1.0)
        if smooth_signal:
            y = np.array([gaussian_filter1d(v, smooth_sigma) for v in y])
        return y

    Rn, An, Tn = map(_noisy, (R, A, T))

    # --------------------------------------------- 4) renormalise after noise
    RAT_noisy = np.stack([Rn, An, Tn], axis=1)          # (N, 3, λ)
    s = RAT_noisy.sum(axis=1, keepdims=True)
    s[s == 0.0] = 1.0
    RAT_noisy = np.clip(RAT_noisy / s, 0.0, 1.0)

    # ------------------------------------------------------- 5) prepare outputs
    wavelengths   = wl
    signals       = np.tile(R, (num_samples, 1))               # pure R
    noisy_signals = RAT_noisy[:, 0, :]                         # noisy R
    RAT_out       = RAT_noisy.reshape(num_samples, -1)         # flattened R|A|T
    # RAT_out = np.insert(RAT_out, 0, np.concatenate([R, A, T]), axis=0)
    RAT_out[0] = np.concatenate([R, A, T])

    return wavelengths, signals, noisy_signals, RAT_out

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter1d


def generate_signal3(
    start_wavelength: float,
    end_wavelength: float,
    step_size: float,
    num_samples: int,
    *,
    peaks: Optional[List[Dict[str, float]]] = None,
    baseline: Union[float, List[Dict[str, float]]] = 0.0,
    noise_std_dev: float = 0.01,
    smooth_signal: bool = False,
    smooth_sigma: float = 1.0,
    pure_signal: Optional[np.ndarray] = None,          # legacy - single curve
    pure_signals: Optional[np.ndarray] = None,         # NEW - many curves
    **legacy_kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate wavelength grid, pure R curves, noisy R curves and flattened RAT.
    ──────────────────────────────────────────────────────────────────────────
    pure_signals (preferred, optional)
        2-D array where each row is **one** curve:
            • len == n_wl      → R only       (A = 0, T = 1-R)
            • len == 2·n_wl    → R|T concat   (A = 1-R-T)
            • len == 3·n_wl    → R|A|T concat
        If given, pure_signal must be None.

    pure_signal (legacy, optional)
        Same as above but 1-D → treated as a single-row input.

    For each input curve we draw `num_samples` noisy realisations **and**
    make sure the first sample of every set is the untouched pure curve.
    """
    # ------------------------------------------------------------------ 1) λ-grid
    wl = np.arange(start_wavelength, end_wavelength + step_size, step_size)
    n_wl = len(wl)

    # -------------------------------------------- helpers for baseline & noise
    def _baseline_curve():
        if isinstance(baseline, (int, float)):
            return np.full(n_wl, baseline, dtype=float)
        out = np.zeros(n_wl, dtype=float)
        for seg in baseline:
            m = (wl >= seg["start"]) & (wl <= seg["end"])
            out[m] = seg["value"]
        return out

    def _noise_std_curve():
        if isinstance(baseline, (int, float)):
            return np.full(n_wl, noise_std_dev, dtype=float)
        out = np.zeros(n_wl, dtype=float)
        for seg in baseline:
            m = (wl >= seg["start"]) & (wl <= seg["end"])
            out[m] = seg.get("noise", 0.0)
        return out + noise_std_dev

    noise_sigma = _noise_std_curve()  # (λ,)

    # ------------------------------------------------ 2) collect pure curves
    if (pure_signals is not None) and (pure_signal is not None):
        raise ValueError("Use either `pure_signals` or `pure_signal`, not both.")

    if pure_signals is not None:                              # NEW: many curves
        arr = np.asarray(pure_signals, dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]                                # make 2-D

    elif pure_signal is not None:                             # legacy: one curve
        arr = np.asarray(pure_signal, dtype=float).ravel()[None, :]

    else:                                                     # build from peaks
        sigma_fac = 1.0 / (2 * np.sqrt(2 * np.log(2)))
        R0 = _baseline_curve()
        if peaks:
            for p in peaks:
                sigma = p["fwhm"] * sigma_fac
                R0 += p["amplitude"] * np.exp(-((wl - p["anchor"]) / sigma) ** 2)
        R0 = np.clip(R0, 0.0, 1.0)
        arr = R0[None, :]                                     # single curve

    n_signals = arr.shape[0]

    # ------------ parse every supplied vector into normalised R, A, T triplets
    def _parse(vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(vec) == n_wl:                      # R only
            R = np.clip(vec, 0.0, 1.0)
            A = np.zeros_like(R)
            T = 1.0 - R
        elif len(vec) == 2 * n_wl:                # R|T
            R, T = vec[:n_wl], vec[n_wl:]
            A = 1.0 - R - T
        elif len(vec) == 3 * n_wl:                # R|A|T
            R, A, T = np.split(vec, 3)
        else:
            raise ValueError(
                "Pure curve must have length n_wl, 2·n_wl or 3·n_wl "
                f"(got {len(vec)}; n_wl={n_wl})."
            )

        RAT = np.clip(np.stack([R, A, T]), 0.0, 1.0)
        s = RAT.sum(axis=0, keepdims=True)
        s[s == 0.0] = 1.0
        RAT /= s
        return RAT                              # (3, λ)

    RAT_pure = np.stack([_parse(vec) for vec in arr])          # (n_signals, 3, λ)
    Rp, Ap, Tp = RAT_pure[:, 0, :], RAT_pure[:, 1, :], RAT_pure[:, 2, :]

    # --------------------------------------------- 3) Monte-Carlo noise + smooth
    def _noisy(base: np.ndarray) -> np.ndarray:
        """
        base: (n_signals, λ)  →  (n_signals, num_samples, λ)
        First slice [:,0,:] is the untouched base curve.
        """
        rep = np.repeat(base[:, None, :], num_samples, axis=1)
        jitter = np.random.normal(0.0, noise_sigma, rep.shape)
        rep += jitter
        rep = np.clip(rep, 0.0, 1.0)
        if smooth_signal:
            rep = np.array(
                [[gaussian_filter1d(v, smooth_sigma) for v in sig_set] for sig_set in rep]
            )
        rep[:, 0, :] = base                               # ensure pure curve kept
        return rep                                        # (n_signals, N, λ)

    Rn = _noisy(Rp)
    An = _noisy(Ap)
    Tn = _noisy(Tp)

    # --------------------------------------------- 4) renormalise after noise
    RAT_noisy = np.stack([Rn, An, Tn], axis=2)           # (n_signals, N, 3, λ)
    s = RAT_noisy.sum(axis=2, keepdims=True)
    s[s == 0.0] = 1.0
    RAT_noisy = np.clip(RAT_noisy / s, 0.0, 1.0)

    # ------------------------------------------------------- 5) prepare outputs
    wavelengths   = wl
    signals       = Rp                                     # (n_signals, λ)

    # flatten the sample axis so downstream code sees one long batch
    k, N = n_signals, num_samples
    noisy_signals = RAT_noisy[:, :, 0, :].reshape(k * N, n_wl)
    RAT_out       = RAT_noisy.reshape(k * N, 3 * n_wl)

    return wavelengths, signals, noisy_signals, RAT_out


def plot_signal(wavelengths, signals, noisy_signals, RAT):
    """
    Plot pure, noisy and RAT signals.
    Unchanged from the original function except for minor label tweaks.
    """
    num_samples = signals.shape[0]

    plt.figure(figsize=(10, 8))
    for i in range(num_samples):
        plt.subplot(3, 1, 1)
        plt.plot(wavelengths, signals[i], alpha=0.1)
        plt.title("Pure Signals")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity")

        plt.subplot(3, 1, 2)
        plt.plot(wavelengths, noisy_signals[i], alpha=0.1)
        plt.title("Signals with Gaussian Noise")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity")
        
        plt.subplot(3, 1, 3)
        plt.plot(RAT[i], alpha=0.1)
        plt.title("RAT Signals")
        plt.xlabel("Index")
        plt.ylabel("Intensity")

    plt.tight_layout()

    # plt.figure(figsize=(10, 4))
    # for i in range(num_samples):
    #     plt.plot(RAT[i], alpha=0.1)
    # plt.title("RAT Signals")
    # plt.xlabel("Index")
    # plt.ylabel("Intensity")

    # plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------
# 1. Composite spectrum with three peaks and a two-level baseline
# ---------------------------------------------------------------
# wl0, wl1, step = 300, 2000, 10
# peaks = [
#     {"anchor": 500,  "fwhm": 80,  "amplitude": 0.6},
#     {"anchor": 1000, "fwhm": 150, "amplitude": 0.8},
#     {"anchor": 1500, "fwhm": 120, "amplitude": 0.4},
# ]
# baseline_segments = [
#     {"start": 300,  "end": 1200, "value": 0.05},
#     {"start": 1200, "end": 2000, "value": 0.15},
# ]

# w, s, ns, rat = generate_signal(
#     "gaussian",
#     wl0, wl1, step,
#     num_samples=100,
#     peaks=peaks,
#     baseline=baseline_segments,
#     noise_std_dev=0.01
# )
# plot_signal(w, s, ns, rat)