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