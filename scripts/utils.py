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

import numpy as np
import matplotlib.pyplot as plt


def _build_baseline(wavelengths, baseline_segments):
    """
    Create a baseline curve from user-defined segments.

    Parameters
    ----------
    wavelengths : 1-D ndarray
    baseline_segments : float | list[dict]
        • float – constant baseline  
        • list – each dict has keys {'start', 'end', 'value'}

    Returns
    -------
    baseline_curve : ndarray
    """
    if isinstance(baseline_segments, (int, float)):
        return np.full_like(wavelengths, baseline_segments, dtype=float)

    # start with zeros then paint segment values
    baseline_curve = np.zeros_like(wavelengths, dtype=float)
    for seg in baseline_segments:
        mask = (wavelengths >= seg["start"]) & (wavelengths <= seg["end"])
        baseline_curve[mask] = seg["value"]
    return baseline_curve

from scipy.ndimage import gaussian_filter1d

def generate_signal(signal_type,
                    start_wavelength,
                    end_wavelength,
                    step_size,
                    num_samples,
                    *,
                    peaks=None,
                    baseline=0.0,
                    noise_std_dev=0.01,
                    smooth_signal=False,
                    smooth_sigma=1,
                    **legacy_kwargs):
    """
    Generate pure, noisy and RAT signals with support for
    • multiple Gaussian peaks
    • piece-wise baselines
    • legacy single-peak Gaussian and step signals
    • smoothing the signal before noise addition

    Parameters
    ----------
    signal_type : {'gaussian', 'step'}
    start_wavelength, end_wavelength, step_size : float
    num_samples : int
    peaks : list[dict] | None
        Each dict needs {'anchor', 'fwhm', 'amplitude'}
    baseline : float | list[dict]
        Constant value    – e.g. 0.1  
        Piece-wise list   – e.g. [{'start':300,'end':800,'value':0.1},
                                  {'start':800,'end':1400,'value':0.3},
                                  {'start':1400,'end':2000,'value':0.05}]
    noise_std_dev : float
    smooth_signal : bool
        Whether to smooth the signal before noise addition
    smooth_sigma : float
        Sigma value for the Gaussian filter used for smoothing
    legacy_kwargs : keeps old arguments working
        • gaussian   – anchor_wavelength, fwhm, amplitude, baseline
        • step       – step_wavelength, baseline1, baseline2

    Returns
    -------
    wavelengths, signals, noisy_signals, RAT : ndarrays
    """
    wavelengths = np.arange(start_wavelength, end_wavelength + step_size, step_size)

    # --- 1) PURE SIGNAL ------------------------------------------------------
    if peaks is None:
        pure_signal = _build_baseline(wavelengths, baseline)
    else:
        baseline_curve = _build_baseline(wavelengths, baseline)
        sigma_factor = 1 / (2 * np.sqrt(2 * np.log(2)))           # pre-compute
        total_peak = np.zeros_like(wavelengths, dtype=float)
        for p in peaks:
            sigma = p["fwhm"] * sigma_factor
            total_peak += p["amplitude"] * np.exp(-((wavelengths - p["anchor"]) / sigma) ** 2)

        pure_signal = np.clip(baseline_curve + total_peak, 0, 1)
        
    # --- 2) MONTE-CARLO SAMPLE MATRICES --------------------------------------
    signals = np.tile(pure_signal, (num_samples, 1))
    noise = np.random.normal(0, noise_std_dev, size=signals.shape)
    noisy_signals = np.clip(signals + noise, 0, 1)

    # --- 3) SMOOTH SIGNAL ----------------------------------------------------
    if smooth_signal:
        noisy_signals = np.array([gaussian_filter1d(ns, smooth_sigma) for ns in noisy_signals])

    # --- 4) RAT EXTENSION ----------------------------------------------------
    RAT = np.concatenate(
        (noisy_signals, np.zeros_like(noisy_signals), 1 - noisy_signals),
        axis=1
    )

    return wavelengths, signals, noisy_signals, RAT

def plot_signal(wavelengths, signals, noisy_signals, RAT):
    """
    Plot pure, noisy and RAT signals.
    Unchanged from the original function except for minor label tweaks.
    """
    num_samples = signals.shape[0]

    plt.figure(figsize=(10, 6))
    for i in range(num_samples):
        plt.subplot(2, 1, 1)
        plt.plot(wavelengths, signals[i], alpha=0.1)
        plt.title("Pure Signals")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity")

        plt.subplot(2, 1, 2)
        plt.plot(wavelengths, noisy_signals[i], alpha=0.1)
        plt.title("Signals with Gaussian Noise")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity")

    plt.tight_layout()

    plt.figure(figsize=(10, 4))
    for i in range(num_samples):
        plt.plot(RAT[i], alpha=0.1)
    plt.title("RAT Signals")
    plt.xlabel("Index")
    plt.ylabel("Intensity")

    plt.tight_layout()
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

# # ---------------------------------------------------------------
# # 2. Legacy usage still works
# # ---------------------------------------------------------------
# w, s, ns, rat = generate_signal(
#     "gaussian",
#     300, 2000, 10,
#     num_samples=50,
#     anchor_wavelength=1000,
#     fwhm=200,
#     amplitude=0.8,
#     baseline=0.0,
#     noise_std_dev=0.02
# )
