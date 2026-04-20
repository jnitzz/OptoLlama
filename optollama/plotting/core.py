from __future__ import annotations

import os

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

import optollama.utils


@dataclass
class PlotBundle:
    wavelengths: Optional[np.ndarray] = None
    roi_min: Optional[float] = None
    roi_max: Optional[float] = None
    mae_grid: Optional[np.ndarray] = None
    ids_grid: Optional[np.ndarray] = None
    pred_spectra_grid: Optional[np.ndarray] = None
    mae_traj_grid: Optional[np.ndarray] = None


def _has_value(value: Any) -> bool:
    """Return whether an optional plotting payload is populated."""
    if value is None:
        return False
    if isinstance(value, (list, tuple, dict, str, bytes)):
        return len(value) > 0
    if isinstance(value, np.ndarray):
        return value.size > 0
    if torch.is_tensor(value):
        return value.numel() > 0
    return True


def _to_numpy(value: Any) -> np.ndarray:
    """Convert tensors and array-like values to a NumPy array."""
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def save_plot_bundle(
    path: str,
    output: dict[str, Any],
    wavelengths: Any = None,
    roi_min: Optional[float] = None,
    roi_max: Optional[float] = None,
) -> None:
    """
    Save plotting-centric inference outputs as a compressed NumPy bundle.

    Args
    ----
    path : str
        Output ``.npz`` path.
    output : dict[str, Any]
        Dictionary returned by ``model_prediction`` / inference.
    wavelengths : array-like, optional
        Wavelength grid of shape ``[W]``.
    roi_min : float, optional
        Lower ROI wavelength bound.
    roi_max : float, optional
        Upper ROI wavelength bound.
    """
    arrays: dict[str, np.ndarray] = {}

    if _has_value(wavelengths):
        arrays["wavelengths"] = _to_numpy(wavelengths).astype(np.float32)
    if roi_min is not None:
        arrays["roi_min"] = np.asarray([roi_min], dtype=np.float32)
    if roi_max is not None:
        arrays["roi_max"] = np.asarray([roi_max], dtype=np.float32)

    for key in ("mae_grid", "ids_grid", "pred_spectra_grid", "mae_traj_grid"):
        value = output.get(key)
        if _has_value(value):
            arrays[key] = _to_numpy(value)

    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez_compressed(path, **arrays)


def load_plot_bundle(path: str) -> PlotBundle:
    """
    Load a compressed plotting bundle produced by :func:`save_plot_bundle`.

    Args
    ----
    path : str
        ``.npz`` bundle path.

    Returns
    -------
    PlotBundle
        Structured access to the saved plotting arrays.
    """
    with np.load(path, allow_pickle=False) as payload:
        return PlotBundle(
            wavelengths=payload["wavelengths"] if "wavelengths" in payload else None,
            roi_min=float(payload["roi_min"][0]) if "roi_min" in payload else None,
            roi_max=float(payload["roi_max"][0]) if "roi_max" in payload else None,
            mae_grid=payload["mae_grid"] if "mae_grid" in payload else None,
            ids_grid=payload["ids_grid"] if "ids_grid" in payload else None,
            pred_spectra_grid=payload["pred_spectra_grid"] if "pred_spectra_grid" in payload else None,
            mae_traj_grid=payload["mae_traj_grid"] if "mae_traj_grid" in payload else None,
        )


def load_results(path: str) -> list[dict[str, Any]]:
    """
    Load the per-sample results JSON produced by inference.

    Args
    ----
    path : str
        JSON path.
    """
    return optollama.utils.load_as_json(path)


def results_target_spectra(results: list[dict[str, Any]]) -> np.ndarray:
    """
    Stack target spectra from inference results into ``[N, 3, W]``.

    Args
    ----
    results : list[dict[str, Any]]
        Per-sample inference results containing ``"rat_target"``.
    """
    spectra = [np.asarray(item["rat_target"], dtype=np.float32) for item in results if "rat_target" in item]
    if not spectra:
        raise ValueError("No 'rat_target' entries were found in the provided results.")
    return np.stack(spectra, axis=0)


def build_pred_tokens_grid(
    ids_grid: np.ndarray,
    idx_to_token: dict[int, str],
    eos: int,
    pad: int,
    msk: int,
) -> list[list[list[str]]]:
    """
    Decode an ``ids_grid`` array into token strings per ``(target, mc)`` cell.

    Args
    ----
    ids_grid : np.ndarray
        Array of shape ``[N, M, S]`` with token ids.
    idx_to_token : dict[int, str]
        Vocabulary mapping from id to token string.
    eos : int
        EOS token id.
    pad : int
        PAD token id.
    msk : int
        MASK token id.
    """
    decoded: list[list[list[str]]] = []
    for row in ids_grid:
        decoded_row: list[list[str]] = []
        for ids in row:
            tokens: list[str] = []
            for token_id in ids.tolist():
                token_id = int(token_id)
                if token_id in (eos, pad, msk):
                    break
                tokens.append(idx_to_token[token_id])
            decoded_row.append(tokens)
        decoded.append(decoded_row)
    return decoded


def select_best_result_index(results: list[dict[str, Any]]) -> int:
    """
    Select the index of the lowest-MAE result when available.

    Falls back to index 0 if MAE is not present.
    """
    mae_pairs = [(float(item["mae"]), i) for i, item in enumerate(results) if "mae" in item]
    if mae_pairs:
        return min(mae_pairs)[1]
    if not results:
        raise ValueError("Cannot select a result from an empty list.")
    return 0

