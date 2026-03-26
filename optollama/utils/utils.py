import csv
import json
import os
import tempfile

from typing import Any, Optional

import pandas as pd
import torch

from safetensors.torch import load_file, save_file
from scipy.interpolate import interp1d
from torch.nn.parallel import DistributedDataParallel

import optollama.data.spectra


def load_as_json(path: str) -> Any:
    """
    Load and return a Python object from a JSON file.

    Args
    ----
    path : str
        Path to the JSON file.

    Returns
    -------
    Any
        The deserialized Python object (typically a ``dict`` or ``list``).
    """
    with open(rf"{path}", "r", encoding="utf-8") as f:
        return json.load(f)


def save_as_json(path: str, pyobj: Any) -> None:
    """
    Save a Python object (e.g., list of strings, dict, etc.) as a JSON file.

    Args
    ----
    path : str
        Destination file path.
    pyobj : Any
        The Python object to serialize.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pyobj, f, indent=2, ensure_ascii=False)


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Strip the ``"module."`` prefix from all keys in a state dict.

    This is needed when loading weights saved from a
    ``DistributedDataParallel``-wrapped model into a plain model.

    Args
    ----
    state_dict : dict[str, torch.Tensor]
        Model state dict, potentially with ``"module."``-prefixed keys.

    Returns
    -------
    dict[str, torch.Tensor]
        State dict with the ``"module."`` prefix removed from all keys.
    """
    if not state_dict:
        return state_dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


@torch.no_grad()
def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
    strict: bool = True,
    scaler: Any = None,
    scheduler: Any = None,
) -> tuple[Optional[int], dict[str, Any]]:
    """
    Load an optollama-style checkpoint, robust to DDP/non-DDP differences.

    Supports full ``.pt`` checkpoints saved by :func:`save_checkpoint` as
    well as weights-only ``.safetensors`` files.

    Args
    ----
    path : str
        Path to the checkpoint file (``.pt`` or ``.safetensors``).
    model : torch.nn.Module
        Model into which weights are loaded (DDP-wrapped or plain).
    optimizer : torch.optim.Optimizer, optional
        Optimizer to restore state into, if present in the checkpoint.
    map_location : str
        Device string passed to ``torch.load`` (default: ``"cpu"``).
    strict : bool
        Whether to enforce strict key matching when loading state dicts.
    scaler : optional
        GradScaler instance to restore state into.
    scheduler : optional
        LR scheduler to restore state into.

    Returns
    -------
    tuple[int | None, dict[str, Any]]
        A 2-tuple of ``(start_epoch, blob)`` where ``start_epoch`` is the
        next epoch to train from (``None`` for safetensors-only checkpoints)
        and ``blob`` is the full checkpoint dictionary.
    """
    # --- weights-only safetensors path ---
    if path.lower().endswith(".safetensors"):
        sd = load_file(path)  # dict[str, Tensor] on CPU
        core = model.module if isinstance(model, DistributedDataParallel) else model
        sd = _strip_module_prefix(sd)
        core.load_state_dict(sd, strict=strict)
        blob = {"model_state": sd}
        print("Checkpoint from .safetensors")
        return None, blob

    # --- regular PyTorch checkpoint (.pt etc.) ---
    blob = torch.load(path, map_location=map_location, weights_only=False)
    sd = None

    if isinstance(blob, dict):
        if "model_state" in blob and isinstance(blob["model_state"], dict):
            sd = blob["model_state"]
        elif "model" in blob and isinstance(blob["model"], dict):
            sd = blob["model"]  # older util format fallback

    if sd is None:
        if isinstance(blob, dict) and any(torch.is_tensor(v) for v in blob.values()):
            sd = blob
        else:
            raise RuntimeError("Unrecognized checkpoint layout; expected dict with 'model_state' (or 'model').")

    core = model.module if isinstance(model, DistributedDataParallel) else model
    if "allowed_vocab_mask" not in sd and hasattr(core, "allowed_vocab_mask"):
        sd["allowed_vocab_mask"] = core.allowed_vocab_mask.detach().clone()
    sd = _strip_module_prefix(sd)
    core.load_state_dict(sd, strict=strict)

    if optimizer is not None and isinstance(blob.get("optimizer_state", None), dict):
        optimizer.load_state_dict(blob["optimizer_state"])
    if scaler is not None and isinstance(blob.get("scaler_state", None), dict):
        scaler.load_state_dict(blob["scaler_state"])
    if scheduler is not None and isinstance(blob.get("scheduler_state", None), dict):
        scheduler.load_state_dict(blob["scheduler_state"])

    start_epoch = None
    if isinstance(blob.get("epoch", None), (int, float)):
        start_epoch = int(blob["epoch"]) + 1  # resume on next epoch

    return start_epoch, blob


@torch.no_grad()
def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    train_losses: Optional[torch.Tensor] = None,
    train_acc: Optional[torch.Tensor] = None,
    test_acc: Optional[torch.Tensor] = None,
    test_mae: Optional[torch.Tensor] = None,
    scaler: Any = None,
    scheduler: Any = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """
    Save a checkpoint as both a ``.pt`` file and a ``.safetensors`` file.

    Handles ``DistributedDataParallel``-wrapped models by unwrapping the
    inner module before saving. Both files are written atomically.

    Args
    ----
    path : str
        Destination path for the ``.pt`` checkpoint file. A sibling
        ``.safetensors`` file is written at the same base path.
    model : torch.nn.Module
        The model to checkpoint (DDP-wrapped or plain).
    optimizer : torch.optim.Optimizer, optional
        Optimizer whose state should be saved.
    epoch : int, optional
        Current training epoch index.
    train_losses : torch.Tensor, optional
        Per-epoch training loss history.
    train_acc : torch.Tensor, optional
        Per-epoch training accuracy history.
    test_acc : torch.Tensor, optional
        Per-epoch validation accuracy history.
    test_mae : torch.Tensor, optional
        Per-epoch validation MAE history.
    scaler : optional
        GradScaler instance (AMP) whose state should be saved.
    scheduler : optional
        LR scheduler whose state should be saved.
    extra : dict[str, Any], optional
        Additional key-value pairs to store in the checkpoint.
    """
    core = model.module if isinstance(model, DistributedDataParallel) else model
    model_state = _strip_module_prefix(core.state_dict())

    state = {
        "epoch": int(epoch) if epoch is not None else None,
        "model_state": model_state,
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "train_losses": train_losses,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "test_mae": test_mae,
        "scaler_state": (scaler.state_dict() if scaler else None),
        "scheduler_state": (scheduler.state_dict() if scheduler else None),
        "format_version": 1,
    }
    if extra:
        state["extra"] = extra

    ckpt_dir = os.path.dirname(path)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    # ---- 1) atomic write of the full PyTorch checkpoint (.pt) ----
    with tempfile.NamedTemporaryFile(dir=ckpt_dir or None, delete=False) as tmp:
        torch.save(state, tmp.name)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name
    os.replace(tmp_path, path)

    # ---- 2) atomic write of weights-only .safetensors next to it ----
    base, _ = os.path.splitext(path)
    safe_path = base + ".safetensors"

    with tempfile.NamedTemporaryFile(dir=ckpt_dir or None, delete=False) as tmp:
        save_file(model_state, tmp.name)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_safe_path = tmp.name
    os.replace(tmp_safe_path, safe_path)


def load_spectra(path: str, cfg: dict) -> torch.Tensor:
    """
    Load a RAT spectrum from a JSON or CSV file with post-processing from cfg.

    Reads wavelengths, noise, smoothing, and mismatch-fill settings directly
    from the configuration object. Always returns a ``[3, W]`` float32 tensor.

    Args
    ----
    path : str
        Path to the JSON or CSV file containing the spectrum.
    cfg : dict
        Configuration object providing:

        - ``cfg["WAVELENGTHS"]`` — 1-D wavelength grid tensor.
        - ``cfg["NOISE"]`` — noise config dict (see :func:`apply_noise`).
        - ``cfg["SMOOTH"]`` — smoothing config dict (see :func:`apply_smoothing`).
        - ``cfg["MISMATCH_FILL_ORDER"]`` — channel priority string, e.g. ``"R>T>A"``.

    Returns
    -------
    torch.Tensor
        Spectrum tensor of shape ``[3, W]`` with values in ``[0, 1]``.
    """
    if path.endswith(".json"):
        data = load_as_json(path)
        x = torch.as_tensor(data["spectra"], dtype=torch.float32)

    else:
        with open(path, "r") as f:
            rows = [[float(v) for v in row] for row in csv.reader(f) if row]
        x = torch.tensor(rows, dtype=torch.float32).T  # [3, W]

    x, _ = optollama.data.spectra.ensure_3w(x)

    return x


def load_materials(path_materials: str, wavelengths: torch.Tensor) -> dict:
    """
    Load complex refractive indices for all materials in a folder.

    Reads CSV files and interpolates the nk data onto the given wavelength
    grid.

    Args
    ----
    path_materials : str
        Directory containing CSV files with columns ``"nm"``, ``"n"``,
        ``"k"``.
    wavelengths : torch.Tensor
        1-D tensor of wavelengths (nm) at which nk is required, shape
        ``[W]``.

    Returns
    -------
    dict
        Mapping from material name (file stem) to complex nk values
        interpolated onto ``wavelengths``. Values are array-like of shape
        ``[W]``.
    """
    material_files = [item[:-4] for item in sorted(os.listdir(path_materials)) if item.lower().endswith(".csv")]

    nk_dict = {}

    for mat in material_files:
        try:
            data_temp = pd.read_csv(os.path.join(path_materials, f"{mat}.csv"))
            wavelength_nm = data_temp["nm"].to_numpy()
            n_vals = data_temp["n"].to_numpy()
            k_vals = data_temp["k"].to_numpy()
        except Exception:
            print("Error: NK file does not have the right format: .csv with columns 'nm', 'n', 'k', comma (,) separated.")
            continue

        n_fn = interp1d(
            wavelength_nm,
            n_vals,
            axis=0,
            bounds_error=False,
            kind="linear",
            fill_value=(n_vals[0], n_vals[-1]),
        )
        k_fn = interp1d(
            wavelength_nm,
            k_vals,
            axis=0,
            bounds_error=False,
            kind="linear",
            fill_value=(k_vals[0], k_vals[-1]),
        )

        nk_dict[mat] = n_fn(wavelengths.cpu().numpy()) + 1j * k_fn(wavelengths.cpu().numpy())

    return nk_dict
