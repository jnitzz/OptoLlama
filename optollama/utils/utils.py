import json
import os
import tempfile
from typing import Any, Optional

import pandas as pd
import torch
import torch.nn.functional as f

from safetensors.torch import load_file, save_file
from scipy.interpolate import interp1d
from torch.nn.parallel import DistributedDataParallel


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


def unique_length_int_generator(start: int, stop: int, amount: int) -> torch.Tensor:
    """
    Generate a tensor of ``amount`` unique, evenly-spaced integer indices.

    Requires ``-1 < start < stop`` and ``0 < amount <= stop``.

    Args
    ----
    start : int
        The start index to subset from.
    stop : int
        The exclusive upper bound for the subset.
    amount : int
        The number of unique indices to return.

    Returns
    -------
    torch.Tensor
        1-D integer tensor of ``amount`` unique indices in ``[start, stop)``.

    Raises
    ------
    ValueError
        If the constraints ``-1 < start < stop`` or ``0 < amount <= stop``
        are not satisfied.
    """
    if not (-1 < start < stop) or not (0 < amount <= stop):
        raise ValueError(
            f"Invalid arguments: start={start}, stop={stop}, amount={amount}. Require (-1 < start < stop) and (0 < amount <= stop)."
        )
    
    len_unique = -1
    amount = amount - 1
    
    while len_unique < amount:
        amount = amount + 1
        subset_idx = torch.linspace(start, stop - 1, amount, dtype=torch.int).unique()
        len_unique = len(subset_idx)

    return subset_idx


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
        # no optimizer/scheduler/epoch information here
        blob = {"model_state": sd}
        print("Checkpoint from .safetensors")
        return None, blob

    # --- regular PyTorch checkpoint (.pt etc.) ---
    blob = torch.load(path, map_location=map_location, weights_only=False)

    # pick a plausible model state dict key
    sd = None
    if isinstance(blob, dict):
        if "model_state" in blob and isinstance(blob["model_state"], dict):
            sd = blob["model_state"]
        elif "model" in blob and isinstance(blob["model"], dict):
            sd = blob["model"]  # older util format fallback

    if sd is None:
        # last fallback: maybe the file is a raw state_dict
        if isinstance(blob, dict) and any(torch.is_tensor(v) for v in blob.values()):
            sd = blob
        else:
            raise RuntimeError("Unrecognized checkpoint layout; expected dict with 'model_state' (or 'model').")

    # always load into the core module; strip 'module.' if present
    core = model.module if isinstance(model, DistributedDataParallel) else model
    # Backward compat: allow new buffers not present in old checkpoints
    if "allowed_vocab_mask" not in sd and hasattr(core, "allowed_vocab_mask"):
        sd["allowed_vocab_mask"] = core.allowed_vocab_mask.detach().clone()
    sd = _strip_module_prefix(sd)
    core.load_state_dict(sd, strict=strict)

    # restore optimizer / scaler / scheduler if available
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
        # optional but useful:
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
        # safetensors stores a simple dict[str, Tensor]
        save_file(model_state, tmp.name)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_safe_path = tmp.name
    os.replace(tmp_safe_path, safe_path)


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
    filter_value: float = -float("inf"),
) -> torch.Tensor:
    """
    Apply combined top-k and top-p (nucleus) filtering to logits.

    Filters along the last dimension of ``logits`` (shape ``[..., V]``).

    Args
    ----
    logits : torch.Tensor
        Raw logits of shape ``[..., V]``.
    top_k : int
        If > 0, keep only the top-k highest-probability tokens.
    top_p : float
        If > 0, keep the smallest set of tokens whose cumulative probability
        exceeds ``top_p`` (nucleus sampling).
    filter_value : float
        Value assigned to filtered-out positions (default: ``-inf``).

    Returns
    -------
    torch.Tensor
        Filtered logits of the same shape as the input.

    Notes
    -----
    Uses a large negative number for NaNs/Infs to keep kernels stable.
    If both ``top_k`` and ``top_p`` are disabled, returns logits unchanged.
    """
    logits = torch.nan_to_num(logits, neginf=-1e9, posinf=1e9)

    v = logits.size(-1)

    if top_k and top_k > 0:
        k = min(int(top_k), v)
        kth = torch.topk(logits, k, dim=-1).values[..., -1].unsqueeze(-1)
        logits = torch.where(logits < kth, torch.full_like(logits, filter_value), logits)

    if top_p and top_p > 0.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)

        remove = cumprobs > float(top_p)
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False

        mask = torch.zeros_like(remove, dtype=torch.bool).scatter(-1, sorted_idx, remove)
        logits = logits.masked_fill(mask, filter_value)

    return logits


def boxcar_kernel(win: int, device: str) -> torch.Tensor:
    """
    Create a normalized boxcar (uniform) smoothing kernel.

    Args
    ----
    win : int
        Kernel window size. Rounded up to the nearest odd integer.
    device : str
        Device on which to allocate the kernel tensor.

    Returns
    -------
    torch.Tensor
        1-D float32 tensor of length ``win`` with all values equal to
        ``1 / win``.
    """
    win = max(1, int(win))
    if win % 2 == 0:
        win += 1
    k = torch.ones(win, device=device, dtype=torch.float32) / float(win)
    return k


def gauss_kernel(win: int, sigma: float, device: str) -> torch.Tensor:
    """
    Create a normalized Gaussian smoothing kernel.

    Args
    ----
    win : int
        Kernel window size. Rounded up to the nearest odd integer.
    sigma : float
        Standard deviation of the Gaussian.
    device : str
        Device on which to allocate the kernel tensor.

    Returns
    -------
    torch.Tensor
        1-D float32 tensor of length ``win`` containing the normalized
        Gaussian weights.
    """
    win = max(1, int(win))
    if win % 2 == 0:
        win += 1
    r = (win - 1) // 2
    xs = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (xs / max(1e-6, float(sigma))) ** 2)
    k = k / k.sum().clamp_min(1e-6)
    return k


def smooth_1d(x_3w: torch.Tensor, method: str, win: int, sigma: float) -> torch.Tensor:
    """
    Apply depthwise 1-D smoothing along the wavelength axis.

    Args
    ----
    x_3w : torch.Tensor
        Input tensor of shape ``[3, W]`` or ``[B, 3, W]``.
    method : str
        Smoothing method: ``"gaussian"`` or ``"boxcar"``.
    win : int
        Kernel window size (rounded up to nearest odd integer).
    sigma : float
        Standard deviation used when ``method="gaussian"``.

    Returns
    -------
    torch.Tensor
        Smoothed tensor of the same shape as ``x_3w``.
    """
    orig_dim = x_3w.dim()
    if orig_dim == 2:
        x_3w = x_3w.unsqueeze(0)  # [1,3,W]
    _, c, w = x_3w.shape
    device = x_3w.device
    if method.lower() == "gaussian":
        k = gauss_kernel(win, sigma, device)
    else:
        k = boxcar_kernel(win, device)
    weight = k.view(1, 1, -1).repeat(c, 1, 1)  # [C,1,K]
    x = f.pad(x_3w, (k.numel() // 2, k.numel() // 2), mode="reflect")
    x = f.conv1d(x, weight, groups=c)
    x = x[:, :, :w]
    return x.squeeze(0) if orig_dim == 2 else x


def parse_order(order_str: str) -> tuple[int, ...]:
    """
    Parse a channel priority string into a tuple of channel indices.

    Args
    ----
    order_str : str
        Priority string such as ``"R>A>T"`` specifying the order in which
        channels are used to fill or crop residual energy.

    Returns
    -------
    tuple[int, ...]
        A 3-tuple of channel indices (0=R, 1=A, 2=T) in the specified
        priority order.
    """
    order_str = (order_str or "R>A>T").upper()
    mapping = {"R": 0, "A": 1, "T": 2}
    seq = [mapping[c.strip()] for c in order_str.split(">") if c.strip() in mapping]
    rest = [i for i in (0, 1, 2) if i not in seq]
    seq.extend(rest)
    return tuple(seq[:3])


def wl_mask(wavelengths: Any, wl_min: float, wl_max: float, device: str) -> Optional[torch.Tensor]:
    """
    Build a boolean mask selecting wavelengths within a given range.

    Args
    ----
    wavelengths : array-like or None
        Wavelength values (nm). If ``None``, returns ``None``.
    wl_min : float
        Lower bound of the wavelength range (inclusive).
    wl_max : float
        Upper bound of the wavelength range (inclusive).
    device : str
        Device on which to allocate the mask tensor.

    Returns
    -------
    torch.Tensor or None
        Boolean tensor of shape ``[W]`` where ``True`` indicates wavelengths
        within ``[wl_min, wl_max]``, or ``None`` if ``wavelengths`` is
        ``None``.
    """
    if wavelengths is None:
        return None
    wl = torch.as_tensor(wavelengths, dtype=torch.float32, device=device)
    return (wl >= float(wl_min)) & (wl <= float(wl_max))


def redistribute_mismatch(tar_spec: torch.Tensor, order: str, target_sum: float = 1.0) -> torch.Tensor:
    """
    Enforce a per-wavelength channel sum by redistributing residual energy.

    Given a spectrum tensor with values in ``[0, 1]``, adjusts channel values
    so that ``R + A + T ≈ target_sum`` at every wavelength, distributing any
    deficit or excess according to the channel priority order.

    Args
    ----
    tar_spec : torch.Tensor
        Spectrum tensor of shape ``[3, W]`` or ``[B, 3, W]`` with values in
        ``[0, 1]``.
    order : str
        Channel priority string (e.g. ``"R>A>T"``) controlling which channel
        absorbs residual energy first.
    target_sum : float
        Desired per-wavelength sum of all channels (default: ``1.0``).

    Returns
    -------
    torch.Tensor
        Adjusted spectrum tensor of the same shape, clamped to ``[0, 1]``.
    """
    orig_dim = tar_spec.dim()
    if orig_dim == 2:
        tar_spec = tar_spec.unsqueeze(0)  # [1,3,W]
    pri = parse_order(order)
    total = tar_spec.sum(dim=1, keepdim=True)  # [B,1,W]
    res = float(target_sum) - total  # +: add, -: remove

    for idx in pri:
        ch = tar_spec[:, idx : idx + 1, :]
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
        tar_spec[:, idx : idx + 1, :] = ch

    tar_spec = tar_spec.clamp_(0.0, 1.0)
    return tar_spec.squeeze(0) if orig_dim == 2 else tar_spec


def apply_noise(tar_spec: torch.Tensor, noise_cfg: Optional[dict[str, Any]], wavelengths: Any) -> torch.Tensor:
    """
    Apply additive Gaussian noise to a spectrum tensor.

    Args
    ----
    tar_spec : torch.Tensor
        Spectrum tensor of shape ``[3, W]`` or ``[B, 3, W]``.
    noise_cfg : dict[str, Any] or None
        Noise configuration dictionary. Expected keys:

        - ``"enabled"`` (bool): whether to apply noise.
        - ``"sigma_abs"`` (float): absolute noise standard deviation.
        - ``"sigma_rel"`` (float): relative noise standard deviation.
        - ``"per_channel"`` (list[float]): per-channel scale factors.
        - ``"wl_min"`` / ``"wl_max"`` (float, optional): wavelength range
          to restrict noise to.
        - ``"clip_0_1"`` (bool): whether to clamp output to ``[0, 1]``.
    wavelengths : array-like or None
        Wavelength values used to build the optional wavelength mask.

    Returns
    -------
    torch.Tensor
        Noised spectrum tensor of the same shape as ``tar_spec``.
    """
    if not noise_cfg or not noise_cfg.get("enabled", False):
        return tar_spec
    orig_dim = tar_spec.dim()
    if orig_dim == 2:
        tar_spec = tar_spec.unsqueeze(0)  # [1,3,W]
    b, c, w = tar_spec.shape
    device = tar_spec.device

    sigma_abs = float(noise_cfg.get("sigma_abs", 0.0))
    sigma_rel = float(noise_cfg.get("sigma_rel", 0.0))
    per_ch = noise_cfg.get("per_channel", [1.0, 1.0, 1.0])
    per_ch = torch.tensor(per_ch, dtype=torch.float32, device=device).view(1, c, 1)

    wl_min = noise_cfg.get("wl_min", None)
    wl_max = noise_cfg.get("wl_max", None)
    mask = None
    if wl_min is not None and wl_max is not None:
        mask = wl_mask(wavelengths, wl_min, wl_max, device) if wavelengths is not None else None

    eps = torch.randn_like(tar_spec) * (sigma_abs + sigma_rel * tar_spec)
    eps = eps * per_ch

    if mask is not None:
        m = mask.view(1, 1, w)
        tar_spec = torch.where(m, tar_spec + eps, tar_spec)
    else:
        tar_spec = tar_spec + eps

    if noise_cfg.get("clip_0_1", True):
        tar_spec = tar_spec.clamp_(0.0, 1.0)

    return tar_spec.squeeze(0) if orig_dim == 2 else tar_spec


def apply_smoothing(tar_spec: torch.Tensor, smooth_cfg: Optional[dict[str, Any]]) -> torch.Tensor:
    """
    Apply 1-D smoothing to a spectrum tensor.

    Args
    ----
    tar_spec : torch.Tensor
        Spectrum tensor of shape ``[3, W]`` or ``[B, 3, W]``.
    smooth_cfg : dict[str, Any] or None
        Smoothing configuration dictionary. Expected keys:

        - ``"enabled"`` (bool): whether to apply smoothing.
        - ``"method"`` (str): ``"gaussian"`` or ``"boxcar"``.
        - ``"win"`` (int): kernel window size.
        - ``"sigma"`` (float): Gaussian standard deviation.

    Returns
    -------
    torch.Tensor
        Smoothed spectrum tensor of the same shape as ``tar_spec``.
    """
    if not smooth_cfg or not smooth_cfg.get("enabled", False):
        return tar_spec
    method = smooth_cfg.get("method", "gaussian")
    win = int(smooth_cfg.get("win", 5))
    sigma = float(smooth_cfg.get("sigma", 1.0))
    return smooth_1d(tar_spec, method, win, sigma)


def ensure_3w(tar_spec: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """
    Ensure a spectrum tensor has shape ``[..., 3, W]``, transposing if needed.

    Args
    ----
    tar_spec : torch.Tensor
        Spectrum tensor of shape ``[3, W]``, ``[W, 3]``, ``[B, 3, W]``, or
        ``[B, W, 3]``.

    Returns
    -------
    tuple[torch.Tensor, bool]
        A 2-tuple of ``(tensor, was_transposed)`` where ``tensor`` has shape
        ``[..., 3, W]`` and ``was_transposed`` is ``True`` if the input was
        transposed.

    Raises
    ------
    ValueError
        If the input shape is not one of the supported formats.
    """
    if tar_spec.dim() == 2:
        if tar_spec.size(0) == 513:
            tar_spec = tar_spec.reshape(3, 171)
        elif tar_spec.size(1) == 513:
            tar_spec = tar_spec.reshape(171, 3)
    if tar_spec.dim() == 2:
        if tar_spec.size(0) == 3:
            return tar_spec, False
        elif tar_spec.size(1) == 3:
            return tar_spec.permute(1, 0).contiguous(), True
    elif tar_spec.dim() == 3:
        if tar_spec.size(1) == 3:
            return tar_spec, False
        elif tar_spec.size(2) == 3:
            return tar_spec.permute(0, 2, 1).contiguous(), True
    raise ValueError(f"Expected shape [...,3,W] or [...,W,3], got {tuple(tar_spec.shape)}")


def normalize_rat_fill_crop(r: torch.Tensor, a: torch.Tensor, t: torch.Tensor, target: float = 1.0) -> torch.Tensor:
    """
    Normalize R, A, T channels so their per-wavelength sum equals ``target``.

    Uses T as the primary filler and crop source, then A, then R.

    Steps per wavelength:

    1. Clamp R, A, T to be non-negative.
    2. If ``R + A + T < target``: add the deficit to T.
    3. If ``R + A + T > target``: reduce T first; if still over, reduce A;
       then R. Values are never reduced below 0.

    Args
    ----
    r : torch.Tensor
        Reflectance channel, shape ``[W]``.
    a : torch.Tensor
        Absorptance channel, shape ``[W]``.
    t : torch.Tensor
        Transmittance channel, shape ``[W]``.
    target : float
        Desired per-wavelength sum (default: ``1.0``).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Normalized ``(r, a, t)`` float32 tensors, each of shape ``[W]``.
    """
    r = torch.clamp(torch.as_tensor(r), min=0.0)
    a = torch.clamp(torch.as_tensor(a), min=0.0)
    t = torch.clamp(torch.as_tensor(t), min=0.0)

    total = r + a + t
    # case 1: fill deficit into T
    deficit = target - total
    need = deficit > 0
    t[need] += deficit[need]  # safe since deficit>0 only at 'need'

    # case 2: crop excess from T, then A, then R
    excess = (r + a + t) - target
    over = torch.clamp(excess, 0.0)
    has_over = over > 0

    if torch.any(has_over):
        # crop from T
        cut = torch.minimum(t[has_over], over[has_over])
        t[has_over] -= cut
        over[has_over] -= cut

        # crop from A
        still = over > 0
        if torch.any(still):
            cut = torch.minimum(a[still], over[still])
            a[still] -= cut
            over[still] -= cut
            # By construction, over should now be 0

        # crop from R
        still = over > 0
        if torch.any(still):
            cut = torch.minimum(r[still], over[still])
            r[still] -= cut
            over[still] -= cut

    return r.to(torch.float32), a.to(torch.float32), t.to(torch.float32)


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
