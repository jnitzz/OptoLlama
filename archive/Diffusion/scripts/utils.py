# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

import json
import jsonpickle
from typing import Any, List, Union, Dict, Optional, Tuple

import tempfile
import datetime

def set_all_seeds(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def set_torch_options() -> None:
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_override_can_fuse_on_cpu(True)
    torch._C._jit_override_can_fuse_on_gpu(True)
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_printoptions(threshold=int(1e10))
    

import torch.distributed as dist
import socket


def _pick_free_port() -> int:
    """Pick an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def init_distributed_old() -> Tuple[str, int, int, int]:
    """
    Initialize (or skip) torch.distributed in a cross-platform way.

    Behavior:
      - If WORLD_SIZE > 1 (torchrun / SLURM / multi-GPU), initialize a process group:
          * backend = 'nccl' if CUDA + NCCL available (typically Linux), else 'gloo'
          * rendezvous = env://
      - If WORLD_SIZE == 1 (most local runs), DO NOT initialize a process group.
        Just select the best device and return (device, 0, 0, 1).

    Returns:
      device_str:  e.g., 'cuda:0' or 'cpu'
      local_rank:  int
      rank:        int
      world_size:  int
    """
    # Infer ranks/world from environment (torchrun sets these; SLURM may too)
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    local_rank_env = os.environ.get("LOCAL_RANK")
    local_rank = int(local_rank_env) if local_rank_env is not None else int(os.environ.get("SLURM_LOCALID", "0"))

    # If single process, skip process-group setup entirely.
    if world_size <= 1:
        # Never disable CUDA on local runs.
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        return device_str, 0, 0, 1

    # Multi-process case: initialize a process group
    # Ensure rendezvous env vars exist
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_pick_free_port()))

    # Choose backend: NCCL if available (Linux + CUDA), otherwise Gloo (Windows/macOS or CPU)
    use_cuda = torch.cuda.is_available()
    try:
        nccl_ok = dist.is_nccl_available()
    except Exception:
        nccl_ok = False
    backend = "nccl" if (use_cuda and nccl_ok) else "gloo"

    # Initialize the process group
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=datetime.timedelta(hours=1),
        world_size=world_size,
        rank=rank,
    )

    # Device selection (bind each rank to its local GPU when CUDA is available)
    if use_cuda:
        # If torchrun provided LOCAL_RANK, use it; otherwise map by global rank
        if local_rank_env is None:
            # Fallback mapping: assume one process per GPU per node
            num_gpus = torch.cuda.device_count()
            local_rank = rank % max(1, num_gpus)
        torch.cuda.set_device(local_rank)
        device_str = f"cuda:{local_rank}"
    else:
        device_str = "cpu"

    return device_str, local_rank, rank, world_size


def init_distributed() -> tuple[str, int, int, int]:
    """
    Initialize torch.distributed and return:
      - device (e.g. 'cuda:0' or 'cpu'),
      - local_rank,
      - global rank,
      - world_size

    Logic:
      1) If we've already got a process group, just re-use it.
      2) If SLURM_NTASKS>1, assume a SLURM multi-node job.
      3) If WORLD_SIZE>1, assume torchrun / torch.distributed.launch.
      4) Otherwise: single-GPU/CPU local fallback (disable CUDA, file-based rendezvous).
    """

    # --- Decide if this is purely local (no SLURM, no torchrun) → turn off CUDA entirely. ---
    slurm_tasks = int(os.getenv("SLURM_NTASKS", "1"))
    run_world  = int(os.getenv("WORLD_SIZE",    "1"))
    # if slurm_tasks == 1 and run_world == 1:
    #     # force CPU fallback on local
    #     os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Helper to grab ranks + set device
    def _finalize() -> tuple[str,int,int,int]:
        rank      = torch.distributed.get_rank()
        world_sz  = torch.distributed.get_world_size()
        local_id  = int(os.getenv("LOCAL_RANK", os.getenv("SLURM_LOCALID", "0")))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_id)
            device = f"cuda:{local_id}"
        else:
            device = "cpu"
        return device, local_id, rank, world_sz

    # 1) Re-use an existing group
    # if torch.distributed.is_initialized():
    #     return _finalize()

    # 2) SLURM multi-node
    if slurm_tasks > 1:
        rank      = int(os.getenv("SLURM_PROCID",   "0"))
        local_id  = int(os.getenv("SLURM_LOCALID",   "0"))
        os.environ["MASTER_ADDR"] = os.getenv("SLURM_LAUNCH_NODE_IPADDR")
        os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "29500")
        os.environ["RANK"]       = str(rank)
        os.environ["WORLD_SIZE"] = str(slurm_tasks)

        torch.distributed.init_process_group(
            backend    = "nccl",
            init_method= "env://",
            timeout    = datetime.timedelta(hours=1)
        )
        return _finalize()

    # 3) env:// multi-GPU (torchrun / torch.distributed.launch)
    if run_world > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        free_port = random.randint(30000, 40000)
        os.environ.setdefault("MASTER_PORT", str(free_port))
        torch.distributed.init_process_group(
            backend    = "nccl",
            init_method= "env://",
            timeout    = datetime.timedelta(hours=1)
        )
        return _finalize()

    # 4) Single-GPU/CPU fallback: file rendezvous under Gloo
    free_port = random.randint(30000, 40000)
    os.environ.update({
        "MASTER_ADDR":    "127.0.0.1",
        "MASTER_PORT":    str(free_port),
        "RANK":           "0",
        "WORLD_SIZE":     "1",
        "LOCAL_RANK":     "0",
    })
    init_file = tempfile.NamedTemporaryFile(delete=False).name
    torch.distributed.init_process_group(
        backend="gloo",
        init_method= f"file://{init_file}",
        rank       = 0,
        world_size = 1,
    )
    return _finalize()


def load_state_dict_flexible(model, state_dict):
    # handle checkpoints saved from DDP (keys start with 'module.')
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)  # keep strict=True so you don't silently miss weights
    return model


from torch.nn.parallel import DistributedDataParallel as DDP
def core_module_crop(model):
    # return the real nn.Module whether wrapped or not
    return model.module if isinstance(model, DDP) else model


def token_accuracy(stacks, preds, eos, pad, msk):
    """
    Compute both global accuracy (weighted) and per-sample accuracy vector.

    Args:
        stacks: target ids [B, L]
        preds:  logits [B, L, V] or ids [B, L]
        eos:    int id of EOS
        pad:    int id of PAD
        msk:    int id of MSK

    Returns:
        global_acc : scalar float tensor (on CPU)
        per_sample : [B] float tensor (on CPU)
    """
    if preds.dim() == 3:
        preds = preds.argmax(dim=-1)

    L = min(stacks.size(1), preds.size(1))
    stacks = stacks[:, :L]
    preds  = preds[:,  :L]

    # positions strictly before first EOS
    is_eos = (stacks == eos)
    before_first_eos = (is_eos.cumsum(dim=1) == 0)

    # exclude PADs and MSKs
    valid = before_first_eos & (stacks != pad) & (stacks != msk)

    correct = (stacks == preds) & valid

    # --- per-sample accuracy ---
    per_correct = correct.sum(dim=1).float()
    per_total   = valid.sum(dim=1).clamp_min(1).float()
    per_sample  = (per_correct / per_total).detach().cpu()

    # --- global weighted accuracy ---
    global_acc = (per_correct.sum() / per_total.sum()).detach().cpu()

    return global_acc, per_sample


def seed_everything(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# def load_JSONPICKLE(PATH: str, name: str) -> Any:
#     with open(f'{PATH}/{name}.json') as f:
#         d = json.load(f)
#     return jsonpickle.decode(d)


# def save_JSONPICKLE(PATH: str, pyobj: Any, name: str) -> None:
#     frozen = jsonpickle.encode(pyobj)
#     with open(f"{PATH}/{name}.json", 'w') as f:
#         json.dump(frozen, f)


def load_JSONPICKLE_NEW(PATH: str, name: Optional[str] = None) -> Any:
    if name == None:
        with open(f'{PATH}', 'r') as f:
            data = f.read()
        return jsonpickle.decode(data)
    else:
        with open(f'{PATH}/{name}.json', 'r') as f:
            data = f.read()
        return jsonpickle.decode(data)


def save_JSONPICKLE_NEW(PATH: str, pyobj: Any, name: str) -> None:
    frozen = jsonpickle.encode(pyobj)
    with open(f"{PATH}/{name}.json", 'w') as f:
        f.write(frozen)


def init_tokenmaps(PATH: str) -> List[str]:                 #TODO type hints correctly
    tokens = load_JSONPICKLE_NEW(PATH, 'tokens')
    
    # Insert special tokens if not present
    PAD_TOKEN = "<PAD>"
    MSK_TOKEN = "<MSK>"
    EOS_TOKEN = "<EOS>"
    for special_tk in [EOS_TOKEN, PAD_TOKEN, MSK_TOKEN]:
        if special_tk not in tokens:
            tokens.append(special_tk)
    
    token_to_idx = {tk: i for i, tk in enumerate(tokens)}
    eos_idx = token_to_idx[EOS_TOKEN]
    pad_idx = token_to_idx[PAD_TOKEN]
    msk_idx = token_to_idx[MSK_TOKEN]
    idx_to_token = {i: tk for i, tk in enumerate(token_to_idx)}
    return tokens, token_to_idx, idx_to_token, EOS_TOKEN, PAD_TOKEN, MSK_TOKEN, eos_idx, pad_idx, msk_idx


import torch
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

def grad_stats(parameters, norm_type=2.0):
    """Return total L2 norm and max abs value of all gradients."""
    total_norm_sq, max_abs = 0.0, 0.0
    for p in parameters:
        if p.grad is None:            # frozen / unused param
            continue
        grad = p.grad.detach()
        total_norm_sq += grad.norm(norm_type).pow(2).item()
        max_abs = max(max_abs, grad.abs().max().item())
    total_norm = total_norm_sq ** 0.5
    return total_norm, max_abs


def entropy_from_unnorm_logp(unnorm_logp: torch.Tensor, dim: int = 1, eps: float = 1e-12):
    """
    unnorm_logp : raw log-probability scores (any real numbers)  
                  shape (..., C, ...)
    dim         : the class/channel dimension

    returns     : Shannon entropy H(p)  with  p = softmax(unnorm_logp)
                  shape is the input shape with the `dim` removed
    """
    # 1) normalise in log-space
    log_Z  = torch.logsumexp(unnorm_logp, dim=dim, keepdim=True)  # log ∑ₖ e^{zₖ}
    log_p  = unnorm_logp - log_Z                                   # == F.log_softmax

    # 2) convert to probs only for the multiply
    p      = log_p.exp()

    # 3) H = −Σ p log p
    H      = -(p * log_p).sum(dim=dim)

    return H


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
    num_noisy_signals = noisy_signals.shape[0]
    num_signals = signals.shape[0]

    plt.figure(figsize=(10, 8))
    for i in range(num_noisy_signals):
        if i > num_signals-1:
            j = num_signals-1
        else:
             j = i
        plt.subplot(3, 1, 1)
        plt.plot(wavelengths, signals[j], alpha=0.2)
        plt.title("Pure Signals")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity")

        plt.subplot(3, 1, 2)
        plt.plot(wavelengths, noisy_signals[i], alpha=0.2)
        plt.title("Signals with Gaussian Noise")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity")
        
        plt.subplot(3, 1, 3)
        plt.plot(RAT[i], alpha=0.2)
        plt.title("RAT Signals")
        plt.xlabel("Index")
        plt.ylabel("Intensity")

    plt.tight_layout()
    plt.show()
    
def load_materialdata_file(filename):
    """
    Reads a material data file where the line immediately above the first numeric data row
    indicates whether wavelength is in nm or um.

    Steps:
    1. Collect all lines except comments, empty lines, lines with references or keys (DATA:, etc.).
    2. Find the first line that parses as three floats -> data start.
       The line just above it is the 'header' line for nm/um detection.
    3. Parse numeric data from that line onward.
    4. Convert wavelength to meters based on whether header says 'nm' or 'um'.
    5. Return: (wavelengths_in_m, n_vals, k_vals, short_name)
    """
    short_name = os.path.splitext(os.path.basename(filename))[0]  # e.g. "Ag" or "ITO"
    
    ignore_keywords = ["REFERENCES:", "COMMENTS:", "DATA:"]
    valid_lines = []

    with open(filename, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                # Skip empty or comment lines
                continue
            if any(keyword in line.upper() for keyword in ignore_keywords):
                # Skip lines with references, comments, data block markers
                continue
            valid_lines.append(line)

    if not valid_lines:
        print(f"Warning: {filename} had no valid lines after filtering.")
        # Return empty arrays
        return np.array([]), np.array([]), np.array([]), short_name

    first_data_idx = None
    for i, line in enumerate(valid_lines):
        parts = line.split()
        if len(parts) == 3:
            try:
                w, n, k = map(float, parts)
                first_data_idx = i
                break
            except ValueError:
                pass

    if first_data_idx is None:
        print(f"Warning: {filename} had no parseable numeric data.")
        return np.array([]), np.array([]), np.array([]), short_name

    # Check line above data (header) for nm/um
    wavelength_unit = None
    if first_data_idx > 0:
        header_line = valid_lines[first_data_idx - 1].lower()
        if "um" in header_line or "μm" in header_line:
            wavelength_unit = "um"
        elif "nm" in header_line:
            wavelength_unit = "nm"

    if wavelength_unit is None:
        # Default to um or nm as you prefer:
        wavelength_unit = "um"
        print(f"Warning: {filename} - no explicit nm/um in header above data. Defaulting to {wavelength_unit}.")

    wavelengths = []
    n_vals = []
    k_vals = []

    # Parse numeric data
    for line in valid_lines[first_data_idx:]:
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            w, n, k = map(float, parts)
            wavelengths.append(w)
            n_vals.append(n)
            k_vals.append(k)
        except ValueError:
            pass

    wavelengths = np.array(wavelengths)
    n_vals = np.array(n_vals)
    k_vals = np.array(k_vals)

    # Convert to meters
    if wavelength_unit == "nm":
        wavelengths_si = wavelengths * 1e-9
    else:  # 'um'
        wavelengths_si = wavelengths * 1e-6

    return wavelengths_si, n_vals, k_vals, short_name


def masked_mae(x, y):
    # treat only finite y as valid (predicted_spectra)
    mask = torch.isfinite(y).all(dim=-1, keepdim=True)  # [B,W,1]
    valid = mask.expand_as(y)
    num = (torch.abs(x - torch.nan_to_num(y))).where(valid, torch.zeros_like(x)).sum(dim=1).sum(dim=1)
    den = valid.sum(dim=1).sum(dim=1).clamp_min(1)
    return num / den


def masked_huber(x, y, delta=0.01):  # tune delta to your spectrum scale
    d = (x - y).abs()
    quad = torch.clamp(d, max=delta)
    lin  = d - quad
    return (0.5 * quad**2 / delta + lin).mean()


from collections import deque
def rolling_mean(lst, window_size):
    """
    Symmetric, edge-shortened rolling mean (length preserved).
    At index i we use a centered window with radius r_i = min(r, i, n-1-i),
    so endpoints use window size 1 → exact original values.

    Note: For even window_size, the effective centered window is (2*r+1)
    with r=(window_size-1)//2 (i.e., the largest odd size ≤ window_size),
    to keep symmetry and exact endpoints.
    """
    n = len(lst)
    if n == 0 or window_size <= 0:
        return
    if window_size == 1:
        for x in lst:
            yield x
        return

    r = (window_size - 1) // 2  # centered radius

    # prefix sums for O(1) range-sum
    s = [0.0]
    for x in lst:
        s.append(s[-1] + x)

    for i in range(n):
        ri = min(r, i, n - 1 - i)     # shrink near edges; 0 at endpoints
        a = i - ri                    # inclusive start
        b = i + ri + 1                # exclusive end
        yield (s[b] - s[a]) / (b - a)

def unique_length_int_generator(start: float, stop: float, amount: float):
    start = int(start)
    stop = int(stop)
    amount = int(amount)
    len_unique=0
    amount = amount-1
    while len_unique<amount: 
        amount = amount+1
        subset_idx = torch.linspace(start, stop, amount, dtype=int).unique()
        len_unique = len(subset_idx)
        # print(len_unique)
    return subset_idx


import re
def extract_mae_values(filepath):
    """
    Extracts all 'min val MAE' and 'last val MAE' values from a training log (.out file).

    Args:
        filepath (str): Path to the .out file.

    Returns:
        tuple[list[float], list[float]]:
            A tuple containing two lists:
                - min_mae_list: all "min val MAE" values
                - last_mae_list: all "last val MAE" values
    """
    min_mae_list = []
    last_mae_list = []

    with open(filepath, "r") as f:
        for line in f:
            m_min = re.search(r"min val MAE:\s*([0-9.]+)", line)
            if m_min:
                min_mae_list.append(float(m_min.group(1)))

            m_last = re.search(r"last val MAE:\s*([0-9.]+)", line)
            if m_last:
                last_mae_list.append(float(m_last.group(1)))

    return min_mae_list, last_mae_list


# Example usage
# %%
d = False
if d:
# %%
    import config_MD50 as cfg
    file_path_DiT = rf'{cfg.PATH_RUN}/results/MD49_18174987.out'
    file_path_T = rf'{cfg.PATH_RUN}/results/MD50_17083190.out'
    min_mae_T, last_mae_T = extract_mae_values(file_path_T)
    min_mae_DiT, last_mae_DiT = extract_mae_values(file_path_DiT)
    plt.plot(list(zip(min_mae_T, last_mae_T)))
    plt.plot(list(zip(min_mae_DiT, last_mae_DiT)))
    plt.gca().set_ylim(0.037,0.045)
    plt.gca().set_xlim(700,1000)
# %%
# %%
# print("MAE pairs:", list(zip(min_mae, last_mae)))
