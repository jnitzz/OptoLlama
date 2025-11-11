import json, os, tempfile
from typing import List, Optional, Any, Dict, Tuple
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
PAD_TOKEN = "<PAD>"
MSK_TOKEN = "<MSK>"
EOS_TOKEN = "<EOS>"

def save_as_json(path: str, pyobj: Any, name: Optional[str] = None) -> None:
    """
    Save a Python object (e.g., list of strings, dict, etc.) as a JSON file.

    Args:
        pyobj: The Python object to serialize.
        path: Destination file path.
        Optional: name: file name, ".json" will be added automatically
    """
    if name is not None:
        path = f"{path}/{name}.json"
    else:
        path = path
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pyobj, f, indent=2, ensure_ascii=False)


def load_as_json(path: str, name: Optional[str] = None) -> Any:
    """
    Load and return a Python object from a JSON file.

    Args:
        path: Destination file path.
        Optional: name: file name, ".json" will be added automatically
    Returns:
        Python object (e.g. list, dict).
    """
    if name == None:
        with open(f'{path}', 'r', encoding="utf-8") as f:
            return json.load(f)
    else:
        with open(f'{path}/{name}.json', 'r', encoding="utf-8") as f:
            return json.load(f)


def init_tokenmaps(path: str) -> List[str]:
    tokens = load_as_json(path, 'tokens')
    # Insert special tokens if not present
    for special_tk in (EOS_TOKEN, PAD_TOKEN, MSK_TOKEN):
        if special_tk not in tokens:
            tokens.append(special_tk)
    
    token_to_idx = {tk: i for i, tk in enumerate(tokens)}
    eos_idx = token_to_idx[EOS_TOKEN]
    pad_idx = token_to_idx[PAD_TOKEN]
    msk_idx = token_to_idx[MSK_TOKEN]
    idx_to_token = {i: tk for i, tk in enumerate(token_to_idx)}
    
    return tokens, token_to_idx, idx_to_token, EOS_TOKEN, PAD_TOKEN, MSK_TOKEN, eos_idx, pad_idx, msk_idx


def unique_length_int_generator(start: float, stop: float, amount: float):
    start = int(start)
    stop = int(stop)
    amount = int(amount)
    if not (-1 < start < stop) or not (0 < amount <= stop):
        print(f"Your start, stop, amount is: {start}, {stop}, {amount}. \
              amount must be (-1 < start < stop) and (0 < amount < stop).")
        return ValueError
    
    len_unique=-1
    amount = amount-1
    while len_unique<amount: 
        amount = amount+1
        subset_idx = torch.linspace(start, stop-1, amount, dtype=int).unique()
        len_unique = len(subset_idx)
    
    return subset_idx


def core_module_crop(model):
    return model.module if isinstance(model, DDP) else model

def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    
    return state_dict

@torch.no_grad()
def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    train_losses: Optional[torch.Tensor] = None,
    train_acc: Optional[torch.Tensor] = None,
    valid_acc: Optional[torch.Tensor] = None,
    valid_MAE: Optional[torch.Tensor] = None,
    *,
    scaler: Any = None,
    scheduler: Any = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save checkpoint in the 'optollama.py' format with DDP-safe model weights (unwrapped, no 'module.').
    """
    core = core_module_crop(model)
    state = {
        "epoch": int(epoch) if epoch is not None else None,
        "model_state": _strip_module_prefix(core.state_dict()),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "train_losses": train_losses,
        "train_acc": train_acc,
        "valid_acc": valid_acc,
        "valid_MAE": valid_MAE,
        # optional but useful:
        "scaler_state": (scaler.state_dict() if scaler else None),
        "scheduler_state": (scheduler.state_dict() if scheduler else None),
        "format_version": 1,
    }
    if extra:
        state["extra"] = extra

    # atomic write to avoid partial files
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=os.path.dirname(path), delete=False) as tmp:
        torch.save(state, tmp.name)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name
    os.replace(tmp_path, path)

@torch.no_grad()
def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    *,
    map_location: str = "cpu",
    strict: bool = True,
    scaler: Any = None,
    scheduler: Any = None,
) -> Tuple[Optional[int], Dict[str, Any]]:
    """
    Load an optollama-style checkpoint, robust to DDP/non-DDP diffs.
    Returns (start_epoch, full_state_dict).
    """
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
            raise RuntimeError(
                "Unrecognized checkpoint layout; expected dict with 'model_state' (or 'model')."
            )

    # always load into the core module; strip 'module.' if present
    core = core_module_crop(model)
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


def apply_sampling_from_sources(model, *, args=None, cfg=None, default_temp=0.0, default_top_k=0, default_top_p=0.0):
    # precedence: CLI args (if present) > cfg attributes > defaults
    temperature = getattr(args, "temperature", None)
    top_k       = getattr(args, "top_k", None)
    top_p       = getattr(args, "top_p", None)

    if temperature is None: temperature = float(getattr(cfg, "TEMPERATURE", default_temp)) if cfg else default_temp
    if top_k       is None: top_k       = int(getattr(cfg, "TOP_K",       default_top_k)) if cfg else default_top_k
    if top_p       is None: top_p       = float(getattr(cfg, "TOP_P",     default_top_p)) if cfg else default_top_p

    # normalize bad/disabled values
    temperature = float(temperature or 0.0)
    top_k = int(top_k or 0)
    top_p = float(top_p or 0.0)

    if hasattr(model, "set_sampling"):
        model.set_sampling(temperature=temperature, top_k=top_k, top_p=top_p)

    # return for logging/saving if you like
    return {"temperature": temperature, "top_k": top_k, "top_p": top_p}


def boxcar_kernel(win: int, device):
    win = max(1, int(win))
    if win % 2 == 0: win += 1
    k = torch.ones(win, device=device, dtype=torch.float32) / float(win)
    return k


def gauss_kernel(win: int, sigma: float, device):
    win = max(1, int(win))
    if win % 2 == 0: win += 1
    r = (win - 1) // 2
    xs = torch.arange(-r, r+1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (xs / max(1e-6, float(sigma)))**2)
    k = k / k.sum().clamp_min(1e-6)
    return k


def smooth_1d(x_3w: torch.Tensor, method: str, win: int, sigma: float) -> torch.Tensor:
    # x_3w: [...,3,W] -> do depthwise 1D conv along last dim
    orig_dim = x_3w.dim()
    if orig_dim == 2:
        x_3w = x_3w.unsqueeze(0)  # [1,3,W]
    _, C, W = x_3w.shape
    device = x_3w.device
    if method.lower() == "gaussian":
        k = gauss_kernel(win, sigma, device)
    else:
        k = boxcar_kernel(win, device)
    weight = k.view(1, 1, -1).repeat(C, 1, 1)  # [C,1,K]
    x = F.pad(x_3w, (k.numel()//2, k.numel()//2), mode="reflect")
    x = F.conv1d(x, weight, groups=C)
    x = x[:, :, :W]
    return x.squeeze(0) if orig_dim == 2 else x


def parse_order(order_str: str):
    order_str = (order_str or "R>A>T").upper()
    mapping = {'R':0, 'A':1, 'T':2}
    seq = [mapping[c.strip()] for c in order_str.split('>') if c.strip() in mapping]
    rest = [i for i in (0,1,2) if i not in seq]
    seq.extend(rest)
    return tuple(seq[:3])


def wl_mask(wavelengths, wl_min: float, wl_max: float, device):
    if wavelengths is None:
        return None
    wl = torch.as_tensor(wavelengths, dtype=torch.float32, device=device)
    return (wl >= float(wl_min)) & (wl <= float(wl_max))


def redistribute_mismatch(x: torch.Tensor, order: str, target_sum: float = 1.0):
    """
    x: [...,3,W] with values in [0,1]. Enforce per-W sum≈target_sum
    by distributing residual in the given priority order.
    """
    orig_dim = x.dim()
    if orig_dim == 2:
        x = x.unsqueeze(0)  # [1,3,W]
    B, C, W = x.shape
    pri = parse_order(order)
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


def apply_noise(x: torch.Tensor, noise_cfg: dict, wavelengths):
    if not noise_cfg or not noise_cfg.get("enabled", False):
        return x
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
        mask = wl_mask(wavelengths, wl_min, wl_max, device) if wavelengths is not None else None

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


def apply_smoothing(x: torch.Tensor, smooth_cfg: dict):
    if not smooth_cfg or not smooth_cfg.get("enabled", False):
        return x
    method = smooth_cfg.get("method", "gaussian")
    win    = int(smooth_cfg.get("win", 5))
    sigma  = float(smooth_cfg.get("sigma", 1.0))
    return smooth_1d(x, method, win, sigma)