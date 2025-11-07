# -*- coding: utf-8 -*-
import torch, json, os, tempfile
from typing import List, Optional, Any, Dict, Tuple
from torch.nn.parallel import DistributedDataParallel as DDP


def save_JSON(PATH: str, pyobj: Any, name: Optional[str] = None) -> None:
    """
    Save a Python object (e.g., list of strings, dict, etc.) as a JSON file.

    Args:
        pyobj: The Python object to serialize.
        PATH: Destination file path.
        Optional: name: file name, ".json" will be added automatically
    """
    if name is not None:
        path = f"{PATH}/{name}.json"
    else:
        path = PATH
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pyobj, f, indent=2, ensure_ascii=False)


def load_JSON(PATH: str, name: Optional[str] = None) -> Any:
    """
    Load and return a Python object from a JSON file.

    Args:
        PATH: Destination file path.
        Optional: name: file name, ".json" will be added automatically
    Returns:
        Python object (e.g. list, dict).
    """
    if name == None:
        with open(f'{PATH}', 'r', encoding="utf-8") as f:
            return json.load(f)
    else:
        with open(f'{PATH}/{name}.json', 'r', encoding="utf-8") as f:
            return json.load(f)


#TODO check bc functional tokens in tokens already
def init_tokenmaps(PATH: str) -> List[str]:
    tokens = load_JSON(PATH, 'tokens')
    # tokens = load_JSONPICKLE(PATH, 'tokens')
    save_JSON(PATH, tokens, 'tokens')
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
