# -*- coding: utf-8 -*-
import os, torch
from typing import List, Optional, Sequence, Any
import jsonpickle
from safetensors.torch import save_file, load_file
#TODO check model und optimizer als safetensor
#TODO save results als .json ohne pickle
def load_JSONPICKLE(PATH: str, name: Optional[str] = None) -> Any:
    if name == None:
        with open(f'{PATH}', 'r') as f:
            data = f.read()
        return jsonpickle.decode(data)
    else:
        with open(f'{PATH}/{name}.json', 'r') as f:
            data = f.read()
        return jsonpickle.decode(data)


def save_JSONPICKLE(PATH: str, pyobj: Any, name: str) -> None:
    frozen = jsonpickle.encode(pyobj)
    with open(f"{PATH}/{name}.json", 'w') as f:
        f.write(frozen)

#TODO nur mit json ohne pickle
def _pack_tokens(tokens: Sequence[str]):
    if not tokens:
        return {
            "data": torch.empty(0, dtype=torch.uint8),
            "starts": torch.empty(0, dtype=torch.int64),
            "lengths": torch.empty(0, dtype=torch.int64),
        }
    btoks = [t.encode("utf-8") for t in tokens]
    lengths = torch.tensor([len(b) for b in btoks], dtype=torch.int64)
    starts = torch.cat([torch.tensor([0]), lengths.cumsum(0)[:-1]])
    data = torch.tensor(list(b"".join(btoks)), dtype=torch.uint8)
    return {"data": data, "starts": starts, "lengths": lengths}


def load_TOKENS_SAFETENSORS(file_path: str, name: Optional[str] = None):
    if name is not None:
        file_path = os.path.join(file_path,f"{name}.safetensors")
    else:
        pass
    tens = load_file(file_path)
    data, starts, lengths = tens["data"].cpu(), tens["starts"].cpu(), tens["lengths"].cpu()
    return [bytes(data[s:s+L].tolist()).decode("utf-8") for s, L in zip(starts.tolist(), lengths.tolist())]


def save_TOKENS_SAFETENSORS(PATH: str, tokens: Sequence[str], name: str = "tokens") -> None:
    os.makedirs(PATH, exist_ok=True)
    save_file(_pack_tokens(tokens), f"{PATH}/{name}.safetensors")
    
#TODO check bc functional tokens in tokens already
def init_tokenmaps(PATH: str) -> List[str]:
    tokens = load_TOKENS_SAFETENSORS(PATH, 'tokens')
    
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


from torch.nn.parallel import DistributedDataParallel as DDP

def core_module_crop(model):
    # return the real nn.Module whether wrapped or not
    return model.module if isinstance(model, DDP) else model


def load_state_dict_flexible(model, state_dict):
    # handle checkpoints saved from DDP (keys start with 'module.')
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    return model


def load_checkpoint(model, path, device="cpu"):
    model_core = core_module_crop(model)
    ckpt = torch.load(path, map_location=device)
    load_state_dict_flexible(model_core, ckpt)
    return model_core


def save_checkpoint(model, path, epoch=None, metrics=None):
    core = core_module_crop(model)
    state = {
        "model": core.state_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
    }
    torch.save(state, path)
