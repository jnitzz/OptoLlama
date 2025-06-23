# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 12:10:52 2025

@author: a3536
"""

# ---------------------------------------------------------------------------
#  save_closest_train.py
# ---------------------------------------------------------------------------
import pathlib, torch, numpy as np, tqdm
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
#  0.  helper – lazy mem-map backed datasets (same as in your plots.py)
# ---------------------------------------------------------------------------
def _read_npy_shape(path: str):
    with open(path, "rb") as fh:
        major, _ = np.lib.format.read_magic(fh)
        if major == 1:
            header_dict, *__ = np.lib.format.read_array_header_1_0(fh)
        elif major in (2, 3):
            header_dict, *__ = np.lib.format.read_array_header_2_0(fh)
        else:
            raise ValueError("Unsupported .npy version")
    return header_dict

class LazyMemmap(Dataset):
    """Generic 2-D mem-mapped float / int array – opens lazily per worker."""
    def __init__(self, npy_path: str, dtype=None):
        self.path = str(npy_path)
        self.shape = _read_npy_shape(self.path)
        self._mm = None
    def _open(self):
        if self._mm is None:
            self._mm = np.load(self.path, mmap_mode="r")
    def __len__(self):  return self.shape[0]
    def __getitem__(self, i):
        self._open()
        return self._mm[i]

# ---------------------------------------------------------------------------
#  1.  closest-spectra search for *one test batch* (GPU, streaming training)
# ---------------------------------------------------------------------------
@torch.no_grad()
def _batch_closest_mse(test_batch: torch.Tensor,
                       train_mm       : LazyMemmap,
                       batch_train    : int,
                       device         : str):
    """
    For *test_batch* [T,513] return three tensors on CPU:

        best_idx   [T]       – index in training set
        best_mse   [T]       – MSE value
        best_spec  [T,513]   – spectrum

    Scans the mem-map in chunks, constant RAM.
    """
    T = test_batch.size(0)
    best_mse  = torch.full((T,), float("inf"), device=device)
    best_idx  = torch.full((T,), -1, dtype=torch.long, device=device)
    best_spec = torch.empty_like(test_batch, device=device)

    N_train   = len(train_mm)
    for s0 in range(0, N_train, batch_train):
        s1   = min(s0 + batch_train, N_train)
        spec = torch.from_numpy(train_mm[s0:s1]).to(device, non_blocking=True)
        mse  = torch.mean((spec.unsqueeze(1) - test_batch)**2, dim=2)  # [B,T]
        chunk_best, chunk_arg = torch.min(mse, dim=0)                  # per test

        upd = chunk_best < best_mse
        if upd.any():
            best_mse[upd]  = chunk_best[upd]
            best_idx[upd]  = chunk_arg[upd] + s0
            best_spec[upd] = spec[chunk_arg[upd]][:]
    return (best_idx.cpu(),
            best_mse.cpu(),
            best_spec.cpu())

# ---------------------------------------------------------------------------
#  2.  high-level driver – operates on a *.pt* validation file
# ---------------------------------------------------------------------------
def save_validation_with_closest(
        val_pt          : str | pathlib.Path,
        train_spec_npy  : str | pathlib.Path,
        train_tok_npy   : str | pathlib.Path,
        out_pt          : str | pathlib.Path,
        *,
        batch_test      = 256,
        batch_train     = 8192,
        device          = "cuda"):
    """
    For every sample in *val_pt* find the closest spectrum in *train_spec_npy*
    (MSE over the 513 values).  Save the spectra **and their label tokens**
    to *out_pt* so that `SinglePTDataset(out_pt)` works immediately.
    """
    # -- 2.1  read validation split completely (small enough for RAM) ----------
    val_spec, val_tok = torch.load(val_pt, map_location="cpu")  # lists
    val_spec, val_tok = val_spec[:1000], val_tok[:1000]
    N_val             = len(val_spec)
    print(f"Validation set: {N_val:,d} spectra")

    # -- 2.2  mem-maps for training data --------------------------------------
    mm_spec = LazyMemmap(train_spec_npy, dtype="float32")
    mm_tok  = LazyMemmap(train_tok_npy , dtype="int16")  # padded matrix

    # -- 2.3  iterate over validation in batches ------------------------------
    out_spec = []
    out_tok  = []

    loader = DataLoader(val_spec, batch_size=batch_test, shuffle=False)

    for offset, batch_cpu in enumerate(tqdm.tqdm(loader, desc="val batches")):
        batch_gpu = batch_cpu.to(device, dtype=torch.float32)
        (idx_cpu, mse_cpu, spec_cpu) = _batch_closest_mse(
                                            batch_gpu, mm_spec, batch_train, device)

        out_spec.extend(spec_cpu)                 # append spectra list
        # gather the *label* rows by index, strip PAD (==0) on the fly
        for j, idx in enumerate(idx_cpu):
            tok_row = mm_tok[int(idx)]
            valid   = tok_row[tok_row != 0]
            out_tok.append(torch.as_tensor(valid, dtype=torch.int16))

    # -- 2.4  write result -----------------------------------------------------
    torch.save((out_spec, out_tok), out_pt)
    print("✅  wrote", out_pt)

# ---------------------------------------------------------------------------
#  3.  CLI / example ---------------------------------------------------------
import config_OL19 as c
if __name__ == "__main__":
    save_validation_with_closest(
        val_pt         = rf"{c.PATH_DATA}/my_dataset_val.pt",
        train_spec_npy = rf"{c.PATH_DATA}/my_dataset_16m.npy",
        train_tok_npy  = rf"{c.PATH_DATA}/my_dataset_16m_tokens.npy",
        out_pt         = rf"{c.PATH_DATA}/my_dataset_val_closest.pt",
        batch_test     = 256,      # tune ↔ GPU RAM
        batch_train    = 512,    # bigger ⇒ better GPU utilisation
        device         = "cuda")

# data = torch.load(r'd:/Profile/a3536/Eigene Dateien/GitHub/OptoLlama/data/RF_T63_16m/my_dataset_val_closest.pt')
