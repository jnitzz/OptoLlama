#!/usr/bin/env python3
import os
import math
import json
import pickle
from typing import List, Tuple

import numpy as np
import torch

# ---- project helpers ----
# tokens & special indices (EOS/PAD/MSK) are read the same way your code does
from utils import init_tokenmaps  # :contentReference[oaicite:3]{index=3}

# ----------------------------------------------------------------------------------------------------------------------
# CONFIG — edit paths to your 10 training shards and the test file (you uploaded data_test_crop.pt)
# ----------------------------------------------------------------------------------------------------------------------
path_base = r"d:\Profile\a3536\Eigene Dateien\GitHub\ColorAppearanceToolbox\Diffusion\data\TF_MA2"
path_base = r"/scratch/htc/jschaibl/repos/ColorAppearanceToolbox/Diffusion/data/TF_MA2"
TRAIN_PATHS = [rf"{path_base}/data_train_0.pt",rf"{path_base}/data_train_1.pt"]
TEST_PATH   = rf"{path_base}/data_test.pt"     # your uploaded partial test split
OUT_BASENAME = rf"{path_base}/nearest_train_for_each_test"   # produces .pkl and .json

# Optional: directory to stash a memmap of all training spectra (speeds up reruns)
MEMMAP_DIR = rf"{path_base}"
MEMMAP_FILE = os.path.join(MEMMAP_DIR, "train_RAT_memmap.npy")
TOKENS_PKL  = os.path.join(MEMMAP_DIR, "train_tokens.pkl")

# Tuning
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_BATCH   = 512           # test slab size
TRAIN_CHUNK  = 8192          # streamed train chunk size
NUM_WORKERS  = 0             # we stream from memmap; no loaders needed

# ----------------------------------------------------------------------------------------------------------------------
def _load_pt_pair(path: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Return (spectra_list, stacks_list) from a .pt produced like your datasets."""
    spectra, stacks = torch.load(path, weights_only=False)
    
    return spectra, stacks

def _count_train_samples(paths: List[str]) -> Tuple[int, int]:
    """Return (N_total, flat_dim) where flat_dim = 3*W (R|A|T concat)."""
    total = 0
    flat_dim = None
    for p in paths:
        spec_list, _ = _load_pt_pair(p)
        total += len(spec_list)
        if flat_dim is None:
            # each item is [W,3]; flatten → 3*W
            s0 = spec_list[0].reshape(3,-1).T  # [W,3] (same as your dataset)  :contentReference[oaicite:4]{index=4}
            flat_dim = s0.shape[0] * s0.shape[1]  # 3*W
    if flat_dim is None:
        raise RuntimeError("No training samples found.")
    return total, flat_dim

def _ensure_memmap(paths: List[str], memmap_file: str, tokens_pkl: str) -> Tuple[str, str, int]:
    """
    Create (or reuse) a memmap of shape [N_train, 3*W] and a tokens.pkl (list of LongTensor stacks).
    Returns (memmap_file, tokens_pkl, N_train).
    """
    os.makedirs(os.path.dirname(memmap_file), exist_ok=True)
    N, D = _count_train_samples(paths)

    # reuse if present with correct shape
    if os.path.exists(memmap_file) and os.path.exists(tokens_pkl):
        hdr = np.load(memmap_file, mmap_mode="r")
        if hdr.shape[0] == N and hdr.shape[1] == D:
            return memmap_file, tokens_pkl, N
        # else fallthrough and rebuild

    # mm = np.memmap(memmap_file, mode="w+", dtype=np.float32, shape=(N, D))
    from numpy.lib.format import open_memmap
    mm = open_memmap(memmap_file, mode="w+", dtype=np.float32, shape=(N, D))
    all_tokens: List[torch.Tensor] = []
    wptr = 0
    for p in paths:
        spec_list, stack_list = _load_pt_pair(p)
        for spec, stk in zip(spec_list, stack_list):
            spec = spec.reshape(3, -1).T  # [W,3]
            flat = spec.T.reshape(-1).to(torch.float32).cpu().numpy()  # [3W]
            mm[wptr, :] = flat
            all_tokens.append(stk.clone().to(torch.long).cpu())
            wptr += 1
    mm.flush()
    with open(tokens_pkl, "wb") as fh:
        pickle.dump(all_tokens, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return memmap_file, tokens_pkl, N

@torch.no_grad()
def _nearest_by_mae(
    test_batch_flat: torch.Tensor,    # [B, D], float32 on DEVICE
    train_mm: np.memmap,              # [N, D], float32 memmap on CPU
    start: int, end: int
):
    """
    Compute MAE between each test vector and train_chunk = train_mm[start:end].
    Returns (best_val[B], best_idx[B]) indices are global (start + argmin).
    """
    # bring chunk to GPU
    chunk = torch.from_numpy(train_mm[start:end]).to(DEVICE, non_blocking=True)  # [T, D]
    # pairwise |T - B| mean over D → [T, B]
    diff = (chunk.unsqueeze(1) - test_batch_flat.unsqueeze(0)).abs().mean(dim=2)
    # min over T for each test item
    vals, argmin = diff.min(dim=0)          # [B]
    return vals, argmin + start

@torch.no_grad()
def build_matches(
    train_paths: List[str],
    test_path: str,
    out_base: str,
    memmap_file: str,
    tokens_pkl: str
):
    # 1) create (or reuse) train memmap and token list
    memmap_file, tokens_pkl, N_train = _ensure_memmap(train_paths, memmap_file, tokens_pkl)
 
    train_mm = np.load(memmap_file, mmap_mode="r") # [N, D] float32

    with open(tokens_pkl, "rb") as fh:
        train_tokens: List[torch.Tensor] = pickle.load(fh)

    # 2) token tables → string mapping, special ids
    #    Same as your code path (returns tokens, maps, EOS/PAD/MSK ids). :contentReference[oaicite:5]{index=5}
    #    We need this only to decode the best train stacks into readable tokens.
    from config_MD49 import PATH_DATA  # adjust config if needed  :contentReference[oaicite:6]{index=6}
    tokens, t2i, i2t, EOS_TOKEN, PAD_TOKEN, MSK_TOKEN, eos, pad, msk = init_tokenmaps(PATH_DATA)

    # 3) load test .pt
    test_specs, test_stacks = _load_pt_pair(test_path)  # we only need spectra to match

    # 4) stream test in batches, find nearest neighbor over train chunks
    results = []
    D = train_mm.shape[1]
    n_test = len(test_specs)

    for t0 in range(0, n_test, TEST_BATCH):
        t1 = min(t0 + TEST_BATCH, n_test)
        # stack and flatten test -> [B, D]
        tb = []
        for i in range(t0, t1):
            s = test_specs[i].reshape(3, -1).T           # [W,3] same as dataset  :contentReference[oaicite:7]{index=7}
            tb.append(s.T.reshape(-1).to(torch.float32))
        test_flat = torch.stack(tb, dim=0).to(DEVICE, non_blocking=True)  # [B,D]

        # init bests for this slab
        best_val = torch.full((t1 - t0,), float("inf"), device=DEVICE)
        best_idx = torch.full((t1 - t0,), -1, dtype=torch.long, device=DEVICE)

        # stream the (possibly huge) train set
        for s0 in range(0, N_train, TRAIN_CHUNK):
            s1 = min(s0 + TRAIN_CHUNK, N_train)
            vals, idxs = _nearest_by_mae(test_flat, train_mm, s0, s1)  # [B], [B]
            mask = vals < best_val
            best_val[mask] = vals[mask]
            best_idx[mask] = idxs[mask]

        # stash per-test results (convert to CPU)
        best_val = best_val.cpu().numpy()
        best_idx = best_idx.cpu().numpy()
        for i_local, (gidx, mae) in enumerate(zip(best_idx, best_val)):
            # reconstruct the training spectrum and decode its tokens
            best_spec = train_mm[gidx].copy()               # [D]
            tok_ids = train_tokens[gidx]                    # 1D tensor of ids (variable len)
            # cut to EOS & drop specials
            ids = tok_ids.tolist()
            if eos in ids:
                ids = ids[:ids.index(eos)]
            ids = [x for x in ids if x not in (pad, msk)]
            seq_tokens = [i2t[int(x)] for x in ids]

            results.append({
                "test_index": t0 + i_local,
                "best_train_index": int(gidx),
                "best_mae": float(mae),
                "best_train_spectrum_flat": best_spec.tolist(),   # R|A|T concat
                "best_train_sequence_tokens": seq_tokens,         # human-readable
            })

        # small progress cue
        done = t1
        print(f"[{done}/{n_test}] processed")

    # 5) save
    with open(out_base + ".pkl", "wb") as fh:
        pickle.dump(results, fh)
    with open(out_base + ".json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Saved: {out_base}.pkl and {out_base}.json")

if __name__ == "__main__":
    assert len(TRAIN_PATHS) == 2, "Please fill TRAIN_PATHS with your 10 training files."
    build_matches(TRAIN_PATHS, TEST_PATH, OUT_BASENAME, MEMMAP_FILE, TOKENS_PKL)
