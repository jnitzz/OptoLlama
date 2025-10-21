# -*- coding: utf-8 -*-
"""
Batched data generation using TMM_FAST (coherent TMM).
Writes (spectra_tensor, all_labels) tuples like the original RayFlare script.

Usage (examples):
    python data_generation_TMMFAST.py --batch 256 --device auto --save-every 100000
    python data_generation_TMMFAST.py --src-csv /path/to/data_test.csv --start 0 --end -1
"""

import os
import argparse
import itertools
import random
import ast
import numpy as np
import torch
import pandas as pd

import config_RF as c
from typing import List, Tuple, Dict
from dataset import parse_tokens
import dataset

# project utilities
from utils import init_tokenmaps, save_JSONPICKLE_NEW, load_JSONPICKLE_NEW

# TMM-FAST helpers
from call_tmm_fast import load_materials, TMMSpectrum

# ----------------------------
# helpers
# ----------------------------
def normalize_token(tok: str) -> str:
    # unify "Ag__100" → "Ag_100"
    t = tok.replace("__", "_")
    # guard accidental multiple underscores
    while "__" in t:
        t = t.replace("__", "_")
    return t

def pad_batch(token_id_lists: List[List[int]], pad_id: int) -> torch.LongTensor:
    max_len = max(len(x) for x in token_id_lists) if token_id_lists else 0
    out = torch.full((len(token_id_lists), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(token_id_lists):
        out[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
    return out

def seq_to_ids(seq_tokens: List[str], token_to_idx: Dict[str, int], eos_token: str) -> List[int]:
    ids = []
    for t in seq_tokens:
        t = normalize_token(t)
        if t == eos_token:
            break
        if t in token_to_idx:
            ids.append(token_to_idx[t])
        # silently drop unknowns (e.g., empty tokens from weird splits)
    return ids

def chunk_iter(it, chunk_size):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == chunk_size:
            yield buf
            buf = []
    if buf:
        yield buf

# ----------------------------
# main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-csv", type=str,
                        default=r'd:\Profile\a3536\Eigene Dateien\GitHub\ColorAppearanceToolbox\Diffusion\data\ma',
                        help="CSV with a 'structure' column listing token sequences")
    parser.add_argument("--outdir", type=str, default=c.PATH_RUN, help="Where to save outputs")
    parser.add_argument("--batch", type=int, default=256, help="TMM batch size")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Compute device")
    parser.add_argument("--save-every", type=int, default=int(256*500), help="Periodic checkpoint (samples)")
    parser.add_argument("--files", type=int, nargs="*", default=list(range(2, 10)), help="Which dataset file indices to generate")
    parser.add_argument("--start", type=int, default=0, help="Row start idx (inclusive)")
    parser.add_argument("--end", type=int, default=-1, help="Row end idx (exclusive), -1 for end")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Config: wavelength grid, materials list, tokens
    c.THICKNESSES = np.arange(c.THICKNESS_MIN, c.THICKNESS_MAX + 1, c.THICKNESS_STEPS)
    c.WAVELENGTHS = np.arange(c.WAVELENGTH_MIN, c.WAVELENGTH_MAX + 1, c.WAVELENGTH_STEPS)

    # Build token vocabulary from materials × thicknesses (e.g., "TiO2_60")
    c.MATERIAL_LIST = [item[:-4] for item in sorted(os.listdir(c.PATH_MATERIALS)) if not item.startswith('XX')]
    base_tokens = [f"{m}_{t}" for m, t in itertools.product(c.MATERIAL_LIST, c.THICKNESSES)]
    save_JSONPICKLE_NEW(args.outdir, base_tokens, 'tokens')  # for traceability

    # 2) Token maps (EOS/PAD/MSK) and nk data
    tokens, token_to_idx, idx_to_token, EOS_TOKEN, PAD_TOKEN, MSK_TOKEN, eos_idx, pad_idx, msk_idx = init_tokenmaps(c.PATH_DATA)
    
    nk_dict = load_materials(c)  # reads from PATH_MATERIALS and interpolates to c.WAVELENGTHS :contentReference[oaicite:10]{index=10}

    # 3) Choose device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    TMM_RAT = TMMSpectrum(nk_dict, idx_to_token, device=device).to(device)
    eos = eos_idx
    pad = pad_idx
    msk = msk_idx
    # 1) spectral / angle setup
    degree = np.pi / 180
    theta  = torch.tensor(c.INCIDENCE_ANGLE * degree,
                          device=device, dtype=torch.complex128).unsqueeze(0)        # [1]
    wl_tensor = torch.tensor(c.WAVELENGTHS, dtype=torch.complex128, device=device)    
    # Loop over the same set of “files” you were generating before
    for datafile in args.files:
        # 4) Load source CSV once (same as your RayFlare script)
        # df = pd.read_csv(os.path.join(args.src_csv, f"data_train_{datafile}.csv"), sep=",")
        df = pd.read_csv(os.path.join(args.src_csv, f"data_test.csv"), sep=",")
        series = df.loc[:, "structure"]
        series = series.apply(ast.literal_eval)
        all_samples = series.tolist()
        if args.end == -1:
            samples = all_samples[args.start:]
        else:
            samples = all_samples[args.start:args.end]
        print(f"[file {datafile}] building stacks from {len(samples)} sequences")

        # Save and reload sampled stacks (keeps parity w/ your original pipeline)
        # save_JSONPICKLE_NEW(args.outdir, samples, f'samples_train_{datafile}')
        # samples_this = load_JSONPICKLE_NEW(args.outdir, f'samples_train_{datafile}')
        save_JSONPICKLE_NEW(args.outdir, samples, f'sample_test')
        samples_this = load_JSONPICKLE_NEW(args.outdir, f'sample_test')

        all_spectra: List[torch.Tensor] = []
        all_labels:  List[torch.Tensor] = []

        print("[TMM_FAST] starting batched simulations…")

        processed = 0
        for batch_stacks in chunk_iter(samples_this, args.batch):
            # 4a) convert token strings to ids and pad
            id_lists = [seq_to_ids(seq, token_to_idx, EOS_TOKEN) for seq in batch_stacks]
            if not id_lists:
                continue
            stacks_ids = pad_batch(id_lists, pad_idx)  # [B, S]
            stacks_ids = stacks_ids.to(device)
            # 4b) run TMM_FAST
            RAT = TMM_RAT(stacks_ids, wl_tensor, theta, eos=eos, pad=pad, msk=msk)  # → [B, 3*W]
            
            # 4c) collect outputs and labels (as in original)
            B = stacks_ids.size(0)
            for b in range(B):
                all_spectra.append(RAT[b].detach().cpu().to(torch.float16))
                all_labels.append(torch.tensor(id_lists[b], dtype=torch.long))

            processed += B
            if args.save_every > 0 and processed % args.save_every == 0:
                # torch_outfile = os.path.join(args.outdir, f"data_train_{datafile}.pt")
                torch_outfile = os.path.join(args.outdir, f"data_test.pt")
                torch.save((all_spectra, all_labels), torch_outfile)
                print(f"💾 checkpoint: {processed} samples → {torch_outfile}")

        # final save
        # torch_outfile = os.path.join(args.outdir, f"data_train_{datafile}.pt")
        torch_outfile = os.path.join(args.outdir, f"data_test.pt")
        torch.save((all_spectra, all_labels), torch_outfile)
        print(f"✅ DONE: saved {len(all_spectra)} spectra to: {torch_outfile}")
        break

if __name__ == "__main__":
    main()
