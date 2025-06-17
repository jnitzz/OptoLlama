# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 16:07:09 2025

@author: a3536
"""
import numpy as np
from data import SinglePTDataset
pt_file_val = r'd:\Profile\a3536\Eigene Dateien\GitHub\OptoLlama\data\RF_T63_16m\my_dataset_val.pt'
ds_val = SinglePTDataset(str(pt_file_val))

target = np.concatenate([np.zeros(30),np.ones(30),np.zeros(111),np.zeros(171),np.ones(30),np.zeros(30),np.ones(111)])


mm = np.memmap("/scratch/htc/jschaibl/repos/ColorAppearanceToolbox/RayFlare/data/samples/RF_T63_1m/my_dataset_1m.npy",
               mode="w+",
               dtype=np.float32,
               shape=(n_rows, n_cols))


# convert_pt_to_npy_memmap.py  -----------------------------------------------
import pathlib, torch, numpy as np, tqdm
from numpy.lib.format import open_memmap          # <— crucial difference!

pt_file   = pathlib.Path("/scratch/htc/jschaibl/repos/ColorAppearanceToolbox/RayFlare/data/samples/RF_T63_16m/my_dataset_16m.pt")
spectra, _ = torch.load(pt_file, map_location="cpu")   # list[Tensor]
n_rows = len(spectra)
n_cols = spectra[0].numel()                         # 513

out_file = pt_file.with_suffix("/scratch/htc/jschaibl/repos/ColorAppearanceToolbox/RayFlare/data/samples/RF_T63_16m/my_dataset_16m.npy")
mm = open_memmap("/scratch/htc/jschaibl/repos/ColorAppearanceToolbox/RayFlare/data/samples/RF_T63_16m/my_dataset_16m.npy", mode="w+", dtype="float32", shape=(n_rows, n_cols))

for i, spec in enumerate(tqdm.tqdm(spectra, desc="writing", unit="spec")):
    mm[i] = spec.reshape(-1).numpy().astype(np.float32, copy=False)

mm.flush()      # header + data now on disk
print(f"✅  wrote numeric .npy mem-map → {out_file}")

_, train_tokens = torch.load(pt_file, map_location="cpu")
n_rows = len(train_tokens)
n_cols = train_tokens[0].numel() 
# out_file = pt_file.with_suffix("/scratch/htc/jschaibl/repos/ColorAppearanceToolbox/RayFlare/data/samples/RF_T63_16m/my_dataset_16m_tokens.npy")
mm = open_memmap("/scratch/htc/jschaibl/repos/ColorAppearanceToolbox/RayFlare/data/samples/RF_T63_16m/my_dataset_16m_tokens.npy", mode="w+", dtype="float16", shape=(n_rows, n_cols))

for i, toks in enumerate(tqdm.tqdm(train_tokens, desc="writing", unit="spec")):
    mm[i] = toks.reshape(-1).numpy().astype(np.float16, copy=False)

mm.flush()      # header + data now on disk
print(f"✅  wrote numeric .npy mem-map → {out_file}")

########################
import torch, numpy as np, pathlib, tqdm
from numpy.lib.format import open_memmap

pt_file   = pathlib.Path("my_dataset_train.pt")
spectra, train_tokens = torch.load(pt_file, map_location="cpu")   # list[list[int]]

pad_id    = 0                 # whatever you use for <PAD>
max_len   = 20                # longest sequence you ever expect
n_rows    = len(train_tokens)

mm = open_memmap("/scratch/htc/jschaibl/repos/ColorAppearanceToolbox/RayFlare/data/samples/RF_T63_16m/my_dataset_16m_tokens.npy",
                     mode="w+",
                     dtype=np.int16,            # fits vocab < 32 768
                     shape=(n_rows, max_len))

for i, seq in enumerate(tqdm.tqdm(train_tokens, desc="tokens", unit="seq")):
    a = np.asarray(seq[:max_len], dtype=np.int16)          # clip if >max_len
    mm[i, :a.size] = a
    if a.size < max_len:                                   # pad right side
        mm[i, a.size:] = pad_id

mm.flush()
print("✅  wrote padded tokens →", tok_mm.filename)


import torch, numpy as np, pathlib, tqdm
from numpy.lib.format import open_memmap

pt_file   = pathlib.Path("my_dataset_train.pt")
spectra, tokens = torch.load(pt_file, map_location="cpu")   # list[list[int]]

pad_id    = 0                 # whatever you use for <PAD>
max_len   = 20                # longest sequence you ever expect
n_rows    = len(tokens)

tok_mm = open_memmap(pt_file.with_suffix(".tokens.npy"),
                     mode="w+",
                     dtype=np.int16,            # fits vocab < 32 768
                     shape=(n_rows, max_len))

for i, seq in enumerate(tqdm.tqdm(tokens, desc="tokens", unit="seq")):
    a = np.asarray(seq[:max_len], dtype=np.int16)          # clip if >max_len
    tok_mm[i, :a.size] = a
    if a.size < max_len:                                   # pad right side
        tok_mm[i, a.size:] = pad_id

tok_mm.flush()
print("✅  wrote padded tokens →", tok_mm.filename)
