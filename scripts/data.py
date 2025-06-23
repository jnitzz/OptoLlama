# data.py
"""
Dataset and sequence-preparation utilities for Opto-Llama/Optical-GPT.
"""
from __future__ import annotations

import os
import math
from functools import partial
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, distributed, random_split

__all__ = [
    "SinglePTDataset",
    "parse_material_and_thickness",
    "pad_collate_fn",
    "create_masks",
    "build_dataloaders",
]

# -----------------------------------------------------------------------------#
# 1.  Dataset                                                                  #
# -----------------------------------------------------------------------------#


class SinglePTDataset(Dataset[Tuple[Tensor, Tensor]]):
    """
    A thin `torch.utils.data.Dataset` wrapper around a single *.pt* file that
    contains two tensors:

    1. `all_spectra`       – shape ``[N, input_dim]``
    2. `all_label_tokens`  – shape ``[N, L]`` (L = variable sequence length)
    """

    def __init__(self, pt_file: str) -> None:
        super().__init__()
        if not os.path.isfile(pt_file):
            raise FileNotFoundError(pt_file)
        print(f"Loading spectra/tokens from ➜  {pt_file}")
        self.all_spectra, self.all_label_tokens = torch.load(pt_file)
        self.all_spectra, self.all_label_tokens = self.all_spectra[:1000], self.all_label_tokens[:1000]

    # --------------------------------------------------------------------- #
    # PyTorch overrides                                                     #
    # --------------------------------------------------------------------- #

    def __len__(self) -> int:  # noqa: D401
        return len(self.all_spectra)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        spectrum: Tensor = self.all_spectra[idx]         # [input_dim]
        tokens:   Tensor = self.all_label_tokens[idx]    # [L]
        return spectrum, tokens


# -----------------------------------------------------------------------------#
# 2.  Token helpers                                                            #
# -----------------------------------------------------------------------------#


def parse_material_and_thickness(
    token_idx: int,
    token_string_map: Mapping[int, str],
    pad_idx: int,
    sos_idx: int,
    eos_idx: int,
    material_to_id: Mapping[str, int],
) -> Tuple[int, float]:
    """
    Maps a token («Ag__100», «<SOS>», …) → ``(material_id, thickness_nm)``.

    Special tokens are mapped to IDs *after* the real materials.

    Returns
    -------
    (material_id, thickness_nm)
    """
    tk_string: str = token_string_map[token_idx]

    # Special tokens
    if token_idx == pad_idx:
        return len(material_to_id), 0.0
    if token_idx == sos_idx:
        return len(material_to_id) + 1, 0.0
    if token_idx == eos_idx:
        return len(material_to_id) + 2, 0.0

    # Regular «MAT__THICKNESS» token
    if "__" not in tk_string:  # defensive fallback
        return 0, 0.0
    mat_str, thick_str = tk_string.split("__", maxsplit=1)
    return material_to_id.get(mat_str, 0), float(thick_str)


# -----------------------------------------------------------------------------#
# 3.  Collate + masking                                                        #
# -----------------------------------------------------------------------------#


def pad_collate_fn(
    batch: Sequence[Tuple[Tensor, Tensor]],
    *,
    max_seq_length: int,
    pad_idx: int,
    sos_idx: int,
    eos_idx: int,
    token_string_map: Mapping[int, str],
    material_to_id: Mapping[str, int],
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    • Inserts «<SOS>» / «<EOS>»  
    • Pads or truncates to ``max_seq_length``  
    • Converts tokens → ``(material_id, thickness)`` pairs

    Returns ``(spectra, material_ids, thicknesses)`` with shapes:

    * spectra        ``[B, input_dim]``
    * material_ids   ``[B, max_seq_length]``
    * thicknesses    ``[B, max_seq_length]``
    """
    spectra, mat_ids, thicks = [], [], []

    for spectrum, tokens in batch:
        # 1) <SOS> / <EOS>
        seq = torch.cat(
            (
                torch.tensor([sos_idx], dtype=tokens.dtype),
                tokens,
                torch.tensor([eos_idx], dtype=tokens.dtype),
            )
        )

        # 2) pad / truncate
        if seq.numel() > max_seq_length:
            seq = seq[:max_seq_length]
        else:
            pad_len = max_seq_length - seq.numel()
            seq = torch.cat((seq, torch.full((pad_len,), pad_idx, dtype=seq.dtype)))

        # 3) map → (id, thickness)
        ids, th = zip(
            *(
                parse_material_and_thickness(
                    tk.item(),
                    token_string_map,
                    pad_idx,
                    sos_idx,
                    eos_idx,
                    material_to_id,
                )
                for tk in seq
            )
        )
        mat_ids.append(torch.tensor(ids, dtype=torch.long))
        thicks.append(torch.tensor(th, dtype=torch.float32))
        spectra.append(spectrum)

    return (
        torch.stack(spectra, dim=0),
        torch.stack(mat_ids, dim=0),
        torch.stack(thicks, dim=0),
    )


def create_masks(
    input_sequence: Tensor, pad_idx: int, n_heads: int
) -> Tuple[Tensor, Tensor]:
    """
    Standard causal + padding masks for multi-head attention.

    Returns
    -------
    attn_mask        : ``[B*n_heads, T, T]``
    key_padding_mask : ``[B, T]`` (True = ignore)
    """
    bsz, t = input_sequence.shape
    device = input_sequence.device

    key_padding_mask = input_sequence.eq(pad_idx)  # [B, T]

    causal_mask = torch.triu(torch.ones((t, t), device=device), diagonal=1).bool()
    attn_mask = causal_mask.unsqueeze(0).expand(bsz * n_heads, t, t)

    return attn_mask, key_padding_mask


# -----------------------------------------------------------------------------#
# 4.  (Optional) Dataloader factory                                            #
# -----------------------------------------------------------------------------#


def build_dataloaders(
    dataset: Dataset[Tuple[Tensor, Tensor]],
    *,
    batch_size: int,
    world_size: int,
    rank: int,
    collate_fn: torch.utils.data.dataloader._collate_fn_t,  # type: ignore
    num_workers: int = 8,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Splits *dataset* 80/20 and constructs **distributed** train/val loaders.
    """
    data_size = len(dataset)
    train_size = int(0.8 * data_size)
    val_size = int(0.2 * data_size)
    # val_size = data_size - train_size

    if rank == 0:
        print(f"Dataset split ➜ train={train_size}, val={val_size}")

    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_sampler = distributed.DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = distributed.DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_dl, val_dl

# -----------------------------------------------------------------------------#
# 5.  Convenience wrapper – *one file → DataLoader*                            #
# -----------------------------------------------------------------------------#
def build_validation_loader(
    pt_file: str,
    *,
    collate_fn,
    batch_size: int = 32,        # overwrite from the caller if you need
    num_workers: int = 0,
    shuffle: bool = False,
) -> DataLoader:
    """
    Shorthand for

        dataset = SinglePTDataset(pt_file)
        loader  = DataLoader(dataset, batch_size=…, collate_fn=…, …)

    so every script can share the exact same “how to build the val-loader”
    definition.
    """
    dataset = SinglePTDataset(pt_file)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

# make it importable with  `from data import build_validation_loader`
__all__.append("build_validation_loader")
