# -*- coding: utf-8 -*-
"""
Created on Wed May  7 15:10:49 2025

@author: a3536
"""
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_accuracy(c, accuracies):
    # accuracies = [i[0] for i in samples]
    plt.figure(figsize=(8, 6))
    plt.hist(accuracies, bins=20, edgecolor='black')
    plt.title(f'Distribution of Per-Example Accuracy [model: {c.RUN_NAME} epoch: {c.RESUME_EPOCH}]')
    plt.xlabel('Accuracy score [20 bins]')
    plt.ylabel('Count')
    # Save the figure in the model directory
    hist_path = os.path.join(c.PATH_RUN, f'{c.RUN_NAME}_{c.TARGET}_{c.RESUME_EPOCH}_distribution_accuracy.svg')
    plt.savefig(hist_path)
    print(f"Accuracy distribution plot saved to: {hist_path}")

    # If running locally and you want to see the plot:
    plt.show()
    
def plot_mse(c, mses):
    # mses = [i[3] for i in samples]
    plt.figure(figsize=(8, 6))
    plt.hist(mses, bins=100,edgecolor='black')
    plt.title(f'Distribution of Mean Squared Error (MSE) [model: {c.RUN_NAME} epoch: {c.RESUME_EPOCH}]')#  [model: {model_folder}]')
    plt.xlabel('MSE [100 bins]')
    plt.ylabel('Count')
    plt.text(0.95, 0.95, f"mean MSE: {np.mean(mses):.3f}", ha='right', fontsize=10, transform=plt.gca().transAxes)
    hist_path = os.path.join(c.PATH_RUN, f'{c.RUN_NAME}_{c.TARGET}_{c.RESUME_EPOCH}_distribution_mse.svg')
    plt.savefig(hist_path)
    print(f"Accuracy distribution plot saved to: {hist_path}")
    plt.show()

def plot_mae(c, maes):
    # mses = [i[3] for i in samples]
    plt.figure(figsize=(8, 6))
    plt.hist(maes, bins=100,edgecolor='black')
    plt.title(f'Distribution of Mean Absolute Error (MAE) [model: {c.RUN_NAME} epoch: {c.RESUME_EPOCH}]')#  [model: {model_folder}]')
    plt.xlabel('MAE [100 bins]')
    plt.ylabel('Count')
    plt.text(0.95, 0.95, f"mean MAE: {np.mean(maes):.3f}", ha='right', fontsize=10, transform=plt.gca().transAxes)
    hist_path = os.path.join(c.PATH_RUN, f'{c.RUN_NAME}_{c.TARGET}_{c.RESUME_EPOCH}_distribution_mae.svg')
    plt.savefig(hist_path)
    print(f"Accuracy distribution plot saved to: {hist_path}")
    plt.show()
    
def plot_samples(c, RAT_pred, RAT_tar, stack_pred, stack_tar, ACC, MSE, number, RAT_tar_mean = None):
    def _strip(seq):
        return [t for t in seq if t not in ("<PAD>", "<SOS>", "<EOS>")]
    stack_tar = _strip(stack_tar)
    stack_pred = _strip(stack_pred)
    colormap = plt.cm.tab20(range(6))
    wls = np.arange(c.WAVELENGTH_MIN,c.WAVELENGTH_MAX+1,c.WAVELENGTH_STEPS)
    plt.plot(wls,RAT_pred[:171], label = 'Prediction (R)', color=colormap[1])
    plt.plot(wls,RAT_tar[:171], label = 'Target (R)', color=colormap[0])
    if RAT_tar_mean is not None:
        plt.plot(wls,RAT_tar_mean[:171], '--', label = 'Target mean(R)', color=colormap[0])
    plt.plot(wls,RAT_pred[171:2*171], label = 'Prediction (A)', color=colormap[3])
    plt.plot(wls,RAT_tar[171:2*171], label = 'Target (A)', color=colormap[2])
    if RAT_tar_mean is not None:
        plt.plot(wls,RAT_tar_mean[171:2*171], '--', label = 'Target mean(A)', color=colormap[2])
    plt.plot(wls,RAT_pred[2*171:], label = 'Prediction (T)', color=colormap[5])
    plt.plot(wls,RAT_tar[2*171:], label = 'Target (T)', color=colormap[4])
    if RAT_tar_mean is not None:
        plt.plot(wls,RAT_tar_mean[2*171:], '--', label = 'Target mean(T)', color=colormap[4])
    plt.title(f'Prediction using OptoLLama [model: {c.RUN_NAME} epoch: {c.RESUME_EPOCH}]', fontsize=10)
    plt.legend(loc = 'upper right')
    plt.xlabel('Wavelengths [nm]', fontsize=10)
    plt.ylabel('Reflectance (R), Transmissance (T), Absorptance (A)', fontsize=10)
    
    # Automatically position the target and prediction text    
    text = '\n'.join(f'{material.split("__")[0]}' for material in stack_tar)
    text2 = '\n'.join(f'{material.split("__")[1]}' for material in stack_tar)
    plt.text(0.02, 1.1, f"Target:\n- - - - - - - - - - - - - -\n{text}\n- - - - - - - - - - - - - -" , ha='left', fontsize=8, transform=plt.gca().transAxes)
    plt.text(0.24, 1.1, f"\n{text2}\n" , ha='right', fontsize=8, transform=plt.gca().transAxes)
        
    text = '\n'.join(f'{material.split("__")[0]}' for material in stack_pred)
    text2 = '\n'.join(f'{material.split("__")[1]}' for material in stack_pred)
    plt.text(0.76, 1.1, f"Prediction:\n- - - - - - - - - - - - - -\n{text}\n- - - - - - - - - - - - - -", ha='left', fontsize=8, transform=plt.gca().transAxes)
    plt.text(0.98, 1.1, f"\n{text2}\n" , ha='right', fontsize=8, transform=plt.gca().transAxes)
    
    # Automatically position the accuracy, MAE, and key text
    # spectrum_rss = np.sum(np.square(RAT_pred - RAT_tar))
    spectrum_mae = np.mean(np.absolute(RAT_pred - RAT_tar))
    spectrum_mae = np.mean(np.absolute(np.concatenate([RAT_pred[10:81],RAT_pred[171*1+10:171*1+81],RAT_pred[171*2+10:171*2+81]])
                                -np.concatenate([RAT_tar[10:81],RAT_tar[171*1+10:171*1+81],RAT_tar[171*2+10:171*2+81]])))
    spectrum_mae = np.mean(np.absolute(np.concatenate([RAT_pred[10:81],RAT_pred[171*2+10:171*2+81]])
                                -np.concatenate([RAT_tar[10:81],RAT_tar[171*2+10:171*2+81]])))
    plt.text(0.36, 1.10, "- - - - - - - - - - - - - -\nKey#:\nMAE:\nAccuracy:\n- - - - - - - - - - - - - -", ha='left', fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.64, 1.10, f"\n{number}\n{spectrum_mae:.2f}\n{ACC:.2f}\n", ha='right', fontsize=10, transform=plt.gca().transAxes)
    
    plt.show()

def plot_mae_comparison(c, model_maes_dict):
    """
    Plot MAE distributions for multiple models.

    Parameters:
    - c: config object with RUN_NAME, RESUME_EPOCH, PATH_RUN
    - model_maes_dict: dict where keys are model names (str), values are lists or arrays of MAEs
    """
    plt.figure(figsize=(8, 6))
    max_vals = max([max(maes) for i, maes in model_maes_dict.items()])
    # Plot each model's MAE histogram
    for model_name, maes in model_maes_dict.items():
        plt.hist(maes, bins=np.linspace(0,max_vals,100), alpha=0.5, label=f'{model_name} (mean: {np.mean(maes):.3f})', edgecolor='black', linewidth=0.5)

    plt.title('MAE Distribution Comparison')
    plt.xlabel('MAE [100 bins]')
    plt.ylabel('Count')
    plt.legend(loc='upper right', fontsize=9)

    # Save and show
    hist_path = os.path.join(c.PATH_RUN, f'{c.RUN_NAME}_{c.TARGET}_{c.RESUME_EPOCH}_mae_comparison.svg')
    plt.tight_layout()
    plt.savefig(hist_path, dpi=400)
    print(f"MAE comparison plot saved to: {hist_path}")
    plt.show()

def plot_mse_comparison(c, model_mses_dict):
    """
    Plot MSE distributions for multiple models.

    Parameters:
    - c: config object with RUN_NAME, RESUME_EPOCH, PATH_RUN
    - model_mses_dict: dict where keys are model names (str), values are lists or arrays of MSEs
    """
    plt.figure(figsize=(8, 6))
    max_vals = max([max(mses) for i, mses in model_mses_dict.items()])
    # Plot each model's MAE histogram
    for model_name, mses in model_mses_dict.items():
        plt.hist(mses, bins=np.linspace(0,max_vals,100), alpha=0.5, label=f'{model_name} (mean: {np.mean(mses):.3f})', edgecolor='black', linewidth=0.5)

    plt.title('MSE Distribution Comparison')
    plt.xlabel('MSE [100 bins]')
    plt.ylabel('Count')
    plt.legend(loc='upper right', fontsize=9)

    # Save and show
    hist_path = os.path.join(c.PATH_RUN, f'{c.RUN_NAME}_{c.TARGET}_{c.RESUME_EPOCH}_mse_comparison.svg')
    plt.tight_layout()
    plt.savefig(hist_path, dpi=400)
    print(f"MSE comparison plot saved to: {hist_path}")
    plt.show()

def plot_acc_comparison(c, model_accs_dict):
    """
    Plot ACC distributions for multiple models.

    Parameters:
    - c: config object with RUN_NAME, RESUME_EPOCH, PATH_RUN
    - model_accs_dict: dict where keys are model names (str), values are lists or arrays of ACCs
    """
    plt.figure(figsize=(8, 6))
    max_vals = max([max(accs) for i, accs in model_accs_dict.items()])
    # Plot each model's MAE histogram
    for model_name, accs in model_accs_dict.items():
        plt.hist(accs, bins=np.linspace(0,max_vals,100), alpha=0.5, label=f'{model_name} (mean: {np.mean(accs):.3f})', edgecolor='black', linewidth=0.5)

    plt.title('Accuracy Distribution Comparison')
    plt.xlabel('Accuracy [100 bins]')
    plt.ylabel('Count')
    plt.legend(loc='upper right', fontsize=9)

    # Save and show
    hist_path = os.path.join(c.PATH_RUN, f'{c.RUN_NAME}_{c.TARGET}_{c.RESUME_EPOCH}_acc_comparison.svg')
    plt.tight_layout()
    plt.savefig(hist_path, dpi=400)
    print(f"ACC comparison plot saved to: {hist_path}")
    plt.show()
    
import matplotlib.pyplot as plt
import numpy as np
import torch           # NEW  – needed when the dataset returns tensors
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np, torch

class MemmapSpectra(Dataset):
    def __init__(self, npy_file: str):
        self.mm = np.load(npy_file, mmap_mode="r", allow_pickle=False)   # read-only mem-map
    def __len__(self):              return self.mm.shape[0]
    def __getitem__(self, idx):     # only the spectrum (as tensor)
        return torch.from_numpy(self.mm[idx])        # [513]
    
import numpy as np, torch
from torch.utils.data import Dataset, get_worker_info

def _read_npy_shape(path: str):
    """
    Return (shape, dtype) for any .npy version (v1.0 – v3.0) without
    mem-mapping the data.  Works no matter whether NumPy returns
    2 or 3 values from read_array_header_*().
    """
    import numpy as np
    with open(path, "rb") as fh:
        major, minor = np.lib.format.read_magic(fh)
        if major == 1:
            header_dict, *__ = np.lib.format.read_array_header_1_0(fh)
        elif major in (2, 3):
            header_dict, *__ = np.lib.format.read_array_header_2_0(fh)
        else:
            raise ValueError(f".npy version {major}.{minor} not supported")
    return header_dict


class LazyMemmapSpectra(Dataset):
    def __init__(self, npy_path: str):
        self.path          = str(npy_path)           # cheap to pickle
        self.shape         = _read_npy_shape(self.path)
        self._mm           = None                    # opened lazily

    def _lazy_open(self):
        if self._mm is None:                         # first use in *this worker*
            self._mm = np.load(self.path, mmap_mode="r")

    def __len__(self):  return self.shape[0]

    def __getitem__(self, idx):
        self._lazy_open()
        return torch.from_numpy(self._mm[idx])

@torch.no_grad()
def find_closest_mae(target: np.ndarray,
                     mm_path,
                     batch_size: int = int(8192),#8192
                     device: str = "cuda",
                     workers: int = 4):

    loader = torch.utils.data.DataLoader(
        LazyMemmapSpectra(mm_path),
        batch_size=batch_size,
        num_workers=workers,     # can be >0 now
        pin_memory=True
    )
    # loader      = DataLoader(dataset,
    #                          batch_size=batch_size,
    #                          num_workers=8,
    #                          pin_memory=True)

    tgt   = torch.as_tensor(target, dtype=torch.float32, device=device)
    best  = float("inf")
    best_spec = None
    best_idx  = -1

    for b, spec in enumerate(loader):            # spec: [B, 513]
        spec = spec.to(device, non_blocking=True)
        mae  = torch.mean(torch.abs(spec - tgt), dim=1)  # [B]

        val, idx = torch.min(mae, dim=0)         # best in this batch
        if val.item() < best:
            best, best_idx = val.item(), b * batch_size + idx.item()
            best_spec      = spec[idx].cpu()     # keep a copy on CPU

    return best_idx, best_spec.numpy(), best     # index, spectrum, MAE

def plot_samples2(
                c,
                RAT_pred: np.ndarray,
                RAT_tar : np.ndarray,
                stack_pred: list[str],
                stack_tar : list[str],
                ACC      : float,
                MSE      : float,
                number   : int,
                *,
                ds_search = None,          # NEW (optional)
                RAT_tar_mean: np.ndarray | None = None,
        ):
    """
    Visualises one sample, *optionally* adding the closest training spectrum
    (w.r.t. MAE in RAT space).

    Parameters
    ----------
    ds_train : Dataset | None
        Any dataset whose __getitem__(idx) returns a tuple with the first item
        being the 3×171-value RAT spectrum. Pass e.g.

            ds_train = SinglePTDataset('…/my_dataset_train.pt')

        and call

            plot_samples(c, RAT_pred, RAT_tar, …, ds_train=ds_train)
    """
    # ---------------------------------------------------------------
    # 1) (Optional) find closest training spectrum
    # ---------------------------------------------------------------
    RAT_closest = None
    mae_closest = None
    if ds_search is not None:
        idx, RAT_closest, mae_closest = find_closest_mae(
            RAT_tar, ds_search, batch_size=int(10*8192))

    # ---------------------------------------------------------------
    # 2) plotting – original curves plus optional “closest train”
    # ---------------------------------------------------------------
    colormap = plt.cm.tab20(range(8))   # a few extra colours
    wls = np.arange(c.WAVELENGTH_MIN,
                    c.WAVELENGTH_MAX + 1,
                    c.WAVELENGTH_STEPS)

    # prediction vs. target
    plt.plot(wls, RAT_pred[:171]      , label='Prediction (R)',  color=colormap[1])
    if RAT_closest is not None:
        plt.plot(wls, RAT_closest[:171]      , '--', label='Closest train (R)', color=colormap[1])
    plt.plot(wls, RAT_tar [:171]      , label='Target     (R)',  color=colormap[0])
    plt.plot(wls, RAT_pred[171:2*171] , label='Prediction (A)',  color=colormap[3])
    if RAT_closest is not None:
        plt.plot(wls, RAT_closest[171:2*171] , '--', label='Closest train (A)', color=colormap[3])
    plt.plot(wls, RAT_tar [171:2*171] , label='Target     (A)',  color=colormap[2])
    plt.plot(wls, RAT_pred[2*171:]    , label='Prediction (T)',  color=colormap[5])
    if RAT_closest is not None:
        plt.plot(wls, RAT_closest[2*171:]    , '--', label='Closest train (T)', color=colormap[5])
    plt.plot(wls, RAT_tar [2*171:]    , label='Target     (T)',  color=colormap[4])

    if RAT_tar_mean is not None:
        plt.plot(wls, RAT_tar_mean[:171]      , '--', label='Target mean (R)', color=colormap[0])
        plt.plot(wls, RAT_tar_mean[171:2*171] , '--', label='Target mean (A)', color=colormap[2])
        plt.plot(wls, RAT_tar_mean[2*171:]    , '--', label='Target mean (T)', color=colormap[4])

    # # NEW – closest train sample (dashed)
    # if RAT_closest is not None:
    #     plt.plot(wls, RAT_closest[:171]      , '--', label='Closest train (R)', color=colormap[1])
    #     plt.plot(wls, RAT_closest[171:2*171] , '--', label='Closest train (A)', color=colormap[3])
    #     plt.plot(wls, RAT_closest[2*171:]    , '--', label='Closest train (T)', color=colormap[5])

    # ---------------------------------------------------------------
    # 3) cosmetics (titles, text blocks, etc.) – mostly unchanged
    # ---------------------------------------------------------------
    plt.title(f'Prediction using OptoLlama [model: {c.RUN_NAME} epoch: {c.RESUME_EPOCH}]',
              fontsize=10)
    plt.xlabel('Wavelength [nm]', fontsize=10)
    plt.ylabel('R - A - T', fontsize=10)
    plt.legend(loc='upper right')

    # compact helper
    def _strip(seq): return [t for t in seq if t not in ("<PAD>", "<SOS>", "<EOS>")]

    stack_tar  = _strip(stack_tar)
    stack_pred = _strip(stack_pred)

    # → Target materials / thicknesses
    plt.text(0.02, 1.10,
             "Target:\n" + "\n".join(f"{m.split('__')[0]}" for m in stack_tar),
             ha='left', fontsize=8, transform=plt.gca().transAxes)
    plt.text(0.24, 1.10,
             "\n".join(f"{m.split('__')[1]}" for m in stack_tar),
             ha='right', fontsize=8, transform=plt.gca().transAxes)

    # → Prediction
    plt.text(0.76, 1.10,
             "Prediction:\n" + "\n".join(f"{m.split('__')[0]}" for m in stack_pred),
             ha='left', fontsize=8, transform=plt.gca().transAxes)
    plt.text(0.98, 1.10,
             "\n".join(f"{m.split('__')[1]}" for m in stack_pred),
             ha='right', fontsize=8, transform=plt.gca().transAxes)

    # → Key figures
    spectrum_mae = np.mean(np.abs(RAT_pred - RAT_tar))
    key_block  = f"Key: {number}\n"
    key_block += f"MAE (pred vs tgt): {spectrum_mae:.4f}\n"
    if mae_closest is not None:
        key_block += f"MAE (train vs tgt): {mae_closest:.4f}\n"
    key_block += f"Token accuracy: {ACC:.2f}"
    plt.text(0.36, 1.10, key_block,
             ha='left', fontsize=9, transform=plt.gca().transAxes)

    # plt.tight_layout()
    plt.show()

import torch, numpy as np
from torch.utils.data import DataLoader
from collections import namedtuple

Nearest = namedtuple("Nearest", "train_idx mae train_tokens")

@torch.no_grad()
def all_closest_mae(test_ds,
                    mm_train_file: str,
                    train_tokens: list,
                    *,
                    batch_test  = 256,
                    batch_train = 8192,
                    device      = "cuda"):
    """
    For every spectrum in `test_ds` return the index in *training* that has the
    lowest mean-squared-error (== L2) and the corresponding tokens.

    Returns
    -------
    list[Nearest]  (same length as test_ds)
    """
    # ---------- 1. bring the *whole* test split onto the GPU in manageable slabs
    results = []

    N_test  = len(test_ds)
    mm_tr   = np.load(mm_train_file, mmap_mode="r")
    N_train = mm_tr.shape[0]

    for t0 in range(0, N_test, batch_test):
        t1  = min(t0 + batch_test, N_test)
        # stack → [T,B] on GPU
        test_batch = torch.stack([test_ds[i][0] for i in range(t0, t1)]).to(device)
        T, D = test_batch.shape

        best_mae  = torch.full((T,), float("inf"), device=device)
        best_idx  = torch.full((T,), -1, dtype=torch.long, device=device)

        # ---------- 2. stream through the (huge) training mem-map
        for s0 in range(0, N_train, batch_train):
            s1 = min(s0 + batch_train, N_train)
            train_chunk = torch.from_numpy(mm_tr[s0:s1]).to(device, non_blocking=True)  # [B,D]

            # pair-wise L2 → mean over D
            diff = train_chunk.unsqueeze(1) - test_batch                # [B,T,D]
            mae  = torch.mean(torch.abs(diff), dim=2)                       # [B,T]

            chunk_best, chunk_arg = torch.min(mae, dim=0)               # per-test
            update = chunk_best < best_mae
            best_mae[update] = chunk_best[update]
            best_idx[update] = chunk_arg[update] + s0                   # global idx

        # ---------- 3. stash results for this slab
        best_mae = best_mae.cpu().numpy()
        best_idx = best_idx.cpu().numpy()
        for local_i, (idx, mae_val) in enumerate(zip(best_idx, best_mae)):
            results.append(
                Nearest(train_idx   = int(idx),
                        mae         = float(mae_val),
                        train_tokens= train_tokens[idx]))
    return results
