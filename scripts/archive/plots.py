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
    # spectrum_mae = np.mean(np.absolute(np.concatenate([RAT_pred[10:81],RAT_pred[171*1+10:171*1+81],RAT_pred[171*2+10:171*2+81]])
    #                             -np.concatenate([RAT_tar[10:81],RAT_tar[171*1+10:171*1+81],RAT_tar[171*2+10:171*2+81]])))
    # spectrum_mae = np.mean(np.absolute(np.concatenate([RAT_pred[10:81],RAT_pred[171*2+10:171*2+81]])
    #                             -np.concatenate([RAT_tar[10:81],RAT_tar[171*2+10:171*2+81]])))
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
    
import torch           # NEW  – needed when the dataset returns tensors
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class MemmapSpectra(Dataset):
    def __init__(self, npy_file: str):
        self.mm = np.load(npy_file, mmap_mode="r", allow_pickle=False)   # read-only mem-map
    def __len__(self):              return self.mm.shape[0]
    def __getitem__(self, idx):     # only the spectrum (as tensor)
        return torch.from_numpy(self.mm[idx])        # [513]
    
from torch.utils.data import get_worker_info

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

def train_data_comp(
                c,
                data,
                ds_search = None,          # NEW (optional)
                ds_search_precalc = None,
                RAT_tar_mean: np.ndarray | None = None,
                sample_plots = False,
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
    
    import pandas as pd
    MAE_scatter = dict()
    if ds_search_precalc is not None:
        from main import _prepare_token_maps
        idx2tk, mat2id, pad_idx, sos_idx, eos_idx = _prepare_token_maps(c)
        if isinstance(ds_search_precalc, str):
            ds_search_precalc_file = torch.load(ds_search_precalc)
        else:
            ds_search_precalc_file = ds_search_precalc
    for number, _ in enumerate(data):
        # print(number)
        stack_tar, stack_pred, RAT_tar, RAT_pred, ACC, MAE = data[number].values()
        # ---------------------------------------------------------------
        # 1) (Optional) find closest training spectrum
        # ---------------------------------------------------------------
        RAT_closest = None
        mae_closest = None
        if ds_search is not None:
            idx, RAT_closest, mae_closest = find_closest_mae(
                RAT_tar, ds_search, batch_size=int(10*8192))
        RAT_closest, SEQ_closest = ds_search_precalc_file[0][number], ds_search_precalc_file[1][number]
        SEQ_closest = [idx2tk[toke] for toke in SEQ_closest.tolist() if toke not in [1100, 1101, 1102]]
        mae_closest = np.mean(np.absolute(RAT_tar - RAT_closest.tolist()))
        spectrum_mae = np.mean(np.abs(RAT_pred - RAT_tar))
        
        # MAE_scatter[number] = [number,spectrum_mae,mae_closest]
        MAE_scatter[number] = {'number': number,'spectrum_mae': spectrum_mae,'mae_closest': mae_closest}
    
        if sample_plots and number%int(len(data)/5)==0:
            # ---------------------------------------------------------------
            # 2) plotting – original curves plus optional “closest train”
            # ---------------------------------------------------------------
            colormap = plt.cm.tab20(range(8))   # a few extra colours
            wls = np.arange(c.WAVELENGTH_MIN,
                            c.WAVELENGTH_MAX + 1,
                            c.WAVELENGTH_STEPS)
        
            plt.figure(figsize=(10,6))
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
        
            # ---------------------------------------------------------------
            # 3) cosmetics (titles, text blocks, etc.) – mostly unchanged
            # ---------------------------------------------------------------
            plt.title(f'Prediction using OptoLlama [model: {c.RUN_NAME} epoch: {c.RESUME_EPOCH}]',
                      fontsize=12)
            plt.xlabel('Wavelength [nm]', fontsize=10)
            plt.ylabel('R - A - T', fontsize=10)
            plt.legend(loc='center right', fontsize=10)
        
            # compact helper
            def _strip(seq): return [t for t in seq if t not in ("<PAD>", "<SOS>", "<EOS>")]
        
            stack_tar  = _strip(stack_tar)
            stack_pred = _strip(stack_pred)
        
            # → Target materials / thicknesses
            plt.text(0.24, -0.12,
                     "Target:\n" + "\n".join(f"{m.split('__')[0]}" for m in stack_tar),
                     ha='left', fontsize=10, transform=plt.gca().transAxes,
                     horizontalalignment='left',verticalalignment='top',)
            plt.text(0.46, -0.16,
                     "\n".join(f"{m.split('__')[1]}" for m in stack_tar),
                     ha='right', fontsize=10, transform=plt.gca().transAxes,
                     horizontalalignment='left',verticalalignment='top',)
        
            # → Prediction
            plt.text(0.50, -0.12,
                     "Prediction:\n" + "\n".join(f"{m.split('__')[0]}" for m in stack_pred),
                     ha='left', fontsize=10, transform=plt.gca().transAxes,
                     horizontalalignment='left',verticalalignment='top',)
            plt.text(0.72, -0.16,
                     "\n".join(f"{m.split('__')[1]}" for m in stack_pred),
                     ha='right', fontsize=10, transform=plt.gca().transAxes,
                     horizontalalignment='left',verticalalignment='top',)
            
            # → Trainset
            plt.text(0.76, -0.12,
                     "Closest in Train:\n" + "\n".join(f"{m.split('__')[0]}" for m in SEQ_closest),
                     ha='left', fontsize=10, transform=plt.gca().transAxes,
                     horizontalalignment='left',verticalalignment='top',)
            plt.text(0.98, -0.16,
                     "\n".join(f"{m.split('__')[1]}" for m in SEQ_closest),
                     ha='right', fontsize=10, transform=plt.gca().transAxes,
                     horizontalalignment='left',verticalalignment='top',)
        
            # → Key figures
            spectrum_mae = np.mean(np.abs(RAT_pred - RAT_tar))
            # key_block  = f"Key: {number}\n"
            # key_block += f"MAE (pred vs tgt):  {spectrum_mae:.4f}\n"
            # if mae_closest is not None:
            #     key_block += f"MAE (train vs tgt):  {mae_closest:.4f}\n"
            # key_block += f"Token accuracy:     {ACC:.2f}"
            # plt.text(0.3, 1.1, key_block,
            #          ha='left', fontsize=9, transform=plt.gca().transAxes)
            
            key_block_str = "Key: \n\nMAE (prediction): \nMAE (trainset): \n\nAccuracy:"
            plt.text(-0.05, -0.12, key_block_str,
                     ha='left', fontsize=10, transform=plt.gca().transAxes,
                     horizontalalignment='left',verticalalignment='top',)
            
            key_block_float = f"{number} \n\n{spectrum_mae:.4f} \n{mae_closest:.4f} \n\n{ACC:.2f}"
            plt.text(0.12, -0.12, key_block_float,
                     ha='left', fontsize=10, transform=plt.gca().transAxes,
                     horizontalalignment='left',verticalalignment='top',)
            
            # plt.tight_layout()
            plt.show()
            
    return MAE_scatter

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

#%%
data = [['Al2O3__48','Al__283','Ge__249','ZnSe__251','AlN__254'],
['Al2O3__49','Al__280','AlN__255','ZnSe__251'],
['Al2O3__55','Al__273','AlN__252','ZnSe__251'],
['Al2O3__70','Al__303','AlN__253','ZnSe__251'],
['Al2O3__65','Al__312','AlN__245','ZnSe__251'],
['Al2O3__68','Al__306','AlN__246','ZnSe__251'],
['Al2O3__58','Al__302','AlN__247','ZnSe__251'],
['Al2O3__48','Al__287','AlN__248','MgF2__247'],
['Al2O3__61','Al__289','AlN__250','MgF2__245'],
['Al2O3__57','Al__294','AlN__249','ZnO__246']]

#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import ast
from collections import Counter

def load_block(filepath, block_id=0):
    """
    Reads the file at `filepath`, finds the line matching block_id,
    and returns the list of (material, thickness) rows for that block.
    """
    with open(filepath, 'r') as f:
        lines = f.read().splitlines()
    # find the line equal to the block label
    idx = lines.index(str(block_id))
    # the data is two lines below that label
    return ast.literal_eval(lines[idx + 2])

def make_plot(data, figsize=(12,6), output_path=None):
    """
    Given a list of lists [(mat, value), ...] for each stack,
    creates the inverted-y, offset-marker plot.
    """
    # extract sequences and unique sequence list
    sequences = [tuple(mat for mat,_ in row) for row in data]
    unique_seqs = list(dict.fromkeys(sequences))
    # assign one distinct color per sequence
    cmap_seq = plt.cm.tab10(np.linspace(0,1,len(unique_seqs)))
    seq_color = {seq: cmap_seq[i] for i,seq in enumerate(unique_seqs)}

    # find max layer depth
    max_layers = max(len(row) for row in data)
    layers = np.arange(1, max_layers+1)

    # collect all materials for marker/color mapping
    all_materials = sorted({mat for row in data for mat,_ in row})
    marker_list = ['o','s','^','D','P','X','*','v','<','>','p','8']
    cmap_mat = plt.cm.tab20(np.linspace(0,1,len(all_materials)))
    marker_map = {m: marker_list[i % len(marker_list)] for i,m in enumerate(all_materials)}
    color_map  = {m: cmap_mat[i] for i,m in enumerate(all_materials)}

    # compute majority material at each layer
    majority = {}
    for L in layers:
        mats = [row[L-1][0] for row in data if len(row)>=L]
        majority[L] = Counter(mats).most_common(1)[0][0]

    # compute small offsets for stacks at each layer
    delta = 0.1
    offsets = {}
    for L in layers:
        idxs = [i for i,row in enumerate(data) if len(row)>=L]
        offs = np.linspace(-delta, delta, len(idxs))
        for i,off in zip(idxs, offs):
            offsets[(i, L)] = off

    # start plotting
    fig, ax = plt.subplots(figsize=figsize)

    # draw background lines per sequence
    for row, seq in zip(data, sequences):
        thickness = [val for _,val in row] + [np.nan]*(max_layers - len(row))
        ax.plot(thickness, layers, color=seq_color[seq],
                linewidth=2, alpha=0.7, zorder=1)

    # draw majority‐layer markers on centerline
    for i,row in enumerate(data):
        for L,(mat,val) in enumerate(row, start=1):
            if mat==majority[L]:
                ax.scatter(val, L,
                           marker=marker_map[mat],
                           color=color_map[mat],
                           s=60, edgecolors='black', zorder=2)

    # draw outlier markers with vertical offset
    for i,row in enumerate(data):
        for L,(mat,val) in enumerate(row, start=1):
            if mat!=majority[L]:
                y = L #+ offsets[(i,L)]
                ax.scatter(val, y,
                           marker=marker_map[mat],
                           color=color_map[mat],
                           s=60, edgecolors='black', zorder=3)

    # invert y-axis so layer 1 is at the top
    ax.set_yticks(layers)
    ax.set_ylim(max_layers+0.5, 0.5)
    ax.grid(True)
    ax.set_xlabel('Thickness Value')
    ax.set_ylabel('Layer Number')
    ax.set_title(f'Predicted Thickness Profiles (Block {block})')

    # build combined legend
    seq_handles = [
        plt.Line2D([], [], color=seq_color[seq], linewidth=2, label='-'.join(seq))
        for seq in unique_seqs
    ]
    mat_handles = [
        plt.Line2D([], [], marker=marker_map[m], color=color_map[m],
                   linestyle='None', markersize=6, markeredgecolor='black', label=m)
        for m in all_materials
    ]
    ax.legend(handles=seq_handles + mat_handles,
              title='Sequences & Materials',
              bbox_to_anchor=(1.05,1), loc='upper left')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()

if __name__ == '__main__':
    import sys
    fp = r'D:\Profile\a3536\Nextcloud\PhD - HEIBRiDS\Conferences\20250801_NatureMachineIntelligence\results\MC00\predictions_10.txt'
    for block in range(10):    
        # block = 2
        if len(sys.argv) > 1:
            fp = sys.argv[1]            # first arg can override file path
        if len(sys.argv) > 2:
            block = int(sys.argv[2])    # second arg can override block ID
        data = load_block(fp, block_id=block)
        make_plot(data)

