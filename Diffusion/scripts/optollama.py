#!/usr/bin/env python

import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import Subset
import tqdm

import config_MD51 as cfg
# import config_DIT006 as cfg
import config_MD16 as cfg
import cli as cli
import dataset
# import model
from model import build_model
# import matplotlib.pyplot as plt

from utils import set_all_seeds, set_torch_options, init_distributed, token_accuracy
from call_tmm_fast import load_materials, TMMSpectrum


def train_loop(arguments: argparse.Namespace, device: str, rank: int, world_size: int, slurm_localid: int) -> None:
    # data
    from utils import init_tokenmaps, save_JSONPICKLE_NEW, load_JSONPICKLE_NEW, masked_mae, unique_length_int_generator
    tokens, token_to_idx, idx_to_token, EOS_TOKEN, PAD_TOKEN, MSK_TOKEN, eos_idx, pad_idx, msk_idx = init_tokenmaps(cfg.PATH_DATA)

    # train_dataset = dataset.SpectraDataset([cfg.PATH_TRAIN_0, cfg.PATH_TRAIN_1, cfg.PATH_TRAIN_2, cfg.PATH_TRAIN_3, cfg.PATH_TRAIN_4, cfg.PATH_TRAIN_5, cfg.PATH_TRAIN_6, cfg.PATH_TRAIN_7, cfg.PATH_TRAIN_8, cfg.PATH_TRAIN_9], tokens, device)
    # train_dataset = dataset.SpectraDataset([cfg.PATH_TRAIN_0, cfg.PATH_TRAIN_1, cfg.PATH_TRAIN_2, cfg.PATH_TRAIN_3], tokens, device)
    train_dataset = dataset.SpectraDataset([cfg.PATH_TRAIN_0], tokens, device)
    val_dataset = dataset.SpectraDataset(cfg.PATH_VALID, tokens, device)
    
    subset_idx = unique_length_int_generator(0e0, len(train_dataset)-1, 10e6)
    train_subset = Subset(train_dataset, subset_idx)
    
    subset_idx = unique_length_int_generator(0e0, len(val_dataset)-1, 1e3)
    val_subset = Subset(val_dataset, subset_idx)
        
    max_stack_depth = train_dataset.get_maximum_depth()
    # 1) spectral / angle setup
    degree = np.pi / 180
    theta  = torch.tensor(cfg.INCIDENCE_ANGLE * degree,
                          device=device, dtype=torch.complex128).unsqueeze(0)        # [1]
    cfg.WAVELENGTHS = np.arange(cfg.WAVELENGTH_MIN,
                              cfg.WAVELENGTH_MAX + 1,
                              cfg.WAVELENGTH_STEPS)
    wl_tensor = torch.tensor(cfg.WAVELENGTHS, dtype=torch.complex128, device=device)    
    nk_dict = load_materials(cfg)
    TMM_RAT = TMMSpectrum(nk_dict, idx_to_token, device=device).to(device)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_subset, shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_subset, shuffle=False)
    train_loader = torch.utils.data.DataLoader(
        train_subset, 
        batch_size=cfg.BATCH, 
        shuffle=False, 
        drop_last=True, 
        sampler=train_sampler,
        pin_memory=True,
        collate_fn=lambda batch: dataset.pad_batch(batch, max_stack_depth, eos_idx, pad_idx)
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset, 
        batch_size=100,   #-val_size/world_size%8
        shuffle=False, 
        drop_last=True, 
        sampler=val_sampler,
        pin_memory=True,
        collate_fn=lambda batch: dataset.pad_batch(batch, max_stack_depth, eos_idx, pad_idx)
    )

    # model
    vocab_size = len(tokens)
    spectrum_sample, _ = train_dataset[0]
    local_model = build_model(
        model_type=getattr(cfg, "ARCH", getattr(cfg, "OL_MODEL", "dit")),
        spectrum_dim=spectrum_sample.shape[-1],
        vocab_size=vocab_size,
        timesteps=cfg.STEPS,
        max_len=spectrum_sample.shape[0],
        max_stack_depth=max_stack_depth,
        mask_idx=msk_idx,
        d_model=cfg.D_MODEL,
        n_blocks=cfg.N_BLOCKS,
        n_heads=cfg.N_HEADS,
        dropout=cfg.DROPOUT,
        idx_to_token=idx_to_token,
        pad_idx=pad_idx,
        eos_idx=eos_idx,
        device=device,
        sample_spectrum=spectrum_sample.unsqueeze(0),  # [1,W,3]
    )
    model_ddp = torch.nn.parallel.DistributedDataParallel(
        local_model,
        device_ids=[slurm_localid], 
        output_device=slurm_localid,
        find_unused_parameters=False
    )
    
    # optimization
    optimizer = torch.optim.Adam(model_ddp.parameters(), lr=1e-4)

    # metrics tracking
    train_losses = torch.zeros(cfg.EPOCHS)
    train_losses_MAE = torch.zeros(cfg.EPOCHS)
    train_losses_H = torch.zeros(cfg.EPOCHS)
    train_losses_CE = torch.zeros(cfg.EPOCHS)
    train_acc = torch.zeros(cfg.EPOCHS)
    val_acc = torch.zeros(cfg.EPOCHS)
    val_MAE = torch.ones(cfg.EPOCHS) * np.inf
    start_epoch = 0
    SUBSTRATE_IDS = torch.tensor(list(range(200,250)), device=device)
    # checkpointing
    checkpoint = f'{cfg.PATH_SAVED}/ol3l-checkpoint.pt'
    if os.path.exists(checkpoint):
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        start_epoch = state['epoch']
        model_ddp.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])
        train_losses = state['train_losses']
        train_acc = state['train_acc']
        val_acc = state['val_acc']
        val_acc_save = np.max(state['val_acc'].numpy())
        try:
            val_MAE = state['val_MAE']
            val_MAE_save = np.min(state['val_MAE'].numpy())
        except:
            val_MAE_save = np.inf
    else:
        val_acc_save = 0.0
        val_MAE_save = np.inf
    
    # optimization loop
    for epoch in range(start_epoch, cfg.EPOCHS):
        if not arguments.notrain:
            model_ddp.train()
            train_loader.sampler.set_epoch(epoch)
            
            if rank == 0:
                train_progressbar = tqdm.tqdm(
                    total=len(train_loader), 
                    desc=f'Epoch {epoch + 1}/{cfg.EPOCHS} train', 
                    leave=True
                )
            
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()
                
                spectra, stacks = batch
                spectra, stacks = spectra.to(device, non_blocking=True), stacks.to(device, non_blocking=True)
                
                
                logits = model_ddp(spectra, stacks) #, loss_scalars
                
                if cfg.MAE:
                # if (i % max(1, 10)) == 0:
                    # Linear warmup for MAE weight
                    if epoch <= cfg.MAE_WARMUP_EPOCHS:
                        w_mae = float(epoch) / float(max(1, cfg.MAE_WARMUP_EPOCHS))
                    else:
                        w_mae = 1.0
                    
                    # logits: [B,S,V]
                    logits = logits.clone()
                    
                    # Build once at init: SUBSTRATE_IDS = torch.tensor(list(range(200,250)), device=...)
                    logits[..., SUBSTRATE_IDS] = -1e9        # forbid EVA anywhere in the stack
                    # # or front-aware:
                    # K_front_block = 8                         # tune 4–16
                    # logits[:, :K_front_block, SUBSTRATE_IDS] = -1e9
                    
                    # keep PAD/MSK out always; you can also block EOS in the first K positions
                    logits[..., pad_idx] = -1e9
                    logits[..., msk_idx] = -1e9
                    # logits[:, :K_front_block, eos_idx] = -1e9
                    
                    tau0, tau_min = 2.0, 0.2
                    tau = max(tau_min, tau0 * (0.85 ** epoch))  # example schedule
                    
                    # logits -> probabilities with temperature
                    probs = torch.softmax(logits / tau, dim=-1)      # [B,S,V]
                    # probs = torch.nn.functional.gumbel_softmax(logits / tau, dim=-1, hard=True)      # [B,S,V]
                    
                    # # confidence and argmax id per position
                    # conf, ids = probs.max(dim=-1, keepdim=True)                # [B,S,1], [B,S,1]
                    # onehot = torch.zeros_like(probs).scatter_(-1, ids, 1.0)
                    # # onehot = torch.nn.functional.one_hot(logits.argmax(-1), num_classes=vocab_size).to(probs.dtype)
                    # # tokens_for_tmm = onehot + probs - probs.detach()
                    
                    # # progress p ∈ [0,1]: how far we are into the ramp
                    # p = min(1.0, (epoch + 1) / max(1, int(0.03 * cfg.EPOCHS)))  # 30% epochs ramp
                    
                    # # global threshold ramps up → becomes stricter over time
                    # th_min, th_max = 0.10, 0.95
                    # thresh_global = th_min + (th_max - th_min) * p
                    
                    # # FRONT-AWARE easing: lower threshold for early positions so they go hard sooner
                    # S = stacks.size(1)
                    # pos = torch.linspace(0, 1, S, device=probs.device).view(1, S, 1)   # 0 at front → 1 at tail
                    # ease = 0.20                                                         # how much easier at front
                    # thresh_pos = thresh_global - ease * (1.0 - pos)                     # early positions: lower thresh
                    
                    # # binary mask of where we trust hard tokens already
                    # use_hard = (conf >= thresh_pos).float()                             # [B,S,1]
                    
                    # # mixed tokens: confident → one-hot, else stay soft
                    # tokens_mix = use_hard * onehot + (1.0 - use_hard) * probs           # [B,S,V]
                    
                    k = 1
                    vals, idxs = (logits / tau).topk(k, dim=-1)                           # [B,S,k]
                    print('\n\n',idxs[-1].T)#,'\n',idxs[-1].T)
                    mask = torch.zeros_like(probs).scatter_(-1, idxs, 1.0)       # [B,S,V]
                    probs_sparse = probs * mask
                    probs_sparse = probs_sparse / probs_sparse.sum(-1, keepdim=True).clamp_min(1e-8)
                    
                    # 2) Convert logits → ids for accuracy, align with targets
                    # pred_ids = logits.argmax(dim=-1) if logits.dim() == 3 else logits  # [B,S]
                    # L = min(stacks.size(1), pred_ids.size(1))
                    # stacks_aligned = stacks[:, :L]
                    # tokens_mix       = pred_ids[:, :L]
                    # print(tokens_mix)
                    tokens_mix = probs_sparse
                    
                    # --- Physics forward on mixed tokens ---
                    res = TMM_RAT(tokens_mix, wl_tensor, theta, eos=eos_idx, pad=pad_idx, msk=msk_idx)
                    
                    # reshape to [B,W,3]
                    B = spectra.size(0)
                    W = wl_tensor.numel()
                    if res.dim() == 3 and res.size(1) == 3:         # [B,3,W] → [B,W,3]
                        predicted_spectra = res.permute(0, 2, 1)
                    elif res.dim() == 2 and res.size(1) == 3*W:     # [B,3W]  → [B,W,3]
                        predicted_spectra = res.view(B, 3, W).permute(0, 2, 1)
                    else:
                        raise RuntimeError(f"Unexpected TMM shape {tuple(res.shape)}")
                    
                    predicted_spectra = torch.nan_to_num(predicted_spectra, nan=0.0, posinf=1.0, neginf=0.0)
                    loss_MAE_BASE = masked_mae(spectra, predicted_spectra)
                    
                    # # (optional) small diversity regulariser you already had
                    # p_dist = probs.clone()
                    # sim_adj = (p_dist[:, 1:, :] * p_dist[:, :-1, :]).sum(-1).mean()
                    # λ_div   = 5e-1
                    # loss_MAE = loss_MAE_BASE + λ_div * sim_adj
                    loss_MAE = loss_MAE_BASE
                    if loss_MAE.dim() > 0:
                        loss_MAE = loss_MAE.mean()
                    # Optional debug: count NaNs from raw TMM every N steps
                    if (i % max(1, cfg.LOG_NAN_EVERY_STEPS)) == 0:
                        n_rats_nans = (~torch.isfinite(predicted_spectra)).sum().item()
                        if rank == 0 and n_rats_nans > 0:
                            print(f"[dbg] step {i} → non-finite entries in RAT: {n_rats_nans}")
                    # print(spectra[0], predicted_spectra[0])
                    
                if cfg.CE:
                    # Linear warmup for MAE weight
                    if epoch <= cfg.CE_WARMUP_EPOCHS:
                        w_ce = 1.0 - min(float(epoch) / float(max(1, cfg.CE_WARMUP_EPOCHS)), 1)
                    else:
                        w_ce = 0.0
                    
                    # Guard the logits → prevents NaNs from propagating into CE 
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    loss_CE_base = torch.nn.NLLLoss(ignore_index=pad_idx)(
                        log_probs.reshape(-1, vocab_size),
                        stacks.reshape(-1)
                    )
                    # # after you compute `logits` (logits) in train loop
                    # probs = torch.softmax(logits, dim=-1)            # [B,S,V]
                    # p_active = 1.0 - (probs[..., eos_idx] + probs[..., pad_idx] + probs[..., msk_idx])  # [B,S]
                    
                    # # optional: increase penalty with position (later layers cost more)
                    # S = p_active.size(1)
                    # pos = torch.arange(S, device=p_active.device).float() / max(S - 1, 1)
                    # w_pos = (0.25 + 0.75 * pos).unsqueeze(0)                   # gentle ramp 0.25→1.0
                    
                    # L_len = (p_active * w_pos).sum(dim=1).mean()               # expected length (weighted)
                    # λ_len = 1e-1                                               # start tiny, tune up
                    
                    # loss_CE = loss_CE_base + λ_len * L_len
                    loss_CE = loss_CE_base
                
                if cfg.H:
                    # # 1) Per-position entropy (encourages each position to keep multiple tokens alive)
                    # logp      = (probs.clamp_min(1e-8)).log()
                    # H_pos     = -(probs * logp).sum(dim=-1).mean()            # scalar
                    
                    # # 2) Sequence-level entropy (encourages different positions to cover the vocab overall)
                    mean_probs = probs.mean(dim=1)                             # [B, V], avg over positions
                    H_seq      = -(mean_probs * mean_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
                    
                    # # 3) Adjacent-position dissimilarity (discourages same token everywhere)
                    # #    High similarity is bad → add to the loss (so it's minimized).
                    # sim_adj   = (probs[:, 1:, :] * probs[:, :-1, :]).sum(dim=-1).mean()
                    
                    # # --- weights (start relatively strong; anneal later) ---
                    # λ_pos  = 1e-3   # per-position entropy
                    # λ_seq  = 1e-3   # sequence-level entropy
                    # λ_adj  = 5e-1   # adjacency similarity penalty (already in your code; keep sign as +)
                    
                    # # --- final diversity term (NOTE THE SIGNS) ---
                    # loss_diversity = (-λ_pos * H_pos) + (-λ_seq * H_seq) + (λ_adj * sim_adj)
                    
                    probs = torch.softmax(logits, dim=-1)
                    # loss_H = -(probs * (probs.clamp_min(1e-12)).log()).sum(dim=-1).mean()
                    # loss_H = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
                    # average probs across sequence
                    mean_probs = probs.mean(dim=1)       # [B, V] average over positions
                    # entropy of this distribution
                    # loss_H = -(mean_probs * mean_probs.clamp_min(1e-8).log()).sum(-1).mean()
                    # loss_H   = (probs[:, 1:, :] * probs[:, :-1, :]).sum(dim=-1).mean()
                    loss_H = H_seq
                    # Linear warmup for entropy (H) weight
                    if epoch <= cfg.H_WARMUP_EPOCHS:
                        w_H = cfg.H_WEIGHT * float(epoch) / float(max(1, cfg.H_WARMUP_EPOCHS))
                    else:
                        w_H = cfg.H_WEIGHT
                    
                if cfg.MAE and cfg.CE and cfg.H:
                    loss = loss_CE + w_mae * loss_MAE + w_H * loss_H
                else:
                    if cfg.MAE and cfg.CE:
                        loss = w_ce * loss_CE + w_mae * loss_MAE
                    elif cfg.MAE and cfg.H:
                        loss = w_H * loss_H + w_mae * loss_MAE
                    else:
                        if cfg.CE:
                            loss = loss_CE
                        if cfg.MAE:
                            loss = loss_MAE #+ λ_div * L_div
                
                # Early NaN diagnostics per loss before weighting (DDP-safe)
                ce_bad  = torch.tensor([float(cfg.CE and (not torch.isfinite(loss_CE)))],  device=spectra.device)
                mae_bad = torch.tensor([float(cfg.MAE and (not torch.isfinite(loss_MAE)))], device=spectra.device)
                torch.distributed.all_reduce(ce_bad,  op=torch.distributed.ReduceOp.MAX)
                torch.distributed.all_reduce(mae_bad, op=torch.distributed.ReduceOp.MAX)
                skip_batch = (ce_bad.item() > 0.0) or (mae_bad.item() > 0.0)
                if skip_batch:
                    if rank == 0:
                        what = ("CE" if ce_bad.item() > 0 else "") + ("+MAE" if mae_bad.item() > 0 else "")
                        print(f"⚠️ non-finite loss ({what}) at epoch {epoch}, step {i} — skipping batch (all ranks)")
                    continue
                
                # --- Backprop (retain_graph to inspect grads)
                loss.backward()
                bad_grad = False
                for p in model_ddp.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        bad_grad = True
                        print('bad_grad, scipping')
                        break
                if bad_grad:
                    optimizer.zero_grad(set_to_none=True)
                    continue  # skip this step safely

                torch.nn.utils.clip_grad_norm_(model_ddp.parameters(), cfg.MAX_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
                log_loss = loss.detach()
                torch.distributed.all_reduce(log_loss)
                log_loss = (log_loss / world_size).item()
                train_losses[epoch] += log_loss
                
# %%
                # import matplotlib.pyplot as plt
                # if (i % max(1, 50)) == 0:
                #     it = 0
                #     stack_pred = logits[int(it)].argmax(dim=-1)
                #     with torch.no_grad():
                #         pred_ids = logits.argmax(-1)
                #         res_hard = TMM_RAT(pred_ids, wl_tensor, theta, eos=eos_idx, pad=pad_idx, msk=msk_idx)
                #         B, W = spectra.size(0), wl_tensor.numel()
                #         pred_spectra_hard = res_hard.permute(0,2,1) if res_hard.dim()==3 else res_hard.view(B,3,W).permute(0,2,1)
                #     plt.plot(spectra[int(it)][:,0].cpu().detach().numpy(), '-', label='target - R')
                #     plt.plot(spectra[int(it)][:,1].cpu().detach().numpy(), '-', label='target - A')
                #     plt.plot(spectra[int(it)][:,2].cpu().detach().numpy(), '-', label='target - T')
                #     plt.plot(pred_spectra_hard[int(it)][:,0].cpu().detach().numpy(), '--', color='C0', label='prediction - R')
                #     plt.plot(pred_spectra_hard[int(it)][:,1].cpu().detach().numpy(), '--', color='C1', label='prediction - A')
                #     plt.plot(pred_spectra_hard[int(it)][:,2].cpu().detach().numpy(), '--', color='C2', label='prediction - T')
                #     plt.legend(loc=1)
                #     stack_tar = stacks[int(it)]
                #     stack_tar    = np.vectorize(idx_to_token.get)(stack_tar.detach().cpu().numpy())
                #     stack_pred = np.vectorize(idx_to_token.get)(stack_pred.detach().cpu().numpy())
                #     ACC, _ = token_accuracy(stacks, logits.argmax(dim=-1), eos_idx, pad_idx, msk_idx)
                #     spectrum_mae = masked_mae(spectra[0], pred_spectra_hard[0])
                #     # Automatically position the target and prediction text    
                #     text = '\n'.join(f'{material}' for material in stack_tar)
                #     # text2 = '\n'.join(f'{material.split("_")[1]}' for material in stack_tar)
                #     plt.text(0.02, 1.1, f"Target:\n- - - - - - - - - - - - - -\n{text}\n- - - - - - - - - - - - - -" , ha='left', fontsize=8, transform=plt.gca().transAxes)
                #     # plt.text(0.24, 1.1, f"\n{text2}\n" , ha='right', fontsize=8, transform=plt.gca().transAxes)
                #     text = '\n'.join(f'{material}' for material in stack_pred)
                #     # text2 = '\n'.join(f'{material.split("_")[1]}' for material in stack_pred)
                #     plt.text(0.76, 1.1, f"Prediction:\n- - - - - - - - - - - - - -\n{text}\n- - - - - - - - - - - - - -", ha='left', fontsize=8, transform=plt.gca().transAxes)
                #     # plt.text(0.98, 1.1, f"\n{text2}\n" , ha='right', fontsize=8, transform=plt.gca().transAxes)
                #     # Automatically position the accuracy, MAE, and key text
                #     # spectrum_mae = np.mean(np.absolute(RAT_pred - RAT_tar))
                #     plt.text(0.36, 1.10, "- - - - - - - - - - - - - -\nKey#:\nMAE:\nAccuracy:\n- - - - - - - - - - - - - -", ha='left', fontsize=10, transform=plt.gca().transAxes)
                #     plt.text(0.64, 1.10, f"\n{int(i)}\n{spectrum_mae:.2f}\n{ACC:.2f}\n", ha='right', fontsize=10, transform=plt.gca().transAxes)
                #     plt.ylim(top=1.03)
                #     plt.show()
# %%
                
                if cfg.CE:
                    log_loss_CE = loss_CE.detach()
                    torch.distributed.all_reduce(log_loss_CE)
                    log_loss_CE = (log_loss_CE / world_size).item()
                    train_losses_CE[epoch] += log_loss_CE
                if cfg.MAE:
                    log_loss_MAE = loss_MAE.detach()
                    torch.distributed.all_reduce(log_loss_MAE)
                    log_loss_MAE = (log_loss_MAE / world_size).item()
                    train_losses_MAE[epoch] += log_loss_MAE
                if cfg.H:
                    log_loss_H = loss_H.detach()
                    torch.distributed.all_reduce(log_loss_H)
                    log_loss_H = (log_loss_H / world_size).item()
                    train_losses_H[epoch] += log_loss_H
                acc, _ = token_accuracy(stacks, logits.argmax(dim=-1), eos_idx, pad_idx, msk_idx)
                train_acc[epoch] += acc
                
                # update the progressbar
                if rank == 0:
                    train_progressbar.set_postfix(
                        loss=f'{train_losses[epoch] / (i + 1):.4f}',
                        loss_MAE=f'{train_losses_MAE[epoch] / (i + 1):.4f}',
                        loss_H=f'{train_losses_H[epoch] / (i + 1):.4f}',
                        loss_CE=f'{train_losses_CE[epoch] / (i + 1):.4f}',
                        acc=f'{train_acc[epoch] / (i + 1) * 100:.4f}%'
                    )
                    train_progressbar.update()
    
            # clean up the progress bar
            if rank == 0:
                train_progressbar.close()
                position = spectra.shape[0] // 2
                for position in range(spectra.shape[0]):
                    if position%100 == 0:
                        print('\n')
                        print(stacks[position])
                        print(logits[position].argmax(dim=-1))
                
        if not arguments.noval:
            if arguments.valsim == 'TMM_FAST':
                # --- Validation with tmm_fast (same physics as training) ---
                from typing import Any, Dict, List
            
                model_ddp.eval()
                val_loader.sampler.set_epoch(epoch)
            
                results: List[Dict[str, Any]] = []
                with torch.no_grad():
                    for j, batch in enumerate(val_loader):
                        spectra, stacks = batch
                        spectra = spectra.to(device, non_blocking=True)   # [B, W, 3]
                        stacks  = stacks.to(device, non_blocking=True)    # [B, S]
            
                        # 1) Forward model → logits [B,S,V]
                        logits, _ = model_ddp(spectra)
                        # logits = model_ddp(spectra)
            
                        # 2) Convert logits → ids for accuracy, align with targets
                        pred_ids = logits.argmax(dim=-1) if logits.dim() == 3 else logits  # [B,S]
                        L = min(stacks.size(1), pred_ids.size(1))
                        stacks_aligned = stacks[:, :L]
                        pred_ids       = pred_ids[:, :L]
            
                        # 3) Token-accuracy (ignoring PAD)
                        acc, _ = token_accuracy(stacks_aligned, pred_ids, eos_idx, pad_idx, msk_idx)
                        val_acc[epoch] += acc
                        
                        # 4) Run differentiable TMM to get predicted spectra
                        #    TMM_RAT returns [B, 3*W] (concat R|A|T). Reshape to [B, W, 3] to compare with data.
                        res = TMM_RAT(pred_ids, wl_tensor, theta, eos=eos_idx, pad=pad_idx, msk=msk_idx)
                        B = spectra.size(0)
                        W = wl_tensor.numel()
                        if res.dim() == 3 and res.size(1) == 3:         # [B,3,W] → [B,W,3]
                            pred_spectra = res.permute(0, 2, 1)
                        elif res.dim() == 2 and res.size(1) == 3*W:     # [B,3W]  → [B,W,3]
                            pred_spectra = res.view(B, 3, W).permute(0, 2, 1)
                        else:
                            raise RuntimeError(f"Unexpected TMM shape {tuple(res.shape)}")

                        # 5) Per-sample MAE and logging payload
                        #    Also map ids → tokens for saving
                        target_tokens    = np.vectorize(idx_to_token.get)(stacks_aligned.detach().cpu().numpy())
                        predicted_tokens = np.vectorize(idx_to_token.get)(pred_ids.detach().cpu().numpy())
                        
                        batch_mae = (pred_spectra - spectra).abs().mean(dim=(1, 2))            # [B]
            
                        for b in range(B):
                            acc_b, _ = token_accuracy(stacks_aligned[b:b+1], pred_ids[b:b+1], eos_idx, pad_idx, msk_idx)
                            tgt_spec_flat = spectra[b].T.detach().cpu().reshape(-1).numpy()   # [3W]
                            pred_spec_flat = pred_spectra[b].T.detach().cpu().reshape(-1).numpy()
                            mae_b = float(batch_mae[b].detach().cpu().item())
            
                            results.append({
                                "target_seq":      np.array(target_tokens[b]),
                                "pred_seq":        np.array(predicted_tokens[b]),
                                "target_spectrum": tgt_spec_flat,
                                "pred_spectrum":   pred_spec_flat,
                                "accuracy":        acc_b,
                                "mae":             mae_b,
                            })
                        # import matplotlib.pyplot as plt
                        # if (j % max(1, 50)) == 0:
                        #     it = 0
                        #     stack_pred = logits[int(it)].argmax(dim=-1)
                        #     with torch.no_grad():
                        #         pred_ids = logits.argmax(-1)
                        #         res_hard = TMM_RAT(pred_ids, wl_tensor, theta, eos=eos_idx, pad=pad_idx, msk=msk_idx)
                        #         B, W = spectra.size(0), wl_tensor.numel()
                        #         pred_spectra_hard = res_hard.permute(0,2,1) if res_hard.dim()==3 else res_hard.view(B,3,W).permute(0,2,1)
                        #     plt.plot(spectra[int(it)][:,0].cpu().detach().numpy(), '-', label='target - R')
                        #     plt.plot(spectra[int(it)][:,1].cpu().detach().numpy(), '-', label='target - A')
                        #     plt.plot(spectra[int(it)][:,2].cpu().detach().numpy(), '-', label='target - T')
                        #     plt.plot(pred_spectra_hard[int(it)][:,0].cpu().detach().numpy(), '--', color='C0', label='prediction - R')
                        #     plt.plot(pred_spectra_hard[int(it)][:,1].cpu().detach().numpy(), '--', color='C1', label='prediction - A')
                        #     plt.plot(pred_spectra_hard[int(it)][:,2].cpu().detach().numpy(), '--', color='C2', label='prediction - T')
                        #     plt.legend(loc=1)
                        #     stack_tar = stacks[int(it)]
                        #     stack_tar    = np.vectorize(idx_to_token.get)(stack_tar.detach().cpu().numpy())
                        #     stack_pred = np.vectorize(idx_to_token.get)(stack_pred.detach().cpu().numpy())
                        #     ACC, _ = token_accuracy(stacks, logits.argmax(dim=-1), eos_idx, pad_idx, msk_idx)
                        #     spectrum_mae = masked_mae(spectra[0], pred_spectra_hard[0])
                        #     # Automatically position the target and prediction text    
                        #     text = '\n'.join(f'{material}' for material in stack_tar)
                        #     # text2 = '\n'.join(f'{material.split("_")[1]}' for material in stack_tar)
                        #     plt.text(0.02, 1.1, f"Target:\n- - - - - - - - - - - - - -\n{text}\n- - - - - - - - - - - - - -" , ha='left', fontsize=8, transform=plt.gca().transAxes)
                        #     # plt.text(0.24, 1.1, f"\n{text2}\n" , ha='right', fontsize=8, transform=plt.gca().transAxes)
                        #     text = '\n'.join(f'{material}' for material in stack_pred)
                        #     # text2 = '\n'.join(f'{material.split("_")[1]}' for material in stack_pred)
                        #     plt.text(0.76, 1.1, f"Prediction:\n- - - - - - - - - - - - - -\n{text}\n- - - - - - - - - - - - - -", ha='left', fontsize=8, transform=plt.gca().transAxes)
                        #     # plt.text(0.98, 1.1, f"\n{text2}\n" , ha='right', fontsize=8, transform=plt.gca().transAxes)
                        #     # Automatically position the accuracy, MAE, and key text
                        #     # spectrum_mae = np.mean(np.absolute(RAT_pred - RAT_tar))
                        #     plt.text(0.36, 1.10, "- - - - - - - - - - - - - -\nKey#:\nMAE:\nAccuracy:\n- - - - - - - - - - - - - -", ha='left', fontsize=10, transform=plt.gca().transAxes)
                        #     plt.text(0.64, 1.10, f"\n{int(i)}\n{spectrum_mae:.2f}\n{ACC:.2f}\n", ha='right', fontsize=10, transform=plt.gca().transAxes)
                        #     plt.ylim(top=1.03)
                        #     plt.show()
            
                    # 6) Gather results across DDP ranks and compute epoch MAE on rank 0
                    gathered = [None for _ in range(world_size)]
                    torch.distributed.all_gather_object(gathered, results)
            
                    if rank == 0:
                        merged_results: List[Dict[str, Any]] = []
                        for sublist in gathered:
                            merged_results.extend(sublist)
            
                        out_name = f"results_{cfg.RUN_NAME}"
                        save_JSONPICKLE_NEW(cfg.PATH_SAVED, merged_results, out_name)
                        print(f"💾 [rank 0] Saved {len(merged_results)} samples → …/{out_name}.json")
                        # Epoch-level MAE
                        val_MAE[epoch] = np.mean([r["mae"] for r in merged_results])
                        # print(f"validation MAE: {np.min(val_MAE.numpy(),initial=np.inf, where=~np.isnan(val_MAE.numpy()))}")
                        print(f" min val MAE: {np.min(val_MAE.numpy(),initial=np.inf, where=~np.isnan(val_MAE.numpy())):.4f}")
                        print(f"last val MAE: {val_MAE[epoch]:.4f}")
                
            elif arguments.valsim == 'RAYFLARE':
                from utils import init_tokenmaps, save_JSONPICKLE_NEW, load_JSONPICKLE_NEW
                from call_rayflare import Call_RayFlare_with_dict
                from plots import plot_samples, plot_mae, train_data_comp, plot_mae_comparison
                from typing import Any, Dict, List, Mapping, Tuple, Union, Optional
                # from solcore import config
                # config.user_folder = rf'{cfg.PATH}/RayFlare/data/Solcore'
                # config['Parameters','custom'] = f'{config.user_folder}/custom_parameters.txt'
                model_ddp.eval()
                val_loader.sampler.set_epoch(epoch)
                # print('rayflare...')
                # print(f"VAL loader: {len(val_loader)} batches, {len(val_loader.dataset)} samples")
                results: List[Dict[str, Any]] = []
                with torch.no_grad():
                    number = 0
                    MAE_list = []
                    for j, batch in enumerate(val_loader):
                        # print(f"j: {j}, number: {number}")
                        if j < 1e10:
                            spectra, stacks = batch
                            spectra, stacks = spectra.to(device), stacks.to(device)
                            
                            # logits, _ = model_ddp(spectra)
                            logits = model_ddp(spectra)

                            # Convert logits→ids if needed and align with targets once
                            if logits.dim() == 3:
                                predicted_ids = logits.argmax(dim=-1)
                            else:
                                predicted_ids = logits
                            
                            L = min(stacks.size(1), predicted_ids.size(1))
                            stacks_aligned = stacks[:, :L]
                            predicted_ids  = predicted_ids[:, :L]
                            
                            acc, _ = token_accuracy(stacks_aligned, predicted_ids, eos_idx, pad_idx, msk_idx)
                            val_acc[epoch] += acc
                            
                            # Map ids → tokens for RayFlare
                            target_tokens    = np.vectorize(idx_to_token.get)(stacks_aligned.cpu().numpy())
                            predicted_tokens = np.vectorize(idx_to_token.get)(predicted_ids.cpu().numpy())
                            
                            # When iterating per-sample, iterate over predicted_ids (not logits)
                            for spectrum, stack, pred_id, target_token, predicted_token in zip(
                                    spectra, stacks_aligned, predicted_ids, target_tokens, predicted_tokens):
                                acc_temp, _ = token_accuracy(stack.unsqueeze(0), pred_id.unsqueeze(0), eos_idx, pad_idx, msk_idx)
                                res = Call_RayFlare_with_dict(cfg, predicted_token, EOS_TOKEN)
                                res = res.unsqueeze(0)
                                pred_spectra = res.permute(0, 2, 1)
                                pred_spectra = pred_spectra.to(device)
                                MAE = (pred_spectra[0] - spectrum).abs().mean()
                                
                                MAE_list.append(MAE)
                                tgt_spec_flat = spectrum.T.detach().cpu().reshape(-1).numpy()   # [3W]
                                pred_spec_flat = pred_spectra[0].T.detach().cpu().reshape(-1).numpy()
                                
                                results.append({
                                    "target_seq": np.array(target_token),
                                    "pred_seq": np.array(predicted_token),
                                    "target_spectrum": tgt_spec_flat,
                                    "pred_spectrum": pred_spec_flat,
                                    "accuracy": acc_temp,
                                    "mae": MAE.item(),
                                })
                                number += 1
                        else:
                            pass
                    
                    # Prepare a holder on each rank
                    gathered = [None for _ in range(world_size)]
                    # This will send `results` from each rank into `gathered[0..world_size-1]` on **every** rank
                    
                    torch.distributed.all_gather_object(gathered, results)
                    if rank == 0:
                        # Merge lists of dicts into one big list
                        merged_results = []
                        for sublist in gathered:
                            merged_results.extend(sublist)
                
                        out_name = f'results_{cfg.RUN_NAME}'
                        save_JSONPICKLE_NEW(cfg.PATH_SAVED, merged_results, out_name)
                        print(f"💾 [rank 0] Saved {len(merged_results)} samples → …/{out_name}.json")
                        val_MAE[epoch] = np.mean([r['mae'] for r in merged_results])
                        print(f" min val MAE: {np.min(val_MAE.numpy(),initial=np.inf, where=~np.isnan(val_MAE.numpy())):.4f}")
                        print(f"last val MAE: {val_MAE[epoch]:.4f}")

            elif not arguments.valsim:
                model_ddp.eval()
                val_loader.sampler.set_epoch(epoch)
        
                with torch.no_grad():
                    for k, batch in enumerate(val_loader):
                        spectra, stacks = batch
                        spectra, stacks = spectra.to(device), stacks.to(device)
    
                        logits = model_ddp(spectra)
                        acc, _ = token_accuracy(stacks, logits, eos_idx, pad_idx, msk_idx)
                        val_acc[epoch] += acc
    
                    print(f'\nValidation accuracy: {val_acc[epoch]:.2f}%\n')
            else:
                if rank == 0:
                    print(f"⚠️ No validation at epoch {epoch}")
                continue

        # save best model
        if not arguments.notrain and rank == 0:
            if not arguments.valsim:
                if np.max(val_acc.numpy()) <= val_acc_save:
                    print(f"Checkpoint saved: {np.mean(val_acc.numpy(), where=~np.isnan(val_acc.numpy()))} <= {val_acc_save} [{epoch}]")
                    val_MAE_save = val_MAE[epoch]
                    state = {
                        'epoch': epoch,
                        'model_state': model_ddp.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        # training
                        'train_losses': train_losses,
                        'train_acc': train_acc,
                        # validation
                        'val_acc': val_acc,
                        'val_MAE': val_MAE
                    }
                    torch.save(state, checkpoint)
            if arguments.valsim:
                if val_MAE[epoch].item() <= val_MAE_save:
                    print(f"Checkpoint saved: {np.min(val_MAE.numpy(),initial=np.inf, where=~np.isnan(val_MAE.numpy()))} <= {val_MAE_save} [{epoch}]")
                    val_MAE_save = val_MAE[epoch]
                    state = {
                        'epoch': epoch,
                        'model_state': model_ddp.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        # training
                        'train_losses': train_losses,
                        'train_acc': train_acc,
                        # validation
                        'val_acc': val_acc,
                        'val_MAE': val_MAE
                    }
                    torch.save(state, checkpoint)


def train(arguments: argparse.Namespace) -> None:
    # distributed setup
    device, slurm_localid, rank, world_size = init_distributed()
    set_all_seeds(random.randint(1,int(1e6)))
    # set_all_seeds(42)
    set_torch_options()

    # create directory
    if rank == 0:
        if not os.path.exists(cfg.PATH_RUN):
            os.mkdir(cfg.PATH_RUN)
        if not os.path.exists(cfg.PATH_SAVED):
            os.mkdir(cfg.PATH_SAVED)

    # main training loop
    train_loop(arguments, device, rank, world_size, slurm_localid)

    # graceful shutdown of ddp
    torch.distributed.destroy_process_group()
    

if __name__ == '__main__':
    try:
        # graceful shutdown of ddp
        torch.distributed.destroy_process_group()
    except:
        pass
    arguments = cli.parse_arguments()
    arguments.notrain = cfg.NOTRAIN
    arguments.valsim = cfg.VALSIM
    arguments.noplot = cfg.NOPLOT
    try:
        train(arguments)
        pass
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    