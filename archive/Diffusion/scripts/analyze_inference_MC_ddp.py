#!/usr/bin/env python3
from __future__ import annotations
import argparse
import importlib
import os
import math
from typing import Dict, Any, List
import random

import numpy as np
import torch
import torch.distributed
from torch.utils.data import DataLoader, Subset

import dataset
from utils import init_tokenmaps, save_JSONPICKLE_NEW, load_JSONPICKLE_NEW, token_accuracy, load_state_dict_flexible, core_module_crop, init_distributed, set_all_seeds, set_torch_options, unique_length_int_generator
from model import build_model
from call_tmm_fast import load_materials, TMMSpectrum
import plots


def pick_device(user_device: str | None = None) -> torch.device:
    if user_device:
        return torch.device(user_device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def spectrum_mae(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - tgt), dim=(1, 2))


def spectrum_mse(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - tgt) ** 2, dim=(1, 2))


def build_dataloader(cfg, tokens: torch.Tensor, limit: int, batch: int, eos: int, pad: int):
    val_dataset = dataset.SpectraDataset(cfg.PATH_VALID, tokens, device='cpu')
    if limit and limit > 0:
        # idx = np.arange(min(limit, len(val_dataset)))
        idx = unique_length_int_generator(0e0, len(val_dataset)-1, limit)
        val_subset = Subset(val_dataset, idx)
    else:
        val_subset = val_dataset

    max_stack_depth = val_dataset.get_maximum_depth()

    sampler = None
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(val_subset, shuffle=False, drop_last=False)

    loader = DataLoader(
        val_subset,
        batch_size=batch,
        shuffle=False if sampler is not None else False,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        sampler=sampler,
        collate_fn=lambda batch: dataset.pad_batch(batch, max_stack_depth, eos, pad)
    )
    return loader, max_stack_depth, sampler


def build_tmm(cfg, device: torch.device, idx_to_token: Dict[int,str]) -> TMMSpectrum:
    degree = math.pi / 180.0
    theta  = torch.tensor(cfg.INCIDENCE_ANGLE * degree, device=device, dtype=torch.complex128).unsqueeze(0)
    cfg.WAVELENGTHS = np.arange(cfg.WAVELENGTH_MIN, cfg.WAVELENGTH_MAX + 1, cfg.WAVELENGTH_STEPS)
    wl_tensor = torch.tensor(cfg.WAVELENGTHS, dtype=torch.complex128, device=device)

    nk_dict = load_materials(cfg)
    TMM_RAT = TMMSpectrum(nk_dict, idx_to_token, device=device).to(device)
    return TMM_RAT, wl_tensor, theta


def logits_to_spectra(
    logits: torch.Tensor,          # [B,S,V] logits  OR  [B,S] token ids
    TMM_RAT: TMMSpectrum,
    wl_tensor: torch.Tensor,
    theta: torch.Tensor,
    eos: int, pad: int, msk: int,
    tau: float
) -> torch.Tensor:
    """
    Accepts either:
      • logits: float tensor [B,S,V]  -> soft/STE path to TMM
      • token ids: long/int tensor [B,S] -> pass directly to TMM (hard sequence)
    Returns:
      predicted_spectra: [B, W, 3]
    """
    # Case 1: sampling mode → model(spectra) returned token IDs [B,S]
    if logits.dim() == 2:
        tokens_for_tmm = logits.to(torch.long)  # pass ids directly; TMMSpectrum handles ints
        res = TMM_RAT(tokens_for_tmm, wl_tensor, theta, eos=eos, pad=pad, msk=msk)

    # Case 2: training/inference with logits [B,S,V]
    elif logits.dim() == 3:
        if tau <= 0.0:
            # STE: argmax one-hot with softmax gradient injection
            hard  = torch.nn.functional.one_hot(logits.argmax(-1), num_classes=logits.shape[-1]).to(logits.dtype)
            probs = torch.softmax(logits, dim=-1)
            tokens_for_tmm = hard + probs - probs.detach()
        else:
            tokens_for_tmm = torch.softmax(logits / tau, dim=-1)  # soft tokens
        res = TMM_RAT(tokens_for_tmm, wl_tensor, theta, eos=eos, pad=pad, msk=msk)

    else:
        raise ValueError(f"Expected logits/tokens with dim 2 or 3, got shape {tuple(logits.shape)}")

    # Normalise output to [B, W, 3]
    B = logits.size(0)
    W = wl_tensor.numel()
    if res.dim() == 3 and res.size(1) == 3:            # [B,3,W] → [B,W,3]
        predicted_spectra = res.permute(0, 2, 1)
    elif res.dim() == 2 and res.size(1) == 3 * W:       # [B,3W]  → [B,W,3]
        predicted_spectra = res.view(B, 3, W).permute(0, 2, 1)
    else:
        raise RuntimeError(f"Unexpected TMM shape {tuple(res.shape)}")

    return torch.nan_to_num(predicted_spectra, nan=0.0, posinf=1.0, neginf=0.0)


def parse_compare_arg(items: List[str]) -> Dict[str, str]:
    """Parse ['label:path.json', ...] into dict."""
    out = {}
    for it in items:
        if ':' not in it:
            raise ValueError(f"--compare expects 'label:path_to_json', got: {it}")
        label, path = it.split(':', 1)
        if not path.endswith('.json'):
            path = path + '.json'
        out[label] = path
    return out


def main():
    ap = argparse.ArgumentParser(description='Inference + plotting + model-to-model comparison (DDP, MC-best-of-N)')
    ap.add_argument('--config', required=True, help="Config module name, e.g. config_MD43")
    ap.add_argument('--checkpoint', default='auto', help="'auto'|'none'|path/to/ckpt.pt")
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default=None)
    ap.add_argument('--device', default=None)
    ap.add_argument('--plots', type=int, default=0, help='How many per-sample plots to generate (0=none)')
    ap.add_argument('--noplot_hists', action='store_true', help='Skip histogram plots')

    # --- New: Sampling / MC search ---
    ap.add_argument('--mc_samples', type=int, default=1, help='Number of stochastic inferences per sample; keep best (lowest MAE).')
    ap.add_argument('--sample_temperature', type=float, default=1.0, help='Sampling temperature (>=0). Use <1 for peakier.')
    ap.add_argument('--top_k', type=int, default=0, help='Top-K filtering (0 disables).')
    ap.add_argument('--top_p', type=float, default=0.0, help='Top-p nucleus sampling (0 disables).')
    ap.add_argument('--seed', type=int, default=1234, help='Base RNG seed (per-rank offsets applied).')

    # Comparison options
    ap.add_argument('--compare', action='append', default=[],
                    help="Repeatable: 'label:path_to_results.json' for model-to-model comparison plots.")
    ap.add_argument('--include_current_in_compare', action='store_true',
                    help='Also include the results from this run in the comparison set.')
    args = ap.parse_args()

    # Load config
    cfg = importlib.import_module(args.config)

    # Distributed setup (mirrors optollama.py)
    device, local_id, rank, world_size = init_distributed()
    randint = random.randint(1,int(1e6))
    set_all_seeds(args.seed + rank + randint)
    set_torch_options()

    # Token tables
    tokens, token_to_idx, idx_to_token, EOS_TOKEN, PAD_TOKEN, MSK_TOKEN, eos, pad, msk = init_tokenmaps(cfg.PATH_DATA)

    # DataLoader
    loader, max_stack_depth, sampler = build_dataloader(cfg, tokens, args.limit, args.batch, eos, pad)
    if rank == 0:
        print("VAL path:", cfg.PATH_VALID)
        # When PATH_VALID is a list, len(loader.dataset) is the sum; when it’s a str, it’s that file’s size.
        print("len(val_dataset) =", len(loader.dataset))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            print("world_size =", world_size)

    # Model dims
    # Model dims — accept loader batches of (spectra, stacks) or (spectra, stacks, idxs)
    _first_batch = next(iter(loader))
    spectra0 = _first_batch[0]           # always the first item
    spectrum_sample = spectra0[0]        # [W,3]
    vocab_size = len(tokens)

    # Model (DDP-style like in optollama.py)
    local_model = build_model(
        model_type=getattr(cfg, 'ARCH', getattr(cfg, 'OL_MODEL', 'dit')),
        spectrum_dim=spectrum_sample.shape[-1],
        vocab_size=vocab_size,
        timesteps=cfg.STEPS,
        max_len=spectrum_sample.shape[0],
        max_stack_depth=max_stack_depth,
        mask_idx=msk,
        d_model=cfg.D_MODEL,
        n_blocks=cfg.N_BLOCKS,
        n_heads=cfg.N_HEADS,
        dropout=cfg.DROPOUT,
        idx_to_token=idx_to_token,
        pad_idx=pad,
        eos_idx=eos,
        device=str(device),
        sample_spectrum=spectrum_sample.unsqueeze(0),
    ).to(torch.float32).to(device)

    # Set stochastic sampling knobs BEFORE wrapping with DDP
    if hasattr(local_model, 'set_sampling'):
        try:
            local_model.set_sampling(
                temperature=args.sample_temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            # if hasattr(local_model, "set_sampling"):
            #     local_model.set_sampling(temperature=args.sample_temperature, top_k=args.top_k, top_p=args.top_p)
            # else:
            #     local_model.sample_temperature = args.sample_temperature
            #     local_model.sample_top_k = args.top_k
            #     local_model.sample_top_p = args.top_p
        except Exception as e:
            if rank == 0:
                print(f"[warn] set_sampling not applied: {e}")

    if torch.distributed.is_available() and torch.distributed.is_initialized() \
       and torch.distributed.get_world_size() > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            local_model,
            device_ids=[int(local_id)] if str(device).startswith('cuda') else None,
            output_device=int(local_id) if str(device).startswith('cuda') else None,
            find_unused_parameters=False
        )
    else:
        model = local_model
    
    # Load checkpoint
    ckpt_path = None
    if args.checkpoint == 'auto':
        ckpt_path = os.path.join(cfg.PATH_SAVED, 'ol3l-checkpoint.pt')
        if not os.path.exists(ckpt_path):
            ckpt_path = None
    elif args.checkpoint != 'none':
        ckpt_path = args.checkpoint
    if ckpt_path:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = load_state_dict_flexible(core_module_crop(model), state["model_state"])
        if rank == 0:
            print(f"Loaded checkpoint: {ckpt_path}")
    else:
        if rank == 0:
            print("No checkpoint loaded (random weights).")

    # TMM
    TMM_RAT, wl_tensor, theta = build_tmm(cfg, device, idx_to_token)

    # Output base (suffix encodes MC/K/P)
    base_out = args.out
    if base_out is None:
        resume = getattr(cfg, 'RESUME_EPOCH', 'XX')
        tgt    = getattr(cfg, 'TARGET', 'valid')
        base_out = os.path.join(cfg.PATH_RUN, f"{cfg.RUN_NAME}_inference_N{args.limit}_MC{args.mc_samples}_K{args.top_k}_P{args.top_p}")
    # suffix = f"_MC{args.mc_samples}_K{args.top_k}_P{args.top_p}"
    # base_out = base_out + suffix
    if rank == 0:
        os.makedirs(os.path.dirname(base_out), exist_ok=True)

    # Eval
    if sampler is not None:
        try:
            sampler.set_epoch(0)
        except Exception:
            pass
    model.eval()

    results: List[Dict[str, Any]] = []
    dn_curve_sum = None  # accumulated sum over steps (CPU numpy array)
    dn_curve_count = 0   # number of curves accumulated
    batch_global_accs: List[float] = []
    total_correct_tokens = 0.0
    total_valid_tokens = 0.0

    rng_base = args.seed + 1337 * (rank + 1)
    with torch.no_grad():
        for batch_out in loader:
            # pad_batch now may return 2 or 3 items
            if len(batch_out) == 3:
                spectra, stacks, idxs = batch_out
            else:
                spectra, stacks = batch_out
                # fall back to a per-rank running counter if indices aren't present
                if 'running_idx' not in locals():
                    running_idx = 0
                idxs = torch.arange(running_idx, running_idx + stacks.size(0))
                running_idx += stacks.size(0)
            if rank == 0:
                print(f'idx: {idxs[0]} - {idxs[-1]}') 
            spectra = spectra.to(device, non_blocking=True)
            stacks  = stacks.to(device, non_blocking=True)

            B = spectra.size(0)
            best_mae = torch.full((B,), float('inf'), device=device)
            best_pred_spectra = None  # [B,W,3]
            best_preds_ids = None     # [B,S]
            best_dn_maes = None

            # Monte-Carlo loop
            for s in range(max(1, args.mc_samples)):
                if rank == 0:
                    print(f'MC sample: {s}')
                # ensure stochasticity per-rank per-try
                torch.manual_seed(rng_base + s)
                np.random.seed(rng_base + 17 * s)

                # Forward
                logits, dn_maes = model(spectra)

                # Normalize denoise curve to [B, STEPS] for per-sample keeping
                dn_curve_batch = None
                try:
                    if isinstance(dn_maes, torch.Tensor):
                        dn_cpu = dn_maes.detach().float()  # keep on current device for where; no grad
                        if dn_cpu.dim() == 2:
                            # Accept [B, STEPS] or [STEPS, B] → make [B, STEPS]
                            if dn_cpu.size(0) == spectra.size(0):
                                dn_curve_batch = dn_cpu
                            elif dn_cpu.size(1) == spectra.size(0):
                                dn_curve_batch = dn_cpu.transpose(0, 1)
                        elif dn_cpu.dim() == 1:
                            # [STEPS] replicated over batch
                            dn_curve_batch = dn_cpu.view(1, -1).expand(spectra.size(0), -1)
                except Exception:
                    dn_curve_batch = None

                # Convert to spectra
                pred_spectra = logits_to_spectra(logits, TMM_RAT, wl_tensor, theta, eos, pad, msk, tau=args.tau)

                # Per-sample MAE
                mae_s = spectrum_mae(pred_spectra, spectra)

                # Candidate predicted token ids for accuracy bookkeeping
                if logits.dim() == 2:
                    preds_ids_s = logits.long()
                elif logits.dim() == 3:
                    preds_ids_s = logits.argmax(dim=-1)
                else:
                    raise RuntimeError("Unexpected logits dim; expected 2 or 3.")

                # Keep elementwise best
                take = mae_s < best_mae
                best_mae = torch.where(take, mae_s, best_mae)
                if best_pred_spectra is None:
                    best_pred_spectra = pred_spectra
                    best_preds_ids = preds_ids_s
                    best_dn_maes = dn_maes
                else:
                    # broadcast take to spectra dims [B,W,3]
                    best_pred_spectra = torch.where(take.view(B, 1, 1), pred_spectra, best_pred_spectra)
                    best_preds_ids = torch.where(take.view(B, 1), preds_ids_s, best_preds_ids)
                    # keep the dn_maes curve corresponding to the chosen MC pass, per sample
                    if dn_curve_batch is not None:
                        if best_dn_maes is None:
                            best_dn_maes = dn_curve_batch
                        else:
                            # both [B, STEPS]
                            best_dn_maes = torch.where(take.view(B, 1), dn_curve_batch, best_dn_maes)

            # Accuracy on best predictions
            global_acc, batch_acc = token_accuracy(stacks, best_preds_ids, eos, pad, msk)
            batch_mae = spectrum_mae(best_pred_spectra, spectra)
            batch_mse = spectrum_mse(best_pred_spectra, spectra)

            batch_global_accs.append(float(global_acc.item()))

            # For overall weighted global accuracy across the entire evaluation,
            # recompute valid/correct token counts for this batch and accumulate.
            L_acc = min(stacks.size(1), best_preds_ids.size(1))
            stacks_aligned_acc = stacks[:, :L_acc]
            preds_aligned_acc  = best_preds_ids[:, :L_acc]
            is_eos_acc = (stacks_aligned_acc == eos)
            before_first_eos_acc = (is_eos_acc.cumsum(dim=1) == 0)
            valid_mask_acc = before_first_eos_acc & (stacks_aligned_acc != pad) & (stacks_aligned_acc != msk)
            correct_mask_acc = (stacks_aligned_acc == preds_aligned_acc) & valid_mask_acc
            total_correct_tokens += float(correct_mask_acc.sum().item())
            total_valid_tokens   += float(valid_mask_acc.sum().item())

            # Build per-sample results
            for i in range(stacks.size(0)):
                # Stable per-sample ID: hash of target tokens up to EOS (avoids DDP padding duplicates)
                tgt_ids = stacks[i].tolist()
                tgt_len = tgt_ids.index(eos) if eos in tgt_ids else len(tgt_ids)
                tgt_core = [int(t) for t in tgt_ids[:tgt_len] if int(t) not in (pad, msk, eos)]
                sample_id = hash(tuple(tgt_core))
                tgt_tokens = [idx_to_token[int(t)] for t in tgt_core]

                pred_ids = best_preds_ids[i].tolist()
                pred_len = pred_ids.index(eos) if eos in pred_ids else len(pred_ids)
                pred_tokens = [idx_to_token[int(t)] for t in pred_ids[:pred_len] if int(t) not in (pad, msk, eos)]

                # Per-sample denoise curve from the chosen MC pass (length = STEPS)
                denoise_steps = None
                if isinstance(best_dn_maes, torch.Tensor):
                    try:
                        denoise_steps = best_dn_maes[i].detach().float().cpu().numpy().tolist()
                    except Exception:
                        denoise_steps = None

                ds_idx = int(idxs[i].item())
                results.append({
                    'id': f"{ds_idx}_{int(sample_id)}",
                    'dataset_index': ds_idx,
                    'acc': float(batch_acc[i].item()),
                    'mae': float(batch_mae[i].item()),
                    'mse': float(batch_mse[i].item()),
                    'stack_target_tokens': tgt_tokens,
                    'stack_pred_tokens': pred_tokens,
                    'RAT_target_flat': spectra[i].T.reshape(-1,).detach().cpu().numpy().tolist(),
                    'RAT_pred_flat':   best_pred_spectra[i].T.reshape(-1).detach().cpu().numpy().tolist(),
                    'denoise_steps': denoise_steps,
                })

    # --- DDP gather of results to rank 0 ---
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        gathered_lists = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(gathered_lists, results)
        if rank == 0:
            merged = []
            for sub in gathered_lists:
                merged.extend(sub)
            # Deduplicate by 'id' (keep first occurrence)
            seen = set()
            dedup = []
            for r in merged:
                rid = r.get('id', None)
                if rid is None or rid not in seen:
                    dedup.append(r)
                    if rid is not None:
                        seen.add(rid)
            results = dedup
            # Enforce --limit if provided (after dedup)
            if args.limit and args.limit > 0 and len(results) > args.limit:
                results = results[:args.limit]

    # ---- Summaries ----
    maes = [r['mae'] for r in results]
    mses = [r['mse'] for r in results]
    accs = [r['acc'] for r in results]

    summary = {
        'n_examples': int(len(results)),
        'acc_mean': float(np.mean(accs)) if accs else None,
        'acc_median': float(np.median(accs)) if accs else None,
        'mae_mean': float(np.mean(maes)) if maes else None,
        'mae_median': float(np.median(maes)) if maes else None,
        'mse_mean': float(np.mean(mses)) if mses else None,
        'mse_median': float(np.median(mses)) if mses else None,
        'mc_samples': int(args.mc_samples),
        'temperature': float(args.sample_temperature),
        'top_k': int(args.top_k),
        'top_p': float(args.top_p),
    }

    # ---- Save (rank 0 only) ----
    if rank == 0:
        # Per-example JSON (portable types only)
        save_JSONPICKLE_NEW(os.path.dirname(base_out), results, os.path.basename(base_out))
        # Also write standard JSON + PKL if you want both
        import json, pickle
        with open(base_out + '.json', 'w') as fh:
            json.dump(results, fh)
        with open(base_out + '.pkl', 'wb') as fh:
            pickle.dump(results, fh)
        with open(base_out + '_summary.json', 'w') as fh:
            json.dump(summary, fh, indent=2)
        print(f"Saved per-example results to: {base_out}.json | .pkl")
        print(f"Saved summary metrics to:     {base_out}_summary.json")

        # Optional plots
        try:
            if not args.noplot_hists and maes:
                plots.plot_mae(cfg, maes)
            if args.plots:
                n_show = int(min(args.plots, len(results)))
                step = max(1, len(results) // max(1, n_show))
                picked = [results[i] for i in range(0, len(results), step)][:n_show]
                for idx, r in enumerate(picked, 1):
                    plots.plot_samples(
                        cfg,
                        np.array(r['RAT_pred_flat']),
                        np.array(r['RAT_target_flat']),
                        r['stack_pred_tokens'],
                        r['stack_target_tokens'],
                        r['acc'],
                        r['mse'],
                        number=idx,
                        RAT_tar_mean=None,
                    )
        except Exception as e:
            print(f"[warn] plotting failed: {e}")


if __name__ == '__main__':
    try:
        import sys
        if len(sys.argv) == 1:
            sys.argv.extend([
                '--config','config_MD49',
                '--checkpoint','auto',
                '--batch','100',
                '--tau','-1',
                '--limit','0',
                '--plots','0',
                '--mc_samples','3',
                '--top_k','3',
                '--top_p','0.0',
            ])
        main()
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
