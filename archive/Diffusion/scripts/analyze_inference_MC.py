
#!/usr/bin/env python3
from __future__ import annotations
import argparse
import importlib
import os
import math
from typing import Dict, Any, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import dataset
from utils import init_tokenmaps, save_JSONPICKLE_NEW, load_JSONPICKLE_NEW, token_accuracy, load_state_dict_flexible, unique_length_int_generator
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
        subset_idx = unique_length_int_generator(0e0, len(val_dataset)-1, limit)
        val_subset = Subset(val_dataset, subset_idx)
    else:
        val_subset = val_dataset

    max_stack_depth = val_dataset.get_maximum_depth()

    loader = DataLoader(
        val_subset,
        batch_size=batch,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=lambda batch: dataset.pad_batch(batch, max_stack_depth, eos, pad)
    )
    return loader, max_stack_depth


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
    ap = argparse.ArgumentParser(description='Inference + plotting + model-to-model comparison')
    ap.add_argument('--config', required=True, help="Config module name, e.g. config_MD43")
    ap.add_argument('--checkpoint', default='auto', help="'auto'|'none'|path/to/ckpt.pt")
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default=None)
    ap.add_argument('--device', default=None)
    ap.add_argument('--plots', type=int, default=0, help='How many per-sample plots to generate (0=none)')
    ap.add_argument('--noplot_hists', action='store_true', help='Skip histogram plots')
    ap.add_argument('--mc_samples', type=int, default=1, help='How many stochastic inference samples per test item; pick best by MAE (default: 1)')
    ap.add_argument('--sample_temperature', type=int, default=0)
    ap.add_argument('--top_k', type=float, default=0.0)
    ap.add_argument('--top_p', type=float, default=0.0)

    # Comparison options
    ap.add_argument('--compare', action='append', default=[],
                    help="Repeatable: 'label:path_to_results.json' for model-to-model comparison plots.")
    ap.add_argument('--include_current_in_compare', action='store_true',
                    help='Also include the results from this run in the comparison set.')
    args = ap.parse_args()

    # Load config
    cfg = importlib.import_module(args.config)
    device = pick_device(args.device)

    # Token tables
    tokens, token_to_idx, idx_to_token, EOS_TOKEN, PAD_TOKEN, MSK_TOKEN, eos, pad, msk = init_tokenmaps(cfg.PATH_DATA)

    # DataLoader
    loader, max_stack_depth = build_dataloader(cfg, tokens, args.limit, args.batch, eos, pad)

    # Model dims
    spectrum_sample, _ = next(iter(loader))
    spectrum_sample = spectrum_sample[0]  # [W,3]
    vocab_size = len(tokens)

    # Model
    model = build_model(
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
    
    # Load checkpoint
    ckpt_path = None
    if args.checkpoint == 'auto':
        ckpt_path = os.path.join(cfg.PATH_SAVED, 'ol3l-checkpoint.pt')
        if not os.path.exists(ckpt_path):
            ckpt_path = None
    elif args.checkpoint != 'none':
        ckpt_path = args.checkpoint

    if ckpt_path:
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model = load_state_dict_flexible(model, state['model_state'])
        # model.load_state_dict(state['model_state'], strict=True)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("No checkpoint loaded (random weights).")
    
    # Example flags: --sample_temperature 1.0 --top_k 50 --top_p 0.95
    if hasattr(model, "set_sampling"):
        model.set_sampling(temperature=args.sample_temperature, top_k=args.top_k, top_p=args.top_p)
    else:
        model.sample_temperature = args.sample_temperature
        model.sample_top_k = args.top_k
        model.sample_top_p = args.top_p
    
    # TMM
    TMM_RAT, wl_tensor, theta = build_tmm(cfg, device, idx_to_token)

    # Output base
    base_out = args.out
    if base_out is None:
        resume = getattr(cfg, 'RESUME_EPOCH', 'XX')
        tgt    = getattr(cfg, 'TARGET', 'valid')
        base_out = os.path.join(cfg.PATH_RUN, f"{cfg.RUN_NAME}_inference_N{args.limit}_MC{args.mc_samples}_K{args.top_k}_P{args.top_p}")
    os.makedirs(os.path.dirname(base_out), exist_ok=True)

    # Eval
    model.eval()
    results: List[Dict[str, Any]] = []
    all_acc, all_mae, all_mse = [], [], []
    batch_global_accs = []
    total_correct_tokens = 0.0
    total_valid_tokens = 0.0
    with torch.no_grad():
        for spectra, stacks in loader:
            spectra = spectra.to(device, non_blocking=True)
            stacks  = stacks.to(device, non_blocking=True)
    
            B = spectra.size(0)
            # Hold “best so far” per item in the batch
            best_mae  = torch.full((B,), float('inf'), device=device)
            best_ids  = None            # [B,S] long
            best_spec = None            # [B,W,3] float
            best_dn   = None            # [B,steps] (if available)
    
            # --- repeat stochastic sampling mc_samples times and keep best by MAE
            for _ in range(max(1, args.mc_samples)):
                try:
                    logits, dn_maes = model(spectra)   # OptoLlama returns (ids, dn_maes)
                except:
                    logits = model(spectra)
                    # dn_maes = None
    
                pred_spectra = logits_to_spectra(logits, TMM_RAT, wl_tensor, theta, eos, pad, msk, tau=args.tau)
                mae_now = spectrum_mae(pred_spectra, spectra)    # [B]
    
                # turn logits to ids where needed
                ids_now = logits if logits.dim() == 2 else logits.argmax(dim=-1)
    
                # first fill
                if best_ids is None:
                    best_ids  = ids_now.clone()
                    best_spec = pred_spectra.clone()
                    best_mae  = mae_now.clone()
                    if dn_maes is not None and torch.is_tensor(dn_maes):
                        best_dn = dn_maes.clone()
                    continue
    
                # update improvements
                improve = mae_now < best_mae                       # [B]
                if improve.any():
                    m_rows = improve.nonzero(as_tuple=False).squeeze(-1)
                    best_mae[m_rows]  = mae_now[m_rows]
                    best_ids[m_rows]  = ids_now[m_rows]
                    best_spec[m_rows] = pred_spectra[m_rows]
                    if dn_maes is not None and torch.is_tensor(dn_maes):
                        if best_dn is None:
                            best_dn = torch.zeros_like(dn_maes)
                        best_dn[m_rows] = dn_maes[m_rows]
    
            # --- metrics for the chosen best per item
            global_acc, batch_acc = token_accuracy(stacks, best_ids, eos, pad, msk)
            batch_mse = spectrum_mse(best_spec, spectra)
    
            all_acc.append(batch_acc.cpu())
            batch_global_accs.append(global_acc.item())
    
            # contribute to global-accuracy reaggregation
            L_acc = min(stacks.size(1), best_ids.size(1))
            stacks_aligned_acc = stacks[:, :L_acc]
            preds_aligned_acc  = best_ids[:, :L_acc]
            is_eos_acc = (stacks_aligned_acc == eos)
            before_first_eos_acc = (is_eos_acc.cumsum(dim=1) == 0)
            valid_mask_acc = before_first_eos_acc & (stacks_aligned_acc != pad) & (stacks_aligned_acc != msk)
            correct_mask_acc = (stacks_aligned_acc == preds_aligned_acc) & valid_mask_acc
            total_correct_tokens += float(correct_mask_acc.sum().item())
            total_valid_tokens   += float(valid_mask_acc.sum().item())
    
            all_mae.append(best_mae.detach().cpu())
            all_mse.append(batch_mse.detach().cpu())
    
            # per-sample payload
            for i in range(stacks.size(0)):
                tgt_ids = stacks[i].tolist()
                tgt_len = tgt_ids.index(eos) if eos in tgt_ids else len(tgt_ids)
                tgt_tokens = [idx_to_token[int(t)] for t in tgt_ids[:tgt_len] if int(t) not in (pad, msk, eos)]
    
                pred_ids = best_ids[i].tolist()
                pred_len = pred_ids.index(eos) if eos in pred_ids else len(pred_ids)
                pred_tokens = [idx_to_token[int(t)] for t in pred_ids[:pred_len] if int(t) not in (pad, msk, eos)]
    
                results.append({
                    'acc': float(batch_acc[i].item()),
                    'mae': float(best_mae[i].item()),
                    'mse': float(batch_mse[i].item()),
                    'stack_target_tokens': tgt_tokens,
                    'stack_pred_tokens': pred_tokens,
                    'RAT_target_flat': spectra[i].T.reshape(-1,).detach().cpu().numpy().tolist(),
                    'RAT_pred_flat':   best_spec[i].T.reshape(-1).detach().cpu().numpy().tolist(),
                    'denoise_steps': None if best_dn is None else best_dn[i].detach().cpu().numpy().tolist(),
                })


    # Aggregate + save
    acc = torch.cat(all_acc).numpy()
    mae = torch.cat(all_mae).numpy()
    mse = torch.cat(all_mse).numpy()
    summary = {
        'n_examples': int(len(results)),
        'acc_mean': float(np.mean(acc)),
        'acc_median': float(np.median(acc)),
        'mae_mean': float(np.mean(mae)),
        'mae_median': float(np.median(mae)),
        'mse_mean': float(np.mean(mse)),
        'mse_median': float(np.median(mse)),
        'global_acc_overall': float((total_correct_tokens / max(1.0, total_valid_tokens))),
        'global_acc_batches_mean': float(np.mean(batch_global_accs)) if batch_global_accs else None,
        # 'global_acc_mean': float(np.mean([global_acc_value])),  # averaged over batches

    }

    save_JSONPICKLE_NEW(os.path.dirname(base_out), results, os.path.basename(base_out))
    import pickle, json
    with open(base_out + '.pkl', 'wb') as fh:
        pickle.dump(results, fh)
    with open(base_out + '_summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2)

    print(f"Saved per-example results to: {base_out}.pkl | .json")
    print(f"Saved summary metrics to:     {base_out}_summary.json")

    # ---------------- PLOTS ----------------
    try:
        maes = [r['mae'] for r in results]
        mses = [r['mse'] for r in results]
        accs = [r['acc'] for r in results]

        if not args.noplot_hists:
            plots.plot_mae(cfg, maes)
            # plots.plot_mse(cfg, mses)
            # plots.plot_accuracy(cfg, accs)

        n_show = int(min(args.plots, len(results)))
        if n_show > 0:
            step = max(1, len(results) // n_show)
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

    # ---------------- COMPARISON ----------------
    try:
        if args.compare or args.include_current_in_compare:
            model_maes_dict = {}
            model_mses_dict = {}
            model_accs_dict = {}

            if args.include_current_in_compare:
                model_maes_dict[f"{cfg.RUN_NAME}_current"] = maes
                model_mses_dict[f"{cfg.RUN_NAME}_current"] = mses
                model_accs_dict[f"{cfg.RUN_NAME}_current"] = accs

            if args.compare:
                pairs = parse_compare_arg(args.compare)
                for label, path in pairs.items():
                    # Use utils loader (jsonpickle)
                    arr = load_JSONPICKLE_NEW(os.path.dirname(path), os.path.basename(path).replace('.json',''))
                    other_mae = [r['mae'] for r in arr]
                    other_mse = [r['mse'] for r in arr]
                    other_acc = [r['acc'] for r in arr]
                    model_maes_dict[label] = other_mae
                    model_mses_dict[label] = other_mse
                    model_accs_dict[label] = other_acc

            if len(model_maes_dict) >= 2 or (args.include_current_in_compare and model_maes_dict):
                plots.plot_mae_comparison(cfg, model_maes_dict)
                plots.plot_mse_comparison(cfg, model_mses_dict)
                plots.plot_acc_comparison(cfg, model_accs_dict)
                print("Saved comparison plots (MAE/MSE/ACC)." )
            else:
                print("[info] not enough models to compare (need at least two sets)." )
    except Exception as e:
        print(f"[warn] comparison plotting failed: {e}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:  # No CLI args given (Spyder case)
        sys.argv.extend([
            # "--config", "config_MD50",       # your config module
            "--config", "config_MD49",       # your config module
            "--checkpoint", "auto",          # or path / none
            "--batch", "1000",
            "--tau", "-1",
            "--limit", "0",
            "--plots", "0",
            "--mc_samples", "3",
            # "--include_current_in_compare"
            # "--compare", "MD43:/path/to/results.json"
            "--sample_temperature", '1',
            "--top_k", "3",
            "--top_p", "0.0",
        ])
    main()
