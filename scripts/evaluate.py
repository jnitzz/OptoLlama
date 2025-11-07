from typing import List, Dict, Any, Optional
import torch
from simulation_TMM_FAST import TMMContext

# ---------------------- metrics & helpers ----------------------
def token_accuracy(stacks, preds, eos, pad, msk):
    """
    Compute both global accuracy (weighted) and per-sample accuracy vector.

    Args:
        stacks: target ids [B, L]
        preds:  logits [B, L, V] or ids [B, L]
        eos:    int id of EOS
        pad:    int id of PAD
        msk:    int id of MSK

    Returns:
        global_acc : scalar float tensor (on CPU)
        per_sample : [B] float tensor (on CPU)
    """
    if preds.dim() == 3:
        preds = preds.argmax(dim=-1)

    L = min(stacks.size(1), preds.size(1))
    stacks = stacks[:, :L]
    preds  = preds[:,  :L]

    # positions strictly before first EOS
    is_eos = (stacks == eos)
    before_first_eos = (is_eos.cumsum(dim=1) == 0)

    # exclude PADs and MSKs
    valid = before_first_eos & (stacks != pad) & (stacks != msk)

    correct = (stacks == preds) & valid

    # --- per-sample accuracy ---
    per_correct = correct.sum(dim=1).float()
    per_total   = valid.sum(dim=1).clamp_min(1).float()
    per_sample  = (per_correct / per_total).detach().cpu()

    # --- global weighted accuracy ---
    global_acc = (per_correct.sum() / per_total.sum()).detach().cpu()

    return global_acc, per_sample


def masked_mae(x, y):
    # treat only finite y as valid (predicted_spectra)
    mask = torch.isfinite(y).all(dim=-1, keepdim=True)  # [B,W,1]
    valid = mask.expand_as(y)
    num = (torch.abs(x - torch.nan_to_num(y))).where(valid, torch.zeros_like(x)).sum(dim=1).sum(dim=1)
    den = valid.sum(dim=1).sum(dim=1).clamp_min(1)
    return num / den


def _simulate_spectra_ids(
    ids: torch.Tensor, tmm_ctx: TMMContext, *, eos: int, pad: int, msk: int
) -> torch.Tensor:
    """
    ids: [B, S] int tokens
    returns: [B, W, 3] float32   (R, A, T along the last dim)
    """
    tmm, wl, theta = tmm_ctx
    out = tmm(ids, wl, theta, eos=eos, pad=pad, msk=msk)  # typically [B, 3, W]
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).clamp_(0.0, 1.0)


# ---------------------- unified validation API ----------------------
@torch.no_grad()
def validate_model(
    model: torch.nn.Module,
    loader,
    *,
    mode: str,                              # 'NOSIM' or 'TMM_FAST'
    eos: int, pad: int, msk: int,
    device: torch.device,
    idx_to_token: Dict[int, str],
    tmm_ctx: Optional[TMMContext] = None,
    mc_samples: int = 1,
    rank: int = 0,
    world_size: int = 1,
    gather: bool = True,
) -> Dict[str, Any]:
    """
    Reusable validation w/ optional Monte-Carlo best-of-N and DDP gathering.

    Returns a dictionary with keys:
      - 'mean_acc' (float)
      - 'mean_mae' (float | None)  # only for TMM_FAST
      - 'results'  (list of per-example dicts)  # Present only on rank 0
    """
    is_ddp = (world_size > 1)
    model.eval()

    do_sim = (mode.upper() == 'TMM_FAST')
    results: List[Dict[str, Any]] = []

    sum_acc = 0.0
    sum_mae = 0.0
    n_batches = 0

    running_idx = 0  # for datasets not returning idxs

    for batch in loader:
        # dataset yields (spectra, stacks[, idxs])
        if len(batch) == 3:
            spectra, stacks, idxs = batch
        else:
            spectra, stacks = batch
            idxs = torch.arange(running_idx, running_idx + stacks.size(0))
            running_idx += stacks.size(0)

        spectra = spectra.to(device, non_blocking=True)  # [B,3,W]
        stacks  = stacks.to(device, non_blocking=True)   # [B,S]
        B = spectra.size(0)

        best_mae = torch.full((B,), float('inf'), device=device)
        best_pred_spectra = None
        best_pred_ids = None

        # ---- MC loop ----
        S = max(1, int(mc_samples))
        for s in range(S):
            logits_or_ids, _ = model(spectra)
            ids = logits_or_ids.argmax(dim=-1) if logits_or_ids.dim() == 3 else logits_or_ids

            if do_sim:
                pred = _simulate_spectra_ids(ids, tmm_ctx, eos=eos, pad=pad, msk=msk)  # [B,W,3]
                mae_s = masked_mae(spectra, pred)  # [B]
            else:
                pred = None
                mae_s = torch.zeros(B, device=device)

            # update best
            take = mae_s < best_mae
            best_mae = torch.where(take, mae_s, best_mae)
            if best_pred_spectra is None:
                best_pred_spectra = pred if pred is not None else None
                best_pred_ids = ids
            else:
                if pred is not None:
                    best_pred_spectra = torch.where(take.view(B, 1, 1), pred, best_pred_spectra)
                best_pred_ids = torch.where(take.view(B, 1), ids, best_pred_ids)

        # Metrics on best
        L = min(stacks.size(1), best_pred_ids.size(1))
        stacks_aligned = stacks[:, :L]
        ids_aligned = best_pred_ids[:, :L]

        acc_g, acc_vec = token_accuracy(stacks_aligned, ids_aligned, eos, pad, msk)
        sum_acc += float(acc_g)
        if do_sim and best_pred_spectra is not None:
            sum_mae += float(masked_mae(spectra, best_pred_spectra).mean().item())
        n_batches += 1

        # per-example records
        for i in range(B):
            tgt_ids = stacks_aligned[i].tolist()
            try:
                tgt_len = tgt_ids.index(eos)
            except ValueError:
                tgt_len = len(tgt_ids)
            tgt_core = [int(t) for t in tgt_ids[:tgt_len] if int(t) not in (pad, msk, eos)]
            tgt_tokens = [idx_to_token[int(t)] for t in tgt_core]

            pred_ids_i = ids_aligned[i].tolist()
            try:
                pred_len = pred_ids_i.index(eos)
            except ValueError:
                pred_len = len(pred_ids_i)
            pred_tokens = [idx_to_token[int(t)] for t in pred_ids_i[:pred_len] if int(t) not in (pad, msk, eos)]

            rec: Dict[str, Any] = {
                "dataset_index": int(idxs[i].item()),
                "acc": float(acc_vec[i].item()),
                "stack_target_tokens": tgt_tokens,
                "stack_pred_tokens": pred_tokens,
            }
            if do_sim and best_pred_spectra is not None:
                rec.update({
                    "mae": float(best_mae[i].item()),
                    "RAT_target_flat": spectra[i].reshape(-1).detach().cpu().numpy().tolist(),
                    "RAT_pred_flat":   best_pred_spectra[i].reshape(-1).detach().cpu().numpy().tolist(),
                    "RAT_target": spectra[i].detach().cpu().numpy().tolist(),
                    "RAT_pred":   best_pred_spectra[i].detach().cpu().numpy().tolist(),
                })
            results.append(rec)

    # DDP gather results
    if gather and is_ddp:
        gathered_lists = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(gathered_lists, results)
        if rank == 0:
            merged: List[Dict[str, Any]] = []
            for sub in gathered_lists:
                merged.extend(sub)
            results = merged

    out: Dict[str, Any] = {
        "mean_acc": (sum_acc / max(n_batches, 1)),
        "mean_mae": (sum_mae / max(n_batches, 1)) if do_sim else None,
    }
    if (not is_ddp) or rank == 0:
        out["results"] = results
    return out