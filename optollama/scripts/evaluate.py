from typing import Any, Dict, List, Literal, Optional

import torch

from optollama.evaluation.metrics import masked_mae_roi, token_accuracy
from optollama.utils.simulation_TMM_FAST import TMMContext


def simulate_spectra_ids(ids: torch.Tensor, tmm_ctx: TMMContext, *, eos: int, pad: int, msk: int) -> torch.Tensor:
    """
    Simulate the RAT spectra from the token ids.

    ids: [B, S] int tokens
    returns: [B, W, 3] float32   (R, A, T along the last dim)
    """
    tmm, wl, theta = tmm_ctx
    out = tmm(ids, wl, theta, eos=eos, pad=pad, msk=msk)  # typically [B, 3, W]

    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).clamp_(0.0, 1.0)


def stack_mc_lists(x: List[torch.Tensor]) -> torch.Tensor:
    """
    Move stack lists of tensors in MC grid.

    Moves stacks of [N,...] tensors into [N, m, ...] (MC dim in axis=1).
    """
    if not x:
        raise ValueError("Expected non-empty list to stack.")
    # input is list length m of [N,...]
    # stack -> [m,N,...] then transpose -> [N,m,...]
    return torch.stack(x, dim=0).transpose(0, 1).contiguous()


@torch.no_grad()
def validate_model(
    model: torch.nn.Module,
    loader: Any,
    *,
    mode: Literal["NOSIM", "TMM_FAST"],
    eos: int,
    pad: int,
    msk: int,
    device: torch.device,
    idx_to_token: Dict[int, str],
    tmm_ctx: Optional[TMMContext] = None,
    mc_samples: int = 1,
    rank: int = 0,
    world_size: int = 1,
    gather: bool = True,
    track_step_mae: bool = False,
    roi_mask: Optional[torch.Tensor] = None,
    record_all_mc: bool = False,
    record_pred_spectra: bool = True,
) -> Dict[str, Any]:
    """
    Reusable validation w/ optional Monte-Carlo best-of-N and DDP gathering.

    Returns a dictionary with keys:
      - 'mean_acc' (float)
      - 'mean_mae' (float | None)  # only for TMM_FAST
      - 'results'  (list of per-example dicts)  # Present only on rank 0

    Optional (rank 0 only) when record_all_mc=True:
      - 'mae_grid' : [N, m] float
      - 'ids_grid' : [N, m, S] long
      - 'pred_spectra_grid' : [N, m, 3, W] float (only if do_sim and record_pred_spectra)
      - 'mae_traj_grid' : [N, m, steps] float (only if track_step_mae True and model provides it)

    Notes
    -----
      - In mode='NOSIM', MAE is set to 0 for all MC samples (grid still returned for ids).
    """
    is_ddp = world_size > 1
    model.eval()

    do_sim = mode.upper() == "TMM_FAST"
    if do_sim and tmm_ctx is None:
        raise ValueError("tmm_ctx must be provided when mode='TMM_FAST'")

    if track_step_mae and not do_sim:
        raise ValueError("track_step_mae requires mode='TMM_FAST'")

    # If the underlying model knows how to track step-wise MAE, enable it
    inner = model.module if hasattr(model, "module") else model
    if track_step_mae and hasattr(inner, "enable_step_mae"):
        inner.enable_step_mae(tmm_ctx)

    # ---- optional MC grid recording (rank 0 / single-process only) ----
    # We record raw ids + MAE (+pred spectra) for *every* MC draw, then stack at end.
    # If DDP + gather, we only support recording on rank 0 after gathering tensors is non-trivial
    # (we'd need all_gather on potentially large tensors). For now, we gate it.
    can_record = (not is_ddp) or (rank == 0 and not gather)
    if record_all_mc and not can_record:
        if rank == 0:
            print("record_all_mc=True is disabled when DDP gather=True (too much data). Use gather=False or single-process.")
        record_all_mc = False

    all_mc_mae: List[List[torch.Tensor]] = []  # list over batches, each: list over s of [B]
    all_mc_ids: List[List[torch.Tensor]] = []  # list over batches, each: list over s of [B,S]
    all_mc_pred: List[List[torch.Tensor]] = []  # list over batches, each: list over s of [B,3,W]
    all_mc_traj: List[List[torch.Tensor]] = []  # list over batches, each: list over s of [B,steps]

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
        stacks = stacks.to(device, non_blocking=True)  # [B,S]
        b = spectra.size(0)

        best_mae = torch.full((b,), float("inf"), device=device)
        best_pred_spectra: Optional[torch.Tensor] = None
        best_pred_ids: Optional[torch.Tensor] = None
        best_step_mae_traj: Optional[torch.Tensor] = None

        # per-batch MC storage
        mc_mae_s: List[torch.Tensor] = []
        mc_ids_s: List[torch.Tensor] = []
        mc_pred_s: List[torch.Tensor] = []
        mc_traj_s: List[torch.Tensor] = []

        # ---- MC loop ----
        for s in range(max(1, int(mc_samples))):
            logits_or_ids, mae_traj_s = model(spectra)

            ids = logits_or_ids.argmax(dim=-1) if logits_or_ids.dim() == 3 else logits_or_ids

            if do_sim:
                assert tmm_ctx is not None
                pred = simulate_spectra_ids(ids, tmm_ctx, eos=eos, pad=pad, msk=msk)  # [B,3,W]
                mae_s = masked_mae_roi(spectra, pred, wl_mask=roi_mask)  # [B]
            else:
                pred = None
                mae_s = torch.zeros(b, device=device)

            # ---- record this MC draw (optional) ----
            if record_all_mc:
                mc_mae_s.append(mae_s.detach().cpu())
                mc_ids_s.append(ids.detach().cpu())
                if do_sim and record_pred_spectra:
                    assert pred is not None
                    mc_pred_s.append(pred.detach().cpu())
                if mae_traj_s is not None:
                    # mae_traj_s: [B,steps]
                    mc_traj_s.append(mae_traj_s.detach().cpu())

            # update best
            take_mae = mae_s < best_mae
            take = take_mae
            best_mae = torch.where(take, mae_s, best_mae)

            if best_pred_spectra is None:
                if pred is not None:
                    best_pred_spectra = pred
                best_pred_ids = ids
            else:
                if pred is not None and best_pred_spectra is not None:
                    best_pred_spectra = torch.where(take.view(b, 1, 1), pred, best_pred_spectra)
                if best_pred_ids is None:
                    best_pred_ids = ids
                else:
                    best_pred_ids = torch.where(take.view(b, 1), ids, best_pred_ids)

            # best step-wise MAE trajectory
            if mae_traj_s is not None:
                if best_step_mae_traj is None:
                    best_step_mae_traj = mae_traj_s
                else:
                    # keep the trajectory from the MC sample that wins for each element in the batch
                    best_step_mae_traj = torch.where(
                        take.view(b, 1),  # [B,1] broadcast over steps
                        mae_traj_s,  # [B,steps] from current MC sample
                        best_step_mae_traj,  # [B,steps] from previous best
                    )

        # finalize per-batch MC recording
        if record_all_mc:
            # Ensure lists have consistent length m
            m = max(1, int(mc_samples))
            if len(mc_mae_s) != m or len(mc_ids_s) != m:
                raise RuntimeError(f"MC recording mismatch: expected m={m}, got mae={len(mc_mae_s)} ids={len(mc_ids_s)}")
            all_mc_mae.append(mc_mae_s)
            all_mc_ids.append(mc_ids_s)
            if do_sim and record_pred_spectra:
                if len(mc_pred_s) != m:
                    raise RuntimeError(f"MC pred recording mismatch: expected m={m}, got pred={len(mc_pred_s)}")
                all_mc_pred.append(mc_pred_s)
            if track_step_mae:
                # if enabled, model should return mae_traj; if not, we skip silently
                if len(mc_traj_s) == m:
                    all_mc_traj.append(mc_traj_s)

        # Metrics on best
        assert best_pred_ids is not None, "MC loop did not produce any predictions"
        len_seq = min(stacks.size(1), best_pred_ids.size(1))
        stacks_aligned = stacks[:, :len_seq]
        ids_aligned = best_pred_ids[:, :len_seq]

        acc_g, acc_vec = token_accuracy(stacks_aligned, ids_aligned, eos, pad, msk)
        sum_acc += float(acc_g)  # TODO check if to use .item() here instead
        if do_sim and best_pred_spectra is not None:
            sum_mae += float(masked_mae_roi(spectra, best_pred_spectra, wl_mask=roi_mask).mean().item())
        n_batches += 1

        # per-example records
        for i in range(b):
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
                rec.update(
                    {
                        "mae": float(best_mae[i].item()),
                        "rat_target": spectra[i].detach().cpu().numpy().tolist(),
                        "rat_pred": best_pred_spectra[i].detach().cpu().numpy().tolist(),
                    }
                )
            if best_step_mae_traj is not None:
                rec["mae_traj"] = best_step_mae_traj[i].detach().cpu().tolist()
            results.append(rec)

    # DDP gather results
    if gather and is_ddp:
        gathered_lists: List[List[Dict[str, Any]]] = [[] for _ in range(world_size)]
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

    # ---- Attach MC grids (rank 0 / single-process) ----
    if record_all_mc and ((not is_ddp) or rank == 0):
        # Concatenate batches for each MC sample:
        # all_mc_* is list over batches; each batch is list over s of [B,...]
        # We want list over s of [N,...] then stack -> [N,m,...]
        m = max(1, int(mc_samples))

        # Build per-sample concatenated tensors
        mae_per_s: List[torch.Tensor] = []
        ids_per_s: List[torch.Tensor] = []
        pred_per_s: List[torch.Tensor] = []
        traj_per_s: List[torch.Tensor] = []

        for s in range(m):
            mae_per_s.append(torch.cat([batch_list[s] for batch_list in all_mc_mae], dim=0))  # [N]
            ids_per_s.append(torch.cat([batch_list[s] for batch_list in all_mc_ids], dim=0))  # [N,S]

        if do_sim and record_pred_spectra and all_mc_pred:
            for s in range(m):
                pred_per_s.append(torch.cat([batch_list[s] for batch_list in all_mc_pred], dim=0))  # [N,3,W]

        if track_step_mae and all_mc_traj:
            # only if we actually recorded all m trajectories in each batch
            for s in range(m):
                traj_per_s.append(torch.cat([batch_list[s] for batch_list in all_mc_traj], dim=0))  # [N,steps]

        out["mae_grid"] = stack_mc_lists(mae_per_s).to(torch.float32)  # [N,m]
        out["ids_grid"] = stack_mc_lists(ids_per_s).to(torch.long)  # [N,m,S]
        if do_sim and record_pred_spectra and pred_per_s:
            out["pred_spectra_grid"] = stack_mc_lists(pred_per_s).to(torch.float32)  # [N,m,3,W]
        if traj_per_s:
            out["mae_traj_grid"] = stack_mc_lists(traj_per_s).to(torch.float32)  # [N,m,steps]

    return out
