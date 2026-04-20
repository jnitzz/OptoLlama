from typing import Any, Optional

import torch
# ruff: noqa: F401

import optollama.evaluation
import optollama.utils

from optollama.evaluation.simulation import TMMContext


def validate_and_setup(
    mode: str,
    tmm_ctx: Optional[TMMContext],
    track_step_mae: bool,
    model: torch.nn.Module,
    ddp: bool,
    rank: int,
    gather: bool,
    record_all_mc: bool,
) -> tuple[bool, bool]:
    """
    Validate inputs and configure the model for the prediction run.

    Checks that required arguments are consistent, enables per-step MAE
    tracking on the model if requested, and disables ``record_all_mc`` when
    it cannot be supported under DDP.

    Args
    ----
    mode : str
        Validation mode; ``"TMM_FAST"`` or ``"NO_SIM"``.
    tmm_ctx : TMMContext or None
        TMM context; required when ``mode="TMM_FAST"``.
    track_step_mae : bool
        Whether to track per-denoising-step MAE.
    model : torch.nn.Module
        The model being evaluated.
    ddp : bool
        Whether running under DistributedDataParallel.
    rank : int
        Global DDP rank.
    gather : bool
        Whether results will be gathered onto rank 0.
    record_all_mc : bool
        Whether to record all MC draws.

    Returns
    -------
    tuple[bool, bool]
        ``(do_sim, record_all_mc)`` — whether simulation is active and
        whether MC recording is active (may be downgraded to ``False``).

    Raises
    ------
    ValueError
        If ``mode="TMM_FAST"`` but ``tmm_ctx`` is ``None``, or if
        ``track_step_mae=True`` but ``mode`` is not ``"TMM_FAST"``.
    """
    do_sim = mode == "TMM_FAST"

    if do_sim and tmm_ctx is None:
        raise ValueError("tmm_ctx must be provided when mode='TMM_FAST'")
    if track_step_mae and not do_sim:
        raise ValueError("track_step_mae requires mode='TMM_FAST'")

    inner = model.module if hasattr(model, "module") else model
    if track_step_mae and hasattr(inner, "enable_step_mae"):
        inner.enable_step_mae(tmm_ctx)

    can_record = not ddp or (rank == 0 and not gather)
    if record_all_mc and not can_record:
        if rank == 0:
            print(
                "record_all_mc=True is disabled when DDP gather=True "
                "(too much data). Use gather=False or single-process."
            )
        record_all_mc = False

    return do_sim, record_all_mc


def unpack_batch(
    batch: tuple,
    running_idx: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Unpack a data-loader batch and move tensors to ``device``.

    Handles both 2-element ``(spectra, stacks)`` and 3-element
    ``(spectra, stacks, idxs)`` batches.

    Args
    ----
    batch : tuple
        Raw batch from the data loader.
    running_idx : int
        Counter used to synthesize indices when the dataset does not
        return them.
    device : torch.device
        Target device.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]
        ``(spectra, stacks, idxs, running_idx)`` where ``running_idx``
        is updated if synthetic indices were generated.
    """
    if len(batch) == 3:
        spectra, stacks, idxs = batch
    else:
        spectra, stacks = batch
        idxs = torch.arange(running_idx, running_idx + stacks.size(0))
        running_idx += stacks.size(0)

    spectra = spectra.to(device, non_blocking=True)
    stacks = stacks.to(device, non_blocking=True)

    return spectra, stacks, idxs, running_idx


def run_mc_batch(
    model: torch.nn.Module,
    spectra: torch.Tensor,
    mc_samples: int,
    do_sim: bool,
    tmm_ctx: Optional[TMMContext],
    eos: int,
    pad: int,
    msk: int,
    roi_mask: Optional[torch.Tensor],
    record_all_mc: bool,
    record_pred_spectra: bool,
    device: torch.device,
) -> tuple[dict, dict]:
    """
    Run the Monte-Carlo sampling loop for a single batch.

    Draws ``mc_samples`` predictions from the model and keeps the
    best-of-N result per example (lowest MAE when simulating, otherwise
    the last draw).

    Args
    ----
    model : torch.nn.Module
        The model to sample from.
    spectra : torch.Tensor
        Conditioning spectra, shape ``[B, 3, W]``.
    mc_samples : int
        Number of MC draws.
    do_sim : bool
        Whether to simulate spectra via TMM.
    tmm_ctx : TMMContext or None
        TMM context used when ``do_sim=True``.
    eos : int
        EOS token id.
    pad : int
        PAD token id.
    msk : int
        MASK token id.
    roi_mask : torch.Tensor or None
        Boolean wavelength mask for ROI-restricted MAE.
    record_all_mc : bool
        Whether to record every MC draw.
    record_pred_spectra : bool
        Whether to include predicted spectra in the MC recording.
    device : torch.device
        Device for intermediate tensors.

    Returns
    -------
    tuple[dict, dict]
        ``(best, draws)`` where:

        - ``best`` contains ``"mae"``, ``"ids"``, ``"pred_spectra"``
          (or ``None``), and ``"step_mae_traj"`` (or ``None``) for the
          winning MC draw per example.
        - ``draws`` contains lists ``"mae"``, ``"ids"``, ``"pred_spectra"``,
          ``"traj"`` accumulated over all MC draws (empty when
          ``record_all_mc=False``).
    """
    b = spectra.size(0)
    m = max(1, mc_samples)

    # Expand MC draws into the batch dimension so sampling and optional TMM
    # simulation run in one vectorized pass.
    spectra_mc = spectra.unsqueeze(1).expand(b, m, *spectra.shape[1:]).reshape(b * m, *spectra.shape[1:])
    logits_or_ids, mae_traj_s = model(spectra_mc)
    ids_flat = logits_or_ids.argmax(dim=-1) if logits_or_ids.dim() == 3 else logits_or_ids
    ids = ids_flat.view(b, m, -1)

    if do_sim:
        pred_flat = optollama.evaluation.simulation.simulate_token_sequence(ids_flat, tmm_ctx, eos=eos, pad=pad, msk=msk)
        mae_flat = optollama.evaluation.metrics.masked_mae_roi(spectra_mc, pred_flat, wl_mask=roi_mask)
        pred = pred_flat.view(b, m, *pred_flat.shape[1:])
        mae = mae_flat.view(b, m)
    else:
        pred = None
        mae = torch.zeros((b, m), device=device)

    traj = None
    if mae_traj_s is not None:
        traj = mae_traj_s.view(b, m, -1)

    draws = {"mae": [], "ids": [], "pred_spectra": [], "traj": []}
    if record_all_mc:
        for sample_idx in range(m):
            draws["mae"].append(mae[:, sample_idx].detach().cpu())
            draws["ids"].append(ids[:, sample_idx].detach().cpu())
            if do_sim and record_pred_spectra and pred is not None:
                draws["pred_spectra"].append(pred[:, sample_idx].detach().cpu())
            if traj is not None:
                draws["traj"].append(traj[:, sample_idx].detach().cpu())

    if do_sim:
        best_indices = mae.argmin(dim=1)
    else:
        # In NO_SIM mode all MAEs are zero, so pick the final MC draw.
        best_indices = torch.full((b,), m - 1, device=device, dtype=torch.long)

    batch_indices = torch.arange(b, device=device)
    best_mae = mae[batch_indices, best_indices]
    best_pred_ids = ids[batch_indices, best_indices]
    best_pred_spectra = pred[batch_indices, best_indices] if pred is not None else None
    best_step_mae_traj = traj[batch_indices, best_indices] if traj is not None else None

    best = {
        "mae": best_mae,
        "ids": best_pred_ids,
        "pred_spectra": best_pred_spectra,
        "step_mae_traj": best_step_mae_traj,
    }

    return best, draws


def accumulate_mc_draws(
    draws: dict,
    mc_samples: int,
    do_sim: bool,
    record_pred_spectra: bool,
    track_step_mae: bool,
    all_mc_mae: list,
    all_mc_ids: list,
    all_mc_pred: list,
    all_mc_traj: list,
) -> None:
    """
    Validate and accumulate per-batch MC draw lists into the global accumulators.

    Args
    ----
    draws : dict
        Per-draw recordings from :func:`_run_mc_batch`.
    mc_samples : int
        Expected number of MC draws.
    do_sim : bool
        Whether simulation was active.
    record_pred_spectra : bool
        Whether predicted spectra were recorded.
    track_step_mae : bool
        Whether step-wise MAE trajectories were recorded.
    all_mc_mae : list
        Global accumulator for MAE draws (mutated in-place).
    all_mc_ids : list
        Global accumulator for id draws (mutated in-place).
    all_mc_pred : list
        Global accumulator for predicted spectra draws (mutated in-place).
    all_mc_traj : list
        Global accumulator for MAE trajectory draws (mutated in-place).

    Raises
    ------
    RuntimeError
        If the number of recorded draws does not match ``mc_samples``.
    """
    m = max(1, mc_samples)
    if len(draws["mae"]) != m or len(draws["ids"]) != m:
        raise RuntimeError(
            f"MC recording mismatch: expected m={m}, "
            f"got mae={len(draws['mae'])} ids={len(draws['ids'])}"
        )
    all_mc_mae.append(draws["mae"])
    all_mc_ids.append(draws["ids"])

    if do_sim and record_pred_spectra:
        if len(draws["pred_spectra"]) != m:
            raise RuntimeError(
                f"MC pred recording mismatch: expected m={m}, "
                f"got pred={len(draws['pred_spectra'])}"
            )
        all_mc_pred.append(draws["pred_spectra"])

    if track_step_mae and len(draws["traj"]) == m:
        all_mc_traj.append(draws["traj"])


def build_example_record(
    i: int,
    idxs: torch.Tensor,
    stacks_aligned: torch.Tensor,
    ids_aligned: torch.Tensor,
    acc_vec: torch.Tensor,
    best_mae: torch.Tensor,
    best_pred_spectra: Optional[torch.Tensor],
    best_step_mae_traj: Optional[torch.Tensor],
    spectra: torch.Tensor,
    idx_to_token: dict[int, str],
    eos: int,
    pad: int,
    msk: int,
    do_sim: bool,
) -> dict:
    """
    Build a result record for a single example.

    Args
    ----
    i : int
        Index within the current batch.
    idxs : torch.Tensor
        Dataset indices for the batch.
    stacks_aligned : torch.Tensor
        Ground-truth token ids, shape ``[B, S]``.
    ids_aligned : torch.Tensor
        Predicted token ids, shape ``[B, S]``.
    acc_vec : torch.Tensor
        Per-example token accuracy, shape ``[B]``.
    best_mae : torch.Tensor
        Per-example best MAE, shape ``[B]``.
    best_pred_spectra : torch.Tensor or None
        Best predicted spectra, shape ``[B, 3, W]``, or ``None``.
    best_step_mae_traj : torch.Tensor or None
        Per-step MAE trajectory, shape ``[B, steps]``, or ``None``.
    spectra : torch.Tensor
        Target spectra, shape ``[B, 3, W]``.
    idx_to_token : dict[int, str]
        Vocabulary mapping.
    eos : int
        EOS token id.
    pad : int
        PAD token id.
    msk : int
        MASK token id.
    do_sim : bool
        Whether simulation was active.

    Returns
    -------
    dict
        Record with keys ``"dataset_index"``, ``"acc"``,
        ``"stack_target_tokens"``, ``"stack_pred_tokens"``, and
        optionally ``"mae"``, ``"rat_target"``, ``"rat_pred"``,
        ``"mae_traj"``.
    """
    special = {pad, msk, eos}

    tgt_ids = stacks_aligned[i].tolist()
    tgt_len = tgt_ids.index(eos) if eos in tgt_ids else len(tgt_ids)
    tgt_tokens = [idx_to_token[int(t)] for t in tgt_ids[:tgt_len] if int(t) not in special]

    pred_ids_i = ids_aligned[i].tolist()
    pred_len = pred_ids_i.index(eos) if eos in pred_ids_i else len(pred_ids_i)
    pred_tokens = [idx_to_token[int(t)] for t in pred_ids_i[:pred_len] if int(t) not in special]

    rec = {
        "dataset_index": int(idxs[i].item()),
        "acc": float(acc_vec[i].item()),
        "stack_target_tokens": tgt_tokens,
        "stack_pred_tokens": pred_tokens,
    }

    if do_sim and best_pred_spectra is not None:
        rec.update({
            "mae": float(best_mae[i].item()),
            "rat_target": spectra[i].detach().cpu().numpy().tolist(),
            "rat_pred": best_pred_spectra[i].detach().cpu().numpy().tolist(),
        })

    if best_step_mae_traj is not None:
        rec["mae_traj"] = best_step_mae_traj[i].detach().cpu().tolist()

    return rec


def gather_ddp_results(results: list, world_size: int, rank: int) -> list:
    """
    Gather per-example result records from all DDP ranks onto rank 0.

    Args
    ----
    results : list
        Local per-example records on this rank.
    world_size : int
        Total number of DDP processes.
    rank : int
        Global rank of this process.

    Returns
    -------
    list
        Merged list of all records (only meaningful on rank 0).
    """
    gathered = [[] for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered, results)
    if rank == 0:
        merged: list = []
        for sub in gathered:
            merged.extend(sub)

        return merged
    return results


def reduce_ddp_metric_sums(
    total_correct: float,
    total_tokens: float,
    total_mae: float,
    total_mae_examples: float,
    ddp: bool,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """
    Reduce metric numerators and denominators across DDP ranks.

    Args
    ----
    total_correct : float
        Local number of correct valid tokens.
    total_tokens : float
        Local number of valid tokens.
    total_mae : float
        Local sum of per-example MAE values.
    total_mae_examples : float
        Local number of examples contributing to MAE.
    ddp : bool
        Whether DistributedDataParallel is active.
    device : torch.device
        Device on which to allocate the reduction tensor.

    Returns
    -------
    tuple[float, float, float, float]
        Globally reduced metric sums.
    """
    if not ddp:
        return total_correct, total_tokens, total_mae, total_mae_examples

    totals = torch.tensor(
        [total_correct, total_tokens, total_mae, total_mae_examples],
        device=device,
        dtype=torch.float64,
    )
    torch.distributed.all_reduce(totals)

    return tuple(float(v.item()) for v in totals)


def assemble_mc_grids(
    all_mc_mae: list,
    all_mc_ids: list,
    all_mc_pred: list,
    all_mc_traj: list,
    m: int,
    do_sim: bool,
    record_pred_spectra: bool,
    track_step_mae: bool,
) -> dict:
    """
    Assemble per-batch MC draw lists into ``[N, m, ...]`` grid tensors.

    Args
    ----
    all_mc_mae : list
        List over batches; each element is a list of ``m`` tensors ``[B]``.
    all_mc_ids : list
        List over batches; each element is a list of ``m`` tensors ``[B, S]``.
    all_mc_pred : list
        List over batches; each element is a list of ``m`` tensors ``[B, 3, W]``.
    all_mc_traj : list
        List over batches; each element is a list of ``m`` tensors ``[B, steps]``.
    m : int
        Number of MC samples.
    do_sim : bool
        Whether simulation was active.
    record_pred_spectra : bool
        Whether predicted spectra were recorded.
    track_step_mae : bool
        Whether step-wise MAE trajectories were recorded.

    Returns
    -------
    dict
        Dictionary with keys ``"mae_grid"``, ``"ids_grid"``, and
        optionally ``"pred_spectra_grid"`` and ``"mae_traj_grid"``.
    """
    def _concat_and_stack(per_batch: list) -> torch.Tensor:
        # per_batch: list[list[Tensor[B,...]]] → [N, m, ...]
        per_s = [torch.cat([b[s] for b in per_batch], dim=0) for s in range(m)]
        return torch.stack(per_s, dim=0).transpose(0, 1).contiguous()

    grids: dict[str, torch.Tensor] = {
        "mae_grid": _concat_and_stack(all_mc_mae).to(torch.float32),
        "ids_grid": _concat_and_stack(all_mc_ids).to(torch.long),
    }

    if do_sim and record_pred_spectra and all_mc_pred:
        grids["pred_spectra_grid"] = _concat_and_stack(all_mc_pred).to(torch.float32)

    if track_step_mae and all_mc_traj:
        grids["mae_traj_grid"] = _concat_and_stack(all_mc_traj).to(torch.float32)

    return grids


@torch.no_grad()
def model_prediction(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    mode: str,
    eos: int,
    pad: int,
    msk: int,
    idx_to_token: dict[int, str],
    tmm_ctx: Optional[TMMContext] = None,
    mc_samples: int = 1,
    track_step_mae: bool = False,
    roi_mask: Optional[torch.Tensor] = None,
    record_all_mc: bool = False,
    record_pred_spectra: bool = True,
    rank: int = 0,
    world_size: int = 1,
    gather: bool = True,
) -> dict[str, Any]:
    """
    Prediction routine with optional Monte-Carlo best-of-N and DDP gathering.

    In mode='NO_SIM', MAE is set to 0.0 for all MC samples (grid still returned for ids).

    Args
    ----
    model : torch.nn.Module
        The model to evaluate.
    loader : torch.utils.data.DataLoader
        The test data loader.
    device : torch.device
        The torch device used to run the simulation on.
    mode : str
        Validation mode; either ``"TMM_FAST"`` (simulate spectra) or
        ``"NO_SIM"`` (skip simulation).
    eos : int
        The index of the EOS token.
    pad : int
        The index of the PAD token.
    msk : int
        The index of the MASK token.
    idx_to_token : dict[int, str]
        Mapping from token indices to their human-readable string
        representation.
    tmm_ctx : TMMContext, optional
        Bundled TMM model and optical grid; required when
        ``mode="TMM_FAST"``.
    mc_samples : int
        Number of Monte-Carlo samples per example (best-of-N selection).
    track_step_mae : bool
        If ``True``, record per-denoising-step MAE (requires
        ``mode="TMM_FAST"``).
    roi_mask : torch.Tensor, optional
        Boolean wavelength mask for ROI-restricted MAE computation.
    record_all_mc : bool
        If ``True``, store raw ids and MAE for every MC draw.
    record_pred_spectra : bool
        If ``True``, include predicted spectra in the MC grid.
    rank : int
        Global DDP rank (default: ``0``).
    world_size : int
        Total number of DDP processes (default: ``1``).
    gather : bool
        If ``True``, gather results onto rank 0 (default: ``True``).

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:

        - ``"mean_acc"`` (float): mean token accuracy.
        - ``"mean_mae"`` (float or None): mean MAE; ``None`` when
          ``mode="NO_SIM"``.
        - ``"results"`` (list[dict]): per-example records (rank 0 only).

        Optional keys (rank 0 only) when ``record_all_mc=True``:

        - ``"mae_grid"`` — shape ``[N, m]``, float.
        - ``"ids_grid"`` — shape ``[N, m, S]``, long.
        - ``"pred_spectra_grid"`` — shape ``[N, m, 3, W]``, float
          (only when ``do_sim`` and ``record_pred_spectra``).
        - ``"mae_traj_grid"`` — shape ``[N, m, steps]``, float
          (only when ``track_step_mae=True`` and model provides it).
    """
    ddp = optollama.utils.is_ddp()
    do_sim, record_all_mc = validate_and_setup(
        mode, tmm_ctx, track_step_mae, model, ddp, rank, gather, record_all_mc
    )

    all_mc_mae, all_mc_ids, all_mc_pred, all_mc_traj, results = [], [], [], [], []
    total_correct, total_tokens = 0.0, 0.0
    total_mae, total_mae_examples, running_idx = 0.0, 0.0, 0

    for batch in loader:
        spectra, stacks, idxs, running_idx = unpack_batch(batch, running_idx, device)
        b = spectra.size(0)

        best, draws = run_mc_batch(
            model, spectra, mc_samples, do_sim, tmm_ctx,
            eos, pad, msk, roi_mask, record_all_mc, record_pred_spectra, device,
        )

        if record_all_mc:
            accumulate_mc_draws(
                draws, mc_samples, do_sim, record_pred_spectra, track_step_mae,
                all_mc_mae, all_mc_ids, all_mc_pred, all_mc_traj,
            )

        len_seq = min(stacks.size(1), best["ids"].size(1))
        stacks_aligned = stacks[:, :len_seq]
        ids_aligned = best["ids"][:, :len_seq]

        correct_count, total_count, _, _ = optollama.evaluation.metrics.token_accuracy_counts(
            stacks_aligned, ids_aligned, eos, pad, msk
        )
        acc_g, acc_vec = optollama.evaluation.metrics.token_accuracy(stacks_aligned, ids_aligned, eos, pad, msk)
        total_correct += float(correct_count.item())
        total_tokens += float(total_count.item())
        if do_sim and best["pred_spectra"] is not None:
            mae_vec = optollama.evaluation.metrics.masked_mae_roi(spectra, best["pred_spectra"], wl_mask=roi_mask)
            total_mae += float(mae_vec.sum().item())
            total_mae_examples += float(mae_vec.numel())

        for i in range(b):
            results.append(build_example_record(
                i, idxs, stacks_aligned, ids_aligned, acc_vec,
                best["mae"], best["pred_spectra"], best["step_mae_traj"],
                spectra, idx_to_token, eos, pad, msk, do_sim,
            ))

    if gather and ddp:
        results = gather_ddp_results(results, world_size, rank)

    total_correct, total_tokens, total_mae, total_mae_examples = reduce_ddp_metric_sums(
        total_correct,
        total_tokens,
        total_mae,
        total_mae_examples,
        ddp,
        device,
    )

    out: dict[str, Any] = {
        "mean_acc": total_correct / max(total_tokens, 1.0),
        "mean_mae": (total_mae / max(total_mae_examples, 1.0)) if do_sim else None,
    }
    if not ddp or rank == 0:
        out["results"] = results
        if record_all_mc:
            out.update(assemble_mc_grids(
                all_mc_mae, all_mc_ids, all_mc_pred, all_mc_traj,
                max(1, mc_samples), do_sim, record_pred_spectra, track_step_mae,
            ))

    return out
