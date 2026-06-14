import time

from typing import Any, Optional

import torch
import tqdm
# ruff: noqa: F401

import optollama.data
import optollama.evaluation
import optollama.utils

from optollama.evaluation.simulation import TMMContext


def sync_for_timing(device: torch.device, enabled: bool) -> None:
    """
    Synchronize CUDA work when wall-clock timing is requested.
    """
    if enabled and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def split_model_sample_output(
    output: Any,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Normalize model sampling output across legacy and factored formats.

    Legacy models return ``(ids_or_logits, mae_traj)``. The continuous
    thickness path returns a dict with hard token ``ids`` plus
    ``thickness_nm``.
    """
    thickness_nm = None
    active_prob = None
    mae_traj = None

    if isinstance(output, dict):
        if "ids" in output:
            logits_or_ids = output["ids"]
        elif "logits" in output:
            logits_or_ids = output["logits"]
        else:
            raise ValueError("Model sample output dict must contain 'ids' or 'logits'.")
        if "mae_traj" in output:
            mae_traj = output["mae_traj"]
        elif "step_mae_traj" in output:
            mae_traj = output["step_mae_traj"]
        thickness_nm = output.get("thickness_nm")
        active_prob = output.get("active_prob")
    elif isinstance(output, tuple):
        if len(output) == 2:
            logits_or_ids, mae_traj = output
        elif len(output) == 3:
            logits_or_ids, mae_traj, thickness_nm = output
        elif len(output) == 4:
            logits_or_ids, mae_traj, thickness_nm, active_prob = output
        else:
            raise ValueError(f"Unsupported model sample tuple length: {len(output)}")
    else:
        logits_or_ids = output

    ids = logits_or_ids.argmax(dim=-1) if logits_or_ids.dim() == 3 else logits_or_ids
    return ids, mae_traj, thickness_nm, active_prob


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
    source_wavelengths: Optional[torch.Tensor],
    common_mae_wavelengths: Optional[torch.Tensor],
    record_all_mc: bool,
    record_pred_spectra: bool,
    device: torch.device,
    score_spectra: Optional[torch.Tensor] = None,
    mae_channel_mask: Optional[torch.Tensor] = None,
    deduplicate_stacks: bool = False,
    profile_timing: bool = False,
) -> tuple[dict, dict, dict[str, float]]:
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
    source_wavelengths : torch.Tensor or None
        Native wavelength grid for the input/predicted spectra.
    common_mae_wavelengths : torch.Tensor or None
        Optional comparison grid used to compute MAE independent of the
        native config wavelength spacing.
    record_all_mc : bool
        Whether to record every MC draw.
    record_pred_spectra : bool
        Whether to include predicted spectra in the MC recording.
    device : torch.device
        Device for intermediate tensors.
    score_spectra : torch.Tensor, optional
        Spectra used for MC ranking and MAE reporting. If omitted, the
        conditioning ``spectra`` are used.
    deduplicate_stacks : bool
        Whether to simulate unique predicted stacks only and scatter spectra
        back to duplicate MC entries.
    profile_timing : bool
        Whether to synchronize CUDA and record coarse timing.

    Returns
    -------
    tuple[dict, dict, dict[str, float]]
        ``(best, draws, timing)`` where:

        - ``best`` contains ``"mae"``, ``"ids"``, ``"pred_spectra"``
          (or ``None``), and ``"step_mae_traj"`` (or ``None``) for the
          winning MC draw per example.
        - ``draws`` contains lists ``"mae"``, ``"ids"``, ``"pred_spectra"``,
          ``"traj"`` accumulated over all MC draws (empty when
          ``record_all_mc=False``).
    """
    b = spectra.size(0)
    m = max(1, mc_samples)
    timing = {
        "model_s": 0.0,
        "tmm_s": 0.0,
        "record_s": 0.0,
        "dedup_input": 0.0,
        "dedup_unique": 0.0,
    }

    # Expand MC draws into the batch dimension so sampling and optional TMM
    # simulation run in one vectorized pass.
    score_spectra = spectra if score_spectra is None else score_spectra
    spectra_mc = spectra.unsqueeze(1).expand(b, m, *spectra.shape[1:]).reshape(b * m, *spectra.shape[1:])
    score_spectra_mc = score_spectra.unsqueeze(1).expand(b, m, *score_spectra.shape[1:]).reshape(
        b * m,
        *score_spectra.shape[1:],
    )
    sync_for_timing(device, profile_timing)
    t0 = time.perf_counter()
    model_output = model(spectra_mc)
    ids_flat, mae_traj_s, thickness_flat, active_flat = split_model_sample_output(model_output)
    ids = ids_flat.view(b, m, -1)
    thickness = thickness_flat.view(b, m, -1) if thickness_flat is not None else None
    active = active_flat.view(b, m, -1) if active_flat is not None else None
    sync_for_timing(device, profile_timing)
    timing["model_s"] = time.perf_counter() - t0

    inner_model = model.module if hasattr(model, "module") else model
    use_continuous_thickness_for_tmm = bool(
        thickness_flat is not None and getattr(inner_model, "continuous_thickness_use_for_tmm", True)
    )
    save_continuous_thickness = bool(
        thickness is not None and getattr(inner_model, "continuous_thickness_save_in_results", True)
    )
    save_active_prob = bool(active is not None and save_continuous_thickness)

    if do_sim:
        sync_for_timing(device, profile_timing)
        t0 = time.perf_counter()
        sim_device = tmm_ctx.wl.device if tmm_ctx is not None else device
        ids_for_sim = ids_flat.to(sim_device, non_blocking=True) if ids_flat.device != sim_device else ids_flat
        thickness_for_sim = None
        if use_continuous_thickness_for_tmm:
            thickness_for_sim = (
                thickness_flat.to(sim_device, non_blocking=True)
                if thickness_flat.device != sim_device
                else thickness_flat
            )
        if deduplicate_stacks and thickness_for_sim is None:
            unique_ids, inverse = torch.unique(ids_for_sim, dim=0, return_inverse=True)
            pred_unique = optollama.evaluation.simulation.simulate_token_sequence(
                unique_ids,
                tmm_ctx,
                eos=eos,
                pad=pad,
                msk=msk,
            )
            pred_flat = pred_unique[inverse].to(device, non_blocking=True)
            timing["dedup_input"] = float(ids_flat.size(0))
            timing["dedup_unique"] = float(unique_ids.size(0))
        else:
            pred_flat = optollama.evaluation.simulation.simulate_token_sequence(
                ids_for_sim,
                tmm_ctx,
                eos=eos,
                pad=pad,
                msk=msk,
                thickness_override=thickness_for_sim,
            ).to(device, non_blocking=True)
            timing["dedup_input"] = float(ids_flat.size(0))
            timing["dedup_unique"] = float(ids_flat.size(0))
        sync_for_timing(device, profile_timing)
        timing["tmm_s"] = time.perf_counter() - t0
        mae_flat = optollama.evaluation.metrics.masked_mae_roi(
            score_spectra_mc,
            pred_flat,
            wl_mask=roi_mask,
            channel_mask=mae_channel_mask,
        )
        mae_common_flat = None
        if source_wavelengths is not None and common_mae_wavelengths is not None:
            mae_common_flat = optollama.evaluation.metrics.resampled_mae(
                score_spectra_mc,
                pred_flat,
                source_wavelengths=source_wavelengths,
                target_wavelengths=common_mae_wavelengths,
                channel_mask=mae_channel_mask,
            )
        pred = pred_flat.view(b, m, *pred_flat.shape[1:])
        mae = mae_flat.view(b, m)
        mae_common = mae_common_flat.view(b, m) if mae_common_flat is not None else None
    else:
        pred = None
        mae = torch.zeros((b, m), device=device)
        mae_common = None

    traj = None
    if mae_traj_s is not None:
        traj = mae_traj_s.view(b, m, -1)

    draws = {
        "mae": [],
        "mae_common": [],
        "ids": [],
        "thickness_nm": [],
        "active_prob": [],
        "pred_spectra": [],
        "traj": [],
    }
    if record_all_mc:
        sync_for_timing(device, profile_timing)
        t0 = time.perf_counter()
        for sample_idx in range(m):
            draws["mae"].append(mae[:, sample_idx].detach().cpu())
            if mae_common is not None:
                draws["mae_common"].append(mae_common[:, sample_idx].detach().cpu())
            draws["ids"].append(ids[:, sample_idx].detach().cpu())
            if save_continuous_thickness:
                draws["thickness_nm"].append(thickness[:, sample_idx].detach().cpu())
            if save_active_prob:
                draws["active_prob"].append(active[:, sample_idx].detach().cpu())
            if do_sim and record_pred_spectra and pred is not None:
                draws["pred_spectra"].append(pred[:, sample_idx].detach().cpu())
            if traj is not None:
                draws["traj"].append(traj[:, sample_idx].detach().cpu())
        sync_for_timing(device, profile_timing)
        timing["record_s"] = time.perf_counter() - t0

    if do_sim:
        best_indices = mae.argmin(dim=1)
    else:
        # In NO_SIM mode all MAEs are zero, so pick the final MC draw.
        best_indices = torch.full((b,), m - 1, device=device, dtype=torch.long)

    batch_indices = torch.arange(b, device=device)
    best_mae = mae[batch_indices, best_indices]
    best_mae_common = mae_common[batch_indices, best_indices] if mae_common is not None else None
    best_pred_ids = ids[batch_indices, best_indices]
    best_thickness_nm = thickness[batch_indices, best_indices] if save_continuous_thickness else None
    best_active_prob = active[batch_indices, best_indices] if save_active_prob else None
    best_pred_spectra = pred[batch_indices, best_indices] if pred is not None else None
    best_step_mae_traj = traj[batch_indices, best_indices] if traj is not None else None

    best = {
        "mae": best_mae,
        "mae_common": best_mae_common,
        "ids": best_pred_ids,
        "thickness_nm": best_thickness_nm,
        "active_prob": best_active_prob,
        "pred_spectra": best_pred_spectra,
        "step_mae_traj": best_step_mae_traj,
    }

    return best, draws, timing


def accumulate_mc_draws(
    draws: dict,
    mc_samples: int,
    do_sim: bool,
    record_pred_spectra: bool,
    track_step_mae: bool,
    all_mc_mae: list,
    all_mc_mae_common: list,
    all_mc_ids: list,
    all_mc_pred: list,
    all_mc_traj: list,
    all_mc_thickness: Optional[list] = None,
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
    all_mc_mae_common : list
        Global accumulator for common-grid MAE draws (mutated in-place).
    all_mc_ids : list
        Global accumulator for id draws (mutated in-place).
    all_mc_pred : list
        Global accumulator for predicted spectra draws (mutated in-place).
    all_mc_traj : list
        Global accumulator for MAE trajectory draws (mutated in-place).
    all_mc_thickness : list, optional
        Global accumulator for continuous thickness draws (mutated in-place).

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
    if draws.get("mae_common"):
        if len(draws["mae_common"]) != m:
            raise RuntimeError(
                f"MC common-MAE recording mismatch: expected m={m}, "
                f"got mae_common={len(draws['mae_common'])}"
            )
        all_mc_mae_common.append(draws["mae_common"])
    all_mc_ids.append(draws["ids"])
    if all_mc_thickness is not None and draws.get("thickness_nm"):
        if len(draws["thickness_nm"]) != m:
            raise RuntimeError(
                f"MC thickness recording mismatch: expected m={m}, "
                f"got thickness_nm={len(draws['thickness_nm'])}"
            )
        all_mc_thickness.append(draws["thickness_nm"])

    if do_sim and record_pred_spectra:
        if len(draws["pred_spectra"]) != m:
            raise RuntimeError(
                f"MC pred recording mismatch: expected m={m}, "
                f"got pred={len(draws['pred_spectra'])}"
            )
        all_mc_pred.append(draws["pred_spectra"])

    if track_step_mae and len(draws["traj"]) == m:
        all_mc_traj.append(draws["traj"])


def continuous_stack_records(
    pred_ids: list[int],
    thickness_nm: torch.Tensor,
    idx_to_token: dict[int, str],
    eos: int,
    special: set[int],
    active_prob: Optional[torch.Tensor] = None,
) -> list[dict[str, Any]]:
    """
    Convert sampled hard materials plus continuous thicknesses to JSON records.
    """
    pred_len = pred_ids.index(eos) if eos in pred_ids else len(pred_ids)
    records: list[dict[str, Any]] = []
    thickness_values = thickness_nm.detach().cpu().tolist()
    active_values = active_prob.detach().cpu().tolist() if active_prob is not None else None

    for pos, token_id_raw in enumerate(pred_ids[:pred_len]):
        token_id = int(token_id_raw)
        if token_id in special:
            continue
        token = idx_to_token[int(token_id)]
        parts = optollama.data.layer_token_parts(token)
        if parts is None:
            continue
        material, token_thickness_nm = parts
        layer = {
            "material": material,
            "thickness_nm": float(thickness_values[pos]),
            "token": token,
            "token_thickness_nm": float(token_thickness_nm),
        }
        if active_values is not None:
            layer["active_prob"] = float(active_values[pos])
        records.append(layer)

    return records


def build_example_record(
    i: int,
    idxs: torch.Tensor,
    stacks_aligned: torch.Tensor,
    ids_aligned: torch.Tensor,
    acc_vec: torch.Tensor,
    best_mae: torch.Tensor,
    best_mae_common: Optional[torch.Tensor],
    best_pred_spectra: Optional[torch.Tensor],
    best_step_mae_traj: Optional[torch.Tensor],
    best_thickness_nm: Optional[torch.Tensor],
    best_active_prob: Optional[torch.Tensor],
    score_spectra: torch.Tensor,
    idx_to_token: dict[int, str],
    eos: int,
    pad: int,
    msk: int,
    do_sim: bool,
    conditioning_spectra: Optional[torch.Tensor] = None,
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
    best_mae_common : torch.Tensor or None
        Per-example best common-grid MAE, shape ``[B]``, or ``None``.
    best_pred_spectra : torch.Tensor or None
        Best predicted spectra, shape ``[B, 3, W]``, or ``None``.
    best_step_mae_traj : torch.Tensor or None
        Per-step MAE trajectory, shape ``[B, steps]``, or ``None``.
    best_thickness_nm : torch.Tensor or None
        Continuous thicknesses for the best predicted hard-token sequence,
        shape ``[B, S]``, or ``None``.
    best_active_prob : torch.Tensor or None
        Active probabilities for the best predicted hard-token sequence,
        shape ``[B, S]``, or ``None``.
    score_spectra : torch.Tensor
        Spectra used for MC ranking and MAE reporting, shape ``[B, 3, W]``.
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
    conditioning_spectra : torch.Tensor, optional
        Spectra passed to the model. Stored only when it differs from the
        scoring target.

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
    if best_thickness_nm is not None:
        rec["stack_pred_continuous"] = continuous_stack_records(
            pred_ids_i,
            best_thickness_nm[i],
            idx_to_token,
            eos=eos,
            special=special,
            active_prob=best_active_prob[i] if best_active_prob is not None else None,
        )

    if do_sim and best_pred_spectra is not None:
        rec.update({
            "mae": float(best_mae[i].item()),
            "rat_target": score_spectra[i].detach().cpu().numpy().tolist(),
            "rat_pred": best_pred_spectra[i].detach().cpu().numpy().tolist(),
        })
        if best_mae_common is not None:
            rec["mae_common"] = float(best_mae_common[i].item())
        if conditioning_spectra is not None:
            rec["rat_conditioning"] = conditioning_spectra[i].detach().cpu().numpy().tolist()

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
    total_mae_common: float,
    total_mae_common_examples: float,
    ddp: bool,
    device: torch.device,
) -> tuple[float, float, float, float, float, float]:
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
    total_mae_common : float
        Local sum of per-example common-grid MAE values.
    total_mae_common_examples : float
        Local number of examples contributing to common-grid MAE.
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
        return total_correct, total_tokens, total_mae, total_mae_examples, total_mae_common, total_mae_common_examples

    totals = torch.tensor(
        [
            total_correct,
            total_tokens,
            total_mae,
            total_mae_examples,
            total_mae_common,
            total_mae_common_examples,
        ],
        device=device,
        dtype=torch.float64,
    )
    torch.distributed.all_reduce(totals)

    return tuple(float(v.item()) for v in totals)


def assemble_mc_grids(
    all_mc_mae: list,
    all_mc_mae_common: list,
    all_mc_ids: list,
    all_mc_pred: list,
    all_mc_traj: list,
    m: int,
    do_sim: bool,
    record_pred_spectra: bool,
    track_step_mae: bool,
    all_mc_thickness: Optional[list] = None,
) -> dict:
    """
    Assemble per-batch MC draw lists into ``[N, m, ...]`` grid tensors.

    Args
    ----
    all_mc_mae : list
        List over batches; each element is a list of ``m`` tensors ``[B]``.
    all_mc_mae_common : list
        List over batches; each element is a list of ``m`` common-grid MAE
        tensors ``[B]``.
    all_mc_ids : list
        List over batches; each element is a list of ``m`` tensors ``[B, S]``.
    all_mc_pred : list
        List over batches; each element is a list of ``m`` tensors ``[B, 3, W]``.
    all_mc_traj : list
        List over batches; each element is a list of ``m`` tensors ``[B, steps]``.
    all_mc_thickness : list, optional
        List over batches; each element is a list of ``m`` tensors ``[B, S]``.
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

    if all_mc_mae_common:
        grids["mae_common_grid"] = _concat_and_stack(all_mc_mae_common).to(torch.float32)

    if do_sim and record_pred_spectra and all_mc_pred:
        grids["pred_spectra_grid"] = _concat_and_stack(all_mc_pred).to(torch.float32)

    if track_step_mae and all_mc_traj:
        grids["mae_traj_grid"] = _concat_and_stack(all_mc_traj).to(torch.float32)

    if all_mc_thickness:
        grids["thickness_nm_grid"] = _concat_and_stack(all_mc_thickness).to(torch.float32)

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
    source_wavelengths: Optional[torch.Tensor] = None,
    common_mae_wavelengths: Optional[torch.Tensor] = None,
    mae_channel_mask: Optional[torch.Tensor] = None,
    score_spectrum: Optional[torch.Tensor] = None,
    record_all_mc: bool = False,
    record_pred_spectra: bool = True,
    rank: int = 0,
    world_size: int = 1,
    gather: bool = True,
    show_progress: bool = False,
    profile_timing: bool = False,
    deduplicate_stacks: bool = False,
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
    source_wavelengths : torch.Tensor, optional
        Native wavelength grid for spectra passed through ``loader``.
    common_mae_wavelengths : torch.Tensor, optional
        Fixed comparison grid used for ``mae_common`` and
        ``mean_mae_common``.
    score_spectrum : torch.Tensor, optional
        Spectrum used for MC ranking and MAE reporting. This allows target
        inference to condition on a physicalized target while ranking against
        the original file target.
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
    show_progress : bool
        Whether to show a tqdm progress bar on rank 0.
    profile_timing : bool
        Whether to collect coarse model/TMM/recording timings.
    deduplicate_stacks : bool
        Whether to simulate unique predicted stacks only before scattering
        spectra back to duplicate MC entries.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:

        - ``"mean_acc"`` (float): mean token accuracy.
        - ``"mean_mae"`` (float or None): mean native/ROI MAE; ``None``
          when ``mode="NO_SIM"``.
        - ``"mean_mae_common"`` (float or None): mean MAE after resampling
          to ``common_mae_wavelengths``; ``None`` when no common grid is
          configured or ``mode="NO_SIM"``.
        - ``"results"`` (list[dict]): per-example records (rank 0 only).

        Optional keys (rank 0 only) when ``record_all_mc=True``:

        - ``"mae_grid"`` — shape ``[N, m]``, native/ROI MAE, float.
        - ``"mae_common_grid"`` — shape ``[N, m]``, common-grid MAE, float
          (only when a common grid is configured).
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

    all_mc_mae, all_mc_mae_common, all_mc_ids, all_mc_pred, all_mc_traj, all_mc_thickness, results = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    total_correct, total_tokens = 0.0, 0.0
    total_mae, total_mae_examples, running_idx = 0.0, 0.0, 0
    total_mae_common, total_mae_common_examples = 0.0, 0.0
    timing_totals = {
        "model_s": 0.0,
        "tmm_s": 0.0,
        "record_s": 0.0,
        "post_s": 0.0,
        "dedup_input": 0.0,
        "dedup_unique": 0.0,
    }

    iterator = tqdm.tqdm(
        loader,
        total=len(loader) if hasattr(loader, "__len__") else None,
        desc="inference",
        disable=(not show_progress) or rank != 0,
    )

    for batch in iterator:
        spectra, stacks, idxs, running_idx = unpack_batch(batch, running_idx, device)
        b = spectra.size(0)
        score_spectra = spectra
        conditioning_spectra = None
        if score_spectrum is not None:
            score_spectrum_d = torch.as_tensor(score_spectrum, dtype=spectra.dtype, device=device)
            if score_spectrum_d.dim() == 2:
                score_spectra = score_spectrum_d.unsqueeze(0).expand(b, -1, -1)
            elif score_spectrum_d.dim() == 3 and score_spectrum_d.size(0) == b:
                score_spectra = score_spectrum_d
            elif score_spectrum_d.dim() == 3 and score_spectrum_d.size(0) == 1:
                score_spectra = score_spectrum_d.expand(b, -1, -1)
            else:
                raise ValueError(
                    "score_spectrum must have shape [3,W], [1,3,W], or [B,3,W], "
                    f"got {tuple(score_spectrum_d.shape)} for batch size {b}"
                )
            conditioning_spectra = spectra

        best, draws, timing = run_mc_batch(
            model, spectra, mc_samples, do_sim, tmm_ctx,
            eos, pad, msk, roi_mask, source_wavelengths, common_mae_wavelengths,
            record_all_mc, record_pred_spectra, device,
            score_spectra=score_spectra,
            mae_channel_mask=mae_channel_mask,
            deduplicate_stacks=deduplicate_stacks,
            profile_timing=profile_timing,
        )
        for key in timing_totals:
            timing_totals[key] += float(timing.get(key, 0.0))

        sync_for_timing(device, profile_timing)
        t0 = time.perf_counter()

        if record_all_mc:
            accumulate_mc_draws(
                draws, mc_samples, do_sim, record_pred_spectra, track_step_mae,
                all_mc_mae, all_mc_mae_common, all_mc_ids, all_mc_pred, all_mc_traj,
                all_mc_thickness,
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
            mae_vec = optollama.evaluation.metrics.masked_mae_roi(
                score_spectra,
                best["pred_spectra"],
                wl_mask=roi_mask,
                channel_mask=mae_channel_mask,
            )
            total_mae += float(mae_vec.sum().item())
            total_mae_examples += float(mae_vec.numel())
            if best["mae_common"] is not None:
                total_mae_common += float(best["mae_common"].sum().item())
                total_mae_common_examples += float(best["mae_common"].numel())

        for i in range(b):
            results.append(build_example_record(
                i, idxs, stacks_aligned, ids_aligned, acc_vec,
                best["mae"], best["mae_common"], best["pred_spectra"], best["step_mae_traj"],
                best["thickness_nm"], best["active_prob"], score_spectra, idx_to_token, eos, pad, msk, do_sim,
                conditioning_spectra=conditioning_spectra,
            ))

        sync_for_timing(device, profile_timing)
        timing_totals["post_s"] += time.perf_counter() - t0

        if show_progress and rank == 0 and profile_timing:
            postfix = {
                "model": f"{timing['model_s']:.2f}s",
                "tmm": f"{timing['tmm_s']:.2f}s",
            }
            if timing.get("dedup_input", 0.0) > 0.0:
                postfix["unique"] = f"{int(timing['dedup_unique'])}/{int(timing['dedup_input'])}"
            iterator.set_postfix(postfix)

    if gather and ddp:
        results = gather_ddp_results(results, world_size, rank)

    (
        total_correct,
        total_tokens,
        total_mae,
        total_mae_examples,
        total_mae_common,
        total_mae_common_examples,
    ) = reduce_ddp_metric_sums(
        total_correct,
        total_tokens,
        total_mae,
        total_mae_examples,
        total_mae_common,
        total_mae_common_examples,
        ddp,
        device,
    )

    out: dict[str, Any] = {
        "mean_acc": total_correct / max(total_tokens, 1.0),
        "mean_mae": (total_mae / max(total_mae_examples, 1.0)) if do_sim else None,
        "mean_mae_common": (
            total_mae_common / max(total_mae_common_examples, 1.0)
            if do_sim and total_mae_common_examples > 0.0
            else None
        ),
    }
    if common_mae_wavelengths is not None:
        out["common_mae_wavelengths"] = torch.as_tensor(common_mae_wavelengths).detach().cpu()
    if profile_timing:
        timing_out = dict(timing_totals)
        if timing_totals["dedup_input"] > 0.0:
            timing_out["dedup_ratio"] = timing_totals["dedup_unique"] / timing_totals["dedup_input"]
        out["timing"] = timing_out
    if not ddp or rank == 0:
        out["results"] = results
        if record_all_mc:
            out.update(assemble_mc_grids(
                all_mc_mae, all_mc_mae_common, all_mc_ids, all_mc_pred, all_mc_traj,
                max(1, mc_samples), do_sim, record_pred_spectra, track_step_mae,
                all_mc_thickness=all_mc_thickness,
            ))

    return out
