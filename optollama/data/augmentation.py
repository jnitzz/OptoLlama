from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch

import optollama.evaluation

from .token import (
    SPECIAL_TOKENS,
    make_material_groups,
    make_material_token_ids,
    material_name,
    token_ids_of,
)


@dataclass(frozen=True)
class ExtensionPool:
    """
    Sampling pool for stack-extension tokens.

    Attributes
    ----------
    allowed_token_ids : torch.Tensor
        1-D long tensor of token ids that may be appended.
    token_material_codes : torch.Tensor
        1-D long tensor mapping every token id in the vocabulary to a compact
        material code. Special tokens map to ``-1``.
    metal_token_mask : torch.Tensor
        Boolean tensor marking tokens in the repo's predefined ``metals``
        group.
    absorber_token_mask : torch.Tensor
        Boolean tensor marking tokens in the predefined ``metals`` or
        ``semiconductors`` groups. This is used for soft penalties, not hard
        constraints.
    """

    allowed_token_ids: torch.Tensor
    token_material_codes: torch.Tensor
    metal_token_mask: torch.Tensor
    absorber_token_mask: torch.Tensor


def count_layer_tokens(
    stacks: torch.Tensor,
    eos: int,
    pad: int,
    msk: int,
) -> torch.Tensor:
    """
    Count material/layer tokens before the first EOS in each sequence.

    Args
    ----
    stacks : torch.Tensor
        Token sequences of shape ``[B, S]``.
    eos : int
        EOS token id.
    pad : int
        PAD token id.
    msk : int
        MASK token id.

    Returns
    -------
    torch.Tensor
        Long tensor of shape ``[B]`` with the number of layer tokens per
        sample.
    """
    is_eos = stacks == eos
    before_first_eos = is_eos.cumsum(dim=1) == 0
    valid = before_first_eos & (stacks != pad) & (stacks != msk)

    return valid.sum(dim=1).to(torch.long)


def reencode_stacks_for_output(
    stacks: torch.Tensor,
    output_seq_len: int,
    eos: int,
    pad: int,
    msk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Re-encode token stacks into a longer fixed-length output format.

    The returned tensor contains layer tokens first, followed by a single EOS,
    followed by PAD tokens.

    Args
    ----
    stacks : torch.Tensor
        Input token sequences of shape ``[B, S_in]``.
    output_seq_len : int
        Desired output sequence length.
    eos : int
        EOS token id.
    pad : int
        PAD token id.
    msk : int
        MASK token id.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A 2-tuple ``(encoded, lengths)`` where ``encoded`` has shape
        ``[B, output_seq_len]`` and ``lengths`` contains the number of layer
        tokens per sample.

    Raises
    ------
    ValueError
        If ``output_seq_len`` is too short to hold the existing layers plus
        EOS.
    """
    device = stacks.device
    lengths = count_layer_tokens(stacks, eos=eos, pad=pad, msk=msk)
    max_layers = int(lengths.max().item()) if lengths.numel() else 0

    if max_layers + 1 > output_seq_len:
        raise ValueError(
            f"output_seq_len={output_seq_len} is too short for existing stack "
            f"length {max_layers} plus EOS."
        )

    out = torch.full((stacks.size(0), output_seq_len), pad, dtype=stacks.dtype, device=device)
    is_eos = stacks == eos
    before_first_eos = is_eos.cumsum(dim=1) == 0
    valid = before_first_eos & (stacks != pad) & (stacks != msk)

    for i in range(stacks.size(0)):
        layer_tokens = stacks[i][valid[i]]
        length_i = int(lengths[i].item())
        if length_i:
            out[i, :length_i] = layer_tokens[:length_i]
        out[i, length_i] = eos

    return out, lengths


def build_extension_pool(
    tokens: list[str],
    token_to_idx: dict[str, int],
    allowed_groups: Optional[Sequence[str]] = None,
    exclude_materials: Optional[Sequence[str]] = None,
    exclude_tokens: Optional[Sequence[str]] = None,
) -> ExtensionPool:
    """
    Build a token pool for suffix generation.

    When ``allowed_groups`` is ``None`` or empty, all non-special tokens are
    eligible for extension. Optional exclusion lists can then be used as a
    small blacklist for known-bad materials or tokens.

    Args
    ----
    tokens : list[str]
        Full vocabulary list.
    token_to_idx : dict[str, int]
        Mapping from token string to token id.
    allowed_groups : Sequence[str], optional
        Material groups from :func:`make_material_groups` that may contribute
        extension tokens. If omitted, all non-special tokens are allowed.
    exclude_materials : Sequence[str], optional
        Base material names to exclude from the pool.
    exclude_tokens : Sequence[str], optional
        Exact token strings to exclude from the pool.

    Returns
    -------
    ExtensionPool
        Pool describing which token ids may be sampled and how token ids map to
        base materials.
    """
    groups = make_material_groups(tokens, token_to_idx)
    material_ids = make_material_token_ids(token_to_idx)

    allowed_parts = []
    if allowed_groups:
        for name in allowed_groups:
            if name not in groups:
                raise ValueError(f"Unknown extension material group: {name!r}")
            allowed_parts.append(groups[name])
    else:
        allowed_parts.append(
            torch.tensor(
                [token_to_idx[t] for t in tokens if t not in SPECIAL_TOKENS],
                dtype=torch.long,
            )
        )

    allowed_token_ids = torch.unique(torch.cat(allowed_parts, dim=0))

    exclude_items: list[Any] = []
    if exclude_materials:
        exclude_items.extend(list(exclude_materials))
    if exclude_tokens:
        exclude_items.extend(list(exclude_tokens))
    if exclude_items:
        exclude_ids = token_ids_of(exclude_items, token_to_idx, material_ids)
        if exclude_ids.numel():
            allowed_token_ids = allowed_token_ids[~torch.isin(allowed_token_ids, exclude_ids)]

    if allowed_token_ids.numel() == 0:
        raise ValueError("No extension tokens remain after applying group and exclusion filters.")

    token_material_codes = torch.full((len(tokens),), -1, dtype=torch.long)
    material_code_map: dict[str, int] = {}
    next_code = 0
    for token, token_id in token_to_idx.items():
        if token in SPECIAL_TOKENS:
            continue

        base = material_name(token)
        if base not in material_code_map:
            material_code_map[base] = next_code
            next_code += 1
        token_material_codes[token_id] = material_code_map[base]

    metal_token_mask = torch.zeros((len(tokens),), dtype=torch.bool)
    absorber_token_mask = torch.zeros((len(tokens),), dtype=torch.bool)
    if groups["metals"].numel():
        metal_token_mask[groups["metals"]] = True
        absorber_token_mask[groups["metals"]] = True
    if groups["semiconductors"].numel():
        absorber_token_mask[groups["semiconductors"]] = True

    return ExtensionPool(
        allowed_token_ids=allowed_token_ids,
        token_material_codes=token_material_codes,
        metal_token_mask=metal_token_mask,
        absorber_token_mask=absorber_token_mask,
    )


def count_group_tokens(
    stacks: torch.Tensor,
    token_mask: torch.Tensor,
    eos: int,
    pad: int,
    msk: int,
) -> torch.Tensor:
    """
    Count tokens from a boolean token mask before the first EOS.

    Args
    ----
    stacks : torch.Tensor
        Token sequences of shape ``[B, S]``.
    token_mask : torch.Tensor
        Boolean mask over the vocabulary, shape ``[V]``.
    eos : int
        EOS token id.
    pad : int
        PAD token id.
    msk : int
        MASK token id.

    Returns
    -------
    torch.Tensor
        Float tensor of shape ``[B]`` with counts of masked tokens.
    """
    is_eos = stacks == eos
    before_first_eos = is_eos.cumsum(dim=1) == 0
    valid = before_first_eos & (stacks != pad) & (stacks != msk)

    return (valid & token_mask[stacks]).sum(dim=1).float()


def sample_extension_tokens(
    previous_tokens: torch.Tensor,
    pool: ExtensionPool,
    num_candidates: int,
    generator: Optional[torch.Generator] = None,
    avoid_same_material: bool = True,
    material_resample_rounds: int = 4,
) -> torch.Tensor:
    """
    Sample candidate extension tokens for a batch of active stacks.

    Args
    ----
    previous_tokens : torch.Tensor
        Last active layer token per sample, shape ``[B]``.
    pool : ExtensionPool
        Sampling pool built by :func:`build_extension_pool`.
    num_candidates : int
        Number of candidate next tokens to draw per sample.
    generator : torch.Generator, optional
        RNG generator controlling reproducibility.
    avoid_same_material : bool
        If ``True``, resample proposals that use the same base material as the
        previous token.
    material_resample_rounds : int
        Maximum number of resampling rounds for same-material collisions.

    Returns
    -------
    torch.Tensor
        Tensor of sampled token ids with shape ``[B, num_candidates]``.
    """
    device = previous_tokens.device
    allowed_token_ids = pool.allowed_token_ids.to(device)
    token_material_codes = pool.token_material_codes.to(device)

    sample_idx = torch.randint(
        low=0,
        high=allowed_token_ids.numel(),
        size=(previous_tokens.size(0), num_candidates),
        generator=generator,
        device=device,
    )
    sampled = allowed_token_ids[sample_idx]

    if not avoid_same_material:
        return sampled

    prev_codes = token_material_codes[previous_tokens]
    cand_codes = token_material_codes[sampled]
    same = cand_codes == prev_codes.unsqueeze(1)

    for _ in range(material_resample_rounds):
        if not same.any():
            break
        repl_idx = torch.randint(
            low=0,
            high=allowed_token_ids.numel(),
            size=(int(same.sum().item()),),
            generator=generator,
            device=device,
        )
        sampled[same] = allowed_token_ids[repl_idx]
        cand_codes = token_material_codes[sampled]
        same = cand_codes == prev_codes.unsqueeze(1)

    return sampled


@torch.no_grad()
def augment_stack_batch(
    spectra: torch.Tensor,
    stacks: torch.Tensor,
    tmm_ctx: "optollama.evaluation.simulation.TMMContext",
    pool: ExtensionPool,
    eos: int,
    pad: int,
    msk: int,
    max_layers: int,
    output_seq_len: int,
    min_layers: int = 20,
    samples_per_input: int = 1,
    num_candidates: int = 8,
    proposal_rounds: int = 4,
    min_delta_mae: float = 0.002,
    lookahead_candidates: int = 2,
    lookahead_topk: Optional[int] = 2,
    lookahead_weight: float = 0.35,
    min_lookahead_delta: float = 0.0005,
    metal_penalty: float = 0.01,
    absorber_penalty: float = 0.005,
    absorber_load_penalty: float = 0.01,
    roi_mask: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """
    Extend a batch of stacks with low-loss suffix layers.

    The augmentation proceeds one layer at a time. For each active sample the
    function samples ``num_candidates`` proposals, simulates them in a single
    batched TMM call, and ranks them by:

    - immediate spectral change relative to the current spectrum,
    - a one-step lookahead score measuring how extensible the candidate
      remains,
    - penalties for metal / absorber-heavy growth.

    Samples that fail all proposal rounds stop early.

    Args
    ----
    spectra : torch.Tensor
        Source spectra of shape ``[B, 3, W]``.
    stacks : torch.Tensor
        Source token sequences of shape ``[B, S]``.
    tmm_ctx : TMMContext
        Batched TMM context used for suffix validation and final spectra.
    pool : ExtensionPool
        Sampling pool for extension tokens.
    eos : int
        EOS token id.
    pad : int
        PAD token id.
    msk : int
        MASK token id.
    max_layers : int
        Maximum number of material layers in the augmented result.
    output_seq_len : int
        Length of the saved token sequence. Must be at least
        ``max_layers + 1`` to hold EOS.
    min_layers : int
        Minimum target length for a generated sample.
    samples_per_input : int
        Number of augmented variants to generate per input example.
    num_candidates : int
        Number of next-token proposals per active sample and round.
    proposal_rounds : int
        Number of resampling rounds when no candidate clears the MAE threshold.
    min_delta_mae : float
        Minimum MAE change required for an appended token to be accepted.
    lookahead_candidates : int
        Number of one-step lookahead proposals sampled per first-step
        candidate.
    lookahead_topk : int, optional
        Number of first-step candidates per sample that receive lookahead
        evaluation. If ``None`` or ``<= 0``, all candidates are evaluated.
    lookahead_weight : float
        Weight of the lookahead delta in the combined proposal score.
    min_lookahead_delta : float
        Minimum best lookahead delta required when the sample still needs at
        least one more appended token after the current step.
    metal_penalty : float
        Score penalty applied when the new token is in the predefined
        ``metals`` group.
    absorber_penalty : float
        Score penalty applied when the new token is in the predefined
        ``metals`` or ``semiconductors`` groups.
    absorber_load_penalty : float
        Additional score penalty proportional to the current absorber fraction
        in the stack.
    roi_mask : torch.Tensor, optional
        Optional wavelength mask restricting the MAE-based acceptance test.
    generator : torch.Generator, optional
        RNG generator controlling reproducibility.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
        ``(aug_spectra, aug_stacks, stats)`` with augmented spectra
        ``[B*samples_per_input, 3, W]`` and token sequences
        ``[B*samples_per_input, output_seq_len]``.
    """
    if output_seq_len < max_layers + 1:
        raise ValueError(
            f"output_seq_len={output_seq_len} must be at least max_layers + 1 "
            f"(={max_layers + 1}) to store EOS."
        )

    if samples_per_input <= 0:
        raise ValueError("samples_per_input must be positive.")

    device = spectra.device
    metal_token_mask = pool.metal_token_mask.to(device)
    absorber_token_mask = pool.absorber_token_mask.to(device)

    if samples_per_input > 1:
        spectra = spectra.repeat_interleave(samples_per_input, dim=0)
        stacks = stacks.repeat_interleave(samples_per_input, dim=0)

    current_stacks, source_lengths = reencode_stacks_for_output(
        stacks,
        output_seq_len=output_seq_len,
        eos=eos,
        pad=pad,
        msk=msk,
    )
    current_spectra = spectra.clone()
    current_lengths = source_lengths.clone().to(device)

    low = torch.maximum(current_lengths, torch.full_like(current_lengths, int(min_layers)))
    high = torch.full_like(current_lengths, int(max_layers))
    if (low > high).any():
        raise ValueError(
            f"Found source stacks longer than max_layers={max_layers}. "
            "Increase max_layers or filter the input data."
        )

    span = (high - low + 1).to(torch.float32)
    u = torch.rand(low.shape, generator=generator, device=device)
    target_lengths = low + torch.floor(u * span).to(torch.long)

    stopped_early = torch.zeros_like(current_lengths, dtype=torch.bool)

    while True:
        active = (~stopped_early) & (current_lengths < target_lengths)
        if not active.any():
            break

        active_rows = active.nonzero(as_tuple=False).squeeze(1)
        active_stacks = current_stacks[active_rows]
        active_spectra = current_spectra[active_rows]
        active_lengths = current_lengths[active_rows]
        active_targets = target_lengths[active_rows]

        chosen_stacks = active_stacks.clone()
        chosen_spectra = active_spectra.clone()
        accepted = torch.zeros(active_rows.size(0), dtype=torch.bool, device=device)

        last_positions = active_lengths - 1
        previous_tokens = active_stacks[torch.arange(active_stacks.size(0), device=device), last_positions]

        for _ in range(proposal_rounds):
            remaining = (~accepted).nonzero(as_tuple=False).squeeze(1)
            if remaining.numel() == 0:
                break

            rem_stacks = active_stacks[remaining]
            rem_spectra = active_spectra[remaining]
            rem_lengths = active_lengths[remaining]
            rem_targets = active_targets[remaining]
            rem_prev = previous_tokens[remaining]

            proposals = sample_extension_tokens(
                rem_prev,
                pool=pool,
                num_candidates=num_candidates,
                generator=generator,
                avoid_same_material=True,
            )

            candidate_stacks = rem_stacks.repeat_interleave(num_candidates, dim=0)
            candidate_rows = torch.arange(candidate_stacks.size(0), device=device)
            insert_positions = rem_lengths.repeat_interleave(num_candidates)
            candidate_stacks[candidate_rows, insert_positions] = proposals.reshape(-1)

            eos_positions = insert_positions + 1
            candidate_stacks[candidate_rows, eos_positions] = eos

            candidate_spectra = optollama.evaluation.simulation.simulate_token_sequence(
                candidate_stacks,
                tmm_ctx,
                eos=eos,
                pad=pad,
                msk=msk,
            )

            _, _, channels, width = candidate_spectra.view(rem_stacks.size(0), num_candidates, *candidate_spectra.shape[1:]).shape
            candidate_stacks = candidate_stacks.view(rem_stacks.size(0), num_candidates, output_seq_len)
            candidate_spectra = candidate_spectra.view(rem_stacks.size(0), num_candidates, channels, width)

            ref = rem_spectra.unsqueeze(1).expand(-1, num_candidates, -1, -1).reshape(-1, channels, width)
            delta = optollama.evaluation.metrics.masked_mae_roi(
                ref,
                candidate_spectra.reshape(-1, channels, width),
                wl_mask=roi_mask,
            ).view(rem_stacks.size(0), num_candidates)

            need_future = (rem_lengths + 1) < rem_targets
            best_lookahead = torch.zeros_like(delta)
            lookahead_mask = torch.zeros_like(delta, dtype=torch.bool)
            if lookahead_candidates > 0 and need_future.any():
                topk = num_candidates if lookahead_topk is None or lookahead_topk <= 0 else min(int(lookahead_topk), num_candidates)
                topk_idx = delta[need_future].topk(k=topk, dim=1).indices  # [Nf, topk]

                future_candidate_stacks = candidate_stacks[need_future]      # [Nf, C, S]
                future_candidate_spectra = candidate_spectra[need_future]    # [Nf, C, 3, W]
                future_candidate_props = proposals[need_future]              # [Nf, C]
                future_candidate_lengths = rem_lengths[need_future]          # [Nf]

                gather_stack = topk_idx.unsqueeze(-1).expand(-1, -1, output_seq_len)
                future_stacks = future_candidate_stacks.gather(dim=1, index=gather_stack).reshape(-1, output_seq_len)

                gather_spec = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, channels, width)
                future_spectra = future_candidate_spectra.gather(dim=1, index=gather_spec).reshape(-1, channels, width)

                future_prev = future_candidate_props.gather(dim=1, index=topk_idx).reshape(-1)
                future_lengths = (
                    (future_candidate_lengths + 1)
                    .unsqueeze(1)
                    .expand(-1, topk)
                    .reshape(-1)
                )

                lookahead_tokens = sample_extension_tokens(
                    future_prev,
                    pool=pool,
                    num_candidates=lookahead_candidates,
                    generator=generator,
                    avoid_same_material=True,
                )

                lookahead_stacks = future_stacks.repeat_interleave(lookahead_candidates, dim=0)
                lookahead_rows = torch.arange(lookahead_stacks.size(0), device=device)
                insert_positions = future_lengths.repeat_interleave(lookahead_candidates)
                lookahead_stacks[lookahead_rows, insert_positions] = lookahead_tokens.reshape(-1)
                lookahead_stacks[lookahead_rows, insert_positions + 1] = eos

                lookahead_spectra = optollama.evaluation.simulation.simulate_token_sequence(
                    lookahead_stacks,
                    tmm_ctx,
                    eos=eos,
                    pad=pad,
                    msk=msk,
                )

                future_ref = future_spectra.repeat_interleave(lookahead_candidates, dim=0)
                lookahead_delta = optollama.evaluation.metrics.masked_mae_roi(
                    future_ref,
                    lookahead_spectra,
                    wl_mask=roi_mask,
                ).view(future_stacks.size(0), lookahead_candidates)

                best_future = lookahead_delta.max(dim=1).values.view(-1, topk)
                best_lookahead[need_future].scatter_(1, topk_idx, best_future)
                lookahead_mask[need_future].scatter_(
                    1,
                    topk_idx,
                    torch.ones_like(best_future, dtype=torch.bool),
                )

            absorber_counts = count_group_tokens(
                rem_stacks,
                absorber_token_mask,
                eos=eos,
                pad=pad,
                msk=msk,
            ).to(device)
            absorber_fraction = absorber_counts / rem_lengths.float().clamp_min(1.0)

            proposal_is_metal = metal_token_mask[proposals].float()
            proposal_is_absorber = absorber_token_mask[proposals].float()
            penalties = (
                metal_penalty * proposal_is_metal
                + absorber_penalty * proposal_is_absorber
                + absorber_load_penalty * absorber_fraction.unsqueeze(1)
            )

            valid = delta >= float(min_delta_mae)
            if lookahead_candidates > 0:
                valid = valid & (
                    ~need_future.unsqueeze(1)
                    | (lookahead_mask & (best_lookahead >= float(min_lookahead_delta)))
                )

            combined_score = delta + (lookahead_weight * best_lookahead) - penalties
            scores = torch.where(valid, combined_score, torch.full_like(combined_score, -float("inf")))
            best_score, best_idx = scores.max(dim=1)
            has_valid = valid.any(dim=1)

            if has_valid.any():
                accept_rows = remaining[has_valid]
                chosen_stacks[accept_rows] = candidate_stacks[has_valid, best_idx[has_valid]]
                chosen_spectra[accept_rows] = candidate_spectra[has_valid, best_idx[has_valid]]
                accepted[accept_rows] = True

        if accepted.any():
            write_rows = active_rows[accepted]
            current_stacks[write_rows] = chosen_stacks[accepted]
            current_spectra[write_rows] = chosen_spectra[accepted]
            current_lengths[write_rows] = current_lengths[write_rows] + 1

        if (~accepted).any():
            stopped_early[active_rows[~accepted]] = True

    stats = {
        "source_lengths": source_lengths.detach().cpu(),
        "target_lengths": target_lengths.detach().cpu(),
        "final_lengths": current_lengths.detach().cpu(),
        "stopped_early": stopped_early.detach().cpu(),
    }

    return current_spectra, current_stacks, stats
