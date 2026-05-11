from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from .augmentation import ExtensionPool, count_layer_tokens
from .token import SPECIAL_TOKENS


@dataclass(frozen=True)
class TokenMutationIndex:
    """
    Lookup tables used by target-driven stack perturbation.

    The tables are intentionally Python dictionaries because perturbation is a
    stochastic proposal step; the expensive part remains the batched TMM
    simulation of the generated proposals.
    """

    token_materials: list[Optional[str]]
    token_thicknesses: list[Optional[int]]
    material_thickness_to_id: dict[str, dict[int, int]]
    allowed_token_ids: torch.Tensor
    allowed_token_id_list: list[int]
    allowed_by_thickness: dict[int, list[int]]


def _parse_token(token: str) -> tuple[str, int] | None:
    """
    Parse a layer token of the form ``material_thickness``.
    """
    if token in SPECIAL_TOKENS or "_" not in token:
        return None

    material, thickness = token.rsplit("_", 1)
    try:
        return material, int(thickness)
    except ValueError:
        return None


def build_token_mutation_index(
    tokens: Sequence[str],
    pool: ExtensionPool,
) -> TokenMutationIndex:
    """
    Build lookup tables for thickness, material, and insertion mutations.

    Args
    ----
    tokens : Sequence[str]
        Vocabulary tokens.
    pool : ExtensionPool
        Allowed proposal-token pool.

    Returns
    -------
    TokenMutationIndex
        Lookup data for fast stochastic perturbations.
    """
    token_materials: list[Optional[str]] = [None] * len(tokens)
    token_thicknesses: list[Optional[int]] = [None] * len(tokens)
    material_thickness_to_id: dict[str, dict[int, int]] = {}
    allowed_by_thickness: dict[int, list[int]] = {}

    allowed_set = {int(v) for v in pool.allowed_token_ids.tolist()}
    for token_id, token in enumerate(tokens):
        parsed = _parse_token(token)
        if parsed is None:
            continue

        material, thickness = parsed
        token_materials[token_id] = material
        token_thicknesses[token_id] = thickness
        material_thickness_to_id.setdefault(material, {})[thickness] = token_id

        if token_id in allowed_set:
            allowed_by_thickness.setdefault(thickness, []).append(token_id)

    return TokenMutationIndex(
        token_materials=token_materials,
        token_thicknesses=token_thicknesses,
        material_thickness_to_id=material_thickness_to_id,
        allowed_token_ids=pool.allowed_token_ids.detach().cpu().long(),
        allowed_token_id_list=[int(v) for v in pool.allowed_token_ids.tolist()],
        allowed_by_thickness=allowed_by_thickness,
    )


def _randint(generator: torch.Generator, high: int) -> int:
    return int(torch.randint(high, (1,), generator=generator, device="cpu").item())


def _rand(generator: torch.Generator) -> float:
    return float(torch.rand((), generator=generator, device="cpu").item())


def _sample_from(values: Sequence[int], generator: torch.Generator) -> int:
    return int(values[_randint(generator, len(values))])


def _nearest_token_for_thickness(
    material: str,
    thickness: int,
    index: TokenMutationIndex,
) -> Optional[int]:
    by_thickness = index.material_thickness_to_id.get(material)
    if not by_thickness:
        return None

    if thickness in by_thickness:
        return int(by_thickness[thickness])

    nearest = min(by_thickness.keys(), key=lambda value: abs(value - thickness))
    return int(by_thickness[nearest])


def _active_layers(
    stack: torch.Tensor,
    eos: int,
    pad: int,
    msk: int,
) -> list[int]:
    layers: list[int] = []
    for token_id in stack.detach().cpu().tolist():
        token_id = int(token_id)
        if token_id == eos:
            break
        if token_id in (pad, msk):
            continue
        layers.append(token_id)
    return layers


def _encode_layers(
    layers: Sequence[int],
    output_seq_len: int,
    eos: int,
    pad: int,
    device: torch.device,
) -> torch.Tensor:
    out = torch.full((output_seq_len,), int(pad), dtype=torch.long, device=device)
    n = min(len(layers), output_seq_len - 1)
    if n:
        out[:n] = torch.as_tensor(layers[:n], dtype=torch.long, device=device)
    out[n] = int(eos)
    return out


def _choose_operation(
    layers: Sequence[int],
    max_layers: int,
    insertion_prob: float,
    material_prob: float,
    thickness_prob: float,
    delete_prob: float,
    generator: torch.Generator,
) -> str:
    ops: list[tuple[str, float]] = []
    if len(layers) < max_layers:
        ops.append(("insert", max(0.0, insertion_prob)))
    if layers:
        ops.append(("material", max(0.0, material_prob)))
        ops.append(("thickness", max(0.0, thickness_prob)))
    if len(layers) > 1:
        ops.append(("delete", max(0.0, delete_prob)))

    ops = [(name, weight) for name, weight in ops if weight > 0.0]
    if not ops:
        return "noop"

    total = sum(weight for _, weight in ops)
    draw = _rand(generator) * total
    accum = 0.0
    for name, weight in ops:
        accum += weight
        if draw <= accum:
            return name
    return ops[-1][0]


def _mutate_layers(
    layers: list[int],
    index: TokenMutationIndex,
    max_layers: int,
    thickness_deltas: Sequence[int],
    insertion_prob: float,
    material_prob: float,
    thickness_prob: float,
    delete_prob: float,
    generator: torch.Generator,
) -> list[int]:
    op = _choose_operation(
        layers,
        max_layers=max_layers,
        insertion_prob=insertion_prob,
        material_prob=material_prob,
        thickness_prob=thickness_prob,
        delete_prob=delete_prob,
        generator=generator,
    )

    out = list(layers)
    if op == "insert":
        pos = _randint(generator, len(out) + 1)
        token_id = _sample_from(index.allowed_token_id_list, generator)
        out.insert(pos, token_id)
    elif op == "delete":
        pos = _randint(generator, len(out))
        del out[pos]
    elif op == "material":
        pos = _randint(generator, len(out))
        old_token = out[pos]
        old_material = index.token_materials[old_token]
        thickness = index.token_thicknesses[old_token]

        choices = index.allowed_by_thickness.get(int(thickness)) if thickness is not None else None
        if choices:
            new_token = _sample_from(choices, generator)
            for _ in range(4):
                if index.token_materials[new_token] != old_material:
                    break
                new_token = _sample_from(choices, generator)
        else:
            new_token = _sample_from(index.allowed_token_id_list, generator)
        out[pos] = new_token
    elif op == "thickness":
        pos = _randint(generator, len(out))
        old_token = out[pos]
        material = index.token_materials[old_token]
        thickness = index.token_thicknesses[old_token]
        if material is not None and thickness is not None and thickness_deltas:
            delta = _sample_from(thickness_deltas, generator)
            new_token = _nearest_token_for_thickness(material, int(thickness) + int(delta), index)
            if new_token is not None:
                out[pos] = new_token

    return out[:max_layers]


def perturb_stack_candidates(
    stacks: torch.Tensor,
    index: TokenMutationIndex,
    eos: int,
    pad: int,
    msk: int,
    max_layers: int,
    output_seq_len: int,
    num_perturbations: int,
    generator: torch.Generator,
    edits_per_perturbation: int = 1,
    insertion_prob: float = 0.25,
    material_prob: float = 0.25,
    thickness_prob: float = 0.45,
    delete_prob: float = 0.05,
    thickness_deltas: Sequence[int] = (-50, -40, -30, -20, -10, 10, 20, 30, 40, 50),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate local perturbations of candidate stacks.

    Args
    ----
    stacks : torch.Tensor
        Base candidate stacks, shape ``[N, S]``.
    index : TokenMutationIndex
        Token mutation lookup tables.
    eos, pad, msk : int
        Special token ids.
    max_layers : int
        Maximum number of layer tokens after mutation.
    output_seq_len : int
        Fixed output token sequence length.
    num_perturbations : int
        Number of perturbed variants per input stack.
    generator : torch.Generator
        CPU RNG controlling perturbation reproducibility.
    edits_per_perturbation : int
        Number of stochastic edit operations per perturbed variant.
    insertion_prob, material_prob, thickness_prob, delete_prob : float
        Relative probabilities for mutation operation selection.
    thickness_deltas : Sequence[int]
        Candidate thickness changes in nm.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(variants, source_indices)`` where ``variants`` has shape
        ``[N * num_perturbations, output_seq_len]`` and ``source_indices``
        maps each variant back to its input row.
    """
    if num_perturbations <= 0:
        raise ValueError("num_perturbations must be positive.")
    if output_seq_len < max_layers + 1:
        raise ValueError("output_seq_len must be at least max_layers + 1.")

    device = stacks.device
    variants: list[torch.Tensor] = []
    source_indices: list[int] = []

    for source_idx in range(stacks.size(0)):
        base_layers = _active_layers(stacks[source_idx], eos=eos, pad=pad, msk=msk)
        for _ in range(num_perturbations):
            layers = list(base_layers)
            for _ in range(max(1, int(edits_per_perturbation))):
                layers = _mutate_layers(
                    layers,
                    index=index,
                    max_layers=max_layers,
                    thickness_deltas=thickness_deltas,
                    insertion_prob=insertion_prob,
                    material_prob=material_prob,
                    thickness_prob=thickness_prob,
                    delete_prob=delete_prob,
                    generator=generator,
                )
            variants.append(_encode_layers(layers, output_seq_len, eos=eos, pad=pad, device=device))
            source_indices.append(source_idx)

    return torch.stack(variants, dim=0), torch.as_tensor(source_indices, dtype=torch.long, device=device)


def generate_filter_targets(
    n_targets: int,
    wavelengths: torch.Tensor,
    roi_min: float,
    roi_max: float,
    generator: torch.Generator,
    min_band_width: float = 80.0,
    max_band_width: float = 320.0,
    edge_width: float = 15.0,
) -> torch.Tensor:
    """
    Generate simple target spectra for self-improving augmentation.

    The generated targets are idealized optical filters with A=0,
    T=1 inside one or two random passbands, and R=1 elsewhere. Smooth
    sigmoid edges avoid hard discontinuities.

    Returns
    -------
    torch.Tensor
        Target spectra of shape ``[n_targets, 3, W]``.
    """
    if n_targets <= 0:
        raise ValueError("n_targets must be positive.")

    wl = wavelengths.to(torch.float32).cpu()
    targets: list[torch.Tensor] = []
    span = float(roi_max - roi_min)
    if span <= min_band_width:
        raise ValueError("ROI width must be larger than min_band_width.")

    for _ in range(n_targets):
        n_bands = 1 + _randint(generator, 2)
        transmission = torch.zeros_like(wl)
        for _ in range(n_bands):
            width = min_band_width + _rand(generator) * max(0.0, max_band_width - min_band_width)
            low = float(roi_min) + _rand(generator) * max(1.0, span - width)
            high = min(float(roi_max), low + width)
            left = torch.sigmoid((wl - low) / float(edge_width))
            right = torch.sigmoid((high - wl) / float(edge_width))
            transmission = torch.maximum(transmission, left * right)

        transmission = transmission.clamp(0.0, 1.0)
        reflectance = 1.0 - transmission
        absorptance = torch.zeros_like(transmission)
        targets.append(torch.stack([reflectance, absorptance, transmission], dim=0))

    return torch.stack(targets, dim=0).to(torch.float32)


def select_top_improved_per_target(
    target_indices: torch.Tensor,
    base_mae: torch.Tensor,
    final_mae: torch.Tensor,
    max_per_target: int,
    min_improvement: float,
    source_lengths: Optional[torch.Tensor] = None,
    final_lengths: Optional[torch.Tensor] = None,
    min_layer_gain: int = 0,
    length_reward: float = 0.0,
    max_mae_regression: float = 0.0,
) -> torch.Tensor:
    """
    Select candidates, keeping at most ``max_per_target`` per target.

    ``length_reward`` is expressed in MAE-equivalent units per added layer.
    It can make longer candidates easier to keep, while ``max_mae_regression``
    limits how much target fit may degrade.
    """
    improvement = base_mae - final_mae
    if source_lengths is None or final_lengths is None:
        layer_gain = torch.zeros_like(improvement)
    else:
        layer_gain = (final_lengths.to(improvement.device) - source_lengths.to(improvement.device)).to(improvement.dtype)

    adjusted_improvement = improvement + float(length_reward) * layer_gain
    valid = (
        (adjusted_improvement >= float(min_improvement))
        & (layer_gain >= int(min_layer_gain))
        & (improvement >= -float(max_mae_regression))
    )
    selection_score = final_mae - float(length_reward) * layer_gain
    selected: list[int] = []

    for target_idx in torch.unique(target_indices[valid]).tolist():
        rows = (target_indices == int(target_idx)).nonzero(as_tuple=False).squeeze(1)
        rows = rows[valid[rows]]
        if rows.numel() == 0:
            continue
        order = selection_score[rows].argsort()
        selected.extend(rows[order[:max(1, int(max_per_target))]].tolist())

    if not selected:
        return torch.empty((0,), dtype=torch.long, device=target_indices.device)

    return torch.as_tensor(selected, dtype=torch.long, device=target_indices.device)
