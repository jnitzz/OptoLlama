import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from optollama.evaluation.metrics import masked_mae_roi

from .augmentation import count_layer_tokens
from .self_improvement import TokenMutationIndex


APPROX_REFRACTIVE_INDEX = {
    "Al2O3": 1.76,
    "AlN": 2.10,
    "EVA": 1.48,
    "HfO2": 2.00,
    "MgF2": 1.38,
    "MgO": 1.73,
    "Si3N4": 2.00,
    "SiO2": 1.46,
    "Ta2O5": 2.10,
    "TiO2": 2.40,
    "ZnO": 2.00,
    "ZnS": 2.30,
    "ZnSe": 2.50,
}


@dataclass(frozen=True)
class LongStackLibrary:
    """
    Token lookup data for physically structured long-stack generation.
    """

    index: TokenMutationIndex
    materials: list[str]
    high_index_materials: list[str]
    low_index_materials: list[str]
    refractive_index: dict[str, float]


def build_long_stack_library(index: TokenMutationIndex) -> LongStackLibrary:
    """
    Build material groups from the token mutation index.

    Args
    ----
    index : TokenMutationIndex
        Token lookup data created from the allowed generation pool.

    Returns
    -------
    LongStackLibrary
        Material groups and approximate refractive indices.
    """
    materials = sorted(
        {
            index.token_materials[int(token_id)]
            for token_id in index.allowed_token_id_list
            if index.token_materials[int(token_id)] is not None
        }
    )
    if len(materials) < 2:
        raise ValueError("Long-stack generation requires at least two allowed materials.")

    refractive_index = {material: APPROX_REFRACTIVE_INDEX.get(material, 1.80) for material in materials}
    ordered = sorted(materials, key=lambda material: refractive_index[material])

    low_index_materials = [m for m in materials if refractive_index[m] <= 1.80]
    high_index_materials = [m for m in materials if refractive_index[m] >= 1.95]

    if not low_index_materials:
        low_index_materials = ordered[: max(1, len(ordered) // 2)]
    if not high_index_materials:
        high_index_materials = ordered[max(1, len(ordered) // 2) :]
    if not high_index_materials or not low_index_materials:
        raise ValueError("Could not split allowed materials into high/low index groups.")

    return LongStackLibrary(
        index=index,
        materials=materials,
        high_index_materials=high_index_materials,
        low_index_materials=low_index_materials,
        refractive_index=refractive_index,
    )


def _rand(generator: torch.Generator) -> float:
    return float(torch.rand((), generator=generator, device="cpu").item())


def _randint(generator: torch.Generator, high: int) -> int:
    return int(torch.randint(high, (1,), generator=generator, device="cpu").item())


def _sample(values: Sequence, generator: torch.Generator):
    return values[_randint(generator, len(values))]


def _randint_inclusive(generator: torch.Generator, low: int, high: int) -> int:
    if high < low:
        raise ValueError(f"Invalid integer range: {low}..{high}")
    return low + _randint(generator, high - low + 1)


def _nearest_token(index: TokenMutationIndex, material: str, thickness: float) -> int:
    by_thickness = index.material_thickness_to_id.get(material)
    if not by_thickness:
        raise ValueError(f"No token thicknesses available for material {material!r}.")

    nearest = min(by_thickness.keys(), key=lambda value: abs(value - thickness))
    return int(by_thickness[nearest])


def _random_token_for_material(index: TokenMutationIndex, material: str, generator: torch.Generator) -> int:
    by_thickness = index.material_thickness_to_id.get(material)
    if not by_thickness:
        raise ValueError(f"No tokens available for material {material!r}.")
    thicknesses = sorted(by_thickness)
    return int(by_thickness[_sample(thicknesses, generator)])


def _quarter_wave_thickness(center_wavelength_nm: float, material: str, library: LongStackLibrary) -> float:
    return center_wavelength_nm / (4.0 * library.refractive_index[material])


def _half_wave_thickness(center_wavelength_nm: float, material: str, library: LongStackLibrary) -> float:
    return center_wavelength_nm / (2.0 * library.refractive_index[material])


def _jitter_thickness(thickness: float, jitter_fraction: float, generator: torch.Generator) -> float:
    if jitter_fraction <= 0.0:
        return thickness
    return thickness * (1.0 + (2.0 * _rand(generator) - 1.0) * jitter_fraction)


def _sample_high_low_pair(library: LongStackLibrary, generator: torch.Generator) -> tuple[str, str]:
    high = _sample(library.high_index_materials, generator)
    low_choices = [material for material in library.low_index_materials if material != high]
    if not low_choices:
        low_choices = [material for material in library.materials if material != high]
    low = _sample(low_choices, generator)
    return high, low


def encode_layer_tokens(
    layers: Sequence[int],
    output_seq_len: int,
    eos: int,
    pad: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Encode layer-token ids as ``layers | EOS | PAD``.
    """
    if output_seq_len < len(layers) + 1:
        raise ValueError("output_seq_len is too short for generated stack plus EOS.")

    out = torch.full((output_seq_len,), int(pad), dtype=torch.long, device=device)
    if layers:
        out[: len(layers)] = torch.as_tensor(layers, dtype=torch.long, device=device)
    out[len(layers)] = int(eos)
    return out


def generate_random_dielectric_layers(
    library: LongStackLibrary,
    min_layers: int,
    max_layers: int,
    generator: torch.Generator,
) -> list[int]:
    """
    Generate a random dielectric long stack with no immediate material repeats.
    """
    length = _randint_inclusive(generator, min_layers, max_layers)
    layers: list[int] = []
    previous_material: Optional[str] = None

    for _ in range(length):
        choices = [material for material in library.materials if material != previous_material]
        material = _sample(choices or library.materials, generator)
        layers.append(_random_token_for_material(library.index, material, generator))
        previous_material = material

    return layers


def generate_random_layers(
    library: LongStackLibrary,
    min_layers: int,
    max_layers: int,
    generator: torch.Generator,
) -> list[int]:
    """
    Generate a completely random stack from the allowed token pool.
    """
    length = _randint_inclusive(generator, min_layers, max_layers)
    return [_sample(library.index.allowed_token_id_list, generator) for _ in range(length)]


def generate_dbr_layers(
    library: LongStackLibrary,
    min_layers: int,
    max_layers: int,
    center_min_nm: float,
    center_max_nm: float,
    jitter_fraction: float,
    generator: torch.Generator,
) -> list[int]:
    """
    Generate an alternating quarter-wave DBR-like stack.
    """
    length = _randint_inclusive(generator, max(2, min_layers), max_layers)
    high, low = _sample_high_low_pair(library, generator)
    center = center_min_nm + _rand(generator) * (center_max_nm - center_min_nm)
    start_high = _rand(generator) < 0.5
    layers: list[int] = []

    for idx in range(length):
        material = high if ((idx % 2 == 0) == start_high) else low
        thickness = _quarter_wave_thickness(center, material, library)
        thickness = _jitter_thickness(thickness, jitter_fraction, generator)
        layers.append(_nearest_token(library.index, material, thickness))

    return layers


def generate_chirped_dbr_layers(
    library: LongStackLibrary,
    min_layers: int,
    max_layers: int,
    center_min_nm: float,
    center_max_nm: float,
    jitter_fraction: float,
    generator: torch.Generator,
) -> list[int]:
    """
    Generate an alternating DBR stack with a wavelength chirp across depth.
    """
    length = _randint_inclusive(generator, max(2, min_layers), max_layers)
    high, low = _sample_high_low_pair(library, generator)
    center_a = center_min_nm + _rand(generator) * (center_max_nm - center_min_nm)
    center_b = center_min_nm + _rand(generator) * (center_max_nm - center_min_nm)
    if abs(center_a - center_b) < 100.0:
        center_b = min(center_max_nm, max(center_min_nm, center_a + (100.0 if center_a < center_max_nm - 100.0 else -100.0)))

    start_high = _rand(generator) < 0.5
    layers: list[int] = []

    for idx in range(length):
        frac = 0.0 if length <= 1 else idx / float(length - 1)
        center = center_a + frac * (center_b - center_a)
        material = high if ((idx % 2 == 0) == start_high) else low
        thickness = _quarter_wave_thickness(center, material, library)
        thickness = _jitter_thickness(thickness, jitter_fraction, generator)
        layers.append(_nearest_token(library.index, material, thickness))

    return layers


def generate_cavity_layers(
    library: LongStackLibrary,
    min_layers: int,
    max_layers: int,
    center_min_nm: float,
    center_max_nm: float,
    jitter_fraction: float,
    generator: torch.Generator,
) -> list[int]:
    """
    Generate a symmetric DBR-cavity-DBR stack.
    """
    pair_min = max(1, math.ceil((min_layers - 1) / 4))
    pair_max = max(1, math.floor((max_layers - 1) / 4))
    if pair_max < pair_min:
        return generate_dbr_layers(
            library,
            min_layers=min_layers,
            max_layers=max_layers,
            center_min_nm=center_min_nm,
            center_max_nm=center_max_nm,
            jitter_fraction=jitter_fraction,
            generator=generator,
        )

    mirror_pairs = _randint_inclusive(generator, pair_min, pair_max)
    high, low = _sample_high_low_pair(library, generator)
    center = center_min_nm + _rand(generator) * (center_max_nm - center_min_nm)
    cavity_material = low if _rand(generator) < 0.7 else high

    high_token = _nearest_token(
        library.index,
        high,
        _jitter_thickness(_quarter_wave_thickness(center, high, library), jitter_fraction, generator),
    )
    low_token = _nearest_token(
        library.index,
        low,
        _jitter_thickness(_quarter_wave_thickness(center, low, library), jitter_fraction, generator),
    )
    cavity_token = _nearest_token(
        library.index,
        cavity_material,
        _jitter_thickness(_half_wave_thickness(center, cavity_material, library), jitter_fraction, generator),
    )

    left: list[int] = []
    for _ in range(mirror_pairs):
        left.extend([high_token, low_token])

    right: list[int] = []
    for _ in range(mirror_pairs):
        right.extend([low_token, high_token])

    return left + [cavity_token] + right


def generate_long_stack_layers(
    family: str,
    library: LongStackLibrary,
    min_layers: int,
    max_layers: int,
    center_min_nm: float,
    center_max_nm: float,
    jitter_fraction: float,
    generator: torch.Generator,
) -> list[int]:
    """
    Generate one tokenized layer list from a named family.
    """
    if family == "random":
        return generate_random_layers(library, min_layers, max_layers, generator)
    if family == "random_dielectric":
        return generate_random_dielectric_layers(library, min_layers, max_layers, generator)
    if family == "dbr":
        return generate_dbr_layers(library, min_layers, max_layers, center_min_nm, center_max_nm, jitter_fraction, generator)
    if family == "chirped_dbr":
        return generate_chirped_dbr_layers(library, min_layers, max_layers, center_min_nm, center_max_nm, jitter_fraction, generator)
    if family == "cavity":
        return generate_cavity_layers(library, min_layers, max_layers, center_min_nm, center_max_nm, jitter_fraction, generator)
    raise ValueError(f"Unknown long-stack family: {family!r}")


def generate_long_stack_batch(
    batch_size: int,
    families: Sequence[str],
    library: LongStackLibrary,
    min_layers: int,
    max_layers: int,
    output_seq_len: int,
    eos: int,
    pad: int,
    center_min_nm: float,
    center_max_nm: float,
    jitter_fraction: float,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """
    Generate a batch of encoded long stacks.
    """
    stacks: list[torch.Tensor] = []
    lengths: list[int] = []
    selected_families: list[str] = []

    for _ in range(batch_size):
        family = _sample(families, generator)
        layers = generate_long_stack_layers(
            family,
            library=library,
            min_layers=min_layers,
            max_layers=max_layers,
            center_min_nm=center_min_nm,
            center_max_nm=center_max_nm,
            jitter_fraction=jitter_fraction,
            generator=generator,
        )
        stacks.append(encode_layer_tokens(layers, output_seq_len, eos=eos, pad=pad, device=device))
        lengths.append(len(layers))
        selected_families.append(family)

    return torch.stack(stacks, dim=0), torch.as_tensor(lengths, dtype=torch.long), selected_families


def build_prefix_variants(
    stacks: torch.Tensor,
    final_lengths: torch.Tensor,
    prefix_lengths: Sequence[int],
    eos: int,
    pad: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build prefix stacks and map each prefix back to its source full stack.
    """
    variants: list[torch.Tensor] = []
    source_indices: list[int] = []
    output_seq_len = stacks.size(1)

    for row in range(stacks.size(0)):
        final_length = int(final_lengths[row].item())
        for prefix_length in prefix_lengths:
            prefix_length = int(prefix_length)
            if prefix_length <= 0 or prefix_length >= final_length:
                continue
            prefix = torch.full((output_seq_len,), int(pad), dtype=torch.long, device=stacks.device)
            prefix[:prefix_length] = stacks[row, :prefix_length]
            prefix[prefix_length] = int(eos)
            variants.append(prefix)
            source_indices.append(row)

    if not variants:
        return (
            torch.empty((0, output_seq_len), dtype=torch.long, device=stacks.device),
            torch.empty((0,), dtype=torch.long, device=stacks.device),
        )

    return torch.stack(variants, dim=0), torch.as_tensor(source_indices, dtype=torch.long, device=stacks.device)


def prefix_filter_mask(
    stacks: torch.Tensor,
    full_spectra: torch.Tensor,
    prefix_lengths: Sequence[int],
    min_prefix_mae: float,
    tmm_ctx,
    eos: int,
    pad: int,
    msk: int,
    roi_mask: Optional[torch.Tensor],
    eval_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Keep stacks whose short prefixes cannot closely reproduce the full spectrum.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(keep_mask, best_prefix_mae)``.
    """
    if min_prefix_mae <= 0.0 or not prefix_lengths:
        return (
            torch.ones((stacks.size(0),), dtype=torch.bool, device=stacks.device),
            torch.full((stacks.size(0),), float("inf"), dtype=torch.float32, device=stacks.device),
        )

    final_lengths = count_layer_tokens(stacks, eos=eos, pad=pad, msk=msk)
    prefixes, source_idx = build_prefix_variants(stacks, final_lengths, prefix_lengths, eos=eos, pad=pad)
    if prefixes.numel() == 0:
        return (
            torch.ones((stacks.size(0),), dtype=torch.bool, device=stacks.device),
            torch.full((stacks.size(0),), float("inf"), dtype=torch.float32, device=stacks.device),
        )

    best_prefix_mae = torch.full((stacks.size(0),), float("inf"), dtype=torch.float32, device=stacks.device)
    for start in range(0, prefixes.size(0), eval_batch_size):
        end = min(start + eval_batch_size, prefixes.size(0))
        pred = tmm_ctx.tmm(prefixes[start:end], tmm_ctx.wl, tmm_ctx.theta, eos=eos, pad=pad, msk=msk)
        mae = masked_mae_roi(
            full_spectra[source_idx[start:end]],
            pred,
            wl_mask=roi_mask,
        )
        best_prefix_mae.scatter_reduce_(
            0,
            source_idx[start:end],
            mae,
            reduce="amin",
            include_self=True,
        )

    return best_prefix_mae >= float(min_prefix_mae), best_prefix_mae
