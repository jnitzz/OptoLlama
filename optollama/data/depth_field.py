from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from optollama.data.token import SPECIAL_TOKENS, layer_token_parts

VOID_MATERIAL = "<VOID>"


@dataclass(frozen=True)
class DepthTokenOption:
    """One tokenized thickness option for a material."""

    thickness_nm: float
    token_id: int
    token: str


@dataclass(frozen=True)
class DepthFieldVocab:
    """Material vocabulary used by the depth-field representation."""

    material_names: tuple[str, ...]
    material_to_id: dict[str, int]
    token_options: dict[str, tuple[DepthTokenOption, ...]]
    void_id: int
    mask_id: int

    @property
    def num_clean_classes(self) -> int:
        """Return the number of denoising classes, including void."""
        return len(self.material_names)

    def material_name(self, material_id: int) -> str:
        """Return a material name for a clean depth-field id."""
        return self.material_names[int(material_id)]

    def to_dict(self) -> dict:
        """Return a JSON/checkpoint friendly vocabulary description."""
        return {
            "material_names": list(self.material_names),
            "void_id": int(self.void_id),
            "mask_id": int(self.mask_id),
        }


def depth_bins_for(max_thickness_nm: float, dz_nm: float) -> int:
    """Return the number of fixed-width bins needed for a depth range."""
    if dz_nm <= 0:
        raise ValueError(f"dz_nm must be positive, got {dz_nm}")
    if max_thickness_nm <= 0:
        raise ValueError(f"max_thickness_nm must be positive, got {max_thickness_nm}")
    return int(round(float(max_thickness_nm) / float(dz_nm)))


def build_depth_field_vocab(tokens: Iterable[str], token_to_idx: dict[str, int]) -> DepthFieldVocab:
    """
    Build a material-level depth-field vocabulary from layer tokens.

    The clean denoising classes are all material names found in layer tokens
    plus ``<VOID>``. ``mask_id`` is intentionally outside the clean classes and
    is used only as the corrupted diffusion input.
    """
    material_names: list[str] = []
    material_seen: set[str] = set()
    options: dict[str, list[DepthTokenOption]] = {}

    for token in tokens:
        if token in SPECIAL_TOKENS:
            continue
        parts = layer_token_parts(token)
        if parts is None:
            continue
        material, thickness_nm = parts
        if material not in material_seen:
            material_seen.add(material)
            material_names.append(material)
        options.setdefault(material, []).append(
            DepthTokenOption(thickness_nm=float(thickness_nm), token_id=int(token_to_idx[token]), token=str(token))
        )

    if not material_names:
        raise ValueError("Depth-field vocabulary found no material layer tokens.")

    for material, items in options.items():
        items.sort(key=lambda item: (item.thickness_nm, item.token))

    material_names.append(VOID_MATERIAL)
    material_to_id = {name: idx for idx, name in enumerate(material_names)}
    void_id = material_to_id[VOID_MATERIAL]

    return DepthFieldVocab(
        material_names=tuple(material_names),
        material_to_id=material_to_id,
        token_options={key: tuple(value) for key, value in options.items()},
        void_id=void_id,
        mask_id=len(material_names),
    )


def rasterize_stack_to_depth_field(
    stacks: torch.Tensor,
    idx_to_token: dict[int, str],
    vocab: DepthFieldVocab,
    *,
    dz_nm: float = 10.0,
    max_thickness_nm: float = 10_000.0,
    eos_idx: int,
    pad_idx: int,
    msk_idx: int,
) -> torch.Tensor:
    """
    Convert token stacks to fixed-depth material fields.

    Each 10 nm depth bin stores the material class occupying that slice. Layers
    beyond ``max_thickness_nm`` are clipped, and remaining bins are ``<VOID>``.
    """
    if stacks.dim() == 1:
        stacks = stacks.unsqueeze(0)
    if stacks.dim() != 2:
        raise ValueError(f"stacks must have shape [B,S] or [S], got {tuple(stacks.shape)}")

    bins = depth_bins_for(max_thickness_nm=max_thickness_nm, dz_nm=dz_nm)
    fields = torch.full((int(stacks.size(0)), bins), int(vocab.void_id), dtype=torch.long, device=stacks.device)

    stacks_cpu = stacks.detach().cpu()
    for batch_idx in range(stacks_cpu.size(0)):
        cursor = 0
        for token_id in stacks_cpu[batch_idx].tolist():
            token_id = int(token_id)
            if token_id == eos_idx:
                break
            if token_id in (pad_idx, msk_idx):
                continue
            token = idx_to_token.get(token_id)
            if token is None:
                continue
            parts = layer_token_parts(token)
            if parts is None:
                continue

            material, thickness_nm = parts
            if thickness_nm <= 0:
                continue
            material_id = vocab.material_to_id.get(material)
            if material_id is None:
                continue

            layer_bins = int(round(float(thickness_nm) / float(dz_nm)))
            if layer_bins <= 0:
                layer_bins = 1
            stop = min(cursor + layer_bins, bins)
            if stop > cursor:
                fields[batch_idx, cursor:stop] = int(material_id)
                cursor = stop
            if cursor >= bins:
                break

    return fields


def depth_field_active_bins(fields: torch.Tensor, void_id: int) -> torch.Tensor:
    """Return non-void bin counts per sample."""
    return (fields != int(void_id)).sum(dim=-1)


def token_stack_total_thickness_nm(
    stacks: torch.Tensor,
    idx_to_token: dict[int, str],
    *,
    eos_idx: int,
    pad_idx: int,
    msk_idx: int,
) -> torch.Tensor:
    """Return total tokenized layer thickness per stack before EOS."""
    if stacks.dim() == 1:
        stacks = stacks.unsqueeze(0)
    if stacks.dim() != 2:
        raise ValueError(f"stacks must have shape [B,S] or [S], got {tuple(stacks.shape)}")

    totals: list[float] = []
    stacks_cpu = stacks.detach().cpu()
    for batch_idx in range(stacks_cpu.size(0)):
        total = 0.0
        for token_id in stacks_cpu[batch_idx].tolist():
            token_id = int(token_id)
            if token_id == eos_idx:
                break
            if token_id in (pad_idx, msk_idx):
                continue
            token = idx_to_token.get(token_id)
            if token is None:
                continue
            parts = layer_token_parts(token)
            if parts is not None:
                total += float(parts[1])
        totals.append(total)

    return torch.tensor(totals, dtype=torch.float32, device=stacks.device)


def depth_field_runs(field: torch.Tensor, vocab: DepthFieldVocab, *, dz_nm: float = 10.0) -> list[dict[str, float | str]]:
    """
    Convert one depth field to material runs, merging equal neighbors.

    Void bins are skipped. Equal material runs separated only by void are merged,
    because void has zero optical effect after decoding.
    """
    if field.dim() != 1:
        raise ValueError(f"field must have shape [D], got {tuple(field.shape)}")

    runs: list[tuple[int, int]] = []
    for value in field.detach().cpu().tolist():
        material_id = int(value)
        if material_id == int(vocab.void_id) or material_id < 0 or material_id >= vocab.num_clean_classes:
            continue
        if runs and runs[-1][0] == material_id:
            runs[-1] = (material_id, runs[-1][1] + 1)
        else:
            runs.append((material_id, 1))

    return [
        {"material": vocab.material_name(material_id), "thickness_nm": float(bin_count) * float(dz_nm)}
        for material_id, bin_count in runs
    ]


def depth_field_material_token_ids(vocab: DepthFieldVocab) -> dict[str, int]:
    """
    Return one representative layer token id per material.

    The representative token supplies only the material identity when TMM is
    called with a continuous thickness override.
    """
    return {
        material: int(options[0].token_id)
        for material, options in vocab.token_options.items()
        if options
    }


def _nearest_option(options: tuple[DepthTokenOption, ...], thickness_nm: float) -> DepthTokenOption:
    return min(options, key=lambda item: (abs(item.thickness_nm - float(thickness_nm)), item.thickness_nm))


def _tokenize_material_run(vocab: DepthFieldVocab, material: str, thickness_nm: float, *, min_run_nm: float) -> list[int]:
    options = vocab.token_options.get(material)
    if not options or thickness_nm < min_run_nm:
        return []

    max_option = options[-1]
    remaining = float(thickness_nm)
    token_ids: list[int] = []

    while remaining > max_option.thickness_nm + 0.5 * min_run_nm:
        token_ids.append(int(max_option.token_id))
        remaining -= max_option.thickness_nm

    if remaining >= min_run_nm:
        token_ids.append(int(_nearest_option(options, remaining).token_id))

    return token_ids


def decode_depth_field_to_tokens(
    fields: torch.Tensor,
    vocab: DepthFieldVocab,
    *,
    output_seq_len: int,
    dz_nm: float = 10.0,
    eos_idx: int,
    pad_idx: int,
    min_run_nm: float | None = None,
) -> torch.Tensor:
    """
    Convert depth fields back to token-id stacks.

    Adjacent equal materials are merged to one optical layer, then long runs are
    split into the largest available tokenized thickness chunks for that
    material. A trailing EOS is written when there is room.
    """
    if fields.dim() == 1:
        fields = fields.unsqueeze(0)
    if fields.dim() != 2:
        raise ValueError(f"fields must have shape [B,D] or [D], got {tuple(fields.shape)}")
    if output_seq_len <= 0:
        raise ValueError(f"output_seq_len must be positive, got {output_seq_len}")

    min_nm = float(dz_nm if min_run_nm is None else min_run_nm)
    out = torch.full((int(fields.size(0)), int(output_seq_len)), int(pad_idx), dtype=torch.long, device=fields.device)

    for batch_idx in range(fields.size(0)):
        write_idx = 0
        for run in depth_field_runs(fields[batch_idx], vocab, dz_nm=dz_nm):
            material = str(run["material"])
            thickness_nm = float(run["thickness_nm"])
            for token_id in _tokenize_material_run(vocab, material, thickness_nm, min_run_nm=min_nm):
                if write_idx >= output_seq_len:
                    break
                out[batch_idx, write_idx] = int(token_id)
                write_idx += 1
            if write_idx >= output_seq_len:
                break

        if write_idx < output_seq_len:
            out[batch_idx, write_idx] = int(eos_idx)

    return out


def token_stack_strings(
    ids: torch.Tensor,
    idx_to_token: dict[int, str],
    *,
    eos_idx: int,
    pad_idx: int,
    include_eos: bool = False,
) -> list[str]:
    """Return readable token strings from one decoded stack."""
    if ids.dim() != 1:
        raise ValueError(f"ids must have shape [S], got {tuple(ids.shape)}")

    tokens: list[str] = []
    for token_id in ids.detach().cpu().tolist():
        token_id = int(token_id)
        if token_id == pad_idx:
            continue
        if token_id == eos_idx:
            if include_eos:
                tokens.append(idx_to_token[token_id])
            break
        tokens.append(idx_to_token.get(token_id, str(token_id)))
    return tokens
