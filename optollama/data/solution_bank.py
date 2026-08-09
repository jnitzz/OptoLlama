from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import safetensors.torch
import torch
from safetensors import safe_open

SOLUTION_BANK_SCHEMA_VERSION = 1
QUALITY_SILVER = 1
QUALITY_GOLD = 2

SOLUTION_BANK_TENSORS = (
    "anchor_spectra",
    "fields",
    "pred_spectra",
    "anchor_indices",
    "candidate_indices",
    "level_mae",
    "derivative_mae",
    "score",
    "quality_tier",
    "topology_hash",
    "run_count",
    "active_bins",
)


def spectrum_error_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    wavelengths: torch.Tensor,
    channels: Sequence[int] = (0, 2),
    roi_min: float | None = None,
    roi_max: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-sample spectrum level and adjacent-difference MAE."""
    if predicted.shape != target.shape or predicted.dim() != 3:
        raise ValueError(f"predicted and target must share shape [B,3,W], got {tuple(predicted.shape)} and {tuple(target.shape)}")
    if not channels:
        raise ValueError("channels must not be empty")

    wl = torch.as_tensor(wavelengths, dtype=torch.float32, device=predicted.device)
    if wl.numel() != predicted.size(-1):
        raise ValueError(f"wavelength grid has {wl.numel()} entries, expected {predicted.size(-1)}")
    mask = torch.ones_like(wl, dtype=torch.bool)
    if roi_min is not None:
        mask &= wl >= float(roi_min)
    if roi_max is not None:
        mask &= wl <= float(roi_max)
    if not bool(mask.any()):
        raise ValueError("spectrum error ROI contains no wavelengths")

    channel_index = torch.as_tensor(tuple(int(value) for value in channels), dtype=torch.long, device=predicted.device)
    pred_selected = predicted.index_select(1, channel_index)
    target_selected = target.index_select(1, channel_index)
    level_mae = (pred_selected[..., mask] - target_selected[..., mask]).abs().mean(dim=(1, 2))

    pair_mask = mask[:-1] & mask[1:]
    if not bool(pair_mask.any()):
        derivative_mae = torch.zeros_like(level_mae)
    else:
        pred_delta = pred_selected[..., 1:] - pred_selected[..., :-1]
        target_delta = target_selected[..., 1:] - target_selected[..., :-1]
        derivative_mae = (pred_delta[..., pair_mask] - target_delta[..., pair_mask]).abs().mean(dim=(1, 2))
    return level_mae, derivative_mae


def field_topology_statistics(
    fields: torch.Tensor,
    *,
    void_id: int,
    dz_nm: float,
    thickness_quantum_nm: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return stable topology hashes, material-run counts, and active-bin counts."""
    if fields.dim() == 1:
        fields = fields.unsqueeze(0)
    if fields.dim() != 2:
        raise ValueError(f"fields must have shape [B,D] or [D], got {tuple(fields.shape)}")
    if thickness_quantum_nm <= 0.0:
        raise ValueError("thickness_quantum_nm must be positive")

    hashes: list[int] = []
    run_counts: list[int] = []
    active_counts: list[int] = []
    for field in fields.detach().cpu():
        active = [int(value) for value in field.tolist() if int(value) != int(void_id)]
        runs: list[tuple[int, int]] = []
        for material_id in active:
            if runs and runs[-1][0] == material_id:
                runs[-1] = (material_id, runs[-1][1] + 1)
            else:
                runs.append((material_id, 1))
        topology = [
            (material_id, max(1, int(round(bin_count * float(dz_nm) / float(thickness_quantum_nm)))))
            for material_id, bin_count in runs
        ]
        digest = hashlib.blake2b(json.dumps(topology, separators=(",", ":")).encode("ascii"), digest_size=8).digest()
        hashes.append(int.from_bytes(digest, byteorder="little", signed=True))
        run_counts.append(len(runs))
        active_counts.append(len(active))

    device = fields.device
    return (
        torch.tensor(hashes, dtype=torch.int64, device=device),
        torch.tensor(run_counts, dtype=torch.int32, device=device),
        torch.tensor(active_counts, dtype=torch.int32, device=device),
    )


class SolutionBankShardWriter:
    """Incrementally write versioned solution-bank safetensor shards."""

    def __init__(
        self,
        out_dir: str | Path,
        *,
        rank: int,
        shard_size: int,
        metadata: dict[str, Any],
    ) -> None:
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rank = int(rank)
        self.shard_size = int(shard_size)
        self.metadata = {
            "schema_version": str(SOLUTION_BANK_SCHEMA_VERSION),
            **{
                str(key): json.dumps(value, sort_keys=True) if not isinstance(value, str) else value
                for key, value in metadata.items()
            },
        }
        self.parts: dict[str, list[torch.Tensor]] = {name: [] for name in SOLUTION_BANK_TENSORS}
        self.buffered = 0
        self.shard_index = 0
        self.paths: list[Path] = []
        self.samples_written = 0

    def append(self, tensors: dict[str, torch.Tensor]) -> None:
        """Append one accepted-candidate batch and flush at the shard limit."""
        missing = [name for name in SOLUTION_BANK_TENSORS if name not in tensors]
        if missing:
            raise KeyError(f"solution-bank append is missing tensors: {missing}")
        count = int(tensors["fields"].size(0))
        if count == 0:
            return
        for name in SOLUTION_BANK_TENSORS:
            tensor = tensors[name]
            if int(tensor.size(0)) != count:
                raise ValueError(f"solution-bank tensor {name!r} has {tensor.size(0)} rows, expected {count}")
            self.parts[name].append(tensor.detach().cpu().contiguous())
        self.buffered += count
        if self.buffered >= self.shard_size:
            self.flush()

    def flush(self) -> Path | None:
        """Write all buffered candidates and return the new shard path."""
        if self.buffered == 0:
            return None
        tensors = {name: torch.cat(parts, dim=0).contiguous() for name, parts in self.parts.items()}
        path = self.out_dir / f"solution-bank-rank{self.rank:05d}-shard{self.shard_index:05d}.safetensors"
        safetensors.torch.save_file(tensors, str(path), metadata=self.metadata)
        self.paths.append(path)
        self.samples_written += int(tensors["fields"].size(0))
        self.shard_index += 1
        self.parts = {name: [] for name in SOLUTION_BANK_TENSORS}
        self.buffered = 0
        return path

    def close(self) -> list[Path]:
        """Flush remaining candidates and return every written shard path."""
        self.flush()
        return list(self.paths)


def collect_solution_bank_paths(paths: Sequence[str | Path] | str | Path) -> list[Path]:
    """Expand solution-bank files/directories into validated shard paths."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    expanded: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            expanded.extend(sorted(path.glob("solution-bank-rank*-shard*.safetensors")))
        elif path.is_file():
            expanded.append(path)
        else:
            raise FileNotFoundError(f"Solution-bank path does not exist: {path}")
    unique = sorted(set(item.resolve() for item in expanded))
    if not unique:
        raise FileNotFoundError("No solution-bank safetensor shards were found.")
    return unique


def _metadata_for_shard(path: Path) -> dict[str, str]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        keys = set(handle.keys())
    missing = set(SOLUTION_BANK_TENSORS) - keys
    if missing:
        raise KeyError(f"Solution-bank shard {path} is missing tensors: {sorted(missing)}")
    version = int(metadata.get("schema_version", 0))
    if version != SOLUTION_BANK_SCHEMA_VERSION:
        raise ValueError(f"Unsupported solution-bank schema {version} in {path}; expected {SOLUTION_BANK_SCHEMA_VERSION}.")
    return metadata


class DepthFieldSolutionBankReplay:
    """Load accepted fields and sample replay rows uniformly by anchor/topology."""

    def __init__(
        self,
        paths: Sequence[str | Path] | str | Path,
        *,
        replay_fraction: float,
        gold_fraction: float,
        seed: int,
        expected_spectrum_shape: Sequence[int],
        expected_depth_bins: int,
        expected_material_names: Sequence[str],
        expected_dz_nm: float,
    ) -> None:
        if not 0.0 <= replay_fraction <= 1.0:
            raise ValueError(f"replay_fraction must be in [0,1], got {replay_fraction}")
        if not 0.0 <= gold_fraction <= 1.0:
            raise ValueError(f"gold_fraction must be in [0,1], got {gold_fraction}")
        self.paths = collect_solution_bank_paths(paths)
        self.replay_fraction = float(replay_fraction)
        self.gold_fraction = float(gold_fraction)
        self.seed = int(seed)

        loaded: dict[str, list[torch.Tensor]] = {name: [] for name in SOLUTION_BANK_TENSORS}
        for path in self.paths:
            metadata = _metadata_for_shard(path)
            material_names = json.loads(metadata.get("material_names", "[]"))
            dz_nm = float(metadata.get("dz_nm", "nan"))
            if list(material_names) != list(expected_material_names):
                raise ValueError(f"Solution-bank material order in {path} does not match the training vocabulary.")
            if not math.isclose(dz_nm, float(expected_dz_nm), rel_tol=0.0, abs_tol=1.0e-9):
                raise ValueError(f"Solution-bank dz_nm={dz_nm:g} in {path}, expected {expected_dz_nm:g}.")
            data = safetensors.torch.load_file(str(path), device="cpu")
            for name in SOLUTION_BANK_TENSORS:
                loaded[name].append(data[name])

        self.tensors = {name: torch.cat(parts, dim=0) for name, parts in loaded.items()}
        self.tensors["anchor_spectra"] = self.tensors["anchor_spectra"].to(torch.float32)
        self.tensors["fields"] = self.tensors["fields"].long()
        if tuple(self.tensors["anchor_spectra"].shape[1:]) != tuple(int(v) for v in expected_spectrum_shape):
            raise ValueError(
                "Solution-bank spectrum shape "
                f"{tuple(self.tensors['anchor_spectra'].shape[1:])} does not match {tuple(expected_spectrum_shape)}."
            )
        if int(self.tensors["fields"].size(1)) != int(expected_depth_bins):
            raise ValueError(f"Solution-bank depth has {self.tensors['fields'].size(1)} bins, expected {expected_depth_bins}.")
        self._groups = self._build_groups()
        if not self._groups:
            raise ValueError("Solution bank contains no Gold or Silver candidates.")

    def _build_groups(self) -> dict[int, dict[int, dict[int, list[int]]]]:
        groups: dict[int, dict[int, dict[int, list[int]]]] = {}
        tiers = self.tensors["quality_tier"].tolist()
        anchors = self.tensors["anchor_indices"].tolist()
        topologies = self.tensors["topology_hash"].tolist()
        for row, (tier, anchor, topology) in enumerate(zip(tiers, anchors, topologies)):
            tier = int(tier)
            if tier not in (QUALITY_GOLD, QUALITY_SILVER):
                continue
            groups.setdefault(tier, {}).setdefault(int(anchor), {}).setdefault(int(topology), []).append(row)
        return groups

    @property
    def samples(self) -> int:
        """Return the number of loaded alternatives."""
        return int(self.tensors["fields"].size(0))

    def summary(self) -> dict[str, Any]:
        """Return JSON-friendly bank provenance and composition statistics."""
        tiers = self.tensors["quality_tier"]
        return {
            "schema_version": SOLUTION_BANK_SCHEMA_VERSION,
            "paths": [str(path) for path in self.paths],
            "samples": self.samples,
            "gold_samples": int((tiers == QUALITY_GOLD).sum().item()),
            "silver_samples": int((tiers == QUALITY_SILVER).sum().item()),
            "anchors": int(torch.unique(self.tensors["anchor_indices"]).numel()),
            "topologies": int(torch.unique(self.tensors["topology_hash"]).numel()),
            "replay_fraction": self.replay_fraction,
            "gold_fraction": self.gold_fraction,
        }

    @staticmethod
    def _randint(generator: torch.Generator, high: int) -> int:
        return int(torch.randint(high, (1,), generator=generator).item())

    def _choose_tier(self, generator: torch.Generator) -> int:
        available = set(self._groups)
        if QUALITY_GOLD in available and QUALITY_SILVER in available:
            return QUALITY_GOLD if float(torch.rand((), generator=generator)) < self.gold_fraction else QUALITY_SILVER
        return QUALITY_GOLD if QUALITY_GOLD in available else QUALITY_SILVER

    def sample_indices(self, count: int, *, generator: torch.Generator) -> torch.Tensor:
        """Sample rows uniformly by tier, anchor, and topology group."""
        rows: list[int] = []
        for _ in range(int(count)):
            tier = self._choose_tier(generator)
            anchors = list(self._groups[tier])
            anchor = anchors[self._randint(generator, len(anchors))]
            topology_groups = self._groups[tier][anchor]
            topologies = list(topology_groups)
            topology = topologies[self._randint(generator, len(topologies))]
            candidates = topology_groups[topology]
            rows.append(candidates[self._randint(generator, len(candidates))])
        return torch.tensor(rows, dtype=torch.long)

    def mix_batch(
        self,
        spectra: torch.Tensor,
        fields: torch.Tensor,
        *,
        epoch: int,
        batch_index: int,
        rank: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Replace a deterministic fraction of a CPU batch with accepted alternatives."""
        if spectra.size(0) != fields.size(0):
            raise ValueError("spectra and fields batch sizes differ")
        count = min(int(spectra.size(0)), int(round(float(spectra.size(0)) * self.replay_fraction)))
        if count <= 0:
            return spectra, fields, 0

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + int(epoch) * 1_000_003 + int(batch_index) * 9_973 + int(rank) * 389)
        positions = torch.randperm(int(spectra.size(0)), generator=generator)[:count]
        rows = self.sample_indices(count, generator=generator)
        mixed_spectra = spectra.clone()
        mixed_fields = fields.clone()
        mixed_spectra[positions] = self.tensors["anchor_spectra"][rows].to(dtype=mixed_spectra.dtype)
        mixed_fields[positions] = self.tensors["fields"][rows].to(dtype=mixed_fields.dtype)
        return mixed_spectra, mixed_fields, count
