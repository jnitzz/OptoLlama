from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from optollama.data.token import SPECIAL_TOKENS, layer_token_parts

CHANNEL_TO_INDEX = {"R": 0, "A": 1, "T": 2}


def _normalize_target_spectrum(values: np.ndarray) -> np.ndarray:
    """Normalize two- or three-channel target values to ``[3,W]`` RAT order."""
    spectra = np.asarray(values, dtype=np.float32)
    if spectra.ndim != 2:
        raise ValueError(f"Target spectrum must be two-dimensional, got {spectra.shape}.")
    if spectra.shape[0] not in {2, 3} and spectra.shape[1] in {2, 3}:
        spectra = spectra.T
    if spectra.shape[0] == 2:
        reflectance, transmittance = spectra
        absorption = np.clip(1.0 - reflectance - transmittance, 0.0, 1.0)
        spectra = np.stack((reflectance, absorption, transmittance), axis=0)
    if spectra.shape[0] != 3:
        raise ValueError(f"Target spectrum must contain R/T or R/A/T channels, got {spectra.shape}.")
    if not np.isfinite(spectra).all():
        raise ValueError("Target spectrum contains non-finite values.")
    return np.clip(spectra, 0.0, 1.0)


def load_open_layer_target(
    path: str | Path,
    fallback_wavelengths_nm: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a legacy target or a coordinate-bearing CSV/JSON target.

    Coordinate-bearing CSV files use columns ``wavelength_nm,R,T`` or
    ``wavelength_nm,R,A,T``. JSON files use ``wavelengths_nm`` and ``spectra``.
    Legacy headerless two/three-row CSVs use ``fallback_wavelengths_nm``.
    """
    target_path = Path(path)
    if not target_path.is_file():
        raise FileNotFoundError(f"Target spectrum does not exist: {target_path}")

    wavelengths: np.ndarray | None = None
    if target_path.suffix.lower() == ".json":
        with target_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or "spectra" not in payload:
            raise ValueError(f"{target_path} must contain a JSON object with a spectra field.")
        spectra = _normalize_target_spectrum(np.asarray(payload["spectra"]))
        if payload.get("wavelengths_nm") is not None:
            wavelengths = np.asarray(payload["wavelengths_nm"], dtype=np.float64).reshape(-1)
    else:
        with target_path.open("r", encoding="utf-8-sig") as handle:
            first_line = handle.readline()
        has_header = any(character.isalpha() for character in first_line)
        if has_header:
            table: Any = np.genfromtxt(target_path, delimiter=",", names=True, dtype=np.float64)
            names = tuple(table.dtype.names or ())
            lowered = {name.lower(): name for name in names}
            wavelength_key = next(
                (lowered[key] for key in ("wavelength_nm", "nm", "wavelength", "lambda", "wl") if key in lowered),
                None,
            )
            if wavelength_key is None or "r" not in lowered or "t" not in lowered:
                raise ValueError(f"{target_path} needs wavelength_nm, R, and T columns.")
            wavelengths = np.asarray(table[wavelength_key], dtype=np.float64).reshape(-1)
            reflectance = np.asarray(table[lowered["r"]], dtype=np.float32).reshape(-1)
            transmittance = np.asarray(table[lowered["t"]], dtype=np.float32).reshape(-1)
            if "a" in lowered:
                absorption = np.asarray(table[lowered["a"]], dtype=np.float32).reshape(-1)
                spectra = _normalize_target_spectrum(np.stack((reflectance, absorption, transmittance), axis=0))
            else:
                spectra = _normalize_target_spectrum(np.stack((reflectance, transmittance), axis=0))
        else:
            spectra = _normalize_target_spectrum(np.loadtxt(target_path, delimiter=",", dtype=np.float32))

    if wavelengths is None:
        if fallback_wavelengths_nm is None:
            raise ValueError(f"{target_path} has no wavelength coordinates and no fallback grid was provided.")
        wavelengths = fallback_wavelengths_nm.detach().cpu().to(dtype=torch.float64).numpy().reshape(-1)
    if len(wavelengths) != spectra.shape[1]:
        raise ValueError(f"Target has {spectra.shape[1]} spectral points but {len(wavelengths)} wavelength coordinates.")
    if not np.isfinite(wavelengths).all() or np.any(wavelengths <= 0):
        raise ValueError("Target wavelengths must be finite and positive.")
    order = np.argsort(wavelengths)
    wavelengths = wavelengths[order]
    if np.any(np.diff(wavelengths) <= 0):
        raise ValueError("Target wavelengths must be unique.")
    spectra = spectra[:, order]
    return torch.from_numpy(spectra).to(dtype=torch.float32), torch.from_numpy(wavelengths.copy()).to(dtype=torch.float32)


@dataclass(frozen=True)
class MaterialCatalog:
    """Native optical-constant curves used to build query-local material banks."""

    names: tuple[str, ...]
    wavelengths_nm: tuple[np.ndarray, ...]
    n_values: tuple[np.ndarray, ...]
    k_values: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        """Validate catalog arrays and material names."""
        count = len(self.names)
        if count == 0:
            raise ValueError("MaterialCatalog requires at least one material.")
        if len(set(self.names)) != count:
            raise ValueError("MaterialCatalog material names must be unique.")
        if not (len(self.wavelengths_nm) == len(self.n_values) == len(self.k_values) == count):
            raise ValueError("MaterialCatalog arrays must have one entry per material.")
        for name, wavelengths, n_values, k_values in zip(
            self.names,
            self.wavelengths_nm,
            self.n_values,
            self.k_values,
            strict=True,
        ):
            if wavelengths.ndim != 1 or n_values.ndim != 1 or k_values.ndim != 1:
                raise ValueError(f"Optical constants for {name!r} must be one-dimensional.")
            if not (len(wavelengths) == len(n_values) == len(k_values)) or len(wavelengths) < 2:
                raise ValueError(f"Optical constants for {name!r} have inconsistent or insufficient samples.")
            if not np.all(np.diff(wavelengths) > 0):
                raise ValueError(f"Wavelengths for {name!r} must be strictly increasing.")
            if not (np.isfinite(wavelengths).all() and np.isfinite(n_values).all() and np.isfinite(k_values).all()):
                raise ValueError(f"Optical constants for {name!r} contain non-finite values.")

    @property
    def name_to_index(self) -> dict[str, int]:
        """Return the stable catalog name-to-index mapping."""
        return {name: idx for idx, name in enumerate(self.names)}

    def interpolate(
        self,
        wavelengths_nm: torch.Tensor,
        material_indices: torch.Tensor | None = None,
        *,
        require_coverage: bool = True,
        coverage_tolerance_nm: float = 0.0,
    ) -> torch.Tensor:
        """Interpolate selected material curves onto a one-dimensional wavelength query."""
        query = wavelengths_nm.detach().to(device="cpu", dtype=torch.float64).numpy()
        if query.ndim != 1 or query.size == 0:
            raise ValueError(f"wavelengths_nm must be non-empty and one-dimensional, got {query.shape}.")
        if not np.isfinite(query).all() or np.any(query <= 0):
            raise ValueError("wavelengths_nm must contain finite positive values.")

        if material_indices is None:
            indices = list(range(len(self.names)))
        else:
            indices = [int(value) for value in material_indices.detach().cpu().reshape(-1).tolist()]

        curves: list[np.ndarray] = []
        for idx in indices:
            if idx < 0 or idx >= len(self.names):
                raise IndexError(f"Material index {idx} is outside [0,{len(self.names)}).")
            native_wavelengths = self.wavelengths_nm[idx]
            below = float(native_wavelengths[0] - query.min())
            above = float(query.max() - native_wavelengths[-1])
            if require_coverage and max(below, above) > float(coverage_tolerance_nm):
                raise ValueError(
                    f"Material {self.names[idx]!r} covers {native_wavelengths[0]:g}-{native_wavelengths[-1]:g} nm, "
                    f"but the query requests {query.min():g}-{query.max():g} nm "
                    f"(allowed endpoint tolerance {coverage_tolerance_nm:g} nm)."
                )
            n_interp = np.interp(query, native_wavelengths, self.n_values[idx])
            k_interp = np.interp(query, native_wavelengths, self.k_values[idx])
            curves.append(np.stack((n_interp, k_interp), axis=-1))

        return torch.from_numpy(np.stack(curves, axis=0)).to(dtype=torch.float32)


def load_material_catalog(
    path: str | Path,
    material_names: Sequence[str] | None = None,
) -> MaterialCatalog:
    """Load native ``wavelength,n,k`` curves without silently extrapolating them."""
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"Material directory does not exist: {root}")

    if material_names is None:
        names = tuple(item.stem for item in sorted(root.glob("*.csv")))
    else:
        names = tuple(str(name) for name in material_names)
    if not names:
        raise ValueError(f"No material CSV files were selected from {root}.")

    wavelengths: list[np.ndarray] = []
    n_values: list[np.ndarray] = []
    k_values: list[np.ndarray] = []
    for name in names:
        material_path = root / f"{name}.csv"
        if not material_path.is_file():
            raise FileNotFoundError(f"Missing optical constants for material {name!r}: {material_path}")
        data = np.genfromtxt(material_path, delimiter=",", names=True)
        fields = set(data.dtype.names or ())
        wavelength_key = next(
            (candidate for candidate in ("nm", "wavelength_nm", "wavelength", "lambda", "wl") if candidate in fields),
            "",
        )
        if wavelength_key not in fields or "n" not in fields or "k" not in fields:
            raise ValueError(f"{material_path} must contain wavelength (nm/wl/wavelength/lambda), n, and k columns.")
        order = np.argsort(np.asarray(data[wavelength_key], dtype=np.float64))
        wavelengths.append(np.asarray(data[wavelength_key], dtype=np.float64)[order])
        n_values.append(np.asarray(data["n"], dtype=np.float64)[order])
        k_values.append(np.asarray(data["k"], dtype=np.float64)[order])

    return MaterialCatalog(
        names=names,
        wavelengths_nm=tuple(wavelengths),
        n_values=tuple(n_values),
        k_values=tuple(k_values),
    )


def material_names_from_tokens(tokens: Iterable[str]) -> tuple[str, ...]:
    """Return unique physical material names in token-vocabulary order."""
    names: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in SPECIAL_TOKENS:
            continue
        parts = layer_token_parts(str(token))
        if parts is None:
            continue
        material, _ = parts
        if material not in seen:
            seen.add(material)
            names.append(material)
    if not names:
        raise ValueError("Token vocabulary contains no physical layer tokens.")
    return tuple(names)


def parse_layer_stack(
    stack_ids: torch.Tensor,
    idx_to_token: dict[int, str],
    material_to_index: dict[str, int],
    *,
    eos_idx: int,
    pad_idx: int,
    msk_idx: int,
    merge_adjacent: bool = True,
) -> tuple[list[int], list[float]]:
    """Parse token IDs into physical material indices and continuous thicknesses."""
    material_ids: list[int] = []
    thickness_nm: list[float] = []
    for value in stack_ids.detach().cpu().reshape(-1).tolist():
        token_id = int(value)
        if token_id == eos_idx:
            break
        if token_id in {pad_idx, msk_idx}:
            continue
        token = idx_to_token.get(token_id)
        if token is None:
            raise KeyError(f"Token id {token_id} is absent from idx_to_token.")
        parts = layer_token_parts(token)
        if parts is None:
            continue
        material, thickness = parts
        if material not in material_to_index:
            raise KeyError(f"Material {material!r} is absent from the material catalog.")
        if thickness <= 0 or not np.isfinite(thickness):
            raise ValueError(f"Layer token {token!r} has invalid thickness {thickness}.")
        material_id = int(material_to_index[material])
        if merge_adjacent and material_ids and material_ids[-1] == material_id:
            thickness_nm[-1] += float(thickness)
        else:
            material_ids.append(material_id)
            thickness_nm.append(float(thickness))
    return material_ids, thickness_nm


@dataclass(frozen=True)
class ThicknessTransform:
    """Invertible log transform for positive layer thicknesses."""

    min_nm: float = 5.0
    max_nm: float = 10_000.0

    def __post_init__(self) -> None:
        """Validate the supported physical range."""
        if self.min_nm <= 0 or self.max_nm <= self.min_nm:
            raise ValueError(f"Invalid thickness range {self.min_nm:g}-{self.max_nm:g} nm.")

    def encode(self, thickness_nm: torch.Tensor) -> torch.Tensor:
        """Map thickness in nm to approximately ``[-1, 1]``."""
        thickness = thickness_nm.to(dtype=torch.float32).clamp(self.min_nm, self.max_nm)
        low = float(np.log(self.min_nm))
        span = float(np.log(self.max_nm) - low)
        return 2.0 * (torch.log(thickness) - low) / span - 1.0

    def decode(self, values: torch.Tensor) -> torch.Tensor:
        """Map normalized log thickness back to nm."""
        low = float(np.log(self.min_nm))
        span = float(np.log(self.max_nm) - low)
        normalized = values.to(dtype=torch.float32).clamp(-1.0, 1.0)
        return torch.exp(low + 0.5 * (normalized + 1.0) * span)


def sample_query_indices(
    width: int,
    *,
    min_points: int,
    max_points: int,
    mode: str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a sorted fixed-grid subset for one collated batch."""
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}.")
    low = max(1, min(int(min_points), width))
    high = max(low, min(int(max_points), width))
    normalized_mode = str(mode).lower().replace("-", "_")
    if normalized_mode == "mixed":
        draw = int(torch.randint(0, 3, (1,), generator=generator).item())
        normalized_mode = ("full", "window", "random")[draw]
    if normalized_mode == "full":
        return torch.arange(width, dtype=torch.long)
    if normalized_mode not in {"window", "random"}:
        raise ValueError(f"Unknown query sampling mode {mode!r}; expected full, window, random, or mixed.")
    count = int(torch.randint(low, high + 1, (1,), generator=generator).item()) if high > low else low
    if normalized_mode == "window":
        start_max = width - count
        start = int(torch.randint(0, start_max + 1, (1,), generator=generator).item()) if start_max > 0 else 0
        return torch.arange(start, start + count, dtype=torch.long)
    return torch.randperm(width, generator=generator)[:count].sort().values


class OpenLayerBatchCollator:
    """Convert existing spectrum/token samples into query-local open-layer batches."""

    def __init__(
        self,
        *,
        wavelengths_nm: torch.Tensor,
        catalog: MaterialCatalog,
        idx_to_token: dict[int, str],
        eos_idx: int,
        pad_idx: int,
        msk_idx: int,
        channels: Sequence[str] = ("R", "T"),
        max_layers: int = 100,
        max_candidates: int = 15,
        min_query_points: int = 64,
        max_query_points: int | None = None,
        query_sampling: str = "mixed",
        randomize_candidates: bool = True,
        random_distractors: bool = True,
        holdout_materials: Sequence[str] = (),
        merge_adjacent: bool = True,
        thickness_transform: ThicknessTransform | None = None,
        coverage_tolerance_nm: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.wavelengths_nm = wavelengths_nm.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        self.catalog = catalog
        self.idx_to_token = dict(idx_to_token)
        self.eos_idx = int(eos_idx)
        self.pad_idx = int(pad_idx)
        self.msk_idx = int(msk_idx)
        self.channels = tuple(str(channel).upper() for channel in channels)
        unknown_channels = set(self.channels) - set(CHANNEL_TO_INDEX)
        if unknown_channels:
            raise ValueError(f"Unknown target channels: {sorted(unknown_channels)}")
        self.max_layers = int(max_layers)
        self.max_candidates = int(max_candidates)
        self.min_query_points = int(min_query_points)
        self.max_query_points = int(max_query_points or len(self.wavelengths_nm))
        self.query_sampling = str(query_sampling)
        self.randomize_candidates = bool(randomize_candidates)
        self.random_distractors = bool(random_distractors)
        unknown_holdouts = set(str(value) for value in holdout_materials) - set(self.catalog.names)
        if unknown_holdouts:
            raise ValueError(f"Unknown holdout materials: {sorted(unknown_holdouts)}")
        self.holdout_indices = frozenset(self.catalog.name_to_index[str(value)] for value in holdout_materials)
        self.merge_adjacent = bool(merge_adjacent)
        self.thickness_transform = thickness_transform or ThicknessTransform()
        self.coverage_tolerance_nm = float(coverage_tolerance_nm)
        self.seed = int(seed)
        self.calls = 0
        if self.max_layers <= 0 or self.max_candidates <= 0:
            raise ValueError("max_layers and max_candidates must be positive.")
        if len(self.catalog.names) > self.max_candidates:
            # Per-sample banks can still be smaller, but the complete catalog is not assumed to fit.
            pass

    def _generator(self) -> torch.Generator:
        generator = torch.Generator()
        worker = torch.utils.data.get_worker_info()
        worker_id = 0 if worker is None else int(worker.id) + 1
        generator.manual_seed(self.seed + self.calls * 1_000_003 + worker_id * 97)
        self.calls += 1
        return generator

    def _candidate_indices(self, true_ids: Sequence[int], generator: torch.Generator) -> torch.Tensor:
        unique_true = list(dict.fromkeys(int(value) for value in true_ids))
        if len(unique_true) > self.max_candidates:
            raise ValueError(f"A stack uses {len(unique_true)} materials, exceeding MAX_CANDIDATES={self.max_candidates}.")
        true_set = set(unique_true)
        distractors = [idx for idx in range(len(self.catalog.names)) if idx not in true_set and idx not in self.holdout_indices]
        if distractors:
            order = torch.randperm(len(distractors), generator=generator).tolist()
            distractors = [distractors[idx] for idx in order]
        selected = unique_true
        if self.random_distractors:
            selected += distractors[: self.max_candidates - len(unique_true)]
        if not selected:
            # Empty stacks are unsupervised, but still need a non-empty bank so
            # their placeholder layer can pass safely through attention.
            selected = [0]
        if self.randomize_candidates and len(selected) > 1:
            order = torch.randperm(len(selected), generator=generator).tolist()
            selected = [selected[idx] for idx in order]
        return torch.tensor(selected, dtype=torch.long)

    def __call__(self, samples: Sequence[tuple[torch.Tensor, torch.Tensor, int]]) -> dict[str, torch.Tensor]:
        """Collate raw dataset tuples into model-ready tensors."""
        if not samples:
            raise ValueError("Cannot collate an empty sample list.")
        generator = self._generator()
        width = int(samples[0][0].shape[-1])
        if width != len(self.wavelengths_nm):
            raise ValueError(f"Dataset spectrum width {width} does not match wavelength grid {len(self.wavelengths_nm)}.")
        query_indices = sample_query_indices(
            width,
            min_points=self.min_query_points,
            max_points=self.max_query_points,
            mode=self.query_sampling,
            generator=generator,
        )
        query_wavelengths = self.wavelengths_nm[query_indices]
        all_curves = self.catalog.interpolate(
            query_wavelengths,
            coverage_tolerance_nm=self.coverage_tolerance_nm,
        )

        batch_size = len(samples)
        query_count = len(query_indices)
        channel_indices = torch.tensor([CHANNEL_TO_INDEX[channel] for channel in self.channels], dtype=torch.long)
        target = torch.empty((batch_size, query_count, len(self.channels)), dtype=torch.float32)
        candidate_nk = torch.zeros((batch_size, self.max_candidates, query_count, 2), dtype=torch.float32)
        candidate_mask = torch.zeros((batch_size, self.max_candidates), dtype=torch.bool)
        candidate_global_ids = torch.full((batch_size, self.max_candidates), -1, dtype=torch.long)
        material_targets = torch.full((batch_size, self.max_layers), -100, dtype=torch.long)
        thickness_nm = torch.zeros((batch_size, self.max_layers), dtype=torch.float32)
        layer_mask = torch.zeros((batch_size, self.max_layers), dtype=torch.bool)
        sample_indices = torch.empty(batch_size, dtype=torch.long)
        sample_mask = torch.ones(batch_size, dtype=torch.bool)

        material_to_index = self.catalog.name_to_index
        for batch_idx, (spectrum, stack, sample_index) in enumerate(samples):
            if spectrum.shape != (3, width):
                raise ValueError(f"Expected spectrum [3,{width}], got {tuple(spectrum.shape)}.")
            global_materials, layer_thickness = parse_layer_stack(
                stack,
                self.idx_to_token,
                material_to_index,
                eos_idx=self.eos_idx,
                pad_idx=self.pad_idx,
                msk_idx=self.msk_idx,
                merge_adjacent=self.merge_adjacent,
            )
            physical_layer_count = len(global_materials)
            if physical_layer_count == 0:
                sample_mask[batch_idx] = False
            if any(material_id in self.holdout_indices for material_id in global_materials):
                sample_mask[batch_idx] = False
            if physical_layer_count > self.max_layers:
                raise ValueError(
                    f"Sample {sample_index} has {physical_layer_count} layers after canonicalization, "
                    f"exceeding MAX_LAYERS={self.max_layers}."
                )
            candidates = self._candidate_indices(global_materials, generator)
            local_lookup = {int(global_id): local_id for local_id, global_id in enumerate(candidates.tolist())}
            local_materials = torch.tensor([local_lookup[value] for value in global_materials], dtype=torch.long)
            active_layer_count = max(physical_layer_count, 1)
            candidate_count = len(candidates)

            target[batch_idx] = spectrum.index_select(0, channel_indices).index_select(1, query_indices).transpose(0, 1)
            candidate_nk[batch_idx, :candidate_count] = all_curves.index_select(0, candidates)
            candidate_mask[batch_idx, :candidate_count] = True
            candidate_global_ids[batch_idx, :candidate_count] = candidates
            if physical_layer_count:
                material_targets[batch_idx, :physical_layer_count] = local_materials
                thickness_nm[batch_idx, :physical_layer_count] = torch.tensor(layer_thickness, dtype=torch.float32)
            layer_mask[batch_idx, :active_layer_count] = True
            sample_indices[batch_idx] = int(sample_index)

        return {
            "wavelengths_nm": query_wavelengths.unsqueeze(0).expand(batch_size, -1).clone(),
            "target_spectrum": target,
            "query_mask": torch.ones((batch_size, query_count), dtype=torch.bool),
            "candidate_nk": candidate_nk,
            "candidate_mask": candidate_mask,
            "candidate_global_ids": candidate_global_ids,
            "material_targets": material_targets,
            "thickness_targets": self.thickness_transform.encode(thickness_nm.clamp_min(self.thickness_transform.min_nm)),
            "thickness_nm": thickness_nm,
            "layer_mask": layer_mask,
            "sample_indices": sample_indices,
            "sample_mask": sample_mask,
        }


def make_open_layer_condition(
    *,
    target_spectrum: torch.Tensor,
    wavelengths_nm: torch.Tensor,
    catalog: MaterialCatalog,
    candidate_names: Sequence[str] | None = None,
    coverage_tolerance_nm: float = 0.0,
    channels: Sequence[str] = ("R", "T"),
) -> dict[str, torch.Tensor]:
    """Build one inference condition using a named query-local material bank."""
    spectrum = target_spectrum.detach().to(device="cpu", dtype=torch.float32)
    wavelengths = wavelengths_nm.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    if spectrum.ndim != 2 or spectrum.shape[0] not in {2, 3} or spectrum.shape[1] != len(wavelengths):
        raise ValueError("target_spectrum must be [2,W] or [3,W] and match wavelengths_nm.")
    normalized_channels = tuple(str(channel).upper() for channel in channels)
    if spectrum.shape[0] == 2 and normalized_channels != ("R", "T"):
        raise ValueError("Two-channel target spectra can only be interpreted as [R,T].")
    channel_indices = [CHANNEL_TO_INDEX[channel] for channel in normalized_channels]
    target_values = spectrum[channel_indices].transpose(0, 1).unsqueeze(0)
    if candidate_names is None:
        candidate_indices = torch.arange(len(catalog.names), dtype=torch.long)
    else:
        mapping = catalog.name_to_index
        missing = [name for name in candidate_names if name not in mapping]
        if missing:
            raise KeyError(f"Unknown candidate materials: {missing}")
        if not candidate_names:
            raise ValueError("At least one candidate material is required.")
        if len(set(candidate_names)) != len(candidate_names):
            raise ValueError("Candidate material names must be unique.")
        candidate_indices = torch.tensor([mapping[name] for name in candidate_names], dtype=torch.long)
    curves = catalog.interpolate(
        wavelengths,
        candidate_indices,
        coverage_tolerance_nm=coverage_tolerance_nm,
    )
    return {
        "wavelengths_nm": wavelengths.unsqueeze(0),
        "target_spectrum": target_values,
        "query_mask": torch.ones((1, len(wavelengths)), dtype=torch.bool),
        "candidate_nk": curves.unsqueeze(0),
        "candidate_mask": torch.ones((1, len(candidate_indices)), dtype=torch.bool),
        "candidate_global_ids": candidate_indices.unsqueeze(0),
    }


def layer_batch_to_runs(
    material_ids: torch.Tensor,
    thickness_nm: torch.Tensor,
    candidate_global_ids: torch.Tensor,
    catalog: MaterialCatalog,
    layer_mask: torch.Tensor,
) -> list[list[dict[str, Any]]]:
    """Decode local material-pointer outputs into TMM-compatible material runs."""
    if material_ids.shape != thickness_nm.shape or material_ids.shape != layer_mask.shape:
        raise ValueError("material_ids, thickness_nm, and layer_mask must have matching [B,L] shapes.")
    runs_batch: list[list[dict[str, Any]]] = []
    for row in range(material_ids.shape[0]):
        runs: list[dict[str, Any]] = []
        for layer in range(material_ids.shape[1]):
            if not bool(layer_mask[row, layer]):
                continue
            local_id = int(material_ids[row, layer].item())
            if local_id < 0 or local_id >= candidate_global_ids.shape[1]:
                raise ValueError(f"Invalid local material id {local_id} at row={row}, layer={layer}.")
            global_id = int(candidate_global_ids[row, local_id].item())
            if global_id < 0 or global_id >= len(catalog.names):
                raise ValueError(f"Local candidate {local_id} is padding at row={row}.")
            material = catalog.names[global_id]
            value = float(thickness_nm[row, layer].item())
            if runs and runs[-1]["material"] == material:
                runs[-1]["thickness_nm"] += value
            else:
                runs.append({"material": material, "thickness_nm": value})
        runs_batch.append(runs)
    return runs_batch
