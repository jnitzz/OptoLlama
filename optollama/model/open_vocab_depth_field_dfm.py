"""Revisable discrete flow over candidate-local depth-field labels."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable

import torch

from .open_vocab_depth_field import OpenVocabularyDepthFieldConfig, OpenVocabularyDepthFieldDiffusion


@dataclass(frozen=True)
class DiscreteFlowConfig:
    """Source-state mixture for candidate-local discrete flow."""

    source_mask_prob: float = 0.8
    source_void_prob: float = 0.1
    source_material_prob: float = 0.1
    source_mode: str = "iid"
    block_min_bins: int = 8
    block_max_bins: int = 64
    block_ranges: tuple[tuple[int, int], ...] = ()
    block_scale_probs: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        """Validate the source and optional block-scale distributions."""
        probabilities = (self.source_mask_prob, self.source_void_prob, self.source_material_prob)
        if not all(math.isfinite(value) and value >= 0 for value in probabilities):
            raise ValueError("All discrete-flow source probabilities must be nonnegative and finite.")
        if not math.isclose(sum(probabilities), 1.0, abs_tol=1.0e-6):
            raise ValueError("Discrete-flow source probabilities must sum to one.")
        if self.source_mode not in {"iid", "block"}:
            raise ValueError("Discrete-flow source_mode must be 'iid' or 'block'.")
        if self.block_min_bins <= 0 or self.block_max_bins < self.block_min_bins:
            raise ValueError("Discrete-flow block sizes must be positive and ordered.")
        if self.block_ranges or self.block_scale_probs:
            if self.source_mode != "block" or not self.block_ranges or len(self.block_ranges) != len(self.block_scale_probs):
                raise ValueError("Block ranges and matching scale probabilities require SOURCE_MODE: block.")
            if any(
                len(bounds) != 2 or any(type(size) is not int for size in bounds) or not 0 < bounds[0] <= bounds[1]
                for bounds in self.block_ranges
            ):
                raise ValueError("Each discrete-flow block range must be an ordered pair of positive integers.")
            if not all(math.isfinite(value) and value >= 0 for value in self.block_scale_probs) or not math.isclose(
                sum(self.block_scale_probs), 1.0, abs_tol=1.0e-6
            ):
                raise ValueError("Block scale probabilities must be finite, nonnegative, and sum to one.")

    @classmethod
    def from_dict(cls, value: dict | None) -> DiscreteFlowConfig:
        """Read source probabilities from the YAML flow block."""
        value = value or {}

        def get(name: str, default):
            return value.get(name, value.get(name.lower(), default))

        return cls(
            source_mask_prob=float(get("SOURCE_MASK_PROB", 0.8)),
            source_void_prob=float(get("SOURCE_VOID_PROB", 0.1)),
            source_material_prob=float(get("SOURCE_MATERIAL_PROB", 0.1)),
            source_mode=str(get("SOURCE_MODE", "iid")).lower(),
            block_min_bins=int(get("BLOCK_MIN_BINS", 8)),
            block_max_bins=int(get("BLOCK_MAX_BINS", 64)),
            block_ranges=tuple(tuple(bounds) for bounds in get("BLOCK_RANGES", ())),
            block_scale_probs=tuple(float(prob) for prob in get("BLOCK_SCALE_PROBS", ())),
        )

    def to_dict(self) -> dict[str, float | int | str | tuple]:
        """Serialize the source distribution into checkpoint metadata."""
        metadata = asdict(self)
        if not self.block_ranges:
            del metadata["block_ranges"]
            del metadata["block_scale_probs"]
        return metadata


class OpenVocabularyDepthFieldDFM(OpenVocabularyDepthFieldDiffusion):
    """Patch4 backbone with a mixture probability path and CTMC sampling."""

    def __init__(self, config: OpenVocabularyDepthFieldConfig, flow_config: DiscreteFlowConfig):
        super().__init__(config)
        self.flow_config = flow_config
        self.mask_only_source = (
            flow_config.source_mask_prob, flow_config.source_void_prob, flow_config.source_material_prob
        ) == (1.0, 0.0, 0.0)
        if flow_config.source_mode == "block":
            ranges = flow_config.block_ranges or ((flow_config.block_min_bins, flow_config.block_max_bins),)
            if any(
                size > self.depth_bins or size % self.depth_patch_size for bounds in ranges for size in bounds
            ):
                raise ValueError("Block sizes must fit the field and be multiples of the depth patch size.")
            if flow_config.block_ranges:
                self.register_buffer(
                    "_block_ranges_patches", torch.tensor(ranges, dtype=torch.long) // self.depth_patch_size, persistent=False
                )
                self.register_buffer("_block_scale_probs", torch.tensor(flow_config.block_scale_probs), persistent=False)
                modal_scale = max(range(len(ranges)), key=flow_config.block_scale_probs.__getitem__)
                self._modal_block_min_patches = ranges[modal_scale][0] // self.depth_patch_size
                min_width = min(lo for (lo, _), prob in zip(ranges, flow_config.block_scale_probs, strict=True) if prob > 0)
                # Reserve space for partial blocks at both field edges, without a device-to-host reduction.
                self._multiscale_block_count = (self.depth_bins + min_width - 2 * self.depth_patch_size) // min_width + 1

    def _sample_block_source(
        self,
        candidate_mask: torch.Tensor,
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return a candidate-valid source and its patch-aligned block IDs."""
        if candidate_mask.ndim != 2 or candidate_mask.shape[1] != self.max_candidates:
            raise ValueError("Expected one candidate mask per sample.")
        if not bool(candidate_mask.any(dim=1).all()):
            raise ValueError("Every source sample needs at least one valid candidate.")
        batch_size = candidate_mask.shape[0]
        patch_size = self.depth_patch_size
        min_patches = self.flow_config.block_min_bins // patch_size
        max_patches = self.flow_config.block_max_bins // patch_size
        device = candidate_mask.device
        if self.flow_config.block_ranges and not deterministic:
            scales = torch.multinomial(self._block_scale_probs, batch_size, replacement=True, generator=generator)
            bounds = self._block_ranges_patches[scales]
            low, high = bounds[:, :1], bounds[:, 1:]
            block_width = low + (torch.rand(low.shape, device=device, generator=generator) * (high - low + 1)).long()
            offset = (torch.rand(low.shape, device=device, generator=generator) * block_width).long()
        elif deterministic:
            if self.flow_config.block_ranges:
                min_patches = self._modal_block_min_patches
            block_width = torch.full((batch_size, 1), min_patches, device=device, dtype=torch.long)
            offset = torch.zeros_like(block_width)
        else:
            block_width = torch.randint(
                min_patches, max_patches + 1, (batch_size, 1), device=device, generator=generator
            )
            offset = (torch.rand((batch_size, 1), device=device, generator=generator) * block_width).long()
        patch_positions = torch.arange(self.depth_bins, device=device).div(patch_size, rounding_mode="floor")
        block_ids = (patch_positions[None, :] + offset).div(block_width, rounding_mode="floor")
        block_count = (
            self._multiscale_block_count if self.flow_config.block_ranges
            else self.depth_bins // self.flow_config.block_min_bins + 1
        )
        if self.mask_only_source:
            return torch.full_like(block_ids, self.mask_id), block_ids, block_count
        if deterministic:
            material_probability = self.flow_config.source_material_prob / candidate_mask.sum(dim=1)
            best_material = candidate_mask.long().argmax(dim=1)
            label = torch.where(
                material_probability > max(self.flow_config.source_mask_prob, self.flow_config.source_void_prob),
                best_material,
                torch.full_like(
                    best_material,
                    self.mask_id
                    if self.flow_config.source_mask_prob >= self.flow_config.source_void_prob
                    else self.void_id,
                ),
            )
            block_labels = label[:, None].expand(-1, block_count)
        else:
            draws = torch.rand((batch_size, block_count), device=device, generator=generator)
            material = torch.multinomial(candidate_mask.float(), block_count, replacement=True, generator=generator)
            block_labels = torch.where(
                draws < self.flow_config.source_mask_prob,
                self.mask_id,
                torch.where(
                    draws < self.flow_config.source_mask_prob + self.flow_config.source_void_prob,
                    self.void_id,
                    material,
                ),
            )
        return block_labels.gather(1, block_ids).long(), block_ids, block_count

    def sample_source(
        self,
        candidate_mask: torch.Tensor,
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw source labels from the per-sample valid candidate bank."""
        if candidate_mask.ndim != 2 or candidate_mask.shape[1] != self.max_candidates:
            raise ValueError("Expected one candidate mask per sample.")
        if not bool(candidate_mask.any(dim=1).all()):
            raise ValueError("Every source sample needs at least one valid candidate.")
        if self.flow_config.source_mode == "block":
            source, _, _ = self._sample_block_source(
                candidate_mask, deterministic=deterministic, generator=generator
            )
            return source
        batch_size = candidate_mask.shape[0]
        if self.mask_only_source:
            return torch.full((batch_size, self.depth_bins), self.mask_id, device=candidate_mask.device, dtype=torch.long)
        if deterministic:
            material_probability = self.flow_config.source_material_prob / candidate_mask.sum(dim=1)
            best_material = candidate_mask.long().argmax(dim=1)
            source_label = torch.where(
                material_probability > max(self.flow_config.source_mask_prob, self.flow_config.source_void_prob),
                best_material,
                torch.full_like(
                    best_material,
                    self.mask_id if self.flow_config.source_mask_prob >= self.flow_config.source_void_prob else self.void_id,
                ),
            )
            return source_label[:, None].expand(batch_size, self.depth_bins).clone()
        draws = torch.rand((batch_size, self.depth_bins), device=candidate_mask.device, generator=generator)
        material = torch.multinomial(candidate_mask.float(), self.depth_bins, replacement=True, generator=generator)
        source = torch.where(
            draws < self.flow_config.source_mask_prob,
            self.mask_id,
            torch.where(
                draws < self.flow_config.source_mask_prob + self.flow_config.source_void_prob,
                self.void_id,
                material,
            ),
        )
        return source.long()

    def corrupt(
        self,
        clean_fields: torch.Tensor,
        timesteps: torch.Tensor,
        candidate_mask: torch.Tensor,
        *,
        random_replace_prob: float = 0.0,
        corruption_config: object = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the mixture path at the legacy noisy-to-clean time."""
        if random_replace_prob != 0.0:
            raise ValueError("DFM uses its source distribution, not RANDOM_REPLACE_PROB.")
        if clean_fields.ndim != 2 or clean_fields.shape != (candidate_mask.shape[0], self.depth_bins):
            raise ValueError("Expected one depth field per candidate bank.")
        if timesteps.numel() != clean_fields.shape[0]:
            raise ValueError("Expected one timestep per depth field.")
        if self.flow_config.source_mode == "block":
            source, block_ids, block_count = self._sample_block_source(candidate_mask, generator=generator)
        else:
            source = self.sample_source(candidate_mask, generator=generator)
        source_probability = self.noise_probability(timesteps).to(device=clean_fields.device)
        if self.flow_config.source_mode == "block":
            from_source = torch.rand(
                (clean_fields.shape[0], block_count), device=clean_fields.device, generator=generator
            ) < source_probability.reshape(-1, 1)
            from_source = from_source.gather(1, block_ids)
        else:
            from_source = torch.rand(clean_fields.shape, device=clean_fields.device, generator=generator)
            from_source = from_source < source_probability.reshape(-1, 1)
        return torch.where(from_source, source, clean_fields), from_source

    def training_loss(self, batch: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        """Use unweighted clean-field prediction for the flow posterior."""
        kwargs.setdefault("random_replace_prob", 0.0)
        kwargs.setdefault("void_loss_weight", 1.0)
        kwargs.setdefault("corrupted_loss_weight", 1.0)
        kwargs.setdefault("uncorrupted_loss_weight", 1.0)
        kwargs.setdefault("boundary_loss_enabled", False)
        return super().training_loss(batch, **kwargs)

    @torch.no_grad()
    def sample(
        self,
        *,
        spectra: torch.Tensor,
        wavelengths_nm: torch.Tensor,
        candidate_nk: torch.Tensor,
        candidate_mask: torch.Tensor,
        incidence_angle_deg: torch.Tensor,
        polarization_id: torch.Tensor,
        steps: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        deterministic: bool = False,
        guidance_scale: float = 1.0,
        remask_strategy: str = "random",
        corruption_config: object = None,
        generator: torch.Generator | None = None,
        trace_step: Callable[[int, float, torch.Tensor, torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        """Simulate source-to-data jumps with a changing endpoint posterior."""
        if steps <= 0:
            raise ValueError("steps must be positive.")
        if remask_strategy != "random":
            raise ValueError("DFM does not use a remasking strategy.")
        batch_size = spectra.shape[0]
        if self.flow_config.source_mode == "block":
            fields, block_ids, block_count = self._sample_block_source(
                candidate_mask, deterministic=deterministic, generator=generator
            )
        else:
            fields = self.sample_source(candidate_mask, deterministic=deterministic, generator=generator)
        material_memory = self.encode_materials(wavelengths_nm, candidate_nk, candidate_mask)
        for step_index in range(steps):
            # Model time runs from noisy 1 to clean 0; flow time runs from source 0 to data 1.
            model_time = 1.0 - step_index / steps
            timesteps = torch.full((batch_size,), model_time, device=spectra.device)

            def predict(condition: torch.Tensor) -> torch.Tensor:
                return self(
                    condition,
                    fields,
                    timesteps,
                    wavelengths_nm=wavelengths_nm,
                    candidate_nk=candidate_nk,
                    candidate_mask=candidate_mask,
                    incidence_angle_deg=incidence_angle_deg,
                    polarization_id=polarization_id,
                    encoded_materials=material_memory,
                )

            if guidance_scale == 1.0:
                logits = predict(spectra)
            else:
                unconditional = predict(torch.zeros_like(spectra))
                logits = unconditional + guidance_scale * (predict(spectra) - unconditional)
            endpoint, _, _ = self._sample_logits(
                logits,
                temperature=temperature,
                top_k=top_k,
                deterministic=deterministic,
                generator=generator,
            )
            # For kappa(s)=1-(1-s)^2, a source coordinate flips by the next grid point
            # with probability 1-((1-s_next)/(1-s_current))^2.
            flip_probability = 1.0 - ((steps - step_index - 1) / (steps - step_index)) ** 2
            if deterministic:
                if flip_probability >= 0.5:
                    fields = endpoint
            else:
                if self.flow_config.source_mode == "block":
                    flip = torch.rand(
                        (batch_size, block_count), device=fields.device, generator=generator
                    ) < flip_probability
                    flip = flip.gather(1, block_ids)
                else:
                    flip = torch.rand(fields.shape, device=fields.device, generator=generator) < flip_probability
                fields = torch.where(flip, endpoint, fields)
            if trace_step is not None:
                trace_step(step_index, model_time, endpoint.detach().cpu().clone(), fields.detach().cpu().clone())
        return fields
