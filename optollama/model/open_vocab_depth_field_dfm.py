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

    def __post_init__(self) -> None:
        """Require a full-support probability distribution."""
        probabilities = (self.source_mask_prob, self.source_void_prob, self.source_material_prob)
        if not all(math.isfinite(value) and value > 0 for value in probabilities):
            raise ValueError("All discrete-flow source probabilities must be positive and finite.")
        if not math.isclose(sum(probabilities), 1.0, abs_tol=1.0e-6):
            raise ValueError("Discrete-flow source probabilities must sum to one.")

    @classmethod
    def from_dict(cls, value: dict | None) -> DiscreteFlowConfig:
        """Read source probabilities from the YAML flow block."""
        value = value or {}
        return cls(
            source_mask_prob=float(value.get("SOURCE_MASK_PROB", 0.8)),
            source_void_prob=float(value.get("SOURCE_VOID_PROB", 0.1)),
            source_material_prob=float(value.get("SOURCE_MATERIAL_PROB", 0.1)),
        )

    def to_dict(self) -> dict[str, float]:
        """Serialize the source distribution into checkpoint metadata."""
        return asdict(self)


class OpenVocabularyDepthFieldDFM(OpenVocabularyDepthFieldDiffusion):
    """Patch4 backbone with a mixture probability path and CTMC sampling."""

    def __init__(self, config: OpenVocabularyDepthFieldConfig, flow_config: DiscreteFlowConfig):
        super().__init__(config)
        self.flow_config = flow_config

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
        batch_size = candidate_mask.shape[0]
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
        source = self.sample_source(candidate_mask, generator=generator)
        source_probability = self.noise_probability(timesteps).to(device=clean_fields.device)
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
                flip = torch.rand(fields.shape, device=fields.device, generator=generator) < flip_probability
                fields = torch.where(flip, endpoint, fields)
            if trace_step is not None:
                trace_step(step_index, model_time, endpoint.detach().cpu().clone(), fields.detach().cpu().clone())
        return fields
