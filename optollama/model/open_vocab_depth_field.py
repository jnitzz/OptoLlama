from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as functional
from torch import nn

from .depth_field_diffusion import (
    DepthFieldCorruptionConfig,
    DepthFieldHybridDiffusion,
    DepthFieldModelConfig,
    depth_field_boundary_mask,
    depth_field_corruption_mask,
    drop_spectrum_condition,
    scheduled_random_replace_probability,
    weighted_depth_field_loss,
)
from .open_layer_flow import (
    MaterialCurveEncoder,
    OpenLayerFlowConfig,
    SinusoidalTimeEmbedding,
    WavelengthFeatures,
)


@dataclass(frozen=True)
class OpenVocabularyDepthFieldConfig:
    """Architecture metadata for candidate-conditioned depth-field diffusion."""

    spectrum_shape: tuple[int, ...]
    depth_bins: int
    max_candidates: int = 24
    d_model: int = 896
    n_blocks: int = 8
    n_heads: int = 8
    ffn_multiplier: float = 4.0
    kernel_size: int = 7
    dropout: float = 0.0
    conv_type: str = "separable"
    hybrid_dilations: tuple[int, ...] = ()
    hybrid_residual_init: float = 1.0e-3
    spectrum_patch_size: int = 4
    spectrum_patch_stride: int = 2
    spectrum_encoder_blocks: int = 4
    spectrum_encoder_heads: int = 8
    spectrum_ffn_multiplier: float = 2.0
    wavelength_scale_nm: float = 1_000.0
    wavelength_fourier_bands: int = 4

    def __post_init__(self) -> None:
        """Normalize dilation defaults and validate structural dimensions."""
        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive.")
        if self.depth_bins <= 0:
            raise ValueError("depth_bins must be positive.")
        if not self.hybrid_dilations:
            base = (1, 2, 4, 8, 16, 32, 64)
            object.__setattr__(
                self,
                "hybrid_dilations",
                tuple(base[min(index, len(base) - 1)] for index in range(self.n_blocks)),
            )
        if len(self.hybrid_dilations) != self.n_blocks:
            raise ValueError("hybrid_dilations must contain one value per block.")

    def to_dict(self) -> dict[str, Any]:
        """Return checkpoint-friendly constructor metadata."""
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "OpenVocabularyDepthFieldConfig":
        """Restore constructor metadata from a checkpoint mapping."""
        values = dict(data)
        values["spectrum_shape"] = tuple(int(value) for value in values["spectrum_shape"])
        values["hybrid_dilations"] = tuple(int(value) for value in values.get("hybrid_dilations", ()))
        return OpenVocabularyDepthFieldConfig(**values)


class OpenVocabularyDepthFieldDiffusion(DepthFieldHybridDiffusion):
    """Continuous-time depth-field model with per-sample optical material banks."""

    def __init__(self, config: OpenVocabularyDepthFieldConfig) -> None:
        base_config = DepthFieldModelConfig(
            spectrum_shape=config.spectrum_shape,
            num_materials=config.max_candidates + 1,
            depth_bins=config.depth_bins,
            model_type="hybrid",
            d_model=config.d_model,
            n_blocks=config.n_blocks,
            kernel_size=config.kernel_size,
            n_heads=config.n_heads,
            ffn_multiplier=config.ffn_multiplier,
            timesteps=1,
            dropout=config.dropout,
            conv_type=config.conv_type,
            hybrid_dilations=config.hybrid_dilations,
            hybrid_residual_init=config.hybrid_residual_init,
            spectrum_patch_size=config.spectrum_patch_size,
            spectrum_patch_stride=config.spectrum_patch_stride,
            spectrum_encoder_blocks=config.spectrum_encoder_blocks,
            spectrum_encoder_heads=config.spectrum_encoder_heads,
            spectrum_ffn_multiplier=config.spectrum_ffn_multiplier,
        )
        super().__init__(base_config)
        self.open_vocab_config = config
        self.max_candidates = int(config.max_candidates)
        self.void_id = self.max_candidates
        self.num_materials = self.max_candidates + 1
        self.mask_id = self.max_candidates + 1

        coordinates = WavelengthFeatures(config.wavelength_scale_nm, config.wavelength_fourier_bands)
        encoder_config = OpenLayerFlowConfig(
            target_channels=int(config.spectrum_shape[0]),
            max_layers=1,
            d_model=config.d_model,
            n_blocks=1,
            n_heads=config.n_heads,
            ffn_multiplier=config.ffn_multiplier,
            query_encoder_blocks=0,
            dropout=config.dropout,
            wavelength_scale_nm=config.wavelength_scale_nm,
            wavelength_fourier_bands=config.wavelength_fourier_bands,
        )
        self.material_encoder = MaterialCurveEncoder(encoder_config, coordinates)
        self.special_embeddings = nn.Parameter(torch.empty(2, config.d_model))
        self.candidate_context = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
        )
        self.optical_condition = nn.Sequential(
            nn.Linear(4, config.d_model, bias=False),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model, bias=False),
        )
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.pointer_query = nn.Linear(config.d_model, config.d_model, bias=False)
        self.pointer_key = nn.Linear(config.d_model, config.d_model, bias=False)
        del self.input_embedding
        del self.output
        nn.init.normal_(self.special_embeddings, mean=0.0, std=0.02)

    def noise_probability(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Map normalized continuous time to the legacy quadratic noise level."""
        return timesteps.to(dtype=torch.float32).clamp(0.0, 1.0).square()

    @staticmethod
    def _optical_features(angles_deg: torch.Tensor, polarization_id: torch.Tensor) -> torch.Tensor:
        """Encode incidence and polarization with an exact zero at normal incidence."""
        angle = torch.deg2rad(angles_deg.to(dtype=torch.float32).reshape(-1))
        polarization_sign = polarization_id.to(dtype=torch.float32).reshape(-1).mul(2.0).sub(1.0)
        sine = torch.sin(angle)
        cosine_offset = 1.0 - torch.cos(angle)
        return torch.stack(
            (sine, cosine_offset, polarization_sign * sine, polarization_sign * cosine_offset),
            dim=-1,
        )

    def encode_materials(
        self,
        wavelengths_nm: torch.Tensor,
        candidate_nk: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode candidate n/k curves once for one or more denoising steps."""
        query_mask = torch.ones_like(wavelengths_nm, dtype=torch.bool)
        return self.material_encoder(wavelengths_nm, candidate_nk, query_mask, candidate_mask)

    def _material_state(self, fields: torch.Tensor, material_memory: torch.Tensor) -> torch.Tensor:
        batch, depth = fields.shape
        special = self.special_embeddings.unsqueeze(0).expand(batch, -1, -1)
        states = torch.cat((material_memory, special), dim=1)
        safe = fields.long().clamp(0, self.mask_id)
        return states.gather(1, safe.unsqueeze(-1).expand(batch, depth, self.d_model))

    def forward(
        self,
        spectra: torch.Tensor,
        noised_fields: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        wavelengths_nm: torch.Tensor,
        candidate_nk: torch.Tensor,
        candidate_mask: torch.Tensor,
        incidence_angle_deg: torch.Tensor,
        polarization_id: torch.Tensor,
        encoded_materials: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict candidate-local material/void logits for every depth bin."""
        if spectra.shape[1:] != self.spectrum_shape:
            raise ValueError(f"Expected spectra shape [B,{self.spectrum_shape}], got {tuple(spectra.shape)}")
        if noised_fields.shape != (spectra.size(0), self.depth_bins):
            raise ValueError(f"Expected fields [B,{self.depth_bins}], got {tuple(noised_fields.shape)}")
        if candidate_mask.shape != (spectra.size(0), self.max_candidates):
            raise ValueError(f"Expected candidate_mask [B,{self.max_candidates}], got {tuple(candidate_mask.shape)}")
        if not torch.all(candidate_mask.any(dim=1)):
            raise ValueError("Each sample requires at least one candidate material.")

        material_memory = (
            self.encode_materials(wavelengths_nm, candidate_nk, candidate_mask) if encoded_materials is None else encoded_materials
        )
        spectrum_tokens = self._spectrum_tokens(spectra)
        optical = self.optical_condition(self._optical_features(incidence_angle_deg, polarization_id)).to(
            dtype=spectrum_tokens.dtype
        )
        spectrum_tokens = spectrum_tokens + optical.unsqueeze(1)
        candidate_weights = candidate_mask.to(dtype=material_memory.dtype).unsqueeze(-1)
        pooled_candidates = (material_memory * candidate_weights).sum(dim=1) / candidate_weights.sum(dim=1).clamp_min(1.0)
        candidate_token = self.candidate_context(pooled_candidates).unsqueeze(1)
        spectrum_tokens = torch.cat((spectrum_tokens[:, :-1], candidate_token, spectrum_tokens[:, -1:]), dim=1)

        depth_tokens = self._material_state(noised_fields, material_memory)
        depth_tokens = depth_tokens + self.positional_encoding(depth_tokens).to(dtype=depth_tokens.dtype)
        time_token = self.time_embedding(timesteps.reshape(-1)).unsqueeze(1)
        depth_tokens = depth_tokens + time_token.to(dtype=depth_tokens.dtype)
        cond = self._block_condition(spectrum_tokens, time_token).to(dtype=depth_tokens.dtype)
        depth_tokens = self._run_depth_blocks(depth_tokens, spectrum_tokens, cond)
        features = self.final_depth_norm(depth_tokens)

        query = self.pointer_query(features)
        keys = self.pointer_key(material_memory)
        material_logits = torch.einsum("bld,bmd->blm", query, keys) / math.sqrt(self.d_model)
        finite_floor = torch.finfo(material_logits.dtype).min
        material_logits = material_logits.masked_fill(~candidate_mask[:, None, :], finite_floor)
        void_key = self.pointer_key(self.special_embeddings[0]).view(1, 1, -1)
        void_logits = (query * void_key).sum(dim=-1, keepdim=True) / math.sqrt(self.d_model)
        return torch.cat((material_logits, void_logits), dim=-1)

    def corrupt(
        self,
        clean_fields: torch.Tensor,
        timesteps: torch.Tensor,
        candidate_mask: torch.Tensor,
        *,
        random_replace_prob: float = 0.10,
        corruption_config: DepthFieldCorruptionConfig | dict | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply hybrid masks and valid candidate-local random replacements."""
        probability = self.noise_probability(timesteps).to(device=clean_fields.device)
        corrupted = depth_field_corruption_mask(
            clean_fields,
            probability,
            config=corruption_config,
            generator=generator,
        )
        replacement_probability = scheduled_random_replace_probability(
            probability,
            random_replace_prob,
            config=corruption_config,
        )
        replace = corrupted & (
            torch.rand(clean_fields.shape, device=clean_fields.device, generator=generator) < replacement_probability.unsqueeze(1)
        )
        candidate_counts = candidate_mask.sum(dim=1).long()
        draws = torch.rand(clean_fields.shape, device=clean_fields.device, generator=generator)
        local = torch.floor(draws * (candidate_counts + 1).unsqueeze(1)).long()
        random_labels = torch.where(local == candidate_counts.unsqueeze(1), self.void_id, local)
        noised = clean_fields.clone()
        noised[replace] = random_labels[replace]
        noised[corrupted & ~replace] = self.mask_id
        return noised, corrupted

    def training_loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        void_loss_weight: float = 0.10,
        random_replace_prob: float = 0.10,
        corruption_config: DepthFieldCorruptionConfig | dict | None = None,
        corrupted_loss_weight: float = 1.0,
        uncorrupted_loss_weight: float = 0.1,
        condition_dropout_prob: float = 0.0,
        boundary_loss_enabled: bool = True,
        boundary_loss_radius_bins: int = 2,
        boundary_loss_weight: float = 2.0,
        timesteps: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute continuous-time candidate-local denoising loss."""
        clean = batch["clean_fields"].long()
        batch_size = clean.shape[0]
        if timesteps is None:
            timesteps = torch.rand(batch_size, device=clean.device, generator=generator)
        timesteps = timesteps.to(device=clean.device, dtype=torch.float32).reshape(-1).clamp(0.0, 1.0)
        noised, corrupted = self.corrupt(
            clean,
            timesteps,
            batch["candidate_mask"],
            random_replace_prob=random_replace_prob,
            corruption_config=corruption_config,
            generator=generator,
        )
        spectra, condition_dropped = drop_spectrum_condition(
            batch["target_spectrum"].transpose(1, 2),
            condition_dropout_prob,
            generator=generator,
        )
        logits = self(
            spectra,
            noised,
            timesteps,
            wavelengths_nm=batch["wavelengths_nm"],
            candidate_nk=batch["candidate_nk"],
            candidate_mask=batch["candidate_mask"],
            incidence_angle_deg=batch["incidence_angle_deg"],
            polarization_id=batch["polarization_id"],
        )
        weights = torch.ones(self.num_materials, dtype=logits.dtype, device=logits.device)
        weights[self.void_id] = float(void_loss_weight)
        loss_per_bin = functional.cross_entropy(
            logits.reshape(-1, self.num_materials),
            clean.reshape(-1),
            weight=weights,
            reduction="none",
        ).view_as(clean)
        boundary = depth_field_boundary_mask(clean, boundary_loss_radius_bins)
        if boundary_loss_enabled:
            loss_per_bin = loss_per_bin * torch.where(
                boundary,
                loss_per_bin.new_full((), float(boundary_loss_weight)),
                loss_per_bin.new_ones(()),
            )
        loss = weighted_depth_field_loss(
            loss_per_bin,
            corrupted,
            corrupted_loss_weight=corrupted_loss_weight,
            uncorrupted_loss_weight=uncorrupted_loss_weight,
        )
        predictions = logits.argmax(dim=-1)
        return {
            "loss": loss,
            "logits": logits,
            "timesteps": timesteps,
            "noised_fields": noised,
            "corrupted": corrupted,
            "condition_dropped": condition_dropped,
            "boundary": boundary,
            "accuracy": (predictions == clean).float().mean(),
            "corrupted_accuracy": (predictions[corrupted] == clean[corrupted]).float().mean()
            if bool(corrupted.any())
            else loss.new_zeros(()),
        }

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
        corruption_config: DepthFieldCorruptionConfig | dict | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample with an arbitrary discretization of the continuous denoising path."""
        if steps <= 0:
            raise ValueError("steps must be positive.")
        strategy = self._normalize_remask_strategy(remask_strategy)
        batch_size = spectra.shape[0]
        fields = torch.full((batch_size, self.depth_bins), self.mask_id, dtype=torch.long, device=spectra.device)
        material_memory = self.encode_materials(wavelengths_nm, candidate_nk, candidate_mask)
        time_grid = torch.linspace(1.0, 0.0, steps + 1, device=spectra.device)
        for step_index in range(steps):
            current_t = time_grid[step_index]
            next_t = time_grid[step_index + 1]
            timesteps = current_t.expand(batch_size)

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
                logits = (
                    unconditional if guidance_scale == 0.0 else unconditional + guidance_scale * (predict(spectra) - unconditional)
                )
            prediction, confidence, _ = self._sample_logits(
                logits,
                temperature=temperature,
                top_k=top_k,
                deterministic=deterministic,
                generator=generator,
            )
            fields = prediction
            if step_index + 1 == steps:
                break
            next_probability = float(self.noise_probability(next_t.reshape(1)).item())
            if strategy == "random":
                remask = depth_field_corruption_mask(
                    fields,
                    fields.new_full((batch_size,), next_probability, dtype=torch.float32),
                    config=corruption_config,
                    generator=generator,
                )
            else:
                mask_count = min(self.depth_bins, int(round(next_probability * self.depth_bins)))
                remask = torch.zeros_like(fields, dtype=torch.bool)
                if mask_count:
                    indices = torch.topk(confidence, k=mask_count, dim=1, largest=False).indices
                    remask.scatter_(1, indices, True)
            fields[remask] = self.mask_id
        return fields
