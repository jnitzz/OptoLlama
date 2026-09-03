from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as functional
from torch import nn

from optollama.data.open_layer import ThicknessTransform


@dataclass(frozen=True)
class OpenLayerFlowConfig:
    """Architecture and process definition for the open-layer MVP."""

    target_channels: int = 2
    max_layers: int = 100
    d_model: int = 512
    n_blocks: int = 12
    n_heads: int = 8
    ffn_multiplier: float = 4.0
    query_encoder_blocks: int = 2
    dropout: float = 0.0
    wavelength_scale_nm: float = 1_000.0
    wavelength_fourier_bands: int = 4
    thickness_loss_weight: float = 1.0
    thickness_huber_delta: float = 0.1
    min_thickness_nm: float = 5.0
    max_thickness_nm: float = 10_000.0
    max_total_thickness_nm: float = 10_000.0

    def __post_init__(self) -> None:
        """Validate values that determine checkpoint compatibility."""
        if self.target_channels <= 0 or self.max_layers <= 0 or self.d_model <= 0:
            raise ValueError("target_channels, max_layers, and d_model must be positive.")
        if self.n_blocks <= 0 or self.n_heads <= 0 or self.query_encoder_blocks < 0:
            raise ValueError("n_blocks/n_heads must be positive and query_encoder_blocks non-negative.")
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}.")
        if self.ffn_multiplier <= 0 or self.wavelength_scale_nm <= 0 or self.wavelength_fourier_bands < 0:
            raise ValueError("FFN, wavelength scale, and Fourier-band settings are invalid.")
        if self.thickness_loss_weight < 0 or self.thickness_huber_delta <= 0:
            raise ValueError("Thickness loss weight must be non-negative and Huber delta positive.")
        ThicknessTransform(self.min_thickness_nm, self.max_thickness_nm)
        if self.max_total_thickness_nm < self.min_thickness_nm:
            raise ValueError("max_total_thickness_nm must permit at least one minimum-thickness layer.")

    def to_dict(self) -> dict[str, Any]:
        """Return a checkpoint-friendly representation."""
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "OpenLayerFlowConfig":
        """Construct a config from serialized metadata."""
        return OpenLayerFlowConfig(**data)


class WavelengthFeatures(nn.Module):
    """Continuous wavelength coordinates without learned grid-index semantics."""

    def __init__(self, scale_nm: float, fourier_bands: int) -> None:
        super().__init__()
        self.scale_nm = float(scale_nm)
        self.fourier_bands = int(fourier_bands)
        frequencies = 2.0 ** torch.arange(self.fourier_bands, dtype=torch.float32)
        self.register_buffer("_frequencies", frequencies, persistent=False)

    @property
    def frequencies(self) -> torch.Tensor:
        """Return the registered Fourier frequencies with a concrete tensor type."""
        return self.get_buffer("_frequencies")

    @property
    def output_dim(self) -> int:
        """Return the number of generated coordinate channels."""
        return 3 + 2 * self.fourier_bands

    def forward(self, wavelengths_nm: torch.Tensor) -> torch.Tensor:
        """Encode positive wavelengths of shape ``[B,Q]``."""
        if wavelengths_nm.ndim != 2:
            raise ValueError(f"wavelengths_nm must be [B,Q], got {tuple(wavelengths_nm.shape)}.")
        wavelengths = wavelengths_nm.to(dtype=torch.float32)
        if torch.any(wavelengths <= 0):
            raise ValueError("wavelengths_nm must be positive.")
        scaled = wavelengths / self.scale_nm
        inverse = self.scale_nm / wavelengths
        features = [scaled.unsqueeze(-1), inverse.unsqueeze(-1), torch.log(scaled).unsqueeze(-1)]
        if self.fourier_bands:
            phase = math.pi * inverse.unsqueeze(-1) * self.frequencies
            features.extend((torch.sin(phase), torch.cos(phase)))
        return torch.cat(features, dim=-1)


class SinusoidalTimeEmbedding(nn.Module):
    """Continuous timestep embedding for a shared material/thickness process."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = int(d_model)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Embed normalized timesteps of shape ``[B]``."""
        half = self.d_model // 2
        exponent = -math.log(10_000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        exponent = exponent / max(half - 1, 1)
        phase = 1_000.0 * timesteps.to(dtype=torch.float32).reshape(-1, 1) * torch.exp(exponent).reshape(1, -1)
        embedding = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
        if embedding.shape[-1] < self.d_model:
            embedding = functional.pad(embedding, (0, self.d_model - embedding.shape[-1]))
        return embedding


class TargetQueryEncoder(nn.Module):
    """Encode variable-grid target spectra into wavelength-aware memory tokens."""

    def __init__(self, config: OpenLayerFlowConfig, coordinates: WavelengthFeatures) -> None:
        super().__init__()
        self.coordinates = coordinates
        self.input = nn.Sequential(
            nn.Linear(config.target_channels + coordinates.output_dim, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.d_model,
                    nhead=config.n_heads,
                    dim_feedforward=int(round(config.d_model * config.ffn_multiplier)),
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config.query_encoder_blocks)
            ]
        )
        self.output_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        wavelengths_nm: torch.Tensor,
        target_spectrum: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return target memory of shape ``[B,Q,D]``."""
        if target_spectrum.shape[:2] != wavelengths_nm.shape:
            raise ValueError("target_spectrum and wavelengths_nm must agree in batch/query dimensions.")
        coordinates = self.coordinates(wavelengths_nm)
        x = self.input(torch.cat((target_spectrum.to(dtype=torch.float32), coordinates), dim=-1))
        padding_mask = ~query_mask.to(dtype=torch.bool)
        for block in self.blocks:
            x = block(x, src_key_padding_mask=padding_mask)
        return self.output_norm(x)


class MaterialCurveEncoder(nn.Module):
    """Encode each candidate from its n/k curve on the current query grid."""

    def __init__(self, config: OpenLayerFlowConfig, coordinates: WavelengthFeatures) -> None:
        super().__init__()
        self.coordinates = coordinates
        self.point_encoder = nn.Sequential(
            nn.Linear(2 + coordinates.output_dim, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
        )
        self.pool_score = nn.Linear(config.d_model, 1)
        self.output = nn.Sequential(nn.Linear(config.d_model, config.d_model), nn.LayerNorm(config.d_model))

    def forward(
        self,
        wavelengths_nm: torch.Tensor,
        candidate_nk: torch.Tensor,
        query_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return one embedding per candidate with shape ``[B,M,D]``."""
        if candidate_nk.ndim != 4 or candidate_nk.shape[-1] != 2:
            raise ValueError(f"candidate_nk must be [B,M,Q,2], got {tuple(candidate_nk.shape)}.")
        batch, candidates, query_count, _ = candidate_nk.shape
        if wavelengths_nm.shape != (batch, query_count):
            raise ValueError("candidate_nk and wavelengths_nm must agree in batch/query dimensions.")
        coordinates = self.coordinates(wavelengths_nm).unsqueeze(1).expand(-1, candidates, -1, -1)
        points = self.point_encoder(torch.cat((candidate_nk.to(dtype=torch.float32), coordinates), dim=-1))
        score = self.pool_score(points).squeeze(-1)
        score = score.masked_fill(~query_mask[:, None, :].to(dtype=torch.bool), -torch.inf)
        weights = torch.softmax(score, dim=-1)
        pooled = torch.sum(weights.unsqueeze(-1) * points, dim=2)
        pooled = self.output(pooled)
        return pooled * candidate_mask.unsqueeze(-1).to(dtype=pooled.dtype)


class OpenLayerDecoderBlock(nn.Module):
    """Layer-token self-attention with separate target and material memories."""

    def __init__(self, config: OpenLayerFlowConfig) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(config.d_model)
        self.target_norm = nn.LayerNorm(config.d_model)
        self.material_norm = nn.LayerNorm(config.d_model)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.self_attention = nn.MultiheadAttention(config.d_model, config.n_heads, dropout=config.dropout, batch_first=True)
        self.target_attention = nn.MultiheadAttention(config.d_model, config.n_heads, dropout=config.dropout, batch_first=True)
        self.material_attention = nn.MultiheadAttention(config.d_model, config.n_heads, dropout=config.dropout, batch_first=True)
        hidden = int(round(config.d_model * config.ffn_multiplier))
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.d_model),
        )
        self.dropout = nn.Dropout(config.dropout)
        self.time_modulation = nn.Sequential(nn.SiLU(), nn.Linear(config.d_model, 2 * config.d_model))

    @staticmethod
    def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + scale[:, None, :]) + shift[:, None, :]

    def forward(
        self,
        x: torch.Tensor,
        *,
        time_embedding: torch.Tensor,
        layer_padding_mask: torch.Tensor,
        target_memory: torch.Tensor,
        query_padding_mask: torch.Tensor,
        material_memory: torch.Tensor,
        candidate_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one conditioned decoder block."""
        shift, scale = self.time_modulation(time_embedding).chunk(2, dim=-1)
        h = self._modulate(self.self_norm(x), shift, scale)
        h, _ = self.self_attention(h, h, h, key_padding_mask=layer_padding_mask, need_weights=False)
        x = x + self.dropout(h)

        h = self._modulate(self.target_norm(x), shift, scale)
        h, _ = self.target_attention(
            h,
            target_memory,
            target_memory,
            key_padding_mask=query_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(h)

        h = self._modulate(self.material_norm(x), shift, scale)
        h, _ = self.material_attention(
            h,
            material_memory,
            material_memory,
            key_padding_mask=candidate_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(h)
        x = x + self.dropout(self.ffn(self._modulate(self.ffn_norm(x), shift, scale)))
        return x.masked_fill(layer_padding_mask.unsqueeze(-1), 0.0)


class OpenLayerFlow(nn.Module):
    """Open-vocabulary layer generator with masked materials and continuous thickness flow."""

    MASK_MATERIAL = -1

    def __init__(self, config: OpenLayerFlowConfig) -> None:
        super().__init__()
        self.config = config
        self.thickness_transform = ThicknessTransform(config.min_thickness_nm, config.max_thickness_nm)
        coordinates = WavelengthFeatures(config.wavelength_scale_nm, config.wavelength_fourier_bands)
        self.target_encoder = TargetQueryEncoder(config, coordinates)
        self.material_encoder = MaterialCurveEncoder(config, coordinates)
        self.mask_embedding = nn.Parameter(torch.empty(config.d_model))
        self.position_embedding = nn.Parameter(torch.empty(config.max_layers, config.d_model))
        self.thickness_embedding = nn.Sequential(
            nn.Linear(1, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.blocks = nn.ModuleList([OpenLayerDecoderBlock(config) for _ in range(config.n_blocks)])
        self.output_norm = nn.LayerNorm(config.d_model)
        self.pointer_query = nn.Linear(config.d_model, config.d_model, bias=False)
        self.pointer_key = nn.Linear(config.d_model, config.d_model, bias=False)
        self.thickness_velocity = nn.Linear(config.d_model, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize learned mask and layer-position states."""
        nn.init.normal_(self.mask_embedding, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

    def _material_state(self, material_ids: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        batch, layers = material_ids.shape
        candidate_count = candidates.shape[1]
        valid = material_ids >= 0
        safe_ids = material_ids.clamp(0, max(candidate_count - 1, 0))
        gathered = candidates.gather(1, safe_ids.unsqueeze(-1).expand(batch, layers, candidates.shape[-1]))
        mask = self.mask_embedding.reshape(1, 1, -1).expand(batch, layers, -1)
        return torch.where(valid.unsqueeze(-1), gathered, mask)

    def _project_total_thickness(self, thickness_nm: torch.Tensor, layer_mask: torch.Tensor) -> torch.Tensor:
        """Project active layers onto the configured total-thickness budget."""
        projected = thickness_nm.clone()
        for row in range(projected.shape[0]):
            active = layer_mask[row]
            count = int(active.sum().item())
            if count == 0:
                continue
            minimum_total = count * self.config.min_thickness_nm
            if minimum_total > self.config.max_total_thickness_nm + 1.0e-6:
                raise ValueError(
                    f"{count} layers at min_thickness_nm={self.config.min_thickness_nm:g} exceed "
                    f"max_total_thickness_nm={self.config.max_total_thickness_nm:g}."
                )
            values = projected[row, active]
            if float(values.sum().item()) <= self.config.max_total_thickness_nm:
                continue
            excess = (values - self.config.min_thickness_nm).clamp_min(0.0)
            excess_budget = self.config.max_total_thickness_nm - minimum_total
            if float(excess.sum().item()) > 0.0:
                values = self.config.min_thickness_nm + excess * (excess_budget / excess.sum())
            else:
                values = torch.full_like(values, self.config.min_thickness_nm)
            projected[row, active] = values
        return projected

    def encode_condition(
        self,
        wavelengths_nm: torch.Tensor,
        target_spectrum: torch.Tensor,
        query_mask: torch.Tensor,
        candidate_nk: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode target and material-bank memories once for repeated sampling steps."""
        target_memory = self.target_encoder(wavelengths_nm, target_spectrum, query_mask)
        material_memory = self.material_encoder(wavelengths_nm, candidate_nk, query_mask, candidate_mask)
        return target_memory, material_memory

    def forward(
        self,
        *,
        wavelengths_nm: torch.Tensor,
        target_spectrum: torch.Tensor,
        query_mask: torch.Tensor,
        candidate_nk: torch.Tensor,
        candidate_mask: torch.Tensor,
        material_ids: torch.Tensor,
        thickness_state: torch.Tensor,
        layer_mask: torch.Tensor,
        timesteps: torch.Tensor,
        encoded_condition: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict query-local material logits and normalized-thickness velocity."""
        if material_ids.shape != thickness_state.shape or material_ids.shape != layer_mask.shape:
            raise ValueError("material_ids, thickness_state, and layer_mask must have matching [B,L] shapes.")
        batch, layers = material_ids.shape
        if layers > self.config.max_layers:
            raise ValueError(f"Layer count {layers} exceeds max_layers={self.config.max_layers}.")
        if timesteps.reshape(-1).shape[0] != batch:
            raise ValueError("timesteps must contain one value per batch item.")
        if not torch.all(candidate_mask.any(dim=1)):
            raise ValueError("Each batch item requires at least one candidate material.")
        if not torch.all(query_mask.any(dim=1)):
            raise ValueError("Each batch item requires at least one wavelength query.")

        if encoded_condition is None:
            target_memory, material_memory = self.encode_condition(
                wavelengths_nm, target_spectrum, query_mask, candidate_nk, candidate_mask
            )
        else:
            target_memory, material_memory = encoded_condition
        time = self.time_embedding(timesteps.reshape(-1))
        x = self._material_state(material_ids, material_memory)
        x = x + self.thickness_embedding(thickness_state.unsqueeze(-1).to(dtype=torch.float32))
        x = x + self.position_embedding[:layers].unsqueeze(0) + time.unsqueeze(1)
        layer_padding = ~layer_mask.to(dtype=torch.bool)
        for block in self.blocks:
            x = block(
                x,
                time_embedding=time,
                layer_padding_mask=layer_padding,
                target_memory=target_memory,
                query_padding_mask=~query_mask.to(dtype=torch.bool),
                material_memory=material_memory,
                candidate_padding_mask=~candidate_mask.to(dtype=torch.bool),
            )
        x = self.output_norm(x)
        pointer_query = self.pointer_query(x)
        pointer_key = self.pointer_key(material_memory)
        logits = torch.einsum("bld,bmd->blm", pointer_query, pointer_key) / math.sqrt(self.config.d_model)
        logits = logits.masked_fill(~candidate_mask[:, None, :].to(dtype=torch.bool), -torch.inf)
        logits = logits.masked_fill(layer_padding.unsqueeze(-1), 0.0)
        velocity = self.thickness_velocity(x).squeeze(-1).masked_fill(layer_padding, 0.0)
        return {"material_logits": logits, "thickness_velocity": velocity}

    def training_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Sample joint corruption and return material/thickness denoising losses."""
        clean_materials = batch["material_targets"].to(dtype=torch.long)
        clean_thickness = batch["thickness_targets"].to(dtype=torch.float32)
        layer_mask = batch["layer_mask"].to(dtype=torch.bool)
        sample_mask = batch.get(
            "sample_mask",
            torch.ones(layer_mask.shape[0], device=layer_mask.device, dtype=torch.bool),
        ).to(dtype=torch.bool)
        supervised_layers = layer_mask & sample_mask[:, None]
        batch_size = clean_materials.shape[0]
        timesteps = torch.rand(batch_size, device=clean_materials.device).clamp_(1.0e-4, 1.0 - 1.0e-4)

        corrupted = torch.rand(clean_materials.shape, device=clean_materials.device) < timesteps[:, None]
        corrupted &= supervised_layers
        # Every sample contributes at least one supervised material decision.
        for row in range(batch_size):
            if bool(sample_mask[row]) and not bool(corrupted[row].any()):
                active = torch.nonzero(supervised_layers[row], as_tuple=False).flatten()
                if active.numel():
                    choice = active[torch.randint(active.numel(), (1,), device=active.device)]
                    corrupted[row, choice] = True
        noised_materials = clean_materials.clone()
        noised_materials[corrupted] = self.MASK_MATERIAL
        noised_materials[~layer_mask] = self.MASK_MATERIAL

        thickness_noise = torch.randn_like(clean_thickness)
        thickness_state = (1.0 - timesteps[:, None]) * clean_thickness + timesteps[:, None] * thickness_noise
        target_velocity = thickness_noise - clean_thickness
        outputs = self(
            wavelengths_nm=batch["wavelengths_nm"],
            target_spectrum=batch["target_spectrum"],
            query_mask=batch["query_mask"],
            candidate_nk=batch["candidate_nk"],
            candidate_mask=batch["candidate_mask"],
            material_ids=noised_materials,
            thickness_state=thickness_state,
            layer_mask=layer_mask,
            timesteps=timesteps,
        )
        if bool(corrupted.any()):
            material_loss = functional.cross_entropy(outputs["material_logits"][corrupted], clean_materials[corrupted])
            material_accuracy = (
                (outputs["material_logits"][corrupted].argmax(dim=-1) == clean_materials[corrupted]).to(dtype=torch.float32).mean()
            )
        else:
            finite_logits = torch.where(
                torch.isfinite(outputs["material_logits"]),
                outputs["material_logits"],
                torch.zeros_like(outputs["material_logits"]),
            )
            material_loss = finite_logits.sum() * 0.0
            material_accuracy = torch.zeros((), device=clean_materials.device)
        if bool(supervised_layers.any()):
            thickness_loss = functional.smooth_l1_loss(
                outputs["thickness_velocity"][supervised_layers],
                target_velocity[supervised_layers],
                beta=self.config.thickness_huber_delta,
            )
        else:
            thickness_loss = outputs["thickness_velocity"].sum() * 0.0
        total = material_loss + self.config.thickness_loss_weight * thickness_loss
        return {
            "loss": total,
            "material_loss": material_loss.detach(),
            "thickness_loss": thickness_loss.detach(),
            "material_accuracy": material_accuracy,
            "mean_timestep": timesteps.mean().detach(),
            "corrupted_fraction": corrupted.sum().to(dtype=torch.float32) / supervised_layers.sum().clamp_min(1),
            "supervised_samples": sample_mask.sum().detach(),
        }

    @torch.no_grad()
    def sample(
        self,
        *,
        wavelengths_nm: torch.Tensor,
        target_spectrum: torch.Tensor,
        query_mask: torch.Tensor,
        candidate_nk: torch.Tensor,
        candidate_mask: torch.Tensor,
        layer_counts: torch.Tensor,
        steps: int = 32,
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Sample fixed-count stacks with monotonic material revelation and Euler flow integration."""
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}.")
        batch = target_spectrum.shape[0]
        layer_counts = layer_counts.to(device=target_spectrum.device, dtype=torch.long).reshape(-1)
        if layer_counts.shape[0] != batch or torch.any(layer_counts < 1) or torch.any(layer_counts > self.config.max_layers):
            raise ValueError(f"layer_counts must be [B] within [1,{self.config.max_layers}].")
        layers = int(layer_counts.max().item())
        positions = torch.arange(layers, device=target_spectrum.device).unsqueeze(0)
        layer_mask = positions < layer_counts.unsqueeze(1)
        material_ids = torch.full((batch, layers), self.MASK_MATERIAL, device=target_spectrum.device, dtype=torch.long)
        thickness_state = torch.randn((batch, layers), device=target_spectrum.device)
        encoded = self.encode_condition(wavelengths_nm, target_spectrum, query_mask, candidate_nk, candidate_mask)

        for step_idx in range(steps):
            current_t = 1.0 - step_idx / steps
            next_t = 1.0 - (step_idx + 1) / steps
            timesteps = torch.full((batch,), current_t, device=target_spectrum.device)
            outputs = self(
                wavelengths_nm=wavelengths_nm,
                target_spectrum=target_spectrum,
                query_mask=query_mask,
                candidate_nk=candidate_nk,
                candidate_mask=candidate_mask,
                material_ids=material_ids,
                thickness_state=thickness_state,
                layer_mask=layer_mask,
                timesteps=timesteps,
                encoded_condition=encoded,
            )
            thickness_state = thickness_state - (current_t - next_t) * outputs["thickness_velocity"]
            probabilities = torch.softmax(outputs["material_logits"] / max(float(temperature), 1.0e-6), dim=-1)
            confidence, greedy = probabilities.max(dim=-1)
            proposals = (
                greedy
                if deterministic or temperature <= 0
                else torch.multinomial(probabilities.reshape(-1, probabilities.shape[-1]), 1).reshape(batch, layers)
            )

            for row in range(batch):
                active_count = int(layer_counts[row].item())
                desired_masked = int(math.ceil(next_t * active_count - 1.0e-9))
                masked_positions = torch.nonzero((material_ids[row] < 0) & layer_mask[row], as_tuple=False).flatten()
                reveal_count = max(0, int(masked_positions.numel()) - desired_masked)
                if reveal_count:
                    selected = masked_positions[torch.topk(confidence[row, masked_positions], k=reveal_count, largest=True).indices]
                    material_ids[row, selected] = proposals[row, selected]

        # Numerical drift outside the trained normalization range is clipped by decode.
        thickness_nm = self.thickness_transform.decode(thickness_state).masked_fill(~layer_mask, 0.0)
        thickness_nm = self._project_total_thickness(thickness_nm, layer_mask)
        if torch.any(material_ids[layer_mask] < 0):
            raise RuntimeError("Sampling ended with unresolved material slots.")
        return {
            "material_ids": material_ids,
            "thickness_nm": thickness_nm,
            "thickness_state": thickness_state,
            "layer_mask": layer_mask,
        }
