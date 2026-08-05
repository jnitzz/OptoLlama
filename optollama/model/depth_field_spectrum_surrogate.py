from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class DepthFieldSpectrumSurrogateConfig:
    """Serializable architecture and representation contract for a forward surrogate."""

    num_materials: int
    void_id: int
    depth_bins: int
    spectrum_width: int
    dz_nm: float
    d_model: int = 128
    conv_dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    kernel_size: int = 7
    depth_pool: int = 16
    decoder_blocks: int = 2
    decoder_heads: int = 8
    ffn_multiplier: float = 2.0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        """Normalize tuple fields and validate architecture dimensions."""
        object.__setattr__(self, "conv_dilations", tuple(int(value) for value in self.conv_dilations))
        if self.num_materials < 2:
            raise ValueError("num_materials must include at least one material and VOID.")
        if not 0 <= self.void_id < self.num_materials:
            raise ValueError(f"void_id={self.void_id} is outside [0,{self.num_materials}).")
        if self.depth_bins <= 0 or self.spectrum_width <= 1:
            raise ValueError("depth_bins and spectrum_width must be positive.")
        if self.dz_nm <= 0.0 or self.d_model <= 0:
            raise ValueError("dz_nm and d_model must be positive.")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        if not self.conv_dilations or any(value <= 0 for value in self.conv_dilations):
            raise ValueError("conv_dilations must contain positive integers.")
        if self.depth_pool <= 0 or self.decoder_blocks < 0 or self.decoder_heads <= 0:
            raise ValueError("depth_pool/decoder_heads must be positive and decoder_blocks non-negative.")
        if self.d_model % self.decoder_heads != 0:
            raise ValueError("d_model must be divisible by decoder_heads.")
        if self.ffn_multiplier <= 0.0 or not math.isfinite(self.ffn_multiplier):
            raise ValueError("ffn_multiplier must be finite and positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1).")

    def to_dict(self) -> dict[str, Any]:
        """Return a checkpoint-safe configuration mapping."""
        data = asdict(self)
        data["conv_dilations"] = list(self.conv_dilations)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DepthFieldSpectrumSurrogateConfig":
        """Build a configuration from checkpoint metadata."""
        return cls(**dict(data))


def straight_through_material_probabilities(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Return hard one-hot materials in the forward pass with softmax gradients."""
    if logits.dim() != 3:
        raise ValueError(f"logits must have shape [B,D,M], got {tuple(logits.shape)}")
    tau = float(temperature)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError(f"temperature must be finite and positive, got {temperature}")
    probabilities = functional.softmax(logits.float() / tau, dim=-1)
    hard = functional.one_hot(probabilities.argmax(dim=-1), num_classes=probabilities.size(-1)).to(probabilities.dtype)
    return hard + probabilities - probabilities.detach()


def compact_depth_field_probabilities(probabilities: torch.Tensor, void_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack non-VOID bins left while preserving gradients through packed values.

    The hard forward path exactly matches ``depth_field_runs``: VOID contributes
    zero thickness and later material bins move left. Scatter destinations are
    selected from the hard forward state, while gradients flow through the
    straight-through material probabilities.
    """
    if probabilities.dim() != 3:
        raise ValueError(f"probabilities must have shape [B,D,M], got {tuple(probabilities.shape)}")
    batch, depth, materials = probabilities.shape
    if not 0 <= int(void_id) < materials:
        raise ValueError(f"void_id={void_id} is outside [0,{materials}).")

    hard_ids = probabilities.detach().argmax(dim=-1)
    hard_active = hard_ids != int(void_id)
    destinations = (hard_active.long().cumsum(dim=1) - 1).clamp(min=0, max=depth - 1)
    raw_positions = torch.arange(depth, device=probabilities.device).unsqueeze(0)
    last_active = torch.where(hard_active, raw_positions, raw_positions.new_full((), -1)).amax(dim=1, keepdim=True)
    gradient_eligible = raw_positions <= (last_active + 1).clamp(max=depth - 1)

    nonvoid_indices = [index for index in range(materials) if index != int(void_id)]
    source = probabilities[..., nonvoid_indices] * gradient_eligible.unsqueeze(-1).to(probabilities.dtype)
    packed = probabilities.new_zeros((batch, depth, materials - 1))
    packed.scatter_add_(1, destinations.unsqueeze(-1).expand_as(source), source)

    active_source = (1.0 - probabilities[..., int(void_id)]) * gradient_eligible.to(probabilities.dtype)
    packed_active = probabilities.new_zeros((batch, depth))
    packed_active.scatter_add_(1, destinations, active_source)
    return packed, packed_active.clamp(0.0, 1.0)


class _SeparableResidualBlock(torch.nn.Module):
    def __init__(self, width: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.norm = torch.nn.GroupNorm(1, width)
        self.depthwise = torch.nn.Conv1d(
            width,
            width,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=width,
        )
        self.pointwise = torch.nn.Conv1d(width, width * 2, kernel_size=1)
        self.output = torch.nn.Conv1d(width, width, kernel_size=1)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.depthwise(functional.silu(self.norm(inputs)))
        values, gates = self.pointwise(hidden).chunk(2, dim=1)
        hidden = values * functional.silu(gates)
        return inputs + self.dropout(self.output(hidden))


class _WavelengthDecoderBlock(torch.nn.Module):
    def __init__(self, width: int, heads: int, ffn_multiplier: float, dropout: float) -> None:
        super().__init__()
        self.query_norm = torch.nn.LayerNorm(width)
        self.memory_norm = torch.nn.LayerNorm(width)
        self.cross_attention = torch.nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        hidden = max(width, int(round(width * ffn_multiplier)))
        self.ffn_norm = torch.nn.LayerNorm(width)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(width, hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, width),
            torch.nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        normalized_queries = self.query_norm(queries)
        attended, _ = self.cross_attention(
            normalized_queries,
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )
        queries = queries + attended
        return queries + self.ffn(self.ffn_norm(queries))


class DepthFieldSpectrumSurrogate(torch.nn.Module):
    """Predict a conserved R/A/T spectrum from a compacted material depth field."""

    def __init__(self, config: DepthFieldSpectrumSurrogateConfig) -> None:
        super().__init__()
        self.config = config
        nonvoid_materials = int(config.num_materials) - 1
        self.material_projection = torch.nn.Linear(nonvoid_materials, config.d_model, bias=False)
        self.active_projection = torch.nn.Linear(1, config.d_model)
        self.register_buffer(
            "depth_positions",
            torch.arange(config.depth_bins, dtype=torch.float32).mul(float(config.dz_nm)),
            persistent=False,
        )
        self.depth_position_projection = torch.nn.Sequential(
            torch.nn.Linear(17, config.d_model),
            torch.nn.SiLU(),
            torch.nn.Linear(config.d_model, config.d_model),
        )
        self.depth_blocks = torch.nn.ModuleList(
            [
                _SeparableResidualBlock(config.d_model, config.kernel_size, dilation, config.dropout)
                for dilation in config.conv_dilations
            ]
        )
        self.depth_pool = torch.nn.AvgPool1d(config.depth_pool, stride=config.depth_pool, ceil_mode=True)
        self.depth_output_norm = torch.nn.LayerNorm(config.d_model)
        self.wavelength_queries = torch.nn.Parameter(torch.empty(config.spectrum_width, config.d_model))
        self.decoder_blocks = torch.nn.ModuleList(
            [
                _WavelengthDecoderBlock(
                    config.d_model,
                    config.decoder_heads,
                    config.ffn_multiplier,
                    config.dropout,
                )
                for _ in range(config.decoder_blocks)
            ]
        )
        self.output_norm = torch.nn.LayerNorm(config.d_model)
        self.output = torch.nn.Linear(config.d_model, 3)
        torch.nn.init.normal_(self.wavelength_queries, mean=0.0, std=0.02)

    def _probabilities(self, fields: torch.Tensor) -> torch.Tensor:
        if fields.dim() == 2:
            ids = fields.long().clamp(0, self.config.num_materials - 1)
            return functional.one_hot(ids, num_classes=self.config.num_materials).to(torch.float32)
        if fields.dim() != 3 or fields.size(-1) != self.config.num_materials:
            raise ValueError(
                f"fields must have shape [B,D] or [B,D,{self.config.num_materials}], got {tuple(fields.shape)}"
            )
        return fields.float()

    def _depth_position_features(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        maximum = max(float(self.config.depth_bins) * float(self.config.dz_nm), 1.0)
        normalized = self.depth_positions.to(device=device, dtype=torch.float32) / maximum
        frequencies = torch.arange(1, 9, device=device, dtype=torch.float32)
        angles = normalized.unsqueeze(-1) * frequencies.unsqueeze(0) * math.pi
        features = torch.cat([normalized.unsqueeze(-1), angles.sin(), angles.cos()], dim=-1)
        return self.depth_position_projection(features.to(dtype=dtype))

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        """Predict conserved R/A/T spectra from hard or soft depth fields."""
        probabilities = self._probabilities(fields)
        if probabilities.size(1) != self.config.depth_bins:
            raise ValueError(f"Expected {self.config.depth_bins} depth bins, got {probabilities.size(1)}")
        packed, active = compact_depth_field_probabilities(probabilities, self.config.void_id)
        hidden = self.material_projection(packed) + self.active_projection(active.unsqueeze(-1))
        hidden = hidden + self._depth_position_features(device=hidden.device, dtype=hidden.dtype).unsqueeze(0)
        hidden = hidden.transpose(1, 2).contiguous()
        for block in self.depth_blocks:
            hidden = block(hidden)
        memory = self.depth_pool(hidden).transpose(1, 2).contiguous()
        memory = self.depth_output_norm(memory)

        queries = self.wavelength_queries.to(dtype=memory.dtype).unsqueeze(0).expand(memory.size(0), -1, -1)
        for block in self.decoder_blocks:
            queries = block(queries, memory)
        rat_logits = self.output(self.output_norm(queries)).float()
        return functional.softmax(rat_logits, dim=-1).transpose(1, 2).contiguous()


def depth_field_spectrum_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    channels: tuple[int, ...] = (0, 2),
    derivative_weight: float = 0.25,
    huber_delta: float = 0.02,
) -> dict[str, torch.Tensor]:
    """Return level and wavelength-derivative normalized Huber losses."""
    if predicted.shape != target.shape or predicted.dim() != 3:
        raise ValueError(f"predicted and target must share [B,3,W], got {predicted.shape} and {target.shape}")
    if not channels or any(index < 0 or index >= predicted.size(1) for index in channels):
        raise ValueError(f"Invalid spectral channels {channels}")
    if not math.isfinite(float(huber_delta)) or float(huber_delta) <= 0.0:
        raise ValueError(f"huber_delta must be finite and positive, got {huber_delta}")
    selected_predicted = predicted[:, channels].float()
    selected_target = target[:, channels].float()
    level = functional.smooth_l1_loss(selected_predicted, selected_target, beta=float(huber_delta))
    derivative = functional.smooth_l1_loss(
        selected_predicted.diff(dim=-1),
        selected_target.diff(dim=-1),
        beta=float(huber_delta),
    )
    total = level + float(derivative_weight) * derivative
    return {"loss": total, "level_loss": level, "derivative_loss": derivative}


def load_depth_field_spectrum_surrogate(
    checkpoint: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[DepthFieldSpectrumSurrogate, dict[str, Any]]:
    """Load a surrogate and its metadata from an OptoLlama checkpoint."""
    blob = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    extra = blob.get("extra") or {}
    config_data = extra.get("surrogate_config")
    if not isinstance(config_data, dict):
        raise RuntimeError(f"Surrogate checkpoint {checkpoint} is missing extra['surrogate_config'].")
    model = DepthFieldSpectrumSurrogate(DepthFieldSpectrumSurrogateConfig.from_dict(config_data))
    model.load_state_dict(blob["model_state"], strict=True)
    model.to(device)
    return model, extra
