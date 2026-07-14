from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .optollama import (
    AdaLayerNormGaussian,
    PositionalEncoding as OptoLlamaPositionalEncoding,
    SpectrumEmbedding as OptoLlamaSpectrumEmbedding,
    TimestepEmbedding as OptoLlamaTimestepEmbedding,
)


def _group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Embed diffusion timesteps."""
        timesteps = timesteps.float()
        half = max(self.dim // 2, 1)
        if half == 1:
            freqs = torch.ones(1, device=timesteps.device, dtype=torch.float32)
        else:
            scale = -math.log(10_000.0) / float(half - 1)
            freqs = torch.exp(torch.arange(half, device=timesteps.device, dtype=torch.float32) * scale)
        args = timesteps[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.size(-1) < self.dim:
            emb = functional.pad(emb, (0, self.dim - emb.size(-1)))
        return emb[:, : self.dim]


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise local filtering followed by pointwise channel mixing."""

    def __init__(self, channels: int, *, kernel_size: int, padding: int, dilation: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply depthwise-separable convolution."""
        return self.pointwise(self.depthwise(x))


def _normalize_conv_type(conv_type: str | None) -> str:
    value = str(conv_type or "full").lower().replace("-", "_")
    aliases = {
        "full": "full",
        "standard": "full",
        "conv": "full",
        "separable": "separable",
        "depthwise": "separable",
        "depthwise_separable": "separable",
    }
    if value not in aliases:
        raise ValueError(f"Unknown depth-field conv_type={conv_type!r}; expected 'full' or 'separable'.")
    return aliases[value]


def _normalize_model_type(model_type: str | None) -> str:
    value = str(model_type or "conv").lower().replace("-", "_")
    aliases = {
        "conv": "conv",
        "convolution": "conv",
        "cnn": "conv",
        "dilated_conv": "conv",
        "attention": "attention",
        "attn": "attention",
        "mha": "attention",
        "multihead_attention": "attention",
        "multi_head_attention": "attention",
        "transformer": "attention",
        "optollama_depth": "optollama_depth",
        "optollama_depth_field": "optollama_depth",
        "opto_depth": "optollama_depth",
        "opto_depth_field": "optollama_depth",
        "dit_depth": "optollama_depth",
        "optollama_depth_windowed": "optollama_depth_windowed",
        "optollama_windowed_depth": "optollama_depth_windowed",
        "optollama_depth_patched": "optollama_depth_windowed",
        "windowed_optollama_depth": "optollama_depth_windowed",
        "windowed_depth": "optollama_depth_windowed",
    }
    if value not in aliases:
        raise ValueError(
            f"Unknown depth-field model_type={model_type!r}; expected 'conv', 'attention', "
            "'optollama_depth', or 'optollama_depth_windowed'."
        )
    return aliases[value]


def _make_depth_conv(channels: int, *, kernel_size: int, padding: int, dilation: int, conv_type: str) -> nn.Module:
    if _normalize_conv_type(conv_type) == "separable":
        return DepthwiseSeparableConv1d(channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
    return nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)


def _match_depth_length(x: torch.Tensor, target_length: int) -> torch.Tensor:
    current_length = int(x.size(-1))
    target_length = int(target_length)
    if current_length == target_length:
        return x
    if current_length > target_length:
        offset = (current_length - target_length) // 2
        return x[..., offset : offset + target_length]
    missing = target_length - current_length
    left = missing // 2
    right = missing - left
    return functional.pad(x, (left, right))


class DepthFieldBlock(nn.Module):
    """Denoising residual block for a depth-field sequence."""

    def __init__(self, channels: int, *, kernel_size: int, dilation: int, dropout: float, conv_type: str = "full") -> None:
        super().__init__()
        padding = int(dilation) * (int(kernel_size) // 2)
        groups = _group_count(channels)
        conv_type = _normalize_conv_type(conv_type)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = _make_depth_conv(channels, kernel_size=kernel_size, padding=padding, dilation=dilation, conv_type=conv_type)
        self.cond_proj = nn.Linear(channels, channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.dropout = nn.Dropout(float(dropout))
        self.conv2 = _make_depth_conv(channels, kernel_size=kernel_size, padding=padding, dilation=dilation, conv_type=conv_type)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply one conditioned residual denoising block."""
        target_length = int(x.size(-1))
        h = self.conv1(functional.silu(self.norm1(x)))
        h = _match_depth_length(h, target_length)
        h = h + self.cond_proj(cond).unsqueeze(-1)
        h = self.conv2(self.dropout(functional.silu(self.norm2(h))))
        h = _match_depth_length(h, target_length)
        return x + h


class DepthFieldAttentionBlock(nn.Module):
    """Conditioned self-attention block for a complete depth-field sequence."""

    def __init__(self, channels: int, *, n_heads: int, dropout: float, ffn_multiplier: float) -> None:
        super().__init__()
        channels = int(channels)
        n_heads = int(n_heads)
        if n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {n_heads}")
        if channels % n_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by n_heads={n_heads}.")
        hidden = max(channels, int(round(channels * float(ffn_multiplier))))
        self.cond_proj = nn.Linear(channels, channels)
        self.norm1 = nn.LayerNorm(channels)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=n_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, channels),
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply global self-attention and position-wise feed-forward mixing."""
        h = self.norm1(x + self.cond_proj(cond).unsqueeze(1))
        attended, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + self.dropout(attended)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class DepthFieldOptoLlamaBlock(nn.Module):
    """OptoLlama-style DiT block adapted to dense depth-field tokens."""

    def __init__(self, channels: int, *, n_heads: int, dropout: float, ffn_multiplier: float) -> None:
        super().__init__()
        channels = int(channels)
        n_heads = int(n_heads)
        if n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {n_heads}")
        if channels % n_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by n_heads={n_heads}.")
        hidden = max(channels, int(round(channels * float(ffn_multiplier))))

        self.cross_attn = nn.MultiheadAttention(embed_dim=channels, num_heads=n_heads, dropout=float(dropout), batch_first=True)
        self.self_attn = nn.MultiheadAttention(embed_dim=channels, num_heads=n_heads, dropout=float(dropout), batch_first=True)
        self.ff1 = nn.Linear(channels, hidden)
        self.ff2 = nn.Linear(hidden, channels)

        self.norm1 = AdaLayerNormGaussian(channels, channels)
        self.norm2 = AdaLayerNormGaussian(channels, channels)
        self.norm3 = AdaLayerNormGaussian(channels, channels)

        self.to_alpha1 = nn.Linear(channels, channels)
        self.to_alpha2 = nn.Linear(channels, channels)
        nn.init.normal_(self.to_alpha1.weight, 0.0, 8e-4)
        nn.init.normal_(self.to_alpha2.weight, 0.0, 8e-4)
        nn.init.zeros_(self.to_alpha1.bias)
        nn.init.zeros_(self.to_alpha2.bias)

        self.dropout = nn.Dropout(float(dropout))

    def forward(self, depth_tokens: torch.Tensor, spectrum_tokens: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply spectrum cross-attention, depth self-attention, and FFN."""
        residual = depth_tokens
        h = self.norm1(depth_tokens, cond)
        h, _ = self.cross_attn(query=h, key=spectrum_tokens, value=spectrum_tokens, need_weights=False)
        h = h * (1.0 + self.to_alpha1(cond).unsqueeze(1))
        depth_tokens = residual + h

        residual = depth_tokens
        h = self.norm2(depth_tokens, cond)
        h, _ = self.self_attn(query=h, key=h, value=h, need_weights=False)
        h = h * (1.0 + self.to_alpha2(cond).unsqueeze(1))
        depth_tokens = residual + h

        residual = depth_tokens
        h = self.norm3(depth_tokens, cond)
        h = self.ff2(functional.silu(self.ff1(h)))
        return residual + self.dropout(h)


class WindowedSpectrumEmbedding(nn.Module):
    """Project overlapping multi-channel wavelength windows into spectrum tokens."""

    def __init__(
        self,
        input_channels: int,
        input_width: int,
        d_model: int,
        *,
        patch_size: int,
        patch_stride: int,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.input_width = int(input_width)
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)

        if self.input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {input_channels}.")
        if self.input_width <= 0:
            raise ValueError(f"input_width must be positive, got {input_width}.")
        if self.patch_size <= 0 or self.patch_size > self.input_width:
            raise ValueError(
                f"spectrum_patch_size must be in [1,{self.input_width}], got {self.patch_size}."
            )
        if self.patch_stride <= 0:
            raise ValueError(f"spectrum_patch_stride must be positive, got {self.patch_stride}.")

        remainder = (self.input_width - self.patch_size) % self.patch_stride
        self.right_padding = (self.patch_stride - remainder) % self.patch_stride
        padded_width = self.input_width + self.right_padding
        self.num_patches = 1 + (padded_width - self.patch_size) // self.patch_stride

        self.projection = nn.Conv1d(
            self.input_channels,
            int(d_model),
            kernel_size=self.patch_size,
            stride=self.patch_stride,
        )
        self.pre_norm = nn.LayerNorm(int(d_model))
        self.mlp = nn.Sequential(
            nn.Linear(int(d_model), int(d_model)),
            nn.SiLU(),
            nn.Linear(int(d_model), int(d_model)),
        )
        self.output_norm = nn.LayerNorm(int(d_model))

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        """Return wavelength-window tokens with shape ``[B,num_patches,d_model]``."""
        if spectra.dim() < 2:
            raise ValueError(f"spectra must include batch and wavelength dimensions, got {tuple(spectra.shape)}")
        spectra = spectra.reshape(spectra.size(0), -1, spectra.size(-1)).float()
        if spectra.shape[1:] != (self.input_channels, self.input_width):
            raise ValueError(
                f"Expected flattened spectra shape [B,{self.input_channels},{self.input_width}], "
                f"got {tuple(spectra.shape)}"
            )
        if self.right_padding:
            spectra = functional.pad(spectra, (0, self.right_padding), mode="replicate")
        tokens = self.projection(spectra).transpose(1, 2).contiguous()
        tokens = tokens + self.mlp(self.pre_norm(tokens))
        return self.output_norm(tokens)


class SpectrumEncoderBlock(nn.Module):
    """Mix wavelength-window tokens before they condition the depth sequence."""

    def __init__(self, d_model: int, *, n_heads: int, dropout: float, ffn_multiplier: float) -> None:
        super().__init__()
        hidden = max(int(d_model), int(round(int(d_model) * float(ffn_multiplier))))
        self.norm1 = nn.LayerNorm(int(d_model))
        self.self_attn = nn.MultiheadAttention(
            embed_dim=int(d_model),
            num_heads=int(n_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(int(d_model))
        self.ff = nn.Sequential(
            nn.Linear(int(d_model), hidden),
            nn.SiLU(),
            nn.Linear(hidden, int(d_model)),
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply spectrum self-attention and a position-wise feed-forward layer."""
        h = self.norm1(tokens)
        attended, _ = self.self_attn(h, h, h, need_weights=False)
        tokens = tokens + self.dropout(attended)
        return tokens + self.dropout(self.ff(self.norm2(tokens)))


@dataclass(frozen=True)
class DepthFieldModelConfig:
    """Serializable constructor config for depth-field diffusion."""

    spectrum_shape: tuple[int, ...]
    num_materials: int
    depth_bins: int
    model_type: str = "conv"
    d_model: int = 192
    n_blocks: int = 12
    kernel_size: int = 7
    n_heads: int = 8
    ffn_multiplier: float = 4.0
    timesteps: int = 100
    dropout: float = 0.0
    conv_type: str = "full"
    spectrum_patch_size: int = 8
    spectrum_patch_stride: int = 4
    spectrum_encoder_blocks: int = 2
    spectrum_encoder_heads: int = 8
    spectrum_ffn_multiplier: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_type", _normalize_model_type(self.model_type))
        object.__setattr__(self, "conv_type", _normalize_conv_type(self.conv_type))
        object.__setattr__(self, "n_heads", int(self.n_heads))
        object.__setattr__(self, "ffn_multiplier", float(self.ffn_multiplier))
        object.__setattr__(self, "spectrum_patch_size", int(self.spectrum_patch_size))
        object.__setattr__(self, "spectrum_patch_stride", int(self.spectrum_patch_stride))
        object.__setattr__(self, "spectrum_encoder_blocks", int(self.spectrum_encoder_blocks))
        object.__setattr__(self, "spectrum_encoder_heads", int(self.spectrum_encoder_heads))
        object.__setattr__(self, "spectrum_ffn_multiplier", float(self.spectrum_ffn_multiplier))
        if int(self.n_heads) <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}.")
        if float(self.ffn_multiplier) <= 0.0:
            raise ValueError(f"ffn_multiplier must be positive, got {self.ffn_multiplier}.")
        attention_models = {"attention", "optollama_depth", "optollama_depth_windowed"}
        if self.model_type in attention_models and int(self.d_model) % int(self.n_heads) != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}.")
        if int(self.spectrum_patch_size) <= 0:
            raise ValueError(f"spectrum_patch_size must be positive, got {self.spectrum_patch_size}.")
        if int(self.spectrum_patch_stride) <= 0:
            raise ValueError(f"spectrum_patch_stride must be positive, got {self.spectrum_patch_stride}.")
        if int(self.spectrum_encoder_blocks) < 0:
            raise ValueError(f"spectrum_encoder_blocks must be non-negative, got {self.spectrum_encoder_blocks}.")
        if int(self.spectrum_encoder_heads) <= 0:
            raise ValueError(f"spectrum_encoder_heads must be positive, got {self.spectrum_encoder_heads}.")
        if float(self.spectrum_ffn_multiplier) <= 0.0:
            raise ValueError(f"spectrum_ffn_multiplier must be positive, got {self.spectrum_ffn_multiplier}.")
        if self.model_type == "optollama_depth_windowed" and int(self.d_model) % int(self.spectrum_encoder_heads) != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by spectrum_encoder_heads={self.spectrum_encoder_heads}."
            )

    def to_dict(self) -> dict:
        """Return a JSON/checkpoint friendly constructor mapping."""
        return {
            "spectrum_shape": list(self.spectrum_shape),
            "num_materials": int(self.num_materials),
            "depth_bins": int(self.depth_bins),
            "model_type": _normalize_model_type(self.model_type),
            "d_model": int(self.d_model),
            "n_blocks": int(self.n_blocks),
            "kernel_size": int(self.kernel_size),
            "n_heads": int(self.n_heads),
            "ffn_multiplier": float(self.ffn_multiplier),
            "timesteps": int(self.timesteps),
            "dropout": float(self.dropout),
            "conv_type": _normalize_conv_type(self.conv_type),
            "spectrum_patch_size": int(self.spectrum_patch_size),
            "spectrum_patch_stride": int(self.spectrum_patch_stride),
            "spectrum_encoder_blocks": int(self.spectrum_encoder_blocks),
            "spectrum_encoder_heads": int(self.spectrum_encoder_heads),
            "spectrum_ffn_multiplier": float(self.spectrum_ffn_multiplier),
        }

    @staticmethod
    def from_dict(data: dict) -> "DepthFieldModelConfig":
        """Build a model config from a checkpoint mapping."""
        return DepthFieldModelConfig(
            spectrum_shape=tuple(int(v) for v in data["spectrum_shape"]),
            num_materials=int(data["num_materials"]),
            depth_bins=int(data["depth_bins"]),
            model_type=_normalize_model_type(data.get("model_type", data.get("type", "conv"))),
            d_model=int(data.get("d_model", 192)),
            n_blocks=int(data.get("n_blocks", 12)),
            kernel_size=int(data.get("kernel_size", 7)),
            n_heads=int(data.get("n_heads", 8)),
            ffn_multiplier=float(data.get("ffn_multiplier", 4.0)),
            timesteps=int(data.get("timesteps", 100)),
            dropout=float(data.get("dropout", 0.0)),
            conv_type=_normalize_conv_type(data.get("conv_type", "full")),
            spectrum_patch_size=int(data.get("spectrum_patch_size", 8)),
            spectrum_patch_stride=int(data.get("spectrum_patch_stride", 4)),
            spectrum_encoder_blocks=int(data.get("spectrum_encoder_blocks", 2)),
            spectrum_encoder_heads=int(data.get("spectrum_encoder_heads", data.get("n_heads", 8))),
            spectrum_ffn_multiplier=float(data.get("spectrum_ffn_multiplier", 2.0)),
        )


class DepthFieldDiffusion(nn.Module):
    """
    Conditional diffusion over material occupancy on a fixed depth grid.

    Clean labels are material ids plus a void class. The diffusion input has
    one additional mask id used only for corrupted/unknown depth bins.
    """

    def __init__(self, config: DepthFieldModelConfig) -> None:
        super().__init__()
        self.config = config
        self.spectrum_shape = tuple(int(v) for v in config.spectrum_shape)
        self.num_materials = int(config.num_materials)
        self.depth_bins = int(config.depth_bins)
        self.d_model = int(config.d_model)
        self.timesteps = int(config.timesteps)
        self.mask_id = self.num_materials
        self.conv_type = _normalize_conv_type(config.conv_type)

        spectrum_dim = int(math.prod(self.spectrum_shape))
        self.spectrum_encoder = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(spectrum_dim),
            nn.Linear(spectrum_dim, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.time_encoder = nn.Sequential(
            SinusoidalTimeEmbedding(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.input_embedding = nn.Embedding(self.num_materials + 1, self.d_model)
        self.position_embedding = nn.Parameter(torch.empty(1, self.depth_bins, self.d_model))

        dilation_cycle = (1, 2, 4, 8, 16, 32)
        self.blocks = nn.ModuleList(
            [
                DepthFieldBlock(
                    self.d_model,
                    kernel_size=int(config.kernel_size),
                    dilation=dilation_cycle[idx % len(dilation_cycle)],
                    dropout=float(config.dropout),
                    conv_type=self.conv_type,
                )
                for idx in range(int(config.n_blocks))
            ]
        )
        self.output_norm = nn.GroupNorm(_group_count(self.d_model), self.d_model)
        self.output = nn.Conv1d(self.d_model, self.num_materials, kernel_size=1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize learned positional embeddings."""
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def noise_probability(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Return the mask/replacement probability for each timestep."""
        if self.timesteps <= 1:
            return torch.ones_like(timesteps, dtype=torch.float32)
        x = (timesteps.float() + 1.0) / float(self.timesteps)
        return x.square().clamp(0.0, 1.0)

    def corrupt(
        self,
        clean_fields: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        random_replace_prob: float = 0.10,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Corrupt clean material fields with mask and random replacement noise."""
        if clean_fields.dim() != 2:
            raise ValueError(f"clean_fields must have shape [B,D], got {tuple(clean_fields.shape)}")
        if clean_fields.size(1) != self.depth_bins:
            raise ValueError(f"Expected {self.depth_bins} depth bins, got {clean_fields.size(1)}")

        clean_fields = clean_fields.long().clamp(0, self.num_materials - 1)
        noise_prob = self.noise_probability(timesteps).to(device=clean_fields.device).unsqueeze(1)
        replace_fraction = float(max(0.0, min(1.0, random_replace_prob)))
        replace_prob = noise_prob * replace_fraction
        uniform = torch.rand(clean_fields.shape, device=clean_fields.device, generator=generator)
        replace_mask = uniform < replace_prob
        mask_mask = (uniform >= replace_prob) & (uniform < noise_prob)

        random_labels = torch.randint(
            low=0,
            high=self.num_materials,
            size=clean_fields.shape,
            device=clean_fields.device,
            generator=generator,
        )
        noised = clean_fields.clone()
        noised[replace_mask] = random_labels[replace_mask]
        noised[mask_mask] = int(self.mask_id)
        return noised, replace_mask | mask_mask

    def forward(self, spectra: torch.Tensor, noised_fields: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Predict clean material logits for every depth bin."""
        if spectra.shape[1:] != self.spectrum_shape:
            raise ValueError(f"Expected spectra shape [B,{self.spectrum_shape}], got {tuple(spectra.shape)}")
        if noised_fields.shape != (spectra.size(0), self.depth_bins):
            raise ValueError(
                f"Expected noised_fields shape [{spectra.size(0)},{self.depth_bins}], got {tuple(noised_fields.shape)}"
            )

        cond = self.spectrum_encoder(spectra.float()) + self.time_encoder(timesteps)
        x = self.input_embedding(noised_fields.long().clamp(0, self.mask_id))
        x = x + self.position_embedding
        x = x.transpose(1, 2).contiguous()
        for block in self.blocks:
            x = block(x, cond)
        logits = self.output(functional.silu(self.output_norm(x))).transpose(1, 2).contiguous()
        return logits

    def training_loss(
        self,
        spectra: torch.Tensor,
        clean_fields: torch.Tensor,
        *,
        void_id: int,
        void_loss_weight: float = 0.25,
        random_replace_prob: float = 0.10,
        loss_on_corrupted_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Return a denoising CE loss dictionary."""
        batch_size = int(clean_fields.size(0))
        timesteps = torch.randint(0, self.timesteps, (batch_size,), device=clean_fields.device)
        noised_fields, corrupted = self.corrupt(
            clean_fields,
            timesteps,
            random_replace_prob=random_replace_prob,
        )
        logits = self(spectra, noised_fields, timesteps)

        weights = torch.ones(self.num_materials, device=clean_fields.device, dtype=logits.dtype)
        if 0 <= int(void_id) < self.num_materials:
            weights[int(void_id)] = float(void_loss_weight)
        loss_per_bin = functional.cross_entropy(
            logits.view(-1, self.num_materials),
            clean_fields.long().view(-1),
            weight=weights,
            reduction="none",
        ).view_as(clean_fields)

        if loss_on_corrupted_only:
            denom = corrupted.float().sum().clamp_min(1.0)
            loss = (loss_per_bin * corrupted.float()).sum() / denom
        else:
            loss = loss_per_bin.mean()

        return {
            "loss": loss,
            "logits": logits,
            "timesteps": timesteps,
            "noised_fields": noised_fields,
            "corrupted": corrupted,
        }

    @staticmethod
    def _filter_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
        if top_k is None or int(top_k) <= 0 or int(top_k) >= logits.size(-1):
            return logits
        values, _ = torch.topk(logits, int(top_k), dim=-1)
        threshold = values[..., -1, None]
        return logits.masked_fill(logits < threshold, float("-inf"))

    def _sample_logits(
        self,
        logits: torch.Tensor,
        *,
        temperature: float,
        top_k: int,
        deterministic: bool,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if deterministic or temperature <= 0.0:
            probs = torch.softmax(logits, dim=-1)
            return logits.argmax(dim=-1), probs.max(dim=-1).values

        scaled = logits / max(float(temperature), 1.0e-6)
        scaled = self._filter_top_k(scaled, top_k)
        probs = torch.softmax(scaled, dim=-1)
        flat = probs.view(-1, probs.size(-1))
        sampled = torch.multinomial(flat, num_samples=1, generator=generator).view(logits.shape[:2])
        return sampled, probs.max(dim=-1).values

    @staticmethod
    def _normalize_remask_strategy(remask_strategy: str) -> str:
        strategy = str(remask_strategy or "confidence").lower().replace("-", "_")
        aliases = {
            "confidence": "confidence",
            "low_confidence": "confidence",
            "least_confidence": "confidence",
            "uncertain": "confidence",
            "random": "random",
            "bernoulli": "random",
        }
        if strategy not in aliases:
            raise ValueError(f"Unknown remask_strategy={remask_strategy!r}; expected 'confidence' or 'random'.")
        return aliases[strategy]

    @torch.no_grad()
    def sample(
        self,
        spectra: torch.Tensor,
        *,
        steps: int | None = None,
        temperature: float = 1.0,
        top_k: int = 0,
        deterministic: bool = False,
        remask_strategy: str = "confidence",
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Sample clean material fields from all-mask initialization.

        At each stage the model predicts a full clean field, then bins are
        re-masked according to the next timestep's noise level. Confidence
        remasking reopens the least confident bins; random remasking uses the
        original OptoLlama-style Bernoulli remask process.
        """
        self.eval()
        batch_size = int(spectra.size(0))
        device = spectra.device
        remask_strategy = self._normalize_remask_strategy(remask_strategy)
        fields = torch.full((batch_size, self.depth_bins), int(self.mask_id), dtype=torch.long, device=device)

        total_steps = int(steps or self.timesteps)
        if total_steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        step_values = torch.linspace(self.timesteps - 1, 0, total_steps, device=device).round().long()
        if step_values[-1].item() != 0:
            step_values = torch.cat([step_values, torch.zeros(1, device=device, dtype=torch.long)])

        for step_idx, timestep in enumerate(step_values):
            timesteps = torch.full((batch_size,), int(timestep.item()), dtype=torch.long, device=device)
            logits = self(spectra, fields, timesteps)
            pred, confidence = self._sample_logits(
                logits,
                temperature=temperature,
                top_k=top_k,
                deterministic=deterministic,
                generator=generator,
            )

            if step_idx == len(step_values) - 1:
                fields = pred
                break

            next_timestep = step_values[step_idx + 1].view(1)
            mask_fraction = float(self.noise_probability(next_timestep).item())
            mask_count = int(round(mask_fraction * float(self.depth_bins)))
            fields = pred
            if mask_count > 0:
                if remask_strategy == "random":
                    remask = torch.rand(pred.shape, device=device, generator=generator) < mask_fraction
                    fields = torch.where(remask, torch.full_like(fields, int(self.mask_id)), fields)
                else:
                    mask_count = min(mask_count, self.depth_bins)
                    low_confidence = torch.topk(confidence, k=mask_count, dim=1, largest=False).indices
                    fields.scatter_(1, low_confidence, int(self.mask_id))

        return fields


class DepthFieldAttentionDiffusion(DepthFieldDiffusion):
    """
    Conditional diffusion model with global multi-head self-attention.

    The denoising/sample API is inherited from :class:`DepthFieldDiffusion`,
    but every residual block attends over the full depth sequence at once.
    """

    def __init__(self, config: DepthFieldModelConfig) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.spectrum_shape = tuple(int(v) for v in config.spectrum_shape)
        self.num_materials = int(config.num_materials)
        self.depth_bins = int(config.depth_bins)
        self.d_model = int(config.d_model)
        self.timesteps = int(config.timesteps)
        self.mask_id = self.num_materials
        self.conv_type = _normalize_conv_type(config.conv_type)

        spectrum_dim = int(math.prod(self.spectrum_shape))
        self.spectrum_encoder = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(spectrum_dim),
            nn.Linear(spectrum_dim, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.time_encoder = nn.Sequential(
            SinusoidalTimeEmbedding(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.input_embedding = nn.Embedding(self.num_materials + 1, self.d_model)
        self.position_embedding = nn.Parameter(torch.empty(1, self.depth_bins, self.d_model))
        self.blocks = nn.ModuleList(
            [
                DepthFieldAttentionBlock(
                    self.d_model,
                    n_heads=int(config.n_heads),
                    dropout=float(config.dropout),
                    ffn_multiplier=float(config.ffn_multiplier),
                )
                for _ in range(int(config.n_blocks))
            ]
        )
        self.output_norm = nn.LayerNorm(self.d_model)
        self.output = nn.Linear(self.d_model, self.num_materials)
        self.reset_parameters()

    def forward(self, spectra: torch.Tensor, noised_fields: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Predict clean material logits for every depth bin with global attention."""
        if spectra.shape[1:] != self.spectrum_shape:
            raise ValueError(f"Expected spectra shape [B,{self.spectrum_shape}], got {tuple(spectra.shape)}")
        if noised_fields.shape != (spectra.size(0), self.depth_bins):
            raise ValueError(
                f"Expected noised_fields shape [{spectra.size(0)},{self.depth_bins}], got {tuple(noised_fields.shape)}"
            )

        cond = self.spectrum_encoder(spectra.float()) + self.time_encoder(timesteps)
        x = self.input_embedding(noised_fields.long().clamp(0, self.mask_id))
        x = x + self.position_embedding
        for block in self.blocks:
            x = block(x, cond)
        logits = self.output(functional.silu(self.output_norm(x)))
        return logits


class DepthFieldOptoLlamaDiffusion(DepthFieldDiffusion):
    """
    Depth-field diffusion with the OptoLlama transformer block structure.

    The representation remains a dense depth grid, but each block follows the
    OptoLlama pattern: spectrum cross-attention, depth self-attention,
    AdaLayerNorm timestep conditioning, alpha gates, and a 4x-style FFN.
    """

    def __init__(self, config: DepthFieldModelConfig) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.spectrum_shape = tuple(int(v) for v in config.spectrum_shape)
        self.num_materials = int(config.num_materials)
        self.depth_bins = int(config.depth_bins)
        self.d_model = int(config.d_model)
        self.timesteps = int(config.timesteps)
        self.mask_id = self.num_materials
        self.conv_type = _normalize_conv_type(config.conv_type)

        if len(self.spectrum_shape) < 1:
            raise ValueError(f"spectrum_shape must include a wavelength dimension, got {self.spectrum_shape}")
        spectrum_width = int(self.spectrum_shape[-1])
        spectral_tokens = int(math.prod(self.spectrum_shape[:-1])) if len(self.spectrum_shape) > 1 else 1
        max_position_len = max(self.depth_bins, spectral_tokens, 1)

        self.spectrum_embedding = OptoLlamaSpectrumEmbedding(spectrum_width, self.d_model)
        self.input_embedding = nn.Embedding(self.num_materials + 1, self.d_model)
        self.time_embedding = OptoLlamaTimestepEmbedding(self.d_model)
        self.positional_encoding = OptoLlamaPositionalEncoding(max_position_len, self.d_model)
        self.blocks = nn.ModuleList(
            [
                DepthFieldOptoLlamaBlock(
                    self.d_model,
                    n_heads=int(config.n_heads),
                    dropout=float(config.dropout),
                    ffn_multiplier=float(config.ffn_multiplier),
                )
                for _ in range(int(config.n_blocks))
            ]
        )
        self.output = nn.Linear(self.d_model, self.num_materials)

    def reset_parameters(self) -> None:
        """Keep API compatibility with the other depth-field backbones."""
        return None

    def _spectrum_tokens(self, spectra: torch.Tensor) -> torch.Tensor:
        embedded = self.spectrum_embedding(spectra.float())
        if embedded.dim() == 2:
            embedded = embedded.unsqueeze(1)
        elif embedded.dim() > 3:
            embedded = embedded.reshape(embedded.size(0), -1, embedded.size(-1))
        return embedded + self.positional_encoding(embedded).to(dtype=embedded.dtype)

    def forward(self, spectra: torch.Tensor, noised_fields: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Predict clean material logits with OptoLlama-style conditioning."""
        if spectra.shape[1:] != self.spectrum_shape:
            raise ValueError(f"Expected spectra shape [B,{self.spectrum_shape}], got {tuple(spectra.shape)}")
        if noised_fields.shape != (spectra.size(0), self.depth_bins):
            raise ValueError(
                f"Expected noised_fields shape [{spectra.size(0)},{self.depth_bins}], got {tuple(noised_fields.shape)}"
            )

        spectrum_tokens = self._spectrum_tokens(spectra)
        depth_tokens = self.input_embedding(noised_fields.long().clamp(0, self.mask_id))
        depth_tokens = depth_tokens + self.positional_encoding(depth_tokens).to(dtype=depth_tokens.dtype)
        time_token = self.time_embedding(timesteps)
        depth_tokens = depth_tokens + time_token.to(dtype=depth_tokens.dtype)
        cond = time_token.squeeze(1).to(dtype=depth_tokens.dtype)

        for block in self.blocks:
            depth_tokens = block(depth_tokens, spectrum_tokens, cond)
        return self.output(depth_tokens)


class DepthFieldWindowedOptoLlamaDiffusion(DepthFieldOptoLlamaDiffusion):
    """OptoLlama depth-field model conditioned by overlapping wavelength windows."""

    def __init__(self, config: DepthFieldModelConfig) -> None:
        super().__init__(config)
        spectrum_width = int(self.spectrum_shape[-1])
        spectrum_channels = int(math.prod(self.spectrum_shape[:-1])) if len(self.spectrum_shape) > 1 else 1
        self.spectrum_embedding = WindowedSpectrumEmbedding(
            spectrum_channels,
            spectrum_width,
            self.d_model,
            patch_size=int(config.spectrum_patch_size),
            patch_stride=int(config.spectrum_patch_stride),
        )
        self.spectrum_positional_encoding = OptoLlamaPositionalEncoding(
            self.spectrum_embedding.num_patches,
            self.d_model,
        )
        self.spectrum_blocks = nn.ModuleList(
            [
                SpectrumEncoderBlock(
                    self.d_model,
                    n_heads=int(config.spectrum_encoder_heads),
                    dropout=float(config.dropout),
                    ffn_multiplier=float(config.spectrum_ffn_multiplier),
                )
                for _ in range(int(config.spectrum_encoder_blocks))
            ]
        )
        self.spectrum_output_norm = nn.LayerNorm(self.d_model)

    def _spectrum_tokens(self, spectra: torch.Tensor) -> torch.Tensor:
        tokens = self.spectrum_embedding(spectra.float())
        tokens = tokens + self.spectrum_positional_encoding(tokens).to(dtype=tokens.dtype)
        for block in self.spectrum_blocks:
            tokens = block(tokens)
        return self.spectrum_output_norm(tokens)


def build_depth_field_model(config: DepthFieldModelConfig) -> DepthFieldDiffusion:
    """Build the configured depth-field diffusion model."""
    model_type = _normalize_model_type(config.model_type)
    if model_type == "optollama_depth_windowed":
        return DepthFieldWindowedOptoLlamaDiffusion(config)
    if model_type == "optollama_depth":
        return DepthFieldOptoLlamaDiffusion(config)
    if model_type == "attention":
        return DepthFieldAttentionDiffusion(config)
    if model_type == "conv":
        return DepthFieldDiffusion(config)
    raise ValueError(f"Unsupported depth-field model_type={config.model_type!r}.")
