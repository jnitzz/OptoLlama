from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as functional


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


class DepthFieldBlock(nn.Module):
    """Denoising residual block for a depth-field sequence."""

    def __init__(self, channels: int, *, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = int(dilation) * (int(kernel_size) // 2)
        groups = _group_count(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.cond_proj = nn.Linear(channels, channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.dropout = nn.Dropout(float(dropout))
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply one conditioned residual denoising block."""
        h = self.conv1(functional.silu(self.norm1(x)))
        h = h + self.cond_proj(cond).unsqueeze(-1)
        h = self.conv2(self.dropout(functional.silu(self.norm2(h))))
        return x + h


@dataclass(frozen=True)
class DepthFieldModelConfig:
    """Serializable constructor config for depth-field diffusion."""

    spectrum_shape: tuple[int, ...]
    num_materials: int
    depth_bins: int
    d_model: int = 192
    n_blocks: int = 12
    kernel_size: int = 7
    timesteps: int = 100
    dropout: float = 0.0

    def to_dict(self) -> dict:
        """Return a JSON/checkpoint friendly constructor mapping."""
        return {
            "spectrum_shape": list(self.spectrum_shape),
            "num_materials": int(self.num_materials),
            "depth_bins": int(self.depth_bins),
            "d_model": int(self.d_model),
            "n_blocks": int(self.n_blocks),
            "kernel_size": int(self.kernel_size),
            "timesteps": int(self.timesteps),
            "dropout": float(self.dropout),
        }

    @staticmethod
    def from_dict(data: dict) -> "DepthFieldModelConfig":
        """Build a model config from a checkpoint mapping."""
        return DepthFieldModelConfig(
            spectrum_shape=tuple(int(v) for v in data["spectrum_shape"]),
            num_materials=int(data["num_materials"]),
            depth_bins=int(data["depth_bins"]),
            d_model=int(data.get("d_model", 192)),
            n_blocks=int(data.get("n_blocks", 12)),
            kernel_size=int(data.get("kernel_size", 7)),
            timesteps=int(data.get("timesteps", 100)),
            dropout=float(data.get("dropout", 0.0)),
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
