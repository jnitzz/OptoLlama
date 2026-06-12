import math
from typing import Optional

import torch
import torch.nn.functional as F

import optollama.data
import optollama.evaluation
import optollama.model
from optollama.evaluation.simulation import TMMContext

# ruff: noqa: D102, D105, D107


class SquareNoise(torch.nn.Module):
    """
    Noise schedule for discrete diffusion.

    This module squares normalized timesteps (t ∈ [0,1]) to obtain a monotonic
    noise level β(t) used during masking / remasking. A small epsilon offset
    prevents degenerate zero noise.

    Args
    ----
    eps : float
        Minimum noise level added to the schedule.
    """

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # Normalize timesteps to [0, 1] if they aren't already
        t = timesteps.clamp(0, 1)
        return (1.0 - self.eps) * t**2


class PositionalEncoding(torch.nn.Module):
    """
    Classic sinusoidal positional encoding.

    Creates a matrix of shape ``[max_len, d_model]`` containing deterministic
    sin/cos positional features. Returned encodings are sliced to match the
    sequence length of the input.

    Args
    ----
    max_len : int
        Maximum supported sequence length.
    d_model : int
        Embedding dimensionality.
    """

    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()

        # create position encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # combine the position and div_term to create the encoding
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # register the pe as a buffer
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[: x.shape[1]]


def _token_metadata(
    idx_to_token: dict[int, str],
    vocab_size: int,
    pad_idx: int,
    eos_idx: int,
    mask_idx: int,
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build material/thickness metadata for joint material-thickness tokens."""
    material_to_id: dict[str, int] = {}
    token_material_ids = torch.zeros(vocab_size, dtype=torch.long)
    token_thickness_nm = torch.zeros(vocab_size, dtype=torch.float32)
    layer_token_mask = torch.zeros(vocab_size, dtype=torch.bool)

    def material_id(name: str) -> int:
        if name not in material_to_id:
            material_to_id[name] = len(material_to_id)
        return material_to_id[name]

    special_ids = {
        int(pad_idx): optollama.data.PAD_TOKEN,
        int(eos_idx): optollama.data.EOS_TOKEN,
        int(mask_idx): optollama.data.MSK_TOKEN,
    }

    for token_id in range(vocab_size):
        token = idx_to_token.get(token_id, "")
        parts = optollama.data.layer_token_parts(token)
        if parts is None:
            material = special_ids.get(int(token_id), token or f"<UNK_{token_id}>")
            thickness = 0.0
            is_layer = False
        else:
            material, thickness = parts
            is_layer = True

        token_material_ids[token_id] = material_id(material)
        token_thickness_nm[token_id] = float(thickness)
        layer_token_mask[token_id] = is_layer

    material_names = [None] * len(material_to_id)
    for name, idx in material_to_id.items():
        material_names[idx] = name

    return material_names, token_material_ids, token_thickness_nm, layer_token_mask


class DepthPositionalEncoding(torch.nn.Module):
    """
    Additive positional encoding using cumulative physical stack depth.

    This is the diffusion-compatible version of PRISM's depth-position idea:
    positions are inferred from the current denoising state. Masked positions
    use a configurable fallback thickness, because their true thickness is
    unknown during sampling.
    """

    def __init__(
        self,
        max_len: int,
        d_model: int,
        token_thickness_nm: torch.Tensor,
        eos_idx: int,
        pad_idx: int,
        mask_idx: int,
        scale_nm: float = 1000.0,
        mask_thickness_nm: float | None = None,
        layer_centers: bool = True,
    ) -> None:
        super().__init__()
        self.eos_idx = int(eos_idx)
        self.pad_idx = int(pad_idx)
        self.mask_idx = int(mask_idx)
        self.scale_nm = max(float(scale_nm), 1e-6)
        self.layer_centers = bool(layer_centers)

        positive = token_thickness_nm[token_thickness_nm > 0]
        fallback = float(mask_thickness_nm) if mask_thickness_nm is not None else float(positive.median().item())
        thickness = token_thickness_nm.to(torch.float32).clone()
        thickness[self.mask_idx] = fallback
        thickness[self.pad_idx] = 0.0
        thickness[self.eos_idx] = 0.0
        self.register_buffer("token_thickness_nm", thickness)

        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        self.register_buffer("div_term", div_term)

        self.max_len = int(max_len)
        self.d_model = int(d_model)

    def _encode_thickness(self, thickness: torch.Tensor) -> torch.Tensor:
        """Encode an already-masked physical thickness sequence."""
        positions = thickness.cumsum(dim=1)
        if self.layer_centers:
            positions = positions - 0.5 * thickness
        positions = positions / self.scale_nm

        angles = positions.unsqueeze(-1) * self.div_term.to(device=positions.device, dtype=positions.dtype)
        pe = torch.zeros((*positions.shape, self.d_model), device=positions.device, dtype=positions.dtype)
        pe[..., 0::2] = torch.sin(angles)
        pe[..., 1::2] = torch.cos(angles[..., : pe[..., 1::2].shape[-1]])
        return pe

    def forward(self, stacks: torch.Tensor, thickness_nm: Optional[torch.Tensor] = None) -> torch.Tensor:
        token_ids = stacks.to(torch.long).clamp(0, self.token_thickness_nm.numel() - 1)
        if thickness_nm is None:
            thickness = self.token_thickness_nm[token_ids]
            is_eos = token_ids == self.eos_idx
            before_eos = is_eos.cumsum(dim=1) == 0
            valid = before_eos & (token_ids != self.pad_idx)
            thickness = thickness * valid.to(thickness.dtype)
        else:
            if thickness_nm.shape != token_ids.shape:
                raise ValueError(
                    "thickness_nm must have the same shape as stacks "
                    f"({tuple(token_ids.shape)}), got {tuple(thickness_nm.shape)}."
                )
            thickness = thickness_nm.to(device=token_ids.device, dtype=torch.float32)

        return self._encode_thickness(thickness)


class FactoredOutputHead(torch.nn.Module):
    """
    Predict material and continuous log-thickness separately, then project back
    to the existing joint token vocabulary for compatibility.
    """

    def __init__(
        self,
        d_model: int,
        idx_to_token: dict[int, str],
        vocab_size: int,
        pad_idx: int,
        eos_idx: int,
        mask_idx: int,
        thickness_log_sigma: float = 0.20,
    ) -> None:
        super().__init__()
        material_names, token_material_ids, token_thickness_nm, layer_token_mask = _token_metadata(
            idx_to_token,
            vocab_size,
            pad_idx,
            eos_idx,
            mask_idx,
        )
        self.material_names = material_names
        self.num_materials = len(material_names)
        self.thickness_log_sigma = max(float(thickness_log_sigma), 1e-4)

        self.register_buffer("token_material_ids", token_material_ids)
        self.register_buffer("token_thickness_nm", token_thickness_nm)
        self.register_buffer("token_log_thickness", torch.log(token_thickness_nm.clamp_min(1.0)))
        self.register_buffer("layer_token_mask", layer_token_mask)
        self.pad_idx = int(pad_idx)
        self.eos_idx = int(eos_idx)
        self.mask_idx = int(mask_idx)
        self.pad_material_id = int(token_material_ids[int(pad_idx)].item())
        self.eos_material_id = int(token_material_ids[int(eos_idx)].item())
        self.mask_material_id = int(token_material_ids[int(mask_idx)].item())
        material_layer_mask = torch.zeros(self.num_materials, dtype=torch.bool)
        if layer_token_mask.any():
            material_layer_mask[token_material_ids[layer_token_mask]] = True
        self.register_buffer("material_layer_mask", material_layer_mask)

        self.material_head = torch.nn.Linear(d_model, self.num_materials)
        self.log_thickness_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.SiLU(),
            torch.nn.Linear(d_model, self.num_materials),
        )

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        material_logits = self.material_head(hidden)  # [B,S,M]
        log_thickness = self.log_thickness_head(hidden)  # [B,S,M]

        token_material_ids = self.token_material_ids.to(device=hidden.device)
        token_log_thickness = self.token_log_thickness.to(device=hidden.device, dtype=hidden.dtype)
        layer_token_mask = self.layer_token_mask.to(device=hidden.device)

        material_joint = material_logits[..., token_material_ids]  # [B,S,V]
        pred_log_for_token = log_thickness[..., token_material_ids]  # [B,S,V]
        thickness_delta = (pred_log_for_token - token_log_thickness.view(1, 1, -1)) / self.thickness_log_sigma
        thickness_score = -(thickness_delta * thickness_delta)
        thickness_score = torch.where(layer_token_mask.view(1, 1, -1), thickness_score, torch.zeros_like(thickness_score))

        return {
            "joint_logits": material_joint + thickness_score,
            "material_logits": material_logits,
            "log_thickness": log_thickness,
        }

    def thickness_for_ids(
        self,
        outputs: dict[str, torch.Tensor],
        ids: torch.Tensor,
        min_nm: float,
        max_nm: float,
        round_step_nm: float | None = None,
    ) -> torch.Tensor:
        """
        Select the predicted continuous thickness for each sampled material.

        The sampled joint token still defines the hard material choice. The
        thickness comes from the material-specific regression head, so it can
        lie between token-bin values when rounding is disabled.
        """
        token_ids = ids.to(torch.long).clamp(0, self.token_material_ids.numel() - 1)
        token_material_ids = self.token_material_ids.to(device=token_ids.device)
        layer_token_mask = self.layer_token_mask.to(device=token_ids.device)
        material_ids = token_material_ids[token_ids]

        log_thickness = outputs["log_thickness"].gather(-1, material_ids.unsqueeze(-1)).squeeze(-1)
        thickness = torch.exp(log_thickness).to(torch.float32)
        thickness = thickness.clamp(float(min_nm), float(max_nm))

        if round_step_nm is not None:
            step = float(round_step_nm)
            if step > 0.0:
                thickness = torch.round(thickness / step) * step
                thickness = thickness.clamp(float(min_nm), float(max_nm))

        is_layer = layer_token_mask[token_ids]
        return thickness * is_layer.to(thickness.dtype)

    def thickness_for_material_ids(
        self,
        outputs: dict[str, torch.Tensor],
        material_ids: torch.Tensor,
        min_nm: float,
        max_nm: float,
        round_step_nm: float | None = None,
    ) -> torch.Tensor:
        """Select predicted continuous thickness for sampled material IDs."""
        mats = material_ids.to(torch.long).clamp(0, self.num_materials - 1)
        log_thickness = outputs["log_thickness"].gather(-1, mats.unsqueeze(-1)).squeeze(-1)
        thickness = torch.exp(log_thickness).to(torch.float32).clamp(float(min_nm), float(max_nm))
        if round_step_nm is not None:
            step = float(round_step_nm)
            if step > 0.0:
                thickness = torch.round(thickness / step) * step
                thickness = thickness.clamp(float(min_nm), float(max_nm))
        is_layer = self.material_layer_mask.to(device=mats.device)[mats]
        return thickness * is_layer.to(thickness.dtype)

    def tokens_for_material_thickness(
        self,
        material_ids: torch.Tensor,
        thickness_nm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert material IDs plus continuous thickness to nearest old tokens.

        This is only a compatibility/export bridge. The material-vocab model
        does not sample from the old joint material-thickness vocabulary.
        """
        mats = material_ids.to(torch.long).clamp(0, self.num_materials - 1)
        thickness = thickness_nm.to(device=mats.device, dtype=torch.float32)
        token_material_ids = self.token_material_ids.to(device=mats.device)
        token_thickness = self.token_thickness_nm.to(device=mats.device, dtype=torch.float32)
        layer_token_mask = self.layer_token_mask.to(device=mats.device)

        candidate = token_material_ids.view(1, 1, -1).eq(mats.unsqueeze(-1)) & layer_token_mask.view(1, 1, -1)
        distance = (token_thickness.view(1, 1, -1) - thickness.unsqueeze(-1)).abs()
        inf = torch.full_like(distance, float("inf"))
        nearest = torch.where(candidate, distance, inf).argmin(dim=-1)

        nearest = torch.where(mats == self.pad_material_id, torch.full_like(nearest, self.pad_idx), nearest)
        nearest = torch.where(mats == self.eos_material_id, torch.full_like(nearest, self.eos_idx), nearest)
        nearest = torch.where(mats == self.mask_material_id, torch.full_like(nearest, self.mask_idx), nearest)
        return nearest

    def loss(
        self,
        outputs: dict[str, torch.Tensor],
        stacks: torch.Tensor,
        thickness_weight: float,
        joint_ce_weight: float,
        label_smoothing: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        token_ids = stacks.to(torch.long).clamp(0, self.token_material_ids.numel() - 1)
        token_material_ids = self.token_material_ids.to(device=token_ids.device)
        target_materials = token_material_ids[token_ids]

        material_loss = F.cross_entropy(
            outputs["material_logits"].reshape(-1, self.num_materials),
            target_materials.reshape(-1),
            ignore_index=self.pad_material_id,
            label_smoothing=float(label_smoothing),
        )

        layer_mask = self.layer_token_mask.to(device=token_ids.device)[token_ids]
        if layer_mask.any():
            predicted = outputs["log_thickness"].gather(-1, target_materials.unsqueeze(-1)).squeeze(-1)
            target = self.token_log_thickness.to(device=token_ids.device, dtype=predicted.dtype)[token_ids]
            thickness_loss = F.mse_loss(predicted[layer_mask], target[layer_mask])
        else:
            thickness_loss = outputs["log_thickness"].sum() * 0.0

        joint_targets = token_ids.masked_fill(token_ids == self.pad_idx, -100)
        joint_loss = F.cross_entropy(
            outputs["joint_logits"].reshape(-1, self.token_material_ids.numel()),
            joint_targets.reshape(-1),
            ignore_index=-100,
        )

        loss = material_loss + float(thickness_weight) * thickness_loss + float(joint_ce_weight) * joint_loss
        parts = {
            "material_loss": material_loss.detach(),
            "thickness_loss": thickness_loss.detach(),
            "joint_loss": joint_loss.detach(),
        }
        return loss, parts


class SpectrumEmbedding(torch.nn.Module):
    """
    Embeds an input spectrum vector into the model's hidden dimension.

    Applies a small MLP + LayerNorm to project spectral inputs
    (e.g., RAT / reflectance curves) into ``d_model``.

    Args
    ----
    input_dim : int
        Dimensionality of the raw spectrum.
    d_model : int
        Model hidden dimension.
    """

    def __init__(self, input_dim: int, d_model: int) -> None:
        super().__init__()

        self.spectrum_embedding = torch.nn.Sequential(
            torch.nn.Linear(input_dim, d_model),
            torch.nn.SiLU(),
            torch.nn.Linear(d_model, d_model, bias=True),
            torch.nn.LayerNorm(d_model),
        )

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        return self.spectrum_embedding(spectra)


class StackEmbedding(torch.nn.Module):
    """
    Standard token embedding for discrete stack tokens.

    Args
    ----
    input_vocab : int
        Vocabulary size for layer/material tokens.
    d_model : int
        Embedding dimensionality.
    """

    def __init__(self, input_vocab: int, d_model: int) -> None:
        super().__init__()

        self.stack_embedding = torch.nn.Embedding(input_vocab, d_model)

    def forward(self, stacks: torch.Tensor) -> torch.Tensor:
        return self.stack_embedding(stacks)


class ThicknessStateEmbedding(torch.nn.Module):
    """
    Embed the current continuous layer-thickness state for hybrid diffusion.

    The input is physical thickness in nm. A log transform keeps 10 nm and
    500 nm values in a compact numeric range before projection.
    """

    def __init__(self, d_model: int, scale_nm: float = 500.0) -> None:
        super().__init__()
        self.scale_nm = max(float(scale_nm), 1.0)
        self.embedding = torch.nn.Sequential(
            torch.nn.Linear(1, d_model),
            torch.nn.SiLU(),
            torch.nn.Linear(d_model, d_model),
            torch.nn.LayerNorm(d_model),
        )

    def forward(self, thickness_nm: torch.Tensor) -> torch.Tensor:
        thickness = thickness_nm.to(torch.float32).clamp_min(0.0)
        scaled = torch.log1p(thickness) / math.log1p(self.scale_nm)
        return self.embedding(scaled.unsqueeze(-1))


class DepthRoPESelfAttention(torch.nn.Module):
    """
    Self-attention with rotary position embeddings from physical stack depth.

    Positions are continuous layer depths in nm. They are scaled before the
    sinusoidal rotation so values remain numerically comparable across stacks.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        scale_nm: float = 1000.0,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}.")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = int(d_model // n_heads)
        self.rotary_dim = (self.head_dim // 2) * 2
        if self.rotary_dim <= 0:
            raise ValueError("DepthRoPESelfAttention requires head_dim >= 2.")
        self.dropout = float(dropout)
        self.scale_nm = max(float(scale_nm), 1e-6)

        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)

        inv_freq = torch.exp(
            -math.log(float(base)) * torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        return x.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)

    def _apply_rope(self, x: torch.Tensor, depth_nm: torch.Tensor) -> torch.Tensor:
        rot = x[..., : self.rotary_dim]
        tail = x[..., self.rotary_dim :]
        pos = depth_nm.to(device=x.device, dtype=x.dtype) / self.scale_nm
        inv_freq = self.inv_freq.to(device=x.device, dtype=x.dtype)
        freqs = pos.unsqueeze(-1) * inv_freq.view(1, 1, -1)  # [B,S,D/2]
        cos = torch.cos(freqs).unsqueeze(1)  # [B,1,S,D/2]
        sin = torch.sin(freqs).unsqueeze(1)

        even = rot[..., 0::2]
        odd = rot[..., 1::2]
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
        if tail.numel() == 0:
            return rotated
        return torch.cat((rotated, tail), dim=-1)

    def forward(self, x: torch.Tensor, depth_nm: torch.Tensor) -> torch.Tensor:
        q = self._shape(self.q_proj(x))
        k = self._shape(self.k_proj(x))
        v = self._shape(self.v_proj(x))

        q = self._apply_rope(q, depth_nm)
        k = self._apply_rope(k, depth_nm)

        dropout_p = self.dropout if self.training else 0.0
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        attended = attended.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.d_model)
        return self.out_proj(attended)


# TimestepEmbedding class from Kuleshov group: https://github.com/kuleshov-group/bd3lms/blob/1c3e8f43d88dfbcee5ff2aa6932a9e74b31ae1d7/models/dit.py#L236
class TimestepEmbedding(torch.nn.Module):
    """
    Fourier timestep embedding.

    Implements the sinusoidal timestep embedding from the BD3LMS / DiT
    architecture (Kuleshov Group), followed by a two-layer MLP.

    Args
    ----
    d_model : int
        Output embedding dimension.
    frequency_embedding_size : int
        Size of the Fourier feature vector.
    """

    def __init__(self, d_model: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()

        self.frequency_embedding_size = frequency_embedding_size

        self.embedding = torch.nn.Sequential(
            torch.nn.Linear(frequency_embedding_size, d_model, bias=True),
            torch.nn.SiLU(),
            torch.nn.Linear(d_model, d_model, bias=True),
            torch.nn.SiLU(),
        )

    def timestep_embedding(self, timesteps: torch.Tensor, max_period: int = 10000) -> torch.Tensor:
        half = self.frequency_embedding_size // 2

        frequencies = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, device=timesteps.device) / half)
        sigmas = timesteps
        projected = sigmas.unsqueeze(-1) * frequencies.unsqueeze(0)

        return torch.cat([torch.cos(projected), torch.sin(projected)], dim=-1)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        frequencies = self.timestep_embedding(timesteps)

        return self.embedding(frequencies).unsqueeze(1)


# Peebles, W., & Xie, S. (2023). Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 4195-4205).
# Arxiv: (https://arxiv.org/pdf/2212.09748)
class AdaLayerNormGaussian(torch.nn.Module):
    """
    Adaptive LayerNorm with Gaussian initialization (Peebles & Xie, 2023).

    Modulates normalization using a conditioning vector ``cond``, producing
    (Δγ, β) shifts that scale and bias normalized activations.

    Args
    ----
    hidden_size : int
        Size of the normalized dimension.
    cond_dim : int
        Dimensionality of the conditioning embedding.
    std_gamma : float
        Initialization std for Δγ parameters.
    std_beta : float
        Initialization std for β parameters.
    """

    def __init__(self, hidden_size: int, cond_dim: int, std_gamma: float = 1.2e-3, std_beta: float = 8e-4):
        super().__init__()
        self.eps = 1e-5
        # no affine params inside the LN itself
        self.to_scale_shift = torch.nn.Linear(cond_dim, 2 * hidden_size, bias=True)

        # --- Gaussian init (key difference from adaLN-Zero) ---
        torch.nn.init.normal_(self.to_scale_shift.weight[:hidden_size], 0.0, std_gamma)  # Δγ
        torch.nn.init.normal_(self.to_scale_shift.weight[hidden_size:], 0.0, std_beta)  # β
        torch.nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sig = x.var(-1, unbiased=False, keepdim=True).add(self.eps).sqrt()
        x_hat = (x - mu) / sig  # weight-free LN

        delta_g, beta = self.to_scale_shift(cond).chunk(2, dim=-1)  # [B, H] each
        y = x_hat * (1 + delta_g).unsqueeze(1) + beta.unsqueeze(1)
        return y


class Block(torch.nn.Module):
    """
    Transformer block with cross-attention, self-attention, and AdaLN-Gaussian.

    A single DiT-style block:

    - Cross-attention over encoded spectra
    - Self-attention over the predicted token stack
    - Feed-forward network
    - α-gates modulating attention with timestep conditioning

    Args
    ----
    d_model : int
        Hidden dimension.
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout probability.
    cond_dim : int
        Dimensionality of the conditional vector used by AdaLN / gates.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        cond_dim: int,
        depth_rope_enabled: bool = False,
        depth_rope_scale_nm: float = 1000.0,
        depth_rope_base: float = 10000.0,
    ):
        super().__init__()

        self.cross_attn = torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.depth_rope_enabled = bool(depth_rope_enabled)
        if self.depth_rope_enabled:
            self.self_attn = DepthRoPESelfAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                scale_nm=depth_rope_scale_nm,
                base=depth_rope_base,
            )
        else:
            self.self_attn = torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ff1 = torch.nn.Linear(d_model, 4 * d_model)
        self.ff2 = torch.nn.Linear(4 * d_model, d_model)

        self.norm1 = AdaLayerNormGaussian(d_model, cond_dim)
        self.norm2 = AdaLayerNormGaussian(d_model, cond_dim)
        self.norm3 = AdaLayerNormGaussian(d_model, cond_dim)

        # α-gates (condition-dependent) ------------------------
        self.to_alpha1 = torch.nn.Linear(cond_dim, d_model)
        self.to_alpha2 = torch.nn.Linear(cond_dim, d_model)
        torch.nn.init.normal_(self.to_alpha1.weight, 0.0, 8e-4)
        torch.nn.init.normal_(self.to_alpha2.weight, 0.0, 8e-4)
        torch.nn.init.zeros_(self.to_alpha1.bias)
        torch.nn.init.zeros_(self.to_alpha2.bias)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self,
        predicted_stack: torch.Tensor,
        spectra: torch.Tensor,
        cond: torch.Tensor,
        depth_positions_nm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = predicted_stack

        # cross-attention
        predicted_stack = self.norm1(predicted_stack, cond)
        predicted_stack, _ = self.cross_attn(query=predicted_stack, key=spectra, value=spectra)
        alpha1 = 1 + self.to_alpha1(cond).unsqueeze(1)  # shape [B,1,H]
        predicted_stack = predicted_stack * alpha1
        predicted_stack += residual
        residual = predicted_stack

        # self-attention
        predicted_stack = self.norm2(predicted_stack, cond)
        if self.depth_rope_enabled:
            if depth_positions_nm is None:
                raise ValueError("depth_positions_nm must be provided when DEPTH_ROPE is enabled.")
            predicted_stack = self.self_attn(predicted_stack, depth_positions_nm)
        else:
            predicted_stack, _ = self.self_attn(query=predicted_stack, key=predicted_stack, value=predicted_stack)
        alpha2 = 1 + self.to_alpha2(cond).unsqueeze(1)  # shape [B,1,H]
        predicted_stack = predicted_stack * alpha2
        predicted_stack += residual
        residual = predicted_stack

        # feedforward
        predicted_stack = self.norm3(predicted_stack, cond)
        # predicted_stack = self.ff(predicted_stack)
        predicted_stack = self.ff1(predicted_stack)
        predicted_stack = torch.nn.functional.silu(predicted_stack)
        predicted_stack = self.ff2(predicted_stack)

        predicted_stack = self.dropout(predicted_stack)
        predicted_stack = predicted_stack + residual
        return predicted_stack


class OptoLlama(torch.nn.Module):
    """
    Discrete diffusion transformer for thin-film stack generation.

    This model predicts a sequence of discrete material/layer tokens conditioned
    on an input spectrum. It follows a DiT-style architecture with:

    - Spectrum embedding
    - Stack token embedding
    - Timestep embedding
    - Multiple conditional Transformer blocks
    - Diffusion-style masking/noising
    - Autoregressive-free sampling

    Args
    ----
    spectra_dim : int
        Dimensionality of the input spectrum.
    vocab_size : int
        Number of discrete material/layer tokens.
    timesteps : int
        Number of diffusion sampling steps.
    max_stack_depth : int
        Maximum token length of the predicted stack.
    eos_idx : int
        EOS token index.
    pad_idx : int
        PAD token index.
    mask_idx : int
        MASK token index for diffusion noising.
    d_model : int
        Transformer hidden size.
    n_blocks : int
        Number of transformer blocks.
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout probability.
    idx_to_token : dict
        Mapping from token ids to token strings.
    temperature : float
        Sampling temperature (``0.0`` = deterministic).
    top_k : int
        Top-k sampling cutoff.
    top_p : float
        Top-p (nucleus) sampling cutoff.
    spectrum_latent : dict, optional
        Optional frozen spectrum-autoencoder latent conditioning config.
    """

    def __init__(
        self,
        spectra_dim: int,
        vocab_size: int,
        timesteps: int,
        max_stack_depth: int,
        eos_idx: int,
        pad_idx: int,
        mask_idx: int,
        d_model: int,
        n_blocks: int,
        n_heads: int,
        dropout: float,
        idx_to_token: dict,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 0.0,
        spectrum_latent: Optional[dict] = None,
        depth_position: Optional[dict] = None,
        depth_rope: Optional[dict] = None,
        factored_output: Optional[dict] = None,
    ) -> None:
        super().__init__()

        self.n_blocks = n_blocks
        self.steps = timesteps
        self.eos = eos_idx
        self.pad = pad_idx
        self.mask = mask_idx
        self.vocab_size = vocab_size
        self.max_stack_depth = max_stack_depth
        self.d_model = d_model
        self.idx_to_token = idx_to_token

        self.spectrum_embedding = SpectrumEmbedding(spectra_dim, d_model)
        self.spectrum_latent_enabled = bool((spectrum_latent or {}).get("ENABLED", False))
        self.spectrum_autoencoder: Optional[torch.nn.Module] = None
        self.spectrum_latent_embedding: Optional[torch.nn.Module] = None
        if self.spectrum_latent_enabled:
            self._init_spectrum_latent_conditioning(spectrum_latent or {}, d_model)

        self.stack_embedding = StackEmbedding(vocab_size, d_model)
        self.time_embedding = TimestepEmbedding(d_model)

        self.positional_encoding = PositionalEncoding(2000, d_model)
        _, _, token_thickness_nm, layer_token_mask = _token_metadata(idx_to_token, vocab_size, pad_idx, eos_idx, mask_idx)
        self.register_buffer("token_thickness_nm", token_thickness_nm.to(torch.float32), persistent=False)
        self.register_buffer("layer_token_mask", layer_token_mask.to(torch.bool), persistent=False)
        depth_cfg = depth_position or {}
        self.depth_position_enabled = bool(depth_cfg.get("ENABLED", False))
        self.depth_positional_encoding: Optional[DepthPositionalEncoding] = None
        if self.depth_position_enabled:
            self.depth_positional_encoding = DepthPositionalEncoding(
                max_len=max_stack_depth,
                d_model=d_model,
                token_thickness_nm=token_thickness_nm,
                eos_idx=eos_idx,
                pad_idx=pad_idx,
                mask_idx=mask_idx,
                scale_nm=float(depth_cfg.get("SCALE_NM", 1000.0)),
                mask_thickness_nm=depth_cfg.get("MASK_THICKNESS_NM"),
                layer_centers=bool(depth_cfg.get("LAYER_CENTERS", True)),
            )
        depth_rope_cfg = depth_rope or {}
        self.depth_rope_enabled = bool(depth_rope_cfg.get("ENABLED", False))
        self.depth_rope_layer_centers = bool(depth_rope_cfg.get("LAYER_CENTERS", True))
        self.depth_rope_scale_nm = float(depth_rope_cfg.get("SCALE_NM", depth_cfg.get("SCALE_NM", 1000.0)))
        self.depth_rope_base = float(depth_rope_cfg.get("BASE", 10000.0))
        self.noise = SquareNoise()

        self.blocks = torch.nn.ModuleList([
            Block(
                d_model,
                n_heads,
                dropout,
                cond_dim=d_model,
                depth_rope_enabled=self.depth_rope_enabled,
                depth_rope_scale_nm=self.depth_rope_scale_nm,
                depth_rope_base=self.depth_rope_base,
            )
            for _ in range(n_blocks)
        ])
        factored_cfg = factored_output or {}
        self.factored_output_enabled = bool(factored_cfg.get("ENABLED", False))
        self.material_vocab_mode = bool(self.factored_output_enabled and factored_cfg.get("MATERIAL_VOCAB_MODE", False))
        self.factored_thickness_weight = float(factored_cfg.get("THICKNESS_LOSS_WEIGHT", 0.25))
        self.factored_joint_ce_weight = float(factored_cfg.get("JOINT_CE_WEIGHT", 0.10))
        self.factored_label_smoothing = float(factored_cfg.get("MATERIAL_LABEL_SMOOTHING", 0.05))
        continuous_cfg = factored_cfg.get("CONTINUOUS_THICKNESS") or {}
        self.continuous_thickness_enabled = bool(
            self.factored_output_enabled and continuous_cfg.get("ENABLED", False)
        )
        self.continuous_thickness_use_for_tmm = bool(continuous_cfg.get("USE_FOR_TMM", True))
        self.continuous_thickness_save_in_results = bool(continuous_cfg.get("SAVE_IN_RESULTS", True))
        self.continuous_thickness_round_step_nm = continuous_cfg.get("ROUND_STEP_NM")
        self.continuous_thickness_min_nm = float(continuous_cfg.get("MIN_NM", 1.0))
        self.continuous_thickness_max_nm = float(continuous_cfg.get("MAX_NM", 500.0))
        positive_thickness = token_thickness_nm[token_thickness_nm > 0]
        default_state_init = float(positive_thickness.median().item()) if positive_thickness.numel() else 100.0
        self.continuous_thickness_state_enabled = bool(
            self.continuous_thickness_enabled and continuous_cfg.get("STATE_ENABLED", False)
        )
        self.continuous_thickness_state_noise_log_std = float(continuous_cfg.get("STATE_NOISE_LOG_STD", 0.30))
        self.continuous_thickness_state_init_nm = float(continuous_cfg.get("STATE_INIT_NM") or default_state_init)
        self.thickness_state_embedding: Optional[ThicknessStateEmbedding] = None
        self.material_state_embedding: Optional[StackEmbedding] = None
        self.material_vocab_size: Optional[int] = None
        self.pad_material_id: Optional[int] = None
        self.eos_material_id: Optional[int] = None
        self.mask_material_id: Optional[int] = None
        if self.continuous_thickness_state_enabled:
            self.thickness_state_embedding = ThicknessStateEmbedding(
                d_model=d_model,
                scale_nm=self.continuous_thickness_max_nm,
            )
        self.factored_head: Optional[FactoredOutputHead] = None
        if self.factored_output_enabled:
            self.factored_head = FactoredOutputHead(
                d_model=d_model,
                idx_to_token=idx_to_token,
                vocab_size=vocab_size,
                pad_idx=pad_idx,
                eos_idx=eos_idx,
                mask_idx=mask_idx,
                thickness_log_sigma=float(factored_cfg.get("THICKNESS_LOG_SIGMA", 0.20)),
            )
            if self.material_vocab_mode:
                if not self.continuous_thickness_state_enabled:
                    raise ValueError(
                        "FACTORED_OUTPUT.MATERIAL_VOCAB_MODE requires "
                        "CONTINUOUS_THICKNESS.ENABLED=true and STATE_ENABLED=true."
                    )
                self.material_vocab_size = int(self.factored_head.num_materials)
                self.material_state_embedding = StackEmbedding(self.material_vocab_size, d_model)
                self.pad_material_id = int(self.factored_head.pad_material_id)
                self.eos_material_id = int(self.factored_head.eos_material_id)
                self.mask_material_id = int(self.factored_head.mask_material_id)
            self.projection = None
        else:
            self.projection = torch.nn.Linear(d_model, vocab_size)
            self.continuous_thickness_enabled = False
            self.continuous_thickness_state_enabled = False
            self.material_vocab_mode = False
            self.thickness_state_embedding = None
            self.material_state_embedding = None

        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        # Inference-time constraint: force solutions to terminate by this length.
        # If set to max_len:
        #   - position max_len-1 is forced to EOS
        #   - positions >= max_len are forced to PAD
        # This does not affect training and does not change model weights.
        self.max_emit_len: Optional[int] = None

        # Optional token constraints for sampling (inference-time only):
        # a boolean mask over vocabulary; False entries cannot be sampled.
        # Default: allow everything.
        self.register_buffer("allowed_vocab_mask", torch.ones(vocab_size, dtype=torch.bool))
        # self.allowed_vocab_mask: Optional[torch.Tensor] = torch.ones(vocab_size, dtype=torch.bool, device=)

        # By default we do not want to *emit* the diffusion mask token as a final output.
        # The diffusion process still uses `self.mask` internally via remasking.
        if 0 <= int(self.mask) < int(vocab_size):
            self.allowed_vocab_mask[int(self.mask)] = False

        # Optional: per-step MAE tracking during sampling
        self._step_mae_enabled: bool = False
        self._step_mae_ctx: Optional[TMMContext] = None

    def _init_spectrum_latent_conditioning(self, cfg: dict, d_model: int) -> None:
        """Load a frozen spectrum AE encoder and trainable latent projection."""
        mode = str(cfg.get("MODE", "token")).lower()
        if mode != "token":
            raise ValueError(f"Unsupported SPECTRUM_LATENT.MODE={mode!r}; currently only 'token' is implemented.")

        checkpoint = cfg.get("CHECKPOINT")
        if not checkpoint:
            raise ValueError("SPECTRUM_LATENT.CHECKPOINT must be set when SPECTRUM_LATENT.ENABLED is true.")

        from optollama.model.spectrum_autoencoder import load_spectrum_autoencoder

        ae, _ = load_spectrum_autoencoder(checkpoint, device="cpu")
        freeze_encoder = bool(cfg.get("FREEZE_ENCODER", True))
        if freeze_encoder:
            for param in ae.parameters():
                param.requires_grad_(False)
        ae.eval()

        self.spectrum_autoencoder = ae
        self.spectrum_latent_embedding = torch.nn.Sequential(
            torch.nn.Linear(ae.latent_dim, d_model),
            torch.nn.SiLU(),
            torch.nn.Linear(d_model, d_model),
            torch.nn.LayerNorm(d_model),
        )

    def _spectrum_latent_token(self, spectra: torch.Tensor) -> Optional[torch.Tensor]:
        """Return one AE latent conditioning token per spectrum, or ``None``."""
        if not self.spectrum_latent_enabled:
            return None
        if self.spectrum_autoencoder is None or self.spectrum_latent_embedding is None:
            raise RuntimeError("Spectrum latent conditioning is enabled but not initialized.")

        self.spectrum_autoencoder.eval()
        with torch.no_grad():
            z = self.spectrum_autoencoder.encode(spectra.to(dtype=torch.float32))
        z = z.to(device=spectra.device, dtype=spectra.dtype)
        return self.spectrum_latent_embedding(z).unsqueeze(1)

    def set_max_emit_len(self, max_len: Optional[int]) -> None:
        """
        Set a hard maximum emitted sequence length (in tokens, incl. EOS position).

        Args
        ----
        max_len : int or None
            Maximum number of emitted tokens (including the EOS position).
            If ``None``, the constraint is disabled. Values ``<= 0`` are
            treated as ``1`` (immediate EOS at position 0).
        """
        if max_len is None:
            self.max_emit_len = None
            return
        ml = int(max_len)
        if ml <= 0:
            ml = 1
        # Can't exceed model capacity
        self.max_emit_len = min(ml, int(self.max_stack_depth))

    def set_token_constraints(
        self,
        allow_ids: Optional[torch.Tensor] = None,
        exclude_ids: Optional[torch.Tensor] = None,
        allow_eos_pad: bool = True,
        allow_msk: bool = False,
    ) -> None:
        """
        Set inference-time sampling constraints.

        This does **not** change model weights; it only changes which tokens
        can be sampled during ``_sample_logits``.

        Args
        ----
        allow_ids : torch.Tensor, optional
            If provided, only these token ids are allowed (plus optionally
            EOS/PAD depending on ``allow_eos_pad``).
        exclude_ids : torch.Tensor, optional
            If provided, these token ids are forbidden.
        allow_eos_pad : bool
            If ``True``, always allow EOS and PAD even in allowlist mode.
        allow_msk : bool
            If ``True``, allow emitting ``<MSK>`` as a sampled output token.
        """
        mask = torch.zeros((self.vocab_size,), dtype=torch.bool, device=self.allowed_vocab_mask.device)
        if allow_ids is None:
            mask[:] = True
        else:
            allow_ids = allow_ids.to(device=mask.device, dtype=torch.long)
            mask[allow_ids] = True

        if allow_eos_pad:
            if 0 <= int(self.eos) < int(self.vocab_size):
                mask[int(self.eos)] = True
            if 0 <= int(self.pad) < int(self.vocab_size):
                mask[int(self.pad)] = True

        if exclude_ids is not None and exclude_ids.numel() > 0:
            exclude_ids = exclude_ids.to(device=mask.device, dtype=torch.long)
            mask[exclude_ids] = False

        # Control whether <MSK> can ever be sampled as an emitted token.
        if not allow_msk and 0 <= int(self.mask) < int(self.vocab_size):
            mask[int(self.mask)] = False

        self.allowed_vocab_mask.copy_(mask)

    def _sample_t(self, batch: torch.Tensor, sampling_eps: float = 1e-3) -> torch.Tensor:
        """
        Sample diffusion timesteps t ∈ (eps, 1].

        Produces a set of timesteps for batched diffusion training. Ensures
        even coverage of the unit interval and avoids t=0.

        Args
        ----
        batch : torch.Tensor
            Token batch whose size determines the number of timesteps.
        sampling_eps : float
            Minimum timestep value to avoid degenerate noise.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``[B]`` with sampled timesteps in ``(eps, 1]``.
        """
        n, device = batch.shape[0], batch.device

        timesteps = torch.rand(n, device=device)
        # antithetic sampling
        offset = torch.arange(n, device=device) / n
        timesteps = (timesteps / n + offset) % 1.0

        return (1.0 - sampling_eps) * timesteps + sampling_eps

    def _depth_positions_nm(
        self,
        noised_stacks: torch.Tensor,
        state_thickness_nm: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Return physical depth positions in nm for depth-RoPE."""
        if not self.depth_rope_enabled:
            return None

        if state_thickness_nm is not None:
            thickness = state_thickness_nm.to(device=noised_stacks.device, dtype=torch.float32).clamp_min(0.0)
        elif self.material_vocab_mode:
            thickness = torch.zeros(noised_stacks.shape, device=noised_stacks.device, dtype=torch.float32)
        else:
            token_ids = noised_stacks.to(torch.long).clamp(0, self.token_thickness_nm.numel() - 1)
            thickness = self.token_thickness_nm.to(device=token_ids.device, dtype=torch.float32)[token_ids]
            is_eos = token_ids == self.eos
            before_eos = is_eos.cumsum(dim=1) == 0
            valid = before_eos & (token_ids != self.pad) & (token_ids != self.mask)
            thickness = thickness * valid.to(thickness.dtype)

        positions = thickness.cumsum(dim=1)
        if self.depth_rope_layer_centers:
            positions = positions - 0.5 * thickness
        return positions

    def _hidden(
        self,
        spectra: torch.Tensor,
        noised_stacks: torch.Tensor,
        timesteps: torch.Tensor,
        state_thickness_nm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the DiT backbone before the output head.

        Embeds spectra, stacks, and timesteps; applies positional encodings and
        conditional DiT blocks.

        Args
        ----
        spectra : torch.Tensor
            Input spectra, shape ``[B, D_spec]``.
        noised_stacks : torch.Tensor
            Masked/noised stack tokens, shape ``[B, S]``.
        timesteps : torch.Tensor
            Diffusion timesteps, shape ``[B]``.
        state_thickness_nm : torch.Tensor, optional
            Current continuous thickness state, shape ``[B, S]``. When
            enabled, this is embedded directly and used for depth positions.

        Returns
        -------
        torch.Tensor
            Hidden stack states, shape ``[B, S, d_model]``.
        """
        embedded_spectra = self.spectrum_embedding(spectra)  # [B, 3, d_model]
        latent_token = self._spectrum_latent_token(spectra)
        if latent_token is not None:
            embedded_spectra = torch.cat([embedded_spectra, latent_token], dim=1)
        embedded_spectra += self.positional_encoding(embedded_spectra)
        if self.material_vocab_mode:
            if self.material_state_embedding is None:
                raise RuntimeError("MATERIAL_VOCAB_MODE is enabled but the material embedding is not initialized.")
            predicted_stacks = self.material_state_embedding(noised_stacks)
        else:
            predicted_stacks = self.stack_embedding(noised_stacks)
        if self.continuous_thickness_state_enabled and state_thickness_nm is not None:
            if self.thickness_state_embedding is None:
                raise RuntimeError("Continuous thickness state is enabled but the embedding was not initialized.")
            predicted_stacks += self.thickness_state_embedding(state_thickness_nm).to(dtype=predicted_stacks.dtype)
        if self.depth_position_enabled:
            if self.depth_positional_encoding is None:
                raise RuntimeError("DEPTH_POSITION is enabled but the encoder was not initialized.")
            depth_thickness = state_thickness_nm if self.continuous_thickness_state_enabled else None
            predicted_stacks += self.depth_positional_encoding(
                noised_stacks,
                thickness_nm=depth_thickness,
            ).to(dtype=predicted_stacks.dtype)
        else:
            predicted_stacks += self.positional_encoding(predicted_stacks)
        predicted_stacks += self.time_embedding(timesteps)  # [B, S, d_model]
        cond = self.time_embedding(timesteps)  # [B, 1, 1024]
        cond = cond.squeeze(1)  # [B, 1024]
        depth_positions_nm = self._depth_positions_nm(noised_stacks, state_thickness_nm)

        for block in self.blocks:
            predicted_stacks = block(
                predicted_stacks,
                embedded_spectra,
                cond,
                depth_positions_nm=depth_positions_nm,
            )

        return predicted_stacks

    def _project_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project backbone hidden states to the active output space."""
        if self.factored_output_enabled:
            if self.factored_head is None:
                raise RuntimeError("FACTORED_OUTPUT is enabled but the factored head was not initialized.")
            outputs = self.factored_head(hidden)
            if self.material_vocab_mode:
                return outputs["material_logits"]
            return outputs["joint_logits"]
        if self.projection is None:
            raise RuntimeError("Joint projection head is not initialized.")
        return self.projection(hidden)

    def _sample_outputs(
        self,
        spectra: torch.Tensor,
        noised_stacks: torch.Tensor,
        timesteps: torch.Tensor,
        state_thickness_nm: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]]]:
        """
        Return joint token logits and optional factored-head outputs.

        Sampling needs the full factored output only when continuous
        thickness is enabled; training and legacy inference keep using the
        joint logits interface.
        """
        hidden = self._hidden(spectra, noised_stacks, timesteps, state_thickness_nm=state_thickness_nm)
        if self.factored_output_enabled:
            if self.factored_head is None:
                raise RuntimeError("FACTORED_OUTPUT is enabled but the factored head was not initialized.")
            outputs = self.factored_head(hidden)
            if self.material_vocab_mode:
                return outputs["material_logits"], outputs
            return outputs["joint_logits"], outputs
        if self.projection is None:
            raise RuntimeError("Joint projection head is not initialized.")
        return self.projection(hidden), None

    def _model(
        self,
        spectra: torch.Tensor,
        noised_stacks: torch.Tensor,
        timesteps: torch.Tensor,
        state_thickness_nm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the DiT and active output head.

        Returns predicted logits over the existing joint token vocabulary,
        shape ``[B, S, vocab_size]``.
        """
        predicted_stacks = self._project_hidden(
            self._hidden(spectra, noised_stacks, timesteps, state_thickness_nm=state_thickness_nm)
        )

        return predicted_stacks

    def _continuous_thickness_for_sample(
        self,
        outputs: Optional[dict[str, torch.Tensor]],
        sampled_stacks: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Compute continuous thicknesses for sampled hard tokens, if enabled.
        """
        if not self.continuous_thickness_enabled:
            return None
        if outputs is None or self.factored_head is None:
            return None
        if self.material_vocab_mode:
            thickness = self.factored_head.thickness_for_material_ids(
                outputs,
                sampled_stacks,
                min_nm=self.continuous_thickness_min_nm,
                max_nm=self.continuous_thickness_max_nm,
                round_step_nm=self.continuous_thickness_round_step_nm,
            )
            material_ids = sampled_stacks.to(torch.long)
            eos_id = int(self.eos_material_id)
            pad_id = int(self.pad_material_id)
            mask_id = int(self.mask_material_id)
            is_eos = material_ids == eos_id
            before_eos = is_eos.cumsum(dim=1) == 0
            valid = before_eos & (material_ids != pad_id) & (material_ids != mask_id)
            return thickness * valid.to(thickness.dtype)

        thickness = self.factored_head.thickness_for_ids(
            outputs,
            sampled_stacks,
            min_nm=self.continuous_thickness_min_nm,
            max_nm=self.continuous_thickness_max_nm,
            round_step_nm=self.continuous_thickness_round_step_nm,
        )
        token_ids = sampled_stacks.to(torch.long)
        is_eos = token_ids == self.eos
        before_eos = is_eos.cumsum(dim=1) == 0
        valid = before_eos & (token_ids != self.pad) & (token_ids != self.mask)
        return thickness * valid.to(thickness.dtype)

    def _material_ids_from_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Convert old joint material-thickness token IDs to material IDs."""
        if self.factored_head is None:
            raise RuntimeError("Material conversion requires FACTORED_OUTPUT.ENABLED=true.")
        ids = token_ids.to(torch.long).clamp(0, self.vocab_size - 1)
        token_material_ids = self.factored_head.token_material_ids.to(device=ids.device)
        return token_material_ids[ids]

    def _tokens_from_material_state(self, material_ids: torch.Tensor, thickness_nm: torch.Tensor) -> torch.Tensor:
        """Convert internal material/thickness state to nearest old tokens."""
        if self.factored_head is None:
            raise RuntimeError("Material-token conversion requires FACTORED_OUTPUT.ENABLED=true.")
        return self.factored_head.tokens_for_material_thickness(material_ids, thickness_nm)

    def _active_sampling_mask(self, device: torch.device) -> torch.Tensor:
        """Return the allowed output mask for the active output vocabulary."""
        if not self.material_vocab_mode:
            return self.allowed_vocab_mask.to(device=device)
        if self.factored_head is None:
            raise RuntimeError("Material sampling mask requires FACTORED_OUTPUT.ENABLED=true.")
        token_allowed = self.allowed_vocab_mask.to(device=device, dtype=torch.bool)
        token_material_ids = self.factored_head.token_material_ids.to(device=device)
        material_mask = torch.zeros((self.factored_head.num_materials,), dtype=torch.bool, device=device)
        material_mask[token_material_ids[token_allowed]] = True
        material_mask[int(self.mask_material_id)] = False
        return material_mask

    def _layer_mask_from_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """Return true material-layer positions before EOS."""
        token_ids = ids.to(torch.long).clamp(0, self.vocab_size - 1)
        is_eos = token_ids == self.eos
        before_eos = is_eos.cumsum(dim=1) == 0
        layer_token_mask = self.layer_token_mask.to(device=token_ids.device)
        return before_eos & layer_token_mask[token_ids]

    def _target_thickness_from_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """Read nominal token thicknesses for material layers only."""
        token_ids = ids.to(torch.long).clamp(0, self.vocab_size - 1)
        thickness = self.token_thickness_nm.to(device=token_ids.device, dtype=torch.float32)[token_ids]
        layer_mask = self._layer_mask_from_ids(token_ids)
        return thickness * layer_mask.to(thickness.dtype)

    def _noise_thickness_state(self, stacks: torch.Tensor, timesteps: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Build the noised continuous thickness state used during training.

        Noise is applied in log-thickness space and only on true material
        layers. Positions after EOS and special tokens remain zero.
        """
        if not self.continuous_thickness_state_enabled:
            return None

        target = self._target_thickness_from_ids(stacks)
        valid = target > 0
        if not valid.any():
            return torch.zeros_like(target)

        std = float(self.continuous_thickness_state_noise_log_std) * timesteps.reshape(-1, 1).clamp(0.0, 1.0)
        log_target = torch.log(target.clamp_min(float(self.continuous_thickness_min_nm)))
        noisy = torch.exp(log_target + torch.randn_like(target) * std)
        noisy = noisy.clamp(float(self.continuous_thickness_min_nm), float(self.continuous_thickness_max_nm))
        return torch.where(valid, noisy, torch.zeros_like(noisy))

    def _initial_thickness_state(self, stacks: torch.Tensor) -> Optional[torch.Tensor]:
        """Initial continuous thickness state for sampling from all-mask tokens."""
        if not self.continuous_thickness_state_enabled:
            return None
        return torch.full(
            stacks.shape,
            float(self.continuous_thickness_state_init_nm),
            dtype=torch.float32,
            device=stacks.device,
        )

    def _train(self, spectra: torch.Tensor, stacks: torch.Tensor) -> torch.Tensor:
        """
        Training-time diffusion step.

        Samples a timestep, applies masking noise to the input stack, and predicts
        denoised logits via the transformer backbone.

        Args
        ----
        spectra : torch.Tensor
            Conditioning spectra, shape ``[B, D_spec]``.
        stacks : torch.Tensor
            Ground-truth token stacks, shape ``[B, S]``.

        Returns
        -------
        torch.Tensor
            Predicted logits for all stack positions, shape ``[B, S, vocab_size]``.
        """
        # sample time points
        timesteps = self._sample_t(stacks)

        # convert into noise
        betas = self.noise(timesteps)

        flip_chance = betas.reshape(-1, 1)
        flipped = torch.rand_like(stacks, dtype=spectra.dtype) < flip_chance
        if self.material_vocab_mode:
            target_state = self._material_ids_from_tokens(stacks)
            noised_stacks = torch.where(flipped, int(self.mask_material_id), target_state)
        else:
            noised_stacks = torch.where(flipped, self.mask, stacks)

        # query model
        state_thickness_nm = self._noise_thickness_state(stacks, timesteps)
        predicted_stacks = self._model(
            spectra,
            noised_stacks,
            timesteps,
            state_thickness_nm=state_thickness_nm,
        )

        return predicted_stacks

    def _noised_training_inputs(
        self,
        spectra: torch.Tensor,
        stacks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Sample diffusion timesteps and mask tokens for training."""
        timesteps = self._sample_t(stacks)
        betas = self.noise(timesteps)

        flip_chance = betas.reshape(-1, 1)
        flipped = torch.rand_like(stacks, dtype=spectra.dtype) < flip_chance
        if self.material_vocab_mode:
            target_state = self._material_ids_from_tokens(stacks)
            noised_stacks = torch.where(flipped, int(self.mask_material_id), target_state)
        else:
            noised_stacks = torch.where(flipped, self.mask, stacks)
        state_thickness_nm = self._noise_thickness_state(stacks, timesteps)
        return timesteps, noised_stacks, state_thickness_nm

    def _factored_training_loss(self, spectra: torch.Tensor, stacks: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute material CE + log-thickness regression for the factored head."""
        if self.factored_head is None:
            raise RuntimeError("Factored training loss requires FACTORED_OUTPUT.ENABLED=true.")

        timesteps, noised_stacks, state_thickness_nm = self._noised_training_inputs(spectra, stacks)
        hidden = self._hidden(spectra, noised_stacks, timesteps, state_thickness_nm=state_thickness_nm)
        outputs = self.factored_head(hidden)
        loss, parts = self.factored_head.loss(
            outputs,
            stacks,
            thickness_weight=self.factored_thickness_weight,
            joint_ce_weight=0.0 if self.material_vocab_mode else self.factored_joint_ce_weight,
            label_smoothing=self.factored_label_smoothing,
        )
        if self.material_vocab_mode:
            parts["joint_loss"] = parts["joint_loss"] * 0.0
        return {
            "loss": loss,
            "logits": outputs["joint_logits"],
            **parts,
        }

    def enable_step_mae(self, tmm_ctx: Optional[TMMContext]) -> None:
        """
        Enable or disable per-step MAE tracking during sampling.

        When ``tmm_ctx`` is not ``None``, ``_sample`` will simulate spectra at
        every denoising step, compute ``masked_mae`` against the conditioning
        spectra, and return a ``[B, steps]`` trajectory as the second output.

        Args
        ----
        tmm_ctx : TMMContext or None
            TMM simulation context used for per-step spectrum evaluation.
            Pass ``None`` to disable tracking.
        """
        self._step_mae_ctx = tmm_ctx
        self._step_mae_enabled = tmm_ctx is not None

    def _sample_logits(
        self,
        logits: torch.Tensor,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        allowed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample token indices from logits using temperature, top-k, and top-p.

        If all sampling knobs are disabled (temperature ≤ 0, no top-k, no top-p),
        falls back to greedy argmax decoding.

        Args
        ----
        logits : torch.Tensor
            Logits over vocabulary, shape ``[B*S, V]``.
        top_k : int, optional
            Override for top-k sampling. Uses ``self.top_k`` if ``None``.
        top_p : float, optional
            Override for top-p sampling. Uses ``self.top_p`` if ``None``.

        Returns
        -------
        torch.Tensor
            Sampled token ids of shape ``[B*S, 1]``.
        """
        # Apply vocabulary constraints (if any). This is inference-time only and does not
        # affect training/weights — it just zeroes probability mass for forbidden tokens.
        if allowed_mask is not None or (hasattr(self, "allowed_vocab_mask") and self.allowed_vocab_mask is not None):
            mask = allowed_mask if allowed_mask is not None else self.allowed_vocab_mask
            if mask.dtype != torch.bool:
                mask = mask.bool()
            mask = mask.to(device=logits.device)
            # If any token is forbidden, mask them out.
            if (~mask).any():
                # Use a very negative number instead of -inf to avoid NaNs in some kernels.
                # Use -65504 for FP16 compatibility (smallest finite FP16 value).
                logits = logits.masked_fill(~mask.unsqueeze(0), -65504.0)

        # defaults from model if not provided
        if top_k is None:
            top_k = getattr(self, "top_k", 0)
        if top_p is None:
            top_p = getattr(self, "top_p", 0.0)
        temperature = getattr(self, "temperature", 0.0)

        # Greedy fallback: fully deterministic DiT decoding when all sampling knobs are "off"
        if (temperature is None or temperature <= 0.0) and (not top_k or top_k <= 0) and (not top_p or top_p <= 0.0):
            return logits.argmax(dim=-1, keepdim=True)  # [B,1]

        # Stochastic path
        logits = torch.nan_to_num(logits, neginf=-1e9, posinf=1e9)

        if temperature is not None and temperature > 0.0:
            logits = logits / temperature

        # apply top-k / top-p if requested
        logits = optollama.model.sampling.top_k_top_p_filtering(logits, top_k=int(top_k or 0), top_p=float(top_p or 0.0))

        probs = torch.softmax(logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)

        return torch.multinomial(probs, num_samples=1)  # [B,1]

    def _sample(
        self, spectra: torch.Tensor, eps: float = 1e-3, remask_prob: float = 0.1
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Perform iterative diffusion sampling to generate a token stack.

        Runs a full denoising trajectory from pure mask tokens toward a clean
        stack, applying the learned denoising model at each timestep.

        Args
        ----
        spectra : torch.Tensor
            Conditioning spectra, shape ``[B, D_spec]``.
        eps : float
            Minimum timestep value used for the final denoising steps.
        remask_prob : float
            Base probability of re-masking tokens between updates (overridden
            by the noise schedule during sampling).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor or None]
            A 2-tuple of ``(stacks, step_mae_traj)`` where:

            - ``stacks`` is the final sampled stack, shape ``[B, S]``.
            - ``step_mae_traj`` is a tensor of shape ``[B, steps]`` with
              per-step MAE if ``_step_mae_enabled`` is set, otherwise
              ``None``.
        """
        timesteps = torch.linspace(1.0, eps, self.steps, device=spectra.device)

        initial_state_id = int(self.mask_material_id) if self.material_vocab_mode else int(self.mask)
        stacks = torch.full(
            (spectra.shape[0], self.max_stack_depth),
            initial_state_id,
            dtype=torch.long,
            device=spectra.device,
        )
        thickness_state = self._initial_thickness_state(stacks)

        beta_sched = self.noise(timesteps)

        # Decide whether we track MAE this run
        track_mae = bool(self._step_mae_enabled and (self._step_mae_ctx is not None))
        mae_per_step = []

        for i in range(self.steps):
            t = torch.full((spectra.shape[0],), timesteps[i], device=spectra.device)
            predicted_stacks, factored_outputs = self._sample_outputs(
                spectra,
                stacks,
                t,
                state_thickness_nm=thickness_state,
            )  # [B, S, V]

            b, s, v = predicted_stacks.shape
            logits = predicted_stacks.view(b * s, v)
            samples = self._sample_logits(logits, allowed_mask=self._active_sampling_mask(logits.device))  # [B*S, 1]
            sampled_stacks = samples.view(b, s)

            # --- enforce max emitted length (inference-time constraint) ---
            # Force EOS at position max_len -1 and PAD after, if enabled.
            if self.max_emit_len is not None:
                max_len = int(self.max_emit_len)
                eos_id = int(self.eos_material_id) if self.material_vocab_mode else int(self.eos)
                pad_id = int(self.pad_material_id) if self.material_vocab_mode else int(self.pad)
                # position L-1 forced to EOS
                eos_pos = max_len - 1
                if 0 <= eos_pos < s:
                    sampled_stacks[:, eos_pos] = eos_id
                # positions >= max_len forced to PAD
                if max_len < s:
                    sampled_stacks[:, max_len:] = pad_id

            sampled_thickness = self._continuous_thickness_for_sample(factored_outputs, sampled_stacks)
            if self.continuous_thickness_state_enabled and sampled_thickness is not None:
                next_thickness_state = sampled_thickness
            else:
                next_thickness_state = thickness_state

            if i < self.steps - 1:
                # overwrite remask_prob with noise schedule
                remask_prob = beta_sched[i].item()
                remask = (torch.rand_like(stacks, dtype=spectra.dtype) < remask_prob).bool()

                # Don't allow remasking of forced EOS/PAD positions
                if self.max_emit_len is not None:
                    max_len = int(self.max_emit_len)
                    if 0 < max_len <= s:
                        remask[:, max_len - 1 :] = False

                mask_id = int(self.mask_material_id) if self.material_vocab_mode else int(self.mask)
                stacks = torch.where(remask, mask_id, sampled_stacks)
                thickness_state = next_thickness_state
            else:
                stacks = sampled_stacks
                thickness_state = next_thickness_state

            # ---- optional per-step MAE tracking ----
            if track_mae:
                sim_stacks = (
                    self._tokens_from_material_state(stacks, thickness_state)
                    if self.material_vocab_mode
                    else stacks
                )
                pred_spec = optollama.evaluation.simulation.simulate_token_sequence(
                    sim_stacks,
                    self._step_mae_ctx,
                    eos=self.eos,
                    pad=self.pad,
                    msk=self.mask,
                    thickness_override=thickness_state if self.continuous_thickness_state_enabled else sampled_thickness,
                )  # [B, 3, W]
                step_mae = optollama.evaluation.metrics.masked_mae(spectra, pred_spec)  # [B]
                mae_per_step.append(step_mae)

        step_mae_traj: Optional[torch.Tensor]
        if track_mae and mae_per_step:
            # mae_per_step: list of [B] → [steps, B] → [B, steps]
            step_mae_traj = torch.stack(mae_per_step, dim=0).transpose(0, 1).contiguous()
        else:
            step_mae_traj = None

        if self.continuous_thickness_enabled:
            output_stacks = (
                self._tokens_from_material_state(stacks, thickness_state)
                if self.material_vocab_mode
                else stacks
            )
            return {
                "ids": output_stacks,
                "material_ids": stacks if self.material_vocab_mode else None,
                "thickness_nm": thickness_state if thickness_state is not None else sampled_thickness,
                "mae_traj": step_mae_traj,
            }

        return stacks, step_mae_traj

    def forward(
        self,
        spectra: torch.Tensor,
        stacks: torch.Tensor = None,
        timesteps: torch.Tensor = None,
        return_loss: bool = False,
    ) -> torch.Tensor:
        """
        Unified forward interface.

        If ``timesteps`` is provided, ``stacks`` is interpreted as an already
        noised stack and the model runs one explicit denoising step.

        - If ``stacks`` is provided → run diffusion training step.
        - If ``stacks`` is ``None`` → run autoregressive-free diffusion sampling.

        Args
        ----
        spectra : torch.Tensor
            Conditioning spectra.
        stacks : torch.Tensor, optional
            Ground-truth token stack for training. If ``None``, sampling mode
            is used.

        Returns
        -------
        torch.Tensor
            Training logits of shape ``[B, S, vocab_size]`` when ``stacks``
            is provided, or sampled stacks of shape ``[B, S]`` otherwise.
        """
        if stacks is None:
            return self._sample(spectra)
        if return_loss:
            if self.factored_output_enabled:
                return self._factored_training_loss(spectra, stacks)
            logits = self._train(spectra, stacks)
            log_probs = torch.nn.functional.log_softmax(
                torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0),
                dim=-1,
            )
            loss = torch.nn.NLLLoss(ignore_index=self.pad)(log_probs.view(-1, self.vocab_size), stacks.view(-1))
            return {"loss": loss, "logits": logits}
        if timesteps is not None:
            return self._model(spectra, stacks, timesteps)
        return self._train(spectra, stacks)
