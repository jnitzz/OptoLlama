"""Open-vocabulary discrete material flow coupled to continuous layer geometry."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, NamedTuple

import torch
from torch import nn
from torch.nn import functional

from .open_layer_flow import (
    MaterialCurveEncoder,
    OpenLayerDecoderBlock,
    OpenLayerFlowConfig,
    SinusoidalTimeEmbedding,
    TargetQueryEncoder,
    WavelengthFeatures,
)
from .open_vocab_depth_field_dfm import dfm_step_flip_probability


@dataclass(frozen=True)
class OpenLayerJointDFMConfig:
    """Serializable network and thickness geometry settings for the joint model."""

    max_layers: int = 100
    max_candidates: int = 24
    d_model: int = 512
    n_blocks: int = 12
    n_heads: int = 8
    ffn_multiplier: float = 4.0
    query_encoder_blocks: int = 2
    query_points: int = 256
    dropout: float = 0.0
    fp32_attention: bool = True
    min_thickness_nm: float = 1.0
    max_total_nm: float = 10000.0
    endpoint_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        """Validate the slot and thickness domains."""
        if self.max_layers < 1 or self.max_candidates < 1 or self.query_points < 4:
            raise ValueError("Require positive layer and candidate counts, and at least four query points.")
        if not 0 < self.min_thickness_nm < self.max_total_nm / self.max_layers:
            raise ValueError("Thickness budget must exceed the minimum for MAX_LAYERS.")
        if not 0 < self.endpoint_epsilon < 0.01:
            raise ValueError("endpoint_epsilon must be in (0, 0.01).")

    def to_dict(self) -> dict[str, Any]:
        """Return the architecture as checkpoint metadata."""
        return asdict(self)

    @classmethod
    def from_mapping(cls, block: dict[str, Any]) -> OpenLayerJointDFMConfig:
        """Read the joint model's dedicated YAML section."""
        model, bank, query, geometry = (block.get(key) or {} for key in ("MODEL", "MATERIAL_BANK", "QUERY", "THICKNESS"))
        if not query.get("ENABLED", True):
            raise ValueError("Joint layer DFM requires coordinate queries.")
        return cls(
            max_layers=int(block.get("MAX_LAYERS", 100)),
            max_candidates=int(bank.get("MAX_CANDIDATES", 24)),
            d_model=int(model.get("D_MODEL", 512)),
            n_blocks=int(model.get("N_BLOCKS", 12)),
            n_heads=int(model.get("N_HEADS", 8)),
            ffn_multiplier=float(model.get("FFN_MULTIPLIER", 4.0)),
            query_encoder_blocks=int(model.get("QUERY_ENCODER_BLOCKS", 2)),
            query_points=int(query.get("POINTS", 256)),
            dropout=float(model.get("DROPOUT", 0.0)),
            fp32_attention=bool(model.get("FP32_ATTENTION", True)),
            min_thickness_nm=float(geometry.get("MIN_NM", 1.0)),
            max_total_nm=float(geometry.get("MAX_TOTAL_NM", 10000.0)),
            endpoint_epsilon=float(geometry.get("ENDPOINT_EPSILON", 1e-6)),
        )


class JointLayerSample(NamedTuple):
    """Continuous physical stacks with separate generated slot counts."""

    material_ids: torch.Tensor
    thickness_nm: torch.Tensor
    layer_counts: torch.Tensor
    layer_mask: torch.Tensor


class OpenLayerJointDFM(nn.Module):
    """Generate candidates by sharing one decoder between categorical jumps and geometry flow."""

    mask_id = -1
    pad_id = -100

    def __init__(self, config: OpenLayerJointDFMConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        shared = OpenLayerFlowConfig(
            target_channels=3,
            max_layers=config.max_layers,
            d_model=d,
            n_blocks=config.n_blocks,
            n_heads=config.n_heads,
            ffn_multiplier=config.ffn_multiplier,
            query_encoder_blocks=config.query_encoder_blocks,
            dropout=config.dropout,
            adaln_zero=True,
            adaln_shift_limit=2.0,
            adaln_scale_limit=1.0,
            adaln_gate_limit=1.0,
        )
        coordinates = WavelengthFeatures(shared.wavelength_scale_nm, shared.wavelength_fourier_bands)
        self.target_encoder = TargetQueryEncoder(shared, coordinates, fp32_attention=config.fp32_attention)
        self.material_encoder = MaterialCurveEncoder(shared, coordinates)
        self.angle_encoder = nn.Linear(2, d)
        self.polarization_embedding = nn.Embedding(2, d)
        self.position = nn.Parameter(torch.randn(config.max_layers + 1, d) * 0.02)
        self.mask_embedding = nn.Parameter(torch.randn(d) * 0.02)
        self.thickness_embedding = nn.Sequential(nn.Linear(3, d), nn.SiLU(), nn.Linear(d, d))
        self.total_embedding = nn.Linear(1, d)
        self.time_encoder = nn.Sequential(SinusoidalTimeEmbedding(d), nn.Linear(d, d), nn.SiLU(), nn.LayerNorm(d))
        self.count_head = nn.Sequential(nn.LayerNorm(3 * d), nn.Linear(3 * d, d), nn.SiLU(), nn.Linear(d, config.max_layers + 1))
        self.blocks = nn.ModuleList(
            OpenLayerDecoderBlock(shared, fp32_attention=config.fp32_attention) for _ in range(config.n_blocks)
        )
        self.output_norm = nn.LayerNorm(d)
        self.pointer_query = nn.Linear(d, d, bias=False)
        self.pointer_key = nn.Linear(d, d, bias=False)
        self.thickness_velocity = nn.Linear(d, 1)
        self.total_velocity = nn.Linear(d, 1)

    def encode_thickness(self, thickness_nm: torch.Tensor, layer_mask: torch.Tensor) -> torch.Tensor:
        """Map native lengths to centered log allocations and a bounded-total logit."""
        count = layer_mask.sum(dim=1, keepdim=True)
        cap = self.config.max_total_nm
        minimum = self.config.min_thickness_nm
        eps = self.config.endpoint_epsilon
        excess = (thickness_nm.float() - minimum).clamp_min(0).masked_fill(~layer_mask, 0)
        total = excess.sum(dim=1, keepdim=True)
        log_share = (excess / total.clamp_min(eps)).clamp_min(eps).log().masked_fill(~layer_mask, 0)
        centered = (log_share - log_share.sum(dim=1, keepdim=True) / count.clamp_min(1)).masked_fill(~layer_mask, 0)
        fraction = total / (cap - minimum * count)
        logit = torch.logit(fraction, eps=eps).masked_fill(count == 0, 0)
        return torch.cat((logit, centered), dim=1)

    def decode_thickness(self, state: torch.Tensor, layer_mask: torch.Tensor) -> torch.Tensor:
        """Decode one budgeted total plus positive allocations, leaving padding at zero."""
        total, allocation = state.float().split((1, layer_mask.shape[1]), dim=1)
        minimum = self.config.min_thickness_nm
        budget = self.config.max_total_nm - minimum * layer_mask.sum(dim=1, keepdim=True)
        # Leave enough budget for roundoff in the FP32 allocations and their sum.
        margin = max(self.config.endpoint_epsilon, 4 * torch.finfo(total.dtype).eps)
        excess = budget * total.sigmoid().clamp_max(1 - margin)
        # Finite masking avoids NaNs for count-zero rows. Their decoded allocations are zeroed below.
        weights = allocation.masked_fill(~layer_mask, torch.finfo(allocation.dtype).min).softmax(dim=1)
        return (minimum + excess * weights).masked_fill(~layer_mask, 0)

    @staticmethod
    def state_mask(layer_mask: torch.Tensor) -> torch.Tensor:
        """Mask both the total latent and layer latents on empty or padded positions."""
        return torch.cat((layer_mask.any(dim=1, keepdim=True), layer_mask), dim=1)

    def encode_condition(
        self,
        *,
        spectra: torch.Tensor,
        wavelengths_nm: torch.Tensor,
        candidate_nk: torch.Tensor,
        candidate_mask: torch.Tensor,
        incidence_angle_deg: torch.Tensor,
        polarization_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode cached optical memories and the target-conditioned count posterior."""
        query_mask = torch.ones_like(wavelengths_nm, dtype=torch.bool)
        radians = torch.deg2rad(incidence_angle_deg.float())
        optical = self.angle_encoder(torch.stack((radians.sin(), radians.cos()), dim=-1))
        optical = optical + self.polarization_embedding(polarization_id.long())
        target = self.target_encoder(wavelengths_nm, spectra.transpose(1, 2), query_mask)
        materials = self.material_encoder(wavelengths_nm, candidate_nk, query_mask, candidate_mask)
        candidate_mean = materials.sum(dim=1) / candidate_mask.sum(dim=1, keepdim=True).clamp_min(1)
        counts = self.count_head(torch.cat((target.mean(dim=1), candidate_mean, optical), dim=-1))
        return target + optical[:, None], materials, counts

    def forward(
        self,
        *,
        material_ids: torch.Tensor,
        thickness_state: torch.Tensor,
        layer_mask: torch.Tensor,
        flow_time: torch.Tensor,
        candidate_mask: torch.Tensor,
        encoded_condition: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        **condition: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict both endpoints from the same joint state, never from clean inactive labels."""
        target, materials, counts = encoded_condition or self.encode_condition(candidate_mask=candidate_mask, **condition)
        _, layers = material_ids.shape
        dim = self.config.d_model
        ids = material_ids.masked_fill(~layer_mask, 0).clamp_min(0).unsqueeze(-1).expand(-1, -1, dim)
        material = materials.gather(1, ids)
        material = torch.where((material_ids == self.mask_id).unsqueeze(-1), self.mask_embedding.to(material.dtype), material)
        state = thickness_state.masked_fill(~self.state_mask(layer_mask), 0)
        thickness = self.decode_thickness(state, layer_mask)
        cap = self.config.max_total_nm
        center = thickness.cumsum(dim=1) - 0.5 * thickness
        geom = torch.stack((state[:, 1:], thickness / cap, center / cap), dim=-1)
        layer = (material + self.thickness_embedding(geom)).masked_fill(~layer_mask.unsqueeze(-1), 0)
        # The always-visible total token also makes self-attention well defined for empty stacks.
        x = torch.cat((self.total_embedding(state[:, :1]).unsqueeze(1), layer), dim=1)
        x = x + self.position[: layers + 1].to(x.dtype)
        time = self.time_encoder(flow_time.float()).to(x.dtype)
        padding = functional.pad(~layer_mask, (1, 0), value=False)
        for block in self.blocks:
            x = block(
                x,
                time_embedding=time,
                layer_padding_mask=padding,
                target_memory=target,
                query_padding_mask=None,
                material_memory=materials,
                candidate_padding_mask=~candidate_mask,
            )
        x = self.output_norm(x)
        logits = torch.einsum("bld,bmd->blm", self.pointer_query(x[:, 1:]), self.pointer_key(materials)) / math.sqrt(dim)
        logits = logits.masked_fill(~candidate_mask[:, None], torch.finfo(logits.dtype).min)
        velocity = torch.cat((self.total_velocity(x[:, 0]), self.thickness_velocity(x[:, 1:]).squeeze(-1)), dim=1)
        return {
            "material_logits": logits,
            "thickness_velocity": velocity.masked_fill(~self.state_mask(layer_mask), 0),
            "count_logits": counts,
        }

    def training_state(
        self,
        batch: dict[str, torch.Tensor],
        flow_time: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draw the material mixture state and continuous linear path at the same time."""
        mask = batch["layer_mask"]
        clean = batch["material_targets"]
        s = flow_time.float()[:, None]
        corrupted = (torch.rand(clean.shape, device=clean.device, generator=generator) < (1 - s).square()) & mask
        material = torch.where(corrupted, self.mask_id, clean).masked_fill(~mask, self.pad_id)
        endpoint = self.encode_thickness(batch["thickness_nm"], mask)
        noise = torch.randn(endpoint.shape, device=endpoint.device, generator=generator).masked_fill(~self.state_mask(mask), 0)
        return material, (1 - s) * noise + s * endpoint, endpoint - noise, corrupted

    @staticmethod
    def _categorical(
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        deterministic: bool,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("Non-finite joint layer DFM logits.")
        if deterministic or temperature == 0:
            return logits.argmax(dim=-1)
        scaled = logits.float() / max(temperature, 1e-6)
        if 0 < top_k < logits.shape[-1]:
            keep = scaled.topk(top_k, dim=-1).indices
            scaled = torch.full_like(scaled, -torch.inf).scatter(-1, keep, scaled.gather(-1, keep))
        probabilities = scaled.softmax(dim=-1).reshape(-1, scaled.shape[-1])
        return torch.multinomial(probabilities, num_samples=1, generator=generator).reshape(logits.shape[:-1])

    @torch.no_grad()
    def sample(
        self,
        *,
        steps: int = 16,
        temperature: float = 1.0,
        top_k: int = 0,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
        **condition: torch.Tensor,
    ) -> JointLayerSample:
        """Sample a count, then co-evolve material jumps and thickness latents."""
        if steps < 1 or not math.isfinite(temperature) or temperature < 0 or top_k < 0:
            raise ValueError("Sampling requires positive steps and finite nonnegative temperature/top-k.")
        bank = condition["candidate_mask"]
        if not bool(bank.any(dim=1).all()):
            raise ValueError("Every candidate needs a nonempty material bank.")
        cached = self.encode_condition(**condition)
        greedy = deterministic or temperature == 0
        count = self._categorical(cached[2], temperature, 0, greedy, generator)
        layers = max(1, int(count.max().item()))
        mask = torch.arange(layers, device=count.device)[None] < count[:, None]
        material = torch.full(mask.shape, self.mask_id, device=count.device, dtype=torch.long).masked_fill(~mask, self.pad_id)
        state = torch.randn((len(count), layers + 1), device=count.device, generator=generator)
        state = state.masked_fill(~self.state_mask(mask), 0)
        for step in range(steps):
            flow_time = torch.full_like(count, step / steps, dtype=torch.float32)
            out = self(
                material_ids=material,
                thickness_state=state,
                layer_mask=mask,
                flow_time=flow_time,
                candidate_mask=bank,
                encoded_condition=cached,
            )
            proposal = self._categorical(out["material_logits"], temperature, top_k, greedy, generator)
            probability = dfm_step_flip_probability(step, steps)
            if greedy:
                if probability >= 0.5:
                    material = torch.where(mask, proposal, material)
            else:
                flip = torch.rand(mask.shape, device=mask.device, generator=generator) < probability
                material = torch.where(mask & flip, proposal, material)
            state = state + out["thickness_velocity"].float() / steps
        if not bool(torch.isfinite(state).all()):
            raise FloatingPointError("Non-finite thickness state after joint layer DFM sampling.")
        return JointLayerSample(material, self.decode_thickness(state, mask), count, mask)
