from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectrumEncoder(nn.Module):
    """
    Encode a RAT spectrum into a compact latent vector.
    """

    def __init__(self, spectra_shape: tuple[int, int], d_model: int, dropout: float) -> None:
        super().__init__()
        self.spectra_shape = tuple(int(v) for v in spectra_shape)
        self.input_dim = int(self.spectra_shape[0] * self.spectra_shape[1])
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        """
        Return encoded spectra of shape ``[B, d_model]``.
        """
        return self.net(spectra.reshape(spectra.size(0), -1).to(torch.float32))


class StackStateEncoder(nn.Module):
    """
    Encode a discrete stack token sequence into a latent vector.
    """

    def __init__(
        self,
        vocab_size: int,
        max_stack_depth: int,
        eos_idx: int,
        pad_idx: int,
        msk_idx: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.eos = int(eos_idx)
        self.pad = int(pad_idx)
        self.msk = int(msk_idx)
        self.max_stack_depth = int(max_stack_depth)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position = nn.Parameter(torch.zeros(1, self.max_stack_depth, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def valid_token_mask(self, stacks: torch.Tensor) -> torch.Tensor:
        """
        Mark tokens up to and including the first EOS, excluding PAD/MSK.
        """
        is_eos = stacks == self.eos
        eos_seen = is_eos.cumsum(dim=1)
        before_first_eos = eos_seen == 0
        first_eos = is_eos & (eos_seen == 1)
        valid = (before_first_eos | first_eos) & (stacks != self.pad) & (stacks != self.msk)
        empty = valid.sum(dim=1, keepdim=True) == 0
        return torch.where(empty, torch.ones_like(valid), valid)

    def forward(self, stacks: torch.Tensor) -> torch.Tensor:
        """
        Return encoded stacks of shape ``[B, d_model]``.
        """
        if stacks.size(1) > self.max_stack_depth:
            raise ValueError(
                f"Stack length {stacks.size(1)} exceeds model max_stack_depth={self.max_stack_depth}."
            )

        stacks = stacks.to(torch.long)
        valid = self.valid_token_mask(stacks)
        x = self.embedding(stacks) + self.position[:, : stacks.size(1)]
        x = self.encoder(x, src_key_padding_mask=~valid)
        x = self.norm(x)
        weights = valid.unsqueeze(-1).to(x.dtype)
        return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class WorldEditScorer(nn.Module):
    """
    Learned local world model for ranking candidate stack edits.

    The model receives a target spectrum, the current simulated spectrum, the
    current stack, and a candidate next stack. It predicts the candidate's
    next-spectrum proxy, absolute next MAE, and MAE delta. The final TMM solver
    remains the source of truth; this model is intended as a cheap proposal
    scorer inside planning loops.
    """

    def __init__(
        self,
        spectra_shape: tuple[int, int],
        vocab_size: int,
        max_stack_depth: int,
        eos_idx: int,
        pad_idx: int,
        msk_idx: int,
        d_model: int = 256,
        n_heads: int = 4,
        stack_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.spectra_shape = tuple(int(v) for v in spectra_shape)
        self.spectrum_encoder = SpectrumEncoder(self.spectra_shape, d_model=d_model, dropout=dropout)
        self.stack_encoder = StackStateEncoder(
            vocab_size=vocab_size,
            max_stack_depth=max_stack_depth,
            eos_idx=eos_idx,
            pad_idx=pad_idx,
            msk_idx=msk_idx,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=stack_layers,
            dropout=dropout,
        )

        fused_dim = 5 * d_model
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 4 * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, 2 * d_model),
            nn.SiLU(),
            nn.LayerNorm(2 * d_model),
        )
        self.cost_head = nn.Linear(2 * d_model, 1)
        self.delta_head = nn.Linear(2 * d_model, 1)
        self.spectrum_head = nn.Linear(2 * d_model, int(self.spectra_shape[0] * self.spectra_shape[1]))

    def forward(
        self,
        target_spectra: torch.Tensor,
        current_spectra: torch.Tensor,
        current_stacks: torch.Tensor,
        next_stacks: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Predict candidate edit consequences.
        """
        target_z = self.spectrum_encoder(target_spectra)
        current_spec_z = self.spectrum_encoder(current_spectra)
        residual_z = self.spectrum_encoder(target_spectra - current_spectra)
        current_stack_z = self.stack_encoder(current_stacks)
        next_stack_z = self.stack_encoder(next_stacks)

        h = self.fusion(torch.cat([target_z, current_spec_z, residual_z, current_stack_z, next_stack_z], dim=-1))
        next_spectrum = torch.sigmoid(self.spectrum_head(h)).view(target_spectra.size(0), *self.spectra_shape)
        return {
            "cost_after": F.softplus(self.cost_head(h)).squeeze(-1),
            "delta_mae": self.delta_head(h).squeeze(-1),
            "next_spectra": next_spectrum,
        }

    def loss(
        self,
        batch: dict[str, torch.Tensor],
        spectrum_weight: float = 0.1,
        cost_weight: float = 1.0,
        delta_weight: float = 1.0,
        roi_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute training loss for a batch of transition rows.
        """
        pred = self(
            batch["target_spectra"],
            batch["current_spectra"],
            batch["current_stacks"],
            batch["next_stacks"],
        )
        target_cost = batch["cost_after"].to(torch.float32)
        target_delta = (batch["cost_after"] - batch["cost_before"]).to(torch.float32)
        target_spectrum = batch["next_spectra"].to(torch.float32)

        cost_loss = F.mse_loss(pred["cost_after"], target_cost)
        delta_loss = F.mse_loss(pred["delta_mae"], target_delta)
        if roi_mask is not None:
            mask = roi_mask.to(device=target_spectrum.device, dtype=torch.bool).view(-1)
            spectrum_loss = F.l1_loss(pred["next_spectra"][..., mask], target_spectrum[..., mask])
        else:
            spectrum_loss = F.l1_loss(pred["next_spectra"], target_spectrum)

        total = cost_weight * cost_loss + delta_weight * delta_loss + spectrum_weight * spectrum_loss
        parts = {
            "loss": total.detach(),
            "cost_loss": cost_loss.detach(),
            "delta_loss": delta_loss.detach(),
            "spectrum_loss": spectrum_loss.detach(),
        }
        return total, parts
