# -*- coding: utf-8 -*-
"""
Created on Mon May  5 14:53:56 2025

@author: a3536
"""
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: Optional[torch.Tensor] = 512) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        pe = self.pe[:, :seq_len, :]
        return x + pe.to(x.device)


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, hidden_dim: int, dropout: float) -> None:
        super(DecoderBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, d_model)
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.norm3   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, spectrum_embedding: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-Attn
        x2 = self.norm1(x)
        attn_output, _ = self.self_attn(
            query=x2, key=x2, value=x2,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        x = x + self.dropout(attn_output)

        # Cross-Attn
        x2 = self.norm2(x)
        attn_output, _ = self.cross_attn(
            query=x2,
            key=spectrum_embedding,
            value=spectrum_embedding
        )
        x = x + self.dropout(attn_output)

        # Feedforward
        x2         = self.norm3(x)
        ffn_output = self.linear2(F.relu(self.linear1(x2)))
        x          = x + self.dropout(ffn_output)
        return x


class OptoLlama(nn.Module):
    """
    We'll define 'num_classes' as the size of the "merged" vocabulary
    (20 materials + 3 special tokens, etc.). The model can output either
    a real material index or a special token like <EOS>.
    """
    def __init__(self, num_classes: int, d_model: int, n_heads: int, n_layers: int, max_seq_length: int, input_dim: int, dropout: float) -> None:
        super(OptoLlama, self).__init__()
        self.n_heads       = n_heads
        self.embedding     = nn.Embedding(num_classes, d_model)
        self.pos_encoding  = PositionalEncoding(d_model, max_seq_length)

        self.spectrum_embedding = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )

        hidden_dim = 4 * d_model
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, hidden_dim, dropout)
            for _ in range(n_layers)
        ])

        # final heads
        self.material_head  = nn.Linear(d_model, num_classes)
        self.thickness_head = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, spectrum: torch.Tensor, mat_input: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        mat_input: [B, seq_len], each entry in [0..num_classes-1].
        """
        x = self.embedding(mat_input)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        spec_emb = self.spectrum_embedding(spectrum).unsqueeze(1)

        for layer in self.layers:
            x = layer(x, spec_emb, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        mat_logits   = self.material_head(x)          # [B, seq_len, num_classes]
        thick_logits = self.thickness_head(x).squeeze(-1)  # [B, seq_len]
        return mat_logits, thick_logits