import copy
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as f

from optollama.utils.utils import top_k_top_p_filtering

# ruff: noqa: D101, D102, D103, N801
# mypy: disable-error-code=no-untyped-def


# ===== Original building blocks (unchanged) =====
# code from https://github.com/taigaoma1997/optogpt/blob/cdedef9526bba02c58fa73181802dc7138fa3900/optogpt/core/models/transformer.py#L26
# (optogpt/core/models/transformer.py)
def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding_Tf(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe_pos = torch.mul(position, div_term)
        pe[:, 0::2] = torch.sin(pe_pos)
        pe[:, 1::2] = torch.cos(pe_pos)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = f.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [
            linears(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for linears, x in zip(self.linears, (query, key, value))
        ]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class LayerNorm_Tf(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / torch.sqrt(std**2 + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm_Tf(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x)))


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, n):
        super().__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm_Tf(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return f.log_softmax(self.proj(x), dim=-1)


# ===== Helpers to interface with the OptoGPT =====
def _filter_logits_topk_topp(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
    filter_value: float = -float("inf"),
) -> torch.Tensor:
    """Backward-compatible wrapper around :func:`utils.top_k_top_p_filtering`."""
    return top_k_top_p_filtering(logits, top_k=int(top_k or 0), top_p=float(top_p or 0.0), filter_value=filter_value)


def _subsequent_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create an autoregressive (causal) mask for a sequence of length `seq_len`.

    The mask allows each position to attend to itself and all previous
    positions, but not to any future positions.

    Args:
        sz: Target sequence length.
        device: Device on which to allocate the mask.

    Returns
    -------
        Boolean tensor of shape [1, seq_len, seq_len] where True indicates allowed
        attention positions.
    """
    return torch.triu(torch.ones(1, seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1) == 0


def _flatten_spectra(spectra: torch.Tensor) -> torch.Tensor:
    """
    Flatten spectra into a 2D [B, D] representation for the MLP encoder.

    Supports either:
      • [B, 3, W]  → flattened to [B, 3*W]
      • [B, D]     → returned unchanged

    Args:
        spectra: Input spectra tensor.

    Returns
    -------
        Flattened spectra tensor of shape [B, D].

    Raises
    ------
        ValueError: If the input rank/shape is not supported.
    """
    # [B,W,3] -> [B,3W]
    if spectra.dim() == 3 and spectra.size(1) == 3:
        b, c, w = spectra.shape
        return spectra.reshape(b, c * w)
    elif spectra.dim() == 2:
        return spectra
    raise ValueError(f"Unexpected spectra shape {tuple(spectra.shape)}")


class OriginalDecoderWrapper(nn.Module):
    """
    Wrapper around the original OptoGPT decoder-only model.

    This exposes a unified API compatible with the rest of this repo:

        forward(spectra, stacks=None) -> logits [B, S, V] or (logits, None)

    When `stacks` are provided, teacher forcing is used (train mode). When
    `stacks` is None, greedy / stochastic decoding is performed up to
    `max_len`, optionally using temperature, top-k, and top-p sampling.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        max_len: int,
        mask_idx: int,
        pad_idx: int,
        eos_idx: int,
        spectrum_flat_dim: int,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 0.0,
    ):
        """
        Initialize the wrapped OptoGPT-style decoder.

        Args:
            vocab_size: Size of the token vocabulary.
            d_model: Transformer hidden size.
            n_layers: Number of decoder layers.
            n_heads: Number of attention heads.
            d_ff: Width of the feed-forward sublayers.
            dropout: Dropout probability.
            max_len: Maximum number of generated tokens in a sequence.
            mask_idx: Index of the special <MASK> token.
            pad_idx: Index of the <PAD> token.
            eos_idx: Index of the <EOS> token.
            spectrum_flat_dim: Flattened spectral feature dimension (input to fc).
            temperature: Sampling temperature (0.0 = deterministic).
            top_k: Top-k sampling cutoff.
            top_p: Top-p (nucleus) sampling cutoff.
        """
        super().__init__()
        self.vocab = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.mask_idx = mask_idx
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        # Build ORIGINAL parts (mirrors make_model_I)
        attn = MultiHeadedAttention(n_heads, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        pos = PositionalEncoding_Tf(d_model, dropout)
        self.fc = nn.Sequential(nn.Linear(spectrum_flat_dim, d_model))  # minimal: one linear as in original
        self.decoder = Decoder(
            DecoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(attn), copy.deepcopy(ff), dropout), n_layers
        )
        self.tgt_embed = nn.Sequential(Embeddings(d_model, vocab_size), pos)
        self.generator = Generator(d_model, vocab_size)  # we'll call .proj for logits

    def _decode_logits(self, memory: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
        """
        Run the decoder stack and return raw logits for target tokens.

        Args:
            memory: Encoded spectrum features, shape [B, 1, D].
            tgt_tokens: Target token ids, shape [B, S].

        Returns
        -------
            Logits over the vocabulary, shape [B, S, V].
        """
        # memory: [B,1,D], tgt_tokens: [B,S]
        x = self.tgt_embed(tgt_tokens)
        tgt_mask = _subsequent_mask(tgt_tokens.size(1), x.device)
        y = self.decoder(x, memory, src_mask=None, tgt_mask=tgt_mask)
        return self.generator.proj(y)  # raw logits [B,S,V]

    def forward(
        self, spectra: torch.Tensor, stacks: torch.Tensor = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Forward pass in either teacher-forcing or autoregressive mode.

        Args:
            spectra: Conditioning spectra, shape [B, 3, W] or [B, D_flat].
            stacks: Optional ground-truth token stacks, shape [B, S].
                If provided, teacher forcing is used and logits for the full
                sequence are returned. If None, tokens are generated step by
                step until `max_len`.

        Returns
        -------
            If `stacks` is not None:
                Logits of shape [B, S, V].
            If `stacks` is None:
                A tuple (logits, None), where logits has shape [B, max_len, V].
        """
        # 1) spectrum -> flatten -> fc -> memory [B,1,D]
        src_vec = _flatten_spectra(spectra)
        memory = self.fc(src_vec).unsqueeze(1)

        if stacks is not None:
            # Teacher forcing: shift-right with <MSK>
            b, s = stacks.shape
            tgt_in = torch.full_like(stacks, self.mask_idx)
            tgt_in[:, 1:] = stacks[:, :-1]
            return self._decode_logits(memory, tgt_in)

        # Greedy inference up to max_len
        b = spectra.size(0)
        out = torch.full((b, 1), self.mask_idx, dtype=torch.long, device=spectra.device)
        logits_accum = []
        for t in range(self.max_len):
            logits_t = self._decode_logits(memory, out)  # [B,t+1,V]
            last = logits_t[:, -1, :]  # [B,V]

            # defaults from model if not provided
            top_k = getattr(self, "top_k", 0)
            top_p = getattr(self, "top_p", 0.0)
            temperature = getattr(self, "temperature", 0.0)

            last = torch.nan_to_num(last, neginf=-1e9, posinf=1e9)

            # Greedy fallback: fully deterministic DiT decoding when all sampling knobs are "off"
            if (temperature is None or temperature <= 0.0) and (not top_k or top_k <= 0) and (not top_p or top_p <= 0.0):
                next_id = last.argmax(dim=-1, keepdim=True)
            else:
                # Stochastic path
                if temperature is not None and temperature > 0.0:
                    logits_adj = last / temperature
                else:
                    logits_adj = last

                # apply top-k / top-p if requested
                logits_adj = _filter_logits_topk_topp(logits_adj, top_k=top_k or 0, top_p=top_p or 0.0)
                probs = torch.softmax(logits_adj, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)  # [B,1]

            logits_accum.append(last.unsqueeze(1))  # keep [B,1,V] for return
            out = torch.cat([out, next_id], dim=1)

        return torch.cat(logits_accum, dim=1), None
