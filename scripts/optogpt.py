import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== Original building blocks (unchanged) =====
# code from https://github.com/taigaoma1997/optogpt/blob/cdedef9526bba02c58fa73181802dc7138fa3900/optogpt/core/models/transformer.py#L26
# (optogpt/core/models/transformer.py)
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


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
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe_pos   = torch.mul(position, div_term)
        pe[:, 0::2] = torch.sin(pe_pos)
        pe[:, 1::2] = torch.cos(pe_pos)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
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
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
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
        std  = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / torch.sqrt(std ** 2 + self.eps) + self.b_2


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
        self.src_attn  = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
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
        return F.log_softmax(self.proj(x), dim=-1)


# ===== Helpers to interface with the OptoGPT =====
def _filter_logits_topk_topp(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0, filter_value: float = -float('inf')):
    """
    logits: [B,V] -> mask tokens outside top-k / nucleus (top-p) by setting to -inf.
    """
    logits = torch.nan_to_num(logits, neginf=-1e9, posinf=1e9)

    if top_k and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1].unsqueeze(-1)  # [B,1]
        logits = torch.where(logits < kth, torch.full_like(logits, filter_value), logits)

    if top_p and top_p > 0.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        remove = cumprobs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        mask = torch.zeros_like(remove, dtype=torch.bool).scatter(1, sorted_idx, remove)
        logits = logits.masked_fill(mask, filter_value)

    return logits


def _subsequent_mask(sz: int, device: torch.device):
    return torch.triu(torch.ones(1, sz, sz, device=device, dtype=torch.bool), diagonal=1) == 0


def _flatten_spectra(spectra: torch.Tensor) -> torch.Tensor:
    # [B,W,3] -> [B,3W]
    if spectra.dim() == 3 and spectra.size(1) == 3:
        B, C, W = spectra.shape
        return spectra.reshape(B, C * W)
    elif spectra.dim() == 2:
        return spectra
    raise ValueError(f"Unexpected spectra shape {tuple(spectra.shape)}")


class OriginalDecoderWrapper(nn.Module):
    """
    Wraps the ORIGINAL decoder-only stack so it matches our repo API:
      forward(spectra, stacks=None) -> logits [B,S,V]
    """
    def __init__(self, *, vocab_size: int, d_model: int, n_layers: int,
                 n_heads: int, d_ff: int, dropout: float,
                 max_len: int, mask_idx: int, pad_idx: int, eos_idx: int,
                 spectrum_flat_dim: int):
        super().__init__()
        self.vocab = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.mask_idx = mask_idx
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx
        self.sample_temperature = 0.0
        self.sample_top_k = 0
        self.sample_top_p = 0.0


        # Build ORIGINAL parts (mirrors make_model_I)
        attn = MultiHeadedAttention(n_heads, d_model, dropout)
        ff   = PositionwiseFeedForward(d_model, d_ff, dropout)
        pos  = PositionalEncoding_Tf(d_model, dropout)
        self.fc = nn.Sequential(nn.Linear(spectrum_flat_dim, d_model))  # minimal: one linear as in original
        self.decoder = Decoder(DecoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(attn), copy.deepcopy(ff), dropout), n_layers)
        self.tgt_embed = nn.Sequential(Embeddings(d_model, vocab_size), pos)
        self.generator = Generator(d_model, vocab_size)  # we'll call .proj for logits
    
    def set_sampling(self, temperature: float = 0.0, top_k: int = 0, top_p: float = 0.0):
        self.sample_temperature = float(temperature)
        self.sample_top_k = int(top_k)
        self.sample_top_p = float(top_p)

    def _decode_logits(self, memory: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
        # memory: [B,1,D], tgt_tokens: [B,S]
        x = self.tgt_embed(tgt_tokens)
        tgt_mask = _subsequent_mask(tgt_tokens.size(1), x.device)
        y = self.decoder(x, memory, src_mask=None, tgt_mask=tgt_mask)
        return self.generator.proj(y)  # raw logits [B,S,V]

    def forward(self, spectra: torch.Tensor, stacks: torch.Tensor = None):
        # 1) spectrum -> flatten -> fc -> memory [B,1,D]
        src_vec = _flatten_spectra(spectra)
        memory = self.fc(src_vec).unsqueeze(1)

        if stacks is not None:
            # Teacher forcing: shift-right with <MSK>
            B, S = stacks.shape
            tgt_in = torch.full_like(stacks, self.mask_idx)
            tgt_in[:, 1:] = stacks[:, :-1]
            return self._decode_logits(memory, tgt_in)

        # Greedy inference up to max_len
        B = spectra.size(0)
        out = torch.full((B, 1), self.mask_idx, dtype=torch.long, device=spectra.device)
        logits_accum = []
        for t in range(self.max_len):
            logits_t = self._decode_logits(memory, out)   # [B,t+1,V]
            last = logits_t[:, -1, :]  # [B,V]
            
            # defaults from model if not provided
            top_k = getattr(self, "sample_top_k", 0)
            top_p = getattr(self, "sample_top_p", 0.0)
            temperature = getattr(self, "sample_temperature", 0.0)
            
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
                
            logits_accum.append(last.unsqueeze(1))                 # keep [B,1,V] for return
            out = torch.cat([out, next_id], dim=1)

        return torch.cat(logits_accum, dim=1), None