import math
import torch
import copy


class LinearNoise(torch.nn.Module):
    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.eps) * timesteps


class SquareNoise(torch.nn.Module):
    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # Normalize timesteps to [0, 1] if they aren't already
        t = timesteps.clamp(0, 1)
        return (1.0 - self.eps) * t**2


class ZeroTerminalNoise(torch.nn.Module):
    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        betas = (1.0 - self.eps) * timesteps**2
        alphas = 1.0 - betas
        # Convert alphas to alphas_bar_sqrt
        alphas_bar = alphas.cumprod(0)
        alphas_bar_sqrt = alphas_bar.sqrt()
        
        # Store old values.
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
        # Shift so last timestep is zero.
        alphas_bar_sqrt -= alphas_bar_sqrt_T
        # Scale so first timestep is back to old value.
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)
        
        # Convert alphas_bar_sqrt to betas
        alphas_bar = alphas_bar_sqrt ** 2
        alphas = alphas_bar[1:] / alphas_bar[:-1]
        alphas = torch.cat([alphas_bar[0:1], alphas])
        betas = 1.0 - alphas
        return betas


class SpectrumEmbedding(torch.nn.Module):
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


class PositionalEncoding(torch.nn.Module):
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
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:        
        return self.pe[:x.shape[1]]


class LearnedPositionalEncoding(torch.nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pe = torch.nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, D]
        S = x.size(1)
        pos = torch.arange(S, device=x.device)
        return self.pe(pos).unsqueeze(0).expand(x.size(0), S, -1)


class StackEmbedding(torch.nn.Module):
    def __init__(self, input_vocab: int, d_model: int) -> None:
        super().__init__()
        
        self.stack_embedding = torch.nn.Embedding(input_vocab, d_model)

    def forward(self, stacks: torch.Tensor) -> torch.Tensor:
        return self.stack_embedding(stacks)


class TimestepEmbedding(torch.nn.Module):
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
        sigmas = -torch.log(timesteps)
        projected = sigmas.unsqueeze(-1) * frequencies.unsqueeze(0)
        
        return torch.cat([torch.cos(projected), torch.sin(projected)], dim=-1)
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        frequencies = self.timestep_embedding(timesteps)
        
        return self.embedding(frequencies).unsqueeze(1)


class AdaLayerNormGaussian(torch.nn.Module):
    def __init__(self, hidden_size: int, cond_dim: int,
                 std_gamma: float = 1.2e-3, std_beta: float = 8e-4):
        super().__init__()
        self.eps = 1e-5
        # no affine params inside the LN itself
        self.to_scale_shift = torch.nn.Linear(cond_dim, 2 * hidden_size, bias=True)

        # --- Gaussian init (key difference from adaLN-Zero) ---
        torch.nn.init.normal_(self.to_scale_shift.weight[:hidden_size], 0.0, std_gamma)  # Δγ
        torch.nn.init.normal_(self.to_scale_shift.weight[hidden_size:], 0.0, std_beta)   # β
        torch.nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x, cond):
        mu  = x.mean(-1, keepdim=True)
        sig  = x.var(-1, unbiased=False, keepdim=True).add(self.eps).sqrt()
        x_hat  = (x - mu) / sig                         # weight-free LN

        delta_g, beta = self.to_scale_shift(cond).chunk(2, dim=-1)   # [B, H] each
        y = x_hat * (1 + delta_g).unsqueeze(1) + beta.unsqueeze(1)
        return y


class PositionBias(torch.nn.Module):
    """
    Learnable bias per sequence position.
    Added to stack embeddings so early tokens can be weighted more.
    """
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(max_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, D]
        S = x.size(1)
        return x + self.bias[:S]


class Block(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, cond_dim: int):
        super().__init__()
        
        self.cross_attn = torch.nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.self_attn = torch.nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.ff = torch.nn.Linear(d_model, d_model)
        
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

    
    def forward(self, predicted_stack: torch.Tensor, spectra: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = predicted_stack
        
        # cross-attention
        predicted_stack = self.norm1(predicted_stack, cond)
        predicted_stack, _ = self.cross_attn(query=predicted_stack, key=spectra, value=spectra)
        alpha1 = 1 + self.to_alpha1(cond).unsqueeze(1)   # shape [B,1,H]
        predicted_stack = predicted_stack * alpha1
        predicted_stack += residual
        residual = predicted_stack

        # self-attention
        predicted_stack = self.norm2(predicted_stack, cond)
        predicted_stack, _ = self.self_attn(query=predicted_stack, key=predicted_stack, value=predicted_stack)
        # S = predicted_stack.size(1)
        # causal = torch.triu(torch.ones(S, S, device=predicted_stack.device, dtype=torch.bool), diagonal=1)
        # # attn_mask=True means "these are positions to mask", so pass causal as attn_mask
        # predicted_stack, _ = self.self_attn(
        #     query=predicted_stack, key=predicted_stack, value=predicted_stack,
        #     attn_mask=causal
        # )

        alpha2 = 1 + self.to_alpha2(cond).unsqueeze(1)   # shape [B,1,H]
        predicted_stack = predicted_stack * alpha2
        predicted_stack += residual
        residual = predicted_stack
        
        # feedforward
        predicted_stack = self.norm3(predicted_stack, cond)
        predicted_stack = self.ff(predicted_stack)
        predicted_stack = torch.nn.functional.silu(predicted_stack)

        predicted_stack = self.dropout(predicted_stack)
        
        return predicted_stack


# %% try later im improved performance
# class AdaLNZero(torch.nn.Module):
#     """
#     adaLN-Zero: produce (gamma, beta, gate) from a cond vector, and
#     apply weight-free LN + FiLM-like modulation.
#     """
#     def __init__(self, hidden_size: int, cond_dim: int):
#         super().__init__()
#         self.ln = torch.nn.LayerNorm(hidden_size, elementwise_affine=False)  # weight-free LN
#         self.to_params = torch.nn.Sequential(
#             torch.nn.SiLU(),
#             torch.nn.Linear(cond_dim, 3 * hidden_size)  # gamma, beta, gate
#         )
#         # zero-init so the block starts as identity
#         torch.nn.init.zeros_(self.to_params[1].weight)
#         torch.nn.init.zeros_(self.to_params[1].bias)

#     def modulate(self, x, gamma, beta):
#         return x * (1 + gamma) + beta

#     def forward(self, x: torch.Tensor, cond: torch.Tensor):
#         x_norm = self.ln(x)
#         gamma, beta, gate = self.to_params(cond).chunk(3, dim=-1)  # [B,H] each
#         return self.modulate(x_norm, gamma.unsqueeze(1), beta.unsqueeze(1)), gate.unsqueeze(1)  # [B,1,H]
        

# class Block(torch.nn.Module):
#     def __init__(self, d_model: int, n_heads: int, dropout: float, cond_dim: int):
#         super().__init__()

#         # Attn modules
#         self.cross_attn = torch.nn.MultiheadAttention(
#             embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
#         )
#         self.self_attn = torch.nn.MultiheadAttention(
#             embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
#         )

#         # FFN (2-layer MLP)
#         self.ff1 = torch.nn.Linear(d_model, 4 * d_model)
#         self.ff2 = torch.nn.Linear(4 * d_model, d_model)

#         # adaLN-Zero modulators (one per sublayer)
#         self.ada1 = AdaLNZero(d_model, cond_dim)  # for cross-attn
#         self.ada2 = AdaLNZero(d_model, cond_dim)  # for self-attn
#         self.ada3 = AdaLNZero(d_model, cond_dim)  # for FFN

#         self.drop = torch.nn.Dropout(dropout)

#         # Optional: LN for spectra keys/values (weight-free), helpful for cross-attn stability
#         self.spectra_ln = torch.nn.LayerNorm(d_model, elementwise_affine=False)

#         # ---- Zero-initialize last projections (DiT trick) ----
#         torch.nn.init.zeros_(self.cross_attn.out_proj.weight)
#         torch.nn.init.zeros_(self.cross_attn.out_proj.bias)
#         torch.nn.init.zeros_(self.self_attn.out_proj.weight)
#         torch.nn.init.zeros_(self.self_attn.out_proj.bias)
#         torch.nn.init.zeros_(self.ff2.weight)
#         torch.nn.init.zeros_(self.ff2.bias)

#     def forward(self, x: torch.Tensor, spectra: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
#         # ---- Cross-attention ----
#         residual = x
#         x_mod, gate1 = self.ada1(x, cond)                                # adaLN-Zero
#         # normalize spectra for K/V (kept unconditional; simple LN is enough)
#         k = v = self.spectra_ln(spectra)
#         xa, _ = self.cross_attn(query=x_mod, key=k, value=v, need_weights=False)
#         x = residual + self.drop(gate1 * xa)                              # gated residual

#         # ---- Self-attention ----
#         residual = x
#         x_mod, gate2 = self.ada2(x, cond)
#         xa, _ = self.self_attn(query=x_mod, key=x_mod, value=x_mod, need_weights=False)
#         x = residual + self.drop(gate2 * xa)

#         # ---- Feed-forward ----
#         residual = x
#         x_mod, gate3 = self.ada3(x, cond)
#         xf = self.ff2(self.drop(torch.nn.functional.silu(self.ff1(x_mod))))
#         x = residual + self.drop(gate3 * xf)

#         return x
# %%


class OptoLlama(torch.nn.Module):
    def __init__(
        self, 
        spectra_dim: int, 
        vocab_size: int,
        timesteps: int,
        max_len: int,
        max_stack_depth: int,
        eos_idx: int,
        pad_idx: int,
        mask_idx: int,
        d_model: int,
        n_blocks: int,
        n_heads: int,
        dropout: float,
        idx_to_token: dict,
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
        self.stack_embedding = StackEmbedding(vocab_size, d_model)
        self.time_embedding = TimestepEmbedding(d_model)
        
        self.positional_encoding = PositionalEncoding(max_len, d_model)
        # self.pos_bias = PositionBias(max_stack_depth, d_model)
        # self.pos_token_bias = torch.nn.Parameter(torch.zeros(self.max_stack_depth, self.vocab_size))
        # self.positional_encoding_SPECTRA = LearnedPositionalEncoding(max_len, d_model)
        # self.positional_encoding_STACKS = LearnedPositionalEncoding(max_len, d_model)
        # self.noise = LinearNoise()
        # self.noise = ZeroTerminalNoise()
        self.noise = SquareNoise()
        
        self.blocks = torch.nn.ModuleList([Block(d_model, n_heads, dropout, cond_dim=d_model) for _ in range(n_blocks)])
        self.projection = torch.nn.Linear(d_model, vocab_size)

        self.sample_temperature = 0.0  # 0 = greedy (deterministic); >0 = stochastic
        self.sample_top_k = 0          # 0 = disabled
        self.sample_top_p = 0.0        # 0 = disabled


    def _sample_t(self, batch: torch.Tensor, sampling_eps: float = 1e-3):
        n, device = batch.shape[0], batch.device
          
        timesteps = torch.rand(n, device=device)
        # antithetic sampling
        offset = torch.arange(n, device=device) / n
        timesteps = (timesteps / n + offset) % 1.0
        
        return (1.0 - sampling_eps) * timesteps + sampling_eps

    def _model(self, spectra: torch.Tensor, noised_stacks: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        embedded_spectra = self.spectrum_embedding(spectra)
        # embedded_spectra += self.positional_encoding_SPECTRA(embedded_spectra)
        embedded_spectra += self.positional_encoding(embedded_spectra)
        
        predicted_stacks = self.stack_embedding(noised_stacks)
        # predicted_stacks += self.positional_encoding_STACKS(predicted_stacks)
        predicted_stacks += self.positional_encoding(predicted_stacks)          
        predicted_stacks += self.time_embedding(timesteps)
        # predicted_stacks = self.pos_bias(predicted_stacks)
        cond = self.time_embedding(timesteps)   # [B, 1, 1024]
        cond = cond.squeeze(1)                  # [B, 1024]
        
        for block in self.blocks:
            predicted_stacks = block(predicted_stacks, embedded_spectra, cond=cond)

        predicted_stacks = self.projection(predicted_stacks)
        
        # # predicted_stacks: [B,S,V]
        # S = predicted_stacks.size(1)
        # predicted_stacks = predicted_stacks + self.pos_token_bias[:S]  # broadcast over batch

        return predicted_stacks

    def _train(self, spectra: torch.Tensor, stacks: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # sample time points
        timesteps = self._sample_t(stacks)

        # convert into noise
        betas = self.noise(timesteps)
        # # optollama._train()
        # betas = self.noise(timesteps)
        # warm = (epoch < int(0.05*cfg.EPOCHS))
        # flip_chance = betas * (0.25 if warm else 1.0)  # 4× less masking early

        flip_chance = betas.reshape(-1, 1)
        flipped = torch.rand_like(stacks, dtype=spectra.dtype) < flip_chance
        noised_stacks = torch.where(flipped, self.mask, stacks)

        # query model
        predicted_stacks = self._model(spectra, noised_stacks, timesteps)
        
        return predicted_stacks
    
    def set_sampling(self, temperature: float = 0.0, top_k: int = 0, top_p: float = 0.0):
        self.sample_temperature = float(temperature)
        self.sample_top_k = int(top_k)
        self.sample_top_p = float(top_p)
        print(self.sample_temperature,self.sample_top_k)

    def _top_k_top_p_filtering(self, logits: torch.Tensor, top_k: int, top_p: float, filter_value: float = -float('inf')) -> torch.Tensor:
        logits = torch.nan_to_num(logits, neginf=-1e9, posinf=1e9)  # guard
                
        if top_k > 0:
            top_k = min(top_k, self.vocab_size)
            kth_values = torch.topk(logits, top_k, dim=-1)[0][:, -1].unsqueeze(-1)
            logits = torch.where(logits < kth_values, torch.full_like(logits, filter_value), logits)
            
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, filter_value)
            
        return logits
    
    # def _sample_logits(self, logits, top_k=1, top_p=0.0):
    #     logits = torch.nan_to_num(logits, neginf=-1e9, posinf=1e9)
    #     logits = self._top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    #     probs  = torch.softmax(logits, dim=-1)
    #     probs  = torch.nan_to_num(probs, nan=0.0)
    #     return torch.multinomial(probs, num_samples=1)
    
    def _sample_logits(self, logits, top_k: int = None, top_p: float = None):
        """
        logits: [B, V]
        Uses self.sample_* as defaults, but allows call-site override.
        - If temperature<=0 and no top-k/top-p: greedy argmax (deterministic).
        """
        # defaults from model if not provided
        if top_k is None: top_k = getattr(self, "sample_top_k", 0)
        if top_p is None: top_p = getattr(self, "sample_top_p", 0.0)
        temperature = getattr(self, "sample_temperature", 0.0)
    
        # Greedy fallback: fully deterministic DiT decoding when all sampling knobs are "off"
        if (temperature is None or temperature <= 0.0) and (not top_k or top_k <= 0) and (not top_p or top_p <= 0.0):
            return logits.argmax(dim=-1, keepdim=True)  # [B,1]
    
        # Stochastic path
        logits = torch.nan_to_num(logits, neginf=-1e9, posinf=1e9)
    
        if temperature is not None and temperature > 0.0:
            logits = logits / temperature
    
        # apply top-k / top-p if requested
        logits = self._top_k_top_p_filtering(logits, top_k=top_k or 0, top_p=top_p or 0.0)
    
        probs = torch.softmax(logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)
        return torch.multinomial(probs, num_samples=1)  # [B,1]

    def _sample(self, spectra: torch.Tensor, eps=1e-3, remask_prob=0.1) -> torch.Tensor:
        timesteps = torch.linspace(1.0, eps, self.steps, device=spectra.device)
        stacks = torch.full((spectra.shape[0], self.max_stack_depth,), self.mask, dtype=torch.long, device=spectra.device)
        
        beta_sched = self.noise(timesteps)
        denoise_steps_sim = False
        if denoise_steps_sim:
            from analyze_inference import logits_to_spectra, build_tmm
            import config_MD49 as cfg
            TMM_RAT, wl_tensor, theta = build_tmm(cfg, spectra.device, self.idx_to_token)
            from utils import masked_mae, save_JSONPICKLE_NEW
        
        dn_maes = torch.full((spectra.shape[0], self.steps), 0, dtype=torch.float, device=spectra.device)
            
        for i in range(self.steps):
            t = torch.full((spectra.shape[0],), timesteps[i], device=spectra.device)
            predictions = self._model(spectra, stacks, t)   # [B,S,V]

            # sampled_tokens = []
            # for pos in range(predictions.size(1)):
            #     pos_logits = predictions[:, pos, :]
            #     next_token = self._sample_logits(pos_logits)
            #     sampled_tokens.append(next_token)
            # sampled_stacks = torch.cat(sampled_tokens, dim=1)
            
            # AFTER: vectorize sampling across all positions
            B, S, V = predictions.shape
            logits = predictions.view(B * S, V)
            samples = self._sample_logits(logits)              # [B*S,1]
            sampled_stacks = samples.view(B, S)
            
            if denoise_steps_sim:
                pred_spectra = logits_to_spectra(sampled_stacks, TMM_RAT, wl_tensor, theta, self.eos, self.pad, self.mask, tau=-1)
                resmae = masked_mae(spectra, pred_spectra)
                dn_maes[..., i] = resmae
                print(i, resmae.mean(dim=0))
            if i < self.steps - 1:
                remask_prob = beta_sched[i].item()
                remask = (torch.rand_like(stacks, dtype=spectra.dtype) < remask_prob).bool()
                stacks = torch.where(remask, self.mask, sampled_stacks)
            else:
                stacks = sampled_stacks
        # out_name = f''
        # save_JSONPICKLE_NEW(cfg.PATH_SAVED, maes, out_name)
        # import matplotlib.pyplot as plt
        # from utils import rolling_mean
        # maes_window = list(rolling_mean(maes, 10))
        # plt.plot(maes)
        # plt.plot(maes_window, '--')
        # plt.show()
        return stacks, dn_maes
            
    def forward(self, spectra: torch.Tensor, stacks: torch.Tensor = None) -> torch.Tensor:
        return self._train(spectra, stacks) if stacks is not None else self._sample(spectra)




# %%
# # ========================= Transformer Baseline (DiT-compatible) =========================
# import math
# from typing import Optional

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # -- tiny utils
# def _subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
#     # [1, sz, sz] causal mask (True=keep / False=mask in our code path)
#     return torch.triu(torch.ones(1, sz, sz, device=device, dtype=torch.bool), diagonal=1) == 0

# # -- layers (prefixed to avoid collisions)
# class XfmrLayerNorm(nn.Module):
#     def __init__(self, d, eps=1e-6):
#         super().__init__()
#         self.w = nn.Parameter(torch.ones(d))
#         self.b = nn.Parameter(torch.zeros(d))
#         self.eps = eps
#     def forward(self, x):
#         m = x.mean(-1, keepdim=True)
#         v = x.var(-1, unbiased=False, keepdim=True)
#         return (x - m) / torch.sqrt(v + self.eps) * self.w + self.b

# class XfmrSublayer(nn.Module):
#     def __init__(self, d_model, dropout):
#         super().__init__()
#         self.norm = XfmrLayerNorm(d_model)
#         self.drop = nn.Dropout(dropout)
#     def forward(self, x, sublayer):
#         return x + self.drop(sublayer(self.norm(x)))

# class XfmrMHA(nn.Module):
#     def __init__(self, h, d_model, dropout=0.1):
#         super().__init__()
#         assert d_model % h == 0
#         self.h = h
#         self.d_k = d_model // h
#         self.qkv = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
#         self.o = nn.Linear(d_model, d_model)
#         self.drop = nn.Dropout(dropout)
#     def forward(self, q, k, v, mask: Optional[torch.Tensor] = None):
#         B, Lq, D = q.shape
#         _, Lk, _ = k.shape
#         def proj(l, x):
#             y = l(x).view(B, -1, self.h, self.d_k).transpose(1, 2)  # [B,h,L,d_k]
#             return y
#         q, k, v = [proj(l, x) for l, x in zip(self.qkv, (q, k, v))]
#         scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)     # [B,h,Lq,Lk]
#         if mask is not None:
#             # mask True = keep, False = -inf
#             scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
#         p = F.softmax(scores, dim=-1)
#         p = self.drop(p)
#         x = p @ v                                               # [B,h,Lq,d_k]
#         x = x.transpose(1, 2).contiguous().view(B, Lq, self.h * self.d_k)
#         return self.o(x)

# class XfmrFF(nn.Module):
#     def __init__(self, d_model, d_ff, dropout=0.1):
#         super().__init__()
#         self.fc1 = nn.Linear(d_model, d_ff)
#         self.fc2 = nn.Linear(d_ff, d_model)
#         self.drop = nn.Dropout(dropout)
#     def forward(self, x):
#         return self.fc2(self.drop(F.gelu(self.fc1(x))))

# class XfmrPosEnc(nn.Module):
#     def __init__(self, d_model, max_len=5000, dropout=0.1):
#         super().__init__()
#         pe = torch.zeros(max_len, d_model)
#         pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
#         div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(pos * div)
#         pe[:, 1::2] = torch.cos(pos * div)
#         self.register_buffer("pe", pe.unsqueeze(0))  # [1,max_len,d_model]
#         self.drop = nn.Dropout(dropout)
#     def forward(self, x):
#         return self.drop(x + self.pe[:, : x.size(1), :].to(x.device))

# class XfmrDecoderLayer(nn.Module):
#     def __init__(self, d_model, n_heads, d_ff, dropout):
#         super().__init__()
#         self.self_attn = XfmrMHA(n_heads, d_model, dropout)
#         self.src_attn  = XfmrMHA(n_heads, d_model, dropout)
#         self.ff        = XfmrFF(d_model, d_ff, dropout)
#         self.s1 = XfmrSublayer(d_model, dropout)
#         self.s2 = XfmrSublayer(d_model, dropout)
#         self.s3 = XfmrSublayer(d_model, dropout)
#     def forward(self, x, memory, tgt_mask, src_mask=None):
#         x = self.s1(x, lambda x: self.self_attn(x, x, x, tgt_mask))
#         # memory is [B,1,D]; allow broadcast mask or skip
#         x = self.s2(x, lambda x: self.src_attn(x, memory, memory, src_mask))
#         return self.s3(x, self.ff)

# class XfmrDecoder(nn.Module):
#     def __init__(self, layer: XfmrDecoderLayer, N: int, d_model: int):
#         super().__init__()
#         self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
#         self.norm = XfmrLayerNorm(d_model)
#     def forward(self, x, memory, tgt_mask, src_mask=None):
#         for lyr in self.layers:
#             x = lyr(x, memory, tgt_mask, src_mask)
#         return self.norm(x)

# class TransformerBaseline(nn.Module):
#     """
#     DiT-compatible baseline:
#       - Input spectra: [B, W, 3]  (R/A/T), concatenated -> [B, 3W]
#       - Teacher-forced training: forward(spectra, stacks) -> logits [B,S,V]
#       - Greedy eval:   forward(spectra) -> logits [B,S,V]  (S = max_len provided at init)
#     """
#     def __init__(
#         self,
#         vocab_size: int,
#         d_model: int,
#         n_layers: int,
#         n_heads: int,
#         d_ff: int,
#         dropout: float,
#         max_len: int,
#         mask_idx: int,
#         pad_idx: int,
#         eos_idx: int,
#         spectrum_flat_dim: int,   # = 3 * W
#     ):
#         super().__init__()
#         self.vocab = vocab_size
#         self.d_model = d_model
#         self.max_len = max_len
#         self.mask_idx = mask_idx
#         self.pad_idx = pad_idx
#         self.eos_idx = eos_idx

#         # spectrum: [B, 3W] -> [B, D] -> memory [B,1,D]
#         self.spec_proj = nn.Sequential(
#             nn.Linear(spectrum_flat_dim, spectrum_flat_dim),
#             nn.GELU(), nn.LayerNorm(spectrum_flat_dim),
#             nn.Linear(spectrum_flat_dim, d_model),
#             nn.GELU(), nn.LayerNorm(d_model),
#         )

#         # decoder input embeddings (tokens)
#         self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
#         self.pos_enc = XfmrPosEnc(d_model, max_len=1000, dropout=dropout)

#         # decoder stack
#         layer = XfmrDecoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
#         self.dec = XfmrDecoder(layer, N=n_layers, d_model=d_model)

#         # output head
#         self.generator = nn.Linear(d_model, vocab_size)

#         # init
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)

#     # -- helpers
#     def _flatten_spectra(self, spectra: torch.Tensor) -> torch.Tensor:
#         # [B,W,3] -> [B,3W]  (channel-major concat: R.. A.. T..)
#         if spectra.dim() == 3 and spectra.size(-1) == 3:
#             B, W, C = spectra.shape
#             return spectra.permute(0, 2, 1).reshape(B, C * W)
#         elif spectra.dim() == 2:
#             return spectra
#         else:
#             raise ValueError(f"Unexpected spectra shape {tuple(spectra.shape)}")

#     def _decode(self, memory: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
#         # tgt_tokens: [B,S]
#         x = self.tok_emb(tgt_tokens)                      # [B,S,D]
#         x = self.pos_enc(x)                               # [B,S,D]
#         tgt_mask = _subsequent_mask(tgt_tokens.size(1), x.device)  # [1,S,S]
#         y = self.dec(x, memory, tgt_mask=tgt_mask)        # [B,S,D]
#         return self.generator(y)                          # [B,S,V]

#     # -- DiT-style interface
#     def forward(self, spectra: torch.Tensor, stacks: Optional[torch.Tensor] = None) -> torch.Tensor:
#         spec_vec = self._flatten_spectra(spectra)         # [B,3W]
#         memory = self.spec_proj(spec_vec).unsqueeze(1)    # [B,1,D]

#         if stacks is not None:
#             # teacher forcing: shift-right with <MSK> at t=0
#             B, S = stacks.shape
#             tgt_in = torch.full_like(stacks, self.mask_idx)
#             tgt_in[:, 1:] = stacks[:, :-1]
#             return self._decode(memory, tgt_in)           # logits aligned to stacks [B,S,V]

#         # inference: greedy AR for max_len
#         B = spectra.size(0)
#         out = torch.full((B, self.max_len), self.mask_idx, dtype=torch.long, device=spectra.device)
#         logits_accum = []
#         for t in range(self.max_len):
#             logits_t = self._decode(memory, out[:, : t + 1])  # [B,t+1,V]
#             logits_last = logits_t[:, -1:, :]                  # [B,1,V]
#             logits_accum.append(logits_last)
#             out[:, t] = logits_last.argmax(dim=-1).squeeze(-1)
#             if (out[:, t] == self.eos_idx).all():
#                 # still return full S logits (stack and pad if needed)
#                 continue
#         return torch.cat(logits_accum, dim=1), None                  # [B,S,V]
# %%
# --- BEGIN minimal drop-in: Original decoder-only + tiny wrapper ---
# Paste this block into model.py, replacing the current "Transformer Baseline" section.
# It keeps your OptoLlama/DiT code intact and only swaps the baseline with a
# very small wrapper around the ORIGINAL code you provided.

import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== Original building blocks (unchanged) =====

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

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
        # NOTE: we will use raw logits in the wrapper, but keep this for completeness
        return F.log_softmax(self.proj(x), dim=-1)

# ===== Tiny helpers to interface with the repo =====

def _subsequent_mask(sz: int, device: torch.device):
    return torch.triu(torch.ones(1, sz, sz, device=device, dtype=torch.bool), diagonal=1) == 0

def _flatten_spectra(spectra: torch.Tensor) -> torch.Tensor:
    # [B,W,3] -> [B,3W]
    if spectra.dim() == 3 and spectra.size(-1) == 3:
        B, W, C = spectra.shape
        return spectra.permute(0, 2, 1).reshape(B, C * W)
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
        print(self.sample_temperature,self.sample_top_k)

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
            # last = logits_t[:, -1:, :]                    # [B,1,V]
            # logits_accum.append(last)
            # next_id = last.argmax(dim=-1)                 # [B,1]
            # out = torch.cat([out, next_id], dim=1)
            last = logits_t[:, -1, :]  # [B,V]
            if self.sample_temperature is not None and self.sample_temperature > 0.0:
                logits_adj = last / self.sample_temperature
                logits_adj = _filter_logits_topk_topp(logits_adj, self.sample_top_k, self.sample_top_p)
                probs = torch.softmax(logits_adj, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)  # [B,1]
            else:
                next_id = last.argmax(dim=-1, keepdim=True)        # [B,1]
            
            logits_accum.append(last.unsqueeze(1))                 # keep [B,1,V] for return
            out = torch.cat([out, next_id], dim=1)

        return torch.cat(logits_accum, dim=1), None

# ===== Hook into build_model() =====
# In your existing build_model(), replace the current transformer branch
# with the following minimal construction:
#
#    if mt == "transformer":
#        if sample_spectrum.dim() == 3:
#            _, W, C = sample_spectrum.shape
#            assert C == 3
#            spec_flat_dim = 3 * W
#        elif sample_spectrum.dim() == 2:
#            spec_flat_dim = sample_spectrum.shape[-1]
#        else:
#            raise ValueError(...)
#        return OriginalDecoderWrapper(
#            vocab_size=vocab_size,
#            d_model=d_model,
#            n_layers=n_blocks,
#            n_heads=n_heads,
#            d_ff=4*d_model,
#            dropout=dropout,
#            max_len=max_stack_depth,
#            mask_idx=mask_idx,
#            pad_idx=pad_idx,
#            eos_idx=eos_idx,
#            spectrum_flat_dim=spec_flat_dim,
#        ).to(torch.float32).to(device)
#
# --- END minimal drop-in ---

# %%




# ---------- Factory used by optollama.py --------------------------------------

def build_model(
    *,
    model_type: str,
    spectrum_dim: int,
    vocab_size: int,
    timesteps: int,
    max_len: int,
    max_stack_depth: int,
    mask_idx: int,
    d_model: int,
    n_blocks: int,
    n_heads: int,
    dropout: float,
    idx_to_token: dict,
    pad_idx: int,
    eos_idx: int,
    device: str,
    sample_spectrum: torch.Tensor,   # provide a [1,W,3] example (you already have one in train_loop)
):
    mt = (model_type or "dit").lower()

    if mt == "transformer":
        pass
        # # infer 3*W from a real sample
        # if sample_spectrum.dim() == 3:
        #     _, W, C = sample_spectrum.shape
        #     assert C == 3, f"Expected channels=3, got {C}"
        #     spec_flat_dim = 3 * W
        # elif sample_spectrum.dim() == 2:
        #     spec_flat_dim = sample_spectrum.shape[-1]
        # else:
        #     raise ValueError(f"Unexpected spectrum sample shape {tuple(sample_spectrum.shape)}")

        # model = TransformerBaseline(
        #     vocab_size=vocab_size,
        #     d_model=d_model,
        #     n_layers=n_blocks,
        #     n_heads=n_heads,
        #     d_ff=4 * d_model,
        #     dropout=dropout,
        #     max_len=max_stack_depth,   # generate up to this many tokens
        #     mask_idx=mask_idx,
        #     pad_idx=pad_idx,
        #     eos_idx=eos_idx,
        #     spectrum_flat_dim=spec_flat_dim,
        # ).to(torch.float32).to(device)
        # return model
    
    if mt == "transformer2":
           if sample_spectrum.dim() == 3:
               _, W, C = sample_spectrum.shape
               assert C == 3
               spec_flat_dim = 3 * W
           elif sample_spectrum.dim() == 2:
               spec_flat_dim = sample_spectrum.shape[-1]
           else:
               raise ValueError(...)
           return OriginalDecoderWrapper(
               vocab_size=vocab_size,
               d_model=d_model,
               n_layers=n_blocks,
               n_heads=n_heads,
               d_ff=4*d_model,
               dropout=dropout,
               max_len=max_stack_depth,
               mask_idx=mask_idx,
               pad_idx=pad_idx,
               eos_idx=eos_idx,
               spectrum_flat_dim=spec_flat_dim,
           ).to(torch.float32).to(device)
    
    # --- your existing DiT branch (unchanged) ---
    dit = OptoLlama(
        spectra_dim=spectrum_dim,
        vocab_size=vocab_size,
        timesteps=timesteps,
        max_len=max_len,
        max_stack_depth=max_stack_depth+1,                                      #account for EOS token padded to the end of the trainingdata of len 20
        eos_idx=eos_idx,
        mask_idx=mask_idx,
        pad_idx=pad_idx,
        d_model=d_model,
        n_blocks=n_blocks,
        n_heads=n_heads,
        dropout=dropout,
        idx_to_token=idx_to_token,
    ).to(torch.float32).to(device)
    return dit
