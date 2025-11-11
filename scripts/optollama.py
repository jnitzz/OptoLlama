import math
import torch

class SquareNoise(torch.nn.Module):
    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # Normalize timesteps to [0, 1] if they aren't already
        t = timesteps.clamp(0, 1)
        return (1.0 - self.eps) * t**2


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
        sigmas = -torch.log(timesteps)                                          #TODO check why here
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


class OptoLlama(torch.nn.Module):
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
        
        self.positional_encoding = PositionalEncoding(2000, d_model)
        self.noise = SquareNoise()
        
        self.blocks = torch.nn.ModuleList([Block(d_model, n_heads, dropout, 
                                                 cond_dim=d_model) for _ 
                                           in range(n_blocks)])
        self.projection = torch.nn.Linear(d_model, vocab_size)

        self.sample_temperature = 0.0  # 0 = greedy
        self.sample_top_k = 0          # 0 = disabled
        self.sample_top_p = 0.0        # 0 = disabled


    def _sample_t(self, batch: torch.Tensor, sampling_eps: float = 1e-3):
        n, device = batch.shape[0], batch.device
          
        timesteps = torch.rand(n, device=device)
        # antithetic sampling
        offset = torch.arange(n, device=device) / n
        timesteps = (timesteps / n + offset) % 1.0
        
        return (1.0 - sampling_eps) * timesteps + sampling_eps

    def _model(self, spectra: torch.Tensor, noised_stacks: torch.Tensor, 
               timesteps: torch.Tensor) -> torch.Tensor:
        embedded_spectra = self.spectrum_embedding(spectra)
        embedded_spectra += self.positional_encoding(embedded_spectra)
        predicted_stacks = self.stack_embedding(noised_stacks)
        predicted_stacks += self.positional_encoding(predicted_stacks)          
        predicted_stacks += self.time_embedding(timesteps)
        cond = self.time_embedding(timesteps)   # [B, 1, 1024]
        cond = cond.squeeze(1)                  # [B, 1024]
        
        for block in self.blocks:
            predicted_stacks = block(predicted_stacks, embedded_spectra, cond=cond)

        predicted_stacks = self.projection(predicted_stacks)
        
        return predicted_stacks

    def _train(self, spectra: torch.Tensor, stacks: torch.Tensor) -> (torch.Tensor):
        # sample time points
        timesteps = self._sample_t(stacks)

        # convert into noise
        betas = self.noise(timesteps)

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
        stacks = torch.full((spectra.shape[0], self.max_stack_depth,), 
                            self.mask, dtype=torch.long, device=spectra.device)
        beta_sched = self.noise(timesteps)
        denoise_steps_sim = False
        if denoise_steps_sim:
            from analyze_inference import logits_to_spectra
            from call_tmm_fast import _build_tmm
            import config_MD49 as cfg
            TMM_RAT, wl_tensor, theta = \
                _build_tmm(incidence_angle=cfg.INCIDENCE_ANGLE, 
                           device=stacks.device,
                           wavelengths=cfg.WAVELENGTHS, 
                           path_materials=cfg.PATH_MATERIALS, 
                           idx_to_token=self.idx_to_token)            
            from utils import masked_mae, save_JSONPICKLE
        
        dn_maes = torch.full((spectra.shape[0], self.steps), 0, 
                             dtype=torch.float, device=spectra.device)
            
        for i in range(self.steps):
            t = torch.full((spectra.shape[0],), timesteps[i], device=spectra.device)
            predicted_stacks = self._model(spectra, stacks, t)   # [B,S,V]
            
            B, S, V = predicted_stacks.shape
            logits = predicted_stacks.view(B * S, V)
            samples = self._sample_logits(logits)              # [B*S,1]
            sampled_stacks = samples.view(B, S)
            
            if denoise_steps_sim:
                pred_spectra = logits_to_spectra(sampled_stacks, TMM_RAT, 
                                                 wl_tensor, theta, self.eos, 
                                                 self.pad, self.mask, tau=-1)
                resmae = masked_mae(spectra, pred_spectra)
                dn_maes[..., i] = resmae
                print(i, resmae.mean(dim=0))
            if i < self.steps - 1:
                remask_prob = beta_sched[i].item()
                remask = (torch.rand_like(stacks, dtype=spectra.dtype) < remask_prob).bool()
                stacks = torch.where(remask, self.mask, sampled_stacks)
            else:
                stacks = sampled_stacks
                
        if denoise_steps_sim:
            out_name = ''
            save_JSONPICKLE(cfg.PATH_SAVED, resmae, out_name)
            import matplotlib.pyplot as plt
            from utils import rolling_mean
            maes_window = list(rolling_mean(resmae, 10))
            plt.plot(resmae)
            plt.plot(maes_window, '--')
            plt.show()
        return stacks, dn_maes
            
    def forward(self, spectra: torch.Tensor, stacks: torch.Tensor = None) -> torch.Tensor:
        return self._train(spectra, stacks) if stacks is not None else self._sample(spectra)
