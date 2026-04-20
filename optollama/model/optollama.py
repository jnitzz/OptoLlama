import math
from typing import Optional

import torch

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

    def __init__(self, d_model: int, n_heads: int, dropout: float, cond_dim: int):
        super().__init__()

        self.cross_attn = torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
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

    def forward(self, predicted_stack: torch.Tensor, spectra: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
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

        self.blocks = torch.nn.ModuleList([Block(d_model, n_heads, dropout, cond_dim=d_model) for _ in range(n_blocks)])
        self.projection = torch.nn.Linear(d_model, vocab_size)

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

    def _model(self, spectra: torch.Tensor, noised_stacks: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DiT.

        Embeds spectra, stacks, and timesteps; applies positional encodings and
        conditional DiT blocks; finally projects to vocabulary logits.

        Args
        ----
        spectra : torch.Tensor
            Input spectra, shape ``[B, D_spec]``.
        noised_stacks : torch.Tensor
            Masked/noised stack tokens, shape ``[B, S]``.
        timesteps : torch.Tensor
            Diffusion timesteps, shape ``[B]``.

        Returns
        -------
        torch.Tensor
            Predicted logits over tokens, shape ``[B, S, vocab_size]``.
        """
        embedded_spectra = self.spectrum_embedding(spectra)  # [B, 3, d_model]
        embedded_spectra += self.positional_encoding(embedded_spectra)
        predicted_stacks = self.stack_embedding(noised_stacks)
        predicted_stacks += self.positional_encoding(predicted_stacks)
        predicted_stacks += self.time_embedding(timesteps)  # [B, S, d_model]
        cond = self.time_embedding(timesteps)  # [B, 1, 1024]
        cond = cond.squeeze(1)  # [B, 1024]

        for block in self.blocks:
            predicted_stacks = block(predicted_stacks, embedded_spectra, cond)

        predicted_stacks = self.projection(predicted_stacks)

        return predicted_stacks

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
        noised_stacks = torch.where(flipped, self.mask, stacks)

        # query model
        predicted_stacks = self._model(spectra, noised_stacks, timesteps)

        return predicted_stacks

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
        if hasattr(self, "allowed_vocab_mask") and self.allowed_vocab_mask is not None:
            mask = self.allowed_vocab_mask
            if mask.dtype != torch.bool:
                mask = mask.bool()
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

        stacks = torch.full(
            (spectra.shape[0], self.max_stack_depth),
            self.mask,
            dtype=torch.long,
            device=spectra.device,
        )

        beta_sched = self.noise(timesteps)

        # Decide whether we track MAE this run
        track_mae = bool(self._step_mae_enabled and (self._step_mae_ctx is not None))
        mae_per_step = []

        for i in range(self.steps):
            t = torch.full((spectra.shape[0],), timesteps[i], device=spectra.device)
            predicted_stacks = self._model(spectra, stacks, t)  # [B, S, V]

            b, s, v = predicted_stacks.shape
            logits = predicted_stacks.view(b * s, v)
            samples = self._sample_logits(logits)  # [B*S, 1]
            sampled_stacks = samples.view(b, s)

            # --- enforce max emitted length (inference-time constraint) ---
            # Force EOS at position max_len -1 and PAD after, if enabled.
            if self.max_emit_len is not None:
                max_len = int(self.max_emit_len)
                # position L-1 forced to EOS
                eos_pos = max_len - 1
                if 0 <= eos_pos < s:
                    sampled_stacks[:, eos_pos] = int(self.eos)
                # positions >= max_len forced to PAD
                if max_len < s:
                    sampled_stacks[:, max_len:] = int(self.pad)

            if i < self.steps - 1:
                # overwrite remask_prob with noise schedule
                remask_prob = beta_sched[i].item()
                remask = (torch.rand_like(stacks, dtype=spectra.dtype) < remask_prob).bool()

                # Don't allow remasking of forced EOS/PAD positions
                if self.max_emit_len is not None:
                    max_len = int(self.max_emit_len)
                    if 0 < max_len <= s:
                        remask[:, max_len - 1 :] = False

                stacks = torch.where(remask, self.mask, sampled_stacks)
            else:
                stacks = sampled_stacks

            # ---- optional per-step MAE tracking ----
            if track_mae:
                pred_spec = optollama.evaluation.simulation.simulate_token_sequence(
                    stacks,
                    self._step_mae_ctx,
                    eos=self.eos,
                    pad=self.pad,
                    msk=self.mask,
                )  # [B, 3, W]
                step_mae = optollama.evaluation.metrics.masked_mae(spectra, pred_spec)  # [B]
                mae_per_step.append(step_mae)

        step_mae_traj: Optional[torch.Tensor]
        if track_mae and mae_per_step:
            # mae_per_step: list of [B] → [steps, B] → [B, steps]
            step_mae_traj = torch.stack(mae_per_step, dim=0).transpose(0, 1).contiguous()
        else:
            step_mae_traj = None

        return stacks, step_mae_traj

    def forward(self, spectra: torch.Tensor, stacks: torch.Tensor = None) -> torch.Tensor:
        """
        Unified forward interface.

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
        return self._train(spectra, stacks) if stacks is not None else self._sample(spectra)
