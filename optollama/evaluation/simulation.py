from typing import Any, NamedTuple, Self, Sequence, Union

import torch
import torch.nn as nn

import optollama.utils


def _coh_tmm(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
    """Import tmm_fast lazily to avoid a Windows import-order crash."""
    from tmm_fast import coh_tmm

    return coh_tmm(*args, **kwargs)


class TMMSpectrum(nn.Module):
    """
    Differentiable TMM model.

    Maps a token sequence describing a layer stack to its optical response:
    concatenated (R | A | T) spectrum over wavelength.

    Tokens encode material and thickness via names like "TiO2_60".
    Special tokens (PAD/EOS/MSK) are mapped to zero-thickness layers.
    """

    def __init__(
        self,
        nk_dict: dict[str, Any],
        idx_to_token: dict[int, str],
        substrate: str = "EVA",
        substrate_thick: float = 5e5,
        device: Union[str, torch.device] = "cuda",
    ) -> None:
        """
        Initialize TMMSpectrum.

        Args
        ----
        nk_dict : dict[str, Any]
            Mapping from material name to complex nk values (array-like of
            shape ``[W]``).
        idx_to_token : dict[int, str]
            Vocabulary mapping from token index to token string (e.g.
            ``"TiO2_60"``).
        substrate : str, optional
            Name of the substrate material (default: ``"EVA"``).
        substrate_thick : float, optional
            Substrate thickness in nm (stored but not used explicitly in the
            forward pass; default: ``5e5``).
        device : str or torch.device, optional
            Device where internal buffers are stored (default: ``"cuda"``).
        """
        super().__init__()
        self.substrate_thick = substrate_thick

        # ---- 1. thickness vector & material lookup ----
        v_length = len(idx_to_token)  # vocabulary size
        thickness = torch.zeros(v_length, dtype=torch.complex128, device=device)

        mat_names = []
        for i, tok in idx_to_token.items():
            if "_" in tok:  # e.g. "TiO2_60"
                mat, th = tok.split("_", 1)
                thickness[i] = float(th)
            else:  # PAD / EOS / MSK or any non-layer token
                mat = substrate  # map to substrate material
                thickness[i] = 0.0  # zero thickness
            mat_names.append(mat)

        # keep material order stable and ensure substrate is included
        uniq_mats = list(dict.fromkeys(mat_names + [substrate]))
        self.mat_to_idx: dict[str, int] = {m: j for j, m in enumerate(uniq_mats)}

        nk_table = torch.stack(
            [torch.as_tensor(nk_dict[m], dtype=torch.complex128, device=device) for m in uniq_mats],
            dim=0,
        )  # [M, W]

        # ---- 2. register params / buffers ----
        self.register_buffer("thickness", thickness)  # [V]
        self.register_buffer("nk_table", nk_table)  # [M, W]

        # token id  →  row index in nk_table
        self.register_buffer(
            "mat_idx_table",
            torch.as_tensor([self.mat_to_idx[m] for m in mat_names], dtype=torch.long, device=device),
        )  # [V]
        self.register_buffer("sub_idx", torch.tensor(self.mat_to_idx[substrate], dtype=torch.long, device=device))

    def forward(
        self,
        stacks: torch.Tensor,
        wl_tensor: torch.Tensor,
        theta: torch.Tensor,
        eos: int,
        pad: int,
        msk: int,
        pol: str = "s",
        thickness_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute R, A, T spectra for a batch of stacks.

        Args
        ----
        stacks : torch.Tensor
            Either:
            - Hard token IDs of shape [B, S] (long), or
            - Soft token probabilities/logits of shape [B, S, V] (float).
        wl_tensor : torch.Tensor
            Wavelengths (nm) as a 1D tensor of shape [W], dtype complex128.
        theta : torch.Tensor
            Incidence angle (rad), scalar or shape [1], dtype complex128.
        eos : int
            EOS token ID.
        pad : int
            PAD token ID.
        msk : int
            MSK token ID; kept for API compatibility, used to mask mixtures in soft mode.
        pol : str, optional
            Polarization string for `coh_tmm`, typically "s" or "p".
        thickness_override : torch.Tensor, optional
            Continuous layer thicknesses in nm for hard-token stacks, shape
            ``[B, S]``. This is useful for simulating manufacturing jitter
            while keeping the saved stack tokenized.

        Returns
        -------
        torch.Tensor
            Concatenated R, A, T spectra of shape [B, 3, W], dtype float32.
        """
        # ---- 1. Build per-layer n, t for each sequence ----
        if stacks.dim() == 3:
            if thickness_override is not None:
                raise ValueError("thickness_override is only supported for hard token-id stacks.")

            # Soft / straight-through path: stacks ~ mixture over tokens [B, S, V]
            nk_per_token = self.nk_table[self.mat_idx_table]  # [V, W]

            # Clone to avoid modifying inputs in-place
            cstacks = stacks.clone()  # [B, S, V], real-valued mixture

            # Remove EOS/PAD/MSK from the mixture
            cstacks[..., [eos, pad, msk]] = 0.0
            zfilter = cstacks.sum(dim=-1, keepdim=True)  # [B, S, 1]

            # (A) safe renormalization: only divide where zfilter > 0
            p_norm = torch.where(zfilter > 0, cstacks / zfilter.clamp_min(1e-8), cstacks)

            p_c = p_norm.to(torch.complex128)

            # (B) compute effective n, t per layer from normalized token mixtures
            n_base = torch.einsum("bsv,vw->bsw", p_c, nk_per_token)  # [B, S, W] complex
            t_base = torch.matmul(p_c, self.thickness)  # [B, S] complex

            # (C) explicit fallback where zfilter == 0 (i.e. only EOS/PAD/MSK present)
            is_zero = zfilter.squeeze(-1) <= 0  # [B, S] boolean
            if is_zero.any():
                sub_n = self.nk_table[self.sub_idx]  # [W], substrate refractive index
                n_base[is_zero] = sub_n  # use substrate nk for "empty" layers
                t_base[is_zero] = 0.0  # and zero thickness there

            # Survival gate from EOS (use raw stacks, not p_norm)
            p_eos = stacks[..., eos].clamp(0, 1)  # [B, S]
            survival = torch.cumprod(1.0 - p_eos + 1e-12, dim=1)  # [B, S]
            active = torch.cat([torch.ones_like(survival[:, :1]), survival[:, :-1]], dim=1)  # [B, S]

            t_base = t_base * active  # gate thickness after EOS
        else:
            # Hard IDs path (validation/inference): stacks are [B, S] longs
            token_ids = stacks.to(torch.long)
            is_eos = token_ids == eos
            active = is_eos.cumsum(dim=1) == 0  # [B, S] True before EOS, False at/after
            valid_layer = active & (token_ids != pad) & (token_ids != msk)

            n_base = self.nk_table[self.mat_idx_table[token_ids]]  # [B, S, W]
            # Zero thickness at/after EOS; keep n as defined for clarity
            if thickness_override is None:
                t_base = self.thickness[token_ids] * active.to(self.thickness.dtype)  # [B, S]
            else:
                if thickness_override.shape != token_ids.shape:
                    raise ValueError(
                        "thickness_override must have the same shape as hard token stacks "
                        f"({tuple(token_ids.shape)}), got {tuple(thickness_override.shape)}."
                    )
                t_base = thickness_override.to(device=token_ids.device, dtype=torch.complex128)
                t_base = t_base * valid_layer.to(t_base.dtype)

        # ---- 2. Prepend/append semi-infinite media (air front & back) ----
        b, _, w = n_base.shape
        one = torch.ones(w, dtype=n_base.dtype, device=n_base.device)  # complex
        big = float("inf")

        # Air front
        front_n = one[None, None, :].expand(b, 1, w)  # [B, 1, W]
        front_t = torch.full((b, 1), big, device=n_base.device, dtype=torch.complex128)  # [B, 1]

        # Air back
        back_n = one[None, None, :].expand(b, 1, w).to(torch.complex128)  # [B, 1, W]
        back_t = torch.full((b, 1), big, device=n_base.device, dtype=torch.complex128)  # [B, 1]

        # Stack: air (front) | layers | air (back)
        n_tensor = torch.cat([front_n, n_base, back_n], dim=1)  # [B, S+2, W]
        t_tensor = torch.cat([front_t, t_base, back_t], dim=1)  # [B, S+2]

        # ---- 3. Coherent TMM solver ----
        res = _coh_tmm(
            pol,
            n_tensor,
            t_tensor,
            theta,
            wl_tensor,
            device=n_tensor.device,
        )

        r = torch.nan_to_num(res["R"], nan=0.0, posinf=0.0, neginf=0.0).float().clamp_(0.0, 1.0)
        t = torch.nan_to_num(res["T"], nan=0.0, posinf=0.0, neginf=0.0).float().clamp_(0.0, 1.0)
        # Sanitize A as well to prevent NaNs sneaking into the loss
        a = torch.nan_to_num(1.0 - r - t, nan=0.0, posinf=0.0, neginf=0.0).clamp_(0.0, 1.0)

        out = torch.cat([r, a, t], dim=1)  # [B, 3, W]
        
        return out


def build_tmm(
    incidence_angle: float,
    device: Union[str, torch.device],
    wavelengths: torch.Tensor,
    path_materials: str,
    idx_to_token: dict[int, str],
) -> tuple[TMMSpectrum, torch.Tensor, torch.Tensor]:
    """
    Build a TMMSpectrum instance plus wavelength and angle tensors.

    Args
    ----
    incidence_angle : float
        Incidence angle in degrees.
    device : str or torch.device
        Device on which to allocate model and buffers.
    wavelengths : torch.Tensor
        1-D tensor of wavelengths (nm), real-valued, shape ``[W]``.
    path_materials : str
        Path to directory with nk CSV files.
    idx_to_token : dict[int, str]
        Vocabulary mapping from token index to token string.

    Returns
    -------
    tuple[TMMSpectrum, torch.Tensor, torch.Tensor]
        A 3-tuple of ``(tmm, wl_tensor, theta)`` where:

        - ``tmm`` is the initialized :class:`TMMSpectrum` model.
        - ``wl_tensor`` is the wavelength tensor on ``device``, dtype
          ``complex128``, shape ``[W]``.
        - ``theta`` is the incidence angle in radians, dtype ``complex128``,
          shape ``[1]``.
    """
    theta = torch.tensor(
        incidence_angle * torch.pi / 180.0,
        device=device,
        dtype=torch.complex128,
    ).unsqueeze(0)

    wl_tensor = wavelengths.to(device=device, dtype=torch.complex128).clone()  # [W]
    nk_dict = optollama.utils.load_materials(path_materials, wavelengths)
    tmm = TMMSpectrum(nk_dict, idx_to_token, device=device).to(device).eval()
    
    return tmm, wl_tensor, theta


@torch.no_grad()
def simulate_token_sequence(
    ids: torch.Tensor,
    tmm_ctx: "TMMContext",
    eos: int,
    pad: int,
    msk: int,
    thickness_override: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Simulate RAT spectra from a batch of token id sequences.

    Args
    ----
    ids : torch.Tensor
        Hard token id sequences of shape ``[B, S]`` (long).
    tmm_ctx : TMMContext
        Bundled TMM model, wavelength tensor, and incidence angle.
    eos : int
        EOS token id used to terminate each sequence.
    pad : int
        PAD token id (zero-thickness layers).
    msk : int
        MASK token id (zero-thickness layers).
    thickness_override : torch.Tensor, optional
        Continuous layer thicknesses in nm for hard-token stacks, shape
        ``[B, S]``. Materials are still taken from ``ids``.

    Returns
    -------
    torch.Tensor
        Simulated RAT spectra of shape ``[B, 3, W]``, float32, clamped to
        ``[0, 1]``.
    """
    if tmm_ctx.realistic_enabled:
        out = simulate_token_sequence_averaged(
            ids,
            tmm=tmm_ctx.tmm,
            wavelengths=tmm_ctx.wl,
            angle_thetas=tmm_ctx.average_thetas,
            angle_weights=tmm_ctx.angle_weights,
            polarizations=tmm_ctx.polarizations,
            jitter_realizations=tmm_ctx.jitter_realizations,
            thickness_jitter_nm=tmm_ctx.thickness_jitter_nm,
            eos=eos,
            pad=pad,
            msk=msk,
            thickness_override=thickness_override,
        )
    else:
        out = tmm_ctx.tmm(
            ids,
            tmm_ctx.wl,
            tmm_ctx.theta,
            eos=eos,
            pad=pad,
            msk=msk,
            thickness_override=thickness_override,
        )  # [B, 3, W]

    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).clamp_(0.0, 1.0)


def material_runs_to_token_tensors(
    runs_batch: Sequence[Sequence[dict[str, Any]]],
    *,
    material_to_token_id: dict[str, int],
    eos: int,
    pad: int,
    device: Union[str, torch.device],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert material runs into token IDs plus continuous thickness overrides.

    Token IDs carry material identity only. The returned thickness tensor carries
    the exact run thickness in nm, avoiding discretized token chunking.
    """
    batch_size = len(runs_batch)
    max_runs = max((len(runs) for runs in runs_batch), default=0)
    seq_len = max(max_runs + 1, 1)
    token_ids = torch.full((batch_size, seq_len), int(pad), dtype=torch.long, device=device)
    thickness = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=device)

    for batch_idx, runs in enumerate(runs_batch):
        write_idx = 0
        for run in runs:
            material = str(run["material"])
            thickness_nm = float(run["thickness_nm"])
            if thickness_nm <= 0.0:
                continue
            token_id = material_to_token_id.get(material)
            if token_id is None:
                raise ValueError(f"No representative token id for material {material!r}.")
            if write_idx >= seq_len - 1:
                break
            token_ids[batch_idx, write_idx] = int(token_id)
            thickness[batch_idx, write_idx] = float(thickness_nm)
            write_idx += 1
        if write_idx < seq_len:
            token_ids[batch_idx, write_idx] = int(eos)

    return token_ids, thickness


@torch.no_grad()
def simulate_material_runs(
    runs_batch: Sequence[Sequence[dict[str, Any]]],
    tmm_ctx: "TMMContext",
    *,
    material_to_token_id: dict[str, int],
    eos: int,
    pad: int,
    msk: int,
) -> torch.Tensor:
    """
    Simulate RAT spectra from material runs with native run thicknesses.

    This is the depth-field-native scoring path: each contiguous material run
    becomes one optical layer with its run thickness, independent of the
    tokenized layer-thickness vocabulary and independent of MAX_SEQ_LEN.
    """
    device = tmm_ctx.wl.device
    token_ids, thickness = material_runs_to_token_tensors(
        runs_batch,
        material_to_token_id=material_to_token_id,
        eos=eos,
        pad=pad,
        device=device,
    )
    return simulate_token_sequence(
        token_ids,
        tmm_ctx,
        eos=eos,
        pad=pad,
        msk=msk,
        thickness_override=thickness,
    )


def active_layer_mask(stacks: torch.Tensor, eos: int, pad: int, msk: int) -> torch.Tensor:
    """Return a boolean mask for material layers before EOS."""
    token_ids = stacks.to(torch.long)
    is_eos = token_ids == eos
    before_eos = is_eos.cumsum(dim=1) == 0
    return before_eos & (token_ids != pad) & (token_ids != msk)


@torch.no_grad()
def simulate_token_sequence_averaged(
    ids: torch.Tensor,
    tmm: torch.nn.Module,
    wavelengths: torch.Tensor,
    angle_thetas: Sequence[torch.Tensor],
    angle_weights: Sequence[float],
    polarizations: Sequence[str],
    jitter_realizations: int,
    thickness_jitter_nm: float,
    eos: int,
    pad: int,
    msk: int,
    thickness_override: torch.Tensor | None = None,
) -> torch.Tensor:
    """Simulate token stacks with angle, polarization, and thickness-jitter averaging."""
    if len(angle_thetas) != len(angle_weights):
        raise ValueError("angle_thetas and angle_weights must have the same length.")
    if len(polarizations) == 0:
        raise ValueError("At least one polarization is required.")
    if jitter_realizations <= 0:
        raise ValueError("jitter_realizations must be positive.")

    token_ids = ids.to(torch.long)
    valid = active_layer_mask(token_ids, eos=eos, pad=pad, msk=msk)
    if thickness_override is None:
        nominal = tmm.thickness[token_ids].real.float() * valid.float()
    else:
        if thickness_override.shape != token_ids.shape:
            raise ValueError(
                "thickness_override must have the same shape as token ids "
                f"({tuple(token_ids.shape)}), got {tuple(thickness_override.shape)}."
            )
        nominal = thickness_override.to(device=token_ids.device, dtype=torch.float32) * valid.float()
    out = torch.zeros((token_ids.size(0), 3, wavelengths.numel()), device=token_ids.device, dtype=torch.float32)

    normalizer = float(sum(float(v) for v in angle_weights) * len(polarizations))
    if normalizer <= 0.0:
        raise ValueError("Averaging weights must have positive total weight.")

    if thickness_jitter_nm > 0.0 and int(jitter_realizations) > 1:
        b, s = token_ids.shape
        jr = int(jitter_realizations)
        sim_ids = token_ids.unsqueeze(0).expand(jr, b, s).reshape(jr * b, s)
        sim_valid = valid.unsqueeze(0).expand(jr, b, s).reshape(jr * b, s)
        sim_nominal = nominal.unsqueeze(0).expand(jr, b, s).reshape(jr * b, s)
        jitter = (torch.rand((jr * b, s), device=token_ids.device) * 2.0 - 1.0) * float(thickness_jitter_nm)
        sim_thickness = (sim_nominal + jitter).clamp_min(0.0) * sim_valid.float()
    elif thickness_jitter_nm > 0.0:
        sim_ids = token_ids
        jitter = (torch.rand(nominal.shape, device=token_ids.device) * 2.0 - 1.0) * float(thickness_jitter_nm)
        sim_thickness = nominal + jitter
        sim_thickness = sim_thickness.clamp_min(0.0) * valid.float()
        jr = 1
    else:
        sim_ids = token_ids
        sim_thickness = nominal
        jr = 1

    for theta, angle_weight in zip(angle_thetas, angle_weights):
        for pol in polarizations:
            simulated = tmm(
                sim_ids,
                wavelengths,
                theta,
                eos=eos,
                pad=pad,
                msk=msk,
                pol=pol,
                thickness_override=sim_thickness,
            )
            if jr > 1:
                simulated = simulated.view(jr, token_ids.size(0), *simulated.shape[1:]).mean(dim=0)
            out.add_(simulated, alpha=float(angle_weight))

    return (out / normalizer).clamp_(0.0, 1.0)


def _realistic_tmm_config(cfg: dict) -> dict:
    block = cfg.get("REALISTIC_TMM") or {}
    if not bool(block.get("ENABLED", False)):
        return {}

    fallback = cfg.get("REALISTIC_DATASET") or {}
    merged = dict(fallback)
    merged.update(block)
    return merged


class TMMContext(NamedTuple):
    """
    Lightweight container bundling TMM model and its optical grid.

    Attributes
    ----------
    tmm : torch.nn.Module
        The TMM model (typically TMMSpectrum).
    wl : torch.Tensor
        Wavelength tensor [W], complex128.
    theta : torch.Tensor
        Incidence angle tensor [], [1], or broadcastable, complex128.
    average_thetas : tuple[torch.Tensor, ...] or None
        Optional incidence-angle tensors used for realistic averaging.
    angle_weights : tuple[float, ...]
        Weights for ``average_thetas``.
    polarizations : tuple[str, ...]
        Polarizations to average when ``average_thetas`` is set.
    jitter_realizations : int
        Number of thickness-jitter realizations used in averaging.
    thickness_jitter_nm : float
        Uniform per-layer thickness jitter range in nm.
    """

    tmm: torch.nn.Module
    wl: torch.Tensor
    theta: torch.Tensor
    average_thetas: tuple[torch.Tensor, ...] | None = None
    angle_weights: tuple[float, ...] = (1.0,)
    polarizations: tuple[str, ...] = ("s",)
    jitter_realizations: int = 1
    thickness_jitter_nm: float = 0.0

    @property
    def realistic_enabled(self) -> bool:
        """Whether this context uses realistic averaged TMM simulation."""
        return self.average_thetas is not None

    @staticmethod
    @torch.no_grad()
    def make(
        cfg: dict,
        idx_to_token: dict[int, str],
        device: Union[str, torch.device],
    ) -> Self:
        """
        Centralized helper to construct a TMMContext from a config object.

        Args
        ----
        cfg : dict
            Configuration dictionary providing at least:

            - ``"INCIDENCE_ANGLE"`` (float, degrees)
            - ``"WAVELENGTHS"`` (torch.Tensor)
            - ``"MATERIALS_PATH"`` (str)
        idx_to_token : dict[int, str]
            Vocabulary mapping from token index to token string.
        device : str or torch.device
            Device for model and buffers.

        Returns
        -------
        TMMContext
            Named tuple bundling ``(tmm, wl_tensor, theta)``.
        """
        tmm, wl_tensor, theta = build_tmm(
            incidence_angle=cfg["INCIDENCE_ANGLE"],
            device=device,
            wavelengths=cfg["WAVELENGTHS"],
            path_materials=cfg["MATERIALS_PATH"],
            idx_to_token=idx_to_token,
        )

        realistic_cfg = _realistic_tmm_config(cfg)
        if realistic_cfg:
            angles = tuple(float(v) for v in realistic_cfg.get("ANGLES", [cfg["INCIDENCE_ANGLE"]]))
            angle_weights = tuple(float(v) for v in realistic_cfg.get("ANGLE_WEIGHTS", [1.0] * len(angles)))
            polarizations = tuple(str(v) for v in realistic_cfg.get("POLARIZATIONS", ["s", "p"]))
            if len(angles) != len(angle_weights):
                raise ValueError("REALISTIC_TMM.ANGLES and REALISTIC_TMM.ANGLE_WEIGHTS must have the same length.")

            average_thetas = tuple(
                torch.tensor(angle * torch.pi / 180.0, device=device, dtype=torch.complex128).unsqueeze(0)
                for angle in angles
            )
            return TMMContext(
                tmm=tmm,
                wl=wl_tensor,
                theta=theta,
                average_thetas=average_thetas,
                angle_weights=angle_weights,
                polarizations=polarizations,
                jitter_realizations=int(realistic_cfg.get("JITTER_REALIZATIONS", 1)),
                thickness_jitter_nm=float(realistic_cfg.get("THICKNESS_JITTER_NM", 0.0)),
            )

        return TMMContext(tmm=tmm, wl=wl_tensor, theta=theta)
