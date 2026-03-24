from typing import Any, NamedTuple, Self, Union

import torch
import torch.nn as nn

from tmm_fast import coh_tmm

import optollama.utils


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

        Returns
        -------
        torch.Tensor
            Concatenated R, A, T spectra of shape [B, 3, W], dtype float32.
        """
        # ---- 1. Build per-layer n, t for each sequence ----
        if stacks.dim() == 3:
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

            n_base = self.nk_table[self.mat_idx_table[token_ids]]  # [B, S, W]
            # Zero thickness at/after EOS; keep n as defined for clarity
            t_base = self.thickness[token_ids] * active.to(self.thickness.dtype)  # [B, S]

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
        res = coh_tmm(
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
def simulate_token_sequence(ids: torch.Tensor, tmm_ctx: "TMMContext", eos: int, pad: int, msk: int) -> torch.Tensor:
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

    Returns
    -------
    torch.Tensor
        Simulated RAT spectra of shape ``[B, 3, W]``, float32, clamped to
        ``[0, 1]``.
    """
    tmm, wl, theta = tmm_ctx
    out = tmm(ids, wl, theta, eos=eos, pad=pad, msk=msk)  # [B, 3, W]

    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).clamp_(0.0, 1.0)


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
    """
    tmm: torch.nn.Module
    wl: torch.Tensor
    theta: torch.Tensor

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

        return TMMContext(tmm=tmm, wl=wl_tensor, theta=theta)
