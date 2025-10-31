# ------------------------------------------------------------
# Differentiable optical-stack layer
# ------------------------------------------------------------
import os
import torch
import torch.nn as nn
from typing import Dict, Any
import numpy as np
from tmm_fast import coh_tmm
from utils import load_materialdata_file
from scipy.interpolate import interp1d
import pandas as pd
# import config_MD47 as c


def load_materials(c):
    '''
    Load material nk and return corresponding interpolators.

    Return:
        nk_dict: dict, key -- material name, value: n, k in the 
        self.wavelength range
    '''
    
    c.THICKNESSES = np.arange(c.THICKNESS_MIN,c.THICKNESS_MAX+1,c.THICKNESS_STEPS)
    c.WAVELENGTHS = np.arange(c.WAVELENGTH_MIN,c.WAVELENGTH_MAX+1,c.WAVELENGTH_STEPS)
    material_files = [item[:-4] for item in sorted(os.listdir(c.PATH_MATERIALS)) 
                      if not item.startswith('XX')]
    file_type = os.listdir(c.PATH_MATERIALS)[0][-4:]
    nk_dict = {}
    for mat in material_files:
        if file_type == '.csv':
            try:
                data_temp = pd.read_csv(os.path.join(c.PATH_MATERIALS, f'{mat}.csv'))
                wavelength_nm, n_vals, k_vals, short_name = data_temp['nm'].to_numpy(), data_temp['n'].to_numpy(), data_temp['k'].to_numpy(), f'{mat}' 
            except:
                print("Error: NK File has not the right format: .csv with Columns 'nm', 'n', 'k', comma (,) separated.")
        elif file_type == '.txt':
            try:
                wavelengths_si, n_vals, k_vals, short_name = load_materialdata_file(os.path.join(c.PATH_MATERIALS, f'{mat}.txt'))
                wavelength_nm = wavelengths_si*1e9
            except:
                print("Error: NK File has not the right format: .txt with Columns 'um', 'n', 'k'.")

        n_fn = interp1d(
                wavelength_nm, n_vals, axis=0, bounds_error=False, kind='linear', fill_value=(n_vals[0], n_vals[-1]))
        k_fn = interp1d(
                wavelength_nm, k_vals, axis=0, bounds_error=False, kind='linear', fill_value=(k_vals[0], k_vals[-1]))
            
        nk_dict[mat] = n_fn(c.WAVELENGTHS) + 1j*k_fn(c.WAVELENGTHS)

    return nk_dict


class TMMSpectrum(nn.Module):
    """
    Differentiable TMM.
    Token sequence  →  concatenated (R|A|T) spectrum.
    """

    # --------------------------- constructor ---------------------------
    def __init__(
        self,
        nk_dict: Dict[str, Any],
        idx_to_token: Dict[int, str],
        *,
        substrate: str = "EVA",
        substrate_thick: float = 5e5,
        device: str | torch.device = "cuda",
        learn_nk: bool = False,
    ):
        super().__init__()
        self.substrate_thick = float(substrate_thick)

        # ---- 1. thickness vector & material lookup ----
        V = len(idx_to_token)                            # vocabulary size
        thickness = torch.zeros(V, dtype=torch.complex128)

        mat_names = []
        for i, tok in idx_to_token.items():
            if "_" in tok:                               # e.g. "TiO2_60"
                mat, th = tok.split("_", 1)
                thickness[i] = float(th)
            else:                                        # PAD / EOS / MSK
                mat = substrate                          # map to substrate
                thickness[i] = 0.0                       # zero thickness
            mat_names.append(mat)

        uniq_mats = list(dict.fromkeys(mat_names + [substrate]))
        self.mat_to_idx = {m: j for j, m in enumerate(uniq_mats)}

        nk_table = torch.stack(
            [torch.as_tensor(nk_dict[m], dtype=torch.complex128) for m in uniq_mats], dim=0
        )
        
        # ---- 2. register params / buffers ---- #TODO check if section is necessary
        self.register_buffer("thickness", thickness)  # keep as buffer
        # if learn_nk:
        #     self.nk_table = nn.Parameter(nk_table)
        # else:
        
        self.register_buffer("nk_table", nk_table)

        # token id  →  row in nk_table
        self.register_buffer(
            "mat_idx_table",
            torch.as_tensor([self.mat_to_idx[m] for m in mat_names], dtype=torch.long),
        )
        self.register_buffer(
            "sub_idx",
            torch.tensor(self.mat_to_idx[substrate], dtype=torch.long),
        )
    
    # --------------------------- forward ---------------------------
    def forward(
        self,
        stacks: torch.Tensor,     # [B,S] ints  *or*  [B,S,V] floats
        wl_tensor: torch.Tensor,     # [W]  wavelengths (nm)
        theta:  torch.Tensor,     # [] or [1] incidence angle (rad)
        *,
        eos: int, pad: int, msk: int,     # kept for API compatibility
        pol: str = "s",
    ) -> torch.Tensor:
        """
        Output: [B, 3*W]  – concatenated R, A, T   (float32)
        """
        def _dbg(x, name):
            print(f"{name}: req={getattr(x,'requires_grad',None)}, dtype={x.dtype}")
            return x
        
        # _dbg(stacks, "in.stacks")
        # -- inside TMMSpectrum.forward, just after you branch on stacks.dim() --
        if stacks.dim() == 3:  # soft / straight-through path
            # --- inside TMMSpectrum.forward, soft / STE path ([B,S,V]) ---
            # stacks_f = stacks.to(torch.complex128)  # real probs cast to complex
            # _dbg(stacks_f, "stacks_f")
            nk_per_token = self.nk_table[self.mat_idx_table]               # [V,W] #TODO make already precalculated
            
            # # 1) take EOS/PAD/MSK out of the mixture
            p = stacks.clone()                                # [B,S,V], real
            # remove specials from the *mixture*
            p[..., [eos, pad, msk]] = 0.0
            Z = p.sum(dim=-1, keepdim=True)                    # [B,S,1]
            
            # (A) safe renorm: only divide where Z>0
            p_norm = torch.where(Z > 0, p / Z.clamp_min(1e-8), p)  # leaves zero rows as all-zeros
            
            p_c = p_norm.to(torch.complex128)
            
            # (B) compute base n, t from normalized real tokens
            n_base = torch.einsum("bsv,vw->bsw", p_c, nk_per_token)  # [B,S,W] (complex)
            t_base = torch.matmul(p_c, self.thickness)               # [B,S]   (complex)
            
            # (C) explicit fallback for positions with Z==0 (i.e., only EOS/PAD/MSK were present)
            is_zero = (Z.squeeze(-1) <= 0)                           # [B,S] boolean
            if is_zero.any():
                sub_n = self.nk_table[self.sub_idx]                  # [W] substrate
                n_base[is_zero] = sub_n                              # set to substrate index for clarity
                t_base[is_zero] = 0.0                                # zero thickness there
            
            # --- survival gate from EOS (use raw stacks, not p_norm) ---
            p_eos   = stacks[..., eos].clamp(0, 1)                   # [B,S]
            survival = torch.cumprod(1.0 - p_eos + 1e-12, dim=1)     # [B,S]
            active   = torch.cat([torch.ones_like(survival[:, :1]), survival[:, :-1]], dim=1)
            
            t_base = t_base * active                                 # gating thickness after EOS
        else:  # hard ids path (validation/inference)
            token_ids = stacks.to(torch.long)
            is_eos = (token_ids == eos)
            active = (is_eos.cumsum(dim=1) == 0)                           # [B,S] True before EOS
            
            n_base = self.nk_table[self.mat_idx_table[token_ids]]          # [B,S,W]
            # end the stack after EOS by zeroing thickness; do not force n to substrate
            t_base = self.thickness[token_ids] * active.to(self.thickness.dtype)

        # ---- 2. prepend / append air & substrate ----
        B, S, W = n_base.shape
        one = torch.ones(W, dtype=n_base.dtype, device=n_base.device)      # complex64
        BIG = float("Inf")    #TODO check if stable

        front_n = one[None, None, :].expand(B, 1, W)        # air (front)
        front_t = torch.full((B, 1), BIG, device=n_base.device).to(torch.complex128)

        sub_n = self.nk_table[self.sub_idx]                 # substrate n-k  [W]
        sub_n_tile = sub_n[None, None, :].expand(B, 1, W).to(torch.complex128)
        sub_t      = torch.full((B, 1), self.substrate_thick, device=n_base.device).to(torch.complex128)

        back_n = one[None, None, :].expand(B, 1, W).to(torch.complex128)         # air (back)
        back_t = torch.full((B, 1), BIG, device=n_base.device).to(torch.complex128)

        n_tensor = torch.cat([front_n, n_base[:,:,:], sub_n_tile, back_n], 1)     # [B,S+3,W]
        t_tensor = torch.cat([front_t, t_base[:,:],   sub_t,      back_t], 1)   # [B,S+3]
        # _dbg(n_tensor, "n_tensor")
        # _dbg(t_tensor, "t_tensor")
        # ---- 3. coherent TMM solver (use clones to avoid in-place issues) ----
        res = coh_tmm(
            pol,
            n_tensor,
            t_tensor,
            theta,
            wl_tensor,
            device=n_tensor.device,
        )
        
        R = torch.nan_to_num(res["R"], nan=0.0, posinf=0.0, neginf=0.0).float().clamp_(0.0, 1.0)
        T = torch.nan_to_num(res["T"], nan=0.0, posinf=0.0, neginf=0.0).float().clamp_(0.0, 1.0)
        # sanitize A as well to prevent NaNs sneaking into the loss
        A = torch.nan_to_num(1.0 - R - T, nan=0.0, posinf=0.0, neginf=0.0).clamp_(0.0, 1.0)
        
        # If your coh_tmm returns [B,1,W] for each, this yields [B,3,W]; if it returns [B,W], it yields [B,3W]
        out = torch.cat([R, A, T], dim=1)
        # _dbg(out, "out")
        return out