import torch
from model import OptoLlama, OriginalDecoderWrapper

def build_model( #TODO optollamam und optoGPT
    *,
    model_type: str,
    sample_spectrum: torch.Tensor,   # provide a [1,3,W] example
    vocab_size: int,
    max_stack_depth: int,
    d_model: int,
    n_blocks: int,
    n_heads: int,
    timesteps: int,
    dropout: float,
    idx_to_token: dict,
    mask_idx: int,
    pad_idx: int,
    eos_idx: int,
    device: str,
):
    mt = (model_type or "dit").lower()
    
    if mt == "transformer":
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
            spectrum_flat_dim=max(sample_spectrum.size())*min(sample_spectrum.size()),
        ).to(torch.float32).to(device)
    elif mt == "dit":
        return OptoLlama(
            spectra_dim=max(sample_spectrum.size()),
            vocab_size=vocab_size,
            timesteps=timesteps,
            max_stack_depth=max_stack_depth,
            eos_idx=eos_idx,
            mask_idx=mask_idx,
            pad_idx=pad_idx,
            d_model=d_model,
            n_blocks=n_blocks,
            n_heads=n_heads,
            dropout=dropout,
            idx_to_token=idx_to_token,
        ).to(torch.float32).to(device)
    else:
        raise ValueError(f"Unsupported model key: {mt}")