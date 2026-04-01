import torch
import torch.nn as nn

from .optogpt import OriginalDecoderWrapper
from .optollama import OptoLlama


def build_model(
    model_type: str,
    sample_spectrum: torch.Tensor,
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
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 0.0,
) -> nn.Module:
    """
    Build and return the configured sequence model (OptoGPT or OptoLlama).

    Args
    ----
    model_type : str
        Which model architecture to build: ``"optogpt"`` or ``"optollama"``.
    sample_spectrum : torch.Tensor
        A representative spectrum of shape ``[1, 3, W]`` or ``[3, W]`` used
        to infer the spectral input dimension.
    vocab_size : int
        Number of discrete material/layer tokens.
    max_stack_depth : int
        Maximum token sequence length.
    d_model : int
        Transformer hidden dimension.
    n_blocks : int
        Number of transformer blocks / decoder layers.
    n_heads : int
        Number of attention heads.
    timesteps : int
        Number of diffusion sampling steps (OptoLlama only).
    dropout : float
        Dropout probability.
    idx_to_token : dict
        Mapping from token index to token string.
    mask_idx : int
        Index of the ``<MASK>`` token.
    pad_idx : int
        Index of the ``<PAD>`` token.
    eos_idx : int
        Index of the ``<EOS>`` token.
    device : str
        Device string (e.g. ``"cuda"`` or ``"cpu"``).
    temperature : float
        Sampling temperature (``0.0`` = greedy; default: ``0.0``).
    top_k : int
        Top-k sampling cutoff (``0`` = disabled; default: ``0``).
    top_p : float
        Top-p (nucleus) sampling cutoff (``0.0`` = disabled; default:
        ``0.0``).

    Returns
    -------
    torch.nn.Module
        Initialized model moved to ``device`` and cast to float32.

    Raises
    ------
    ValueError
        If ``model_type`` is not ``"optogpt"`` or ``"optollama"``.
    """
    if model_type == "optogpt":
        return (
            OriginalDecoderWrapper(
                vocab_size=vocab_size,
                d_model=d_model,
                n_layers=n_blocks,
                n_heads=n_heads,
                d_ff=4 * d_model,
                dropout=dropout,
                max_len=max_stack_depth,
                mask_idx=mask_idx,
                pad_idx=pad_idx,
                eos_idx=eos_idx,
                spectrum_flat_dim=max(sample_spectrum.size()) * min(sample_spectrum.size()),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            .to(torch.float32)
            .to(device)
        )
    elif model_type == "optollama":
        return (
            OptoLlama(
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
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            .to(torch.float32)
            .to(device)
        )
    else:
        raise ValueError(f"Unsupported model key: {model_type}")
