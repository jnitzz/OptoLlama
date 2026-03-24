import torch


def token_accuracy(
    stacks: torch.Tensor,
    preds: torch.Tensor,
    eos: int,
    pad: int,
    msk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute weighted global token accuracy and per-sample accuracy.

    Args
    ----
    stacks : torch.Tensor
        Target token IDs of shape ``[B, L]``.
    preds : torch.Tensor
        Either predicted logits of shape ``[B, L, V]`` or predicted token
        IDs of shape ``[B, L]``.
    eos : int
        Token ID for EOS.
    pad : int
        Token ID for PAD (ignored in accuracy computation).
    msk : int
        Token ID for MSK (ignored in accuracy computation).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A 2-tuple of:

        - Scalar (0-D) float tensor on CPU with weighted global accuracy.
        - Float tensor of shape ``[B]`` on CPU with accuracy per batch sample.
    """
    # Convert logits to token IDs
    if preds.dim() == 3:
        preds = preds.argmax(dim=-1)

    len_stack = min(stacks.size(1), preds.size(1))
    stacks = stacks[:, :len_stack]
    preds = preds[:, :len_stack]

    # Identify valid evaluation positions (before first EOS)
    is_eos = stacks == eos
    before_first_eos = is_eos.cumsum(dim=1) == 0

    # Exclude PAD & MSK tokens
    valid = before_first_eos & (stacks != pad) & (stacks != msk)

    # Correct predictions at valid positions
    correct = (stacks == preds) & valid

    # Per-sample accuracy
    per_correct = correct.sum(dim=1).float()
    per_total = valid.sum(dim=1).clamp_min(1).float()
    per_sample = (per_correct / per_total).detach().cpu()

    # Global weighted accuracy
    global_acc = (per_correct.sum() / per_total.sum()).detach().cpu()

    return global_acc, per_sample


def masked_mae(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Mean Absolute Error over only valid (finite) predictions.

    Args
    ----
    x : torch.Tensor
        Ground-truth spectra, shape ``[B, C, W]``.
    y : torch.Tensor
        Predicted spectra, same shape. Non-finite entries are ignored.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``[B]`` containing per-sample masked MAE.
    """
    # Valid mask: all channels finite
    mask = torch.isfinite(y).all(dim=-1, keepdim=True)  # [B, C, 1]
    valid = mask.expand_as(y)  # [B, C, W]

    abs_err = torch.abs(x - torch.nan_to_num(y))
    masked_err = abs_err.where(valid, torch.zeros_like(abs_err))

    num = masked_err.sum(dim=1).sum(dim=1)
    den = valid.sum(dim=1).sum(dim=1).clamp_min(1)

    return num / den


def masked_mae_roi(
    x: torch.Tensor,
    y: torch.Tensor,
    wl_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute Mean Absolute Error over finite predictions and a wavelength ROI.

    Args
    ----
    x : torch.Tensor
        Ground-truth spectra, shape ``[B, C, W]``.
    y : torch.Tensor
        Predicted spectra, same shape. Non-finite entries are ignored.
    wl_mask : torch.Tensor, optional
        Boolean mask of shape ``[W]`` selecting the Region of Interest (ROI).
        ``True`` entries are included in the MAE computation.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``[B]`` containing per-sample masked MAE.
    """
    # x,y: [B,3,W]
    # wl_mask: [W] bool, True = included in MAE

    # finite-mask logic
    finite_mask = torch.isfinite(y).all(dim=-1, keepdim=True)  # [B,3,1]
    valid = finite_mask.expand_as(y)  # [B,3,W]

    if wl_mask is not None:
        wl_mask = wl_mask.view(1, 1, -1)  # [1,1,W]
        valid = valid & wl_mask  # [B,3,W]

    abs_err = torch.abs(x - torch.nan_to_num(y))
    masked_err = abs_err.where(valid, torch.zeros_like(abs_err))

    num = masked_err.sum(dim=1).sum(dim=1)
    den = valid.sum(dim=1).sum(dim=1).clamp_min(1)

    return num / den
