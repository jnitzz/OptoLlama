import torch


def token_accuracy_counts(
    stacks: torch.Tensor,
    preds: torch.Tensor,
    eos: int,
    pad: int,
    msk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute raw token-accuracy counts after applying the evaluation mask.

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
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        A 4-tuple of:

        - ``correct_count``: scalar float tensor on CPU with the number of
          correct valid tokens in the batch.
        - ``total_count``: scalar float tensor on CPU with the number of
          valid tokens in the batch.
        - ``per_correct``: float tensor ``[B]`` with correct-token counts.
        - ``per_total``: float tensor ``[B]`` with valid-token counts.
    """
    if preds.dim() == 3:
        preds = preds.argmax(dim=-1)

    len_stack = min(stacks.size(1), preds.size(1))
    stacks = stacks[:, :len_stack]
    preds = preds[:, :len_stack]

    is_eos = stacks == eos
    before_first_eos = is_eos.cumsum(dim=1) == 0
    valid = before_first_eos & (stacks != pad) & (stacks != msk)
    correct = (stacks == preds) & valid

    per_correct = correct.sum(dim=1).float()
    per_total = valid.sum(dim=1).float()

    correct_count = per_correct.sum().detach().cpu()
    total_count = per_total.sum().detach().cpu()

    return correct_count, total_count, per_correct, per_total


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
    correct_count, total_count, per_correct, per_total = token_accuracy_counts(
        stacks,
        preds,
        eos,
        pad,
        msk,
    )
    per_total = per_total.clamp_min(1.0)
    per_sample = (per_correct / per_total).detach().cpu()

    global_acc = (correct_count / total_count.clamp_min(1.0)).detach().cpu()

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
    channel_mask: torch.Tensor = None,
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

    if channel_mask is not None:
        channel_mask = channel_mask.to(device=valid.device, dtype=torch.bool).view(1, -1, 1)
        valid = valid & channel_mask

    abs_err = torch.abs(x - torch.nan_to_num(y))
    masked_err = abs_err.where(valid, torch.zeros_like(abs_err))

    num = masked_err.sum(dim=1).sum(dim=1)
    den = valid.sum(dim=1).sum(dim=1).clamp_min(1)

    return num / den


def interpolate_spectra_to_wavelengths(
    spectra: torch.Tensor,
    source_wavelengths: torch.Tensor,
    target_wavelengths: torch.Tensor,
) -> torch.Tensor:
    """
    Linearly interpolate spectra from one wavelength grid to another.

    Args
    ----
    spectra : torch.Tensor
        Spectra with shape ``[B, C, W]``.
    source_wavelengths : torch.Tensor
        Monotonic wavelength grid of shape ``[W]`` for ``spectra``.
    target_wavelengths : torch.Tensor
        Target wavelength grid of shape ``[W2]``.

    Returns
    -------
    torch.Tensor
        Interpolated spectra with shape ``[B, C, W2]``.
    """
    if spectra.dim() != 3:
        raise ValueError(f"Expected spectra shape [B, C, W], got {tuple(spectra.shape)}")

    device = spectra.device
    dtype = torch.float32
    source = torch.as_tensor(source_wavelengths, device=device, dtype=dtype)
    target = torch.as_tensor(target_wavelengths, device=device, dtype=dtype)
    spectra = spectra.to(dtype)

    if source.numel() != spectra.size(-1):
        raise ValueError(
            f"Source wavelength count ({source.numel()}) does not match spectra width ({spectra.size(-1)})."
        )
    if source.numel() < 2:
        raise ValueError("At least two source wavelengths are required for interpolation.")
    if target.numel() == 0:
        raise ValueError("Target wavelength grid is empty.")
    if torch.any(source[1:] <= source[:-1]):
        raise ValueError("Source wavelengths must be strictly increasing.")

    if source.numel() == target.numel() and torch.allclose(source, target):
        return spectra

    if torch.any(target < source[0]) or torch.any(target > source[-1]):
        raise ValueError(
            "Target wavelengths must be inside the source wavelength range "
            f"({float(source[0]):g}-{float(source[-1]):g} nm)."
        )

    idx_hi = torch.searchsorted(source, target).clamp(1, source.numel() - 1)
    idx_lo = idx_hi - 1

    src_lo = source.index_select(0, idx_lo)
    src_hi = source.index_select(0, idx_hi)
    denom = (src_hi - src_lo).clamp_min(torch.finfo(dtype).eps)
    weight = ((target - src_lo) / denom).view(1, 1, -1)

    y_lo = spectra.index_select(-1, idx_lo)
    y_hi = spectra.index_select(-1, idx_hi)
    return y_lo + (y_hi - y_lo) * weight


def resampled_mae(
    x: torch.Tensor,
    y: torch.Tensor,
    source_wavelengths: torch.Tensor,
    target_wavelengths: torch.Tensor,
    channel_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute MAE after resampling both spectra to a common wavelength grid.

    The same finite-prediction masking as :func:`masked_mae` is used after
    interpolation.
    """
    x_resampled = interpolate_spectra_to_wavelengths(x, source_wavelengths, target_wavelengths)
    y_resampled = interpolate_spectra_to_wavelengths(y, source_wavelengths, target_wavelengths)
    return masked_mae_roi(x_resampled, y_resampled, channel_mask=channel_mask)
