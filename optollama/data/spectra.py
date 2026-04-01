from typing import Any, Optional

import torch
import torch.nn.functional as f

# ruff: noqa: E731


def ensure_3w(spectrum: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """
    Ensure a spectrum tensor has shape ``[..., 3, W]``, transposing if needed.

    Args
    ----
    spectrum : torch.Tensor
        Spectrum tensor of shape ``[3, W]``, ``[W, 3]``, ``[B, 3, W]``, or
        ``[B, W, 3]``.

    Returns
    -------
    tuple[torch.Tensor, bool]
        A 2-tuple of ``(tensor, was_transposed)`` where ``tensor`` has shape
        ``[..., 3, W]`` and ``was_transposed`` is ``True`` if the input was
        transposed.

    Raises
    ------
    ValueError
        If the input shape is not one of the supported formats.
    """
    if spectrum.dim() == 2:
        if spectrum.size(0) == 513:
            spectrum = spectrum.reshape(3, 171)
        elif spectrum.size(1) == 513:
            spectrum = spectrum.reshape(171, 3)
    if spectrum.dim() == 2:
        if spectrum.size(0) == 3:
            return spectrum, False
        elif spectrum.size(1) == 3:
            return spectrum.permute(1, 0).contiguous(), True
    elif spectrum.dim() == 3:
        if spectrum.size(1) == 3:
            return spectrum, False
        elif spectrum.size(2) == 3:
            return spectrum.permute(0, 2, 1).contiguous(), True

    raise ValueError(f"Expected shape [...,3,W] or [...,W,3], got {tuple(spectrum.shape)}")


def redistribute_mismatch(
    spectrum: torch.Tensor,
    order: str,
    target_sum: float = 1.0,
) -> torch.Tensor:
    """
    Enforce a per-wavelength channel sum by redistributing residual energy.

    Given a spectrum tensor with values in ``[0, 1]``, adjusts channel values
    so that ``R + A + T ≈ target_sum`` at every wavelength, distributing any
    deficit or excess according to the channel priority order.

    Args
    ----
    spectrum : torch.Tensor
        Spectrum tensor of shape ``[3, W]`` or ``[B, 3, W]`` with values in
        ``[0, 1]``.
    order : str
        Channel priority string (e.g. ``"R>A>T"``) controlling which channel
        absorbs residual energy first.
    target_sum : float
        Desired per-wavelength sum of all channels (default: ``1.0``).

    Returns
    -------
    torch.Tensor
        Adjusted spectrum tensor of the same shape, clamped to ``[0, 1]``.
    """
    orig_dim = spectrum.dim()
    if orig_dim == 2:
        spectrum = spectrum.unsqueeze(0)  # [1, 3, W]
    pri = parse_order(order)
    total = spectrum.sum(dim=1, keepdim=True)  # [B, 1, W]
    res = target_sum - total

    for idx in pri:
        ch = spectrum[:, idx : idx + 1, :]
        if (res.abs() < 1e-12).all():
            break
        add_capacity = (1.0 - ch).clamp_min(0.0)
        add = torch.sign(res) * torch.minimum(res.clamp_min(0.0), add_capacity)
        ch = ch + add
        res = res - add
        rem_capacity = ch.clamp_max(1.0)
        rem = -torch.minimum((-res).clamp_min(0.0), rem_capacity)
        ch = ch + rem
        res = res - rem
        spectrum[:, idx : idx + 1, :] = ch

    spectrum = spectrum.clamp_(0.0, 1.0)

    return spectrum.squeeze(0) if orig_dim == 2 else spectrum


def parse_order(order_str: str) -> tuple[int, ...]:
    """
    Parse a channel priority string into a tuple of channel indices.

    Args
    ----
    order_str : str
        Priority string such as ``"R>A>T"`` specifying the order in which
        channels are used to fill or crop residual energy.

    Returns
    -------
    tuple[int, ...]
        A 3-tuple of channel indices (0=R, 1=A, 2=T) in the specified
        priority order.
    """
    order_str = (order_str or "R>A>T").upper()
    mapping = {"R": 0, "A": 1, "T": 2}
    seq = [mapping[c.strip()] for c in order_str.split(">") if c.strip() in mapping]
    rest = [i for i in (0, 1, 2) if i not in seq]
    seq.extend(rest)

    return tuple(seq[:3])


def apply_stochastic_filler(
    spectrum: torch.Tensor,
    wavelengths: torch.Tensor,
    cfg: dict,
    seed: int,
    roi: list,
) -> torch.Tensor:
    """
    Modify ``spectrum`` outside a given ROI according to ``cfg``.

    The ROI region stays untouched. Outside the ROI, channel values are
    overwritten or perturbed depending on the chosen mode, then the full
    spectrum is renormalized to a physical RAT spectrum.

    Args
    ----
    spectrum : torch.Tensor
        Input RAT spectrum of shape ``[3, W]``.
    wavelengths : torch.Tensor
        Wavelength grid of shape ``[W]``.
    cfg : dict
        Configuration dict with keys (lowercase or uppercase accepted):

        - ``"ENABLED"`` (bool): if ``False``, return ``base`` unchanged.
        - ``"MODE"`` (str): one of ``"flat_random"``, ``"tilted_random"``,
          ``"smooth_random"``, or ``"prior_plus_noise"``.
        - ``"STRENGTH"`` (float): scale of the random perturbation.
        - ``"KERNEL_SIZE"`` (int): smoothing kernel size for smooth modes.
    seed : int
        The random seed.
    roi : list[float], optional
        Explicit ``[lo, hi]`` wavelength range to preserve.

    Returns
    -------
    torch.Tensor
        Augmented RAT spectrum of shape ``[3, W]``, normalized to sum to 1
        per wavelength.

    Raises
    ------
    ValueError
        If no ROI can be determined, or if ``mode`` is not recognized.
    """
    if not cfg["ENABLED"]:
        return spectrum

    low, high = roi[0], roi[1]
    roi_mask = (wavelengths >= low) & (wavelengths <= high)
    outside = ~roi_mask

    if not torch.any(outside):
        # ROI covers all wavelengths; nothing to fill
        return spectrum

    mode = cfg["MODE"]
    strength = cfg["STRENGTH"]
    kernel_size = cfg["KERNEL_SIZE"]

    # Local random helpers (optional reproducibility)
    g = torch.Generator(device=spectrum.device)
    g.manual_seed(seed)
    rand = lambda shape: torch.rand(shape, generator=g, device=spectrum.device)
    randn = lambda shape: torch.randn(shape, generator=g, device=spectrum.device)

    spectrum = spectrum.clone()
    r, a, t = spectrum[0], spectrum[1], spectrum[2]

    if mode == "flat_random":
        # Constant R/A outside ROI
        r0 = rand(()) * strength
        a0 = rand(()) * strength
        r[outside] = r0
        a[outside] = a0

    elif mode == "tilted_random":
        # Linear tilt of R/A over wavelength
        x = (wavelengths - wavelengths.min()) / (wavelengths.max() - wavelengths.min() + 1e-8)
        slope_r = (rand(()) * 2.0 - 1.0) * strength
        offset_r = rand(()) * strength
        slope_a = (rand(()) * 2.0 - 1.0) * strength
        offset_a = rand(()) * strength

        r_fill = torch.clamp(offset_r + slope_r * x, 0.0, 1.0)
        a_fill = torch.clamp(offset_a + slope_a * x, 0.0, 1.0)

        r[outside] = r_fill[outside]
        a[outside] = a_fill[outside]

    elif mode in ("smooth_random", "prior_plus_noise"):
        # Smooth random fields for R/A
        noise_r = randn(r.shape) * strength
        noise_a = randn(a.shape) * strength

        noise_r = smooth_1d_reflect(noise_r, kernel_size=kernel_size)
        noise_a = smooth_1d_reflect(noise_a, kernel_size=kernel_size)

        if mode == "smooth_random":
            r_fill = torch.clamp(noise_r, 0.0, 1.0)
            a_fill = torch.clamp(noise_a, 0.0, 1.0)
            r[outside] = r_fill[outside]
            a[outside] = a_fill[outside]
        else:  # prior_plus_noise
            r[outside] = torch.clamp(r[outside] + noise_r[outside], 0.0, 1.0)
            a[outside] = torch.clamp(a[outside] + noise_a[outside], 0.0, 1.0)

    else:
        raise ValueError(f"Unknown stochastic_filler.mode: {mode}")

    # Normalize to a physical RAT spectrum
    rn, an, tn = normalize_rat_fill_crop(r, a, t, target=1.0)

    return torch.stack([rn, an, tn], dim=0)


def smooth_1d_reflect(v: torch.Tensor, kernel_size: int = 15) -> torch.Tensor:
    """
    Smooth a 1-D tensor with a uniform moving-average and reflect padding.

    Args
    ----
    v : torch.Tensor
        1-D input tensor of shape ``[W]``.
    kernel_size : int
        Width of the uniform averaging kernel (default: ``15``).

    Returns
    -------
    torch.Tensor
        Smoothed 1-D tensor of the same shape as ``v``.
    """
    if kernel_size <= 1:
        return v
    
    pad = kernel_size // 2
    kernel = torch.ones(kernel_size, device=v.device, dtype=v.dtype) / kernel_size
    v_pad = torch.nn.functional.pad(v[None, None, :], (pad, pad), mode="reflect")
    k = kernel[None, None, :]

    return torch.nn.functional.conv1d(v_pad, k)[0, 0, : v.shape.numel()]


def normalize_rat_fill_crop(
    r: torch.Tensor,
    a: torch.Tensor,
    t: torch.Tensor,
    target: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize R, A, T channels so their per-wavelength sum equals ``target``.

    Uses T as the primary filler and crop source, then A, then R.

    Steps per wavelength:

    1. Clamp R, A, T to be non-negative.
    2. If ``R + A + T < target``: add the deficit to T.
    3. If ``R + A + T > target``: reduce T first; if still over, reduce A;
       then R. Values are never reduced below 0.

    Args
    ----
    r : torch.Tensor
        Reflectance channel, shape ``[W]``.
    a : torch.Tensor
        Absorptance channel, shape ``[W]``.
    t : torch.Tensor
        Transmittance channel, shape ``[W]``.
    target : float
        Desired per-wavelength sum (default: ``1.0``).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Normalized ``(r, a, t)`` float32 tensors, each of shape ``[W]``.
    """
    r = torch.clamp(torch.as_tensor(r), min=0.0)
    a = torch.clamp(torch.as_tensor(a), min=0.0)
    t = torch.clamp(torch.as_tensor(t), min=0.0)

    total = r + a + t
    deficit = target - total
    need = deficit > 0
    t[need] += deficit[need]

    excess = (r + a + t) - target
    over = torch.clamp(excess, 0.0)
    has_over = over > 0

    if torch.any(has_over):
        cut = torch.minimum(t[has_over], over[has_over])
        t[has_over] -= cut
        over[has_over] -= cut

        still = over > 0
        if torch.any(still):
            cut = torch.minimum(a[still], over[still])
            a[still] -= cut
            over[still] -= cut

        still = over > 0
        if torch.any(still):
            cut = torch.minimum(r[still], over[still])
            r[still] -= cut
            over[still] -= cut

    return r.to(torch.float32), a.to(torch.float32), t.to(torch.float32)


def apply_noise(
    spectrum: torch.Tensor,
    cfg: dict[str, Any],
    wavelengths: Any,
) -> torch.Tensor:
    """
    Apply additive Gaussian noise to a spectrum tensor.

    Args
    ----
    spectrum : torch.Tensor
        Spectrum tensor of shape ``[3, W]`` or ``[B, 3, W]``.
    cfg : dict[str, Any] or None
        Noise configuration dictionary. Keys may be lowercase or uppercase:

        - ``"ENABLED"`` (bool): whether to apply noise.
        - ``"SIGMA_ABS"`` (float): absolute noise standard deviation.
        - ``"SIGMA_REL"`` (float): relative noise standard deviation.
        - ``"PER_CHANNEL"`` (list[float]): per-channel scale factors.
        - ``"WL_MIN"`` and ``"WL_MAX"`` (float, optional):
          wavelength range to restrict noise to.
        - ``"CLIP_0_1"`` (bool): whether to clamp output to ``[0, 1]``.
    wavelengths : array-like or None
        Wavelength values used to build the optional wavelength mask.

    Returns
    -------
    torch.Tensor
        Noised spectrum tensor of the same shape as ``spectrum``.
    """
    if not cfg["ENABLED"]:
        return spectrum

    orig_dim = spectrum.dim()
    if orig_dim == 2:
        spectrum = spectrum.unsqueeze(0)  # [1, 3, W]
    _, c, w = spectrum.shape
    device = spectrum.device

    sigma_abs = cfg["SIGMA_ABS"]
    sigma_rel = cfg["SIGMA_REL"]

    per_ch = torch.tensor(cfg["PER_CHANNEL"], dtype=torch.float32, device=device).view(1, c, 1)

    mask = wavelength_mask(wavelengths, cfg["WL_MIN"], cfg["WL_MAX"], device)
    eps = torch.randn_like(spectrum) * (sigma_abs + sigma_rel * spectrum) * per_ch

    if mask is not None:
        spectrum = torch.where(mask.view(1, 1, w), spectrum + eps, spectrum)
    else:
        spectrum = spectrum + eps

    if cfg["CLIP_0_1"]:
        spectrum = spectrum.clamp_(0.0, 1.0)

    return spectrum.squeeze(0) if orig_dim == 2 else spectrum


def wavelength_mask(wavelengths: Any, wl_min: float, wl_max: float, device: str) -> Optional[torch.Tensor]:
    """
    Build a boolean mask that is ``True`` within a wavelength range.

    Args
    ----
    wavelengths : array-like or None
        1-D sequence of wavelength values of length ``W``. If ``None``,
        the function returns ``None`` immediately.
    wl_min : float
        Lower bound of the wavelength range (inclusive).
    wl_max : float
        Upper bound of the wavelength range (inclusive).
    device : str or torch.device
        Device on which to create the output tensor.

    Returns
    -------
    torch.Tensor or None
        Boolean tensor of shape ``[W]`` where entry ``i`` is ``True`` if
        ``wl_min <= wavelengths[i] <= wl_max``, or ``None`` if
        ``wavelengths`` is ``None``.
    """
    if wavelengths is None:
        return None
    wl = torch.as_tensor(wavelengths, dtype=torch.float32, device=device)
    return (wl >= float(wl_min)) & (wl <= float(wl_max))


def apply_smoothing(
    spectrum: torch.Tensor,
    cfg: Optional[dict[str, Any]],
) -> torch.Tensor:
    """
    Apply 1-D smoothing to a spectrum tensor.

    Args
    ----
    spectrum : torch.Tensor
        Spectrum tensor of shape ``[3, W]`` or ``[B, 3, W]``.
    cfg : dict[str, Any] or None
        Smoothing configuration dictionary. Keys may be lowercase or uppercase:

        - ``"enabled"`` / ``"ENABLED"`` (bool): whether to apply smoothing.
        - ``"method"`` / ``"METHOD"`` (str): ``"gaussian"`` or ``"boxcar"``.
        - ``"win"`` / ``"WIN"`` (int): kernel window size.
        - ``"sigma"`` / ``"SIGMA"`` (float): Gaussian standard deviation.

    Returns
    -------
    torch.Tensor
        Smoothed spectrum tensor of the same shape as ``spectrum``.
    """
    if not cfg["ENABLED"]:
        return spectrum

    return smooth_1d(
        spectrum,
        method=cfg["METHOD"],
        sigma=cfg["SIGMA"],
        win=cfg["WIN"],
    )


def smooth_1d(x_3w: torch.Tensor, method: str, win: int, sigma: float) -> torch.Tensor:
    """
    Apply depthwise 1-D smoothing along the wavelength axis.

    Args
    ----
    x_3w : torch.Tensor
        Input tensor of shape ``[3, W]`` or ``[B, 3, W]``.
    method : str
        Smoothing method: ``"gaussian"`` or ``"boxcar"``.
    win : int
        Kernel window size (rounded up to nearest odd integer).
    sigma : float
        Standard deviation used when ``method="gaussian"``.

    Returns
    -------
    torch.Tensor
        Smoothed tensor of the same shape as ``x_3w``.
    """
    orig_dim = x_3w.dim()
    if orig_dim == 2:
        x_3w = x_3w.unsqueeze(0)  # [1, 3, W]
    _, c, w = x_3w.shape
    device = x_3w.device

    k = gauss_kernel(win, sigma, device) if method == "gaussian" else boxcar_kernel(win, device)
    weight = k.view(1, 1, -1).repeat(c, 1, 1)  # [C, 1, K]
    x = f.pad(x_3w, (k.numel() // 2, k.numel() // 2), mode="reflect")
    x = f.conv1d(x, weight, groups=c)[:, :, :w]

    return x.squeeze(0) if orig_dim == 2 else x


def gauss_kernel(win: int, sigma: float, device: torch.device) -> torch.Tensor:
    """
    Build a normalised 1-D Gaussian kernel.

    The kernel length is rounded up to the nearest odd integer so that it
    has a well-defined centre. Values are computed from the standard
    Gaussian formula and then normalised to sum to 1.

    Parameters
    ----------
    win : int
        Desired kernel window size. Rounded up to the nearest odd integer
        if even.
    sigma : float
        Standard deviation of the Gaussian in samples.
    device : torch.device
        Device on which to create the kernel tensor.

    Returns
    -------
    torch.Tensor
        1-D float32 tensor of shape ``[win]`` (after odd-rounding) whose
        values sum to 1.
    """
    if win % 2 == 0:
        win += 1
    half = win // 2
    x = torch.arange(-half, half + 1, dtype=torch.float32, device=device)
    kernel = torch.exp(-0.5 * (x / max(sigma, 1e-6)) ** 2)

    return kernel / kernel.sum()


def boxcar_kernel(win: int, device: torch.device) -> torch.Tensor:
    """
    Build a normalised 1-D boxcar (uniform moving-average) kernel.

    The kernel length is rounded up to the nearest odd integer so that it
    has a well-defined centre. All values are equal and sum to 1.

    Parameters
    ----------
    win : int
        Desired kernel window size. Rounded up to the nearest odd integer
        if even.
    device : torch.device
        Device on which to create the kernel tensor.

    Returns
    -------
    torch.Tensor
        1-D float32 tensor of shape ``[win]`` (after odd-rounding) with
        all values equal to ``1 / win``.
    """
    if win % 2 == 0:
        win += 1

    return torch.ones(win, dtype=torch.float32, device=device) / win
