from __future__ import annotations

import re
from pathlib import Path
from types import ModuleType
from typing import Any, List, Optional, Tuple, Union

import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset
from utils import apply_noise, apply_smoothing, normalize_rat_fill_crop, redistribute_mismatch, unique_length_int_generator

# ruff: noqa: E731


class SpectraDataset(torch.utils.data.Dataset):
    """
    Dataset for Hugging Face `.safetensors` shards.

    Each file must contain:
      - 'spectra'     : Float tensor of shape [N, 3, W]
      - 'thin_films'  : Long tensor of shape [N, S]

    Returns from __getitem__:
      (spectrum_[3,W] float32, stack_[S] long, index)
    """

    def __init__(self, paths: List[str] | str):
        super().__init__()

        # normalize paths (accept dir or list of files)
        if isinstance(paths, str):
            paths = [paths]

        files: list[Path] = []
        for p in map(Path, paths):
            if p.is_dir():
                files.extend(p.glob("*.safetensors"))
            elif p.suffix == ".safetensors":
                files.append(p)
            else:
                raise ValueError(
                    f"Unsupported path \
                                 (expect dir or .safetensors): {p}"
                )

        if not files:
            raise FileNotFoundError(
                "No .safetensors files found \
                                    in the provided paths."
            )

        def shard_sort_key(path: Path) -> Tuple[str, Union[int, float]]:
            m = re.match(r"^(.*?)(\d+)$", path.stem.lower())
            if m:
                prefix, num = m.groups()
                return (prefix, int(num))
            return (path.stem.lower(), float("inf"))

        files = sorted(files, key=shard_sort_key)

        spectra_list, stacks_list = [], []
        for fp in files:
            data = load_file(str(fp))
            if "spectra" not in data or "thin_films" not in data:
                raise KeyError(
                    f"{fp} must contain 'spectra' and \
                               'thin_films' tensors."
                )
            spectra_list.append(data["spectra"].to(torch.float32))
            stacks_list.append(data["thin_films"].long())

        self.spectra = torch.cat(spectra_list, dim=0)  # [N, 3, W]
        self.stacks = torch.cat(stacks_list, dim=0)  # [N, S]
        self.maximum_depth = int(self.stacks.size(1))
        self.length_dataset = int(self.spectra.size(0))

        if self.spectra.size(0) != self.stacks.size(0):
            raise RuntimeError(
                "Mismatched number of samples between spectra \
                               and thin_films."
            )

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.length_dataset

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Return specra, stacks and index."""
        return self.spectra[index], self.stacks[index], index


def make_loader(
    cfg: ModuleType, split: str, subset_n: Optional[int] = None, ddp: bool = False
) -> Tuple[Union[SpectraDataset, Subset[SpectraDataset]], DataLoader, Optional[DistributedSampler]]:
    """
    Build dataset, optional subset, sampler, and DataLoader in train/valid.

    Args:
        cfg: configuration module or object (must define dataset paths, batch size, etc.)
        split: 'train' or 'val' / 'validation'
        subset_n: optional number of samples to subset for quick runs
        ddp: whether to use DistributedSampler

    Returns
    -------
        dataset, loader, sampler
    """
    split_lower = split.lower()
    if split_lower in ("train", "training"):
        try:
            dataset_path = sorted([getattr(cfg, k) for k in dir(cfg) if k.startswith("PATH_TRAIN")])
        except ValueError:
            raise ValueError(
                "cfg must have at least one argument \
                             staring with: 'PATH_TRAIN'"
            )
    elif split_lower in ("valid", "validation", "test"):
        try:
            dataset_path = sorted([getattr(cfg, k) for k in dir(cfg) if k.startswith("PATH_VALID")])
        except ValueError:
            raise ValueError(
                "cfg must have at least one argument \
                             staring with: 'PATH_VALID'"
            )
    else:
        raise ValueError(f"Unknown split '{split}' — must be 'train' or 'valid'.")

    # --- build dataset ---
    ds = SpectraDataset(dataset_path)

    # --- optional subset for quick debugging ---
    if subset_n is not None and subset_n < ds.length_dataset:
        idxs = unique_length_int_generator(0e0, ds.length_dataset - 1, subset_n)
        ds = Subset(ds, idxs)

    # --- configure sampler and shuffling ---
    if split_lower == "train":
        sampler = DistributedSampler(ds, shuffle=True) if ddp else None
        shuffle = not ddp
        drop_last = True
    elif split_lower == "valid":
        sampler = DistributedSampler(ds, shuffle=False) if ddp else None
        shuffle = False
        drop_last = False
    else:
        raise ValueError("either 'train' or 'valid'")

    # --- build DataLoader ---
    batch_size = getattr(cfg, "BATCH_SIZE", getattr(cfg, "batch_size", 256))
    num_workers = getattr(cfg, "NUM_WORKERS", getattr(cfg, "num_workers", 16))

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )

    return ds, loader, sampler


def smooth_1d_reflect(v: torch.Tensor, kernel_size: int = 15) -> torch.Tensor:
    """Smooth 1D moving-average with reflect padding."""
    if kernel_size <= 1:
        return v
    pad = kernel_size // 2
    kernel = torch.ones(kernel_size, device=v.device, dtype=v.dtype) / kernel_size
    v_pad = torch.nn.functional.pad(v[None, None, :], (pad, pad), mode="reflect")
    k = kernel[None, None, :]
    return torch.nn.functional.conv1d(v_pad, k)[0, 0, : v.shape.numel()]


def apply_stochastic_filler(
    base: torch.Tensor,
    wl: torch.Tensor,
    data: dict,
    *,
    seed_override: Optional[int] = None,
) -> torch.Tensor:
    """
    Modify `base` outside a given ROI according to `data['stochastic_filler']`.

    - ROI stays untouched.
    - Outside ROI is overwritten / perturbed depending on mode.
    """
    stoch = data.get("stochastic_filler")
    if not stoch:
        return base

    roi = stoch.get("roi") or stoch.get("ROI")
    if not roi or len(roi) != 2:
        raise ValueError("stochastic_filler.roi must be a 2-element list [lo, hi].")

    lo, hi = float(roi[0]), float(roi[1])
    roi_mask = (wl >= lo) & (wl <= hi)
    outside = ~roi_mask
    if not torch.any(outside):
        # ROI covers all wavelengths; nothing to fill
        return base

    mode = str(stoch.get("mode", "smooth_random")).lower()
    strength = float(stoch.get("strength", 0.3))
    kernel_size = int(stoch.get("kernel_size", 15))
    seed = seed_override if seed_override is not None else stoch.get("seed", None)

    # Local random helpers (optional reproducibility)
    if seed is not None:
        g = torch.Generator(device=base.device)
        g.manual_seed(int(seed))
        rand = lambda shape: torch.rand(shape, generator=g, device=base.device)
        randn = lambda shape: torch.randn(shape, generator=g, device=base.device)
    else:
        rand = lambda shape: torch.rand(shape, device=base.device)
        randn = lambda shape: torch.randn(shape, device=base.device)

    base = base.clone()
    r, a, t = base[0], base[1], base[2]

    if mode == "flat_random":
        # Constant R/A outside ROI
        r0 = rand(()) * strength
        a0 = rand(()) * strength
        r[outside] = r0
        a[outside] = a0

    elif mode == "tilted_random":
        # Linear tilt of R/A over wavelength
        x = (wl - wl.min()) / (wl.max() - wl.min() + 1e-8)
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


class RepeatedSpecDataset(Dataset):
    """
    Dataset that repeats a *base* [3,W] spectrum N times.

    If NOISE.enabled=True, each item draws fresh noise (and smoothing).
    """

    def __init__(
        self,
        base_spec_3w: torch.Tensor,  # [3,W] before noise/smoothing
        n_items: int,
        max_stack_depth: int,
        pad_idx: int,
        wavelengths: Any,  # np.ndarray [W] (used for wl masks)
        noise_cfg: dict | None,
        smooth_cfg: dict | None,
        mismatch_order: str = "R>A>T",
        stochastic_filler_cfg: dict | None = None,
        seed: int = 0,
    ):
        assert base_spec_3w.ndim == 2 and base_spec_3w.shape[0] == 3, "base_spec must be [3,W]"
        self.base = base_spec_3w.detach().clone()  # untouched template
        self.n = int(n_items)
        self.maximum_depth = int(max_stack_depth)
        self.pad_idx = int(pad_idx)
        self.wavelengths = wavelengths
        self.noise_cfg = noise_cfg or {"enabled": False}
        self.smooth_cfg = smooth_cfg or {"enabled": False}
        self.mismatch_order = mismatch_order
        self.stochastic_filler_cfg = stochastic_filler_cfg or {}
        self.seed = int(seed)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.n

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return specraand stacks."""
        stacks = torch.full((self.maximum_depth,), self.pad_idx)  # dummy; eval uses TMM
        specs = self.base.clone()  # [3,W]

        # Zeroth target: exact original, no stochastic filling
        if index == 0 and self.stochastic_filler_cfg.get("skip_index0", True):
            specs = redistribute_mismatch(specs, self.mismatch_order, target_sum=1.0)
            return specs, stacks

        # 1) vary outside ROI first (keeps ROI untouched)
        seed_i = self.seed + int(index)  # optionally also + 100000 * rank
        specs = apply_stochastic_filler(
            specs, self.wavelengths, {"stochastic_filler": self.stochastic_filler_cfg}, seed_override=seed_i
        )

        # 2) optionally also apply noise/smoothing (if enabled)
        specs = apply_noise(specs, self.noise_cfg, self.wavelengths)
        specs = apply_smoothing(specs, self.smooth_cfg)

        # 3) enforce sum-to-1
        specs = redistribute_mismatch(specs, self.mismatch_order, target_sum=1.0)
        return specs, stacks


def make_repeated_spec_loader(
    base_spec_3w: torch.Tensor,
    n_items: int,
    max_stack_depth: int,
    pad_idx: int,
    wavelengths: Any,
    cfg: None = None,
    batch_size: int = 1,
) -> Tuple[Union[SpectraDataset, Subset[SpectraDataset]], DataLoader, None]:
    """
    Build dataset, sampler, and DataLoader for inference.

    Args:
        base_spec_3w: base spectrum target (3,W)
        n_items: number of noise variations for the base spectrum
        max_stack_depth: maximum stack depth allowed
        pad_idx: token <PAD> to fill voide
        wavelengths: to apply noise in the correct format
        cfg: config
        batch_size: batch_size

    Returns
    -------
        dataset, loader, sampler
    """
    roi_min = getattr(cfg, "ROI_MIN", None)
    roi_max = getattr(cfg, "ROI_MAX", None)

    stochastic_filler_cfg = getattr(cfg, "FILL_OUTSIDE_ROI", None) if cfg is not None else None
    if stochastic_filler_cfg is not None and roi_min is not None and roi_max is not None:
        stochastic_filler_cfg = dict(stochastic_filler_cfg)  # shallow copy
        stochastic_filler_cfg["roi"] = [roi_min, roi_max]
    noise_cfg = getattr(cfg, "NOISE", None) if cfg is not None else None
    smooth_cfg = getattr(cfg, "SMOOTH", None) if cfg is not None else None
    mismatch_order = getattr(cfg, "MISMATCH_FILL_ORDER", "R>A>T") if cfg is not None else "R>A>T"
    seed = int(getattr(cfg, "SEED", 0)) if cfg is not None else 0

    ds = RepeatedSpecDataset(
        base_spec_3w,
        n_items,
        max_stack_depth,
        pad_idx,
        wavelengths=wavelengths,
        noise_cfg=noise_cfg,
        smooth_cfg=smooth_cfg,
        mismatch_order=mismatch_order,
        stochastic_filler_cfg=stochastic_filler_cfg,
        seed=seed,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return ds, loader, None
