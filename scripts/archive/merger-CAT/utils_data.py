import torch
from torch.utils.data import DataLoader, Subset, DistributedSampler
from utils import unique_length_int_generator
from pathlib import Path
from typing import List, Tuple, Optional
from safetensors.torch import load_file

class SpectraDataset(torch.utils.data.Dataset): #TODO in separate dataset.py
    """
    Dataset for Hugging Face `.safetensors` shards.

    Each file must contain:
      - 'spectra'     : Float tensor of shape [N, W, 3]
      - 'thin_films'  : Long tensor of shape [N, S]

    Returns from __getitem__:
      (spectrum_[W,3] float32, stack_[S] long, index)
    """

    def __init__(self, paths: List[str] | str, device: str = "cpu"):
        super().__init__()

        # normalize paths (accept dir or list of files)
        if isinstance(paths, str):
            paths = [paths]

        files: list[Path] = []
        for p in map(Path, paths):
            if p.is_dir():
                files.extend(sorted(p.glob("*.safetensors")))
            elif p.suffix == ".safetensors":
                files.append(p)
            else:
                raise ValueError(f"Unsupported path \
                                 (expect dir or .safetensors): {p}")

        if not files:
            raise FileNotFoundError("No .safetensors files found \
                                    in the provided paths.")

        spectra_list, stacks_list = [], []
        for fp in files:
            data = load_file(str(fp))
            if "spectra" not in data or "thin_films" not in data:
                raise KeyError(f"{fp} must contain 'spectra' and \
                               'thin_films' tensors.")
            spectra_list.append(data["spectra"].to(torch.float32))
            stacks_list.append(data["thin_films"].long())

        self.spectra = torch.cat(spectra_list, dim=0)  # [N, W, 3]
        self.stacks = torch.cat(stacks_list, dim=0)    # [N, S]
        self.maximum_depth = int(self.stacks.size(1))

        if self.spectra.size(0) != self.stacks.size(0):
            raise RuntimeError("Mismatched number of samples between spectra \
                               and thin_films.")

    def __len__(self) -> int:
        return int(self.spectra.size(0))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        return self.spectra[index], self.stacks[index], index


def make_loader(
    cfg,
    split: str,
    subset_n: Optional[int] = None,
    ddp: bool = False,
):
    """
    Build dataset, optional subset, sampler, and DataLoader
    for training or validation.

    Args:
        cfg: configuration module or object (must define dataset paths, batch size, etc.)
        split: 'train' or 'val' / 'validation'
        subset_n: optional number of samples to subset for quick runs
        ddp: whether to use DistributedSampler

    Returns:
        dataset, loader, sampler
    """
    # --- select dataset path(s) ---
    split_lower = split.lower()
    if split_lower in ("train", "training"):
        try:
            dataset_path = sorted([
                getattr(cfg, k) for k in dir(cfg) if k.startswith("PATH_TRAIN")])
        except:
            raise ValueError("cfg must have at least one argument \
                             staring with: 'PATH_TRAIN'")
    elif split_lower in ("valid", "validation", "test"):
        try:
            dataset_path = sorted([
                getattr(cfg, k) for k in dir(cfg) if k.startswith("PATH_VALID")])
        except:
            raise ValueError("cfg must have at least one argument \
                             staring with: 'PATH_VALID'")
    else:
        raise ValueError(f"Unknown split '{split}' — must be 'train' or 'valid'.")

    # --- build dataset ---
    ds = SpectraDataset(dataset_path, device="cpu")

    # --- optional subset for quick debugging ---
    if subset_n is not None and subset_n < len(ds):
        idxs = unique_length_int_generator(0e0, len(ds)-1, subset_n)
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
