import re
from bisect import bisect_right
from pathlib import Path
from typing import Any, Optional, Self, Union

import safetensors.torch
import torch
from safetensors import safe_open
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset, Subset

import optollama.data.spectra

# ruff: noqa: E731


def collect_safetensor_paths(paths: list[str] | str) -> list[str]:
    """Expand configured dataset paths into sorted ``.safetensors`` shard paths."""
    if isinstance(paths, str):
        paths = [paths]

    expanded_paths: list[str] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            expanded_paths.extend(str(fp) for fp in sorted(path.glob("*.safetensors")))
        else:
            expanded_paths.append(str(path))

    sorted_paths = sorted(expanded_paths, key=SpectraDataset.shard_sort_key)
    if not sorted_paths:
        raise FileNotFoundError("No .safetensors files found for SpectraDataset.")
    return sorted_paths


def safetensor_shape(path: str, tensor_name: str) -> tuple[int, ...]:
    """Read a tensor shape from a safetensors shard without loading its data."""
    with safe_open(path, framework="pt", device="cpu") as f:
        if tensor_name not in f.keys():
            raise KeyError(f"{path} must contain {tensor_name!r} tensor.")
        return tuple(int(x) for x in f.get_slice(tensor_name).get_shape())


class SpectraDataset(torch.utils.data.Dataset):
    """
    Dataset for Hugging Face `.safetensors` shards.

    Each file must contain:
      - 'spectra'    : float tensor of shape [N, 3, W]
      - 'thin_films' : long tensor of shape [N, S]
    """

    def __init__(self, paths: list[str] | str):
        super().__init__()

        paths = collect_safetensor_paths(paths)

        spectra_list, stacks_list = [], []
        for fp in paths:
            data = safetensors.torch.load_file(fp, device="cpu")
            spectra_list.append(data["spectra"].to(torch.float32))
            stacks_list.append(data["thin_films"].long())

        self.spectra = torch.cat(spectra_list, dim=0)  # [N, 3, W]
        self.stacks = torch.cat(stacks_list, dim=0)  # [N, S]
        self.maximum_depth = int(self.stacks.size(1))
        self.length_dataset = int(self.spectra.size(0))

        if self.spectra.size(0) != self.stacks.size(0):
            raise RuntimeError("Mismatched number of samples between spectra and thin_films.")

    @staticmethod
    def indices_of_unique_equidistant_subset(start: int, stop: int, amount: int) -> torch.Tensor:
        """
        Generate a tensor of ``amount`` unique, evenly-spaced integer indices.

        Requires ``-1 < start < stop`` and ``0 < amount <= stop``.

        Args
        ----
        start : int
            The start index to subset from.
        stop : int
            The exclusive upper bound for the subset.
        amount : int
            The number of unique indices to return.

        Returns
        -------
        torch.Tensor
            1-D integer tensor of ``amount`` unique indices in ``[start, stop)``.

        Raises
        ------
        ValueError
            If the constraints ``-1 < start < stop`` or ``0 < amount <= stop``
            are not satisfied.
        """
        if not (-1 < start < stop) or not (0 < amount <= stop):
            raise ValueError(
                f"Invalid arguments: start={start}, stop={stop}, amount={amount}. Require (-1 < start < stop) and (0 < amount <= stop)."
            )

        len_unique = -1
        amount = amount - 1

        while len_unique < amount:
            amount = amount + 1
            subset_idx = torch.linspace(start, stop - 1, amount, dtype=torch.int).unique()
            len_unique = len(subset_idx)

        return subset_idx

    @staticmethod
    def shard_sort_key(path: str) -> tuple[str, int]:
        """
        Sorting lambda that sort file name lexicographic for their path prefixes and integer-based for their suffixes.

        Args
        ----
        path: str
            The path name to convert into a sorting key.

        Returns
        -------
        tuple[str, int]
            The file prefix and number.
        """
        m = re.match(r"^(.*?)(\d+)$", path)
        if m:
            prefix, num = m.groups()
            return (
                prefix,
                int(num),
            )

        return (
            path,
            float("inf"),
        )

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns
        -------
        int
            The number of items.
        """
        return self.length_dataset

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return specra, stacks and index.

        Args
        ----
        index: int
            The index of the data items to return.

        Returns
        -------
        tuple
            Spectrum of shape [3,W] in float32 and stack tokens of shape [S] as longs and the passed index number.
        """
        return self.spectra[index], self.stacks[index], index

    @classmethod
    def make_loader(
        cls, cfg: dict, split: str, subset_n: int = None, ddp: bool = False
    ) -> tuple[Union[Self, Subset[Self]], DataLoader, DistributedSampler]:
        """
        Build dataset, optional subset, sampler, and DataLoader in train/test.

        Args
        ----
        cfg: dict
            Configuration mapping.
        split: str
            'train' or 'test'.
        subset_n: int
            Optional number of samples to subset for quick runs, defaults to None (aka all items).
        ddp: bool
            Whether to use in data parallel mode, defaults to False.

        Returns
        -------
        Union[SpectraDataset, Subset[SpectraDataset]]
            The loaded dataset
        DataLoader
            The dataloader wrapping the above dataset.
        DistributedSampler
            The indices sampler.
        """
        split_lower = split.lower()
        if split_lower not in ("train", "test"):
            raise ValueError(f"Unknown data split {split_lower}, expected 'train' or 'test'")

        search_string = "DATA_PATH_TRAIN" if split_lower == "train" else "DATA_PATH_TEST"
        dataset_path = sorted([cfg[k] for k in cfg.keys() if k.startswith(search_string)])
        batch_size = cfg["TRAIN_BATCH_SIZE"] if split_lower == "train" else cfg["TEST_BATCH_SIZE"]
        num_workers = cfg["NUM_WORKERS"]

        if cfg.get("SHARDED_LOADING", False):
            if ddp and torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
            else:
                rank = 0
                world_size = 1

            dataset = ShardedSpectraDataset(
                dataset_path,
                split=split_lower,
                subset_n=subset_n,
                rank=rank,
                world_size=world_size,
                seed=int(cfg.get("SEED") or 0),
                shuffle=(split_lower == "train"),
            )

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                sampler=None,
                num_workers=0,
                pin_memory=not torch.mps.is_available(),
                drop_last=(split_lower == "train"),
            )
            return dataset, loader, None

        # --- build dataset ---
        dataset = cls(dataset_path)

        # --- optional subset for quick debugging ---
        if subset_n is not None and subset_n < dataset.length_dataset:
            idxs = cls.indices_of_unique_equidistant_subset(0, dataset.length_dataset - 1, subset_n)
            dataset = Subset(dataset, idxs)

        # --- configure sampler and shuffling ---
        if split_lower == "train":
            sampler = DistributedSampler(dataset, shuffle=True) if ddp else None
            shuffle = not ddp
            drop_last = True
        else:
            sampler = DistributedSampler(dataset, shuffle=False) if ddp else None
            shuffle = False
            drop_last = False

        # --- build DataLoader ---
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=not torch.mps.is_available(),
            drop_last=drop_last,
        )

        return dataset, loader, sampler


class ShardedSpectraDataset(IterableDataset):
    """
    Iterable dataset that streams one safetensors shard at a time.

    This avoids the eager ``torch.cat`` used by :class:`SpectraDataset`, which
    is too memory-heavy for multi-million-sample DDP training. DDP ranks receive
    disjoint contiguous global ranges with equal length so each rank produces
    the same number of batches.
    """

    def __init__(
        self,
        paths: list[str] | str,
        *,
        split: str,
        subset_n: Optional[int] = None,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")
        if not (0 <= rank < world_size):
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")

        self.paths = collect_safetensor_paths(paths)
        self.split = split
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed or 0)
        self.shuffle = bool(shuffle)
        self.epoch = 0

        self.shard_lengths: list[int] = []
        self.shard_offsets: list[int] = [0]
        spectra_shape = None
        stack_shape = None
        for path in self.paths:
            spectra_shape_i = safetensor_shape(path, "spectra")
            stack_shape_i = safetensor_shape(path, "thin_films")
            if len(spectra_shape_i) != 3 or spectra_shape_i[1] != 3:
                raise ValueError(f"'spectra' in {path} must be [N,3,W], got {spectra_shape_i}")
            if len(stack_shape_i) != 2:
                raise ValueError(f"'thin_films' in {path} must be [N,S], got {stack_shape_i}")
            if spectra_shape_i[0] != stack_shape_i[0]:
                raise RuntimeError(f"Mismatched number of samples in {path}: {spectra_shape_i[0]} vs {stack_shape_i[0]}")
            if spectra_shape is None:
                spectra_shape = spectra_shape_i
                stack_shape = stack_shape_i
            elif spectra_shape_i[1:] != spectra_shape[1:] or stack_shape_i[1:] != stack_shape[1:]:
                raise ValueError(
                    f"Inconsistent shard shapes in {path}: spectra {spectra_shape_i}, thin_films {stack_shape_i}; "
                    f"expected spectra [N,{spectra_shape[1]},{spectra_shape[2]}] and thin_films [N,{stack_shape[1]}]"
                )

            n = int(spectra_shape_i[0])
            self.shard_lengths.append(n)
            self.shard_offsets.append(self.shard_offsets[-1] + n)

        self.total_available = int(self.shard_offsets[-1])
        requested_total = self.total_available if subset_n is None else min(int(subset_n), self.total_available)
        if requested_total <= 0:
            raise ValueError(f"{split} subset contains no samples: subset_n={subset_n}, available={self.total_available}")

        self.total_samples = int(requested_total)
        self.samples_per_rank = self.total_samples // self.world_size
        if self.samples_per_rank <= 0:
            raise ValueError(
                f"{split} subset has {self.total_samples} samples, fewer than world_size={self.world_size}; "
                "increase NUM_SAMPLES_* or reduce DDP world size."
            )
        trimmed_total = self.samples_per_rank * self.world_size
        self.rank_start = self.rank * self.samples_per_rank
        self.rank_stop = self.rank_start + self.samples_per_rank
        self.dropped_tail = self.total_samples - trimmed_total
        self.maximum_depth = int(stack_shape[1])
        self.spectrum_width = int(spectra_shape[2])

    @property
    def length_dataset(self) -> int:
        """Return the number of samples visible to this rank."""
        return self.samples_per_rank

    def __len__(self) -> int:
        """Return the number of samples visible to this rank."""
        return self.samples_per_rank

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch seed used for deterministic per-shard shuffling."""
        self.epoch = int(epoch)

    def example_spectrum(self) -> torch.Tensor:
        """Load one representative spectrum without materializing the dataset."""
        data = safetensors.torch.load_file(self.paths[0], device="cpu")
        return data["spectra"][0].to(torch.float32)

    def _rank_shard_indices(self) -> list[int]:
        """Return shard indices that overlap this rank's global sample range."""
        start_shard = max(0, bisect_right(self.shard_offsets, self.rank_start) - 1)
        end_shard = max(0, bisect_right(self.shard_offsets, self.rank_stop - 1) - 1)
        shard_indices = list(range(start_shard, end_shard + 1))
        if self.shuffle and self.split == "train":
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch * 1_000_003 + self.rank * 97)
            order = torch.randperm(len(shard_indices), generator=generator).tolist()
            shard_indices = [shard_indices[i] for i in order]
        return shard_indices

    def __iter__(self):
        """Yield ``(spectrum, stack, global_index)`` samples for this rank."""
        for shard_idx in self._rank_shard_indices():
            shard_start = self.shard_offsets[shard_idx]
            shard_stop = self.shard_offsets[shard_idx + 1]
            local_start = max(self.rank_start, shard_start) - shard_start
            local_stop = min(self.rank_stop, shard_stop) - shard_start
            if local_start >= local_stop:
                continue

            data = safetensors.torch.load_file(self.paths[shard_idx], device="cpu")
            spectra = data["spectra"].to(torch.float32)
            stacks = data["thin_films"].long()

            if self.shuffle and self.split == "train":
                generator = torch.Generator()
                generator.manual_seed(self.seed + self.epoch * 1_000_003 + shard_idx * 193 + self.rank * 389)
                indices = torch.randperm(local_stop - local_start, generator=generator) + local_start
            else:
                indices = torch.arange(local_start, local_stop)

            for local_idx_t in indices:
                local_idx = int(local_idx_t.item())
                yield spectra[local_idx], stacks[local_idx], shard_start + local_idx


class RepeatedSpectrumDataset(Dataset):
    """
    Dataset that repeats a *base* ``[3, W]`` spectrum ``n_targets`` times.

    Each item returns the (possibly augmented) spectrum paired with an
    all-padding token sequence. If noise/smoothing/stochastic-filler are
    enabled in ``cfg``, every item beyond index 0 draws fresh stochastic
    augmentations so that the model sees varied inputs for the same target.

    Args
    ----
    spectrum : torch.Tensor
        Base RAT spectrum of shape ``[3, W]`` used as the template.
    n_targets : int
        Number of times the spectrum is repeated (i.e. dataset length).
    cfg : dict
        Configuration mapping providing keys such as ``ROI_MIN``,
        ``ROI_MAX``, ``FILL_OUTSIDE_ROI``, ``MAX_SEQ_LEN``,
        ``MISMATCH_FILL_ORDER``, ``WAVELENGTHS``, ``NOISE``, and
        ``SMOOTH``.
    msk_idx : int
        Token index used to fill the placeholder stack sequence.
    """

    def __init__(self, spectrum: torch.Tensor, n_targets: int, cfg: dict, msk_idx: int):
        self.spectrum = spectrum.detach().clone()  # untouched template
        self.n_targets = n_targets
        self.cfg = cfg
        self.msk_idx = msk_idx

        roi = [cfg["ROI_MIN"], cfg["ROI_MAX"]]
        self.roi = roi if cfg["FILL_OUTSIDE_ROI"]["ENABLED"] else None

    def __len__(self) -> int:
        """
        Return the number of spectrum repetitions in the dataset.

        Returns
        -------
        int
            The number of items (``n_targets``).
        """
        return self.n_targets

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return an augmented spectrum and a placeholder stack for the given index.

        Index 0 returns the base spectrum (no stochastic filling, only
        mismatch redistribution). All subsequent indices apply stochastic
        outside-ROI filling followed by noise, smoothing, and mismatch
        redistribution.

        Args
        ----
        index : int
            The index of the item to return, in ``[0, n_targets)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A 2-tuple of:

            - **spectrum** — augmented RAT spectrum of shape ``[3, W]``
              in float32.
            - **stack** — all-padding token sequence of shape
              ``[MAX_SEQ_LEN]`` as long integers.
        """
        stack = torch.full((self.cfg["MAX_SEQ_LEN"],), self.msk_idx)
        spectrum = self.spectrum.clone()
        mismatch_order = self.cfg["MISMATCH_FILL_ORDER"]

        # Zeroth index: exact original, no stochastic filling
        if index == 0 and self.cfg["FILL_OUTSIDE_ROI"]["SKIP_INDEX_0"]:
            spectrum = optollama.data.spectra.redistribute_mismatch(spectrum, mismatch_order, target_sum=1.0)

            # Ensure spectrum is on CPU for DataLoader pinning
            spectrum = spectrum.cpu() if spectrum.is_cuda else spectrum
            return spectrum, stack

        # --- vary outside ROI first (keeps ROI untouched) ---
        wavelengths = self.cfg["WAVELENGTHS"]

        spectrum = optollama.data.spectra.apply_stochastic_filler(
            spectrum,
            wavelengths,
            self.cfg["FILL_OUTSIDE_ROI"],
            seed=self.cfg["SEED"] + index,
            roi=[self.cfg["ROI_MIN"], self.cfg["ROI_MAX"]],
        )

        # --- optionally also apply noise/smoothing (if enabled) ---
        spectrum = optollama.data.spectra.apply_noise(spectrum, self.cfg["NOISE"], wavelengths)
        spectrum = optollama.data.spectra.apply_smoothing(spectrum, self.cfg["SMOOTH"])

        # --- enforce sum-to-1 ---
        spectrum = optollama.data.spectra.redistribute_mismatch(spectrum, mismatch_order, target_sum=1.0)

        # Ensure spectrum is on CPU for DataLoader pinning
        spectrum = spectrum.cpu() if spectrum.is_cuda else spectrum
        return spectrum, stack

    @classmethod
    def make_loader(
        cls,
        spectrum: torch.Tensor,
        cfg: dict,
        msk_idx: int,
    ) -> tuple[Union[SpectraDataset, Subset[SpectraDataset]], DataLoader]:
        """
        Build a :class:`RepeatedSpectrumDataset` and its :class:`~torch.utils.data.DataLoader` for inference.

        The number of repetitions is read from ``cfg["N_TARGETS"]``. If
        that value is ``<= 0`` it is clamped to ``1``. The batch size is
        taken from ``cfg["TEST_BATCH_SIZE"]``, capped at ``n_targets``.

        Args
        ----
        spectrum : torch.Tensor
            Base RAT spectrum of shape ``[3, W]`` to repeat.
        cfg : dict
            Configuration mapping providing at minimum ``N_TARGETS``,
            ``TEST_BATCH_SIZE``, ``MAX_SEQ_LEN``, ``ROI_MIN``,
            ``ROI_MAX``, ``FILL_OUTSIDE_ROI``, ``MISMATCH_FILL_ORDER``,
            ``WAVELENGTHS``, ``NOISE``, and ``SMOOTH``.
        msk_idx : int
            Token index used to fill the placeholder stack sequences.

        Returns
        -------
        RepeatedSpectrumDataset
            The constructed dataset.
        DataLoader
            DataLoader wrapping the dataset with ``shuffle=False``.
        """
        n_targets = cfg["N_TARGETS"]
        if n_targets <= 0:
            print(f"N_TARGETS in the configuration was {n_targets}, using 1 instead")
            n_targets = 1

        dataset = RepeatedSpectrumDataset(spectrum, n_targets, cfg, msk_idx)

        loader = DataLoader(
            dataset, batch_size=min(n_targets, cfg["TEST_BATCH_SIZE"]), shuffle=False, pin_memory=not torch.mps.is_available()
        )

        return dataset, loader
