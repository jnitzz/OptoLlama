import re
from typing import Any, Optional, Self, Union

import safetensors.torch
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset

import optollama.data.spectra

# ruff: noqa: E731


class SpectraDataset(torch.utils.data.Dataset):
    """
    Dataset for Hugging Face `.safetensors` shards.

    Each file must contain:
      - 'spectra'    : float tensor of shape [N, 3, W]
      - 'thin_films' : long tensor of shape [N, S]
    """

    def __init__(self, paths: list[str] | str):
        super().__init__()

        # normalize paths (accept dir or list of files)
        if isinstance(paths, str):
            paths = [paths]

        paths = sorted(paths, key=SpectraDataset.shard_sort_key)

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
        batch_size = cfg["TRAIN_BATCH_SIZE"] if split == "train" else cfg["TEST_BATCH_SIZE"]
        num_workers = cfg["NUM_WORKERS"]

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
