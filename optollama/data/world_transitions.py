from pathlib import Path
from typing import Union

import safetensors.torch
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset


WORLD_TRANSITION_KEYS = (
    "target_spectra",
    "current_stacks",
    "current_spectra",
    "next_stacks",
    "next_spectra",
    "cost_before",
    "cost_after",
)


def expand_world_transition_paths(paths: list[str] | str) -> list[str]:
    """
    Expand a file, directory, or list of paths into sorted safetensor files.
    """
    if isinstance(paths, str):
        paths = [paths]

    expanded: list[str] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            expanded.extend(str(fp) for fp in sorted(path.glob("*.safetensors")))
        else:
            expanded.append(str(path))

    out = sorted(expanded)
    if not out:
        raise FileNotFoundError("No .safetensors files found for world-transition data.")
    return out


class WorldTransitionDataset(Dataset):
    """
    Dataset of local design transitions for learned world-model scoring.

    Each row describes a transition from a current stack to a candidate next
    stack for the same target spectrum. The learned scorer uses this data to
    predict whether the candidate edit improves the optical objective.
    """

    def __init__(self, paths: list[str] | str) -> None:
        super().__init__()
        self.paths = expand_world_transition_paths(paths)

        buckets: dict[str, list[torch.Tensor]] = {key: [] for key in WORLD_TRANSITION_KEYS}
        for path in self.paths:
            data = safetensors.torch.load_file(path, device="cpu")
            missing = [key for key in WORLD_TRANSITION_KEYS if key not in data]
            if missing:
                raise KeyError(f"{path} is missing world-transition keys: {missing}")
            for key in WORLD_TRANSITION_KEYS:
                buckets[key].append(data[key])

        self.data = {key: torch.cat(values, dim=0) for key, values in buckets.items()}
        self.length_dataset = int(self.data["target_spectra"].size(0))
        for key, value in self.data.items():
            if value.size(0) != self.length_dataset:
                raise RuntimeError(f"World-transition key {key!r} has mismatched row count.")

    def __len__(self) -> int:
        """
        Return the number of transition rows.
        """
        return self.length_dataset

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """
        Return one transition row as a tensor dictionary.
        """
        return {key: value[index] for key, value in self.data.items()}

    @classmethod
    def make_loader(
        cls,
        cfg: dict,
        split: str,
        subset_n: int | None = None,
        ddp: bool = False,
    ) -> tuple[Union["WorldTransitionDataset", Subset["WorldTransitionDataset"]], DataLoader, DistributedSampler | None]:
        """
        Build a world-transition dataset and dataloader from config paths.
        """
        split_lower = split.lower()
        if split_lower not in ("train", "test"):
            raise ValueError(f"Unknown data split {split_lower!r}, expected 'train' or 'test'.")

        prefix = "WORLD_DATA_PATH_TRAIN" if split_lower == "train" else "WORLD_DATA_PATH_TEST"
        paths = sorted([cfg[key] for key in cfg if key.startswith(prefix)])
        if not paths:
            raise KeyError(f"No config entries found with prefix {prefix!r}.")

        dataset: WorldTransitionDataset | Subset[WorldTransitionDataset] = cls(paths)
        if subset_n is not None and subset_n < len(dataset):
            idxs = torch.linspace(0, len(dataset) - 1, int(subset_n), dtype=torch.long).unique()
            dataset = Subset(dataset, idxs)

        if split_lower == "train":
            sampler = DistributedSampler(dataset, shuffle=True) if ddp else None
            shuffle = not ddp
            drop_last = True
            batch_size = int(cfg["WORLD_TRAIN_BATCH_SIZE"])
        else:
            sampler = DistributedSampler(dataset, shuffle=False) if ddp else None
            shuffle = False
            drop_last = False
            batch_size = int(cfg["WORLD_TEST_BATCH_SIZE"])

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=int(cfg.get("NUM_WORKERS", 0)),
            pin_memory=not torch.mps.is_available(),
            drop_last=drop_last,
        )
        return dataset, loader, sampler
