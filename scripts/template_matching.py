#!/usr/bin/env python

import argparse
import os
from pathlib import Path
from typing import Any, Optional, Union

import torch
import tqdm
from safetensors import safe_open
from optollama.data.dataset import SpectraDataset
from optollama.evaluation.metrics import masked_mae_roi
from optollama.data.spectra import ensure_3w
from optollama.utils.utils import save_as_json
from safetensors.torch import load_file


def collect_safetensor_files(paths: Union[list[str], str]) -> list[Path]:
    """Collect safetensor shards from directories/files using dataset-compatible sorting."""
    if isinstance(paths, str):
        paths = [paths]

    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files.extend(fp for fp in p.glob("*.safetensors"))
        elif p.suffix == ".safetensors":
            files.append(p)
        else:
            raise ValueError(f"Unsupported path (expect dir or .safetensors): {p}")

    if not files:
        raise FileNotFoundError("No .safetensors files found in the provided paths.")

    return sorted(files, key=shard_sort_key)


def shard_sort_key(path: Path) -> tuple[str, Union[int, float]]:
    """Sort shards by prefix and numeric suffix."""
    import re

    m = re.match(r"^(.*?)(\d+)$", path.stem.lower())
    if m:
        prefix, num = m.groups()
        return (prefix, int(num))
    return (path.stem.lower(), float("inf"))


def safetensor_first_dim(path: Path, tensor_name: str = "spectra") -> int:
    """Read the first dimension of a safetensors tensor without materializing all shards."""
    with safe_open(str(path), framework="pt", device="cpu") as f:
        if tensor_name not in f.keys():
            raise KeyError(f"{path} must contain {tensor_name!r} tensor.")
        return int(f.get_slice(tensor_name).get_shape()[0])


def load_train_sample_by_global_id(
    global_id: int,
    train_paths: Union[list[str], str],
) -> tuple[torch.Tensor, torch.Tensor, str, int]:
    """
    Load a single training sample by its global index across all shards.

    Args
    ----
    global_id : int
        Global position of the sample in the sorted training set.
    train_paths : list[str] or str
        Directory/directories or ``.safetensors`` files containing the
        training shards.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, str, int]
        A 4-tuple of ``(spectrum, thin_films, file_path, local_idx)`` where:

        - ``spectrum`` has shape ``[3, W]``.
        - ``thin_films`` contains token ids for the stack.
        - ``file_path`` is the path to the shard file.
        - ``local_idx`` is the index inside that shard.

    Raises
    ------
    FileNotFoundError
        If no ``.safetensors`` files are found.
    IndexError
        If ``global_id`` exceeds the total dataset size.
    """
    files = collect_safetensor_files(train_paths)

    # --- iterate until we find the correct shard ---
    offset = 0
    for shard_path in files:
        data = load_file(str(shard_path))
        spectra = data["spectra"]  # [N,3,W]
        thin_films = data["thin_films"]
        n = spectra.size(0)

        # check if global_id is inside this shard
        if global_id < offset + n:
            local_idx = global_id - offset
            sample_spectrum = spectra[local_idx]  # [3,W]
            sample_thin_films = thin_films[local_idx]
            return sample_spectrum, sample_thin_films, str(shard_path), int(local_idx)

        offset += n

    raise IndexError(f"Global id {global_id} exceeds dataset size {offset}.")


@torch.no_grad()
def find_best_train_for_target(
    target_spec: torch.Tensor,
    train_paths: Union[list[str], str],
    train_chunk_size: int = 1024,
    device: Optional[Union[str, torch.device]] = None,
    wl_range: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    """
    Find the closest training sample to a target spectrum via masked MAE.

    Streams over training shards one ``.safetensors`` file at a time so the
    full training set is never held in memory simultaneously.

    Args
    ----
    target_spec : torch.Tensor
        Target spectrum of shape ``[3, W]``, ``[1, 3, W]``, or ``[W, 3]``.
    train_paths : list[str] or str
        Directory/directories or ``.safetensors`` file(s) containing the
        training shards.
    train_chunk_size : int
        Number of training samples to process per device batch (default:
        ``1024``).
    device : str or torch.device, optional
        Computation device. Defaults to CUDA if available, else CPU.
    wl_range : torch.Tensor, optional
        Boolean wavelength mask of shape ``[W]`` restricting the MAE
        computation to a region of interest.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:

        - ``"best_mae"`` (float): minimum MAE found.
        - ``"best_global_index"`` (int): index in the concatenated train set.
        - ``"best_file"`` (str): path to the shard file.
        - ``"best_index_in_file"`` (int): index inside that shard.

    Raises
    ------
    FileNotFoundError
        If no ``.safetensors`` files are found.
    RuntimeError
        If no training spectra were processed.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    # --- normalize target to [3,W] on device ---
    t = target_spec
    if t.dim() == 3 and t.size(0) == 1:
        t = t[0]  # [3,W] or [W,3]
    if t.dim() != 2:
        raise ValueError(f"target_spec must be [3,W] or [1,3,W]/[W,3], got {tuple(t.shape)}")

    t_3w, transposed = ensure_3w(t)  # guarantees [...,3,W]  :contentReference[oaicite:5]{index=5}
    if t_3w.dim() == 3:
        t_3w = t_3w[0]
    target_3w = t_3w.to(torch.float32).to(device_t)  # [3,W]
    w = target_3w.size(-1)

    files = collect_safetensor_files(train_paths)

    best_mae_val = float("inf")
    best_global_idx = -1
    best_file = None
    best_idx_in_file = -1

    global_offset = 0  # how many samples we've seen so far

    # --- iterate over shards, one at a time ---
    for shard_path in files:
        data = load_file(str(shard_path))  # only this file in RAM now
        if "spectra" not in data:
            raise KeyError(f"{shard_path} must contain 'spectra' tensor.")
        train_spectra = data["spectra"].to(torch.float32)  # [N_i, 3, W_i]

        if train_spectra.dim() != 3 or train_spectra.size(1) != 3:
            raise ValueError(f"'spectra' in {shard_path} must be [N,3,W], got {tuple(train_spectra.shape)}")
        if train_spectra.size(2) != w:
            raise ValueError(f"Wavelength dimension mismatch in {shard_path}: train W={train_spectra.size(2)}, target W={w}")

        n_train = train_spectra.size(0)

        # loop over this shard in chunks → send chunk to device
        for start in range(0, n_train, train_chunk_size):
            end = min(start + train_chunk_size, n_train)
            # print(end)
            chunk_cpu = train_spectra[start:end]  # [B,3,W] on CPU
            chunk = chunk_cpu.to(device_t, non_blocking=True)  # [B,3,W] on device

            # repeat target to match chunk size
            t_rep = target_3w.unsqueeze(0).expand(chunk.size(0), -1, -1)  # [B,3,W]

            mae_vals = masked_mae_roi(t_rep, chunk, wl_range)  # [B]  :contentReference[oaicite:7]{index=7}

            chunk_min_val, chunk_min_idx = torch.min(mae_vals, dim=0)
            chunk_min_val_f = float(chunk_min_val.item())

            if chunk_min_val_f < best_mae_val:
                best_mae_val = chunk_min_val_f
                best_global_idx = global_offset + start + int(chunk_min_idx.item())
                best_file = str(shard_path)
                best_idx_in_file = start + int(chunk_min_idx.item())

        global_offset += n_train

    if best_file is None:
        raise RuntimeError("No training spectra processed; check your train_paths.")

    return {
        "best_mae": best_mae_val,
        "best_global_index": int(best_global_idx),
        "best_file": best_file,
        "best_index_in_file": int(best_idx_in_file),
    }


@torch.no_grad()
def find_nearest_neighbors(
    train_ds: SpectraDataset,
    test_ds: SpectraDataset,
    train_chunk_size: int = 1024,
    device: Optional[Union[str, torch.device]] = None,
) -> list[dict[str, Any]]:
    """
    For each test spectrum, find the closest training spectrum by MAE.

    Uses :func:`~optollama.evaluation.metrics.masked_mae_roi` as the
    distance metric (mean absolute error over R/A/T channels).

    Args
    ----
    train_ds : SpectraDataset
        Dataset containing training spectra of shape ``[N_train, 3, W]``.
    test_ds : SpectraDataset
        Dataset containing test spectra of shape ``[N_test, 3, W]``.
    train_chunk_size : int
        Number of training samples to process per device batch to avoid
        out-of-memory errors (default: ``1024``).
    device : str or torch.device, optional
        Computation device. Defaults to CUDA if available, else CPU.

    Returns
    -------
    list[dict[str, Any]]
        One dict per test sample with keys:

        - ``"test_index"`` (int): index in the test dataset.
        - ``"best_train_index"`` (int): index in the training dataset.
        - ``"mae"`` (float): minimum MAE value.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    train_spectra = train_ds.spectra  # [N_train, 3, W]
    test_spectra = test_ds.spectra  # [N_test, 3, W]

    n_train = train_spectra.size(0)
    n_test = test_spectra.size(0)

    results = []

    # Loop over test points
    for test_idx in tqdm.tqdm(range(n_test), desc="Matching test -> train"):
        print(f"Processing test index {test_idx + 1}/{n_test}", end="\r")
        spec_test = test_spectra[test_idx].unsqueeze(0).to(device_t)  # [1, 3, W]

        best_mae_val = float("inf")
        best_train_idx = -1

        # Loop over train points in chunks to save memory
        for start in range(0, n_train, train_chunk_size):
            end = min(start + train_chunk_size, n_train)
            print(end)
            train_chunk = train_spectra[start:end].to(device_t)  # [B, 3, W]
            # Repeat test spectrum B times to match chunk size
            spec_test_rep = spec_test.expand(train_chunk.size(0), -1, -1)  # [B, 3, W]

            # masked_mae_roi returns [B] per-sample MAE
            mae_vals = masked_mae_roi(spec_test_rep, train_chunk)  # [B]

            # Find best in this chunk
            chunk_min_val, chunk_min_idx = torch.min(mae_vals, dim=0)
            chunk_min_val_f = float(chunk_min_val.item())

            if chunk_min_val_f < best_mae_val:
                best_mae_val = chunk_min_val_f
                best_train_idx = int(start + chunk_min_idx.item())

        results.append(
            {
                "test_index": int(test_idx),
                "best_train_index": best_train_idx,
                "mae": best_mae_val,
            }
        )

    return results


def pairwise_masked_mae(
    test_spectra: torch.Tensor,
    train_spectra: torch.Tensor,
    wl_range: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute pairwise MAE between test and train spectra as [n_test, n_train]."""
    if test_spectra.dim() != 3 or train_spectra.dim() != 3:
        raise ValueError(
            f"Expected test/train spectra with shape [N,3,W], got {tuple(test_spectra.shape)} and {tuple(train_spectra.shape)}"
        )
    if test_spectra.size(1) != 3 or train_spectra.size(1) != 3:
        raise ValueError(
            f"Expected channel dimension 3, got {test_spectra.size(1)} and {train_spectra.size(1)}"
        )
    if test_spectra.size(2) != train_spectra.size(2):
        raise ValueError(
            f"Wavelength dimension mismatch: test W={test_spectra.size(2)}, train W={train_spectra.size(2)}"
        )

    finite = torch.isfinite(train_spectra).all(dim=-1, keepdim=True).unsqueeze(0)
    valid = finite.expand(test_spectra.size(0), -1, -1, train_spectra.size(-1))
    if wl_range is not None:
        valid = valid & wl_range.to(device=train_spectra.device, dtype=torch.bool).view(1, 1, 1, -1)

    abs_err = torch.abs(test_spectra.unsqueeze(1) - torch.nan_to_num(train_spectra).unsqueeze(0))
    masked_err = abs_err.where(valid, torch.zeros_like(abs_err))
    num = masked_err.sum(dim=(2, 3))
    den = valid.sum(dim=(2, 3)).clamp_min(1)
    return num / den


@torch.no_grad()
def find_nearest_neighbors_streaming(
    train_paths: Union[list[str], str],
    test_paths: Union[list[str], str],
    train_chunk_size: int = 1024,
    test_chunk_size: int = 16,
    max_test_samples: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    inner_progress: bool = True,
) -> list[dict[str, Any]]:
    """
    Find nearest training spectra for test spectra without loading full datasets.

    This scans training shards once per test chunk instead of once per test
    sample, which is much more practical for multi-million-sample datasets.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    train_files = collect_safetensor_files(train_paths)
    test_files = collect_safetensor_files(test_paths)
    total_train = sum(safetensor_first_dim(path) for path in train_files)
    total_test_all = sum(safetensor_first_dim(path) for path in test_files)
    total_test = min(total_test_all, max_test_samples) if max_test_samples is not None else total_test_all
    print(
        f"Exact NN search: {total_test} test spectra x {total_train} train spectra "
        f"({len(test_files)} test shards, {len(train_files)} train shards)"
    )

    results: list[dict[str, Any]] = []
    test_global_offset = 0
    outer = tqdm.tqdm(total=total_test, desc="Matching test -> train", unit="sample")

    try:
        for test_path in test_files:
            test_data = load_file(str(test_path))
            if "spectra" not in test_data:
                raise KeyError(f"{test_path} must contain 'spectra' tensor.")
            test_spectra_all = test_data["spectra"].to(torch.float32)
            if test_spectra_all.dim() != 3 or test_spectra_all.size(1) != 3:
                raise ValueError(f"'spectra' in {test_path} must be [N,3,W], got {tuple(test_spectra_all.shape)}")

            n_test_in_shard = test_spectra_all.size(0)
            if max_test_samples is not None:
                remaining = max_test_samples - len(results)
                if remaining <= 0:
                    break
                n_test_in_shard = min(n_test_in_shard, remaining)

            for test_start in range(0, n_test_in_shard, test_chunk_size):
                test_end = min(test_start + test_chunk_size, n_test_in_shard)
                test_batch = test_spectra_all[test_start:test_end].to(device_t, non_blocking=True)
                n_test_batch = test_batch.size(0)
                w = test_batch.size(-1)

                best_mae = torch.full((n_test_batch,), float("inf"), device=device_t)
                best_global = torch.full((n_test_batch,), -1, dtype=torch.long, device=device_t)
                best_local = [-1 for _ in range(n_test_batch)]
                best_file = ["" for _ in range(n_test_batch)]

                train_global_offset = 0
                train_iter = train_files
                if inner_progress:
                    train_iter = tqdm.tqdm(
                        train_files,
                        desc=f"scan train for test {test_global_offset + test_start}-{test_global_offset + test_end - 1}",
                        leave=False,
                        unit="shard",
                    )

                for train_path in train_iter:
                    train_data = load_file(str(train_path))
                    if "spectra" not in train_data:
                        raise KeyError(f"{train_path} must contain 'spectra' tensor.")
                    train_spectra = train_data["spectra"].to(torch.float32)
                    if train_spectra.dim() != 3 or train_spectra.size(1) != 3:
                        raise ValueError(f"'spectra' in {train_path} must be [N,3,W], got {tuple(train_spectra.shape)}")
                    if train_spectra.size(2) != w:
                        raise ValueError(
                            f"Wavelength dimension mismatch in {train_path}: train W={train_spectra.size(2)}, test W={w}"
                        )

                    n_train = train_spectra.size(0)
                    for train_start in range(0, n_train, train_chunk_size):
                        train_end = min(train_start + train_chunk_size, n_train)
                        train_chunk = train_spectra[train_start:train_end].to(device_t, non_blocking=True)
                        mae = pairwise_masked_mae(test_batch, train_chunk)
                        chunk_best, chunk_idx = mae.min(dim=1)
                        improved = chunk_best < best_mae
                        if improved.any():
                            best_mae[improved] = chunk_best[improved]
                            best_global[improved] = train_global_offset + train_start + chunk_idx[improved]
                            improved_indices = torch.nonzero(improved, as_tuple=False).flatten().detach().cpu().tolist()
                            chunk_idx_cpu = chunk_idx.detach().cpu().tolist()
                            for i in improved_indices:
                                best_local[i] = train_start + int(chunk_idx_cpu[i])
                                best_file[i] = str(train_path)

                    train_global_offset += n_train

                best_mae_cpu = best_mae.detach().cpu().tolist()
                best_global_cpu = best_global.detach().cpu().tolist()
                for i in range(n_test_batch):
                    results.append(
                        {
                            "test_index": int(test_global_offset + test_start + i),
                            "test_file": str(test_path),
                            "test_index_in_file": int(test_start + i),
                            "best_train_index": int(best_global_cpu[i]),
                            "best_global_index": int(best_global_cpu[i]),
                            "best_file": best_file[i],
                            "best_index_in_file": int(best_local[i]),
                            "mae": float(best_mae_cpu[i]),
                        }
                    )

                outer.update(n_test_batch)

            test_global_offset += test_spectra_all.size(0)
            if max_test_samples is not None and len(results) >= max_test_samples:
                break
    finally:
        outer.close()

    return results


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for the match-test-to-train script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes:

        - ``train`` — one or more folders/files with training
          ``.safetensors``.
        - ``test`` — one or more folders/files with test ``.safetensors``.
        - ``out_dir`` — directory where the JSON result will be written.
        - ``name`` — base filename (default: ``"nearest_neighbors"``).
        - ``chunk_size`` — training chunk size for distance computation.
        - ``test_chunk_size`` — number of test spectra matched together.
        - ``max_test_samples`` — optional cap for exact matching.
        - ``device`` — e.g. ``"cuda"``, ``"cuda:0"``, or ``"cpu"``.
    """
    p = argparse.ArgumentParser(description="Match test spectra to nearest training spectra (by MAE).")
    p.add_argument(
        "--train",
        nargs="+",
        required=True,
        help="Training data path(s): dir(s) or .safetensors file(s).",
    )
    p.add_argument(
        "--test",
        nargs="+",
        required=True,
        help="Test data path(s): dir(s) or .safetensors file(s).",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory where the JSON result will be saved.",
    )
    p.add_argument(
        "--name",
        type=str,
        default="nearest_neighbors",
        help="Base name for the JSON file (default: nearest_neighbors).",
    )
    p.add_argument(
        "--chunk_size",
        type=int,
        default=1024,
        help="Chunk size for iterating over training samples (default: 1024).",
    )
    p.add_argument(
        "--test_chunk_size",
        type=int,
        default=16,
        help="Number of test spectra matched against each training chunk (default: 16).",
    )
    p.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help="Optional cap on the number of test spectra to match, taken from the start of the sorted test shards.",
    )
    p.add_argument(
        "--no_inner_progress",
        action="store_true",
        help="Disable the per-test-chunk train-shard progress bar.",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device to use, e.g. "cuda", "cuda:0", or "cpu". Default: auto-detect CUDA if available.',
    )
    return p.parse_args()


def main() -> None:
    """
    Entry point for the match-test-to-train CLI script.

    Loads training and test datasets, computes nearest-neighbour matches by
    MAE, and saves the result as a JSON file.
    """
    args = parse_args()

    # SpectraDataset already knows how to handle:
    #   - directories containing .safetensors
    #   - individual .safetensors files
    #
    # It will concatenate all shards it finds.
    if len(args.train) == 1:
        train_paths = args.train[0]
    else:
        train_paths = args.train

    if len(args.test) == 1:
        test_paths = args.test[0]
    else:
        test_paths = args.test

    results = find_nearest_neighbors_streaming(
        train_paths,
        test_paths,
        train_chunk_size=args.chunk_size,
        test_chunk_size=args.test_chunk_size,
        max_test_samples=args.max_test_samples,
        device=args.device,
        inner_progress=not args.no_inner_progress,
    )

    # Save mapping as JSON using the project helper
    out_path = os.path.join(args.out_dir, args.name + ".json")
    os.makedirs(args.out_dir, exist_ok=True)
    save_as_json(out_path, results)
    print(f"Saved {len(results)} test->train mappings to '{out_path}'")


if __name__ == "__main__":
    main()

# python match_test_to_train.py \
#   --train /path/to/train_safetensors_dir \
#   --test /path/to/test_safetensors_dir \
#   --out-dir /path/to/output \
#   --name test_to_train_nn \
#   --chunk-size 2048 \
#   --device cuda
