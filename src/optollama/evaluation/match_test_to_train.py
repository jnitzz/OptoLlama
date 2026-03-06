import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import tqdm
from optollama.dataloader.dataset import SpectraDataset
from optollama.evaluation.metrics import masked_mae_roi
from safetensors.torch import load_file
from optollama.utils.utils import ensure_3w, save_as_json


def load_train_sample_by_global_id(
    global_id: int,
    train_paths: Union[List[str], str],
) -> Tuple[torch.Tensor, torch.Tensor, str, int]:
    """
    Load a single training sample by its GLOBAL index across all shards.

    Parameters
    ----------
    global_id : int
        Global position of the sample in the sorted training set.
    train_paths : list[str] | str
        Directory/directories or .safetensors files containing the train shards.

    Returns
    -------
    sample : torch.Tensor [3, W]
    sample_thin_films : torch.Tensor  # [S] token ids / stack
    file_path : str                 # shard file it comes from
    local_idx : int                 # index inside the shard
    """
    # --- normalize to list ---
    if isinstance(train_paths, str):
        train_paths = [train_paths]

    # --- collect safetensors files ---
    files: list[Path] = []
    for p in map(Path, train_paths):
        if p.is_dir():
            files.extend(sorted(fp for fp in p.glob("*.safetensors")))
        elif p.suffix == ".safetensors":
            files.append(p)
        else:
            raise ValueError(f"Invalid train path: {p}")

    if not files:
        raise FileNotFoundError("No .safetensors files found")

    # --- same sort logic as dataset ---
    def shard_sort_key(path: Path) -> Tuple[str, Union[int, float]]:
        import re

        m = re.match(r"^(.*?)(\d+)$", path.stem.lower())
        if m:
            prefix, num = m.groups()
            return (prefix, int(num))
        return (path.stem.lower(), float("inf"))

    files = sorted(files, key=shard_sort_key)

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
    target_spec: torch.Tensor,  # [3,W] or [1,3,W] or [W,3]
    train_paths: Union[List[str], str],  # dir(s) or .safetensors file(s)
    *,
    train_chunk_size: int = 1024,
    device: Optional[Union[str, torch.device]] = None,
    wl_range: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Find closest training sample for a given target via masked MAE.

    Given a single target spectrum [3,W], stream over the training shards
    (one .safetensors file at a time) and find the closest training sample
    by masked MAE.

    It never holds the whole train set in memory at once.

    Returns
    -------
    {
        "best_mae": float,
        "best_global_index": int,       # index in concatenated train set
        "best_file": str,               # path to safetensors file
        "best_index_in_file": int,      # index inside that file
    }
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

    # --- collect all safetensors files, but don't load them yet ---
    if isinstance(train_paths, str):
        train_paths = [train_paths]

    files: list[Path] = []
    for p in map(Path, train_paths):
        if p.is_dir():
            files.extend(sorted(fp for fp in p.glob("*.safetensors")))
        elif p.suffix == ".safetensors":
            files.append(p)
        else:
            raise ValueError(f"Unsupported train path (expect dir or .safetensors): {p}")

    if not files:
        raise FileNotFoundError("No .safetensors files found in the provided train paths.")

    # mimic the same sort logic as SpectraDataset for consistent indices :contentReference[oaicite:6]{index=6}
    def shard_sort_key(path: Path) -> Tuple[str, Union[int, float]]:
        import re

        m = re.match(r"^(.*?)(\d+)$", path.stem.lower())
        if m:
            prefix, num = m.groups()
            return (prefix, int(num))
        return (path.stem.lower(), float("inf"))

    files = sorted(files, key=shard_sort_key)

    best_mae_val = float("inf")
    best_global_idx = -1
    best_file: Optional[str] = None
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
    *,
    train_chunk_size: int = 1024,
    device: Optional[Union[str, torch.device]] = None,
) -> List[Dict[str, Any]]:
    """
    For each test spectrum, find the closest training spectrum by MAE.

    Distance metric:
        - Mean Absolute Error on the spectra (R/A/T over wavelength).
        - Uses `metrics.masked_mae_roi` for robustness.

    Parameters
    ----------
    train_ds : SpectraDataset
        Dataset containing training spectra (train_ds.spectra: [N_train, 3, W]).
    test_ds : SpectraDataset
        Dataset containing test spectra (test_ds.spectra: [N_test, 3, W]).
    train_chunk_size : int, optional
        Number of training samples to process at once on the device,
        to avoid running out of memory. Default: 1024.
    device : str or torch.device, optional
        Device used for computation. If None, chooses CUDA if available,
        else CPU.

    Returns
    -------
    List[Dict[str, Any]]
        One dict per test sample with keys:
            - "test_index": index in the test dataset
            - "best_train_index": index in the training dataset
            - "mae": minimal MAE value
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    train_spectra = train_ds.spectra  # [N_train, 3, W]
    test_spectra = test_ds.spectra  # [N_test, 3, W]

    n_train = train_spectra.size(0)
    n_test = test_spectra.size(0)

    results: List[Dict[str, Any]] = []

    # Loop over test points
    for test_idx in tqdm.tqdm(range(n_test), desc="Matching test → train"):
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


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    CLI arguments:

        --train PATH ...   one or more folders/files with training .safetensors
        --test PATH ...    one or more folders/files with test .safetensors
        --out_dir PATH     directory where JSON will be written
        --name NAME        base filename (default: nearest_neighbors)
        --chunk_size N     training chunk size for distance computation
        --device DEV       e.g. "cuda", "cuda:0", or "cpu"
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
        "--device",
        type=str,
        default=None,
        help='Device to use, e.g. "cuda", "cuda:0", or "cpu". Default: auto-detect CUDA if available.',
    )
    return p.parse_args()


def main() -> None:
    """Run main function."""
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

    train_ds = SpectraDataset(train_paths)
    test_ds = SpectraDataset(test_paths)

    results = find_nearest_neighbors(
        train_ds,
        test_ds,
        train_chunk_size=args.chunk_size,
        device=args.device,
    )

    # Save mapping as JSON using the project helper
    save_as_json(args.out_dir, results, name=args.name)
    print(f"Saved {len(results)} test→train mappings to '{args.out_dir}/{args.name}.json'")


if __name__ == "__main__":
    if "--train" not in sys.argv:
        sys.argv.extend(["--train", "D:/Profile/a3536/Eigene Dateien/Github/OptoLlama/data/TF_safetensors/dtrain"])
    if "--test" not in sys.argv:
        sys.argv.extend(["--test", "D:/Profile/a3536/Eigene Dateien/Github/OptoLlama/data/TF_safetensors/dtest"])
    if "--out_dir" not in sys.argv:
        sys.argv.extend(["--out_dir", "D:/Profile/a3536/Eigene Dateien/Github/OptoLlama/data/TF_safetensors"])
    if "--device" not in sys.argv:
        sys.argv.extend(["--device", "cuda"])
    if "--name" not in sys.argv:
        sys.argv.extend(["--name", "test_to_train_nn"])
    if "--chunk_size" not in sys.argv:
        sys.argv.extend(["--chunk_size", "10240"])
    main()

# python match_test_to_train.py \
#   --train /path/to/train_safetensors_dir \
#   --test /path/to/test_safetensors_dir \
#   --out-dir /path/to/output \
#   --name test_to_train_nn \
#   --chunk-size 2048 \
#   --device cuda
