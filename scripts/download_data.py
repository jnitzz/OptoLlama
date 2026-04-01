#!/usr/bin/env python

import argparse

from huggingface_hub import login, snapshot_download


def setup_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the dataset download script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments containing:
        - ``token`` (str): HuggingFace API token.
        - ``dest`` (str): Local directory to save the downloaded files.
    """
    parser = argparse.ArgumentParser(description="Download a private Hugging Face dataset.")

    # huggingface API token
    parser.add_argument(
        "--token", 
        type=str, 
        required=True, 
        help="HuggingFace API token"
    )

    # destination (optional, defaults to local dir)
    parser.add_argument(
        "--dest", 
        type=str, 
        default=".", 
        help="Local directory to save files (default: current directory)"
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """
    Authenticate with HuggingFace and download the MultiLayerThinFilms dataset.

    Args
    ----
    args : argparse.Namespace
        Parsed CLI arguments. Expected attributes:

        - ``token`` (str): HuggingFace API token used for authentication.
        - ``dest`` (str): Local directory where the dataset will be saved.
    """
    # authenticate
    login(token=args.token)

    print("Starting targeted download...")
    repo_id = "HZBSolarOptics/MultiLayerThinFilms"

    try:
        snapshot_download(
            repo_id=repo_id, 
            repo_type="dataset", 
            local_dir=args.dest, 
            ignore_patterns=["README.md"]
        )
        print("All data download")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    args = setup_args()
    main(args)
