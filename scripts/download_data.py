#!/usr/bin/env python

import argparse
import os

from huggingface_hub import login, snapshot_download


def setup_args():
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


def main(args: argparse.Namespace):
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
