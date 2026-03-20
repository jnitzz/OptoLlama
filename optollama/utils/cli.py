import argparse
import os

import omegaconf
import torch


def parse_arguments() -> argparse.Namespace:
    """
    Parse namespace arguments.

    Returns
    -------
        The CLI arguments as a Namespace.
    """
    p = argparse.ArgumentParser()
    
    p.add_argument(
        "--config",
        type=str,
        default="configs/optollama.yaml",
        help=("Path to YAML config file (.yaml/.yml), e.g. configs/config_optollama.yaml"),
    )
    
    return p.parse_args()


def load_config_file(path: str) -> dict:
    """
    Load config from a YAML file.

    Args
    ----
        path: Path to the config file.

    Returns
    -------
        The configuration as a dictionary.
    """
    cfg = omegaconf.OmegaConf.load(path)
    
    return omegaconf.OmegaConf.to_container(cfg, resolve=True)


def load_config(args: argparse.Namespace) -> dict:
    """
    Load config from YAML and enrich wavelengths.

    Args
    ----
        path: Path to the config file.

    Returns
    -------
        The configuration as a dictionary.
    """
    cfg = load_config_file(args.config)

    # --- build WAVELENGTHS if needed ---
    wl_min = int(cfg["WAVELENGTH_MIN"])
    wl_max = int(cfg["WAVELENGTH_MAX"])
    wl_step = int(cfg["WAVELENGTH_STEPS"])
    
    cfg["WAVELENGTHS"] = torch.arange(wl_min, wl_max + 1, wl_step, dtype=torch.int)

    return cfg