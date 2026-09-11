"""Generate eight hand-picked four-layer thin-film examples with OptoLlama's TMM."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import safetensors.torch
import torch

from optollama.data.token import EOS_TOKEN, MSK_TOKEN, PAD_TOKEN
from optollama.evaluation.simulation import build_tmm


SAMPLES: tuple[dict[str, Any], ...] = (
    {
        "name": "bragg_visible",
        "family": "quarter_wave_bragg",
        "layers": ["TiO2_60", "SiO2_90", "TiO2_60", "SiO2_90"],
        "description": "Two high/low-index quarter-wave pairs, approximately centered in the visible.",
    },
    {
        "name": "bragg_near_ir",
        "family": "quarter_wave_bragg",
        "layers": ["TiO2_110", "SiO2_170", "TiO2_110", "SiO2_170"],
        "description": "The same Bragg motif scaled to a near-infrared stopband.",
    },
    {
        "name": "chirped_dielectric_mild",
        "family": "chirped_dielectric",
        "layers": ["TiO2_50", "SiO2_80", "TiO2_80", "SiO2_130"],
        "description": "A mildly chirped high/low-index stack with two optical-thickness scales.",
    },
    {
        "name": "chirped_dielectric_strong",
        "family": "chirped_dielectric",
        "layers": ["TiO2_40", "SiO2_60", "TiO2_120", "SiO2_200"],
        "description": "A strongly chirped version that separates the two interference scales and broadens the structure.",
    },
    {
        "name": "silver_cavity_visible",
        "family": "metal_dielectric_cavity",
        "layers": ["Ag_20", "SiO2_140", "Ag_20", "MgF2_80"],
        "description": "A thin-metal Fabry-Perot cavity with a visible-scale dielectric spacer.",
    },
    {
        "name": "silver_cavity_near_ir",
        "family": "metal_dielectric_cavity",
        "layers": ["Ag_20", "SiO2_300", "Ag_20", "MgF2_140"],
        "description": "A thicker-spacer version of the silver cavity, shifting its resonances toward the near infrared.",
    },
    {
        "name": "metal_backed_absorber_visible",
        "family": "metal_backed_absorber",
        "layers": ["TiN_20", "SiO2_100", "TiO2_60", "Ag_100"],
        "description": "A lossy matching layer and dielectric spacer in front of an optically thick silver reflector.",
    },
    {
        "name": "metal_backed_absorber_near_ir",
        "family": "metal_backed_absorber",
        "layers": ["TiN_20", "SiO2_250", "TiO2_110", "Ag_100"],
        "description": "A thicker metal-backed absorber intended to move interference-assisted absorption into the near infrared.",
    },
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/examples/four_layer_interesting"))
    parser.add_argument("--tokens", type=Path, default=Path("data/tokens.json"))
    parser.add_argument("--materials", type=Path, default=Path("data/materials"))
    parser.add_argument("--wavelength-min", type=float, default=400.0, help="First wavelength in nm.")
    parser.add_argument("--wavelength-max", type=float, default=1600.0, help="Last wavelength in nm.")
    parser.add_argument("--wavelength-step", type=float, default=10.0, help="Wavelength spacing in nm.")
    parser.add_argument("--incidence-angle", type=float, default=0.0, help="Incidence angle in degrees.")
    parser.add_argument("--polarization", choices=("s", "p"), default="s")
    parser.add_argument("--sequence-length", type=int, default=21, help="Encoded length, including EOS and PAD tokens.")
    parser.add_argument("--device", default="auto", help="Torch device, or 'auto'.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_vocabulary(path: Path) -> tuple[list[str], dict[str, int]]:
    """Load and validate the project's token vocabulary."""
    with path.open("r", encoding="utf-8") as handle:
        tokens = json.load(handle)
    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        raise ValueError(f"{path} must contain a JSON list of token strings.")
    token_to_idx = {token: index for index, token in enumerate(tokens)}
    required = {PAD_TOKEN, EOS_TOKEN, MSK_TOKEN}
    required.update(layer for sample in SAMPLES for layer in sample["layers"])
    missing = sorted(required.difference(token_to_idx))
    if missing:
        raise ValueError(f"Vocabulary {path} is missing required tokens: {missing}")
    return tokens, token_to_idx


def encode_samples(token_to_idx: dict[str, int], sequence_length: int) -> torch.Tensor:
    """Encode four layers, EOS, and optional PAD tokens for every example."""
    if sequence_length < 5:
        raise ValueError("--sequence-length must be at least 5 (four layers plus EOS).")
    rows: list[list[int]] = []
    for sample in SAMPLES:
        row = [token_to_idx[layer] for layer in sample["layers"]]
        row.append(token_to_idx[EOS_TOKEN])
        row.extend([token_to_idx[PAD_TOKEN]] * (sequence_length - len(row)))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.long)


def resolve_device(value: str) -> torch.device:
    """Resolve the requested compute device."""
    if value.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def spectral_summary(wavelengths: torch.Tensor, spectrum: torch.Tensor) -> dict[str, float]:
    """Return compact extrema that make each example easy to inspect."""
    names = ("R", "A", "T")
    summary: dict[str, float] = {}
    for channel, name in enumerate(names):
        values = spectrum[channel]
        maximum, max_index = values.max(dim=0)
        minimum, min_index = values.min(dim=0)
        summary[f"{name}_max"] = round(float(maximum), 6)
        summary[f"{name}_max_nm"] = float(wavelengths[max_index])
        summary[f"{name}_min"] = round(float(minimum), 6)
        summary[f"{name}_min_nm"] = float(wavelengths[min_index])
    return summary


def write_csv(path: Path, wavelengths: torch.Tensor, spectra: torch.Tensor) -> None:
    """Write a long-form, human-readable copy of the spectra."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sample_index", "sample_name", "wavelength_nm", "R", "A", "T"))
        for sample_index, sample in enumerate(SAMPLES):
            for wavelength_index, wavelength in enumerate(wavelengths.tolist()):
                writer.writerow(
                    (
                        sample_index,
                        sample["name"],
                        wavelength,
                        *[float(spectra[sample_index, channel, wavelength_index]) for channel in range(3)],
                    )
                )


def main() -> None:
    """Simulate and save the curated dataset."""
    args = parse_args()
    if args.wavelength_step <= 0 or args.wavelength_max < args.wavelength_min:
        raise ValueError("The wavelength range must be increasing and --wavelength-step must be positive.")
    if not args.tokens.is_file():
        raise FileNotFoundError(f"Token vocabulary does not exist: {args.tokens}")
    if not args.materials.is_dir():
        raise FileNotFoundError(f"Materials directory does not exist: {args.materials}")

    outputs = ("dataset.safetensors", "spectra.csv", "manifest.json", "tokens.json")
    existing = [args.out_dir / name for name in outputs if (args.out_dir / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Output already exists ({existing[0]}). Pass --overwrite to replace this example dataset.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tokens, token_to_idx = load_vocabulary(args.tokens)
    stacks = encode_samples(token_to_idx, args.sequence_length)
    wavelengths = torch.arange(
        args.wavelength_min,
        args.wavelength_max + args.wavelength_step * 0.5,
        args.wavelength_step,
        dtype=torch.float32,
    )
    device = resolve_device(args.device)
    idx_to_token = dict(enumerate(tokens))
    tmm, wavelength_device, theta = build_tmm(
        incidence_angle=args.incidence_angle,
        device=device,
        wavelengths=wavelengths,
        path_materials=str(args.materials),
        idx_to_token=idx_to_token,
    )
    with torch.no_grad():
        spectra = tmm(
            stacks.to(device),
            wavelength_device,
            theta,
            eos=token_to_idx[EOS_TOKEN],
            pad=token_to_idx[PAD_TOKEN],
            msk=token_to_idx[MSK_TOKEN],
            pol=args.polarization,
        ).cpu()

    safetensors.torch.save_file(
        {"spectra": spectra.contiguous(), "thin_films": stacks.contiguous()},
        str(args.out_dir / "dataset.safetensors"),
    )
    write_csv(args.out_dir / "spectra.csv", wavelengths, spectra)
    shutil.copyfile(args.tokens, args.out_dir / "tokens.json")

    manifest = {
        "format": "optollama-four-layer-examples-v1",
        "dataset_file": "dataset.safetensors",
        "tensor_shapes": {"spectra": list(spectra.shape), "thin_films": list(stacks.shape)},
        "layer_order": "incident_air_to_exit_air",
        "simulation": {
            "solver": "optollama.evaluation.simulation.TMMSpectrum (tmm_fast coherent TMM)",
            "wavelength_min_nm": args.wavelength_min,
            "wavelength_max_nm": args.wavelength_max,
            "wavelength_step_nm": args.wavelength_step,
            "incidence_angle_deg": args.incidence_angle,
            "polarization": args.polarization,
            "incident_medium": "air",
            "exit_medium": "air",
            "coherence": "all finite layers coherent",
        },
        "samples": [
            {
                "index": index,
                **sample,
                "summary": spectral_summary(wavelengths, spectra[index]),
            }
            for index, sample in enumerate(SAMPLES)
        ],
    }
    with (args.out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print(f"Saved {len(SAMPLES)} samples to {args.out_dir / 'dataset.safetensors'}")
    print(f"spectra shape={tuple(spectra.shape)}, thin_films shape={tuple(stacks.shape)}, device={device}")


if __name__ == "__main__":
    main()
