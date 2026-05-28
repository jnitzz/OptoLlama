#!/usr/bin/env python

import argparse
import collections
import math
from pathlib import Path
from typing import Sequence

import safetensors.torch
import torch
import tqdm

import optollama.data
import optollama.evaluation
import optollama.utils


SUPPORTED_FAMILIES = (
    "dbr",
    "chirped_dbr",
    "cavity",
    "symmetric_motif",
    "rugate",
    "apodized_dbr",
    "multi_cavity",
    "edge_filter",
    "multi_band_filter",
    "metal_dielectric_metal",
    "dielectric_metal_dielectric",
    "absorber_backed",
    "hybrid_dbr_metal",
    "sparse_metal_random",
    "tco_stack",
    "random_dielectric",
    "random",
)

HYBRID_FAMILIES = {
    "metal_dielectric_metal",
    "dielectric_metal_dielectric",
    "absorber_backed",
    "hybrid_dbr_metal",
    "sparse_metal_random",
    "tco_stack",
}

DEFAULT_FAMILY_WEIGHTS = {
    "dbr": 0.055,
    "chirped_dbr": 0.055,
    "cavity": 0.055,
    "symmetric_motif": 0.055,
    "rugate": 0.055,
    "apodized_dbr": 0.055,
    "multi_cavity": 0.055,
    "edge_filter": 0.055,
    "multi_band_filter": 0.055,
    "metal_dielectric_metal": 0.045,
    "dielectric_metal_dielectric": 0.045,
    "absorber_backed": 0.045,
    "hybrid_dbr_metal": 0.045,
    "sparse_metal_random": 0.045,
    "tco_stack": 0.045,
    "random_dielectric": 0.13,
    "random": 0.105,
}


CONFIG_DEFAULT = "configs/optollama.yaml"

CONFIG_DEFAULT_MAP = {
    "OUT_DIR": "out_dir",
    "NUM_SAMPLES": "num_samples",
    "SHARD_SIZE": "shard_size",
    "BATCH_SIZE": "batch_size",
    "FAMILY_WEIGHTS": "family_weights",
    "SEED": "seed",
    "DEVICE": "device",
    "OVERWRITE": "overwrite",
    "MIN_LAYERS": "min_layers",
    "MAX_LAYERS": "max_layers",
    "OUTPUT_SEQ_LEN": "output_seq_len",
    "DBR_MIN_LAYERS": "dbr_min_layers",
    "CHIRPED_DBR_MIN_LAYERS": "chirped_dbr_min_layers",
    "CAVITY_MIN_LAYERS": "cavity_min_layers",
    "SYMMETRIC_MOTIF_MIN_LAYERS": "symmetric_motif_min_layers",
    "RUGATE_MIN_LAYERS": "rugate_min_layers",
    "APODIZED_DBR_MIN_LAYERS": "apodized_dbr_min_layers",
    "MULTI_CAVITY_MIN_LAYERS": "multi_cavity_min_layers",
    "EDGE_FILTER_MIN_LAYERS": "edge_filter_min_layers",
    "MULTI_BAND_FILTER_MIN_LAYERS": "multi_band_filter_min_layers",
    "METAL_DIELECTRIC_METAL_MIN_LAYERS": "metal_dielectric_metal_min_layers",
    "DIELECTRIC_METAL_DIELECTRIC_MIN_LAYERS": "dielectric_metal_dielectric_min_layers",
    "ABSORBER_BACKED_MIN_LAYERS": "absorber_backed_min_layers",
    "HYBRID_DBR_METAL_MIN_LAYERS": "hybrid_dbr_metal_min_layers",
    "SPARSE_METAL_RANDOM_MIN_LAYERS": "sparse_metal_random_min_layers",
    "TCO_STACK_MIN_LAYERS": "tco_stack_min_layers",
    "CENTER_MIN": "center_min",
    "CENTER_MAX": "center_max",
    "STRUCTURE_JITTER_FRACTION": "structure_jitter_fraction",
    "THICKNESS_MIN": "thickness_min",
    "THICKNESS_MAX": "thickness_max",
    "THICKNESS_STEP": "thickness_step",
    "WAVELENGTH_MIN": "wavelength_min",
    "WAVELENGTH_MAX": "wavelength_max",
    "WAVELENGTH_STEP": "wavelength_step",
    "ANGLES": "angles",
    "ANGLE_WEIGHTS": "angle_weights",
    "POLARIZATIONS": "polarizations",
    "JITTER_REALIZATIONS": "jitter_realizations",
    "THICKNESS_JITTER_NM": "thickness_jitter_nm",
    "SAVE_DTYPE": "save_dtype",
}


def realistic_defaults_from_config(path: str) -> dict[str, object]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}

    cfg = optollama.utils.load_config_file(str(cfg_path))
    block = cfg.get("REALISTIC_DATASET") or {}
    defaults: dict[str, object] = {}
    for cfg_key, arg_name in CONFIG_DEFAULT_MAP.items():
        if cfg_key in block:
            defaults[arg_name] = block[cfg_key]

    return defaults


def parse_family_weights(value: str) -> dict[str, float]:
    """
    Parse comma-separated family weights, e.g. ``dbr=0.1,random=0.2``.
    """
    weights: dict[str, float] = {}
    for item in str(value).split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError("Family weights must use name=value pairs separated by commas.")
        family, weight = item.split("=", 1)
        family = family.strip()
        try:
            weights[family] = float(weight)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid weight for family {family!r}: {weight!r}") from exc
    return weights


def parse_args() -> argparse.Namespace:
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--config", type=str, default=CONFIG_DEFAULT, help="Project config YAML.")
    known, _ = base.parse_known_args()

    p = argparse.ArgumentParser(
        description=(
            "Generate tokenized thin-film stacks with realistic spectra from "
            "angle, polarization, and thickness-jitter averaged TMM."
        ),
        parents=[base],
    )
    p.add_argument("--out-dir", type=str, default=None, help="Output directory for shards and generated tokens.json.")
    p.add_argument("--num-samples", type=int, default=None, help="Total number of samples to write.")
    p.add_argument("--shard-size", type=int, default=100000, help="Samples per safetensors shard.")
    p.add_argument("--batch-size", type=int, default=128, help="Stacks simulated per averaged TMM batch.")
    p.add_argument(
        "--family-weights",
        type=parse_family_weights,
        default=DEFAULT_FAMILY_WEIGHTS,
        help=(
            "Comma-separated generation weights, e.g. "
            "'dbr=0.08,chirped_dbr=0.08,random_dielectric=0.14,random=0.14'."
        ),
    )
    p.add_argument("--seed", type=int, default=None, help="Random seed. Defaults to config SEED.")
    p.add_argument("--device", type=str, default=None, help='Execution device, e.g. "cuda", "cuda:0", or "cpu".')
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting existing realistic shards in out-dir.")

    p.add_argument("--min-layers", type=int, default=0, help="Global minimum layer count.")
    p.add_argument("--max-layers", type=int, default=100, help="Maximum layer count.")
    p.add_argument("--output-seq-len", type=int, default=None, help="Saved thin_films width. Defaults to max_layers + 1.")
    p.add_argument("--dbr-min-layers", type=int, default=2, help="Minimum layers for DBR samples.")
    p.add_argument("--chirped-dbr-min-layers", type=int, default=2, help="Minimum layers for chirped DBR samples.")
    p.add_argument("--cavity-min-layers", type=int, default=5, help="Minimum layers for cavity samples.")
    p.add_argument("--symmetric-motif-min-layers", type=int, default=4, help="Minimum layers for symmetric motif samples.")
    p.add_argument("--rugate-min-layers", type=int, default=8, help="Minimum layers for rugate-like samples.")
    p.add_argument("--apodized-dbr-min-layers", type=int, default=4, help="Minimum layers for apodized DBR samples.")
    p.add_argument("--multi-cavity-min-layers", type=int, default=7, help="Minimum layers for multi-cavity samples.")
    p.add_argument("--edge-filter-min-layers", type=int, default=4, help="Minimum layers for edge-filter samples.")
    p.add_argument("--multi-band-filter-min-layers", type=int, default=6, help="Minimum layers for multi-band samples.")
    p.add_argument("--metal-dielectric-metal-min-layers", type=int, default=3, help="Minimum layers for MDM samples.")
    p.add_argument("--dielectric-metal-dielectric-min-layers", type=int, default=3, help="Minimum layers for DMD samples.")
    p.add_argument("--absorber-backed-min-layers", type=int, default=2, help="Minimum layers for absorber-backed samples.")
    p.add_argument("--hybrid-dbr-metal-min-layers", type=int, default=3, help="Minimum layers for hybrid DBR-metal samples.")
    p.add_argument("--sparse-metal-random-min-layers", type=int, default=3, help="Minimum layers for sparse-metal random samples.")
    p.add_argument("--tco-stack-min-layers", type=int, default=3, help="Minimum layers for TCO-stack samples.")
    p.add_argument("--center-min", type=float, default=300.0, help="Minimum structured-family design center wavelength.")
    p.add_argument("--center-max", type=float, default=1700.0, help="Maximum structured-family design center wavelength.")
    p.add_argument(
        "--structure-jitter-fraction",
        type=float,
        default=0.0,
        help="Relative token-grid jitter used while constructing structured stacks.",
    )

    p.add_argument("--thickness-min", type=int, default=10, help="Minimum token thickness in nm.")
    p.add_argument("--thickness-max", type=int, default=500, help="Maximum token thickness in nm.")
    p.add_argument("--thickness-step", type=int, default=5, help="Token thickness step in nm.")
    p.add_argument("--wavelength-min", type=int, default=300, help="Minimum simulated wavelength in nm.")
    p.add_argument("--wavelength-max", type=int, default=1700, help="Maximum simulated wavelength in nm.")
    p.add_argument("--wavelength-step", type=int, default=5, help="Wavelength step in nm.")

    p.add_argument("--angles", nargs="+", type=float, default=[0.0, 1.0, 2.0], help="Averaged incidence angles in degrees.")
    p.add_argument(
        "--angle-weights",
        nargs="+",
        type=float,
        default=[1.0, 1.0, 1.0],
        help="Angle weights. Must match --angles.",
    )
    p.add_argument("--polarizations", nargs="+", choices=["s", "p"], default=["s", "p"], help="Polarizations to average.")
    p.add_argument("--jitter-realizations", type=int, default=5, help="Thickness-jitter realizations per stack.")
    p.add_argument("--thickness-jitter-nm", type=float, default=2.0, help="Uniform per-layer jitter range, +/- nm.")
    p.add_argument("--save-dtype", choices=["float16", "float32"], default="float16", help="Saved spectra dtype.")
    p.set_defaults(**realistic_defaults_from_config(known.config))
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.out_dir is None:
        raise ValueError("--out-dir must be set or REALISTIC_DATASET.OUT_DIR must exist in the config.")
    if args.num_samples is None:
        raise ValueError("--num-samples must be set or REALISTIC_DATASET.NUM_SAMPLES must exist in the config.")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    validate_family_weights(args.family_weights)
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.min_layers < 0:
        raise ValueError("--min-layers must be >= 0.")
    if args.max_layers < args.min_layers:
        raise ValueError("--max-layers must be >= --min-layers.")
    if args.thickness_min <= 0 or args.thickness_step <= 0:
        raise ValueError("--thickness-min and --thickness-step must be positive.")
    if args.thickness_max < args.thickness_min:
        raise ValueError("--thickness-max must be >= --thickness-min.")
    if (args.thickness_max - args.thickness_min) % args.thickness_step != 0:
        raise ValueError("Thickness range must be divisible by --thickness-step.")
    if args.wavelength_max < args.wavelength_min:
        raise ValueError("--wavelength-max must be >= --wavelength-min.")
    if args.wavelength_step <= 0:
        raise ValueError("--wavelength-step must be positive.")
    if len(args.angles) != len(args.angle_weights):
        raise ValueError("--angle-weights must have the same length as --angles.")
    if sum(args.angle_weights) <= 0:
        raise ValueError("--angle-weights must have positive total weight.")
    if args.jitter_realizations <= 0:
        raise ValueError("--jitter-realizations must be positive.")
    if args.thickness_jitter_nm < 0:
        raise ValueError("--thickness-jitter-nm must be non-negative.")
    if args.structure_jitter_fraction < 0:
        raise ValueError("--structure-jitter-fraction must be non-negative.")
    if args.center_max <= args.center_min:
        raise ValueError("--center-max must be greater than --center-min.")

    family_min = {
        "dbr": args.dbr_min_layers,
        "chirped_dbr": args.chirped_dbr_min_layers,
        "cavity": args.cavity_min_layers,
        "symmetric_motif": args.symmetric_motif_min_layers,
        "rugate": args.rugate_min_layers,
        "apodized_dbr": args.apodized_dbr_min_layers,
        "multi_cavity": args.multi_cavity_min_layers,
        "edge_filter": args.edge_filter_min_layers,
        "multi_band_filter": args.multi_band_filter_min_layers,
        "metal_dielectric_metal": args.metal_dielectric_metal_min_layers,
        "dielectric_metal_dielectric": args.dielectric_metal_dielectric_min_layers,
        "absorber_backed": args.absorber_backed_min_layers,
        "hybrid_dbr_metal": args.hybrid_dbr_metal_min_layers,
        "sparse_metal_random": args.sparse_metal_random_min_layers,
        "tco_stack": args.tco_stack_min_layers,
    }
    for family, min_layers in family_min.items():
        if min_layers < 0:
            raise ValueError(f"--{family.replace('_', '-')}-min-layers must be >= 0.")
        if max(args.min_layers, min_layers) > args.max_layers:
            raise ValueError(f"{family} minimum layers exceed --max-layers.")


def validate_family_weights(weights: dict[str, float]) -> None:
    if not weights:
        raise ValueError("At least one family weight must be configured.")
    unknown = sorted(set(weights) - set(SUPPORTED_FAMILIES))
    if unknown:
        raise ValueError(f"Unknown family weights: {unknown}. Supported families: {list(SUPPORTED_FAMILIES)}")
    if any(float(weight) < 0.0 for weight in weights.values()):
        raise ValueError("Family weights must be non-negative.")
    if sum(float(weight) for weight in weights.values()) <= 0.0:
        raise ValueError("At least one family weight must be positive.")


def normalized_family_weights(weights: dict[str, float]) -> dict[str, float]:
    validate_family_weights(weights)
    ordered = {family: float(weights.get(family, 0.0)) for family in SUPPORTED_FAMILIES}
    total = sum(ordered.values())
    return {family: weight / total for family, weight in ordered.items() if weight > 0.0}


def load_base_material_order(cfg: dict) -> list[str]:
    tokens_path = Path(cfg["TOKENS_PATH"])
    if tokens_path.exists():
        tokens = optollama.utils.load_as_json(str(tokens_path))
        materials: list[str] = []
        for token in tokens:
            if token in optollama.data.SPECIAL_TOKENS or "_" not in token:
                continue
            material = token.rsplit("_", 1)[0]
            if material not in materials:
                materials.append(material)
        if materials:
            return materials

    materials_path = Path(cfg["MATERIALS_PATH"])
    return [path.stem for path in sorted(materials_path.glob("*.csv"))]


def build_tokens(materials: Sequence[str], thickness_min: int, thickness_max: int, thickness_step: int) -> list[str]:
    thicknesses = range(int(thickness_min), int(thickness_max) + 1, int(thickness_step))
    tokens = [f"{material}_{thickness}" for material in materials for thickness in thicknesses]
    tokens.extend([optollama.data.EOS_TOKEN, optollama.data.PAD_TOKEN, optollama.data.MSK_TOKEN])
    return tokens


def token_maps(tokens: Sequence[str]) -> tuple[dict[str, int], dict[int, str], int, int, int]:
    token_to_idx = {token: idx for idx, token in enumerate(tokens)}
    idx_to_token = {idx: token for idx, token in enumerate(tokens)}
    eos = token_to_idx[optollama.data.EOS_TOKEN]
    pad = token_to_idx[optollama.data.PAD_TOKEN]
    msk = token_to_idx[optollama.data.MSK_TOKEN]
    return token_to_idx, idx_to_token, eos, pad, msk


def weighted_family_counts(num_samples: int, family_weights: dict[str, float]) -> dict[str, int]:
    raw = {family: num_samples * weight for family, weight in family_weights.items()}
    counts = {family: int(math.floor(value)) for family, value in raw.items()}
    remainder = num_samples - sum(counts.values())
    order = sorted(family_weights, key=lambda family: (raw[family] - counts[family], family_weights[family]), reverse=True)
    for family in order[:remainder]:
        counts[family] += 1
    return counts


def _randint(generator: torch.Generator, high: int) -> int:
    return int(torch.randint(high, (1,), generator=generator, device="cpu").item())


def draw_family_batch(
    remaining: dict[str, int],
    family_weights: dict[str, float],
    batch_size: int,
    generator: torch.Generator,
) -> list[str]:
    families: list[str] = []
    for _ in range(batch_size):
        total = sum(remaining.values())
        if total <= 0:
            break
        draw = _randint(generator, total)
        for family in family_weights:
            count = remaining[family]
            if draw < count:
                remaining[family] -= 1
                families.append(family)
                break
            draw -= count
    return families


def family_min_layers(args: argparse.Namespace, family: str) -> int:
    defaults = {
        "dbr": int(args.dbr_min_layers),
        "chirped_dbr": int(args.chirped_dbr_min_layers),
        "cavity": int(args.cavity_min_layers),
        "symmetric_motif": int(args.symmetric_motif_min_layers),
        "rugate": int(args.rugate_min_layers),
        "apodized_dbr": int(args.apodized_dbr_min_layers),
        "multi_cavity": int(args.multi_cavity_min_layers),
        "edge_filter": int(args.edge_filter_min_layers),
        "multi_band_filter": int(args.multi_band_filter_min_layers),
        "metal_dielectric_metal": int(args.metal_dielectric_metal_min_layers),
        "dielectric_metal_dielectric": int(args.dielectric_metal_dielectric_min_layers),
        "absorber_backed": int(args.absorber_backed_min_layers),
        "hybrid_dbr_metal": int(args.hybrid_dbr_metal_min_layers),
        "sparse_metal_random": int(args.sparse_metal_random_min_layers),
        "tco_stack": int(args.tco_stack_min_layers),
    }
    return max(int(args.min_layers), defaults.get(family, int(args.min_layers)))


def build_libraries(tokens: list[str], token_to_idx: dict[str, int]):
    all_pool = optollama.data.build_extension_pool(tokens, token_to_idx, allowed_groups=None)
    dielectric_pool = optollama.data.build_extension_pool(tokens, token_to_idx, allowed_groups=["dielectrics"])
    all_index = optollama.data.build_token_mutation_index(tokens, all_pool)
    dielectric_index = optollama.data.build_token_mutation_index(tokens, dielectric_pool)
    return {
        "all": optollama.data.build_long_stack_library(all_index),
        "dielectric": optollama.data.build_long_stack_library(dielectric_index),
    }


def generate_stack_batch(
    families: Sequence[str],
    libraries: dict[str, optollama.data.LongStackLibrary],
    args: argparse.Namespace,
    eos: int,
    pad: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    stacks: list[torch.Tensor] = []
    lengths: list[int] = []
    output_seq_len = int(args.output_seq_len or (args.max_layers + 1))

    for family in families:
        library = libraries["all"] if family == "random" or family in HYBRID_FAMILIES else libraries["dielectric"]
        layers = optollama.data.generate_long_stack_layers(
            family,
            library=library,
            min_layers=family_min_layers(args, family),
            max_layers=int(args.max_layers),
            center_min_nm=float(args.center_min),
            center_max_nm=float(args.center_max),
            jitter_fraction=float(args.structure_jitter_fraction),
            generator=generator,
        )
        stacks.append(
            optollama.data.encode_layer_tokens(
                layers,
                output_seq_len=output_seq_len,
                eos=eos,
                pad=pad,
                device=device,
            )
        )
        lengths.append(len(layers))

    return torch.stack(stacks, dim=0), torch.as_tensor(lengths, dtype=torch.long)


@torch.no_grad()
def simulate_realistic_batch(
    stacks: torch.Tensor,
    tmm: optollama.evaluation.simulation.TMMSpectrum,
    wavelengths: torch.Tensor,
    angle_thetas: Sequence[torch.Tensor],
    angle_weights: Sequence[float],
    polarizations: Sequence[str],
    jitter_realizations: int,
    thickness_jitter_nm: float,
    eos: int,
    pad: int,
    msk: int,
) -> torch.Tensor:
    return optollama.evaluation.simulation.simulate_token_sequence_averaged(
        stacks,
        tmm=tmm,
        wavelengths=wavelengths,
        angle_thetas=angle_thetas,
        angle_weights=angle_weights,
        polarizations=polarizations,
        jitter_realizations=jitter_realizations,
        thickness_jitter_nm=thickness_jitter_nm,
        eos=eos,
        pad=pad,
        msk=msk,
    )


def save_shard(
    out_dir: Path,
    shard_idx: int,
    spectra: torch.Tensor,
    stacks: torch.Tensor,
    save_dtype: torch.dtype,
) -> Path:
    out_path = out_dir / f"realistic-{shard_idx:05d}.safetensors"
    safetensors.torch.save_file(
        {
            "spectra": spectra.to(save_dtype).contiguous().cpu(),
            "thin_films": stacks.long().contiguous().cpu(),
        },
        str(out_path),
    )
    return out_path


def flush_shards(
    out_dir: Path,
    shard_idx: int,
    spectra_parts: list[torch.Tensor],
    stack_parts: list[torch.Tensor],
    buffered_count: int,
    shard_size: int,
    save_dtype: torch.dtype,
    written_paths: list[str],
    force: bool = False,
) -> tuple[int, list[torch.Tensor], list[torch.Tensor], int]:
    if buffered_count == 0 or (buffered_count < shard_size and not force):
        return shard_idx, spectra_parts, stack_parts, buffered_count

    spectra = torch.cat(spectra_parts, dim=0)
    stacks = torch.cat(stack_parts, dim=0)
    write_count = spectra.size(0) if force else (spectra.size(0) // shard_size) * shard_size

    start = 0
    while start < write_count:
        end = min(start + shard_size, write_count)
        out_path = save_shard(out_dir, shard_idx, spectra[start:end], stacks[start:end], save_dtype)
        written_paths.append(str(out_path))
        print(f"Saved {out_path} ({end - start} samples)")
        shard_idx += 1
        start = end

    if write_count >= spectra.size(0):
        return shard_idx, [], [], 0

    return shard_idx, [spectra[write_count:].cpu()], [stacks[write_count:].cpu()], spectra.size(0) - write_count


def main() -> None:
    args = parse_args()
    validate_args(args)

    cfg = optollama.utils.load_config_file(args.config)
    seed = int(args.seed if args.seed is not None else cfg["SEED"])
    optollama.utils.set_all_seeds(seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.overwrite and any(out_dir.glob("realistic-*.safetensors")):
        raise FileExistsError(f"{out_dir} already contains realistic shards. Use --overwrite to replace/add shards.")

    output_seq_len = int(args.output_seq_len or (args.max_layers + 1))
    if output_seq_len < args.max_layers + 1:
        raise ValueError("--output-seq-len must be at least --max-layers + 1.")

    wavelengths_real = torch.arange(
        int(args.wavelength_min),
        int(args.wavelength_max) + 1,
        int(args.wavelength_step),
        dtype=torch.float32,
    )
    wavelengths = wavelengths_real.to(device=device, dtype=torch.complex128)
    angle_thetas = [
        torch.tensor(angle * math.pi / 180.0, device=device, dtype=torch.complex128).unsqueeze(0)
        for angle in args.angles
    ]
    save_dtype = torch.float16 if args.save_dtype == "float16" else torch.float32

    materials = load_base_material_order(cfg)
    tokens = build_tokens(materials, int(args.thickness_min), int(args.thickness_max), int(args.thickness_step))
    token_to_idx, idx_to_token, eos, pad, msk = token_maps(tokens)
    optollama.utils.save_as_json(str(out_dir / "tokens.json"), tokens)

    libraries = build_libraries(tokens, token_to_idx)
    nk_dict = optollama.utils.load_materials(cfg["MATERIALS_PATH"], wavelengths_real)
    tmm = optollama.evaluation.simulation.TMMSpectrum(nk_dict, idx_to_token, device=device).to(device).eval()

    family_weights = normalized_family_weights(args.family_weights)
    family_targets = weighted_family_counts(int(args.num_samples), family_weights)
    remaining = dict(family_targets)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 101)

    print(f"Using device: {device}")
    print(f"Wavelengths: {args.wavelength_min}-{args.wavelength_max} nm step {args.wavelength_step} ({wavelengths.numel()} points)")
    print(f"Vocabulary: {len(tokens)} tokens ({len(materials)} materials, {args.thickness_step} nm thickness grid)")
    print(f"Family weights: {family_weights}")
    print(f"Family targets: {family_targets}")
    print(
        "Averaging: "
        f"{len(args.angles)} angles x {len(args.polarizations)} polarizations x "
        f"{args.jitter_realizations} jitter realizations"
    )

    shard_idx = 0
    written = 0
    written_paths: list[str] = []
    spectra_buffer: list[torch.Tensor] = []
    stack_buffer: list[torch.Tensor] = []
    buffered_count = 0
    family_written = collections.Counter()
    length_hist = collections.Counter()

    pbar = tqdm.tqdm(total=int(args.num_samples), desc="realistic-generate")
    while written < int(args.num_samples):
        batch_count = min(int(args.batch_size), int(args.num_samples) - written)
        batch_families = draw_family_batch(remaining, family_weights, batch_count, generator)
        if not batch_families:
            break

        stacks, lengths = generate_stack_batch(
            batch_families,
            libraries=libraries,
            args=args,
            eos=eos,
            pad=pad,
            generator=generator,
            device=device,
        )
        spectra = simulate_realistic_batch(
            stacks,
            tmm=tmm,
            wavelengths=wavelengths,
            angle_thetas=angle_thetas,
            angle_weights=args.angle_weights,
            polarizations=args.polarizations,
            jitter_realizations=int(args.jitter_realizations),
            thickness_jitter_nm=float(args.thickness_jitter_nm),
            eos=eos,
            pad=pad,
            msk=msk,
        )

        spectra_buffer.append(spectra.detach().to(save_dtype).cpu())
        stack_buffer.append(stacks.detach().cpu())
        buffered_count += stacks.size(0)
        written += stacks.size(0)

        family_written.update(batch_families)
        length_hist.update(int(v) for v in lengths.tolist())
        pbar.update(stacks.size(0))

        shard_idx, spectra_buffer, stack_buffer, buffered_count = flush_shards(
            out_dir,
            shard_idx,
            spectra_buffer,
            stack_buffer,
            buffered_count,
            int(args.shard_size),
            save_dtype,
            written_paths,
        )
        pbar.set_postfix(shard=shard_idx, buffered=buffered_count)

    pbar.close()

    shard_idx, spectra_buffer, stack_buffer, buffered_count = flush_shards(
        out_dir,
        shard_idx,
        spectra_buffer,
        stack_buffer,
        buffered_count,
        int(args.shard_size),
        save_dtype,
        written_paths,
        force=True,
    )

    summary = {
        "num_samples": int(written),
        "shard_size": int(args.shard_size),
        "shards": written_paths,
        "tokens_path": str(out_dir / "tokens.json"),
        "vocab_size": len(tokens),
        "materials": materials,
        "family_weights": family_weights,
        "family_targets": family_targets,
        "family_written": dict(family_written),
        "length_histogram": {str(k): int(v) for k, v in sorted(length_hist.items())},
        "min_layers": int(args.min_layers),
        "max_layers": int(args.max_layers),
        "output_seq_len": int(output_seq_len),
        "family_min_layers": {
            "dbr": family_min_layers(args, "dbr"),
            "chirped_dbr": family_min_layers(args, "chirped_dbr"),
            "cavity": family_min_layers(args, "cavity"),
            "symmetric_motif": family_min_layers(args, "symmetric_motif"),
            "rugate": family_min_layers(args, "rugate"),
            "apodized_dbr": family_min_layers(args, "apodized_dbr"),
            "multi_cavity": family_min_layers(args, "multi_cavity"),
            "edge_filter": family_min_layers(args, "edge_filter"),
            "multi_band_filter": family_min_layers(args, "multi_band_filter"),
            "metal_dielectric_metal": family_min_layers(args, "metal_dielectric_metal"),
            "dielectric_metal_dielectric": family_min_layers(args, "dielectric_metal_dielectric"),
            "absorber_backed": family_min_layers(args, "absorber_backed"),
            "hybrid_dbr_metal": family_min_layers(args, "hybrid_dbr_metal"),
            "sparse_metal_random": family_min_layers(args, "sparse_metal_random"),
            "tco_stack": family_min_layers(args, "tco_stack"),
            "random_dielectric": family_min_layers(args, "random_dielectric"),
            "random": family_min_layers(args, "random"),
        },
        "thickness_grid_nm": {
            "min": int(args.thickness_min),
            "max": int(args.thickness_max),
            "step": int(args.thickness_step),
        },
        "wavelength_grid_nm": {
            "min": int(args.wavelength_min),
            "max": int(args.wavelength_max),
            "step": int(args.wavelength_step),
            "count": int(wavelengths.numel()),
        },
        "averaging": {
            "angles_deg": [float(v) for v in args.angles],
            "angle_weights": [float(v) for v in args.angle_weights],
            "polarizations": list(args.polarizations),
            "jitter_realizations": int(args.jitter_realizations),
            "thickness_jitter_nm": float(args.thickness_jitter_nm),
        },
        "save_dtype": args.save_dtype,
        "seed": int(seed),
    }
    summary_path = out_dir / "realistic_generation_summary.json"
    optollama.utils.save_as_json(str(summary_path), summary)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
