# Open-layer flow MVP

This package is an independent layer-space experiment. It does not change the
existing depth-field trainers or their checkpoint format.

## Representation

Each generated slot is one physical optical layer:

- material: a pointer into the material bank supplied with the query;
- thickness: a continuous value in nm;
- layer count: supplied at inference time or enumerated over configured values.

The target encoder receives spectral values together with continuous wavelength
features. The material encoder receives each candidate's `n(lambda), k(lambda)`
curve on the same wavelength query. Candidate order is randomized during
training, and the output head scores candidate embeddings instead of fixed
material class IDs.

The first version uses masked material denoising and rectified flow for
normalized log thickness. It does not yet implement CTMC material flow,
angle/polarization conditioning, learned layer count, spectral reward training,
or depth-field convolutional refinement.

## Training

Set the HPC `PATHS.DATA_PATH` in `configs/open_layer_flow_01.yaml`, then launch a
single process per GPU. For example, on four nodes with four GPUs each:

```bash
sbatch --nodes=4 --ntasks-per-node=4 --gres=gpu:4 \
  --job-name=open-layer-01 --output=logs/open-layer-01_%j.out \
  --error=logs/open-layer-01_%j.err --wrap \
  "srun --gpu-bind=closest python -u scripts/train_open_layer_flow.py \
  --config configs/open_layer_flow_01.yaml"
```

For a bounded smoke run:

```bash
python scripts/train_open_layer_flow.py \
  --config configs/open_layer_flow_01.yaml \
  --batch-size 2 --epochs 1 \
  --max-train-samples 8 --max-val-samples 4 --max-train-steps 2
```

Checkpoints and history are written to `OPEN_LAYER.OUT_DIR`:

- `open-layer-best.pt` and `open-layer-best.safetensors`;
- `open-layer-last.pt` and `open-layer-last.safetensors`;
- `open-layer-history.json`.

The full configuration is approximately 77.85 million trainable parameters.

## Inference

The default inference enumerates configured layer counts, generates
`MC_SAMPLES` for each count, and selects the best candidate by exact TMM R/T
MAE:

```bash
python scripts/inference_open_layer_flow.py \
  --config configs/open_layer_flow_01.yaml \
  --checkpoint data/output_OPEN_LAYER_FLOW_01/open-layer-best.pt \
  --target data/targets/target_R_BR_realistic.csv
```

Restricting the query-local bank also reduces inference cost:

```bash
python scripts/inference_open_layer_flow.py \
  --config configs/open_layer_flow_01.yaml \
  --checkpoint data/output_OPEN_LAYER_FLOW_01/open-layer-best.pt \
  --target data/targets/target_R_BR_realistic.csv \
  --candidate-materials Ag,Al,SiO2,TiO2,MgF2 \
  --layer-counts 1,2,4,8,12 --mc-samples 20
```

Results use the existing per-target output convention. The default location is
`<OPEN_LAYER.OUT_DIR>/<target-stem>/samples-YYMMDD-HHMM.json`, with a sibling
`plot-bundle.npz` for the wavelength grid.

## Target formats

Existing headerless `[R,A,T] x W` CSV files remain supported and use the grid
from the config. Coordinate-bearing targets can use either:

```csv
wavelength_nm,R,T
400,0.1,0.9
425,0.2,0.8
```

or JSON:

```json
{
  "wavelengths_nm": [400, 425],
  "spectra": [[0.1, 0.2], [0.0, 0.0], [0.9, 0.8]]
}
```

The model accepts arbitrary query spacing and point count, but the first
training corpus only supports learned behavior within 300-1700 nm. Optical
constant curves must cover a requested grid. The configured 100 nm endpoint
tolerance exists solely because the current corpus was generated with clamped
material tables, notably TiO2 below approximately 396 nm.

## Compatibility and limits

An inference material does not need a token-vocabulary entry. Its CSV must be
present in `MATERIALS_PATH`, and it must be included in the candidate bank. The
exact TMM path creates an identity-only token internally and uses the generated
continuous thickness override.

This architecture enables unseen-material evaluation, but useful zero-shot
selection is an empirical question. Set, for example,
`OPEN_LAYER.MATERIAL_BANK.HOLDOUT_MATERIALS: [TiO2]` before a fresh run. Samples
containing TiO2 then produce no training gradient, and TiO2 is excluded from
training distractor banks; validation and inference still expose its n/k curve.
Likewise, variable wavelength coordinates remove the fixed-grid architectural
restriction but do not by themselves establish extrapolation beyond the
training range.
