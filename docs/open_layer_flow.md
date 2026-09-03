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
normalized log thickness. Two material processes are available:

- `monotonic` (the `open_layer_flow_01.yaml` baseline) reveals masked slots once;
- `full_remask` predicts every active material slot at every step, then randomly
  masks a new mixture of isolated slots and contiguous layer spans.

`full_remask` keeps active-layer, material-MASK, and padding states separate. A
masked layer still carries its evolving continuous thickness state. The pointer
head never predicts MASK; the final step contains only query-local candidate
materials. The model does not yet implement CTMC material flow,
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

Sharded loading intentionally uses `NUM_WORKERS: 0`. `ShardedSpectraDataset`
already partitions samples across DDP ranks; additional DataLoader workers would
each replay the complete rank-local range and multiply the apparent epoch
length. With 20 million samples, four ranks, and batch size 64 per rank, the
correct progress-bar length is 78,125 batches.

Training skips isolated non-finite forward or gradient steps synchronously on
all ranks. The progress bar reports `nf` and `amp_skip`; the history records
`nonfinite_forward_steps`, `nonfinite_gradient_steps`, `amp_skipped_steps`, and
`optimizer_steps`. `MAX_CONSECUTIVE_NONFINITE_STEPS` controls when repeated
failures abort instead of being hidden.

The live `loss`, `mat`, and `full` values are cumulative, sample-weighted
averages for the current epoch on the displayed rank. Final epoch metrics are
reduced across all ranks. `grad` remains the current step's gradient norm so
that short numerical spikes stay visible.

With `SYNC_SHAPES_ACROSS_RANKS: true`, all DDP ranks use the same query sampling
mode and wavelength count on a given batch. Window starts, random wavelength
positions, candidate order, and distractors remain rank-local. This balances
per-rank tensor shapes without reducing the diversity of query content.

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

### Full-prediction random-remask experiment

`configs/open_layer_flow_02_remask.yaml` enables the layer-space counterpart of
the depth-field denoising policy:

- 30% of the corruption budget is allocated to isolated layer slots;
- 70% is allocated to contiguous spans of 2-8 physical layer slots;
- span length grows with noise level;
- up to 10% of corrupted slots use a wrong query-local material instead of MASK,
  with replacement probability decaying to zero at the all-MASK endpoint;
- material CE weights are 1.0 on corrupted slots and 0.1 on visible slots.

Training and sampling use the same span policy. Sampling starts from all MASK,
rewrites every active material slot, randomly constructs the next mask pattern,
and independently advances continuous thickness flow. Existing checkpoints that
do not contain these settings load as the monotonic baseline.

Launch it with the same job script as the baseline after changing `CONFIG` to:

```bash
CONFIG=/scratch/htc/jschaibl/repos/OptoLlama/configs/open_layer_flow_02_remask.yaml
```

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
