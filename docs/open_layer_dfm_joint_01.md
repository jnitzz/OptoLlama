# Open-layer joint DFM, Run 01

`OPEN_LAYER_DFM_JOINT_01` trains one layer transformer from scratch. Its material
pointer and thickness velocity use the same current material IDs, continuous
geometry, and optical conditions. The count head uses only the conditions.
Training uses the reference layer count as an active-slot mask. Sampling draws a
count once per MC candidate (zero is valid), then starts with all MASK materials
and Gaussian thickness latents. Every step makes revisable layerwise material
jumps and advances the thickness latents with Euler integration. The decoder
allocates a learned total under 10,000 nm across the active slots. Adjacent
equal-material slots are kept separately in training and saved output.

Training logs and `open-layer-dfm-joint-history.json` report material CE,
thickness velocity MSE, and count CE independently. The in-run validation is a
denoising-loss pass over 10,240 examples, not TMM. The best checkpoint is selected
by that loss on EMA weights, not by spectral MAE. Evaluate optics separately.

## HPC training

From the OptoLlama repository root:

```bash
mkdir -p logs
sbatch jobscripts/train_open_layer_dfm_joint_01.sh
```

This requests one node with four H200s for 2 days, 23 hours, and 59 minutes. The
config uses all four training folders (20M total), batch 64 per GPU, and a
50-epoch learning-rate schedule. Slurm may stop a job mid-epoch. The trainer
updates `progress.pt` every 10,000 optimizer steps (2.56M examples at the
configured batch size) and `last.pt` at epoch end.

```bash
out="${OPTOLLAMA_DATA_PATH:-$PWD/data}/output_OPEN_LAYER_DFM_JOINT_01"
ls -lt "$out"/open-layer-dfm-joint-{progress,last}.pt
RESUME_CHECKPOINT="$out/open-layer-dfm-joint-progress.pt" \
  sbatch jobscripts/train_open_layer_dfm_joint_01.sh
```

Use `open-layer-dfm-joint-last.pt` instead if it is newer. A mid-epoch resume
requires the same dataset paths, training sample count, world size, and per-rank
batch size. Do not submit two training jobs against the same output folder.

## Wrong-label training fork

Take one completed checkpoint and submit a matched continuation. These are
training branches, not Git branches. The hard link keeps the source fixed even
as Run 01 atomically replaces its own `last.pt`; both paths must be on the same
filesystem. The launcher runs exactly one additional epoch per arm.

```bash
mkdir -p logs
out="${OPTOLLAMA_DATA_PATH:-$PWD/data}/output_OPEN_LAYER_DFM_JOINT_01"
fork="$out/forks/material-recovery-$(date +%y%m%d-%H%M%S)"
mkdir -p "$fork"
ln "$out/open-layer-dfm-joint-last.pt" "$fork/source.pt" &&
  sbatch --time=06:00:00 --job-name=joint-mat-control jobscripts/train_open_layer_dfm_joint_01.sh "$fork" control &&
  sbatch --time=06:00:00 --job-name=joint-mat-noise jobscripts/train_open_layer_dfm_joint_01.sh "$fork" wrong-label
```

Both arms use the Run 01 config, including batch 64 per rank, sample-based LR,
and 20M examples per epoch. The treatment applies a training-only replacement
probability of `0.10 * 4 * s * (1-s)` to visible materials that have at least
one valid alternative. The control uses zero replacement. Losses and the
validation corruption are identical. The treatment logs the fraction of
visible slots replaced and the accuracy of recovering their original labels.
Re-submit the same arm with the same fork directory to continue from its
progress checkpoint, after the original job has stopped.

For the matched benchmark below, test each arm's own
`$fork/<arm>/open-layer-dfm-joint-last.pt` with `--weights ema` and save into
that arm's directory. Use the normal sampler, without reference inputs.

## Optical inference

Use the same `out` variable and the repository's Python environment. Omit
`--query-bands` to prompt the entire target. A comma separates disjoint bands.
To override the material bank, add `--candidate-materials MgF2,Ta2O5,SiO2`.
To use another set of n,k files, also set `--materials-dir`; exact TMM needs
the candidate files and the EVA substrate file in that directory. Angle and
polarization are supported inputs, but this run trains only at 0 degrees/s.

```bash
srun -p gpu --constraint=H200 --nodes=1 --ntasks=1 --gres=gpu:1 --mem=0 --cpu-bind=none \
  python -u scripts/inference_open_vocab_depth_field.py \
  --config configs/open_layer_dfm_joint_01.yaml \
  --checkpoint "$out/open-layer-dfm-joint-best.pt" --weights ema \
  --target data/targets/target_R_BR_realistic.csv --query-bands 400:1100 \
  --steps 16 --mc-samples 16 --mc-batch-size 4 --device cuda \
  --save "$out/plant-400-1100.json"
```

For a matched 512-target benchmark, use the same test selection and budget as
the earlier depth-field runs:

```bash
srun -p gpu --constraint=H200 --nodes=1 --ntasks=1 --gres=gpu:1 --mem=0 --cpu-bind=none \
  python -u scripts/inference_open_vocab_depth_field.py \
  --config configs/open_layer_dfm_joint_01.yaml \
  --checkpoint "$out/open-layer-dfm-joint-best.pt" --weights ema \
  --split test --max-samples 512 --seed 4 \
  --batch-size 8 --mc-samples 4 --mc-batch-size 4 --steps 16 \
  --query-mode full --full-grid-metrics --record-spectra --record-fields --device cuda \
  --save "$out/rat-test-512-mc04-steps-16.json"
```

The benchmark records the actual scored count, continuous stacks, saved
spectra, and separate query-grid and original-grid exact-TMM RAT metrics. Use
`summary.full_grid` for full-grid scores and `summary.nonempty_reference.full_grid`
for comparisons with runs that skipped empty reference stacks. Best-of-MC is
an oracle score selected by TMM; use single-draw MAE to measure one attempt.

## Reference-count diagnostic

Use one saved checkpoint and run the split benchmark twice, with
`--layer-count-mode predicted` and `--layer-count-mode reference`. Keep the
split, seed, batch size, MC batch size, step count, and MC count identical, and
use different output files. A 128-target prefix at 16 steps, MC4, and batch 1
is a practical local test.

Both modes pad sampling to the same slot shape to pair the random draws.
Reference mode supplies only the number of slots from the reference stack;
materials and thicknesses are still generated from noise. Its JSON reports
`applied_layer_count` rather than `predicted_layer_count`. This is a diagnostic
using held-out labels, not an inference setting for new optical targets.

To isolate the effect of thickness, use the reference-count run as the control
and repeat it with `--reference-thickness`. Both arms start from the same
Gaussian thickness state; the new arm follows the forward-noised reference
trajectory and scores the generated materials at the exact native reference
thicknesses. No reference material labels are supplied to the sampler.

For the reverse test, use `--reference-materials` instead of
`--reference-thickness`. The same material jumps reveal reference labels;
all thicknesses still evolve from Gaussian noise using the learned velocity.
