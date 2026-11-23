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
