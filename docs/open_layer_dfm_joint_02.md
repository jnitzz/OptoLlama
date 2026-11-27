# Joint DFM Run 02: high-noise exposure

Run 02 starts from scratch on the same four shards (20M examples). It uses a
14-block stack transformer at width 512 and eight heads, instead of 12 blocks.
All loss weights remain 1.0; visible-material replacement peaks at 0.10 from
epoch 1. The 50-epoch cosine LR schedule and sampler are unchanged.

For each training example, flow time is uniform on [0, 1) with probability
0.80 and uniform on [0, 0.35) otherwise. With the unchanged mask probability
`(1-s)^2`, this raises expected masking from 33.3% to 40.5%. Held-out denoising
validation still uses uniform time. Run 02 is a combined experiment: it changes
time sampling and depth, and uses wrong labels from the start rather than
introducing them after pretraining. To isolate time sampling, compare
with a fresh 14-block run with the same replacement peak and zero high-noise
fraction in another output directory.

## Slurm

From the repository root, launch a new run on one node with four H200s:

```bash
mkdir -p logs
env -u RESUME_CHECKPOINT OPTOLLAMA_JOINT_RUN=02 \
  sbatch --job-name=joint-high-noise-02 jobscripts/train_open_layer_dfm_joint_01.sh
```

The launcher writes to `data/output_OPEN_LAYER_DFM_JOINT_02` by default and
refuses an implicit restart if checkpoints exist. To resume after the three-day
limit, pick the newest checkpoint in that directory and set its path explicitly:

```bash
out="${OPTOLLAMA_DATA_PATH:-$PWD/data}/output_OPEN_LAYER_DFM_JOINT_02"
ls -lt "$out"/open-layer-dfm-joint-{last,progress}.pt
OPTOLLAMA_JOINT_RUN=02 RESUME_CHECKPOINT="$out/open-layer-dfm-joint-progress.pt" \
  sbatch --job-name=joint-high-noise-02 jobscripts/train_open_layer_dfm_joint_01.sh
```

Substitute `last.pt` when that file is newer than the mid-epoch save. Do not
start a second job against the same output directory while one is active.

## Optical checkpoints

The trainer retains immutable E02, E04, E06 and E08 `.pt` files. Exact TMM is
not part of training. Score each retained checkpoint on the same 512 targets
used for Run 01:

```bash
out="${OPTOLLAMA_DATA_PATH:-$PWD/data}/output_OPEN_LAYER_DFM_JOINT_02"
epoch=0002  # Also test 0004, 0006 and 0008.
srun -p gpu --nodes=1 --ntasks=1 --gres=gpu:1 --mem=0 --cpu-bind=none --gpu-bind=closest \
  "$HOME/.venv/optollama/bin/python" -u scripts/inference_open_vocab_depth_field.py \
  --config configs/open_layer_dfm_joint_02.yaml \
  --checkpoint "$out/open-layer-dfm-joint-epoch-$epoch.pt" --weights ema \
  --split test --seed 4 --batch-size 8 --layer-count-mode predicted \
  --query-mode full --full-grid-metrics --record-spectra --record-fields --device cuda \
  --save "$out/rat-test-e$epoch-512-mc04-steps-16.json" \
  --max-samples 512 --steps 16 --mc-samples 4 --mc-batch-size 4
```
