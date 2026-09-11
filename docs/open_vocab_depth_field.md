# Open-vocabulary depth field

This experimental path keeps the 5 nm fixed-depth representation while replacing the global material classifier with a per-sample material bank. Each candidate is represented by its interpolated `n(lambda), k(lambda)` curve. Candidate order can therefore change without changing the physical meaning of a prediction, and a new material can be offered at inference without adding a learned vocabulary row.

## Time and sampling

Training draws continuous `t ~ Uniform(0,1)` and applies the depth-field noise law `p(t) = t^2`. `OPEN_VOCAB_DEPTH_FIELD.EVAL.SAMPLING_STEPS` only controls inference discretization and is not checkpoint metadata. The initial default is 64 rather than the fixed depth model's 200 steps.

Use the same checkpoint to measure the speed/quality tradeoff:

```powershell
python scripts/inference_open_vocab_depth_field.py `
  --config configs/depth_field_open_vocab_01.yaml `
  --checkpoint data/output_DF_OPEN_VOCAB_01/open-vocab-depth-best.pt `
  --steps 32
```

Repeat with `--steps 64`, `--steps 100`, and `--steps 200`. Thin layers and material boundaries are expected to degrade before broad spectral features as the step count is reduced.

## Training

```bash
srun --cpu-bind=none --export=ALL --kill-on-bad-exit=1 \
  python -u scripts/train_open_vocab_depth_field.py \
  --config configs/depth_field_open_vocab_01.yaml
```

Run the complete data/checkpoint path with a small model first:

```bash
python scripts/train_open_vocab_depth_field.py \
  --config configs/depth_field_open_vocab_01.yaml \
  --device cpu \
  --smoke-test
```

The full checkpoint stores raw and EMA states. Inference uses EMA by default; pass `--weights raw` for comparison.

## Conditions and new materials

The current data configuration supplies `0` degrees and `s` polarization to every sample. The model interface accepts per-sample angle and polarization tensors, so future datasets can vary them without an architecture change. At normal incidence the optical condition is exactly zero and `s`/`p` are intentionally identical.

Use a different material directory and bank at inference:

```bash
python scripts/inference_open_vocab_depth_field.py \
  --config configs/depth_field_open_vocab_01.yaml \
  --checkpoint data/output_DF_OPEN_VOCAB_01/open-vocab-depth-best.pt \
  --materials-dir data/materials_extended \
  --candidate-materials Ag,SiO2,NewMaterial \
  --angle-deg 30 \
  --polarization p
```

Generation supports unseen candidate curves directly. Exact TMM scoring currently requires each selected name to exist in the token/TMM catalog; otherwise inference still saves generated material runs and reports that scoring was skipped.
