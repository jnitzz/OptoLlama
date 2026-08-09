# Depth-Field Self-Training

This workflow mines alternative depth fields from a trained inverse model, verifies every retained candidate with exact TMM, and replays accepted alternatives during ordinary depth-field training.

## Safety Boundary

- Mine anchor spectra only from the training split.
- Never add validation or external benchmark targets to the solution bank.
- Exact TMM determines Gold/Silver acceptance. The frozen spectrum surrogate is only an auxiliary training gradient.
- Keep the original dataset in every self-training cycle to prevent model-only feedback drift.

## 1. Freeze the Run 21 Source

The Run 24 config expects the best Run 21 EMA checkpoint under an immutable name:

```bash
cp -p data/output_DF_HYBRID_SPECFT_21/depth-field-best.pt \
  data/output_DF_HYBRID_SPECFT_21/depth-field-best-70M-frozen.pt
```

Confirm that `depth-field-best.pt` still corresponds to the 70M validation minimum before copying it.

## 2. Mine a Pilot Bank

The checked-in config mines 1,000 training anchors with 16 candidates each. Under an allocation with one task per GPU:

```bash
srun --gpu-bind=closest python -u scripts/mine_depth_field_solutions.py \
  --config configs/depth_field_hybrid_24_selftrain.yaml
```

Each DDP rank receives a disjoint contiguous anchor range and writes independent safetensor shards. Rank 0 writes the combined `manifest.json` after every rank reaches the final barrier.

The default bank directory is:

```text
data/solution_banks/df_hybrid_21_cycle_01/
```

Inspect `manifest.json` before training. It reports processed anchors, generated candidates, and Gold/Silver acceptance counts. A zero or very small Gold count should be investigated before increasing the mining scale.

Useful pilot overrides:

```bash
python scripts/mine_depth_field_solutions.py \
  --config configs/depth_field_hybrid_24_selftrain.yaml \
  --max-targets 100 \
  --mc-samples 4 \
  --mc-batch-size 2
```

For the first production bank, change `SELF_TRAIN.MINING.MAX_TARGETS` to `10000` or `25000`. Keep the acceptance thresholds fixed so pilot and production acceptance rates remain comparable.

## 3. Run the Replay Fine-Tune

After the bank contains accepted candidates:

```bash
srun --gpu-bind=closest python -u scripts/train_depth_field_diffusion.py \
  --config configs/depth_field_hybrid_24_selftrain.yaml
```

Run 24 starts from the frozen Run 21 model with a fresh optimizer. It trains for 20M global samples with:

- 70% original training examples;
- 20% Gold alternatives;
- 10% Silver alternatives;
- spectral auxiliary weight `0.10`;
- validation every 5M global samples.

The progress line includes `bank=30.0%`. History and checkpoints record the realized replay count and complete solution-bank provenance.

## 4. Promote or Reject the Cycle

Compare Run 24 to the frozen Run 21 benchmark using untouched validation targets. Promotion requires improvement in all-MC exact TMM quality or success rate, not only a better selected lucky sample. Also inspect successful topology count, run count, and active thickness.

If Run 24 is promoted, mine cycle 2 from its frozen best checkpoint into a new directory. Advance `ANCHOR_OFFSET` to avoid repeating cycle-1 anchors, keep prior accepted banks immutable, and list both bank directories under `DEPTH_FIELD.TRAIN.SOLUTION_BANK.PATHS` for cumulative replay.

Do not overwrite a prior bank or point two concurrent miners at the same output directory.
