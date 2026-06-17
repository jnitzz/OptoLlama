Archived experiment entry points.

These scripts are kept for reference but are not part of the active depth-field
training/inference path on this branch:

- `train_ram_reward.py`: RAM-style online TMM reward training for the original
  token/factored OptoLlama model.
- `self_improve_lite.py`: local TMM-ranked self-improvement prototype.
- `self_improve_tmm_ranked.py`: TMM-ranked self-improvement prototype.

Shared helper modules remain in `optollama/` when they are still used by kept
dataset-generation or target-processing code.
