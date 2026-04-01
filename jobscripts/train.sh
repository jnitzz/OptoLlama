#!/bin/bash
#SBATCH --partition=...
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=2-00:00:00
#SBATCH --job-name=optollama-train
#SBATCH --output=train-log-%x_%j.out
#SBATCH --error=error-log-%x_%j.err

srun --gpu-bind=closest python -u ../scripts/train.py --config ../configs/optollama.yaml
