#!/bin/bash
#SBATCH --partition=...
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:full:1
#SBATCH --time=2-00:00:00
#SBATCH --job-name=optollama-train
#SBATCH --output=train-log-%x_%j.out
#SBATCH --error=error-log-%x_%j.err

source ~/.venv/optollama/bin/activate
srun --gpu-bind=closest python -u ../scripts/inference.py --config ../configs/optollama.yaml
