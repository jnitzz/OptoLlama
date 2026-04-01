#!/bin/bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:full:4
#SBATCH --mem=501600mb
#SBATCH --time=0-12:00:00
#SBATCH --job-name=optollama-train
#SBATCH --output=train-log-%x_%j.out
#SBATCH --error=error-log-%x_%j.err

source ~/.venv/optollama/bin/activate
srun python -u ../scripts/train.py --config ../configs/optollama.yaml
