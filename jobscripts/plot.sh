#!/bin/bash
#SBATCH --partition=...
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-00:30:00
#SBATCH --job-name=optollama-plot
#SBATCH --output=plot-log-%x_%j.out
#SBATCH --error=error-log-%x_%j.err

source ~/.venv/optollama/bin/activate

# Default dashboard plot from the latest inference outputs
python -u ../scripts/plot_results.py dashboard --config ../configs/optollama.yaml

# Example sample plot:
# python -u ../scripts/plot_results.py sample --config ../configs/optollama.yaml --index 0
