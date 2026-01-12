#!/bin/bash
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1               # 1 GPU per task  (total 4 on the node)
#SBATCH --cpus-per-task=10          # (optional) keep 32 logical CPUs overall
#SBATCH --time=01:00:00
#SBATCH --mem=0                 # grab all the node’s RAM instead of a fixed 500 G
#SBATCH --job-name=OptoLlama
#SBATCH --account=hai_1044
#SBATCH --output=/p/project1/hai_1044/oezdemir/optollama_new/OptoLlama/logs/%x_%j.out
#SBATCH --error=/p/project1/hai_1044/oezdemir/optollama_new/OptoLlama/logs/%x_%j.err

module purge
module load Stages/2025
module load GCCcore/.13.3.0
module load Python/3.12.3
source /p/project1/hai_1044/oezdemir/optollama_new/OptoLlama/.venv/bin/activate

export MASTER_PORT=12342

# Get the first node name as master address.
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# one rank ↔ one GPU
# srun --gpu-bind=closest python -u Diffusion/scripts/optollama.py
srun --gpu-bind=closest python -u /p/project1/hai_1044/oezdemir/optollama_new/OptoLlama/scripts/variability_analysis.py \
#srun python -u /p/project1/hai_1044/oezdemir/optollama_new/OptoLlama/scripts/minimal_example.py

