#!/bin/bash
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1               # 1 GPU per task  (total 4 on the node)
#SBATCH --time=01:00:00
#SBATCH --job-name=OptoLlama
#SBATCH --account=hai_1044
#SBATCH --output=/p/project1/hai_1044/oezdemir/optollama_new/OptoLlama/logs/%x_%j.out
#SBATCH --error=/p/project1/hai_1044/oezdemir/optollama_new/OptoLlama/logs/%x_%j.err

module purge
module load Stages/2025
module load GCCcore/.13.3.0
module load Python/3.12.3
source /p/project1/hai_1044/oezdemir/optollama_new/OptoLlama/newvenv/bin/activate

export MASTER_PORT=12342

# Get the first node name as master address.
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

srun --gpu-bind=closest python -u /p/project1/hai_1044/oezdemir/optollama_new/OptoLlama/scripts/bragg_stack_finder_test_data.py

