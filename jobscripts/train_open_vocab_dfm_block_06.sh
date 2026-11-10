#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=2-23:59:00
#SBATCH --job-name=dfm-block-06
#SBATCH --output=./logs/log-%x_%j.out
#SBATCH --error=./logs/log-%x_%j.err

set -euo pipefail

ROOT=/scratch/htc/jschaibl/repos/OptoLlama
VENV_PY=/home/htc/jschaibl/.venv/optollama/bin/python
CONFIG="$ROOT/configs/depth_field_open_vocab_dfm_block_06.yaml"
CHECKPOINT="$ROOT/data/output_DF_OPEN_VOCAB_DFM_BLOCK_06/open-vocab-depth-last.pt"

cd "$ROOT"
source /home/htc/jschaibl/.venv/optollama/bin/activate
unset RDZV_ENDPOINT RDZV_ID RDZV_BACKEND

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="$((29500 + SLURM_JOB_ID % 1000))"
export OPTOLLAMA_DDP_TIMEOUT_MINUTES=20
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT SLURM_NTASKS=$SLURM_NTASKS"
"$VENV_PY" -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.is_bf16_supported())'

nvidia-smi --loop=30 \
    --format=csv,noheader,nounits \
    --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total \
    >> "./logs/log-${SLURM_JOB_NAME}_${SLURM_JOB_ID}.smi.log" &
telemetry_pid=$!
trap 'kill "$telemetry_pid" 2>/dev/null || true' EXIT

resume_args=()
if [[ -f "$CHECKPOINT" ]]; then
    resume_args=(--resume "$CHECKPOINT")
fi

srun --export=ALL --cpu-bind=none --kill-on-bad-exit=1 \
    "$VENV_PY" -u "$ROOT/scripts/train_open_vocab_depth_field.py" \
    --config "$CONFIG" "${resume_args[@]}"
