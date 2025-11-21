#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4               # 1 GPU per task  (total 4 on the node)
#SBATCH --cpus-per-task=16          # (optional) keep 32 logical CPUs overall
#SBATCH --time=30-00:00:00
#SBATCH --mem=0                 # grab all the node’s RAM instead of a fixed 500 G
#SBATCH --job-name=MD49
#SBATCH --output=/scratch/htc/jschaibl/repos/ColorAppearanceToolbox/logs/%x_%j.out
#SBATCH --error=/scratch/htc/jschaibl/repos/ColorAppearanceToolbox/logs/%x_%j.err

# eval "$(mamba shell.bash hook)"
# source /scratch/htc/jschaibl/miniforge3/envs/GPT/bin/activate

(
  while true
  do
    # Capture date + nvidia-smi output
    NEW_LOG="$(date)\n$(nvidia-smi)\n----------------------------"

    # Prepend it to the beginning of existing log
    # 1. Echo the new log
    # 2. Cat old log (if it exists)
    # 3. Write out to a tmp file, then overwrite the original
    if [ -f /scratch/htc/jschaibl/repos/ColorAppearanceToolbox/logs/${SLURM_JOB_NAME}_$SLURM_JOB_ID.log ]; then
      (echo -e "$NEW_LOG"; cat /scratch/htc/jschaibl/repos/ColorAppearanceToolbox/logs/${SLURM_JOB_NAME}_$SLURM_JOB_ID.log) > /scratch/htc/jschaibl/repos/ColorAppearanceToolbox/logs/${SLURM_JOB_NAME}_$SLURM_JOB_ID.tmp
    else
      echo -e "$NEW_LOG" > /scratch/htc/jschaibl/repos/ColorAppearanceToolbox/logs/${SLURM_JOB_NAME}_$SLURM_JOB_ID.tmp
    fi
    mv /scratch/htc/jschaibl/repos/ColorAppearanceToolbox/logs/${SLURM_JOB_NAME}_$SLURM_JOB_ID.tmp /scratch/htc/jschaibl/repos/ColorAppearanceToolbox/logs/${SLURM_JOB_NAME}_$SLURM_JOB_ID.gpu.log

    # Sleep a bit before the next sample
    sleep 29
  done
) &

export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=120  # Increased timeout

# ✅ 🔍 Force TCP Communication if InfiniBand is Unstable
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0  # Adjust based on the actual network interface
export NCCL_P2P_DISABLE=1

# one rank ↔ one GPU
# srun --gpu-bind=closest python -u Diffusion/scripts/optollama.py
srun --gpu-bind=closest python -u analyze_inference_MC.py \
  --config config_MD49 \
  --checkpoint auto \
  --batch 128 \
  --tau -1 \
  --limit 10 \
  --mc_samples 3 \
  --sample_temperature 1 \
  --top_k 5 \
  --top_p 0.0
