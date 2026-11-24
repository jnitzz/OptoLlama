#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --constraint=H200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=2-23:59:00
#SBATCH --job-name=open-layer-dfm-joint-01
#SBATCH --output=./logs/log-%x_%j.out
#SBATCH --error=./logs/log-%x_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:?Submit from the OptoLlama repository root}"
VENV_PY="${VENV_PY:-$HOME/.venv/optollama/bin/python}"
CONFIG="$ROOT/configs/open_layer_dfm_joint_01.yaml"
export OPTOLLAMA_DATA_PATH="${OPTOLLAMA_DATA_PATH:-$ROOT/data}"
OUT_DIR="$OPTOLLAMA_DATA_PATH/output_OPEN_LAYER_DFM_JOINT_01"
cd "$ROOT"

train_args=(--config "$CONFIG")
if (( $# > 0 )); then
    if (( $# != 2 )); then
        echo "Usage: sbatch $0 [FORK_DIR {control|wrong-label}]" >&2
        exit 1
    fi
    case "$2" in
        control) peak=0 ;;
        wrong-label) peak=0.10 ;;
        *) echo "Unknown fork arm: $2" >&2; exit 1 ;;
    esac
    source_checkpoint="$1/source.pt"
    OUT_DIR="$1/$2"
    if [[ ! -s "$source_checkpoint" ]]; then
        echo "Missing frozen fork source: $source_checkpoint" >&2
        exit 1
    fi
    if [[ -s "$OUT_DIR/open-layer-dfm-joint-last.pt" ]]; then
        echo "Fork arm $2 already completed: $OUT_DIR"
        exit 0
    fi
    end_epoch="$("$VENV_PY" -c '
import sys, torch
p = torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=False)
extra = p.get("extra") or {}
if (not isinstance(p.get("epoch"), int) or not p.get("optimizer_state")
        or extra.get("resume_cursor") is not None or extra.get("denoising_process") != "joint_layer_dfm"):
    raise SystemExit("source.pt must be a completed-epoch joint layer DFM checkpoint with optimizer state")
print(p["epoch"] + 2)
' "$source_checkpoint")"
    RESUME_CHECKPOINT="$source_checkpoint"
    if [[ -s "$OUT_DIR/open-layer-dfm-joint-progress.pt" ]]; then
        RESUME_CHECKPOINT="$OUT_DIR/open-layer-dfm-joint-progress.pt"
    fi
    echo "Fork arm=$2 peak=$peak through epoch=$end_epoch resume=$RESUME_CHECKPOINT"
    train_args+=(--output-dir "$OUT_DIR" --epochs "$end_epoch" --visible-material-replace-peak "$peak")
fi

resume_args=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    if [[ ! -s "$RESUME_CHECKPOINT" ]]; then
        echo "Resume checkpoint not found or empty: $RESUME_CHECKPOINT" >&2
        exit 1
    fi
    resume_args=(--resume "$RESUME_CHECKPOINT")
elif [[ -e "$OUT_DIR/open-layer-dfm-joint-progress.pt" || -e "$OUT_DIR/open-layer-dfm-joint-last.pt" || -e "$OUT_DIR/open-layer-dfm-joint-best.pt" ]]; then
    echo "Run 01 already has checkpoints. Set RESUME_CHECKPOINT explicitly." >&2
    exit 1
fi

unset RDZV_ENDPOINT RDZV_ID RDZV_BACKEND
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="$((29500 + SLURM_JOB_ID % 1000))"
export OPTOLLAMA_DDP_TIMEOUT_MINUTES=20
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT SLURM_NTASKS=$SLURM_NTASKS"

nvidia-smi --loop=30 --format=csv,noheader,nounits \
    --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total \
    >> "./logs/log-${SLURM_JOB_NAME}_${SLURM_JOB_ID}.smi.log" &
telemetry_pid=$!
trap 'kill "$telemetry_pid" 2>/dev/null || true' EXIT

srun --export=ALL --cpu-bind=none --kill-on-bad-exit=1 \
    "$VENV_PY" -u "$ROOT/scripts/train_open_vocab_depth_field.py" \
    "${train_args[@]}" "${resume_args[@]}"
