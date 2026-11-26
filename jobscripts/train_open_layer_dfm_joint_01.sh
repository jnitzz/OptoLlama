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
    if (( $# < 2 || $# > 3 )); then
        echo "Usage: sbatch $0 [FORK_DIR {control|wrong-label|material-recovery} [END_EPOCH|+N]]" >&2
        exit 1
    fi
    case "$2" in
        control) peak=0; visible_weight=1 ;;
        wrong-label) peak=0.10; visible_weight=1 ;;
        material-recovery) peak=0.10; visible_weight="${CORRECT_VISIBLE_MATERIAL_WEIGHT:-0.25}" ;;
        *) echo "Unknown fork arm: $2" >&2; exit 1 ;;
    esac
    source_checkpoint="$1/source.pt"
    OUT_DIR="$1/$2"
    if [[ ! -s "$source_checkpoint" ]]; then
        echo "Missing frozen fork source: $source_checkpoint" >&2
        exit 1
    fi
    selection="$("$VENV_PY" -c '
import sys, torch
from pathlib import Path

def position(path, *, progress=False, peak=None, visible_weight=None):
    p = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    extra = p.get("extra") or {}
    cursor = extra.get("resume_cursor")
    if (not isinstance(p.get("epoch"), int) or not p.get("optimizer_state")
            or (cursor is not None) != progress or extra.get("denoising_process") != "joint_layer_dfm"):
        raise SystemExit(f"Invalid resumable joint layer DFM checkpoint: {path}")
    if peak is not None and float(extra.get("visible_material_replace_peak", 0.0)) != peak:
        raise SystemExit(f"Checkpoint replacement probability differs from the fork arm: {path}")
    if visible_weight is not None and float(extra.get("correct_visible_material_weight", 1.0)) != visible_weight:
        raise SystemExit(f"Checkpoint correctly visible material weight differs from the fork arm: {path}")
    # A completed save wins over progress from the same zero-based epoch.
    return (p["epoch"], not progress, int(extra.get("global_optimizer_steps", 0)))

source, out = map(Path, sys.argv[1:3])
peak, visible_weight = map(float, sys.argv[3:5])
if not 0 <= visible_weight <= 1:
    raise SystemExit("CORRECT_VISIBLE_MATERIAL_WEIGHT must be between zero and one")
source_position = position(source)
end = sys.argv[5] or "+1"
number = end.removeprefix("+")
if not number.isascii() or not number.isdecimal() or int(number) < 1:
    raise SystemExit("END_EPOCH must be a positive integer or +N")
end_epoch = int(number) + (source_position[0] + 1 if end.startswith("+") else 0)
latest, resume = source_position, source
for kind in ("last", "progress"):
    path = out / f"open-layer-dfm-joint-{kind}.pt"
    if path.is_file() and path.stat().st_size:
        key = position(path, progress=kind == "progress", peak=peak, visible_weight=visible_weight)
        if key > latest:
            latest, resume = key, path
if latest[0] + int(latest[1]) >= end_epoch:
    print(end_epoch)
else:
    print(end_epoch, resume, sep="\t")
' "$source_checkpoint" "$OUT_DIR" "$peak" "$visible_weight" "${3:-}")"
    IFS=$'\t' read -r end_epoch RESUME_CHECKPOINT <<< "$selection"
    if [[ -z "$RESUME_CHECKPOINT" ]]; then
        echo "Fork arm $2 already reached epoch $end_epoch: $OUT_DIR"
        exit 0
    fi
    echo "Fork arm=$2 peak=$peak correct_visible_weight=$visible_weight through epoch=$end_epoch resume=$RESUME_CHECKPOINT"
    train_args+=(--output-dir "$OUT_DIR" --epochs "$end_epoch" --visible-material-replace-peak "$peak"
                 --correct-visible-material-weight "$visible_weight")
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
