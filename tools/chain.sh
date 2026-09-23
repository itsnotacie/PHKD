#!/bin/bash
# Sequentially launch training/eval jobs on one GPU after a given PID exits.
# Usage: chain.sh <gpu_id> <wait_pid> <config1> [config2 ...]
# Each config runs to completion before the next starts. Logs go to runs/chain_<name>_gpu<gpu>.log
set -u
GPU="$1"; shift
WAIT_PID="$1"; shift
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "[chain gpu$GPU] waiting for pid $WAIT_PID to finish..."
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "[chain gpu$GPU] pid $WAIT_PID done at $(date). Starting queue."

for cfg in "$@"; do
  name=$(basename "$cfg" .yaml)
  log="runs/chain_${name}_gpu${GPU}.log"
  echo "[chain gpu$GPU] launching $name -> $log at $(date)"
  WANDB_MODE=offline CUDA_VISIBLE_DEVICES="$GPU" python -u train_geokd.py --config "$cfg" > "$log" 2>&1
  echo "[chain gpu$GPU] finished $name (exit $?) at $(date)"
done
echo "[chain gpu$GPU] queue complete at $(date)."
