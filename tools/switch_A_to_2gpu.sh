#!/bin/bash
# When D finishes, switch A from single-GPU to 2-GPU DDP (GPU 0 + 2).
# Fallback to single-GPU A on GPU0 if the 2-GPU DDP run fails within 5 min.
# Usage: switch_A_to_2gpu.sh <D_pid> <singleGPU_A_pid>
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export TMPDIR=/home/yiru_fang/torchtmp
D_PID="$1"; A_PID="$2"

echo "[switch] waiting for D pid $D_PID to finish..."
while kill -0 "$D_PID" 2>/dev/null; do sleep 60; done
echo "[switch] D done at $(date). Stopping single-GPU A ($A_PID)."
kill -9 "$A_PID" 2>/dev/null
sleep 15

echo "[switch] launching 2-GPU DDP A on GPU 0,2 at $(date)"
start=$(date +%s)
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,2 torchrun --nproc_per_node=2 --master_port=29533 \
  train_geokd.py --config dataset/config_vigor_supervised_ddp.yaml > runs/vigor_A_supervised_2gpu.log 2>&1
code=$?
elapsed=$(( $(date +%s) - start ))
echo "[switch] 2-GPU A exited code=$code after ${elapsed}s at $(date)"

if [ "$code" -ne 0 ] && [ "$elapsed" -lt 300 ]; then
  echo "[switch] 2-GPU DDP failed fast; falling back to single-GPU A on GPU0"
  WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 python -u train_geokd.py \
    --config dataset/config_vigor_supervised.yaml > runs/vigor_A_supervised_gpu0_fallback.log 2>&1
  echo "[switch] fallback single-GPU A exited code=$? at $(date)"
fi
