#!/bin/bash
# Wait for a pid to exit, then run a CCVPE distillation experiment on a given GPU.
# Usage: watch_then_run.sh <wait_pid> <gpu> <distill_mode> <name> <batch_size>
set -u
WP="$1"; GPU="$2"; MODE="$3"; NAME="$4"; BS="$5"
cd /home/yiru_fang/geokd-main
source .venv/bin/activate
export TMPDIR=/home/yiru_fang/torchtmp
echo "[watch] waiting for pid $WP before running $NAME on gpu $GPU..."
while kill -0 "$WP" 2>/dev/null; do sleep 60; done
echo "[watch] pid $WP done at $(date). Launching $NAME (mode=$MODE bs=$BS) on gpu $GPU"
CUDA_VISIBLE_DEVICES="$GPU" python -u train_ccvpe_distill.py \
  --config dataset/config_vigor_kd.yaml --distill_mode "$MODE" --batch_size "$BS" --name "$NAME" \
  > "runs/${NAME}.log" 2>&1
echo "[watch] $NAME finished (exit $?) at $(date)"
