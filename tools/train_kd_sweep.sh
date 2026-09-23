#!/usr/bin/env bash
# Train 4 new distillation loss variants sequentially on a single GPU.
# Base: rand-init student (ViT-S/14 + SlimDPT w=0.5), batch 56, 10 epochs.
set -euo pipefail

REPO=/home/yiru_fang/geokd-main
PY=/home/yiru_fang/geokd-main/.venv/bin/python
LOGDIR=/tmp/track_a_kd_sweep
GPU=${GPU:-4}
mkdir -p "$LOGDIR"

declare -a RUNS=(
  "dist         config_vigor_dist.yaml"
  "mse          config_vigor_mse.yaml"
  "dkd          config_vigor_dkd.yaml"
  "peak_hard_ce config_vigor_peak_hard_ce.yaml"
)

cd "$REPO"
for entry in "${RUNS[@]}"; do
  NAME=$(echo "$entry" | awk '{print $1}')
  CFG=$(echo "$entry"  | awk '{print $2}')
  LOG=$LOGDIR/${NAME}.log
  echo "=========================================="
  echo "  $(date) starting $NAME on GPU $GPU"
  echo "  cfg: dataset/$CFG"
  echo "  log: $LOG"
  echo "=========================================="
  WANDB_MODE=offline CUDA_VISIBLE_DEVICES=$GPU "$PY" -u train_geokd.py \
    --config "dataset/$CFG" > "$LOG" 2>&1
  echo "  --> finished $NAME. Recent epochs:"
  grep -E "^Epoch [0-9]+/[0-9]+:" "$LOG" | tail -5 | sed 's/^/    /'
done

echo ""
echo "=========================================="
echo "  SUMMARY (best mean on internal val)"
echo "=========================================="
for entry in "${RUNS[@]}"; do
  NAME=$(echo "$entry" | awk '{print $1}')
  LOG=$LOGDIR/${NAME}.log
  BEST=$(grep -oE "mean=[0-9.]+" "$LOG" | awk -F= '{print $2}' | sort -n | head -1)
  printf "  %-14s best_mean=%s\n" "$NAME" "${BEST:-?}"
done
