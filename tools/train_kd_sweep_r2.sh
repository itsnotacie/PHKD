#!/usr/bin/env bash
# Round-2 KD sweep: 3 new SOTA-inspired losses.
#   peak_hard_zscore : peak_hard + Sun et al. CVPR 2024 logit Z-score standardization
#   ce_zscore        : plain Hinton CE + Z-score standardization (sanity control)
#   wavelet          : Haar DWT LL/{LH,HL,HH} + CE (simplified DS²D², TGRS 2025)
#
# Waits for GPU 4 to be idle before starting each run, so it is safe to launch
# alongside a still-running round-1 sweep — it will wait automatically.
set -euo pipefail

REPO=/home/yiru_fang/geokd-main
PY=/home/yiru_fang/geokd-main/.venv/bin/python
LOGDIR=/tmp/track_a_kd_sweep_r2
GPU=${GPU:-4}
mkdir -p "$LOGDIR"

declare -a RUNS=(
  "peak_hard_zscore config_vigor_peak_hard_zscore.yaml"
  "ce_zscore        config_vigor_ce_zscore.yaml"
  "wavelet          config_vigor_wavelet.yaml"
)

wait_for_gpu() {
  local gpu=$1
  while true; do
    local util
    util=$(nvidia-smi --id="$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    local mem
    mem=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
    if [[ "$util" -lt 5 && "$mem" -lt 500 ]]; then
      echo "  GPU $gpu is idle (util=$util%, mem=${mem}MB)"
      return
    fi
    sleep 60
  done
}

cd "$REPO"
for entry in "${RUNS[@]}"; do
  NAME=$(echo "$entry" | awk '{print $1}')
  CFG=$(echo "$entry"  | awk '{print $2}')
  LOG=$LOGDIR/${NAME}.log
  echo "=========================================="
  echo "  $(date) — waiting for GPU $GPU to free up..."
  wait_for_gpu "$GPU"
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
echo "  ROUND-2 SUMMARY (best val mean, 10 ep)"
echo "=========================================="
for entry in "${RUNS[@]}"; do
  NAME=$(echo "$entry" | awk '{print $1}')
  LOG=$LOGDIR/${NAME}.log
  BEST=$(grep -oE "mean=[0-9.]+" "$LOG" 2>/dev/null | awk -F= '{print $2}' | sort -n | head -1)
  printf "  %-20s best_mean=%s\n" "$NAME" "${BEST:-?}"
done
