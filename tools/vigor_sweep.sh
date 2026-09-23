#!/usr/bin/env bash
# Generic VIGOR sweep launcher: sequentially run a list of configs on one GPU.
# For each config, probe batch sizes [96, 72, 56, 40, 32, 24, 16] largest-first,
# accept the largest that stays under 76 GiB after a 150s warm-up.
#
# Usage:
#   GPU=0 CONFIGS="peak_hard_r3 peak_hard_topk4 peak_hard_hn_topk64" \
#     ./tools/vigor_sweep.sh
set -u

: "${GPU:?set GPU env var}"
: "${CONFIGS:?set CONFIGS env var (space-separated stems)}"
GEOKD=/home/yiru_fang/geokd-main
LOGS=$GEOKD/runs/logs
mkdir -p "$LOGS"

STATE=/tmp/vigor_sweep_gpu${GPU}.state
: > "$STATE"

log_line() { echo "[$(date +'%F %T')][GPU$GPU] $*" | tee -a "$STATE"; }

BS_CANDIDATES=(96 72 56 40 32 24 16)

for STEM in $CONFIGS; do
  CFG=$GEOKD/dataset/config_vigor_${STEM}.yaml
  if [[ ! -f "$CFG" ]]; then
    log_line "ERROR: config missing: $CFG"; continue
  fi

  log_line "==============================================================="
  log_line "config=$CFG"

  ACCEPTED_BS=""
  ACCEPTED_PID=""
  ACCEPTED_LOG=""

  for BS in "${BS_CANDIDATES[@]}"; do
    log_line "  [probe] trying batch_size_per_gpu=$BS"
    sed -i "s/^  batch_size_per_gpu:.*/  batch_size_per_gpu: $BS/" "$CFG"
    STAMP=$(date +%Y%m%d_%H%M%S)
    PROBE_LOG=$LOGS/vigor_sweep_${STEM}_bs${BS}_gpu${GPU}_${STAMP}.log

    CUDA_VISIBLE_DEVICES=$GPU nohup "$GEOKD/.venv/bin/python" -u \
      "$GEOKD/train_geokd.py" --config "$CFG" > "$PROBE_LOG" 2>&1 &
    PID=$!
    log_line "  [probe] launched PID=$PID  log=$PROBE_LOG"

    sleep 150

    if ! ps -p "$PID" > /dev/null 2>&1; then
      if grep -qE "out of memory|OutOfMemoryError|CUDA out of memory" "$PROBE_LOG"; then
        log_line "  [probe] bs=$BS OOM -> try smaller"
        continue
      else
        log_line "  [probe] bs=$BS crashed for other reason. Last 30 log lines:"
        tail -30 "$PROBE_LOG" | tee -a "$STATE"
        log_line "  [probe] aborting config=$STEM"
        break
      fi
    fi

    MEM_USED=$(nvidia-smi --id=$GPU --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    log_line "  [probe] bs=$BS alive at 150s, GPU mem=${MEM_USED} MiB"

    if [[ "$MEM_USED" =~ ^[0-9]+$ ]] && (( MEM_USED > 76000 )); then
      log_line "  [probe] bs=$BS mem too high (>76 GiB), killing and try smaller"
      kill $PID 2>/dev/null; sleep 3; kill -9 $PID 2>/dev/null; sleep 5
      continue
    fi

    ACCEPTED_BS=$BS
    ACCEPTED_PID=$PID
    ACCEPTED_LOG=$PROBE_LOG
    log_line "  [accept] $STEM bs=$BS PID=$PID mem=${MEM_USED} MiB"
    break
  done

  if [[ -z "$ACCEPTED_BS" ]]; then
    log_line "ERROR: no batch size worked for $STEM. Moving on."
    continue
  fi

  log_line "  [run] waiting for $STEM PID=$ACCEPTED_PID to finish..."
  while ps -p "$ACCEPTED_PID" > /dev/null 2>&1; do
    sleep 600
  done
  log_line "  [done] $STEM finished."

  log_line "  [result] $STEM per-epoch val:"
  grep -E "Epoch [0-9]+/[0-9]+:" "$ACCEPTED_LOG" | tee -a "$STATE"
  BEST=$(grep -oE "mean=[0-9.]+" "$ACCEPTED_LOG" | awk -F= '{print $2}' | sort -n | head -1)
  log_line "  [result] $STEM BEST val mean = ${BEST} m"
  sleep 15
done

log_line "==============================================================="
log_line "GPU $GPU sweep complete."
