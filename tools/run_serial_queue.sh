#!/usr/bin/env bash
# Serial queue runner: wait for a PID to exit, then run configs sequentially on one GPU.
# Usage: run_serial_queue.sh <GPU> <WAIT_PID> <QUEUE_TAG> <cfg1> [cfg2 ...]
set -uo pipefail

GPU="$1"
WAIT_PID="$2"
QUEUE_TAG="$3"
shift 3

REPO=/home/yiru_fang/geokd-main
PY=$REPO/.venv/bin/python
LOGDIR=$REPO/runs/logs
STATE=/tmp/${QUEUE_TAG}.state
mkdir -p "$LOGDIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')][$QUEUE_TAG][GPU$GPU] $*" | tee -a "$STATE"; }

log "queue start; waiting for PID=$WAIT_PID to exit; queue=($*)"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
log "PID=$WAIT_PID exited; starting queue"

for cfg in "$@"; do
  name=$(basename "$cfg" .yaml | sed 's/^config_//')
  tag="serial_${QUEUE_TAG}_${name}"
  ts=$(date +%Y%m%d_%H%M%S)
  child_log=$LOGDIR/${tag}_gpu${GPU}_${ts}.log
  log "==============================================="
  log "starting cfg=$cfg  log=$child_log"
  CUDA_VISIBLE_DEVICES=$GPU $PY $REPO/train_geokd.py --config "$REPO/$cfg" > "$child_log" 2>&1
  rc=$?
  log "cfg=$cfg finished (rc=$rc)"
  best=$(grep -E "^Epoch [0-9]+/10:" "$child_log" | awk -F 'mean=' '{print $2}' | awk -F ',' '{print $1}' | sort -g | head -1)
  log "  best val mean = ${best:-N/A}"
done

log "queue complete."
