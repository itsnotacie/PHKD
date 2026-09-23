#!/usr/bin/env bash
# Auto-launcher for KITTI KD students.
# Waits for the teacher run to finish, then trains three students sequentially.
# For each student, it probes batch sizes (larger -> smaller) and picks the largest
# that fits under 76 GiB on the target GPU. Restarts training with that batch size.
set -u

GPU=${GPU:-3}
GEOKD=/home/yiru_fang/geokd-main
LOGS=$GEOKD/runs/logs
mkdir -p "$LOGS"

STATE=/tmp/kitti_students.state
: > "$STATE"

TEACHER_PID_FILE=/tmp/kitti_teacher.pid
TEACHER_LOG_FILE=/tmp/kitti_teacher.log_path
TEACHER_CKPT=$GEOKD/checkpoints/kitti/geokd/student/kitti-teacher-vitl14_best.pth

log_line() {
  echo "[$(date +'%F %T')] $*" | tee -a "$STATE"
}

# ---------- 1) Wait for teacher to finish ----------
if [[ ! -f "$TEACHER_PID_FILE" ]]; then
  log_line "ERROR: teacher pid file missing: $TEACHER_PID_FILE"
  exit 1
fi
TEACHER_PID=$(cat "$TEACHER_PID_FILE")
log_line "Waiting for teacher PID=$TEACHER_PID to finish..."
while ps -p "$TEACHER_PID" > /dev/null 2>&1; do
  sleep 300
done
log_line "Teacher PID=$TEACHER_PID has exited."

sleep 15  # let file system flush

if [[ ! -f "$TEACHER_CKPT" ]]; then
  log_line "ERROR: teacher checkpoint missing after training: $TEACHER_CKPT"
  ls -lh "$GEOKD/checkpoints/kitti/geokd/student/" | tee -a "$STATE"
  exit 1
fi
SIZE=$(du -h "$TEACHER_CKPT" | cut -f1)
log_line "Teacher checkpoint OK: $TEACHER_CKPT ($SIZE)"

# ---------- 2) Print teacher final val (last few epochs) ----------
TLOG=$(cat "$TEACHER_LOG_FILE" 2>/dev/null || echo "")
if [[ -f "$TLOG" ]]; then
  log_line "Teacher final val epochs:"
  grep -E "Epoch [0-9]+/10:" "$TLOG" | tee -a "$STATE"
fi

# ---------- 3) For each student mode, adaptive-bs launch ----------
MODES=(peak_hard peak_hard_ce peak_hard_zscore)
# Ordered from largest to smallest; first one that fits <=76 GiB wins.
BS_CANDIDATES=(56 48 40 32 24 16)

for MODE in "${MODES[@]}"; do
  CFG=$GEOKD/dataset/config_kitti_${MODE}.yaml
  if [[ ! -f "$CFG" ]]; then
    log_line "ERROR: config missing: $CFG"; continue
  fi

  log_line "==============================================================="
  log_line "student mode=$MODE  config=$CFG"

  ACCEPTED_BS=""
  ACCEPTED_PID=""
  ACCEPTED_LOG=""

  for BS in "${BS_CANDIDATES[@]}"; do
    log_line "  [probe] trying batch_size_per_gpu=$BS"
    # patch config
    sed -i "s/^  batch_size_per_gpu:.*/  batch_size_per_gpu: $BS/" "$CFG"
    STAMP=$(date +%Y%m%d_%H%M%S)
    PROBE_LOG=$LOGS/kitti_student_${MODE}_bs${BS}_${STAMP}.log

    CUDA_VISIBLE_DEVICES=$GPU nohup "$GEOKD/.venv/bin/python" -u \
      "$GEOKD/train_geokd.py" --config "$CFG" > "$PROBE_LOG" 2>&1 &
    PID=$!
    log_line "  [probe] launched PID=$PID  log=$PROBE_LOG"

    # Warmup window: 150s should let it load DINO, do teacher forward, one train iter
    sleep 150

    if ! ps -p "$PID" > /dev/null 2>&1; then
      # crashed within 150s
      if grep -qE "out of memory|OutOfMemoryError|CUDA out of memory" "$PROBE_LOG"; then
        log_line "  [probe] bs=$BS OOM -> try smaller"
        continue
      else
        log_line "  [probe] bs=$BS crashed for other reason. Last 30 log lines:"
        tail -30 "$PROBE_LOG" | tee -a "$STATE"
        log_line "  [probe] aborting mode=$MODE"
        break
      fi
    fi

    # Alive; check GPU mem
    MEM_USED=$(nvidia-smi --id=$GPU --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    log_line "  [probe] bs=$BS alive at 150s, GPU mem=${MEM_USED} MiB"

    if [[ "$MEM_USED" =~ ^[0-9]+$ ]] && (( MEM_USED > 76000 )); then
      log_line "  [probe] bs=$BS mem too high (>76 GiB), killing and try smaller"
      kill $PID 2>/dev/null; sleep 3; kill -9 $PID 2>/dev/null; sleep 5
      continue
    fi

    # Accept
    ACCEPTED_BS=$BS
    ACCEPTED_PID=$PID
    ACCEPTED_LOG=$PROBE_LOG
    log_line "  [accept] mode=$MODE bs=$BS PID=$PID mem=${MEM_USED} MiB"
    break
  done

  if [[ -z "$ACCEPTED_BS" ]]; then
    log_line "ERROR: no batch size worked for mode=$MODE. Moving on."
    continue
  fi

  # Wait for full training to complete
  log_line "  [run] waiting for mode=$MODE PID=$ACCEPTED_PID to finish..."
  while ps -p "$ACCEPTED_PID" > /dev/null 2>&1; do
    sleep 600
  done
  log_line "  [done] mode=$MODE finished."

  # Report final val
  log_line "  [result] mode=$MODE per-epoch val:"
  grep -E "Epoch [0-9]+/[0-9]+:" "$ACCEPTED_LOG" | tee -a "$STATE"
  BEST=$(grep -oE "mean=[0-9.]+" "$ACCEPTED_LOG" | awk -F= '{print $2}' | sort -n | head -1)
  log_line "  [result] mode=$MODE BEST val mean = ${BEST} m"
  sleep 15
done

log_line "==============================================================="
log_line "All KITTI students done."
log_line "See $STATE for full summary."
