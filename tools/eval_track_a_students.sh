#!/usr/bin/env bash
# Evaluate all Track A student checkpoints on VIGOR cross-area test set.
# Runs sequentially on GPU 5 (GPU 3 is busy with DINOv3 stage1 eval).
set -euo pipefail

REPO=/home/yiru_fang/geokd-main
PY=/home/yiru_fang/geokd-main/.venv/bin/python
TEMPLATE=$REPO/dataset/config_vigor_student_eval_template.yaml
LOGDIR=/tmp/track_a_eval
GPU=${GPU:-5}
mkdir -p "$LOGDIR"

declare -a RUNS=(
  "A-supervised|$REPO/checkpoints/vigor/geokd/student/vigor-cross-supervised-baseline_best.pth"
  "C-randinit-peak_hard|$REPO/checkpoints/vigor/geokd/student/vigor-cross-C-randinit_best.pth"
  "D-teacherinit-peak_hard|$REPO/checkpoints/vigor/geokd/student/vigor-cross-peak-hard_best.pth"
  "freq-SDKD|$REPO/checkpoints/vigor/geokd/student/vigor-cross-freq_best.pth"
  "boundary-GaitKD|$REPO/checkpoints/vigor/geokd/student/vigor-cross-boundary_best.pth"
)

cd "$REPO"
for entry in "${RUNS[@]}"; do
  NAME="${entry%%|*}"
  CKPT="${entry##*|}"
  if [ ! -f "$CKPT" ]; then
    echo "SKIP: $NAME (missing $CKPT)"
    continue
  fi
  CFG=$LOGDIR/config_$NAME.yaml
  LOG=$LOGDIR/${NAME}.log
  sed -e "s#__STUDENT_CKPT__#$CKPT#" -e "s#__NAME__#eval-$NAME#" "$TEMPLATE" > "$CFG"
  echo "=== $NAME -> $LOG ==="
  WANDB_MODE=offline CUDA_VISIBLE_DEVICES=$GPU "$PY" -u train_geokd.py --config "$CFG" > "$LOG" 2>&1
  grep -E "Test mean|Test median|Test FPS" "$LOG" | sed "s/^/  /"
done

echo ""
echo "=== SUMMARY ==="
for entry in "${RUNS[@]}"; do
  NAME="${entry%%|*}"
  LOG=$LOGDIR/${NAME}.log
  if [ -f "$LOG" ]; then
    MEAN=$(grep "Test mean distance" "$LOG" | tail -1 | awk '{print $NF}')
    MEDIAN=$(grep "Test median distance" "$LOG" | tail -1 | awk '{print $NF}')
    printf "%-30s mean=%s  median=%s\n" "$NAME" "${MEAN:-?}" "${MEDIAN:-?}"
  fi
done
