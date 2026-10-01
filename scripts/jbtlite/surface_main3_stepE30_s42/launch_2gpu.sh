#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT="${PROJECT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
PYTHON="${PYTHON:-/home/tsz-25/miniconda3/envs/py3.10torch2.9.1cu128/bin/python}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT/surface_main3_stepE30_s42_results}"
GPU_A="${GPU_A:-2}"  # balanced: BTMRI alone
GPU_B="${GPU_B:-4}"  # Kvasir -> ISIC sequentially

"$PYTHON" "$PROJECT/scripts/jbtlite/surface_main3_stepE30_s42/verify_fairness.py" "$PROJECT"
if [[ -z "${STUDY_ID:-}" ]]; then STUDY_ID="SA_MAIN3_S42_$(date +%Y%m%d_%H%M%S)"; export STUDY_ID; fi
LOGDIR="$PROJECT/logs/$STUDY_ID"
if [[ -s "$LOGDIR/MAIN3_PIDS.txt" ]]; then
  echo "[FAIL] study already launched: $STUDY_ID" >&2; cat "$LOGDIR/MAIN3_PIDS.txt" >&2; exit 9
fi
mkdir -p "$LOGDIR" "$RESULT_ROOT/$STUDY_ID"
printf '%s\n' "$STUDY_ID" > "$PROJECT/SURFACE_MAIN3_STEPE30_S42_ACTIVE.txt"

nohup env PROJECT="$PROJECT" PYTHON="$PYTHON" RESULT_ROOT="$RESULT_ROOT" STUDY_ID="$STUDY_ID" \
  bash "$PROJECT/scripts/jbtlite/surface_main3_stepE30_s42/run_queue.sh" "$GPU_A" BTMRI \
  > "$LOGDIR/gpu${GPU_A}_BTMRI.log" 2>&1 &
echo "queueA gpu=$GPU_A pid=$! datasets=BTMRI" | tee -a "$LOGDIR/MAIN3_PIDS.txt"

nohup env PROJECT="$PROJECT" PYTHON="$PYTHON" RESULT_ROOT="$RESULT_ROOT" STUDY_ID="$STUDY_ID" \
  bash "$PROJECT/scripts/jbtlite/surface_main3_stepE30_s42/run_queue.sh" "$GPU_B" Kvasir ISIC \
  > "$LOGDIR/gpu${GPU_B}_Kvasir_ISIC.log" 2>&1 &
echo "queueB gpu=$GPU_B pid=$! datasets=Kvasir ISIC" | tee -a "$LOGDIR/MAIN3_PIDS.txt"

echo "STUDY_ID=$STUDY_ID"
echo "Results=$RESULT_ROOT/$STUDY_ID"
echo "Logs=$LOGDIR"
