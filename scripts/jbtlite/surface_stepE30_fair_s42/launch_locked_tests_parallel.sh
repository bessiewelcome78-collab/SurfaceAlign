#!/usr/bin/env bash
set -Eeuo pipefail
# Opens Test for the six test-track arms at seed 42 (6 jobs), across 6 GPU
# queues. Requires:
#   1. Those six validation runs completed (VAL_ONLY_COMPLETE.txt present).
#   2. STUDY_ID pointing at that same validation study.
#   3. I_ACKNOWLEDGE_SFE30S42_TEST=YES exported, confirming that no config
#      or checkpoint was changed after looking at validation results.
# Radius and weight arms are excluded by design and will be refused.

PROJECT="${PROJECT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
PYTHON="${PYTHON:-/home/tsz-25/miniconda3/envs/py3.10torch2.9.1cu128/bin/python}"
DATA_DIR="${DATA_DIR:-/home/tsz-25/MedCLIPSeg-main/data}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT/surface_stepE30_fair_s42_results}"
STUDY_ID="${STUDY_ID:?set STUDY_ID (see SURFACE_STEPE30_FAIR_S42_ACTIVE.txt)}"
: "${I_ACKNOWLEDGE_SFE30S42_TEST:?export I_ACKNOWLEDGE_SFE30S42_TEST=YES after freezing validation}"
SEED=42
LOGDIR="$PROJECT/logs/${STUDY_ID}_test"
mkdir -p "$LOGDIR"

launch_one() {
  local gpu="$1" arm="$2"
  nohup env PROJECT="$PROJECT" PYTHON="$PYTHON" DATA_DIR="$DATA_DIR" \
    RESULT_ROOT="$RESULT_ROOT" STUDY_ID="$STUDY_ID" \
    I_ACKNOWLEDGE_SFE30S42_TEST="$I_ACKNOWLEDGE_SFE30S42_TEST" \
    bash "$PROJECT/scripts/jbtlite/surface_stepE30_fair_s42/run_locked_test_one.sh" \
      "$arm" "$gpu" "$SEED" \
    > "$LOGDIR/${arm}.test.log" 2>&1 &
  echo "$arm gpu=$gpu pid=$!" | tee -a "$LOGDIR/TEST_PIDS.txt"
}

launch_one "${GPU0:-0}" DECODER_ONLY_STEP_E30
launch_one "${GPU1:-1}" LOSS_BOUNDARY_STEP_E30
launch_one "${GPU2:-2}" LOSS_HD_STEP_E30
launch_one "${GPU3:-3}" LOSS_ACTIVE_CONTOUR_STEP_E30
launch_one "${GPU4:-4}" SCHED_COSINE
launch_one "${GPU5:-5}" SCHED_WARMUP_E30

echo "STUDY_ID=$STUDY_ID"
echo "Logs: $LOGDIR"
