#!/usr/bin/env bash
set -Eeuo pipefail
# Launches all 13 new single-factor arms at seed 42 (13 train+val jobs)
# across parallel GPU queues. Each queue is sequential on its own GPU.
# Override GPU0..GPU6 to remap physical device ids.
#
# Does NOT run BASE / LOCAL_ALWAYS / FULL_STEP_E30: those are reused as-is
# from the existing surface_ablation_redesign_v2 study.

PROJECT="${PROJECT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
PYTHON="${PYTHON:-/home/tsz-25/miniconda3/envs/py3.10torch2.9.1cu128/bin/python}"
DATA_DIR="${DATA_DIR:-/home/tsz-25/MedCLIPSeg-main/data}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT/surface_stepE30_fair_s42_results}"
STUDY_ID="${STUDY_ID:-SFE30S42_$(date +%Y%m%d_%H%M%S)}"
SEED=42
LOGDIR="$PROJECT/logs/$STUDY_ID"
mkdir -p "$LOGDIR" "$RESULT_ROOT/$STUDY_ID"
printf '%s\n' "$STUDY_ID" > "$PROJECT/SURFACE_STEPE30_FAIR_S42_ACTIVE.txt"

bash "$PROJECT/scripts/jbtlite/surface_stepE30_fair_s42/verify_patch.sh"

launch_queue() {
  local gpu="$1" name="$2"
  shift 2
  nohup env PROJECT="$PROJECT" PYTHON="$PYTHON" DATA_DIR="$DATA_DIR" \
    RESULT_ROOT="$RESULT_ROOT" STUDY_ID="$STUDY_ID" \
    bash "$PROJECT/scripts/jbtlite/surface_stepE30_fair_s42/run_queue.sh" "$gpu" "$@" \
    > "$LOGDIR/${name}.train_val.log" 2>&1 &
  echo "$name gpu=$gpu pid=$! jobs=$*" | tee -a "$LOGDIR/TRAIN_VAL_PIDS.txt"
}

# 13 jobs over 7 GPU queues (6 queues get 2 jobs, 1 queue gets 1).
launch_queue "${GPU0:-0}" gpu0 DECODER_ONLY_STEP_E30:$SEED RADIUS_R5_STEP_E30:$SEED
launch_queue "${GPU1:-1}" gpu1 LOSS_BOUNDARY_STEP_E30:$SEED WEIGHT_L010_STEP_E30:$SEED
launch_queue "${GPU2:-2}" gpu2 LOSS_HD_STEP_E30:$SEED WEIGHT_L015_STEP_E30:$SEED
launch_queue "${GPU3:-3}" gpu3 LOSS_ACTIVE_CONTOUR_STEP_E30:$SEED WEIGHT_L020_STEP_E30:$SEED
launch_queue "${GPU4:-4}" gpu4 SCHED_COSINE:$SEED WEIGHT_L025_STEP_E30:$SEED
launch_queue "${GPU5:-5}" gpu5 SCHED_WARMUP_E30:$SEED RADIUS_R2_STEP_E30:$SEED
launch_queue "${GPU6:-6}" gpu6 RADIUS_R3_STEP_E30:$SEED

echo "STUDY_ID=$STUDY_ID"
echo "Results: $RESULT_ROOT/$STUDY_ID"
echo "Logs:    $LOGDIR"
echo "Status:  STUDY_ID=$STUDY_ID bash scripts/jbtlite/surface_stepE30_fair_s42/status.sh"
