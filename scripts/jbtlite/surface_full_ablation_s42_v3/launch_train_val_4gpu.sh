#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT="${PROJECT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
PYTHON="${PYTHON:-/home/tsz-25/miniconda3/envs/py3.10torch2.9.1cu128/bin/python}"
DATA_DIR="${DATA_DIR:-/home/tsz-25/MedCLIPSeg-main/data}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT/surface_full_ablation_s42_v3_results}"
STUDY_ID="${STUDY_ID:-SAFA3_S42_$(date +%Y%m%d_%H%M%S)}"
LOGDIR="$PROJECT/logs/$STUDY_ID"

# Refuse duplicate launcher invocation for an existing study.
if [[ -s "$LOGDIR/TRAIN_VAL_PIDS.txt" ]]; then
  echo "[FAIL] STUDY_ID=$STUDY_ID has already been launched." >&2
  echo "[FAIL] Existing queue record:" >&2
  cat "$LOGDIR/TRAIN_VAL_PIDS.txt" >&2
  echo "[FAIL] Do not launch the same study twice." >&2
  exit 9
fi

mkdir -p "$LOGDIR" "$RESULT_ROOT/$STUDY_ID"
printf '%s\n' "$STUDY_ID" > "$PROJECT/SURFACE_FULL_ABLATION_S42_V3_ACTIVE.txt"

bash "$PROJECT/scripts/jbtlite/surface_full_ablation_s42_v3/verify_patch.sh"

launch() {
  local gpu="$1" name="$2"; shift 2
  nohup env PROJECT="$PROJECT" PYTHON="$PYTHON" DATA_DIR="$DATA_DIR" RESULT_ROOT="$RESULT_ROOT" STUDY_ID="$STUDY_ID" \
    bash "$PROJECT/scripts/jbtlite/surface_full_ablation_s42_v3/run_queue.sh" "$gpu" "$@" \
    > "$LOGDIR/${name}.train_val.log" 2>&1 &
  echo "$name gpu=$gpu pid=$! arms=$*" | tee -a "$LOGDIR/TRAIN_VAL_PIDS.txt"
}

# 16 unique 100-epoch jobs, exactly 4 queues / 4 GPUs.
launch "${GPU0:-0}" gpu0 BASE DECODER_ONLY_STEP_E30 RADIUS_R2_STEP_E30 WEIGHT_L020_STEP_E30
launch "${GPU1:-1}" gpu1 SURFACE_ALWAYS LOSS_BOUNDARY_STEP_E30 RADIUS_R3_STEP_E30 WEIGHT_L025_STEP_E30
launch "${GPU2:-2}" gpu2 FULL_STEP_E30 LOSS_HD_STEP_E30 RADIUS_R5_STEP_E30 WEIGHT_L010_STEP_E30
launch "${GPU3:-3}" gpu3 SCHED_COSINE SCHED_WARMUP_E30 LOSS_ACTIVE_CONTOUR_STEP_E30 WEIGHT_L015_STEP_E30

echo "STUDY_ID=$STUDY_ID"
echo "Results=$RESULT_ROOT/$STUDY_ID"
echo "Logs=$LOGDIR"
echo "Status: STUDY_ID=$STUDY_ID bash scripts/jbtlite/surface_full_ablation_s42_v3/status.sh"
