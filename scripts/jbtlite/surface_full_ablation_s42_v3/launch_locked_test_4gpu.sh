#!/usr/bin/env bash
set -Eeuo pipefail
[[ "${I_ACKNOWLEDGE_SAFA3_TEST:-NO}" == YES ]] || { echo '[LOCKED] export I_ACKNOWLEDGE_SAFA3_TEST=YES after freezing validation' >&2; exit 20; }
PROJECT="${PROJECT:-$(cd "$(dirname "$0")/../../.." && pwd)}"; PYTHON="${PYTHON:-/home/tsz-25/miniconda3/envs/py3.10torch2.9.1cu128/bin/python}"; DATA_DIR="${DATA_DIR:-/home/tsz-25/MedCLIPSeg-main/data}"; RESULT_ROOT="${RESULT_ROOT:-$PROJECT/surface_full_ablation_s42_v3_results}"; STUDY_ID="${STUDY_ID:-$(cat "$PROJECT/SURFACE_FULL_ABLATION_S42_V3_ACTIVE.txt")}"; LOGDIR="$PROJECT/logs/$STUDY_ID"
[[ -f "$RESULT_ROOT/$STUDY_ID/FROZEN_TEST_MANIFEST.json" ]] || { echo '[FAIL] run freeze_test_manifest.py first' >&2; exit 21; }
launch(){ local gpu="$1" name="$2"; shift 2; nohup env PROJECT="$PROJECT" PYTHON="$PYTHON" DATA_DIR="$DATA_DIR" RESULT_ROOT="$RESULT_ROOT" STUDY_ID="$STUDY_ID" I_ACKNOWLEDGE_SAFA3_TEST=YES bash "$PROJECT/scripts/jbtlite/surface_full_ablation_s42_v3/run_test_queue.sh" "$gpu" "$@" > "$LOGDIR/${name}.locked_test.log" 2>&1 & echo "$name gpu=$gpu pid=$! arms=$*" | tee -a "$LOGDIR/TEST_PIDS.txt"; }
launch "${GPU0:-0}" test_gpu0 BASE LOSS_BOUNDARY_STEP_E30 FULL_STEP_E30
launch "${GPU1:-1}" test_gpu1 SURFACE_ALWAYS LOSS_HD_STEP_E30
launch "${GPU2:-2}" test_gpu2 DECODER_ONLY_STEP_E30 LOSS_ACTIVE_CONTOUR_STEP_E30
launch "${GPU3:-3}" test_gpu3 SCHED_COSINE SCHED_WARMUP_E30
echo "Locked tests launched for STUDY_ID=$STUDY_ID"
