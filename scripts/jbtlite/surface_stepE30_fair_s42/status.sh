#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT="${PROJECT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT/surface_stepE30_fair_s42_results}"
STUDY_ID="${STUDY_ID:-$(cat "$PROJECT/SURFACE_STEPE30_FAIR_S42_ACTIVE.txt" 2>/dev/null || true)}"
[[ -n "$STUDY_ID" ]] || { echo "[FAIL] set STUDY_ID or run launch_train_val_parallel.sh first" >&2; exit 1; }
SEED=42
TEST_TRACK="DECODER_ONLY_STEP_E30 LOSS_BOUNDARY_STEP_E30 LOSS_HD_STEP_E30 LOSS_ACTIVE_CONTOUR_STEP_E30 SCHED_COSINE SCHED_WARMUP_E30"
VAL_ONLY="RADIUS_R2_STEP_E30 RADIUS_R3_STEP_E30 RADIUS_R5_STEP_E30 WEIGHT_L010_STEP_E30 WEIGHT_L015_STEP_E30 WEIGHT_L020_STEP_E30 WEIGHT_L025_STEP_E30"
echo "STUDY_ID=$STUDY_ID  (seed $SEED only)"
printf "%-32s %-6s %-14s\n" ARM VAL TEST
dv=0; tv=0; tt=0; nv=0
for ARM in $TEST_TRACK $VAL_ONLY; do
  RUN="$RESULT_ROOT/$STUDY_ID/BUSI/$ARM/seed$SEED"
  nv=$((nv+1)); V="-"; T="-"
  [[ -f "$RUN/VAL_ONLY_COMPLETE.txt" ]] && { V="OK"; dv=$((dv+1)); }
  case " $TEST_TRACK " in
    *" $ARM "*) tt=$((tt+1)); [[ -f "$RUN/TEST_COMPLETE.txt" ]] && { T="OK"; tv=$((tv+1)); } ;;
    *) T="n/a (val-only)" ;;
  esac
  printf "%-32s %-6s %-14s\n" "$ARM" "$V" "$T"
done
echo "----"
echo "Validation:  $dv / $nv complete"
echo "Locked test: $tv / $tt complete (test-track arms only)"
