#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT="${PROJECT:-$(cd "$(dirname "$0")/../../.." && pwd)}"; RESULT_ROOT="${RESULT_ROOT:-$PROJECT/surface_full_ablation_s42_v3_results}"; STUDY_ID="${STUDY_ID:-$(cat "$PROJECT/SURFACE_FULL_ABLATION_S42_V3_ACTIVE.txt" 2>/dev/null || true)}"; [[ -n "$STUDY_ID" ]] || { echo '[FAIL] no STUDY_ID' >&2; exit 1; }
ALL=(BASE SURFACE_ALWAYS DECODER_ONLY_STEP_E30 FULL_STEP_E30 SCHED_COSINE SCHED_WARMUP_E30 LOSS_BOUNDARY_STEP_E30 LOSS_HD_STEP_E30 LOSS_ACTIVE_CONTOUR_STEP_E30 RADIUS_R2_STEP_E30 RADIUS_R3_STEP_E30 RADIUS_R5_STEP_E30 WEIGHT_L010_STEP_E30 WEIGHT_L015_STEP_E30 WEIGHT_L020_STEP_E30 WEIGHT_L025_STEP_E30)
TEST=" BASE SURFACE_ALWAYS DECODER_ONLY_STEP_E30 FULL_STEP_E30 SCHED_COSINE SCHED_WARMUP_E30 LOSS_BOUNDARY_STEP_E30 LOSS_HD_STEP_E30 LOSS_ACTIVE_CONTOUR_STEP_E30 "
val_done=0; test_done=0
echo "STUDY_ID=$STUDY_ID"
printf '%-34s %-12s %-12s\n' ARM VAL TEST
for a in "${ALL[@]}"; do
 r="$RESULT_ROOT/$STUDY_ID/BUSI/$a/seed42"; v=RUN; [[ -f "$r/VAL_ONLY_COMPLETE.txt" ]] && { v=DONE; val_done=$((val_done+1)); }
 if [[ "$TEST" == *" $a "* ]]; then t=WAIT; [[ -f "$r/TEST_COMPLETE.txt" ]] && { t=DONE; test_done=$((test_done+1)); }; else t=VAL_ONLY; fi
 printf '%-34s %-12s %-12s\n' "$a" "$v" "$t"
done
echo "VAL=$val_done/16  TEST=$test_done/9"
[[ -f "$RESULT_ROOT/$STUDY_ID/FROZEN_TEST_MANIFEST.json" ]] && echo 'FROZEN_TEST_MANIFEST=YES' || echo 'FROZEN_TEST_MANIFEST=NO'
