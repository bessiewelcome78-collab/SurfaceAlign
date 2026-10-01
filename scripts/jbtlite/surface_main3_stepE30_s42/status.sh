#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT="${PROJECT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT/surface_main3_stepE30_s42_results}"
STUDY_ID="${STUDY_ID:-$(cat "$PROJECT/SURFACE_MAIN3_STEPE30_S42_ACTIVE.txt")}" 
printf '%-10s %-12s %-12s\n' DATASET VAL TEST
for DS in Kvasir ISIC BTMRI; do
  R="$RESULT_ROOT/$STUDY_ID/$DS/FULL_STEP_E30/seed42"
  V=WAIT; T=WAIT
  [[ -f "$R/VAL_COMPLETE.txt" ]] && V=DONE
  [[ -f "$R/TEST_COMPLETE.txt" ]] && T=DONE
  if [[ "$V" == WAIT ]] && pgrep -af "FULL_STEP_E30_${DS}.yaml" >/dev/null 2>&1; then V=RUN; fi
  if [[ "$V" == DONE && "$T" == WAIT ]] && pgrep -af "FULL_STEP_E30_${DS}.yaml" >/dev/null 2>&1; then T=RUN; fi
  printf '%-10s %-12s %-12s\n' "$DS" "$V" "$T"
done
