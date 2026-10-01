#!/usr/bin/env bash
set -uo pipefail
GPU="${1:?GPU required}"; shift
FAIL=0
for ARM in "$@"; do
  if I_ACKNOWLEDGE_SAFA3_TEST=YES bash "${PROJECT:?}/scripts/jbtlite/surface_full_ablation_s42_v3/run_locked_test_one.sh" "$ARM" "$GPU"; then
    echo "[TEST_QUEUE_OK] gpu=$GPU arm=$ARM"
  else
    rc=$?; FAIL=1; echo "[TEST_QUEUE_FAIL] gpu=$GPU arm=$ARM rc=$rc" >&2
    mkdir -p "$PROJECT/logs/${STUDY_ID:?}"; echo "gpu=$GPU arm=$ARM rc=$rc" >> "$PROJECT/logs/$STUDY_ID/FAILED_TEST.txt"
  fi
done
exit "$FAIL"
