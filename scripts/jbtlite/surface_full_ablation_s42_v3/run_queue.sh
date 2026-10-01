#!/usr/bin/env bash
set -uo pipefail
GPU="${1:?GPU required}"; shift
FAIL=0
for ARM in "$@"; do
  echo "[QUEUE] gpu=$GPU arm=$ARM start=$(date '+%F %T')"
  if bash "${PROJECT:?}/scripts/jbtlite/surface_full_ablation_s42_v3/run_train_val_one.sh" "$ARM" "$GPU" 42; then
    echo "[QUEUE_OK] gpu=$GPU arm=$ARM"
  else
    rc=$?; FAIL=1
    echo "[QUEUE_FAIL] gpu=$GPU arm=$ARM rc=$rc" >&2
    mkdir -p "$PROJECT/logs/${STUDY_ID:?}"
    echo "gpu=$GPU arm=$ARM rc=$rc" >> "$PROJECT/logs/$STUDY_ID/FAILED_TRAIN_VAL.txt"
  fi
done
exit "$FAIL"
