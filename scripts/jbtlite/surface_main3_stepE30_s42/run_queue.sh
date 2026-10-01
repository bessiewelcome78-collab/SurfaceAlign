#!/usr/bin/env bash
set -Eeuo pipefail
GPU="${1:?GPU required}"; shift
for DS in "$@"; do
  echo "[QUEUE] gpu=$GPU dataset=$DS start=$(date '+%F %T')"
  if bash "$(dirname "$0")/run_formal_one.sh" "$DS" "$GPU" 42; then
    echo "[QUEUE_OK] gpu=$GPU dataset=$DS end=$(date '+%F %T')"
  else
    rc=$?
    echo "[QUEUE_FAIL] gpu=$GPU dataset=$DS rc=$rc end=$(date '+%F %T')" >&2
    exit "$rc"
  fi
done
