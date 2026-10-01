#!/usr/bin/env bash
set -Eeuo pipefail
GPU="${1:?GPU required}"
shift
for item in "$@"; do
  ARM="${item%%:*}"
  SEED="${item##*:}"
  bash "${PROJECT:?}/scripts/jbtlite/surface_stepE30_fair_s42/run_train_val_one.sh" "$ARM" "$GPU" "$SEED"
done
