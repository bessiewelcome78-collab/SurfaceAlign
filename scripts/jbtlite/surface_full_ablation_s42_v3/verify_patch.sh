#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT="${PROJECT:-$(cd "$(dirname "$0")/../../.." && pwd)}"; PYTHON="${PYTHON:-/home/tsz-25/miniconda3/envs/py3.10torch2.9.1cu128/bin/python}"
[[ -f "$PROJECT/train.py" && -f "$PROJECT/utils/jbtl_rbal_loss.py" ]] || { echo '[FAIL] run from MedCLIPSeg_SURFACE_FAIR_SEP14 project' >&2; exit 2; }
grep -q 'hard_cutoff' "$PROJECT/train.py" || { echo '[FAIL] hard_cutoff schedule support missing' >&2; exit 3; }
grep -q 'decoder_only' "$PROJECT/train.py" || { echo '[FAIL] decoder-only gradient routing support missing' >&2; exit 3; }
grep -q 'compute_boundary_loss' "$PROJECT/utils/jbtl_rbal_loss.py" || { echo '[FAIL] auxiliary objective support missing' >&2; exit 3; }
"$PYTHON" "$PROJECT/scripts/jbtlite/surface_full_ablation_s42_v3/verify_patch.py" "$PROJECT"
echo '[PASS] patch is additive: no train.py/model/loss file is replaced.'
