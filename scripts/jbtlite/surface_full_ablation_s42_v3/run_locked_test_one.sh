#!/usr/bin/env bash
set -Eeuo pipefail
ARM="${1:?ARM required}"; GPU="${2:?GPU required}"
[[ "${I_ACKNOWLEDGE_SAFA3_TEST:-NO}" == YES ]] || { echo '[LOCKED] set I_ACKNOWLEDGE_SAFA3_TEST=YES after freeze_test_manifest.py' >&2; exit 20; }
TEST_ARMS="BASE SURFACE_ALWAYS DECODER_ONLY_STEP_E30 FULL_STEP_E30 SCHED_COSINE SCHED_WARMUP_E30 LOSS_BOUNDARY_STEP_E30 LOSS_HD_STEP_E30 LOSS_ACTIVE_CONTOUR_STEP_E30"
case " $TEST_ARMS " in *" $ARM "*) ;; *) echo "[FAIL] Test forbidden for validation-only arm=$ARM" >&2; exit 22;; esac
PROJECT="${PROJECT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
PYTHON="${PYTHON:-/home/tsz-25/miniconda3/envs/py3.10torch2.9.1cu128/bin/python}"
DATA_DIR="${DATA_DIR:-/home/tsz-25/MedCLIPSeg-main/data}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT/surface_full_ablation_s42_v3_results}"
STUDY_ID="${STUDY_ID:-$(cat "$PROJECT/SURFACE_FULL_ABLATION_S42_V3_ACTIVE.txt")}" 
RUN_ROOT="$RESULT_ROOT/$STUDY_ID/BUSI/$ARM/seed42"
CFG="$PROJECT/configs/jbtlite/surface_full_ablation_s42_v3/${ARM}.yaml"
MANIFEST="$RESULT_ROOT/$STUDY_ID/FROZEN_TEST_MANIFEST.json"
[[ -f "$MANIFEST" ]] || { echo '[FAIL] frozen manifest missing' >&2; exit 21; }
if [[ -e "$RUN_ROOT/TEST_COMPLETE.txt" ]]; then echo "[SKIP] test complete arm=$ARM"; exit 0; fi

readarray -t LOCK < <("$PYTHON" - "$MANIFEST" "$ARM" <<'PY'
import json,sys
m=json.load(open(sys.argv[1])); a=sys.argv[2]; r=m['runs'][a]
assert r['test_allowed'] is True
print(r['checkpoint']); print(r['checkpoint_sha256']); print(r['config_sha256'])
PY
)
CKPT="${LOCK[0]}"; EXPECT_CKPT="${LOCK[1]}"; EXPECT_CFG="${LOCK[2]}"
[[ "$(sha256sum "$CKPT" | awk '{print $1}')" == "$EXPECT_CKPT" ]] || { echo '[FAIL] checkpoint hash changed' >&2; exit 23; }
[[ "$(sha256sum "$CFG" | awk '{print $1}')" == "$EXPECT_CFG" ]] || { echo '[FAIL] config hash changed' >&2; exit 23; }

export CUDA_VISIBLE_DEVICES="$GPU" PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}" TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT"; mkdir -p "$RUN_ROOT"/{test,logs}
TAG="SAFA3_BUSI_${ARM}_PAPER100_S42"
OVR=(DATASET.NAME BUSI DATASET.TRAIN_PATH "$DATA_DIR/BUSI/Train_Folder/" DATASET.VAL_PATH "$DATA_DIR/BUSI/Val_Folder/" DATASET.TEST_PATH "$DATA_DIR/BUSI/Test_Folder/" DATASET.TEXT_PROMPT_PATH "$DATA_DIR/BUSI/Prompts_Folder/" TRAIN.RUN_TAG "$TAG")

echo "[SAFA3_TEST_START] arm=$ARM gpu=$GPU checkpoint=$CKPT"
"$PYTHON" -u test.py --config-file "$CFG" --seed 42 --split test --prompt_design original --num-samples 30 \
  --checkpoint "$CKPT" --export-mode base --inference-batch-size 1 --output-dir "$RUN_ROOT/test" "${OVR[@]}" 2>&1 | tee "$RUN_ROOT/logs/test_mc30.log"
for MODE in true2d paper_legacy; do
  "$PYTHON" -u utils/eval.py --config-file "$CFG" --seed 42 --split test --prompt_design original --output-dir "$RUN_ROOT/test" \
    --csv-name "${ARM,,}_test_mc30_${MODE}.csv" --nsd-mode "$MODE" "${OVR[@]}" 2>&1 | tee "$RUN_ROOT/logs/eval_test_mc30_${MODE}.log"
done
printf '%s\n' LOCKED_TEST_COMPLETE > "$RUN_ROOT/TEST_COMPLETE.txt"
echo "[SAFA3_TEST_DONE] arm=$ARM"
