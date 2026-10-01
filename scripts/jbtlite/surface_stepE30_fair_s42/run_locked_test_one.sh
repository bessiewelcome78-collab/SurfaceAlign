#!/usr/bin/env bash
set -Eeuo pipefail

ARM="${1:?ARM required}"
GPU="${2:?GPU required}"
SEED="${3:-42}"

[[ "${I_ACKNOWLEDGE_SFE30S42_TEST:-NO}" == YES ]] || {
  echo '[LOCKED] set I_ACKNOWLEDGE_SFE30S42_TEST=YES only after all validation runs' >&2
  echo '         for this arm/seed are frozen (VAL_ONLY_COMPLETE.txt present).' >&2
  exit 20
}

# Only the six "test-track" arms are allowed to open Test. Radius and weight
# sensitivity arms are validation-only by design (same policy as Tables 5-6
# in the paper: no test-set hyperparameter selection).
case "$ARM" in
  DECODER_ONLY_STEP_E30|LOSS_BOUNDARY_STEP_E30|LOSS_HD_STEP_E30|LOSS_ACTIVE_CONTOUR_STEP_E30|\
  SCHED_COSINE|SCHED_WARMUP_E30) ;;
  RADIUS_R2_STEP_E30|RADIUS_R3_STEP_E30|RADIUS_R5_STEP_E30|\
  WEIGHT_L010_STEP_E30|WEIGHT_L015_STEP_E30|WEIGHT_L020_STEP_E30|WEIGHT_L025_STEP_E30)
    echo "[FAIL] $ARM is validation-only by design; Test is forbidden (avoids test-set tuning)." >&2
    exit 22 ;;
  *) echo "[FAIL] unknown arm: $ARM" >&2; exit 2 ;;
esac

PROJECT="${PROJECT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
PYTHON="${PYTHON:-/home/tsz-25/miniconda3/envs/py3.10torch2.9.1cu128/bin/python}"
DATA_DIR="${DATA_DIR:-/home/tsz-25/MedCLIPSeg-main/data}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT/surface_stepE30_fair_s42_results}"
STUDY_ID="${STUDY_ID:?set STUDY_ID of the validation-locked run}"
CFG="$PROJECT/configs/jbtlite/surface_stepE30_fair_s42/${ARM}.yaml"
RUN_ROOT="$RESULT_ROOT/$STUDY_ID/BUSI/$ARM/seed$SEED"
TAG="SFE30S42_BUSI_${ARM}_PAPER100_S${SEED}"

grep -qx VAL_LOCKED_NO_TEST_OPENED "$RUN_ROOT/VAL_ONLY_COMPLETE.txt" || {
  echo "[FAIL] validation lock marker missing/invalid: $ARM seed=$SEED" >&2
  exit 21
}
CKPT="$(cat "$RUN_ROOT/LOCKED_CHECKPOINT.txt")"
[[ -s "$CKPT" ]] || { echo "[FAIL] locked checkpoint missing: $ARM" >&2; exit 21; }
sha256sum -c "$RUN_ROOT/LOCKED_CHECKPOINT.sha256"
sha256sum -c "$RUN_ROOT/LOCKED_CONFIG.sha256"
if [[ -e "$RUN_ROOT/TEST_COMPLETE.txt" ]]; then
  echo "[SKIP] locked Test already complete: $ARM seed=$SEED"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="$GPU" PYTHONHASHSEED="$SEED" CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT"
mkdir -p "$RUN_ROOT"/{test,logs}

OVR=(
  DATASET.NAME BUSI
  DATASET.TRAIN_PATH "$DATA_DIR/BUSI/Train_Folder/"
  DATASET.VAL_PATH "$DATA_DIR/BUSI/Val_Folder/"
  DATASET.TEST_PATH "$DATA_DIR/BUSI/Test_Folder/"
  DATASET.TEXT_PROMPT_PATH "$DATA_DIR/BUSI/Prompts_Folder/"
  TRAIN.RUN_TAG "$TAG"
)

echo "[SFE30S42_TEST_START] arm=$ARM gpu=$GPU seed=$SEED checkpoint=$CKPT"
"$PYTHON" -u test.py --config-file "$CFG" --seed "$SEED" --split test \
  --prompt_design original --num-samples 30 --checkpoint "$CKPT" --export-mode base \
  --inference-batch-size 1 --output-dir "$RUN_ROOT/test" "${OVR[@]}" 2>&1 | tee "$RUN_ROOT/logs/test_mc30.log"

for MODE in true2d paper_legacy; do
  "$PYTHON" -u utils/eval.py --config-file "$CFG" --seed "$SEED" --split test \
    --prompt_design original --output-dir "$RUN_ROOT/test" --csv-name "${ARM,,}_test_mc30_${MODE}.csv" \
    --nsd-mode "$MODE" "${OVR[@]}" 2>&1 | tee "$RUN_ROOT/logs/eval_test_mc30_${MODE}.log"
done

printf '%s\n' LOCKED_TEST_COMPLETE > "$RUN_ROOT/TEST_COMPLETE.txt"
echo "[SFE30S42_TEST_DONE] arm=$ARM seed=$SEED"
