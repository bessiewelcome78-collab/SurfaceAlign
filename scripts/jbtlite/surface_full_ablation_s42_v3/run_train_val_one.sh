#!/usr/bin/env bash
set -Eeuo pipefail
ARM="${1:?ARM required}"
GPU="${2:?GPU required}"
SEED="${3:-42}"

ALL_ARMS="BASE SURFACE_ALWAYS DECODER_ONLY_STEP_E30 FULL_STEP_E30 SCHED_COSINE SCHED_WARMUP_E30 LOSS_BOUNDARY_STEP_E30 LOSS_HD_STEP_E30 LOSS_ACTIVE_CONTOUR_STEP_E30 RADIUS_R2_STEP_E30 RADIUS_R3_STEP_E30 RADIUS_R5_STEP_E30 WEIGHT_L010_STEP_E30 WEIGHT_L015_STEP_E30 WEIGHT_L020_STEP_E30 WEIGHT_L025_STEP_E30"
case " $ALL_ARMS " in *" $ARM "*) ;; *) echo "[FAIL] unknown arm=$ARM" >&2; exit 2;; esac
[[ "$SEED" == 42 ]] || { echo "[FAIL] this protocol is seed=42 only" >&2; exit 2; }

PROJECT="${PROJECT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
PYTHON="${PYTHON:-/home/tsz-25/miniconda3/envs/py3.10torch2.9.1cu128/bin/python}"
DATA_DIR="${DATA_DIR:-/home/tsz-25/MedCLIPSeg-main/data}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT/surface_full_ablation_s42_v3_results}"
STUDY_ID="${STUDY_ID:?set STUDY_ID}"
CFG="$PROJECT/configs/jbtlite/surface_full_ablation_s42_v3/${ARM}.yaml"
RUN_ROOT="$RESULT_ROOT/$STUDY_ID/BUSI/$ARM/seed42"
TAG="SAFA3_BUSI_${ARM}_PAPER100_S42"

[[ -x "$PYTHON" ]] || { echo "[FAIL] python not executable: $PYTHON" >&2; exit 3; }
[[ -f "$CFG" ]] || { echo "[FAIL] missing config: $CFG" >&2; exit 3; }
for d in Train_Folder Val_Folder Test_Folder Prompts_Folder; do
  [[ -d "$DATA_DIR/BUSI/$d" ]] || { echo "[FAIL] missing $DATA_DIR/BUSI/$d" >&2; exit 3; }
done
if [[ -e "$RUN_ROOT/VAL_ONLY_COMPLETE.txt" ]]; then
  echo "[SKIP] val complete arm=$ARM"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="$GPU" PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT"
mkdir -p "$RUN_ROOT"/{train,val,logs}
exec > >(tee -a "$RUN_ROOT/logs/driver.log") 2>&1

OVR=(
  DATASET.NAME BUSI
  DATASET.TRAIN_PATH "$DATA_DIR/BUSI/Train_Folder/"
  DATASET.VAL_PATH "$DATA_DIR/BUSI/Val_Folder/"
  DATASET.TEST_PATH "$DATA_DIR/BUSI/Test_Folder/"
  DATASET.TEXT_PROMPT_PATH "$DATA_DIR/BUSI/Prompts_Folder/"
  TRAIN.RUN_TAG "$TAG"
)

echo "[SAFA3_VAL_START] arm=$ARM gpu=$GPU seed=42 root=$RUN_ROOT"
"$PYTHON" -u train.py --config-file "$CFG" --output-dir "$RUN_ROOT/train" \
  --seed 42 "${OVR[@]}" 2>&1 | tee "$RUN_ROOT/logs/train.log"

CKPT_DIR="$RUN_ROOT/train/BUSI/trained_models/seed42"
CKPT="$(find "$CKPT_DIR" -maxdepth 1 -type f -name '*_best_val.pth' -print | sort | tail -n1)"
[[ -n "$CKPT" && -s "$CKPT" ]] || { echo "[FAIL] best-val checkpoint missing: $ARM" >&2; exit 4; }
printf '%s\n' "$CKPT" > "$RUN_ROOT/LOCKED_CHECKPOINT.txt"
sha256sum "$CKPT" > "$RUN_ROOT/LOCKED_CHECKPOINT.sha256"
sha256sum "$CFG" > "$RUN_ROOT/LOCKED_CONFIG.sha256"

"$PYTHON" -u test.py --config-file "$CFG" --seed 42 --split val \
  --prompt_design original --num-samples 10 --checkpoint "$CKPT" \
  --export-mode base --inference-batch-size 1 --output-dir "$RUN_ROOT/val" \
  "${OVR[@]}" 2>&1 | tee "$RUN_ROOT/logs/val_mc10.log"

for MODE in true2d paper_legacy; do
  "$PYTHON" -u utils/eval.py --config-file "$CFG" --seed 42 \
    --split val --prompt_design original --output-dir "$RUN_ROOT/val" \
    --csv-name "${ARM,,}_val_mc10_${MODE}.csv" --nsd-mode "$MODE" \
    "${OVR[@]}" 2>&1 | tee "$RUN_ROOT/logs/eval_val_mc10_${MODE}.log"
done

printf '%s\n' VAL_LOCKED_NO_TEST_OPENED > "$RUN_ROOT/VAL_ONLY_COMPLETE.txt"
echo "[SAFA3_VAL_DONE] arm=$ARM checkpoint=$CKPT TEST_NOT_OPENED"
