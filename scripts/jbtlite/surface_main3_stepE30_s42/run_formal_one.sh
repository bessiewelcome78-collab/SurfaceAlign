#!/usr/bin/env bash
set -Eeuo pipefail
DATASET="${1:?DATASET required: Kvasir|ISIC|BTMRI}"
GPU="${2:?GPU required}"
SEED="${3:-42}"
case "$DATASET" in Kvasir|ISIC|BTMRI) ;; *) echo "[FAIL] unsupported dataset=$DATASET" >&2; exit 2;; esac
[[ "$SEED" == 42 ]] || { echo "[FAIL] formal protocol is seed=42 only" >&2; exit 2; }

PROJECT="${PROJECT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
PYTHON="${PYTHON:-/home/tsz-25/miniconda3/envs/py3.10torch2.9.1cu128/bin/python}"
DATA_DIR="${DATA_DIR:-/home/tsz-25/MedCLIPSeg-main/data}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT/surface_main3_stepE30_s42_results}"
STUDY_ID="${STUDY_ID:?set STUDY_ID}"
CFG="$PROJECT/configs/jbtlite/surface_main3_stepE30_s42/FULL_STEP_E30_${DATASET}.yaml"
RUN_ROOT="$RESULT_ROOT/$STUDY_ID/$DATASET/FULL_STEP_E30/seed42"
TAG="SA_MAIN3_${DATASET}_FULL_STEP_E30_PAPER100_S42"

[[ -x "$PYTHON" ]] || { echo "[FAIL] python not executable: $PYTHON" >&2; exit 3; }
[[ -f "$CFG" ]] || { echo "[FAIL] config missing: $CFG" >&2; exit 3; }
for d in Train_Folder Val_Folder Test_Folder Prompts_Folder; do
  [[ -d "$DATA_DIR/$DATASET/$d" ]] || { echo "[FAIL] missing $DATA_DIR/$DATASET/$d" >&2; exit 3; }
done
if [[ -f "$RUN_ROOT/FORMAL_COMPLETE.txt" ]]; then
  echo "[SKIP] formal complete dataset=$DATASET"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="$GPU" PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT"
mkdir -p "$RUN_ROOT"/{train,val,test,logs}
exec > >(tee -a "$RUN_ROOT/logs/driver.log") 2>&1

OVR=(
  DATASET.NAME "$DATASET"
  DATASET.TRAIN_PATH "$DATA_DIR/$DATASET/Train_Folder/"
  DATASET.VAL_PATH "$DATA_DIR/$DATASET/Val_Folder/"
  DATASET.TEST_PATH "$DATA_DIR/$DATASET/Test_Folder/"
  DATASET.TEXT_PROMPT_PATH "$DATA_DIR/$DATASET/Prompts_Folder/"
  TRAIN.RUN_TAG "$TAG"
)

echo "[MAIN3_START] dataset=$DATASET gpu=$GPU seed=42 root=$RUN_ROOT"
sha256sum "$CFG" > "$RUN_ROOT/LOCKED_CONFIG.sha256"

# Train 100 epochs; checkpoint selection is validation-only MC10/native metrics.
"$PYTHON" -u train.py --config-file "$CFG" --output-dir "$RUN_ROOT/train" \
  --seed 42 "${OVR[@]}" 2>&1 | tee "$RUN_ROOT/logs/train.log"

CKPT_DIR="$RUN_ROOT/train/$DATASET/trained_models/seed42"
CKPT="$(find "$CKPT_DIR" -maxdepth 1 -type f -name '*_best_val.pth' -print | sort | tail -n1)"
[[ -n "$CKPT" && -s "$CKPT" ]] || { echo "[FAIL] best-val checkpoint missing: $DATASET" >&2; exit 4; }
printf '%s\n' "$CKPT" > "$RUN_ROOT/LOCKED_CHECKPOINT.txt"
sha256sum "$CKPT" > "$RUN_ROOT/LOCKED_CHECKPOINT.sha256"

# Locked validation export: MC10.
"$PYTHON" -u test.py --config-file "$CFG" --seed 42 --split val \
  --prompt_design original --num-samples 10 --checkpoint "$CKPT" \
  --export-mode base --inference-batch-size 1 --output-dir "$RUN_ROOT/val" \
  "${OVR[@]}" 2>&1 | tee "$RUN_ROOT/logs/val_mc10.log"
for MODE in true2d paper_legacy; do
  "$PYTHON" -u utils/eval.py --config-file "$CFG" --seed 42 --split val \
    --prompt_design original --output-dir "$RUN_ROOT/val" \
    --csv-name "${DATASET,,}_full_step_e30_val_mc10_${MODE}.csv" --nsd-mode "$MODE" \
    "${OVR[@]}" 2>&1 | tee "$RUN_ROOT/logs/eval_val_mc10_${MODE}.log"
done
printf '%s\n' VAL_COMPLETE > "$RUN_ROOT/VAL_COMPLETE.txt"

# Freeze provenance before opening Test.
"$PYTHON" - "$DATASET" "$CFG" "$CKPT" "$RUN_ROOT" <<'PY'
import json, hashlib, pathlib, sys, time
DS, cfg, ckpt, root = sys.argv[1:]
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
out={
  "dataset":DS,"seed":42,"recipe":"FULL_STEP_E30","radius":1,"lambda0":0.05,
  "schedule":"hard_cutoff","cutoff_epoch":30,"grad_scope":"all","aux_loss":"surface",
  "val_mc":10,"test_mc":30,"config":str(pathlib.Path(cfg).resolve()),"config_sha256":sha(cfg),
  "checkpoint":str(pathlib.Path(ckpt).resolve()),"checkpoint_sha256":sha(ckpt),
  "frozen_at_unix":time.time(),
}
path=pathlib.Path(root)/"FROZEN_TEST_MANIFEST.json"
path.write_text(json.dumps(out,indent=2)+"\n")
print("[FROZEN]",path)
PY

# Verify frozen hashes, then open Test exactly once: MC30.
EXPECT_CFG="$(awk '{print $1}' "$RUN_ROOT/LOCKED_CONFIG.sha256")"
EXPECT_CKPT="$(awk '{print $1}' "$RUN_ROOT/LOCKED_CHECKPOINT.sha256")"
[[ "$(sha256sum "$CFG" | awk '{print $1}')" == "$EXPECT_CFG" ]] || { echo "[FAIL] config changed before Test" >&2; exit 5; }
[[ "$(sha256sum "$CKPT" | awk '{print $1}')" == "$EXPECT_CKPT" ]] || { echo "[FAIL] checkpoint changed before Test" >&2; exit 5; }

"$PYTHON" -u test.py --config-file "$CFG" --seed 42 --split test \
  --prompt_design original --num-samples 30 --checkpoint "$CKPT" \
  --export-mode base --inference-batch-size 1 --output-dir "$RUN_ROOT/test" \
  "${OVR[@]}" 2>&1 | tee "$RUN_ROOT/logs/test_mc30.log"
for MODE in true2d paper_legacy; do
  "$PYTHON" -u utils/eval.py --config-file "$CFG" --seed 42 --split test \
    --prompt_design original --output-dir "$RUN_ROOT/test" \
    --csv-name "${DATASET,,}_full_step_e30_test_mc30_${MODE}.csv" --nsd-mode "$MODE" \
    "${OVR[@]}" 2>&1 | tee "$RUN_ROOT/logs/eval_test_mc30_${MODE}.log"
done
printf '%s\n' LOCKED_TEST_COMPLETE > "$RUN_ROOT/TEST_COMPLETE.txt"
printf '%s\n' FORMAL_COMPLETE > "$RUN_ROOT/FORMAL_COMPLETE.txt"
echo "[MAIN3_DONE] dataset=$DATASET checkpoint=$CKPT"
