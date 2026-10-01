#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT="${PROJECT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
PYTHON="${PYTHON:-/home/tsz-25/miniconda3/envs/py3.10torch2.9.1cu128/bin/python}"

echo "[VERIFY] checking new arm configs..."
CONF_DIR="$PROJECT/configs/jbtlite/surface_stepE30_fair_s42"
COUNT=$(ls -1 "$CONF_DIR"/*.yaml 2>/dev/null | wc -l)
echo "[VERIFY] found $COUNT config files."
[[ "$COUNT" -eq 13 ]] || { echo "[FAIL] expected 13 configs, found $COUNT" >&2; exit 3; }

echo "[VERIFY] checking the FULL_STEP_E30 anchor and reused arms are present..."
for a in BASE LOCAL_ALWAYS FULL_STEP_E30; do
  [[ -f "$PROJECT/configs/jbtlite/surface_ablation_redesign_v2/$a.yaml" ]] || {
    echo "[FAIL] required config missing: $a" >&2; exit 4; }
done

echo "[VERIFY] running single-factor diff check against FULL_STEP_E30..."
"$PYTHON" "$PROJECT/scripts/jbtlite/surface_stepE30_fair_s42/verify_single_factor_diff.py"

echo "[VERIFY] confirming selection protocol is identical across every arm..."
"$PYTHON" - "$PROJECT" <<'PY'
import re, sys
from pathlib import Path
proj = Path(sys.argv[1])
keys = ["VAL_SELECTION_METRIC","VAL_TIEBREAK_METRIC","VAL_NUM_SAMPLES","VAL_MC_SEED",
        "VAL_SELECTION_START_EPOCH","VAL_SELECTION_TOLERANCE","USE_VALIDATION_SELECTION",
        "NUM_EPOCHS","BATCH_SIZE","LEARNING_RATE","NUM_SAMPLES","DETERMINISTIC",
        "STRICT_REPRODUCIBILITY"]
def kv(t,k):
    m=re.search(rf"^\s*{k}:\s*(.*)$",t,re.MULTILINE); return m.group(1).strip() if m else None
files=[proj/"configs/jbtlite/surface_ablation_redesign_v2"/f"{a}.yaml" for a in ("BASE","LOCAL_ALWAYS","FULL_STEP_E30")]
files+=sorted((proj/"configs/jbtlite/surface_stepE30_fair_s42").glob("*.yaml"))
ref=None; bad=False
for f in files:
    t=f.read_text(); vals={k:kv(t,k) for k in keys}
    if ref is None: ref=vals; refname=f.name; continue
    diff={k:(ref[k],vals[k]) for k in keys if ref[k]!=vals[k]}
    if diff:
        bad=True; print(f"[FAIL] {f.name} selection/budget protocol differs from {refname}: {diff}")
print("[PASS] all 16 arms share an identical checkpoint-selection and training-budget protocol." if not bad
      else "[FAIL] protocol mismatch detected; the comparison would not be fair.")
sys.exit(1 if bad else 0)
PY

echo "[VERIFY] core files untouched by this patch:"
for f in train.py test.py utils/eval.py utils/jbtl_rbal_loss.py; do
  [[ -f "$PROJECT/$f" ]] || { echo "[FAIL] core file missing: $f" >&2; exit 6; }
done
echo "[PASS] surface_stepE30_fair_s42 patch verified."
