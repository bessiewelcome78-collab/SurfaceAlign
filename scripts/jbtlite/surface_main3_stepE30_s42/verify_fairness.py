#!/usr/bin/env python3
from pathlib import Path
import sys, json, hashlib
try:
    import yaml
except Exception as e:
    raise SystemExit(f"[FAIL] PyYAML unavailable: {e}")

project = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd().resolve()
anchor = project / "configs/jbtlite/surface_full_ablation_s42_v3/FULL_STEP_E30.yaml"
root = project / "configs/jbtlite/surface_main3_stepE30_s42"
if not anchor.exists():
    raise SystemExit(f"[FAIL] BUSI anchor missing: {anchor}")

def load(p):
    with p.open() as f: return yaml.safe_load(f)

def normalized(d):
    x = json.loads(json.dumps(d))
    # Dataset identity is the only allowed dataset-specific factor.
    x["DATASET"]["NAME"] = "<DATASET>"
    for k in ("TRAIN_PATH","VAL_PATH","TEST_PATH","TEXT_PROMPT_PATH"):
        x["DATASET"][k] = f"<DATASET>/{k}"
    # Tags only identify output provenance.
    if "M1" in x: x["M1"]["RUN_TAG"] = "<TAG>"
    x["TRAIN"]["RUN_TAG"] = "<TAG>"
    return x

A = normalized(load(anchor))
contracts = {}
for ds in ("Kvasir","ISIC","BTMRI"):
    p = root / f"FULL_STEP_E30_{ds}.yaml"
    if not p.exists(): raise SystemExit(f"[FAIL] missing {p}")
    raw = load(p)
    if normalized(raw) != A:
        # Print top-level differing normalized blocks for fast debugging.
        diffs=[k for k in sorted(set(A)|set(normalized(raw))) if A.get(k)!=normalized(raw).get(k)]
        raise SystemExit(f"[FAIL] {ds} differs from BUSI anchor outside allowed dataset/tag fields: {diffs}")
    tr=raw["TRAIN"]; te=raw["TEST"]
    assert raw["DATASET"]["SIZE"] == 224
    assert tr["NUM_EPOCHS"] == 100
    assert tr["BATCH_SIZE"] == 24
    assert abs(float(tr["LEARNING_RATE"])-3e-4) < 1e-12
    assert tr["OPTIMIZER"].lower() == "adam"
    assert tr["VAL_NUM_SAMPLES"] == 10
    assert te["NUM_SAMPLES"] == 30
    assert abs(float(tr["RBAL_EDGE_WEIGHT"])-0.05) < 1e-12
    assert tr["RBAL_BOUNDARY_RADIUS_PX"] == 1
    assert tr["RBAL_SCHEDULE_ENABLED"] is True
    assert tr["RBAL_SCHEDULE_TYPE"] == "hard_cutoff"
    assert tr["RBAL_FULL_WEIGHT_EPOCHS"] == 30
    assert tr["RBAL_DECAY_END_EPOCH"] == 30
    assert tr["RBAL_EDGE_GRAD_SCOPE"] == "all"
    assert tr["RBAL_AUX_LOSS_TYPE"] == "surface"
    contracts[ds] = hashlib.sha256(p.read_bytes()).hexdigest()

print("[PASS] Kvasir / ISIC / BTMRI are exact FULL_STEP_E30 copies of BUSI except dataset identity/paths and RUN_TAG.")
print("[LOCKED] seed=42, 100ep, B24, Adam 3e-4, r=1, lambda0=0.05, hard-cutoff e30, full-path, Val MC10, Test MC30.")
for ds,h in contracts.items(): print(f"[SHA256] {ds:7s} {h}")
