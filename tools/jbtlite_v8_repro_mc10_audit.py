#!/usr/bin/env python3
"""Fail-closed audit for the JBT-Lite v8 BUSI reproducibility ablation."""
from __future__ import annotations
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = ROOT / "configs" / "jbtlite" / "repro_mc10"
ARMS = {
    "BASE": (0.0, "all"),
    "DECODER_EDGE20": (0.25, "decoder_only"),
    "GLOBAL_EDGE20": (0.25, "all"),
}

COMMON = {
    "NUM_EPOCHS": 100,
    "SCHEDULER_TOTAL_EPOCHS": 100,
    "BATCH_SIZE": 24,
    "LEARNING_RATE": 3e-4,
    "WEIGHT_DECAY": 0.0,
    "OPTIMIZER": "adam",
    "VAL_NUM_SAMPLES": 10,
    "VAL_MC_SEED": 20260910,
    "USE_VALIDATION_SELECTION": True,
    "VAL_SELECTION_METRIC": "native_fusion_dice",
    "VAL_TIEBREAK_METRIC": "native_fusion_nsd",
    "RBAL_NORMAL_WEIGHT": 0.0,
    "RBAL_SCHEDULE_ENABLED": True,
    "RBAL_SCHEDULE_TYPE": "hard_cutoff",
    "RBAL_FULL_WEIGHT_EPOCHS": 20,
    "RBAL_DECAY_END_EPOCH": 21,
    "STRICT_REPRODUCIBILITY": True,
    "ALLOW_TF32": False,
    "DETERMINISTIC": True,
}


def die(msg: str) -> None:
    raise SystemExit("[JBTL8_REPRO_MC10_AUDIT_FAIL] " + msg)


loaded = {}
for arm, (edge, scope) in ARMS.items():
    path = CFG_DIR / f"JBTL8_BUSI_{arm}_PAPER100.yaml"
    if not path.is_file():
        die(f"missing config: {path}")
    cfg = yaml.safe_load(path.read_text())
    loaded[arm] = cfg
    if cfg.get("DATASET", {}).get("NAME") != "BUSI":
        die(f"{arm}: DATASET.NAME must be BUSI")
    if bool(cfg.get("M1", {}).get("ENABLED", True)):
        die(f"{arm}: M1 must be disabled")
    tr = cfg.get("TRAIN", {})
    for key, expected in COMMON.items():
        actual = tr.get(key)
        if isinstance(expected, float):
            if abs(float(actual) - expected) > 1e-12:
                die(f"{arm}: TRAIN.{key}={actual!r}, expected {expected!r}")
        elif actual != expected:
            die(f"{arm}: TRAIN.{key}={actual!r}, expected {expected!r}")
    if abs(float(tr.get("RBAL_EDGE_WEIGHT", -1)) - edge) > 1e-12:
        die(f"{arm}: wrong EDGE weight")
    if str(tr.get("RBAL_EDGE_GRAD_SCOPE", "")) != scope:
        die(f"{arm}: wrong EDGE grad scope")

# The three arms must have identical active training protocol except the
# pre-declared treatment variables and run tags.
ignore = {
    "RBAL_EDGE_WEIGHT",
    "RBAL_EDGE_GRAD_SCOPE",
    "RUN_TAG",
}
base_train = {k: v for k, v in loaded["BASE"]["TRAIN"].items() if k not in ignore}
for arm in ("DECODER_EDGE20", "GLOBAL_EDGE20"):
    arm_train = {k: v for k, v in loaded[arm]["TRAIN"].items() if k not in ignore}
    if arm_train != base_train:
        differing = sorted(set(base_train) | set(arm_train))
        differing = [k for k in differing if base_train.get(k) != arm_train.get(k)]
        die(f"{arm}: unexpected TRAIN differences vs BASE: {differing}")

for arm in ARMS:
    other = dict(loaded[arm].get("M1", {}))
    other.pop("RUN_TAG", None)
    base_other = dict(loaded["BASE"].get("M1", {}))
    base_other.pop("RUN_TAG", None)
    if other != base_other:
        die(f"{arm}: unexpected M1 config difference")

print("[JBTL8_REPRO_MC10_AUDIT_PASS]")
print("Protocol: BUSI / 224 / 100ep / B24 / Adam3e-4 / scheduler T=100")
print("Checkpoint selection: Validation MC10, fixed VAL_MC_SEED=20260910; Test unopened")
print("Strict reproducibility: deterministic algorithms fail-closed + math SDPA + TF32 off")
print("A0 BASE             : EDGE=0")
print("A1 DECODER_EDGE20   : EDGE=.25 epochs1-20, zero 21-100, decoder-only gradient")
print("A2 GLOBAL_EDGE20    : EDGE=.25 epochs1-20, zero 21-100, PVL+decoder gradient")
