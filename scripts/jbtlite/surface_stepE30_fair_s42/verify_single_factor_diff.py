#!/usr/bin/env python3
"""Verify that every new arm in surface_stepE30_fair_s42 differs from the
frozen FULL_STEP_E30 anchor (DSC/NSD = 87.15/59.85/89.84, seed 42) by
exactly one *conceptual* factor.

This is a fairness gate: if this script fails, at least one config secretly
changed more than the thing it claims to isolate, and the resulting ablation
row would not be a valid single-factor comparison.

Schedule-shape is treated as one conceptual factor even though the "cosine"
schedule needs two correlated YAML keys (RBAL_SCHEDULE_TYPE and
RBAL_FULL_WEIGHT_EPOCHS) to reproduce the exact hold-then-decay shape used
in the paper (hold=10, decay_end=30), whereas "hard_cutoff" only needs one
(hold=30). Every other factor is a single YAML key.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
PROJECT = HERE.parents[3]
FULL_CFG = PROJECT / "configs/jbtlite/surface_ablation_redesign_v2/FULL_STEP_E30.yaml"
VDIR = PROJECT / "configs/jbtlite/surface_stepE30_fair_s42"

ALL_FIELDS = [
    "RBAL_EDGE_WEIGHT",
    "RBAL_BOUNDARY_RADIUS_PX",
    "RBAL_SCHEDULE_TYPE",
    "RBAL_FULL_WEIGHT_EPOCHS",
    "RBAL_DECAY_END_EPOCH",
    "RBAL_EDGE_GRAD_SCOPE",
    "RBAL_AUX_LOSS_TYPE",
    "RBAL_WARMUP_END_EPOCH",
    "RBAL_WARMUP_START_SCALE",
]

# arm -> (human label of the one factor it isolates, set of YAML keys that
#         are allowed/required to change to express that one factor)
EXPECTED = {
    "DECODER_ONLY_STEP_E30": ("gradient scope (full-path vs decoder-only)", {"RBAL_EDGE_GRAD_SCOPE"}),
    "LOSS_BOUNDARY_STEP_E30": ("auxiliary objective identity", {"RBAL_AUX_LOSS_TYPE"}),
    "LOSS_HD_STEP_E30": ("auxiliary objective identity", {"RBAL_AUX_LOSS_TYPE"}),
    "LOSS_ACTIVE_CONTOUR_STEP_E30": ("auxiliary objective identity", {"RBAL_AUX_LOSS_TYPE"}),
    "SCHED_COSINE": ("schedule shape", {"RBAL_SCHEDULE_TYPE", "RBAL_FULL_WEIGHT_EPOCHS"}),
    "SCHED_WARMUP_E30": ("schedule shape", {"RBAL_SCHEDULE_TYPE"}),
    "RADIUS_R2_STEP_E30": ("surface radius r", {"RBAL_BOUNDARY_RADIUS_PX"}),
    "RADIUS_R3_STEP_E30": ("surface radius r", {"RBAL_BOUNDARY_RADIUS_PX"}),
    "RADIUS_R5_STEP_E30": ("surface radius r", {"RBAL_BOUNDARY_RADIUS_PX"}),
    "WEIGHT_L010_STEP_E30": ("initial weight lambda0", {"RBAL_EDGE_WEIGHT"}),
    "WEIGHT_L015_STEP_E30": ("initial weight lambda0", {"RBAL_EDGE_WEIGHT"}),
    "WEIGHT_L020_STEP_E30": ("initial weight lambda0", {"RBAL_EDGE_WEIGHT"}),
    "WEIGHT_L025_STEP_E30": ("initial weight lambda0", {"RBAL_EDGE_WEIGHT"}),
}


def get_kv(text: str, key: str) -> str | None:
    m = re.search(rf"^\s*{key}:\s*(.*)$", text, re.MULTILINE)
    return m.group(1).strip() if m else None


def main() -> int:
    if not FULL_CFG.is_file():
        print(f"[FAIL] anchor config missing: {FULL_CFG}", file=sys.stderr)
        return 2
    full_text = FULL_CFG.read_text()
    full_vals = {f: get_kv(full_text, f) for f in ALL_FIELDS}

    ok = True
    missing_arms = []
    for arm, (label, allowed_keys) in EXPECTED.items():
        cfg = VDIR / f"{arm}.yaml"
        if not cfg.is_file():
            missing_arms.append(arm)
            ok = False
            continue
        text = cfg.read_text()
        changed = {
            f: (full_vals[f], get_kv(text, f))
            for f in ALL_FIELDS
            if get_kv(text, f) != full_vals[f]
        }
        changed_keys = set(changed.keys())
        status = "OK" if changed_keys == allowed_keys else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"{status:5s} {arm:32s} isolates: {label:38s} diff={changed}")

    if missing_arms:
        print(f"[FAIL] missing config files: {missing_arms}", file=sys.stderr)

    print("\n[PASS] every new arm changes exactly the one factor it claims to." if ok
          else "\n[FAIL] at least one arm changes more (or less) than its claimed factor.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
