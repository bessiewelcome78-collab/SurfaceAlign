#!/usr/bin/env python3
"""Generate the frozen one-factor configs for SurfaceAlign Tables 3--6."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/jbtlite/surface_weight_rescue/SWR_BUSI_L005_PAPER100.yaml"
OUT = ROOT / "configs/jbtlite/surface_tables_3_6_fair"


ARMS = {
    # Table 3: schedule only.  COSINE/FULL is reused from the frozen reference.
    "T3_ALWAYS": {"RBAL_SCHEDULE_TYPE": "always_on"},
    "T3_STEP_E30": {
        "RBAL_SCHEDULE_TYPE": "hard_cutoff",
        "RBAL_FULL_WEIGHT_EPOCHS": 30,
    },
    "T3_WARMUP_E30": {
        "RBAL_SCHEDULE_TYPE": "warmup",
        "RBAL_WARMUP_END_EPOCH": 30,
        "RBAL_WARMUP_START_SCALE": 0.0,
    },
    # Table 4: auxiliary objective only.  SURFACE/FULL is reused.
    "T4_BOUNDARY": {"RBAL_AUX_LOSS_TYPE": "boundary"},
    "T4_HD": {"RBAL_AUX_LOSS_TYPE": "hausdorff_dt"},
    "T4_ACTIVE_CONTOUR": {"RBAL_AUX_LOSS_TYPE": "active_contour"},
    # Table 5: radius only.  R1/FULL is reused.
    "T5_R2": {"RBAL_BOUNDARY_RADIUS_PX": 2},
    "T5_R3": {"RBAL_BOUNDARY_RADIUS_PX": 3},
    "T5_R5": {"RBAL_BOUNDARY_RADIUS_PX": 5},
    # Table 6: initial auxiliary weight only.  L005/FULL is reused.
    "T6_L000": {"RBAL_EDGE_WEIGHT": 0.0},
    "T6_L010": {"RBAL_EDGE_WEIGHT": 0.10},
    "T6_L015": {"RBAL_EDGE_WEIGHT": 0.15},
    "T6_L020": {"RBAL_EDGE_WEIGHT": 0.20},
    "T6_L025": {"RBAL_EDGE_WEIGHT": 0.25},
}


COMMON = {
    "RBAL_AUX_LOSS_TYPE": "surface",
    "RBAL_HD_ALPHA": 2.0,
    "RBAL_ACTIVE_CONTOUR_LENGTH_WEIGHT": 1.0,
    "RBAL_ACTIVE_CONTOUR_REGION_WEIGHT": 1.0,
    "RBAL_WARMUP_END_EPOCH": 30,
    "RBAL_WARMUP_START_SCALE": 0.0,
}


def main() -> None:
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    for arm, override in ARMS.items():
        cfg = copy.deepcopy(base)
        cfg["TRAIN"].update(COMMON)
        cfg["TRAIN"].update(override)
        tag = f"S36_BUSI_{arm}_PAPER100"
        cfg["TRAIN"]["RUN_TAG"] = tag
        cfg["M1"]["RUN_TAG"] = tag
        path = OUT / f"{arm}.yaml"
        path.write_text(
            yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
