#!/usr/bin/env python3
"""Isolate M2's effect on the catastrophic subset, from the per-case CSVs
test.py already writes -- no training-loop changes, no re-running anything.

Why this exists
----------------
The dataset-mean `topoedit_minus_m1` comparison (in *_comparison.json) is
dominated by the 80-97% of cases that are already fine, where the local-edit
ceiling is ~0.001-0.002 DSC and easily drowned out by noise. If a redesign
(macro actions, case-weighted training) is working, its signature shows up
FIRST and MOST CLEARLY on the catastrophic subset specifically -- this script
computes exactly that, by joining the Base/M1/TopoEdit per-case CSVs on case
ID and filtering to cases where M1's DSC is below a threshold (default 0.50,
matching TRAIN.VAL_CATASTROPHIC_DSC_THRESHOLD).

Usage
-----
    python tools/analyze_catastrophic_subset.py \\
        --m1 outputs_topoedit_v2diag3/BUSI/seg_results/seed42/BUSI_M1_test_true2d.csv \\
        --m2 outputs_topoedit_v2diag3/BUSI/seg_results/seed42/BUSI_TopoEditV2_test_true2d.csv \\
        --threshold 0.50
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict


def _read_dice_by_case(path: Path) -> Dict[str, float]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        case_col = next(
            (c for c in fields if c.lower() in ("case", "case_id", "name", "image", "id")),
            fields[0] if fields else None,
        )
        dice_col = next(
            (c for c in fields if "dsc" in c.lower() or "dice" in c.lower()), None
        )
        if case_col is None or dice_col is None:
            raise ValueError(f"{path}: could not find case/DSC columns in {fields}")
        out: Dict[str, float] = {}
        for row in reader:
            key = row[case_col]
            try:
                out[key] = float(row[dice_col])
            except (TypeError, ValueError):
                continue
        return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m1", required=True, help="M1-only per-case CSV")
    parser.add_argument("--m2", required=True, help="M1+M2 (TopoEdit) per-case CSV")
    parser.add_argument(
        "--threshold", type=float, default=0.50,
        help="M1 DSC below this = catastrophic (default matches "
             "TRAIN.VAL_CATASTROPHIC_DSC_THRESHOLD)",
    )
    args = parser.parse_args()

    m1_scores = _read_dice_by_case(Path(args.m1))
    m2_scores = _read_dice_by_case(Path(args.m2))
    shared = sorted(set(m1_scores) & set(m2_scores))
    if not shared:
        print("[FAIL] no shared case IDs between the two CSVs -- check the "
              "case/name column, or that both files are from the same run.")
        return 1

    all_gains = [m2_scores[k] - m1_scores[k] for k in shared]
    catastrophic = [k for k in shared if m1_scores[k] < args.threshold]
    normal = [k for k in shared if k not in catastrophic]

    def _report(label: str, keys) -> None:
        if not keys:
            print(f"{label}: 0 cases")
            return
        gains = [m2_scores[k] - m1_scores[k] for k in keys]
        m1_mean = sum(m1_scores[k] for k in keys) / len(keys)
        m2_mean = sum(m2_scores[k] for k in keys) / len(keys)
        beneficial = sum(1 for g in gains if g > 1e-9)
        harmful = sum(1 for g in gains if g < -1e-9)
        equal = len(keys) - beneficial - harmful
        print(
            f"{label}: n={len(keys)}  M1_mean={m1_mean:.4f}  M2_mean={m2_mean:.4f}  "
            f"mean_gain={sum(gains)/len(gains):+.4f}  "
            f"beneficial={beneficial}  harmful={harmful}  equal={equal}"
        )

    print(f"threshold (catastrophic if M1 DSC < {args.threshold}):")
    print(f"  catastrophic cases: {len(catastrophic)} / {len(shared)} "
          f"({100*len(catastrophic)/len(shared):.1f}%)")
    print()
    _report("ALL CASES        ", shared)
    _report("CATASTROPHIC ONLY", catastrophic)
    _report("NORMAL ONLY      ", normal)
    print()

    if catastrophic:
        cat_contribution = sum(m2_scores[k] - m1_scores[k] for k in catastrophic) / len(shared)
        normal_contribution = sum(m2_scores[k] - m1_scores[k] for k in normal) / len(shared) if normal else 0.0
        print("dataset-level DSC contribution (mean_gain * subset_share):")
        print(f"  from catastrophic subset: {cat_contribution:+.5f}")
        print(f"  from normal subset:       {normal_contribution:+.5f}")
        print(f"  total (should match ALL CASES mean_gain): "
              f"{cat_contribution + normal_contribution:+.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
