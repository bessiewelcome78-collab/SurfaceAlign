#!/usr/bin/env python3
"""Compare controlled V544 A0/A1/A2/A3 logs without post-hoc metric changes."""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from statistics import mean

NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def parse_log(path: Path, tail_n: int):
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("M1_DIAG:"):
            rows.append({
                k: float(v)
                for k, v in re.findall(rf"([A-Za-z0-9_]+)=({NUMBER})", line)
            })
    if not rows:
        raise RuntimeError(f"No M1_DIAG rows in {path}")
    tail = rows[-max(1, tail_n):]
    def a(key):
        vals = [r[key] for r in tail if key in r]
        return mean(vals) if vals else float("nan")
    return {
        "log": str(path),
        "purity": a("v538_component_purity"),
        "capture": a("v538_component_capture_ratio"),
        "mask_oracle": a("v544_mask_oracle_gain"),
        "full_oracle": a("v544_full_oracle_gain"),
        "benefit_sign": a("v544_benefit_gain_positive_rate_global"),
        "harm_sign": a("v544_harm_gain_negative_rate_global"),
        "signed_benefit": a("v544_signed_outcome_mean_on_benefit_global"),
        "zero_benefit": a("v544_zero_benefit_batch_rate"),
        "m1_grad": a("v538_m1_grad_norm_preclip"),
        "m2_grad": a("v538_m2_grad_norm_preclip"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--tail", type=int, default=3)
    args = parser.parse_args()
    rows = [parse_log(Path(p), args.tail) for p in args.logs]
    columns = [
        ("purity", "Purity"),
        ("capture", "Capture"),
        ("mask_oracle", "MaskOra"),
        ("full_oracle", "FullOra"),
        ("benefit_sign", "BenSign"),
        ("harm_sign", "HarmSign"),
        ("signed_benefit", "SignedBen"),
        ("zero_benefit", "ZeroBenB"),
        ("m1_grad", "M1Grad"),
        ("m2_grad", "M2Grad"),
    ]
    print("=" * 150)
    print("V544 controlled ablation comparison")
    print("=" * 150)
    print(f"{'Run':<38}" + "".join(f"{label:>11}" for _, label in columns))
    for row in rows:
        name = Path(row["log"]).stem[-36:]
        print(f"{name:<38}" + "".join(f"{row[key]:>11.5f}" for key, _ in columns))

    if len(rows) >= 3:
        a0, a1, a2 = rows[:3]
        mask_improved = (
            a1["purity"] > a0["purity"]
            and a1["full_oracle"] >= a0["full_oracle"]
        )
        selector_improved = (
            a2["benefit_sign"] > a1["benefit_sign"]
            and a2["harm_sign"] >= a1["harm_sign"] - 0.05
            and a2["signed_benefit"] > a1["signed_benefit"]
        )
        print()
        print(f"A1 region-mask causal improvement: {'PASS' if mask_improved else 'FAIL'}")
        print(f"A2 minimal-M2 causal improvement : {'PASS' if selector_improved else 'FAIL'}")
        if len(rows) >= 4:
            a3 = rows[3]
            replay_needed = a2["zero_benefit"] > 0.30
            replay_helped = (
                a3["benefit_sign"] > a2["benefit_sign"]
                and a3["harm_sign"] >= a2["harm_sign"] - 0.05
            )
            print(f"A3 replay was indicated          : {'YES' if replay_needed else 'NO'}")
            print(f"A3 replay causal improvement     : {'PASS' if replay_helped else 'FAIL'}")


if __name__ == "__main__":
    main()
