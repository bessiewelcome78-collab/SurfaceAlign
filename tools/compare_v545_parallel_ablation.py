#!/usr/bin/env python3
"""Compare V545 root-causal ablations and select a winner without weighted scoring."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from statistics import mean

NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def avg_tail(rows, key, tail_n):
    vals = [r[key] for r in rows[-max(1, tail_n):] if key in r]
    return mean(vals) if vals else float("nan")


def parse(path: Path, tail_n: int):
    diag = []
    val = []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("M1_DIAG:"):
            diag.append({k: float(v) for k, v in re.findall(rf"([A-Za-z0-9_]+)=({NUMBER})", line)})
        elif line.startswith("VAL_NATIVE epoch="):
            m = re.search(
                rf"base DSC/NSD=({NUMBER})/({NUMBER}).*componentOracle DSC/NSD=({NUMBER})/({NUMBER})",
                line,
            )
            if m:
                b_d, b_n, o_d, o_n = map(float, m.groups())
                val.append({"base": b_d, "base_nsd": b_n, "comp": o_d, "comp_nsd": o_n})
    if not diag:
        raise RuntimeError(f"No M1_DIAG rows: {path}")
    if not val:
        raise RuntimeError(f"No VAL_NATIVE rows: {path}")
    return {
        "name": path.stem,
        "path": str(path),
        "purity": avg_tail(diag, "v538_component_purity", tail_n),
        "capture": avg_tail(diag, "v538_component_capture_ratio", tail_n),
        "mask_oracle": avg_tail(diag, "v544_mask_oracle_gain", tail_n),
        "full_oracle": avg_tail(diag, "v544_full_oracle_gain", tail_n),
        "benefit_sign": avg_tail(diag, "v544_benefit_gain_positive_rate_global", tail_n),
        "harm_sign": avg_tail(diag, "v544_harm_gain_negative_rate_global", tail_n),
        "signed_benefit": avg_tail(diag, "v544_signed_outcome_mean_on_benefit_global", tail_n),
        "zero_benefit": avg_tail(diag, "v544_zero_benefit_batch_rate", tail_n),
        "utility_gain": avg_tail(diag, "v545_utility_gain_mean", tail_n),
        "direction_active": avg_tail(diag, "v545_direction_update_active", tail_n),
        "val_component_gain": mean([(r["comp"] - r["base"]) for r in val[-max(1, tail_n):]]),
        "val_component_nsd_gain": mean([(r["comp_nsd"] - r["base_nsd"]) for r in val[-max(1, tail_n):]]),
    }


def ge(a, b):
    return a >= b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--tail", type=int, default=3)
    args = ap.parse_args()
    rows = [parse(Path(x), args.tail) for x in args.logs]
    print("=" * 176)
    print("V545 root-causal parallel ablation")
    print("=" * 176)
    labels = [
        ("purity", "Purity"), ("capture", "Capture"),
        ("mask_oracle", "MaskOra"), ("full_oracle", "FullOra"),
        ("benefit_sign", "BenSign"), ("harm_sign", "HarmSign"),
        ("signed_benefit", "SignedBen"), ("zero_benefit", "ZeroBen"),
        ("utility_gain", "UtilGain"), ("direction_active", "DirActive"),
        ("val_component_gain", "ValComp"), ("val_component_nsd_gain", "ValNSD"),
    ]
    print(f"{'Run':<42}" + ''.join(f"{lab:>11}" for _, lab in labels))
    for r in rows:
        print(f"{r['name'][-40:]:<42}" + ''.join(f"{r[k]:>11.5f}" for k, _ in labels))

    if len(rows) < 6:
        print("\n[NO AUTO SELECTION] Expected C0..C5 logs.")
        return
    c0, c1, c2, c3, c4, c5 = rows[:6]
    print("\n[Causal checks]")
    checks = {
        "C1 minimal M2 isolated": (
            c1["benefit_sign"] > c0["benefit_sign"]
            and c1["signed_benefit"] > c0["signed_benefit"]
        ),
        "C2 class-complete direction": (
            c2["benefit_sign"] > c1["benefit_sign"]
            and c2["signed_benefit"] > c1["signed_benefit"]
        ),
        "C3 exact train/deploy alignment": (
            c3["full_oracle"] > c0["full_oracle"]
            and c3["val_component_gain"] > c0["val_component_gain"]
        ),
        "C4 matched locality": (
            c4["purity"] > c3["purity"]
            and c4["full_oracle"] >= c3["full_oracle"]
        ),
        "C5 combined root fix": (
            c5["full_oracle"] > c0["full_oracle"]
            and c5["val_component_gain"] > c0["val_component_gain"]
            and c5["benefit_sign"] > c1["benefit_sign"]
            and c5["signed_benefit"] > c1["signed_benefit"]
        ),
    }
    for name, passed in checks.items():
        print(f"{name:<36}: {'PASS' if passed else 'FAIL'}")

    # No weighted composite and no manually chosen tolerance.  A candidate may
    # win only if it improves both the actual candidate ceiling and Benefit
    # direction relative to the corresponding controls.  Among eligible rows,
    # lexicographic selection uses deployment-relevant metrics only.
    eligible = [
        r for r in rows
        if ge(r["full_oracle"], c0["full_oracle"])
        and ge(r["val_component_gain"], c0["val_component_gain"])
        and ge(r["benefit_sign"], c1["benefit_sign"])
        and ge(r["signed_benefit"], c1["signed_benefit"])
    ]
    print("\n[Automatic selection]")
    if not eligible:
        print("V545_AUTO_SELECTED=NONE")
        print("No variant simultaneously improved candidate utility and Benefit direction.")
        return
    winner = max(
        eligible,
        key=lambda r: (
            r["val_component_gain"],
            r["full_oracle"],
            r["benefit_sign"],
            r["signed_benefit"],
        ),
    )
    print(f"V545_AUTO_SELECTED={winner['name']}")
    print(f"V545_AUTO_SELECTED_LOG={winner['path']}")


if __name__ == "__main__":
    main()
