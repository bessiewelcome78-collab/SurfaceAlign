#!/usr/bin/env python3
"""Compare a complete 2^3 V546F factorial ablation without weighted scoring."""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from statistics import mean

NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
BITS = tuple(f"{i:03b}" for i in range(8))
MODULES = {0: "OptimalMatch", 1: "SlotCompetition", 2: "BalancedSoftmax"}


def parse_diag(line: str) -> dict[str, float]:
    return {key: float(value) for key, value in re.findall(rf"([A-Za-z0-9_]+)=({NUMBER})", line)}


def parse_log(path: Path, tail: int) -> dict:
    text = path.read_text(errors="replace")
    tag_match = re.search(r"RUN_TAG:\s*(V546F_F([01]{3})_[A-Za-z0-9_]+)", text)
    if not tag_match:
        tag_match = re.search(r"V546F_F([01]{3})_[A-Za-z0-9_]+", path.name)
        if not tag_match:
            raise RuntimeError(f"Cannot identify factorial bits from {path}")
        bits = tag_match.group(1)
        tag = tag_match.group(0)
    else:
        tag, bits = tag_match.group(1), tag_match.group(2)

    seed_match = re.search(r"seed(\d+)", path.name)
    seed = int(seed_match.group(1)) if seed_match else -1

    diag_by_epoch: dict[int, dict[str, float]] = {}
    val_by_epoch: dict[int, dict[str, float]] = {}
    shadow_by_epoch: dict[int, dict[str, float]] = {}
    improved_epochs: list[int] = []
    current_epoch: int | None = None
    for line in text.splitlines():
        epoch_match = re.search(r"EPOCH:\s*(\d+)\s*\|", line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
        elif line.startswith("M1_DIAG:") and current_epoch is not None:
            diag_by_epoch[current_epoch] = parse_diag(line)
        elif line.startswith("VAL_NATIVE epoch="):
            match = re.search(
                rf"VAL_NATIVE epoch=(\d+).*?base DSC/NSD=({NUMBER})/({NUMBER}).*?"
                rf"componentOracle DSC/NSD=({NUMBER})/({NUMBER}).*?qualifies=(\w+)",
                line,
            )
            if match:
                epoch = int(match.group(1))
                base_d, base_n, comp_d, comp_n = map(float, match.groups()[1:5])
                val_by_epoch[epoch] = {
                    "base_dice": base_d,
                    "base_nsd": base_n,
                    "component_dice": comp_d,
                    "component_nsd": comp_n,
                    "component_gain": comp_d - base_d,
                    "component_nsd_gain": comp_n - base_n,
                    "qualifies": 1.0 if match.group(6) == "True" else 0.0,
                }
        elif line.startswith("V546_SHADOW_VAL epoch="):
            match = re.search(
                rf"epoch=(\d+).*?shadowM2 DSC/NSD=({NUMBER})/({NUMBER})\s+"
                rf"gain=({NUMBER})/({NUMBER})\s+\| cat=({NUMBER})",
                line,
            )
            if match:
                epoch = int(match.group(1))
                shadow_by_epoch[epoch] = {
                    "shadow_dice": float(match.group(2)),
                    "shadow_nsd": float(match.group(3)),
                    "shadow_gain": float(match.group(4)),
                    "shadow_nsd_gain": float(match.group(5)),
                    "shadow_cat": float(match.group(6)),
                }
        elif line.startswith("VAL epoch=") and "improved=True" in line:
            match = re.search(r"VAL epoch=(\d+)", line)
            if match:
                improved_epochs.append(int(match.group(1)))

    if not diag_by_epoch or not val_by_epoch or not shadow_by_epoch:
        raise RuntimeError(f"Incomplete formal log: {path}")

    safe_epoch = improved_epochs[-1] if improved_epochs else None
    report_epoch = safe_epoch if safe_epoch is not None else max(set(diag_by_epoch) & set(val_by_epoch) & set(shadow_by_epoch))
    diag = diag_by_epoch.get(report_epoch, diag_by_epoch[max(diag_by_epoch)])
    val = val_by_epoch.get(report_epoch, val_by_epoch[max(val_by_epoch)])
    shadow = shadow_by_epoch.get(report_epoch, shadow_by_epoch[max(shadow_by_epoch)])

    tail_epochs = sorted(diag_by_epoch)[-max(1, tail):]
    def tail_mean(key: str) -> float:
        values = [diag_by_epoch[e][key] for e in tail_epochs if key in diag_by_epoch[e]]
        return mean(values) if values else float("nan")

    return {
        "name": tag,
        "bits": bits,
        "seed": seed,
        "path": str(path),
        "safe": safe_epoch is not None,
        "epoch": report_epoch,
        "shadow_gain": shadow["shadow_gain"],
        "shadow_nsd_gain": shadow["shadow_nsd_gain"],
        "shadow_cat": shadow["shadow_cat"],
        "component_gain": val["component_gain"],
        "component_nsd_gain": val["component_nsd_gain"],
        "full_oracle": diag.get("v544_full_oracle_gain", float("nan")),
        "mask_oracle": diag.get("v544_mask_oracle_gain", float("nan")),
        "capture": diag.get("v538_component_capture_ratio", float("nan")),
        "purity": diag.get("v538_component_purity", float("nan")),
        "benefit_sign": diag.get("v544_benefit_gain_positive_rate_global", float("nan")),
        "harm_sign": diag.get("v544_harm_gain_negative_rate_global", float("nan")),
        "balanced_sign": diag.get("v544_balanced_sign_accuracy_global", float("nan")),
        "signed_benefit": diag.get("v544_signed_outcome_mean_on_benefit_global", float("nan")),
        "signed_harm": diag.get("v544_signed_outcome_mean_on_harm_global", float("nan")),
        "zero_benefit": diag.get("v544_zero_benefit_batch_rate", float("nan")),
        "matching_gain": tail_mean("v546_matching_gain"),
        "raw_overlap": tail_mean("v546_raw_slot_overlap_mass"),
        "projected_overlap": tail_mean("v546_competition_overlap_mass"),
    }


def module_pairs(bit_index: int) -> list[tuple[str, str]]:
    pairs = []
    for bits in BITS:
        if bits[bit_index] == "0":
            enabled = bits[:bit_index] + "1" + bits[bit_index + 1:]
            pairs.append((bits, enabled))
    return pairs


def summarize_effect(rows: dict[str, dict], bit_index: int, metric: str, higher_is_better: bool = True) -> tuple[list[float], float]:
    effects = []
    for off, on in module_pairs(bit_index):
        raw = rows[on][metric] - rows[off][metric]
        effects.append(raw if higher_is_better else -raw)
    return effects, mean(effects)


def dominates(a: dict, b: dict) -> bool:
    higher = ("shadow_gain", "shadow_nsd_gain", "component_gain", "full_oracle")
    lower = ("shadow_cat",)
    no_worse = all(a[k] >= b[k] for k in higher) and all(a[k] <= b[k] for k in lower)
    strict = any(a[k] > b[k] for k in higher) or any(a[k] < b[k] for k in lower)
    return no_worse and strict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--tail", type=int, default=5)
    args = parser.parse_args()
    parsed = [parse_log(Path(p), args.tail) for p in args.logs]
    rows = {row["bits"]: row for row in parsed}
    missing = [b for b in BITS if b not in rows]
    if missing:
        raise SystemExit(f"Missing factorial cells: {missing}")

    print("=" * 198)
    print("V546F complete 2^3 factorial ablation")
    print("=" * 198)
    headers = (
        ("safe", "Safe"), ("epoch", "Epoch"),
        ("shadow_gain", "ShDSC"), ("shadow_nsd_gain", "ShNSD"),
        ("component_gain", "CompOra"), ("full_oracle", "TrainOra"),
        ("capture", "Capture"), ("purity", "Purity"),
        ("benefit_sign", "BenSign"), ("harm_sign", "HarmSign"),
        ("signed_benefit", "SignedB"), ("shadow_cat", "Cat"),
    )
    print(f"{'Cell':<7}{'Run':<43}" + "".join(f"{label:>11}" for _, label in headers))
    for bits in BITS:
        row = rows[bits]
        vals = []
        for key, _ in headers:
            value = row[key]
            if isinstance(value, bool): vals.append(f"{'Y' if value else 'N':>11}")
            elif isinstance(value, int): vals.append(f"{value:>11d}")
            else: vals.append(f"{value:>11.6f}")
        print(f"F{bits:<6}{row['name'][:41]:<43}" + "".join(vals))

    print("\n[Factorial causal contrasts: positive means improvement]")
    metrics = (
        ("shadow_gain", True, "Shadow DSC"),
        ("shadow_nsd_gain", True, "Shadow NSD"),
        ("component_gain", True, "Component Oracle"),
        ("full_oracle", True, "Train Full Oracle"),
        ("balanced_sign", True, "Balanced Sign"),
        ("shadow_cat", False, "Catastrophic safety"),
    )
    for idx, module in MODULES.items():
        print(f"\n{module}:")
        for metric, higher, label in metrics:
            effects, avg = summarize_effect(rows, idx, metric, higher)
            signs = sum(v > 0 for v in effects)
            print(f"  {label:<22} effects=" + ",".join(f"{v:+.6f}" for v in effects) + f" | mean={avg:+.6f} | positive={signs}/4")

        primary, primary_mean = summarize_effect(rows, idx, "shadow_gain", True)
        safety, safety_mean = summarize_effect(rows, idx, "shadow_cat", False)
        if all(x >= -1e-8 for x in primary) and primary_mean > 0 and all(x >= -1e-8 for x in safety):
            signal = "FAVORABLE_IN_THIS_SEED"
        elif primary_mean <= 0:
            signal = "UNFAVORABLE_IN_THIS_SEED"
        else:
            signal = "MIXED_INTERACTION"
        print(f"  V546F_{module.upper()}_SEED_SIGNAL={signal}")

    eligible = [
        r for r in rows.values()
        if r["safe"] and r["shadow_gain"] > 0 and r["shadow_nsd_gain"] >= 0
        and r["component_gain"] > 0 and math.isfinite(r["full_oracle"])
    ]
    print("\n[Pareto selection; no weighted score]")
    if not eligible:
        print("V546F_AUTO_SELECTED=NONE")
        return
    front = [r for r in eligible if not any(dominates(o, r) for o in eligible if o is not r)]
    if len(front) == 1:
        w = front[0]
        print(f"V546F_AUTO_SELECTED={w['name']}")
        print(f"V546F_AUTO_SELECTED_LOG={w['path']}")
        print(f"V546F_AUTO_SELECTED_EPOCH={w['epoch']}")
    else:
        print("V546F_AUTO_SELECTED=AMBIGUOUS")
        print("V546F_PARETO_FRONT=" + ",".join(r["name"] for r in sorted(front, key=lambda x: x["bits"])))


if __name__ == "__main__":
    main()
