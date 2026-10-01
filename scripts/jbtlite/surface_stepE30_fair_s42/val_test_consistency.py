#!/usr/bin/env python3
"""Validation -> Test selection-reliability audit.

THE QUESTION THIS ANSWERS
-------------------------
"How do you guarantee that the checkpoint that looked best on validation is
also the best on test?"

You cannot guarantee it, and any procedure that tried to would be test-set
tuning. What you *can* do is measure, honestly and after the fact, how well
validation ranking predicted test ranking across the arms of this study. That
measurement is itself a reportable result, and it is the correct scientific
answer to the question.

This script computes, across all arms that have BOTH a validation (MC10) and a
locked test (MC30) result:
  * Spearman rank correlation between the validation metric and the test metric,
  * Kendall's tau (more robust for a small number of arms),
  * the rank each arm held on validation versus on test,
  * whether the validation-best arm was also the test-best arm.

A high correlation supports "validation selection was informative here."
A low correlation is also a legitimate finding and should be reported as a
limitation rather than hidden. Either way, nothing in this script changes
which checkpoint was selected: selection already happened, on validation only.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

METRIC = re.compile(r"Average (DSC|NSD).*?:\s*([0-9.]+)%")


def read_log(log: Path) -> dict[str, float] | None:
    if not log.is_file():
        return None
    vals = {k: float(v) for k, v in METRIC.findall(log.read_text(errors="replace"))}
    return vals if {"DSC", "NSD"} <= set(vals) else None


def run_metrics(run_root: Path, split_tag: str, nsd_mode: str) -> dict[str, float] | None:
    return read_log(run_root / "logs" / f"eval_{split_tag}_{nsd_mode}.log")


def ranks(values: list[float]) -> list[float]:
    """Average ranks, descending (rank 1 = largest value = best)."""
    order = sorted(range(len(values)), key=lambda i: -values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def spearman(x: list[float], y: list[float]) -> float:
    rx, ry = ranks(x), ranks(y)
    n = len(rx)
    if n < 2:
        return float("nan")
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def kendall_tau(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            sx = (x[i] > x[j]) - (x[i] < x[j])
            sy = (y[i] > y[j]) - (y[i] < y[j])
            prod = sx * sy
            if prod > 0:
                conc += 1
            elif prod < 0:
                disc += 1
    total = conc + disc
    return (conc - disc) / total if total else float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--new-result-root", default=None)
    p.add_argument("--new-study-id", required=True)
    p.add_argument("--anchor-result-root", default=None)
    p.add_argument("--anchor-study-id", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nsd-mode", default="true2d", choices=["true2d", "paper_legacy"])
    a = p.parse_args()

    project = Path(a.project)
    new_root = Path(a.new_result_root) if a.new_result_root else project / "surface_stepE30_fair_s42_results"
    anchor_root = Path(a.anchor_result_root) if a.anchor_result_root else project / "surface_ablation_redesign_v2_results"

    # Arms that legitimately have BOTH val and locked test in this design.
    candidates: list[tuple[str, Path]] = []
    for arm in ["BASE", "LOCAL_ALWAYS", "FULL_STEP_E30"]:
        candidates.append((arm, anchor_root / a.anchor_study_id / "BUSI" / arm / f"seed{a.seed}"))
    for arm in ["DECODER_ONLY_STEP_E30", "LOSS_BOUNDARY_STEP_E30", "LOSS_HD_STEP_E30",
                "LOSS_ACTIVE_CONTOUR_STEP_E30", "SCHED_COSINE", "SCHED_WARMUP_E30"]:
        candidates.append((arm, new_root / a.new_study_id / "BUSI" / arm / f"seed{a.seed}"))

    rows = []
    for arm, run in candidates:
        v = run_metrics(run, "val_mc10", a.nsd_mode)
        t = run_metrics(run, "test_mc30", a.nsd_mode)
        if v is None or t is None:
            print(f"[skip] {arm}: val={'ok' if v else 'missing'} test={'ok' if t else 'missing'}")
            continue
        rows.append({"arm": arm, "val_dsc": v["DSC"], "val_nsd": v["NSD"],
                     "test_dsc": t["DSC"], "test_nsd": t["NSD"]})

    if len(rows) < 3:
        raise SystemExit(f"[FAIL] need >=3 arms with both val and test; found {len(rows)}")

    for key_v, key_t, label in (("val_dsc", "test_dsc", "DSC"), ("val_nsd", "test_nsd", "NSD")):
        xs = [r[key_v] for r in rows]
        ys = [r[key_t] for r in rows]
        rv, rt = ranks(xs), ranks(ys)
        for r, a_, b_ in zip(rows, rv, rt):
            r[f"val_rank_{label}"] = a_
            r[f"test_rank_{label}"] = b_
        r_s = spearman(xs, ys)
        tau = kendall_tau(xs, ys)
        best_val = rows[max(range(len(rows)), key=lambda i: xs[i])]["arm"]
        best_test = rows[max(range(len(rows)), key=lambda i: ys[i])]["arm"]
        print(f"\n=== {label} ({a.nsd_mode}, seed {a.seed}, {len(rows)} arms) ===")
        print(f"Spearman(val, test) = {r_s:.3f}   Kendall tau = {tau:.3f}")
        print(f"validation-best arm = {best_val}")
        print(f"test-best arm       = {best_test}")
        print("agreement: " + ("YES" if best_val == best_test else
                               "NO  (validation selection did not pick the test-best arm)"))
        print(f"{'arm':32s} {'val':>8s} {'test':>8s} {'val_rk':>7s} {'test_rk':>8s}")
        for r in sorted(rows, key=lambda d: d[f"val_rank_{label}"]):
            print(f"{r['arm']:32s} {r[key_v]:8.2f} {r[key_t]:8.2f} "
                  f"{r[f'val_rank_{label}']:7.1f} {r[f'test_rank_{label}']:8.1f}")

    out_dir = new_root / a.new_study_id / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"val_test_consistency_{a.nsd_mode}.csv"
    fields = ["arm", "val_dsc", "test_dsc", "val_rank_DSC", "test_rank_DSC",
              "val_nsd", "test_nsd", "val_rank_NSD", "test_rank_NSD"]
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWritten: {out_csv}")


if __name__ == "__main__":
    main()
