#!/usr/bin/env python3
"""Paired per-case statistics for the seed-42 ablation.

WHY THIS EXISTS
---------------
With one seed and a 78-image BUSI test set, a bare "86.65 vs 87.15" comparison
carries no uncertainty information: the reader cannot tell a real effect from
sampling noise. But every arm is evaluated on *exactly the same 78 cases*, so
the comparison is naturally PAIRED. That turns "one number vs one number" into
78 paired observations, which supports a real significance test and a real
confidence interval without needing extra training seeds.

This script therefore reports, for each arm versus the FULL_STEP_E30 anchor:
  * mean paired difference (arm - anchor) over the 78 shared cases,
  * a BCa-free percentile bootstrap 95% CI of that mean difference,
  * a two-sided Wilcoxon signed-rank p-value (exact-ish, via scipy if present,
    else a normal approximation implemented here),
  * the number of cases that improved / worsened / tied.

WHAT IT DOES NOT DO
-------------------
It does not, and cannot, tell you which arm will be best on the test set ahead
of time. Checkpoint selection stays on validation. This script only quantifies
how confident you may be about a difference you already committed to measuring.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

ANCHOR_ARM = "FULL_STEP_E30"


def read_cases(csv_path: Path) -> dict[str, tuple[float, float]]:
    """Return {case_id: (DSC, NSD)} from one per-case eval CSV."""
    if not csv_path.is_file():
        return {}
    out: dict[str, tuple[float, float]] = {}
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            key = (rec.get("Case_ID") or rec.get("Name") or "").strip()
            if not key:
                continue
            try:
                out[key] = (float(rec["DSC"]), float(rec["NSD"]))
            except (KeyError, ValueError):
                continue
    return out


def find_case_csv(run_root: Path, arm: str, split_tag: str, nsd_mode: str) -> Path | None:
    """Locate the per-case CSV written by utils/eval.py for one run."""
    split = "test" if split_tag.startswith("test") else "val"
    stem = f"{arm.lower()}_{split_tag}_{nsd_mode}.csv"
    direct = run_root / split / "BUSI" / "seg_results"
    for cand in list(direct.rglob(stem)) + list((run_root / split).rglob(stem)):
        if cand.is_file():
            return cand
    hits = sorted((run_root / split).rglob(f"*{split_tag}_{nsd_mode}.csv"))
    return hits[0] if hits else None


def percentile_ci(samples: list[float], alpha: float = 0.05) -> tuple[float, float]:
    ordered = sorted(samples)
    n = len(ordered)
    lo = ordered[max(0, int(math.floor((alpha / 2) * n)) - 1)]
    hi = ordered[min(n - 1, int(math.ceil((1 - alpha / 2) * n)) - 1)]
    return lo, hi


def bootstrap_mean_diff(diffs: list[float], iters: int, rng: random.Random) -> tuple[float, float]:
    n = len(diffs)
    if n == 0:
        return float("nan"), float("nan")
    means = []
    for _ in range(iters):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    return percentile_ci(means)


def wilcoxon_signed_rank(diffs: list[float]) -> float:
    """Two-sided Wilcoxon signed-rank p-value. Uses scipy when available."""
    nz = [d for d in diffs if d != 0.0]
    if len(nz) < 1:
        return float("nan")
    try:
        from scipy.stats import wilcoxon  # type: ignore
        return float(wilcoxon(nz, alternative="two-sided").pvalue)
    except Exception:
        pass
    # Normal approximation with tie correction.
    n = len(nz)
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(r for r, d in zip(ranks, nz) if d > 0)
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    if var_w <= 0:
        return float("nan")
    z = (w_plus - mean_w) / math.sqrt(var_w)
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def compare(arm_cases, anchor_cases, index: int):
    shared = sorted(set(arm_cases) & set(anchor_cases))
    diffs = [arm_cases[c][index] - anchor_cases[c][index] for c in shared]
    return shared, diffs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--new-result-root", default=None)
    p.add_argument("--new-study-id", required=True)
    p.add_argument("--anchor-result-root", default=None)
    p.add_argument("--anchor-study-id", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-tag", default="test_mc30", choices=["test_mc30", "val_mc10"])
    p.add_argument("--nsd-mode", default="true2d", choices=["true2d", "paper_legacy"])
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--arms", default="")
    a = p.parse_args()

    project = Path(a.project)
    new_root = Path(a.new_result_root) if a.new_result_root else project / "surface_stepE30_fair_s42_results"
    anchor_root = Path(a.anchor_result_root) if a.anchor_result_root else project / "surface_ablation_redesign_v2_results"
    rng = random.Random(20260921)

    anchor_run = anchor_root / a.anchor_study_id / "BUSI" / ANCHOR_ARM / f"seed{a.seed}"
    anchor_csv = find_case_csv(anchor_run, ANCHOR_ARM, a.split_tag, a.nsd_mode)
    if anchor_csv is None:
        raise SystemExit(
            f"[FAIL] anchor per-case CSV not found under {anchor_run}.\n"
            f"       The {ANCHOR_ARM} arm must have completed {a.split_tag} first."
        )
    anchor_cases = read_cases(anchor_csv)
    print(f"[anchor] {ANCHOR_ARM} seed{a.seed}: {len(anchor_cases)} cases from {anchor_csv}")

    if a.arms.strip():
        arms = [s.strip() for s in a.arms.split(",") if s.strip()]
    else:
        arms = sorted(
            d.name for d in (new_root / a.new_study_id / "BUSI").iterdir() if d.is_dir()
        ) if (new_root / a.new_study_id / "BUSI").is_dir() else []

    rows = []
    for arm in arms:
        run = new_root / a.new_study_id / "BUSI" / arm / f"seed{a.seed}"
        csv_path = find_case_csv(run, arm, a.split_tag, a.nsd_mode)
        if csv_path is None:
            print(f"[skip] {arm}: no per-case CSV yet")
            continue
        arm_cases = read_cases(csv_path)
        for metric_name, idx in (("DSC", 0), ("NSD", 1)):
            shared, diffs = compare(arm_cases, anchor_cases, idx)
            if not diffs:
                continue
            mean_d = sum(diffs) / len(diffs)
            lo, hi = bootstrap_mean_diff(diffs, a.bootstrap, rng)
            pval = wilcoxon_signed_rank(diffs)
            better = sum(1 for d in diffs if d > 0)
            worse = sum(1 for d in diffs if d < 0)
            tie = sum(1 for d in diffs if d == 0)
            rows.append({
                "arm": arm,
                "metric": metric_name,
                "n_paired_cases": len(shared),
                "mean_diff_vs_anchor": f"{mean_d * 100:+.2f}",
                "ci95_low": f"{lo * 100:+.2f}",
                "ci95_high": f"{hi * 100:+.2f}",
                "wilcoxon_p": f"{pval:.4g}",
                "cases_better": better,
                "cases_worse": worse,
                "cases_tied": tie,
                "significant_at_0.05": "yes" if (pval == pval and pval < 0.05) else "no",
            })

    out_dir = new_root / a.new_study_id / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"paired_stats_vs_anchor_{a.split_tag}_{a.nsd_mode}.csv"
    fields = ["arm", "metric", "n_paired_cases", "mean_diff_vs_anchor", "ci95_low",
              "ci95_high", "wilcoxon_p", "cases_better", "cases_worse", "cases_tied",
              "significant_at_0.05"]
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\nPaired comparison vs {ANCHOR_ARM} ({a.split_tag}, {a.nsd_mode}, seed {a.seed})")
    print("All values are percentage points; positive = arm better than anchor.\n")
    hdr = f"{'arm':32s} {'metric':6s} {'n':>4s} {'mean_diff':>10s} {'95% CI':>18s} {'p':>10s} {'+/-/=':>12s}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        ci = f"[{r['ci95_low']},{r['ci95_high']}]"
        bwt = f"{r['cases_better']}/{r['cases_worse']}/{r['cases_tied']}"
        print(f"{r['arm']:32s} {r['metric']:6s} {r['n_paired_cases']:>4d} "
              f"{r['mean_diff_vs_anchor']:>10s} {ci:>18s} {r['wilcoxon_p']:>10s} {bwt:>12s}")
    print(f"\nWritten: {out_csv}")


if __name__ == "__main__":
    main()
