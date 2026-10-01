#!/usr/bin/env python3
"""Build consolidated ablation tables for the seed-42-only
surface_stepE30_fair_s42 study, anchored on FULL_STEP_E30
(DSC/NSD/legacy = 87.15/59.85/89.84 at seed 42, once its formal locked
test exists).

Reuses BASE / LOCAL_ALWAYS / FULL_STEP_E30 from the existing
surface_ablation_redesign_v2_results study (no retraining) and reads the
13 new arms from surface_stepE30_fair_s42_results. Never fabricates a
missing seed: each cell reports n=<count of seeds actually found>.
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics as st
from pathlib import Path

METRIC = re.compile(r"Average (DSC|NSD).*?:\s*([0-9.]+)%")


def read_metrics(log: Path) -> dict[str, float] | None:
    if not log.is_file():
        return None
    text = log.read_text(errors="replace")
    values = {k: float(v) for k, v in METRIC.findall(text)}
    if {"DSC", "NSD"} - set(values):
        return None
    return values


def triple(run_root: Path, split_tag: str) -> tuple[float, float, float] | None:
    true2d = read_metrics(run_root / "logs" / f"eval_{split_tag}_true2d.log")
    legacy = read_metrics(run_root / "logs" / f"eval_{split_tag}_paper_legacy.log")
    if true2d is None or legacy is None:
        return None
    if abs(true2d["DSC"] - legacy["DSC"]) > 0.01:
        raise ValueError(f"DSC mismatch between true2d/legacy evaluators at {run_root}")
    return true2d["DSC"], true2d["NSD"], legacy["NSD"]


def fmt(values: list[float]) -> str:
    if not values:
        return "n=0 (missing)"
    if len(values) == 1:
        return f"{values[0]:.2f} (n=1)"
    return f"{st.mean(values):.2f}\u00b1{st.pstdev(values):.2f} (n={len(values)})"


def collect(result_root: Path, study_id: str, arm: str, seeds: list[int], split_tag: str):
    dsc, nsd2d, nsdleg = [], [], []
    for seed in seeds:
        run_root = result_root / study_id / "BUSI" / arm / f"seed{seed}"
        t = triple(run_root, split_tag)
        if t is not None:
            dsc.append(t[0]); nsd2d.append(t[1]); nsdleg.append(t[2])
    return dsc, nsd2d, nsdleg


def row(label: str, dsc, nsd2d, nsdleg) -> dict:
    return {
        "setting": label,
        "dsc": fmt(dsc),
        "true2d_nsd": fmt(nsd2d),
        "legacy_nsd": fmt(nsdleg),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["setting", "dsc", "true2d_nsd", "legacy_nsd"])
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--reuse-result-root", default=None)
    p.add_argument("--reuse-study-id", required=True)
    p.add_argument("--new-result-root", default=None)
    p.add_argument("--new-study-id", required=True)
    p.add_argument("--seeds", default="42")
    a = p.parse_args()

    project = Path(a.project)
    reuse_root = Path(a.reuse_result_root) if a.reuse_result_root else project / "surface_ablation_redesign_v2_results"
    new_root = Path(a.new_result_root) if a.new_result_root else project / "surface_stepE30_fair_s42_results"
    seeds = [int(s) for s in a.seeds.split(",")]

    def reuse(arm, split_tag):
        return collect(reuse_root, a.reuse_study_id, arm, seeds, split_tag)

    def new(arm, split_tag):
        return collect(new_root, a.new_study_id, arm, seeds, split_tag)

    out = project / "surface_stepE30_fair_s42_results" / a.new_study_id / "tables"

    tA = [
        row("BASE (no surface term)", *reuse("BASE", "test_mc30")),
        row("+ local surface, always-on (LOCAL_ALWAYS)", *reuse("LOCAL_ALWAYS", "test_mc30")),
        row("+ local surface, step-e30, decoder-only grad", *new("DECODER_ONLY_STEP_E30", "test_mc30")),
        row("+ local surface, step-e30, full-path grad (FULL, anchor)", *reuse("FULL_STEP_E30", "test_mc30")),
    ]
    tB = [
        row("Boundary Loss (step-e30 matched)", *new("LOSS_BOUNDARY_STEP_E30", "test_mc30")),
        row("HD loss (step-e30 matched)", *new("LOSS_HD_STEP_E30", "test_mc30")),
        row("Active Contour Loss (step-e30 matched)", *new("LOSS_ACTIVE_CONTOUR_STEP_E30", "test_mc30")),
        row("SurfaceAlign (FULL, anchor)", *reuse("FULL_STEP_E30", "test_mc30")),
    ]
    tC = [
        row("No aux term (BASE, floor)", *reuse("BASE", "test_mc30")),
        row("Always-on (no withdrawal)", *reuse("LOCAL_ALWAYS", "test_mc30")),
        row("Cosine hold10/decay30", *new("SCHED_COSINE", "test_mc30")),
        row("Warm-up to epoch 30", *new("SCHED_WARMUP_E30", "test_mc30")),
        row("Step cutoff at epoch 30 (FULL, anchor)", *reuse("FULL_STEP_E30", "test_mc30")),
    ]
    tD = [
        row("r=1 (FULL, anchor)", *reuse("FULL_STEP_E30", "val_mc10")),
        row("r=2", *new("RADIUS_R2_STEP_E30", "val_mc10")),
        row("r=3", *new("RADIUS_R3_STEP_E30", "val_mc10")),
        row("r=5", *new("RADIUS_R5_STEP_E30", "val_mc10")),
    ]
    tE = [
        row("lambda0=0 (= BASE, reused)", *reuse("BASE", "val_mc10")),
        row("lambda0=0.05 (FULL, anchor)", *reuse("FULL_STEP_E30", "val_mc10")),
        row("lambda0=0.10", *new("WEIGHT_L010_STEP_E30", "val_mc10")),
        row("lambda0=0.15", *new("WEIGHT_L015_STEP_E30", "val_mc10")),
        row("lambda0=0.20", *new("WEIGHT_L020_STEP_E30", "val_mc10")),
        row("lambda0=0.25", *new("WEIGHT_L025_STEP_E30", "val_mc10")),
    ]

    tables = {
        "tableA_component_test.csv": tA,
        "tableB_auxloss_test.csv": tB,
        "tableC_schedule_test.csv": tC,
        "tableD_radius_val.csv": tD,
        "tableE_weight_val.csv": tE,
    }
    for name, rows in tables.items():
        write_csv(out / name, rows)

    md = ["# SurfaceAlign unified ablation (seed 42), anchored on FULL_STEP_E30\n"]
    md.append(f"Reused study (BASE/LOCAL_ALWAYS/FULL_STEP_E30): `{a.reuse_study_id}`\n")
    md.append(f"New study (13 single-factor arms):              `{a.new_study_id}`\n")

    def render(title, rows):
        md.append(f"\n## {title}\n")
        md.append("| Setting | DSC | true-2D NSD | Legacy NSD |")
        md.append("|---|---|---|---|")
        for r in rows:
            md.append(f"| {r['setting']} | {r['dsc']} | {r['true2d_nsd']} | {r['legacy_nsd']} |")

    render("Table A - Component ablation (Test, MC30)", tA)
    render("Table B - Auxiliary-objective identity, step-e30 matched (Test, MC30)", tB)
    render("Table C - Schedule shape, r=1/lambda0=0.05/full-path matched (Test, MC30)", tC)
    render("Table D - Surface radius sensitivity, step-e30 matched (Val, MC10, pre-freeze)", tD)
    render("Table E - Initial weight sensitivity, step-e30 matched (Val, MC10, pre-freeze)", tE)

    out.mkdir(parents=True, exist_ok=True)
    (out / "RESULTS.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nWritten to: {out}")


if __name__ == "__main__":
    main()
