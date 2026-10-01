#!/usr/bin/env python3
"""Aggregate GEOTR-M1 final paired reports across datasets and seeds."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


DATASETS = ("BUSI", "BTMRI", "ISIC", "Kvasir")
METRICS = ("DSC", "NSD")


def parse_seeds(value: str):
    return [int(x) for x in value.replace(",", " ").split() if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, help="Root passed to the multi-seed launcher")
    parser.add_argument("--seeds", default="42,123,789")
    parser.add_argument("--output-prefix", default="")
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    seeds = parse_seeds(args.seeds)
    rows = []
    missing = []
    for dataset in DATASETS:
        for seed in seeds:
            report_path = root / dataset / f"seed{seed}" / "m1_test" / dataset / "seg_results" / f"seed{seed}" / "paired_true2d.json"
            if not report_path.is_file():
                missing.append(str(report_path))
                continue
            report = json.loads(report_path.read_text(encoding="utf-8"))
            row = {"dataset": dataset, "seed": seed, "report": str(report_path)}
            for metric in METRICS:
                item = report["metrics"][metric]
                row[f"base_{metric.lower()}"] = float(item["base_mean"])
                row[f"m1_{metric.lower()}"] = float(item["m1_mean"])
                row[f"delta_{metric.lower()}"] = float(item["mean_delta"])
                row[f"ci_low_{metric.lower()}"] = float(item["bootstrap_95_ci"][0])
                row[f"ci_high_{metric.lower()}"] = float(item["bootstrap_95_ci"][1])
                row[f"p_{metric.lower()}"] = float(item["paired_sign_flip_p_two_sided"])
            rows.append(row)
    if missing:
        raise SystemExit("[FAIL] missing paired_true2d.json files:\n" + "\n".join(missing))

    summary = {}
    for dataset in DATASETS:
        selected = [row for row in rows if row["dataset"] == dataset]
        if len(selected) != len(seeds):
            raise SystemExit(f"[FAIL] {dataset}: expected {len(seeds)} seeds, found {len(selected)}")
        item = {"dataset": dataset, "seeds": seeds}
        for metric in METRICS:
            suffix = metric.lower()
            for field in ("base", "m1", "delta", "ci_low", "ci_high", "p"):
                values = [row[f"{field}_{suffix}"] for row in selected]
                item[f"{field}_{suffix}_mean"] = statistics.mean(values)
                item[f"{field}_{suffix}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            item[f"all_seeds_positive_{suffix}"] = all(row[f"delta_{suffix}"] > 0.0 for row in selected)
            item[f"all_case_bootstrap_ci_lower_gt_zero_{suffix}"] = all(row[f"ci_low_{suffix}"] > 0.0 for row in selected)
            item[f"all_p_lt_0_05_{suffix}"] = all(row[f"p_{suffix}"] < 0.05 for row in selected)
        item["stable_dsc_and_nsd"] = bool(
            item["all_seeds_positive_dsc"] and item["all_seeds_positive_nsd"]
            and item["all_case_bootstrap_ci_lower_gt_zero_dsc"]
            and item["all_case_bootstrap_ci_lower_gt_zero_nsd"]
        )
        summary[dataset] = item

    prefix = Path(args.output_prefix).resolve() if args.output_prefix else root / "geotr_m1_final_summary"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "seed"] + [key for key in rows[0] if key not in {"dataset", "seed", "report"}] + ["report"]
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# GEOTR-M1 Final Paper100 多 seed 汇总",
        "",
        "> 稳定提升定义：3 个 seed 的 DSC/NSD 配对均值都大于 0，且每个 seed 的病例级 bootstrap 95% CI 下界都大于 0。跨 seed 的训练显著性还应结合 seed 数量增加后的统计检验，不得只看单个 seed。",
        "",
        "| 数据集 | DSC Base | DSC M1 | ΔDSC | NSD Base | NSD M1 | ΔNSD | 所有 seed 提升 | 病例 CI 下界均>0 |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for dataset in DATASETS:
        item = summary[dataset]
        fmt = lambda key: f"{100 * item[key + '_mean']:.2f}±{100 * item[key + '_std']:.2f}"
        stable = item["stable_dsc_and_nsd"]
        ci_ok = item["all_case_bootstrap_ci_lower_gt_zero_dsc"] and item["all_case_bootstrap_ci_lower_gt_zero_nsd"]
        lines.append(
            f"| {dataset} | {fmt('base_dsc')} | {fmt('m1_dsc')} | {fmt('delta_dsc')} | "
            f"{fmt('base_nsd')} | {fmt('m1_nsd')} | {fmt('delta_nsd')} | "
            f"{'YES' if stable else 'NO'} | {'YES' if ci_ok else 'NO'} |"
        )
    lines.extend(["", "## 逐 seed 原始结果", ""])
    for row in rows:
        lines.append(
            f"- {row['dataset']} seed{row['seed']}: "
            f"ΔDSC={100*row['delta_dsc']:+.3f} pp, "
            f"ΔNSD={100*row['delta_nsd']:+.3f} pp, "
            f"p(DSC)={row['p_dsc']:.5g}, p(NSD)={row['p_nsd']:.5g}"
        )
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    prefix.with_suffix(".json").write_text(
        json.dumps({"run_root": str(root), "seeds": seeds, "summary": summary, "rows": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(prefix.with_suffix(".md"))
    print(prefix.with_suffix(".csv"))
    print(prefix.with_suffix(".json"))


if __name__ == "__main__":
    main()
