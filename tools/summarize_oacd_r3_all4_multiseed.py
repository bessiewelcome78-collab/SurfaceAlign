#!/usr/bin/env python3
"""Summarize all-dataset paired reports without pooling correlated cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["BUSI", "Kvasir", "ISIC", "BTMRI"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 789])
    parser.add_argument("--mode", choices=["true2d", "paper_legacy"], default="true2d")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def report_path(root: Path, dataset: str, seed: int, mode: str) -> Path:
    return (
        root / dataset / f"seed{seed}" / "formal_test" / dataset /
        "seg_results" / f"seed{seed}" / f"paired_{mode}.json"
    )


def mean_sd(values):
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def main() -> None:
    args = arguments()
    per_seed = []
    aggregates = []
    for dataset in args.datasets:
        rows = []
        for seed in args.seeds:
            path = report_path(args.formal_root, dataset, seed, args.mode)
            if not path.is_file():
                raise FileNotFoundError(path)
            report = json.loads(path.read_text(encoding="utf-8"))
            row = {"dataset": dataset, "seed": seed, "cases": report["cases"]}
            for metric in ("DSC", "NSD"):
                item = report["metrics"][metric]
                prefix = metric.lower()
                row[f"{prefix}_base"] = item["base_mean"]
                row[f"{prefix}_m1"] = item["m1_mean"]
                row[f"{prefix}_delta"] = item["mean_delta"]
                row[f"{prefix}_ci_low"] = item["bootstrap_95_ci"][0]
                row[f"{prefix}_ci_high"] = item["bootstrap_95_ci"][1]
                row[f"{prefix}_p"] = item["paired_sign_flip_p_two_sided"]
            rows.append(row)
            per_seed.append(row)
        aggregate = {"dataset": dataset, "seeds": len(rows), "seed_values": list(args.seeds)}
        for metric in ("dsc", "nsd"):
            for quantity in ("base", "m1", "delta"):
                mean, sd = mean_sd([row[f"{metric}_{quantity}"] for row in rows])
                aggregate[f"{metric}_{quantity}_mean"] = mean
                aggregate[f"{metric}_{quantity}_sd_across_seeds"] = sd
        aggregates.append(aggregate)

    payload = {
        "protocol": f"OACD-R3-HCRV Paper100 {args.mode}",
        "formal_root": str(args.formal_root.resolve()),
        "note": "Case-level tests remain per seed; cases are not pooled across seeds.",
        "per_seed": per_seed,
        "aggregate_across_seeds": aggregates,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"all4_multiseed_{args.mode}.json"
    csv_path = args.output_dir / f"all4_multiseed_{args.mode}.csv"
    md_path = args.output_dir / f"all4_multiseed_{args.mode}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(per_seed).to_csv(csv_path, index=False)

    lines = [
        f"# OACD-R3-HCRV Paper100 ({args.mode})",
        "",
        "| Dataset | Seeds | DSC Base | DSC M1 | DSC gain | NSD Base | NSD M1 | NSD gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        def fmt(metric, quantity):
            return f"{100*row[f'{metric}_{quantity}_mean']:.3f}±{100*row[f'{metric}_{quantity}_sd_across_seeds']:.3f}"
        lines.append(
            f"| {row['dataset']} | {row['seeds']} | {fmt('dsc','base')} | {fmt('dsc','m1')} | "
            f"{fmt('dsc','delta')} pp | {fmt('nsd','base')} | {fmt('nsd','m1')} | {fmt('nsd','delta')} pp |"
        )
    lines.extend([
        "",
        "> 均值±标准差以独立训练 seed 为统计单位。病例级 bootstrap/符号翻转检验保留在每个 seed 的 paired JSON/Markdown 中，未把同一病例跨 seed 当成独立样本。",
    ])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
