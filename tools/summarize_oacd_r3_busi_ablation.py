#!/usr/bin/env python3
"""Build the preregistered BUSI component-ablation table from per-case CSVs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ARMS = (
    "R8_SEARCH_RANGE",
    "NO_CENSORED_SUPERVISION",
    "NO_RELATIONAL_VOLUME",
    "NO_HIERARCHICAL_DECISION",
)
METRICS = ("DSC", "NSD")


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-base-csv", type=Path, required=True)
    parser.add_argument("--full-m1-csv", type=Path, required=True)
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--permutation-samples", type=int, default=100000)
    parser.add_argument("--rng-seed", type=int, default=20260904)
    return parser.parse_args()


def load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"Case_ID", *METRICS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns {sorted(missing)}")
    if frame["Case_ID"].duplicated().any():
        raise ValueError(f"{path} contains duplicate Case_ID")
    return frame.set_index("Case_ID").sort_index()


def align(reference: pd.DataFrame, other: pd.DataFrame, label: str) -> pd.DataFrame:
    if set(reference.index) != set(other.index):
        raise ValueError(f"Case_ID mismatch for {label}")
    return other.loc[reference.index]


def paired_stats(a, b, rng, bootstrap_samples, permutation_samples):
    """Return paired B-A statistics; positive Full-Abl means a useful component."""
    delta = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    n = int(delta.size)
    if n <= 0:
        raise ValueError("empty paired sample")
    bootstrap = np.empty(bootstrap_samples, dtype=np.float64)
    chunk = 4096
    for start in range(0, bootstrap_samples, chunk):
        current = min(chunk, bootstrap_samples - start)
        ids = rng.integers(0, n, size=(current, n))
        bootstrap[start:start + current] = delta[ids].mean(axis=1)
    ci = np.quantile(bootstrap, [0.025, 0.975])
    observed = abs(float(delta.mean()))
    extreme = 0
    remaining = permutation_samples
    while remaining > 0:
        current = min(chunk, remaining)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(current, n))
        extreme += int((np.abs((signs * delta[None]).mean(1)) >= observed - 1e-15).sum())
        remaining -= current
    return {
        "mean_delta": float(delta.mean()),
        "bootstrap_95_ci": [float(ci[0]), float(ci[1])],
        "paired_sign_flip_p_two_sided": float((extreme + 1) / (permutation_samples + 1)),
        "benefit_cases": int((delta > 1e-12).sum()),
        "harm_cases": int((delta < -1e-12).sum()),
    }


def arm_csv(root: Path, arm: str, seed: int, name: str) -> Path:
    path = root / arm / f"seed{seed}" / "formal_test" / "BUSI" / "seg_results" / f"seed{seed}" / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    args = arguments()
    rng = np.random.default_rng(args.rng_seed)
    full_base = load(args.full_base_csv)
    full_m1 = align(full_base, load(args.full_m1_csv), "FULL M1")
    result = {
        "protocol": "BUSI OACD-R3-HCRV one-component-removal Ablation100",
        "seed": args.seed,
        "cases": int(len(full_base)),
        "single_seed_training_variance_limitation": True,
        "rows": {},
    }
    full_row = {"metrics": {}}
    for metric in METRICS:
        full_row["metrics"][metric] = {
            "base_mean": float(full_base[metric].mean()),
            "m1_mean": float(full_m1[metric].mean()),
            "m1_minus_base": paired_stats(
                full_base[metric], full_m1[metric], rng,
                args.bootstrap_samples, args.permutation_samples,
            ),
            "full_minus_ablation": None,
        }
    result["rows"]["FULL"] = full_row

    for arm in ARMS:
        base = align(full_base, load(arm_csv(args.ablation_root, arm, args.seed, "test_BaseNative_true2d.csv")), f"{arm} Base")
        m1 = align(full_base, load(arm_csv(args.ablation_root, arm, args.seed, "test_M1Native_true2d.csv")), f"{arm} M1")
        max_base_delta = max(
            float(np.max(np.abs(base[metric].to_numpy() - full_base[metric].to_numpy())))
            for metric in METRICS
        )
        row = {
            "base_parity_max_abs_delta": max_base_delta,
            "base_parity_pass": bool(max_base_delta <= 1.0e-8),
            "metrics": {},
        }
        for metric in METRICS:
            row["metrics"][metric] = {
                "base_mean": float(base[metric].mean()),
                "m1_mean": float(m1[metric].mean()),
                "m1_minus_base": paired_stats(
                    base[metric], m1[metric], rng,
                    args.bootstrap_samples, args.permutation_samples,
                ),
                "full_minus_ablation": paired_stats(
                    m1[metric], full_m1[metric], rng,
                    args.bootstrap_samples, args.permutation_samples,
                ),
            }
        result["rows"][arm] = row

    result["all_ablation_base_outputs_match_full"] = all(
        row.get("base_parity_pass", True) for row in result["rows"].values()
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "BUSI_OACD_R3_HCRV_ABL100_summary.json"
    csv_path = args.output_dir / "BUSI_OACD_R3_HCRV_ABL100_summary.csv"
    md_path = args.output_dir / "BUSI_OACD_R3_HCRV_ABL100_summary.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    csv_rows = []
    for arm, row in result["rows"].items():
        for metric in METRICS:
            item = row["metrics"][metric]
            contribution = item["full_minus_ablation"]
            csv_rows.append({
                "arm": arm,
                "metric": metric,
                "base_mean": item["base_mean"],
                "m1_mean": item["m1_mean"],
                "m1_minus_base": item["m1_minus_base"]["mean_delta"],
                "full_minus_ablation": None if contribution is None else contribution["mean_delta"],
                "full_minus_ablation_ci_low": None if contribution is None else contribution["bootstrap_95_ci"][0],
                "full_minus_ablation_ci_high": None if contribution is None else contribution["bootstrap_95_ci"][1],
                "full_minus_ablation_p": None if contribution is None else contribution["paired_sign_flip_p_two_sided"],
                "base_parity_pass": row.get("base_parity_pass", True),
            })
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    lines = [
        "# BUSI OACD-R3-HCRV Ablation100",
        "",
        "正的 `Full-Ablated` 表示被移除组件对完整方法有贡献。病例级 CI/p 值不替代多 seed 的训练方差。",
        "",
        "| Arm | DSC Base | DSC M1 | M1-Base | Full-Ablated | 95% CI | p | Base parity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, row in result["rows"].items():
        item = row["metrics"]["DSC"]
        contribution = item["full_minus_ablation"]
        if contribution is None:
            delta, ci, p = "—", "—", "—"
        else:
            delta = f"{100 * contribution['mean_delta']:+.3f} pp"
            ci = f"[{100 * contribution['bootstrap_95_ci'][0]:+.3f}, {100 * contribution['bootstrap_95_ci'][1]:+.3f}]"
            p = f"{contribution['paired_sign_flip_p_two_sided']:.4g}"
        lines.append(
            f"| {arm} | {100 * item['base_mean']:.3f}% | {100 * item['m1_mean']:.3f}% | "
            f"{100 * item['m1_minus_base']['mean_delta']:+.3f} pp | {delta} | {ci} | {p} | "
            f"{'PASS' if row.get('base_parity_pass', True) else 'FAIL'} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)
    print(csv_path)
    print(json_path)
    if not result["all_ablation_base_outputs_match_full"]:
        raise SystemExit(11)


if __name__ == "__main__":
    main()
