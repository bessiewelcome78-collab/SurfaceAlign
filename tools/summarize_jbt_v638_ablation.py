#!/usr/bin/env python3
"""Write one ablation result and atomically refresh the shared comparison table."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path


EXPECTED_CASES = {"BUSI": 78, "Kvasir": 100, "ISIC": 379}
DATASET_ORDER = {"BUSI": 0, "Kvasir": 1, "ISIC": 2}
ARM_ORDER = {"FULL": 0, "NO_SEMANTIC": 1, "NO_DISPLACEMENT": 2, "NO_UTILITY": 3}
ARM_LABEL = {
    "FULL": "完整模型",
    "NO_SEMANTIC": "去掉语义图条件",
    "NO_DISPLACEMENT": "去掉有符号位移监督",
    "NO_UTILITY": "去掉候选效用选择器",
}
FIELDS = (
    "dataset", "arm", "arm_label", "seed", "cases",
    "paper_legacy_dsc", "paper_legacy_nsd", "true2d_dsc", "true2d_nsd",
    "delta_dsc_vs_full_pp", "delta_nsd_vs_full_pp",
    "selected_strength", "val_dsc_gain_pp", "val_nsd_gain_pp",
    "base_hash_all_epochs_match", "run_root", "completed_utc",
)


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(content, encoding="utf-8")
    os.replace(temp, path)


def mean_metrics(path: Path) -> tuple[float, float, int]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty evaluation CSV: {path}")
    dsc = [float(row["DSC"]) for row in rows]
    nsd = [float(row["NSD"]) for row in rows]
    if not all(math.isfinite(value) for value in dsc + nsd):
        raise ValueError(f"non-finite metric in {path}")
    return sum(dsc) / len(dsc), sum(nsd) / len(nsd), len(rows)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def full_row(dataset: str, seed: int, full_result: Path) -> dict:
    source = read_json(full_result)
    if source.get("dataset") != dataset or int(source.get("seed", -1)) != seed:
        raise ValueError(f"FULL result identity mismatch: {full_result}")
    return {
        "dataset": dataset,
        "arm": "FULL",
        "arm_label": ARM_LABEL["FULL"],
        "seed": seed,
        "cases": int(source["cases"]),
        "paper_legacy_dsc": float(source["final_dsc"]),
        "paper_legacy_nsd": float(source["final_nsd"]),
        "true2d_dsc": float(source["true2d_dsc"]),
        "true2d_nsd": float(source["true2d_nsd"]),
        "delta_dsc_vs_full_pp": 0.0,
        "delta_nsd_vs_full_pp": 0.0,
        "selected_strength": source["selected_strength"],
        "val_dsc_gain_pp": float(source["val_dsc_gain_pp"]),
        "val_nsd_gain_pp": float(source["val_nsd_gain_pp"]),
        "base_hash_all_epochs_match": bool(source["base_hash_all_epochs_match"]),
        "run_root": source["run_root"],
        "completed_utc": source["completed_utc"],
    }


def refresh_shared(summary_dir: Path, rows_to_add: list[dict]) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    with (summary_dir / ".ablation_summary.lock").open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        csv_path = summary_dir / "JBT_V638_ABLATION_SUMMARY.csv"
        old_rows = []
        if csv_path.exists():
            with csv_path.open(newline="", encoding="utf-8-sig") as handle:
                old_rows = list(csv.DictReader(handle))
        keyed = {(row["dataset"], row["arm"], str(row["seed"])): row for row in old_rows}
        for row in rows_to_add:
            keyed[(row["dataset"], row["arm"], str(row["seed"]))] = {
                field: str(row.get(field, "")) for field in FIELDS
            }
        records = sorted(
            keyed.values(),
            key=lambda row: (
                DATASET_ORDER.get(row["dataset"], 99),
                ARM_ORDER.get(row["arm"], 99),
                int(row["seed"]),
            ),
        )
        temp = csv_path.with_name(csv_path.name + f".tmp.{os.getpid()}")
        with temp.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(records)
        os.replace(temp, csv_path)

        lines = [
            "# JBT-v6.3.8 正式消融实验动态汇总",
            "",
            "> FULL 是同一数据集本次 fresh Paper100 正式结果；所有消融共享该数据集同一个冻结 Base100，",
            "> 但辅助模块均从头独立训练。每完成一个消融项，本表自动更新。",
            "",
            "| 数据集 | 模型 | N | DSC | 相对FULL | NSD | 相对FULL | true2d NSD | Val强度 | Base哈希 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ]
        for row in records:
            lines.append(
                "| {dataset} | {label} | {cases} | {dsc:.2f}% | {dd:+.2f} pp | "
                "{nsd:.2f}% | {dn:+.2f} pp | {tn:.2f}% | {strength} | {parity} |".format(
                    dataset=row["dataset"],
                    label=row["arm_label"],
                    cases=row["cases"],
                    dsc=100 * float(row["paper_legacy_dsc"]),
                    dd=float(row["delta_dsc_vs_full_pp"]),
                    nsd=100 * float(row["paper_legacy_nsd"]),
                    dn=float(row["delta_nsd_vs_full_pp"]),
                    tn=100 * float(row["true2d_nsd"]),
                    strength=row["selected_strength"],
                    parity="PASS" if row["base_hash_all_epochs_match"] == "True" else "FAIL",
                )
            )
        completed = sum(1 for row in records if row["arm"] != "FULL")
        lines += [
            "",
            f"- 已完成消融：{completed}/9；FULL 参考行不计入这 9 项。",
            "- 主表采用论文一致的 paper_legacy DSC/NSD；true2d NSD 作为边界补充指标。",
        ]
        atomic_text(
            summary_dir / "JBT_V638_ABLATION_SUMMARY.md",
            "\n".join(lines) + "\n",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(EXPECTED_CASES), required=True)
    parser.add_argument("--arm", choices=tuple(key for key in ARM_ORDER if key != "FULL"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--final-legacy", type=Path, required=True)
    parser.add_argument("--final-true2d", type=Path, required=True)
    parser.add_argument("--val-selection", type=Path, required=True)
    parser.add_argument("--base-hash-audit", type=Path, required=True)
    parser.add_argument("--full-result", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path, required=True)
    args = parser.parse_args()

    full = full_row(args.dataset, args.seed, args.full_result)
    dsc, nsd, count = mean_metrics(args.final_legacy)
    true_dsc, true_nsd, true_count = mean_metrics(args.final_true2d)
    if count != EXPECTED_CASES[args.dataset] or true_count != count:
        raise SystemExit(
            f"case-count contract failed for {args.dataset}: "
            f"legacy={count}, true2d={true_count}, expected={EXPECTED_CASES[args.dataset]}"
        )
    audit = read_json(args.base_hash_audit)
    if not bool(audit.get("all_epoch_hashes_match")):
        raise SystemExit("protected Base/PVL changed during ablation training")
    selection = read_json(args.val_selection)
    if args.arm == "NO_UTILITY":
        selected_strength = "1.0 fixed"
        candidate = min(
            selection["candidates"],
            key=lambda row: abs(float(row["strength"]) - 1.0),
        )
        val_dsc_gain_pp = 100 * float(candidate["dsc_gain"])
        val_nsd_gain_pp = 100 * float(candidate["nsd_gain"])
    else:
        selected_strength = selection["selected_strength"]
        val_dsc_gain_pp = 100 * float(selection["selected_val_dsc_gain"])
        val_nsd_gain_pp = 100 * float(selection["selected_val_nsd_gain"])

    row = {
        "dataset": args.dataset,
        "arm": args.arm,
        "arm_label": ARM_LABEL[args.arm],
        "seed": args.seed,
        "cases": count,
        "paper_legacy_dsc": dsc,
        "paper_legacy_nsd": nsd,
        "true2d_dsc": true_dsc,
        "true2d_nsd": true_nsd,
        "delta_dsc_vs_full_pp": 100 * (dsc - float(full["paper_legacy_dsc"])),
        "delta_nsd_vs_full_pp": 100 * (nsd - float(full["paper_legacy_nsd"])),
        "selected_strength": selected_strength,
        "val_dsc_gain_pp": val_dsc_gain_pp,
        "val_nsd_gain_pp": val_nsd_gain_pp,
        "base_hash_all_epochs_match": True,
        "run_root": str(args.run_root.resolve()),
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    diagnosis = args.run_root / "diagnosis"
    diagnosis.mkdir(parents=True, exist_ok=True)
    atomic_text(
        diagnosis / "JBT_V638_ABLATION_RESULT.json",
        json.dumps(row, indent=2, ensure_ascii=False) + "\n",
    )
    atomic_text(
        diagnosis / "JBT_V638_ABLATION_RESULT.md",
        "\n".join([
            f"# {args.dataset} / {ARM_LABEL[args.arm]}",
            "",
            f"- Paper legacy DSC/NSD: {100*dsc:.2f}% / {100*nsd:.2f}%",
            f"- 相对完整模型 DSC/NSD: {row['delta_dsc_vs_full_pp']:+.2f} / {row['delta_nsd_vs_full_pp']:+.2f} pp",
            f"- true2d DSC/NSD: {100*true_dsc:.2f}% / {100*true_nsd:.2f}%",
            f"- Validation部署强度: {selected_strength}",
            "- Base/PVL hash parity: PASS",
            "",
        ]) + "\n",
    )
    refresh_shared(args.summary_dir, [full, row])
    print(diagnosis / "JBT_V638_ABLATION_RESULT.md")
    print(args.summary_dir / "JBT_V638_ABLATION_SUMMARY.md")


if __name__ == "__main__":
    main()
