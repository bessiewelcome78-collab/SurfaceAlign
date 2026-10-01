#!/usr/bin/env python3
"""Create per-run and cumulative summaries for JBT-v6.3.7 Paper100.

The paper benchmark is descriptive, never a Test-time tuning target.  This tool
does not fail a scientifically valid run merely because it is below a published
number; it records the margin transparently.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path


PAPER = {
    "BUSI": {"DSC": 0.8572, "NSD": 0.8835, "cases": 78},
    "BTMRI": {"DSC": 0.8803, "NSD": 0.9178, "cases": 1005},
    "ISIC": {"DSC": 0.9254, "NSD": 0.9358, "cases": 379},
    "Kvasir": {"DSC": 0.9015, "NSD": 0.9232, "cases": 100},
}
ORDER = {name: index for index, name in enumerate(("BUSI", "Kvasir", "ISIC", "BTMRI"))}
FIELDS = (
    "dataset", "seed", "cases", "final_dsc", "final_nsd",
    "paper_dsc", "paper_nsd", "dsc_margin_pp", "nsd_margin_pp",
    "above_paper_dsc", "above_paper_nsd", "above_paper_both",
    "selected_strength", "val_dsc_gain_pp", "val_nsd_gain_pp",
    "true2d_dsc", "true2d_nsd", "base_hash_all_epochs_match",
    "run_root", "completed_utc",
)


def mean_metrics(path: Path) -> tuple[float, float, int]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty evaluation CSV: {path}")
    try:
        dsc = [float(row["DSC"]) for row in rows]
        nsd = [float(row["NSD"]) for row in rows]
    except KeyError as exc:
        raise ValueError(f"{path} lacks DSC/NSD columns") from exc
    if not all(math.isfinite(value) for value in dsc + nsd):
        raise ValueError(f"non-finite metric in {path}")
    return sum(dsc) / len(dsc), sum(nsd) / len(nsd), len(rows)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def write_shared(summary_dir: Path, row: dict) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    lock_path = summary_dir / ".summary.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        csv_path = summary_dir / "JBT_V637_PAPER100_ALL4_SUMMARY.csv"
        prior = []
        if csv_path.exists():
            with csv_path.open(newline="", encoding="utf-8-sig") as handle:
                prior = list(csv.DictReader(handle))
        keyed = {(item["dataset"], str(item["seed"])): item for item in prior}
        keyed[(row["dataset"], str(row["seed"]))] = {key: str(row.get(key, "")) for key in FIELDS}
        records = sorted(
            keyed.values(),
            key=lambda item: (ORDER.get(item["dataset"], 99), int(item["seed"])),
        )
        temporary = csv_path.with_name(csv_path.name + f".tmp.{os.getpid()}")
        with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(records)
        os.replace(temporary, csv_path)

        lines = [
            "# JBT-v6.3.7 Paper100 四数据集动态汇总", "",
            "> 每个数据集完成后自动更新。论文值只用于最终对照，不参与训练或 Test 调参。", "",
            "| 数据集 | Seed | N | Final DSC | 论文 DSC | 差值 | Final NSD | 论文 NSD | 差值 | Val强度 | 同时超过 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ]
        for item in records:
            lines.append(
                "| {dataset} | {seed} | {cases} | {fd:.2f}% | {pd:.2f}% | {dm:+.2f} pp | "
                "{fn:.2f}% | {pn:.2f}% | {nm:+.2f} pp | {strength} | {both} |".format(
                    dataset=item["dataset"], seed=item["seed"], cases=item["cases"],
                    fd=100 * float(item["final_dsc"]), pd=100 * float(item["paper_dsc"]),
                    dm=float(item["dsc_margin_pp"]), fn=100 * float(item["final_nsd"]),
                    pn=100 * float(item["paper_nsd"]), nm=float(item["nsd_margin_pp"]),
                    strength=item["selected_strength"],
                    both="YES" if item["above_paper_both"] == "True" else "NO",
                )
            )
        lines += ["", f"- 已完成：{len(records)}/4 个数据集（当前 seed 集合按行统计）"]
        atomic_text(summary_dir / "JBT_V637_PAPER100_ALL4_SUMMARY.md", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(PAPER), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--final-legacy", type=Path, required=True)
    parser.add_argument("--final-true2d", type=Path, required=True)
    parser.add_argument("--val-selection", type=Path, required=True)
    parser.add_argument("--base-hash-audit", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path, required=True)
    args = parser.parse_args()

    dsc, nsd, count = mean_metrics(args.final_legacy)
    true_dsc, true_nsd, true_count = mean_metrics(args.final_true2d)
    expected = PAPER[args.dataset]
    if count != expected["cases"] or true_count != count:
        raise SystemExit(
            f"case-count contract failed for {args.dataset}: "
            f"paper_legacy={count}, true2d={true_count}, expected={expected['cases']}"
        )
    selection = load_json(args.val_selection)
    hash_audit = load_json(args.base_hash_audit)
    if not bool(hash_audit.get("all_epoch_hashes_match")):
        raise SystemExit("protected Base/PVL changed during JBT training")

    row = {
        "dataset": args.dataset, "seed": args.seed, "cases": count,
        "final_dsc": dsc, "final_nsd": nsd,
        "paper_dsc": expected["DSC"], "paper_nsd": expected["NSD"],
        "dsc_margin_pp": 100 * (dsc - expected["DSC"]),
        "nsd_margin_pp": 100 * (nsd - expected["NSD"]),
        "above_paper_dsc": dsc > expected["DSC"],
        "above_paper_nsd": nsd > expected["NSD"],
        "above_paper_both": dsc > expected["DSC"] and nsd > expected["NSD"],
        "selected_strength": selection["selected_strength"],
        "val_dsc_gain_pp": 100 * float(selection["selected_val_dsc_gain"]),
        "val_nsd_gain_pp": 100 * float(selection["selected_val_nsd_gain"]),
        "true2d_dsc": true_dsc, "true2d_nsd": true_nsd,
        "base_hash_all_epochs_match": True,
        "run_root": str(args.run_root.resolve()),
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    per_json = args.run_root / "diagnosis" / "JBT_V637_FINAL_RESULT.json"
    atomic_text(per_json, json.dumps(row, indent=2, ensure_ascii=False) + "\n")
    per_md = args.run_root / "diagnosis" / "JBT_V637_FINAL_RESULT.md"
    atomic_text(
        per_md,
        "\n".join([
            f"# JBT-v6.3.7 Paper100 最终结果：{args.dataset}", "",
            f"- Seed: {args.seed}", f"- 病例数: {count}",
            f"- Final paper_legacy DSC: {100*dsc:.2f}%（论文 {100*expected['DSC']:.2f}%，差值 {row['dsc_margin_pp']:+.2f} pp）",
            f"- Final paper_legacy NSD: {100*nsd:.2f}%（论文 {100*expected['NSD']:.2f}%，差值 {row['nsd_margin_pp']:+.2f} pp）",
            f"- Final true2d DSC/NSD: {100*true_dsc:.2f}% / {100*true_nsd:.2f}%",
            f"- Validation锁定强度: {selection['selected_strength']}",
            f"- Validation DSC/NSD增益: {row['val_dsc_gain_pp']:+.3f} / {row['val_nsd_gain_pp']:+.3f} pp",
            f"- 同时超过论文 DSC 与 NSD: {row['above_paper_both']}",
            "- Base/PVL 在JBT阶段全程哈希一致: True", "",
            "> 论文值没有参与训练、Validation选择或Test决策。", "",
        ])
    )
    write_shared(args.summary_dir, row)
    print(per_md)
    print(args.summary_dir / "JBT_V637_PAPER100_ALL4_SUMMARY.md")


if __name__ == "__main__":
    main()

