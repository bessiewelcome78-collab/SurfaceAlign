#!/usr/bin/env python3
"""Atomically merge FULL/ablation extended metrics into one shared table."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path


METRICS = (
    "DSC", "IoU", "Precision", "Recall", "Specificity", "Accuracy",
    "NSD2_ModelPx", "HD95_ModelPx", "ASSD_ModelPx", "AreaAbsError",
)
LOWER_BETTER = {"HD95_ModelPx", "ASSD_ModelPx", "AreaAbsError"}
DATASET_ORDER = {"BUSI": 0, "Kvasir": 1, "ISIC": 2}
ARM_ORDER = {"FULL": 0, "NO_SEMANTIC": 1, "NO_DISPLACEMENT": 2, "NO_UTILITY": 3}
LABEL = {
    "FULL": "完整模型",
    "NO_SEMANTIC": "去掉语义图条件",
    "NO_DISPLACEMENT": "去掉有符号位移监督",
    "NO_UTILITY": "去掉候选效用选择器",
}
IDENTITY_FIELDS = ("dataset", "arm", "arm_label", "cases", "run_root")


def finite(value: object) -> float:
    number = float(value)
    return number if math.isfinite(number) else float("nan")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def make_row(dataset: str, arm: str, summary: dict, run_root: Path) -> dict:
    row: dict[str, object] = {
        "dataset": dataset,
        "arm": arm,
        "arm_label": LABEL[arm],
        "cases": int(summary["cases"]),
        "run_root": str(run_root.resolve()),
        "gt_empty_cases": int(summary["gt_empty_cases"]),
        "pred_empty_cases": int(summary["pred_empty_cases"]),
        "empty_mismatch_cases": int(summary["empty_mismatch_cases"]),
    }
    comparison = summary.get("paired_vs_reference", {})
    for metric in METRICS:
        item = summary["metrics"][metric]
        row[f"{metric}_mean"] = finite(item["mean"])
        row[f"{metric}_median"] = finite(item["median"])
        row[f"{metric}_valid_cases"] = int(item["valid_cases"])
        paired = comparison.get(metric, {})
        row[f"{metric}_raw_delta_vs_full"] = finite(paired.get("raw_delta", 0.0 if arm == "FULL" else float("nan")))
        row[f"{metric}_p_vs_full"] = finite(paired.get("paired_sign_flip_p_two_sided", float("nan")))
        row[f"{metric}_improved"] = int(paired.get("improved", 0))
        row[f"{metric}_harmed"] = int(paired.get("harmed", 0))
        row[f"{metric}_tied"] = int(paired.get("tied", int(summary["cases"]) if arm == "FULL" else 0))
    return row


def render(records: list[dict[str, str]]) -> str:
    def value(row: dict[str, str], key: str) -> float:
        return float(row[key])

    lines = [
        "# JBT-v6.3.8 扩展消融指标动态汇总", "",
        "> 所有消融均与同数据集 FULL 逐病例配对；FULL 与消融共享冻结的 fresh Base100，辅助模块独立训练。",
        "> 距离指标使用模型等价像素；HD95/ASSD 仅在 GT 与预测均非空的病例上统计。", "",
        "## 区域、分类与校准指标", "",
        "| 数据集 | 模型 | DSC | IoU | Precision | Recall | Specificity | Accuracy | 面积绝对误差 | 空掩膜错配 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            "| {dataset} | {label} | {dsc:.2f}% | {iou:.2f}% | {precision:.2f}% | "
            "{recall:.2f}% | {specificity:.2f}% | {accuracy:.2f}% | {area:.3f}% | {empty}/{cases} |".format(
                dataset=row["dataset"], label=row["arm_label"],
                dsc=100 * value(row, "DSC_mean"), iou=100 * value(row, "IoU_mean"),
                precision=100 * value(row, "Precision_mean"), recall=100 * value(row, "Recall_mean"),
                specificity=100 * value(row, "Specificity_mean"), accuracy=100 * value(row, "Accuracy_mean"),
                area=100 * value(row, "AreaAbsError_mean"), empty=row["empty_mismatch_cases"], cases=row["cases"],
            )
        )
    lines += [
        "", "## 边界指标", "",
        "| 数据集 | 模型 | NSD@2px | HD95↓ | ASSD↓ | HD95有效N |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            "| {dataset} | {label} | {nsd:.2f}% | {hd:.3f} | {assd:.3f} | {valid} |".format(
                dataset=row["dataset"], label=row["arm_label"],
                nsd=100 * value(row, "NSD2_ModelPx_mean"), hd=value(row, "HD95_ModelPx_mean"),
                assd=value(row, "ASSD_ModelPx_mean"), valid=row["HD95_ModelPx_valid_cases"],
            )
        )
    lines += [
        "", "## 相对 FULL 的逐病例配对差值与显著性", "",
        "| 数据集 | 消融 | ΔDSC | p | ΔIoU | p | ΔNSD | p | ΔHD95 | p |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        if row["arm"] == "FULL":
            continue
        lines.append(
            "| {dataset} | {label} | {dd:+.2f} pp | {pd:.3g} | {di:+.2f} pp | {pi:.3g} | "
            "{dn:+.2f} pp | {pn:.3g} | {dh:+.3f} | {ph:.3g} |".format(
                dataset=row["dataset"], label=row["arm_label"],
                dd=100 * value(row, "DSC_raw_delta_vs_full"), pd=value(row, "DSC_p_vs_full"),
                di=100 * value(row, "IoU_raw_delta_vs_full"), pi=value(row, "IoU_p_vs_full"),
                dn=100 * value(row, "NSD2_ModelPx_raw_delta_vs_full"), pn=value(row, "NSD2_ModelPx_p_vs_full"),
                dh=value(row, "HD95_ModelPx_raw_delta_vs_full"), ph=value(row, "HD95_ModelPx_p_vs_full"),
            )
        )
    lines += [
        "",
        "- 对 DSC/IoU/NSD，负差值表示删除组件后变差；对 HD95/ASSD/面积误差，正差值表示变差。",
        "- CSV 同时保留全部指标的均值、中位数、有效病例数、改善/损害/持平例数和配对 p 值。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASET_ORDER), required=True)
    parser.add_argument("--arm", choices=tuple(key for key in ARM_ORDER if key != "FULL"), required=True)
    parser.add_argument("--full-summary", type=Path, required=True)
    parser.add_argument("--ablation-summary", type=Path, required=True)
    parser.add_argument("--full-run-root", type=Path, required=True)
    parser.add_argument("--ablation-run-root", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path, required=True)
    args = parser.parse_args()
    full = load(args.full_summary)
    ablation = load(args.ablation_summary)
    if int(full["cases"]) != int(ablation["cases"]):
        raise SystemExit("FULL and ablation case counts differ")
    additions = [
        make_row(args.dataset, "FULL", full, args.full_run_root),
        make_row(args.dataset, args.arm, ablation, args.ablation_run_root),
    ]
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    with (args.summary_dir / ".extended_ablation.lock").open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        path = args.summary_dir / "JBT_V638_ABLATION_EXTENDED_SUMMARY.csv"
        previous: list[dict[str, str]] = []
        if path.exists():
            with path.open(newline="", encoding="utf-8-sig") as handle:
                previous = list(csv.DictReader(handle))
        keyed = {(row["dataset"], row["arm"]): row for row in previous}
        for row in additions:
            keyed[(str(row["dataset"]), str(row["arm"]))] = {key: str(value) for key, value in row.items()}
        records = sorted(keyed.values(), key=lambda row: (DATASET_ORDER[row["dataset"]], ARM_ORDER[row["arm"]]))
        fieldnames = list(IDENTITY_FIELDS)
        for extra in ("gt_empty_cases", "pred_empty_cases", "empty_mismatch_cases"):
            fieldnames.append(extra)
        for metric in METRICS:
            fieldnames.extend((
                f"{metric}_mean", f"{metric}_median", f"{metric}_valid_cases",
                f"{metric}_raw_delta_vs_full", f"{metric}_p_vs_full",
                f"{metric}_improved", f"{metric}_harmed", f"{metric}_tied",
            ))
        temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
        with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        os.replace(temporary, path)
        atomic_write(args.summary_dir / "JBT_V638_ABLATION_EXTENDED_SUMMARY.md", render(records))
    print(path)
    print(args.summary_dir / "JBT_V638_ABLATION_EXTENDED_SUMMARY.md")


if __name__ == "__main__":
    main()
