#!/usr/bin/env python3
"""Audit FORMAL55/CAUSAL56/LeST57 logs without promoting incomplete runs."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


VARIANT_TOKEN = re.compile(
    r"GEOTR_M1_(?:FORMAL55|CAUSAL56|LEST57(?:01)?)_[A-Z0-9_]+?_"
    r"(FULL|NO_TEXT|NO_SEMANTIC|IMAGE_ONLY|NO_ANCHOR|PROB_WARP|"
    r"NO_DEFORM|EDGE_DEFORM|CORE_EDGE|CORE|SEMANTIC|REWRITE|FREE2D|"
    r"FREE2D_GATE|NORMAL1D|LEST|LEST_GLOBAL|LEST_PROB)_"
    r"(?:PAPER100|BASEFROZEN100)"
)
NOHUP_TOKEN = re.compile(
    r"GEOTR_M1_(?:FORMAL55|CAUSAL56|LEST57(?:01)?)_[A-Za-z0-9_]+?_([a-z0-9_]+)_val_seed"
)
EPOCH = re.compile(r"^EPOCH:\s*(\d+)\s*\|", re.MULTILINE)
DIAG_ITEM = re.compile(r"([A-Za-z0-9_]+)=([-+0-9.eE]+)")
AVERAGE = re.compile(
    r"Average (DSC|NSD) for .*_(BaseNative|M1Native)_Val .*: ([0-9.]+)%"
)


def variant_from_name(path: Path) -> str | None:
    upper = VARIANT_TOKEN.search(path.name)
    if upper:
        return upper.group(1).lower()
    lower = NOHUP_TOKEN.search(path.name)
    return lower.group(1).lower() if lower else None


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")


def last_diagnostic(text: str) -> dict[str, float]:
    lines = [line for line in text.splitlines() if line.startswith("M1_DIAG:")]
    if not lines:
        return {}
    return {key: float(value) for key, value in DIAG_ITEM.findall(lines[-1])}


def failure_reason(text: str) -> str | None:
    if "ModuleNotFoundError: No module named 'torch'" in text:
        return "python_without_torch"
    if "dataset split audit failed" in text:
        return "obsolete_split_audit_failure"
    if "Traceback (most recent call last)" in text:
        tail = text.split("Traceback (most recent call last)")[-1].strip().splitlines()
        return tail[-1].strip() if tail else "python_traceback"
    return None


def parse_eval(text: str) -> dict:
    mode = None
    parsed: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for line in text.splitlines():
        if "CSV output" in line or "Per-case CSV" in line:
            if "true2d" in line:
                mode = "true2d"
            elif "paper_legacy" in line:
                mode = "paper_legacy"
        match = AVERAGE.search(line)
        if match and mode is not None:
            metric, branch, value = match.groups()
            parsed[mode][branch][metric] = float(value) / 100.0
    return {
        mode_name: {branch: dict(metrics) for branch, metrics in branches.items()}
        for mode_name, branches in parsed.items()
    }


def merge_logs(paths: list[Path], protocol: str) -> dict:
    texts = [(path, read_text(path)) for path in paths]
    epochs = [int(item) for _, text in texts for item in EPOCH.findall(text)]
    diagnostics = [last_diagnostic(text) for _, text in texts]
    diagnostics = [item for item in diagnostics if item]
    evaluation = {}
    for _, text in texts:
        candidate = parse_eval(text)
        if candidate:
            evaluation.update(candidate)
    failures = [failure_reason(text) for _, text in texts]
    failures = [item for item in failures if item]
    marker = "[PASS] CAUSAL56" if protocol in {"causal56", "lest57"} else "[PASS] FORMAL55"
    completed = any(marker in text and "/val complete" in text for _, text in texts)
    status = "complete" if completed else "training_incomplete" if epochs else "failed_to_start"
    record = {
        "status": status,
        "max_epoch": max(epochs, default=0),
        "completed": completed,
        # A variant can have an earlier failed launch and a later live run.
        # Keep those events for diagnosis without mislabelling the live run.
        "observed_failed_attempts": sorted(set(failures)),
        "evaluation": evaluation,
        "last_diagnostic": diagnostics[-1] if diagnostics else {},
        "files": [str(path) for path in sorted(paths)],
    }
    true2d = evaluation.get("true2d", {})
    base = true2d.get("BaseNative", {})
    m1 = true2d.get("M1Native", {})
    if {"DSC", "NSD"} <= base.keys() and {"DSC", "NSD"} <= m1.keys():
        record["true2d_gain"] = {
            metric: m1[metric] - base[metric] for metric in ("DSC", "NSD")
        }
    legacy = evaluation.get("paper_legacy", {})
    base = legacy.get("BaseNative", {})
    m1 = legacy.get("M1Native", {})
    if {"DSC", "NSD"} <= base.keys() and {"DSC", "NSD"} <= m1.keys():
        record["paper_legacy_gain"] = {
            metric: m1[metric] - base[metric] for metric in ("DSC", "NSD")
        }
    return record


def pct(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.2f}"


def make_markdown(records: dict[str, dict], protocol: str) -> str:
    title = "CAUSAL56 同一冻结 Base 消融审计" if protocol == "causal56" else "FORMAL55 消融日志审计"
    lines = [
        f"# GEOTR-M1 {title}",
        "",
        "> 只有 `complete` 且包含 Base/M1 配对评估的运行才是实验结果；训练中断或启动失败不能用于模块优劣结论。",
        "",
        "| 变体 | 状态 | 最后轮次 | Base DSC | M1 DSC | ΔDSC | Base true2d NSD | M1 true2d NSD | ΔNSD |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, record in sorted(records.items()):
        true2d = record.get("evaluation", {}).get("true2d", {})
        base = true2d.get("BaseNative", {})
        m1 = true2d.get("M1Native", {})
        gain = record.get("true2d_gain", {})
        lines.append(
            f"| {variant} | {record['status']} | {record['max_epoch']} | "
            f"{pct(base.get('DSC'))} | {pct(m1.get('DSC'))} | {pct(gain.get('DSC'))} | "
            f"{pct(base.get('NSD'))} | {pct(m1.get('NSD'))} | {pct(gain.get('NSD'))} |"
        )
    lines.extend([
        "",
        "## 自动判读",
        "",
    ])
    complete = [name for name, item in records.items() if item["status"] == "complete"]
    incomplete = [name for name, item in records.items() if item["status"] != "complete"]
    lines.append(f"- 可用于最终数值比较：{', '.join(sorted(complete)) or '无'}。")
    lines.append(f"- 不可用于模块结论：{', '.join(sorted(incomplete)) or '无'}。")
    if "full" in records and "no_deform" in records:
        full = records["full"]
        no_deform = records["no_deform"]
        if full.get("true2d_gain") and no_deform.get("true2d_gain"):
            full_base = full["evaluation"]["true2d"]["BaseNative"]["DSC"]
            nd_base = no_deform["evaluation"]["true2d"]["BaseNative"]["DSC"]
            lines.extend([
                "",
                "### Full 与 no_deform 的解释边界",
                "",
                f"- 两次运行的 Base DSC 相差 {pct(nd_base - full_base)} 个百分点；因此不能用 M1 绝对分数差直接归因于形变正则。",
                f"- Full 的配对增益为 DSC {pct(full['true2d_gain']['DSC'])}、NSD {pct(full['true2d_gain']['NSD'])} 个百分点。",
                f"- no_deform 的配对增益为 DSC {pct(no_deform['true2d_gain']['DSC'])}、NSD {pct(no_deform['true2d_gain']['NSD'])} 个百分点。",
                "- 单 seed 只能筛选结构；最终版本仍须三 seed、跨数据集和跨 Base 复核。",
            ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", required=True)
    parser.add_argument("--protocol", choices=("formal55", "causal56", "lest57"), default="formal55")
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    logs_dir = Path(args.logs_dir)
    grouped: dict[str, list[Path]] = defaultdict(list)
    accepted_prefixes = (
        ("GEOTR_M1_LEST57_", "GEOTR_M1_LEST5701_")
        if args.protocol == "lest57"
        else (f"GEOTR_M1_{args.protocol.upper()}_",)
    )
    for path in sorted(logs_dir.glob("*.log")):
        if not any(prefix in path.name for prefix in accepted_prefixes):
            continue
        variant = variant_from_name(path)
        if variant is not None:
            grouped[variant].append(path)
    if not grouped:
        raise SystemExit(f"[FAIL] no {args.protocol.upper()} logs found below {logs_dir}")
    records = {
        variant: merge_logs(paths, args.protocol) for variant, paths in grouped.items()
    }
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    prefix.with_suffix(".md").write_text(
        make_markdown(records, args.protocol), encoding="utf-8"
    )
    print(prefix.with_suffix(".md"))
    print(prefix.with_suffix(".json"))


if __name__ == "__main__":
    main()
