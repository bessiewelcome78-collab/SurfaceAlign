#!/usr/bin/env python3
"""Collect the five-row seed-42 paper table and paired FULL-vs-control effects."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ARMS = (
    ("A000_JOINT_CONTROL", "joint control (no three constraints)"),
    ("A011_NO_ALIGN", "w/o operator-coordinate alignment"),
    ("A101_NO_CDF", "CDF proper score -> mean regression"),
    ("A110_NO_REACH", "w/o reachable-domain matching"),
    ("FULL", "full method"),
)


def lock_value(path: Path, key: str) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(key + "="):
            return line.split("=", 1)[1]
    raise RuntimeError(f"{path}: missing {key}")


def load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if len(frame) != 78 or not {"Case_ID", "DSC", "NSD"}.issubset(frame.columns):
        raise RuntimeError(f"invalid BUSI result CSV: {path}")
    return frame.set_index("Case_ID").sort_index()


def paired_effect(full: np.ndarray, control: np.ndarray, rng: np.random.Generator) -> dict:
    delta = full - control
    indices = rng.integers(0, len(delta), size=(20000, len(delta)))
    means = delta[indices].mean(axis=1)
    return {
        "mean": float(delta.mean()),
        "median": float(np.median(delta)),
        "ci95": [float(x) for x in np.quantile(means, [0.025, 0.975])],
        "full_better_cases": int((delta > 1e-12).sum()),
        "control_better_cases": int((delta < -1e-12).sum()),
        "equal_cases": int((np.abs(delta) <= 1e-12).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", required=True)
    args = parser.parse_args()
    root = Path(args.study_root).resolve()
    if not (root / "formal_state/CHECKPOINT_PARITY_SUCCESS.lock").is_file():
        raise SystemExit("[FAIL] checkpoint parity audit is missing")
    full_lock = root / "formal_state/FULL_REFERENCE.lock"
    full_root = Path(lock_value(full_lock, "result_root"))

    roots = {"FULL": full_root}
    for arm, _ in ARMS[:-1]:
        lock = root / arm / "formal_state/EVAL_SUCCESS.lock"
        if not lock.is_file():
            raise SystemExit(f"[FAIL] evaluation incomplete: {arm}")
        roots[arm] = Path(lock_value(lock, "result_root"))

    data: dict[str, dict[str, dict[str, pd.DataFrame]]] = {}
    for arm, _ in ARMS:
        data[arm] = {}
        for protocol in ("true2d", "paper_legacy"):
            data[arm][protocol] = {
                "base": load(roots[arm] / f"test_BaseNative_{protocol}.csv"),
                "m1": load(roots[arm] / f"test_M1Native_{protocol}.csv"),
            }

    reference_index = data["FULL"]["true2d"]["base"].index
    reference_base = data["FULL"]["true2d"]["base"][["DSC", "NSD"]].to_numpy()
    for arm, _ in ARMS:
        for protocol in ("true2d", "paper_legacy"):
            for kind in ("base", "m1"):
                if not data[arm][protocol][kind].index.equals(reference_index):
                    raise SystemExit(f"[FAIL] Case_ID mismatch: {arm}/{protocol}/{kind}")
        current = data[arm]["true2d"]["base"][["DSC", "NSD"]].to_numpy()
        if not np.array_equal(reference_base, current):
            raise SystemExit(f"[FAIL] Base per-case metrics are not bit-identical: {arm}")

    rows = []
    effects = {}
    rng = np.random.default_rng(42)
    full_true = data["FULL"]["true2d"]["m1"]
    for arm, label in ARMS:
        true_base, true_m1 = data[arm]["true2d"]["base"], data[arm]["true2d"]["m1"]
        paper_m1 = data[arm]["paper_legacy"]["m1"]
        rows.append({
            "arm": arm,
            "paper_label": label,
            "base_dsc": true_base["DSC"].mean(),
            "m1_dsc": true_m1["DSC"].mean(),
            "delta_dsc_vs_base": (true_m1["DSC"] - true_base["DSC"]).mean(),
            "m1_true2d_nsd": true_m1["NSD"].mean(),
            "delta_true2d_nsd_vs_base": (true_m1["NSD"] - true_base["NSD"]).mean(),
            "m1_paper_legacy_nsd": paper_m1["NSD"].mean(),
        })
        if arm != "FULL":
            effects[arm] = {
                metric: paired_effect(full_true[metric].to_numpy(), true_m1[metric].to_numpy(), rng)
                for metric in ("DSC", "NSD")
            }

    table = pd.DataFrame(rows)
    csv_path = root / "BUSI_OACD_MINABL100_SEED42_TABLE.csv"
    json_path = root / "BUSI_OACD_MINABL100_SEED42_EFFECTS.json"
    md_path = root / "BUSI_OACD_MINABL100_SEED42_REPORT.md"
    table.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(effects, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# BUSI OACD minimal ablation (seed=42)", "",
        "> 单训练种子只能作为 seed-42 机制证据；bootstrap 区间是病例不确定性，不是跨训练种子显著性。", "",
        "| Arm | Setting | DSC | ΔDSC vs Base | true-2D NSD | ΔNSD vs Base | paper-legacy NSD |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['arm']} | {row['paper_label']} | {100*row['m1_dsc']:.2f} | "
            f"{100*row['delta_dsc_vs_base']:+.2f} | {100*row['m1_true2d_nsd']:.2f} | "
            f"{100*row['delta_true2d_nsd_vs_base']:+.2f} | {100*row['m1_paper_legacy_nsd']:.2f} |"
        )
    lines.extend(["", "## FULL relative to each control (paired BUSI cases)", ""])
    for arm, item in effects.items():
        lines.append(
            f"- {arm}: DSC {100*item['DSC']['mean']:+.2f} pp "
            f"(95% case-bootstrap [{100*item['DSC']['ci95'][0]:+.2f}, {100*item['DSC']['ci95'][1]:+.2f}]); "
            f"true-2D NSD {100*item['NSD']['mean']:+.2f} pp "
            f"([{100*item['NSD']['ci95'][0]:+.2f}, {100*item['NSD']['ci95'][1]:+.2f}])."
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "formal_state/FORMAL_STUDY_SUCCESS.lock").write_text(
        f"seed=42\nreport={md_path}\ntable={csv_path}\n", encoding="utf-8"
    )
    print(f"[PASS] five-row seed-42 minimal ablation collected: {md_path}")


if __name__ == "__main__":
    main()
