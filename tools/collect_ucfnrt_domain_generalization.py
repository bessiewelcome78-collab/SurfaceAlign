#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PAPER_MEDCLIPSEG = {
    "BUSI": 85.72,
    "BUSBRA": 75.06,
    "BUSUC": 84.37,
    "BUID": 78.99,
    "UDIAT": 74.64,
    "Kvasir-SEG": 90.15,
    "ColonDB": 71.90,
    "ClinicDB": 80.80,
    "CVC300": 80.82,
    "BKAI": 79.15,
    "BTMRI": 88.03,
    "BRISC": 80.92,
    "ISIC": 92.54,
    "UWaterloo": 83.53,
}

SOURCE_RESULT_CSV = {
    "BUSI": "runs/UCFNRT_FORMAL100_BUSI_S42_20260831_032834/formal_test/BUSI/seg_results/seed42/test_M1Native_paper_legacy.csv",
    "Kvasir-SEG": "runs/UCFNRT_FORMAL100_Kvasir_S42_20260831_061201/formal_test/Kvasir/seg_results/seed42/test_M1Native_paper_legacy.csv",
    "BTMRI": "runs/UCFNRT_FORMAL100_BTMRI_S42_20260831_032834/formal_test/BTMRI/seg_results/seed42/test_M1Native_paper_legacy.csv",
    "ISIC": "runs/UCFNRT_FORMAL100_ISIC_S42_20260831_032834/formal_test/ISIC/seg_results/seed42/test_M1Native_paper_legacy.csv",
}

ORDER = [
    "BUSI", "BUSBRA", "BUSUC", "BUID", "UDIAT",
    "Kvasir-SEG", "ColonDB", "ClinicDB", "CVC300", "BKAI",
    "BTMRI", "BRISC", "ISIC", "UWaterloo",
]


def mean_metric_csv(path: Path) -> tuple[int, float, float]:
    df = pd.read_csv(path)
    return len(df), float(df["DSC"].mean() * 100.0), float(df["NSD"].mean() * 100.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()
    project = Path(args.project).resolve()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    run_root = Path(manifest["run_root"])
    final_dir = run_root / "final_table"
    final_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    # Existing source-domain results: no retraining/retesting for DG.
    for display, rel in SOURCE_RESULT_CSV.items():
        p = project / rel
        if not p.is_file():
            raise FileNotFoundError(f"Missing existing source result CSV: {p}")
        n, dsc, nsd = mean_metric_csv(p)
        rows.append({
            "Dataset": display,
            "Role": "Source (existing)",
            "Cases": n,
            "DSC_percent": dsc,
            "NSD_paper_legacy_percent": nsd,
            "NSD_true2d_percent": None,
            "Source": display,
            "Result": str(p),
        })

    for e in manifest["targets"]:
        result_json = Path(e["state_dir"]) / "RESULT.json"
        if not result_json.is_file():
            raise FileNotFoundError(f"Target incomplete: {result_json}")
        r = json.loads(result_json.read_text(encoding="utf-8"))
        rows.append({
            "Dataset": r["display_target"],
            "Role": "OOD target",
            "Cases": r["cases"],
            "DSC_percent": r["m1_dsc_percent"],
            "NSD_paper_legacy_percent": r["m1_nsd_paper_legacy_percent"],
            "NSD_true2d_percent": r["m1_nsd_true2d_percent"],
            "Source": r["display_source"],
            "Result": r["result_root"],
        })

    df = pd.DataFrame(rows)
    rank = {k: i for i, k in enumerate(ORDER)}
    df["_order"] = df["Dataset"].map(rank)
    df = df.sort_values("_order").drop(columns="_order")
    df.to_csv(final_dir / "UC_FNRT_DOMAIN_GENERALIZATION_TABLE2_LONG.csv", index=False)

    vals = {r["Dataset"]: float(r["DSC_percent"]) for _, r in df.iterrows()}
    wide = pd.DataFrame([{k: vals[k] for k in ORDER}])
    wide.to_csv(final_dir / "UC_FNRT_DOMAIN_GENERALIZATION_TABLE2_WIDE.csv", index=False)

    comp = pd.DataFrame([
        {
            "Dataset": k,
            "MedCLIPSeg_paper_DSC": PAPER_MEDCLIPSEG[k],
            "UC_FNRT_DSC": vals[k],
            "Delta_pp": vals[k] - PAPER_MEDCLIPSEG[k],
        }
        for k in ORDER
    ])
    comp.to_csv(final_dir / "UC_FNRT_VS_MEDCLIPSEG_TABLE2.csv", index=False)

    latex_vals = " & ".join(f"{vals[k]:.2f}" for k in ORDER)
    (final_dir / "UC_FNRT_TABLE2_LATEX_ROW.tex").write_text(
        "\\textbf{UC-FNRT (Ours)} & " + latex_vals + " \\\\\n",
        encoding="utf-8",
    )

    source_keys = {"BUSI", "Kvasir-SEG", "BTMRI", "ISIC"}
    id_mean = sum(vals[k] for k in source_keys) / len(source_keys)
    ood_keys = [k for k in ORDER if k not in source_keys]
    ood_mean = sum(vals[k] for k in ood_keys) / len(ood_keys)
    hm = 2 * id_mean * ood_mean / (id_mean + ood_mean)
    summary = {
        "ID_macro_DSC_percent": id_mean,
        "OOD_macro_DSC_percent": ood_mean,
        "HM_DSC_percent": hm,
        "paper_MedCLIPSeg_ID_macro_DSC_percent": sum(PAPER_MEDCLIPSEG[k] for k in source_keys) / len(source_keys),
        "paper_MedCLIPSeg_OOD_macro_DSC_percent": sum(PAPER_MEDCLIPSEG[k] for k in ood_keys) / len(ood_keys),
    }
    (final_dir / "UC_FNRT_DOMAIN_GENERALIZATION_SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("=" * 100)
    print("UC-FNRT DOMAIN GENERALIZATION TABLE 2 (DSC %)" )
    print(" | ".join(ORDER))
    print(" | ".join(f"{vals[k]:.2f}" for k in ORDER))
    print("=" * 100)
    print(f"ID macro DSC  = {id_mean:.2f}%")
    print(f"OOD macro DSC = {ood_mean:.2f}%")
    print(f"HM DSC        = {hm:.2f}%")
    print(f"[PASS] final files: {final_dir}")


if __name__ == "__main__":
    main()
