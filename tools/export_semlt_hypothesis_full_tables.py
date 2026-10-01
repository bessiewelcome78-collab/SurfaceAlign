#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export a comprehensive four-dataset SemLT hypothesis diagnosis package.

Reads existing hypothesis_diagnosis/<seed>/<DATASET>/summary.json and
per_case_hypothesis_metrics.csv. It DOES NOT rerun inference or diagnosis.
Outputs detailed CSV/Markdown/LaTeX tables plus a merged per-case CSV and a
complete flattened JSON summary table so no diagnostic field is silently lost.

Stdlib-only: no pandas/openpyxl dependency.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import tarfile
from collections import OrderedDict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

DATASETS = ["BUSI", "BTMRI", "ISIC", "Kvasir"]


def finite(v: Any) -> bool:
    try:
        return math.isfinite(float(v))
    except Exception:
        return False


def fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "--"
    if isinstance(v, bool):
        return "Yes" if v else "No"
    try:
        x = float(v)
        if math.isnan(x):
            return "--"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return f"{x:.{digits}f}"
    except Exception:
        return str(v)


def fmt_pct(v: Any, digits: int = 2) -> str:
    if not finite(v):
        return "--"
    return f"{100.0 * float(v):.{digits}f}%"


def fmt_pp(v: Any, digits: int = 2) -> str:
    if not finite(v):
        return "--"
    return f"{100.0 * float(v):+.{digits}f} pp"


def fmt_p(v: Any) -> str:
    if v is None:
        return "--"
    try:
        x = float(v)
    except Exception:
        return str(v)
    if math.isnan(x):
        return "--"
    if x == 0.0:
        return "< machine precision"
    if x < 1e-4:
        return f"{x:.2e}"
    return f"{x:.4f}"


def stat(d: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    x = d.get(key, {})
    return x if isinstance(x, Mapping) else {}


def sval(d: Mapping[str, Any], key: str, field: str = "mean") -> Any:
    return stat(d, key).get(field)


def macro(summary: Mapping[str, Any], metric: str, r: int, field: str = "mean") -> Any:
    return sval(summary.get("macro", {}), f"{metric}_r{r}", field)


def micro(summary: Mapping[str, Any], metric: str, r: int) -> Any:
    return summary.get("micro", {}).get(f"{metric}_r{r}")


def flatten(obj: Any, prefix: str = "") -> List[Tuple[str, Any]]:
    out: List[Tuple[str, Any]] = []
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            out.extend(flatten(v, p))
    elif isinstance(obj, list):
        if all(not isinstance(v, (dict, list)) for v in obj):
            out.append((prefix, json.dumps(obj, ensure_ascii=False)))
        else:
            for i, v in enumerate(obj):
                out.extend(flatten(v, f"{prefix}[{i}]"))
    else:
        out.append((prefix, obj))
    return out


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        keys: List[str] = []
        seen = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def esc(x: Any) -> str:
        return str(x).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(map(esc, headers)) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(esc(x) for x in row) + " |")
    return "\n".join(lines)


def latex_escape(s: Any) -> str:
    text = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")]:
        text = text.replace(a, b)
    return text


def load_summaries(root: Path, require_all4: bool) -> OrderedDict[str, Dict[str, Any]]:
    data: OrderedDict[str, Dict[str, Any]] = OrderedDict()
    missing = []
    for ds in DATASETS:
        p = root / ds / "summary.json"
        if not p.is_file():
            missing.append(str(p))
            continue
        with p.open(encoding="utf-8") as f:
            data[ds] = json.load(f)
    if require_all4 and len(data) != 4:
        raise SystemExit("[FAIL] Missing dataset summaries:\n  " + "\n  ".join(missing))
    if not data:
        raise SystemExit(f"[FAIL] No summary.json found below {root}")
    return data


def per_case_structure(root: Path, ds: str) -> Dict[str, Any]:
    p = root / ds / "per_case_hypothesis_metrics.csv"
    if not p.is_file():
        return {"dataset": ds}
    n = gt_empty = error_cases = fn_cases = fp_cases = improved = harmed = unchanged = 0
    with p.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            n += 1
            def num(k: str) -> float:
                try:
                    return float(row.get(k, "nan"))
                except Exception:
                    return float("nan")
            if str(row.get("gt_empty", "")).lower() in {"1", "true", "yes"}:
                gt_empty += 1
            if num("error_pixels") > 0: error_cases += 1
            if num("fn_pixels") > 0: fn_cases += 1
            if num("fp_pixels") > 0: fp_cases += 1
            dd = num("delta_dsc")
            if finite(dd):
                if dd > 1e-12: improved += 1
                elif dd < -1e-12: harmed += 1
                else: unchanged += 1
    return {
        "dataset": ds, "n_cases": n, "gt_empty_cases": gt_empty,
        "gt_empty_fraction": gt_empty / n if n else None,
        "cases_with_base_error": error_cases,
        "cases_with_fn": fn_cases, "cases_with_fp": fp_cases,
        "cases_delta_dsc_positive": improved,
        "cases_delta_dsc_negative": harmed,
        "cases_delta_dsc_zero": unchanged,
        "positive_case_fraction_delta_dsc": improved / n if n else None,
    }


def core_row(ds: str, s: Mapping[str, Any], r: int) -> Dict[str, Any]:
    nt = s.get("null_excess_tests", {}).get(f"ltr_excess_fn_r{r}", {})
    nt_fp = s.get("null_excess_tests", {}).get(f"ltr_excess_fp_r{r}", {})
    mech = s.get(f"mechanism_enrichment_r{r}", s.get("mechanism_enrichment_r5", {}))
    od = macro(s, "oracle_delta_dsc", r)
    ad = sval(s, "delta_dsc")
    capture = (float(ad) / float(od)) if finite(ad) and finite(od) and float(od) != 0 else None
    gap = (float(od) - float(ad)) if finite(ad) and finite(od) else None
    return OrderedDict([
        ("Dataset", ds), ("N", s.get("n_cases")),
        ("Base_DSC", sval(s,"base_dsc")), ("Base_DSC_CI_low", sval(s,"base_dsc","ci95_low")), ("Base_DSC_CI_high", sval(s,"base_dsc","ci95_high")),
        ("Refined_DSC", sval(s,"refined_dsc")), ("Refined_DSC_CI_low", sval(s,"refined_dsc","ci95_low")), ("Refined_DSC_CI_high", sval(s,"refined_dsc","ci95_high")),
        ("Actual_Delta_DSC", ad), ("Actual_Delta_DSC_CI_low", sval(s,"delta_dsc","ci95_low")), ("Actual_Delta_DSC_CI_high", sval(s,"delta_dsc","ci95_high")),
        ("Base_NSD_diagnosis", sval(s,"base_nsd")), ("Refined_NSD_diagnosis", sval(s,"refined_nsd")), ("Actual_Delta_NSD_diagnosis", sval(s,"delta_nsd")),
        (f"LTR_FN_r{r}", macro(s,"ltr_fn",r)), (f"LTR_FN_CI_low_r{r}", macro(s,"ltr_fn",r,"ci95_low")), (f"LTR_FN_CI_high_r{r}", macro(s,"ltr_fn",r,"ci95_high")),
        (f"Null_FN_r{r}", macro(s,"ltr_null_fn",r)), (f"Excess_FN_r{r}", macro(s,"ltr_excess_fn",r)), (f"Excess_FN_CI_low_r{r}", macro(s,"ltr_excess_fn",r,"ci95_low")), (f"Excess_FN_CI_high_r{r}", macro(s,"ltr_excess_fn",r,"ci95_high")),
        (f"FN_excess_positive_case_fraction_r{r}", nt.get("positive_case_fraction")), (f"FN_excess_Wilcoxon_greater_p_r{r}", nt.get("wilcoxon_greater_p")),
        (f"LTR_FP_r{r}", macro(s,"ltr_fp",r)), (f"Null_FP_r{r}", macro(s,"ltr_null_fp",r)), (f"Excess_FP_r{r}", macro(s,"ltr_excess_fp",r)),
        (f"FP_excess_positive_case_fraction_r{r}", nt_fp.get("positive_case_fraction")), (f"FP_excess_Wilcoxon_greater_p_r{r}", nt_fp.get("wilcoxon_greater_p")),
        (f"BER_r{r}", macro(s,"ber_all",r)), (f"BER_CI_low_r{r}", macro(s,"ber_all",r,"ci95_low")), (f"BER_CI_high_r{r}", macro(s,"ber_all",r,"ci95_high")),
        (f"Oracle_Delta_DSC_r{r}", od), (f"Oracle_Delta_DSC_CI_low_r{r}", macro(s,"oracle_delta_dsc",r,"ci95_low")), (f"Oracle_Delta_DSC_CI_high_r{r}", macro(s,"oracle_delta_dsc",r,"ci95_high")),
        (f"Oracle_Delta_NSD_r{r}", macro(s,"oracle_delta_nsd",r)),
        ("Opportunity_gap_DSC", gap), ("Oracle_capture_ratio_Actual_over_Oracle", capture),
        ("Error_correction_recall", sval(s,"correction_recall")),
        ("Introduced_error_rate_over_Base_correct", sval(s,"introduced_error_rate_over_base_correct")),
        (f"Corrected_error_transportable_precision_r{r}", sval(s,f"correction_transportable_precision_r{r}") if f"correction_transportable_precision_r{r}" in s else sval(s,"correction_transportable_precision_r5")),
        (f"Transportable_error_realization_r{r}", sval(s,f"transportable_realization_r{r}") if f"transportable_realization_r{r}" in s else sval(s,"transportable_realization_r5")),
        (f"Base_transportable_error_fraction_r{r}", sval(s,f"base_transportable_error_fraction_r{r}") if f"base_transportable_error_fraction_r{r}" in s else sval(s,"base_transportable_error_fraction_r5")),
        (f"Correction_transport_enrichment_diff_r{r}", sval(s,f"correction_transport_enrichment_diff_r{r}") if f"correction_transport_enrichment_diff_r{r}" in s else sval(s,"correction_transport_enrichment_diff_r5")),
        (f"Mechanism_odds_ratio_r{r}", mech.get("odds_ratio")), (f"Mechanism_Fisher_greater_p_r{r}", mech.get("fisher_greater_p")),
        ("Recommended_displacement_cap_px", s.get("radius_recommendation",{}).get("recommended_px")),
        ("Max_macro_oracle_delta_DSC", s.get("radius_recommendation",{}).get("max_macro_oracle_delta_dsc")),
    ])


def make_report(root: Path, out: Path, summaries: OrderedDict[str, Dict[str, Any]], r: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    core = [core_row(ds, s, r) for ds, s in summaries.items()]
    write_csv(out / "01_core_hypothesis_r5.csv", core)

    structures = [per_case_structure(root, ds) for ds in summaries]
    write_csv(out / "02_case_structure.csv", structures)

    # Full radius macro profile.
    radius_rows = []
    for ds, s in summaries.items():
        for rr in s.get("radii", []):
            rr = int(rr)
            row = OrderedDict([("Dataset", ds), ("r", rr)])
            for metric in ["ltr_all","ltr_fn","ltr_fp","ber_all","ltr_null_all","ltr_null_fn","ltr_null_fp","ltr_excess_all","ltr_excess_fn","ltr_excess_fp","oracle_delta_dsc","oracle_delta_nsd"]:
                for fld, suf in [("mean","mean"),("ci95_low","ci_low"),("ci95_high","ci_high"),("n","n")]:
                    row[f"{metric}_{suf}"] = macro(s, metric, rr, fld)
            for metric in ["ltr_all","ltr_fn","ltr_fp","ber_all"]:
                row[f"micro_{metric}"] = micro(s, metric, rr)
            for typ in ["all","fn","fp"]:
                nt = s.get("null_excess_tests", {}).get(f"ltr_excess_{typ}_r{rr}", {})
                row[f"excess_{typ}_positive_case_fraction"] = nt.get("positive_case_fraction")
                row[f"excess_{typ}_wilcoxon_greater_p"] = nt.get("wilcoxon_greater_p")
            radius_rows.append(row)
    write_csv(out / "03_radius_profile_full.csv", radius_rows)

    # Performance / realization table with CIs.
    perf_rows = []
    for ds, s in summaries.items():
        row = OrderedDict([("Dataset", ds), ("N", s.get("n_cases"))])
        for k in ["base_dsc","refined_dsc","delta_dsc","base_nsd","refined_nsd","delta_nsd","correction_recall","introduced_error_rate_over_base_correct","correction_transportable_precision_r5","transportable_realization_r5","base_transportable_error_fraction_r5","correction_transport_enrichment_diff_r5"]:
            st = stat(s,k)
            row[f"{k}_mean"] = st.get("mean")
            row[f"{k}_ci_low"] = st.get("ci95_low")
            row[f"{k}_ci_high"] = st.get("ci95_high")
            row[f"{k}_n"] = st.get("n")
        oracle = macro(s,"oracle_delta_dsc",r)
        actual = sval(s,"delta_dsc")
        row["oracle_delta_dsc_r5"] = oracle
        row["actual_over_oracle"] = float(actual)/float(oracle) if finite(actual) and finite(oracle) and float(oracle) != 0 else None
        row["oracle_minus_actual_gap"] = float(oracle)-float(actual) if finite(actual) and finite(oracle) else None
        perf_rows.append(row)
    write_csv(out / "04_performance_and_realization.csv", perf_rows)

    # Correlations.
    corr_rows = []
    for ds, s in summaries.items():
        for name, c in s.get("correlations", {}).items():
            corr_rows.append(OrderedDict([
                ("Dataset", ds), ("Analysis", name), ("N", c.get("n")),
                ("Pearson_r", c.get("pearson_r")), ("Pearson_p", c.get("pearson_p")),
                ("Spearman_rho", c.get("spearman_rho")), ("Spearman_p", c.get("spearman_p")),
            ]))
    write_csv(out / "05_case_level_correlations.csv", corr_rows)

    # Null significance.
    null_rows = []
    for ds, s in summaries.items():
        for name, x in s.get("null_excess_tests", {}).items():
            null_rows.append(OrderedDict([
                ("Dataset", ds), ("Test", name), ("N", x.get("n")),
                ("Positive_case_fraction", x.get("positive_case_fraction")),
                ("Wilcoxon_greater_p", x.get("wilcoxon_greater_p")),
            ]))
    write_csv(out / "06_null_excess_significance.csv", null_rows)

    # Mechanism enrichment.
    mech_rows = []
    for ds, s in summaries.items():
        key = f"mechanism_enrichment_r{r}"
        m = s.get(key, s.get("mechanism_enrichment_r5", {}))
        table = m.get("table_corrected_vs_remaining__transportable_vs_nontransportable", [[None,None],[None,None]])
        mech_rows.append(OrderedDict([
            ("Dataset", ds), ("r", r),
            ("Corrected_transportable", table[0][0] if len(table)>0 and len(table[0])>0 else None),
            ("Corrected_nontransportable", table[0][1] if len(table)>0 and len(table[0])>1 else None),
            ("Remaining_transportable", table[1][0] if len(table)>1 and len(table[1])>0 else None),
            ("Remaining_nontransportable", table[1][1] if len(table)>1 and len(table[1])>1 else None),
            ("Odds_ratio", m.get("odds_ratio")), ("Fisher_greater_p", m.get("fisher_greater_p")),
        ]))
    write_csv(out / "07_mechanism_enrichment.csv", mech_rows)

    # Radius recommendations.
    rec_rows = []
    for ds, s in summaries.items():
        x = s.get("radius_recommendation", {})
        rec_rows.append({"Dataset": ds, **x})
    write_csv(out / "08_radius_recommendation.csv", rec_rows)

    # Definitions / protocol.
    def_rows = []
    for ds, s in summaries.items():
        for k,v in s.get("definitions",{}).items():
            def_rows.append({"Dataset": ds, "Term": k, "Definition": v})
    write_csv(out / "09_definitions.csv", def_rows)

    # Complete flatten table: guarantees every summary.json field is exported.
    flat_rows = []
    for ds, s in summaries.items():
        for key, val in flatten(s):
            flat_rows.append({"Dataset": ds, "JSON_path": key, "Value": val})
    write_csv(out / "10_summary_json_complete_flatten.csv", flat_rows)

    # Merge per-case tables (all 127+ columns) with dataset name.
    all_headers: List[str] = ["dataset"]
    seen = {"dataset"}
    per_paths = []
    for ds in summaries:
        p = root / ds / "per_case_hypothesis_metrics.csv"
        if p.is_file():
            with p.open(newline="", encoding="utf-8-sig") as f:
                rdr = csv.DictReader(f)
                for h in (rdr.fieldnames or []):
                    if h not in seen:
                        seen.add(h); all_headers.append(h)
            per_paths.append((ds,p))
    with (out / "11_per_case_all_datasets.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=all_headers)
        w.writeheader()
        for ds,p in per_paths:
            with p.open(newline="", encoding="utf-8-sig") as src:
                for row in csv.DictReader(src):
                    w.writerow({"dataset": ds, **row})

    # Index/copy the per-dataset raw summaries and figures; also index formal paired artifacts.
    index_rows = []
    raw_out = out / "raw_and_figures"
    raw_out.mkdir(exist_ok=True)
    for ds in summaries:
        dsout = raw_out / ds; dsout.mkdir(exist_ok=True)
        for name in ["summary.json","summary.md","per_case_hypothesis_metrics.csv","ltr_profile.png","hypothesis_profile.png","motivation_figure_representative.png","motivation_figure_mechanism_rich.png"]:
            src = root / ds / name
            if src.is_file():
                shutil.copy2(src, dsout / name)
                index_rows.append({"Dataset": ds,"Type":"diagnosis","Name":name,"Source":str(src),"Export":str(dsout/name)})
        # Find formal paired reports upward under the same experiment run tree.
        exp_run = root.parents[2] if len(root.parents) >= 3 else root
        candidates = list(exp_run.glob(f"{ds}/seed*/formal_test/{ds}/seg_results/seed*/paired_true2d.*")) + list(exp_run.glob(f"{ds}/seed*/formal_test/{ds}/seg_results/seed*/paired_paper_legacy.*"))
        formalout = dsout / "formal_test_reports"; formalout.mkdir(exist_ok=True)
        for src in candidates:
            dst = formalout / src.name
            shutil.copy2(src,dst)
            index_rows.append({"Dataset": ds,"Type":"formal_test","Name":src.name,"Source":str(src),"Export":str(dst)})
    write_csv(out / "12_artifact_index.csv", index_rows)

    # Cross-dataset aggregate summary.
    agg_rows=[]
    metrics=[f"LTR_FN_r{r}",f"Null_FN_r{r}",f"Excess_FN_r{r}",f"LTR_FP_r{r}",f"Null_FP_r{r}",f"Excess_FP_r{r}",f"BER_r{r}",f"Oracle_Delta_DSC_r{r}","Actual_Delta_DSC","Oracle_capture_ratio_Actual_over_Oracle","Error_correction_recall",f"Transportable_error_realization_r{r}"]
    Ns=[float(x.get("N") or 0) for x in core]
    for m in metrics:
        vals=[float(x[m]) for x in core if finite(x.get(m))]
        if not vals: continue
        macro_avg=sum(vals)/len(vals)
        pairs=[(float(x[m]),float(x.get("N") or 0)) for x in core if finite(x.get(m)) and float(x.get("N") or 0)>0]
        wavg=sum(v*n for v,n in pairs)/sum(n for _,n in pairs) if pairs else None
        agg_rows.append({"Metric":m,"Dataset_macro_mean":macro_avg,"Case_count_weighted_mean":wavg,"N_datasets":len(vals),"Note":"Case-weighted mean can be dominated by BTMRI; use dataset-macro mean for cross-dataset narrative."})
    write_csv(out / "13_cross_dataset_aggregates.csv", agg_rows)

    # Comprehensive Markdown report.
    lines=[]
    lines.append("# SemLT four-dataset hypothesis diagnosis — comprehensive report\n")
    lines.append(f"- Source root: `{root}`")
    lines.append(f"- Datasets: {', '.join(summaries.keys())}")
    lines.append(f"- Mechanism radius: **r={r} model-grid pixels**")
    lines.append("- This is post-hoc diagnosis: GT is used for diagnosis/oracle construction only, not as inference input.\n")

    headers=["Dataset","N","BER@5","FN-LTR@5","Null-FN","Excess-FN","FN Wilcoxon p","Oracle ΔDSC","Actual ΔDSC","Actual/Oracle","Realization","Intro. error rate"]
    mdrows=[]
    for x in core:
        mdrows.append([
            x["Dataset"],x["N"],fmt_pct(x[f"BER_r{r}"]),fmt_pct(x[f"LTR_FN_r{r}"]),fmt_pct(x[f"Null_FN_r{r}"]),fmt_pp(x[f"Excess_FN_r{r}"]),fmt_p(x[f"FN_excess_Wilcoxon_greater_p_r{r}"]),
            fmt_pp(x[f"Oracle_Delta_DSC_r{r}"]),fmt_pp(x["Actual_Delta_DSC"]),fmt_pct(x["Oracle_capture_ratio_Actual_over_Oracle"]),fmt_pct(x[f"Transportable_error_realization_r{r}"]),fmt_pct(x["Introduced_error_rate_over_Base_correct"])
        ])
    lines.append("## 1. Core hypothesis evidence\n")
    lines.append(markdown_table(headers,mdrows)+"\n")
    lines.append("**Interpretation rule:** the geometry/local-recoverability claim should be based jointly on boundary concentration (BER), FN-LTR above the spatial-shift null (Excess-FN + one-sided Wilcoxon), and positive local-oracle gain. LTR alone is insufficient.\n")

    headers2=["Dataset","Base DSC","SemLT DSC","ΔDSC [95% CI]","Base diag-NSD","SemLT diag-NSD","Δdiag-NSD","Correction recall","Transportable precision","Transportable realization","Enrichment","Fisher p"]
    rows2=[]
    for x in core:
        rows2.append([x["Dataset"],fmt_pct(x["Base_DSC"]),fmt_pct(x["Refined_DSC"]),f"{fmt_pp(x['Actual_Delta_DSC'])} [{fmt_pp(x['Actual_Delta_DSC_CI_low'])}, {fmt_pp(x['Actual_Delta_DSC_CI_high'])}]",fmt_pct(x["Base_NSD_diagnosis"]),fmt_pct(x["Refined_NSD_diagnosis"]),fmt_pp(x["Actual_Delta_NSD_diagnosis"]),fmt_pct(x["Error_correction_recall"]),fmt_pct(x[f"Corrected_error_transportable_precision_r{r}"]),fmt_pct(x[f"Transportable_error_realization_r{r}"]),fmt_pp(x[f"Correction_transport_enrichment_diff_r{r}"]),fmt_p(x[f"Mechanism_Fisher_greater_p_r{r}"])])
    lines.append("## 2. Actual SemLT performance and mechanism realization\n")
    lines.append(markdown_table(headers2,rows2)+"\n")
    lines.append("`diag-NSD` above is the NSD implementation stored by the hypothesis diagnosis. For the paper's formal NSD protocol, use the copied `paired_true2d.*` reports in `raw_and_figures/<Dataset>/formal_test_reports/`.\n")

    lines.append("## 3. Full radius profile\n")
    hr=["Dataset","r","BER","FN-LTR","Null-FN","Excess-FN","FP-LTR","Null-FP","Excess-FP","Oracle ΔDSC","Oracle ΔNSD"]
    rrrows=[]
    for row in radius_rows:
        rrrows.append([row["Dataset"],row["r"],fmt_pct(row["ber_all_mean"]),fmt_pct(row["ltr_fn_mean"]),fmt_pct(row["ltr_null_fn_mean"]),fmt_pp(row["ltr_excess_fn_mean"]),fmt_pct(row["ltr_fp_mean"]),fmt_pct(row["ltr_null_fp_mean"]),fmt_pp(row["ltr_excess_fp_mean"]),fmt_pp(row["oracle_delta_dsc_mean"]),fmt_pp(row["oracle_delta_nsd_mean"])])
    lines.append(markdown_table(hr,rrrows)+"\n")

    lines.append("## 4. Dataset-specific evidence statements\n")
    for x in core:
        ds=x["Dataset"]
        fn_p=x[f"FN_excess_Wilcoxon_greater_p_r{r}"]
        fp_p=x[f"FP_excess_Wilcoxon_greater_p_r{r}"]
        lines.append(f"### {ds}")
        lines.append(f"- Boundary concentration: BER@{r} = **{fmt_pct(x[f'BER_r{r}'])}**.")
        lines.append(f"- FN local recoverability: FN-LTR@{r} = **{fmt_pct(x[f'LTR_FN_r{r}'])}**, shift-null = {fmt_pct(x[f'Null_FN_r{r}'])}, Excess-FN = **{fmt_pp(x[f'Excess_FN_r{r}'])}**, one-sided Wilcoxon p = {fmt_p(fn_p)}.")
        lines.append(f"- Local recoverable upper bound: oracle ΔDSC@{r} = **{fmt_pp(x[f'Oracle_Delta_DSC_r{r}'])}**; actual SemLT ΔDSC = **{fmt_pp(x['Actual_Delta_DSC'])}**; current capture = **{fmt_pct(x['Oracle_capture_ratio_Actual_over_Oracle'])}**.")
        lines.append(f"- Mechanism: corrected-error transportable precision = {fmt_pct(x[f'Corrected_error_transportable_precision_r{r}'])}; transportable realization = {fmt_pct(x[f'Transportable_error_realization_r{r}'])}; enrichment = {fmt_pp(x[f'Correction_transport_enrichment_diff_r{r}'])}; Fisher p = {fmt_p(x[f'Mechanism_Fisher_greater_p_r{r}'])}.")
        lines.append(f"- FP evidence is reported separately: Excess-FP@{r} = {fmt_pp(x[f'Excess_FP_r{r}'])}, Wilcoxon p = {fmt_p(fp_p)}. Do not infer FN/FP symmetry unless this statistic supports it.\n")

    lines.append("## 5. Statistical and causal reporting cautions\n")
    lines.append("- Do **not** claim that host semantics are globally correct. The supported claim is local availability of host-correct target-class decisions around a subset of residual errors.")
    lines.append("- Do **not** use LTR alone. Report FN/FP separately and jointly interpret BER, shift-null Excess-LTR, local-oracle gain, actual correction enrichment, and realization.")
    lines.append("- BUSI/BTMRI contain empty-GT cases, so FP-LTR is especially susceptible to background prevalence; the shift-null control is essential.")
    lines.append("- `Actual/Oracle` measures an opportunity–realization gap; a small value supports improving source identification/transport realization, not necessarily rejecting the transport hypothesis.")
    lines.append("- Cross-dataset weighted averages are dominated by BTMRI (1005 cases); prefer dataset-macro averages when making modality-level claims.\n")

    lines.append("## 6. Exported files\n")
    lines.append("- `01_core_hypothesis_r5.csv`: compact but comprehensive r=5 evidence table.")
    lines.append("- `03_radius_profile_full.csv`: all radii, macro CI, micro metrics, null tests, oracle gains.")
    lines.append("- `04_performance_and_realization.csv`: Base/SemLT performance, CIs, correction/harm/realization.")
    lines.append("- `05_case_level_correlations.csv`: Pearson/Spearman mechanism correlations.")
    lines.append("- `06_null_excess_significance.csv`: all one-sided Wilcoxon Excess-LTR tests.")
    lines.append("- `07_mechanism_enrichment.csv`: pooled Fisher contingency tables and odds ratios.")
    lines.append("- `10_summary_json_complete_flatten.csv`: every field in every `summary.json` (no omitted analysis result).")
    lines.append("- `11_per_case_all_datasets.csv`: all per-case diagnostic columns merged across four datasets.")
    lines.append("- `raw_and_figures/`: original summaries, per-case CSVs, diagnosis plots, motivation figures, and formal paired reports where available.\n")
    (out/"00_FULL_DIAGNOSIS_REPORT.md").write_text("\n".join(lines),encoding="utf-8")

    # Paper-ready LaTeX core tables.
    tex=[]
    tex.append(r"% Auto-generated SemLT diagnosis tables. Values are fractions converted to percentages where noted.")
    tex.append(r"\begin{table*}[t]")
    tex.append(r"\centering\small")
    tex.append(r"\caption{Hypothesis verification at $r=5$ model-grid pixels. BER measures boundary concentration; FN-LTR is compared with an exact circular-shift spatial null; the local oracle changes only transportable Base errors and never alters Base-correct pixels.}")
    tex.append(r"\label{tab:semlt_hypothesis}")
    tex.append(r"\begin{tabular}{lrrrrrrrr}")
    tex.append(r"\toprule")
    tex.append(r"Dataset & BER$@5$ & FN-LTR$@5$ & Null & Excess & Oracle $\Delta$DSC & Actual $\Delta$DSC & Capture & Realize. \\")
    tex.append(r"\midrule")
    for x in core:
        tex.append(f"{latex_escape(x['Dataset'])} & {100*x[f'BER_r{r}']:.2f} & {100*x[f'LTR_FN_r{r}']:.2f} & {100*x[f'Null_FN_r{r}']:.2f} & {100*x[f'Excess_FN_r{r}']:.2f} & {100*x[f'Oracle_Delta_DSC_r{r}']:.2f} & {100*x['Actual_Delta_DSC']:.2f} & {100*x['Oracle_capture_ratio_Actual_over_Oracle']:.1f} & {100*x[f'Transportable_error_realization_r{r}']:.1f} \\")
    tex.append(r"\bottomrule\end{tabular}")
    tex.append(r"\end{table*}")
    (out/"14_paper_tables.tex").write_text("\n".join(tex),encoding="utf-8")

    # Bundle everything.
    bundle = out.parent / f"{out.name}.tar.gz"
    with tarfile.open(bundle,"w:gz") as tf:
        tf.add(out, arcname=out.name)

    print("[PASS] Comprehensive SemLT diagnosis export complete")
    print(f"       report = {out/'00_FULL_DIAGNOSIS_REPORT.md'}")
    print(f"       core   = {out/'01_core_hypothesis_r5.csv'}")
    print(f"       radius = {out/'03_radius_profile_full.csv'}")
    print(f"       percase= {out/'11_per_case_all_datasets.csv'}")
    print(f"       latex  = {out/'14_paper_tables.tex'}")
    print(f"       bundle = {bundle}")


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True,type=Path,help=".../hypothesis_diagnosis/seed42")
    ap.add_argument("--output-dir",required=True,type=Path)
    ap.add_argument("--radius",type=int,default=5)
    ap.add_argument("--allow-missing",action="store_true",help="Allow <4 summaries (for debugging only)")
    args=ap.parse_args()
    summaries=load_summaries(args.root, require_all4=not args.allow_missing)
    make_report(args.root,args.output_dir,summaries,args.radius)

if __name__=="__main__":
    main()
