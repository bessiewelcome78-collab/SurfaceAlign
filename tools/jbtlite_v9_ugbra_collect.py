#!/usr/bin/env python3
"""Collect JBT-Lite v9 BUSI 2x2 Validation MC10 results and quantify interaction."""
from __future__ import annotations
import argparse
import csv
import re
from pathlib import Path

ARMS = ("BASE", "EDGE20", "UGBRA", "FULL")
PAT_DSC = re.compile(r"Average DSC .* split=val:\s*([0-9.]+)%")
PAT_NSD = re.compile(r"Average NSD .* split=val:\s*([0-9.]+)%")
PAT_BEST = re.compile(r"best_val_epoch=(\d+)\s*\|\s*best_selection_value=([0-9.]+)")

ap = argparse.ArgumentParser()
ap.add_argument("--study-root", required=True)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--full-vs-base-dsc", type=float, default=0.50,
                help="minimum FULL-BASE DSC improvement (pp)")
ap.add_argument("--full-vs-base-nsd", type=float, default=1.00,
                help="minimum FULL-BASE NSD improvement (pp)")
ap.add_argument("--module-vs-edge-dsc", type=float, default=0.20,
                help="minimum FULL-EDGE20 DSC improvement proving module contribution (pp)")
ap.add_argument("--module-vs-edge-nsd", type=float, default=0.50,
                help="minimum FULL-EDGE20 NSD improvement proving module contribution (pp)")
args = ap.parse_args()
root = Path(args.study_root)
rows = {}
for arm in ARMS:
    r = root / "BUSI" / arm / f"seed{args.seed}"
    elog = r / "logs" / "eval_val_mc10_true2d.log"
    tlog = r / "logs" / "train.log"
    if not elog.is_file() or not tlog.is_file():
        print(f"{arm:8s} MISSING  run_root={r}")
        continue
    et = elog.read_text(errors="replace")
    tt = tlog.read_text(errors="replace")
    md, mn, mb = PAT_DSC.findall(et), PAT_NSD.findall(et), PAT_BEST.findall(tt)
    if not md or not mn:
        print(f"{arm:8s} INCOMPLETE metrics")
        continue
    dsc, nsd = float(md[-1]), float(mn[-1])
    ep, sel = (int(mb[-1][0]), float(mb[-1][1])) if mb else (-1, float("nan"))
    rows[arm] = (dsc, nsd, ep, sel)

print("\nJBT-Lite v9 UGBRA | BUSI Validation MC10 | true2d")
print("ARM          DSC      NSD  BEST_EP   SEL_MC10    dDSC    dNSD")
if "BASE" in rows:
    bd, bn = rows["BASE"][0], rows["BASE"][1]
else:
    bd = bn = float("nan")
for arm in ARMS:
    if arm not in rows:
        continue
    d, n, ep, sel = rows[arm]
    print(f"{arm:8s} {d:8.2f} {n:8.2f} {ep:8d} {sel:10.6f} {d-bd:+7.2f} {n-bn:+7.2f}")

out = root / f"JBTL9_BUSI_FACTORIAL_S{args.seed}.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["arm", "dsc_pct", "nsd_pct", "best_epoch", "selection_value", "delta_dsc_pp", "delta_nsd_pp"])
    for arm in ARMS:
        if arm in rows:
            d, n, ep, sel = rows[arm]
            w.writerow([arm, d, n, ep, sel, d-bd, n-bn])
print(f"[CSV] {out}")

if len(rows) != 4:
    print("[VAL_LOCK_NONE] Need all 4 factorial arms before drawing a module conclusion.")
    raise SystemExit(0)

B, E, U, F = rows["BASE"], rows["EDGE20"], rows["UGBRA"], rows["FULL"]
edge_dsc, edge_nsd = E[0]-B[0], E[1]-B[1]
module_alone_dsc, module_alone_nsd = U[0]-B[0], U[1]-B[1]
full_dsc, full_nsd = F[0]-B[0], F[1]-B[1]
module_on_edge_dsc, module_on_edge_nsd = F[0]-E[0], F[1]-E[1]
interaction_dsc = F[0] - E[0] - U[0] + B[0]
interaction_nsd = F[1] - E[1] - U[1] + B[1]

print("\nFactorial decomposition (percentage points)")
print(f"Existing Surface effect (EDGE20-BASE) : DSC {edge_dsc:+.2f} | NSD {edge_nsd:+.2f}")
print(f"UGBRA-alone effect (UGBRA-BASE)       : DSC {module_alone_dsc:+.2f} | NSD {module_alone_nsd:+.2f}")
print(f"UGBRA on current method (FULL-EDGE20) : DSC {module_on_edge_dsc:+.2f} | NSD {module_on_edge_nsd:+.2f}")
print(f"FULL total effect (FULL-BASE)         : DSC {full_dsc:+.2f} | NSD {full_nsd:+.2f}")
print(f"2x2 interaction                       : DSC {interaction_dsc:+.2f} | NSD {interaction_nsd:+.2f}")

pass_total = full_dsc >= args.full_vs_base_dsc and full_nsd >= args.full_vs_base_nsd
pass_module = module_on_edge_dsc >= args.module_vs_edge_dsc and module_on_edge_nsd >= args.module_vs_edge_nsd
not_cancelling = F[0] >= min(E[0], U[0]) and F[1] >= min(E[1], U[1])
if pass_total and pass_module and not_cancelling:
    print("[VAL_LOCK_FULL] FULL passes total-gain + incremental-module gates. Freeze FULL before BUSI Test MC30.")
else:
    print("[VAL_LOCK_NONE] Do NOT open BUSI Test yet.")
    print(f"  total gate:  FULL-BASE >= {args.full_vs_base_dsc:.2f} DSC / {args.full_vs_base_nsd:.2f} NSD pp -> {pass_total}")
    print(f"  module gate: FULL-EDGE >= {args.module_vs_edge_dsc:.2f} DSC / {args.module_vs_edge_nsd:.2f} NSD pp -> {pass_module}")
    print(f"  no-cancellation sanity -> {not_cancelling}")
