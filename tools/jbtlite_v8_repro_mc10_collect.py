#!/usr/bin/env python3
"""Collect BUSI v8 Validation MC10 true2d results and apply the locked gate."""
from __future__ import annotations
import argparse, re
from pathlib import Path

ARMS = ("BASE", "DECODER_EDGE20", "GLOBAL_EDGE20")
PAT_DSC = re.compile(r"Average DSC .* split=val:\s*([0-9.]+)%")
PAT_NSD = re.compile(r"Average NSD .* split=val:\s*([0-9.]+)%")
PAT_BEST = re.compile(r"best_val_epoch=(\d+)\s*\|\s*best_selection_value=([0-9.]+)")

ap = argparse.ArgumentParser()
ap.add_argument("--study-root", required=True)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--gate-dsc-pp", type=float, default=0.30)
ap.add_argument("--gate-nsd-pp", type=float, default=1.00)
args = ap.parse_args()
root = Path(args.study_root)
rows = {}
for arm in ARMS:
    r = root / "BUSI" / arm / f"seed{args.seed}"
    elog = r / "logs" / "eval_val_mc10_true2d.log"
    tlog = r / "logs" / "train.log"
    if not elog.is_file() or not tlog.is_file():
        print(f"{arm:18s} MISSING  run_root={r}")
        continue
    et = elog.read_text(errors="replace")
    tt = tlog.read_text(errors="replace")
    md, mn = PAT_DSC.findall(et), PAT_NSD.findall(et)
    mb = PAT_BEST.findall(tt)
    if not md or not mn:
        print(f"{arm:18s} INCOMPLETE metrics")
        continue
    dsc, nsd = float(md[-1]), float(mn[-1])
    ep, sel = (int(mb[-1][0]), float(mb[-1][1])) if mb else (-1, float('nan'))
    rows[arm] = (dsc, nsd, ep, sel)

print("\nJBT-Lite v8 BUSI Validation MC10 true2d")
print("ARM                 DSC      NSD   BEST_EP  SEL_MC10   dDSC    dNSD")
if "BASE" in rows:
    bd, bn, _, _ = rows["BASE"]
else:
    bd = bn = float('nan')
for arm in ARMS:
    if arm not in rows:
        continue
    d, n, ep, sel = rows[arm]
    dd, dn = d-bd, n-bn
    print(f"{arm:18s} {d:7.2f} {n:8.2f} {ep:8d} {sel:9.6f} {dd:+7.2f} {dn:+7.2f}")

if len(rows) == 3:
    passed = []
    for arm in ("DECODER_EDGE20", "GLOBAL_EDGE20"):
        d, n, _, _ = rows[arm]
        dd, dn = d-bd, n-bn
        ok = dd >= args.gate_dsc_pp and dn >= args.gate_nsd_pp
        passed.append((arm, ok, dd, dn))
        print(f"[VAL_GATE_{'PASS' if ok else 'FAIL'}] {arm}: dDSC={dd:+.2f}pp dNSD={dn:+.2f}pp "
              f"required>={args.gate_dsc_pp:.2f}/{args.gate_nsd_pp:.2f}pp")
    winners = [x for x in passed if x[1]]
    if winners:
        # Predeclared primary ordering: DSC first, then NSD.
        winners.sort(key=lambda x: (x[2], x[3]), reverse=True)
        print(f"[VAL_LOCK_CANDIDATE] {winners[0][0]} -- freeze this arm before any new Test run.")
    else:
        print("[VAL_LOCK_NONE] No EDGE arm passes the predeclared Validation gate; do not open Test.")
