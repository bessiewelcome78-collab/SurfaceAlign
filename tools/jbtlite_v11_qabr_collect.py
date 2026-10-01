#!/usr/bin/env python3
"""Collect BUSI BASE/EDGE20/QABR/FULL validation results and quantify synergy."""
from __future__ import annotations
import argparse, csv, re
from pathlib import Path

ARMS=("BASE","EDGE20","QABR","FULL")
PAT_DSC=re.compile(r"Average DSC .* split=val:\s*([0-9.]+)%")
PAT_NSD=re.compile(r"Average NSD .* split=val:\s*([0-9.]+)%")
PAT_BEST=re.compile(r"best_val_epoch=(\d+)\s*\|\s*best_selection_value=([0-9.]+)")

ap=argparse.ArgumentParser()
ap.add_argument('--study-root',required=True)
ap.add_argument('--seed',type=int,default=42)
ap.add_argument('--full-vs-base-dsc',type=float,default=0.50)
ap.add_argument('--full-vs-base-nsd',type=float,default=1.00)
ap.add_argument('--module-vs-edge-dsc',type=float,default=0.20)
ap.add_argument('--module-vs-edge-nsd',type=float,default=0.50)
ap.add_argument('--module-alone-dsc',type=float,default=0.00,
                help='QABR-BASE Dice floor; v10 explicitly targets the v9 negative-Dice issue')
ap.add_argument('--module-alone-nsd',type=float,default=0.75)
args=ap.parse_args(); root=Path(args.study_root)
rows={}
for arm in ARMS:
    r=root/'BUSI'/arm/f'seed{args.seed}'
    elog=r/'logs'/'eval_val_mc10_true2d.log'; tlog=r/'logs'/'train.log'
    if not elog.is_file() or not tlog.is_file():
        print(f'{arm:8s} MISSING run_root={r}'); continue
    et=elog.read_text(errors='replace'); tt=tlog.read_text(errors='replace')
    md,mn,mb=PAT_DSC.findall(et),PAT_NSD.findall(et),PAT_BEST.findall(tt)
    if not md or not mn:
        print(f'{arm:8s} INCOMPLETE metrics'); continue
    d,n=float(md[-1]),float(mn[-1])
    ep,sel=(int(mb[-1][0]),float(mb[-1][1])) if mb else (-1,float('nan'))
    rows[arm]=(d,n,ep,sel)

print('\nJBT-Lite v10 QABR | BUSI Validation MC10 | true2d')
print('ARM          DSC      NSD  BEST_EP   SEL_MC10    dDSC    dNSD')
bd,bn=(rows['BASE'][0],rows['BASE'][1]) if 'BASE' in rows else (float('nan'),float('nan'))
for arm in ARMS:
    if arm in rows:
        d,n,ep,sel=rows[arm]
        print(f'{arm:8s} {d:8.2f} {n:8.2f} {ep:8d} {sel:10.6f} {d-bd:+7.2f} {n-bn:+7.2f}')

out=root/f'JBTL11_BUSI_FACTORIAL_S{args.seed}.csv'; out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w',newline='') as f:
    w=csv.writer(f); w.writerow(['arm','dsc_pct','nsd_pct','best_epoch','selection_value','delta_dsc_pp','delta_nsd_pp'])
    for arm in ARMS:
        if arm in rows:
            d,n,ep,sel=rows[arm]; w.writerow([arm,d,n,ep,sel,d-bd,n-bn])
print(f'[CSV] {out}')
if len(rows)!=4:
    print('[VAL_LOCK_NONE] Need all 4 factorial arms.'); raise SystemExit(0)
B,E,Q,F=rows['BASE'],rows['EDGE20'],rows['QABR'],rows['FULL']
edge=(E[0]-B[0],E[1]-B[1]); alone=(Q[0]-B[0],Q[1]-B[1]); onedge=(F[0]-E[0],F[1]-E[1]); total=(F[0]-B[0],F[1]-B[1]); inter=(F[0]-E[0]-Q[0]+B[0],F[1]-E[1]-Q[1]+B[1])
print('\nFactorial decomposition (percentage points)')
print(f'Existing Surface effect (EDGE20-BASE): DSC {edge[0]:+.2f} | NSD {edge[1]:+.2f}')
print(f'QABR-alone effect (QABR-BASE)       : DSC {alone[0]:+.2f} | NSD {alone[1]:+.2f}')
print(f'QABR on Surface (FULL-EDGE20)        : DSC {onedge[0]:+.2f} | NSD {onedge[1]:+.2f}')
print(f'FULL total effect (FULL-BASE)        : DSC {total[0]:+.2f} | NSD {total[1]:+.2f}')
print(f'2x2 interaction                      : DSC {inter[0]:+.2f} | NSD {inter[1]:+.2f}')
pass_total=total[0]>=args.full_vs_base_dsc and total[1]>=args.full_vs_base_nsd
pass_increment=onedge[0]>=args.module_vs_edge_dsc and onedge[1]>=args.module_vs_edge_nsd
pass_alone=alone[0]>=args.module_alone_dsc and alone[1]>=args.module_alone_nsd
positive_synergy=inter[0]>=0.0 and inter[1]>=0.0
if pass_total and pass_increment and pass_alone and positive_synergy:
    print('[VAL_LOCK_FULL] QABR v10 passes total + standalone + incremental + synergy gates. Freeze FULL before Test MC30.')
else:
    print('[VAL_LOCK_NONE] Do NOT open BUSI Test yet.')
    print(f'  total      >= {args.full_vs_base_dsc:.2f}/{args.full_vs_base_nsd:.2f} DSC/NSD pp -> {pass_total}')
    print(f'  standalone >= {args.module_alone_dsc:.2f}/{args.module_alone_nsd:.2f} DSC/NSD pp -> {pass_alone}')
    print(f'  incremental>= {args.module_vs_edge_dsc:.2f}/{args.module_vs_edge_nsd:.2f} DSC/NSD pp -> {pass_increment}')
    print(f'  interaction non-negative in both metrics -> {positive_synergy}')
