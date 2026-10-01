#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, re
from pathlib import Path
PAT_DSC = re.compile(r"Average DSC .*? split=val:\s*([0-9.]+)%")
PAT_NSD = re.compile(r"Average NSD .*? split=val:\s*([0-9.]+)%")
ARMS = ('BASE','GLOBAL_EDGE20','GLOBAL_BUDGET20')

def parse(path: Path):
    text = path.read_text(errors='ignore')
    d, n = PAT_DSC.findall(text), PAT_NSD.findall(text)
    if not d or not n: raise RuntimeError(f'Validation true2d metrics not found in {path}')
    return float(d[-1]), float(n[-1])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--gate', action='store_true')
    ap.add_argument('--min-full-dsc-gain', type=float, default=0.30)
    ap.add_argument('--min-full-nsd-gain', type=float, default=1.00)
    a=ap.parse_args(); root=Path(a.root)
    rows=[]
    for arm in ARMS:
        p=root/'BUSI'/arm/f'seed{a.seed}'/'logs'/'eval_val_true2d.log'
        if not p.is_file(): raise RuntimeError(f'missing: {p}')
        rows.append((arm,*parse(p)))
    base=rows[0]
    print('ARM\tDSC\tNSD\tΔDSC\tΔNSD')
    for arm,dsc,nsd in rows:
        print(f'{arm}\t{dsc:.2f}\t{nsd:.2f}\t{dsc-base[1]:+.2f}\t{nsd-base[2]:+.2f}')
    out=root/'BUSI'/'JBTL6_GLOBALBUDGET_DIAG40_summary.csv'; out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['arm','dsc','nsd','delta_dsc','delta_nsd'])
        for arm,dsc,nsd in rows: w.writerow([arm,f'{dsc:.4f}',f'{nsd:.4f}',f'{dsc-base[1]:.4f}',f'{nsd-base[2]:.4f}'])
    print(f'summary={out}')
    if a.gate:
        full=rows[2]; dd=full[1]-base[1]; dn=full[2]-base[2]
        if dd >= a.min_full_dsc_gain and dn >= a.min_full_nsd_gain:
            print('[JBTL6_GLOBALBUDGET_VAL_GATE_PASS] A2 clears the predeclared Validation gate. Formal Test may proceed with locked config.')
            return 0
        print('[JBTL6_GLOBALBUDGET_VAL_GATE_FAIL] Do NOT open Test.')
        print(f'  A2 ΔDSC={dd:+.2f} pp; required >= +{a.min_full_dsc_gain:.2f} pp')
        print(f'  A2 ΔNSD={dn:+.2f} pp; required >= +{a.min_full_nsd_gain:.2f} pp')
        return 20
    return 0
if __name__=='__main__': raise SystemExit(main())
