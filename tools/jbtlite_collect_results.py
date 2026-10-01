#!/usr/bin/env python3
"""Collect JBT-Lite v3 metrics and fail closed on weak validation ablations."""
from __future__ import annotations
import argparse, csv, re
from pathlib import Path

PAT_DSC = re.compile(r"Average DSC .*? split=(?:val|test):\s*([0-9.]+)%")
PAT_NSD = re.compile(r"Average NSD .*? split=(?:val|test):\s*([0-9.]+)%")


def parse(path: Path):
    text = path.read_text(errors="ignore")
    d = PAT_DSC.findall(text); n = PAT_NSD.findall(text)
    if not d or not n:
        raise RuntimeError(f"metrics not found: {path}")
    return float(d[-1]), float(n[-1])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--split', choices=['val','test'], required=True)
    ap.add_argument('--gate', action='store_true')
    ap.add_argument('--max-dsc-drop', type=float, default=0.05,
                    help='max tolerated DSC drop vs BASE, in percentage points')
    ap.add_argument('--min-dsc-gain', type=float, default=0.05,
                    help='minimum DSC gain required for each module')
    ap.add_argument('--min-nsd-gain', type=float, default=0.50,
                    help='minimum true2d NSD gain required for each module')
    ap.add_argument('--full-extra-nsd', type=float, default=0.10,
                    help='FULL should beat the stronger single module in NSD by this margin unless DSC adds more')
    args=ap.parse_args()
    root=Path(args.root)
    arms=['BASE','EDGE','NORMAL','FULL']
    rows=[]
    for arm in arms:
        matches=list((root/arm).glob(f'seed*/logs/eval_{args.split}_true2d.log'))
        if len(matches)!=1:
            raise RuntimeError(f"cannot resolve {arm} log under {root}: {matches}")
        dsc,nsd=parse(matches[0]); rows.append([arm,dsc,nsd])
    base=rows[0]
    print('ARM\tDSC\tNSD\tΔDSC\tΔNSD')
    for arm,dsc,nsd in rows:
        print(f'{arm}\t{dsc:.2f}\t{nsd:.2f}\t{dsc-base[1]:+.2f}\t{nsd-base[2]:+.2f}')
    out=root/f'JBTL3_{args.split}_summary.csv'
    with out.open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['arm','dsc','nsd','delta_dsc','delta_nsd'])
        for arm,dsc,nsd in rows:
            w.writerow([arm,f'{dsc:.4f}',f'{nsd:.4f}',f'{dsc-base[1]:.4f}',f'{nsd-base[2]:.4f}'])
    print(f'summary={out}')

    if args.gate:
        failures=[]
        for arm,dsc,nsd in rows[1:3]:
            dd, dn = dsc-base[1], nsd-base[2]
            if dd < -args.max_dsc_drop:
                failures.append(f'{arm}: DSC drop {dd:+.2f} exceeds {-args.max_dsc_drop:.2f}')
            if dd < args.min_dsc_gain:
                failures.append(f'{arm}: DSC gain {dd:+.2f} < +{args.min_dsc_gain:.2f}')
            if dn < args.min_nsd_gain:
                failures.append(f'{arm}: NSD gain {dn:+.2f} < +{args.min_nsd_gain:.2f}')
        full=rows[3]; edge=rows[1]; normal=rows[2]
        fdd, fdn = full[1]-base[1], full[2]-base[2]
        if fdd < max(args.min_dsc_gain, edge[1]-base[1]-0.05, normal[1]-base[1]-0.05):
            failures.append(f'FULL: DSC gain {fdd:+.2f} does not preserve single-module gains')
        if fdn < max(args.min_nsd_gain, edge[2]-base[2]-0.10, normal[2]-base[2]-0.10):
            failures.append(f'FULL: NSD gain {fdn:+.2f} does not preserve single-module gains')
        if failures:
            print('[JBTL3_VAL_GATE_FAIL]')
            for x in failures: print(' -',x)
            print('Do NOT open Test. Tune only on Validation or remove the failing module.')
            return 20
        print('[JBTL3_VAL_GATE_PASS] Both modules and FULL pass the validation-only causal gate. Test remains unopened.')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
