#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import re
from pathlib import Path

PAT_DSC = re.compile(r"Average DSC .*? split=val:\s*([0-9.]+)%")
PAT_NSD = re.compile(r"Average NSD .*? split=val:\s*([0-9.]+)%")


def parse(path: Path):
    text = path.read_text(errors='ignore')
    d = PAT_DSC.findall(text)
    n = PAT_NSD.findall(text)
    if not d or not n:
        raise RuntimeError(f'Validation true2d metrics not found in {path}')
    return float(d[-1]), float(n[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='Study root containing BUSI/{BASE,EDGEISO,FULL}/seedXX')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--gate', action='store_true')
    ap.add_argument('--min-full-dsc-gain', type=float, default=0.30, help='percentage points')
    ap.add_argument('--min-full-nsd-gain', type=float, default=1.00, help='percentage points')
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    for arm in ('BASE', 'EDGEISO', 'FULL'):
        path = root / 'BUSI' / arm / f'seed{args.seed}' / 'logs' / 'eval_val_true2d.log'
        if not path.is_file():
            raise RuntimeError(f'missing: {path}')
        dsc, nsd = parse(path)
        rows.append((arm, dsc, nsd))

    base = rows[0]
    print('ARM\tDSC\tNSD\tΔDSC\tΔNSD')
    for arm, dsc, nsd in rows:
        print(f'{arm}\t{dsc:.2f}\t{nsd:.2f}\t{dsc-base[1]:+.2f}\t{nsd-base[2]:+.2f}')

    out = root / 'BUSI' / 'JBTL5_R50_DIAG40_summary.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['arm','dsc','nsd','delta_dsc','delta_nsd'])
        for arm, dsc, nsd in rows:
            w.writerow([arm, f'{dsc:.4f}', f'{nsd:.4f}', f'{dsc-base[1]:.4f}', f'{nsd-base[2]:.4f}'])
    print(f'summary={out}')

    if args.gate:
        full = rows[2]
        dd = full[1] - base[1]
        dn = full[2] - base[2]
        if dd >= args.min_full_dsc_gain and dn >= args.min_full_nsd_gain:
            print('[JBTL5_R50_VAL_GATE_PASS] FULL clears the predeclared Validation effect-size gate. Test remains unopened.')
            return 0
        print('[JBTL5_R50_VAL_GATE_FAIL] Do NOT open Test.')
        print(f'  FULL ΔDSC={dd:+.2f} pp; required >= +{args.min_full_dsc_gain:.2f} pp')
        print(f'  FULL ΔNSD={dn:+.2f} pp; required >= +{args.min_full_nsd_gain:.2f} pp')
        return 20
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
