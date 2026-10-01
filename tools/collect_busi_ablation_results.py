#!/usr/bin/env python3
from pathlib import Path
import csv, argparse

def readmean(p, names):
    if p is None or not p.exists():
        return None
    with p.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    k = next((x for x in names if x in rows[0]), None)
    if not k:
        return None
    vals = []
    for r in rows:
        try:
            vals.append(float(r[k]))
        except Exception:
            pass
    return sum(vals) / len(vals) if vals else None

def latest(d, name):
    if not d.exists():
        return None
    hits = [p for p in d.rglob(name) if 'archive_' not in str(p)]
    return max(hits, key=lambda p: p.stat().st_mtime) if hits else None

ap = argparse.ArgumentParser()
ap.add_argument('--root', default='/home/tsz-25/MedCLIPSeg-pristine/runs/UCFNRT_CAUSAL_ABL_20260831_132730/seed42/BUSI')
a = ap.parse_args()
root = Path(a.root)
rows = []

for v in ('NO_POSTERIOR','NO_RAY','DIRECT_SIGNED','SEG_ONLY'):
    d = root / v
    bt = latest(d, 'test_BaseNative_true2d.csv')
    mt = latest(d, 'test_M1Native_true2d.csv')
    ml = latest(d, 'test_M1Native_paper_legacy.csv')

    bd = readmean(bt, ['DSC','Dice','dice','dsc'])
    md = readmean(mt, ['DSC','Dice','dice','dsc'])
    bn = readmean(bt, ['NSD','nsd'])
    mn = readmean(mt, ['NSD','nsd'])
    mln = readmean(ml, ['NSD','nsd'])

    rows.append([
        v,
        'COMPLETE' if md is not None else 'PENDING',
        '' if bd is None else f'{100*bd:.4f}',
        '' if md is None else f'{100*md:.4f}',
        '' if bn is None else f'{100*bn:.4f}',
        '' if mn is None else f'{100*mn:.4f}',
        '' if mln is None else f'{100*mln:.4f}',
        '' if bd is None or md is None else f'{100*(md-bd):+.4f}',
        '' if bn is None or mn is None else f'{100*(mn-bn):+.4f}',
        str(mt) if mt else '',
    ])

out = root / 'ABLATION_CURRENT_RESULTS.csv'
with out.open('w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow([
        'Variant','Status','Base_DSC_pct','M1_DSC_pct',
        'Base_true2d_NSD_pct','M1_true2d_NSD_pct','M1_legacy_NSD_pct',
        'Delta_DSC_pp','Delta_true2d_NSD_pp','Source_CSV'
    ])
    w.writerows(rows)

print(out)
for r in rows:
    print('\t'.join(r))
