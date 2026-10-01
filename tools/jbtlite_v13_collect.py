#!/usr/bin/env python3
from __future__ import annotations
import argparse,re
from pathlib import Path

p=argparse.ArgumentParser(); p.add_argument('--study-root',required=True); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
root=Path(a.study_root)
arms=['BASE','SURFACE','QABR','FULL']
rows={}
for arm in arms:
    rr=root/'BUSI'/arm/f'seed{a.seed}'
    elog=rr/'logs'/'eval_val_mc10_true2d.log'
    tlog=rr/'logs'/'train.log'
    if not elog.exists():
        print(f'[MISSING] {arm}: {elog}'); continue
    txt=elog.read_text(errors='ignore')
    md=re.findall(r'Average DSC .*?:\s*([0-9.]+)%',txt)
    mn=re.findall(r'Average NSD .*?:\s*([0-9.]+)%',txt)
    best=None; bestv=None
    if tlog.exists():
        tt=tlog.read_text(errors='ignore')
        m=re.findall(r'best_val_epoch=(\d+) \| best_selection_value=([0-9.]+)',tt)
        if m: best,bestv=m[-1]
    if md and mn:
        rows[arm]=(float(md[-1]),float(mn[-1]),best,bestv)
for arm in arms:
    if arm in rows:
        d,n,e,v=rows[arm]; print(f'{arm:8s} DSC={d:6.2f} NSD={n:6.2f} best_epoch={e or "?"} native_dice={v or "?"}')
if all(x in rows for x in arms):
    b,s,q,f=(rows[x] for x in arms)
    idsc=f[0]-s[0]-q[0]+b[0]; insd=f[1]-s[1]-q[1]+b[1]
    print(f'INTERACTION DSC={idsc:+.2f}pp NSD={insd:+.2f}pp')
    print(f'FULL-vs-BASE    DSC={f[0]-b[0]:+.2f}pp NSD={f[1]-b[1]:+.2f}pp')
    print(f'FULL-vs-SURFACE DSC={f[0]-s[0]:+.2f}pp NSD={f[1]-s[1]:+.2f}pp')
    print(f'FULL-vs-QABR    DSC={f[0]-q[0]:+.2f}pp NSD={f[1]-q[1]:+.2f}pp')
    if f[0] > max(s[0],q[0]) and f[1] > max(s[1],q[1]):
        print('[JBTL13_PARETO_PASS] FULL beats both single modules on both external Val metrics.')
    else:
        print('[JBTL13_PARETO_FAIL] Do not open Test; inspect cooperative diagnostics.')
