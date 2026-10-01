#!/usr/bin/env python3
import argparse, pathlib, re, csv, statistics
p=argparse.ArgumentParser(); p.add_argument('--project',required=True); p.add_argument('--root-id',required=True); p.add_argument('--seeds',default='42 123 789'); a=p.parse_args()
project=pathlib.Path(a.project); seeds=[int(x) for x in a.seeds.split()]; arms=['BASE','SURFACE','QABR','FULL']
pd=re.compile(r'Average DSC .*?:\s*([0-9.]+)%'); pn=re.compile(r'Average NSD .*?:\s*([0-9.]+)%')
rows=[]
for seed in seeds:
    sid=f'{a.root_id}_S{seed}'
    for arm in arms:
        log=project/f'formal_results_jbtlite_v15/{sid}/BUSI/{arm}/seed{seed}/logs/eval_val_mc10_true2d.log'
        if not log.exists(): continue
        t=log.read_text(errors='ignore'); d=pd.findall(t); n=pn.findall(t)
        if d and n: rows.append({'seed':seed,'arm':arm,'dice':float(d[-1]),'nsd':float(n[-1]),'study_id':sid})
outdir=project/f'formal_results_jbtlite_v15/{a.root_id}_MULTISEED_SUMMARY'; outdir.mkdir(parents=True,exist_ok=True)
with (outdir/'per_seed.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['seed','arm','dice','nsd','study_id']); w.writeheader(); w.writerows(rows)
print('ARM      Dice(mean±sd)   NSD(mean±sd)')
agg=[]
for arm in arms:
    rr=[r for r in rows if r['arm']==arm]
    if not rr: continue
    ds=[r['dice'] for r in rr]; ns=[r['nsd'] for r in rr]
    md=statistics.mean(ds); mn=statistics.mean(ns); sd=statistics.stdev(ds) if len(ds)>1 else 0.; sn=statistics.stdev(ns) if len(ns)>1 else 0.
    agg.append({'arm':arm,'dice_mean':md,'dice_sd':sd,'nsd_mean':mn,'nsd_sd':sn,'n':len(rr)})
    print(f'{arm:<8} {md:.3f}±{sd:.3f}   {mn:.3f}±{sn:.3f}')
with (outdir/'aggregate.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['arm','dice_mean','dice_sd','nsd_mean','nsd_sd','n']); w.writeheader(); w.writerows(agg)
A={x['arm']:x for x in agg}
if all(k in A for k in arms):
    print(f"FULL-BASE    Dice {A['FULL']['dice_mean']-A['BASE']['dice_mean']:+.3f} pp | NSD {A['FULL']['nsd_mean']-A['BASE']['nsd_mean']:+.3f} pp")
    print(f"FULL-SURFACE Dice {A['FULL']['dice_mean']-A['SURFACE']['dice_mean']:+.3f} pp | NSD {A['FULL']['nsd_mean']-A['SURFACE']['nsd_mean']:+.3f} pp")
    print(f"QABR-BASE    Dice {A['QABR']['dice_mean']-A['BASE']['dice_mean']:+.3f} pp | NSD {A['QABR']['nsd_mean']-A['BASE']['nsd_mean']:+.3f} pp")
print('SUMMARY_DIR=',outdir)
