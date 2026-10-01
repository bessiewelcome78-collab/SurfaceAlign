#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,math,random
from pathlib import Path
TEST=['BASE','SURFACE_ALWAYS','DECODER_ONLY_STEP_E30','SCHED_COSINE','SCHED_WARMUP_E30','LOSS_BOUNDARY_STEP_E30','LOSS_HD_STEP_E30','LOSS_ACTIVE_CONTOUR_STEP_E30']

def find_csv(rr,arm):
    stem=f'{arm.lower()}_test_mc30_true2d.csv'
    hits=list((rr/'test').rglob(stem)) if (rr/'test').exists() else []
    return hits[0] if hits else None

def read(p):
    out={}
    if not p:return out
    with p.open(newline='',encoding='utf-8') as f:
      for r in csv.DictReader(f):
        k=(r.get('Case_ID') or r.get('Name') or '').strip()
        if k:
          try: out[k]=(float(r['DSC']),float(r['NSD']))
          except: pass
    return out

def ci(ds,n=10000):
    rng=random.Random(20260921); N=len(ds); vals=[sum(ds[rng.randrange(N)] for _ in range(N))/N for __ in range(n)]; vals.sort(); return vals[int(.025*n)],vals[min(n-1,int(.975*n))]

def pval(ds):
    try:
      from scipy.stats import wilcoxon
      nz=[x for x in ds if x!=0]; return float(wilcoxon(nz).pvalue) if nz else float('nan')
    except: return float('nan')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',required=True);ap.add_argument('--result-root',default=None);ap.add_argument('--study-id',required=True);a=ap.parse_args(); p=Path(a.project); root=(Path(a.result_root) if a.result_root else p/'surface_full_ablation_s42_v3_results')/a.study_id/'BUSI'; anchor=read(find_csv(root/'FULL_STEP_E30'/'seed42','FULL_STEP_E30')); rows=[]
    if not anchor: raise SystemExit('[FAIL] FULL_STEP_E30 per-case CSV missing')
    for arm in TEST:
      dat=read(find_csv(root/arm/'seed42',arm)); shared=sorted(set(dat)&set(anchor))
      for metric,ix in [('DSC',0),('NSD',1)]:
        ds=[dat[k][ix]-anchor[k][ix] for k in shared]
        if not ds: continue
        lo,hi=ci(ds); pv=pval(ds); rows.append({'arm':arm,'metric':metric,'n':len(ds),'mean_diff_pp':f'{100*sum(ds)/len(ds):+.2f}','ci95_low_pp':f'{100*lo:+.2f}','ci95_high_pp':f'{100*hi:+.2f}','wilcoxon_p':f'{pv:.4g}' if pv==pv else 'NA','better':sum(x>0 for x in ds),'worse':sum(x<0 for x in ds),'tie':sum(x==0 for x in ds)})
    out=root.parent/'tables'/'paired_stats_vs_FULL_STEP_E30.csv'; out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(f'Written: {out}')
if __name__=='__main__':main()
