#!/usr/bin/env python3
"""Publication-grade collector for the locked SemLT A0-A6 ablation.

Key safeguards:
- verifies every variant used the same per-case Base predictions for a dataset/seed;
- reports mean±SD over seeds (never case-level SD as seed uncertainty);
- preregistered contrasts average each case over seeds before paired inference;
- applies Holm correction over the five planned contrasts within dataset/metric;
- gives BUSI/BTMRI empty/non-empty stratification when GT area is available.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from pathlib import Path
import numpy as np
import pandas as pd

DATASETS=['BUSI','BTMRI','ISIC','Kvasir']
VARIANTS=['A1_REWRITE','A2_PROB','A3_NORMAL1D','A4_FREE2D','A5_SUPPORT','A6_SEMLT']
METRICS=['DSC','NSD']

def ms(xs):
    a=np.asarray(xs,float); return float(a.mean()), float(a.std(ddof=1)) if len(a)>1 else 0.0

def pct_ms(x): return f"{100*x[0]:.2f}±{100*x[1]:.2f}"
def pct(x): return f"{100*float(x):.2f}"

def canonical_frame_hash(df):
    cols=[c for c in ['Case_ID','DSC','NSD','GT_Area_Fraction'] if c in df.columns]
    d=df[cols].copy().sort_values('Case_ID')
    payload=d.to_csv(index=False,float_format='%.12g').encode(); return hashlib.sha256(payload).hexdigest()

def paired_stats(a,b,rng,boot,perm):
    # delta = a - b; positive supports the preregistered "better" method.
    a=np.asarray(a,float); b=np.asarray(b,float); d=a-b; n=len(d)
    idx=rng.integers(0,n,size=(boot,n)); bm=d[idx].mean(1); lo,hi=np.quantile(bm,[.025,.975])
    observed=abs(float(d.mean())); extreme=0; remaining=perm; chunk=2048
    while remaining:
        c=min(chunk,remaining); signs=rng.choice(np.array([-1.,1.]),size=(c,n)); vals=np.abs((signs*d).mean(1)); extreme += int((vals>=observed-1e-15).sum()); remaining-=c
    p=(extreme+1)/(perm+1)
    return {'mean_delta':float(d.mean()),'median_delta':float(np.median(d)),'ci95':[float(lo),float(hi)],'p_raw':float(p),'benefit_rate':float((d>1e-12).mean()),'harm_gt_1pp_rate':float((d<-.01).mean()),'n':int(n)}

def holm(pvals):
    m=len(pvals); order=sorted(range(m),key=lambda i:pvals[i]); out=[1.0]*m; running=0.0
    for rank,i in enumerate(order):
        adj=min(1.0,(m-rank)*pvals[i]); running=max(running,adj); out[i]=running
    return out

def load_case(root,root_id,seed,ds,var,split,protocol,which):
    p=root/f'runs/SEMLT_ABLATION_{root_id}/seed{seed}/{ds}/variants/{var}/formal_{split}/{ds}/seg_results/seed{seed}/{split}_{which}Native_{protocol}.csv'
    if not p.is_file(): raise FileNotFoundError(p)
    df=pd.read_csv(p)
    if df['Case_ID'].duplicated().any(): raise ValueError(f'duplicate Case_ID: {p}')
    return p,df.set_index('Case_ID').sort_index()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project-root',default='.'); ap.add_argument('--root-id',required=True); ap.add_argument('--split',choices=['val','test'],default='test'); ap.add_argument('--protocol',choices=['true2d','paper_legacy'],default='true2d'); ap.add_argument('--seeds',default='42,123,789'); ap.add_argument('--bootstrap',type=int,default=20000); ap.add_argument('--permutations',type=int,default=100000); ap.add_argument('--output-dir',default=''); a=ap.parse_args()
    root=Path(a.project_root).resolve(); seeds=[int(x) for x in a.seeds.split(',') if x]; spec=json.loads((root/'configs/SEMLT_ABLATION_LOCKED_VARIANTS.json').read_text()); names={v:spec['variants'][v]['paper_name'] for v in VARIANTS}; contrasts=spec['preregistered_contrasts']
    outdir=Path(a.output_dir) if a.output_dir else root/f'logs/SEMLT_ABLATION_{a.root_id}_{a.split}_{a.protocol}'; outdir.mkdir(parents=True,exist_ok=True)
    # Load all case tables and verify common Base identity across variants.
    data={}; base_hashes={}
    for ds in DATASETS:
      for seed in seeds:
        reference_cases=None; hashes=[]
        for v in VARIANTS:
          _,b=load_case(root,a.root_id,seed,ds,v,a.split,a.protocol,'Base'); _,m=load_case(root,a.root_id,seed,ds,v,a.split,a.protocol,'M1')
          if set(b.index)!=set(m.index): raise SystemExit(f'[FAIL] Base/M1 case mismatch {ds} seed{seed} {v}')
          if reference_cases is None: reference_cases=list(b.index)
          elif reference_cases!=list(b.index): raise SystemExit(f'[FAIL] case ordering/content mismatch {ds} seed{seed} {v}')
          hashes.append(canonical_frame_hash(b)); data[(ds,seed,v,'base')]=b; data[(ds,seed,v,'m1')]=m
        if len(set(hashes))!=1: raise SystemExit(f'[FAIL] common Base prediction mismatch across variants: {ds} seed{seed}')
        base_hashes[(ds,seed)]=hashes[0]
    # Seed-level main table.
    seed_rows=[]
    for ds in DATASETS:
      for seed in seeds:
        b=data[(ds,seed,'A4_FREE2D','base')]
        row={'dataset':ds,'seed':seed,'method':'A0_BASE','paper_name':'Base'}
        for met in METRICS: row[met.lower()]=float(b[met].mean())
        seed_rows.append(row)
        for v in VARIANTS:
          m=data[(ds,seed,v,'m1')]; row={'dataset':ds,'seed':seed,'method':v,'paper_name':names[v]}
          for met in METRICS: row[met.lower()]=float(m[met].mean())
          seed_rows.append(row)
    pd.DataFrame(seed_rows).to_csv(outdir/'seed_level_results.csv',index=False)
    methods=['A0_BASE']+VARIANTS; labels={'A0_BASE':'Base',**names}; summary={}
    lines=[f'# SemLT locked ablation — {a.split} / {a.protocol}','',f'Seeds: {seeds}. Values are mean±SD across seeds. Dataset macro-average weights each dataset equally.','', '| Method | BUSI DSC/NSD | BTMRI DSC/NSD | ISIC DSC/NSD | Kvasir DSC/NSD | Macro DSC/NSD |','|---|---:|---:|---:|---:|---:|']
    for method in methods:
      summary[method]={}; cells=[]; macro_by_seed={s:{'DSC':[],'NSD':[]} for s in seeds}
      for ds in DATASETS:
        rr=[r for r in seed_rows if r['dataset']==ds and r['method']==method]; d=ms([r['dsc'] for r in rr]); n=ms([r['nsd'] for r in rr]); summary[method][ds]={'DSC':{'mean':d[0],'std':d[1]},'NSD':{'mean':n[0],'std':n[1]}}; cells.append(f"{pct_ms(d)} / {pct_ms(n)}")
        for r in rr: macro_by_seed[r['seed']]['DSC'].append(r['dsc']); macro_by_seed[r['seed']]['NSD'].append(r['nsd'])
      md=ms([np.mean(macro_by_seed[s]['DSC']) for s in seeds]); mn=ms([np.mean(macro_by_seed[s]['NSD']) for s in seeds]); summary[method]['MACRO']={'DSC':{'mean':md[0],'std':md[1]},'NSD':{'mean':mn[0],'std':mn[1]}}; cells.append(f"{pct_ms(md)} / {pct_ms(mn)}")
      bold='**' if method=='A6_SEMLT' else ''; lines.append(f"| {bold}{labels[method]}{bold} | "+' | '.join(cells)+' |')
    (outdir/'TABLE_MAIN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Preregistered direct contrasts using per-case seed-average scores.
    rng=np.random.default_rng(20260829); contrast_rows=[]
    for ds in DATASETS:
      for met in METRICS:
        local=[]
        for c in contrasts:
          better,ref=c['better'],c['reference']; case_ids=list(data[(ds,seeds[0],better,'m1')].index)
          ba=[]; rb=[]
          for seed in seeds:
            x=data[(ds,seed,better,'m1')]; y=data[(ds,seed,ref,'m1')]
            if list(x.index)!=case_ids or list(y.index)!=case_ids: raise SystemExit('[FAIL] case mismatch across seeds')
            ba.append(x[met].to_numpy(float)); rb.append(y[met].to_numpy(float))
          stats=paired_stats(np.mean(ba,0),np.mean(rb,0),rng,a.bootstrap,a.permutations)
          row={'dataset':ds,'metric':met,'contrast_id':c['id'],'better':better,'reference':ref,'claim':c['claim'],**stats}; local.append(row)
        adj=holm([r['p_raw'] for r in local])
        for r,padj in zip(local,adj): r['p_holm']=padj; r['direction_supported']=r['mean_delta']>0; r['significant_holm_005']=r['mean_delta']>0 and padj<.05; contrast_rows.append(r)
    with (outdir/'planned_contrasts.csv').open('w',newline='',encoding='utf-8') as f:
      w=csv.DictWriter(f,fieldnames=list(contrast_rows[0])); w.writeheader(); w.writerows(contrast_rows)
    clines=['# Preregistered causal contrasts','', 'Each case is averaged over seeds first; inference is paired across cases. Holm correction is applied to the five planned contrasts within each dataset/metric.','', '| Dataset | Metric | Contrast | Δ (better-reference) | 95% CI | p raw | p Holm | Direction | Holm<.05 |','|---|---|---|---:|---:|---:|---:|---|---|']
    for r in contrast_rows:
      clines.append(f"| {r['dataset']} | {r['metric']} | {r['contrast_id']} | {pct(r['mean_delta'])} | [{pct(r['ci95'][0])}, {pct(r['ci95'][1])}] | {r['p_raw']:.4g} | {r['p_holm']:.4g} | {'✓' if r['direction_supported'] else '✗'} | {'✓' if r['significant_holm_005'] else '✗'} |")
    (outdir/'TABLE_CONTRASTS.md').write_text('\n'.join(clines)+'\n',encoding='utf-8')
    # Cross-dataset direction consistency for each planned causal claim.
    consistency=[]
    for c in contrasts:
      for met in METRICS:
        rr=[r for r in contrast_rows if r['contrast_id']==c['id'] and r['metric']==met]; consistency.append({'contrast_id':c['id'],'metric':met,'positive_datasets':sum(r['direction_supported'] for r in rr),'significant_datasets_holm':sum(r['significant_holm_005'] for r in rr),'datasets':len(rr)})
    # Full SemLT vs Base mechanism/safety report, seed-averaged per case.
    full_vs_base=[]; strata=[]
    for ds in DATASETS:
      ids=list(data[(ds,seeds[0],'A6_SEMLT','m1')].index)
      for met in METRICS:
        full=np.mean([data[(ds,s,'A6_SEMLT','m1')][met].to_numpy(float) for s in seeds],0); base=np.mean([data[(ds,s,'A4_FREE2D','base')][met].to_numpy(float) for s in seeds],0); st=paired_stats(full,base,rng,a.bootstrap,a.permutations); full_vs_base.append({'dataset':ds,'metric':met,**st})
      b0=data[(ds,seeds[0],'A4_FREE2D','base')]
      if 'GT_Area_Fraction' in b0.columns:
        empty=b0['GT_Area_Fraction'].to_numpy(float)<=0
        for label,mask in [('empty_gt',empty),('nonempty_gt',~empty)]:
          if not mask.any(): continue
          for met in METRICS:
            full=np.mean([data[(ds,s,'A6_SEMLT','m1')][met].to_numpy(float) for s in seeds],0)[mask]; base=np.mean([data[(ds,s,'A4_FREE2D','base')][met].to_numpy(float) for s in seeds],0)[mask]; st=paired_stats(full,base,rng,min(a.bootstrap,10000),min(a.permutations,20000)); strata.append({'dataset':ds,'stratum':label,'metric':met,**st})
    payload={'protocol':spec['protocol'],'root_id':a.root_id,'split':a.split,'metric_protocol':a.protocol,'seeds':seeds,'base_hashes':{f'{d}_seed{s}':h for (d,s),h in base_hashes.items()},'summary':summary,'planned_contrasts':contrast_rows,'consistency':consistency,'full_vs_base':full_vs_base,'strata':strata}
    (outdir/'SUMMARY.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(outdir/'TABLE_MAIN.md'); print(outdir/'TABLE_CONTRASTS.md'); print(outdir/'SUMMARY.json')
if __name__=='__main__': main()
