#!/usr/bin/env python3
"""Collect LeST59 paired true2d results into publication-friendly CSV/Markdown."""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path

DATASETS=['BUSI','BTMRI','ISIC','Kvasir']
METRICS=['DSC','NSD']

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project-root', default='.')
    ap.add_argument('--root-id', required=True, help='ROOT_ID used by launch_lest59_all4_3seeds.sh')
    ap.add_argument('--split', choices=['val','test'], default='test')
    ap.add_argument('--seeds', default='42,123,789')
    ap.add_argument('--output-prefix', default='')
    a=ap.parse_args(); root=Path(a.project_root).resolve(); seeds=[int(x) for x in a.seeds.split(',')]
    rows=[]; missing=[]
    for ds in DATASETS:
        for seed in seeds:
            exp=f'{a.root_id}_seed{seed}'
            p=root/f'runs/LEST59_ALL4_{exp}/{ds}/seed{seed}/unified_aniso/formal_{a.split}/{ds}/seg_results/seed{seed}/paired_true2d.json'
            if not p.is_file(): missing.append(str(p)); continue
            r=json.loads(p.read_text())
            row={'dataset':ds,'seed':seed}
            for metric in METRICS:
                x=r['metrics'][metric]
                row[f'base_{metric.lower()}']=x['base_mean']; row[f'm1_{metric.lower()}']=x['m1_mean']; row[f'delta_{metric.lower()}']=x['mean_delta']; row[f'p_{metric.lower()}']=x['paired_sign_flip_p_two_sided']
            rows.append(row)
    if missing:
        raise SystemExit('[FAIL] missing reports:\n'+'\n'.join(missing))
    prefix=Path(a.output_prefix) if a.output_prefix else root/f'logs/LEST59_{a.root_id}_{a.split}_summary'
    prefix.parent.mkdir(parents=True,exist_ok=True)
    import csv
    with prefix.with_suffix('.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=[f'# LeST59 {a.split} 3-seed summary','', '| Dataset | DSC Base | DSC LeST59 | ΔDSC | NSD Base | NSD LeST59 | ΔNSD |','|---|---:|---:|---:|---:|---:|---:|']
    summary={}
    for ds in DATASETS:
        rr=[r for r in rows if r['dataset']==ds]; summary[ds]={}
        vals={}
        for key in ['base_dsc','m1_dsc','delta_dsc','base_nsd','m1_nsd','delta_nsd']:
            xs=[r[key] for r in rr]; vals[key]=(statistics.mean(xs),statistics.stdev(xs) if len(xs)>1 else 0.0); summary[ds][key]={'mean':vals[key][0],'std':vals[key][1]}
        fmt=lambda k:f"{100*vals[k][0]:.2f}±{100*vals[k][1]:.2f}"
        lines.append(f"| {ds} | {fmt('base_dsc')} | {fmt('m1_dsc')} | {fmt('delta_dsc')} | {fmt('base_nsd')} | {fmt('m1_nsd')} | {fmt('delta_nsd')} |")
    prefix.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    prefix.with_suffix('.json').write_text(json.dumps({'root_id':a.root_id,'split':a.split,'seeds':seeds,'summary':summary,'rows':rows},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(prefix.with_suffix('.md'))

if __name__=='__main__': main()
