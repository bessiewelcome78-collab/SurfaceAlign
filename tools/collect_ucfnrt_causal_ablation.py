#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
VARIANTS=('NO_POSTERIOR','NO_RAY','DIRECT_SIGNED','SEG_ONLY')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project',required=True); ap.add_argument('--root-id',required=True); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--datasets',nargs='+',required=True); a=ap.parse_args()
    root=Path(a.project)/'runs'/f'UCFNRT_CAUSAL_ABL_{a.root_id}'/f'seed{a.seed}'
    rows=[]
    for ds in a.datasets:
        for var in VARIANTS:
            rr=root/ds/var/'formal_test'/ds/'seg_results'/f'seed{a.seed}'
            reports={}
            for mode in ('true2d','paper_legacy'):
                p=rr/f'paired_{mode}.json'
                if not p.is_file(): continue
                reports[mode]=json.loads(p.read_text())
            if not reports: continue
            t=reports.get('true2d',{}).get('metrics',{})
            l=reports.get('paper_legacy',{}).get('metrics',{})
            rows.append({
                'Dataset':ds,'Variant':var,
                'Base_DSC':t.get('DSC',{}).get('base_mean'),
                'M1_DSC':t.get('DSC',{}).get('m1_mean'),
                'Delta_DSC':t.get('DSC',{}).get('mean_delta'),
                'DSC_CI_low':(t.get('DSC',{}).get('bootstrap_95_ci') or [None,None])[0],
                'DSC_CI_high':(t.get('DSC',{}).get('bootstrap_95_ci') or [None,None])[1],
                'DSC_p':t.get('DSC',{}).get('paired_sign_flip_p_two_sided'),
                'Base_NSD_true2d':t.get('NSD',{}).get('base_mean'),
                'M1_NSD_true2d':t.get('NSD',{}).get('m1_mean'),
                'Delta_NSD_true2d':t.get('NSD',{}).get('mean_delta'),
                'NSD_CI_low':(t.get('NSD',{}).get('bootstrap_95_ci') or [None,None])[0],
                'NSD_CI_high':(t.get('NSD',{}).get('bootstrap_95_ci') or [None,None])[1],
                'NSD_p':t.get('NSD',{}).get('paired_sign_flip_p_two_sided'),
                'Benefit_cases_DSC':t.get('DSC',{}).get('benefit_cases'),
                'Harm_cases_DSC':t.get('DSC',{}).get('harm_cases'),
                'Delta_NSD_paper_legacy':l.get('NSD',{}).get('mean_delta'),
            })
    df=pd.DataFrame(rows)
    out=root/'UC_FNRT_CAUSAL_ABLATION_SUMMARY.csv'; out.parent.mkdir(parents=True,exist_ok=True); df.to_csv(out,index=False)
    print(out)
    if not df.empty:
        print(df.to_string(index=False))
if __name__=='__main__': main()
