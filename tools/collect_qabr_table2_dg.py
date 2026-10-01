#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,re
from pathlib import Path
ORDER=[('BUSI','BUSBRA'),('BUSI','BUSUC'),('BUSI','BUID'),('BUSI','UDIAT'),('Kvasir','ColonDB'),('Kvasir','ClinicDB'),('Kvasir','CVC300'),('Kvasir','BKAI'),('BTMRI','BRISC'),('ISIC','UWaterlooSkinCancer')]

def metric(p:Path,key:str):
    if not p.is_file(): return None
    txt=p.read_text(errors='ignore')
    m=re.findall(rf'Average {key} .*?:\s*([0-9.]+)%',txt)
    return float(m[-1]) if m else None

def cases(p:Path):
    if not p.is_file(): return None
    m=re.findall(r'Cases evaluated:\s*([0-9]+)',p.read_text(errors='ignore'))
    return int(m[-1]) if m else None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project',required=True); ap.add_argument('--dg-id',required=True); ap.add_argument('--seed',type=int,default=42); a=ap.parse_args()
    root=Path(a.project)/'domain_generalization_qabr_v15'/a.dg_id
    rows=[]
    for s,t in ORDER:
        d=root/f'{s}_to_{t}'/f'seed{a.seed}'/'logs'
        tr=d/'eval_test_mc30_true2d.log'; lg=d/'eval_test_mc30_paper_legacy.log'
        rows.append((s,t,metric(tr,'DSC'),metric(tr,'NSD'),metric(lg,'NSD'),cases(tr)))
    print(f"{'SOURCE':<10} {'TARGET':<22} {'DSC':>8} {'TRUE2D_NSD':>12} {'LEGACY_NSD':>12} {'N':>6}")
    print('-'*76)
    for r in rows:
        s,t,d,n,l,c=r
        fmt=lambda x:'MISSING' if x is None else f'{x:.2f}'
        print(f'{s:<10} {t:<22} {fmt(d):>8} {fmt(n):>12} {fmt(l):>12} {str(c or "-"):>6}')
    csv=root/'QABR_TABLE2_DG_RESULTS.csv'
    csv.write_text('source,target,dsc,true2d_nsd,legacy_nsd,cases\n'+'\n'.join(f'{s},{t},{"" if d is None else d},{"" if n is None else n},{"" if l is None else l},{"" if c is None else c}' for s,t,d,n,l,c in rows)+'\n')
    print(f'\nCSV: {csv}')
if __name__=='__main__':main()
