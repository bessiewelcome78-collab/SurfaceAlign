#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,re
from pathlib import Path
ORDER=[
 ('BUSI','BUSBRA'),('BUSI','BUSUC'),('BUSI','BUID'),('BUSI','UDIAT'),
 ('Kvasir','ColonDB'),('Kvasir','ClinicDB'),('Kvasir','CVC300'),('Kvasir','BKAI'),
 ('BTMRI','BRISC'),('ISIC','UWaterlooSkinCancer')]
DISPLAY={'Kvasir':'Kvasir-SEG','UWaterlooSkinCancer':'UWaterloo'}

def metric(p:Path,key:str):
    if not p.is_file(): return None
    m=re.findall(rf'Average {key} .*?:\s*([0-9.]+)%',p.read_text(errors='ignore'))
    return float(m[-1]) if m else None

def cases(p:Path):
    if not p.is_file(): return None
    m=re.findall(r'Cases evaluated:\s*([0-9]+)',p.read_text(errors='ignore'))
    return int(m[-1]) if m else None

def fmt(x): return 'MISSING' if x is None else f'{x:.2f}'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project',required=True); ap.add_argument('--dg-id',required=True); ap.add_argument('--seed',type=int,default=42); a=ap.parse_args()
    root=Path(a.project)/'domain_generalization_base_s42'/a.dg_id
    resolved=json.loads((root/'resolved.json').read_text())
    rows=[]
    for s,t in ORDER:
        d=root/f'{s}_to_{t}'/f'seed{a.seed}'/'logs'
        tr=d/'eval_test_mc30_true2d.log'; lg=d/'eval_test_mc30_paper_legacy.log'
        rows.append({'source':s,'target':t,'dsc':metric(tr,'DSC'),'true2d_nsd':metric(tr,'NSD'),'legacy_nsd':metric(lg,'NSD'),'cases':cases(tr)})
    print(f"{'SOURCE':<10} {'TARGET':<22} {'DSC':>8} {'TRUE2D_NSD':>12} {'LEGACY_NSD':>12} {'N':>6}")
    print('-'*76)
    for r in rows:
        print(f"{r['source']:<10} {r['target']:<22} {fmt(r['dsc']):>8} {fmt(r['true2d_nsd']):>12} {fmt(r['legacy_nsd']):>12} {str(r['cases'] or '-'):>6}")
    csvp=root/'BASE_TABLE2_DG_RESULTS.csv'
    with csvp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['source','target','dsc','true2d_nsd','legacy_nsd','cases']); w.writeheader(); w.writerows(rows)
    # Table-2-style row: source ID value, then all OOD targets.
    idm={s:resolved['sources'][s]['expected_id_metrics']['dsc'] for s in resolved['sources']}
    lookup={(r['source'],r['target']):r['dsc'] for r in rows}
    table2=[('BUSI',None),('BUSI','BUSBRA'),('BUSI','BUSUC'),('BUSI','BUID'),('BUSI','UDIAT'),
            ('Kvasir',None),('Kvasir','ColonDB'),('Kvasir','ClinicDB'),('Kvasir','CVC300'),('Kvasir','BKAI'),
            ('BTMRI',None),('BTMRI','BRISC'),('ISIC',None),('ISIC','UWaterlooSkinCancer')]
    vals=[]
    for s,t in table2: vals.append(idm[s] if t is None else lookup.get((s,t)))
    latex='Matched Base (seed 42) & ' + ' & '.join('MISSING' if v is None else f'{v:.2f}' for v in vals) + r' \\'
    (root/'TABLE2_LATEX_ROW.txt').write_text(latex+'\n')
    md=['# BASE Table-2 Domain Generalization','',f'DG_ID: `{a.dg_id}`','',
        '| Source | Target | DSC | true2D NSD | paper-legacy NSD | N |','|---|---:|---:|---:|---:|---:|']
    for r in rows: md.append(f"| {DISPLAY.get(r['source'],r['source'])} | {DISPLAY.get(r['target'],r['target'])} | {fmt(r['dsc'])} | {fmt(r['true2d_nsd'])} | {fmt(r['legacy_nsd'])} | {r['cases'] or '-'} |")
    md += ['', 'Table-2-style DSC row:', '', '```latex', latex, '```']
    (root/'RESULTS.md').write_text('\n'.join(md)+'\n')
    print(f'\nCSV: {csvp}')
    print(f'LaTeX: {root/"TABLE2_LATEX_ROW.txt"}')
    print(f'Markdown: {root/"RESULTS.md"}')
    print('\n'+latex)
if __name__=='__main__':main()
