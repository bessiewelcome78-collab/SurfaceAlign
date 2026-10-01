#!/usr/bin/env python3
"""Collect completed four-dataset E2E100 one-shot tests into Table-1 rows."""
from __future__ import annotations
import argparse, json, csv
from pathlib import Path

ORDER=('BUSI','BTMRI','ISIC','Kvasir')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-root',required=True); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--output-prefix',default='')
    a=ap.parse_args(); root=Path(a.run_root).resolve()
    rows=[]
    for ds in ORDER:
        man=root/ds/f'seed{a.seed}'/'SEMLT_E2E100_manifest.json'
        if not man.is_file(): raise SystemExit(f'[FAIL] missing manifest {man}')
        m=json.loads(man.read_text())
        if not m.get('test_success'): raise SystemExit(f'[FAIL] Test incomplete: {ds}')
        report=Path(m['table1_report']); r=json.loads(report.read_text())
        d=r['metrics']['DSC']; n=r['metrics']['NSD']
        rows.append({'dataset':ds,'base_dsc':d['base_mean'],'semlt_dsc':d['m1_mean'],'delta_dsc':d['mean_delta'],
                     'base_nsd':n['base_mean'],'semlt_nsd':n['m1_mean'],'delta_nsd':n['mean_delta'],
                     'dsc_p':d['paired_sign_flip_p_two_sided'],'nsd_p':n['paired_sign_flip_p_two_sided']})
    avg={k:sum(x[k] for x in rows)/len(rows) for k in ('base_dsc','semlt_dsc','delta_dsc','base_nsd','semlt_nsd','delta_nsd')}
    prefix=Path(a.output_prefix) if a.output_prefix else root/f'TABLE1_seed{a.seed}'
    prefix.parent.mkdir(parents=True,exist_ok=True)
    with prefix.with_suffix('.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    def pc(x): return f'{100*x:.2f}'
    lines=['# SemLT E2E100 Table-1 collector','',f'Seed: {a.seed}','',
           '| Method | BUSI DSC/NSD | BTMRI DSC/NSD | ISIC DSC/NSD | Kvasir DSC/NSD | Avg DSC/NSD |',
           '|---|---:|---:|---:|---:|---:|']
    lines.append('| Joint-run Base | '+' | '.join(f"{pc(x['base_dsc'])}/{pc(x['base_nsd'])}" for x in rows)+f" | {pc(avg['base_dsc'])}/{pc(avg['base_nsd'])} |")
    lines.append('| **SemLT-E2E100 (Ours)** | '+' | '.join(f"**{pc(x['semlt_dsc'])}/{pc(x['semlt_nsd'])}**" for x in rows)+f" | **{pc(avg['semlt_dsc'])}/{pc(avg['semlt_nsd'])}** |")
    lines.append('| Δ over joint-run Base | '+' | '.join(f"{pc(x['delta_dsc'])}/{pc(x['delta_nsd'])}" for x in rows)+f" | {pc(avg['delta_dsc'])}/{pc(avg['delta_nsd'])} |")
    lines.extend(['','> Table-1 NSD uses the paper-compatible `paper_legacy` evaluator. Use `paired_true2d` only for the strict 2-D boundary analysis table.',''])
    prefix.with_suffix('.md').write_text('\n'.join(lines),encoding='utf-8')
    print(prefix.with_suffix('.md')); print(prefix.with_suffix('.csv'))

if __name__=='__main__': main()
