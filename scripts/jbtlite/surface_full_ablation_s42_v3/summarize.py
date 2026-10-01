#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,re
from pathlib import Path
METRIC=re.compile(r"Average (DSC|NSD).*?:\s*([0-9.]+)%")

def metrics(log):
    p=Path(log)
    if not p.is_file(): return None
    vals={k:float(v) for k,v in METRIC.findall(p.read_text(errors='replace'))}
    return vals if 'DSC' in vals and 'NSD' in vals else None

def triple(root,tag):
    a=metrics(root/'logs'/f'eval_{tag}_true2d.log'); b=metrics(root/'logs'/f'eval_{tag}_paper_legacy.log')
    if not a or not b: return None
    return round(a['DSC'],2),round(a['NSD'],2),round(b['NSD'],2)

def row(root,arm,tag,label,**extra):
    t=triple(root/'BUSI'/arm/'seed42',tag)
    d={'setting':label,'arm':arm,'DSC': '' if not t else f'{t[0]:.2f}','true2D_NSD':'' if not t else f'{t[1]:.2f}','Legacy_NSD':'' if not t else f'{t[2]:.2f}'}; d.update(extra); return d

def write(path,rows,fields=None):
    path.parent.mkdir(parents=True,exist_ok=True); fields=fields or list(rows[0])
    with path.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def md_table(rows, cols):
    out=['| '+' | '.join(h for h,_ in cols)+' |','|'+ '|'.join(['---']*len(cols))+'|']
    for r in rows: out.append('| '+' | '.join(str(r.get(k,'')) for _,k in cols)+' |')
    return '\n'.join(out)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project',required=True); ap.add_argument('--result-root',default=None); ap.add_argument('--study-id',required=True); a=ap.parse_args()
    project=Path(a.project); rr=Path(a.result_root) if a.result_root else project/'surface_full_ablation_s42_v3_results'; root=rr/a.study_id; out=root/'tables'
    comp=[
      row(root,'BASE','test_mc30','Base',Surface='No',FiniteHorizon='No',FullPath='--'),
      row(root,'SURFACE_ALWAYS','test_mc30','+ Surface loss (always-on)',Surface='Yes',FiniteHorizon='No',FullPath='Yes'),
      row(root,'DECODER_ONLY_STEP_E30','test_mc30','+ Surface loss + Step-e30, decoder-only',Surface='Yes',FiniteHorizon='Yes',FullPath='No'),
      row(root,'FULL_STEP_E30','test_mc30','Full SurfaceAlign (Step-e30, full-path)',Surface='Yes',FiniteHorizon='Yes',FullPath='Yes')]
    sched=[row(root,'BASE','test_mc30','No auxiliary term'),row(root,'SURFACE_ALWAYS','test_mc30','Always-on'),row(root,'SCHED_COSINE','test_mc30','Cosine hold10→0@30'),row(root,'SCHED_WARMUP_E30','test_mc30','Warm-up→30'),row(root,'FULL_STEP_E30','test_mc30','Step-e30')]
    obj=[row(root,'LOSS_BOUNDARY_STEP_E30','test_mc30','Boundary Loss'),row(root,'LOSS_HD_STEP_E30','test_mc30','HD-DT Loss'),row(root,'LOSS_ACTIVE_CONTOUR_STEP_E30','test_mc30','Active Contour Loss'),row(root,'FULL_STEP_E30','test_mc30','SurfaceAlign')]
    rad=[row(root,'FULL_STEP_E30','val_mc10','r=1'),row(root,'RADIUS_R2_STEP_E30','val_mc10','r=2'),row(root,'RADIUS_R3_STEP_E30','val_mc10','r=3'),row(root,'RADIUS_R5_STEP_E30','val_mc10','r=5')]
    wei=[row(root,'BASE','val_mc10','lambda0=0'),row(root,'FULL_STEP_E30','val_mc10','lambda0=0.05'),row(root,'WEIGHT_L010_STEP_E30','val_mc10','lambda0=0.10'),row(root,'WEIGHT_L015_STEP_E30','val_mc10','lambda0=0.15'),row(root,'WEIGHT_L020_STEP_E30','val_mc10','lambda0=0.20'),row(root,'WEIGHT_L025_STEP_E30','val_mc10','lambda0=0.25')]
    write(out/'table_component_test.csv',comp); write(out/'table_schedule_test.csv',sched); write(out/'table_aux_objective_test.csv',obj); write(out/'table_radius_val.csv',rad); write(out/'table_weight_val.csv',wei)
    allrows=[]
    for group, rows in [('component',comp),('schedule',sched),('objective',obj),('radius',rad),('weight',wei)]:
        for r in rows: allrows.append({'group':group,**r})
    write(out/'table_all_results.csv',allrows)
    md=['# SurfaceAlign seed-42 unified ablation — FULL_STEP_E30 anchor','',f'Study: `{a.study_id}`','',
        '## A. Core contribution ablation (Test MC30)',md_table(comp,[('Setting','setting'),('Surface','Surface'),('Finite horizon','FiniteHorizon'),('Full path','FullPath'),('DSC','DSC'),('true-2D NSD','true2D_NSD'),('Legacy NSD','Legacy_NSD')]),'',
        '## B. Schedule analysis (Test MC30)',md_table(sched,[('Setting','setting'),('DSC','DSC'),('true-2D NSD','true2D_NSD'),('Legacy NSD','Legacy_NSD')]),'',
        '## C. Auxiliary-objective control (Test MC30)',md_table(obj,[('Setting','setting'),('DSC','DSC'),('true-2D NSD','true2D_NSD'),('Legacy NSD','Legacy_NSD')]),'',
        '## D. Radius sensitivity (Val MC10 only)',md_table(rad,[('Setting','setting'),('DSC','DSC'),('true-2D NSD','true2D_NSD'),('Legacy NSD','Legacy_NSD')]),'',
        '## E. Weight sensitivity (Val MC10 only)',md_table(wei,[('Setting','setting'),('DSC','DSC'),('true-2D NSD','true2D_NSD'),('Legacy NSD','Legacy_NSD')]),'']
    (out/'RESULTS.md').write_text('\n'.join(md),encoding='utf-8'); print('\n'.join(md)); print(f'\nWritten: {out}')
if __name__=='__main__': main()
