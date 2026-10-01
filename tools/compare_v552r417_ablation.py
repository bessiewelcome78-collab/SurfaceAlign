#!/usr/bin/env python3
from __future__ import annotations
import argparse,math,re
from pathlib import Path
PAIR=re.compile(r'([A-Za-z0-9_]+)=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')
VAL=re.compile(r'VAL_NATIVE epoch=(\d+).*?base DSC/NSD=([0-9.]+)/([0-9.]+).*?M2 DSC/NSD=([0-9.]+)/([0-9.]+).*?componentOracle DSC/NSD=([0-9.]+)/([0-9.]+)')
def parse(p):
 lines=p.read_text(errors='replace').splitlines(); rows=[{k:float(v) for k,v in PAIR.findall(x)} for x in lines if 'M1_DIAG:' in x]; vals=[tuple(map(float,m.groups())) for x in lines if (m:=VAL.search(x))]; r=rows[-1] if rows else {}
 teacher=r.get('v538_action_realizable_teacher_oracle_gain',math.nan); student=r.get('v538_component_oracle_gain',math.nan)
 return dict(loc=r.get('v552r417_location_center_recall',math.nan),peak=r.get('v552r417_pre_topk_peak_recall',math.nan),sp=r.get('v552r417_selected_spatial_coverage',math.nan),id=r.get('v552r415_identity_match_rate',math.nan),box=r.get('v552r413_proposal_box_iou',math.nan),student=student,retain=student/teacher if teacher and math.isfinite(teacher) else math.nan,best=max((v[3] for v in vals),default=math.nan),complete=any('TRAIN COMPLETE' in x for x in lines))
def F(x): return 'n/a' if not math.isfinite(x) else f'{x:.4f}'
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--mode',default='SMOKE20'); ap.add_argument('--stamp',required=True); a=ap.parse_args(); root=Path('logs')
 names=['A0_R415_Control','A1_LocationOnly','A2_LocationSharedOffset']; data={}
 for n in names:
  fs=sorted(root.glob(f'V552R417_ABL_{n}_{a.mode}_seed42_gpu*_{a.stamp}.log')); data[n]=parse(fs[-1]) if fs else {}
 print('Variant                LocR   PeakR  SpCov  IdCov  BoxIoU SOracle Retain BestM2 Complete')
 for n in names:
  d=data[n]; print(f'{n:<22} {F(d.get("loc",math.nan)):>6} {F(d.get("peak",math.nan)):>6} {F(d.get("sp",math.nan)):>6} {F(d.get("id",math.nan)):>6} {F(d.get("box",math.nan)):>6} {F(d.get("student",math.nan)):>7} {F(d.get("retain",math.nan)):>6} {F(d.get("best",math.nan)):>6} {d.get("complete",False)}')
 full=data.get('A2_LocationSharedOffset',{}); ready=all([full.get('complete',False),full.get('peak',0)>=.75,full.get('sp',0)>=.60,full.get('id',0)>=.50,full.get('box',0)>=.30,full.get('student',0)>=.005,full.get('retain',0)>=.20])
 print('\n[FORMAL READINESS]', 'READY' if ready else 'NOT_READY')
 if not ready: print('Do not run FORMAL150 yet; diagnose the first failed stage in A2.')
if __name__=='__main__': main()
