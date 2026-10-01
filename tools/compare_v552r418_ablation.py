#!/usr/bin/env python3
from __future__ import annotations
import argparse, math, re
from pathlib import Path
PAIR=re.compile(r'([A-Za-z0-9_]+)=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')
VAL=re.compile(r'VAL_NATIVE epoch=(\d+).*?base DSC/NSD=([0-9.]+)/([0-9.]+).*?M2 DSC/NSD=([0-9.]+)/([0-9.]+).*?componentOracle DSC/NSD=([0-9.]+)/([0-9.]+)')
def parse(p):
    lines=p.read_text(errors='replace').splitlines(); rows=[{k:float(v) for k,v in PAIR.findall(x)} for x in lines if 'M1_DIAG:' in x]; vals=[tuple(map(float,m.groups())) for x in lines if (m:=VAL.search(x))]; r=rows[-1] if rows else {}
    teacher=r.get('v538_action_realizable_teacher_oracle_gain',math.nan); student=r.get('v538_component_oracle_gain',math.nan)
    return dict(loc=r.get('v552r417_location_center_recall',math.nan),native=r.get('v552r418_native_mask_matching_dice',math.nan),pair=r.get('v552r418_paired_mask_dice',math.nan),cons=r.get('v552r418_paired_target_consistency',math.nan),capture=r.get('v538_component_capture_ratio',math.nan),student=student,retain=student/teacher if math.isfinite(teacher) and abs(teacher)>1e-12 else math.nan,gain=r.get('v552r44_quality_candidate_audit_mean_gain',math.nan),harm=r.get('v552r44_quality_candidate_audit_harm_rate',math.nan),best=max((v[3] for v in vals),default=math.nan),complete=any('TRAIN COMPLETE' in x for x in lines))
def F(x): return 'n/a' if not math.isfinite(x) else f'{x:.4f}'
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',default='SMOKE20'); ap.add_argument('--stamp',required=True); ap.add_argument('--seed',type=int,default=42); a=ap.parse_args(); root=Path('logs')
    names=['A0_R417_Control','A1_BoxFreeMaskSet','A2_FULL']; data={}
    for n in names:
        fs=sorted(root.glob(f'V552R418_ABL_{n}_{a.mode}_seed{a.seed}_gpu*_{a.stamp}.log')); data[n]=parse(fs[-1]) if fs else {}
    print('Variant              LocR NativeMatch PairDice PairCons Capture SOracle Retain CandGain CandHarm BestM2 Complete')
    for n in names:
        d=data[n]; print(f'{n:<20} {F(d.get("loc",math.nan)):>5} {F(d.get("native",math.nan)):>11} {F(d.get("pair",math.nan)):>8} {F(d.get("cons",math.nan)):>8} {F(d.get("capture",math.nan)):>7} {F(d.get("student",math.nan)):>7} {F(d.get("retain",math.nan)):>6} {F(d.get("gain",math.nan)):>8} {F(d.get("harm",math.nan)):>8} {F(d.get("best",math.nan)):>6} {d.get("complete",False)}')
    a0=data.get('A0_R417_Control',{}); a1=data.get('A1_BoxFreeMaskSet',{}); a2=data.get('A2_FULL',{})
    def delta(k,x,y):
        a=x.get(k,math.nan); b=y.get(k,math.nan); return b-a if math.isfinite(a) and math.isfinite(b) else math.nan
    print('\n[Causal deltas]')
    print(f'A1-A0 (remove center/box/ROI bottleneck): ΔCapture={F(delta("capture",a0,a1))} ΔStudentOracle={F(delta("student",a0,a1))} ΔBestM2={F(delta("best",a0,a1))}')
    print(f'A2-A1 (paired stable coarse supervision): ΔNativeMatch={F(delta("native",a1,a2))} ΔCapture={F(delta("capture",a1,a2))} ΔStudentOracle={F(delta("student",a1,a2))} ΔBestM2={F(delta("best",a1,a2))}')
    ready=all([a2.get('complete',False),a2.get('cons',0)>=.999,a2.get('pair',0)>=.50,a2.get('native',0)>=.20,a2.get('capture',0)>=.15,a2.get('student',0)>=.005,a2.get('retain',0)>=.20,a2.get('gain',-1)>0,a2.get('harm',1)<=.10])
    print('\n[FORMAL READINESS]', 'READY' if ready else 'NOT_READY')
    if ready: print('A2 closes the current M1 root chain. Freeze the structure and select formal checkpoints by VAL native_m2_dice only; do not use Test to re-select.')
    else: print('Do not run FORMAL150 yet. Diagnose the first failed R4.18 stage; do not lower the deployment gate to manufacture execution.')
if __name__=='__main__': main()
