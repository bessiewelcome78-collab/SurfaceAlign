#!/usr/bin/env python3
from __future__ import annotations
import argparse,math,re
from pathlib import Path
PAIR=re.compile(r'([A-Za-z0-9_]+)=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')
VAL=re.compile(r'VAL_NATIVE epoch=(\d+).*?base DSC/NSD=([0-9.]+)/([0-9.]+).*?M2 DSC/NSD=([0-9.]+)/([0-9.]+).*?componentOracle DSC/NSD=([0-9.]+)/([0-9.]+)')
def parse(p):
    lines=p.read_text(errors='replace').splitlines(); rows=[{k:float(v) for k,v in PAIR.findall(x)} for x in lines if 'M1_DIAG:' in x]; vals=[tuple(map(float,m.groups())) for x in lines if (m:=VAL.search(x))]; r=rows[-1] if rows else {}
    teacher=r.get('v538_action_realizable_teacher_oracle_gain',math.nan); student=r.get('v538_component_oracle_gain',math.nan)
    native=r.get('v552r420_native_mask_matching_dice',r.get('v552r419_native_mask_matching_dice',math.nan)); purity=r.get('v552r420_native_mask_soft_purity',r.get('v552r419_native_mask_soft_purity',math.nan)); coverage=r.get('v552r420_native_mask_soft_coverage',r.get('v552r419_native_mask_soft_coverage',math.nan))
    return dict(loc=r.get('v552r417_location_center_recall',math.nan),peak=r.get('v552r417_pre_topk_peak_recall',math.nan),native=native,purity=purity,coverage=coverage,capture=r.get('v538_component_capture_ratio',math.nan),student=student,retain=student/teacher if math.isfinite(teacher) and abs(teacher)>1e-12 else math.nan,gain=r.get('v552r44_quality_candidate_audit_mean_gain',math.nan),harm=r.get('v552r44_quality_candidate_audit_harm_rate',math.nan),best=max((v[3] for v in vals),default=math.nan),complete=any('TRAIN COMPLETE' in x for x in lines),fatal=any('Traceback (most recent call last)' in x for x in lines))
def F(x): return 'n/a' if not math.isfinite(x) else f'{x:.4f}'
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',default='SMOKE20'); ap.add_argument('--stamp',required=True); ap.add_argument('--seed',type=int,default=42); a=ap.parse_args(); root=Path('logs')
    names=['A0_R419_SeededMaskedAttention','A1_DynamicResidualMask','A2_DynamicTypeDecoupled']; data={}
    for n in names:
        fs=sorted(root.glob(f'V552R420_ABL_{n}_{a.mode}_seed{a.seed}_gpu*_{a.stamp}.log')); data[n]=parse(fs[-1]) if fs else {}
    print('Variant                         LocR PeakR NativeMatch Purity Coverage Capture SOracle Retain CandGain Harm BestM2 Complete')
    for n in names:
        d=data[n]; print(f'{n:<31} {F(d.get("loc",math.nan)):>5} {F(d.get("peak",math.nan)):>5} {F(d.get("native",math.nan)):>11} {F(d.get("purity",math.nan)):>6} {F(d.get("coverage",math.nan)):>8} {F(d.get("capture",math.nan)):>7} {F(d.get("student",math.nan)):>7} {F(d.get("retain",math.nan)):>6} {F(d.get("gain",math.nan)):>8} {F(d.get("harm",math.nan)):>5} {F(d.get("best",math.nan)):>6} {d.get("complete",False) and not d.get("fatal",False)}')
    a0,a1,a2=(data[n] for n in names)
    def D(k,x,y):
        u=x.get(k,math.nan);v=y.get(k,math.nan);return v-u if math.isfinite(u) and math.isfinite(v) else math.nan
    print('\n[Causal deltas]')
    print(f'A1-A0 dynamic-mask decoder: ΔNativeMatch={F(D("native",a0,a1))} ΔPurity={F(D("purity",a0,a1))} ΔCapture={F(D("capture",a0,a1))} ΔStudentOracle={F(D("student",a0,a1))}')
    print(f'A2-A1 type/shape decouple:  ΔNativeMatch={F(D("native",a1,a2))} ΔPurity={F(D("purity",a1,a2))} ΔCapture={F(D("capture",a1,a2))} ΔStudentOracle={F(D("student",a1,a2))}')
    ready=all([a2.get('complete',False),not a2.get('fatal',True),a2.get('loc',0)>=.90,a2.get('peak',0)>=.75,a2.get('native',0)>=.20,a2.get('purity',0)>=.12,a2.get('capture',0)>=.15,a2.get('student',0)>=.005,a2.get('retain',0)>=.20,a2.get('gain',-1)>0,a2.get('harm',1)<=.10])
    print('\n[FORMAL READINESS]', 'READY' if ready else 'NOT_READY')
    if not ready: print('Do NOT run FORMAL150 or loosen M2 gates. First verify A1-A0 closes mask realization; only then treat PeakR or utility as the next bottleneck.')
    else: print('R4.20 closes the current M1 realization chain. Freeze M1 structure before any M2 redesign.')
if __name__=='__main__': main()
