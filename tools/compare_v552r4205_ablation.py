#!/usr/bin/env python3
from __future__ import annotations
import argparse, math, re
from pathlib import Path
PAIR=re.compile(r'([A-Za-z0-9_]+)=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')
EP=re.compile(r'EPOCH:\s*(\d+)')
VAL=re.compile(r'VAL_NATIVE epoch=(\d+).*?base DSC/NSD=([0-9.]+)/([0-9.]+).*?M2 DSC/NSD=([0-9.]+)/([0-9.]+).*?actionOracle DSC/NSD=([0-9.]+)/([0-9.]+).*?componentOracle DSC/NSD=([0-9.]+)/([0-9.]+)')
FP=re.compile(r'\[V552R4204_BASE_FINGERPRINT\]\s+sha256=([0-9a-f]+)')
def parse(p):
    ep=None; rows=[]; vals=[]; lines=Path(p).read_text(errors='replace').splitlines()
    for line in lines:
        m=EP.search(line)
        if m: ep=int(m.group(1))
        if 'M1_DIAG:' in line: rows.append((ep,{k:float(v) for k,v in PAIR.findall(line)}))
        m=VAL.search(line)
        if m: vals.append(tuple(map(float,m.groups())))
    fps=[FP.search(x).group(1) for x in lines if FP.search(x)]
    return {'path':Path(p),'rows':rows,'vals':vals,'fp':fps[-1] if fps else None}
def dlast(x): return x['rows'][-1][1] if x['rows'] else {}
def g(x,k,alt=None):
    d=dlast(x)
    if k in d:return d[k]
    if alt and alt in d:return d[alt]
    return float('nan')
def f(x,n=6):return 'n/a' if not math.isfinite(x) else f'{x:.{n}f}'
def ratio(a,b):return a/b if math.isfinite(a) and math.isfinite(b) and abs(b)>1e-12 else float('nan')
def one(pattern):
    xs=sorted(Path('.').glob(pattern),key=lambda p:p.stat().st_mtime,reverse=True)
    if not xs: raise SystemExit('[FAIL] missing '+pattern)
    return xs[0]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stamp',required=True); ap.add_argument('--seed',type=int,default=42); a=ap.parse_args()
    p0=one(f'logs/V552R4205_A0_R4204_FairControl_SMOKE20_seed{a.seed}_gpu*_{a.stamp}.log')
    p1=one(f'logs/V552R4205_A1_CapacityConsistentOverflow_SMOKE20_seed{a.seed}_gpu*_{a.stamp}.log')
    c,t=parse(p0),parse(p1)
    print('='*170);print('V552-R4.20.5 CAUSAL ABLATION | R4.20.4 CONTROL vs CAPACITY-CONSISTENT OVERFLOW');print('='*170)
    print('Metric                         Control        R4205        Delta')
    metrics=[
      ('NativeMatch','v552r4204_native_mask_matching_dice'),('Purity','v552r4204_native_mask_soft_purity'),
      ('Capture','v538_component_capture_ratio'),('StudentOracle','v538_component_oracle_gain'),
      ('Retention',None),('OccupancyDice','v552r4204_occupancy_soft_dice')]
    for name,key in metrics:
        if name=='Retention':
            x=ratio(g(c,'v538_component_oracle_gain'),g(c,'v538_action_realizable_teacher_oracle_gain'))
            y=ratio(g(t,'v538_component_oracle_gain'),g(t,'v538_action_realizable_teacher_oracle_gain'))
        else: x,y=g(c,key),g(t,key)
        print(f'{name:<28} {f(x):>12} {f(y):>12} {f(y-x):>+12}')
    print('\n[R4205 CAPACITY DIAGNOSTICS]')
    for key in ['v552r4205_overflow_target_residual_fraction','v552r4205_overflow_predicted_residual_fraction','v552r4205_overflow_soft_dice','v552r4205_editable_on_overflow_leakage','v552r4205_overflow_on_retained_leakage','v552r4205_target_decomposition_error','v552r4205_prediction_decomposition_error','v552r4205_final_logits_finite_fraction']:
        print(f'{key:<52} {f(g(t,key))}')
    print('\n[FAIRNESS]')
    print('[PASS] identical Base/PVL fingerprint '+str(c['fp']) if c['fp'] and c['fp']==t['fp'] else '[FAIL] fingerprints differ/missing: '+str((c['fp'],t['fp'])))
    if c['vals'] and t['vals']:
        cm={int(v[0]):v for v in c['vals']}; tm={int(v[0]):v for v in t['vals']}; common=sorted(set(cm)&set(tm))
        if common:
            gaps=[abs(cm[e][1]-tm[e][1]) for e in common]
            print('max Base DSC trajectory gap='+f(max(gaps)))
    print('\n[DECISION]')
    match=g(t,'v552r4204_native_mask_matching_dice'); pur=g(t,'v552r4204_native_mask_soft_purity'); cap=g(t,'v538_component_capture_ratio'); so=g(t,'v538_component_oracle_gain'); ret=ratio(so,g(t,'v538_action_realizable_teacher_oracle_gain'))
    ready=match>=.20 and pur>=.12 and cap>=.15 and so>=.005 and ret>=.20
    print(f'R4205 M1 readiness={"PASS" if ready else "FAIL"} Match={f(match)} Purity={f(pur)} Capture={f(cap)} SOracle={f(so)} Retention={f(ret)}')
    if ready: print('Next: audit M2 signed utility. Do not add positional identity unless an isolated identity failure remains.')
    elif ret>=.20 and match<.20: print('Capacity allocation improved but identity still fails: next ablation should be EXOGENOUS positional identity, not self-derived centroid feedback.')
    else: print('Capacity/identity is still not ready: inspect overflow learning and editable leakage before adding any new module or tuning M2.')
if __name__=='__main__':main()
