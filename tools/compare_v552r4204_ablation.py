#!/usr/bin/env python3
from __future__ import annotations
import argparse, math, re
from pathlib import Path

PAIR=re.compile(r'([A-Za-z0-9_]+)=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')
EP=re.compile(r'EPOCH:\s*(\d+)')
VAL=re.compile(r'VAL_NATIVE epoch=(\d+).*?base DSC/NSD=([0-9.]+)/([0-9.]+).*?M2 DSC/NSD=([0-9.]+)/([0-9.]+).*?actionOracle DSC/NSD=([0-9.]+)/([0-9.]+).*?componentOracle DSC/NSD=([0-9.]+)/([0-9.]+)')
FP=re.compile(r'\[V552R4204_BASE_FINGERPRINT\]\s+sha256=([0-9a-f]+)')

def parse(path):
    lines=Path(path).read_text(errors='replace').splitlines(); ep=None; rows=[]; vals=[]
    for line in lines:
        m=EP.search(line)
        if m: ep=int(m.group(1))
        if 'M1_DIAG:' in line: rows.append((ep,{k:float(v) for k,v in PAIR.findall(line)}))
        m=VAL.search(line)
        if m: vals.append(tuple(map(float,m.groups())))
    fps=[FP.search(x).group(1) for x in lines if FP.search(x)]
    return {'path':Path(path),'rows':rows,'vals':vals,'fingerprint':fps[-1] if fps else None}

def last(d): return d['rows'][-1][1] if d['rows'] else {}
def g(d,k,*fallback):
    r=last(d)
    for x in (k,)+fallback:
        if x in r:return r[x]
    return float('nan')
def ratio(a,b): return a/b if math.isfinite(a) and math.isfinite(b) and abs(b)>1e-12 else float('nan')
def f(x,n=6):return 'n/a' if not math.isfinite(x) else f'{x:.{n}f}'

def find_logs(stamp,seed):
    pats=[
      f'logs/V552R4204_A0_R4203_FairControl_SMOKE20_seed{seed}_gpu*_{stamp}.log',
      f'logs/V552R4204_A1_FactorizedExistenceIdentity_SMOKE20_seed{seed}_gpu*_{stamp}.log',
      f'logs/V552R4204_A2_SelfDerivedSpatialIdentity_SMOKE20_seed{seed}_gpu*_{stamp}.log',
    ]
    out=[]
    for pat in pats:
        xs=sorted(Path('.').glob(pat),key=lambda p:p.stat().st_mtime,reverse=True)
        if not xs: raise SystemExit('[FAIL] missing '+pat)
        out.append(xs[0])
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stamp',required=True); ap.add_argument('--seed',type=int,default=42); args=ap.parse_args()
    names=['A0_R4203_FairControl','A1_FactorizedExistenceIdentity','A2_SelfDerivedSpatialIdentity']
    ds=[parse(p) for p in find_logs(args.stamp,args.seed)]
    print('='*180);print('V552-R4.20.4 ABLATION | FACTORIZED RESIDUAL EXISTENCE-IDENTITY');print('='*180)
    print('Variant                    Match   Purity   Cover Capture SOracle Retain OccDice KeepR UnkeepR CondH CentSep HarmNeg  BaseDSC   M2Gain CompOracle')
    for n,d in zip(names,ds):
        match=g(d,'v552r4204_native_mask_matching_dice','v552r4203_native_mask_matching_dice')
        pur=g(d,'v552r4204_native_mask_soft_purity','v552r4203_native_mask_soft_purity')
        cov=g(d,'v552r4204_native_mask_soft_coverage','v552r4203_native_mask_soft_coverage')
        cap=g(d,'v538_component_capture_ratio'); so=g(d,'v538_component_oracle_gain'); to=g(d,'v538_action_realizable_teacher_oracle_gain'); ret=ratio(so,to)
        val=d['vals'][-1] if d['vals'] else None
        if val:
            _,bd,bn,md,mn,ad,an,cd,cn=val; mg=md-bd; co=cd-bd
        else: bd=mg=co=float('nan')
        print(f'{n:<27} {f(match,3):>6} {f(pur,3):>7} {f(cov,3):>7} {f(cap,3):>7} {f(so,5):>7} {f(ret,3):>6} '
              f'{f(g(d,"v552r4204_occupancy_soft_dice"),3):>7} {f(g(d,"v552r4204_retained_teacher_residual_fraction"),3):>5} {f(g(d,"v552r4204_unretained_teacher_residual_fraction"),3):>7} '
              f'{f(g(d,"v552r4204_conditional_slot_entropy"),3):>5} {f(g(d,"v552r4204_centroid_separation"),3):>7} '
              f'{f(g(d,"v542_harm_gain_negative_rate"),3):>7} {f(bd,6):>9} {f(mg,6):>8} {f(co,6):>10}')
    print('\n[BASE FAIRNESS]')
    fps=[d['fingerprint'] for d in ds]
    if all(fps) and len(set(fps))==1: print('[PASS] shared Base/PVL initialization fingerprints are identical:',fps[0])
    else: print('[FAIL] Base/PVL fingerprints differ or are missing:',fps)
    maps=[{int(v[0]):v for v in d['vals']} for d in ds]
    common=sorted(set(maps[0])&set(maps[1])&set(maps[2]))
    if common:
        gaps=[]
        for e in common:
            bases=[m[e][1] for m in maps]; gaps.append(max(bases)-min(bases))
        print('mean cross-variant Base DSC range =',f(sum(gaps)/len(gaps)))
        print('max  cross-variant Base DSC range =',f(max(gaps)))
        if max(gaps)<=0.002: print('[PASS] Base trajectories are close enough for mechanism attribution')
        else: print('[WARN] Base trajectories still diverge >0.002; do not attribute final DSC differences only to M1')
    print('\n[OCCUPANCY SEMANTICS / CAPACITY]')
    for n,d in zip(names,ds):
        if g(d,'v552r4204_rootfix_enabled') < .5:
            continue
        mae=g(d,'v552r4204_occupancy_target_teacher_error_mae')
        kept=g(d,'v552r4204_retained_teacher_residual_fraction')
        unkept=g(d,'v552r4204_unretained_teacher_residual_fraction')
        print(f'{n}: targetMAE={f(mae)} retainedResidual={f(kept)} unretainedResidual={f(unkept)}')
        if math.isfinite(mae) and mae > 1e-8:
            print('[FAIL] occupancy target is not the full effective residual field')
        if math.isfinite(unkept) and unkept > 0.05:
            print('[WARN] material residual mass is outside retained K Teacher components; inspect Teacher component-count overflow before changing slot identity.')

    print('\n[CAUSAL EFFECTS]')
    a0,a1,a2=ds
    for key,label in [
      ('v552r4204_native_mask_matching_dice','Match'),('v552r4204_native_mask_soft_purity','Purity'),
      ('v538_component_capture_ratio','Capture'),('v538_component_oracle_gain','StudentOracle')]:
        x=g(a0,key,key.replace('v552r4204_','v552r4203_')); y=g(a1,key,key.replace('v552r4204_','v552r4203_')); z=g(a2,key,key.replace('v552r4204_','v552r4203_'))
        print(f'{label:<14} A0={f(x)} A1={f(y)} Δfactorization={f(y-x)} A2={f(z)} Δspatial={f(z-y)}')
    print('\n[GO / NO-GO]')
    best=max(ds[1:],key=lambda d:g(d,'v538_component_oracle_gain'))
    match=g(best,'v552r4204_native_mask_matching_dice'); pur=g(best,'v552r4204_native_mask_soft_purity'); cap=g(best,'v538_component_capture_ratio'); so=g(best,'v538_component_oracle_gain'); ret=ratio(so,g(best,'v538_action_realizable_teacher_oracle_gain'))
    ok=match>=.20 and pur>=.12 and cap>=.15 and so>=.005 and ret>=.20
    print('[M1 READINESS]', 'PASS' if ok else 'FAIL', f'Match={f(match)} Purity={f(pur)} Capture={f(cap)} SOracle={f(so)} Retention={f(ret)}')
    if not ok: print('Do NOT run FORMAL150 and do NOT tune M2 gates. Fix the first failing M1 mechanism.')
    else: print('M1 supply is ready. Next audit signed Harm utility before allowing deployment/Formal150.')

if __name__=='__main__':main()
