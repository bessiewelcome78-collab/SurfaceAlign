#!/usr/bin/env python3
from __future__ import annotations
import re,sys,math
from pathlib import Path
PAIR=re.compile(r'([A-Za-z0-9_]+)=([-+0-9.eE]+)')
VAL=re.compile(r'VAL_NATIVE epoch=(\d+).*?base DSC/NSD=([0-9.]+)/([0-9.]+).*?M1Native DSC/NSD=([0-9.]+)/([0-9.]+).*?componentOracle DSC/NSD=([0-9.]+)/([0-9.]+)')
BEST=re.compile(r'best_m1_epoch=(\d+) best_m1_native_dice=([0-9.]+)')
def parse(p):
    rows=[]; vals=[]; best=(None,float('nan'))
    for line in Path(p).read_text(errors='replace').splitlines():
        if 'M1_DIAG:' in line: rows.append({k:float(v) for k,v in PAIR.findall(line)})
        m=VAL.search(line)
        if m: vals.append(tuple(map(float,m.groups())))
        m=BEST.search(line)
        if m: best=(int(m.group(1)),float(m.group(2)))
    return (rows[-1] if rows else {}),(vals[-1] if vals else (float('nan'),)*7),best
def stage(p):
    n=Path(p).name
    for x in ('D0_HardSeedGateControl','D1_ProposalExistence','D2_GeometryOverflow','D3_FullDecoupledM1'):
        if x in n:return x
    return n[:25]
def g(d,k):return d.get(k,float('nan'))
def f(x):return 'n/a' if not math.isfinite(x) else f'{x:.6f}'
metrics=[
 ('PeakRecall','v552r417_pre_topk_peak_recall'),('ProposalSeeds','v552r4211_proposal_seed_count'),('TeacherCount','v552r4210_teacher_component_count'),('ProposalConfidence','v552r4211_proposal_confidence_mean'),
 ('SeedPrecision','v552r4208_seed_teacher_precision'),('TeacherCoverage','v552r4208_teacher_seed_coverage'),('FalseSeed','v552r4208_seed_without_teacher_rate'),('Duplicate','v552r4208_seed_duplicate_teacher_rate'),
 ('PresenceExpected','v552r4211_presence_expected_count'),('PresenceHard','v552r4211_presence_hard_count'),('PresenceGap','v552r4211_presence_minus_teacher_count'),
 ('Q0Identity','v552r4208_q0_seed_identity_cosine'),('Q1Identity','v552r4208_q1_seed_identity_cosine'),('Collision','v552r4207_teacher_best_slot_collision_rate'),('Margin','v552r4207_teacher_best_slot_margin'),
 ('NativeMatch','v552r4204_native_mask_matching_dice'),('Purity','v552r4204_native_mask_soft_purity'),('Capture','v538_component_capture_ratio'),('StudentOracle','v538_component_oracle_gain'),
 ('GeoEffectiveL1','v552r4211_geometry_effective_l1'),('OverflowDice','v552r4205_overflow_soft_dice'),('EditableLeak','v552r4205_editable_on_overflow_leakage')]
items=[(stage(p),)+parse(p) for p in sys.argv[1:]]
order={'D0':0,'D1':1,'D2':2,'D3':3};items.sort(key=lambda z:order.get(z[0][:2],99))
print('='*185);print('V552-R4.21.1 CAUSAL ABLATION | D0 hard gate -> D1 proposal/existence -> D2 geometry/overflow -> D3 deployable alignment');print('='*185)
print(f'{"Metric":<22}'+''.join(f'{x[0]:>31}' for x in items))
for n,k in metrics:print(f'{n:<22}'+''.join(f'{f(g(x[1],k)):>31}' for x in items))
print(f'{"VAL Base":<22}'+''.join(f'{f(x[2][1]):>31}' for x in items))
print(f'{"VAL M1Native":<22}'+''.join(f'{f(x[2][3]):>31}' for x in items))
print(f'{"VAL NativeGain":<22}'+''.join(f'{f(x[2][3]-x[2][1]):>31}' for x in items))
print(f'{"VAL ComponentOracle":<22}'+''.join(f'{f(x[2][5]):>31}' for x in items))
print(f'{"Best M1Native":<22}'+''.join(f'{f(x[3][1]):>31}' for x in items))
