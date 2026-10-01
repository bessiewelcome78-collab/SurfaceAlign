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
    d=rows[-1] if rows else {}
    v=vals[-1] if vals else (float('nan'),)*7
    return d,v,best
def stage(p):
    name=Path(p).name
    for x in ('C0_AnchorControl','C1_InteriorAnchor','C2_VariableSeeds','C3_IndependentOverflow','C4_FullM1'):
        if x in name:return x
    return name[:22]
def g(d,k):return d.get(k,float('nan'))
def f(x):return 'n/a' if not math.isfinite(x) else f'{x:.6f}'
metrics=[
 ('BBoxInside','v552r4210_bbox_center_inside_teacher_rate'),('PeakRecall','v552r417_pre_topk_peak_recall'),
 ('ValidSeeds','v552r4210_valid_seed_count'),('SeedPrecision','v552r4208_seed_teacher_precision'),
 ('TeacherCoverage','v552r4208_teacher_seed_coverage'),('FalseSeed','v552r4208_seed_without_teacher_rate'),
 ('Duplicate','v552r4208_seed_duplicate_teacher_rate'),('NativeMatch','v552r4204_native_mask_matching_dice'),
 ('Capture','v538_component_capture_ratio'),('StudentOracle','v538_component_oracle_gain'),
 ('OverflowDice','v552r4205_overflow_soft_dice'),('EditableLeak','v552r4205_editable_on_overflow_leakage')]
items=[(stage(p),)+parse(p) for p in sys.argv[1:]]
items.sort(key=lambda z:('C0','C1','C2','C3','C4').index(z[0][:2]) if z[0][:2] in ('C0','C1','C2','C3','C4') else 99)
print('='*180);print('V552-R4.21.0 CAUSAL ABLATION | C0 -> C1 -> C2 -> C3 -> C4');print('='*180)
print(f'{"Metric":<22}' + ''.join(f'{x[0]:>28}' for x in items))
for n,k in metrics:
    print(f'{n:<22}'+''.join(f'{f(g(x[1],k)):>28}' for x in items))
print(f'{"VAL Base":<22}'+''.join(f'{f(x[2][1]):>28}' for x in items))
print(f'{"VAL M1Native":<22}'+''.join(f'{f(x[2][3]):>28}' for x in items))
print(f'{"VAL NativeGain":<22}'+''.join(f'{f(x[2][3]-x[2][1]):>28}' for x in items))
print(f'{"VAL ComponentOracle":<22}'+''.join(f'{f(x[2][5]):>28}' for x in items))
print(f'{"Best M1Native":<22}'+''.join(f'{f(x[3][1]):>28}' for x in items))
