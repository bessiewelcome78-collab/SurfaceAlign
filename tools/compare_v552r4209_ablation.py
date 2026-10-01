#!/usr/bin/env python3
import argparse, math, re
from pathlib import Path
PAIR=re.compile(r'([A-Za-z0-9_]+)=([-+0-9.eE]+)')
VAL=re.compile(r'VAL_NATIVE epoch=(\d+).*?base DSC/NSD=([0-9.]+)/([0-9.]+).*?(?:M1Native DSC/NSD=([0-9.]+)/([0-9.]+).*?)?M2 DSC/NSD=([0-9.]+)/([0-9.]+).*?componentOracle DSC/NSD=([0-9.]+)/([0-9.]+)')
def read(p):
    rows=[]; vals=[]
    for x in Path(p).read_text(errors='replace').splitlines():
        if 'M1_DIAG:' in x: rows.append({k:float(v) for k,v in PAIR.findall(x)})
        m=VAL.search(x)
        if m: vals.append(tuple(float(z) if z is not None else float('nan') for z in m.groups()))
    return (rows[-1] if rows else {}),(vals[-1] if vals else None)
def g(d,k): return d.get(k,float('nan'))
def fmt(x): return 'n/a' if not math.isfinite(x) else f'{x:.6f}'
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('logs',nargs='+'); a=ap.parse_args()
    parsed=[(Path(p).name,)+read(p) for p in a.logs]
    metrics=[('PeakRecall','v552r417_pre_topk_peak_recall'),('SeedPrec','v552r4208_seed_teacher_precision'),('TeacherCov','v552r4208_teacher_seed_coverage'),('FalseSeed','v552r4208_seed_without_teacher_rate'),('Q0Id','v552r4208_q0_seed_identity_cosine'),('Q1Id','v552r4208_q1_seed_identity_cosine'),('Collision','v552r4207_teacher_best_slot_collision_rate'),('Margin','v552r4207_teacher_best_slot_margin'),('Match','v552r4204_native_mask_matching_dice'),('Purity','v552r4204_native_mask_soft_purity'),('Capture','v538_component_capture_ratio'),('StudentOracle','v538_component_oracle_gain'),('OverflowDice','v552r4205_overflow_soft_dice'),('OverflowLeak','v552r4205_editable_on_overflow_leakage')]
    print('Metric'.ljust(20)+''.join(n[:18].center(20) for n,_,_ in parsed))
    for label,key in metrics: print(label.ljust(20)+''.join(fmt(g(d,key)).center(20) for _,d,_ in parsed))
    print('Retention'.ljust(20)+''.join(fmt(g(d,'v538_component_oracle_gain')/g(d,'v538_action_realizable_teacher_oracle_gain') if abs(g(d,'v538_action_realizable_teacher_oracle_gain'))>1e-12 else float('nan')).center(20) for _,d,_ in parsed))
    print('ValM1Native'.ljust(20)+''.join(fmt(v[3] if v else float('nan')).center(20) for _,_,v in parsed))
    print('ValCompOracle'.ljust(20)+''.join(fmt(v[7] if v else float('nan')).center(20) for _,_,v in parsed))
if __name__=='__main__': main()
