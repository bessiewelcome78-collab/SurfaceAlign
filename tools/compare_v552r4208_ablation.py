#!/usr/bin/env python3
from __future__ import annotations
import argparse, math, re
from pathlib import Path
PAIR=re.compile(r'([A-Za-z0-9_]+)=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)'); EP=re.compile(r'EPOCH:\s*(\d+)'); FP=re.compile(r'\[V552R4204_BASE_FINGERPRINT\]\s+sha256=([0-9a-f]+)')
def parse(p):
    ep=None; rows=[]; lines=Path(p).read_text(errors='replace').splitlines()
    for line in lines:
        m=EP.search(line)
        if m: ep=int(m.group(1))
        if 'M1_DIAG:' in line: rows.append((ep,{k:float(v) for k,v in PAIR.findall(line)}))
    fps=[FP.search(x).group(1) for x in lines if FP.search(x)]
    return {'p':Path(p),'d':rows[-1][1] if rows else {},'fp':fps[-1] if fps else None}
def f(x,n=6): return 'n/a' if not math.isfinite(x) else f'{x:.{n}f}'
def signed(x): return 'n/a' if not math.isfinite(x) else f'{x:+.6f}'
def g(x,k): return x['d'].get(k,float('nan'))
def retention(x):
    a=g(x,'v538_component_oracle_gain'); b=g(x,'v538_action_realizable_teacher_oracle_gain'); return a/b if math.isfinite(a) and math.isfinite(b) and abs(b)>1e-12 else float('nan')
def latest(pattern):
    xs=sorted(Path('logs').glob(pattern),key=lambda p:p.stat().st_mtime,reverse=True); return xs[0] if xs else None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stamp',required=True); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--include-a4',action='store_true'); a=ap.parse_args()
    patterns=[('A0','V552R4208_A0_R4205_Control_SMOKE20'),('A2','V552R4208_A2_NormalizedFusion_SMOKE20'),('A3','V552R4208_A3_PersistentIdentity_SMOKE20')]
    if a.include_a4: patterns.append(('A4','V552R4208_A4_SeedConsistentMatching_SMOKE20'))
    runs=[]
    for label,tag in patterns:
        p=latest(f'{tag}_seed{a.seed}_gpu*_{a.stamp}.log')
        if p is None: raise SystemExit('[FAIL] missing '+tag)
        runs.append((label,parse(p)))
    print('='*220); print('V552-R4.20.8 CAUSAL LADDER | A0 FIXED → A2 NORMALIZED → A3 PERSISTENT' + (' → A4 SEED-MATCH' if a.include_a4 else '')); print('='*220)
    headers='Metric'.ljust(34)+''.join(label.rjust(15) for label,_ in runs)
    print(headers)
    metrics=[
      ('SeedTeacherPrecision','v552r4208_seed_teacher_precision'),('TeacherSeedCoverage','v552r4208_teacher_seed_coverage'),('SeedDuplicateRate','v552r4208_seed_duplicate_teacher_rate'),
      ('LearnedQueryNorm','v552r4208_learned_query_norm'),('SeedFeatureNorm','v552r4208_seed_feature_norm'),('Seed/LearnedNorm','v552r4208_seed_to_learned_norm_ratio'),
      ('Q0SeedIdentity','v552r4208_q0_seed_identity_cosine'),('Q1SeedIdentity','v552r4208_q1_seed_identity_cosine'),('IdentityDelta','v552r4208_identity_retention_delta'),
      ('Collision','v552r4207_teacher_best_slot_collision_rate'),('Margin','v552r4207_teacher_best_slot_margin'),('BestSlotDice','v552r4207_teacher_best_slot_dice'),
      ('NativeMatch','v552r4204_native_mask_matching_dice'),('Purity','v552r4204_native_mask_soft_purity'),('Capture','v538_component_capture_ratio'),('StudentOracle','v538_component_oracle_gain'),('TeacherOracle','v538_action_realizable_teacher_oracle_gain'),('BenefitRate','v541_benefit_target_rate'),('LockedFraction','v552r4208_seed_locked_fraction')]
    for name,key in metrics: print(name.ljust(34)+''.join(f(g(x,key)).rjust(15) for _,x in runs))
    print('Retention'.ljust(34)+''.join(f(retention(x)).rjust(15) for _,x in runs))
    print('\n[FAIRNESS]')
    fps=[x['fp'] for _,x in runs]; print('[PASS] identical Base fingerprint '+str(fps[0]) if fps and all(z==fps[0] and z for z in fps) else '[FAIL] fingerprint mismatch '+str(fps))
    base=runs[0][1]
    print('\n[DELTA vs A0]')
    for label,x in runs[1:]:
        print(label, 'Collision',signed(g(x,'v552r4207_teacher_best_slot_collision_rate')-g(base,'v552r4207_teacher_best_slot_collision_rate')),
              'Margin',signed(g(x,'v552r4207_teacher_best_slot_margin')-g(base,'v552r4207_teacher_best_slot_margin')),
              'Match',signed(g(x,'v552r4204_native_mask_matching_dice')-g(base,'v552r4204_native_mask_matching_dice')),
              'Capture',signed(g(x,'v538_component_capture_ratio')-g(base,'v538_component_capture_ratio')),
              'Oracle',signed(g(x,'v538_component_oracle_gain')-g(base,'v538_component_oracle_gain')),
              'Retention',signed(retention(x)-retention(base)))
    print('\nInterpretation order: seed-source audit → q0/q1 identity retention → collision/margin → Match → Capture/Oracle/Retention. Do not select a variant from Purity alone.')
if __name__=='__main__': main()
