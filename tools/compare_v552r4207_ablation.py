#!/usr/bin/env python3
from __future__ import annotations

import argparse, math, re
from pathlib import Path

PAIR = re.compile(r'([A-Za-z0-9_]+)=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')
EP = re.compile(r'EPOCH:\s*(\d+)')
FP = re.compile(r'\[V552R4204_BASE_FINGERPRINT\]\s+sha256=([0-9a-f]+)')
VAL = re.compile(r'VAL_NATIVE epoch=(\d+).*?base DSC/NSD=([0-9.]+)/([0-9.]+).*?M2 DSC/NSD=([0-9.]+)/([0-9.]+).*?actionOracle DSC/NSD=([0-9.]+)/([0-9.]+).*?componentOracle DSC/NSD=([0-9.]+)/([0-9.]+)')


def parse(p):
    ep = None; rows = []; vals = []
    lines = Path(p).read_text(errors='replace').splitlines()
    for line in lines:
        m = EP.search(line)
        if m: ep = int(m.group(1))
        if 'M1_DIAG:' in line:
            rows.append((ep, {k: float(v) for k, v in PAIR.findall(line)}))
        m = VAL.search(line)
        if m: vals.append(tuple(map(float, m.groups())))
    fps = [FP.search(x).group(1) for x in lines if FP.search(x)]
    return {'p': Path(p), 'r': rows, 'v': vals, 'fp': fps[-1] if fps else None}


def one(pat):
    xs = sorted(Path('.').glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
    if not xs: raise SystemExit('[FAIL] missing ' + pat)
    return xs[0]


def last(x): return x['r'][-1][1]
def g(x,k): return last(x).get(k,float('nan'))
def f(x,n=6): return 'n/a' if not math.isfinite(x) else f'{x:.{n}f}'
def ratio(a,b): return a/b if math.isfinite(a) and math.isfinite(b) and abs(b)>1e-12 else float('nan')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--stamp', required=True); ap.add_argument('--seed', type=int, default=42); a = ap.parse_args()
    c = parse(one(f'logs/V552R4207_A0_R4205_Control_SMOKE20_seed{a.seed}_gpu*_{a.stamp}.log'))
    t = parse(one(f'logs/V552R4207_A1_DynamicVisualBinding_SMOKE20_seed{a.seed}_gpu*_{a.stamp}.log'))
    print('='*190)
    print('V552-R4.20.7 CAUSAL ABLATION | R4.20.5 FIXED QUERY vs DYNAMIC VISUAL INSTANCE BINDING')
    print('='*190)
    print('Metric                                  A0 R4205        A1 R4207           Delta')
    items = [
        ('TeacherBestSlotCollision','v552r4207_teacher_best_slot_collision_rate'),
        ('TeacherBestSlotMargin','v552r4207_teacher_best_slot_margin'),
        ('TeacherBestSlotDice','v552r4207_teacher_best_slot_dice'),
        ('NativeMatch','v552r4204_native_mask_matching_dice'),
        ('Purity','v552r4204_native_mask_soft_purity'),
        ('Coverage','v552r4204_native_mask_soft_coverage'),
        ('Capture','v538_component_capture_ratio'),
        ('StudentOracle','v538_component_oracle_gain'),
        ('TeacherOracle','v538_action_realizable_teacher_oracle_gain'),
        ('BenefitTargetRate','v541_benefit_target_rate'),
        ('OverflowSoftDice','v552r4205_overflow_soft_dice'),
        ('EditableOnOverflowLeak','v552r4205_editable_on_overflow_leakage'),
        ('Q0PairCosine','v552r4207_q0_pairwise_cosine'),
        ('Q1PairCosine','v552r4207_q1_pairwise_cosine'),
    ]
    for n,k in items:
        x,y=g(c,k),g(t,k)
        print(f'{n:<38} {f(x):>14} {f(y):>14} {f"{y-x:+.6f}":>14}')
    cr = ratio(g(c,'v538_component_oracle_gain'),g(c,'v538_action_realizable_teacher_oracle_gain'))
    tr = ratio(g(t,'v538_component_oracle_gain'),g(t,'v538_action_realizable_teacher_oracle_gain'))
    print(f'{"Retention":<38} {f(cr):>14} {f(tr):>14} {f"{tr-cr:+.6f}":>14}')

    print('\n[FAIRNESS]')
    print('[PASS] identical Base/PVL fingerprint ' + str(c['fp']) if c['fp'] and c['fp']==t['fp'] else '[FAIL] fingerprint mismatch ' + str((c['fp'],t['fp'])))
    if c['v'] and t['v']:
        cm={int(v[0]):v for v in c['v']}; tm={int(v[0]):v for v in t['v']}; common=sorted(set(cm)&set(tm))
        gaps=[abs(cm[e][1]-tm[e][1]) for e in common]
        print('max Base DSC trajectory gap=' + f(max(gaps)) if gaps else 'no common val epochs')

    print('\n[ROOT-CAUSE TEST]')
    collision_delta = g(t,'v552r4207_teacher_best_slot_collision_rate') - g(c,'v552r4207_teacher_best_slot_collision_rate')
    margin_delta = g(t,'v552r4207_teacher_best_slot_margin') - g(c,'v552r4207_teacher_best_slot_margin')
    match_delta = g(t,'v552r4204_native_mask_matching_dice') - g(c,'v552r4204_native_mask_matching_dice')
    capture_delta = g(t,'v538_component_capture_ratio') - g(c,'v538_component_capture_ratio')
    oracle_delta = g(t,'v538_component_oracle_gain') - g(c,'v538_component_oracle_gain')
    retention_delta = tr-cr
    print('Binding evidence expected: Collision↓, Margin↑, Match↑, while Capture/StudentOracle do not trade off.')
    print(f'CollisionDelta={f(collision_delta)} MarginDelta={f(margin_delta)} MatchDelta={f(match_delta)} CaptureDelta={f(capture_delta)} OracleDelta={f(oracle_delta)} RetentionDelta={f(retention_delta)}')
    structural_support = (collision_delta < 0 or margin_delta > 0) and match_delta > 0
    utility_preserved = capture_delta >= -1e-4 and oracle_delta >= -1e-4
    if structural_support and utility_preserved and retention_delta > 0:
        print('[SUPPORTED] Dynamic instance binding improves set identity without sacrificing corrective utility.')
    elif structural_support and not utility_preserved:
        print('[PARTIAL/FAIL] Binding cleanliness improved but corrective utility fell. Do NOT stack another loss; inspect seed recall/seed quality before changing objective.')
    else:
        print('[NOT CONFIRMED] Dynamic visual binding did not repair the predicted binding gap. Re-evaluate R417 seed quality before any CE/M2 modification.')

    ma=g(t,'v552r4204_native_mask_matching_dice'); pu=g(t,'v552r4204_native_mask_soft_purity'); ca=g(t,'v538_component_capture_ratio'); so=g(t,'v538_component_oracle_gain'); retn=tr
    ready=ma>=.20 and pu>=.12 and ca>=.15 and so>=.005 and retn>=.20
    print('\n[M1 READINESS]', 'PASS' if ready else 'FAIL', f'Match={f(ma)} Purity={f(pu)} Capture={f(ca)} SOracle={f(so)} Retention={f(retn)}')
    if ready:
        print('Next isolated step: audit/repair the conditional identity objective while preserving this binding mechanism; only after M1 remains ready should M2 signed utility be changed.')
    else:
        print('NO-GO FORMAL150. Do not tune M2 gates or add CE weights until this binding ablation is causally interpreted.')


if __name__ == '__main__': main()
