#!/usr/bin/env python3
from __future__ import annotations
import argparse,math,re
from pathlib import Path
PAIR=re.compile(r'([A-Za-z0-9_]+)=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)');EP=re.compile(r'EPOCH:\s*(\d+)');FP=re.compile(r'\[V552R4204_BASE_FINGERPRINT\]\s+sha256=([0-9a-f]+)');VAL=re.compile(r'VAL_NATIVE epoch=(\d+).*?base DSC/NSD=([0-9.]+)/([0-9.]+).*?M2 DSC/NSD=([0-9.]+)/([0-9.]+).*?actionOracle DSC/NSD=([0-9.]+)/([0-9.]+).*?componentOracle DSC/NSD=([0-9.]+)/([0-9.]+)')
def parse(p):
 ep=None;rows=[];vals=[];lines=Path(p).read_text(errors='replace').splitlines()
 for line in lines:
  m=EP.search(line)
  if m:ep=int(m.group(1))
  if 'M1_DIAG:' in line:rows.append((ep,{k:float(v) for k,v in PAIR.findall(line)}))
  m=VAL.search(line)
  if m:vals.append(tuple(map(float,m.groups())))
 fps=[FP.search(x).group(1) for x in lines if FP.search(x)];return {'p':Path(p),'r':rows,'v':vals,'fp':fps[-1] if fps else None}
def last(x):return x['r'][-1][1]
def g(x,k):return last(x).get(k,float('nan'))
def f(x,n=6):return 'n/a' if not math.isfinite(x) else f'{x:.{n}f}'
def ratio(a,b):return a/b if math.isfinite(a) and math.isfinite(b) and abs(b)>1e-12 else float('nan')
def one(pat):
 xs=sorted(Path('.').glob(pat),key=lambda p:p.stat().st_mtime,reverse=True)
 if not xs:raise SystemExit('[FAIL] missing '+pat)
 return xs[0]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stamp',required=True);ap.add_argument('--seed',type=int,default=42);a=ap.parse_args()
 c=parse(one(f'logs/V552R4206_A0_R4205_Control_SMOKE20_seed{a.seed}_gpu*_{a.stamp}.log'));t=parse(one(f'logs/V552R4206_A1_ConditionalIdentityCE_SMOKE20_seed{a.seed}_gpu*_{a.stamp}.log'))
 print('='*185);print('V552-R4.20.6 CAUSAL ABLATION | R4.20.5 CONTROL vs RESIDUAL-CONDITIONAL K+1 CE');print('='*185)
 print('Metric                           R4205 control      R4206 CE          Delta')
 items=[('NativeMatch','v552r4204_native_mask_matching_dice'),('Purity','v552r4204_native_mask_soft_purity'),('Capture','v538_component_capture_ratio'),('StudentOracle','v538_component_oracle_gain'),('OccupancyDice','v552r4204_occupancy_soft_dice'),('OverflowSoftDice','v552r4205_overflow_soft_dice'),('EditableOnOverflowLeak','v552r4205_editable_on_overflow_leakage')]
 for n,k in items:
  x,y=g(c,k),g(t,k);print(f'{n:<31} {f(x):>14} {f(y):>14} {f(y-x):>+14}')
 x=ratio(g(c,'v538_component_oracle_gain'),g(c,'v538_action_realizable_teacher_oracle_gain'));y=ratio(g(t,'v538_component_oracle_gain'),g(t,'v538_action_realizable_teacher_oracle_gain'));print(f'{"Retention":<31} {f(x):>14} {f(y):>14} {f(y-x):>+14}')
 print('\n[R4206 CONDITIONAL TARGET]')
 for k in ['v552r4206_conditional_identity_loss','v552r4206_shape_dice_loss','v552r4206_conditional_accuracy','v552r4206_editable_accuracy','v552r4206_overflow_target_residual_fraction','v552r4206_overflow_predicted_residual_fraction','v552r4206_overflow_recall','v552r4206_overflow_precision','v552r4206_true_class_probability','v552r4206_supervised_residual_fraction','v552r4206_target_decomposition_error','v552r4206_teacher_overlap_rate']:
  print(f'{k:<56} {f(g(t,k))}')
 print('\n[FAIRNESS]')
 print('[PASS] identical Base/PVL fingerprint '+str(c['fp']) if c['fp'] and c['fp']==t['fp'] else '[FAIL] fingerprint mismatch '+str((c['fp'],t['fp'])))
 if c['v'] and t['v']:
  cm={int(v[0]):v for v in c['v']};tm={int(v[0]):v for v in t['v']};common=sorted(set(cm)&set(tm));gaps=[abs(cm[e][1]-tm[e][1]) for e in common];print('max Base DSC trajectory gap='+f(max(gaps)) if gaps else 'no common val epochs')
 print('\n[DECISION]')
 ma=g(t,'v552r4204_native_mask_matching_dice');pu=g(t,'v552r4204_native_mask_soft_purity');ca=g(t,'v538_component_capture_ratio');so=g(t,'v538_component_oracle_gain');retn=ratio(so,g(t,'v538_action_realizable_teacher_oracle_gain'));ready=ma>=.20 and pu>=.12 and ca>=.15 and so>=.005 and retn>=.20
 print(f'R4206 M1 readiness={"PASS" if ready else "FAIL"} Match={f(ma)} Purity={f(pu)} Capture={f(ca)} SOracle={f(so)} Retention={f(retn)}')
 if ready:print('Next: isolate M2 signed-utility/sign-objective failure. Do not alter M1 again unless a separate diagnostic fails.')
 elif g(t,'v552r4206_overflow_recall')>g(c,'v552r4205_overflow_soft_coverage') and retn>ratio(g(c,'v538_component_oracle_gain'),g(c,'v538_action_realizable_teacher_oracle_gain')):print('Conditional CE is causally moving overflow/retention correctly; identity capacity remains the next M1 question only if Match still fails.')
 else:print('Conditional CE did not sufficiently repair overflow/retention; next isolated ablation should be a learnable per-pixel overflow score, not positional geometry or gate tuning.')
if __name__=='__main__':main()
