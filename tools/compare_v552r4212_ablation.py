#!/usr/bin/env python3
from __future__ import annotations
import argparse,math,re
from pathlib import Path
PAIR=re.compile(r'([A-Za-z0-9_]+)=([-+0-9.eE]+)')
EPOCH_RE=re.compile(r'EPOCH:\s*(\d+)')
VAL_RE=re.compile(r'VAL_NATIVE epoch=(\d+).*?base DSC/NSD=([0-9.]+)/([0-9.]+).*?(?:M1Native DSC/NSD=([0-9.]+)/([0-9.]+).*?)?M2 DSC/NSD=([0-9.]+)/([0-9.]+).*?componentOracle DSC/NSD=([0-9.]+)/([0-9.]+)')
BEST_RE=re.compile(r'best_m1_epoch=(\d+)\s+best_m1_native_dice=([0-9.]+)')
def g(d,k): return float(d.get(k,float('nan')))
def f(x,n=5): return 'n/a' if not math.isfinite(x) else f'{x:.{n}f}'
def parse(p):
 lines=Path(p).read_text(errors='replace').splitlines();rows=[];vals=[];best=[];ep=0
 for line in lines:
  m=EPOCH_RE.search(line)
  if m: ep=int(m.group(1))
  if 'M1_DIAG:' in line: rows.append((ep,{k:float(v) for k,v in PAIR.findall(line)}))
  m=VAL_RE.search(line)
  if m:
   z=m.groups(); vals.append((int(z[0]),)+tuple(float(x) if x is not None else float('nan') for x in z[1:]))
  m=BEST_RE.search(line)
  if m: best.append((int(m.group(1)),float(m.group(2))))
 if not rows: raise RuntimeError(f'no M1_DIAG: {p}')
 d=rows[-1]; v=vals[-1] if vals else (float('nan'),)*9
 return {'path':str(p),'d':d[1],'epoch':d[0],'val':v,'best':best[-1] if best else (None,float('nan')),'complete':any('TRAIN COMPLETE' in x for x in lines),'fatal':any('Traceback (most recent call last)' in x for x in lines)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('logs',nargs='+');a=ap.parse_args()
 rows=[]
 for p in a.logs:
  r=parse(p);d=r['d']; ve,bd,bn,m1d,m1n,m2d,m2n,cod,con=r['val'];tea=g(d,'v538_action_realizable_teacher_oracle_gain');stu=g(d,'v538_component_oracle_gain')
  stage=int(round(sum(g(d,k)>=.5 for k in ['v552r4212_independent_candidate_set_enabled','v552r4212_visual_seed_identity_disabled','v552r4212_candidate_alignment_enabled','v552r4212_direct_delta_utility_enabled','v552r4212_zero_stop_one_step_enabled'])))
  rows.append((stage,Path(p).name,r,bd,m1d,m2d,cod,stu,tea))
 rows.sort(key=lambda x:x[0])
 print('='*190);print('V552-R4.21.2 FORMAL100 CAUSAL ABLATION COMPARISON');print('='*190)
 print('Stg Run                  Ep Done Base     M1Native M1Gain    M2       M2Gain    CompOra  StudOra  Retain  ExistT ExistP DepN  U_MAE   USign TStep PStep TGain   PGain   BestM1')
 for st,name,r,bd,m1d,m2d,cod,stu,tea in rows:
  d=r['d'];ret=stu/tea if math.isfinite(tea) and abs(tea)>1e-12 else float('nan')
  print(f'E{st:<2} {name[:20]:<20} {r["epoch"]:3d} {int(r["complete"]):4d} {f(bd):>8} {f(m1d):>8} {f(m1d-bd):>9} {f(m2d):>8} {f(m2d-bd):>9} {f(cod):>8} {f(stu):>8} {f(ret,3):>6} {f(g(d,"v552r4212_presence_target_count"),2):>6} {f(g(d,"v552r4212_parent_existence_count"),2):>6} {f(g(d,"v552r4212_deployment_candidate_count"),2):>5} {f(g(d,"v552r4212_direct_delta_utility_mae")):>7} {f(g(d,"v552r4212_direct_delta_sign_accuracy"),3):>5} {f(g(d,"v552_composer_teacher_step_count"),3):>5} {f(g(d,"v552_composer_predicted_step_count"),3):>5} {f(g(d,"v552_composer_teacher_total_gain")):>7} {f(g(d,"v552_composer_predicted_total_gain")):>7} {f(r["best"][1]):>8}')
 print('\nCausal read: compare only adjacent stages. E1 should raise candidate coverage/oracle; E2 tests seed removal; E3 tests candidate-target alignment; E4 tests physical utility; E5 tests Stop=0 realization.')
 print('A stage is not a success merely because its own diagnostic improves: it must preserve upstream candidate oracle and improve the downstream quantity it owns.')
if __name__=='__main__':main()
