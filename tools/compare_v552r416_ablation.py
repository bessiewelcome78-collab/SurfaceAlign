#!/usr/bin/env python3
from __future__ import annotations
import argparse,glob,math,re
from pathlib import Path
PAIR=re.compile(r'([A-Za-z0-9_]+)=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')
VAL=re.compile(r'VAL_NATIVE epoch=(\d+).*?base DSC/NSD=([0-9.]+)/([0-9.]+).*?M2 DSC/NSD=([0-9.]+)/([0-9.]+).*?actionOracle DSC/NSD=([0-9.]+)/([0-9.]+).*?componentOracle DSC/NSD=([0-9.]+)/([0-9.]+)')
NAMES=['A0_R415_Control','A1_UniquePoint','A2_AsymmetricLTRB','A3_FineContext','A4_UniquePointLTRB','A5_FULL']
def num(d,k): return d.get(k,float('nan'))
def fmt(x,n=4): return 'NA' if not math.isfinite(x) else f'{x:.{n}f}'
def parse(path):
 rows=[]; vals=[]
 for line in Path(path).read_text(errors='replace').splitlines():
  if 'M1_DIAG:' in line: rows.append({k:float(v) for k,v in PAIR.findall(line)})
  m=VAL.search(line)
  if m: vals.append(tuple(map(float,m.groups())))
 r=rows[-1] if rows else {}; best=max(vals,key=lambda x:x[3]) if vals else (math.nan,)*9
 teacher=num(r,'v538_action_realizable_teacher_oracle_gain'); student=num(r,'v538_component_oracle_gain'); ret=student/teacher if teacher and math.isfinite(teacher) else math.nan
 return dict(pre=num(r,'v552r416_pre_topk_peak_recall'),id=num(r,'v552r415_identity_match_rate'),legacy=num(r,'v552r416_legacy_topk_unique_fraction'),edge=num(r,'v552r416_edge_offset_mae_px'),box=num(r,'v552r413_proposal_box_iou'),native=num(r,'v552r412_native_box_canonical_dice'),capture=num(r,'v538_component_capture_ratio'),student=student,ret=ret,cgain=num(r,'v552r44_quality_candidate_audit_mean_gain'),bestm2=best[3],bestbase=best[1],complete='TRAIN COMPLETE' in Path(path).read_text(errors='replace'))
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--mode',default='SMOKE20'); ap.add_argument('--stamp',default=''); ap.add_argument('--log-dir',default='logs'); a=ap.parse_args()
 print('Experiment                PreTopK IdCov LegacyU EdgeMAE PropIoU NBoxD Capture SOracle Retain CandGain BestM2 Complete')
 data={}
 for n in NAMES:
  pat=f'{a.log_dir}/V552R416_ABL_{n}_{a.mode}_seed*_gpu*_{a.stamp or "*"}.log'; fs=sorted(glob.glob(pat),key=lambda p:Path(p).stat().st_mtime,reverse=True)
  if not fs: print(f'{n:<25} MISSING'); continue
  d=parse(fs[0]); data[n]=d
  print(f"{n:<25} {fmt(d['pre'],3):>7} {fmt(d['id'],3):>5} {fmt(d['legacy'],3):>7} {fmt(d['edge'],2):>7} {fmt(d['box'],3):>7} {fmt(d['native'],3):>5} {fmt(d['capture'],3):>7} {fmt(d['student'],4):>7} {fmt(d['ret'],3):>6} {fmt(d['cgain'],5):>8} {fmt(d['bestm2'],4):>6} {d['complete']}")
 if 'A0_R415_Control' in data:
  b=data['A0_R415_Control']; print('\n[Causal deltas vs A0]')
  for n in NAMES[1:]:
   if n not in data: continue
   d=data[n]; print(f"{n:<25} ΔIdCov={d['id']-b['id']:+.3f} ΔBoxIoU={d['box']-b['box']:+.3f} ΔStudentOracle={d['student']-b['student']:+.4f} ΔBestM2={d['bestm2']-b['bestm2']:+.4f}")
 print('\n[Readiness for FORMAL150]')
 ready=[]
 for n,d in data.items():
  ok=(d['complete'] and math.isfinite(d['pre']) and d['pre']>=0.80 and math.isfinite(d['id']) and d['id']>=0.50 and math.isfinite(d['box']) and d['box']>=0.30 and math.isfinite(d['native']) and d['native']>=0.40 and math.isfinite(d['student']) and d['student']>=0.005 and math.isfinite(d['ret']) and d['ret']>=0.20 and math.isfinite(d['cgain']) and d['cgain']>0)
  print(f"{n:<25} {'READY' if ok else 'NOT_READY'}")
  if ok: ready.append((d['bestm2'],d['student'],d['box'],n))
 if ready:
  ready.sort(reverse=True)
  print(f"[RECOMMEND] Freeze structure using validation/root metrics: {ready[0][3]} (highest BestM2 among root-ready variants). Do not use Test to re-select.")
 else:
  print('[NO FORMAL WINNER] No variant closes the current M1 root chain. Do NOT start FORMAL150 merely because one metric or the gate looks better.')
 print('\n[Decision rule] A1-A0 isolates TopK duplication; A2-A0 isolates asymmetric geometry; A3-A0 isolates context resolution; A4 tests Unique+LTRB interaction; A5-A4 tests fine context after geometry is fixed. Prefer improvements in IdCov + ProposalIoU + StudentOracle together.')
if __name__=='__main__': main()
