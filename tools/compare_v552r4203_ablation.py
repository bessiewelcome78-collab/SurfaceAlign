#!/usr/bin/env python3
from __future__ import annotations
import argparse,math,re
from pathlib import Path
PAIR=re.compile(r'([A-Za-z0-9_]+)=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')
VAL=re.compile(r'VAL_NATIVE epoch=(\d+).*?base DSC/NSD=([0-9.]+)/([0-9.]+).*?M2 DSC/NSD=([0-9.]+)/([0-9.]+)')
def parse(p):
 lines=p.read_text(errors='replace').splitlines();rows=[{k:float(v) for k,v in PAIR.findall(x)} for x in lines if 'M1_DIAG:' in x];r=rows[-1] if rows else {}; vals=[tuple(map(float,m.groups())) for x in lines if (m:=VAL.search(x))]
 teacher=r.get('v538_action_realizable_teacher_oracle_gain',math.nan);student=r.get('v538_component_oracle_gain',math.nan)
 return {'loc':r.get('v552r417_location_center_recall',math.nan),'peak':r.get('v552r417_pre_topk_peak_recall',math.nan),'match':r.get('v552r4203_native_mask_matching_dice',r.get('v552r420_native_mask_matching_dice',math.nan)),'purity':r.get('v552r4203_native_mask_soft_purity',r.get('v552r420_native_mask_soft_purity',math.nan)),'coverage':r.get('v552r4203_native_mask_soft_coverage',r.get('v552r420_native_mask_soft_coverage',math.nan)),'capture':r.get('v538_component_capture_ratio',math.nan),'student':student,'retain':student/teacher if math.isfinite(teacher) and abs(teacher)>1e-12 else math.nan,'harm':r.get('v552r44_quality_candidate_audit_harm_rate',math.nan),'best':max((v[3] for v in vals),default=math.nan),'complete':any('TRAIN COMPLETE' in x for x in lines),'fatal':any(q in x for x in lines for q in ('Traceback (most recent call last)','RuntimeError:','CUDA out of memory')),'path':p}
def F(x):return 'n/a' if not math.isfinite(x) else f'{x:.4f}'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stamp',required=True);ap.add_argument('--seed',type=int,default=42);a=ap.parse_args();root=Path('logs')
 specs=[('A0_R4201_PointDynamic',f'V552R4203_A0_R4201_SMOKE20_seed{a.seed}_gpu*_{a.stamp}.log'),('A1_R4203_DenseSet',f'V552R4203_A1_DenseSet_SMOKE20_seed{a.seed}_gpu*_{a.stamp}.log')]
 data={}
 for n,p in specs:
  fs=sorted(root.glob(p));data[n]=parse(fs[-1]) if fs else {}
 print('Variant                    LocR PeakR Match Purity Coverage Capture SOracle Retain Harm BestM2 Complete')
 for n,_ in specs:
  d=data[n];print(f'{n:<26} {F(d.get("loc",math.nan)):>5} {F(d.get("peak",math.nan)):>5} {F(d.get("match",math.nan)):>5} {F(d.get("purity",math.nan)):>6} {F(d.get("coverage",math.nan)):>8} {F(d.get("capture",math.nan)):>7} {F(d.get("student",math.nan)):>7} {F(d.get("retain",math.nan)):>6} {F(d.get("harm",math.nan)):>5} {F(d.get("best",math.nan)):>6} {d.get("complete",False) and not d.get("fatal",False)}')
 a0,a1=(data[n] for n,_ in specs)
 def D(k):
  x=a0.get(k,math.nan);y=a1.get(k,math.nan);return y-x if math.isfinite(x) and math.isfinite(y) else math.nan
 print('\n[Causal delta: remove Dense->TopK->Point->DynamicMask bottleneck]')
 print(f'ΔMatch={F(D("match"))} ΔPurity={F(D("purity"))} ΔCoverage={F(D("coverage"))} ΔCapture={F(D("capture"))} ΔStudentOracle={F(D("student"))} ΔRetention={F(D("retain"))}')
 ready=all([a1.get('complete',False),not a1.get('fatal',True),a1.get('loc',0)>=.90,a1.get('match',0)>=.20,a1.get('purity',0)>=.12,a1.get('capture',0)>=.15,a1.get('student',0)>=.005,a1.get('retain',0)>=.20,a1.get('harm',1)<=.10])
 print('\n[FORMAL READINESS]', 'READY' if ready else 'NOT_READY')
 if not ready:print('Do NOT run FORMAL150. Dense-set architecture removed the point bottleneck structurally; capability must still satisfy the pre-registered mask/capture criteria.')
if __name__=='__main__':main()
