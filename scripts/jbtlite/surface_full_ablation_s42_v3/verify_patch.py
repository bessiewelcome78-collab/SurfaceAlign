#!/usr/bin/env python3
from __future__ import annotations
import sys,yaml,json
from pathlib import Path
project=Path(sys.argv[1]) if len(sys.argv)>1 else Path.cwd(); cfgdir=project/'configs/jbtlite/surface_full_ablation_s42_v3'; anchor=yaml.safe_load((cfgdir/'FULL_STEP_E30.yaml').read_text())
expected={
'BASE':{'TRAIN.RBAL_EDGE_WEIGHT':0.0},'SURFACE_ALWAYS':{'TRAIN.RBAL_SCHEDULE_TYPE':'always_on'},'DECODER_ONLY_STEP_E30':{'TRAIN.RBAL_EDGE_GRAD_SCOPE':'decoder_only'},
'FULL_STEP_E30':{},'SCHED_COSINE':{'TRAIN.RBAL_SCHEDULE_TYPE':'cosine','TRAIN.RBAL_FULL_WEIGHT_EPOCHS':10},'SCHED_WARMUP_E30':{'TRAIN.RBAL_SCHEDULE_TYPE':'warmup'},
'LOSS_BOUNDARY_STEP_E30':{'TRAIN.RBAL_AUX_LOSS_TYPE':'boundary'},'LOSS_HD_STEP_E30':{'TRAIN.RBAL_AUX_LOSS_TYPE':'hausdorff_dt'},'LOSS_ACTIVE_CONTOUR_STEP_E30':{'TRAIN.RBAL_AUX_LOSS_TYPE':'active_contour'},
'RADIUS_R2_STEP_E30':{'TRAIN.RBAL_BOUNDARY_RADIUS_PX':2},'RADIUS_R3_STEP_E30':{'TRAIN.RBAL_BOUNDARY_RADIUS_PX':3},'RADIUS_R5_STEP_E30':{'TRAIN.RBAL_BOUNDARY_RADIUS_PX':5},
'WEIGHT_L010_STEP_E30':{'TRAIN.RBAL_EDGE_WEIGHT':0.10},'WEIGHT_L015_STEP_E30':{'TRAIN.RBAL_EDGE_WEIGHT':0.15},'WEIGHT_L020_STEP_E30':{'TRAIN.RBAL_EDGE_WEIGHT':0.20},'WEIGHT_L025_STEP_E30':{'TRAIN.RBAL_EDGE_WEIGHT':0.25}}

def get(d,path):
    for x in path.split('.'): d=d[x]
    return d
ignore={'M1.RUN_TAG','TRAIN.RUN_TAG'}
def flat(d,p=''):
    o={}
    if isinstance(d,dict):
      for k,v in d.items():o.update(flat(v,f'{p}.{k}' if p else k))
    else:o[p]=d
    return o
fa=flat(anchor)
assert get(anchor,'TRAIN.NUM_EPOCHS')==100 and get(anchor,'TRAIN.BATCH_SIZE')==24 and abs(get(anchor,'TRAIN.LEARNING_RATE')-3e-4)<1e-12
assert get(anchor,'TRAIN.RBAL_EDGE_WEIGHT')==0.05 and get(anchor,'TRAIN.RBAL_BOUNDARY_RADIUS_PX')==1 and get(anchor,'TRAIN.RBAL_SCHEDULE_TYPE')=='hard_cutoff' and get(anchor,'TRAIN.RBAL_FULL_WEIGHT_EPOCHS')==30 and get(anchor,'TRAIN.RBAL_EDGE_GRAD_SCOPE')=='all' and get(anchor,'TRAIN.RBAL_AUX_LOSS_TYPE')=='surface'
for arm,exp in expected.items():
    f=cfgdir/f'{arm}.yaml'; assert f.is_file(),f
    d=yaml.safe_load(f.read_text()); fd=flat(d)
    diffs={k for k in set(fa)|set(fd) if fa.get(k)!=fd.get(k) and k not in ignore}
    allowed=set(exp)
    if diffs!=allowed: raise AssertionError(f'{arm}: unexpected diffs {sorted(diffs)} expected {sorted(allowed)}')
    for k,v in exp.items():
      if get(d,k)!=v: raise AssertionError(f'{arm}: {k}={get(d,k)!r}, expected {v!r}')
print('[PASS] 16 configs share one seed-42 protocol and differ only by declared factors.')
contract=json.loads((cfgdir/'ANCHOR_CONTRACT.json').read_text()); print('[ANCHOR]',contract['final_anchor'])
