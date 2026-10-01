#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

ALL = [
'BASE','SURFACE_ALWAYS','DECODER_ONLY_STEP_E30','FULL_STEP_E30','SCHED_COSINE','SCHED_WARMUP_E30',
'LOSS_BOUNDARY_STEP_E30','LOSS_HD_STEP_E30','LOSS_ACTIVE_CONTOUR_STEP_E30',
'RADIUS_R2_STEP_E30','RADIUS_R3_STEP_E30','RADIUS_R5_STEP_E30',
'WEIGHT_L010_STEP_E30','WEIGHT_L015_STEP_E30','WEIGHT_L020_STEP_E30','WEIGHT_L025_STEP_E30']
TEST = ALL[:9]

def sha(p: Path): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project',required=True); ap.add_argument('--result-root',default=None); ap.add_argument('--study-id',required=True)
    a=ap.parse_args(); project=Path(a.project); root=Path(a.result_root) if a.result_root else project/'surface_full_ablation_s42_v3_results'; study=root/a.study_id
    runs={}; missing=[]
    for arm in ALL:
        rr=study/'BUSI'/arm/'seed42'; marker=rr/'VAL_ONLY_COMPLETE.txt'; ckf=rr/'LOCKED_CHECKPOINT.txt'; cfg=project/'configs/jbtlite/surface_full_ablation_s42_v3'/f'{arm}.yaml'
        if not marker.is_file() or marker.read_text().strip()!='VAL_LOCKED_NO_TEST_OPENED' or not ckf.is_file() or not cfg.is_file():
            missing.append(arm); continue
        ck=Path(ckf.read_text().strip())
        if not ck.is_file(): missing.append(arm+':checkpoint'); continue
        runs[arm]={'checkpoint':str(ck),'checkpoint_sha256':sha(ck),'config':str(cfg),'config_sha256':sha(cfg),'test_allowed':arm in TEST}
    if missing:
        raise SystemExit('[FAIL] validation not complete for: '+', '.join(missing))
    data={'protocol':'SAFA3_S42','seed':42,'gate':'validation-complete-no-outcome-selection','test_arms':TEST,'validation_only_arms':[x for x in ALL if x not in TEST],'runs':runs}
    p=study/'FROZEN_TEST_MANIFEST.json'; p.write_text(json.dumps(data,indent=2)+'\n')
    print(f'[PASS] frozen manifest: {p}'); print('Test arms: '+', '.join(TEST))
if __name__=='__main__': main()
