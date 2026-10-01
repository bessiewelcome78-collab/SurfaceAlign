#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, hashlib, json
from pathlib import Path
import yaml

VARIANTS={
    'NO_POSTERIOR':'no_posterior_uncertainty',
    'NO_RAY':'no_normal_ray_evidence',
    'DIRECT_SIGNED':'direct_signed',
    'SEG_ONLY':'segmentation_only',
}
DATASETS=('BUSI','BTMRI','ISIC','Kvasir')

def sha(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()

def normalized(cfg):
    c=copy.deepcopy(cfg)
    c.setdefault('M1',{}).pop('SEMLT_UC_FNRT_ABLATION',None)
    if 'M1' in c: c['M1']['RUN_TAG']='__RUN_TAG__'
    if 'TRAIN' in c: c['TRAIN']['RUN_TAG']='__RUN_TAG__'
    return c

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project',required=True)
    a=ap.parse_args(); root=Path(a.project)
    spec=root/'configs/ucfnrt_causal_ablation/UC_FNRT_CAUSAL_ABLATION_SPEC.json'
    data=json.loads(spec.read_text())
    assert data['protocol']=='UC-FNRT_causal_ablation_v1'
    for ds in DATASETS:
        full_path=root/f'configs/{ds}_SEMLT_UC_FNRT_FORMAL100.yaml'
        full=yaml.safe_load(full_path.read_text())
        assert full['M1']['SEMLT_UC_FNRT'] is True
        assert int(full['M1']['GEOTR_TRAIN_POSTERIOR_SAMPLES'])==10
        assert int(full['TEST']['NUM_SAMPLES'])==30
        assert int(full['TRAIN']['NUM_EPOCHS'])==100
        assert int(full['TRAIN']['BATCH_SIZE'])==24
        assert full['TRAIN']['USE_VALIDATION_SELECTION'] is False
        for key,mode in VARIANTS.items():
            p=root/f'configs/ucfnrt_causal_ablation/{ds}_{key}_PAPER100.yaml'
            c=yaml.safe_load(p.read_text())
            got=str(c['M1'].get('SEMLT_UC_FNRT_ABLATION','')).lower()
            if got!=mode: raise SystemExit(f'[FAIL] {p}: ablation={got}, expected={mode}')
            if normalized(c)!=normalized(full):
                # concise top-level diff for debugging
                bad=[k for k in sorted(set(c)|set(full)) if normalized(c).get(k)!=normalized(full).get(k)]
                raise SystemExit(f'[FAIL] {p}: differs from Formal100 beyond RUN_TAG/ablation: {bad}')
            print(f'[PASS] {ds:6s} {key:13s} mode={mode:27s} sha256={sha(p)}')
    print('[PASS] all 16 causal-ablation configs are matched to Formal100')
if __name__=='__main__': main()
