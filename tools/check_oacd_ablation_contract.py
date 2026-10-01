#!/usr/bin/env python3
from pathlib import Path
import sys, yaml

project=Path(sys.argv[1] if len(sys.argv)>1 else '.').resolve()
base=project/'configs/ucfnrt_oacd_busi_minabl100'
arms={
'A000_JOINT_CONTROL':(False,False,False),
'A011_NO_ALIGN':(False,True,True),
'A101_NO_CDF':(True,False,True),
'A110_NO_REACH':(True,True,False),
'FULL':(True,True,True),
}
keys=['SEMLT_UC_OPERATOR_ALIGNED_CANDIDATES','SEMLT_UC_ORDERED_CDF_LOSS','SEMLT_UC_REACHABLE_MATCH_ONLY']
loaded={}
for arm,vals in arms.items():
    p=base/f'{arm}.yaml'
    assert p.is_file(), p
    d=yaml.safe_load(p.read_text())
    loaded[arm]=d
    m=d['M1']; tr=d['TRAIN']; te=d['TEST']; model=d['MODEL']
    got=tuple(bool(m[k]) for k in keys)
    assert got==vals,(arm,got,vals)

    # Exact routing/protocol invariants required by the repository's current train.py.
    assert m['PROTOCOL']=='semlt_autozero'
    assert m['CANDIDATE_MODE']=='semlt_autozero_transport'
    assert m['LOSS_MODE']=='semlt_autozero'
    assert m['M1_LOSS_VERSION']=='semlt_uc_fnrt'
    assert m['TRAIN_MODE']=='e2e'
    assert m['INFERENCE_MODE']=='unified_action_cf_selection'
    assert m['INIT_CHECKPOINT']==''
    assert m['SEMLT_LST_V3_MAIN'] is True
    assert m['V469_FREEZE_BASE'] is False
    assert m['SEMLT_AUTOZERO_DETACH_CONDITIONERS'] is True
    assert m['GEOTOPO_REFINEMENT_ENABLED'] is True
    assert str(m['GEOTOPO_MODE']).lower()=='geometry'
    assert m['SEMLT_UC_FNRT'] is True
    assert m['SEMLT_UC_FNRT_ABLATION']=='full'
    assert m['M1_TRAIN_NUM_SAMPLES']==1
    assert m['GEOTR_TRAIN_POSTERIOR_SAMPLES']==10
    assert m['MHCS_RNG_ISOLATION'] is True
    assert m['SEMLT_GRADIENT_AUDIT'] is True
    assert m['SEMLT_LOCAL_RADIUS_PX']==8
    assert m['SEMLT_UC_OFFSET_DISTRIBUTION'] is True
    assert m['SEMLT_UC_OFFSET_HEAD_REGISTERED'] is True
    assert m['SEMLT_UC_HRCV'] is True
    assert m['SEMLT_UC_HRCV_REGISTERED'] is True
    assert m['SEMLT_UC_HRCV_CANDIDATE_CONDITIONED'] is True

    # Paper100 fairness budget.
    assert tr['NUM_EPOCHS']==100 and tr['SCHEDULER_TOTAL_EPOCHS']==100
    assert tr['BATCH_SIZE']==24 and tr['GRAD_ACCUMULATION_STEPS']==1
    assert abs(float(tr['LEARNING_RATE'])-3e-4)<1e-12 and tr['OPTIMIZER']=='adam'
    assert float(tr['WEIGHT_DECAY'])==0.0
    assert float(tr['CE_WEIGHT'])==0.5 and float(tr['DICE_WEIGHT'])==0.5 and float(tr['CLIP_WEIGHT'])==0.1
    assert tr['WARMUP_EPOCHS']==0 and float(tr['MIN_LR_RATIO'])==0.0
    assert tr['USE_VALIDATION_SELECTION'] is False
    assert te['USE_LATEST'] is True and te['NUM_SAMPLES']==30
    assert float(model['BETA'])==2.35 and model['ADAPTER_DIM']==256 and model['NUM_UPSCALE']==2
    assert float(model['TEMPERATURE'])==0.2

# Only the three causal interventions and names may differ across arms.
ref=loaded['FULL']
def flatten(x,p=''):
    out={}
    if isinstance(x,dict):
        for k,v in x.items(): out.update(flatten(v,p+'.'+k if p else k))
    else: out[p]=x
    return out
rf=flatten(ref)
allowed=set('M1.'+k for k in keys)|{'M1.RUN_TAG','TRAIN.RUN_TAG'}
for arm,d in loaded.items():
    f=flatten(d)
    assert set(f)==set(rf), (arm, 'key-set differs')
    diff={k for k in rf if rf[k]!=f[k]}
    assert diff<=allowed,(arm,sorted(diff-allowed))

# Source integration: search all relevant UC files jointly, not brittle file-by-file placement.
source_paths=[
    project/'train.py',
    project/'trainers/medclipseg_unimedclip.py',
    project/'utils/semlt_autozero_transport.py',
    project/'utils/semlt_autozero_loss.py',
]
text='\n'.join(p.read_text(errors='ignore') for p in source_paths if p.is_file())
for k in keys:
    assert k in text, f'runtime source tree does not reference {k}'
assert 'GEOTOPO_REFINEMENT_ENABLED' in text, 'runtime source tree lacks GEOTOPO protocol marker'
print('[PASS] OACD rigorous static config + source contract')
