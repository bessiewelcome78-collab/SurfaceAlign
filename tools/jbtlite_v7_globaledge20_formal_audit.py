#!/usr/bin/env python3
from pathlib import Path
import sys, yaml
root=Path(__file__).resolve().parents[1]
cfgdir=root/'configs/jbtlite/globaledge20_formal'
errors=[]

def read(name):
    p=cfgdir/name
    if not p.exists():
        errors.append(f'missing {p}')
        return None
    with p.open('r',encoding='utf-8') as f: return yaml.safe_load(f)

def check_common(cfg,name):
    if cfg is None: return
    tr=cfg['TRAIN']; te=cfg['TEST']; m1=cfg['M1']
    expected={
        'NUM_EPOCHS':100,'SCHEDULER_TOTAL_EPOCHS':100,'BATCH_SIZE':24,
        'LEARNING_RATE':0.0003,'USE_VALIDATION_SELECTION':True,
        'VAL_SELECTION_METRIC':'native_fusion_dice','VAL_TIEBREAK_METRIC':'native_fusion_nsd',
    }
    for k,v in expected.items():
        if tr.get(k)!=v: errors.append(f'{name}: {k}={tr.get(k)!r}, expected {v!r}')
    if te.get('NUM_SAMPLES')!=30: errors.append(f'{name}: TEST.NUM_SAMPLES !=30')
    if m1.get('ENABLED') is not False: errors.append(f'{name}: M1.ENABLED must be false')
    if float(tr.get('RBAL_NORMAL_WEIGHT',999))!=0.0: errors.append(f'{name}: NORMAL must be 0')

base=read('JBTL7_BUSI_BASE_PAPER100.yaml'); check_common(base,'BASE')
if base:
    t=base['TRAIN']
    if float(t.get('RBAL_EDGE_WEIGHT',999))!=0: errors.append('BASE: EDGE must be 0')

for ds in ['BUSI','Kvasir','ISIC','BTMRI']:
    name=f'JBTL7_{ds}_FULL_PAPER100.yaml'; cfg=read(name); check_common(cfg,name)
    if cfg:
        t=cfg['TRAIN']
        checks={
            'RBAL_EDGE_WEIGHT':0.25,'RBAL_SCHEDULE_ENABLED':True,
            'RBAL_SCHEDULE_TYPE':'hard_cutoff','RBAL_FULL_WEIGHT_EPOCHS':20,
            'RBAL_DECAY_END_EPOCH':21,'RBAL_EDGE_GRAD_SCOPE':'all',
        }
        for k,v in checks.items():
            if t.get(k)!=v: errors.append(f'{name}: {k}={t.get(k)!r}, expected {v!r}')

dec=read('JBTL7_BUSI_DECODER_EDGE20_PAPER100.yaml'); check_common(dec,'DECODER_EDGE20')
if dec:
    t=dec['TRAIN']
    if t.get('RBAL_EDGE_GRAD_SCOPE')!='decoder_only': errors.append('DECODER_EDGE20: grad scope must be decoder_only')
    if t.get('RBAL_SCHEDULE_TYPE')!='hard_cutoff' or t.get('RBAL_FULL_WEIGHT_EPOCHS')!=20 or float(t.get('RBAL_EDGE_WEIGHT',0))!=0.25:
        errors.append('DECODER_EDGE20: must match FULL except gradient scope')

if errors:
    print('[JBTL7_FORMAL_AUDIT_FAIL]')
    for e in errors: print(' -',e)
    sys.exit(2)
print('[JBTL7_FORMAL_AUDIT_PASS]')
print('MAIN FULL = global Surface Alignment, EDGE=.25 for epochs 1-20, then exactly 0.')
print('Gradient scope = all trainable PVL adapters + decoder; M1/NORMAL/BAND/RING inactive.')
print('Formal = 100 epochs / scheduler T=100 / Val-best / Test MC30 exactly once.')
print('BUSI causal ablation: BASE vs DECODER_EDGE20 vs FULL(global EDGE20).')
