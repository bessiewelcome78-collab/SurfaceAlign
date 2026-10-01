#!/usr/bin/env python3
from pathlib import Path
import sys, yaml
root=Path(__file__).resolve().parents[1]
errs=[]
for ds in ('Kvasir','ISIC','BTMRI'):
    for arm in ('BASE','GLOBAL_EDGE20'):
        p=root/'configs/jbtlite/repro_mc10'/f'JBTL8_{ds}_{arm}_PAPER100.yaml'
        if not p.exists(): errs.append(f'missing {p}'); continue
        c=yaml.safe_load(p.read_text())
        tr=c['TRAIN']
        exp=0.0 if arm=='BASE' else 0.25
        checks={
            'dataset': c['DATASET']['NAME']==ds,
            'epochs': tr.get('NUM_EPOCHS')==100,
            'schedT': tr.get('SCHEDULER_TOTAL_EPOCHS')==100,
            'strict': tr.get('STRICT_REPRODUCIBILITY') is True,
            'val_mc10': tr.get('VAL_NUM_SAMPLES')==10,
            'val_select': tr.get('USE_VALIDATION_SELECTION') is True,
            'edge': abs(float(tr.get('RBAL_EDGE_WEIGHT',-1))-exp)<1e-12,
            'scope_all': tr.get('RBAL_EDGE_GRAD_SCOPE')=='all',
            'cut20': tr.get('RBAL_SCHEDULE_TYPE')=='hard_cutoff' and tr.get('RBAL_FULL_WEIGHT_EPOCHS')==20 and tr.get('RBAL_DECAY_END_EPOCH')==21,
            'normal_off': float(tr.get('RBAL_NORMAL_WEIGHT',-1))==0.0,
            'test_mc30': c['TEST'].get('NUM_SAMPLES')==30,
            'm1_off': c['M1'].get('ENABLED') is False,
        }
        bad=[k for k,v in checks.items() if not v]
        if bad: errs.append(f'{p.name}: failed {bad}')
if errs:
    print('[JBTL8_1_OTHER3_AUDIT_FAIL]')
    print('\n'.join(errs)); sys.exit(1)
print('[JBTL8_1_OTHER3_AUDIT_PASS]')
print('Locked method: GLOBAL_EDGE20 = edge 0.25 epochs1-20, zero 21-100, grad_scope=all.')
print('Formal protocol: 100ep / T=100 / Val MC10 best checkpoint / Test MC30 once / strict reproducibility.')
print('Other datasets: Kvasir, ISIC, BTMRI; paired BASE + GLOBAL_EDGE20.')
