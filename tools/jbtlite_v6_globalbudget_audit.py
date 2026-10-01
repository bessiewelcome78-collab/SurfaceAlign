#!/usr/bin/env python3
from __future__ import annotations
import math
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / 'configs' / 'jbtlite' / 'globalbudget'


def cosine_lr(base_lr: float, eta_min: float, total: int, epoch_1based: int) -> float:
    t = max(0, epoch_1based - 1)
    t = min(t, total)
    return eta_min + 0.5 * (base_lr - eta_min) * (1.0 + math.cos(math.pi * t / total))


def matched_weight(epoch: int, base_weight: float = 0.25) -> float:
    if epoch > 20:
        return 0.0
    ref = cosine_lr(3e-4, 1e-4, 20, epoch)
    target = cosine_lr(3e-4, 1e-4, 100, epoch)
    return base_weight * ref / target


def load(name: str):
    p = CFG / name
    if not p.is_file():
        raise AssertionError(f'missing config: {p}')
    return yaml.safe_load(p.read_text())


def check_common(c, epochs):
    t = c['TRAIN']
    assert c['M1']['ENABLED'] is False
    assert t['NUM_EPOCHS'] == epochs
    assert t['SCHEDULER_TOTAL_EPOCHS'] == 100
    assert t['USE_VALIDATION_SELECTION'] is True
    assert t['RBAL_NORMAL_WEIGHT'] == 0.0
    assert t['RBAL_EDGE_GRAD_SCOPE'] == 'all'
    assert c['TEST']['NUM_SAMPLES'] == 30


def main():
    b = load('JBTL6_BUSI_BASE_DIAG40.yaml')
    e = load('JBTL6_BUSI_GLOBAL_EDGE20_DIAG40.yaml')
    m = load('JBTL6_BUSI_GLOBAL_BUDGET20_DIAG40.yaml')
    for c in (b,e,m):
        check_common(c, 40)
    assert b['TRAIN']['RBAL_EDGE_WEIGHT'] == 0.0
    assert e['TRAIN']['RBAL_EDGE_WEIGHT'] == 0.25
    assert e['TRAIN']['RBAL_SCHEDULE_TYPE'] == 'hard_cutoff'
    assert e['TRAIN']['RBAL_FULL_WEIGHT_EPOCHS'] == 20
    assert m['TRAIN']['RBAL_EDGE_WEIGHT'] == 0.25
    assert m['TRAIN']['RBAL_SCHEDULE_TYPE'] == 'matched_budget'
    assert m['TRAIN']['RBAL_BUDGET_ACTIVE_EPOCHS'] == 20
    assert m['TRAIN']['RBAL_BUDGET_REFERENCE_EPOCHS'] == 20
    assert abs(m['TRAIN']['RBAL_BUDGET_ETA_MIN'] - 1e-4) < 1e-12

    for arm in ('BASE','GLOBAL_EDGE20','GLOBAL_BUDGET20'):
        check_common(load(f'JBTL6_BUSI_{arm}_PAPER100.yaml'), 100)
    for ds in ('Kvasir','ISIC','BTMRI'):
        c = load(f'JBTL6_{ds}_GLOBAL_BUDGET20_PAPER100.yaml')
        check_common(c, 100)
        assert c['DATASET']['NAME'] == ds
        assert c['TRAIN']['RBAL_SCHEDULE_TYPE'] == 'matched_budget'

    src = (ROOT / 'train.py').read_text(errors='ignore')
    required = [
        'if schedule == "hard_cutoff"',
        'if schedule == "matched_budget"',
        'eta_ref / eta_target',
        'if rbal_edge_weight > 0.0 and not edge_decoder_only:',
        'loss = loss + rbal_edge_weight * rbal_edge',
    ]
    for token in required:
        assert token in src, f'missing train.py contract token: {token}'

    print('[JBTL6_GLOBALBUDGET_AUDIT_PASS]')
    print('A0 BASE             : EDGE=0')
    print('A1 GLOBAL_EDGE20    : global EDGE=.25 for epochs 1-20, then 0')
    print('A2 GLOBAL_BUDGET20  : global EDGE update budget matched to the old successful T=20 cosine trajectory')
    print('Formal Base schedule: T=100; Val selects checkpoint; Test MC30 once only after the Val gate.')
    print('Matched-budget effective EDGE weights:')
    for e in (1,5,10,15,20,21,40,100):
        print(f'  epoch={e:03d} edge_eff={matched_weight(e):.6f}')
    print('Expected: epoch1=.250000, epoch20≈.089519, epoch21+=0.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
