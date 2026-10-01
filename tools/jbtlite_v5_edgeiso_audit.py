#!/usr/bin/env python3
from pathlib import Path
import math
import yaml

ROOT = Path(__file__).resolve().parents[1]
TRAIN = (ROOT / 'train.py').read_text()
LOSS = (ROOT / 'utils/jbtl_rbal_loss.py').read_text()

for marker in (
    'RBAL_EDGE_GRAD_SCOPE',
    '[JBTL5_EDGEISO]',
    'torch.autograd.grad(',
    'compute_edge_alignment_loss',
    '_jbtl_edge_decoder_only_enabled',
    '_jbtl_decoder_params',
):
    assert marker in TRAIN, f'missing train marker: {marker}'

assert 'def compute_edge_alignment_loss(' in LOSS
assert 'def compute_normal_margin_loss(' in LOSS

CFGDIR = ROOT / 'configs/jbtlite/edgeiso'


def load(name):
    path = CFGDIR / name
    assert path.is_file(), f'missing config: {path}'
    return path, yaml.safe_load(path.read_text())


def common(path, cfg, epochs):
    t = cfg['TRAIN']
    assert int(t['NUM_EPOCHS']) == epochs, path
    assert int(t['SCHEDULER_TOTAL_EPOCHS']) == 100, path
    assert int(t['BATCH_SIZE']) == 24, path
    assert abs(float(t['LEARNING_RATE']) - 3e-4) < 1e-12, path
    assert str(t['OPTIMIZER']).lower() == 'adam', path
    assert bool(t['USE_VALIDATION_SELECTION']) is True, path
    assert str(t['VAL_SELECTION_METRIC']).lower() == 'native_fusion_dice', path
    assert str(t['VAL_TIEBREAK_METRIC']).lower() == 'native_fusion_nsd', path
    assert int(cfg['TEST']['NUM_SAMPLES']) == 30, path
    assert bool(cfg['M1']['ENABLED']) is False, path
    assert float(t['RBAL_NORMAL_WEIGHT']) == 0.0, path
    assert bool(t['RBAL_SCHEDULE_ENABLED']) is True, path
    assert str(t['RBAL_SCHEDULE_TYPE']).lower() == 'cosine', path
    assert int(t['RBAL_FULL_WEIGHT_EPOCHS']) == 20, path


def assert_arm(path, cfg, arm):
    t = cfg['TRAIN']
    if arm == 'BASE':
        assert float(t['RBAL_EDGE_WEIGHT']) == 0.0, path
        # decay is irrelevant when EDGE=0; keep it explicit for protocol traceability.
        assert str(t['RBAL_EDGE_GRAD_SCOPE']).lower() == 'all', path
    elif arm == 'EDGEISO':
        assert float(t['RBAL_EDGE_WEIGHT']) == 0.25, path
        assert str(t['RBAL_EDGE_GRAD_SCOPE']).lower() == 'decoder_only', path
        # Control arm: semantic gradient shield, but slower historical release.
        assert int(t['RBAL_DECAY_END_EPOCH']) == 80, path
    elif arm == 'FULL':
        assert float(t['RBAL_EDGE_WEIGHT']) == 0.25, path
        assert str(t['RBAL_EDGE_GRAD_SCOPE']).lower() == 'decoder_only', path
        # Proposed final schedule: full 1-20 -> cosine release -> zero from 50.
        assert int(t['RBAL_DECAY_END_EPOCH']) == 50, path
    else:
        raise AssertionError(arm)


for arm in ('BASE', 'EDGEISO', 'FULL'):
    p, c = load(f'JBTL5_BUSI_{arm}_DIAG40.yaml')
    common(p, c, 40)
    assert_arm(p, c, arm)

    p, c = load(f'JBTL5_BUSI_{arm}_PAPER100.yaml')
    common(p, c, 100)
    assert_arm(p, c, arm)

for dataset in ('Kvasir', 'ISIC', 'BTMRI'):
    p, c = load(f'JBTL5_{dataset}_FULL_PAPER100.yaml')
    common(p, c, 100)
    assert_arm(p, c, 'FULL')


def scale(epoch, hold=20, end=50):
    if epoch <= hold:
        return 1.0
    if epoch >= end:
        return 0.0
    q = (epoch - hold) / float(end - hold)
    return 0.5 * (1.0 + math.cos(math.pi * q))

print('[JBTL5_R50_AUDIT_PASS]')
print('Protocol: DIAG40 uses scheduler horizon 100; Validation selects checkpoint; Test remains unopened.')
print('Ablation contract:')
print('  BASE    : no EDGE')
print('  EDGEISO : EDGE=.25 + decoder-only gradient routing + release@80 control')
print('  FULL    : EDGE=.25 + decoder-only gradient routing + early release@50')
print('Final FULL schedule:')
for e in (1, 20, 30, 40, 50, 100):
    s = scale(e)
    print(f'  epoch={e:03d} scale={s:.6f} edge_eff={0.25*s:.6f}')
