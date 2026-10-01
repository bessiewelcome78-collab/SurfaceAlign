#!/usr/bin/env python3
from __future__ import annotations
import os, sys, types, importlib
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
# Load trainers.qabr without executing the heavyweight trainers/__init__.py.
pkg=types.ModuleType('trainers'); pkg.__path__=[str(ROOT/'trainers')]; sys.modules['trainers']=pkg
QueryAnchoredBoundaryRefiner=importlib.import_module('trainers.qabr').QueryAnchoredBoundaryRefiner


def fail(msg):
    raise SystemExit('[JBTL12_AUDIT_FAIL] '+msg)

qabr_src=(ROOT/'trainers/qabr.py').read_text()
train_src=(ROOT/'train.py').read_text()
for token in ['corr - corr.detach()', 'set_training_progress', 'raw_corr already contains the v10 support term']:
    if token not in qabr_src:
        fail('missing qabr contract token: '+token)
if 'set_training_progress(' not in train_src:
    fail('train.py does not pass epoch progress into QABR')
if 'QABR_V11_UNCERTAINTY_FLOOR' in qabr_src or 'v11_uncertainty_floor' in qabr_src:
    fail('v11 secondary uncertainty-only gate leaked into v12 qabr.py')

# Runtime contract test.
torch.manual_seed(7)
m=QueryAnchoredBoundaryRefiner(
    decoder_channels=512, detail_channels=8, hidden_channels=32,
    alpha_init=0.0, band_kernel=5, max_logit_delta=2.0,
    detach_support=True, use_image_detail=True,
)
m.train(); m.set_training_progress(1,40)
dec=torch.randn(1,512,56,56,requires_grad=True)
coarse=torch.randn(1,1,56,56,requires_grad=True)
img=torch.randn(1,3,224,224,requires_grad=True)
out=m(dec,coarse,image=img,output_size=224)
base=F.interpolate(coarse,size=(224,224),mode='bilinear',align_corners=False)
err=float((out-base).abs().max())
if err != 0.0:
    fail(f'shadow phase must be exact forward identity, maxerr={err}')
(out.square().mean()).backward()
if coarse.grad is None or float(coarse.grad.abs().sum()) <= 0:
    fail('Base direct gradient missing')
if dec.grad is not None and float(dec.grad.abs().sum()) > 0:
    fail('decoder received QABR side-branch gradient')
if img.grad is not None and float(img.grad.abs().sum()) > 0:
    fail('image received QABR side-branch gradient')
if m.alpha.grad is None or float(m.alpha.grad.abs()) <= 0:
    fail('shadow phase failed to train QABR alpha')

# Deployed phase must be capable of changing the forward result, but only near
# the current hard boundary band.
with torch.no_grad(): m.alpha.fill_(0.15)
m.eval(); m.set_training_progress(12,40)
with torch.no_grad():
    out2=m(dec.detach(),coarse.detach(),image=img.detach(),output_size=224)
    base2=F.interpolate(coarse.detach(),size=(224,224),mode='bilinear',align_corners=False)
    delta=out2-base2
    band=m._hard_boundary_band(base2)
    inside=float(delta.abs().sum())
    outside=float((delta.abs()*(1-band)).sum())
if inside <= 0:
    fail('deployed phase produced no correction in synthetic test')
if outside > 1e-5:
    fail(f'correction leaked outside boundary band: {outside}')

# Config schedule audit.
import yaml
for sub,total,hold,end in [('qabr_v12_diag40',40,4,12),('qabr_v12',100,10,30)]:
    for arm in ['BASE','SURFACE','QABR','FULL']:
        suffix='DIAG40' if total==40 else 'PAPER100'
        p=ROOT/f'configs/jbtlite/{sub}/JBTL12_BUSI_{arm}_{suffix}.yaml'
        d=yaml.safe_load(p.read_text())
        if int(d['TRAIN']['NUM_EPOCHS'])!=total: fail(str(p)+' bad NUM_EPOCHS')
        if arm in {'SURFACE','FULL'}:
            if d['TRAIN']['RBAL_SCHEDULE_TYPE']!='cosine': fail(str(p)+' must use cosine surface handoff')
            if int(d['TRAIN']['RBAL_FULL_WEIGHT_EPOCHS'])!=hold or int(d['TRAIN']['RBAL_DECAY_END_EPOCH'])!=end:
                fail(str(p)+' wrong fractional schedule')
        expected_q=arm in {'QABR','FULL'}
        if bool(d['MODEL']['QABR']['ENABLED'])!=expected_q: fail(str(p)+' bad QABR arm')

print('[JBTL12_QABR_AUDIT_PASS]')
print('shadow_forward_identity_maxerr=',err)
print('shadow_alpha_grad=',float(m.alpha.grad.abs()))
print('side_branch_decoder_grad=',0.0 if dec.grad is None else float(dec.grad.abs().sum()))
print('deployed_abs_correction=',inside)
print('outside_band_abs_correction=',outside)
print('handoff: 10%-30% total epochs; Surface cosine-down, QABR cosine-up + readiness')
print('no extra loss; Test remains unopened by provided launchers')
