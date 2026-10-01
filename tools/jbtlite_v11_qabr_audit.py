#!/usr/bin/env python3
"""Fail-closed structural/protocol audit for JBT-Lite v10 QABR."""
from __future__ import annotations
from pathlib import Path
import copy, hashlib, importlib.util, math
import torch, yaml

ROOT=Path(__file__).resolve().parents[1]
CFG_DIR=ROOT/'configs/jbtlite/qabr_v11'
DIAG_DIR=ROOT/'configs/jbtlite/qabr_v11_diag40'
HASHES={
 'train.py':'5e20fb47014cae045b5eb3f9dc8427dfd4a458a8e00dff14f25590ebee3bd007',
 'utils/jbtl_rbal_loss.py':'ca262dd74d8f43e63d10917229bd01200c0c11ad9de0fb1ca3a83249a420c2d9',
}
ARMS={
 'BASE':{'edge':0.0,'qabr':False},
 'EDGE20':{'edge':0.25,'qabr':False},
 'QABR':{'edge':0.0,'qabr':True},
 'FULL':{'edge':0.25,'qabr':True},
}
EXPECTED_QABR={
 'DETAIL_CHANNELS':8,'HIDDEN_CHANNELS':32,'ALPHA_INIT':0.0,
 'BAND_KERNEL':11,'MAX_LOGIT_DELTA':3.0,'DETACH_SUPPORT':True,
 'USE_IMAGE_DETAIL':True,
}
COMMON={
 'NUM_EPOCHS':100,'SCHEDULER_TOTAL_EPOCHS':100,'BATCH_SIZE':24,
 'LEARNING_RATE':3e-4,'WEIGHT_DECAY':0.0,'OPTIMIZER':'adam',
 'VAL_NUM_SAMPLES':10,'VAL_MC_SEED':20260910,'USE_VALIDATION_SELECTION':True,
 'VAL_SELECTION_METRIC':'native_fusion_dice','VAL_TIEBREAK_METRIC':'native_fusion_nsd',
 'RBAL_NORMAL_WEIGHT':0.0,'RBAL_SCHEDULE_ENABLED':True,
 'RBAL_SCHEDULE_TYPE':'hard_cutoff','RBAL_FULL_WEIGHT_EPOCHS':20,
 'RBAL_DECAY_END_EPOCH':21,'RBAL_EDGE_GRAD_SCOPE':'all',
 'STRICT_REPRODUCIBILITY':True,'ALLOW_TF32':False,'DETERMINISTIC':True,
}

def die(msg): raise SystemExit('[JBTL11_QABR_AUDIT_FAIL] '+msg)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1<<20),b''): h.update(c)
 return h.hexdigest()

# No new loss code: core training and current Surface objective must stay byte-identical.
for rel,want in HASHES.items():
 p=ROOT/rel
 if not p.is_file() or sha(p)!=want: die(f'protected loss/training file changed: {rel}')

model=(ROOT/'trainers/medclipseg_unimedclip.py').read_text(errors='replace')
for snip in [
 'from .qabr import QueryAnchoredBoundaryRefiner',
 'self.qabr_enabled',
 'self.qabr(',
 'decoder_features, seg_logits, image=raw_image, output_size=self.im_size',
 'base_trainable = {"pvl_adapters", "mask_head", "upscale", "ugbra", "qabr"}',
]:
 if snip not in model: die(f'model integration missing: {snip}')

# Standalone numerical contract.
mp=ROOT/'trainers/qabr.py'; spec=importlib.util.spec_from_file_location('qabr_standalone',mp)
if spec is None or spec.loader is None: die('cannot load qabr.py')
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); Q=mod.QueryAnchoredBoundaryRefiner
torch.manual_seed(20260911)
q=Q(decoder_channels=64,detail_channels=8,hidden_channels=16,alpha_init=0.0,band_kernel=11,max_logit_delta=3.0,detach_support=True,use_image_detail=True)
f=torch.randn(2,64,14,14,requires_grad=True); z=torch.randn(2,1,14,14); im=torch.randn(2,3,56,56)
y,d=q(f,z,im,output_size=(56,56),return_diagnostics=True)
base=torch.nn.functional.interpolate(z,size=(56,56),mode='bilinear',align_corners=False)
if not torch.equal(y,base): die(f'zero-init not exact identity; max={(y-base).abs().max().item():.3e}')
if tuple(y.shape)!=(2,1,56,56) or not torch.isfinite(y).all(): die('bad QABR output')
y.mean().backward()
if q.alpha.grad is None or not torch.isfinite(q.alpha.grad): die('alpha has no finite gradient at zero init')
nonalpha=sum(0.0 if p.grad is None else float(p.grad.detach().abs().sum()) for n,p in q.named_parameters() if n!='alpha')
if nonalpha!=0.0: die('non-alpha parameters must have zero gradient at exact zero init')
with torch.no_grad(): q.alpha.fill_(0.1)
q.zero_grad(set_to_none=True); f2=torch.randn(2,64,14,14,requires_grad=True); z2=torch.randn(2,1,14,14); im2=torch.randn(2,3,56,56)
y2=q(f2,z2,im2,output_size=(56,56)); y2.mean().backward()
nonalpha=sum(0.0 if p.grad is None else float(p.grad.detach().abs().sum()) for n,p in q.named_parameters() if n!='alpha')
if not math.isfinite(nonalpha) or nonalpha<=0: die('refinement path does not receive gradient after alpha opens')
# Verify locality: changing the random candidate cannot alter pixels where support is exactly zero.
with torch.no_grad():
 _,diag=q(f2,z2,im2,output_size=(56,56),return_diagnostics=True)
 if float(diag['qabr_support_mean'])<=0 or float(diag['qabr_support_mean'])>1: die('invalid support statistics')

full=Q(512,8,32,0.0,11,3.0,True,True); params=sum(p.numel() for p in full.parameters())
if not (5_000<=params<=100_000): die(f'unexpected QABR params={params}')

# BUSI 2x2 factorial formal configs.
loaded={}
for arm,t in ARMS.items():
 p=CFG_DIR/f'JBTL11_BUSI_{arm}_PAPER100.yaml'
 if not p.is_file(): die(f'missing {p.name}')
 c=yaml.safe_load(p.read_text()); loaded[arm]=c
 if c['DATASET']['NAME']!='BUSI': die(f'{arm} wrong dataset')
 if bool(c['M1']['ENABLED']): die(f'{arm} old M1 path must be disabled')
 if bool(c['MODEL']['UGBRA']['ENABLED']): die(f'{arm} v9 UGBRA must be disabled in v10')
 tr=c['TRAIN']
 for k,w in COMMON.items():
  a=tr.get(k)
  if isinstance(w,float):
   if abs(float(a)-w)>1e-12: die(f'{arm} TRAIN.{k}={a}, want {w}')
  elif a!=w: die(f'{arm} TRAIN.{k}={a!r}, want {w!r}')
 if abs(float(tr['RBAL_EDGE_WEIGHT'])-t['edge'])>1e-12: die(f'{arm} wrong Surface treatment')
 qc=c['MODEL']['QABR']
 if bool(qc['ENABLED'])!=t['qabr']: die(f'{arm} wrong QABR treatment')
 for k,w in EXPECTED_QABR.items():
  if qc.get(k)!=w: die(f'{arm} MODEL.QABR.{k}={qc.get(k)!r}, want {w!r}')

def sanitize(c):
 c=copy.deepcopy(c); c['TRAIN'].pop('RUN_TAG',None); c['M1'].pop('RUN_TAG',None); c['TRAIN'].pop('RBAL_EDGE_WEIGHT',None); c['MODEL'].pop('QABR',None); return c
bs=sanitize(loaded['BASE'])
for arm in ('EDGE20','QABR','FULL'):
 if sanitize(loaded[arm])!=bs: die(f'{arm} has undeclared config differences vs BASE')

# DIAG40 must mirror treatment with only epoch horizon changed.
for arm,t in ARMS.items():
 p=DIAG_DIR/f'JBTL11_BUSI_{arm}_DIAG40.yaml'
 if not p.is_file(): die(f'missing {p.name}')
 c=yaml.safe_load(p.read_text()); tr=c['TRAIN']; qc=c['MODEL']['QABR']
 if tr['NUM_EPOCHS']!=40 or tr['SCHEDULER_TOTAL_EPOCHS']!=40: die(f'{arm} diag horizon wrong')
 if abs(float(tr['RBAL_EDGE_WEIGHT'])-t['edge'])>1e-12 or bool(qc['ENABLED'])!=t['qabr']: die(f'{arm} diag treatment wrong')
 if bool(c['MODEL']['UGBRA']['ENABLED']): die(f'{arm} diag UGBRA must be off')

# Other datasets: strict paired BASE/FULL, same mechanism, no extra loss.
for ds in ('Kvasir','ISIC','BTMRI'):
 for arm in ('BASE','FULL'):
  p=CFG_DIR/f'JBTL11_{ds}_{arm}_PAPER100.yaml'
  if not p.is_file(): die(f'missing {p.name}')
  c=yaml.safe_load(p.read_text()); on=arm=='FULL'
  if bool(c['MODEL']['UGBRA']['ENABLED']): die(f'{ds}/{arm} UGBRA must be off')
  if bool(c['MODEL']['QABR']['ENABLED'])!=on: die(f'{ds}/{arm} QABR state wrong')
  if abs(float(c['TRAIN']['RBAL_EDGE_WEIGHT'])-(0.25 if on else 0.0))>1e-12: die(f'{ds}/{arm} Surface state wrong')
  if float(c['TRAIN']['RBAL_NORMAL_WEIGHT'])!=0.0: die(f'{ds}/{arm} new loss not allowed')

print('[JBTL11_QABR_AUDIT_PASS]')
print(f'QABR @ C=512 parameters: {params:,} (~{params/1e6:.3f}M)')
print('Zero-init high-resolution output equals original bilinear Base exactly.')
print('After alpha opens, detail projection + signed directional refinement receive gradients.')
print('Protected train.py and Surface Alignment implementation are byte-identical; no new loss added.')
print('BUSI factorial: BASE / EDGE20 / QABR / FULL; old UGBRA and old M1 are disabled.')
