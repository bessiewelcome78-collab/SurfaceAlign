#!/usr/bin/env python3
from pathlib import Path
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument('--root', default='/home/tsz-25/MedCLIPSeg-pristine/runs/UCFNRT_CAUSAL_ABL_20260831_132730/seed42/BUSI')
a = ap.parse_args()
root = Path(a.root)

for v in ('NO_POSTERIOR','NO_RAY','DIRECT_SIGNED','SEG_ONLY'):
    d = root / v
    state = d / 'formal_state'
    lock_candidates = [state/'TRAIN_SUCCESS.lock', d/'TRAIN_SUCCESS.lock']
    manifest_candidates = [state/'TRAIN_MANIFEST.json', d/'TRAIN_MANIFEST.json']
    lock = next((p for p in lock_candidates if p.is_file()), None)
    manifest = next((p for p in manifest_candidates if p.is_file()), None)
    ckpts = list(d.rglob('*_last_epoch.pth')) if d.exists() else []
    ck = max(ckpts, key=lambda p: p.stat().st_mtime) if ckpts else None

    manifest_ckpt = None
    sha_ok = None
    if manifest:
        try:
            obj = json.loads(manifest.read_text(encoding='utf-8'))
            cp = obj.get('checkpoint')
            if cp:
                manifest_ckpt = Path(cp)
                if manifest_ckpt.is_file():
                    ck = manifest_ckpt
        except Exception:
            pass

    if lock and manifest and ck:
        status = 'FORMAL_COMPLETE'
    elif ck:
        status = 'CHECKPOINT_ONLY'
    else:
        status = 'NOT_COMPLETE'

    print(
        f'{v:15s} {status:16s} '
        f'lock={str(lock) if lock else "NONE"} '
        f'manifest={str(manifest) if manifest else "NONE"} '
        f'ckpt={str(ck) if ck else "NONE"}'
    )
