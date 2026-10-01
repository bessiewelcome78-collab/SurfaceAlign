#!/usr/bin/env python3
"""Open one fixed LeST59 Test split exactly once from a train+val manifest.

This script is intentionally fail-closed. It verifies the method/config/base
checkpoint bound by the manifest, creates a TEST_OPENED lock before inference,
and never trains or selects a checkpoint on Test.
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024), b''): h.update(c)
    return h.hexdigest()

def run(cmd, env=None):
    print('+', ' '.join(map(str,cmd)), flush=True)
    subprocess.run([str(x) for x in cmd], check=True, env=env)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--gpu', required=True)
    ap.add_argument('--python', default=sys.executable)
    args=ap.parse_args()
    project=Path.cwd().resolve()
    mf=Path(args.manifest).resolve()
    m=json.loads(mf.read_text(encoding='utf-8'))
    if m.get('protocol')!='LeST59_fixed_unified_aniso_formal100':
        raise SystemExit('[FAIL] manifest protocol mismatch')
    if not m.get('method_locked_before_test'):
        raise SystemExit('[FAIL] method was not locked before Test')
    dataset=m['dataset']; seed=int(m['seed']); tag=m['m1_tag']
    config=project/m['m1_config']
    m1_run=Path(m['m1_run_dir'])
    if not m1_run.is_absolute(): m1_run=project/m1_run
    base_ckpt=Path(m['base_checkpoint'])
    val_lock=m1_run/'FORMAL_VAL_SUCCESS.lock'
    open_lock=m1_run/'FORMAL_TEST_OPENED.lock'
    success=m1_run/'FORMAL_TEST_SUCCESS.lock'
    if not val_lock.is_file(): raise SystemExit(f'[FAIL] validation success lock missing: {val_lock}')
    if open_lock.exists() or success.exists() or m.get('test_opened'):
        raise SystemExit(f'[FAIL] Test already opened for {dataset} seed{seed}')
    if sha256(base_ckpt)!=m['base_checkpoint_sha256']:
        raise SystemExit('[FAIL] Base checkpoint SHA256 changed after validation')
    run_name=f'MedCLIPSeg_unimedclip_ViT-B-16_{tag}'
    ckpt=m1_run/dataset/'trained_models'/f'seed{seed}'/f'{run_name}_last_epoch.pth'
    if not ckpt.is_file(): raise SystemExit(f'[FAIL] M1 checkpoint missing: {ckpt}')
    out=m1_run/'formal_test'
    result_root=out/dataset/'seg_results'/f'seed{seed}'
    overrides=[
        'TRAIN.RUN_TAG',tag,'M1.RUN_TAG',tag,
        'DATASET.NAME',dataset,
        'DATASET.TRAIN_PATH',m['train_path'],'DATASET.VAL_PATH',m['val_path'],
        'DATASET.TEST_PATH',m['test_path'],'DATASET.TEXT_PROMPT_PATH',m['prompt_path'],
    ]
    open_lock.write_text(
        f'opened_at_protocol=LeST59_fixed_before_test\ndataset={dataset}\nseed={seed}\ncheckpoint={ckpt}\n',
        encoding='utf-8')
    env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(args.gpu); env['TOKENIZERS_PARALLELISM']='false'
    env['PYTHONPATH']=str(project)+((':'+env['PYTHONPATH']) if env.get('PYTHONPATH') else '')
    try:
        run([args.python,'-u','test.py','--config-file',config,'--seed',seed,'--split','test','--num-samples','1',
             '--checkpoint',ckpt,'--output-dir',out,*overrides],env)
        for result,csv_name,mode in [
            (f'{run_name}_BaseNative','test_BaseNative_true2d.csv','true2d'),
            (f'{run_name}_M1Native','test_M1Native_true2d.csv','true2d'),
            (f'{run_name}_BaseNative','test_BaseNative_paper_legacy.csv','paper_legacy'),
            (f'{run_name}_M1Native','test_M1Native_paper_legacy.csv','paper_legacy')]:
            run([args.python,'-u','utils/eval.py','--config-file',config,'--seed',seed,'--split','test','--output-dir',out,
                 '--result-name',result,'--csv-name',csv_name,'--nsd-mode',mode,*overrides],env)
        run([args.python,'tools/compare_geotr_m1_paired.py','--base-csv',result_root/'test_BaseNative_true2d.csv',
             '--m1-csv',result_root/'test_M1Native_true2d.csv','--output-prefix',result_root/'paired_true2d',
             '--protocol-label',f'{dataset}_test_corrected_true_2d_nsd'],env)
        run([args.python,'tools/compare_geotr_m1_paired.py','--base-csv',result_root/'test_BaseNative_paper_legacy.csv',
             '--m1-csv',result_root/'test_M1Native_paper_legacy.csv','--output-prefix',result_root/'paired_paper_legacy',
             '--protocol-label',f'{dataset}_test_reference_paper_singleton_depth_nsd'],env)
        test_registry=m1_run/f'test_common_base_registry_seed{seed}.json'
        run([args.python,'tools/verify_geotr_m1_common_base.py','--checkpoint',base_ckpt,
             '--base-csv',result_root/'test_BaseNative_true2d.csv','--registry',test_registry],env)
        success.write_text(f'completed_dataset={dataset}\nseed={seed}\nreport={result_root / "paired_true2d.json"}\n',encoding='utf-8')
        m['test_opened']=True; m['test_success']=True; m['test_report']=str((result_root/'paired_true2d.json').resolve())
        mf.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(f'[PASS] LeST59 Test complete: {dataset} seed{seed}')
        print(result_root/'paired_true2d.md')
    except Exception:
        print('[FAIL] Test was opened but did not complete. Lock is intentionally retained; do not silently rerun.', file=sys.stderr)
        raise

if __name__=='__main__': main()
