#!/usr/bin/env python3
"""One-shot Test + paired evaluation for matched-budget SemLT E2E100."""
from __future__ import annotations

import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024), b''):
            h.update(c)
    return h.hexdigest()


def run(cmd, cwd, env=None):
    print('+', ' '.join(map(str,cmd)), flush=True)
    subprocess.run(list(map(str,cmd)), cwd=str(cwd), env=env, check=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--gpu', required=True)
    ap.add_argument('--python', required=True)
    args=ap.parse_args()

    manifest=Path(args.manifest).resolve()
    if not manifest.is_file():
        raise SystemExit(f'[FAIL] missing manifest: {manifest}')
    m=json.loads(manifest.read_text())
    if m.get('protocol') != 'SEMLT_E2E100_MATCHED_BUDGET_MAIN_TABLE':
        raise SystemExit(f"[FAIL] wrong protocol: {m.get('protocol')}")
    if m.get('total_training_epochs') != 100 or not m.get('base_and_semlt_same_run'):
        raise SystemExit('[FAIL] manifest is not matched-budget E2E100')
    if m.get('validation_selection') is not False or m.get('checkpoint_policy') != 'physical_last_epoch_100':
        raise SystemExit('[FAIL] checkpoint selection protocol mismatch')
    if not m.get('semlt_base_evidence_detached'):
        raise SystemExit('[FAIL] gradient-isolation contract not declared')
    if m.get('test_opened') or m.get('test_success'):
        raise SystemExit('[FAIL] Test already opened/completed according to manifest')

    run_dir=Path(m['run_dir']).resolve()
    project=manifest.parents[5] if False else Path(__file__).resolve().parents[1]
    # Prefer code tree containing this script; the manifest must point inside the same project run tree.
    project=project.resolve()
    train_lock=run_dir/'FORMAL_TRAIN_SUCCESS.lock'
    open_lock=run_dir/'FORMAL_TEST_OPENED.lock'
    success_lock=run_dir/'FORMAL_TEST_SUCCESS.lock'
    if not train_lock.is_file():
        raise SystemExit(f'[FAIL] training success lock missing: {train_lock}')
    if open_lock.exists() or success_lock.exists():
        raise SystemExit('[FAIL] Test lock already exists; refusing second opening')

    config=Path(m['config']).resolve(); ckpt=Path(m['checkpoint']).resolve()
    if not config.is_file() or not ckpt.is_file():
        raise SystemExit('[FAIL] config/checkpoint missing')
    if sha256(config) != m['config_sha256']:
        raise SystemExit('[FAIL] config SHA changed after training')
    if sha256(ckpt) != m['checkpoint_sha256']:
        raise SystemExit('[FAIL] checkpoint SHA changed after training')

    dataset=m['dataset']; seed=int(m['seed']); run_tag=m['run_tag']
    run_name=f'MedCLIPSeg_unimedclip_ViT-B-16_{run_tag}'
    out=run_dir/'formal_test'
    result_root=out/dataset/'seg_results'/f'seed{seed}'

    open_lock.write_text(
        f"opened_at={datetime.now(timezone.utc).isoformat()}\n"
        f"dataset={dataset}\nseed={seed}\ncheckpoint={ckpt}\n",
        encoding='utf-8'
    )
    m['test_opened']=True
    m['test_opened_at']=datetime.now(timezone.utc).isoformat()
    manifest.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

    env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(args.gpu); env['PYTHONPATH']=str(project)+((':'+env['PYTHONPATH']) if env.get('PYTHONPATH') else '')
    env['TOKENIZERS_PARALLELISM']='false'
    try:
        overrides=['TRAIN.RUN_TAG',run_tag,'M1.RUN_TAG',run_tag]
        run([args.python,'-u','test.py','--config-file',config,'--seed',seed,'--split','test','--num-samples','1',
             '--checkpoint',ckpt,'--output-dir',out,*overrides], project, env)

        base_name=run_name+'_BaseNative'; m1_name=run_name+'_M1Native'
        for protocol in ('true2d','paper_legacy'):
            for name, stem in ((base_name,'BaseNative'),(m1_name,'M1Native')):
                csv=result_root/f'test_{stem}_{protocol}.csv'
                run([args.python,'-u','utils/eval.py','--config-file',config,'--seed',seed,'--split','test',
                     '--output-dir',out,'--result-name',name,'--csv-name',csv.name,'--nsd-mode',protocol,*overrides], project, env)
            run([args.python,'tools/compare_geotr_m1_paired.py',
                 '--base-csv',result_root/f'test_BaseNative_{protocol}.csv',
                 '--m1-csv',result_root/f'test_M1Native_{protocol}.csv',
                 '--output-prefix',result_root/f'paired_{protocol}',
                 '--protocol-label',f'{dataset}_test_{protocol}_SEMLT_E2E100'], project, env)

        success_lock.write_text(
            f"completed_at={datetime.now(timezone.utc).isoformat()}\n"
            f"dataset={dataset}\nseed={seed}\nreport={result_root/'paired_paper_legacy.json'}\n",
            encoding='utf-8'
        )
        m=json.loads(manifest.read_text())
        m['test_success']=True
        m['test_success_at']=datetime.now(timezone.utc).isoformat()
        m['result_root']=str(result_root)
        m['table1_report']=str(result_root/'paired_paper_legacy.json')
        m['secondary_true2d_report']=str(result_root/'paired_true2d.json')
        manifest.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print('[PASS] one-shot Test complete')
        print('       TABLE1=', result_root/'paired_paper_legacy.md')
        print('       TRUE2D=', result_root/'paired_true2d.md')
    except Exception:
        print('[FAIL] Test was opened but did not complete. The OPENED lock is intentionally retained.', file=sys.stderr)
        raise

if __name__=='__main__':
    main()
