#!/usr/bin/env python3
"""One-shot Test finalizer for one locked SemLT ablation variant."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path

def sha(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()

def run(cmd,env):
    cmd=[str(x) for x in cmd]; print('+',' '.join(cmd),flush=True); subprocess.run(cmd,check=True,env=env)

def resolve_overrides(project: Path, py: str, spec: Path, variant: str):
    out=subprocess.check_output([py,str(project/'tools/semlt_ablation_spec.py'),'--spec',str(spec),'--variant',variant,'--emit-overrides'],text=True)
    pairs=[]
    for line in out.splitlines():
        k,v=line.split('\t',1); pairs += [k,v]
    return pairs

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',required=True); ap.add_argument('--gpu',required=True); ap.add_argument('--python',default=sys.executable); a=ap.parse_args()
    project=Path.cwd().resolve(); mf=Path(a.manifest).resolve(); m=json.loads(mf.read_text(encoding='utf-8'))
    if m.get('protocol')!='SemLT_locked_causal_ablation_v1' or not m.get('method_locked_before_test'): raise SystemExit('[FAIL] invalid/unlocked manifest')
    ds=m['dataset']; seed=int(m['seed']); variant=m['variant']; run_dir=Path(m['run_dir']); ckpt=Path(m['checkpoint']); base=Path(m['base_checkpoint'])
    config=(project/m['config']).resolve() if not Path(m['config']).is_absolute() else Path(m['config'])
    spec=(project/m['spec']).resolve() if not Path(m['spec']).is_absolute() else Path(m['spec'])
    checks=[(config,m['config_sha256'],'config'),(spec,m['spec_sha256'],'spec'),(project/'utils/geotr_m1_transport.py',m['transport_sha256'],'transport source'),(project/'utils/geotr_m1_loss.py',m['loss_sha256'],'loss source'),(base,m['base_checkpoint_sha256'],'Base checkpoint')]
    for p,expected,label in checks:
        if not p.is_file() or sha(p)!=expected: raise SystemExit(f'[FAIL] {label} changed after validation: {p}')
    val_lock=run_dir/'FORMAL_VAL_SUCCESS.lock'; open_lock=run_dir/'FORMAL_TEST_OPENED.lock'; success=run_dir/'FORMAL_TEST_SUCCESS.lock'
    if not val_lock.is_file(): raise SystemExit('[FAIL] validation was not completed')
    if open_lock.exists() or success.exists() or m.get('test_opened'): raise SystemExit(f'[FAIL] Test already opened: {ds} seed{seed} {variant}')
    if not ckpt.is_file(): raise SystemExit(f'[FAIL] checkpoint missing: {ckpt}')
    tag=m['run_tag']; run_name=f'MedCLIPSeg_unimedclip_ViT-B-16_{tag}'; out=run_dir/'formal_test'; result_root=out/ds/'seg_results'/f'seed{seed}'
    overrides=['TRAIN.RUN_TAG',tag,'M1.RUN_TAG',tag,'DATASET.NAME',ds,'DATASET.TRAIN_PATH',m['train_path'],'DATASET.VAL_PATH',m['val_path'],'DATASET.TEST_PATH',m['test_path'],'DATASET.TEXT_PROMPT_PATH',m['prompt_path']]
    overrides += resolve_overrides(project,a.python,spec,variant)
    dataset_root=run_dir.parent.parent; test_registry=dataset_root/'test_common_base_registry.json'
    open_lock.write_text(f'opened_at={__import__("datetime").datetime.now().astimezone().isoformat()}\ndataset={ds}\nseed={seed}\nvariant={variant}\ncheckpoint={ckpt}\n',encoding='utf-8')
    env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(a.gpu); env['TOKENIZERS_PARALLELISM']='false'; env['PYTHONPATH']=str(project)+((':'+env['PYTHONPATH']) if env.get('PYTHONPATH') else '')
    try:
        run([a.python,'-u','test.py','--config-file',config,'--seed',seed,'--split','test','--num-samples','1','--checkpoint',ckpt,'--output-dir',out,*overrides],env)
        for result,csv_name,mode in [(f'{run_name}_BaseNative','test_BaseNative_true2d.csv','true2d'),(f'{run_name}_M1Native','test_M1Native_true2d.csv','true2d'),(f'{run_name}_BaseNative','test_BaseNative_paper_legacy.csv','paper_legacy'),(f'{run_name}_M1Native','test_M1Native_paper_legacy.csv','paper_legacy')]:
            run([a.python,'-u','utils/eval.py','--config-file',config,'--seed',seed,'--split','test','--output-dir',out,'--result-name',result,'--csv-name',csv_name,'--nsd-mode',mode,*overrides],env)
        for mode in ('true2d','paper_legacy'):
            run([a.python,'tools/compare_geotr_m1_paired.py','--base-csv',result_root/f'test_BaseNative_{mode}.csv','--m1-csv',result_root/f'test_M1Native_{mode}.csv','--output-prefix',result_root/f'paired_{mode}','--protocol-label',f'{ds}_test_{mode}_SemLT_locked_ablation'],env)
        run([a.python,'tools/verify_geotr_m1_common_base.py','--checkpoint',base,'--base-csv',result_root/'test_BaseNative_true2d.csv','--registry',test_registry],env)
        success.write_text(f'completed_dataset={ds}\nseed={seed}\nvariant={variant}\nreport={result_root / "paired_true2d.json"}\n',encoding='utf-8')
        m['test_opened']=True; m['test_success']=True; m['test_report']=str((result_root/'paired_true2d.json').resolve()); m['test_common_base_registry']=str(test_registry.resolve()); mf.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(f'[PASS] Test complete: {ds} seed{seed} {variant}')
    except Exception:
        print('[FAIL] Test was opened but did not complete. The lock is intentionally retained.',file=sys.stderr); raise
if __name__=='__main__': main()
