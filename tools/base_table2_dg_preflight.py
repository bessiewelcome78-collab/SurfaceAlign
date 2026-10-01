#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Optional
import yaml

IMG_EXT={'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}


def count_files(p: Path):
    return sum(1 for x in p.iterdir() if x.is_file() and x.suffix.lower() in IMG_EXT)


def resolve_target_root(data_dir: Path, target: str):
    aliases={
        'UWaterlooSkinCancer':['UWaterlooSkinCancer','UWaterloo'],
        'ColonDB':['ColonDB','CVC-ColonDB'],
        'ClinicDB':['ClinicDB','CVC-ClinicDB'],
        'CVC300':['CVC300','CVC-300'],
    }
    names=aliases.get(target,[target])
    for n in names:
        root=data_dir/n
        for cand in [root/'Test_Folder', root/'test', root/'Test']:
            if (cand/'img').is_dir() and (cand/'label').is_dir():
                return root, cand
    raise FileNotFoundError(f'{target}: no test img/label folder under {data_dir}')


def resolve_prompt_dir(root: Path, target: str, data_dir: Path):
    candidates=[root/'Prompts_Folder']
    for n in [target, 'UWaterlooSkinCancer' if target=='UWaterlooSkinCancer' else target,
              'UWaterloo' if target=='UWaterlooSkinCancer' else target,
              'CVC-ColonDB' if target=='ColonDB' else target,
              'CVC-ClinicDB' if target=='ClinicDB' else target,
              'CVC-300' if target=='CVC300' else target]:
        candidates.append(data_dir/n/'Prompts_Folder')
    seen=set()
    for p in candidates:
        p=p.resolve() if p.exists() else p
        if str(p) in seen: continue
        seen.add(str(p))
        if p.is_dir() and (p/'Test_text_original.xlsx').is_file():
            return p
    raise FileNotFoundError(f'{target}: Test_text_original.xlsx not found')


def parse_eval_log(path: Path):
    if not path.is_file(): return None
    txt=path.read_text(errors='ignore')
    md=re.findall(r'Average DSC .*?:\s*([0-9.]+)%',txt)
    mn=re.findall(r'Average NSD .*?:\s*([0-9.]+)%',txt)
    mc=re.findall(r'Cases evaluated:\s*([0-9]+)',txt)
    if not md: return None
    return {'dsc':float(md[-1]), 'nsd':float(mn[-1]) if mn else None, 'cases':int(mc[-1]) if mc else None, 'log':str(path)}


def source_metrics_near_lock(lock: Path):
    seed=lock.parent
    out={}
    for mode in ['paper_legacy','true2d']:
        cands=[seed/'logs'/f'eval_test_mc30_{mode}.log', seed/f'eval_test_mc30_{mode}.log']
        for p in cands:
            v=parse_eval_log(p)
            if v:
                out[mode]=v; break
    return out


def repair_checkpoint(lock: Path, raw: str, project: Path) -> Optional[Path]:
    raw=raw.strip()
    if not raw: return None
    cp=Path(raw)
    if cp.is_file() and cp.stat().st_size>0: return cp.resolve()
    # Most copied studies keep the same result-tree-relative checkpoint, while
    # LOCKED_CHECKPOINT.txt can still contain the old project root.
    local=lock.parent/'train'/lock.parent.parent.parent.name/'trained_models'/'seed42'/cp.name
    if local.is_file() and local.stat().st_size>0: return local.resolve()
    # Generic local search inside the selected seed root first.
    for p in lock.parent.rglob(cp.name):
        if p.is_file() and p.stat().st_size>0: return p.resolve()
    # Last resort: same checkpoint basename elsewhere in current project.
    hits=[]
    try:
        for p in project.rglob(cp.name):
            if p.is_file() and p.stat().st_size>0: hits.append(p.resolve())
            if len(hits)>2: break
    except OSError:
        pass
    return hits[0] if len(hits)==1 else None


def expand_lock_path(project: Path, s: str, study: str):
    s=s.replace('{base_crossdataset_study_id}',study)
    p=Path(s)
    return p if p.is_absolute() else project/p


def preferred_locks(project: Path, spec: dict, study: str):
    out=[]
    for s in spec.get('preferred_lock_files',[]):
        p=expand_lock_path(project,s,study)
        if p not in out: out.append(p)
    return out


def fallback_locks(project: Path, spec: dict):
    out=[]
    for g in spec.get('fallback_lock_globs',[]):
        if g.startswith('/'):
            continue
        for p in sorted(project.glob(g)):
            if p not in out: out.append(p)
    return out


def evaluate_lock(project: Path, src: str, spec: dict, expected: dict, lock: Path):
    if not lock.is_file():
        return None, f'{lock}: lock missing'
    raw=lock.read_text(errors='ignore').strip()
    cp=repair_checkpoint(lock,raw,project)
    if cp is None:
        return None, f'{lock}: checkpoint missing/stale: {raw}'
    name=cp.name
    bad=[tok for tok in spec.get('checkpoint_name_must_contain',[]) if tok not in name]
    if bad:
        return None, f'{lock}: checkpoint name lacks {bad}: {name}'
    if not (name.endswith('_best_val.pth') or name.endswith('_best_base_val.pth')):
        return None, f'{lock}: not a best-Val checkpoint: {name}'
    metrics=source_metrics_near_lock(lock)
    score=0
    leg=metrics.get('paper_legacy')
    tru=metrics.get('true2d')
    if leg and expected:
        if abs(leg['dsc']-float(expected['dsc'])) <= 0.06: score += 5
        else: score -= 20
        if 'legacy_nsd' in expected and leg.get('nsd') is not None:
            if abs(leg['nsd']-float(expected['legacy_nsd'])) <= 0.06: score += 5
            else: score -= 20
    if tru and 'true2d_nsd' in expected and tru.get('nsd') is not None:
        if abs(tru['nsd']-float(expected['true2d_nsd'])) <= 0.06: score += 3
        else: score -= 10
    return (score, lock.resolve(), cp, metrics), None


def select_checkpoint(project: Path, src: str, spec: dict, study: str, expected: dict):
    rejects=[]
    # Fast path: exact known run identities. This avoids recursively scanning a
    # very large results tree during normal use.
    preferred=[]
    for lock in preferred_locks(project,spec,study):
        item,err=evaluate_lock(project,src,spec,expected,lock)
        if item is not None: preferred.append(item)
        elif err: rejects.append(err)
    if preferred:
        preferred.sort(key=lambda x:x[0], reverse=True)
        best=preferred[0]
        # score >= 0 means either metrics match or no local metric log exists;
        # in the latter case the exact run identity is intentionally trusted.
        if best[0] >= 0:
            return {'lock_file':str(best[1]), 'checkpoint':str(best[2]), 'checkpoint_name':best[2].name,
                    'checkpoint_bytes':best[2].stat().st_size, 'source_metric_evidence':best[3], 'selection_score':best[0]}
        rejects.append(f'{best[1]}: local ID metrics contradict expected paper BASE row: {best[3]} expected={expected}')
    # Slow fallback only when an exact preferred run is genuinely unavailable
    # or contradicted by its own evaluation log.
    usable=[]
    seen=set(str(p) for p in preferred_locks(project,spec,study))
    for lock in fallback_locks(project,spec):
        if str(lock) in seen: continue
        item,err=evaluate_lock(project,src,spec,expected,lock)
        if item is not None: usable.append(item)
        elif err: rejects.append(err)
    if not usable:
        raise RuntimeError(f'{src}: no usable BASE checkpoint. ' + '; '.join(rejects[:12]))
    usable.sort(key=lambda x:x[0], reverse=True)
    best=usable[0]
    if best[0] < -5:
        raise RuntimeError(f'{src}: fallback checkpoint(s) exist but local ID metrics contradict expected paper BASE row. best={best[1]} metrics={best[3]} expected={expected}')
    return {'lock_file':str(best[1]), 'checkpoint':str(best[2]), 'checkpoint_name':best[2].name,
            'checkpoint_bytes':best[2].stat().st_size, 'source_metric_evidence':best[3], 'selection_score':best[0]}

def choose_config(project: Path, src: str, spec: dict):
    errors=[]
    for rel in spec.get('config_candidates',[]):
        p=project/rel
        if not p.is_file():
            errors.append(f'{rel}: missing'); continue
        cfg=yaml.safe_load(p.read_text()) or {}
        checks={
          'DATASET.NAME': cfg.get('DATASET',{}).get('NAME')==src,
          'M1.ENABLED': cfg.get('M1',{}).get('ENABLED') is False,
          'MODEL.QABR.ENABLED': cfg.get('MODEL',{}).get('QABR',{}).get('ENABLED') is False,
          'MODEL.UGBRA.ENABLED': cfg.get('MODEL',{}).get('UGBRA',{}).get('ENABLED') is False,
          'TRAIN.NUM_EPOCHS': int(cfg.get('TRAIN',{}).get('NUM_EPOCHS',-1))==100,
          'TRAIN.BATCH_SIZE': int(cfg.get('TRAIN',{}).get('BATCH_SIZE',-1))==24,
          'TRAIN.LEARNING_RATE': abs(float(cfg.get('TRAIN',{}).get('LEARNING_RATE',-1))-3e-4)<1e-12,
          'TRAIN.VAL_NUM_SAMPLES': int(cfg.get('TRAIN',{}).get('VAL_NUM_SAMPLES',-1))==10,
          'TEST.NUM_SAMPLES': int(cfg.get('TEST',{}).get('NUM_SAMPLES',-1))==30,
          'TRAIN.RBAL_EDGE_WEIGHT': float(cfg.get('TRAIN',{}).get('RBAL_EDGE_WEIGHT',999))==0.0,
          'TRAIN.RBAL_NORMAL_WEIGHT': float(cfg.get('TRAIN',{}).get('RBAL_NORMAL_WEIGHT',999))==0.0,
        }
        bad=[k for k,v in checks.items() if not v]
        if bad:
            errors.append(f'{rel}: contract failed {bad}'); continue
        return p.resolve(), checks
    raise RuntimeError(f'{src}: no valid BASE config: ' + '; '.join(errors))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project',required=True)
    ap.add_argument('--data-dir',required=True)
    ap.add_argument('--manifest',default='configs/domain_generalization_base/BASE_TABLE2_DG_S42.json')
    ap.add_argument('--write-resolved',required=True)
    a=ap.parse_args()
    project=Path(a.project).resolve(); data_dir=Path(a.data_dir).resolve()
    manifest=json.loads((project/a.manifest).read_text())
    active=project/'BASE_CROSSDATASET_S42_ACTIVE.txt'
    study=active.read_text().strip() if active.is_file() else ''
    resolved={'protocol':manifest['protocol'],'seed':manifest['seed'],'mc_samples':manifest['mc_samples'],
              'base_crossdataset_study_id':study,'sources':{},'targets':{}}
    errors=[]
    print(f'[INFO] project={project}')
    print(f'[INFO] base_crossdataset_study_id={study or "<not found>"}')
    for src,spec in manifest['sources'].items():
        try:
            cfg,_=choose_config(project,src,spec)
            ck=select_checkpoint(project,src,spec,study,manifest.get('source_id_reference',{}).get(src,{}))
            resolved['sources'][src]={'config':str(cfg), **ck,
                'expected_id_metrics':manifest.get('source_id_reference',{}).get(src,{}),
                'targets':spec['targets']}
            print(f'[PASS] SOURCE {src}')
            print(f'       config={cfg}')
            print(f'       lock={ck["lock_file"]}')
            print(f'       checkpoint={ck["checkpoint"]}')
            if ck['source_metric_evidence']:
                print(f'       local_source_metrics={ck["source_metric_evidence"]}')
            else:
                print(f'       local_source_metrics=<not found; using exact known run identity>')
        except Exception as e:
            errors.append(str(e)); continue
        for tgt in spec['targets']:
            if tgt in resolved['targets']: continue
            try:
                root,test=resolve_target_root(data_dir,tgt)
                prompt=resolve_prompt_dir(root,tgt,data_dir)
                ni=count_files(test/'img'); nl=count_files(test/'label'); exp=int(manifest['expected_cases'][tgt])
                if ni!=exp or nl<exp:
                    raise RuntimeError(f'case count img={ni}, label={nl}, expected img={exp}, label>={exp}')
                resolved['targets'][tgt]={'dataset_root':str(root.resolve()),'test_path':str(test.resolve())+'/',
                    'prompt_path':str(prompt.resolve())+'/', 'images':ni,'labels':nl,'expected':exp}
                print(f'[PASS] TARGET {tgt}: images={ni} labels={nl} prompt=yes')
            except Exception as e:
                errors.append(f'{src}->{tgt}: {e}')
    if errors:
        print('\n[FAIL] preflight:')
        for e in errors: print(' -',e)
        raise SystemExit(20)
    out=Path(a.write_resolved); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(resolved,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'\n[PASS] BASE Table-2 DG preflight complete')
    print(f'[RESOLVED] {out}')

if __name__=='__main__': main()
