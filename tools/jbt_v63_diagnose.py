#!/usr/bin/env python3
"""Automatic paired diagnostic for JBT-v6.

Separates: (A) exact-Base trajectory drift, (B) module gain from the SAME joint
checkpoint, (C) net gain, and (D) candidate-capacity vs selector realization.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd

PAPER = {
    'BUSI': {'DSC': 0.8572, 'NSD': 0.8835},
    'Kvasir': {'DSC': 0.9015, 'NSD': 0.9232},
}

def load_csv(path):
    f = pd.read_csv(path)
    need = {'Case_ID','DSC','NSD'}
    miss = need - set(f.columns)
    if miss: raise ValueError(f'{path} missing {sorted(miss)}')
    if f.Case_ID.duplicated().any(): raise ValueError(f'{path} duplicate Case_ID')
    return f.set_index('Case_ID').sort_index()

def align(*fs):
    ids = set(fs[0].index)
    if any(set(f.index) != ids for f in fs[1:]): raise ValueError('CSV case sets differ')
    order = sorted(ids)
    return [f.loc[order] for f in fs]

def stat(a,b):
    a,b = np.asarray(a,float),np.asarray(b,float); d=b-a
    return {'a_mean':float(a.mean()),'b_mean':float(b.mean()),'mean_delta':float(d.mean()),
            'benefit_fraction':float((d>1e-12).mean()),'harm_fraction':float((d<-1e-12).mean()),
            'worst_delta':float(d.min()),'best_delta':float(d.max())}

def parse_diag(path):
    if not path or not Path(path).exists(): return []
    rows=[]
    for line in Path(path).read_text(errors='ignore').splitlines():
        if 'M1_DIAG:' not in line: continue
        r={}
        for k,v in re.findall(r'([A-Za-z0-9_]+)=([-+0-9.eE]+)',line):
            try:r[k]=float(v)
            except:pass
        if r:rows.append(r)
    return rows

def parse_test(path):
    out={}
    if not path or not Path(path).exists(): return out
    text=Path(path).read_text(errors='ignore')
    m=re.findall(r'Deployment change summary: changed_cases=(\d+)/(\d+) \(([^)]+)\), mean_changed_pixel_fraction=([-+0-9.eE]+)',text)
    if m:
        x,n,r,p=m[-1]; out.update(changed_cases=int(x),total_cases=int(n),changed_case_fraction=float(r),mean_changed_pixel_fraction=float(p))
    m=re.findall(r'effective_MC=(\d+)',text)
    if m: out['effective_mc']=int(m[-1])
    return out

def load_json(path):
    return json.loads(Path(path).read_text()) if path and Path(path).exists() else {}

def trend(rows,key):
    v=[r[key] for r in rows if key in r and np.isfinite(r[key])]
    if not v:return {}
    return {'first':v[0],'last':v[-1],'mean':float(np.mean(v)),'max':float(np.max(v)),'positive_fraction':float(np.mean(np.asarray(v)>0))}

def pct(x):return f'{100*x:.2f}%'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',required=True,choices=['BUSI','Kvasir','ISIC','BTMRI'])
    for n in ('matched-base-legacy','joint-base-legacy','final-legacy','matched-base-true2d','joint-base-true2d','final-true2d'):
        p.add_argument('--'+n,required=True)
    p.add_argument('--train-log',default=''); p.add_argument('--test-log',default='')
    p.add_argument('--strength-oracle-legacy',default=''); p.add_argument('--strength-oracle-true2d',default='')
    p.add_argument('--base-hash-audit',default='')
    p.add_argument('--val-selection',default='')
    p.add_argument('--output-prefix',required=True)
    a=p.parse_args()
    ml,jl,fl=align(load_csv(a.matched_base_legacy),load_csv(a.joint_base_legacy),load_csv(a.final_legacy))
    mt,jt,ft=align(load_csv(a.matched_base_true2d),load_csv(a.joint_base_true2d),load_csv(a.final_true2d))
    rows=parse_diag(a.train_log); last=rows[-1] if rows else {}
    train_text=(Path(a.train_log).read_text(errors='ignore')
                if a.train_log and Path(a.train_log).exists() else '')
    fixed_host_mode='[JBT_V636_FIXED_BASE_REFERENCE]' in train_text
    or_l=load_json(a.strength_oracle_legacy); or_t=load_json(a.strength_oracle_true2d)
    hash_audit=load_json(a.base_hash_audit)
    val_selection=load_json(a.val_selection)
    report={'dataset':a.dataset,'cases':len(ml),'paper_legacy':{},'true2d':{},'last_train_diag':last,
            'test_diag':parse_test(a.test_log),'strength_oracle_legacy':or_l,'strength_oracle_true2d':or_t,'base_hash_audit':hash_audit,
            'val_selection':val_selection}
    for proto,mb,jb,fn in [('paper_legacy',ml,jl,fl),('true2d',mt,jt,ft)]:
        for metric in ('DSC','NSD'):
            report[proto][metric]={'base_drift':stat(mb[metric],jb[metric]),'module_gain':stat(jb[metric],fn[metric]),'net_gain':stat(mb[metric],fn[metric])}
    paper=PAPER.get(a.dataset)
    if paper: report['paper_margin']={m:float(fl[m].mean()-paper[m]) for m in ('DSC','NSD')}
    drift=report['paper_legacy']['DSC']['base_drift']['mean_delta']; gain=report['paper_legacy']['DSC']['module_gain']['mean_delta']; n2=report['true2d']['NSD']['module_gain']['mean_delta']
    oracle_dsc=float(or_l.get('oracle',{}).get('DSC',{}).get('case_oracle_gain',0.0))
    realization=float(or_l.get('oracle',{}).get('DSC',{}).get('oracle_realization_ratio_global',0.0))
    # v6.3.6 fixed-host recovery and from-scratch joint training have different
    # applicable contracts.  The old diagnostic incorrectly required a live
    # Base optimizer and MC3 even when Base is intentionally immutable, then
    # also treated a heuristic amplitude ratio as a hard correctness flag.
    mechanics={
      ('exact_fixed_host_optimizer' if fixed_host_mode else 'exact_base_dual_optimizer'):
          (('[JBT_V636_FIXED_BASE_OPTIMIZER]' in train_text and
            "('official_base_exact', 0.0)" in train_text and
            "('jbt_aux_exact', 0.0004)" in train_text)
           if fixed_host_mode else
           float(last.get('jbt_v6_separate_base_optimizer',0.0))>0.5),
      'base_init_hash_match':bool(hash_audit.get('init_match',False)) if hash_audit else True,
      'base_full_trajectory_match':bool(hash_audit.get('all_epoch_hashes_match',False)) if hash_audit else False,
      'base24_aux_microbatch':float(last.get('jbt_v63_exact_base_aux_microbatch',0))>0.5,
      'base_gradient_isolated':abs(float(last.get('v470_aux_to_base_grad_scale',999)))<1e-9,
      ('fixed_host_single_posterior_expected' if fixed_host_mode else 'train_mc3'):
          (abs(float(last.get('jbt_train_posterior_sample_count',-1)))<1e-9
           if fixed_host_mode else
           float(last.get('jbt_train_posterior_sample_count',0))>=2.5),
      'direct_signed_flow':float(last.get('jbt_v6_direct_signed_flow',0))>0.5,
      'candidate_specific_utility':float(last.get('jbt_v6_candidate_utility_enabled',0))>0.5,
      'flow_alive':float(last.get('geotr_m1_flow_rms_px',0))>1e-3,
      'signed_target_present':float(last.get('jbt_v6_signed_disp_owner_fraction',0))>0,
      'normal_ray_target':float(last.get('jbt_v62_normal_ray_target_enabled',0))>0.5,
      'normal_transport':abs(float(last.get('geotr_m1_tangent_energy_ratio',1)))<1e-2,
      'expected_gain_selector':float(last.get('jbt_v63_enabled',0))>0.5,
      'active_range_preserved':float(last.get('geotr_m1_range_violation_fraction',1))<1e-3,
    }
    # Screening heuristics only; not significance thresholds.
    report['mechanics_mode']='fixed_host' if fixed_host_mode else 'joint'
    report['mechanics']=mechanics
    report['amplitude_diagnostic']={
      'active_flow_to_scalar_ratio':float(last.get('jbt_v62_active_flow_to_scalar_ratio',0)),
      'legacy_heuristic_threshold':0.70,
      'legacy_heuristic_pass':float(last.get('jbt_v62_active_flow_to_scalar_ratio',0))>=0.70,
      'hard_contract':False,
    }
    report['screening']={
      'base_parity_0p2pp':abs(drift)<=0.002,
      'candidate_oracle_at_least_1pp':oracle_dsc>=0.01,
      'module_dsc_at_least_0p4pp':gain>=0.004,
      'true2d_nsd_at_least_2pp':n2>=0.02,
      'mechanics_pass':all(mechanics.values()),
      'oracle_realization_global':realization,
    }
    pref=Path(a.output_prefix); pref.parent.mkdir(parents=True,exist_ok=True)
    pref.with_suffix('.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    lines=[f'# JBT-v6.3 自动诊断：{a.dataset}','',f'- 病例数：{len(ml)}','',
      '| 协议/指标 | Matched Base | Joint BaseNative | JBT-v6.3 Final | Base漂移 | 模块增益 | 净增益 |','|---|---:|---:|---:|---:|---:|---:|']
    for proto,mb,jb,fn in [('paper_legacy',ml,jl,fl),('true2d',mt,jt,ft)]:
        for m in ('DSC','NSD'):
            r=report[proto][m]; lines.append(f'| {proto}/{m} | {pct(mb[m].mean())} | {pct(jb[m].mean())} | {pct(fn[m].mean())} | {pct(r["base_drift"]["mean_delta"])} | {pct(r["module_gain"]["mean_delta"])} | {pct(r["net_gain"]["mean_delta"])} |')
    if paper:
        lines += ['', '## 与论文 paper_legacy（仅同协议比较）','',f'- Published DSC {pct(paper["DSC"])}；Final margin {pct(report["paper_margin"]["DSC"])}',f'- Published NSD {pct(paper["NSD"])}；Final margin {pct(report["paper_margin"]["NSD"])}']
    lines += ['', '## 机制合同','',
              f'- mode: {report["mechanics_mode"]}']+[f'- {k}: {v}' for k,v in mechanics.items()]
    amp=report['amplitude_diagnostic']
    lines += ['', '## 幅度诊断（非硬合同）','',
              f'- active_flow_to_scalar_ratio: {amp["active_flow_to_scalar_ratio"]:.6f}',
              f'- legacy threshold 0.70 passed: {amp["legacy_heuristic_pass"]}',
              '- 该比率是观察性启发指标；只要范围、梯度、候选能力与Val/Test收益正常，不应单独令机制合同失败。']
    if hash_audit:
        lines += ['', '## Protected Base/PVL hash parity','',
                  f'- init_match: {hash_audit.get("init_match")}',
                  f'- step1_match: {hash_audit.get("step1_match")}',
                  f'- all_epoch_hashes_match: {hash_audit.get("all_epoch_hashes_match")}',
                  f'- mismatch_epochs: {hash_audit.get("mismatch_epochs",[])}']
    if val_selection:
        lines += ['', '## Validation锁定的部署后备','',
                  f'- selection rule: {val_selection.get("selection_rule")}',
                  f'- selected strength: {val_selection.get("selected_strength")}',
                  f'- Val DSC gain: {100*float(val_selection.get("selected_val_dsc_gain",0)):+.3f} pp',
                  f'- Val NSD gain: {100*float(val_selection.get("selected_val_nsd_gain",0)):+.3f} pp']
    if or_l:
        o=or_l['oracle']['DSC']; lines += ['', '## Candidate capacity / selector','',f'- paper_legacy DSC CaseOracle gain: {100*o["case_oracle_gain"]:+.3f} pp',f'- Final module gain: {100*o["final_gain"]:+.3f} pp',f'- Oracle-benefit cases: {100*o["oracle_benefit_fraction"]:.1f}%',f'- Global Oracle realization: {100*o.get("oracle_realization_ratio_global",0):.1f}%']
    keys=['jbt_v63_exact_base_aux_microbatch','jbt_v63_aux_microbatch_size','v471_candidate_ratio_effective','v470_aux_to_base_grad_scale','jbt_train_posterior_sample_count','mhcs_final_gain','geotr_m1_flow_rms_px','geotr_m1_flow_max_px','jbt_v6_scalar_abs_mean','jbt_v6_direction_abs_mean','jbt_v6_error_selected_support_fraction','jbt_v5_broad_support_fraction','jbt_v5_effective_support_fraction','jbt_v62_actual_support_mean','jbt_v62_active_flow_to_scalar_ratio','jbt_v6_signed_displacement_loss','jbt_v6_signed_disp_target_abs_mean','jbt_v6_signed_disp_pred_abs_mean','jbt_v6_signed_disp_direction_accuracy','jbt_v6_signed_disp_owner_fraction','jbt_v62_normal_ray_hit_fraction','jbt_v62_curriculum_stage','jbt_v62_effective_utility_weight','jbt_v62_effective_capacity_weight','jbt_v62_effective_displacement_weight','jbt_v6_utility_classification_loss','jbt_v6_utility_regression_loss','jbt_v6_utility_ranking_loss','jbt_v62_utility_preserve_ranking_loss','jbt_v62_gain_mean_mae','jbt_v62_gain_sigma_mean','jbt_v62_lcb_score_mean','jbt_v5_oracle_capacity_gain','jbt_v5_utility_selected_gain','jbt_v5_utility_accept_rate','geotr_m1_range_violation_fraction']
    if last:
        lines += ['', '## 最后一轮训练诊断','']
        for k in keys:
            if k in last: lines.append(f'- `{k}` = {last[k]:.8g}')
        tg=trend(rows,'mhcs_final_gain')
        if tg: lines += ['',f'- train final-gain mean: {tg["mean"]:.8g}; positive epoch fraction: {tg["positive_fraction"]:.3f}']
    sc=report['screening']; lines += ['', '## 20轮结构筛选（不是统计显著性标准）','',f'- Base parity |drift|<=0.2pp: {sc["base_parity_0p2pp"]}',f'- CaseOracle DSC >= +1.0pp: {sc["candidate_oracle_at_least_1pp"]}',f'- Final module DSC >= +0.4pp: {sc["module_dsc_at_least_0p4pp"]}',f'- true2d NSD >= +2.0pp: {sc["true2d_nsd_at_least_2pp"]}',f'- mechanics PASS: {sc["mechanics_pass"]}']
    lines += ['', '> 20轮只用于结构筛选。测试集已被多次查看，后续不要继续按 Test 阈值调参；锁定结构后应以 Val 调参，再做100轮、多seed和配对统计。']
    pref.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(pref.with_suffix('.md')); print(pref.with_suffix('.json'))

if __name__=='__main__': main()
