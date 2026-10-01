#!/usr/bin/env python3
"""Automatic JBT-v3 paired diagnostic report.

The report separates matched Base trajectory, joint-Base drift, module gain,
and net gain.  It also audits the v3 mechanics that directly address the v2
failure modes: full branch loss, posterior-domain identity, selective error
gating, bounded feature-logit feedback, and range-preserving normal transport.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd

PAPER={
    'BUSI':{'DSC':0.8572,'NSD':0.8835},
    'Kvasir':{'DSC':0.9015,'NSD':0.9232},
}

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',required=True,choices=['BUSI','Kvasir','ISIC','BTMRI'])
    p.add_argument('--matched-base-legacy',required=True)
    p.add_argument('--joint-base-legacy',required=True)
    p.add_argument('--final-legacy',required=True)
    p.add_argument('--matched-base-true2d',required=True)
    p.add_argument('--joint-base-true2d',required=True)
    p.add_argument('--final-true2d',required=True)
    p.add_argument('--train-log',default='')
    p.add_argument('--test-log',default='')
    p.add_argument('--output-prefix',required=True)
    return p.parse_args()

def load(path):
    f=pd.read_csv(path)
    need={'Case_ID','DSC','NSD'}
    miss=need-set(f.columns)
    if miss: raise ValueError(f'{path} missing {sorted(miss)}')
    if f.Case_ID.duplicated().any(): raise ValueError(f'{path} duplicate Case_ID')
    return f.set_index('Case_ID').sort_index()

def align(*frames):
    ids=set(frames[0].index)
    for f in frames[1:]:
        if set(f.index)!=ids: raise ValueError('CSV case sets differ')
    order=sorted(ids)
    return [f.loc[order] for f in frames]

def stat(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float); d=b-a
    return {
        'a_mean':float(a.mean()),'b_mean':float(b.mean()),'mean_delta':float(d.mean()),
        'median_delta':float(np.median(d)),'benefit_cases':int((d>1e-12).sum()),
        'harm_cases':int((d<-1e-12).sum()),'equal_cases':int((np.abs(d)<=1e-12).sum()),
        'benefit_fraction':float((d>1e-12).mean()),'harm_fraction':float((d<-1e-12).mean()),
        'worst_delta':float(d.min()),'best_delta':float(d.max()),
    }

def parse_diag_series(path):
    if not path or not Path(path).exists(): return []
    rows=[]
    for line in Path(path).read_text(errors='ignore').splitlines():
        if 'M1_DIAG:' not in line: continue
        row={}
        for key,val in re.findall(r'([A-Za-z0-9_]+)=([-+0-9.eE]+)',line):
            try: row[key]=float(val)
            except Exception: pass
        if row: rows.append(row)
    return rows

def parse_test(path):
    if not path or not Path(path).exists(): return {}
    text=Path(path).read_text(errors='ignore')
    out={}
    m=re.findall(r'Deployment change summary: changed_cases=(\d+)/(\d+) \(([^)]+)\), mean_changed_pixel_fraction=([-+0-9.eE]+)',text)
    if m:
        a,b,r,f=m[-1]
        out={'changed_cases':int(a),'total_cases':int(b),'changed_case_fraction':float(r),'mean_changed_pixel_fraction':float(f)}
    m2=re.findall(r'effective_MC=(\d+)',text)
    if m2: out['effective_mc']=int(m2[-1])
    return out

def summarize_series(rows):
    if not rows: return {}
    keys={k for r in rows for k in r}
    out={}
    for k in sorted(keys):
        vals=[r[k] for r in rows if k in r and np.isfinite(r[k])]
        if not vals: continue
        out[k]={
            'first':float(vals[0]),'last':float(vals[-1]),'mean':float(np.mean(vals)),
            'min':float(np.min(vals)),'max':float(np.max(vals)),
        }
        if 'gain' in k:
            out[k]['positive_fraction']=float(np.mean(np.asarray(vals)>0))
    return out

def pct(x): return f'{100*x:.2f}%'

def main():
    a=parse_args()
    mb_l,jb_l,fn_l=align(load(a.matched_base_legacy),load(a.joint_base_legacy),load(a.final_legacy))
    mb_t,jb_t,fn_t=align(load(a.matched_base_true2d),load(a.joint_base_true2d),load(a.final_true2d))
    rows=parse_diag_series(a.train_log)
    trend=summarize_series(rows)
    report={'dataset':a.dataset,'cases':len(mb_l),'paper_legacy':{},'true2d':{},
            'train_diag_trend':trend,'train_diag_last':rows[-1] if rows else {},
            'test_diag':parse_test(a.test_log)}
    for metric in ('DSC','NSD'):
        report['paper_legacy'][metric]={
            'base_drift':stat(mb_l[metric],jb_l[metric]),
            'module_gain':stat(jb_l[metric],fn_l[metric]),
            'net_gain':stat(mb_l[metric],fn_l[metric]),
        }
        report['true2d'][metric]={
            'base_drift':stat(mb_t[metric],jb_t[metric]),
            'module_gain':stat(jb_t[metric],fn_t[metric]),
            'net_gain':stat(mb_t[metric],fn_t[metric]),
        }
    paper=PAPER.get(a.dataset)
    if paper:
        report['published_reference']=paper
        report['paper_margin']={m:float(fn_l[m].mean()-paper[m]) for m in ('DSC','NSD')}

    dsc_drift=report['paper_legacy']['DSC']['base_drift']['mean_delta']
    dsc_gain=report['paper_legacy']['DSC']['module_gain']['mean_delta']
    nsd2_gain=report['true2d']['NSD']['module_gain']['mean_delta']
    last=report['train_diag_last']
    mechanics={
        'full_branch_loss_active': float(last.get('v471_candidate_ratio_effective',0.0))>=0.95,
        'train_mean_then_refine_mc3': float(last.get('jbt_train_posterior_sample_count',0.0))>=2.5,
        'normal_transport': abs(float(last.get('geotr_m1_tangent_energy_ratio',1.0)))<1e-4,
        'active_range_preserved': float(last.get('geotr_m1_range_violation_fraction',1.0))<1e-3,
        'bounded_feature_delta': float(last.get('geotr_m1_feature_feedback_logit_change_max',999.0))<=1.251,
    }
    report['mechanics']=mechanics
    report['screening']={
        'base_preserved_within_0p2pp':bool(dsc_drift>=-0.002),
        'module_dsc_gain_at_least_0p4pp':bool(dsc_gain>=0.004),
        'module_true2d_nsd_gain_at_least_2pp':bool(nsd2_gain>=0.02),
        'mechanics_pass':bool(all(mechanics.values())),
    }
    report['screening']['ready_for_100ep_screen']=bool(
        report['screening']['base_preserved_within_0p2pp'] and
        report['screening']['module_dsc_gain_at_least_0p4pp'] and
        report['screening']['module_true2d_nsd_gain_at_least_2pp'] and
        report['screening']['mechanics_pass']
    )

    prefix=Path(a.output_prefix); prefix.parent.mkdir(parents=True,exist_ok=True)
    jpath=prefix.with_suffix('.json'); mpath=prefix.with_suffix('.md')
    jpath.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    lines=[f'# JBT-v3 自动诊断：{a.dataset}','',f'- 病例数：{len(mb_l)}','',
           '| 协议/指标 | Matched Base20 | Joint BaseNative | JBT-v3 Final | Base漂移 | 模块增益 | 净增益 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for proto,mb,jb,fn in [('paper_legacy',mb_l,jb_l,fn_l),('true2d',mb_t,jb_t,fn_t)]:
        for metric in ('DSC','NSD'):
            r=report[proto][metric]
            lines.append(f'| {proto}/{metric} | {pct(mb[metric].mean())} | {pct(jb[metric].mean())} | {pct(fn[metric].mean())} | {pct(r["base_drift"]["mean_delta"])} | {pct(r["module_gain"]["mean_delta"])} | {pct(r["net_gain"]["mean_delta"])} |')
    if paper:
        lines += ['', '## 与论文 paper_legacy 数值', '',
                  f'- Published DSC: {pct(paper["DSC"])}; JBT-v3: {pct(fn_l["DSC"].mean())}; margin: {pct(report["paper_margin"]["DSC"])}',
                  f'- Published NSD: {pct(paper["NSD"])}; JBT-v3: {pct(fn_l["NSD"].mean())}; margin: {pct(report["paper_margin"]["NSD"])}']
    lines += ['', '## 结构/训练合同', '']
    for k,v in mechanics.items(): lines.append(f'- {k}: {v}')
    sc=report['screening']
    lines += ['', '## 20轮筛选结论', '',
              f'- Base 保持（漂移 >= -0.2pp）：{sc["base_preserved_within_0p2pp"]}',
              f'- JBT-v3 DSC 增益 >= +0.4pp：{sc["module_dsc_gain_at_least_0p4pp"]}',
              f'- JBT-v3 true2d NSD 增益 >= +2.0pp：{sc["module_true2d_nsd_gain_at_least_2pp"]}',
              f'- mechanics PASS：{sc["mechanics_pass"]}',
              f'- 建议进入100轮：{sc["ready_for_100ep_screen"]}']
    if rows:
        keys=['v471_candidate_ratio_effective','v470_aux_to_base_grad_scale','jbt_train_posterior_sample_count',
              'mhcs_final_gain','geotr_m1_flow_rms_px','geotr_m1_flow_max_px',
              'geotr_m1_feature_feedback_logit_change','geotr_m1_feature_feedback_logit_change_max',
              'geotr_m1_posterior_reconstruction_mismatch','geotr_m1_error_gate_fraction',
              'jbt_error_supervision_loss','jbt_nondegrade_loss','geotr_m1_posterior_evidence_mean',
              'geotr_m1_deadzone_fraction','geotr_m1_tangent_energy_ratio','geotr_m1_range_violation_fraction']
        lines += ['', '## 最后一轮训练诊断', '']
        for k in keys:
            if k in last: lines.append(f'- `{k}` = {last[k]:.8g}')
        if 'mhcs_final_gain' in trend:
            g=trend['mhcs_final_gain']
            lines += ['', f'- 20轮 train final-gain 均值：{g["mean"]:.8g}',
                      f'- train final-gain 为正的 epoch 比例：{g.get("positive_fraction",0):.3f}']
    ts=report['test_diag']
    if ts:
        lines += ['', '## 测试部署诊断', '']+[f'- `{k}` = {v}' for k,v in ts.items()]
    lines += ['', '> 20轮只用于结构筛选。100轮、多 seed 和配对统计才用于论文主结论；不要用同一 test set 继续反复调阈值。']
    mpath.write_text('\n'.join(lines)+'\n')
    print(mpath); print(jpath)

if __name__=='__main__': main()
