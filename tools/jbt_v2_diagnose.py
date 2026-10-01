#!/usr/bin/env python3
"""Automatic JBT-v2 paired diagnostic report.

Reads only already-produced per-case CSVs and logs. It does not touch masks,
model weights or checkpoint selection. The report separates:
  1) matched Base20 -> joint-checkpoint BaseNative (Base trajectory drift),
  2) joint-checkpoint BaseNative -> JBT Final (module gain),
  3) matched Base20 -> JBT Final (net system gain),
and compares paper_legacy metrics with the published reference numbers.
"""
from __future__ import annotations
import argparse, json, math, re
from pathlib import Path
import numpy as np
import pandas as pd

PAPER = {
    "BUSI": {"DSC": 0.8572, "NSD": 0.8835},
    "Kvasir": {"DSC": 0.9015, "NSD": 0.9232},
}


def args():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=['BUSI','Kvasir','ISIC','BTMRI'])
    p.add_argument('--matched-base-legacy', required=True)
    p.add_argument('--joint-base-legacy', required=True)
    p.add_argument('--final-legacy', required=True)
    p.add_argument('--matched-base-true2d', required=True)
    p.add_argument('--joint-base-true2d', required=True)
    p.add_argument('--final-true2d', required=True)
    p.add_argument('--train-log', default='')
    p.add_argument('--test-log', default='')
    p.add_argument('--output-prefix', required=True)
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
        'a_mean':float(a.mean()), 'b_mean':float(b.mean()), 'mean_delta':float(d.mean()),
        'median_delta':float(np.median(d)), 'benefit_cases':int((d>1e-12).sum()),
        'harm_cases':int((d<-1e-12).sum()), 'equal_cases':int((np.abs(d)<=1e-12).sum()),
        'worst_delta':float(d.min()), 'best_delta':float(d.max()),
    }


def parse_last_diag(path):
    if not path or not Path(path).exists(): return {}
    text=Path(path).read_text(errors='ignore')
    lines=[x for x in text.splitlines() if 'M1_DIAG:' in x]
    if not lines: return {}
    line=lines[-1]
    out={}
    for key,val in re.findall(r'([A-Za-z0-9_]+)=([-+0-9.eE]+)', line):
        try: out[key]=float(val)
        except Exception: pass
    return out


def parse_test(path):
    if not path or not Path(path).exists(): return {}
    text=Path(path).read_text(errors='ignore')
    out={}
    m=re.findall(r'Deployment change summary: changed_cases=(\d+)/(\d+) \(([^)]+)\), mean_changed_pixel_fraction=([-+0-9.eE]+)',text)
    if m:
        a,b,r,f=m[-1]; out={'changed_cases':int(a),'total_cases':int(b),'changed_case_fraction':float(r),'mean_changed_pixel_fraction':float(f)}
    m2=re.findall(r'effective_MC=(\d+)',text)
    if m2: out['effective_mc']=int(m2[-1])
    return out


def pct(x): return f'{100*x:.2f}%'

def main():
    a=args()
    mb_l,jb_l,fn_l=align(load(a.matched_base_legacy),load(a.joint_base_legacy),load(a.final_legacy))
    mb_t,jb_t,fn_t=align(load(a.matched_base_true2d),load(a.joint_base_true2d),load(a.final_true2d))
    report={'dataset':a.dataset,'cases':len(mb_l),'paper_legacy':{},'true2d':{},'train_diag_last':parse_last_diag(a.train_log),'test_diag':parse_test(a.test_log)}
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

    # Diagnostic gates are screening criteria, not claims of statistical significance.
    dsc_drift=report['paper_legacy']['DSC']['base_drift']['mean_delta']
    dsc_gain=report['paper_legacy']['DSC']['module_gain']['mean_delta']
    nsd2_gain=report['true2d']['NSD']['module_gain']['mean_delta']
    report['screening']={
        'base_preserved_within_0p2pp': bool(dsc_drift >= -0.002),
        'module_dsc_gain_at_least_0p4pp': bool(dsc_gain >= 0.004),
        'module_true2d_nsd_gain_at_least_2pp': bool(nsd2_gain >= 0.02),
        'ready_for_100ep_screen': bool(dsc_drift >= -0.002 and dsc_gain >= 0.004 and nsd2_gain >= 0.02),
    }
    if paper:
        report['screening']['final_exceeds_published_dsc_at_20ep']=bool(report['paper_margin']['DSC']>0)
        report['screening']['final_exceeds_published_legacy_nsd_at_20ep']=bool(report['paper_margin']['NSD']>0)

    prefix=Path(a.output_prefix); prefix.parent.mkdir(parents=True,exist_ok=True)
    jpath=prefix.with_suffix('.json'); mpath=prefix.with_suffix('.md')
    jpath.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    lines=[f'# JBT-v2 自动诊断：{a.dataset}','',f'- 病例数：{len(mb_l)}','',
           '| 协议/指标 | Matched Base20 | Joint BaseNative | JBT Final | Base漂移 | JBT模块增益 | 净增益 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for proto,mb,jb,fn in [('paper_legacy',mb_l,jb_l,fn_l),('true2d',mb_t,jb_t,fn_t)]:
        for metric in ('DSC','NSD'):
            bd=report[proto][metric]['base_drift']['mean_delta']; mg=report[proto][metric]['module_gain']['mean_delta']; ng=report[proto][metric]['net_gain']['mean_delta']
            lines.append(f'| {proto}/{metric} | {pct(mb[metric].mean())} | {pct(jb[metric].mean())} | {pct(fn[metric].mean())} | {pct(bd)} | {pct(mg)} | {pct(ng)} |')
    if paper:
        lines += ['', '## 与论文 paper_legacy 数值', '',
                  f"- Published DSC: {pct(paper['DSC'])}; JBT-v2: {pct(fn_l['DSC'].mean())}; margin: {pct(report['paper_margin']['DSC'])}",
                  f"- Published NSD: {pct(paper['NSD'])}; JBT-v2: {pct(fn_l['NSD'].mean())}; margin: {pct(report['paper_margin']['NSD'])}"]
    sc=report['screening']
    lines += ['', '## 20轮筛选结论', '',
              f"- Base 保持（漂移 >= -0.2pp）：{sc['base_preserved_within_0p2pp']}",
              f"- JBT DSC 增益 >= +0.4pp：{sc['module_dsc_gain_at_least_0p4pp']}",
              f"- JBT true2d NSD 增益 >= +2.0pp：{sc['module_true2d_nsd_gain_at_least_2pp']}",
              f"- 建议进入100轮：{sc['ready_for_100ep_screen']}"]
    td=report['train_diag_last']
    if td:
        keys=['v470_aux_to_base_grad_scale','geotr_m1_flow_rms_px','geotr_m1_flow_max_px','geotr_m1_geometry_abs_change','geotr_m1_feature_feedback_abs','geotr_m1_feature_feedback_logit_change','geotr_m1_posterior_evidence_mean','geotr_m1_deadzone_fraction','geotr_m1_tangent_energy_ratio']
        lines += ['', '## 最后一轮训练诊断', '']
        for k in keys:
            if k in td: lines.append(f'- `{k}` = {td[k]:.8g}')
    ts=report['test_diag']
    if ts:
        lines += ['', '## 测试部署诊断', ''] + [f'- `{k}` = {v}' for k,v in ts.items()]
    lines += ['', '> 20轮只是结构筛选；是否“显著优于论文”必须由100轮/多seed与配对统计共同支持。']
    mpath.write_text('\n'.join(lines)+'\n')
    print(mpath); print(jpath)

if __name__=='__main__': main()
