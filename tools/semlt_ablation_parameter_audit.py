#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
from types import SimpleNamespace as NS
import yaml, torch
from utils.geotr_m1_transport import ExactGeometryTransportSegmenter

MAP={"operator":"GEOTR_M1_OPERATOR","transport_space":"GEOTR_M1_TRANSPORT_SPACE","support_ratio":"GEOTR_M1_UNIFIED_SUPPORT_RATIO","tangent_ratio":"GEOTR_M1_UNIFIED_TANGENT_RATIO"}
def ns(x):
    if isinstance(x,dict): return NS(**{k:ns(v) for k,v in x.items()})
    if isinstance(x,list): return [ns(v) for v in x]
    return x

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/SEMLT_ABLATION_FROZEN100.yaml'); ap.add_argument('--spec',default='configs/SEMLT_ABLATION_LOCKED_VARIANTS.json'); ap.add_argument('--output',default='audits/SEMLT_ABLATION_parameter_audit.json'); a=ap.parse_args()
    base=ns(yaml.safe_load(Path(a.config).read_text())); spec=json.loads(Path(a.spec).read_text())
    rows=[]
    for i,(v,item) in enumerate(spec['variants'].items()):
        cfg=copy.deepcopy(base)
        for k,val in spec['fixed_overrides'].items(): setattr(getattr(cfg,k.split('.')[0]),k.split('.')[1],val)
        for src,dst in MAP.items(): setattr(cfg.M1,dst,item[src])
        torch.manual_seed(12345)
        m=ExactGeometryTransportSegmenter(cfg)
        total=sum(p.numel() for p in m.parameters()); trainable=sum(p.numel() for p in m.parameters() if p.requires_grad)
        rows.append({'variant':v,'paper_name':item['paper_name'],'parameters':total,'trainable_parameters':trainable,'operator':item['operator'],'transport_space':item['transport_space']})
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps({'protocol':spec['protocol'],'rows':rows},indent=2)+'\n')
    for r in rows: print(f"{r['variant']}: {r['trainable_parameters']:,} trainable")
    print(out)
if __name__=='__main__': main()
