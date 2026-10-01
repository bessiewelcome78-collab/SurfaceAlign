#!/usr/bin/env python3
"""Fail-closed contract for the locked SemLT A0-A6 causal ablation."""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
from types import SimpleNamespace as NS
import torch, yaml
from utils.geotr_m1_transport import ExactGeometryTransportSegmenter
from utils.geotr_m1_loss import compute_geotr_m1_loss

PAIRWISE_EXPECTED = {
    ("A4_FREE2D","A1_REWRITE"): {"M1.GEOTR_M1_OPERATOR"},
    ("A4_FREE2D","A2_PROB"): {"M1.GEOTR_M1_TRANSPORT_SPACE"},
    ("A4_FREE2D","A3_NORMAL1D"): {"M1.GEOTR_M1_OPERATOR"},
    ("A5_SUPPORT","A4_FREE2D"): {"M1.GEOTR_M1_UNIFIED_SUPPORT_RATIO"},
    ("A6_SEMLT","A5_SUPPORT"): {"M1.GEOTR_M1_UNIFIED_TANGENT_RATIO"},
}
VARIANT_KEYS={
    "operator":"M1.GEOTR_M1_OPERATOR",
    "transport_space":"M1.GEOTR_M1_TRANSPORT_SPACE",
    "support_ratio":"M1.GEOTR_M1_UNIFIED_SUPPORT_RATIO",
    "tangent_ratio":"M1.GEOTR_M1_UNIFIED_TANGENT_RATIO",
}

def ns(x):
    if isinstance(x,dict): return NS(**{k:ns(v) for k,v in x.items()})
    if isinstance(x,list): return [ns(v) for v in x]
    return x

def set_path(data,key,value):
    a,b=key.split('.',1); setattr(getattr(data,a),b,value)

def merged_overrides(spec,variant):
    out=dict(spec['fixed_overrides']); item=spec['variants'][variant]
    for src,dst in VARIANT_KEYS.items(): out[dst]=item[src]
    return out

def cfg_for(base_cfg,spec,variant):
    c=copy.deepcopy(base_cfg)
    for k,v in merged_overrides(spec,variant).items(): set_path(c,k,v)
    return c

def flat_relevant(spec,variant):
    return merged_overrides(spec,variant)

def assert_config(cfg):
    m1=cfg.M1; tr=cfg.TRAIN
    assert bool(m1.GEOTR_M1_CAUSAL_ABLATION)
    assert bool(m1.V469_FREEZE_BASE)
    assert bool(m1.GEOTR_M1_DETACH_CONDITIONERS)
    assert not bool(m1.GEOTR_M1_USE_SEMANTIC_CONDITIONING)
    assert not bool(m1.GEOTR_M1_USE_TEXT_CONDITIONING)
    assert bool(m1.GEOTR_M1_USE_ANCHOR_CUES)
    assert str(m1.GEOTR_M1_GATE_MODE).lower()=="none"
    assert str(m1.GEOTR_M1_DEFORM_MODE).lower()=="unified"
    assert float(m1.GEOTOPO_SMOOTHNESS_WEIGHT)==0.0
    assert float(m1.GEOTR_M1_FOLDING_WEIGHT)==0.001
    assert float(m1.GEOTR_M1_UNIFIED_WEIGHT)==0.001
    assert float(m1.GEOTR_M1_UNIFIED_SMOOTH_RATIO)==0.0
    assert float(m1.M2_LEARNING_RATE)==0.0 and float(m1.M3_LEARNING_RATE)==0.0
    assert float(m1.M2_LOSS_WEIGHT)==0.0
    assert int(tr.NUM_EPOCHS)==100 and int(tr.SCHEDULER_TOTAL_EPOCHS)==100
    assert int(tr.BATCH_SIZE)*int(tr.GRAD_ACCUMULATION_STEPS)==24
    assert not bool(tr.USE_VALIDATION_SELECTION)
    assert bool(tr.DETERMINISTIC)

def assert_single_variable_contrasts(spec):
    for (left,right),expected in PAIRWISE_EXPECTED.items():
        a=flat_relevant(spec,left); b=flat_relevant(spec,right)
        diff={k for k in sorted(set(a)|set(b)) if a.get(k)!=b.get(k)}
        assert diff==expected, (left,right,diff,expected)

def assert_identity_and_gradients(base_cfg,spec):
    for i,variant in enumerate(spec['variants']):
        torch.manual_seed(20260829+i)
        cfg=cfg_for(base_cfg,spec,variant)
        model=ExactGeometryTransportSegmenter(cfg)
        base=torch.randn(2,1,20,20,requires_grad=True)
        image=torch.randn(2,3,20,20,requires_grad=True)
        semantic=torch.randn(2,512,10,10,requires_grad=True)
        text=torch.randn(2,512,requires_grad=True)
        _,aux=model.generate(base,image,semantic,text)
        # Every variant is zero-initialized to the exact same factual Base output.
        assert torch.allclose(aux['geotopo_final_logits'],base.detach(),atol=1e-5,rtol=1e-5), variant
        with torch.no_grad():
            model.mean_head.flow_out.weight.normal_(0.0,0.02)
        _,aux=model.generate(base,image,semantic,text)
        target=torch.randint(0,2,(2,20,20)).float()
        loss,_=compute_geotr_m1_loss(cfg,aux['candidates'],target,aux)
        assert torch.isfinite(loss), variant
        loss.backward()
        assert base.grad is None, variant
        assert semantic.grad is None, variant
        assert text.grad is None, variant
        assert image.grad is not None and float(image.grad.abs().sum())>0, variant
        g=model.mean_head.flow_out.weight.grad
        assert g is not None and float(g.abs().sum())>0, variant

def assert_identical_initialization_for_same_arch(base_cfg,spec):
    # A2/A4/A5/A6 have identical architecture; same seed must yield byte-identical
    # initial parameters. Thus their only interventions are forward/loss semantics.
    variants=['A2_PROB','A4_FREE2D','A5_SUPPORT','A6_SEMLT']
    states=[]
    for v in variants:
        torch.manual_seed(777)
        model=ExactGeometryTransportSegmenter(cfg_for(base_cfg,spec,v))
        states.append({k:t.detach().cpu().clone() for k,t in model.state_dict().items()})
    for s in states[1:]:
        assert s.keys()==states[0].keys()
        for k in s: assert torch.equal(s[k],states[0][k]), (k,variants)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='configs/SEMLT_ABLATION_FROZEN100.yaml')
    ap.add_argument('--spec',default='configs/SEMLT_ABLATION_LOCKED_VARIANTS.json')
    a=ap.parse_args(); cp=Path(a.config); sp=Path(a.spec)
    cfg=ns(yaml.safe_load(cp.read_text(encoding='utf-8')))
    spec=json.loads(sp.read_text(encoding='utf-8'))
    assert spec['protocol']=='SemLT_locked_causal_ablation_v1'
    assert list(spec['variants'])==['A1_REWRITE','A2_PROB','A3_NORMAL1D','A4_FREE2D','A5_SUPPORT','A6_SEMLT']
    assert_config(cfg)
    for v in spec['variants']: assert_config(cfg_for(cfg,spec,v))
    assert_single_variable_contrasts(spec)
    assert_identical_initialization_for_same_arch(cfg,spec)
    assert_identity_and_gradients(cfg,spec)
    print('[PASS] SemLT locked ablation contract: A1-A6 exact identity init, frozen Base causal isolation, '
          'single-variable preregistered contrasts, identical initialization for same-architecture variants, no M2/M3.')
if __name__=='__main__': main()
