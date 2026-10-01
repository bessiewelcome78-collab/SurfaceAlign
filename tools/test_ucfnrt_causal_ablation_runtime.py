#!/usr/bin/env python3
from __future__ import annotations
from types import SimpleNamespace as NS
import copy, torch
from utils.semlt_autozero_transport import AutoZeroSemanticTransportSegmenter
from utils.semlt_autozero_loss import _compute_uc_fnrt_loss

def cfg(mode=None):
    m=dict(SEMLT_UC_FNRT=True,SEMLT_LOCAL_RADIUS_PX=8,SEMLT_AUTOZERO_DETACH_CONDITIONERS=True,SEMLT_GRADIENT_AUDIT=False)
    if mode is not None: m['SEMLT_UC_FNRT_ABLATION']=mode
    return NS(MODEL=NS(ADAPTER_DIM=32,BACKBONE='ViT-B/16'),M1=NS(**m))

def inputs():
    torch.manual_seed(7); b=h=w=1
    H=W=32
    yy,xx=torch.meshgrid(torch.arange(H),torch.arange(W),indexing='ij')
    dist=((xx-15.5)**2+(yy-15.5)**2).sqrt()
    base_prob=torch.sigmoid((9.0-dist)/1.2)[None,None]
    factual=torch.logit(base_prob.clamp(1e-4,1-1e-4))
    img=torch.randn(1,32,H,W); sem=torch.randn(1,32,H,W)
    soft=(4*base_prob*(1-base_prob)).clamp(0,1)
    mcstd=torch.rand_like(base_prob)*0.08+0.01; mcdis=torch.rand_like(base_prob)*0.12+0.01
    return factual,base_prob,img,sem,soft,soft,mcstd,mcdis

def randomize_heads(m):
    torch.manual_seed(99)
    with torch.no_grad():
        m.direction_controller.weight.normal_(0,0.02); m.direction_controller.bias.normal_(0,0.01)
        m.magnitude_controller.weight.normal_(0,0.02); m.magnitude_controller.bias.normal_(0,0.01)

def forward(m,inp):
    return m._generate_uc_fnrt(*inp)

def main():
    inp=inputs()
    # explicit full must be numerically identical to missing/default full.
    a=AutoZeroSemanticTransportSegmenter(cfg(None)).eval(); randomize_heads(a)
    b=AutoZeroSemanticTransportSegmenter(cfg('full')).eval(); b.load_state_dict(a.state_dict())
    oa,aa=forward(a,inp); ob,ab=forward(b,inp)
    assert torch.equal(oa,ob), (oa-ob).abs().max().item()
    print('[PASS] default/full forward is bitwise identical')

    u=AutoZeroSemanticTransportSegmenter(cfg('no_posterior_uncertainty')).eval(); u.load_state_dict(a.state_dict())
    ou,au=forward(u,inp)
    assert float(au['geotr_m1_mc_std_mean'].mean())>0
    assert float(au['geotr_m1_effective_mc_std_mean'].abs().max())==0.0
    assert float(au['geotr_m1_effective_mc_disagreement_mean'].abs().max())==0.0
    assert not torch.equal(ou,oa)
    print('[PASS] no-posterior masks only effective MC dispersion while preserving raw audit')

    r=AutoZeroSemanticTransportSegmenter(cfg('no_normal_ray_evidence')).eval(); r.load_state_dict(a.state_dict())
    or_,ar=forward(r,inp); assert not torch.equal(or_,oa)
    print('[PASS] no-ray is an active causal intervention with identical state layout')

    d1=AutoZeroSemanticTransportSegmenter(cfg('direct_signed')).eval(); d1.load_state_dict(a.state_dict())
    d2=copy.deepcopy(d1)
    with torch.no_grad(): d2.magnitude_controller.weight.add_(10.0); d2.magnitude_controller.bias.add_(10.0)
    od1,ad1=forward(d1,inp); od2,ad2=forward(d2,inp)
    assert torch.equal(ad1['geotr_m1_predicted_owner_sample_offset_px'],ad2['geotr_m1_predicted_owner_sample_offset_px'])
    assert torch.equal(od1,od2)
    print('[PASS] direct-signed deployment is independent of magnitude head')

    g=AutoZeroSemanticTransportSegmenter(cfg('segmentation_only')).eval(); g.load_state_dict(a.state_dict())
    og,ag=forward(g,inp)
    # simple non-empty GT circle, deliberately shifted to create geometric target
    gt=(inp[1][:,0].roll(shifts=1,dims=-1)>0.5).float()
    loss,diag=_compute_uc_fnrt_loss(cfg('segmentation_only'),gt,ag,0)
    assert torch.allclose(loss.detach(),diag['geotr_m1_segmentation_loss'],atol=0,rtol=0)
    print('[PASS] segmentation-only objective removes explicit geometry terms from optimization')

    # State layouts must be identical for every intervention, protecting model-construction RNG.
    keys=set(a.state_dict())
    for mode in ('no_posterior_uncertainty','no_normal_ray_evidence','direct_signed','segmentation_only'):
        assert set(AutoZeroSemanticTransportSegmenter(cfg(mode)).state_dict())==keys
    print('[PASS] all interventions have identical state-dict architecture')
if __name__=='__main__': main()
