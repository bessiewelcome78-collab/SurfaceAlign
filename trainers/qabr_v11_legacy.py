# Auto-generated JBT-Lite v11 safe wrapper around the original v10 QABR.
# Original implementation is preserved verbatim in qabr_v10_legacy.py.
from .qabr_v10_legacy import *  # noqa: F401,F403
from .qabr_v10_legacy import QueryAnchoredBoundaryRefiner as _LegacyQABR
import os
import torch
import torch.nn.functional as F

class QueryAnchoredBoundaryRefiner(_LegacyQABR):
    """Coarse-to-fine, gradient-isolated safety wrapper for v10 QABR.

    Contracts:
      1) Phase A: QABR is exactly identity until START_STEP (BUSI default 20 epochs=520 steps).
      2) Legacy QABR reads detached coarse/decoder/image tensors, so its side branch cannot
         send a second gradient path into PVL/decoder. Base still receives the direct dL/dz path.
      3) Correction is restricted to uncertain coarse-logit pixels and spatially mean-centered,
         reducing the area bias observed in QABR-only while preserving local boundary motion.
      4) Original v10 module, configs and results remain untouched in the source project.
    """
    _v11_base_arg_index = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('_jbtl11_train_step', torch.zeros((), dtype=torch.long), persistent=True)
        self.v11_start_step = int(os.environ.get('QABR_V11_START_STEP', '520'))
        self.v11_ramp_steps = max(int(os.environ.get('QABR_V11_RAMP_STEPS', '130')), 1)
        self.v11_uncertainty_floor = float(os.environ.get('QABR_V11_UNCERTAINTY_FLOOR', '0.30'))
        self.v11_gate_power = float(os.environ.get('QABR_V11_GATE_POWER', '1.5'))
        self.v11_corr_clip = float(os.environ.get('QABR_V11_CORR_CLIP', '1.25'))
        self.v11_corr_scale = float(os.environ.get('QABR_V11_CORR_SCALE', '0.90'))
        self.v11_center = os.environ.get('QABR_V11_CENTER_RESIDUAL', '1') != '0'
        self.v11_print_every = int(os.environ.get('QABR_V11_PRINT_EVERY', '130'))

    @staticmethod
    def _detach_tree(x):
        if torch.is_tensor(x):
            return x.detach()
        if isinstance(x, tuple):
            return tuple(QueryAnchoredBoundaryRefiner._detach_tree(v) for v in x)
        if isinstance(x, list):
            return [QueryAnchoredBoundaryRefiner._detach_tree(v) for v in x]
        if isinstance(x, dict):
            return {k: QueryAnchoredBoundaryRefiner._detach_tree(v) for k,v in x.items()}
        return x

    def _find_base(self, args, kwargs):
        for key in ('base_logits','coarse_logits','seg_logits','logits','mask_logits'):
            v=kwargs.get(key, None)
            if torch.is_tensor(v) and v.ndim==4 and v.shape[1]==1:
                return v
        idx=self._v11_base_arg_index
        if 0 <= idx < len(args):
            v=args[idx]
            if torch.is_tensor(v) and v.ndim==4 and v.shape[1]==1:
                return v
        c=[]
        for v in list(args)+list(kwargs.values()):
            if torch.is_tensor(v) and v.ndim==4 and v.shape[1]==1 and min(v.shape[-2:]) >= 64:
                c.append(v)
        if len(c)==1:
            return c[0]
        raise RuntimeError('JBTL11 cannot uniquely identify QABR coarse/base logit; candidates=' + str([tuple(x.shape) for x in c]))

    @staticmethod
    def _extract_tensor(result):
        candidates=[]
        if torch.is_tensor(result) and result.ndim==4 and result.shape[1]==1:
            candidates.append((result, ('tensor', None)))
        if isinstance(result, (tuple,list)):
            for i,v in enumerate(result):
                if torch.is_tensor(v) and v.ndim==4 and v.shape[1]==1:
                    candidates.append((v, ('seq', i)))
        if isinstance(result, dict):
            for key in ('refined_logits','seg_logits','logits','output','mask_logits'):
                v=result.get(key, None)
                if torch.is_tensor(v) and v.ndim==4 and v.shape[1]==1:
                    candidates.append((v, ('dict', key)))
            if not candidates:
                for key,v in result.items():
                    if torch.is_tensor(v) and v.ndim==4 and v.shape[1]==1:
                        candidates.append((v, ('dict', key)))
        if not candidates:
            raise RuntimeError('JBTL11 cannot identify legacy QABR output tensor')
        candidates.sort(key=lambda x: int(x[0].shape[-2])*int(x[0].shape[-1]), reverse=True)
        return candidates[0]

    @staticmethod
    def _put_tensor(result, where, new):
        kind,key=where
        if kind=='tensor': return new
        if kind=='seq':
            vals=list(result); vals[key]=new
            return tuple(vals) if isinstance(result,tuple) else vals
        if kind=='dict':
            vals=dict(result); vals[key]=new; return vals
        raise AssertionError(where)

    def forward(self, *args, **kwargs):
        base = self._find_base(args, kwargs)
        # Side-branch isolation: QABR may learn, but cannot directly rewrite Base/PVL/decoder features.
        dargs = self._detach_tree(args)
        dkwargs = self._detach_tree(kwargs)
        legacy_result = super().forward(*dargs, **dkwargs)
        legacy_logits, where = self._extract_tensor(legacy_result)
        if tuple(base.shape[-2:]) != tuple(legacy_logits.shape[-2:]):
            base_hr = F.interpolate(base, size=legacy_logits.shape[-2:], mode='bilinear', align_corners=False)
        else:
            base_hr = base
        base_d = base_hr.detach()
        raw_corr = legacy_logits - base_d

        if self.training:
            self._jbtl11_train_step.add_(1)
        step = int(self._jbtl11_train_step.item())
        ramp = max(0.0, min(1.0, (step - self.v11_start_step) / float(self.v11_ramp_steps)))

        # Residual-only trust region: edit only genuinely uncertain boundary pixels.
        p = torch.sigmoid(base_d.float())
        u = (4.0 * p * (1.0 - p)).clamp(0.0, 1.0)
        floor = min(max(self.v11_uncertainty_floor, 0.0), 0.95)
        gate = ((u - floor) / max(1.0-floor, 1.0e-6)).clamp(0.0,1.0)
        gate = gate.pow(max(self.v11_gate_power, 0.25)).to(raw_corr.dtype).detach()

        corr = raw_corr
        if self.v11_center:
            dims=tuple(range(2, corr.ndim))
            denom=gate.sum(dim=dims, keepdim=True).clamp_min(1.0)
            mean=(gate*corr).sum(dim=dims, keepdim=True)/denom
            corr=corr-mean
        corr = corr.clamp(-self.v11_corr_clip, self.v11_corr_clip) * self.v11_corr_scale
        safe_corr = float(ramp) * gate * corr
        out_logits = base_hr + safe_corr

        if self.training and self.v11_print_every > 0 and step % self.v11_print_every == 0:
            with torch.no_grad():
                print('[JBTL11_QABR_SAFE] step=%d ramp=%.3f gate=%.4f raw_abs=%.5f safe_abs=%.5f alpha=%.5f' % (
                    step, ramp, float(gate.mean()), float(raw_corr.abs().mean()), float(safe_corr.abs().mean()),
                    float(torch.tanh(self.alpha.detach())) if hasattr(self,'alpha') else float('nan')
                ), flush=True)
        return self._put_tensor(legacy_result, where, out_logits)
