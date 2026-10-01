"""JBT-Lite v15 Boundary-Tangent QABR (BT-QABR).

Root-cause fix after the v14 deploy-aligned regression.

V14 correctly kept Base/Surface factual gradients isolated, but added a full
finite Dice+CE auxiliary objective to the tiny QABR head.  On BUSI this drove
the residual magnitude up, improved boundary NSD, and simultaneously degraded
region Dice.  The failure mode is a boundary/area trade-off: a local logit
residual can translate a contour but can also inflate/deflate foreground mass.

V15 therefore returns to the proven v13 ownership contract (zero-forward
straight-through training, no finite auxiliary segmentation loss) and changes
the *operator*, not the loss.  Before deployment/training-shadow, the QABR
residual is projected onto the first-order tangent space of constant foreground
probability mass:

    sum_x sigmoid(z_x)(1-sigmoid(z_x)) * delta_x = 0.

This makes QABR behave like a local boundary redistribution/displacement rather
than a second region segmenter.  The host trajectory is exactly preserved, QABR
still learns from the existing segmentation objective through a zero-forward
shadow residual, and evaluation deploys the finite tangent correction.
"""
from __future__ import annotations

import math
import os
from typing import Any

import torch
import torch.nn.functional as F

from .qabr_v10_legacy import *  # noqa: F401,F403
from .qabr_v10_legacy import QueryAnchoredBoundaryRefiner as _LegacyQABR


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name, "1" if default else "0").strip().lower()
    return raw not in {"0", "false", "no", "off"}


class QueryAnchoredBoundaryRefiner(_LegacyQABR):
    """QABR with strict host isolation and probability-mass tangent projection."""

    _base_arg_index = 1
    _image_arg_index = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("_jbtl13_epoch_1based", torch.ones((), dtype=torch.long), persistent=True)
        self.register_buffer("_jbtl13_total_epochs", torch.tensor(100, dtype=torch.long), persistent=True)

        self.v15_start_frac = _env_float("QABR_V15_DEPLOY_START_FRAC", 0.10)
        self.v15_end_frac = _env_float("QABR_V15_DEPLOY_END_FRAC", 0.30)
        # Shadow training keeps factual forward exactly unchanged; no finite auxiliary loss is used.
        self.v15_shadow_weight = _env_float("QABR_V15_SHADOW_WEIGHT", 1.00)
        self.v15_eval_scale = _env_float("QABR_V15_EVAL_SCALE", 1.00)
        self.v15_alpha_ready = _env_float("QABR_V15_ALPHA_READY", 0.04)
        self.v15_corr_clip = _env_float("QABR_V15_CORR_CLIP", 1.25)
        self.v15_corr_scale = _env_float("QABR_V15_CORR_SCALE", 0.90)
        # V15 replaces ordinary logit-mean centering with a probability-mass
        # tangent projection.  sigmoid'(z)=p(1-p) is the first-order conversion
        # from a logit displacement to foreground-probability mass change.
        self.v15_center = _env_bool("QABR_V15_TANGENT_PROJECTION", True)
        self.v15_tangent_passes = max(1, int(_env_float("QABR_V15_TANGENT_PASSES", 2)))
        self.v15_readiness_enabled = _env_bool("QABR_V15_READINESS_GATE", True)
        self.v15_train_shadow_only = _env_bool("QABR_V15_TRAIN_SHADOW_ONLY", True)
        self.v15_debug = _env_bool("QABR_V15_DEBUG", True)
        # Strong structural ablation: remove strict local support and let the
        # residual act globally. OFF in FULL; useful to demonstrate why local
        # constrained correction is needed instead of a second segmenter.
        self.ablate_global_support = _env_bool("QABR_ABL_GLOBAL_SUPPORT", False)
        self._v15_last_debug_epoch = None

        if not (0.0 <= self.v15_start_frac < self.v15_end_frac <= 1.0):
            raise ValueError(
                "QABR_V13 deployment fractions must satisfy 0 <= start < end <= 1, "
                f"got {self.v15_start_frac}, {self.v15_end_frac}"
            )
        if self.v15_shadow_weight <= 0.0:
            raise ValueError("QABR_V15_SHADOW_WEIGHT must be > 0")
        if not (0.0 <= self.v15_eval_scale <= 1.5):
            raise ValueError("QABR_V15_EVAL_SCALE must be in [0, 1.5]")
        if self.v15_alpha_ready <= 0.0:
            raise ValueError("QABR_V15_ALPHA_READY must be > 0")

    def set_training_progress(self, epoch_1based: int, total_epochs: int) -> None:
        epoch_1based = max(1, int(epoch_1based))
        total_epochs = max(epoch_1based, int(total_epochs))
        self._jbtl13_epoch_1based.fill_(epoch_1based)
        self._jbtl13_total_epochs.fill_(total_epochs)

    @staticmethod
    def _detach_tree(x: Any):
        if torch.is_tensor(x):
            return x.detach()
        if isinstance(x, tuple):
            return tuple(QueryAnchoredBoundaryRefiner._detach_tree(v) for v in x)
        if isinstance(x, list):
            return [QueryAnchoredBoundaryRefiner._detach_tree(v) for v in x]
        if isinstance(x, dict):
            return {k: QueryAnchoredBoundaryRefiner._detach_tree(v) for k, v in x.items()}
        return x

    def _find_base(self, args, kwargs):
        for key in ("base_logits", "coarse_logits", "seg_logits", "logits", "mask_logits"):
            v = kwargs.get(key, None)
            if torch.is_tensor(v) and v.ndim == 4 and v.shape[1] == 1:
                return v
        if 0 <= self._base_arg_index < len(args):
            v = args[self._base_arg_index]
            if torch.is_tensor(v) and v.ndim == 4 and v.shape[1] == 1:
                return v
        candidates = [
            v for v in list(args) + list(kwargs.values())
            if torch.is_tensor(v) and v.ndim == 4 and v.shape[1] == 1 and min(v.shape[-2:]) >= 32
        ]
        if len(candidates) == 1:
            return candidates[0]
        raise RuntimeError(
            "JBTL15 cannot uniquely identify the QABR coarse logit; candidates="
            + str([tuple(v.shape) for v in candidates])
        )

    @staticmethod
    def _extract_tensor(result):
        candidates = []
        if torch.is_tensor(result) and result.ndim == 4 and result.shape[1] == 1:
            candidates.append((result, ("tensor", None)))
        if isinstance(result, (tuple, list)):
            for i, v in enumerate(result):
                if torch.is_tensor(v) and v.ndim == 4 and v.shape[1] == 1:
                    candidates.append((v, ("seq", i)))
        if isinstance(result, dict):
            for key in ("refined_logits", "seg_logits", "logits", "output", "mask_logits"):
                v = result.get(key, None)
                if torch.is_tensor(v) and v.ndim == 4 and v.shape[1] == 1:
                    candidates.append((v, ("dict", key)))
            if not candidates:
                for key, v in result.items():
                    if torch.is_tensor(v) and v.ndim == 4 and v.shape[1] == 1:
                        candidates.append((v, ("dict", key)))
        if not candidates:
            raise RuntimeError("JBTL15 cannot identify legacy QABR output tensor")
        candidates.sort(key=lambda x: int(x[0].shape[-2]) * int(x[0].shape[-1]), reverse=True)
        return candidates[0]

    @staticmethod
    def _put_tensor(result, where, new):
        kind, key = where
        if kind == "tensor":
            return new
        if kind == "seq":
            vals = list(result)
            vals[key] = new
            return tuple(vals) if isinstance(result, tuple) else vals
        if kind == "dict":
            vals = dict(result)
            vals[key] = new
            return vals
        raise AssertionError(where)

    @staticmethod
    def _cosine_ramp(x: float) -> float:
        x = max(0.0, min(1.0, float(x)))
        return 0.5 * (1.0 - math.cos(math.pi * x))

    def _time_deploy(self) -> float:
        epoch = float(self._jbtl13_epoch_1based.item())
        total = max(float(self._jbtl13_total_epochs.item()), 1.0)
        frac = max(0.0, min(1.0, epoch / total))
        if frac <= self.v15_start_frac:
            return 0.0
        if frac >= self.v15_end_frac:
            return 1.0
        r = (frac - self.v15_start_frac) / (self.v15_end_frac - self.v15_start_frac)
        return self._cosine_ramp(r)

    def _readiness(self) -> float:
        if not self.v15_readiness_enabled:
            return 1.0
        with torch.no_grad():
            a = abs(float(torch.tanh(self.alpha.detach()))) if hasattr(self, "alpha") else 0.0
        return self._cosine_ramp(min(a / self.v15_alpha_ready, 1.0))

    def _hard_boundary_band(self, base_hr_detached: torch.Tensor) -> torch.Tensor:
        if self.ablate_global_support:
            return torch.ones_like(base_hr_detached).detach()
        p = torch.sigmoid(base_hr_detached.float())
        hard = (p >= 0.5).to(base_hr_detached.dtype)
        dil = F.max_pool2d(hard, 3, stride=1, padding=1)
        ero = 1.0 - F.max_pool2d(1.0 - hard, 3, stride=1, padding=1)
        boundary = (dil - ero).clamp(0.0, 1.0)
        return F.max_pool2d(
            boundary,
            kernel_size=self.band_kernel,
            stride=1,
            padding=self.band_kernel // 2,
        ).clamp(0.0, 1.0).detach()

    def _probability_tangent_project(
        self, corr: torch.Tensor, base_hr_detached: torch.Tensor, band: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project local logit correction to zero first-order foreground-mass change.

        For p=sigmoid(z), a small logit perturbation d changes foreground mass by
        sum p(1-p)d.  Subtracting the weighted mean inside the boundary band
        removes the normal component that globally inflates/deflates the mask,
        while preserving local signed redistribution along the contour.
        """
        p = torch.sigmoid(base_hr_detached.float()).to(corr.dtype)
        slope = (p * (1.0 - p)).detach()
        w = (band * slope).detach()
        dims = tuple(range(2, corr.ndim))
        denom = w.sum(dim=dims, keepdim=True).clamp_min(1e-6)

        # Clip before projection, then alternate clip/project a small fixed number
        # of times.  A final projection is left unclipped so the tangent equality
        # is exact up to floating point; observed QABR corrections are far below
        # the safety cap, so this does not increase practical amplitude.
        out = (band * corr).clamp(-self.v15_corr_clip, self.v15_corr_clip)
        for _ in range(self.v15_tangent_passes):
            mu = (w * out).sum(dim=dims, keepdim=True) / denom
            out = band * (out - mu)
            out = out.clamp(-self.v15_corr_clip, self.v15_corr_clip)
        mu = (w * out).sum(dim=dims, keepdim=True) / denom
        out = band * (out - mu)
        tangent_mass = (w * out).sum(dim=dims, keepdim=True)
        return out, tangent_mass

    def forward(self, *args, **kwargs):
        base = self._find_base(args, kwargs)

        # QABR side inputs are factual observers only; no QABR gradient is allowed
        # to enter Base/PVL/decoder/image through the side branch.
        dargs = self._detach_tree(args)
        dkwargs = self._detach_tree(kwargs)
        legacy_result = super().forward(*dargs, **dkwargs)
        legacy_logits, where = self._extract_tensor(legacy_result)

        if tuple(base.shape[-2:]) != tuple(legacy_logits.shape[-2:]):
            base_hr = F.interpolate(base, size=legacy_logits.shape[-2:], mode="bilinear", align_corners=False)
        else:
            base_hr = base
        base_d = base_hr.detach()

        raw_corr = legacy_logits - base_d
        band = self._hard_boundary_band(base_d)
        corr = raw_corr
        if self.v15_center:
            corr, tangent_mass = self._probability_tangent_project(corr, base_d, band)
        else:
            corr = band * corr.clamp(-self.v15_corr_clip, self.v15_corr_clip)
            tangent_mass = corr.new_zeros((corr.shape[0], corr.shape[1], 1, 1))
        corr = corr * self.v15_corr_scale

        time_deploy = self._time_deploy()
        readiness = self._readiness()
        deploy = float(time_deploy * readiness * self.v15_eval_scale)

        if self.training and self.v15_train_shadow_only:
            # Core V15 ownership contract:
            #   forward(shadow) == 0 for every epoch, even after QABR is ready;
            #   d(shadow)/d(theta_qabr) != 0.
            # Hence L is evaluated at the exact factual Base/Surface logits, so
            # shared-branch gradients are identical to the QABR-disabled arm.
            shadow_coeff = float(self.v15_shadow_weight)
            shadow = shadow_coeff * (corr - corr.detach())
            visible = corr * 0.0
        else:
            # Evaluation/test deploys the learned local residual.  The optional
            # compatibility path for TRAIN_SHADOW_ONLY=0 reproduces the v12-style
            # cross-fade, but is deliberately not the default.
            visible = deploy * corr
            if self.training:
                shadow_coeff = float(self.v15_shadow_weight * (1.0 - time_deploy))
                shadow = shadow_coeff * (corr - corr.detach())
            else:
                shadow_coeff = 0.0
                shadow = corr * 0.0

        out_logits = base_hr + visible + shadow

        epoch = int(self._jbtl13_epoch_1based.item())
        if self.training and self.v15_debug and self._v15_last_debug_epoch != epoch:
            self._v15_last_debug_epoch = epoch
            with torch.no_grad():
                p = torch.sigmoid(base_d.float())
                uncertainty = 4.0 * p * (1.0 - p)
                print(
                    "[JBTL15_QABR_TANGENT] epoch=%03d/%03d train_shadow_only=%d "
                    "time=%.4f ready=%.4f eval_deploy=%.4f shadow=%.4f "
                    "band=%.4f u=%.4f raw_abs=%.6f corr_abs=%.6f "
                    "mass1=%.3e train_visible_abs=%.6f alpha=%+.5f"
                    % (
                        epoch,
                        int(self._jbtl13_total_epochs.item()),
                        int(self.v15_train_shadow_only),
                        time_deploy,
                        readiness,
                        deploy,
                        shadow_coeff,
                        float(band.mean()),
                        float(uncertainty.mean()),
                        float(raw_corr.abs().mean()),
                        float(corr.abs().mean()),
                        float(tangent_mass.abs().max()),
                        float(visible.abs().mean()),
                        float(torch.tanh(self.alpha.detach())) if hasattr(self, "alpha") else float("nan"),
                    ),
                    flush=True,
                )

        return self._put_tensor(legacy_result, where, out_logits)
