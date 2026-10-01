"""JBT-Lite v12 Cooperative QABR.

This wrapper keeps the original v10 QABR operator intact while fixing the
v11 FULL interaction failure:

1) Shadow learning without forward perturbation: QABR learns from epoch 1 via
   a straight-through zero-forward residual, so handoff is never a cold start.
2) Fractional cosine ownership transfer: deployment is controlled by training
   progress (default 10%-30%), not a fixed 20-epoch switch.
3) No secondary uncertainty-only gate: v10 already contains a boundary band,
   uncertainty, image-edge support and a 25% sharp-but-misplaced floor. v11's
   second uncertainty gate starved QABR after EDGE sharpened the boundary.
4) Side-branch isolation remains strict: QABR reads detached Base/decoder/image
   tensors, while Base still gets the direct dL/dz path through base_hr.
5) Area-neutral local residual: correction is centered inside the current hard
   boundary band and remains zero outside that band.

No extra task, target or auxiliary loss is introduced.
"""
from __future__ import annotations

import math
import os
from typing import Any

import torch
import torch.nn.functional as F

from .qabr_v10_legacy import *  # noqa: F401,F403
from .qabr_v10_legacy import QueryAnchoredBoundaryRefiner as _LegacyQABR


class QueryAnchoredBoundaryRefiner(_LegacyQABR):
    """Cooperative, shadow-pretrained wrapper around the v10 QABR operator."""

    _base_arg_index = 1
    _image_arg_index = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Persist training progress so a locked checkpoint reproduces exactly
        # the deployment state used when it was selected.
        self.register_buffer("_jbtl12_epoch_1based", torch.ones((), dtype=torch.long), persistent=True)
        self.register_buffer("_jbtl12_total_epochs", torch.tensor(100, dtype=torch.long), persistent=True)

        self.v12_start_frac = float(os.environ.get("QABR_V12_DEPLOY_START_FRAC", "0.10"))
        self.v12_end_frac = float(os.environ.get("QABR_V12_DEPLOY_END_FRAC", "0.30"))
        self.v12_shadow_weight = float(os.environ.get("QABR_V12_SHADOW_WEIGHT", "0.50"))
        self.v12_alpha_ready = float(os.environ.get("QABR_V12_ALPHA_READY", "0.04"))
        self.v12_corr_clip = float(os.environ.get("QABR_V12_CORR_CLIP", "1.25"))
        self.v12_corr_scale = float(os.environ.get("QABR_V12_CORR_SCALE", "0.90"))
        self.v12_center = os.environ.get("QABR_V12_CENTER_RESIDUAL", "1") != "0"
        self.v12_readiness_enabled = os.environ.get("QABR_V12_READINESS_GATE", "1") != "0"
        self.v12_debug = os.environ.get("QABR_V12_DEBUG", "1") != "0"
        self._v12_last_debug_epoch = None

        if not (0.0 <= self.v12_start_frac < self.v12_end_frac <= 1.0):
            raise ValueError(
                "QABR_V12 deployment fractions must satisfy 0 <= start < end <= 1, "
                f"got {self.v12_start_frac}, {self.v12_end_frac}"
            )
        if self.v12_shadow_weight < 0.0:
            raise ValueError("QABR_V12_SHADOW_WEIGHT must be >= 0")
        if self.v12_alpha_ready <= 0.0:
            raise ValueError("QABR_V12_ALPHA_READY must be > 0")

    def set_training_progress(self, epoch_1based: int, total_epochs: int) -> None:
        epoch_1based = max(1, int(epoch_1based))
        total_epochs = max(epoch_1based, int(total_epochs))
        self._jbtl12_epoch_1based.fill_(epoch_1based)
        self._jbtl12_total_epochs.fill_(total_epochs)

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
            "JBTL12 cannot uniquely identify the QABR coarse logit; candidates="
            + str([tuple(v.shape) for v in candidates])
        )

    def _find_image(self, args, kwargs):
        image = kwargs.get("image", None)
        if torch.is_tensor(image) and image.ndim == 4:
            return image
        if 0 <= self._image_arg_index < len(args):
            image = args[self._image_arg_index]
            if torch.is_tensor(image) and image.ndim == 4:
                return image
        return None

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
            raise RuntimeError("JBTL12 cannot identify legacy QABR output tensor")
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
        epoch = float(self._jbtl12_epoch_1based.item())
        total = max(float(self._jbtl12_total_epochs.item()), 1.0)
        # Epoch-fraction schedule is aligned exactly with the Surface schedule:
        # 40ep -> 4..12, 100ep -> 10..30 for the default 10%-30% handoff.
        frac = max(0.0, min(1.0, epoch / total))
        if frac <= self.v12_start_frac:
            return 0.0
        if frac >= self.v12_end_frac:
            return 1.0
        r = (frac - self.v12_start_frac) / (self.v12_end_frac - self.v12_start_frac)
        return self._cosine_ramp(r)

    def _readiness(self) -> float:
        if not self.v12_readiness_enabled:
            return 1.0
        with torch.no_grad():
            a = abs(float(torch.tanh(self.alpha.detach()))) if hasattr(self, "alpha") else 0.0
        # Smooth readiness avoids a second hard switch.
        return self._cosine_ramp(min(a / self.v12_alpha_ready, 1.0))

    def _hard_boundary_band(self, base_hr_detached: torch.Tensor) -> torch.Tensor:
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

    def forward(self, *args, **kwargs):
        base = self._find_base(args, kwargs)

        # Strict side-branch gradient isolation. QABR learns its own parameters,
        # but cannot create a second gradient path into Base/PVL/decoder/image.
        dargs = self._detach_tree(args)
        dkwargs = self._detach_tree(kwargs)
        legacy_result = super().forward(*dargs, **dkwargs)
        legacy_logits, where = self._extract_tensor(legacy_result)

        if tuple(base.shape[-2:]) != tuple(legacy_logits.shape[-2:]):
            base_hr = F.interpolate(base, size=legacy_logits.shape[-2:], mode="bilinear", align_corners=False)
        else:
            base_hr = base
        base_d = base_hr.detach()

        # IMPORTANT: raw_corr already contains the v10 support term:
        # band * (0.25 + 0.75 * max(uncertainty, image_edge * band)).
        # Do NOT apply the v11 uncertainty-only gate again.
        raw_corr = legacy_logits - base_d
        band = self._hard_boundary_band(base_d)
        corr = raw_corr
        if self.v12_center:
            dims = tuple(range(2, corr.ndim))
            denom = band.sum(dim=dims, keepdim=True).clamp_min(1.0)
            mean = (band * corr).sum(dim=dims, keepdim=True) / denom
            corr = band * (corr - mean)
        else:
            corr = band * corr
        corr = corr.clamp(-self.v12_corr_clip, self.v12_corr_clip) * self.v12_corr_scale

        time_deploy = self._time_deploy()
        readiness = self._readiness()
        deploy = float(time_deploy * readiness)

        # Zero-forward / non-zero-backward shadow pretraining.
        # Forward value of (corr - corr.detach()) is exactly zero, but its
        # derivative trains QABR using the SAME existing segmentation/Surface
        # objective. No auxiliary loss is added.
        shadow_coeff = 0.0
        shadow = corr * 0.0
        if self.training and self.v12_shadow_weight > 0.0:
            shadow_coeff = float(self.v12_shadow_weight * (1.0 - time_deploy))
            shadow = shadow_coeff * (corr - corr.detach())

        visible = deploy * corr
        out_logits = base_hr + visible + shadow

        epoch = int(self._jbtl12_epoch_1based.item())
        if self.training and self.v12_debug and self._v12_last_debug_epoch != epoch:
            self._v12_last_debug_epoch = epoch
            with torch.no_grad():
                p = torch.sigmoid(base_d.float())
                uncertainty = 4.0 * p * (1.0 - p)
                print(
                    "[JBTL12_QABR_COOP] epoch=%03d/%03d time=%.4f ready=%.4f deploy=%.4f "
                    "shadow=%.4f band=%.4f u=%.4f raw_abs=%.6f visible_abs=%.6f alpha=%+.5f"
                    % (
                        epoch,
                        int(self._jbtl12_total_epochs.item()),
                        time_deploy,
                        readiness,
                        deploy,
                        shadow_coeff,
                        float(band.mean()),
                        float(uncertainty.mean()),
                        float(raw_corr.abs().mean()),
                        float(visible.abs().mean()),
                        float(torch.tanh(self.alpha.detach())) if hasattr(self, "alpha") else float("nan"),
                    ),
                    flush=True,
                )

        return self._put_tensor(legacy_result, where, out_logits)
