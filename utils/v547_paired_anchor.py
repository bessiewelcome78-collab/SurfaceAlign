"""V547 paired residual-anchor replay.

The replay is used only while training M1/M2.  It creates a synthetic factual
anchor and the component teacher from the *same* anchor, avoiding the V538
mismatch where labels were changed after the refiner had already observed the
unmodified Base prediction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

EPS = 1.0e-6


def _as_b1hw(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 4:
        return value[:, :1]
    if value.ndim == 3:
        return value[:, None]
    raise ValueError(f"Expected [B,H,W] or [B,1,H,W], got {tuple(value.shape)}")


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    radius = max(int(radius), 1)
    kernel = 2 * radius + 1
    return F.max_pool2d(mask, kernel, stride=1, padding=radius)


def _erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    radius = max(int(radius), 1)
    kernel = 2 * radius + 1
    return 1.0 - F.max_pool2d(1.0 - mask, kernel, stride=1, padding=radius)


@dataclass(frozen=True)
class V547PairedAnchor:
    probability: torch.Tensor
    replay_case: torch.Tensor
    replay_type: torch.Tensor
    replay_radius: torch.Tensor
    real_residual_fraction: torch.Tensor
    synthetic_residual_fraction: torch.Tensor


def build_v547_paired_anchor(
    *,
    base_probability: torch.Tensor,
    target: torch.Tensor,
    epoch: int,
    residual_floor: float = 0.0025,
    radius_min: int = 1,
    radius_max: int = 3,
    force_confidence: float = 0.90,
) -> V547PairedAnchor:
    """Create a deterministic FP/FN replay anchor when real residuals vanish.

    Replay type alternates by epoch and sample:
      0: inner FN boundary band;
      1: outer FP boundary band.

    The original Base probability is retained outside the injected band.  The
    returned anchor is detached and must be used consistently by the error head,
    candidate generator, exact candidate outcome, and component teacher.
    """
    base = _as_b1hw(base_probability).detach().clamp(EPS, 1.0 - EPS)
    gt = (_as_b1hw(target).detach() >= 0.5).to(base.dtype)
    hard_base = (base >= 0.5).to(base.dtype)
    real_residual = (hard_base != gt).to(base.dtype).flatten(1).mean(dim=1)

    floor = max(float(residual_floor), 0.0)
    replay_case = real_residual < floor
    b = base.shape[0]
    radius_min = max(int(radius_min), 1)
    radius_max = max(int(radius_max), radius_min)
    confidence = min(max(float(force_confidence), 0.50 + EPS), 1.0 - EPS)
    low = 1.0 - confidence
    high = confidence

    anchor = base.clone()
    replay_type = torch.full((b,), -1, device=base.device, dtype=torch.long)
    replay_radius = torch.zeros((b,), device=base.device, dtype=torch.long)
    radius_span = radius_max - radius_min + 1

    for sample in range(b):
        if not bool(replay_case[sample].item()):
            continue
        mode = (int(epoch) + sample) % 2
        radius = radius_min + ((int(epoch) // 2 + sample) % radius_span)
        replay_type[sample] = mode
        replay_radius[sample] = radius

        sample_gt = gt[sample : sample + 1]
        if mode == 0:
            band = (sample_gt - _erode(sample_gt, radius)).clamp(0.0, 1.0)
            # Very small lesions can disappear completely after erosion.  In that
            # case the entire lesion is a valid FN replay rather than an empty one.
            if float(band.sum().item()) < 1.0:
                band = sample_gt
            forced = torch.minimum(
                anchor[sample : sample + 1],
                anchor.new_full(anchor[sample : sample + 1].shape, low),
            )
        else:
            band = (_dilate(sample_gt, radius) - sample_gt).clamp(0.0, 1.0)
            if float(band.sum().item()) < 1.0:
                # Empty masks are not expected for BUSI, but keep the contract
                # total and deterministic if they occur.
                band = (_dilate(sample_gt, max(radius, 1)) - sample_gt).clamp(0.0, 1.0)
            forced = torch.maximum(
                anchor[sample : sample + 1],
                anchor.new_full(anchor[sample : sample + 1].shape, high),
            )
        anchor[sample : sample + 1] = (
            anchor[sample : sample + 1] * (1.0 - band) + forced * band
        )

    synthetic_residual = (
        (anchor >= 0.5).to(gt.dtype) != gt
    ).to(base.dtype).flatten(1).mean(dim=1)
    return V547PairedAnchor(
        probability=anchor.detach(),
        replay_case=replay_case.detach(),
        replay_type=replay_type.detach(),
        replay_radius=replay_radius.detach(),
        real_residual_fraction=real_residual.detach(),
        synthetic_residual_fraction=synthetic_residual.detach(),
    )
