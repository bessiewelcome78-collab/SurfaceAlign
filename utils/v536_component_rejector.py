"""V536 case/component rejector.

This module adapts two ideas from 2026 open-source segmentation work without
copying task-specific anatomical priors:

* R2-Seg: turn dense predictions into connected candidate regions and reject
  unsupported components instead of accepting every positive pixel.
* PGR-Net: keep only the highest-confidence ROI (Top-1 in the first safety
  stage) so sparse errors cannot be scattered over every case.

The selector is deliberately parameter-free.  It uses the policy margin,
M1 cause support, candidate/base hard-mask change and spatial uncertainty.  A
component is accepted only when a conservative lower-confidence score is
positive.  This zero boundary has a direct meaning: the component evidence
must remain positive after uncertainty and local-control correction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

try:
    from scipy import ndimage
except Exception as exc:  # pragma: no cover - environment contract catches it
    ndimage = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None


@dataclass(frozen=True)
class ComponentDecision:
    accepted: bool
    action_index: int
    area: int
    score: float
    evidence_mean: float
    uncertainty_mean: float
    cause_mean: float


def _normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(probabilities, 1.0e-8, 1.0)
    entropy = -(p * np.log(p)).sum(axis=0)
    return entropy / np.log(float(probabilities.shape[0]))


def _component_score(
    component: np.ndarray,
    margin: np.ndarray,
    cause: np.ndarray,
    uncertainty: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Spatially-aware lower-confidence evidence score.

    The component evidence is compared with a one-pixel local control ring.
    No tunable weighted sum is used: a component must pass both a within-region
    lower-confidence bound and a positive local contrast, so the final score is
    the minimum of the two.
    """
    values = margin[component] * cause[component] * (1.0 - uncertainty[component])
    if values.size == 0:
        return float("-inf"), 0.0, 1.0, 0.0
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    lcb = mean - std / max(np.sqrt(float(values.size)), 1.0)

    dilated = ndimage.binary_dilation(component, iterations=1)
    ring = np.logical_and(dilated, np.logical_not(component))
    ring_values = margin[ring] * cause[ring] * (1.0 - uncertainty[ring])
    local_control = float(ring_values.mean()) if ring_values.size else 0.0
    contrast = mean - local_control
    score = min(lcb, contrast)
    return score, mean, float(uncertainty[component].mean()), float(cause[component].mean())


def select_top1_component(
    *,
    policy_soft: torch.Tensor,
    policy_hard_index: torch.Tensor,
    case_action_index: torch.Tensor,
    action_candidates: torch.Tensor,
    base_probability: torch.Tensor,
    cause_probability: torch.Tensor,
    min_pixels: int = 4,
) -> Tuple[torch.Tensor, List[ComponentDecision]]:
    """Return one accepted component per case, or exact Preserve.

    Args use deployment-resolution tensors:
      policy_soft: [B,5,H,W]
      policy_hard_index: [B,1,H,W], 0 Preserve and 1..4 actions
      case_action_index: [B], 0 Preserve and 1..4 actions
      action_candidates: [B,4,H,W]
      base_probability: [B,1,H,W]
      cause_probability: [B,4,H,W]

    The returned mask is detached because component extraction is a deployment
    rejector.  Policy heads are trained by explicit pixel- and case-level
    objectives, not through a straight-through connected-component surrogate.
    """
    if ndimage is None:
        raise RuntimeError(
            "V536 component rejector requires scipy.ndimage; original error: "
            + repr(_SCIPY_IMPORT_ERROR)
        )
    if policy_soft.ndim != 4 or policy_soft.shape[1] != 5:
        raise ValueError(f"policy_soft must be [B,5,H,W], got {tuple(policy_soft.shape)}")

    device = policy_soft.device
    dtype = policy_soft.dtype
    b, _, h, w = policy_soft.shape
    output = torch.zeros((b, 1, h, w), device=device, dtype=dtype)
    decisions: List[ComponentDecision] = []

    soft_np = policy_soft.detach().float().cpu().numpy()
    hard_np = policy_hard_index.detach().cpu().numpy()[:, 0]
    case_np = case_action_index.detach().cpu().numpy()
    cand_np = action_candidates.detach().float().cpu().numpy()
    base_np = base_probability.detach().float().cpu().numpy()[:, 0]
    cause_np = cause_probability.detach().float().cpu().numpy()

    for sample in range(b):
        action = int(case_np[sample])
        if action <= 0 or action > 4:
            decisions.append(ComponentDecision(False, 0, 0, 0.0, 0.0, 1.0, 0.0))
            continue

        action_channel = action - 1
        base_hard = base_np[sample] >= 0.5
        candidate_hard = cand_np[sample, action_channel] >= 0.5
        changes = candidate_hard != base_hard
        predicted = hard_np[sample] == action
        binary = np.logical_and(changes, predicted)
        labels, count = ndimage.label(binary)
        if count <= 0:
            decisions.append(ComponentDecision(False, action, 0, 0.0, 0.0, 1.0, 0.0))
            continue

        uncertainty = _normalized_entropy(soft_np[sample])
        preserve_log_probability = np.log(np.clip(soft_np[sample, 0], 1.0e-8, 1.0))
        action_log_probability = np.log(
            np.clip(soft_np[sample, action], 1.0e-8, 1.0)
        )
        margin = action_log_probability - preserve_log_probability
        cause = cause_np[sample, action_channel]

        best = None
        best_mask = None
        for label_id in range(1, count + 1):
            component = labels == label_id
            area = int(component.sum())
            if area < max(int(min_pixels), 1):
                continue
            score, evidence_mean, uncertainty_mean, cause_mean = _component_score(
                component, margin, cause, uncertainty
            )
            decision = ComponentDecision(
                accepted=bool(score > 0.0),
                action_index=action,
                area=area,
                score=float(score),
                evidence_mean=float(evidence_mean),
                uncertainty_mean=float(uncertainty_mean),
                cause_mean=float(cause_mean),
            )
            if decision.accepted and (best is None or decision.score > best.score):
                best = decision
                best_mask = component

        if best is None or best_mask is None:
            decisions.append(ComponentDecision(False, action, 0, 0.0, 0.0, 1.0, 0.0))
            continue

        output[sample, 0] = torch.from_numpy(best_mask.astype(np.float32)).to(
            device=device, dtype=dtype
        )
        decisions.append(best)

    return output.detach(), decisions
