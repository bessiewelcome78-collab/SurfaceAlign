"""V486 fixed-base/local-repair proposal loss.

The public function remains ``compute_v484_loss`` for the existing train.py
loss dispatcher.  This version is intentionally designed for the V486 probe:

1. C0 is treated as a detached factual prediction.
2. The loss directly trains local counterfactual repair candidates, not only
   their support maps.
3. An edit-floor term prevents the trivial no-op solution when FP/FN/boundary
   error exists.
4. Diagnostics are emitted with v484_*, v485_* and v486_* names so older logging
   code remains compatible.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import math
import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy import ndimage
except Exception as exc:  # pragma: no cover - contract reports missing scipy
    ndimage = None
    _V536_SCIPY_IMPORT_ERROR = exc
else:
    _V536_SCIPY_IMPORT_ERROR = None

from utils.v484_error_state_causal import (
    ErrorStateHead,
    _as_b1hw,
    _soft_boundary,
    _soft_dice_probs,
)
from utils.v503_factual_atomic_causal import build_factual_cause_targets
from utils.v504_realizable_policy import build_realizable_action_targets
from utils.v538_loss import compute_v538_online_component_loss
from utils.v505_interactive_region_causal import (
    OUTCOME_BENEFIT,
    OUTCOME_HARM,
    OUTCOME_NEUTRAL,
    build_region_potential_outcomes,
)

EPS = 1.0e-6


def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def _m1(cfg: Any, key: str, default: Any = None) -> Any:
    return _cfg_get(_cfg_get(cfg, "M1", None), key, default)


def _soft_dice_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    pred = _as_b1hw(pred).clamp(EPS, 1.0 - EPS)
    target = _as_b1hw(target).clamp(0.0, 1.0)
    if mask is None:
        mask = torch.ones_like(target)
    mask = _as_b1hw(mask).clamp(0.0, 1.0)
    inter = (pred * target * mask).flatten(1).sum(dim=1)
    den = ((pred + target) * mask).flatten(1).sum(dim=1)
    valid = mask.flatten(1).sum(dim=1) > 1.0
    loss = 1.0 - (2.0 * inter + 1.0e-6) / (den + 1.0e-6)
    if valid.any():
        return loss[valid].mean()
    return pred.sum() * 0.0


def _support_loss(pred: torch.Tensor, target: torch.Tensor, pos_weight: float = 4.0) -> torch.Tensor:
    pred = _as_b1hw(pred).clamp(EPS, 1.0 - EPS)
    target = _as_b1hw(target).clamp(0.0, 1.0)
    weight = 1.0 + float(pos_weight) * target
    bce = F.binary_cross_entropy(pred, target, weight=weight, reduction="mean")
    return bce + _soft_dice_loss(pred, target)


def _masked_bce_dice(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, pos_weight: float = 1.0) -> torch.Tensor:
    pred = _as_b1hw(pred).clamp(EPS, 1.0 - EPS)
    target = _as_b1hw(target).clamp(0.0, 1.0)
    mask = _as_b1hw(mask).clamp(0.0, 1.0)
    valid = mask.flatten(1).sum(dim=1) > 1.0
    if not valid.any():
        return pred.sum() * 0.0
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    weight = 1.0 + float(pos_weight) * target
    bce = (bce * weight * mask).flatten(1).sum(dim=1) / mask.flatten(1).sum(dim=1).clamp_min(1.0)
    dice = 1.0 - (
        2.0 * (pred * target * mask).flatten(1).sum(dim=1) + 1.0e-6
    ) / (((pred + target) * mask).flatten(1).sum(dim=1) + 1.0e-6)
    return (bce + dice)[valid].mean()


def _iou_soft(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = _as_b1hw(pred).clamp(0.0, 1.0)
    target = _as_b1hw(target).clamp(0.0, 1.0)
    inter = (pred * target).flatten(1).sum(dim=1)
    union = (pred + target - pred * target).flatten(1).sum(dim=1).clamp_min(EPS)
    return (inter / union).mean()


def _region_mean(x: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
    x = _as_b1hw(x)
    region = _as_b1hw(region)
    num = (x * region).flatten(1).sum(dim=1)
    den = region.flatten(1).sum(dim=1).clamp_min(1.0)
    return num / den


def _merge_regions(*regions: torch.Tensor) -> torch.Tensor:
    out = regions[0].clone().clamp(0.0, 1.0)
    for r in regions[1:]:
        out = torch.maximum(out, _as_b1hw(r).clamp(0.0, 1.0))
    return out



def _prob_bce_dice(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = _as_b1hw(pred).clamp(EPS, 1.0 - EPS)
    target = _as_b1hw(target).clamp(0.0, 1.0)
    bce = F.binary_cross_entropy(pred, target)
    inter = (pred * target).flatten(1).sum(dim=1)
    den = pred.flatten(1).sum(dim=1) + target.flatten(1).sum(dim=1)
    dice = 1.0 - (2.0 * inter + EPS) / (den + EPS)
    return bce + dice.mean()


def _boundary_l1(pred: torch.Tensor, target: torch.Tensor, radius: int = 1) -> torch.Tensor:
    pred_b = _soft_boundary(_as_b1hw(pred), radius=radius)
    target_b = _soft_boundary(_as_b1hw(target), radius=radius)
    return F.l1_loss(pred_b, target_b)


def _v488_pixel_targets(
    candidate_probs: torch.Tensor,
    gt: torch.Tensor,
    tie_edit_penalty: float,
) -> Dict[str, torch.Tensor]:
    """Build correction-only PWO supervision with Preserve-favouring ties."""
    cand = candidate_probs.detach().clamp(EPS, 1.0 - EPS)
    b, k, h, w = cand.shape
    gt1 = _as_b1hw(gt).clamp(0.0, 1.0)
    gt_k = gt1.expand(-1, k, -1, -1)
    c0 = cand[:, :1]
    c0_hard = c0 >= 0.5
    cand_hard = cand >= 0.5
    gt_hard = gt_k >= 0.5
    base_correct = c0_hard == gt_hard[:, :1]
    candidate_correct = cand_hard == gt_hard
    nonbase_corrects_error = candidate_correct[:, 1:] & (~base_correct.expand(-1, k - 1, -1, -1))
    any_correction = nonbase_corrects_error.any(dim=1, keepdim=True)

    target_dist = cand.new_zeros((b, k, h, w))
    preserve_region = base_correct | (~any_correction)
    target_dist[:, 0:1] = preserve_region.to(cand.dtype)
    if k > 1:
        edit = (cand[:, 1:] - c0).abs()
        correction_weight = nonbase_corrects_error.to(cand.dtype) * torch.exp(-float(tie_edit_penalty) * edit)
        correction_weight = correction_weight / correction_weight.sum(dim=1, keepdim=True).clamp_min(EPS)
        target_dist[:, 1:] = correction_weight * (~preserve_region).to(cand.dtype)

    # Hard PWO is exactly GT where some non-base candidate corrects a Base
    # error; elsewhere it is Preserve/C0.  This is achievable by a pixel-wise
    # candidate selector and is the relevant upper-bound target for M2.
    pwo_target = torch.where(any_correction, gt1, (c0_hard.to(cand.dtype))).detach()
    base_abs_error = (c0 - gt1).abs()
    candidate_abs_error = (cand - gt_k).abs()
    effect_target = (base_abs_error - candidate_abs_error).detach()

    evidence_target = cand.new_zeros((b, k, h, w))
    evidence_target[:, 0:1] = preserve_region.to(cand.dtype)
    if k > 1:
        evidence_target[:, 1:] = nonbase_corrects_error.to(cand.dtype)

    return {
        "target_dist": target_dist,
        "pwo_target": pwo_target,
        "effect_target": effect_target,
        "evidence_target": evidence_target,
        "any_correction": any_correction.to(cand.dtype),
        "base_correct": base_correct.to(cand.dtype),
    }


def _compute_v488_m2m3_loss(
    cfg: Any,
    masks: torch.Tensor,
    aux: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """V488 loss: train only pixel composer M2 and safe deployer M3."""
    gt = (_as_b1hw(masks) > 0.5).float()
    candidate_probs = aux.get("candidate_probs")
    if not isinstance(candidate_probs, torch.Tensor) or candidate_probs.ndim != 4:
        raise RuntimeError("V488 requires aux['candidate_probs'] with shape [B,K,H,W].")
    candidate_probs = candidate_probs.detach().clamp(EPS, 1.0 - EPS)
    c0 = candidate_probs[:, :1]

    required = (
        "m2_pixel_effect",
        "m2_pixel_evidence_logit",
        "m2_pixel_log_variance",
        "m2_pixel_policy_logits",
        "m2_pixel_weights",
        "m2_fused_probs",
        "m3_gate_logit",
        "m3_gate_prob",
        "final_probs",
    )
    missing = [key for key in required if key not in aux]
    if missing:
        raise RuntimeError("V488 loss missing M2/M3 outputs: " + str(missing))

    targets = _v488_pixel_targets(
        candidate_probs,
        gt,
        tie_edit_penalty=float(_m1(cfg, "V488_PWO_TIE_EDIT_PENALTY", 2.0)),
    )
    target_dist = targets["target_dist"]
    effect_target = targets["effect_target"]
    evidence_target = targets["evidence_target"]
    pwo_target = targets["pwo_target"]

    effect_pred = aux["m2_pixel_effect"]
    evidence_logit = aux["m2_pixel_evidence_logit"]
    log_variance = aux["m2_pixel_log_variance"].clamp(-6.0, 4.0)
    policy_logits = aux["m2_pixel_policy_logits"]
    m2_fused = _as_b1hw(aux["m2_fused_probs"])
    final_probs = _as_b1hw(aux["final_probs"])

    log_policy = F.log_softmax(policy_logits, dim=1)
    policy_loss = -(target_dist * log_policy).sum(dim=1).mean()
    effect_reg = F.smooth_l1_loss(effect_pred, effect_target, beta=0.10)
    effect_residual = (effect_pred.detach() - effect_target).abs()
    uncertainty_target = effect_residual.clamp(0.0, 1.0)
    uncertainty_pred = torch.sqrt(F.softplus(log_variance) + EPS)
    uncertainty_loss = F.smooth_l1_loss(uncertainty_pred, uncertainty_target, beta=0.10)

    # Sparse correction labels are strongly imbalanced; positive weighting is
    # bounded to avoid unstable gradients on tiny lesions.
    pos_fraction = evidence_target.mean().detach().clamp_min(1.0e-4)
    evidence_pos_weight = float(_m1(cfg, "V488_M2_EVIDENCE_POS_WEIGHT", 4.0))
    dynamic_pos_weight = float((1.0 / pos_fraction).clamp_max(12.0).detach().cpu().item())
    evidence_weight = 1.0 + evidence_target * min(evidence_pos_weight, dynamic_pos_weight)
    evidence_loss = F.binary_cross_entropy_with_logits(
        evidence_logit,
        evidence_target,
        weight=evidence_weight,
    )

    m2_pwo_loss = _prob_bce_dice(m2_fused, pwo_target)
    m2_gt_loss = _prob_bce_dice(m2_fused, gt)
    boundary_radius = int(_m1(cfg, "V488_BOUNDARY_RADIUS", 1))
    m2_boundary_loss = _boundary_l1(m2_fused, gt, radius=boundary_radius)

    # M3 learns whether M2 is locally better than Preserve.  The target is
    # intentionally derived from GT only inside the training loss; it is never
    # supplied to the inference graph.
    base_error = (c0 - gt).abs()
    m2_error = (m2_fused.detach() - gt).abs()
    deploy_margin = float(_m1(cfg, "V488_M3_TARGET_MARGIN", 0.01))
    deploy_target = (m2_error + deploy_margin < base_error).float()
    gate_logit = _as_b1hw(aux["m3_gate_logit"])
    gate_pos_weight = float(_m1(cfg, "V488_M3_GATE_POS_WEIGHT", 3.0))
    gate_weight = 1.0 + gate_pos_weight * deploy_target
    m3_gate_loss = F.binary_cross_entropy_with_logits(
        gate_logit,
        deploy_target,
        weight=gate_weight,
    )
    m3_seg_loss = _prob_bce_dice(final_probs, gt)
    m3_boundary_loss = _boundary_l1(final_probs, gt, radius=boundary_radius)
    final_error = (final_probs - gt).abs()
    m3_no_harm_loss = F.relu(final_error - base_error.detach()).mean()
    m3_rollback_loss = (
        (final_probs - c0).abs() * (1.0 - deploy_target)
    ).sum() / (1.0 - deploy_target).sum().clamp_min(1.0)

    total = (
        float(_m1(cfg, "V488_M2_POLICY_WEIGHT", 1.0)) * policy_loss
        + float(_m1(cfg, "V488_M2_EFFECT_WEIGHT", 0.5)) * effect_reg
        + float(_m1(cfg, "V488_M2_EVIDENCE_LOSS_WEIGHT", 0.5)) * evidence_loss
        + float(_m1(cfg, "V488_M2_UNCERTAINTY_LOSS_WEIGHT", 0.10)) * uncertainty_loss
        + float(_m1(cfg, "V488_M2_PWO_SEG_WEIGHT", 1.0)) * m2_pwo_loss
        + float(_m1(cfg, "V488_M2_GT_SEG_WEIGHT", 0.5)) * m2_gt_loss
        + float(_m1(cfg, "V488_M2_BOUNDARY_WEIGHT", 0.25)) * m2_boundary_loss
        + float(_m1(cfg, "V488_M3_GATE_LOSS_WEIGHT", 1.0)) * m3_gate_loss
        + float(_m1(cfg, "V488_M3_SEG_WEIGHT", 1.0)) * m3_seg_loss
        + float(_m1(cfg, "V488_M3_BOUNDARY_WEIGHT", 0.35)) * m3_boundary_loss
        + float(_m1(cfg, "V488_M3_NO_HARM_WEIGHT", 0.75)) * m3_no_harm_loss
        + float(_m1(cfg, "V488_M3_ROLLBACK_WEIGHT", 0.25)) * m3_rollback_loss
    )

    with torch.no_grad():
        base_dice = _soft_dice_probs((c0 >= 0.5).float(), gt)[:, 0]
        candidate_hard = (candidate_probs >= 0.5).float()
        slot_dice = _soft_dice_probs(candidate_hard, gt)
        global_oracle = slot_dice.max(dim=1).values
        pwo_dice = _soft_dice_probs((pwo_target >= 0.5).float(), gt)[:, 0]
        m2_dice = _soft_dice_probs((m2_fused >= 0.5).float(), gt)[:, 0]
        final_dice = _soft_dice_probs((final_probs >= 0.5).float(), gt)[:, 0]
        nonbase_weight = aux["m2_pixel_weights"][:, 1:].sum(dim=1).mean()
        gate_rate = aux["m3_gate_prob"].mean()
        correction_coverage = targets["any_correction"].mean()

    diag = {
        "v488_total_loss": total.detach(),
        "v488_m2_policy_loss": policy_loss.detach(),
        "v488_m2_effect_loss": effect_reg.detach(),
        "v488_m2_evidence_loss": evidence_loss.detach(),
        "v488_m2_uncertainty_loss": uncertainty_loss.detach(),
        "v488_m2_pwo_seg_loss": m2_pwo_loss.detach(),
        "v488_m2_gt_seg_loss": m2_gt_loss.detach(),
        "v488_m2_boundary_loss": m2_boundary_loss.detach(),
        "v488_m3_gate_loss": m3_gate_loss.detach(),
        "v488_m3_seg_loss": m3_seg_loss.detach(),
        "v488_m3_boundary_loss": m3_boundary_loss.detach(),
        "v488_m3_no_harm_loss": m3_no_harm_loss.detach(),
        "v488_m3_rollback_loss": m3_rollback_loss.detach(),
        "v488_base_dice": base_dice.mean().detach(),
        "v488_global_oracle_dice": global_oracle.mean().detach(),
        "v488_pwo_dice": pwo_dice.mean().detach(),
        "v488_pwo_gap_over_global": (pwo_dice - global_oracle).mean().detach(),
        "v488_m2_dice": m2_dice.mean().detach(),
        "v488_m3_final_dice": final_dice.mean().detach(),
        "v488_m2_gain_vs_base": (m2_dice - base_dice).mean().detach(),
        "v488_m3_gain_vs_m2": (final_dice - m2_dice).mean().detach(),
        "v488_nonbase_weight": nonbase_weight.detach(),
        "v488_m3_gate_rate": gate_rate.detach(),
        "v488_correctable_pixel_rate": correction_coverage.detach(),
        # Compatibility aliases used by existing log aggregation.
        "v485_m2_local_loss": (policy_loss + effect_reg + evidence_loss).detach(),
        "v484_m2_local_loss": (policy_loss + effect_reg + evidence_loss).detach(),
        "v484_m3_regret_loss": (m3_gate_loss + m3_no_harm_loss).detach(),
    }
    return total, diag

def _v489_dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    mask = _as_b1hw(mask).clamp(0.0, 1.0)
    radius = max(0, int(radius))
    if radius <= 0:
        return mask
    kernel = 2 * radius + 1
    return F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=radius)


def _v489_masked_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    value = _as_b1hw(value)
    weight = _as_b1hw(weight).to(value.dtype).clamp_min(0.0)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _v489_balanced_soft_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    region: torch.Tensor,
    positive_threshold: float = 0.05,
) -> torch.Tensor:
    logits = _as_b1hw(logits)
    target = _as_b1hw(target).clamp(0.0, 1.0)
    region = _as_b1hw(region).clamp(0.0, 1.0)
    element = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    positive = region * (target > float(positive_threshold)).to(region.dtype)
    negative = region * (target <= float(positive_threshold)).to(region.dtype)
    terms = []
    if bool((positive.sum() > 0).item()):
        terms.append(_v489_masked_mean(element, positive))
    if bool((negative.sum() > 0).item()):
        terms.append(_v489_masked_mean(element, negative))
    if not terms:
        return logits.sum() * 0.0
    return torch.stack(terms).mean()



def _v495_balanced_risk_sign_loss(
    predicted_proposal_risk: torch.Tensor,
    teacher_proposal_risk: torch.Tensor,
    relevance: torch.Tensor,
    margin: float,
    temperature: float,
) -> Dict[str, torch.Tensor]:
    """Directly align M3 training with the deployed ``risk < 0`` decision.

    The teacher risk is Preserve-relative, so its sign has the exact deployment
    semantics: negative means that the routed
    proposal is safer/better than Preserve.  Pixels inside ``[-margin, margin]``
    are intentionally excluded because their ordering is not stable enough to
    justify a hard decision. Positive and negative class masses are averaged
    separately to prevent the dominant reject class from hiding sparse useful
    edits.
    """
    predicted = _as_b1hw(predicted_proposal_risk)
    teacher = _as_b1hw(teacher_proposal_risk).detach()
    relevance = _as_b1hw(relevance).detach().clamp(0.0, 1.0)
    margin = max(0.0, float(margin))
    temperature = max(EPS, float(temperature))

    target = (teacher < -margin).to(predicted.dtype)
    valid = (teacher.abs() > margin).to(predicted.dtype) * relevance
    logits = -predicted / temperature
    element = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    positive_region = valid * target
    negative_region = valid * (1.0 - target)
    terms = []
    if bool((positive_region.sum() > 0).item()):
        terms.append(_v489_masked_mean(element, positive_region))
    if bool((negative_region.sum() > 0).item()):
        terms.append(_v489_masked_mean(element, negative_region))
    loss = torch.stack(terms).mean() if terms else predicted.sum() * 0.0

    with torch.no_grad():
        mass = valid.sum().clamp_min(1.0)
        prediction = (predicted < 0.0).to(target.dtype)
        accuracy = (
            (prediction == target).to(valid.dtype) * valid
        ).sum() / mass
        positive_rate = (target * valid).sum() / mass
        supervision_fraction = (valid > 0.0).to(valid.dtype).mean()

    return {
        "loss": loss,
        "accuracy": accuracy,
        "positive_rate": positive_rate,
        "supervision_fraction": supervision_fraction,
    }


def _v490_balanced_continuous_gate_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    positive_relevance: torch.Tensor,
    negative_relevance: torch.Tensor,
) -> torch.Tensor:
    """Balanced Gate supervision on the feasible convex-composition target.

    ``target`` is the optimal interpolation coefficient in [0, 1].  Its value
    must not also be used as the sample weight: a tiny but positive candidate
    gain can require a full coefficient of one.  ``positive_relevance`` carries
    the actual achievable error reduction, while ``negative_relevance`` carries
    harmful or unnecessary edit evidence.
    """
    logits = _as_b1hw(logits)
    target = _as_b1hw(target).clamp(0.0, 1.0)
    positive_relevance = _as_b1hw(positive_relevance).clamp(0.0, 1.0)
    negative_relevance = _as_b1hw(negative_relevance).clamp(0.0, 1.0)
    element = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    positive_membership = positive_relevance
    negative_membership = (1.0 - positive_relevance) * negative_relevance
    terms = []
    if bool((positive_membership.sum() > 0).item()):
        terms.append(_v489_masked_mean(element, positive_membership))
    if bool((negative_membership.sum() > 0).item()):
        terms.append(_v489_masked_mean(element, negative_membership))
    return torch.stack(terms).mean() if terms else logits.sum() * 0.0



def _v492_balanced_benefit_loss(
    logits: torch.Tensor,
    positive_weight: torch.Tensor,
    negative_weight: torch.Tensor,
) -> torch.Tensor:
    """Class-mass-balanced logistic loss for sparse beneficial pixels.

    Positive and negative masses contribute equally regardless of foreground or
    candidate-support prevalence.  Continuous ITE magnitude is used as sample
    weight, so a tiny numerical gain cannot dominate a clinically meaningful
    correction.
    """
    logits = _as_b1hw(logits)
    positive_weight = _as_b1hw(positive_weight).detach().clamp_min(0.0)
    negative_weight = _as_b1hw(negative_weight).detach().clamp_min(0.0)
    positive = (
        F.softplus(-logits) * positive_weight
    ).sum() / positive_weight.sum().clamp_min(EPS)
    negative = (
        F.softplus(logits) * negative_weight
    ).sum() / negative_weight.sum().clamp_min(EPS)
    return 0.5 * (positive + negative)


def _v492_benefit_rank_loss(
    logits: torch.Tensor,
    positive_weight: torch.Tensor,
    negative_weight: torch.Tensor,
) -> torch.Tensor:
    """Threshold-free separation of beneficial and non-beneficial regions."""
    logits = _as_b1hw(logits)
    positive_weight = _as_b1hw(positive_weight).detach().clamp_min(0.0)
    negative_weight = _as_b1hw(negative_weight).detach().clamp_min(0.0)
    if not bool((positive_weight.sum() > 0).item()) or not bool(
        (negative_weight.sum() > 0).item()
    ):
        return logits.sum() * 0.0
    positive_mean = (
        logits * positive_weight
    ).sum() / positive_weight.sum().clamp_min(EPS)
    negative_mean = (
        logits * negative_weight
    ).sum() / negative_weight.sum().clamp_min(EPS)
    return F.softplus(-(positive_mean - negative_mean))

def _v493_mutually_exclusive_benefit_weights(
    positive_relevance: torch.Tensor,
    negative_relevance: torch.Tensor,
    benefit_target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create disjoint Benefit supervision for the shared edit decision.

    A pixel is positive when at least one realizable intervention improves C0.
    It is negative only when no intervention improves C0. Candidate-specific
    harm is supervised by the candidate utility head instead of being allowed to
    contradict the shared existence-of-benefit label.
    """
    positive_relevance = _as_b1hw(positive_relevance).detach().clamp(0.0, 1.0)
    negative_relevance = _as_b1hw(negative_relevance).detach().clamp(0.0, 1.0)
    target = _as_b1hw(benefit_target).detach().to(positive_relevance.dtype)
    target = (target > 0.5).to(positive_relevance.dtype)
    positive_weight = positive_relevance * target
    negative_weight = negative_relevance * (1.0 - target)
    return positive_weight, negative_weight


def _v493_balanced_candidate_utility_loss(
    logits: torch.Tensor,
    positive_weight: torch.Tensor,
    negative_weight: torch.Tensor,
) -> torch.Tensor:
    """Mass-balanced candidate-conditioned utility classification.

    ``logits`` and weights are [B,N,H,W]. Positive and negative supervision is
    mutually exclusive by construction. Each class contributes one half of the
    loss when present, preventing sparse beneficial candidates from being
    overwhelmed by background or by a different harmful candidate.
    """
    if logits.ndim != 4:
        raise ValueError(f"candidate utility logits must be [B,N,H,W], got {tuple(logits.shape)}")
    positive_weight = positive_weight.detach().to(logits.dtype).clamp_min(0.0)
    negative_weight = negative_weight.detach().to(logits.dtype).clamp_min(0.0)
    if positive_weight.shape != logits.shape or negative_weight.shape != logits.shape:
        raise ValueError("candidate utility weight shapes must match logits")
    terms = []
    if bool((positive_weight.sum() > 0).item()):
        terms.append(
            (F.softplus(-logits) * positive_weight).sum()
            / positive_weight.sum().clamp_min(EPS)
        )
    if bool((negative_weight.sum() > 0).item()):
        terms.append(
            (F.softplus(logits) * negative_weight).sum()
            / negative_weight.sum().clamp_min(EPS)
        )
    return torch.stack(terms).mean() if terms else logits.sum() * 0.0



def _v498_balanced_action_loss(
    action_logits: torch.Tensor,
    action_target: torch.Tensor,
) -> torch.Tensor:
    """Parameter-free class-balanced Preserve/candidate action loss.

    The Preserve class dominates spatially, so a plain pixel mean would learn
    an almost-always-Preserve classifier. This objective computes a mean loss
    inside every action class present in the current batch and then averages
    those class means. It adapts automatically to dataset and batch frequencies
    without a hand-tuned Preserve weight.
    """
    if action_logits.ndim != 4:
        raise ValueError(
            "V498 action_logits must be [B,A,H,W], got "
            f"{tuple(action_logits.shape)}"
        )
    if action_target.ndim != 3:
        raise ValueError(
            "V498 action_target must be [B,H,W], got "
            f"{tuple(action_target.shape)}"
        )
    element = F.cross_entropy(
        action_logits, action_target.long(), reduction="none"
    )
    class_terms = []
    for index in range(action_logits.shape[1]):
        mask = action_target == index
        if bool(mask.any().item()):
            class_terms.append(element[mask].mean())
    if not class_terms:
        return action_logits.sum() * 0.0
    return torch.stack(class_terms).mean()


def _v500_balanced_presence_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Sparse binary candidate-presence loss without a hand-tuned class weight."""
    logits = _as_b1hw(logits)
    target = _as_b1hw(target).to(logits.dtype).clamp(0.0, 1.0)
    element = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    positive = target > 0.5
    negative = ~positive
    terms = []
    if bool(positive.any().item()):
        terms.append(element[positive].mean())
    if bool(negative.any().item()):
        terms.append(element[negative].mean())
    balanced_bce = (
        torch.stack(terms).mean() if terms else logits.sum() * 0.0
    )
    probability = torch.sigmoid(logits)
    inter = (probability * target).flatten(1).sum(dim=1)
    den = probability.flatten(1).sum(dim=1) + target.flatten(1).sum(dim=1)
    dice = 1.0 - (2.0 * inter + EPS) / (den + EPS)
    return balanced_bce + dice.mean()


def _v499_causal_action_targets(
    c0: torch.Tensor,
    nonbase: torch.Tensor,
    gt: torch.Tensor,
    candidate_gain: torch.Tensor,
    edit_eps: float,
    boundary_radius: int,
) -> Dict[str, torch.Tensor]:
    """Build Preserve/candidate targets only on factual error geometry.

    V498 marked a candidate positive whenever its continuous probability error
    was smaller than C0 by any non-zero amount.  Because the candidate bank
    contains interventions in both directions and the support maps have soft
    tails, almost every pixel had a tiny positive-gain candidate; Preserve
    therefore disappeared from the action teacher.

    V499 reuses the *existing* M1 error semantics instead of introducing a new
    dataset-specific gain threshold.  A candidate action is eligible only when:

    1. the factual hard mask disagrees with GT, or its hard boundary disagrees;
    2. the candidate makes a non-trivial edit according to the already existing
       ``V489_DECISION_EDIT_EPS`` contract; and
    3. that candidate has positive realizable gain.

    Everywhere else the reference action is Preserve.  This target is used only
    during training; M2 still emits one complete non-Base proposal and M3 remains
    the sole irreversible Preserve-vs-proposal decision.
    """
    c0 = _as_b1hw(c0).detach().clamp(EPS, 1.0 - EPS)
    if nonbase.ndim != 4 or nonbase.shape[1] < 1:
        raise ValueError("V499 nonbase must be [B,N,H,W] with N>=1")
    nonbase = nonbase.detach().clamp(EPS, 1.0 - EPS)
    gt = _as_b1hw(gt).detach().clamp(0.0, 1.0)
    if candidate_gain.shape != nonbase.shape:
        raise ValueError(
            "V499 candidate_gain shape must match nonbase: "
            f"{tuple(candidate_gain.shape)} vs {tuple(nonbase.shape)}"
        )
    candidate_gain = candidate_gain.detach().clamp_min(0.0)

    c0_hard = (c0 >= 0.5).to(c0.dtype)
    gt_hard = (gt >= 0.5).to(c0.dtype)
    hard_error = (c0_hard != gt_hard).to(c0.dtype)
    radius = max(1, int(boundary_radius))
    boundary_error = (
        _soft_boundary(c0_hard, radius=radius)
        - _soft_boundary(gt_hard, radius=radius)
    ).abs().clamp(0.0, 1.0)
    causal_region = torch.maximum(hard_error, boundary_error).detach()

    candidate_edit = (nonbase - c0.expand_as(nonbase)).abs()
    meaningful_edit = candidate_edit > max(float(edit_eps), EPS)
    eligible = meaningful_edit & (causal_region.expand_as(nonbase) > 0.0)
    causal_gain = candidate_gain * eligible.to(candidate_gain.dtype)
    best_gain, best_index = causal_gain.max(dim=1, keepdim=True)
    has_action = best_gain > EPS
    action_target = torch.where(
        has_action,
        best_index + 1,
        torch.zeros_like(best_index),
    )[:, 0].long()

    route_target = F.one_hot(
        best_index[:, 0], num_classes=nonbase.shape[1]
    ).permute(0, 3, 1, 2).to(nonbase.dtype)
    uniform = torch.full_like(route_target, 1.0 / float(nonbase.shape[1]))
    route_target = torch.where(
        has_action.expand_as(route_target), route_target, uniform
    )

    best_candidate = nonbase.gather(1, best_index)
    oracle_target = torch.where(has_action, best_candidate, c0)

    # V500 also defines a route teacher where no beneficial candidate exists.
    # Among candidates that actually edit C0, select the minimum-error option.
    # This does not accept the intervention; it only teaches the mandatory M2
    # proposal to be the least-regret proposal available to M3.
    candidate_abs_error = (nonbase - gt.expand_as(nonbase)).abs()
    candidate_edit = (nonbase - c0.expand_as(nonbase)).abs()
    meaningful = candidate_edit > max(float(edit_eps), EPS)
    has_meaningful = meaningful.any(dim=1, keepdim=True)
    large = torch.finfo(candidate_abs_error.dtype).max / 16.0
    safe_error = candidate_abs_error.masked_fill(~meaningful, large)
    safe_min_error, safe_index = safe_error.min(dim=1, keepdim=True)
    safe_index = torch.where(
        has_meaningful, safe_index, torch.zeros_like(safe_index)
    )
    safe_route_target = F.one_hot(
        safe_index[:, 0], num_classes=nonbase.shape[1]
    ).permute(0, 3, 1, 2).to(nonbase.dtype)
    uniform_safe = torch.full_like(
        safe_route_target, 1.0 / float(nonbase.shape[1])
    )
    safe_route_target = torch.where(
        has_meaningful.expand_as(safe_route_target),
        safe_route_target,
        uniform_safe,
    )
    max_meaningful_edit = (
        candidate_edit * meaningful.to(candidate_edit.dtype)
    ).max(dim=1, keepdim=True).values
    safe_route_weight = (
        max_meaningful_edit * has_meaningful.to(candidate_edit.dtype)
    ).detach()
    safe_min_error = torch.where(
        has_meaningful, safe_min_error, (c0 - gt).abs()
    )

    return {
        "action_target": action_target.detach(),
        "route_target": route_target.detach(),
        "oracle_target": oracle_target.detach(),
        "causal_region": causal_region.detach(),
        "eligible_candidates": eligible.detach(),
        "causal_gain": causal_gain.detach(),
        "best_gain": best_gain.detach(),
        "best_index": best_index.detach(),
        "candidate_target_rate": has_action.float().mean().detach(),
        "causal_region_rate": causal_region.mean().detach(),
        "eligible_candidate_rate": eligible.float().mean().detach(),
        "safe_route_target": safe_route_target.detach(),
        "safe_route_weight": safe_route_weight.detach(),
        "safe_route_index": safe_index.detach(),
        "safe_min_error": safe_min_error.detach(),
        "meaningful_candidates": meaningful.detach(),
        "meaningful_edit_rate": meaningful.float().mean().detach(),
    }


def _v493_imagewise_relative_risk_normalize(
    risk: torch.Tensor,
    relevance: torch.Tensor,
    floor: float = 1.0e-3,
    clip: float = 8.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize binary relative risk with one shared scale per image.

    V492 divided every two-expert pixel by its own absolute risk. With only
    Preserve and M2 this degenerates to sign(r), erasing whether a proposal is
    marginally or strongly beneficial. V493 estimates a detached weighted mean
    absolute proposal risk over the whole relevant image. Division by this
    positive shared scale preserves both sign and within-image magnitude ratios.
    """
    if risk.ndim != 4 or risk.shape[1] < 2:
        raise ValueError("risk must be [B,E,H,W] with Preserve plus proposal")
    relevance = _as_b1hw(relevance).to(risk.dtype).clamp(0.0, 1.0)
    centered = risk - risk[:, :1]
    proposal = centered[:, 1:]
    rel = relevance.expand(-1, proposal.shape[1], -1, -1)
    rel_mass = rel.sum(dim=(1, 2, 3), keepdim=True)
    weighted = (proposal.abs() * rel).sum(dim=(1, 2, 3), keepdim=True)
    weighted_scale = weighted / rel_mass.clamp_min(EPS)
    fallback_scale = proposal.abs().mean(dim=(1, 2, 3), keepdim=True)
    scale = torch.where(rel_mass > EPS, weighted_scale, fallback_scale)
    scale = scale.detach().clamp_min(max(float(floor), EPS))
    normalized = centered / scale
    if float(clip) > 0:
        normalized = normalized.clamp(-float(clip), float(clip))
    return normalized, scale


def _v490_optimal_convex_targets(
    c0: torch.Tensor,
    nonbase: torch.Tensor,
    gt: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Exact stop-gradient teacher for the constrained M2 convex composer.

    M2 can only produce ``C0 + g * (sum_k q_k Ck - C0)``.  For every candidate
    this function projects the ground-truth probability onto the line segment
    from C0 to Ck.  The candidate with the largest achievable error reduction
    supplies a one-hot route target, and its projection coefficient supplies
    the Gate target.  The resulting teacher is exactly realizable by M2.
    """
    c0 = _as_b1hw(c0).detach().clamp(EPS, 1.0 - EPS)
    if nonbase.ndim != 4 or nonbase.shape[1] < 1:
        raise ValueError("nonbase must be [B,N,H,W] with N>=1")
    nonbase = nonbase.detach().clamp(EPS, 1.0 - EPS)
    gt = _as_b1hw(gt).detach().clamp(0.0, 1.0)
    target = gt.expand_as(nonbase)
    base = c0.expand_as(nonbase)
    delta = nonbase - base

    alpha = ((target - base) * delta) / delta.square().clamp_min(EPS)
    alpha = alpha.clamp(0.0, 1.0)
    projected = base + alpha * delta

    base_error = (c0 - gt).abs()
    projected_error = (projected - target).abs()
    achievable_gain = (
        base_error.expand_as(projected_error) - projected_error
    ).clamp_min(0.0)
    candidate_relative_gain = (
        achievable_gain / base_error.expand_as(achievable_gain).clamp_min(EPS)
    ).clamp(0.0, 1.0)

    best_gain, best_index = achievable_gain.max(dim=1, keepdim=True)
    best_relative_gain = candidate_relative_gain.gather(1, best_index)
    best_alpha = alpha.gather(1, best_index)
    valid = best_gain > EPS
    gate_target = torch.where(valid, best_alpha, torch.zeros_like(best_alpha))

    route_target = F.one_hot(
        best_index[:, 0], num_classes=nonbase.shape[1]
    ).permute(0, 3, 1, 2).to(nonbase.dtype)
    uniform = torch.full_like(route_target, 1.0 / float(nonbase.shape[1]))
    route_target = torch.where(valid.expand_as(route_target), route_target, uniform)

    best_candidate = nonbase.gather(1, best_index)
    soft_oracle_target = (
        c0 + gate_target * (best_candidate - c0)
    ).clamp(EPS, 1.0 - EPS)

    return {
        "gate_target": gate_target.detach(),
        "route_target": route_target.detach(),
        "soft_oracle_target": soft_oracle_target.detach(),
        "positive_relevance": best_relative_gain.detach(),
        "best_achievable_gain": best_gain.detach(),
        "candidate_achievable_gain": achievable_gain.detach(),
        "candidate_relative_gain": candidate_relative_gain.detach(),
        "candidate_alpha": alpha.detach(),
        "best_index": best_index.detach(),
    }


def _v490_normalize_relative_risk(
    risk: torch.Tensor,
    floor: float = 1.0e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Centre and scale expert risk without changing the hard argmin.

    The per-pixel scale is detached so the network cannot lower the loss by
    manipulating its denominator.  Division by a positive scalar preserves the
    selected expert while giving list/rank distillation a dimensionless scale.
    """
    if risk.ndim != 4 or risk.shape[1] < 2:
        raise ValueError("risk must be [B,E,H,W] with E>=2")
    centered = risk - risk[:, :1]
    scale = centered[:, 1:].abs().mean(dim=1, keepdim=True).detach()
    scale = scale.clamp_min(max(float(floor), EPS))
    return centered / scale, scale


def _v490_context_scales(cfg: Any) -> tuple[int, ...]:
    raw = _m1(cfg, "V490_M3_CONTEXT_SCALES", [1, 3, 7, 15])
    if isinstance(raw, str):
        raw = [int(part.strip()) for part in raw.split(",") if part.strip()]
    scales = tuple(sorted({max(1, int(scale)) | 1 for scale in raw}))
    return scales or (1, 3, 7, 15)


def _v490_multiscale_relative_risk(
    value: torch.Tensor,
    relevance: torch.Tensor,
    scales: tuple[int, ...],
) -> torch.Tensor:
    """Use the same continuous multi-scale geometry as the V490.2 selector."""
    if value.ndim != 4 or relevance.ndim != 4 or relevance.shape[1] != 1:
        raise ValueError("value must be [B,E,H,W] and relevance [B,1,H,W]")
    relevance = relevance.to(value.dtype).clamp(0.0, 1.0)
    outputs = []
    for scale in scales:
        padding = scale // 2
        mass = F.avg_pool2d(
            relevance, kernel_size=scale, stride=1, padding=padding
        )
        numerator = F.avg_pool2d(
            value * relevance,
            kernel_size=scale,
            stride=1,
            padding=padding,
        )
        outputs.append(
            torch.where(
                mass > EPS,
                numerator / mass.clamp_min(EPS),
                torch.zeros_like(numerator),
            )
        )
    result = torch.stack(outputs, dim=0).mean(dim=0)
    return result - result[:, :1]


def _v490_dense_risk_losses(
    predicted_risk: torch.Tensor,
    teacher_risk: torch.Tensor,
    relevance: torch.Tensor,
    teacher_temperature: float,
    student_temperature: float,
    scale_invariant: bool = False,
    risk_scale_floor: float = 1.0e-3,
) -> Dict[str, torch.Tensor]:
    """Dense relative-risk regression, list distillation, and ranking.

    The weighting field is continuous candidate edit relevance.  It is never
    thresholded into connected components, so no region is discarded and two
    different candidate error modes are not forced to share one regional label.
    """
    if predicted_risk.shape != teacher_risk.shape:
        raise ValueError(
            f"risk shape mismatch: {tuple(predicted_risk.shape)} vs "
            f"{tuple(teacher_risk.shape)}"
        )
    batch, experts, height, width = predicted_risk.shape
    relevance = _as_b1hw(relevance).to(predicted_risk.dtype).clamp(0.0, 1.0)
    expert_weight = relevance.expand(-1, experts, -1, -1)
    expert_denom = expert_weight.sum().clamp_min(1.0)

    risk_element = F.smooth_l1_loss(
        predicted_risk,
        teacher_risk,
        beta=0.10,
        reduction="none",
    )
    risk_regression = (risk_element * expert_weight).sum() / expert_denom
    risk_mae = (
        (predicted_risk.detach() - teacher_risk).abs() * expert_weight
    ).sum() / expert_denom

    teacher_temperature = max(float(teacher_temperature), 1.0e-4)
    student_temperature = max(float(student_temperature), 1.0e-4)
    if bool(scale_invariant):
        teacher_for_order, teacher_scale = _v490_normalize_relative_risk(
            teacher_risk, floor=risk_scale_floor
        )
        predicted_for_order, predicted_scale = _v490_normalize_relative_risk(
            predicted_risk, floor=risk_scale_floor
        )
    else:
        teacher_for_order = teacher_risk
        predicted_for_order = predicted_risk
        teacher_scale = teacher_risk.new_ones((batch, 1, height, width))
        predicted_scale = predicted_risk.new_ones((batch, 1, height, width))

    teacher_distribution = F.softmax(
        -teacher_for_order / teacher_temperature, dim=1
    )
    teacher_log_distribution = teacher_distribution.clamp_min(EPS).log()
    student_log_distribution = F.log_softmax(
        -predicted_for_order / student_temperature, dim=1
    )
    # KL has the same student gradient as cross-entropy but removes the
    # teacher-entropy constant from the diagnostic value.
    list_element = (
        teacher_distribution
        * (teacher_log_distribution - student_log_distribution)
    ).sum(dim=1, keepdim=True)
    list_loss = (list_element * relevance).sum() / relevance.sum().clamp_min(1.0)
    teacher_entropy = -(
        teacher_distribution * teacher_log_distribution
    ).sum(dim=1, keepdim=True)
    teacher_entropy = (
        teacher_entropy * relevance
    ).sum() / relevance.sum().clamp_min(1.0)

    rank_terms = []
    pair_correct_num = predicted_risk.new_zeros(())
    pair_correct_den = predicted_risk.new_zeros(())
    for left in range(experts):
        for right in range(left + 1, experts):
            # Positive true/predicted gap means ``left`` has lower risk.
            true_gap = teacher_for_order[:, right] - teacher_for_order[:, left]
            predicted_gap = (
                predicted_for_order[:, right] - predicted_for_order[:, left]
            )
            magnitude = true_gap.abs().detach() * relevance[:, 0]
            preference = true_gap.sign().detach()
            element = F.softplus(
                -preference * predicted_gap / student_temperature
            )
            rank_terms.append(
                (element * magnitude).sum() / magnitude.sum().clamp_min(1.0)
            )
            valid = magnitude > 0.0
            if bool(valid.any().item()):
                pair_correct_num = pair_correct_num + (
                    (predicted_gap.sign() == preference).to(magnitude.dtype)
                    * magnitude
                ).sum()
                pair_correct_den = pair_correct_den + magnitude.sum()
    rank_loss = (
        torch.stack(rank_terms).mean()
        if rank_terms else predicted_risk.sum() * 0.0
    )
    pairwise_accuracy = pair_correct_num / pair_correct_den.clamp_min(1.0)

    oracle_index = teacher_risk.argmin(dim=1)
    predicted_index = predicted_risk.argmin(dim=1)
    gather_predicted = predicted_index[:, None]
    selected_teacher_risk = teacher_risk.gather(1, gather_predicted)[:, 0]
    oracle_teacher_risk = teacher_risk.min(dim=1).values
    relevance_2d = relevance[:, 0]
    pixel_denom = relevance_2d.sum().clamp_min(1.0)
    selection_accuracy = (
        (predicted_index == oracle_index).to(relevance_2d.dtype)
        * relevance_2d
    ).sum() / pixel_denom
    oracle_regret = (
        (selected_teacher_risk - oracle_teacher_risk).clamp_min(0.0)
        * relevance_2d
    ).sum() / pixel_denom

    oracle_one_hot = F.one_hot(
        oracle_index, num_classes=experts
    ).permute(0, 3, 1, 2).to(predicted_risk.dtype)
    predicted_one_hot = F.one_hot(
        predicted_index, num_classes=experts
    ).permute(0, 3, 1, 2).to(predicted_risk.dtype)
    histogram_denom = relevance.sum().clamp_min(1.0)
    oracle_hist = (
        oracle_one_hot * relevance
    ).sum(dim=(0, 2, 3)) / histogram_denom
    predicted_hist = (
        predicted_one_hot * relevance
    ).sum(dim=(0, 2, 3)) / histogram_denom

    student_probability = F.softmax(
        -predicted_for_order / student_temperature, dim=1
    )
    entropy = -(
        student_probability.clamp_min(EPS)
        * student_probability.clamp_min(EPS).log()
    ).sum(dim=1, keepdim=True)
    selection_entropy = (entropy * relevance).sum() / relevance.sum().clamp_min(1.0)

    switch_num = predicted_risk.new_zeros(())
    switch_den = predicted_risk.new_zeros(())
    if width > 1:
        horizontal_weight = torch.minimum(
            relevance_2d[:, :, :-1], relevance_2d[:, :, 1:]
        )
        switch_num = switch_num + (
            (predicted_index[:, :, :-1] != predicted_index[:, :, 1:]).to(
                horizontal_weight.dtype
            ) * horizontal_weight
        ).sum()
        switch_den = switch_den + horizontal_weight.sum()
    if height > 1:
        vertical_weight = torch.minimum(
            relevance_2d[:, :-1, :], relevance_2d[:, 1:, :]
        )
        switch_num = switch_num + (
            (predicted_index[:, :-1, :] != predicted_index[:, 1:, :]).to(
                vertical_weight.dtype
            ) * vertical_weight
        ).sum()
        switch_den = switch_den + vertical_weight.sum()
    spatial_switch_rate = switch_num / switch_den.clamp_min(1.0)

    return {
        "risk_regression": risk_regression,
        "risk_mae": risk_mae,
        "list_loss": list_loss,
        "teacher_entropy": teacher_entropy,
        "teacher_risk_scale": teacher_scale.mean(),
        "predicted_risk_scale": predicted_scale.mean(),
        "rank_loss": rank_loss,
        "pairwise_accuracy": pairwise_accuracy,
        "selection_accuracy": selection_accuracy,
        "oracle_regret": oracle_regret,
        "selection_entropy": selection_entropy,
        "spatial_switch_rate": spatial_switch_rate,
        "oracle_hist": oracle_hist,
        "predicted_hist": predicted_hist,
    }


def _v489_region_pool(x: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
    return F.adaptive_avg_pool2d(x, output_size=output_size)


def _v489_loss_ramp(epoch: int | None, start: float, final: float, ramp_epochs: int) -> float:
    current = 0 if epoch is None else max(0, int(epoch))
    ramp_epochs = max(1, int(ramp_epochs))
    progress = min(1.0, float(current + 1) / float(ramp_epochs))
    return float(start) + (float(final) - float(start)) * progress


def _v501_delayed_ramp(
    epoch: int | None,
    start_epoch: int,
    ramp_epochs: int,
    final: float = 1.0,
) -> float:
    """A true delayed curriculum used by V501.

    ``start_epoch`` and ``epoch`` are zero based.  The objective is exactly
    inactive before ``start_epoch`` and then increases linearly to ``final``.
    This differs from the historical V500 ramp where start/final were both 1.
    """
    current = 0 if epoch is None else max(0, int(epoch))
    start_epoch = max(0, int(start_epoch))
    if current < start_epoch:
        return 0.0
    ramp_epochs = max(1, int(ramp_epochs))
    progress = min(1.0, float(current - start_epoch + 1) / float(ramp_epochs))
    return float(final) * progress



def _v502_weighted_local_mean(
    value: torch.Tensor,
    weight: torch.Tensor,
    radius: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stable tensor-only local aggregation used by the V502 teacher."""
    radius = max(0, int(radius))
    if radius == 0:
        return value * (weight > EPS).to(value.dtype), weight
    kernel = 2 * radius + 1
    numerator = F.avg_pool2d(
        value * weight,
        kernel_size=kernel,
        stride=1,
        padding=radius,
    )
    mass = F.avg_pool2d(
        weight,
        kernel_size=kernel,
        stride=1,
        padding=radius,
    )
    return numerator / mass.clamp_min(EPS), mass


def _v502_utility_teacher_targets(
    c0: torch.Tensor,
    nonbase: torch.Tensor,
    gt: torch.Tensor,
    supports: torch.Tensor,
    local_radius: int = 2,
    boundary_radius: int = 1,
    edit_eps: float = 0.005,
    boundary_overlap_floor: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """Build a Base-relative, Pareto-safe soft routing teacher.

    The teacher uses decomposable local absolute-error improvements rather than
    a fictitious per-pixel Dice delta.  Boundary quality is a hard eligibility
    constraint in boundary regions, so a candidate cannot trade a large Dice
    gain for an NSD-damaging boundary regression.
    """
    c0 = _as_b1hw(c0).clamp(EPS, 1.0 - EPS)
    gt = (_as_b1hw(gt) > 0.5).float()
    if nonbase.ndim != 4:
        raise ValueError(f"nonbase must be [B,N,H,W], got {tuple(nonbase.shape)}")
    if supports.shape != nonbase.shape:
        raise ValueError(
            "supports must match nonbase; "
            f"got {tuple(supports.shape)} vs {tuple(nonbase.shape)}"
        )
    n = nonbase.shape[1]
    c0e = c0.expand(-1, n, -1, -1)
    gte = gt.expand(-1, n, -1, -1)
    edit_magnitude = (nonbase - c0e).abs()
    geometry = torch.maximum(
        supports.detach().clamp(0.0, 1.0),
        edit_magnitude.detach(),
    ).clamp(0.0, 1.0)

    base_seg_error = (c0e - gte).abs()
    candidate_seg_error = (nonbase - gte).abs()
    seg_gain_pixel = base_seg_error - candidate_seg_error
    local_seg_gain, local_mass = _v502_weighted_local_mean(
        seg_gain_pixel, geometry, local_radius
    )
    local_base_error, _ = _v502_weighted_local_mean(
        base_seg_error, geometry, local_radius
    )
    local_edit, _ = _v502_weighted_local_mean(
        edit_magnitude, geometry, local_radius
    )

    base_boundary = _soft_boundary(c0, radius=max(1, int(boundary_radius)))
    gt_boundary = _soft_boundary(gt, radius=max(1, int(boundary_radius)))
    candidate_boundary = _soft_boundary(
        nonbase.reshape(-1, 1, *nonbase.shape[-2:]),
        radius=max(1, int(boundary_radius)),
    ).reshape_as(nonbase)
    base_boundary_error = (
        base_boundary - gt_boundary
    ).abs().expand(-1, n, -1, -1)
    candidate_boundary_error = (
        candidate_boundary - gt_boundary.expand(-1, n, -1, -1)
    ).abs()
    boundary_gain_pixel = base_boundary_error - candidate_boundary_error
    local_boundary_gain, _ = _v502_weighted_local_mean(
        boundary_gain_pixel, geometry, local_radius
    )
    local_base_boundary_error, _ = _v502_weighted_local_mean(
        base_boundary_error, geometry, local_radius
    )
    boundary_context = torch.maximum(base_boundary, gt_boundary).expand(
        -1, n, -1, -1
    )
    boundary_overlap, _ = _v502_weighted_local_mean(
        boundary_context, geometry, local_radius
    )
    touches_boundary = boundary_overlap >= max(float(boundary_overlap_floor), 0.0)

    has_geometry = local_mass > EPS
    meaningful_edit = local_edit >= max(float(edit_eps), 0.0)
    segmentation_improves = local_seg_gain > 0.0
    boundary_safe = (~touches_boundary) | (local_boundary_gain >= 0.0)
    eligible = has_geometry & meaningful_edit & segmentation_improves & boundary_safe

    relative_seg_gain = (
        local_seg_gain.clamp_min(0.0) / local_base_error.clamp_min(EPS)
    ).clamp(0.0, 1.0)
    candidate_utility = relative_seg_gain * eligible.to(relative_seg_gain.dtype)
    utility_sum = candidate_utility.sum(dim=1, keepdim=True)
    route_target = torch.where(
        utility_sum > EPS,
        candidate_utility / utility_sum.clamp_min(EPS),
        torch.zeros_like(candidate_utility),
    )
    edit_target = candidate_utility.max(dim=1, keepdim=True).values

    seg_harm = (
        (-local_seg_gain).clamp_min(0.0) / local_base_error.clamp_min(EPS)
    ).clamp(0.0, 1.0)
    boundary_harm = (
        (-local_boundary_gain).clamp_min(0.0)
        / local_base_boundary_error.clamp_min(EPS)
    ).clamp(0.0, 1.0)
    candidate_harm = torch.maximum(
        seg_harm,
        boundary_harm * touches_boundary.to(boundary_harm.dtype),
    )
    candidate_harm = candidate_harm * has_geometry.to(candidate_harm.dtype)

    return {
        "edit_target": edit_target.detach(),
        "route_target": route_target.detach(),
        "candidate_utility": candidate_utility.detach(),
        "candidate_harm": candidate_harm.detach(),
        "eligible": eligible.detach(),
        "local_seg_gain": local_seg_gain.detach(),
        "local_boundary_gain": local_boundary_gain.detach(),
        "teacher_region": (local_mass.max(dim=1, keepdim=True).values > EPS).float().detach(),
    }


def _compute_v502_m2_loss(
    cfg: Any,
    gt: torch.Tensor,
    c0: torch.Tensor,
    nonbase: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    epoch: int | None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """V502 hierarchical utility-teacher soft-routing objective."""
    edit_logit = _as_b1hw(aux["m2_candidate_presence_logit"])
    edit_prob = _as_b1hw(aux["m2_candidate_presence_prob"])
    route_probs = aux["m2_route_probs"]
    route_logits = aux["m2_route_logits"]
    m2_probs = _as_b1hw(aux["m2_training_probs"])
    supports = aux.get("m2_effective_supports")
    if not isinstance(supports, torch.Tensor) or supports.shape != nonbase.shape:
        raise RuntimeError(
            "V502 requires m2_effective_supports aligned with non-Base candidates."
        )

    with torch.no_grad():
        teacher = _v502_utility_teacher_targets(
            c0.detach(),
            nonbase.detach(),
            gt,
            supports.detach(),
            local_radius=int(_m1(cfg, "V502_TEACHER_LOCAL_RADIUS", 2)),
            boundary_radius=int(_m1(cfg, "V502_TEACHER_BOUNDARY_RADIUS", 1)),
            edit_eps=float(_m1(cfg, "V502_TEACHER_EDIT_EPS", 0.005)),
            boundary_overlap_floor=float(
                _m1(cfg, "V502_BOUNDARY_OVERLAP_FLOOR", 0.01)
            ),
        )
    edit_target = teacher["edit_target"]
    route_target = teacher["route_target"]
    candidate_harm = teacher["candidate_harm"]

    eligibility_element = F.smooth_l1_loss(
        edit_prob,
        edit_target,
        reduction="none",
        beta=max(float(_m1(cfg, "V502_ELIGIBILITY_BETA", 0.10)), 1.0e-4),
    )
    eligibility_calibration = eligibility_element.mean()
    eligibility_positive = _v489_masked_mean(
        eligibility_element, edit_target
    )
    eligibility_loss = eligibility_calibration + float(
        _m1(cfg, "V502_POSITIVE_FOCUS_WEIGHT", 0.25)
    ) * eligibility_positive

    route_ce = -(
        route_target * F.log_softmax(route_logits, dim=1)
    ).sum(dim=1, keepdim=True)
    route_loss = _v489_masked_mean(route_ce, edit_target)

    expected_candidate_harm_map = (
        route_probs * candidate_harm
    ).sum(dim=1, keepdim=True)
    expected_candidate_harm_loss = _v489_masked_mean(
        edit_prob * expected_candidate_harm_map,
        teacher["teacher_region"],
    )

    base_seg_error = (c0.detach() - gt).abs()
    m2_seg_error = (m2_probs - gt).abs()
    seg_violation_map = F.relu(m2_seg_error - base_seg_error)
    boundary_radius = int(_m1(cfg, "V502_TEACHER_BOUNDARY_RADIUS", 1))
    base_boundary_error = (
        _soft_boundary(c0.detach(), radius=boundary_radius)
        - _soft_boundary(gt, radius=boundary_radius)
    ).abs()
    m2_boundary_error = (
        _soft_boundary(m2_probs, radius=boundary_radius)
        - _soft_boundary(gt, radius=boundary_radius)
    ).abs()
    boundary_violation_map = F.relu(
        m2_boundary_error - base_boundary_error
    )
    edit_relevance = (m2_probs - c0.detach()).abs().detach().clamp(0.0, 1.0)
    seg_violation = _v489_masked_mean(
        seg_violation_map, edit_relevance
    )
    boundary_violation = _v489_masked_mean(
        boundary_violation_map,
        torch.maximum(
            edit_relevance,
            _soft_boundary(edit_relevance, radius=max(1, boundary_radius)),
        ).clamp(0.0, 1.0),
    )
    final_noharm_loss = torch.maximum(seg_violation, boundary_violation)

    task_loss = _masked_bce_dice(
        m2_probs,
        gt,
        edit_target,
        pos_weight=1.0,
    )

    teacher_weight = _v489_loss_ramp(
        epoch,
        float(_m1(cfg, "V502_TEACHER_WEIGHT_START", 1.0)),
        float(_m1(cfg, "V502_TEACHER_WEIGHT_FINAL", 0.25)),
        int(_m1(cfg, "V502_TEACHER_WEIGHT_RAMP_EPOCHS", 60)),
    )
    total = (
        teacher_weight
        * (
            float(_m1(cfg, "V502_ELIGIBILITY_WEIGHT", 1.0))
            * eligibility_loss
            + float(_m1(cfg, "V502_ROUTE_TEACHER_WEIGHT", 1.0))
            * route_loss
        )
        + float(_m1(cfg, "V502_EXPECTED_HARM_WEIGHT", 1.0))
        * expected_candidate_harm_loss
        + float(_m1(cfg, "V502_FINAL_NOHARM_WEIGHT", 2.0))
        * final_noharm_loss
        + float(_m1(cfg, "V502_MASKED_TASK_WEIGHT", 0.25))
        * task_loss
    )

    # Entropy regularization (Pereyra et al. ICLR 2017):
    # Penalize overly confident (low-entropy) route distributions to prevent
    # the soft router from collapsing to deterministic "preserve" output.
    route_entropy_for_loss = aux.get(
        "m2_route_entropy",
        -(route_probs * torch.log(route_probs.clamp_min(EPS))).sum(
            dim=1, keepdim=True
        ),
    ).mean()
    entropy_weight = float(_m1(cfg, "V502_ROUTE_ENTROPY_WEIGHT", 0.0))
    if entropy_weight > 0.0:
        total = total - entropy_weight * route_entropy_for_loss

    with torch.no_grad():
        positive = edit_target > 0.0
        predicted_positive = edit_prob >= 0.5
        true_positive = (predicted_positive & positive).float().sum()
        precision = true_positive / predicted_positive.float().sum().clamp_min(1.0)
        recall = true_positive / positive.float().sum().clamp_min(1.0)
        route_valid = edit_target[:, 0] > 0.0
        if bool(route_valid.any().item()):
            route_accuracy = (
                route_probs.argmax(dim=1)[route_valid]
                == route_target.argmax(dim=1)[route_valid]
            ).float().mean()
        else:
            route_accuracy = edit_prob.new_zeros(())
        expected_utility = (
            route_probs * teacher["candidate_utility"]
        ).sum(dim=1, keepdim=True)
        soft_beneficial_mass = (edit_prob * expected_utility).mean()
        soft_harmful_mass = (edit_prob * expected_candidate_harm_map).mean()

    diagnostics = {
        "v502_m2_loss": total.detach(),
        "v502_teacher_weight": edit_prob.new_tensor(teacher_weight),
        "v502_route_entropy_weight": edit_prob.new_tensor(entropy_weight),
        "v502_eligibility_loss": eligibility_loss.detach(),
        "v502_eligibility_calibration_loss": eligibility_calibration.detach(),
        "v502_eligibility_positive_loss": eligibility_positive.detach(),
        "v502_route_teacher_loss": route_loss.detach(),
        "v502_expected_candidate_harm_loss": expected_candidate_harm_loss.detach(),
        "v502_final_noharm_loss": final_noharm_loss.detach(),
        "v502_seg_violation": seg_violation.detach(),
        "v502_boundary_violation": boundary_violation.detach(),
        "v502_masked_task_loss": task_loss.detach(),
        "v502_teacher_edit_rate": edit_target.mean().detach(),
        "v502_predicted_edit_rate": edit_prob.mean().detach(),
        "v502_edit_rate_ratio": (
            edit_prob.mean() / edit_target.mean().clamp_min(EPS)
        ).detach(),
        "v502_edit_precision": precision.detach(),
        "v502_edit_recall": recall.detach(),
        "v502_route_accuracy": route_accuracy.detach(),
        "v502_eligible_candidate_rate": teacher["eligible"].float().mean().detach(),
        "v502_soft_beneficial_mass": soft_beneficial_mass.detach(),
        "v502_soft_harmful_mass": soft_harmful_mass.detach(),
        "v502_route_entropy": aux.get(
            "m2_route_entropy",
            -(route_probs * torch.log(route_probs.clamp_min(EPS))).sum(
                dim=1, keepdim=True
            ),
        ).mean().detach(),
        "v502_route_margin": aux.get(
            "m2_route_margin", edit_prob.new_zeros(edit_prob.shape)
        ).mean().detach(),
    }
    return total, diagnostics



def _v503_masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    value = _as_b1hw(value)
    mask = _as_b1hw(mask).to(value.dtype).clamp(0.0, 1.0)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def _compute_v503_m2_loss(
    cfg: Any,
    gt: torch.Tensor,
    c0: torch.Tensor,
    nonbase: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    epoch: int | None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Fixed factual-action supervision for the V503 atomic M2 policy."""
    del epoch  # The target is factual and does not anneal with candidate quality.
    action_logits = aux["m2_action_logits"]
    action_probs = aux["m2_action_probs"]
    action_index = aux["m2_action_index"]
    m2_probs = _as_b1hw(aux["m2_training_probs"])
    include_global = nonbase.shape[1] > 4 and action_logits.shape[1] > 5
    target = build_factual_cause_targets(
        c0.detach(),
        gt,
        boundary_radius=int(_m1(cfg, "V503_FACTUAL_BOUNDARY_RADIUS", 2)),
        failure_dice=float(_m1(cfg, "V503_FAILURE_DICE", 0.55)),
        include_global_action=include_global,
    )
    action_target = target["action_target"]
    if int(action_target.max()) >= action_logits.shape[1]:
        raise RuntimeError(
            f"V503 action target requires {int(action_target.max()) + 1} actions, "
            f"but M2 exposes {action_logits.shape[1]}"
        )

    log_probs = F.log_softmax(action_logits, dim=1)
    ce_map = F.nll_loss(log_probs, action_target, reduction="none")[:, None]
    factual_error = target["factual_error"]
    base_correct = target["base_correct"]
    error_action_loss = _v503_masked_mean(ce_map, factual_error)
    preserve_action_loss = _v503_masked_mean(ce_map, base_correct)
    action_loss = (
        float(_m1(cfg, "V503_ERROR_ACTION_WEIGHT", 1.0)) * error_action_loss
        + float(_m1(cfg, "V503_PRESERVE_ACTION_WEIGHT", 1.0)) * preserve_action_loss
    )

    repair_loss = _masked_bce_dice(
        m2_probs,
        gt,
        factual_error,
        pos_weight=float(_m1(cfg, "V503_REPAIR_POS_WEIGHT", 2.0)),
    )
    preserve_l1 = _v503_masked_mean((m2_probs - c0.detach()).abs(), base_correct)

    margin = float(_m1(cfg, "V503_BINARY_SAFETY_MARGIN", 0.02))
    base_hard = target["base_hard"]
    positive_guard = F.relu(0.5 + margin - m2_probs) * base_hard
    negative_guard = F.relu(m2_probs - (0.5 - margin)) * (1.0 - base_hard)
    crossing_loss = _v503_masked_mean(positive_guard + negative_guard, base_correct)
    noharm_loss = preserve_l1 + crossing_loss

    boundary_region = target["cause_targets"][:, 2:4].amax(dim=1, keepdim=True)
    boundary_error = (
        _soft_boundary(m2_probs, radius=int(_m1(cfg, "V503_FACTUAL_BOUNDARY_RADIUS", 2)))
        - _soft_boundary(gt, radius=int(_m1(cfg, "V503_FACTUAL_BOUNDARY_RADIUS", 2)))
    ).abs()
    boundary_loss = _v503_masked_mean(boundary_error, boundary_region)

    total = (
        float(_m1(cfg, "V503_ACTION_CE_WEIGHT", 1.0)) * action_loss
        + float(_m1(cfg, "V503_OUTCOME_REPAIR_WEIGHT", 1.0)) * repair_loss
        + float(_m1(cfg, "V503_EXACT_NOHARM_WEIGHT", 2.0)) * noharm_loss
        + float(_m1(cfg, "V503_BOUNDARY_OUTCOME_WEIGHT", 0.5)) * boundary_loss
    )

    with torch.no_grad():
        pred_action = action_index[:, 0] if action_index.ndim == 4 else action_index
        if pred_action.ndim == 3:
            pass
        else:
            pred_action = action_probs.argmax(dim=1)
        error_mask = factual_error[:, 0] > 0.5
        correct_mask = base_correct[:, 0] > 0.5
        action_accuracy = (
            (pred_action[error_mask] == action_target[error_mask]).float().mean()
            if bool(error_mask.any().item()) else m2_probs.new_zeros(())
        )
        preserve_accuracy = (
            (pred_action[correct_mask] == 0).float().mean()
            if bool(correct_mask.any().item()) else m2_probs.new_zeros(())
        )
        c0_hard = target["base_hard"] >= 0.5
        m2_hard = m2_probs >= 0.5
        gt_hard = target["gt_hard"] >= 0.5
        changed = m2_hard != c0_hard
        harmful = changed & (c0_hard == gt_hard)
        beneficial = changed & (c0_hard != gt_hard) & (m2_hard == gt_hard)
        changed_mass = changed.float().sum().clamp_min(1.0)
        conditional_harm = harmful.float().sum() / changed_mass
        conditional_benefit = beneficial.float().sum() / changed_mass

    diag = {
        "v503_m2_loss": total.detach(),
        "v503_action_loss": action_loss.detach(),
        "v503_error_action_loss": error_action_loss.detach(),
        "v503_preserve_action_loss": preserve_action_loss.detach(),
        "v503_outcome_repair_loss": repair_loss.detach(),
        "v503_preserve_l1_loss": preserve_l1.detach(),
        "v503_binary_crossing_loss": crossing_loss.detach(),
        "v503_exact_noharm_loss": noharm_loss.detach(),
        "v503_boundary_outcome_loss": boundary_loss.detach(),
        "v503_teacher_edit_rate": factual_error.mean().detach(),
        "v503_predicted_action_edit_rate": (pred_action > 0).float().mean().detach(),
        "v503_error_action_accuracy": action_accuracy.detach(),
        "v503_preserve_action_accuracy": preserve_accuracy.detach(),
        "v503_changed_pixel_rate": changed.float().mean().detach(),
        "v503_conditional_harm_rate": conditional_harm.detach(),
        "v503_conditional_benefit_rate": conditional_benefit.detach(),
        "v503_failure_case_rate": target["failure_target"].mean().detach(),
    }
    return total, diag



def _compute_v504_m2_loss(
    cfg: Any,
    gt: torch.Tensor,
    c0: torch.Tensor,
    nonbase: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    epoch: int | None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Candidate-realizable potential-outcome supervision for V504.

    The factual region is fixed by detached Base/GT.  Candidate outputs are
    detached potential outcomes and may decide only whether a typed intervention
    is executable inside that fixed region.  Preserve is the target everywhere
    no candidate actually repairs the Base hard decision.
    """
    del epoch
    action_logits = aux["m2_action_logits"]
    action_probs = aux["m2_action_probs"]
    action_index = aux["m2_action_index"]
    route_logits = aux["m2_route_logits"]
    presence_logit = _as_b1hw(aux["m2_candidate_presence_logit"])
    utility_logits = aux["m2_candidate_utility_map_logits"]
    m2_probs = _as_b1hw(aux["m2_training_probs"])

    teacher = build_realizable_action_targets(
        c0.detach(),
        nonbase.detach(),
        gt,
        boundary_radius=int(_m1(cfg, "V503_FACTUAL_BOUNDARY_RADIUS", 2)),
        failure_dice=float(_m1(cfg, "V503_FAILURE_DICE", 0.55)),
    )
    action_target = teacher["action_target"]
    has_action = teacher["has_action_target"]
    if int(action_target.max()) >= action_logits.shape[1]:
        raise RuntimeError(
            f"V504 action target requires {int(action_target.max()) + 1} actions, "
            f"but M2 exposes {action_logits.shape[1]}"
        )

    # 1) Calibrated Preserve-vs-edit decision.  Unlike V503, this is averaged
    # over the true pixel prevalence; factual errors are not artificially given
    # the same total mass as all Base-correct pixels.
    presence_loss = F.binary_cross_entropy_with_logits(
        presence_logit,
        has_action,
        reduction="mean",
    )

    # 2) Conditional route classification is learned only where at least one
    # candidate actually realizes a repair.  This prevents non-executable errors
    # from being mislabeled as mandatory Delete/Fill/Trim/Expand actions.
    route_target = (action_target - 1).clamp_min(0)
    route_ce = F.cross_entropy(route_logits, route_target, reduction="none")[:, None]
    route_loss = _v503_masked_mean(route_ce, has_action)

    # 3) Candidate-conditioned potential outcome.  Each utility head predicts
    # whether its own candidate realizes a typed binary repair in the fixed
    # factual region.  Natural prevalence is retained for calibration.
    executable_target = teacher["candidate_executable_targets"]
    if utility_logits.shape != executable_target.shape:
        raise RuntimeError(
            "V504 candidate utility shape mismatch: "
            f"pred={tuple(utility_logits.shape)} target={tuple(executable_target.shape)}"
        )
    utility_loss = F.binary_cross_entropy_with_logits(
        utility_logits,
        executable_target,
        reduction="mean",
    )

    # 4) Outcome supervision is restricted to realizable edits.  V503 used all
    # factual errors and therefore forced M2 to edit even where M1 had no action
    # capable of fixing the hard decision.
    repair_loss = _masked_bce_dice(
        m2_probs,
        gt,
        has_action,
        pos_weight=1.0,
    )

    base_correct = teacher["base_correct"]
    preserve_l1 = _v503_masked_mean((m2_probs - c0.detach()).abs(), base_correct)
    margin = float(_m1(cfg, "V503_BINARY_SAFETY_MARGIN", 0.02))
    base_hard = teacher["base_hard"]
    positive_guard = F.relu(0.5 + margin - m2_probs) * base_hard
    negative_guard = F.relu(m2_probs - (0.5 - margin)) * (1.0 - base_hard)
    crossing_loss = _v503_masked_mean(positive_guard + negative_guard, base_correct)
    noharm_loss = preserve_l1 + crossing_loss

    # Optional boundary outcome is evaluated only where the selected executable
    # action is a directional boundary action (Trim/Expand).
    boundary_action = ((action_target == 3) | (action_target == 4)).float()[:, None]
    boundary_error = (
        _soft_boundary(m2_probs, radius=int(_m1(cfg, "V503_FACTUAL_BOUNDARY_RADIUS", 2)))
        - _soft_boundary(gt, radius=int(_m1(cfg, "V503_FACTUAL_BOUNDARY_RADIUS", 2)))
    ).abs()
    boundary_loss = _v503_masked_mean(boundary_error, boundary_action)

    total = (
        float(_m1(cfg, "V504_PRESENCE_WEIGHT", 1.0)) * presence_loss
        + float(_m1(cfg, "V504_ROUTE_WEIGHT", 1.0)) * route_loss
        + float(_m1(cfg, "V504_UTILITY_WEIGHT", 1.0)) * utility_loss
        + float(_m1(cfg, "V504_REPAIR_WEIGHT", 1.0)) * repair_loss
        + float(_m1(cfg, "V504_EXACT_NOHARM_WEIGHT", 2.0)) * noharm_loss
        + float(_m1(cfg, "V504_BOUNDARY_WEIGHT", 0.5)) * boundary_loss
    )

    with torch.no_grad():
        pred_action = action_index[:, 0] if action_index.ndim == 4 else action_index
        if pred_action.ndim != 3:
            pred_action = action_probs.argmax(dim=1)
        edit_target_mask = has_action[:, 0] > 0.5
        preserve_target_mask = ~edit_target_mask
        action_accuracy = (
            (pred_action[edit_target_mask] == action_target[edit_target_mask]).float().mean()
            if bool(edit_target_mask.any().item()) else m2_probs.new_zeros(())
        )
        preserve_accuracy = (
            (pred_action[preserve_target_mask] == 0).float().mean()
            if bool(preserve_target_mask.any().item()) else m2_probs.new_zeros(())
        )
        c0_hard = teacher["base_hard"] >= 0.5
        m2_hard = m2_probs >= 0.5
        gt_hard = teacher["gt_hard"] >= 0.5
        changed = m2_hard != c0_hard
        harmful = changed & (c0_hard == gt_hard)
        beneficial = changed & (c0_hard != gt_hard) & (m2_hard == gt_hard)
        changed_mass = changed.float().sum().clamp_min(1.0)
        conditional_harm = harmful.float().sum() / changed_mass
        conditional_benefit = beneficial.float().sum() / changed_mass
        pred_edit = pred_action > 0
        edit_precision = (
            (pred_edit & edit_target_mask).float().sum() / pred_edit.float().sum().clamp_min(1.0)
        )
        edit_recall = (
            (pred_edit & edit_target_mask).float().sum() / edit_target_mask.float().sum().clamp_min(1.0)
        )

    diag = {
        "v504_m2_loss": total.detach(),
        "v504_presence_loss": presence_loss.detach(),
        "v504_route_loss": route_loss.detach(),
        "v504_candidate_utility_loss": utility_loss.detach(),
        "v504_outcome_repair_loss": repair_loss.detach(),
        "v504_exact_noharm_loss": noharm_loss.detach(),
        "v504_boundary_outcome_loss": boundary_loss.detach(),
        "v504_factual_error_rate": teacher["factual_error"].mean().detach(),
        "v504_teacher_edit_rate": has_action.mean().detach(),
        "v504_realizable_error_fraction": teacher["realizable_error_fraction"].detach(),
        "v504_candidate_harm_fraction": teacher["candidate_harm_fraction"].detach(),
        "v504_predicted_action_edit_rate": (pred_action > 0).float().mean().detach(),
        "v504_edit_precision": edit_precision.detach(),
        "v504_edit_recall": edit_recall.detach(),
        "v504_action_accuracy": action_accuracy.detach(),
        "v504_preserve_accuracy": preserve_accuracy.detach(),
        "v504_changed_pixel_rate": changed.float().mean().detach(),
        "v504_conditional_harm_rate": conditional_harm.detach(),
        "v504_conditional_benefit_rate": conditional_benefit.detach(),
    }
    return total, diag


def _compute_v504_m3_loss(
    cfg: Any,
    gt: torch.Tensor,
    c0: torch.Tensor,
    aux: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Baseline-bootstrapped verifier for the realized V504 intervention."""
    m2 = _as_b1hw(aux["m2_fused_probs"]).detach()
    final = _as_b1hw(aux["final_probs"])
    accept_prob = _as_b1hw(aux.get("m3_accept_probability", aux["m3_gate_prob"]))
    c0_hard = c0.detach() >= 0.5
    m2_hard = m2 >= 0.5
    gt_hard = gt >= 0.5
    actual_edit = (m2_hard != c0_hard).float()
    beneficial = ((c0_hard != gt_hard) & (m2_hard == gt_hard)).float()
    harmful = ((c0_hard == gt_hard) & (m2_hard != gt_hard)).float()
    accept_target = beneficial * actual_edit

    bce = F.binary_cross_entropy(
        accept_prob.clamp(EPS, 1.0 - EPS), accept_target, reduction="none"
    )
    # Every realized edit contributes at its natural prevalence. Harmful edits
    # receive an explicit rejection cost but are not rebalanced to 50/50.
    verifier_loss = _v503_masked_mean(bce, actual_edit)
    harmful_accept_loss = _v503_masked_mean(accept_prob, harmful)

    base_error = (c0_hard != gt_hard).float()
    base_correct = 1.0 - base_error
    final_repair = _masked_bce_dice(final, gt, base_error, pos_weight=1.0)
    final_preserve = _v503_masked_mean((final - c0.detach()).abs(), base_correct)
    total = (
        float(_m1(cfg, "V504_M3_VERIFIER_WEIGHT", 1.0)) * verifier_loss
        + float(_m1(cfg, "V504_M3_HARM_REJECT_WEIGHT", 1.0)) * harmful_accept_loss
        + float(_m1(cfg, "V504_M3_REPAIR_WEIGHT", 0.5)) * final_repair
        + float(_m1(cfg, "V504_M3_NOHARM_WEIGHT", 2.0)) * final_preserve
    )
    with torch.no_grad():
        relevant = actual_edit > 0.5
        pred_accept = accept_prob >= 0.5
        accuracy = (
            (pred_accept[relevant] == (accept_target[relevant] > 0.5)).float().mean()
            if bool(relevant.any().item()) else final.new_zeros(())
        )
        harmful_reject = (
            (~pred_accept[harmful > 0.5]).float().mean()
            if bool((harmful > 0.5).any().item()) else final.new_zeros(())
        )
    return total, {
        "v504_m3_loss": total.detach(),
        "v504_m3_verifier_loss": verifier_loss.detach(),
        "v504_m3_harmful_accept_loss": harmful_accept_loss.detach(),
        "v504_m3_final_repair_loss": final_repair.detach(),
        "v504_m3_final_preserve_loss": final_preserve.detach(),
        "v504_m3_accept_target_rate": accept_target.mean().detach(),
        "v504_m3_accept_probability": accept_prob.mean().detach(),
        "v504_m3_verifier_accuracy": accuracy.detach(),
        "v504_m3_harmful_reject_rate": harmful_reject.detach(),
    }

def _compute_v503_m3_loss(
    cfg: Any,
    gt: torch.Tensor,
    c0: torch.Tensor,
    aux: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Train M3 only as a verifier of the realized detached M2 intervention."""
    m2 = _as_b1hw(aux["m2_fused_probs"]).detach()
    final = _as_b1hw(aux["final_probs"])
    accept_prob = _as_b1hw(aux.get("m3_accept_probability", aux["m3_gate_prob"]))
    c0_hard = c0.detach() >= 0.5
    m2_hard = m2 >= 0.5
    gt_hard = gt >= 0.5
    actual_edit = (m2_hard != c0_hard).float()
    beneficial = ((c0_hard != gt_hard) & (m2_hard == gt_hard)).float()
    accept_target = beneficial * actual_edit
    bce = F.binary_cross_entropy(
        accept_prob.clamp(EPS, 1.0 - EPS), accept_target, reduction="none"
    )
    verifier_loss = _v503_masked_mean(bce, actual_edit)

    base_correct = (c0_hard == gt_hard).float()
    final_repair = _masked_bce_dice(final, gt, (c0_hard != gt_hard).float(), pos_weight=1.0)
    final_preserve = _v503_masked_mean((final - c0.detach()).abs(), base_correct)
    total = (
        float(_m1(cfg, "V503_M3_VERIFIER_WEIGHT", 1.0)) * verifier_loss
        + float(_m1(cfg, "V503_M3_REPAIR_WEIGHT", 0.5)) * final_repair
        + float(_m1(cfg, "V503_M3_NOHARM_WEIGHT", 2.0)) * final_preserve
    )
    with torch.no_grad():
        pred_accept = accept_prob >= 0.5
        relevant = actual_edit > 0.5
        accuracy = (
            (pred_accept[relevant] == (accept_target[relevant] > 0.5)).float().mean()
            if bool(relevant.any().item()) else final.new_zeros(())
        )
    return total, {
        "v503_m3_loss": total.detach(),
        "v503_m3_verifier_loss": verifier_loss.detach(),
        "v503_m3_final_repair_loss": final_repair.detach(),
        "v503_m3_final_preserve_loss": final_preserve.detach(),
        "v503_m3_accept_target_rate": accept_target.mean().detach(),
        "v503_m3_accept_probability": accept_prob.mean().detach(),
        "v503_m3_verifier_accuracy": accuracy.detach(),
    }



def _v505_binary_focal_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Parameter-free class-mass-balanced binary focal loss.

    V505 averaged every pixel together.  Because factual cause maps are very
    sparse, predicting zero everywhere produced the tiny logged cause/support
    losses while eliminating every candidate.  This implementation computes a
    positive and negative mean independently for each channel and then averages
    the channels.  It changes no labels and needs no dataset-specific positive
    weight.
    """
    target = target.to(dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = torch.exp(-bce)
    per_item = ((1.0 - pt) ** float(gamma)) * bce

    if per_item.ndim == 1:
        per_item = per_item[:, None, None]
        target = target[:, None, None]
    elif per_item.ndim == 2:
        per_item = per_item[:, :, None]
        target = target[:, :, None]
    else:
        per_item = per_item.reshape(per_item.shape[0], per_item.shape[1], -1)
        target = target.reshape(target.shape[0], target.shape[1], -1)

    channel_terms = []
    for channel in range(per_item.shape[1]):
        loss_c = per_item[:, channel]
        target_c = target[:, channel]
        positive = target_c > 0.5
        negative = ~positive
        class_terms = []
        if bool(positive.any().item()):
            class_terms.append(loss_c[positive].mean())
        if bool(negative.any().item()):
            class_terms.append(loss_c[negative].mean())
        if class_terms:
            channel_terms.append(torch.stack(class_terms).mean())
    return (
        torch.stack(channel_terms).mean()
        if channel_terms
        else logits.sum() * 0.0
    )



def _v507_dynamic_sparse_focal_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    max_positive_weight: float = 8.0,
) -> torch.Tensor:
    """Sparse focal loss with a bounded, data-derived positive weight.

    V506 independently averaged positive and negative masses.  On a 1%-sparse
    cause map that makes a handful of positive pixels contribute as much as the
    complete background and encourages broad false-positive cause/support maps.
    V507 derives the positive weight from the actual batch prevalence but caps
    it, so sparse positives remain learnable without receiving unbounded mass.
    """
    target = target.to(dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = torch.exp(-bce)
    focal = ((1.0 - pt) ** float(gamma)) * bce

    if focal.ndim == 1:
        focal = focal[:, None, None]
        target = target[:, None, None]
    elif focal.ndim == 2:
        focal = focal[:, :, None]
        target = target[:, :, None]
    else:
        focal = focal.reshape(focal.shape[0], focal.shape[1], -1)
        target = target.reshape(target.shape[0], target.shape[1], -1)

    terms = []
    cap = max(float(max_positive_weight), 1.0)
    for channel in range(focal.shape[1]):
        loss_c = focal[:, channel]
        target_c = target[:, channel]
        positive = target_c > 0.5
        negative = ~positive
        num_pos = positive.sum().to(dtype=loss_c.dtype)
        num_neg = negative.sum().to(dtype=loss_c.dtype)
        if bool(positive.any().item()):
            positive_weight = (num_neg / num_pos.clamp_min(1.0)).clamp(1.0, cap)
        else:
            positive_weight = loss_c.new_tensor(1.0)
        weights = torch.where(positive, positive_weight, torch.ones_like(loss_c))
        terms.append((loss_c * weights).sum() / weights.sum().clamp_min(1.0))
    return torch.stack(terms).mean() if terms else logits.sum() * 0.0


def _v507_precision_tversky_loss(
    probability: torch.Tensor,
    target: torch.Tensor,
    fp_weight: float = 0.70,
    fn_weight: float = 0.30,
) -> torch.Tensor:
    """Precision-first Tversky term used to suppress broad false cause maps."""
    probability = probability.clamp(EPS, 1.0 - EPS)
    target = target.to(dtype=probability.dtype).clamp(0.0, 1.0)
    probability = probability.reshape(probability.shape[0], probability.shape[1], -1)
    target = target.reshape(target.shape[0], target.shape[1], -1)
    tp = (probability * target).sum(dim=-1)
    fp = (probability * (1.0 - target)).sum(dim=-1)
    fn = ((1.0 - probability) * target).sum(dim=-1)
    score = (tp + EPS) / (
        tp + float(fp_weight) * fp + float(fn_weight) * fn + EPS
    )
    active = target.sum(dim=-1) > 0.0
    if bool(active.any().item()):
        return (1.0 - score[active]).mean()
    return probability.mean()


def _v507_prevalence_calibration_loss(
    probability: torch.Tensor,
    target: torch.Tensor,
    floor: float = 0.01,
) -> torch.Tensor:
    """Match predicted cause/support prevalence to factual prevalence.

    The denominator is detached and floored, making this scale-adaptive while
    avoiding a dataset-specific positive-class constant.
    """
    probability = probability.clamp(0.0, 1.0)
    target = target.to(dtype=probability.dtype).clamp(0.0, 1.0)
    pred_rate = probability.flatten(2).mean(dim=-1)
    target_rate = target.flatten(2).mean(dim=-1)
    scale = target_rate.detach().clamp_min(float(floor))
    return F.smooth_l1_loss(
        pred_rate / scale,
        target_rate / scale,
        reduction="mean",
    )


def _v507_straight_through_hard(probability: torch.Tensor) -> torch.Tensor:
    hard = (probability >= 0.5).to(dtype=probability.dtype)
    return hard + probability - probability.detach()


def _v507_candidate_hard_objectives(
    c0: torch.Tensor,
    local: torch.Tensor,
    gt: torch.Tensor,
    cause_target: torch.Tensor,
    gain_margin: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Optimize the exact thresholded candidate that is deployed.

    Soft absolute-error improvement is insufficient: a candidate may move from
    0.95 to 0.55 and reduce soft error while leaving the binary mask unchanged.
    This straight-through objective directly trains candidate-level hard Dice,
    penalizes any global Dice regression, and penalizes threshold crossings on
    Base-correct/out-of-cause pixels.
    """
    base_hard = (c0 >= 0.5).to(dtype=local.dtype).expand_as(local)
    gt_expand = gt.expand_as(local)
    local_hard = _v507_straight_through_hard(local)

    def dice_per_candidate(mask: torch.Tensor) -> torch.Tensor:
        intersection = (mask * gt_expand).flatten(2).sum(dim=2)
        denominator = mask.flatten(2).sum(dim=2) + gt_expand.flatten(2).sum(dim=2)
        return (2.0 * intersection + EPS) / (denominator + EPS)

    candidate_dice = dice_per_candidate(local_hard)
    base_dice = dice_per_candidate(base_hard).detach()
    delta = candidate_dice - base_dice
    active = cause_target.flatten(2).sum(dim=2) > 0.5
    active_float = active.to(dtype=local.dtype)
    active_denominator = active_float.sum().clamp_min(1.0)

    hard_gain = (
        F.relu(float(gain_margin) - delta) * active_float
    ).sum() / active_denominator
    hard_noharm = (
        F.relu(-delta) * active_float
    ).sum() / active_denominator

    changed = (local_hard - base_hard).abs()
    base_correct = (base_hard == gt_expand).to(dtype=local.dtype)
    changed_mass = changed.flatten(2).sum(dim=2).clamp_min(1.0)
    harmful_fraction = (changed * base_correct).flatten(2).sum(dim=2) / changed_mass
    outside_fraction = (
        changed * (1.0 - cause_target)
    ).flatten(2).sum(dim=2) / changed_mass
    hard_scope = (
        (harmful_fraction + outside_fraction) * active_float
    ).sum() / active_denominator

    with torch.no_grad():
        hard_benefit_rate = (
            ((delta > 0.0) & active).float().sum() / active_float.sum().clamp_min(1.0)
        )
        hard_harm_rate = (
            ((delta < 0.0) & active).float().sum() / active_float.sum().clamp_min(1.0)
        )
        mean_delta = (delta * active_float).sum() / active_denominator

    return hard_gain + hard_noharm + hard_scope, {
        "hard_gain_loss": hard_gain,
        "hard_noharm_loss": hard_noharm,
        "hard_scope_loss": hard_scope,
        "hard_benefit_rate": hard_benefit_rate,
        "hard_harm_rate": hard_harm_rate,
        "hard_mean_delta_dice": mean_delta,
    }

def _v505_multiclass_focal(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Balanced Neutral/Benefit/Harm focal objective over valid proposals."""
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_target = target.reshape(-1).long()
    log_probability = F.log_softmax(flat_logits, dim=-1)
    selected_log_probability = log_probability.gather(
        1, flat_target[:, None]
    )[:, 0]
    selected_probability = selected_log_probability.exp()
    per_item = -(
        (1.0 - selected_probability) ** float(gamma)
    ) * selected_log_probability
    if valid_mask is None:
        valid = torch.ones_like(flat_target, dtype=torch.bool)
    else:
        valid = valid_mask.reshape(-1).bool()

    class_terms = []
    for class_id in (OUTCOME_NEUTRAL, OUTCOME_BENEFIT, OUTCOME_HARM):
        mask = valid & (flat_target == class_id)
        if bool(mask.any().item()):
            class_terms.append(per_item[mask].mean())
    return (
        torch.stack(class_terms).mean()
        if class_terms
        else logits.sum() * 0.0
    )


def _v505_masked_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    per_item = F.smooth_l1_loss(prediction, target, reduction="none")
    mask = valid_mask.to(dtype=per_item.dtype)
    return (per_item * mask).sum() / mask.sum().clamp_min(1.0)


def _v505_listwise_preserve_decision_loss(
    score: torch.Tensor,
    outcome_class: torch.Tensor,
    delta_dice: torch.Tensor,
    delta_surface: torch.Tensor,
    surface_weight: float,
    proposal_valid: torch.Tensor | None = None,
    selection_margin: float = 0.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Choose one beneficial proposal or Preserve with the deployed margin."""
    if proposal_valid is None:
        proposal_valid = torch.ones_like(score, dtype=torch.bool)
    else:
        proposal_valid = proposal_valid.bool()
    utility = delta_dice + float(surface_weight) * delta_surface
    valid_benefit = (outcome_class == OUTCOME_BENEFIT) & proposal_valid
    selection_floor = -1.0e4 if score.dtype in (torch.float16, torch.bfloat16) else -1.0e9
    masked_utility = utility.masked_fill(~valid_benefit, selection_floor)
    best_utility, best_index = masked_utility.max(dim=-1)
    has_benefit = valid_benefit.any(dim=-1) & (best_utility > 0.0)
    target = torch.where(has_benefit, best_index + 1, torch.zeros_like(best_index))
    preserve_score = torch.zeros_like(score[..., :1])
    temp = max(float(temperature), 1.0e-3)
    selectable_score = ((score - float(selection_margin)) / temp).masked_fill(
        ~proposal_valid, selection_floor
    )
    decision_logits = torch.cat([preserve_score, selectable_score], dim=-1)
    return F.cross_entropy(
        decision_logits.reshape(-1, decision_logits.shape[-1]),
        target.reshape(-1),
        reduction="mean",
    )


def _v507_soft_preserve_decision_loss(
    score: torch.Tensor,
    outcome_class: torch.Tensor,
    delta_dice: torch.Tensor,
    delta_surface: torch.Tensor,
    surface_weight: float,
    proposal_valid: torch.Tensor,
    selection_margin: float,
    teacher_temperature: float,
    student_temperature: float,
) -> torch.Tensor:
    """Distributional listwise supervision over Preserve and all good proposals.

    One-hot best-proposal labels oscillate when several proposals have almost
    equal utility.  V507 gives every valid beneficial proposal probability mass,
    while Preserve remains the only target when no proposal has positive utility.
    """
    valid = proposal_valid.bool()
    utility = delta_dice + float(surface_weight) * delta_surface
    beneficial = valid & (outcome_class == OUTCOME_BENEFIT) & (utility > 0.0)
    floor = -1.0e4 if score.dtype in (torch.float16, torch.bfloat16) else -1.0e9

    teacher_temp = max(float(teacher_temperature), 1.0e-3)
    preserve_teacher = torch.zeros_like(score[..., :1])
    proposal_teacher = (utility / teacher_temp).masked_fill(~beneficial, floor)
    teacher_logits = torch.cat([preserve_teacher, proposal_teacher], dim=-1)
    teacher_probability = torch.softmax(teacher_logits, dim=-1).detach()

    student_temp = max(float(student_temperature), 1.0e-3)
    preserve_student = torch.zeros_like(score[..., :1])
    proposal_student = ((score - float(selection_margin)) / student_temp).masked_fill(
        ~valid, floor
    )
    student_log_probability = torch.log_softmax(
        torch.cat([preserve_student, proposal_student], dim=-1), dim=-1
    )
    return -(teacher_probability * student_log_probability).sum(dim=-1).mean()

def _v505_region_ranking_loss(
    score: torch.Tensor,
    outcome_class: torch.Tensor,
    proposal_valid: torch.Tensor | None = None,
    selection_margin: float = 0.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Calibrate scores around the exact Preserve/deploy margin."""
    if proposal_valid is None:
        valid = torch.ones_like(score, dtype=torch.bool)
    else:
        valid = proposal_valid.bool()
    temp = max(float(temperature), 1.0e-3)
    margin = float(selection_margin)
    class_terms = []
    for class_id in (OUTCOME_NEUTRAL, OUTCOME_BENEFIT, OUTCOME_HARM):
        mask = valid & (outcome_class == class_id)
        if not bool(mask.any().item()):
            continue
        selected = score[mask]
        if class_id == OUTCOME_BENEFIT:
            term = F.softplus((margin - selected) / temp).mean()
        elif class_id == OUTCOME_HARM:
            term = F.softplus((margin + selected) / temp).mean()
        else:
            term = (0.10 * selected.square()).mean()
        class_terms.append(term)
    return torch.stack(class_terms).mean() if class_terms else score.sum() * 0.0

def _v506_all_proposal_verifier_loss(
    score: torch.Tensor,
    outcome_class: torch.Tensor,
    delta_dice: torch.Tensor,
    delta_surface: torch.Tensor,
    surface_weight: float,
    proposal_valid: torch.Tensor,
    utility_margin: float,
    selection_margin: float = 0.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Train M3 on every deployable proposal at its actual accept boundary."""
    utility = delta_dice + float(surface_weight) * delta_surface
    valid = proposal_valid.bool()
    positive = (
        valid
        & (outcome_class == OUTCOME_BENEFIT)
        & (utility > float(utility_margin))
    )
    negative = valid & (~positive)
    temp = max(float(temperature), 1.0e-3)
    margin = float(selection_margin)
    terms = []
    if bool(positive.any().item()):
        terms.append(F.softplus((margin - score[positive]) / temp).mean())
    if bool(negative.any().item()):
        terms.append(F.softplus((margin + score[negative]) / temp).mean())
    return torch.stack(terms).mean() if terms else score.sum() * 0.0

def _v505_selected_verifier_decision_loss(
    score: torch.Tensor,
    outcome_class: torch.Tensor,
    delta_dice: torch.Tensor,
    delta_surface: torch.Tensor,
    surface_weight: float,
    selected: torch.Tensor,
    proposal_valid: torch.Tensor,
    selection_margin: float = 0.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Train M3 on the proposal M2 asks it to verify, at deploy margin."""
    safe = selected.clamp_min(0)
    selected_score = score.gather(2, safe[:, :, None])[:, :, 0]
    selected_class = outcome_class.gather(2, safe[:, :, None])[:, :, 0]
    selected_utility = (
        delta_dice + float(surface_weight) * delta_surface
    ).gather(2, safe[:, :, None])[:, :, 0]
    selected_valid = proposal_valid.gather(2, safe[:, :, None])[:, :, 0].bool()
    active = (selected >= 0) & selected_valid
    accept_target = (
        active
        & (selected_class == OUTCOME_BENEFIT)
        & (selected_utility > 0.0)
    ).long()
    temp = max(float(temperature), 1.0e-3)
    accept_logit = (selected_score - float(selection_margin)) / temp
    logits = torch.stack([torch.zeros_like(accept_logit), accept_logit], dim=-1)
    per_step = F.cross_entropy(
        logits.reshape(-1, 2),
        accept_target.reshape(-1),
        reduction="none",
    ).reshape_as(selected_score)
    active_float = active.to(dtype=per_step.dtype)
    return (per_step * active_float).sum() / active_float.sum().clamp_min(1.0)

def _v505_gather_selected(
    values: torch.Tensor,
    selected: torch.Tensor,
    preserve_value: float = 0.0,
) -> torch.Tensor:
    safe = selected.clamp_min(0)
    gathered = values.gather(2, safe[:, :, None])[:, :, 0]
    return torch.where(
        selected >= 0,
        gathered,
        values.new_full(gathered.shape, float(preserve_value)),
    )



def _v518_balanced_ohem_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    negative_ratio: int = 4,
    min_negatives: int = 256,
) -> torch.Tensor:
    """Keep every error pixel and only the hardest bounded background set."""
    target = target.to(dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    terms = []
    for channel in range(logits.shape[1]):
        loss_c = loss[:, channel].reshape(-1)
        target_c = target[:, channel].reshape(-1)
        positive = target_c > 0.5
        negative = ~positive
        pieces = []
        if bool(positive.any().item()):
            pieces.append(loss_c[positive].mean())
        negative_loss = loss_c[negative]
        if negative_loss.numel() > 0:
            positive_count = int(positive.sum().item())
            keep = max(int(min_negatives), int(negative_ratio) * max(positive_count, 1))
            keep = min(keep, int(negative_loss.numel()))
            pieces.append(torch.topk(negative_loss, k=keep).values.mean())
        if pieces:
            terms.append(torch.stack(pieces).mean())
    return torch.stack(terms).mean() if terms else logits.sum() * 0.0


def _v518_masked_subtype_ce(
    logits: torch.Tensor,
    target_class: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    per_pixel = F.cross_entropy(logits, target_class.long(), reduction="none")
    valid = valid_mask[:, 0] > 0.5
    if bool(valid.any().item()):
        return per_pixel[valid].mean()
    return logits.sum() * 0.0


def _v518_hierarchical_locator_loss(
    cfg: Any,
    cause_target: torch.Tensor,
    aux: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    polarity_logits = aux.get("v518_error_polarity_logits")
    fp_subtype_logits = aux.get("v518_fp_subtype_logits")
    fn_subtype_logits = aux.get("v518_fn_subtype_logits")
    if not all(isinstance(x, torch.Tensor) for x in (
        polarity_logits, fp_subtype_logits, fn_subtype_logits
    )):
        zero = cause_target.sum() * 0.0
        return zero, {
            "polarity_loss": zero,
            "fp_subtype_loss": zero,
            "fn_subtype_loss": zero,
        }

    delete = cause_target[:, 0:1]
    fill = cause_target[:, 1:2]
    trim = cause_target[:, 2:3]
    expand = cause_target[:, 3:4]
    fp_target = torch.maximum(delete, trim)
    fn_target = torch.maximum(fill, expand)
    polarity_target = torch.cat([fp_target, fn_target], dim=1)
    polarity_loss = _v518_balanced_ohem_bce_with_logits(
        polarity_logits,
        polarity_target,
        negative_ratio=int(_m1(cfg, "V518_OHEM_NEGATIVE_RATIO", 4)),
        min_negatives=int(_m1(cfg, "V518_OHEM_MIN_NEGATIVES", 256)),
    )
    fp_class = (trim[:, 0] > delete[:, 0]).long()
    fn_class = (expand[:, 0] > fill[:, 0]).long()
    fp_subtype_loss = _v518_masked_subtype_ce(
        fp_subtype_logits, fp_class, fp_target
    )
    fn_subtype_loss = _v518_masked_subtype_ce(
        fn_subtype_logits, fn_class, fn_target
    )
    subtype_loss = 0.5 * (fp_subtype_loss + fn_subtype_loss)
    total = (
        float(_m1(cfg, "V518_POLARITY_WEIGHT", 1.0)) * polarity_loss
        + float(_m1(cfg, "V518_SUBTYPE_WEIGHT", 0.5)) * subtype_loss
    )
    return total, {
        "polarity_loss": polarity_loss,
        "fp_subtype_loss": fp_subtype_loss,
        "fn_subtype_loss": fn_subtype_loss,
    }


def _v518_trainable_candidates(
    candidate_probs: torch.Tensor,
    aux: Dict[str, torch.Tensor],
) -> torch.Tensor:
    mask = aux.get("v518_trainable_candidate_mask")
    if isinstance(mask, torch.Tensor) and mask.ndim == 1 and mask.numel() == candidate_probs.shape[1]:
        indices = torch.nonzero(mask, as_tuple=False).flatten()
        indices = indices[indices > 0]
        if indices.numel() > 0:
            return candidate_probs.index_select(1, indices)
    return candidate_probs[:, 1:]


def _v518_bank_objectives(
    cfg: Any,
    candidate_probs: torch.Tensor,
    c0: torch.Tensor,
    gt: torch.Tensor,
    aux: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    candidates = _v518_trainable_candidates(candidate_probs, aux)
    if candidates.shape[1] == 0:
        zero = candidate_probs.sum() * 0.0
        return zero, {
            "bank_best_loss": zero,
            "coverage_loss": zero,
            "diversity_loss": zero,
            "soft_best_gain": zero.detach(),
            "soft_error_coverage": zero.detach(),
        }

    gt_k = gt.expand(-1, candidates.shape[1], -1, -1)
    base_dice = _soft_dice_probs(c0, gt)[:, 0]
    candidate_dice = _soft_dice_probs(candidates, gt_k)
    gains = candidate_dice - base_dice[:, None]
    temperature = max(float(_m1(cfg, "V518_BANK_TEMPERATURE", 0.02)), 1.0e-4)
    # Normalised log-sum-exp is a smooth best-of-K without a candidate-count bonus.
    soft_best_gain = temperature * (
        torch.logsumexp(gains / temperature, dim=1)
        - math.log(max(candidates.shape[1], 1))
    )
    margin = float(_m1(cfg, "V518_BANK_GAIN_MARGIN", 0.005))
    bank_best_loss = F.relu(margin - soft_best_gain).mean()

    gt_hard = (gt >= 0.5).to(candidates.dtype)
    base_error = ((c0 >= 0.5) != (gt >= 0.5)).to(candidates.dtype)
    correct_probability = (
        candidates * gt_hard.expand_as(candidates)
        + (1.0 - candidates) * (1.0 - gt_hard.expand_as(candidates))
    )
    best_correct_probability = correct_probability.max(dim=1, keepdim=True).values
    soft_error_coverage = (
        best_correct_probability * base_error
    ).sum() / base_error.sum().clamp_min(1.0)
    coverage_loss = 1.0 - soft_error_coverage

    # Diversity is evaluated only among the four typed primary candidates and
    # only on factual Base-error pixels. Dose copies are intentionally similar.
    primary = candidate_probs[:, 1:5]
    edit = (primary - c0).abs() * base_error
    flat = F.normalize(edit.flatten(2), dim=2, eps=1.0e-6)
    similarity = torch.bmm(flat, flat.transpose(1, 2))
    eye = torch.eye(4, device=similarity.device, dtype=similarity.dtype)[None]
    diversity_loss = (similarity * (1.0 - eye)).sum() / (
        similarity.shape[0] * 12.0
    )

    total = (
        float(_m1(cfg, "V518_BANK_BEST_WEIGHT", 2.0)) * bank_best_loss
        + float(_m1(cfg, "V518_COVERAGE_WEIGHT", 1.0)) * coverage_loss
        + float(_m1(cfg, "V518_DIVERSITY_WEIGHT", 0.05)) * diversity_loss
    )
    return total, {
        "bank_best_loss": bank_best_loss,
        "coverage_loss": coverage_loss,
        "diversity_loss": diversity_loss,
        "soft_best_gain": soft_best_gain.mean().detach(),
        "soft_error_coverage": soft_error_coverage.detach(),
    }


def _v519_prob_bce_dice(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    pred = _as_b1hw(pred).clamp(EPS, 1.0 - EPS)
    gt = _as_b1hw(gt).clamp(0.0, 1.0)
    bce = F.binary_cross_entropy(pred, gt)
    inter = (pred * gt).flatten(1).sum(dim=1)
    den = pred.flatten(1).sum(dim=1) + gt.flatten(1).sum(dim=1)
    return bce + (1.0 - (2.0 * inter + EPS) / (den + EPS)).mean()


def _v520_balanced_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Equalise positive and negative supervision without copying samples."""
    target = target.to(dtype=logits.dtype)
    if valid is None:
        valid = torch.ones_like(target, dtype=torch.bool)
    else:
        valid = valid.bool()
    per = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    positive = valid & (target > 0.5)
    negative = valid & ~positive
    terms = []
    if bool(positive.any().item()):
        terms.append(per[positive].mean())
    if bool(negative.any().item()):
        terms.append(per[negative].mean())
    return torch.stack(terms).mean() if terms else logits.sum() * 0.0


def _v520_region_average(value: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return value
    b, n, h, w = value.shape
    kernel = 2 * radius + 1
    return F.avg_pool2d(
        value.reshape(b * n, 1, h, w),
        kernel_size=kernel,
        stride=1,
        padding=radius,
    ).reshape(b, n, h, w)


def _v520_region_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    """Dense local hard-Dice map with the same neighbourhood as deployment."""
    pred = pred.float()
    target = target.float()
    inter = _v520_region_average(pred * target, radius)
    den = _v520_region_average(pred + target, radius)
    return (2.0 * inter + EPS) / (den + EPS)


def _compute_v520_m1_objective(
    cfg: Any,
    gt: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    epoch: int | None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Real V518 M1 objective used during the joint V520 run.

    This function deliberately does not consume any M2 output.  M1 learns from
    its factual cause, support, candidate-repair, coverage and bank objectives;
    M2 receives detached copies of the resulting bank in the forward pass.
    """
    candidate_probs = aux.get("candidate_probs")
    if not isinstance(candidate_probs, torch.Tensor) or candidate_probs.ndim != 4:
        raise RuntimeError("V520 M1 requires aux['candidate_probs'] [B,K,H,W]")
    if candidate_probs.shape[1] < 5:
        raise RuntimeError("V520 M1 requires Preserve + four typed local candidates")

    gt = (_as_b1hw(gt) >= 0.5).float()
    c0 = candidate_probs[:, :1].detach().clamp(EPS, 1.0 - EPS)
    local = candidate_probs[:, 1:5].clamp(EPS, 1.0 - EPS)
    factual = build_factual_cause_targets(
        c0,
        gt,
        boundary_radius=int(_m1(cfg, "V503_FACTUAL_BOUNDARY_RADIUS", 2)),
        failure_dice=float(_m1(cfg, "V503_FAILURE_DICE", 0.55)),
        include_global_action=False,
    )
    cause_target = factual["cause_targets"]

    cause_logits = aux.get("error_cause_map_logits")
    failure_logit = aux.get("failure_state_logit")
    raw_support = aux.get("local_raw_supports", aux.get("local_supports"))
    if not isinstance(cause_logits, torch.Tensor):
        raise RuntimeError("V520 M1 missing error_cause_map_logits")
    if not isinstance(failure_logit, torch.Tensor):
        raise RuntimeError("V520 M1 missing failure_state_logit")
    if not isinstance(raw_support, torch.Tensor):
        raise RuntimeError("V520 M1 missing local_raw_supports/local_supports")
    raw_support = raw_support[:, :4].clamp(EPS, 1.0 - EPS)

    gamma = float(_m1(cfg, "V505_FOCAL_GAMMA", 2.0))
    cause_loss = _v505_binary_focal_with_logits(cause_logits, cause_target, gamma)
    failure_loss = _v505_binary_focal_with_logits(
        failure_logit,
        factual["failure_target"],
        gamma,
    )
    support_loss = _v505_binary_focal_with_logits(
        torch.logit(raw_support),
        cause_target,
        gamma,
    )
    locator_loss, locator_parts = _v518_hierarchical_locator_loss(
        cfg,
        cause_target,
        aux,
    )
    bank_loss, bank_parts = _v518_bank_objectives(
        cfg,
        candidate_probs,
        c0,
        gt,
        aux,
    )

    c0_bank = c0.expand_as(local)
    gt_bank = gt.expand_as(local)
    base_error = (c0_bank - gt_bank).abs()
    candidate_error = (local - gt_bank).abs()
    repair_gain = base_error - candidate_error
    actual_change = (local - c0_bank).abs()
    temperature = max(
        float(_m1(cfg, "V505_NET_REGRET_TEMPERATURE", 0.10)),
        1.0e-3,
    )
    margin = max(float(_m1(cfg, "V506_REPAIR_GAIN_MARGIN", 0.01)), 0.0)
    repair_regret = (
        F.softplus((margin - repair_gain) / temperature) * cause_target
    ).sum() / cause_target.sum().clamp_min(1.0)
    outside_weight = actual_change.detach() * (1.0 - cause_target)
    outside_harm_regret = (
        F.softplus((candidate_error - base_error) / temperature) * outside_weight
    ).sum() / outside_weight.sum().clamp_min(1.0)
    scope_loss = (
        actual_change * (1.0 - cause_target)
    ).sum() / actual_change.sum().clamp_min(1.0)
    safety_loss = (
        float(_m1(cfg, "V505_NET_REGRET_WEIGHT", 1.0))
        * (repair_regret + outside_harm_regret)
        + float(_m1(cfg, "V505_SCOPE_PURITY_WEIGHT", 1.0)) * scope_loss
    )
    localization_loss = (
        float(_m1(cfg, "V505_CAUSE_WEIGHT", 1.0)) * cause_loss
        + float(_m1(cfg, "V505_FAILURE_WEIGHT", 0.5)) * failure_loss
        + float(_m1(cfg, "V505_SUPPORT_WEIGHT", 1.0)) * support_loss
        + locator_loss
    )

    epoch_value = int(epoch or 0)
    locator_only_epochs = max(int(_m1(cfg, "V518_LOCATOR_ONLY_EPOCHS", 8)), 0)
    bank_start = max(
        int(_m1(cfg, "V518_BANK_LOSS_START_EPOCH", locator_only_epochs)),
        locator_only_epochs,
    )
    bank_ramp_epochs = max(int(_m1(cfg, "V518_BANK_RAMP_EPOCHS", 8)), 1)
    safety_ramp_epochs = max(int(_m1(cfg, "V518_SAFETY_RAMP_EPOCHS", 12)), 1)
    if epoch_value < locator_only_epochs:
        bank_scale = 0.0
        safety_scale = 0.0
    else:
        bank_scale = min(
            max((epoch_value - bank_start + 1) / float(bank_ramp_epochs), 0.0),
            1.0,
        )
        safety_scale = min(
            max((epoch_value - locator_only_epochs + 1) / float(safety_ramp_epochs), 0.0),
            1.0,
        )
    m1_loss = (
        localization_loss
        + bank_scale * bank_loss
        + safety_scale * safety_loss
    )
    return m1_loss, {
        "v520_m1_loss": m1_loss.detach(),
        "v520_m1_localization_loss": localization_loss.detach(),
        "v520_m1_safety_loss": safety_loss.detach(),
        "v520_m1_cause_loss": cause_loss.detach(),
        "v520_m1_failure_loss": failure_loss.detach(),
        "v520_m1_support_loss": support_loss.detach(),
        "v520_m1_locator_loss": locator_loss.detach(),
        "v520_m1_polarity_loss": locator_parts["polarity_loss"].detach(),
        "v520_m1_fp_subtype_loss": locator_parts["fp_subtype_loss"].detach(),
        "v520_m1_fn_subtype_loss": locator_parts["fn_subtype_loss"].detach(),
        "v520_m1_bank_loss": bank_loss.detach(),
        "v520_m1_bank_best_loss": bank_parts["bank_best_loss"].detach(),
        "v520_m1_coverage_loss": bank_parts["coverage_loss"].detach(),
        "v520_m1_diversity_loss": bank_parts["diversity_loss"].detach(),
        "v520_m1_soft_best_gain": bank_parts["soft_best_gain"],
        "v520_m1_soft_error_coverage": bank_parts["soft_error_coverage"],
        "v520_m1_bank_scale": m1_loss.new_tensor(bank_scale),
        "v520_m1_safety_scale": m1_loss.new_tensor(safety_scale),
    }


def _compute_v520_gate_selector_loss(
    cfg: Any,
    gt: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    epoch: int | None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Joint V518-M1 + V520 two-stage regional M2 objective."""
    gt = (_as_b1hw(gt) >= 0.5).float()
    candidate_probs = aux.get("candidate_probs")
    if not isinstance(candidate_probs, torch.Tensor) or candidate_probs.ndim != 4:
        raise RuntimeError("V520 requires aux['candidate_probs'] [B,K,H,W]")
    required = (
        "v520_gate_logit",
        "v520_soft_gate",
        "v520_hard_gate",
        "v520_selector_logits",
        "v520_soft_selector_weights",
        "v520_hard_selector_weights",
        "v520_soft_route_weights",
        "v520_hard_route_weights",
        "v520_route_weights",
        "v520_utility_pred",
        "v520_harm_logit",
        "v520_executable_mask",
        "v520_abs_edit",
        "v520_deploy_mask",
        "m2_training_probs",
        "m2_fused_probs",
    )
    missing = [key for key in required if key not in aux]
    if missing:
        raise RuntimeError("V520 loss missing outputs: " + str(missing))

    # M1 remains live through its own objective.  All M2 teacher quantities are
    # built from detached factual observations so the selector cannot move its
    # labels by changing Base or M1.
    m1_loss, m1_diag = _compute_v520_m1_objective(cfg, gt, aux, epoch)
    candidates = candidate_probs.detach().clamp(EPS, 1.0 - EPS)
    c0 = candidates[:, :1]
    nonbase = candidates[:, 1:]
    b, n, h, w = nonbase.shape
    gt_bank = gt.expand(-1, n, -1, -1)
    c0_bank = c0.expand_as(nonbase)

    executable = aux["v520_executable_mask"].bool()
    deploy_mask = aux["v520_deploy_mask"][1:].bool()
    executable = executable & deploy_mask[None, :, None, None]
    abs_edit = aux["v520_abs_edit"].detach()
    changed = (
        (nonbase >= 0.5) != (c0_bank >= 0.5)
    ) & executable

    radius = max(int(_m1(cfg, "V520_REGION_RADIUS", 2)), 0)
    base_hard = (c0_bank >= 0.5).float()
    candidate_hard = (nonbase >= 0.5).float()
    gt_hard = (gt_bank >= 0.5).float()
    base_region_dice = _v520_region_dice(base_hard, gt_hard, radius)
    candidate_region_dice = _v520_region_dice(candidate_hard, gt_hard, radius)
    delta_region_dice = candidate_region_dice - base_region_dice

    # Boundary agreement is a differentiable local proxy for NSD.  Positive
    # delta means the candidate boundary is closer to GT than Base.
    base_boundary = _soft_boundary(c0, radius=max(radius, 1)).detach()
    gt_boundary = _soft_boundary(gt, radius=max(radius, 1)).detach()
    candidate_boundary = _soft_boundary(
        nonbase.reshape(b * n, 1, h, w),
        radius=max(radius, 1),
    ).reshape(b, n, h, w).detach()
    base_boundary_error = (base_boundary - gt_boundary).abs().expand_as(nonbase)
    candidate_boundary_error = (
        candidate_boundary - gt_boundary.expand_as(nonbase)
    ).abs()
    delta_surface = _v520_region_average(
        base_boundary_error - candidate_boundary_error,
        radius,
    )

    base_wrong = (c0_bank >= 0.5) != (gt_bank >= 0.5)
    candidate_correct = (nonbase >= 0.5) == (gt_bank >= 0.5)
    benefit_pixel = changed & base_wrong & candidate_correct
    harm_pixel = changed & (~base_wrong) & (~candidate_correct)
    changed_float = changed.float()
    benefit_fraction = _v520_region_average(benefit_pixel.float(), radius)
    harm_fraction = _v520_region_average(harm_pixel.float(), radius)
    edit_fraction = _v520_region_average(changed_float, radius)

    utility_target = (
        delta_region_dice
        + float(_m1(cfg, "V520_SURFACE_UTILITY_WEIGHT", 0.25)) * delta_surface
        + float(_m1(cfg, "V520_BENEFIT_FRACTION_WEIGHT", 0.25)) * benefit_fraction
        - float(_m1(cfg, "V520_HARM_FRACTION_WEIGHT", 0.50)) * harm_fraction
        - float(_m1(cfg, "V520_TEACHER_EDIT_PENALTY", 0.02)) * edit_fraction
    ).detach()
    utility_target = utility_target.masked_fill(~executable, -1.0)
    positive_margin = float(_m1(cfg, "V520_POSITIVE_UTILITY_MARGIN", 0.002))
    harmful_margin = float(_m1(cfg, "V520_HARMFUL_UTILITY_MARGIN", 0.0))
    beneficial = executable & (utility_target > positive_margin)
    harmful = executable & (
        (utility_target < -harmful_margin)
        | (harm_fraction > benefit_fraction)
    )
    gate_target = beneficial.any(dim=1, keepdim=True).float()
    best_utility, best_index = utility_target.masked_fill(~beneficial, -1.0e9).max(dim=1)
    positive_region = gate_target[:, 0] > 0.5

    gate_logit = aux["v520_gate_logit"]
    selector_logits = aux["v520_selector_logits"]
    utility_pred = aux["v520_utility_pred"]
    harm_logit = aux["v520_harm_logit"]
    gate_loss = _v520_balanced_bce_with_logits(
        gate_logit,
        gate_target,
        valid=aux.get("v520_any_executable", torch.ones_like(gate_target)).bool(),
    )

    valid_float = executable.float()
    valid_den = valid_float.sum().clamp_min(1.0)
    utility_loss = (
        F.smooth_l1_loss(
            utility_pred,
            utility_target.clamp(-1.0, 1.0),
            reduction="none",
            beta=float(_m1(cfg, "V520_UTILITY_BETA", 0.02)),
        ) * valid_float
    ).sum() / valid_den
    harm_loss = _v520_balanced_bce_with_logits(
        harm_logit,
        harmful.float(),
        valid=executable,
    )

    if bool(positive_region.any().item()):
        selector_ce_map = F.cross_entropy(
            selector_logits,
            best_index,
            reduction="none",
        )
        selector_loss = selector_ce_map[positive_region].mean()
        best_predicted_score = selector_logits.gather(
            1,
            best_index[:, None],
        )[:, 0]
        positive_rank_loss = F.relu(
            float(_m1(cfg, "V520_PRESERVE_RANK_MARGIN", 0.05))
            - best_predicted_score[positive_region]
        ).mean()

        other_mask = executable.clone()
        other_mask.scatter_(1, best_index[:, None], False)
        hardest_other = selector_logits.masked_fill(~other_mask, -1.0e9).amax(dim=1)
        pairwise_rank_loss = F.relu(
            float(_m1(cfg, "V520_SELECTOR_RANK_MARGIN", 0.05))
            - best_predicted_score[positive_region]
            + hardest_other[positive_region]
        ).mean()
    else:
        selector_loss = selector_logits.sum() * 0.0
        positive_rank_loss = selector_logits.sum() * 0.0
        pairwise_rank_loss = selector_logits.sum() * 0.0

    if bool(harmful.any().item()):
        harmful_rank_loss = F.relu(
            float(_m1(cfg, "V520_HARM_RANK_MARGIN", 0.05))
            + selector_logits[harmful]
        ).mean()
    else:
        harmful_rank_loss = selector_logits.sum() * 0.0
    ranking_loss = positive_rank_loss + pairwise_rank_loss + harmful_rank_loss

    training_prob = _as_b1hw(aux["m2_training_probs"])
    deployed_prob = _as_b1hw(aux["m2_fused_probs"])
    segmentation_loss = _v519_prob_bce_dice(training_prob, gt)
    selected_change = (training_prob - c0).abs()
    changed_support = (selected_change > float(_m1(cfg, "V520_EDIT_EPS", 1.0e-4))).float()
    base_error_single = (c0 - gt).abs()
    m2_error = (training_prob - gt).abs()
    noharm_loss = (
        F.relu(m2_error - base_error_single) * changed_support
    ).sum() / changed_support.sum().clamp_min(1.0)
    correct_base = ((c0 >= 0.5) == (gt >= 0.5)).float()
    selected_preserve_loss = (
        selected_change * correct_base * changed_support
    ).sum() / (correct_base * changed_support).sum().clamp_min(1.0)

    soft_gate = aux["v520_soft_gate"]
    dx = (soft_gate[..., :, 1:] - soft_gate[..., :, :-1]).abs().mean()
    dy = (soft_gate[..., 1:, :] - soft_gate[..., :-1, :]).abs().mean()
    coherence_loss = dx + dy

    m2_loss = (
        float(_m1(cfg, "V520_GATE_WEIGHT", 1.0)) * gate_loss
        + float(_m1(cfg, "V520_UTILITY_WEIGHT", 1.0)) * utility_loss
        + float(_m1(cfg, "V520_SELECTOR_WEIGHT", 1.0)) * selector_loss
        + float(_m1(cfg, "V520_RANKING_WEIGHT", 1.0)) * ranking_loss
        + float(_m1(cfg, "V520_HARM_WEIGHT", 0.5)) * harm_loss
        + float(_m1(cfg, "V520_SEGMENTATION_WEIGHT", 1.0)) * segmentation_loss
        + float(_m1(cfg, "V520_NOHARM_WEIGHT", 1.0)) * noharm_loss
        + float(_m1(cfg, "V520_SELECTED_PRESERVE_WEIGHT", 0.0))
        * selected_preserve_loss
        + float(_m1(cfg, "V520_COHERENCE_WEIGHT", 0.02)) * coherence_loss
    )

    epoch_value = int(epoch or 0)
    m1_scale = _v501_delayed_ramp(
        epoch_value,
        int(_m1(cfg, "V520_M1_START_EPOCH", 0)),
        int(_m1(cfg, "V520_M1_RAMP_EPOCHS", 3)),
        float(_m1(cfg, "V520_M1_FINAL_WEIGHT", 1.0)),
    )
    m2_scale = _v501_delayed_ramp(
        epoch_value,
        int(_m1(cfg, "V520_M2_START_EPOCH", 0)),
        int(_m1(cfg, "V520_M2_RAMP_EPOCHS", 5)),
        float(_m1(cfg, "V520_M2_FINAL_WEIGHT", 1.0)),
    )
    total = m1_scale * m1_loss + m2_scale * m2_loss

    base_dice = _soft_dice_probs(c0, gt).mean()
    training_dice = _soft_dice_probs(training_prob, gt).mean()
    deployed_dice = _soft_dice_probs(deployed_prob, gt).mean()
    base_hard_single = c0 >= 0.5
    deployed_hard = deployed_prob >= 0.5
    gt_hard_single = gt >= 0.5
    deployed_changed = deployed_hard != base_hard_single
    beneficial_change = (
        deployed_changed
        & (base_hard_single != gt_hard_single)
        & (deployed_hard == gt_hard_single)
    )
    harmful_change = (
        deployed_changed
        & (base_hard_single == gt_hard_single)
        & (deployed_hard != gt_hard_single)
    )
    changed_mass = deployed_changed.float().sum().clamp_min(1.0)

    hard_route = aux["v520_hard_route_weights"]
    soft_route = aux["v520_soft_route_weights"]
    selected_utility = utility_target.gather(
        1,
        aux["v520_selected_nonbase_index"][:, None],
    )[:, 0]
    zero = total * 0.0
    diag: Dict[str, torch.Tensor] = {
        **m1_diag,
        "v520_total_loss": total.detach(),
        "v520_m2_loss": m2_loss.detach(),
        "v520_gate_loss": gate_loss.detach(),
        "v520_utility_loss": utility_loss.detach(),
        "v520_selector_loss": selector_loss.detach(),
        "v520_ranking_loss": ranking_loss.detach(),
        "v520_positive_rank_loss": positive_rank_loss.detach(),
        "v520_pairwise_rank_loss": pairwise_rank_loss.detach(),
        "v520_harmful_rank_loss": harmful_rank_loss.detach(),
        "v520_harm_loss": harm_loss.detach(),
        "v520_segmentation_loss": segmentation_loss.detach(),
        "v520_noharm_loss": noharm_loss.detach(),
        "v520_selected_preserve_loss": selected_preserve_loss.detach(),
        "v520_coherence_loss": coherence_loss.detach(),
        "v520_teacher_positive_region_rate": gate_target.mean().detach(),
        "v520_teacher_beneficial_candidate_rate": beneficial.float().mean().detach(),
        "v520_teacher_harmful_candidate_rate": harmful.float().mean().detach(),
        "v520_teacher_mean_best_utility": torch.where(
            positive_region,
            best_utility,
            torch.zeros_like(best_utility),
        ).sum().detach() / positive_region.float().sum().clamp_min(1.0),
        "v520_predicted_soft_gate_rate": soft_gate.mean().detach(),
        "v520_predicted_hard_gate_rate": aux["v520_hard_gate"].mean().detach(),
        "v520_soft_nonbase_mass": (1.0 - soft_route[:, :1]).mean().detach(),
        "v520_hard_nonbase_mass": (1.0 - hard_route[:, :1]).mean().detach(),
        "v520_executable_rate": executable.float().mean().detach(),
        "v520_selected_mean_teacher_utility": selected_utility.mean().detach(),
        "v520_base_dice": base_dice.detach(),
        "v520_training_dice": training_dice.detach(),
        "v520_deployed_dice": deployed_dice.detach(),
        "v520_training_gain": (training_dice - base_dice).detach(),
        "v520_deployed_gain": (deployed_dice - base_dice).detach(),
        "v520_changed_pixel_rate": deployed_changed.float().mean().detach(),
        "v520_conditional_benefit_rate": (
            beneficial_change.float().sum() / changed_mass
        ).detach(),
        "v520_conditional_harm_rate": (
            harmful_change.float().sum() / changed_mass
        ).detach(),
        # Existing V490 router compatibility.  Crucially, both live objectives
        # are non-zero graph tensors and M3 remains bypassed.
        "v505_m1_loss": m1_loss.detach(),
        "v505_m2_loss": m2_loss.detach(),
        "v505_m3_loss": zero.detach(),
        "v505_m1_scale": total.new_tensor(m1_scale),
        "v505_m2_scale": total.new_tensor(m2_scale),
        "v505_m3_scale": zero.detach(),
        "v505_base_dice": base_dice.detach(),
        "v505_m2_dice": deployed_dice.detach(),
        "v505_final_dice": deployed_dice.detach(),
        "v505_m2_gain_vs_base": (deployed_dice - base_dice).detach(),
        "v505_final_gain_vs_base": (deployed_dice - base_dice).detach(),
        "v505_m2_changed_pixel_rate": deployed_changed.float().mean().detach(),
        "v505_m2_conditional_benefit_rate": (
            beneficial_change.float().sum() / changed_mass
        ).detach(),
        "v505_m2_conditional_harm_rate": (
            harmful_change.float().sum() / changed_mass
        ).detach(),
        "v505_selected_interaction_rate": aux["v520_hard_gate"].mean().detach(),
        "v505_m3_accept_rate": zero.detach(),
        "v505_interactive_region_causal_enabled": total.new_tensor(1.0),
        "v489_total_loss": total.detach(),
        "v489_m1_loss": m1_loss.detach(),
        "v489_m2_loss": m2_loss.detach(),
        "v489_m3_loss": zero.detach(),
        "v489_base_dice": base_dice.detach(),
        "v489_m2_dice": deployed_dice.detach(),
        "v489_final_dice": deployed_dice.detach(),
        "v489_m2_gain_vs_base": (deployed_dice - base_dice).detach(),
        "v489_final_gain_vs_m2": zero.detach(),
        "v488_total_loss": total.detach(),
        "v488_base_dice": base_dice.detach(),
        "v488_m2_dice": deployed_dice.detach(),
        "v488_m3_final_dice": deployed_dice.detach(),
        "v488_m2_gain_vs_base": (deployed_dice - base_dice).detach(),
        "v488_m3_gain_vs_m2": zero.detach(),
        "v484_m2_loss": m2_loss.detach(),
        "v484_m3_regret_loss": zero.detach(),
        "_v490_m1_objective": m1_scale * m1_loss,
        "_v490_m2_objective": m2_scale * m2_loss,
        "_v490_m3_objective": zero,
    }
    return total, diag

def _compute_v519_region_composer_loss(
    cfg: Any,
    gt: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    epoch: int | None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Train the frozen-bank, family-aware dense regional M2 composer.

    The supervision is Base-relative and deployment aligned:
      * per-candidate local effect regression;
      * calibrated uncertainty and explicit harm prediction;
      * Preserve-first route distillation at every pixel;
      * final segmentation, no-harm and route-coherence objectives.
    """
    gt = (_as_b1hw(gt) >= 0.5).float()
    candidate_probs = aux.get("candidate_probs")
    if not isinstance(candidate_probs, torch.Tensor) or candidate_probs.ndim != 4:
        raise RuntimeError("V519 requires aux['candidate_probs'] [B,K,H,W]")
    required = (
        "v519_route_logits",
        "v519_soft_route_weights",
        "v519_effect_mean",
        "v519_log_sigma",
        "v519_harm_logit",
        "v519_eligibility_logit",
        "v519_executable_mask",
        "v519_abs_edit",
        "v519_deploy_mask",
        "m2_training_probs",
        "m2_fused_probs",
    )
    missing = [key for key in required if key not in aux]
    if missing:
        raise RuntimeError("V519 loss missing outputs: " + str(missing))

    # Frozen observations/labels. M2 cannot improve its label by changing M1.
    candidates = candidate_probs.detach().clamp(EPS, 1.0 - EPS)
    c0 = candidates[:, :1]
    nonbase = candidates[:, 1:]
    b, n, h, w = nonbase.shape
    gt_bank = gt.expand(-1, n, -1, -1)
    c0_bank = c0.expand_as(nonbase)
    base_error = (c0_bank - gt_bank).abs()
    candidate_error = (nonbase - gt_bank).abs()
    effect_target = (base_error - candidate_error).detach()

    base_hard = c0_bank >= 0.5
    candidate_hard = nonbase >= 0.5
    gt_hard = gt_bank >= 0.5
    base_wrong = base_hard != gt_hard
    candidate_correct = candidate_hard == gt_hard
    candidate_wrong = ~candidate_correct
    benefit_target = (base_wrong & candidate_correct).float()
    harm_target = ((~base_wrong) & candidate_wrong).float()

    executable = aux["v519_executable_mask"].bool()
    deploy_mask = aux["v519_deploy_mask"][1:].bool()
    executable = executable & deploy_mask[None, :, None, None]
    abs_edit = aux["v519_abs_edit"].detach()
    cause_evidence = aux.get("v519_candidate_causes", candidates)[:, 1:].detach()
    relevance = executable.float() * (0.25 + 0.75 * cause_evidence)
    relevance_den = relevance.sum().clamp_min(1.0)

    # Region-aligned potential outcomes. V518 showed that strong candidates are
    # locally useful but often globally harmful.  Supervision is therefore
    # aggregated over the same neighbourhood used by the deploy-time composer.
    region_radius = max(int(_m1(cfg, "V519_REGION_RADIUS", 2)), 0)
    kernel = 2 * region_radius + 1

    def region_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if region_radius <= 0:
            return value
        bn = value.shape[0] * value.shape[1]
        numerator = F.avg_pool2d(
            (value * weight).reshape(bn, 1, h, w),
            kernel_size=kernel,
            stride=1,
            padding=region_radius,
        ).reshape(b, n, h, w)
        denominator = F.avg_pool2d(
            weight.reshape(bn, 1, h, w),
            kernel_size=kernel,
            stride=1,
            padding=region_radius,
        ).reshape(b, n, h, w).clamp_min(EPS)
        return numerator / denominator

    edit_presence = (abs_edit > float(_m1(cfg, "V519_EDIT_EPS", 1.0e-4))).float()
    region_effect_target = region_mean(effect_target, abs_edit)
    region_benefit_fraction = region_mean(benefit_target, edit_presence)
    region_harm_fraction = region_mean(harm_target, edit_presence)
    region_edit_magnitude = region_mean(abs_edit, edit_presence)
    region_net_outcome = region_benefit_fraction - region_harm_fraction

    effect_pred = aux["v519_effect_mean"]
    log_sigma = aux["v519_log_sigma"].clamp(-6.0, 3.0)
    residual = effect_pred - effect_target
    effect_loss = (
        F.smooth_l1_loss(effect_pred, effect_target, reduction="none", beta=0.05)
        * relevance
    ).sum() / relevance_den
    uncertainty_loss = (
        0.5 * (torch.exp(-log_sigma) * residual.square() + log_sigma)
        * relevance
    ).sum() / relevance_den

    predicted_region_effect = region_mean(effect_pred, abs_edit)
    region_effect_loss = (
        F.smooth_l1_loss(
            predicted_region_effect,
            region_effect_target,
            reduction="none",
            beta=0.05,
        ) * relevance
    ).sum() / relevance_den

    harm_logit = aux["v519_harm_logit"]
    eligibility_logit = aux["v519_eligibility_logit"]
    region_harm_target = (
        (region_harm_fraction > region_benefit_fraction)
        | (region_effect_target < 0.0)
    ).float()
    harm_loss = (
        F.binary_cross_entropy_with_logits(
            harm_logit, region_harm_target, reduction="none"
        ) * relevance
    ).sum() / relevance_den
    effect_margin = float(_m1(cfg, "V519_PIXEL_GAIN_MARGIN", 0.05))
    net_margin = float(_m1(cfg, "V519_REGION_NET_MARGIN", 0.02))
    eligibility_target = (
        (region_effect_target > effect_margin)
        & (region_net_outcome > net_margin)
        & executable
    ).float()
    eligibility_loss = (
        F.binary_cross_entropy_with_logits(
            eligibility_logit, eligibility_target, reduction="none"
        ) * relevance
    ).sum() / relevance_den

    # Preserve-first regional teacher.  A candidate receives mass only when its
    # local neighbourhood has positive net repair after accounting for nearby
    # harm and edit size.  Different regions may select different candidates.
    teacher_temperature = max(
        float(_m1(cfg, "V519_TEACHER_TEMPERATURE", 0.05)), 1.0e-3
    )
    edit_tie = float(_m1(cfg, "V519_TEACHER_EDIT_PENALTY", 0.05))
    outcome_weight = float(_m1(cfg, "V519_REGION_OUTCOME_WEIGHT", 0.5))
    utility = (
        region_effect_target
        + outcome_weight * region_net_outcome
        - edit_tie * region_edit_magnitude
    )
    beneficial = (
        executable
        & (region_effect_target > effect_margin)
        & (region_net_outcome > net_margin)
    )
    floor = -1.0e4 if utility.dtype in (torch.float16, torch.bfloat16) else -1.0e9
    candidate_teacher_logits = (utility / teacher_temperature).masked_fill(
        ~beneficial, floor
    )
    preserve_teacher = torch.zeros((b, 1, h, w), device=gt.device, dtype=gt.dtype)
    teacher_logits = torch.cat([preserve_teacher, candidate_teacher_logits], dim=1)
    teacher_distribution = F.softmax(teacher_logits, dim=1).detach()
    student_log_probability = F.log_softmax(
        aux["v519_route_logits"]
        / max(float(_m1(cfg, "V519_STUDENT_TEMPERATURE", 0.35)), 1.0e-3),
        dim=1,
    )
    route_loss = -(
        teacher_distribution * student_log_probability
    ).sum(dim=1).mean()

    training_prob = _as_b1hw(aux["m2_training_probs"])
    deployed_prob = _as_b1hw(aux["m2_fused_probs"])
    segmentation_loss = _v519_prob_bce_dice(training_prob, gt)

    base_error_single = (c0 - gt).abs()
    m2_error = (training_prob - gt).abs()
    noharm_loss = F.relu(m2_error - base_error_single.detach()).mean()
    correct_base_weight = ((c0 >= 0.5) == (gt >= 0.5)).float()
    preserve_correct_loss = (
        (training_prob - c0).abs() * correct_base_weight
    ).sum() / correct_base_weight.sum().clamp_min(1.0)

    route_weights = aux["v519_soft_route_weights"]
    dx = (route_weights[..., :, 1:] - route_weights[..., :, :-1]).abs().mean()
    dy = (route_weights[..., 1:, :] - route_weights[..., :-1, :]).abs().mean()
    coherence_loss = dx + dy
    nonbase_weight = 1.0 - route_weights[:, :1]
    edit_budget_loss = (
        nonbase_weight * (training_prob - c0).abs()
    ).mean()

    # Candidate-level effect calibration: metadata-aware M2 must distinguish
    # strong specialised candidates rather than treating every dose/radius as
    # an anonymous slot.
    edit_mass = aux["v519_abs_edit"].detach().flatten(2).sum(dim=2).clamp_min(1.0)
    predicted_candidate_effect = (
        effect_pred * aux["v519_abs_edit"].detach()
    ).flatten(2).sum(dim=2) / edit_mass
    base_dice_case = _soft_dice_probs(c0, gt)[:, 0]
    candidate_dice_case = _soft_dice_probs(nonbase, gt_bank)
    candidate_delta_dice = (candidate_dice_case - base_dice_case[:, None]).detach()
    candidate_effect_loss = F.smooth_l1_loss(
        predicted_candidate_effect,
        candidate_delta_dice,
        beta=0.01,
    )

    m2_loss = (
        float(_m1(cfg, "V519_EFFECT_WEIGHT", 1.0)) * effect_loss
        + float(_m1(cfg, "V519_UNCERTAINTY_WEIGHT", 0.25)) * uncertainty_loss
        + float(_m1(cfg, "V519_HARM_WEIGHT", 1.0)) * harm_loss
        + float(_m1(cfg, "V519_ELIGIBILITY_WEIGHT", 1.0)) * eligibility_loss
        + float(_m1(cfg, "V519_ROUTE_WEIGHT", 2.0)) * route_loss
        + float(_m1(cfg, "V519_SEGMENTATION_WEIGHT", 1.0)) * segmentation_loss
        + float(_m1(cfg, "V519_NOHARM_WEIGHT", 2.0)) * noharm_loss
        + float(_m1(cfg, "V519_PRESERVE_CORRECT_WEIGHT", 0.5)) * preserve_correct_loss
        + float(_m1(cfg, "V519_COHERENCE_WEIGHT", 0.05)) * coherence_loss
        + float(_m1(cfg, "V519_EDIT_BUDGET_WEIGHT", 0.05)) * edit_budget_loss
        + float(_m1(cfg, "V519_CANDIDATE_EFFECT_WEIGHT", 0.5)) * candidate_effect_loss
        + float(_m1(cfg, "V519_REGION_EFFECT_WEIGHT", 0.5)) * region_effect_loss
    )

    base_dice = _soft_dice_probs(c0, gt).mean()
    training_dice = _soft_dice_probs(training_prob, gt).mean()
    deployed_dice = _soft_dice_probs(deployed_prob, gt).mean()
    deployed_hard = deployed_prob >= 0.5
    base_hard_single = c0 >= 0.5
    gt_hard_single = gt >= 0.5
    changed = deployed_hard != base_hard_single
    beneficial_change = changed & (base_hard_single != gt_hard_single) & (deployed_hard == gt_hard_single)
    harmful_change = changed & (base_hard_single == gt_hard_single) & (deployed_hard != gt_hard_single)
    changed_mass = changed.float().sum().clamp_min(1.0)

    zero = m2_loss * 0.0
    diag: Dict[str, torch.Tensor] = {
        "v519_total_loss": m2_loss.detach(),
        "v519_effect_loss": effect_loss.detach(),
        "v519_uncertainty_loss": uncertainty_loss.detach(),
        "v519_harm_loss": harm_loss.detach(),
        "v519_eligibility_loss": eligibility_loss.detach(),
        "v519_route_loss": route_loss.detach(),
        "v519_segmentation_loss": segmentation_loss.detach(),
        "v519_noharm_loss": noharm_loss.detach(),
        "v519_preserve_correct_loss": preserve_correct_loss.detach(),
        "v519_coherence_loss": coherence_loss.detach(),
        "v519_edit_budget_loss": edit_budget_loss.detach(),
        "v519_candidate_effect_loss": candidate_effect_loss.detach(),
        "v519_region_effect_loss": region_effect_loss.detach(),
        "v519_teacher_region_benefit": region_benefit_fraction.mean().detach(),
        "v519_teacher_region_harm": region_harm_fraction.mean().detach(),
        "v519_teacher_region_net": region_net_outcome.mean().detach(),
        "v519_teacher_nonbase_rate": (1.0 - teacher_distribution[:, :1]).mean().detach(),
        "v519_predicted_nonbase_rate": (1.0 - route_weights[:, :1]).mean().detach(),
        "v519_executable_rate": executable.float().mean().detach(),
        "v519_predicted_harm_probability": torch.sigmoid(harm_logit).mean().detach(),
        "v519_base_dice": base_dice.detach(),
        "v519_training_dice": training_dice.detach(),
        "v519_deployed_dice": deployed_dice.detach(),
        "v519_training_gain": (training_dice - base_dice).detach(),
        "v519_deployed_gain": (deployed_dice - base_dice).detach(),
        "v519_changed_pixel_rate": changed.float().mean().detach(),
        "v519_conditional_benefit_rate": (
            beneficial_change.float().sum() / changed_mass
        ).detach(),
        "v519_conditional_harm_rate": (
            harmful_change.float().sum() / changed_mass
        ).detach(),
        # V505/V490 compatibility for the existing loss router/logger.
        "v505_m1_loss": zero.detach(),
        "v505_m2_loss": m2_loss.detach(),
        "v505_m3_loss": zero.detach(),
        "v505_m1_scale": zero.detach(),
        "v505_m2_scale": m2_loss.new_tensor(1.0),
        "v505_m3_scale": zero.detach(),
        "v505_base_dice": base_dice.detach(),
        "v505_m2_dice": deployed_dice.detach(),
        "v505_final_dice": deployed_dice.detach(),
        "v505_m2_gain_vs_base": (deployed_dice - base_dice).detach(),
        "v505_final_gain_vs_base": (deployed_dice - base_dice).detach(),
        "v505_m2_changed_pixel_rate": changed.float().mean().detach(),
        "v505_m2_conditional_benefit_rate": (
            beneficial_change.float().sum() / changed_mass
        ).detach(),
        "v505_m2_conditional_harm_rate": (
            harmful_change.float().sum() / changed_mass
        ).detach(),
        "v505_selected_interaction_rate": (1.0 - route_weights[:, :1]).mean().detach(),
        "v505_m3_accept_rate": zero.detach(),
        "v505_interactive_region_causal_enabled": m2_loss.new_tensor(1.0),
        "v489_total_loss": m2_loss.detach(),
        "v489_m1_loss": zero.detach(),
        "v489_m2_loss": m2_loss.detach(),
        "v489_m3_loss": zero.detach(),
        "v489_base_dice": base_dice.detach(),
        "v489_m2_dice": deployed_dice.detach(),
        "v489_final_dice": deployed_dice.detach(),
        "v489_m2_gain_vs_base": (deployed_dice - base_dice).detach(),
        "v489_final_gain_vs_m2": zero.detach(),
        "v488_total_loss": m2_loss.detach(),
        "v488_base_dice": base_dice.detach(),
        "v488_m2_dice": deployed_dice.detach(),
        "v488_m3_final_dice": deployed_dice.detach(),
        "v488_m2_gain_vs_base": (deployed_dice - base_dice).detach(),
        "v488_m3_gain_vs_m2": zero.detach(),
        "v484_m2_local_loss": m2_loss.detach(),
        "v484_m3_regret_loss": zero.detach(),
        "_v490_m1_objective": zero,
        "_v490_m2_objective": m2_loss,
        "_v490_m3_objective": zero,
    }
    return m2_loss, diag


def _compute_v505_interactive_region_loss(
    cfg: Any,
    gt: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    epoch: int | None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """V505 loss with one statistical unit: a complete region interaction.

    M1 learns cause-gated, net-improving candidates. M2 and M3 predict the
    region-level potential outcome of every typed box proposal.  All proposal
    losses are averaged over the natural proposal distribution; positive and
    negative subsets are never independently re-normalised.
    """
    candidate_probs = aux.get("candidate_probs")
    if not isinstance(candidate_probs, torch.Tensor) or candidate_probs.ndim != 4:
        raise RuntimeError("V505 requires aux['candidate_probs'] [B,K,H,W]")
    if candidate_probs.shape[1] < 5:
        raise RuntimeError("V505 requires Preserve + Delete/Fill/Trim/Expand")
    c0 = candidate_probs[:, :1].detach().clamp(EPS, 1.0 - EPS)
    local = candidate_probs[:, 1:5].clamp(EPS, 1.0 - EPS)
    gt = (_as_b1hw(gt) >= 0.5).float()

    required = (
        "error_cause_map_logits",
        "failure_state_logit",
        "local_raw_supports",
        "local_supports",
        "v505_proposal_masks",
        "v505_proposal_action_ids",
        "v505_m2_step_current_probs",
        "v505_m2_outcome_logits",
        "v505_m2_delta_dice_pred",
        "v505_m2_delta_surface_pred",
        "v505_m2_proposal_score",
        "v505_m2_proposal_available",
        "v505_m2_selected_proposal",
        "v505_m3_step_current_probs",
        "v505_m3_outcome_logits",
        "v505_m3_delta_dice_pred",
        "v505_m3_delta_surface_pred",
        "v505_m3_proposal_score",
        "v505_m3_proposal_valid",
        "v505_m3_accept",
        "m2_fused_probs",
        "final_probs",
    )
    missing = [key for key in required if key not in aux]
    if missing:
        raise RuntimeError("V505 loss missing outputs: " + str(missing))

    factual = build_factual_cause_targets(
        c0,
        gt,
        boundary_radius=int(_m1(cfg, "V503_FACTUAL_BOUNDARY_RADIUS", 2)),
        failure_dice=float(_m1(cfg, "V503_FAILURE_DICE", 0.55)),
        include_global_action=False,
    )
    cause_target = factual["cause_targets"]
    cause_logits = aux["error_cause_map_logits"]
    failure_logit = aux["failure_state_logit"]
    raw_support = aux["local_raw_supports"][:, :4]
    effective_support = aux["local_supports"][:, :4]

    focal_gamma = float(_m1(cfg, "V505_FOCAL_GAMMA", 2.0))
    v507_precision_first = bool(
        _m1(cfg, "V507_PRECISION_FIRST_CAUSE_SUPERVISION", False)
    )
    if v507_precision_first:
        max_positive_weight = float(_m1(cfg, "V507_MAX_POSITIVE_WEIGHT", 8.0))
        cause_loss = _v507_dynamic_sparse_focal_with_logits(
            cause_logits, cause_target, focal_gamma, max_positive_weight
        )
        failure_loss = _v507_dynamic_sparse_focal_with_logits(
            failure_logit,
            factual["failure_target"],
            focal_gamma,
            max(2.0, min(max_positive_weight, 4.0)),
        )
        support_logits = torch.logit(raw_support.clamp(EPS, 1.0 - EPS))
        support_loss = _v507_dynamic_sparse_focal_with_logits(
            support_logits, cause_target, focal_gamma, max_positive_weight
        )
        cause_probability = torch.sigmoid(cause_logits)
        cause_tversky = _v507_precision_tversky_loss(
            cause_probability,
            cause_target,
            fp_weight=float(_m1(cfg, "V507_TVERSKY_FP_WEIGHT", 0.70)),
            fn_weight=float(_m1(cfg, "V507_TVERSKY_FN_WEIGHT", 0.30)),
        )
        support_tversky = _v507_precision_tversky_loss(
            raw_support,
            cause_target,
            fp_weight=float(_m1(cfg, "V507_TVERSKY_FP_WEIGHT", 0.70)),
            fn_weight=float(_m1(cfg, "V507_TVERSKY_FN_WEIGHT", 0.30)),
        )
        cause_prevalence = _v507_prevalence_calibration_loss(
            cause_probability, cause_target
        )
        support_prevalence = _v507_prevalence_calibration_loss(
            raw_support, cause_target
        )
    else:
        cause_loss = _v505_binary_focal_with_logits(cause_logits, cause_target, focal_gamma)
        failure_loss = _v505_binary_focal_with_logits(
            failure_logit,
            factual["failure_target"],
            focal_gamma,
        )
        support_loss = _v505_binary_focal_with_logits(
            torch.logit(raw_support.clamp(EPS, 1.0 - EPS)),
            cause_target,
            focal_gamma,
        )
        cause_tversky = cause_loss * 0.0
        support_tversky = support_loss * 0.0
        cause_prevalence = cause_loss * 0.0
        support_prevalence = support_loss * 0.0

    v518_enabled = bool(_m1(cfg, "V518_ENABLED", False))
    if v518_enabled:
        v518_locator_loss, v518_locator_parts = _v518_hierarchical_locator_loss(
            cfg, cause_target, aux
        )
        v518_bank_loss, v518_bank_parts = _v518_bank_objectives(
            cfg, candidate_probs, c0, gt, aux
        )
    else:
        zero_v518 = candidate_probs.sum() * 0.0
        v518_locator_loss = zero_v518
        v518_bank_loss = zero_v518
        v518_locator_parts = {
            "polarity_loss": zero_v518,
            "fp_subtype_loss": zero_v518,
            "fn_subtype_loss": zero_v518,
        }
        v518_bank_parts = {
            "bank_best_loss": zero_v518,
            "coverage_loss": zero_v518,
            "diversity_loss": zero_v518,
            "soft_best_gain": zero_v518.detach(),
            "soft_error_coverage": zero_v518.detach(),
        }

    base_error = (c0 - gt).abs().expand_as(local)
    candidate_error = (local - gt.expand_as(local)).abs()
    repair_gain = base_error - candidate_error
    actual_change = (local - c0).abs()
    temperature = max(
        float(_m1(cfg, "V505_NET_REGRET_TEMPERATURE", 0.10)),
        1.0e-3,
    )
    repair_margin = max(
        float(_m1(cfg, "V506_REPAIR_GAIN_MARGIN", 0.01)),
        0.0,
    )

    # On factual cause pixels, no-op is not an acceptable optimum: the
    # candidate must improve the absolute error by a positive margin.  Outside
    # its typed cause, only actual changes are penalised, so Preserve itself is
    # not punished.
    repair_term = F.softplus(
        (repair_margin - repair_gain) / temperature
    )
    repair_regret = (
        repair_term * cause_target
    ).sum() / cause_target.sum().clamp_min(1.0)

    outside_change = actual_change.detach() * (1.0 - cause_target)
    outside_harm = F.softplus(
        (candidate_error - base_error) / temperature
    )
    outside_harm_regret = (
        outside_harm * outside_change
    ).sum() / outside_change.sum().clamp_min(1.0)
    net_regret = repair_regret + outside_harm_regret

    scope_loss = (
        actual_change * (1.0 - cause_target)
    ).sum() / actual_change.sum().clamp_min(1.0)

    margin = float(_m1(cfg, "V505_EXECUTABILITY_MARGIN", 0.02))
    negative_action = torch.tensor([1.0, 0.0, 1.0, 0.0], device=local.device, dtype=local.dtype)[None, :, None, None]
    positive_action = 1.0 - negative_action
    execution_violation = (
        negative_action * F.relu(local - (0.5 - margin))
        + positive_action * F.relu((0.5 + margin) - local)
    )
    executability_loss = (
        execution_violation * cause_target
    ).sum() / cause_target.sum().clamp_min(1.0)

    v507_hard_enabled = bool(
        _m1(cfg, "V507_HARD_DEPLOYMENT_OBJECTIVE_ENABLED", False)
    )
    if v507_hard_enabled:
        _, v507_hard_parts = _v507_candidate_hard_objectives(
            c0,
            local,
            gt,
            cause_target,
            gain_margin=float(_m1(cfg, "V507_HARD_GAIN_MARGIN", 0.002)),
        )
    else:
        zero = local.sum() * 0.0
        v507_hard_parts = {
            "hard_gain_loss": zero,
            "hard_noharm_loss": zero,
            "hard_scope_loss": zero,
            "hard_benefit_rate": zero.detach(),
            "hard_harm_rate": zero.detach(),
            "hard_mean_delta_dice": zero.detach(),
        }

    localization_loss = (
        float(_m1(cfg, "V505_CAUSE_WEIGHT", 1.0)) * cause_loss
        + float(_m1(cfg, "V505_FAILURE_WEIGHT", 0.5)) * failure_loss
        + float(_m1(cfg, "V505_SUPPORT_WEIGHT", 1.0)) * support_loss
        + float(_m1(cfg, "V507_CAUSE_TVERSKY_WEIGHT", 0.0)) * cause_tversky
        + float(_m1(cfg, "V507_SUPPORT_TVERSKY_WEIGHT", 0.0)) * support_tversky
        + float(_m1(cfg, "V507_PREVALENCE_WEIGHT", 0.0))
        * (cause_prevalence + support_prevalence)
        + v518_locator_loss
    )
    safety_loss = (
        float(_m1(cfg, "V505_NET_REGRET_WEIGHT", 1.0)) * net_regret
        + float(_m1(cfg, "V505_SCOPE_PURITY_WEIGHT", 1.0)) * scope_loss
        + float(_m1(cfg, "V505_EXECUTABILITY_WEIGHT", 1.0)) * executability_loss
        + float(_m1(cfg, "V507_HARD_GAIN_WEIGHT", 0.0))
        * v507_hard_parts["hard_gain_loss"]
        + float(_m1(cfg, "V507_HARD_NOHARM_WEIGHT", 0.0))
        * v507_hard_parts["hard_noharm_loss"]
        + float(_m1(cfg, "V507_HARD_SCOPE_WEIGHT", 0.0))
        * v507_hard_parts["hard_scope_loss"]
    )
    if v518_enabled:
        epoch_value = int(epoch or 0)
        locator_only_epochs = max(
            int(_m1(cfg, "V518_LOCATOR_ONLY_EPOCHS", 8)), 0
        )
        bank_start = max(
            int(_m1(cfg, "V518_BANK_LOSS_START_EPOCH", locator_only_epochs)),
            locator_only_epochs,
        )
        bank_ramp_epochs = max(int(_m1(cfg, "V518_BANK_RAMP_EPOCHS", 8)), 1)
        safety_ramp_epochs = max(int(_m1(cfg, "V518_SAFETY_RAMP_EPOCHS", 12)), 1)
        if epoch_value < locator_only_epochs:
            bank_scale = 0.0
            safety_scale = 0.0
        else:
            bank_scale = min(
                max((epoch_value - bank_start + 1) / float(bank_ramp_epochs), 0.0),
                1.0,
            )
            safety_scale = min(
                max((epoch_value - locator_only_epochs + 1) / float(safety_ramp_epochs), 0.0),
                1.0,
            )
        m1_loss = localization_loss + bank_scale * v518_bank_loss + safety_scale * safety_loss
    else:
        bank_scale = 0.0
        safety_scale = 1.0
        m1_loss = localization_loss + safety_loss

    proposal_masks = aux["v505_proposal_masks"]
    action_ids = aux["v505_proposal_action_ids"]
    surface_radius = int(_m1(cfg, "V505_SURFACE_TOLERANCE_RADIUS", 2))
    chunk_size = int(_m1(cfg, "V505_TEACHER_CHUNK_SIZE", 8))
    teacher_size = int(_m1(cfg, "V505_TEACHER_SIZE", 56))

    def make_teacher(step_current: torch.Tensor) -> Dict[str, torch.Tensor]:
        teachers = []
        for step in range(step_current.shape[1]):
            teachers.append(
                build_region_potential_outcomes(
                    step_current[:, step:step + 1],
                    candidate_probs,
                    proposal_masks,
                    action_ids,
                    gt,
                    surface_radius=surface_radius,
                    chunk_size=chunk_size,
                    teacher_size=teacher_size,
                )
            )
        keys = teachers[0].keys()
        return {key: torch.stack([teacher[key] for teacher in teachers], dim=1) for key in keys}

    m2_teacher = make_teacher(aux["v505_m2_step_current_probs"])
    m3_teacher = make_teacher(aux["v505_m3_step_current_probs"])

    def outcome_objective(
        prefix: str,
        teacher: Dict[str, torch.Tensor],
        outcome_valid: torch.Tensor,
        decision_valid: torch.Tensor,
        selected_for_verifier: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        logits = aux[f"v505_{prefix}_outcome_logits"]
        pred_dice = aux[f"v505_{prefix}_delta_dice_pred"]
        pred_surface = aux[f"v505_{prefix}_delta_surface_pred"]
        score = aux[f"v505_{prefix}_proposal_score"]
        if not bool(torch.isfinite(score).all().item()):
            raise RuntimeError(
                f"V505 {prefix} learned proposal_score contains non-finite values"
            )
        outcome_valid = outcome_valid.bool()
        decision_valid = decision_valid.bool()
        class_loss = _v505_multiclass_focal(
            logits,
            teacher["outcome_class"],
            focal_gamma,
            outcome_valid,
        )
        dice_regression = _v505_masked_smooth_l1(
            pred_dice, teacher["delta_dice"], outcome_valid
        )
        surface_regression = _v505_masked_smooth_l1(
            pred_surface, teacher["delta_surface"], outcome_valid
        )
        margin_aligned = bool(_m1(cfg, "V507_MARGIN_ALIGNED_DECISION", False))
        selection_margin = (
            float(_m1(cfg, "V506_SELECTION_MARGIN", 0.05))
            if margin_aligned else 0.0
        )
        score_temperature = float(_m1(cfg, "V507_SCORE_TEMPERATURE", 0.25))
        ranking = _v505_region_ranking_loss(
            score,
            teacher["outcome_class"],
            decision_valid,
            selection_margin=selection_margin,
            temperature=score_temperature if margin_aligned else 1.0,
        )
        all_proposal_decision = score.sum() * 0.0
        selected_decision = score.sum() * 0.0
        if selected_for_verifier is None:
            if bool(_m1(cfg, "V507_SOFT_LISTWISE_DECISION", False)):
                decision = _v507_soft_preserve_decision_loss(
                    score,
                    teacher["outcome_class"],
                    teacher["delta_dice"],
                    teacher["delta_surface"],
                    float(_m1(cfg, "V505_SURFACE_SCORE_WEIGHT", 0.5)),
                    decision_valid,
                    selection_margin=selection_margin,
                    teacher_temperature=float(
                        _m1(cfg, "V507_TEACHER_UTILITY_TEMPERATURE", 0.02)
                    ),
                    student_temperature=score_temperature,
                )
            else:
                decision = _v505_listwise_preserve_decision_loss(
                    score,
                    teacher["outcome_class"],
                    teacher["delta_dice"],
                    teacher["delta_surface"],
                    float(_m1(cfg, "V505_SURFACE_SCORE_WEIGHT", 0.5)),
                    decision_valid,
                    selection_margin=selection_margin,
                    temperature=score_temperature if margin_aligned else 1.0,
                )
        else:
            all_proposal_decision = _v506_all_proposal_verifier_loss(
                score,
                teacher["outcome_class"],
                teacher["delta_dice"],
                teacher["delta_surface"],
                float(_m1(cfg, "V505_SURFACE_SCORE_WEIGHT", 0.5)),
                decision_valid,
                float(_m1(cfg, "V506_VERIFIER_UTILITY_MARGIN", 0.0)),
                selection_margin=selection_margin,
                temperature=score_temperature if margin_aligned else 1.0,
            )
            selected_decision = _v505_selected_verifier_decision_loss(
                score,
                teacher["outcome_class"],
                teacher["delta_dice"],
                teacher["delta_surface"],
                float(_m1(cfg, "V505_SURFACE_SCORE_WEIGHT", 0.5)),
                selected_for_verifier,
                decision_valid,
                selection_margin=selection_margin,
                temperature=score_temperature if margin_aligned else 1.0,
            )
            decision = (
                float(_m1(cfg, "V506_VERIFIER_ALL_PROPOSAL_WEIGHT", 1.0))
                * all_proposal_decision
                + float(_m1(cfg, "V506_SELECTED_VERIFIER_WEIGHT", 0.5))
                * selected_decision
            )
        total = (
            float(_m1(cfg, "V505_OUTCOME_CLASS_WEIGHT", 1.0)) * class_loss
            + float(_m1(cfg, "V505_DELTA_DICE_WEIGHT", 1.0)) * dice_regression
            + float(_m1(cfg, "V505_DELTA_SURFACE_WEIGHT", 0.5)) * surface_regression
            + float(_m1(cfg, "V505_RANKING_WEIGHT", 1.0)) * ranking
            + float(_m1(cfg, "V505_DECISION_CE_WEIGHT", 1.0)) * decision
        )
        return total, {
            "class_loss": class_loss,
            "dice_regression": dice_regression,
            "surface_regression": surface_regression,
            "ranking_loss": ranking,
            "decision_loss": decision,
            "all_proposal_decision_loss": all_proposal_decision,
            "selected_decision_loss": selected_decision,
            "outcome_valid_rate": outcome_valid.float().mean(),
            "decision_valid_rate": decision_valid.float().mean(),
        }

    selected = aux["v505_m2_selected_proposal"]
    m2_available = aux["v505_m2_proposal_available"].bool()
    m2_train_valid = aux.get(
        "v505_m2_proposal_train_valid",
        aux.get("v505_m2_proposal_valid", m2_available),
    ).bool()
    m2_deploy_valid = aux.get(
        "v505_m2_proposal_deploy_valid",
        m2_available,
    ).bool()
    m3_train_valid = aux.get(
        "v505_m3_proposal_train_valid",
        aux["v505_m3_proposal_valid"],
    ).bool()
    m3_deploy_valid = aux.get(
        "v505_m3_proposal_deploy_valid",
        aux["v505_m3_proposal_valid"],
    ).bool()

    # Classification/regression may learn from soft cause/support evidence, but
    # ranking and Preserve-vs-edit decisions are trained only on proposals that
    # are executable under the exact deployment threshold.
    use_dual_valid = bool(_m1(cfg, "V507_DEPLOY_VALID_DECISION_ONLY", False))
    m2_decision_valid = m2_deploy_valid if use_dual_valid else m2_train_valid
    m3_decision_valid = m3_deploy_valid if use_dual_valid else m3_train_valid
    m2_loss, m2_parts = outcome_objective(
        "m2", m2_teacher, m2_train_valid, m2_decision_valid
    )
    m3_loss, m3_parts = outcome_objective(
        "m3",
        m3_teacher,
        m3_train_valid,
        m3_decision_valid,
        selected_for_verifier=selected,
    )

    epoch_value = 0 if epoch is None else int(epoch)
    m1_scale = _v501_delayed_ramp(
        epoch_value,
        int(_m1(cfg, "V505_M1_START_EPOCH", 0)),
        int(_m1(cfg, "V505_M1_RAMP_EPOCHS", 3)),
        float(_m1(cfg, "V505_M1_FINAL_WEIGHT", 1.0)),
    )
    m2_scale = _v501_delayed_ramp(
        epoch_value,
        int(_m1(cfg, "V505_M2_START_EPOCH", 3)),
        int(_m1(cfg, "V505_M2_RAMP_EPOCHS", 5)),
        float(_m1(cfg, "V505_M2_FINAL_WEIGHT", 1.0)),
    )
    m3_scale = _v501_delayed_ramp(
        epoch_value,
        int(_m1(cfg, "V505_M3_START_EPOCH", 5)),
        int(_m1(cfg, "V505_M3_RAMP_EPOCHS", 5)),
        float(_m1(cfg, "V505_M3_FINAL_WEIGHT", 0.5)),
    )
    total = m1_scale * m1_loss + m2_scale * m2_loss + m3_scale * m3_loss

    selected_class = _v505_gather_selected(
        m2_teacher["outcome_class"].float(), selected, preserve_value=float(OUTCOME_NEUTRAL)
    ).long()
    selected_benefit = (selected_class == OUTCOME_BENEFIT).float()
    selected_harm = (selected_class == OUTCOME_HARM).float()
    selected_nonpreserve = (selected >= 0).float()
    accept = aux["v505_m3_accept"].float()

    m2 = _as_b1hw(aux["m2_fused_probs"])
    final = _as_b1hw(aux["final_probs"])
    base_hard = c0 >= 0.5
    m2_hard = m2 >= 0.5
    final_hard = final >= 0.5
    gt_hard = gt >= 0.5
    m2_changed = m2_hard != base_hard
    final_changed = final_hard != base_hard
    m2_benefit_pixels = m2_changed & (base_hard != gt_hard) & (m2_hard == gt_hard)
    m2_harm_pixels = m2_changed & (base_hard == gt_hard) & (m2_hard != gt_hard)
    final_benefit_pixels = final_changed & (base_hard != gt_hard) & (final_hard == gt_hard)
    final_harm_pixels = final_changed & (base_hard == gt_hard) & (final_hard != gt_hard)

    def conditional_rate(mask: torch.Tensor, denominator_mask: torch.Tensor) -> torch.Tensor:
        return mask.float().sum() / denominator_mask.float().sum().clamp_min(1.0)

    def proposal_rate(condition: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        valid = valid_mask.float()
        return (condition.float() * valid).sum() / valid.sum().clamp_min(1.0)

    base_dice = _soft_dice_probs(c0, gt).mean()
    m2_dice = _soft_dice_probs(m2, gt).mean()
    final_dice = _soft_dice_probs(final, gt).mean()

    diag: Dict[str, torch.Tensor] = {
        "v505_total_loss": total.detach(),
        "v505_m1_loss": m1_loss.detach(),
        "v518_localization_loss": localization_loss.detach(),
        "v518_safety_loss": safety_loss.detach(),
        "v518_locator_loss": v518_locator_loss.detach(),
        "v518_polarity_loss": v518_locator_parts["polarity_loss"].detach(),
        "v518_fp_subtype_loss": v518_locator_parts["fp_subtype_loss"].detach(),
        "v518_fn_subtype_loss": v518_locator_parts["fn_subtype_loss"].detach(),
        "v518_bank_loss": v518_bank_loss.detach(),
        "v518_bank_best_loss": v518_bank_parts["bank_best_loss"].detach(),
        "v518_coverage_loss": v518_bank_parts["coverage_loss"].detach(),
        "v518_diversity_loss": v518_bank_parts["diversity_loss"].detach(),
        "v518_soft_best_gain": v518_bank_parts["soft_best_gain"],
        "v518_soft_error_coverage": v518_bank_parts["soft_error_coverage"],
        "v518_bank_scale": m1_loss.new_tensor(float(bank_scale)),
        "v518_safety_scale": m1_loss.new_tensor(float(safety_scale)),
        "v505_m2_loss": m2_loss.detach(),
        "v505_m3_loss": m3_loss.detach(),
        "v505_m1_scale": c0.new_tensor(m1_scale),
        "v505_m2_scale": c0.new_tensor(m2_scale),
        "v505_m3_scale": c0.new_tensor(m3_scale),
        "v505_cause_loss": cause_loss.detach(),
        "v505_failure_loss": failure_loss.detach(),
        "v505_support_loss": support_loss.detach(),
        "v505_m1_net_regret_loss": net_regret.detach(),
        "v505_m1_scope_purity_loss": scope_loss.detach(),
        "v505_m1_executability_loss": executability_loss.detach(),
        "v507_cause_tversky_loss": cause_tversky.detach(),
        "v507_support_tversky_loss": support_tversky.detach(),
        "v507_cause_prevalence_loss": cause_prevalence.detach(),
        "v507_support_prevalence_loss": support_prevalence.detach(),
        "v507_cause_target_rate": cause_target.mean().detach(),
        "v507_cause_prediction_rate": torch.sigmoid(cause_logits).mean().detach(),
        "v507_support_prediction_rate": raw_support.mean().detach(),
        "v507_m1_hard_gain_loss": v507_hard_parts["hard_gain_loss"].detach(),
        "v507_m1_hard_noharm_loss": v507_hard_parts["hard_noharm_loss"].detach(),
        "v507_m1_hard_scope_loss": v507_hard_parts["hard_scope_loss"].detach(),
        "v507_m1_hard_benefit_rate": v507_hard_parts["hard_benefit_rate"].detach(),
        "v507_m1_hard_harm_rate": v507_hard_parts["hard_harm_rate"].detach(),
        "v507_m1_hard_mean_delta_dice": v507_hard_parts["hard_mean_delta_dice"].detach(),
        "v505_m2_outcome_class_loss": m2_parts["class_loss"].detach(),
        "v505_m2_delta_dice_loss": m2_parts["dice_regression"].detach(),
        "v505_m2_delta_surface_loss": m2_parts["surface_regression"].detach(),
        "v505_m2_ranking_loss": m2_parts["ranking_loss"].detach(),
        "v505_m2_decision_loss": m2_parts["decision_loss"].detach(),
        "v505_m3_outcome_class_loss": m3_parts["class_loss"].detach(),
        "v505_m3_delta_dice_loss": m3_parts["dice_regression"].detach(),
        "v505_m3_delta_surface_loss": m3_parts["surface_regression"].detach(),
        "v505_m3_ranking_loss": m3_parts["ranking_loss"].detach(),
        "v505_m3_decision_loss": m3_parts["decision_loss"].detach(),
        "v506_m3_all_proposal_decision_loss": m3_parts[
            "all_proposal_decision_loss"
        ].detach(),
        "v506_m3_selected_decision_loss": m3_parts[
            "selected_decision_loss"
        ].detach(),
        "v505_m2_available_proposal_rate": m2_available.float().mean().detach(),
        "v506_m2_train_valid_proposal_rate": m2_parts["outcome_valid_rate"].detach(),
        "v507_m2_decision_valid_proposal_rate": m2_parts["decision_valid_rate"].detach(),
        "v505_m3_active_valid_proposal_rate": aux.get(
            "v505_m3_proposal_deploy_valid",
            aux["v505_m3_proposal_valid"],
        ).float().mean().detach(),
        "v506_m3_train_valid_proposal_rate": m3_parts["outcome_valid_rate"].detach(),
        "v507_m3_decision_valid_proposal_rate": m3_parts["decision_valid_rate"].detach(),
        "v505_teacher_benefit_proposal_rate": proposal_rate(
            m2_teacher["outcome_class"] == OUTCOME_BENEFIT, m2_available
        ).detach(),
        "v505_teacher_harm_proposal_rate": proposal_rate(
            m2_teacher["outcome_class"] == OUTCOME_HARM, m2_available
        ).detach(),
        "v505_selected_interaction_rate": selected_nonpreserve.mean().detach(),
        "v505_selected_region_benefit_rate": (
            (selected_benefit * selected_nonpreserve).sum() / selected_nonpreserve.sum().clamp_min(1.0)
        ).detach(),
        "v505_selected_region_harm_rate": (
            (selected_harm * selected_nonpreserve).sum() / selected_nonpreserve.sum().clamp_min(1.0)
        ).detach(),
        "v505_m3_accept_rate": accept.mean().detach(),
        "v505_m3_harmful_accept_rate": (
            (accept * selected_harm).sum() / accept.sum().clamp_min(1.0)
        ).detach(),
        "v505_m3_beneficial_accept_recall": (
            (accept * selected_benefit).sum() / selected_benefit.sum().clamp_min(1.0)
        ).detach(),
        "v505_m2_changed_pixel_rate": m2_changed.float().mean().detach(),
        "v505_m2_conditional_benefit_rate": conditional_rate(m2_benefit_pixels, m2_changed).detach(),
        "v505_m2_conditional_harm_rate": conditional_rate(m2_harm_pixels, m2_changed).detach(),
        "v505_final_changed_pixel_rate": final_changed.float().mean().detach(),
        "v505_final_conditional_benefit_rate": conditional_rate(final_benefit_pixels, final_changed).detach(),
        "v505_final_conditional_harm_rate": conditional_rate(final_harm_pixels, final_changed).detach(),
        "v505_base_dice": base_dice.detach(),
        "v505_m2_dice": m2_dice.detach(),
        "v505_final_dice": final_dice.detach(),
        "v505_m2_gain_vs_base": (m2_dice - base_dice).detach(),
        "v505_final_gain_vs_base": (final_dice - base_dice).detach(),
        "v505_interactive_region_causal_enabled": c0.new_tensor(1.0),
        # Compatibility aliases consumed by train.py summaries.
        "v489_total_loss": total.detach(),
        "v489_m1_loss": m1_loss.detach(),
        "v489_m2_loss": m2_loss.detach(),
        "v489_m3_loss": m3_loss.detach(),
        "v489_base_dice": base_dice.detach(),
        "v489_m2_dice": m2_dice.detach(),
        "v489_final_dice": final_dice.detach(),
        "v489_m2_gain_vs_base": (m2_dice - base_dice).detach(),
        "v489_final_gain_vs_m2": (final_dice - m2_dice).detach(),
        "v488_total_loss": total.detach(),
        "v488_base_dice": base_dice.detach(),
        "v488_m2_dice": m2_dice.detach(),
        "v488_m3_final_dice": final_dice.detach(),
        "v488_m2_gain_vs_base": (m2_dice - base_dice).detach(),
        "v488_m3_gain_vs_m2": (final_dice - m2_dice).detach(),
        "v484_m2_local_loss": m2_loss.detach(),
        "v484_m3_regret_loss": m3_loss.detach(),
        "_v490_m1_objective": m1_scale * m1_loss,
        "_v490_m2_objective": m2_scale * m2_loss,
        "_v490_m3_objective": m3_scale * m3_loss,
    }
    return total, diag



def _v521_region_sum(value: torch.Tensor, region_masks: torch.Tensor) -> torch.Tensor:
    """Sum [B,N,H,W] values inside [B,R,H,W] -> [B,R,N]."""
    return torch.einsum("bnhw,brhw->brn", value, region_masks)


def _v521_safe_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if bool(mask.any().item()):
        return value[mask].mean()
    return value.sum() * 0.0



def _v523_balanced_mean(value: torch.Tensor, masks: list[torch.Tensor]) -> torch.Tensor:
    """Average equally across non-empty decision groups.

    V523 correctable pixels are extremely rare.  Equal group aggregation keeps
    repair, preserve-correct and uncorrectable decisions visible without using
    dataset-specific hand-tuned class weights.
    """
    parts = []
    for mask in masks:
        mask = mask.bool()
        if bool(mask.any().item()):
            parts.append(value[mask].mean())
    if parts:
        return torch.stack(parts).mean()
    return value.sum() * 0.0


def _compute_v523_sea_level_utility_loss(
    cfg: Any,
    gt: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    epoch: int | None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Exact-repair, risk-coupled V523 objective.

    The deployed task is binary segmentation, so every candidate that actually
    flips the current hard label has only two possible factual outcomes: it
    repairs a wrong Base pixel or harms a correct Base pixel.  V523.2 trains
    those outcomes directly and routes only among hard-changing candidates.
    Preserve remains action zero.  Soft BCE improvement is used only as a small
    tie-break between multiple exact repairs, never as permission to edit.
    """
    gt = (_as_b1hw(gt) >= 0.5).float()
    candidate_probs = aux.get("candidate_probs")
    if not isinstance(candidate_probs, torch.Tensor) or candidate_probs.ndim != 4:
        raise RuntimeError("V523 requires aux['candidate_probs'] [B,K,H,W]")
    required = (
        "v523_step_current_before",
        "v523_step_candidate_score",
        "v523_step_candidate_utility",
        "v523_step_candidate_benefit_logit",
        "v523_step_candidate_harm_logit",
        "v523_step_candidate_valid",
        "v523_step_action_score",
        "v523_step_soft_route",
        "v523_step_hard_route",
        "v523_step_selected_index",
        "v523_step_hard_edit_mask",
        "v523_step_edit_prob",
        "v523_step_top_gap",
        "v523_step_selected_risk",
        "v523_step_selected_benefit",
        "v523_step_selected_coherence",
        "v523_step_edge_map",
        "v523_deploy_mask",
        "m2_training_probs",
        "m2_fused_probs",
    )
    missing = [key for key in required if key not in aux]
    if missing:
        raise RuntimeError("V523 loss missing outputs: " + str(missing))

    # M1 owns candidate generation.  M2 observes detached Base/candidates and
    # cannot move its own factual labels through the routing objective.
    m1_loss, raw_m1_diag = _compute_v520_m1_objective(cfg, gt, aux, epoch)
    m1_diag = dict(raw_m1_diag)
    for key, value in raw_m1_diag.items():
        if key.startswith("v520_m1_"):
            m1_diag["v523_m1_" + key[len("v520_m1_"):]] = value

    candidates = candidate_probs.detach().clamp(EPS, 1.0 - EPS)
    c0 = candidates[:, :1]
    nonbase = candidates[:, 1:]
    _, n, _, _ = nonbase.shape
    gt_bank = gt.expand(-1, n, -1, -1)

    step_current = aux["v523_step_current_before"]
    step_action_score = aux["v523_step_action_score"]
    step_utility = aux["v523_step_candidate_utility"]
    step_benefit_logit = aux["v523_step_candidate_benefit_logit"]
    step_harm_logit = aux["v523_step_candidate_harm_logit"]
    step_valid = aux["v523_step_candidate_valid"].bool()
    step_soft_route = aux["v523_step_soft_route"]
    step_hard_edit = aux["v523_step_hard_edit_mask"] > 0.5
    step_edit_prob = aux["v523_step_edit_prob"]
    step_top_gap = aux["v523_step_top_gap"]
    step_selected_risk = aux["v523_step_selected_risk"]
    step_selected_benefit = aux["v523_step_selected_benefit"]
    step_selected_coherence = aux["v523_step_selected_coherence"]
    step_edge = aux["v523_step_edge_map"].detach().clamp(0.0, 1.0)
    steps = int(step_current.shape[1])
    if steps < 1:
        raise RuntimeError("V523 must expose at least one sequential step")

    teacher_temperature = max(float(_m1(cfg, "V523_TEACHER_TEMPERATURE", 0.10)), 1.0e-3)
    student_temperature = max(float(_m1(cfg, "V523_STUDENT_TEMPERATURE", 0.20)), 1.0e-3)
    soft_tiebreak_weight = float(_m1(cfg, "V523_SOFT_UTILITY_WEIGHT", 0.10))
    utility_clip = max(float(_m1(cfg, "V523_UTILITY_CLIP", 2.0)), 1.0e-3)
    rank_margin = float(_m1(cfg, "V523_RANK_MARGIN", 0.50))
    floor = -1.0e4 if nonbase.dtype in (torch.float16, torch.bfloat16) else -1.0e9

    listwise_losses = []
    route_outcome_losses = []
    harmful_route_losses = []
    benefit_losses = []
    harm_losses = []
    rank_losses = []
    utility_losses = []
    preserve_losses = []
    coherence_losses = []

    teacher_correctable_rates = []
    teacher_positive_candidate_rates = []
    route_top1_accuracies = []
    repair_route_accuracies = []
    preserve_route_accuracies = []
    route_positive_precisions = []
    route_positive_recalls = []
    selected_benefit_rates = []
    selected_harm_rates = []
    edit_rates = []
    top_gap_values = []
    selected_risk_values = []
    selected_benefit_confidences = []
    selected_coherence_values = []
    teacher_oracle_probs = []

    for step_id in range(steps):
        current = step_current[:, step_id].detach().clamp(EPS, 1.0 - EPS)
        current_bank = current.expand(-1, n, -1, -1)
        valid = step_valid[:, step_id]
        valid_union = valid.any(dim=1, keepdim=True)

        current_hard = current >= 0.5
        candidate_hard = nonbase >= 0.5
        gt_hard = gt >= 0.5
        changed_hard = candidate_hard != current_hard.expand_as(candidate_hard)
        # Contract check: V523.2 deployment domain contains only real hard edits.
        if bool((valid & (~changed_hard)).any().item()):
            raise RuntimeError("V523 hard-change-only contract violated")

        current_wrong = current_hard != gt_hard
        repair = valid & current_wrong.expand_as(valid) & (candidate_hard == gt_hard.expand_as(valid))
        # With binary labels and a real hard flip, every non-repair is harmful.
        harm = valid & (~repair)
        correctable = repair.any(dim=1, keepdim=True)
        preserve_required = valid_union & (~correctable)

        current_bce = F.binary_cross_entropy(current_bank, gt_bank, reduction="none")
        candidate_bce = F.binary_cross_entropy(nonbase, gt_bank, reduction="none")
        soft_improvement = (current_bce - candidate_bce).clamp(-utility_clip, utility_clip)
        teacher_utility = torch.where(
            repair,
            1.0 + soft_tiebreak_weight * soft_improvement.clamp_min(0.0),
            -1.0 - soft_tiebreak_weight * (-soft_improvement).clamp_min(0.0),
        ).detach().clamp(-utility_clip, utility_clip)
        teacher_utility = teacher_utility.masked_fill(~valid, floor)

        teacher_action_score = torch.cat([torch.zeros_like(current), teacher_utility], dim=1)
        teacher_prob = F.softmax(teacher_action_score / teacher_temperature, dim=1)
        student_action_score = step_action_score[:, step_id]
        student_log_prob = F.log_softmax(student_action_score / student_temperature, dim=1)
        teacher_log_prob = torch.log(teacher_prob.clamp_min(EPS))
        kl_per_pixel = (teacher_prob * (teacher_log_prob - student_log_prob)).sum(dim=1)
        listwise_losses.append(
            _v523_balanced_mean(
                kl_per_pixel,
                [correctable[:, 0], preserve_required[:, 0]],
            )
        )

        candidate_route = step_soft_route[:, step_id, 1:]
        preserve_route = step_soft_route[:, step_id, :1]
        repair_mass = (candidate_route * repair.to(candidate_route.dtype)).sum(dim=1, keepdim=True)
        harm_mass = (candidate_route * harm.to(candidate_route.dtype)).sum(dim=1, keepdim=True)
        route_error = torch.zeros_like(preserve_route)
        route_error = torch.where(
            correctable,
            -torch.log(repair_mass.clamp_min(EPS)),
            route_error,
        )
        route_error = torch.where(
            preserve_required,
            -torch.log(preserve_route.clamp_min(EPS)),
            route_error,
        )
        route_outcome_losses.append(
            _v523_balanced_mean(
                route_error[:, 0],
                [correctable[:, 0], preserve_required[:, 0]],
            )
        )
        harmful_route_losses.append(
            _v523_balanced_mean(
                harm_mass[:, 0],
                [correctable[:, 0], preserve_required[:, 0]],
            )
        )

        benefit_logit = step_benefit_logit[:, step_id]
        benefit_bce = F.binary_cross_entropy_with_logits(
            benefit_logit, repair.to(benefit_logit.dtype), reduction="none"
        )
        benefit_losses.append(_v523_balanced_mean(benefit_bce, [repair, harm]))

        harm_logit = step_harm_logit[:, step_id]
        harm_bce = F.binary_cross_entropy_with_logits(
            harm_logit, harm.to(harm_logit.dtype), reduction="none"
        )
        harm_losses.append(_v523_balanced_mean(harm_bce, [harm, repair]))

        utility = step_utility[:, step_id]
        utility_error = F.smooth_l1_loss(
            utility,
            teacher_utility.clamp(-utility_clip, utility_clip),
            reduction="none",
            beta=0.10,
        )
        utility_losses.append(_v523_balanced_mean(utility_error, [repair, harm]))
        repair_rank = F.relu(rank_margin - utility)
        harm_rank = F.relu(rank_margin + utility)
        rank_losses.append(
            _v523_balanced_mean(
                torch.cat([repair_rank, harm_rank], dim=1),
                [
                    torch.cat([repair, torch.zeros_like(harm)], dim=1),
                    torch.cat([torch.zeros_like(repair), harm], dim=1),
                ],
            )
        )

        edit_prob = step_edit_prob[:, step_id]
        preserve_losses.append(
            edit_prob[preserve_required].mean()
            if bool(preserve_required.any().item())
            else edit_prob.sum() * 0.0
        )

        edge = step_edge[:, step_id]
        dx = (edit_prob[..., :, 1:] - edit_prob[..., :, :-1]).abs()
        dy = (edit_prob[..., 1:, :] - edit_prob[..., :-1, :]).abs()
        wx = 1.0 - torch.maximum(edge[..., :, 1:], edge[..., :, :-1])
        wy = 1.0 - torch.maximum(edge[..., 1:, :], edge[..., :-1, :])
        coherence_losses.append((dx * wx).mean() + (dy * wy).mean())

        with torch.no_grad():
            teacher_best = teacher_action_score.argmax(dim=1)
            student_best = student_action_score.argmax(dim=1)
            student_candidate = (student_best - 1).clamp_min(0)
            student_selected_repair = repair.gather(1, student_candidate[:, None])[:, 0]
            action_correct = (
                (correctable[:, 0] & student_selected_repair)
                | (preserve_required[:, 0] & (student_best == 0))
            )
            route_top1_accuracies.append(
                action_correct[valid_union[:, 0]].float().mean()
                if bool(valid_union.any().item()) else gt.new_zeros(())
            )
            repair_route_accuracies.append(
                student_selected_repair[correctable[:, 0]].float().mean()
                if bool(correctable.any().item()) else gt.new_zeros(())
            )
            preserve_route_accuracies.append(
                (student_best[preserve_required[:, 0]] == 0).float().mean()
                if bool(preserve_required.any().item()) else gt.new_zeros(())
            )

            predicted_edit = step_hard_edit[:, step_id]
            selected_index = aux["v523_step_selected_index"][:, step_id]
            bank_hard = torch.cat([current_hard, candidate_hard], dim=1)
            selected_hard = bank_hard.gather(1, selected_index[:, None])
            selected_changed = selected_hard != current_hard
            selected_benefit = selected_changed & current_wrong & (selected_hard == gt_hard)
            selected_harm = selected_changed & (~current_wrong) & (selected_hard != gt_hard)
            exact_true_positive = selected_benefit.float().sum()
            route_positive_precisions.append(
                exact_true_positive / predicted_edit.float().sum().clamp_min(1.0)
            )
            route_positive_recalls.append(
                exact_true_positive / correctable.float().sum().clamp_min(1.0)
            )
            selected_mass = selected_changed.float().sum().clamp_min(1.0)
            selected_benefit_rates.append(selected_benefit.float().sum() / selected_mass)
            selected_harm_rates.append(selected_harm.float().sum() / selected_mass)
            teacher_correctable_rates.append(
                correctable.float().sum() / valid_union.float().sum().clamp_min(1.0)
            )
            teacher_positive_candidate_rates.append(
                repair.float().sum() / valid.float().sum().clamp_min(1.0)
            )
            edit_rates.append(predicted_edit.float().mean())
            top_gap_values.append(step_top_gap[:, step_id].mean())
            selected_risk_values.append(step_selected_risk[:, step_id].mean())
            selected_benefit_confidences.append(step_selected_benefit[:, step_id].mean())
            selected_coherence_values.append(step_selected_coherence[:, step_id].mean())
            teacher_bank = torch.cat([current, nonbase], dim=1)
            teacher_oracle_probs.append(
                teacher_bank.gather(1, teacher_best[:, None]).clamp(EPS, 1.0 - EPS)
            )

    listwise_loss = torch.stack(listwise_losses).mean()
    route_outcome_loss = torch.stack(route_outcome_losses).mean()
    harmful_route_loss = torch.stack(harmful_route_losses).mean()
    benefit_loss = torch.stack(benefit_losses).mean()
    harm_loss = torch.stack(harm_losses).mean()
    rank_loss = torch.stack(rank_losses).mean()
    utility_loss = torch.stack(utility_losses).mean()
    preserve_loss = torch.stack(preserve_losses).mean()
    coherence_loss = torch.stack(coherence_losses).mean()
    uncorrectable_loss = preserve_loss * 0.0

    training_prob = _as_b1hw(aux["m2_training_probs"])
    deployed_prob = _as_b1hw(aux["m2_fused_probs"])
    deploy_seg_loss = _v519_prob_bce_dice(training_prob, gt)
    boundary_loss = (
        _soft_boundary(training_prob, radius=1) - _soft_boundary(gt, radius=1)
    ).abs().mean()

    base_soft_dice_case = _soft_dice_probs(c0, gt)[:, 0]
    final_soft_dice_case = _soft_dice_probs(training_prob, gt)[:, 0]
    case_regret = F.relu(
        base_soft_dice_case
        - final_soft_dice_case
        + float(_m1(cfg, "V523_CASE_NOHARM_MARGIN", 0.0))
    )
    case_noharm_loss = case_regret.mean()

    m2_loss = (
        float(_m1(cfg, "V523_LISTWISE_WEIGHT", 1.0)) * listwise_loss
        + float(_m1(cfg, "V523_ROUTE_OUTCOME_WEIGHT", 2.0)) * route_outcome_loss
        + float(_m1(cfg, "V523_HARM_ROUTE_WEIGHT", 2.0)) * harmful_route_loss
        + float(_m1(cfg, "V523_BENEFIT_WEIGHT", 2.0)) * benefit_loss
        + float(_m1(cfg, "V523_HARM_WEIGHT", 2.0)) * harm_loss
        + float(_m1(cfg, "V523_RANK_WEIGHT", 0.5)) * rank_loss
        + float(_m1(cfg, "V523_UTILITY_REG_WEIGHT", 0.25)) * utility_loss
        + float(_m1(cfg, "V523_PRESERVE_WEIGHT", 1.0)) * preserve_loss
        + float(_m1(cfg, "V523_COHERENCE_WEIGHT", 0.05)) * coherence_loss
        + float(_m1(cfg, "V523_DEPLOY_SEG_WEIGHT", 0.5)) * deploy_seg_loss
        + float(_m1(cfg, "V523_BOUNDARY_WEIGHT", 0.10)) * boundary_loss
        + float(_m1(cfg, "V523_CASE_NOHARM_WEIGHT", 1.0)) * case_noharm_loss
    )

    epoch_value = int(epoch or 0)
    m1_scale = _v501_delayed_ramp(
        epoch_value,
        int(_m1(cfg, "V523_M1_START_EPOCH", 0)),
        int(_m1(cfg, "V523_M1_RAMP_EPOCHS", 10)),
        float(_m1(cfg, "V523_M1_FINAL_WEIGHT", 1.0)),
    )
    m2_scale = _v501_delayed_ramp(
        epoch_value,
        int(_m1(cfg, "V523_M2_START_EPOCH", 0)),
        int(_m1(cfg, "V523_M2_RAMP_EPOCHS", 10)),
        float(_m1(cfg, "V523_M2_FINAL_WEIGHT", 1.0)),
    )
    total = m1_scale * m1_loss + m2_scale * m2_loss

    base_dice = base_soft_dice_case.mean()
    deployed_dice = _soft_dice_probs(deployed_prob, gt).mean()
    oracle_prob = teacher_oracle_probs[-1]
    oracle_dice = _soft_dice_probs(oracle_prob, gt).mean()
    oracle_gain = oracle_dice - base_dice
    deployed_gain = deployed_dice - base_dice
    capture_rate = torch.where(
        oracle_gain > 1.0e-6,
        deployed_gain / oracle_gain.clamp_min(1.0e-6),
        torch.zeros_like(deployed_gain),
    )

    base_hard = c0 >= 0.5
    deployed_hard = deployed_prob >= 0.5
    gt_hard = gt >= 0.5
    changed = deployed_hard != base_hard
    beneficial = changed & (base_hard != gt_hard) & (deployed_hard == gt_hard)
    harmful = changed & (base_hard == gt_hard) & (deployed_hard != gt_hard)
    changed_mass = changed.float().sum().clamp_min(1.0)
    zero = total * 0.0

    diag: Dict[str, torch.Tensor] = {
        **m1_diag,
        "v523_total_loss": total.detach(),
        "v523_m2_loss": m2_loss.detach(),
        "v523_dense_teacher_listwise_loss": listwise_loss.detach(),
        "v523_route_outcome_loss": route_outcome_loss.detach(),
        "v523_harmful_route_mass_loss": harmful_route_loss.detach(),
        "v523_benefit_classification_loss": benefit_loss.detach(),
        "v523_harm_classification_loss": harm_loss.detach(),
        "v523_preserve_rank_loss": rank_loss.detach(),
        "v523_utility_reg_loss": utility_loss.detach(),
        "v523_preserve_loss": preserve_loss.detach(),
        "v523_uncorrectable_loss": uncorrectable_loss.detach(),
        "v523_spatial_coherence_loss": coherence_loss.detach(),
        "v523_deploy_seg_loss": deploy_seg_loss.detach(),
        "v523_boundary_loss": boundary_loss.detach(),
        "v523_case_noharm_loss": case_noharm_loss.detach(),
        "v523_teacher_correctable_pixel_rate": torch.stack(teacher_correctable_rates).mean().detach(),
        "v523_teacher_positive_candidate_rate": torch.stack(teacher_positive_candidate_rates).mean().detach(),
        "v523_route_top1_accuracy": torch.stack(route_top1_accuracies).mean().detach(),
        "v523_repair_route_accuracy": torch.stack(repair_route_accuracies).mean().detach(),
        "v523_preserve_route_accuracy": torch.stack(preserve_route_accuracies).mean().detach(),
        "v523_beneficial_edit_precision": torch.stack(route_positive_precisions).mean().detach(),
        "v523_beneficial_edit_recall": torch.stack(route_positive_recalls).mean().detach(),
        "v523_selected_benefit_rate": torch.stack(selected_benefit_rates).mean().detach(),
        "v523_selected_harm_rate": torch.stack(selected_harm_rates).mean().detach(),
        "v523_hard_edit_pixel_rate": torch.stack(edit_rates).mean().detach(),
        "v523_mean_top_gap": torch.stack(top_gap_values).mean().detach(),
        "v523_mean_selected_risk": torch.stack(selected_risk_values).mean().detach(),
        "v523_mean_selected_benefit_confidence": torch.stack(selected_benefit_confidences).mean().detach(),
        "v523_mean_selected_coherence": torch.stack(selected_coherence_values).mean().detach(),
        "v523_base_dice": base_dice.detach(),
        "v523_deployed_dice": deployed_dice.detach(),
        "v523_deployed_gain": deployed_gain.detach(),
        "v523_teacher_pixel_oracle_dice": oracle_dice.detach(),
        "v523_teacher_pixel_oracle_gain": oracle_gain.detach(),
        "v523_pwo_capture_rate": capture_rate.detach(),
        "v523_changed_pixel_rate": changed.float().mean().detach(),
        "v523_conditional_benefit_rate": (beneficial.float().sum() / changed_mass).detach(),
        "v523_conditional_harm_rate": (harmful.float().sum() / changed_mass).detach(),
        "v523_worst_case_regret": case_regret.max().detach(),
        "v523_num_steps": aux["v523_num_steps"].float().mean().detach(),
        "v523_deploy_phase": aux["v523_deploy_phase"].float().mean().detach(),
        # Existing training/validation router compatibility.
        "v505_m1_loss": m1_loss.detach(),
        "v505_m2_loss": m2_loss.detach(),
        "v505_m3_loss": zero.detach(),
        "v505_m1_scale": total.new_tensor(m1_scale),
        "v505_m2_scale": total.new_tensor(m2_scale),
        "v505_m3_scale": zero.detach(),
        "v505_base_dice": base_dice.detach(),
        "v505_m2_dice": deployed_dice.detach(),
        "v505_final_dice": deployed_dice.detach(),
        "v505_m2_gain_vs_base": deployed_gain.detach(),
        "v505_final_gain_vs_base": deployed_gain.detach(),
        "v505_m2_changed_pixel_rate": changed.float().mean().detach(),
        "v505_m2_conditional_benefit_rate": (beneficial.float().sum() / changed_mass).detach(),
        "v505_m2_conditional_harm_rate": (harmful.float().sum() / changed_mass).detach(),
        "v505_selected_interaction_rate": torch.stack(edit_rates).mean().detach(),
        "v505_m3_accept_rate": zero.detach(),
        "v505_interactive_region_causal_enabled": total.new_tensor(1.0),
        "v489_total_loss": total.detach(),
        "v489_m1_loss": m1_loss.detach(),
        "v489_m2_loss": m2_loss.detach(),
        "v489_m3_loss": zero.detach(),
        "v489_base_dice": base_dice.detach(),
        "v489_m2_dice": deployed_dice.detach(),
        "v489_final_dice": deployed_dice.detach(),
        "v489_m2_gain_vs_base": deployed_gain.detach(),
        "v489_final_gain_vs_m2": zero.detach(),
        "v488_total_loss": total.detach(),
        "v488_base_dice": base_dice.detach(),
        "v488_m2_dice": deployed_dice.detach(),
        "v488_m3_final_dice": deployed_dice.detach(),
        "v488_m2_gain_vs_base": deployed_gain.detach(),
        "v488_m3_gain_vs_m2": zero.detach(),
        "v484_m2_loss": m2_loss.detach(),
        "v484_m3_regret_loss": zero.detach(),
        "_v490_m1_objective": m1_scale * m1_loss,
        "_v490_m2_objective": m2_scale * m2_loss,
        "_v490_m3_objective": zero,
    }
    return total, diag


def _compute_v522_pwo_distilled_sequential_loss(
    cfg: Any,
    gt: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    epoch: int | None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Joint V518-M1 + V522 PWO-distilled sequential local objective.

    PWO is used only as a privileged training teacher.  At each on-policy step
    the teacher marks a pixel correctable when the current deployed state is
    wrong and at least one *deployable, differing* M1 candidate is correct.
    Candidate attribution is multi-positive: every correcting candidate is a
    valid source, so equivalent candidates are never turned into false
    negatives.
    """
    gt = (_as_b1hw(gt) >= 0.5).float()
    candidate_probs = aux.get("candidate_probs")
    if not isinstance(candidate_probs, torch.Tensor) or candidate_probs.ndim != 4:
        raise RuntimeError("V522 requires aux['candidate_probs'] [B,K,H,W]")
    required = (
        "v522_step_current_before",
        "v522_step_gate_logit",
        "v522_step_gate_prob",
        "v522_step_source_logits",
        "v522_step_source_hard",
        "v522_step_utility_mean",
        "v522_step_utility_logvar",
        "v522_step_candidate_valid",
        "v522_step_hard_edit_mask",
        "v522_step_roi_mask",
        "v522_step_selected_pixel_index",
        "v522_deploy_mask",
        "m2_training_probs",
        "m2_fused_probs",
    )
    missing = [key for key in required if key not in aux]
    if missing:
        raise RuntimeError("V522 loss missing outputs: " + str(missing))

    # M1 remains live through the validated factual/candidate-bank objective.
    # M2 observes detached candidates and therefore cannot reshape its teacher.
    m1_loss, raw_m1_diag = _compute_v520_m1_objective(cfg, gt, aux, epoch)
    m1_diag = dict(raw_m1_diag)
    for key, value in raw_m1_diag.items():
        if key.startswith("v520_m1_"):
            m1_diag["v522_m1_" + key[len("v520_m1_"):]] = value

    candidates = candidate_probs.detach().clamp(EPS, 1.0 - EPS)
    c0 = candidates[:, :1]
    nonbase = candidates[:, 1:]
    b, n, h, w = nonbase.shape
    gt_bank = gt.expand(-1, n, -1, -1)

    step_current = aux["v522_step_current_before"]
    step_gate_logit = aux["v522_step_gate_logit"]
    step_gate_prob = aux["v522_step_gate_prob"]
    step_source_logits = aux["v522_step_source_logits"]
    step_source_hard = aux["v522_step_source_hard"]
    step_utility_mean = aux["v522_step_utility_mean"]
    step_utility_logvar = aux["v522_step_utility_logvar"].clamp(-6.0, 4.0)
    step_valid = aux["v522_step_candidate_valid"].bool()
    step_hard_edit = aux["v522_step_hard_edit_mask"] > 0.5
    step_roi = aux["v522_step_roi_mask"] > 0.5
    steps = int(step_current.shape[1])
    if steps < 1:
        raise RuntimeError("V522 must expose at least one sequential step")

    gate_losses = []
    preserve_losses = []
    uncorrectable_losses = []
    source_losses = []
    utility_losses = []
    gate_precision_values = []
    gate_recall_values = []
    source_hit_values = []
    correctable_rates = []
    candidate_positive_rates = []
    roi_rates = []
    hard_edit_rates = []
    step_benefit_rates = []
    step_harm_rates = []

    focal_gamma = float(_m1(cfg, "V522_GATE_FOCAL_GAMMA", 2.0))
    gate_threshold = float(_m1(cfg, "V522_GATE_THRESHOLD", 0.55))
    utility_floor = -1.0e4 if nonbase.dtype in (torch.float16, torch.bfloat16) else -1.0e9

    for step_id in range(steps):
        current = step_current[:, step_id].detach().clamp(EPS, 1.0 - EPS)
        current_hard = current >= 0.5
        current_bank = current_hard.expand(-1, n, -1, -1)
        candidate_hard = nonbase >= 0.5
        changed_hard = candidate_hard != current_bank
        current_wrong = current_hard != (gt >= 0.5)
        candidate_correct = candidate_hard == (gt_bank >= 0.5)
        valid = step_valid[:, step_id]

        repair = valid & changed_hard & current_wrong.expand_as(valid) & candidate_correct
        harm = valid & changed_hard & (~current_wrong.expand_as(valid)) & (~candidate_correct)
        valid_union = valid.any(dim=1, keepdim=True)
        correctable = repair.any(dim=1, keepdim=True)
        preserve_pixel = valid_union & (~current_wrong)
        uncorrectable = valid_union & current_wrong & (~correctable)

        gate_logit = step_gate_logit[:, step_id]
        gate_prob = step_gate_prob[:, step_id]
        if bool(valid_union.any().item()):
            gate_losses.append(
                _v505_binary_focal_with_logits(
                    gate_logit[valid_union],
                    correctable.float()[valid_union],
                    gamma=focal_gamma,
                )
            )
        else:
            gate_losses.append(gate_logit.sum() * 0.0)
        preserve_losses.append(
            gate_prob[preserve_pixel].mean()
            if bool(preserve_pixel.any().item())
            else gate_prob.sum() * 0.0
        )
        uncorrectable_losses.append(
            gate_prob[uncorrectable].mean()
            if bool(uncorrectable.any().item())
            else gate_prob.sum() * 0.0
        )

        # Multi-positive source attribution: probability mass assigned to the
        # union of all correcting candidates, not to an arbitrary single ID.
        source_logits = step_source_logits[:, step_id]
        all_log_z = torch.logsumexp(source_logits, dim=1)
        positive_logits = source_logits.masked_fill(~repair, utility_floor)
        positive_log_z = torch.logsumexp(positive_logits, dim=1)
        source_per_pixel = -(positive_log_z - all_log_z)
        source_per_pixel = torch.nan_to_num(source_per_pixel, nan=0.0, posinf=0.0, neginf=0.0)
        source_losses.append(
            source_per_pixel[correctable[:, 0]].mean()
            if bool(correctable.any().item())
            else source_logits.sum() * 0.0
        )

        # Heteroscedastic utility target supplies dense ordering information.
        # Positive means moving from the current state to the candidate reduces
        # absolute segmentation error; negative means it increases error.
        soft_utility_target = (
            (current.expand(-1, n, -1, -1) - gt_bank).abs()
            - (nonbase - gt_bank).abs()
        ).detach().clamp(-1.0, 1.0)
        utility_mean = step_utility_mean[:, step_id]
        utility_logvar = step_utility_logvar[:, step_id]
        utility_nll = 0.5 * (
            utility_logvar
            + (soft_utility_target - utility_mean).pow(2)
            * torch.exp(-utility_logvar)
        )
        utility_losses.append(
            utility_nll[valid].mean()
            if bool(valid.any().item())
            else utility_mean.sum() * 0.0
        )

        with torch.no_grad():
            pred_gate = (gate_prob >= gate_threshold) & valid_union
            true_positive = (pred_gate & correctable).float().sum()
            gate_precision_values.append(
                true_positive / pred_gate.float().sum().clamp_min(1.0)
            )
            gate_recall_values.append(
                true_positive / correctable.float().sum().clamp_min(1.0)
            )
            source_hit = (step_source_hard[:, step_id].bool() & repair).any(dim=1, keepdim=True)
            source_hit_values.append(
                source_hit[correctable].float().mean()
                if bool(correctable.any().item())
                else gt.new_zeros(())
            )
            correctable_rates.append(
                correctable.float().sum() / valid_union.float().sum().clamp_min(1.0)
            )
            candidate_positive_rates.append(
                repair.float().sum() / valid.float().sum().clamp_min(1.0)
            )
            roi_rates.append(step_roi[:, step_id].float().mean())
            hard_edit = step_hard_edit[:, step_id]
            hard_edit_rates.append(hard_edit.float().mean())
            selected_candidate_hard = (
                step_source_hard[:, step_id] * candidate_hard.float()
            ).sum(dim=1, keepdim=True) >= 0.5
            selected_changed = hard_edit & (selected_candidate_hard != current_hard)
            selected_benefit = selected_changed & current_wrong & (selected_candidate_hard == (gt >= 0.5))
            selected_harm = selected_changed & (~current_wrong) & (selected_candidate_hard != (gt >= 0.5))
            selected_mass = selected_changed.float().sum().clamp_min(1.0)
            step_benefit_rates.append(selected_benefit.float().sum() / selected_mass)
            step_harm_rates.append(selected_harm.float().sum() / selected_mass)

    gate_loss = torch.stack(gate_losses).mean()
    preserve_loss = torch.stack(preserve_losses).mean()
    uncorrectable_loss = torch.stack(uncorrectable_losses).mean()
    source_loss = torch.stack(source_losses).mean()
    utility_loss = torch.stack(utility_losses).mean()

    training_prob = _as_b1hw(aux["m2_training_probs"])
    deployed_prob = _as_b1hw(aux["m2_fused_probs"])
    deploy_seg_loss = _v519_prob_bce_dice(training_prob, gt)
    boundary_loss = (
        _soft_boundary(training_prob, radius=1)
        - _soft_boundary(gt, radius=1)
    ).abs().mean()

    base_soft_dice_case = _soft_dice_probs(c0, gt)[:, 0]
    final_soft_dice_case = _soft_dice_probs(training_prob, gt)[:, 0]
    case_regret = F.relu(
        base_soft_dice_case
        - final_soft_dice_case
        + float(_m1(cfg, "V522_CASE_NOHARM_MARGIN", 0.0))
    )
    case_noharm_loss = case_regret.mean()
    tail_fraction = min(max(float(_m1(cfg, "V522_TAIL_FRACTION", 0.2)), 0.0), 1.0)
    tail_count = max(1, int(math.ceil(case_regret.numel() * tail_fraction)))
    tail_noharm_loss = case_regret.topk(tail_count).values.mean()

    m2_loss = (
        float(_m1(cfg, "V522_GATE_WEIGHT", 1.0)) * gate_loss
        + float(_m1(cfg, "V522_PRESERVE_WEIGHT", 2.0)) * preserve_loss
        + float(_m1(cfg, "V522_UNCORRECTABLE_WEIGHT", 1.0)) * uncorrectable_loss
        + float(_m1(cfg, "V522_SOURCE_WEIGHT", 1.0)) * source_loss
        + float(_m1(cfg, "V522_UTILITY_WEIGHT", 0.5)) * utility_loss
        + float(_m1(cfg, "V522_DEPLOY_SEG_WEIGHT", 1.0)) * deploy_seg_loss
        + float(_m1(cfg, "V522_BOUNDARY_WEIGHT", 0.25)) * boundary_loss
        + float(_m1(cfg, "V522_CASE_NOHARM_WEIGHT", 1.0)) * case_noharm_loss
        + float(_m1(cfg, "V522_TAIL_NOHARM_WEIGHT", 1.0)) * tail_noharm_loss
    )

    epoch_value = int(epoch or 0)
    m1_scale = _v501_delayed_ramp(
        epoch_value,
        int(_m1(cfg, "V522_M1_START_EPOCH", 0)),
        int(_m1(cfg, "V522_M1_RAMP_EPOCHS", 10)),
        float(_m1(cfg, "V522_M1_FINAL_WEIGHT", 1.0)),
    )
    m2_scale = _v501_delayed_ramp(
        epoch_value,
        int(_m1(cfg, "V522_M2_START_EPOCH", 0)),
        int(_m1(cfg, "V522_M2_RAMP_EPOCHS", 10)),
        float(_m1(cfg, "V522_M2_FINAL_WEIGHT", 1.0)),
    )
    total = m1_scale * m1_loss + m2_scale * m2_loss

    base_dice = base_soft_dice_case.mean()
    deployed_dice = _soft_dice_probs(deployed_prob, gt).mean()
    base_hard = c0 >= 0.5
    deployed_hard = deployed_prob >= 0.5
    gt_hard = gt >= 0.5
    changed = deployed_hard != base_hard
    beneficial = changed & (base_hard != gt_hard) & (deployed_hard == gt_hard)
    harmful = changed & (base_hard == gt_hard) & (deployed_hard != gt_hard)
    changed_mass = changed.float().sum().clamp_min(1.0)
    zero = total * 0.0

    diag: Dict[str, torch.Tensor] = {
        **m1_diag,
        "v522_total_loss": total.detach(),
        "v522_m2_loss": m2_loss.detach(),
        "v522_gate_loss": gate_loss.detach(),
        "v522_preserve_loss": preserve_loss.detach(),
        "v522_uncorrectable_loss": uncorrectable_loss.detach(),
        "v522_multi_positive_source_loss": source_loss.detach(),
        "v522_utility_nll": utility_loss.detach(),
        "v522_deploy_seg_loss": deploy_seg_loss.detach(),
        "v522_boundary_loss": boundary_loss.detach(),
        "v522_case_noharm_loss": case_noharm_loss.detach(),
        "v522_tail_noharm_loss": tail_noharm_loss.detach(),
        "v522_gate_precision": torch.stack(gate_precision_values).mean().detach(),
        "v522_gate_recall": torch.stack(gate_recall_values).mean().detach(),
        "v522_source_multi_positive_hit_rate": torch.stack(source_hit_values).mean().detach(),
        "v522_teacher_correctable_pixel_rate": torch.stack(correctable_rates).mean().detach(),
        "v522_teacher_positive_candidate_rate": torch.stack(candidate_positive_rates).mean().detach(),
        "v522_dynamic_roi_pixel_rate": torch.stack(roi_rates).mean().detach(),
        "v522_hard_edit_pixel_rate": torch.stack(hard_edit_rates).mean().detach(),
        "v522_step_conditional_benefit_rate": torch.stack(step_benefit_rates).mean().detach(),
        "v522_step_conditional_harm_rate": torch.stack(step_harm_rates).mean().detach(),
        "v522_base_dice": base_dice.detach(),
        "v522_deployed_dice": deployed_dice.detach(),
        "v522_deployed_gain": (deployed_dice - base_dice).detach(),
        "v522_changed_pixel_rate": changed.float().mean().detach(),
        "v522_conditional_benefit_rate": (beneficial.float().sum() / changed_mass).detach(),
        "v522_conditional_harm_rate": (harmful.float().sum() / changed_mass).detach(),
        "v522_worst_case_regret": case_regret.max().detach(),
        "v522_tail_mean_regret": tail_noharm_loss.detach(),
        "v522_num_steps": aux["v522_num_steps"].float().mean().detach(),
        "v522_deploy_phase": aux["v522_deploy_phase"].float().mean().detach(),
        # Existing training/validation router compatibility.
        "v505_m1_loss": m1_loss.detach(),
        "v505_m2_loss": m2_loss.detach(),
        "v505_m3_loss": zero.detach(),
        "v505_m1_scale": total.new_tensor(m1_scale),
        "v505_m2_scale": total.new_tensor(m2_scale),
        "v505_m3_scale": zero.detach(),
        "v505_base_dice": base_dice.detach(),
        "v505_m2_dice": deployed_dice.detach(),
        "v505_final_dice": deployed_dice.detach(),
        "v505_m2_gain_vs_base": (deployed_dice - base_dice).detach(),
        "v505_final_gain_vs_base": (deployed_dice - base_dice).detach(),
        "v505_m2_changed_pixel_rate": changed.float().mean().detach(),
        "v505_m2_conditional_benefit_rate": (beneficial.float().sum() / changed_mass).detach(),
        "v505_m2_conditional_harm_rate": (harmful.float().sum() / changed_mass).detach(),
        "v505_selected_interaction_rate": torch.stack(hard_edit_rates).mean().detach(),
        "v505_m3_accept_rate": zero.detach(),
        "v505_interactive_region_causal_enabled": total.new_tensor(1.0),
        "v489_total_loss": total.detach(),
        "v489_m1_loss": m1_loss.detach(),
        "v489_m2_loss": m2_loss.detach(),
        "v489_m3_loss": zero.detach(),
        "v489_base_dice": base_dice.detach(),
        "v489_m2_dice": deployed_dice.detach(),
        "v489_final_dice": deployed_dice.detach(),
        "v489_m2_gain_vs_base": (deployed_dice - base_dice).detach(),
        "v489_final_gain_vs_m2": zero.detach(),
        "v488_total_loss": total.detach(),
        "v488_base_dice": base_dice.detach(),
        "v488_m2_dice": deployed_dice.detach(),
        "v488_m3_final_dice": deployed_dice.detach(),
        "v488_m2_gain_vs_base": (deployed_dice - base_dice).detach(),
        "v488_m3_gain_vs_m2": zero.detach(),
        "v484_m2_loss": m2_loss.detach(),
        "v484_m3_regret_loss": zero.detach(),
        "_v490_m1_objective": m1_scale * m1_loss,
        "_v490_m2_objective": m2_scale * m2_loss,
        "_v490_m3_objective": zero,
    }
    return total, diag


def _compute_v521_candidate_conditional_region_loss(
    cfg: Any,
    gt: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    epoch: int | None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Joint V518-M1 + V521 candidate-conditional region objective."""
    gt = (_as_b1hw(gt) >= 0.5).float()
    candidate_probs = aux.get("candidate_probs")
    if not isinstance(candidate_probs, torch.Tensor) or candidate_probs.ndim != 4:
        raise RuntimeError("V521 requires aux['candidate_probs'] [B,K,H,W]")
    required = (
        "v521_region_masks",
        "v521_region_active",
        "v521_region_candidate_valid",
        "v521_utility_pred",
        "v521_harm_logit",
        "v521_validity_logit",
        "v521_nonbase_score",
        "v521_score_all",
        "v521_soft_route_weights",
        "v521_hard_route_weights",
        "v521_route_weights",
        "v521_selected_region_index",
        "v521_candidate_edit_fraction",
        "v521_deploy_mask",
        "m2_training_probs",
        "m2_fused_probs",
    )
    missing = [key for key in required if key not in aux]
    if missing:
        raise RuntimeError("V521 loss missing outputs: " + str(missing))

    # M1 stays live only through its own factual/candidate-bank objective.  M2
    # teacher outcomes are computed from detached candidate observations.
    m1_loss, raw_m1_diag = _compute_v520_m1_objective(cfg, gt, aux, epoch)
    m1_diag = dict(raw_m1_diag)
    for key, value in raw_m1_diag.items():
        if key.startswith("v520_m1_"):
            m1_diag["v521_m1_" + key[len("v520_m1_"):]] = value

    candidates = candidate_probs.detach().clamp(EPS, 1.0 - EPS)
    c0 = candidates[:, :1]
    nonbase = candidates[:, 1:]
    b, n, h, w = nonbase.shape
    region_masks = aux["v521_region_masks"].detach().clamp(0.0, 1.0)
    region_active = aux["v521_region_active"].bool()
    r = region_masks.shape[1]
    valid = aux["v521_region_candidate_valid"].bool()
    deploy_mask = aux["v521_deploy_mask"][1:].bool()
    valid = valid & deploy_mask[None, None, :]

    base_hard = (c0 >= 0.5).float()
    candidate_hard = (nonbase >= 0.5).float()
    gt_hard = (gt >= 0.5).float()
    gt_bank = gt_hard.expand(-1, n, -1, -1)
    base_bank = base_hard.expand_as(candidate_hard)

    # Exact image-level hard Dice change after applying one candidate inside one
    # region.  This is the deployed intervention contract, not a pixel proxy.
    base_inter_global = (base_hard * gt_hard).flatten(1).sum(dim=1)
    base_pred_global = base_hard.flatten(1).sum(dim=1)
    gt_global = gt_hard.flatten(1).sum(dim=1)
    base_dice_case = (
        2.0 * base_inter_global + EPS
    ) / (base_pred_global + gt_global + EPS)

    base_inter_inside = torch.einsum(
        "bchw,brhw->br", base_hard * gt_hard, region_masks
    )
    base_pred_inside = torch.einsum("bchw,brhw->br", base_hard, region_masks)
    cand_inter_inside = _v521_region_sum(candidate_hard * gt_bank, region_masks)
    cand_pred_inside = _v521_region_sum(candidate_hard, region_masks)
    new_inter = (
        base_inter_global[:, None, None]
        - base_inter_inside[:, :, None]
        + cand_inter_inside
    )
    new_pred = (
        base_pred_global[:, None, None]
        - base_pred_inside[:, :, None]
        + cand_pred_inside
    )
    candidate_dice = (2.0 * new_inter + EPS) / (
        new_pred + gt_global[:, None, None] + EPS
    )
    delta_dice = candidate_dice - base_dice_case[:, None, None]

    base_boundary = _soft_boundary(base_hard, radius=1).detach()
    gt_boundary = _soft_boundary(gt_hard, radius=1).detach()
    candidate_boundary = _soft_boundary(
        candidate_hard.reshape(b * n, 1, h, w), radius=1
    ).reshape(b, n, h, w).detach()
    base_boundary_error = (base_boundary - gt_boundary).abs()
    candidate_boundary_error = (
        candidate_boundary - gt_boundary.expand_as(candidate_boundary)
    ).abs()
    region_area = region_masks.flatten(2).sum(dim=2).clamp_min(1.0)
    base_boundary_region = torch.einsum(
        "bchw,brhw->br", base_boundary_error, region_masks
    ) / region_area
    candidate_boundary_region = _v521_region_sum(
        candidate_boundary_error, region_masks
    ) / region_area[:, :, None]
    delta_boundary = base_boundary_region[:, :, None] - candidate_boundary_region

    changed = candidate_hard != base_bank
    base_wrong = base_bank != gt_bank
    candidate_correct = candidate_hard == gt_bank
    benefit_pixel = changed & base_wrong & candidate_correct
    harm_pixel = changed & (~base_wrong) & (~candidate_correct)
    changed_count = _v521_region_sum(changed.float(), region_masks)
    benefit_count = _v521_region_sum(benefit_pixel.float(), region_masks)
    harm_count = _v521_region_sum(harm_pixel.float(), region_masks)
    benefit_fraction = benefit_count / changed_count.clamp_min(1.0)
    harm_fraction = harm_count / changed_count.clamp_min(1.0)
    edit_fraction = changed_count / region_area[:, :, None]

    teacher_utility = (
        float(_m1(cfg, "V521_TEACHER_DICE_WEIGHT", 1.0)) * delta_dice
        + float(_m1(cfg, "V521_TEACHER_SURFACE_WEIGHT", 0.25)) * delta_boundary
        + float(_m1(cfg, "V521_TEACHER_BENEFIT_WEIGHT", 0.25)) * benefit_fraction
        - float(_m1(cfg, "V521_TEACHER_HARM_WEIGHT", 0.75)) * harm_fraction
        - float(_m1(cfg, "V521_TEACHER_EDIT_WEIGHT", 0.02)) * edit_fraction
    ).detach()
    teacher_utility = teacher_utility.masked_fill(~valid, -1.0)
    positive_margin = float(_m1(cfg, "V521_POSITIVE_UTILITY_MARGIN", 0.002))
    harmful_margin = float(_m1(cfg, "V521_HARMFUL_UTILITY_MARGIN", 0.002))
    beneficial = valid & (teacher_utility >= positive_margin)
    harmful = valid & (
        (teacher_utility <= -harmful_margin)
        | (harm_fraction > benefit_fraction)
    )

    utility_pred = aux["v521_utility_pred"]
    nonbase_score = aux["v521_nonbase_score"]
    harm_logit = aux["v521_harm_logit"]
    validity_logit = aux["v521_validity_logit"]
    valid_float = valid.float()
    valid_den = valid_float.sum().clamp_min(1.0)
    utility_loss = (
        F.smooth_l1_loss(
            utility_pred,
            teacher_utility.clamp(-1.0, 1.0),
            reduction="none",
            beta=float(_m1(cfg, "V521_UTILITY_BETA", 0.02)),
        ) * valid_float
    ).sum() / valid_den
    harm_loss = _v520_balanced_bce_with_logits(
        harm_logit, harmful.float(), valid=valid
    )
    validity_loss = _v520_balanced_bce_with_logits(
        validity_logit, beneficial.float(), valid=valid
    )

    teacher_full = torch.cat(
        [torch.zeros((b, r, 1), device=gt.device, dtype=gt.dtype), teacher_utility],
        dim=2,
    )
    full_valid = torch.cat(
        [torch.ones((b, r, 1), device=gt.device, dtype=torch.bool), valid],
        dim=2,
    )
    floor = -1.0e4 if teacher_full.dtype in (torch.float16, torch.bfloat16) else -1.0e9
    teacher_temperature = max(
        float(_m1(cfg, "V521_TEACHER_TEMPERATURE", 0.02)), 1.0e-3
    )
    student_temperature = max(
        float(_m1(cfg, "V521_STUDENT_TEMPERATURE", 0.10)), 1.0e-3
    )
    teacher_logits = (teacher_full / teacher_temperature).masked_fill(~full_valid, floor)
    student_logits = (aux["v521_score_all"] / student_temperature).masked_fill(~full_valid, floor)
    teacher_dist = F.softmax(teacher_logits, dim=2).detach()
    student_log_dist = F.log_softmax(student_logits, dim=2)
    listwise_per_region = (
        teacher_dist
        * (torch.log(teacher_dist.clamp_min(EPS)) - student_log_dist)
    ).sum(dim=2)
    listwise_loss = _v521_safe_mean(listwise_per_region, region_active)

    preserve_rank_margin = float(_m1(cfg, "V521_PRESERVE_RANK_MARGIN", 0.002))
    positive_rank_loss = _v521_safe_mean(
        F.relu(preserve_rank_margin - nonbase_score), beneficial
    )
    harmful_rank_loss = _v521_safe_mean(
        F.relu(preserve_rank_margin + nonbase_score), harmful
    )
    preserve_ranking_loss = positive_rank_loss + harmful_rank_loss

    selected_index = aux["v521_selected_region_index"].long()
    selected_teacher_utility = teacher_full.gather(
        2, selected_index[..., None]
    )[..., 0]
    best_teacher_utility, best_teacher_index = teacher_full.masked_fill(
        ~full_valid, floor
    ).max(dim=2)
    selected_edit = selected_index > 0
    epoch_value = int(epoch or 0)
    deploy_start = int(_m1(cfg, "V521_M2_DEPLOY_START_EPOCH", 5))
    deploy_enabled = epoch_value >= deploy_start
    deploy_region_mask = region_active & selected_edit
    if deploy_enabled:
        selected_regret_loss = _v521_safe_mean(
            F.relu(
                best_teacher_utility
                - selected_teacher_utility
                - float(_m1(cfg, "V521_REGRET_TOLERANCE", 0.001))
            ),
            region_active,
        )
        selected_harm_loss = _v521_safe_mean(
            F.relu(
                float(_m1(cfg, "V521_SELECTED_HARM_MARGIN", 0.002))
                - selected_teacher_utility
            ),
            deploy_region_mask,
        )
        no_positive = best_teacher_utility <= positive_margin
        false_edit_loss = _v521_safe_mean(
            F.relu(
                aux["v521_selected_region_score"]
                + float(_m1(cfg, "V521_FALSE_EDIT_MARGIN", 0.002))
            ),
            deploy_region_mask & no_positive,
        )
    else:
        zero_graph = nonbase_score.sum() * 0.0
        selected_regret_loss = zero_graph
        selected_harm_loss = zero_graph
        false_edit_loss = zero_graph

    training_prob = _as_b1hw(aux["m2_training_probs"])
    deployed_prob = _as_b1hw(aux["m2_fused_probs"])
    deploy_seg_loss = _v519_prob_bce_dice(training_prob, gt)
    base_soft_dice_case = _soft_dice_probs(c0, gt)[:, 0]
    final_soft_dice_case = _soft_dice_probs(training_prob, gt)[:, 0]
    case_regret = F.relu(
        base_soft_dice_case
        - final_soft_dice_case
        + float(_m1(cfg, "V521_CASE_NOHARM_MARGIN", 0.0))
    )
    case_noharm_loss = case_regret.mean()
    tail_fraction = min(max(float(_m1(cfg, "V521_TAIL_FRACTION", 0.2)), 0.0), 1.0)
    tail_count = max(1, int(math.ceil(case_regret.numel() * tail_fraction)))
    tail_noharm_loss = case_regret.topk(tail_count).values.mean()

    m2_loss = (
        float(_m1(cfg, "V521_UTILITY_WEIGHT", 1.0)) * utility_loss
        + float(_m1(cfg, "V521_LISTWISE_WEIGHT", 1.0)) * listwise_loss
        + float(_m1(cfg, "V521_PRESERVE_RANK_WEIGHT", 1.0)) * preserve_ranking_loss
        + float(_m1(cfg, "V521_HARM_WEIGHT", 1.0)) * harm_loss
        + float(_m1(cfg, "V521_VALIDITY_WEIGHT", 0.5)) * validity_loss
        + float(_m1(cfg, "V521_SELECTED_REGRET_WEIGHT", 2.0)) * selected_regret_loss
        + float(_m1(cfg, "V521_SELECTED_HARM_WEIGHT", 4.0)) * selected_harm_loss
        + float(_m1(cfg, "V521_FALSE_EDIT_WEIGHT", 1.0)) * false_edit_loss
        + float(_m1(cfg, "V521_DEPLOY_SEG_WEIGHT", 1.0)) * deploy_seg_loss
        + float(_m1(cfg, "V521_CASE_NOHARM_WEIGHT", 3.0)) * case_noharm_loss
        + float(_m1(cfg, "V521_TAIL_NOHARM_WEIGHT", 2.0)) * tail_noharm_loss
    )

    m1_scale = _v501_delayed_ramp(
        epoch_value,
        int(_m1(cfg, "V521_M1_START_EPOCH", 0)),
        int(_m1(cfg, "V521_M1_RAMP_EPOCHS", 10)),
        float(_m1(cfg, "V521_M1_FINAL_WEIGHT", 1.0)),
    )
    m2_scale = _v501_delayed_ramp(
        epoch_value,
        int(_m1(cfg, "V521_M2_UTILITY_START_EPOCH", 0)),
        int(_m1(cfg, "V521_M2_RAMP_EPOCHS", 10)),
        float(_m1(cfg, "V521_M2_FINAL_WEIGHT", 1.0)),
    )
    total = m1_scale * m1_loss + m2_scale * m2_loss

    base_dice = base_soft_dice_case.mean()
    deployed_dice = _soft_dice_probs(deployed_prob, gt).mean()
    base_hard_single = c0 >= 0.5
    deployed_hard = deployed_prob >= 0.5
    gt_hard_single = gt >= 0.5
    deployed_changed = deployed_hard != base_hard_single
    beneficial_change = (
        deployed_changed
        & (base_hard_single != gt_hard_single)
        & (deployed_hard == gt_hard_single)
    )
    harmful_change = (
        deployed_changed
        & (base_hard_single == gt_hard_single)
        & (deployed_hard != gt_hard_single)
    )
    changed_mass = deployed_changed.float().sum().clamp_min(1.0)
    selected_negative = deploy_region_mask & (selected_teacher_utility < 0.0)
    selected_positive = deploy_region_mask & (selected_teacher_utility > 0.0)
    harmful_rejected = harmful & (nonbase_score < 0.0)
    positive_selected = beneficial & (nonbase_score > 0.0)
    top3 = teacher_full.masked_fill(~full_valid, floor).topk(
        min(3, teacher_full.shape[2]), dim=2
    ).indices
    selected_in_top3 = (top3 == selected_index[..., None]).any(dim=2)
    expected_teacher_utility = (
        aux["v521_soft_route_weights"] * teacher_full
    ).sum(dim=2)
    zero = total * 0.0

    diag: Dict[str, torch.Tensor] = {
        **m1_diag,
        "v521_total_loss": total.detach(),
        "v521_m2_loss": m2_loss.detach(),
        "v521_utility_loss": utility_loss.detach(),
        "v521_listwise_loss": listwise_loss.detach(),
        "v521_preserve_ranking_loss": preserve_ranking_loss.detach(),
        "v521_positive_rank_loss": positive_rank_loss.detach(),
        "v521_harmful_rank_loss": harmful_rank_loss.detach(),
        "v521_harm_loss": harm_loss.detach(),
        "v521_validity_loss": validity_loss.detach(),
        "v521_selected_regret_loss": selected_regret_loss.detach(),
        "v521_selected_harm_loss": selected_harm_loss.detach(),
        "v521_false_edit_loss": false_edit_loss.detach(),
        "v521_deploy_seg_loss": deploy_seg_loss.detach(),
        "v521_case_noharm_loss": case_noharm_loss.detach(),
        "v521_tail_noharm_loss": tail_noharm_loss.detach(),
        "v521_teacher_positive_region_rate": (
            beneficial.any(dim=2) & region_active
        ).float().sum().detach() / region_active.float().sum().clamp_min(1.0),
        "v521_teacher_beneficial_candidate_rate": beneficial.float().sum().detach() / valid_float.sum().clamp_min(1.0),
        "v521_teacher_harmful_candidate_rate": harmful.float().sum().detach() / valid_float.sum().clamp_min(1.0),
        "v521_teacher_mean_best_utility": _v521_safe_mean(best_teacher_utility, region_active).detach(),
        "v521_expected_teacher_utility": _v521_safe_mean(expected_teacher_utility, region_active).detach(),
        "v521_selected_region_rate": deploy_region_mask.float().sum().detach() / region_active.float().sum().clamp_min(1.0),
        "v521_selected_mean_teacher_utility": _v521_safe_mean(selected_teacher_utility, deploy_region_mask).detach(),
        "v521_selected_negative_utility_rate": selected_negative.float().sum().detach() / deploy_region_mask.float().sum().clamp_min(1.0),
        "v521_selected_positive_utility_rate": selected_positive.float().sum().detach() / deploy_region_mask.float().sum().clamp_min(1.0),
        "v521_selected_mean_regret": _v521_safe_mean(best_teacher_utility - selected_teacher_utility, region_active).detach(),
        "v521_top1_within_teacher_top3_rate": _v521_safe_mean(selected_in_top3.float(), region_active).detach(),
        "v521_positive_candidate_recall": positive_selected.float().sum().detach() / beneficial.float().sum().clamp_min(1.0),
        "v521_harmful_candidate_rejection_rate": harmful_rejected.float().sum().detach() / harmful.float().sum().clamp_min(1.0),
        "v521_region_active_rate": region_active.float().mean().detach(),
        "v521_candidate_valid_rate": valid.float().mean().detach(),
        "v521_base_dice": base_dice.detach(),
        "v521_deployed_dice": deployed_dice.detach(),
        "v521_deployed_gain": (deployed_dice - base_dice).detach(),
        "v521_changed_pixel_rate": deployed_changed.float().mean().detach(),
        "v521_conditional_benefit_rate": (beneficial_change.float().sum() / changed_mass).detach(),
        "v521_conditional_harm_rate": (harmful_change.float().sum() / changed_mass).detach(),
        "v521_worst_case_regret": case_regret.max().detach(),
        "v521_tail_mean_regret": tail_noharm_loss.detach(),
        "v521_deploy_phase": aux["v521_deploy_phase"].float().mean().detach(),
        # Existing training router compatibility.
        "v505_m1_loss": m1_loss.detach(),
        "v505_m2_loss": m2_loss.detach(),
        "v505_m3_loss": zero.detach(),
        "v505_m1_scale": total.new_tensor(m1_scale),
        "v505_m2_scale": total.new_tensor(m2_scale),
        "v505_m3_scale": zero.detach(),
        "v505_base_dice": base_dice.detach(),
        "v505_m2_dice": deployed_dice.detach(),
        "v505_final_dice": deployed_dice.detach(),
        "v505_m2_gain_vs_base": (deployed_dice - base_dice).detach(),
        "v505_final_gain_vs_base": (deployed_dice - base_dice).detach(),
        "v505_m2_changed_pixel_rate": deployed_changed.float().mean().detach(),
        "v505_m2_conditional_benefit_rate": (beneficial_change.float().sum() / changed_mass).detach(),
        "v505_m2_conditional_harm_rate": (harmful_change.float().sum() / changed_mass).detach(),
        "v505_selected_interaction_rate": deploy_region_mask.float().mean().detach(),
        "v505_m3_accept_rate": zero.detach(),
        "v505_interactive_region_causal_enabled": total.new_tensor(1.0),
        "v489_total_loss": total.detach(),
        "v489_m1_loss": m1_loss.detach(),
        "v489_m2_loss": m2_loss.detach(),
        "v489_m3_loss": zero.detach(),
        "v489_base_dice": base_dice.detach(),
        "v489_m2_dice": deployed_dice.detach(),
        "v489_final_dice": deployed_dice.detach(),
        "v489_m2_gain_vs_base": (deployed_dice - base_dice).detach(),
        "v489_final_gain_vs_m2": zero.detach(),
        "v488_total_loss": total.detach(),
        "v488_base_dice": base_dice.detach(),
        "v488_m2_dice": deployed_dice.detach(),
        "v488_m3_final_dice": deployed_dice.detach(),
        "v488_m2_gain_vs_base": (deployed_dice - base_dice).detach(),
        "v488_m3_gain_vs_m2": zero.detach(),
        "v484_m2_loss": m2_loss.detach(),
        "v484_m3_regret_loss": zero.detach(),
        "_v490_m1_objective": m1_scale * m1_loss,
        "_v490_m2_objective": m2_scale * m2_loss,
        "_v490_m3_objective": zero,
    }
    return total, diag



def _v524_balanced_binary_bce(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Class-balanced BCE without dataset-specific positive weights."""
    target = target.to(dtype=logit.dtype)
    loss = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    positive = target > 0.5
    negative = ~positive
    parts = []
    if bool(positive.any().item()):
        parts.append(loss[positive].mean())
    if bool(negative.any().item()):
        parts.append(loss[negative].mean())
    if parts:
        return torch.stack(parts).mean()
    return loss.sum() * 0.0


def _v524_balanced_group_mean(
    value: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    parts = []
    for mask in (first.bool(), second.bool()):
        if bool(mask.any().item()):
            parts.append(value[mask].mean())
    if parts:
        return torch.stack(parts).mean()
    return value.sum() * 0.0


def _compute_v524_counterfactual_prompted_region_loss(
    cfg: Any,
    gt: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    epoch: int | None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Region-potential-outcome supervision for V524.

    GT is used only here.  Every candidate-region action is evaluated relative
    to the detached factual Base.  The teacher jointly supervises family
    activation, positive/negative/boundary prompts, region set routing,
    candidate utility/risk/uncertainty, selective rejection and the final
    candidate-constrained deployment tensor.
    """
    gt = (_as_b1hw(gt) >= 0.5).float()
    candidate_probs = aux.get("candidate_probs")
    if not isinstance(candidate_probs, torch.Tensor) or candidate_probs.ndim != 4:
        raise RuntimeError("V524 requires aux['candidate_probs'] [B,K,H,W]")
    required = (
        "v524_candidate_valid",
        "v524_dense_utility_mu",
        "v524_dense_logvar",
        "v524_dense_harm_logit",
        "v524_prompt_logits",
        "v524_family_logits",
        "v524_region_action_score",
        "v524_region_soft_route",
        "v524_region_selected_index",
        "v524_region_candidate_valid",
        "v524_region_utility_mu",
        "v524_region_logvar",
        "v524_region_harm_logit",
        "v524_region_prompt_support",
        "v524_refiner_gate_logit",
        "v524_selected_index_map",
        "v524_pre_guard_prob",
        "v524_case_gain_mu",
        "v524_case_logvar",
        "v524_case_harm_logit",
        "v524_case_accept",
        "v524_deploy_gate",
        "v524_family_ids",
        "m2_training_probs",
        "m2_fused_probs",
    )
    missing = [key for key in required if key not in aux]
    if missing:
        raise RuntimeError("V524 loss missing outputs: " + str(missing))

    m1_loss, raw_m1_diag = _compute_v520_m1_objective(cfg, gt, aux, epoch)
    m1_diag = dict(raw_m1_diag)
    for key, value in raw_m1_diag.items():
        if key.startswith("v520_m1_"):
            m1_diag["v524_m1_" + key[len("v520_m1_"):]] = value

    candidates = candidate_probs.detach().clamp(EPS, 1.0 - EPS)
    c0 = candidates[:, :1]
    nonbase = candidates[:, 1:]
    b, n, h, w = nonbase.shape
    valid = aux["v524_candidate_valid"].bool()
    if valid.shape != nonbase.shape:
        raise RuntimeError("V524 candidate-valid shape mismatch")

    base_hard = c0 >= 0.5
    candidate_hard = nonbase >= 0.5
    gt_hard = gt >= 0.5
    base_wrong = base_hard != gt_hard
    repair = valid & base_wrong.expand_as(valid) & (candidate_hard == gt_hard.expand_as(valid))
    harm = valid & (~repair)
    if bool((valid & (~((candidate_hard != base_hard.expand_as(candidate_hard))))).any().item()):
        raise RuntimeError("V524 candidate-realizable hard-change contract violated")

    # Pixel-level dense guidance teacher.
    pixel_teacher_utility = repair.to(c0.dtype) - harm.to(c0.dtype)
    dense_mu = aux["v524_dense_utility_mu"]
    dense_logvar = aux["v524_dense_logvar"].clamp(
        float(_m1(cfg, "V526_DENSE_LOGVAR_MIN", -6.0)),
        float(_m1(cfg, "V526_DENSE_LOGVAR_MAX", 3.0)),
    )
    dense_harm_logit = aux["v524_dense_harm_logit"]
    dense_utility_error = F.smooth_l1_loss(
        dense_mu, pixel_teacher_utility, reduction="none", beta=0.10
    )
    dense_utility_loss = _v523_balanced_mean(dense_utility_error, [repair, harm])
    dense_harm_loss = _v524_balanced_binary_bce(dense_harm_logit[valid], harm[valid]) if bool(valid.any().item()) else dense_harm_logit.sum() * 0.0
    dense_nll = 0.5 * torch.exp(-dense_logvar) * (pixel_teacher_utility - dense_mu).pow(2) + 0.5 * dense_logvar
    dense_nll_loss = _v523_balanced_mean(dense_nll, [repair, harm])

    # Counterfactual prompt supervision.  Positive prompts are supported 0->1
    # repairs, negative prompts are supported 1->0 repairs, and boundary prompts
    # focus those repairs around factual/GT boundaries.
    add_repair = repair & (~base_hard.expand_as(repair)) & candidate_hard
    remove_repair = repair & base_hard.expand_as(repair) & (~candidate_hard)
    add_target = add_repair.any(dim=1, keepdim=True).float()
    remove_target = remove_repair.any(dim=1, keepdim=True).float()
    boundary_band = torch.maximum(_soft_boundary(c0, radius=2), _soft_boundary(gt, radius=2)).detach()
    repair_union = repair.any(dim=1, keepdim=True).float()
    boundary_target = (repair_union * (boundary_band > 0.05).float()).clamp(0.0, 1.0)
    prompt_target = torch.cat([add_target, remove_target, boundary_target], dim=1)
    prompt_logits = aux["v524_prompt_logits"]
    prompt_loss = torch.stack([
        _v524_balanced_binary_bce(prompt_logits[:, index:index + 1], prompt_target[:, index:index + 1])
        for index in range(3)
    ]).mean()

    grid = int(round(float(aux["v524_region_grid_size"][0].detach().cpu())))
    if grid < 2 or grid * grid != aux["v524_region_action_score"].shape[1]:
        raise RuntimeError("V524 invalid region-grid contract")

    def pool_candidate(value: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(value.reshape(b * n, 1, h, w), (grid, grid))
        return pooled.reshape(b, n, grid, grid).permute(0, 2, 3, 1).reshape(b, grid * grid, n)

    valid_fraction = pool_candidate(valid.float())
    decision_region_valid = aux["v524_region_candidate_valid"].bool()
    geometric_region_valid = aux.get("v526_region_geometric_valid")
    if not isinstance(geometric_region_valid, torch.Tensor):
        geometric_region_valid = decision_region_valid
    geometric_region_valid = geometric_region_valid.bool()
    if bool(_m1(cfg, "V527_ORACLE_PRESERVING_VALIDITY", False)):
        region_valid = geometric_region_valid
    else:
        region_valid = decision_region_valid

    # Exact candidate-region potential outcome teacher.  Each action is the
    # complete valid support of one real candidate inside one region.  The
    # teacher evaluates the resulting hard mask with the same per-case Dice
    # geometry used by validation instead of a repair-minus-harm heuristic.
    region_ids = F.interpolate(
        torch.arange(grid * grid, device=c0.device, dtype=c0.dtype).reshape(1, 1, grid, grid),
        size=(h, w),
        mode="nearest",
    )[0, 0].long()
    base_tp = (base_hard & gt_hard).flatten(1).sum(dim=1).to(c0.dtype)
    base_fp = (base_hard & (~gt_hard)).flatten(1).sum(dim=1).to(c0.dtype)
    base_fn = ((~base_hard) & gt_hard).flatten(1).sum(dim=1).to(c0.dtype)
    base_dice_exact = (2.0 * base_tp + EPS) / (2.0 * base_tp + base_fp + base_fn + EPS)
    boundary_band = torch.maximum(_soft_boundary(c0, radius=2), _soft_boundary(gt, radius=2)).detach()
    boundary_den = boundary_band.flatten(1).sum(dim=1).clamp_min(1.0)
    exact_dice_gain = c0.new_zeros(b, grid * grid, n)
    exact_boundary_gain = c0.new_zeros(b, grid * grid, n)
    exact_fn_fix = c0.new_zeros(b, grid * grid, n)
    exact_fp_fix = c0.new_zeros(b, grid * grid, n)
    exact_tn_harm = c0.new_zeros(b, grid * grid, n)
    exact_tp_harm = c0.new_zeros(b, grid * grid, n)
    v528_outcome_contract_error = c0.new_zeros(())
    base_correct = base_hard == gt_hard
    candidate_correct = candidate_hard == gt_hard.expand_as(candidate_hard)
    for region_index in range(grid * grid):
        spatial = (region_ids == region_index)[None, None]
        action_mask = valid & spatial
        old_tp = (action_mask & base_hard.expand_as(action_mask) & gt_hard.expand_as(action_mask)).flatten(2).sum(dim=2).to(c0.dtype)
        old_fp = (action_mask & base_hard.expand_as(action_mask) & (~gt_hard.expand_as(action_mask))).flatten(2).sum(dim=2).to(c0.dtype)
        old_fn = (action_mask & (~base_hard.expand_as(action_mask)) & gt_hard.expand_as(action_mask)).flatten(2).sum(dim=2).to(c0.dtype)
        new_tp = (action_mask & candidate_hard & gt_hard.expand_as(action_mask)).flatten(2).sum(dim=2).to(c0.dtype)
        new_fp = (action_mask & candidate_hard & (~gt_hard.expand_as(action_mask))).flatten(2).sum(dim=2).to(c0.dtype)
        new_fn = (action_mask & (~candidate_hard) & gt_hard.expand_as(action_mask)).flatten(2).sum(dim=2).to(c0.dtype)
        tp = base_tp[:, None] + new_tp - old_tp
        fp = base_fp[:, None] + new_fp - old_fp
        fn = base_fn[:, None] + new_fn - old_fn
        action_dice = (2.0 * tp + EPS) / (2.0 * tp + fp + fn + EPS)
        exact_dice_gain[:, region_index] = action_dice - base_dice_exact[:, None]

        # V528 physically conserved outcome decomposition.
        fn_fix = (
            action_mask & (~base_hard.expand_as(action_mask))
            & candidate_hard & gt_hard.expand_as(action_mask)
        ).flatten(2).sum(dim=2).to(c0.dtype)
        tn_harm = (
            action_mask & (~base_hard.expand_as(action_mask))
            & candidate_hard & (~gt_hard.expand_as(action_mask))
        ).flatten(2).sum(dim=2).to(c0.dtype)
        fp_fix = (
            action_mask & base_hard.expand_as(action_mask)
            & (~candidate_hard) & (~gt_hard.expand_as(action_mask))
        ).flatten(2).sum(dim=2).to(c0.dtype)
        tp_harm = (
            action_mask & base_hard.expand_as(action_mask)
            & (~candidate_hard) & gt_hard.expand_as(action_mask)
        ).flatten(2).sum(dim=2).to(c0.dtype)
        image_pixels = float(max(h * w, 1))
        exact_fn_fix[:, region_index] = fn_fix / image_pixels
        exact_tn_harm[:, region_index] = tn_harm / image_pixels
        exact_fp_fix[:, region_index] = fp_fix / image_pixels
        exact_tp_harm[:, region_index] = tp_harm / image_pixels

        outcome_tp = base_tp[:, None] + fn_fix - tp_harm
        outcome_fp = base_fp[:, None] + tn_harm - fp_fix
        outcome_fn = base_fn[:, None] - fn_fix + tp_harm
        outcome_dice = (
            2.0 * outcome_tp + EPS
        ) / (2.0 * outcome_tp + outcome_fp + outcome_fn + EPS)
        v528_outcome_contract_error = torch.maximum(
            v528_outcome_contract_error,
            (outcome_dice - action_dice).abs().max(),
        )

        correctness_delta = (candidate_correct.to(c0.dtype) - base_correct.expand_as(candidate_correct).to(c0.dtype))
        boundary_delta = (correctness_delta * action_mask.to(c0.dtype) * boundary_band.expand_as(action_mask)).flatten(2).sum(dim=2)
        exact_boundary_gain[:, region_index] = boundary_delta / boundary_den[:, None]

    boundary_weight = float(_m1(cfg, "V524_EXACT_REGION_BOUNDARY_WEIGHT", 0.25))
    v528_enabled = bool(_m1(cfg, "V528_OUTCOME_COMPOSER_ENABLED", False))
    v529_enabled = bool(_m1(cfg, "V529_CALIBRATION_FIRST_OUTCOME_ENABLED", False))
    v530_enabled = bool(_m1(cfg, "V530_PROBABILITY_CALIBRATED_OUTCOME_ENABLED", False))
    raw_region_utility = (
        exact_dice_gain
        if (v528_enabled or v529_enabled or v530_enabled)
        else exact_dice_gain + boundary_weight * exact_boundary_gain
    ).detach()
    raw_region_utility = raw_region_utility.masked_fill(
        ~geometric_region_valid, 0.0
    )
    # Selector supervision uses the actually available action set after sparse
    # family routing / causal compatibility.  The raw utility is retained for
    # an unfiltered action-space oracle diagnostic.
    region_utility = raw_region_utility.masked_fill(~region_valid, 0.0)

    positive_margin = float(_m1(cfg, "V524_POSITIVE_UTILITY_MARGIN", 1.0e-4))
    negative_margin = float(_m1(cfg, "V524_NEGATIVE_UTILITY_MARGIN", 1.0e-4))
    positive = region_valid & (region_utility > positive_margin)
    harmful = region_valid & (region_utility < -negative_margin)
    repairable = positive.any(dim=2)
    harmful_present = harmful.any(dim=2)

    # Dense teacher family routing.  Families are scored by their best true
    # region action, so all families receive comparative supervision.
    family_logits = aux["v524_family_logits"]
    family_ids = aux["v524_family_ids"][1:].long().to(region_utility.device)
    family_count = family_logits.shape[1]
    floor = -1.0e4 if region_utility.dtype in (torch.float16, torch.bfloat16) else -1.0e9
    family_teacher_score = region_utility.new_full((b, family_count), floor)
    family_present = torch.zeros((family_count,), device=region_utility.device, dtype=torch.bool)
    for family_index in range(family_count):
        candidate_mask = family_ids == family_index
        if bool(candidate_mask.any().item()):
            family_present[family_index] = True
            values = region_utility[:, :, candidate_mask]
            valid_values = region_valid[:, :, candidate_mask]
            values = values.masked_fill(~valid_values, floor)
            family_teacher_score[:, family_index] = values.flatten(1).max(dim=1).values
    family_teacher_score = family_teacher_score.masked_fill(~family_present[None], floor)
    family_temperature = max(float(_m1(cfg, "V524_FAMILY_TEACHER_TEMPERATURE", 0.20)), 1.0e-3)
    family_teacher_prob = F.softmax(family_teacher_score / family_temperature, dim=1).detach()
    family_student_log = F.log_softmax(family_logits, dim=1)
    family_kl_case = (family_teacher_prob * (torch.log(family_teacher_prob.clamp_min(EPS)) - family_student_log)).sum(dim=1)
    correctable_case = repairable.any(dim=1)
    family_loss = family_kl_case[correctable_case].mean() if bool(correctable_case.any().item()) else family_kl_case.sum() * 0.0

    action_score = aux["v524_region_action_score"]
    route_temperature = max(float(_m1(cfg, "V524_ROUTE_TEMPERATURE", 0.50)), 1.0e-5)
    action_logits = action_score / route_temperature

    # ------------------------------------------------------------------
    # V526/V527 counterfactual selector objectives.
    # ------------------------------------------------------------------
    v526_enabled = bool(_m1(cfg, "V526_PREFERENCE_SELECTOR_ENABLED", False))
    v527_enabled = bool(_m1(cfg, "V527_FACTORIZED_RANKER_ENABLED", False))
    v528_enabled = bool(_m1(cfg, "V528_OUTCOME_COMPOSER_ENABLED", False))
    v529_enabled = bool(_m1(cfg, "V529_CALIBRATION_FIRST_OUTCOME_ENABLED", False))
    full_valid = torch.cat(
        [torch.ones_like(repairable[..., None]), region_valid], dim=2
    )
    full_utility = torch.cat(
        [torch.zeros_like(region_utility[..., :1]), region_utility], dim=2
    ).masked_fill(~full_valid, floor)

    best_candidate_utility, best_candidate_index = region_utility.masked_fill(
        ~region_valid, floor
    ).max(dim=2)
    oracle_edits = best_candidate_utility > positive_margin
    oracle_index = torch.where(
        oracle_edits, best_candidate_index + 1, torch.zeros_like(best_candidate_index)
    )
    oracle_utility = torch.where(
        oracle_edits, best_candidate_utility, torch.zeros_like(best_candidate_utility)
    )

    editability_loss = action_score.sum() * 0.0
    conditional_top1_loss = action_score.sum() * 0.0
    hard_pair_loss = action_score.sum() * 0.0
    preserve_noharm_loss = action_score.sum() * 0.0
    v528_base_confusion_loss = action_score.sum() * 0.0
    v528_outcome_correctness_loss = action_score.sum() * 0.0
    v528_outcome_mass_loss = action_score.sum() * 0.0
    v528_gain_loss = action_score.sum() * 0.0
    v528_gain_sign_loss = action_score.sum() * 0.0
    v528_expected_regret_loss = action_score.sum() * 0.0
    v528_harm_mass_loss = action_score.sum() * 0.0
    v528_hard_pair_loss = action_score.sum() * 0.0
    v528_outcome_nll_loss = action_score.sum() * 0.0
    v528_expected_utility_case = action_score.new_zeros(b)
    v528_oracle_utility_case = action_score.new_zeros(b)

    v529_base_error_loss = action_score.sum() * 0.0
    v529_outcome_pixel_loss = action_score.sum() * 0.0
    v529_outcome_mass_loss = action_score.sum() * 0.0
    v529_gain_loss = action_score.sum() * 0.0
    v529_gain_sign_loss = action_score.sum() * 0.0
    v529_expected_regret_loss = action_score.sum() * 0.0
    v529_harm_mass_loss = action_score.sum() * 0.0
    v529_hard_pair_loss = action_score.sum() * 0.0
    v529_outcome_nll_loss = action_score.sum() * 0.0
    v529_expected_utility_case = action_score.new_zeros(b)
    v529_oracle_utility_case = action_score.new_zeros(b)
    v529_base_dice_mae = action_score.sum() * 0.0
    v529_tp_mass_mae = action_score.sum() * 0.0
    v529_fp_mass_mae = action_score.sum() * 0.0
    v529_fn_mass_mae = action_score.sum() * 0.0
    v529_add_rate_mae = action_score.sum() * 0.0
    v529_remove_rate_mae = action_score.sum() * 0.0
    v529_gain_mae = action_score.sum() * 0.0
    v529_gain_sign_accuracy = action_score.sum() * 0.0
    v529_positive_precision = action_score.sum() * 0.0
    v529_positive_recall = action_score.sum() * 0.0
    v529_sigma_p50 = action_score.sum() * 0.0
    v529_sigma_p90 = action_score.sum() * 0.0
    v529_sigma_p99 = action_score.sum() * 0.0

    v530_base_pixel_loss = action_score.sum() * 0.0
    v530_base_brier_loss = action_score.sum() * 0.0
    v530_base_mass_loss = action_score.sum() * 0.0
    v530_outcome_pixel_loss = action_score.sum() * 0.0
    v530_outcome_brier_loss = action_score.sum() * 0.0
    v530_rate_loss = action_score.sum() * 0.0
    v530_outcome_mass_loss = action_score.sum() * 0.0
    v530_gain_loss = action_score.sum() * 0.0
    v530_gain_sign_loss = action_score.sum() * 0.0
    v530_expected_regret_loss = action_score.sum() * 0.0
    v530_harm_mass_loss = action_score.sum() * 0.0
    v530_hard_pair_loss = action_score.sum() * 0.0
    v530_outcome_nll_loss = action_score.sum() * 0.0
    v530_expected_utility_case = action_score.new_zeros(b)
    v530_oracle_utility_case = action_score.new_zeros(b)
    v530_base_dice_mae = action_score.sum() * 0.0
    v530_tp_mass_mae = action_score.sum() * 0.0
    v530_fp_mass_mae = action_score.sum() * 0.0
    v530_fn_mass_mae = action_score.sum() * 0.0
    v530_add_rate_mae = action_score.sum() * 0.0
    v530_remove_rate_mae = action_score.sum() * 0.0
    v530_gain_mae = action_score.sum() * 0.0
    v530_positive_gain_mae = action_score.sum() * 0.0
    v530_harmful_gain_mae = action_score.sum() * 0.0
    v530_neutral_gain_mae = action_score.sum() * 0.0
    v530_gain_sign_accuracy = action_score.sum() * 0.0
    v530_positive_precision = action_score.sum() * 0.0
    v530_positive_recall = action_score.sum() * 0.0
    v530_sigma_p50 = action_score.sum() * 0.0
    v530_sigma_p90 = action_score.sum() * 0.0
    v530_sigma_p99 = action_score.sum() * 0.0

    if v530_enabled:
        required_v530 = (
            "v530_base_error_pixel_logit",
            "v530_pred_base_tp",
            "v530_pred_base_fp",
            "v530_pred_base_fn",
            "v530_pred_base_dice",
            "v530_outcome_pixel_logit",
            "v530_outcome_logvar",
            "v530_pred_fn_fix",
            "v530_pred_fp_fix",
            "v530_pred_tn_harm",
            "v530_pred_tp_harm",
            "v530_predicted_gain",
            "v530_normalized_score",
            "v530_predicted_sigma",
            "v530_predicted_harm_fraction",
        )
        missing_v530 = [key for key in required_v530 if key not in aux]
        if missing_v530:
            raise RuntimeError("V530 loss missing outputs: " + str(missing_v530))

        image_pixels = float(max(h * w, 1))
        true_tp_fraction = base_tp / image_pixels
        true_fp_fraction = base_fp / image_pixels
        true_fn_fraction = base_fn / image_pixels
        true_base_dice = base_dice_exact

        # Calibrated factual error supervision.  No class rebalancing is used
        # because these sigmoid values are integrated as probabilities.
        base_logits = aux["v530_base_error_pixel_logit"]
        fp_domain = base_hard
        fn_domain = ~base_hard
        fp_target_pixel = (~gt_hard).to(base_logits.dtype)
        fn_target_pixel = gt_hard.to(base_logits.dtype)

        def case_domain_mean(
            value: torch.Tensor, domain: torch.Tensor
        ) -> torch.Tensor:
            domain_float = domain.to(value.dtype)
            numerator = (value * domain_float).flatten(1).sum(dim=1)
            denominator = domain_float.flatten(1).sum(dim=1)
            valid_case = denominator > 0
            if bool(valid_case.any().item()):
                return (
                    numerator[valid_case] / denominator[valid_case].clamp_min(1.0)
                ).mean()
            return value.sum() * 0.0

        fp_bce_map = F.binary_cross_entropy_with_logits(
            base_logits[:, 0:1], fp_target_pixel, reduction="none"
        )
        fn_bce_map = F.binary_cross_entropy_with_logits(
            base_logits[:, 1:2], fn_target_pixel, reduction="none"
        )
        v530_base_pixel_loss = 0.5 * (
            case_domain_mean(fp_bce_map, fp_domain)
            + case_domain_mean(fn_bce_map, fn_domain)
        )
        base_prob = torch.sigmoid(base_logits)
        v530_base_brier_loss = 0.5 * (
            case_domain_mean((base_prob[:, 0:1] - fp_target_pixel).pow(2), fp_domain)
            + case_domain_mean((base_prob[:, 1:2] - fn_target_pixel).pow(2), fn_domain)
        )
        base_mass_scale = max(
            float(_m1(cfg, "V530_BASE_MASS_SCALE", 0.01)), 1.0e-6
        )
        base_mass_pred = torch.stack(
            [aux["v530_pred_base_fp"], aux["v530_pred_base_fn"]], dim=1
        )
        base_mass_true = torch.stack(
            [true_fp_fraction, true_fn_fraction], dim=1
        )
        v530_base_mass_loss = F.smooth_l1_loss(
            base_mass_pred / base_mass_scale,
            base_mass_true.detach() / base_mass_scale,
            beta=max(float(_m1(cfg, "V530_BASE_MASS_HUBER_BETA", 0.10)), 1.0e-4),
        )

        add_full_teacher = (
            valid & (~base_hard.expand_as(candidate_hard)) & candidate_hard
        )
        remove_full_teacher = (
            valid & base_hard.expand_as(candidate_hard) & (~candidate_hard)
        )
        add_fix_target_pixel = gt_hard.expand_as(candidate_hard)
        remove_fix_target_pixel = (~gt_hard).expand_as(candidate_hard)
        pixel_logits = aux["v530_outcome_pixel_logit"]
        grid_size = int(round(float(region_valid.shape[1]) ** 0.5))
        if grid_size * grid_size != int(region_valid.shape[1]):
            raise RuntimeError("V530 requires a square region grid")
        region_ids_loss = F.interpolate(
            torch.arange(
                grid_size * grid_size, device=c0.device, dtype=c0.dtype
            ).reshape(1, 1, grid_size, grid_size),
            size=(h, w), mode="nearest",
        )[0, 0].long()

        def calibrated_region_edit_loss(
            logit: torch.Tensor, target: torch.Tensor, domain: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            raw_bce = F.binary_cross_entropy_with_logits(
                logit, target.to(logit.dtype), reduction="none"
            )
            raw_brier = (torch.sigmoid(logit) - target.to(logit.dtype)).pow(2)
            bce_parts = []
            brier_parts = []
            for region_index in range(grid_size * grid_size):
                spatial = (region_ids_loss == region_index)[None, None]
                mask = domain & spatial
                count = mask.flatten(2).sum(dim=2)
                valid_pair = count > 0
                if bool(valid_pair.any().item()):
                    mask_float = mask.to(logit.dtype)
                    bce_mean = (
                        raw_bce * mask_float
                    ).flatten(2).sum(dim=2) / count.clamp_min(1).to(logit.dtype)
                    brier_mean = (
                        raw_brier * mask_float
                    ).flatten(2).sum(dim=2) / count.clamp_min(1).to(logit.dtype)
                    bce_parts.append(bce_mean[valid_pair])
                    brier_parts.append(brier_mean[valid_pair])
            if not bce_parts:
                zero_local = logit.sum() * 0.0
                return zero_local, zero_local
            return torch.cat(bce_parts).mean(), torch.cat(brier_parts).mean()

        add_bce, add_brier = calibrated_region_edit_loss(
            pixel_logits[:, :, 0], add_fix_target_pixel, add_full_teacher
        )
        remove_bce, remove_brier = calibrated_region_edit_loss(
            pixel_logits[:, :, 1], remove_fix_target_pixel, remove_full_teacher
        )
        pixel_terms = []
        brier_terms = []
        if bool(add_full_teacher.any().item()):
            pixel_terms.append(add_bce)
            brier_terms.append(add_brier)
        if bool(remove_full_teacher.any().item()):
            pixel_terms.append(remove_bce)
            brier_terms.append(remove_brier)
        v530_outcome_pixel_loss = (
            torch.stack(pixel_terms).mean()
            if pixel_terms else pixel_logits.sum() * 0.0
        )
        v530_outcome_brier_loss = (
            torch.stack(brier_terms).mean()
            if brier_terms else pixel_logits.sum() * 0.0
        )

        exact_add = exact_fn_fix + exact_tn_harm
        exact_remove = exact_fp_fix + exact_tp_harm
        pred_add_rate = aux["v530_pred_fn_fix"] / (
            aux["v530_pred_fn_fix"] + aux["v530_pred_tn_harm"]
        ).clamp_min(EPS)
        pred_remove_rate = aux["v530_pred_fp_fix"] / (
            aux["v530_pred_fp_fix"] + aux["v530_pred_tp_harm"]
        ).clamp_min(EPS)
        true_add_rate = exact_fn_fix / exact_add.clamp_min(EPS)
        true_remove_rate = exact_fp_fix / exact_remove.clamp_min(EPS)
        add_domain = region_valid & (exact_add > 0.0)
        remove_domain = region_valid & (exact_remove > 0.0)
        rate_parts = []
        if bool(add_domain.any().item()):
            rate_parts.append(F.smooth_l1_loss(
                pred_add_rate[add_domain], true_add_rate[add_domain].detach(),
                beta=max(float(_m1(cfg, "V530_RATE_HUBER_BETA", 0.05)), 1.0e-4),
            ))
        if bool(remove_domain.any().item()):
            rate_parts.append(F.smooth_l1_loss(
                pred_remove_rate[remove_domain], true_remove_rate[remove_domain].detach(),
                beta=max(float(_m1(cfg, "V530_RATE_HUBER_BETA", 0.05)), 1.0e-4),
            ))
        v530_rate_loss = (
            torch.stack(rate_parts).mean()
            if rate_parts else pixel_logits.sum() * 0.0
        )

        true_outcome = torch.stack(
            [exact_fn_fix, exact_fp_fix, exact_tn_harm, exact_tp_harm], dim=-1
        )
        pred_outcome = torch.stack(
            [
                aux["v530_pred_fn_fix"], aux["v530_pred_fp_fix"],
                aux["v530_pred_tn_harm"], aux["v530_pred_tp_harm"],
            ], dim=-1,
        )
        outcome_scale = max(
            float(_m1(cfg, "V530_OUTCOME_MASS_SCALE", 0.005)), 1.0e-6
        )
        outcome_error = F.smooth_l1_loss(
            pred_outcome / outcome_scale,
            true_outcome.detach() / outcome_scale,
            reduction="none",
            beta=max(float(_m1(cfg, "V530_OUTCOME_HUBER_BETA", 0.10)), 1.0e-4),
        ).mean(dim=-1)
        v530_outcome_mass_loss = (
            outcome_error[region_valid].mean()
            if bool(region_valid.any().item()) else outcome_error.sum() * 0.0
        )

        predicted_gain = aux["v530_predicted_gain"]
        v530_utility_scale = max(
            float(_m1(cfg, "V530_UTILITY_SCALE", 0.005)), 1.0e-6
        )
        utility_scale = v530_utility_scale
        predicted_gain_norm = predicted_gain / v530_utility_scale
        exact_gain_norm = exact_dice_gain.detach() / v530_utility_scale
        gain_error = F.smooth_l1_loss(
            predicted_gain_norm, exact_gain_norm, reduction="none",
            beta=max(float(_m1(cfg, "V530_GAIN_HUBER_BETA", 0.10)), 1.0e-4),
        )
        positive_dice = region_valid & (exact_dice_gain > positive_margin)
        harmful_dice = region_valid & (exact_dice_gain < -negative_margin)
        neutral_dice = region_valid & (~positive_dice) & (~harmful_dice)
        gain_parts = []
        gain_weights = []
        for mask, weight in (
            (positive_dice, 1.0),
            (harmful_dice, 1.0),
            (neutral_dice, float(_m1(cfg, "V530_NEUTRAL_GAIN_WEIGHT", 0.25))),
        ):
            if bool(mask.any().item()):
                gain_parts.append(weight * gain_error[mask].mean())
                gain_weights.append(weight)
        v530_gain_loss = (
            torch.stack(gain_parts).sum() / max(sum(gain_weights), 1.0e-6)
            if gain_parts else gain_error.sum() * 0.0
        )
        sign_margin_norm = float(_m1(cfg, "V530_SIGN_MARGIN", 1.0e-4)) / v530_utility_scale
        sign_temperature = max(
            float(_m1(cfg, "V530_SIGN_TEMPERATURE", 0.5)), 1.0e-4
        )
        sign_parts = []
        if bool(positive_dice.any().item()):
            sign_parts.append(F.softplus(
                (sign_margin_norm - predicted_gain_norm[positive_dice])
                / sign_temperature
            ).mean())
        if bool(harmful_dice.any().item()):
            sign_parts.append(F.softplus(
                (sign_margin_norm + predicted_gain_norm[harmful_dice])
                / sign_temperature
            ).mean())
        v530_gain_sign_loss = (
            torch.stack(sign_parts).mean()
            if sign_parts else predicted_gain.sum() * 0.0
        )

        candidate_route = aux["v524_region_soft_route"][:, :, 1:]
        v530_expected_utility_case = (
            candidate_route * exact_dice_gain.detach()
        ).sum(dim=(1, 2))
        flat_exact = exact_dice_gain.masked_fill(~region_valid, floor).reshape(b, -1)
        v530_oracle_utility_case = flat_exact.max(dim=1).values.clamp_min(0.0)
        if bool(_m1(cfg, "V530_SELECTOR_ENABLED", False)):
            v530_expected_regret_loss = (
                v530_oracle_utility_case - v530_expected_utility_case
            ).clamp_min(0.0).mean() / v530_utility_scale
            harmful_probability_mass = (
                candidate_route * harmful_dice.to(candidate_route.dtype)
            ).sum(dim=(1, 2))
            v530_harm_mass_loss = harmful_probability_mass.mean()
            normalized_score = aux["v530_normalized_score"].masked_fill(
                ~region_valid, floor
            )
            positive_score = normalized_score.masked_fill(
                ~positive_dice, floor
            ).reshape(b, -1)
            harmful_score = normalized_score.masked_fill(
                ~harmful_dice, floor
            ).reshape(b, -1)
            pair_domain = (
                positive_dice.reshape(b, -1).any(dim=1)
                & harmful_dice.reshape(b, -1).any(dim=1)
            )
            pair_margin = float(_m1(cfg, "V530_HARD_PAIR_MARGIN", 0.10))
            pair_temperature = max(
                float(_m1(cfg, "V530_HARD_PAIR_TEMPERATURE", 0.50)), 1.0e-4
            )
            pair_raw = F.softplus((
                pair_margin + harmful_score.max(dim=1).values
                - positive_score.max(dim=1).values
            ) / pair_temperature)
            v530_hard_pair_loss = (
                pair_raw[pair_domain].mean()
                if bool(pair_domain.any().item()) else pair_raw.sum() * 0.0
            )

        if bool(_m1(cfg, "V530_UNCERTAINTY_ENABLED", False)):
            outcome_logvar = aux["v530_outcome_logvar"].clamp(
                float(_m1(cfg, "V530_LOGVAR_MIN", -16.0)),
                float(_m1(cfg, "V530_LOGVAR_MAX", -8.0)),
            )
            residual_norm = exact_gain_norm - predicted_gain_norm
            outcome_nll = (
                0.5 * torch.exp(-outcome_logvar) * residual_norm.pow(2)
                + 0.5 * outcome_logvar
            )
            v530_outcome_nll_loss = (
                outcome_nll[region_valid].mean()
                if bool(region_valid.any().item()) else outcome_nll.sum() * 0.0
            )

        v530_base_dice_mae = (
            aux["v530_pred_base_dice"] - true_base_dice
        ).abs().mean()
        v530_tp_mass_mae = (
            aux["v530_pred_base_tp"] - true_tp_fraction
        ).abs().mean()
        v530_fp_mass_mae = (
            aux["v530_pred_base_fp"] - true_fp_fraction
        ).abs().mean()
        v530_fn_mass_mae = (
            aux["v530_pred_base_fn"] - true_fn_fraction
        ).abs().mean()
        if bool(add_domain.any().item()):
            v530_add_rate_mae = (
                pred_add_rate[add_domain] - true_add_rate[add_domain]
            ).abs().mean()
        if bool(remove_domain.any().item()):
            v530_remove_rate_mae = (
                pred_remove_rate[remove_domain] - true_remove_rate[remove_domain]
            ).abs().mean()
        if bool(region_valid.any().item()):
            absolute_gain_error = (predicted_gain - exact_dice_gain).abs()
            v530_gain_mae = absolute_gain_error[region_valid].mean()
            if bool(positive_dice.any().item()):
                v530_positive_gain_mae = absolute_gain_error[positive_dice].mean()
            if bool(harmful_dice.any().item()):
                v530_harmful_gain_mae = absolute_gain_error[harmful_dice].mean()
            if bool(neutral_dice.any().item()):
                v530_neutral_gain_mae = absolute_gain_error[neutral_dice].mean()
        predicted_positive = predicted_gain > 0.0
        true_positive = exact_dice_gain > positive_margin
        true_harmful = exact_dice_gain < -negative_margin
        sign_domain = region_valid & (true_positive | true_harmful)
        if bool(sign_domain.any().item()):
            target_sign = true_positive[sign_domain]
            pred_sign = predicted_positive[sign_domain]
            v530_gain_sign_accuracy = (pred_sign == target_sign).float().mean()
            tp_sign = (pred_sign & target_sign).float().sum()
            v530_positive_precision = tp_sign / pred_sign.float().sum().clamp_min(1.0)
            v530_positive_recall = tp_sign / target_sign.float().sum().clamp_min(1.0)
        sigma_valid = aux["v530_predicted_sigma"][region_valid]
        if sigma_valid.numel() > 0:
            quantiles = torch.quantile(
                sigma_valid.float(),
                sigma_valid.new_tensor([0.50, 0.90, 0.99]).float(),
            ).to(c0.dtype)
            v530_sigma_p50, v530_sigma_p90, v530_sigma_p99 = quantiles.unbind()

        oracle_utility = torch.where(
            oracle_edits, best_candidate_utility, torch.zeros_like(best_candidate_utility)
        )
        expected_utility = (candidate_route * exact_dice_gain.detach()).sum(dim=2)
        expected_regret = (oracle_utility - expected_utility).clamp_min(0.0)
        expected_regret_loss = v530_expected_regret_loss
        top1_loss = v530_gain_loss
        utility_kl_loss = v530_harm_mass_loss
        preference_loss = v530_hard_pair_loss
        region_utility_loss = v530_outcome_mass_loss
        region_harm_loss = v530_outcome_pixel_loss
        region_nll_loss = v530_outcome_nll_loss

    if v529_enabled:
        required_v529 = (
            "v529_base_error_logit",
            "v529_pred_base_tp",
            "v529_pred_base_fp",
            "v529_pred_base_fn",
            "v529_pred_base_dice",
            "v529_outcome_pixel_logit",
            "v529_outcome_logvar",
            "v529_pred_fn_fix",
            "v529_pred_fp_fix",
            "v529_pred_tn_harm",
            "v529_pred_tp_harm",
            "v529_predicted_gain",
            "v529_normalized_score",
            "v529_predicted_sigma",
            "v529_predicted_harm_fraction",
        )
        missing_v529 = [key for key in required_v529 if key not in aux]
        if missing_v529:
            raise RuntimeError("V529 loss missing outputs: " + str(missing_v529))

        image_pixels = float(max(h * w, 1))
        base_fg_count = base_hard.flatten(1).sum(dim=1).to(c0.dtype)
        base_bg_count = (~base_hard).flatten(1).sum(dim=1).to(c0.dtype)
        true_fp_rate = base_fp / base_fg_count.clamp_min(1.0)
        true_fn_rate = base_fn / base_bg_count.clamp_min(1.0)
        base_error_target = torch.stack([true_fp_rate, true_fn_rate], dim=1)
        v529_base_error_loss = F.binary_cross_entropy_with_logits(
            aux["v529_base_error_logit"], base_error_target
        )

        # Dense supervision is evaluated only on pixels that the real candidate
        # actually adds or removes.  Positive/negative subclasses are averaged
        # separately so common TN/TP pixels cannot dominate the result.
        pixel_logits = aux["v529_outcome_pixel_logit"]
        add_full_teacher = (
            valid & (~base_hard.expand_as(candidate_hard)) & candidate_hard
        )
        remove_full_teacher = (
            valid & base_hard.expand_as(candidate_hard) & (~candidate_hard)
        )
        add_fix_target_pixel = gt_hard.expand_as(candidate_hard)
        remove_fix_target_pixel = (~gt_hard).expand_as(candidate_hard)

        def balanced_edit_bce(
            logit: torch.Tensor, target: torch.Tensor, domain: torch.Tensor
        ) -> torch.Tensor:
            raw = F.binary_cross_entropy_with_logits(
                logit, target.to(logit.dtype), reduction="none"
            )
            parts = []
            positive_domain = domain & target
            negative_domain = domain & (~target)
            if bool(positive_domain.any().item()):
                parts.append(raw[positive_domain].mean())
            if bool(negative_domain.any().item()):
                parts.append(raw[negative_domain].mean())
            return (
                torch.stack(parts).mean()
                if parts else raw.sum() * 0.0
            )

        add_pixel_loss = balanced_edit_bce(
            pixel_logits[:, :, 0], add_fix_target_pixel, add_full_teacher
        )
        remove_pixel_loss = balanced_edit_bce(
            pixel_logits[:, :, 1], remove_fix_target_pixel, remove_full_teacher
        )
        pixel_parts = []
        if bool(add_full_teacher.any().item()):
            pixel_parts.append(add_pixel_loss)
        if bool(remove_full_teacher.any().item()):
            pixel_parts.append(remove_pixel_loss)
        v529_outcome_pixel_loss = (
            torch.stack(pixel_parts).mean()
            if pixel_parts else pixel_logits.sum() * 0.0
        )

        true_outcome = torch.stack(
            [exact_fn_fix, exact_fp_fix, exact_tn_harm, exact_tp_harm], dim=-1
        )
        pred_outcome = torch.stack(
            [
                aux["v529_pred_fn_fix"],
                aux["v529_pred_fp_fix"],
                aux["v529_pred_tn_harm"],
                aux["v529_pred_tp_harm"],
            ],
            dim=-1,
        )
        outcome_scale = max(
            float(_m1(cfg, "V529_OUTCOME_MASS_SCALE", 0.005)), 1.0e-6
        )
        outcome_error = F.smooth_l1_loss(
            pred_outcome / outcome_scale,
            true_outcome.detach() / outcome_scale,
            reduction="none",
            beta=max(float(_m1(cfg, "V529_OUTCOME_HUBER_BETA", 0.10)), 1.0e-4),
        ).mean(dim=-1)
        v529_outcome_mass_loss = (
            outcome_error[region_valid].mean()
            if bool(region_valid.any().item())
            else outcome_error.sum() * 0.0
        )

        predicted_gain = aux["v529_predicted_gain"]
        v529_utility_scale = max(
            float(_m1(cfg, "V529_UTILITY_SCALE", 0.005)), 1.0e-6
        )
        utility_scale = v529_utility_scale
        predicted_gain_norm = predicted_gain / v529_utility_scale
        exact_gain_norm = exact_dice_gain.detach() / v529_utility_scale
        gain_error = F.smooth_l1_loss(
            predicted_gain_norm, exact_gain_norm, reduction="none",
            beta=max(float(_m1(cfg, "V529_GAIN_HUBER_BETA", 0.10)), 1.0e-4),
        )
        positive_dice = region_valid & (exact_dice_gain > positive_margin)
        harmful_dice = region_valid & (exact_dice_gain < -negative_margin)
        neutral_dice = region_valid & (~positive_dice) & (~harmful_dice)
        gain_parts = []
        gain_weights = []
        for mask, weight in (
            (positive_dice, 1.0),
            (harmful_dice, 1.0),
            (neutral_dice, float(_m1(cfg, "V529_NEUTRAL_GAIN_WEIGHT", 0.25))),
        ):
            if bool(mask.any().item()):
                gain_parts.append(weight * gain_error[mask].mean())
                gain_weights.append(weight)
        v529_gain_loss = (
            torch.stack(gain_parts).sum() / max(sum(gain_weights), 1.0e-6)
            if gain_parts else gain_error.sum() * 0.0
        )

        sign_margin_norm = float(_m1(cfg, "V529_SIGN_MARGIN", 1.0e-4)) / v529_utility_scale
        sign_temperature = max(
            float(_m1(cfg, "V529_SIGN_TEMPERATURE", 0.5)), 1.0e-4
        )
        sign_parts = []
        if bool(positive_dice.any().item()):
            sign_parts.append(F.softplus(
                (sign_margin_norm - predicted_gain_norm[positive_dice])
                / sign_temperature
            ).mean())
        if bool(harmful_dice.any().item()):
            sign_parts.append(F.softplus(
                (sign_margin_norm + predicted_gain_norm[harmful_dice])
                / sign_temperature
            ).mean())
        v529_gain_sign_loss = (
            torch.stack(sign_parts).mean()
            if sign_parts else predicted_gain.sum() * 0.0
        )

        candidate_route = aux["v524_region_soft_route"][:, :, 1:]
        v529_expected_utility_case = (
            candidate_route * exact_dice_gain.detach()
        ).sum(dim=(1, 2))
        flat_exact = exact_dice_gain.masked_fill(~region_valid, floor).reshape(b, -1)
        v529_oracle_utility_case = flat_exact.max(dim=1).values.clamp_min(0.0)
        if bool(_m1(cfg, "V529_SELECTOR_ENABLED", False)):
            v529_expected_regret_loss = (
                v529_oracle_utility_case - v529_expected_utility_case
            ).clamp_min(0.0).mean() / v529_utility_scale
            harmful_probability_mass = (
                candidate_route * harmful_dice.to(candidate_route.dtype)
            ).sum(dim=(1, 2))
            v529_harm_mass_loss = harmful_probability_mass.mean()

            normalized_score = aux["v529_normalized_score"].masked_fill(
                ~region_valid, floor
            )
            positive_score = normalized_score.masked_fill(
                ~positive_dice, floor
            ).reshape(b, -1)
            harmful_score = normalized_score.masked_fill(
                ~harmful_dice, floor
            ).reshape(b, -1)
            pair_domain = (
                positive_dice.reshape(b, -1).any(dim=1)
                & harmful_dice.reshape(b, -1).any(dim=1)
            )
            pair_margin = float(_m1(cfg, "V529_HARD_PAIR_MARGIN", 0.10))
            pair_temperature = max(
                float(_m1(cfg, "V529_HARD_PAIR_TEMPERATURE", 0.50)), 1.0e-4
            )
            pair_raw = F.softplus(
                (
                    pair_margin
                    + harmful_score.max(dim=1).values
                    - positive_score.max(dim=1).values
                ) / pair_temperature
            )
            v529_hard_pair_loss = (
                pair_raw[pair_domain].mean()
                if bool(pair_domain.any().item())
                else pair_raw.sum() * 0.0
            )

        if bool(_m1(cfg, "V529_UNCERTAINTY_ENABLED", False)):
            outcome_logvar = aux["v529_outcome_logvar"].clamp(
                float(_m1(cfg, "V529_LOGVAR_MIN", -8.0)),
                float(_m1(cfg, "V529_LOGVAR_MAX", 2.0)),
            )
            residual_norm = exact_gain_norm - predicted_gain_norm
            outcome_nll = (
                0.5 * torch.exp(-outcome_logvar) * residual_norm.pow(2)
                + 0.5 * outcome_logvar
            )
            v529_outcome_nll_loss = (
                outcome_nll[region_valid].mean()
                if bool(region_valid.any().item())
                else outcome_nll.sum() * 0.0
            )

        # Calibration diagnostics pinpoint whether error comes from the factual
        # Base state or from Add/Remove outcome estimation.
        true_tp_fraction = base_tp / image_pixels
        true_fp_fraction = base_fp / image_pixels
        true_fn_fraction = base_fn / image_pixels
        true_base_dice = base_dice_exact
        v529_base_dice_mae = (aux["v529_pred_base_dice"] - true_base_dice).abs().mean()
        v529_tp_mass_mae = (aux["v529_pred_base_tp"] - true_tp_fraction).abs().mean()
        v529_fp_mass_mae = (aux["v529_pred_base_fp"] - true_fp_fraction).abs().mean()
        v529_fn_mass_mae = (aux["v529_pred_base_fn"] - true_fn_fraction).abs().mean()
        exact_add = exact_fn_fix + exact_tn_harm
        exact_remove = exact_fp_fix + exact_tp_harm
        pred_add_rate = aux["v529_pred_fn_fix"] / (
            aux["v529_pred_fn_fix"] + aux["v529_pred_tn_harm"]
        ).clamp_min(EPS)
        pred_remove_rate = aux["v529_pred_fp_fix"] / (
            aux["v529_pred_fp_fix"] + aux["v529_pred_tp_harm"]
        ).clamp_min(EPS)
        true_add_rate = exact_fn_fix / exact_add.clamp_min(EPS)
        true_remove_rate = exact_fp_fix / exact_remove.clamp_min(EPS)
        add_domain = region_valid & (exact_add > 0.0)
        remove_domain = region_valid & (exact_remove > 0.0)
        if bool(add_domain.any().item()):
            v529_add_rate_mae = (
                pred_add_rate[add_domain] - true_add_rate[add_domain]
            ).abs().mean()
        if bool(remove_domain.any().item()):
            v529_remove_rate_mae = (
                pred_remove_rate[remove_domain] - true_remove_rate[remove_domain]
            ).abs().mean()
        if bool(region_valid.any().item()):
            v529_gain_mae = (
                predicted_gain[region_valid] - exact_dice_gain[region_valid]
            ).abs().mean()
        predicted_positive = predicted_gain > 0.0
        true_positive = exact_dice_gain > positive_margin
        true_harmful = exact_dice_gain < -negative_margin
        sign_domain = region_valid & (true_positive | true_harmful)
        if bool(sign_domain.any().item()):
            target_sign = true_positive[sign_domain]
            pred_sign = predicted_positive[sign_domain]
            v529_gain_sign_accuracy = (pred_sign == target_sign).float().mean()
            tp_sign = (pred_sign & target_sign).float().sum()
            v529_positive_precision = tp_sign / pred_sign.float().sum().clamp_min(1.0)
            v529_positive_recall = tp_sign / target_sign.float().sum().clamp_min(1.0)
        sigma_valid = aux["v529_predicted_sigma"][region_valid]
        if sigma_valid.numel() > 0:
            quantiles = torch.quantile(
                sigma_valid.float(), sigma_valid.new_tensor([0.50, 0.90, 0.99]).float()
            ).to(c0.dtype)
            v529_sigma_p50, v529_sigma_p90, v529_sigma_p99 = quantiles.unbind()

        oracle_utility = torch.where(
            oracle_edits, best_candidate_utility, torch.zeros_like(best_candidate_utility)
        )
        expected_utility = (candidate_route * exact_dice_gain.detach()).sum(dim=2)
        expected_regret = (oracle_utility - expected_utility).clamp_min(0.0)
        expected_regret_loss = v529_expected_regret_loss
        top1_loss = v529_gain_loss
        utility_kl_loss = v529_expected_regret_loss
        preference_loss = v529_hard_pair_loss
    elif v528_enabled:
        required_v528 = (
            "v528_base_confusion_logit",
            "v528_base_confusion_prob",
            "v528_outcome_correctness_logit",
            "v528_outcome_logvar",
            "v528_pred_fn_fix",
            "v528_pred_fp_fix",
            "v528_pred_tn_harm",
            "v528_pred_tp_harm",
            "v528_predicted_gain",
            "v528_gain_lcb",
            "v528_predicted_harm_fraction",
        )
        missing_v528 = [key for key in required_v528 if key not in aux]
        if missing_v528:
            raise RuntimeError("V528 loss missing outputs: " + str(missing_v528))

        image_pixels = float(max(h * w, 1))
        true_tn = (
            (~base_hard) & (~gt_hard)
        ).flatten(1).sum(dim=1).to(c0.dtype)
        base_confusion_target = torch.stack(
            [base_tp, base_fp, base_fn, true_tn], dim=1
        ) / image_pixels
        base_confusion_log_prob = F.log_softmax(
            aux["v528_base_confusion_logit"], dim=1
        )
        v528_base_confusion_loss = -(
            base_confusion_target * base_confusion_log_prob
        ).sum(dim=1).mean()

        exact_add = exact_fn_fix + exact_tn_harm
        exact_remove = exact_fp_fix + exact_tp_harm
        add_target = exact_fn_fix / exact_add.clamp_min(EPS)
        remove_target = exact_fp_fix / exact_remove.clamp_min(EPS)
        outcome_logits = aux["v528_outcome_correctness_logit"]
        add_loss = F.binary_cross_entropy_with_logits(
            outcome_logits[..., 0], add_target, reduction="none"
        )
        remove_loss = F.binary_cross_entropy_with_logits(
            outcome_logits[..., 1], remove_target, reduction="none"
        )
        outcome_parts = []
        add_domain = region_valid & (exact_add > 0.0)
        remove_domain = region_valid & (exact_remove > 0.0)
        if bool(add_domain.any().item()):
            outcome_parts.append(add_loss[add_domain].mean())
        if bool(remove_domain.any().item()):
            outcome_parts.append(remove_loss[remove_domain].mean())
        v528_outcome_correctness_loss = (
            torch.stack(outcome_parts).mean()
            if outcome_parts else outcome_logits.sum() * 0.0
        )

        true_outcome = torch.stack(
            [exact_fn_fix, exact_fp_fix, exact_tn_harm, exact_tp_harm],
            dim=-1,
        )
        pred_outcome = torch.stack(
            [
                aux["v528_pred_fn_fix"],
                aux["v528_pred_fp_fix"],
                aux["v528_pred_tn_harm"],
                aux["v528_pred_tp_harm"],
            ],
            dim=-1,
        )
        outcome_mass_error = F.smooth_l1_loss(
            pred_outcome, true_outcome, reduction="none", beta=1.0e-4
        ).mean(dim=-1)
        v528_outcome_mass_loss = (
            outcome_mass_error[region_valid].mean() / 1.0e-3
            if bool(region_valid.any().item())
            else outcome_mass_error.sum() * 0.0
        )

        predicted_gain = aux["v528_predicted_gain"]
        gain_error = F.smooth_l1_loss(
            predicted_gain, exact_dice_gain.detach(),
            reduction="none",
            beta=max(float(_m1(cfg, "V528_GAIN_HUBER_BETA", 5.0e-4)), 1.0e-6),
        )
        positive_dice = region_valid & (exact_dice_gain > positive_margin)
        harmful_dice = region_valid & (exact_dice_gain < -negative_margin)
        neutral_dice = region_valid & (~positive_dice) & (~harmful_dice)
        gain_parts = []
        for mask, weight in (
            (positive_dice, 1.0),
            (harmful_dice, 1.0),
            (neutral_dice, float(_m1(cfg, "V528_NEUTRAL_GAIN_WEIGHT", 0.25))),
        ):
            if bool(mask.any().item()):
                gain_parts.append(weight * gain_error[mask].mean())
        v528_gain_loss = (
            torch.stack(gain_parts).sum() / max(sum(
                [1.0 if bool(positive_dice.any().item()) else 0.0,
                 1.0 if bool(harmful_dice.any().item()) else 0.0,
                 float(_m1(cfg, "V528_NEUTRAL_GAIN_WEIGHT", 0.25))
                 if bool(neutral_dice.any().item()) else 0.0]
            ), 1.0e-6)
            if gain_parts else gain_error.sum() * 0.0
        )

        sign_temperature = max(
            float(_m1(cfg, "V528_SIGN_TEMPERATURE", 5.0e-4)), 1.0e-6
        )
        sign_margin = float(_m1(cfg, "V528_SIGN_MARGIN", 1.0e-4))
        sign_parts = []
        if bool(positive_dice.any().item()):
            sign_parts.append(F.softplus(
                (sign_margin - predicted_gain[positive_dice]) / sign_temperature
            ).mean())
        if bool(harmful_dice.any().item()):
            sign_parts.append(F.softplus(
                (sign_margin + predicted_gain[harmful_dice]) / sign_temperature
            ).mean())
        v528_gain_sign_loss = (
            torch.stack(sign_parts).mean()
            if sign_parts else predicted_gain.sum() * 0.0
        )

        candidate_route = aux["v524_region_soft_route"][:, :, 1:]
        v528_expected_utility_case = (
            candidate_route * exact_dice_gain.detach()
        ).sum(dim=(1, 2))
        flat_exact = exact_dice_gain.masked_fill(~region_valid, floor).reshape(b, -1)
        v528_oracle_utility_case = flat_exact.max(dim=1).values.clamp_min(0.0)
        utility_scale = max(float(_m1(cfg, "V528_UTILITY_SCALE", 0.005)), 1.0e-6)
        v528_expected_regret_loss = (
            v528_oracle_utility_case - v528_expected_utility_case
        ).clamp_min(0.0).mean() / utility_scale

        harmful_probability_mass = (
            candidate_route * harmful_dice.to(candidate_route.dtype)
        ).sum(dim=(1, 2))
        v528_harm_mass_loss = harmful_probability_mass.mean()

        gain_lcb = aux["v528_gain_lcb"].masked_fill(~region_valid, floor)
        positive_score = gain_lcb.masked_fill(~positive_dice, floor).reshape(b, -1)
        harmful_score = gain_lcb.masked_fill(~harmful_dice, floor).reshape(b, -1)
        positive_exists = positive_dice.reshape(b, -1).any(dim=1)
        harmful_exists = harmful_dice.reshape(b, -1).any(dim=1)
        pair_domain = positive_exists & harmful_exists
        best_positive_score = positive_score.max(dim=1).values
        hardest_harm_score = harmful_score.max(dim=1).values
        pair_margin = float(_m1(cfg, "V528_HARD_PAIR_MARGIN", 0.25))
        pair_temperature = max(
            float(_m1(cfg, "V528_HARD_PAIR_TEMPERATURE", 0.25)), 1.0e-5
        )
        pair_raw = F.softplus(
            (pair_margin + hardest_harm_score - best_positive_score)
            / pair_temperature
        )
        v528_hard_pair_loss = (
            pair_raw[pair_domain].mean()
            if bool(pair_domain.any().item()) else pair_raw.sum() * 0.0
        )

        outcome_logvar = aux["v528_outcome_logvar"].clamp(
            float(_m1(cfg, "V526_LOGVAR_MIN", -14.0)),
            float(_m1(cfg, "V526_LOGVAR_MAX", 2.0)),
        )
        gain_residual = exact_dice_gain.detach() - predicted_gain
        outcome_nll = 0.5 * torch.exp(-outcome_logvar) * gain_residual.pow(2) + 0.5 * outcome_logvar
        v528_outcome_nll_loss = (
            outcome_nll[region_valid].mean()
            if bool(region_valid.any().item()) else outcome_nll.sum() * 0.0
        )

        # Compatibility diagnostics remain region-shaped while the actual V528
        # objective uses the single global action distribution above.
        oracle_utility = torch.where(
            oracle_edits, best_candidate_utility, torch.zeros_like(best_candidate_utility)
        )
        expected_utility = (
            candidate_route * exact_dice_gain.detach()
        ).sum(dim=2)
        expected_regret = (oracle_utility - expected_utility).clamp_min(0.0)
        expected_regret_loss = v528_expected_regret_loss
        top1_loss = v528_gain_loss
        utility_kl_loss = v528_expected_regret_loss
        preference_loss = v528_hard_pair_loss
    elif v527_enabled:
        editability_logit = aux.get("v527_region_editability_logit")
        conditional_score = aux.get("v527_region_conditional_score")
        if not isinstance(editability_logit, torch.Tensor):
            raise RuntimeError("V527 requires v527_region_editability_logit")
        if not isinstance(conditional_score, torch.Tensor):
            raise RuntimeError("V527 requires v527_region_conditional_score")
        if conditional_score.shape != region_utility.shape:
            raise RuntimeError("V527 conditional-score shape mismatch")

        editability_loss = _v524_balanced_binary_bce(
            editability_logit, oracle_edits
        )
        edit_prob = torch.sigmoid(editability_logit)

        conditional_temperature = max(
            float(_m1(cfg, "V527_CONDITIONAL_TEMPERATURE", 0.25)), 1.0e-5
        )
        masked_conditional_score = conditional_score.masked_fill(~region_valid, floor)
        conditional_logits = masked_conditional_score / conditional_temperature
        conditional_top1_raw = F.cross_entropy(
            conditional_logits.reshape(-1, conditional_logits.shape[-1]),
            best_candidate_index.reshape(-1),
            reduction="none",
        ).reshape_as(best_candidate_index)
        conditional_top1_loss = (
            conditional_top1_raw[oracle_edits].mean()
            if bool(oracle_edits.any().item())
            else conditional_top1_raw.sum() * 0.0
        )
        top1_loss = conditional_top1_loss

        valid_count = region_valid.float().sum(dim=2).clamp_min(1.0)
        utility_mean = (
            region_utility.masked_fill(~region_valid, 0.0).sum(dim=2)
            / valid_count
        )
        utility_var = (
            ((region_utility - utility_mean[..., None]).pow(2)
             * region_valid.to(region_utility.dtype)).sum(dim=2)
            / valid_count
        )
        utility_std = torch.sqrt(utility_var + 1.0e-12)
        teacher_temp = (
            utility_std * float(_m1(cfg, "V527_TEACHER_TEMP_SCALE", 1.0))
        ).clamp(
            min=float(_m1(cfg, "V527_TEACHER_TEMP_MIN", 2.5e-4)),
            max=float(_m1(cfg, "V527_TEACHER_TEMP_MAX", 5.0e-3)),
        )
        teacher_logits = (
            region_utility / teacher_temp[..., None]
        ).masked_fill(~region_valid, floor)
        teacher_prob = F.softmax(teacher_logits, dim=2).detach()
        student_log_prob_candidate = F.log_softmax(conditional_logits, dim=2)
        candidate_prob = student_log_prob_candidate.exp()
        utility_kl_raw = (
            teacher_prob
            * (
                torch.log(teacher_prob.clamp_min(EPS))
                - student_log_prob_candidate
            )
        ).sum(dim=2)
        utility_kl_loss = (
            utility_kl_raw[oracle_edits].mean()
            if bool(oracle_edits.any().item())
            else utility_kl_raw.sum() * 0.0
        )

        student_prob = torch.cat(
            [
                (1.0 - edit_prob)[..., None],
                edit_prob[..., None] * candidate_prob,
            ],
            dim=2,
        )
        expected_utility = (
            student_prob * full_utility.masked_fill(~full_valid, 0.0)
        ).sum(dim=2)
        expected_regret = (oracle_utility - expected_utility).clamp_min(0.0)
        utility_scale = max(
            float(_m1(cfg, "V526_UTILITY_SCALE", 0.005)), 1.0e-6
        )
        expected_regret_loss = _v524_balanced_group_mean(
            expected_regret / utility_scale, oracle_edits, ~oracle_edits
        )

        oracle_candidate_score = masked_conditional_score.gather(
            2, best_candidate_index[..., None]
        ).squeeze(-1)
        candidate_ids = torch.arange(
            n, device=region_utility.device
        ).view(1, 1, n)
        wrong_valid = region_valid & (
            candidate_ids != best_candidate_index[..., None]
        )
        wrong_score, wrong_index = masked_conditional_score.masked_fill(
            ~wrong_valid, floor
        ).max(dim=2)
        wrong_exists = wrong_valid.any(dim=2) & oracle_edits
        wrong_utility = region_utility.gather(
            2, wrong_index[..., None]
        ).squeeze(-1)
        wrong_gap = (oracle_utility - wrong_utility).clamp_min(0.0)

        harmful_score, harmful_index = masked_conditional_score.masked_fill(
            ~harmful, floor
        ).max(dim=2)
        harmful_exists_for_rank = harmful.any(dim=2) & oracle_edits
        harmful_utility = region_utility.gather(
            2, harmful_index[..., None]
        ).squeeze(-1)
        harmful_gap = (oracle_utility - harmful_utility).clamp_min(0.0)

        pair_temperature = max(
            float(_m1(cfg, "V527_HARD_PAIR_TEMPERATURE", 0.25)), 1.0e-5
        )
        margin_base = float(_m1(cfg, "V527_HARD_PAIR_MARGIN_BASE", 0.0))
        margin_scale = float(_m1(cfg, "V527_HARD_PAIR_MARGIN_SCALE", 50.0))
        wrong_pair = F.softplus(
            (
                margin_base
                + margin_scale * wrong_gap.detach()
                + wrong_score
                - oracle_candidate_score
            )
            / pair_temperature
        )
        harmful_pair = F.softplus(
            (
                margin_base
                + margin_scale * harmful_gap.detach()
                + harmful_score
                - oracle_candidate_score
            )
            / pair_temperature
        )
        pair_parts = []
        if bool(wrong_exists.any().item()):
            pair_parts.append(wrong_pair[wrong_exists].mean())
        if bool(harmful_exists_for_rank.any().item()):
            pair_parts.append(harmful_pair[harmful_exists_for_rank].mean())
        hard_pair_loss = (
            torch.stack(pair_parts).mean()
            if pair_parts
            else wrong_pair.sum() * 0.0
        )
        preference_loss = hard_pair_loss

        harmful_candidate_mass = (
            candidate_prob * harmful.to(candidate_prob.dtype)
        ).sum(dim=2)
        preserve_noharm_raw = edit_prob * harmful_candidate_mass
        preserve_noharm_loss = (
            preserve_noharm_raw[~oracle_edits].mean()
            if bool((~oracle_edits).any().item())
            else preserve_noharm_raw.sum() * 0.0
        )
    else:
        top1_raw = F.cross_entropy(
            action_logits.reshape(-1, action_logits.shape[-1]),
            oracle_index.reshape(-1),
            reduction="none",
        ).reshape_as(oracle_index)
        top1_loss = _v524_balanced_group_mean(
            top1_raw, oracle_edits, ~oracle_edits
        )

        teacher_temperature = max(
            float(_m1(cfg, "V526_TEACHER_TEMPERATURE", 0.002)), 1.0e-5
        )
        student_temperature = max(
            float(_m1(cfg, "V526_STUDENT_TEMPERATURE", route_temperature)),
            1.0e-5,
        )
        teacher_logits = (
            full_utility / teacher_temperature
        ).masked_fill(~full_valid, floor)
        teacher_prob = F.softmax(teacher_logits, dim=2).detach()
        student_logits = (
            action_score / student_temperature
        ).masked_fill(~full_valid, floor)
        student_log_prob = F.log_softmax(student_logits, dim=2)
        student_prob = student_log_prob.exp()
        utility_kl_raw = (
            teacher_prob
            * (
                torch.log(teacher_prob.clamp_min(EPS))
                - student_log_prob
            )
        ).sum(dim=2)
        utility_kl_loss = _v524_balanced_group_mean(
            utility_kl_raw, oracle_edits, ~oracle_edits
        )

        expected_utility = (
            student_prob * full_utility.masked_fill(~full_valid, 0.0)
        ).sum(dim=2)
        expected_regret = (oracle_utility - expected_utility).clamp_min(0.0)
        utility_scale = max(
            float(_m1(cfg, "V526_UTILITY_SCALE", 0.005)), 1.0e-6
        )
        expected_regret_loss = _v524_balanced_group_mean(
            expected_regret / utility_scale, oracle_edits, ~oracle_edits
        )

        oracle_score = action_score.gather(
            2, oracle_index[..., None]
        ).squeeze(-1)
        utility_gap = (oracle_utility[..., None] - full_utility).clamp_min(0.0)
        preference_gap = float(
            _m1(cfg, "V526_PREFERENCE_UTILITY_GAP", 1.0e-4)
        )
        oracle_onehot = F.one_hot(
            oracle_index, num_classes=action_score.shape[2]
        ).bool()
        preference_mask = (
            full_valid & (~oracle_onehot) & (utility_gap >= preference_gap)
        )
        preference_margin = (
            float(_m1(cfg, "V526_PREFERENCE_MARGIN_BASE", 0.0))
            + float(_m1(cfg, "V526_PREFERENCE_MARGIN_SCALE", 1.0))
            * utility_gap
        )
        preference_temperature = max(
            float(_m1(cfg, "V526_PREFERENCE_TEMPERATURE", 0.002)), 1.0e-5
        )
        preference_raw = F.softplus(
            (
                preference_margin
                + action_score
                - oracle_score[..., None]
            )
            / preference_temperature
        )
        preference_loss = (
            preference_raw[preference_mask].mean()
            if bool(preference_mask.any().item())
            else preference_raw.sum() * 0.0
        )

    # V524 compatibility objectives remain available for clean ablations.
    target_mask = torch.cat([(~repairable)[..., None], positive], dim=2)
    target_logits = action_logits.masked_fill(~target_mask, floor)
    listwise_per_region = torch.logsumexp(action_logits, dim=2) - torch.logsumexp(target_logits, dim=2)
    listwise_loss = _v524_balanced_group_mean(listwise_per_region, repairable, ~repairable)

    candidate_score = action_score[:, :, 1:]
    positive_score = candidate_score.masked_fill(~positive, floor)
    positive_lse = torch.logsumexp(positive_score, dim=2)
    positive_reference = torch.where(repairable, positive_lse, action_score[:, :, 0])
    harmful_score = candidate_score.masked_fill(~harmful, floor).max(dim=2).values
    rank_margin = float(_m1(cfg, "V524_HARD_NEGATIVE_MARGIN", 0.50))
    hard_negative_per_region = F.relu(rank_margin + harmful_score - positive_reference)
    hard_negative_loss = hard_negative_per_region[harmful_present].mean() if bool(harmful_present.any().item()) else hard_negative_per_region.sum() * 0.0

    region_mu = aux["v524_region_utility_mu"]
    region_logvar = aux["v524_region_logvar"].clamp(
        float(_m1(cfg, "V526_LOGVAR_MIN", -14.0)),
        float(_m1(cfg, "V526_LOGVAR_MAX", 2.0)),
    )
    region_harm_logit = aux["v524_region_harm_logit"]
    utility_error = F.smooth_l1_loss(
        region_mu,
        region_utility,
        reduction="none",
        beta=max(float(_m1(cfg, "V526_UTILITY_HUBER_BETA", 0.002)), 1.0e-6),
    )
    region_utility_loss = utility_error[region_valid].mean() if bool(region_valid.any().item()) else utility_error.sum() * 0.0
    region_harm_loss = _v524_balanced_binary_bce(region_harm_logit[region_valid], harmful[region_valid]) if bool(region_valid.any().item()) else region_harm_logit.sum() * 0.0
    region_nll = 0.5 * torch.exp(-region_logvar) * (region_utility - region_mu).pow(2) + 0.5 * region_logvar
    region_nll_loss = region_nll[region_valid].mean() if bool(region_valid.any().item()) else region_nll.sum() * 0.0

    # Selective risk and adaptive coverage.  Preserve is action zero.
    soft_route = aux["v524_region_soft_route"]
    candidate_route = soft_route[:, :, 1:]
    accept_prob = candidate_route.sum(dim=2)
    harmful_mass = (candidate_route * harmful.to(candidate_route.dtype)).sum(dim=2)
    selective_risk = harmful_mass.sum() / accept_prob.sum().clamp_min(EPS)
    risk_budget = float(_m1(cfg, "V524_SELECTIVE_RISK_BUDGET", 0.10))
    selective_risk_loss = F.relu(selective_risk - risk_budget)
    target_coverage = repairable.float().mean().detach() * float(_m1(cfg, "V524_COVERAGE_FRACTION", 0.80))
    coverage = accept_prob.mean()
    coverage_loss = F.relu(target_coverage - coverage)
    accept_bce = F.binary_cross_entropy(accept_prob.clamp(EPS, 1.0 - EPS), repairable.float())

    # Complete-region execution removes the learned pixel-fragmentation path.
    # The legacy refiner loss is kept only when explicitly requested.
    selected_index_map = aux["v524_selected_index_map"].long().clamp(0, n)
    bank_hard = torch.cat([base_hard, candidate_hard], dim=1)
    selected_hard = bank_hard.gather(1, selected_index_map[:, None])
    selected_action = selected_index_map[:, None] > 0
    selected_repair = selected_action & base_wrong & (selected_hard == gt_hard)
    refiner_target = selected_repair.float()
    refiner_domain = selected_action | repair.any(dim=1, keepdim=True)
    refiner_logit = aux["v524_refiner_gate_logit"]
    if bool(_m1(cfg, "V524_FULL_REGION_EXECUTION", True)):
        refiner_loss = refiner_logit.sum() * 0.0
    else:
        refiner_raw = F.binary_cross_entropy_with_logits(refiner_logit, refiner_target, reduction="none")
        refiner_loss = refiner_raw[refiner_domain].mean() if bool(refiner_domain.any().item()) else refiner_raw.sum() * 0.0

    pre_guard_prob = _as_b1hw(aux["v524_pre_guard_prob"])
    training_prob = pre_guard_prob
    deployed_prob = _as_b1hw(aux["m2_fused_probs"])
    deploy_seg_loss = _v519_prob_bce_dice(training_prob, gt)
    boundary_loss = (_soft_boundary(training_prob, radius=1) - _soft_boundary(gt, radius=1)).abs().mean()
    base_dice_case = _soft_dice_probs(c0, gt)[:, 0]
    proposal_dice_case = _soft_dice_probs(pre_guard_prob, gt)[:, 0]
    deployed_dice_case = _soft_dice_probs(deployed_prob, gt)[:, 0]
    proposal_gain_case = (proposal_dice_case - base_dice_case).detach()
    case_regret = F.relu(
        base_dice_case - proposal_dice_case + float(_m1(cfg, "V524_CASE_NOHARM_MARGIN", 0.0))
    )
    case_noharm_loss = case_regret.mean()

    case_mu = aux["v524_case_gain_mu"]
    case_logvar = aux["v524_case_logvar"].clamp(
        float(_m1(cfg, "V526_LOGVAR_MIN", -14.0)),
        float(_m1(cfg, "V526_LOGVAR_MAX", 2.0)),
    )
    case_harm_logit = aux["v524_case_harm_logit"]
    case_target = (proposal_dice_case - base_dice_case).detach()
    case_gain_loss = F.smooth_l1_loss(case_mu, case_target, beta=0.01)
    case_nll_loss = (0.5 * torch.exp(-case_logvar) * (case_target - case_mu).pow(2) + 0.5 * case_logvar).mean()
    case_harm_target = (case_target < -float(_m1(cfg, "V524_NEGATIVE_UTILITY_MARGIN", 1.0e-4))).float()
    case_harm_loss = _v524_balanced_binary_bce(case_harm_logit, case_harm_target)

    m2_loss = (
        float(_m1(cfg, "V524_FAMILY_WEIGHT", 0.50)) * family_loss
        + float(_m1(cfg, "V524_PROMPT_WEIGHT_LOSS", 1.00)) * prompt_loss
        + float(_m1(cfg, "V524_DENSE_UTILITY_WEIGHT", 0.50)) * dense_utility_loss
        + float(_m1(cfg, "V524_DENSE_HARM_WEIGHT", 0.50)) * dense_harm_loss
        + float(_m1(cfg, "V524_DENSE_NLL_WEIGHT", 0.10)) * dense_nll_loss
        + float(_m1(cfg, "V526_TOP1_WEIGHT", 0.0 if not v526_enabled else 2.0)) * top1_loss
        + float(_m1(cfg, "V526_UTILITY_KL_WEIGHT", 0.0 if not v526_enabled else 1.0)) * utility_kl_loss
        + float(_m1(cfg, "V526_EXPECTED_REGRET_WEIGHT", 0.0 if not v526_enabled else 2.0)) * expected_regret_loss
        + float(_m1(cfg, "V526_PREFERENCE_WEIGHT", 0.0 if not v526_enabled else 1.0)) * preference_loss
        + float(_m1(cfg, "V528_BASE_CONFUSION_WEIGHT", 0.0)) * v528_base_confusion_loss
        + float(_m1(cfg, "V528_OUTCOME_CORRECTNESS_WEIGHT", 0.0)) * v528_outcome_correctness_loss
        + float(_m1(cfg, "V528_OUTCOME_MASS_WEIGHT", 0.0)) * v528_outcome_mass_loss
        + float(_m1(cfg, "V528_GAIN_WEIGHT", 0.0)) * v528_gain_loss
        + float(_m1(cfg, "V528_GAIN_SIGN_WEIGHT", 0.0)) * v528_gain_sign_loss
        + float(_m1(cfg, "V528_EXPECTED_REGRET_WEIGHT", 0.0)) * v528_expected_regret_loss
        + float(_m1(cfg, "V528_HARM_MASS_WEIGHT", 0.0)) * v528_harm_mass_loss
        + float(_m1(cfg, "V528_HARD_PAIR_WEIGHT", 0.0)) * v528_hard_pair_loss
        + float(_m1(cfg, "V528_OUTCOME_NLL_WEIGHT", 0.0)) * v528_outcome_nll_loss
        + float(_m1(cfg, "V530_BASE_PIXEL_WEIGHT", 0.0)) * v530_base_pixel_loss
        + float(_m1(cfg, "V530_BASE_BRIER_WEIGHT", 0.0)) * v530_base_brier_loss
        + float(_m1(cfg, "V530_BASE_MASS_WEIGHT", 0.0)) * v530_base_mass_loss
        + float(_m1(cfg, "V530_OUTCOME_PIXEL_WEIGHT", 0.0)) * v530_outcome_pixel_loss
        + float(_m1(cfg, "V530_OUTCOME_BRIER_WEIGHT", 0.0)) * v530_outcome_brier_loss
        + float(_m1(cfg, "V530_RATE_WEIGHT", 0.0)) * v530_rate_loss
        + float(_m1(cfg, "V530_OUTCOME_MASS_WEIGHT", 0.0)) * v530_outcome_mass_loss
        + float(_m1(cfg, "V530_GAIN_WEIGHT", 0.0)) * v530_gain_loss
        + float(_m1(cfg, "V530_GAIN_SIGN_WEIGHT", 0.0)) * v530_gain_sign_loss
        + float(_m1(cfg, "V530_EXPECTED_REGRET_WEIGHT", 0.0)) * v530_expected_regret_loss
        + float(_m1(cfg, "V530_HARM_MASS_WEIGHT", 0.0)) * v530_harm_mass_loss
        + float(_m1(cfg, "V530_HARD_PAIR_WEIGHT", 0.0)) * v530_hard_pair_loss
        + float(_m1(cfg, "V530_OUTCOME_NLL_WEIGHT", 0.0)) * v530_outcome_nll_loss
        + float(_m1(cfg, "V529_BASE_ERROR_WEIGHT", 0.0)) * v529_base_error_loss
        + float(_m1(cfg, "V529_OUTCOME_PIXEL_WEIGHT", 0.0)) * v529_outcome_pixel_loss
        + float(_m1(cfg, "V529_OUTCOME_MASS_WEIGHT", 0.0)) * v529_outcome_mass_loss
        + float(_m1(cfg, "V529_GAIN_WEIGHT", 0.0)) * v529_gain_loss
        + float(_m1(cfg, "V529_GAIN_SIGN_WEIGHT", 0.0)) * v529_gain_sign_loss
        + float(_m1(cfg, "V529_EXPECTED_REGRET_WEIGHT", 0.0)) * v529_expected_regret_loss
        + float(_m1(cfg, "V529_HARM_MASS_WEIGHT", 0.0)) * v529_harm_mass_loss
        + float(_m1(cfg, "V529_HARD_PAIR_WEIGHT", 0.0)) * v529_hard_pair_loss
        + float(_m1(cfg, "V529_OUTCOME_NLL_WEIGHT", 0.0)) * v529_outcome_nll_loss
        + float(_m1(cfg, "V527_EDITABILITY_WEIGHT", 0.0)) * editability_loss
        + float(_m1(cfg, "V527_CONDITIONAL_TOP1_WEIGHT", 0.0)) * conditional_top1_loss
        + float(_m1(cfg, "V527_UTILITY_KL_WEIGHT", 0.0)) * utility_kl_loss
        + float(_m1(cfg, "V527_HARD_PAIR_WEIGHT", 0.0)) * hard_pair_loss
        + float(_m1(cfg, "V527_EXPECTED_REGRET_WEIGHT", 0.0)) * expected_regret_loss
        + float(_m1(cfg, "V527_PRESERVE_NOHARM_WEIGHT", 0.0)) * preserve_noharm_loss
        + float(_m1(cfg, "V527_VALUE_WEIGHT", 0.0)) * region_utility_loss
        + float(_m1(cfg, "V527_HARM_AUX_WEIGHT", 0.0)) * region_harm_loss
        + float(_m1(cfg, "V524_LISTWISE_WEIGHT", 2.00)) * listwise_loss
        + float(_m1(cfg, "V524_HARD_NEGATIVE_WEIGHT", 1.00)) * hard_negative_loss
        + float(_m1(cfg, "V524_REGION_UTILITY_WEIGHT", 0.50)) * region_utility_loss
        + float(_m1(cfg, "V524_REGION_HARM_WEIGHT", 0.50)) * region_harm_loss
        + float(_m1(cfg, "V524_REGION_NLL_WEIGHT", 0.10)) * region_nll_loss
        + float(_m1(cfg, "V524_ACCEPT_WEIGHT", 0.50)) * accept_bce
        + float(_m1(cfg, "V524_SELECTIVE_RISK_WEIGHT", 2.00)) * selective_risk_loss
        + float(_m1(cfg, "V524_COVERAGE_WEIGHT", 0.25)) * coverage_loss
        + float(_m1(cfg, "V524_REFINER_WEIGHT", 1.00)) * refiner_loss
        + float(_m1(cfg, "V524_DEPLOY_SEG_WEIGHT", 0.50)) * deploy_seg_loss
        + float(_m1(cfg, "V524_BOUNDARY_WEIGHT", 0.25)) * boundary_loss
        + float(_m1(cfg, "V524_CASE_NOHARM_WEIGHT", 1.00)) * case_noharm_loss
        + float(_m1(cfg, "V524_CASE_GAIN_WEIGHT", 1.00)) * case_gain_loss
        + float(_m1(cfg, "V524_CASE_HARM_WEIGHT", 2.00)) * case_harm_loss
        + float(_m1(cfg, "V524_CASE_NLL_WEIGHT", 0.25)) * case_nll_loss
    )

    epoch_value = int(epoch or 0)
    m1_scale = _v501_delayed_ramp(
        epoch_value,
        int(_m1(cfg, "V524_M1_START_EPOCH", 0)),
        int(_m1(cfg, "V524_M1_RAMP_EPOCHS", 1)),
        float(_m1(cfg, "V524_M1_FINAL_WEIGHT", 0.0)),
    )
    m2_scale = _v501_delayed_ramp(
        epoch_value,
        int(_m1(cfg, "V524_M2_START_EPOCH", 0)),
        int(_m1(cfg, "V524_M2_RAMP_EPOCHS", 10)),
        float(_m1(cfg, "V524_M2_FINAL_WEIGHT", 1.0)),
    )
    total = m1_scale * m1_loss + m2_scale * m2_loss

    # Region oracle: one candidate per grid region, but only on that candidate's
    # real valid support.  It diagnoses whether region construction preserves
    # enough of the original M1 candidate ceiling.
    def compose_region_oracle(
        utility: torch.Tensor, available: torch.Tensor
    ) -> torch.Tensor:
        best_utility, best_candidate = utility.masked_fill(
            ~available, floor
        ).max(dim=2)
        region_index = torch.where(
            best_utility > positive_margin,
            best_candidate + 1,
            torch.zeros_like(best_candidate),
        )
        route = torch.zeros_like(action_score).scatter_(
            2, region_index[..., None], 1.0
        )
        route_grid = route.transpose(1, 2).reshape(b, n + 1, grid, grid)
        pixel_route = F.interpolate(
            route_grid, size=(h, w), mode="nearest"
        )[:, 1:]
        oracle_candidate = (pixel_route * nonbase).sum(dim=1, keepdim=True)
        oracle_edit = (pixel_route * valid.float()).sum(
            dim=1, keepdim=True
        ).clamp(0.0, 1.0)
        return (c0 + oracle_edit * (oracle_candidate - c0)).clamp(
            EPS, 1.0 - EPS
        )

    region_oracle_prob = compose_region_oracle(
        raw_region_utility, geometric_region_valid
    )
    decision_region_oracle_prob = compose_region_oracle(
        region_utility, region_valid
    )

    base_dice = base_dice_case.mean()
    deployed_dice = _soft_dice_probs(deployed_prob, gt).mean()
    region_oracle_dice = _soft_dice_probs(region_oracle_prob, gt).mean()
    decision_region_oracle_dice = _soft_dice_probs(
        decision_region_oracle_prob, gt
    ).mean()
    region_oracle_gain = region_oracle_dice - base_dice
    decision_region_oracle_gain = decision_region_oracle_dice - base_dice
    deployed_gain = deployed_dice - base_dice
    capture_rate = torch.where(
        region_oracle_gain > 1.0e-6,
        deployed_gain / region_oracle_gain.clamp_min(1.0e-6),
        torch.zeros_like(deployed_gain),
    )
    proposal_gain = proposal_gain_case.mean()
    proposal_capture_rate = torch.where(
        decision_region_oracle_gain > 1.0e-6,
        proposal_gain / decision_region_oracle_gain.clamp_min(1.0e-6),
        torch.zeros_like(proposal_gain),
    )
    raw_proposal_capture_rate = torch.where(
        region_oracle_gain > 1.0e-6,
        proposal_gain / region_oracle_gain.clamp_min(1.0e-6),
        torch.zeros_like(proposal_gain),
    )

    if v530_enabled:
        normalized_score = aux["v530_normalized_score"].masked_fill(
            ~region_valid, floor
        )
        predicted_candidate_index = normalized_score.argmax(dim=2)
        predicted_true_utility = exact_dice_gain.gather(
            2, predicted_candidate_index[..., None]
        ).squeeze(-1)
        near_eps = float(_m1(cfg, "V530_NEAR_OPTIMAL_EPS", 2.0e-4))
        best_true = exact_dice_gain.masked_fill(~region_valid, floor).max(dim=2).values
        conditional_top1_correct = predicted_true_utility >= (best_true - near_eps)
        topk = min(3, n)
        top3_idx = normalized_score.topk(topk, dim=2).indices
        top3_true = exact_dice_gain.gather(2, top3_idx)
        conditional_top3_recall = (
            top3_true >= best_true[..., None] - near_eps
        ).any(dim=2)
        predicted_index = aux["v524_region_selected_index"]
        predicted_edit = predicted_index > 0
        top1_correct = torch.where(
            oracle_edits, conditional_top1_correct, ~predicted_edit
        )
        top3_recall = torch.where(
            oracle_edits, conditional_top3_recall, ~predicted_edit
        )
        edit_precision = total * 0.0
        edit_recall = total * 0.0
        preserve_false_edit_rate = total * 0.0
    elif v529_enabled:
        normalized_score = aux["v529_normalized_score"].masked_fill(
            ~region_valid, floor
        )
        predicted_candidate_index = normalized_score.argmax(dim=2)
        predicted_true_utility = exact_dice_gain.gather(
            2, predicted_candidate_index[..., None]
        ).squeeze(-1)
        near_eps = float(_m1(cfg, "V529_NEAR_OPTIMAL_EPS", 2.0e-4))
        best_true = exact_dice_gain.masked_fill(~region_valid, floor).max(dim=2).values
        conditional_top1_correct = predicted_true_utility >= (best_true - near_eps)
        topk = min(3, n)
        top3_idx = normalized_score.topk(topk, dim=2).indices
        top3_true = exact_dice_gain.gather(2, top3_idx)
        conditional_top3_recall = (
            top3_true >= best_true[..., None] - near_eps
        ).any(dim=2)
        predicted_index = aux["v524_region_selected_index"]
        predicted_edit = predicted_index > 0
        top1_correct = torch.where(
            oracle_edits, conditional_top1_correct, ~predicted_edit
        )
        top3_recall = torch.where(
            oracle_edits, conditional_top3_recall, ~predicted_edit
        )
        edit_precision = total * 0.0
        edit_recall = total * 0.0
        preserve_false_edit_rate = total * 0.0
    elif v528_enabled:
        predicted_index = aux["v524_region_selected_index"]
        predicted_edit = predicted_index > 0
        predicted_candidate_index = (predicted_index - 1).clamp_min(0)
        predicted_true_utility = exact_dice_gain.gather(
            2, predicted_candidate_index[..., None]
        ).squeeze(-1)
        conditional_top1_correct = predicted_true_utility >= (
            exact_dice_gain.masked_fill(~region_valid, floor).max(dim=2).values
            - float(_m1(cfg, "V528_NEAR_OPTIMAL_EPS", 2.0e-4))
        )
        topk = min(3, n)
        top3_idx = aux["v528_gain_lcb"].masked_fill(~region_valid, floor).topk(
            topk, dim=2
        ).indices
        top3_true = exact_dice_gain.gather(2, top3_idx)
        best_true = exact_dice_gain.masked_fill(~region_valid, floor).max(dim=2).values
        conditional_top3_recall = (
            top3_true >= best_true[..., None] - float(
                _m1(cfg, "V528_NEAR_OPTIMAL_EPS", 2.0e-4)
            )
        ).any(dim=2)
        top1_correct = predicted_index == oracle_index
        top3_recall = torch.where(
            oracle_edits, conditional_top3_recall, ~predicted_edit
        )
        edit_precision = total * 0.0
        edit_recall = total * 0.0
        preserve_false_edit_rate = total * 0.0
    elif v527_enabled:
        editability_prob = torch.sigmoid(aux["v527_region_editability_logit"])
        predicted_edit = editability_prob >= float(
            _m1(cfg, "V527_EDITABILITY_THRESHOLD", 0.50)
        )
        conditional_score = aux["v527_region_conditional_score"].masked_fill(
            ~region_valid, floor
        )
        predicted_candidate_index = conditional_score.argmax(dim=2)
        predicted_edit = predicted_edit & region_valid.any(dim=2)
        predicted_index = torch.where(
            predicted_edit,
            predicted_candidate_index + 1,
            torch.zeros_like(predicted_candidate_index),
        )
        conditional_top1_correct = predicted_candidate_index == best_candidate_index
        conditional_topk = min(3, conditional_score.shape[2])
        conditional_top3_index = conditional_score.topk(
            k=conditional_topk, dim=2
        ).indices
        conditional_top3_recall = (
            conditional_top3_index == best_candidate_index[..., None]
        ).any(dim=2)
        top1_correct = predicted_index == oracle_index
        top3_recall = torch.where(
            oracle_edits,
            conditional_top3_recall,
            ~predicted_edit,
        )
        edit_true_positive = predicted_edit & oracle_edits
        edit_precision = (
            edit_true_positive.float().sum()
            / predicted_edit.float().sum().clamp_min(1.0)
        )
        edit_recall = (
            edit_true_positive.float().sum()
            / oracle_edits.float().sum().clamp_min(1.0)
        )
        preserve_false_edit_rate = (
            (predicted_edit & (~oracle_edits)).float().sum()
            / (~oracle_edits).float().sum().clamp_min(1.0)
        )
    else:
        predicted_index = action_score.argmax(dim=2)
        top1_correct = predicted_index == oracle_index
        topk = min(3, action_score.shape[2])
        top3_index = action_score.topk(k=topk, dim=2).indices
        top3_recall = (top3_index == oracle_index[..., None]).any(dim=2)
        predicted_edit = predicted_index > 0
        predicted_candidate_index = (predicted_index - 1).clamp_min(0)
        conditional_top1_correct = top1_correct
        conditional_top3_recall = top3_recall
        edit_precision = total * 0.0
        edit_recall = total * 0.0
        preserve_false_edit_rate = total * 0.0

    predicted_utility = full_utility.gather(
        2, predicted_index[..., None]
    ).squeeze(-1)
    positive_action_precision = (
        ((predicted_utility > positive_margin) & predicted_edit).float().sum()
        / predicted_edit.float().sum().clamp_min(1.0)
    )
    causal_valid = aux.get("v526_region_causal_valid")
    if isinstance(causal_valid, torch.Tensor):
        chosen_causal_valid = causal_valid.bool().gather(
            2, predicted_candidate_index[..., None]
        ).squeeze(-1)
        causal_invalid_rate = (
            (predicted_edit & (~chosen_causal_valid)).float().sum()
            / predicted_edit.float().sum().clamp_min(1.0)
        )
    else:
        causal_invalid_rate = total * 0.0

    oracle_retention = torch.where(
        region_oracle_gain > 1.0e-6,
        decision_region_oracle_gain / region_oracle_gain.clamp_min(1.0e-6),
        torch.ones_like(region_oracle_gain),
    )

    pre_guard_hard = pre_guard_prob >= 0.5
    pre_guard_changed = pre_guard_hard != base_hard
    pre_guard_beneficial = (
        pre_guard_changed & base_wrong & (pre_guard_hard == gt_hard)
    )
    pre_guard_harmful = (
        pre_guard_changed & (~base_wrong) & (pre_guard_hard != gt_hard)
    )
    pre_guard_changed_mass = pre_guard_changed.float().sum().clamp_min(1.0)

    deployed_hard = deployed_prob >= 0.5
    changed = deployed_hard != base_hard
    beneficial = changed & base_wrong & (deployed_hard == gt_hard)
    harmful_changed = changed & (~base_wrong) & (deployed_hard != gt_hard)
    changed_mass = changed.float().sum().clamp_min(1.0)
    selected_region = aux["v524_region_selected_index"] > 0
    selected_region_positive = positive.gather(2, (aux["v524_region_selected_index"] - 1).clamp_min(0)[..., None]).squeeze(-1) & selected_region
    selected_region_harm = harmful.gather(2, (aux["v524_region_selected_index"] - 1).clamp_min(0)[..., None]).squeeze(-1) & selected_region
    selected_region_mass = selected_region.float().sum().clamp_min(1.0)
    zero = total * 0.0
    v529_rank_harmful_top1 = zero
    v529_rank_selected_true = zero
    v529_route_entropy = zero
    v529_preserve_probability = zero
    v530_rank_harmful_top1 = zero
    v530_rank_selected_true = zero
    v530_route_entropy = zero
    v530_preserve_probability = zero
    if v530_enabled:
        v530_score = aux["v530_normalized_score"].masked_fill(~region_valid, floor)
        v530_rank_index = v530_score.argmax(dim=2)
        v530_rank_harm = harmful.gather(
            2, v530_rank_index[..., None]
        ).squeeze(-1)
        v530_rank_domain = region_valid.any(dim=2)
        if bool(v530_rank_domain.any().item()):
            v530_rank_harmful_top1 = v530_rank_harm[
                v530_rank_domain
            ].float().mean()
            v530_rank_selected_true = exact_dice_gain.gather(
                2, v530_rank_index[..., None]
            ).squeeze(-1)[v530_rank_domain].mean()
        route_all = aux["v524_region_soft_route"].clamp_min(EPS)
        v530_route_entropy = -(route_all * torch.log(route_all)).sum(dim=2).mean()
        v530_preserve_probability = route_all[:, :, 0].mean()
    if v529_enabled:
        v529_score = aux["v529_normalized_score"].masked_fill(~region_valid, floor)
        v529_rank_index = v529_score.argmax(dim=2)
        v529_rank_harm = harmful.gather(
            2, v529_rank_index[..., None]
        ).squeeze(-1)
        v529_rank_domain = region_valid.any(dim=2)
        if bool(v529_rank_domain.any().item()):
            v529_rank_harmful_top1 = v529_rank_harm[
                v529_rank_domain
            ].float().mean()
            v529_rank_selected_true = exact_dice_gain.gather(
                2, v529_rank_index[..., None]
            ).squeeze(-1)[v529_rank_domain].mean()
        route_all = aux["v524_region_soft_route"].clamp_min(EPS)
        v529_route_entropy = -(route_all * torch.log(route_all)).sum(dim=2).mean()
        v529_preserve_probability = route_all[:, :, 0].mean()

    diag: Dict[str, torch.Tensor] = {
        **m1_diag,
        "v524_total_loss": total.detach(),
        "v524_m2_loss": m2_loss.detach(),
        "v524_family_teacher_loss": family_loss.detach(),
        "v524_prompt_loss": prompt_loss.detach(),
        "v524_dense_utility_loss": dense_utility_loss.detach(),
        "v524_dense_harm_loss": dense_harm_loss.detach(),
        "v524_dense_nll_loss": dense_nll_loss.detach(),
        "v524_region_listwise_loss": listwise_loss.detach(),
        "v524_hard_negative_loss": hard_negative_loss.detach(),
        "v526_top1_loss": top1_loss.detach(),
        "v526_utility_kl_loss": utility_kl_loss.detach(),
        "v526_expected_regret_loss": expected_regret_loss.detach(),
        "v526_preference_loss": preference_loss.detach(),
        "v527_editability_loss": editability_loss.detach(),
        "v527_conditional_top1_loss": conditional_top1_loss.detach(),
        "v527_hard_pair_loss": hard_pair_loss.detach(),
        "v527_preserve_noharm_loss": preserve_noharm_loss.detach(),
        "v524_region_utility_loss": region_utility_loss.detach(),
        "v524_region_harm_loss": region_harm_loss.detach(),
        "v524_region_nll_loss": region_nll_loss.detach(),
        "v524_accept_loss": accept_bce.detach(),
        "v524_selective_risk_loss": selective_risk_loss.detach(),
        "v524_coverage_loss": coverage_loss.detach(),
        "v524_refiner_loss": refiner_loss.detach(),
        "v524_deploy_seg_loss": deploy_seg_loss.detach(),
        "v524_boundary_loss": boundary_loss.detach(),
        "v524_case_noharm_loss": case_noharm_loss.detach(),
        "v524_case_gain_loss": case_gain_loss.detach(),
        "v524_case_harm_loss": case_harm_loss.detach(),
        "v524_case_nll_loss": case_nll_loss.detach(),
        "v524_proposal_gain": proposal_gain_case.mean().detach(),
        "v524_case_accept_rate": aux["v524_case_accept"].float().mean().detach(),
        "v524_teacher_repairable_region_rate": repairable.float().mean().detach(),
        "v524_teacher_positive_candidate_rate": positive.float().sum().div(region_valid.float().sum().clamp_min(1.0)).detach(),
        "v524_predicted_region_coverage": coverage.detach(),
        "v524_selective_risk": selective_risk.detach(),
        "v524_selected_region_benefit_rate": selected_region_positive.float().sum().div(selected_region_mass).detach(),
        "v524_selected_region_harm_rate": selected_region_harm.float().sum().div(selected_region_mass).detach(),
        "v524_base_dice": base_dice.detach(),
        "v524_deployed_dice": deployed_dice.detach(),
        "v524_deployed_gain": deployed_gain.detach(),
        "v524_region_oracle_dice": region_oracle_dice.detach(),
        "v524_region_oracle_gain": region_oracle_gain.detach(),
        "v524_region_oracle_capture_rate": capture_rate.detach(),
        "v526_decision_region_oracle_dice": decision_region_oracle_dice.detach(),
        "v526_decision_region_oracle_gain": decision_region_oracle_gain.detach(),
        "v526_proposal_oracle_capture_rate": proposal_capture_rate.detach(),
        "v526_raw_proposal_oracle_capture_rate": raw_proposal_capture_rate.detach(),
        "v526_oracle_action_top1_accuracy": top1_correct.float().mean().detach(),
        "v526_oracle_action_top1_accuracy_repairable": (
            top1_correct[oracle_edits].float().mean().detach()
            if bool(oracle_edits.any().item()) else zero.detach()
        ),
        "v526_oracle_action_top3_recall": top3_recall.float().mean().detach(),
        "v526_oracle_action_top3_recall_repairable": (
            top3_recall[oracle_edits].float().mean().detach()
            if bool(oracle_edits.any().item()) else zero.detach()
        ),
        "v526_preserve_accuracy": (
            top1_correct[~oracle_edits].float().mean().detach()
            if bool((~oracle_edits).any().item()) else zero.detach()
        ),
        "v526_expected_utility": expected_utility.mean().detach(),
        "v526_oracle_utility": oracle_utility.mean().detach(),
        "v526_expected_regret": expected_regret.mean().detach(),
        "v526_normalized_regret": (expected_regret.mean() / max(utility_scale, 1.0e-6)).detach(),
        "v526_positive_action_precision": positive_action_precision.detach(),
        "v526_causal_invalid_action_rate": causal_invalid_rate.detach(),
        "v527_oracle_retention": oracle_retention.detach(),
        "v527_editability_precision": edit_precision.detach(),
        "v527_editability_recall": edit_recall.detach(),
        "v527_preserve_false_edit_rate": preserve_false_edit_rate.detach(),
        "v527_conditional_top1_accuracy_repairable": (
            conditional_top1_correct[oracle_edits].float().mean().detach()
            if bool(oracle_edits.any().item()) else zero.detach()
        ),
        "v527_conditional_top3_recall_repairable": (
            conditional_top3_recall[oracle_edits].float().mean().detach()
            if bool(oracle_edits.any().item()) else zero.detach()
        ),
        "v527_pre_guard_changed_pixel_rate": pre_guard_changed.float().mean().detach(),
        "v527_pre_guard_benefit_rate": pre_guard_beneficial.float().sum().div(pre_guard_changed_mass).detach(),
        "v527_pre_guard_harm_rate": pre_guard_harmful.float().sum().div(pre_guard_changed_mass).detach(),
        "v528_base_confusion_loss": v528_base_confusion_loss.detach(),
        "v528_outcome_correctness_loss": v528_outcome_correctness_loss.detach(),
        "v528_outcome_mass_loss": v528_outcome_mass_loss.detach(),
        "v528_gain_loss": v528_gain_loss.detach(),
        "v528_gain_sign_loss": v528_gain_sign_loss.detach(),
        "v528_expected_regret_loss": v528_expected_regret_loss.detach(),
        "v528_harm_mass_loss": v528_harm_mass_loss.detach(),
        "v528_hard_pair_loss": v528_hard_pair_loss.detach(),
        "v528_outcome_nll_loss": v528_outcome_nll_loss.detach(),
        "v528_outcome_contract_error": v528_outcome_contract_error.detach(),
        "v528_expected_utility": v528_expected_utility_case.mean().detach(),
        "v528_oracle_utility": v528_oracle_utility_case.mean().detach(),
        "v528_predicted_gain_mean": aux.get(
            "v528_predicted_gain", exact_dice_gain * 0.0
        )[region_valid].mean().detach() if bool(region_valid.any().item()) else zero.detach(),
        "v528_gain_lcb_mean": aux.get(
            "v528_gain_lcb", exact_dice_gain * 0.0
        )[region_valid].mean().detach() if bool(region_valid.any().item()) else zero.detach(),
        "v528_predicted_harm_mean": aux.get(
            "v528_predicted_harm_fraction", exact_dice_gain * 0.0
        )[region_valid].mean().detach() if bool(region_valid.any().item()) else zero.detach(),
        "v528_near_optimal_top1": (
            conditional_top1_correct[repairable].float().mean().detach()
            if bool(repairable.any().item()) else zero.detach()
        ),
        "v528_near_optimal_top3": (
            conditional_top3_recall[repairable].float().mean().detach()
            if bool(repairable.any().item()) else zero.detach()
        ),
        "v528_harmful_top1_rate": (
            selected_region_harm.float().sum().div(selected_region_mass).detach()
        ),
        "v528_selected_true_utility": (
            predicted_true_utility[predicted_edit].mean().detach()
            if v528_enabled and bool(predicted_edit.any().item()) else zero.detach()
        ),
        "v530_base_pixel_loss": v530_base_pixel_loss.detach(),
        "v530_base_brier_loss": v530_base_brier_loss.detach(),
        "v530_base_mass_loss": v530_base_mass_loss.detach(),
        "v530_outcome_pixel_loss": v530_outcome_pixel_loss.detach(),
        "v530_outcome_brier_loss": v530_outcome_brier_loss.detach(),
        "v530_rate_loss": v530_rate_loss.detach(),
        "v530_outcome_mass_loss": v530_outcome_mass_loss.detach(),
        "v530_gain_loss": v530_gain_loss.detach(),
        "v530_gain_sign_loss": v530_gain_sign_loss.detach(),
        "v530_expected_regret_loss": v530_expected_regret_loss.detach(),
        "v530_harm_mass_loss": v530_harm_mass_loss.detach(),
        "v530_hard_pair_loss": v530_hard_pair_loss.detach(),
        "v530_outcome_nll_loss": v530_outcome_nll_loss.detach(),
        "v530_outcome_contract_error": v528_outcome_contract_error.detach(),
        "v530_expected_utility": v530_expected_utility_case.mean().detach(),
        "v530_oracle_utility": v530_oracle_utility_case.mean().detach(),
        "v530_predicted_gain_mean": (
            aux["v530_predicted_gain"][region_valid].mean().detach()
            if v530_enabled and bool(region_valid.any().item()) else zero.detach()
        ),
        "v530_predicted_harm_mean": (
            aux["v530_predicted_harm_fraction"][region_valid].mean().detach()
            if v530_enabled and bool(region_valid.any().item()) else zero.detach()
        ),
        "v530_base_dice_mae": v530_base_dice_mae.detach(),
        "v530_tp_mass_mae": v530_tp_mass_mae.detach(),
        "v530_fp_mass_mae": v530_fp_mass_mae.detach(),
        "v530_fn_mass_mae": v530_fn_mass_mae.detach(),
        "v530_add_correct_rate_mae": v530_add_rate_mae.detach(),
        "v530_remove_correct_rate_mae": v530_remove_rate_mae.detach(),
        "v530_gain_mae": v530_gain_mae.detach(),
        "v530_positive_gain_mae": v530_positive_gain_mae.detach(),
        "v530_harmful_gain_mae": v530_harmful_gain_mae.detach(),
        "v530_neutral_gain_mae": v530_neutral_gain_mae.detach(),
        "v530_gain_sign_accuracy": v530_gain_sign_accuracy.detach(),
        "v530_positive_gain_precision": v530_positive_precision.detach(),
        "v530_positive_gain_recall": v530_positive_recall.detach(),
        "v530_sigma_p50": v530_sigma_p50.detach(),
        "v530_sigma_p90": v530_sigma_p90.detach(),
        "v530_sigma_p99": v530_sigma_p99.detach(),
        "v530_near_optimal_top1": (
            conditional_top1_correct[repairable].float().mean().detach()
            if v530_enabled and bool(repairable.any().item()) else zero.detach()
        ),
        "v530_near_optimal_top3": (
            conditional_top3_recall[repairable].float().mean().detach()
            if v530_enabled and bool(repairable.any().item()) else zero.detach()
        ),
        "v530_harmful_top1_rate": v530_rank_harmful_top1.detach(),
        "v530_rank_selected_true_utility": v530_rank_selected_true.detach(),
        "v530_route_entropy": v530_route_entropy.detach(),
        "v530_preserve_probability": v530_preserve_probability.detach(),
        "v530_executed_selected_true_utility": (
            predicted_true_utility[predicted_edit].mean().detach()
            if v530_enabled and bool(predicted_edit.any().item()) else zero.detach()
        ),
        "v529_base_error_loss": v529_base_error_loss.detach(),
        "v529_outcome_pixel_loss": v529_outcome_pixel_loss.detach(),
        "v529_outcome_mass_loss": v529_outcome_mass_loss.detach(),
        "v529_gain_loss": v529_gain_loss.detach(),
        "v529_gain_sign_loss": v529_gain_sign_loss.detach(),
        "v529_expected_regret_loss": v529_expected_regret_loss.detach(),
        "v529_harm_mass_loss": v529_harm_mass_loss.detach(),
        "v529_hard_pair_loss": v529_hard_pair_loss.detach(),
        "v529_outcome_nll_loss": v529_outcome_nll_loss.detach(),
        "v529_outcome_contract_error": v528_outcome_contract_error.detach(),
        "v529_expected_utility": v529_expected_utility_case.mean().detach(),
        "v529_oracle_utility": v529_oracle_utility_case.mean().detach(),
        "v529_predicted_gain_mean": (
            aux["v529_predicted_gain"][region_valid].mean().detach()
            if v529_enabled and bool(region_valid.any().item()) else zero.detach()
        ),
        "v529_predicted_harm_mean": (
            aux["v529_predicted_harm_fraction"][region_valid].mean().detach()
            if v529_enabled and bool(region_valid.any().item()) else zero.detach()
        ),
        "v529_base_dice_mae": v529_base_dice_mae.detach(),
        "v529_tp_mass_mae": v529_tp_mass_mae.detach(),
        "v529_fp_mass_mae": v529_fp_mass_mae.detach(),
        "v529_fn_mass_mae": v529_fn_mass_mae.detach(),
        "v529_add_correct_rate_mae": v529_add_rate_mae.detach(),
        "v529_remove_correct_rate_mae": v529_remove_rate_mae.detach(),
        "v529_gain_mae": v529_gain_mae.detach(),
        "v529_gain_sign_accuracy": v529_gain_sign_accuracy.detach(),
        "v529_positive_gain_precision": v529_positive_precision.detach(),
        "v529_positive_gain_recall": v529_positive_recall.detach(),
        "v529_sigma_p50": v529_sigma_p50.detach(),
        "v529_sigma_p90": v529_sigma_p90.detach(),
        "v529_sigma_p99": v529_sigma_p99.detach(),
        "v529_near_optimal_top1": (
            conditional_top1_correct[repairable].float().mean().detach()
            if v529_enabled and bool(repairable.any().item()) else zero.detach()
        ),
        "v529_near_optimal_top3": (
            conditional_top3_recall[repairable].float().mean().detach()
            if v529_enabled and bool(repairable.any().item()) else zero.detach()
        ),
        "v529_harmful_top1_rate": v529_rank_harmful_top1.detach(),
        "v529_rank_selected_true_utility": v529_rank_selected_true.detach(),
        "v529_route_entropy": v529_route_entropy.detach(),
        "v529_preserve_probability": v529_preserve_probability.detach(),
        "v529_executed_selected_true_utility": (
            predicted_true_utility[predicted_edit].mean().detach()
            if v529_enabled and bool(predicted_edit.any().item()) else zero.detach()
        ),
        "v524_changed_pixel_rate": changed.float().mean().detach(),
        "v524_beneficial_edit_precision": beneficial.float().sum().div(changed_mass).detach(),
        "v524_conditional_benefit_rate": beneficial.float().sum().div(changed_mass).detach(),
        "v524_conditional_harm_rate": harmful_changed.float().sum().div(changed_mass).detach(),
        "v524_worst_case_regret": case_regret.max().detach(),
        "v524_deploy_phase": aux["v524_deploy_phase"].float().mean().detach(),
        # Existing logger/validation compatibility.
        "v505_m1_loss": m1_loss.detach(),
        "v505_m2_loss": m2_loss.detach(),
        "v505_m3_loss": zero.detach(),
        "v505_m1_scale": total.new_tensor(m1_scale),
        "v505_m2_scale": total.new_tensor(m2_scale),
        "v505_m3_scale": zero.detach(),
        "v505_base_dice": base_dice.detach(),
        "v505_m2_dice": deployed_dice.detach(),
        "v505_final_dice": deployed_dice.detach(),
        "v505_m2_gain_vs_base": deployed_gain.detach(),
        "v505_final_gain_vs_base": deployed_gain.detach(),
        "v505_m2_changed_pixel_rate": changed.float().mean().detach(),
        "v505_m2_conditional_benefit_rate": beneficial.float().sum().div(changed_mass).detach(),
        "v505_m2_conditional_harm_rate": harmful_changed.float().sum().div(changed_mass).detach(),
        "v505_selected_interaction_rate": selected_region.float().mean().detach(),
        "v505_m3_accept_rate": zero.detach(),
        "v489_total_loss": total.detach(),
        "v489_m1_loss": m1_loss.detach(),
        "v489_m2_loss": m2_loss.detach(),
        "v489_m3_loss": zero.detach(),
        "v489_base_dice": base_dice.detach(),
        "v489_m2_dice": deployed_dice.detach(),
        "v489_final_dice": deployed_dice.detach(),
        "v489_m2_gain_vs_base": deployed_gain.detach(),
        "v489_final_gain_vs_m2": zero.detach(),
        "v488_total_loss": total.detach(),
        "v488_base_dice": base_dice.detach(),
        "v488_m2_dice": deployed_dice.detach(),
        "v488_m3_final_dice": deployed_dice.detach(),
        "v488_m2_gain_vs_base": deployed_gain.detach(),
        "v488_m3_gain_vs_m2": zero.detach(),
        "v484_m2_loss": m2_loss.detach(),
        "v484_m3_regret_loss": zero.detach(),
        "_v490_m1_objective": m1_scale * m1_loss,
        "_v490_m2_objective": m2_scale * m2_loss,
        "_v490_m3_objective": zero,
    }
    return total, diag


def _v532_ohem_binary_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    negative_ratio: int,
    min_negatives: int,
    positive_weight: float = 1.0,
    negative_weight: float = 1.0,
) -> torch.Tensor:
    """Positive + hard-negative BCE with configurable precision/recall cost."""
    logits = logits.float()
    target = target.to(logits.dtype).clamp(0.0, 1.0)
    element = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    positive = target > 0.5
    negative = ~positive
    terms = []
    weights = []
    if bool(positive.any().item()):
        terms.append(element[positive].mean())
        weights.append(max(float(positive_weight), 0.0))
        positive_count = int(positive.sum().detach().cpu())
    else:
        positive_count = 0
    if bool(negative.any().item()):
        requested = max(
            int(min_negatives),
            int(max(positive_count, 1) * max(int(negative_ratio), 1)),
        )
        negative_loss = element[negative]
        keep = min(requested, int(negative_loss.numel()))
        if keep > 0:
            terms.append(torch.topk(negative_loss, k=keep, largest=True).values.mean())
            weights.append(max(float(negative_weight), 0.0))
    if not terms:
        return logits.sum() * 0.0
    weight_tensor = logits.new_tensor(weights)
    if float(weight_tensor.sum().detach().cpu()) <= 0.0:
        return torch.stack(terms).mean()
    return (torch.stack(terms) * weight_tensor).sum() / weight_tensor.sum()


def _v532_minimal_alpha_target(
    c0: torch.Tensor,
    cause_target: torch.Tensor,
    probability_margin: float,
) -> torch.Tensor:
    """Minimum monotone movement needed to cross the hard decision safely."""
    c0 = _as_b1hw(c0).detach().clamp(EPS, 1.0 - EPS)
    margin = min(max(float(probability_margin), 0.0), 0.49)
    low = 0.5 - margin
    high = 0.5 + margin
    negative_needed = ((c0 - low) / c0.clamp_min(EPS)).clamp(0.0, 1.0)
    positive_needed = ((high - c0) / (1.0 - c0).clamp_min(EPS)).clamp(0.0, 1.0)
    target = torch.cat(
        [negative_needed, positive_needed, negative_needed, positive_needed],
        dim=1,
    )
    return (target * cause_target).detach()


def _v532_balanced_outcome_loss(
    outcome_logits: torch.Tensor,
    outcome_target: torch.Tensor,
    domain_weight: torch.Tensor,
) -> torch.Tensor:
    """Average Neutral/Benefit/Harm class means when each class is present."""
    if outcome_logits.ndim != 5 or outcome_logits.shape[2] != 3:
        raise ValueError(
            "V532 outcome logits must be [B,4,3,H,W], got "
            + str(tuple(outcome_logits.shape))
        )
    b, actions, _, h, w = outcome_logits.shape
    logits = outcome_logits.reshape(b * actions, 3, h, w)
    target = outcome_target.reshape(b * actions, h, w).long()
    weight = domain_weight.reshape(b * actions, h, w).to(logits.dtype)
    element = F.cross_entropy(logits, target, reduction="none")
    terms = []
    for outcome_class in range(3):
        membership = weight * (target == outcome_class).to(weight.dtype)
        if bool((membership.sum() > 0).item()):
            terms.append(
                (element * membership).sum() / membership.sum().clamp_min(1.0)
            )
    return torch.stack(terms).mean() if terms else outcome_logits.sum() * 0.0


def _v533_ste_hard_mask(probability: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Hard-forward mask with identity-gradient surrogate."""
    hard = (probability >= float(threshold)).to(probability.dtype)
    return hard.detach() - probability.detach() + probability


def _v533_binary_prf(
    prediction: torch.Tensor,
    target: torch.Tensor,
    domain: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prediction = prediction.bool() & domain.bool()
    target = target.bool() & domain.bool()
    tp = (prediction & target).float().sum()
    fp = (prediction & (~target) & domain.bool()).float().sum()
    fn = ((~prediction) & target).float().sum()
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(EPS)
    return precision, recall, f1


def _v535_sparse_channel_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Sparse Dice that does not let empty channels dominate learning.

    Non-empty channel/case pairs receive Dice supervision.  Empty pairs receive
    only a false-positive mass penalty.  The two terms are averaged when both
    exist, so no class-frequency weight needs to be tuned.
    """
    prediction = prediction.clamp(EPS, 1.0 - EPS)
    target = target.clamp(0.0, 1.0)
    pred_sum = prediction.flatten(2).sum(dim=2)
    target_sum = target.flatten(2).sum(dim=2)
    intersection = (prediction * target).flatten(2).sum(dim=2)
    nonempty = target_sum > 0.0
    terms = []
    if bool(nonempty.any().item()):
        dice_loss = 1.0 - (
            2.0 * intersection + EPS
        ) / (pred_sum + target_sum + EPS)
        terms.append(dice_loss[nonempty].mean())
    empty = ~nonempty
    if bool(empty.any().item()):
        spatial_size = float(prediction.shape[-2] * prediction.shape[-1])
        false_positive_mass = pred_sum / max(spatial_size, 1.0)
        terms.append(false_positive_mass[empty].mean())
    return torch.stack(terms).mean() if terms else prediction.sum() * 0.0


def _v535_adaptive_policy_objective(
    policy_logits: torch.Tensor,
    policy_target: torch.Tensor,
    action_utility_logits: torch.Tensor,
    action_utility_target: torch.Tensor,
    harm_mask: torch.Tensor,
    utility_domain: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Direct Preserve+four-action objective with no execute threshold.

    All non-Preserve targets are retained.  Preserve pixels are selected by
    online hard-negative mining in a one-to-one ratio with positive targets.
    Batches without a beneficial action do not push the policy further toward
    all-Preserve; safety losses still handle harmful actions.
    """
    if policy_logits.ndim != 4 or policy_logits.shape[1] != 5:
        raise ValueError(
            "V535 policy logits must be [B,5,H,W], got "
            + str(tuple(policy_logits.shape))
        )
    policy_target = policy_target.long()
    element = F.cross_entropy(policy_logits, policy_target, reduction="none")
    positive = policy_target > 0
    preserve = ~positive
    positive_count = int(positive.sum().detach().cpu())
    if positive_count > 0:
        positive_term = element[positive].mean()
        preserve_loss = element[preserve]
        keep = min(max(positive_count, 1), int(preserve_loss.numel()))
        if keep > 0:
            preserve_term = torch.topk(
                preserve_loss, k=keep, largest=True
            ).values.mean()
            policy_ce = 0.5 * (positive_term + preserve_term)
        else:
            policy_ce = positive_term
    else:
        policy_ce = policy_logits.sum() * 0.0

    utility_prediction = torch.tanh(action_utility_logits)
    utility_element = F.smooth_l1_loss(
        utility_prediction,
        action_utility_target,
        beta=0.10,
        reduction="none",
    )
    domain = utility_domain > 0.5
    utility_terms = []
    for value in (-1.0, 0.0, 1.0):
        membership = domain & (action_utility_target == value)
        if bool(membership.any().item()):
            utility_terms.append(utility_element[membership].mean())
    utility_loss = (
        torch.stack(utility_terms).mean()
        if utility_terms
        else action_utility_logits.sum() * 0.0
    )

    policy_probability = F.softmax(policy_logits, dim=1)
    action_probability = policy_probability[:, 1:]
    harm_weight = harm_mask.to(action_probability.dtype) * utility_domain
    harm_loss = (
        action_probability * harm_weight
    ).sum() / harm_weight.sum().clamp_min(1.0)

    expected_utility = (
        action_probability * action_utility_target
    ).sum(dim=1)
    active_domain = (utility_domain.sum(dim=1) > 0.0).to(expected_utility.dtype)
    regret_loss = (
        F.relu(-expected_utility) * active_domain
    ).sum() / active_domain.sum().clamp_min(1.0)

    objective = policy_ce + utility_loss + harm_loss + regret_loss
    prediction = policy_logits.argmax(dim=1)
    policy_accuracy = (prediction == policy_target).float().mean()
    positive_accuracy = (
        (prediction[positive] == policy_target[positive]).float().mean()
        if positive_count > 0
        else objective.new_zeros(())
    )
    execute_probability = 1.0 - policy_probability[:, 0]
    entropy = -(
        policy_probability.clamp_min(EPS)
        * policy_probability.clamp_min(EPS).log()
    ).sum(dim=1).mean()
    selected_utility = torch.cat(
        [torch.zeros_like(action_utility_logits[:, :1]), action_utility_logits],
        dim=1,
    ).gather(1, prediction[:, None]).mean()
    diagnostics = {
        "v535_policy_objective": objective.detach(),
        "v535_policy_ce": policy_ce.detach(),
        "v535_utility_loss": utility_loss.detach(),
        "v535_policy_harm_loss": harm_loss.detach(),
        "v535_policy_regret_loss": regret_loss.detach(),
        "v535_policy_target_execute_rate": positive.float().mean().detach(),
        "v535_policy_predicted_execute_rate": (prediction > 0).float().mean().detach(),
        "v535_policy_soft_execute_rate": execute_probability.mean().detach(),
        "v535_policy_accuracy": policy_accuracy.detach(),
        "v535_positive_action_accuracy": positive_accuracy.detach(),
        "v535_policy_entropy": entropy.detach(),
        "v535_selected_utility": selected_utility.detach(),
    }
    return objective, diagnostics




def _binary_dice_numpy(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.astype(bool, copy=False)
    target = target.astype(bool, copy=False)
    intersection = float(np.logical_and(prediction, target).sum())
    denominator = float(prediction.sum() + target.sum())
    if denominator <= 0.0:
        return 1.0
    return (2.0 * intersection) / denominator


def _v536_best_component_targets(
    base_hard: torch.Tensor,
    action_candidates: torch.Tensor,
    gt: torch.Tensor,
    *,
    min_pixels: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build deployment-aligned Top-1 component targets.

    For each case and action, enumerate connected hard-mask changes and compute
    the *actual case Dice gain* obtained by applying only that component to the
    Base mask.  The case target therefore matches V536 deployment (one
    component), unlike the old whole-action candidate target.

    Returns:
      case_action_gain: [B,4], best component Dice gain for each action;
      component_policy_target: [B,H,W], Preserve=0 / action=1..4 for every
        action's best positive component, with overlaps resolved by larger gain;
      positive_component_mask: [B,4,H,W], best positive component per action.
    """
    if ndimage is None:
        raise RuntimeError(
            "V536 component targets require scipy.ndimage; original error: "
            + repr(_V536_SCIPY_IMPORT_ERROR)
        )
    base = (_as_b1hw(base_hard).detach() >= 0.5).cpu().numpy()[:, 0]
    candidates = (
        action_candidates.detach() >= 0.5
    ).cpu().numpy()
    target = (_as_b1hw(gt).detach() >= 0.5).cpu().numpy()[:, 0]
    b, actions, h, w = candidates.shape
    gains = np.zeros((b, actions), dtype=np.float32)
    positive_masks = np.zeros((b, actions, h, w), dtype=np.float32)
    target_class = np.zeros((b, h, w), dtype=np.int64)
    target_gain = np.full((b, h, w), -np.inf, dtype=np.float32)

    for sample in range(b):
        base_mask = base[sample]
        gt_mask = target[sample]
        base_dice = _binary_dice_numpy(base_mask, gt_mask)
        for action in range(actions):
            candidate = candidates[sample, action]
            changed = np.logical_xor(candidate, base_mask)
            labels, count = ndimage.label(changed)
            if count <= 0:
                gains[sample, action] = 0.0
                continue

            best_gain = -1.0
            best_component = None
            for label_id in range(1, count + 1):
                component = labels == label_id
                if int(component.sum()) < max(int(min_pixels), 1):
                    continue
                proposal = base_mask.copy()
                proposal[component] = candidate[component]
                delta = _binary_dice_numpy(proposal, gt_mask) - base_dice
                if delta > best_gain:
                    best_gain = float(delta)
                    best_component = component

            if best_component is None:
                gains[sample, action] = 0.0
                continue
            gains[sample, action] = best_gain
            if best_gain > 0.0:
                positive_masks[sample, action, best_component] = 1.0
                update = np.logical_and(best_component, best_gain > target_gain[sample])
                target_class[sample, update] = action + 1
                target_gain[sample, update] = best_gain

    device = action_candidates.device
    dtype = action_candidates.dtype
    return (
        torch.from_numpy(gains).to(device=device, dtype=dtype),
        torch.from_numpy(target_class).to(device=device, dtype=torch.long),
        torch.from_numpy(positive_masks).to(device=device, dtype=dtype),
    )




def _v537_component_gain_targets(
    base_hard: torch.Tensor,
    action_candidates: torch.Tensor,
    candidate_masks: torch.Tensor,
    candidate_actions: torch.Tensor,
    candidate_valid: torch.Tensor,
    gt: torch.Tensor,
) -> torch.Tensor:
    """Exact hard-Dice gain for each deployed action-component candidate."""
    base = (_as_b1hw(base_hard).detach() >= 0.5)
    target = (_as_b1hw(gt).detach() >= 0.5)
    candidate_hard = action_candidates.detach() >= 0.5
    masks = candidate_masks.detach() >= 0.5
    actions = candidate_actions.detach().clamp(0, 3)
    valid = candidate_valid.detach().bool()
    b, k, h, w = masks.shape
    gather_index = actions[:, :, None, None, None].expand(b, k, 1, h, w)
    selected_action = candidate_hard[:, None].expand(b, k, 4, h, w).gather(
        2, gather_index
    ).squeeze(2)
    base_expand = base[:, 0][:, None].expand(b, k, h, w)
    proposal = torch.where(masks, selected_action, base_expand)
    target_expand = target[:, 0][:, None].expand_as(proposal)

    def _dice(value: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        value = value.to(action_candidates.dtype)
        truth = truth.to(action_candidates.dtype)
        inter = (value * truth).flatten(2).sum(dim=2)
        den = value.flatten(2).sum(dim=2) + truth.flatten(2).sum(dim=2)
        return (2.0 * inter + EPS) / (den + EPS)

    proposal_dice = _dice(proposal, target_expand)
    base_dice = _dice(base_expand[:, :1], target_expand[:, :1])[:, 0:1]
    gains = proposal_dice - base_dice
    return torch.where(valid, gains, torch.zeros_like(gains)).detach()


def _v537_pwo_candidate_losses(
    action_candidates: torch.Tensor,
    c0: torch.Tensor,
    factual_error: torch.Tensor,
    cause_target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Directly align the M1 proposal union with the PWO residual error map.

    Coverage prevents missing most PWO pixels; purity prevents the trivial
    whole-image proposal.  Typed precision/recall are computed on the same
    monotone action changes used by deployment rather than on an unrelated map.
    """
    change = (action_candidates - c0.expand_as(action_candidates)).abs().clamp(0.0, 1.0)
    union = change.amax(dim=1, keepdim=True)
    error = factual_error.to(union.dtype)
    overlap = (union * error).sum()
    coverage = overlap / error.sum().clamp_min(1.0)
    purity = overlap / union.sum().clamp_min(1.0)
    coverage_loss = 1.0 - coverage
    purity_loss = 1.0 - purity

    hard_base = c0.detach() >= 0.5
    hard_action = action_candidates.detach() >= 0.5
    hard_union = (hard_action != hard_base.expand_as(hard_action)).any(dim=1, keepdim=True)
    hard_overlap = (hard_union.to(error.dtype) * error).sum()
    hard_coverage = hard_overlap / error.sum().clamp_min(1.0)
    hard_purity = hard_overlap / hard_union.to(error.dtype).sum().clamp_min(1.0)

    typed_overlap = (change * cause_target).flatten(2).sum(dim=2)
    typed_coverage = typed_overlap / cause_target.flatten(2).sum(dim=2).clamp_min(1.0)
    typed_purity = typed_overlap / change.flatten(2).sum(dim=2).clamp_min(1.0)
    typed_present = cause_target.flatten(2).sum(dim=2) > 0.0
    typed_coverage_mean = (
        typed_coverage[typed_present].mean()
        if bool(typed_present.any().item()) else union.new_zeros(())
    )
    proposed = change.flatten(2).sum(dim=2) > 0.0
    typed_purity_mean = (
        typed_purity[proposed].mean()
        if bool(proposed.any().item()) else union.new_ones(())
    )
    diag = {
        "v537_pwo_candidate_coverage": coverage.detach(),
        "v537_pwo_candidate_purity": purity.detach(),
        "v537_hard_candidate_coverage": hard_coverage.detach(),
        "v537_hard_candidate_purity": hard_purity.detach(),
        "v537_typed_candidate_coverage": typed_coverage_mean.detach(),
        "v537_typed_candidate_purity": typed_purity_mean.detach(),
        "v537_pwo_error_fraction": error.mean().detach(),
        "v537_candidate_union_fraction": union.mean().detach(),
    }
    return coverage_loss, purity_loss, diag




def _v537_gt_error_component_oracle(
    base_hard: torch.Tensor,
    gt: torch.Tensor,
    *,
    top_k: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Oracle gains from exact GT FP/FN connected components.

    This diagnoses whether the residual error is genuinely expressible by a
    small number of local regions, independently of the learned M1 proposals.
    """
    if ndimage is None:
        raise RuntimeError(
            "V537 oracle ladder requires scipy.ndimage; original error: "
            + repr(_V536_SCIPY_IMPORT_ERROR)
        )
    base_np = (_as_b1hw(base_hard).detach() >= 0.5).cpu().numpy()[:, 0]
    gt_np = (_as_b1hw(gt).detach() >= 0.5).cpu().numpy()[:, 0]
    one = []
    many = []
    structure = np.ones((3, 3), dtype=np.uint8)
    for base_mask, target_mask in zip(base_np, gt_np):
        base_dice = _binary_dice_numpy(base_mask, target_mask)
        components = []
        for error in (
            np.logical_and(base_mask, np.logical_not(target_mask)),
            np.logical_and(np.logical_not(base_mask), target_mask),
        ):
            labels, count = ndimage.label(error, structure=structure)
            for label_id in range(1, count + 1):
                component = labels == label_id
                proposal = base_mask.copy()
                proposal[component] = target_mask[component]
                gain = _binary_dice_numpy(proposal, target_mask) - base_dice
                components.append((float(gain), component))
        components.sort(key=lambda item: item[0], reverse=True)
        one.append(max(0.0, components[0][0]) if components else 0.0)
        proposal = base_mask.copy()
        for gain, component in components[: max(int(top_k), 1)]:
            if gain <= 0.0:
                continue
            proposal[component] = target_mask[component]
        many.append(max(0.0, _binary_dice_numpy(proposal, target_mask) - base_dice))
    device = base_hard.device
    dtype = gt.dtype
    return (
        torch.tensor(one, device=device, dtype=dtype),
        torch.tensor(many, device=device, dtype=dtype),
    )


def _v537_m1_component_topk_gain(
    *,
    base_hard: torch.Tensor,
    action_candidates: torch.Tensor,
    candidate_masks: torch.Tensor,
    candidate_actions: torch.Tensor,
    candidate_valid: torch.Tensor,
    candidate_gains: torch.Tensor,
    gt: torch.Tensor,
    top_k: int = 3,
) -> torch.Tensor:
    """Apply the top individual positive M1 components and measure exact gain."""
    base = (_as_b1hw(base_hard).detach() >= 0.5)
    target = (_as_b1hw(gt).detach() >= 0.5)
    candidate_hard = action_candidates.detach() >= 0.5
    masks = candidate_masks.detach() >= 0.5
    actions = candidate_actions.detach().clamp(0, 3)
    valid = candidate_valid.detach().bool()
    b, k, h, w = masks.shape
    selected_count = min(max(int(top_k), 1), k)
    safe_gains = candidate_gains.masked_fill(~valid, -1.0e4)
    top_values, top_indices = torch.topk(safe_gains, k=selected_count, dim=1)
    current = base[:, 0].clone()
    for rank in range(selected_count):
        index = top_indices[:, rank]
        gain = top_values[:, rank]
        mask = masks.gather(1, index[:, None, None, None].expand(b, 1, h, w))[:, 0]
        action = actions.gather(1, index[:, None])[:, 0]
        action_value = candidate_hard.gather(
            1, action[:, None, None, None].expand(b, 1, h, w)
        )[:, 0]
        apply = (gain > 0.0)[:, None, None] & mask
        current = torch.where(apply, action_value, current)

    current_float = current.to(gt.dtype)[:, None]
    base_float = base.to(gt.dtype)
    return (
        _soft_dice_probs(current_float, target.to(gt.dtype))[:, 0]
        - _soft_dice_probs(base_float, target.to(gt.dtype))[:, 0]
    ).clamp_min(0.0).detach()

def _v537_component_utility_objective(
    *,
    scores: torch.Tensor,
    candidate_masks: torch.Tensor,
    candidate_actions: torch.Tensor,
    candidate_valid: torch.Tensor,
    base_hard: torch.Tensor,
    action_candidates: torch.Tensor,
    gt: torch.Tensor,
    gain_epsilon: float,
    regression_beta: float,
    pair_temperature: float,
    preserve_margin: float,
    regression_weight: float,
    pair_weight: float,
    preserve_weight: float,
    oracle_ladder_diagnostics: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Gain regression + within-case ordering + explicit Preserve boundary."""
    gains = _v537_component_gain_targets(
        base_hard,
        action_candidates,
        candidate_masks,
        candidate_actions,
        candidate_valid,
        gt,
    )
    valid = candidate_valid.bool()
    gain_epsilon = max(float(gain_epsilon), 0.0)
    positive = valid & (gains > gain_epsilon)
    negative = valid & (gains < -gain_epsilon)
    neutral = valid & (~positive) & (~negative)

    regression_map = F.smooth_l1_loss(
        scores,
        gains,
        beta=max(float(regression_beta), 1.0e-6),
        reduction="none",
    )
    regression_terms = []
    for group in (positive, negative, neutral):
        if bool(group.any().item()):
            regression_terms.append(regression_map[group].mean())
    regression_loss = (
        torch.stack(regression_terms).mean()
        if regression_terms else scores.sum() * 0.0
    )

    # Pair every candidate with all other candidates and with Preserve=0.
    preserve_score = scores.new_zeros((scores.shape[0], 1))
    preserve_gain = gains.new_zeros((gains.shape[0], 1))
    preserve_valid = torch.ones((valid.shape[0], 1), device=valid.device, dtype=torch.bool)
    all_scores = torch.cat([preserve_score, scores], dim=1)
    all_gains = torch.cat([preserve_gain, gains], dim=1)
    all_valid = torch.cat([preserve_valid, valid], dim=1)
    pair_terms = []
    pair_correct_terms = []
    temperature = max(float(pair_temperature), 1.0e-6)
    for sample in range(scores.shape[0]):
        score_difference = all_scores[sample, :, None] - all_scores[sample, None, :]
        gain_difference = all_gains[sample, :, None] - all_gains[sample, None, :]
        pair_valid = all_valid[sample, :, None] & all_valid[sample, None, :]
        upper = torch.triu(torch.ones_like(pair_valid), diagonal=1)
        informative = pair_valid & upper & (gain_difference.abs() > gain_epsilon)
        if bool(informative.any().item()):
            sign = gain_difference[informative].sign()
            predicted_difference = score_difference[informative]
            pair_terms.append(F.softplus(-sign * predicted_difference / temperature).mean())
            pair_correct_terms.append(
                ((predicted_difference * sign) > 0.0).float().mean()
            )
    pair_loss = torch.stack(pair_terms).mean() if pair_terms else scores.sum() * 0.0
    pair_accuracy = (
        torch.stack(pair_correct_terms).mean()
        if pair_correct_terms else scores.new_zeros(())
    )

    margin = max(float(preserve_margin), 0.0)
    preserve_terms = []
    if bool(positive.any().item()):
        preserve_terms.append(F.relu(margin - scores[positive]).mean())
    if bool(negative.any().item()):
        preserve_terms.append(F.relu(margin + scores[negative]).mean())
    if bool(neutral.any().item()):
        preserve_terms.append(0.25 * scores[neutral].abs().mean())
    preserve_loss = (
        torch.stack(preserve_terms).mean()
        if preserve_terms else scores.sum() * 0.0
    )

    objective = (
        float(regression_weight) * regression_loss
        + float(pair_weight) * pair_loss
        + float(preserve_weight) * preserve_loss
    )

    masked_scores = scores.masked_fill(~valid, -1.0e4)
    best_score, best_index = masked_scores.max(dim=1)
    has_candidate = valid.any(dim=1)
    predicted_execute = has_candidate & (best_score > 0.0)
    best_gain, oracle_index = gains.masked_fill(~valid, -1.0e4).max(dim=1)
    positive_case = has_candidate & (best_gain > gain_epsilon)
    selected_gain = gains.gather(1, best_index[:, None])[:, 0]
    selected_gain = torch.where(predicted_execute, selected_gain, torch.zeros_like(selected_gain))
    oracle_gain = best_gain.clamp_min(0.0)

    tp = (predicted_execute & positive_case).float().sum()
    fp = (predicted_execute & (~positive_case)).float().sum()
    fn = ((~predicted_execute) & positive_case).float().sum()
    execute_precision = tp / (tp + fp).clamp_min(1.0)
    execute_recall = tp / (tp + fn).clamp_min(1.0)
    top1_hit = (
        (best_index[positive_case] == oracle_index[positive_case]).float().mean()
        if bool(positive_case.any().item()) else scores.new_zeros(())
    )
    positive_above_preserve = (
        (scores[positive] > 0.0).float().mean()
        if bool(positive.any().item()) else scores.new_zeros(())
    )
    negative_below_preserve = (
        (scores[negative] < 0.0).float().mean()
        if bool(negative.any().item()) else scores.new_ones(())
    )
    base_float = (_as_b1hw(base_hard).detach() >= 0.5).to(gt.dtype)
    base_dice_case = _soft_dice_probs(base_float, _as_b1hw(gt).detach())[:, 0]
    pwo_gain_case = (1.0 - base_dice_case).clamp_min(0.0)
    gt_component1 = scores.new_zeros(scores.shape[0])
    gt_component3 = scores.new_zeros(scores.shape[0])
    m1_component3 = scores.new_zeros(scores.shape[0])
    if bool(oracle_ladder_diagnostics):
        gt_component1, gt_component3 = _v537_gt_error_component_oracle(
            base_hard, gt, top_k=3
        )
        m1_component3 = _v537_m1_component_topk_gain(
            base_hard=base_hard,
            action_candidates=action_candidates,
            candidate_masks=candidate_masks,
            candidate_actions=candidate_actions,
            candidate_valid=candidate_valid,
            candidate_gains=gains,
            gt=gt,
            top_k=3,
        )
    diagnostics = {
        "v537_component_objective": objective.detach(),
        "v537_gain_regression_loss": regression_loss.detach(),
        "v537_pairwise_loss": pair_loss.detach(),
        "v537_preserve_margin_loss": preserve_loss.detach(),
        "v537_pairwise_accuracy": pair_accuracy.detach(),
        "v537_candidate_count": valid.float().sum(dim=1).mean().detach(),
        "v537_positive_candidate_rate": (
            positive.float().sum() / valid.float().sum().clamp_min(1.0)
        ).detach(),
        "v537_target_execute_case_rate": positive_case.float().mean().detach(),
        "v537_predicted_execute_case_rate": predicted_execute.float().mean().detach(),
        "v537_execute_precision": execute_precision.detach(),
        "v537_execute_recall": execute_recall.detach(),
        "v537_top1_oracle_hit_rate": top1_hit.detach(),
        "v537_positive_above_preserve_rate": positive_above_preserve.detach(),
        "v537_negative_below_preserve_rate": negative_below_preserve.detach(),
        "v537_component_oracle_gain": oracle_gain.mean().detach(),
        "v537_m1_component_top3_oracle_gain": m1_component3.mean().detach(),
        "v537_gt_component_top1_oracle_gain": gt_component1.mean().detach(),
        "v537_gt_component_top3_oracle_gain": gt_component3.mean().detach(),
        "v537_pwo_hard_gain": pwo_gain_case.mean().detach(),
        "v537_local_expressibility_ratio": (
            gt_component3.mean() / pwo_gain_case.mean().clamp_min(EPS)
        ).detach(),
        "v537_m1_candidate_capture_ratio": (
            oracle_gain.mean() / gt_component1.mean().clamp_min(EPS)
        ).detach(),
        "v537_m1_top3_candidate_capture_ratio": (
            m1_component3.mean() / gt_component3.mean().clamp_min(EPS)
        ).detach(),
        "v537_selected_gain": selected_gain.mean().detach(),
        "v537_selector_capture_ratio": (
            selected_gain.mean().clamp_min(0.0) / oracle_gain.mean().clamp_min(EPS)
        ).detach(),
        "v537_improved_case_rate": (selected_gain > gain_epsilon).float().mean().detach(),
        "v537_harmful_case_rate": (selected_gain < -gain_epsilon).float().mean().detach(),
        "v537_best_candidate_gain": best_gain.mean().detach(),
        "v537_best_predicted_score": best_score.mean().detach(),
    }
    return objective, diagnostics

def _v536_prior_aligned_objective(
    policy_logits: torch.Tensor,
    policy_target: torch.Tensor,
    action_utility_logits: torch.Tensor,
    action_utility_target: torch.Tensor,
    harm_mask: torch.Tensor,
    utility_domain: torch.Tensor,
    case_policy_logits: torch.Tensor,
    case_action_gain: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Prior-aligned pixel policy plus case-level no-harm selection.

    Unlike V535, Preserve examples are not resampled to a synthetic 1:1 prior.
    Pixel CE is evaluated on the complete deployment-relevant domain.  A second
    five-way case policy is supervised by the actual whole-case Dice gain of
    the four action candidates.  Negative-only batches do not keep pushing the
    case head toward all-Preserve after residual training errors disappear.
    """
    if policy_logits.ndim != 4 or policy_logits.shape[1] != 5:
        raise ValueError(f"V536 policy_logits must be [B,5,H,W], got {tuple(policy_logits.shape)}")
    if case_policy_logits.ndim != 2 or case_policy_logits.shape[1] != 5:
        raise ValueError(
            f"V536 case_policy_logits must be [B,5], got {tuple(case_policy_logits.shape)}"
        )
    if case_action_gain.ndim != 2 or case_action_gain.shape[1] != 4:
        raise ValueError(
            f"V536 case_action_gain must be [B,4], got {tuple(case_action_gain.shape)}"
        )

    policy_target = policy_target.long()
    active_pixel = (utility_domain.sum(dim=1) > 0.0) | (policy_target > 0)
    pixel_element = F.cross_entropy(policy_logits, policy_target, reduction="none")
    pixel_ce = (
        pixel_element[active_pixel].mean()
        if bool(active_pixel.any().item())
        else policy_logits.sum() * 0.0
    )

    utility_prediction = torch.tanh(action_utility_logits)
    utility_element = F.smooth_l1_loss(
        utility_prediction,
        action_utility_target,
        beta=0.10,
        reduction="none",
    )
    domain = utility_domain > 0.5
    utility_terms = []
    for value in (-1.0, 0.0, 1.0):
        membership = domain & (action_utility_target == value)
        if bool(membership.any().item()):
            utility_terms.append(utility_element[membership].mean())
    utility_loss = (
        torch.stack(utility_terms).mean()
        if utility_terms
        else action_utility_logits.sum() * 0.0
    )

    policy_probability = F.softmax(policy_logits, dim=1)
    action_probability = policy_probability[:, 1:]
    harm_weight = harm_mask.to(action_probability.dtype) * utility_domain
    harm_loss = (
        action_probability * harm_weight
    ).sum() / harm_weight.sum().clamp_min(1.0)
    expected_utility = (action_probability * action_utility_target).sum(dim=1)
    active_domain = (utility_domain.sum(dim=1) > 0.0).to(expected_utility.dtype)
    regret_loss = (
        F.relu(-expected_utility) * active_domain
    ).sum() / active_domain.sum().clamp_min(1.0)

    best_case_gain, best_case_action = case_action_gain.max(dim=1)
    case_target = torch.where(
        best_case_gain > 0.0,
        best_case_action + 1,
        torch.zeros_like(best_case_action),
    ).long()
    positive_case = case_target > 0
    case_element = F.cross_entropy(case_policy_logits, case_target, reduction="none")
    if bool(positive_case.any().item()):
        case_terms = [case_element[positive_case].mean()]
        negative_case = ~positive_case
        if bool(negative_case.any().item()):
            case_terms.append(case_element[negative_case].mean())
        case_ce = torch.stack(case_terms).mean()
    else:
        # Do not erase a learned rare-positive selector once the training Base
        # has memorised the current batch and no residual positive remains.
        case_ce = case_policy_logits.sum() * 0.0

    case_probability = F.softmax(case_policy_logits, dim=1)
    case_action_probability = case_probability[:, 1:]
    negative_case_gain = F.relu(-case_action_gain)
    case_harm_loss = (
        case_action_probability * negative_case_gain
    ).sum() / negative_case_gain.sum().clamp_min(EPS)
    expected_case_gain = (
        case_action_probability * case_action_gain
    ).sum(dim=1)
    case_regret_loss = F.relu(-expected_case_gain).mean()

    objective = (
        pixel_ce
        + utility_loss
        + harm_loss
        + regret_loss
        + case_ce
        + case_harm_loss
        + case_regret_loss
    )

    pixel_prediction = policy_logits.argmax(dim=1)
    pixel_execute_target = policy_target > 0
    pixel_execute_prediction = pixel_prediction > 0
    tp = (pixel_execute_prediction & pixel_execute_target & active_pixel).float().sum()
    fp = (pixel_execute_prediction & (~pixel_execute_target) & active_pixel).float().sum()
    fn = ((~pixel_execute_prediction) & pixel_execute_target & active_pixel).float().sum()
    execute_precision = tp / (tp + fp).clamp_min(1.0)
    execute_recall = tp / (tp + fn).clamp_min(1.0)

    case_prediction = case_policy_logits.argmax(dim=1)
    case_execute_prediction = case_prediction > 0
    case_tp = (case_execute_prediction & positive_case).float().sum()
    case_fp = (case_execute_prediction & (~positive_case)).float().sum()
    case_fn = ((~case_execute_prediction) & positive_case).float().sum()
    case_precision = case_tp / (case_tp + case_fp).clamp_min(1.0)
    case_recall = case_tp / (case_tp + case_fn).clamp_min(1.0)

    entropy = -(
        policy_probability.clamp_min(EPS)
        * policy_probability.clamp_min(EPS).log()
    ).sum(dim=1).mean()
    diagnostics = {
        "v536_policy_objective": objective.detach(),
        "v536_pixel_ce": pixel_ce.detach(),
        "v536_utility_loss": utility_loss.detach(),
        "v536_pixel_harm_loss": harm_loss.detach(),
        "v536_pixel_regret_loss": regret_loss.detach(),
        "v536_case_ce": case_ce.detach(),
        "v536_case_harm_loss": case_harm_loss.detach(),
        "v536_case_regret_loss": case_regret_loss.detach(),
        "v536_target_execute_rate": pixel_execute_target.float().mean().detach(),
        "v536_predicted_execute_rate": pixel_execute_prediction.float().mean().detach(),
        "v536_execute_precision": execute_precision.detach(),
        "v536_execute_recall": execute_recall.detach(),
        "v536_target_execute_case_rate": positive_case.float().mean().detach(),
        "v536_predicted_execute_case_rate": case_execute_prediction.float().mean().detach(),
        "v536_case_execute_precision": case_precision.detach(),
        "v536_case_execute_recall": case_recall.detach(),
        "v536_case_best_gain": best_case_gain.mean().detach(),
        "v536_case_expected_gain": expected_case_gain.mean().detach(),
        "v536_policy_entropy": entropy.detach(),
        "v536_all_preserve_accuracy": (1.0 - pixel_execute_target.float().mean()).detach(),
    }
    return objective, diagnostics

def _compute_v532_unified_sparse_refiner_loss(
    cfg: Any,
    masks: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    epoch: int | None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """One-run objective for V532 unified typed sparse refinement.

    The objective separates four responsibilities:
      * M1: factual typed error localisation and minimally sufficient action dose;
      * router: Preserve/Edit and conditional typed action prediction;
      * outcome: actual Neutral/Benefit/Harm of each generated action;
      * final: segmentation quality with explicit pixel- and case-level no-harm.
    """
    current_epoch = max(0, int(epoch or 0))

    def _scale(
        prefix: str,
        *,
        default_start: int,
        default_ramp: int,
        default_initial: float,
        default_final: float = 1.0,
    ) -> float:
        start = max(0, int(_m1(cfg, f"V532_{prefix}_START_EPOCH", default_start)))
        ramp = max(1, int(_m1(cfg, f"V532_{prefix}_RAMP_EPOCHS", default_ramp)))
        initial = max(0.0, float(_m1(cfg, f"V532_{prefix}_START_SCALE", default_initial)))
        final = max(0.0, float(_m1(cfg, f"V532_{prefix}_FINAL_SCALE", default_final)))
        if current_epoch < start:
            return 0.0
        if ramp <= 1:
            return final
        progress = min(1.0, max(0.0, float(current_epoch - start) / float(ramp - 1)))
        return initial + (final - initial) * progress

    m1_scale = _scale(
        "M1", default_start=0, default_ramp=10, default_initial=0.50
    )
    route_scale = _scale(
        "ROUTE", default_start=3, default_ramp=12, default_initial=0.10
    )
    final_scale = _scale(
        "FINAL", default_start=5, default_ramp=20, default_initial=0.05
    )
    safety_scale = _scale(
        "SAFETY", default_start=10, default_ramp=15, default_initial=0.0
    )
    refiner_global_scale = max(route_scale, final_scale, safety_scale)

    gt = (_as_b1hw(masks) >= 0.5).float()
    c0 = _as_b1hw(aux["c0_prob_online"]).clamp(EPS, 1.0 - EPS)
    factual = build_factual_cause_targets(
        c0.detach(),
        gt.detach(),
        boundary_radius=int(_m1(cfg, "V503_FACTUAL_BOUNDARY_RADIUS", 2)),
        failure_dice=float(_m1(cfg, "V503_FAILURE_DICE", 0.55)),
        include_global_action=False,
    )
    cause_target = factual["cause_targets"]
    action_target = factual["action_target"].long()
    base_hard = factual["base_hard"]
    base_correct = factual["base_correct"]
    factual_error = factual["factual_error"]

    # ------------------------------------------------------------- typed M1.
    cause_logits = aux["error_cause_map_logits"]
    cause_probs = aux["error_cause_map_probs"].clamp(EPS, 1.0 - EPS)
    if cause_logits.shape[-2:] != gt.shape[-2:]:
        cause_logits = F.interpolate(
            cause_logits, size=gt.shape[-2:], mode="bilinear", align_corners=False
        )
        cause_probs = torch.sigmoid(cause_logits)
    negative_ratio = int(_m1(cfg, "V532_OHEM_NEGATIVE_RATIO", 4))
    min_negatives = int(_m1(cfg, "V532_OHEM_MIN_NEGATIVES", 256))
    cause_terms = []
    for action_index in range(4):
        cause_terms.append(
            _v532_ohem_binary_bce(
                cause_logits[:, action_index],
                cause_target[:, action_index],
                negative_ratio=negative_ratio,
                min_negatives=min_negatives,
            )
        )
    cause_bce = torch.stack(cause_terms).mean()
    v535_enabled = bool(
        _m1(cfg, "V535_ADAPTIVE_UTILITY_POLICY_ENABLED", False)
    )
    v536_enabled = bool(
        _m1(cfg, "V536_PRIOR_ALIGNED_CASE_COMPONENT_ENABLED", False)
    )
    v537_enabled = bool(
        _m1(cfg, "V537_COMPONENT_UTILITY_RANKER_ENABLED", False)
    )
    m1_protocol = str(_m1(cfg, "PROTOCOL", "")).strip().lower()
    clean_dynamic_component_set = m1_protocol in {"clean_dynamic_component_set", "tc_drcs"}
    v538_enabled = bool(
        clean_dynamic_component_set
        or _m1(cfg, "V538_ONLINE_COMPONENT_REFINER_ENABLED", False)
    )
    cause_dice = (
        _v535_sparse_channel_loss(cause_probs, cause_target)
        if v535_enabled
        else _v531_channel_dice_loss(cause_probs, cause_target)
    )

    # V533 directly supervises the hierarchy instead of relying only on the
    # products P(FP)*P(Delete|FP), etc.  This avoids gradient attenuation in
    # rare FP/FN regions and makes polarity/subtype errors auditable.
    zero_hierarchy = cause_logits.sum() * 0.0
    polarity_loss = zero_hierarchy
    fp_subtype_loss = zero_hierarchy
    fn_subtype_loss = zero_hierarchy
    if all(
        key in aux
        for key in (
            "v518_error_polarity_logits",
            "v518_fp_subtype_logits",
            "v518_fn_subtype_logits",
        )
    ):
        polarity_logits = aux["v518_error_polarity_logits"]
        fp_subtype_logits = aux["v518_fp_subtype_logits"]
        fn_subtype_logits = aux["v518_fn_subtype_logits"]
        if polarity_logits.shape[-2:] != gt.shape[-2:]:
            polarity_logits = F.interpolate(
                polarity_logits, size=gt.shape[-2:], mode="bilinear", align_corners=False
            )
            fp_subtype_logits = F.interpolate(
                fp_subtype_logits, size=gt.shape[-2:], mode="bilinear", align_corners=False
            )
            fn_subtype_logits = F.interpolate(
                fn_subtype_logits, size=gt.shape[-2:], mode="bilinear", align_corners=False
            )
        fp_target = (cause_target[:, 0:1] + cause_target[:, 2:3]).clamp(0.0, 1.0)
        fn_target = (cause_target[:, 1:2] + cause_target[:, 3:4]).clamp(0.0, 1.0)
        polarity_loss = 0.5 * (
            _v532_ohem_binary_bce(
                polarity_logits[:, 0:1], fp_target,
                negative_ratio=negative_ratio, min_negatives=min_negatives,
            )
            + _v532_ohem_binary_bce(
                polarity_logits[:, 1:2], fn_target,
                negative_ratio=negative_ratio, min_negatives=min_negatives,
            )
        )
        fp_mask = fp_target[:, 0] > 0.5
        fn_mask = fn_target[:, 0] > 0.5
        if bool(fp_mask.any().item()):
            fp_subtype_target = (cause_target[:, 2] > 0.5).long()
            fp_subtype_loss = F.cross_entropy(
                fp_subtype_logits, fp_subtype_target, reduction="none"
            )[fp_mask].mean()
        if bool(fn_mask.any().item()):
            fn_subtype_target = (cause_target[:, 3] > 0.5).long()
            fn_subtype_loss = F.cross_entropy(
                fn_subtype_logits, fn_subtype_target, reduction="none"
            )[fn_mask].mean()

    cause_loss = (
        float(_m1(cfg, "V533_POLARITY_WEIGHT", 1.0)) * polarity_loss
        + float(_m1(cfg, "V533_FP_SUBTYPE_WEIGHT", 0.5)) * fp_subtype_loss
        + float(_m1(cfg, "V533_FN_SUBTYPE_WEIGHT", 0.5)) * fn_subtype_loss
        + float(_m1(cfg, "V533_COMBINED_CAUSE_BCE_WEIGHT", 0.25)) * cause_bce
        + float(_m1(cfg, "V533_COMBINED_CAUSE_DICE_WEIGHT", 0.25)) * cause_dice
    )

    alpha = aux["local_primary_action_alpha"].clamp(0.0, 1.0)
    if alpha.shape[-2:] != gt.shape[-2:]:
        alpha = F.interpolate(
            alpha, size=gt.shape[-2:], mode="bilinear", align_corners=False
        )
    alpha_target = _v532_minimal_alpha_target(
        c0,
        cause_target,
        probability_margin=float(_m1(cfg, "V532_ALPHA_PROBABILITY_MARGIN", 0.05)),
    )
    alpha_error = F.smooth_l1_loss(
        alpha, alpha_target, beta=0.10, reduction="none"
    )
    alpha_inside_loss = (
        alpha_error * cause_target
    ).sum() / cause_target.sum().clamp_min(1.0)
    alpha_outside_loss = (
        alpha * (1.0 - cause_target)
    ).sum() / (1.0 - cause_target).sum().clamp_min(1.0)

    action_candidates = aux["v532_action_candidate_probs"].clamp(EPS, 1.0 - EPS)
    gt_actions = gt.expand(-1, 4, -1, -1)
    action_repair_bce_map = F.binary_cross_entropy(
        action_candidates, gt_actions, reduction="none"
    )
    action_repair_loss = (
        action_repair_bce_map * cause_target
    ).sum() / cause_target.sum().clamp_min(1.0)

    v538_m1_objective = c0.sum() * 0.0
    v538_m2_objective = c0.sum() * 0.0
    v538_diag: Dict[str, torch.Tensor] = {}
    if v538_enabled:
        v538_m1_objective, v538_m2_objective, v538_diag = (
            compute_v538_online_component_loss(
                cfg=cfg,
                masks=gt,
                aux=aux,
                epoch=current_epoch,
                base_probability=c0,
            )
        )

    v537_coverage_loss = c0.sum() * 0.0
    v537_purity_loss = c0.sum() * 0.0
    v537_pwo_diag: Dict[str, torch.Tensor] = {}
    if v537_enabled:
        v537_coverage_loss, v537_purity_loss, v537_pwo_diag = (
            _v537_pwo_candidate_losses(
                action_candidates,
                c0,
                factual_error,
                cause_target,
            )
        )

    # -------------------------------------------------------- factorised route.
    edit_logit = _as_b1hw(aux["m2_edit_logit"])
    edit_target = factual_error
    edit_loss = _v532_ohem_binary_bce(
        edit_logit,
        edit_target,
        negative_ratio=negative_ratio,
        min_negatives=min_negatives,
        positive_weight=float(_m1(cfg, "V534_EDIT_POSITIVE_WEIGHT", 1.0)),
        negative_weight=float(_m1(cfg, "V534_EDIT_NEGATIVE_WEIGHT", 1.0)),
    )

    action_logits = aux["m2_action_logits"]
    error_mask = action_target > 0
    if bool(error_mask.any().item()):
        conditional_target = (action_target - 1).clamp(0, 3)
        action_ce_map = F.cross_entropy(
            action_logits, conditional_target, reduction="none"
        )
        action_loss = action_ce_map[error_mask].mean()
    else:
        action_loss = action_logits.sum() * 0.0

    # ----------------------------------------------------- actual action outcome.
    base_abs_error = (c0.detach() - gt).abs()
    candidate_abs_error = (
        action_candidates.detach() - gt_actions
    ).abs()
    signed_gain = base_abs_error.expand(-1, 4, -1, -1) - candidate_abs_error
    candidate_hard = action_candidates.detach() >= 0.5
    gt_hard = gt_actions >= 0.5
    base_wrong = (base_hard != gt).expand(-1, 4, -1, -1)
    base_right = ~base_wrong
    candidate_correct = candidate_hard == gt_hard
    margin = max(0.0, float(_m1(cfg, "V532_OUTCOME_GAIN_MARGIN", 0.01)))
    benefit = (signed_gain > margin) | (base_wrong & candidate_correct)
    harm = (signed_gain < -margin) | (base_right & (~candidate_correct))
    both = benefit & harm
    benefit = benefit & ((~both) | (signed_gain >= 0.0))
    harm = harm & ((~both) | (signed_gain < 0.0))
    outcome_target = torch.zeros_like(signed_gain, dtype=torch.long)
    outcome_target[benefit] = 1
    outcome_target[harm] = 2

    outcome_logits = aux["m2_action_outcome_logits"]
    outcome_probs = aux["m2_action_outcome_probs"].clamp(EPS, 1.0 - EPS)
    candidate_change = (
        action_candidates.detach() - c0.detach().expand(-1, 4, -1, -1)
    ).abs()
    # V534 aligns Outcome supervision with actual deployment relevance.  Tiny
    # probability shifts that do not change the hard mask are Neutral noise and
    # must not dominate Benefit/Harm learning.
    outcome_change_threshold = max(
        0.0, float(_m1(cfg, "V533_OUTCOME_CHANGE_THRESHOLD", 0.02))
    )
    hard_candidate_change = (
        candidate_hard != base_hard.expand(-1, 4, -1, -1).bool()
    )
    meaningful_soft_change = candidate_change > outcome_change_threshold
    deployment_relevant_change = (
        hard_candidate_change | meaningful_soft_change
    ).to(cause_target.dtype)
    outcome_domain = torch.maximum(cause_target, deployment_relevant_change)

    # V535 direct utility policy.  Preserve is class 0.  An action is a target
    # only when it physically changes the hard mask and is truly beneficial.
    v535_policy_objective = c0.sum() * 0.0
    v535_diag: Dict[str, torch.Tensor] = {}
    if v535_enabled:
        valid_benefit = benefit & (~harm) & hard_candidate_change
        masked_gain = torch.where(
            valid_benefit,
            signed_gain,
            signed_gain.new_full(signed_gain.shape, -1.0e6),
        )
        _, best_action_index = masked_gain.max(dim=1)
        has_beneficial_action = valid_benefit.any(dim=1)
        policy_target = torch.where(
            has_beneficial_action,
            best_action_index + 1,
            torch.zeros_like(best_action_index),
        )
        action_utility_target = torch.where(
            benefit & hard_candidate_change,
            torch.ones_like(signed_gain),
            torch.where(
                harm & hard_candidate_change,
                -torch.ones_like(signed_gain),
                torch.zeros_like(signed_gain),
            ),
        ).detach()
        if v538_enabled:
            # M1 slot generation is trained separately by differentiable soft
            # gain.  The adaptive route receives only the quality-gated
            # Preserve/listwise utility objective.
            v535_policy_objective = v538_m2_objective
            v535_diag = v538_diag
        elif v537_enabled:
            v535_policy_objective, v535_diag = _v537_component_utility_objective(
                scores=aux["v537_candidate_scores"],
                candidate_masks=aux["v537_candidate_masks"],
                candidate_actions=aux["v537_candidate_actions"],
                candidate_valid=aux["v537_candidate_valid"],
                base_hard=base_hard,
                action_candidates=action_candidates,
                gt=gt,
                gain_epsilon=float(_m1(cfg, "V537_GAIN_EPSILON", 1.0e-4)),
                regression_beta=float(_m1(cfg, "V537_REGRESSION_BETA", 0.005)),
                pair_temperature=float(_m1(cfg, "V537_PAIR_TEMPERATURE", 0.01)),
                preserve_margin=float(_m1(cfg, "V537_PRESERVE_MARGIN", 0.002)),
                regression_weight=float(_m1(cfg, "V537_REGRESSION_WEIGHT", 1.0)),
                pair_weight=float(_m1(cfg, "V537_PAIR_WEIGHT", 1.0)),
                preserve_weight=float(_m1(cfg, "V537_PRESERVE_WEIGHT", 1.0)),
                oracle_ladder_diagnostics=bool(
                    _m1(cfg, "V537_ORACLE_LADDER_DIAGNOSTICS", True)
                ),
            )
        elif v536_enabled:
            case_action_gain, component_policy_target, positive_components = (
                _v536_best_component_targets(
                    base_hard,
                    action_candidates,
                    gt,
                    min_pixels=int(_m1(cfg, "V536_COMPONENT_MIN_PIXELS", 4)),
                )
            )
            # Match the pixel selector to the same deployment unit: only the
            # best positive connected component for each action is an Execute
            # target.  All harmful/neutral/scattered changes remain Preserve.
            policy_target = component_policy_target
            component_domain = torch.maximum(
                positive_components,
                hard_candidate_change.to(positive_components.dtype),
            )
            v535_policy_objective, v535_diag = _v536_prior_aligned_objective(
                aux["v535_policy_logits"],
                policy_target,
                aux["v535_action_utility_logits"],
                action_utility_target,
                harm,
                component_domain,
                aux["v536_case_policy_logits"],
                case_action_gain,
            )
        else:
            v535_policy_objective, v535_diag = _v535_adaptive_policy_objective(
                aux["v535_policy_logits"],
                policy_target,
                aux["v535_action_utility_logits"],
                action_utility_target,
                harm,
                outcome_domain,
            )

    outcome_ce = _v532_balanced_outcome_loss(
        outcome_logits, outcome_target, outcome_domain
    )
    outcome_one_hot = F.one_hot(outcome_target, num_classes=3).permute(0, 1, 4, 2, 3)
    outcome_brier = (
        (outcome_probs - outcome_one_hot.to(outcome_probs.dtype)).square()
        * outcome_domain[:, :, None]
    ).sum() / (
        outcome_domain.sum().clamp_min(1.0) * 3.0
    )

    predicted_utility = aux["m2_action_utility"]
    utility_scale = max(float(_m1(cfg, "V532_UTILITY_TARGET_SCALE", 0.25)), EPS)
    utility_target = (signed_gain / utility_scale).clamp(-1.0, 1.0)
    utility_regression = (
        F.smooth_l1_loss(
            predicted_utility, utility_target, beta=0.10, reduction="none"
        ) * outcome_domain
    ).sum() / outcome_domain.sum().clamp_min(1.0)

    # V534 directly supervises the exact deployed Execute decision.  The target
    # is positive only when the currently selected action is truly beneficial.
    selected_action_index = action_logits.detach().argmax(dim=1, keepdim=True)
    selected_benefit_target = torch.gather(
        benefit.to(c0.dtype), 1, selected_action_index
    )
    selected_harm_target = torch.gather(
        harm.to(c0.dtype), 1, selected_action_index
    )
    execute_target = (
        factual_error
        * selected_benefit_target
        * (1.0 - selected_harm_target)
    ).detach()
    execute_probability = _as_b1hw(aux["m2_execute_probability"]).clamp(
        EPS, 1.0 - EPS
    )
    execute_loss = _v532_ohem_binary_bce(
        torch.logit(execute_probability),
        execute_target,
        negative_ratio=int(_m1(cfg, "V534_EXECUTE_NEGATIVE_RATIO", 8)),
        min_negatives=int(_m1(cfg, "V534_EXECUTE_MIN_NEGATIVES", 512)),
        positive_weight=float(_m1(cfg, "V534_EXECUTE_POSITIVE_WEIGHT", 1.0)),
        negative_weight=float(_m1(cfg, "V534_EXECUTE_NEGATIVE_WEIGHT", 4.0)),
    )

    benefit_probability = outcome_probs[:, :, 1]
    harm_probability = outcome_probs[:, :, 2]
    nonbenefit_domain = outcome_domain * (outcome_target != 1).to(outcome_domain.dtype)
    harm_target_domain = outcome_domain * (outcome_target == 2).to(outcome_domain.dtype)
    false_benefit_loss = (
        benefit_probability * nonbenefit_domain
    ).sum() / nonbenefit_domain.sum().clamp_min(1.0)
    harm_miss_loss = (
        (1.0 - harm_probability) * harm_target_domain
    ).sum() / harm_target_domain.sum().clamp_min(1.0)

    # -------------------------------------------------------- final segmentation.
    final_prob = _as_b1hw(aux["m2_fused_probs"]).clamp(EPS, 1.0 - EPS)
    final_bce = F.binary_cross_entropy(final_prob, gt)
    final_dice_loss = _soft_dice_loss(final_prob, gt)
    final_hard_st = _v533_ste_hard_mask(final_prob)
    final_hard_dice_loss = _soft_dice_loss(final_hard_st, gt)
    final_boundary_loss = _boundary_l1(
        final_prob,
        gt,
        radius=int(_m1(cfg, "V532_BOUNDARY_RADIUS", 1)),
    )

    soft_base_dice_case = _soft_dice_probs(c0, gt)[:, 0]
    soft_final_dice_case = _soft_dice_probs(final_prob, gt)[:, 0]
    base_hard_float = (c0.detach() >= 0.5).to(gt.dtype)
    hard_base_dice_case = _soft_dice_probs(base_hard_float, gt)[:, 0]
    hard_final_dice_case = _soft_dice_probs(final_hard_st, gt)[:, 0]

    hard_case_regret = F.relu(
        hard_base_dice_case - hard_final_dice_case
        + float(_m1(cfg, "V532_CASE_REGRET_MARGIN", 0.0))
    )
    soft_case_regret = F.relu(soft_base_dice_case - soft_final_dice_case)
    case_regret = (
        float(_m1(cfg, "V533_HARD_REGRET_WEIGHT", 1.0)) * hard_case_regret
        + float(_m1(cfg, "V533_SOFT_REGRET_WEIGHT", 0.25)) * soft_case_regret
    )
    tail_fraction = min(
        max(float(_m1(cfg, "V532_REGRET_TAIL_FRACTION", 0.25)), 0.0), 1.0
    )
    tail_count = max(1, int(math.ceil(case_regret.numel() * max(tail_fraction, 1.0e-6))))
    regret_cvar = torch.topk(case_regret, k=min(tail_count, case_regret.numel())).values.mean()
    regret_loss = case_regret.mean() + float(
        _m1(cfg, "V532_REGRET_CVAR_WEIGHT", 0.5)
    ) * regret_cvar

    # Safety is evaluated only on physically changed hard pixels.  This removes
    # the previous pathology where infinitesimal soft shifts changed every case.
    final_abs_error = (final_prob - gt).abs()
    error_delta = final_abs_error - base_abs_error
    hard_changed = (final_hard_st.detach() != base_hard_float).to(gt.dtype)
    execute_gate_st = _as_b1hw(aux["m2_execute_gate"]).clamp(0.0, 1.0)
    pixel_margin = max(0.0, float(_m1(cfg, "V532_PIXEL_HARM_MARGIN", 0.0)))
    actual_harm_mass = F.relu(error_delta - pixel_margin) * hard_changed
    actual_benefit_mass = F.relu(-error_delta - pixel_margin) * hard_changed
    changed_mass = hard_changed.sum().clamp_min(1.0)
    harmful_edit_loss = actual_harm_mass.sum() / changed_mass
    # V533 used detached hard_changed here, so these losses had values but zero
    # gate gradient.  V534 uses the hard-forward/soft-backward Execute gate.
    preserve_correct_loss = (
        execute_gate_st * base_correct
    ).sum() / base_correct.sum().clamp_min(1.0)
    edit_fraction_case = hard_changed.flatten(1).mean(dim=1)
    execute_fraction_case = execute_gate_st.flatten(1).mean(dim=1)
    edit_budget = float(_m1(cfg, "V532_MAX_EDIT_FRACTION", 0.035))
    edit_budget_loss = F.relu(execute_fraction_case - edit_budget).mean()

    if clean_dynamic_component_set:
        # CLEAN formal path: historical cause/alpha/action-repair objectives are
        # diagnostics only.  Component M1 has one owner and one objective.
        m1_raw = v538_m1_objective
    else:
        m1_raw = (
            float(_m1(cfg, "V532_CAUSE_WEIGHT", 1.0)) * cause_loss
            + float(_m1(cfg, "V532_ALPHA_WEIGHT", 1.0)) * alpha_inside_loss
            + float(_m1(cfg, "V532_ALPHA_OUTSIDE_WEIGHT", 0.50)) * alpha_outside_loss
            + float(_m1(cfg, "V532_ACTION_REPAIR_WEIGHT", 0.50)) * action_repair_loss
            + float(_m1(cfg, "V537_PWO_COVERAGE_WEIGHT", 0.0)) * v537_coverage_loss
            + float(_m1(cfg, "V537_PWO_PURITY_WEIGHT", 0.0)) * v537_purity_loss
            + float(_m1(cfg, "V538_M1_GLOBAL_WEIGHT", 1.0)) * v538_m1_objective
        )
    if v535_enabled:
        # One aligned objective replaces Edit/Action/Outcome/product-gate losses.
        route_raw = v535_policy_objective
    else:
        route_raw = (
            float(_m1(cfg, "V532_EDIT_EXISTENCE_WEIGHT", 1.0)) * edit_loss
            + float(_m1(cfg, "V532_ACTION_TYPE_WEIGHT", 1.0)) * action_loss
            + float(_m1(cfg, "V532_OUTCOME_CE_WEIGHT", 1.0)) * outcome_ce
            + float(_m1(cfg, "V532_OUTCOME_BRIER_WEIGHT", 0.25)) * outcome_brier
            + float(_m1(cfg, "V532_UTILITY_REG_WEIGHT", 0.50)) * utility_regression
            + float(_m1(cfg, "V534_EXECUTE_WEIGHT", 0.0)) * execute_loss
            + float(_m1(cfg, "V534_FALSE_BENEFIT_WEIGHT", 0.0)) * false_benefit_loss
            + float(_m1(cfg, "V534_HARM_RECALL_WEIGHT", 0.0)) * harm_miss_loss
        )
    final_raw = (
        float(_m1(cfg, "V532_FINAL_BCE_WEIGHT", 0.50)) * final_bce
        + float(_m1(cfg, "V532_FINAL_DICE_WEIGHT", 0.50)) * final_dice_loss
        + float(_m1(cfg, "V533_FINAL_HARD_DICE_WEIGHT", 1.00)) * final_hard_dice_loss
        + float(_m1(cfg, "V532_FINAL_BOUNDARY_WEIGHT", 0.25)) * final_boundary_loss
    )
    safety_raw = (
        float(_m1(cfg, "V532_REGRET_WEIGHT", 1.00)) * regret_loss
        + float(_m1(cfg, "V532_HARM_WEIGHT", 1.00)) * harmful_edit_loss
        + float(_m1(cfg, "V532_PRESERVE_CORRECT_WEIGHT", 0.50)) * preserve_correct_loss
        + float(_m1(cfg, "V532_EDIT_BUDGET_WEIGHT", 0.25)) * edit_budget_loss
    )

    m1_objective = m1_scale * m1_raw
    refiner_objective = (
        route_scale * route_raw
        + final_scale * final_raw
        + safety_scale * safety_raw
    )
    zero = refiner_objective * 0.0
    total = m1_objective + refiner_objective

    # ------------------------------------------------------------- diagnostics.
    route_probs = aux["m2_route_probs"]
    route_soft_probs = aux.get("m2_route_soft_probs", route_probs)
    selected = route_soft_probs.argmax(dim=1)
    edit_prediction = aux["m2_edit_probability"] > 0.5
    deployed_edit_prediction = aux.get(
        "m2_execute_gate", aux["m2_edit_probability"]
    ) > 0.5
    edit_target_bool = edit_target[:, 0] > 0.5
    tp = (edit_prediction & edit_target_bool).float().sum()
    fp = (edit_prediction & (~edit_target_bool)).float().sum()
    fn = ((~edit_prediction) & edit_target_bool).float().sum()
    edit_precision = tp / (tp + fp).clamp_min(1.0)
    edit_recall = tp / (tp + fn).clamp_min(1.0)
    dep_tp = (deployed_edit_prediction & edit_target_bool).float().sum()
    dep_fp = (deployed_edit_prediction & (~edit_target_bool)).float().sum()
    dep_fn = ((~deployed_edit_prediction) & edit_target_bool).float().sum()
    deployed_edit_precision = dep_tp / (dep_tp + dep_fp).clamp_min(1.0)
    deployed_edit_recall = dep_tp / (dep_tp + dep_fn).clamp_min(1.0)
    action_accuracy = (
        (selected[error_mask] == action_target[error_mask]).float().mean()
        if bool(error_mask.any().item()) else total.new_zeros(())
    )
    outcome_prediction = outcome_probs.argmax(dim=2)
    outcome_domain_bool = outcome_domain > 0.5
    outcome_accuracy = (
        (outcome_prediction == outcome_target).to(outcome_domain.dtype)
        * outcome_domain
    ).sum() / outcome_domain.sum().clamp_min(1.0)
    neutral_precision, neutral_recall, neutral_f1 = _v533_binary_prf(
        outcome_prediction == 0, outcome_target == 0, outcome_domain_bool
    )
    benefit_precision, benefit_recall, benefit_f1 = _v533_binary_prf(
        outcome_prediction == 1, outcome_target == 1, outcome_domain_bool
    )
    harm_precision, harm_recall, harm_f1 = _v533_binary_prf(
        outcome_prediction == 2, outcome_target == 2, outcome_domain_bool
    )
    outcome_macro_f1 = (neutral_f1 + benefit_f1 + harm_f1) / 3.0

    candidate_probs = torch.cat([c0, action_candidates], dim=1)
    soft_candidate_dice = _soft_dice_probs(candidate_probs, gt)
    soft_oracle_dice_case = soft_candidate_dice.max(dim=1).values
    candidate_hard_st = _v533_ste_hard_mask(candidate_probs)
    hard_candidate_dice = _soft_dice_probs(candidate_hard_st, gt)
    hard_oracle_dice_case = hard_candidate_dice.max(dim=1).values

    soft_oracle_gain_case = (soft_oracle_dice_case - soft_base_dice_case).clamp_min(0.0)
    hard_oracle_gain_case = (hard_oracle_dice_case - hard_base_dice_case).clamp_min(0.0)
    soft_final_gain_case = soft_final_dice_case - soft_base_dice_case
    hard_final_gain_case = hard_final_dice_case - hard_base_dice_case

    # Primary V532 diagnostics are now hard-metric aligned; soft values remain
    # available under explicit V533 names for calibration analysis.
    base_dice_case = hard_base_dice_case
    final_dice_case = hard_final_dice_case
    oracle_dice_case = hard_oracle_dice_case
    oracle_gain_case = hard_oracle_gain_case
    final_gain_case = hard_final_gain_case
    oracle_gain = oracle_gain_case.mean()
    final_gain = final_gain_case.mean()
    capture_ratio = final_gain.clamp_min(0.0) / oracle_gain.clamp_min(EPS)
    changed_case = edit_fraction_case > 0.0
    hard_benefit_count = (
        hard_changed * factual_error * (final_hard_st.detach() == gt).to(gt.dtype)
    ).sum()
    hard_harm_count = (
        hard_changed * base_correct * (final_hard_st.detach() != gt).to(gt.dtype)
    ).sum()
    changed_pixel_correctness = hard_benefit_count / (
        hard_benefit_count + hard_harm_count
    ).clamp_min(1.0)

    diag: Dict[str, torch.Tensor] = {
        "v532_total_loss": total.detach(),
        "v532_m1_objective": m1_objective.detach(),
        "v532_refiner_objective": refiner_objective.detach(),
        "v532_m1_raw": m1_raw.detach(),
        "v532_route_raw": route_raw.detach(),
        "v532_final_raw": final_raw.detach(),
        "v532_safety_raw": safety_raw.detach(),
        "v532_m1_scale": total.new_tensor(m1_scale),
        "v532_route_scale": total.new_tensor(route_scale),
        "v532_final_scale": total.new_tensor(final_scale),
        "v532_safety_scale": total.new_tensor(safety_scale),
        "v532_refiner_global_scale": total.new_tensor(refiner_global_scale),
        "v532_cause_loss": cause_loss.detach(),
        "v532_cause_bce": cause_bce.detach(),
        "v532_cause_dice_loss": cause_dice.detach(),
        "v533_polarity_loss": polarity_loss.detach(),
        "v533_fp_subtype_loss": fp_subtype_loss.detach(),
        "v533_fn_subtype_loss": fn_subtype_loss.detach(),
        "v532_alpha_inside_loss": alpha_inside_loss.detach(),
        "v532_alpha_outside_loss": alpha_outside_loss.detach(),
        "v532_action_repair_loss": action_repair_loss.detach(),
        "v537_pwo_coverage_loss": v537_coverage_loss.detach(),
        "v537_pwo_purity_loss": v537_purity_loss.detach(),
        "v538_m1_objective": v538_m1_objective.detach(),
        "v538_m2_objective": v538_m2_objective.detach(),
        "v532_edit_loss": edit_loss.detach(),
        "v532_action_loss": action_loss.detach(),
        "v532_outcome_ce": outcome_ce.detach(),
        "v532_outcome_brier": outcome_brier.detach(),
        "v532_utility_regression": utility_regression.detach(),
        "v534_execute_loss": execute_loss.detach(),
        "v534_execute_target_rate": execute_target.mean().detach(),
        "v534_false_benefit_loss": false_benefit_loss.detach(),
        "v534_harm_miss_loss": harm_miss_loss.detach(),
        "v534_execute_fraction": execute_fraction_case.mean().detach(),
        "v532_final_bce": final_bce.detach(),
        "v532_final_dice_loss": final_dice_loss.detach(),
        "v533_final_hard_dice_loss": final_hard_dice_loss.detach(),
        "v532_final_boundary_loss": final_boundary_loss.detach(),
        "v532_regret_loss": regret_loss.detach(),
        "v532_regret_cvar": regret_cvar.detach(),
        "v532_harmful_edit_loss": harmful_edit_loss.detach(),
        "v532_preserve_correct_loss": preserve_correct_loss.detach(),
        "v532_edit_budget_loss": edit_budget_loss.detach(),
        "v532_base_dice": base_dice_case.mean().detach(),
        "v532_final_dice": final_dice_case.mean().detach(),
        "v532_final_gain": final_gain.detach(),
        "v532_oracle_dice": oracle_dice_case.mean().detach(),
        "v532_oracle_gain": oracle_gain.detach(),
        "v532_oracle_capture_ratio": capture_ratio.detach(),
        "v532_improved_case_rate": (final_gain_case > 1.0e-4).float().mean().detach(),
        "v532_harmful_case_rate": (final_gain_case < -1.0e-4).float().mean().detach(),
        "v532_changed_case_rate": changed_case.float().mean().detach(),
        "v532_edit_fraction": edit_fraction_case.mean().detach(),
        "v532_changed_pixel_correctness": changed_pixel_correctness.detach(),
        "v532_actual_benefit_mass": actual_benefit_mass.mean().detach(),
        "v532_actual_harm_mass": actual_harm_mass.mean().detach(),
        "v532_edit_precision": edit_precision.detach(),
        "v532_edit_recall": edit_recall.detach(),
        "v532_action_accuracy": action_accuracy.detach(),
        "v532_outcome_accuracy": outcome_accuracy.detach(),
        "v533_outcome_macro_f1": outcome_macro_f1.detach(),
        "v533_neutral_precision": neutral_precision.detach(),
        "v533_neutral_recall": neutral_recall.detach(),
        "v533_benefit_precision": benefit_precision.detach(),
        "v533_benefit_recall": benefit_recall.detach(),
        "v533_harm_precision": harm_precision.detach(),
        "v533_harm_recall": harm_recall.detach(),
        "v533_deployed_edit_precision": deployed_edit_precision.detach(),
        "v533_deployed_edit_recall": deployed_edit_recall.detach(),
        "v533_soft_base_dice": soft_base_dice_case.mean().detach(),
        "v533_soft_final_dice": soft_final_dice_case.mean().detach(),
        "v533_soft_final_gain": soft_final_gain_case.mean().detach(),
        "v533_soft_oracle_dice": soft_oracle_dice_case.mean().detach(),
        "v533_soft_oracle_gain": soft_oracle_gain_case.mean().detach(),
        "v533_hard_base_dice": hard_base_dice_case.mean().detach(),
        "v533_hard_final_dice": hard_final_dice_case.mean().detach(),
        "v533_hard_final_gain": hard_final_gain_case.mean().detach(),
        "v533_hard_oracle_dice": hard_oracle_dice_case.mean().detach(),
        "v533_hard_oracle_gain": hard_oracle_gain_case.mean().detach(),
        "v532_route_preserve_probability": aux["v532_route_preserve_probability"].mean().detach(),
        "v532_deployed_preserve_probability": aux["v532_deployed_preserve_probability"].mean().detach(),
        "v532_m1_gradient_scale": aux["v532_m1_gradient_scale"].mean().detach(),
        "v532_outcome_partition_error": aux["v532_outcome_partition_error"].amax().detach(),
        "v536_component_accept_rate": aux.get(
            "v536_component_accept", total.new_zeros((1,))
        ).float().mean().detach(),
        "v536_component_area_fraction": aux.get(
            "v536_component_area_fraction", total.new_zeros((1,))
        ).float().mean().detach(),
        "v536_component_score": aux.get(
            "v536_component_score", total.new_zeros((1,))
        ).float().mean().detach(),
        "v537_raw_component_count": aux.get(
            "v537_raw_component_count", total.new_zeros((1,))
        ).float().mean().detach(),
        "v537_retained_component_count": aux.get(
            "v537_retained_component_count", total.new_zeros((1,))
        ).float().mean().detach(),
        "v503_factual_atomic_causal_enabled": total.new_ones(()),
        "v484_base_dice": base_dice_case.mean().detach(),
        "v484_local_oracle_gain": oracle_gain.detach(),
        # V538 keeps its live objectives separate from the legacy V532 bundle.
        # train.py routes these tensors through an independent Base-relative
        # budget, while the detached copies above remain diagnostics only.
        "_v538_m1_objective": v538_m1_objective,
        "_v538_m2_objective": v538_m2_objective,
        "_v490_m1_objective": m1_objective,
        "_v490_m2_objective": refiner_objective,
        "_v490_m3_objective": zero,
    }
    diag.update(v535_diag)
    diag.update(v538_diag)
    diag.update(v537_pwo_diag)
    diag["v535_sparse_cause_loss"] = cause_dice.detach()
    return total, diag


def _v531_channel_dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    pred = pred.clamp(EPS, 1.0 - EPS)
    target = target.clamp(0.0, 1.0)
    inter = (pred * target).flatten(2).sum(dim=2)
    den = pred.flatten(2).sum(dim=2) + target.flatten(2).sum(dim=2)
    return (1.0 - (2.0 * inter + EPS) / (den + EPS)).mean()


def _compute_v531_typed_sparse_refiner_loss(
    cfg: Any,
    masks: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    epoch: int | None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Clean V531 objective for typed sparse monotone refinement.

    V531 deliberately bypasses the V524--V530 candidate/region ranking losses.
    M1 is supervised by factual typed errors and monotone action support; M2 is
    supervised by spatial routing, calibrated edit correctness, final
    segmentation, per-case regret, and true harmful edit mass.  There is no
    separate M3 objective: Preserve is an explicit route class.
    """
    current_epoch = max(0, int(epoch or 0))

    def _progressive_scale(
        prefix: str,
        *,
        default_start_epoch: int,
        default_ramp_epochs: int,
        default_start_scale: float,
        default_final_scale: float = 1.0,
    ) -> float:
        start_epoch = max(
            0,
            int(_m1(cfg, f"V531_{prefix}_START_EPOCH", default_start_epoch)),
        )
        ramp_epochs = max(
            1,
            int(_m1(cfg, f"V531_{prefix}_RAMP_EPOCHS", default_ramp_epochs)),
        )
        start_scale = float(
            _m1(cfg, f"V531_{prefix}_START_SCALE", default_start_scale)
        )
        final_scale = float(
            _m1(cfg, f"V531_{prefix}_FINAL_SCALE", default_final_scale)
        )
        start_scale = max(0.0, start_scale)
        final_scale = max(0.0, final_scale)
        if current_epoch < start_epoch:
            return 0.0
        if ramp_epochs <= 1:
            return final_scale
        progress = min(
            1.0,
            max(0.0, float(current_epoch - start_epoch) / float(ramp_epochs - 1)),
        )
        return start_scale + (final_scale - start_scale) * progress

    # A single-run curriculum: task heads are all instantiated and trainable
    # from epoch 1, while their loss influence is progressively increased.
    # This is not staged checkpoint training and does not freeze/reload M1.
    m1_scale = _progressive_scale(
        "M1", default_start_epoch=0, default_ramp_epochs=10,
        default_start_scale=0.50,
    )
    route_scale = _progressive_scale(
        "ROUTE", default_start_epoch=5, default_ramp_epochs=20,
        default_start_scale=0.10,
    )
    final_scale = _progressive_scale(
        "FINAL", default_start_epoch=0, default_ramp_epochs=30,
        default_start_scale=0.20,
    )
    safety_scale = _progressive_scale(
        "SAFETY", default_start_epoch=10, default_ramp_epochs=20,
        default_start_scale=0.0,
    )
    m2_global_scale = max(route_scale, final_scale, safety_scale)

    gt = (_as_b1hw(masks) >= 0.5).float()
    c0 = _as_b1hw(aux["c0_prob_online"]).clamp(EPS, 1.0 - EPS)
    factual = build_factual_cause_targets(
        c0.detach(),
        gt.detach(),
        boundary_radius=int(_m1(cfg, "V503_FACTUAL_BOUNDARY_RADIUS", 2)),
        failure_dice=float(_m1(cfg, "V503_FAILURE_DICE", 0.55)),
        include_global_action=False,
    )
    cause_target = factual["cause_targets"]
    action_target = factual["action_target"].long()
    base_hard = factual["base_hard"]
    base_correct = factual["base_correct"]
    fp_target = cause_target[:, 0:1] + cause_target[:, 2:3]
    fn_target = cause_target[:, 1:2] + cause_target[:, 3:4]

    # ------------------------------------------------------------ M1 locator.
    cause_logits = aux["error_cause_map_logits"]
    cause_probs = aux["error_cause_map_probs"].clamp(EPS, 1.0 - EPS)
    if cause_logits.shape[-2:] != gt.shape[-2:]:
        cause_logits = F.interpolate(
            cause_logits, size=gt.shape[-2:], mode="bilinear", align_corners=False
        )
        cause_probs = torch.sigmoid(cause_logits)
    pos = cause_target.flatten(2).sum(dim=2)
    neg = cause_target.shape[-2] * cause_target.shape[-1] - pos
    pos_weight = (neg / pos.clamp_min(1.0)).clamp(
        1.0, float(_m1(cfg, "V531_CAUSE_POS_WEIGHT_CAP", 20.0))
    )
    cause_bce_terms = []
    for action_index in range(4):
        cause_bce_terms.append(
            F.binary_cross_entropy_with_logits(
                cause_logits[:, action_index],
                cause_target[:, action_index],
                pos_weight=pos_weight[:, action_index].mean().detach(),
            )
        )
    cause_bce = torch.stack(cause_bce_terms).mean()
    cause_dice = _v531_channel_dice_loss(cause_probs, cause_target)
    cause_loss = cause_bce + cause_dice

    alpha = aux["local_primary_action_alpha"].clamp(EPS, 1.0 - EPS)
    if alpha.shape[-2:] != gt.shape[-2:]:
        alpha = F.interpolate(
            alpha, size=gt.shape[-2:], mode="bilinear", align_corners=False
        )
    alpha_weight = 1.0 + float(_m1(cfg, "V531_ALPHA_POS_WEIGHT", 4.0)) * cause_target
    alpha_bce = (
        F.binary_cross_entropy(alpha, cause_target, reduction="none")
        * alpha_weight
    ).mean()
    alpha_dice = _v531_channel_dice_loss(alpha, cause_target)
    alpha_loss = alpha_bce + alpha_dice
    alpha_outside_loss = (alpha * (1.0 - cause_target)).mean()

    # -------------------------------------------------------- spatial routing.
    route_logits = aux["m2_route_logits"]
    route_probs = aux["m2_route_probs"].clamp(EPS, 1.0 - EPS)
    route_ce = F.cross_entropy(route_logits, action_target, reduction="none")
    target_probability = route_probs.gather(1, action_target[:, None])[:, 0]
    focal_gamma = float(_m1(cfg, "V531_ROUTE_FOCAL_GAMMA", 1.0))
    focal = (1.0 - target_probability).pow(focal_gamma)
    preserve_weight = float(_m1(cfg, "V531_ROUTE_PRESERVE_WEIGHT", 0.10))
    route_weight = torch.where(
        action_target == 0,
        route_ce.new_full((), preserve_weight),
        route_ce.new_ones(()),
    )
    route_loss = (route_ce * focal * route_weight).sum() / route_weight.sum().clamp_min(1.0)

    # -------------------------------------------- calibrated edit correctness.
    correctness_logits = aux["m2_action_correctness_logits"]
    correctness_probs = aux["m2_action_correctness_probs"].clamp(EPS, 1.0 - EPS)
    correctness_target = torch.cat(
        [fp_target, fn_target, fp_target, fn_target], dim=1
    ).clamp(0.0, 1.0)
    correctness_domain = torch.cat(
        [base_hard, 1.0 - base_hard, base_hard, 1.0 - base_hard], dim=1
    )
    risk_floor = float(_m1(cfg, "V531_RISK_DOMAIN_FLOOR", 0.05))
    risk_weight = correctness_domain * (
        risk_floor + alpha.detach() + cause_target
    )
    risk_bce_map = F.binary_cross_entropy_with_logits(
        correctness_logits, correctness_target, reduction="none"
    )
    risk_bce = (risk_bce_map * risk_weight).sum() / risk_weight.sum().clamp_min(1.0)
    risk_brier = (
        (correctness_probs - correctness_target).square() * risk_weight
    ).sum() / risk_weight.sum().clamp_min(1.0)

    log_variance = aux.get("m2_action_log_variance")
    if isinstance(log_variance, torch.Tensor):
        squared_error = (correctness_probs - correctness_target).square().detach()
        uncertainty_map = 0.5 * torch.exp(-log_variance) * squared_error + 0.5 * log_variance
        uncertainty_loss = (uncertainty_map * risk_weight).sum() / risk_weight.sum().clamp_min(1.0)
    else:
        uncertainty_loss = route_loss * 0.0

    # ------------------------------------------------------- final segmentation.
    final_prob = _as_b1hw(aux["m2_fused_probs"]).clamp(EPS, 1.0 - EPS)
    final_bce = F.binary_cross_entropy(final_prob, gt)
    final_dice_loss = _soft_dice_loss(final_prob, gt)
    final_boundary_loss = _boundary_l1(
        final_prob,
        gt,
        radius=int(_m1(cfg, "V531_BOUNDARY_RADIUS", 1)),
    )

    base_dice_case = _soft_dice_probs(c0, gt)[:, 0]
    final_dice_case = _soft_dice_probs(final_prob, gt)[:, 0]
    regret_margin = float(_m1(cfg, "V531_REGRET_MARGIN", 0.0))
    regret_loss = F.relu(base_dice_case - final_dice_case + regret_margin).mean()

    edit_mass = aux["v531_per_action_edit_mass"]
    true_fix_mass_map = edit_mass * correctness_target
    true_harm_mass_map = edit_mass * (1.0 - correctness_target)
    edit_case = edit_mass.flatten(1).sum(dim=1)
    fix_case = true_fix_mass_map.flatten(1).sum(dim=1)
    harm_case = true_harm_mass_map.flatten(1).sum(dim=1)
    changed_case = edit_case > float(_m1(cfg, "V531_CHANGED_CASE_EPS", 1.0e-6))
    harm_fraction_case = harm_case / edit_case.clamp_min(EPS)
    fix_fraction_case = fix_case / edit_case.clamp_min(EPS)
    if bool(changed_case.any().item()):
        harmful_edit_loss = harm_fraction_case[changed_case].mean()
    else:
        harmful_edit_loss = edit_mass.sum() * 0.0

    preserve_loss = (
        (final_prob - c0).abs() * base_correct
    ).sum() / base_correct.sum().clamp_min(1.0)
    edit_fraction_case = (final_prob - c0).abs().flatten(1).mean(dim=1)
    edit_budget = float(_m1(cfg, "V531_MAX_EDIT_FRACTION", 0.035))
    edit_budget_loss = F.relu(edit_fraction_case - edit_budget).mean()

    m1_objective_raw = (
        float(_m1(cfg, "V531_CAUSE_WEIGHT", 1.0)) * cause_loss
        + float(_m1(cfg, "V531_ALPHA_WEIGHT", 1.0)) * alpha_loss
        + float(_m1(cfg, "V531_ALPHA_OUTSIDE_WEIGHT", 0.25)) * alpha_outside_loss
    )
    route_risk_objective_raw = (
        float(_m1(cfg, "V531_ROUTE_WEIGHT", 1.0)) * route_loss
        + float(_m1(cfg, "V531_RISK_BCE_WEIGHT", 0.50)) * risk_bce
        + float(_m1(cfg, "V531_RISK_BRIER_WEIGHT", 0.50)) * risk_brier
        + float(_m1(cfg, "V531_UNCERTAINTY_WEIGHT", 0.05)) * uncertainty_loss
    )
    final_seg_objective_raw = (
        float(_m1(cfg, "V531_FINAL_BCE_WEIGHT", 0.50)) * final_bce
        + float(_m1(cfg, "V531_FINAL_DICE_WEIGHT", 1.00)) * final_dice_loss
        + float(_m1(cfg, "V531_FINAL_BOUNDARY_WEIGHT", 0.25)) * final_boundary_loss
    )
    safety_objective_raw = (
        float(_m1(cfg, "V531_REGRET_WEIGHT", 1.00)) * regret_loss
        + float(_m1(cfg, "V531_HARM_WEIGHT", 0.75)) * harmful_edit_loss
        + float(_m1(cfg, "V531_PRESERVE_WEIGHT", 0.50)) * preserve_loss
        + float(_m1(cfg, "V531_EDIT_BUDGET_WEIGHT", 0.25)) * edit_budget_loss
    )

    m1_objective = m1_scale * m1_objective_raw
    m2_objective = (
        route_scale * route_risk_objective_raw
        + final_scale * final_seg_objective_raw
        + safety_scale * safety_objective_raw
    )
    m3_objective = m2_objective * 0.0
    total = m1_objective + m2_objective

    # ------------------------------------------------------------- diagnostics.
    error_mask = action_target > 0
    preserve_mask = action_target == 0
    selected = route_probs.argmax(dim=1)
    route_error_accuracy = (
        (selected[error_mask] == action_target[error_mask]).float().mean()
        if bool(error_mask.any().item())
        else total.new_zeros(())
    )
    preserve_accuracy = (
        (selected[preserve_mask] == 0).float().mean()
        if bool(preserve_mask.any().item())
        else total.new_zeros(())
    )

    candidate_probs = aux["candidate_probs"].clamp(EPS, 1.0 - EPS)
    candidate_dice = _soft_dice_probs(candidate_probs, gt)
    oracle_dice_case = candidate_dice.max(dim=1).values
    oracle_gain_case = (oracle_dice_case - base_dice_case).clamp_min(0.0)
    final_gain_case = final_dice_case - base_dice_case
    oracle_gain = oracle_gain_case.mean()
    final_gain = final_gain_case.mean()
    capture_ratio = final_gain.clamp_min(0.0) / oracle_gain.clamp_min(EPS)
    improved_rate = (final_gain_case > 1.0e-4).float().mean()
    harmful_case_rate = (final_gain_case < -1.0e-4).float().mean()
    changed_pixel_correctness = (
        fix_case[changed_case].sum() / edit_case[changed_case].sum().clamp_min(EPS)
        if bool(changed_case.any().item())
        else total.new_zeros(())
    )
    outcome_contract_error = aux["v531_outcome_contract_error"].amax()

    diag: Dict[str, torch.Tensor] = {
        "v531_total_loss": total.detach(),
        "v531_m1_objective": m1_objective.detach(),
        "v531_m2_objective": m2_objective.detach(),
        "v531_m1_objective_raw": m1_objective_raw.detach(),
        "v531_route_risk_objective_raw": route_risk_objective_raw.detach(),
        "v531_final_seg_objective_raw": final_seg_objective_raw.detach(),
        "v531_safety_objective_raw": safety_objective_raw.detach(),
        "v531_m1_scale": total.new_tensor(m1_scale),
        "v531_route_scale": total.new_tensor(route_scale),
        "v531_final_scale": total.new_tensor(final_scale),
        "v531_safety_scale": total.new_tensor(safety_scale),
        "v531_m2_global_scale": total.new_tensor(m2_global_scale),
        "v531_cause_loss": cause_loss.detach(),
        "v531_cause_bce": cause_bce.detach(),
        "v531_cause_dice_loss": cause_dice.detach(),
        "v531_alpha_loss": alpha_loss.detach(),
        "v531_alpha_outside_loss": alpha_outside_loss.detach(),
        "v531_route_loss": route_loss.detach(),
        "v531_risk_bce": risk_bce.detach(),
        "v531_risk_brier": risk_brier.detach(),
        "v531_uncertainty_loss": uncertainty_loss.detach(),
        "v531_final_bce": final_bce.detach(),
        "v531_final_dice_loss": final_dice_loss.detach(),
        "v531_final_boundary_loss": final_boundary_loss.detach(),
        "v531_regret_loss": regret_loss.detach(),
        "v531_harmful_edit_loss": harmful_edit_loss.detach(),
        "v531_preserve_loss": preserve_loss.detach(),
        "v531_edit_budget_loss": edit_budget_loss.detach(),
        "v531_base_dice": base_dice_case.mean().detach(),
        "v531_final_dice": final_dice_case.mean().detach(),
        "v531_final_gain": final_gain.detach(),
        "v531_oracle_dice": oracle_dice_case.mean().detach(),
        "v531_oracle_gain": oracle_gain.detach(),
        "v531_oracle_capture_ratio": capture_ratio.detach(),
        "v531_improved_case_rate": improved_rate.detach(),
        "v531_harmful_case_rate": harmful_case_rate.detach(),
        "v531_route_error_accuracy": route_error_accuracy.detach(),
        "v531_preserve_accuracy": preserve_accuracy.detach(),
        "v531_changed_case_rate": changed_case.float().mean().detach(),
        "v531_changed_pixel_correctness": changed_pixel_correctness.detach(),
        "v531_edit_fraction": edit_fraction_case.mean().detach(),
        "v531_fix_fraction": (
            fix_fraction_case[changed_case].mean().detach()
            if bool(changed_case.any().item()) else total.new_zeros(())
        ),
        "v531_harm_fraction": (
            harm_fraction_case[changed_case].mean().detach()
            if bool(changed_case.any().item()) else total.new_zeros(())
        ),
        "v531_outcome_contract_error": outcome_contract_error.detach(),
        "v531_preserve_probability": route_probs[:, 0].mean().detach(),
        "v531_delete_rate": (selected == 1).float().mean().detach(),
        "v531_fill_rate": (selected == 2).float().mean().detach(),
        "v531_trim_rate": (selected == 3).float().mean().detach(),
        "v531_expand_rate": (selected == 4).float().mean().detach(),
        "v503_factual_atomic_causal_enabled": total.new_ones(()),
        "v484_base_dice": base_dice_case.mean().detach(),
        "v484_local_oracle_gain": oracle_gain.detach(),
        "_v490_m1_objective": m1_objective,
        "_v490_m2_objective": m2_objective,
        "_v490_m3_objective": m3_objective,
    }
    return total, diag


def _compute_v489_end_to_end_loss(
    cfg: Any,
    masks: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    epoch: int | None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """V489 joint loss for Base/PVL→M1→M2→M3 end-to-end training.

    Prediction tensors are never detached from the forward graph.  Targets that
    are computed from the model's current candidates are stop-gradient labels;
    otherwise a candidate could reduce its loss by changing the label itself.
    This target stop-gradient does not freeze M1: M1 still receives gradients
    through candidate repair, coverage, M2 composition, M3 final segmentation,
    and the ordinary Base loss in train.py.
    """
    gt = (_as_b1hw(masks) > 0.5).float()
    v505_enabled = bool(
        _m1(cfg, "V505_INTERACTIVE_REGION_CAUSAL_ENABLED", False)
    )
    v524_enabled = bool(
        _m1(cfg, "V524_COUNTERFACTUAL_PROMPTED_REGION_ENABLED", False)
    )
    if v524_enabled:
        return _compute_v524_counterfactual_prompted_region_loss(cfg, gt, aux, epoch)
    v523_enabled = bool(
        _m1(cfg, "V523_SEA_LEVEL_UTILITY_COMPOSER_ENABLED", False)
    )
    if v523_enabled:
        return _compute_v523_sea_level_utility_loss(cfg, gt, aux, epoch)
    v522_enabled = bool(
        _m1(cfg, "V522_PWO_DISTILLED_SEQUENTIAL_LOCAL_ENABLED", False)
    )
    if v522_enabled:
        return _compute_v522_pwo_distilled_sequential_loss(cfg, gt, aux, epoch)
    v521_enabled = bool(
        _m1(cfg, "V521_CANDIDATE_CONDITIONAL_REGION_COMPOSER_ENABLED", False)
    )
    if v521_enabled:
        return _compute_v521_candidate_conditional_region_loss(cfg, gt, aux, epoch)
    v520_enabled = bool(
        _m1(cfg, "V520_GATE_SELECTOR_REGION_COMPOSER_ENABLED", False)
    )
    if v520_enabled:
        return _compute_v520_gate_selector_loss(cfg, gt, aux, epoch)
    v519_enabled = bool(
        _m1(cfg, "V519_FAMILY_RISK_REGION_COMPOSER_ENABLED", False)
    )
    if v519_enabled:
        return _compute_v519_region_composer_loss(cfg, gt, aux, epoch)
    if v505_enabled:
        return _compute_v505_interactive_region_loss(cfg, gt, aux, epoch)
    v490_enabled = bool(_m1(cfg, "V490_ROOT_CAUSE_ENABLED", False))
    v491_enabled = bool(
        _m1(cfg, "V491_PRESERVE_FIRST_ENABLED", False)
    ) and v490_enabled
    v492_enabled = bool(
        _m1(cfg, "V492_CAUSAL_LOCAL_EDITOR_ENABLED", False)
    ) and v491_enabled
    v493_enabled = bool(
        _m1(cfg, "V493_CANDIDATE_CONDITIONED_CAUSAL_ENABLED", False)
    ) and v492_enabled
    v494_enabled = bool(
        _m1(cfg, "V494_DIRECT_CAUSAL_DOSE_ENABLED", False)
    ) and v493_enabled
    v495_enabled = bool(
        _m1(cfg, "V495_SINGLE_DECISION_ENABLED", False)
    ) and v494_enabled
    v498_enabled = bool(
        _m1(cfg, "V498_CONSISTENT_FULL_PROPOSAL_ENABLED", False)
    ) and v495_enabled
    v499_enabled = bool(
        _m1(cfg, "V499_CAUSAL_PRESERVE_REFERENCE_ENABLED", False)
    ) and v498_enabled
    v500_enabled = bool(
        _m1(cfg, "V500_HIERARCHICAL_SAFE_ROUTE_ENABLED", False)
    ) and v499_enabled
    v501_enabled = bool(
        _m1(cfg, "V501_BASE_ANCHORED_SELECTIVE_REPAIR_ENABLED", False)
    ) and v500_enabled
    v502_enabled = bool(
        _m1(cfg, "V502_HIERARCHICAL_UTILITY_SOFT_ROUTER_ENABLED", False)
    ) and v501_enabled
    v504_enabled = bool(
        _m1(cfg, "V504_REALIZABLE_POTENTIAL_OUTCOME_ENABLED", False)
    ) and v502_enabled
    v503_enabled = (
        bool(_m1(cfg, "V503_FACTUAL_ATOMIC_CAUSAL_ENABLED", False))
        or v504_enabled
    ) and v502_enabled
    optimal_convex_enabled = bool(
        _m1(cfg, "V490_OPTIMAL_CONVEX_TEACHER_ENABLED", False)
    )
    scale_invariant_risk_enabled = bool(
        _m1(cfg, "V490_SCALE_INVARIANT_RISK_ENABLED", False)
    )
    candidate_probs = aux.get("candidate_probs")
    if not isinstance(candidate_probs, torch.Tensor) or candidate_probs.ndim != 4:
        raise RuntimeError("V489/V490 requires aux['candidate_probs'] with shape [B,K,H,W].")
    candidate_probs = candidate_probs.clamp(EPS, 1.0 - EPS)
    if candidate_probs.shape[1] < 2:
        raise RuntimeError("V489 requires Preserve plus at least one M1 candidate.")
    c0 = candidate_probs[:, :1]
    nonbase = candidate_probs[:, 1:]
    m2_teacher_nonbase = nonbase
    if v492_enabled:
        effective = aux.get("m2_effective_candidates")
        if not isinstance(effective, torch.Tensor) or effective.shape != nonbase.shape:
            raise RuntimeError(
                "V492 requires support-constrained m2_effective_candidates "
                f"with shape {tuple(nonbase.shape)}"
            )
        m2_teacher_nonbase = effective.clamp(EPS, 1.0 - EPS)
    b, k, h, w = candidate_probs.shape
    n = k - 1

    required = [
        "m2_edit_gate_logit",
        "m2_edit_gate_prob",
        "m2_route_logits",
        "m2_route_probs",
        "m2_convex_probs",
        "m2_fused_probs",
        "m2_residual_map",
        "m3_expert_probs",
        "m3_pixel_weights",
        "final_probs",
    ]
    if v490_enabled:
        required.extend([
            "m3_predicted_risk_map",
            "m3_edit_relevance",
            "m3_intervention_mask",
        ])
        if v491_enabled:
            required.append("m3_predicted_log_variance_map")
        if v492_enabled:
            required.extend([
                "m2_benefit_logit",
                "m2_benefit_prob",
                "m2_amplitude_logit",
                "m2_amplitude_prob",
                "m2_structural_support",
                "m2_effective_candidates",
                "m2_candidate_utility_logits",
                "m2_route_policy",
            ])
            if v493_enabled:
                required.extend([
                    "m2_candidate_utility_map_logits",
                    "m2_candidate_utility_map_prob",
                    "m2_selected_utility_logit",
                ])
                if v494_enabled:
                    required.append("m2_direct_causal_accept")
                if v495_enabled:
                    required.extend(["m2_proposal_probs", "m2_training_probs"])
                if v498_enabled:
                    required.extend([
                        "m2_action_logits",
                        "m2_action_probs",
                        "m2_action_policy",
                        "m2_action_index",
                        "m2_preserve_reference_prob",
                    ])
                    if v500_enabled:
                        required.extend([
                            "m2_candidate_presence_logit",
                            "m2_candidate_presence_prob",
                        ])
                        if v501_enabled:
                            required.extend([
                                "m2_full_proposal_probs",
                                "m2_noop_gate_prob",
                                "m2_noop_gate",
                            ])
                            if v502_enabled:
                                required.extend([
                                    "m2_soft_router_enabled",
                                    "m2_route_entropy",
                                    "m2_route_margin",
                                    "m2_candidate_variance",
                                ])
    else:
        required.append("m3_region_logits")
    missing = [key for key in required if key not in aux]
    if missing:
        raise RuntimeError("V489 loss missing outputs: " + str(missing))

    # ------------------------------------------------------------------
    # Stop-gradient continuous correctability targets for M2.
    # V490 deliberately removes the self-referential hard decision mask: the
    # amount of supervision is the actual continuous improvement/harm supplied
    # by the current candidate set, not a thresholded declaration that the
    # router has already succeeded.
    # ------------------------------------------------------------------
    with torch.no_grad():
        c0_target = c0.detach()
        nonbase_target = m2_teacher_nonbase.detach()
        base_abs_error = (c0_target - gt).abs()
        candidate_abs_error = (
            nonbase_target - gt.expand(-1, n, -1, -1)
        ).abs()
        signed_improvement = base_abs_error - candidate_abs_error
        improvement = signed_improvement.clamp_min(0.0)
        harm = (-signed_improvement).clamp_min(0.0)
        best_improvement = improvement.max(dim=1, keepdim=True).values
        worst_harm = harm.max(dim=1, keepdim=True).values

        relative_improvement = (
            improvement / base_abs_error.clamp_min(EPS)
        ).clamp(0.0, 1.0)
        relative_harm = (
            harm / (1.0 - base_abs_error).clamp_min(EPS)
        ).clamp(0.0, 1.0)
        soft_oracle_target = None
        optimal_gate_target = c0_target.new_zeros(c0_target.shape)
        optimal_gain_relevance = c0_target.new_zeros(c0_target.shape)
        best_achievable_gain = improvement.max(dim=1, keepdim=True).values
        candidate_achievable_gain = improvement
        candidate_alpha = c0_target.new_zeros(nonbase_target.shape)

        if v490_enabled and optimal_convex_enabled:
            optimal = _v490_optimal_convex_targets(
                c0_target, nonbase_target, gt
            )
            gate_target = optimal["gate_target"]
            route_target = optimal["route_target"]
            positive_relevance = optimal["positive_relevance"]
            relative_improvement = optimal["candidate_relative_gain"]
            soft_oracle_target = optimal["soft_oracle_target"]
            optimal_gate_target = gate_target
            optimal_gain_relevance = positive_relevance
            best_achievable_gain = optimal["best_achievable_gain"]
            candidate_achievable_gain = optimal["candidate_achievable_gain"]
            candidate_alpha = optimal["candidate_alpha"]
        else:
            gate_target = relative_improvement.max(dim=1, keepdim=True).values
            route_sum = relative_improvement.sum(dim=1, keepdim=True)
            route_target = relative_improvement / route_sum.clamp_min(EPS)
            positive_relevance = gate_target

        # V499 restricts candidate actions to the factual FP/FN/boundary error
        # geometry and to non-trivial candidate edits.  This removes the V498
        # degeneracy where soft candidate tails made Preserve nearly absent.
        if v499_enabled:
            v499_targets = _v499_causal_action_targets(
                c0_target,
                nonbase_target,
                gt,
                candidate_achievable_gain,
                edit_eps=float(_m1(cfg, "V489_DECISION_EDIT_EPS", 0.005)),
                boundary_radius=int(_m1(cfg, "V489_BOUNDARY_RADIUS", 1)),
            )
            v499_action_target = v499_targets["action_target"]
            v499_route_target = v499_targets["route_target"]
            v499_oracle_target = v499_targets["oracle_target"]
            v499_causal_region = v499_targets["causal_region"]
            v499_candidate_eligible = v499_targets["eligible_candidates"]
            v499_positive_relevance = (
                positive_relevance * v499_causal_region
            ).detach()
            v500_safe_route_target = v499_targets["safe_route_target"]
            v500_safe_route_weight = v499_targets["safe_route_weight"]
            v500_safe_route_index = v499_targets["safe_route_index"]
            v500_safe_min_error = v499_targets["safe_min_error"]
            v500_meaningful_edit_rate_target = v499_targets[
                "meaningful_edit_rate"
            ]
        else:
            v499_targets = None
            v499_action_target = None
            v499_route_target = route_target
            v499_oracle_target = soft_oracle_target
            v499_causal_region = torch.ones_like(c0_target)
            v499_candidate_eligible = torch.ones_like(
                candidate_achievable_gain, dtype=torch.bool
            )
            v499_positive_relevance = positive_relevance
            v500_safe_route_target = route_target
            v500_safe_route_weight = torch.zeros_like(c0_target)
            v500_safe_route_index = route_target.argmax(dim=1, keepdim=True)
            v500_safe_min_error = base_abs_error
            v500_meaningful_edit_rate_target = candidate_probs.new_zeros(())

        # V501 uses a continuous, prior-preserving estimate of *relative*
        # improvement.  The target is zero unless a causally eligible candidate
        # clears a minimum relative-gain margin.  Unlike the old balanced BCE,
        # this target does not pretend that sparse positive and abundant
        # negative pixels have a 50/50 prior.
        if v501_enabled:
            eligible_gain = candidate_achievable_gain * (
                v499_candidate_eligible.to(candidate_achievable_gain.dtype)
                if v499_enabled else 1.0
            )
            best_eligible_gain = eligible_gain.max(dim=1, keepdim=True).values
            relative_best_gain = (
                best_eligible_gain / base_abs_error.clamp_min(EPS)
            ).clamp(0.0, 1.0)
            minimum_relative_gain = max(
                float(_m1(cfg, "V501_M2_MIN_RELATIVE_GAIN", 0.05)), 0.0
            )
            minimum_absolute_gain = max(
                float(_m1(cfg, "V501_M2_MIN_ABSOLUTE_GAIN", 0.002)), 0.0
            )
            # A relative ratio can be large when the factual error is tiny.
            # Requiring an absolute gain as well prevents such numerically large
            # but clinically meaningless ratios from opening the edit gate.
            v501_gain_is_meaningful = (
                (relative_best_gain >= minimum_relative_gain)
                & (best_eligible_gain >= minimum_absolute_gain)
            )
            v501_gate_target = torch.where(
                v501_gain_is_meaningful,
                relative_best_gain,
                torch.zeros_like(relative_best_gain),
            ) * v499_causal_region
            gate_target = v501_gate_target.detach()
            positive_relevance = gate_target
            v499_positive_relevance = gate_target
        else:
            v501_gate_target = gate_target

        max_abs_edit = (
            nonbase_target - c0_target
        ).abs().max(dim=1, keepdim=True).values
        negative_relevance = torch.maximum(
            relative_harm.max(dim=1, keepdim=True).values,
            max_abs_edit * (1.0 - positive_relevance),
        ).clamp(0.0, 1.0)
        supervision_weight = (
            positive_relevance + negative_relevance
        ).clamp(0.0, 1.0)

        # Retained only as a diagnostic name for existing log readers.  In V490
        # this is a continuous supervision-mass map, not a hard loss mask.
        if v490_enabled:
            decision_region = supervision_weight
            hard_negative = negative_relevance
        else:
            theta = float(_m1(cfg, "V489_GATE_TARGET_THRESHOLD", 0.05))
            tau = max(
                float(_m1(cfg, "V489_GATE_TARGET_TEMPERATURE", 0.10)),
                1.0e-4,
            )
            raw_target = torch.sigmoid((best_improvement - theta) / tau)
            zero_level = torch.sigmoid(
                best_improvement.new_tensor(-theta / tau)
            )
            gate_target = (
                (raw_target - zero_level)
                / (1.0 - zero_level).clamp_min(EPS)
            ).clamp(0.0, 1.0)
            gate_target = gate_target * (
                best_improvement > 0.0
            ).to(gate_target.dtype)
            route_sum = relative_improvement.sum(dim=1, keepdim=True)
            uniform_route = torch.full_like(
                relative_improvement, 1.0 / float(max(n, 1))
            )
            route_target = torch.where(
                route_sum > EPS,
                relative_improvement / route_sum.clamp_min(EPS),
                uniform_route,
            )
            decision_eps = float(
                _m1(cfg, "V489_DECISION_EDIT_EPS", 0.005)
            )
            decision_seed = (
                (max_abs_edit > decision_eps) | (best_improvement > 0.0)
            ).float()
            radius_ratio = float(
                _m1(cfg, "V489_DECISION_RADIUS_RATIO", 0.01)
            )
            radius_min = int(_m1(cfg, "V489_DECISION_RADIUS_MIN", 1))
            radius_max = int(_m1(cfg, "V489_DECISION_RADIUS_MAX", 4))
            radius = int(round(radius_ratio * float(min(h, w))))
            radius = max(radius_min, min(radius_max, radius))
            decision_region = _v489_dilate(decision_seed, radius)
            c0_hard = c0_target >= 0.5
            gt_hard = gt >= 0.5
            base_correct = (c0_hard == gt_hard).float()
            hard_negative = base_correct * decision_region

        # M2 teacher is constructed from the same continuous improvement
        # definition used by Gate and Route.  This removes the previous
        # contradiction where Gate/Route rewarded a probability improvement
        # that did not yet cross 0.5 while the hard-PWO loss forced Preserve.
        if v490_enabled and soft_oracle_target is None:
            preserve_weight = 1.0 - gate_target
            nonbase_weight = gate_target * route_target
            soft_oracle_weights = torch.cat(
                [preserve_weight, nonbase_weight], dim=1
            )
            teacher_candidate_stack = torch.cat(
                [c0_target, nonbase_target], dim=1
            )
            soft_oracle_target = (
                soft_oracle_weights * teacher_candidate_stack
            ).sum(dim=1, keepdim=True).detach()

        # Hard PWO remains an evaluation upper-bound diagnostic only.
        targets = _v488_pixel_targets(
            candidate_probs.detach(),
            gt,
            tie_edit_penalty=float(
                _m1(cfg, "V489_PWO_TIE_EDIT_PENALTY", 2.0)
            ),
        )
        pwo_target = targets["pwo_target"]

    # ----------------------------- M1 supervision and candidate protection.
    local_supports = aux.get("local_supports")
    if not isinstance(local_supports, torch.Tensor):
        local_supports = (nonbase[:, : min(4, n)] - c0).abs()
    local_supports = local_supports[:, : min(4, n)].clamp(0.0, 1.0)
    local_probs = nonbase[:, : min(4, n)]
    local_count = local_probs.shape[1]
    c0_hard_float = (c0.detach() >= 0.5).float()

    if v503_enabled:
        factual = build_factual_cause_targets(
            c0.detach(),
            gt,
            boundary_radius=int(_m1(cfg, "V503_FACTUAL_BOUNDARY_RADIUS", 2)),
            failure_dice=float(_m1(cfg, "V503_FAILURE_DICE", 0.55)),
            include_global_action=nonbase.shape[1] > 4,
        )
        cause_targets = factual["cause_targets"][:, :local_count]
        labels = torch.where(
            factual["failure_target"][:, 0] > 0.5,
            torch.full_like(factual["failure_target"][:, 0], 4, dtype=torch.long),
            torch.zeros_like(factual["failure_target"][:, 0], dtype=torch.long),
        )
        cause_logits = aux.get("error_cause_map_logits")
        failure_logit = aux.get("failure_state_logit")
        if not isinstance(cause_logits, torch.Tensor) or cause_logits.shape[1] < local_count:
            raise RuntimeError("V503 requires four pixel-level error_cause_map_logits")
        cause_logits = cause_logits[:, :local_count]
        cause_prob = torch.sigmoid(cause_logits)
        pos_weight = float(_m1(cfg, "V503_CAUSE_POS_WEIGHT", 6.0))
        cause_bce = F.binary_cross_entropy_with_logits(
            cause_logits, cause_targets, reduction="none"
        )
        cause_bce = (cause_bce * (1.0 + (pos_weight - 1.0) * cause_targets)).mean()
        cause_dice_terms = []
        for index in range(local_count):
            cause_dice_terms.append(_soft_dice_loss(
                cause_prob[:, index:index + 1], cause_targets[:, index:index + 1]
            ))
        cause_dice = torch.stack(cause_dice_terms).mean() if cause_dice_terms else candidate_probs.sum() * 0.0
        failure_loss = (
            F.binary_cross_entropy_with_logits(
                failure_logit, factual["failure_target"]
            )
            if isinstance(failure_logit, torch.Tensor)
            else candidate_probs.sum() * 0.0
        )
        error_state_loss = cause_bce + cause_dice + float(
            _m1(cfg, "V503_FAILURE_STATE_WEIGHT", 0.5)
        ) * failure_loss
        fp_target = (cause_targets[:, 0:1] + (cause_targets[:, 2:3] if local_count > 2 else 0.0)).clamp(0.0, 1.0)
        fn_target = (cause_targets[:, 1:2] + (cause_targets[:, 3:4] if local_count > 3 else 0.0)).clamp(0.0, 1.0)
        boundary_target = cause_targets[:, 2:4].amax(dim=1, keepdim=True) if local_count > 2 else torch.zeros_like(fp_target)
        boundary_band = boundary_target
        support_losses = [
            _support_loss(
                local_supports[:, index:index + 1],
                cause_targets[:, index:index + 1],
                float(_m1(cfg, "V503_SUPPORT_POS_WEIGHT", 6.0)),
            )
            for index in range(local_count)
        ]
        support_supervision_loss = torch.stack(support_losses).sum() if support_losses else candidate_probs.sum() * 0.0
        repair_pos_weight = float(_m1(cfg, "V503_REPAIR_POS_WEIGHT", 2.0))
        valid_regions = [cause_targets[:, index:index + 1] for index in range(local_count)]
        repair_losses = [
            _masked_bce_dice(
                local_probs[:, index:index + 1], gt, valid_regions[index], repair_pos_weight
            )
            for index in range(local_count)
        ]
        candidate_repair_loss = torch.stack(repair_losses).sum() if repair_losses else candidate_probs.sum() * 0.0
        v504_executability_loss = candidate_probs.sum() * 0.0
        v504_executable_rates = []
        if v504_enabled and local_count > 0:
            margin = float(_m1(cfg, "V504_EXECUTABILITY_MARGIN", 0.02))
            executable_terms = []
            for index in range(local_count):
                region = cause_targets[:, index:index + 1]
                candidate = local_probs[:, index:index + 1]
                target_y = gt
                positive_guard = F.relu(0.5 + margin - candidate) * target_y
                negative_guard = F.relu(candidate - (0.5 - margin)) * (1.0 - target_y)
                executable_terms.append(_v503_masked_mean(positive_guard + negative_guard, region))
                with torch.no_grad():
                    success = ((candidate >= 0.5) == (target_y >= 0.5)).float()
                    v504_executable_rates.append(_v503_masked_mean(success, region))
            v504_executability_loss = torch.stack(executable_terms).mean()
    else:
        v504_executability_loss = candidate_probs.sum() * 0.0
        v504_executable_rates = []
        labels = ErrorStateHead.make_training_labels(
            c0.detach(),
            gt,
            fp_threshold=float(_m1(cfg, "V485_FP_STATE_THRESHOLD", 0.001)),
            fn_threshold=float(_m1(cfg, "V485_FN_STATE_THRESHOLD", 0.001)),
            boundary_threshold=float(_m1(cfg, "V485_BOUNDARY_STATE_THRESHOLD", 0.003)),
            failure_dice=float(_m1(cfg, "V485_FAILURE_DICE", 0.55)),
        )
        error_logits = aux.get("error_state_logits")
        error_state_loss = (
            F.cross_entropy(error_logits, labels)
            if isinstance(error_logits, torch.Tensor)
            else candidate_probs.sum() * 0.0
        )
        fp_target = (c0_hard_float * (1.0 - gt)).detach()
        fn_target = ((1.0 - c0_hard_float) * gt).detach()
        c0_boundary = _soft_boundary(c0_hard_float, radius=1).detach()
        gt_boundary = _soft_boundary(gt, radius=1).detach()
        boundary_target = (c0_boundary - gt_boundary).abs().clamp(0.0, 1.0).detach()
        boundary_band = torch.maximum(c0_boundary, gt_boundary)
        support_losses = []
        if local_count >= 1:
            support_losses.append(_support_loss(local_supports[:, 0:1], fp_target, float(_m1(cfg, "V485_SUPPORT_POS_WEIGHT", 6.0))))
        if local_count >= 2:
            support_losses.append(_support_loss(local_supports[:, 1:2], fn_target, float(_m1(cfg, "V485_SUPPORT_POS_WEIGHT", 6.0))))
        for index in range(2, local_count):
            support_losses.append(0.5 * _support_loss(local_supports[:, index:index + 1], boundary_target, float(_m1(cfg, "V485_SUPPORT_POS_WEIGHT", 6.0))))
        support_supervision_loss = torch.stack(support_losses).sum() if support_losses else candidate_probs.sum() * 0.0
        repair_losses = []
        valid_regions = []
        repair_pos_weight = float(_m1(cfg, "V486_REPAIR_POS_WEIGHT", 2.0))
        if local_count >= 1:
            delete_region = _merge_regions(fp_target, boundary_band * float(_m1(cfg, "V486_DELETE_BOUNDARY_MIX", 0.25)))
            repair_losses.append(_masked_bce_dice(local_probs[:, 0:1], gt, delete_region, repair_pos_weight))
            valid_regions.append(delete_region)
        if local_count >= 2:
            fill_region = _merge_regions(fn_target, boundary_band * float(_m1(cfg, "V486_FILL_BOUNDARY_MIX", 0.25)))
            repair_losses.append(_masked_bce_dice(local_probs[:, 1:2], gt, fill_region, repair_pos_weight))
            valid_regions.append(fill_region)
        for index in range(2, local_count):
            repair_losses.append(0.5 * _masked_bce_dice(local_probs[:, index:index + 1], gt, boundary_band, repair_pos_weight))
            valid_regions.append(boundary_band)
        candidate_repair_loss = torch.stack(repair_losses).sum() if repair_losses else candidate_probs.sum() * 0.0
    if valid_regions:
        valid_stack = torch.cat(valid_regions, dim=1)
        outside_preserve_loss = ((local_probs - c0).abs() * (1.0 - valid_stack)).mean()
    else:
        outside_preserve_loss = candidate_probs.sum() * 0.0

    # V501 gives the boundary candidates an explicit surface objective.  This
    # complements region Dice/BCE instead of replacing it, which is important
    # because NSD is driven by where the surface lies rather than only by mask
    # overlap.
    v501_m1_boundary_loss = candidate_probs.sum() * 0.0
    if v501_enabled and local_count > 2:
        boundary_terms = [
            _boundary_l1(
                local_probs[:, index:index + 1],
                gt,
                radius=int(_m1(cfg, "V501_BOUNDARY_RADIUS", 2)),
            )
            for index in range(2, local_count)
        ]
        if boundary_terms:
            v501_m1_boundary_loss = torch.stack(boundary_terms).mean()

    # C5 is an independently generated failure-rediscovery candidate.  It is
    # supervised on catastrophic Base cases and is explicitly encouraged to
    # preserve C0 elsewhere.  Training the independent direct probability (not
    # merely a residual around C0) is what makes empty-mask recovery possible.
    v501_global_rediscovery_loss = candidate_probs.sum() * 0.0
    v501_global_preserve_loss = candidate_probs.sum() * 0.0
    v501_failure_case_rate = candidate_probs.new_zeros(())
    if v501_enabled:
        direct_global = aux.get("global_direct_probs")
        global_candidate = aux.get("global_candidate_probs")
        if isinstance(direct_global, torch.Tensor) and direct_global.ndim == 4 and direct_global.shape[1] > 0:
            failure_case = (labels == 4).to(candidate_probs.dtype)[:, None, None, None]
            nonfailure_case = 1.0 - failure_case
            v501_failure_case_rate = failure_case.mean().detach()
            rediscovery_terms = []
            preserve_terms = []
            for index in range(direct_global.shape[1]):
                rediscovery_terms.append(
                    _masked_bce_dice(
                        direct_global[:, index:index + 1],
                        gt,
                        failure_case.expand_as(gt),
                        pos_weight=float(_m1(cfg, "V501_GLOBAL_POS_WEIGHT", 2.0)),
                    )
                )
                if isinstance(global_candidate, torch.Tensor) and global_candidate.shape[1] > index:
                    preserve_terms.append(
                        _v489_masked_mean(
                            (global_candidate[:, index:index + 1] - c0.detach()).abs(),
                            nonfailure_case.expand_as(gt),
                        )
                    )
            if rediscovery_terms:
                v501_global_rediscovery_loss = torch.stack(rediscovery_terms).mean()
            if preserve_terms:
                v501_global_preserve_loss = torch.stack(preserve_terms).mean()

    coverage_temperature = max(float(_m1(cfg, "V489_COVERAGE_TEMPERATURE", 0.10)), 1.0e-4)
    all_abs_error = (candidate_probs - gt.expand(-1, k, -1, -1)).abs()
    softmin_error = (
        -coverage_temperature * torch.logsumexp(-all_abs_error / coverage_temperature, dim=1, keepdim=True)
        + coverage_temperature * math.log(float(k))
    )
    base_error_region = ((c0.detach() >= 0.5) != (gt >= 0.5)).float()
    coverage_loss = (
        candidate_probs.sum() * 0.0
        if v503_enabled
        else _v489_masked_mean(softmin_error, base_error_region)
    )

    signed_edits = nonbase - c0
    flat = signed_edits.flatten(2)
    normed = flat / flat.norm(dim=-1, keepdim=True).clamp_min(EPS)
    decor_terms = []
    decor_threshold = float(_m1(cfg, "V489_DECORRELATION_COSINE_THRESHOLD", 0.80))
    for i in range(n):
        for j in range(i + 1, n):
            cosine = (normed[:, i] * normed[:, j]).sum(dim=-1)
            active = ((flat[:, i].abs().sum(dim=-1) > EPS) & (flat[:, j].abs().sum(dim=-1) > EPS)).float()
            decor_terms.append((F.relu(cosine - decor_threshold) * active).sum() / active.sum().clamp_min(1.0))
    decorrelation_loss = (
        candidate_probs.sum() * 0.0
        if v503_enabled
        else (torch.stack(decor_terms).mean() if decor_terms else candidate_probs.sum() * 0.0)
    )

    m1_loss = (
        float(_m1(cfg, "V489_M1_ERROR_STATE_WEIGHT", 1.0)) * error_state_loss
        + float(_m1(cfg, "V489_M1_SUPPORT_WEIGHT", 1.0)) * support_supervision_loss
        + float(_m1(cfg, "V489_M1_REPAIR_WEIGHT", 1.0)) * candidate_repair_loss
        + float(_m1(cfg, "V489_M1_OUTSIDE_PRESERVE_WEIGHT", 0.5)) * outside_preserve_loss
        + float(_m1(cfg, "V489_COVERAGE_WEIGHT", 0.5)) * coverage_loss
        + float(_m1(cfg, "V489_DECORRELATION_WEIGHT", 0.05)) * decorrelation_loss
        + float(_m1(cfg, "V501_M1_BOUNDARY_LOSS_WEIGHT", 0.25))
        * v501_m1_boundary_loss
        + float(_m1(cfg, "V501_GLOBAL_REDISCOVERY_WEIGHT", 1.0))
        * v501_global_rediscovery_loss
        + float(_m1(cfg, "V501_GLOBAL_PRESERVE_WEIGHT", 0.25))
        * v501_global_preserve_loss
        + float(_m1(cfg, "V504_M1_EXECUTABILITY_WEIGHT", 1.0))
        * v504_executability_loss
    )

    # -------------------------------------------------- M2 causal local editing.
    gate_logit = _as_b1hw(aux["m2_edit_gate_logit"])
    gate_prob = _as_b1hw(aux["m2_edit_gate_prob"])
    route_logits = aux["m2_route_logits"]
    m2_convex = _as_b1hw(aux["m2_convex_probs"])
    m2_fused = _as_b1hw(aux["m2_fused_probs"])
    m2_training = _as_b1hw(
        aux.get("m2_training_probs", aux["m2_fused_probs"])
    )
    # FIX: M2 optimization target MUST be the gated (gate-filtered) output,
    # NOT the raw proposal. When the gate closes (gate=0), m2_objective_probs
    # equals c0, gradients from M2 losses stop flowing to the proposal
    # generator. This breaks the adversarial deadlock where:
    #   - Gate learns to close (minimize edits)
    #   - Proposal generator is forced to match GT (maximize edits)
    # Using m2_training_probs (c0 + gate * delta) ensures both modules
    # optimize in the same direction.
    if v495_enabled:
        # m2_training_probs is already the gated result: c0 + gate * (proposal - c0)
        m2_objective_probs = _as_b1hw(aux["m2_training_probs"])
    else:
        m2_objective_probs = m2_training if v495_enabled else m2_fused
    residual_map = _as_b1hw(aux["m2_residual_map"])

    # V492-specific diagnostics/losses default to graph-connected zeros so the
    # legacy V489/V490 branch remains fully compatible.
    v492_benefit_loss = candidate_probs.sum() * 0.0
    v492_benefit_rank_loss = candidate_probs.sum() * 0.0
    v492_amplitude_loss = candidate_probs.sum() * 0.0
    v492_utility_loss = candidate_probs.sum() * 0.0
    v492_preserve_loss = candidate_probs.sum() * 0.0
    v492_no_harm_loss = candidate_probs.sum() * 0.0
    v492_local_seg_loss = candidate_probs.sum() * 0.0
    v492_local_boundary_loss = candidate_probs.sum() * 0.0
    v492_effective_oracle_loss = candidate_probs.sum() * 0.0
    v492_benefit_prob = gate_prob
    v492_amplitude_prob = torch.ones_like(gate_prob)
    v492_structural_support = aux.get(
        "m2_support_union", torch.zeros_like(gate_prob)
    )
    v493_candidate_utility_loss = candidate_probs.sum() * 0.0
    v493_candidate_utility_reg_loss = candidate_probs.sum() * 0.0
    v493_selected_benefit_loss = candidate_probs.sum() * 0.0
    v493_selected_benefit_rank_loss = candidate_probs.sum() * 0.0
    v493_selected_preserve_loss = candidate_probs.sum() * 0.0
    v493_benefit_conflict_rate = candidate_probs.new_zeros(())
    v493_old_label_overlap_rate = candidate_probs.new_zeros(())
    v493_selected_route_safe_rate = candidate_probs.new_zeros(())
    v493_selected_route_gain = candidate_probs.new_zeros(())
    v493_candidate_utility_accuracy = candidate_probs.new_zeros(())
    v493_candidate_utility_positive_mean = candidate_probs.new_zeros(())
    v493_candidate_utility_negative_mean = candidate_probs.new_zeros(())
    v493_selected_positive_weight = torch.zeros_like(gate_prob)
    v493_selected_negative_weight = torch.zeros_like(gate_prob)
    v493_selected_alpha_target = torch.zeros_like(gate_prob)
    v493_selected_gain_target = torch.zeros_like(gate_prob)
    v494_alpha_full_dose_rate = candidate_probs.new_zeros(())
    v494_direct_accept_positive_mean = candidate_probs.new_zeros(())
    v494_direct_accept_negative_mean = candidate_probs.new_zeros(())
    v494_m3_soft_hard_agreement = candidate_probs.new_zeros(())
    v494_m3_ucb_rejection_gap = candidate_probs.new_zeros(())
    v495_proposal_edit_rate = candidate_probs.new_zeros(())
    v495_training_edit_rate = candidate_probs.new_zeros(())
    v495_actual_to_potential_relevance = candidate_probs.new_zeros(())
    v495_proposal_visibility_ratio = candidate_probs.new_zeros(())
    v495_selected_support_alignment = candidate_probs.new_zeros(())
    v498_action_loss = candidate_probs.sum() * 0.0
    v498_action_accuracy = candidate_probs.new_zeros(())
    v498_preserve_target_rate = candidate_probs.new_zeros(())
    v498_preserve_reference_prob = candidate_probs.new_zeros(())
    v498_full_proposal_no_harm_loss = candidate_probs.sum() * 0.0
    v498_route_expected_harm_loss = candidate_probs.sum() * 0.0
    v498_full_proposal_harm_fraction = candidate_probs.new_zeros(())
    v498_train_deploy_l1 = candidate_probs.new_zeros(())
    v498_train_deploy_alignment = candidate_probs.new_zeros(())
    v499_causal_region_rate = candidate_probs.new_zeros(())
    v499_candidate_target_rate = candidate_probs.new_zeros(())
    v499_eligible_candidate_rate = candidate_probs.new_zeros(())
    v499_target_outside_causal_rate = candidate_probs.new_zeros(())
    v499_preserve_outside_causal_accuracy = candidate_probs.new_zeros(())
    v499_candidate_action_accuracy = candidate_probs.new_zeros(())
    v499_predicted_candidate_rate = candidate_probs.new_zeros(())
    v500_candidate_presence_loss = candidate_probs.sum() * 0.0
    v500_candidate_presence_prob = candidate_probs.new_zeros(())
    v500_candidate_rate_gap = candidate_probs.new_zeros(())
    v500_safe_route_loss = candidate_probs.sum() * 0.0
    v500_route_min_regret_loss = candidate_probs.sum() * 0.0
    v500_safe_route_accuracy = candidate_probs.new_zeros(())
    v500_meaningful_edit_rate = candidate_probs.new_zeros(())
    v501_gate_regression_loss = candidate_probs.sum() * 0.0
    v501_false_edit_loss = candidate_probs.sum() * 0.0
    v501_gate_precision = candidate_probs.new_zeros(())
    v501_gate_recall = candidate_probs.new_zeros(())
    v501_gate_f1 = candidate_probs.new_zeros(())
    v501_gate_false_positive_rate = candidate_probs.new_zeros(())
    v501_stable_boundary_penalty = candidate_probs.sum() * 0.0
    v501_gain_is_meaningful = torch.zeros_like(c0, dtype=torch.bool)

    if v492_enabled:
        benefit_logit = _as_b1hw(aux["m2_benefit_logit"])
        benefit_prob = _as_b1hw(aux["m2_benefit_prob"])
        amplitude_prob = _as_b1hw(aux["m2_amplitude_prob"])
        structural_support = _as_b1hw(aux["m2_structural_support"]).detach()
        utility_logits = aux["m2_candidate_utility_logits"]
        v492_benefit_prob = benefit_prob
        v492_amplitude_prob = amplitude_prob
        v492_structural_support = structural_support
        if v495_enabled:
            proposal_probs = _as_b1hw(aux["m2_proposal_probs"])
            v495_proposal_edit_rate = (proposal_probs - c0).abs().mean().detach()
            v495_training_edit_rate = (m2_training - c0).abs().mean().detach()
        if v498_enabled:
            action_logits = aux["m2_action_logits"]
            action_probs = aux["m2_action_probs"]
            best_gain, best_index = candidate_achievable_gain.max(
                dim=1, keepdim=True
            )
            if v499_enabled:
                action_target = v499_action_target
            else:
                action_target = torch.where(
                    best_gain > EPS,
                    best_index + 1,
                    torch.zeros_like(best_index),
                )[:, 0].long()
            if v500_enabled:
                presence_logit = _as_b1hw(
                    aux["m2_candidate_presence_logit"]
                )
                if v501_enabled:
                    presence_target = v501_gate_target.detach().clamp(0.0, 1.0)
                    presence_prob_map = torch.sigmoid(presence_logit)
                    # FIX: Replace Smooth L1 with Asymmetric Focal BCE Loss.
                    # Smooth L1 fails catastrophically on 99.56% negative /
                    # 0.44% positive targets (v501_m2_gate_target_mean=0.0044).
                    # The regressor learns a global near-zero bias, producing
                    # high false-positive rates.  Focal BCE with dynamic
                    # positive up-weighting forces the model to pay attention
                    # to the rare edit-required pixels.
                    bce = F.binary_cross_entropy(
                        presence_prob_map, presence_target, reduction="none"
                    )
                    # gamma=2 down-weights easy negatives
                    focal_factor = (1.0 - presence_prob_map).pow(2)
                    # Up-weight positive (edit-needed) pixels by 6x
                    positive_weight = 1.0 + 5.0 * presence_target
                    v501_gate_regression_loss = (
                        focal_factor * bce * positive_weight
                    ).mean()
                    # Penalize probability mass on a true no-edit target. This
                    # directly controls the false-edit prior without inventing
                    # an artificial balanced class distribution.
                    no_edit_target = (presence_target <= 0.0).to(presence_prob_map.dtype)
                    v501_false_edit_loss = (
                        presence_prob_map * no_edit_target
                    ).sum() / no_edit_target.sum().clamp_min(1.0)
                    v500_candidate_presence_loss = (
                        v501_gate_regression_loss
                        + float(_m1(cfg, "V501_M2_FALSE_EDIT_WEIGHT", 0.50))
                        * v501_false_edit_loss
                    )
                else:
                    presence_target = (action_target > 0).to(
                        candidate_probs.dtype
                    )[:, None]
                    v500_candidate_presence_loss = (
                        _v500_balanced_presence_loss(
                            presence_logit, presence_target
                        )
                    )
                v498_action_loss = v500_candidate_presence_loss
                action_pred = aux["m2_action_index"][:, 0].long()
                v500_candidate_presence_prob = _as_b1hw(
                    aux["m2_candidate_presence_prob"]
                ).mean().detach()
                v500_candidate_rate_gap = (
                    v500_candidate_presence_prob
                    - presence_target.mean().detach()
                )
                if v501_enabled:
                    threshold = float(_m1(cfg, "V501_M2_GATE_THRESHOLD", 0.05))
                    pred_positive = presence_prob_map >= threshold
                    true_positive = presence_target >= threshold
                    tp = (pred_positive & true_positive).float().sum()
                    fp = (pred_positive & (~true_positive)).float().sum()
                    fn = ((~pred_positive) & true_positive).float().sum()
                    tn = ((~pred_positive) & (~true_positive)).float().sum()
                    v501_gate_precision = (tp / (tp + fp).clamp_min(1.0)).detach()
                    v501_gate_recall = (tp / (tp + fn).clamp_min(1.0)).detach()
                    v501_gate_f1 = (
                        2.0 * v501_gate_precision * v501_gate_recall
                        / (v501_gate_precision + v501_gate_recall).clamp_min(EPS)
                    ).detach()
                    v501_gate_false_positive_rate = (
                        fp / (fp + tn).clamp_min(1.0)
                    ).detach()
            else:
                v498_action_loss = _v498_balanced_action_loss(
                    action_logits, action_target
                )
                action_pred = action_logits.argmax(dim=1)
            v498_action_accuracy = (
                action_pred == action_target
            ).float().mean().detach()
            v498_preserve_target_rate = (
                action_target == 0
            ).float().mean().detach()
            v498_preserve_reference_prob = action_probs[:, :1].mean().detach()
            if v499_enabled:
                v499_causal_region_rate = v499_targets[
                    "causal_region_rate"
                ]
                v499_candidate_target_rate = v499_targets[
                    "candidate_target_rate"
                ]
                v499_eligible_candidate_rate = v499_targets[
                    "eligible_candidate_rate"
                ]
                outside = v499_causal_region[:, 0] <= 0.0
                outside_mass = outside.float().sum().clamp_min(1.0)
                v499_target_outside_causal_rate = (
                    ((action_target > 0) & outside).float().sum() / outside_mass
                ).detach()
                v499_preserve_outside_causal_accuracy = (
                    ((action_pred == 0) & outside).float().sum() / outside_mass
                ).detach()
                candidate_target = action_target > 0
                candidate_mass = candidate_target.float().sum().clamp_min(1.0)
                v499_candidate_action_accuracy = (
                    ((action_pred == action_target) & candidate_target)
                    .float().sum() / candidate_mass
                ).detach()
                v499_predicted_candidate_rate = (
                    action_pred > 0
                ).float().mean().detach()
            v498_train_deploy_l1 = (
                m2_training - proposal_probs
            ).abs().mean().detach()
            v498_train_deploy_alignment = (
                1.0 - v498_train_deploy_l1
            ).clamp(0.0, 1.0)

        # Individual treatment effect: each M1 candidate is a distinct
        # intervention and C0 is factual. V493 makes the global existence label
        # and the actually selected intervention label mutually exclusive.
        benefit_target = (
            (v499_action_target[:, None] > 0).to(candidate_probs.dtype)
            if v499_enabled
            else (best_achievable_gain > EPS).to(candidate_probs.dtype)
        )
        if v493_enabled:
            # Eliminate the V492 contradiction: a pixel with one beneficial and
            # one harmful candidate must never be simultaneously positive and
            # negative for the same Benefit logit.
            positive_weight, negative_weight = (
                _v493_mutually_exclusive_benefit_weights(
                    v499_positive_relevance,
                    negative_relevance,
                    benefit_target,
                )
            )
            v493_old_label_overlap_rate = (
                (positive_relevance > 0.0) & (negative_relevance > 0.0)
            ).float().mean().detach()
            v493_benefit_conflict_rate = (
                (positive_weight > 0.0) & (negative_weight > 0.0)
            ).float().mean().detach()

            route_policy = aux["m2_route_policy"].detach()
            selected_index = route_policy.argmax(dim=1, keepdim=True)
            selected_gain = candidate_achievable_gain.gather(1, selected_index)
            selected_relative_gain = relative_improvement.gather(1, selected_index)
            selected_alpha = candidate_alpha.gather(1, selected_index)
            candidate_abs_edit = (m2_teacher_nonbase.detach() - c0.detach()).abs()
            selected_abs_edit = candidate_abs_edit.gather(1, selected_index)
            selected_eligible = v499_candidate_eligible.gather(
                1, selected_index
            )
            selected_positive = (
                (selected_gain > EPS) & selected_eligible
            ).to(candidate_probs.dtype)
            selected_positive_weight = selected_relative_gain * selected_positive
            selected_negative_weight = selected_abs_edit * (1.0 - selected_positive)
            v493_selected_positive_weight = selected_positive_weight
            v493_selected_negative_weight = selected_negative_weight
            v493_selected_alpha_target = selected_alpha
            v493_selected_gain_target = selected_gain

            oracle_benefit_loss = _v492_balanced_benefit_loss(
                benefit_logit, positive_weight, negative_weight
            )
            selected_benefit_loss = _v492_balanced_benefit_loss(
                benefit_logit, selected_positive_weight, selected_negative_weight
            )
            v493_selected_benefit_loss = selected_benefit_loss
            oracle_rank = _v492_benefit_rank_loss(
                benefit_logit, positive_weight, negative_weight
            )
            selected_rank = _v492_benefit_rank_loss(
                benefit_logit, selected_positive_weight, selected_negative_weight
            )
            v493_selected_benefit_rank_loss = selected_rank

            if v494_enabled:
                # The deployed Benefit logit is already the selected candidate's
                # causal utility. Training it again against an oracle-existence
                # target would reintroduce the cross-candidate contradiction.
                v492_benefit_loss = selected_benefit_loss
                v492_benefit_rank_loss = selected_rank
                # For binary GT and bounded probabilities, every strictly
                # improving candidate has full-dose alpha=1.  The learned V493
                # amplitude was therefore a redundant attenuator.  Keep the
                # compatibility loss at exact zero and audit the theorem below.
                v492_amplitude_loss = amplitude_prob.sum() * 0.0
                positive_alpha_mass = selected_positive_weight.sum()
                if bool((positive_alpha_mass > 0).item()):
                    v494_alpha_full_dose_rate = (
                        ((selected_alpha >= 1.0 - 1.0e-5).to(selected_alpha.dtype)
                         * selected_positive_weight).sum()
                        / positive_alpha_mass.clamp_min(1.0)
                    ).detach()
                v494_direct_accept_positive_mean = _v489_masked_mean(
                    benefit_prob, selected_positive_weight
                ).detach()
                v494_direct_accept_negative_mean = _v489_masked_mean(
                    benefit_prob, selected_negative_weight
                ).detach()
            else:
                v492_benefit_loss = 0.5 * (
                    oracle_benefit_loss + selected_benefit_loss
                )
                v492_benefit_rank_loss = 0.5 * (oracle_rank + selected_rank)
                v492_amplitude_loss = _v489_masked_mean(
                    F.smooth_l1_loss(
                        amplitude_prob,
                        selected_alpha,
                        reduction="none",
                        beta=float(_m1(cfg, "V492_M2_AMPLITUDE_BETA", 0.10)),
                    ),
                    selected_positive_weight,
                )

            # Direct candidate-conditioned utility supervision. Positive and
            # negative candidates are distinct tensors, so one harmful option
            # cannot suppress a different beneficial option at the same pixel.
            utility_map_logits = aux["m2_candidate_utility_map_logits"]
            candidate_positive = (
                (candidate_achievable_gain > EPS) & v499_candidate_eligible
            ).to(candidate_probs.dtype)
            candidate_positive_weight = relative_improvement * candidate_positive
            candidate_negative_weight = candidate_abs_edit * (1.0 - candidate_positive)
            v493_candidate_utility_loss = _v493_balanced_candidate_utility_loss(
                utility_map_logits,
                candidate_positive_weight,
                candidate_negative_weight,
            )
            utility_prob = torch.sigmoid(utility_map_logits)
            utility_reg_weight = (
                candidate_positive_weight + candidate_negative_weight
            ).clamp(0.0, 1.0)
            v493_candidate_utility_reg_loss = (
                F.smooth_l1_loss(
                    utility_prob,
                    relative_improvement,
                    reduction="none",
                    beta=0.10,
                ) * utility_reg_weight
            ).sum() / utility_reg_weight.sum().clamp_min(1.0)
            utility_truth = candidate_positive.bool()
            utility_pred = utility_prob >= 0.5
            utility_valid = utility_reg_weight > 0.0
            if bool(utility_valid.any().item()):
                v493_candidate_utility_accuracy = (
                    utility_pred == utility_truth
                )[utility_valid].float().mean().detach()
            v493_candidate_utility_positive_mean = (
                (utility_prob * candidate_positive_weight).sum()
                / candidate_positive_weight.sum().clamp_min(1.0)
            ).detach()
            v493_candidate_utility_negative_mean = (
                (utility_prob * candidate_negative_weight).sum()
                / candidate_negative_weight.sum().clamp_min(1.0)
            ).detach()
            selected_mass = (
                selected_positive_weight + selected_negative_weight
            ).sum().clamp_min(1.0)
            v493_selected_route_safe_rate = (
                selected_positive * (selected_positive_weight + selected_negative_weight)
            ).sum().detach() / selected_mass.detach()
            v493_selected_route_gain = (
                selected_gain * (selected_positive_weight + selected_negative_weight)
            ).sum().detach() / selected_mass.detach()
        else:
            positive_weight = positive_relevance.detach()
            negative_weight = negative_relevance.detach()
            v492_benefit_loss = _v492_balanced_benefit_loss(
                benefit_logit, positive_weight, negative_weight
            )
            v492_benefit_rank_loss = _v492_benefit_rank_loss(
                benefit_logit, positive_weight, negative_weight
            )
            v492_amplitude_loss = _v489_masked_mean(
                F.smooth_l1_loss(
                    amplitude_prob,
                    gate_target,
                    reduction="none",
                    beta=float(_m1(cfg, "V492_M2_AMPLITUDE_BETA", 0.10)),
                ),
                positive_weight,
            )

        route_log_prob = F.log_softmax(route_logits, dim=1)
        route_teacher = v499_route_target if v499_enabled else route_target
        route_element = -(route_teacher * route_log_prob).sum(dim=1, keepdim=True)
        conditional_route_loss = _v489_masked_mean(
            route_element,
            v499_positive_relevance if v499_enabled else positive_weight,
        )
        # Preserve is supervised as a reference action, not deployed in M2.
        # V500 adds a minimum-error route teacher on every meaningful edit so
        # that the mandatory full proposal is useful (or least harmful) even
        # where no candidate is better than Preserve.
        if v500_enabled:
            route_terms = [v500_candidate_presence_loss]
            if bool((v499_positive_relevance.sum() > 0).item()):
                route_terms.append(conditional_route_loss)
            if not v501_enabled:
                safe_route_element = -(
                    v500_safe_route_target * route_log_prob
                ).sum(dim=1, keepdim=True)
                v500_safe_route_loss = _v489_masked_mean(
                    safe_route_element, v500_safe_route_weight
                )
                if bool((v500_safe_route_weight.sum() > 0).item()):
                    route_terms.append(v500_safe_route_loss)
            route_loss = torch.stack(route_terms).mean()
            safe_route_pred = route_logits.argmax(dim=1, keepdim=True)
            safe_valid = (v500_safe_route_weight > 0.0) & (not v501_enabled)
            if bool(safe_valid.any().item()):
                v500_safe_route_accuracy = (
                    (safe_route_pred == v500_safe_route_index)[safe_valid]
                    .float().mean().detach()
                )
            v500_meaningful_edit_rate = (
                v500_meaningful_edit_rate_target.detach()
            )
        else:
            route_loss = (
                0.5 * (conditional_route_loss + v498_action_loss)
                if v498_enabled else conditional_route_loss
            )

        # Image-level candidate utility provides a sparse DES-style prior while
        # pixel-wise routing still determines the local intervention.
        case_gain = (
            candidate_achievable_gain
            * v499_candidate_eligible.to(candidate_achievable_gain.dtype)
            if v499_enabled
            else candidate_achievable_gain
        )
        case_utility = case_gain.flatten(2).sum(dim=-1)
        utility_target = case_utility.argmax(dim=1)
        valid_case = (case_utility.max(dim=1).values > EPS).to(candidate_probs.dtype)
        utility_element = F.cross_entropy(
            utility_logits, utility_target, reduction="none"
        )
        v492_utility_loss = (
            utility_element * valid_case
        ).sum() / valid_case.sum().clamp_min(1.0)

        oracle_teacher = (
            v499_oracle_target if v499_enabled else soft_oracle_target
        )
        v492_effective_oracle_loss = _v489_masked_mean(
            (m2_objective_probs - oracle_teacher).pow(2),
            (
                v499_positive_relevance
                if v499_enabled
                else positive_relevance if v498_enabled else supervision_weight
            ),
        )
        if v493_enabled:
            selected_supervision = (
                v493_selected_positive_weight + v493_selected_negative_weight
            ).clamp(0.0, 1.0)
            preserve_weight = torch.maximum(
                negative_weight, v493_selected_negative_weight
            )
            if v498_enabled:
                # Preserve is learned by the explicit all-action classifier.
                # Forcing a non-Base proposal itself to equal C0 would collapse
                # the candidate bank and contradict M3's sole-decision role.
                v493_selected_preserve_loss = candidate_probs.sum() * 0.0
                v492_preserve_loss = candidate_probs.sum() * 0.0
            else:
                v493_selected_preserve_loss = _v489_masked_mean(
                    (m2_objective_probs - c0).abs(),
                    v493_selected_negative_weight,
                )
                v492_preserve_loss = _v489_masked_mean(
                    (m2_objective_probs - c0).abs(), preserve_weight
                )
            causal_supervision = torch.maximum(
                supervision_weight, selected_supervision
            ).clamp(0.0, 1.0)
        else:
            preserve_weight = negative_weight
            causal_supervision = supervision_weight
        base_pixel_error = (c0.detach() - gt).abs()
        m2_pixel_error = (m2_objective_probs - gt).abs()
        if v498_enabled:
            proposal_edit = (m2_objective_probs - c0.detach()).abs()
            proposal_relevance = torch.maximum(
                structural_support, proposal_edit.detach()
            ).clamp(0.0, 1.0)
            full_harm_map = F.relu(m2_pixel_error - base_pixel_error)
            v498_full_proposal_no_harm_loss = _v489_masked_mean(
                full_harm_map, proposal_relevance
            )

            route_prob = aux["m2_route_probs"]
            candidate_harm_map = F.relu(
                candidate_abs_error
                - base_pixel_error.expand(-1, n, -1, -1)
            )
            expected_harm_map = (
                route_prob * candidate_harm_map
            ).sum(dim=1, keepdim=True)
            route_relevance = max_abs_edit.detach().clamp(0.0, 1.0)
            v498_route_expected_harm_loss = _v489_masked_mean(
                expected_harm_map, route_relevance
            )
            if v500_enabled:
                expected_candidate_error = (
                    route_prob * candidate_abs_error
                ).sum(dim=1, keepdim=True)
                route_regret = F.relu(
                    expected_candidate_error - v500_safe_min_error
                )
                v500_route_min_regret_loss = _v489_masked_mean(
                    route_regret, v500_safe_route_weight
                )
                # Equal averaging preserves the historical scale while adding
                # direct minimum-regret route supervision.
                if v501_enabled:
                    # V501 has a true no-op.  There is no reason to train a
                    # least-bad candidate on negative pixels; doing so was the
                    # source of the mandatory harmful proposal in V500.
                    v500_route_min_regret_loss = candidate_probs.sum() * 0.0
                    v492_no_harm_loss = 0.5 * (
                        v498_full_proposal_no_harm_loss
                        + v498_route_expected_harm_loss
                    )
                else:
                    v492_no_harm_loss = (
                        v498_full_proposal_no_harm_loss
                        + v498_route_expected_harm_loss
                        + v500_route_min_regret_loss
                    ) / 3.0
            else:
                # Equal averaging keeps the historical no-harm scale while
                # supervising both the hard proposal and the soft STE route.
                v492_no_harm_loss = 0.5 * (
                    v498_full_proposal_no_harm_loss
                    + v498_route_expected_harm_loss
                )
            edited = proposal_edit > float(
                _m1(cfg, "V489_DECISION_EDIT_EPS", 0.005)
            )
            if bool(edited.any().item()):
                v498_full_proposal_harm_fraction = (
                    (full_harm_map > EPS)[edited].float().mean().detach()
                )
            positive_supervision = (
                v499_positive_relevance if v499_enabled else positive_relevance
            ).clamp(0.0, 1.0)
            v492_local_seg_loss = _masked_bce_dice(
                m2_objective_probs,
                gt,
                positive_supervision,
                pos_weight=1.0,
            )
            boundary_supervision = positive_supervision
        else:
            v492_no_harm_loss = _v489_masked_mean(
                F.relu(m2_pixel_error - base_pixel_error), causal_supervision
            )
            v492_local_seg_loss = _masked_bce_dice(
                m2_objective_probs, gt, causal_supervision, pos_weight=1.0
            )
            boundary_supervision = causal_supervision
        boundary_element = (
            _soft_boundary(
                m2_objective_probs,
                radius=int(_m1(cfg, "V489_BOUNDARY_RADIUS", 1)),
            )
            - _soft_boundary(
                gt, radius=int(_m1(cfg, "V489_BOUNDARY_RADIUS", 1))
            )
        ).abs()
        v492_local_boundary_loss = _v489_masked_mean(
            boundary_element, boundary_supervision
        )
        if v501_enabled:
            # FIX: Disabled. Penalizing M2 from editing boundaries where Base
            # and GT agree (stable_boundary) is counter-productive in medical
            # imaging: Base boundaries are inherently uncertain and are where
            # the largest DSC gains reside.  This penalty was the primary
            # reason v501_m2_stable_boundary_penalty=0.341 and M2 gain_vs_base
            # was negative (-0.018).  The term is computed for diagnostics but
            # zero-weighted.
            boundary_radius_v501 = int(_m1(cfg, "V501_BOUNDARY_RADIUS", 2))
            base_boundary_soft = _soft_boundary(c0.detach(), radius=boundary_radius_v501)
            gt_boundary_soft = _soft_boundary(gt, radius=boundary_radius_v501)
            stable_boundary = torch.minimum(
                base_boundary_soft, gt_boundary_soft
            ).detach().clamp(0.0, 1.0)
            v501_stable_boundary_penalty = _v489_masked_mean(
                (m2_objective_probs - c0.detach()).abs(), stable_boundary
            ) * 0.0

        # Compatibility aliases consumed by existing diagnostics.
        gate_loss = v492_benefit_loss
        gate_smooth_l1 = v492_amplitude_loss
        correction_loss = _v489_masked_mean(
            (m2_objective_probs - gt).abs(), positive_weight
        )
        harm_loss = v492_preserve_loss + v492_no_harm_loss
        pwo_loss = v492_effective_oracle_loss
        m2_seg_loss = v492_local_seg_loss
        m2_boundary_loss = v492_local_boundary_loss
        residual_l1 = residual_map.abs().mean()
        effect_loss = candidate_probs.sum() * 0.0
        evidence_loss = candidate_probs.sum() * 0.0
        uncertainty_loss = candidate_probs.sum() * 0.0

        m2_loss = (
            float(_m1(cfg, "V492_M2_BENEFIT_WEIGHT", 1.0))
            * v492_benefit_loss
            + float(_m1(cfg, "V492_M2_BENEFIT_RANK_WEIGHT", 0.25))
            * v492_benefit_rank_loss
            + float(_m1(cfg, "V492_M2_AMPLITUDE_WEIGHT", 1.0))
            * v492_amplitude_loss
            + float(_m1(cfg, "V492_M2_ROUTE_WEIGHT", 1.0))
            * route_loss
            + float(_m1(cfg, "V492_M2_UTILITY_WEIGHT", 0.25))
            * v492_utility_loss
            + (
                float(_m1(cfg, "V493_M2_CANDIDATE_UTILITY_WEIGHT", 1.0))
                * v493_candidate_utility_loss
                + float(_m1(cfg, "V493_M2_CANDIDATE_UTILITY_REG_WEIGHT", 0.25))
                * v493_candidate_utility_reg_loss
                + float(_m1(cfg, "V493_M2_SELECTED_PRESERVE_WEIGHT", 1.0))
                * v493_selected_preserve_loss
                if v493_enabled else candidate_probs.sum() * 0.0
            )
            + float(_m1(cfg, "V492_M2_ORACLE_WEIGHT", 1.0))
            * v492_effective_oracle_loss
            + float(_m1(cfg, "V492_M2_PRESERVE_WEIGHT", 1.0))
            * v492_preserve_loss
            + float(_m1(cfg, "V492_M2_NO_HARM_WEIGHT", 1.0))
            * v492_no_harm_loss
            + float(_m1(cfg, "V492_M2_LOCAL_SEG_WEIGHT", 0.25))
            * v492_local_seg_loss
            + float(_m1(cfg, "V492_M2_LOCAL_BOUNDARY_WEIGHT", 0.25))
            * v492_local_boundary_loss
            + float(_m1(cfg, "V501_M2_STABLE_BOUNDARY_WEIGHT", 0.50))
            * v501_stable_boundary_penalty
        )
    else:
        gate_logit = _as_b1hw(aux["m2_edit_gate_logit"])
        gate_prob = _as_b1hw(aux["m2_edit_gate_prob"])
        route_logits = aux["m2_route_logits"]
        m2_convex = _as_b1hw(aux["m2_convex_probs"])
        m2_fused = _as_b1hw(aux["m2_fused_probs"])
        residual_map = _as_b1hw(aux["m2_residual_map"])

        if v490_enabled:
            gate_loss = _v490_balanced_continuous_gate_loss(
                gate_logit,
                gate_target,
                positive_relevance,
                negative_relevance,
            )
            gate_smooth_l1 = _v489_masked_mean(
                F.smooth_l1_loss(
                    gate_prob,
                    gate_target,
                    reduction="none",
                    beta=0.10,
                ),
                supervision_weight,
            )
        else:
            gate_loss = _v489_balanced_soft_bce(
                gate_logit,
                gate_target,
                decision_region,
                positive_threshold=float(
                    _m1(cfg, "V489_GATE_POSITIVE_THRESHOLD", 0.05)
                ),
            )
            gate_smooth_l1 = _v489_masked_mean(
                F.smooth_l1_loss(
                    gate_prob,
                    gate_target,
                    reduction="none",
                    beta=0.10,
                ),
                decision_region,
            )

        route_log_prob = F.log_softmax(route_logits, dim=1)
        route_element = -(route_target * route_log_prob).sum(dim=1, keepdim=True)
        route_weight = positive_relevance if v490_enabled else decision_region * best_improvement
        route_loss = _v489_masked_mean(route_element, route_weight)

        correction_weight = positive_relevance if v490_enabled else decision_region * gate_target
        correction_loss = _v489_masked_mean(
            (m2_fused - gt).abs(), correction_weight
        )
        harm_loss = _v489_masked_mean(
            (m2_fused - c0).abs(), hard_negative
        )
        pwo_loss = _v489_masked_mean(
            (
                m2_convex
                - (soft_oracle_target if v490_enabled else pwo_target)
            ).pow(2),
            supervision_weight if v490_enabled else decision_region,
        )
        m2_seg_loss = _prob_bce_dice(m2_fused, gt)
        boundary_radius = int(_m1(cfg, "V489_BOUNDARY_RADIUS", 1))
        m2_boundary_loss = _boundary_l1(m2_fused, gt, radius=boundary_radius)
        residual_l1 = residual_map.abs().mean()

        effect_loss = candidate_probs.sum() * 0.0
        evidence_loss = candidate_probs.sum() * 0.0
        uncertainty_loss = candidate_probs.sum() * 0.0
        if isinstance(aux.get("m2_pixel_effect"), torch.Tensor):
            effect_target = (
                (c0.detach() - gt).abs()
                - (candidate_probs.detach() - gt.expand(-1, k, -1, -1)).abs()
            )
            effect_loss = F.smooth_l1_loss(aux["m2_pixel_effect"], effect_target, beta=0.10)
        if isinstance(aux.get("m2_pixel_evidence_logit"), torch.Tensor):
            evidence_target = torch.cat([torch.ones_like(gate_target), relative_improvement], dim=1)
            evidence_loss = F.binary_cross_entropy_with_logits(aux["m2_pixel_evidence_logit"], evidence_target)
        if isinstance(aux.get("m2_pixel_log_variance"), torch.Tensor) and isinstance(aux.get("m2_pixel_effect"), torch.Tensor):
            uncertainty_pred = torch.sqrt(F.softplus(aux["m2_pixel_log_variance"].clamp(-6.0, 4.0)) + EPS)
            uncertainty_target = (aux["m2_pixel_effect"].detach() - effect_target).abs().clamp(0.0, 1.0)
            uncertainty_loss = F.smooth_l1_loss(uncertainty_pred, uncertainty_target, beta=0.10)

        m2_loss = (
            float(_m1(cfg, "V489_M2_GATE_WEIGHT", 1.0)) * gate_loss
            + float(_m1(cfg, "V489_M2_GATE_REG_WEIGHT", 0.25)) * gate_smooth_l1
            + float(_m1(cfg, "V489_M2_ROUTE_WEIGHT", 1.0)) * route_loss
            + float(_m1(cfg, "V489_M2_CORRECTION_WEIGHT", 1.0)) * correction_loss
            + float(_m1(cfg, "V489_M2_HARM_WEIGHT", 1.0)) * harm_loss
            + float(_m1(cfg, "V489_M2_PWO_WEIGHT", 0.10)) * pwo_loss
            + float(_m1(cfg, "V489_M2_SEG_WEIGHT", 0.50)) * m2_seg_loss
            + float(_m1(cfg, "V489_M2_BOUNDARY_WEIGHT", 0.50)) * m2_boundary_loss
            + float(_m1(cfg, "V489_M2_RESIDUAL_L1_WEIGHT", 0.25)) * residual_l1
            + float(_m1(cfg, "V489_M2_EFFECT_WEIGHT", 0.10)) * effect_loss
            + float(_m1(cfg, "V489_M2_EVIDENCE_WEIGHT", 0.10)) * evidence_loss
            + float(_m1(cfg, "V489_M2_UNCERTAINTY_WEIGHT", 0.05)) * uncertainty_loss
        )

    v502_diag: Dict[str, torch.Tensor] = {}
    v503_diag: Dict[str, torch.Tensor] = {}
    v504_diag: Dict[str, torch.Tensor] = {}
    if v504_enabled:
        m2_loss, v504_diag = _compute_v504_m2_loss(
            cfg, gt, c0, m2_teacher_nonbase, aux, epoch
        )
        gate_loss = v504_diag["v504_presence_loss"]
        gate_smooth_l1 = v504_diag["v504_candidate_utility_loss"]
        route_loss = v504_diag["v504_route_loss"]
        correction_loss = v504_diag["v504_outcome_repair_loss"]
        harm_loss = v504_diag["v504_exact_noharm_loss"]
        pwo_loss = candidate_probs.sum() * 0.0
        m2_seg_loss = v504_diag["v504_outcome_repair_loss"]
        m2_boundary_loss = v504_diag["v504_boundary_outcome_loss"]
    elif v503_enabled:
        m2_loss, v503_diag = _compute_v503_m2_loss(
            cfg, gt, c0, m2_teacher_nonbase, aux, epoch
        )
        gate_loss = v503_diag["v503_action_loss"]
        gate_smooth_l1 = v503_diag["v503_preserve_action_loss"]
        route_loss = v503_diag["v503_error_action_loss"]
        correction_loss = v503_diag["v503_outcome_repair_loss"]
        harm_loss = v503_diag["v503_exact_noharm_loss"]
        pwo_loss = candidate_probs.sum() * 0.0
        m2_seg_loss = v503_diag["v503_outcome_repair_loss"]
        m2_boundary_loss = v503_diag["v503_boundary_outcome_loss"]
    elif v502_enabled:
        # Legacy V502 is retained for reproducibility only.
        m2_loss, v502_diag = _compute_v502_m2_loss(
            cfg, gt, c0, m2_teacher_nonbase, aux, epoch
        )
        gate_loss = v502_diag["v502_eligibility_loss"]
        gate_smooth_l1 = v502_diag["v502_eligibility_calibration_loss"]
        route_loss = v502_diag["v502_route_teacher_loss"]
        correction_loss = v502_diag["v502_masked_task_loss"]
        harm_loss = (
            v502_diag["v502_expected_candidate_harm_loss"]
            + v502_diag["v502_final_noharm_loss"]
        )
        pwo_loss = candidate_probs.sum() * 0.0
        m2_seg_loss = v502_diag["v502_masked_task_loss"]
        m2_boundary_loss = v502_diag["v502_boundary_violation"]

    # ------------------------------------------ M3 direct expert-risk learning.
    expert_probs = aux["m3_expert_probs"].clamp(EPS, 1.0 - EPS)
    final_probs = _as_b1hw(aux["final_probs"])
    e = expert_probs.shape[1]
    gt_e = gt.expand(-1, e, -1, -1)
    boundary_radius = int(_m1(cfg, "V489_BOUNDARY_RADIUS", 1))
    expert_boundary = _soft_boundary(
        expert_probs.reshape(b * e, 1, h, w), radius=boundary_radius
    ).reshape(b, e, h, w)
    gt_boundary_e = _soft_boundary(
        gt, radius=boundary_radius
    ).expand(-1, e, -1, -1)
    boundary_error = (expert_boundary - gt_boundary_e).abs()
    edit_magnitude = (expert_probs - c0).abs()

    m3_selection_accuracy = candidate_probs.new_zeros(())
    m3_oracle_regret = candidate_probs.new_zeros(())
    m3_pairwise_accuracy = candidate_probs.new_zeros(())
    m3_selection_entropy = candidate_probs.new_zeros(())
    m3_spatial_switch_rate = candidate_probs.new_zeros(())
    m3_risk_mae = candidate_probs.new_zeros(())
    m3_edit_relevance_mean = candidate_probs.new_zeros(())
    m3_oracle_hist = candidate_probs.new_zeros(e)
    m3_predicted_hist = candidate_probs.new_zeros(e)
    m3_teacher_entropy = candidate_probs.new_zeros(())
    m3_teacher_risk_scale = candidate_probs.new_ones(())
    m3_predicted_risk_scale = candidate_probs.new_ones(())
    risk_regression_loss = candidate_probs.sum() * 0.0
    expert_ce_loss = candidate_probs.sum() * 0.0
    rank_loss = candidate_probs.sum() * 0.0
    coherence_loss = candidate_probs.sum() * 0.0
    uncertainty_nll_loss = candidate_probs.sum() * 0.0
    v495_m3_sign_bce_loss = candidate_probs.sum() * 0.0
    v495_m3_sign_accuracy = candidate_probs.new_zeros(())
    v495_m3_sign_positive_rate = candidate_probs.new_zeros(())
    v495_m3_sign_supervision_fraction = candidate_probs.new_zeros(())
    v492_teacher_raw_scale = candidate_probs.new_ones(())

    if v490_enabled:
        # The selector predicts a dense *relative* risk field already centred
        # on Preserve/C0 and aggregated with continuous multi-scale geometry.
        predicted_risk = aux["m3_predicted_risk_map"]
        edit_relevance = _as_b1hw(aux["m3_edit_relevance"]).detach().clamp(0.0, 1.0)
        m3_edit_relevance_mean = edit_relevance.mean()

        pixel_bce_risk = F.binary_cross_entropy(
            expert_probs,
            gt_e,
            reduction="none",
        )
        teacher_absolute_risk = (
            pixel_bce_risk
            + float(_m1(cfg, "V490_M3_TEACHER_BOUNDARY_WEIGHT", 0.50))
            * boundary_error
        )
        teacher_relative_risk = (
            teacher_absolute_risk
            - teacher_absolute_risk[:, :1]
        ).detach()
        teacher_risk = _v490_multiscale_relative_risk(
            teacher_relative_risk,
            edit_relevance,
            _v490_context_scales(cfg),
        ).detach()
        # Keep the raw Preserve-relative risk for the deployment-sign target.
        # Its ambiguity margin reuses the experiment's existing M3 target
        # margin, instead of introducing a separately tuned normalized cutoff.
        teacher_risk_for_sign = teacher_risk
        v492_teacher_raw_scale = teacher_risk.new_ones(())
        if v492_enabled:
            # V493 uses one relevance-weighted scale per image. V492's
            # per-pixel two-expert normalization reduced every non-zero target
            # to +/-1 and erased risk magnitude.
            if v493_enabled and bool(
                _m1(cfg, "V493_M3_IMAGEWISE_RISK_NORMALIZATION", True)
            ):
                teacher_risk, raw_scale = _v493_imagewise_relative_risk_normalize(
                    teacher_risk,
                    edit_relevance,
                    floor=float(_m1(cfg, "V490_M3_RISK_SCALE_FLOOR", 1.0e-3)),
                    clip=float(_m1(cfg, "V493_M3_NORMALIZED_RISK_CLIP", 8.0)),
                )
            else:
                teacher_risk, raw_scale = _v490_normalize_relative_risk(
                    teacher_risk,
                    floor=float(_m1(cfg, "V490_M3_RISK_SCALE_FLOOR", 1.0e-3)),
                )
            v492_teacher_raw_scale = raw_scale.mean().detach()

        if v491_enabled:
            predicted_log_variance = aux["m3_predicted_log_variance_map"]
            if predicted_log_variance.shape != predicted_risk.shape:
                raise RuntimeError(
                    "V491 predicted risk/log-variance shapes must match: "
                    f"{tuple(predicted_risk.shape)} vs "
                    f"{tuple(predicted_log_variance.shape)}"
                )
            proposal_log_variance = predicted_log_variance[:, 1:2].clamp(
                -6.0, 4.0
            )
            proposal_residual_sq = (
                predicted_risk[:, 1:2] - teacher_risk[:, 1:2]
            ).pow(2)
            # Non-negative heteroscedastic calibration.  ``softplus`` keeps
            # the variance regularizer stable when log-variance is negative,
            # avoiding a large negative auxiliary objective during scratch
            # initialization while retaining the residual-dependent precision.
            gaussian_nll = 0.5 * (
                torch.exp(-proposal_log_variance) * proposal_residual_sq
                + F.softplus(proposal_log_variance)
            )
            uncertainty_nll_loss = _v489_masked_mean(
                gaussian_nll, edit_relevance
            )

        risk_terms = _v490_dense_risk_losses(
            predicted_risk=predicted_risk,
            teacher_risk=teacher_risk,
            relevance=edit_relevance,
            teacher_temperature=float(
                _m1(cfg, "V490_M3_TEACHER_TEMPERATURE", 0.10)
            ),
            student_temperature=float(
                _m1(cfg, "V490_M3_STUDENT_TEMPERATURE", 0.50)
            ),
            scale_invariant=(scale_invariant_risk_enabled and not v492_enabled),
            risk_scale_floor=float(
                _m1(cfg, "V490_M3_RISK_SCALE_FLOOR", 1.0e-3)
            ),
        )
        risk_regression_loss = risk_terms["risk_regression"]
        m3_risk_mae = risk_terms["risk_mae"]
        expert_ce_loss = risk_terms["list_loss"]
        rank_loss = risk_terms["rank_loss"]
        m3_pairwise_accuracy = risk_terms["pairwise_accuracy"]
        m3_selection_accuracy = risk_terms["selection_accuracy"]
        m3_oracle_regret = risk_terms["oracle_regret"]
        m3_selection_entropy = risk_terms["selection_entropy"]
        m3_spatial_switch_rate = risk_terms["spatial_switch_rate"]
        m3_oracle_hist = risk_terms["oracle_hist"]
        m3_predicted_hist = risk_terms["predicted_hist"]
        m3_teacher_entropy = risk_terms["teacher_entropy"]
        m3_teacher_risk_scale = risk_terms["teacher_risk_scale"]
        m3_predicted_risk_scale = risk_terms["predicted_risk_scale"]

        if v495_enabled:
            # Deployment accepts the proposal iff predicted relative risk < 0.
            # Existing regression/list/rank terms improve ordering but do not
            # directly force the prediction across that exact zero boundary.
            # This balanced sign objective uses the same normalized teacher risk
            # already used by M3, ignores ambiguous near-zero targets, and is
            # restricted to the selected proposal's actual edit relevance.
            sign_terms = _v495_balanced_risk_sign_loss(
                predicted_proposal_risk=predicted_risk[:, 1:2],
                teacher_proposal_risk=teacher_risk_for_sign[:, 1:2],
                relevance=edit_relevance,
                margin=float(
                    _m1(
                        cfg,
                        "V495_M3_SIGN_MARGIN",
                        _m1(cfg, "V489_M3_TARGET_MARGIN", 0.002),
                    )
                ),
                temperature=float(
                    _m1(cfg, "V490_M3_STUDENT_TEMPERATURE", 0.50)
                ),
            )
            v495_m3_sign_bce_loss = sign_terms["loss"]
            v495_m3_sign_accuracy = sign_terms["accuracy"]
            v495_m3_sign_positive_rate = sign_terms["positive_rate"]
            v495_m3_sign_supervision_fraction = sign_terms[
                "supervision_fraction"
            ]

        # Spatial consistency is imposed on the soft expert distribution only
        # where experts actually differ.  It replaces the old hard component
        # grouping without forcing all pixels of a large region to one expert.
        m3_soft = aux["m3_pixel_weights"]
        coherence_num = candidate_probs.new_zeros(())
        coherence_den = candidate_probs.new_zeros(())
        if w > 1:
            weight_x = torch.minimum(
                edit_relevance[..., :, :-1], edit_relevance[..., :, 1:]
            )
            coherence_num = coherence_num + (
                (m3_soft[..., :, 1:] - m3_soft[..., :, :-1]).abs()
                * weight_x
            ).sum()
            coherence_den = coherence_den + (
                weight_x.sum() * float(e)
            )
        if h > 1:
            weight_y = torch.minimum(
                edit_relevance[..., :-1, :], edit_relevance[..., 1:, :]
            )
            coherence_num = coherence_num + (
                (m3_soft[..., 1:, :] - m3_soft[..., :-1, :]).abs()
                * weight_y
            ).sum()
            coherence_den = coherence_den + (
                weight_y.sum() * float(e)
            )
        coherence_loss = coherence_num / coherence_den.clamp_min(1.0)
    else:
        region_logits = aux["m3_region_logits"]
        region_size = (region_logits.shape[-2], region_logits.shape[-1])
        seg_error = (expert_probs - gt_e).abs()
        base_seg_error = seg_error[:, :1]
        base_boundary_error = boundary_error[:, :1]
        region_seg_gain = _v489_region_pool(
            base_seg_error - seg_error, region_size
        )
        region_boundary_gain = _v489_region_pool(
            base_boundary_error - boundary_error, region_size
        )
        region_edit = _v489_region_pool(edit_magnitude, region_size)
        utility = (
            region_seg_gain
            + float(_m1(cfg, "V489_M3_BOUNDARY_UTILITY_WEIGHT", 0.50))
            * region_boundary_gain
            - float(_m1(cfg, "V489_M3_EDIT_UTILITY_PENALTY", 0.02))
            * region_edit
        ).detach()
        utility[:, 0:1] = 0.0
        target_margin = float(_m1(cfg, "V489_M3_TARGET_MARGIN", 0.002))
        utility_for_target = utility.clone()
        if e > 1:
            utility_for_target[:, 1:] -= target_margin
        expert_target = utility_for_target.argmax(dim=1)
        expert_ce_loss = F.cross_entropy(region_logits, expert_target)
        best_pred = region_logits.gather(1, expert_target[:, None])
        best_utility = utility.gather(1, expert_target[:, None])
        utility_gap = (best_utility - utility).clamp_min(0.0)
        pair_valid = (utility_gap > target_margin).to(region_logits.dtype)
        pair_valid.scatter_(1, expert_target[:, None], 0.0)
        rank_margin = float(_m1(cfg, "V489_M3_RANK_MARGIN", 0.10))
        rank_element = F.relu(rank_margin - best_pred + region_logits)
        rank_loss = (
            (rank_element * pair_valid * utility_gap).sum()
            / (pair_valid * utility_gap).sum().clamp_min(1.0)
        )

    final_seg_loss = _prob_bce_dice(final_probs, gt)
    final_boundary_loss = _boundary_l1(
        final_probs, gt, radius=boundary_radius
    )
    v501_m3_boundary_no_harm_loss = candidate_probs.sum() * 0.0
    if v501_enabled:
        v501_boundary_radius = int(_m1(cfg, "V501_BOUNDARY_RADIUS", 2))
        base_boundary_error_map = (
            _soft_boundary(c0.detach(), radius=v501_boundary_radius)
            - _soft_boundary(gt, radius=v501_boundary_radius)
        ).abs()
        final_boundary_error_map = (
            _soft_boundary(final_probs, radius=v501_boundary_radius)
            - _soft_boundary(gt, radius=v501_boundary_radius)
        ).abs()
        v501_m3_boundary_no_harm_loss = _v489_masked_mean(
            F.relu(final_boundary_error_map - base_boundary_error_map),
            _as_b1hw(aux["m3_edit_relevance"]).detach().clamp(0.0, 1.0),
        )
    base_online_error = (c0 - gt).abs()
    final_error = (final_probs - gt).abs()
    if v491_enabled:
        final_no_harm_loss = _v489_masked_mean(
            F.relu(final_error - base_online_error),
            _as_b1hw(aux["m3_edit_relevance"]).detach().clamp(0.0, 1.0),
        )
    else:
        final_no_harm_loss = F.relu(
            final_error - base_online_error
        ).mean()
    if v490_enabled:
        m3_loss = (
            float(_m1(cfg, "V490_M3_RISK_REGRESSION_WEIGHT", 1.0))
            * risk_regression_loss
            + float(_m1(cfg, "V490_M3_LIST_WEIGHT", 1.0))
            * expert_ce_loss
            + float(_m1(cfg, "V490_M3_RANK_WEIGHT", 0.50))
            * rank_loss
            + float(_m1(cfg, "V490_M3_FINAL_SEG_WEIGHT", 1.0))
            * final_seg_loss
            + float(_m1(cfg, "V490_M3_FINAL_BOUNDARY_WEIGHT", 0.50))
            * final_boundary_loss
            + float(_m1(cfg, "V490_M3_COHERENCE_WEIGHT", 0.0))
            * coherence_loss
            + float(_m1(cfg, "V491_M3_UNCERTAINTY_NLL_WEIGHT", 0.0))
            * uncertainty_nll_loss
            + float(_m1(cfg, "V490_M3_NO_HARM_WEIGHT", 0.0))
            * final_no_harm_loss
            + float(_m1(cfg, "V501_M3_BOUNDARY_NO_HARM_WEIGHT", 0.75))
            * v501_m3_boundary_no_harm_loss
            + (
                float(_m1(cfg, "V495_M3_SIGN_BCE_WEIGHT", 1.0))
                * v495_m3_sign_bce_loss
                if v495_enabled else candidate_probs.sum() * 0.0
            )
        )
    else:
        m3_loss = (
            float(_m1(cfg, "V489_M3_EXPERT_CE_WEIGHT", 1.0))
            * expert_ce_loss
            + float(_m1(cfg, "V489_M3_RANK_WEIGHT", 0.50))
            * rank_loss
            + float(_m1(cfg, "V489_M3_FINAL_SEG_WEIGHT", 1.0))
            * final_seg_loss
            + float(_m1(cfg, "V489_M3_FINAL_BOUNDARY_WEIGHT", 0.50))
            * final_boundary_loss
            + float(_m1(cfg, "V489_M3_NO_HARM_WEIGHT", 0.25))
            * final_no_harm_loss
        )

    v503_m3_diag: Dict[str, torch.Tensor] = {}
    v504_m3_diag: Dict[str, torch.Tensor] = {}
    if v504_enabled:
        m3_loss, v504_m3_diag = _compute_v504_m3_loss(cfg, gt, c0, aux)
    elif v503_enabled:
        m3_loss, v503_m3_diag = _compute_v503_m3_loss(cfg, gt, c0, aux)

    if v501_enabled:
        m1_scale = _v501_delayed_ramp(
            epoch,
            int(_m1(cfg, "V501_M1_START_EPOCH", 0)),
            int(_m1(cfg, "V501_M1_RAMP_EPOCHS", 20)),
            float(_m1(cfg, "V501_M1_FINAL_WEIGHT", 1.0)),
        )
        if v502_enabled:
            m2_scale = _v501_delayed_ramp(
                epoch,
                int(_m1(cfg, "V502_M2_START_EPOCH", 20)),
                int(_m1(cfg, "V502_M2_RAMP_EPOCHS", 20)),
                float(_m1(cfg, "V502_M2_FINAL_WEIGHT", 1.0)),
            )
            # M3 is intentionally delayed until the soft M2 has received a
            # substantial teacher-only training window.  It still sees the same
            # proposal during forward, but its objective is inactive here.
            m3_scale = _v501_delayed_ramp(
                epoch,
                int(_m1(cfg, "V502_M3_START_EPOCH", 70)),
                int(_m1(cfg, "V502_M3_RAMP_EPOCHS", 20)),
                float(_m1(cfg, "V502_M3_FINAL_WEIGHT", 0.5)),
            )
        else:
            m2_scale = _v501_delayed_ramp(
                epoch,
                int(_m1(cfg, "V501_M2_START_EPOCH", 10)),
                int(_m1(cfg, "V501_M2_RAMP_EPOCHS", 20)),
                float(_m1(cfg, "V501_M2_FINAL_WEIGHT", 1.0)),
            )
            m3_scale = _v501_delayed_ramp(
                epoch,
                int(_m1(cfg, "V501_M3_START_EPOCH", 25)),
                int(_m1(cfg, "V501_M3_RAMP_EPOCHS", 20)),
                float(_m1(cfg, "V501_M3_FINAL_WEIGHT", 0.5)),
            )
    else:
        m1_scale = 1.0
        m2_scale = _v489_loss_ramp(
            epoch,
            float(_m1(cfg, "V489_M2_LOSS_START", 0.25)),
            float(_m1(cfg, "V489_M2_LOSS_FINAL", 1.00)),
            int(_m1(cfg, "V489_M2_LOSS_RAMP_EPOCHS", 10)),
        )
        m3_scale = _v489_loss_ramp(
            epoch,
            float(_m1(cfg, "V489_M3_LOSS_START", 0.05)),
            float(_m1(cfg, "V489_M3_LOSS_FINAL", 0.50)),
            int(_m1(cfg, "V489_M3_LOSS_RAMP_EPOCHS", 15)),
        )
    total = m1_scale * m1_loss + m2_scale * m2_loss + m3_scale * m3_loss

    # --------------------------------------------------------------- diagnostics.
    with torch.no_grad():
        base_dice = _soft_dice_probs((c0 >= 0.5).float(), gt)[:, 0]
        candidate_hard = (candidate_probs >= 0.5).float()
        slot_dice = _soft_dice_probs(candidate_hard, gt)
        global_oracle = slot_dice.max(dim=1).values
        pwo_dice = _soft_dice_probs((pwo_target >= 0.5).float(), gt)[:, 0]
        convex_dice = _soft_dice_probs((m2_convex >= 0.5).float(), gt)[:, 0]
        m2_dice = _soft_dice_probs((m2_fused >= 0.5).float(), gt)[:, 0]
        final_dice = _soft_dice_probs((final_probs >= 0.5).float(), gt)[:, 0]

        gate_diag_threshold = 0.50 if v492_enabled else (0.10 if v490_enabled else 0.50)
        gate_source = v492_benefit_prob if v492_enabled else gate_prob
        gate_pred = gate_source >= gate_diag_threshold
        gate_truth = (
            (v493_selected_gain_target > EPS)
            if v493_enabled else
            ((best_achievable_gain > EPS) if v492_enabled else (gate_target >= gate_diag_threshold))
        )
        diag_region = (
            supervision_weight > 0.0
            if v490_enabled else decision_region > 0.5
        )
        tp = (gate_pred & gate_truth & diag_region).float().sum()
        fp = (gate_pred & (~gate_truth) & diag_region).float().sum()
        fn = ((~gate_pred) & gate_truth & diag_region).float().sum()
        gate_precision = tp / (tp + fp).clamp_min(1.0)
        gate_recall = tp / (tp + fn).clamp_min(1.0)

        route_probability = F.softmax(route_logits, dim=1)
        target_top2 = route_target.topk(k=min(2, route_target.shape[1]), dim=1)
        pred_top2 = route_probability.topk(k=min(2, route_probability.shape[1]), dim=1)
        if route_target.shape[1] > 1:
            target_gap = target_top2.values[:, 0] - target_top2.values[:, 1]
            pred_gap = pred_top2.values[:, 0] - pred_top2.values[:, 1]
        else:
            target_gap = target_top2.values[:, 0]
            pred_gap = pred_top2.values[:, 0]
        route_target_unique = (target_gap > 0.05) & (gate_target[:, 0] > 0.0)
        route_pred_confident = pred_gap > 0.01
        if bool(route_target_unique.any().item()):
            route_pred = route_logits.argmax(dim=1)
            route_true = route_target.argmax(dim=1)
            route_accuracy = (
                (route_pred == route_true)
                & route_pred_confident
            )[route_target_unique].float().mean()
            route_pred_confident_fraction = route_pred_confident[
                route_target_unique
            ].float().mean()
        else:
            route_accuracy = candidate_probs.new_tensor(0.0)
            route_pred_confident_fraction = candidate_probs.new_tensor(0.0)
        route_unique_fraction = route_target_unique.float().mean()
        route_target_entropy = -(
            route_target.clamp_min(EPS) * route_target.clamp_min(EPS).log()
        ).sum(dim=1)
        route_target_entropy = _v489_masked_mean(
            route_target_entropy[:, None], gate_target
        )

        c0_binary = c0 >= 0.5
        m2_binary = m2_fused >= 0.5
        harmful_edit = ((m2_binary != c0_binary) & (c0_binary == (gt >= 0.5))).float().mean()
        selection_rate = aux.get("m3_expert_selection_rate")
        if not isinstance(selection_rate, torch.Tensor):
            selection_rate = candidate_probs.new_zeros((b, e))

        if v494_enabled:
            accept_probability = _as_b1hw(aux["m3_accept_probability"])
            hard_gate = _as_b1hw(aux["m3_hard_gate"]) >= 0.5
            relevance = _as_b1hw(aux["m3_edit_relevance"]).detach() > 0.0
            real_edit = (expert_probs[:, 1:2] - c0).abs() > EPS
            valid = relevance & real_edit
            probability_choice = accept_probability >= 0.5
            valid_mass = valid.float().sum().clamp_min(1.0)
            v494_m3_soft_hard_agreement = (
                (probability_choice == hard_gate).float() * valid.float()
            ).sum() / valid_mass
            ucb_accept = _as_b1hw(aux["m3_upper_confidence_risk"]) < 0.0
            expected_accept = _as_b1hw(aux["m3_deployment_risk"]) < 0.0
            v494_m3_ucb_rejection_gap = (
                (expected_accept & (~ucb_accept) & valid).float().sum()
                / valid_mass
            )

    if v495_enabled:
        actual_rel = _as_b1hw(
            aux.get("m3_actual_edit_relevance", aux["m3_edit_relevance"])
        ).mean()
        potential_rel = _as_b1hw(
            aux.get("m3_potential_edit_relevance", aux["m3_edit_relevance"])
        ).mean()
        v495_actual_to_potential_relevance = (
            actual_rel / potential_rel.clamp_min(EPS)
        ).detach()
        proposal = _as_b1hw(aux["m2_proposal_probs"])
        proposal_delta = (proposal - c0).abs()
        proposal_scale = proposal_delta.flatten(1).amax(dim=1).view(
            -1, 1, 1, 1
        )
        proposal_relevance = (
            proposal_delta / proposal_scale.clamp_min(EPS)
        ).detach().clamp(0.0, 1.0)
        v495_proposal_visibility_ratio = (
            actual_rel / proposal_relevance.mean().clamp_min(EPS)
        ).detach()
        selected_support = _as_b1hw(aux["m2_structural_support"]).detach()
        m3_selected_support = _as_b1hw(
            aux.get("m3_selected_support", aux["m3_support_union"])
        ).detach()
        support_intersection = torch.minimum(
            selected_support, m3_selected_support
        ).sum()
        support_union_mass = torch.maximum(
            selected_support, m3_selected_support
        ).sum()
        v495_selected_support_alignment = (
            (support_intersection + EPS) / (support_union_mass + EPS)
        ).detach()

    diag: Dict[str, torch.Tensor] = {
        "v489_total_loss": total.detach(),
        "v489_m1_loss": m1_loss.detach(),
        "v501_m1_loss_scale": candidate_probs.new_tensor(m1_scale),
        "v489_error_state_loss": error_state_loss.detach(),
        "v489_support_loss": support_supervision_loss.detach(),
        "v489_candidate_repair_loss": candidate_repair_loss.detach(),
        "v489_outside_preserve_loss": outside_preserve_loss.detach(),
        "v489_coverage_loss": coverage_loss.detach(),
        "v489_decorrelation_loss": decorrelation_loss.detach(),
        "v489_m2_loss": m2_loss.detach(),
        "v489_m2_loss_scale": candidate_probs.new_tensor(m2_scale),
        "v489_gate_loss": gate_loss.detach(),
        "v489_gate_smooth_l1": gate_smooth_l1.detach(),
        "v489_route_loss": route_loss.detach(),
        "v489_correction_loss": correction_loss.detach(),
        "v489_harm_loss": harm_loss.detach(),
        "v489_pwo_loss": pwo_loss.detach(),
        "v490_m2_soft_oracle_loss": (
            pwo_loss.detach() if v490_enabled else candidate_probs.new_zeros(())
        ),
        "v489_m2_seg_loss": m2_seg_loss.detach(),
        "v489_m2_boundary_loss": m2_boundary_loss.detach(),
        "v489_residual_l1": residual_l1.detach(),
        "v489_m3_loss": m3_loss.detach(),
        "v489_m3_loss_scale": candidate_probs.new_tensor(m3_scale),
        "v489_m3_expert_ce_loss": expert_ce_loss.detach(),
        "v489_m3_rank_loss": rank_loss.detach(),
        "v490_m3_risk_regression_loss": risk_regression_loss.detach(),
        "v490_m3_list_loss": expert_ce_loss.detach(),
        "v490_m3_rank_loss": rank_loss.detach(),
        "v490_m3_selection_accuracy": m3_selection_accuracy.detach(),
        "v490_m3_oracle_regret": m3_oracle_regret.detach(),
        "v490_m3_edit_relevance_mean": m3_edit_relevance_mean.detach(),
        "v490_m3_risk_mae": m3_risk_mae.detach(),
        "v490_m3_pairwise_accuracy": m3_pairwise_accuracy.detach(),
        "v490_m3_selection_entropy": m3_selection_entropy.detach(),
        "v490_m3_teacher_entropy": m3_teacher_entropy.detach(),
        "v490_m3_teacher_risk_scale": m3_teacher_risk_scale.detach(),
        "v490_m3_predicted_risk_scale": m3_predicted_risk_scale.detach(),
        "v490_m3_spatial_switch_rate": m3_spatial_switch_rate.detach(),
        "v490_m3_coherence_loss": coherence_loss.detach(),
        "v491_m3_uncertainty_nll_loss": uncertainty_nll_loss.detach(),
        "v491_preserve_first_enabled": candidate_probs.new_tensor(
            float(v491_enabled)
        ),
        "v492_causal_local_editor_enabled": candidate_probs.new_tensor(
            float(v492_enabled)
        ),
        "v493_candidate_conditioned_causal_enabled": candidate_probs.new_tensor(
            float(v493_enabled)
        ),
        "v494_direct_causal_dose_enabled": candidate_probs.new_tensor(
            float(v494_enabled)
        ),
        "v495_single_decision_enabled": candidate_probs.new_tensor(
            float(v495_enabled)
        ),
        "v498_consistent_full_proposal_enabled": candidate_probs.new_tensor(
            float(v498_enabled)
        ),
        "v498_m2_action_loss": v498_action_loss.detach(),
        "v498_m2_action_accuracy": v498_action_accuracy.detach(),
        "v498_m2_preserve_target_rate": v498_preserve_target_rate.detach(),
        "v498_m2_preserve_reference_prob": (
            v498_preserve_reference_prob.detach()
        ),
        "v498_m2_full_proposal_no_harm_loss": (
            v498_full_proposal_no_harm_loss.detach()
        ),
        "v498_m2_route_expected_harm_loss": (
            v498_route_expected_harm_loss.detach()
        ),
        "v498_m2_full_proposal_harm_fraction": (
            v498_full_proposal_harm_fraction.detach()
        ),
        "v498_m2_train_deploy_l1": v498_train_deploy_l1.detach(),
        "v498_m2_train_deploy_alignment": (
            v498_train_deploy_alignment.detach()
        ),
        "v499_causal_preserve_reference_enabled": candidate_probs.new_tensor(
            float(v499_enabled)
        ),
        "v499_m2_causal_region_rate": v499_causal_region_rate.detach(),
        "v499_m2_candidate_target_rate": v499_candidate_target_rate.detach(),
        "v499_m2_eligible_candidate_rate": (
            v499_eligible_candidate_rate.detach()
        ),
        "v499_m2_target_outside_causal_rate": (
            v499_target_outside_causal_rate.detach()
        ),
        "v499_m2_preserve_outside_causal_accuracy": (
            v499_preserve_outside_causal_accuracy.detach()
        ),
        "v499_m2_candidate_action_accuracy": (
            v499_candidate_action_accuracy.detach()
        ),
        "v499_m2_predicted_candidate_rate": (
            v499_predicted_candidate_rate.detach()
        ),
        "v500_hierarchical_safe_route_enabled": candidate_probs.new_tensor(
            float(v500_enabled)
        ),
        "v500_m2_candidate_presence_loss": (
            v500_candidate_presence_loss.detach()
        ),
        "v500_m2_candidate_presence_prob": (
            v500_candidate_presence_prob.detach()
        ),
        "v500_m2_candidate_rate_gap": v500_candidate_rate_gap.detach(),
        "v500_m2_safe_route_loss": v500_safe_route_loss.detach(),
        "v500_m2_route_min_regret_loss": (
            v500_route_min_regret_loss.detach()
        ),
        "v500_m2_safe_route_accuracy": v500_safe_route_accuracy.detach(),
        "v500_m2_meaningful_edit_rate": v500_meaningful_edit_rate.detach(),
        "v501_base_anchored_selective_repair_enabled": candidate_probs.new_tensor(
            float(v501_enabled)
        ),
        "v501_m1_boundary_loss": v501_m1_boundary_loss.detach(),
        "v501_global_rediscovery_loss": v501_global_rediscovery_loss.detach(),
        "v501_global_preserve_loss": v501_global_preserve_loss.detach(),
        "v501_failure_case_rate": v501_failure_case_rate.detach(),
        "v501_m2_gate_target_mean": v501_gate_target.mean().detach(),
        "v501_m2_gain_eligible_rate": (
            v501_gain_is_meaningful.float().mean().detach()
        ),
        "v501_m2_gate_regression_loss": v501_gate_regression_loss.detach(),
        "v501_m2_false_edit_loss": v501_false_edit_loss.detach(),
        "v501_m2_gate_precision": v501_gate_precision.detach(),
        "v501_m2_gate_recall": v501_gate_recall.detach(),
        "v501_m2_gate_f1": v501_gate_f1.detach(),
        "v501_m2_gate_false_positive_rate": (
            v501_gate_false_positive_rate.detach()
        ),
        "v501_m2_stable_boundary_penalty": (
            v501_stable_boundary_penalty.detach()
        ),
        "v501_m3_boundary_no_harm_loss": (
            v501_m3_boundary_no_harm_loss.detach()
        ),
        "v502_hierarchical_utility_soft_router_enabled": candidate_probs.new_tensor(
            float(v502_enabled)
        ),
        **v502_diag,
        "v495_m2_proposal_edit_rate": v495_proposal_edit_rate.detach(),
        "v495_m2_training_edit_rate": v495_training_edit_rate.detach(),
        "v495_selected_to_all_candidate_relevance": (
            v495_actual_to_potential_relevance.detach()
        ),
        "v495_m3_proposal_visibility_ratio": (
            v495_proposal_visibility_ratio.detach()
        ),
        "v495_m3_selected_support_alignment": (
            v495_selected_support_alignment.detach()
        ),
        "v495_m3_sign_bce_loss": v495_m3_sign_bce_loss.detach(),
        "v495_m3_sign_accuracy": v495_m3_sign_accuracy.detach(),
        "v495_m3_sign_positive_rate": (
            v495_m3_sign_positive_rate.detach()
        ),
        "v495_m3_sign_supervision_fraction": (
            v495_m3_sign_supervision_fraction.detach()
        ),
        "v492_m2_benefit_loss": v492_benefit_loss.detach(),
        "v492_m2_benefit_rank_loss": v492_benefit_rank_loss.detach(),
        "v492_m2_amplitude_loss": v492_amplitude_loss.detach(),
        "v492_m2_utility_loss": v492_utility_loss.detach(),
        "v492_m2_preserve_loss": v492_preserve_loss.detach(),
        "v492_m2_no_harm_loss": v492_no_harm_loss.detach(),
        "v492_m2_local_seg_loss": v492_local_seg_loss.detach(),
        "v492_m2_local_boundary_loss": v492_local_boundary_loss.detach(),
        "v492_m2_effective_oracle_loss": v492_effective_oracle_loss.detach(),
        "v492_m2_benefit_mean": v492_benefit_prob.mean().detach(),
        "v492_m2_benefit_positive_mean": _v489_masked_mean(
            v492_benefit_prob, positive_relevance
        ).detach(),
        "v492_m2_benefit_negative_mean": _v489_masked_mean(
            v492_benefit_prob, negative_relevance
        ).detach(),
        "v492_m2_amplitude_mean": v492_amplitude_prob.mean().detach(),
        "v492_m2_amplitude_target_mean": _v489_masked_mean(
            gate_target, positive_relevance
        ).detach(),
        "v492_m2_amplitude_mae": _v489_masked_mean(
            (v492_amplitude_prob - gate_target).abs(), positive_relevance
        ).detach(),
        "v492_m2_structural_support_mean": v492_structural_support.mean().detach(),
        "v493_m2_candidate_utility_loss": v493_candidate_utility_loss.detach(),
        "v493_m2_candidate_utility_reg_loss": v493_candidate_utility_reg_loss.detach(),
        "v493_m2_selected_benefit_loss": v493_selected_benefit_loss.detach(),
        "v493_m2_selected_benefit_rank_loss": v493_selected_benefit_rank_loss.detach(),
        "v493_m2_selected_preserve_loss": v493_selected_preserve_loss.detach(),
        "v493_m2_benefit_conflict_rate": v493_benefit_conflict_rate.detach(),
        "v493_m2_old_label_overlap_rate": v493_old_label_overlap_rate.detach(),
        "v493_m2_selected_route_safe_rate": v493_selected_route_safe_rate.detach(),
        "v493_m2_selected_route_gain": v493_selected_route_gain.detach(),
        "v493_m2_candidate_utility_accuracy": v493_candidate_utility_accuracy.detach(),
        "v493_m2_candidate_utility_positive_mean": v493_candidate_utility_positive_mean.detach(),
        "v493_m2_candidate_utility_negative_mean": v493_candidate_utility_negative_mean.detach(),
        "v493_m2_selected_benefit_positive_mean": _v489_masked_mean(
            v492_benefit_prob, v493_selected_positive_weight
        ).detach(),
        "v493_m2_selected_benefit_negative_mean": _v489_masked_mean(
            v492_benefit_prob, v493_selected_negative_weight
        ).detach(),
        "v493_m2_selected_amplitude_target_mean": _v489_masked_mean(
            v493_selected_alpha_target, v493_selected_positive_weight
        ).detach(),
        "v493_m2_selected_amplitude_mae": _v489_masked_mean(
            (v492_amplitude_prob - v493_selected_alpha_target).abs(),
            v493_selected_positive_weight,
        ).detach(),
        "v494_m2_alpha_full_dose_rate": v494_alpha_full_dose_rate.detach(),
        "v494_m2_direct_accept_positive_mean": v494_direct_accept_positive_mean.detach(),
        "v494_m2_direct_accept_negative_mean": v494_direct_accept_negative_mean.detach(),
        "v494_m3_soft_hard_agreement": v494_m3_soft_hard_agreement.detach(),
        "v494_m3_ucb_rejection_gap": v494_m3_ucb_rejection_gap.detach(),
        "v492_m3_teacher_raw_scale": v492_teacher_raw_scale.detach(),
        "v493_m3_actual_edit_relevance_mean": _as_b1hw(
            aux.get("m3_actual_edit_relevance", aux["m3_edit_relevance"])
        ).mean().detach(),
        "v493_m3_potential_edit_relevance_mean": _as_b1hw(
            aux.get("m3_potential_edit_relevance", aux["m3_edit_relevance"])
        ).mean().detach(),
        # Compatibility fields are fixed at zero because V490.2 has no
        # connected components and cannot discard intervention regions.
        "v490_m3_component_count": candidate_probs.new_zeros(()),
        "v490_m3_raw_component_count": candidate_probs.new_zeros(()),
        "v490_m3_intervention_pixel_rate": m3_edit_relevance_mean.detach(),
        "v490_m3_dropped_intervention_fraction": candidate_probs.new_zeros(()),
        "v489_final_seg_loss": final_seg_loss.detach(),
        "v489_final_boundary_loss": final_boundary_loss.detach(),
        "v489_final_no_harm_loss": final_no_harm_loss.detach(),
        "v489_base_dice": base_dice.mean().detach(),
        "v489_global_oracle_dice": global_oracle.mean().detach(),
        "v489_pwo_dice": pwo_dice.mean().detach(),
        "v489_m2_convex_dice": convex_dice.mean().detach(),
        "v489_m2_dice": m2_dice.mean().detach(),
        "v489_final_dice": final_dice.mean().detach(),
        "v489_m2_gain_vs_base": (m2_dice - base_dice).mean().detach(),
        "v489_final_gain_vs_m2": (final_dice - m2_dice).mean().detach(),
        "v489_pwo_gap_to_m2_convex": (pwo_dice - convex_dice).mean().detach(),
        "v489_decision_pixel_rate": decision_region.mean().detach(),
        "v489_correctable_pixel_rate": (
            positive_relevance.mean().detach()
            if v490_enabled else gate_target.mean().detach()
        ),
        "v490_optimal_gate_target_mean": optimal_gate_target.mean().detach(),
        "v490_achievable_gain_relevance": optimal_gain_relevance.mean().detach(),
        "v490_supervision_mass": supervision_weight.mean().detach(),
        "v490_negative_relevance": negative_relevance.mean().detach(),
        "v490_gate_positive_mean": _v489_masked_mean(
            v492_benefit_prob if v492_enabled else gate_prob,
            positive_relevance if v492_enabled else gate_target,
        ).detach(),
        "v490_gate_negative_mean": _v489_masked_mean(
            v492_benefit_prob if v492_enabled else gate_prob, negative_relevance
        ).detach(),
        "v489_edit_gate_rate": gate_prob.mean().detach(),
        "v489_edit_gate_precision": gate_precision.detach(),
        "v489_edit_gate_recall": gate_recall.detach(),
        "v489_route_top1_accuracy": route_accuracy.detach(),
        "v490_route_unique_target_fraction": route_unique_fraction.detach(),
        "v490_route_pred_confident_fraction": route_pred_confident_fraction.detach(),
        "v490_route_target_entropy": route_target_entropy.detach(),
        "v489_m2_hard_edit_rate": (m2_binary != c0_binary).float().mean().detach(),
        "v489_m2_harmful_edit_rate": harmful_edit.detach(),
        "v489_residual_active_rate": (residual_map.abs() > 1.0e-4).float().mean().detach(),
        "v489_m3_nonbase_rate": (1.0 - selection_rate[:, :1]).mean().detach(),
        "v489_m3_m2_selection_rate": selection_rate[:, -1].mean().detach(),
        "v489_m3_temperature": aux.get("m3_temperature", candidate_probs.new_ones((b,))).mean().detach(),
        "v489_m3_m2_margin": aux.get("m3_m2_margin", candidate_probs.new_zeros((b,))).mean().detach(),
        # Compatibility aliases for existing log/validation readers.
        "v488_total_loss": total.detach(),
        "v488_base_dice": base_dice.mean().detach(),
        "v488_global_oracle_dice": global_oracle.mean().detach(),
        "v488_pwo_dice": pwo_dice.mean().detach(),
        "v488_m2_dice": m2_dice.mean().detach(),
        "v488_m3_final_dice": final_dice.mean().detach(),
        "v488_m2_gain_vs_base": (m2_dice - base_dice).mean().detach(),
        "v488_m3_gain_vs_m2": (final_dice - m2_dice).mean().detach(),
        "v488_nonbase_weight": gate_prob.mean().detach(),
        "v488_m3_gate_rate": (1.0 - selection_rate[:, :1]).mean().detach(),
        "v488_correctable_pixel_rate": (
            positive_relevance.mean().detach()
            if v490_enabled else gate_target.mean().detach()
        ),
        "v485_m2_local_loss": m2_loss.detach(),
        "v484_m2_local_loss": m2_loss.detach(),
        "v484_m3_regret_loss": m3_loss.detach(),
    }
    diag.update(v502_diag)
    diag.update(v503_diag)
    diag.update(v503_m3_diag)
    diag.update(v504_diag)
    diag.update(v504_m3_diag)
    diag["v503_factual_atomic_causal_enabled"] = candidate_probs.new_tensor(float(v503_enabled))
    diag["v504_realizable_potential_outcome_enabled"] = candidate_probs.new_tensor(float(v504_enabled))
    diag["v504_m1_executability_loss"] = v504_executability_loss.detach()
    if v504_executable_rates:
        for index, value in enumerate(v504_executable_rates):
            diag[f"v504_m1_action_{index + 1}_executable_rate"] = value.detach()
        diag["v504_m1_mean_executable_rate"] = torch.stack(v504_executable_rates).mean().detach()
    for index in range(selection_rate.shape[1]):
        diag[f"v489_m3_expert_{index}_selection_rate"] = selection_rate[:, index].mean().detach()
    if v490_enabled:
        for index in range(e):
            diag[f"v490_m3_oracle_expert_{index}_rate"] = m3_oracle_hist[index].detach()
            diag[f"v490_m3_predicted_expert_{index}_rate"] = m3_predicted_hist[index].detach()
        # Live tensors are consumed and removed by train.py.  They split the
        # objective routing so the legacy M1 balance no longer suppresses M2/M3.
        diag["_v490_m1_objective"] = m1_scale * m1_loss
        diag["_v490_m2_objective"] = m2_scale * m2_loss
        diag["_v490_m3_objective"] = m3_scale * m3_loss
    return total, diag

def compute_v484_loss(
    cfg: Any,
    candidates: torch.Tensor,
    masks: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    epoch=None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if (
        str(_m1(cfg, "PROTOCOL", "")).strip().lower() in {"clean_dynamic_component_set", "tc_drcs"}
        or bool(_m1(cfg, "V532_UNIFIED_SPARSE_REFINER_ENABLED", False))
    ):
        return _compute_v532_unified_sparse_refiner_loss(cfg, masks, aux, epoch)
    if bool(_m1(cfg, "V531_TYPED_SPARSE_REFINER_ENABLED", False)):
        return _compute_v531_typed_sparse_refiner_loss(cfg, masks, aux, epoch)
    if bool(_m1(cfg, "V490_ROOT_CAUSE_ENABLED", False)):
        return _compute_v489_end_to_end_loss(cfg, masks, aux, epoch)
    if bool(_m1(cfg, "V489_END_TO_END_ENABLED", False)):
        return _compute_v489_end_to_end_loss(cfg, masks, aux, epoch)
    if bool(_m1(cfg, "V488_M2M3_ONLY", False)):
        return _compute_v488_m2m3_loss(cfg, masks, aux)

    gt = (_as_b1hw(masks) > 0.5).float()
    c0 = aux.get("c0_prob_detached", None)
    if c0 is None:
        c0 = torch.sigmoid(_as_b1hw(candidates[:, :1] if candidates.ndim == 4 else candidates).detach())
    c0 = _as_b1hw(c0).detach().clamp(EPS, 1.0 - EPS)

    cand_probs = aux.get("candidate_probs", None)
    if cand_probs is None:
        cand_probs = torch.sigmoid(candidates).clamp(EPS, 1.0 - EPS)
    if cand_probs.ndim == 3:
        cand_probs = cand_probs[:, None]
    cand_probs = cand_probs.clamp(EPS, 1.0 - EPS)
    if cand_probs.shape[1] < 5:
        pad = cand_probs[:, :1].expand(-1, 5 - cand_probs.shape[1], -1, -1)
        cand_probs = torch.cat([cand_probs, pad], dim=1)

    local_probs = cand_probs[:, 1:5]
    local_supports = aux.get("local_supports")
    if local_supports is None:
        local_supports = (local_probs - c0).abs()
    local_supports = local_supports[:, :4].clamp(0.0, 1.0)

    local_deltas = aux.get("local_deltas")
    if local_deltas is None:
        local_deltas = local_probs - c0
    local_deltas = local_deltas[:, :4]

    local_soft_gates = aux.get("local_soft_gates", None)
    if local_soft_gates is None:
        local_soft_gates = torch.ones(local_probs.shape[:2], device=local_probs.device, dtype=local_probs.dtype)
    local_soft_gates = local_soft_gates[:, :4].clamp(0.0, 1.0)

    labels = ErrorStateHead.make_training_labels(
        c0,
        gt,
        fp_threshold=float(_m1(cfg, "V485_FP_STATE_THRESHOLD", 0.001)),
        fn_threshold=float(_m1(cfg, "V485_FN_STATE_THRESHOLD", 0.001)),
        boundary_threshold=float(_m1(cfg, "V485_BOUNDARY_STATE_THRESHOLD", 0.003)),
        failure_dice=float(_m1(cfg, "V485_FAILURE_DICE", 0.55)),
    )
    error_logits = aux.get("error_state_logits")
    error_loss = F.cross_entropy(error_logits, labels) if error_logits is not None else cand_probs.sum() * 0.0

    c0_hard = (c0 >= 0.5).float()
    fp_target = (c0_hard * (1.0 - gt)).detach()
    fn_target = ((1.0 - c0_hard) * gt).detach()
    tp_region = (c0_hard * gt).detach()
    bg_region = ((1.0 - c0_hard) * (1.0 - gt)).detach()
    c0_boundary = _soft_boundary(c0_hard, radius=1).detach()
    gt_boundary = _soft_boundary(gt, radius=1).detach()
    boundary_target = (c0_boundary - gt_boundary).abs().clamp(0.0, 1.0).detach()
    boundary_band = torch.maximum(c0_boundary, gt_boundary).detach()

    pos_weight = float(_m1(cfg, "V485_SUPPORT_POS_WEIGHT", 6.0))
    delete_support_loss = _support_loss(local_supports[:, 0:1], fp_target, pos_weight)
    fill_support_loss = _support_loss(local_supports[:, 1:2], fn_target, pos_weight)
    trim_support_loss = _support_loss(local_supports[:, 2:3], boundary_target, pos_weight)
    expand_support_loss = _support_loss(local_supports[:, 3:4], boundary_target, pos_weight)
    support_supervision_loss = delete_support_loss + fill_support_loss + 0.5 * (trim_support_loss + expand_support_loss)

    delete_edit = (-local_deltas[:, 0:1]).clamp_min(0.0) * local_supports[:, 0:1]
    fill_edit = local_deltas[:, 1:2].clamp_min(0.0) * local_supports[:, 1:2]
    trim_edit = (-local_deltas[:, 2:3]).clamp_min(0.0) * local_supports[:, 2:3]
    expand_edit = local_deltas[:, 3:4].clamp_min(0.0) * local_supports[:, 3:4]

    fp_removed = _region_mean(delete_edit, fp_target).mean()
    tp_removed = _region_mean(delete_edit, tp_region).mean()
    fn_added = _region_mean(fill_edit, fn_target).mean()
    bg_added = _region_mean(fill_edit, bg_region).mean()
    trim_outside = (trim_edit * (1.0 - boundary_band)).flatten(1).sum(dim=1) / trim_edit.flatten(1).sum(dim=1).clamp_min(EPS)
    expand_outside = (expand_edit * (1.0 - boundary_band)).flatten(1).sum(dim=1) / expand_edit.flatten(1).sum(dim=1).clamp_min(EPS)
    boundary_outside = 0.5 * (trim_outside.mean() + expand_outside.mean())

    direction_loss = (
        F.relu(float(_m1(cfg, "V485_MIN_FP_REMOVED", 0.0005)) - fp_removed)
        + F.relu(float(_m1(cfg, "V485_MIN_FN_ADDED", 0.0005)) - fn_added)
        + float(_m1(cfg, "V485_TP_REMOVAL_WEIGHT", 3.0)) * tp_removed
        + float(_m1(cfg, "V485_BG_LEAKAGE_WEIGHT", 3.0)) * bg_added
        + float(_m1(cfg, "V485_BOUNDARY_OUTSIDE_WEIGHT", 1.0)) * boundary_outside
    )

    # V486 direct candidate-mask repair.  The candidate itself must approach GT
    # in the typed target region while staying close to C0 outside that region.
    repair_pos_weight = float(_m1(cfg, "V486_REPAIR_POS_WEIGHT", 2.0))
    delete_region = _merge_regions(fp_target, boundary_band * float(_m1(cfg, "V486_DELETE_BOUNDARY_MIX", 0.25)))
    fill_region = _merge_regions(fn_target, boundary_band * float(_m1(cfg, "V486_FILL_BOUNDARY_MIX", 0.25)))
    boundary_region = boundary_band
    delete_repair_loss = _masked_bce_dice(local_probs[:, 0:1], gt, delete_region, repair_pos_weight)
    fill_repair_loss = _masked_bce_dice(local_probs[:, 1:2], gt, fill_region, repair_pos_weight)
    trim_repair_loss = _masked_bce_dice(local_probs[:, 2:3], gt, boundary_region, repair_pos_weight)
    expand_repair_loss = _masked_bce_dice(local_probs[:, 3:4], gt, boundary_region, repair_pos_weight)
    candidate_repair_loss = delete_repair_loss + fill_repair_loss + 0.5 * (trim_repair_loss + expand_repair_loss)

    valid_regions = torch.cat([delete_region, fill_region, boundary_region, boundary_region], dim=1)
    outside_preserve_loss = ((local_probs - c0).abs() * (1.0 - valid_regions).clamp(0.0, 1.0)).mean()

    base_soft_dice = _soft_dice_probs(c0, gt)[:, 0]
    local_soft_dice = _soft_dice_probs(local_probs, gt)
    soft_gain = local_soft_dice - base_soft_dice[:, None]
    active_weight = local_soft_gates.clamp(0.0, 1.0)
    gated_gain = soft_gain * active_weight
    best_gain = gated_gain.max(dim=1).values
    gain_margin = float(_m1(cfg, "V486_GAIN_MARGIN", _m1(cfg, "V485_GAIN_MARGIN", 0.002)))
    harm_margin = float(_m1(cfg, "V485_HARM_MARGIN", _m1(cfg, "V484_HARM_MARGIN", 0.002)))
    existence_loss = F.relu(gain_margin - best_gain).mean()
    purity_loss = (F.relu(harm_margin - soft_gain) * active_weight).sum() / active_weight.sum().clamp_min(1.0)

    edit_area = local_supports.flatten(2).mean(dim=-1)
    edit_budget = float(_m1(cfg, "V486_LOCAL_EDIT_BUDGET", _m1(cfg, "V485_LOCAL_EDIT_BUDGET", 0.06)))
    edit_budget_loss = F.relu(edit_area - edit_budget).mean()

    candidate_abs_change = (local_probs - c0).abs()
    change_per_candidate = candidate_abs_change.flatten(2).mean(dim=-1)
    target_area = torch.cat([fp_target, fn_target, boundary_target, boundary_target], dim=1).flatten(2).mean(dim=-1)
    target_exists = (target_area > float(_m1(cfg, "V486_TARGET_EXISTS_AREA", 0.0005))).float()
    min_edit = float(_m1(cfg, "V486_MIN_EDIT_PROB", 0.002))
    edit_floor_loss = (F.relu(min_edit - change_per_candidate) * target_exists * active_weight).sum() / (target_exists * active_weight).sum().clamp_min(1.0)

    # V487_PRECISION_PER_EARLY_BEGIN
    # Compute actual edit-in-target precision before optional M2/support-head
    # supervision.  This is needed by the V487 gated verifier branch below.
    all_edit = torch.cat(
        [delete_edit, fill_edit, trim_edit, expand_edit],
        dim=1,
    )
    target_union = torch.cat(
        [fp_target, fn_target, boundary_target, boundary_target],
        dim=1,
    )
    edit_mass = all_edit.flatten(2).sum(dim=-1)
    inside_mass = (all_edit * target_union).flatten(2).sum(dim=-1)
    precision_per = inside_mass / edit_mass.clamp_min(EPS)
    support_precision = precision_per.mean()
    outside_ratio = (1.0 - precision_per).mean()
    outside_ratio = torch.where(
        edit_mass.mean() <= EPS,
        outside_ratio.new_tensor(1.0),
        outside_ratio,
    )
    # V487_PRECISION_PER_EARLY_END

    m2_loss = cand_probs.sum() * 0.0
    pred_delta = aux.get("local_m2_local_pred_delta_dsc")
    pred_harm_logit = aux.get("local_m2_local_pred_harm_logit")
    if bool(_m1(cfg, "V486_ENABLE_M2_LOSS", False)) and pred_delta is not None and pred_harm_logit is not None:
        active_bool = active_weight > 0.05
        if active_bool.any():
            scale = float(_m1(cfg, "V485_GAIN_SCALE", _m1(cfg, "V484_GAIN_SCALE", 0.03)))
            target_gain = (soft_gain.detach() / scale).clamp(-1.0, 1.0)
            reg = F.smooth_l1_loss(pred_delta[:, :4][active_bool], target_gain[active_bool], beta=0.1)
            harm_target = (soft_gain.detach() < -harm_margin).float()
            harm = F.binary_cross_entropy_with_logits(pred_harm_logit[:, :4][active_bool], harm_target[active_bool])
            # V487: train the deployment gate's support-precision head from the
            # actual edit-in-target precision.  Without this term the gate can
            # only learn gain/harm; it cannot distinguish a clean repair from a
            # spatially diffuse edit that happens to look promising in logits.
            pred_support_logit = aux.get("local_m2_local_pred_support_precision_logit")
            if pred_support_logit is not None:
                true_precision = precision_per.detach().clamp(0.0, 1.0)
                support_reg = F.binary_cross_entropy_with_logits(
                    pred_support_logit[:, :4][active_bool],
                    true_precision[active_bool],
                )
            else:
                support_reg = harm.new_tensor(0.0)
            m2_loss = reg + harm + float(_m1(cfg, "V487_M2_SUPPORT_PRECISION_WEIGHT", 0.5)) * support_reg

    total = (
        float(_m1(cfg, "V485_ERROR_STATE_WEIGHT", 1.0)) * error_loss
        + float(_m1(cfg, "V485_SUPPORT_SUPERVISION_WEIGHT", 1.0)) * support_supervision_loss
        + float(_m1(cfg, "V485_DIRECTION_WEIGHT", 1.0)) * direction_loss
        + float(_m1(cfg, "V486_CANDIDATE_REPAIR_WEIGHT", 2.0)) * candidate_repair_loss
        + float(_m1(cfg, "V486_OUTSIDE_PRESERVE_WEIGHT", 1.0)) * outside_preserve_loss
        + float(_m1(cfg, "V486_EDIT_FLOOR_WEIGHT", 0.5)) * edit_floor_loss
        + float(_m1(cfg, "V486_ORACLE_GAIN_WEIGHT", 1.0)) * existence_loss
        + float(_m1(cfg, "V485_PURITY_WEIGHT", 0.5)) * purity_loss
        + float(_m1(cfg, "V485_EDIT_BUDGET_WEIGHT", 0.25)) * edit_budget_loss
        + float(_m1(cfg, "V485_M2_LOCAL_WEIGHT", 0.0)) * m2_loss
    )

    mean_abs_change = candidate_abs_change.flatten(1).mean()
    max_abs_change = candidate_abs_change.flatten(1).max()
    non_noop_rate = (change_per_candidate > float(_m1(cfg, "V485_NON_NOOP_EPS", 1.0e-5))).float().mean()
    hard_gain = _soft_dice_probs((local_probs >= 0.5).float(), gt) - _soft_dice_probs((c0 >= 0.5).float(), gt)[:, :1]
    hard_oracle_gain = hard_gain.max(dim=1).values.mean()
    positive_rate = (hard_gain > 1.0e-4).float().mean()
    harmful_rate = (hard_gain < -1.0e-4).float().mean()

    all_edit = torch.cat([delete_edit, fill_edit, trim_edit, expand_edit], dim=1)
    target_union = torch.cat([fp_target, fn_target, boundary_target, boundary_target], dim=1)
    edit_mass = all_edit.flatten(2).sum(dim=-1)
    inside_mass = (all_edit * target_union).flatten(2).sum(dim=-1)
    precision_per = inside_mass / edit_mass.clamp_min(EPS)
    support_precision = precision_per.mean()
    outside_ratio = (1.0 - precision_per).mean()
    outside_ratio = torch.where(edit_mass.mean() <= EPS, outside_ratio.new_tensor(1.0), outside_ratio)

    diag: Dict[str, torch.Tensor] = {
        "v485_error_state_loss": error_loss.detach(),
        "v485_support_supervision_loss": support_supervision_loss.detach(),
        "v485_delete_support_loss": delete_support_loss.detach(),
        "v485_fill_support_loss": fill_support_loss.detach(),
        "v485_boundary_support_loss": (trim_support_loss + expand_support_loss).detach() * 0.5,
        "v485_direction_loss": direction_loss.detach(),
        "v485_existence_loss": existence_loss.detach(),
        "v485_purity_loss": purity_loss.detach(),
        "v485_edit_budget_loss": edit_budget_loss.detach(),
        "v485_m2_local_loss": m2_loss.detach(),
        "v485_base_dice": base_soft_dice.mean().detach(),
        "v485_local_oracle_gain": hard_oracle_gain.detach(),
        "v485_positive_candidate_rate": positive_rate.detach(),
        "v485_harmful_candidate_rate": harmful_rate.detach(),
        "v485_candidate_mean_abs_change": mean_abs_change.detach(),
        "v485_candidate_max_abs_change": max_abs_change.detach(),
        "v485_candidate_non_noop_rate": non_noop_rate.detach(),
        "v485_support_precision": support_precision.detach(),
        "v485_correction_outside_ratio": outside_ratio.detach(),
        "v485_delete_support_iou": _iou_soft(local_supports[:, 0:1], fp_target).detach(),
        "v485_fill_support_iou": _iou_soft(local_supports[:, 1:2], fn_target).detach(),
        "v485_boundary_support_iou": (0.5 * (_iou_soft(local_supports[:, 2:3], boundary_target) + _iou_soft(local_supports[:, 3:4], boundary_target))).detach(),
        "v485_fp_removed": fp_removed.detach(),
        "v485_fn_added": fn_added.detach(),
        "v485_tp_removed": tp_removed.detach(),
        "v485_bg_added": bg_added.detach(),
        "v485_local_active_rate": (local_soft_gates > 0.15).float().mean().detach(),
        "v486_candidate_repair_loss": candidate_repair_loss.detach(),
        "v486_delete_repair_loss": delete_repair_loss.detach(),
        "v486_fill_repair_loss": fill_repair_loss.detach(),
        "v486_boundary_repair_loss": (0.5 * (trim_repair_loss + expand_repair_loss)).detach(),
        "v486_outside_preserve_loss": outside_preserve_loss.detach(),
        "v486_edit_floor_loss": edit_floor_loss.detach(),
        "v486_target_exists_rate": target_exists.mean().detach(),
        "v486_mean_target_area": target_area.mean().detach(),
    }
    diag.update({
        "v484_error_state_loss": diag["v485_error_state_loss"],
        "v484_existence_loss": diag["v485_existence_loss"],
        "v484_purity_loss": diag["v485_purity_loss"],
        "v484_fp_delete_loss": diag["v485_delete_support_loss"],
        "v484_fn_fill_loss": diag["v485_fill_support_loss"],
        "v484_boundary_loss": diag["v485_boundary_support_loss"],
        "v484_m2_local_loss": diag["v485_m2_local_loss"],
        "v484_m3_regret_loss": cand_probs.new_tensor(0.0),
        "v484_base_dice": diag["v485_base_dice"],
        "v484_local_oracle_gain": diag["v485_local_oracle_gain"],
        "v484_local_harmful_rate": diag["v485_harmful_candidate_rate"],
        "v484_local_active_rate": diag["v485_local_active_rate"],
        "v484_support_precision": diag["v485_support_precision"],
        "v484_correction_outside_ratio": diag["v485_correction_outside_ratio"],
    })
    return total, diag
