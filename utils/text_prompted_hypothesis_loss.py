#/home/tsz-25/MedCLIPSeg-pristine/utils/text_prompted_hypothesis_loss.py 
"""Compositional signed-error candidate-set objective.

This file intentionally keeps the historical public function name so existing
training code does not need a parallel loss module.  The active CEM path uses:

  1. connected signed-error components from the current consensus prediction,
  2. Hungarian set matching between components and learned correction modes,
  3. candidate-set coverage/risk optimization over single and composed modes,
  4. relative candidate-vs-Preserve utility, uncertainty calibration and LCB selection.

The consensus prediction is produced in the same forward pass and remains fully
trainable.  Detached tensors are used only to construct stable supervision
labels; no separately trained B0 checkpoint is assumed.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

EPS = 1.0e-6


def _m1(cfg: Any, key: str, default: Any) -> Any:
    node = getattr(cfg, "M1", None)
    if node is None:
        return default
    return node.get(key, default) if isinstance(node, dict) else getattr(node, key, default)


def _foreground_mask(masks: torch.Tensor) -> torch.Tensor:
    x = masks.float()
    while x.ndim > 3:
        middle = list(range(1, x.ndim - 2))
        binary = [d for d in middle if x.shape[d] == 2]
        if binary:
            x = x.select(binary[-1], 1)
            continue
        singleton = [d for d in middle if x.shape[d] == 1]
        if singleton:
            x = x.select(singleton[0], 0)
            continue
        raise ValueError(f"Cannot infer foreground mask from {tuple(masks.shape)}")
    return (x > 0.5).float()


def _soft_erode(x: torch.Tensor, radius: int = 1) -> torch.Tensor:
    radius = max(1, int(radius))
    return -F.max_pool2d(-x, 2 * radius + 1, 1, radius)


def _soft_dilate(x: torch.Tensor, radius: int = 1) -> torch.Tensor:
    radius = max(1, int(radius))
    return F.max_pool2d(x, 2 * radius + 1, 1, radius)


def _soft_boundary(x: torch.Tensor, radius: int = 1) -> torch.Tensor:
    return (_soft_dilate(x, radius) - _soft_erode(x, radius)).clamp(0.0, 1.0)


def _soft_dice_probs(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prob.ndim == 3:
        if target.ndim == 4:
            target = target[:, 0]
        inter = (prob * target).flatten(1).sum(dim=-1)
        den = prob.flatten(1).sum(dim=-1) + target.flatten(1).sum(dim=-1)
        return (2.0 * inter + EPS) / (den + EPS)
    if prob.ndim == 4:
        if target.ndim == 3:
            target = target[:, None].expand_as(prob)
        inter = (prob * target).flatten(2).sum(dim=-1)
        den = prob.flatten(2).sum(dim=-1) + target.flatten(2).sum(dim=-1)
        return (2.0 * inter + EPS) / (den + EPS)
    raise ValueError(f"Unexpected probability shape: {tuple(prob.shape)}")


def _hard_dice_probs(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = (prob >= 0.5).float()
    return _soft_dice_probs(pred, target)


def _segmentation_loss_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    boundary_weight: float = 0.25,
) -> torch.Tensor:
    if logits.ndim != 4:
        raise ValueError(f"Expected [B,K,H,W], got {tuple(logits.shape)}")
    target_k = target[:, None].expand_as(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target_k)
    prob = torch.sigmoid(logits).clamp(EPS, 1.0 - EPS)
    dice = 1.0 - _soft_dice_probs(prob, target).mean()
    if boundary_weight <= 0:
        return bce + dice
    pred_boundary = _soft_boundary(prob.reshape(-1, 1, *prob.shape[-2:]), 1)
    pred_boundary = pred_boundary.reshape_as(prob)
    gt_boundary = _soft_boundary(target[:, None], 1).expand_as(prob)
    boundary_loss = 1.0 - _soft_dice_probs(pred_boundary, gt_boundary).mean()
    return bce + dice + float(boundary_weight) * boundary_loss


def _segmentation_loss_from_probs(
    prob: torch.Tensor,
    target: torch.Tensor,
    boundary_weight: float = 0.25,
) -> torch.Tensor:
    p = prob.clamp(EPS, 1.0 - EPS)
    bce = F.binary_cross_entropy(p, target)
    dice = 1.0 - _soft_dice_probs(p, target).mean()
    pred_boundary = _soft_boundary(p[:, None], 1)[:, 0]
    gt_boundary = _soft_boundary(target[:, None], 1)[:, 0]
    boundary_loss = 1.0 - _soft_dice_probs(pred_boundary, gt_boundary).mean()
    return bce + dice + float(boundary_weight) * boundary_loss


def _build_signed_component_targets(
    gt: torch.Tensor,
    consensus_prob: torch.Tensor,
    num_modes: int,
    threshold: float,
    connect_radius: int,
    min_pixels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [B,K,H,W] signed residual components and component counts.

    Nearby hard errors of the same sign are grouped by a small binary dilation,
    while the target values retain the original undilated error pixels and the
    detached soft residual magnitude.
    """
    device, dtype = gt.device, gt.dtype
    batch, height, width = gt.shape
    out = torch.zeros(batch, num_modes, height, width, device=device, dtype=dtype)
    counts = torch.zeros(batch, device=device, dtype=dtype)

    gt_np = (gt.detach().cpu().numpy() > 0.5)
    prob_np = consensus_prob.detach().cpu().numpy()
    pred_np = prob_np >= float(threshold)
    residual_np = gt_np.astype(np.float32) - prob_np.astype(np.float32)

    structure = ndimage.generate_binary_structure(2, 2)
    for b in range(batch):
        components: list[tuple[int, np.ndarray]] = []
        signed_masks = (
            (1, gt_np[b] & (~pred_np[b])),
            (-1, (~gt_np[b]) & pred_np[b]),
        )
        for sign, binary in signed_masks:
            if not binary.any():
                continue
            grouped = binary
            if int(connect_radius) > 0:
                grouped = ndimage.binary_dilation(
                    binary,
                    structure=structure,
                    iterations=int(connect_radius),
                )
            labels, n_labels = ndimage.label(grouped, structure=structure)
            sign_components: list[tuple[int, np.ndarray]] = []
            for label_id in range(1, int(n_labels) + 1):
                original = binary & (labels == label_id)
                area = int(original.sum())
                if area < int(min_pixels):
                    continue
                target = np.zeros((height, width), dtype=np.float32)
                target[original] = residual_np[b][original]
                # Numerical protection: retain the requested sign even when the
                # consensus probability is extremely close to the hard target.
                if sign > 0:
                    target[original] = np.maximum(target[original], 0.05)
                else:
                    target[original] = np.minimum(target[original], -0.05)
                sign_components.append((area, target))

            # Do not silently discard a real error direction merely because all
            # of its components are smaller than the configured minimum.  In
            # that rare case use one aggregate low-area target for this sign.
            if not sign_components:
                original = binary
                target = np.zeros((height, width), dtype=np.float32)
                target[original] = residual_np[b][original]
                if sign > 0:
                    target[original] = np.maximum(target[original], 0.05)
                else:
                    target[original] = np.minimum(target[original], -0.05)
                sign_components.append((int(original.sum()), target))
            components.extend(sign_components)

        components.sort(key=lambda item: item[0], reverse=True)
        selected = components[:num_modes]
        counts[b] = float(len(selected))
        for k, (_, target_np) in enumerate(selected):
            out[b, k] = torch.from_numpy(target_np).to(device=device, dtype=dtype)

    return out, counts



def _build_typed_logit_targets(
    gt: torch.Tensor,
    consensus_prob: torch.Tensor,
    num_modes: int,
    max_atom_delta: float,
    boundary_radius: int,
    target_smoothing: float,
    min_pixels: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """V479 direct fixed-semantic targets for Delete/Fill/Trim/Expand.

    Targets are mutually exclusive and corrections are expressed in logit
    space, exactly matching the quantity predicted by the candidate generator.
    """
    hard = (consensus_prob.detach() >= 0.5).to(gt.dtype)
    fp = hard * (1.0 - gt)
    fn = (1.0 - hard) * gt
    boundary = _soft_boundary(gt[:, None], max(1, int(boundary_radius)))[:, 0]
    boundary = (boundary > 0.0).to(gt.dtype)
    interior = 1.0 - boundary
    typed = [fp * interior, fn * interior, fp * boundary, fn * boundary]
    support = gt.new_zeros(gt.shape[0], num_modes, *gt.shape[-2:])
    for index in range(min(num_modes, 4)):
        support[:, index] = typed[index]
    eps = min(max(float(target_smoothing), 1.0e-4), 0.49)
    y_smooth = eps + (1.0 - 2.0 * eps) * gt
    target_delta = torch.logit(y_smooth.clamp(EPS, 1.0 - EPS)) - torch.logit(
        consensus_prob.detach().clamp(EPS, 1.0 - EPS)
    )
    target_delta = target_delta.clamp(-float(max_atom_delta), float(max_atom_delta))
    correction = support * target_delta[:, None]
    presence = support.flatten(2).sum(dim=-1) >= float(max(1, min_pixels))
    return support, correction, presence.to(gt.dtype)

def _hungarian_align_targets(
    attention: torch.Tensor,
    correction: torch.Tensor,
    targets: torch.Tensor,
    max_atom_delta: float,
) -> torch.Tensor:
    """Match unordered predicted modes to unordered signed error components."""
    batch, modes = attention.shape[:2]
    aligned = torch.zeros_like(targets)
    pred_norm = (correction / max(float(max_atom_delta), EPS)).clamp(-1.0, 1.0)

    with torch.no_grad():
        for b in range(batch):
            costs = attention.new_zeros((modes, modes))
            for i in range(modes):
                pred_support = attention[b, i]
                pred_field = pred_norm[b, i]
                support_mass = pred_support.sum().clamp_min(EPS)
                for j in range(modes):
                    target = targets[b, j]
                    target_abs = (target.abs() > 0).to(target.dtype)
                    target_area = target_abs.sum()
                    if float(target_area) < 0.5:
                        costs[i, j] = pred_support.mean() + pred_field.abs().mean()
                        continue
                    inter = (pred_support * target_abs).sum()
                    dice_cost = 1.0 - (2.0 * inter + EPS) / (
                        support_mass + target_area + EPS
                    )
                    target_sign = target.sign()
                    sign_cost = (
                        F.relu(-pred_field * target_sign) * target_abs
                    ).sum() / target_area.clamp_min(1.0)
                    leakage = (
                        pred_support * (1.0 - target_abs)
                    ).sum() / support_mass
                    costs[i, j] = dice_cost + 0.50 * sign_cost + 0.20 * leakage

            rows, cols = linear_sum_assignment(costs.detach().cpu().numpy())
            for row, col in zip(rows.tolist(), cols.tolist()):
                aligned[b, row] = targets[b, col]
    return aligned



def _per_case_segmentation_loss_from_probs(
    prob: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Per-case BCE+Dice used for final-policy tail control."""
    p = prob.clamp(EPS, 1.0 - EPS)
    y = target.to(dtype=p.dtype)
    bce = F.binary_cross_entropy(p, y, reduction="none").flatten(1).mean(dim=1)
    inter = (p * y).flatten(1).sum(dim=1)
    den = p.flatten(1).sum(dim=1) + y.flatten(1).sum(dim=1)
    dice = 1.0 - (2.0 * inter + EPS) / (den + EPS)
    return bce + dice


def _balanced_support_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    positive = target.sum(dim=(-2, -1), keepdim=True)
    negative = (1.0 - target).sum(dim=(-2, -1), keepdim=True)
    positive_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 8.0)
    positive_loss = F.softplus(-logits) * target * positive_weight
    negative_loss = F.softplus(logits) * (1.0 - target)
    denom = (
        target * positive_weight + (1.0 - target)
    ).sum().clamp_min(1.0)
    return (positive_loss + negative_loss).sum() / denom


def _mode_redundancy(attention: torch.Tensor, correction: torch.Tensor) -> torch.Tensor:
    _, modes, _, _ = attention.shape
    if modes <= 1:
        return attention.new_zeros(())
    penalties = []
    for i in range(modes):
        for j in range(i + 1, modes):
            overlap = (attention[:, i] * attention[:, j]).mean(dim=(-2, -1))
            cosine = F.cosine_similarity(
                correction[:, i].flatten(1),
                correction[:, j].flatten(1),
                dim=1,
                eps=EPS,
            ).clamp_min(0.0)
            penalties.append(overlap * cosine)
    return torch.stack(penalties, dim=1).mean()


def _pairwise_ranking_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    margin: float,
    temperature: float,
) -> torch.Tensor:
    candidates = predicted.shape[1]
    losses = []
    for i in range(candidates):
        for j in range(i + 1, candidates):
            target_diff = target[:, i] - target[:, j]
            valid = target_diff.abs() > float(margin)
            if not bool(valid.any()):
                continue
            sign = target_diff.sign()
            pred_diff = (predicted[:, i] - predicted[:, j]) / max(
                float(temperature), 1.0e-4
            )
            losses.append(F.softplus(-sign[valid] * pred_diff[valid]).mean())
    return torch.stack(losses).mean() if losses else predicted.new_zeros(())


def _legacy_direct_hypothesis_loss(
    cfg: Any,
    candidates: torch.Tensor,
    masks: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Safe compatibility path for old direct-hypothesis checkpoints.

    It deliberately removes morphology-derived pseudo-role targets and global
    diversity forcing.  New experiments should use the CEM auxiliary fields.
    """
    gt = _foreground_mask(masks).to(candidates.dtype)
    c0 = candidates[:, 0]
    hypotheses = candidates[:, 1:]
    loss = _segmentation_loss_from_logits(hypotheses, gt, boundary_weight=0.25)
    base_hard = _hard_dice_probs(torch.sigmoid(c0), gt)
    hyp_hard = _hard_dice_probs(torch.sigmoid(hypotheses), gt)
    best, best_idx = hyp_hard.max(dim=1)
    gain = hyp_hard - base_hard[:, None]
    diagnostics = {
        "tpmhg_loss": loss.detach(),
        "tpmhg_base_dice": base_hard.mean().detach(),
        "tpmhg_best_dice": best.mean().detach(),
        "tpmhg_oracle_gain": (best - base_hard).mean().detach(),
        "tpmhg_positive_candidate_rate": (gain > 5.0e-4).float().mean().detach(),
        "tpmhg_harmful_candidate_rate": (gain < -5.0e-4).float().mean().detach(),
        "tpmhg_positive_case_rate": (gain > 5.0e-4).any(dim=1).float().mean().detach(),
        "tpmhg_best_slot_mean": (best_idx.float() + 1.0).mean().detach(),
    }
    return loss, diagnostics


def _compute_text_prompted_hypothesis_loss_v480(
    cfg: Any,
    candidates: torch.Tensor,
    masks: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    epoch=None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if candidates.ndim != 4 or candidates.shape[1] < 2:
        raise ValueError(f"CEM expects candidates [B,M,H,W], got {tuple(candidates.shape)}")

    required = {
        "cem_atom_attention_logits",
        "cem_atom_attention",
        "cem_atom_delta_raw",
        "cem_atom_corrections",
        "cem_quality_utility",
        "cem_quality_sigma",
        "cem_quality_harm_logits",
        "cem_failure_logit",
        "cem_selection_scores",
        "cem_st_selected_probs",
    }
    if not required.issubset(aux):
        return _legacy_direct_hypothesis_loss(cfg, candidates, masks)

    gt = _foreground_mask(masks).to(device=candidates.device, dtype=candidates.dtype)
    consensus_logits = candidates[:, 0]
    candidate_probs = torch.sigmoid(candidates).clamp(EPS, 1.0 - EPS)
    consensus_prob = candidate_probs[:, 0]
    hypotheses = candidates[:, 1:]

    attention_logits = aux["cem_atom_attention_logits"]
    attention = aux["cem_atom_attention"]
    delta_raw = aux["cem_atom_delta_raw"]
    corrections = aux["cem_atom_corrections"]
    num_modes = corrections.shape[1]
    max_atom_delta = float(_m1(cfg, "CEM_MAX_ATOM_LOGIT_DELTA", 3.5))

    direct_typed_targets = bool(_m1(cfg, "CEM_TYPED_DIRECT_TARGETS", False))
    if direct_typed_targets and num_modes >= 4:
        target_support, target_correction, target_presence = _build_typed_logit_targets(
            gt=gt,
            consensus_prob=consensus_prob,
            num_modes=num_modes,
            max_atom_delta=max_atom_delta,
            boundary_radius=int(_m1(cfg, "CEM_TYPED_BOUNDARY_RADIUS", 2)),
            target_smoothing=float(_m1(cfg, "CEM_LOGIT_TARGET_SMOOTHING", 0.05)),
            min_pixels=int(_m1(cfg, "CEM_COMPONENT_MIN_PIXELS", 3)),
        )
        component_count = target_presence.sum(dim=1)
        aligned_targets = target_correction / max(max_atom_delta, EPS)
    else:
        component_targets, component_count = _build_signed_component_targets(
            gt=gt,
            consensus_prob=consensus_prob.detach(),
            num_modes=num_modes,
            threshold=float(_m1(cfg, "CEM_COMPONENT_THRESHOLD", 0.50)),
            connect_radius=int(_m1(cfg, "CEM_COMPONENT_CONNECT_RADIUS", 1)),
            min_pixels=int(_m1(cfg, "CEM_COMPONENT_MIN_PIXELS", 3)),
        )
        aligned_targets = _hungarian_align_targets(
            attention=attention,
            correction=corrections,
            targets=component_targets,
            max_atom_delta=max_atom_delta,
        )
        target_support = (aligned_targets.abs() > 0).to(candidates.dtype)
        target_correction = aligned_targets * max_atom_delta
        target_presence = (target_support.flatten(2).sum(dim=-1) > 0).to(candidates.dtype)
    target_sign = target_correction.sign()
    target_magnitude = target_correction.abs().clamp(0.0, max_atom_delta)

    support_bce = _balanced_support_bce(attention_logits, target_support)
    support_dice = 1.0 - _soft_dice_probs(attention, target_support).mean()

    pred_field = (corrections / max(max_atom_delta, EPS)).clamp(-1.0, 1.0)
    inside_weight = target_support
    inside_denom = inside_weight.sum().clamp_min(1.0)
    signed_field_loss = (
        F.smooth_l1_loss(
            corrections,
            target_correction,
            beta=0.25,
            reduction="none",
        )
        * inside_weight
    ).sum() / inside_denom
    typed_modes = bool(float(aux.get(
        "cem_typed_modes", candidates.new_zeros(())
    ).detach().mean().cpu()))
    if typed_modes:
        # Direction is structurally fixed by the typed candidate mechanism;
        # only support and magnitude need supervision.
        sign_loss = delta_raw.sum() * 0.0
    else:
        sign_loss = (
            F.softplus(-target_sign * delta_raw) * inside_weight
        ).sum() / inside_denom
    outside_sparsity = (
        pred_field.abs() * (1.0 - target_support)
    ).mean()

    # V474.2 precision controls.  The old outside loss was averaged over the
    # whole image, so a spatially diffuse correction could receive a small loss
    # simply because the lesion occupies few pixels.  These normalized ratios
    # directly optimize support/correction precision independent of image size.
    support_tp = (attention * target_support).flatten(2).sum(dim=-1)
    support_pred = attention.flatten(2).sum(dim=-1)
    support_precision = (support_tp + EPS) / (support_pred + EPS)
    target_present = target_support.flatten(2).sum(dim=-1) > 0
    support_precision_loss = (
        (1.0 - support_precision[target_present]).mean()
        if bool(target_present.any())
        else attention.mean()
    )

    correction_abs = pred_field.abs()
    correction_total = correction_abs.flatten(2).sum(dim=-1)
    correction_outside = (
        correction_abs * (1.0 - target_support)
    ).flatten(2).sum(dim=-1)
    correction_outside_ratio = correction_outside / correction_total.clamp_min(EPS)
    correction_precision_loss = (
        correction_outside_ratio[target_present].mean()
        if bool(target_present.any())
        else correction_abs.mean()
    )

    no_op_loss = (
        attention.mean(dim=(-2, -1)) + pred_field.abs().mean(dim=(-2, -1))
    )
    no_op_loss = (
        no_op_loss[~target_present].mean()
        if bool((~target_present).any())
        else no_op_loss.new_zeros(())
    )

    predicted_gates = aux.get("cem_atom_gates")
    if isinstance(predicted_gates, torch.Tensor):
        gate_loss = F.binary_cross_entropy(
            predicted_gates.clamp(EPS, 1.0 - EPS), target_presence
        )
    else:
        gate_loss = attention.sum() * 0.0
    leakage_loss = (
        corrections.abs() * (1.0 - target_support)
    ).flatten(2).sum(dim=-1) / corrections.abs().flatten(2).sum(dim=-1).clamp_min(EPS)
    leakage_loss = leakage_loss.mean()

    component_union = target_support.amax(dim=1)
    hard_consensus = consensus_prob.detach() >= float(
        _m1(cfg, "CEM_COMPONENT_THRESHOLD", 0.50)
    )
    error_union = ((gt > 0.5) != hard_consensus).to(candidates.dtype)
    predicted_union = 1.0 - torch.prod(1.0 - attention.clamp(0.0, 1.0), dim=1)
    error_case = error_union.flatten(1).sum(dim=1) > 0
    union_dice = _soft_dice_probs(predicted_union, error_union)
    coverage_loss = (
        (1.0 - union_dice[error_case]).mean()
        if bool(error_case.any())
        else predicted_union.mean()
    )
    redundancy_loss = _mode_redundancy(attention, corrections)

    # Candidate-level utility combines overlap, boundary agreement and actual
    # harmful changed mass.  All targets used by quality heads are detached.
    dice_soft = _soft_dice_probs(candidate_probs, gt)
    candidate_boundary = _soft_boundary(
        candidate_probs.reshape(-1, 1, *candidate_probs.shape[-2:]), 1
    ).reshape_as(candidate_probs)
    gt_boundary = _soft_boundary(gt[:, None], 1).expand_as(candidate_probs)
    boundary_dice = _soft_dice_probs(candidate_boundary, gt_boundary)

    base = candidate_probs[:, :1]
    add = (candidate_probs - base).relu()
    remove = (base - candidate_probs).relu()
    beneficial_mass = (
        add * gt[:, None] + remove * (1.0 - gt[:, None])
    ).mean(dim=(-2, -1))
    harmful_mass = (
        add * (1.0 - gt[:, None]) + remove * gt[:, None]
    ).mean(dim=(-2, -1))
    edit_mass = (add + remove).mean(dim=(-2, -1))
    harm_fraction = harmful_mass / edit_mass.clamp_min(EPS)
    harm_fraction[:, 0] = 0.0

    # Separate the two clinically relevant harmful directions.  The observed
    # BUSI failures are dominated by false-positive expansion, so background
    # additions receive an explicit penalty rather than being diluted inside a
    # generic mean harmful mass.
    false_positive_edit_mass = (add * (1.0 - gt[:, None])).mean(dim=(-2, -1))
    true_positive_removal_mass = (remove * gt[:, None]).mean(dim=(-2, -1))
    candidate_fp_edit_loss = false_positive_edit_mass[:, 1:].mean()
    candidate_tp_removal_loss = true_positive_removal_mass[:, 1:].mean()

    boundary_utility_weight = float(_m1(cfg, "CEM_UTILITY_BOUNDARY_WEIGHT", 0.25))
    hard_candidate_dice = _hard_dice_probs(candidate_probs.detach(), gt)
    hard_delta_dice = hard_candidate_dice - hard_candidate_dice[:, :1]
    # M2 is trained on the same candidate-minus-Preserve objective used for
    # deployment.  Pixel-level harmful mass remains a generator diagnostic,
    # not an absolute gate that can reject a net-positive candidate.
    relative_utility = (
        hard_delta_dice
        + boundary_utility_weight * (boundary_dice - boundary_dice[:, :1]).detach()
    )
    relative_utility[:, 0] = 0.0

    # Local typed candidates are directly supervised because each one is a
    # constrained correction of the factual prediction. Discovery candidates
    # are intentionally supervised with a set-level soft oracle below: forcing
    # every independent discovery proposal to match the same GT makes the
    # queries collapse and destroys the alternative-location hypothesis that is
    # required for catastrophic factual localisation failures.
    local_hypotheses = hypotheses[:, :num_modes]
    candidate_validity_loss = _segmentation_loss_from_logits(
        local_hypotheses,
        gt,
        boundary_weight=float(_m1(cfg, "CEM_CANDIDATE_BOUNDARY_WEIGHT", 0.25)),
    )

    discovery_probs = candidate_probs[:, 1 + num_modes :]
    if discovery_probs.shape[1] > 0:
        discovery_boundary_weight = float(
            _m1(cfg, "CEM_DISCOVERY_BOUNDARY_WEIGHT", 0.25)
        )
        discovery_quality = (
            dice_soft[:, 1 + num_modes :]
            + discovery_boundary_weight * boundary_dice[:, 1 + num_modes :]
        ) / (1.0 + discovery_boundary_weight)
        discovery_temperature = max(
            float(_m1(cfg, "CEM_DISCOVERY_ORACLE_TEMPERATURE", 0.05)),
            1.0e-4,
        )
        discovery_weights = torch.softmax(
            discovery_quality / discovery_temperature, dim=1
        )
        discovery_oracle_quality = (
            discovery_weights * discovery_quality
        ).sum(dim=1)
        discovery_oracle_loss = (1.0 - discovery_oracle_quality).mean()
    else:
        discovery_oracle_quality = candidate_probs.new_zeros(
            candidate_probs.shape[0]
        )
        discovery_oracle_loss = candidate_probs.sum() * 0.0

    oracle_temperature = max(
        float(_m1(cfg, "CEM_ORACLE_TEMPERATURE", 0.02)), 1.0e-4
    )
    oracle_weights = torch.softmax(relative_utility / oracle_temperature, dim=1)
    soft_oracle_gain = (oracle_weights * relative_utility).sum(dim=1)
    target_oracle_gain = float(_m1(cfg, "CEM_TARGET_ORACLE_GAIN", 0.003))
    oracle_margin_loss = F.relu(target_oracle_gain - soft_oracle_gain).mean()

    max_edit_fraction = float(_m1(cfg, "CEM_MAX_EDIT_FRACTION", 0.08))
    edit_budget_loss = F.relu(edit_mass[:, 1:] - max_edit_fraction).pow(2).mean()
    candidate_harm_limit = float(_m1(cfg, "CEM_TARGET_MAX_HARM", 0.50))
    candidate_harm_loss = (
        F.relu(harm_fraction[:, 1:] - candidate_harm_limit)
        * edit_mass[:, 1:]
    ).mean()
    harm_fraction_penalty = F.relu(
        harm_fraction[:, 1:] - candidate_harm_limit
    ).pow(2).mean()

    family_effect_enabled = all(
        key in aux
        for key in (
            "cem_cf_mean_dsc", "cem_cf_q10_dsc",
            "cem_cf_mean_nsd", "cem_cf_q10_nsd",
        )
    )
    cf_mean_loss = candidates.new_zeros(())
    cf_quantile_loss = candidates.new_zeros(())
    cf_q10_coverage = candidates.new_zeros(())
    if family_effect_enabled:
        target_dsc_effect = hard_delta_dice.detach()
        target_nsd_effect = (boundary_dice - boundary_dice[:, :1]).detach()
        mean_dsc_pred = aux["cem_cf_mean_dsc"]
        q_dsc_pred = aux["cem_cf_q10_dsc"]
        mean_nsd_pred = aux["cem_cf_mean_nsd"]
        q_nsd_pred = aux["cem_cf_q10_nsd"]
        gain_weight = 1.0 + float(_m1(cfg, "CEM_CF_HIGH_GAIN_WEIGHT", 12.0)) * (
            target_dsc_effect.abs().clamp_max(0.10)
        )
        catastrophic = (hard_candidate_dice[:, :1] < float(
            _m1(cfg, "CEM_CATASTROPHIC_BASE_DICE", 0.5)
        )).to(gain_weight.dtype)
        high_gain = (target_dsc_effect > float(
            _m1(cfg, "CEM_CF_HIGH_GAIN_MARGIN", 0.03)
        )).to(gain_weight.dtype)
        gain_weight = gain_weight * (1.0 + catastrophic * high_gain)
        cf_mean_loss = (
            F.smooth_l1_loss(mean_dsc_pred, target_dsc_effect, beta=0.01, reduction="none")
            + float(_m1(cfg, "CEM_CF_NSD_LOSS_WEIGHT", 0.5))
            * F.smooth_l1_loss(mean_nsd_pred, target_nsd_effect, beta=0.01, reduction="none")
        )
        cf_mean_loss = (cf_mean_loss * gain_weight).sum() / gain_weight.sum().clamp_min(1.0)
        alpha = float(_m1(cfg, "CEM_CF_QUANTILE_ALPHA", 0.10))
        def _pinball(pred, target):
            residual = target - pred
            return torch.maximum(alpha * residual, (alpha - 1.0) * residual)
        cf_quantile_matrix = _pinball(q_dsc_pred, target_dsc_effect) + float(
            _m1(cfg, "CEM_CF_NSD_LOSS_WEIGHT", 0.5)
        ) * _pinball(q_nsd_pred, target_nsd_effect)
        cf_quantile_loss = (cf_quantile_matrix * gain_weight).sum() / gain_weight.sum().clamp_min(1.0)
        cf_q10_coverage = (target_dsc_effect >= q_dsc_pred).float().mean()

    predicted_utility = aux["cem_quality_utility"]
    predicted_sigma = aux["cem_quality_sigma"].clamp_min(EPS)
    predicted_benefit_logits = aux["cem_quality_benefit_logits"]
    predicted_harm_logits = aux["cem_quality_harm_logits"]
    if predicted_utility.shape != relative_utility.shape:
        raise ValueError(
            "CEM utility shape mismatch: "
            f"pred={tuple(predicted_utility.shape)} target={tuple(relative_utility.shape)}"
        )

    target = relative_utility.detach()
    error = predicted_utility[:, 1:] - target[:, 1:]
    sigma_actions = predicted_sigma[:, 1:].clamp_min(EPS)
    base_hard_dice = hard_candidate_dice[:, 0]
    catastrophic_threshold = float(
        _m1(cfg, "CEM_CATASTROPHIC_BASE_DICE", 0.50)
    )
    catastrophic_weight = float(
        _m1(cfg, "CEM_CATASTROPHIC_CASE_WEIGHT", 2.0)
    )
    gain_weight = float(_m1(cfg, "CEM_GAIN_IMPORTANCE_WEIGHT", 4.0))
    quality_weight = 1.0 + gain_weight * target[:, 1:].abs()
    quality_weight = quality_weight * torch.where(
        (base_hard_dice < catastrophic_threshold)[:, None],
        quality_weight.new_full(quality_weight.shape, catastrophic_weight),
        quality_weight.new_ones(quality_weight.shape),
    )
    # V477.1: learn the relative utility mean and explicitly calibrate the
    # predicted uncertainty to the detached absolute residual.  This remains
    # strictly non-negative, which is important because the outer trainer
    # uses the proposal-loss magnitude to cap the candidate/base loss ratio.
    utility_mean_error = F.smooth_l1_loss(
        predicted_utility[:, 1:],
        target[:, 1:],
        beta=float(_m1(cfg, "CEM_UTILITY_REG_BETA", 0.01)),
        reduction="none",
    )
    utility_mean_loss = (
        quality_weight * utility_mean_error
    ).sum() / quality_weight.sum().clamp_min(1.0)
    sigma_target = error.detach().abs().clamp(
        min=float(_m1(cfg, "CEM_SELECTOR_MIN_SIGMA", 0.002)),
        max=float(_m1(cfg, "CEM_SELECTOR_MAX_SIGMA", 0.10)),
    )
    sigma_calibration_loss = F.smooth_l1_loss(
        sigma_actions,
        sigma_target,
        beta=float(_m1(cfg, "CEM_SIGMA_CALIBRATION_BETA", 0.005)),
    )
    utility_regression_loss = (
        utility_mean_loss
        + float(_m1(cfg, "CEM_SIGMA_CALIBRATION_WEIGHT", 0.50))
        * sigma_calibration_loss
    )

    positive_gain_margin = float(_m1(cfg, "CEM_POSITIVE_GAIN_MARGIN", 0.0005))
    harm_margin = float(_m1(cfg, "CEM_HARM_GAIN_MARGIN", 0.0005))
    benefit_target = (target[:, 1:] > positive_gain_margin).to(candidates.dtype)
    harmful_target = (target[:, 1:] < -harm_margin).to(candidates.dtype)
    positive_count = benefit_target.sum()
    negative_count = benefit_target.numel() - positive_count
    benefit_pos_weight = (
        negative_count / positive_count.clamp_min(1.0)
    ).clamp(1.0, float(_m1(cfg, "CEM_BENEFIT_POS_WEIGHT_MAX", 8.0)))
    benefit_classification_loss = F.binary_cross_entropy_with_logits(
        predicted_benefit_logits[:, 1:],
        benefit_target,
        pos_weight=benefit_pos_weight.detach(),
    )
    harm_regression_loss = F.binary_cross_entropy_with_logits(
        predicted_harm_logits[:, 1:],
        harmful_target,
    )
    candidate_accept_loss = candidates.new_zeros(())
    candidate_accept_logits = aux.get("cem_candidate_accept_logits")
    if isinstance(candidate_accept_logits, torch.Tensor):
        # The final Pareto-safe target is constructed below; this placeholder
        # is overwritten after DSC/NSD-safe targets are available.
        candidate_accept_loss = candidate_accept_logits.sum() * 0.0
    ranking_loss = _pairwise_ranking_loss(
        predicted_utility,
        target,
        margin=float(_m1(cfg, "CEM_RANK_MARGIN", 0.0005)),
        temperature=float(_m1(cfg, "CEM_RANK_TEMPERATURE", 0.02)),
    )

    # V478 M2 counterfactual supervision.  Every candidate is an explicit
    # model intervention and C0 is the factual no-intervention outcome.  The
    # pairwise target is a smooth function of the true intervention-effect
    # difference, avoiding brittle one-hot labels for nearly tied candidates.
    cf_pairwise_logits = aux.get("cem_cf_pairwise_logits")
    causal_pairwise_loss = candidates.new_zeros(())
    causal_pairwise_accuracy = candidates.new_zeros(())
    causal_null_loss = predicted_utility[:, 0].abs().mean()
    if isinstance(cf_pairwise_logits, torch.Tensor):
        if cf_pairwise_logits.shape[:2] != target.shape or cf_pairwise_logits.shape[2] != target.shape[1]:
            raise ValueError(
                "CEM counterfactual pairwise shape mismatch: "
                f"pred={tuple(cf_pairwise_logits.shape)} target={tuple(target.shape)}"
            )
        effect_difference = target[:, :, None] - target[:, None, :]
        pair_temperature = max(
            float(_m1(cfg, "CEM_CF_PAIRWISE_TEMPERATURE", 0.01)), 1.0e-4
        )
        pair_target = torch.sigmoid(effect_difference / pair_temperature)
        pair_weight = 1.0 + float(
            _m1(cfg, "CEM_CF_PAIRWISE_GAIN_WEIGHT", 8.0)
        ) * effect_difference.abs()
        candidate_count = target.shape[1]
        pair_mask = torch.triu(
            torch.ones(
                candidate_count,
                candidate_count,
                device=target.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )[None].expand(target.shape[0], -1, -1)
        pair_bce = F.binary_cross_entropy_with_logits(
            cf_pairwise_logits, pair_target, reduction="none"
        )
        causal_pairwise_loss = (
            pair_bce[pair_mask] * pair_weight[pair_mask]
        ).sum() / pair_weight[pair_mask].sum().clamp_min(1.0)
        non_tie = effect_difference.abs() > float(
            _m1(cfg, "CEM_CF_PAIRWISE_MARGIN", 0.0005)
        )
        eval_mask = pair_mask & non_tie
        if bool(eval_mask.any()):
            predicted_order = cf_pairwise_logits > 0
            true_order = effect_difference > 0
            causal_pairwise_accuracy = (
                predicted_order[eval_mask] == true_order[eval_mask]
            ).float().mean()

    predicted_lcb_source = aux.get(
        "cem_cf_conservative_score", aux["cem_selection_scores"]
    )
    predicted_lcb = predicted_lcb_source[:, 1:]
    positive_lcb_loss = (
        F.relu(positive_gain_margin - predicted_lcb).pow(2)
        * benefit_target
    ).sum() / benefit_target.sum().clamp_min(1.0)
    harmful_lcb_loss = (
        F.relu(predicted_lcb).pow(2) * harmful_target
    ).sum() / harmful_target.sum().clamp_min(1.0)
    lcb_calibration_loss = positive_lcb_loss + harmful_lcb_loss

    minimum_gain = float(_m1(cfg, "CEM_SELECTOR_TARGET_GAIN", 0.001))
    if bool(_m1(cfg, "CEM_PARETO_SAFE_TARGETS", False)) and family_effect_enabled:
        safe_target = (
            (target_dsc_effect >= float(_m1(cfg, "CEM_SAFE_DSC_FLOOR", 0.0)))
            & (target_nsd_effect >= float(_m1(cfg, "CEM_SAFE_NSD_FLOOR", 0.0)))
            & (target >= minimum_gain)
        )
    else:
        safe_target = target >= minimum_gain
    safe_target = safe_target.clone()
    safe_target[:, 0] = False
    safe_utility_target = target.masked_fill(~safe_target, -1.0e4)
    best_safe_utility, best_safe_index = safe_utility_target.max(dim=1)
    failure_target = safe_target[:, 1:].any(dim=1).to(candidates.dtype)

    if isinstance(candidate_accept_logits, torch.Tensor):
        accept_target = safe_target[:, 1:].to(candidates.dtype)
        accept_positive = accept_target.sum()
        accept_negative = accept_target.numel() - accept_positive
        accept_pos_weight = (
            accept_negative / accept_positive.clamp_min(1.0)
        ).clamp(1.0, float(_m1(cfg, "CEM_ACCEPT_POS_WEIGHT_MAX", 8.0)))
        candidate_accept_loss = F.binary_cross_entropy_with_logits(
            candidate_accept_logits[:, 1:],
            accept_target,
            pos_weight=accept_pos_weight.detach(),
        )
    failure_loss = F.binary_cross_entropy_with_logits(
        aux["cem_failure_logit"], failure_target
    )
    target_index = torch.where(
        failure_target > 0.5,
        best_safe_index,
        torch.zeros_like(best_safe_index),
    )
    selection_loss = F.cross_entropy(
        aux["cem_selection_scores"]
        / max(float(_m1(cfg, "CEM_SELECTION_LOSS_TEMPERATURE", 0.10)), 1.0e-4),
        target_index,
    )

    selected_segmentation_loss = _segmentation_loss_from_probs(
        aux["cem_st_selected_probs"],
        gt,
        boundary_weight=float(_m1(cfg, "CEM_SELECTED_BOUNDARY_WEIGHT", 0.35)),
    )
    expected_harm_loss = (
        aux["cem_selection_soft"] * harm_fraction.detach()
    ).sum(dim=1).mean()
    harmful_candidate_indicator = torch.cat(
        [harmful_target.new_zeros((harmful_target.shape[0], 1)), harmful_target],
        dim=1,
    )
    expected_negative_gain_loss = (
        aux["cem_selection_soft"] * harmful_candidate_indicator.detach()
    ).sum(dim=1).mean()

    # M3 learns the final intervention policy from M2 causal evidence.  Policy
    # regret directly measures how much true candidate utility is left on the
    # table, while the tail term prevents a small set of catastrophic cases
    # from being hidden by a good mean.
    policy_expected_utility = (
        aux["cem_selection_soft"] * target
    ).sum(dim=1)
    policy_oracle_utility = target.max(dim=1).values
    policy_regret_loss = (
        policy_oracle_utility - policy_expected_utility
    ).clamp_min(0.0).mean()
    soft_final_probs = aux["cem_soft_selected_probs"]
    final_case_loss = _per_case_segmentation_loss_from_probs(
        soft_final_probs, gt
    )
    tail_fraction = float(_m1(cfg, "CEM_FINAL_TAIL_FRACTION", 0.20))
    tail_fraction = min(max(tail_fraction, 1.0 / max(final_case_loss.numel(), 1)), 1.0)
    tail_k = max(1, int(math.ceil(final_case_loss.numel() * tail_fraction)))
    final_tail_loss = torch.topk(
        final_case_loss, k=tail_k, largest=True
    ).values.mean()

    discovery_coherence_loss = aux.get(
        "cem_discovery_coherence_energy", candidates.new_zeros(())
    )
    discovery_diversity_loss = aux.get(
        "cem_discovery_diversity_loss", candidates.new_zeros(())
    )

    current_epoch = int(epoch or 0)
    selector_start = int(_m1(cfg, "CEM_SELECTOR_START_EPOCH", 5))
    selector_ramp_epochs = max(1, int(_m1(cfg, "CEM_SELECTOR_RAMP_EPOCHS", 10)))
    selector_scale = min(
        max((current_epoch - selector_start + 1) / float(selector_ramp_epochs), 0.0),
        1.0,
    )

    compact_local_loss = (
        float(_m1(cfg, "CEM_V479_SUPPORT_WEIGHT", 0.5)) * (support_bce + support_dice)
        + float(_m1(cfg, "CEM_V479_LOGIT_WEIGHT", 0.5)) * signed_field_loss
        + float(_m1(cfg, "CEM_V479_GATE_WEIGHT", 0.25)) * gate_loss
        + float(_m1(cfg, "CEM_V479_LEAKAGE_WEIGHT", 0.25)) * leakage_loss
    ) if direct_typed_targets else candidates.new_zeros(())

    total = (
        float(_m1(cfg, "CEM_V479_COMPACT_LOCAL_WEIGHT", 0.0)) * compact_local_loss
        + float(_m1(cfg, "CEM_CF_MEAN_WEIGHT", 0.0)) * cf_mean_loss
        + float(_m1(cfg, "CEM_CF_QUANTILE_WEIGHT", 0.0)) * cf_quantile_loss
        + float(_m1(cfg, "CEM_ATOM_SUPPORT_WEIGHT", 0.60))
        * (support_bce + support_dice)
        + float(_m1(cfg, "CEM_ATOM_SIGNED_WEIGHT", 0.75))
        * (signed_field_loss + 0.50 * sign_loss)
        + float(_m1(cfg, "CEM_ATOM_OUTSIDE_WEIGHT", 0.20)) * outside_sparsity
        + float(_m1(cfg, "CEM_SUPPORT_PRECISION_WEIGHT", 0.0))
        * support_precision_loss
        + float(_m1(cfg, "CEM_CORRECTION_PRECISION_WEIGHT", 0.0))
        * correction_precision_loss
        + float(_m1(cfg, "CEM_ATOM_NOOP_WEIGHT", 0.20)) * no_op_loss
        + float(_m1(cfg, "CEM_ERROR_COVERAGE_WEIGHT", 0.75)) * coverage_loss
        + float(_m1(cfg, "CEM_REDUNDANCY_WEIGHT", 0.08)) * redundancy_loss
        + float(_m1(cfg, "CEM_CANDIDATE_VALIDITY_WEIGHT", 0.20))
        * candidate_validity_loss
        + float(_m1(cfg, "CEM_ORACLE_WEIGHT", 1.0)) * oracle_margin_loss
        + float(_m1(cfg, "CEM_EDIT_BUDGET_WEIGHT", 0.10)) * edit_budget_loss
        + float(_m1(cfg, "CEM_CANDIDATE_HARM_WEIGHT", 0.25))
        * candidate_harm_loss
        + float(_m1(cfg, "CEM_HARM_FRACTION_WEIGHT", 0.0))
        * harm_fraction_penalty
        + float(_m1(cfg, "CEM_FP_EDIT_WEIGHT", 0.0))
        * candidate_fp_edit_loss
        + float(_m1(cfg, "CEM_TP_REMOVAL_WEIGHT", 0.0))
        * candidate_tp_removal_loss
        # V476.1: candidate-quality supervision must be active before
        # deployment starts. Previously every quality loss was multiplied by
        # selector_scale, so utility/benefit/harm received no gradient before
        # CEM_SELECTOR_START_EPOCH. When the ramp began, the majority Preserve
        # target pushed all action scores negative.
        + float(
            _m1(
                cfg,
                "CEM_UTILITY_DISTRIBUTION_WEIGHT",
                _m1(cfg, "CEM_UTILITY_NLL_WEIGHT", _m1(cfg, "CEM_UTILITY_REG_WEIGHT", 0.50)),
            )
        )
        * utility_regression_loss
        + float(_m1(cfg, "CEM_RANK_WEIGHT", 0.35)) * ranking_loss
        + float(_m1(cfg, "CEM_LCB_CALIBRATION_WEIGHT", 0.50))
        * lcb_calibration_loss
        + float(_m1(cfg, "CEM_CF_PAIRWISE_WEIGHT", 0.0))
        * causal_pairwise_loss
        + float(_m1(cfg, "CEM_CF_NULL_WEIGHT", 0.0))
        * causal_null_loss
        + float(_m1(cfg, "CEM_DISCOVERY_COHERENCE_WEIGHT", 0.0))
        * discovery_coherence_loss
        + float(_m1(cfg, "CEM_DISCOVERY_DIVERSITY_WEIGHT", 0.0))
        * discovery_diversity_loss
        + float(_m1(cfg, "CEM_DISCOVERY_ORACLE_WEIGHT", 0.0))
        * discovery_oracle_loss
        + selector_scale
        * float(_m1(cfg, "CEM_M3_POLICY_REGRET_WEIGHT", 0.0))
        * policy_regret_loss
        + selector_scale
        * float(_m1(cfg, "CEM_FINAL_TAIL_WEIGHT", 0.0))
        * final_tail_loss
        # Legacy probability losses are optional diagnostics.  The default
        # V477 selector is driven only by relative mean, uncertainty and LCB.
        + float(_m1(cfg, "CEM_BENEFIT_WEIGHT", 0.0))
        * benefit_classification_loss
        + float(_m1(cfg, "CEM_HARM_REG_WEIGHT", 0.0))
        * harm_regression_loss
        + float(_m1(cfg, "CEM_ACCEPT_WEIGHT", 0.0))
        * candidate_accept_loss
        + float(_m1(cfg, "CEM_FAILURE_WEIGHT", 0.30)) * failure_loss
        + selector_scale
        * (
            float(_m1(cfg, "CEM_SELECTION_WEIGHT", 0.60))
            * selection_loss
            + float(_m1(cfg, "CEM_SELECTED_SEGMENTATION_WEIGHT", 0.75))
            * selected_segmentation_loss
            + float(_m1(cfg, "CEM_EXPECTED_HARM_WEIGHT", 0.20))
            * expected_harm_loss
            + float(_m1(cfg, "CEM_EXPECTED_NEGATIVE_GAIN_WEIGHT", 0.0))
            * expected_negative_gain_loss
        )
    )

    hard_dice = _hard_dice_probs(candidate_probs, gt)
    base_hard = hard_dice[:, 0]
    best_hard, best_index = hard_dice.max(dim=1)
    selected_hard = _hard_dice_probs(aux["cem_hard_selected_probs"], gt)
    candidate_gain = hard_dice[:, 1:] - base_hard[:, None]
    positive_margin = float(_m1(cfg, "CEM_POSITIVE_GAIN_MARGIN", 0.0005))
    harm_margin = float(_m1(cfg, "CEM_HARM_GAIN_MARGIN", 0.0005))

    error_pixels = error_union.flatten(1).sum(dim=1)
    predicted_error = (predicted_union >= 0.5).to(gt.dtype)
    true_positive = (predicted_error * error_union).flatten(1).sum(dim=1)
    predicted_positive = predicted_error.flatten(1).sum(dim=1)
    coverage_precision = true_positive / predicted_positive.clamp_min(1.0)
    coverage_recall = true_positive / error_pixels.clamp_min(1.0)

    target_utility_centered = relative_utility.detach() - relative_utility.detach().mean(dim=1, keepdim=True)
    pred_utility_centered = predicted_utility.detach() - predicted_utility.detach().mean(dim=1, keepdim=True)
    utility_corr = (
        (target_utility_centered * pred_utility_centered).sum(dim=1)
        / (
            target_utility_centered.square().sum(dim=1).sqrt()
            * pred_utility_centered.square().sum(dim=1).sqrt()
            + EPS
        )
    ).mean()
    predicted_benefit = (
        torch.sigmoid(predicted_benefit_logits[:, 1:]) >= 0.5
    ).to(candidates.dtype)
    benefit_tp = (predicted_benefit * benefit_target).sum()
    benefit_precision = benefit_tp / predicted_benefit.sum().clamp_min(1.0)
    benefit_recall = benefit_tp / benefit_target.sum().clamp_min(1.0)
    benefit_accuracy = (predicted_benefit == benefit_target).float().mean()
    deployable_prediction = (
        predicted_lcb
        > float(_m1(cfg, "CEM_SELECTOR_UTILITY_MARGIN", 0.001))
    )

    diagnostics: Dict[str, torch.Tensor] = {
        # Legacy names retained for the existing logger/audit scripts.
        "cem_v479_compact_local_loss": compact_local_loss.detach(),
        "cem_v479_gate_loss": gate_loss.detach(),
        "cem_v479_leakage_loss": leakage_loss.detach(),
        "cem_cf_mean_loss": cf_mean_loss.detach(),
        "cem_cf_quantile_loss": cf_quantile_loss.detach(),
        "cem_cf_q10_coverage": cf_q10_coverage.detach(),
        "tpmhg_loss": total.detach(),
        "tpmhg_oracle_loss": oracle_margin_loss.detach(),
        "tpmhg_bce_loss": candidate_validity_loss.detach(),
        "tpmhg_role_loss": (support_bce + signed_field_loss).detach(),
        "tpmhg_diversity_loss": redundancy_loss.detach(),
        "tpmhg_area_loss": edit_budget_loss.detach(),
        "tpmhg_base_dice": base_hard.mean().detach(),
        "tpmhg_best_dice": best_hard.mean().detach(),
        "tpmhg_oracle_gain": (best_hard - base_hard).mean().detach(),
        "tpmhg_pairwise_l1": aux["cem_pairwise_l1_forward"].mean().detach(),
        "tpmhg_positive_candidate_rate": (
            candidate_gain > positive_margin
        ).float().mean().detach(),
        "tpmhg_harmful_candidate_rate": (
            candidate_gain < -harm_margin
        ).float().mean().detach(),
        "tpmhg_positive_case_rate": (
            candidate_gain > positive_margin
        ).any(dim=1).float().mean().detach(),
        "tpmhg_best_slot_mean": best_index.float().mean().detach(),
        "tpmhg_hyp_area": candidate_probs[:, 1:].mean().detach(),

        # CEM-specific diagnostics.
        "cem_loss": total.detach(),
        "cem_atom_support_bce": support_bce.detach(),
        "cem_atom_support_dice_loss": support_dice.detach(),
        "cem_atom_signed_loss": signed_field_loss.detach(),
        "cem_atom_sign_loss": sign_loss.detach(),
        "cem_atom_outside_loss": outside_sparsity.detach(),
        "cem_support_precision_loss": support_precision_loss.detach(),
        "cem_support_precision": support_precision[target_present].mean().detach()
        if bool(target_present.any()) else support_precision.new_zeros(()),
        "cem_correction_precision_loss": correction_precision_loss.detach(),
        "cem_correction_outside_ratio": correction_outside_ratio[target_present].mean().detach()
        if bool(target_present.any()) else correction_outside_ratio.new_zeros(()),
        "cem_atom_noop_loss": no_op_loss.detach(),
        "cem_component_count": component_count.mean().detach(),
        "cem_error_coverage_loss": coverage_loss.detach(),
        "cem_component_union_fraction": component_union.mean().detach(),
        "cem_full_error_fraction": error_union.mean().detach(),
        "cem_error_coverage_precision": coverage_precision.mean().detach(),
        "cem_error_coverage_recall": coverage_recall.mean().detach(),
        "cem_redundancy_loss": redundancy_loss.detach(),
        "cem_candidate_validity_loss": candidate_validity_loss.detach(),
        "cem_soft_oracle_gain": soft_oracle_gain.mean().detach(),
        "cem_hard_oracle_gain": (best_hard - base_hard).mean().detach(),
        "cem_single_best_gain": (
            hard_dice[:, 1 : 1 + num_modes].max(dim=1).values - base_hard
        ).mean().detach(),
        "cem_composed_best_gain": (
            hard_dice[:, 1 + num_modes :].max(dim=1).values - base_hard
        ).mean().detach()
        if hard_dice.shape[1] > 1 + num_modes
        else hard_dice.new_zeros(()),
        "cem_discovery_best_gain": (
            hard_dice[:, 1 + num_modes :].max(dim=1).values - base_hard
        ).mean().detach()
        if hard_dice.shape[1] > 1 + num_modes
        else hard_dice.new_zeros(()),
        "cem_candidate_harm_fraction": harm_fraction[:, 1:].mean().detach(),
        "cem_harm_fraction_penalty": harm_fraction_penalty.detach(),
        "cem_candidate_fp_edit_mass": false_positive_edit_mass[:, 1:].mean().detach(),
        "cem_candidate_tp_removal_mass": true_positive_removal_mass[:, 1:].mean().detach(),
        "cem_candidate_beneficial_mass": beneficial_mass[:, 1:].mean().detach(),
        "cem_candidate_edit_mass": edit_mass[:, 1:].mean().detach(),
        "cem_utility_reg_loss": utility_regression_loss.detach(),
        "cem_utility_mean_loss": utility_mean_loss.detach(),
        "cem_sigma_calibration_loss": sigma_calibration_loss.detach(),
        "cem_lcb_calibration_loss": lcb_calibration_loss.detach(),
        "cem_predicted_sigma_mean": predicted_sigma[:, 1:].mean().detach(),
        "cem_predicted_lcb_mean": predicted_lcb.mean().detach(),
        "cem_positive_lcb_pass_rate": (
            ((predicted_lcb > positive_gain_margin).to(candidates.dtype)
             * benefit_target).sum()
            / benefit_target.sum().clamp_min(1.0)
        ).detach(),
        "cem_harmful_lcb_reject_rate": (
            ((predicted_lcb <= 0).to(candidates.dtype)
             * harmful_target).sum()
            / harmful_target.sum().clamp_min(1.0)
        ).detach(),
        "cem_benefit_loss": benefit_classification_loss.detach(),
        "cem_benefit_target_rate": benefit_target.mean().detach(),
        "cem_benefit_predicted_rate": predicted_benefit.mean().detach(),
        "cem_benefit_precision": benefit_precision.detach(),
        "cem_benefit_recall": benefit_recall.detach(),
        "cem_benefit_accuracy": benefit_accuracy.detach(),
        "cem_deployable_candidate_rate": deployable_prediction.float().mean().detach(),
        "cem_best_action_score": aux["cem_best_action_score"].mean().detach(),
        "cem_best_action_harm": aux["cem_best_action_harm"].mean().detach(),
        "cem_best_action_benefit": aux["cem_best_action_benefit"].mean().detach(),
        "cem_harm_reg_loss": harm_regression_loss.detach(),
        "cem_candidate_accept_loss": candidate_accept_loss.detach(),
        "cem_expected_negative_gain_loss": expected_negative_gain_loss.detach(),
        "cem_rank_loss": ranking_loss.detach(),
        "cem_cf_pairwise_loss": causal_pairwise_loss.detach(),
        "cem_cf_pairwise_accuracy": causal_pairwise_accuracy.detach(),
        "cem_cf_null_loss": causal_null_loss.detach(),
        "cem_m3_policy_regret_loss": policy_regret_loss.detach(),
        "cem_m3_expected_utility": policy_expected_utility.mean().detach(),
        "cem_m3_oracle_utility": policy_oracle_utility.mean().detach(),
        "cem_final_tail_loss": final_tail_loss.detach(),
        "cem_discovery_coherence_loss": discovery_coherence_loss.detach(),
        "cem_discovery_diversity_loss": discovery_diversity_loss.detach(),
        "cem_discovery_oracle_loss": discovery_oracle_loss.detach(),
        "cem_discovery_oracle_quality": discovery_oracle_quality.mean().detach(),
        "cem_failure_loss": failure_loss.detach(),
        "cem_failure_target_rate": failure_target.mean().detach(),
        "cem_selection_loss": selection_loss.detach(),
        "cem_selector_scale": candidates.new_tensor(selector_scale),
        "cem_selected_segmentation_loss": selected_segmentation_loss.detach(),
        "cem_selected_dice": selected_hard.mean().detach(),
        "cem_selected_delta_dice": (selected_hard - base_hard).mean().detach(),
        "cem_changed_rate": aux["cem_accept"].mean().detach(),
        "cem_quality_correlation": utility_corr.detach(),
        "cem_candidate_count": candidates.new_tensor(float(candidates.shape[1])),
    }
    return total, diagnostics


# ---------------------------------------------------------------------------
# V481 wrapper: keep the original CEM objective, add residual-repair precision
# and no-harm supervision.
# ---------------------------------------------------------------------------
def compute_text_prompted_hypothesis_loss(
    cfg: Any,
    candidates: torch.Tensor,
    masks: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    epoch=None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    loss_candidates = candidates
    loss_aux = aux

    # V482: detach the Preserve/C0 slot inside proposal losses.
    # This does not freeze Base. Base still receives gradients from the
    # main segmentation loss. It only prevents the M1 proposal objective
    # from optimizing C0 directly and damaging the base segmenter.
    if bool(_m1(cfg, "CEM_V482_DETACH_C0_IN_PROPOSAL_LOSS", True)):
        loss_candidates = candidates.clone()
        loss_candidates[:, :1] = loss_candidates[:, :1].detach()
        loss_aux = dict(aux)
        if isinstance(aux.get("candidate_probs"), torch.Tensor):
            cp = aux["candidate_probs"].clone()
            cp[:, :1] = cp[:, :1].detach()
            loss_aux["candidate_probs"] = cp
        if isinstance(aux.get("tpmhg_hypothesis_probs"), torch.Tensor):
            loss_aux["tpmhg_hypothesis_probs"] = aux["tpmhg_hypothesis_probs"]
        if isinstance(aux.get("tpmhg_hypothesis_logits"), torch.Tensor):
            loss_aux["tpmhg_hypothesis_logits"] = aux["tpmhg_hypothesis_logits"]

    base_loss, diagnostics = _compute_text_prompted_hypothesis_loss_v480(
        cfg,
        loss_candidates,
        masks,
        loss_aux,
        epoch=epoch,
    )

    if bool(_m1(cfg, "CEM_V481_ENABLED", False)):
        from utils.v481_m1_residual_repair import (
            compute_v481_m1_precision_loss,
        )

        v481_loss, v481_diag = compute_v481_m1_precision_loss(
            cfg,
            loss_candidates,
            masks,
            loss_aux,
        )

        weight = float(_m1(cfg, "CEM_V481_LOSS_WEIGHT", 0.35))
        loss = base_loss + weight * v481_loss
        diagnostics.update(v481_diag)
        diagnostics["cem_v481_weight"] = candidates.new_tensor(weight).detach()
        diagnostics["cem_v481_total_loss"] = loss.detach()
        return loss, diagnostics

    return base_loss, diagnostics
