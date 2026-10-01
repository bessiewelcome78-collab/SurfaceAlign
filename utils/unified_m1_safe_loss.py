# -*- coding: utf-8 -*-
"""V420 unified M1-only safe-fusion objective.

This module trains one complete candidate generator rather than a legacy
A1--A8 bank followed by separately appended C9/C10 branches.

The generator is image/Base driven and exports:
  C0 Preserve,
  C1--C2 sparse-island delete,
  C3--C4 inner-boundary trim,
  C5--C6 connected outer-boundary fill,
  C7--C8 interior-hole fill.

M2 is intentionally not used here.  M1 is accepted only if its own hard
one-action-or-Preserve output improves both validation DSC and NSD under the
pre-registered audit.  Ground truth occurs only inside this training loss.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

EPS = 1.0e-6


def _m1(cfg, key: str, default):
    node = getattr(cfg, "M1", None)
    if node is None:
        return default
    return node.get(key, default) if isinstance(node, dict) else getattr(node, key, default)


def _foreground(mask: torch.Tensor) -> torch.Tensor:
    x = mask.float()
    if x.ndim == 4:
        if x.shape[1] == 1:
            x = x[:, 0]
        elif x.shape[1] == 2:
            x = x[:, 1]
        else:
            raise ValueError(f"Cannot infer foreground from {tuple(mask.shape)}")
    if x.ndim != 3:
        raise ValueError(f"Expected mask [B,H,W] or [B,1,H,W], got {tuple(mask.shape)}")
    return (x > 0.5).float()


def _inner_boundary(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask[:, None]
    eroded = -F.max_pool2d(-mask.float(), kernel_size=3, stride=1, padding=1)
    return (mask.float() - eroded).clamp(0.0, 1.0)


@torch.no_grad()
def _hard_dice(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Hard Dice; logits [B,S,H,W] -> [B,S]."""
    if logits.ndim == 3:
        logits = logits[:, None]
    gt = _foreground(target)[:, None]
    hard = (torch.sigmoid(logits) >= 0.5).float()
    inter = (hard * gt).sum(dim=(-2, -1))
    denom = hard.sum(dim=(-2, -1)) + gt.sum(dim=(-2, -1))
    return (2.0 * inter + EPS) / (denom + EPS)


@torch.no_grad()
def _hard_nsd(logits: torch.Tensor, target: torch.Tensor, tolerance: int) -> torch.Tensor:
    """Train-resolution surface-Dice proxy with the project's inner-boundary rule."""
    if logits.ndim == 3:
        logits = logits[:, None]
    gt = _foreground(target)[:, None]
    hard = (torch.sigmoid(logits) >= 0.5).float()
    gt = gt.expand_as(hard)
    b, s, h, w = hard.shape
    x = hard.reshape(b * s, 1, h, w)
    y = gt.reshape(b * s, 1, h, w)
    bx, by = _inner_boundary(x), _inner_boundary(y)
    r = max(0, int(tolerance))
    near_x = F.max_pool2d(bx, 2 * r + 1, 1, r)
    near_y = F.max_pool2d(by, 2 * r + 1, 1, r)
    close_x = (bx * near_y).sum(dim=(1, 2, 3))
    close_y = (by * near_x).sum(dim=(1, 2, 3))
    denom = bx.sum(dim=(1, 2, 3)) + by.sum(dim=(1, 2, 3))
    score = (close_x + close_y) / denom.clamp_min(EPS)
    area_x = x.sum(dim=(1, 2, 3))
    area_y = y.sum(dim=(1, 2, 3))
    score = torch.where((area_x == 0) & (area_y == 0), torch.ones_like(score), score)
    score = torch.where((area_x == 0) ^ (area_y == 0), torch.zeros_like(score), score)
    return score.reshape(b, s)


def _soft_dice(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prob.ndim == 3:
        prob = prob[:, None]
    gt = _foreground(target)[:, None].to(prob.dtype)
    inter = (prob * gt).sum(dim=(-2, -1))
    denom = prob.sum(dim=(-2, -1)) + gt.sum(dim=(-2, -1))
    return (2.0 * inter + EPS) / (denom + EPS)


def _soft_boundary_dice(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prob.ndim == 3:
        prob = prob[:, None]
    gt = _foreground(target)[:, None].to(prob.dtype)
    return _soft_dice(_inner_boundary(prob), _inner_boundary(gt))


def _balanced_bce(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
                  pos_min: float = 1.0, pos_max: float = 25.0) -> torch.Tensor:
    valid = valid.float()
    if valid.sum() <= 0:
        return logits.sum() * 0.0
    x = logits[valid > 0.5]
    y = target.float()[valid > 0.5]
    pos = y.sum()
    neg = y.numel() - pos
    pos_weight = (neg / pos.clamp_min(1.0)).clamp(float(pos_min), float(pos_max))
    return F.binary_cross_entropy_with_logits(x, y, pos_weight=pos_weight)


def _pairwise_value_loss(score: torch.Tensor, positive: torch.Tensor,
                         nonpositive: torch.Tensor, valid: torch.Tensor,
                         margin: float) -> torch.Tensor:
    terms = []
    for i in range(score.shape[0]):
        p = score[i][positive[i] & valid[i]]
        n = score[i][nonpositive[i] & valid[i]]
        if p.numel() and n.numel():
            terms.append(F.softplus(float(margin) - (p[:, None] - n[None, :])).mean())
    return torch.stack(terms).mean() if terms else score.sum() * 0.0


def compute_unified_m1_safe_fusion_loss(
    cfg,
    candidates: torch.Tensor,
    masks: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    epoch=None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Train the unified M1 candidate bank and its M1-only safe output.

    The deployed hard output selects exactly one typed local candidate or
    Preserve.  The differentiable soft mixture is a training surrogate only.
    """
    required = (
        "v20_action_supports",
        "v20_control_supports",
        "v20_type_supports",
        "v20_action_types",
        "v20_actionness_logits",
        "v20_delta_maps",
        "m1_value_delta",
        "m1_benefit_logits",
        "m1_harm_logits",
        "m1_score",
        "m1_choice_logits",
        "m1_selector_soft",
        "m1_soft_fused_logits",
        "m1_hard_fused_logits",
        "m1_valid_action",
        "m1_local_area",
    )
    missing = [key for key in required if key not in aux]
    if missing:
        raise KeyError(f"Unified M1 auxiliary outputs missing: {missing}")
    if candidates.ndim != 4 or candidates.shape[1] < 2:
        raise ValueError(f"Expected [B,1+K,H,W] candidate logits, got {tuple(candidates.shape)}")

    m1 = cfg.M1
    base = candidates[:, :1]
    actions = candidates[:, 1:]
    b, k, h, w = actions.shape
    gt = _foreground(masks).to(candidates.dtype)

    supports = aux["v20_action_supports"].float()
    controls = aux["v20_control_supports"].float()
    type_supports = aux["v20_type_supports"].float()
    action_types = aux["v20_action_types"].long()
    actionness = aux["v20_actionness_logits"]
    delta_maps = aux["v20_delta_maps"].float()
    if supports.shape != actions.shape or controls.shape != actions.shape or actionness.shape != actions.shape or delta_maps.shape != actions.shape:
        raise ValueError(
            "Unified M1 action/support/actionness shape mismatch: "
            f"actions={tuple(actions.shape)}, supports={tuple(supports.shape)}, "
            f"controls={tuple(controls.shape)}, delta_maps={tuple(delta_maps.shape)}, "
            f"actionness={tuple(actionness.shape)}"
        )

    dsc_eps = max(float(_m1(cfg, "UNIFIED_M1_DSC_EPS", 0.001)), 0.0)
    nsd_eps = max(float(_m1(cfg, "UNIFIED_M1_NSD_EPS", 0.001)), 0.0)
    tolerance = int(_m1(cfg, "UNIFIED_M1_NSD_TOLERANCE", 2))
    max_edit = max(float(_m1(cfg, "UNIFIED_M1_MAX_EDIT_FRACTION", 0.035)), 1.0e-4)
    utility_dsc_weight = float(_m1(cfg, "UNIFIED_M1_DSC_UTILITY_WEIGHT", 0.60))
    utility_nsd_weight = float(_m1(cfg, "UNIFIED_M1_NSD_UTILITY_WEIGHT", 0.40))
    utility_sum = max(utility_dsc_weight + utility_nsd_weight, EPS)
    utility_dsc_weight /= utility_sum
    utility_nsd_weight /= utility_sum

    with torch.no_grad():
        all_logits = torch.cat([base, actions], dim=1)
        dsc_all = _hard_dice(all_logits, gt)
        nsd_all = _hard_nsd(all_logits, gt, tolerance)
        dd = dsc_all[:, 1:] - dsc_all[:, :1]
        dn = nsd_all[:, 1:] - nsd_all[:, :1]

        is_delete = (action_types.to(base.device).long()[None, :, None, None] <= 1)
        sign = torch.where(is_delete, -torch.ones_like(actions), torch.ones_like(actions))
        control_actions = base + sign * controls * delta_maps
        control_all_logits = torch.cat([base, control_actions], dim=1)
        control_dsc_all = _hard_dice(control_all_logits, gt)
        control_nsd_all = _hard_nsd(control_all_logits, gt, tolerance)
        cdd = control_dsc_all[:, 1:] - control_dsc_all[:, :1]
        cdn = control_nsd_all[:, 1:] - control_nsd_all[:, :1]

        base_hard = (torch.sigmoid(base) >= 0.5)
        action_hard = (torch.sigmoid(actions) >= 0.5)
        changed = (action_hard != base_hard).float().mean(dim=(-2, -1))
        support_nonempty = supports.sum(dim=(-2, -1)) > 0.5
        candidate_valid = support_nonempty & (changed > 0.0) & (changed <= max_edit)

        factual_area = supports.sum(dim=(-2, -1))
        control_area = controls.sum(dim=(-2, -1))
        area_ratio = control_area / factual_area.clamp_min(EPS)
        control_overlap = (supports * controls).sum(dim=(-2, -1)) / factual_area.clamp_min(EPS)
        computed_pair_valid = (
            candidate_valid
            & (control_area > 0.5)
            & ((area_ratio - 1.0).abs() <= float(_m1(cfg, "UNIFIED_M1_CONTROL_AREA_TOLERANCE", 0.15)))
            & (control_overlap <= float(_m1(cfg, "UNIFIED_M1_CONTROL_OVERLAP_TOLERANCE", 1.0e-6)))
        )
        pair_valid_aux = aux.get("v457_train_pair_valid", aux.get("v25_valid_control"))
        if isinstance(pair_valid_aux, torch.Tensor) and pair_valid_aux.shape == computed_pair_valid.shape:
            pair_valid = candidate_valid & pair_valid_aux.bool()
        else:
            pair_valid = computed_pair_valid

        target_clip = max(float(_m1(cfg, "UNIFIED_M1_TARGET_CLAMP", 0.05)), max(dsc_eps, nsd_eps) * 2.0)
        dd = torch.nan_to_num(dd, nan=0.0, posinf=0.0, neginf=0.0).clamp(-target_clip, target_clip)
        dn = torch.nan_to_num(dn, nan=0.0, posinf=0.0, neginf=0.0).clamp(-target_clip, target_clip)
        cdd = torch.nan_to_num(cdd, nan=0.0, posinf=0.0, neginf=0.0).clamp(-target_clip, target_clip)
        cdn = torch.nan_to_num(cdn, nan=0.0, posinf=0.0, neginf=0.0).clamp(-target_clip, target_clip)

        utility = utility_dsc_weight * dd + utility_nsd_weight * dn
        control_utility = utility_dsc_weight * cdd + utility_nsd_weight * cdn
        paired_utility = (utility - control_utility).clamp(-target_clip, target_clip)
        paired_dsc = (dd - cdd).clamp(-target_clip, target_clip)
        paired_nsd = (dn - cdn).clamp(-target_clip, target_clip)
        pair_eps = max(dsc_eps, nsd_eps)

        # V459 root fix: do not use exact-control availability as the only
        # source of selector supervision.  Exact controls are a stronger
        # relative-quality target when available, but BUSI ActionBank can have
        # sparse pair-valid controls early in training.  If pair_valid is used
        # as the sole valid mask, almost every action becomes Preserve and the
        # candidate bank loses its oracle ceiling.  We therefore train two
        # aligned targets:
        #   factual target: candidate improves Base without obvious DSC/NSD harm;
        #   paired target : factual improvement is better than matched control.
        # The selector learns paired residuals where pair_valid exists and falls
        # back to factual utility only for candidate-valid actions without a
        # usable exact control.  This is not freezing and not a hand threshold;
        # it removes the label-starvation bug while preserving paired-control
        # evidence when it is actually valid.
        utility_eps = max(dsc_eps, nsd_eps)
        factual_no_harm = (dd >= -dsc_eps) & (dn >= -nsd_eps)
        factual_positive = candidate_valid & factual_no_harm & (utility > utility_eps)
        factual_harmful = candidate_valid & (
            (dd < -dsc_eps) | (dn < -nsd_eps) | (utility < -utility_eps)
        )
        pair_positive = pair_valid & factual_no_harm & (utility > 0.0) & (paired_utility > pair_eps)
        pair_harmful = pair_valid & (
            (dd < -dsc_eps) | (dn < -nsd_eps) | (paired_utility < -pair_eps)
        )

        require_control = bool(_m1(cfg, "UNIFIED_M1_REQUIRE_CONTROL_FOR_SELECTOR", False))
        if bool(_m1(cfg, "UNIFIED_M1_PAIRED_CONTROL_SELECTOR", True)) and require_control:
            valid = pair_valid
            positive = pair_positive
            harmful = pair_harmful
            selector_utility = paired_utility
        elif bool(_m1(cfg, "UNIFIED_M1_PAIRED_CONTROL_SELECTOR", True)):
            valid = candidate_valid
            positive = pair_positive | ((~pair_valid) & factual_positive)
            harmful = pair_harmful | ((~pair_valid) & factual_harmful)
            selector_utility = torch.where(pair_valid, paired_utility, utility)
        else:
            valid = candidate_valid
            positive = factual_positive
            harmful = factual_harmful
            selector_utility = utility

        deploy_valid = valid
        neutral = valid & ~(positive | harmful)
        deploy_positive = deploy_valid & positive
        deploy_harmful = deploy_valid & harmful
        deploy_neutral = deploy_valid & ~(deploy_positive | deploy_harmful)

        best_utility, best_index = selector_utility.masked_fill(~positive, -1e4).max(dim=1)
        choice_target = torch.where(
            best_utility > -1e3,
            best_index + 1,
            torch.zeros_like(best_index),
        )

    # Dense M1 proposal supervision. The type carrier and residual target are
    # built from B0-vs-GT errors; no M2 score is used.
    type_targets = []
    type_carriers = []
    base_hard = (base[:, 0] >= 0.0).float()
    for type_id in action_types.tolist():
        carrier = type_supports[:, int(type_id)]
        if int(type_id) in (0, 1):
            target = base_hard * (1.0 - gt) * carrier
        else:
            target = (1.0 - base_hard) * gt * carrier
        type_targets.append(target)
        type_carriers.append(carrier)
    dense_target = torch.stack(type_targets, dim=1)
    dense_carrier = torch.stack(type_carriers, dim=1)

    proposal_loss = _balanced_bce(
        actionness,
        dense_target,
        dense_carrier,
        pos_min=float(_m1(cfg, "UNIFIED_M1_PROPOSAL_POS_WEIGHT_MIN", 2.0)),
        pos_max=float(_m1(cfg, "UNIFIED_M1_PROPOSAL_POS_WEIGHT_MAX", 25.0)),
    )

    target_expanded = gt[:, None].expand_as(actions)
    repair_raw = F.binary_cross_entropy_with_logits(actions, target_expanded, reduction="none")
    repair_loss = (repair_raw * supports).sum() / supports.sum().clamp_min(1.0)

    # Explicit anti-collapse term: every train-time residual component must
    # receive proposal mass from at least one slot of its corresponding type.
    proposal_prob = torch.sigmoid(actionness)
    cover_num = (proposal_prob * dense_target).sum(dim=(-2, -1))
    cover_den = dense_target.sum(dim=(-2, -1))
    cover_valid = cover_den > 0.5
    coverage_loss = (
        (1.0 - cover_num[cover_valid] / cover_den[cover_valid].clamp_min(1.0)).mean()
        if cover_valid.any() else proposal_loss * 0.0
    )

    value = aux["m1_value_delta"]
    if value.shape != (b, k, 2):
        raise ValueError(f"m1_value_delta expected {(b,k,2)}, got {tuple(value.shape)}")
    benefit_logits = aux["m1_benefit_logits"]
    harm_logits = aux["m1_harm_logits"]
    score = aux["m1_score"]
    if any(x.shape != (b, k) for x in (benefit_logits, harm_logits, score)):
        raise ValueError("Unified M1 value-head shape mismatch.")

    factual_value_target = torch.stack([dd, dn], dim=-1)
    paired_value_target = torch.stack([paired_dsc, paired_nsd], dim=-1)
    if bool(_m1(cfg, "UNIFIED_M1_PAIRED_CONTROL_SELECTOR", True)):
        value_target = torch.where(pair_valid[..., None], paired_value_target, factual_value_target)
    else:
        value_target = factual_value_target
    value_loss = (
        F.smooth_l1_loss(
            value[valid],
            value_target[valid],
            beta=float(_m1(cfg, "UNIFIED_M1_VALUE_HUBER_BETA", 0.01)),
        ) if valid.any() else value.sum() * 0.0
    )
    benefit_loss = _balanced_bce(
        benefit_logits,
        positive.float(),
        valid,
        pos_min=1.0,
        pos_max=float(_m1(cfg, "UNIFIED_M1_VALUE_POS_WEIGHT_MAX", 20.0)),
    )
    harm_loss = _balanced_bce(
        harm_logits,
        harmful.float(),
        valid,
        pos_min=1.0,
        pos_max=float(_m1(cfg, "UNIFIED_M1_VALUE_POS_WEIGHT_MAX", 20.0)),
    )

    # V456: the loss uses un-gated train logits when available; deployment
    # logits may be adaptively masked at inference.  This decouples learning
    # relative quality from the abstention gate.
    choice_logits = aux.get("m1_train_choice_logits", aux["m1_choice_logits"])
    if choice_logits.shape != (b, k + 1):
        raise ValueError(f"m1_choice_logits expected {(b,k+1)}, got {tuple(choice_logits.shape)}")

    # Dataset/batch adaptive class balancing.  No fixed BUSI-specific preserve
    # or action weights are needed: weights are derived from the current
    # oracle target distribution and normalised to mean one.
    preserve_freq = (choice_target == 0).float().mean().clamp_min(EPS)
    action_freq = (choice_target > 0).float().mean().clamp_min(EPS)
    class_weight = choice_logits.new_ones(k + 1)
    class_weight[0] = 0.5 / preserve_freq
    class_weight[1:] = 0.5 / action_freq
    class_weight = class_weight / class_weight.mean().clamp_min(EPS)
    choice_loss = F.cross_entropy(choice_logits, choice_target, weight=class_weight)

    choice_prob = torch.softmax(
        choice_logits / max(float(_m1(cfg, "UNIFIED_M1_LOSS_TEMPERATURE", 1.0)), EPS),
        dim=1,
    )
    has_positive = positive.any(dim=1)
    preserve_penalty = (
        (-torch.log((1.0 - choice_prob[:, 0]).clamp_min(EPS)))[has_positive].mean()
        if has_positive.any() else choice_logits.sum() * 0.0
    )

    # V456 adaptive loss rebalancing.  When harmful candidates dominate the
    # batch, automatically weaken anti-preserve pressure and strengthen
    # downside/harm terms; when positives dominate, allow more action pressure.
    positive_rate = positive.float().mean().detach()
    harmful_rate = harmful.float().mean().detach()
    safe_action_ratio = positive_rate / (positive_rate + harmful_rate + EPS)
    risk_ratio = (harmful_rate / (positive_rate + EPS)).clamp(
        max=float(_m1(cfg, "UNIFIED_M1_MAX_RISK_RATIO", 4.0))
    )

    positive_score_margin = float(_m1(cfg, "UNIFIED_M1_POSITIVE_SCORE_MARGIN", 0.10))
    harmful_score_margin = float(_m1(cfg, "UNIFIED_M1_HARMFUL_SCORE_MARGIN", 0.05))
    positive_score_loss = (
        F.softplus(positive_score_margin - score[positive]).mean()
        if positive.any() else score.sum() * 0.0
    )
    harmful_score_loss = (
        F.softplus(harmful_score_margin + score[harmful]).mean()
        if harmful.any() else score.sum() * 0.0
    )

    pairwise_loss = _pairwise_value_loss(
        score,
        positive,
        harmful | neutral,
        valid,
        margin=float(_m1(cfg, "UNIFIED_M1_PAIRWISE_MARGIN", 0.02)),
    )

    soft_choice = aux["m1_selector_soft"]
    soft_actions = soft_choice[:, 1:]
    expected_utility_loss = -(
        soft_actions * selector_utility.detach() * valid.float()
    ).sum(dim=1).mean()
    downside_loss = (
        soft_actions
        * F.relu(-selector_utility.detach() + max(dsc_eps, nsd_eps))
        * valid.float()
    ).sum(dim=1).mean()

    soft_fused_logits = aux["m1_soft_fused_logits"]
    hard_fused_logits = aux["m1_hard_fused_logits"]
    if soft_fused_logits.ndim != 3 or hard_fused_logits.ndim != 3:
        raise ValueError("Unified M1 fused logits must be [B,H,W].")
    soft_prob = torch.sigmoid(soft_fused_logits)
    base_prob = torch.sigmoid(base[:, 0])
    fused_bce = F.binary_cross_entropy_with_logits(soft_fused_logits, gt)
    fused_dice = 1.0 - _soft_dice(soft_prob, gt).mean()
    fused_boundary = 1.0 - _soft_boundary_dice(soft_prob, gt).mean()

    base_soft_dice = _soft_dice(base_prob, gt)
    fused_soft_dice = _soft_dice(soft_prob, gt)
    base_soft_nsd = _soft_boundary_dice(base_prob, gt)
    fused_soft_nsd = _soft_boundary_dice(soft_prob, gt)
    no_reg_dice = F.relu(
        base_soft_dice - fused_soft_dice + float(_m1(cfg, "UNIFIED_M1_SOFT_DICE_MARGIN", 0.0))
    ).mean()
    no_reg_nsd = F.relu(
        base_soft_nsd - fused_soft_nsd + float(_m1(cfg, "UNIFIED_M1_SOFT_NSD_MARGIN", 0.0))
    ).mean()

    local_area = aux["m1_local_area"]
    expected_area = (soft_actions * local_area).sum(dim=1)
    budget_loss = F.relu(expected_area - max_edit).mean()

    total = (
        float(_m1(cfg, "UNIFIED_M1_PROPOSAL_WEIGHT", 1.0)) * proposal_loss
        + float(_m1(cfg, "UNIFIED_M1_REPAIR_WEIGHT", 0.75)) * repair_loss
        + float(_m1(cfg, "UNIFIED_M1_COVERAGE_WEIGHT", 1.0)) * coverage_loss
        + float(_m1(cfg, "UNIFIED_M1_VALUE_WEIGHT", 2.0)) * value_loss
        + float(_m1(cfg, "UNIFIED_M1_BENEFIT_WEIGHT", 0.50)) * benefit_loss
        + float(_m1(cfg, "UNIFIED_M1_HARM_WEIGHT", 0.75)) * harm_loss
        + float(_m1(cfg, "UNIFIED_M1_CHOICE_WEIGHT", 2.0)) * choice_loss
        + float(_m1(cfg, "UNIFIED_M1_PRESERVE_PENALTY_WEIGHT", 1.0)) * safe_action_ratio * preserve_penalty
        + float(_m1(cfg, "UNIFIED_M1_POSITIVE_SCORE_WEIGHT", 1.0)) * positive_score_loss
        + float(_m1(cfg, "UNIFIED_M1_HARMFUL_SCORE_WEIGHT", 1.0)) * (1.0 + risk_ratio) * harmful_score_loss
        + float(_m1(cfg, "UNIFIED_M1_PAIRWISE_WEIGHT", 0.75)) * pairwise_loss
        + float(_m1(cfg, "UNIFIED_M1_EXPECTED_UTILITY_WEIGHT", 0.75)) * expected_utility_loss
        + float(_m1(cfg, "UNIFIED_M1_DOWNSIDE_WEIGHT", 1.50)) * (1.0 + risk_ratio) * downside_loss
        + float(_m1(cfg, "UNIFIED_M1_FUSED_BCE_WEIGHT", 0.50)) * fused_bce
        + float(_m1(cfg, "UNIFIED_M1_FUSED_DICE_WEIGHT", 1.00)) * fused_dice
        + float(_m1(cfg, "UNIFIED_M1_FUSED_BOUNDARY_WEIGHT", 1.00)) * fused_boundary
        + float(_m1(cfg, "UNIFIED_M1_NO_REG_DICE_WEIGHT", 1.00)) * no_reg_dice
        + float(_m1(cfg, "UNIFIED_M1_NO_REG_NSD_WEIGHT", 1.00)) * no_reg_nsd
        + float(_m1(cfg, "UNIFIED_M1_BUDGET_WEIGHT", 1.00)) * budget_loss
    )

    with torch.no_grad():
        hard_base_dsc = _hard_dice(base, gt)[:, 0]
        hard_base_nsd = _hard_nsd(base, gt, tolerance)[:, 0]
        hard_fused_dsc = _hard_dice(hard_fused_logits, gt)[:, 0]
        hard_fused_nsd = _hard_nsd(hard_fused_logits, gt, tolerance)[:, 0]

        # Preserve has index 0. torch.gather does not accept -1, and
        # torch.where evaluates both branches, so clamp before gathering.
        selected = choice_logits.argmax(dim=1)
        selected_action = (selected - 1).clamp(min=0, max=k - 1)

        selected_positive = torch.where(
            selected > 0,
            positive.gather(1, selected_action[:, None])[:, 0],
            torch.zeros_like(selected, dtype=torch.bool),
        )
        selected_harmful = torch.where(
            selected > 0,
            harmful.gather(1, selected_action[:, None])[:, 0],
            torch.zeros_like(selected, dtype=torch.bool),
        )

    return total, {
        "unified_m1_loss": total.detach(),
        "unified_m1_proposal_loss": proposal_loss.detach(),
        "unified_m1_repair_loss": repair_loss.detach(),
        "unified_m1_coverage_loss": coverage_loss.detach(),
        "unified_m1_value_loss": value_loss.detach(),
        "unified_m1_benefit_loss": benefit_loss.detach(),
        "unified_m1_harm_loss": harm_loss.detach(),
        "unified_m1_choice_loss": choice_loss.detach(),
        "unified_m1_preserve_penalty": preserve_penalty.detach(),
        "unified_m1_safe_action_ratio": safe_action_ratio.detach(),
        "unified_m1_risk_ratio": risk_ratio.detach(),
        "unified_m1_auto_preserve_weight": safe_action_ratio.detach(),
        "unified_m1_auto_risk_weight": (1.0 + risk_ratio).detach(),
        "unified_m1_positive_score_loss": positive_score_loss.detach(),
        "unified_m1_harmful_score_loss": harmful_score_loss.detach(),
        "unified_m1_pairwise_loss": pairwise_loss.detach(),
        "unified_m1_expected_utility_loss": expected_utility_loss.detach(),
        "unified_m1_downside_loss": downside_loss.detach(),
        "unified_m1_fused_bce": fused_bce.detach(),
        "unified_m1_fused_dice_loss": fused_dice.detach(),
        "unified_m1_fused_boundary_loss": fused_boundary.detach(),
        "unified_m1_no_reg_dice": no_reg_dice.detach(),
        "unified_m1_no_reg_nsd": no_reg_nsd.detach(),
        "unified_m1_budget_loss": budget_loss.detach(),
        "unified_m1_valid_action_rate": valid.float().mean().detach(),
        "unified_m1_candidate_valid_action_rate": candidate_valid.float().mean().detach(),
        "unified_m1_pair_valid_action_rate": pair_valid.float().mean().detach(),
        "unified_m1_factual_positive_action_rate": factual_positive.float().mean().detach(),
        "unified_m1_pair_positive_action_rate": pair_positive.float().mean().detach(),
        "unified_m1_factual_harmful_action_rate": factual_harmful.float().mean().detach(),
        "unified_m1_pair_harmful_action_rate": pair_harmful.float().mean().detach(),
        "unified_m1_deploy_valid_action_rate": deploy_valid.float().mean().detach(),
        "unified_m1_deploy_positive_action_rate": deploy_positive.float().mean().detach(),
        "unified_m1_deploy_harmful_action_rate": deploy_harmful.float().mean().detach(),
        "unified_m1_positive_action_rate": positive.float().mean().detach(),
        "unified_m1_harmful_action_rate": harmful.float().mean().detach(),
        "unified_m1_positive_case_rate": positive.any(dim=1).float().mean().detach(),
        "unified_m1_deploy_positive_case_rate": deploy_positive.any(dim=1).float().mean().detach(),
        "unified_m1_selector_utility_mean": selector_utility[valid].mean().detach() if valid.any() else selector_utility.sum().detach() * 0.0,
        "unified_m1_paired_utility_mean": paired_utility[pair_valid].mean().detach() if pair_valid.any() else paired_utility.sum().detach() * 0.0,
        "unified_m1_factual_utility_mean": utility[candidate_valid].mean().detach() if candidate_valid.any() else utility.sum().detach() * 0.0,
        "unified_m1_control_utility_mean": control_utility[pair_valid].mean().detach() if pair_valid.any() else control_utility.sum().detach() * 0.0,
        "unified_m1_choice_target_preserve_rate": (choice_target == 0).float().mean().detach(),
        "unified_m1_hard_preserve_rate": (choice_logits.argmax(dim=1) == 0).float().mean().detach(),
        "unified_m1_selected_positive_rate": selected_positive.float().mean().detach(),
        "unified_m1_selected_harmful_rate": selected_harmful.float().mean().detach(),
        "unified_m1_hard_delta_dice": (hard_fused_dsc - hard_base_dsc).mean().detach(),
        "unified_m1_hard_delta_nsd": (hard_fused_nsd - hard_base_nsd).mean().detach(),
        "unified_m1_hard_fusion_dice": hard_fused_dsc.mean().detach(),
        "unified_m1_hard_fusion_nsd": hard_fused_nsd.mean().detach(),
        "unified_m1_expected_area": expected_area.mean().detach(),
    }
