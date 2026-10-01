# /home/tsz-25/MedCLIPSeg-pristine/utils/candidate_consensus_loss.py-*- coding: utf-8 -*-
"""V38 train-only supervision for casewise M2 falsification + delta-consensus M3.

M1 remains a trainable all-type local candidate generator. M2 is supervised
with Train-only candidate outcomes, but its deployment input is still only
matched factual/control image-text evidence.  M3 has no learned gain/risk head:
it deterministically selects only M2-qualified hypotheses using local signed
edit-delta structural consistency.
"""
from __future__ import annotations
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

EPS = 1e-6
DELETE_TYPES = (0, 1)


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


def _hard_dice(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = (torch.sigmoid(logits) >= 0.5).float()
    tgt = (target > 0.5).float()
    if logits.ndim == 4:
        tgt = tgt[:, None]
        inter = (pred * tgt).flatten(2).sum(dim=-1)
        den = pred.flatten(2).sum(dim=-1) + tgt.flatten(2).sum(dim=-1)
    elif logits.ndim == 3:
        inter = (pred * tgt).flatten(1).sum(dim=-1)
        den = pred.flatten(1).sum(dim=-1) + tgt.flatten(1).sum(dim=-1)
    else:
        raise ValueError(f"Expected [B,H,W] or [B,K,H,W], got {tuple(logits.shape)}")
    return (2.0 * inter + EPS) / (den + EPS)


def _balanced_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    pos_min: float = 1.0,
    pos_max: float = 25.0,
) -> torch.Tensor:
    mask = mask.float()
    if mask.sum() <= 0:
        return logits.sum() * 0.0
    pos = (target * mask).sum()
    neg = ((1.0 - target) * mask).sum()
    pos_weight = (neg / pos.clamp_min(1.0)).clamp(pos_min, pos_max)
    raw = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction="none"
    )
    return (raw * mask).sum() / mask.sum().clamp_min(1.0)


def _masked_ce(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    valid = valid.float()
    if valid.sum() <= 0:
        return logits.sum() * 0.0
    classes = logits.shape[-1]
    raw = F.cross_entropy(
        logits.reshape(-1, classes),
        target.reshape(-1),
        reduction="none",
    ).reshape_as(valid)
    return (raw * valid).sum() / valid.sum().clamp_min(1.0)


def _geometry_targets(supports: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    b, _, h, w = supports.shape
    area = supports.mean(dim=(-2, -1))
    area_target = torch.bucketize(
        area.detach(), area.new_tensor([0.0003, 0.0010, 0.0040])
    )
    yy = torch.arange(h, device=supports.device, dtype=supports.dtype).view(1, 1, h, 1)
    xx = torch.arange(w, device=supports.device, dtype=supports.dtype).view(1, 1, 1, w)
    mass = supports.sum(dim=(-2, -1)).clamp_min(1.0)
    cy = (supports * yy).sum(dim=(-2, -1)) / mass
    cx = (supports * xx).sum(dim=(-2, -1)) / mass
    position_target = (
        (cy * 3.0).long().clamp(0, 2) * 3
        + (cx * 3.0).long().clamp(0, 2)
    )
    return area_target, position_target


def _dense_action_targets(
    base_logits: torch.Tensor,
    gt: torch.Tensor,
    type_supports: torch.Tensor,
    action_types: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    base_hard = (base_logits >= 0.0).float()
    targets, carriers = [], []
    for action_type in action_types.tolist():
        carrier = type_supports[:, int(action_type)]
        if int(action_type) in DELETE_TYPES:
            target = base_hard * (1.0 - gt) * carrier
        else:
            target = (1.0 - base_hard) * gt * carrier
        targets.append(target)
        carriers.append(carrier)
    return torch.stack(targets, dim=1), torch.stack(carriers, dim=1)


def _source_residual_loss(
    cfg: Any,
    base_logits: torch.Tensor,
    gt: torch.Tensor,
    aux: Dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    zero = base_logits.sum() * 0.0
    required = (
        "v35_residual_logit_map",
        "v20_actionness_logits",
        "v20_type_supports",
        "v20_action_types",
    )
    if any(key not in aux for key in required):
        return zero, zero.detach()
    residual = aux["v35_residual_logit_map"]
    actionness = aux["v20_actionness_logits"]
    type_supports = aux["v20_type_supports"].float()
    action_types = aux["v20_action_types"].to(base_logits.device)
    island_idx = torch.where(action_types == 0)[0]
    if island_idx.numel() == 0:
        return zero, zero.detach()
    base_hard = (torch.sigmoid(base_logits) >= 0.5).float()
    fp = base_hard * (1.0 - gt)
    carrier = type_supports[:, :1]
    source_target = fp[:, None] * carrier
    island_actionness = actionness.index_select(1, island_idx).mean(
        dim=1, keepdim=True
    )
    source_logits = residual + island_actionness
    carrier_mask = carrier > 0.5
    if not carrier_mask.any():
        return zero, zero.detach()
    target_c = source_target[carrier_mask]
    logits_c = source_logits[carrier_mask]
    gamma = max(float(_m1(cfg, "V35_SOURCE_FOCAL_GAMMA", 2.0)), 0.0)
    pos_weight = max(float(_m1(cfg, "V35_SOURCE_POS_WEIGHT", 6.0)), 1.0)
    bce = F.binary_cross_entropy_with_logits(logits_c, target_c, reduction="none")
    prob = torch.sigmoid(logits_c)
    pt = prob * target_c + (1.0 - prob) * (1.0 - target_c)
    alpha = torch.where(
        target_c > 0.5,
        torch.full_like(target_c, pos_weight),
        torch.ones_like(target_c),
    )
    focal = (alpha * (1.0 - pt).pow(gamma) * bce).mean()
    pred = torch.sigmoid(source_logits) * carrier
    inter = (pred * source_target).sum(dim=(-2, -1))
    den = pred.sum(dim=(-2, -1)) + source_target.sum(dim=(-2, -1))
    dice_loss = (1.0 - (2.0 * inter + EPS) / (den + EPS)).mean()
    union = ((pred >= 0.5).float() + source_target).clamp(max=1.0)
    source_iou = (
        ((pred >= 0.5).float() * source_target).sum(dim=(-2, -1))
        / union.sum(dim=(-2, -1)).clamp_min(1.0)
    ).mean()
    return focal + dice_loss, source_iou.detach()


def _casewise_pair_loss(
    scores: torch.Tensor,
    positive: torch.Tensor,
    harmful: torch.Tensor,
    valid: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train same-image positive > harmful. Returns loss, accuracy, case count."""
    terms, correct, count = [], scores.new_zeros(()), scores.new_zeros(())
    for batch_index in range(scores.shape[0]):
        pos = scores[batch_index][positive[batch_index] & valid[batch_index]]
        harm = scores[batch_index][harmful[batch_index] & valid[batch_index]]
        if pos.numel() == 0 or harm.numel() == 0:
            continue
        diff = pos[:, None] - harm[None, :]
        terms.append(F.softplus(float(margin) - diff).mean())
        correct = correct + (diff > 0.0).float().sum()
        count = count + torch.tensor(float(diff.numel()), device=scores.device)
    if not terms:
        return scores.sum() * 0.0, correct, count
    return torch.stack(terms).mean(), correct, count


def _preserve_aware_loss(
    scores: torch.Tensor,
    positive: torch.Tensor,
    harmful: torch.Tensor,
    valid: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Positive candidates must clear Preserve=0; harmful candidates must not."""
    terms = []
    pos_above, harmful_above = scores.new_zeros(()), scores.new_zeros(())
    for batch_index in range(scores.shape[0]):
        allowed = valid[batch_index]
        if not allowed.any():
            continue
        pos = scores[batch_index][positive[batch_index] & allowed]
        harm = scores[batch_index][harmful[batch_index] & allowed]
        if pos.numel() > 0:
            terms.append(F.softplus(float(margin) - pos.max()))
            pos_above = pos_above + (pos.max() > 0.0).float()
        else:
            # In no-positive images, every admissible candidate must stay below
            # the Preserve logit. This includes neutral actions conservatively.
            terms.append(F.softplus(float(margin) + scores[batch_index][allowed]).mean())
        if harm.numel() > 0:
            terms.append(F.softplus(float(margin) + harm).mean())
            harmful_above = harmful_above + (harm >= 0.0).float().mean()
    if not terms:
        return scores.sum() * 0.0, pos_above, harmful_above
    return torch.stack(terms).mean(), pos_above, harmful_above


def _combo_logits(
    base_logits: torch.Tensor,
    action_logits: torch.Tensor,
    pair_indices: torch.Tensor,
) -> torch.Tensor:
    if pair_indices.numel() == 0:
        return action_logits.new_zeros((action_logits.shape[0], 0, *action_logits.shape[-2:]))
    deltas = action_logits - base_logits[:, None]
    di = pair_indices[:, 0].long()
    fi = pair_indices[:, 1].long()
    return base_logits[:, None] + deltas.index_select(1, di) + deltas.index_select(1, fi)


def compute_candidate_consensus_loss(
    cfg: Any,
    candidate_logits: torch.Tensor,
    target_mask: torch.Tensor,
    aux: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    required = (
        "v20_action_supports", "v20_control_supports", "v20_type_supports",
        "v20_action_types", "v20_actionness_logits", "v20_fused_logits",
        "v20_selector_hard", "v38_atomic_cf_logit", "v38_atomic_cf_swap_logit",
        "v38_atomic_available", "v38_context_clean", "v38_atomic_qualified",
        "v38_pos_delta", "v38_neg_delta", "v38_swap_delta",
        "v38_type_probe", "v38_area_probe", "v38_position_probe",
        "v38_combo_pair_indices", "v38_combo_cf_logit", "v38_combo_available",
        "v38_combo_qualified", "v38_consensus_active", "v38_multi_active",
        "v38_singleton_active", "v38_cluster_size",
    )
    missing = [key for key in required if key not in aux]
    if missing:
        raise KeyError(f"V38 auxiliary outputs missing: {missing}")
    if candidate_logits.ndim != 4 or candidate_logits.shape[1] < 2:
        raise ValueError("V38 expects candidate logits [B,1+K,H,W].")

    gt = _foreground_mask(target_mask).to(candidate_logits.dtype)
    base = candidate_logits[:, 0]
    actions = candidate_logits[:, 1:]
    batch, action_count, _, _ = actions.shape
    supports = aux["v20_action_supports"].float()
    controls = aux["v20_control_supports"].float()
    type_supports = aux["v20_type_supports"].float()
    types = aux["v20_action_types"].to(actions.device)
    actionness = aux["v20_actionness_logits"]
    if supports.shape != actions.shape or controls.shape != actions.shape:
        raise ValueError("V38 action/support/control shape mismatch.")

    gain_margin = max(float(_m1(cfg, "CONSENSUS_GAIN_MARGIN", 0.002)), 0.0)
    base_dice = _hard_dice(base, gt)
    action_dice = _hard_dice(actions, gt)
    gain = action_dice - base_dice[:, None]
    positive = gain > gain_margin
    harmful = gain < -gain_margin
    neutral = ~(positive | harmful)

    # M1: full, all-type proposal supervision.
    dense_target, dense_carrier = _dense_action_targets(base, gt, type_supports, types)
    proposal_loss = _balanced_bce(
        actionness,
        dense_target,
        dense_carrier,
        pos_min=float(_m1(cfg, "ACTION_BANK_POS_WEIGHT_MIN", 2.0)),
        pos_max=float(_m1(cfg, "ACTION_BANK_POS_WEIGHT_MAX", 25.0)),
    )
    repair_raw = F.binary_cross_entropy_with_logits(
        actions, gt[:, None].expand_as(actions), reduction="none"
    )
    repair_loss = (repair_raw * supports).sum() / supports.sum().clamp_min(1.0)

    source_loss, source_iou = _source_residual_loss(cfg, base, gt, aux)
    island_idx = torch.where(types == 0)[0]
    if island_idx.numel() > 0:
        base_prob = torch.sigmoid(base)
        island_prob = torch.sigmoid(actions.index_select(1, island_idx))
        deletion = (base_prob[:, None] - island_prob).clamp_min(0.0)
        base_hard = (base_prob >= 0.5).float()
        fp = base_hard * (1.0 - gt)
        tp = base_hard * gt
        fp_removed = (deletion * fp[:, None]).sum(dim=(-2, -1)) / fp[:, None].sum(dim=(-2, -1)).clamp_min(1.0)
        tp_removed = (deletion * tp[:, None]).sum(dim=(-2, -1)) / tp[:, None].sum(dim=(-2, -1)).clamp_min(1.0)
        purification_loss = (
            -fp_removed.mean()
            + float(_m1(cfg, "CONSENSUS_TP_PRESERVE_WEIGHT", 2.0)) * tp_removed.mean()
        )
    else:
        purification_loss = source_loss * 0.0

    # M2 atomic: factual/control class loss + same-case ranking + Preserve calibration.
    available = aux["v38_atomic_available"].bool()
    context_clean = aux["v38_context_clean"].bool()
    valid = available & context_clean
    informative = positive | harmful
    supervised = valid & informative
    atomic_logits = aux["v38_atomic_cf_logit"]
    atomic_swap_logits = aux["v38_atomic_cf_swap_logit"]
    atomic_bce = _balanced_bce(
        atomic_logits,
        positive.float(),
        supervised.float(),
        pos_min=float(_m1(cfg, "CONSENSUS_CF_POS_WEIGHT_MIN", 1.0)),
        pos_max=float(_m1(cfg, "CONSENSUS_CF_POS_WEIGHT_MAX", 12.0)),
    )
    pair_margin = float(_m1(cfg, "CONSENSUS_PAIR_MARGIN", 0.20))
    preserve_margin = float(_m1(cfg, "CONSENSUS_PRESERVE_MARGIN", 0.10))
    atomic_pair_loss, atomic_pair_correct, atomic_pair_count = _casewise_pair_loss(
        atomic_logits, positive, harmful, valid, pair_margin
    )
    atomic_preserve_loss, pos_above_count, harm_above_sum = _preserve_aware_loss(
        atomic_logits, positive, harmful, valid, preserve_margin
    )

    pos_verified = positive & valid
    pos_delta = aux["v38_pos_delta"]
    neg_delta = aux["v38_neg_delta"]
    swap_delta = aux["v38_swap_delta"]
    contrast_margin = float(_m1(cfg, "CONSENSUS_CONTRAST_MARGIN", 0.0))
    swap_margin = float(_m1(cfg, "CONSENSUS_SWAP_MARGIN", 0.0))
    contrast_loss = (
        F.relu(contrast_margin - (pos_delta + neg_delta)) * pos_verified.float()
    ).sum() / pos_verified.float().sum().clamp_min(1.0)
    swap_loss = (
        F.relu(swap_margin - (pos_delta - swap_delta)) * pos_verified.float()
    ).sum() / pos_verified.float().sum().clamp_min(1.0)
    antisymmetry_loss = (
        (atomic_logits + atomic_swap_logits).pow(2) * valid.float()
    ).sum() / valid.float().sum().clamp_min(1.0)

    # M2 composition: the composition head is trained against the actual
    # Train-only outcome of Base + delete-delta + fill-delta.
    pair_indices = aux["v38_combo_pair_indices"].to(actions.device)
    combo_logits_model = aux["v38_combo_cf_logit"]
    combo_available = aux["v38_combo_available"].bool()
    composed = _combo_logits(base, actions, pair_indices)
    if composed.shape[1] > 0:
        combo_dice = _hard_dice(composed, gt)
        combo_gain = combo_dice - base_dice[:, None]
        combo_positive = combo_gain > gain_margin
        combo_harmful = combo_gain < -gain_margin
        combo_supervised = combo_available & (combo_positive | combo_harmful)
        combo_bce = _balanced_bce(
            combo_logits_model,
            combo_positive.float(),
            combo_supervised.float(),
            pos_min=float(_m1(cfg, "CONSENSUS_CF_POS_WEIGHT_MIN", 1.0)),
            pos_max=float(_m1(cfg, "CONSENSUS_CF_POS_WEIGHT_MAX", 12.0)),
        )
        combo_pair_loss, combo_pair_correct, combo_pair_count = _casewise_pair_loss(
            combo_logits_model,
            combo_positive,
            combo_harmful,
            combo_available,
            pair_margin,
        )
        combo_preserve_loss, combo_pos_above, combo_harm_above = _preserve_aware_loss(
            combo_logits_model,
            combo_positive,
            combo_harmful,
            combo_available,
            preserve_margin,
        )
    else:
        combo_gain = actions.new_zeros((batch, 0))
        combo_positive = combo_gain.bool()
        combo_harmful = combo_gain.bool()
        combo_bce = atomic_logits.sum() * 0.0
        combo_pair_loss = atomic_logits.sum() * 0.0
        combo_preserve_loss = atomic_logits.sum() * 0.0
        combo_pair_correct = atomic_logits.new_zeros(())
        combo_pair_count = atomic_logits.new_zeros(())
        combo_pos_above = atomic_logits.new_zeros(())
        combo_harm_above = atomic_logits.new_zeros(())

    # Geometry adversarial probes: train-only deconfounding regularizers.
    type_target = types[None].expand(batch, -1)
    area_target, position_target = _geometry_targets(supports)
    geom_type = _masked_ce(aux["v38_type_probe"], type_target, valid)
    geom_area = _masked_ce(aux["v38_area_probe"], area_target, valid)
    geom_pos = _masked_ce(aux["v38_position_probe"], position_target, valid)
    geometry_adv_loss = (geom_type + geom_area + geom_pos) / 3.0

    control_overlap = (
        supports * controls
    ).sum(dim=(-2, -1)) / supports.sum(dim=(-2, -1)).clamp_min(1.0)

    total = (
        float(_m1(cfg, "CONSENSUS_SOURCE_WEIGHT", 1.0)) * source_loss
        + float(_m1(cfg, "CONSENSUS_PURIFICATION_WEIGHT", 1.0)) * purification_loss
        + float(_m1(cfg, "CONSENSUS_PROPOSAL_WEIGHT", 1.0)) * proposal_loss
        + float(_m1(cfg, "CONSENSUS_REPAIR_WEIGHT", 0.75)) * repair_loss
        + float(_m1(cfg, "CONSENSUS_ATOMIC_BCE_WEIGHT", 1.0)) * atomic_bce
        + float(_m1(cfg, "CONSENSUS_ATOMIC_PAIR_WEIGHT", 1.5)) * atomic_pair_loss
        + float(_m1(cfg, "CONSENSUS_ATOMIC_PRESERVE_WEIGHT", 1.0)) * atomic_preserve_loss
        + float(_m1(cfg, "CONSENSUS_COMBO_BCE_WEIGHT", 0.75)) * combo_bce
        + float(_m1(cfg, "CONSENSUS_COMBO_PAIR_WEIGHT", 1.0)) * combo_pair_loss
        + float(_m1(cfg, "CONSENSUS_COMBO_PRESERVE_WEIGHT", 0.75)) * combo_preserve_loss
        + float(_m1(cfg, "CONSENSUS_CONTRAST_WEIGHT", 0.25)) * contrast_loss
        + float(_m1(cfg, "CONSENSUS_SWAP_WEIGHT", 0.25)) * swap_loss
        + float(_m1(cfg, "CONSENSUS_ANTISYMMETRY_WEIGHT", 0.10)) * antisymmetry_loss
        + float(_m1(cfg, "CONSENSUS_GEOMETRY_ADV_WEIGHT", 0.10)) * geometry_adv_loss
        + float(_m1(cfg, "CONSENSUS_CONTROL_OVERLAP_WEIGHT", 0.50)) * control_overlap.mean()
    )

    # Train-only diagnostics. M3 itself remains deterministic and untrained.
    hard_selected = aux["v20_selector_hard"].float()
    selected_gain = (hard_selected * gain).sum(dim=1)
    selected_positive = (hard_selected * positive.float()).sum(dim=1)
    selected_harmful = (hard_selected * harmful.float()).sum(dim=1)
    selected_neutral = (hard_selected * neutral.float()).sum(dim=1)
    preserve = hard_selected.sum(dim=1) <= 0.0
    candidate_oracle_gain = gain.masked_fill(~valid, -1e4).max(dim=1).values
    candidate_oracle_gain = torch.where(
        candidate_oracle_gain < -1e3,
        torch.zeros_like(candidate_oracle_gain),
        candidate_oracle_gain,
    )
    fused = aux["v20_fused_logits"]
    fused_gain = _hard_dice(fused, gt) - base_dice
    atomic_prob = torch.sigmoid(atomic_logits)
    atomic_threshold = float(_m1(cfg, "CONSENSUS_ATOMIC_TEXT_THRESHOLD", 0.50))
    combo_threshold = float(_m1(cfg, "CONSENSUS_COMBO_TEXT_THRESHOLD", 0.55))
    atomic_pos_recall = (
        ((atomic_prob >= atomic_threshold) & positive & valid).float().sum()
        / (positive & valid).float().sum().clamp_min(1.0)
    )
    atomic_harm_accept = (
        ((atomic_prob >= atomic_threshold) & harmful & valid).float().sum()
        / (harmful & valid).float().sum().clamp_min(1.0)
    )
    combo_prob = torch.sigmoid(combo_logits_model)
    combo_pos_recall = (
        ((combo_prob >= combo_threshold) & combo_positive & combo_available).float().sum()
        / (combo_positive & combo_available).float().sum().clamp_min(1.0)
    )
    combo_harm_accept = (
        ((combo_prob >= combo_threshold) & combo_harmful & combo_available).float().sum()
        / (combo_harmful & combo_available).float().sum().clamp_min(1.0)
    )

    z = total.detach() * 0.0
    diagnostics = {
        "v38_source_loss": source_loss.detach(),
        "v38_source_iou": source_iou.detach(),
        "v38_purification_loss": purification_loss.detach(),
        "v38_proposal_loss": proposal_loss.detach(),
        "v38_repair_loss": repair_loss.detach(),
        "v38_atomic_bce_loss": atomic_bce.detach(),
        "v38_atomic_pair_loss": atomic_pair_loss.detach(),
        "v38_atomic_preserve_loss": atomic_preserve_loss.detach(),
        "v38_combo_bce_loss": combo_bce.detach(),
        "v38_combo_pair_loss": combo_pair_loss.detach(),
        "v38_combo_preserve_loss": combo_preserve_loss.detach(),
        "v38_contrast_loss": contrast_loss.detach(),
        "v38_swap_loss": swap_loss.detach(),
        "v38_antisymmetry_loss": antisymmetry_loss.detach(),
        "v38_geometry_adv_loss": geometry_adv_loss.detach(),
        "v38_control_overlap": control_overlap.mean().detach(),
        "v38_valid_action_rate": valid.float().mean().detach(),
        "v38_context_clean_rate": context_clean.float().mean().detach(),
        "v38_atomic_qualified_rate": aux["v38_atomic_qualified"].float().mean().detach(),
        "v38_atomic_positive_recall": atomic_pos_recall.detach(),
        "v38_atomic_harmful_false_accept": atomic_harm_accept.detach(),
        "v38_atomic_pair_rank_accuracy": atomic_pair_correct.detach() / atomic_pair_count.clamp_min(1.0),
        "v38_atomic_pair_count": atomic_pair_count.detach(),
        "v38_atomic_positive_above_preserve": pos_above_count.detach(),
        "v38_atomic_harmful_above_preserve": harm_above_sum.detach(),
        "v38_combo_available_rate": combo_available.float().mean().detach() if combo_available.numel() else z,
        "v38_combo_qualified_rate": aux["v38_combo_qualified"].float().mean().detach() if combo_available.numel() else z,
        "v38_combo_positive_recall": combo_pos_recall.detach(),
        "v38_combo_harmful_false_accept": combo_harm_accept.detach(),
        "v38_combo_pair_rank_accuracy": combo_pair_correct.detach() / combo_pair_count.clamp_min(1.0),
        "v38_combo_pair_count": combo_pair_count.detach(),
        "v38_combo_positive_above_preserve": combo_pos_above.detach(),
        "v38_combo_harmful_above_preserve": combo_harm_above.detach(),
        "v38_consensus_active_rate": aux["v38_consensus_active"].float().mean().detach(),
        "v38_multi_active_rate": aux["v38_multi_active"].float().mean().detach(),
        "v38_singleton_active_rate": aux["v38_singleton_active"].float().mean().detach(),
        "v38_cluster_size": aux["v38_cluster_size"].float().mean().detach(),
        "v38_selected_positive_rate": (selected_positive > 0).float().mean().detach(),
        "v38_selected_harmful_rate": (selected_harmful > 0).float().mean().detach(),
        "v38_selected_neutral_rate": (selected_neutral > 0).float().mean().detach(),
        "v38_preserve_rate": preserve.float().mean().detach(),
        "v38_selected_gain": selected_gain.mean().detach(),
        "v38_oracle_gain": candidate_oracle_gain.mean().detach(),
        "v38_fusion_gain": fused_gain.mean().detach(),
        # Generic trainer diagnostics.
        "proposal_quality_loss": proposal_loss.detach(),
        "proposal_fp_gate_loss": proposal_loss.detach(),
        "proposal_fn_gate_loss": z,
        "proposal_correction_loss": repair_loss.detach(),
        "proposal_safety_loss": (selected_harmful > 0).float().mean().detach(),
        "proposal_locality_loss": z,
        "proposal_diversity_loss": z,
        "fusion_quality_loss": z,
        "fusion_direction_loss": atomic_bce.detach(),
        "fusion_noedit_loss": z,
        "fp_target_fraction": ((base >= 0.0).float() * (1.0 - gt)).mean().detach(),
        "fn_target_fraction": ((base < 0.0).float() * gt).mean().detach(),
        "proposal_improved_rate": (gain > 0).float().mean().detach(),
        "fusion_abs_change": (torch.sigmoid(fused) - torch.sigmoid(base)).abs().mean().detach(),
        "shrink_changed_fraction": z,
        "expand_changed_fraction": z,
        "fusion_changed_fraction": (fused_gain.abs() > 1e-8).float().mean().detach(),
        "candidate_base_detached": torch.tensor(
            float(bool(aux.get("candidate_base_detached", False))),
            device=total.device,
        ),
        "m1_train_ratio": z + 1.0,
        "m2_ratio": z + 1.0,
    }
    return total, diagnostics
