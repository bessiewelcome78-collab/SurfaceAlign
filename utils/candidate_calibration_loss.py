#/home/tsz-25/MedCLIPSeg-pristine/utils/candidate_calibration_loss.py -*- coding: utf-8 -*-
"""V381 legacy + V382 action-conditional quantile loss.

V382 retains the original all-action V381 candidate generator but replaces its
single calibrated text logit with a spatial action-conditional q10/q50/q90
outcome model.  Train GT is used only to construct the current batch's signed
hard-Dice gain targets; inference consumes no GT, no validation-derived
threshold, and no singleton/consensus admission rule.
"""
from __future__ import annotations
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from .candidate_consensus_loss import (
    EPS,
    _m1,
    _foreground_mask,
    _hard_dice,
    _balanced_bce,
    _dense_action_targets,
    _source_residual_loss,
    _casewise_pair_loss,
    _preserve_aware_loss,
)


def _compute_v381_legacy_loss(
    cfg: Any,
    candidate_logits: torch.Tensor,
    target_mask: torch.Tensor,
    aux: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    required = (
        "v20_action_supports", "v20_control_supports", "v20_type_supports",
        "v20_action_types", "v20_actionness_logits", "v20_fused_logits",
        "v20_selector_hard", "v381_atomic_logit", "v381_atomic_available",
        "v381_context_clean", "v381_atomic_qualified",
        "v381_lesionness_contrast", "v381_pos_delta", "v381_neg_delta",
        "v381_consensus_active", "v381_multi_active", "v381_singleton_active",
        "v381_cluster_size",
    )
    missing = [key for key in required if key not in aux]
    if missing:
        raise KeyError(f"V381 auxiliary outputs missing: {missing}")
    if candidate_logits.ndim != 4 or candidate_logits.shape[1] < 2:
        raise ValueError("V381 expects candidate logits [B,1+K,H,W].")

    gt = _foreground_mask(target_mask).to(candidate_logits.dtype)
    base = candidate_logits[:, 0]
    actions = candidate_logits[:, 1:]
    batch = actions.shape[0]
    supports = aux["v20_action_supports"].float()
    controls = aux["v20_control_supports"].float()
    type_supports = aux["v20_type_supports"].float()
    types = aux["v20_action_types"].to(actions.device)
    actionness = aux["v20_actionness_logits"]
    if supports.shape != actions.shape or controls.shape != actions.shape:
        raise ValueError("V381 action/support/control shape mismatch.")

    gain_margin = max(float(_m1(cfg, "CALIBRATION_GAIN_MARGIN", 0.002)), 0.0)
    base_dice = _hard_dice(base, gt)
    action_dice = _hard_dice(actions, gt)
    gain = action_dice - base_dice[:, None]
    positive = gain > gain_margin
    harmful = gain < -gain_margin
    neutral = ~(positive | harmful)

    # M1 proposal objectives are deliberately unchanged from V38.
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
            + float(_m1(cfg, "CALIBRATION_TP_PRESERVE_WEIGHT", 2.0)) * tp_removed.mean()
        )
    else:
        purification_loss = source_loss * 0.0

    # M2: one lesion-vs-background factual/control score, trained and deployed
    # against the same Preserve=0 reference.  No swap, no raw-delta AND gates.
    available = aux["v381_atomic_available"].bool()
    context_clean = aux["v381_context_clean"].bool()
    valid = available & context_clean
    informative = positive | harmful
    supervised = valid & informative
    atomic_logits = aux["v381_atomic_logit"]
    atomic_bce = _balanced_bce(
        atomic_logits,
        positive.float(),
        supervised.float(),
        pos_min=float(_m1(cfg, "CALIBRATION_CF_POS_WEIGHT_MIN", 1.0)),
        pos_max=float(_m1(cfg, "CALIBRATION_CF_POS_WEIGHT_MAX", 12.0)),
    )
    pair_margin = float(_m1(cfg, "CALIBRATION_PAIR_MARGIN", 0.10))
    preserve_margin = float(_m1(cfg, "CALIBRATION_PRESERVE_MARGIN", 0.05))
    atomic_pair_loss, atomic_pair_correct, atomic_pair_count = _casewise_pair_loss(
        atomic_logits, positive, harmful, valid, pair_margin
    )
    atomic_preserve_loss, pos_above_count, harm_above_sum = _preserve_aware_loss(
        atomic_logits, positive, harmful, valid, preserve_margin
    )

    total = (
        float(_m1(cfg, "CALIBRATION_SOURCE_WEIGHT", 1.0)) * source_loss
        + float(_m1(cfg, "CALIBRATION_PURIFICATION_WEIGHT", 1.0)) * purification_loss
        + float(_m1(cfg, "CALIBRATION_PROPOSAL_WEIGHT", 1.0)) * proposal_loss
        + float(_m1(cfg, "CALIBRATION_REPAIR_WEIGHT", 0.75)) * repair_loss
        + float(_m1(cfg, "CALIBRATION_ATOMIC_BCE_WEIGHT", 1.0)) * atomic_bce
        + float(_m1(cfg, "CALIBRATION_ATOMIC_PAIR_WEIGHT", 1.5)) * atomic_pair_loss
        + float(_m1(cfg, "CALIBRATION_ATOMIC_PRESERVE_WEIGHT", 1.0)) * atomic_preserve_loss
    )

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
    deploy_threshold = float(_m1(cfg, "CALIBRATION_DEPLOY_LOGIT_THRESHOLD", 0.0))
    atomic_pos_recall = (
        ((atomic_logits >= deploy_threshold) & positive & valid).float().sum()
        / (positive & valid).float().sum().clamp_min(1.0)
    )
    atomic_harm_accept = (
        ((atomic_logits >= deploy_threshold) & harmful & valid).float().sum()
        / (harmful & valid).float().sum().clamp_min(1.0)
    )

    contrast = aux["v381_lesionness_contrast"]
    pos_valid = positive & valid
    harm_valid = harmful & valid
    z = total.detach() * 0.0
    diagnostics = {
        "v381_source_loss": source_loss.detach(),
        "v381_source_iou": source_iou.detach(),
        "v381_purification_loss": purification_loss.detach(),
        "v381_proposal_loss": proposal_loss.detach(),
        "v381_repair_loss": repair_loss.detach(),
        "v381_atomic_bce_loss": atomic_bce.detach(),
        "v381_atomic_pair_loss": atomic_pair_loss.detach(),
        "v381_atomic_preserve_loss": atomic_preserve_loss.detach(),
        "v381_valid_action_rate": valid.float().mean().detach(),
        "v381_context_clean_rate": context_clean.float().mean().detach(),
        "v381_atomic_qualified_rate": aux["v381_atomic_qualified"].float().mean().detach(),
        "v381_atomic_positive_recall": atomic_pos_recall.detach(),
        "v381_atomic_harmful_false_accept": atomic_harm_accept.detach(),
        "v381_atomic_pair_rank_accuracy": atomic_pair_correct.detach() / atomic_pair_count.clamp_min(1.0),
        "v381_atomic_pair_count": atomic_pair_count.detach(),
        "v381_atomic_positive_above_preserve": pos_above_count.detach(),
        "v381_atomic_harmful_above_preserve": harm_above_sum.detach(),
        "v381_positive_contrast": (
            (contrast * pos_valid.float()).sum() / pos_valid.float().sum().clamp_min(1.0)
        ).detach(),
        "v381_harmful_contrast": (
            (contrast * harm_valid.float()).sum() / harm_valid.float().sum().clamp_min(1.0)
        ).detach(),
        "v381_consensus_active_rate": aux["v381_consensus_active"].float().mean().detach(),
        "v381_multi_active_rate": aux["v381_multi_active"].float().mean().detach(),
        "v381_singleton_active_rate": aux["v381_singleton_active"].float().mean().detach(),
        "v381_cluster_size": aux["v381_cluster_size"].float().mean().detach(),
        "v381_selected_positive_rate": (selected_positive > 0).float().mean().detach(),
        "v381_selected_harmful_rate": (selected_harmful > 0).float().mean().detach(),
        "v381_selected_neutral_rate": (selected_neutral > 0).float().mean().detach(),
        "v381_preserve_rate": preserve.float().mean().detach(),
        "v381_selected_gain": selected_gain.mean().detach(),
        "v381_oracle_gain": candidate_oracle_gain.mean().detach(),
        "v381_fusion_gain": fused_gain.mean().detach(),
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
            float(bool(aux.get("candidate_base_detached", False))), device=total.device
        ),
        "m1_train_ratio": z + 1.0,
        "m2_ratio": z + 1.0,
    }
    return total, diagnostics



def _pinball(prediction: torch.Tensor, target: torch.Tensor, tau: float) -> torch.Tensor:
    error = target - prediction
    return torch.maximum(float(tau) * error, (float(tau) - 1.0) * error)


def _compute_v382_action_conditional_quantile_loss(
    cfg: Any,
    candidate_logits: torch.Tensor,
    target_mask: torch.Tensor,
    aux: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    required = (
        "v20_action_supports", "v20_control_supports", "v20_type_supports",
        "v20_action_types", "v20_actionness_logits", "v20_fused_logits",
        "v20_selector_hard", "v382_quantile_gain", "v382_lower_gain",
        "v382_outcome_logits", "v382_class_logits", "v382_valid_action",
        "v382_pre_veto_valid", "v382_structural_safe",
        "v382_selected_action", "v382_soft_choice_probs",
    )
    missing = [key for key in required if key not in aux]
    if missing:
        raise KeyError(f"V382 auxiliary outputs missing: {missing}")
    if candidate_logits.ndim != 4 or candidate_logits.shape[1] < 2:
        raise ValueError("V382 expects candidate logits [B,1+K,H,W].")

    gt = _foreground_mask(target_mask).to(candidate_logits.dtype)
    base = candidate_logits[:, 0]
    actions = candidate_logits[:, 1:]
    b, k, _, _ = actions.shape
    supports = aux["v20_action_supports"].float()
    controls = aux["v20_control_supports"].float()
    type_supports = aux["v20_type_supports"].float()
    types = aux["v20_action_types"].to(actions.device)
    actionness = aux["v20_actionness_logits"]
    if supports.shape != actions.shape or controls.shape != actions.shape:
        raise ValueError("V382 action/support/control shape mismatch.")

    quantiles = aux["v382_quantile_gain"]
    lower = aux["v382_lower_gain"]
    outcome_logits = aux["v382_outcome_logits"]
    class_logits = aux["v382_class_logits"]
    valid = aux["v382_valid_action"].bool()
    soft_choice = aux["v382_soft_choice_probs"]
    if quantiles.shape != (b, k, 3) or lower.shape != (b, k):
        raise ValueError("V382 quantile tensor shape mismatch.")
    if outcome_logits.shape != (b, k, 3):
        raise ValueError("V382 outcome tensor shape mismatch.")
    if class_logits.shape != (b, k + 1) or soft_choice.shape != (b, k + 1):
        raise ValueError("V382 case-choice tensor shape mismatch.")
    if valid.shape != (b, k):
        raise ValueError("V382 valid-action tensor shape mismatch.")

    gain_scale = max(float(_m1(cfg, "V382_GAIN_SCALE", 0.06)), 1e-5)
    gain_clip = max(float(_m1(cfg, "V382_GAIN_CLIP", 0.06)), gain_scale)
    huber_beta = max(float(_m1(cfg, "V382_HUBER_BETA", 0.20)), 1e-5)
    benefit_margin = max(float(_m1(cfg, "V382_DSC_BENEFIT_MARGIN", 0.002)), 0.0)
    harm_margin = max(float(_m1(cfg, "V382_DSC_HARM_MARGIN", 0.002)), 0.0)
    severe_harm_weight = max(float(_m1(cfg, "V382_SEVERE_HARM_WEIGHT", 3.0)), 1.0)
    pair_margin = max(float(_m1(cfg, "V382_PAIRWISE_MARGIN", 0.002)), 0.0)

    with torch.no_grad():
        base_dice = _hard_dice(base, gt)
        action_dice = _hard_dice(actions, gt)
        gain = action_dice - base_dice[:, None]
        clipped_gain = gain.clamp(-gain_clip, gain_clip)
        beneficial = valid & (gain > benefit_margin)
        harmful = valid & (gain < -harm_margin)
        neutral = valid & ~(beneficial | harmful)
        outcome_target = torch.full(
            (b, k), 1, dtype=torch.long, device=actions.device
        )
        outcome_target[harmful] = 0
        outcome_target[beneficial] = 2
        best_gain, best_index = gain.masked_fill(~valid, -1e4).max(dim=1)
        target_choice = torch.where(
            best_gain > benefit_margin,
            best_index + 1,
            torch.zeros_like(best_index),
        )

    zero = quantiles.sum() * 0.0
    normalized_target = clipped_gain / gain_scale
    normalized_quantiles = quantiles / gain_scale
    if valid.any():
        q10_loss = _pinball(
            normalized_quantiles[..., 0][valid], normalized_target[valid], 0.10
        )
        q90_loss = _pinball(
            normalized_quantiles[..., 2][valid], normalized_target[valid], 0.90
        )
        action_weight = torch.where(
            harmful[valid],
            torch.full_like(q10_loss, severe_harm_weight),
            torch.ones_like(q10_loss),
        )
        quantile_loss = ((q10_loss + q90_loss) * 0.5 * action_weight).mean()
        median_loss = F.smooth_l1_loss(
            normalized_quantiles[..., 1][valid],
            normalized_target[valid],
            beta=huber_beta,
        )
        class_weight = torch.tensor(
            [
                float(_m1(cfg, "V382_HARM_CLASS_WEIGHT", 2.0)),
                1.0,
                float(_m1(cfg, "V382_BENEFIT_CLASS_WEIGHT", 4.0)),
            ],
            device=actions.device,
            dtype=outcome_logits.dtype,
        )
        outcome_loss = F.cross_entropy(
            outcome_logits[valid], outcome_target[valid], weight=class_weight
        )
        q50_mae = (quantiles[..., 1][valid] - clipped_gain[valid]).abs().mean()
    else:
        quantile_loss = zero
        median_loss = zero
        outcome_loss = zero
        q50_mae = zero.detach()

    # This directly matches the deployed candidate set: Preserve has class 0
    # and score 0; a positive gain candidate is the only positive class target.
    choice_loss = F.cross_entropy(class_logits, target_choice)
    action_probability = soft_choice[:, 1:]
    expected_gain_loss = -(
        action_probability * normalized_target * valid.float()
    ).sum(dim=1).mean()
    downside = F.relu(-normalized_target - harm_margin / gain_scale)
    downside_loss = (
        action_probability * downside * valid.float()
    ).sum(dim=1).mean()

    pair_terms = []
    for bi in range(b):
        positive_q10 = lower[bi][beneficial[bi]]
        harmful_q10 = lower[bi][harmful[bi]]
        if positive_q10.numel() and harmful_q10.numel():
            pair_terms.append(
                F.softplus(
                    pair_margin / gain_scale
                    - (positive_q10[:, None] - harmful_q10[None, :]) / gain_scale
                ).mean()
            )
    pairwise_loss = torch.stack(pair_terms).mean() if pair_terms else zero

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
        fp_removed = (
            (deletion * fp[:, None]).sum(dim=(-2, -1))
            / fp[:, None].sum(dim=(-2, -1)).clamp_min(1.0)
        )
        tp_removed = (
            (deletion * tp[:, None]).sum(dim=(-2, -1))
            / tp[:, None].sum(dim=(-2, -1)).clamp_min(1.0)
        )
        purification_loss = (
            -fp_removed.mean()
            + float(_m1(cfg, "V382_TP_PRESERVE_WEIGHT", 2.0)) * tp_removed.mean()
        )
    else:
        purification_loss = source_loss * 0.0

    total = (
        float(_m1(cfg, "V382_SOURCE_WEIGHT", 1.0)) * source_loss
        + float(_m1(cfg, "V382_PURIFICATION_WEIGHT", 1.0)) * purification_loss
        + float(_m1(cfg, "V382_PROPOSAL_WEIGHT", 1.0)) * proposal_loss
        + float(_m1(cfg, "V382_REPAIR_WEIGHT", 0.75)) * repair_loss
        + float(_m1(cfg, "V382_QUANTILE_WEIGHT", 1.0)) * quantile_loss
        + float(_m1(cfg, "V382_MEDIAN_WEIGHT", 0.75)) * median_loss
        + float(_m1(cfg, "V382_OUTCOME_WEIGHT", 0.50)) * outcome_loss
        + float(_m1(cfg, "V382_CHOICE_WEIGHT", 2.0)) * choice_loss
        + float(_m1(cfg, "V382_EXPECTED_GAIN_WEIGHT", 1.0)) * expected_gain_loss
        + float(_m1(cfg, "V382_DOWNSIDE_WEIGHT", 4.0)) * downside_loss
        + float(_m1(cfg, "V382_PAIRWISE_WEIGHT", 0.75)) * pairwise_loss
    )

    selected = aux["v382_selected_action"].float()
    selected_gain = (selected * gain).sum(dim=1)
    selected_positive = (selected * beneficial.float()).sum(dim=1)
    selected_harmful = (selected * harmful.float()).sum(dim=1)
    selected_neutral = (selected * neutral.float()).sum(dim=1)
    preserve = selected.sum(dim=1) <= 0.0
    oracle_gain = gain.masked_fill(~valid, -1e4).max(dim=1).values.clamp_min(0.0)
    fused_gain = _hard_dice(aux["v20_fused_logits"], gt) - base_dice
    q10_positive = lower[beneficial]
    q10_harmful = lower[harmful]
    harm_probability = torch.softmax(outcome_logits, dim=-1)[..., 0]
    q_order_ok = (
        (quantiles[..., 0] <= quantiles[..., 1] + 1e-7)
        & (quantiles[..., 1] <= quantiles[..., 2] + 1e-7)
    ).float().mean()
    z = total.detach() * 0.0

    diagnostics = {
        "v382_source_loss": source_loss.detach(),
        "v382_source_iou": source_iou.detach(),
        "v382_purification_loss": purification_loss.detach(),
        "v382_proposal_loss": proposal_loss.detach(),
        "v382_repair_loss": repair_loss.detach(),
        "v382_quantile_loss": quantile_loss.detach(),
        "v382_median_loss": median_loss.detach(),
        "v382_outcome_loss": outcome_loss.detach(),
        "v382_choice_loss": choice_loss.detach(),
        "v382_expected_gain_loss": expected_gain_loss.detach(),
        "v382_downside_loss": downside_loss.detach(),
        "v382_pairwise_loss": pairwise_loss.detach(),
        "v382_q50_mae": q50_mae.detach(),
        "v382_valid_action_rate": valid.float().mean().detach(),
        "v382_pre_veto_action_rate": aux["v382_pre_veto_valid"].float().mean().detach(),
        "v382_structural_safe_rate": aux["v382_structural_safe"].float().mean().detach(),
        "v382_beneficial_rate": beneficial.float().mean().detach(),
        "v382_harmful_rate": harmful.float().mean().detach(),
        "v382_oracle_gain": oracle_gain.mean().detach(),
        "v382_selected_gain": selected_gain.mean().detach(),
        "v382_fusion_gain": fused_gain.mean().detach(),
        "v382_selected_positive_rate": (selected_positive > 0).float().mean().detach(),
        "v382_selected_harmful_rate": (selected_harmful > 0).float().mean().detach(),
        "v382_selected_neutral_rate": (selected_neutral > 0).float().mean().detach(),
        "v382_preserve_rate": preserve.float().mean().detach(),
        "v382_selected_actions_per_case": selected.sum(dim=1).mean().detach(),
        "v382_target_choice_accuracy": class_logits.argmax(dim=1).eq(target_choice).float().mean().detach(),
        "v382_q10_positive_mean": q10_positive.mean().detach() if q10_positive.numel() else z,
        "v382_q10_harmful_mean": q10_harmful.mean().detach() if q10_harmful.numel() else z,
        "v382_harm_probability_mean": harm_probability[valid].mean().detach() if valid.any() else z,
        "v382_quantile_order_rate": q_order_ok.detach(),
        "v382_singleton_active_rate": z,
        "v382_consensus_active_rate": z,
        # Generic trainer diagnostics.
        "proposal_quality_loss": proposal_loss.detach(),
        "proposal_fp_gate_loss": proposal_loss.detach(),
        "proposal_fn_gate_loss": z,
        "proposal_correction_loss": repair_loss.detach(),
        "proposal_safety_loss": (selected_harmful > 0).float().mean().detach(),
        "proposal_locality_loss": z,
        "proposal_diversity_loss": z,
        "fusion_quality_loss": z,
        "fusion_direction_loss": quantile_loss.detach(),
        "fusion_noedit_loss": choice_loss.detach(),
        "fp_target_fraction": ((base >= 0.0).float() * (1.0 - gt)).mean().detach(),
        "fn_target_fraction": ((base < 0.0).float() * gt).mean().detach(),
        "proposal_improved_rate": (gain > 0).float().mean().detach(),
        "fusion_abs_change": (torch.sigmoid(aux["v20_fused_logits"]) - torch.sigmoid(base)).abs().mean().detach(),
        "shrink_changed_fraction": z,
        "expand_changed_fraction": z,
        "fusion_changed_fraction": (fused_gain.abs() > 1e-8).float().mean().detach(),
        "candidate_base_detached": torch.tensor(
            float(bool(aux.get("candidate_base_detached", False))), device=total.device
        ),
        "m1_train_ratio": z + 1.0,
        "m2_ratio": z + 1.0,
    }
    return total, diagnostics


def compute_candidate_calibration_loss(
    cfg: Any,
    candidate_logits: torch.Tensor,
    target_mask: torch.Tensor,
    aux: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Dispatch the preserved V381 loss or the V382 quantile-value loss."""
    if "v382_quantile_gain" in aux:
        return _compute_v382_action_conditional_quantile_loss(
            cfg, candidate_logits, target_mask, aux
        )
    return _compute_v381_legacy_loss(cfg, candidate_logits, target_mask, aux)
