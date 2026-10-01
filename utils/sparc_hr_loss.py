"""Losses and diagnostics for SPARC-HR3.1 safety-constrained routing.

The teacher is local and candidate-relative: at every router cell it asks
which available source reduces the current M1 error most.  M1 itself is source
zero, so a cell with no positive attainable advantage receives the exact
preserve label.  This produces thousands of supervised decisions per image and
matches the one-shot deployment operator.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

EPS = 1.0e-6


def _cfg_get(obj, name: str, default):
    return getattr(obj, name, default) if obj is not None else default


def _mask4(target: torch.Tensor) -> torch.Tensor:
    if target.ndim == 3:
        target = target[:, None]
    if target.ndim != 4 or target.shape[1] != 1:
        raise ValueError(f"segmentation target must be [B,H,W] or [B,1,H,W], got {tuple(target.shape)}")
    return target.float()


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(value)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _dice_loss(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, prob.ndim))
    inter = (prob * target).sum(dim=dims)
    den = (prob + target).sum(dim=dims)
    return (1.0 - (2.0 * inter + EPS) / (den + EPS)).mean()


def _seg_loss(prob: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prob = prob.clamp(EPS, 1.0 - EPS)
    bce = F.binary_cross_entropy(prob, target)
    dice = _dice_loss(prob, target)
    return 0.5 * bce + 0.5 * dice, bce, dice


def _boundary(prob: torch.Tensor) -> torch.Tensor:
    return (F.max_pool2d(prob, 3, 1, 1) + F.max_pool2d(-prob, 3, 1, 1)).clamp(0.0, 1.0)


def _hard_boundary(mask: torch.Tensor) -> torch.Tensor:
    dilated = F.max_pool2d(mask, 3, 1, 1)
    eroded = -F.max_pool2d(-mask, 3, 1, 1)
    return (dilated != eroded).float()


def _hard_quality(prob: torch.Tensor, target: torch.Tensor, tolerance: int) -> torch.Tensor:
    pred, gt = (prob >= 0.5).float(), (target >= 0.5).float()
    inter = (pred * gt).sum(dim=(-3, -2, -1))
    den = (pred + gt).sum(dim=(-3, -2, -1))
    dice = (2.0 * inter + EPS) / (den + EPS)
    pb, gb = _hard_boundary(pred), _hard_boundary(gt)
    radius = max(0, int(tolerance))
    if radius:
        kernel = 2 * radius + 1
        near_g = F.max_pool2d(gb, kernel, 1, radius)
        near_p = F.max_pool2d(pb, kernel, 1, radius)
    else:
        near_g, near_p = gb, pb
    hits = (pb * near_g).sum(dim=(-3, -2, -1)) + (gb * near_p).sum(dim=(-3, -2, -1))
    surf_den = pb.sum(dim=(-3, -2, -1)) + gb.sum(dim=(-3, -2, -1))
    surface = (hits + EPS) / (surf_den + EPS)
    return 0.5 * dice + 0.5 * surface


def _balanced_binary_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    positive = target > 0.5
    negative = ~positive
    pos_loss = _masked_mean(-F.logsigmoid(logits), positive)
    neg_loss = _masked_mean(-F.logsigmoid(-logits), negative)
    has_pos = positive.any().to(logits)
    return 0.5 * has_pos * pos_loss + (1.0 - 0.5 * has_pos) * neg_loss


def _router_teacher(cfg, target: torch.Tensor, aux: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Dense attainable-advantage labels in the exact deployment coordinates."""
    m1 = _cfg_get(cfg, "M1", None)
    sources = aux["sparc_source_probs_hr"].detach().clamp(EPS, 1.0 - EPS)
    logits = aux["sparc_router_logits"]
    b, j, _, hh, hw = sources.shape
    grid = tuple(logits.shape[-2:])
    gt = target[:, None].expand(-1, j, -1, -1, -1)
    anchor = sources[:, :1]
    anchor_hard_error = ((anchor >= 0.5) != (target[:, None] >= 0.5)).float()
    source_hard_error = ((sources >= 0.5) != (gt >= 0.5)).float()
    hard_adv_hr = anchor_hard_error - source_hard_error
    hard_adv = F.adaptive_avg_pool2d(hard_adv_hr.reshape(b * j, 1, hh, hw), grid)
    hard_adv = hard_adv.reshape(b, j, *grid)

    anchor_bce = F.binary_cross_entropy(anchor.expand_as(gt), gt, reduction="none")
    source_bce = F.binary_cross_entropy(sources, gt, reduction="none")
    soft_adv_hr = (anchor_bce - source_bce).clamp(-2.0, 2.0)
    soft_adv = F.adaptive_avg_pool2d(soft_adv_hr.reshape(b * j, 1, hh, hw), grid)
    soft_adv = soft_adv.reshape(b, j, *grid)
    change_hr = ((sources >= 0.5) != (anchor >= 0.5)).float()
    change = F.adaptive_avg_pool2d(change_hr.reshape(b * j, 1, hh, hw), grid)
    change = change.reshape(b, j, *grid)

    soft_weight = float(_cfg_get(m1, "GEOTR_SPARC_ROUTER_SOFT_ADVANTAGE_WEIGHT", 0.0))
    edit_cost = float(_cfg_get(m1, "GEOTR_SPARC_ROUTER_EDIT_COST", 0.02))
    advantage = hard_adv + soft_weight * soft_adv - edit_cost * change
    advantage[:, 0] = 0.0
    valid = aux.get("sparc_router_valid_mask")
    if not isinstance(valid, torch.Tensor):
        valid = torch.ones_like(advantage, dtype=torch.bool)
    else:
        valid = valid.detach().bool()
    valid[:, 0] = True
    # A finite sentinel avoids 0 * inf in expected-advantage losses while
    # making invalid/no-op sources strictly worse than KEEP.
    advantage = advantage.masked_fill(~valid, -1.0)
    best_advantage, teacher = advantage.max(dim=1)
    minimum = float(_cfg_get(m1, "GEOTR_SPARC_ROUTER_MIN_ADVANTAGE", 0.0))
    actionable = best_advantage > minimum
    teacher = torch.where(actionable, teacher, torch.zeros_like(teacher))
    return {
        "advantage": advantage,
        "hard_advantage": hard_adv,
        "change": change,
        "teacher": teacher,
        "actionable": actionable,
        "best_advantage": best_advantage.clamp_min(0.0),
        "anchor_error_hr": anchor_hard_error[:, 0],
        "valid": valid,
    }


def _compose_from_grid(sources: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    b, j, _, hh, hw = sources.shape
    hard = F.one_hot(labels, num_classes=j).permute(0, 3, 1, 2).to(sources)
    hard_hr = F.interpolate(hard, size=(hh, hw), mode="nearest")
    return (hard_hr[:, :, None] * sources).sum(dim=1).clamp(EPS, 1.0 - EPS)


def _family_oracle_gain(
    sources: torch.Tensor,
    family_ids: torch.Tensor,
    advantage: torch.Tensor,
    target: torch.Tensor,
    anchor_quality: torch.Tensor,
    tolerance: int,
    family: int,
) -> torch.Tensor:
    allowed = (family_ids == family)
    allowed[0] = True
    masked = advantage.masked_fill(~allowed[None, :, None, None], -torch.inf)
    labels = masked.argmax(dim=1)
    labels = torch.where(masked.max(dim=1).values > 0.0, labels, torch.zeros_like(labels))
    prob = _compose_from_grid(sources, labels)
    return (_hard_quality(prob, target, tolerance) - anchor_quality).clamp_min(0.0)


def _routing_statistics(cfg, target: torch.Tensor, aux: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    m1 = _cfg_get(cfg, "M1", None)
    teacher = _router_teacher(cfg, target, aux)
    sources = aux["sparc_source_probs_hr"]
    selected = aux["sparc_router_selected_source"]
    labels = teacher["teacher"]
    advantage = teacher["advantage"]
    selected_adv = advantage.gather(1, selected[:, None])[:, 0]
    edited = selected != 0
    actionable = teacher["actionable"]
    tolerance = int(_cfg_get(m1, "GEOTR_SPARC_SURFACE_TOLERANCE_HR_PX", 4))
    anchor_q = _hard_quality(aux["sparc_anchor_prob_hr"], target, tolerance)
    final_q = _hard_quality(aux["sparc_final_prob_hr"], target, tolerance)
    oracle_prob = _compose_from_grid(sources, labels)
    oracle_q = _hard_quality(oracle_prob, target, tolerance)
    proposal_q = _hard_quality(aux["sparc_learned_proposal_prob_hr"], target, tolerance)
    quality_gain = final_q - anchor_q
    oracle_gain = (oracle_q - anchor_q).clamp_min(0.0)
    family_ids = aux["sparc_source_family_ids"]
    source_oracles = {
        1: _family_oracle_gain(sources, family_ids, advantage, target, anchor_q, tolerance, 1),
        2: _family_oracle_gain(sources, family_ids, advantage, target, anchor_q, tolerance, 2),
        3: _family_oracle_gain(sources, family_ids, advantage, target, anchor_q, tolerance, 3),
        4: _family_oracle_gain(sources, family_ids, advantage, target, anchor_q, tolerance, 4),
    }
    pred_hard = aux["sparc_final_prob_hr"] >= 0.5
    anchor_hard = aux["sparc_anchor_prob_hr"] >= 0.5
    gt_hard = target >= 0.5
    corrected = ((pred_hard == gt_hard) & (anchor_hard != gt_hard)).float().sum()
    introduced = ((pred_hard != gt_hard) & (anchor_hard == gt_hard)).float().sum()
    positive_selected = edited & (selected_adv > 0.0)
    harmful_selected = edited & (selected_adv < 0.0)
    beneficial_any = (advantage[:, 1:] > 0.0) & teacher["valid"][:, 1:]
    top_accuracy = (selected == labels).float().mean()
    stop_accuracy = ((selected == 0) == (labels == 0)).float().mean()
    execution_case = edited.flatten(1).any(1).float().mean()
    execution_cell = edited.float().mean()
    selected_precision = _masked_mean(positive_selected.float(), edited)
    selected_harm = _masked_mean(harmful_selected.float(), edited)
    selected_mean_benefit = _masked_mean(selected_adv.clamp_min(0.0), edited)
    selected_mean_harm = _masked_mean((-selected_adv).clamp_min(0.0), edited)
    benefit_recall = _masked_mean(positive_selected.float(), actionable)
    change = teacher["change"][:, 1:]
    no_op = (change <= 0.0).float().mean()
    duplicate_mask = aux.get("sparc_router_duplicate_mask")
    duplicate_rate = (
        duplicate_mask[:, 1:].float().mean()
        if isinstance(duplicate_mask, torch.Tensor) and duplicate_mask.shape[1] > 1
        else no_op.new_zeros(())
    )
    valid_rate = teacher["valid"][:, 1:].float().mean()
    family_mass = []
    for family in range(5):
        family_mass.append((family_ids[selected] == family).float().mean())
    return {
        **teacher,
        "anchor_quality": anchor_q, "final_quality": final_q,
        "proposal_quality": proposal_q, "quality_gain": quality_gain,
        "oracle_gain": oracle_gain, "selected_advantage": selected_adv,
        "corrected": corrected, "introduced": introduced,
        "top_accuracy": top_accuracy, "stop_accuracy": stop_accuracy,
        "execution_case": execution_case, "execution_cell": execution_cell,
        "selected_precision": selected_precision, "selected_harm": selected_harm,
        "selected_mean_benefit": selected_mean_benefit,
        "selected_mean_harm": selected_mean_harm,
        "benefit_recall": benefit_recall,
        "candidate_benefit_rate": beneficial_any.float().mean(),
        "no_op_rate": no_op, "teacher_stop_rate": (labels == 0).float().mean(),
        "duplicate_rate": duplicate_rate, "valid_rate": valid_rate,
        "posterior_oracle": source_oracles[1], "proposal_oracle": source_oracles[2],
        "add_oracle": source_oracles[3], "remove_oracle": source_oracles[4],
        "family_mass": torch.stack(family_mass),
    }


def compute_sparc_hr_loss(
    cfg,
    target: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    mode: str,
    ce_weight: float,
    dice_weight: float,
    *,
    epoch: int = 0,
):
    del mode, ce_weight, dice_weight
    required = (
        "sparc_hr_target", "sparc_anchor_prob_hr", "sparc_final_prob_hr",
        "sparc_source_probs_hr", "sparc_router_logits",
        "sparc_router_soft_weights_grid", "sparc_router_selected_source",
        "sparc_learned_proposal_prob_hr", "sparc_typed_gate_logits_hr",
    )
    missing = [key for key in required if key not in aux]
    if missing:
        raise KeyError(f"SPARC-HR3 loss missing outputs: {missing}")
    m1 = _cfg_get(cfg, "M1", None)
    target_lr = _mask4(target)
    target_hr = (_mask4(aux["sparc_hr_target"]) > 0.5).to(aux["sparc_final_prob_hr"])
    geometry_prob = aux["geotopo_geometry_probs"].clamp(EPS, 1.0 - EPS)
    if tuple(target_lr.shape[-2:]) != tuple(geometry_prob.shape[-2:]):
        target_lr = F.interpolate(target_lr, size=geometry_prob.shape[-2:], mode="nearest")
    geometry_loss, geometry_bce, geometry_dice = _seg_loss(geometry_prob, target_lr)
    final_prob = aux["sparc_final_prob_hr"]
    final_loss, final_bce, final_dice = _seg_loss(final_prob, target_hr)
    proposal_loss, proposal_bce, proposal_dice = _seg_loss(
        aux["sparc_learned_proposal_prob_hr"], target_hr
    )

    anchor = aux["sparc_anchor_prob_hr"]
    edit_band = aux.get("sparc_boundary_edit_band_hr")
    if not isinstance(edit_band, torch.Tensor):
        edit_band = torch.ones_like(anchor)
    edit_band = edit_band > 0.5
    anchor_error = ((anchor >= 0.5) != (target_hr >= 0.5)) & edit_band
    preserve_support = (~anchor_error) & edit_band
    pixel_bce = F.binary_cross_entropy(final_prob.clamp(EPS, 1.0 - EPS), target_hr, reduction="none")
    correction_loss = _masked_mean(pixel_bce, anchor_error)
    preserve_loss = _masked_mean(pixel_bce, preserve_support)
    has_error = anchor_error.any().to(final_prob)
    correction_preserve = 0.5 * has_error * correction_loss + (1.0 - 0.5 * has_error) * preserve_loss
    boundary_target = _hard_boundary(target_hr)
    boundary_pred = _boundary(final_prob)
    boundary_loss = _dice_loss(boundary_pred, boundary_target)

    fn = (((anchor < 0.5) & (target_hr >= 0.5)) & edit_band).float()
    fp = (((anchor >= 0.5) & (target_hr < 0.5)) & edit_band).float()
    typed_target = torch.cat([fn, fp], dim=1)
    typed_logits = aux["sparc_typed_gate_logits_hr"]
    typed_gate_loss = _balanced_binary_logits(typed_logits[:, :1], fn)
    typed_gate_loss = typed_gate_loss + _balanced_binary_logits(typed_logits[:, 1:2], fp)
    typed_gate_loss = 0.5 * typed_gate_loss
    crossing = float(_cfg_get(m1, "GEOTR_SPARC_MINIMUM_CROSSING_MARGIN", 0.05))
    add_violation = F.relu(0.5 + crossing - aux["sparc_typed_add_prob_hr"])
    remove_violation = F.relu(aux["sparc_typed_remove_prob_hr"] - (0.5 - crossing))
    predicted_dose = aux["sparc_typed_dose_hr"]
    # The typed construction already guarantees threshold reachability for
    # every dose.  A crossing-only objective is therefore identically zero and
    # would leave ``typed_dose_head`` dead.  Supervise the *smallest* sufficient
    # intervention instead: dose=0 gives exactly 0.55/0.45, while positive
    # dose is allowed by the actor but explicitly priced as excess magnitude.
    crossing_loss = _masked_mean(add_violation, fn > 0.5)
    crossing_loss = crossing_loss + _masked_mean(remove_violation, fp > 0.5)
    minimum_add_dose = _masked_mean(predicted_dose[:, :1], fn > 0.5)
    minimum_remove_dose = _masked_mean(predicted_dose[:, 1:2], fp > 0.5)
    typed_dose_loss = 0.5 * crossing_loss + 0.5 * (
        minimum_add_dose + minimum_remove_dose
    )
    excess = _masked_mean((aux["sparc_typed_add_prob_hr"] - anchor).abs(), fn < 0.5)
    excess = excess + _masked_mean((aux["sparc_typed_remove_prob_hr"] - anchor).abs(), fp < 0.5)
    typed_excess = 0.5 * excess
    typed_loss = typed_gate_loss + 0.5 * typed_dose_loss + 0.25 * typed_excess

    teacher = _router_teacher(cfg, target_hr, aux)
    router_logits = aux["sparc_router_logits"]
    temperature = max(0.05, float(_cfg_get(m1, "GEOTR_SPARC_ROUTER_TEMPERATURE", 0.5)))
    policy_logits = router_logits / temperature
    soft = F.softmax(policy_logits, dim=1)
    # Use the exact same temperature coordinate for supervision, diagnostics
    # and the deployment confidence gate.
    ce_map = F.cross_entropy(policy_logits, teacher["teacher"], reduction="none")
    target_probability = soft.gather(1, teacher["teacher"][:, None])[:, 0]
    focal_gamma = max(0.0, float(_cfg_get(m1, "GEOTR_SPARC_ROUTER_FOCAL_GAMMA", 2.0)))
    ce_map = (1.0 - target_probability).pow(focal_gamma) * ce_map
    actionable_weight = min(
        0.8, max(0.05, float(_cfg_get(m1, "GEOTR_SPARC_ROUTER_ACTIONABLE_WEIGHT", 0.40)))
    )
    router_ce = actionable_weight * _masked_mean(ce_map, teacher["actionable"])
    router_ce = router_ce + (1.0 - actionable_weight) * _masked_mean(
        ce_map, ~teacher["actionable"]
    )
    expected = (soft * teacher["advantage"]).sum(dim=1)
    regret_map = (teacher["best_advantage"] - expected).clamp_min(0.0)
    router_regret = actionable_weight * _masked_mean(regret_map, teacher["actionable"])
    router_regret = router_regret + (1.0 - actionable_weight) * _masked_mean(
        regret_map, ~teacher["actionable"]
    )
    router_harm = (soft[:, 1:] * (-teacher["advantage"][:, 1:]).clamp_min(0.0)).sum(dim=1).mean()
    tv_h = (soft[:, :, 1:] - soft[:, :, :-1]).abs().mean()
    tv_w = (soft[:, :, :, 1:] - soft[:, :, :, :-1]).abs().mean()
    router_tv = tv_h + tv_w

    start_epoch = max(0, int(_cfg_get(m1, "GEOTR_SPARC_ROUTER_START_EPOCH", 2)))
    ramp_epochs = max(1, int(_cfg_get(m1, "GEOTR_SPARC_LOSS_RAMP_EPOCHS", 4)))
    ramp = 0.0 if epoch < start_epoch else min(
        1.0, float(epoch - start_epoch + 1) / float(ramp_epochs)
    )
    shares = {
        "geometry": float(_cfg_get(m1, "GEOTR_SPARC_OBJECTIVE_GEOMETRY_SHARE", 0.30)),
        "final": float(_cfg_get(m1, "GEOTR_SPARC_OBJECTIVE_FINAL_SHARE", 0.15)),
        "local": float(_cfg_get(m1, "GEOTR_SPARC_OBJECTIVE_LOCAL_SHARE", 0.10)),
        "boundary": float(_cfg_get(m1, "GEOTR_SPARC_OBJECTIVE_BOUNDARY_SHARE", 0.05)),
        "proposal": float(_cfg_get(m1, "GEOTR_SPARC_OBJECTIVE_PROPOSAL_SHARE", 0.07)),
        "typed": float(_cfg_get(m1, "GEOTR_SPARC_OBJECTIVE_TYPED_SHARE", 0.08)),
        "router_ce": float(_cfg_get(m1, "GEOTR_SPARC_OBJECTIVE_ROUTER_CE_SHARE", 0.15)),
        "regret": float(_cfg_get(m1, "GEOTR_SPARC_OBJECTIVE_REGRET_SHARE", 0.07)),
        "harm": float(_cfg_get(m1, "GEOTR_SPARC_OBJECTIVE_HARM_SHARE", 0.03)),
    }
    total_share = max(EPS, sum(max(0.0, value) for value in shares.values()))
    shares = {key: max(0.0, value) / total_share for key, value in shares.items()}
    objective = shares["geometry"] * geometry_loss + ramp * (
        shares["final"] * final_loss
        + shares["local"] * correction_preserve
        + shares["boundary"] * boundary_loss
        + shares["proposal"] * proposal_loss
        + shares["typed"] * typed_loss
        + shares["router_ce"] * router_ce
        + shares["regret"] * router_regret
        + shares["harm"] * router_harm
        + float(_cfg_get(m1, "GEOTR_SPARC_ROUTER_TV_WEIGHT", 0.005)) * router_tv
    )

    stats = _routing_statistics(cfg, target_hr, aux)
    selected_positive = (stats["selected_advantage"] > 0.0)
    selected_negative = (stats["selected_advantage"] < 0.0)
    realized = stats["quality_gain"]
    neutral = float(_cfg_get(m1, "GEOTR_SPARC_NEUTRAL_UTILITY_MARGIN", 2.0e-4))
    typed_pred = torch.sigmoid(typed_logits) >= 0.5
    add_recall = _masked_mean(typed_pred[:, :1].float(), fn > 0.5)
    remove_recall = _masked_mean(typed_pred[:, 1:2].float(), fp > 0.5)
    add_precision = _masked_mean(fn, typed_pred[:, :1])
    remove_precision = _masked_mean(fp, typed_pred[:, 1:2])
    add_f1 = 2 * add_precision * add_recall / (add_precision + add_recall + EPS)
    remove_f1 = 2 * remove_precision * remove_recall / (remove_precision + remove_recall + EPS)
    posterior = aux["sparc_posterior_views_hr"]
    posterior_diversity = posterior.std(dim=1, unbiased=False).mean()
    zero = objective.new_zeros(())
    metrics = {
        "sparc_hr_enabled": objective.new_ones(()), "sparc_hr_protocol_version": objective.new_tensor(3.1),
        "sparc_hr_loss_ramp": objective.new_tensor(ramp),
        "sparc_hr_objective": objective.detach(),
        "sparc_hr_objective_geometry_share": objective.new_tensor(shares["geometry"]),
        "sparc_hr_objective_router_share": objective.new_tensor(
            shares["router_ce"] + shares["regret"] + shares["harm"]
        ),
        "sparc_hr_final_loss": final_loss.detach(), "sparc_hr_final_bce": final_bce.detach(),
        "sparc_hr_final_dice_loss": final_dice.detach(),
        "sparc_hr_geometry_loss": geometry_loss.detach(), "sparc_hr_geometry_bce": geometry_bce.detach(),
        "sparc_hr_geometry_dice_loss": geometry_dice.detach(),
        "sparc_hr_proposal_loss": proposal_loss.detach(), "sparc_hr_proposal_bce": proposal_bce.detach(),
        "sparc_hr_proposal_dice_loss": proposal_dice.detach(),
        "sparc_hr_typed_loss": typed_loss.detach(), "sparc_hr_typed_bce": typed_gate_loss.detach(),
        "sparc_hr_typed_tversky_loss": typed_gate_loss.detach(),
        "sparc_hr_typed_excess_mass_loss": typed_excess.detach(),
        "sparc_hr_typed_dice_loss": typed_gate_loss.detach(),
        "sparc_hr_typed_minimum_dose_loss": typed_dose_loss.detach(),
        "sparc_hr_router_ce_loss": router_ce.detach(), "sparc_hr_router_regret_loss": router_regret.detach(),
        "sparc_hr_router_harm_loss": router_harm.detach(), "sparc_hr_router_tv_loss": router_tv.detach(),
        "sparc_hr_correction_loss": correction_loss.detach(), "sparc_hr_preserve_loss": preserve_loss.detach(),
        "sparc_hr_boundary_loss": boundary_loss.detach(),
        # Compatibility aliases keep the central logger stable while exposing HR3 semantics.
        "sparc_hr_selector_loss": router_ce.detach(), "sparc_hr_selector_bce": router_ce.detach(),
        "sparc_hr_selector_exact_bce": router_ce.detach(), "sparc_hr_selector_rank_loss": router_regret.detach(),
        "sparc_hr_utility_outcome_loss": router_harm.detach(), "sparc_hr_utility_gain_loss": router_regret.detach(),
        "sparc_hr_utility_quantile_loss": zero, "sparc_hr_utility_risk_loss": router_harm.detach(),
        "sparc_hr_policy_listwise_loss": router_ce.detach(), "sparc_hr_utility_regression_loss": router_regret.detach(),
        "sparc_hr_utility_sign_loss": router_harm.detach(), "sparc_hr_utility_rank_loss": router_ce.detach(),
        "sparc_hr_policy_top_choice_accuracy": stats["top_accuracy"].detach(),
        "sparc_hr_policy_stop_accuracy": stats["stop_accuracy"].detach(),
        "sparc_hr_teacher_stop_rate": stats["teacher_stop_rate"].detach(),
        "sparc_hr_lcb_positive_precision": stats["selected_precision"].detach(),
        "sparc_hr_selected_lcb": stats["selected_advantage"].mean().detach(),
        "sparc_hr_anchor_quality": stats["anchor_quality"].mean().detach(),
        "sparc_hr_final_quality": stats["final_quality"].mean().detach(),
        "sparc_hr_proposal_quality": stats["proposal_quality"].mean().detach(),
        "sparc_hr_realized_gain": realized.mean().detach(),
        "sparc_hr_oracle_atomic_gain": stats["oracle_gain"].mean().detach(),
        "sparc_hr_on_policy_oracle_gain": stats["oracle_gain"].mean().detach(),
        "sparc_hr_selected_utility": stats["selected_advantage"].mean().detach(),
        "sparc_hr_policy_regret": (stats["oracle_gain"] - realized).clamp_min(0.0).mean().detach(),
        "sparc_hr_oracle_capture_ratio": (realized / stats["oracle_gain"].clamp_min(EPS)).clamp(-2, 2).mean().detach(),
        "sparc_hr_beneficial_candidate_rate": stats["candidate_benefit_rate"].detach(),
        "sparc_hr_execution_rate": stats["execution_case"].detach(),
        "sparc_hr_step_execution_rate": stats["execution_cell"].detach(),
        "sparc_hr_benefit_case_rate": (realized > neutral).float().mean().detach(),
        "sparc_hr_harm_case_rate": (realized < -neutral).float().mean().detach(),
        "sparc_hr_corrected_count": stats["corrected"].detach(),
        "sparc_hr_introduced_count": stats["introduced"].detach(),
        "sparc_hr_net_correction_count": (stats["corrected"] - stats["introduced"]).detach(),
        "sparc_hr_selected_action_precision": stats["selected_precision"].detach(),
        "sparc_hr_selected_action_harm_rate": stats["selected_harm"].detach(),
        "sparc_hr_posterior_only_oracle_gain": stats["posterior_oracle"].mean().detach(),
        "sparc_hr_proposal_only_oracle_gain": stats["proposal_oracle"].mean().detach(),
        "sparc_hr_add_only_oracle_gain": stats["add_oracle"].mean().detach(),
        "sparc_hr_remove_only_oracle_gain": stats["remove_oracle"].mean().detach(),
        "sparc_hr_no_op_candidate_rate": stats["no_op_rate"].detach(),
        "sparc_hr_duplicate_candidate_rate": stats["duplicate_rate"].detach(),
        "sparc_hr_valid_candidate_rate": stats["valid_rate"].detach(),
        "sparc_hr_typed_add_recall": add_recall.detach(), "sparc_hr_typed_remove_recall": remove_recall.detach(),
        "sparc_hr_typed_add_precision": add_precision.detach(), "sparc_hr_typed_remove_precision": remove_precision.detach(),
        "sparc_hr_typed_add_f1": add_f1.detach(), "sparc_hr_typed_remove_f1": remove_f1.detach(),
        "sparc_hr_posterior_diversity": posterior_diversity.detach(),
        "sparc_hr_router_anchor_mass": stats["family_mass"][0].detach(),
        "sparc_hr_router_posterior_mass": stats["family_mass"][1].detach(),
        "sparc_hr_router_proposal_mass": stats["family_mass"][2].detach(),
        "sparc_hr_router_add_mass": stats["family_mass"][3].detach(),
        "sparc_hr_router_remove_mass": stats["family_mass"][4].detach(),
        "sparc_hr_router_actionable_rate": teacher["actionable"].float().mean().detach(),
    }
    return objective, metrics


@torch.no_grad()
def compute_sparc_hr_validation_diagnostics(
    cfg, target_hr: torch.Tensor, pred: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    required = (
        "sparc_anchor_prob_hr", "sparc_final_prob_hr", "sparc_source_probs_hr",
        "sparc_router_logits", "sparc_router_selected_source", "sparc_source_family_ids",
    )
    if any(key not in pred for key in required):
        return {}
    target = (_mask4(target_hr) > 0.5).to(pred["sparc_final_prob_hr"])
    stats = _routing_statistics(cfg, target, pred)
    gain = stats["quality_gain"]
    oracle = stats["oracle_gain"]
    neutral = float(_cfg_get(_cfg_get(cfg, "M1", None), "GEOTR_SPARC_NEUTRAL_UTILITY_MARGIN", 2.0e-4))
    edited = pred["sparc_router_selected_source"] != 0
    selected = stats["selected_advantage"]
    values = {
        "val_sparc_hr_quality_gain": gain.mean(),
        "val_sparc_hr_atomic_oracle_gain": oracle.mean(),
        "val_sparc_hr_on_policy_oracle_gain": oracle.mean(),
        "val_sparc_hr_learner_state_oracle_sum": oracle.mean(),
        "val_sparc_hr_true_sequential_oracle_gain": oracle.mean(),
        "val_sparc_hr_selected_utility": selected.mean(),
        "val_sparc_hr_policy_regret": (oracle - gain).clamp_min(0.0).mean(),
        "val_sparc_hr_policy_top_choice_accuracy": stats["top_accuracy"],
        "val_sparc_hr_policy_stop_accuracy": stats["stop_accuracy"],
        "val_sparc_hr_selected_action_precision": stats["selected_precision"],
        "val_sparc_hr_selected_action_harm_rate": stats["selected_harm"],
        "val_sparc_hr_selected_mean_benefit": stats["selected_mean_benefit"],
        "val_sparc_hr_selected_mean_harm_magnitude": stats["selected_mean_harm"],
        "val_sparc_hr_candidate_benefit_rate": stats["candidate_benefit_rate"],
        "val_sparc_hr_no_op_candidate_rate": stats["no_op_rate"],
        "val_sparc_hr_duplicate_candidate_rate": stats["duplicate_rate"],
        "val_sparc_hr_valid_candidate_rate": stats["valid_rate"],
        "val_sparc_hr_execution_rate": stats["execution_case"],
        "val_sparc_hr_step_execution_rate": stats["execution_cell"],
        "val_sparc_hr_benefit_case_rate": (gain > neutral).float().mean(),
        "val_sparc_hr_harm_case_rate": (gain < -neutral).float().mean(),
        "val_sparc_hr_corrected_count": stats["corrected"],
        "val_sparc_hr_introduced_count": stats["introduced"],
        "val_sparc_hr_net_correction_count": stats["corrected"] - stats["introduced"],
        "val_sparc_hr_teacher_stop_rate": stats["teacher_stop_rate"],
        "val_sparc_hr_lcb_positive_precision": stats["selected_precision"],
        "val_sparc_hr_quantile_positive_precision": stats["selected_precision"],
        "val_sparc_hr_benefit_precision": stats["selected_precision"],
        "val_sparc_hr_benefit_recall": stats["benefit_recall"],
        "val_sparc_hr_quantile_recall": stats["benefit_recall"],
        "val_sparc_hr_joint_precision": stats["selected_precision"],
        "val_sparc_hr_joint_recall": stats["benefit_recall"],
        "val_sparc_hr_policy_gate_pass_rate": stats["execution_cell"],
        "val_sparc_hr_benefit_candidate_pass_rate": stats["candidate_benefit_rate"],
        "val_sparc_hr_quantile_candidate_pass_rate": stats["candidate_benefit_rate"],
        "val_sparc_hr_joint_candidate_pass_rate": stats["candidate_benefit_rate"],
        "val_sparc_hr_posterior_only_oracle_gain": stats["posterior_oracle"].mean(),
        "val_sparc_hr_proposal_only_oracle_gain": stats["proposal_oracle"].mean(),
        "val_sparc_hr_add_only_oracle_gain": stats["add_oracle"].mean(),
        "val_sparc_hr_remove_only_oracle_gain": stats["remove_oracle"].mean(),
        "val_sparc_hr_true_oracle_execution_rate": stats["actionable"].float().mean(),
        "val_sparc_hr_true_oracle_step1_gain": oracle.mean(),
        "val_sparc_hr_true_oracle_step2_gain": gain.new_zeros(()),
        "val_sparc_hr_true_oracle_step3_gain": gain.new_zeros(()),
        "val_sparc_hr_selected_step1_utility": selected.mean(),
        "val_sparc_hr_selected_step2_utility": gain.new_zeros(()),
        "val_sparc_hr_selected_step3_utility": gain.new_zeros(()),
        "val_sparc_hr_router_actionable_rate": stats["actionable"].float().mean(),
        "val_sparc_hr_router_anchor_mass": stats["family_mass"][0],
        "val_sparc_hr_router_posterior_mass": stats["family_mass"][1],
        "val_sparc_hr_router_proposal_mass": stats["family_mass"][2],
        "val_sparc_hr_router_add_mass": stats["family_mass"][3],
        "val_sparc_hr_router_remove_mass": stats["family_mass"][4],
    }
    return values
