"""SemLT-LST v3/v3.1 + eligibility-conditioned gate root-fix loss.

V3.1 fixes the over-edit failure mode by separating *selection* from *correction*:

1. The deployment gate is supervised only by a GT-derived edit-benefit target.
2. The transport editor is trained through a GT-derived teacher gate on genuinely
   correctable pixels, so a conservative deployment gate cannot starve learning.
3. Deployment is physically sparse/hard and restricted to the local transition
   band; KEEP therefore means exact Base logits.
4. Safety is harm-only: improvements are never penalized merely for differing
   from Base, while degradations relative to Base are penalized.

GT is used only to construct training targets. It is never consumed by generate()
or by test-time inference.

Eligibility-conditioned gate extension:
- the physical deployment domain is the deterministic transition band;
- the gate is therefore trained as P(beneficial | transition-eligible), not on
  pixels that can never execute a transport action;
- the 0.5 deployment threshold remains fixed and is not tuned on Val/Test.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import math
import torch
import torch.nn.functional as F

from utils.semlt_sdf_geometry import gather_owner_field, operator_matched_normal_ray_target

EPS = 1.0e-6


def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def _target3(masks: torch.Tensor, hw) -> torch.Tensor:
    target = masks.float()
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    elif target.ndim != 3:
        raise ValueError(f"Expected masks [B,H,W] or [B,1,H,W], got {tuple(target.shape)}")
    if tuple(target.shape[-2:]) != tuple(hw):
        target = F.interpolate(target[:, None], size=hw, mode="nearest")[:, 0]
    return target.clamp(0.0, 1.0)


def _dice_per_case(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if probability.ndim == 4:
        probability = probability[:, 0]
    p, t = probability.flatten(1), target.flatten(1)
    return (2.0 * (p * t).sum(1) + EPS) / (p.sum(1) + t.sum(1) + EPS)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(value)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def _case_balanced_masked_mean(
    value: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Average valid pixels per case first, then average valid cases.

    Contour length varies by more than an order of magnitude across BUSI,
    BTMRI, ISIC and Kvasir. A global owner-pixel mean therefore lets the
    largest objects dominate correspondence learning. This helper gives each
    image with at least one valid owner equal weight and keeps the legitimate
    all-empty case differentiable with an exact zero.
    """
    if value.shape != mask.shape:
        raise ValueError(
            "case-balanced masked mean requires equal shapes, got "
            f"value={tuple(value.shape)} mask={tuple(mask.shape)}"
        )
    if value.ndim < 1:
        raise ValueError("case-balanced masked mean requires a batch dimension")
    weight = mask.to(value)
    value_flat = value.reshape(value.shape[0], -1)
    weight_flat = weight.reshape(weight.shape[0], -1)
    count = weight_flat.sum(dim=1)
    valid_case = count > 0
    per_case = (value_flat * weight_flat).sum(dim=1) / count.clamp_min(1.0)
    if bool(valid_case.any()):
        return per_case[valid_case].mean()
    return value.sum() * 0.0


def _class_case_balanced_masked_mean(
    value: torch.Tensor,
    target_class: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Equalize present target classes without sacrificing case balance."""
    if target_class.shape != value.shape or mask.shape != value.shape:
        raise ValueError("class/case-balanced mean expects matching [B,...] tensors")
    terms: List[torch.Tensor] = []
    for class_index in range(int(num_classes)):
        class_mask = mask & (target_class == class_index)
        if bool(class_mask.any()):
            terms.append(_case_balanced_masked_mean(value, class_mask))
    return torch.stack(terms).mean() if terms else value.sum() * 0.0


def _proper_sign_log_score(
    sign_probs: torch.Tensor,
    sign_target_prob: torch.Tensor,
    mask: torch.Tensor,
    case_balanced: bool,
) -> torch.Tensor:
    """Proper three-way sign log score under the selected owner measure.

    The returned probabilities are consumed as a posterior by the deployed
    expectation decoder.  Consequently NEGATIVE/KEEP/POSITIVE must not be
    equalized by target class here: class equalization changes the effective
    class prior and makes the probabilities unsuitable for posterior-mean
    decoding.  ``case_balanced`` changes only the case sampling measure.
    """
    if sign_probs.ndim != 4 or int(sign_probs.shape[1]) != 3:
        raise ValueError("sign_probs must be [B,3,H,W]")
    if sign_target_prob.shape != sign_probs.shape:
        raise ValueError("sign_target_prob must match sign_probs")
    if mask.shape != sign_probs.shape[:1] + sign_probs.shape[2:]:
        raise ValueError("sign-score mask must be [B,H,W]")
    target = sign_target_prob.to(sign_probs)
    nll_map = -(target * sign_probs.float().clamp_min(EPS).log()).sum(dim=1)
    reducer = _case_balanced_masked_mean if case_balanced else _masked_mean
    return reducer(nll_map.to(sign_probs), mask)


def _balanced_binary_bce(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Balanced auxiliary discriminator; never interpreted as a posterior."""
    loss = F.binary_cross_entropy_with_logits(
        logit.float(), target.float(), reduction="none"
    ).to(logit)
    terms: List[torch.Tensor] = []
    for value in (False, True):
        mask = target == value
        if bool(mask.any()):
            terms.append(loss[mask].mean())
    return torch.stack(terms).mean() if terms else loss.mean()


def _binary_rank_metrics_histogram(
    probability: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    bins: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Approximate AUROC/AUPRC on a masked binary decision domain.

    This is diagnostics-only.  A fixed histogram avoids sorting tens of
    thousands of dense pixels every batch and introduces no trainable or
    dataset-tuned hyperparameter.  The returned validity flag is one only when
    both positive and negative examples exist in the masked domain.
    """
    with torch.no_grad():
        p = probability.detach().float()[mask].clamp(0.0, 1.0)
        y = target.detach().bool()[mask]
        zero = probability.detach().new_zeros(())
        if p.numel() == 0:
            return zero, zero, zero
        pos_total = y.float().sum()
        neg_total = (~y).float().sum()
        if float(pos_total) <= 0.0 or float(neg_total) <= 0.0:
            return zero, zero, zero
        bins = max(int(bins), 8)
        index = torch.floor(p * float(bins - 1)).long().clamp(0, bins - 1)
        pos_hist = torch.bincount(index[y], minlength=bins).float()
        neg_hist = torch.bincount(index[~y], minlength=bins).float()
        tp = torch.flip(pos_hist, dims=(0,)).cumsum(0)
        fp = torch.flip(neg_hist, dims=(0,)).cumsum(0)
        recall = tp / pos_total.clamp_min(1.0)
        precision = tp / (tp + fp).clamp_min(1.0)
        recall_prev = torch.cat([recall.new_zeros(1), recall[:-1]])
        auprc = ((recall - recall_prev).clamp_min(0.0) * precision).sum()
        tpr = recall
        fpr = fp / neg_total.clamp_min(1.0)
        tpr_curve = torch.cat([tpr.new_zeros(1), tpr])
        fpr_curve = torch.cat([fpr.new_zeros(1), fpr])
        auroc = torch.trapz(tpr_curve, fpr_curve).clamp(0.0, 1.0)
        return auroc.to(probability), auprc.to(probability), probability.detach().new_ones(())




def _marginal_hard_dice_utility(
    base_hard: torch.Tensor,
    candidate_hard: torch.Tensor,
    target_hard: torch.Tensor,
) -> torch.Tensor:
    """Exact one-pixel marginal Dice value of candidate vs KEEP.

    For every pixel, compute the Dice score that would result if *only that
    pixel* changed from the factual Base hard decision to the candidate hard
    decision while all other pixels stayed fixed.  This yields a task-aligned,
    parameter-free signed value: positive=improves Dice, negative=harms Dice,
    zero=no mask-level effect.
    """
    p = base_hard.float()
    q = candidate_hard.float()
    t = target_hard.float()
    intersection = (p * t).flatten(1).sum(1)
    pred_mass = p.flatten(1).sum(1)
    target_mass = t.flatten(1).sum(1)
    numerator = (2.0 * intersection + EPS)[:, None, None]
    denominator = (pred_mass + target_mass + EPS)[:, None, None]
    delta = q - p
    candidate_num = numerator + 2.0 * delta * t
    candidate_den = (denominator + delta).clamp_min(EPS)
    base_dice = numerator / denominator
    candidate_dice = candidate_num / candidate_den
    return candidate_dice - base_dice


def _value_weighted_policy_loss(
    edit_logit: torch.Tensor,
    signed_value: torch.Tensor,
    eligible: torch.Tensor,
) -> torch.Tensor:
    """Outcome-weighted logistic policy loss on the executable domain.

    Minimizing E[|V| softplus(-sign(V) s)] makes the sign of the optimal score
    follow the sign of conditional expected value.  Therefore deployment at
    score>=0 (sigmoid>=0.5) is a value decision, not an arbitrary probability
    calibration threshold.  Zero-value actions carry zero policy weight.
    """
    value = signed_value.detach().to(edit_logit)
    domain = eligible.bool()
    nonzero = domain & (value.abs() > 0.0)
    if not bool(nonzero.any()):
        return edit_logit.sum() * 0.0
    sign = torch.where(value > 0.0, torch.ones_like(value), -torch.ones_like(value))
    weight = value.abs()
    per_pixel = F.softplus(-sign * edit_logit.float()).to(edit_logit)
    return (per_pixel[nonzero] * weight[nonzero]).sum() / weight[nonzero].sum().clamp_min(EPS)

def _mean_aux(aux: Dict[str, torch.Tensor], key: str, reference: torch.Tensor) -> torch.Tensor:
    value = aux.get(key)
    if not isinstance(value, torch.Tensor):
        return reference.detach().new_zeros(())
    return value.float().mean().to(reference).detach()


def _compute_v31_rootfix_loss(
    cfg: Any,
    masks: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    epoch: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    keys = (
        "geotr_m1_final_logits", "geotr_m1_base_logits",
        "geotr_m1_edit_logit", "geotr_m1_locator_logit", "geotr_m1_type_logits",
        "geotr_m1_add_valid_source_mass", "geotr_m1_remove_valid_source_mass",
        "geotr_m1_add_source_exists", "geotr_m1_remove_source_exists",
        "geotr_m1_add_logits", "geotr_m1_remove_logits", "geotr_m1_edited_logits",
        "geotr_m1_change_gate", "geotr_m1_masked_edit_probability",
        "geotr_m1_transition_band", "geotr_m1_local_radius_px",
    )
    if any(not isinstance(aux.get(key), torch.Tensor) for key in keys):
        missing = [key for key in keys if not isinstance(aux.get(key), torch.Tensor)]
        raise RuntimeError(f"SemLT-LST v3.1 loss is missing statistics: {missing}")

    m1_cfg = _cfg_get(cfg, "M1", None)
    utility_margin = float(_cfg_get(m1_cfg, "SEMLT_UTILITY_MARGIN", 0.0))
    gate_warmup_epochs = int(_cfg_get(m1_cfg, "SEMLT_GATE_TEACHER_WARMUP_EPOCHS", 3))
    gate_loss_domain = str(
        _cfg_get(m1_cfg, "SEMLT_GATE_LOSS_DOMAIN", "whole_image")
    ).strip().lower()
    eligible_gate_rootfix = bool(
        _cfg_get(m1_cfg, "SEMLT_LST_ELIGIBLE_GATE_ROOTFIX", False)
    )
    action_value_policy = bool(
        _cfg_get(m1_cfg, "SEMLT_ACTION_CONSISTENT_VALUE_POLICY", False)
    )
    if utility_margin < 0.0:
        raise ValueError("M1.SEMLT_UTILITY_MARGIN must be >= 0")
    if gate_warmup_epochs < 0:
        raise ValueError("M1.SEMLT_GATE_TEACHER_WARMUP_EPOCHS must be >= 0")
    if gate_loss_domain not in {"whole_image", "transition_eligible"}:
        raise ValueError(
            "M1.SEMLT_GATE_LOSS_DOMAIN must be whole_image or transition_eligible"
        )
    if eligible_gate_rootfix and gate_loss_domain != "transition_eligible":
        raise ValueError(
            "SEMLT_LST_ELIGIBLE_GATE_ROOTFIX requires "
            "SEMLT_GATE_LOSS_DOMAIN=transition_eligible"
        )

    deploy_logits = aux["geotr_m1_final_logits"]
    base_logits = aux["geotr_m1_base_logits"]
    edit_logit = aux["geotr_m1_edit_logit"]
    locator_logit = aux["geotr_m1_locator_logit"]
    type_logits = aux["geotr_m1_type_logits"]
    add_logits = aux["geotr_m1_add_logits"]
    remove_logits = aux["geotr_m1_remove_logits"]
    edited_logits = aux["geotr_m1_edited_logits"]
    physical_gate = aux["geotr_m1_change_gate"]
    gate_probability = aux["geotr_m1_masked_edit_probability"]
    transition_band = aux["geotr_m1_transition_band"].detach().bool()
    add_valid_source_mass = aux["geotr_m1_add_valid_source_mass"]
    remove_valid_source_mass = aux["geotr_m1_remove_valid_source_mass"]
    add_source_exists = aux["geotr_m1_add_source_exists"].detach().bool()
    remove_source_exists = aux["geotr_m1_remove_source_exists"].detach().bool()

    for name in ("deploy_logits", "base_logits", "add_logits", "remove_logits", "physical_gate", "gate_probability"):
        value = locals()[name]
        if value.ndim == 3:
            locals()[name] = value[:, None]
    if deploy_logits.ndim == 3:
        deploy_logits = deploy_logits[:, None]
    if base_logits.ndim == 3:
        base_logits = base_logits[:, None]
    if add_logits.ndim == 3:
        add_logits = add_logits[:, None]
    if remove_logits.ndim == 3:
        remove_logits = remove_logits[:, None]
    if edited_logits.ndim == 3:
        edited_logits = edited_logits[:, None]
    if physical_gate.ndim == 3:
        physical_gate = physical_gate[:, None]
    if gate_probability.ndim == 3:
        gate_probability = gate_probability[:, None]

    target = _target3(masks, deploy_logits.shape[-2:]).to(deploy_logits)
    target_hard = target >= 0.5
    base_probability = torch.sigmoid(base_logits.detach())
    base_hard = base_probability[:, 0] >= 0.5
    base_error = base_hard != target_hard

    source_exists = torch.where(target_hard, add_source_exists, remove_source_exists)
    correctable = base_error & source_exists

    # Counterfactual action target: GT chooses only the *training-time action
    # semantics* (ADD for foreground, REMOVE for background); the source itself
    # is still predicted from Base/image/semantic evidence.
    action_logits = torch.where(target_hard[:, None], add_logits, remove_logits)
    base_bce_map = F.binary_cross_entropy_with_logits(
        base_logits[:, 0].detach().float(), target.float(), reduction="none"
    ).to(deploy_logits)
    action_bce_map = F.binary_cross_entropy_with_logits(
        action_logits[:, 0].float(), target.float(), reduction="none"
    ).to(deploy_logits)
    # Deployment-exact counterfactual candidate.  In the action-value policy
    # route, geotr_m1_edited_logits is the hard predicted ADD/REMOVE action,
    # exactly matching inference rather than a soft mixture of opposite actions.
    edited_bce_map = F.binary_cross_entropy_with_logits(
        edited_logits[:, 0].float(), target.float(), reduction="none"
    ).to(deploy_logits)
    candidate_hard = torch.sigmoid(edited_logits.detach())[:, 0] >= 0.5

    # Legacy BCE utility remains available for diagnostics/backward-compatible
    # runs, but the action-value policy is supervised by exact marginal hard-Dice
    # value because DSC is the deployed segmentation objective.  This avoids
    # labeling an infinitesimal confidence-only BCE improvement as equivalent to
    # a true mask correction.
    bce_utility_map = (base_bce_map - edited_bce_map).detach()
    dice_value_map = _marginal_hard_dice_utility(
        base_hard.detach(), candidate_hard.detach(), target_hard.detach()
    ).detach()
    teacher_candidate_hard = torch.sigmoid(action_logits.detach())[:, 0] >= 0.5
    teacher_dice_value_map = _marginal_hard_dice_utility(
        base_hard.detach(), teacher_candidate_hard, target_hard.detach()
    ).detach()
    teacher_action_utility_map = (base_bce_map - action_bce_map).detach()

    if action_value_policy:
        utility_map = dice_value_map
        beneficial = transition_band & (utility_map > 0.0)
        harmful_candidate = transition_band & (utility_map < 0.0)
        neutral_candidate = transition_band & (utility_map == 0.0)
        gate_target = beneficial
        gate_target_outside_eligible = gate_target & ~transition_band
        if bool(gate_target_outside_eligible.any()):
            raise RuntimeError(
                "SemLT action-value contract violated: positive policy value "
                "outside physical transition eligibility."
            )
        deploy_gate_loss = _value_weighted_policy_loss(
            edit_logit, utility_map, transition_band
        )
        # Brier/BCE are diagnostics only in value-policy mode; the learned score
        # is a decision score and is not claimed to be a calibrated posterior.
        raw_gate_probability = torch.sigmoid(edit_logit)
        deploy_gate_map = F.binary_cross_entropy_with_logits(
            edit_logit.float(), gate_target.float(), reduction="none"
        ).to(deploy_logits)
        gate_brier_map = (raw_gate_probability - gate_target.float()).square()
        gate_brier = _masked_mean(gate_brier_map, transition_band)
        deploy_gate_loss_global = deploy_gate_map.mean().detach()
        gate_brier_global = gate_brier_map.mean().detach()
    else:
        utility_map = bce_utility_map
        beneficial = correctable & (utility_map > utility_margin)
        harmful_candidate = transition_band & (utility_map < -utility_margin)
        neutral_candidate = transition_band & ~(beneficial | harmful_candidate)
        gate_target = correctable if int(epoch) < gate_warmup_epochs else beneficial
        gate_target_outside_eligible = gate_target & ~transition_band
        if eligible_gate_rootfix and bool(gate_target_outside_eligible.any()):
            raise RuntimeError(
                "SemLT eligibility contract violated: a positive gate target lies "
                "outside the transition band, where deployment cannot edit."
            )
        deploy_gate_map = F.binary_cross_entropy_with_logits(
            edit_logit.float(), gate_target.float(), reduction="none"
        ).to(deploy_logits)
        raw_gate_probability = torch.sigmoid(edit_logit)
        gate_brier_map = (raw_gate_probability - gate_target.float()).square()
        if gate_loss_domain == "transition_eligible":
            deploy_gate_loss = _masked_mean(deploy_gate_map, transition_band)
            gate_brier = _masked_mean(gate_brier_map, transition_band)
        else:
            deploy_gate_loss = deploy_gate_map.mean()
            gate_brier = gate_brier_map.mean()
        deploy_gate_loss_global = deploy_gate_map.mean().detach()
        gate_brier_global = gate_brier_map.mean().detach()

    # High-recall representation head remains auxiliary and balanced.
    locator_loss = _balanced_binary_bce(locator_logit, correctable)

    # Editor responsibilities.  GT is a teacher only during training and never
    # enters generate()/inference.  The teacher gate prevents a cautious deploy
    # selector from starving the transport of positive correction examples.
    type_target = (~target_hard).long()  # ADD=0, REMOVE=1
    type_map = F.cross_entropy(type_logits.float(), type_target, reduction="none").to(deploy_logits)
    type_loss = _masked_mean(type_map, correctable)

    valid_source_mass = torch.where(
        target_hard, add_valid_source_mass, remove_valid_source_mass
    ).clamp_min(EPS)
    source_nll = _masked_mean(-torch.log(valid_source_mass), correctable)

    teacher_gate = correctable[:, None].to(action_logits)
    teacher_logits = base_logits.detach() + teacher_gate * (action_logits - base_logits.detach())
    teacher_probability = torch.sigmoid(teacher_logits)
    teacher_bce_map = F.binary_cross_entropy_with_logits(
        teacher_logits[:, 0].float(), target.float(), reduction="none"
    ).to(deploy_logits)
    teacher_bce = teacher_bce_map.mean()
    teacher_dice_loss = (1.0 - _dice_per_case(teacher_probability, target)).mean()
    teacher_segmentation = 0.5 * (teacher_bce + teacher_dice_loss)
    correction_loss = _masked_mean(action_bce_map, correctable)

    # Harm-only deployment safety.  Improvements are not penalized merely for
    # being different from Base; only positive excess loss / Dice degradation is.
    deploy_probability = torch.sigmoid(deploy_logits)
    deploy_bce_map = F.binary_cross_entropy_with_logits(
        deploy_logits[:, 0].float(), target.float(), reduction="none"
    ).to(deploy_logits)
    deploy_bce = deploy_bce_map.mean()
    deploy_dice_loss = (1.0 - _dice_per_case(deploy_probability, target)).mean()
    deploy_segmentation = 0.5 * (deploy_bce + deploy_dice_loss)
    harm_bce = F.relu(deploy_bce_map - base_bce_map).mean()
    base_soft_dice = _dice_per_case(base_probability, target).detach()
    deploy_soft_dice = _dice_per_case(deploy_probability, target)
    nondegradation = F.relu(base_soft_dice - deploy_soft_dice).mean()
    safety_loss = 0.5 * (harm_bce + nondegradation)

    selection_loss = torch.stack([deploy_gate_loss, locator_loss]).mean()
    transport_loss = torch.stack([
        type_loss, source_nll, correction_loss, teacher_segmentation
    ]).mean()
    objective = torch.stack([selection_loss, transport_loss, safety_loss]).mean()

    # Diagnostics are based on the *physical* sparse deployment, not merely the
    # mean sigmoid score.  This makes over-edit directly interpretable.
    hard_edit = physical_gate[:, 0].detach() > 0.5
    predicted_edit = hard_edit.float().mean()
    predicted_probability = gate_probability[:, 0].detach().mean()
    target_edit = gate_target.float().mean().detach()
    correctable_fraction = correctable.float().mean().detach()
    beneficial_fraction = beneficial.float().mean().detach()
    edit_tp = (hard_edit & gate_target).float().sum()
    edit_precision = edit_tp / hard_edit.float().sum().clamp_min(1.0)
    edit_recall = edit_tp / gate_target.float().sum().clamp_min(1.0)
    correctable_recall = (hard_edit & correctable).float().sum() / correctable.float().sum().clamp_min(1.0)
    false_edit_rate = (hard_edit & ~gate_target).float().sum() / (~gate_target).float().sum().clamp_min(1.0)
    transition_coverage = (correctable & transition_band).float().sum() / correctable.float().sum().clamp_min(1.0)
    harmful_edit = hard_edit & (deploy_bce_map.detach() > base_bce_map + 1.0e-8)
    harmful_edit_fraction = harmful_edit.float().sum() / hard_edit.float().sum().clamp_min(1.0)

    # Decision-domain diagnostics.  These distinguish a well-calibrated
    # conditional gate from a globally sparse gate that merely predicts the
    # whole-image class prior.
    eligible_count = transition_band.float().sum().clamp_min(1.0)
    eligible_target_count = (gate_target & transition_band).float().sum()
    eligible_negative = transition_band & ~gate_target
    eligible_predicted_count = (hard_edit & transition_band).float().sum()
    eligible_target_rate = eligible_target_count / eligible_count
    eligible_predicted_rate = eligible_predicted_count / eligible_count
    eligible_overedit_ratio = eligible_predicted_rate / eligible_target_rate.clamp_min(EPS)
    eligible_false_edit_rate = (hard_edit & eligible_negative).float().sum() / eligible_negative.float().sum().clamp_min(1.0)
    positive_probability = _masked_mean(raw_gate_probability.detach(), gate_target & transition_band)
    negative_probability = _masked_mean(raw_gate_probability.detach(), eligible_negative)
    probability_margin = positive_probability - negative_probability
    eligible_auroc, eligible_auprc, eligible_rank_metric_valid = _binary_rank_metrics_histogram(
        raw_gate_probability, gate_target, transition_band
    )
    target_outside_eligible_fraction = gate_target_outside_eligible.float().mean().detach()

    teacher_dice = _dice_per_case(teacher_probability.detach(), target)
    deploy_dice = _dice_per_case(deploy_probability.detach(), target)
    utility_on_correctable = _masked_mean(utility_map, correctable).detach()
    utility_on_beneficial = _masked_mean(utility_map, beneficial).detach()
    teacher_action_utility = _masked_mean(teacher_action_utility_map, correctable).detach()
    predicted_type = type_logits.detach().argmax(dim=1)
    type_target_diag = (~target_hard).long()
    type_accuracy_correctable = _masked_mean(
        (predicted_type == type_target_diag).float(), correctable
    ).detach()
    candidate_changed = candidate_hard != base_hard
    candidate_resolved = base_error & (candidate_hard == target_hard)
    candidate_introduced = (~base_error) & (candidate_hard != target_hard)
    value_nonzero = transition_band & (utility_map != 0.0)
    policy_value_precision = (hard_edit & beneficial).float().sum() / hard_edit.float().sum().clamp_min(1.0)
    policy_value_recall = (hard_edit & beneficial).float().sum() / beneficial.float().sum().clamp_min(1.0)
    policy_negative_execution = (hard_edit & harmful_candidate).float().sum() / hard_edit.float().sum().clamp_min(1.0)
    zero, one = objective.detach().new_zeros(()), objective.detach().new_ones(())

    diagnostics = {
        "mhcs_total_loss": objective.detach(),
        "mhcs_final_loss": deploy_segmentation.detach(),
        "mhcs_ce_loss": deploy_bce.detach(),
        "mhcs_dice_loss": deploy_dice_loss.detach(),
        "mhcs_m1_objective": objective.detach(), "mhcs_m2_objective": zero,
        "mhcs_final_gain": (deploy_dice - base_soft_dice).mean(),
        "mhcs_base_dice": base_soft_dice.mean(), "mhcs_final_dice": deploy_dice.mean(),
        "geotopo_total_loss": objective.detach(),
        "geotopo_final_loss": deploy_segmentation.detach(),
        "geotopo_geometry_loss": transport_loss.detach(),
        "geotopo_geometry_bce_loss": correction_loss.detach(),
        "geotopo_geometry_dice_loss": teacher_dice_loss.detach(),
        "geotopo_geometry_boundary_loss": zero,
        "geotopo_reconstruction_loss": safety_loss.detach(),
        "geotopo_flow_smoothness": zero, "geotopo_flow_folding_penalty": zero,
        "geotr_m1_total_loss": objective.detach(),
        "geotr_m1_segmentation_loss": teacher_segmentation.detach(),
        "geotr_m1_deploy_segmentation_loss": deploy_segmentation.detach(),
        "geotr_m1_bce_loss": deploy_bce.detach(),
        "geotr_m1_dice_loss": deploy_dice_loss.detach(),
        "geotr_m1_boundary_loss": zero,
        "geotr_m1_state_loss": selection_loss.detach(),
        "geotr_m1_deploy_gate_loss": deploy_gate_loss.detach(),
        "geotr_m1_locator_loss": locator_loss.detach(),
        "geotr_m1_type_loss": type_loss.detach(),
        "geotr_m1_gate_brier": gate_brier.detach(),
        "geotr_m1_gate_brier_global": gate_brier_global,
        "geotr_m1_deploy_gate_loss_global": deploy_gate_loss_global,
        "geotr_m1_gate_domain_transition_eligible": objective.detach().new_tensor(
            float(gate_loss_domain == "transition_eligible")
        ),
        "geotr_m1_gate_domain_fraction": transition_band.float().mean().detach()
            if gate_loss_domain == "transition_eligible" else objective.detach().new_ones(()),
        "geotr_m1_gate_target_outside_eligible_fraction": target_outside_eligible_fraction,
        "geotr_m1_gate_target_rate_eligible": eligible_target_rate.detach(),
        "geotr_m1_gate_predicted_rate_eligible": eligible_predicted_rate.detach(),
        "geotr_m1_gate_overedit_ratio_eligible": eligible_overedit_ratio.detach(),
        "geotr_m1_gate_false_edit_rate_eligible": eligible_false_edit_rate.detach(),
        "geotr_m1_gate_positive_probability_eligible": positive_probability.detach(),
        "geotr_m1_gate_negative_probability_eligible": negative_probability.detach(),
        "geotr_m1_gate_probability_margin_eligible": probability_margin.detach(),
        "geotr_m1_gate_auroc_eligible": eligible_auroc.detach(),
        "geotr_m1_gate_auprc_eligible": eligible_auprc.detach(),
        "geotr_m1_gate_rank_metric_valid": eligible_rank_metric_valid.detach(),
        "geotr_m1_action_value_policy": objective.detach().new_tensor(float(action_value_policy)),
        "geotr_m1_policy_value_mean_eligible": _masked_mean(utility_map, transition_band).detach(),
        "geotr_m1_policy_value_positive_rate_eligible": (beneficial.float().sum() / eligible_count).detach(),
        "geotr_m1_policy_value_negative_rate_eligible": (harmful_candidate.float().sum() / eligible_count).detach(),
        "geotr_m1_policy_value_nonzero_rate_eligible": (value_nonzero.float().sum() / eligible_count).detach(),
        "geotr_m1_policy_value_precision": policy_value_precision.detach(),
        "geotr_m1_policy_value_recall": policy_value_recall.detach(),
        "geotr_m1_policy_negative_execution_fraction": policy_negative_execution.detach(),
        "geotr_m1_type_accuracy_correctable": type_accuracy_correctable,
        "geotr_m1_candidate_changed_fraction": candidate_changed.float().mean().detach(),
        "geotr_m1_candidate_resolved_fraction": candidate_resolved.float().mean().detach(),
        "geotr_m1_candidate_introduced_fraction": candidate_introduced.float().mean().detach(),
        "geotr_m1_candidate_dice_value_mean_eligible": _masked_mean(dice_value_map, transition_band).detach(),
        "geotr_m1_teacher_dice_value_mean_correctable": _masked_mean(teacher_dice_value_map, correctable).detach(),
        "geotr_m1_candidate_bce_utility_mean_eligible": _masked_mean(bce_utility_map, transition_band).detach(),
        "geotr_m1_predicted_edit_fraction": predicted_edit.detach(),
        "geotr_m1_edit_probability_mean": predicted_probability.detach(),
        "geotr_m1_target_edit_fraction": target_edit,
        "geotr_m1_gate_target_fraction": target_edit,
        "geotr_m1_overedit_ratio": predicted_edit / target_edit.clamp_min(EPS),
        "geotr_m1_edit_precision": edit_precision.detach(),
        "geotr_m1_edit_recall": edit_recall.detach(),
        "geotr_m1_correctable_recall": correctable_recall.detach(),
        "geotr_m1_false_edit_rate": false_edit_rate.detach(),
        "geotr_m1_harmful_edit_fraction": harmful_edit_fraction.detach(),
        "geotr_m1_source_nll": source_nll.detach(),
        "geotr_m1_correction_loss": correction_loss.detach(),
        "geotr_m1_teacher_segmentation_loss": teacher_segmentation.detach(),
        "geotr_m1_teacher_gate_fraction": teacher_gate.mean().detach(),
        "geotr_m1_teacher_dice": teacher_dice.mean().detach(),
        "geotr_m1_teacher_gain": (teacher_dice - base_soft_dice).mean().detach(),
        "geotr_m1_preserve_loss": harm_bce.detach(),
        "geotr_m1_harm_only_loss": harm_bce.detach(),
        "geotr_m1_nondegradation_loss": nondegradation.detach(),
        "geotr_m1_base_error_fraction": base_error.float().mean().detach(),
        "geotr_m1_correctable_fraction": correctable_fraction,
        "geotr_m1_beneficial_fraction": beneficial_fraction,
        "geotr_m1_transition_band_fraction": transition_band.float().mean().detach(),
        "geotr_m1_transition_correctable_coverage": transition_coverage.detach(),
        "geotr_m1_utility_mean_correctable": utility_on_correctable,
        "geotr_m1_utility_mean_beneficial": utility_on_beneficial,
        "geotr_m1_teacher_action_utility_mean": teacher_action_utility,
        "geotr_m1_selection_loss": selection_loss.detach(),
        "geotr_m1_transport_loss": transport_loss.detach(),
        "geotr_m1_safety_loss": safety_loss.detach(),
        "geotr_m1_gate_teacher_warmup_active": objective.detach().new_tensor(
            0.0 if action_value_policy else float(int(epoch) < gate_warmup_epochs)
        ),
        "geotr_m1_utility_margin": objective.detach().new_tensor(utility_margin),
        "geotr_m1_local_oracle_error_coverage": (
            correctable.float().sum().detach() / base_error.float().sum().clamp_min(1.0)
        ),
        "geotr_m1_valid_source_mass": _masked_mean(valid_source_mass.detach(), correctable),
        "geotr_m1_flow_smoothness": zero, "geotr_m1_flow_global_smoothness": zero,
        "geotr_m1_flow_folding_penalty": zero,
        "geotr_m1_flow_rms_px": _mean_aux(aux, "geotopo_flow_rms_px", objective),
        "geotr_m1_flow_max_px": _mean_aux(aux, "geotopo_flow_max_px", objective),
        "geotr_m1_flow_jacobian_mean": _mean_aux(aux, "geotopo_flow_jacobian_mean", objective),
        "geotr_m1_flow_folding_fraction": _mean_aux(aux, "geotopo_flow_folding_fraction", objective),
        "geotr_m1_geometry_abs_change": _mean_aux(aux, "geotopo_geometry_abs_change", objective),
        "geotr_m1_range_violation_fraction": _mean_aux(aux, "geotr_m1_range_violation_fraction", objective),
        "geotr_m1_range_violation_max": _mean_aux(aux, "geotr_m1_range_violation_max", objective),
        "geotr_m1_autozero_trust_mean": _mean_aux(aux, "geotr_m1_autozero_trust_mean", objective),
        "geotr_m1_autozero_trust_std": _mean_aux(aux, "geotr_m1_autozero_trust_std", objective),
        "geotr_m1_source_chunk_size": _mean_aux(aux, "geotr_m1_source_chunk_size", objective),
        "geotr_m1_source_chunk_count": _mean_aux(aux, "geotr_m1_source_chunk_count", objective),
        "geotr_m1_memory_exact": one,
        "geotr_m1_v31_rootfix": one,
        "geotr_m1_eligible_gate_rootfix": objective.detach().new_tensor(float(eligible_gate_rootfix)),
        "geotr_m1_autozero_objective": one, "geotr_m1_has_m2": zero,
    }
    return objective, diagnostics


def _compute_v3_legacy_loss(
    cfg: Any,
    candidate_logits: torch.Tensor,
    masks: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    epoch: int = 0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Frozen compatibility implementation for archived v3 configs.
    del cfg, candidate_logits, epoch
    keys = (
        "geotr_m1_final_logits", "geotr_m1_base_logits",
        "geotr_m1_edit_logit", "geotr_m1_locator_logit", "geotr_m1_type_logits",
        "geotr_m1_add_valid_source_mass", "geotr_m1_remove_valid_source_mass",
        "geotr_m1_add_source_exists", "geotr_m1_remove_source_exists",
        "geotr_m1_local_radius_px",
    )
    if any(not isinstance(aux.get(key), torch.Tensor) for key in keys):
        raise RuntimeError("SemLT-LST v3 loss is missing calibrated transport statistics")
    logits = aux["geotr_m1_final_logits"]
    base_logits = aux["geotr_m1_base_logits"]
    edit_logit = aux["geotr_m1_edit_logit"]
    locator_logit = aux["geotr_m1_locator_logit"]
    type_logits = aux["geotr_m1_type_logits"]
    add_valid_source_mass = aux["geotr_m1_add_valid_source_mass"]
    remove_valid_source_mass = aux["geotr_m1_remove_valid_source_mass"]
    add_source_exists = aux["geotr_m1_add_source_exists"].detach().bool()
    remove_source_exists = aux["geotr_m1_remove_source_exists"].detach().bool()
    if logits.ndim == 3: logits = logits[:, None]
    if base_logits.ndim == 3: base_logits = base_logits[:, None]
    target = _target3(masks, logits.shape[-2:]).to(logits)
    probability = torch.sigmoid(logits)
    base_probability = torch.sigmoid(base_logits.detach())
    base_hard = base_probability[:, 0] >= 0.5
    target_hard = target >= 0.5
    source_exists = torch.where(target_hard, add_source_exists, remove_source_exists)
    base_error = base_hard != target_hard
    correctable = base_error & source_exists
    preserve = ~correctable
    deploy_gate_map = F.binary_cross_entropy_with_logits(edit_logit.float(), correctable.float(), reduction="none").to(logits)
    deploy_gate_loss = deploy_gate_map.mean()
    gate_probability = torch.sigmoid(edit_logit)
    gate_brier = (gate_probability - correctable.float()).square().mean()
    locator_loss = _balanced_binary_bce(locator_logit, correctable)
    type_target = (~target_hard).long()
    type_map = F.cross_entropy(type_logits.float(), type_target, reduction="none").to(logits)
    type_loss = _masked_mean(type_map, correctable)
    valid_source_mass = torch.where(target_hard, add_valid_source_mass, remove_valid_source_mass).clamp_min(EPS)
    source_nll = _masked_mean(-torch.log(valid_source_mass), correctable)
    final_bce_map = F.binary_cross_entropy_with_logits(logits[:, 0].float(), target.float(), reduction="none").to(logits)
    final_bce = final_bce_map.mean()
    final_dice_loss = (1.0 - _dice_per_case(probability, target)).mean()
    segmentation = 0.5 * (final_bce + final_dice_loss)
    correction_loss = _masked_mean(final_bce_map, correctable)
    preserve_bce = _masked_mean(final_bce_map, preserve)
    base_soft_dice = _dice_per_case(base_probability, target).detach()
    final_soft_dice = _dice_per_case(probability, target)
    nondegradation = F.relu(base_soft_dice - final_soft_dice).mean()
    safety_loss = 0.5 * (preserve_bce + nondegradation)
    structure_loss = torch.stack([locator_loss, type_loss, source_nll]).mean()
    objective = torch.stack([segmentation, deploy_gate_loss, correction_loss, structure_loss, safety_loss]).mean()
    final_dice_detached = _dice_per_case(probability.detach(), target)
    predicted_edit = gate_probability.detach().mean()
    target_edit = correctable.float().mean().detach()
    hard_edit = gate_probability.detach() >= 0.5
    edit_tp = (hard_edit & correctable).float().sum()
    edit_precision = edit_tp / hard_edit.float().sum().clamp_min(1.0)
    edit_recall = edit_tp / correctable.float().sum().clamp_min(1.0)
    false_edit_rate = (hard_edit & preserve).float().sum() / preserve.float().sum().clamp_min(1.0)
    zero, one = objective.detach().new_zeros(()), objective.detach().new_ones(())
    diagnostics = {
        "mhcs_total_loss": objective.detach(), "mhcs_final_loss": segmentation.detach(),
        "mhcs_ce_loss": final_bce.detach(), "mhcs_dice_loss": final_dice_loss.detach(),
        "mhcs_m1_objective": objective.detach(), "mhcs_m2_objective": zero,
        "mhcs_final_gain": (final_dice_detached - base_soft_dice).mean(),
        "mhcs_base_dice": base_soft_dice.mean(), "mhcs_final_dice": final_dice_detached.mean(),
        "geotopo_total_loss": objective.detach(), "geotopo_final_loss": segmentation.detach(),
        "geotopo_geometry_loss": structure_loss.detach(), "geotopo_geometry_bce_loss": correction_loss.detach(),
        "geotopo_geometry_dice_loss": final_dice_loss.detach(), "geotopo_geometry_boundary_loss": zero,
        "geotopo_reconstruction_loss": safety_loss.detach(), "geotopo_flow_smoothness": zero,
        "geotopo_flow_folding_penalty": zero, "geotr_m1_total_loss": objective.detach(),
        "geotr_m1_segmentation_loss": segmentation.detach(), "geotr_m1_bce_loss": final_bce.detach(),
        "geotr_m1_dice_loss": final_dice_loss.detach(), "geotr_m1_boundary_loss": zero,
        "geotr_m1_state_loss": deploy_gate_loss.detach(), "geotr_m1_deploy_gate_loss": deploy_gate_loss.detach(),
        "geotr_m1_locator_loss": locator_loss.detach(), "geotr_m1_type_loss": type_loss.detach(),
        "geotr_m1_gate_brier": gate_brier.detach(), "geotr_m1_predicted_edit_fraction": predicted_edit,
        "geotr_m1_target_edit_fraction": target_edit, "geotr_m1_overedit_ratio": predicted_edit / target_edit.clamp_min(EPS),
        "geotr_m1_edit_precision": edit_precision.detach(), "geotr_m1_edit_recall": edit_recall.detach(),
        "geotr_m1_false_edit_rate": false_edit_rate.detach(), "geotr_m1_source_nll": source_nll.detach(),
        "geotr_m1_correction_loss": correction_loss.detach(), "geotr_m1_preserve_loss": preserve_bce.detach(),
        "geotr_m1_nondegradation_loss": nondegradation.detach(), "geotr_m1_base_error_fraction": base_error.float().mean().detach(),
        "geotr_m1_correctable_fraction": target_edit,
        "geotr_m1_local_oracle_error_coverage": correctable.float().sum().detach() / base_error.float().sum().clamp_min(1.0),
        "geotr_m1_valid_source_mass": _masked_mean(valid_source_mass.detach(), correctable),
        "geotr_m1_flow_smoothness": zero, "geotr_m1_flow_global_smoothness": zero,
        "geotr_m1_flow_folding_penalty": zero,
        "geotr_m1_flow_rms_px": _mean_aux(aux, "geotopo_flow_rms_px", objective),
        "geotr_m1_flow_max_px": _mean_aux(aux, "geotopo_flow_max_px", objective),
        "geotr_m1_flow_jacobian_mean": _mean_aux(aux, "geotopo_flow_jacobian_mean", objective),
        "geotr_m1_flow_folding_fraction": _mean_aux(aux, "geotopo_flow_folding_fraction", objective),
        "geotr_m1_geometry_abs_change": _mean_aux(aux, "geotopo_geometry_abs_change", objective),
        "geotr_m1_range_violation_fraction": _mean_aux(aux, "geotr_m1_range_violation_fraction", objective),
        "geotr_m1_range_violation_max": _mean_aux(aux, "geotr_m1_range_violation_max", objective),
        "geotr_m1_autozero_trust_mean": _mean_aux(aux, "geotr_m1_autozero_trust_mean", objective),
        "geotr_m1_autozero_trust_std": _mean_aux(aux, "geotr_m1_autozero_trust_std", objective),
        "geotr_m1_source_chunk_size": _mean_aux(aux, "geotr_m1_source_chunk_size", objective),
        "geotr_m1_source_chunk_count": _mean_aux(aux, "geotr_m1_source_chunk_count", objective),
        "geotr_m1_memory_exact": one, "geotr_m1_autozero_objective": one, "geotr_m1_has_m2": zero,
    }
    return objective, diagnostics




def _binary_boundary3(mask: torch.Tensor) -> torch.Tensor:
    """Two-sided binary morphological boundary, returned as [B,H,W] bool."""
    if mask.ndim == 3:
        x = mask[:, None].float()
    elif mask.ndim == 4 and mask.shape[1] == 1:
        x = mask.float()
    else:
        raise ValueError(f"Expected [B,H,W] or [B,1,H,W], got {tuple(mask.shape)}")
    dilated = F.max_pool2d(x, 3, stride=1, padding=1)
    eroded = -F.max_pool2d(-x, 3, stride=1, padding=1)
    return ((dilated - eroded) > 0.5)[:, 0]


def _nearest_gt_boundary_sample_offset(
    base_boundary: torch.Tensor,
    gt_boundary: torch.Tensor,
    foreground_normal: torch.Tensor,
    radius_px: float,
    chunk_size: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact local boundary correspondence target on sparse predicted-boundary points.

    For every Base boundary point, find the nearest GT boundary point in Euclidean
    image coordinates.  The desired *contour motion* is projected onto the Base
    foreground normal.  ``grid_sample`` uses inverse warping, so its supervised
    sampling offset is the negative of that projected contour motion.

    Returns
    -------
    target_sample_offset_px : [B,H,W]
    reachable               : [B,H,W] bool, nearest GT boundary <= radius
    nearest_distance_px     : [B,H,W]
    normal_alignment        : [B,H,W], |normal projection| / Euclidean distance
    """
    if base_boundary.ndim != 3 or gt_boundary.ndim != 3:
        raise ValueError("base_boundary and gt_boundary must be [B,H,W]")
    if foreground_normal.ndim != 4 or foreground_normal.shape[1] != 2:
        raise ValueError("foreground_normal must be [B,2,H,W]")
    b, h, w = base_boundary.shape
    target = foreground_normal.new_zeros((b, h, w))
    reachable = torch.zeros((b, h, w), device=base_boundary.device, dtype=torch.bool)
    nearest_distance = foreground_normal.new_zeros((b, h, w))
    alignment = foreground_normal.new_zeros((b, h, w))
    radius = float(radius_px)
    chunk_size = max(int(chunk_size), 64)

    # Coordinates and nearest-neighbour assignment are target construction only.
    with torch.no_grad():
        for bi in range(b):
            base_coords = torch.nonzero(base_boundary[bi], as_tuple=False)  # [N,2] y,x
            gt_coords = torch.nonzero(gt_boundary[bi], as_tuple=False)      # [M,2] y,x
            if base_coords.numel() == 0 or gt_coords.numel() == 0:
                continue
            gt_float = gt_coords.float()
            for start in range(0, base_coords.shape[0], chunk_size):
                coords = base_coords[start:start + chunk_size]
                coords_f = coords.float()
                distances = torch.cdist(coords_f, gt_float, p=2.0)
                min_dist, min_index = distances.min(dim=1)
                nearest = gt_coords[min_index]
                yy, xx = coords[:, 0], coords[:, 1]
                vec_y = (nearest[:, 0] - yy).to(foreground_normal)
                vec_x = (nearest[:, 1] - xx).to(foreground_normal)
                nx = foreground_normal[bi, 0, yy, xx]
                ny = foreground_normal[bi, 1, yy, xx]
                contour_motion = vec_x * nx + vec_y * ny
                sample_offset = -contour_motion
                is_reachable = min_dist <= radius + 1.0e-6
                sample_offset = sample_offset.clamp(-radius, radius)
                target[bi, yy, xx] = sample_offset
                nearest_distance[bi, yy, xx] = min_dist.to(foreground_normal)
                alignment[bi, yy, xx] = (
                    contour_motion.abs() / min_dist.to(foreground_normal).clamp_min(1.0e-6)
                ).clamp(0.0, 1.0)
                reachable[bi, yy, xx] = is_reachable
    return target, reachable, nearest_distance, alignment


def _flow_geometry_regularization(
    scalar_offset_px: torch.Tensor,
    flow_px: torch.Tensor,
    support: torch.Tensor,
    radius_px: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Parameter-free contour-field regularity: support TV + true fold penalty."""
    if scalar_offset_px.ndim == 4:
        scalar = scalar_offset_px[:, 0]
    else:
        scalar = scalar_offset_px
    normalized = scalar / max(float(radius_px), 1.0)
    support = support.bool()
    dx = (normalized[:, :, 1:] - normalized[:, :, :-1]).abs()
    dy = (normalized[:, 1:, :] - normalized[:, :-1, :]).abs()
    mx = support[:, :, 1:] & support[:, :, :-1]
    my = support[:, 1:, :] & support[:, :-1, :]
    terms: List[torch.Tensor] = []
    if bool(mx.any()):
        terms.append(dx[mx].mean())
    if bool(my.any()):
        terms.append(dy[my].mean())
    tv = torch.stack(terms).mean() if terms else scalar.sum() * 0.0

    if flow_px.shape[-2] < 2 or flow_px.shape[-1] < 2:
        fold = flow_px.sum() * 0.0
    else:
        ux, uy = flow_px[:, 0], flow_px[:, 1]
        dux_dx = ux[:, :-1, 1:] - ux[:, :-1, :-1]
        dux_dy = ux[:, 1:, :-1] - ux[:, :-1, :-1]
        duy_dx = uy[:, :-1, 1:] - uy[:, :-1, :-1]
        duy_dy = uy[:, 1:, :-1] - uy[:, :-1, :-1]
        det = (1.0 + dux_dx) * (1.0 + duy_dy) - dux_dy * duy_dx
        # Penalize only actual orientation reversals; no arbitrary margin knob.
        fold = F.relu(-det).mean()
    geometry = torch.stack([tv, fold]).mean()
    return geometry, tv, fold


def _compute_boundary_normal_warp_loss(
    cfg: Any,
    masks: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    epoch: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """SemLT Boundary-Normal Warp objective.

    The method has no Gate/Type/Action-Value policy.  A single bounded signed
    inverse-warp offset moves detached Base logits along the Base foreground
    normal.  GT enters only here to supervise current Base boundary points with
    nearest-GT-boundary correspondences; inference never sees GT.
    """
    del epoch
    required = (
        "geotr_m1_final_logits",
        "geotr_m1_base_logits",
        "geotr_m1_predicted_sample_offset_px",
        "geotr_m1_flow_px",
        "geotr_m1_boundary_normal",
        "geotr_m1_base_boundary_mask",
        "geotr_m1_warp_support",
        "geotr_m1_local_radius_px",
    )
    missing = [k for k in required if not isinstance(aux.get(k), torch.Tensor)]
    if missing:
        raise RuntimeError(f"SemLT Boundary-Normal Warp loss missing statistics: {missing}")

    final_logits = aux["geotr_m1_final_logits"]
    base_logits = aux["geotr_m1_base_logits"].detach()
    predicted_offset = aux["geotr_m1_predicted_sample_offset_px"]
    flow_px = aux["geotr_m1_flow_px"]
    normal = aux["geotr_m1_boundary_normal"].detach()
    base_boundary = aux["geotr_m1_base_boundary_mask"].detach().bool()
    support = aux["geotr_m1_warp_support"].detach().bool()
    radius = float(aux["geotr_m1_local_radius_px"].detach().float().mean().item())
    if radius <= 0.0:
        raise ValueError("Boundary-Normal Warp requires positive SEMLT_LOCAL_RADIUS_PX")

    target = _target3(masks, final_logits.shape[-2:])
    target_hard = target >= 0.5
    gt_boundary = _binary_boundary3(target_hard)
    target_offset, reachable, nearest_distance, normal_alignment = (
        _nearest_gt_boundary_sample_offset(
            base_boundary=base_boundary,
            gt_boundary=gt_boundary,
            foreground_normal=normal,
            radius_px=radius,
        )
    )

    # 1) Geometry correspondence: normalized so r is architecture scale, not a loss weight.
    disp_map = F.smooth_l1_loss(
        predicted_offset / radius,
        target_offset.to(predicted_offset) / radius,
        reduction="none",
    )
    displacement_loss = _masked_mean(disp_map, reachable)

    # 2) The actual warped mask must solve segmentation, not merely match a vector target.
    final_probability = torch.sigmoid(final_logits).clamp(EPS, 1.0 - EPS)
    base_probability = torch.sigmoid(base_logits).clamp(EPS, 1.0 - EPS)
    bce_loss = F.binary_cross_entropy_with_logits(
        final_logits[:, 0].float(), target.float(), reduction="mean"
    ).to(final_logits)
    final_dice_case = _dice_per_case(final_probability, target)
    base_dice_case = _dice_per_case(base_probability, target)
    dice_loss = 1.0 - final_dice_case.mean()
    segmentation_loss = torch.stack([bce_loss, dice_loss]).mean()

    # 3) A contour displacement field should be locally coherent and non-folding.
    geometry_loss, tv_loss, fold_loss = _flow_geometry_regularization(
        scalar_offset_px=predicted_offset,
        flow_px=flow_px,
        support=support,
        radius_px=radius,
    )

    objective = torch.stack([
        displacement_loss,
        segmentation_loss,
        geometry_loss,
    ]).mean()

    with torch.no_grad():
        reachable_count = reachable.float().sum().clamp_min(1.0)
        boundary_count = base_boundary.float().sum().clamp_min(1.0)
        support_count = support.float().sum().clamp_min(1.0)
        offset_error = (predicted_offset.detach() - target_offset.to(predicted_offset)).abs()
        target_abs = target_offset.abs().to(predicted_offset)
        predicted_abs = predicted_offset.detach().abs()
        signed_cos = torch.zeros_like(target_offset).to(predicted_offset)
        nonzero_target = reachable & (target_abs > 1.0e-6)
        if bool(nonzero_target.any()):
            signed_cos[nonzero_target] = torch.sign(
                predicted_offset.detach()[nonzero_target]
                * target_offset.to(predicted_offset)[nonzero_target]
            )
        direction_accuracy = (
            (signed_cos[nonzero_target] > 0).float().mean()
            if bool(nonzero_target.any()) else predicted_offset.new_zeros(())
        )
        base_error = ((base_probability[:, 0] >= 0.5) != target_hard)
        final_error = ((final_probability[:, 0] >= 0.5) != target_hard)
        corrected = base_error & ~final_error
        introduced = ~base_error & final_error
        boundary_corrected = corrected & support
        boundary_introduced = introduced & support
        flow_mag = torch.linalg.vector_norm(flow_px.detach(), dim=1)
        zero = objective.detach().new_zeros(())

    final_gain = (final_dice_case - base_dice_case).mean()
    diagnostics: Dict[str, torch.Tensor] = {
        "mhcs_total_loss": objective.detach(),
        "mhcs_final_loss": segmentation_loss.detach(),
        "mhcs_ce_loss": bce_loss.detach(),
        "mhcs_dice_loss": dice_loss.detach(),
        "mhcs_m1_objective": objective.detach(),
        "mhcs_m2_objective": zero,
        "mhcs_final_gain": final_gain.detach(),
        "mhcs_base_dice": base_dice_case.mean().detach(),
        "mhcs_final_dice": final_dice_case.mean().detach(),
        "geotopo_total_loss": objective.detach(),
        "geotopo_final_loss": segmentation_loss.detach(),
        "geotopo_geometry_loss": geometry_loss.detach(),
        "geotopo_geometry_bce_loss": displacement_loss.detach(),
        "geotopo_geometry_dice_loss": dice_loss.detach(),
        "geotopo_geometry_boundary_loss": displacement_loss.detach(),
        "geotopo_reconstruction_loss": geometry_loss.detach(),
        "geotopo_flow_smoothness": tv_loss.detach(),
        "geotopo_flow_folding_penalty": fold_loss.detach(),
        "geotr_m1_total_loss": objective.detach(),
        "geotr_m1_segmentation_loss": segmentation_loss.detach(),
        "geotr_m1_deploy_segmentation_loss": segmentation_loss.detach(),
        "geotr_m1_bce_loss": bce_loss.detach(),
        "geotr_m1_dice_loss": dice_loss.detach(),
        "geotr_m1_boundary_loss": displacement_loss.detach(),
        "geotr_m1_state_loss": displacement_loss.detach(),
        "geotr_m1_deploy_gate_loss": zero,
        "geotr_m1_locator_loss": zero,
        "geotr_m1_type_loss": zero,
        "geotr_m1_gate_brier": zero,
        "geotr_m1_source_nll": zero,
        "geotr_m1_correction_loss": displacement_loss.detach(),
        "geotr_m1_teacher_segmentation_loss": zero,
        "geotr_m1_teacher_gate_fraction": zero,
        "geotr_m1_teacher_dice": base_dice_case.mean().detach(),
        "geotr_m1_teacher_gain": zero,
        "geotr_m1_preserve_loss": zero,
        "geotr_m1_harm_only_loss": zero,
        "geotr_m1_nondegradation_loss": zero,
        "geotr_m1_base_error_fraction": base_error.float().mean().detach(),
        "geotr_m1_correctable_fraction": (base_error & support).float().mean().detach(),
        "geotr_m1_beneficial_fraction": boundary_corrected.float().mean().detach(),
        "geotr_m1_transition_band_fraction": support.float().mean().detach(),
        "geotr_m1_transition_correctable_coverage": (
            (base_error & support).float().sum() / base_error.float().sum().clamp_min(1.0)
        ).detach(),
        "geotr_m1_utility_mean_correctable": zero,
        "geotr_m1_utility_mean_beneficial": zero,
        "geotr_m1_teacher_action_utility_mean": zero,
        "geotr_m1_selection_loss": zero,
        "geotr_m1_transport_loss": torch.stack([displacement_loss, segmentation_loss]).mean().detach(),
        "geotr_m1_safety_loss": geometry_loss.detach(),
        "geotr_m1_gate_teacher_warmup_active": zero,
        "geotr_m1_utility_margin": zero,
        "geotr_m1_local_oracle_error_coverage": (
            reachable.float().sum() / boundary_count
        ).detach(),
        "geotr_m1_valid_source_mass": zero,
        "geotr_m1_flow_smoothness": tv_loss.detach(),
        "geotr_m1_flow_global_smoothness": tv_loss.detach(),
        "geotr_m1_flow_folding_penalty": fold_loss.detach(),
        "geotr_m1_flow_rms_px": flow_mag.square().mean().sqrt().detach(),
        "geotr_m1_flow_max_px": flow_mag.max().detach(),
        "geotr_m1_geometry_abs_change": (final_probability - base_probability).abs().mean().detach(),
        "geotr_m1_range_violation_fraction": _mean_aux(aux, "geotr_m1_range_violation_fraction", objective),
        "geotr_m1_range_violation_max": _mean_aux(aux, "geotr_m1_range_violation_max", objective),
        "geotr_m1_autozero_trust_mean": _mean_aux(aux, "geotr_m1_autozero_trust_mean", objective),
        "geotr_m1_autozero_trust_std": _mean_aux(aux, "geotr_m1_autozero_trust_std", objective),
        "geotr_m1_memory_exact": objective.detach().new_ones(()),
        "geotr_m1_v31_rootfix": zero,
        "geotr_m1_eligible_gate_rootfix": zero,
        "geotr_m1_autozero_objective": objective.detach().new_ones(()),
        "geotr_m1_has_m2": zero,
        # Boundary-Normal Warp diagnostics.
        "geotr_m1_boundary_normal_warp": objective.detach().new_ones(()),
        "geotr_m1_displacement_loss": displacement_loss.detach(),
        "geotr_m1_geometry_regularization": geometry_loss.detach(),
        "geotr_m1_boundary_tv_loss": tv_loss.detach(),
        "geotr_m1_fold_loss": fold_loss.detach(),
        "geotr_m1_base_boundary_fraction": base_boundary.float().mean().detach(),
        "geotr_m1_warp_support_fraction": support.float().mean().detach(),
        "geotr_m1_boundary_reachable_fraction": (reachable.float().sum() / boundary_count).detach(),
        "geotr_m1_reachable_support_fraction": (reachable.float().sum() / support_count).detach(),
        "geotr_m1_target_sample_offset_abs_px": _masked_mean(target_abs, reachable).detach(),
        "geotr_m1_predicted_sample_offset_abs_px": _masked_mean(predicted_abs, reachable).detach(),
        "geotr_m1_sample_offset_mae_px": _masked_mean(offset_error, reachable).detach(),
        "geotr_m1_displacement_direction_accuracy": direction_accuracy.detach(),
        "geotr_m1_gt_correspondence_distance_px": _masked_mean(nearest_distance.to(predicted_offset), reachable).detach(),
        "geotr_m1_gt_normal_alignment": _masked_mean(normal_alignment.to(predicted_offset), reachable).detach(),
        "geotr_m1_corrected_fraction": corrected.float().mean().detach(),
        "geotr_m1_introduced_fraction": introduced.float().mean().detach(),
        "geotr_m1_boundary_corrected_fraction": boundary_corrected.float().mean().detach(),
        "geotr_m1_boundary_introduced_fraction": boundary_introduced.float().mean().detach(),
        "geotr_m1_net_error_change_fraction": (introduced.float().mean() - corrected.float().mean()).detach(),
    }
    return objective, diagnostics

def _operator_matched_geometry_regularization(
    extended_offset_px: torch.Tensor,
    extended_normal: torch.Tensor,
    flow_px: torch.Tensor,
    support: torch.Tensor,
    radius_px: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Contour-tangent smoothness + physical fold penalty.

    The offset field is already nearest-owner extended.  Tangent smoothness is
    therefore evaluated on the physical contour-owned field instead of forcing
    arbitrary normal-direction equality against unsupervised dense-band pixels.
    """
    scalar = extended_offset_px[:, None] if extended_offset_px.ndim == 3 else extended_offset_px
    normal = extended_normal
    normalized = scalar / max(float(radius_px), 1.0)
    kx = normalized.new_tensor([[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]])[:, None] / 8.0
    ky = normalized.new_tensor([[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]])[:, None] / 8.0
    gx = F.conv2d(normalized, kx, padding=1)[:, 0]
    gy = F.conv2d(normalized, ky, padding=1)[:, 0]
    tx = -normal[:, 1]
    ty = normal[:, 0]
    tangent_derivative = (gx * tx + gy * ty).abs()
    tangent_smooth = _masked_mean(tangent_derivative, support.bool())

    if flow_px.shape[-2] < 2 or flow_px.shape[-1] < 2:
        fold = flow_px.sum() * 0.0
    else:
        ux, uy = flow_px[:, 0], flow_px[:, 1]
        dux_dx = ux[:, :-1, 1:] - ux[:, :-1, :-1]
        dux_dy = ux[:, 1:, :-1] - ux[:, :-1, :-1]
        duy_dx = uy[:, :-1, 1:] - uy[:, :-1, :-1]
        duy_dy = uy[:, 1:, :-1] - uy[:, :-1, :-1]
        det = (1.0 + dux_dx) * (1.0 + duy_dy) - dux_dy * duy_dx
        fold = F.relu(-det).mean()
    geometry = torch.stack([tangent_smooth, fold]).mean()
    return geometry, tangent_smooth, fold


def _compute_sdf_operator_matched_warp_loss(
    cfg: Any,
    masks: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    epoch: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """SDF-guided operator-matched contour transport objective.

    Geometry supervision is defined on a one-sided Base contour and uses the
    same normal ray as the physical inverse grid warp.  Hard examples outside
    the one-step radius are clipped, never discarded.  The actual segmentation
    BCE is normalized on the swept support, while Dice remains global.
    """
    del epoch
    required = (
        "geotr_m1_final_logits",
        "geotr_m1_base_logits",
        "geotr_m1_predicted_owner_sample_offset_px",
        "geotr_m1_extended_sample_offset_px",
        "geotr_m1_flow_px",
        "geotr_m1_contour_owner_normal",
        "geotr_m1_boundary_normal",
        "geotr_m1_base_sdf",
        "geotr_m1_contour_owner_mask",
        "geotr_m1_owner_distance_px",
        "geotr_m1_warp_support",
        "geotr_m1_local_radius_px",
    )
    missing = [k for k in required if not isinstance(aux.get(k), torch.Tensor)]
    if missing:
        raise RuntimeError(f"SemLT SDF operator-matched warp loss missing statistics: {missing}")

    final_logits = aux["geotr_m1_final_logits"]
    base_logits = aux["geotr_m1_base_logits"].detach()
    predicted_owner_offset = aux["geotr_m1_predicted_owner_sample_offset_px"]
    extended_offset = aux["geotr_m1_extended_sample_offset_px"]
    flow_px = aux["geotr_m1_flow_px"]
    owner_normal = aux["geotr_m1_contour_owner_normal"].detach()
    extended_normal = aux["geotr_m1_boundary_normal"].detach()
    base_sdf = aux["geotr_m1_base_sdf"].detach()
    contour_owner = aux["geotr_m1_contour_owner_mask"].detach().bool()
    support = aux["geotr_m1_warp_support"].detach().bool()
    radius = int(round(float(aux["geotr_m1_local_radius_px"].detach().float().mean().item())))
    if radius <= 0:
        raise ValueError("SDF operator-matched warp requires positive SEMLT_LOCAL_RADIUS_PX")

    target = _target3(masks, final_logits.shape[-2:])
    target_hard = target >= 0.5
    target_offset, valid_owner, reachable, clipped, motion_abs, normal_alignment = (
        operator_matched_normal_ray_target(
            base_owner=contour_owner,
            base_sdf=base_sdf,
            base_normal=owner_normal,
            gt_mask=target_hard,
            radius_px=radius,
        )
    )

    # 1) Operator-matched contour displacement.  All valid contour owners learn;
    # unreachable one-step cases receive the correctly clipped endpoint target.
    disp_map = F.smooth_l1_loss(
        predicted_owner_offset / float(radius),
        target_offset.to(predicted_owner_offset) / float(radius),
        reduction="none",
    )
    displacement_loss = _masked_mean(disp_map, valid_owner)

    # 2) Actual task loss.  BCE is normalized on the only region M1 can change;
    # global Dice keeps the final object-level objective coupled to deployment.
    final_probability = torch.sigmoid(final_logits).clamp(EPS, 1.0 - EPS)
    base_probability = torch.sigmoid(base_logits).clamp(EPS, 1.0 - EPS)
    bce_map = F.binary_cross_entropy_with_logits(
        final_logits[:, 0].float(), target.float(), reduction="none"
    ).to(final_logits)
    local_bce = _masked_mean(bce_map, support)
    final_dice_case = _dice_per_case(final_probability, target)
    base_dice_case = _dice_per_case(base_probability, target)
    dice_loss = 1.0 - final_dice_case.mean()
    segmentation_loss = torch.stack([local_bce, dice_loss]).mean()

    # 3) Geometry regularity follows the contour-owned field.
    geometry_loss, tangent_loss, fold_loss = _operator_matched_geometry_regularization(
        extended_offset_px=extended_offset,
        extended_normal=extended_normal,
        flow_px=flow_px,
        support=support,
        radius_px=float(radius),
    )

    objective = torch.stack([displacement_loss, segmentation_loss, geometry_loss]).mean()

    with torch.no_grad():
        owner_count = valid_owner.float().sum().clamp_min(1.0)
        support_count = support.float().sum().clamp_min(1.0)
        reachable_count = reachable.float().sum()
        target_abs = target_offset.abs().to(predicted_owner_offset)
        predicted_abs = predicted_owner_offset.detach().abs()
        offset_error = (predicted_owner_offset.detach() - target_offset.to(predicted_owner_offset)).abs()
        nonzero_target = valid_owner & (target_abs > 1.0e-6)
        direction_accuracy = (
            (predicted_owner_offset.detach()[nonzero_target] * target_offset.to(predicted_owner_offset)[nonzero_target] > 0).float().mean()
            if bool(nonzero_target.any()) else predicted_owner_offset.new_zeros(())
        )
        base_error = ((base_probability[:, 0] >= 0.5) != target_hard)
        final_error = ((final_probability[:, 0] >= 0.5) != target_hard)
        corrected = base_error & ~final_error
        introduced = ~base_error & final_error
        zero = objective.detach().new_zeros(())
        clipped_fraction = clipped.float().sum() / owner_count
        reachable_fraction = reachable_count / owner_count
        amplitude_ratio = _masked_mean(predicted_abs, valid_owner) / _masked_mean(target_abs, valid_owner).clamp_min(EPS)
        align_mask = reachable & (normal_alignment > 0)

    diagnostics: Dict[str, torch.Tensor] = {
        "mhcs_total_loss": objective.detach(),
        "mhcs_final_loss": segmentation_loss.detach(),
        "mhcs_ce_loss": local_bce.detach(),
        "mhcs_dice_loss": dice_loss.detach(),
        "mhcs_m1_objective": objective.detach(),
        "mhcs_m2_objective": zero,
        "mhcs_final_gain": (final_dice_case.mean() - base_dice_case.mean()).detach(),
        "mhcs_base_dice": base_dice_case.mean().detach(),
        "mhcs_final_dice": final_dice_case.mean().detach(),
        "geotopo_total_loss": objective.detach(),
        "geotopo_final_loss": segmentation_loss.detach(),
        "geotopo_geometry_loss": geometry_loss.detach(),
        "geotopo_geometry_bce_loss": displacement_loss.detach(),
        "geotopo_geometry_dice_loss": dice_loss.detach(),
        "geotopo_geometry_boundary_loss": displacement_loss.detach(),
        "geotopo_reconstruction_loss": geometry_loss.detach(),
        "geotopo_flow_smoothness": tangent_loss.detach(),
        "geotopo_flow_folding_penalty": fold_loss.detach(),
        "geotr_m1_total_loss": objective.detach(),
        "geotr_m1_segmentation_loss": segmentation_loss.detach(),
        "geotr_m1_deploy_segmentation_loss": segmentation_loss.detach(),
        "geotr_m1_bce_loss": local_bce.detach(),
        "geotr_m1_dice_loss": dice_loss.detach(),
        "geotr_m1_boundary_loss": displacement_loss.detach(),
        "geotr_m1_state_loss": displacement_loss.detach(),
        "geotr_m1_deploy_gate_loss": zero,
        "geotr_m1_locator_loss": zero,
        "geotr_m1_type_loss": zero,
        "geotr_m1_gate_brier": zero,
        "geotr_m1_source_nll": zero,
        "geotr_m1_correction_loss": displacement_loss.detach(),
        "geotr_m1_teacher_segmentation_loss": zero,
        "geotr_m1_teacher_gate_fraction": zero,
        "geotr_m1_teacher_dice": base_dice_case.mean().detach(),
        "geotr_m1_teacher_gain": zero,
        "geotr_m1_preserve_loss": zero,
        "geotr_m1_harm_only_loss": zero,
        "geotr_m1_nondegradation_loss": zero,
        "geotr_m1_base_error_fraction": base_error.float().mean().detach(),
        "geotr_m1_correctable_fraction": valid_owner.float().mean().detach(),
        "geotr_m1_beneficial_fraction": corrected.float().mean().detach(),
        "geotr_m1_transition_band_fraction": support.float().mean().detach(),
        "geotr_m1_transition_correctable_coverage": reachable_fraction.detach(),
        "geotr_m1_utility_mean_correctable": zero,
        "geotr_m1_utility_mean_beneficial": zero,
        "geotr_m1_teacher_action_utility_mean": zero,
        "geotr_m1_selection_loss": zero,
        "geotr_m1_transport_loss": torch.stack([displacement_loss, segmentation_loss]).mean().detach(),
        "geotr_m1_safety_loss": geometry_loss.detach(),
        "geotr_m1_gate_teacher_warmup_active": zero,
        "geotr_m1_utility_margin": zero,
        "geotr_m1_local_oracle_error_coverage": reachable_fraction.detach(),
        "geotr_m1_valid_source_mass": zero,
        "geotr_m1_flow_smoothness": tangent_loss.detach(),
        "geotr_m1_flow_global_smoothness": tangent_loss.detach(),
        "geotr_m1_flow_folding_penalty": fold_loss.detach(),
        "geotr_m1_flow_rms_px": _mean_aux(aux, "geotopo_flow_rms_px", objective),
        "geotr_m1_flow_max_px": _mean_aux(aux, "geotopo_flow_max_px", objective),
        "geotr_m1_geometry_abs_change": _mean_aux(aux, "geotopo_geometry_abs_change", objective),
        "geotr_m1_range_violation_fraction": _mean_aux(aux, "geotr_m1_range_violation_fraction", objective),
        "geotr_m1_range_violation_max": _mean_aux(aux, "geotr_m1_range_violation_max", objective),
        "geotr_m1_autozero_trust_mean": _mean_aux(aux, "geotr_m1_autozero_trust_mean", objective),
        "geotr_m1_autozero_trust_std": _mean_aux(aux, "geotr_m1_autozero_trust_std", objective),
        "geotr_m1_memory_exact": objective.detach().new_ones(()),
        "geotr_m1_v31_rootfix": zero,
        "geotr_m1_eligible_gate_rootfix": zero,
        "geotr_m1_autozero_objective": objective.detach().new_ones(()),
        "geotr_m1_has_m2": zero,
        "geotr_m1_boundary_normal_warp": zero,
        "geotr_m1_sdf_operator_matched_warp": objective.detach().new_ones(()),
        "geotr_m1_displacement_loss": displacement_loss.detach(),
        "geotr_m1_geometry_regularization": geometry_loss.detach(),
        "geotr_m1_boundary_tv_loss": tangent_loss.detach(),
        "geotr_m1_fold_loss": fold_loss.detach(),
        "geotr_m1_base_boundary_fraction": contour_owner.float().mean().detach(),
        "geotr_m1_contour_owner_fraction": contour_owner.float().mean().detach(),
        "geotr_m1_warp_support_fraction": support.float().mean().detach(),
        "geotr_m1_boundary_reachable_fraction": reachable_fraction.detach(),
        "geotr_m1_reachable_support_fraction": (reachable_count / support_count).detach(),
        "geotr_m1_target_clipped_fraction": clipped_fraction.detach(),
        "geotr_m1_target_sample_offset_abs_px": _masked_mean(target_abs, valid_owner).detach(),
        "geotr_m1_predicted_sample_offset_abs_px": _masked_mean(predicted_abs, valid_owner).detach(),
        "geotr_m1_sample_offset_mae_px": _masked_mean(offset_error, valid_owner).detach(),
        "geotr_m1_displacement_amplitude_ratio": amplitude_ratio.detach(),
        "geotr_m1_displacement_direction_accuracy": direction_accuracy.detach(),
        "geotr_m1_gt_correspondence_distance_px": _masked_mean(motion_abs.to(predicted_owner_offset), valid_owner).detach(),
        "geotr_m1_gt_normal_alignment": _masked_mean(normal_alignment.to(predicted_owner_offset), align_mask).detach(),
        "geotr_m1_local_bce_loss": local_bce.detach(),
        "geotr_m1_corrected_fraction": corrected.float().mean().detach(),
        "geotr_m1_introduced_fraction": introduced.float().mean().detach(),
        "geotr_m1_boundary_corrected_fraction": (corrected & support).float().mean().detach(),
        "geotr_m1_boundary_introduced_fraction": (introduced & support).float().mean().detach(),
        "geotr_m1_net_error_change_fraction": (introduced.float().mean() - corrected.float().mean()).detach(),
    }
    return objective, diagnostics



def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    w = weight.to(value)
    return (value * w).sum() / w.sum().clamp_min(1.0)


def _ordered_offset_crps_map(
    offset_probs: torch.Tensor,
    target_offset: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    """Discrete CRPS for the physically ordered offset bins.

    Unlike a regression loss on E[d], this score identifies the full posterior:
    a broad or wrong-sign distribution cannot obtain the same optimum merely by
    matching its first moment.  The final CDF entry is omitted because both CDFs
    are exactly one there.
    """
    if offset_probs.ndim != 4 or offset_probs.shape[1] < 2:
        raise ValueError("ordered offset CRPS expects [B,K,H,W] with K>=2")
    k = int(offset_probs.shape[1])
    if k != 2 * int(radius) + 1:
        raise ValueError("ordered offset CRPS bin/radius mismatch")
    if target_offset.shape != offset_probs.shape[:1] + offset_probs.shape[2:]:
        raise ValueError("ordered offset CRPS target shape mismatch")

    target_clamped = target_offset.to(offset_probs).clamp(
        -float(radius), float(radius)
    )
    shifted = target_clamped + float(radius)
    lower = torch.floor(shifted).long().clamp(0, k - 1)
    upper = (lower + 1).clamp(0, k - 1)
    lower_w = 1.0 - (shifted - lower.to(shifted)).clamp(0.0, 1.0)

    # Stream across the 17 ordered thresholds instead of materialising two
    # dense one-hot target volumes plus two dense CDF volumes.  This matters at
    # the locked physical batch=24 and is exactly the same discrete CRPS.
    pred_cdf = torch.zeros_like(target_clamped, dtype=torch.float32)
    score = torch.zeros_like(pred_cdf)
    for threshold in range(k - 1):
        pred_cdf = pred_cdf + offset_probs[:, threshold].float()
        target_cdf = torch.where(
            threshold < lower,
            torch.zeros_like(pred_cdf),
            torch.where(
                threshold < upper,
                lower_w.float(),
                torch.ones_like(pred_cdf),
            ),
        )
        score = score + (pred_cdf - target_cdf).square()
    return (score / float(k - 1)).to(offset_probs)


def _ordered_magnitude_crps_map(
    magnitude_probs: torch.Tensor,
    target_magnitude: torch.Tensor,
) -> torch.Tensor:
    """Discrete CRPS for conditional magnitude bins ``1..R``.

    Sign is scored separately by the hierarchical objective. Conditioning
    magnitude on the observed sign prevents positive and negative probability
    mass from cancelling before the distance distribution is learned.
    """
    if magnitude_probs.ndim != 4 or magnitude_probs.shape[1] < 1:
        raise ValueError("magnitude CRPS expects [B,R,H,W]")
    radius = int(magnitude_probs.shape[1])
    if target_magnitude.shape != (
        magnitude_probs.shape[0],
        magnitude_probs.shape[2],
        magnitude_probs.shape[3],
    ):
        raise ValueError("magnitude CRPS target shape mismatch")
    if radius == 1:
        return target_magnitude.to(magnitude_probs) * 0.0

    shifted = target_magnitude.to(magnitude_probs).clamp(1.0, float(radius)) - 1.0
    lower = torch.floor(shifted).long().clamp(0, radius - 1)
    upper = (lower + 1).clamp(0, radius - 1)
    lower_w = 1.0 - (shifted - lower.to(shifted)).clamp(0.0, 1.0)
    pred_cdf = torch.zeros_like(shifted, dtype=torch.float32)
    score = torch.zeros_like(pred_cdf)
    for threshold in range(radius - 1):
        pred_cdf = pred_cdf + magnitude_probs[:, threshold].float()
        target_cdf = torch.where(
            threshold < lower,
            torch.zeros_like(pred_cdf),
            torch.where(
                threshold < upper,
                lower_w.float(),
                torch.ones_like(pred_cdf),
            ),
        )
        score = score + (pred_cdf - target_cdf).square()
    return (score / float(radius - 1)).to(magnitude_probs)


def _posterior_stable_geometry_regularization(
    predicted_extended_offset_px: torch.Tensor,
    target_extended_offset_px: torch.Tensor,
    extended_normal: torch.Tensor,
    predicted_flow_px: torch.Tensor,
    target_flow_px: torch.Tensor,
    band_weight: torch.Tensor,
    radius_px: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Regularize *geometry error*, not the correct deformation itself.

    If prediction equals the operator-matched GT displacement, both terms are
    exactly zero (up to floating point).  This removes the archived SDF-OMW bias
    where a spatially varying but correct contour deformation was penalized.
    """
    pred = predicted_extended_offset_px
    tgt = target_extended_offset_px
    if pred.ndim == 3:
        pred = pred[:, None]
    if tgt.ndim == 3:
        tgt = tgt[:, None]
    error = (pred - tgt) / max(float(radius_px), 1.0)
    kx = error.new_tensor([[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]])[:, None] / 8.0
    ky = error.new_tensor([[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]])[:, None] / 8.0
    gx = F.conv2d(error, kx, padding=1)[:, 0]
    gy = F.conv2d(error, ky, padding=1)[:, 0]
    tx = -extended_normal[:, 1]
    ty = extended_normal[:, 0]
    tangent_error = (gx * tx + gy * ty).abs()
    tangent_loss = _weighted_mean(tangent_error, band_weight)

    # Jacobian regularity is also applied to the *error flow*.  Therefore the
    # exact target field has identity error transform and zero regularization.
    error_flow = predicted_flow_px - target_flow_px
    if error_flow.shape[-2] < 2 or error_flow.shape[-1] < 2:
        fold_error = error_flow.sum() * 0.0
    else:
        ux, uy = error_flow[:, 0], error_flow[:, 1]
        dux_dx = ux[:, :-1, 1:] - ux[:, :-1, :-1]
        dux_dy = ux[:, 1:, :-1] - ux[:, :-1, :-1]
        duy_dx = uy[:, :-1, 1:] - uy[:, :-1, :-1]
        duy_dy = uy[:, 1:, :-1] - uy[:, :-1, :-1]
        det = (1.0 + dux_dx) * (1.0 + duy_dy) - dux_dy * duy_dx
        cell_weight = 0.25 * (
            band_weight[:, :-1, :-1] + band_weight[:, :-1, 1:]
            + band_weight[:, 1:, :-1] + band_weight[:, 1:, 1:]
        )
        fold_error = _weighted_mean(F.relu(-det), cell_weight)
    geometry = torch.stack([tangent_loss, fold_error]).mean()
    return geometry, tangent_loss, fold_error


def _warp_logits_with_contour_offset(
    base_logits: torch.Tensor,
    owner_offset_px: torch.Tensor,
    owner_flat_index: torch.Tensor,
    owner_case_valid: torch.Tensor,
    extended_normal: torch.Tensor,
    band_weight: torch.Tensor,
) -> torch.Tensor:
    """Diagnostics-only warp using the exact posterior-stable physical operator."""
    if owner_offset_px.ndim == 3:
        owner_offset_px = owner_offset_px[:, None]
    extended = gather_owner_field(owner_offset_px, owner_flat_index, owner_case_valid)
    flow = extended * band_weight[:, None].to(extended) * extended_normal
    b, _, h, w = base_logits.shape
    ys = torch.linspace(-1.0, 1.0, h, device=base_logits.device, dtype=base_logits.dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=base_logits.device, dtype=base_logits.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    base_grid = torch.stack([gx, gy], dim=-1)[None].expand(b, h, w, 2)
    grid_x = base_grid[..., 0] + (2.0 * flow[:, 0] / float(max(w - 1, 1)) if w > 1 else 0.0)
    grid_y = base_grid[..., 1] + (2.0 * flow[:, 1] / float(max(h - 1, 1)) if h > 1 else 0.0)
    grid = torch.stack([grid_x, grid_y], dim=-1)
    warped = F.grid_sample(base_logits, grid, mode="bilinear", padding_mode="border", align_corners=True)
    identity = F.grid_sample(base_logits, base_grid, mode="bilinear", padding_mode="border", align_corners=True)
    return base_logits + (warped - identity)


def _grad_proxy(loss: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
    grad = torch.autograd.grad(loss, field, retain_graph=True, allow_unused=True)[0]
    if grad is None:
        return torch.zeros_like(field)
    return grad


def _cosine_flat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    av, bv = a.reshape(-1), b.reshape(-1)
    denom = av.norm() * bv.norm()
    if float(denom.detach()) <= 1.0e-12:
        return denom.detach().new_zeros(())
    return (av * bv).sum() / denom.clamp_min(1.0e-12)


def _compute_posterior_stable_operator_warp_loss(
    cfg: Any,
    masks: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    epoch: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Posterior-stable, continuously extended operator-matched contour loss."""
    del epoch
    required = (
        "geotr_m1_base_logits", "geotr_m1_final_logits",
        "geotr_m1_predicted_owner_sample_offset_px", "geotr_m1_extended_sample_offset_px",
        "geotr_m1_flow_px", "geotr_m1_contour_owner_normal", "geotr_m1_boundary_normal",
        "geotr_m1_base_sdf", "geotr_m1_contour_owner_mask", "geotr_m1_owner_flat_index",
        "geotr_m1_owner_case_valid", "geotr_m1_band_weight", "geotr_m1_warp_support",
        "geotr_m1_local_radius_px",
    )
    missing = [k for k in required if k not in aux]
    if missing:
        raise RuntimeError(f"Posterior-stable operator warp missing aux keys: {missing}")

    base_logits = aux["geotr_m1_base_logits"].detach()
    final_logits = aux["geotr_m1_final_logits"]
    predicted_owner = aux["geotr_m1_predicted_owner_sample_offset_px"]
    extended_pred = aux["geotr_m1_extended_sample_offset_px"]
    flow_px = aux["geotr_m1_flow_px"]
    owner_normal = aux["geotr_m1_contour_owner_normal"].detach()
    extended_normal = aux["geotr_m1_boundary_normal"].detach()
    base_sdf = aux["geotr_m1_base_sdf"].detach()
    owner = aux["geotr_m1_contour_owner_mask"].detach().bool()
    owner_idx = aux["geotr_m1_owner_flat_index"].detach().long()
    owner_case_valid = aux["geotr_m1_owner_case_valid"].detach().bool()
    band_weight = aux["geotr_m1_band_weight"].detach().to(final_logits)
    support = aux["geotr_m1_warp_support"].detach().bool()
    radius = int(round(float(aux["geotr_m1_local_radius_px"].detach().float().mean().item())))
    if radius <= 0:
        raise ValueError("Posterior-stable operator warp requires a positive radius")

    target = _target3(masks, final_logits.shape[-2:])
    target_hard = target >= 0.5
    target_offset, valid_owner, reachable, clipped, motion_abs, normal_alignment = (
        operator_matched_normal_ray_target(
            base_owner=owner,
            base_sdf=base_sdf,
            base_normal=owner_normal,
            gt_mask=target_hard,
            radius_px=radius,
        )
    )

    disp_map = F.smooth_l1_loss(
        predicted_owner / float(radius),
        target_offset.to(predicted_owner) / float(radius),
        reduction="none",
    )
    displacement_loss = _masked_mean(disp_map, valid_owner)

    final_prob = torch.sigmoid(final_logits).clamp(EPS, 1.0 - EPS)
    base_prob = torch.sigmoid(base_logits).clamp(EPS, 1.0 - EPS)
    bce_map = F.binary_cross_entropy_with_logits(
        final_logits[:, 0].float(), target.float(), reduction="none"
    ).to(final_logits)
    local_bce = _weighted_mean(bce_map, band_weight)
    final_dice_case = _dice_per_case(final_prob, target)
    base_dice_case = _dice_per_case(base_prob, target)
    dice_loss = 1.0 - final_dice_case.mean()
    segmentation_loss = torch.stack([local_bce, dice_loss]).mean()

    target_owner_field = target_offset.to(predicted_owner)[:, None]
    extended_target = gather_owner_field(target_owner_field, owner_idx, owner_case_valid)
    target_flow = extended_target * band_weight[:, None] * extended_normal
    geometry_loss, tangent_error_loss, fold_error_loss = _posterior_stable_geometry_regularization(
        predicted_extended_offset_px=extended_pred,
        target_extended_offset_px=extended_target,
        extended_normal=extended_normal,
        predicted_flow_px=flow_px,
        target_flow_px=target_flow,
        band_weight=band_weight,
        radius_px=float(radius),
    )
    objective = torch.stack([displacement_loss, segmentation_loss, geometry_loss]).mean()

    # Gradient audit must differentiate with respect to the parent tensor that
    # actually feeds both owner extension and the physical warp.  Auditing a
    # sibling view (owner_scalar[:,0]) makes task/geometry gradients look unused.
    audit_field = aux.get("geotr_m1_owner_scalar_field", predicted_owner)
    grad_audit = bool(_cfg_get(_cfg_get(cfg, "M1", None), "SEMLT_GRADIENT_AUDIT", False))
    if grad_audit and torch.is_grad_enabled() and audit_field.requires_grad:
        gd = _grad_proxy(displacement_loss, audit_field)
        gs = _grad_proxy(segmentation_loss, audit_field)
        gg = _grad_proxy(geometry_loss, audit_field)
        gd_norm, gs_norm, gg_norm = gd.norm(), gs.norm(), gg.norm()
        cos_ds, cos_dg, cos_sg = _cosine_flat(gd, gs), _cosine_flat(gd, gg), _cosine_flat(gs, gg)
    else:
        z = objective.detach().new_zeros(())
        gd_norm = gs_norm = gg_norm = cos_ds = cos_dg = cos_sg = z

    with torch.no_grad():
        owner_count = valid_owner.float().sum().clamp_min(1.0)
        weighted_band_count = band_weight.sum().clamp_min(1.0)
        reachable_count = reachable.float().sum()
        target_abs = target_offset.abs().to(predicted_owner)
        pred_det = predicted_owner.detach()
        pred_abs = pred_det.abs()
        offset_error = (pred_det - target_offset.to(predicted_owner)).abs()
        nonzero = valid_owner & (target_abs > 1.0e-6)
        tpos = valid_owner & (target_offset > 1.0e-6)
        tneg = valid_owner & (target_offset < -1.0e-6)
        ppos = valid_owner & (pred_det > 0)
        pneg = valid_owner & (pred_det < 0)
        correct_pos = (pred_det[tpos] > 0).float().mean() if bool(tpos.any()) else pred_det.new_zeros(())
        correct_neg = (pred_det[tneg] < 0).float().mean() if bool(tneg.any()) else pred_det.new_zeros(())
        balanced_dir = 0.5 * (correct_pos + correct_neg) if bool(tpos.any()) and bool(tneg.any()) else (
            correct_pos if bool(tpos.any()) else correct_neg
        )
        direction_accuracy = (
            (pred_det[nonzero] * target_offset.to(predicted_owner)[nonzero] > 0).float().mean()
            if bool(nonzero.any()) else pred_det.new_zeros(())
        )
        tp = ((pred_det > 0) & tpos).float().sum(); tn = ((pred_det < 0) & tneg).float().sum()
        fp = ((pred_det > 0) & tneg).float().sum(); fn = ((pred_det < 0) & tpos).float().sum()
        mcc_den = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)).sqrt()
        direction_mcc = ((tp*tn - fp*fn) / mcc_den.clamp_min(1.0)) if bool(nonzero.any()) else pred_det.new_zeros(())

        base_error = ((base_prob[:, 0] >= 0.5) != target_hard)
        final_error = ((final_prob[:, 0] >= 0.5) != target_hard)
        corrected = base_error & ~final_error
        introduced = ~base_error & final_error
        zero = objective.detach().new_zeros(())
        clipped_fraction = clipped.float().sum() / owner_count
        reachable_fraction = reachable_count / owner_count
        amplitude_ratio = _masked_mean(pred_abs, valid_owner) / _masked_mean(target_abs, valid_owner).clamp_min(EPS)
        align_mask = reachable & (normal_alignment > 0)

        # Causal oracle decomposition using exactly the deployed physical operator.
        sign_t = torch.sign(target_offset.to(predicted_owner))
        sign_p = torch.sign(pred_det)
        oracle_sign_owner = sign_t * pred_abs
        oracle_mag_owner = sign_p * target_abs
        full_oracle_owner = target_offset.to(predicted_owner)
        oracle_sign_logits = _warp_logits_with_contour_offset(
            base_logits, oracle_sign_owner, owner_idx, owner_case_valid, extended_normal, band_weight
        )
        oracle_mag_logits = _warp_logits_with_contour_offset(
            base_logits, oracle_mag_owner, owner_idx, owner_case_valid, extended_normal, band_weight
        )
        full_oracle_logits = _warp_logits_with_contour_offset(
            base_logits, full_oracle_owner, owner_idx, owner_case_valid, extended_normal, band_weight
        )
        oracle_sign_dice = _dice_per_case(torch.sigmoid(oracle_sign_logits), target).mean()
        oracle_mag_dice = _dice_per_case(torch.sigmoid(oracle_mag_logits), target).mean()
        full_oracle_dice = _dice_per_case(torch.sigmoid(full_oracle_logits), target).mean()
        base_dice = base_dice_case.mean()

    diagnostics: Dict[str, torch.Tensor] = {
        "mhcs_total_loss": objective.detach(),
        "mhcs_final_loss": segmentation_loss.detach(),
        "mhcs_ce_loss": local_bce.detach(),
        "mhcs_dice_loss": dice_loss.detach(),
        "mhcs_m1_objective": objective.detach(),
        "mhcs_m2_objective": objective.detach().new_zeros(()),
        "geotr_m1_total_loss": objective.detach(),
        "geotr_m1_segmentation_loss": segmentation_loss.detach(),
        "geotr_m1_deploy_segmentation_loss": segmentation_loss.detach(),
        "geotr_m1_bce_loss": local_bce.detach(),
        "geotr_m1_dice_loss": dice_loss.detach(),
        "geotr_m1_boundary_loss": displacement_loss.detach(),
        "geotr_m1_state_loss": displacement_loss.detach(),
        "geotr_m1_correction_loss": displacement_loss.detach(),
        "geotr_m1_transport_loss": torch.stack([displacement_loss, segmentation_loss]).mean().detach(),
        "geotr_m1_safety_loss": geometry_loss.detach(),
        "geotr_m1_displacement_loss": displacement_loss.detach(),
        "geotr_m1_geometry_regularization": geometry_loss.detach(),
        "geotr_m1_boundary_tv_loss": tangent_error_loss.detach(),
        "geotr_m1_fold_loss": fold_error_loss.detach(),
        "geotr_m1_tangent_error_loss": tangent_error_loss.detach(),
        "geotr_m1_error_fold_loss": fold_error_loss.detach(),
        "geotr_m1_base_error_fraction": base_error.float().mean().detach(),
        "geotr_m1_correctable_fraction": valid_owner.float().mean().detach(),
        "geotr_m1_beneficial_fraction": corrected.float().mean().detach(),
        "geotr_m1_base_boundary_fraction": owner.float().mean().detach(),
        "geotr_m1_contour_owner_fraction": owner.float().mean().detach(),
        "geotr_m1_warp_support_fraction": support.float().mean().detach(),
        "geotr_m1_weighted_band_fraction": band_weight.mean().detach(),
        "geotr_m1_boundary_reachable_fraction": reachable_fraction.detach(),
        "geotr_m1_reachable_support_fraction": (reachable_count / weighted_band_count).detach(),
        "geotr_m1_target_clipped_fraction": clipped_fraction.detach(),
        "geotr_m1_target_sample_offset_abs_px": _masked_mean(target_abs, valid_owner).detach(),
        "geotr_m1_predicted_sample_offset_abs_px": _masked_mean(pred_abs, valid_owner).detach(),
        "geotr_m1_sample_offset_mae_px": _masked_mean(offset_error, valid_owner).detach(),
        "geotr_m1_displacement_amplitude_ratio": amplitude_ratio.detach(),
        "geotr_m1_displacement_direction_accuracy": direction_accuracy.detach(),
        "geotr_m1_direction_balanced_accuracy": balanced_dir.detach(),
        "geotr_m1_direction_mcc": direction_mcc.detach(),
        "geotr_m1_target_positive_fraction": (tpos.float().sum()/owner_count).detach(),
        "geotr_m1_target_negative_fraction": (tneg.float().sum()/owner_count).detach(),
        "geotr_m1_predicted_positive_fraction": (ppos.float().sum()/owner_count).detach(),
        "geotr_m1_predicted_negative_fraction": (pneg.float().sum()/owner_count).detach(),
        "geotr_m1_gt_correspondence_distance_px": _masked_mean(motion_abs.to(predicted_owner), valid_owner).detach(),
        "geotr_m1_gt_normal_alignment": _masked_mean(normal_alignment.to(predicted_owner), align_mask).detach(),
        "geotr_m1_local_bce_loss": local_bce.detach(),
        "geotr_m1_corrected_fraction": corrected.float().mean().detach(),
        "geotr_m1_introduced_fraction": introduced.float().mean().detach(),
        "geotr_m1_boundary_corrected_fraction": (corrected & support).float().mean().detach(),
        "geotr_m1_boundary_introduced_fraction": (introduced & support).float().mean().detach(),
        "geotr_m1_net_error_change_fraction": (introduced.float().mean() - corrected.float().mean()).detach(),
        "geotr_m1_flow_rms_px": _mean_aux(aux, "geotopo_flow_rms_px", objective),
        "geotr_m1_flow_max_px": _mean_aux(aux, "geotopo_flow_max_px", objective),
        "geotr_m1_geometry_abs_change": _mean_aux(aux, "geotopo_geometry_abs_change", objective),
        "geotr_m1_range_violation_fraction": _mean_aux(aux, "geotr_m1_range_violation_fraction", objective),
        "geotr_m1_range_violation_max": _mean_aux(aux, "geotr_m1_range_violation_max", objective),
        "geotr_m1_autozero_trust_mean": _mean_aux(aux, "geotr_m1_autozero_trust_mean", objective),
        "geotr_m1_autozero_trust_std": _mean_aux(aux, "geotr_m1_autozero_trust_std", objective),
        "geotr_m1_posterior_stable_operator_warp": objective.detach().new_ones(()),
        "geotr_m1_sdf_operator_matched_warp": zero,
        "geotr_m1_boundary_normal_warp": zero,
        "geotr_m1_memory_exact": objective.detach().new_ones(()),
        "geotr_m1_autozero_objective": objective.detach().new_ones(()),
        "geotr_m1_has_m2": zero,
        "geotr_m1_grad_disp_norm": gd_norm.detach(),
        "geotr_m1_grad_seg_norm": gs_norm.detach(),
        "geotr_m1_grad_geo_norm": gg_norm.detach(),
        "geotr_m1_grad_cos_disp_seg": cos_ds.detach(),
        "geotr_m1_grad_cos_disp_geo": cos_dg.detach(),
        "geotr_m1_grad_cos_seg_geo": cos_sg.detach(),
        "geotr_m1_oracle_sign_dice": oracle_sign_dice.detach(),
        "geotr_m1_oracle_magnitude_dice": oracle_mag_dice.detach(),
        "geotr_m1_oracle_full_dice": full_oracle_dice.detach(),
        "geotr_m1_oracle_sign_gain": (oracle_sign_dice-base_dice).detach(),
        "geotr_m1_oracle_magnitude_gain": (oracle_mag_dice-base_dice).detach(),
        "geotr_m1_oracle_full_gain": (full_oracle_dice-base_dice).detach(),
        "geotr_m1_train_posterior_samples": _mean_aux(aux, "geotr_train_posterior_samples", objective),
    }
    return objective, diagnostics



def _compute_uc_fnrt_loss(
    cfg: Any,
    masks: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    epoch: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """UC-FNRT: uncertainty-aware factorized normal-ray transport loss.

    Only presence-compatible contours receive explicit direction/magnitude
    geometry supervision.  Empty/full-GT cases remain visible to the real
    segmentation task but do not fabricate a +/-radius geometric correspondence.
    """
    del epoch
    m1_cfg = _cfg_get(cfg, "M1", None)
    uc_ablation = str(
        _cfg_get(m1_cfg, "SEMLT_UC_FNRT_ABLATION", "full") or "full"
    ).strip().lower()
    offset_distribution = bool(
        _cfg_get(m1_cfg, "SEMLT_UC_OFFSET_DISTRIBUTION", False)
    )
    mrm_structured_loss = bool(
        _cfg_get(m1_cfg, "SEMLT_UC_MRM_STRUCTURED_LOSS", False)
    )
    ordered_cdf_loss = bool(
        _cfg_get(m1_cfg, "SEMLT_UC_ORDERED_CDF_LOSS", False)
    )
    hierarchical_ordinal_loss = bool(
        _cfg_get(m1_cfg, "SEMLT_UC_HIERARCHICAL_ORDINAL_LOSS", False)
    )
    hierarchical_joint_proper_score = bool(
        _cfg_get(
            m1_cfg,
            "SEMLT_UC_HIERARCHICAL_JOINT_PROPER_SCORE",
            False,
        )
    )
    case_balanced_owners = bool(
        _cfg_get(m1_cfg, "SEMLT_UC_CASE_BALANCED_OWNERS", False)
    )
    sign_class_balanced_legacy = bool(
        _cfg_get(m1_cfg, "SEMLT_UC_SIGN_CLASS_BALANCED_LEGACY", False)
    )
    reachable_match_only = bool(
        _cfg_get(m1_cfg, "SEMLT_UC_REACHABLE_MATCH_ONLY", False)
    )
    censored_endpoint_action = bool(
        _cfg_get(m1_cfg, "SEMLT_UC_CENSORED_ENDPOINT_ACTION", False)
    )
    if mrm_structured_loss and not offset_distribution:
        raise RuntimeError("MRM structured loss requires distributional offsets")
    if ordered_cdf_loss and not offset_distribution:
        raise RuntimeError("ordered CDF loss requires distributional offsets")
    if ordered_cdf_loss and mrm_structured_loss:
        raise RuntimeError(
            "ordered CDF loss replaces, rather than stacks with, MRM structured ranking"
        )
    if hierarchical_ordinal_loss and not offset_distribution:
        raise RuntimeError("hierarchical ordinal loss requires distributional offsets")
    if hierarchical_ordinal_loss and (ordered_cdf_loss or mrm_structured_loss):
        raise RuntimeError(
            "hierarchical ordinal loss replaces joint CDF/structured objectives; "
            "do not stack them"
        )
    if hierarchical_joint_proper_score and not hierarchical_ordinal_loss:
        raise RuntimeError(
            "hierarchical joint proper score requires hierarchical ordinal loss"
        )
    required = (
        "geotr_m1_base_logits", "geotr_m1_final_logits",
        "geotr_m1_predicted_owner_sample_offset_px", "geotr_m1_extended_sample_offset_px",
        "geotr_m1_flow_px", "geotr_m1_contour_owner_normal", "geotr_m1_boundary_normal",
        "geotr_m1_base_sdf", "geotr_m1_contour_owner_mask", "geotr_m1_owner_flat_index",
        "geotr_m1_owner_case_valid", "geotr_m1_band_weight", "geotr_m1_warp_support",
        "geotr_m1_local_radius_px", "geotr_m1_direction_raw", "geotr_m1_magnitude_raw",
        "geotr_m1_magnitude_px",
    )
    missing = [k for k in required if k not in aux]
    if offset_distribution:
        missing.extend(
            k
            for k in (
                "geotr_m1_offset_logits",
                "geotr_m1_offset_probs",
                "geotr_m1_offset_keep_prob",
                "geotr_m1_offset_entropy",
            )
            if k not in aux
        )
    if missing:
        raise RuntimeError(f"UC-FNRT missing aux keys: {sorted(set(missing))}")

    base_logits = aux["geotr_m1_base_logits"].detach()
    final_logits = aux["geotr_m1_final_logits"]
    predicted_owner = aux["geotr_m1_predicted_owner_sample_offset_px"]
    direction_raw_field = aux["geotr_m1_direction_raw"]
    magnitude_raw_field = aux["geotr_m1_magnitude_raw"]
    direction_raw = direction_raw_field[:, 0]
    magnitude_raw = magnitude_raw_field[:, 0]
    magnitude_px = aux["geotr_m1_magnitude_px"][:, 0]
    offset_logits_field = aux.get("geotr_m1_offset_logits", None)
    offset_probs_field = aux.get("geotr_m1_offset_probs", None)
    offset_entropy_field = aux.get("geotr_m1_offset_entropy", None)
    offset_keep_prob_field = aux.get("geotr_m1_offset_keep_prob", None)
    offset_global_expectation_field = aux.get("geotr_m1_offset_global_expectation_px", None)
    offset_dominant_center_field = aux.get("geotr_m1_offset_dominant_center_px", None)
    offset_dominant_mass_field = aux.get("geotr_m1_offset_dominant_mass", None)
    extended_pred = aux["geotr_m1_extended_sample_offset_px"]
    flow_px = aux["geotr_m1_flow_px"]
    owner_normal = aux["geotr_m1_contour_owner_normal"].detach()
    extended_normal = aux["geotr_m1_boundary_normal"].detach()
    base_sdf = aux["geotr_m1_base_sdf"].detach()
    owner = aux["geotr_m1_contour_owner_mask"].detach().bool()
    owner_idx = aux["geotr_m1_owner_flat_index"].detach().long()
    owner_case_valid = aux["geotr_m1_owner_case_valid"].detach().bool()
    band_weight = aux["geotr_m1_band_weight"].detach().to(final_logits)
    support = aux["geotr_m1_warp_support"].detach().bool()
    radius = int(round(float(aux["geotr_m1_local_radius_px"].detach().float().mean().item())))
    if radius <= 0:
        raise ValueError("UC-FNRT requires a positive radius")

    target = _target3(masks, final_logits.shape[-2:])
    target_hard = target >= 0.5
    target_offset_raw, valid_owner_raw, reachable_raw, clipped_raw, motion_abs, normal_alignment = (
        operator_matched_normal_ray_target(
            base_owner=owner,
            base_sdf=base_sdf,
            base_normal=owner_normal,
            gt_mask=target_hard,
            radius_px=radius,
            censored_endpoint_action=censored_endpoint_action,
        )
    )

    # Geometry is defined only when both Base and GT possess an actual contour.
    # This removes the archived empty-GT -> -radius pseudo-correspondence bias.
    gt_nonempty_case = target_hard.flatten(1).any(1)
    gt_full_case = target_hard.flatten(1).all(1)
    geometry_case = gt_nonempty_case & (~gt_full_case) & owner_case_valid
    valid_owner = valid_owner_raw & geometry_case[:, None, None]
    target_offset = torch.where(valid_owner, target_offset_raw, torch.zeros_like(target_offset_raw))
    reachable = reachable_raw & valid_owner
    clipped = clipped_raw & valid_owner

    target_abs = target_offset.abs().to(predicted_owner)
    nonzero_owner = valid_owner & (target_abs > 1.0e-6)
    direction_target = target_offset > 0

    signed_map = F.smooth_l1_loss(
        predicted_owner.float(), target_offset.to(predicted_owner).float(), beta=1.0, reduction="none"
    ).to(predicted_owner) / float(radius)
    reduce_owner = (
        _case_balanced_masked_mean if case_balanced_owners else _masked_mean
    )
    signed_loss = reduce_owner(signed_map, valid_owner)

    offset_nll_loss = signed_loss.detach() * 0.0
    offset_expected_reg_loss = signed_loss.detach() * 0.0
    offset_structured_rank_loss = signed_loss.detach() * 0.0
    offset_ordered_cdf_loss = signed_loss.detach() * 0.0
    offset_sign_loss = signed_loss.detach() * 0.0
    offset_magnitude_nll_loss = signed_loss.detach() * 0.0
    offset_magnitude_cdf_loss = signed_loss.detach() * 0.0
    offset_pred_negative_measure_mean = signed_loss.detach() * 0.0
    offset_pred_keep_measure_mean = signed_loss.detach() * 0.0
    offset_pred_positive_measure_mean = signed_loss.detach() * 0.0
    offset_target_negative_measure_mean = signed_loss.detach() * 0.0
    offset_target_keep_measure_mean = signed_loss.detach() * 0.0
    offset_target_positive_measure_mean = signed_loss.detach() * 0.0
    offset_sign_calibration_l1 = signed_loss.detach() * 0.0
    distribution_owner = reachable if reachable_match_only else valid_owner
    if offset_distribution:
        if not isinstance(offset_logits_field, torch.Tensor):
            raise RuntimeError("DNR loss requires geotr_m1_offset_logits.")
        k = int(offset_logits_field.shape[1])
        expected_k = 2 * radius + 1
        if k != expected_k:
            raise RuntimeError(
                f"DNR offset bins mismatch: got={k}, expected={expected_k}"
            )

        # Continuous operator-matched target -> two neighbouring integer bins.
        # This retains sub-pixel supervision without imposing a hard quantizer.
        target_clamped = target_offset.to(offset_logits_field).clamp(
            -float(radius), float(radius)
        )
        shifted = target_clamped + float(radius)
        lower = torch.floor(shifted).long().clamp(0, k - 1)
        upper = (lower + 1).clamp(0, k - 1)
        upper_w = (shifted - lower.to(shifted)).clamp(0.0, 1.0)
        lower_w = 1.0 - upper_w
        log_prob = F.log_softmax(offset_logits_field.float(), dim=1)
        lower_logp = log_prob.gather(1, lower[:, None])[:, 0]
        upper_logp = log_prob.gather(1, upper[:, None])[:, 0]
        nll_map = -(
            lower_w.float() * lower_logp
            + upper_w.float() * upper_logp
        ).to(predicted_owner)

        # Normalize by log(K): a uniform 17-bin head starts at exactly 1.0,
        # keeping the displacement-group scale comparable to the archived loss.
        offset_nll_loss = reduce_owner(nll_map, distribution_owner) / max(
            math.log(float(k)), EPS
        )
        offset_expected_reg_loss = signed_loss

        if hierarchical_ordinal_loss:
            if not isinstance(offset_probs_field, torch.Tensor):
                raise RuntimeError(
                    "hierarchical ordinal loss requires geotr_m1_offset_probs"
                )

            # Factor the existing joint distribution into three-way sign and
            # sign-conditional magnitude probabilities. No new head,
            # temperature, or dataset-specific coefficient is introduced.
            negative_probs = torch.flip(
                offset_probs_field[:, :radius], dims=(1,)
            )
            keep_prob = offset_probs_field[:, radius]
            positive_probs = offset_probs_field[:, radius + 1 :]
            negative_mass = negative_probs.sum(dim=1)
            positive_mass = positive_probs.sum(dim=1)
            sign_probs = torch.stack(
                [negative_mass, keep_prob, positive_mass], dim=1
            ).clamp_min(EPS)

            lower_signed = lower - radius
            upper_signed = upper - radius
            lower_sign_class = torch.where(
                lower_signed < 0,
                torch.zeros_like(lower_signed),
                torch.where(
                    lower_signed > 0,
                    torch.full_like(lower_signed, 2),
                    torch.ones_like(lower_signed),
                ),
            )
            upper_sign_class = torch.where(
                upper_signed < 0,
                torch.zeros_like(upper_signed),
                torch.where(
                    upper_signed > 0,
                    torch.full_like(upper_signed, 2),
                    torch.ones_like(upper_signed),
                ),
            )
            sign_target_parts = []
            for sign_class in range(3):
                sign_target_parts.append(
                    lower_w.float() * (lower_sign_class == sign_class).float()
                    + upper_w.float() * (upper_sign_class == sign_class).float()
                )
            sign_target_prob = torch.stack(sign_target_parts, dim=1)
            if sign_class_balanced_legacy:
                # Exact archived R2 behaviour, retained only for a matched
                # causal control.  Do not use its probabilities as a calibrated
                # deployment posterior in the R2.1-CAL arm.
                sign_nll_map = -(
                    sign_target_prob * sign_probs.float().log()
                ).sum(dim=1).to(predicted_owner)
                dominant_sign_target = sign_target_prob.argmax(dim=1)
                offset_sign_loss = _class_case_balanced_masked_mean(
                    sign_nll_map,
                    dominant_sign_target,
                    distribution_owner,
                    num_classes=3,
                ) / max(math.log(3.0), EPS)
            else:
                offset_sign_loss = _proper_sign_log_score(
                    sign_probs,
                    sign_target_prob,
                    distribution_owner,
                    case_balanced=case_balanced_owners,
                ) / max(math.log(3.0), EPS)

            # Calibration diagnostics use exactly the same owner measure as
            # the optimized proper sign score.  This avoids comparing a
            # case-balanced loss with a globally pixel-weighted diagnostic.
            offset_pred_negative_measure_mean = reduce_owner(
                negative_mass, distribution_owner
            )
            offset_pred_keep_measure_mean = reduce_owner(
                keep_prob, distribution_owner
            )
            offset_pred_positive_measure_mean = reduce_owner(
                positive_mass, distribution_owner
            )
            offset_target_negative_measure_mean = reduce_owner(
                sign_target_prob[:, 0], distribution_owner
            )
            offset_target_keep_measure_mean = reduce_owner(
                sign_target_prob[:, 1], distribution_owner
            )
            offset_target_positive_measure_mean = reduce_owner(
                sign_target_prob[:, 2], distribution_owner
            )
            offset_sign_calibration_l1 = torch.stack(
                [
                    (offset_pred_negative_measure_mean - offset_target_negative_measure_mean).abs(),
                    (offset_pred_keep_measure_mean - offset_target_keep_measure_mean).abs(),
                    (offset_pred_positive_measure_mean - offset_target_positive_measure_mean).abs(),
                ]
            ).sum()

            target_positive = target_offset.to(predicted_owner) > 0
            chosen_mass = torch.where(
                target_positive, positive_mass, negative_mass
            ).clamp_min(EPS)
            conditional_magnitude_probs = torch.where(
                target_positive[:, None], positive_probs, negative_probs
            ) / chosen_mass[:, None]
            conditional_magnitude_probs = conditional_magnitude_probs.clamp_min(EPS)
            conditional_magnitude_probs = conditional_magnitude_probs / (
                conditional_magnitude_probs.sum(dim=1, keepdim=True).clamp_min(EPS)
            )

            magnitude_shifted = target_abs.clamp(1.0, float(radius)) - 1.0
            magnitude_lower = torch.floor(magnitude_shifted).long().clamp(0, radius - 1)
            magnitude_upper = (magnitude_lower + 1).clamp(0, radius - 1)
            magnitude_upper_w = (
                magnitude_shifted - magnitude_lower.to(magnitude_shifted)
            ).clamp(0.0, 1.0)
            magnitude_lower_w = 1.0 - magnitude_upper_w
            magnitude_log_prob = conditional_magnitude_probs.float().log()
            magnitude_nll_map = -(
                magnitude_lower_w.float()
                * magnitude_log_prob.gather(1, magnitude_lower[:, None])[:, 0]
                + magnitude_upper_w.float()
                * magnitude_log_prob.gather(1, magnitude_upper[:, None])[:, 0]
            ).to(predicted_owner)
            magnitude_owner = distribution_owner & (target_abs >= 0.5)
            offset_magnitude_nll_loss = reduce_owner(
                magnitude_nll_map, magnitude_owner
            ) / max(math.log(float(max(radius, 2))), EPS)
            magnitude_cdf_map = _ordered_magnitude_crps_map(
                conditional_magnitude_probs,
                target_abs,
            )
            offset_magnitude_cdf_loss = reduce_owner(
                magnitude_cdf_map, magnitude_owner
            )
            magnitude_score = torch.stack(
                [offset_magnitude_nll_loss, offset_magnitude_cdf_loss]
            ).mean()
            hierarchical_displacement_loss = torch.stack(
                [offset_sign_loss, magnitude_score]
            ).mean()
            if hierarchical_joint_proper_score:
                # R2.2-JPS: the R2.1 factorized score calibrates the sign
                # marginal and the magnitude conditional on the observed sign,
                # but the conditional normalization cannot distinguish whether
                # probability on the *opposite* side is one or eight pixels
                # away.  Deployment consumes the full signed posterior through
                # E[D], so add a full-joint log-score + discrete-CRPS group over
                # the same 17 logits and the same reachable owner measure.
                #
                # Both groups are proper scores. Their fixed arithmetic mean
                # introduces no dataset-specific coefficient, new head,
                # temperature, threshold, or decoder change.  The R2.1 path is
                # bit-for-bit retained when this explicit flag is false.
                signed_cdf_map = _ordered_offset_crps_map(
                    offset_probs_field,
                    target_offset,
                    radius,
                )
                offset_ordered_cdf_loss = reduce_owner(
                    signed_cdf_map, distribution_owner
                )
                joint_proper_score = torch.stack(
                    [offset_nll_loss, offset_ordered_cdf_loss]
                ).mean()
                displacement_loss = torch.stack(
                    [hierarchical_displacement_loss, joint_proper_score]
                ).mean()
            else:
                displacement_loss = hierarchical_displacement_loss

        elif ordered_cdf_loss:
            if not isinstance(offset_probs_field, torch.Tensor):
                raise RuntimeError("ordered CDF loss requires geotr_m1_offset_probs")
            cdf_map = _ordered_offset_crps_map(
                offset_probs_field,
                target_offset,
                radius,
            )
            offset_ordered_cdf_loss = reduce_owner(cdf_map, distribution_owner)
            # Log score and CRPS are both proper scores.  Their parameter-free
            # mean trains probability mass at the correct ordered bins without
            # the under-identified E[d] regression or a tunable ranking margin.
            displacement_loss = torch.stack(
                [offset_nll_loss, offset_ordered_cdf_loss]
            ).mean()

        elif mrm_structured_loss:
            # Distance-aware structured ranking on the ordered ray.  The two
            # neighbouring target bins define a soft target score; every farther
            # hypothesis must be lower by a margin proportional to its physical
            # displacement error.  This directly trains relative ordering of the
            # cost volume instead of relying on posterior-mean regression.
            logits_float = offset_logits_field.float()
            lower_logit = logits_float.gather(1, lower[:, None])[:, 0]
            upper_logit = logits_float.gather(1, upper[:, None])[:, 0]
            target_score = (
                lower_w.float() * lower_logit
                + upper_w.float() * upper_logit
            )
            values = torch.arange(
                -radius, radius + 1,
                device=logits_float.device,
                dtype=logits_float.dtype,
            ).view(1, k, 1, 1)
            distance_margin = (
                values - target_clamped[:, None].float()
            ).abs() / float(radius)
            # Do not rank the two interpolating target bins against themselves.
            near_target = torch.zeros_like(logits_float, dtype=torch.bool)
            near_target.scatter_(1, lower[:, None], True)
            near_target.scatter_(1, upper[:, None], True)
            rank_mask = valid_owner[:, None] & (~near_target)
            rank_term = F.softplus(
                logits_float - target_score[:, None] + distance_margin
            )
            rank_weight = rank_mask.to(rank_term)
            offset_structured_rank_loss = (
                (rank_term * rank_weight).sum()
                / rank_weight.sum().clamp_min(1.0)
            ).to(predicted_owner)
            displacement_loss = torch.stack(
                [
                    offset_nll_loss,
                    offset_expected_reg_loss,
                    offset_structured_rank_loss,
                ]
            ).mean()
        else:
            displacement_loss = torch.stack(
                [offset_nll_loss, offset_expected_reg_loss]
            ).mean()

        # Legacy factor losses are intentionally not optimized in DNR; the
        # offset distribution jointly owns sign, magnitude and KEEP.
        direction_loss = direction_raw.sum() * 0.0
        magnitude_loss = magnitude_raw.sum() * 0.0
    elif uc_ablation == "direct_signed":
        # F-control uses direct signed pixel-space regression; decomposition
        # losses are zero diagnostics, not hidden auxiliary supervision.
        direction_loss = direction_raw.sum() * 0.0
        magnitude_loss = magnitude_raw.sum() * 0.0
        displacement_loss = signed_loss
    else:
        # Explicit direction supervision is class-balanced on the valid
        # geometric domain, preventing a global inward/outward prior.
        direction_terms: List[torch.Tensor] = []
        direction_bce_map = F.binary_cross_entropy_with_logits(
            direction_raw.float(), direction_target.float(), reduction="none"
        ).to(direction_raw)
        for cls in (False, True):
            m = nonzero_owner & (direction_target == cls)
            if bool(m.any()):
                direction_terms.append(direction_bce_map[m].mean())
        direction_loss = torch.stack(direction_terms).mean() if direction_terms else direction_raw.sum() * 0.0

        magnitude_map = F.smooth_l1_loss(
            magnitude_px.float(), target_abs.float(), beta=1.0, reduction="none"
        ).to(magnitude_px) / float(radius)
        magnitude_loss = _masked_mean(magnitude_map, valid_owner)
        displacement_loss = torch.stack([direction_loss, magnitude_loss, signed_loss]).mean()

    final_prob = torch.sigmoid(final_logits).clamp(EPS, 1.0 - EPS)
    base_prob = torch.sigmoid(base_logits).clamp(EPS, 1.0 - EPS)
    bce_map = F.binary_cross_entropy_with_logits(
        final_logits[:, 0].float(), target.float(), reduction="none"
    ).to(final_logits)
    local_bce = _weighted_mean(bce_map, band_weight)
    final_dice_case = _dice_per_case(final_prob, target)
    base_dice_case = _dice_per_case(base_prob, target)
    dice_loss = 1.0 - final_dice_case.mean()
    segmentation_loss = torch.stack([local_bce, dice_loss]).mean()

    target_owner_field = target_offset.to(predicted_owner)[:, None]
    extended_target = gather_owner_field(target_owner_field, owner_idx, owner_case_valid)
    geometry_band_weight = band_weight * geometry_case[:, None, None].to(band_weight)
    target_flow = extended_target * geometry_band_weight[:, None] * extended_normal
    geometry_loss, tangent_error_loss, fold_error_loss = _posterior_stable_geometry_regularization(
        predicted_extended_offset_px=extended_pred,
        target_extended_offset_px=extended_target,
        extended_normal=extended_normal,
        predicted_flow_px=flow_px,
        target_flow_px=target_flow,
        band_weight=geometry_band_weight,
        radius_px=float(radius),
    )
    if uc_ablation == "segmentation_only":
        # G-control leaves the deployed operator untouched but removes all
        # explicit geometric supervision from the optimized M1 objective.
        objective = segmentation_loss
    else:
        objective = torch.stack([displacement_loss, segmentation_loss, geometry_loss]).mean()

    # Gradient audit follows the *actual deployment parent*.  Archived UC-FNRT
    # audits direction/magnitude parents; DNR audits the joint offset logits.
    def _factor_grad(loss_tensor: torch.Tensor):
        grads = torch.autograd.grad(
            loss_tensor,
            (direction_raw_field, magnitude_raw_field),
            retain_graph=True,
            allow_unused=True,
        )
        gd = torch.zeros_like(direction_raw_field) if grads[0] is None else grads[0]
        gm = torch.zeros_like(magnitude_raw_field) if grads[1] is None else grads[1]
        return gd, gm, torch.cat([gd.reshape(-1), gm.reshape(-1)])

    def _offset_grad(loss_tensor: torch.Tensor):
        if not isinstance(offset_logits_field, torch.Tensor):
            raise RuntimeError("DNR gradient audit requires offset logits.")
        grad = torch.autograd.grad(
            loss_tensor,
            offset_logits_field,
            retain_graph=True,
            allow_unused=True,
        )[0]
        return (
            torch.zeros_like(offset_logits_field)
            if grad is None
            else grad
        )

    grad_audit = bool(
        _cfg_get(_cfg_get(cfg, "M1", None), "SEMLT_GRADIENT_AUDIT", False)
    )
    z = objective.detach().new_zeros(())
    grad_offset_disp_norm = grad_offset_seg_norm = grad_offset_geo_norm = z
    if (
        offset_distribution
        and grad_audit
        and torch.is_grad_enabled()
        and isinstance(offset_logits_field, torch.Tensor)
        and offset_logits_field.requires_grad
    ):
        od = _offset_grad(displacement_loss)
        os = _offset_grad(segmentation_loss)
        og = _offset_grad(geometry_loss)
        gdisp, gseg, ggeo = od.reshape(-1), os.reshape(-1), og.reshape(-1)
        gd_norm, gs_norm, gg_norm = gdisp.norm(), gseg.norm(), ggeo.norm()
        cos_ds = _cosine_flat(gdisp, gseg)
        cos_dg = _cosine_flat(gdisp, ggeo)
        cos_sg = _cosine_flat(gseg, ggeo)
        grad_offset_disp_norm = gd_norm
        grad_offset_seg_norm = gs_norm
        grad_offset_geo_norm = gg_norm
        disp_dir_norm = disp_mag_norm = z
        seg_dir_norm = seg_mag_norm = z
        geo_dir_norm = geo_mag_norm = z
    elif (
        grad_audit
        and torch.is_grad_enabled()
        and direction_raw_field.requires_grad
        and magnitude_raw_field.requires_grad
    ):
        dd, dm, gdisp = _factor_grad(displacement_loss)
        sd, sm, gseg = _factor_grad(segmentation_loss)
        gd_, gm_, ggeo = _factor_grad(geometry_loss)
        gd_norm, gs_norm, gg_norm = gdisp.norm(), gseg.norm(), ggeo.norm()
        cos_ds = _cosine_flat(gdisp, gseg)
        cos_dg = _cosine_flat(gdisp, ggeo)
        cos_sg = _cosine_flat(gseg, ggeo)
        disp_dir_norm, disp_mag_norm = dd.norm(), dm.norm()
        seg_dir_norm, seg_mag_norm = sd.norm(), sm.norm()
        geo_dir_norm, geo_mag_norm = gd_.norm(), gm_.norm()
    else:
        gd_norm = gs_norm = gg_norm = cos_ds = cos_dg = cos_sg = z
        disp_dir_norm = disp_mag_norm = z
        seg_dir_norm = seg_mag_norm = z
        geo_dir_norm = geo_mag_norm = z

    with torch.no_grad():
        owner_count = valid_owner.float().sum().clamp_min(1.0)
        weighted_band_count = geometry_band_weight.sum().clamp_min(1.0)
        reachable_count = reachable.float().sum()
        pred_det = predicted_owner.detach()
        pred_abs = pred_det.abs()
        offset_error = (pred_det - target_offset.to(predicted_owner)).abs()
        tpos = valid_owner & (target_offset > 1.0e-6)
        tneg = valid_owner & (target_offset < -1.0e-6)
        ppos = valid_owner & (pred_det > 0)
        pneg = valid_owner & (pred_det < 0)
        correct_pos = (pred_det[tpos] > 0).float().mean() if bool(tpos.any()) else pred_det.new_zeros(())
        correct_neg = (pred_det[tneg] < 0).float().mean() if bool(tneg.any()) else pred_det.new_zeros(())
        balanced_dir = 0.5 * (correct_pos + correct_neg) if bool(tpos.any()) and bool(tneg.any()) else (
            correct_pos if bool(tpos.any()) else correct_neg
        )
        direction_accuracy = (
            (pred_det[nonzero_owner] * target_offset.to(predicted_owner)[nonzero_owner] > 0).float().mean()
            if bool(nonzero_owner.any()) else pred_det.new_zeros(())
        )
        tp = ((pred_det > 0) & tpos).float().sum(); tn = ((pred_det < 0) & tneg).float().sum()
        fp = ((pred_det > 0) & tneg).float().sum(); fn = ((pred_det < 0) & tpos).float().sum()
        mcc_den = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)).sqrt()
        direction_mcc = ((tp*tn - fp*fn) / mcc_den.clamp_min(1.0)) if bool(nonzero_owner.any()) else pred_det.new_zeros(())

        base_error = ((base_prob[:, 0] >= 0.5) != target_hard)
        final_error = ((final_prob[:, 0] >= 0.5) != target_hard)
        corrected = base_error & ~final_error
        introduced = ~base_error & final_error
        clipped_fraction = clipped.float().sum() / owner_count
        reachable_fraction = reachable_count / owner_count
        amplitude_ratio = _masked_mean(pred_abs, valid_owner) / _masked_mean(target_abs, valid_owner).clamp_min(EPS)

        # ROOTCAUSE-A2 realization-chain diagnostics.  These metrics do not
        # change the optimized objective; they expose exactly where physical
        # displacement amplitude is lost.
        if offset_distribution and isinstance(offset_probs_field, torch.Tensor):
            p_pos_det = offset_probs_field[:, radius + 1 :].sum(1).detach()
            p_neg_det = offset_probs_field[:, :radius].sum(1).detach()
            direction_conf_det = (p_pos_det - p_neg_det).abs()
        else:
            direction_conf_det = torch.tanh(direction_raw).detach().abs()
        direction_conf_owner_mean = _masked_mean(direction_conf_det, valid_owner)
        magnitude_target_ratio = (
            _masked_mean(magnitude_px.detach().abs(), valid_owner)
            / _masked_mean(target_abs, valid_owner).clamp_min(EPS)
        )
        preagreement_owner = aux.get(
            "geotr_m1_preagreement_owner_sample_offset_px", predicted_owner
        ).detach()
        preagreement_amplitude_ratio = (
            _masked_mean(preagreement_owner.abs(), valid_owner)
            / _masked_mean(target_abs, valid_owner).clamp_min(EPS)
        )
        postagreement_amplitude_ratio = amplitude_ratio
        agreement_map = aux.get("geotr_m1_ray_agreement_map", None)
        if isinstance(agreement_map, torch.Tensor):
            agreement_det = agreement_map.detach().to(predicted_owner)
            ray_agreement_owner_mean = _masked_mean(agreement_det, valid_owner)
            ray_agreement_nonzero_owner_mean = _masked_mean(agreement_det, nonzero_owner)
            ray_agreement_reachable_mean = _masked_mean(agreement_det, reachable)
        else:
            ray_agreement_owner_mean = pred_det.new_ones(())
            ray_agreement_nonzero_owner_mean = pred_det.new_ones(())
            ray_agreement_reachable_mean = pred_det.new_ones(())
        coherence_suppression_ratio = (
            _masked_mean(pred_abs, valid_owner)
            / _masked_mean(preagreement_owner.abs(), valid_owner).clamp_min(EPS)
        )
        align_mask = reachable & (normal_alignment > 0)

        # DNR-specific calibration/correspondence diagnostics.
        offset_entropy_owner_mean = pred_det.new_zeros(())
        offset_keep_owner_mean = pred_det.new_zeros(())
        offset_mode_mae_px = pred_det.new_zeros(())
        offset_top1_within1px = pred_det.new_zeros(())
        offset_top1_within2px = pred_det.new_zeros(())
        offset_peak_prob_mean = pred_det.new_zeros(())
        offset_peak_margin_mean = pred_det.new_zeros(())
        offset_target_mass_within1px = pred_det.new_zeros(())
        offset_opposite_sign_mass = pred_det.new_zeros(())
        offset_entropy_high_fraction = pred_det.new_zeros(())
        offset_target_near_fraction = pred_det.new_zeros(())
        offset_target_mid_fraction = pred_det.new_zeros(())
        offset_target_far_fraction = pred_det.new_zeros(())
        offset_target_edge_fraction = pred_det.new_zeros(())
        offset_normal_alignment_mean = pred_det.new_zeros(())
        offset_entropy_correct_sign = pred_det.new_zeros(())
        offset_entropy_wrong_sign = pred_det.new_zeros(())
        offset_global_expectation_abs_px = pred_det.new_zeros(())
        offset_dominant_mode_abs_px = pred_det.new_zeros(())
        offset_dominant_center_mae_px = pred_det.new_zeros(())
        offset_dominant_mass_mean = pred_det.new_zeros(())
        offset_cancellation_ratio = pred_det.new_zeros(())
        offset_dominant_sign_accuracy = pred_det.new_zeros(())
        if (
            offset_distribution
            and isinstance(offset_probs_field, torch.Tensor)
            and isinstance(offset_entropy_field, torch.Tensor)
            and isinstance(offset_keep_prob_field, torch.Tensor)
        ):
            probs_det = offset_probs_field.detach()
            entropy_det = offset_entropy_field[:, 0].detach()
            keep_det = offset_keep_prob_field[:, 0].detach()
            offset_entropy_owner_mean = _masked_mean(entropy_det, valid_owner)
            offset_keep_owner_mean = _masked_mean(keep_det, valid_owner)
            values_det = torch.arange(
                -radius,
                radius + 1,
                device=probs_det.device,
                dtype=predicted_owner.dtype,
            )
            mode_idx = probs_det.argmax(dim=1)
            mode_offset = values_det[mode_idx]
            mode_error = (mode_offset - target_offset.to(mode_offset)).abs()
            offset_mode_mae_px = _masked_mean(mode_error, valid_owner)
            offset_top1_within1px = _masked_mean(
                (mode_error <= 1.0).to(predicted_owner), valid_owner
            )
            offset_top1_within2px = _masked_mean(
                (mode_error <= 2.0).to(predicted_owner), valid_owner
            )
            top2 = probs_det.topk(k=min(2, probs_det.shape[1]), dim=1).values
            offset_peak_prob_mean = _masked_mean(top2[:, 0], valid_owner)
            if top2.shape[1] > 1:
                offset_peak_margin_mean = _masked_mean(
                    top2[:, 0] - top2[:, 1], valid_owner
                )
            values_grid = values_det.view(1, -1, 1, 1)
            target_grid = target_offset.to(probs_det)[:, None]
            within1 = (values_grid - target_grid).abs() <= 1.0
            target_mass = (probs_det * within1.to(probs_det)).sum(dim=1)
            offset_target_mass_within1px = _masked_mean(target_mass, valid_owner)
            pos_mass = probs_det[:, radius + 1 :].sum(dim=1)
            neg_mass = probs_det[:, :radius].sum(dim=1)
            target_pos = target_offset.to(probs_det) > 1.0e-6
            target_neg = target_offset.to(probs_det) < -1.0e-6
            opposite = torch.where(target_pos, neg_mass, torch.where(target_neg, pos_mass, torch.zeros_like(pos_mass)))
            offset_opposite_sign_mass = _masked_mean(opposite, nonzero_owner)
            offset_entropy_high_fraction = _masked_mean(
                (entropy_det >= 0.80).to(predicted_owner), valid_owner
            )
            target_abs_det = target_offset.to(predicted_owner).abs()
            offset_target_near_fraction = _masked_mean(
                (target_abs_det <= 1.0).to(predicted_owner), valid_owner
            )
            offset_target_mid_fraction = _masked_mean(
                ((target_abs_det > 1.0) & (target_abs_det <= 4.0)).to(predicted_owner), valid_owner
            )
            offset_target_far_fraction = _masked_mean(
                (target_abs_det > 4.0).to(predicted_owner), valid_owner
            )
            offset_target_edge_fraction = _masked_mean(
                (target_abs_det >= float(radius) - 0.5).to(predicted_owner), valid_owner
            )
            offset_normal_alignment_mean = _masked_mean(
                normal_alignment.to(predicted_owner), valid_owner
            )
            sign_correct_mask = nonzero_owner & (
                pred_det * target_offset.to(predicted_owner) > 0
            )
            sign_wrong_mask = nonzero_owner & (~sign_correct_mask)
            if bool(sign_correct_mask.any()):
                offset_entropy_correct_sign = _masked_mean(
                    entropy_det, sign_correct_mask
                )
            if bool(sign_wrong_mask.any()):
                offset_entropy_wrong_sign = _masked_mean(
                    entropy_det, sign_wrong_mask
                )

            if isinstance(offset_global_expectation_field, torch.Tensor):
                global_exp = offset_global_expectation_field[:, 0].detach().to(predicted_owner)
            else:
                global_exp = pred_det
            offset_global_expectation_abs_px = _masked_mean(global_exp.abs(), valid_owner)
            offset_dominant_mode_abs_px = _masked_mean(pred_det.abs(), valid_owner)
            offset_cancellation_ratio = (
                offset_global_expectation_abs_px
                / _masked_mean(magnitude_px.detach().abs(), valid_owner).clamp_min(EPS)
            )
            if isinstance(offset_dominant_center_field, torch.Tensor):
                dominant_center = offset_dominant_center_field[:, 0].detach().to(predicted_owner)
                offset_dominant_center_mae_px = _masked_mean(
                    (dominant_center - target_offset.to(dominant_center)).abs(), valid_owner
                )
                dominant_sign_ok = (
                    torch.sign(dominant_center) == torch.sign(target_offset.to(dominant_center))
                ).to(predicted_owner)
                offset_dominant_sign_accuracy = _masked_mean(
                    dominant_sign_ok, nonzero_owner
                )
            if isinstance(offset_dominant_mass_field, torch.Tensor):
                dominant_mass_det = offset_dominant_mass_field[:, 0].detach().to(predicted_owner)
                offset_dominant_mass_mean = _masked_mean(dominant_mass_det, valid_owner)

        sign_t = torch.sign(target_offset.to(predicted_owner))
        sign_p = torch.sign(pred_det)
        oracle_sign_owner = sign_t * pred_abs
        oracle_mag_owner = sign_p * target_abs
        full_oracle_owner = target_offset.to(predicted_owner)
        oracle_sign_logits = _warp_logits_with_contour_offset(
            base_logits, oracle_sign_owner, owner_idx, owner_case_valid, extended_normal, band_weight
        )
        oracle_mag_logits = _warp_logits_with_contour_offset(
            base_logits, oracle_mag_owner, owner_idx, owner_case_valid, extended_normal, band_weight
        )
        full_oracle_logits = _warp_logits_with_contour_offset(
            base_logits, full_oracle_owner, owner_idx, owner_case_valid, extended_normal, band_weight
        )
        oracle_sign_dice = _dice_per_case(torch.sigmoid(oracle_sign_logits), target).mean()
        oracle_mag_dice = _dice_per_case(torch.sigmoid(oracle_mag_logits), target).mean()
        full_oracle_dice = _dice_per_case(torch.sigmoid(full_oracle_logits), target).mean()
        base_dice = base_dice_case.mean(); final_dice = final_dice_case.mean()
        gt_empty_case = ~gt_nonempty_case
        empty_gain = (final_dice_case[gt_empty_case]-base_dice_case[gt_empty_case]).mean() if bool(gt_empty_case.any()) else base_dice.new_zeros(())
        nonempty_gain = (final_dice_case[gt_nonempty_case]-base_dice_case[gt_nonempty_case]).mean() if bool(gt_nonempty_case.any()) else base_dice.new_zeros(())

    zero = objective.detach().new_zeros(())
    diagnostics: Dict[str, torch.Tensor] = {
        "mhcs_total_loss": objective.detach(),
        "mhcs_final_loss": segmentation_loss.detach(),
        "mhcs_ce_loss": local_bce.detach(),
        "mhcs_dice_loss": dice_loss.detach(),
        "mhcs_base_dice": base_dice.detach(),
        "mhcs_final_dice": final_dice.detach(),
        "mhcs_final_gain": (final_dice-base_dice).detach(),
        "mhcs_m1_objective": objective.detach(),
        "mhcs_m2_objective": zero,
        "geotr_m1_total_loss": objective.detach(),
        "geotr_m1_segmentation_loss": segmentation_loss.detach(),
        "geotr_m1_deploy_segmentation_loss": segmentation_loss.detach(),
        "geotr_m1_bce_loss": local_bce.detach(),
        "geotr_m1_dice_loss": dice_loss.detach(),
        "geotr_m1_displacement_loss": displacement_loss.detach(),
        "geotr_m1_direction_loss": direction_loss.detach(),
        "geotr_m1_magnitude_loss": magnitude_loss.detach(),
        "geotr_m1_signed_consistency_loss": signed_loss.detach(),
        "geotr_m1_offset_distribution_enabled": objective.detach().new_tensor(
            float(offset_distribution)
        ),
        "geotr_m1_offset_nll_loss": offset_nll_loss.detach(),
        "geotr_m1_offset_expected_reg_loss": offset_expected_reg_loss.detach(),
        "geotr_m1_offset_structured_rank_loss": offset_structured_rank_loss.detach(),
        "geotr_m1_offset_ordered_cdf_loss": offset_ordered_cdf_loss.detach(),
        "geotr_m1_offset_hierarchical_ordinal_enabled": objective.detach().new_tensor(
            float(hierarchical_ordinal_loss)
        ),
        "geotr_m1_offset_hierarchical_joint_proper_score_enabled": objective.detach().new_tensor(
            float(hierarchical_joint_proper_score)
        ),
        "geotr_m1_offset_sign_loss": offset_sign_loss.detach(),
        "geotr_m1_offset_magnitude_nll_loss": offset_magnitude_nll_loss.detach(),
        "geotr_m1_offset_magnitude_cdf_loss": offset_magnitude_cdf_loss.detach(),
        "geotr_m1_offset_sign_class_balanced_legacy": objective.detach().new_tensor(
            float(sign_class_balanced_legacy)
        ),
        "geotr_m1_offset_pred_negative_measure_mean": offset_pred_negative_measure_mean.detach(),
        "geotr_m1_offset_pred_keep_measure_mean": offset_pred_keep_measure_mean.detach(),
        "geotr_m1_offset_pred_positive_measure_mean": offset_pred_positive_measure_mean.detach(),
        "geotr_m1_offset_target_negative_measure_mean": offset_target_negative_measure_mean.detach(),
        "geotr_m1_offset_target_keep_measure_mean": offset_target_keep_measure_mean.detach(),
        "geotr_m1_offset_target_positive_measure_mean": offset_target_positive_measure_mean.detach(),
        "geotr_m1_offset_sign_calibration_l1": offset_sign_calibration_l1.detach(),
        "geotr_m1_case_balanced_owners": objective.detach().new_tensor(
            float(case_balanced_owners)
        ),
        "geotr_m1_offset_ordered_cdf_enabled": objective.detach().new_tensor(
            float(ordered_cdf_loss)
        ),
        "geotr_m1_offset_reachable_match_only": objective.detach().new_tensor(
            float(reachable_match_only)
        ),
        "geotr_m1_censored_endpoint_action": objective.detach().new_tensor(
            float(censored_endpoint_action)
        ),
        "geotr_m1_offset_match_owner_fraction": (
            distribution_owner.float().sum()
            / valid_owner.float().sum().clamp_min(1.0)
        ).detach(),
        "geotr_m1_offset_entropy_owner_mean": offset_entropy_owner_mean.detach(),
        "geotr_m1_offset_keep_owner_mean": offset_keep_owner_mean.detach(),
        "geotr_m1_offset_mode_mae_px": offset_mode_mae_px.detach(),
        "geotr_m1_offset_top1_within1px": offset_top1_within1px.detach(),
        "geotr_m1_offset_top1_within2px": offset_top1_within2px.detach(),
        "geotr_m1_offset_peak_prob_mean": offset_peak_prob_mean.detach(),
        "geotr_m1_offset_peak_margin_mean": offset_peak_margin_mean.detach(),
        "geotr_m1_offset_target_mass_within1px": offset_target_mass_within1px.detach(),
        "geotr_m1_offset_opposite_sign_mass": offset_opposite_sign_mass.detach(),
        "geotr_m1_offset_entropy_high_fraction": offset_entropy_high_fraction.detach(),
        "geotr_m1_offset_target_near_fraction": offset_target_near_fraction.detach(),
        "geotr_m1_offset_target_mid_fraction": offset_target_mid_fraction.detach(),
        "geotr_m1_offset_target_far_fraction": offset_target_far_fraction.detach(),
        "geotr_m1_offset_target_edge_fraction": offset_target_edge_fraction.detach(),
        "geotr_m1_offset_normal_alignment_mean": offset_normal_alignment_mean.detach(),
        "geotr_m1_hrcv_enabled": _mean_aux(aux, "geotr_m1_hrcv_enabled", objective),
        "geotr_m1_hrcv_candidate_conditioned": _mean_aux(
            aux, "geotr_m1_hrcv_candidate_conditioned", objective
        ),
        "geotr_m1_hrcv_delta_px": _mean_aux(aux, "geotr_m1_hrcv_delta_px", objective),
        "geotr_m1_offset_entropy_correct_sign": offset_entropy_correct_sign.detach(),
        "geotr_m1_offset_entropy_wrong_sign": offset_entropy_wrong_sign.detach(),
        "geotr_m1_offset_global_expectation_abs_px": offset_global_expectation_abs_px.detach(),
        "geotr_m1_offset_dominant_mode_abs_px": offset_dominant_mode_abs_px.detach(),
        "geotr_m1_offset_dominant_center_mae_px": offset_dominant_center_mae_px.detach(),
        "geotr_m1_offset_dominant_mass_mean": offset_dominant_mass_mean.detach(),
        "geotr_m1_offset_cancellation_ratio": offset_cancellation_ratio.detach(),
        "geotr_m1_offset_dominant_sign_accuracy": offset_dominant_sign_accuracy.detach(),
        "geotr_m1_mrm_registered": _mean_aux(aux, "geotr_m1_mrm_registered", objective),
        "geotr_m1_mrm_relational_cost": _mean_aux(aux, "geotr_m1_mrm_relational_cost", objective),
        "geotr_m1_mrm_ordered_aggregation": _mean_aux(aux, "geotr_m1_mrm_ordered_aggregation", objective),
        "geotr_m1_mrm_dominant_mode": _mean_aux(aux, "geotr_m1_mrm_dominant_mode", objective),
        "geotr_m1_mrm_structured_loss": objective.detach().new_tensor(float(mrm_structured_loss)),
        "geotr_m1_hierarchical_confidence_decoder": _mean_aux(
            aux, "geotr_m1_hierarchical_confidence_decoder", objective
        ),
        "geotr_m1_radius_balanced_keep_prior": _mean_aux(
            aux, "geotr_m1_radius_balanced_keep_prior", objective
        ),
        "geotr_m1_hierarchical_sign_confidence_mean": (
            _masked_mean(
                aux["geotr_m1_hierarchical_sign_confidence"][:, 0].detach(),
                valid_owner,
            )
            if isinstance(aux.get("geotr_m1_hierarchical_sign_confidence"), torch.Tensor)
            else objective.detach().new_zeros(())
        ),
        "geotr_m1_hierarchical_conditional_offset_abs_px": (
            _masked_mean(
                aux["geotr_m1_hierarchical_conditional_offset_px"][:, 0].detach().abs(),
                valid_owner,
            )
            if isinstance(aux.get("geotr_m1_hierarchical_conditional_offset_px"), torch.Tensor)
            else objective.detach().new_zeros(())
        ),
        "geotr_m1_operator_aligned_candidates": _mean_aux(
            aux, "geotr_m1_operator_aligned_candidates", objective
        ),
        "geotr_m1_geometry_regularization": geometry_loss.detach(),
        "geotr_m1_boundary_tv_loss": tangent_error_loss.detach(),
        "geotr_m1_fold_loss": fold_error_loss.detach(),
        "geotr_m1_tangent_error_loss": tangent_error_loss.detach(),
        "geotr_m1_error_fold_loss": fold_error_loss.detach(),
        "geotr_m1_base_error_fraction": base_error.float().mean().detach(),
        "geotr_m1_correctable_fraction": valid_owner.float().mean().detach(),
        "geotr_m1_beneficial_fraction": corrected.float().mean().detach(),
        "geotr_m1_base_boundary_fraction": owner.float().mean().detach(),
        "geotr_m1_contour_owner_fraction": owner.float().mean().detach(),
        "geotr_m1_warp_support_fraction": support.float().mean().detach(),
        "geotr_m1_weighted_band_fraction": band_weight.mean().detach(),
        "geotr_m1_geometry_case_fraction": geometry_case.float().mean().detach(),
        "geotr_m1_gt_empty_case_fraction": gt_empty_case.float().mean().detach(),
        "geotr_m1_boundary_reachable_fraction": reachable_fraction.detach(),
        "geotr_m1_reachable_support_fraction": (reachable_count / weighted_band_count).detach(),
        "geotr_m1_target_clipped_fraction": clipped_fraction.detach(),
        "geotr_m1_target_sample_offset_abs_px": _masked_mean(target_abs, valid_owner).detach(),
        "geotr_m1_predicted_sample_offset_abs_px": _masked_mean(pred_abs, valid_owner).detach(),
        "geotr_m1_predicted_magnitude_abs_px": _masked_mean(magnitude_px.detach(), valid_owner).detach(),
        "geotr_m1_sample_offset_mae_px": _masked_mean(offset_error, valid_owner).detach(),
        "geotr_m1_displacement_amplitude_ratio": amplitude_ratio.detach(),
        "geotr_m1_direction_confidence_abs_owner_mean": direction_conf_owner_mean.detach(),
        "geotr_m1_magnitude_target_ratio": magnitude_target_ratio.detach(),
        "geotr_m1_preagreement_amplitude_ratio": preagreement_amplitude_ratio.detach(),
        "geotr_m1_postagreement_amplitude_ratio": postagreement_amplitude_ratio.detach(),
        "geotr_m1_coherence_suppression_ratio": coherence_suppression_ratio.detach(),
        "geotr_m1_ray_agreement_owner_mean": ray_agreement_owner_mean.detach(),
        "geotr_m1_ray_agreement_nonzero_owner_mean": ray_agreement_nonzero_owner_mean.detach(),
        "geotr_m1_ray_agreement_reachable_mean": ray_agreement_reachable_mean.detach(),
        "geotr_m1_displacement_direction_accuracy": direction_accuracy.detach(),
        "geotr_m1_direction_balanced_accuracy": balanced_dir.detach(),
        "geotr_m1_direction_mcc": direction_mcc.detach(),
        "geotr_m1_target_positive_fraction": (tpos.float().sum()/owner_count).detach(),
        "geotr_m1_target_negative_fraction": (tneg.float().sum()/owner_count).detach(),
        "geotr_m1_predicted_positive_fraction": (ppos.float().sum()/owner_count).detach(),
        "geotr_m1_predicted_negative_fraction": (pneg.float().sum()/owner_count).detach(),
        "geotr_m1_gt_correspondence_distance_px": _masked_mean(motion_abs.to(predicted_owner), valid_owner).detach(),
        "geotr_m1_gt_normal_alignment": _masked_mean(normal_alignment.to(predicted_owner), align_mask).detach(),
        "geotr_m1_local_bce_loss": local_bce.detach(),
        "geotr_m1_corrected_fraction": corrected.float().mean().detach(),
        "geotr_m1_introduced_fraction": introduced.float().mean().detach(),
        "geotr_m1_boundary_corrected_fraction": (corrected & support).float().mean().detach(),
        "geotr_m1_boundary_introduced_fraction": (introduced & support).float().mean().detach(),
        "geotr_m1_net_error_change_fraction": (introduced.float().mean()-corrected.float().mean()).detach(),
        "geotr_m1_soft_base_dice": base_dice.detach(),
        "geotr_m1_soft_final_dice": final_dice.detach(),
        "geotr_m1_soft_final_gain": (final_dice-base_dice).detach(),
        "geotr_m1_soft_gain_gt_empty": empty_gain.detach(),
        "geotr_m1_soft_gain_gt_nonempty": nonempty_gain.detach(),
        "geotr_m1_flow_rms_px": _mean_aux(aux, "geotopo_flow_rms_px", objective),
        "geotr_m1_flow_max_px": _mean_aux(aux, "geotopo_flow_max_px", objective),
        "geotr_m1_geometry_abs_change": _mean_aux(aux, "geotopo_geometry_abs_change", objective),
        "geotr_m1_range_violation_fraction": _mean_aux(aux, "geotr_m1_range_violation_fraction", objective),
        "geotr_m1_range_violation_max": _mean_aux(aux, "geotr_m1_range_violation_max", objective),
        "geotr_m1_autozero_trust_mean": _mean_aux(aux, "geotr_m1_autozero_trust_mean", objective),
        "geotr_m1_autozero_trust_std": _mean_aux(aux, "geotr_m1_autozero_trust_std", objective),
        "geotr_m1_mc_std_mean": _mean_aux(aux, "geotr_m1_mc_std_mean", objective),
        "geotr_m1_mc_disagreement_mean": _mean_aux(aux, "geotr_m1_mc_disagreement_mean", objective),
        "geotr_m1_effective_mc_std_mean": _mean_aux(aux, "geotr_m1_effective_mc_std_mean", objective),
        "geotr_m1_effective_mc_disagreement_mean": _mean_aux(aux, "geotr_m1_effective_mc_disagreement_mean", objective),
        "geotr_m1_ds_uc_fnrt": _mean_aux(aux, "geotr_m1_ds_uc_fnrt", objective),
        "geotr_m1_ds_relative_uncertainty": _mean_aux(aux, "geotr_m1_ds_relative_uncertainty", objective),
        "geotr_m1_ds_normalized_ray": _mean_aux(aux, "geotr_m1_ds_normalized_ray", objective),
        "geotr_m1_ds_ray_agreement": _mean_aux(aux, "geotr_m1_ds_ray_agreement", objective),
        "geotr_m1_ds_ray_agreement_apply": _mean_aux(aux, "geotr_m1_ds_ray_agreement_apply", objective),
        "geotr_m1_ds_relative_mc_std_abs_mean": _mean_aux(aux, "geotr_m1_ds_relative_mc_std_abs_mean", objective),
        "geotr_m1_ds_relative_mc_disagreement_abs_mean": _mean_aux(aux, "geotr_m1_ds_relative_mc_disagreement_abs_mean", objective),
        "geotr_m1_ds_ray_agreement_mean": _mean_aux(aux, "geotr_m1_ds_ray_agreement_mean", objective),
        "geotr_m1_ds_ray_agreement_low_fraction": _mean_aux(aux, "geotr_m1_ds_ray_agreement_low_fraction", objective),
        "geotr_m1_ablation_no_posterior": objective.detach().new_tensor(float(uc_ablation == "no_posterior_uncertainty")),
        "geotr_m1_ablation_no_ray": objective.detach().new_tensor(float(uc_ablation == "no_normal_ray_evidence")),
        "geotr_m1_ablation_direct_signed": objective.detach().new_tensor(float(uc_ablation == "direct_signed")),
        "geotr_m1_ablation_segmentation_only": objective.detach().new_tensor(float(uc_ablation == "segmentation_only")),
        "geotr_m1_uc_fnrt": objective.detach().new_ones(()),
        "geotr_m1_posterior_stable_operator_warp": zero,
        "geotr_m1_sdf_operator_matched_warp": zero,
        "geotr_m1_boundary_normal_warp": zero,
        "geotr_m1_memory_exact": objective.detach().new_ones(()),
        "geotr_m1_autozero_objective": objective.detach().new_ones(()),
        "geotr_m1_has_m2": zero,
        "geotr_m1_grad_disp_norm": gd_norm.detach(),
        "geotr_m1_grad_seg_norm": gs_norm.detach(),
        "geotr_m1_grad_geo_norm": gg_norm.detach(),
        "geotr_m1_grad_cos_disp_seg": cos_ds.detach(),
        "geotr_m1_grad_cos_disp_geo": cos_dg.detach(),
        "geotr_m1_grad_cos_seg_geo": cos_sg.detach(),
        "geotr_m1_grad_disp_direction_norm": disp_dir_norm.detach(),
        "geotr_m1_grad_disp_magnitude_norm": disp_mag_norm.detach(),
        "geotr_m1_grad_seg_direction_norm": seg_dir_norm.detach(),
        "geotr_m1_grad_seg_magnitude_norm": seg_mag_norm.detach(),
        "geotr_m1_grad_geo_direction_norm": geo_dir_norm.detach(),
        "geotr_m1_grad_geo_magnitude_norm": geo_mag_norm.detach(),
        "geotr_m1_grad_offset_disp_norm": grad_offset_disp_norm.detach(),
        "geotr_m1_grad_offset_seg_norm": grad_offset_seg_norm.detach(),
        "geotr_m1_grad_offset_geo_norm": grad_offset_geo_norm.detach(),
        "geotr_m1_oracle_sign_dice": oracle_sign_dice.detach(),
        "geotr_m1_oracle_magnitude_dice": oracle_mag_dice.detach(),
        "geotr_m1_oracle_full_dice": full_oracle_dice.detach(),
        "geotr_m1_oracle_sign_gain": (oracle_sign_dice-base_dice).detach(),
        "geotr_m1_oracle_magnitude_gain": (oracle_mag_dice-base_dice).detach(),
        "geotr_m1_oracle_full_gain": (full_oracle_dice-base_dice).detach(),
        "geotr_m1_train_posterior_samples": _mean_aux(aux, "geotr_train_posterior_samples", objective),
    }
    return objective, diagnostics


def compute_semlt_autozero_loss(
    cfg: Any,
    candidate_logits: torch.Tensor,
    masks: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    epoch: int = 0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    m1_cfg = _cfg_get(cfg, "M1", None)
    if bool(_cfg_get(m1_cfg, "SEMLT_UC_FNRT", False)):
        return _compute_uc_fnrt_loss(cfg, masks, aux, epoch)
    if bool(_cfg_get(m1_cfg, "SEMLT_POSTERIOR_STABLE_OPERATOR_WARP", False)):
        return _compute_posterior_stable_operator_warp_loss(cfg, masks, aux, epoch)
    if bool(_cfg_get(m1_cfg, "SEMLT_SDF_OPERATOR_MATCHED_WARP", False)):
        return _compute_sdf_operator_matched_warp_loss(cfg, masks, aux, epoch)
    if bool(_cfg_get(m1_cfg, "SEMLT_BOUNDARY_NORMAL_WARP", False)):
        return _compute_boundary_normal_warp_loss(cfg, masks, aux, epoch)
    if bool(_cfg_get(m1_cfg, "SEMLT_LST_V31_ROOTFIX", False)):
        return _compute_v31_rootfix_loss(cfg, masks, aux, epoch)
    return _compute_v3_legacy_loss(cfg, candidate_logits, masks, aux, epoch)
