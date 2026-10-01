"""Standard training objectives and exact local teachers for V551.

No new exotic loss is introduced.  The implementation uses only cross entropy,
binary cross entropy, Smooth-L1, Dice, and L1 regularisation.  Ground truth is
used exclusively in this training-time file to construct exact local potential
outcomes; the V551 model itself remains inference-safe.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Sequence, Tuple

import torch
import torch.nn.functional as F

EPS = 1.0e-6


def _m1(cfg: Any, key: str, default: Any) -> Any:
    node = getattr(cfg, "M1", cfg)
    return getattr(node, key, default)


def _as_b1hw(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 3:
        return value[:, None]
    if value.ndim == 4 and value.shape[1] == 1:
        return value
    raise ValueError(f"Expected [B,H,W] or [B,1,H,W], got {tuple(value.shape)}")


def _safe_logit(probability: torch.Tensor) -> torch.Tensor:
    probability = probability.clamp(EPS, 1.0 - EPS)
    return torch.log(probability) - torch.log1p(-probability)


def _dice_many(probability: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Dice for ``probability=[B,N,H,W]`` and ``gt=[B,1,H,W]``."""
    gt_expand = gt.expand(-1, probability.shape[1], -1, -1)
    intersection = (probability * gt_expand).flatten(2).sum(dim=2)
    denominator = probability.flatten(2).sum(dim=2) + gt_expand.flatten(2).sum(dim=2)
    return (2.0 * intersection + EPS) / (denominator + EPS)


def _hard_dice_many(probability: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return _dice_many((probability >= 0.5).to(gt.dtype), gt)


def _hard_dice_route_bank(
    probability: torch.Tensor,
    gt: torch.Tensor,
) -> torch.Tensor:
    """Hard Dice for ``[B,N,R,H,W]`` route candidates."""
    if probability.ndim != 5:
        raise ValueError(f"Expected [B,N,R,H,W], got {tuple(probability.shape)}")
    b, n, r, h, w = probability.shape
    flat = probability.reshape(b, n * r, h, w)
    return _hard_dice_many(flat, gt).reshape(b, n, r)


def _balanced_index_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Equal class-mean CE over classes that are present in the tensor."""
    per_item = F.cross_entropy(
        logits.reshape(-1, num_classes),
        target.reshape(-1),
        reduction="none",
    ).reshape_as(target)
    pieces = []
    for cls in range(num_classes):
        mask = valid & (target == cls)
        if bool(mask.any().item()):
            pieces.append(per_item[mask].mean())
    return torch.stack(pieces).mean() if pieces else logits.sum() * 0.0


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not bool(mask.any().item()):
        return value.sum() * 0.0
    return value[mask].mean()


def _soft_dice_loss(
    probability: torch.Tensor,
    target: torch.Tensor,
    region: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    probability = probability * region
    target = target * region
    intersection = (probability * target).flatten(2).sum(dim=2)
    denominator = probability.flatten(2).sum(dim=2) + target.flatten(2).sum(dim=2)
    loss = 1.0 - (2.0 * intersection + EPS) / (denominator + EPS)
    return _masked_mean(loss, valid)


def _area_scale_target(
    area_fraction: torch.Tensor,
    thresholds: Sequence[float],
    num_scales: int,
) -> torch.Tensor:
    boundaries = torch.as_tensor(
        list(thresholds), device=area_fraction.device, dtype=area_fraction.dtype
    )
    if boundaries.numel() != max(int(num_scales) - 1, 0):
        # Log-spaced defaults cover tiny boundary fragments through large
        # under-segmented regions without dataset-specific hand tuning.
        if num_scales <= 1:
            boundaries = area_fraction.new_empty((0,))
        else:
            boundaries = torch.logspace(
                math.log10(5.0e-4),
                math.log10(3.0e-2),
                steps=num_scales - 1,
                device=area_fraction.device,
                dtype=area_fraction.dtype,
            )
    return torch.bucketize(area_fraction.detach(), boundaries).clamp(0, num_scales - 1)


def _curriculum(epoch: int, start: int, ramp: int) -> float:
    start = max(int(start), 0)
    ramp = max(int(ramp), 1)
    if int(epoch) < start:
        return 0.0
    if ramp <= 1:
        return 1.0
    return min(1.0, max(0.0, float(int(epoch) - start + 1) / float(ramp)))


@torch.no_grad()
def _greedy_nonoverlap_oracle(
    *,
    base: torch.Tensor,
    candidates: torch.Tensor,
    masks: torch.Tensor,
    valid: torch.Tensor,
    gt: torch.Tensor,
    max_steps: int,
    max_overlap: float,
) -> torch.Tensor:
    """Vectorized exact sequential oracle with only a short step loop.

    The former implementation iterated over every case and atom and called
    ``.item()`` repeatedly, forcing CUDA synchronization in every training
    batch.  This version evaluates all remaining atom interventions in parallel
    and loops only over the configured number of composition steps.
    """
    b, n, h, w = candidates.shape
    current = base[:, :1].clone()
    base_dice = _hard_dice_many(current[:, 0][:, None], gt)[:, 0]
    current_dice = base_dice.clone()
    occupied = torch.zeros((b, h, w), device=base.device, dtype=torch.bool)
    remaining = valid.clone()
    batch = torch.arange(b, device=base.device)
    for _ in range(min(max(int(max_steps), 1), n)):
        mask_bool = masks.bool()
        area = mask_bool.flatten(2).sum(dim=2)
        overlap = (mask_bool & occupied[:, None]).flatten(2).sum(dim=2).to(base.dtype) / area.clamp_min(1).to(base.dtype)
        admissible = remaining & (area > 0) & (overlap <= float(max_overlap))
        proposal = torch.where(mask_bool, candidates, current.expand(-1, n, -1, -1))
        proposal_dice = _hard_dice_many(proposal, gt)
        gain = (proposal_dice - current_dice[:, None]).masked_fill(~admissible, -1.0e4)
        best_gain, best_index = gain.max(dim=1)
        accept = best_gain > 0.0
        selected_candidate = candidates[batch, best_index][:, None]
        selected_mask = mask_bool[batch, best_index] & accept[:, None, None]
        current = torch.where(selected_mask[:, None], selected_candidate, current)
        current_dice = torch.where(accept, proposal_dice[batch, best_index], current_dice)
        occupied |= selected_mask
        remaining[batch, best_index] &= ~accept
    return (current_dice - base_dice).mean()


def _compute_exact_composer_teacher(
    *,
    gt: torch.Tensor,
    editor_logit_delta: torch.Tensor,
    step_logits: torch.Tensor,
    step_candidate_scores: torch.Tensor,
    step_state_probs: torch.Tensor,
    step_eligible: torch.Tensor,
    step_active: torch.Tensor,
    stop_margin: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Exact on-policy supervision for the M2 sequential set composer.

    Each predicted composition state is treated as the current intervention.
    All still-eligible edited atoms are then applied in parallel and evaluated
    against GT.  The target is the best positive marginal atom, or the explicit
    Stop token when no atom improves the current composition.
    """
    b, steps, classes = step_logits.shape
    n = classes - 1
    zero = step_logits.sum() * 0.0
    ce_terms = []
    regression_terms = []
    correct = zero.new_zeros(())
    active_count = zero.new_zeros(())
    stop_correct = zero.new_zeros(())
    stop_count = zero.new_zeros(())
    teacher_steps = zero.new_zeros(())
    marginal_abs_error = zero.new_zeros(())
    marginal_count = zero.new_zeros(())
    gt_b1 = gt[:, :1]
    for step in range(steps):
        active = step_active[:, step].bool()
        eligible = step_eligible[:, step].bool() & active[:, None]
        state = step_state_probs[:, step].detach().clamp(EPS, 1.0 - EPS)
        state_dice = _hard_dice_many(state[:, 0][:, None], gt_b1)[:, 0]
        proposal = torch.sigmoid(
            _safe_logit(state)[:, 0][:, None] + editor_logit_delta.detach()
        ).clamp(EPS, 1.0 - EPS)
        marginal = _hard_dice_many(proposal, gt_b1) - state_dice[:, None]
        masked = marginal.masked_fill(~eligible, -1.0e4)
        best_gain, best_index = masked.max(dim=1)
        choose_stop = (~eligible.any(dim=1)) | (best_gain <= float(stop_margin))
        target = torch.where(
            choose_stop,
            best_index.new_full(best_index.shape, n),
            best_index,
        )
        if bool(active.any().item()):
            ce_terms.append(F.cross_entropy(step_logits[active, step], target[active]))
            prediction = step_logits[:, step].detach().argmax(dim=1)
            correct = correct + ((prediction == target) & active).float().sum()
            active_count = active_count + active.float().sum()
            stop_mask = active & choose_stop
            stop_correct = stop_correct + ((prediction == n) & stop_mask).float().sum()
            stop_count = stop_count + stop_mask.float().sum()
            teacher_steps = teacher_steps + ((~choose_stop) & active).float().sum()
        if bool(eligible.any().item()):
            pred = step_candidate_scores[:, step]
            regression_terms.append(
                F.smooth_l1_loss(pred[eligible], marginal[eligible])
            )
            marginal_abs_error = marginal_abs_error + (
                pred.detach()[eligible] - marginal[eligible]
            ).abs().sum()
            marginal_count = marginal_count + eligible.float().sum()
    selection_loss = torch.stack(ce_terms).mean() if ce_terms else zero
    marginal_loss = torch.stack(regression_terms).mean() if regression_terms else zero
    diagnostics = {
        "v552_composer_selection_loss": selection_loss.detach(),
        "v552_composer_marginal_gain_loss": marginal_loss.detach(),
        "v552_composer_selection_accuracy": correct / active_count.clamp_min(1.0),
        "v552_composer_stop_accuracy": stop_correct / stop_count.clamp_min(1.0),
        "v552_composer_teacher_step_count": teacher_steps / float(max(b, 1)),
        "v552_composer_marginal_gain_mae": marginal_abs_error / marginal_count.clamp_min(1.0),
    }
    return selection_loss, marginal_loss, diagnostics



@torch.no_grad()
def build_v552r2_exact_composer_teacher(
    *,
    base: torch.Tensor,
    editor_logit_delta: torch.Tensor,
    masks: torch.Tensor,
    valid: torch.Tensor,
    gt: torch.Tensor,
    max_steps: int,
    stop_margin: float,
    max_overlap: float = 0.20,
    max_total_edit_fraction: float = 0.35,
) -> Dict[str, torch.Tensor]:
    """Build an exact GT rollout independent of student gates and Stop choices.

    This function is deliberately pure training-time code.  It starts from Base,
    evaluates every remaining physically-valid atom against GT at each step,
    selects the best positive marginal atom, and only then advances its own
    teacher state.  No Benefit/Harm/Gain threshold, persistent quality latch, or
    student-selected action can remove supervision from the rollout.
    """
    base = _as_b1hw(base).detach().clamp(EPS, 1.0 - EPS)
    gt = _as_b1hw(gt).detach().to(base.dtype)
    delta = editor_logit_delta.detach()
    masks_bool = masks.detach().bool()
    valid = valid.detach().bool()
    b, n, h, w = delta.shape
    steps = min(max(int(max_steps), 1), n)
    batch = torch.arange(b, device=base.device)
    stop_index = n
    max_pixels = max(int(round(float(max_total_edit_fraction) * h * w)), 1)

    state_logit = _safe_logit(base)[:, 0]
    occupied = torch.zeros((b, h, w), dtype=torch.bool, device=base.device)
    remaining = valid.clone()
    teacher_alive = torch.ones((b,), dtype=torch.bool, device=base.device)

    targets = []
    marginals = []
    eligibles = []
    actives = []
    step_gains = []
    state_probs = []
    for _ in range(steps):
        state = torch.sigmoid(state_logit)[:, None].clamp(EPS, 1.0 - EPS)
        state_probs.append(state)
        current_dice = _hard_dice_many(state[:, 0][:, None], gt)[:, 0]

        area = masks_bool.flatten(2).sum(dim=2)
        overlap_pixels = (masks_bool & occupied[:, None]).flatten(2).sum(dim=2)
        overlap = overlap_pixels.to(base.dtype) / area.clamp_min(1).to(base.dtype)
        new_mask = masks_bool & (~occupied[:, None])
        new_area = new_mask.flatten(2).sum(dim=2)
        current_area = occupied.flatten(1).sum(dim=1)
        eligible = (
            remaining
            & teacher_alive[:, None]
            & (area > 0)
            & (new_area > 0)
            & (overlap <= float(max_overlap))
            & ((current_area[:, None] + new_area) <= max_pixels)
        )
        proposal_logit = state_logit[:, None] + delta * new_mask.to(delta.dtype)
        proposal = torch.sigmoid(proposal_logit).clamp(EPS, 1.0 - EPS)
        marginal = _hard_dice_many(proposal, gt) - current_dice[:, None]
        masked = marginal.masked_fill(~eligible, -1.0e4)
        best_gain, best_index = masked.max(dim=1)
        should_stop = (~eligible.any(dim=1)) | (best_gain <= float(stop_margin))
        target = torch.where(
            should_stop,
            best_index.new_full(best_index.shape, stop_index),
            best_index,
        )

        targets.append(target)
        marginals.append(marginal)
        eligibles.append(eligible)
        actives.append(teacher_alive.clone())
        step_gains.append(torch.where(should_stop, torch.zeros_like(best_gain), best_gain))

        accept = teacher_alive & (~should_stop)
        chosen_new_mask = new_mask[batch, best_index] & accept[:, None, None]
        chosen_delta = delta[batch, best_index]
        state_logit = state_logit + chosen_delta * chosen_new_mask.to(delta.dtype)
        occupied |= chosen_new_mask
        remaining[batch, best_index] &= ~accept
        teacher_alive = teacher_alive & (~should_stop)

    final_probability = torch.sigmoid(state_logit)[:, None].clamp(EPS, 1.0 - EPS)
    return {
        "target_indices": torch.stack(targets, dim=1),
        "target_marginal_gains": torch.stack(marginals, dim=1),
        "target_eligible": torch.stack(eligibles, dim=1),
        "target_active": torch.stack(actives, dim=1),
        "target_step_gains": torch.stack(step_gains, dim=1),
        "teacher_state_probs": torch.stack(state_probs, dim=1),
        "teacher_final_probs": final_probability,
    }


def v552r2_editor_modify_target(
    m1_gain: torch.Tensor,
    editor_gain: torch.Tensor,
    valid: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Public contract helper: edit only when Editor beats the original M1."""
    return valid.bool() & ((editor_gain - m1_gain) > float(margin))


def _balanced_outcome_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    per_item = F.cross_entropy(
        logits.reshape(-1, 3), target.reshape(-1), reduction="none"
    ).reshape_as(target)
    nominal = (0.25, 0.375, 0.375)
    pieces = []
    weights = []
    for cls, weight in enumerate(nominal):
        mask = valid & (target == cls)
        if bool(mask.any().item()):
            pieces.append(per_item[mask].mean())
            weights.append(weight)
    if not pieces:
        return logits.sum() * 0.0
    normalizer = max(sum(weights), EPS)
    return sum(piece * (weight / normalizer) for piece, weight in zip(pieces, weights))


def _equal_class_outcome_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Dynamic equal-class Cross Entropy with no fixed Benefit/Harm prior.

    Every class that is actually present contributes one class mean.  Therefore
    the per-example weight adapts automatically to the current batch counts and
    Benefit/Harm always receive exactly the same aggregate gradient budget.
    """
    per_item = F.cross_entropy(
        logits.reshape(-1, 3), target.reshape(-1), reduction="none"
    ).reshape_as(target)
    return _equal_class_mean(per_item, target, valid, num_classes=3)


def _equal_class_mean(
    per_item: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    num_classes: int,
) -> torch.Tensor:
    pieces = []
    for cls in range(int(num_classes)):
        mask = valid & (target == cls)
        if bool(mask.any().item()):
            pieces.append(per_item[mask].mean())
    return torch.stack(pieces).mean() if pieces else per_item.sum() * 0.0


def _balanced_binary_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    per_item = F.binary_cross_entropy_with_logits(
        logits, target.to(logits.dtype), reduction="none"
    )
    return _equal_class_mean(
        per_item,
        target.long(),
        valid,
        num_classes=2,
    )


def _active_mean(*losses: Tuple[torch.Tensor, bool]) -> torch.Tensor:
    active = [loss for loss, enabled in losses if bool(enabled)]
    if not active:
        reference = losses[0][0]
        return reference.sum() * 0.0
    return torch.stack(active).mean()


def _quota_replay_mask(
    *,
    queue_target: torch.Tensor,
    current_target: torch.Tensor,
    current_valid: torch.Tensor,
    minimum_per_class: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select only the class deficit from a detached replay queue."""
    replay_mask = torch.zeros_like(queue_target, dtype=torch.bool)
    replay_counts = torch.zeros(
        (3,), dtype=torch.long, device=queue_target.device
    )
    minimum = max(int(minimum_per_class), 0)
    if minimum <= 0 or queue_target.numel() == 0:
        return replay_mask, replay_counts
    for cls in range(3):
        current_count = int(
            (current_valid & (current_target == cls)).sum().detach().item()
        )
        needed = max(minimum - current_count, 0)
        if needed <= 0:
            continue
        indices = torch.nonzero(queue_target == cls, as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        selected = indices[-min(needed, int(indices.numel())) :]
        replay_mask[selected] = True
        replay_counts[cls] = int(selected.numel())
    return replay_mask, replay_counts


def _r42_critic_terms(
    *,
    logits: torch.Tensor,
    benefit_magnitude: torch.Tensor,
    harm_magnitude: torch.Tensor,
    exact_gain: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    gain_unit: float,
) -> Dict[str, torch.Tensor]:
    """Independent proper-classification and conditional-magnitude losses.

    The Outcome logits receive gradients only from equal-class Cross Entropy.
    Conditional magnitudes receive Smooth-L1 gradients.  Detached Outcome
    probabilities are used to calibrate the expected gain and ordering without
    reopening the R4.1 shortcut from gain error into the class boundary.
    """
    zero = logits.sum() * 0.0
    unit = max(float(gain_unit), 1.0e-5)
    ce = _equal_class_outcome_ce(logits, target, valid)
    probabilities = F.softmax(logits, dim=-1)
    detached_probabilities = probabilities.detach()

    benefit_mask = valid & (target == 1)
    harm_mask = valid & (target == 2)
    neutral_mask = valid & (target == 0)
    positive_target = exact_gain.detach().clamp_min(0.0)
    negative_target = (-exact_gain.detach()).clamp_min(0.0)

    benefit_magnitude_loss = _masked_mean(
        F.smooth_l1_loss(
            benefit_magnitude / unit,
            (positive_target / unit).clamp(0.0, 1.0),
            beta=0.10,
            reduction="none",
        ),
        benefit_mask,
    )
    harm_magnitude_loss = _masked_mean(
        F.smooth_l1_loss(
            harm_magnitude / unit,
            (negative_target / unit).clamp(0.0, 1.0),
            beta=0.10,
            reduction="none",
        ),
        harm_mask,
    )
    magnitude_loss = _active_mean(
        (benefit_magnitude_loss, bool(benefit_mask.any().item())),
        (harm_magnitude_loss, bool(harm_mask.any().item())),
    )

    isolated_expected_gain = (
        detached_probabilities[..., 1] * benefit_magnitude
        - detached_probabilities[..., 2] * harm_magnitude
    )
    expected_per_item = F.smooth_l1_loss(
        isolated_expected_gain / unit,
        (exact_gain.detach() / unit).clamp(-1.0, 1.0),
        beta=0.10,
        reduction="none",
    )
    expected_gain_loss = _equal_class_mean(
        expected_per_item,
        target,
        valid,
        num_classes=3,
    )

    return {
        "ce": ce,
        "magnitude_loss": magnitude_loss,
        "benefit_magnitude_loss": benefit_magnitude_loss,
        "harm_magnitude_loss": harm_magnitude_loss,
        "expected_gain_loss": expected_gain_loss,
        "isolated_expected_gain": isolated_expected_gain,
        "probabilities": probabilities,
        "benefit_mask": benefit_mask,
        "harm_mask": harm_mask,
        "neutral_mask": neutral_mask,
        "zero": zero,
    }


def _v552r2_student_soft_final(
    *,
    base: torch.Tensor,
    step_logits: torch.Tensor,
    editor_logit_delta: torch.Tensor,
) -> torch.Tensor:
    """Differentiable low-weight final-composition auxiliary target."""
    b, steps, classes = step_logits.shape
    n = classes - 1
    if editor_logit_delta.shape[1] != n:
        raise ValueError("Composer class count and candidate delta count disagree")
    current_logit = _safe_logit(base)[:, 0]
    remaining_soft = torch.ones((b, n), device=base.device, dtype=base.dtype)
    for step in range(steps):
        probabilities = F.softmax(step_logits[:, step], dim=1)
        candidate_weight = probabilities[:, :n] * remaining_soft
        current_logit = current_logit + (
            candidate_weight[:, :, None, None] * editor_logit_delta
        ).sum(dim=1)
        remaining_soft = remaining_soft * (1.0 - candidate_weight).clamp(0.0, 1.0)
    return torch.sigmoid(current_logit)[:, None].clamp(EPS, 1.0 - EPS)



def _compute_v552r4_loss(
    *,
    cfg: Any,
    masks: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    epoch: int,
    base_probability: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """V552-R4 unified-reference training contract.

    Five exact routes (Preserve + four typed actions) are supervised for every
    physical atom.  The Editor Safety Critic predicts route value relative to
    M1, while the Candidate Utility Critic predicts value relative to Base.
    Composer supervision consumes the exact M1-safe handoff.
    """
    base = _as_b1hw(base_probability).detach().clamp(EPS, 1.0 - EPS)
    gt = (_as_b1hw(masks) >= 0.5).to(base.dtype)
    zero = base.sum() * 0.0
    r42_enabled = bool(_m1(cfg, "V552R42_DECOUPLED_CRITIC_ENABLED", False))
    r43_enabled = bool(_m1(cfg, "V552R43_ROOTFIX_ENABLED", False))
    r44_enabled = bool(_m1(cfg, "V552R44_AUDIT_GATE_ROOTFIX_ENABLED", False))
    r45_enabled = bool(_m1(cfg, "V552R45_ROOTFIX_ENABLED", False))
    r46_enabled = bool(_m1(cfg, "V552R46_ROOTFIX_ENABLED", False))
    r4212_requested = bool(_m1(cfg, "V552R4212_ROOTFIX_ENABLED", False))
    r4212_existence_no_object = r4212_requested and bool(
        _m1(cfg, "V552R4212_EXISTENCE_NO_OBJECT_ENABLED", False)
    )
    r4212_direct_delta_utility = r4212_requested and bool(
        _m1(cfg, "V552R4212_DIRECT_DELTA_UTILITY_ENABLED", False)
    )
    r4212_zero_stop_one_step = r4212_requested and bool(
        _m1(cfg, "V552R4212_ZERO_STOP_ONE_STEP_ENABLED", False)
    )
    v560_clean_core = bool(_m1(cfg, "V560_CLEAN_CORE_ENABLED", False))
    required = [
        "v551_scale_logits",
        "v551_editor_route_logits",
        "v551_editor_region",
        "v551_local_residual",
        "v545_slot_hard_masks",
        "v552_m1_supervision_valid",
        "v538_slot_presence_logits",
        "v552_atom_quality_logits",
        "v552r4_route_candidate_bank",
        "v552r4_route_delta_bank",
        "v552r4_route_valid",
        "v552r4_editor_safety_logits_bank",
        "v552r4_editor_incremental_gain_bank",
        "v552r4_candidate_utility_logits_bank",
        "v552r4_candidate_absolute_gain_bank",
        "v552r4_teacher_safe_route_index",
        "v552r4_teacher_safe_delta",
        "v552r4_teacher_safe_candidate",
        "v552r4_student_safe_route_index",
        "v552r4_student_safe_candidate",
        "v552r4_predicted_editor_safe",
        "v552_composer_step_logits",
        "v552_composer_step_candidate_scores",
        "v552r41_teacher_target_indices",
        "v552r41_teacher_target_marginal_gains",
        "v552r41_teacher_target_eligible",
        "v552r41_teacher_target_active",
        "v552r41_teacher_target_step_gains",
        "v552r41_teacher_state_probs",
        "v552r41_teacher_final_probs",
    ]
    if r45_enabled:
        required.extend([
            "v552r45_error_prone_logits",
            "v552r45_editor_safety_benefit_logits_bank",
            "v552r45_editor_safety_harm_logits_bank",
            "v552r45_candidate_direct_signed_gain_bank",
        ])
    if r42_enabled:
        required.extend([
            "v552r42_editor_benefit_magnitude_bank",
            "v552r42_editor_harm_magnitude_bank",
            "v552r42_candidate_benefit_magnitude_bank",
            "v552r42_candidate_harm_magnitude_bank",
        ])
    missing = [key for key in required if key not in aux]
    if missing:
        raise KeyError(f"V552-R4 missing model outputs: {missing}")

    atom_masks = aux["v545_slot_hard_masks"].detach().clamp(0.0, 1.0)
    valid = aux["v552_m1_supervision_valid"].bool()
    route_valid = aux["v552r4_route_valid"].bool()
    route_candidates = aux["v552r4_route_candidate_bank"].clamp(EPS, 1.0 - EPS)
    route_delta_bank = aux["v552r4_route_delta_bank"]
    route_logits = aux["v551_editor_route_logits"]
    safety_logits = aux["v552r4_editor_safety_logits_bank"]
    safety_gain_pred = aux["v552r4_editor_incremental_gain_bank"]
    utility_logits = aux["v552r4_candidate_utility_logits_bank"]
    utility_gain_pred = aux["v552r4_candidate_absolute_gain_bank"]
    safety_benefit_binary_logits = aux.get(
        "v552r45_editor_safety_benefit_logits_bank"
    )
    safety_harm_binary_logits = aux.get(
        "v552r45_editor_safety_harm_logits_bank"
    )
    utility_direct_signed_gain = aux.get(
        "v552r45_candidate_direct_signed_gain_bank", utility_gain_pred
    )
    safety_benefit_magnitude = aux.get(
        "v552r42_editor_benefit_magnitude_bank", safety_gain_pred.abs()
    )
    safety_harm_magnitude = aux.get(
        "v552r42_editor_harm_magnitude_bank", safety_gain_pred.abs()
    )
    utility_benefit_magnitude = aux.get(
        "v552r42_candidate_benefit_magnitude_bank", utility_gain_pred.abs()
    )
    utility_harm_magnitude = aux.get(
        "v552r42_candidate_harm_magnitude_bank", utility_gain_pred.abs()
    )
    teacher_safe_index_model = aux["v552r4_teacher_safe_route_index"].long()
    teacher_safe_delta = aux["v552r4_teacher_safe_delta"]
    teacher_safe_candidate = aux["v552r4_teacher_safe_candidate"].clamp(EPS, 1.0 - EPS)
    student_safe_index = aux["v552r4_student_safe_route_index"].long()
    student_safe_candidate = aux["v552r4_student_safe_candidate"].clamp(EPS, 1.0 - EPS)
    predicted_editor_safe = aux["v552r4_predicted_editor_safe"].bool()
    editor_region = aux["v551_editor_region"].clamp(0.0, 1.0)
    local_residual = aux["v551_local_residual"]
    b, n, routes, h, w = route_candidates.shape
    if routes != 5:
        raise ValueError(f"V552-R4 requires exactly 5 routes, got {routes}")

    base_dice = _hard_dice_many(base[:, 0][:, None], gt)[:, 0]

    # R4.5 Error-Prone Region supervision.  This is a standard BCE + Tversky
    # objective on the Base error map.  The Tversky beta defaults to 0.7 so the
    # current high-precision/low-recall failure is corrected by explicitly
    # penalising false negatives more than false positives.
    error_region_bce = zero
    error_region_tversky = zero
    error_region_precision = zero
    error_region_recall = zero
    if r45_enabled:
        error_logits = aux["v552r45_error_prone_logits"]
        if error_logits.shape[-2:] != gt.shape[-2:]:
            error_logits = F.interpolate(
                error_logits, size=gt.shape[-2:], mode="bilinear", align_corners=False
            )
        base_error_target = ((base >= 0.5) != (gt >= 0.5)).to(base.dtype)
        error_region_bce = F.binary_cross_entropy_with_logits(
            error_logits, base_error_target
        )
        error_prob = torch.sigmoid(error_logits)
        alpha = float(_m1(cfg, "V552R45_TVERSKY_ALPHA", 0.30))
        beta = float(_m1(cfg, "V552R45_TVERSKY_BETA", 0.70))
        tp = (error_prob * base_error_target).flatten(1).sum(dim=1)
        fp = (error_prob * (1.0 - base_error_target)).flatten(1).sum(dim=1)
        fn = ((1.0 - error_prob) * base_error_target).flatten(1).sum(dim=1)
        tversky = (tp + EPS) / (tp + alpha * fp + beta * fn + EPS)
        error_region_tversky = (1.0 - tversky).mean()
        hard_error = error_prob.detach() >= 0.5
        target_error = base_error_target.bool()
        error_tp = (hard_error & target_error).flatten(1).sum(dim=1).to(base.dtype)
        error_fp = (hard_error & (~target_error)).flatten(1).sum(dim=1).to(base.dtype)
        error_fn = ((~hard_error) & target_error).flatten(1).sum(dim=1).to(base.dtype)
        error_region_precision = (
            error_tp / (error_tp + error_fp).clamp_min(1.0)
        ).mean()
        error_region_recall = (
            error_tp / (error_tp + error_fn).clamp_min(1.0)
        ).mean()

    route_dice = _hard_dice_route_bank(route_candidates.detach(), gt)
    m1_dice = route_dice[:, :, 0]
    incremental_gain = route_dice - m1_dice[:, :, None]
    absolute_gain = route_dice - base_dice[:, None, None]
    m1_gain = absolute_gain[:, :, 0]

    relative_margin = max(
        float(
            _m1(
                cfg,
                "V552R43_SAFETY_CLASS_MARGIN",
                _m1(cfg, "V552_EDITOR_RELATIVE_MARGIN", 1.0e-3),
            )
            if r43_enabled
            else _m1(cfg, "V552_EDITOR_RELATIVE_MARGIN", 1.0e-3)
        ),
        0.0,
    )
    absolute_margin = max(
        float(
            _m1(
                cfg,
                "V552R43_UTILITY_CLASS_MARGIN",
                _m1(cfg, "V551_BENEFIT_EPSILON", 1.0e-4),
            )
            if r43_enabled
            else _m1(cfg, "V551_BENEFIT_EPSILON", 1.0e-4)
        ),
        0.0,
    )
    best_relative_gain, best_action_offset = incremental_gain[:, :, 1:].max(dim=2)
    route_target = torch.where(
        valid & (best_relative_gain > relative_margin),
        best_action_offset + 1,
        torch.zeros_like(best_action_offset),
    )
    teacher_contract_mismatch = (
        (teacher_safe_index_model != route_target) & valid
    ).float().sum() / valid.float().sum().clamp_min(1.0)

    route_loss = _balanced_index_ce(
        route_logits, route_target, valid, num_classes=5
    )
    preserve_logit = route_logits[:, :, 0]
    edit_logit = torch.logsumexp(route_logits[:, :, 1:], dim=2)
    should_modify = valid & (route_target > 0)
    editability_loss = _masked_mean(
        F.binary_cross_entropy_with_logits(
            edit_logit - preserve_logit,
            should_modify.to(route_logits.dtype),
            reduction="none",
        ),
        valid,
    )
    route_prediction = route_logits.detach().argmax(dim=2)
    selected_route_onehot = F.one_hot(route_prediction, num_classes=5).to(base.dtype)
    selected_incremental_gain_true = (
        selected_route_onehot * incremental_gain
    ).sum(dim=2)
    selected_absolute_gain_true = (
        selected_route_onehot * absolute_gain
    ).sum(dim=2)
    selected_raw_candidate = (
        selected_route_onehot[:, :, :, None, None] * route_candidates
    ).sum(dim=2)

    # Exact local target is the same shared-executor route selected by Teacher.
    teacher_onehot = F.one_hot(route_target, num_classes=5).to(base.dtype)
    exact_teacher_candidate = (
        teacher_onehot[:, :, :, None, None] * route_candidates.detach()
    ).sum(dim=2)
    local_bce = F.binary_cross_entropy(
        selected_raw_candidate, exact_teacher_candidate, reduction="none"
    )
    local_bce = (
        (local_bce * editor_region).flatten(2).sum(dim=2)
        / editor_region.flatten(2).sum(dim=2).clamp_min(1.0)
    )
    local_bce_loss = _masked_mean(local_bce, valid)
    local_dice_loss = _soft_dice_loss(
        selected_raw_candidate, exact_teacher_candidate, editor_region, valid
    )
    outside = 1.0 - editor_region
    outside_preserve = (
        ((selected_raw_candidate - route_candidates[:, :, 0].detach()).abs() * outside)
        .flatten(2).sum(dim=2)
        / outside.flatten(2).sum(dim=2).clamp_min(1.0)
    )
    outside_preserve_loss = _masked_mean(outside_preserve, valid)
    residual_l1 = _masked_mean(
        local_residual.abs().flatten(2).mean(dim=2), valid
    )
    m1_soft_dice = _dice_many(
        route_candidates[:, :, 0].detach(), gt
    )
    selected_soft_dice = _dice_many(selected_raw_candidate, gt)
    relative_safety_loss = _masked_mean(
        F.relu(m1_soft_dice.detach() - selected_soft_dice + relative_margin),
        valid & (route_prediction > 0),
    )

    # ------------------------------------------------------------------
    # Relative-to-M1 Editor Safety Critic and absolute-to-Base Utility
    # Critic.  R4.2 removes the R4.1 gradient shortcut: Outcome logits are
    # optimized only by a proper equal-class CE, while two conditional
    # magnitude heads learn positive and negative utility independently.
    # ------------------------------------------------------------------
    safety_target = torch.zeros_like(incremental_gain, dtype=torch.long)
    safety_target = torch.where(
        incremental_gain > relative_margin,
        torch.ones_like(safety_target),
        safety_target,
    )
    safety_target = torch.where(
        incremental_gain < -relative_margin,
        torch.full_like(safety_target, 2),
        safety_target,
    )
    utility_target = torch.zeros_like(absolute_gain, dtype=torch.long)
    utility_target = torch.where(
        absolute_gain > absolute_margin,
        torch.ones_like(utility_target),
        utility_target,
    )
    utility_target = torch.where(
        absolute_gain < -absolute_margin,
        torch.full_like(utility_target, 2),
        utility_target,
    )
    gain_unit = max(float(_m1(cfg, "V552R3_GAIN_UNIT", 0.01)), 1.0e-5)

    if r42_enabled:
        safety_terms = _r42_critic_terms(
            logits=safety_logits,
            benefit_magnitude=safety_benefit_magnitude,
            harm_magnitude=safety_harm_magnitude,
            exact_gain=incremental_gain,
            target=safety_target,
            valid=route_valid,
            gain_unit=gain_unit,
        )
        utility_terms = _r42_critic_terms(
            logits=utility_logits,
            benefit_magnitude=utility_benefit_magnitude,
            harm_magnitude=utility_harm_magnitude,
            exact_gain=absolute_gain,
            target=utility_target,
            valid=route_valid,
            gain_unit=gain_unit,
        )
        safety_ce = safety_terms["ce"]
        safety_magnitude_loss = safety_terms["magnitude_loss"]
        safety_benefit_magnitude_loss = safety_terms["benefit_magnitude_loss"]
        safety_harm_magnitude_loss = safety_terms["harm_magnitude_loss"]
        safety_gain_loss = safety_terms["expected_gain_loss"]
        safety_gain_for_loss = safety_terms["isolated_expected_gain"]
        safety_benefit = safety_terms["benefit_mask"]
        safety_harm = safety_terms["harm_mask"]
        safety_neutral = safety_terms["neutral_mask"]

        utility_ce = utility_terms["ce"]
        utility_magnitude_loss = utility_terms["magnitude_loss"]
        utility_benefit_magnitude_loss = utility_terms["benefit_magnitude_loss"]
        utility_harm_magnitude_loss = utility_terms["harm_magnitude_loss"]
        utility_gain_loss = utility_terms["expected_gain_loss"]
        utility_gain_for_loss = utility_terms["isolated_expected_gain"]
        utility_benefit = utility_terms["benefit_mask"]
        utility_harm = utility_terms["harm_mask"]
        utility_neutral = utility_terms["neutral_mask"]
    else:
        safety_ce = _balanced_outcome_ce(safety_logits, safety_target, route_valid)
        safety_gain_target_norm = (
            incremental_gain.detach() / gain_unit
        ).clamp(-1.0, 1.0)
        safety_gain_pred_norm = safety_gain_pred / gain_unit
        safety_gain_loss = _masked_mean(
            F.smooth_l1_loss(
                safety_gain_pred_norm,
                safety_gain_target_norm,
                beta=0.10,
                reduction="none",
            ),
            route_valid,
        )
        safety_magnitude_loss = safety_gain_loss
        safety_benefit_magnitude_loss = zero
        safety_harm_magnitude_loss = zero
        safety_gain_for_loss = safety_gain_pred
        safety_benefit = route_valid & (safety_target == 1)
        safety_harm = route_valid & (safety_target == 2)
        safety_neutral = route_valid & (safety_target == 0)

        utility_ce = _balanced_outcome_ce(utility_logits, utility_target, route_valid)
        utility_gain_target_norm = (
            absolute_gain.detach() / gain_unit
        ).clamp(-1.0, 1.0)
        utility_gain_pred_norm = utility_gain_pred / gain_unit
        utility_gain_loss = _masked_mean(
            F.smooth_l1_loss(
                utility_gain_pred_norm,
                utility_gain_target_norm,
                beta=0.10,
                reduction="none",
            ),
            route_valid,
        )
        utility_magnitude_loss = utility_gain_loss
        utility_benefit_magnitude_loss = zero
        utility_harm_magnitude_loss = zero
        utility_gain_for_loss = utility_gain_pred
        utility_benefit = route_valid & (utility_target == 1)
        utility_harm = route_valid & (utility_target == 2)
        utility_neutral = route_valid & (utility_target == 0)

    if r45_enabled:
        if not isinstance(safety_benefit_binary_logits, torch.Tensor) or not isinstance(
            safety_harm_binary_logits, torch.Tensor
        ):
            raise RuntimeError("R4.5 factorized Safety logits are missing")
        safety_benefit = route_valid & (safety_target == 1)
        safety_harm = route_valid & (safety_target == 2)
        safety_neutral = route_valid & (safety_target == 0)
        safety_benefit_binary_loss = _balanced_binary_bce_with_logits(
            safety_benefit_binary_logits, safety_benefit, route_valid
        )
        safety_harm_binary_loss = _balanced_binary_bce_with_logits(
            safety_harm_binary_logits, safety_harm, route_valid
        )
        # Keep the compatibility name ``safety_ce`` but in R4.5 it is the sum
        # of two independent balanced BCE objectives, not a 3-way softmax CE.
        safety_ce = safety_benefit_binary_loss + safety_harm_binary_loss
        safety_magnitude_loss = zero
        safety_benefit_magnitude_loss = zero
        safety_harm_magnitude_loss = zero
        safety_gain_loss = zero
        safety_gain_for_loss = safety_gain_pred.detach()

        # Utility classification remains available as a diagnostic.  R4.21.2
        # makes the deployed one-step utility a *physical* signed DeltaDice.
        # The regressor therefore uses a proper raw L1 target in the same unit
        # consumed by the selector: no gain_unit, no clipping, no class-balanced
        # rescaling, and no sign hinge.  This preserves magnitude ordering among
        # strong positive candidates and makes Stop=0 dimensionally meaningful.
        utility_ce = _balanced_outcome_ce(utility_logits, utility_target, route_valid)
        utility_gain_for_loss = utility_direct_signed_gain
        if r4212_direct_delta_utility:
            utility_gain_loss = _masked_mean(
                F.l1_loss(
                    utility_direct_signed_gain,
                    absolute_gain.detach(),
                    reduction="none",
                ),
                route_valid,
            )
        else:
            utility_gain_target_norm = (
                absolute_gain.detach() / gain_unit
            ).clamp(-1.0, 1.0)
            utility_gain_loss = _equal_class_mean(
                F.smooth_l1_loss(
                    utility_direct_signed_gain / gain_unit,
                    utility_gain_target_norm,
                    beta=0.10,
                    reduction="none",
                ),
                utility_target,
                route_valid,
                num_classes=3,
            )
        utility_magnitude_loss = zero
        utility_benefit_magnitude_loss = zero
        utility_harm_magnitude_loss = zero
        utility_benefit = route_valid & (utility_target == 1)
        utility_harm = route_valid & (utility_target == 2)
        utility_neutral = route_valid & (utility_target == 0)
    else:
        safety_benefit_binary_loss = zero
        safety_harm_binary_loss = zero

    # Sign metrics remain diagnostics in R4.2.  They are deliberately excluded
    # from the R4.2 objective because conditional non-negative magnitudes make
    # the sign structural rather than another asymmetric hinge objective.
    safety_gain_pred_norm = safety_gain_for_loss / gain_unit
    utility_gain_pred_norm = utility_gain_for_loss / gain_unit
    safety_benefit_sign = _masked_mean(
        F.relu(0.10 - safety_gain_pred_norm), safety_benefit
    )
    safety_harm_sign = _masked_mean(
        F.relu(0.10 + safety_gain_pred_norm), safety_harm
    )
    safety_neutral_band = _masked_mean(
        F.relu(safety_gain_pred_norm.abs() - 0.05), safety_neutral
    )
    utility_benefit_sign = _masked_mean(
        F.relu(0.10 - utility_gain_pred_norm), utility_benefit
    )
    utility_harm_sign = _masked_mean(
        F.relu(0.10 + utility_gain_pred_norm), utility_harm
    )
    utility_neutral_band = _masked_mean(
        F.relu(utility_gain_pred_norm.abs() - 0.05), utility_neutral
    )

    utility_prediction = utility_logits.detach().argmax(dim=-1)
    utility_recalls = []
    if r45_enabled:
        safety_benefit_prediction = torch.sigmoid(
            safety_benefit_binary_logits.detach()
        ) >= 0.5
        safety_harm_prediction = torch.sigmoid(
            safety_harm_binary_logits.detach()
        ) >= 0.5
        safety_benefit_recall = (
            (safety_benefit_prediction & safety_benefit).float().sum()
            / safety_benefit.float().sum().clamp_min(1.0)
        )
        safety_harm_recall = (
            (safety_harm_prediction & safety_harm).float().sum()
            / safety_harm.float().sum().clamp_min(1.0)
        )
        safety_neutral_correct = (
            (~safety_benefit_prediction) & (~safety_harm_prediction) & safety_neutral
        ).float().sum() / safety_neutral.float().sum().clamp_min(1.0)
        safety_recalls = [
            safety_neutral_correct, safety_benefit_recall, safety_harm_recall
        ]
    else:
        safety_prediction = safety_logits.detach().argmax(dim=-1)
        safety_recalls = []
        for cls in range(3):
            safety_mask = route_valid & (safety_target == cls)
            safety_recalls.append(
                ((safety_prediction == cls) & safety_mask).float().sum()
                / safety_mask.float().sum().clamp_min(1.0)
            )
    for cls in range(3):
        utility_mask = route_valid & (utility_target == cls)
        utility_recalls.append(
            ((utility_prediction == cls) & utility_mask).float().sum()
            / utility_mask.float().sum().clamp_min(1.0)
        )

    # ------------------------------------------------------------------
    # Cross-batch replay.  R4.2 fills a class quota instead of replaying only
    # when a class is completely absent.  Queue predictions are computed from
    # the pre-update queue in forward(), so a current example is never replayed
    # twice in the same optimization step.
    # ------------------------------------------------------------------
    safety_queue_logits = aux.get("v552r4_safety_queue_logits")
    safety_queue_gain_pred = aux.get("v552r4_safety_queue_gain_pred")
    safety_queue_target = aux.get("v552r4_safety_queue_target")
    safety_queue_gain_value = aux.get("v552r4_safety_queue_gain_value")
    utility_queue_logits = aux.get("v552r4_utility_queue_logits")
    utility_queue_gain_pred = aux.get("v552r4_utility_queue_gain_pred")
    utility_queue_target = aux.get("v552r4_utility_queue_target")
    utility_queue_gain_value = aux.get("v552r4_utility_queue_gain_value")
    safety_queue_benefit_magnitude = aux.get(
        "v552r42_safety_queue_benefit_magnitude"
    )
    safety_queue_harm_magnitude = aux.get(
        "v552r42_safety_queue_harm_magnitude"
    )
    utility_queue_benefit_magnitude = aux.get(
        "v552r42_utility_queue_benefit_magnitude"
    )
    utility_queue_harm_magnitude = aux.get(
        "v552r42_utility_queue_harm_magnitude"
    )
    safety_queue_ce = zero
    safety_queue_gain_loss = zero
    safety_queue_magnitude_loss = zero
    utility_queue_ce = zero
    utility_queue_gain_loss = zero
    utility_queue_magnitude_loss = zero
    safety_queue_used = zero
    utility_queue_used = zero
    safety_replay_counts = torch.zeros(3, dtype=torch.long, device=base.device)
    utility_replay_counts = torch.zeros(3, dtype=torch.long, device=base.device)

    safety_present = torch.stack([
        (route_valid & (safety_target == cls)).any() for cls in range(3)
    ])
    utility_present = torch.stack([
        (route_valid & (utility_target == cls)).any() for cls in range(3)
    ])
    minimum_replay = max(
        int(_m1(cfg, "V552R42_REPLAY_MIN_PER_CLASS", 16)), 0
    )

    if isinstance(safety_queue_logits, torch.Tensor) and safety_queue_logits.numel() > 0:
        queue_target = safety_queue_target.long()
        if r42_enabled:
            replay_mask, safety_replay_counts = _quota_replay_mask(
                queue_target=queue_target,
                current_target=safety_target,
                current_valid=route_valid,
                minimum_per_class=minimum_replay,
            )
            if bool(replay_mask.any().item()):
                replay_terms = _r42_critic_terms(
                    logits=safety_queue_logits[replay_mask],
                    benefit_magnitude=safety_queue_benefit_magnitude[replay_mask],
                    harm_magnitude=safety_queue_harm_magnitude[replay_mask],
                    exact_gain=safety_queue_gain_value[replay_mask],
                    target=queue_target[replay_mask],
                    valid=torch.ones_like(queue_target[replay_mask], dtype=torch.bool),
                    gain_unit=gain_unit,
                )
                safety_queue_ce = replay_terms["ce"]
                if not bool(_m1(cfg, "V552R43_QUEUE_CE_ONLY_ENABLED", False)):
                    safety_queue_gain_loss = replay_terms["expected_gain_loss"]
                    safety_queue_magnitude_loss = replay_terms["magnitude_loss"]
                safety_queue_used = replay_mask.float().mean()
        else:
            missing_class = ~safety_present.to(queue_target.device)
            replay_mask = missing_class[queue_target]
            if bool(replay_mask.any().item()):
                safety_queue_ce = F.cross_entropy(
                    safety_queue_logits[replay_mask], queue_target[replay_mask]
                )
                safety_queue_gain_loss = F.smooth_l1_loss(
                    safety_queue_gain_pred[replay_mask] / gain_unit,
                    (safety_queue_gain_value[replay_mask] / gain_unit).clamp(-1.0, 1.0),
                    beta=0.10,
                )
                safety_queue_magnitude_loss = safety_queue_gain_loss
                safety_queue_used = replay_mask.float().mean()

    if isinstance(utility_queue_logits, torch.Tensor) and utility_queue_logits.numel() > 0:
        queue_target = utility_queue_target.long()
        if r42_enabled:
            replay_mask, utility_replay_counts = _quota_replay_mask(
                queue_target=queue_target,
                current_target=utility_target,
                current_valid=route_valid,
                minimum_per_class=minimum_replay,
            )
            if bool(replay_mask.any().item()):
                replay_terms = _r42_critic_terms(
                    logits=utility_queue_logits[replay_mask],
                    benefit_magnitude=utility_queue_benefit_magnitude[replay_mask],
                    harm_magnitude=utility_queue_harm_magnitude[replay_mask],
                    exact_gain=utility_queue_gain_value[replay_mask],
                    target=queue_target[replay_mask],
                    valid=torch.ones_like(queue_target[replay_mask], dtype=torch.bool),
                    gain_unit=gain_unit,
                )
                utility_queue_ce = replay_terms["ce"]
                if not bool(_m1(cfg, "V552R43_QUEUE_CE_ONLY_ENABLED", False)):
                    utility_queue_gain_loss = replay_terms["expected_gain_loss"]
                    utility_queue_magnitude_loss = replay_terms["magnitude_loss"]
                utility_queue_used = replay_mask.float().mean()
        else:
            missing_class = ~utility_present.to(queue_target.device)
            replay_mask = missing_class[queue_target]
            if bool(replay_mask.any().item()):
                utility_queue_ce = F.cross_entropy(
                    utility_queue_logits[replay_mask], queue_target[replay_mask]
                )
                utility_queue_gain_loss = F.smooth_l1_loss(
                    utility_queue_gain_pred[replay_mask] / gain_unit,
                    (utility_queue_gain_value[replay_mask] / gain_unit).clamp(-1.0, 1.0),
                    beta=0.10,
                )
                utility_queue_magnitude_loss = utility_queue_gain_loss
                utility_queue_used = replay_mask.float().mean()

    # ------------------------------------------------------------------
    # Useful/Null Atom target. R4.2 measured how much of a broad atom mask
    # overlaps Base error, but did not ask whether the selected Delete/Fill
    # action actually corrects those pixels. As Base improves and its residual
    # error becomes thin, that support-overlap purity necessarily collapses.
    # R4.3 instead measures action-aware correction precision and capture using
    # the exact five-route bank already used by Teacher/deployment.
    # ------------------------------------------------------------------
    base_hard = (base >= 0.5)[:, 0]
    gt_hard = (gt >= 0.5)[:, 0]
    error = base_hard != gt_hard
    error_pixels = error.flatten(1).sum(dim=1).clamp_min(1).to(base.dtype)
    if r43_enabled and bool(
        _m1(cfg, "V552R43_ACTION_AWARE_ATOM_TARGET_ENABLED", True)
    ):
        route_hard = route_candidates.detach() >= 0.5
        changed = route_hard != base_hard[:, None, None]
        corrected = changed & (route_hard == gt_hard[:, None, None]) & (
            base_hard[:, None, None] != gt_hard[:, None, None]
        )
        damaged = changed & (route_hard != gt_hard[:, None, None]) & (
            base_hard[:, None, None] == gt_hard[:, None, None]
        )
        corrected_pixels = corrected.flatten(3).sum(dim=3).to(base.dtype)
        damaged_pixels = damaged.flatten(3).sum(dim=3).to(base.dtype)
        changed_pixels = changed.flatten(3).sum(dim=3).to(base.dtype)
        route_precision = corrected_pixels / (
            corrected_pixels + damaged_pixels
        ).clamp_min(1.0)
        route_capture = corrected_pixels / error_pixels[:, None, None]
        best_absolute_gain, best_absolute_route = absolute_gain.max(dim=2)
        gather_index = best_absolute_route[:, :, None]
        atom_quality_target = route_precision.gather(2, gather_index).squeeze(2).detach()
        atom_capture_target = route_capture.gather(2, gather_index).squeeze(2).detach()
        best_changed_pixels = changed_pixels.gather(2, gather_index).squeeze(2).detach()
        best_corrected_pixels = corrected_pixels.gather(2, gather_index).squeeze(2).detach()
        best_damaged_pixels = damaged_pixels.gather(2, gather_index).squeeze(2).detach()
    else:
        correct_edit_mass = (
            atom_masks.bool() & error[:, None]
        ).flatten(2).sum(dim=2).to(base.dtype)
        atom_area_pixels = atom_masks.flatten(2).sum(dim=2).clamp_min(1.0)
        atom_quality_target = (correct_edit_mass / atom_area_pixels).detach()
        atom_capture_target = (
            correct_edit_mass / error_pixels[:, None]
        ).detach()
        best_absolute_gain, _ = absolute_gain.max(dim=2)
        best_changed_pixels = atom_area_pixels.detach()
        best_corrected_pixels = correct_edit_mass.detach()
        best_damaged_pixels = (atom_area_pixels - correct_edit_mass).clamp_min(0.0).detach()
    atom_quality_probs = torch.sigmoid(aux["v552_atom_quality_logits"])
    atom_quality_loss = _masked_mean(
        F.smooth_l1_loss(
            atom_quality_probs,
            atom_quality_target,
            beta=0.10,
            reduction="none",
        ),
        valid,
    )
    useful_quality = max(
        float(_m1(cfg, "V552R4_USEFUL_ATOM_MIN_PURITY",
                  _m1(cfg, "V552R3_USEFUL_ATOM_MIN_PURITY", 0.18))), 0.0
    )
    useful_gain = max(
        float(_m1(cfg, "V552R4_USEFUL_ATOM_MIN_GAIN",
                  _m1(cfg, "V552R3_USEFUL_ATOM_MIN_GAIN", 5.0e-4))), 0.0
    )
    max_useful_atoms = max(
        int(_m1(cfg, "V552R4_MAX_USEFUL_ATOMS",
                _m1(cfg, "V552R3_MAX_USEFUL_ATOMS", 4))), 1
    )
    minimum_changed_pixels = max(
        int(_m1(cfg, "V552R43_MIN_CHANGED_PIXELS", 4)), 1
    )
    if r43_enabled:
        # Net-positive exact edits are useful. Correction precision remains a
        # soft ranking/quality target instead of a brittle broad-mask gate.
        eligible_useful = (
            valid
            & (best_absolute_gain > useful_gain)
            & (best_changed_pixels >= minimum_changed_pixels)
            & (best_corrected_pixels > best_damaged_pixels)
        )
    else:
        eligible_useful = (
            valid
            & (best_absolute_gain > useful_gain)
            & (atom_quality_target >= useful_quality)
        )
    useful_score = (
        best_absolute_gain
        + 0.25 * atom_quality_target
        + 0.10 * atom_capture_target
    ).masked_fill(~eligible_useful, -1.0e4)
    top_k = min(max_useful_atoms, n)
    top_values, top_indices = useful_score.topk(top_k, dim=1)
    useful_atom_target = torch.zeros_like(valid)
    useful_atom_target.scatter_(1, top_indices, top_values > -1.0e3)

    presence_logits = aux["v538_slot_presence_logits"]
    if r43_enabled and bool(_m1(cfg, "V552R43_BALANCED_PRESENCE_ENABLED", True)):
        useful_presence_loss = _balanced_binary_bce_with_logits(
            presence_logits, useful_atom_target, valid
        )
    else:
        presence_element = F.binary_cross_entropy_with_logits(
            presence_logits,
            useful_atom_target.to(presence_logits.dtype),
            reduction="none",
        )
        positive = useful_atom_target.float().sum().detach()
        negative = (~useful_atom_target).float().sum().detach()
        dynamic_positive_weight = (
            negative / positive.clamp_min(1.0)
        ).clamp(0.5, 2.0)
        presence_weight = torch.where(
            useful_atom_target,
            presence_element.new_full(
                presence_element.shape, float(dynamic_positive_weight)
            ),
            torch.ones_like(presence_element),
        )
        useful_presence_loss = (
            presence_element * presence_weight
        ).sum() / presence_weight.sum().clamp_min(1.0)
    predicted_presence = torch.sigmoid(presence_logits)
    target_cardinality = useful_atom_target.float().sum(dim=1)
    predicted_cardinality = predicted_presence.sum(dim=1)
    # R4.1: cardinality is measured in atoms, not divided by the slot count.
    # The previous /N normalization reduced its gradient by roughly N^2 and
    # allowed 3-4 active predictions for only ~2 useful atoms.
    cardinality_loss = F.smooth_l1_loss(
        predicted_cardinality,
        target_cardinality,
        beta=0.50,
    )
    null_mask = ~useful_atom_target
    false_atom_loss = (
        predicted_presence
        * (1.0 - atom_quality_target)
        * null_mask.to(base.dtype)
    ).sum() / null_mask.float().sum().clamp_min(1.0)

    scale_logits = aux["v551_scale_logits"]
    area_fraction = atom_masks.mean(dim=(-2, -1))
    scale_target = _area_scale_target(
        area_fraction,
        _m1(cfg, "V551_SCALE_AREA_THRESHOLDS", [0.001, 0.006, 0.02]),
        int(scale_logits.shape[2]),
    )
    scale_per = F.cross_entropy(
        scale_logits.reshape(b * n, -1),
        scale_target.reshape(-1),
        reduction="none",
    ).reshape(b, n)
    scale_loss = _masked_mean(scale_per, valid)

    # ------------------------------------------------------------------
    # Composer exact Teacher: one source of truth.
    # ------------------------------------------------------------------
    # The model forward already built the exact GT rollout and used those same
    # choices to advance the teacher-forced state.  Rebuilding a second rollout
    # here is forbidden: even a slightly different Top-K pool creates masked
    # labels and state/target disagreement.
    teacher = {
        "target_indices": aux["v552r41_teacher_target_indices"].detach().long(),
        "target_marginal_gains": aux[
            "v552r41_teacher_target_marginal_gains"
        ].detach(),
        "target_eligible": aux["v552r41_teacher_target_eligible"].detach().bool(),
        "target_active": aux["v552r41_teacher_target_active"].detach().bool(),
        "target_step_gains": aux["v552r41_teacher_target_step_gains"].detach(),
        "teacher_state_probs": aux["v552r41_teacher_state_probs"].detach(),
        "teacher_final_probs": aux["v552r41_teacher_final_probs"].detach(),
    }
    step_logits = aux["v552_composer_step_logits"]
    step_scores = aux["v552_composer_step_candidate_scores"]
    steps = min(step_logits.shape[1], teacher["target_indices"].shape[1])
    if step_logits.shape[2] != n + 1:
        raise RuntimeError(
            f"R4.1 composer logits must have N+1 classes, got "
            f"{step_logits.shape[2]} for N={n}"
        )
    forced_target = aux.get("v552_composer_forced_target_indices")
    if not isinstance(forced_target, torch.Tensor):
        raise RuntimeError("R4.1 requires the forward teacher-forced target trace")
    if forced_target.shape[0] != b or forced_target.shape[1] < steps:
        raise RuntimeError("R4.1 forced target trace has an invalid shape")
    forced_mismatch_mask = (
        forced_target[:, :steps].detach().long()
        != teacher["target_indices"][:, :steps]
    ) & teacher["target_active"][:, :steps]
    if bool(forced_mismatch_mask.any().item()):
        raise RuntimeError(
            "R4.1 single-teacher contract violated: forward state and loss "
            "targets are not identical"
        )

    ce_terms = []
    stop_terms = []
    continue_terms = []
    marginal_terms = []
    selection_correct = zero.new_zeros(())
    selection_count = zero.new_zeros(())
    stop_correct = zero.new_zeros(())
    stop_count = zero.new_zeros(())
    marginal_abs = zero.new_zeros(())
    marginal_count = zero.new_zeros(())
    target_masked_count = zero.new_zeros(())
    target_active_count = zero.new_zeros(())
    composer_marginal_neutral = zero
    composer_marginal_benefit = zero
    composer_marginal_harm = zero
    marginal_class_sums = [zero.new_zeros(()) for _ in range(3)]
    marginal_class_counts = [zero.new_zeros(()) for _ in range(3)]
    selective_negative_mass = zero.new_zeros(())
    selective_positive_mass = zero.new_zeros(())
    selective_coverage_mass = zero.new_zeros(())
    selective_active_count = zero.new_zeros(())
    for step in range(steps):
        active = teacher["target_active"][:, step]
        target = teacher["target_indices"][:, step]
        eligible = teacher["target_eligible"][:, step]
        teacher_logits = step_logits[:, step]
        if bool(active.any().item()):
            target_logit = teacher_logits.gather(1, target[:, None])[:, 0]
            masked_target = active & (target_logit < -1.0e3)
            target_masked_count += masked_target.float().sum()
            target_active_count += active.float().sum()
            if bool(masked_target.any().item()):
                raise RuntimeError(
                    "R4.1 single-teacher contract violated: an active teacher "
                    "target is masked in the exact forward trace"
                )
            if r46_enabled:
                # Single one-step set decision: Preserve is class N with exact
                # zero utility, candidates are classes 0..N-1.  This is the
                # same action space used by formal deployment.
                ce_terms.append(
                    F.cross_entropy(teacher_logits[active], target[active])
                )
            elif r42_enabled:
                # Factor Stop/Continue from candidate identity.  The original
                # N+1 CE lets the frequent Stop class suppress all candidate
                # logits at once and then over-corrects when execution starts.
                stop_target = target == n
                stop_logit = (
                    teacher_logits[:, n]
                    - torch.logsumexp(teacher_logits[:, :n], dim=1)
                )
                stop_terms.append(
                    _balanced_binary_bce_with_logits(
                        stop_logit, stop_target, active
                    )
                )
                continue_mask = active & (~stop_target)
                if bool(continue_mask.any().item()):
                    continue_terms.append(
                        F.cross_entropy(
                            teacher_logits[continue_mask, :n],
                            target[continue_mask],
                        )
                    )
            else:
                ce_terms.append(
                    F.cross_entropy(teacher_logits[active], target[active])
                )
            prediction = teacher_logits.detach().argmax(dim=1)
            selection_correct += ((prediction == target) & active).float().sum()
            selection_count += active.float().sum()
            stop_mask = active & (target == n)
            stop_correct += ((prediction == n) & stop_mask).float().sum()
            stop_count += stop_mask.float().sum()
        if bool(eligible.any().item()):
            exact_marginal = teacher["target_marginal_gains"][:, step]
            if (
                r45_enabled
                or (
                    r44_enabled
                    and bool(_m1(cfg, "V552R44_COMPOSER_SELECTIVE_RISK_ENABLED", True))
                )
            ) and bool(active.any().item()):
                # Candidate probability mass is the differentiable coverage.
                # Penalize probability assigned to truly harmful marginal
                # actions and retain a minimum coverage so the trivial all-Stop
                # policy cannot minimize risk.
                full_probability = F.softmax(teacher_logits[active], dim=1)
                candidate_probability = full_probability[:, :n]
                active_eligible = eligible[active].to(candidate_probability.dtype)
                candidate_probability = candidate_probability * active_eligible
                exact_normalized = (
                    exact_marginal[active] / gain_unit
                ).clamp(-1.0, 1.0)
                selective_negative_mass += (
                    candidate_probability * F.relu(-exact_normalized)
                ).sum()
                selective_positive_mass += (
                    candidate_probability * F.relu(exact_normalized)
                ).sum()
                selective_coverage_mass += candidate_probability.sum()
                selective_active_count += active.float().sum()
            per_marginal = F.smooth_l1_loss(
                step_scores[:, step] / gain_unit,
                (exact_marginal / gain_unit).clamp(-1.0, 1.0),
                beta=0.10,
                reduction="none",
            )
            if r42_enabled:
                marginal_target = torch.zeros_like(
                    exact_marginal, dtype=torch.long
                )
                marginal_target = torch.where(
                    exact_marginal > absolute_margin,
                    torch.ones_like(marginal_target),
                    marginal_target,
                )
                marginal_target = torch.where(
                    exact_marginal < -absolute_margin,
                    torch.full_like(marginal_target, 2),
                    marginal_target,
                )
                marginal_terms.append(
                    _equal_class_mean(
                        per_marginal,
                        marginal_target,
                        eligible,
                        num_classes=3,
                    )
                )
                for cls in range(3):
                    cls_mask = eligible & (marginal_target == cls)
                    if bool(cls_mask.any().item()):
                        marginal_class_sums[cls] += per_marginal[cls_mask].mean()
                        marginal_class_counts[cls] += 1.0
            else:
                marginal_terms.append(per_marginal[eligible].mean())
            marginal_abs += (
                step_scores[:, step].detach()[eligible] - exact_marginal[eligible]
            ).abs().sum()
            marginal_count += eligible.float().sum()
    if r46_enabled:
        composer_stop_loss = zero
        composer_continue_loss = zero
        composer_selection_loss = torch.stack(ce_terms).mean() if ce_terms else zero
        composer_marginal_neutral = (
            marginal_class_sums[0] / marginal_class_counts[0].clamp_min(1.0)
        ) if bool(marginal_class_counts[0].item() > 0) else zero
        composer_marginal_benefit = (
            marginal_class_sums[1] / marginal_class_counts[1].clamp_min(1.0)
        ) if bool(marginal_class_counts[1].item() > 0) else zero
        composer_marginal_harm = (
            marginal_class_sums[2] / marginal_class_counts[2].clamp_min(1.0)
        ) if bool(marginal_class_counts[2].item() > 0) else zero
    elif r42_enabled:
        composer_stop_loss = torch.stack(stop_terms).mean() if stop_terms else zero
        composer_continue_loss = (
            torch.stack(continue_terms).mean() if continue_terms else zero
        )
        composer_selection_loss = composer_stop_loss + composer_continue_loss
        composer_marginal_neutral = (
            marginal_class_sums[0] / marginal_class_counts[0].clamp_min(1.0)
        )
        composer_marginal_benefit = (
            marginal_class_sums[1] / marginal_class_counts[1].clamp_min(1.0)
        )
        composer_marginal_harm = (
            marginal_class_sums[2] / marginal_class_counts[2].clamp_min(1.0)
        )
    else:
        composer_stop_loss = zero
        composer_continue_loss = zero
        composer_selection_loss = torch.stack(ce_terms).mean() if ce_terms else zero
    composer_marginal_loss = (
        torch.stack(marginal_terms).mean() if marginal_terms else zero
    )
    composer_selective_coverage = (
        selective_coverage_mass / selective_active_count.clamp_min(1.0)
    )
    composer_selective_risk = (
        selective_negative_mass / selective_coverage_mass.clamp_min(EPS)
    )
    composer_selective_positive_gain = (
        selective_positive_mass / selective_coverage_mass.clamp_min(EPS)
    )
    selective_coverage_target = max(
        float(_m1(cfg, "V552R44_COMPOSER_TARGET_COVERAGE", 0.10)), 0.0
    )
    composer_coverage_penalty = F.relu(
        composer_selective_coverage.new_tensor(selective_coverage_target)
        - composer_selective_coverage
    ).square()
    student_final = _v552r2_student_soft_final(
        base=base,
        step_logits=step_logits[:, :steps],
        editor_logit_delta=teacher_safe_delta,
    )
    composer_final_bce = F.binary_cross_entropy(student_final, gt)
    intersection = (student_final * gt).flatten(1).sum(dim=1)
    denominator = student_final.flatten(1).sum(dim=1) + gt.flatten(1).sum(dim=1)
    composer_final_dice = (
        1.0 - (2.0 * intersection + EPS) / (denominator + EPS)
    ).mean()

    pairwise_loss = zero
    pairwise_accuracy = zero
    pairwise_scale = _curriculum(
        epoch,
        int(_m1(cfg, "V552_PAIRWISE_START_EPOCH", 10)),
        int(_m1(cfg, "V552_PAIRWISE_RAMP_EPOCHS", 4)),
    )
    if pairwise_scale > 0.0:
        left_true = absolute_gain[:, :, :-1]
        right_true = absolute_gain[:, :, 1:]
        true_diff = left_true - right_true
        pairwise_gain_source = utility_gain_for_loss if r42_enabled else utility_gain_pred
        pred_diff = (
            pairwise_gain_source[:, :, :-1]
            - pairwise_gain_source[:, :, 1:]
        )
        pair_valid = route_valid[:, :, :-1] & route_valid[:, :, 1:]
        non_tie = pair_valid & (true_diff.abs() > absolute_margin)
        if bool(non_tie.any().item()):
            pairwise_loss = F.softplus(
                -torch.sign(true_diff[non_tie])
                * (pred_diff[non_tie] / gain_unit)
            ).mean()
            pairwise_accuracy = (
                pred_diff[non_tie] * true_diff[non_tie] > 0
            ).float().mean()

    editor_scale = _curriculum(
        epoch,
        int(_m1(cfg, "V551_EDITOR_START_EPOCH", 3)),
        int(_m1(cfg, "V551_EDITOR_RAMP_EPOCHS", 5)),
    )
    outcome_scale = _curriculum(
        epoch,
        int(_m1(cfg, "V552_OUTCOME_START_EPOCH", 4)),
        int(_m1(cfg, "V552_OUTCOME_RAMP_EPOCHS", 5)),
    )
    composer_scale = _curriculum(
        epoch,
        int(_m1(cfg, "V552_COMPOSER_START_EPOCH", 8)),
        int(_m1(cfg, "V552_COMPOSER_RAMP_EPOCHS", 5)),
    )

    if r45_enabled:
        # In R4.5 the Safety value/magnitude path is not part of the optimizer
        # objective: Safety is a factorized veto/accept classifier. Utility is
        # sign classification plus direct signed regression.
        safety_critic_objective = (
            float(_m1(cfg, "V552R45_SAFETY_BENEFIT_WEIGHT", 1.0))
            * safety_benefit_binary_loss
            + float(_m1(cfg, "V552R45_SAFETY_HARM_WEIGHT", 1.0))
            * safety_harm_binary_loss
        )
        utility_critic_objective = (
            (0.0 if r46_enabled else float(
                _m1(cfg, "V552R45_UTILITY_SIGN_WEIGHT", 1.0)
            )) * utility_ce
            + float(_m1(cfg, "V552R45_UTILITY_VALUE_WEIGHT", 1.0))
            * utility_gain_loss
        )
        editor_objective = (
            float(_m1(cfg, "V551_ROUTE_LOSS_WEIGHT", 1.0)) * route_loss
            + float(_m1(cfg, "V551_EDITABILITY_LOSS_WEIGHT", 1.0)) * editability_loss
            + safety_critic_objective
            + float(_m1(cfg, "V551_LOCAL_BCE_WEIGHT", 0.50)) * local_bce_loss
            + float(_m1(cfg, "V551_LOCAL_DICE_WEIGHT", 0.50)) * local_dice_loss
            + float(_m1(cfg, "V551_OUTSIDE_PRESERVE_WEIGHT", 1.0))
            * outside_preserve_loss
            + float(_m1(cfg, "V551_LOCAL_RESIDUAL_L1_WEIGHT", 0.30))
            * residual_l1
            + float(_m1(cfg, "V552_EDITOR_RELATIVE_SAFETY_WEIGHT", 6.0))
            * relative_safety_loss
        )
        outcome_objective = utility_critic_objective
        magnitude_scale = 0.0
        expected_gain_scale = 1.0
    elif r42_enabled:
        critic_ce_weight = float(
            _m1(cfg, "V552R42_CRITIC_OUTCOME_CE_WEIGHT", 1.0)
        )
        critic_magnitude_weight = float(
            _m1(cfg, "V552R42_CRITIC_CONDITIONAL_MAGNITUDE_WEIGHT", 1.0)
        )
        critic_expected_gain_weight = float(
            _m1(cfg, "V552R42_CRITIC_EXPECTED_GAIN_WEIGHT", 1.0)
        )
        queue_aux_weight = float(
            _m1(cfg, "V552R42_QUEUE_AUX_WEIGHT", 0.25)
        )
        if r43_enabled:
            magnitude_scale = _curriculum(
                epoch,
                int(_m1(cfg, "V552R43_MAGNITUDE_START_EPOCH", 7)),
                int(_m1(cfg, "V552R43_MAGNITUDE_RAMP_EPOCHS", 4)),
            )
            expected_gain_scale = _curriculum(
                epoch,
                int(_m1(cfg, "V552R43_EXPECTED_GAIN_START_EPOCH", 10)),
                int(_m1(cfg, "V552R43_EXPECTED_GAIN_RAMP_EPOCHS", 5)),
            )
        else:
            magnitude_scale = 1.0
            expected_gain_scale = 1.0
        safety_critic_objective = (
            critic_ce_weight * safety_ce
            + critic_magnitude_weight * float(magnitude_scale) * safety_magnitude_loss
            + critic_expected_gain_weight * float(expected_gain_scale) * safety_gain_loss
            + queue_aux_weight
            * (
                safety_queue_ce
                + safety_queue_magnitude_loss
                + safety_queue_gain_loss
            )
        )
        utility_critic_objective = (
            critic_ce_weight * utility_ce
            + critic_magnitude_weight * float(magnitude_scale) * utility_magnitude_loss
            + critic_expected_gain_weight * float(expected_gain_scale) * utility_gain_loss
            + queue_aux_weight
            * (
                utility_queue_ce
                + utility_queue_magnitude_loss
                + utility_queue_gain_loss
            )
        )
        editor_objective = (
            float(_m1(cfg, "V551_ROUTE_LOSS_WEIGHT", 1.0)) * route_loss
            + float(_m1(cfg, "V551_EDITABILITY_LOSS_WEIGHT", 1.0)) * editability_loss
            + safety_critic_objective
            + float(_m1(cfg, "V551_LOCAL_BCE_WEIGHT", 0.50)) * local_bce_loss
            + float(_m1(cfg, "V551_LOCAL_DICE_WEIGHT", 0.50)) * local_dice_loss
            + float(_m1(cfg, "V551_OUTSIDE_PRESERVE_WEIGHT", 1.0))
            * outside_preserve_loss
            + float(_m1(cfg, "V551_LOCAL_RESIDUAL_L1_WEIGHT", 0.30))
            * residual_l1
            + float(_m1(cfg, "V552_EDITOR_RELATIVE_SAFETY_WEIGHT", 6.0))
            * relative_safety_loss
        )
        outcome_objective = (
            utility_critic_objective
            + float(_m1(cfg, "V552_PAIRWISE_WEIGHT", 0.05))
            * float(pairwise_scale)
            * pairwise_loss
        )
    else:
        safety_critic_objective = (
            float(_m1(cfg, "V552R4_EDITOR_SAFETY_CE_WEIGHT", 2.0)) * safety_ce
            + float(_m1(cfg, "V552R4_EDITOR_INCREMENTAL_GAIN_WEIGHT", 2.0))
            * safety_gain_loss
            + float(_m1(cfg, "V552R4_EDITOR_BENEFIT_SIGN_WEIGHT", 2.0))
            * safety_benefit_sign
            + float(_m1(cfg, "V552R4_EDITOR_HARM_SIGN_WEIGHT", 3.0))
            * safety_harm_sign
            + float(_m1(cfg, "V552R4_EDITOR_NEUTRAL_SIGN_WEIGHT", 0.5))
            * safety_neutral_band
            + float(_m1(cfg, "V552R4_EDITOR_QUEUE_CE_WEIGHT", 1.0))
            * safety_queue_ce
            + float(_m1(cfg, "V552R4_EDITOR_QUEUE_GAIN_WEIGHT", 0.5))
            * safety_queue_gain_loss
        )
        utility_critic_objective = (
            float(_m1(cfg, "V552R4_UTILITY_CE_WEIGHT", 2.0)) * utility_ce
            + float(_m1(cfg, "V552R4_UTILITY_GAIN_WEIGHT", 2.0))
            * utility_gain_loss
            + float(_m1(cfg, "V552R4_UTILITY_BENEFIT_SIGN_WEIGHT", 2.0))
            * utility_benefit_sign
            + float(_m1(cfg, "V552R4_UTILITY_HARM_SIGN_WEIGHT", 3.0))
            * utility_harm_sign
            + float(_m1(cfg, "V552R4_UTILITY_NEUTRAL_SIGN_WEIGHT", 0.5))
            * utility_neutral_band
            + float(_m1(cfg, "V552R4_UTILITY_QUEUE_CE_WEIGHT", 1.0))
            * utility_queue_ce
            + float(_m1(cfg, "V552R4_UTILITY_QUEUE_GAIN_WEIGHT", 0.5))
            * utility_queue_gain_loss
        )
        editor_objective = (
            float(_m1(cfg, "V551_ROUTE_LOSS_WEIGHT", 1.0)) * route_loss
            + float(_m1(cfg, "V551_EDITABILITY_LOSS_WEIGHT", 1.0)) * editability_loss
            + safety_critic_objective
            + float(_m1(cfg, "V551_LOCAL_BCE_WEIGHT", 0.50)) * local_bce_loss
            + float(_m1(cfg, "V551_LOCAL_DICE_WEIGHT", 0.50)) * local_dice_loss
            + float(_m1(cfg, "V551_OUTSIDE_PRESERVE_WEIGHT", 1.0))
            * outside_preserve_loss
            + float(_m1(cfg, "V551_LOCAL_RESIDUAL_L1_WEIGHT", 0.30))
            * residual_l1
            + float(_m1(cfg, "V552_EDITOR_RELATIVE_SAFETY_WEIGHT", 6.0))
            * relative_safety_loss
        )
        outcome_objective = (
            utility_critic_objective
            + float(_m1(cfg, "V552_PAIRWISE_WEIGHT", 0.05))
            * float(pairwise_scale)
            * pairwise_loss
        )

    if r45_enabled:
        # R4.21.2 one-step policy has no learned Composer parameters in the
        # deployed decision: it is argmax([Stop=0, predicted DeltaDice_i]).
        # Keep all legacy Composer losses as diagnostics, but do not optimize an
        # unused learned Stop/correction head in this mode.
        if r4212_zero_stop_one_step:
            composer_objective = zero
        else:
            composer_objective = (
                (
                    float(_m1(cfg, "V552R46_COMPOSER_CHOICE_WEIGHT", 1.0))
                    * composer_selection_loss
                    + float(_m1(cfg, "V552R46_COMPOSER_VALUE_WEIGHT", 0.50))
                    * composer_marginal_loss
                ) if r46_enabled else (
                    float(_m1(cfg, "V552R45_COMPOSER_EXECUTE_WEIGHT", 1.0))
                    * composer_stop_loss
                    + float(_m1(cfg, "V552R45_COMPOSER_CHOICE_WEIGHT", 1.0))
                    * composer_continue_loss
                    + float(_m1(cfg, "V552R45_COMPOSER_VALUE_WEIGHT", 0.5))
                    * composer_marginal_loss
                    + float(_m1(cfg, "V552R45_COMPOSER_FINAL_BCE_WEIGHT", 0.05))
                    * composer_final_bce
                    + float(_m1(cfg, "V552R45_COMPOSER_FINAL_DICE_WEIGHT", 0.05))
                    * composer_final_dice
                )
            )
    else:
        composer_objective = (
            float(_m1(cfg, "V552_COMPOSER_SELECTION_WEIGHT", 1.0))
            * composer_selection_loss
            + float(
                _m1(
                    cfg,
                    "V552_COMPOSER_MARGINAL_GAIN_WEIGHT",
                    1.0 if r42_enabled else 2.0,
                )
            )
            * composer_marginal_loss
            + float(_m1(cfg, "V552_COMPOSER_FINAL_BCE_WEIGHT", 0.10))
            * composer_final_bce
            + float(_m1(cfg, "V552_COMPOSER_FINAL_DICE_WEIGHT", 0.10))
            * composer_final_dice
            + (
                float(_m1(cfg, "V552R44_COMPOSER_SELECTIVE_RISK_WEIGHT", 0.50))
                * composer_selective_risk
                + float(_m1(cfg, "V552R44_COMPOSER_COVERAGE_WEIGHT", 0.25))
                * composer_coverage_penalty
            )
            * (
                1.0
                if r44_enabled and bool(
                    _m1(cfg, "V552R44_COMPOSER_SELECTIVE_RISK_ENABLED", True)
                )
                else 0.0
            )
        )
    if v560_clean_core:
        # Clean M2 owns one and only one physical target: the exact signed
        # DeltaDice of the *M1 route itself* (route index 0).  The feature bank
        # is detached in the model, so this objective trains the utility head
        # without distorting M1 geometry.
        v560_valid = route_valid[:, :, 0] & valid
        v560_pred_delta = utility_direct_signed_gain[:, :, 0]
        v560_true_delta = absolute_gain[:, :, 0].detach()
        v560_direct_delta_l1 = _masked_mean(
            F.l1_loss(v560_pred_delta, v560_true_delta, reduction="none"),
            v560_valid,
        )
        v560_sign_valid = v560_valid & (v560_true_delta.abs() > absolute_margin)
        v560_direct_delta_sign_accuracy = _masked_mean(
            ((v560_pred_delta.detach() * v560_true_delta) > 0).to(base.dtype),
            v560_sign_valid,
        )
        editor_objective = zero
        outcome_objective = v560_direct_delta_l1
        composer_objective = zero
    else:
        v560_direct_delta_l1 = zero
        v560_direct_delta_sign_accuracy = zero

    # R4.21.2 makes slot presence literal existence/no-object in the M1 set
    # loss.  The legacy R552R3 "useful atom" presence/cardinality/false-atom
    # terms encode *positive utility*, so keeping them would give the same
    # presence logit contradictory labels (real-but-harmful component = 1 in
    # M1 set existence, but 0 here).  Utility now belongs exclusively to M2.
    useful_presence_scale_r4212 = 0.0 if r4212_existence_no_object else 1.0
    m1_extra = (
        float(_m1(cfg, "V551_SCALE_LOSS_WEIGHT", 0.50)) * scale_loss
        + float(_m1(cfg, "V552_ATOM_QUALITY_WEIGHT", 0.50)) * atom_quality_loss
        + useful_presence_scale_r4212
        * float(_m1(cfg, "V552R3_USEFUL_ATOM_PRESENCE_WEIGHT", 1.0))
        * useful_presence_loss
        + useful_presence_scale_r4212
        * float(_m1(cfg, "V552R3_CARDINALITY_WEIGHT", 0.25))
        * cardinality_loss
        + useful_presence_scale_r4212
        * float(
            _m1(
                cfg,
                "V552R43_FALSE_ATOM_WEIGHT",
                _m1(cfg, "V552R3_FALSE_ATOM_WEIGHT", 0.25),
            )
            if r43_enabled
            else _m1(cfg, "V552R3_FALSE_ATOM_WEIGHT", 0.25)
        )
        * false_atom_loss
        + (
            float(_m1(cfg, "V552R45_ERROR_BCE_WEIGHT", 1.0)) * error_region_bce
            + float(_m1(cfg, "V552R45_ERROR_TVERSKY_WEIGHT", 1.0))
            * error_region_tversky
        )
        * (1.0 if r45_enabled else 0.0)
    )
    if r46_enabled:
        # Strict single-teacher M1: all current-student usefulness, cardinality,
        # quality and EPR losses are diagnostic-only.
        m1_extra = m1_extra * 0.0
    m2_extra = (
        editor_objective * base.new_tensor(editor_scale)
        + outcome_objective * base.new_tensor(outcome_scale)
        + composer_objective * base.new_tensor(composer_scale)
    )

    if v560_clean_core:
        # v538_loss owns the minimal M1 set objective.  V551 contributes only
        # the physical utility regressor; no Safety/Stop/readiness surrogate is
        # allowed to acquire optimizer ownership in the Clean Core path.
        m1_extra = m1_extra * 0.0
        m2_extra = v560_direct_delta_l1

    teacher_final_gain = (
        _hard_dice_many(teacher["teacher_final_probs"][:, 0][:, None], gt)[:, 0]
        - base_dice
    )
    teacher_steps = (teacher["target_indices"] < n).float().sum(dim=1)
    predicted_step_count = aux.get(
        "v552_composer_predicted_step_count", zero.new_zeros((b,))
    ).float()
    ungated_step_count = aux.get(
        "v552_composer_ungated_step_count", zero.new_zeros((b,))
    ).float()
    ungated_final = aux.get("v552_composer_ungated_final_probability")
    if isinstance(ungated_final, torch.Tensor):
        ungated_gain_case = (
            _hard_dice_many(ungated_final.detach()[:, 0][:, None], gt)[:, 0]
            - base_dice
        )
    else:
        ungated_gain_case = torch.zeros_like(base_dice)
    forced_target_mismatch = forced_mismatch_mask.float().mean()

    useful_union = (
        atom_masks.bool() & useful_atom_target[:, :, None, None]
    ).any(dim=1)
    if r43_enabled:
        corrected_selected = (
            best_corrected_pixels
            * useful_atom_target.to(best_corrected_pixels.dtype)
        ).sum(dim=1)
        capture = (corrected_selected / error_pixels).clamp_max(1.0).mean()
        selected_precision = _masked_mean(
            atom_quality_target, useful_atom_target
        )
    else:
        capture = (
            (useful_union & error).flatten(1).sum(dim=1).to(base.dtype)
            / error_pixels
        ).mean()
        selected_precision = _masked_mean(atom_quality_target, valid)
    m1_oracle = torch.cat(
        [torch.zeros_like(base_dice[:, None]), m1_gain.masked_fill(~valid, -1.0e4)],
        dim=1,
    ).max(dim=1).values.mean()
    teacher_safe_gain = (
        _hard_dice_many(teacher_safe_candidate.detach(), gt) - base_dice[:, None]
    )
    editor_oracle = torch.cat(
        [
            torch.zeros_like(base_dice[:, None]),
            teacher_safe_gain.masked_fill(~valid, -1.0e4),
        ],
        dim=1,
    ).max(dim=1).values.mean()

    selected_utility_logits = (
        F.one_hot(student_safe_index, num_classes=5).to(base.dtype)[:, :, :, None]
        * utility_logits
    ).sum(dim=2)
    selected_utility_pred = selected_utility_logits.detach().argmax(dim=2)
    selected_utility_target = (
        F.one_hot(student_safe_index, num_classes=5).to(base.dtype)
        * utility_target
    ).sum(dim=2).long()
    selected_utility_gain_pred = (
        F.one_hot(student_safe_index, num_classes=5).to(base.dtype)
        * utility_gain_pred
    ).sum(dim=2)
    utility_benefit_selected = valid & (selected_utility_target == 1)
    utility_harm_selected = valid & (selected_utility_target == 2)

    safety_queue_count = aux.get(
        "v552r4_safety_queue_count", torch.zeros(3, device=base.device)
    ).to(base.dtype)
    utility_queue_count = aux.get(
        "v552r4_utility_queue_count", torch.zeros(3, device=base.device)
    ).to(base.dtype)

    diagnostics = {
        "v551_enabled": zero.new_ones(()),
        "v552r2_enabled": zero.new_ones(()),
        "v552r4_enabled": zero.new_ones(()),
        "v551_scale_loss": scale_loss.detach(),
        "v552_atom_quality_loss": atom_quality_loss.detach(),
        "v552_atom_quality_target_mean": _masked_mean(
            atom_quality_target, valid
        ).detach(),
        "v552r3_useful_atom_presence_loss": useful_presence_loss.detach(),
        "v552r3_cardinality_loss": cardinality_loss.detach(),
        "v552r3_false_atom_loss": false_atom_loss.detach(),
        "v552r3_useful_atom_count": useful_atom_target.float().sum(dim=1).mean().detach(),
        "v552r3_predicted_active_atom_count": (
            predicted_presence >= 0.5
        ).float().sum(dim=1).mean().detach(),
        "v552r4_useful_target_consistency": zero.new_ones(()),
        "v552r43_rootfix_enabled": zero.new_tensor(1.0 if r43_enabled else 0.0),
        "v552r43_action_aware_atom_target_enabled": zero.new_tensor(
            1.0
            if r43_enabled and bool(
                _m1(cfg, "V552R43_ACTION_AWARE_ATOM_TARGET_ENABLED", True)
            )
            else 0.0
        ),
        "v552r43_balanced_presence_enabled": zero.new_tensor(
            1.0
            if r43_enabled and bool(
                _m1(cfg, "V552R43_BALANCED_PRESENCE_ENABLED", True)
            )
            else 0.0
        ),
        "v552r43_atom_correction_precision": selected_precision.detach(),
        "v552r43_atom_corrected_pixel_capture": capture.detach(),
        "v552r43_atom_changed_pixels": _masked_mean(
            best_changed_pixels, valid
        ).detach(),
        "v551_route_loss": route_loss.detach(),
        "v551_editability_loss": editability_loss.detach(),
        "v552_editor_relative_gain_loss": safety_gain_loss.detach(),
        "v552_editor_relative_gain_mean": _masked_mean(
            selected_incremental_gain_true, valid
        ).detach(),
        "v552_editor_teacher_relative_gain_mean": _masked_mean(
            best_relative_gain, valid
        ).detach(),
        "v552_editor_relative_safety_loss": relative_safety_loss.detach(),
        "v552_editor_should_modify_rate": (
            should_modify.float().sum() / valid.float().sum().clamp_min(1.0)
        ).detach(),
        "v551_harm_veto_loss": relative_safety_loss.detach(),
        "v551_local_bce_loss": local_bce_loss.detach(),
        "v551_local_dice_loss": local_dice_loss.detach(),
        "v551_outside_preserve_loss": outside_preserve_loss.detach(),
        "v551_local_residual_l1": residual_l1.detach(),
        "v551_editor_train_scale": zero.new_tensor(float(editor_scale)),
        "v552_outcome_train_scale": zero.new_tensor(float(outcome_scale)),
        "v552_composer_train_scale": zero.new_tensor(float(composer_scale)),
        "v552r43_magnitude_train_scale": zero.new_tensor(
            float(magnitude_scale) if r42_enabled else 1.0
        ),
        "v552r43_expected_gain_train_scale": zero.new_tensor(
            float(expected_gain_scale) if r42_enabled else 1.0
        ),
        "v552r43_queue_ce_only_enabled": zero.new_tensor(
            1.0
            if r43_enabled and bool(
                _m1(cfg, "V552R43_QUEUE_CE_ONLY_ENABLED", True)
            )
            else 0.0
        ),
        "v552r44_rootfix_enabled": zero.new_tensor(1.0 if r44_enabled else 0.0),
        "v552r44_action_aware_quality_gate_enabled": zero.new_tensor(
            1.0 if r44_enabled and bool(
                _m1(cfg, "V552R44_ACTION_AWARE_QUALITY_GATE_ENABLED", True)
            ) else 0.0
        ),
        "v552r44_class_value_decoupling_enabled": aux.get(
            "v552r44_class_value_decoupling_enabled", zero
        ).detach(),
        "v552r44_semantic_deployment_enabled": aux.get(
            "v552r44_semantic_deployment_enabled", zero
        ).detach(),
        "v552r44_audit_shadow_enabled": aux.get(
            "v552r44_audit_shadow_enabled", zero
        ).detach(),
        "v552r45_rootfix_enabled": zero.new_tensor(1.0 if r45_enabled else 0.0),
        "v552r46_rootfix_enabled": zero.new_tensor(1.0 if r46_enabled else 0.0),
        "v552r46_single_m1_teacher_enabled": zero.new_tensor(1.0 if r46_enabled else 0.0),
        "v552r46_utility_classification_aux_only": zero.new_tensor(1.0 if r46_enabled else 0.0),
        "v552r46_one_step_composer_enabled": zero.new_tensor(1.0 if r46_enabled else 0.0),
        "v552r46_same_candidate_contract_enabled": zero.new_tensor(1.0 if r46_enabled else 0.0),
        "v552r46_formal_policy_audit_enabled": zero.new_tensor(1.0 if r46_enabled else 0.0),
        "v552r46_secondary_m1_objective_disabled": zero.new_tensor(1.0 if r46_enabled else 0.0),
        "v552r45_error_aware_enabled": aux.get(
            "v552r45_error_aware_enabled", zero
        ).detach(),
        "v552r45_factorized_safety_enabled": aux.get(
            "v552r45_factorized_safety_enabled", zero
        ).detach(),
        "v552r45_direct_signed_utility_enabled": aux.get(
            "v552r45_direct_signed_utility_enabled", zero
        ).detach(),
        "v552r4212_direct_delta_utility_enabled": zero.new_tensor(
            1.0 if r4212_direct_delta_utility else 0.0
        ),
        "v552r4212_zero_stop_one_step_enabled": zero.new_tensor(
            1.0 if r4212_zero_stop_one_step else 0.0
        ),
        "v560_clean_core_enabled": zero.new_tensor(1.0 if v560_clean_core else 0.0),
        "v560_direct_delta_l1": v560_direct_delta_l1.detach(),
        "v560_direct_delta_sign_accuracy": v560_direct_delta_sign_accuracy.detach(),
        "v552r4212_direct_delta_utility_mae": _masked_mean(
            (utility_gain_for_loss.detach() - absolute_gain.detach()).abs(),
            route_valid,
        ).detach(),
        "v552r4212_direct_delta_sign_accuracy": _masked_mean(
            ((utility_gain_for_loss.detach() * absolute_gain.detach()) > 0).to(base.dtype),
            route_valid & (absolute_gain.detach().abs() > absolute_margin),
        ).detach(),
        "v552r45_factorized_composer_enabled": aux.get(
            "v552r45_factorized_composer_enabled", zero
        ).detach(),
        "v552r45_error_region_bce": error_region_bce.detach(),
        "v552r45_error_region_tversky": error_region_tversky.detach(),
        "v552r45_error_region_precision": error_region_precision.detach(),
        "v552r45_error_region_recall": error_region_recall.detach(),
        "v552r45_safety_benefit_bce": safety_benefit_binary_loss.detach(),
        "v552r45_safety_harm_bce": safety_harm_binary_loss.detach(),
        "v552r45_utility_signed_gain_loss": (
            utility_gain_loss.detach() if r45_enabled else zero.detach()
        ),
        "v552r45_utility_signed_gain_mean_benefit": _masked_mean(
            utility_gain_for_loss, utility_benefit
        ).detach(),
        "v552r45_utility_signed_gain_mean_harm": _masked_mean(
            utility_gain_for_loss, utility_harm
        ).detach(),
        "v552r45_composer_execute_loss": composer_stop_loss.detach(),
        "v552r45_composer_choice_loss": composer_continue_loss.detach(),
        "v552r45_composer_value_loss": composer_marginal_loss.detach(),
        "v552r4_editor_safety_ce_loss": safety_ce.detach(),
        "v552r4_editor_safety_gain_loss": safety_gain_loss.detach(),
        "v552r4_editor_safety_balanced_accuracy": torch.stack(
            safety_recalls
        ).mean().detach(),
        "v552r4_editor_safety_benefit_recall": safety_recalls[1].detach(),
        "v552r4_editor_safety_harm_recall": safety_recalls[2].detach(),
        "v552r4_utility_ce_loss": utility_ce.detach(),
        "v552r4_utility_gain_loss": utility_gain_loss.detach(),
        "v552r4_utility_balanced_accuracy": torch.stack(
            utility_recalls
        ).mean().detach(),
        "v552r4_utility_benefit_recall": utility_recalls[1].detach(),
        "v552r4_utility_harm_recall": utility_recalls[2].detach(),
        "v552r4_safety_queue_ce_loss": safety_queue_ce.detach(),
        "v552r4_utility_queue_ce_loss": utility_queue_ce.detach(),
        "v552r4_safety_queue_neutral_count": safety_queue_count[0].detach(),
        "v552r4_safety_queue_benefit_count": safety_queue_count[1].detach(),
        "v552r4_safety_queue_harm_count": safety_queue_count[2].detach(),
        "v552r4_utility_queue_neutral_count": utility_queue_count[0].detach(),
        "v552r4_utility_queue_benefit_count": utility_queue_count[1].detach(),
        "v552r4_utility_queue_harm_count": utility_queue_count[2].detach(),
        "v552r41_safety_queue_replay_fraction": safety_queue_used.detach(),
        "v552r41_utility_queue_replay_fraction": utility_queue_used.detach(),
        "v552r41_single_teacher_contract_enabled": zero.new_ones(()),
        "v552r41_gain_outcome_coupled": zero.new_tensor(0.0 if r42_enabled else 1.0),
        "v552r41_decision_score_gain_only": zero.new_ones(()),
        "v552r41_cardinality_unscaled": zero.new_ones(()),
        "v552r42_decoupled_critic_enabled": zero.new_tensor(
            1.0 if r42_enabled else 0.0
        ),
        "v552r42_task_specific_route_features_enabled": zero.new_tensor(
            1.0 if r42_enabled else 0.0
        ),
        "v552r43_spatial_route_evidence_enabled": aux.get(
            "v552r43_spatial_route_evidence_enabled", zero
        ).detach(),
        "v552r42_composer_critic_grad_isolated": zero.new_tensor(
            1.0 if r42_enabled else 0.0
        ),
        "v552r42_outcome_gain_grad_isolated": zero.new_tensor(
            1.0 if r42_enabled else 0.0
        ),
        "v552r42_pairwise_outcome_grad_isolated": zero.new_tensor(
            1.0 if r42_enabled else 0.0
        ),
        "v552r42_equal_class_mean_enabled": zero.new_tensor(
            1.0 if r42_enabled else 0.0
        ),
        "v552r42_quota_replay_enabled": zero.new_tensor(
            1.0 if r42_enabled else 0.0
        ),
        "v552r42_factorized_composer_loss_enabled": zero.new_tensor(
            1.0 if r42_enabled else 0.0
        ),
        "v552r42_safety_magnitude_loss": safety_magnitude_loss.detach(),
        "v552r42_safety_benefit_magnitude_loss": (
            safety_benefit_magnitude_loss.detach()
        ),
        "v552r42_safety_harm_magnitude_loss": (
            safety_harm_magnitude_loss.detach()
        ),
        "v552r42_utility_magnitude_loss": utility_magnitude_loss.detach(),
        "v552r42_utility_benefit_magnitude_loss": (
            utility_benefit_magnitude_loss.detach()
        ),
        "v552r42_utility_harm_magnitude_loss": (
            utility_harm_magnitude_loss.detach()
        ),
        "v552r42_safety_queue_magnitude_loss": (
            safety_queue_magnitude_loss.detach()
        ),
        "v552r42_utility_queue_magnitude_loss": (
            utility_queue_magnitude_loss.detach()
        ),
        "v552r42_safety_replay_neutral_count": (
            safety_replay_counts[0].to(base.dtype).detach()
        ),
        "v552r42_safety_replay_benefit_count": (
            safety_replay_counts[1].to(base.dtype).detach()
        ),
        "v552r42_safety_replay_harm_count": (
            safety_replay_counts[2].to(base.dtype).detach()
        ),
        "v552r42_utility_replay_neutral_count": (
            utility_replay_counts[0].to(base.dtype).detach()
        ),
        "v552r42_utility_replay_benefit_count": (
            utility_replay_counts[1].to(base.dtype).detach()
        ),
        "v552r42_utility_replay_harm_count": (
            utility_replay_counts[2].to(base.dtype).detach()
        ),
        "v552r42_safety_current_neutral_count": (
            safety_neutral.float().sum().detach()
        ),
        "v552r42_safety_current_benefit_count": (
            safety_benefit.float().sum().detach()
        ),
        "v552r42_safety_current_harm_count": (
            safety_harm.float().sum().detach()
        ),
        "v552r42_utility_current_neutral_count": (
            utility_neutral.float().sum().detach()
        ),
        "v552r42_utility_current_benefit_count": (
            utility_benefit.float().sum().detach()
        ),
        "v552r42_utility_current_harm_count": (
            utility_harm.float().sum().detach()
        ),
        "v552r42_safety_benefit_magnitude_mean": _masked_mean(
            safety_benefit_magnitude, safety_benefit
        ).detach(),
        "v552r42_safety_harm_magnitude_mean": _masked_mean(
            safety_harm_magnitude, safety_harm
        ).detach(),
        "v552r42_utility_benefit_magnitude_mean": _masked_mean(
            utility_benefit_magnitude, utility_benefit
        ).detach(),
        "v552r42_utility_harm_magnitude_mean": _masked_mean(
            utility_harm_magnitude, utility_harm
        ).detach(),
        "v552r4_teacher_executor_route_mismatch": teacher_contract_mismatch.detach(),
        "v552r4_student_editor_safe_rate": (
            predicted_editor_safe & valid
        ).float().sum().detach() / valid.float().sum().clamp_min(1.0),
        "v552r4_student_fallback_to_m1_rate": (
            (~predicted_editor_safe) & (route_prediction > 0) & valid
        ).float().sum().detach() / valid.float().sum().clamp_min(1.0),
        "v552r41_student_preserve_route_rate": (
            (route_prediction == 0) & valid
        ).float().sum().detach() / valid.float().sum().clamp_min(1.0),
        "v552r41_student_nonpreserve_proposal_rate": (
            (route_prediction > 0) & valid
        ).float().sum().detach() / valid.float().sum().clamp_min(1.0),
        "v552r41_student_nonpreserve_accept_rate": (
            (route_prediction > 0) & predicted_editor_safe & valid
        ).float().sum().detach() / valid.float().sum().clamp_min(1.0),
        "v552r41_student_nonpreserve_veto_rate": (
            (route_prediction > 0) & (~predicted_editor_safe) & valid
        ).float().sum().detach() / valid.float().sum().clamp_min(1.0),
        "v552_outcome_ce_loss": utility_ce.detach(),
        "v552_outcome_magnitude_loss": utility_gain_loss.detach(),
        "v552_outcome_signed_gain_loss": utility_gain_loss.detach(),
        "v552r3_benefit_sign_loss": utility_benefit_sign.detach(),
        "v552r3_harm_sign_loss": utility_harm_sign.detach(),
        "v552r3_neutral_sign_loss": utility_neutral_band.detach(),
        "v552r3_gain_unit": zero.new_tensor(float(gain_unit)),
        "v552_outcome_pairwise_loss": pairwise_loss.detach(),
        "v552_outcome_pair_accuracy": pairwise_accuracy.detach(),
        "v552_outcome_balanced_accuracy": torch.stack(
            utility_recalls
        ).mean().detach(),
        "v552_benefit_recall": utility_recalls[1].detach(),
        "v552_harm_recall": utility_recalls[2].detach(),
        "v552_gain_mean_on_benefit": _masked_mean(
            selected_utility_gain_pred, utility_benefit_selected
        ).detach(),
        "v552_gain_mean_on_harm": _masked_mean(
            selected_utility_gain_pred, utility_harm_selected
        ).detach(),
        "v552_composer_selection_loss": composer_selection_loss.detach(),
        "v552r42_composer_stop_loss": composer_stop_loss.detach(),
        "v552r42_composer_continue_loss": composer_continue_loss.detach(),
        "v552r42_composer_marginal_neutral_loss": (
            composer_marginal_neutral.detach()
        ),
        "v552r42_composer_marginal_benefit_loss": (
            composer_marginal_benefit.detach()
        ),
        "v552r42_composer_marginal_harm_loss": (
            composer_marginal_harm.detach()
        ),
        "v552_composer_marginal_gain_loss": composer_marginal_loss.detach(),
        "v552r44_composer_selective_risk": composer_selective_risk.detach(),
        "v552r44_composer_selective_coverage": composer_selective_coverage.detach(),
        "v552r44_composer_selective_positive_gain": (
            composer_selective_positive_gain.detach()
        ),
        "v552r44_composer_coverage_penalty": composer_coverage_penalty.detach(),
        "v552r44_composer_selective_risk_enabled": zero.new_tensor(
            1.0 if r44_enabled and bool(
                _m1(cfg, "V552R44_COMPOSER_SELECTIVE_RISK_ENABLED", True)
            ) else 0.0
        ),
        "v552_composer_final_bce_loss": composer_final_bce.detach(),
        "v552_composer_final_dice_loss": composer_final_dice.detach(),
        "v552_composer_selection_accuracy": (
            selection_correct / selection_count.clamp_min(1.0)
        ).detach(),
        "v552_composer_stop_accuracy": (
            stop_correct / stop_count.clamp_min(1.0)
        ).detach(),
        "v552_composer_marginal_gain_mae": (
            marginal_abs / marginal_count.clamp_min(1.0)
        ).detach(),
        "v552r3_composer_target_masked_rate": (
            target_masked_count / target_active_count.clamp_min(1.0)
        ).detach(),
        "v552r3_teacher_forced_target_mismatch": forced_target_mismatch.detach(),
        "v552_composer_teacher_step_count": teacher_steps.mean().detach(),
        "v552_composer_predicted_step_count": predicted_step_count.mean().detach(),
        "v552r3_ungated_student_step_count": ungated_step_count.mean().detach(),
        "v552r3_ungated_student_gain": ungated_gain_case.mean().detach(),
        "v552r3_ungated_student_execute_rate": (
            ungated_step_count > 0
        ).float().mean().detach(),
        "v552r3_ungated_student_harm_rate": (
            ungated_gain_case < -absolute_margin
        ).float().mean().detach(),
        "v552_composer_teacher_total_gain": teacher_final_gain.mean().detach(),
        "v552_composer_predicted_total_gain": ungated_gain_case.mean().detach(),
        "v552_composer_step1_gain": teacher["target_step_gains"][:, 0].mean().detach(),
        "v552_composer_step2_gain": (
            teacher["target_step_gains"][:, 1].mean().detach()
            if steps > 1 else zero
        ),
        "v552_composer_step3_gain": (
            teacher["target_step_gains"][:, 2].mean().detach()
            if steps > 2 else zero
        ),
        "v551_atom_count": valid.float().sum(dim=1).mean().detach(),
        "v551_atom_purity": _masked_mean(atom_quality_target, valid).detach(),
        "v551_atom_capture_ratio": capture.detach(),
        "v551_m1_atom_gain_mean": _masked_mean(m1_gain, valid).detach(),
        "v551_editor_atom_gain_mean": _masked_mean(
            teacher_safe_gain, valid
        ).detach(),
        "v551_m1_atom_oracle_gain": m1_oracle.detach(),
        "v551_editor_atom_oracle_gain": editor_oracle.detach(),
        "v551_harmful_editor_atom_rate": (
            (valid & (selected_incremental_gain_true < -relative_margin)).float().sum()
            / valid.float().sum().clamp_min(1.0)
        ).detach(),
        "v551_preserve_prediction_rate": (
            ((route_prediction == 0) & valid).float().sum()
            / valid.float().sum().clamp_min(1.0)
        ).detach(),
        "v552_safe_calibrated_execution_enabled": aux.get(
            "v552_safe_calibrated_execution_enabled", zero
        ).detach(),
        "v552_atom_quality_gate_active": aux.get(
            "v552_atom_quality_gate_active", zero
        ).detach(),
        "v552_atom_quality_score": aux.get(
            "v552_atom_quality_score", zero
        ).detach(),
        "v552_editor_strength": aux.get(
            "v552_editor_strength", zero
        ).detach(),
        "_v552r3_editor_objective": editor_objective,
        "_v552r3_outcome_objective": outcome_objective,
        "_v552r3_composer_objective": composer_objective,
        "_v552r42_safety_critic_objective": safety_critic_objective,
        "_v552r42_utility_critic_objective": utility_critic_objective,
    }
    return m1_extra, m2_extra, diagnostics


def _compute_v552r2_loss(
    *,
    cfg: Any,
    masks: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    epoch: int,
    base_probability: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """V552-R3 root-calibrated loss built on the R2 unified heads.

    The function keeps the R2 public entry point for compatibility, but adds
    strict Teacher-state supervision, normalized signed-Gain targets, typed
    Editor safety, useful/Null Atom supervision and three live objectives for
    independent routing in ``train.py``.
    """
    base = _as_b1hw(base_probability).detach().clamp(EPS, 1.0 - EPS)
    gt = (_as_b1hw(masks) >= 0.5).to(base.dtype)
    zero = base.sum() * 0.0
    required = (
        "v551_scale_logits",
        "v551_editor_route_logits",
        "v551_editor_dose_adjust",
        "v551_editor_region",
        "v551_local_residual",
        "v551_m1_exact_candidate_st",
        "v551_editor_exact_candidate_st",
        "v551_editor_logit_delta",
        "v545_slot_hard_masks",
        "v552_m1_supervision_valid",
        "v552_composer_teacher_valid",
        "v552_deployment_valid",
        "v552_atom_quality_logits",
        "v552_editor_relative_gain_pred",
        "v538_slot_presence_logits",
        "v532_action_candidate_probs",
        "v552_composer_step_logits",
        "v552_composer_step_candidate_scores",
        "v543_slot_outcome_logits",
        "v538_slot_gain_scores",
        "v552r2_benefit_magnitude",
        "v552r2_harm_magnitude",
    )
    missing = [key for key in required if key not in aux]
    if missing:
        raise KeyError(f"V552-R3 missing model outputs: {missing}")

    atom_masks = aux["v545_slot_hard_masks"].detach().clamp(0.0, 1.0)
    valid = aux["v552_m1_supervision_valid"].bool()
    teacher_valid = aux["v552_composer_teacher_valid"].bool()
    deployment_valid = aux["v552_deployment_valid"].bool()
    scale_logits = aux["v551_scale_logits"]
    route_logits = aux["v551_editor_route_logits"]
    editor_region = aux["v551_editor_region"].clamp(0.0, 1.0)
    local_residual = aux["v551_local_residual"]
    m1_candidate = aux["v551_m1_exact_candidate_st"].clamp(EPS, 1.0 - EPS)
    editor_candidate = aux["v551_editor_exact_candidate_st"].clamp(EPS, 1.0 - EPS)
    editor_delta = aux["v551_editor_logit_delta"]
    b, n, h, w = atom_masks.shape
    base_many = base[:, 0][:, None].expand(-1, n, -1, -1)
    base_dice = _hard_dice_many(base[:, 0][:, None], gt)[:, 0]
    m1_gain = _hard_dice_many(m1_candidate.detach(), gt) - base_dice[:, None]
    editor_gain = _hard_dice_many(editor_candidate.detach(), gt) - base_dice[:, None]
    relative_gain = editor_gain - m1_gain

    # ------------------------------------------------------------------
    # Typed Editor Teacher: Preserve or best realizable typed action.
    # ------------------------------------------------------------------
    margin = max(float(_m1(cfg, "V552_EDITOR_RELATIVE_MARGIN", 1.0e-4)), 0.0)
    action_candidates = aux["v532_action_candidate_probs"].detach().clamp(
        EPS, 1.0 - EPS
    )
    if action_candidates.shape[-2:] != (h, w):
        action_candidates = F.interpolate(
            action_candidates, size=(h, w), mode="bilinear", align_corners=False
        )
    action_gain = []
    for action in range(4):
        action_map = action_candidates[:, action][:, None].expand(-1, n, -1, -1)
        candidate = torch.where(atom_masks.bool(), action_map, base_many)
        action_gain.append(_hard_dice_many(candidate, gt) - base_dice[:, None])
    action_gain = torch.stack(action_gain, dim=2)
    best_action_gain, best_action = action_gain.max(dim=2)
    oracle_relative_gain = best_action_gain - m1_gain
    should_modify = valid & (oracle_relative_gain > margin)
    route_target = torch.where(
        should_modify, best_action + 1, torch.zeros_like(best_action)
    )
    route_per = F.cross_entropy(
        route_logits.reshape(b * n, 5),
        route_target.reshape(-1),
        reduction="none",
    ).reshape(b, n)
    positive_weight = max(
        float(_m1(cfg, "V551_EDITOR_POSITIVE_WEIGHT", 4.0)), 1.0
    )
    route_weight = torch.where(
        should_modify,
        route_per.new_full(route_per.shape, positive_weight),
        torch.ones_like(route_per),
    )
    route_mask_weight = route_weight * valid.to(route_per.dtype)
    route_loss = (route_per * route_mask_weight).sum() / route_mask_weight.sum().clamp_min(1.0)

    preserve_logit = route_logits[:, :, 0]
    edit_logit = torch.logsumexp(route_logits[:, :, 1:], dim=2)
    modify_probability = torch.sigmoid(edit_logit - preserve_logit)
    editability_loss = _masked_mean(
        F.binary_cross_entropy_with_logits(
            edit_logit - preserve_logit,
            should_modify.to(route_logits.dtype),
            reduction="none",
        ),
        valid,
    )
    relative_pred = aux["v552_editor_relative_gain_pred"]
    relative_gain_loss = _masked_mean(
        F.smooth_l1_loss(
            relative_pred,
            oracle_relative_gain.detach(),
            beta=1.0e-3,
            reduction="none",
        ),
        valid,
    )
    editor_damage = (m1_gain - editor_gain).clamp_min(0.0).detach()
    harm_veto_loss = _masked_mean(modify_probability * editor_damage, valid)

    area_fraction = atom_masks.mean(dim=(-2, -1))
    scale_target = _area_scale_target(
        area_fraction,
        _m1(cfg, "V551_SCALE_AREA_THRESHOLDS", [0.001, 0.006, 0.02]),
        int(scale_logits.shape[2]),
    )
    scale_per = F.cross_entropy(
        scale_logits.reshape(b * n, -1),
        scale_target.reshape(-1),
        reduction="none",
    ).reshape(b, n)
    scale_loss = _masked_mean(scale_per, valid)

    action_bank = action_candidates[:, None].expand(-1, n, -1, -1, -1)
    best_action_map = action_bank.gather(
        2,
        best_action[:, :, None, None, None].expand(-1, -1, 1, h, w),
    ).squeeze(2)
    best_action_candidate = torch.where(
        atom_masks.bool(), best_action_map, base_many
    ).detach()
    local_target = torch.where(
        should_modify[:, :, None, None],
        best_action_candidate,
        m1_candidate.detach(),
    )
    local_bce = F.binary_cross_entropy(
        editor_candidate, local_target, reduction="none"
    )
    local_bce = (
        (local_bce * editor_region).flatten(2).sum(dim=2)
        / editor_region.flatten(2).sum(dim=2).clamp_min(1.0)
    )
    local_bce_loss = _masked_mean(local_bce, valid)
    local_dice_loss = _soft_dice_loss(
        editor_candidate, local_target, editor_region, valid
    )
    outside = 1.0 - editor_region
    outside_preserve = (
        ((editor_candidate - m1_candidate.detach()).abs() * outside)
        .flatten(2)
        .sum(dim=2)
        / outside.flatten(2).sum(dim=2).clamp_min(1.0)
    )
    outside_preserve_loss = _masked_mean(outside_preserve, valid)
    residual_l1 = _masked_mean(
        local_residual.abs().flatten(2).mean(dim=2), valid
    )

    # Protect already-good M1 atoms even when the average Editor improves.
    m1_soft_dice = _dice_many(m1_candidate.detach(), gt)
    editor_soft_dice = _dice_many(editor_candidate, gt)
    safety_margin = max(
        float(_m1(cfg, "V552_EDITOR_RELATIVE_SAFETY_MARGIN", 5.0e-4)), 0.0
    )
    relative_safety_per = F.relu(
        m1_soft_dice.detach() - editor_soft_dice + safety_margin
    )
    protect_weight = torch.where(
        m1_gain > 1.0e-4,
        relative_safety_per.new_full(relative_safety_per.shape, 4.0),
        torch.ones_like(relative_safety_per),
    )
    safety_weight = protect_weight * valid.to(relative_safety_per.dtype)
    relative_safety_loss = (
        relative_safety_per * safety_weight
    ).sum() / safety_weight.sum().clamp_min(1.0)

    # ------------------------------------------------------------------
    # Useful/Null Atom supervision.
    # ------------------------------------------------------------------
    error = ((base >= 0.5) != (gt >= 0.5))[:, 0]
    correct_edit_mass = (
        atom_masks.bool() & error[:, None]
    ).flatten(2).sum(dim=2).to(base.dtype)
    atom_area_pixels = atom_masks.flatten(2).sum(dim=2).clamp_min(1.0)
    atom_quality_target = (correct_edit_mass / atom_area_pixels).detach()
    atom_quality_probs = torch.sigmoid(aux["v552_atom_quality_logits"])
    atom_quality_loss = _masked_mean(
        F.smooth_l1_loss(
            atom_quality_probs,
            atom_quality_target,
            beta=0.10,
            reduction="none",
        ),
        valid,
    )
    presence_logits = aux["v538_slot_presence_logits"]
    useful_quality = max(
        float(_m1(cfg, "V552R3_USEFUL_ATOM_MIN_PURITY", 0.15)), 0.0
    )
    useful_gain = max(
        float(_m1(cfg, "V552R3_USEFUL_ATOM_MIN_GAIN", 1.0e-4)), 0.0
    )
    useful_atom_target = valid & (
        (best_action_gain > useful_gain)
        | (atom_quality_target >= useful_quality)
    )
    presence_element = F.binary_cross_entropy_with_logits(
        presence_logits,
        useful_atom_target.to(presence_logits.dtype),
        reduction="none",
    )
    useful_positive_weight = max(
        float(_m1(cfg, "V552R3_USEFUL_ATOM_POSITIVE_WEIGHT", 2.0)), 1.0
    )
    presence_weight = torch.where(
        useful_atom_target,
        presence_element.new_full(
            presence_element.shape, useful_positive_weight
        ),
        torch.ones_like(presence_element),
    )
    useful_presence_loss = (
        presence_element * presence_weight
    ).sum() / presence_weight.sum().clamp_min(1.0)
    predicted_presence = torch.sigmoid(presence_logits)
    max_useful_atoms = max(int(_m1(cfg, "V552R3_MAX_USEFUL_ATOMS", 4)), 1)
    target_cardinality = useful_atom_target.float().sum(dim=1).clamp(
        max=float(max_useful_atoms)
    )
    predicted_cardinality = predicted_presence.sum(dim=1)
    cardinality_loss = F.smooth_l1_loss(
        predicted_cardinality / float(max(n, 1)),
        target_cardinality / float(max(n, 1)),
        beta=0.10,
    )
    null_mask = ~useful_atom_target
    false_atom_loss = (
        predicted_presence
        * (1.0 - atom_quality_target)
        * null_mask.to(base.dtype)
    ).sum() / null_mask.float().sum().clamp_min(1.0)

    # ------------------------------------------------------------------
    # Unified three-class Outcome + normalized signed Gain.
    # ------------------------------------------------------------------
    epsilon = max(float(_m1(cfg, "V551_BENEFIT_EPSILON", 1.0e-4)), 0.0)
    outcome_target = torch.zeros_like(editor_gain, dtype=torch.long)
    outcome_target = torch.where(
        editor_gain > epsilon, torch.ones_like(outcome_target), outcome_target
    )
    outcome_target = torch.where(
        editor_gain < -epsilon,
        torch.full_like(outcome_target, 2),
        outcome_target,
    )
    outcome_logits = aux["v543_slot_outcome_logits"]
    outcome_ce = _balanced_outcome_ce(outcome_logits, outcome_target, valid)
    benefit_mask = valid & (outcome_target == 1)
    harm_mask = valid & (outcome_target == 2)
    neutral_mask = valid & (outcome_target == 0)

    gain_scale = max(float(_m1(cfg, "V541_GAIN_SCALE", 1000.0)), 1.0)
    gain_unit = max(float(_m1(cfg, "V552R3_GAIN_UNIT", 0.01)), 1.0e-5)
    gain_target_normalized = (editor_gain.detach() / gain_unit).clamp(-1.0, 1.0)
    magnitude_target_normalized = gain_target_normalized.abs()
    benefit_magnitude = aux["v552r2_benefit_magnitude"] / (
        gain_scale * gain_unit
    )
    harm_magnitude = aux["v552r2_harm_magnitude"] / (
        gain_scale * gain_unit
    )
    benefit_mag_loss = _masked_mean(
        F.smooth_l1_loss(
            benefit_magnitude,
            magnitude_target_normalized,
            beta=0.10,
            reduction="none",
        ),
        benefit_mask,
    )
    harm_mag_loss = _masked_mean(
        F.smooth_l1_loss(
            harm_magnitude,
            magnitude_target_normalized,
            beta=0.10,
            reduction="none",
        ),
        harm_mask,
    )
    magnitude_loss = benefit_mag_loss + harm_mag_loss
    signed_gain_pred = aux["v538_slot_gain_scores"]
    signed_gain_pred_normalized = signed_gain_pred / gain_unit
    signed_gain_loss = _masked_mean(
        F.smooth_l1_loss(
            signed_gain_pred_normalized,
            gain_target_normalized,
            beta=0.10,
            reduction="none",
        ),
        valid,
    )
    benefit_sign_margin = max(
        float(_m1(cfg, "V552R3_BENEFIT_SIGN_MARGIN", 0.10)), 0.0
    )
    harm_sign_margin = max(
        float(_m1(cfg, "V552R3_HARM_SIGN_MARGIN", 0.10)), 0.0
    )
    neutral_band = max(
        float(_m1(cfg, "V552R3_NEUTRAL_GAIN_BAND", 0.05)), 0.0
    )
    benefit_sign_loss = _masked_mean(
        F.relu(benefit_sign_margin - signed_gain_pred_normalized),
        benefit_mask,
    )
    harm_sign_loss = _masked_mean(
        F.relu(harm_sign_margin + signed_gain_pred_normalized),
        harm_mask,
    )
    neutral_sign_loss = _masked_mean(
        F.relu(signed_gain_pred_normalized.abs() - neutral_band),
        neutral_mask,
    )

    pairwise_loss = zero
    pairwise_accuracy = zero.new_zeros(())
    pairwise_scale = _curriculum(
        epoch,
        int(_m1(cfg, "V552_PAIRWISE_START_EPOCH", 15)),
        int(_m1(cfg, "V552_PAIRWISE_RAMP_EPOCHS", 5)),
    )
    if n > 1 and pairwise_scale > 0.0:
        pair_valid = valid[:, :-1] & valid[:, 1:]
        true_diff = editor_gain[:, :-1] - editor_gain[:, 1:]
        pred_diff = (
            signed_gain_pred_normalized[:, :-1]
            - signed_gain_pred_normalized[:, 1:]
        )
        non_tie = pair_valid & (true_diff.abs() > epsilon)
        if bool(non_tie.any().item()):
            pairwise_loss = F.softplus(
                -torch.sign(true_diff[non_tie]) * pred_diff[non_tie]
            ).mean()
            pairwise_accuracy = (
                (pred_diff[non_tie] * true_diff[non_tie]) > 0
            ).float().mean().detach()

    # ------------------------------------------------------------------
    # Exact Composer Teacher aligned with strict Teacher-forced model states.
    # ------------------------------------------------------------------
    teacher_pool_size = max(
        int(_m1(cfg, "V552_COMPOSER_TEACHER_POOL_SIZE", 4)), 1
    )
    if teacher_pool_size < n:
        teacher_score = atom_quality_target + 0.25 * (
            correct_edit_mass > 0
        ).to(base.dtype)
        teacher_score = teacher_score.masked_fill(~teacher_valid, -1.0e4)
        indices = teacher_score.topk(teacher_pool_size, dim=1).indices
        pool_mask = torch.zeros_like(teacher_valid)
        pool_mask.scatter_(1, indices, True)
        teacher_valid = teacher_valid & pool_mask

    teacher = build_v552r2_exact_composer_teacher(
        base=base,
        editor_logit_delta=editor_delta,
        masks=aux.get("v552_composer_support", atom_masks),
        valid=teacher_valid,
        gt=gt,
        max_steps=int(_m1(cfg, "V538_COMPOSER_MAX_STEPS", 3)),
        stop_margin=float(
            _m1(cfg, "V552_COMPOSER_TEACHER_STOP_MARGIN", 0.0)
        ),
        max_overlap=float(_m1(cfg, "V538_COMPOSER_MAX_OVERLAP", 0.20)),
        max_total_edit_fraction=float(
            _m1(cfg, "V538_COMPOSER_MAX_EDIT_FRACTION", 0.35)
        ),
    )
    step_logits = aux["v552_composer_step_logits"]
    step_scores = aux["v552_composer_step_candidate_scores"]
    steps = min(step_logits.shape[1], teacher["target_indices"].shape[1])
    ce_terms = []
    marginal_terms = []
    # Every mutable counter owns independent storage.  Initializing all from
    # the same ``zero.detach()`` aliases them and makes ratios exceed one.
    selection_correct = zero.new_zeros(())
    selection_count = zero.new_zeros(())
    stop_correct = zero.new_zeros(())
    stop_count = zero.new_zeros(())
    marginal_abs = zero.new_zeros(())
    marginal_count = zero.new_zeros(())
    target_masked_count = zero.new_zeros(())
    target_active_count = zero.new_zeros(())
    for step in range(steps):
        active = teacher["target_active"][:, step]
        target = teacher["target_indices"][:, step]
        teacher_eligible = teacher["target_eligible"][:, step]
        candidate_teacher_logits = step_scores[:, step].masked_fill(
            ~teacher_eligible, -1.0e4
        )
        teacher_logits = torch.cat(
            [candidate_teacher_logits, step_logits[:, step, -1:]], dim=1
        )
        if bool(active.any().item()):
            original_target_logit = step_logits[:, step].gather(
                1, target[:, None]
            )[:, 0]
            target_masked_count = target_masked_count + (
                active & (original_target_logit < -1.0e3)
            ).float().sum()
            target_active_count = target_active_count + active.float().sum()
            ce_terms.append(
                F.cross_entropy(teacher_logits[active], target[active])
            )
            prediction = teacher_logits.detach().argmax(dim=1)
            selection_correct = selection_correct + (
                (prediction == target) & active
            ).float().sum()
            selection_count = selection_count + active.float().sum()
            is_stop = active & (target == n)
            stop_correct = stop_correct + (
                (prediction == n) & is_stop
            ).float().sum()
            stop_count = stop_count + is_stop.float().sum()
        eligible = teacher_eligible
        if bool(eligible.any().item()):
            exact_marginal = teacher["target_marginal_gains"][:, step]
            pred_normalized = step_scores[:, step] / gain_unit
            target_normalized = (exact_marginal / gain_unit).clamp(-1.0, 1.0)
            marginal_terms.append(
                F.smooth_l1_loss(
                    pred_normalized[eligible],
                    target_normalized[eligible],
                    beta=0.10,
                )
            )
            marginal_abs = marginal_abs + (
                step_scores[:, step].detach()[eligible]
                - exact_marginal[eligible]
            ).abs().sum()
            marginal_count = marginal_count + eligible.float().sum()
    composer_selection_loss = (
        torch.stack(ce_terms).mean() if ce_terms else zero
    )
    composer_marginal_loss = (
        torch.stack(marginal_terms).mean() if marginal_terms else zero
    )
    student_final = _v552r2_student_soft_final(
        base=base,
        step_logits=step_logits[:, :steps],
        editor_logit_delta=editor_delta,
    )
    composer_final_bce = F.binary_cross_entropy(student_final, gt)
    intersection = (student_final * gt).flatten(1).sum(dim=1)
    denominator = (
        student_final.flatten(1).sum(dim=1)
        + gt.flatten(1).sum(dim=1)
    )
    composer_final_dice = (
        1.0 - (2.0 * intersection + EPS) / (denominator + EPS)
    ).mean()

    editor_scale = _curriculum(
        epoch,
        int(_m1(cfg, "V551_EDITOR_START_EPOCH", 6)),
        int(_m1(cfg, "V551_EDITOR_RAMP_EPOCHS", 8)),
    )
    outcome_scale = _curriculum(
        epoch,
        int(_m1(cfg, "V552_OUTCOME_START_EPOCH", 2)),
        int(_m1(cfg, "V552_OUTCOME_RAMP_EPOCHS", 8)),
    )
    composer_scale = _curriculum(
        epoch,
        int(_m1(cfg, "V552_COMPOSER_START_EPOCH", 10)),
        int(_m1(cfg, "V552_COMPOSER_RAMP_EPOCHS", 8)),
    )

    editor_objective = (
        float(_m1(cfg, "V551_ROUTE_LOSS_WEIGHT", 1.0)) * route_loss
        + float(_m1(cfg, "V551_EDITABILITY_LOSS_WEIGHT", 1.0))
        * editability_loss
        + float(_m1(cfg, "V552_EDITOR_RELATIVE_GAIN_WEIGHT", 1.0))
        * relative_gain_loss
        + float(_m1(cfg, "V551_LOCAL_BCE_WEIGHT", 0.50))
        * local_bce_loss
        + float(_m1(cfg, "V551_LOCAL_DICE_WEIGHT", 0.50))
        * local_dice_loss
        + float(_m1(cfg, "V551_OUTSIDE_PRESERVE_WEIGHT", 1.0))
        * outside_preserve_loss
        + float(_m1(cfg, "V551_LOCAL_RESIDUAL_L1_WEIGHT", 0.30))
        * residual_l1
        + float(_m1(cfg, "V551_HARM_VETO_WEIGHT", 6.0))
        * harm_veto_loss
        + float(_m1(cfg, "V552_EDITOR_RELATIVE_SAFETY_WEIGHT", 4.0))
        * relative_safety_loss
    )
    outcome_objective = (
        float(_m1(cfg, "V552_OUTCOME_CE_WEIGHT", 2.0)) * outcome_ce
        + float(_m1(cfg, "V552_MAGNITUDE_WEIGHT", 1.0)) * magnitude_loss
        + float(_m1(cfg, "V552_SIGNED_GAIN_WEIGHT", 1.0))
        * signed_gain_loss
        + float(_m1(cfg, "V552R3_BENEFIT_SIGN_WEIGHT", 2.0))
        * benefit_sign_loss
        + float(_m1(cfg, "V552R3_HARM_SIGN_WEIGHT", 2.0))
        * harm_sign_loss
        + float(_m1(cfg, "V552R3_NEUTRAL_SIGN_WEIGHT", 0.5))
        * neutral_sign_loss
        + float(_m1(cfg, "V552_PAIRWISE_WEIGHT", 0.10))
        * float(pairwise_scale)
        * pairwise_loss
    )
    composer_objective = (
        float(_m1(cfg, "V552_COMPOSER_SELECTION_WEIGHT", 1.0))
        * composer_selection_loss
        + float(_m1(cfg, "V552_COMPOSER_MARGINAL_GAIN_WEIGHT", 1.0))
        * composer_marginal_loss
        + float(_m1(cfg, "V552_COMPOSER_FINAL_BCE_WEIGHT", 0.10))
        * composer_final_bce
        + float(_m1(cfg, "V552_COMPOSER_FINAL_DICE_WEIGHT", 0.10))
        * composer_final_dice
    )
    m1_extra = (
        float(_m1(cfg, "V551_SCALE_LOSS_WEIGHT", 0.50)) * scale_loss
        + float(_m1(cfg, "V552_ATOM_QUALITY_WEIGHT", 0.50))
        * atom_quality_loss
        + float(_m1(cfg, "V552R3_USEFUL_ATOM_PRESENCE_WEIGHT", 1.0))
        * useful_presence_loss
        + float(_m1(cfg, "V552R3_CARDINALITY_WEIGHT", 0.25))
        * cardinality_loss
        + float(_m1(cfg, "V552R3_FALSE_ATOM_WEIGHT", 0.25))
        * false_atom_loss
    )
    # Compatibility objective for old routing.  V552-R3 consumes the live
    # unscaled branch objectives below and applies each curriculum exactly once.
    m2_extra = (
        editor_objective * base.new_tensor(editor_scale)
        + outcome_objective * base.new_tensor(outcome_scale)
        + composer_objective * base.new_tensor(composer_scale)
    )

    route_prediction = route_logits.argmax(dim=2)
    outcome_prediction = outcome_logits.argmax(dim=2)
    teacher_steps = (teacher["target_indices"] < n).float().sum(dim=1)
    teacher_final_gain = (
        _hard_dice_many(
            teacher["teacher_final_probs"][:, 0][:, None], gt
        )[:, 0]
        - base_dice
    )
    predicted_step_count = aux.get(
        "v552_composer_predicted_step_count", zero.new_zeros((b,))
    ).float()
    ungated_step_count = aux.get(
        "v552_composer_ungated_step_count", zero.new_zeros((b,))
    ).float()
    ungated_final = aux.get("v552_composer_ungated_final_probability")
    if isinstance(ungated_final, torch.Tensor):
        ungated_gain_case = (
            _hard_dice_many(ungated_final.detach()[:, 0][:, None], gt)[:, 0]
            - base_dice
        )
    else:
        ungated_gain_case = torch.zeros_like(base_dice)
    forced_target = aux.get("v552_composer_forced_target_indices")
    if (
        isinstance(forced_target, torch.Tensor)
        and forced_target[:, :steps].shape
        == teacher["target_indices"][:, :steps].shape
    ):
        forced_target_mismatch = (
            forced_target[:, :steps].detach()
            != teacher["target_indices"][:, :steps]
        ).float().mean()
    else:
        forced_target_mismatch = zero.new_zeros(())

    error_pixels = error.flatten(1).sum(dim=1).clamp_min(1.0)
    positive_union = (
        atom_masks.bool() & should_modify[:, :, None, None]
    ).any(dim=1)
    capture = (
        (positive_union & error).flatten(1).sum(dim=1).to(base.dtype)
        / error_pixels
    ).mean()
    m1_oracle = torch.cat(
        [
            torch.zeros_like(base_dice[:, None]),
            m1_gain.masked_fill(~valid, -1.0e4),
        ],
        dim=1,
    ).max(dim=1).values.mean()
    editor_oracle = torch.cat(
        [
            torch.zeros_like(base_dice[:, None]),
            editor_gain.masked_fill(~valid, -1.0e4),
        ],
        dim=1,
    ).max(dim=1).values.mean()

    class_recalls = []
    for cls in range(3):
        cls_mask = valid & (outcome_target == cls)
        class_recalls.append(
            (
                ((outcome_prediction == cls) & cls_mask).float().sum()
                / cls_mask.float().sum().clamp_min(1.0)
            )
        )
    diagnostics = {
        "v551_enabled": zero.new_ones(()),
        "v552r2_enabled": zero.new_ones(()),
        "v551_scale_loss": scale_loss.detach(),
        "v552_atom_quality_loss": atom_quality_loss.detach(),
        "v552_atom_quality_target_mean": _masked_mean(
            atom_quality_target, valid
        ).detach(),
        "v552r3_useful_atom_presence_loss": useful_presence_loss.detach(),
        "v552r3_cardinality_loss": cardinality_loss.detach(),
        "v552r3_false_atom_loss": false_atom_loss.detach(),
        "v552r3_useful_atom_count": useful_atom_target.float().sum(
            dim=1
        ).mean().detach(),
        "v552r3_predicted_active_atom_count": (
            predicted_presence >= 0.5
        ).float().sum(dim=1).mean().detach(),
        "v551_route_loss": route_loss.detach(),
        "v551_editability_loss": editability_loss.detach(),
        "v552_editor_relative_gain_loss": relative_gain_loss.detach(),
        "v552_editor_relative_gain_mean": _masked_mean(
            relative_gain, valid
        ).detach(),
        "v552_editor_teacher_relative_gain_mean": _masked_mean(
            oracle_relative_gain, valid
        ).detach(),
        "v552_editor_relative_safety_loss": relative_safety_loss.detach(),
        "v552_editor_should_modify_rate": (
            should_modify.float().sum() / valid.float().sum().clamp_min(1.0)
        ).detach(),
        "v551_harm_veto_loss": harm_veto_loss.detach(),
        "v551_local_bce_loss": local_bce_loss.detach(),
        "v551_local_dice_loss": local_dice_loss.detach(),
        "v551_outside_preserve_loss": outside_preserve_loss.detach(),
        "v551_local_residual_l1": residual_l1.detach(),
        "v551_editor_train_scale": zero.new_tensor(float(editor_scale)),
        "v552_outcome_train_scale": zero.new_tensor(float(outcome_scale)),
        "v552_composer_train_scale": zero.new_tensor(float(composer_scale)),
        "v552_outcome_ce_loss": outcome_ce.detach(),
        "v552_outcome_magnitude_loss": magnitude_loss.detach(),
        "v552_outcome_signed_gain_loss": signed_gain_loss.detach(),
        "v552r3_benefit_sign_loss": benefit_sign_loss.detach(),
        "v552r3_harm_sign_loss": harm_sign_loss.detach(),
        "v552r3_neutral_sign_loss": neutral_sign_loss.detach(),
        "v552r3_gain_unit": zero.new_tensor(float(gain_unit)),
        "v552_outcome_pairwise_loss": pairwise_loss.detach(),
        "v552_outcome_pair_accuracy": pairwise_accuracy.detach(),
        "v552_outcome_balanced_accuracy": torch.stack(
            class_recalls
        ).mean().detach(),
        "v552_benefit_recall": class_recalls[1].detach(),
        "v552_harm_recall": class_recalls[2].detach(),
        "v552_gain_mean_on_benefit": _masked_mean(
            signed_gain_pred, benefit_mask
        ).detach(),
        "v552_gain_mean_on_harm": _masked_mean(
            signed_gain_pred, harm_mask
        ).detach(),
        "v552_composer_selection_loss": composer_selection_loss.detach(),
        "v552_composer_marginal_gain_loss": composer_marginal_loss.detach(),
        "v552_composer_final_bce_loss": composer_final_bce.detach(),
        "v552_composer_final_dice_loss": composer_final_dice.detach(),
        "v552_composer_selection_accuracy": (
            selection_correct / selection_count.clamp_min(1.0)
        ).detach(),
        "v552_composer_stop_accuracy": (
            stop_correct / stop_count.clamp_min(1.0)
        ).detach(),
        "v552_composer_marginal_gain_mae": (
            marginal_abs / marginal_count.clamp_min(1.0)
        ).detach(),
        "v552r3_composer_target_masked_rate": (
            target_masked_count / target_active_count.clamp_min(1.0)
        ).detach(),
        "v552r3_teacher_forced_target_mismatch": forced_target_mismatch.detach(),
        "v552_composer_teacher_step_count": teacher_steps.mean().detach(),
        "v552_composer_predicted_step_count": predicted_step_count.mean().detach(),
        "v552r3_ungated_student_step_count": ungated_step_count.mean().detach(),
        "v552r3_ungated_student_gain": ungated_gain_case.mean().detach(),
        "v552r3_ungated_student_execute_rate": (
            ungated_step_count > 0
        ).float().mean().detach(),
        "v552r3_ungated_student_harm_rate": (
            ungated_gain_case < -epsilon
        ).float().mean().detach(),
        "v552_composer_teacher_total_gain": teacher_final_gain.mean().detach(),
        "v552_composer_predicted_total_gain": ungated_gain_case.mean().detach(),
        "v552_composer_step1_gain": teacher["target_step_gains"][:, 0]
        .mean()
        .detach(),
        "v552_composer_step2_gain": (
            teacher["target_step_gains"][:, 1].mean().detach()
            if steps > 1 else zero.new_zeros(())
        ),
        "v552_composer_step3_gain": (
            teacher["target_step_gains"][:, 2].mean().detach()
            if steps > 2 else zero.new_zeros(())
        ),
        "v552_composer_teacher_pool_count": teacher_valid.float().sum(
            dim=1
        ).mean().detach(),
        "v552_composer_deploy_pool_count": deployment_valid.float().sum(
            dim=1
        ).mean().detach(),
        "v552_composer_conflict_reject_rate": aux.get(
            "v552_composer_conflict_reject_rate", zero
        ).detach(),
        "v552_composer_budget_reject_rate": aux.get(
            "v552_composer_budget_reject_rate", zero
        ).detach(),
        "v551_atom_count": valid.float().sum(dim=1).mean().detach(),
        "v551_atom_purity": _masked_mean(atom_quality_target, valid).detach(),
        "v551_atom_capture_ratio": capture.detach(),
        "v551_m1_atom_gain_mean": _masked_mean(m1_gain, valid).detach(),
        "v551_editor_atom_gain_mean": _masked_mean(editor_gain, valid).detach(),
        "v551_m1_atom_oracle_gain": m1_oracle.detach(),
        "v551_editor_atom_oracle_gain": editor_oracle.detach(),
        "v551_harmful_editor_atom_rate": (
            (valid & (relative_gain < -epsilon)).float().sum()
            / valid.float().sum().clamp_min(1.0)
        ).detach(),
        "v551_preserve_prediction_rate": (
            ((route_prediction == 0) & valid).float().sum()
            / valid.float().sum().clamp_min(1.0)
        ).detach(),
        "v552_safe_calibrated_execution_enabled": aux.get(
            "v552_safe_calibrated_execution_enabled", zero
        ).detach(),
        "v552_atom_quality_gate_active": aux.get(
            "v552_atom_quality_gate_active", zero
        ).detach(),
        "v552_atom_quality_score": aux.get(
            "v552_atom_quality_score", zero
        ).detach(),
        "v552_editor_strength": aux.get(
            "v552_editor_strength", zero
        ).detach(),
        # Live tensors consumed only by train.py's independent R3 router.
        "_v552r3_editor_objective": editor_objective,
        "_v552r3_outcome_objective": outcome_objective,
        "_v552r3_composer_objective": composer_objective,
    }
    return m1_extra, m2_extra, diagnostics

def compute_v551_multiscale_editor_loss(
    *,
    cfg: Any,
    masks: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    epoch: int,
    base_probability: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Return extra M1 loss, extra M2-editor loss, and diagnostics."""
    base = _as_b1hw(base_probability).detach().clamp(EPS, 1.0 - EPS)
    zero = base.sum() * 0.0
    if not bool(_m1(cfg, "V551_MULTISCALE_TYPED_EDITOR_ENABLED", False)):
        return zero, zero, {"v551_enabled": zero.detach()}
    if bool(_m1(cfg, "V552R4_UNIFIED_REFERENCE_CONTRACT_ENABLED", False)):
        return _compute_v552r4_loss(
            cfg=cfg,
            masks=masks,
            aux=aux,
            epoch=epoch,
            base_probability=base_probability,
        )
    if bool(_m1(cfg, "V552R2_TEACHER_DECOUPLED_ENABLED", False)):
        return _compute_v552r2_loss(
            cfg=cfg,
            masks=masks,
            aux=aux,
            epoch=epoch,
            base_probability=base_probability,
        )

    required = (
        "v551_scale_logits",
        "v551_editor_route_logits",
        "v551_editor_dose_adjust",
        "v551_editor_region",
        "v551_local_residual",
        "v551_m1_exact_candidate_st",
        "v551_editor_exact_candidate_st",
        "v545_slot_hard_masks",
        "v538_slot_valid",
        "v532_action_candidate_probs",
    )
    missing = [key for key in required if key not in aux]
    if missing:
        raise KeyError(f"V551 missing model outputs: {missing}")

    gt = (_as_b1hw(masks) >= 0.5).to(base.dtype)
    composer_enabled = bool(
        _m1(cfg, "V552_MULTICANDIDATE_COMPOSER_ENABLED", False)
    )
    composer_required = (
        "v552_composer_step_logits",
        "v552_composer_step_candidate_scores",
        "v552_composer_step_state_probs",
        "v552_composer_step_eligible",
        "v552_composer_step_active",
        "v551_editor_logit_delta",
    )
    if composer_enabled:
        composer_missing = [key for key in composer_required if key not in aux]
        if composer_missing:
            raise KeyError(f"V552-R1 missing composer outputs: {composer_missing}")
    atom_masks = aux["v545_slot_hard_masks"].detach().clamp(0.0, 1.0)
    atom_valid = aux["v538_slot_valid"].bool()
    scale_logits = aux["v551_scale_logits"]
    route_logits = aux["v551_editor_route_logits"]
    dose_adjust = aux["v551_editor_dose_adjust"]
    editor_region = aux["v551_editor_region"].clamp(0.0, 1.0)
    local_residual = aux["v551_local_residual"]
    m1_candidate = aux["v551_m1_exact_candidate_st"].clamp(EPS, 1.0 - EPS)
    editor_candidate = aux["v551_editor_exact_candidate_st"].clamp(EPS, 1.0 - EPS)
    action_candidates = aux["v532_action_candidate_probs"].detach().clamp(
        EPS, 1.0 - EPS
    )
    if action_candidates.shape[-2:] != atom_masks.shape[-2:]:
        action_candidates = F.interpolate(
            action_candidates,
            size=atom_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    b, n, h, w = atom_masks.shape
    if scale_logits.shape[:2] != (b, n):
        raise ValueError(
            f"V551 scale logits {tuple(scale_logits.shape)} incompatible with atoms {(b, n)}"
        )

    base_many = base[:, 0][:, None].expand(-1, n, -1, -1)
    gt_many = gt.expand(-1, n, -1, -1)
    hard_base_dice = _hard_dice_many(base[:, 0][:, None], gt)[:, 0]

    # ------------------------------------------------------------------
    # Exact local action teacher: Preserve + four typed interventions.
    # ------------------------------------------------------------------
    action_gains = []
    for action in range(4):
        action_map = action_candidates[:, action][:, None].expand(-1, n, -1, -1)
        candidate = torch.where(atom_masks.bool(), action_map, base_many)
        gain = _hard_dice_many(candidate, gt) - hard_base_dice[:, None]
        action_gains.append(gain)
    action_gain = torch.stack(action_gains, dim=2)  # [B,N,4]
    best_action_gain, best_action = action_gain.max(dim=2)
    gain_epsilon = max(float(_m1(cfg, "V551_BENEFIT_EPSILON", 1.0e-4)), 0.0)
    harmful_epsilon = max(float(_m1(cfg, "V551_HARM_EPSILON", gain_epsilon)), 0.0)
    benefit_target = atom_valid & (best_action_gain > gain_epsilon)
    route_target = torch.where(
        benefit_target,
        best_action + 1,
        torch.zeros_like(best_action),
    )

    route_element = F.cross_entropy(
        route_logits.reshape(b * n, 5),
        route_target.reshape(b * n),
        reduction="none",
    ).reshape(b, n)
    positive_weight = max(float(_m1(cfg, "V551_EDITOR_POSITIVE_WEIGHT", 4.0)), 1.0)
    route_weight = torch.where(
        benefit_target,
        route_element.new_full(route_element.shape, positive_weight),
        torch.ones_like(route_element),
    )
    route_loss = (
        (route_element * route_weight * atom_valid.to(route_element.dtype)).sum()
        / (route_weight * atom_valid.to(route_element.dtype)).sum().clamp_min(1.0)
    )

    preserve_logit = route_logits[:, :, 0]
    edit_logit = torch.logsumexp(route_logits[:, :, 1:], dim=2)
    editability_logit = edit_logit - preserve_logit
    editability_element = F.binary_cross_entropy_with_logits(
        editability_logit,
        benefit_target.to(editability_logit.dtype),
        reduction="none",
    )
    editability_loss = (
        editability_element * atom_valid.to(editability_element.dtype)
    ).sum() / atom_valid.float().sum().clamp_min(1.0)

    # ------------------------------------------------------------------
    # Dynamic scale teacher from executable atom area.
    # ------------------------------------------------------------------
    area_fraction = atom_masks.mean(dim=(-2, -1))
    thresholds = _m1(cfg, "V551_SCALE_AREA_THRESHOLDS", [0.001, 0.006, 0.02])
    scale_target = _area_scale_target(
        area_fraction, thresholds, int(scale_logits.shape[2])
    )
    scale_element = F.cross_entropy(
        scale_logits.reshape(b * n, -1),
        scale_target.reshape(b * n),
        reduction="none",
    ).reshape(b, n)
    scale_loss = (
        scale_element * atom_valid.to(scale_element.dtype)
    ).sum() / atom_valid.float().sum().clamp_min(1.0)

    # ------------------------------------------------------------------
    # Exact dose-adjustment teacher over a small standard factor grid.
    # ------------------------------------------------------------------
    factor_values = _m1(cfg, "V551_DOSE_TEACHER_FACTORS", [0.5, 1.0, 1.5])
    factors = base.new_tensor([float(value) for value in factor_values])
    if factors.numel() == 0:
        factors = base.new_tensor([1.0])
    base_logit = _safe_logit(base)[:, 0][:, None]
    action_delta_bank = (
        _safe_logit(action_candidates)
        - _safe_logit(base).expand(-1, 4, -1, -1)
    )
    selected_delta = action_delta_bank.gather(
        1,
        best_action[:, :, None, None].expand(-1, -1, h, w),
    )
    dose_gains = []
    for factor in factors:
        candidate = torch.sigmoid(
            base_logit + atom_masks * selected_delta * factor
        )
        dose_gains.append(
            _hard_dice_many(candidate, gt) - hard_base_dice[:, None]
        )
    dose_gain = torch.stack(dose_gains, dim=2)
    best_factor_index = dose_gain.argmax(dim=2)
    dose_target = factors[best_factor_index]
    dose_loss_element = F.smooth_l1_loss(
        dose_adjust,
        dose_target,
        beta=max(float(_m1(cfg, "V551_DOSE_HUBER_BETA", 0.1)), 1.0e-6),
        reduction="none",
    )
    dose_train_mask = benefit_target
    dose_loss = _masked_mean(dose_loss_element, dose_train_mask)

    # ------------------------------------------------------------------
    # Local editor reconstruction.  Preserve atoms target the unchanged Base;
    # beneficial atoms target GT inside the bounded editor region.
    # ------------------------------------------------------------------
    local_target = torch.where(
        benefit_target[:, :, None, None], gt_many, base_many
    ).detach()
    local_bce = F.binary_cross_entropy(
        editor_candidate,
        local_target,
        reduction="none",
    )
    local_bce = (
        local_bce * editor_region
    ).flatten(2).sum(dim=2) / editor_region.flatten(2).sum(dim=2).clamp_min(1.0)
    local_bce_loss = _masked_mean(local_bce, atom_valid)
    local_dice_loss = _soft_dice_loss(
        editor_candidate, local_target, editor_region, atom_valid
    )
    outside = 1.0 - editor_region
    outside_preserve = (
        (editor_candidate - base_many).abs() * outside
    ).flatten(2).sum(dim=2) / outside.flatten(2).sum(dim=2).clamp_min(1.0)
    outside_preserve_loss = _masked_mean(outside_preserve, atom_valid)
    residual_l1 = _masked_mean(
        local_residual.abs().flatten(2).mean(dim=2), atom_valid
    )

    # Exact closure diagnostics for M1 and the final M2-edited atom.
    with torch.no_grad():
        m1_gain = _hard_dice_many(m1_candidate.detach(), gt) - hard_base_dice[:, None]
        editor_gain = (
            _hard_dice_many(editor_candidate.detach(), gt) - hard_base_dice[:, None]
        )
        editor_harm_target = atom_valid & (editor_gain < -harmful_epsilon)
        m1_oracle = torch.cat(
            [torch.zeros_like(hard_base_dice[:, None]), m1_gain.masked_fill(~atom_valid, -1.0e4)],
            dim=1,
        ).max(dim=1).values.mean()
        editor_oracle = torch.cat(
            [
                torch.zeros_like(hard_base_dice[:, None]),
                editor_gain.masked_fill(~atom_valid, -1.0e4),
            ],
            dim=1,
        ).max(dim=1).values.mean()
        error = ((base >= 0.5) != (gt >= 0.5))[:, 0]
        atom_error = atom_masks.bool() & error[:, None]
        atom_purity = atom_error.flatten(2).sum(dim=2).float() / atom_masks.flatten(
            2
        ).sum(dim=2).clamp_min(1.0)
        positive_union = (
            atom_masks.bool() & benefit_target[:, :, None, None]
        ).any(dim=1)
        capture = (
            (positive_union & error).flatten(1).sum(dim=1).float()
            / error.flatten(1).sum(dim=1).clamp_min(1.0)
        ).mean()
        sequential_oracle = _greedy_nonoverlap_oracle(
            base=base,
            candidates=editor_candidate.detach(),
            masks=atom_masks,
            valid=atom_valid,
            gt=gt,
            max_steps=int(_m1(cfg, "V538_COMPOSER_MAX_STEPS", 3)),
            max_overlap=float(_m1(cfg, "V538_COMPOSER_MAX_OVERLAP", 0.20)),
        )

    # M2 Harm veto directly supervises the final edited intervention, not the
    # original M1 candidate.  The inherited V538/V550 selector loss also sees
    # this same final candidate through v541_slot_exact_candidate_probs.
    harm_logit = aux.get("v541_slot_harm_logits")
    if isinstance(harm_logit, torch.Tensor):
        harm_element = F.binary_cross_entropy_with_logits(
            harm_logit,
            editor_harm_target.to(harm_logit.dtype),
            reduction="none",
        )
        harm_veto_loss = (
            harm_element * atom_valid.to(harm_element.dtype)
        ).sum() / atom_valid.float().sum().clamp_min(1.0)
    else:
        harm_veto_loss = zero

    composer_selection_loss = zero
    composer_marginal_loss = zero
    composer_diagnostics: Dict[str, torch.Tensor] = {}
    if composer_enabled:
        (
            composer_selection_loss,
            composer_marginal_loss,
            composer_diagnostics,
        ) = _compute_exact_composer_teacher(
            gt=gt,
            editor_logit_delta=aux["v551_editor_logit_delta"],
            step_logits=aux["v552_composer_step_logits"],
            step_candidate_scores=aux["v552_composer_step_candidate_scores"],
            step_state_probs=aux["v552_composer_step_state_probs"],
            step_eligible=aux["v552_composer_step_eligible"],
            step_active=aux["v552_composer_step_active"],
            stop_margin=float(_m1(cfg, "V552_COMPOSER_TEACHER_STOP_MARGIN", 0.0)),
        )

    m1_extra = (
        float(_m1(cfg, "V551_SCALE_LOSS_WEIGHT", 0.50)) * scale_loss
    )
    editor_scale = _curriculum(
        epoch,
        int(_m1(cfg, "V551_EDITOR_START_EPOCH", 8)),
        int(_m1(cfg, "V551_EDITOR_RAMP_EPOCHS", 8)),
    )
    m2_extra_unscaled = (
        float(_m1(cfg, "V551_ROUTE_LOSS_WEIGHT", 1.0)) * route_loss
        + float(_m1(cfg, "V551_EDITABILITY_LOSS_WEIGHT", 1.0)) * editability_loss
        + float(_m1(cfg, "V551_DOSE_ADJUST_LOSS_WEIGHT", 0.25)) * dose_loss
        + float(_m1(cfg, "V551_LOCAL_BCE_WEIGHT", 0.50)) * local_bce_loss
        + float(_m1(cfg, "V551_LOCAL_DICE_WEIGHT", 0.50)) * local_dice_loss
        + float(_m1(cfg, "V551_OUTSIDE_PRESERVE_WEIGHT", 1.0))
        * outside_preserve_loss
        + float(_m1(cfg, "V551_LOCAL_RESIDUAL_L1_WEIGHT", 0.05)) * residual_l1
        + float(_m1(cfg, "V551_HARM_VETO_WEIGHT", 1.0)) * harm_veto_loss
    )
    composer_scale = _curriculum(
        epoch,
        int(_m1(cfg, "V552_COMPOSER_START_EPOCH", 8)),
        int(_m1(cfg, "V552_COMPOSER_RAMP_EPOCHS", 8)),
    )
    composer_objective = (
        float(_m1(cfg, "V552_COMPOSER_SELECTION_WEIGHT", 1.0))
        * composer_selection_loss
        + float(_m1(cfg, "V552_COMPOSER_MARGINAL_GAIN_WEIGHT", 1.0))
        * composer_marginal_loss
    )
    m2_extra = (
        m2_extra_unscaled * editor_candidate.new_tensor(editor_scale)
        + composer_objective * editor_candidate.new_tensor(composer_scale)
    )

    route_prediction = route_logits.argmax(dim=2)
    diagnostics = {
        "v551_enabled": zero.new_ones(()),
        "v551_fast_parent_atoms_active": aux.get(
            "v551_fast_parent_atoms_active", zero
        ).detach(),
        "v551_editor_active": aux.get(
            "v551_editor_active", zero
        ).detach(),
        "v551_gpu_atomizer_enabled": aux.get(
            "v551_gpu_atomizer_enabled", zero
        ).detach(),
        "v551_single_pass_editor_enabled": aux.get(
            "v551_single_pass_editor_enabled", zero
        ).detach(),
        "v551_scale_loss": scale_loss.detach(),
        "v551_scale_accuracy": (
            (scale_logits.argmax(dim=2) == scale_target)[atom_valid].float().mean()
            if bool(atom_valid.any().item())
            else zero.detach()
        ),
        "v551_route_loss": route_loss.detach(),
        "v551_route_accuracy": (
            (route_prediction == route_target)[atom_valid].float().mean()
            if bool(atom_valid.any().item())
            else zero.detach()
        ),
        "v551_editability_loss": editability_loss.detach(),
        "v551_dose_adjust_loss": dose_loss.detach(),
        "v551_local_bce_loss": local_bce_loss.detach(),
        "v551_local_dice_loss": local_dice_loss.detach(),
        "v551_outside_preserve_loss": outside_preserve_loss.detach(),
        "v551_local_residual_l1": residual_l1.detach(),
        "v551_harm_veto_loss": harm_veto_loss.detach(),
        "v551_editor_train_scale": zero.new_tensor(float(editor_scale)),
        "v552_multicandidate_composer_enabled": aux.get(
            "v552_multicandidate_composer_enabled", zero
        ).detach(),
        "v552_composer_train_scale": zero.new_tensor(float(composer_scale)),
        "v552_composer_predicted_step_count": aux.get(
            "v552_composer_predicted_step_count", zero.new_zeros((base.shape[0],))
        ).float().mean().detach(),
        "v552_safe_calibrated_execution_enabled": aux.get(
            "v552_safe_calibrated_execution_enabled", zero
        ).detach(),
        "v552_atom_quality_gate_active": aux.get(
            "v552_atom_quality_gate_active", zero
        ).detach(),
        "v552_atom_quality_score": aux.get(
            "v552_atom_quality_score", zero
        ).detach(),
        "v552_editor_strength": aux.get(
            "v552_editor_strength", zero
        ).detach(),
        "v551_atom_dense_capacity": aux.get(
            "v551_atom_dense_capacity",
            zero.new_tensor(float(atom_valid.shape[1])),
        ).detach(),
        "v551_atom_valid_count": aux.get(
            "v551_atom_valid_count",
            atom_valid.float().sum(dim=1).mean(),
        ).detach(),
        "v551_atom_count": atom_valid.float().sum(dim=1).mean().detach(),
        "v551_positive_atom_rate": (
            benefit_target.float().sum() / atom_valid.float().sum().clamp_min(1.0)
        ).detach(),
        "v551_harmful_editor_atom_rate": (
            editor_harm_target.float().sum() / atom_valid.float().sum().clamp_min(1.0)
        ).detach(),
        "v551_atom_purity": (
            atom_purity[atom_valid].mean() if bool(atom_valid.any().item()) else zero
        ).detach(),
        "v551_atom_capture_ratio": capture.detach(),
        "v551_m1_atom_gain_mean": (
            m1_gain[atom_valid].mean() if bool(atom_valid.any().item()) else zero
        ).detach(),
        "v551_editor_atom_gain_mean": (
            editor_gain[atom_valid].mean() if bool(atom_valid.any().item()) else zero
        ).detach(),
        "v551_m1_atom_oracle_gain": m1_oracle.detach(),
        "v551_editor_atom_oracle_gain": editor_oracle.detach(),
        "v551_sequential_topk_oracle_gain": sequential_oracle.detach(),
        "v551_preserve_prediction_rate": (
            ((route_prediction == 0) & atom_valid).float().sum()
            / atom_valid.float().sum().clamp_min(1.0)
        ).detach(),
        "v551_mean_dose_adjust": (
            dose_adjust[atom_valid].mean() if bool(atom_valid.any().item()) else zero
        ).detach(),
        "v551_editor_region_fraction": (
            editor_region[atom_valid].mean() if bool(atom_valid.any().item()) else zero
        ).detach(),
    }
    diagnostics.update(composer_diagnostics)
    return m1_extra, m2_extra, diagnostics
