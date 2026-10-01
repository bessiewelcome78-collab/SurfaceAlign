# /home/tsz-25/MedCLIPSeg-pristine/utils/candidate_edit_control_loss.py
"""V395 direct counterfactual-gain policy loss.

This file keeps the original filename so the project still uses the existing
dispatcher.  The previous V393 loss supervised a learned gain scalar but
deployment ignored that scalar and ranked actions mainly with a frozen V381
counterfactual logit.  V395 removes that train/deploy mismatch:

  q(a) is the only Preserve-vs-action score in both training and inference;
  target q(a) is the candidate's clipped, normalised hard-Dice gain;
  Preserve has a fixed score of zero;
  selection, value regression, pairwise ranking, and policy losses all use
  the same q(a).

Ground truth is used only to construct training targets.  Inference receives
only image, text, B0 probabilities, and candidate masks.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F

from utils.candidate_calibration_loss import (
    compute_candidate_calibration_loss,
)
from utils.candidate_consensus_loss import (
    _dense_action_targets,
    _source_residual_loss,
)


def _cfg(node, key, default):
    return getattr(node, key, default)


# V422_M1_SAFE_CANDIDATE_BANK_BEGIN
def _v422_safe_candidate_bank_loss(
    m1,
    base_logits: torch.Tensor,
    action_logits: torch.Tensor,
    supports: torch.Tensor,
    action_types: torch.Tensor,
    gt: torch.Tensor,
):
    """Direction-aware M1 supervision for the existing C1..C8 bank.

    This loss changes only candidate generation:
      delete actions: reward FP removal, penalise TP removal;
      boundary fill: reward FN recovery, strongly penalise BG expansion;
      hole fill: reward FN recovery with a milder BG penalty.

    No M2/M3 score, text feature, selector output, or deployment decision is
    used here. At inference the module still emits exactly the original
    candidate slots and tensor schema.
    """
    zero = action_logits.sum() * 0.0

    is_reference_mode = (
        str(_cfg(m1, "CANDIDATE_MODE", "")).strip().lower()
        == "mechanism"
    )
    if not (is_reference_mode or bool(_cfg(m1, "M1_SAFE_CANDIDATE_BANK", False))):
        return zero, {
            "v422_safe_bank_loss": zero.detach(),
            "v422_delete_fp_removed": zero.detach(),
            "v422_delete_tp_removed": zero.detach(),
            "v422_boundary_fill_fn_added": zero.detach(),
            "v422_boundary_fill_bg_added": zero.detach(),
            "v422_hole_fill_fn_added": zero.detach(),
            "v422_hole_fill_bg_added": zero.detach(),
            "v422_fill_budget_loss": zero.detach(),
        }

    if supports.shape != action_logits.shape:
        raise RuntimeError(
            "V422 support/action shape mismatch: "
            f"{tuple(supports.shape)} vs {tuple(action_logits.shape)}"
        )

    types = action_types.to(action_logits.device).long().reshape(-1)
    if types.numel() != action_logits.shape[1]:
        raise RuntimeError(
            "V422 action type count does not match candidate count: "
            f"{types.numel()} vs {action_logits.shape[1]}"
        )

    gt = gt.float()
    base_prob = torch.sigmoid(base_logits)
    action_prob = torch.sigmoid(action_logits)

    base_hard = (base_prob >= 0.5).float()
    fp = base_hard * (1.0 - gt)
    tp = base_hard * gt
    fn = (1.0 - base_hard) * gt
    bg = (1.0 - base_hard) * (1.0 - gt)

    signed_edit = (
        action_prob - base_prob[:, None]
    ) * supports.float()

    def _region_fraction(values: torch.Tensor, region: torch.Tensor):
        numerator = (values * region[:, None]).sum(dim=(-2, -1))
        denominator = region.sum(dim=(-2, -1)).clamp_min(1.0)
        return numerator / denominator[:, None]

    delete_idx = torch.where((types == 0) | (types == 1))[0]
    boundary_fill_idx = torch.where(types == 2)[0]
    hole_fill_idx = torch.where(types == 3)[0]

    delete_fp_removed = zero
    delete_tp_removed = zero
    delete_loss = zero

    if delete_idx.numel() > 0:
        deletion = (
            -signed_edit.index_select(1, delete_idx)
        ).clamp_min(0.0)

        delete_fp_removed = _region_fraction(deletion, fp)
        delete_tp_removed = _region_fraction(deletion, tp)

        delete_target = max(
            float(_cfg(m1, "M1_SAFE_DELETE_FP_TARGET", 0.002)),
            0.0,
        )
        delete_tp_weight = max(
            float(_cfg(m1, "M1_SAFE_DELETE_TP_WEIGHT", 2.0)),
            0.0,
        )

        delete_loss = (
            F.relu(delete_target - delete_fp_removed).mean()
            + delete_tp_weight * delete_tp_removed.mean()
        )

    def _fill_loss(
        indices: torch.Tensor,
        fn_target_key: str,
        bg_weight_key: str,
        confident_bg_weight_key: str,
        budget_key: str,
    ):
        if indices.numel() == 0:
            return zero, zero, zero, zero

        addition = signed_edit.index_select(1, indices).clamp_min(0.0)

        fn_added = _region_fraction(addition, fn)
        bg_added = _region_fraction(addition, bg)

        confident_bg_threshold = min(
            max(
                float(_cfg(
                    m1,
                    "M1_SAFE_CONFIDENT_BG_THRESHOLD",
                    0.15,
                )),
                0.0,
            ),
            0.5,
        )

        confident_bg = bg * (
            base_prob <= confident_bg_threshold
        ).float()

        confident_bg_added = _region_fraction(addition, confident_bg)

        fn_target = max(float(_cfg(m1, fn_target_key, 0.001)), 0.0)
        bg_weight = max(float(_cfg(m1, bg_weight_key, 1.0)), 0.0)
        confident_bg_weight = max(
            float(_cfg(m1, confident_bg_weight_key, 0.0)),
            0.0,
        )
        edit_budget = max(float(_cfg(m1, budget_key, 0.02)), 0.0)

        edit_fraction = addition.mean(dim=(-2, -1))
        budget_loss = F.relu(edit_fraction - edit_budget).mean()

        loss_value = (
            F.relu(fn_target - fn_added).mean()
            + bg_weight * bg_added.mean()
            + confident_bg_weight * confident_bg_added.mean()
        )

        return loss_value, fn_added, bg_added, budget_loss

    (
        boundary_fill_loss,
        boundary_fill_fn_added,
        boundary_fill_bg_added,
        boundary_fill_budget,
    ) = _fill_loss(
        boundary_fill_idx,
        "M1_SAFE_BOUNDARY_FILL_FN_TARGET",
        "M1_SAFE_BOUNDARY_FILL_BG_WEIGHT",
        "M1_SAFE_BOUNDARY_FILL_CONFIDENT_BG_WEIGHT",
        "M1_SAFE_BOUNDARY_FILL_EDIT_BUDGET",
    )

    (
        hole_fill_loss,
        hole_fill_fn_added,
        hole_fill_bg_added,
        hole_fill_budget,
    ) = _fill_loss(
        hole_fill_idx,
        "M1_SAFE_HOLE_FILL_FN_TARGET",
        "M1_SAFE_HOLE_FILL_BG_WEIGHT",
        "M1_SAFE_HOLE_FILL_CONFIDENT_BG_WEIGHT",
        "M1_SAFE_HOLE_FILL_EDIT_BUDGET",
    )

    safety_weight = max(
        float(_cfg(m1, "M1_SAFE_BANK_WEIGHT", 1.0)),
        0.0,
    )
    budget_weight = max(
        float(_cfg(m1, "M1_SAFE_FILL_BUDGET_WEIGHT", 0.50)),
        0.0,
    )

    total = safety_weight * (
        delete_loss
        + boundary_fill_loss
        + hole_fill_loss
        + budget_weight * (
            boundary_fill_budget + hole_fill_budget
        )
    )

    return total, {
        "v422_safe_bank_loss": total.detach(),
        "v422_delete_fp_removed": delete_fp_removed.mean().detach(),
        "v422_delete_tp_removed": delete_tp_removed.mean().detach(),
        "v422_boundary_fill_fn_added": boundary_fill_fn_added.mean().detach(),
        "v422_boundary_fill_bg_added": boundary_fill_bg_added.mean().detach(),
        "v422_hole_fill_fn_added": hole_fill_fn_added.mean().detach(),
        "v422_hole_fill_bg_added": hole_fill_bg_added.mean().detach(),
        "v422_fill_budget_loss": (
            boundary_fill_budget + hole_fill_budget
        ).detach(),
    }
# V422_M1_SAFE_CANDIDATE_BANK_END


def _hard_dice(probs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Hard Dice for each candidate; output shape [B, K]."""
    if masks.ndim == 4:
        masks = masks[:, 0]
    gt = (masks > 0.5).float()
    pred = (probs > 0.5).float()
    inter = (pred * gt[:, None]).sum(dim=(-2, -1))
    den = pred.sum(dim=(-2, -1)) + gt[:, None].sum(dim=(-2, -1))
    return (2.0 * inter + 1e-7) / (den + 1e-7)


def _action_tensor(aux, keys: Iterable[str], action_count: int) -> torch.Tensor:
    """Read an action tensor, accepting [B,K] and legacy [B,1+K] layouts."""
    for key in keys:
        value = aux.get(key)
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            continue
        if value.shape[1] == action_count + 1:
            value = value[:, 1:]
        if value.shape[1] == action_count:
            return value
    raise KeyError(
        f"Missing direct-gain action tensor. Tried={tuple(keys)}, "
        f"expected K={action_count}."
    )


def _nonempty_action_mask(
    candidates: torch.Tensor,
    aux,
    action_count: int,
) -> torch.Tensor:
    """Only reject physically degenerate actions.

    Invalid matched controls are intentionally *not* hard-rejected here:
    V393's old control gate removed nearly all positive candidates.  The
    direct q-head receives control-validity as an input feature and learns the
    empirical gain target instead.
    """
    supports = aux.get("v20_action_supports")
    if isinstance(supports, torch.Tensor):
        if supports.ndim == 4 and supports.shape[1] == action_count:
            return supports.detach().abs().sum(dim=(-2, -1)) > 1e-8

    # Fallback for compatibility: an action is usable only if its hard mask
    # differs from Preserve somewhere.
    base = (torch.sigmoid(candidates[:, :1]) > 0.5)
    action = (torch.sigmoid(candidates[:, 1:]) > 0.5)
    return (action != base).flatten(2).any(dim=-1)


def _balanced_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    max_pos_weight: float = 20.0,
) -> torch.Tensor:
    """Class-balanced BCE over the current valid action rows."""
    if not valid.any():
        return logits.sum() * 0.0
    y = targets[valid]
    x = logits[valid]
    positives = y.sum()
    negatives = y.numel() - positives
    if positives.item() > 0.0:
        pos_weight = (negatives / positives).clamp(1.0, max_pos_weight)
    else:
        pos_weight = x.new_tensor(1.0)
    return F.binary_cross_entropy_with_logits(x, y, pos_weight=pos_weight)


def _casewise_pairwise_loss(
    q: torch.Tensor,
    beneficial: torch.Tensor,
    nonbeneficial: torch.Tensor,
    valid: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Require every beneficial action to outrank harmful/neutral actions."""
    terms = []
    for batch_index in range(q.shape[0]):
        pos = q[batch_index][beneficial[batch_index] & valid[batch_index]]
        neg = q[batch_index][nonbeneficial[batch_index] & valid[batch_index]]
        if pos.numel() and neg.numel():
            terms.append(
                F.softplus(
                    q.new_tensor(float(margin))
                    - (pos[:, None] - neg[None, :])
                ).mean()
            )
    return torch.stack(terms).mean() if terms else q.sum() * 0.0


def compute_candidate_edit_control_loss(
    cfg,
    candidates: torch.Tensor,
    masks: torch.Tensor,
    aux,
    epoch=None,
):
    """Metric-aligned V395 direct-gain policy.

    Existing callers intentionally retain the V393 function name.  The
    implementation is selected by ``M1.V395_DIRECT_GAIN_POLICY``.  It raises
    a clear error instead of silently using the unsafe legacy deployment
    objective when that flag is absent.
    """
    m1 = cfg.M1
    if not bool(_cfg(m1, "V395_DIRECT_GAIN_POLICY", False)):
        raise RuntimeError(
            "V395_DIRECT_GAIN_POLICY must be true. The old V393 objective is "
            "disabled because it trains a value signal that deployment ignores."
        )

    legacy_loss, diagnostics = (
        compute_candidate_calibration_loss(
            cfg, candidates, masks, aux
        )
    )

    if candidates.ndim != 4 or candidates.shape[1] < 2:
        raise RuntimeError(
            f"V395 expects candidates [B, 1+K, H, W], got {tuple(candidates.shape)}"
        )

    probabilities = torch.sigmoid(candidates)
    action_count = probabilities.shape[1] - 1

    with torch.no_grad():
        base_dice = _hard_dice(probabilities[:, :1], masks)[:, 0]
        action_dice = _hard_dice(probabilities[:, 1:], masks)
        raw_gain = action_dice - base_dice[:, None]

    gain_clip = max(float(_cfg(m1, "V395_GAIN_CLIP", 0.03)), 1.0e-4)
    gain_margin = max(float(_cfg(m1, "V395_GAIN_MARGIN", 0.002)), 0.0)
    huber_beta = max(float(_cfg(m1, "V395_HUBER_BETA", 0.10)), 1.0e-4)

    valid = _nonempty_action_mask(candidates, aux, action_count)
    scaled_gain = raw_gain.clamp(-gain_clip, gain_clip) / gain_clip
    scaled_margin = gain_margin / gain_clip

    beneficial = valid & (raw_gain > gain_margin)
    harmful = valid & (raw_gain < -gain_margin)
    neutral = valid & ~(beneficial | harmful)

    # q is the exact score that deployment compares against Preserve=0.
    q_value = _action_tensor(
        aux,
        ("v395_direct_gain_value", "v393_action_value"),
        action_count,
    )
    benefit_logit = _action_tensor(aux, ("v393_benefit_logit",), action_count)
    harm_logit = _action_tensor(aux, ("v393_harm_logit",), action_count)

    value_loss = (
        F.smooth_l1_loss(
            q_value[valid],
            scaled_gain[valid],
            beta=huber_beta,
        )
        if valid.any()
        else q_value.sum() * 0.0
    )

    benefit_loss = _balanced_bce(
        benefit_logit,
        beneficial.float(),
        valid,
        max_pos_weight=float(_cfg(m1, "V395_MAX_POS_WEIGHT", 20.0)),
    )
    harm_loss = _balanced_bce(
        harm_logit,
        harmful.float(),
        valid,
        max_pos_weight=float(_cfg(m1, "V395_MAX_POS_WEIGHT", 20.0)),
    )

    # Same action scores used by inference: class 0 is Preserve and has score 0.
    choice_logits = torch.cat(
        [
            q_value.new_zeros(q_value.shape[0], 1),
            q_value.masked_fill(~valid, -20.0),
        ],
        dim=1,
    )
    best_gain = raw_gain.masked_fill(~valid, -1e4).max(dim=1).values
    best_index = raw_gain.masked_fill(~valid, -1e4).max(dim=1).indices
    choice_target = torch.where(
        best_gain > gain_margin,
        best_index + 1,
        torch.zeros_like(best_index),
    )
    choice_loss = F.cross_entropy(choice_logits, choice_target)

    action_probability = torch.softmax(choice_logits, dim=1)[:, 1:]
    expected_gain_loss = -(
        action_probability * scaled_gain * valid.float()
    ).sum(dim=1).mean()
    downside_loss = (
        action_probability
        * F.relu(-scaled_gain - scaled_margin)
        * valid.float()
    ).sum(dim=1).mean()

    positive_preserve_loss = (
        F.softplus(
            q_value.new_tensor(scaled_margin) - q_value[beneficial]
        ).mean()
        if beneficial.any()
        else q_value.sum() * 0.0
    )
    harmful_preserve_loss = (
        F.softplus(
            q_value.new_tensor(scaled_margin) + q_value[harmful]
        ).mean()
        if harmful.any()
        else q_value.sum() * 0.0
    )
    neutral_preserve_loss = (
        F.softplus(q_value[neutral]).mean()
        if neutral.any()
        else q_value.sum() * 0.0
    )

    pairwise_loss = _casewise_pairwise_loss(
        q_value,
        beneficial,
        harmful | neutral,
        valid,
        margin=float(_cfg(m1, "V395_PAIRWISE_MARGIN", scaled_margin)),
    )

    # Invalid controls are noisy evidence, not an impossibility proof.  Keep
    # their direct Dice supervision, while discouraging unsupported large q.
    control_valid_raw = aux.get("v393_control_valid")
    if isinstance(control_valid_raw, torch.Tensor):
        control_valid = control_valid_raw
        if control_valid.ndim == 2 and control_valid.shape[1] == action_count + 1:
            control_valid = control_valid[:, 1:]
        invalid_control = valid & ~control_valid.bool()
        invalid_control_loss = (
            q_value[invalid_control].square().mean()
            if invalid_control.any()
            else q_value.sum() * 0.0
        )
    else:
        invalid_control_loss = q_value.sum() * 0.0

    freeze_after = int(_cfg(m1, "M1_FREEZE_GENERATOR_AFTER_EPOCH", 10**9))
    phase_a_legacy_weight = float(_cfg(m1, "V395_PHASE_A_LEGACY_WEIGHT", 1.0))
    phase_b_legacy_weight = float(_cfg(m1, "V395_PHASE_B_LEGACY_WEIGHT", 0.0))
    legacy_weight = (
        phase_a_legacy_weight
        if epoch is None or int(epoch) < freeze_after
        else phase_b_legacy_weight
    )

    total = (
        legacy_weight * legacy_loss
        + float(_cfg(m1, "V395_VALUE_WEIGHT", 3.0)) * value_loss
        + float(_cfg(m1, "V395_BENEFIT_WEIGHT", 0.35)) * benefit_loss
        + float(_cfg(m1, "V395_HARM_WEIGHT", 0.35)) * harm_loss
        + float(_cfg(m1, "V395_CHOICE_WEIGHT", 2.0)) * choice_loss
        + float(_cfg(m1, "V395_EXPECTED_GAIN_WEIGHT", 1.0)) * expected_gain_loss
        + float(_cfg(m1, "V395_DOWNSIDE_WEIGHT", 2.0)) * downside_loss
        + float(_cfg(m1, "V395_POSITIVE_PRESERVE_WEIGHT", 1.5))
        * positive_preserve_loss
        + float(_cfg(m1, "V395_HARM_PRESERVE_WEIGHT", 2.0))
        * harmful_preserve_loss
        + float(_cfg(m1, "V395_NEUTRAL_PRESERVE_WEIGHT", 0.25))
        * neutral_preserve_loss
        + float(_cfg(m1, "V395_PAIRWISE_WEIGHT", 1.0)) * pairwise_loss
        + float(_cfg(m1, "V395_INVALID_CONTROL_WEIGHT", 0.10))
        * invalid_control_loss
    )

    selected_index = choice_logits.argmax(dim=1)
    selected_action = (selected_index - 1).clamp_min(0)
    selected_gain = torch.where(
        selected_index > 0,
        raw_gain.gather(1, selected_action[:, None])[:, 0],
        raw_gain.new_zeros(raw_gain.shape[0]),
    )
    selected_positive = torch.where(
        selected_index > 0,
        beneficial.gather(1, selected_action[:, None])[:, 0],
        torch.zeros_like(selected_index, dtype=torch.bool),
    )
    selected_harmful = torch.where(
        selected_index > 0,
        harmful.gather(1, selected_action[:, None])[:, 0],
        torch.zeros_like(selected_index, dtype=torch.bool),
    )
    oracle_gain = raw_gain.masked_fill(~valid, -1e4).max(dim=1).values
    oracle_gain = torch.where(
        oracle_gain < -1e3,
        torch.zeros_like(oracle_gain),
        oracle_gain,
    )

    diagnostics = dict(diagnostics or {})
    diagnostics.update(
        {
            "v395_legacy_loss": legacy_loss.detach(),
            "v395_legacy_weight": q_value.new_tensor(legacy_weight),
            "v395_value_loss": value_loss.detach(),
            "v395_benefit_loss": benefit_loss.detach(),
            "v395_harm_loss": harm_loss.detach(),
            "v395_choice_loss": choice_loss.detach(),
            "v395_expected_gain_loss": expected_gain_loss.detach(),
            "v395_downside_loss": downside_loss.detach(),
            "v395_positive_preserve_loss": positive_preserve_loss.detach(),
            "v395_harmful_preserve_loss": harmful_preserve_loss.detach(),
            "v395_neutral_preserve_loss": neutral_preserve_loss.detach(),
            "v395_pairwise_loss": pairwise_loss.detach(),
            "v395_invalid_control_loss": invalid_control_loss.detach(),
            "v395_value_mae": (
                (q_value[valid] - scaled_gain[valid]).abs().mean()
                if valid.any()
                else q_value.sum() * 0.0
            ).detach(),
            "v395_valid_action_rate": valid.float().mean().detach(),
            "v395_positive_rate": beneficial.float().mean().detach(),
            "v395_harmful_rate": harmful.float().mean().detach(),
            "v395_selected_gain": selected_gain.mean().detach(),
            "v395_selected_positive_rate": selected_positive.float().mean().detach(),
            "v395_selected_harmful_rate": selected_harmful.float().mean().detach(),
            "v395_preserve_rate": (selected_index == 0).float().mean().detach(),
            "v395_oracle_gain": oracle_gain.mean().detach(),
            "v395_value_mean": q_value.detach().mean(),
        }
    )
    return total, diagnostics


# ---------------------------------------------------------------------------
# V396/FECG CoverageScaled: clean proposal + scaled visual-gain objective.
# ---------------------------------------------------------------------------
def _v396_dense_balanced_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    carrier: torch.Tensor,
    pos_min: float,
    pos_max: float,
) -> torch.Tensor:
    """Balanced BCE over dense proposal carriers, robust to empty positives."""
    valid = carrier > 0.5
    if not valid.any():
        return logits.sum() * 0.0

    x = logits[valid]
    y = targets[valid].float()
    positives = y.sum()
    negatives = y.numel() - positives

    if positives.item() > 0.0:
        pos_weight = (negatives / positives).clamp(
            min=float(pos_min),
            max=float(pos_max),
        )
    else:
        pos_weight = x.new_tensor(1.0)

    return F.binary_cross_entropy_with_logits(
        x,
        y,
        pos_weight=pos_weight,
    )


def _v396_class_balanced_regression(
    prediction: torch.Tensor,
    target: torch.Tensor,
    positive: torch.Tensor,
    harmful: torch.Tensor,
    neutral: torch.Tensor,
    beta: float,
    neutral_weight: float,
) -> torch.Tensor:
    """Equalise rare beneficial/harmful action gradients against neutral rows."""
    terms = []
    for mask, weight in (
        (positive, 1.0),
        (harmful, 1.0),
        (neutral, float(neutral_weight)),
    ):
        if mask.any() and weight > 0.0:
            terms.append(
                float(weight)
                * F.smooth_l1_loss(
                    prediction[mask],
                    target[mask],
                    beta=float(beta),
                )
            )
    return torch.stack(terms).mean() if terms else prediction.sum() * 0.0



def _adaptive_c6_diagnostics(
    candidates: torch.Tensor,
    aux,
):
    """Log the learned C6 residual without adding hand-tuned safety weights."""
    zero = candidates.sum() * 0.0
    increment = aux.get("adaptive_c6_increment")
    scale = aux.get("adaptive_c6_scale")
    if not isinstance(increment, torch.Tensor):
        increment = zero
    if not isinstance(scale, torch.Tensor):
        scale = zero
    return {
        "adaptive_c6_increment_abs_mean": increment.detach().abs().mean(),
        "adaptive_c6_scale": scale.detach().mean(),
    }


def _v426_family_soft_oracle_loss(
    m1,
    candidates: torch.Tensor,
    gt: torch.Tensor,
    aux=None,
):
    """Family-aware candidate oracle loss for a variable-size M1 action bank.

    The old V426 loss was hard-coded to C1/C2 delete and C5/C6 fill.
    This replacement reads v20_action_types when available:
      type 0/1 -> Delete family
      type 2/3 -> Fill family

    It still preserves the existing C0..C8 contract when K=8, but it also
    remains valid when ACTION_BANK_ACTIONS_PER_TYPE is increased later.
    No Val/Test label, selector output, M2 output, text score, or deployment
    threshold is used here.
    """
    zero = candidates.sum() * 0.0
    zero_diag = {
        "v426_family_oracle_loss": zero.detach(),
        "v426_action_quality_loss": zero.detach(),
        "v426_action_no_harm_loss": zero.detach(),
        "v426_action_advantage_loss": zero.detach(),
        "v426_delete_soft_dice": zero.detach(),
        "v426_fill_soft_dice": zero.detach(),
        "v426_base_dice": zero.detach(),
        "v426_delete_action_best_dice": zero.detach(),
        "v426_fill_action_best_dice": zero.detach(),
        "v426_global_action_best_dice": zero.detach(),
        "v426_delete_best_gain_vs_c0": zero.detach(),
        "v426_fill_best_gain_vs_c0": zero.detach(),
        "v426_global_best_gain_vs_c0": zero.detach(),
        "v426_delete_best_gain_vs_c0": zero.detach(),
        "v426_fill_best_gain_vs_c0": zero.detach(),
        "v426_delete_residual_case_rate": zero.detach(),
        "v426_fill_residual_case_rate": zero.detach(),
        "v426_delete_action_count": zero.detach(),
        "v426_fill_action_count": zero.detach(),
        "v426_c1_dice": zero.detach(),
        "v426_c2_dice": zero.detach(),
        "v426_c5_dice": zero.detach(),
        "v426_c6_dice": zero.detach(),
    }

    is_reference_mode = (
        str(_cfg(m1, "CANDIDATE_MODE", "")).strip().lower()
        == "mechanism"
    ) or bool(_cfg(m1, "V426_MECHANISM_DUAL_EXPERT", False))

    if not is_reference_mode:
        return zero, zero, zero_diag

    if candidates.ndim != 4 or candidates.shape[1] < 2:
        raise RuntimeError("V426 family loss expects candidate logits [B,1+K,H,W].")

    if gt.ndim == 4:
        if gt.shape[1] != 1:
            raise RuntimeError("V426 family loss expects gt [B,H,W] or [B,1,H,W].")
        gt = gt[:, 0]

    if gt.ndim != 3:
        raise RuntimeError(f"V426 family loss expects gt [B,H,W], got {tuple(gt.shape)}.")

    probs = torch.sigmoid(candidates)
    target = gt.float()[:, None]
    action_count = probs.shape[1] - 1

    action_types = None
    if isinstance(aux, dict):
        raw_types = aux.get("v20_action_types")
        if isinstance(raw_types, torch.Tensor):
            action_types = raw_types.to(device=probs.device, dtype=torch.long).reshape(-1)

    if action_types is not None and action_types.numel() == action_count:
        delete_action_ids = torch.where((action_types == 0) | (action_types == 1))[0] + 1
        fill_action_ids = torch.where((action_types == 2) | (action_types == 3))[0] + 1
    else:
        delete_action_ids = torch.tensor(
            [slot for slot in (1, 2) if slot < probs.shape[1]],
            device=probs.device,
            dtype=torch.long,
        )
        fill_action_ids = torch.tensor(
            [slot for slot in (5, 6) if slot < probs.shape[1]],
            device=probs.device,
            dtype=torch.long,
        )

    if delete_action_ids.numel() == 0 or fill_action_ids.numel() == 0:
        return zero, zero, zero_diag

    temperature = max(
        float(_cfg(m1, "V426_FAMILY_ORACLE_TEMPERATURE", 0.05)),
        1e-4,
    )

    def dice_for(slot_ids):
        if isinstance(slot_ids, torch.Tensor):
            family = probs.index_select(1, slot_ids.to(probs.device).long())
        else:
            family = probs[:, slot_ids]
        intersection = (family * target).sum(dim=(-2, -1))
        denominator = family.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
        return (2.0 * intersection + 1e-6) / (denominator + 1e-6)

    def soft_best(dice):
        count = max(int(dice.shape[1]), 1)
        return (
            temperature * torch.logsumexp(dice / temperature, dim=1)
            - temperature * torch.log(dice.new_tensor(float(count)))
        )

    preserve_ids = torch.zeros(1, device=probs.device, dtype=torch.long)
    base_dice = dice_for(preserve_ids)[:, 0]

    delete_dice = dice_for(torch.cat([preserve_ids, delete_action_ids]))
    fill_dice = dice_for(torch.cat([preserve_ids, fill_action_ids]))
    delete_action_dice = dice_for(delete_action_ids)
    fill_action_dice = dice_for(fill_action_ids)

    delete_best = soft_best(delete_dice)
    fill_best = soft_best(fill_dice)
    delete_action_best = soft_best(delete_action_dice)
    fill_action_best = soft_best(fill_action_dice)

    family_loss = 0.5 * (
        (1.0 - delete_best).mean()
        + (1.0 - fill_best).mean()
    )

    no_harm_loss = zero
    advantage_loss = zero
    action_quality_loss = zero
    delete_need = torch.zeros_like(delete_best, dtype=torch.bool)
    fill_need = torch.zeros_like(fill_best, dtype=torch.bool)

    if is_reference_mode or bool(_cfg(m1, "V426_ACTION_QUALITY_SUPERVISION", False)):
        active_action_dice = torch.cat([delete_action_dice, fill_action_dice], dim=1)

        no_harm_margin = max(
            float(_cfg(m1, "V426_ACTION_NO_HARM_MARGIN", 0.0)),
            0.0,
        )

        no_harm = F.relu(
            base_dice[:, None]
            + no_harm_margin
            - active_action_dice
        )
        no_harm_loss = no_harm.mean()

        base_hard = (probs[:, 0:1].detach() >= 0.5).float()
        target_hard = (target.detach() >= 0.5).float()

        delete_residual = (
            base_hard * (1.0 - target_hard)
        ).mean(dim=(-3, -2, -1))

        fill_residual = (
            (1.0 - base_hard) * target_hard
        ).mean(dim=(-3, -2, -1))

        min_residual_fraction = max(
            float(_cfg(m1, "V426_ACTION_MIN_RESIDUAL_FRACTION", 0.001)),
            0.0,
        )

        delete_need = delete_residual >= min_residual_fraction
        fill_need = fill_residual >= min_residual_fraction

        advantage_margin = max(
            float(_cfg(m1, "V426_ACTION_ADVANTAGE_MARGIN", 0.0005)),
            0.0,
        )

        def masked_mean(values, mask):
            if not mask.any():
                return values.sum() * 0.0
            return values[mask].sum() / mask.float().sum().clamp_min(1.0)

        delete_advantage = F.relu(
            base_dice
            + advantage_margin
            - delete_action_best
        )

        fill_advantage = F.relu(
            base_dice
            + advantage_margin
            - fill_action_best
        )

        advantage_loss = 0.5 * (
            masked_mean(delete_advantage, delete_need)
            + masked_mean(fill_advantage, fill_need)
        )

        action_quality_loss = (
            max(float(_cfg(m1, "V426_ACTION_NO_HARM_WEIGHT", 1.0)), 0.0)
            * no_harm_loss
            + max(float(_cfg(m1, "V426_ACTION_ADVANTAGE_WEIGHT", 1.0)), 0.0)
            * advantage_loss
        )

    def slot_mean(slot: int):
        if slot < probs.shape[1]:
            return dice_for([slot])[:, 0].mean().detach()
        return zero.detach()

    return family_loss, action_quality_loss, {
        "v426_family_oracle_loss": family_loss.detach(),
        "v426_action_quality_loss": action_quality_loss.detach(),
        "v426_action_no_harm_loss": no_harm_loss.detach(),
        "v426_action_advantage_loss": advantage_loss.detach(),
        "v426_delete_soft_dice": delete_best.mean().detach(),
        "v426_fill_soft_dice": fill_best.mean().detach(),
        "v426_base_dice": base_dice.mean().detach(),
        "v426_delete_action_best_dice": delete_action_best.mean().detach(),
        "v426_fill_action_best_dice": fill_action_best.mean().detach(),
        "v426_global_action_best_dice": torch.maximum(delete_action_best, fill_action_best).mean().detach(),
        "v426_delete_best_gain_vs_c0": (delete_action_best - base_dice).mean().detach(),
        "v426_fill_best_gain_vs_c0": (fill_action_best - base_dice).mean().detach(),
        "v426_global_best_gain_vs_c0": (torch.maximum(delete_action_best, fill_action_best) - base_dice).mean().detach(),
        "v426_delete_best_gain_vs_c0": (delete_action_best - base_dice).mean().detach(),
        "v426_fill_best_gain_vs_c0": (fill_action_best - base_dice).mean().detach(),
        "v426_delete_residual_case_rate": delete_need.float().mean().detach(),
        "v426_fill_residual_case_rate": fill_need.float().mean().detach(),
        "v426_delete_action_count": probs.new_tensor(float(delete_action_ids.numel())).detach(),
        "v426_fill_action_count": probs.new_tensor(float(fill_action_ids.numel())).detach(),
        "v426_c1_dice": slot_mean(1),
        "v426_c2_dice": slot_mean(2),
        "v426_c5_dice": slot_mean(5),
        "v426_c6_dice": slot_mean(6),
    }

def _v428_adaptive_coverage_purity_loss(
    m1,
    actionness: torch.Tensor,
    dense_target: torch.Tensor,
    dense_carrier: torch.Tensor,
    selected_supports: torch.Tensor,
    action_types: torch.Tensor,
    proposal_reference: torch.Tensor,
):
    """Adaptive coverage-purity loss.

    V428 applies this to all action families.
    V429 applies it only to Delete action types {0, 1}; Fill actions keep
    evidence-qualified carrier, repair loss, V422 safety and family oracle.
    """
    zero = actionness.sum() * 0.0
    delete_only = bool(
        _cfg(m1, "V429_ASYMMETRIC_ADAPTIVE", False)
    )

    zero_diag = {
        "v428_adaptive_loss": zero.detach(),
        "v428_soft_top_coverage": zero.detach(),
        "v428_proposal_purity": zero.detach(),
        "v428_active_action_rows": zero.detach(),
        "v429_delete_only_adaptive": zero.new_tensor(float(delete_only)),
    }

    if not bool(_cfg(m1, "V428_ADAPTIVE_DUAL_EXPERT", False)):
        return zero, zero_diag

    if (
        actionness.shape != dense_target.shape
        or actionness.shape != dense_carrier.shape
        or actionness.shape != selected_supports.shape
    ):
        raise RuntimeError(
            "V428/V429 actionness-target-carrier-support shape mismatch: "
            f"{tuple(actionness.shape)}, "
            f"{tuple(dense_target.shape)}, "
            f"{tuple(dense_carrier.shape)}, "
            f"{tuple(selected_supports.shape)}"
        )

    types = action_types.to(
        device=actionness.device,
        dtype=torch.long,
    ).reshape(-1)

    if types.numel() != actionness.shape[1]:
        raise RuntimeError(
            "V428/V429 action-type count mismatch: "
            f"{types.numel()} vs {actionness.shape[1]}"
        )

    raw_terms = []
    coverage_terms = []
    purity_terms = []
    active_rows = 0

    for slot in range(actionness.shape[1]):
        action_type = int(types[slot].item())

        # V429 dynamically learns coverage only for Delete. Fill remains
        # precision-first and is protected by the original evidence carrier.
        if delete_only and action_type not in (0, 1):
            continue

        carrier = dense_carrier[:, slot] > 0.5
        target = dense_target[:, slot].float() * carrier.float()

        valid = (
            (carrier.flatten(1).sum(dim=1) > 0)
            & (target.flatten(1).sum(dim=1) > 0)
            & (selected_supports[:, slot].flatten(1).sum(dim=1) > 0)
        )

        if not valid.any():
            continue

        logits = actionness[:, slot][valid]
        carrier_v = carrier[valid]
        target_v = target[valid]

        ranking_logits = logits.masked_fill(
            ~carrier_v,
            torch.finfo(logits.dtype).min,
        )

        selection = torch.softmax(
            ranking_logits.flatten(1),
            dim=1,
        ).reshape_as(logits)

        soft_top_coverage = (
            selection * target_v
        ).sum(dim=(-2, -1))

        proposal_mass = (
            torch.sigmoid(logits) * carrier_v.float()
        )

        proposal_purity = (
            (proposal_mass * target_v).sum(dim=(-2, -1))
            / proposal_mass.sum(dim=(-2, -1)).clamp_min(1.0e-6)
        )

        harmonic = (
            2.0
            * soft_top_coverage
            * proposal_purity
            / (soft_top_coverage + proposal_purity + 1.0e-6)
        )

        raw_terms.append(
            -torch.log(harmonic.clamp_min(1.0e-6)).mean()
        )
        coverage_terms.append(soft_top_coverage.mean())
        purity_terms.append(proposal_purity.mean())
        active_rows += int(valid.sum().item())

    if not raw_terms:
        return zero, zero_diag

    raw_loss = torch.stack(raw_terms).mean()

    # Dynamic normalization only: no manually chosen V429 loss coefficient.
    adaptive_loss = raw_loss * (
        proposal_reference.detach()
        / raw_loss.detach().clamp_min(1.0e-6)
    )

    return adaptive_loss, {
        "v428_adaptive_loss": adaptive_loss.detach(),
        "v428_soft_top_coverage": torch.stack(
            coverage_terms
        ).mean().detach(),
        "v428_proposal_purity": torch.stack(
            purity_terms
        ).mean().detach(),
        "v428_active_action_rows": actionness.new_tensor(
            float(active_rows)
        ),
        "v429_delete_only_adaptive": actionness.new_tensor(
            float(delete_only)
        ),
    }



def _v396_clean_proposal_loss(
    cfg,
    candidates: torch.Tensor,
    masks: torch.Tensor,
    aux,
) -> tuple[torch.Tensor, dict]:
    """Only the image-side proposal and local repair supervision.

    This intentionally excludes V381's historical atomic text/calibrator,
    consensus and deployment objectives.  Actionness uses dense residual
    supervision because Top-k window selection is discrete; repair trains the
    actual edited logits; source supervision trains the optional island
    residual field.
    """
    m1 = cfg.M1

    if candidates.ndim != 4 or candidates.shape[1] < 2:
        raise RuntimeError("V396 proposal loss expects [B,1+K,H,W].")

    base = candidates[:, 0]
    actions = candidates[:, 1:]
    supports = aux.get("v20_action_supports")
    type_supports = aux.get("v20_type_supports")
    action_types = aux.get("v20_action_types")
    actionness = aux.get("v20_actionness_logits")

    if not all(
        isinstance(value, torch.Tensor)
        for value in (supports, type_supports, action_types, actionness)
    ):
        raise RuntimeError(
            "V396 proposal loss requires V20 supports, type supports, types "
            "and dense actionness logits."
        )

    supports = supports.float()
    type_supports = type_supports.float()

    if supports.shape != actions.shape:
        raise RuntimeError(
            "V396 candidate/action-support shape mismatch: "
            f"{tuple(actions.shape)} vs {tuple(supports.shape)}."
        )

    if masks.ndim == 4:
        gt = (masks[:, 0] > 0.5).to(dtype=candidates.dtype)
    else:
        gt = (masks > 0.5).to(dtype=candidates.dtype)

    dense_target, dense_carrier = _dense_action_targets(
        base,
        gt,
        type_supports,
        action_types.to(actions.device),
    )

    proposal_loss = _v396_dense_balanced_bce(
        actionness,
        dense_target,
        dense_carrier,
        pos_min=float(_cfg(m1, "EVIDENCE_GUIDED_PROPOSAL_POS_WEIGHT_MIN", 2.0)),
        pos_max=float(_cfg(m1, "EVIDENCE_GUIDED_PROPOSAL_POS_WEIGHT_MAX", 25.0)),
    )

    v428_adaptive_loss, v428_adaptive_diag = (
        _v428_adaptive_coverage_purity_loss(
            m1=m1,
            actionness=actionness,
            dense_target=dense_target,
            dense_carrier=dense_carrier,
            selected_supports=supports,
            action_types=action_types,
            proposal_reference=proposal_loss,
        )
    )

    repair_raw = F.binary_cross_entropy_with_logits(
        actions,
        gt[:, None].expand_as(actions),
        reduction="none",
    )
    repair_loss = (
        repair_raw * supports
    ).sum() / supports.sum().clamp_min(1.0)

    if isinstance(aux.get("v35_residual_logit_map"), torch.Tensor):
        source_loss, source_iou = _source_residual_loss(
            cfg,
            base,
            gt,
            aux,
        )
    else:
        source_loss = proposal_loss * 0.0
        source_iou = proposal_loss.detach() * 0.0

    v422_safe_loss, v422_safe_diag = _v422_safe_candidate_bank_loss(
        m1=m1,
        base_logits=base,
        action_logits=actions,
        supports=supports,
        action_types=action_types.to(actions.device),
        gt=gt,
    )

    (
        v426_oracle_loss,
        v426_action_quality_loss,
        v426_oracle_diag,
    ) = _v426_family_soft_oracle_loss(
        m1=m1,
        candidates=candidates,
        gt=gt,
        aux=aux,
    )

    total = (
        float(_cfg(m1, "EVIDENCE_GUIDED_PROPOSAL_WEIGHT", 1.0)) * proposal_loss
        + float(_cfg(m1, "EVIDENCE_GUIDED_REPAIR_WEIGHT", 0.50)) * repair_loss
        + float(_cfg(m1, "EVIDENCE_GUIDED_SOURCE_WEIGHT", 0.25)) * source_loss
        + v422_safe_loss
        + float(_cfg(m1, "V426_FAMILY_ORACLE_WEIGHT", 0.20))
        * v426_oracle_loss
        + (
            1.0
            if str(_cfg(m1, "CANDIDATE_MODE", "")).strip().lower()
            == "mechanism"
            else float(_cfg(m1, "V426_ACTION_QUALITY_WEIGHT", 0.0))
        ) * v426_action_quality_loss
        + v428_adaptive_loss
    )

    return total, {
        "v396_proposal_dense_loss": proposal_loss.detach(),
        "v396_repair_loss": repair_loss.detach(),
        "v396_source_loss": source_loss.detach(),
        "v396_source_iou": source_iou.detach(),
        **v422_safe_diag,
        **v426_oracle_diag,
        **_adaptive_c6_diagnostics(candidates, aux),
        **v428_adaptive_diag,
    }


def compute_reference_candidate_loss(
    cfg,
    candidates: torch.Tensor,
    masks: torch.Tensor,
    aux,
    epoch=None,
):
    """Reference M1 objective: fixed V430-quality loss plus adaptive C6 residual.

    All geometric and objective coefficients are implementation constants from
    the validated reference path. The only newly learned quantity is the C6
    residual scale in ``ReferenceAdaptiveC6Bank``.
    """
    del epoch
    return _v396_clean_proposal_loss(cfg, candidates, masks, aux)


def compute_evidence_guided_candidate_loss(
    cfg,
    candidates: torch.Tensor,
    masks: torch.Tensor,
    aux,
    epoch=None,
):
    """Clean V396/FECG objective.

    1. Image-only proposal supervision creates useful local actions.
    2. A frozen-observer visual critic regresses *scaled signed* Dice gain.
    3. Preserve-vs-action q(a) regresses only positive gain above margin:
       q(a)=fixed_text_gate(a) × ReLU(visual_gain(a)).
    4. The target-text 2x2 evidence is a hard deployment gate and therefore
       cannot be bypassed by the learned visual critic.
    """
    del epoch
    m1 = cfg.M1

    if bool(_cfg(m1, "SAFE_RESIDUAL_ONLY_TRAINING", False)):
        return compute_safe_residual_candidate_loss(
            cfg, candidates, masks, aux
        )

    if not bool(_cfg(m1, "EVIDENCE_GUIDED_FECG_ENABLED", False)):
        raise RuntimeError("EVIDENCE_GUIDED_FECG_ENABLED must be true for evidence_guided_candidate_control loss.")

    if candidates.ndim != 4 or candidates.shape[1] < 2:
        raise RuntimeError(
            f"V396 expects candidate logits [B,1+K,H,W], got {tuple(candidates.shape)}"
        )

    proposal_core, proposal_diag = _v396_clean_proposal_loss(
        cfg,
        candidates,
        masks,
        aux,
    )

    # V423_STRICT_CANDIDATE_PROPOSAL_ONLY
    # Strict candidate pretraining must not use frozen/random q, visual-gain,
    # text-evidence, cf-verifier, or selector heads as supervision.  Keep the
    # FECG flag enabled only because this loss family owns the shared proposal
    # pipeline; return its image-side proposal/repair/source/V422-safe loss.
    if bool(_cfg(m1, "M1_STRICT_CANDIDATE_ONLY", False)):
        diagnostics = dict(proposal_diag)
        diagnostics.update({
            "v423_strict_candidate_only": proposal_core.detach().new_tensor(1.0),
            "v396_proposal_core_loss": proposal_core.detach(),
            "v396_visual_gain_loss": proposal_core.detach() * 0.0,
            "v396_action_value_loss": proposal_core.detach() * 0.0,
            "v396_preserve_loss": proposal_core.detach() * 0.0,
            "v396_pair_loss": proposal_core.detach() * 0.0,
            "v396_geometric_valid_fraction": proposal_core.detach() * 0.0,
            "v396_positive_fraction": proposal_core.detach() * 0.0,
            "v396_harmful_fraction": proposal_core.detach() * 0.0,
            "v396_neutral_fraction": proposal_core.detach() * 0.0,
        })
        return proposal_core, diagnostics

    probabilities = torch.sigmoid(candidates)
    action_count = probabilities.shape[1] - 1

    with torch.no_grad():
        base_dice = _hard_dice(probabilities[:, :1], masks)[:, 0]
        action_dice = _hard_dice(probabilities[:, 1:], masks)
        raw_gain = action_dice - base_dice[:, None]

    q = _action_tensor(aux, ("v396_action_value",), action_count)
    visual_gain = _action_tensor(aux, ("v396_visual_gain",), action_count)
    evidence = _action_tensor(aux, ("v396_text_evidence",), action_count)
    paraphrase_evidence = _action_tensor(
        aux,
        ("v396_paraphrase_evidence", "v396_swap_evidence"),
        action_count,
    )

    control_valid = _action_tensor(
        aux,
        ("v393_control_valid",),
        action_count,
    ).bool()
    control_ratio = _action_tensor(
        aux,
        ("v393_control_area_ratio",),
        action_count,
    )
    control_overlap = _action_tensor(
        aux,
        ("v393_control_overlap",),
        action_count,
    )
    context_overlap = _action_tensor(
        aux,
        ("v393_context_overlap",),
        action_count,
    )

    supports = aux.get("v20_action_supports")
    if not isinstance(supports, torch.Tensor) or supports.ndim != 4:
        raise RuntimeError("V396 requires v20_action_supports [B,K,H,W].")
    if supports.shape[1] != action_count:
        raise RuntimeError(
            "V396 action support count does not match candidate action count."
        )

    action_area = supports.detach().float().mean(dim=(-2, -1))
    area_tol = max(
        float(_cfg(m1, "EVIDENCE_GUIDED_CONTROL_AREA_TOLERANCE", 1e-3)),
        0.0,
    )
    max_edit = max(
        float(_cfg(m1, "EVIDENCE_GUIDED_MAX_EDIT_FRACTION", 0.035)),
        0.0,
    )

    # This is exactly the geometry safety contract used during deployment.
    valid = (
        control_valid
        & ((control_ratio - 1.0).abs() <= area_tol)
        & (control_overlap <= 1e-6)
        & (context_overlap <= 1e-6)
        & (action_area > 0.0)
        & (action_area <= max_edit)
    )

    margin = max(float(_cfg(m1, "EVIDENCE_GUIDED_GAIN_MARGIN", 0.002)), 0.0)
    gain_scale = max(
        float(_cfg(m1, "EVIDENCE_GUIDED_GAIN_TARGET_SCALE", 0.010)),
        1e-5,
    )
    target_clip = max(
        float(_cfg(m1, "EVIDENCE_GUIDED_GAIN_TARGET_CLIP", 1.0)),
        1e-4,
    )
    huber_beta = max(
        float(_cfg(m1, "EVIDENCE_GUIDED_HUBER_BETA", 0.10)),
        1e-5,
    )

    scaled_signed_gain = (
        raw_gain / gain_scale
    ).clamp(-target_clip, target_clip).detach()

    # q is an acceptance score against Preserve=0, so its supervised target is
    # positive gain *beyond* the raw Dice margin. Signed harm belongs to the
    # visual critic, not q(a), otherwise the evidence gate and negative-q
    # regression objectives contradict one another.
    scaled_benefit = (
        (raw_gain - margin) / gain_scale
    ).clamp(0.0, target_clip).detach()

    positive = valid & (raw_gain > margin)
    harmful = valid & (raw_gain < -margin)
    neutral = valid & ~(positive | harmful)

    visual_loss = _v396_class_balanced_regression(
        visual_gain,
        scaled_signed_gain,
        positive,
        harmful,
        neutral,
        beta=huber_beta,
        neutral_weight=float(
            _cfg(m1, "EVIDENCE_GUIDED_NEUTRAL_VISUAL_WEIGHT", 0.25)
        ),
    )

    action_value_loss = _v396_class_balanced_regression(
        q,
        scaled_benefit,
        positive,
        harmful,
        neutral,
        beta=huber_beta,
        neutral_weight=float(
            _cfg(m1, "EVIDENCE_GUIDED_NEUTRAL_Q_WEIGHT", 0.50)
        ),
    )

    preserve_terms = []
    if positive.any():
        preserve_terms.append(
            F.relu(
                scaled_benefit[positive] - q[positive]
            ).mean()
        )
    nonpositive = valid & ~positive
    if nonpositive.any():
        preserve_terms.append(q[nonpositive].mean())
    preserve_loss = (
        torch.stack(preserve_terms).mean()
        if preserve_terms else q.sum() * 0.0
    )

    pair_terms = []
    pair_margin = max(
        float(_cfg(m1, "EVIDENCE_GUIDED_PAIRWISE_MARGIN", 0.10)),
        0.0,
    )
    for batch_index in range(q.shape[0]):
        pos = q[batch_index][positive[batch_index]]
        neg = q[batch_index][(harmful | neutral)[batch_index]]
        if pos.numel() and neg.numel():
            pair_terms.append(
                F.relu(
                    q.new_tensor(pair_margin)
                    - (pos[:, None] - neg[None, :])
                ).mean()
            )
    pair_loss = (
        torch.stack(pair_terms).mean()
        if pair_terms else q.sum() * 0.0
    )

    # Evidence is immutable by design. Log its falsification quality, but do
    # not train a semantic adapter that could reintroduce segmentation-label
    # shortcuts. The paraphrase value is an invariance audit, not a batch swap.
    if positive.any():
        positive_evidence = evidence[positive].mean()
    else:
        positive_evidence = evidence.sum() * 0.0
    if harmful.any():
        harmful_evidence = evidence[harmful].mean()
    else:
        harmful_evidence = evidence.sum() * 0.0
    if valid.any():
        paraphrase_gap = (
            evidence[valid] - paraphrase_evidence[valid]
        ).abs().mean()
    else:
        paraphrase_gap = evidence.sum() * 0.0

    total = (
        proposal_core
        + float(_cfg(m1, "EVIDENCE_GUIDED_VISUAL_GAIN_WEIGHT", 3.0)) * visual_loss
        + float(_cfg(m1, "EVIDENCE_GUIDED_ACTION_VALUE_WEIGHT", 2.0)) * action_value_loss
        + float(_cfg(m1, "EVIDENCE_GUIDED_PRESERVE_WEIGHT", 1.0)) * preserve_loss
        + float(_cfg(m1, "EVIDENCE_GUIDED_PAIRWISE_WEIGHT", 1.0)) * pair_loss
    )

    chosen = aux.get("v20_selector_hard")
    if isinstance(chosen, torch.Tensor) and chosen.shape == raw_gain.shape:
        selected_gain = (
            raw_gain * chosen.detach()
        ).sum() / chosen.detach().sum().clamp_min(1.0)
    else:
        selected_gain = q.sum() * 0.0

    oracle_gain = raw_gain.masked_fill(~valid, -1e4).max(dim=1).values
    oracle_gain = torch.where(
        oracle_gain < -1e3,
        torch.zeros_like(oracle_gain),
        oracle_gain,
    )

    diagnostics = dict(proposal_diag)
    diagnostics.update({
        "v396_proposal_core_loss": proposal_core.detach(),
        "v396_visual_gain_loss": visual_loss.detach(),
        "v396_action_value_loss": action_value_loss.detach(),
        "v396_preserve_loss": preserve_loss.detach(),
        "v396_pair_loss": pair_loss.detach(),
        "v396_geometric_valid_fraction": valid.float().mean().detach(),
        "v396_positive_fraction": positive.float().mean().detach(),
        "v396_harmful_fraction": harmful.float().mean().detach(),
        "v396_neutral_fraction": neutral.float().mean().detach(),
        "v396_mean_visual_gain": visual_gain.detach().mean(),
        "v396_mean_q": q.detach().mean(),
        "v396_positive_text_evidence": positive_evidence.detach(),
        "v396_harmful_text_evidence": harmful_evidence.detach(),
        "v396_paraphrase_gap": paraphrase_gap.detach(),
        "v396_selected_gain": selected_gain.detach(),
        "v396_oracle_gain": oracle_gain.mean().detach(),
        "v396_scaled_target_abs_mean": (
            scaled_signed_gain[valid].abs().mean()
            if valid.any() else q.sum() * 0.0
        ).detach(),
    })
    return total, diagnostics



def _safe_residual_boundary(probability: torch.Tensor) -> torch.Tensor:
    eroded = -F.max_pool2d(-probability, 3, 1, 1)
    return (probability - eroded).abs().clamp(0.0, 1.0)


def _safe_residual_dice(
    probability: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if weight is None:
        weight = torch.ones_like(probability)
    inter = (probability * target * weight).sum(dim=(-2, -1))
    den = ((probability + target) * weight).sum(dim=(-2, -1))
    return (2.0 * inter + 1e-6) / (den + 1e-6)


def _compute_safe_residual_candidate_loss_legacy(cfg, candidates, masks, aux):
    """M1-only safety objective for diagnostic C9/C10 residual candidates."""
    logits = aux.get("safe_residual_candidate_logits")
    boundary_carrier = aux.get("safe_boundary_carrier")
    context_carrier = aux.get("safe_context_carrier")
    direction_probability = aux.get("safe_context_fill_probability")
    needed = (logits, boundary_carrier, context_carrier, direction_probability)
    if not all(isinstance(x, torch.Tensor) for x in needed):
        raise RuntimeError("Safe residual training requires C9/C10 auxiliary tensors.")
    if logits.ndim != 4 or logits.shape[1] != 2:
        raise RuntimeError(
            "Safe residual logits must have shape [B,2,H,W], got "
            f"{tuple(logits.shape)}."
        )
    if masks.ndim == 4:
        target = (masks[:, :1] > 0.5).to(logits.dtype)
    else:
        target = (masks > 0.5).to(logits.dtype)[:, None]
    base_prob = torch.sigmoid(candidates[:, :1]).detach()
    base_hard = (base_prob >= 0.5).to(target.dtype)
    c9_prob = torch.sigmoid(logits[:, :1])
    c10_prob = torch.sigmoid(logits[:, 1:2])
    mode = str(_cfg(cfg.M1, "SAFE_RESIDUAL_MODE", "both")).lower()
    use_c9 = mode in {"boundary", "both"}
    use_c10 = mode in {"context", "both"}
    if not (use_c9 or use_c10):
        raise ValueError("SAFE_RESIDUAL_MODE must be boundary, context, or both.")
    error = (base_hard != target).to(target.dtype)
    error_weight = float(_cfg(cfg.M1, "SAFE_ERROR_WEIGHT", 4.0))
    b_weight = 0.20 + boundary_carrier * (1.0 + error_weight * error)
    c_weight = 0.20 + context_carrier * (1.0 + error_weight * error)
    zero = logits.sum() * 0.0
    bce_terms, dice_terms = [], []
    active_probs, active_carriers = [], []
    if use_c9:
        bce = F.binary_cross_entropy_with_logits(logits[:, :1], target, reduction="none")
        bce_terms.append((bce * b_weight).sum() / b_weight.sum().clamp_min(1.0))
        dice_terms.append(1.0 - _safe_residual_dice(c9_prob, target).mean())
        active_probs.append(c9_prob); active_carriers.append(boundary_carrier)
    if use_c10:
        bce = F.binary_cross_entropy_with_logits(logits[:, 1:2], target, reduction="none")
        bce_terms.append((bce * c_weight).sum() / c_weight.sum().clamp_min(1.0))
        dice_terms.append(1.0 - _safe_residual_dice(c10_prob, target).mean())
        active_probs.append(c10_prob); active_carriers.append(context_carrier)
    region_bce = torch.stack(bce_terms).mean() if bce_terms else zero
    region_dice = torch.stack(dice_terms).mean() if dice_terms else zero
    boundary_dice = (
        1.0 - _safe_residual_dice(
            _safe_residual_boundary(c9_prob),
            _safe_residual_boundary(target),
            0.20 + boundary_carrier,
        ).mean()
        if use_c9 else zero
    )
    low=float(_cfg(cfg.M1, "SAFE_PRESERVE_LOW", 0.15))
    high=float(_cfg(cfg.M1, "SAFE_PRESERVE_HIGH", 0.85))
    confident_correct=(((base_prob<=low)&(target<0.5))|((base_prob>=high)&(target>=0.5))).to(logits.dtype)
    stacked_prob=torch.cat(active_probs,dim=1)
    stacked_carrier=torch.cat(active_carriers,dim=1)
    preserve=((stacked_prob-base_prob).abs()*confident_correct).sum()/(
        confident_correct.sum().clamp_min(1.0)*stacked_prob.shape[1]
    )
    margin=float(_cfg(cfg.M1,"SAFE_HARD_FLIP_MARGIN",0.04))
    protected_fg=confident_correct*target
    protected_bg=confident_correct*(1.0-target)
    hard_flip=(
        F.relu((0.50+margin)-stacked_prob)*protected_fg
    ).sum()+(
        F.relu(stacked_prob-(0.50-margin))*protected_bg
    ).sum()
    hard_flip=hard_flip/(
        (protected_fg.sum()+protected_bg.sum()).clamp_min(1.0)*stacked_prob.shape[1]
    )
    outside=((stacked_prob-base_prob).abs()*(1.0-stacked_carrier)).mean()
    change=(stacked_prob-base_prob).abs().mean(dim=(-2,-1))
    budgets=[]
    if use_c9:
        budgets.append(F.relu(change[:,0]-float(_cfg(cfg.M1,"SAFE_BOUNDARY_CHANGE_BUDGET",0.006))).mean())
    if use_c10:
        budgets.append(F.relu(change[:,-1]-float(_cfg(cfg.M1,"SAFE_CONTEXT_CHANGE_BUDGET",0.010))).mean())
    budget=torch.stack(budgets).mean() if budgets else zero
    if use_c10:
        ir=int(_cfg(cfg.M1,"SAFE_CONTEXT_INNER_RADIUS",2))
        orad=int(_cfg(cfg.M1,"SAFE_CONTEXT_OUTER_RADIUS",6))
        inner=(base_hard-(-F.max_pool2d(-base_hard,2*ir+1,1,ir))).clamp(0.0,1.0)
        outer=(F.max_pool2d(base_hard,2*orad+1,1,orad)-base_hard).clamp(0.0,1.0)
        fill_need=((1.0-base_hard)*target*outer).sum(dim=(-2,-1))
        delete_need=(base_hard*(1.0-target)*inner).sum(dim=(-2,-1))
        valid_direction=(fill_need+delete_need)>0
        direction_target=(fill_need>delete_need).to(direction_probability.dtype).reshape(-1)
        direction_loss=(
            F.binary_cross_entropy(direction_probability.reshape(-1)[valid_direction.reshape(-1)], direction_target[valid_direction.reshape(-1)])
            if valid_direction.any() else zero
        )
        direction_confidence=(direction_probability*(1.0-direction_probability)).mean()
    else:
        direction_loss=zero; direction_confidence=zero
    base_dice=_safe_residual_dice(base_prob,target)
    improve_margin=float(_cfg(cfg.M1,"SAFE_SOFT_IMPROVEMENT_MARGIN",0.0005))
    improve=[]
    if use_c9:
        gain=_safe_residual_dice(c9_prob,target)[:,0]-base_dice[:,0]
        present=(error*boundary_carrier).sum(dim=(-2,-1))[:,0]>0.5
        improve.append((F.relu(improve_margin-gain)*present.to(logits.dtype)).sum()/present.sum().clamp_min(1.0))
    if use_c10:
        gain=_safe_residual_dice(c10_prob,target)[:,0]-base_dice[:,0]
        present=(error*context_carrier).sum(dim=(-2,-1))[:,0]>0.5
        improve.append((F.relu(improve_margin-gain)*present.to(logits.dtype)).sum()/present.sum().clamp_min(1.0))
    improve_loss=torch.stack(improve).mean() if improve else zero
    total=(
        float(_cfg(cfg.M1,"SAFE_REGION_BCE_WEIGHT",1.0))*region_bce+
        float(_cfg(cfg.M1,"SAFE_REGION_DICE_WEIGHT",1.0))*region_dice+
        float(_cfg(cfg.M1,"SAFE_BOUNDARY_DICE_WEIGHT",0.80))*boundary_dice+
        float(_cfg(cfg.M1,"SAFE_PRESERVE_WEIGHT",1.50))*preserve+
        float(_cfg(cfg.M1,"SAFE_HARD_FLIP_WEIGHT",3.0))*hard_flip+
        float(_cfg(cfg.M1,"SAFE_OUTSIDE_WEIGHT",1.0))*outside+
        float(_cfg(cfg.M1,"SAFE_BUDGET_WEIGHT",1.0))*budget+
        float(_cfg(cfg.M1,"SAFE_DIRECTION_WEIGHT",0.50))*direction_loss+
        float(_cfg(cfg.M1,"SAFE_DIRECTION_CONFIDENCE_WEIGHT",0.15))*direction_confidence+
        float(_cfg(cfg.M1,"SAFE_IMPROVEMENT_WEIGHT",0.75))*improve_loss
    )
    hard_base=base_prob>=0.5
    return total,{
        "safe_residual_loss":total.detach(),
        "safe_region_bce":region_bce.detach(),
        "safe_region_dice":region_dice.detach(),
        "safe_boundary_dice":boundary_dice.detach(),
        "safe_preserve_loss":preserve.detach(),
        "safe_hard_flip_loss":hard_flip.detach(),
        "safe_outside_loss":outside.detach(),
        "safe_budget_loss":budget.detach(),
        "safe_direction_loss":direction_loss.detach(),
        "safe_direction_confidence":direction_confidence.detach(),
        "safe_improvement_loss":improve_loss.detach(),
        "safe_c9_soft_dice_gain":(_safe_residual_dice(c9_prob,target)[:,0]-base_dice[:,0]).mean().detach(),
        "safe_c10_soft_dice_gain":(_safe_residual_dice(c10_prob,target)[:,0]-base_dice[:,0]).mean().detach(),
        "safe_c9_hard_change":((c9_prob>=0.5)!=hard_base).float().mean().detach(),
        "safe_c10_hard_change":((c10_prob>=0.5)!=hard_base).float().mean().detach(),
    }


def _safe_probability_bce(
    probability: torch.Tensor,
    target: torch.Tensor,
    max_pos_weight: float = 20.0,
) -> torch.Tensor:
    """Balanced BCE for a probability-valued binary certificate."""
    probability = probability.clamp(1e-5, 1.0 - 1e-5)
    target = target.to(probability.dtype)
    positives = target.sum()
    negatives = target.numel() - positives
    if positives.item() > 0.0:
        pos_weight = (negatives / positives).clamp(1.0, max_pos_weight)
    else:
        pos_weight = probability.new_tensor(1.0)
    return F.binary_cross_entropy(
        probability,
        target,
        weight=torch.where(target > 0.5, pos_weight, torch.ones_like(target)),
    )


def _compute_safe_context_certified_loss(cfg, candidates, masks, aux):
    """A3 loss: C9 fixed, C10 certified, local, and no-regression constrained.

    C10 is supervised only as an independently trainable candidate. It may
    learn a direction only when one polarity clearly dominates the train-split
    residual; ambiguous/no-error cases have certificate target 0 and therefore
    return Preserve exactly in forward inference.
    """
    logits = aux.get("safe_residual_candidate_logits")
    context_carrier = aux.get("safe_context_carrier")
    direction_probability = aux.get("safe_context_fill_probability")
    certificate_probability = aux.get(
        "safe_context_direction_certificate_soft"
    )
    near_threshold_gate = aux.get("safe_context_near_threshold_gate")
    needed = (
        logits,
        context_carrier,
        direction_probability,
        certificate_probability,
        near_threshold_gate,
    )
    if not all(isinstance(value, torch.Tensor) for value in needed):
        raise RuntimeError(
            "A3 certified C10 requires logits, carrier, direction, "
            "certificate, and near-threshold diagnostics."
        )
    if logits.ndim != 4 or logits.shape[1] != 2:
        raise RuntimeError(
            "A3 expects safe residual logits [B,2,H,W], got "
            f"{tuple(logits.shape)}."
        )
    if masks.ndim == 4:
        target = (masks[:, :1] > 0.5).to(logits.dtype)
    else:
        target = (masks > 0.5).to(logits.dtype)[:, None]

    base_prob = torch.sigmoid(candidates[:, :1]).detach()
    base_hard = (base_prob >= 0.5).to(target.dtype)
    c10_prob = torch.sigmoid(logits[:, 1:2])
    error = (base_hard != target).to(target.dtype)
    zero = logits.sum() * 0.0

    # Train-time direction target. C10 may only request a non-Preserve
    # intervention when one polarity clearly dominates within the small local
    # context carrier domain.
    inner_radius = max(
        1, int(_cfg(cfg.M1, "SAFE_CONTEXT_INNER_RADIUS", 1))
    )
    outer_radius = max(
        1, int(_cfg(cfg.M1, "SAFE_CONTEXT_OUTER_RADIUS", 3))
    )
    inner = (
        base_hard
        - (-F.max_pool2d(
            -base_hard,
            kernel_size=2 * inner_radius + 1,
            stride=1,
            padding=inner_radius,
        ))
    ).clamp(0.0, 1.0)
    outer = (
        F.max_pool2d(
            base_hard,
            kernel_size=2 * outer_radius + 1,
            stride=1,
            padding=outer_radius,
        )
        - base_hard
    ).clamp(0.0, 1.0)

    fill_need = ((1.0 - base_hard) * target * outer).sum(
        dim=(-2, -1)
    )[:, 0]
    delete_need = (base_hard * (1.0 - target) * inner).sum(
        dim=(-2, -1)
    )[:, 0]
    total_need = fill_need + delete_need
    dominance = (fill_need - delete_need).abs() / total_need.clamp_min(1.0)
    dominance_min = float(
        _cfg(cfg.M1, "SAFE_CONTEXT_DIRECTION_DOMINANCE_MIN", 0.50)
    )
    min_pixels = max(
        1,
        int(_cfg(cfg.M1, "SAFE_CONTEXT_DIRECTION_MIN_PIXELS", 4)),
    )
    clear = (
        (total_need >= float(min_pixels))
        & (dominance >= dominance_min)
    )
    direction_target = (fill_need > delete_need).to(logits.dtype)

    certificate_probability = certificate_probability.reshape(-1)
    direction_probability = direction_probability.reshape(-1)
    certificate_loss = _safe_probability_bce(
        certificate_probability,
        clear.to(logits.dtype),
        max_pos_weight=float(
            _cfg(cfg.M1, "SAFE_DIRECTION_CERTIFICATE_MAX_POS_WEIGHT", 20.0)
        ),
    )
    direction_loss = (
        F.binary_cross_entropy(
            direction_probability[clear],
            direction_target[clear],
        )
        if clear.any()
        else zero
    )

    # Pixel reconstruction is restricted to the C10 carrier, with the frozen
    # B0 error pixels upweighted. It improves the proposed edit without
    # introducing any GT-dependent inference input.
    error_weight = float(_cfg(cfg.M1, "SAFE_ERROR_WEIGHT", 4.0))
    context_weight = (
        0.10 + context_carrier * (1.0 + error_weight * error)
    )
    region_bce = (
        F.binary_cross_entropy_with_logits(
            logits[:, 1:2],
            target,
            reduction="none",
        )
        * context_weight
    ).sum() / context_weight.sum().clamp_min(1.0)
    region_dice = 1.0 - _safe_residual_dice(
        c10_prob,
        target,
        0.10 + context_carrier,
    ).mean()

    base_dice = _safe_residual_dice(base_prob, target)[:, 0]
    c10_dice = _safe_residual_dice(c10_prob, target)[:, 0]
    global_dice_gain = c10_dice - base_dice
    global_dice_margin = float(
        _cfg(cfg.M1, "SAFE_CONTEXT_GLOBAL_DICE_MARGIN", 0.0)
    )
    global_dice_no_regression = F.relu(
        global_dice_margin - global_dice_gain
    ).mean()

    base_boundary = _safe_residual_dice(
        _safe_residual_boundary(base_prob),
        _safe_residual_boundary(target),
    )[:, 0]
    c10_boundary = _safe_residual_dice(
        _safe_residual_boundary(c10_prob),
        _safe_residual_boundary(target),
    )[:, 0]
    global_boundary_gain = c10_boundary - base_boundary
    global_boundary_margin = float(
        _cfg(cfg.M1, "SAFE_CONTEXT_GLOBAL_BOUNDARY_MARGIN", 0.0)
    )
    global_boundary_no_regression = F.relu(
        global_boundary_margin - global_boundary_gain
    ).mean()

    # Only clinically/geometry-unambiguous cases request an improvement. The
    # old A2 loss asked C10 to improve any error-bearing case, including cases
    # containing contradictory fill and delete needs.
    local_gain = c10_dice - base_dice
    present = (
        clear
        & (
            (error * context_carrier)
            .sum(dim=(-2, -1))[:, 0]
            > 0.5
        )
    )
    improve_margin = float(
        _cfg(cfg.M1, "SAFE_SOFT_IMPROVEMENT_MARGIN", 0.0005)
    )
    improvement = (
        (
            F.relu(improve_margin - local_gain)
            * present.to(logits.dtype)
        ).sum()
        / present.sum().clamp_min(1.0)
    )

    local_base_boundary = _safe_residual_dice(
        _safe_residual_boundary(base_prob),
        _safe_residual_boundary(target),
        0.10 + context_carrier,
    )[:, 0]
    local_c10_boundary = _safe_residual_dice(
        _safe_residual_boundary(c10_prob),
        _safe_residual_boundary(target),
        0.10 + context_carrier,
    )[:, 0]
    local_boundary_margin = float(
        _cfg(
            cfg.M1,
            "SAFE_CONTEXT_BOUNDARY_IMPROVEMENT_MARGIN",
            0.0,
        )
    )
    local_boundary_improvement = (
        (
            F.relu(
                local_boundary_margin
                - (local_c10_boundary - local_base_boundary)
            )
            * present.to(logits.dtype)
        ).sum()
        / present.sum().clamp_min(1.0)
    )

    # Preserve correct B0 decisions. This is directed: deletion of correct
    # foreground and filling of correct background are both penalised.
    correct = (base_hard == target).to(logits.dtype)
    wrong_motion = (
        target * F.relu(base_prob - c10_prob)
        + (1.0 - target) * F.relu(c10_prob - base_prob)
    )
    correct_preserve = (
        wrong_motion * correct * (0.10 + context_carrier)
    ).sum() / (
        correct * (0.10 + context_carrier)
    ).sum().clamp_min(1.0)

    low = float(_cfg(cfg.M1, "SAFE_PRESERVE_LOW", 0.15))
    high = float(_cfg(cfg.M1, "SAFE_PRESERVE_HIGH", 0.85))
    confident_correct = (
        ((base_prob <= low) & (target < 0.5))
        | ((base_prob >= high) & (target >= 0.5))
    ).to(logits.dtype)
    confident_preserve = (
        (c10_prob - base_prob).abs() * confident_correct
    ).sum() / confident_correct.sum().clamp_min(1.0)

    flip_margin = float(
        _cfg(cfg.M1, "SAFE_HARD_FLIP_MARGIN", 0.06)
    )
    protected_fg = confident_correct * target
    protected_bg = confident_correct * (1.0 - target)
    hard_flip = (
        F.relu((0.50 + flip_margin) - c10_prob) * protected_fg
    ).sum() + (
        F.relu(c10_prob - (0.50 - flip_margin)) * protected_bg
    ).sum()
    hard_flip = hard_flip / (
        protected_fg.sum() + protected_bg.sum()
    ).clamp_min(1.0)

    outside = (
        (c10_prob - base_prob).abs() * (1.0 - context_carrier)
    ).mean()
    change = (c10_prob - base_prob).abs().mean(dim=(-2, -1))[:, 0]
    change_budget = float(
        _cfg(cfg.M1, "SAFE_CONTEXT_CHANGE_BUDGET", 0.0020)
    )
    budget = F.relu(change - change_budget).mean()

    total = (
        float(_cfg(cfg.M1, "SAFE_REGION_BCE_WEIGHT", 1.0))
        * region_bce
        + float(_cfg(cfg.M1, "SAFE_REGION_DICE_WEIGHT", 1.0))
        * region_dice
        + float(_cfg(cfg.M1, "SAFE_CONTEXT_GLOBAL_DICE_WEIGHT", 3.0))
        * global_dice_no_regression
        + float(
            _cfg(cfg.M1, "SAFE_CONTEXT_GLOBAL_BOUNDARY_WEIGHT", 4.0)
        )
        * global_boundary_no_regression
        + float(
            _cfg(cfg.M1, "SAFE_CONTEXT_BOUNDARY_IMPROVEMENT_WEIGHT", 2.0)
        )
        * local_boundary_improvement
        + float(
            _cfg(cfg.M1, "SAFE_CONTEXT_CORRECT_PRESERVE_WEIGHT", 4.0)
        )
        * correct_preserve
        + float(_cfg(cfg.M1, "SAFE_PRESERVE_WEIGHT", 3.0))
        * confident_preserve
        + float(_cfg(cfg.M1, "SAFE_HARD_FLIP_WEIGHT", 6.0))
        * hard_flip
        + float(_cfg(cfg.M1, "SAFE_CONTEXT_OUTSIDE_WEIGHT", 3.0))
        * outside
        + float(_cfg(cfg.M1, "SAFE_BUDGET_WEIGHT", 4.0))
        * budget
        + float(_cfg(cfg.M1, "SAFE_DIRECTION_WEIGHT", 1.0))
        * direction_loss
        + float(
            _cfg(cfg.M1, "SAFE_DIRECTION_CERTIFICATE_WEIGHT", 1.0)
        )
        * certificate_loss
        + float(_cfg(cfg.M1, "SAFE_IMPROVEMENT_WEIGHT", 0.75))
        * improvement
    )

    hard_base = base_prob >= 0.5
    return total, {
        "safe_residual_loss": total.detach(),
        "safe_region_bce": region_bce.detach(),
        "safe_region_dice": region_dice.detach(),
        "safe_context_global_dice_no_regression": (
            global_dice_no_regression.detach()
        ),
        "safe_context_global_boundary_no_regression": (
            global_boundary_no_regression.detach()
        ),
        "safe_context_local_boundary_improvement": (
            local_boundary_improvement.detach()
        ),
        "safe_context_correct_preserve": correct_preserve.detach(),
        "safe_confident_preserve": confident_preserve.detach(),
        "safe_hard_flip_loss": hard_flip.detach(),
        "safe_context_outside": outside.detach(),
        "safe_budget_loss": budget.detach(),
        "safe_direction_loss": direction_loss.detach(),
        "safe_direction_certificate_loss": certificate_loss.detach(),
        "safe_direction_clear_fraction": clear.float().mean().detach(),
        "safe_direction_dominance": dominance.mean().detach(),
        "safe_improvement_loss": improvement.detach(),
        "safe_c10_soft_dice_gain": global_dice_gain.mean().detach(),
        "safe_c10_soft_boundary_gain": global_boundary_gain.mean().detach(),
        "safe_c10_hard_change": (
            ((c10_prob >= 0.5) != hard_base).float().mean().detach()
        ),
        "safe_c10_mean_change": change.mean().detach(),
        "safe_context_certificate_mean": (
            certificate_probability.mean().detach()
        ),
        "safe_context_near_threshold_mean": (
            near_threshold_gate.mean().detach()
        ),
    }


def compute_safe_residual_candidate_loss(cfg, candidates, masks, aux):
    """Dispatch legacy A1/A2 or certified A3 safety objective."""
    if bool(_cfg(cfg.M1, "SAFE_CONTEXT_CERTIFIED_ONLY", False)):
        return _compute_safe_context_certified_loss(
            cfg,
            candidates,
            masks,
            aux,
        )
    return _compute_safe_residual_candidate_loss_legacy(
        cfg,
        candidates,
        masks,
        aux,
    )


