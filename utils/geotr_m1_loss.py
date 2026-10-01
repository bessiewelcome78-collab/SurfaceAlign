"""Exact objective used by the previously successful GEOTR Stage-1 M1."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F


EPS = 1.0e-4


def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def _target3(masks: torch.Tensor, hw) -> torch.Tensor:
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise ValueError(f"Expected masks [B,H,W] or [B,1,H,W], got {tuple(masks.shape)}")
    target = (masks > 0.5).float()
    if tuple(target.shape[-2:]) != tuple(hw):
        target = F.interpolate(target[:, None], size=hw, mode="nearest")[:, 0]
    return target


def _dice_per_case(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prob.ndim == 4 and prob.shape[1] == 1:
        prob = prob[:, 0]
    inter = (prob * target).sum(dim=(-2, -1))
    den = prob.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return (2.0 * inter + EPS) / (den + EPS)


def _flow_smoothness(flow: torch.Tensor) -> torch.Tensor:
    dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    return 0.5 * (dx.abs().mean() + dy.abs().mean())


def _edge_aware_flow_smoothness(
    flow: torch.Tensor,
    base_probability: torch.Tensor,
    power: float,
    floor: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Regularize background flow while preserving boundary flexibility.

    The Base uncertainty/boundary band is detached: it defines where a
    correction is plausible but cannot turn the deformation term into another
    gradient path to Base.  A non-zero floor retains weak control inside the
    band and avoids an unconstrained seam.
    """
    if base_probability.ndim == 3:
        base_probability = base_probability[:, None]
    probability = base_probability.detach().clamp(0.0, 1.0)
    uncertainty = (4.0 * probability * (1.0 - probability)).clamp(0.0, 1.0)
    band = torch.maximum(_soft_boundary(probability), uncertainty)
    stiffness = float(floor) + (1.0 - float(floor)) * (
        1.0 - band
    ).pow(max(float(power), 0.0))

    dx = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs().mean(dim=1)
    dy = (flow[:, :, 1:, :] - flow[:, :, :-1, :]).abs().mean(dim=1)
    wx = 0.5 * (stiffness[:, 0, :, 1:] + stiffness[:, 0, :, :-1])
    wy = 0.5 * (stiffness[:, 0, 1:, :] + stiffness[:, 0, :-1, :])
    dx_term = (dx * wx).sum() / wx.sum().clamp_min(EPS)
    dy_term = (dy * wy).sum() / wy.sum().clamp_min(EPS)
    return 0.5 * (dx_term + dy_term), band.mean()


def _soft_boundary(probability: torch.Tensor) -> torch.Tensor:
    if probability.ndim == 3:
        probability = probability[:, None]
    maximum = F.max_pool2d(probability, 3, stride=1, padding=1)
    minimum = -F.max_pool2d(-probability, 3, stride=1, padding=1)
    return (maximum - minimum).clamp(0.0, 1.0)


def _boundary_dice_loss(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    predicted_boundary = _soft_boundary(probability)
    target_boundary = _soft_boundary(target[:, None])
    intersection = (predicted_boundary * target_boundary).sum(dim=(1, 2, 3))
    denominator = predicted_boundary.sum(dim=(1, 2, 3)) + target_boundary.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + EPS) / (denominator + EPS)).mean()


def _surface_band_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    outer_radius_px: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GT-surface-focused BCE aligned to a small 2-D boundary tolerance.

    A nested 1/2/R pixel band emphasizes mistakes close to the true contour
    without duplicating the whole-image BCE.  Empty-GT cases legitimately have
    no surface-band term; the ordinary BCE/Dice objectives still supervise them.
    """
    if logits.ndim == 3:
        logits = logits[:, None]
    target4 = target[:, None].to(logits)
    boundary = _soft_boundary(target4).detach()
    radius = max(1, int(outer_radius_px))

    def dilate(x: torch.Tensor, r: int) -> torch.Tensor:
        r = max(0, int(r))
        if r == 0:
            return x
        k = 2 * r + 1
        return F.max_pool2d(x, kernel_size=k, stride=1, padding=r)

    b1 = dilate(boundary, 1)
    b2 = dilate(boundary, min(2, radius))
    br = dilate(boundary, radius)
    # Nearer contour pixels receive stronger supervision.  The bands are
    # disjoint after subtraction, so a pixel cannot be counted multiple times.
    w = (3.0 * b1 + 2.0 * (b2 - b1).clamp_min(0.0) + (br - b2).clamp_min(0.0)).detach()
    per_pixel = F.binary_cross_entropy_with_logits(
        logits.float(), target4.float(), reduction="none"
    ).to(logits)
    numerator = (per_pixel * w).flatten(1).sum(dim=1)
    denominator = w.flatten(1).sum(dim=1)
    valid = denominator > 0
    if valid.any():
        loss = (numerator[valid] / denominator[valid].clamp_min(EPS)).mean()
    else:
        loss = logits.sum() * 0.0
    band_fraction = (w > 0).float().mean()
    return loss, band_fraction


def _jbt_error_supervision_loss(
    cfg,
    aux: Dict[str, torch.Tensor],
    base_probability: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Class-balanced soft FN/FP supervision concentrated near Base errors.

    v3 used an unweighted whole-image BCE.  On sparse boundary residuals that
    objective is dominated by easy true negatives and drives both error channels
    toward the prior.  v4 balances each channel by its soft positive mass and
    gives more weight to Base/GT boundary support, without consulting dataset
    identity or Test labels.
    """
    logits = aux.get("geotr_m1_error_logits")
    if not isinstance(logits, torch.Tensor):
        zero = base_probability.sum() * 0.0
        return zero, {
            "fn_target_mean": zero.detach(),
            "fp_target_mean": zero.detach(),
            "error_prob_mean": zero.detach(),
            "error_pos_weight_mean": zero.detach(),
        }
    m1 = _cfg_get(cfg, "M1", None)
    target4 = target[:, None].to(base_probability)
    base = base_probability.detach().clamp(0.0, 1.0)
    fn_target = (target4 * (1.0 - base)).detach()
    fp_target = ((1.0 - target4) * base).detach()
    target_pair = torch.cat([fn_target, fp_target], dim=1)

    positive_mass = target_pair.mean(dim=(0, 2, 3)).clamp_min(1.0e-5)
    pos_cap = max(1.0, float(_cfg_get(m1, "JBT_ERROR_POS_WEIGHT_MAX", 12.0)))
    pos_weight = ((1.0 - positive_mass) / positive_mass).clamp(1.0, pos_cap)
    per_pixel = F.binary_cross_entropy_with_logits(
        logits.float(), target_pair.float(), reduction="none"
    ).to(base_probability)
    class_weight = 1.0 + (pos_weight[None, :, None, None].to(per_pixel) - 1.0) * target_pair
    base_support = torch.maximum(_soft_boundary(base), 4.0 * base * (1.0 - base))
    gt_support = _soft_boundary(target4).detach()
    support = torch.maximum(base_support.detach(), gt_support)
    spatial_weight = 0.25 + 0.75 * support
    gamma = max(0.0, float(_cfg_get(m1, "JBT_ERROR_FOCAL_GAMMA", 1.0)))
    if gamma > 0.0:
        probs = torch.sigmoid(logits.detach())
        focal = (probs - target_pair).abs().clamp_min(0.05).pow(gamma)
    else:
        focal = torch.ones_like(per_pixel)
    weight = class_weight * spatial_weight * focal
    loss = (per_pixel * weight).sum() / weight.sum().clamp_min(EPS)
    probs = torch.sigmoid(logits.detach())
    return loss, {
        "fn_target_mean": fn_target.mean(),
        "fp_target_mean": fp_target.mean(),
        "error_prob_mean": probs.mean(),
        "error_pos_weight_mean": pos_weight.mean().detach(),
    }


def _jbt_case_utility_loss(
    cfg,
    aux: Dict[str, torch.Tensor],
    base_probability: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Risk-aware outcome supervision for Preserve vs multi-strength edits.

    JBT-v4 trained one binary ``benefit`` probability for one moving candidate.
    When the candidate was nearly neutral, the head converged below a fixed 0.5
    threshold and deployment preserved every BUSI/Kvasir case.  JBT-v5 instead
    labels each shared-field strength as benefit / neutral / harm.  Inference
    scores benefit minus a configurable harm risk and always keeps Preserve as
    the exact zero-utility action.
    """
    logits = aux.get("jbt_case_utility_logits")
    strength_probs = aux.get("jbt_v5_strength_candidate_probs")
    if (
        isinstance(logits, torch.Tensor)
        and logits.ndim == 3
        and logits.shape[-1] >= 3
        and isinstance(strength_probs, torch.Tensor)
        and strength_probs.ndim == 4
        and strength_probs.shape[1] == logits.shape[1] + 1
    ):
        m1 = _cfg_get(cfg, "M1", None)
        base_dice = _dice_per_case(base_probability.detach(), target).detach()
        nonbase = strength_probs[:, 1:]
        b, k, h, w = nonbase.shape
        target_expanded = target[:, None].expand(b, k, *target.shape[-2:])
        inter = (nonbase * target_expanded).flatten(2).sum(dim=2)
        den = nonbase.flatten(2).sum(dim=2) + target_expanded.flatten(2).sum(dim=2)
        candidate_dice = (2.0 * inter + EPS) / (den + EPS)
        gains = (candidate_dice.detach() - base_dice[:, None]).detach()
        margin = max(0.0, float(_cfg_get(m1, "JBT_CASE_UTILITY_GAIN_MARGIN", 0.001)))
        labels = torch.ones_like(gains, dtype=torch.long)  # neutral=1
        labels = torch.where(gains > margin, torch.zeros_like(labels), labels)  # benefit=0
        labels = torch.where(gains < -margin, torch.full_like(labels, 2), labels)  # harm=2

        class_logits = logits[..., :3]
        flat_labels = labels.reshape(-1)
        flat_logits = class_logits.reshape(-1, 3)
        counts = torch.stack([(flat_labels == c).float().sum() for c in range(3)])
        total = float(max(flat_labels.numel(), 1))
        # Balanced but bounded weights: neutral remains informative without
        # drowning the sparse beneficial/harmful counterfactuals.
        class_w = (total / (3.0 * counts.clamp_min(1.0))).clamp(0.35, 6.0).to(flat_logits)
        ce = F.cross_entropy(flat_logits, flat_labels, weight=class_w)

        # v6: classification alone was too coarse when counterfactual gains were
        # only a few 1e-3.  Add continuous gain calibration and within-case
        # pairwise ranking, while keeping the editor detached from this loss.
        outcome_live = torch.softmax(class_logits, dim=-1)
        utility_live = outcome_live[..., 0] - outcome_live[..., 2]
        gain_scale = max(1.0e-4, float(_cfg_get(m1, "JBT_V6_UTILITY_GAIN_SCALE", 0.02)))
        v62 = bool(_cfg_get(m1, "JBT_V62_ENABLED", False)) and logits.shape[-1] >= 5
        if v62:
            gain_scale = max(1.0e-4, float(
                _cfg_get(m1, "JBT_V62_GAIN_SCALE", gain_scale)
            ))
            gain_mean = gain_scale * torch.tanh(logits[..., 3])
            gain_sigma = gain_scale * (F.softplus(logits[..., 4]) + 1.0e-3)
            v63 = bool(_cfg_get(m1, "JBT_V63_ENABLED", False))
            if v63:
                # v6.2's heteroscedastic NLL learned sigma=0.0115 on BUSI while
                # the complete candidate Oracle was only 0.0062.  Subtracting
                # that sigma at deployment forced 0% acceptance.  v6.3 directly
                # calibrates expected gain in normalized Dice units; sigma is
                # retained for diagnostics but no longer gets rewarded as an
                # escape route for regression error.
                utility_reg = F.smooth_l1_loss(
                    gain_mean / gain_scale,
                    gains / gain_scale,
                    beta=max(1.0e-3, float(_cfg_get(
                        m1, "JBT_V63_GAIN_HUBER_BETA", 0.05
                    ))),
                )
            else:
                normalized_error = (gains - gain_mean) / gain_sigma.clamp_min(1.0e-5)
                # Proper heteroscedastic regression retained for exact v6.2
                # compatibility.
                utility_reg = (
                    0.5 * normalized_error.square()
                    + torch.log(gain_sigma.clamp_min(1.0e-5) / gain_scale)
                ).mean()
            rank_source = gain_mean
            target_utility = (gains / gain_scale).clamp(-1.0, 1.0)
        else:
            target_utility = torch.tanh(gains / gain_scale)
            utility_reg = F.smooth_l1_loss(utility_live, target_utility, beta=0.20)
            rank_source = utility_live
        gain_gap = gains[:, :, None] - gains[:, None, :]
        score_gap = rank_source[:, :, None] - rank_source[:, None, :]
        pair_mask = gain_gap.abs() > max(margin, 2.5e-4)
        rank_temp = max(1.0e-5, float(_cfg_get(
            m1,
            "JBT_V62_RANK_TEMPERATURE" if v62 else "JBT_V6_UTILITY_RANK_TEMPERATURE",
            0.002 if v62 else 0.20,
        )))
        if bool(pair_mask.any()):
            pair_sign = gain_gap.sign()
            rank_term = F.softplus(-pair_sign * score_gap / rank_temp)
            utility_rank = rank_term[pair_mask].mean()
        else:
            utility_rank = ce * 0.0
        reg_weight = max(0.0, float(_cfg_get(m1, "JBT_V6_UTILITY_REG_WEIGHT", 0.50)))
        rank_weight = max(0.0, float(_cfg_get(m1, "JBT_V6_UTILITY_RANK_WEIGHT", 0.50)))
        # Explicitly rank every candidate against Preserve=0.  The v6 loss only
        # ranked non-Preserve candidates against one another, so a perfectly
        # ordered but uniformly harmful bank could still be accepted.
        if v62:
            preserve_mask = gains.abs() > max(margin, 2.5e-4)
            preserve_rank_all = F.softplus(
                -gains.sign() * gain_mean / rank_temp
            )
            preserve_rank = (
                preserve_rank_all[preserve_mask].mean()
                if bool(preserve_mask.any()) else ce * 0.0
            )
            preserve_rank_weight = max(0.0, float(
                _cfg_get(m1, "JBT_V62_PRESERVE_RANK_WEIGHT", 1.0)
            ))
        else:
            preserve_rank = ce * 0.0
            preserve_rank_weight = 0.0
        utility_loss = (
            ce + reg_weight * utility_reg + rank_weight * utility_rank
            + preserve_rank_weight * preserve_rank
        )

        outcome = torch.softmax(class_logits.detach(), dim=-1)
        risk_lambda = max(0.0, float(_cfg_get(m1, "JBT_V5_UTILITY_RISK_LAMBDA", 1.5)))
        if v62:
            lcb_lambda = max(0.0, float(_cfg_get(m1, "JBT_V62_LCB_LAMBDA", 1.0)))
            harm_penalty = max(0.0, float(_cfg_get(m1, "JBT_V62_HARM_PROB_PENALTY", 0.25)))
            if bool(_cfg_get(m1, "JBT_V63_ENABLED", False)):
                class_weight = max(0.0, float(_cfg_get(
                    m1, "JBT_V63_CLASS_SCORE_WEIGHT", 0.25
                )))
                v63_harm_penalty = max(0.0, float(_cfg_get(
                    m1, "JBT_V63_HARM_PROB_PENALTY", 0.05
                )))
                scores = (
                    gain_mean.detach()
                    + class_weight * gain_scale
                    * (outcome[..., 0] - outcome[..., 2])
                    - v63_harm_penalty * gain_scale * outcome[..., 2]
                )
                accept_threshold = float(_cfg_get(
                    m1, "JBT_V63_ACCEPT_MARGIN", 0.0
                ))
            else:
                scores = (
                    gain_mean.detach()
                    - lcb_lambda * gain_sigma.detach()
                    - harm_penalty * gain_scale * outcome[..., 2]
                )
                accept_threshold = max(0.0, float(
                    _cfg_get(m1, "JBT_V62_PRESERVE_MARGIN", 5.0e-4)
                ))
        else:
            scores = outcome[..., 0] - risk_lambda * outcome[..., 2]
            accept_threshold = float(_cfg_get(m1, "JBT_CASE_UTILITY_THRESHOLD", 0.05))
        best_score, best_idx = scores.max(dim=1)
        best_gain = gains.gather(1, best_idx[:, None])[:, 0]
        accept = best_score > accept_threshold
        selected_gain = torch.where(accept, best_gain, torch.zeros_like(best_gain))
        oracle_gain = gains.clamp_min(0.0).max(dim=1).values
        realization = torch.where(
            oracle_gain > 1.0e-6,
            selected_gain.clamp_min(0.0) / oracle_gain.clamp_min(1.0e-6),
            torch.zeros_like(oracle_gain),
        )
        zero = base_probability.sum() * 0.0
        return utility_loss, {
            "case_gain_mean": gains[:, min(1, k - 1)].mean() if k else zero.detach(),
            "utility_classification_loss": ce.detach(),
            "utility_regression_loss": utility_reg.detach(),
            "utility_ranking_loss": utility_rank.detach(),
            "utility_preserve_ranking_loss": preserve_rank.detach(),
            "gain_mean_mae": (
                (gain_mean.detach() - gains).abs().mean()
                if v62 else zero.detach()
            ),
            "gain_sigma_mean": (
                gain_sigma.detach().mean() if v62 else zero.detach()
            ),
            "lcb_score_mean": scores.mean().detach(),
            "utility_target_abs_mean": target_utility.detach().abs().mean(),
            "case_benefit_target_rate": (labels == 0).float().mean(),
            "case_harm_target_rate": (labels == 2).float().mean(),
            "case_neutral_rate": (labels == 1).float().mean(),
            "case_accept_prob_mean": outcome[..., 0].mean(),
            "strength_case_oracle_gain": oracle_gain.mean(),
            "strength_any_benefit_rate": (oracle_gain > margin).float().mean(),
            "utility_selected_gain": selected_gain.mean(),
            "oracle_realization_ratio": realization.mean(),
            "utility_accept_rate": accept.float().mean(),
        }

    # Backward-compatible v4 single-candidate path.
    raw_probability = aux.get("geotr_m1_raw_geometry_probs")
    if not isinstance(logits, torch.Tensor) or not isinstance(raw_probability, torch.Tensor):
        zero = base_probability.sum() * 0.0
        return zero, {
            "case_gain_mean": zero.detach(),
            "case_benefit_target_rate": zero.detach(),
            "case_accept_prob_mean": zero.detach(),
        }
    m1 = _cfg_get(cfg, "M1", None)
    base_dice = _dice_per_case(base_probability.detach(), target).detach()
    raw_dice = _dice_per_case(raw_probability, target)
    gain = (raw_dice.detach() - base_dice).detach()
    margin = max(0.0, float(_cfg_get(m1, "JBT_CASE_UTILITY_GAIN_MARGIN", 0.001)))
    positive = gain > margin
    harmful = gain < -margin
    neutral = ~(positive | harmful)
    labels = positive.to(logits.dtype)
    pos_count = positive.float().sum().clamp_min(1.0)
    neg_count = harmful.float().sum().clamp_min(1.0)
    pos_w = (0.5 * logits.numel() / pos_count).clamp(0.5, 6.0)
    neg_w = (0.5 * logits.numel() / neg_count).clamp(0.5, 6.0)
    weights = torch.where(positive, pos_w, torch.where(harmful, neg_w, logits.new_tensor(0.25)))
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    loss = (bce * weights).sum() / weights.sum().clamp_min(EPS)
    return loss, {
        "case_gain_mean": gain.mean(),
        "case_benefit_target_rate": positive.float().mean(),
        "case_harm_target_rate": harmful.float().mean(),
        "case_accept_prob_mean": torch.sigmoid(logits.detach()).mean(),
        "case_neutral_rate": neutral.float().mean(),
    }


def _jbt_v5_oracle_capacity_loss(
    cfg,
    aux: Dict[str, torch.Tensor],
    base_probability: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Make at least one shared-field strength a strong counterfactual.

    Candidate generation and safe execution have different jobs.  This loss is
    intentionally optimistic: for each training case it backpropagates through
    the currently best non-Preserve strength.  The utility/risk head, not a tiny
    proposal band, is responsible for refusing harmful candidates at inference.
    """
    probs = aux.get("jbt_v5_strength_candidate_probs")
    if not isinstance(probs, torch.Tensor) or probs.ndim != 4 or probs.shape[1] < 2:
        z = base_probability.sum() * 0.0
        return z, {"oracle_gain": z.detach(), "oracle_benefit_rate": z.detach()}
    nonbase = probs[:, 1:]
    b, k, h, w = nonbase.shape
    gt = target[:, None].expand(b, k, *target.shape[-2:]).to(nonbase)
    inter = (nonbase * gt).flatten(2).sum(dim=2)
    den = nonbase.flatten(2).sum(dim=2) + gt.flatten(2).sum(dim=2)
    dice = (2.0 * inter + EPS) / (den + EPS)
    best_idx = dice.detach().argmax(dim=1)
    best = dice.gather(1, best_idx[:, None])[:, 0]
    base_d = _dice_per_case(base_probability.detach(), target).detach()
    gain = best.detach() - base_d
    # Dice-first capacity objective plus an explicit Base-relative gain margin.
    # v5 optimized absolute candidate Dice; for a strong Base, near-identity
    # candidates already have low loss and the editor can settle at sub-pixel
    # motion.  v6 explicitly asks the best counterfactual to beat Base by a
    # small target margin whenever possible.
    m1 = _cfg_get(cfg, "M1", None)
    target_gain = max(0.0, float(_cfg_get(m1, "JBT_V6_CAPACITY_TARGET_GAIN", 0.01)))
    target_dice = (base_d + target_gain).clamp(max=1.0)
    margin_penalty = F.relu(target_dice - best).mean()
    absolute_weight = max(0.0, float(_cfg_get(m1, "JBT_V6_CAPACITY_ABSOLUTE_WEIGHT", 0.25)))
    loss = margin_penalty + absolute_weight * (1.0 - best).mean()
    margin = max(0.0, float(_cfg_get(m1, "JBT_CASE_UTILITY_GAIN_MARGIN", 0.001)))
    return loss, {
        "capacity_margin_penalty": margin_penalty.detach(),
        "capacity_target_gain": best.detach().new_tensor(target_gain),
        "oracle_gain": gain.clamp_min(0.0).mean(),
        "oracle_signed_gain": gain.mean(),
        "oracle_benefit_rate": (gain > margin).float().mean(),
    }

def _jbt_direction_alignment_loss(
    aux: Dict[str, torch.Tensor],
    base_probability: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Teach normal transport sign: FN expands, FP contracts."""
    scalar = aux.get("geotr_m1_scalar_field")
    if not isinstance(scalar, torch.Tensor):
        return base_probability.sum() * 0.0
    target4 = target[:, None].to(base_probability)
    base = base_probability.detach().clamp(0.0, 1.0)
    signed = (target4 - base).detach()
    weight = signed.abs()
    sign = signed.sign()
    # softplus(-s*d) is small when scalar displacement has the desired sign.
    per_pixel = F.softplus(-scalar.float() * sign.float()).to(base_probability)
    return (per_pixel * weight).sum() / weight.sum().clamp_min(EPS)


def _jbt_v6_signed_displacement_loss(
    cfg,
    aux: Dict[str, torch.Tensor],
    base_probability: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Supervise the inverse-warp displacement on the deployed Base-normal ray.

    v6 used an isotropic nearest-boundary distance and assigned its sign from
    the GT class at the Base contour.  That target can point tangentially while
    deployment can move only along the Base normal.  v6.2 samples the GT contour
    on x+d*n(x), selects the closest ray intersection, and uses -d because
    ``grid_sample`` is an inverse warp: sampling inward moves the output contour
    outward.  GT is used only to build this training target.
    """
    m1 = _cfg_get(cfg, "M1", None)
    enabled = bool(_cfg_get(m1, "JBT_V6_SIGNED_DISPLACEMENT_SUPERVISION", False))
    scalar = aux.get("geotr_m1_scalar_field")
    if (not enabled) or not isinstance(scalar, torch.Tensor):
        z = base_probability.sum() * 0.0
        return z, {
            "signed_disp_target_abs_mean": z.detach(),
            "signed_disp_pred_abs_mean": z.detach(),
            "signed_disp_direction_accuracy": z.detach(),
            "signed_disp_owner_fraction": z.detach(),
            "normal_ray_hit_fraction": z.detach(),
            "normal_ray_target_enabled": z.detach(),
        }
    radius = max(1, int(_cfg_get(m1, "JBT_V6_DISPLACEMENT_RADIUS_PX", 8)))
    target4 = target[:, None].to(base_probability)
    base_hard = (base_probability.detach() >= 0.5).to(base_probability)
    gt_hard = (target4 >= 0.5).to(base_probability)
    base_boundary = (_soft_boundary(base_hard) > 0.0).to(base_probability)
    gt_boundary = (_soft_boundary(gt_hard) > 0.0).to(base_probability)
    v62 = bool(_cfg_get(m1, "JBT_V62_ENABLED", False))
    normal = aux.get("geotr_m1_normal_field")
    if v62 and isinstance(normal, torch.Tensor):
        b, _, h, w = base_probability.shape
        step = max(0.25, float(_cfg_get(m1, "JBT_V62_NORMAL_RAY_STEP_PX", 0.5)))
        offsets = torch.arange(
            -float(radius), float(radius) + 0.5 * step, step,
            device=base_probability.device, dtype=base_probability.dtype,
        )
        yy = torch.arange(h, device=base_probability.device, dtype=base_probability.dtype) + 0.5
        xx = torch.arange(w, device=base_probability.device, dtype=base_probability.dtype) + 0.5
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        nx, ny = normal[:, 0].detach(), normal[:, 1].detach()
        samples = []
        for d in offsets:
            sx = gx[None] + d * nx
            sy = gy[None] + d * ny
            grid = torch.stack(
                [2.0 * sx / float(w) - 1.0, 2.0 * sy / float(h) - 1.0], dim=-1
            )
            samples.append(F.grid_sample(
                gt_boundary, grid, mode="bilinear",
                padding_mode="zeros", align_corners=False,
            ))
        ray_evidence = torch.cat(samples, dim=1)
        # Prefer the closest intersection when a thick contour produces ties.
        tie = 1.0e-3 * offsets.abs()[None, :, None, None] / float(max(radius, 1))
        best_index = (ray_evidence - tie).argmax(dim=1, keepdim=True)
        best_evidence = ray_evidence.gather(1, best_index)
        best_offset = offsets[best_index]
        signed_target = -best_offset
        min_hit = max(0.0, float(_cfg_get(m1, "JBT_V62_NORMAL_RAY_MIN_HIT", 0.10)))
        ray_hit = (best_evidence >= min_hit).to(base_probability)
        owner = base_boundary * ray_hit
        distance = signed_target.abs()
        ray_confidence = best_evidence.detach().clamp(0.0, 1.0)
    else:
        # Exact v6 compatibility path.
        distance = torch.full_like(gt_boundary, float(radius))
        reached = gt_boundary > 0.0
        distance = torch.where(reached, torch.zeros_like(distance), distance)
        dilated = gt_boundary
        for r in range(1, radius + 1):
            dilated = F.max_pool2d(dilated, 3, stride=1, padding=1)
            new = (dilated > 0.0) & (~reached)
            distance = torch.where(new, distance.new_full((), float(r)), distance)
            reached = reached | new
        sign = torch.where(gt_hard > 0.5, torch.ones_like(gt_hard), -torch.ones_like(gt_hard))
        signed_target = sign * distance
        owner = base_boundary
        ray_hit = torch.ones_like(owner)
        ray_confidence = torch.ones_like(owner)
    # If broad support exists, use it as a soft confidence multiplier without
    # removing Base-boundary owners completely.
    proposal_gate = aux.get("geotr_m1_change_gate")
    if isinstance(proposal_gate, torch.Tensor):
        owner_weight = owner * ray_confidence * (
            0.75 + 0.25 * proposal_gate.detach().clamp(0.0, 1.0)
            if v62 else 0.35 + 0.65 * proposal_gate.detach().clamp(0.0, 1.0)
        )
    else:
        owner_weight = owner
    pred = scalar
    beta = max(0.05, float(_cfg_get(m1, "JBT_V6_DISPLACEMENT_HUBER_BETA_PX", 1.0)))
    per = F.smooth_l1_loss(pred.float(), signed_target.float(), reduction="none", beta=beta).to(pred)
    valid_dir = owner * (distance > 0.5).to(owner)

    # v6.3.4: BUSI has long, heterogeneous contours and only ~0.9% valid ray
    # owners.  A global pixel reduction lets a few large lesions determine the
    # update and can learn the majority sign while losing small cases.  Give
    # every case with at least one owner equal weight, then add an explicitly
    # sign-balanced logistic objective on the operator-correct inverse-warp
    # target.  This targets the observed 47% direction accuracy without using
    # dataset-specific thresholds or Test labels.
    reduce_dims = (1, 2, 3)
    owner_mass = owner_weight.sum(dim=reduce_dims)
    magnitude_per_case = (per * owner_weight).sum(dim=reduce_dims) / owner_mass.clamp_min(EPS)
    active_case = owner_mass > 0.0
    case_balanced = bool(_cfg_get(m1, "JBT_V634_CASE_BALANCED_DISPLACEMENT", False))
    if case_balanced:
        magnitude_loss = (
            magnitude_per_case[active_case].mean()
            if bool(active_case.any()) else per.sum() * 0.0
        )
    else:
        magnitude_loss = (per * owner_weight).sum() / owner_mass.sum().clamp_min(EPS)

    sign_temperature = max(0.05, float(_cfg_get(
        m1, "JBT_V634_SIGN_TEMPERATURE_PX", 1.0
    )))
    sign_per = F.softplus(
        -pred.float() * signed_target.sign().float() / sign_temperature
    ).to(pred)
    positive = valid_dir * (signed_target > 0.0).to(valid_dir)
    negative = valid_dir * (signed_target < 0.0).to(valid_dir)
    positive_mass = positive.sum(dim=reduce_dims)
    negative_mass = negative.sum(dim=reduce_dims)
    positive_loss = (sign_per * positive).sum(dim=reduce_dims) / positive_mass.clamp_min(1.0)
    negative_loss = (sign_per * negative).sum(dim=reduce_dims) / negative_mass.clamp_min(1.0)
    positive_present = positive_mass > 0.0
    negative_present = negative_mass > 0.0
    present_count = positive_present.to(pred).add(negative_present.to(pred))
    sign_per_case = (
        positive_loss * positive_present.to(pred)
        + negative_loss * negative_present.to(pred)
    ) / present_count.clamp_min(1.0)
    sign_active = present_count > 0.0
    sign_loss = (
        sign_per_case[sign_active].mean()
        if bool(sign_active.any()) else pred.sum() * 0.0
    )
    sign_weight = max(0.0, float(_cfg_get(m1, "JBT_V634_SIGN_LOSS_WEIGHT", 0.0)))
    loss = magnitude_loss + sign_weight * sign_loss

    correct = ((pred.detach().sign() == signed_target.sign()).to(owner) * valid_dir).sum()
    direction_acc = correct / valid_dir.sum().clamp_min(1.0)
    return loss, {
        "signed_disp_target_abs_mean": (signed_target.abs() * owner).sum().detach() / owner.sum().clamp_min(1.0),
        "signed_disp_pred_abs_mean": (pred.detach().abs() * owner).sum() / owner.sum().clamp_min(1.0),
        "signed_disp_direction_accuracy": direction_acc.detach(),
        "signed_disp_owner_fraction": owner.mean().detach(),
        "normal_ray_hit_fraction": (
            (base_boundary * ray_hit).sum() / base_boundary.sum().clamp_min(1.0)
        ).detach(),
        "normal_ray_target_enabled": owner.new_tensor(1.0 if v62 else 0.0),
        "signed_disp_magnitude_loss": magnitude_loss.detach(),
        "signed_disp_balanced_sign_loss": sign_loss.detach(),
        "signed_disp_positive_owner_fraction": positive.mean().detach(),
        "signed_disp_negative_owner_fraction": negative.mean().detach(),
        "signed_disp_active_case_fraction": active_case.float().mean().detach(),
    }


def _jbt_nondegrade_and_preserve_losses(
    probability: torch.Tensor,
    base_probability: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Base-relative safety objective used only during training.

    The hinge prevents the residual branch from improving a surface term by
    sacrificing case-level Dice.  The preserve term discourages unnecessary
    probability motion where Base already agrees with GT.
    """
    base_dice = _dice_per_case(base_probability.detach(), target).detach()
    final_dice = _dice_per_case(probability, target)
    nondegrade = F.relu(base_dice - final_dice).mean()
    target4 = target[:, None].to(probability)
    correctness = (1.0 - (target4 - base_probability.detach()).abs()).clamp(0.0, 1.0)
    preserve = (
        (probability - base_probability.detach()).abs() * correctness.square()
    ).sum() / correctness.square().sum().clamp_min(EPS)
    return nondegrade, preserve


def _flow_folding_penalty(flow: torch.Tensor, minimum_jacobian: float) -> torch.Tensor:
    """Differentiable penalty for locally non-invertible transport fields."""
    if flow.shape[-2] < 2 or flow.shape[-1] < 2:
        return flow.sum() * 0.0
    ux, uy = flow[:, 0], flow[:, 1]
    dux_dx = ux[:, :-1, 1:] - ux[:, :-1, :-1]
    dux_dy = ux[:, 1:, :-1] - ux[:, :-1, :-1]
    duy_dx = uy[:, :-1, 1:] - uy[:, :-1, :-1]
    duy_dy = uy[:, 1:, :-1] - uy[:, :-1, :-1]
    determinant = (1.0 + dux_dx) * (1.0 + duy_dy) - dux_dy * duy_dx
    return F.relu(float(minimum_jacobian) - determinant).mean()


def _levelset_frame_and_reliability(
    base_logits: torch.Tensor,
    smooth_kernel: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Detached local level-set frame and per-pixel normal reliability.

    The frame is estimated from a lightly smoothed factual Base logit field.
    Reliability is normalized per image by the mean gradient magnitude, so
    tangent suppression is strong only where a meaningful level-set normal is
    actually observed.  This avoids the hard self-referential constraint of
    strict normal-1D transport in flat/noisy regions.
    """
    if base_logits.ndim == 3:
        base_logits = base_logits[:, None]
    k = max(1, int(smooth_kernel))
    if k % 2 == 0:
        k += 1
    work = base_logits.detach().float()
    if k > 1:
        work = F.avg_pool2d(work, kernel_size=k, stride=1, padding=k // 2)
    padded = F.pad(work, (1, 1, 1, 1), mode="replicate")
    gx = 0.5 * (padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2])
    gy = 0.5 * (padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1])
    mag = torch.sqrt(gx.square() + gy.square() + 1.0e-12)
    normal = torch.cat([gx / mag.clamp_min(1.0e-6), gy / mag.clamp_min(1.0e-6)], dim=1)
    tangent = torch.cat([-normal[:, 1:2], normal[:, 0:1]], dim=1)
    scale = mag.flatten(1).mean(dim=1)[:, None, None, None].clamp_min(1.0e-6)
    reliability = (mag / (mag + scale)).clamp(0.0, 1.0)
    return normal.to(base_logits), tangent.to(base_logits), reliability.to(base_logits)


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(1)
    weight = weight.to(value)
    return (value * weight).sum() / weight.sum().clamp_min(EPS)


def _unified_transport_energy(
    flow: torch.Tensor,
    base_logits: torch.Tensor,
    evidence_band: torch.Tensor,
    smooth_kernel: int,
    support_power: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Single reliability-adaptive anisotropic geometry energy.

    E_geo = E_support + E_tangent + E_smooth.

    * support: discourages displacement where Base evidence is confident and
      away from a boundary, replacing the old output-space deadzone gate;
    * tangent: discourages tangent motion only when the Base level-set normal is
      reliable, replacing the strict normal-1D architectural bottleneck;
    * smooth: enforces stronger coherence away from ambiguous boundaries,
      subsuming the previous independent edge-aware smoothness switch.
    """
    band = evidence_band.detach().clamp(0.0, 1.0)
    normal, tangent, reliability = _levelset_frame_and_reliability(
        base_logits, smooth_kernel=smooth_kernel
    )
    magnitude = torch.linalg.vector_norm(flow, dim=1, keepdim=True)
    support_weight = (1.0 - band).pow(max(float(support_power), 0.0))
    support = _weighted_mean(magnitude, support_weight)

    tangent_component = (flow * tangent).sum(dim=1, keepdim=True).abs()
    tangent_weight = band * reliability
    tangent_term = _weighted_mean(tangent_component, tangent_weight)

    dx = torch.linalg.vector_norm(flow[:, :, :, 1:] - flow[:, :, :, :-1], dim=1, keepdim=True)
    dy = torch.linalg.vector_norm(flow[:, :, 1:, :] - flow[:, :, :-1, :], dim=1, keepdim=True)
    wx = 0.5 * (support_weight[:, :, :, 1:] + support_weight[:, :, :, :-1])
    wy = 0.5 * (support_weight[:, :, 1:, :] + support_weight[:, :, :-1, :])
    smooth = 0.5 * (_weighted_mean(dx, wx) + _weighted_mean(dy, wy))

    normal_component = (flow * normal).sum(dim=1, keepdim=True).abs()
    stats = {
        "support": support,
        "tangent": tangent_term,
        "smooth": smooth,
        "reliability": reliability.mean(),
        "normal_abs": normal_component.mean(),
        "tangent_abs": tangent_component.mean(),
        "support_fraction": band.mean(),
    }
    return support + tangent_term + smooth, stats


def _mean_aux(aux: Dict[str, torch.Tensor], key: str, reference: torch.Tensor) -> torch.Tensor:
    value = aux.get(key)
    if isinstance(value, torch.Tensor):
        return value.detach().float().mean().to(reference)
    return reference.detach().new_zeros(())


def compute_geotr_m1_loss(
    cfg,
    candidate_logits: torch.Tensor,
    masks: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    epoch: int = 0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Geometry objective with region, contour and deformation safeguards."""
    del candidate_logits
    m1 = _cfg_get(cfg, "M1", None)
    train = _cfg_get(cfg, "TRAIN", None)
    logits = aux["geotopo_geometry_logits"]
    probability = aux["geotopo_geometry_probs"]
    base_probability = aux["geotopo_base_probs"]
    flow = aux["geotopo_geometry_flow_px"]
    operator = str(_cfg_get(m1, "GEOTR_M1_OPERATOR", "free_2d")).strip().lower()
    if operator not in {"free_2d", "normal_1d", "residual_rewrite"}:
        raise ValueError(
            "M1.GEOTR_M1_OPERATOR must be free_2d, normal_1d or residual_rewrite"
        )
    regularized_field = aux.get("geotr_m1_regularized_field", flow)
    target = _target3(masks, probability.shape[-2:]).to(probability)

    ce_weight = max(0.0, float(_cfg_get(m1, "GEOTR_M1_BCE_WEIGHT", _cfg_get(train, "CE_WEIGHT", 0.5))))
    dice_weight = max(0.0, float(_cfg_get(m1, "GEOTR_M1_DICE_WEIGHT", _cfg_get(train, "DICE_WEIGHT", 0.5))))
    boundary_weight = max(0.0, float(_cfg_get(m1, "GEOTR_M1_BOUNDARY_WEIGHT", 0.0)))
    surface_band_weight = max(0.0, float(
        _cfg_get(m1, "GEOTR_M1_SURFACE_BAND_WEIGHT", 0.0)
    ))
    total_weight = max(
        ce_weight + dice_weight + boundary_weight + surface_band_weight, 1.0e-8
    )
    ce_weight /= total_weight
    dice_weight /= total_weight
    boundary_weight /= total_weight
    surface_band_weight /= total_weight
    bce = F.binary_cross_entropy_with_logits(logits.float(), target[:, None].float()).to(probability)
    dice_loss = (1.0 - _dice_per_case(probability, target)).mean()
    boundary_loss = _boundary_dice_loss(probability, target)
    surface_band_loss, surface_band_fraction = _surface_band_bce_loss(
        logits,
        target,
        outer_radius_px=int(_cfg_get(m1, "GEOTR_M1_SURFACE_BAND_RADIUS_PX", 4)),
    )
    segmentation = (
        ce_weight * bce
        + dice_weight * dice_loss
        + boundary_weight * boundary_loss
        + surface_band_weight * surface_band_loss
    )

    error_supervision, error_stats = _jbt_error_supervision_loss(
        cfg, aux, base_probability, target
    )
    case_utility_loss, case_utility_stats = _jbt_case_utility_loss(
        cfg, aux, base_probability, target
    )
    v5_capacity_loss, v5_capacity_stats = _jbt_v5_oracle_capacity_loss(
        cfg, aux, base_probability, target
    )
    direction_alignment = _jbt_direction_alignment_loss(
        aux, base_probability, target
    )
    signed_displacement_loss, signed_displacement_stats = _jbt_v6_signed_displacement_loss(
        cfg, aux, base_probability, target
    )
    nondegrade_loss, preserve_loss = _jbt_nondegrade_and_preserve_losses(
        probability, base_probability, target
    )
    error_weight = max(0.0, float(_cfg_get(m1, "JBT_ERROR_SUPERVISION_WEIGHT", 0.0)))
    direction_weight = max(0.0, float(_cfg_get(m1, "JBT_DIRECTION_ALIGNMENT_WEIGHT", 0.0)))
    nondegrade_weight = max(0.0, float(_cfg_get(m1, "JBT_NONDEGRADE_WEIGHT", 0.0)))
    preserve_weight = max(0.0, float(_cfg_get(m1, "JBT_PRESERVE_WEIGHT", 0.0)))
    case_utility_weight = max(0.0, float(
        _cfg_get(m1, "JBT_CASE_UTILITY_WEIGHT", 0.0)
    ))
    v5_capacity_weight = max(0.0, float(
        _cfg_get(m1, "JBT_V5_ORACLE_CAPACITY_WEIGHT", 0.0)
    ))
    signed_displacement_weight = max(0.0, float(
        _cfg_get(m1, "JBT_V6_SIGNED_DISPLACEMENT_WEIGHT", 0.0)
    ))
    v62_enabled = bool(_cfg_get(m1, "JBT_V62_ENABLED", False))
    curriculum_stage = 0
    if v62_enabled:
        # Generator-first curriculum inside one uninterrupted end-to-end run.
        # Epoch is zero-based here: stages are 1--5, 6--10 and 11+.
        if int(epoch) < 5:
            curriculum_stage = 1
            case_utility_weight = max(0.0, float(_cfg_get(
                m1, "JBT_V62_UTILITY_WEIGHT_STAGE1", 0.05
            )))
            v5_capacity_weight = max(0.0, float(_cfg_get(
                m1, "JBT_V62_CAPACITY_WEIGHT_STAGE1", 0.50
            )))
            signed_displacement_weight = max(0.0, float(_cfg_get(
                m1, "JBT_V62_DISPLACEMENT_WEIGHT_STAGE1", 0.50
            )))
        elif int(epoch) < 10:
            curriculum_stage = 2
            case_utility_weight = max(0.0, float(_cfg_get(
                m1, "JBT_V62_UTILITY_WEIGHT_STAGE2", 0.15
            )))
            v5_capacity_weight = max(0.0, float(_cfg_get(
                m1, "JBT_V62_CAPACITY_WEIGHT_STAGE2", 0.50
            )))
            signed_displacement_weight = max(0.0, float(_cfg_get(
                m1, "JBT_V62_DISPLACEMENT_WEIGHT_STAGE2", 0.50
            )))
        else:
            curriculum_stage = 3
            case_utility_weight = max(0.0, float(_cfg_get(
                m1, "JBT_V62_UTILITY_WEIGHT_STAGE3", 0.30
            )))
            v5_capacity_weight = max(0.0, float(_cfg_get(
                m1, "JBT_V62_CAPACITY_WEIGHT_STAGE3", v5_capacity_weight
            )))
            signed_displacement_weight = max(0.0, float(_cfg_get(
                m1, "JBT_V62_DISPLACEMENT_WEIGHT_STAGE3", signed_displacement_weight
            )))
    jbt_safety_objective = (
        error_weight * error_supervision
        + direction_weight * direction_alignment
        + nondegrade_weight * nondegrade_loss
        + preserve_weight * preserve_loss
        + case_utility_weight * case_utility_loss
        + v5_capacity_weight * v5_capacity_loss
        + signed_displacement_weight * signed_displacement_loss
    )
    global_smoothness = _flow_smoothness(regularized_field)
    deform_mode = str(
        _cfg_get(m1, "GEOTR_M1_DEFORM_MODE", "global")
    ).strip().lower()
    if deform_mode not in {"global", "edge_aware", "none", "unified"}:
        raise ValueError(
            "M1.GEOTR_M1_DEFORM_MODE must be global, edge_aware, unified or none"
        )
    edge_band_fraction = flow.detach().new_zeros(())
    unified_energy = flow.sum() * 0.0
    unified_stats = {
        "support": unified_energy.detach(), "tangent": unified_energy.detach(),
        "smooth": unified_energy.detach(), "reliability": unified_energy.detach(),
        "normal_abs": unified_energy.detach(), "tangent_abs": unified_energy.detach(),
        "support_fraction": unified_energy.detach(),
    }
    if deform_mode == "unified":
        evidence_band = aux.get("geotr_m1_evidence_band")
        if not isinstance(evidence_band, torch.Tensor):
            raise RuntimeError("unified transport energy requires geotr_m1_evidence_band")
        unified_energy, unified_stats = _unified_transport_energy(
            flow,
            aux["geotopo_base_logits"],
            evidence_band,
            smooth_kernel=int(_cfg_get(m1, "GEOTR_M1_UNIFIED_NORMAL_SMOOTH_KERNEL", 5)),
            support_power=float(_cfg_get(m1, "GEOTR_M1_UNIFIED_SUPPORT_POWER", 2.0)),
        )
        smoothness = unified_stats["smooth"]
        edge_band_fraction = unified_stats["support_fraction"].detach()
    elif deform_mode == "edge_aware":
        edge_power = float(_cfg_get(m1, "GEOTR_M1_EDGE_STIFFNESS_POWER", 2.0))
        edge_floor = float(_cfg_get(m1, "GEOTR_M1_EDGE_STIFFNESS_FLOOR", 0.05))
        if edge_power < 0.0 or not 0.0 <= edge_floor <= 1.0:
            raise ValueError(
                "edge-aware deformation requires power >= 0 and floor in [0, 1]"
            )
        smoothness, edge_band_fraction = _edge_aware_flow_smoothness(
            regularized_field,
            base_probability,
            power=edge_power,
            floor=edge_floor,
        )
    elif deform_mode == "none":
        smoothness = global_smoothness.detach() * 0.0
    else:
        smoothness = global_smoothness
    smoothness_weight = float(_cfg_get(m1, "GEOTOPO_SMOOTHNESS_WEIGHT", 0.0))
    minimum_jacobian = float(_cfg_get(m1, "GEOTR_M1_MIN_JACOBIAN", 0.0))
    folding_penalty = (
        flow.sum() * 0.0
        if operator == "residual_rewrite"
        else _flow_folding_penalty(flow, minimum_jacobian)
    )
    folding_weight = max(0.0, float(_cfg_get(m1, "GEOTR_M1_FOLDING_WEIGHT", 0.0)))
    if deform_mode == "none":
        smoothness_weight = 0.0
        folding_weight = 0.0
    if deform_mode == "unified":
        unified_weight = max(0.0, float(_cfg_get(m1, "GEOTR_M1_UNIFIED_WEIGHT", 0.001)))
        support_ratio = max(0.0, float(_cfg_get(m1, "GEOTR_M1_UNIFIED_SUPPORT_RATIO", 1.0)))
        tangent_ratio = max(0.0, float(_cfg_get(m1, "GEOTR_M1_UNIFIED_TANGENT_RATIO", 1.0)))
        smooth_ratio = max(0.0, float(_cfg_get(m1, "GEOTR_M1_UNIFIED_SMOOTH_RATIO", 1.0)))
        unified_energy = (
            support_ratio * unified_stats["support"]
            + tangent_ratio * unified_stats["tangent"]
            + smooth_ratio * unified_stats["smooth"]
        )
        # Old independent smoothness is disabled: the full geometry prior is one
        # coupled energy.  Folding remains a topology safeguard shared by all
        # transport variants, not an additional refinement module.
        objective = (
            segmentation
            + jbt_safety_objective
            + unified_weight * unified_energy
            + folding_weight * folding_penalty
        )
    else:
        unified_weight = 0.0
        objective = (
            segmentation
            + jbt_safety_objective
            + max(0.0, smoothness_weight) * smoothness
            + folding_weight * folding_penalty
        )

    # Preserve the historical compatibility parameter without changing the objective.
    compat = aux.get("mhcs_m1_distribution_log_var")
    if isinstance(compat, torch.Tensor):
        objective = objective + 0.0 * compat

    base_dice = _dice_per_case(base_probability.detach(), target)
    final_dice = _dice_per_case(probability.detach(), target)
    zero = objective.detach().new_zeros(())
    diagnostics = {
        "mhcs_total_loss": objective.detach(),
        "mhcs_final_loss": segmentation.detach(),
        "mhcs_ce_loss": bce.detach(),
        "mhcs_dice_loss": dice_loss.detach(),
        "mhcs_m1_objective": objective.detach(),
        "mhcs_m2_objective": zero,
        "mhcs_final_gain": (final_dice - base_dice).mean().detach(),
        "mhcs_base_dice": base_dice.mean().detach(),
        "mhcs_final_dice": final_dice.mean().detach(),
        "geotopo_total_loss": objective.detach(),
        "geotopo_final_loss": segmentation.detach(),
        "geotopo_geometry_loss": segmentation.detach(),
        "geotopo_geometry_bce_loss": bce.detach(),
        "geotopo_geometry_dice_loss": dice_loss.detach(),
        "geotopo_geometry_boundary_loss": boundary_loss.detach(),
        "geotopo_geometry_surface_band_loss": surface_band_loss.detach(),
        "geotopo_surface_band_fraction": surface_band_fraction.detach(),
        "geotopo_reconstruction_loss": zero,
        "geotopo_flow_smoothness": smoothness.detach(),
        "geotopo_flow_folding_penalty": folding_penalty.detach(),
        "geotr_m1_total_loss": objective.detach(),
        "geotr_m1_segmentation_loss": segmentation.detach(),
        "geotr_m1_bce_loss": bce.detach(),
        "geotr_m1_dice_loss": dice_loss.detach(),
        "geotr_m1_boundary_loss": boundary_loss.detach(),
        "geotr_m1_surface_band_loss": surface_band_loss.detach(),
        "geotr_m1_surface_band_fraction": surface_band_fraction.detach(),
        "geotr_m1_surface_band_weight": objective.detach().new_tensor(float(surface_band_weight)),
        "jbt_error_supervision_loss": error_supervision.detach(),
        "jbt_direction_alignment_loss": direction_alignment.detach(),
        "jbt_nondegrade_loss": nondegrade_loss.detach(),
        "jbt_preserve_loss": preserve_loss.detach(),
        "jbt_case_utility_loss": case_utility_loss.detach(),
        "jbt_v6_utility_classification_loss": case_utility_stats.get("utility_classification_loss", zero).detach(),
        "jbt_v6_utility_regression_loss": case_utility_stats.get("utility_regression_loss", zero).detach(),
        "jbt_v6_utility_ranking_loss": case_utility_stats.get("utility_ranking_loss", zero).detach(),
        "jbt_v62_utility_preserve_ranking_loss": case_utility_stats.get("utility_preserve_ranking_loss", zero).detach(),
        "jbt_v62_gain_mean_mae": case_utility_stats.get("gain_mean_mae", zero).detach(),
        "jbt_v62_gain_sigma_mean": case_utility_stats.get("gain_sigma_mean", zero).detach(),
        "jbt_v62_lcb_score_mean": case_utility_stats.get("lcb_score_mean", zero).detach(),
        "jbt_v6_utility_target_abs_mean": case_utility_stats.get("utility_target_abs_mean", zero).detach(),
        "jbt_v6_signed_displacement_loss": signed_displacement_loss.detach(),
        "jbt_v6_signed_disp_target_abs_mean": signed_displacement_stats["signed_disp_target_abs_mean"].detach(),
        "jbt_v6_signed_disp_pred_abs_mean": signed_displacement_stats["signed_disp_pred_abs_mean"].detach(),
        "jbt_v6_signed_disp_direction_accuracy": signed_displacement_stats["signed_disp_direction_accuracy"].detach(),
        "jbt_v6_signed_disp_owner_fraction": signed_displacement_stats["signed_disp_owner_fraction"].detach(),
        "jbt_v62_normal_ray_hit_fraction": signed_displacement_stats["normal_ray_hit_fraction"].detach(),
        "jbt_v62_normal_ray_target_enabled": signed_displacement_stats["normal_ray_target_enabled"].detach(),
        "jbt_v634_signed_disp_magnitude_loss": signed_displacement_stats.get("signed_disp_magnitude_loss", zero).detach(),
        "jbt_v634_balanced_sign_loss": signed_displacement_stats.get("signed_disp_balanced_sign_loss", zero).detach(),
        "jbt_v634_positive_owner_fraction": signed_displacement_stats.get("signed_disp_positive_owner_fraction", zero).detach(),
        "jbt_v634_negative_owner_fraction": signed_displacement_stats.get("signed_disp_negative_owner_fraction", zero).detach(),
        "jbt_v634_active_case_fraction": signed_displacement_stats.get("signed_disp_active_case_fraction", zero).detach(),
        "jbt_v5_oracle_capacity_loss": v5_capacity_loss.detach(),
        "jbt_v6_capacity_margin_penalty": v5_capacity_stats.get("capacity_margin_penalty", zero).detach(),
        "jbt_v6_capacity_target_gain": v5_capacity_stats.get("capacity_target_gain", zero).detach(),
        "jbt_v5_oracle_capacity_gain": v5_capacity_stats.get("oracle_gain", zero).detach(),
        "jbt_v5_oracle_capacity_signed_gain": v5_capacity_stats.get("oracle_signed_gain", zero).detach(),
        "jbt_v5_oracle_capacity_benefit_rate": v5_capacity_stats.get("oracle_benefit_rate", zero).detach(),
        "jbt_v5_strength_case_oracle_gain": case_utility_stats.get("strength_case_oracle_gain", zero).detach(),
        "jbt_v5_strength_any_benefit_rate": case_utility_stats.get("strength_any_benefit_rate", zero).detach(),
        "jbt_v5_utility_selected_gain": case_utility_stats.get("utility_selected_gain", zero).detach(),
        "jbt_v5_oracle_realization_ratio": case_utility_stats.get("oracle_realization_ratio", zero).detach(),
        "jbt_v5_utility_accept_rate": case_utility_stats.get("utility_accept_rate", zero).detach(),
        "jbt_case_gain_mean": case_utility_stats["case_gain_mean"].detach(),
        "jbt_case_benefit_target_rate": case_utility_stats["case_benefit_target_rate"].detach(),
        "jbt_case_harm_target_rate": case_utility_stats.get("case_harm_target_rate", zero).detach(),
        "jbt_case_accept_prob_mean": case_utility_stats["case_accept_prob_mean"].detach(),
        "jbt_case_neutral_rate": case_utility_stats.get("case_neutral_rate", zero).detach(),
        "jbt_error_supervision_weight": objective.detach().new_tensor(float(error_weight)),
        "jbt_direction_alignment_weight": objective.detach().new_tensor(float(direction_weight)),
        "jbt_nondegrade_weight": objective.detach().new_tensor(float(nondegrade_weight)),
        "jbt_preserve_weight": objective.detach().new_tensor(float(preserve_weight)),
        "jbt_case_utility_weight": objective.detach().new_tensor(float(case_utility_weight)),
        "jbt_v5_oracle_capacity_weight": objective.detach().new_tensor(float(v5_capacity_weight)),
        "jbt_v6_signed_displacement_weight": objective.detach().new_tensor(float(signed_displacement_weight)),
        "jbt_v62_curriculum_stage": objective.detach().new_tensor(float(curriculum_stage)),
        "jbt_v63_enabled": objective.detach().new_tensor(
            1.0 if bool(_cfg_get(m1, "JBT_V63_ENABLED", False)) else 0.0
        ),
        "jbt_v634_utility_detached_from_generator": _mean_aux(
            aux, "jbt_v634_utility_detached_from_generator", objective
        ),
        "jbt_v62_effective_utility_weight": objective.detach().new_tensor(float(case_utility_weight)),
        "jbt_v62_effective_capacity_weight": objective.detach().new_tensor(float(v5_capacity_weight)),
        "jbt_v62_effective_displacement_weight": objective.detach().new_tensor(float(signed_displacement_weight)),
        "jbt_fn_target_mean": error_stats["fn_target_mean"].detach(),
        "jbt_fp_target_mean": error_stats["fp_target_mean"].detach(),
        "jbt_error_prob_mean": error_stats["error_prob_mean"].detach(),
        "jbt_error_pos_weight_mean": error_stats.get("error_pos_weight_mean", zero).detach(),
        "jbt_train_posterior_sample_count": _mean_aux(
            aux, "jbt_train_posterior_sample_count", objective
        ),
        "v470_aux_to_base_grad_scale": _mean_aux(
            aux, "v470_aux_to_base_grad_scale", objective
        ),
        "geotr_m1_error_gate_fraction": _mean_aux(aux, "geotr_m1_error_gate_fraction", objective),
        "geotr_m1_dynamic_error_gate_enabled": _mean_aux(aux, "geotr_m1_dynamic_error_gate_enabled", objective),
        "geotr_m1_error_direction_coupling": _mean_aux(aux, "geotr_m1_error_direction_coupling", objective),
        "jbt_v5_support_radius_px": _mean_aux(aux, "jbt_v5_support_radius_px", objective),
        "jbt_v5_broad_support_fraction": _mean_aux(aux, "jbt_v5_broad_support_fraction", objective),
        "jbt_v5_effective_support_fraction": _mean_aux(aux, "jbt_v5_effective_support_fraction", objective),
        "jbt_v5_selected_strength": _mean_aux(aux, "jbt_v5_selected_strength", objective),
        "jbt_v5_multistrength_enabled": _mean_aux(aux, "jbt_v5_multistrength_enabled", objective),
        "jbt_v6_direct_signed_flow": _mean_aux(aux, "jbt_v6_direct_signed_flow", objective),
        "jbt_v6_candidate_utility_enabled": _mean_aux(aux, "jbt_v6_candidate_utility_enabled", objective),
        "jbt_v6_direction_abs_mean": _mean_aux(aux, "jbt_v6_direction_abs_mean", objective),
        "jbt_v6_scalar_abs_mean": _mean_aux(aux, "jbt_v6_scalar_abs_mean", objective),
        "jbt_v62_enabled": _mean_aux(aux, "jbt_v62_enabled", objective),
        "jbt_v62_actual_support_mean": _mean_aux(aux, "jbt_v62_actual_support_mean", objective),
        "jbt_v62_active_flow_to_scalar_ratio": _mean_aux(aux, "jbt_v62_active_flow_to_scalar_ratio", objective),
        "geotr_m1_posterior_reconstruction_mismatch": _mean_aux(
            aux, "geotr_m1_posterior_reconstruction_mismatch", objective
        ),
        "geotr_m1_posterior_reconstruction_prob_mismatch": _mean_aux(
            aux, "geotr_m1_posterior_reconstruction_prob_mismatch", objective
        ),
        "jbt_case_accept_fraction": _mean_aux(
            aux, "jbt_case_accept_fraction", objective
        ),
        "geotr_m1_feature_feedback_logit_change_max": _mean_aux(
            aux, "geotr_m1_feature_feedback_logit_change_max", objective
        ),
        "geotr_m1_logit_bounded_feature_feedback": _mean_aux(
            aux, "geotr_m1_logit_bounded_feature_feedback", objective
        ),
        "geotr_m1_error_gate_enabled": _mean_aux(
            aux, "geotr_m1_error_gate_enabled", objective
        ),
        "geotr_m1_flow_smoothness": smoothness.detach(),
        "geotr_m1_flow_global_smoothness": global_smoothness.detach(),
        "geotr_m1_edge_band_fraction": edge_band_fraction.detach(),
        "geotr_m1_deform_mode_id": objective.detach().new_tensor(
            {"none": 0.0, "global": 1.0, "edge_aware": 2.0, "unified": 3.0}[deform_mode]
        ),
        "geotr_m1_flow_folding_penalty": folding_penalty.detach(),
        "geotr_m1_flow_rms_px": _mean_aux(aux, "geotopo_flow_rms_px", objective),
        "geotr_m1_flow_max_px": _mean_aux(aux, "geotopo_flow_max_px", objective),
        "geotr_m1_flow_jacobian_mean": _mean_aux(aux, "geotopo_flow_jacobian_mean", objective),
        "geotr_m1_flow_folding_fraction": _mean_aux(aux, "geotopo_flow_folding_fraction", objective),
        "geotr_m1_geometry_abs_change": _mean_aux(aux, "geotopo_geometry_abs_change", objective),
        "geotr_m1_flow_scale_px": _mean_aux(aux, "geotr_m1_flow_scale_px", objective),
        "geotr_m1_context_gate": _mean_aux(aux, "geotr_m1_context_gate", objective),
        "geotr_m1_context_film_abs": _mean_aux(
            aux, "geotr_m1_context_film_abs", objective
        ),
        "geotr_m1_text_latent_abs": _mean_aux(
            aux, "geotr_m1_text_latent_abs", objective
        ),
        "geotr_m1_anchor_boundary_fraction": _mean_aux(
            aux, "geotr_m1_anchor_boundary_fraction", objective
        ),
        "geotr_m1_posterior_evidence_mean": _mean_aux(
            aux, "geotr_m1_posterior_evidence", objective
        ),
        "geotr_m1_feature_feedback_abs": _mean_aux(
            aux, "geotr_m1_feature_feedback_abs", objective
        ),
        "geotr_m1_feature_feedback_logit_change": _mean_aux(
            aux, "geotr_m1_feature_feedback_logit_change", objective
        ),
        "geotr_m1_feature_feedback_scale": _mean_aux(
            aux, "geotr_m1_feature_feedback_scale", objective
        ),
        "geotr_m1_feature_feedback_enabled": _mean_aux(
            aux, "geotr_m1_feature_feedback_enabled", objective
        ),
        "geotr_m1_posterior_uncertainty_enabled": _mean_aux(
            aux, "geotr_m1_posterior_uncertainty_enabled", objective
        ),
        "geotr_m1_conditioners_detached": _mean_aux(
            aux, "geotr_m1_conditioners_detached", objective
        ),
        "geotr_m1_operator_id": _mean_aux(aux, "geotr_m1_operator_id", objective),
        "geotr_m1_gate_mode_id": _mean_aux(aux, "geotr_m1_gate_mode_id", objective),
        "geotr_m1_deadzone_fraction": _mean_aux(aux, "geotr_m1_deadzone_fraction", objective),
        "geotr_m1_exact_identity_error": _mean_aux(aux, "geotr_m1_exact_identity_error", objective),
        "geotr_m1_tangent_energy_ratio": _mean_aux(aux, "geotr_m1_tangent_energy_ratio", objective),
        "geotr_m1_range_violation_fraction": _mean_aux(aux, "geotr_m1_range_violation_fraction", objective),
        "geotr_m1_range_violation_max": _mean_aux(aux, "geotr_m1_range_violation_max", objective),
        "geotr_m1_unified_energy": unified_energy.detach(),
        "geotr_m1_unified_weight": objective.detach().new_tensor(float(unified_weight)),
        "geotr_m1_unified_support": unified_stats["support"].detach(),
        "geotr_m1_unified_tangent": unified_stats["tangent"].detach(),
        "geotr_m1_unified_smooth": unified_stats["smooth"].detach(),
        "geotr_m1_normal_reliability": unified_stats["reliability"].detach(),
        "geotr_m1_unified_normal_abs": unified_stats["normal_abs"].detach(),
        "geotr_m1_unified_tangent_abs": unified_stats["tangent_abs"].detach(),
        "geotr_m1_has_m2": zero,
    }
    return objective, diagnostics
