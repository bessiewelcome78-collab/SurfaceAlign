from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

EPS = 1.0e-6


def _scale_gradient(value: torch.Tensor, scale: float) -> torch.Tensor:
    """Forward identity with a bounded, non-zero upstream gradient."""
    scale = float(scale)
    if scale >= 1.0:
        return value
    if scale <= 0.0:
        return value.detach()
    return value.detach() + scale * (value - value.detach())


def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def _to_b1hw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        return x[:, None]
    if x.dim() == 4 and x.shape[1] != 1:
        return x[:, :1]
    return x


def _hard_dice(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = (_to_b1hw(mask) > 0.5).float()
    target = (_to_b1hw(target) > 0.5).float()
    inter = (mask * target).sum(dim=(-2, -1))
    den = mask.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return ((2.0 * inter + EPS) / (den + EPS))[:, 0]


def _boundary_proxy(mask: torch.Tensor) -> torch.Tensor:
    mask = (_to_b1hw(mask) > 0.5).float()
    dil = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    ero = 1.0 - F.max_pool2d(
        1.0 - mask, kernel_size=3, stride=1, padding=1
    )
    return (dil - ero).clamp(0.0, 1.0)


def _boundary_dice(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bm = _boundary_proxy(mask)
    bt = _boundary_proxy(target)
    inter = (bm * bt).sum(dim=(-2, -1))
    den = bm.sum(dim=(-2, -1)) + bt.sum(dim=(-2, -1))
    return ((2.0 * inter + EPS) / (den + EPS))[:, 0]


def _soft_dice_bk(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_k = target.expand(-1, prob.shape[1], -1, -1)
    inter = (prob * target_k).sum(dim=(-2, -1))
    den = prob.sum(dim=(-2, -1)) + target_k.sum(dim=(-2, -1))
    return (2.0 * inter + EPS) / (den + EPS)


def _soft_boundary_bk(prob: torch.Tensor) -> torch.Tensor:
    b, k, h, w = prob.shape
    flat = prob.reshape(b * k, 1, h, w)
    dil = F.max_pool2d(flat, kernel_size=3, stride=1, padding=1)
    ero = -F.max_pool2d(-flat, kernel_size=3, stride=1, padding=1)
    return (dil - ero).clamp(0.0, 1.0).reshape(b, k, h, w)


def _soft_boundary_dice_bk(
    prob: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    bp = _soft_boundary_bk(prob)
    bt = _soft_boundary_bk(target).expand(-1, prob.shape[1], -1, -1)
    inter = (bp * bt).sum(dim=(-2, -1))
    den = bp.sum(dim=(-2, -1)) + bt.sum(dim=(-2, -1))
    return (2.0 * inter + EPS) / (den + EPS)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float()
    return (values * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def _masked_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    raw = F.smooth_l1_loss(
        pred.float(), target.float(), beta=beta, reduction="none"
    )
    return _masked_mean(raw, mask)


def _masked_bce_dynamic(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    max_weight: float,
) -> torch.Tensor:
    target = target.float()
    mask_f = mask.float()
    with torch.no_grad():
        pos = (target * mask_f).sum().clamp_min(1.0)
        neg = ((1.0 - target) * mask_f).sum().clamp_min(1.0)
        pos_weight = (neg / pos).clamp(0.5, max_weight)
    raw = F.binary_cross_entropy_with_logits(
        logits.float(), target, pos_weight=pos_weight, reduction="none"
    )
    return _masked_mean(raw, mask)


def _masked_focal_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    domain: torch.Tensor,
    gamma: float = 2.0,
    max_pos_weight: float = 200.0,
) -> torch.Tensor:
    target = target.float()
    domain = domain.float()
    with torch.no_grad():
        pos = (target * domain).sum().clamp_min(1.0)
        neg = ((1.0 - target) * domain).sum().clamp_min(1.0)
        pos_weight = (neg / pos).clamp(1.0, max_pos_weight)
    raw = F.binary_cross_entropy_with_logits(
        logits.float(), target, pos_weight=pos_weight, reduction="none"
    )
    prob = torch.sigmoid(logits.float())
    pt = prob * target + (1.0 - prob) * (1.0 - target)
    focal = (1.0 - pt).pow(gamma) * raw
    return (focal * domain).sum() / domain.sum().clamp_min(1.0)


def _masked_tversky_loss(
    prob: torch.Tensor,
    target: torch.Tensor,
    domain: torch.Tensor,
    alpha: float = 0.70,
    beta: float = 0.30,
) -> torch.Tensor:
    prob = prob.float() * domain.float()
    target = target.float() * domain.float()
    tp = (prob * target).sum(dim=(-2, -1))
    fp = (prob * (1.0 - target)).sum(dim=(-2, -1))
    fn = ((1.0 - prob) * target).sum(dim=(-2, -1))
    score = (tp + EPS) / (tp + alpha * fp + beta * fn + EPS)
    positive_case = target.sum(dim=(-2, -1)) > 0.5
    if positive_case.any():
        return (1.0 - score)[positive_case].mean()
    return prob.sum() * 0.0


def _binary_metrics(
    prob: torch.Tensor,
    target: torch.Tensor,
    domain: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = (prob >= 0.5).float() * domain.float()
    target = target.float() * domain.float()
    tp = (pred * target).sum()
    fp = (pred * (1.0 - target)).sum()
    fn = ((1.0 - pred) * target).sum()
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(EPS)
    return precision, recall, f1


def _ramp_weight(epoch: int, start: int, ramp: int) -> float:
    if epoch < start:
        return 0.0
    if ramp <= 0:
        return 1.0
    return min(1.0, float(epoch - start + 1) / float(ramp))


def _action_masks(
    aux: Dict[str, torch.Tensor],
    k: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    supports = aux.get("v20_action_supports")
    if isinstance(supports, torch.Tensor):
        valid = supports[:, :k].flatten(2).sum(dim=-1) > 0.5
    else:
        batch = aux["candidates"].shape[0]
        valid = torch.ones((batch, k), device=device, dtype=torch.bool)

    enabled = aux.get("v469_action_enabled_mask")
    if isinstance(enabled, torch.Tensor):
        enabled = enabled.to(device=device, dtype=torch.bool).reshape(-1)[:k]
    else:
        enabled = torch.ones((k,), device=device, dtype=torch.bool)
    valid = valid & enabled[None]

    action_types = aux.get("v20_action_types")
    if isinstance(action_types, torch.Tensor):
        action_types = action_types.to(device=device).long().reshape(-1)[:k]
    else:
        action_types = torch.zeros((k,), device=device, dtype=torch.long)
    return valid, enabled, action_types


def _candidate_oracle_coverage_loss(
    cfg: Any,
    candidates: torch.Tensor,
    base_logits: torch.Tensor,
    gt: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    fp_t: torch.Tensor,
    fn_t: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Candidate-ceiling objective with explicit coverage/non-empty guards."""
    m1 = _cfg_get(cfg, "M1", None)
    if bool(_cfg_get(m1, "V470_STRICT_JOINT_E2E", False)):
        base_logits = _scale_gradient(
            base_logits,
            float(_cfg_get(m1, "V470_CANDIDATE_TO_BASE_GRAD_SCALE", 0.10)),
        )
    if candidates is None or candidates.shape[1] <= 1:
        zero = base_logits.sum() * 0.0
        return zero, {}

    base_prob = torch.sigmoid(_to_b1hw(base_logits)).clamp(EPS, 1.0 - EPS)
    cand_prob = torch.sigmoid(candidates[:, 1:].float()).clamp(EPS, 1.0 - EPS)
    b, k, h, w = cand_prob.shape
    gt = _to_b1hw(gt).float()
    if gt.shape[-2:] != (h, w):
        gt = F.interpolate(gt, size=(h, w), mode="nearest")

    valid_action, enabled, action_types = _action_masks(aux, k, cand_prob.device)
    base_k = base_prob.expand(-1, k, -1, -1)
    base_dice = _soft_dice_bk(base_prob, gt)[:, 0]
    cand_dice = _soft_dice_bk(cand_prob, gt)
    base_boundary = _soft_boundary_dice_bk(base_prob, gt)[:, 0]
    cand_boundary = _soft_boundary_dice_bk(cand_prob, gt)
    delta_dice = cand_dice - base_dice[:, None]
    delta_boundary = cand_boundary - base_boundary[:, None]

    wd = float(_cfg_get(m1, "M2_UTILITY_DSC_WEIGHT", 0.60))
    wn = float(_cfg_get(m1, "M2_UTILITY_NSD_WEIGHT", 0.40))
    norm = max(wd + wn, EPS)
    wd, wn = wd / norm, wn / norm
    utility = wd * delta_dice + wn * delta_boundary

    temperature = max(
        1.0e-4, float(_cfg_get(m1, "V468_ORACLE_TEMPERATURE", 0.002))
    )
    masked_utility = utility.masked_fill(~valid_action, -1.0e4)
    with torch.no_grad():
        oracle_weight = torch.softmax(masked_utility / temperature, dim=1)
        oracle_weight = oracle_weight * valid_action.float()
        oracle_weight = oracle_weight / oracle_weight.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
    oracle_utility = (oracle_weight * utility).sum(dim=1)
    has_valid = valid_action.any(dim=1)
    gain_scale = max(
        1.0e-4, float(_cfg_get(m1, "V469_CANDIDATE_GAIN_SCALE", 0.01))
    )
    oracle_loss = -_masked_mean(
        oracle_utility / gain_scale, has_valid
    )

    delete = action_types[None, :, None, None] < 2
    removed = (base_k - cand_prob).relu()
    added = (cand_prob - base_k).relu()
    change = torch.where(delete, removed, added)
    benefit_map = torch.where(
        delete,
        removed * fp_t.expand(-1, k, -1, -1),
        added * fn_t.expand(-1, k, -1, -1),
    )
    harm_map = torch.where(
        delete,
        removed * gt.expand(-1, k, -1, -1),
        added * (1.0 - gt).expand(-1, k, -1, -1),
    )
    change_mass = change.sum(dim=(-2, -1))
    benefit_fraction = benefit_map.sum(dim=(-2, -1)) / change_mass.clamp_min(EPS)
    harm_fraction = harm_map.sum(dim=(-2, -1)) / change_mass.clamp_min(EPS)
    valid_change = (change_mass > 0.5) & valid_action
    direction_raw = (1.0 - benefit_fraction) + harm_fraction
    direction_weight = oracle_weight * valid_change.float()
    direction_loss = (
        (direction_weight * direction_raw).sum()
        / direction_weight.sum().clamp_min(1.0)
    )

    no_harm_margin = float(
        _cfg_get(m1, "V468_CANDIDATE_NO_HARM_MARGIN", 0.0)
    )
    downside = F.relu((no_harm_margin - utility) / gain_scale)
    downside_loss = _masked_mean(downside, valid_action)

    target_gain = float(_cfg_get(m1, "V468_ORACLE_TARGET_GAIN", 0.004))
    target_case = has_valid & (
        (fp_t.sum(dim=(-2, -1))[:, 0] > 0.5)
        | (fn_t.sum(dim=(-2, -1))[:, 0] > 0.5)
    )
    target_loss = _masked_mean(
        F.relu((target_gain - oracle_utility) / gain_scale),
        target_case,
    )

    supports = aux.get("v20_action_supports")
    coverage_loss = cand_prob.sum() * 0.0
    gate_nonempty_loss = cand_prob.sum() * 0.0
    if isinstance(supports, torch.Tensor):
        supports = supports[:, :k].float()
        delete_slot = ((action_types < 2) & enabled).view(1, k, 1, 1)
        fill_slot = ((action_types >= 2) & enabled).view(1, k, 1, 1)
        delete_union = (
            supports.masked_fill(~delete_slot, 0.0).amax(dim=1, keepdim=True)
        )
        fill_union = (
            supports.masked_fill(~fill_slot, 0.0).amax(dim=1, keepdim=True)
        )
        fp_case = fp_t.sum(dim=(-2, -1))[:, 0] > 0.5
        fn_case = fn_t.sum(dim=(-2, -1))[:, 0] > 0.5
        delete_dice = 1.0 - _soft_dice_bk(delete_union, fp_t)[:, 0]
        fill_dice = 1.0 - _soft_dice_bk(fill_union, fn_t)[:, 0]
        coverage_loss = _masked_mean(delete_dice, fp_case)
        if bool(fill_slot.any().item()):
            coverage_loss = coverage_loss + _masked_mean(fill_dice, fn_case)

        raw_supports = aux.get("v469_raw_action_supports")
        if isinstance(raw_supports, torch.Tensor):
            raw_supports = raw_supports[:, :k].float()
            raw_mass = raw_supports.flatten(2).sum(dim=-1)
            kept_mass = supports.flatten(2).sum(dim=-1)
            raw_valid = (raw_mass > 0.5) & enabled[None]
            keep_ratio = kept_mass / raw_mass.clamp_min(1.0)
            minimum_ratio = float(
                _cfg_get(m1, "V469_GATE_MIN_KEEP_RATIO", 0.20)
            )
            gate_nonempty_loss = _masked_mean(
                F.relu(minimum_ratio - keep_ratio), raw_valid
            )

    total = (
        float(_cfg_get(m1, "V468_CANDIDATE_ORACLE_WEIGHT", 1.0))
        * oracle_loss
        + float(_cfg_get(m1, "V468_CANDIDATE_DIRECTION_WEIGHT", 0.25))
        * direction_loss
        + float(_cfg_get(m1, "V468_CANDIDATE_DOWNSIDE_WEIGHT", 0.25))
        * downside_loss
        + float(_cfg_get(m1, "V468_CANDIDATE_TARGET_WEIGHT", 1.0))
        * target_loss
        + float(_cfg_get(m1, "V469_CANDIDATE_COVERAGE_WEIGHT", 1.0))
        * coverage_loss
        + float(_cfg_get(m1, "V469_GATE_NONEMPTY_WEIGHT", 1.0))
        * gate_nonempty_loss
    )

    preserve_zero = delta_dice.new_zeros((b, 1))
    hard_oracle_dsc = torch.cat(
        [preserve_zero, delta_dice.masked_fill(~valid_action, -1.0e4)], dim=1
    ).max(dim=1).values
    hard_oracle_boundary = torch.cat(
        [preserve_zero, delta_boundary.masked_fill(~valid_action, -1.0e4)], dim=1
    ).max(dim=1).values
    diag = {
        "v468_candidate_loss": float(total.detach().item()),
        "v468_candidate_oracle_loss": float(oracle_loss.detach().item()),
        "v468_candidate_direction_loss": float(direction_loss.detach().item()),
        "v468_candidate_downside_loss": float(downside_loss.detach().item()),
        "v468_soft_oracle_utility": float(oracle_utility.detach().mean().item()),
        "v468_soft_oracle_delta_dsc": float(
            hard_oracle_dsc.detach().mean().item()
        ),
        "v468_soft_oracle_delta_boundary": float(
            hard_oracle_boundary.detach().mean().item()
        ),
        "v468_candidate_benefit_fraction": float(
            benefit_fraction[valid_change].detach().mean().item()
            if valid_change.any() else 0.0
        ),
        "v468_candidate_harm_fraction": float(
            harm_fraction[valid_change].detach().mean().item()
            if valid_change.any() else 0.0
        ),
        "v469_candidate_coverage_loss": float(coverage_loss.detach().item()),
        "v469_gate_nonempty_loss": float(gate_nonempty_loss.detach().item()),
        "v469_valid_action_rate": float(
            valid_action.float().detach().mean().item()
        ),
    }
    return total, diag


def _ccv_targets(
    candidates: torch.Tensor,
    base_logits: torch.Tensor,
    gt: torch.Tensor,
    dsc_eps: float,
    nsd_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    base_prob = torch.sigmoid(_to_b1hw(base_logits)).clamp(EPS, 1.0 - EPS)
    cand_prob = torch.sigmoid(candidates[:, 1:].float()).clamp(EPS, 1.0 - EPS)
    _, k, h, w = cand_prob.shape
    gt = _to_b1hw(gt).float()
    if gt.shape[-2:] != (h, w):
        gt = F.interpolate(gt, size=(h, w), mode="nearest")
    with torch.no_grad():
        base_dsc = _hard_dice(base_prob, gt)
        base_nsd = _boundary_dice(base_prob, gt)
        cand_dsc, cand_nsd = [], []
        for i in range(k):
            cand_dsc.append(_hard_dice(cand_prob[:, i:i + 1], gt))
            cand_nsd.append(_boundary_dice(cand_prob[:, i:i + 1], gt))
        cand_dsc = torch.stack(cand_dsc, dim=1)
        cand_nsd = torch.stack(cand_nsd, dim=1)
        delta_dsc = cand_dsc - base_dsc[:, None]
        delta_nsd = cand_nsd - base_nsd[:, None]
        harm = ((delta_dsc < -dsc_eps) | (delta_nsd < -nsd_eps)).float()
        pareto = (
            (delta_dsc >= -dsc_eps)
            & (delta_nsd >= -nsd_eps)
            & ((delta_dsc > dsc_eps) | (delta_nsd > nsd_eps))
        ).float()
    return delta_dsc, delta_nsd, harm, pareto


def compute_v463_residual_ccv_joint_loss(
    cfg: Any,
    candidates: torch.Tensor,
    masks: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    epoch: int = 0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """V470 fair one-run joint objective.

    Preserve/Base, M1 and CCV are all trainable from the same task run.  Online
    FP/FN labels are formed from a detached snapshot of the current Preserve
    probability (label construction is not a model freeze).  Cross-module
    gradients remain non-zero but are bounded in the model/loss.  Empty actions
    are excluded from CCV supervision and Test is not involved.
    """
    m1 = _cfg_get(cfg, "M1", None)
    base_logits = aux.get("base_logits")
    if base_logits is None:
        base_logits = candidates[:, 0]
    base_prob = torch.sigmoid(_to_b1hw(base_logits)).clamp(EPS, 1.0 - EPS)
    gt = _to_b1hw(masks).float()
    if gt.shape[-2:] != base_prob.shape[-2:]:
        gt = F.interpolate(gt, size=base_prob.shape[-2:], mode="nearest")
    # Online targets use the current Preserve prediction, but target creation
    # is detached to avoid the degenerate solution of changing Base merely to
    # change its own labels.  Base itself remains trainable through the primary
    # segmentation loss and bounded auxiliary gradients.
    base_target_prob = base_prob.detach()
    hard_base = (base_target_prob > 0.5).float()
    hard_fp_t = hard_base * (1.0 - gt)
    hard_fn_t = (1.0 - hard_base) * gt
    if bool(_cfg_get(m1, "V470_SOFT_ONLINE_RESIDUAL_TARGETS", True)):
        fp_t = base_target_prob * (1.0 - gt)
        fn_t = (1.0 - base_target_prob) * gt
        tp_t = base_target_prob * gt
        bg_t = (1.0 - base_target_prob) * (1.0 - gt)
    else:
        fp_t, fn_t = hard_fp_t, hard_fn_t
        tp_t = hard_base * gt
        bg_t = (1.0 - hard_base) * (1.0 - gt)

    total = base_prob.sum() * 0.0
    diag: Dict[str, float] = {}

    res_logits = aux.get("v463_residual_logits")
    conditional = bool(_cfg_get(m1, "V469_CONDITIONAL_RESIDUAL", False))
    if res_logits is not None:
        if res_logits.shape[-2:] != gt.shape[-2:]:
            res_logits = F.interpolate(
                res_logits,
                size=gt.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if conditional:
            fp_logit = aux.get("v469_fp_binary_logit", res_logits[:, 0:1])
            fn_logit = aux.get("v469_fn_binary_logit", res_logits[:, 1:2])
            if fp_logit.shape[-2:] != gt.shape[-2:]:
                fp_logit = F.interpolate(
                    fp_logit, size=gt.shape[-2:], mode="bilinear", align_corners=False
                )
                fn_logit = F.interpolate(
                    fn_logit, size=gt.shape[-2:], mode="bilinear", align_corners=False
                )

            domain_floor = min(max(
                float(_cfg_get(m1, "V470_RESIDUAL_DOMAIN_FLOOR", 0.15)),
                0.0,
            ), 1.0)
            # Soft conditional domains remain informative from epoch one, when
            # the randomly initialized segmentation head may predict ~0.5 over
            # most pixels.  They progressively specialize as Preserve improves.
            fp_domain = (
                domain_floor + (1.0 - domain_floor) * base_target_prob
            ).detach()
            radius = max(1, int(_cfg_get(m1, "V469_FN_BAND_RADIUS", 8)))
            outer = (
                F.max_pool2d(
                    hard_base,
                    kernel_size=2 * radius + 1,
                    stride=1,
                    padding=radius,
                )
                - hard_base
            ).clamp(0.0, 1.0)
            fn_domain = torch.maximum(
                domain_floor + (1.0 - domain_floor) * (1.0 - base_target_prob),
                outer,
            ).detach()

            gamma = float(_cfg_get(m1, "V469_RESIDUAL_FOCAL_GAMMA", 2.0))
            max_pw = float(_cfg_get(m1, "V469_RESIDUAL_MAX_POS_WEIGHT", 200.0))
            fp_focal = _masked_focal_bce(
                fp_logit, fp_t, fp_domain, gamma=gamma, max_pos_weight=max_pw
            )
            fn_focal = _masked_focal_bce(
                fn_logit, fn_t, fn_domain, gamma=gamma, max_pos_weight=max_pw
            )
            fp_prob = torch.sigmoid(fp_logit)
            fn_prob = torch.sigmoid(fn_logit)
            fp_tversky = _masked_tversky_loss(fp_prob, fp_t, fp_domain)
            fn_tversky = _masked_tversky_loss(fn_prob, fn_t, fn_domain)
            tversky_weight = float(
                _cfg_get(m1, "V469_RESIDUAL_TVERSKY_WEIGHT", 1.0)
            )
            residual_loss = (
                fp_focal + fn_focal
                + tversky_weight * (fp_tversky + fn_tversky)
            )
            fp_precision, fp_recall, fp_f1 = _binary_metrics(
                fp_prob, hard_fp_t, torch.ones_like(fp_domain)
            )
            fn_precision, fn_recall, fn_f1 = _binary_metrics(
                fn_prob, hard_fn_t, torch.ones_like(fn_domain)
            )
            diag.update({
                "v468_residual_ce_loss": float(
                    (fp_focal + fn_focal).detach().item()
                ),
                "v468_residual_sparse_dice_loss": float(
                    (fp_tversky + fn_tversky).detach().item()
                ),
                "v469_fp_focal_loss": float(fp_focal.detach().item()),
                "v469_fn_focal_loss": float(fn_focal.detach().item()),
                "v469_fp_tversky_loss": float(fp_tversky.detach().item()),
                "v469_fn_tversky_loss": float(fn_tversky.detach().item()),
                "v469_fp_precision": float(fp_precision.detach().item()),
                "v469_fp_recall": float(fp_recall.detach().item()),
                "v469_fp_f1": float(fp_f1.detach().item()),
                "v469_fn_precision": float(fn_precision.detach().item()),
                "v469_fn_recall": float(fn_recall.detach().item()),
                "v469_fn_f1": float(fn_f1.detach().item()),
            })
        else:
            multiclass = bool(
                _cfg_get(m1, "V468_RESIDUAL_MULTICLASS_CE", False)
            )
            if multiclass:
                labels = torch.full_like(gt[:, 0], 3, dtype=torch.long)
                labels = torch.where(tp_t[:, 0] > 0.5, 2, labels)
                labels = torch.where(fn_t[:, 0] > 0.5, 1, labels)
                labels = torch.where(fp_t[:, 0] > 0.5, 0, labels)
                with torch.no_grad():
                    counts = torch.stack(
                        [(labels == i).sum() for i in range(4)]
                    ).float().clamp_min(1.0)
                    class_weight = counts.sum() / (4.0 * counts)
                    class_weight = class_weight.clamp(0.25, 500.0)
                ce = F.cross_entropy(
                    res_logits.float(),
                    labels,
                    weight=class_weight.to(res_logits.device),
                )
                probs = torch.softmax(res_logits.float(), dim=1)
                sparse = 0.5 * (
                    _masked_tversky_loss(probs[:, 0:1], fp_t, torch.ones_like(fp_t))
                    + _masked_tversky_loss(probs[:, 1:2], fn_t, torch.ones_like(fn_t))
                )
                residual_loss = ce + float(
                    _cfg_get(m1, "V468_RESIDUAL_SPARSE_DICE_WEIGHT", 2.0)
                ) * sparse
                diag["v468_residual_ce_loss"] = float(ce.detach().item())
                diag["v468_residual_sparse_dice_loss"] = float(sparse.detach().item())
            else:
                residual_loss = (
                    _masked_focal_bce(
                        res_logits[:, 0:1], fp_t, torch.ones_like(fp_t)
                    )
                    + _masked_focal_bce(
                        res_logits[:, 1:2], fn_t, torch.ones_like(fn_t)
                    )
                )

        total = total + float(
            _cfg_get(m1, "V463_RESIDUAL_LOSS_WEIGHT", 1.0)
        ) * residual_loss
        diag.update({
            "v463_residual_loss": float(residual_loss.detach().item()),
            "v463_fp_target_area": float(hard_fp_t.detach().mean().item()),
            "v463_fn_target_area": float(hard_fn_t.detach().mean().item()),
            "v470_soft_fp_target_area": float(fp_t.detach().mean().item()),
            "v470_soft_fn_target_area": float(fn_t.detach().mean().item()),
        })

    candidate_loss, candidate_diag = _candidate_oracle_coverage_loss(
        cfg, candidates, base_logits, gt, aux, fp_t, fn_t
    )
    total = total + candidate_loss
    diag.update(candidate_diag)

    tau_dsc = aux.get("v463_ccv_tau_dsc")
    tau_nsd = aux.get("v463_ccv_tau_nsd")
    if (
        tau_dsc is not None
        and tau_nsd is not None
        and candidates is not None
        and candidates.shape[1] > 1
    ):
        k = candidates.shape[1] - 1
        valid_action, _, _ = _action_masks(aux, k, candidates.device)
        dsc_eps = float(_cfg_get(m1, "V463_CCV_DSC_EPS", 5.0e-4))
        nsd_eps = float(_cfg_get(m1, "V463_CCV_NSD_EPS", 5.0e-4))
        delta_dsc, delta_nsd, harm_t, pareto_t = _ccv_targets(
            candidates, base_logits, gt, dsc_eps, nsd_eps
        )
        delta_dsc = delta_dsc.clamp(-0.05, 0.05)
        delta_nsd = delta_nsd.clamp(-0.05, 0.05)
        harm_t = harm_t * valid_action.float()
        pareto_t = pareto_t * valid_action.float()

        target_scale = max(
            1.0e-4, float(_cfg_get(m1, "V468_CCV_TARGET_SCALE", 0.01))
        )
        reg_beta = float(_cfg_get(m1, "V468_CCV_REG_BETA", 0.20))
        reg_loss = _masked_smooth_l1(
            tau_dsc / target_scale,
            delta_dsc / target_scale,
            valid_action,
            beta=reg_beta,
        ) + _masked_smooth_l1(
            tau_nsd / target_scale,
            delta_nsd / target_scale,
            valid_action,
            beta=reg_beta,
        )
        harm_loss = _masked_bce_dynamic(
            aux["v463_ccv_harm_logit"], harm_t, valid_action, 20.0
        )
        pareto_loss = _masked_bce_dynamic(
            aux["v463_ccv_pareto_logit"], pareto_t, valid_action, 30.0
        )

        utility_pred = aux.get(
            "v469_ccv_direct_utility", aux["v463_ccv_utility"]
        )
        wd = float(_cfg_get(m1, "M2_UTILITY_DSC_WEIGHT", 0.60))
        wn = float(_cfg_get(m1, "M2_UTILITY_NSD_WEIGHT", 0.40))
        norm = max(wd + wn, EPS)
        wd, wn = wd / norm, wn / norm
        utility_t = wd * delta_dsc + wn * delta_nsd
        direct_utility_reg = _masked_smooth_l1(
            utility_pred / target_scale,
            utility_t / target_scale,
            valid_action,
            beta=reg_beta,
        )

        pos = (pareto_t > 0.5) & valid_action
        neg = (harm_t > 0.5) & valid_action
        valid_rank_case = pos.any(dim=1) & neg.any(dim=1)
        if valid_rank_case.any():
            pos_score = utility_pred.masked_fill(~pos, -1.0e4).max(dim=1).values
            neg_score = utility_pred.masked_fill(~neg, -1.0e4).max(dim=1).values
            rank_margin = float(
                _cfg_get(m1, "V463_CCV_RANK_MARGIN", 0.002)
            )
            rank_loss = _masked_mean(
                F.relu(rank_margin - (pos_score - neg_score)),
                valid_rank_case,
            )
        else:
            rank_loss = utility_pred.sum() * 0.0

        pos_margin = float(_cfg_get(m1, "V467_CCV_POS_MARGIN", 0.002))
        harm_margin = float(_cfg_get(m1, "V467_CCV_HARM_MARGIN", 0.002))
        case_margin = float(
            _cfg_get(m1, "V467_CCV_CASE_POS_MARGIN", 0.002)
        )
        deploy_pos_loss = _masked_mean(
            F.relu(pos_margin - utility_pred), pos
        )
        deploy_harm_loss = _masked_mean(
            F.relu(harm_margin + utility_pred), neg
        )

        has_valid = valid_action.any(dim=1)
        has_pos = pos.any(dim=1)
        best_pos = utility_pred.masked_fill(~pos, -1.0e4).max(dim=1).values
        case_pos_loss = _masked_mean(
            F.relu(case_margin - best_pos), has_pos
        )
        no_pos_valid = has_valid & ~has_pos
        best_valid = utility_pred.masked_fill(
            ~valid_action, -1.0e4
        ).max(dim=1).values
        no_pos_loss = _masked_mean(F.relu(best_valid), no_pos_valid)
        deploy_case_loss = case_pos_loss + no_pos_loss + rank_loss
        deploy_margin_loss = (
            deploy_pos_loss + deploy_harm_loss + deploy_case_loss
        )

        ccv_core = (
            float(_cfg_get(m1, "V463_CCV_DELTA_WEIGHT", 0.5)) * reg_loss
            + float(_cfg_get(m1, "V463_CCV_HARM_WEIGHT", 1.0)) * harm_loss
            + float(_cfg_get(m1, "V463_CCV_PARETO_WEIGHT", 1.0)) * pareto_loss
            + float(_cfg_get(m1, "V469_DIRECT_UTILITY_REG_WEIGHT", 1.0))
            * direct_utility_reg
            + float(_cfg_get(m1, "V463_CCV_RANK_WEIGHT", 1.0)) * rank_loss
        )
        ccv_start = int(_cfg_get(
            m1, "V470_CCV_START_EPOCH",
            _cfg_get(m1, "V469_CCV_START_EPOCH", 0),
        ))
        ccv_ramp = int(_cfg_get(
            m1, "V470_CCV_RAMP_EPOCHS",
            _cfg_get(m1, "V469_CCV_RAMP_EPOCHS", 10),
        ))
        deploy_start = int(_cfg_get(
            m1, "V470_DEPLOY_START_EPOCH",
            _cfg_get(m1, "V469_DEPLOY_START_EPOCH", 5),
        ))
        deploy_ramp = int(_cfg_get(
            m1, "V470_DEPLOY_RAMP_EPOCHS",
            _cfg_get(m1, "V469_DEPLOY_RAMP_EPOCHS", 10),
        ))
        ccv_weight = _ramp_weight(epoch, ccv_start, ccv_ramp)
        deploy_weight = _ramp_weight(epoch, deploy_start, deploy_ramp)
        deploy_strength = float(
            _cfg_get(m1, "V467_CCV_DEPLOY_MARGIN_WEIGHT", 1.0)
        )
        ccv_loss = ccv_core + deploy_weight * deploy_strength * deploy_margin_loss
        total = total + ccv_weight * float(
            _cfg_get(m1, "V463_CCV_LOSS_WEIGHT", 1.0)
        ) * ccv_loss

        changed = aux.get(
            "v463_ccv_changed", harm_t.new_zeros(harm_t.shape[0])
        )
        valid_count = valid_action.float().sum().clamp_min(1.0)
        valid_positive_rate = (pareto_t * valid_action.float()).sum() / valid_count
        valid_harm_rate = (harm_t * valid_action.float()).sum() / valid_count
        masked_delta_dsc = delta_dsc.masked_fill(~valid_action, -1.0e4)
        oracle_with_preserve = torch.cat(
            [delta_dsc.new_zeros((delta_dsc.shape[0], 1)), masked_delta_dsc],
            dim=1,
        ).max(dim=1).values
        detached_flag = aux.get("v469_ccv_inputs_detached")
        ccv_base_scale = aux.get("v470_ccv_to_base_grad_scale")
        ccv_m1_scale = aux.get("v470_ccv_to_m1_grad_scale")
        aux_base_scale = aux.get("v470_aux_to_base_grad_scale")
        diag.update({
            "v463_ccv_loss": float(ccv_loss.detach().item()),
            "v463_ccv_reg_loss": float(reg_loss.detach().item()),
            "v463_ccv_harm_loss": float(harm_loss.detach().item()),
            "v463_ccv_pareto_loss": float(pareto_loss.detach().item()),
            "v463_ccv_rank_loss": float(rank_loss.detach().item()),
            "v463_ccv_rank_valid_case_rate": float(
                valid_rank_case.float().mean().detach().item()
            ),
            "v463_oracle_delta_dsc": float(
                oracle_with_preserve.detach().mean().item()
            ),
            "v463_positive_candidate_rate": float(
                valid_positive_rate.detach().item()
            ),
            "v463_harmful_candidate_rate": float(
                valid_harm_rate.detach().item()
            ),
            "v463_ccv_changed_rate": float(
                changed.detach().float().mean().item()
            ),
            "v467_deploy_margin_loss": float(
                deploy_margin_loss.detach().item()
            ),
            "v467_deploy_pos_loss": float(deploy_pos_loss.detach().item()),
            "v467_deploy_harm_loss": float(deploy_harm_loss.detach().item()),
            "v467_deploy_case_loss": float(deploy_case_loss.detach().item()),
            "v467_utility_mean": float(
                _masked_mean(utility_pred.detach(), valid_action).item()
            ),
            "v467_utility_max": float(
                best_valid[has_valid].detach().mean().item()
                if has_valid.any() else 0.0
            ),
            "v468_utility_target_mean": float(
                _masked_mean(utility_t.detach(), valid_action).item()
            ),
            "v469_direct_utility_reg_loss": float(
                direct_utility_reg.detach().item()
            ),
            "v469_ccv_weight": float(ccv_weight),
            "v469_deploy_weight": float(deploy_weight),
            "v469_ccv_inputs_detached_rate": float(
                detached_flag.detach().float().mean().item()
                if isinstance(detached_flag, torch.Tensor) else 0.0
            ),
            "v470_ccv_to_base_grad_scale": float(
                ccv_base_scale.detach().float().mean().item()
                if isinstance(ccv_base_scale, torch.Tensor) else 1.0
            ),
            "v470_ccv_to_m1_grad_scale": float(
                ccv_m1_scale.detach().float().mean().item()
                if isinstance(ccv_m1_scale, torch.Tensor) else 1.0
            ),
            "v470_aux_to_base_grad_scale": float(
                aux_base_scale.detach().float().mean().item()
                if isinstance(aux_base_scale, torch.Tensor) else 1.0
            ),
        })

    return total, diag
