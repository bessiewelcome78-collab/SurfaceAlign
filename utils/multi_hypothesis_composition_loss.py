"""Losses and root-cause diagnostics for GEOTR-V4C/V4D/V4E/V4F/V4G.

V4C keeps Stage-1 Transport unchanged and makes Stage-2 explicit about error
responsibility:

    class 0: Correct / Preserve
    class 1: False Negative -> Add probability mass only
    class 2: False Positive -> Remove probability mass only

The localizer uses standard class-weighted cross entropy + per-type soft Dice.
The HOW branch is optimized by the project's standard BCE + Dice segmentation
loss.  A small L1 preserve term on GT-correct anchor pixels is optional and is
only a regularizer; no oracle, benefit/harm gate or selector is used for the
training/deployed prediction.

GT-only oracle and FN/FP pairing statistics below are diagnostics.  They are
never used to form the training objective or the deployed output.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple
import math
import torch
import torch.nn.functional as F

from .sparc_hr_loss import compute_sparc_hr_loss

EPS = 1.0e-4


def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def _first_present(mapping: Dict[str, Any], *keys: str) -> Any:
    """Return the first explicitly present compatibility key.

    ``dict.get(new_key, mapping[legacy_key])`` is not a lazy fallback in
    Python: ``mapping[legacy_key]`` is evaluated before ``dict.get`` runs.
    SLR 2.3 intentionally permits current-only and legacy-only auxiliary
    dictionaries, so compatibility lookups must test membership explicitly.
    """
    for key in keys:
        if key in mapping:
            return mapping[key]
    raise KeyError(f"None of the required auxiliary keys is present: {keys}")


def _target_3d(masks: torch.Tensor) -> torch.Tensor:
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise ValueError(f"Expected masks [B,H,W] or [B,1,H,W], got {tuple(masks.shape)}")
    return (masks > 0.5).to(dtype=masks.dtype)


def _joint_oracle_envelope(
    candidate_probs: torch.Tensor,
    target: torch.Tensor,
    patch_size: int,
    iterations: int = 1,
) -> Dict[str, torch.Tensor]:
    """GT-only legacy validation helper; never contributes to training."""
    del iterations
    if candidate_probs.ndim != 4:
        raise ValueError(f"candidate_probs must be [B,K,H,W], got {tuple(candidate_probs.shape)}")
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    if target.ndim != 3:
        raise ValueError(f"target must be [B,H,W] or [B,1,H,W], got {tuple(target.shape)}")
    b, k, h, w = candidate_probs.shape
    if target.shape != (b, h, w):
        raise ValueError(f"target shape {tuple(target.shape)} does not match candidates {(b,h,w)}")
    ps = max(1, int(patch_size))
    p = candidate_probs.detach().clamp(EPS, 1.0 - EPS)
    y = target.detach().to(p)[:, None].expand_as(p)
    risk = F.binary_cross_entropy(p, y, reduction="none")
    pad_h, pad_w = (-h) % ps, (-w) % ps
    risk_pad = F.pad(risk, (0, pad_w, 0, pad_h), value=0.0)
    valid = F.pad(
        torch.ones(1, 1, h, w, device=p.device, dtype=p.dtype),
        (0, pad_w, 0, pad_h), value=0.0,
    )
    hp, wp = h + pad_h, w + pad_w
    patch_sum = F.avg_pool2d(
        risk_pad.reshape(b * k, 1, hp, wp), ps, stride=ps, divisor_override=1
    ).reshape(b, k, hp // ps, wp // ps)
    valid_count = F.avg_pool2d(valid, ps, stride=ps, divisor_override=1).clamp_min(1.0)
    idx = (patch_sum / valid_count).argmin(dim=1)
    idx_up = F.interpolate(idx[:, None].float(), size=(hp, wp), mode="nearest").long()[:, 0, :h, :w]
    out = p.gather(1, idx_up[:, None])[:, 0]
    return {"prob": out, "index": idx}


def _dice_per_case(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prob.ndim == 4 and prob.shape[1] == 1:
        prob = prob[:, 0]
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    inter = (prob * target).sum(dim=(-2, -1))
    den = prob.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return (2.0 * inter + EPS) / (den + EPS)


def _dice_loss(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (1.0 - _dice_per_case(prob, target)).mean()


def _flow_smoothness(flow: torch.Tensor) -> torch.Tensor:
    dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    return 0.5 * (dx.abs().mean() + dy.abs().mean())


def _seg_loss(
    logits: torch.Tensor,
    prob: torch.Tensor,
    target: torch.Tensor,
    ce_weight: float,
    dice_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target4 = target[:, None]
    bce = F.binary_cross_entropy_with_logits(logits.float(), target4.float()).to(prob)
    dice = _dice_loss(prob[:, 0], target)
    return ce_weight * bce + dice_weight * dice, bce, dice


def _typed_targets(anchor: torch.Tensor, target: torch.Tensor):
    """Return class target {0=correct,1=FN,2=FP} and binary masks."""
    pred = anchor[:, 0].detach() >= 0.5
    gt = target.detach() >= 0.5
    fn = gt & (~pred)
    fp = (~gt) & pred
    correct = ~(fn | fp)
    cls = torch.zeros_like(target, dtype=torch.long)
    cls[fn] = 1
    cls[fp] = 2
    return cls, correct, fn, fp


def _typed_localization_loss(
    typed_logits: torch.Tensor,
    typed_prob: torch.Tensor,
    cls_target: torch.Tensor,
    fn: torch.Tensor,
    fp: torch.Tensor,
    ce_weight: float,
    dice_weight: float,
    error_class_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Standard class-weighted CE handles the extreme Correct:Error imbalance.
    weights = typed_logits.new_tensor([1.0, error_class_weight, error_class_weight])
    ce = F.cross_entropy(typed_logits.float(), cls_target.long(), weight=weights.float()).to(typed_prob)
    fn_dice_loss = _dice_loss(typed_prob[:, 1], fn.to(typed_prob))
    fp_dice_loss = _dice_loss(typed_prob[:, 2], fp.to(typed_prob))
    typed_dice = 0.5 * (fn_dice_loss + fp_dice_loss)
    loss = ce_weight * ce + dice_weight * typed_dice
    return loss, ce, typed_dice, fn_dice_loss, fp_dice_loss


def _safe_ratio(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    return torch.where(den > 0, num / den.clamp_min(EPS), torch.zeros_like(num))


def _binary_stats(prob: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    if prob.ndim == 4:
        prob = prob[:, 0]
    pred = prob.detach() >= 0.5
    gt = target.detach() >= 0.5
    tp = (pred & gt).float().sum()
    fp = (pred & ~gt).float().sum()
    fn = (~pred & gt).float().sum()
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _class_stats(pred_cls: torch.Tensor, gt_mask: torch.Tensor, cls_id: int) -> Dict[str, torch.Tensor]:
    pred = pred_cls.detach() == int(cls_id)
    gt = gt_mask.detach()
    tp = (pred & gt).float().sum()
    fp = (pred & ~gt).float().sum()
    fn = (~pred & gt).float().sum()
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    maskf = mask.to(x)
    return (x * maskf).sum() / maskf.sum().clamp_min(1.0)


def _aefr_five_state_targets(
    anchor: torch.Tensor,
    target: torch.Tensor,
    region: torch.Tensor,
    boundary_mask: torch.Tensor,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """Construct GT-only typed residual-action labels for training/diagnostics.

    Class semantics inside the deterministic ROI are:
      0 KEEP
      1 BOUNDARY_ADD     (FN near current anchor boundary)
      2 BOUNDARY_REMOVE  (FP near current anchor boundary)
      3 INTERIOR_ADD     (FN outside the boundary band)
      4 INTERIOR_REMOVE  (FP outside the boundary band)

    ``anchor`` is detached before deriving hard state.  These labels never enter
    the deployable forward; they are ordinary supervised targets analogous to a
    segmentation label, but identify the *residual action* relative to the
    current factual anchor.
    """
    if anchor.ndim != 4 or anchor.shape[1] != 1:
        raise ValueError(f"anchor must be [B,1,H,W], got {tuple(anchor.shape)}")
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    if target.ndim != 3:
        raise ValueError(f"target must be [B,H,W], got {tuple(target.shape)}")
    roi = region[:, 0].detach() > 0.5
    bnd = boundary_mask[:, 0].detach() > 0.5
    pred = anchor[:, 0].detach() >= 0.5
    gt = target.detach() >= 0.5
    fn = roi & (~pred) & gt
    fp = roi & pred & (~gt)
    badd = fn & bnd
    bremove = fp & bnd
    iadd = fn & (~bnd)
    iremove = fp & (~bnd)
    keep = roi & (~(fn | fp))
    cls = torch.zeros_like(target, dtype=torch.long)
    cls[badd] = 1
    cls[bremove] = 2
    cls[iadd] = 3
    cls[iremove] = 4
    return cls, (keep, badd, bremove, iadd, iremove)


def _macro_present_class_ce(
    logits: torch.Tensor,
    cls_target: torch.Tensor,
    region: torch.Tensor,
    num_classes: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standard cross-entropy, macro-averaged over classes present in the ROI.

    This avoids an arbitrary hand-tuned class-weight vector when KEEP dominates
    sparse post-Geometry ROIs.  It remains ordinary CE; only the reduction is
    class-balanced.  Returns (loss, present_class_count).
    """
    if logits.ndim != 4 or logits.shape[1] != num_classes:
        raise ValueError(f"state logits must be [B,{num_classes},H,W], got {tuple(logits.shape)}")
    roi = region[:, 0].detach() > 0.5
    elem = F.cross_entropy(logits.float(), cls_target.long(), reduction="none").to(logits)
    terms = []
    for cid in range(num_classes):
        mask = roi & (cls_target == cid)
        if bool(mask.any()):
            terms.append(elem[mask].mean())
    if not terms:
        return logits.sum() * 0.0, logits.new_zeros(())
    return torch.stack(terms).mean(), logits.new_tensor(float(len(terms)))


def _five_state_metrics(
    logits: torch.Tensor,
    cls_target: torch.Tensor,
    region: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Pixel-level precision/recall/F1 for the five residual states in ROI."""
    roi = region[:, 0].detach() > 0.5
    pred = logits.detach().argmax(dim=1)
    out: Dict[str, torch.Tensor] = {}
    names = ("keep", "boundary_add", "boundary_remove", "interior_add", "interior_remove")
    f1_terms = []
    for cid, name in enumerate(names):
        gt_c = roi & (cls_target == cid)
        pred_c = roi & (pred == cid)
        tp = (gt_c & pred_c).float().sum()
        fp = ((~gt_c) & pred_c & roi).float().sum()
        fn = (gt_c & (~pred_c)).float().sum()
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
        out[f"{name}_precision"] = precision
        out[f"{name}_recall"] = recall
        out[f"{name}_f1"] = f1
        out[f"{name}_count"] = gt_c.float().sum()
        if bool(gt_c.any()):
            f1_terms.append(f1)
    out["macro_f1"] = torch.stack(f1_terms).mean() if f1_terms else logits.new_zeros(())
    out["pred_edit_fraction"] = _safe_ratio(((pred != 0) & roi).float().sum(), roi.float().sum())
    return out



def _aefr_soft_ownership_targets(
    anchor: torch.Tensor,
    target: torch.Tensor,
    region: torch.Tensor,
) -> torch.Tensor:
    """Continuous KEEP/ADD/REMOVE ownership target in the factual ROI.

    For probability p and binary label y:
      KEEP   = y*p + (1-y)*(1-p)
      ADD    = y*(1-p)
      REMOVE = (1-y)*p
    The distribution sums to one and satisfies ADD-REMOVE == y-p, i.e. the
    signed ownership target is exactly the probability residual and opposite
    the BCE logit gradient.  The target is detached and used only for training
    and diagnostics; GT never enters the deployable refiner forward.
    """
    if anchor.ndim != 4 or anchor.shape[1] != 1:
        raise ValueError(f"anchor must be [B,1,H,W], got {tuple(anchor.shape)}")
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    if target.ndim != 3:
        raise ValueError(f"target must be [B,H,W], got {tuple(target.shape)}")
    p = anchor[:, 0].detach().clamp(EPS, 1.0 - EPS)
    y = target.detach().to(p).clamp(0.0, 1.0)
    keep = y * p + (1.0 - y) * (1.0 - p)
    add = y * (1.0 - p)
    remove = (1.0 - y) * p
    out = torch.stack([keep, add, remove], dim=1)
    # Numerical contract: valid probability simplex everywhere.
    out = out / out.sum(dim=1, keepdim=True).clamp_min(EPS)
    return out


def _soft_ownership_ce(
    logits: torch.Tensor,
    soft_target: torch.Tensor,
    region: torch.Tensor,
) -> torch.Tensor:
    """Standard soft-label cross entropy, averaged by pixel in the ROI.

    No class reweighting, focal term, utility loss or safety gate is used.
    Sparse residual ownership is represented by the target distribution itself.
    """
    if logits.ndim != 4 or logits.shape[1] != 3:
        raise ValueError(f"ownership logits must be [B,3,H,W], got {tuple(logits.shape)}")
    if soft_target.shape != logits.shape:
        raise ValueError(f"ownership target shape mismatch: {tuple(soft_target.shape)} vs {tuple(logits.shape)}")
    roi = region[:, 0].detach() > 0.5
    elem = -(soft_target.to(logits) * F.log_softmax(logits.float(), dim=1).to(logits)).sum(dim=1)
    if not bool(roi.any()):
        return logits.sum() * 0.0
    return elem[roi].mean()


def _three_ownership_metrics(
    logits: torch.Tensor,
    soft_target: torch.Tensor,
    anchor: torch.Tensor,
    target: torch.Tensor,
    region: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Diagnostics for SRO-Exact. Hard labels are diagnostic-only.

    Training remains fully soft. Hard KEEP/ADD/REMOVE P/R/F1 is reported only
    to make over-editing and direction confusion interpretable.
    """
    roi = region[:, 0].detach() > 0.5
    probs = torch.softmax(logits.detach(), dim=1)
    pred_cls = probs.argmax(dim=1)
    pred_edit_mass = (probs[:, 1] + probs[:, 2])
    target_edit_mass = soft_target[:, 1] + soft_target[:, 2]
    pred_signed = probs[:, 1] - probs[:, 2]
    target_signed = soft_target[:, 1] - soft_target[:, 2]

    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    y = target.detach() >= 0.5
    ah = anchor[:, 0].detach() >= 0.5
    hard = torch.zeros_like(target, dtype=torch.long)
    hard[(~ah) & y] = 1  # ADD
    hard[ah & (~y)] = 2  # REMOVE

    out: Dict[str, torch.Tensor] = {}
    f1s = []
    for cid, name in enumerate(("keep", "add", "remove")):
        gt_c = roi & (hard == cid)
        pr_c = roi & (pred_cls == cid)
        tp = (gt_c & pr_c).float().sum()
        fp = ((~gt_c) & pr_c & roi).float().sum()
        fn = (gt_c & (~pr_c)).float().sum()
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
        out[f"{name}_precision"] = precision
        out[f"{name}_recall"] = recall
        out[f"{name}_f1"] = f1
        out[f"{name}_count"] = gt_c.float().sum()
        if bool(gt_c.any()):
            f1s.append(f1)
    out["macro_f1"] = torch.stack(f1s).mean() if f1s else logits.new_zeros(())
    out["pred_edit_fraction"] = _safe_ratio(((pred_cls != 0) & roi).float().sum(), roi.float().sum())
    out["pred_edit_mass"] = _masked_mean(pred_edit_mass, roi)
    out["target_edit_mass"] = _masked_mean(target_edit_mass, roi)
    out["edit_mass_abs_error"] = (out["pred_edit_mass"] - out["target_edit_mass"]).abs()
    out["signed_action_mae"] = _masked_mean((pred_signed - target_signed).abs(), roi)
    out["signed_target_abs_mean"] = _masked_mean(target_signed.abs(), roi)
    return out


def _aefr_intervention_targets(
    anchor: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hard post-Geometry intervention targets.

    Returns [B,1,H,W] tensors for error/editness, ADD-direction target and
    signed action target, plus boolean FN/FP masks.  Correct hard predictions
    have exactly zero edit target even when their probabilities are not 0/1.
    """
    if anchor.ndim != 4 or anchor.shape[1] != 1:
        raise ValueError(f"anchor must be [B,1,H,W], got {tuple(anchor.shape)}")
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    if target.ndim != 3:
        raise ValueError(f"target must be [B,H,W], got {tuple(target.shape)}")
    y = target.detach().to(anchor) >= 0.5
    ah = anchor[:, 0].detach() >= 0.5
    fn = (~ah) & y
    fp = ah & (~y)
    err = fn | fp
    error_target = err[:, None].to(anchor)
    direction_target = fn[:, None].to(anchor)  # ADD=1, REMOVE=0 on true edits.
    signed_target = fn[:, None].to(anchor) - fp[:, None].to(anchor)
    return error_target, direction_target, signed_target, fn, fp


def _aefr_boundary_magnitude_target(
    anchor: torch.Tensor,
    max_displacement_px: float,
) -> torch.Tensor:
    """Approximate minimum boundary displacement in pixel-center units.

    The target is derived from the *current factual anchor geometry*, while GT
    is used only by the caller to decide which residual pixels require action.
    For each pixel we find the first Chebyshev ring containing the current hard
    decision boundary and add half a pixel to cross the bilinear decision
    surface. Values are capped by the deployed boundary operator range.

    This is an operator-aligned regression target, not an oracle deployment
    input: it is used only during training and never enters forward inference.
    """
    if anchor.ndim != 4 or anchor.shape[1] != 1:
        raise ValueError(f"anchor must be [B,1,H,W], got {tuple(anchor.shape)}")
    dmax = float(max(max_displacement_px, 0.5))
    hard = (anchor.detach() >= 0.5).to(anchor)
    dil = F.max_pool2d(hard, kernel_size=3, stride=1, padding=1)
    ero = -F.max_pool2d(-hard, kernel_size=3, stride=1, padding=1)
    edge = (dil - ero).abs() > 0.5
    dist = torch.full_like(anchor, dmax)
    dist = torch.where(edge, torch.zeros_like(dist), dist)
    assigned = edge.clone()
    for d in range(1, int(math.ceil(dmax)) + 1):
        k = 2 * d + 1
        near = F.max_pool2d(edge.to(anchor), kernel_size=k, stride=1, padding=d) > 0.5
        new = near & (~assigned)
        dist = torch.where(new, torch.full_like(dist, float(d)), dist)
        assigned = assigned | near
    return (dist + 0.5).clamp(min=0.5, max=dmax)


def _masked_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"SmoothL1 shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    m = mask.to(pred)
    if m.shape != pred.shape:
        m = m.expand_as(pred)
    if not bool((m > 0).any()):
        return pred.sum() * 0.0
    raw = F.smooth_l1_loss(pred.float(), target.float(), reduction="none").to(pred)
    return (raw * m).sum() / m.sum().clamp_min(1.0)


def _masked_bce_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    """Standard BCEWithLogits on a declared support; empty support -> zero."""
    if logits.shape != target.shape:
        raise ValueError(f"BCE shape mismatch: {tuple(logits.shape)} vs {tuple(target.shape)}")
    m = mask.to(logits)
    if m.shape != logits.shape:
        m = m.expand_as(logits)
    if not bool((m > 0).any()):
        return logits.sum() * 0.0
    pw = logits.new_tensor(float(max(pos_weight, 1.0e-6)))
    raw = F.binary_cross_entropy_with_logits(
        logits.float(), target.float(), reduction="none", pos_weight=pw.float()
    ).to(logits)
    return (raw * m).sum() / m.sum().clamp_min(1.0)


def _masked_binary_dice_loss(
    prob: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Standard soft Dice loss on a declared support."""
    m = mask.to(prob)
    if m.shape != prob.shape:
        m = m.expand_as(prob)
    inter = (prob * target.to(prob) * m).sum(dim=(1, 2, 3))
    den = ((prob + target.to(prob)) * m).sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * inter + EPS) / (den + EPS)).mean()


def _aefr_intervention_metrics(
    error_prob: torch.Tensor,
    edit_prob: torch.Tensor,
    direction_prob: torch.Tensor,
    signed_action: torch.Tensor,
    anchor: torch.Tensor,
    target: torch.Tensor,
    region: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Diagnostics for intervention-factorized residual refinement."""
    error_target, direction_target, signed_target, fn, fp = _aefr_intervention_targets(anchor, target)
    err = error_target[:, 0] > 0.5
    roi = region[:, 0].detach() > 0.5

    def cls_stats(pred: torch.Tensor, gt: torch.Tensor, support: torch.Tensor):
        pred = pred & support; gt = gt & support
        tp = (pred & gt).float().sum()
        fpp = (pred & (~gt) & support).float().sum()
        fnn = ((~pred) & gt & support).float().sum()
        p = _safe_ratio(tp, tp + fpp); r = _safe_ratio(tp, tp + fnn)
        f = _safe_ratio(2.0 * p * r, p + r)
        return p, r, f

    full = torch.ones_like(err, dtype=torch.bool)
    loc_pred = error_prob[:, 0].detach() >= 0.5
    loc_p, loc_r, loc_f1 = cls_stats(loc_pred, err, full)

    edit_pred = edit_prob[:, 0].detach() >= 0.5
    edit_p, edit_r, edit_f1 = cls_stats(edit_pred, err, roi)
    dir_add = direction_prob[:, 0].detach() >= 0.5
    true_edit_roi = roi & err
    direction_correct = (dir_add & fn) | ((~dir_add) & fp)
    direction_acc = _safe_ratio((direction_correct & true_edit_roi).float().sum(), true_edit_roi.float().sum())

    pred_add = roi & edit_pred & dir_add
    pred_remove = roi & edit_pred & (~dir_add)
    add_p, add_r, add_f1 = cls_stats(pred_add, fn, roi)
    rem_p, rem_r, rem_f1 = cls_stats(pred_remove, fp, roi)

    signed = signed_action[:, 0].detach()
    signed_t = signed_target[:, 0]
    signed_mae = _masked_mean((signed - signed_t).abs(), roi)
    target_abs = _masked_mean(signed_t.abs(), roi)
    return {
        "localizer_precision": loc_p, "localizer_recall": loc_r, "localizer_f1": loc_f1,
        "edit_precision": edit_p, "edit_recall": edit_r, "edit_f1": edit_f1,
        "direction_accuracy": direction_acc,
        "add_precision": add_p, "add_recall": add_r, "add_f1": add_f1,
        "remove_precision": rem_p, "remove_recall": rem_r, "remove_f1": rem_f1,
        "pred_edit_fraction": _safe_ratio((edit_pred & roi).float().sum(), roi.float().sum()),
        "target_edit_fraction": _safe_ratio((err & roi).float().sum(), roi.float().sum()),
        "signed_action_mae": signed_mae,
        "zero_action_mae": target_abs,
        "signed_advantage": target_abs - signed_mae,
        "error_prob_mean_error": _masked_mean(error_prob[:, 0].detach(), err),
        "error_prob_mean_correct": _masked_mean(error_prob[:, 0].detach(), ~err),
        "edit_prob_mean_error_roi": _masked_mean(edit_prob[:, 0].detach(), roi & err),
        "edit_prob_mean_correct_roi": _masked_mean(edit_prob[:, 0].detach(), roi & (~err)),
    }


def _pairing_stats(anchor: torch.Tensor, target: torch.Tensor, radius: int) -> Dict[str, torch.Tensor]:
    """Local FN/FP co-occurrence proxy for residual transportability.

    This is deliberately called a *proxy*: it does not solve optimal transport or
    one-to-one matching.  A residual FN/FP pixel is considered locally pairable
    when the opposite error type lies within ``radius`` pixels.
    """
    _, _, fn, fp = _typed_targets(anchor, target)
    r = max(0, int(radius))
    if r == 0:
        near_fp = fp
        near_fn = fn
    else:
        k = 2 * r + 1
        near_fp = F.max_pool2d(fp.float()[:, None], k, stride=1, padding=r)[:, 0] > 0
        near_fn = F.max_pool2d(fn.float()[:, None], k, stride=1, padding=r)[:, 0] > 0
    paired_fn = fn & near_fp
    paired_fp = fp & near_fn

    fn_n = fn.float().sum(dim=(-2, -1))
    fp_n = fp.float().sum(dim=(-2, -1))
    err_n = fn_n + fp_n
    paired_fn_n = paired_fn.float().sum(dim=(-2, -1))
    paired_fp_n = paired_fp.float().sum(dim=(-2, -1))
    paired_n = paired_fn_n + paired_fp_n
    return {
        "fn_rate": fn.float().mean(dim=(-2, -1)).mean(),
        "fp_rate": fp.float().mean(dim=(-2, -1)).mean(),
        "paired_fn_fraction": _safe_ratio(paired_fn_n, fn_n).mean(),
        "paired_fp_fraction": _safe_ratio(paired_fp_n, fp_n).mean(),
        "paired_error_fraction": _safe_ratio(paired_n, err_n).mean(),
        "unmatched_error_fraction": _safe_ratio((err_n - paired_n).clamp_min(0.0), err_n).mean(),
    }


def _compose_typed(
    anchor: torch.Tensor,
    q_fn: torch.Tensor,
    q_fp: torch.Tensor,
    add_magnitude: torch.Tensor,
    remove_magnitude: torch.Tensor,
) -> torch.Tensor:
    out = anchor + q_fn * add_magnitude * (1.0 - anchor) - q_fp * remove_magnitude * anchor
    return out.clamp(EPS, 1.0 - EPS)



def _masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Standard Smooth-L1 restricted to a binary support; empty support -> 0."""
    if pred.ndim == 4 and pred.shape[1] == 1:
        pred = pred[:, 0]
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    maskf = mask.to(pred)
    raw = F.smooth_l1_loss(pred.float(), target.float(), reduction="none").to(pred)
    return (raw * maskf).sum() / maskf.sum().clamp_min(1.0)


def _weighted_smooth_l1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    raw = F.smooth_l1_loss(pred.float(), target.float(), reduction="none").to(pred)
    w = weight.to(pred)
    return (raw * w).sum() / w.sum().clamp_min(1.0)


def _compose_v4d(
    anchor: torch.Tensor,
    severity_fn: torch.Tensor,
    severity_fp: torch.Tensor,
    add_logit_magnitude: torch.Tensor,
    remove_logit_magnitude: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Monotonic logit-space V4D deployment operator."""
    z = torch.logit(anchor.clamp(EPS, 1.0 - EPS))
    logits = z + severity_fn * add_logit_magnitude - severity_fp * remove_logit_magnitude
    return torch.sigmoid(logits).clamp(EPS, 1.0 - EPS), logits




def _v4f_region_balanced_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """Balanced binary BCE inside the proposal pool.

    Positive and negative candidate regions are averaged independently before
    combination.  This prevents the large NoEdit area from dominating merely
    by pixel count while preserving an ordinary BCE objective.
    """
    if logits.ndim == 4:
        logits = logits[:, 0]
    if candidate_mask.ndim == 4:
        candidate_mask = candidate_mask[:, 0]
    cand = candidate_mask.detach() >= 0.5
    tgt = target.detach() >= 0.5
    elem = F.binary_cross_entropy_with_logits(logits.float(), tgt.float(), reduction="none").to(logits)
    pos = cand & tgt
    neg = cand & (~tgt)
    pos_loss = _masked_mean(elem, pos)
    neg_loss = _masked_mean(elem, neg)
    pos_n = pos.float().sum()
    neg_n = neg.float().sum()
    if float(pos_n.detach()) <= 0.0:
        return neg_loss
    if float(neg_n.detach()) <= 0.0:
        return pos_loss
    return 0.5 * (pos_loss + neg_loss)


def _v4f_direction_ce(
    direction_logits: torch.Tensor,
    fn: torch.Tensor,
    fp: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """Unweighted Add/Remove CE only where a true residual edit is proposed."""
    if candidate_mask.ndim == 4:
        candidate_mask = candidate_mask[:, 0]
    support = (candidate_mask.detach() >= 0.5) & (fn | fp)
    if float(support.float().sum().detach()) <= 0.0:
        return direction_logits.sum() * 0.0
    target = torch.zeros_like(fn, dtype=torch.long)
    target[fp] = 1  # 0=Add(FN), 1=Remove(FP)
    elem = F.cross_entropy(direction_logits.float(), target.long(), reduction="none").to(direction_logits)
    return _masked_mean(elem, support)


def _compose_v4f_teacher(
    anchor: torch.Tensor,
    fn: torch.Tensor,
    fp: torch.Tensor,
    dose_add: torch.Tensor,
    dose_remove: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GT-support bounded fractional dose teacher; policy owns NoEdit."""
    fn4 = fn[:, None].to(anchor)
    fp4 = fp[:, None].to(anchor)
    prob = anchor + fn4 * dose_add * (1.0 - anchor) - fp4 * dose_remove * anchor
    prob = prob.clamp(EPS, 1.0 - EPS)
    return prob, torch.logit(prob)


def _compose_v4f_policy(
    anchor: torch.Tensor,
    candidate_mask: torch.Tensor,
    edit_prob: torch.Tensor,
    direction_prob: torch.Tensor,
    dose_add: torch.Tensor,
    dose_remove: torch.Tensor,
    *,
    hard: bool,
    threshold: float,
) -> torch.Tensor:
    """Compose either differentiable expected action or hard selective action."""
    cand = (candidate_mask >= 0.5)
    add_action = (anchor + dose_add * (1.0 - anchor)).clamp(EPS, 1.0 - EPS)
    rem_action = (anchor - dose_remove * anchor).clamp(EPS, 1.0 - EPS)
    if not hard:
        p_edit = cand.to(anchor) * edit_prob
        p_add = p_edit * direction_prob[:, 0:1]
        p_rem = p_edit * direction_prob[:, 1:2]
        p_no = (1.0 - p_add - p_rem).clamp(0.0, 1.0)
        return (p_no * anchor + p_add * add_action + p_rem * rem_action).clamp(EPS, 1.0 - EPS)
    edit = cand & (edit_prob >= float(threshold))
    direction = direction_prob.argmax(dim=1, keepdim=True)
    add = edit & (direction == 0)
    rem = edit & (direction == 1)
    out = torch.where(add, add_action, anchor)
    out = torch.where(rem, rem_action, out)
    return out.clamp(EPS, 1.0 - EPS)


def _v4f_policy_stats(
    edit_prob: torch.Tensor,
    direction_prob: torch.Tensor,
    candidate_mask: torch.Tensor,
    fn: torch.Tensor,
    fp: torch.Tensor,
    threshold: float,
) -> Dict[str, torch.Tensor]:
    cand = candidate_mask[:, 0].detach() >= 0.5
    pred_edit = cand & (edit_prob[:, 0].detach() >= float(threshold))
    true_edit = fn | fp
    tp = (pred_edit & true_edit).float().sum()
    fp_count = (pred_edit & ~true_edit).float().sum()
    fn_count = ((~pred_edit) & true_edit).float().sum()
    precision = _safe_ratio(tp, tp + fp_count)
    recall = _safe_ratio(tp, tp + fn_count)
    direction = direction_prob.detach().argmax(dim=1)
    dir_ok = ((direction == 0) & fn) | ((direction == 1) & fp)
    direction_acc = _safe_ratio((pred_edit & true_edit & dir_ok).float().sum(), (pred_edit & true_edit).float().sum())
    proposal_recall = _safe_ratio((cand & true_edit).float().sum(), true_edit.float().sum())
    proposal_precision = _safe_ratio((cand & true_edit).float().sum(), cand.float().sum())
    return {
        "precision": precision,
        "recall": recall,
        "direction_accuracy": direction_acc,
        "proposal_recall": proposal_recall,
        "proposal_precision": proposal_precision,
        "candidate_rate": cand.float().mean(),
        "execution_rate": pred_edit.float().mean(),
    }




def _slr_ucdrt_patch_targets(
    anchor_patch: torch.Tensor,
    local_target: torch.Tensor,
    anchor_sdf_patch: torch.Tensor,
    target_sdf_patch: torch.Tensor,
    valid_patch: torch.Tensor,
    *,
    boundary_radius_px: float,
    max_boundary_displacement_px: float,
    max_interior_logit_step: float,
    target_margin: float,
):
    """Build KEEP/MOVE/ADD/REMOVE targets for local UCDRT patches.

    Shapes are [B,K,1,R,R] except state target [B,K,R,R].  MOVE owns hard
    residuals inside the deterministic anchor-SDF boundary band; ADD/REMOVE own
    hard FN/FP residuals outside that band. KEEP owns every other valid pixel.
    """
    ap = anchor_patch.detach().clamp(EPS, 1.0 - EPS)
    tp = local_target.detach().to(ap)
    if tp.shape != ap.shape:
        raise ValueError(f"UCDRT target patch mismatch: {tuple(tp.shape)} vs {tuple(ap.shape)}")
    valid = valid_patch.detach().to(ap) > 0.5
    ah = ap >= 0.5
    gt = tp >= 0.5
    fn = valid & (~ah) & gt
    fp = valid & ah & (~gt)
    bnd = valid & (anchor_sdf_patch.detach().abs() <= float(boundary_radius_px))
    move = (fn | fp) & bnd
    add = fn & (~bnd)
    remove = fp & (~bnd)
    keep = valid & (~(move | add | remove))

    state = torch.zeros(ap.shape[0], ap.shape[1], ap.shape[-2], ap.shape[-1],
                        dtype=torch.long, device=ap.device)
    state[move[:, :, 0]] = 1
    state[add[:, :, 0]] = 2
    state[remove[:, :, 0]] = 3

    max_d = max(float(max_boundary_displacement_px), 1.0e-6)
    move_target = (target_sdf_patch.detach().to(ap) - anchor_sdf_patch.detach().to(ap)).clamp(-max_d, max_d)
    move_target = move_target * move.to(ap)

    m = min(max(float(target_margin), 1.0e-3), 0.49)
    hi = ap.new_tensor(0.5 + m)
    lo = ap.new_tensor(0.5 - m)
    z = torch.logit(ap)
    hi_z = torch.logit(hi)
    lo_z = torch.logit(lo)
    max_step = max(float(max_interior_logit_step), 1.0e-6)
    add_mag = (hi_z - z).clamp(min=0.0, max=max_step)
    remove_mag = (z - lo_z).clamp(min=0.0, max=max_step)
    mag_target = add.to(ap) * add_mag + remove.to(ap) * remove_mag
    return state, (keep, move, add, remove), move_target, mag_target


def _slr_ucdrt_state_loss(
    logits: torch.Tensor,
    state_target: torch.Tensor,
    valid_patch: torch.Tensor,
):
    """Macro-present-class CE so sparse residual states cannot be buried by KEEP."""
    if logits.ndim != 5 or logits.shape[2] != 4:
        raise ValueError(f"UCDRT state logits must be [B,K,4,R,R], got {tuple(logits.shape)}")
    b, k, _, r, _ = logits.shape
    l = logits.reshape(b*k, 4, r, r)
    t = state_target.reshape(b*k, r, r)
    v = valid_patch.reshape(b*k, 1, r, r).detach().to(logits)
    return _macro_present_class_ce(l, t, v, num_classes=4)


def _slr_ucdrt_action_loss(
    move_px: torch.Tensor,
    interior_magnitude: torch.Tensor,
    move_target: torch.Tensor,
    magnitude_target: torch.Tensor,
    masks,
):
    keep, move, add, remove = masks
    move_mask = move.to(move_px)
    int_mask = (add | remove).to(interior_magnitude)
    move_elem = F.smooth_l1_loss(move_px, move_target.to(move_px), reduction="none", beta=0.5)
    mag_elem = F.smooth_l1_loss(interior_magnitude, magnitude_target.to(interior_magnitude), reduction="none", beta=0.5)
    move_loss = (move_elem * move_mask).sum() / move_mask.sum().clamp_min(1.0)
    mag_loss = (mag_elem * int_mask).sum() / int_mask.sum().clamp_min(1.0)
    live_move = (move_mask.sum() > 0).to(move_px)
    live_int = (int_mask.sum() > 0).to(move_px)
    denom = (live_move + live_int).clamp_min(1.0)
    return (live_move * move_loss + live_int * mag_loss) / denom, move_loss, mag_loss


def _slr_patch_dice_per_candidate(
    prob: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Soft Dice [B,K], respecting exactly the sampled valid pixels."""
    p = prob.float()
    t = target.to(p).float()
    v = valid.to(p).float()
    inter = (p * t * v).sum(dim=(2,3,4))
    den = ((p + t) * v).sum(dim=(2,3,4))
    return (2.0 * inter + EPS) / (den + EPS)


def _slr_ucdrt_quality_per_candidate(
    prob: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Empty-aware local quality [B,K] used exclusively as VALUE target.

    Foreground-containing patches use soft Dice. Target-empty patches use
    1-mean foreground probability, so false-positive removal has a meaningful
    utility signal on BUSI/BTMRI empty-mask cases instead of an epsilon-sized
    Dice difference.
    """
    p = prob.float(); t = target.to(p).float(); v = valid.to(p).float()
    inter = (p*t*v).sum(dim=(2,3,4))
    den = ((p+t)*v).sum(dim=(2,3,4))
    dice = (2.0*inter + EPS)/(den + EPS)
    vcount = v.sum(dim=(2,3,4)).clamp_min(1.0)
    empty_quality = (1.0 - (p*v).sum(dim=(2,3,4))/vcount).clamp(0.0,1.0)
    target_mass = (t*v).sum(dim=(2,3,4))
    return torch.where(target_mass > 0.5, dice, empty_quality)


def _slr_ucdrt_masked_deploy_loss(
    logits: torch.Tensor, prob: torch.Tensor, target4: torch.Tensor, support4: torch.Tensor,
    ce_weight: float, dice_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Residual-focused BCE + empty-aware Dice surrogate for UCDRT deployment."""
    mask = support4.to(prob)
    bce_map = F.binary_cross_entropy_with_logits(logits.float(), target4.float(), reduction="none").to(prob)
    bce = (bce_map*mask).sum()/mask.sum().clamp_min(1.0)
    inter=(prob*target4*mask).sum(dim=(1,2,3))
    den=((prob+target4)*mask).sum(dim=(1,2,3))
    dice_loss=1.0-(2.0*inter+EPS)/(den+EPS)
    support_count=mask.sum(dim=(1,2,3))
    target_mass=(target4*mask).sum(dim=(1,2,3))
    empty_loss=(prob*mask).sum(dim=(1,2,3))/support_count.clamp_min(1.0)
    structural=torch.where(target_mass>0.5,dice_loss,empty_loss)
    live=support_count>0
    structural=(structural*live.to(structural)).sum()/live.float().sum().clamp_min(1.0)
    return ce_weight*bce+dice_weight*structural,bce,structural


def _slr_ucdrt_state_metrics(
    logits: torch.Tensor,
    state_target: torch.Tensor,
    valid_patch: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    pred = logits.detach().argmax(dim=2)
    valid = valid_patch.detach()[:, :, 0] > 0.5
    names = ("keep", "move", "add", "remove")
    out: Dict[str, torch.Tensor] = {}
    f1s = []
    for cid, name in enumerate(names):
        gt_c = valid & (state_target == cid)
        pr_c = valid & (pred == cid)
        tp = (gt_c & pr_c).float().sum()
        fp = ((~gt_c) & pr_c & valid).float().sum()
        fn = (gt_c & (~pr_c)).float().sum()
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
        out[f"{name}_precision"] = precision
        out[f"{name}_recall"] = recall
        out[f"{name}_f1"] = f1
        out[f"{name}_count"] = gt_c.float().sum()
        if bool(gt_c.any()):
            f1s.append(f1)
    out["macro_f1"] = torch.stack(f1s).mean() if f1s else logits.new_zeros(())
    out["pred_edit_fraction"] = _safe_ratio(((pred != 0) & valid).float().sum(), valid.float().sum())
    return out


def _slr_ucdrt_r2_targets(
    anchor_patch: torch.Tensor,
    local_target: torch.Tensor,
    valid_patch: torch.Tensor,
    boundary_patch: torch.Tensor,
    move_dictionary_probs: torch.Tensor,
    *,
    target_margin: float = 0.05,
    move_improvement_eps: float = 1.0e-4,
):
    """Operator-consistent EDIT/TYPE/MOVE/dose targets for UCDRT-R2.

    MOVE is assigned only when one of the *actual deployed warp dictionary*
    candidates reduces per-pixel BCE relative to KEEP.  Boundary errors that are
    not executable/helpful as MOVE fall back to signed ADD/REMOVE, which remain
    finite and crossable in probability space.
    """
    ap = anchor_patch.detach().clamp(EPS, 1.0 - EPS)
    tp = local_target.detach().to(ap).clamp(0.0, 1.0)
    valid = valid_patch.detach().to(ap) > 0.5
    bnd = (boundary_patch.detach().to(ap) > 0.5) & valid
    ah, gt = ap >= 0.5, tp >= 0.5
    fn = valid & (~ah) & gt
    fp = valid & ah & (~gt)
    err = fn | fp

    md = move_dictionary_probs.detach().to(ap).clamp(EPS, 1.0 - EPS)
    if md.ndim != 6 or md.shape[:2] != ap.shape[:2] or md.shape[3:] != ap.shape[2:]:
        raise ValueError(
            f"R2 move dictionary must be [B,K,D,1,R,R], got {tuple(md.shape)}"
        )
    td = tp[:, :, None].expand(-1, -1, md.shape[2], -1, -1, -1)
    move_bce = F.binary_cross_entropy(md, td, reduction="none")[:, :, :, 0]
    best_bce, best_idx = move_bce.min(dim=2)
    keep_bce = F.binary_cross_entropy(ap, tp, reduction="none")[:, :, 0]
    move_gain = keep_bce - best_bce
    move = err[:, :, 0] & bnd[:, :, 0] & (move_gain > float(move_improvement_eps))
    add = fn[:, :, 0] & (~move)
    remove = fp[:, :, 0] & (~move)
    edit = err[:, :, 0]

    type_target = torch.zeros_like(edit, dtype=torch.long)
    type_target[add] = 1
    type_target[remove] = 2
    # MOVE remains class 0 by construction.

    m = min(max(float(target_margin), 1.0e-3), 0.49)
    hi, lo = ap.new_tensor(0.5 + m), ap.new_tensor(0.5 - m)
    add_dose_target = ((hi - ap) / (1.0 - ap).clamp_min(EPS)).clamp(0.0, 1.0)
    remove_dose_target = ((ap - lo) / ap.clamp_min(EPS)).clamp(0.0, 1.0)
    add_dose_target = add_dose_target * add[:, :, None].to(ap)
    remove_dose_target = remove_dose_target * remove[:, :, None].to(ap)
    return {
        "edit": edit,
        "type": type_target,
        "move_bin": best_idx,
        "move_gain": move_gain,
        "move": move,
        "add": add,
        "remove": remove,
        "fn": fn[:, :, 0],
        "fp": fp[:, :, 0],
        "add_dose": add_dose_target,
        "remove_dose": remove_dose_target,
    }


def _slr_ucdrt_r2_focal_edit_loss(
    edit_logits: torch.Tensor,
    edit_target: torch.Tensor,
    valid_patch: torch.Tensor,
    gamma: float = 1.5,
):
    target = edit_target[:, :, None].to(edit_logits)
    valid = valid_patch.detach().to(edit_logits)
    bce = F.binary_cross_entropy_with_logits(edit_logits, target, reduction="none")
    p = torch.sigmoid(edit_logits)
    pt = target * p + (1.0 - target) * (1.0 - p)
    focal = (1.0 - pt).clamp_min(0.0).pow(float(gamma)) * bce
    return (focal * valid).sum() / valid.sum().clamp_min(1.0)


def _slr_ucdrt_r2_type_loss(
    type_logits: torch.Tensor,
    type_target: torch.Tensor,
    edit_target: torch.Tensor,
    valid_patch: torch.Tensor,
):
    """Conditional MOVE/ADD/REMOVE classification; balancing cannot alter EDIT prior."""
    b, k, c, r, _ = type_logits.shape
    l = type_logits.reshape(b * k, c, r, r)
    t = type_target.reshape(b * k, r, r)
    support = (
        edit_target[:, :, None].to(type_logits)
        * valid_patch.detach().to(type_logits)
    ).reshape(b * k, 1, r, r)
    return _macro_present_class_ce(l, t, support, num_classes=3)


def _slr_ucdrt_r2_parameter_loss(
    move_bin_logits: torch.Tensor,
    add_dose: torch.Tensor,
    remove_dose: torch.Tensor,
    targets: Dict[str, torch.Tensor],
):
    b, k, d, r, _ = move_bin_logits.shape
    move_support = targets["move"]
    ce = F.cross_entropy(
        move_bin_logits.reshape(b * k, d, r, r),
        targets["move_bin"].reshape(b * k, r, r),
        reduction="none",
    ).view(b, k, r, r)
    move_loss = (ce * move_support.to(ce)).sum() / move_support.float().sum().clamp_min(1.0)
    add_mask = targets["add"][:, :, None].to(add_dose)
    rem_mask = targets["remove"][:, :, None].to(remove_dose)
    add_elem = F.smooth_l1_loss(add_dose, targets["add_dose"].to(add_dose), reduction="none", beta=0.05)
    rem_elem = F.smooth_l1_loss(remove_dose, targets["remove_dose"].to(remove_dose), reduction="none", beta=0.05)
    add_loss = (add_elem * add_mask).sum() / add_mask.sum().clamp_min(1.0)
    rem_loss = (rem_elem * rem_mask).sum() / rem_mask.sum().clamp_min(1.0)
    live_m = (move_support.sum() > 0).to(add_dose)
    live_a = (add_mask.sum() > 0).to(add_dose)
    live_r = (rem_mask.sum() > 0).to(add_dose)
    denom = (live_m + live_a + live_r).clamp_min(1.0)
    total = (live_m * move_loss + live_a * add_loss + live_r * rem_loss) / denom
    return total, move_loss, add_loss, rem_loss


def _slr_ucdrt_r2_actor_exec_loss(
    hard_prob: torch.Tensor,
    soft_prob: torch.Tensor,
    anchor_prob: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    edit_prob: torch.Tensor,
):
    """Ungated executable actor outcome loss.

    Factual residual pixels always receive unit mass; predicted-edit pixels are
    additionally penalized in proportion to detached edit probability.  Thus the
    actor cannot win by preserving all correct context, and false edits remain
    explicitly costly.
    """
    a = anchor_prob.detach().clamp(EPS, 1.0 - EPS)
    t = target.detach().to(a)
    v = valid.detach().to(a)
    err = ((a >= 0.5) != (t >= 0.5)).to(a)
    support = torch.maximum(err, edit_prob.detach().to(a)) * v
    support_count = support.sum().clamp_min(1.0)

    def one(q):
        q = q.clamp(EPS, 1.0 - EPS)
        bce = F.binary_cross_entropy(q, t, reduction="none")
        bce = (bce * support).sum() / support_count
        inter = (q * t * support).sum(dim=(2, 3, 4))
        den = ((q + t) * support).sum(dim=(2, 3, 4))
        dice_loss = 1.0 - (2.0 * inter + EPS) / (den + EPS)
        sm = support.sum(dim=(2, 3, 4))
        tm = (t * support).sum(dim=(2, 3, 4))
        empty = (q * support).sum(dim=(2, 3, 4)) / sm.clamp_min(1.0)
        structural = torch.where(tm > 0.5, dice_loss, empty)
        live = sm > 0
        structural = (structural * live.to(structural)).sum() / live.float().sum().clamp_min(1.0)
        return 0.5 * bce + 0.5 * structural

    hard_loss = one(hard_prob)
    soft_loss = one(soft_prob)
    return 0.7 * hard_loss + 0.3 * soft_loss, hard_loss, soft_loss


def _slr_ucdrt_r2_full_quality(prob: torch.Tensor, target4: torch.Tensor, boundary_weight: float = 0.25):
    """Per-case full-map quality used only for detached critic targets/diagnostics."""
    p = prob.float().clamp(0.0, 1.0)
    t = target4.to(p).float()
    inter = (p * t).sum(dim=(-3, -2, -1))
    den = (p + t).sum(dim=(-3, -2, -1))
    dice = (2.0 * inter + EPS) / (den + EPS)
    empty = 1.0 - p.mean(dim=(-3, -2, -1))
    target_mass = t.sum(dim=(-3, -2, -1))
    region_q = torch.where(target_mass > 0.5, dice, empty)
    bw = min(max(float(boundary_weight), 0.0), 0.75)
    if bw <= 0:
        return region_q
    pd = F.max_pool2d(p, 3, stride=1, padding=1)
    pe = 1.0 - F.max_pool2d(1.0 - p, 3, stride=1, padding=1)
    td = F.max_pool2d(t, 3, stride=1, padding=1)
    te = 1.0 - F.max_pool2d(1.0 - t, 3, stride=1, padding=1)
    pb = (pd - pe).clamp(0.0, 1.0)
    tb = (td - te).clamp(0.0, 1.0)
    bi = (pb * tb).sum(dim=(-3, -2, -1))
    bd = (pb + tb).sum(dim=(-3, -2, -1))
    bq = (2.0 * bi + EPS) / (bd + EPS)
    bq = torch.where(target_mass > 0.5, bq, region_q)
    return (1.0 - bw) * region_q + bw * bq


def _slr_ucdrt_r2_critic_targets(
    anchor_prob: torch.Tensor,
    target4: torch.Tensor,
    candidate_full_num: torch.Tensor,
    candidate_full_den: torch.Tensor,
    step_selected_index: torch.Tensor,
    step_available: torch.Tensor,
    *,
    boundary_weight: float,
    gain_scale: float,
):
    """Exact marginal WOLA advantage target for every recorded critic step."""
    with torch.no_grad():
        a = anchor_prob.detach().clamp(EPS, 1.0 - EPS)
        t = target4.detach().to(a)
        num_c = candidate_full_num.detach().to(a)
        den_c = candidate_full_den.detach().to(a)
        b, k = num_c.shape[:2]
        steps = step_selected_index.shape[1]
        base_logits = torch.logit(a)
        accum_num = torch.zeros_like(a)
        accum_den = torch.zeros_like(a)
        targets = []
        for si in range(steps):
            cur_delta = torch.where(
                accum_den > 1.0e-8,
                accum_num / accum_den.clamp_min(1.0e-8),
                torch.zeros_like(accum_num),
            )
            cur_prob = torch.sigmoid(base_logits + cur_delta)
            cur_q = _slr_ucdrt_r2_full_quality(cur_prob, t, boundary_weight)
            hn = accum_num[:, None] + num_c
            hd = accum_den[:, None] + den_c
            hdlt = torch.where(hd > 1.0e-8, hn / hd.clamp_min(1.0e-8), torch.zeros_like(hn))
            hp = torch.sigmoid(base_logits[:, None] + hdlt)
            ht = t[:, None].expand(-1, k, -1, -1, -1).reshape(b * k, 1, *t.shape[-2:])
            hq = _slr_ucdrt_r2_full_quality(
                hp.reshape(b * k, 1, *a.shape[-2:]), ht, boundary_weight
            ).view(b, k)
            gain = (hq - cur_q[:, None]) * float(gain_scale)
            avail = step_available[:, si].detach() > 0.5
            targets.append(torch.where(avail, gain.clamp(-1.0, 1.0), torch.zeros_like(gain)))
            idx = step_selected_index[:, si].detach().long()
            take = idx >= 0
            if bool(take.any()):
                oh = F.one_hot(idx.clamp_min(0), num_classes=k).to(a) * take[:, None].to(a)
                accum_num = accum_num + (num_c * oh[:, :, None, None, None]).sum(dim=1)
                accum_den = accum_den + (den_c * oh[:, :, None, None, None]).sum(dim=1)
        return torch.stack(targets, dim=1)




def _slr_ucdrt_r2_greedy_oracle_set_gain(
    anchor_prob: torch.Tensor,
    target4: torch.Tensor,
    candidate_full_num: torch.Tensor,
    candidate_full_den: torch.Tensor,
    *,
    boundary_weight: float,
):
    """Greedy exact marginal oracle over the same WOLA candidate set (K<=8)."""
    with torch.no_grad():
        a = anchor_prob.detach().clamp(EPS, 1.0 - EPS)
        t = target4.detach().to(a)
        num_c = candidate_full_num.detach().to(a)
        den_c = candidate_full_den.detach().to(a)
        b, k = num_c.shape[:2]
        base_logits = torch.logit(a)
        accum_num = torch.zeros_like(a)
        accum_den = torch.zeros_like(a)
        used = torch.zeros((b, k), dtype=torch.bool, device=a.device)
        q0 = _slr_ucdrt_r2_full_quality(a, t, boundary_weight)
        for _ in range(k):
            cur_delta = torch.where(accum_den > 1e-8, accum_num/accum_den.clamp_min(1e-8), torch.zeros_like(accum_num))
            cur_prob = torch.sigmoid(base_logits + cur_delta)
            cur_q = _slr_ucdrt_r2_full_quality(cur_prob, t, boundary_weight)
            hn = accum_num[:,None] + num_c
            hd = accum_den[:,None] + den_c
            dlt = torch.where(hd > 1e-8, hn/hd.clamp_min(1e-8), torch.zeros_like(hn))
            hp = torch.sigmoid(base_logits[:,None] + dlt)
            ht = t[:,None].expand(-1,k,-1,-1,-1).reshape(b*k,1,*t.shape[-2:])
            hq = _slr_ucdrt_r2_full_quality(hp.reshape(b*k,1,*a.shape[-2:]), ht, boundary_weight).view(b,k)
            gain = (hq-cur_q[:,None]).masked_fill(used, -1e4)
            best, idx = gain.max(dim=1)
            take = best > 0
            if not bool(take.any()):
                break
            oh = F.one_hot(idx, num_classes=k).to(a) * take[:,None].to(a)
            used = used | (oh > .5)
            accum_num = accum_num + (num_c*oh[:,:,None,None,None]).sum(dim=1)
            accum_den = accum_den + (den_c*oh[:,:,None,None,None]).sum(dim=1)
        final_delta = torch.where(accum_den > 1e-8, accum_num/accum_den.clamp_min(1e-8), torch.zeros_like(accum_num))
        final_prob = torch.sigmoid(base_logits + final_delta)
        qf = _slr_ucdrt_r2_full_quality(final_prob, t, boundary_weight)
        return (qf-q0).mean()

def _slr_ucdrt_r2_critic_loss(
    step_logits: torch.Tensor,
    step_values: torch.Tensor,
    targets: torch.Tensor,
    available: torch.Tensor,
    *,
    rank_margin: float = 0.05,
):
    mask = available.detach().to(step_values)
    value_t = targets.detach().to(step_values)
    reg = F.smooth_l1_loss(step_values, value_t, reduction="none", beta=0.05)
    reg = (reg * mask).sum() / mask.sum().clamp_min(1.0)
    positive = (value_t > 0).to(step_logits)
    sign = F.binary_cross_entropy_with_logits(step_logits, positive, reduction="none")
    sign = (sign * mask).sum() / mask.sum().clamp_min(1.0)
    posm = (value_t > 0).to(step_values) * mask
    negm = (value_t < 0).to(step_values) * mask
    m = float(rank_margin)
    rank_pos = (F.relu(m - step_values) * posm).sum() / posm.sum().clamp_min(1.0)
    rank_neg = (F.relu(m + step_values) * negm).sum() / negm.sum().clamp_min(1.0)
    rank = 0.5 * (rank_pos + rank_neg)
    total = 0.4 * reg + 0.4 * sign + 0.2 * rank
    pred_pos = step_values.detach() > 0
    gt_pos = value_t > 0
    vm = mask > 0
    tp = (pred_pos & gt_pos & vm).float().sum()
    fp = (pred_pos & (~gt_pos) & vm).float().sum()
    fn = ((~pred_pos) & gt_pos & vm).float().sum()
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    mae = ((step_values.detach() - value_t).abs() * mask).sum() / mask.sum().clamp_min(1.0)
    vv = step_values.detach()[vm]
    tt = value_t[vm]
    corr = step_values.new_zeros(())
    if vv.numel() >= 2:
        v0, t0 = vv - vv.mean(), tt - tt.mean()
        corr = (v0 * t0).sum() / torch.sqrt((v0.square().sum() * t0.square().sum()).clamp_min(EPS))
    return total, {
        "reg": reg.detach(), "sign": sign.detach(), "rank": rank.detach(),
        "precision": precision.detach(), "recall": recall.detach(),
        "mae": mae.detach(), "corr": corr.detach(),
        "positive_rate": _safe_ratio((gt_pos & vm).float().sum(), vm.float().sum()).detach(),
    }


def _slr_ucdrt_r2_factor_metrics(
    edit_logits: torch.Tensor,
    type_logits: torch.Tensor,
    targets: Dict[str, torch.Tensor],
    valid_patch: torch.Tensor,
):
    valid = valid_patch.detach()[:, :, 0] > 0.5
    gt_edit = targets["edit"] & valid
    pred_edit = (torch.sigmoid(edit_logits.detach())[:, :, 0] >= 0.5) & valid
    tp = (pred_edit & gt_edit).float().sum()
    fp = (pred_edit & (~gt_edit) & valid).float().sum()
    fn = ((~pred_edit) & gt_edit).float().sum()
    ep = _safe_ratio(tp, tp + fp); er = _safe_ratio(tp, tp + fn)
    ef1 = _safe_ratio(2 * ep * er, ep + er)
    pred_type = type_logits.detach().argmax(dim=2)
    names = ("move", "add", "remove")
    out = {
        "edit_precision": ep, "edit_recall": er, "edit_f1": ef1,
        "target_edit_fraction": _safe_ratio(gt_edit.float().sum(), valid.float().sum()),
        "pred_edit_fraction": _safe_ratio(pred_edit.float().sum(), valid.float().sum()),
    }
    f1s = []
    for cid, name in enumerate(names):
        gm = gt_edit & (targets["type"] == cid)
        pm = pred_edit & (pred_type == cid)
        tpc = (gm & pm).float().sum(); fpc = ((~gm) & pm & valid).float().sum(); fnc = (gm & (~pm)).float().sum()
        pc = _safe_ratio(tpc, tpc + fpc); rc = _safe_ratio(tpc, tpc + fnc); fc = _safe_ratio(2 * pc * rc, pc + rc)
        out[name + "_f1"] = fc
        if bool(gm.any()): f1s.append(fc)
    out["type_macro_f1"] = torch.stack(f1s).mean() if f1s else edit_logits.new_zeros(())
    return out


def _compose_v4e(
    anchor: torch.Tensor,
    support_fn: torch.Tensor,
    support_fp: torch.Tensor,
    add_logit_magnitude: torch.Tensor,
    remove_logit_magnitude: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Operator-consistent V4E: support selects action; magnitude is full dose."""
    z = torch.logit(anchor.clamp(EPS, 1.0 - EPS))
    logits = z + support_fn * add_logit_magnitude - support_fp * remove_logit_magnitude
    return torch.sigmoid(logits).clamp(EPS, 1.0 - EPS), logits


def _v4e_targets(
    anchor: torch.Tensor,
    target: torch.Tensor,
    fn: torch.Tensor,
    fp: torch.Tensor,
    margin: float,
    max_logit_step: float,
):
    """Single-support/full-dose targets used by loss, teacher and deployment audit.

    Only hard FN/FP pixels own a non-zero correction target.  The magnitude is
    the complete monotonic logit step required to reach 0.5+margin / 0.5-margin.
    Off-support target is exactly zero, removing V4D's undefined-magnitude region.
    """
    m = float(min(max(margin, 1.0e-3), 0.49))
    hi = anchor.new_tensor(0.5 + m)
    lo = anchor.new_tensor(0.5 - m)
    z = torch.logit(anchor.detach().clamp(EPS, 1.0 - EPS))
    hi_z = torch.logit(hi)
    lo_z = torch.logit(lo)
    max_step = float(max(max_logit_step, 1.0e-3))
    fn4 = fn[:, None].to(anchor)
    fp4 = fp[:, None].to(anchor)
    raw_add = (hi_z - z).clamp_min(0.0)
    raw_remove = (z - lo_z).clamp_min(0.0)
    add_target = fn4 * raw_add.clamp_max(max_step)
    remove_target = fp4 * raw_remove.clamp_max(max_step)
    add_capped = fn4 * (raw_add >= max_step).to(anchor)
    remove_capped = fp4 * (raw_remove >= max_step).to(anchor)
    return add_target, remove_target, add_capped, remove_capped


def _weighted_support_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    support: torch.Tensor,
    off_support_weight: float,
) -> torch.Tensor:
    """Directly supervise full dose on support and zero dose off support."""
    elem = F.smooth_l1_loss(pred, target, reduction="none")
    sup4 = support[:, None].to(pred)
    w0 = float(min(max(off_support_weight, 0.0), 1.0))
    weight = sup4 + (1.0 - sup4) * w0
    return (elem * weight).sum() / weight.sum().clamp_min(EPS)


def _v4e_oracle_diagnostics(
    anchor: torch.Tensor,
    target: torch.Tensor,
    pred_fn: torch.Tensor,
    pred_fp: torch.Tensor,
    add_magnitude: torch.Tensor,
    remove_magnitude: torch.Tensor,
    add_target: torch.Tensor,
    remove_target: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Factorial V4E decomposition with one operator for every branch."""
    _, _, fn, fp = _typed_targets(anchor, target)
    fn4 = fn[:, None].to(anchor)
    fp4 = fp[:, None].to(anchor)
    pfn4 = pred_fn[:, None].to(anchor)
    pfp4 = pred_fp[:, None].to(anchor)
    base_d = _dice_per_case(anchor[:, 0], target)
    pred_p, _ = _compose_v4e(anchor, pfn4, pfp4, add_magnitude, remove_magnitude)
    gt_support_p, _ = _compose_v4e(anchor, fn4, fp4, add_magnitude, remove_magnitude)
    target_mag_p, _ = _compose_v4e(anchor, pfn4, pfp4, add_target, remove_target)
    full_p, _ = _compose_v4e(anchor, fn4, fp4, add_target, remove_target)
    def pack(name, prob):
        d = _dice_per_case(prob[:, 0], target)
        return {f"{name}_dice": d.mean(), f"{name}_gain": (d-base_d).mean()}
    out = {}
    for name, prob in (("deployed", pred_p), ("oracle_support", gt_support_p),
                       ("oracle_magnitude", target_mag_p), ("oracle_full", full_p)):
        out.update(pack(name, prob))
    return out

def _v4d_targets(
    anchor: torch.Tensor,
    target: torch.Tensor,
    fn: torch.Tensor,
    fp: torch.Tensor,
    margin: float,
    max_logit_step: float,
):
    """Continuous severity + minimum sufficient logit-step targets.

    Severity is a continuous margin-based *need-to-edit* target split into two
    non-negative channels.  It remains smooth around the 0.5 decision boundary
    but becomes zero once a pixel is confidently correct beyond the symmetric
    margin:

        S*_FN = Y * relu((0.5+m)-P)/(0.5+m)
        S*_FP = (1-Y) * relu(P-(0.5-m))/(0.5+m)

    Magnitude targets are only supervised on actual hard FN/FP pixels and ask
    for the minimum monotonic logit step that crosses a symmetric confidence
    margin (0.5+margin for FN, 0.5-margin for FP).
    """
    target4 = target[:, None].to(anchor)
    m = float(min(max(margin, 1.0e-3), 0.49))
    hi = anchor.new_tensor(0.5 + m)
    lo = anchor.new_tensor(0.5 - m)

    # Continuous *need-to-edit* severity around the decision boundary.  Unlike
    # a hard FN/FP mask, this does not jump at 0.5; unlike the raw Y-P residual,
    # it becomes exactly zero once a pixel is confidently correct beyond the
    # pre-declared symmetric margin.
    p = anchor.detach()
    severity_fn_target = target4 * ((hi - p) / hi.clamp_min(EPS)).clamp(0.0, 1.0)
    severity_fp_target = (1.0 - target4) * ((p - lo) / (1.0 - lo).clamp_min(EPS)).clamp(0.0, 1.0)

    z = torch.logit(anchor.detach().clamp(EPS, 1.0 - EPS))
    hi_z = torch.logit(hi)
    lo_z = torch.logit(lo)
    max_step = float(max(max_logit_step, 1.0e-3))
    add_target = (hi_z - z).clamp(min=0.0, max=max_step)
    remove_target = (z - lo_z).clamp(min=0.0, max=max_step)
    return severity_fn_target, severity_fp_target, add_target, remove_target


def _v4d_oracle_diagnostics(
    anchor: torch.Tensor,
    target: torch.Tensor,
    severity_fn: torch.Tensor,
    severity_fp: torch.Tensor,
    add_magnitude: torch.Tensor,
    remove_magnitude: torch.Tensor,
    add_target: torch.Tensor,
    remove_target: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """GT-only V4D support/type/magnitude decomposition.

    These tensors are diagnostics only and never enter the objective.
    """
    _, correct, fn, fp = _typed_targets(anchor, target)
    fn4 = fn[:, None].to(anchor)
    fp4 = fp[:, None].to(anchor)
    err4 = (fn | fp)[:, None].to(anchor)
    correct4 = correct[:, None].to(anchor)
    s_total = (severity_fn + severity_fp).clamp_min(EPS)
    cond_fn = severity_fn / s_total
    cond_fp = severity_fp / s_total

    # Exact WHERE support, learned conditional type and learned magnitude.
    p_where, _ = _compose_v4d(anchor, err4 * cond_fn, err4 * cond_fp, add_magnitude, remove_magnitude)
    # Exact sign on true errors while retaining false support on correct pixels.
    sign_fn = correct4 * severity_fn + fn4 * s_total
    sign_fp = correct4 * severity_fp + fp4 * s_total
    p_sign, _ = _compose_v4d(anchor, sign_fn, sign_fp, add_magnitude, remove_magnitude)
    # Exact typed support, learned magnitude.
    p_typed, _ = _compose_v4d(anchor, fn4, fp4, add_magnitude, remove_magnitude)
    # Exact typed support + minimum-sufficient direct magnitude target.  This is
    # a much tighter diagnostic than V4C's maximal probability overwrite.
    p_full, _ = _compose_v4d(anchor, fn4, fp4, add_target, remove_target)

    anchor_dice = _dice_per_case(anchor[:, 0], target)
    return {
        "oracle_where_dice": _dice_per_case(p_where[:, 0], target).mean(),
        "oracle_sign_dice": _dice_per_case(p_sign[:, 0], target).mean(),
        "oracle_typed_dice": _dice_per_case(p_typed[:, 0], target).mean(),
        "oracle_full_dice": _dice_per_case(p_full[:, 0], target).mean(),
        "oracle_where_gain": (_dice_per_case(p_where[:, 0], target) - anchor_dice).mean(),
        "oracle_sign_gain": (_dice_per_case(p_sign[:, 0], target) - anchor_dice).mean(),
        "oracle_typed_gain": (_dice_per_case(p_typed[:, 0], target) - anchor_dice).mean(),
        "oracle_full_gain": (_dice_per_case(p_full[:, 0], target) - anchor_dice).mean(),
    }


def _edit_stats(final_prob: torch.Tensor, anchor: torch.Tensor, err: torch.Tensor, correct: torch.Tensor):
    edit = (final_prob[:, 0] - anchor[:, 0]).abs()
    errf = err.to(edit)
    correctf = correct.to(edit)
    total_mass = edit.sum().clamp_min(EPS)
    err_mass = (edit * errf).sum()
    edit_precision = err_mass / total_mass
    edit_error_mean = _masked_mean(edit, err)
    edit_correct_mean = _masked_mean(edit, correct)
    edit_ratio = edit_error_mean / edit_correct_mean.clamp_min(EPS)
    return edit_precision, edit_error_mean, edit_correct_mean, edit_ratio


def _oracle_diagnostics(
    anchor: torch.Tensor,
    target: torch.Tensor,
    q_fn: torch.Tensor,
    q_fp: torch.Tensor,
    add_magnitude: torch.Tensor,
    remove_magnitude: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """GT-only decomposition of WHERE, SIGN and MAGNITUDE bottlenecks."""
    cls, correct, fn, fp = _typed_targets(anchor, target)
    del cls
    fn4 = fn[:, None].to(anchor)
    fp4 = fp[:, None].to(anchor)
    err4 = (fn | fp)[:, None].to(anchor)
    correct4 = correct[:, None].to(anchor)
    q_error = (q_fn + q_fp).clamp_min(EPS)

    # WHERE oracle: exact error support, but keep predicted conditional FN/FP
    # proportions.  This isolates false-positive/false-negative support errors.
    cond_fn = q_fn / q_error
    cond_fp = q_fp / q_error
    where_fn = err4 * cond_fn
    where_fp = err4 * cond_fp
    p_where = _compose_typed(anchor, where_fn, where_fp, add_magnitude, remove_magnitude)

    # SIGN oracle: on true error pixels, keep the predicted total support but
    # route it to the GT direction. On correct pixels retain current predictions,
    # so false WHERE activations are deliberately not fixed here.
    sign_fn = correct4 * q_fn + fn4 * q_error
    sign_fp = correct4 * q_fp + fp4 * q_error
    p_sign = _compose_typed(anchor, sign_fn, sign_fp, add_magnitude, remove_magnitude)

    # WHERE+SIGN oracle: exact typed support, learned magnitude retained.
    p_typed = _compose_typed(anchor, fn4, fp4, add_magnitude, remove_magnitude)

    # Full operator upper bound: exact typed support and maximal bounded edit.
    ones = torch.ones_like(add_magnitude)
    p_full = _compose_typed(anchor, fn4, fp4, ones, ones)

    anchor_dice = _dice_per_case(anchor[:, 0], target)
    return {
        "oracle_where_dice": _dice_per_case(p_where[:, 0], target).mean(),
        "oracle_sign_dice": _dice_per_case(p_sign[:, 0], target).mean(),
        "oracle_typed_dice": _dice_per_case(p_typed[:, 0], target).mean(),
        "oracle_full_dice": _dice_per_case(p_full[:, 0], target).mean(),
        "oracle_where_gain": (_dice_per_case(p_where[:, 0], target) - anchor_dice).mean(),
        "oracle_sign_gain": (_dice_per_case(p_sign[:, 0], target) - anchor_dice).mean(),
        "oracle_typed_gain": (_dice_per_case(p_typed[:, 0], target) - anchor_dice).mean(),
        "oracle_full_gain": (_dice_per_case(p_full[:, 0], target) - anchor_dice).mean(),
    }



def _masked_bce_with_logits(
    logits: torch.Tensor,
    target4: torch.Tensor,
    mask4: torch.Tensor,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits.float(), target4.float(), reduction="none").to(logits)
    mask = mask4.to(loss)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)




def _masked_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask4: torch.Tensor,
    *,
    beta: float = 0.1,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred.float(), target.float(), reduction="none", beta=float(max(beta, 1.0e-6))).to(pred)
    mask = mask4.to(loss)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)

def _selection_stats(mask4: torch.Tensor, anchor: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    sel = mask4[:, 0].detach() > 0.5
    pred = anchor[:, 0].detach() >= 0.5
    gt = target.detach() >= 0.5
    err = pred != gt
    tp = (sel & err).float().sum()
    fp = (sel & (~err)).float().sum()
    fn = ((~sel) & err).float().sum()
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "coverage": sel.float().mean(),
        "error_rate": err.float().mean(),
    }


def _point_accuracy(prob: torch.Tensor, target: torch.Tensor, mask4: torch.Tensor) -> torch.Tensor:
    sel = mask4[:, 0].detach() > 0.5
    if int(sel.sum().item()) == 0:
        return prob.new_zeros(())
    pred = prob[:, 0].detach() >= 0.5
    gt = target.detach() >= 0.5
    return (pred[sel] == gt[sel]).float().mean()


def _average_precision_binary(score: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Global batch AP for diagnostics only; no gradient or model selection use."""
    s = score.detach().float().flatten()
    y = target.detach().bool().flatten()
    positives = int(y.sum().item())
    if positives <= 0:
        return score.new_zeros(())
    order = torch.argsort(s, descending=True)
    ys = y[order].float()
    tp = torch.cumsum(ys, dim=0)
    rank = torch.arange(1, ys.numel() + 1, device=ys.device, dtype=tp.dtype)
    precision = tp / rank
    return (precision * ys).sum() / float(positives)


def _topk_recall(score: torch.Tensor, target: torch.Tensor, coverage: float) -> torch.Tensor:
    if score.ndim == 4:
        score = score[:, 0]
    b, h, w = score.shape
    n = h * w
    k = max(1, min(n, int(math.ceil(float(coverage) * n)))) if 'math' in globals() else max(1, min(n, int((float(coverage) * n) + 0.999999)))
    vals = []
    for bi in range(b):
        gt = target[bi].detach().bool().flatten()
        pos = gt.float().sum()
        if float(pos) <= 0:
            vals.append(score.new_zeros(()))
            continue
        idx = score[bi].detach().flatten().topk(k, largest=True, sorted=False).indices
        hit = gt[idx].float().sum()
        vals.append(hit / pos.clamp_min(1.0))
    return torch.stack(vals).mean()



def _masked_region_seg_loss(
    logits: torch.Tensor,
    prob: torch.Tensor,
    target4: torch.Tensor,
    region4: torch.Tensor,
    ce_weight: float,
    dice_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standard BCE+Dice restricted to a pre-declared inspection region."""
    mask = region4.to(prob)
    bce_map = F.binary_cross_entropy_with_logits(
        logits.float(), target4.float(), reduction="none"
    ).to(prob)
    bce = (bce_map * mask).sum() / mask.sum().clamp_min(1.0)
    inter = (prob * target4 * mask).sum(dim=(1, 2, 3))
    den = ((prob + target4) * mask).sum(dim=(1, 2, 3))
    dice = (1.0 - (2.0 * inter + EPS) / (den + EPS)).mean()
    return ce_weight * bce + dice_weight * dice, bce, dice


def _slr_patch_seg_loss(
    logits: torch.Tensor,
    probs: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    ce_weight: float,
    dice_weight: float,
    support: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standard BCE+Dice on BxK local patches.

    ``support`` is a deterministic factual structure/error mask.  It is used
    only to focus the direct local segmentation task; Final supervision still
    uses the exact deployed union.
    """
    if logits.ndim != 5:
        raise ValueError(f"SLR patch logits must be [B,K,1,R,R], got {tuple(logits.shape)}")
    b, k, c, r, _ = logits.shape
    lg = logits.reshape(b * k, c, r, r)
    pr = probs.reshape(b * k, c, r, r)
    tg = target.to(pr).reshape(b * k, c, r, r)
    vm = valid.to(pr).reshape(b * k, c, r, r)
    if support is not None:
        sm = support.to(pr).reshape(b * k, c, r, r)
        vm = vm * sm
    bce_map = F.binary_cross_entropy_with_logits(lg.float(), tg.float(), reduction="none").to(pr)
    bce = (bce_map * vm).sum() / vm.sum().clamp_min(1.0)
    inter = (pr * tg * vm).sum(dim=(1,2,3))
    den = ((pr + tg) * vm).sum(dim=(1,2,3))
    dice = (1.0 - (2.0 * inter + EPS) / (den + EPS)).mean()
    return ce_weight * bce + dice_weight * dice, bce, dice


def _slr_typed_actor_loss(
    state_logits: torch.Tensor,
    dose: torch.Tensor,
    signed_action: torch.Tensor,
    state_target: torch.Tensor,
    dose_target: torch.Tensor,
    valid: torch.Tensor,
    support: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Operator-consistent KEEP/ADD/REMOVE + minimum-dose objective.

    Type CE is macro-averaged over classes present in the current batch instead
    of using a BUSI-specific fixed class weight. Dose is supervised only on true
    edits, while correct pixels explicitly teach the actor exact KEEP identity.
    """
    if state_logits.ndim != 5 or state_logits.shape[2] != 3:
        raise ValueError(
            f"SLR actor state logits must be [B,K,3,R,R], got {tuple(state_logits.shape)}"
        )
    b, k, _, r, _ = state_logits.shape
    lg = state_logits.reshape(b * k, 3, r, r)
    tg = state_target.long().reshape(b * k, r, r)
    vm = (valid.to(dose) * support.to(dose)).reshape(b * k, 1, r, r)[:, 0]
    ce_map = F.cross_entropy(lg.float(), tg, reduction="none").to(dose)
    class_terms = []
    for cls in range(3):
        cm = vm * (tg == cls).to(vm)
        if bool((cm.sum() > 0).detach()):
            class_terms.append((ce_map * cm).sum() / cm.sum().clamp_min(1.0))
    type_loss = torch.stack(class_terms).mean() if class_terms else ce_map.sum() * 0.0

    dose4 = dose.reshape(b * k, 1, r, r)[:, 0]
    dt4 = dose_target.to(dose).reshape(b * k, 1, r, r)[:, 0]
    signed4 = signed_action.reshape(b * k, 1, r, r)[:, 0]
    edit = vm * (tg != 0).to(vm)
    # Supervise both the latent non-negative dose and the *executed* signed
    # action.  The latter is essential: before type confidence saturates,
    # (P_add-P_remove) attenuates the latent dose and can otherwise recreate the
    # same reachability failure that OCRA is intended to remove.
    sign_target = (tg == 1).to(dose) - (tg == 2).to(dose)
    signed_target = sign_target * dt4
    latent_dose_elem = F.smooth_l1_loss(
        dose4.float(), dt4.float(), reduction="none"
    ).to(dose)
    executed_dose_elem = F.smooth_l1_loss(
        signed4.float(), signed_target.float(), reduction="none"
    ).to(dose)
    dose_elem = 0.5 * latent_dose_elem + 0.5 * executed_dose_elem
    dose_loss = (dose_elem * edit).sum() / edit.sum().clamp_min(1.0)

    keep = vm * (tg == 0).to(vm)
    keep_loss = (signed4.abs() * keep).sum() / keep.sum().clamp_min(1.0)
    total = 0.45 * type_loss + 0.35 * dose_loss + 0.20 * keep_loss
    return total, type_loss, dose_loss, keep_loss


def _slr_sdf_loss(
    pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, support: torch.Tensor, radius: float
) -> torch.Tensor:
    """Dimensionless full-valid Smooth-L1 on absolute SDF.

    ``support`` is retained in the signature for overlay compatibility but is
    intentionally not used: after normalization by R the far field no longer
    dominates numerically, and keeping every valid pixel gives the absolute-SDF
    auxiliary task a stationary target everywhere in the local patch.
    """
    del support
    radius = max(float(radius), 1.0)
    vm = valid.to(pred)
    loss = F.smooth_l1_loss(
        (pred.float() / radius), (target.to(pred).float() / radius), reduction="none"
    ).to(pred)
    return (loss * vm).sum() / vm.sum().clamp_min(1.0)


def _slr_truncated_signed_distance(mask4: torch.Tensor, radius: int) -> torch.Tensor:
    """Validation-side twin of the label-faithful pixel-centred SDF."""
    radius = max(int(radius), 1)
    m = mask4 > 0.5
    inf = float(radius) + 2.0

    def shift_inf(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
        h, w = x.shape[-2:]
        pad_l, pad_r = max(dx, 0), max(-dx, 0)
        pad_t, pad_b = max(dy, 0), max(-dy, 0)
        y = F.pad(x, (pad_l, pad_r, pad_t, pad_b), value=inf)
        return y[..., pad_b:pad_b+h, pad_r:pad_r+w]

    def distance_to(seed_mask: torch.Tensor) -> torch.Tensor:
        dist = torch.where(seed_mask, mask4.new_zeros(()), mask4.new_full((), inf))
        diag = 2.0 ** 0.5
        for _ in range(radius + 1):
            cand = [dist]
            for dy, dx, cost in (
                (-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0),
                (-1,-1,diag),(-1,1,diag),(1,-1,diag),(1,1,diag),
            ):
                cand.append(shift_inf(dist,dy,dx)+cost)
            dist = torch.stack(cand, dim=0).amin(dim=0)
        return dist

    d_to_bg = distance_to(~m)
    d_to_fg = distance_to(m)
    inside = (d_to_bg - 0.5).clamp(min=0.5, max=float(radius))
    outside = (d_to_fg - 0.5).clamp(min=0.5, max=float(radius))
    return torch.where(m, inside, -outside).to(mask4)

def _compute_c2r_loss(
    cfg,
    target: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    mode: str,
    ce_weight: float,
    dice_weight: float,
    epoch: int = 0,
):
    """Counterfactual-Consensus Regional Reconstruction objective.

    Three GT-free anchor views share one regional decoder.  Every view is trained
    against the *same* local GT segmentation; there is no residual/action label.
    The optional final loss supervises the actually committed output, whose edit
    mask is a detached deterministic unanimity decision.
    """
    m1 = _cfg_get(cfg, "M1", None)
    base_prob = aux["geotopo_base_probs"]
    geo_prob = aux["geotopo_geometry_probs"]
    geo_logits = aux["geotopo_geometry_logits"]
    final_prob = aux["geotopo_final_probs"]
    final_logits = aux["geotopo_final_logits"]
    flow = aux["geotopo_geometry_flow_px"]
    prefix = "geotopo_reconstruction_after_geometry" if mode == "full" else "geotopo_reconstruction_base"
    anchor = (geo_prob if mode == "full" else base_prob).detach()
    region = aux[prefix + "_c2r_region_mask"]
    centers = aux[prefix + "_c2r_center_mask"]
    view_logits = aux[prefix + "_c2r_view_logits"]
    view_probs = aux[prefix + "_c2r_view_probs"]
    mean_prob = aux[prefix + "_c2r_mean_prob"]
    consensus = aux[prefix + "_c2r_consensus_mask"]
    edit = aux[prefix + "_c2r_edit_mask"]
    margin = aux[prefix + "_v4g_margin_uncertainty"]
    mc_std = aux[prefix + "_v4g_mc_std_map"]
    mc_dis = aux[prefix + "_v4g_mc_disagreement_map"]
    entropy = aux[prefix + "_v4g_entropy_map"]
    candidate_mask = aux.get(prefix + "_c2r_candidate_mask", torch.zeros_like(anchor))
    commit_mask = aux.get(prefix + "_c2r_commit_mask", edit)
    roi_overlap = aux.get(prefix + "_c2r_roi_overlap_pixel_count", final_prob.new_zeros(()))
    roi_unique = aux.get(prefix + "_c2r_roi_unique_pixel_count", final_prob.new_zeros(()))
    center_min_dist = aux.get(prefix + "_c2r_center_min_chebyshev_distance", final_prob.new_zeros(()))
    candidate_components = aux.get(prefix + "_c2r_candidate_component_count", final_prob.new_zeros(()))
    committed_components = aux.get(prefix + "_c2r_committed_component_count", final_prob.new_zeros(()))
    candidate_component_area = aux.get(prefix + "_c2r_candidate_component_area_mean", final_prob.new_zeros(()))
    committed_component_area = aux.get(prefix + "_c2r_committed_component_area_mean", final_prob.new_zeros(()))
    component_agreement = aux.get(prefix + "_c2r_component_agreement_mean", final_prob.new_zeros(()))
    component_spread = aux.get(prefix + "_c2r_component_spread_q90_mean", final_prob.new_zeros(()))
    component_confidence = aux.get(prefix + "_c2r_component_confidence_q10_mean", final_prob.new_zeros(()))
    aefr_stability = aux.get(prefix + "_aefr_posterior_stability_support", final_prob.new_zeros(()))
    aefr_stability_improvement = aux.get(prefix + "_aefr_posterior_stability_improvement", final_prob.new_zeros(()))
    aefr_dis_pre = aux.get(prefix + "_aefr_posterior_disagreement_pre", final_prob.new_zeros(()))
    aefr_dis_post = aux.get(prefix + "_aefr_posterior_disagreement_post", final_prob.new_zeros(()))
    aefr_support_fraction = aux.get(prefix + "_aefr_action_support_fraction", final_prob.new_zeros(()))
    aefr_posterior_div = aux.get(prefix + "_aefr_posterior_diversity_all", final_prob.new_zeros(()))
    aefr_center_bias_abs = aux.get(prefix + "_aefr_posterior_center_bias_abs", final_prob.new_zeros(()))
    aefr_center_bias_signed = aux.get(prefix + "_aefr_posterior_center_bias_signed", final_prob.new_zeros(()))
    aefr_boundary_mask = aux.get(prefix + "_aefr_boundary_mask", torch.zeros_like(anchor))
    aefr_displacement = aux.get(prefix + "_aefr_boundary_displacement_px", torch.zeros_like(anchor))
    aefr_interior_delta = aux.get(prefix + "_aefr_interior_delta_logit", torch.zeros_like(anchor))
    aefr_state_logits = aux.get(
        prefix + "_aefr_state_logits",
        anchor.new_zeros((anchor.shape[0], 5, *anchor.shape[-2:])),
    )
    aefr_state_probs = aux.get(
        prefix + "_aefr_state_probs",
        anchor.new_zeros((anchor.shape[0], 5, *anchor.shape[-2:])),
    )
    aefr_ownership_logits = aux.get(
        prefix + "_aefr_ownership_logits",
        anchor.new_zeros((anchor.shape[0], 3, *anchor.shape[-2:])),
    )
    aefr_ownership_probs = aux.get(
        prefix + "_aefr_ownership_probs",
        anchor.new_zeros((anchor.shape[0], 3, *anchor.shape[-2:])),
    )
    aefr_boundary_magnitude = aux.get(prefix + "_aefr_boundary_magnitude_px", torch.zeros_like(anchor))
    aefr_signed_boundary = aux.get(prefix + "_aefr_signed_boundary_action", torch.zeros_like(anchor))
    aefr_signed_interior = aux.get(prefix + "_aefr_signed_interior_action", torch.zeros_like(anchor))
    aefr_error_localizer_logit = aux.get(prefix + "_aefr_error_localizer_logit", torch.zeros_like(anchor))
    aefr_error_localizer_prob = aux.get(prefix + "_aefr_error_localizer_prob", torch.zeros_like(anchor))
    aefr_edit_logit = aux.get(prefix + "_aefr_edit_logit", torch.zeros_like(anchor))
    aefr_edit_prob = aux.get(prefix + "_aefr_edit_prob", torch.zeros_like(anchor))
    aefr_direction_logit = aux.get(prefix + "_aefr_direction_logit", torch.zeros_like(anchor))
    aefr_direction_prob = aux.get(prefix + "_aefr_direction_prob", torch.zeros_like(anchor))
    aefr_signed_action = aux.get(prefix + "_aefr_signed_action", torch.zeros_like(anchor))
    aefr_commit_mask = aux.get(prefix + "_aefr_commit_mask", torch.zeros_like(anchor))

    target4 = target[:, None].to(final_prob)
    if target4.shape[-2:] != anchor.shape[-2:]:
        target4 = F.interpolate(target4.float(), size=anchor.shape[-2:], mode="nearest").to(final_prob)
        target = target4[:, 0]
    else:
        target = target.to(final_prob)

    geo_loss, geo_bce, geo_dice_loss = _seg_loss(
        geo_logits, geo_prob, target, ce_weight, dice_weight
    )
    final_loss, final_bce, final_dice_loss = _seg_loss(
        final_logits, final_prob, target, ce_weight, dice_weight
    )

    # Canonical full-view tensors are [B,V,H,W]. Accept the historical
    # [B,V,1,H,W] spelling through explicit normalization.
    if view_logits.ndim == 5 and view_logits.shape[2] == 1:
        view_logits = view_logits[:, :, 0]
        view_probs = view_probs[:, :, 0]
    if view_logits.ndim != 4 or view_probs.ndim != 4:
        raise ValueError(
            "C2R full-view logits/probs must be [B,V,H,W], got "
            f"{tuple(view_logits.shape)} and {tuple(view_probs.shape)}"
        )
    if view_logits.shape != view_probs.shape or view_logits.shape[1] < 1:
        raise ValueError(
            "C2R full-view logits/probs must have identical non-empty shapes, got "
            f"{tuple(view_logits.shape)} and {tuple(view_probs.shape)}"
        )

    view_losses, view_bces, view_dices = [], [], []
    for vi in range(view_logits.shape[1]):
        li, bi, di = _masked_region_seg_loss(
            view_logits[:, vi:vi+1],
            view_probs[:, vi:vi+1],
            target4,
            region,
            ce_weight,
            dice_weight,
        )
        view_losses.append(li); view_bces.append(bi); view_dices.append(di)
    view_loss = torch.stack(view_losses).mean()
    view_bce = torch.stack(view_bces).mean()
    view_dice_loss = torch.stack(view_dices).mean()
    view_w = max(float(_cfg_get(m1, "GEOTR_C2R_VIEW_LOSS_WEIGHT", 1.0)), 0.0)
    c2r_v2 = bool(_cfg_get(m1, "GEOTR_C2R_CANONICAL_ROI_ENABLED", False))
    pc2r_v3 = bool(_cfg_get(m1, "GEOTR_PC2R_POSTERIOR_V3_ENABLED", False))
    pc2r_v32 = bool(_cfg_get(m1, "GEOTR_PC2R_OPERATOR_ALIGNED_V32_ENABLED", False))
    aefr_enabled = bool(_cfg_get(m1, "GEOTR_AEFR_ENABLED", False))
    aefr_stage = str(_cfg_get(m1, "GEOTR_AEFR_STAGE", "single_atomic")).strip().lower()
    canonical_logits = torch.logit(mean_prob.clamp(EPS, 1.0 - EPS))
    canonical_loss, canonical_bce, canonical_dice_loss = _masked_region_seg_loss(
        canonical_logits, mean_prob, target4, region, ce_weight, dice_weight
    )
    # AEFR E2/E3 train the actual continuous deployed operator inside the
    # deterministic ROI from step zero.  This is a standard BCE+Dice objective
    # over the deployed ROI; no learned gate/reward/utility target is introduced.
    deployed_roi_loss, deployed_roi_bce, deployed_roi_dice_loss = _masked_region_seg_loss(
        final_logits, final_prob, target4, region, ce_weight, dice_weight
    )
    typed_state_stage = aefr_enabled and aefr_stage in {"typed_state_taylor", "typed_state_exact"}
    soft_ownership_stage = aefr_enabled and aefr_stage == "soft_ownership_exact"
    intervention_stage = aefr_enabled and aefr_stage == "intervention_factorized"
    smi_stage = aefr_enabled and aefr_stage == "selective_minimal_intervention"
    slr_stage = aefr_enabled and aefr_stage == "sparse_local_rerendering"
    slr_ucdrt_r2 = slr_stage and bool(_cfg_get(m1, "GEOTR_SLR_UCDRT_R2_ENABLED", False))
    slr_ucdrt = slr_stage and (slr_ucdrt_r2 or bool(_cfg_get(m1, "GEOTR_SLR_UCDRT_ENABLED", False)))
    state_target = torch.zeros_like(target, dtype=torch.long)
    state_masks = tuple(torch.zeros_like(target, dtype=torch.bool) for _ in range(5))
    state_loss = final_prob.sum() * 0.0
    state_present_classes = final_prob.new_zeros(())
    state_metrics = {
        "macro_f1": final_prob.new_zeros(()),
        "pred_edit_fraction": final_prob.new_zeros(()),
    }
    for _name in ("keep", "boundary_add", "boundary_remove", "interior_add", "interior_remove"):
        state_metrics[_name + "_precision"] = final_prob.new_zeros(())
        state_metrics[_name + "_recall"] = final_prob.new_zeros(())
        state_metrics[_name + "_f1"] = final_prob.new_zeros(())
        state_metrics[_name + "_count"] = final_prob.new_zeros(())
    ownership_loss = final_prob.sum() * 0.0
    ownership_target = final_prob.new_zeros((final_prob.shape[0], 3, *final_prob.shape[-2:]))
    ownership_metrics = {
        "macro_f1": final_prob.new_zeros(()), "pred_edit_fraction": final_prob.new_zeros(()),
        "pred_edit_mass": final_prob.new_zeros(()), "target_edit_mass": final_prob.new_zeros(()),
        "edit_mass_abs_error": final_prob.new_zeros(()), "signed_action_mae": final_prob.new_zeros(()),
        "signed_target_abs_mean": final_prob.new_zeros(()),
    }
    for _name in ("keep", "add", "remove"):
        for _metric in ("precision", "recall", "f1", "count"):
            ownership_metrics[f"{_name}_{_metric}"] = final_prob.new_zeros(())

    intervention_error_loss = final_prob.sum() * 0.0
    intervention_error_bce = final_prob.sum() * 0.0
    intervention_error_dice = final_prob.sum() * 0.0
    intervention_edit_loss = final_prob.sum() * 0.0
    intervention_direction_loss = final_prob.sum() * 0.0
    smi_boundary_magnitude_loss = final_prob.sum() * 0.0
    smi_selective_deploy_loss = final_prob.sum() * 0.0
    smi_selective_deploy_bce = final_prob.sum() * 0.0
    smi_selective_deploy_dice = final_prob.sum() * 0.0
    smi_boundary_target_mean = final_prob.sum() * 0.0
    smi_commit_fraction = final_prob.sum() * 0.0
    intervention_metrics = {
        k: final_prob.new_zeros(()) for k in (
            "localizer_precision", "localizer_recall", "localizer_f1",
            "edit_precision", "edit_recall", "edit_f1", "direction_accuracy",
            "add_precision", "add_recall", "add_f1",
            "remove_precision", "remove_recall", "remove_f1",
            "pred_edit_fraction", "target_edit_fraction",
            "signed_action_mae", "zero_action_mae", "signed_advantage",
            "error_prob_mean_error", "error_prob_mean_correct",
            "edit_prob_mean_error_roi", "edit_prob_mean_correct_roi",
        )
    }
    if typed_state_stage:
        state_target, state_masks = _aefr_five_state_targets(
            anchor, target, region, aefr_boundary_mask
        )
        state_loss, state_present_classes = _macro_present_class_ce(
            aefr_state_logits, state_target, region, num_classes=5
        )
        state_metrics = _five_state_metrics(aefr_state_logits, state_target, region)
    if soft_ownership_stage:
        ownership_target = _aefr_soft_ownership_targets(anchor, target, region)
        ownership_loss = _soft_ownership_ce(aefr_ownership_logits, ownership_target, region)
        ownership_metrics = _three_ownership_metrics(
            aefr_ownership_logits, ownership_target, anchor, target, region
        )
    if intervention_stage:
        error_target, direction_target, _signed_target, _fn, _fp = _aefr_intervention_targets(anchor, target)
        full_support = torch.ones_like(error_target)
        roi_support = region.detach().to(error_target)
        true_edit_support = roi_support * error_target
        pos_weight = float(_cfg_get(m1, "GEOTR_AEFR_INTERVENTION_ERROR_POS_WEIGHT", 6.0))
        intervention_error_bce = _masked_bce_logits(
            aefr_error_localizer_logit, error_target, full_support, pos_weight=pos_weight
        )
        intervention_error_dice = _masked_binary_dice_loss(
            aefr_error_localizer_prob, error_target, full_support
        )
        intervention_error_loss = 0.5 * (intervention_error_bce + intervention_error_dice)
        intervention_edit_loss = _masked_bce_logits(
            aefr_edit_logit, error_target, roi_support, pos_weight=pos_weight
        )
        intervention_direction_loss = _masked_bce_logits(
            aefr_direction_logit, direction_target, true_edit_support, pos_weight=1.0
        )
        intervention_metrics = _aefr_intervention_metrics(
            aefr_error_localizer_prob, aefr_edit_prob, aefr_direction_prob, aefr_signed_action,
            anchor, target, region
        )
    if smi_stage:
        error_target, direction_target, _signed_target, _fn, _fp = _aefr_intervention_targets(anchor, target)
        full_support = torch.ones_like(error_target)
        roi_support = region.detach().to(error_target)
        true_edit_support = roi_support * error_target

        # Proper scoring rule: if the output is used as P(error) and hard COMMIT,
        # do not distort its optimum with class weighting. Dice is retained only
        # as a diagnostic so calibration and ranking can be inspected separately.
        intervention_error_bce = _masked_bce_logits(
            aefr_error_localizer_logit, error_target, full_support, pos_weight=1.0
        )
        intervention_error_dice = _masked_binary_dice_loss(
            aefr_error_localizer_prob, error_target, full_support
        )
        intervention_error_loss = intervention_error_bce
        intervention_edit_loss = final_prob.sum() * 0.0  # WHERE == WHETHER in SMI.
        intervention_direction_loss = _masked_bce_logits(
            aefr_direction_logit, direction_target, true_edit_support, pos_weight=1.0
        )

        max_disp = float(_cfg_get(m1, "GEOTR_AEFR_BOUNDARY_MAX_DISPLACEMENT_PX", 4.0))
        boundary_target = _aefr_boundary_magnitude_target(anchor, max_disp)
        boundary_true_support = true_edit_support * (aefr_boundary_mask.detach() > 0.5).to(error_target)
        smi_boundary_magnitude_loss = _masked_smooth_l1(
            aefr_boundary_magnitude, boundary_target, boundary_true_support
        )
        smi_boundary_target_mean = _masked_mean(boundary_target[:,0], boundary_true_support[:,0] > 0.5)

        # Context ROI is not an action domain. Train the deployed segmentation
        # only where a true residual exists OR where the model actually commits.
        # Thus already-correct, uncommitted context pixels receive no probability-
        # sharpening gradient, while false-positive commits are still penalized.
        commit_support = (aefr_commit_mask.detach() > 0.5).to(error_target) * roi_support
        selective_support = torch.maximum(true_edit_support, commit_support)
        smi_selective_deploy_loss, smi_selective_deploy_bce, smi_selective_deploy_dice = _masked_region_seg_loss(
            final_logits, final_prob, target4, selective_support, ce_weight, dice_weight
        )
        smi_commit_fraction = _safe_ratio(commit_support.sum(), roi_support.sum())

        # Metrics reuse the IFR names for longitudinal comparability. In SMI the
        # edit probability is exactly the calibrated error posterior.
        intervention_metrics = _aefr_intervention_metrics(
            aefr_error_localizer_prob, aefr_error_localizer_prob, aefr_direction_prob, aefr_signed_action,
            anchor, target, region
        )
    # SLR native supervision.  The deployed patch itself is permanently and
    # directly supervised.  Training-only positive/clean samples enlarge the
    # same reconstruction distribution; there is no decaying teacher branch.
    slr_selector_loss = final_prob.sum() * 0.0
    slr_selector_mae = final_prob.sum() * 0.0
    slr_pred_patch_loss = final_prob.sum() * 0.0
    slr_pred_patch_bce = final_prob.sum() * 0.0
    slr_pred_patch_dice = final_prob.sum() * 0.0
    slr_positive_seg_loss = final_prob.sum() * 0.0
    slr_positive_bce = final_prob.sum() * 0.0
    slr_positive_dice = final_prob.sum() * 0.0
    slr_clean_seg_loss = final_prob.sum() * 0.0
    slr_clean_bce = final_prob.sum() * 0.0
    slr_clean_dice = final_prob.sum() * 0.0
    slr_patch_loss = final_prob.sum() * 0.0
    slr_patch_seg_diagnostic = final_prob.sum() * 0.0
    slr_patch_bce = final_prob.sum() * 0.0
    slr_patch_dice = final_prob.sum() * 0.0
    slr_actor_type_loss = final_prob.sum() * 0.0
    slr_actor_dose_loss = final_prob.sum() * 0.0
    slr_actor_keep_loss = final_prob.sum() * 0.0
    slr_sdf_loss = final_prob.sum() * 0.0
    slr_pred_sdf_loss = final_prob.sum() * 0.0
    slr_positive_sdf_loss = final_prob.sum() * 0.0
    slr_clean_sdf_loss = final_prob.sum() * 0.0
    slr_deploy_loss = final_prob.sum() * 0.0
    slr_deploy_bce = final_prob.sum() * 0.0
    slr_deploy_dice = final_prob.sum() * 0.0
    slr_clean_residual_abs = final_prob.sum() * 0.0
    slr_clean_target_residual_abs = final_prob.sum() * 0.0
    # Historical metric aliases retained so old log consumers do not break.
    slr_oracle_seg_loss = final_prob.sum() * 0.0
    slr_oracle_bce = final_prob.sum() * 0.0
    slr_oracle_dice = final_prob.sum() * 0.0
    slr_oracle_sdf_loss = final_prob.sum() * 0.0
    slr_curriculum_progress = final_prob.sum() * 0.0
    slr_oracle_weight_effective = final_prob.sum() * 0.0
    slr_deploy_weight_effective = final_prob.sum() * 0.0
    # UCDRT-native objectives/diagnostics.
    slr_ucdrt_state_loss = final_prob.sum() * 0.0
    slr_ucdrt_state_present = final_prob.new_zeros(())
    slr_ucdrt_utility_loss = final_prob.sum() * 0.0
    slr_ucdrt_action_loss = final_prob.sum() * 0.0
    slr_ucdrt_move_loss = final_prob.sum() * 0.0
    slr_ucdrt_interior_loss = final_prob.sum() * 0.0
    slr_ucdrt_paired_loss = final_prob.sum() * 0.0
    slr_ucdrt_utility_mae = final_prob.new_zeros(())
    slr_ucdrt_utility_corr = final_prob.new_zeros(())
    slr_ucdrt_commit_rate = final_prob.new_zeros(())
    slr_ucdrt_commit_precision = final_prob.new_zeros(())
    slr_ucdrt_deploy_support_fraction = final_prob.new_zeros(())
    # UCDRT-R2 operator-consistent objectives/diagnostics.
    slr_ucdrt_r2_edit_loss = final_prob.sum() * 0.0
    slr_ucdrt_r2_type_loss = final_prob.sum() * 0.0
    slr_ucdrt_r2_param_loss = final_prob.sum() * 0.0
    slr_ucdrt_r2_actor_loss = final_prob.sum() * 0.0
    slr_ucdrt_r2_actor_hard_loss = final_prob.sum() * 0.0
    slr_ucdrt_r2_actor_soft_loss = final_prob.sum() * 0.0
    slr_ucdrt_r2_critic_loss = final_prob.sum() * 0.0
    slr_ucdrt_r2_paired_exec_loss = final_prob.sum() * 0.0
    slr_ucdrt_r2_paired_reachability = final_prob.new_zeros(())
    slr_ucdrt_r2_hard_mean_gain = final_prob.new_zeros(())
    slr_ucdrt_r2_hard_positive_rate = final_prob.new_zeros(())
    slr_ucdrt_r2_hard_oracle_gain = final_prob.new_zeros(())
    slr_ucdrt_r2_soft_mean_gain = final_prob.new_zeros(())
    slr_ucdrt_r2_soft_positive_rate = final_prob.new_zeros(())
    slr_ucdrt_r2_soft_oracle_gain = final_prob.new_zeros(())
    slr_ucdrt_r2_hard_soft_gap = final_prob.new_zeros(())
    slr_ucdrt_r2_critic_precision = final_prob.new_zeros(())
    slr_ucdrt_r2_critic_recall = final_prob.new_zeros(())
    slr_ucdrt_r2_critic_positive_rate = final_prob.new_zeros(())
    slr_ucdrt_r2_factor_metrics = {
        "edit_f1": final_prob.new_zeros(()), "edit_precision": final_prob.new_zeros(()),
        "edit_recall": final_prob.new_zeros(()), "target_edit_fraction": final_prob.new_zeros(()),
        "pred_edit_fraction": final_prob.new_zeros(()), "type_macro_f1": final_prob.new_zeros(()),
        "move_f1": final_prob.new_zeros(()), "add_f1": final_prob.new_zeros(()),
        "remove_f1": final_prob.new_zeros(()),
    }
    slr_ucdrt_state_metrics = {
        "macro_f1": final_prob.new_zeros(()), "pred_edit_fraction": final_prob.new_zeros(()),
    }
    for _n in ("keep", "move", "add", "remove"):
        for _m in ("precision", "recall", "f1", "count"):
            slr_ucdrt_state_metrics[f"{_n}_{_m}"] = final_prob.new_zeros(())
    if slr_stage:
        slr_selector_logits = aux[prefix + "_slr_selector_logits"]
        slr_selector_prob = aux[prefix + "_slr_selector_prob"]
        slr_selector_target = aux[prefix + "_slr_selector_target"].detach().to(slr_selector_prob)
        slr_selector_loss = F.binary_cross_entropy_with_logits(
            slr_selector_logits.float(), slr_selector_target.float()
        ).to(final_prob)
        slr_selector_mae = (slr_selector_prob.detach() - slr_selector_target).abs().mean()

        # 1) Prediction/deployment alignment: local prediction is supervised in
        # its *raw* coordinate. Boundary attenuation belongs only to deployment.
        pl = _first_present(
            aux,
            prefix + "_slr_raw_patch_logits",
            prefix + "_slr_pred_patch_logits",
        )
        pp = _first_present(
            aux,
            prefix + "_slr_raw_patch_probs",
            prefix + "_slr_pred_patch_probs",
        )
        pv = aux[prefix + "_slr_pred_patch_valid"]
        pt = aux[prefix + "_slr_pred_patch_target"]
        psdfs = aux[prefix + "_slr_pred_sdf_support"]
        slr_pred_patch_loss, slr_pred_patch_bce, slr_pred_patch_dice = _slr_patch_seg_loss(
            pl, pp, pt, pv, ce_weight, dice_weight, support=psdfs
        )

        # 2) Training-only residual-positive samples; compatibility falls back
        # to the historical oracle names when loading an older overlay.
        pol = _first_present(
            aux,
            prefix + "_slr_positive_patch_logits",
            prefix + "_slr_oracle_patch_logits",
        )
        pop = _first_present(
            aux,
            prefix + "_slr_positive_patch_probs",
            prefix + "_slr_oracle_patch_probs",
        )
        pov = _first_present(
            aux,
            prefix + "_slr_positive_patch_valid",
            prefix + "_slr_oracle_patch_valid",
        )
        pot = _first_present(
            aux,
            prefix + "_slr_positive_patch_target",
            prefix + "_slr_oracle_patch_target",
        )
        pos_sdfs = _first_present(
            aux,
            prefix + "_slr_positive_sdf_support",
            prefix + "_slr_oracle_sdf_support",
        )
        slr_positive_seg_loss, slr_positive_bce, slr_positive_dice = _slr_patch_seg_loss(
            pol, pop, pot, pov, ce_weight, dice_weight, support=pos_sdfs
        )

        # 3) Training-only clean patches are *structure-preservation teachers*.
        # They intentionally receive no BCE/Dice confidence-sharpening gradient;
        # their stationary absolute-SDF regression below is the supervision.
        cl = aux[prefix + "_slr_clean_patch_logits"]
        cp = aux[prefix + "_slr_clean_patch_probs"]
        cv = aux[prefix + "_slr_clean_patch_valid"]
        ct = aux[prefix + "_slr_clean_patch_target"]
        slr_clean_seg_loss = final_prob.sum() * 0.0
        slr_clean_bce = final_prob.sum() * 0.0
        slr_clean_dice = final_prob.sum() * 0.0
        cs_abs = _first_present(
            aux,
            prefix + "_slr_clean_sdf_absolute",
            prefix + "_slr_clean_sdf_delta",
        )
        cs_anchor = aux.get(prefix + "_slr_clean_sdf_anchor", torch.zeros_like(cs_abs))
        slr_clean_residual_abs = ((cs_abs - cs_anchor).abs() * cv.to(cs_abs)).sum() / cv.to(cs_abs).sum().clamp_min(1.0)
        cs_target_for_diag = aux[prefix + "_slr_clean_sdf_target"].to(cs_abs)
        slr_clean_target_residual_abs = ((cs_target_for_diag - cs_anchor).abs() * cv.to(cs_abs)).sum() / cv.to(cs_abs).sum().clamp_min(1.0)

        # Patch BCE/Dice is retained as an actor-quality diagnostic. The actual
        # actor objective below is typed and minimum-dose, and includes clean
        # KEEP teachers explicitly.
        all_l = torch.cat([pl, pol], dim=1)
        all_p = torch.cat([pp, pop], dim=1)
        all_t = torch.cat([pt, pot], dim=1)
        all_v = torch.cat([pv, pov], dim=1)
        all_s = torch.cat([psdfs, pos_sdfs], dim=1)
        slr_patch_seg_diagnostic, slr_patch_bce, slr_patch_dice = _slr_patch_seg_loss(
            all_l, all_p, all_t, all_v, ce_weight, dice_weight, support=all_s
        )

        # OCRA replaces only the sparse direct refiner.  The older UCDRT
        # transition refiner owns a different four-state objective below and
        # must not be forced to manufacture OCRA tensors.
        if not slr_ucdrt:
            pred_state_logits = aux[prefix + "_slr_pred_action_state_logits"]
            pred_dose = aux[prefix + "_slr_pred_action_dose"]
            pred_signed = aux[prefix + "_slr_pred_signed_action"]
            pred_type_target = aux[prefix + "_slr_pred_action_type_target"]
            pred_dose_target = aux[prefix + "_slr_pred_action_dose_target"]
            pred_action_support = aux[prefix + "_slr_pred_patch_action_support"]
            pos_state_logits = aux[prefix + "_slr_positive_action_state_logits"]
            pos_dose = aux[prefix + "_slr_positive_action_dose"]
            pos_signed = aux[prefix + "_slr_positive_signed_action"]
            pos_type_target = aux[prefix + "_slr_positive_action_type_target"]
            pos_dose_target = aux[prefix + "_slr_positive_action_dose_target"]
            pos_action_support = aux[prefix + "_slr_positive_action_support"]
            clean_state_logits = aux[prefix + "_slr_clean_action_state_logits"]
            clean_dose = aux[prefix + "_slr_clean_action_dose"]
            clean_signed = aux[prefix + "_slr_clean_signed_action"]
            clean_type_target = aux[prefix + "_slr_clean_action_type_target"]
            clean_dose_target = aux[prefix + "_slr_clean_action_dose_target"]
            clean_action_support = aux[prefix + "_slr_clean_action_support"]
            slr_patch_loss, slr_actor_type_loss, slr_actor_dose_loss, slr_actor_keep_loss = _slr_typed_actor_loss(
                torch.cat([pred_state_logits, pos_state_logits, clean_state_logits], dim=1),
                torch.cat([pred_dose, pos_dose, clean_dose], dim=1),
                torch.cat([pred_signed, pos_signed, clean_signed], dim=1),
                torch.cat([pred_type_target, pos_type_target, clean_type_target], dim=1),
                torch.cat([pred_dose_target, pos_dose_target, clean_dose_target], dim=1),
                torch.cat([pv, pov, cv], dim=1),
                torch.cat([pred_action_support, pos_action_support, clean_action_support], dim=1),
            )
        else:
            slr_patch_loss = slr_patch_seg_diagnostic

        psdf = _first_present(
            aux,
            prefix + "_slr_pred_sdf_absolute",
            prefix + "_slr_pred_sdf_delta",
        )
        psdft = aux[prefix + "_slr_pred_sdf_target"]
        pos_sdf = _first_present(
            aux,
            prefix + "_slr_positive_sdf_absolute",
            prefix + "_slr_positive_sdf_delta",
            prefix + "_slr_oracle_sdf_delta",
        )
        pos_sdft = _first_present(
            aux,
            prefix + "_slr_positive_sdf_target",
            prefix + "_slr_oracle_sdf_target",
        )
        csdf = _first_present(
            aux,
            prefix + "_slr_clean_sdf_absolute",
            prefix + "_slr_clean_sdf_delta",
        )
        csdft = aux[prefix + "_slr_clean_sdf_target"]
        csdfs = aux[prefix + "_slr_clean_sdf_support"]
        slr_sdf_radius = max(float(_cfg_get(m1, "GEOTR_SLR_SDF_RADIUS_PX", 8)), 1.0)
        slr_pred_sdf_loss = _slr_sdf_loss(psdf, psdft, pv, psdfs, slr_sdf_radius)
        slr_positive_sdf_loss = _slr_sdf_loss(pos_sdf, pos_sdft, pov, pos_sdfs, slr_sdf_radius)
        slr_clean_sdf_loss = _slr_sdf_loss(csdf, csdft, cv, csdfs, slr_sdf_radius)
        slr_sdf_loss = _slr_sdf_loss(
            torch.cat([psdf, pos_sdf, csdf], dim=1),
            torch.cat([psdft, pos_sdft, csdft], dim=1),
            torch.cat([pv, pov, cv], dim=1),
            torch.cat([psdfs, pos_sdfs, csdfs], dim=1),
            slr_sdf_radius,
        )

        # Composition supervision must match the physical executor. OCRA and
        # UCDRT supervise only factual residual pixels plus pixels on which the
        # model actually
        # proposes an action. This prevents the dominant KEEP majority in a 33x33
        # ROI from turning the deploy objective back into local reconstruction,
        # while still penalizing every false action it introduces.
        action_region = aux.get(prefix + "_slr_action_region_mask", region).to(final_prob)
        if slr_ucdrt:
            factual_residual = (
                ((anchor.detach() >= 0.5) != (target4.detach() >= 0.5)).to(final_prob)
                * (action_region > 0).to(final_prob)
            )
            predicted_action = aux.get(
                prefix + "_slr_ucdrt_action_mask_full", torch.zeros_like(action_region)
            ).detach().to(final_prob)
            deploy_support = torch.maximum(factual_residual, (predicted_action > 0).to(final_prob))
            slr_ucdrt_deploy_support_fraction = _safe_ratio(
                deploy_support.sum(), (action_region > 0).to(final_prob).sum()
            )
        else:
            # OCRA trains the exact executor on every factual error in the
            # selected ROI plus every action the model actually proposes. This
            # preserves correct inactive pixels without hiding false actions.
            factual_residual = (
                ((anchor.detach() >= 0.5) != (target4.detach() >= 0.5)).to(final_prob)
                * (action_region > 0).to(final_prob)
            )
            action_delta_full = aux.get(
                prefix + "_slr_action_delta_full", torch.zeros_like(action_region)
            ).to(final_prob)
            predicted_action = (action_delta_full.detach().abs() > 1.0e-6).to(final_prob)
            deploy_support = torch.maximum(factual_residual, predicted_action * (action_region > 0).to(final_prob))
        if slr_ucdrt:
            slr_deploy_loss, slr_deploy_bce, slr_deploy_dice = _slr_ucdrt_masked_deploy_loss(
                final_logits, final_prob, target4, deploy_support, ce_weight, dice_weight
            )
        else:
            slr_deploy_loss, slr_deploy_bce, slr_deploy_dice = _masked_region_seg_loss(
                final_logits, final_prob, target4, deploy_support, ce_weight, dice_weight
            )

        if slr_ucdrt_r2:
            # -----------------------------------------------------------------
            # UCDRT-R2: operator-consistent actor before selective policy.
            # -----------------------------------------------------------------
            tgt_margin = float(_cfg_get(m1, "GEOTR_SLR_UCDRT_TARGET_MARGIN", 0.05))
            edit_gamma = float(_cfg_get(m1, "GEOTR_SLR_UCDRT_R2_EDIT_FOCAL_GAMMA", 1.5))
            gain_scale = float(_cfg_get(m1, "GEOTR_SLR_UCDRT_R2_UTILITY_GAIN_SCALE", 20.0))
            boundary_qw = float(_cfg_get(m1, "GEOTR_SLR_UCDRT_R2_UTILITY_BOUNDARY_WEIGHT", 0.25))
            rank_margin = float(_cfg_get(m1, "GEOTR_SLR_UCDRT_R2_UTILITY_RANK_MARGIN", 0.05))

            pred_anchor = aux[prefix + "_slr_pred_patch_anchor_prob"]
            edit_logits = aux[prefix + "_slr_ucdrt_r2_edit_logits"]
            edit_prob = aux[prefix + "_slr_ucdrt_r2_edit_probs"]
            type_logits = aux[prefix + "_slr_ucdrt_r2_type_logits"]
            move_bin_logits = aux[prefix + "_slr_ucdrt_r2_move_bin_logits"]
            move_dict = aux[prefix + "_slr_ucdrt_r2_move_dictionary_probs"]
            add_dose = aux[prefix + "_slr_ucdrt_r2_add_dose"]
            remove_dose = aux[prefix + "_slr_ucdrt_r2_remove_dose"]
            hard_candidate = aux[prefix + "_slr_ucdrt_r2_hard_candidate_probs"]
            soft_candidate = aux[prefix + "_slr_ucdrt_r2_soft_candidate_probs"]
            boundary_patch = aux[prefix + "_slr_ucdrt_boundary_mask"]

            r2_targets = _slr_ucdrt_r2_targets(
                pred_anchor, pt, pv, boundary_patch, move_dict,
                target_margin=tgt_margin,
            )
            slr_ucdrt_r2_edit_loss = _slr_ucdrt_r2_focal_edit_loss(
                edit_logits, r2_targets["edit"], pv, gamma=edit_gamma
            )
            slr_ucdrt_r2_type_loss, slr_ucdrt_state_present = _slr_ucdrt_r2_type_loss(
                type_logits, r2_targets["type"], r2_targets["edit"], pv
            )
            (
                slr_ucdrt_r2_param_loss,
                slr_ucdrt_move_loss,
                _r2_add_param_loss,
                _r2_remove_param_loss,
            ) = _slr_ucdrt_r2_parameter_loss(
                move_bin_logits, add_dose, remove_dose, r2_targets
            )
            (
                slr_ucdrt_r2_actor_loss,
                slr_ucdrt_r2_actor_hard_loss,
                slr_ucdrt_r2_actor_soft_loss,
            ) = _slr_ucdrt_r2_actor_exec_loss(
                hard_candidate, soft_candidate, pred_anchor, pt, pv, edit_prob
            )
            slr_ucdrt_r2_factor_metrics = _slr_ucdrt_r2_factor_metrics(
                edit_logits, type_logits, r2_targets, pv
            )

            # Exact action-conditioned marginal critic targets under the same WOLA
            # accumulation and the same predicted selection trace as deployment.
            step_logits = aux[prefix + "_slr_ucdrt_r2_critic_step_logits"]
            step_values = aux[prefix + "_slr_ucdrt_r2_critic_step_values"]
            step_available = aux[prefix + "_slr_ucdrt_r2_critic_step_available"]
            step_selected = aux[prefix + "_slr_ucdrt_r2_selection_step_index"]
            cand_num = aux[prefix + "_slr_ucdrt_r2_candidate_full_num"]
            cand_den = aux[prefix + "_slr_ucdrt_r2_candidate_full_den"]
            critic_target = _slr_ucdrt_r2_critic_targets(
                anchor.detach(), target4.detach(), cand_num, cand_den,
                step_selected, step_available, boundary_weight=boundary_qw,
                gain_scale=gain_scale,
            )
            slr_ucdrt_r2_critic_loss, critic_metrics = _slr_ucdrt_r2_critic_loss(
                step_logits, step_values, critic_target, step_available,
                rank_margin=rank_margin,
            )
            slr_ucdrt_utility_loss = slr_ucdrt_r2_critic_loss
            slr_ucdrt_utility_mae = critic_metrics["mae"]
            slr_ucdrt_utility_corr = critic_metrics["corr"]
            slr_ucdrt_r2_critic_precision = critic_metrics["precision"]
            slr_ucdrt_r2_critic_recall = critic_metrics["recall"]
            slr_ucdrt_r2_critic_positive_rate = critic_metrics["positive_rate"]

            selected = aux[prefix + "_slr_ucdrt_r2_selected_candidate_mask"].detach().to(final_prob)
            candidate_valid = (pv.sum(dim=(2,3,4)) > 0).to(selected)
            slr_ucdrt_commit_rate = (selected * candidate_valid).sum() / candidate_valid.sum().clamp_min(1.0)
            slr_ucdrt_commit_precision = slr_ucdrt_r2_critic_precision

            # Actor diagnostics: these audit the action family before the critic can
            # hide a bad actor by selecting STOP.
            anchor_q = _slr_ucdrt_quality_per_candidate(pred_anchor, pt, pv)
            hard_q = _slr_ucdrt_quality_per_candidate(hard_candidate, pt, pv)
            soft_q = _slr_ucdrt_quality_per_candidate(soft_candidate, pt, pv)
            hard_gain = (hard_q - anchor_q) * candidate_valid
            soft_gain = (soft_q - anchor_q) * candidate_valid
            denom_cv = candidate_valid.sum().clamp_min(1.0)
            slr_ucdrt_r2_hard_mean_gain = hard_gain.sum() / denom_cv
            slr_ucdrt_r2_soft_mean_gain = soft_gain.sum() / denom_cv
            slr_ucdrt_r2_hard_positive_rate = ((hard_gain > 0).to(hard_gain) * candidate_valid).sum() / denom_cv
            slr_ucdrt_r2_soft_positive_rate = ((soft_gain > 0).to(soft_gain) * candidate_valid).sum() / denom_cv
            slr_ucdrt_r2_hard_oracle_gain = hard_gain.clamp_min(0).max(dim=1).values.mean()
            slr_ucdrt_r2_soft_oracle_gain = soft_gain.clamp_min(0).max(dim=1).values.mean()
            slr_ucdrt_r2_hard_soft_gap = slr_ucdrt_r2_soft_oracle_gain - slr_ucdrt_r2_hard_oracle_gain

            # R1 compatibility metrics map to the factorized R2 semantics.
            slr_ucdrt_state_loss = slr_ucdrt_r2_edit_loss + slr_ucdrt_r2_type_loss
            slr_ucdrt_action_loss = slr_ucdrt_r2_param_loss
            slr_ucdrt_interior_loss = 0.5 * (_r2_add_param_loss + _r2_remove_param_loss)
            slr_ucdrt_state_metrics["macro_f1"] = 0.5 * (
                slr_ucdrt_r2_factor_metrics["edit_f1"] + slr_ucdrt_r2_factor_metrics["type_macro_f1"]
            )
            slr_ucdrt_state_metrics["pred_edit_fraction"] = slr_ucdrt_r2_factor_metrics["pred_edit_fraction"]
            slr_ucdrt_state_metrics["keep_f1"] = slr_ucdrt_r2_factor_metrics["edit_f1"]
            slr_ucdrt_state_metrics["move_f1"] = slr_ucdrt_r2_factor_metrics["move_f1"]
            slr_ucdrt_state_metrics["add_f1"] = slr_ucdrt_r2_factor_metrics["add_f1"]
            slr_ucdrt_state_metrics["remove_f1"] = slr_ucdrt_r2_factor_metrics["remove_f1"]

            # Operator-closed paired branch includes an *executable outcome* loss,
            # not only state/parameter labels.
            if prefix + "_slr_ucdrt_r2_paired_edit_logits" in aux:
                pair_anchor = aux[prefix + "_slr_ucdrt_paired_anchor_prob"]
                pair_target = aux[prefix + "_slr_ucdrt_paired_target"]
                pair_valid = aux[prefix + "_slr_ucdrt_paired_valid"]
                pair_boundary = (
                    aux[prefix + "_slr_ucdrt_paired_anchor_sdf"].detach().abs()
                    <= float(_cfg_get(m1, "GEOTR_SLR_UCDRT_BOUNDARY_RADIUS_PX", 5))
                ).to(pair_anchor) * pair_valid.to(pair_anchor)
                pair_targets = _slr_ucdrt_r2_targets(
                    pair_anchor, pair_target, pair_valid, pair_boundary,
                    aux[prefix + "_slr_ucdrt_r2_paired_move_dictionary_probs"],
                    target_margin=tgt_margin,
                )
                pair_edit_loss = _slr_ucdrt_r2_focal_edit_loss(
                    aux[prefix + "_slr_ucdrt_r2_paired_edit_logits"],
                    pair_targets["edit"], pair_valid, gamma=edit_gamma
                )
                pair_type_loss, _ = _slr_ucdrt_r2_type_loss(
                    aux[prefix + "_slr_ucdrt_r2_paired_type_logits"],
                    pair_targets["type"], pair_targets["edit"], pair_valid
                )
                pair_param_loss, _, _, _ = _slr_ucdrt_r2_parameter_loss(
                    aux[prefix + "_slr_ucdrt_r2_paired_move_bin_logits"],
                    aux[prefix + "_slr_ucdrt_r2_paired_add_dose"],
                    aux[prefix + "_slr_ucdrt_r2_paired_remove_dose"],
                    pair_targets,
                )
                pair_edit_prob = torch.sigmoid(aux[prefix + "_slr_ucdrt_r2_paired_edit_logits"])
                pair_exec, _, _ = _slr_ucdrt_r2_actor_exec_loss(
                    aux[prefix + "_slr_ucdrt_r2_paired_hard_candidate_probs"],
                    aux[prefix + "_slr_ucdrt_r2_paired_soft_candidate_probs"],
                    pair_anchor, pair_target, pair_valid, pair_edit_prob,
                )
                slr_ucdrt_r2_paired_exec_loss = pair_exec
                slr_ucdrt_paired_loss = (
                    0.20 * pair_edit_loss + 0.20 * pair_type_loss
                    + 0.20 * pair_param_loss + 0.40 * pair_exec
                )
                pair_err = (
                    (pair_anchor.detach() >= 0.5) != (pair_target.detach() >= 0.5)
                ) & (pair_valid.detach() > 0.5)
                # Probability-dose ADD/REMOVE can cross every binary FN/FP; MOVE
                # targets are assigned only when the real dictionary improves.
                slr_ucdrt_r2_paired_reachability = torch.where(
                    pair_err.sum() > 0, pair_err.new_tensor(1.0, dtype=final_prob.dtype),
                    pair_err.new_tensor(1.0, dtype=final_prob.dtype)
                )

        elif slr_ucdrt:
            boundary_radius = float(_cfg_get(m1, "GEOTR_SLR_UCDRT_BOUNDARY_RADIUS_PX", 5))
            max_disp = float(_cfg_get(m1, "GEOTR_SLR_UCDRT_MAX_BOUNDARY_DISPLACEMENT_PX", 4.0))
            max_step = float(_cfg_get(m1, "GEOTR_SLR_UCDRT_MAX_INTERIOR_LOGIT_STEP", 4.0))
            tgt_margin = float(_cfg_get(m1, "GEOTR_SLR_UCDRT_TARGET_MARGIN", 0.05))

            def _transition_pack(tag, anchor_patch, local_target, anchor_sdf_patch, target_sdf_patch, valid_patch):
                nonlocal slr_ucdrt_state_present
                logits = aux[prefix + f"_slr_ucdrt_{tag}state_logits"] if tag else aux[prefix + "_slr_ucdrt_state_logits"]
                move_px = aux[prefix + f"_slr_ucdrt_{tag}move_px"] if tag else aux[prefix + "_slr_ucdrt_move_px"]
                mag = aux[prefix + f"_slr_ucdrt_{tag}interior_magnitude"] if tag else aux[prefix + "_slr_ucdrt_interior_magnitude"]
                st, masks, move_t, mag_t = _slr_ucdrt_patch_targets(
                    anchor_patch, local_target, anchor_sdf_patch, target_sdf_patch, valid_patch,
                    boundary_radius_px=boundary_radius,
                    max_boundary_displacement_px=max_disp,
                    max_interior_logit_step=max_step,
                    target_margin=tgt_margin,
                )
                st_loss, present = _slr_ucdrt_state_loss(logits, st, valid_patch)
                act_loss, mv_loss, in_loss = _slr_ucdrt_action_loss(
                    move_px, mag, move_t, mag_t, masks
                )
                return st_loss, present, act_loss, mv_loss, in_loss, st, masks

            # Factual deployed proposal set.
            pred_anchor = aux[prefix + "_slr_pred_patch_anchor_prob"]
            pred_anchor_sdf = aux[prefix + "_slr_pred_sdf_anchor"]
            pred_tgt_sdf = aux[prefix + "_slr_pred_sdf_target"]
            (st_loss, present, act_loss, mv_loss, in_loss, st_target, st_masks) = _transition_pack(
                "", pred_anchor, pt, pred_anchor_sdf, pred_tgt_sdf, pv
            )
            slr_ucdrt_state_loss = st_loss
            slr_ucdrt_state_present = present
            slr_ucdrt_action_loss = act_loss
            slr_ucdrt_move_loss = mv_loss
            slr_ucdrt_interior_loss = in_loss
            slr_ucdrt_state_metrics = _slr_ucdrt_state_metrics(
                aux[prefix + "_slr_ucdrt_state_logits"], st_target, pv
            )

            # Training-only selector-positive and clean patches contribute to the
            # explicit state/action task; clean patches are legitimate KEEP labels.
            extra_state_terms = []
            extra_action_terms = []
            if prefix + "_slr_ucdrt_positive_state_logits" in aux:
                pos_anchor = aux[prefix + "_slr_ucdrt_positive_anchor_prob"]
                pos_anchor_sdf = aux[prefix + "_slr_ucdrt_positive_anchor_sdf"]
                ps = _transition_pack(
                    "positive_", pos_anchor, pot, pos_anchor_sdf,
                    aux[prefix + "_slr_positive_sdf_target"], pov
                )
                extra_state_terms.append(ps[0]); extra_action_terms.append(ps[2])
            if prefix + "_slr_ucdrt_clean_state_logits" in aux:
                cs = _transition_pack(
                    "clean_", aux[prefix + "_slr_clean_patch_anchor_prob"], ct,
                    aux[prefix + "_slr_ucdrt_clean_anchor_sdf"],
                    aux[prefix + "_slr_clean_sdf_target"], cv
                )
                extra_state_terms.append(cs[0]); extra_action_terms.append(cs[2])
            if extra_state_terms:
                slr_ucdrt_state_loss = torch.stack([slr_ucdrt_state_loss] + extra_state_terms).mean()
            if extra_action_terms:
                slr_ucdrt_action_loss = torch.stack([slr_ucdrt_action_loss] + extra_action_terms).mean()

            # VALUE is supervised by the *current executable operator outcome* on
            # the same factual candidate, not by residual density.  This target can
            # be negative and therefore teaches the candidate to lose against KEEP=0.
            soft_candidate = aux[prefix + "_slr_ucdrt_candidate_soft_probs"]
            util_value = aux[prefix + "_slr_ucdrt_utility_value"]
            util_commit = aux[prefix + "_slr_ucdrt_commit"]
            anchor_d = _slr_ucdrt_quality_per_candidate(pred_anchor, pt, pv)
            cand_d = _slr_ucdrt_quality_per_candidate(soft_candidate, pt, pv)
            util_target = (cand_d - anchor_d).detach().clamp(-1.0, 1.0)
            center_valid = (pv.sum(dim=(2,3,4)) > 0).to(util_value)
            util_elem = F.smooth_l1_loss(util_value, util_target.to(util_value), reduction="none", beta=0.02)
            slr_ucdrt_utility_loss = (util_elem * center_valid).sum() / center_valid.sum().clamp_min(1.0)
            slr_ucdrt_utility_mae = ((util_value.detach() - util_target).abs() * center_valid).sum() / center_valid.sum().clamp_min(1.0)
            uv = util_value.detach()[center_valid > 0]
            ut = util_target[center_valid > 0]
            if uv.numel() >= 2:
                uv0 = uv - uv.mean(); ut0 = ut - ut.mean()
                slr_ucdrt_utility_corr = (uv0 * ut0).sum() / torch.sqrt((uv0.square().sum() * ut0.square().sum()).clamp_min(EPS))
            slr_ucdrt_commit_rate = (util_commit.detach() * center_valid).sum() / center_valid.sum().clamp_min(1.0)
            positive_gain = util_target > 0
            committed = util_commit.detach() > 0.5
            slr_ucdrt_commit_precision = _safe_ratio(
                (positive_gain & committed & (center_valid > 0)).float().sum(),
                (committed & (center_valid > 0)).float().sum(),
            )

            # Paired stable residual branch.  Its anchor is explicitly supplied by
            # the forward, so input corruption and correction target are paired.
            if prefix + "_slr_ucdrt_paired_state_logits" in aux:
                pair_valid = aux[prefix + "_slr_ucdrt_paired_valid"]
                pair_pack = _transition_pack(
                    "paired_", aux[prefix + "_slr_ucdrt_paired_anchor_prob"],
                    aux[prefix + "_slr_ucdrt_paired_target"],
                    aux[prefix + "_slr_ucdrt_paired_anchor_sdf"],
                    aux[prefix + "_slr_ucdrt_paired_target_sdf"], pair_valid
                )
                slr_ucdrt_paired_loss = 0.5 * pair_pack[0] + 0.5 * pair_pack[2]

        # Historical aliases now mean positive-sample diagnostics only.
        slr_oracle_seg_loss = slr_positive_seg_loss
        slr_oracle_bce = slr_positive_bce
        slr_oracle_dice = slr_positive_dice
        slr_oracle_sdf_loss = slr_positive_sdf_loss

    if aefr_enabled:
        if aefr_stage == "single_atomic":
            # E1 isolates action/evidence factorization while retaining the old
            # hard deployment.  Keep the same canonical+Final causal control.
            canonical_w = 0.5
            deploy_w = 0.5
            stage2_loss = 0.5 * canonical_loss + 0.5 * final_loss
        elif aefr_stage in {"single_continuous", "hybrid_geometry"}:
            # E2/E3: train == deploy inside ROI.  Outside ROI Final is exact
            # anchor identity and is intentionally excluded from Stage-2 loss.
            canonical_w = 0.0
            deploy_w = 1.0
            stage2_loss = deployed_roi_loss
        elif aefr_stage in {"typed_state_taylor", "typed_state_exact"}:
            # Historical R1/R2 controls.
            canonical_w = 0.0
            deploy_w = 0.5
            stage2_loss = 0.5 * state_loss + 0.5 * deployed_roi_loss
        elif aefr_stage == "soft_ownership_exact":
            # Historical SRO-Exact ablation.
            canonical_w = 0.0
            deploy_w = 0.5
            stage2_loss = 0.5 * ownership_loss + 0.5 * deployed_roi_loss
        elif aefr_stage == "intervention_factorized":
            # Historical IFR control.
            w_loc = max(float(_cfg_get(m1, "GEOTR_AEFR_INTERVENTION_LOCALIZER_WEIGHT", 0.30)), 0.0)
            w_edit = max(float(_cfg_get(m1, "GEOTR_AEFR_INTERVENTION_EDIT_WEIGHT", 0.25)), 0.0)
            w_dir = max(float(_cfg_get(m1, "GEOTR_AEFR_INTERVENTION_DIRECTION_WEIGHT", 0.15)), 0.0)
            w_deploy = max(float(_cfg_get(m1, "GEOTR_AEFR_INTERVENTION_DEPLOY_WEIGHT", 0.30)), 0.0)
            denom = max(w_loc + w_edit + w_dir + w_deploy, 1.0e-8)
            canonical_w = 0.0
            deploy_w = w_deploy / denom
            stage2_loss = (
                w_loc * intervention_error_loss
                + w_edit * intervention_edit_loss
                + w_dir * intervention_direction_loss
                + w_deploy * deployed_roi_loss
            ) / denom
        elif aefr_stage == "selective_minimal_intervention":
            # Historical SMI control.
            w_loc = max(float(_cfg_get(m1, "GEOTR_AEFR_SMI_LOCALIZER_WEIGHT", 0.35)), 0.0)
            w_dir = max(float(_cfg_get(m1, "GEOTR_AEFR_SMI_DIRECTION_WEIGHT", 0.20)), 0.0)
            w_mag = max(float(_cfg_get(m1, "GEOTR_AEFR_SMI_BOUNDARY_MAG_WEIGHT", 0.20)), 0.0)
            w_deploy = max(float(_cfg_get(m1, "GEOTR_AEFR_SMI_DEPLOY_WEIGHT", 0.25)), 0.0)
            denom = max(w_loc + w_dir + w_mag + w_deploy, 1.0e-8)
            canonical_w = 0.0
            deploy_w = w_deploy / denom
            stage2_loss = (
                w_loc * intervention_error_loss
                + w_dir * intervention_direction_loss
                + w_mag * smi_boundary_magnitude_loss
                + w_deploy * smi_selective_deploy_loss
            ) / denom
        elif aefr_stage == "sparse_local_rerendering":
            canonical_w = 0.0
            slr_oracle_weight_effective = final_prob.new_zeros(())
            if slr_ucdrt_r2:
                # R2 actor-before-policy objective.  Ungated executable actor and
                # exact marginal critic are first-class objectives; deploy remains
                # the final system objective but can no longer starve the actor.
                w_selector = max(float(_cfg_get(m1, "GEOTR_SLR_UCDRT_R2_SELECTOR_WEIGHT", 0.10)), 0.0)
                w_edit = max(float(_cfg_get(m1, "GEOTR_SLR_UCDRT_R2_EDIT_WEIGHT", 0.15)), 0.0)
                w_type_param = max(float(_cfg_get(m1, "GEOTR_SLR_UCDRT_R2_TYPE_PARAM_WEIGHT", 0.10)), 0.0)
                w_actor = max(float(_cfg_get(m1, "GEOTR_SLR_UCDRT_R2_ACTOR_WEIGHT", 0.25)), 0.0)
                w_utility = max(float(_cfg_get(m1, "GEOTR_SLR_UCDRT_R2_UTILITY_WEIGHT", 0.15)), 0.0)
                w_final = max(float(_cfg_get(m1, "GEOTR_SLR_UCDRT_R2_FINAL_WEIGHT", 0.20)), 0.0)
                w_paired = max(float(_cfg_get(m1, "GEOTR_SLR_UCDRT_R2_PAIRED_WEIGHT", 0.05)), 0.0)
                denom = max(w_selector + w_edit + w_type_param + w_actor + w_utility + w_final + w_paired, 1.0e-8)
                slr_deploy_weight_effective = final_prob.new_tensor(w_final / denom)
                deploy_w = w_final / denom
                stage2_loss = (
                    w_selector * slr_selector_loss
                    + w_edit * slr_ucdrt_r2_edit_loss
                    + w_type_param * (0.5 * slr_ucdrt_r2_type_loss + 0.5 * slr_ucdrt_r2_param_loss)
                    + w_actor * slr_ucdrt_r2_actor_loss
                    + w_utility * slr_ucdrt_r2_critic_loss
                    + w_final * slr_deploy_loss
                    + w_paired * slr_ucdrt_paired_loss
                ) / denom
            elif slr_ucdrt:
                # Historical UCDRT-R1 control retained unchanged.
                w_selector = max(float(_cfg_get(m1, "GEOTR_SLR_UCDRT_SELECTOR_WEIGHT", 0.05)), 0.0)
                w_state = max(float(_cfg_get(m1, "GEOTR_SLR_UCDRT_STATE_WEIGHT", 0.15)), 0.0)
                w_utility = max(float(_cfg_get(m1, "GEOTR_SLR_UCDRT_UTILITY_WEIGHT", 0.15)), 0.0)
                w_action = max(float(_cfg_get(m1, "GEOTR_SLR_UCDRT_ACTION_WEIGHT", 0.15)), 0.0)
                w_final = max(float(_cfg_get(m1, "GEOTR_SLR_UCDRT_FINAL_WEIGHT", 0.45)), 0.0)
                w_paired = max(float(_cfg_get(m1, "GEOTR_SLR_UCDRT_PAIRED_WEIGHT", 0.05)), 0.0)
                denom = max(w_selector + w_state + w_utility + w_action + w_final + w_paired, 1.0e-8)
                slr_deploy_weight_effective = final_prob.new_tensor(w_final / denom)
                deploy_w = w_final / denom
                stage2_loss = (
                    w_selector * slr_selector_loss
                    + w_state * slr_ucdrt_state_loss
                    + w_utility * slr_ucdrt_utility_loss
                    + w_action * slr_ucdrt_action_loss
                    + w_final * slr_deploy_loss
                    + w_paired * slr_ucdrt_paired_loss
                ) / denom
            else:
                # SLR2.4/OCRA: WHERE + typed/dose actor + exact deployment,
                # with a decoupled low-weight absolute-SDF diagnostic head.
                w_selector = max(float(_cfg_get(m1, "GEOTR_SLR_SELECTOR_WEIGHT", 0.15)), 0.0)
                w_patch = max(float(_cfg_get(m1, "GEOTR_SLR_PATCH_WEIGHT", 0.35)), 0.0)
                w_final = max(float(_cfg_get(m1, "GEOTR_SLR_FINAL_WEIGHT", 0.40)), 0.0)
                w_sdf = max(float(_cfg_get(m1, "GEOTR_SLR_SDF_WEIGHT", 0.10)), 0.0)
                slr_deploy_weight_effective = final_prob.new_tensor(w_final)
                denom = max(w_selector + w_patch + w_final + w_sdf, 1.0e-8)
                deploy_w = w_final / denom
                stage2_loss = (
                    w_selector * slr_selector_loss
                    + w_patch * slr_patch_loss
                    + w_final * slr_deploy_loss
                    + w_sdf * slr_sdf_loss
                ) / denom
        else:
            raise ValueError(f"Unknown GEOTR_AEFR_STAGE={aefr_stage!r}")
    elif pc2r_v32:
        # PC2R-v3.2 Operator-Aligned WHAT.  The three posterior-conditioned
        # view losses remain visible as diagnostics but no longer dominate the
        # training objective.  We optimize the factual canonical proposal and
        # the exact deployed Final produced by the deterministic atomic commit.
        # commit_mask itself is detached/non-differentiable, but gradients flow
        # through canonical probabilities on the actually committed support.
        canonical_w = max(float(_cfg_get(m1, "GEOTR_PC2R_OPERATOR_CANONICAL_WEIGHT", 0.5)), 0.0)
        deploy_w = max(float(_cfg_get(m1, "GEOTR_PC2R_OPERATOR_FINAL_WEIGHT", 0.5)), 0.0)
        denom = max(canonical_w + deploy_w, 1.0e-8)
        stage2_loss = (canonical_w * canonical_loss + deploy_w * final_loss) / denom
    elif c2r_v2:
        # Canonical C2R-v2 deliberately does not train through the hard atomic
        # commit.  The commit is a non-differentiable deployment certificate.
        # Instead, all CF views and their canonical mean are directly trained
        # against the same GT ROI from step zero.
        canonical_w = max(float(_cfg_get(m1, "GEOTR_C2R_CANONICAL_SEG_WEIGHT", 0.5)), 0.0)
        denom = max(view_w + canonical_w, 1.0e-8)
        stage2_loss = (view_w * view_loss + canonical_w * canonical_loss) / denom
        deploy_w = 0.0
    else:
        final_w = max(float(_cfg_get(m1, "GEOTR_C2R_FINAL_SEG_WEIGHT", 0.25)), 0.0)
        denom = max(view_w + final_w, 1.0e-8)
        stage2_loss = (view_w * view_loss + final_w * final_loss) / denom
        canonical_w = 0.0
        deploy_w = final_w

    if mode == "base":
        objective = final_loss * 0.0
    elif mode == "geometry":
        objective = geo_loss
    elif mode == "residual":
        objective = stage2_loss
    else:
        objective = geo_loss + stage2_loss

    smooth = _flow_smoothness(flow)
    smooth_w = float(_cfg_get(m1, "GEOTOPO_SMOOTHNESS_WEIGHT", 0.0))
    if mode in {"geometry", "full"} and smooth_w > 0.0:
        objective = objective + smooth_w * smooth
    compat = aux.get("mhcs_m1_distribution_log_var")
    if isinstance(compat, torch.Tensor):
        objective = objective + 0.0 * compat

    base_dice = _dice_per_case(base_prob[:, 0].detach(), target)
    geo_dice = _dice_per_case(geo_prob[:, 0].detach(), target)
    final_dice = _dice_per_case(final_prob[:, 0].detach(), target)
    anchor_dice = _dice_per_case(anchor[:, 0], target)
    stage_gain_pc = final_dice - anchor_dice

    region_b = region[:, 0].detach() > 0.5
    center_b = centers[:, 0].detach() > 0.5
    consensus_b = consensus[:, 0].detach() > 0.5

    edit_b = edit[:, 0].detach() > 0.5
    gt = target >= 0.5
    anchor_h = anchor[:, 0] >= 0.5
    final_h = final_prob[:, 0].detach() >= 0.5
    hard_err = anchor_h != gt

    # PC2R-v3.2 reachability audit.  With bounded tanh residuals, a wrong hard
    # label is actionable only when the factual anchor logit can cross zero
    # within the configured residual range.  This is a diagnostic; it does not
    # alter deployment.
    anchor_logit_map = torch.logit(anchor[:, 0].clamp(EPS, 1.0 - EPS)).detach()
    residual_scale = float(_cfg_get(m1, "GEOTR_AEFR_SINGLE_RESIDUAL_LOGIT_SCALE", _cfg_get(m1, "GEOTR_PC2R_RESIDUAL_LOGIT_SCALE", 2.0)))
    if aefr_enabled and aefr_stage in {"hybrid_geometry", "typed_state_taylor", "typed_state_exact", "soft_ownership_exact", "intervention_factorized", "selective_minimal_intervention", "sparse_local_rerendering"}:
        # E3 interior action is anchor-adaptive and has no fixed-confidence
        # crossing ceiling; mark the deterministic ROI error support as
        # representationally reachable for this diagnostic.
        reachable_err = region_b & hard_err
        unreachable_err = torch.zeros_like(reachable_err)
    else:
        reachable_err = region_b & hard_err & (anchor_logit_map.abs() < max(residual_scale, 1.0e-6))
        unreachable_err = region_b & hard_err & (~reachable_err)
    fn_err = region_b & (~anchor_h) & gt
    fp_err = region_b & anchor_h & (~gt)
    unreachable_fn = fn_err & (~reachable_err)
    unreachable_fp = fp_err & (~reachable_err)

    # Post-anchor residual geometry audit.  Distance is approximated by exact
    # binary morphological neighborhoods of the anchor boundary at model
    # resolution.  Cumulative <=1/2/3/5px fractions tell us whether the
    # remaining error mass is predominantly boundary-local or interior.
    ah = anchor_h.float()[:, None]
    dil = F.max_pool2d(ah, kernel_size=3, stride=1, padding=1)
    ero = -F.max_pool2d(-ah, kernel_size=3, stride=1, padding=1)
    anchor_boundary = (dil != ero)[:, 0]
    def _near_boundary(radius: int):
        if radius <= 0:
            return anchor_boundary
        return F.max_pool2d(anchor_boundary.float()[:, None], kernel_size=2*radius+1, stride=1, padding=radius)[:, 0] > 0.5
    near1 = _near_boundary(1)
    near2 = _near_boundary(2)
    near3 = _near_boundary(3)
    near5 = _near_boundary(5)

    corrected = edit_b & hard_err & (final_h == gt)
    introduced = edit_b & (~hard_err) & (final_h != gt)
    corrected_count = corrected.float().sum()
    introduced_count = introduced.float().sum()
    net = corrected_count - introduced_count
    total_err = hard_err.float().sum()
    region_err = (region_b & hard_err).float().sum()
    region_count = region_b.float().sum()
    edit_count = edit_b.float().sum()
    consensus_count = consensus_b.float().sum()

    region_oracle = anchor.clone()
    region_oracle[:, 0] = torch.where(region_b & hard_err, target, anchor[:, 0])
    region_oracle_gain = (
        _dice_per_case(region_oracle[:, 0], target) - anchor_dice
    ).mean()
    reachable_oracle = anchor.clone()
    reachable_oracle[:, 0] = torch.where(reachable_err, target, anchor[:, 0])
    reachable_oracle_gain = (_dice_per_case(reachable_oracle[:, 0], target) - anchor_dice).mean()
    boundary5_oracle = anchor.clone()
    boundary5_oracle[:, 0] = torch.where(region_b & hard_err & near5, target, anchor[:, 0])
    boundary5_oracle_gain = (_dice_per_case(boundary5_oracle[:, 0], target) - anchor_dice).mean()
    interior_oracle = anchor.clone()
    interior_oracle[:, 0] = torch.where(region_b & hard_err & (~near5), target, anchor[:, 0])
    interior_oracle_gain = (_dice_per_case(interior_oracle[:, 0], target) - anchor_dice).mean()
    all_error_oracle = anchor.clone()
    all_error_oracle[:, 0] = torch.where(hard_err, target, anchor[:, 0])
    all_error_oracle_gain = (
        _dice_per_case(all_error_oracle[:, 0], target) - anchor_dice
    ).mean()

    selection_stats = _selection_stats(region, anchor, target)
    point_before = _point_accuracy(anchor, target, region)
    point_after = _point_accuracy(final_prob, target, region)
    margin_ap = _average_precision_binary(margin[:, 0], hard_err)
    std_ap = _average_precision_binary(mc_std[:, 0], hard_err)
    dis_ap = _average_precision_binary(mc_dis[:, 0], hard_err)
    entropy_ap = _average_precision_binary(entropy[:, 0], hard_err)

    # Exact BCE gradient magnitude wrt canonical logits is |p-y| (up to the
    # global BCE normalization/weight).  Report error-vs-correct mass so we can
    # audit whether easy correct ROI pixels actually dominate the BCE signal
    # instead of inferring that from pixel counts alone.
    bce_grad_mag = (mean_prob[:, 0].detach() - target).abs()
    error_grad_mass = (bce_grad_mag * (region_b & hard_err).to(bce_grad_mag)).sum()
    correct_grad_mass = (bce_grad_mag * (region_b & (~hard_err)).to(bce_grad_mag)).sum()
    candidate_b = candidate_mask[:, 0].detach() > 0.5
    commit_b = commit_mask[:, 0].detach() > 0.5
    candidate_count = candidate_b.float().sum()
    commit_count = commit_b.float().sum()

    recon_base = aux["geotopo_residual_only_probs"]
    recon_geo = aux["geotopo_reconstruction_after_geometry_probs"]
    recon_base_dice = _dice_per_case(recon_base[:, 0].detach(), target)
    recon_geo_dice = _dice_per_case(recon_geo[:, 0].detach(), target)

    diagnostics = {
        "mhcs_objective": objective.detach(),
        "mhcs_m1_objective": objective.detach(),
        "mhcs_m2_objective": stage2_loss.detach() if mode in {"residual", "full"} else objective.detach() * 0.0,
        "mhcs_final_gain": (final_dice - base_dice).mean().detach(),
        "mhcs_base_dice": base_dice.mean().detach(),
        "mhcs_final_dice": final_dice.mean().detach(),
        "mhcs_candidate_mean_gain": (final_dice - base_dice).mean().detach(),
        "mhcs_candidate_harm_rate": (final_dice < base_dice).float().mean().detach(),
        "geotopo_final_loss": final_loss.detach(),
        "geotopo_geometry_loss": geo_loss.detach(),
        "geotopo_bce_loss": final_bce.detach(),
        "geotopo_dice_loss": final_dice_loss.detach(),
        "geotopo_geometry_bce_loss": geo_bce.detach(),
        "geotopo_geometry_dice_loss": geo_dice_loss.detach(),
        "geotopo_base_dice": base_dice.mean().detach(),
        "geotopo_geometry_dice": geo_dice.mean().detach(),
        "geotopo_reconstruction_base_dice": recon_base_dice.mean().detach(),
        "geotopo_reconstruction_after_geometry_dice": recon_geo_dice.mean().detach(),
        "geotopo_final_dice": final_dice.mean().detach(),
        "geotopo_geometry_gain": (geo_dice - base_dice).mean().detach(),
        "geotopo_reconstruction_gain_without_geometry": (recon_base_dice - base_dice).mean().detach(),
        "geotopo_reconstruction_gain_after_geometry": (recon_geo_dice - geo_dice).mean().detach(),
        "geotopo_final_gain": (final_dice - base_dice).mean().detach(),
        "geotr_v4g_enabled": final_prob.new_tensor(1.0).detach(),
        "geotr_v4g_r2_enabled": final_prob.new_tensor(0.0).detach(),
        "geotr_v4g_factual_point_loss": view_loss.detach(),
        "geotr_v4g_correction_loss": view_loss.detach(),
        "geotr_v4g_preserve_loss": final_prob.new_zeros(()),
        "geotr_v4g_denoise_point_loss": final_prob.new_zeros(()),
        "geotr_v4g_denoise_correction_loss": final_prob.new_zeros(()),
        "geotr_v4g_denoise_preserve_loss": final_prob.new_zeros(()),
        "geotr_v4g_selection_coverage": selection_stats["coverage"].detach(),
        "geotr_v4g_selection_precision": selection_stats["precision"].detach(),
        "geotr_v4g_selection_recall": selection_stats["recall"].detach(),
        "geotr_v4g_anchor_error_rate": selection_stats["error_rate"].detach(),
        "geotr_v4g_point_accuracy_before": point_before.detach(),
        "geotr_v4g_point_accuracy_after": point_after.detach(),
        "geotr_v4g_point_accuracy_gain": (point_after - point_before).detach(),
        "geotr_v4g_selected_abs_change": _masked_mean((final_prob[:,0].detach()-anchor[:,0]).abs(), region_b).detach(),
        "geotr_v4g_denoise_corruption_rate": final_prob.new_zeros(()),
        "geotr_v4g_denoise_selected_rate": final_prob.new_zeros(()),
        "geotr_v4g_margin_error_ap": margin_ap.detach(),
        "geotr_v4g_mc_std_error_ap": std_ap.detach(),
        "geotr_v4g_mc_disagreement_error_ap": dis_ap.detach(),
        "geotr_v4g_entropy_error_ap": entropy_ap.detach(),
        "geotr_v4g_margin_recall_at_05": _topk_recall(margin[:,0], hard_err, 0.05).detach(),
        "geotr_v4g_margin_recall_at_10": _topk_recall(margin[:,0], hard_err, 0.10).detach(),
        "geotr_v4g_mc_std_recall_at_05": _topk_recall(mc_std[:,0], hard_err, 0.05).detach(),
        "geotr_v4g_mc_std_recall_at_10": _topk_recall(mc_std[:,0], hard_err, 0.10).detach(),
        "geotr_v4g_mc_disagreement_recall_at_05": _topk_recall(mc_dis[:,0], hard_err, 0.05).detach(),
        "geotr_v4g_mc_disagreement_recall_at_10": _topk_recall(mc_dis[:,0], hard_err, 0.10).detach(),
        "geotr_v4g_corrected_error_count": corrected_count.detach(),
        "geotr_v4g_introduced_error_count": introduced_count.detach(),
        "geotr_v4g_net_correction_count": net.detach(),
        "geotr_v4g_correction_recall": _safe_ratio(corrected_count, total_err).detach(),
        "geotr_v4g_introduction_rate": _safe_ratio(introduced_count, region_count).detach(),
        "geotr_v4g_pred_selector_oracle_gain": region_oracle_gain.detach(),
        "geotr_v4g_full_residual_oracle_gain": all_error_oracle_gain.detach(),
        "geotr_v4g_stage2_benefit_rate": (stage_gain_pc > 1.0e-8).float().mean().detach(),
        "geotr_v4g_stage2_harm_rate": (stage_gain_pc < -1.0e-8).float().mean().detach(),
        "geotr_v4g_stage2_soft_gain": stage_gain_pc.mean().detach(),
        "geotr_v4g_r3_enabled": final_prob.new_zeros(()),
        "geotr_v4g_r4_enabled": final_prob.new_zeros(()),
        "geotr_v4g_r4_flip_loss": final_prob.new_zeros(()),
        "geotr_c2r_enabled": final_prob.new_tensor(1.0).detach(),
        "geotr_c2r_view_loss": view_loss.detach(),
        "geotr_c2r_view_bce": view_bce.detach(),
        "geotr_c2r_view_dice_loss": view_dice_loss.detach(),
        "geotr_c2r_final_loss": final_loss.detach(),
        "geotr_c2r_canonical_loss": canonical_loss.detach(),
        "geotr_c2r_canonical_bce": canonical_bce.detach(),
        "geotr_c2r_canonical_dice_loss": canonical_dice_loss.detach(),
        "geotr_c2r_canonical_seg_weight": final_prob.new_tensor(canonical_w).detach(),
        "geotr_pc2r_v32_operator_aligned": final_prob.new_tensor(1.0 if pc2r_v32 else 0.0).detach(),
        "geotr_pc2r_v32_operator_canonical_weight": final_prob.new_tensor(canonical_w if pc2r_v32 else 0.0).detach(),
        "geotr_pc2r_v32_operator_final_weight": final_prob.new_tensor(deploy_w if pc2r_v32 else 0.0).detach(),
        "geotr_pc2r_v32_view_loss_diagnostic_only": final_prob.new_tensor(1.0 if pc2r_v32 else 0.0).detach(),
        "geotr_aefr_enabled": final_prob.new_tensor(1.0 if aefr_enabled else 0.0).detach(),
        "geotr_aefr_stage_id": final_prob.new_tensor({"single_atomic":1.0,"single_continuous":2.0,"hybrid_geometry":3.0,"typed_state_taylor":4.0,"typed_state_exact":5.0,"soft_ownership_exact":6.0,"intervention_factorized":7.0,"selective_minimal_intervention":8.0,"sparse_local_rerendering":9.0}.get(aefr_stage,0.0) if aefr_enabled else 0.0).detach(),
        "geotr_aefr_deployed_roi_loss": deployed_roi_loss.detach(),
        "geotr_aefr_deployed_roi_bce": deployed_roi_bce.detach(),
        "geotr_aefr_deployed_roi_dice_loss": deployed_roi_dice_loss.detach(),
        "geotr_slr_selector_loss": slr_selector_loss.detach(),
        "geotr_slr_selector_mae": slr_selector_mae.detach(),
        "geotr_slr_pred_patch_loss": slr_pred_patch_loss.detach(),
        "geotr_slr_pred_patch_bce": slr_pred_patch_bce.detach(),
        "geotr_slr_pred_patch_dice_loss": slr_pred_patch_dice.detach(),
        "geotr_slr_positive_seg_loss": slr_positive_seg_loss.detach(),
        "geotr_slr_positive_bce": slr_positive_bce.detach(),
        "geotr_slr_positive_dice_loss": slr_positive_dice.detach(),
        "geotr_slr_clean_seg_loss": slr_clean_seg_loss.detach(),
        "geotr_slr_clean_bce": slr_clean_bce.detach(),
        "geotr_slr_clean_dice_loss": slr_clean_dice.detach(),
        "geotr_slr_patch_loss": slr_patch_loss.detach(),
        "geotr_slr_patch_seg_diagnostic": slr_patch_seg_diagnostic.detach(),
        "geotr_slr_actor_type_loss": slr_actor_type_loss.detach(),
        "geotr_slr_actor_dose_loss": slr_actor_dose_loss.detach(),
        "geotr_slr_actor_keep_loss": slr_actor_keep_loss.detach(),
        "geotr_slr24_residual_actor_active": aux.get(prefix + "_slr_pred_signed_action", torch.tensor(0.0, device=final_prob.device)).float().abs().mean().detach(),
        "geotr_slr_patch_bce": slr_patch_bce.detach(),
        "geotr_slr_patch_dice_loss": slr_patch_dice.detach(),
        "geotr_slr_clean_residual_abs": slr_clean_residual_abs.detach(),
        "geotr_slr_clean_target_residual_abs": slr_clean_target_residual_abs.detach(),
        "geotr_slr_oracle_seg_loss": slr_oracle_seg_loss.detach(),
        "geotr_slr_oracle_bce": slr_oracle_bce.detach(),
        "geotr_slr_oracle_dice_loss": slr_oracle_dice.detach(),
        "geotr_slr_sdf_loss": slr_sdf_loss.detach(),
        "geotr_slr_pred_sdf_loss": slr_pred_sdf_loss.detach(),
        "geotr_slr_positive_sdf_loss": slr_positive_sdf_loss.detach(),
        "geotr_slr_clean_sdf_loss": slr_clean_sdf_loss.detach(),
        "geotr_slr_oracle_sdf_loss": slr_oracle_sdf_loss.detach(),
        "geotr_slr_curriculum_progress": slr_curriculum_progress.detach(),
        "geotr_slr_oracle_weight_effective": slr_oracle_weight_effective.detach(),
        "geotr_slr_deploy_weight_effective": slr_deploy_weight_effective.detach(),
        "geotr_slr_deploy_loss": slr_deploy_loss.detach(),
        "geotr_slr_deploy_bce": slr_deploy_bce.detach(),
        "geotr_slr_deploy_dice_loss": slr_deploy_dice.detach(),
        "geotr_slr_ucdrt_enabled": final_prob.new_tensor(1.0 if slr_ucdrt else 0.0).detach(),
        "geotr_slr_ucdrt_state_loss": slr_ucdrt_state_loss.detach(),
        "geotr_slr_ucdrt_state_present_class_count": slr_ucdrt_state_present.detach(),
        "geotr_slr_ucdrt_state_macro_f1": slr_ucdrt_state_metrics["macro_f1"].detach(),
        "geotr_slr_ucdrt_keep_f1": slr_ucdrt_state_metrics["keep_f1"].detach(),
        "geotr_slr_ucdrt_move_f1": slr_ucdrt_state_metrics["move_f1"].detach(),
        "geotr_slr_ucdrt_add_f1": slr_ucdrt_state_metrics["add_f1"].detach(),
        "geotr_slr_ucdrt_remove_f1": slr_ucdrt_state_metrics["remove_f1"].detach(),
        "geotr_slr_ucdrt_state_pred_edit_fraction": slr_ucdrt_state_metrics["pred_edit_fraction"].detach(),
        "geotr_slr_ucdrt_utility_loss": slr_ucdrt_utility_loss.detach(),
        "geotr_slr_ucdrt_utility_mae": slr_ucdrt_utility_mae.detach(),
        "geotr_slr_ucdrt_utility_corr": slr_ucdrt_utility_corr.detach(),
        "geotr_slr_ucdrt_commit_rate": slr_ucdrt_commit_rate.detach(),
        "geotr_slr_ucdrt_commit_precision": slr_ucdrt_commit_precision.detach(),
        "geotr_slr_ucdrt_deploy_support_fraction": slr_ucdrt_deploy_support_fraction.detach(),
        "geotr_slr_ucdrt_action_loss": slr_ucdrt_action_loss.detach(),
        "geotr_slr_ucdrt_move_loss": slr_ucdrt_move_loss.detach(),
        "geotr_slr_ucdrt_interior_loss": slr_ucdrt_interior_loss.detach(),
        "geotr_slr_ucdrt_paired_loss": slr_ucdrt_paired_loss.detach(),
        "geotr_slr_ucdrt_r2_enabled": final_prob.new_tensor(1.0 if slr_ucdrt_r2 else 0.0).detach(),
        "geotr_slr_ucdrt_r2_edit_loss": slr_ucdrt_r2_edit_loss.detach(),
        "geotr_slr_ucdrt_r2_type_loss": slr_ucdrt_r2_type_loss.detach(),
        "geotr_slr_ucdrt_r2_param_loss": slr_ucdrt_r2_param_loss.detach(),
        "geotr_slr_ucdrt_r2_actor_loss": slr_ucdrt_r2_actor_loss.detach(),
        "geotr_slr_ucdrt_r2_actor_hard_loss": slr_ucdrt_r2_actor_hard_loss.detach(),
        "geotr_slr_ucdrt_r2_actor_soft_loss": slr_ucdrt_r2_actor_soft_loss.detach(),
        "geotr_slr_ucdrt_r2_critic_loss": slr_ucdrt_r2_critic_loss.detach(),
        "geotr_slr_ucdrt_r2_paired_exec_loss": slr_ucdrt_r2_paired_exec_loss.detach(),
        "geotr_slr_ucdrt_r2_paired_reachability": slr_ucdrt_r2_paired_reachability.detach(),
        "geotr_slr_ucdrt_r2_edit_f1": slr_ucdrt_r2_factor_metrics["edit_f1"].detach(),
        "geotr_slr_ucdrt_r2_edit_precision": slr_ucdrt_r2_factor_metrics["edit_precision"].detach(),
        "geotr_slr_ucdrt_r2_edit_recall": slr_ucdrt_r2_factor_metrics["edit_recall"].detach(),
        "geotr_slr_ucdrt_r2_target_edit_fraction": slr_ucdrt_r2_factor_metrics["target_edit_fraction"].detach(),
        "geotr_slr_ucdrt_r2_pred_edit_fraction": slr_ucdrt_r2_factor_metrics["pred_edit_fraction"].detach(),
        "geotr_slr_ucdrt_r2_type_macro_f1": slr_ucdrt_r2_factor_metrics["type_macro_f1"].detach(),
        "geotr_slr_ucdrt_r2_move_f1": slr_ucdrt_r2_factor_metrics["move_f1"].detach(),
        "geotr_slr_ucdrt_r2_add_f1": slr_ucdrt_r2_factor_metrics["add_f1"].detach(),
        "geotr_slr_ucdrt_r2_remove_f1": slr_ucdrt_r2_factor_metrics["remove_f1"].detach(),
        "geotr_slr_ucdrt_r2_hard_mean_gain": slr_ucdrt_r2_hard_mean_gain.detach(),
        "geotr_slr_ucdrt_r2_hard_positive_rate": slr_ucdrt_r2_hard_positive_rate.detach(),
        "geotr_slr_ucdrt_r2_hard_oracle_gain": slr_ucdrt_r2_hard_oracle_gain.detach(),
        "geotr_slr_ucdrt_r2_soft_mean_gain": slr_ucdrt_r2_soft_mean_gain.detach(),
        "geotr_slr_ucdrt_r2_soft_positive_rate": slr_ucdrt_r2_soft_positive_rate.detach(),
        "geotr_slr_ucdrt_r2_soft_oracle_gain": slr_ucdrt_r2_soft_oracle_gain.detach(),
        "geotr_slr_ucdrt_r2_hard_soft_gap": slr_ucdrt_r2_hard_soft_gap.detach(),
        "geotr_slr_ucdrt_r2_critic_precision": slr_ucdrt_r2_critic_precision.detach(),
        "geotr_slr_ucdrt_r2_critic_recall": slr_ucdrt_r2_critic_recall.detach(),
        "geotr_slr_ucdrt_r2_critic_positive_rate": slr_ucdrt_r2_critic_positive_rate.detach(),
        "geotr_aefr_state_loss": state_loss.detach(),
        "geotr_aefr_state_present_class_count": state_present_classes.detach(),
        "geotr_aefr_state_macro_f1": state_metrics["macro_f1"].detach(),
        "geotr_aefr_state_pred_edit_fraction": state_metrics["pred_edit_fraction"].detach(),
        "geotr_aefr_state_keep_precision": state_metrics["keep_precision"].detach(),
        "geotr_aefr_state_keep_recall": state_metrics["keep_recall"].detach(),
        "geotr_aefr_state_keep_f1": state_metrics["keep_f1"].detach(),
        "geotr_aefr_state_boundary_add_precision": state_metrics["boundary_add_precision"].detach(),
        "geotr_aefr_state_boundary_add_recall": state_metrics["boundary_add_recall"].detach(),
        "geotr_aefr_state_boundary_add_f1": state_metrics["boundary_add_f1"].detach(),
        "geotr_aefr_state_boundary_remove_precision": state_metrics["boundary_remove_precision"].detach(),
        "geotr_aefr_state_boundary_remove_recall": state_metrics["boundary_remove_recall"].detach(),
        "geotr_aefr_state_boundary_remove_f1": state_metrics["boundary_remove_f1"].detach(),
        "geotr_aefr_state_interior_add_precision": state_metrics["interior_add_precision"].detach(),
        "geotr_aefr_state_interior_add_recall": state_metrics["interior_add_recall"].detach(),
        "geotr_aefr_state_interior_add_f1": state_metrics["interior_add_f1"].detach(),
        "geotr_aefr_state_interior_remove_precision": state_metrics["interior_remove_precision"].detach(),
        "geotr_aefr_state_interior_remove_recall": state_metrics["interior_remove_recall"].detach(),
        "geotr_aefr_state_interior_remove_f1": state_metrics["interior_remove_f1"].detach(),
        "geotr_aefr_state_keep_count": state_metrics["keep_count"].detach(),
        "geotr_aefr_state_boundary_add_count": state_metrics["boundary_add_count"].detach(),
        "geotr_aefr_state_boundary_remove_count": state_metrics["boundary_remove_count"].detach(),
        "geotr_aefr_state_interior_add_count": state_metrics["interior_add_count"].detach(),
        "geotr_aefr_state_interior_remove_count": state_metrics["interior_remove_count"].detach(),
        "geotr_aefr_ownership_loss": ownership_loss.detach(),
        "geotr_aefr_ownership_macro_f1": ownership_metrics["macro_f1"].detach(),
        "geotr_aefr_ownership_pred_edit_fraction": ownership_metrics["pred_edit_fraction"].detach(),
        "geotr_aefr_ownership_pred_edit_mass": ownership_metrics["pred_edit_mass"].detach(),
        "geotr_aefr_ownership_target_edit_mass": ownership_metrics["target_edit_mass"].detach(),
        "geotr_aefr_ownership_edit_mass_abs_error": ownership_metrics["edit_mass_abs_error"].detach(),
        "geotr_aefr_ownership_signed_action_mae": ownership_metrics["signed_action_mae"].detach(),
        "geotr_aefr_ownership_signed_target_abs_mean": ownership_metrics["signed_target_abs_mean"].detach(),
        "geotr_aefr_ownership_keep_precision": ownership_metrics["keep_precision"].detach(),
        "geotr_aefr_ownership_keep_recall": ownership_metrics["keep_recall"].detach(),
        "geotr_aefr_ownership_keep_f1": ownership_metrics["keep_f1"].detach(),
        "geotr_aefr_ownership_add_precision": ownership_metrics["add_precision"].detach(),
        "geotr_aefr_ownership_add_recall": ownership_metrics["add_recall"].detach(),
        "geotr_aefr_ownership_add_f1": ownership_metrics["add_f1"].detach(),
        "geotr_aefr_ownership_remove_precision": ownership_metrics["remove_precision"].detach(),
        "geotr_aefr_ownership_remove_recall": ownership_metrics["remove_recall"].detach(),
        "geotr_aefr_ownership_remove_f1": ownership_metrics["remove_f1"].detach(),
        "geotr_aefr_intervention_error_loss": intervention_error_loss.detach(),
        "geotr_aefr_intervention_error_bce": intervention_error_bce.detach(),
        "geotr_aefr_intervention_error_dice_loss": intervention_error_dice.detach(),
        "geotr_aefr_intervention_edit_loss": intervention_edit_loss.detach(),
        "geotr_aefr_intervention_direction_loss": intervention_direction_loss.detach(),
        "geotr_aefr_smi_boundary_magnitude_loss": smi_boundary_magnitude_loss.detach(),
        "geotr_aefr_smi_selective_deploy_loss": smi_selective_deploy_loss.detach(),
        "geotr_aefr_smi_selective_deploy_bce": smi_selective_deploy_bce.detach(),
        "geotr_aefr_smi_selective_deploy_dice_loss": smi_selective_deploy_dice.detach(),
        "geotr_aefr_smi_boundary_target_mean_px": smi_boundary_target_mean.detach(),
        "geotr_aefr_smi_commit_fraction": smi_commit_fraction.detach(),
        "geotr_aefr_intervention_localizer_precision": intervention_metrics["localizer_precision"].detach(),
        "geotr_aefr_intervention_localizer_recall": intervention_metrics["localizer_recall"].detach(),
        "geotr_aefr_intervention_localizer_f1": intervention_metrics["localizer_f1"].detach(),
        "geotr_aefr_intervention_edit_precision": intervention_metrics["edit_precision"].detach(),
        "geotr_aefr_intervention_edit_recall": intervention_metrics["edit_recall"].detach(),
        "geotr_aefr_intervention_edit_f1": intervention_metrics["edit_f1"].detach(),
        "geotr_aefr_intervention_direction_accuracy": intervention_metrics["direction_accuracy"].detach(),
        "geotr_aefr_intervention_add_precision": intervention_metrics["add_precision"].detach(),
        "geotr_aefr_intervention_add_recall": intervention_metrics["add_recall"].detach(),
        "geotr_aefr_intervention_add_f1": intervention_metrics["add_f1"].detach(),
        "geotr_aefr_intervention_remove_precision": intervention_metrics["remove_precision"].detach(),
        "geotr_aefr_intervention_remove_recall": intervention_metrics["remove_recall"].detach(),
        "geotr_aefr_intervention_remove_f1": intervention_metrics["remove_f1"].detach(),
        "geotr_aefr_intervention_pred_edit_fraction": intervention_metrics["pred_edit_fraction"].detach(),
        "geotr_aefr_intervention_target_edit_fraction": intervention_metrics["target_edit_fraction"].detach(),
        "geotr_aefr_intervention_signed_action_mae": intervention_metrics["signed_action_mae"].detach(),
        "geotr_aefr_intervention_zero_action_mae": intervention_metrics["zero_action_mae"].detach(),
        "geotr_aefr_intervention_signed_advantage": intervention_metrics["signed_advantage"].detach(),
        "geotr_aefr_intervention_error_prob_mean_error": intervention_metrics["error_prob_mean_error"].detach(),
        "geotr_aefr_intervention_error_prob_mean_correct": intervention_metrics["error_prob_mean_correct"].detach(),
        "geotr_aefr_intervention_edit_prob_mean_error_roi": intervention_metrics["edit_prob_mean_error_roi"].detach(),
        "geotr_aefr_intervention_edit_prob_mean_correct_roi": intervention_metrics["edit_prob_mean_correct_roi"].detach(),
        "geotr_aefr_state_mean_boundary_magnitude_px": _masked_mean(aefr_boundary_magnitude[:,0].detach(), region_b).detach(),
        "geotr_aefr_state_mean_abs_signed_boundary": _masked_mean(aefr_signed_boundary[:,0].detach().abs(), region_b).detach(),
        "geotr_aefr_state_mean_abs_signed_interior": _masked_mean(aefr_signed_interior[:,0].detach().abs(), region_b).detach(),
        "geotr_aefr_posterior_stability_support": aefr_stability.detach(),
        "geotr_aefr_posterior_stability_improvement": aefr_stability_improvement.detach(),
        "geotr_aefr_posterior_disagreement_pre": aefr_dis_pre.detach(),
        "geotr_aefr_posterior_disagreement_post": aefr_dis_post.detach(),
        "geotr_aefr_action_support_fraction": aefr_support_fraction.detach(),
        "geotr_aefr_posterior_diversity_all": aefr_posterior_div.detach(),
        "geotr_aefr_posterior_center_bias_abs": aefr_center_bias_abs.detach(),
        "geotr_aefr_posterior_center_bias_signed": aefr_center_bias_signed.detach(),
        "geotr_aefr_boundary_fraction_in_region": _safe_ratio((aefr_boundary_mask[:,0].detach()>0.5).float().sum(), region_count).detach(),
        "geotr_aefr_mean_abs_boundary_displacement_px": _masked_mean(aefr_displacement[:,0].detach().abs(), region_b).detach(),
        "geotr_aefr_mean_abs_interior_delta_logit": _masked_mean(aefr_interior_delta[:,0].detach().abs(), region_b).detach(),
        "geotr_pc2r_reachable_region_error_count": reachable_err.float().sum().detach(),
        "geotr_pc2r_unreachable_region_error_count": unreachable_err.float().sum().detach(),
        "geotr_pc2r_reachable_region_error_fraction": _safe_ratio(reachable_err.float().sum(), region_err).detach(),
        "geotr_pc2r_unreachable_fn_count": unreachable_fn.float().sum().detach(),
        "geotr_pc2r_unreachable_fp_count": unreachable_fp.float().sum().detach(),
        "geotr_pc2r_reachable_oracle_gain": reachable_oracle_gain.detach(),
        "geotr_pc2r_error_near_boundary_1_fraction": _safe_ratio((region_b & hard_err & near1).float().sum(), region_err).detach(),
        "geotr_pc2r_error_near_boundary_2_fraction": _safe_ratio((region_b & hard_err & near2).float().sum(), region_err).detach(),
        "geotr_pc2r_error_near_boundary_3_fraction": _safe_ratio((region_b & hard_err & near3).float().sum(), region_err).detach(),
        "geotr_pc2r_error_near_boundary_5_fraction": _safe_ratio((region_b & hard_err & near5).float().sum(), region_err).detach(),
        "geotr_pc2r_error_beyond_boundary_5_fraction": _safe_ratio((region_b & hard_err & (~near5)).float().sum(), region_err).detach(),
        "geotr_pc2r_boundary5_oracle_gain": boundary5_oracle_gain.detach(),
        "geotr_pc2r_interior_oracle_gain": interior_oracle_gain.detach(),
        "geotr_c2r_region_coverage": region_b.float().mean().detach(),
        "geotr_c2r_center_count": center_b.float().sum().detach(),
        "geotr_c2r_region_error_density": _safe_ratio(region_err, region_count).detach(),
        "geotr_c2r_region_error_recall": _safe_ratio(region_err, total_err).detach(),
        "geotr_c2r_consensus_rate_in_region": _safe_ratio(consensus_count, region_count).detach(),
        "geotr_c2r_edit_rate_in_region": _safe_ratio(edit_count, region_count).detach(),
        "geotr_c2r_edit_precision": _safe_ratio(corrected_count, corrected_count + introduced_count).detach(),
        "geotr_c2r_region_oracle_gain": region_oracle_gain.detach(),
        "geotr_c2r_stage2_gain": stage_gain_pc.mean().detach(),
        "geotr_c2r_candidate_pixel_count": candidate_count.detach(),
        "geotr_c2r_commit_pixel_count": commit_count.detach(),
        "geotr_c2r_roi_overlap_pixel_count": roi_overlap.detach(),
        "geotr_c2r_roi_unique_pixel_count": roi_unique.detach(),
        "geotr_c2r_center_min_chebyshev_distance": center_min_dist.detach(),
        "geotr_c2r_candidate_component_count": candidate_components.detach(),
        "geotr_c2r_committed_component_count": committed_components.detach(),
        "geotr_c2r_candidate_component_area_mean": candidate_component_area.detach(),
        "geotr_c2r_committed_component_area_mean": committed_component_area.detach(),
        "geotr_c2r_component_agreement_mean": component_agreement.detach(),
        "geotr_c2r_component_spread_q90_mean": component_spread.detach(),
        "geotr_c2r_component_confidence_q10_mean": component_confidence.detach(),
        "geotr_c2r_bce_grad_mass_error": error_grad_mass.detach(),
        "geotr_c2r_bce_grad_mass_correct": correct_grad_mass.detach(),
        "geotr_c2r_bce_grad_mass_error_fraction": _safe_ratio(error_grad_mass, error_grad_mass + correct_grad_mass).detach(),
        "geotopo_error_target_rate": hard_err.float().mean().detach(),
        "geotopo_error_precision": final_prob.new_zeros(()),
        "geotopo_error_recall": final_prob.new_zeros(()),
        "geotopo_error_f1": final_prob.new_zeros(()),
        "geotr_v4f_enabled": final_prob.new_zeros(()),
        "geotr_v4e_enabled": final_prob.new_zeros(()),
        "geotr_v4d_enabled": final_prob.new_zeros(()),
    }
    return objective, diagnostics


def _compute_v4g_loss(
    cfg,
    target: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    mode: str,
    ce_weight: float,
    dice_weight: float,
    epoch: int = 0,
):
    """V4G/V4G-R2/V4G-R3 sparse-refinement objective.

    R1 directly reclassifies every selected point against hard GT.

    R2 implements an explicit correction/preserve contract:
      * selected hard-error points -> standard GT BCE (Correction);
      * selected already-correct points -> detached anchor soft-target BCE
        (Preserve, whose optimum is exactly zero residual delta);
      * denoising points use the known synthetic corruption support, not a
        predicted error class;
      * region means are normalized independently so the many preserve points
        cannot numerically swamp the sparse correction points.

    R3 removes the R2 action-prior distortion and the unbounded reclassification
    objective.  The sparse head regresses a *minimal signed logit intervention*:
      * selected already-correct points -> delta*=0 exactly;
      * selected FN/FP points -> the smallest signed delta that reaches the
        pre-declared 0.55/0.45 decision margin (or its configured equivalent);
      * all selected points are averaged together, so the optimization prior
        matches the empirical deployment prevalence instead of forcing 50:50
        correction/preserve pressure;
      * synthetic denoising is disabled in the first root-causal R3 probe.
    """
    m1 = _cfg_get(cfg, "M1", None)
    if bool(_cfg_get(m1, "GEOTR_SPARC_HR_ENABLED", False)):
        return compute_sparc_hr_loss(
            cfg, target, aux, mode, ce_weight, dice_weight, epoch=epoch
        )
    if bool(_cfg_get(m1, "GEOTR_C2R_ENABLED", False)):
        return _compute_c2r_loss(cfg, target, aux, mode, ce_weight, dice_weight, epoch=epoch)
    base_prob = aux["geotopo_base_probs"]
    geo_prob = aux["geotopo_geometry_probs"]
    geo_logits = aux["geotopo_geometry_logits"]
    final_prob = aux["geotopo_final_probs"]
    final_logits = aux["geotopo_final_logits"]
    flow = aux["geotopo_geometry_flow_px"]
    prefix = "geotopo_reconstruction_after_geometry" if mode == "full" else "geotopo_reconstruction_base"
    anchor = geo_prob.detach() if mode == "full" else base_prob.detach()
    selection = aux[prefix + "_v4g_selection_mask"]
    refined_logits = aux[prefix + "_v4g_refined_logits"]
    refined_prob = aux[prefix + "_v4g_refined_prob"]
    delta_pred = aux.get(prefix + "_v4g_delta_logit", refined_logits - torch.logit(anchor.clamp(EPS, 1.0 - EPS)))
    dn_selection = aux[prefix + "_v4g_dn_selection_mask"]
    dn_refined_logits = aux[prefix + "_v4g_dn_refined_logits"]
    dn_anchor = aux[prefix + "_v4g_dn_anchor_prob"]
    dn_corruption = aux[prefix + "_v4g_dn_corruption_mask"]
    margin = aux[prefix + "_v4g_margin_uncertainty"]
    mc_std = aux[prefix + "_v4g_mc_std_map"]
    mc_dis = aux[prefix + "_v4g_mc_disagreement_map"]
    entropy = aux[prefix + "_v4g_entropy_map"]
    r4_flip_logits = aux.get(prefix + "_v4g_r4_flip_logits")
    r4_flip_prob = aux.get(prefix + "_v4g_r4_flip_prob")
    r4_synth_flip_logits = aux.get(prefix + "_v4g_r4_synth_flip_logits")
    r4_synth_target = aux.get(prefix + "_v4g_r4_synth_target")

    target4 = target[:, None].to(final_prob)
    geo_loss, geo_bce, geo_dice_loss = _seg_loss(geo_logits, geo_prob, target, ce_weight, dice_weight)
    final_loss, final_bce, final_dice_loss = _seg_loss(final_logits, final_prob, target, ce_weight, dice_weight)
    r2 = bool(_cfg_get(m1, "GEOTR_V4G_R2_CORRECTION_PRESERVE_ENABLED", False))
    r3 = r2 and bool(_cfg_get(m1, "GEOTR_V4G_R3_MINIMAL_INTERVENTION_ENABLED", False))
    r4 = r2 and bool(_cfg_get(m1, "GEOTR_V4G_R4_EXOGENOUS_PATCH_FLIP_ENABLED", False))

    hard_err4 = (((anchor >= 0.5) != (target4 >= 0.5))).detach()
    selected_b = selection.detach() > 0.5
    correction_mask = selected_b & hard_err4
    preserve_mask = selected_b & (~hard_err4)

    if r4:
        if not isinstance(r4_synth_flip_logits, torch.Tensor) or not isinstance(r4_synth_target, torch.Tensor):
            raise RuntimeError("GEOTR-V4G-R4 requires synthetic flip logits/targets from the forward path")
        synth_sel = dn_selection.detach() > 0.5
        # Stable exogenous supervision: FLIP iff the synthetic coarse mask was
        # corrupted. No current-anchor residual label enters the R4 objective.
        r4_flip_loss = _masked_bce_with_logits(
            r4_synth_flip_logits, r4_synth_target.detach(), synth_sel
        )
        factual_point_loss = r4_flip_loss
        correction_loss = r4_flip_loss
        preserve_loss = r4_flip_loss.detach() * 0.0
        dn_correction_loss = r4_flip_loss.detach() * 0.0
        dn_preserve_loss = r4_flip_loss.detach() * 0.0
        denoise_point_loss = r4_flip_loss
        stage2_loss = r4_flip_loss
        delta_target = torch.zeros_like(delta_pred)
        gt_sign = torch.where(target4 >= 0.5, torch.ones_like(target4), -torch.ones_like(target4))
    elif r3:
        # Minimal signed intervention target.  This is deliberately defined in
        # the exact deployed variable (delta-logit), not in a second absolute
        # segmentation probability.  Correct selected pixels have an exact
        # zero target; hard-error pixels receive only the smallest step needed
        # to reach a small GT-side decision margin.
        prob_margin = float(_cfg_get(m1, "GEOTR_V4G_R3_TARGET_PROB_MARGIN", 0.05))
        prob_margin = min(max(prob_margin, 1.0e-4), 0.49)
        pos_prob = anchor.new_tensor(0.5 + prob_margin)
        target_logit_margin = torch.logit(pos_prob)
        anchor_logit = torch.logit(anchor.clamp(EPS, 1.0 - EPS)).detach()
        gt_sign = torch.where(target4 >= 0.5, torch.ones_like(target4), -torch.ones_like(target4))
        required_mag = (target_logit_margin - gt_sign * anchor_logit).clamp_min(0.0)
        max_step = float(_cfg_get(m1, "GEOTR_V4G_R3_MAX_ABS_DELTA_LOGIT", 1.0))
        required_mag = required_mag.clamp_max(max(max_step, 1.0e-3))
        delta_target = torch.where(hard_err4, gt_sign * required_mag, torch.zeros_like(required_mag)).detach()

        beta = float(_cfg_get(m1, "GEOTR_V4G_R3_SMOOTH_L1_BETA", 0.10))
        # One empirical-risk mean over *all selected points*.  This preserves
        # the real correction/preserve prevalence instead of the R2 50:50
        # region-mean prior.
        factual_point_loss = _masked_smooth_l1(delta_pred, delta_target, selected_b, beta=beta)
        correction_loss = _masked_smooth_l1(delta_pred, delta_target, correction_mask, beta=beta)
        preserve_loss = _masked_smooth_l1(delta_pred, delta_target, preserve_mask, beta=beta)
        dn_correction_loss = factual_point_loss.detach() * 0.0
        dn_preserve_loss = factual_point_loss.detach() * 0.0
        denoise_point_loss = factual_point_loss.detach() * 0.0
        final_w = max(float(_cfg_get(m1, "GEOTR_V4G_FINAL_SEG_WEIGHT", 0.0)), 0.0)
        stage2_loss = factual_point_loss + final_w * final_loss
    elif r2:
        correction_loss = _masked_bce_with_logits(refined_logits, target4, correction_mask)
        # Soft-target BCE is a standard consistency/distillation objective.  Its
        # derivative wrt the residual logit is p_refined - p_anchor, therefore
        # delta=0 is an exact optimum on already-correct selected locations.
        preserve_loss = _masked_bce_with_logits(refined_logits, anchor.detach(), preserve_mask)

        dn_selected_b = dn_selection.detach() > 0.5
        dn_corr_mask = dn_selected_b & (dn_corruption.detach() > 0.5)
        dn_keep_mask = dn_selected_b & (~(dn_corruption.detach() > 0.5))
        dn_correction_loss = _masked_bce_with_logits(dn_refined_logits, target4, dn_corr_mask)
        dn_preserve_loss = _masked_bce_with_logits(dn_refined_logits, dn_anchor.detach(), dn_keep_mask)
        dn_terms = []
        if float(dn_corr_mask.float().sum().detach()) > 0:
            dn_terms.append(dn_correction_loss)
        if float(dn_keep_mask.float().sum().detach()) > 0:
            dn_terms.append(dn_preserve_loss)
        denoise_point_loss = (
            sum(dn_terms) / float(len(dn_terms)) if dn_terms else correction_loss.detach() * 0.0
        )

        corr_w = max(float(_cfg_get(m1, "GEOTR_V4G_R2_CORRECTION_WEIGHT", 1.0)), 0.0)
        keep_w = max(float(_cfg_get(m1, "GEOTR_V4G_R2_PRESERVE_WEIGHT", 1.0)), 0.0)
        denoise_w = max(float(_cfg_get(m1, "GEOTR_V4G_R2_DENOISE_WEIGHT", 1.0)), 0.0)
        final_w = max(float(_cfg_get(m1, "GEOTR_V4G_FINAL_SEG_WEIGHT", 0.0)), 0.0)
        terms = []
        weights = []
        if corr_w > 0.0:
            terms.append(corr_w * correction_loss); weights.append(corr_w)
        if keep_w > 0.0:
            terms.append(keep_w * preserve_loss); weights.append(keep_w)
        if denoise_w > 0.0:
            terms.append(denoise_w * denoise_point_loss); weights.append(denoise_w)
        if final_w > 0.0:
            terms.append(final_w * final_loss); weights.append(final_w)
        stage2_loss = sum(terms) / max(sum(weights), 1.0e-8)
        factual_point_loss = 0.5 * (correction_loss + preserve_loss)
    else:
        factual_point_loss = _masked_bce_with_logits(refined_logits, target4, selection)
        correction_loss = factual_point_loss
        preserve_loss = factual_point_loss.detach() * 0.0
        dn_correction_loss = factual_point_loss.detach() * 0.0
        dn_preserve_loss = factual_point_loss.detach() * 0.0
        dn_active = dn_selection.detach().sum() > 0
        if bool(dn_active):
            denoise_point_loss = _masked_bce_with_logits(dn_refined_logits, target4, dn_selection)
        else:
            denoise_point_loss = factual_point_loss.detach() * 0.0
        factual_w = float(_cfg_get(m1, "GEOTR_V4G_FACTUAL_POINT_WEIGHT", 1.0))
        denoise_w = float(_cfg_get(m1, "GEOTR_V4G_DENOISE_POINT_WEIGHT", 1.0))
        final_w = float(_cfg_get(m1, "GEOTR_V4G_FINAL_SEG_WEIGHT", 0.25))
        denom = max(max(factual_w, 0.0) + max(denoise_w, 0.0) + max(final_w, 0.0), 1.0e-8)
        stage2_loss = (
            max(factual_w, 0.0) * factual_point_loss
            + max(denoise_w, 0.0) * denoise_point_loss
            + max(final_w, 0.0) * final_loss
        ) / denom

    if mode == "base":
        objective = final_loss * 0.0
    elif mode == "geometry":
        objective = geo_loss
    elif mode == "residual":
        objective = stage2_loss
    else:
        objective = geo_loss + stage2_loss

    smooth = _flow_smoothness(flow)
    smooth_w = float(_cfg_get(m1, "GEOTOPO_SMOOTHNESS_WEIGHT", 0.0))
    if mode in {"geometry", "full"} and smooth_w > 0.0:
        objective = objective + smooth_w * smooth
    compat = aux.get("mhcs_m1_distribution_log_var")
    if isinstance(compat, torch.Tensor):
        objective = objective + 0.0 * compat

    base_dice = _dice_per_case(base_prob[:, 0].detach(), target)
    geo_dice = _dice_per_case(geo_prob[:, 0].detach(), target)
    recon_base = aux["geotopo_residual_only_probs"]
    recon_geo = aux["geotopo_reconstruction_after_geometry_probs"]
    recon_base_dice = _dice_per_case(recon_base[:, 0].detach(), target)
    recon_geo_dice = _dice_per_case(recon_geo[:, 0].detach(), target)
    final_dice = _dice_per_case(final_prob[:, 0].detach(), target)
    anchor_dice = _dice_per_case(anchor[:, 0], target)
    stage_gain_pc = final_dice - anchor_dice
    selection_stats = _selection_stats(selection, anchor, target)
    before_point_acc = _point_accuracy(anchor, target, selection)
    after_point_acc = _point_accuracy(refined_prob, target, selection)
    hard_err = ((anchor[:, 0] >= 0.5) != (target >= 0.5))
    margin_ap = _average_precision_binary(margin[:, 0], hard_err)
    std_ap = _average_precision_binary(mc_std[:, 0], hard_err)
    dis_ap = _average_precision_binary(mc_dis[:, 0], hard_err)
    entropy_ap = _average_precision_binary(entropy[:, 0], hard_err)

    # Hard correction accounting: unlike BCE, these quantities directly answer
    # whether Stage-2 fixed more anchor errors than it introduced.
    refined_hard = refined_prob[:, 0].detach() >= 0.5
    gt_hard = target >= 0.5
    sel3 = selection[:, 0].detach() > 0.5
    corrected = sel3 & hard_err & (refined_hard == gt_hard)
    introduced = sel3 & (~hard_err) & (refined_hard != gt_hard)
    corrected_count = corrected.float().sum()
    introduced_count = introduced.float().sum()
    net_correction = corrected_count - introduced_count
    correction_recall = _safe_ratio(corrected_count, hard_err.float().sum())
    introduction_rate = _safe_ratio(introduced_count, sel3.float().sum())

    # Orthogonal selector ceilings with GT used strictly as diagnostics.
    pred_selector_oracle = anchor.clone()
    pred_fix = sel3 & hard_err
    pred_selector_oracle[:, 0] = torch.where(pred_fix, target, anchor[:, 0])
    full_residual_oracle = anchor.clone()
    full_residual_oracle[:, 0] = torch.where(hard_err, target, anchor[:, 0])
    pred_selector_oracle_gain = (
        _dice_per_case(pred_selector_oracle[:, 0], target) - anchor_dice
    ).mean()
    full_residual_oracle_gain = (
        _dice_per_case(full_residual_oracle[:, 0], target) - anchor_dice
    ).mean()

    if r3 and not r4:
        sel_bool4 = selected_b
        err_sel = correction_mask
        keep_sel = preserve_mask
        pred_abs = delta_pred.detach().abs()
        tgt_abs = delta_target.detach().abs()
        sign_ok = ((delta_pred.detach() * gt_sign) > 0.0) & err_sel
        diagnostics_r3 = {
            "geotr_v4g_r3_enabled": final_prob.new_tensor(1.0).detach(),
            "geotr_v4g_r3_delta_loss": factual_point_loss.detach(),
            "geotr_v4g_r3_pred_abs_delta": _masked_mean(pred_abs[:,0], sel_bool4[:,0]).detach(),
            "geotr_v4g_r3_target_abs_delta": _masked_mean(tgt_abs[:,0], sel_bool4[:,0]).detach(),
            "geotr_v4g_r3_error_pred_abs_delta": _masked_mean(pred_abs[:,0], err_sel[:,0]).detach(),
            "geotr_v4g_r3_keep_pred_abs_delta": _masked_mean(pred_abs[:,0], keep_sel[:,0]).detach(),
            "geotr_v4g_r3_error_sign_accuracy": _safe_ratio(sign_ok.float().sum(), err_sel.float().sum()).detach(),
            "geotr_v4g_r3_empirical_error_fraction_selected": _safe_ratio(err_sel.float().sum(), sel_bool4.float().sum()).detach(),
        }
    else:
        diagnostics_r3 = {
            "geotr_v4g_r3_enabled": final_prob.new_tensor(0.0).detach(),
            "geotr_v4g_r3_delta_loss": final_prob.new_tensor(0.0).detach(),
            "geotr_v4g_r3_pred_abs_delta": final_prob.new_tensor(0.0).detach(),
            "geotr_v4g_r3_target_abs_delta": final_prob.new_tensor(0.0).detach(),
            "geotr_v4g_r3_error_pred_abs_delta": final_prob.new_tensor(0.0).detach(),
            "geotr_v4g_r3_keep_pred_abs_delta": final_prob.new_tensor(0.0).detach(),
            "geotr_v4g_r3_error_sign_accuracy": final_prob.new_tensor(0.0).detach(),
            "geotr_v4g_r3_empirical_error_fraction_selected": final_prob.new_tensor(0.0).detach(),
        }

    if r4 and isinstance(r4_flip_prob, torch.Tensor):
        flip_pred = (r4_flip_prob[:, 0].detach() >= float(_cfg_get(m1, "GEOTR_V4G_R4_FLIP_THRESHOLD", 0.50))) & sel3
        selected_error = sel3 & hard_err
        selected_correct = sel3 & (~hard_err)
        flip_tp = (flip_pred & selected_error).float().sum()
        flip_fp = (flip_pred & selected_correct).float().sum()
        flip_fn = ((~flip_pred) & selected_error).float().sum()
        diagnostics_r4 = {
            "geotr_v4g_r4_enabled": final_prob.new_tensor(1.0).detach(),
            "geotr_v4g_r4_flip_loss": stage2_loss.detach(),
            "geotr_v4g_r4_flip_probability_mean": _masked_mean(r4_flip_prob[:,0].detach(), sel3).detach(),
            "geotr_v4g_r4_flip_probability_error": _masked_mean(r4_flip_prob[:,0].detach(), selected_error).detach(),
            "geotr_v4g_r4_flip_probability_correct": _masked_mean(r4_flip_prob[:,0].detach(), selected_correct).detach(),
            "geotr_v4g_r4_flip_tp_count": flip_tp.detach(),
            "geotr_v4g_r4_flip_fp_count": flip_fp.detach(),
            "geotr_v4g_r4_flip_fn_count": flip_fn.detach(),
            "geotr_v4g_r4_selected_error_count": selected_error.float().sum().detach(),
            "geotr_v4g_r4_selected_correct_count": selected_correct.float().sum().detach(),
            "geotr_v4g_r4_total_error_count": hard_err.float().sum().detach(),
            "geotr_v4g_r4_flip_precision": _safe_ratio(flip_tp, flip_tp + flip_fp).detach(),
            "geotr_v4g_r4_flip_recall_selected": _safe_ratio(flip_tp, selected_error.float().sum()).detach(),
            "geotr_v4g_r4_false_flip_rate": _safe_ratio(flip_fp, selected_correct.float().sum()).detach(),
            "geotr_v4g_r4_effective_edit_rate": _safe_ratio(flip_pred.float().sum(), sel3.float().sum()).detach(),
        }
    else:
        diagnostics_r4 = {
            "geotr_v4g_r4_enabled": final_prob.new_tensor(0.0).detach(),
            "geotr_v4g_r4_flip_loss": final_prob.new_tensor(0.0).detach(),
        }

    diagnostics = {
        "mhcs_objective": objective.detach(),
        "mhcs_m1_objective": objective.detach(),
        "mhcs_m2_objective": stage2_loss.detach() if mode in {"residual", "full"} else objective.detach() * 0.0,
        "mhcs_final_gain": (final_dice - base_dice).mean().detach(),
        "mhcs_base_dice": base_dice.mean().detach(),
        "mhcs_final_dice": final_dice.mean().detach(),
        "mhcs_candidate_mean_gain": (final_dice - base_dice).mean().detach(),
        "mhcs_candidate_harm_rate": (final_dice < base_dice).float().mean().detach(),
        "geotopo_final_loss": final_loss.detach(),
        "geotopo_geometry_loss": geo_loss.detach(),
        "geotopo_bce_loss": final_bce.detach(),
        "geotopo_dice_loss": final_dice_loss.detach(),
        "geotopo_geometry_bce_loss": geo_bce.detach(),
        "geotopo_geometry_dice_loss": geo_dice_loss.detach(),
        "geotopo_base_dice": base_dice.mean().detach(),
        "geotopo_geometry_dice": geo_dice.mean().detach(),
        "geotopo_reconstruction_base_dice": recon_base_dice.mean().detach(),
        "geotopo_reconstruction_after_geometry_dice": recon_geo_dice.mean().detach(),
        "geotopo_final_dice": final_dice.mean().detach(),
        "geotopo_geometry_gain": (geo_dice - base_dice).mean().detach(),
        "geotopo_reconstruction_gain_without_geometry": (recon_base_dice - base_dice).mean().detach(),
        "geotopo_reconstruction_gain_after_geometry": (recon_geo_dice - geo_dice).mean().detach(),
        "geotopo_final_gain": (final_dice - base_dice).mean().detach(),
        "geotr_v4g_enabled": final_prob.new_tensor(1.0).detach(),
        "geotr_v4g_r2_enabled": final_prob.new_tensor(float(r2)).detach(),
        "geotr_v4g_factual_point_loss": factual_point_loss.detach(),
        "geotr_v4g_correction_loss": correction_loss.detach(),
        "geotr_v4g_preserve_loss": preserve_loss.detach(),
        "geotr_v4g_denoise_point_loss": denoise_point_loss.detach(),
        "geotr_v4g_denoise_correction_loss": dn_correction_loss.detach(),
        "geotr_v4g_denoise_preserve_loss": dn_preserve_loss.detach(),
        "geotr_v4g_selection_coverage": selection_stats["coverage"].detach(),
        "geotr_v4g_selection_precision": selection_stats["precision"].detach(),
        "geotr_v4g_selection_recall": selection_stats["recall"].detach(),
        "geotr_v4g_anchor_error_rate": selection_stats["error_rate"].detach(),
        "geotr_v4g_point_accuracy_before": before_point_acc.detach(),
        "geotr_v4g_point_accuracy_after": after_point_acc.detach(),
        "geotr_v4g_point_accuracy_gain": (after_point_acc - before_point_acc).detach(),
        "geotr_v4g_selected_abs_change": _masked_mean((refined_prob[:,0].detach()-anchor[:,0]).abs(), selection[:,0] > 0.5).detach(),
        "geotr_v4g_denoise_corruption_rate": dn_corruption.float().mean().detach(),
        "geotr_v4g_denoise_selected_rate": dn_selection.float().mean().detach(),
        "geotr_v4g_margin_error_ap": margin_ap.detach(),
        "geotr_v4g_mc_std_error_ap": std_ap.detach(),
        "geotr_v4g_mc_disagreement_error_ap": dis_ap.detach(),
        "geotr_v4g_entropy_error_ap": entropy_ap.detach(),
        "geotr_v4g_margin_recall_at_05": _topk_recall(margin[:,0], hard_err, 0.05).detach(),
        "geotr_v4g_margin_recall_at_10": _topk_recall(margin[:,0], hard_err, 0.10).detach(),
        "geotr_v4g_mc_std_recall_at_05": _topk_recall(mc_std[:,0], hard_err, 0.05).detach(),
        "geotr_v4g_mc_std_recall_at_10": _topk_recall(mc_std[:,0], hard_err, 0.10).detach(),
        "geotr_v4g_mc_disagreement_recall_at_05": _topk_recall(mc_dis[:,0], hard_err, 0.05).detach(),
        "geotr_v4g_mc_disagreement_recall_at_10": _topk_recall(mc_dis[:,0], hard_err, 0.10).detach(),
        "geotr_v4g_corrected_error_count": corrected_count.detach(),
        "geotr_v4g_introduced_error_count": introduced_count.detach(),
        "geotr_v4g_net_correction_count": net_correction.detach(),
        "geotr_v4g_correction_recall": correction_recall.detach(),
        "geotr_v4g_introduction_rate": introduction_rate.detach(),
        "geotr_v4g_pred_selector_oracle_gain": pred_selector_oracle_gain.detach(),
        "geotr_v4g_full_residual_oracle_gain": full_residual_oracle_gain.detach(),
        "geotr_v4g_stage2_benefit_rate": (stage_gain_pc > 1.0e-8).float().mean().detach(),
        "geotr_v4g_stage2_harm_rate": (stage_gain_pc < -1.0e-8).float().mean().detach(),
        "geotr_v4g_stage2_soft_gain": stage_gain_pc.mean().detach(),
        "geotopo_error_target_rate": hard_err.float().mean().detach(),
        "geotopo_error_precision": final_prob.new_zeros(()),
        "geotopo_error_recall": final_prob.new_zeros(()),
        "geotopo_error_f1": final_prob.new_zeros(()),
        "geotr_v4f_enabled": final_prob.new_zeros(()),
        "geotr_v4e_enabled": final_prob.new_zeros(()),
        "geotr_v4d_enabled": final_prob.new_zeros(()),
    }
    diagnostics.update(diagnostics_r3)
    diagnostics.update(diagnostics_r4)
    return objective, diagnostics

def compute_multi_hypothesis_composition_loss(
    cfg,
    candidate_logits: torch.Tensor,
    masks: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    epoch: int = 0,
):
    del candidate_logits
    m1 = _cfg_get(cfg, "M1", None)
    train = _cfg_get(cfg, "TRAIN", None)
    mode = str(_cfg_get(m1, "GEOTOPO_MODE", "full")).strip().lower()
    if mode not in {"base", "geometry", "residual", "full"}:
        raise ValueError(f"Unsupported GEOTOPO_MODE={mode!r}")
    v4g = bool(_cfg_get(m1, "GEOTR_V4G_SPARSE_DIRECT_REFINER_ENABLED", False))
    v4f = bool(_cfg_get(m1, "GEOTR_V4F_SELECTIVE_INTERVENTION_ENABLED", False)) and not v4g
    v4e = bool(_cfg_get(m1, "GEOTR_V4E_OPERATOR_CONSISTENT_ENABLED", False)) and not v4f and not v4g
    v4d = bool(_cfg_get(m1, "GEOTR_V4D_ROOT_FIX_ENABLED", False)) and not v4e and not v4f and not v4g

    target = _target_3d(masks)
    final_logits = aux.get("geotopo_final_logits", aux.get("mhcs_final_logits"))
    final_prob = aux.get("geotopo_final_probs", aux.get("mhcs_final_probs"))
    base_prob = aux.get("geotopo_base_probs")
    geo_logits = aux.get("geotopo_geometry_logits")
    geo_prob = aux.get("geotopo_geometry_probs")
    recon_base_prob = aux.get("geotopo_residual_only_probs")
    recon_geo_prob = aux.get("geotopo_reconstruction_after_geometry_probs")
    flow = aux.get("geotopo_geometry_flow_px")

    prefix = "geotopo_reconstruction_after_geometry" if mode == "full" else "geotopo_reconstruction_base"
    error_prob = aux.get(prefix + "_error_probs")
    typed_logits = aux.get(prefix + "_typed_logits")
    typed_prob = aux.get(prefix + "_typed_probs")
    q_fn = aux.get(prefix + "_q_fn")
    q_fp = aux.get(prefix + "_q_fp")
    severity_fn = aux.get(prefix + "_severity_fn")
    severity_fp = aux.get(prefix + "_severity_fp")
    add_magnitude = aux.get(prefix + "_add_magnitude")
    remove_magnitude = aux.get(prefix + "_remove_magnitude")
    v4f_candidate_mask = aux.get(prefix + "_v4f_proposal_candidate_mask")
    v4f_edit_logit = aux.get(prefix + "_v4f_policy_edit_logit")
    v4f_edit_prob = aux.get(prefix + "_v4f_policy_edit_prob")
    v4f_direction_logits = aux.get(prefix + "_v4f_policy_direction_logits")
    v4f_direction_prob = aux.get(prefix + "_v4f_policy_direction_prob")
    v4f_soft_prob = aux.get(prefix + "_v4f_soft_prob")
    v4f_hard_prob = aux.get(prefix + "_v4f_hard_prob")

    required = (final_logits, final_prob, base_prob, geo_logits, geo_prob, recon_base_prob, recon_geo_prob, flow)
    if not all(isinstance(x, torch.Tensor) for x in required):
        raise RuntimeError("GEOTR-V4C/V4D/V4E loss is missing required geotopo_* tensors.")

    target = target.to(final_prob)
    ce_weight = float(_cfg_get(train, "CE_WEIGHT", 0.5))
    dice_weight = float(_cfg_get(train, "DICE_WEIGHT", 0.5))
    total_w = max(ce_weight + dice_weight, 1.0e-8)
    ce_weight, dice_weight = ce_weight / total_w, dice_weight / total_w

    final_loss, final_bce, final_dice_loss = _seg_loss(final_logits, final_prob, target, ce_weight, dice_weight)
    geo_loss, geo_bce, geo_dice_loss = _seg_loss(geo_logits, geo_prob, target, ce_weight, dice_weight)

    if v4g:
        return _compute_v4g_loss(cfg, target, aux, mode, ce_weight, dice_weight, epoch=epoch)

    zero = final_loss.detach() * 0.0
    typed_loss = typed_ce = typed_dice = fn_dice_loss = fp_dice_loss = zero
    preserve_loss = zero
    severity_loss = severity_fn_loss = severity_fp_loss = zero
    magnitude_loss = magnitude_fn_loss = magnitude_fp_loss = zero
    teacher_loss = teacher_bce = teacher_dice_loss = zero
    teacher_prob = None
    teacher_gain = teacher_dice = zero
    severity_fn_mae = severity_fp_mae = zero
    severity_fn_error_mae = severity_fp_error_mae = zero
    magnitude_fn_mae = magnitude_fp_mae = zero
    magnitude_fn_target_mean = magnitude_fp_target_mean = zero
    edit_precision = edit_error_mean = edit_correct_mean = edit_ratio = zero
    stage2_benefit_rate = stage2_harm_rate = stage2_mean_benefit = stage2_mean_harm = zero
    cls_target = torch.zeros_like(target, dtype=torch.long)
    correct = torch.ones_like(target, dtype=torch.bool)
    fn = torch.zeros_like(target, dtype=torch.bool)
    fp = torch.zeros_like(target, dtype=torch.bool)
    typed_stats = {
        "fn_precision": zero, "fn_recall": zero, "fn_f1": zero,
        "fp_precision": zero, "fp_recall": zero, "fp_f1": zero,
        "macro_f1": zero,
    }
    error_stats = {"precision": zero, "recall": zero, "f1": zero}
    direction = {"fn": zero, "fp": zero, "all": zero}
    oracle = {k: zero for k in (
        "oracle_where_dice", "oracle_sign_dice", "oracle_typed_dice", "oracle_full_dice",
        "oracle_where_gain", "oracle_sign_gain", "oracle_typed_gain", "oracle_full_gain",
    )}
    q_correct_fp_mass = q_error_tp_mass = q_soft_precision = zero
    severity_correct_mass = severity_error_mass = severity_soft_precision = zero
    v4f_policy_presence_loss = v4f_policy_direction_loss = v4f_policy_loss = zero
    v4f_proposal_recall = v4f_proposal_precision = zero
    v4f_execution_precision = v4f_execution_recall = v4f_direction_accuracy = zero
    v4f_candidate_rate = v4f_execution_rate = zero
    v4f_teacher_gain = v4f_teacher_dice = zero
    v4f_soft_gain = v4f_hard_gain = zero
    v4f_false_action_dose = v4f_true_action_dose = zero

    if mode in {"residual", "full"}:
        active_required = (error_prob, typed_logits, typed_prob, q_fn, q_fp, add_magnitude, remove_magnitude)
        if not all(isinstance(x, torch.Tensor) for x in active_required):
            raise RuntimeError("GEOTR Stage-2 requires typed localization and directional correction outputs.")
        if v4d and not all(isinstance(x, torch.Tensor) for x in (severity_fn, severity_fp)):
            raise RuntimeError("GEOTR-V4D requires explicit severity_fn/severity_fp outputs.")
        if v4f and not all(isinstance(x, torch.Tensor) for x in (
            v4f_candidate_mask, v4f_edit_logit, v4f_edit_prob,
            v4f_direction_logits, v4f_direction_prob, v4f_soft_prob, v4f_hard_prob,
        )):
            raise RuntimeError("GEOTR-V4F requires proposal/execution/bounded-dose outputs.")

        anchor = base_prob.detach() if mode == "residual" else geo_prob.detach()
        cls_target, correct, fn, fp = _typed_targets(anchor, target)
        error_target = (fn | fp).to(target)

        # WHERE/TYPE remains an auxiliary semantic identity task.  Its weighted
        # CE probabilities are NOT used as the V4D edit amplitude.
        error_class_weight = float(_cfg_get(m1, "GEOTR_STAGE2_ERROR_CLASS_WEIGHT", 6.0))
        typed_loss, typed_ce, typed_dice, fn_dice_loss, fp_dice_loss = _typed_localization_loss(
            typed_logits, typed_prob, cls_target, fn, fp, ce_weight, dice_weight, error_class_weight
        )
        error_stats = _binary_stats(error_prob, error_target)
        pred_cls = typed_prob.detach().argmax(dim=1)
        fn_stats = _class_stats(pred_cls, fn, 1)
        fp_stats = _class_stats(pred_cls, fp, 2)
        typed_stats = {
            "fn_precision": fn_stats["precision"], "fn_recall": fn_stats["recall"], "fn_f1": fn_stats["f1"],
            "fp_precision": fp_stats["precision"], "fp_recall": fp_stats["recall"], "fp_f1": fp_stats["f1"],
            "macro_f1": 0.5 * (fn_stats["f1"] + fp_stats["f1"]),
        }

        q_error = (q_fn + q_fp).detach()[:, 0]
        q_correct_fp_mass = _masked_mean(q_error, correct)
        q_error_tp_mass = _masked_mean(q_error, fn | fp)
        q_soft_precision = (q_error * (fn | fp).to(q_error)).sum() / q_error.sum().clamp_min(EPS)

        if not v4d and not v4e and not v4f:
            # Exact V4C optimization contract retained as an explicit ablation.
            change = (final_prob[:, 0] - anchor[:, 0]).abs()
            preserve_loss = _masked_mean(change, correct)
            localizer_w = float(_cfg_get(m1, "GEOTR_STAGE2_LOCALIZER_WEIGHT", 0.5))
            segmentation_w = float(_cfg_get(m1, "GEOTR_STAGE2_SEGMENTATION_WEIGHT", 0.5))
            norm = max(localizer_w + segmentation_w, 1.0e-8)
            localizer_w, segmentation_w = localizer_w / norm, segmentation_w / norm
            preserve_w = float(_cfg_get(m1, "GEOTR_STAGE2_PRESERVE_WEIGHT", 0.05))
            stage2_loss = localizer_w * typed_loss + segmentation_w * final_loss + preserve_w * preserve_loss

            add_score = (q_fn.detach() * add_magnitude.detach() * (1.0 - anchor))[:, 0]
            remove_score = (q_fp.detach() * remove_magnitude.detach() * anchor)[:, 0]
            pred_add = add_score >= remove_score
            if bool(_cfg_get(m1, "GEOTR_STAGE2_ORACLE_DIAGNOSTICS", True)):
                oracle = _oracle_diagnostics(
                    anchor, target, q_fn.detach(), q_fp.detach(),
                    add_magnitude.detach(), remove_magnitude.detach(),
                )
        elif v4f:
            # Proposal is deliberately high-recall and never deployed directly.
            # Execution is separately optimized inside the fixed-coverage pool.
            policy_presence = _v4f_region_balanced_bce(v4f_edit_logit, error_target, v4f_candidate_mask)
            policy_direction = _v4f_direction_ce(v4f_direction_logits, fn, fp, v4f_candidate_mask)
            v4f_policy_presence_loss = policy_presence
            v4f_policy_direction_loss = policy_direction
            v4f_policy_loss = 0.5 * (policy_presence + policy_direction)

            # HOW MUCH is learned only conditional on a true edit. No off-support
            # zero target is used: NoEdit has a single owner, the execution policy.
            teacher_prob, teacher_logits = _compose_v4f_teacher(
                anchor, fn, fp, add_magnitude, remove_magnitude
            )
            teacher_loss, teacher_bce, teacher_dice_loss = _seg_loss(
                teacher_logits, teacher_prob, target, ce_weight, dice_weight
            )
            anchor_dice_pc = _dice_per_case(anchor[:, 0], target)
            teacher_dice_pc = _dice_per_case(teacher_prob[:, 0], target)
            teacher_dice = teacher_dice_pc.mean()
            teacher_gain = (teacher_dice_pc - anchor_dice_pc).mean()
            v4f_teacher_gain = teacher_gain
            v4f_teacher_dice = teacher_dice

            # The forward training prediction is the differentiable expected
            # action. Hence deploy segmentation loss directly updates execution
            # and dose. Validation/test switch to hard thresholded execution.
            soft_prob = v4f_soft_prob
            soft_logits = torch.logit(soft_prob.clamp(EPS, 1.0 - EPS))
            deploy_loss, _, _ = _seg_loss(soft_logits, soft_prob, target, ce_weight, dice_weight)

            weights = {
                "proposal": float(_cfg_get(m1, "GEOTR_V4F_PROPOSAL_LOSS_WEIGHT", 0.20)),
                "policy": float(_cfg_get(m1, "GEOTR_V4F_POLICY_LOSS_WEIGHT", 0.25)),
                "dose_teacher": float(_cfg_get(m1, "GEOTR_V4F_DOSE_TEACHER_WEIGHT", 0.25)),
                "deploy": float(_cfg_get(m1, "GEOTR_V4F_DEPLOY_LOSS_WEIGHT", 0.30)),
            }
            norm = max(sum(max(x, 0.0) for x in weights.values()), 1.0e-8)
            stage2_loss = (
                max(weights["proposal"], 0.0) * typed_loss
                + max(weights["policy"], 0.0) * v4f_policy_loss
                + max(weights["dose_teacher"], 0.0) * teacher_loss
                + max(weights["deploy"], 0.0) * deploy_loss
            ) / norm
            preserve_w = float(_cfg_get(m1, "GEOTR_V4F_PRESERVE_WEIGHT", 0.0))
            if preserve_w > 0.0:
                preserve_loss = _masked_mean((soft_prob[:, 0] - anchor[:, 0]).abs(), correct)
                stage2_loss = stage2_loss + preserve_w * preserve_loss

            threshold = float(_cfg_get(m1, "GEOTR_V4F_EXECUTION_THRESHOLD", 0.65))
            pst = _v4f_policy_stats(
                v4f_edit_prob, v4f_direction_prob, v4f_candidate_mask, fn, fp, threshold
            )
            v4f_proposal_recall = pst["proposal_recall"]
            v4f_proposal_precision = pst["proposal_precision"]
            v4f_execution_precision = pst["precision"]
            v4f_execution_recall = pst["recall"]
            v4f_direction_accuracy = pst["direction_accuracy"]
            v4f_candidate_rate = pst["candidate_rate"]
            v4f_execution_rate = pst["execution_rate"]

            hard_prob = v4f_hard_prob
            soft_d = _dice_per_case(soft_prob[:, 0].detach(), target)
            hard_d = _dice_per_case(hard_prob[:, 0].detach(), target)
            v4f_soft_gain = (soft_d - anchor_dice_pc).mean()
            v4f_hard_gain = (hard_d - anchor_dice_pc).mean()
            hard_edit = ((hard_prob - anchor).abs()[:, 0] > 1.0e-8)
            true_edit = fn | fp
            mean_dose = 0.5 * (add_magnitude[:, 0].detach() + remove_magnitude[:, 0].detach())
            v4f_true_action_dose = _masked_mean(mean_dose, hard_edit & true_edit)
            v4f_false_action_dose = _masked_mean(mean_dose, hard_edit & (~true_edit))

            # Compatibility diagnostics use the hard inference policy, not the
            # proposal argmax.  The bounded dose is alpha, not a logit magnitude.
            pred_add = (v4f_hard_prob[:, 0] > anchor[:, 0] + 1.0e-8)
            edit_precision, edit_error_mean, edit_correct_mean, edit_ratio = _edit_stats(
                hard_prob.detach(), anchor, fn | fp, correct
            )
            stage2_gain_pc = hard_d - anchor_dice_pc
            stage2_benefit_rate = (stage2_gain_pc > 1.0e-8).float().mean()
            stage2_harm_rate = (stage2_gain_pc < -1.0e-8).float().mean()
            stage2_mean_benefit = _safe_ratio(stage2_gain_pc.clamp_min(0.0).sum(), (stage2_gain_pc > 1.0e-8).float().sum())
            stage2_mean_harm = _safe_ratio(stage2_gain_pc.clamp_max(0.0).sum(), (stage2_gain_pc < -1.0e-8).float().sum())

            # Existing oracle fields are intentionally not reused for V4F: its
            # validation routine reports decomposed proposal/policy/dose audits.
        elif v4e:
            margin = float(_cfg_get(m1, "GEOTR_STAGE2_TARGET_MARGIN", 0.05))
            max_step = float(_cfg_get(m1, "GEOTR_STAGE2_MAX_LOGIT_STEP", 8.0))
            add_t, rem_t, add_cap_t, rem_cap_t = _v4e_targets(anchor, target, fn, fp, margin, max_step)
            pred_cls = typed_prob.detach().argmax(dim=1)
            pred_fn = pred_cls == 1
            pred_fp = pred_cls == 2

            # Full-dose magnitude target is defined everywhere: positive target
            # on the owning hard error support, exactly zero elsewhere.  A small
            # off-support weight prevents background domination while removing
            # V4D's unconstrained off-support magnitude extrapolation.
            off_w = float(_cfg_get(m1, "GEOTR_STAGE2_OFF_SUPPORT_MAG_WEIGHT", 0.05))
            magnitude_fn_loss = _weighted_support_smooth_l1(add_magnitude, add_t, fn, off_w)
            magnitude_fp_loss = _weighted_support_smooth_l1(remove_magnitude, rem_t, fp, off_w)
            magnitude_loss = 0.5 * (magnitude_fn_loss + magnitude_fp_loss)
            magnitude_fn_mae = _masked_mean((add_magnitude.detach() - add_t).abs()[:, 0], fn)
            magnitude_fp_mae = _masked_mean((remove_magnitude.detach() - rem_t).abs()[:, 0], fp)
            magnitude_fn_target_mean = _masked_mean(add_t[:, 0], fn)
            magnitude_fp_target_mean = _masked_mean(rem_t[:, 0], fp)

            fn4 = fn[:, None].to(anchor)
            fp4 = fp[:, None].to(anchor)
            teacher_prob, teacher_logits = _compose_v4e(anchor, fn4, fp4, add_magnitude, remove_magnitude)
            teacher_loss, teacher_bce, teacher_dice_loss = _seg_loss(
                teacher_logits, teacher_prob, target, ce_weight, dice_weight
            )
            anchor_dice_pc = _dice_per_case(anchor[:, 0], target)
            teacher_dice_pc = _dice_per_case(teacher_prob[:, 0], target)
            teacher_dice = teacher_dice_pc.mean()
            teacher_gain = (teacher_dice_pc - anchor_dice_pc).mean()

            change = (final_prob[:, 0] - anchor[:, 0]).abs()
            preserve_loss = _masked_mean(change, correct)

            weights = {
                "type": float(_cfg_get(m1, "GEOTR_STAGE2_TYPE_WEIGHT", 0.30)),
                "magnitude": float(_cfg_get(m1, "GEOTR_STAGE2_MAGNITUDE_WEIGHT", 0.30)),
                "teacher": float(_cfg_get(m1, "GEOTR_STAGE2_TEACHER_SEG_WEIGHT", 0.20)),
                "deploy": float(_cfg_get(m1, "GEOTR_STAGE2_DEPLOY_SEG_WEIGHT", 0.20)),
            }
            core_norm = max(sum(max(v, 0.0) for v in weights.values()), 1.0e-8)
            stage2_loss = (
                max(weights["type"], 0.0) * typed_loss
                + max(weights["magnitude"], 0.0) * magnitude_loss
                + max(weights["teacher"], 0.0) * teacher_loss
                + max(weights["deploy"], 0.0) * final_loss
            ) / core_norm
            preserve_w = float(_cfg_get(m1, "GEOTR_STAGE2_PRESERVE_WEIGHT", 0.02))
            stage2_loss = stage2_loss + preserve_w * preserve_loss

            pred_add = pred_fn
            # V4E compatibility fields: hard support is the deployed WHERE.
            severity_error = (pred_fn | pred_fp).float()
            severity_correct_mass = _masked_mean(severity_error, correct)
            severity_error_mass = _masked_mean(severity_error, fn | fp)
            severity_soft_precision = (
                severity_error * (fn | fp).to(severity_error)
            ).sum() / severity_error.sum().clamp_min(EPS)

            edit_precision, edit_error_mean, edit_correct_mean, edit_ratio = _edit_stats(
                final_prob.detach(), anchor, fn | fp, correct
            )
            stage2_gain_pc = _dice_per_case(final_prob[:, 0].detach(), target) - _dice_per_case(anchor[:, 0], target)
            stage2_benefit_rate = (stage2_gain_pc > 1.0e-8).float().mean()
            stage2_harm_rate = (stage2_gain_pc < -1.0e-8).float().mean()
            stage2_mean_benefit = _safe_ratio(
                stage2_gain_pc.clamp_min(0.0).sum(), (stage2_gain_pc > 1.0e-8).float().sum()
            )
            stage2_mean_harm = _safe_ratio(
                stage2_gain_pc.clamp_max(0.0).sum(), (stage2_gain_pc < -1.0e-8).float().sum()
            )
            if bool(_cfg_get(m1, "GEOTR_STAGE2_ORACLE_DIAGNOSTICS", True)):
                v4e_oracle = _v4e_oracle_diagnostics(
                    anchor, target, pred_fn, pred_fp, add_magnitude.detach(), remove_magnitude.detach(),
                    add_t.detach(), rem_t.detach(),
                )
                oracle = {
                    "oracle_where_dice": v4e_oracle["oracle_support_dice"],
                    "oracle_sign_dice": v4e_oracle["oracle_magnitude_dice"],
                    "oracle_typed_dice": v4e_oracle["oracle_support_dice"],
                    "oracle_full_dice": v4e_oracle["oracle_full_dice"],
                    "oracle_where_gain": v4e_oracle["oracle_support_gain"],
                    "oracle_sign_gain": v4e_oracle["oracle_magnitude_gain"],
                    "oracle_typed_gain": v4e_oracle["oracle_support_gain"],
                    "oracle_full_gain": v4e_oracle["oracle_full_gain"],
                }
        else:
            assert isinstance(severity_fn, torch.Tensor) and isinstance(severity_fp, torch.Tensor)
            margin = float(_cfg_get(m1, "GEOTR_STAGE2_TARGET_MARGIN", 0.05))
            max_step = float(_cfg_get(m1, "GEOTR_STAGE2_MAX_LOGIT_STEP", 4.0))
            sev_fn_t, sev_fp_t, add_t, rem_t = _v4d_targets(anchor, target, fn, fp, margin, max_step)

            # Direct continuous severity supervision.  Error pixels can receive
            # extra weight, but the target itself remains a calibrated residual
            # severity rather than a class-weighted posterior.
            sev_err_w = float(_cfg_get(m1, "GEOTR_STAGE2_SEVERITY_ERROR_WEIGHT", 4.0))
            err4 = (fn | fp)[:, None].to(anchor)
            sev_weight = 1.0 + (max(sev_err_w, 1.0) - 1.0) * err4
            severity_fn_loss = _weighted_smooth_l1(severity_fn, sev_fn_t, sev_weight)
            severity_fp_loss = _weighted_smooth_l1(severity_fp, sev_fp_t, sev_weight)
            severity_loss = 0.5 * (severity_fn_loss + severity_fp_loss)
            severity_fn_mae = (severity_fn.detach() - sev_fn_t).abs().mean()
            severity_fp_mae = (severity_fp.detach() - sev_fp_t).abs().mean()
            severity_fn_error_mae = _masked_mean((severity_fn.detach() - sev_fn_t).abs()[:, 0], fn | fp)
            severity_fp_error_mae = _masked_mean((severity_fp.detach() - sev_fp_t).abs()[:, 0], fn | fp)

            # HOW is directly supervised on true residual support; therefore a
            # wrong predicted WHERE cannot suppress its learning signal.
            magnitude_fn_loss = _masked_smooth_l1(add_magnitude, add_t, fn)
            magnitude_fp_loss = _masked_smooth_l1(remove_magnitude, rem_t, fp)
            magnitude_loss = 0.5 * (magnitude_fn_loss + magnitude_fp_loss)
            magnitude_fn_mae = _masked_mean((add_magnitude.detach() - add_t).abs()[:, 0], fn)
            magnitude_fp_mae = _masked_mean((remove_magnitude.detach() - rem_t).abs()[:, 0], fp)
            magnitude_fn_target_mean = _masked_mean(add_t[:, 0], fn)
            magnitude_fp_target_mean = _masked_mean(rem_t[:, 0], fp)

            # GT-support teacher HOW path: training-only, same computation graph,
            # never used for Val/Test deployment or checkpoint selection.
            fn4 = fn[:, None].to(anchor)
            fp4 = fp[:, None].to(anchor)
            anchor_logits = torch.logit(anchor.clamp(EPS, 1.0 - EPS))
            teacher_logits = anchor_logits + fn4 * add_magnitude - fp4 * remove_magnitude
            teacher_prob = torch.sigmoid(teacher_logits).clamp(EPS, 1.0 - EPS)
            teacher_loss, teacher_bce, teacher_dice_loss = _seg_loss(
                teacher_logits, teacher_prob, target, ce_weight, dice_weight
            )
            anchor_dice_pc = _dice_per_case(anchor[:, 0], target)
            teacher_dice_pc = _dice_per_case(teacher_prob[:, 0], target)
            teacher_dice = teacher_dice_pc.mean()
            teacher_gain = (teacher_dice_pc - anchor_dice_pc).mean()

            change = (final_prob[:, 0] - anchor[:, 0]).abs()
            preserve_loss = _masked_mean(change, correct)

            # Five interpretable standard-loss terms.  Core weights are
            # normalized so changing the decomposition does not arbitrarily
            # rescale the whole Stage-2 objective.
            weights = {
                "type": float(_cfg_get(m1, "GEOTR_STAGE2_TYPE_WEIGHT", 0.25)),
                "severity": float(_cfg_get(m1, "GEOTR_STAGE2_SEVERITY_WEIGHT", 0.25)),
                "magnitude": float(_cfg_get(m1, "GEOTR_STAGE2_MAGNITUDE_WEIGHT", 0.20)),
                "teacher": float(_cfg_get(m1, "GEOTR_STAGE2_TEACHER_SEG_WEIGHT", 0.20)),
                "deploy": float(_cfg_get(m1, "GEOTR_STAGE2_DEPLOY_SEG_WEIGHT", 0.10)),
            }
            core_norm = max(sum(max(v, 0.0) for v in weights.values()), 1.0e-8)
            stage2_loss = (
                max(weights["type"], 0.0) * typed_loss
                + max(weights["severity"], 0.0) * severity_loss
                + max(weights["magnitude"], 0.0) * magnitude_loss
                + max(weights["teacher"], 0.0) * teacher_loss
                + max(weights["deploy"], 0.0) * final_loss
            ) / core_norm
            preserve_w = float(_cfg_get(m1, "GEOTR_STAGE2_PRESERVE_WEIGHT", 0.02))
            stage2_loss = stage2_loss + preserve_w * preserve_loss

            # Deployment direction is determined by severity*logit-magnitude,
            # not by class-weighted q.
            add_score = (severity_fn.detach() * add_magnitude.detach())[:, 0]
            remove_score = (severity_fp.detach() * remove_magnitude.detach())[:, 0]
            pred_add = add_score >= remove_score

            severity_error = (severity_fn.detach() + severity_fp.detach())[:, 0]
            severity_correct_mass = _masked_mean(severity_error, correct)
            severity_error_mass = _masked_mean(severity_error, fn | fp)
            severity_soft_precision = (
                severity_error * (fn | fp).to(severity_error)
            ).sum() / severity_error.sum().clamp_min(EPS)

            edit_precision, edit_error_mean, edit_correct_mean, edit_ratio = _edit_stats(
                final_prob.detach(), anchor, fn | fp, correct
            )
            stage2_gain_pc = _dice_per_case(final_prob[:, 0].detach(), target) - _dice_per_case(anchor[:, 0], target)
            stage2_benefit_rate = (stage2_gain_pc > 1.0e-8).float().mean()
            stage2_harm_rate = (stage2_gain_pc < -1.0e-8).float().mean()
            stage2_mean_benefit = _safe_ratio(
                stage2_gain_pc.clamp_min(0.0).sum(), (stage2_gain_pc > 1.0e-8).float().sum()
            )
            stage2_mean_harm = _safe_ratio(
                stage2_gain_pc.clamp_max(0.0).sum(), (stage2_gain_pc < -1.0e-8).float().sum()
            )

            if bool(_cfg_get(m1, "GEOTR_STAGE2_ORACLE_DIAGNOSTICS", True)):
                oracle = _v4d_oracle_diagnostics(
                    anchor, target, severity_fn.detach(), severity_fp.detach(),
                    add_magnitude.detach(), remove_magnitude.detach(),
                    add_t.detach(), rem_t.detach(),
                )

        fn_n = fn.float().sum().clamp_min(1.0)
        fp_n = fp.float().sum().clamp_min(1.0)
        err_n = (fn | fp).float().sum().clamp_min(1.0)
        direction = {
            "fn": (pred_add & fn).float().sum() / fn_n,
            "fp": ((~pred_add) & fp).float().sum() / fp_n,
            "all": ((pred_add & fn) | ((~pred_add) & fp)).float().sum() / err_n,
        }
    else:
        error_target = target.new_zeros(target.shape)
        typed_loss = error_loss = stage2_loss = zero

    # Historical aliases retained for trainer/logger compatibility.
    error_loss = typed_loss
    error_bce = typed_ce
    error_dice_loss = typed_dice

    # Causal optimization contract: A3 Transport gradient is identical to A1.
    if mode == "base":
        objective = final_loss * 0.0
    elif mode == "geometry":
        objective = geo_loss
    elif mode == "residual":
        objective = stage2_loss
    else:
        objective = geo_loss + stage2_loss

    smooth = _flow_smoothness(flow)
    smooth_w = float(_cfg_get(m1, "GEOTOPO_SMOOTHNESS_WEIGHT", 0.0))
    if mode in {"geometry", "full"} and smooth_w > 0.0:
        objective = objective + smooth_w * smooth

    compat = aux.get("mhcs_m1_distribution_log_var")
    if isinstance(compat, torch.Tensor):
        objective = objective + 0.0 * compat

    base_dice = _dice_per_case(base_prob[:, 0].detach(), target)
    geo_dice = _dice_per_case(geo_prob[:, 0].detach(), target)
    recon_base_dice = _dice_per_case(recon_base_prob[:, 0].detach(), target)
    recon_geo_dice = _dice_per_case(recon_geo_prob[:, 0].detach(), target)
    final_dice = _dice_per_case(final_prob[:, 0].detach(), target)
    flow_mag = torch.sqrt(flow.detach()[:, 0].square() + flow.detach()[:, 1].square() + 1e-12)

    pair_radius = int(_cfg_get(m1, "GEOTR_STAGE2_PAIR_RADIUS_PX", 5))
    base_pair = _pairing_stats(base_prob.detach(), target, pair_radius)
    geo_pair = _pairing_stats(geo_prob.detach(), target, pair_radius)

    diagnostics = {
        # Legacy keys required by the existing trainer/logger.
        "mhcs_objective": objective.detach(),
        "mhcs_m1_objective": objective.detach(),
        "mhcs_m2_objective": stage2_loss.detach() if mode in {"residual", "full"} else final_loss.detach(),
        "mhcs_final_gain": (final_dice - base_dice).mean().detach(),
        "mhcs_base_dice": base_dice.mean().detach(),
        "mhcs_final_dice": final_dice.mean().detach(),
        "mhcs_best_single_dice": torch.maximum(base_dice, final_dice).mean().detach(),
        "mhcs_generated_best_dice": final_dice.mean().detach(),
        "mhcs_distribution_mean_dice": final_dice.mean().detach(),
        "mhcs_surface_hard_envelope_dice": final_dice.mean().detach(),
        "mhcs_global_selected_dice": final_dice.mean().detach(),
        "mhcs_local_dice": final_dice.mean().detach(),
        "mhcs_candidate_mean_gain": (final_dice - base_dice).mean().detach(),
        "mhcs_candidate_harm_rate": (final_dice < base_dice).float().mean().detach(),

        "geotopo_final_loss": final_loss.detach(),
        "geotopo_geometry_loss": geo_loss.detach(),
        "geotopo_bce_loss": final_bce.detach(),
        "geotopo_dice_loss": final_dice_loss.detach(),
        "geotopo_geometry_bce_loss": geo_bce.detach(),
        "geotopo_geometry_dice_loss": geo_dice_loss.detach(),
        "geotopo_error_localization_loss": error_loss.detach(),
        "geotopo_error_localization_bce": error_bce.detach(),
        "geotopo_error_localization_dice_loss": error_dice_loss.detach(),
        "geotopo_error_target_rate": error_target.mean().detach(),
        "geotopo_error_precision": error_stats["precision"].detach(),
        "geotopo_error_recall": error_stats["recall"].detach(),
        "geotopo_error_f1": error_stats["f1"].detach(),

        # V4C/V4D typed-localizer diagnostics.
        "geotr_v4c_typed_ce": typed_ce.detach(),
        "geotr_v4c_typed_dice_loss": typed_dice.detach(),
        "geotr_v4c_fn_dice_loss": fn_dice_loss.detach(),
        "geotr_v4c_fp_dice_loss": fp_dice_loss.detach(),
        "geotr_v4c_fn_target_rate": fn.float().mean().detach(),
        "geotr_v4c_fp_target_rate": fp.float().mean().detach(),
        "geotr_v4c_correct_target_rate": correct.float().mean().detach(),
        "geotr_v4c_fn_precision": typed_stats["fn_precision"].detach(),
        "geotr_v4c_fn_recall": typed_stats["fn_recall"].detach(),
        "geotr_v4c_fn_f1": typed_stats["fn_f1"].detach(),
        "geotr_v4c_fp_precision": typed_stats["fp_precision"].detach(),
        "geotr_v4c_fp_recall": typed_stats["fp_recall"].detach(),
        "geotr_v4c_fp_f1": typed_stats["fp_f1"].detach(),
        "geotr_v4c_typed_macro_f1": typed_stats["macro_f1"].detach(),
        "geotr_v4c_preserve_loss": preserve_loss.detach(),
        "geotr_v4c_q_false_positive_mass_correct": q_correct_fp_mass.detach(),
        "geotr_v4c_q_true_error_mass": q_error_tp_mass.detach(),
        "geotr_v4c_q_soft_precision": q_soft_precision.detach(),
        "geotr_v4c_fn_direction_accuracy": direction["fn"].detach(),
        "geotr_v4c_fp_direction_accuracy": direction["fp"].detach(),
        "geotr_v4c_direction_accuracy": direction["all"].detach(),

        # V4E operator-consistency diagnostics.
        "geotr_v4e_enabled": final_prob.new_tensor(float(v4e)).detach(),
        "geotr_v4e_predicted_support_rate": ((typed_prob.detach().argmax(dim=1) != 0).float().mean() if isinstance(typed_prob, torch.Tensor) else zero).detach(),
        "geotr_v4e_true_support_rate": (fn | fp).float().mean().detach(),

        # V4F selective-intervention diagnostics.
        "geotr_v4f_enabled": final_prob.new_tensor(float(v4f)).detach(),
        "geotr_v4f_policy_presence_loss": v4f_policy_presence_loss.detach(),
        "geotr_v4f_policy_direction_loss": v4f_policy_direction_loss.detach(),
        "geotr_v4f_policy_loss": v4f_policy_loss.detach(),
        "geotr_v4f_proposal_recall": v4f_proposal_recall.detach(),
        "geotr_v4f_proposal_precision": v4f_proposal_precision.detach(),
        "geotr_v4f_execution_precision": v4f_execution_precision.detach(),
        "geotr_v4f_execution_recall": v4f_execution_recall.detach(),
        "geotr_v4f_direction_accuracy": v4f_direction_accuracy.detach(),
        "geotr_v4f_candidate_rate": v4f_candidate_rate.detach(),
        "geotr_v4f_execution_rate": v4f_execution_rate.detach(),
        "geotr_v4f_teacher_gain": v4f_teacher_gain.detach(),
        "geotr_v4f_teacher_dice": v4f_teacher_dice.detach(),
        "geotr_v4f_soft_gain": v4f_soft_gain.detach(),
        "geotr_v4f_hard_gain": v4f_hard_gain.detach(),
        "geotr_v4f_true_action_dose": v4f_true_action_dose.detach(),
        "geotr_v4f_false_action_dose": v4f_false_action_dose.detach(),

        # V4D root-fix diagnostics.
        "geotr_v4d_enabled": final_prob.new_tensor(float(v4d)).detach(),
        "geotr_v4d_severity_loss": severity_loss.detach(),
        "geotr_v4d_severity_fn_loss": severity_fn_loss.detach(),
        "geotr_v4d_severity_fp_loss": severity_fp_loss.detach(),
        "geotr_v4d_severity_fn_mae": severity_fn_mae.detach(),
        "geotr_v4d_severity_fp_mae": severity_fp_mae.detach(),
        "geotr_v4d_severity_fn_error_mae": severity_fn_error_mae.detach(),
        "geotr_v4d_severity_fp_error_mae": severity_fp_error_mae.detach(),
        "geotr_v4d_severity_true_error_mass": severity_error_mass.detach(),
        "geotr_v4d_severity_correct_mass": severity_correct_mass.detach(),
        "geotr_v4d_severity_soft_precision": severity_soft_precision.detach(),
        "geotr_v4d_magnitude_loss": magnitude_loss.detach(),
        "geotr_v4d_magnitude_fn_loss": magnitude_fn_loss.detach(),
        "geotr_v4d_magnitude_fp_loss": magnitude_fp_loss.detach(),
        "geotr_v4d_magnitude_fn_mae": magnitude_fn_mae.detach(),
        "geotr_v4d_magnitude_fp_mae": magnitude_fp_mae.detach(),
        "geotr_v4d_magnitude_fn_target_mean": magnitude_fn_target_mean.detach(),
        "geotr_v4d_magnitude_fp_target_mean": magnitude_fp_target_mean.detach(),
        "geotr_v4d_teacher_how_loss": teacher_loss.detach(),
        "geotr_v4d_teacher_how_bce": teacher_bce.detach(),
        "geotr_v4d_teacher_how_dice_loss": teacher_dice_loss.detach(),
        "geotr_v4d_teacher_how_dice": teacher_dice.detach(),
        "geotr_v4d_teacher_how_gain": teacher_gain.detach(),
        "geotr_v4d_edit_precision": edit_precision.detach(),
        "geotr_v4d_edit_error_mean": edit_error_mean.detach(),
        "geotr_v4d_edit_correct_mean": edit_correct_mean.detach(),
        "geotr_v4d_edit_error_correct_ratio": edit_ratio.detach(),
        "geotr_v4d_stage2_benefit_rate": stage2_benefit_rate.detach(),
        "geotr_v4d_stage2_harm_rate": stage2_harm_rate.detach(),
        "geotr_v4d_stage2_mean_benefit": stage2_mean_benefit.detach(),
        "geotr_v4d_stage2_mean_harm": stage2_mean_harm.detach(),

        # GT-only root-cause oracles. These never enter objective.
        "geotr_v4c_oracle_where_dice": oracle["oracle_where_dice"].detach(),
        "geotr_v4c_oracle_sign_dice": oracle["oracle_sign_dice"].detach(),
        "geotr_v4c_oracle_typed_dice": oracle["oracle_typed_dice"].detach(),
        "geotr_v4c_oracle_full_dice": oracle["oracle_full_dice"].detach(),
        "geotr_v4c_oracle_where_gain": oracle["oracle_where_gain"].detach(),
        "geotr_v4c_oracle_sign_gain": oracle["oracle_sign_gain"].detach(),
        "geotr_v4c_oracle_typed_gain": oracle["oracle_typed_gain"].detach(),
        "geotr_v4c_oracle_full_gain": oracle["oracle_full_gain"].detach(),

        # Pairing is explicitly a local co-occurrence proxy, not an OT claim.
        "geotr_v4c_base_fn_rate": base_pair["fn_rate"].detach(),
        "geotr_v4c_base_fp_rate": base_pair["fp_rate"].detach(),
        "geotr_v4c_base_paired_error_fraction": base_pair["paired_error_fraction"].detach(),
        "geotr_v4c_base_unmatched_error_fraction": base_pair["unmatched_error_fraction"].detach(),
        "geotr_v4c_geo_fn_rate": geo_pair["fn_rate"].detach(),
        "geotr_v4c_geo_fp_rate": geo_pair["fp_rate"].detach(),
        "geotr_v4c_geo_paired_fn_fraction": geo_pair["paired_fn_fraction"].detach(),
        "geotr_v4c_geo_paired_fp_fraction": geo_pair["paired_fp_fraction"].detach(),
        "geotr_v4c_geo_paired_error_fraction": geo_pair["paired_error_fraction"].detach(),
        "geotr_v4c_geo_unmatched_error_fraction": geo_pair["unmatched_error_fraction"].detach(),
        "geotr_v4c_geometry_paired_error_reduction": (
            base_pair["paired_error_fraction"] - geo_pair["paired_error_fraction"]
        ).detach(),

        "geotopo_base_dice": base_dice.mean().detach(),
        "geotopo_geometry_dice": geo_dice.mean().detach(),
        "geotopo_residual_dice": recon_base_dice.mean().detach(),
        "geotopo_reconstruction_after_geometry_dice": recon_geo_dice.mean().detach(),
        "geotopo_final_dice": final_dice.mean().detach(),
        "geotopo_final_gain": (final_dice - base_dice).mean().detach(),
        "geotopo_geometry_gain": (geo_dice - base_dice).mean().detach(),
        "geotopo_residual_gain": (recon_base_dice - base_dice).mean().detach(),
        "geotopo_reconstruction_gain_after_geometry": (recon_geo_dice - geo_dice).mean().detach(),
        "geotopo_full_gain_over_geometry": (final_dice - geo_dice).mean().detach(),
        "geotopo_flow_mean_px": flow_mag.mean().detach(),
        "geotopo_flow_rms_px": flow_mag.square().mean().sqrt().detach(),
        "geotopo_flow_max_px": flow_mag.max().detach(),
        "geotopo_flow_smoothness": smooth.detach(),
        "geotopo_abs_change": (final_prob.detach() - base_prob.detach()).abs().mean(),
    }
    for key in (
        "geotopo_flow_jacobian_mean", "geotopo_flow_folding_fraction",
        "geotopo_residual_rms", "geotopo_reconstruction_strength_mean_abs",
        "geotopo_reconstruction_strength_p95_proxy", "geotopo_reconstruction_saturation_fraction",
        "geotopo_error_probability_mean", "geotopo_error_probability_p95",
        "geotopo_q_correct_mean", "geotopo_q_fn_mean", "geotopo_q_fp_mean",
        "geotr_v4d_severity_fn_mean", "geotr_v4d_severity_fp_mean",
        "geotopo_add_magnitude_mean", "geotopo_remove_magnitude_mean",
        "geotopo_geometry_abs_change", "geotopo_reconstruction_base_abs_change",
        "geotopo_reconstruction_after_geometry_abs_change",
        "mhcs_candidate_disagreement_mean", "mhcs_route_set_std_mean",
        "mhcs_route_raw_abs_disagreement_mean", "mhcs_candidate_probability_novelty_mean",
        "mhcs_global_adapter_delta_rms", "mhcs_local_adapter_delta_rms",
        "mhcs_context_gate", "mhcs_effective_rank", "mhcs_surface_nonbase_mass",
    ):
        value = aux.get(key)
        if isinstance(value, torch.Tensor):
            diagnostics[key] = value.detach().mean()
    return objective, diagnostics

@torch.no_grad()
def compute_geotr_v4c_validation_diagnostics(
    cfg,
    masks: torch.Tensor,
    pred: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Model-resolution GT-only V4C diagnostics for validation.

    The returned values are diagnostics only.  The function does not modify the
    prediction and is intentionally separate from checkpoint selection.
    """
    m1 = _cfg_get(cfg, "M1", None)
    mode = str(_cfg_get(m1, "GEOTOPO_MODE", "full")).strip().lower()
    if mode not in {"residual", "full"}:
        return {}

    target = _target_3d(masks)
    base = pred.get("geotopo_base_probs")
    geo = pred.get("geotopo_geometry_probs")
    prefix = "geotopo_reconstruction_after_geometry" if mode == "full" else "geotopo_reconstruction_base"
    q_fn = pred.get(prefix + "_q_fn")
    q_fp = pred.get(prefix + "_q_fp")
    add_magnitude = pred.get(prefix + "_add_magnitude")
    remove_magnitude = pred.get(prefix + "_remove_magnitude")
    typed_prob = pred.get(prefix + "_typed_probs")
    if not all(isinstance(x, torch.Tensor) for x in (base, geo, q_fn, q_fp, add_magnitude, remove_magnitude, typed_prob)):
        return {}

    anchor = geo if mode == "full" else base
    if anchor.ndim == 3:
        anchor = anchor[:, None]
    if target.shape[-2:] != anchor.shape[-2:]:
        target = F.interpolate(target[:, None].float(), size=anchor.shape[-2:], mode="nearest")[:, 0]
    target = target.to(anchor)

    cls_target, correct, fn, fp = _typed_targets(anchor, target)
    pred_cls = typed_prob.argmax(dim=1)
    fn_stats = _class_stats(pred_cls, fn, 1)
    fp_stats = _class_stats(pred_cls, fp, 2)
    q_error = (q_fn + q_fp).clamp(0.0, 1.0)[:, 0]
    err = fn | fp

    add_score = (q_fn * add_magnitude * (1.0 - anchor))[:, 0]
    remove_score = (q_fp * remove_magnitude * anchor)[:, 0]
    pred_add = add_score >= remove_score
    fn_n = fn.float().sum().clamp_min(1.0)
    fp_n = fp.float().sum().clamp_min(1.0)
    err_n = err.float().sum().clamp_min(1.0)

    oracle = _oracle_diagnostics(anchor, target, q_fn, q_fp, add_magnitude, remove_magnitude)
    radius = int(_cfg_get(m1, "GEOTR_STAGE2_PAIR_RADIUS_PX", 5))
    base_pair = _pairing_stats(base if base.ndim == 4 else base[:, None], target, radius)
    geo_pair = _pairing_stats(geo if geo.ndim == 4 else geo[:, None], target, radius)

    return {
        "val_geotr_v4c_error_rate": err.float().mean(),
        "val_geotr_v4c_fn_rate": fn.float().mean(),
        "val_geotr_v4c_fp_rate": fp.float().mean(),
        "val_geotr_v4c_fn_f1": fn_stats["f1"],
        "val_geotr_v4c_fp_f1": fp_stats["f1"],
        "val_geotr_v4c_typed_macro_f1": 0.5 * (fn_stats["f1"] + fp_stats["f1"]),
        "val_geotr_v4c_q_false_positive_mass_correct": _masked_mean(q_error, correct),
        "val_geotr_v4c_q_true_error_mass": _masked_mean(q_error, err),
        "val_geotr_v4c_q_soft_precision": (q_error * err.to(q_error)).sum() / q_error.sum().clamp_min(EPS),
        "val_geotr_v4c_fn_direction_accuracy": (pred_add & fn).float().sum() / fn_n,
        "val_geotr_v4c_fp_direction_accuracy": ((~pred_add) & fp).float().sum() / fp_n,
        "val_geotr_v4c_direction_accuracy": ((pred_add & fn) | ((~pred_add) & fp)).float().sum() / err_n,
        "val_geotr_v4c_oracle_where_gain": oracle["oracle_where_gain"],
        "val_geotr_v4c_oracle_sign_gain": oracle["oracle_sign_gain"],
        "val_geotr_v4c_oracle_typed_gain": oracle["oracle_typed_gain"],
        "val_geotr_v4c_oracle_full_gain": oracle["oracle_full_gain"],
        "val_geotr_v4c_base_paired_error_fraction": base_pair["paired_error_fraction"],
        "val_geotr_v4c_geo_paired_error_fraction": geo_pair["paired_error_fraction"],
        "val_geotr_v4c_geometry_paired_error_reduction": base_pair["paired_error_fraction"] - geo_pair["paired_error_fraction"],
    }

@torch.no_grad()
def compute_geotr_v4d_validation_diagnostics(
    cfg,
    masks: torch.Tensor,
    pred: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Validation-only V4D diagnostics with Stage2-vs-anchor case statistics.

    All GT use is diagnostic.  Nothing here changes predictions or checkpoint
    selection.  Count fields are intentionally returned so ``train.py`` can
    aggregate global pixel F1 exactly rather than averaging per-batch F1.
    """
    m1 = _cfg_get(cfg, "M1", None)
    if not bool(_cfg_get(m1, "GEOTR_V4D_ROOT_FIX_ENABLED", False)):
        return {}
    mode = str(_cfg_get(m1, "GEOTOPO_MODE", "full")).strip().lower()
    if mode not in {"residual", "full"}:
        return {}

    target = _target_3d(masks)
    base = pred.get("geotopo_base_probs")
    geo = pred.get("geotopo_geometry_probs")
    final = pred.get("geotopo_final_probs")
    prefix = "geotopo_reconstruction_after_geometry" if mode == "full" else "geotopo_reconstruction_base"
    typed_prob = pred.get(prefix + "_typed_probs")
    severity_fn = pred.get(prefix + "_severity_fn")
    severity_fp = pred.get(prefix + "_severity_fp")
    add_magnitude = pred.get(prefix + "_add_magnitude")
    remove_magnitude = pred.get(prefix + "_remove_magnitude")
    if not all(isinstance(x, torch.Tensor) for x in (
        base, geo, final, typed_prob, severity_fn, severity_fp, add_magnitude, remove_magnitude
    )):
        return {}

    anchor = geo if mode == "full" else base
    if anchor.ndim == 3:
        anchor = anchor[:, None]
    if final.ndim == 3:
        final = final[:, None]
    if target.shape[-2:] != anchor.shape[-2:]:
        target = F.interpolate(target[:, None].float(), size=anchor.shape[-2:], mode="nearest")[:, 0]
    target = target.to(anchor)

    _, correct, fn, fp = _typed_targets(anchor, target)
    err = fn | fp
    pred_cls = typed_prob.argmax(dim=1)
    pred_fn = pred_cls == 1
    pred_fp = pred_cls == 2

    fn_tp = (pred_fn & fn).float().sum()
    fn_fp_count = (pred_fn & ~fn).float().sum()
    fn_fn_count = ((~pred_fn) & fn).float().sum()
    fp_tp = (pred_fp & fp).float().sum()
    fp_fp_count = (pred_fp & ~fp).float().sum()
    fp_fn_count = ((~pred_fp) & fp).float().sum()
    fn_stats = _class_stats(pred_cls, fn, 1)
    fp_stats = _class_stats(pred_cls, fp, 2)

    margin = float(_cfg_get(m1, "GEOTR_STAGE2_TARGET_MARGIN", 0.05))
    max_step = float(_cfg_get(m1, "GEOTR_STAGE2_MAX_LOGIT_STEP", 4.0))
    sev_fn_t, sev_fp_t, add_t, rem_t = _v4d_targets(anchor, target, fn, fp, margin, max_step)

    # Teacher-HOW diagnostic uses exact hard support but learned magnitude.
    fn4 = fn[:, None].to(anchor)
    fp4 = fp[:, None].to(anchor)
    z = torch.logit(anchor.clamp(EPS, 1.0 - EPS))
    teacher_logits = z + fn4 * add_magnitude - fp4 * remove_magnitude
    teacher_prob = torch.sigmoid(teacher_logits).clamp(EPS, 1.0 - EPS)

    anchor_dice = _dice_per_case(anchor[:, 0], target)
    final_dice = _dice_per_case(final[:, 0], target)
    teacher_dice = _dice_per_case(teacher_prob[:, 0], target)
    gain = final_dice - anchor_dice
    teacher_gain = teacher_dice - anchor_dice

    benefit = gain > 1.0e-8
    harm = gain < -1.0e-8
    benefit_n = benefit.float().sum()
    harm_n = harm.float().sum()
    mean_benefit = _safe_ratio(gain.clamp_min(0.0).sum(), benefit_n)
    mean_harm = _safe_ratio(gain.clamp_max(0.0).sum(), harm_n)

    severity_error = (severity_fn + severity_fp)[:, 0]
    edit_precision, edit_error_mean, edit_correct_mean, edit_ratio = _edit_stats(final, anchor, err, correct)
    oracle = _v4d_oracle_diagnostics(
        anchor, target, severity_fn, severity_fp, add_magnitude, remove_magnitude, add_t, rem_t
    )

    return {
        "val_geotr_v4d_error_rate": err.float().mean(),
        "val_geotr_v4d_fn_rate": fn.float().mean(),
        "val_geotr_v4d_fp_rate": fp.float().mean(),
        "val_geotr_v4d_fn_f1": fn_stats["f1"],
        "val_geotr_v4d_fp_f1": fp_stats["f1"],
        "val_geotr_v4d_typed_macro_f1": 0.5 * (fn_stats["f1"] + fp_stats["f1"]),
        "val_geotr_v4d_fn_tp_count": fn_tp,
        "val_geotr_v4d_fn_fp_count": fn_fp_count,
        "val_geotr_v4d_fn_fn_count": fn_fn_count,
        "val_geotr_v4d_fp_tp_count": fp_tp,
        "val_geotr_v4d_fp_fp_count": fp_fp_count,
        "val_geotr_v4d_fp_fn_count": fp_fn_count,
        "val_geotr_v4d_severity_true_error_mass": _masked_mean(severity_error, err),
        "val_geotr_v4d_severity_correct_mass": _masked_mean(severity_error, correct),
        "val_geotr_v4d_severity_soft_precision": (
            severity_error * err.to(severity_error)
        ).sum() / severity_error.sum().clamp_min(EPS),
        "val_geotr_v4d_severity_fn_mae": (severity_fn - sev_fn_t).abs().mean(),
        "val_geotr_v4d_severity_fp_mae": (severity_fp - sev_fp_t).abs().mean(),
        "val_geotr_v4d_magnitude_fn_mae": _masked_mean((add_magnitude - add_t).abs()[:, 0], fn),
        "val_geotr_v4d_magnitude_fp_mae": _masked_mean((remove_magnitude - rem_t).abs()[:, 0], fp),
        "val_geotr_v4d_magnitude_fn_target_mean": _masked_mean(add_t[:, 0], fn),
        "val_geotr_v4d_magnitude_fp_target_mean": _masked_mean(rem_t[:, 0], fp),
        "val_geotr_v4d_teacher_how_gain": teacher_gain.mean(),
        "val_geotr_v4d_teacher_how_dice": teacher_dice.mean(),
        "val_geotr_v4d_stage2_gain": gain.mean(),
        "val_geotr_v4d_stage2_benefit_rate": benefit.float().mean(),
        "val_geotr_v4d_stage2_harm_rate": harm.float().mean(),
        "val_geotr_v4d_stage2_mean_benefit": mean_benefit,
        "val_geotr_v4d_stage2_mean_harm": mean_harm,
        "val_geotr_v4d_edit_precision": edit_precision,
        "val_geotr_v4d_edit_error_mean": edit_error_mean,
        "val_geotr_v4d_edit_correct_mean": edit_correct_mean,
        "val_geotr_v4d_edit_error_correct_ratio": edit_ratio,
        "val_geotr_v4d_oracle_where_gain": oracle["oracle_where_gain"],
        "val_geotr_v4d_oracle_sign_gain": oracle["oracle_sign_gain"],
        "val_geotr_v4d_oracle_typed_gain": oracle["oracle_typed_gain"],
        "val_geotr_v4d_oracle_target_magnitude_gain": oracle["oracle_full_gain"],
    }



def compute_geotr_v4e_validation_diagnostics(cfg, masks, pred):
    """Validation-only operator-consistency diagnostics for GEOTR-V4E."""
    m1 = _cfg_get(cfg, "M1", None)
    if not bool(_cfg_get(m1, "GEOTR_V4E_OPERATOR_CONSISTENT_ENABLED", False)):
        return {}
    mode = str(_cfg_get(m1, "GEOTOPO_MODE", "full")).lower()
    if mode not in {"residual", "full"}:
        return {}
    target = _target_3d(masks)
    base = pred.get("geotopo_base_probs")
    geo = pred.get("geotopo_geometry_probs")
    final = pred.get("geotopo_final_probs")
    prefix = "geotopo_reconstruction_after_geometry" if mode == "full" else "geotopo_reconstruction_base"
    typed_prob = pred.get(prefix + "_typed_probs")
    add_magnitude = pred.get(prefix + "_add_magnitude")
    remove_magnitude = pred.get(prefix + "_remove_magnitude")
    if not all(isinstance(x, torch.Tensor) for x in (base, geo, final, typed_prob, add_magnitude, remove_magnitude)):
        return {}
    anchor = geo if mode == "full" else base
    if anchor.ndim == 3: anchor = anchor[:, None]
    if final.ndim == 3: final = final[:, None]
    if target.shape[-2:] != anchor.shape[-2:]:
        target = F.interpolate(target[:, None].float(), size=anchor.shape[-2:], mode="nearest")[:, 0]
    target = target.to(anchor)
    _, correct, fn, fp = _typed_targets(anchor, target)
    err = fn | fp
    pred_cls = typed_prob.argmax(dim=1)
    pred_fn = pred_cls == 1
    pred_fp = pred_cls == 2
    fn_stats = _class_stats(pred_cls, fn, 1)
    fp_stats = _class_stats(pred_cls, fp, 2)
    margin = float(_cfg_get(m1, "GEOTR_STAGE2_TARGET_MARGIN", 0.05))
    max_step = float(_cfg_get(m1, "GEOTR_STAGE2_MAX_LOGIT_STEP", 8.0))
    add_t, rem_t, add_cap_t, rem_cap_t = _v4e_targets(anchor, target, fn, fp, margin, max_step)
    fn4 = fn[:, None].to(anchor); fp4 = fp[:, None].to(anchor)
    teacher_prob, _ = _compose_v4e(anchor, fn4, fp4, add_magnitude, remove_magnitude)
    anchor_d = _dice_per_case(anchor[:,0], target)
    final_d = _dice_per_case(final[:,0], target)
    teacher_d = _dice_per_case(teacher_prob[:,0], target)
    gain = final_d-anchor_d; tgain=teacher_d-anchor_d
    benefit = gain>1e-8; harm=gain<-1e-8
    oracle = _v4e_oracle_diagnostics(anchor,target,pred_fn,pred_fp,add_magnitude,remove_magnitude,add_t,rem_t)
    edit = (final-anchor).abs()[:,0]
    # Three disjoint edit regions: hard error, low-margin correct, confident correct.
    p = anchor[:,0]
    gt = target>=0.5
    margin_correct = correct & (((gt) & (p < 0.55)) | ((~gt) & (p > 0.45)))
    confident_correct = correct & ~margin_correct
    def mm(x,m): return _masked_mean(x,m)
    cap = float(max_step)
    return {
        "val_geotr_v4e_stage2_soft_gain": gain.mean(),
        "val_geotr_v4e_stage2_soft_benefit_rate": benefit.float().mean(),
        "val_geotr_v4e_stage2_soft_harm_rate": harm.float().mean(),
        "val_geotr_v4e_teacher_how_gain": tgain.mean(),
        "val_geotr_v4e_global_typed_macro_f1": 0.5*(fn_stats["f1"]+fp_stats["f1"]),
        "val_geotr_v4e_fn_f1": fn_stats["f1"], "val_geotr_v4e_fp_f1": fp_stats["f1"],
        "val_geotr_v4e_true_support_rate": err.float().mean(),
        "val_geotr_v4e_predicted_support_rate": (pred_fn|pred_fp).float().mean(),
        "val_geotr_v4e_support_precision": ((pred_fn|pred_fp)&err).float().sum()/(pred_fn|pred_fp).float().sum().clamp_min(EPS),
        "val_geotr_v4e_support_recall": ((pred_fn|pred_fp)&err).float().sum()/err.float().sum().clamp_min(EPS),
        "val_geotr_v4e_edit_hard_error_mean": mm(edit,err),
        "val_geotr_v4e_edit_margin_correct_mean": mm(edit,margin_correct),
        "val_geotr_v4e_edit_confident_correct_mean": mm(edit,confident_correct),
        "val_geotr_v4e_magnitude_fn_mae": mm((add_magnitude-add_t).abs()[:,0],fn),
        "val_geotr_v4e_magnitude_fp_mae": mm((remove_magnitude-rem_t).abs()[:,0],fp),
        "val_geotr_v4e_offsupport_add_mean": mm(add_magnitude[:,0],~fn),
        "val_geotr_v4e_offsupport_remove_mean": mm(remove_magnitude[:,0],~fp),
        "val_geotr_v4e_predicted_magnitude_cap_fraction": (((add_magnitude>=0.99*cap)|(remove_magnitude>=0.99*cap)).float().mean()),
        "val_geotr_v4e_target_fn_cap_fraction": add_cap_t.sum()/fn.float().sum().clamp_min(1.0),
        "val_geotr_v4e_target_fp_cap_fraction": rem_cap_t.sum()/fp.float().sum().clamp_min(1.0),
        "val_geotr_v4e_oracle_support_gain": oracle["oracle_support_gain"],
        "val_geotr_v4e_oracle_magnitude_gain": oracle["oracle_magnitude_gain"],
        "val_geotr_v4e_oracle_full_gain": oracle["oracle_full_gain"],
    }



def compute_geotr_v4f_validation_diagnostics(cfg, masks, pred):
    """Validation-only V4F proposal/execution/dose and risk-coverage audit.

    No GT quantity enters deployment. Counterfactuals below are diagnostic only.
    """
    m1 = _cfg_get(cfg, "M1", None)
    if not bool(_cfg_get(m1, "GEOTR_V4F_SELECTIVE_INTERVENTION_ENABLED", False)):
        return {}
    mode = str(_cfg_get(m1, "GEOTOPO_MODE", "full")).lower()
    if mode not in {"residual", "full"}:
        return {}
    target = _target_3d(masks)
    base = pred.get("geotopo_base_probs")
    geo = pred.get("geotopo_geometry_probs")
    final = pred.get("geotopo_final_probs")
    prefix = "geotopo_reconstruction_after_geometry" if mode == "full" else "geotopo_reconstruction_base"
    typed_prob = pred.get(prefix + "_typed_probs")
    cand = pred.get(prefix + "_v4f_proposal_candidate_mask")
    edit_prob = pred.get(prefix + "_v4f_policy_edit_prob")
    dir_prob = pred.get(prefix + "_v4f_policy_direction_prob")
    dose_add = pred.get(prefix + "_add_magnitude")
    dose_remove = pred.get(prefix + "_remove_magnitude")
    soft_prob = pred.get(prefix + "_v4f_soft_prob")
    hard_prob = pred.get(prefix + "_v4f_hard_prob")
    if not all(isinstance(x, torch.Tensor) for x in (
        base, geo, final, typed_prob, cand, edit_prob, dir_prob,
        dose_add, dose_remove, soft_prob, hard_prob,
    )):
        return {}
    anchor = geo if mode == "full" else base
    if anchor.ndim == 3: anchor = anchor[:, None]
    if target.shape[-2:] != anchor.shape[-2:]:
        target = F.interpolate(target[:, None].float(), size=anchor.shape[-2:], mode="nearest")[:, 0]
    target = target.to(anchor)
    _, correct, fn, fp = _typed_targets(anchor, target)
    err = fn | fp
    anchor_d = _dice_per_case(anchor[:, 0], target)
    hard_d = _dice_per_case(hard_prob[:, 0], target)
    soft_d = _dice_per_case(soft_prob[:, 0], target)
    threshold = float(_cfg_get(m1, "GEOTR_V4F_EXECUTION_THRESHOLD", 0.65))
    stats = _v4f_policy_stats(edit_prob, dir_prob, cand, fn, fp, threshold)

    teacher_prob, _ = _compose_v4f_teacher(anchor, fn, fp, dose_add, dose_remove)
    teacher_d = _dice_per_case(teacher_prob[:, 0], target)

    hard_edit = (hard_prob[:, 0] - anchor[:, 0]).abs() > 1.0e-8
    hard_add = hard_prob[:, 0] > anchor[:, 0] + 1.0e-8
    hard_remove = hard_prob[:, 0] < anchor[:, 0] - 1.0e-8
    direction_valid = (hard_add & (target >= 0.5)) | (hard_remove & (target < 0.5))
    repair_valid = (hard_add & fn) | (hard_remove & fp)
    exec_n = hard_edit.float().sum()
    direction_precision = _safe_ratio((hard_edit & direction_valid).float().sum(), exec_n)
    repair_precision = _safe_ratio((hard_edit & repair_valid).float().sum(), exec_n)

    # Counterfactual 1: keep predicted candidate/policy/dose but suppress only
    # hard false actions using GT. This isolates false-execution harm without
    # changing dose on true executed actions.
    suppress_false = torch.where((hard_edit & ~repair_valid)[:, None], anchor, hard_prob)
    suppress_false_d = _dice_per_case(suppress_false[:, 0], target)

    # Counterfactual 2: exact GT support with learned bounded dose (dose ceiling).
    gt_support_prob, _ = _compose_v4f_teacher(anchor, fn, fp, dose_add, dose_remove)
    gt_support_d = _dice_per_case(gt_support_prob[:, 0], target)

    # Candidate recall ceiling: if GT error is outside proposal pool, execution
    # cannot recover it.  Report separately from policy recall.
    cand_bool = cand[:, 0] >= 0.5
    proposal_recall = _safe_ratio((cand_bool & err).float().sum(), err.float().sum())

    out = {
        "val_geotr_v4f_modelres_soft_gain": (soft_d - anchor_d).mean(),
        "val_geotr_v4f_modelres_hard_gain": (hard_d - anchor_d).mean(),
        "val_geotr_v4f_teacher_dose_gain": (teacher_d - anchor_d).mean(),
        "val_geotr_v4f_proposal_recall": proposal_recall,
        "val_geotr_v4f_proposal_precision": stats["proposal_precision"],
        "val_geotr_v4f_candidate_rate": stats["candidate_rate"],
        "val_geotr_v4f_execution_precision_repair": repair_precision,
        "val_geotr_v4f_execution_precision_direction": direction_precision,
        "val_geotr_v4f_execution_recall": stats["recall"],
        "val_geotr_v4f_execution_rate": stats["execution_rate"],
        "val_geotr_v4f_direction_accuracy_on_true_exec": stats["direction_accuracy"],
        "val_geotr_v4f_true_action_dose": _masked_mean(0.5*(dose_add[:,0]+dose_remove[:,0]), hard_edit & err),
        "val_geotr_v4f_false_action_dose": _masked_mean(0.5*(dose_add[:,0]+dose_remove[:,0]), hard_edit & ~err),
        "val_geotr_v4f_false_action_suppression_gain": (suppress_false_d - hard_d).mean(),
        "val_geotr_v4f_gt_support_pred_dose_gain": (gt_support_d - anchor_d).mean(),
    }
    # Risk/coverage curve over execution threshold; proposal coverage is fixed.
    thresholds = _cfg_get(m1, "GEOTR_V4F_DIAG_THRESHOLDS", [0.30,0.40,0.50,0.60,0.65,0.70,0.80,0.90])
    if isinstance(thresholds, str):
        thresholds = [float(x) for x in thresholds.split(",") if x.strip()]
    for tau in thresholds:
        tau = float(tau)
        hp = _compose_v4f_policy(anchor, cand, edit_prob, dir_prob, dose_add, dose_remove, hard=True, threshold=tau)
        d = _dice_per_case(hp[:,0], target)
        edit = (hp[:,0]-anchor[:,0]).abs() > 1.0e-8
        add = hp[:,0] > anchor[:,0] + 1.0e-8
        rem = hp[:,0] < anchor[:,0] - 1.0e-8
        repair = (add & fn) | (rem & fp)
        precision = _safe_ratio((edit & repair).float().sum(), edit.float().sum())
        recall = _safe_ratio((edit & repair).float().sum(), err.float().sum())
        tag = str(int(round(tau*100))).zfill(2)
        out[f"val_geotr_v4f_tau{tag}_gain"] = (d-anchor_d).mean()
        out[f"val_geotr_v4f_tau{tag}_precision"] = precision
        out[f"val_geotr_v4f_tau{tag}_recall"] = recall
        out[f"val_geotr_v4f_tau{tag}_execution_rate"] = edit.float().mean()
    return out



def _compute_c2r_validation_diagnostics(cfg, masks, pred):
    """Validation-only GT audit for the GT-free C2R deployable forward.

    Canonical C2R-v2 diagnostics deliberately separate WHERE (context ROI),
    WHAT (canonical reconstruction), VERIFY (counterfactual agreement/spread),
    and COMMIT (atomic component intervention).  All GT use is audit-only.
    """
    m1 = _cfg_get(cfg, "M1", None)
    mode = str(_cfg_get(m1, "GEOTOPO_MODE", "full")).strip().lower()
    if mode not in {"residual", "full"}:
        return {}
    target = _target_3d(masks)
    base = pred.get("geotopo_base_probs")
    geo = pred.get("geotopo_geometry_probs")
    final = pred.get("geotopo_final_probs")
    prefix = "geotopo_reconstruction_after_geometry" if mode == "full" else "geotopo_reconstruction_base"
    region = pred.get(prefix + "_c2r_region_mask")
    centers = pred.get(prefix + "_c2r_center_mask")
    views = pred.get(prefix + "_c2r_view_probs")
    mean_prob = pred.get(prefix + "_c2r_mean_prob")
    consensus = pred.get(prefix + "_c2r_consensus_mask")
    edit = pred.get(prefix + "_c2r_edit_mask")
    candidate = pred.get(prefix + "_c2r_candidate_mask", edit)
    commit = pred.get(prefix + "_c2r_commit_mask", edit)
    margin = pred.get(prefix + "_v4g_margin_uncertainty")
    mc_std = pred.get(prefix + "_v4g_mc_std_map")
    mc_dis = pred.get(prefix + "_v4g_mc_disagreement_map")
    entropy = pred.get(prefix + "_v4g_entropy_map")
    aefr_state_logits = pred.get(prefix + "_aefr_state_logits")
    aefr_ownership_logits = pred.get(prefix + "_aefr_ownership_logits")
    aefr_error_localizer_prob = pred.get(prefix + "_aefr_error_localizer_prob")
    aefr_edit_prob = pred.get(prefix + "_aefr_edit_prob")
    aefr_direction_prob = pred.get(prefix + "_aefr_direction_prob")
    aefr_signed_action = pred.get(prefix + "_aefr_signed_action")
    aefr_boundary_mask = pred.get(prefix + "_aefr_boundary_mask")
    aefr_boundary_magnitude = pred.get(prefix + "_aefr_boundary_magnitude_px")
    slr_selector_prob = pred.get(prefix + "_slr_selector_prob")
    slr_sdf_delta_full = pred.get(prefix + "_slr_sdf_delta_full")
    slr_sdf_absolute_full = pred.get(prefix + "_slr_sdf_absolute_full")
    slr_blend_weight_sum = pred.get(prefix + "_slr_blend_weight_sum")
    slr_pred_patch_probs = pred.get(prefix + "_slr_pred_patch_probs")
    slr_raw_patch_probs = pred.get(prefix + "_slr_raw_patch_probs")
    slr_pred_patch_action_support = pred.get(prefix + "_slr_pred_patch_action_support")
    slr_action_weight_full = pred.get(prefix + "_slr_action_weight_full")
    slr_action_region_mask = pred.get(prefix + "_slr_action_region_mask")
    slr_action_delta_full = pred.get(prefix + "_slr_action_delta_full")
    slr_positive_from_selector_fraction_raw = pred.get(prefix + "_slr_positive_from_selector_fraction")
    slr_pred_patch_valid = pred.get(prefix + "_slr_pred_patch_valid")
    slr_pred_patch_grid = pred.get(prefix + "_slr_pred_patch_grid")
    slr_pred_patch_anchor_prob = pred.get(prefix + "_slr_pred_patch_anchor_prob")
    slr_overlap_disagreement_raw = pred.get(prefix + "_slr_overlap_disagreement")
    slr_overlap_sdf_disagreement_raw = pred.get(prefix + "_slr_overlap_sdf_disagreement")
    # UCDRT-native outputs are optional because OCRA owns the non-UCDRT route.
    slr_ucdrt_enabled_raw = pred.get(prefix + "_slr_ucdrt_enabled")
    slr_ucdrt_state_logits = pred.get(prefix + "_slr_ucdrt_state_logits")
    slr_ucdrt_move_px = pred.get(prefix + "_slr_ucdrt_move_px")
    slr_ucdrt_interior_magnitude = pred.get(prefix + "_slr_ucdrt_interior_magnitude")
    slr_ucdrt_candidate_soft_probs = pred.get(prefix + "_slr_ucdrt_candidate_soft_probs")
    slr_ucdrt_utility_value = pred.get(prefix + "_slr_ucdrt_utility_value")
    slr_ucdrt_commit = pred.get(prefix + "_slr_ucdrt_commit")
    slr_ucdrt_boundary_displacement_full = pred.get(prefix + "_slr_ucdrt_boundary_displacement_full")
    slr_ucdrt_interior_action_full = pred.get(prefix + "_slr_ucdrt_interior_action_full")
    slr_ucdrt_r2_edit_logits = pred.get(prefix + "_slr_ucdrt_r2_edit_logits")
    slr_ucdrt_r2_edit_probs = pred.get(prefix + "_slr_ucdrt_r2_edit_probs")
    slr_ucdrt_r2_type_logits = pred.get(prefix + "_slr_ucdrt_r2_type_logits")
    slr_ucdrt_r2_move_bin_logits = pred.get(prefix + "_slr_ucdrt_r2_move_bin_logits")
    slr_ucdrt_r2_move_dictionary_probs = pred.get(prefix + "_slr_ucdrt_r2_move_dictionary_probs")
    slr_ucdrt_r2_add_dose = pred.get(prefix + "_slr_ucdrt_r2_add_dose")
    slr_ucdrt_r2_remove_dose = pred.get(prefix + "_slr_ucdrt_r2_remove_dose")
    slr_ucdrt_r2_hard_candidate_probs = pred.get(prefix + "_slr_ucdrt_r2_hard_candidate_probs")
    slr_ucdrt_r2_soft_candidate_probs = pred.get(prefix + "_slr_ucdrt_r2_soft_candidate_probs")
    slr_ucdrt_r2_critic_step_logits = pred.get(prefix + "_slr_ucdrt_r2_critic_step_logits")
    slr_ucdrt_r2_critic_step_values = pred.get(prefix + "_slr_ucdrt_r2_critic_step_values")
    slr_ucdrt_r2_critic_step_available = pred.get(prefix + "_slr_ucdrt_r2_critic_step_available")
    slr_ucdrt_r2_selection_step_index = pred.get(prefix + "_slr_ucdrt_r2_selection_step_index")
    slr_ucdrt_r2_candidate_full_num = pred.get(prefix + "_slr_ucdrt_r2_candidate_full_num")
    slr_ucdrt_r2_candidate_full_den = pred.get(prefix + "_slr_ucdrt_r2_candidate_full_den")
    slr_ucdrt_r2_selected_candidate_mask = pred.get(prefix + "_slr_ucdrt_r2_selected_candidate_mask")
    slr_pred_sdf_anchor = pred.get(prefix + "_slr_pred_sdf_anchor")
    if not all(isinstance(x, torch.Tensor) for x in (
        base, geo, final, region, centers, views, mean_prob, consensus, edit,
        candidate, commit, margin, mc_std, mc_dis, entropy,
    )):
        return {}
    anchor = geo if mode == "full" else base
    if anchor.ndim == 3:
        anchor = anchor[:, None]
    if target.shape[-2:] != anchor.shape[-2:]:
        target = F.interpolate(target[:, None].float(), size=anchor.shape[-2:], mode="nearest")[:, 0]
    target = target.to(anchor)
    gt = target >= 0.5
    base_h = base[:, 0].detach() >= 0.5
    anchor_h = anchor[:, 0].detach() >= 0.5
    final_h = final[:, 0].detach() >= 0.5
    err = anchor_h != gt

    # Geometry -> Stage2 interference matrix (GT is diagnostic-only).
    # In A3/full these four sets distinguish what the upstream Geometry stage
    # actually did before Stage2 acts.  In A2/residual, anchor == Base, so
    # G_fix/G_harm are exactly empty and provide a built-in negative control.
    g_fix = (base_h != gt) & (anchor_h == gt)      # Geometry repaired a Base error.
    g_harm = (base_h == gt) & (anchor_h != gt)     # Geometry introduced an error.
    g_miss = (base_h != gt) & (anchor_h != gt)     # Error remains after Geometry.
    g_keep = (base_h == gt) & (anchor_h == gt)     # Correct before and after Geometry.
    stage2_edit = final_h != anchor_h
    final_correct = final_h == gt

    def _masked_rate(mask: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        den = mask.float().sum()
        return (mask & event).float().sum() / den.clamp_min(1.0)

    # ``break`` and ``repair`` directly quantify correction interference.
    gfix_preserve_rate = _masked_rate(g_fix, final_correct)
    gfix_break_rate = _masked_rate(g_fix, ~final_correct)
    gharm_repair_rate = _masked_rate(g_harm, final_correct)
    gmiss_repair_rate = _masked_rate(g_miss, final_correct)
    gkeep_break_rate = _masked_rate(g_keep, ~final_correct)
    gfix_edit_rate = _masked_rate(g_fix, stage2_edit)
    gharm_edit_rate = _masked_rate(g_harm, stage2_edit)
    gmiss_edit_rate = _masked_rate(g_miss, stage2_edit)
    gkeep_edit_rate = _masked_rate(g_keep, stage2_edit)
    stage2_abs_dz = (
        torch.logit(final[:, 0].detach().clamp(EPS, 1.0 - EPS))
        - torch.logit(anchor[:, 0].detach().clamp(EPS, 1.0 - EPS))
    ).abs()
    gfix_action_abs = _masked_mean(stage2_abs_dz, g_fix)
    gharm_action_abs = _masked_mean(stage2_abs_dz, g_harm)
    gmiss_action_abs = _masked_mean(stage2_abs_dz, g_miss)
    gkeep_action_abs = _masked_mean(stage2_abs_dz, g_keep)
    region_b = region[:, 0].detach() > 0.5
    center_b = centers[:, 0].detach() > 0.5
    consensus_b = consensus[:, 0].detach() > 0.5

    # Root Autopsy reachability must match the active action coordinate.
    # PC2R / AEFR-E1/E2 use a fixed bounded logit step; AEFR-E3 uses a
    # hybrid coordinate where the boundary band is corrected by a bounded
    # normal displacement and the ROI interior uses an anchor-adaptive logit
    # residual whose magnitude can cross any finite factual logit.
    anchor_logit_map = torch.logit(anchor[:, 0].detach().clamp(EPS, 1.0 - EPS))
    residual_scale = float(_cfg_get(m1, "GEOTR_AEFR_SINGLE_RESIDUAL_LOGIT_SCALE", _cfg_get(m1, "GEOTR_PC2R_RESIDUAL_LOGIT_SCALE", 2.0)))
    region_err_mask = region_b & err
    fn_err = region_b & (~anchor_h) & gt
    fp_err = region_b & anchor_h & (~gt)
    ah = anchor_h.float()[:, None]
    dil = F.max_pool2d(ah, kernel_size=3, stride=1, padding=1)
    ero = -F.max_pool2d(-ah, kernel_size=3, stride=1, padding=1)
    anchor_boundary = (dil != ero)[:, 0]
    def _near_boundary(radius: int):
        if radius <= 0:
            return anchor_boundary
        return F.max_pool2d(anchor_boundary.float()[:, None], kernel_size=2*radius+1, stride=1, padding=radius)[:, 0] > 0.5
    near1 = _near_boundary(1)
    near2 = _near_boundary(2)
    near3 = _near_boundary(3)
    near5 = _near_boundary(5)

    _aefr_enabled = bool(_cfg_get(m1, "GEOTR_AEFR_ENABLED", False))
    _aefr_stage = str(_cfg_get(m1, "GEOTR_AEFR_STAGE", "single_atomic")).strip().lower()
    _slr_ucdrt_r2_enabled = bool(_cfg_get(m1, "GEOTR_SLR_UCDRT_R2_ENABLED", False))
    _slr_ucdrt_enabled = _slr_ucdrt_r2_enabled or bool(_cfg_get(m1, "GEOTR_SLR_UCDRT_ENABLED", False))
    if _aefr_enabled and _aefr_stage == "sparse_local_rerendering" and _slr_ucdrt_r2_enabled:
        # R2 probability-dose ADD/REMOVE can cross every finite binary FN/FP in
        # a selected patch; MOVE is optional geometry, not the sole boundary path.
        reachable_err = region_err_mask
    elif _aefr_enabled and _aefr_stage == "sparse_local_rerendering" and _slr_ucdrt_enabled:
        # Historical UCDRT-R1 reachability.
        _br = max(1, int(_cfg_get(m1, "GEOTR_SLR_UCDRT_BOUNDARY_RADIUS_PX", 5)))
        _dmax = max(0, int(float(_cfg_get(m1, "GEOTR_SLR_UCDRT_MAX_BOUNDARY_DISPLACEMENT_PX", 4.0))))
        _zmax = max(float(_cfg_get(m1, "GEOTR_SLR_UCDRT_MAX_INTERIOR_LOGIT_STEP", 4.0)), 1.0e-6)
        _boundary_band = _near_boundary(_br)
        _move_reach = _near_boundary(_dmax) if _dmax > 0 else torch.zeros_like(region_err_mask)
        _interior_reach = (~_boundary_band) & (anchor_logit_map.abs() < _zmax)
        reachable_err = region_err_mask & ((_boundary_band & _move_reach) | _interior_reach)
    elif _aefr_enabled and _aefr_stage == "sparse_local_rerendering":
        # SLR2.4/OCRA predicts an uncapped typed dose and deploys that exact
        # action throughout the selected WOLA support. Capacity reachability is
        # therefore spatial; predicted dose sufficiency is audited separately.
        reachable_err = region_err_mask
    elif _aefr_enabled and _aefr_stage in {"hybrid_geometry", "typed_state_taylor", "typed_state_exact", "soft_ownership_exact", "intervention_factorized", "selective_minimal_intervention"}:
        _band_r = max(1, int(_cfg_get(m1, "GEOTR_AEFR_BOUNDARY_RADIUS_PX", 5)))
        _dmax = max(0, int(float(_cfg_get(m1, "GEOTR_AEFR_BOUNDARY_MAX_DISPLACEMENT_PX", 4.0))))
        _boundary_band_reach = _near_boundary(_band_r)
        _spatial_reach = _near_boundary(_dmax) if _dmax > 0 else torch.zeros_like(region_err_mask)
        # Pixels outside the deterministic boundary band are owned by the
        # anchor-adaptive interior residual and are representationally
        # crossable.  Within the band we use a conservative spatial-distance
        # test against the maximum normal displacement.
        reachable_err = region_err_mask & (_spatial_reach | (~_boundary_band_reach))
    else:
        reachable_err = region_err_mask & (anchor_logit_map.abs() < max(residual_scale, 1.0e-6))
    unreachable_err = region_err_mask & (~reachable_err)
    unreachable_fn = fn_err & (~reachable_err)
    unreachable_fp = fp_err & (~reachable_err)

    edit_b = edit[:, 0].detach() > 0.5
    candidate_b = candidate[:, 0].detach() > 0.5
    commit_b = commit[:, 0].detach() > 0.5
    corrected = commit_b & err & (final_h == gt)
    introduced = commit_b & (~err) & (final_h != gt)
    corrected_count = corrected.float().sum()
    introduced_count = introduced.float().sum()
    region_err_count = (region_b & err).float().sum()
    region_correct_count = (region_b & (~err)).float().sum()
    region_count = region_b.float().sum()
    total_err_count = err.float().sum()
    total_pix_count = err.new_tensor(float(err.numel()), dtype=anchor.dtype)
    consensus_count = consensus_b.float().sum()
    edit_count = edit_b.float().sum()
    candidate_count = candidate_b.float().sum()
    commit_count = commit_b.float().sum()

    anchor_d = _dice_per_case(anchor[:, 0], target)
    final_d = _dice_per_case(final[:, 0], target)
    mean_d = _dice_per_case(mean_prob[:, 0].detach(), target)
    view_d = [
        _dice_per_case(views[:, vi].detach(), target).mean()
        for vi in range(views.shape[1])
    ]

    # Model-resolution ceilings.  Native-resolution counterparts are computed
    # in train.evaluate_validation using the exact eval.py resize protocol.
    context_oracle = anchor.clone()
    context_oracle[:, 0] = torch.where(region_b & err, target, anchor[:, 0])
    context_oracle_gain = (_dice_per_case(context_oracle[:, 0], target) - anchor_d).mean()
    reachable_oracle = anchor.clone()
    reachable_oracle[:, 0] = torch.where(reachable_err, target, anchor[:, 0])
    reachable_oracle_gain = (_dice_per_case(reachable_oracle[:, 0], target) - anchor_d).mean()
    boundary5_oracle = anchor.clone()
    boundary5_oracle[:, 0] = torch.where(region_err_mask & near5, target, anchor[:, 0])
    boundary5_oracle_gain = (_dice_per_case(boundary5_oracle[:, 0], target) - anchor_d).mean()
    interior_oracle = anchor.clone()
    interior_oracle[:, 0] = torch.where(region_err_mask & (~near5), target, anchor[:, 0])
    interior_oracle_gain = (_dice_per_case(interior_oracle[:, 0], target) - anchor_d).mean()
    candidate_oracle = anchor.clone()
    candidate_oracle[:, 0] = torch.where(candidate_b & err, target, anchor[:, 0])
    candidate_oracle_gain = (_dice_per_case(candidate_oracle[:, 0], target) - anchor_d).mean()
    all_oracle = anchor.clone()
    all_oracle[:, 0] = torch.where(err, target, anchor[:, 0])
    all_oracle_gain = (_dice_per_case(all_oracle[:, 0], target) - anchor_d).mean()

    hard_views = views.detach() >= 0.5
    unanimous = (hard_views == hard_views[:, :1]).all(dim=1)
    view_spread = views.detach().max(dim=1).values - views.detach().min(dim=1).values
    agree_err_mask = region_b & err
    agree_correct_mask = region_b & (~err)
    agree_candidate_mask = candidate_b
    agreement_error_count = (unanimous & agree_err_mask).float().sum()
    agreement_correct_count = (unanimous & agree_correct_mask).float().sum()
    agreement_candidate_count = (unanimous & agree_candidate_mask).float().sum()
    spread_error_sum = (view_spread * agree_err_mask.to(view_spread)).sum()
    spread_correct_sum = (view_spread * agree_correct_mask.to(view_spread)).sum()
    spread_candidate_sum = (view_spread * agree_candidate_mask.to(view_spread)).sum()

    stage_gain = final_d - anchor_d
    benefit_case_count = (stage_gain > 1.0e-8).float().sum()
    harm_case_count = (stage_gain < -1.0e-8).float().sum()
    case_count = stage_gain.new_tensor(float(stage_gain.numel()))

    typed_state_stage = bool(_aefr_enabled and _aefr_stage in {"typed_state_taylor", "typed_state_exact"})
    soft_ownership_stage = bool(_aefr_enabled and _aefr_stage == "soft_ownership_exact")
    intervention_stage = bool(_aefr_enabled and _aefr_stage in {"intervention_factorized", "selective_minimal_intervention"})
    state_val = None
    ownership_val = None
    intervention_val = None
    if typed_state_stage:
        if not isinstance(aefr_state_logits, torch.Tensor) or not isinstance(aefr_boundary_mask, torch.Tensor):
            raise RuntimeError("Typed AEFR validation requires state logits and boundary mask in forward outputs")
        state_target, _state_masks = _aefr_five_state_targets(
            anchor, target, region, aefr_boundary_mask
        )
        state_val = _five_state_metrics(aefr_state_logits, state_target, region)
    if soft_ownership_stage:
        if not isinstance(aefr_ownership_logits, torch.Tensor):
            raise RuntimeError("SRO-Exact validation requires ownership logits in forward outputs")
        own_target = _aefr_soft_ownership_targets(anchor, target, region)
        ownership_val = _three_ownership_metrics(
            aefr_ownership_logits, own_target, anchor, target, region
        )
    if intervention_stage:
        if not all(isinstance(x, torch.Tensor) for x in (
            aefr_error_localizer_prob, aefr_edit_prob, aefr_direction_prob, aefr_signed_action
        )):
            raise RuntimeError("IFR/SMI validation requires error/edit/direction/signed-action outputs")
        intervention_val = _aefr_intervention_metrics(
            aefr_error_localizer_prob, aefr_edit_prob, aefr_direction_prob, aefr_signed_action,
            anchor, target, region
        )

    smi_mag_mae = anchor.new_zeros(())
    smi_mag_pred_mean = anchor.new_zeros(())
    smi_mag_target_mean = anchor.new_zeros(())
    if _aefr_enabled and _aefr_stage == "selective_minimal_intervention":
        if not isinstance(aefr_boundary_magnitude, torch.Tensor) or not isinstance(aefr_boundary_mask, torch.Tensor):
            raise RuntimeError("SMI validation requires boundary magnitude and boundary mask")
        max_disp = float(_cfg_get(m1, "GEOTR_AEFR_BOUNDARY_MAX_DISPLACEMENT_PX", 4.0))
        btarget = _aefr_boundary_magnitude_target(anchor, max_disp)
        b_support = region_b & err & (aefr_boundary_mask[:,0].detach() > 0.5)
        smi_mag_mae = _masked_mean((aefr_boundary_magnitude[:,0].detach() - btarget[:,0]).abs(), b_support)
        smi_mag_pred_mean = _masked_mean(aefr_boundary_magnitude[:,0].detach(), b_support)
        smi_mag_target_mean = _masked_mean(btarget[:,0], b_support)

    # SLR validation: compare selector output with its *absolute patch-error
    # density* target.  Pixel-error AP is intentionally not used because the
    # selector is a compute-value map, not a pixel posterior.
    slr_selector_ap = anchor.new_zeros(())  # compatibility key: patch-support AP
    slr_selector_value_mae = anchor.new_zeros(())
    slr_selector_value_corr = anchor.new_zeros(())
    slr_changed_precision = anchor.new_zeros(())
    slr_changed_recall = anchor.new_zeros(())
    slr_context_realization_num = anchor.new_zeros(())
    slr_context_realization_den = anchor.new_zeros(())
    slr_blend_overlap_fraction = anchor.new_zeros(())
    slr_blend_weight_mean = anchor.new_zeros(())
    slr_sdf_band_mae = anchor.new_zeros(())
    slr_anchor_sdf_band_mae = anchor.new_zeros(())
    slr_sdf_gain = anchor.new_zeros(())
    slr_sdf_sign_acc = anchor.new_zeros(())
    slr_sdf_zero_cross_dice = anchor.new_zeros(())
    slr_boundary_changed_precision = anchor.new_zeros(())
    slr_boundary_changed_recall = anchor.new_zeros(())
    slr_interior_change_rate = anchor.new_zeros(())
    slr_pred_patch_dice_gain = anchor.new_zeros(())  # compatibility = deploy patch gain
    slr_raw_patch_dice_gain = anchor.new_zeros(())
    slr_deploy_patch_dice_gain = anchor.new_zeros(())
    slr_actionable_selector_recall = anchor.new_zeros(())
    # Corrected semantics: training loss explicitly masks to action support, so
    # supervision-outside-action must be zero.  The historical quantity is kept
    # separately as PatchOutsideActionFrac.
    slr_unreachable_supervision_fraction = anchor.new_zeros(())
    slr_patch_outside_action_fraction = anchor.new_zeros(())
    slr_positive_from_selector_fraction = anchor.new_zeros(())
    slr_actor_direction_accuracy = anchor.new_zeros(())
    slr_actor_dose_reachability = anchor.new_zeros(())
    slr_actor_dose_ratio = anchor.new_zeros(())
    slr_actor_action_abs_mean = anchor.new_zeros(())
    slr_overlap_disagreement = anchor.new_zeros(())
    slr_ucdrt_state_macro_f1 = anchor.new_zeros(())
    slr_ucdrt_keep_f1 = anchor.new_zeros(())
    slr_ucdrt_move_f1 = anchor.new_zeros(())
    slr_ucdrt_add_f1 = anchor.new_zeros(())
    slr_ucdrt_remove_f1 = anchor.new_zeros(())
    slr_ucdrt_utility_mae = anchor.new_zeros(())
    slr_ucdrt_utility_corr = anchor.new_zeros(())
    slr_ucdrt_commit_rate = anchor.new_zeros(())
    slr_ucdrt_commit_precision = anchor.new_zeros(())
    slr_ucdrt_move_mae = anchor.new_zeros(())
    slr_ucdrt_interior_mae = anchor.new_zeros(())
    # UCDRT-R2 validation diagnostics.
    slr_ucdrt_r2_edit_f1 = anchor.new_zeros(())
    slr_ucdrt_r2_edit_precision = anchor.new_zeros(())
    slr_ucdrt_r2_edit_recall = anchor.new_zeros(())
    slr_ucdrt_r2_target_edit_fraction = anchor.new_zeros(())
    slr_ucdrt_r2_pred_edit_fraction = anchor.new_zeros(())
    slr_ucdrt_r2_type_macro_f1 = anchor.new_zeros(())
    slr_ucdrt_r2_move_f1 = anchor.new_zeros(())
    slr_ucdrt_r2_add_f1 = anchor.new_zeros(())
    slr_ucdrt_r2_remove_f1 = anchor.new_zeros(())
    slr_ucdrt_r2_hard_mean_gain = anchor.new_zeros(())
    slr_ucdrt_r2_hard_positive_rate = anchor.new_zeros(())
    slr_ucdrt_r2_hard_oracle_gain = anchor.new_zeros(())
    slr_ucdrt_r2_soft_mean_gain = anchor.new_zeros(())
    slr_ucdrt_r2_soft_positive_rate = anchor.new_zeros(())
    slr_ucdrt_r2_soft_oracle_gain = anchor.new_zeros(())
    slr_ucdrt_r2_hard_soft_gap = anchor.new_zeros(())
    slr_ucdrt_r2_critic_precision = anchor.new_zeros(())
    slr_ucdrt_r2_critic_recall = anchor.new_zeros(())
    slr_ucdrt_r2_critic_positive_rate = anchor.new_zeros(())
    slr_ucdrt_r2_set_oracle_gain = anchor.new_zeros(())
    slr_ucdrt_r2_set_predicted_gain = anchor.new_zeros(())
    slr_ucdrt_r2_set_realization = anchor.new_zeros(())
    slr_overlap_sdf_disagreement = anchor.new_zeros(())
    if _aefr_enabled and _aefr_stage == "sparse_local_rerendering" and isinstance(slr_selector_prob, torch.Tensor):
        r = int(_cfg_get(m1, "GEOTR_SLR_REGION_SIZE", _cfg_get(m1, "GEOTR_C2R_REGION_SIZE", 33)))
        if isinstance(slr_action_weight_full, torch.Tensor):
            aw = slr_action_weight_full.detach().to(anchor)[:,0].clamp(0,1)
        else:
            rad_tmp=max(int(_cfg_get(m1,"GEOTR_SLR_SDF_RADIUS_PX",8)),1)
            sdf_tmp=_slr_truncated_signed_distance(anchor_h[:,None].to(anchor),rad_tmp)
            aw=(1.0-sdf_tmp[:,0].abs()/float(rad_tmp)).clamp(0,1)
        # OCRA and UCDRT both expose an executable action throughout the selected
        # WOLA patch. Geometry boundary weight is evidence, not reachability.
        selector_error = err.float()
        actionable_err_b = err
        value_target = F.avg_pool2d(selector_error[:,None], kernel_size=r, stride=1, padding=r//2).clamp(0.0, 1.0)
        patch_support = value_target[:,0] > 0.0
        action_region_b = (slr_action_region_mask[:,0] > 0.5) if isinstance(slr_action_region_mask,torch.Tensor) else (region_b & (aw>0))
        slr_actionable_selector_recall = _safe_ratio((action_region_b & actionable_err_b).float().sum(), actionable_err_b.float().sum())
        if isinstance(slr_positive_from_selector_fraction_raw, torch.Tensor):
            slr_positive_from_selector_fraction = slr_positive_from_selector_fraction_raw.detach().to(anchor).mean()
        slr_selector_ap = _average_precision_binary(slr_selector_prob[:,0], patch_support)
        predv = slr_selector_prob.detach()
        tv = value_target.to(predv)
        slr_selector_value_mae = (predv - tv).abs().mean()
        pa = predv.flatten(1) - predv.flatten(1).mean(dim=1, keepdim=True)
        ta = tv.flatten(1) - tv.flatten(1).mean(dim=1, keepdim=True)
        corr = (pa * ta).sum(dim=1) / (
            torch.sqrt(pa.square().sum(dim=1) * ta.square().sum(dim=1)).clamp_min(1.0e-8)
        )
        slr_selector_value_corr = corr.mean()
        changed = final_h != anchor_h
        repaired = changed & err & (final_h == gt)
        slr_changed_precision = _safe_ratio(repaired.float().sum(), changed.float().sum())
        slr_changed_recall = _safe_ratio(repaired.float().sum(), err.float().sum())
        if isinstance(slr_action_delta_full, torch.Tensor):
            ad = slr_action_delta_full.detach().to(anchor)[:,0]
            selected_error = action_region_b & err
            proposed_on_error = selected_error & (ad.abs() > 1.0e-6)
            desired_sign = torch.where(gt, torch.ones_like(ad), -torch.ones_like(ad))
            direction_ok = (ad * desired_sign) > 0
            slr_actor_direction_accuracy = _safe_ratio(
                (direction_ok & proposed_on_error).float().sum(), proposed_on_error.float().sum()
            )
            margin_prob = float(_cfg_get(m1, "GEOTR_SLR_ACTION_MARGIN_PROB", 0.05))
            margin_prob = min(max(margin_prob, 1.0e-4), 0.4999)
            z_margin = float(torch.logit(torch.tensor(0.5 + margin_prob)).item())
            az = torch.logit(anchor[:,0].detach().clamp(EPS,1.0-EPS))
            required = torch.where(gt, (z_margin-az).clamp_min(0.0), (az+z_margin).clamp_min(0.0))
            dose_ok = direction_ok & (ad.abs() + 1.0e-6 >= required)
            slr_actor_dose_reachability = _safe_ratio(
                (dose_ok & selected_error).float().sum(), selected_error.float().sum()
            )
            slr_actor_dose_ratio = _masked_mean(
                (ad.abs() / required.clamp_min(1.0e-6)).clamp_max(10.0), selected_error
            )
            slr_actor_action_abs_mean = _masked_mean(ad.abs(), action_region_b)
        # Export numerator/denominator separately; validation aggregation can
        # average them safely, then the logger computes ratio-of-means.
        slr_context_realization_num = stage_gain.mean()
        slr_context_realization_den = context_oracle_gain.abs()
        if isinstance(slr_blend_weight_sum, torch.Tensor):
            bw = slr_blend_weight_sum.detach().to(anchor)
            slr_blend_weight_mean = _masked_mean(bw[:,0], region_b)
            slr_blend_overlap_fraction = (bw[:,0] > 1.0 + 1.0e-6).float().mean()
        if isinstance(slr_overlap_disagreement_raw, torch.Tensor):
            slr_overlap_disagreement = slr_overlap_disagreement_raw.detach().to(anchor).mean()
        if isinstance(slr_overlap_sdf_disagreement_raw, torch.Tensor):
            slr_overlap_sdf_disagreement = slr_overlap_sdf_disagreement_raw.detach().to(anchor).mean()
        if all(isinstance(x, torch.Tensor) for x in (
            slr_pred_patch_probs, slr_pred_patch_valid, slr_pred_patch_grid, slr_pred_patch_anchor_prob
        )):
            dp = slr_pred_patch_probs.detach().to(anchor)
            rp = (slr_raw_patch_probs.detach().to(anchor) if isinstance(slr_raw_patch_probs,torch.Tensor) else dp)
            pa = slr_pred_patch_anchor_prob.detach().to(anchor)
            pv = slr_pred_patch_valid.detach().to(anchor)
            bsz, kpatch, _, rr, _ = dp.shape
            grid = slr_pred_patch_grid.detach().to(anchor)
            gt4 = target[:, None].to(anchor)
            expanded = gt4[:, None].expand(bsz, kpatch, 1, *gt4.shape[-2:]).reshape(bsz*kpatch,1,*gt4.shape[-2:])
            gtp = F.grid_sample(expanded, grid, mode="nearest", padding_mode="border", align_corners=True).view(bsz,kpatch,1,rr,rr)
            vm_valid=pv
            vm=(pv*(slr_pred_patch_action_support.detach().to(anchor)>0).to(pv)
                if isinstance(slr_pred_patch_action_support,torch.Tensor) else pv)
            def _patch_gain(q):
                qi=(q*gtp*vm).sum(dim=(2,3,4)); qd=((q+gtp)*vm).sum(dim=(2,3,4))
                ai=(pa*gtp*vm).sum(dim=(2,3,4)); ad=((pa+gtp)*vm).sum(dim=(2,3,4))
                qdice=(2*qi+EPS)/(qd+EPS); adice=(2*ai+EPS)/(ad+EPS)
                valid_patch=vm.sum(dim=(2,3,4))>0
                return ((qdice-adice)*valid_patch.float()).sum()/valid_patch.float().sum().clamp_min(1.0)
            slr_raw_patch_dice_gain=_patch_gain(rp)
            slr_deploy_patch_dice_gain=_patch_gain(dp)
            slr_pred_patch_dice_gain=slr_deploy_patch_dice_gain
            if isinstance(slr_pred_patch_action_support,torch.Tensor):
                sup=slr_pred_patch_action_support.detach().to(anchor)
                outside=(vm_valid*(1.0-(sup>0).to(vm_valid))).sum()
                slr_patch_outside_action_fraction=outside/vm_valid.sum().clamp_min(1.0)
                # The actual training supervision is vm*sup, therefore by
                # construction it contains no sample outside action support.
                slr_unreachable_supervision_fraction = anchor.new_zeros(())

            # UCDRT state/action/value audit is computed with the exact same
            # factual candidate patches and executable candidate probabilities.
            if _slr_ucdrt_r2_enabled and all(isinstance(x, torch.Tensor) for x in (
                slr_ucdrt_r2_edit_logits, slr_ucdrt_r2_edit_probs,
                slr_ucdrt_r2_type_logits, slr_ucdrt_r2_move_bin_logits,
                slr_ucdrt_r2_move_dictionary_probs, slr_ucdrt_r2_add_dose,
                slr_ucdrt_r2_remove_dose, slr_ucdrt_r2_hard_candidate_probs,
                slr_ucdrt_r2_soft_candidate_probs, slr_ucdrt_r2_critic_step_logits,
                slr_ucdrt_r2_critic_step_values, slr_ucdrt_r2_critic_step_available,
                slr_ucdrt_r2_selection_step_index, slr_ucdrt_r2_candidate_full_num,
                slr_ucdrt_r2_candidate_full_den, slr_ucdrt_r2_selected_candidate_mask,
            )):
                boundary_patch = (
                    slr_pred_sdf_anchor.detach().to(anchor).abs()
                    <= float(_cfg_get(m1, "GEOTR_SLR_UCDRT_BOUNDARY_RADIUS_PX", 5))
                ).to(anchor) * pv
                r2t = _slr_ucdrt_r2_targets(
                    pa, gtp, pv, boundary_patch,
                    slr_ucdrt_r2_move_dictionary_probs.detach().to(anchor),
                    target_margin=float(_cfg_get(m1, "GEOTR_SLR_UCDRT_TARGET_MARGIN", 0.05)),
                )
                fm = _slr_ucdrt_r2_factor_metrics(
                    slr_ucdrt_r2_edit_logits.detach().to(anchor),
                    slr_ucdrt_r2_type_logits.detach().to(anchor), r2t, pv
                )
                slr_ucdrt_r2_edit_f1=fm["edit_f1"]; slr_ucdrt_r2_edit_precision=fm["edit_precision"]
                slr_ucdrt_r2_edit_recall=fm["edit_recall"]; slr_ucdrt_r2_target_edit_fraction=fm["target_edit_fraction"]
                slr_ucdrt_r2_pred_edit_fraction=fm["pred_edit_fraction"]; slr_ucdrt_r2_type_macro_f1=fm["type_macro_f1"]
                slr_ucdrt_r2_move_f1=fm["move_f1"]; slr_ucdrt_r2_add_f1=fm["add_f1"]; slr_ucdrt_r2_remove_f1=fm["remove_f1"]
                # Compatibility display.
                slr_ucdrt_state_macro_f1=0.5*(slr_ucdrt_r2_edit_f1+slr_ucdrt_r2_type_macro_f1)
                slr_ucdrt_keep_f1=slr_ucdrt_r2_edit_f1; slr_ucdrt_move_f1=slr_ucdrt_r2_move_f1
                slr_ucdrt_add_f1=slr_ucdrt_r2_add_f1; slr_ucdrt_remove_f1=slr_ucdrt_r2_remove_f1

                cv=(pv.sum(dim=(2,3,4))>0).to(anchor)
                aq=_slr_ucdrt_quality_per_candidate(pa,gtp,pv)
                hq=_slr_ucdrt_quality_per_candidate(slr_ucdrt_r2_hard_candidate_probs.detach().to(anchor),gtp,pv)
                sq=_slr_ucdrt_quality_per_candidate(slr_ucdrt_r2_soft_candidate_probs.detach().to(anchor),gtp,pv)
                hg=(hq-aq)*cv; sg=(sq-aq)*cv; denv=cv.sum().clamp_min(1.0)
                slr_ucdrt_r2_hard_mean_gain=hg.sum()/denv
                slr_ucdrt_r2_soft_mean_gain=sg.sum()/denv
                slr_ucdrt_r2_hard_positive_rate=((hg>0).to(hg)*cv).sum()/denv
                slr_ucdrt_r2_soft_positive_rate=((sg>0).to(sg)*cv).sum()/denv
                slr_ucdrt_r2_hard_oracle_gain=hg.clamp_min(0).max(dim=1).values.mean()
                slr_ucdrt_r2_soft_oracle_gain=sg.clamp_min(0).max(dim=1).values.mean()
                slr_ucdrt_r2_hard_soft_gap=slr_ucdrt_r2_soft_oracle_gain-slr_ucdrt_r2_hard_oracle_gain

                gain_scale=float(_cfg_get(m1,"GEOTR_SLR_UCDRT_R2_UTILITY_GAIN_SCALE",20.0))
                bqw=float(_cfg_get(m1,"GEOTR_SLR_UCDRT_R2_UTILITY_BOUNDARY_WEIGHT",0.25))
                ctarget=_slr_ucdrt_r2_critic_targets(
                    anchor.detach(), target[:,None].detach(),
                    slr_ucdrt_r2_candidate_full_num, slr_ucdrt_r2_candidate_full_den,
                    slr_ucdrt_r2_selection_step_index, slr_ucdrt_r2_critic_step_available,
                    boundary_weight=bqw,gain_scale=gain_scale,
                )
                _,cm=_slr_ucdrt_r2_critic_loss(
                    slr_ucdrt_r2_critic_step_logits.detach().to(anchor),
                    slr_ucdrt_r2_critic_step_values.detach().to(anchor),
                    ctarget,slr_ucdrt_r2_critic_step_available.detach().to(anchor),
                    rank_margin=float(_cfg_get(m1,"GEOTR_SLR_UCDRT_R2_UTILITY_RANK_MARGIN",0.05)),
                )
                slr_ucdrt_utility_mae=cm["mae"]; slr_ucdrt_utility_corr=cm["corr"]
                slr_ucdrt_r2_critic_precision=cm["precision"]; slr_ucdrt_r2_critic_recall=cm["recall"]
                slr_ucdrt_r2_critic_positive_rate=cm["positive_rate"]
                sel=slr_ucdrt_r2_selected_candidate_mask.detach().to(anchor)
                slr_ucdrt_commit_rate=(sel*cv).sum()/cv.sum().clamp_min(1.0)
                slr_ucdrt_commit_precision=slr_ucdrt_r2_critic_precision
                slr_ucdrt_r2_set_oracle_gain=_slr_ucdrt_r2_greedy_oracle_set_gain(
                    anchor.detach(),target[:,None].detach(),
                    slr_ucdrt_r2_candidate_full_num,slr_ucdrt_r2_candidate_full_den,
                    boundary_weight=bqw,
                )
                q0=_slr_ucdrt_r2_full_quality(anchor.detach(),target[:,None].detach(),bqw)
                qf=_slr_ucdrt_r2_full_quality(final.detach(),target[:,None].detach(),bqw)
                slr_ucdrt_r2_set_predicted_gain=(qf-q0).mean()
                slr_ucdrt_r2_set_realization=slr_ucdrt_r2_set_predicted_gain/slr_ucdrt_r2_set_oracle_gain.abs().clamp_min(1e-8)
            elif _slr_ucdrt_enabled and all(isinstance(x, torch.Tensor) for x in (
                slr_ucdrt_state_logits, slr_ucdrt_move_px,
                slr_ucdrt_interior_magnitude, slr_ucdrt_candidate_soft_probs,
                slr_ucdrt_utility_value, slr_ucdrt_commit, slr_pred_sdf_anchor,
            )):
                rad = max(int(_cfg_get(m1, "GEOTR_SLR_SDF_RADIUS_PX", 8)), 1)
                gt_sdf_full = _slr_truncated_signed_distance(gt[:,None].to(anchor), rad)
                sdf_expand = gt_sdf_full[:,None].expand(bsz,kpatch,1,*gt_sdf_full.shape[-2:]).reshape(
                    bsz*kpatch,1,*gt_sdf_full.shape[-2:]
                )
                gt_sdf_patch = F.grid_sample(
                    sdf_expand, grid, mode="bilinear", padding_mode="border", align_corners=True
                ).view(bsz,kpatch,1,rr,rr)
                st, masks, move_t, mag_t = _slr_ucdrt_patch_targets(
                    pa, gtp, slr_pred_sdf_anchor.detach().to(anchor), gt_sdf_patch, pv,
                    boundary_radius_px=float(_cfg_get(m1,"GEOTR_SLR_UCDRT_BOUNDARY_RADIUS_PX",5)),
                    max_boundary_displacement_px=float(_cfg_get(m1,"GEOTR_SLR_UCDRT_MAX_BOUNDARY_DISPLACEMENT_PX",4.0)),
                    max_interior_logit_step=float(_cfg_get(m1,"GEOTR_SLR_UCDRT_MAX_INTERIOR_LOGIT_STEP",4.0)),
                    target_margin=float(_cfg_get(m1,"GEOTR_SLR_UCDRT_TARGET_MARGIN",0.05)),
                )
                sm = _slr_ucdrt_state_metrics(slr_ucdrt_state_logits.detach().to(anchor), st, pv)
                slr_ucdrt_state_macro_f1 = sm["macro_f1"]
                slr_ucdrt_keep_f1 = sm["keep_f1"]
                slr_ucdrt_move_f1 = sm["move_f1"]
                slr_ucdrt_add_f1 = sm["add_f1"]
                slr_ucdrt_remove_f1 = sm["remove_f1"]
                _keep_mask, mv_mask, _add_mask, _remove_mask = masks
                in_mask = _add_mask | _remove_mask
                slr_ucdrt_move_mae = _masked_mean(
                    (slr_ucdrt_move_px.detach().to(anchor)[:, :, 0] - move_t[:, :, 0]).abs(),
                    mv_mask[:, :, 0],
                )
                slr_ucdrt_interior_mae = _masked_mean(
                    (slr_ucdrt_interior_magnitude.detach().to(anchor)[:, :, 0] - mag_t[:, :, 0]).abs(),
                    in_mask[:, :, 0],
                )
                util_v = slr_ucdrt_utility_value.detach().to(anchor)
                util_c = slr_ucdrt_commit.detach().to(anchor)
                anchor_patch_d = _slr_ucdrt_quality_per_candidate(pa, gtp, pv)
                candidate_patch_d = _slr_ucdrt_quality_per_candidate(
                    slr_ucdrt_candidate_soft_probs.detach().to(anchor), gtp, pv
                )
                util_t = (candidate_patch_d-anchor_patch_d).clamp(-1,1)
                cv = (pv.sum(dim=(2,3,4))>0).to(util_v)
                slr_ucdrt_utility_mae = ((util_v-util_t).abs()*cv).sum()/cv.sum().clamp_min(1.0)
                uu = util_v[cv>0]; tt = util_t[cv>0]
                if uu.numel() >= 2:
                    u0=uu-uu.mean(); t0=tt-tt.mean()
                    slr_ucdrt_utility_corr=(u0*t0).sum()/torch.sqrt((u0.square().sum()*t0.square().sum()).clamp_min(EPS))
                slr_ucdrt_commit_rate=(util_c*cv).sum()/cv.sum().clamp_min(1.0)
                comm=(util_c>0.5)&(cv>0); posgain=(util_t>0)&(cv>0)
                slr_ucdrt_commit_precision=_safe_ratio((comm&posgain).float().sum(),comm.float().sum())
        if isinstance(slr_sdf_absolute_full, torch.Tensor):
            rad = max(int(_cfg_get(m1, "GEOTR_SLR_SDF_RADIUS_PX", 8)), 1)
            sdf_a = _slr_truncated_signed_distance(anchor_h[:,None].to(anchor), rad)
            sdf_t = _slr_truncated_signed_distance(gt[:,None].to(anchor), rad)
            sdf_band = (sdf_a.abs() < float(rad)) | (sdf_t.abs() < float(rad)) | err[:,None]
            sdf_pred = slr_sdf_absolute_full.detach().to(anchor)
            slr_sdf_band_mae = _masked_mean(
                (sdf_pred[:,0] - sdf_t[:,0]).abs(),
                sdf_band[:,0],
            )
            slr_anchor_sdf_band_mae = _masked_mean(
                (sdf_a[:,0] - sdf_t[:,0]).abs(),
                sdf_band[:,0],
            )
            slr_sdf_gain = slr_anchor_sdf_band_mae - slr_sdf_band_mae

            # Batch-safe ROI shapes: all boolean tensors are [B,H,W].  The old
            # [B,1,H,W] & [B,H,W] broadcast produced impossible accuracies >1.
            roi = region_b
            sdf_hard = sdf_pred[:,0] > 0
            gt_hard = gt.bool()
            slr_sdf_sign_acc = _masked_mean((sdf_hard == gt_hard).float(), roi)
            pred_fg = sdf_hard & roi
            gt_fg = gt_hard & roi
            inter = (pred_fg & gt_fg).float().sum()
            den = pred_fg.float().sum() + gt_fg.float().sum()
            slr_sdf_zero_cross_dice = (2.0 * inter + EPS) / (den + EPS)

            boundary_support = sdf_a[:,0].abs() < float(rad)
            interior_support = ~boundary_support
            changed_now = final_h != anchor_h
            repaired_now = changed_now & err & (final_h == gt)
            b_changed = changed_now & boundary_support & region_b
            b_err = err & boundary_support & region_b
            slr_boundary_changed_precision = _safe_ratio(
                (repaired_now & boundary_support & region_b).float().sum(), b_changed.float().sum()
            )
            slr_boundary_changed_recall = _safe_ratio(
                (repaired_now & boundary_support & region_b).float().sum(), b_err.float().sum()
            )
            slr_interior_change_rate = _safe_ratio(
                (changed_now & interior_support & region_b).float().sum(),
                (interior_support & region_b).float().sum(),
            )

    stats = _selection_stats(region, anchor, target)
    out = {
        "val_geotr_v4g_modelres_soft_gain": stage_gain.mean(),
        "val_geotr_v4g_selection_coverage": stats["coverage"],
        "val_geotr_v4g_selection_precision": stats["precision"],
        "val_geotr_v4g_selection_recall": stats["recall"],
        "val_geotr_slr_selector_error_ap": slr_selector_ap,
        "val_geotr_slr_selector_value_mae": slr_selector_value_mae,
        "val_geotr_slr_selector_value_corr": slr_selector_value_corr,
        "val_geotr_slr_changed_precision": slr_changed_precision,
        "val_geotr_slr_changed_recall": slr_changed_recall,
        "val_geotr_slr_context_realization_num": slr_context_realization_num,
        "val_geotr_slr_context_realization_den": slr_context_realization_den,
        "val_geotr_slr_context_realization": slr_context_realization_num / slr_context_realization_den.clamp_min(1.0e-8),
        "val_geotr_slr_blend_overlap_fraction": slr_blend_overlap_fraction,
        "val_geotr_slr_blend_weight_mean": slr_blend_weight_mean,
        "val_geotr_slr_sdf_band_mae": slr_sdf_band_mae,
        "val_geotr_slr_anchor_sdf_band_mae": slr_anchor_sdf_band_mae,
        "val_geotr_slr_sdf_gain": slr_sdf_gain,
        "val_geotr_slr_sdf_sign_acc": slr_sdf_sign_acc,
        "val_geotr_slr_sdf_zero_cross_dice": slr_sdf_zero_cross_dice,
        "val_geotr_slr_boundary_changed_precision": slr_boundary_changed_precision,
        "val_geotr_slr_boundary_changed_recall": slr_boundary_changed_recall,
        "val_geotr_slr_interior_change_rate": slr_interior_change_rate,
        "val_geotr_slr_pred_patch_dice_gain": slr_pred_patch_dice_gain,
        "val_geotr_slr_raw_patch_dice_gain": slr_raw_patch_dice_gain,
        "val_geotr_slr_deploy_patch_dice_gain": slr_deploy_patch_dice_gain,
        "val_geotr_slr_actionable_selector_recall": slr_actionable_selector_recall,
        "val_geotr_slr_unreachable_supervision_fraction": slr_unreachable_supervision_fraction,
        "val_geotr_slr_patch_outside_action_fraction": slr_patch_outside_action_fraction,
        "val_geotr_slr_positive_from_selector_fraction": slr_positive_from_selector_fraction,
        "val_geotr_slr_actor_direction_accuracy": slr_actor_direction_accuracy,
        "val_geotr_slr_actor_dose_reachability": slr_actor_dose_reachability,
        "val_geotr_slr_actor_dose_ratio": slr_actor_dose_ratio,
        "val_geotr_slr_actor_action_abs_mean": slr_actor_action_abs_mean,
        "val_geotr_slr_ucdrt_state_macro_f1": slr_ucdrt_state_macro_f1,
        "val_geotr_slr_ucdrt_keep_f1": slr_ucdrt_keep_f1,
        "val_geotr_slr_ucdrt_move_f1": slr_ucdrt_move_f1,
        "val_geotr_slr_ucdrt_add_f1": slr_ucdrt_add_f1,
        "val_geotr_slr_ucdrt_remove_f1": slr_ucdrt_remove_f1,
        "val_geotr_slr_ucdrt_utility_mae": slr_ucdrt_utility_mae,
        "val_geotr_slr_ucdrt_utility_corr": slr_ucdrt_utility_corr,
        "val_geotr_slr_ucdrt_commit_rate": slr_ucdrt_commit_rate,
        "val_geotr_slr_ucdrt_commit_precision": slr_ucdrt_commit_precision,
        "val_geotr_slr_ucdrt_move_mae": slr_ucdrt_move_mae,
        "val_geotr_slr_ucdrt_interior_mae": slr_ucdrt_interior_mae,
        "val_geotr_slr_ucdrt_r2_edit_f1": slr_ucdrt_r2_edit_f1,
        "val_geotr_slr_ucdrt_r2_edit_precision": slr_ucdrt_r2_edit_precision,
        "val_geotr_slr_ucdrt_r2_edit_recall": slr_ucdrt_r2_edit_recall,
        "val_geotr_slr_ucdrt_r2_target_edit_fraction": slr_ucdrt_r2_target_edit_fraction,
        "val_geotr_slr_ucdrt_r2_pred_edit_fraction": slr_ucdrt_r2_pred_edit_fraction,
        "val_geotr_slr_ucdrt_r2_type_macro_f1": slr_ucdrt_r2_type_macro_f1,
        "val_geotr_slr_ucdrt_r2_move_f1": slr_ucdrt_r2_move_f1,
        "val_geotr_slr_ucdrt_r2_add_f1": slr_ucdrt_r2_add_f1,
        "val_geotr_slr_ucdrt_r2_remove_f1": slr_ucdrt_r2_remove_f1,
        "val_geotr_slr_ucdrt_r2_hard_mean_gain": slr_ucdrt_r2_hard_mean_gain,
        "val_geotr_slr_ucdrt_r2_hard_positive_rate": slr_ucdrt_r2_hard_positive_rate,
        "val_geotr_slr_ucdrt_r2_hard_oracle_gain": slr_ucdrt_r2_hard_oracle_gain,
        "val_geotr_slr_ucdrt_r2_soft_mean_gain": slr_ucdrt_r2_soft_mean_gain,
        "val_geotr_slr_ucdrt_r2_soft_positive_rate": slr_ucdrt_r2_soft_positive_rate,
        "val_geotr_slr_ucdrt_r2_soft_oracle_gain": slr_ucdrt_r2_soft_oracle_gain,
        "val_geotr_slr_ucdrt_r2_hard_soft_gap": slr_ucdrt_r2_hard_soft_gap,
        "val_geotr_slr_ucdrt_r2_critic_precision": slr_ucdrt_r2_critic_precision,
        "val_geotr_slr_ucdrt_r2_critic_recall": slr_ucdrt_r2_critic_recall,
        "val_geotr_slr_ucdrt_r2_critic_positive_rate": slr_ucdrt_r2_critic_positive_rate,
        "val_geotr_slr_ucdrt_r2_set_oracle_gain": slr_ucdrt_r2_set_oracle_gain,
        "val_geotr_slr_ucdrt_r2_set_predicted_gain": slr_ucdrt_r2_set_predicted_gain,
        "val_geotr_slr_ucdrt_r2_set_realization": slr_ucdrt_r2_set_realization,
        "val_geotr_slr_overlap_disagreement": slr_overlap_disagreement,
        "val_geotr_slr_overlap_sdf_disagreement": slr_overlap_sdf_disagreement,
        "val_geotr_v4g_anchor_error_rate": stats["error_rate"],
        "val_geotr_v4g_selected_error_count": region_err_count,
        "val_geotr_v4g_selected_correct_count": region_correct_count,
        "val_geotr_v4g_selected_count": region_count,
        "val_geotr_v4g_total_error_count": total_err_count,
        "val_geotr_v4g_total_pixel_count": total_pix_count,
        "val_geotr_v4g_point_accuracy_before": _point_accuracy(anchor, target, region),
        "val_geotr_v4g_point_accuracy_after": _point_accuracy(final, target, region),
        "val_geotr_v4g_point_accuracy_gain": _point_accuracy(final, target, region) - _point_accuracy(anchor, target, region),
        "val_geotr_v4g_selected_abs_change": _masked_mean((final[:,0]-anchor[:,0]).abs(), region_b),
        "val_geotr_v4g_margin_error_ap": _average_precision_binary(margin[:,0], err),
        "val_geotr_v4g_mc_std_error_ap": _average_precision_binary(mc_std[:,0], err),
        "val_geotr_v4g_mc_disagreement_error_ap": _average_precision_binary(mc_dis[:,0], err),
        "val_geotr_v4g_entropy_error_ap": _average_precision_binary(entropy[:,0], err),
        "val_geotr_aefr_intervention_error_ap": (
            _average_precision_binary(aefr_error_localizer_prob[:,0], err)
            if intervention_stage else anchor.new_zeros(())
        ),
        "val_geotr_aefr_intervention_error_recall_at_05": (
            _topk_recall(aefr_error_localizer_prob[:,0], err, 0.05)
            if intervention_stage else anchor.new_zeros(())
        ),
        "val_geotr_aefr_intervention_error_recall_at_10": (
            _topk_recall(aefr_error_localizer_prob[:,0], err, 0.10)
            if intervention_stage else anchor.new_zeros(())
        ),
        "val_geotr_aefr_smi_boundary_mag_mae": smi_mag_mae,
        "val_geotr_aefr_smi_boundary_mag_pred_mean": smi_mag_pred_mean,
        "val_geotr_aefr_smi_boundary_mag_target_mean": smi_mag_target_mean,
        "val_geotr_v4g_margin_recall_at_05": _topk_recall(margin[:,0], err, 0.05),
        "val_geotr_v4g_margin_recall_at_10": _topk_recall(margin[:,0], err, 0.10),
        "val_geotr_v4g_mc_std_recall_at_05": _topk_recall(mc_std[:,0], err, 0.05),
        "val_geotr_v4g_mc_std_recall_at_10": _topk_recall(mc_std[:,0], err, 0.10),
        "val_geotr_v4g_mc_disagreement_recall_at_05": _topk_recall(mc_dis[:,0], err, 0.05),
        "val_geotr_v4g_mc_disagreement_recall_at_10": _topk_recall(mc_dis[:,0], err, 0.10),
        "val_geotr_v4g_corrected_error_count": corrected_count,
        "val_geotr_v4g_introduced_error_count": introduced_count,
        "val_geotr_v4g_net_correction_count": corrected_count - introduced_count,
        "val_geotr_v4g_correction_recall": _safe_ratio(corrected_count, total_err_count),
        "val_geotr_v4g_introduction_rate": _safe_ratio(introduced_count, region_count),
        "val_geotr_v4g_pred_selector_oracle_gain": context_oracle_gain,
        "val_geotr_v4g_full_residual_oracle_gain": all_oracle_gain,
        "val_geotr_v4g_benefit_case_rate": (stage_gain > 1.0e-8).float().mean(),
        "val_geotr_v4g_harm_case_rate": (stage_gain < -1.0e-8).float().mean(),
        "val_geotr_c2r_region_coverage": region_b.float().mean(),
        "val_geotr_c2r_center_count": center_b.float().sum(),
        "val_geotr_c2r_region_error_density": _safe_ratio(region_err_count, region_count),
        "val_geotr_c2r_region_error_recall": _safe_ratio(region_err_count, total_err_count),
        "val_pc2r_reachable_region_error_count": reachable_err.float().sum(),
        "val_pc2r_unreachable_region_error_count": unreachable_err.float().sum(),
        "val_pc2r_reachable_region_error_fraction": _safe_ratio(reachable_err.float().sum(), region_err_count),
        "val_pc2r_unreachable_fn_count": unreachable_fn.float().sum(),
        "val_pc2r_unreachable_fp_count": unreachable_fp.float().sum(),
        "val_pc2r_reachable_oracle_gain": reachable_oracle_gain,
        "val_pc2r_error_near_boundary_1_fraction": _safe_ratio((region_err_mask & near1).float().sum(), region_err_count),
        "val_pc2r_error_near_boundary_2_fraction": _safe_ratio((region_err_mask & near2).float().sum(), region_err_count),
        "val_pc2r_error_near_boundary_3_fraction": _safe_ratio((region_err_mask & near3).float().sum(), region_err_count),
        "val_pc2r_error_near_boundary_5_fraction": _safe_ratio((region_err_mask & near5).float().sum(), region_err_count),
        "val_pc2r_error_beyond_boundary_5_fraction": _safe_ratio((region_err_mask & (~near5)).float().sum(), region_err_count),
        "val_pc2r_boundary5_oracle_gain": boundary5_oracle_gain,
        "val_pc2r_interior_oracle_gain": interior_oracle_gain,
        "val_geotr_c2r_consensus_count": consensus_count,
        "val_geotr_c2r_consensus_rate_in_region": _safe_ratio(consensus_count, region_count),
        "val_geotr_c2r_edit_count": edit_count,
        "val_geotr_c2r_edit_rate_in_region": _safe_ratio(edit_count, region_count),
        "val_geotr_c2r_edit_precision": _safe_ratio(corrected_count, corrected_count + introduced_count),
        "val_geotr_c2r_view_spread_in_region": _masked_mean(view_spread, region_b),
        "val_geotr_c2r_region_oracle_gain": context_oracle_gain,
        "val_geotr_c2r_modelres_context_oracle_gain": context_oracle_gain,
        "val_geotr_c2r_modelres_candidate_oracle_gain": candidate_oracle_gain,
        "val_geotr_c2r_canonical_mean_gain": (mean_d - anchor_d).mean(),
        "val_geotr_c2r_stage2_gain": stage_gain.mean(),
        "val_geotr_c2r_harm_case_rate": (stage_gain < -1.0e-8).float().mean(),
        "val_geotr_c2r_benefit_case_count": benefit_case_count,
        "val_geotr_c2r_harm_case_count": harm_case_count,
        "val_geotr_c2r_case_count": case_count,
        "val_geotr_c2r_candidate_pixel_count": candidate_count,
        "val_geotr_c2r_commit_pixel_count": commit_count,
        "val_geotr_c2r_agreement_error_count": agreement_error_count,
        "val_geotr_c2r_agreement_error_total_count": agree_err_mask.float().sum(),
        "val_geotr_c2r_agreement_correct_count": agreement_correct_count,
        "val_geotr_c2r_agreement_correct_total_count": agree_correct_mask.float().sum(),
        "val_geotr_c2r_agreement_candidate_count": agreement_candidate_count,
        "val_geotr_c2r_agreement_candidate_total_count": agree_candidate_mask.float().sum(),
        "val_geotr_c2r_spread_error_sum": spread_error_sum,
        "val_geotr_c2r_spread_error_count": agree_err_mask.float().sum(),
        "val_geotr_c2r_spread_correct_sum": spread_correct_sum,
        "val_geotr_c2r_spread_correct_count": agree_correct_mask.float().sum(),
        "val_geotr_c2r_spread_candidate_sum": spread_candidate_sum,
        "val_geotr_c2r_spread_candidate_count": agree_candidate_mask.float().sum(),
        # Geometry->Stage2 interference matrix.  These are validation-only,
        # never used by forward, loss, checkpoint selection, or deployment.
        "val_geotr_aefr_gfix_count": g_fix.float().sum(),
        "val_geotr_aefr_gharm_count": g_harm.float().sum(),
        "val_geotr_aefr_gmiss_count": g_miss.float().sum(),
        "val_geotr_aefr_gkeep_count": g_keep.float().sum(),
        "val_geotr_aefr_gfix_preserve_rate": gfix_preserve_rate,
        "val_geotr_aefr_gfix_break_rate": gfix_break_rate,
        "val_geotr_aefr_gharm_repair_rate": gharm_repair_rate,
        "val_geotr_aefr_gmiss_repair_rate": gmiss_repair_rate,
        "val_geotr_aefr_gkeep_break_rate": gkeep_break_rate,
        "val_geotr_aefr_gfix_edit_rate": gfix_edit_rate,
        "val_geotr_aefr_gharm_edit_rate": gharm_edit_rate,
        "val_geotr_aefr_gmiss_edit_rate": gmiss_edit_rate,
        "val_geotr_aefr_gkeep_edit_rate": gkeep_edit_rate,
        "val_geotr_aefr_gfix_action_abs_dz": gfix_action_abs,
        "val_geotr_aefr_gharm_action_abs_dz": gharm_action_abs,
        "val_geotr_aefr_gmiss_action_abs_dz": gmiss_action_abs,
        "val_geotr_aefr_gkeep_action_abs_dz": gkeep_action_abs,
    }
    if state_val is not None:
        for _k, _v in state_val.items():
            out["val_geotr_aefr_state_" + _k] = _v
    if ownership_val is not None:
        for _k, _v in ownership_val.items():
            out["val_geotr_aefr_ownership_" + _k] = _v
    if intervention_val is not None:
        for _k, _v in intervention_val.items():
            out["val_geotr_aefr_intervention_" + _k] = _v
    for vi, vd in enumerate(view_d):
        out[f"val_geotr_c2r_view{vi}_dice"] = vd

    # Structural forward-only diagnostics are exported by Canonical C2R-v2.
    for suffix in (
        "roi_overlap_pixel_count",
        "roi_unique_pixel_count",
        "center_min_chebyshev_distance",
        "candidate_component_count",
        "committed_component_count",
        "candidate_component_area_total",
        "committed_component_area_total",
        "candidate_component_area_mean",
        "committed_component_area_mean",
        "component_agreement_mean",
        "component_spread_q90_mean",
        "component_confidence_q10_mean",
    ):
        value = pred.get(prefix + "_c2r_" + suffix)
        if isinstance(value, torch.Tensor):
            key = "val_geotr_c2r_" + suffix
            # Totals/counts must survive as additive quantities; train.py sums
            # *_count and *_sum fields across validation batches.
            if suffix.endswith("_total"):
                key = key + "_sum"
            out[key] = value.detach().mean()

    # PC2R-v3 forward-only diagnostics: certificate pass decomposition, actual
    # posterior diversity, correction magnitude, and reliance probes.
    for suffix in (
        "component_risk_overlap_mean",
        "component_raw_count",
        "component_area_pass_count",
        "component_risk_pass_count",
        "component_direction_pass_count",
        "component_spread_pass_count",
        "component_strength_pass_count",
        "component_all_pass_count",
        "selected_posterior_diversity",
        "selected_center_bias_abs",
        "selected_center_bias_signed",
        "all_center_bias_logit_abs",
        "all_center_bias_logit_signed",
        "all_center_bias_prob_abs",
        "selected_center_bias_prob_abs",
        "selected_vs_all_logit_bias_abs",
        "branch_deviation_rms",
        "branch_direction_unanimity",
        "mean_abs_delta_logit",
        "reliance_factualized",
        "reliance_shuffled",
    ):
        value = pred.get(prefix + "_pc2r_" + suffix)
        if isinstance(value, torch.Tensor):
            out["val_pc2r_" + suffix] = value.detach().mean()
    raw_corr = pred.get(prefix + "_c2r_raw_correction_mask")
    risk_support = pred.get(prefix + "_c2r_risk_support_mask")
    if isinstance(raw_corr, torch.Tensor):
        out["val_pc2r_raw_correction_pixel_count"] = (raw_corr.detach() > 0.5).float().sum()
    if isinstance(risk_support, torch.Tensor):
        out["val_pc2r_risk_support_pixel_count"] = (risk_support.detach() > 0.5).float().sum()

    # AEFR forward-only evidence/action diagnostics.  These values are GT-free
    # and never influence deployment or checkpoint selection.  They are folded
    # into the same validation accumulator so E1/E2/E3 can be compared under
    # exactly the existing native protocol.
    if bool(_cfg_get(m1, "GEOTR_AEFR_ENABLED", False)):
        for suffix in (
            "posterior_stability_support",
            "posterior_stability_improvement",
            "posterior_disagreement_pre",
            "posterior_disagreement_post",
            "action_support_fraction",
            "posterior_diversity_all",
            "posterior_center_bias_abs",
            "posterior_center_bias_signed",
            "transition_abs_mean",
            "transition_active_fraction",
            "transition_flip_fraction",
        ):
            value = pred.get(prefix + "_aefr_" + suffix)
            if isinstance(value, torch.Tensor):
                out["val_geotr_aefr_" + suffix] = value.detach().mean()
        bmask = pred.get(prefix + "_aefr_boundary_mask")
        disp = pred.get(prefix + "_aefr_boundary_displacement_px")
        idelta = pred.get(prefix + "_aefr_interior_delta_logit")
        if isinstance(bmask, torch.Tensor):
            bm = (bmask.detach()[:, 0] > 0.5) & region_b
            out["val_geotr_aefr_boundary_fraction_in_region"] = _safe_ratio(bm.float().sum(), region_count)
        else:
            bm = torch.zeros_like(region_b)
        if isinstance(disp, torch.Tensor):
            out["val_geotr_aefr_mean_abs_boundary_displacement_px"] = _masked_mean(disp.detach()[:, 0].abs(), bm)
        if isinstance(idelta, torch.Tensor):
            im = region_b & (~bm)
            out["val_geotr_aefr_mean_abs_interior_delta_logit"] = _masked_mean(idelta.detach()[:, 0].abs(), im)
    return out


def compute_geotr_v4g_validation_diagnostics(cfg, masks, pred):
    """Validation-only V4G/V4G-R2 audit.

    GT is used strictly for diagnostics.  The deployed forward remains GT-free.
    R2 adds hard correction accounting and selector ceilings so WHERE and HOW
    failure can be separated without relying on loss values alone.
    """
    m1 = _cfg_get(cfg, "M1", None)
    if bool(_cfg_get(m1, "GEOTR_C2R_ENABLED", False)):
        return _compute_c2r_validation_diagnostics(cfg, masks, pred)
    if not bool(_cfg_get(m1, "GEOTR_V4G_SPARSE_DIRECT_REFINER_ENABLED", False)):
        return {}
    mode = str(_cfg_get(m1, "GEOTOPO_MODE", "full")).strip().lower()
    if mode not in {"residual", "full"}:
        return {}
    target = _target_3d(masks)
    base = pred.get("geotopo_base_probs")
    geo = pred.get("geotopo_geometry_probs")
    final = pred.get("geotopo_final_probs")
    prefix = "geotopo_reconstruction_after_geometry" if mode == "full" else "geotopo_reconstruction_base"
    selection = pred.get(prefix + "_v4g_selection_mask")
    refined = pred.get(prefix + "_v4g_refined_prob")
    delta = pred.get(prefix + "_v4g_delta_logit")
    margin = pred.get(prefix + "_v4g_margin_uncertainty")
    mc_std = pred.get(prefix + "_v4g_mc_std_map")
    mc_dis = pred.get(prefix + "_v4g_mc_disagreement_map")
    entropy = pred.get(prefix + "_v4g_entropy_map")
    r4_flip_prob = pred.get(prefix + "_v4g_r4_flip_prob")
    if not all(isinstance(x, torch.Tensor) for x in (base, geo, final, selection, refined, margin, mc_std, mc_dis, entropy)):
        return {}
    anchor = geo if mode == "full" else base
    if anchor.ndim == 3:
        anchor = anchor[:, None]
    if target.shape[-2:] != anchor.shape[-2:]:
        target = F.interpolate(target[:, None].float(), size=anchor.shape[-2:], mode="nearest")[:, 0]
    target = target.to(anchor)
    gt = target >= 0.5
    hard_err = ((anchor[:, 0] >= 0.5) != gt)
    stats = _selection_stats(selection, anchor, target)
    anchor_d = _dice_per_case(anchor[:, 0], target)
    final_d = _dice_per_case(final[:, 0], target)
    point_before = _point_accuracy(anchor, target, selection)
    point_after = _point_accuracy(refined, target, selection)

    sel = selection[:, 0] > 0.5
    rh = refined[:, 0] >= 0.5
    corrected = sel & hard_err & (rh == gt)
    introduced = sel & (~hard_err) & (rh != gt)
    corrected_count = corrected.float().sum()
    introduced_count = introduced.float().sum()
    net = corrected_count - introduced_count

    # Pred-selector / perfect-correction ceiling: preserve every selected correct
    # pixel and replace only selected true errors with GT.  This isolates the
    # maximum utility available from the current selector.
    selector_oracle = anchor.clone()
    fix = sel & hard_err
    selector_oracle[:, 0] = torch.where(fix, target, anchor[:, 0])
    selector_oracle_gain = (
        _dice_per_case(selector_oracle[:, 0], target) - anchor_d
    ).mean()
    all_error_oracle = anchor.clone()
    all_error_oracle[:, 0] = torch.where(hard_err, target, anchor[:, 0])
    all_error_oracle_gain = (
        _dice_per_case(all_error_oracle[:, 0], target) - anchor_d
    ).mean()

    r3_cfg = bool(_cfg_get(_cfg_get(cfg, "M1", None), "GEOTR_V4G_R3_MINIMAL_INTERVENTION_ENABLED", False))
    if r3_cfg and isinstance(delta, torch.Tensor):
        gt_sign3 = torch.where(gt, torch.ones_like(target), -torch.ones_like(target))
        err_sel3 = sel & hard_err
        keep_sel3 = sel & (~hard_err)
        pred_abs3 = delta[:,0].detach().abs()
        sign_ok3 = ((delta[:,0].detach() * gt_sign3) > 0.0) & err_sel3
        r3_val = {
            "val_geotr_v4g_r3_pred_abs_delta": _masked_mean(pred_abs3, sel),
            "val_geotr_v4g_r3_error_pred_abs_delta": _masked_mean(pred_abs3, err_sel3),
            "val_geotr_v4g_r3_keep_pred_abs_delta": _masked_mean(pred_abs3, keep_sel3),
            "val_geotr_v4g_r3_error_sign_accuracy": _safe_ratio(sign_ok3.float().sum(), err_sel3.float().sum()),
            "val_geotr_v4g_r3_empirical_error_fraction_selected": _safe_ratio(err_sel3.float().sum(), sel.float().sum()),
        }
    else:
        r3_val = {}

    selected_error_count = (sel & hard_err).float().sum()
    selected_correct_count = (sel & (~hard_err)).float().sum()
    selected_count = sel.float().sum()
    total_error_count = hard_err.float().sum()
    total_pixel_count = hard_err.new_tensor(float(hard_err.numel()), dtype=anchor.dtype)

    r4_cfg = bool(_cfg_get(_cfg_get(cfg, "M1", None), "GEOTR_V4G_R4_EXOGENOUS_PATCH_FLIP_ENABLED", False))
    if r4_cfg and isinstance(r4_flip_prob, torch.Tensor):
        tau4 = float(_cfg_get(_cfg_get(cfg, "M1", None), "GEOTR_V4G_R4_FLIP_THRESHOLD", 0.50))
        flip4 = sel & (r4_flip_prob[:,0].detach() >= tau4)
        tp4 = (flip4 & hard_err).float().sum()
        fp4 = (flip4 & (~hard_err)).float().sum()
        fn4 = ((~flip4) & sel & hard_err).float().sum()
        r4_val = {
            "val_geotr_v4g_r4_flip_probability_mean": _masked_mean(r4_flip_prob[:,0].detach(), sel),
            "val_geotr_v4g_r4_flip_probability_error": _masked_mean(r4_flip_prob[:,0].detach(), sel & hard_err),
            "val_geotr_v4g_r4_flip_probability_correct": _masked_mean(r4_flip_prob[:,0].detach(), sel & (~hard_err)),
            "val_geotr_v4g_r4_flip_tp_count": tp4,
            "val_geotr_v4g_r4_flip_fp_count": fp4,
            "val_geotr_v4g_r4_flip_fn_count": fn4,
            "val_geotr_v4g_r4_flip_precision": _safe_ratio(tp4, tp4 + fp4),
            "val_geotr_v4g_r4_flip_recall_selected": _safe_ratio(tp4, selected_error_count),
            "val_geotr_v4g_r4_false_flip_rate": _safe_ratio(fp4, selected_correct_count),
            "val_geotr_v4g_r4_effective_edit_rate": _safe_ratio(flip4.float().sum(), selected_count),
        }
    else:
        r4_val = {}

    out = {
        "val_geotr_v4g_modelres_soft_gain": (final_d - anchor_d).mean(),
        "val_geotr_v4g_selection_coverage": stats["coverage"],
        "val_geotr_v4g_selection_precision": stats["precision"],
        "val_geotr_v4g_selection_recall": stats["recall"],
        "val_geotr_v4g_anchor_error_rate": stats["error_rate"],
        "val_geotr_v4g_selected_error_count": selected_error_count,
        "val_geotr_v4g_selected_correct_count": selected_correct_count,
        "val_geotr_v4g_selected_count": selected_count,
        "val_geotr_v4g_total_error_count": total_error_count,
        "val_geotr_v4g_total_pixel_count": total_pixel_count,
        "val_geotr_v4g_point_accuracy_before": point_before,
        "val_geotr_v4g_point_accuracy_after": point_after,
        "val_geotr_v4g_point_accuracy_gain": point_after - point_before,
        "val_geotr_v4g_selected_abs_change": _masked_mean((refined[:,0]-anchor[:,0]).abs(), sel),
        "val_geotr_v4g_margin_error_ap": _average_precision_binary(margin[:,0], hard_err),
        "val_geotr_v4g_mc_std_error_ap": _average_precision_binary(mc_std[:,0], hard_err),
        "val_geotr_v4g_mc_disagreement_error_ap": _average_precision_binary(mc_dis[:,0], hard_err),
        "val_geotr_v4g_entropy_error_ap": _average_precision_binary(entropy[:,0], hard_err),
        "val_geotr_v4g_margin_recall_at_05": _topk_recall(margin[:,0], hard_err, 0.05),
        "val_geotr_v4g_margin_recall_at_10": _topk_recall(margin[:,0], hard_err, 0.10),
        "val_geotr_v4g_mc_std_recall_at_05": _topk_recall(mc_std[:,0], hard_err, 0.05),
        "val_geotr_v4g_mc_std_recall_at_10": _topk_recall(mc_std[:,0], hard_err, 0.10),
        "val_geotr_v4g_mc_disagreement_recall_at_05": _topk_recall(mc_dis[:,0], hard_err, 0.05),
        "val_geotr_v4g_mc_disagreement_recall_at_10": _topk_recall(mc_dis[:,0], hard_err, 0.10),
        "val_geotr_v4g_corrected_error_count": corrected_count,
        "val_geotr_v4g_introduced_error_count": introduced_count,
        "val_geotr_v4g_net_correction_count": net,
        "val_geotr_v4g_correction_recall": _safe_ratio(corrected_count, hard_err.float().sum()),
        "val_geotr_v4g_introduction_rate": _safe_ratio(introduced_count, sel.float().sum()),
        "val_geotr_v4g_pred_selector_oracle_gain": selector_oracle_gain,
        "val_geotr_v4g_full_residual_oracle_gain": all_error_oracle_gain,
        "val_geotr_v4g_benefit_case_rate": ((final_d-anchor_d) > 1.0e-8).float().mean(),
        "val_geotr_v4g_harm_case_rate": ((final_d-anchor_d) < -1.0e-8).float().mean(),
    }
    out.update(r3_val)
    out.update(r4_val)
    return out
