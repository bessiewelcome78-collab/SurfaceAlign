# -*- coding: utf-8 -*-
"""V481 M1: failure-aware residual candidate repair.

This module is intentionally independent from M2/M3.  It changes only the M1
candidate geometry and M1 supervision:

  1) local candidates are re-projected as Base-preserving residual edits;
  2) discovery candidates are admitted only through a failure-aware gate;
  3) M1 supervision becomes precision-first and no-harm aware;
  4) diagnostics expose whether M1 is helping or contaminating Base.

No dataset name is used anywhere.  All budgets are computed from the current
train batch distribution: Base area, boundary mass and uncertainty.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

EPS = 1.0e-6


def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def _m1(cfg: Any, key: str, default: Any = None) -> Any:
    return _cfg_get(_cfg_get(cfg, "M1", None), key, default)


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


def _as_bhw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4 and x.shape[1] == 1:
        return x[:, 0]
    if x.ndim == 3:
        return x
    if x.ndim == 2:
        return x[None]
    raise ValueError(f"Expected [B,H,W] or [B,1,H,W], got {tuple(x.shape)}")


def _soft_dilate(x: torch.Tensor, radius: int) -> torch.Tensor:
    radius = max(1, int(radius))
    return F.max_pool2d(x, 2 * radius + 1, 1, radius)


def _soft_erode(x: torch.Tensor, radius: int) -> torch.Tensor:
    radius = max(1, int(radius))
    return -F.max_pool2d(-x, 2 * radius + 1, 1, radius)


def _soft_boundary(prob_bhw: torch.Tensor, radius: int = 1) -> torch.Tensor:
    x = prob_bhw[:, None].clamp(0.0, 1.0)
    return (_soft_dilate(x, radius) - _soft_erode(x, radius)).clamp(0.0, 1.0)[:, 0]


def _entropy(prob_bhw: torch.Tensor) -> torch.Tensor:
    p = prob_bhw.clamp(EPS, 1.0 - EPS)
    h = -(p * p.log() + (1.0 - p) * (1.0 - p).log())
    return (h / 0.6931471805599453).clamp(0.0, 1.0)


def _safe_logit(prob: torch.Tensor) -> torch.Tensor:
    return torch.logit(prob.clamp(EPS, 1.0 - EPS))


def _soft_dice(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prob.ndim == 3:
        inter = (prob * target).flatten(1).sum(dim=1)
        den = prob.flatten(1).sum(dim=1) + target.flatten(1).sum(dim=1)
        return (2.0 * inter + EPS) / (den + EPS)
    if prob.ndim == 4:
        if target.ndim == 3:
            target = target[:, None].expand_as(prob)
        inter = (prob * target).flatten(2).sum(dim=2)
        den = prob.flatten(2).sum(dim=2) + target.flatten(2).sum(dim=2)
        return (2.0 * inter + EPS) / (den + EPS)
    raise ValueError(tuple(prob.shape))


def _top_cvar(values: torch.Tensor, fraction: float) -> torch.Tensor:
    values = values.flatten()
    if values.numel() == 0:
        return values.new_zeros(())
    fraction = min(max(float(fraction), 1.0 / max(values.numel(), 1)), 1.0)
    k = max(1, int(float(values.numel()) * fraction + 0.999))
    return torch.topk(values, k=k, largest=True).values.mean()


def _adaptive_budget(
    cfg: Any,
    base_prob: torch.Tensor,
    base_boundary: torch.Tensor,
    base_entropy: torch.Tensor,
) -> torch.Tensor:
    """Per-case edit budget from current prediction statistics.

    This is the cross-dataset adaptive part.  It uses no dataset name and no
    validation/test labels.
    """
    b_area = base_prob.flatten(1).mean(dim=1)
    b_boundary = base_boundary.flatten(1).mean(dim=1)
    b_entropy = base_entropy.flatten(1).mean(dim=1)

    min_budget = float(_m1(cfg, "CEM_V481_MIN_EDIT_BUDGET", 0.0025))
    max_budget = float(_m1(cfg, "CEM_V481_MAX_EDIT_BUDGET", 0.0800))

    base_factor = float(_m1(cfg, "CEM_V481_BUDGET_BASE_FACTOR", 0.12))
    boundary_factor = float(_m1(cfg, "CEM_V481_BUDGET_BOUNDARY_FACTOR", 0.35))
    entropy_factor = float(_m1(cfg, "CEM_V481_BUDGET_UNCERT_FACTOR", 0.25))

    raw = (
        base_factor * b_area
        + boundary_factor * b_boundary
        + entropy_factor * b_entropy
    )

    budget = raw.clamp(min_budget, max_budget)
    return budget[:, None, None]


def _action_types_from_aux(
    aux: Dict[str, torch.Tensor],
    action_count: int,
    device,
) -> torch.Tensor:
    value = aux.get("v20_action_types")
    if isinstance(value, torch.Tensor):
        out = value.detach().to(device=device, dtype=torch.long).reshape(-1)
        if out.numel() >= action_count:
            return out[:action_count]
    # fallback: 0 delete, 1 fill, 2 trim, 3 expand, remaining discovery
    if action_count <= 0:
        return torch.zeros(0, device=device, dtype=torch.long)
    fallback = torch.arange(action_count, device=device, dtype=torch.long)
    return fallback.clamp_max(4)


def _action_supports_from_aux(
    aux: Dict[str, torch.Tensor],
    action_count: int,
    base_boundary: torch.Tensor,
    base_entropy: torch.Tensor,
) -> torch.Tensor:
    value = aux.get("v20_action_supports")
    if isinstance(value, torch.Tensor) and value.ndim == 4:
        support = value.float()
        if support.shape[1] >= action_count:
            return support[:, :action_count].clamp(0.0, 1.0)

    fallback = (0.65 * base_boundary + 0.35 * base_entropy).clamp(0.0, 1.0)
    return fallback[:, None].expand(-1, action_count, -1, -1).contiguous()


def _failure_gate_from_aux(
    cfg: Any,
    aux: Dict[str, torch.Tensor],
    base_prob: torch.Tensor,
    base_entropy: torch.Tensor,
) -> torch.Tensor:
    value = aux.get("cem_failure_logit")
    if isinstance(value, torch.Tensor):
        logit = value.float().reshape(base_prob.shape[0], -1).mean(dim=1)
    else:
        # fallback proxy: empty/very uncertain predictions should admit rediscovery.
        area = base_prob.flatten(1).mean(dim=1)
        ent = base_entropy.flatten(1).mean(dim=1)
        low_area = torch.sigmoid((0.01 - area) / 0.01)
        high_unc = torch.sigmoid((ent - 0.25) / 0.10)
        proxy = 0.5 * low_area + 0.5 * high_unc
        logit = _safe_logit(proxy.clamp(0.01, 0.99))

    threshold = float(_m1(cfg, "CEM_V481_DISCOVERY_FAILURE_THRESHOLD", 0.50))
    temperature = max(float(_m1(cfg, "CEM_V481_DISCOVERY_TEMPERATURE", 0.20)), 1.0e-3)
    return torch.sigmoid((logit - threshold) / temperature)[:, None, None]


def v481_reproject_m1_candidates(
    cfg: Any,
    base_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    aux: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Re-project generated candidates into safe V481 geometry.

    C0 is preserved exactly.
    Local C1..Ck candidates become bounded residual edits around C0.
    Discovery candidates are allowed to differ globally only when the failure
    gate is high.
    """
    if not bool(_m1(cfg, "CEM_V481_ENABLED", False)):
        return candidate_logits, aux

    if candidate_logits.ndim != 4 or candidate_logits.shape[1] < 2:
        return candidate_logits, aux

    base_slot = candidate_logits[:, :1]

    # V482: Base is still trainable through the main segmentation loss,
    # but candidate auxiliary losses must not optimize Base by using C0 as
    # a shortcut.  The residual candidates are therefore anchored on a
    # detached Base reference, while C0 itself is preserved as the first slot.
    detach_base_anchor = bool(_m1(cfg, "CEM_V482_DETACH_BASE_ANCHOR", True))
    base_anchor = base_slot.detach() if detach_base_anchor else base_slot
    base_prob = torch.sigmoid(base_anchor[:, 0]).clamp(EPS, 1.0 - EPS)

    action_count = candidate_logits.shape[1] - 1
    action_types = _action_types_from_aux(
        aux,
        action_count=action_count,
        device=candidate_logits.device,
    )

    base_boundary = _soft_boundary(base_prob, radius=int(_m1(cfg, "CEM_V481_BOUNDARY_RADIUS", 1)))
    base_entropy = _entropy(base_prob)

    support = _action_supports_from_aux(
        aux,
        action_count=action_count,
        base_boundary=base_boundary,
        base_entropy=base_entropy,
    ).to(device=candidate_logits.device, dtype=candidate_logits.dtype)

    # Local support must stay near Base boundary / uncertainty.  This prevents
    # "full-mask replacement" behavior for local candidates.
    local_prior = (
        float(_m1(cfg, "CEM_V481_BOUNDARY_PRIOR_WEIGHT", 0.65)) * base_boundary
        + float(_m1(cfg, "CEM_V481_ENTROPY_PRIOR_WEIGHT", 0.35)) * base_entropy
    ).clamp(0.0, 1.0)

    local_prior = local_prior[:, None].expand_as(support)
    support = (support * local_prior).clamp(0.0, 1.0)

    # Per-case adaptive support area budget.
    budget = _adaptive_budget(cfg, base_prob, base_boundary, base_entropy)
    area = support.flatten(2).mean(dim=2).clamp_min(EPS)
    scale = (budget[:, 0, 0][:, None] / area).clamp(max=1.0)
    support = support * scale[:, :, None, None]

    old_action_logits = candidate_logits[:, 1:]
    raw_delta = old_action_logits - base_anchor
    local_delta_cap = float(_m1(cfg, "CEM_V481_LOCAL_DELTA_CAP", 2.00))
    global_delta_cap = float(_m1(cfg, "CEM_V481_GLOBAL_DELTA_CAP", 4.00))

    # Typed sign constraints:
    #   0 / 2 : delete / trim      -> negative residual
    #   1 / 3 : fill / expand      -> positive residual
    #   >= 4  : discovery/global   -> gated full proposal
    abs_delta = raw_delta.abs()
    neg_delta = -abs_delta.clamp(max=local_delta_cap)
    pos_delta = abs_delta.clamp(max=local_delta_cap)
    free_delta = raw_delta.clamp(-local_delta_cap, local_delta_cap)

    at = action_types[None, :, None, None]
    signed_delta = torch.where(
        (at == 0) | (at == 2),
        neg_delta,
        torch.where((at == 1) | (at == 3), pos_delta, free_delta),
    )

    local_logits = base_anchor + support * signed_delta

    # Discovery/global candidates are not removed.  They are failure-admitted:
    # if Base seems reliable, they collapse toward C0; if Base seems failed,
    # they can move further to rediscover the target.
    failure_gate = _failure_gate_from_aux(cfg, aux, base_prob, base_entropy)
    old_prob = torch.sigmoid(old_action_logits).clamp(EPS, 1.0 - EPS)
    global_prob = (
        (1.0 - failure_gate[:, None]) * base_prob[:, None]
        + failure_gate[:, None] * old_prob
    ).clamp(EPS, 1.0 - EPS)
    global_logits = _safe_logit(global_prob).clamp(
        base_anchor - global_delta_cap,
        base_anchor + global_delta_cap,
    )

    is_global = (action_types >= 4)[None, :, None, None]
    new_action_logits = torch.where(is_global, global_logits, local_logits)
    new_logits = torch.cat([base_slot, new_action_logits], dim=1)

    new_probs = torch.sigmoid(new_logits).clamp(EPS, 1.0 - EPS)
    edit_fraction = (new_probs[:, 1:] - new_probs[:, :1]).abs().flatten(2).mean(dim=2)

    aux = dict(aux)
    aux["candidate_probs"] = new_probs
    aux["tpmhg_hypothesis_logits"] = new_logits[:, 1:]
    aux["tpmhg_hypothesis_probs"] = new_probs[:, 1:]
    aux["v20_budget"] = edit_fraction
    aux["local_action_area"] = edit_fraction
    aux["v481_reprojected"] = new_logits.new_ones(new_logits.shape[0])
    aux["v481_adaptive_edit_budget"] = budget[:, 0, 0].detach()
    aux["v481_failure_gate"] = failure_gate[:, 0, 0].detach()
    aux["v481_local_support_area"] = support.flatten(2).mean(dim=2).mean(dim=1).detach()
    aux["v481_mean_edit_fraction"] = edit_fraction.mean(dim=1).detach()
    aux["v481_discovery_action_rate"] = (
        (action_types >= 4).float().mean().expand(new_logits.shape[0])
    )

    # If a hard selected index is already in aux, recompute the selected mask
    # under the re-projected candidate geometry.
    for key in ("selected_index", "cem_selected_index"):
        if key in aux and isinstance(aux[key], torch.Tensor):
            idx = aux[key].long().clamp(0, new_probs.shape[1] - 1)
            gather = idx[:, None, None, None].expand(
                -1, 1, new_probs.shape[-2], new_probs.shape[-1]
            )
            hard = new_probs.gather(1, gather)[:, 0]
            aux["hard_final"] = hard
            aux["m1_hard_fused_probs"] = hard
            aux["direct_fused_probs"] = hard
            aux["router_fused_probs"] = hard

    return new_logits, aux


def _targets_for_action_type(
    gt: torch.Tensor,
    base_prob: torch.Tensor,
    action_type: int,
    boundary_radius: int,
) -> torch.Tensor:
    pred = (base_prob >= 0.5).float()
    fp = (pred * (1.0 - gt)).clamp(0.0, 1.0)
    fn = ((1.0 - pred) * gt).clamp(0.0, 1.0)
    gt_boundary = _soft_boundary(gt, radius=boundary_radius)
    base_boundary = _soft_boundary(base_prob, radius=boundary_radius)
    boundary = (gt_boundary + base_boundary).clamp(0.0, 1.0)

    if action_type == 0:      # FP delete
        return fp
    if action_type == 1:      # FN fill
        return fn
    if action_type == 2:      # boundary trim
        return (fp + boundary * pred).clamp(0.0, 1.0)
    if action_type == 3:      # boundary expand
        return (fn + boundary * (1.0 - pred)).clamp(0.0, 1.0)
    return torch.ones_like(gt)


def compute_v481_m1_precision_loss(
    cfg: Any,
    candidates: torch.Tensor,
    masks: torch.Tensor,
    aux: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Precision-first M1 regularizer.

    It does not replace the old CEM objective; it adds constraints that force
    candidates to be local, precise and no-harm.
    """
    if not bool(_m1(cfg, "CEM_V481_ENABLED", False)):
        return candidates.new_zeros(()), {}

    gt = _foreground_mask(masks).to(device=candidates.device, dtype=candidates.dtype)
    probs = torch.sigmoid(candidates).clamp(EPS, 1.0 - EPS)
    base = probs[:, 0]
    actions = probs[:, 1:]

    if actions.numel() == 0:
        return candidates.new_zeros(()), {}

    action_count = actions.shape[1]
    action_types = _action_types_from_aux(aux, action_count, candidates.device)

    base_boundary = _soft_boundary(base, radius=int(_m1(cfg, "CEM_V481_BOUNDARY_RADIUS", 1)))
    base_entropy = _entropy(base)
    budget = _adaptive_budget(cfg, base, base_boundary, base_entropy)[:, 0, 0]

    edit = (actions - base[:, None]).abs()
    edit_area = edit.flatten(2).mean(dim=2)

    precision_losses = []
    correction_losses = []
    tp_remove_losses = []
    bg_leak_losses = []

    for k in range(action_count):
        at = int(action_types[k].detach().cpu())
        target = _targets_for_action_type(
            gt=gt,
            base_prob=base,
            action_type=at,
            boundary_radius=int(_m1(cfg, "CEM_V481_BOUNDARY_RADIUS", 1)),
        )

        e = edit[:, k]
        denom = e.flatten(1).sum(dim=1).clamp_min(EPS)
        outside = (e * (1.0 - target)).flatten(1).sum(dim=1) / denom
        inside = (e * target).flatten(1).sum(dim=1) / denom

        precision_losses.append(outside)
        correction_losses.append(1.0 - inside)

        # Penalize removing GT foreground and adding background mass.
        delete_mass = (base - actions[:, k]).relu()
        add_mass = (actions[:, k] - base).relu()
        tp_remove_losses.append((delete_mass * gt).flatten(1).mean(dim=1))
        bg_leak_losses.append((add_mass * (1.0 - gt)).flatten(1).mean(dim=1))

    precision_loss = torch.stack(precision_losses, dim=1).mean()
    correction_precision_loss = torch.stack(correction_losses, dim=1).mean()
    tp_remove_loss = torch.stack(tp_remove_losses, dim=1).mean()
    bg_leak_loss = torch.stack(bg_leak_losses, dim=1).mean()

    base_dice = _soft_dice(base, gt)
    action_dice = _soft_dice(actions, gt)
    harm = (base_dice[:, None] - action_dice).relu()
    harm_mean = harm.mean()
    harm_tail = _top_cvar(
        harm,
        fraction=float(_m1(cfg, "CEM_V481_TAIL_FRACTION", 0.25)),
    )

    edit_budget_loss = (edit_area - budget[:, None]).relu().pow(2).mean()

    w_precision = float(_m1(cfg, "CEM_V481_SUPPORT_PRECISION_WEIGHT", 1.00))
    w_correction = float(_m1(cfg, "CEM_V481_CORRECTION_PRECISION_WEIGHT", 1.00))
    w_tp = float(_m1(cfg, "CEM_V481_TP_REMOVAL_WEIGHT", 1.00))
    w_bg = float(_m1(cfg, "CEM_V481_BG_LEAKAGE_WEIGHT", 1.00))
    w_harm = float(_m1(cfg, "CEM_V481_NO_HARM_WEIGHT", 1.25))
    w_tail = float(_m1(cfg, "CEM_V481_TAIL_WEIGHT", 1.00))
    w_budget = float(_m1(cfg, "CEM_V481_EDIT_BUDGET_WEIGHT", 0.75))

    loss = (
        w_precision * precision_loss
        + w_correction * correction_precision_loss
        + w_tp * tp_remove_loss
        + w_bg * bg_leak_loss
        + w_harm * harm_mean
        + w_tail * harm_tail
        + w_budget * edit_budget_loss
    )

    diagnostics = {
        "cem_v481_loss": loss.detach(),
        "cem_v481_support_precision_loss": precision_loss.detach(),
        "cem_v481_correction_precision_loss": correction_precision_loss.detach(),
        "cem_v481_tp_removal_loss": tp_remove_loss.detach(),
        "cem_v481_bg_leakage_loss": bg_leak_loss.detach(),
        "cem_v481_harm_mean": harm_mean.detach(),
        "cem_v481_harm_tail": harm_tail.detach(),
        "cem_v481_edit_budget_loss": edit_budget_loss.detach(),
        "cem_v481_mean_edit_fraction": edit_area.mean().detach(),
        "cem_v481_adaptive_budget_mean": budget.mean().detach(),
        "cem_v481_candidate_oracle_dice": action_dice.max(dim=1).values.mean().detach(),
        "cem_v481_base_dice": base_dice.mean().detach(),
        "cem_v481_oracle_gain": (
            action_dice.max(dim=1).values - base_dice
        ).mean().detach(),
        "cem_v481_harmful_candidate_rate": (
            (action_dice < base_dice[:, None] - 5.0e-4).float().mean().detach()
        ),
    }

    return loss, diagnostics
