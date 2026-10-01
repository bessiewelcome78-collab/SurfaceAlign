"""Losses and teacher construction for V538 one-run component refinement."""
from __future__ import annotations

from typing import Any, Dict, Tuple
import math

import numpy as np
import torch
import torch.nn.functional as F

from utils.v551_loss import compute_v551_multiscale_editor_loss

try:
    from scipy import ndimage
except Exception as exc:  # pragma: no cover
    ndimage = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None

EPS = 1.0e-6

# V546 keeps the empirical outcome prior inside the current process and resets
# it at every epoch.  This is deliberately an exact cumulative count, not an
# EMA with a tuned decay.  Each formal ablation runs in its own process.
_V546_STREAMING_PRIOR_STATE: Dict[str, Dict[str, Any]] = {}

# V549 stores detached post-trunk selector features.  It is retained only for
# backward-compatible regression tests; V550 disables it because both the
# representation and its Benefit/Harm label become stale in joint E2E training.
_V549_BALANCED_QUEUE_STATE: Dict[str, Dict[str, Any]] = {}

# V550 keeps only exact cumulative class counts for the current epoch.  No
# feature, mask, image, label, or computation graph is replayed.  Every V550
# Direction/Sign gradient is therefore computed from the current selector
# representation and the current exact candidate Dice gain.
_V550_STREAMING_BALANCE_STATE: Dict[str, Dict[str, Any]] = {}


def reset_v549_balanced_queue_state() -> None:
    """Clear the process-local V549 Benefit/Harm feature queues."""
    _V549_BALANCED_QUEUE_STATE.clear()


def reset_v550_streaming_balance_state() -> None:
    """Clear process-local V550 current-semantic class-count state."""
    _V550_STREAMING_BALANCE_STATE.clear()


def reset_v546_streaming_prior_state() -> None:
    """Clear process-local V546 prior counts (used by contract tests)."""
    _V546_STREAMING_PRIOR_STATE.clear()


def _v546_streaming_class_counts(
    cfg: Any,
    epoch: int,
    batch_counts: torch.Tensor,
) -> torch.Tensor:
    run_tag = str(_m1(cfg, "RUN_TAG", "V546_DEFAULT"))
    state = _V546_STREAMING_PRIOR_STATE.get(run_tag)
    if state is None or int(state.get("epoch", -1)) != int(epoch):
        state = {
            "epoch": int(epoch),
            "counts": torch.zeros(3, dtype=torch.float64),
        }
        _V546_STREAMING_PRIOR_STATE[run_tag] = state
    state["counts"] += batch_counts.detach().to("cpu", torch.float64)
    return state["counts"].to(
        device=batch_counts.device,
        dtype=batch_counts.dtype,
    )


def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def _r4209_centernet_focal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Canonical CenterNet-style modified focal loss for center heatmaps.

    The exponents (positive/negative focusing 2 and Gaussian negative weight 4)
    are the standard CenterNet formulation, not tunable project-specific
    hyperparameters.  Exact center pixels must be 1; Gaussian shoulders are
    soft negatives.
    """
    pred = torch.sigmoid(logits).clamp(1.0e-6, 1.0 - 1.0e-6)
    pos = target.eq(1.0).to(pred.dtype)
    neg = target.lt(1.0).to(pred.dtype)
    neg_weight = (1.0 - target).pow(4.0)
    pos_loss = torch.log(pred) * (1.0 - pred).pow(2.0) * pos
    neg_loss = torch.log(1.0 - pred) * pred.pow(2.0) * neg_weight * neg
    num_pos = pos.sum()
    if bool((num_pos > 0).item()):
        return -(pos_loss.sum() + neg_loss.sum()) / num_pos
    return -neg_loss.sum()


def _r4210_component_interior_anchor(
    teacher_masks: torch.Tensor,
    teacher_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return one deterministic, guaranteed-inside anchor per Teacher mask.

    Bounding-box centres are not valid instance representatives for crescent,
    ring, boundary-band, or other irregular residual components.  R4.21.0 uses
    the mask centroid only as a *reference* and selects the actual component
    pixel nearest to that centroid.  The returned anchor therefore always lies
    on the residual component while introducing no distance threshold, radius,
    learned offset, or extra parameter.

    Returns:
        anchor_xy: [B,K,2] in pixel-centre coordinates ((x+0.5)/W,(y+0.5)/H)
        anchor_index: [B,K] flattened integer pixel index
        inside_rate: scalar diagnostic (must be 1 for every valid component)
    """
    if teacher_masks.ndim != 4:
        raise ValueError(f"teacher_masks must be [B,K,H,W], got {tuple(teacher_masks.shape)}")
    b, k, h, w = teacher_masks.shape
    dtype = teacher_masks.dtype
    device = teacher_masks.device
    mask = (teacher_masks > 0.5) & teacher_valid[:, :, None, None]
    yy = torch.arange(h, device=device, dtype=dtype)[None, None, :, None]
    xx = torch.arange(w, device=device, dtype=dtype)[None, None, None, :]
    mass = mask.to(dtype).sum(dim=(-2, -1)).clamp_min(1.0)
    cx = (mask.to(dtype) * xx).sum(dim=(-2, -1)) / mass
    cy = (mask.to(dtype) * yy).sum(dim=(-2, -1)) / mass
    dist2 = (xx - cx[:, :, None, None]).square() + (yy - cy[:, :, None, None]).square()
    inf = torch.finfo(dtype).max
    dist2 = torch.where(mask, dist2, dist2.new_full((), inf))
    flat_index = dist2.flatten(2).argmin(dim=2)
    py = torch.div(flat_index, w, rounding_mode="floor")
    px = flat_index.remainder(w)
    anchor_x = (px.to(dtype) + 0.5) / float(max(w, 1))
    anchor_y = (py.to(dtype) + 0.5) / float(max(h, 1))
    anchor_xy = torch.stack([anchor_x, anchor_y], dim=2)
    anchor_xy = torch.where(teacher_valid[:, :, None], anchor_xy, torch.zeros_like(anchor_xy))
    batch = torch.arange(b, device=device)[:, None].expand(b, k)
    slot = torch.arange(k, device=device)[None, :].expand(b, k)
    anchor_inside = mask[batch, slot, py, px]
    if bool(teacher_valid.any().item()):
        inside_rate = anchor_inside[teacher_valid].to(dtype).mean()
    else:
        inside_rate = teacher_masks.new_ones(())
    return anchor_xy, flat_index, inside_rate.detach()


def _m1(cfg: Any, key: str, default: Any = None) -> Any:
    return _cfg_get(_cfg_get(cfg, "M1", None), key, default)


def _as_b1hw(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 4:
        return value[:, :1]
    if value.ndim == 3:
        return value[:, None]
    raise ValueError(f"Expected [B,H,W] or [B,1,H,W], got {tuple(value.shape)}")


def _dice_many(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Dice for ``pred=[B,K,H,W]`` and ``target=[B,1,H,W]``."""
    target = _as_b1hw(target)[:, 0][:, None].to(pred.dtype)
    inter = (pred * target).sum(dim=(-2, -1))
    den = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return (2.0 * inter + EPS) / (den + EPS)


def _dice_binary_many(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return _dice_many(pred.to(target.dtype), target)


def _morphology(mask: torch.Tensor, radius: int, mode: str) -> torch.Tensor:
    radius = max(int(radius), 1)
    kernel = 2 * radius + 1
    if mode == "dilate":
        return F.max_pool2d(mask, kernel, stride=1, padding=radius)
    if mode == "erode":
        return 1.0 - F.max_pool2d(1.0 - mask, kernel, stride=1, padding=radius)
    raise ValueError(mode)


def _dose_grid(cfg: Any, reference: torch.Tensor) -> torch.Tensor:
    values = _m1(cfg, "V540_TEACHER_DOSE_GRID", [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    if not isinstance(values, (list, tuple)) or not values:
        values = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    minimum = max(float(_m1(cfg, "V540_MINIMUM_DOSE", 0.25)), 0.0)
    maximum = max(float(_m1(cfg, "V540_MAXIMUM_DOSE", 16.0)), minimum + EPS)
    cleaned = sorted({min(max(float(value), minimum), maximum) for value in values})
    if not cleaned:
        cleaned = [minimum, maximum]
    return reference.new_tensor(cleaned)


def _build_teacher_components(
    *,
    teacher_probability: torch.Tensor,
    gt: torch.Tensor,
    num_slots: int,
    min_pixels: int,
    current_epoch: int,
    replay_enabled: bool,
    replay_error_floor: float,
    replay_radius: int,
    utility_rank_enabled: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Build atomic FP/FN component teachers from a lagged Base.

    When the lagged training Base has almost no residual error, deterministic
    erosion/dilation replay supplies boundary FP/FN components.  This prevents
    the candidate-positive distribution from collapsing while retaining a
    one-run, from-scratch protocol.
    """
    if ndimage is None:
        raise RuntimeError(
            "V538 component teacher requires scipy.ndimage; original error: "
            + repr(_SCIPY_IMPORT_ERROR)
        )
    teacher = (_as_b1hw(teacher_probability).detach() >= 0.5).float()
    target = (_as_b1hw(gt).detach() >= 0.5).float()
    residual_fraction = (teacher != target).float().flatten(1).mean(dim=1)
    replay_case = residual_fraction < max(float(replay_error_floor), 0.0)

    if replay_enabled and bool(replay_case.any().item()):
        eroded = _morphology(target, replay_radius, "erode")
        dilated = _morphology(target, replay_radius, "dilate")
        # Alternate replay polarity across epochs and samples so both FP and FN
        # components remain represented without introducing random labels.
        replay_masks = teacher.clone()
        for sample in range(teacher.shape[0]):
            if not bool(replay_case[sample].item()):
                continue
            use_dilation = ((int(current_epoch) + sample) % 2) == 1
            replay_masks[sample] = dilated[sample] if use_dilation else eroded[sample]
        teacher = torch.where(replay_case[:, None, None, None], replay_masks, teacher)
    else:
        replay_case = torch.zeros_like(replay_case)

    b, _, h, w = teacher.shape
    k = max(int(num_slots), 1)
    device = teacher.device
    dtype = teacher.dtype
    component_masks = torch.zeros((b, k, h, w), device=device, dtype=dtype)
    component_actions = torch.zeros((b, k), device=device, dtype=torch.long)
    component_valid = torch.zeros((b, k), device=device, dtype=torch.bool)
    component_area = torch.zeros((b, k), device=device, dtype=dtype)
    raw_component_count = torch.zeros((b,), device=device, dtype=dtype)
    structure = np.ones((3, 3), dtype=np.uint8)

    teacher_np = teacher[:, 0].cpu().numpy().astype(bool)
    target_np = target[:, 0].cpu().numpy().astype(bool)
    min_pixels = max(int(min_pixels), 1)

    for sample in range(b):
        base_np = teacher_np[sample]
        gt_np = target_np[sample]
        fp_np = base_np & (~gt_np)
        fn_np = (~base_np) & gt_np
        gt_dilated = ndimage.binary_dilation(gt_np, structure=structure)
        base_dilated = ndimage.binary_dilation(base_np, structure=structure)
        entries = []

        fp_labels, fp_count = ndimage.label(fp_np, structure=structure)
        for label_id in range(1, fp_count + 1):
            component = fp_labels == label_id
            area = int(component.sum())
            if area < min_pixels:
                continue
            touches_gt = bool((component & gt_dilated).any())
            action = 2 if touches_gt else 0  # Trim or Delete.
            entries.append((area, action, component))

        fn_labels, fn_count = ndimage.label(fn_np, structure=structure)
        for label_id in range(1, fn_count + 1):
            component = fn_labels == label_id
            area = int(component.sum())
            if area < min_pixels:
                continue
            touches_base = bool((component & base_dilated).any())
            action = 3 if touches_base else 1  # Expand or Fill.
            entries.append((area, action, component))

        # V552-R4.11: teacher capacity must be spent on components that can
        # actually improve the current Native Base, not merely on the largest
        # residual regions.  Every component consists only of current Base
        # errors, so its exact hard correction is available without inventing
        # a proxy objective.  Gain is used first; area only breaks ties.
        base_intersection = float((base_np & gt_np).sum())
        base_denominator = float(base_np.sum() + gt_np.sum())
        base_dice = (2.0 * base_intersection + 1.0e-6) / (base_denominator + 1.0e-6)
        utility_entries = []
        for area, action, component in entries:
            if utility_rank_enabled:
                corrected = base_np.copy()
                corrected[component] = gt_np[component]
                corrected_intersection = float((corrected & gt_np).sum())
                corrected_denominator = float(corrected.sum() + gt_np.sum())
                corrected_dice = (2.0 * corrected_intersection + 1.0e-6) / (
                    corrected_denominator + 1.0e-6
                )
                utility_gain = corrected_dice - base_dice
            else:
                utility_gain = 0.0
            utility_entries.append((utility_gain, area, action, component))

        if utility_rank_enabled:
            utility_entries.sort(key=lambda item: (item[0], item[1]), reverse=True)
        else:
            utility_entries.sort(key=lambda item: item[1], reverse=True)
        raw_component_count[sample] = float(len(utility_entries))
        for slot, (_utility_gain, area, action, component) in enumerate(utility_entries[:k]):
            component_masks[sample, slot] = torch.from_numpy(
                component.astype(np.float32)
            ).to(device=device, dtype=dtype)
            component_actions[sample, slot] = int(action)
            component_valid[sample, slot] = True
            component_area[sample, slot] = float(area) / float(max(h * w, 1))

    effective_teacher_error = (teacher != target).to(dtype)
    return (
        component_masks,
        component_actions,
        component_valid,
        component_area,
        replay_case,
        teacher,
        effective_teacher_error,
        raw_component_count,
    )


def _greedy_unique_match(
    pair_dice: torch.Tensor,
    target_valid: torch.Tensor,
    target_area: torch.Tensor,
    target_priority: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match each teacher component to one unique predicted slot.

    V541 processes high-utility teacher components before low-utility residuals
    when slots are limited.  Spatial matching itself remains Dice based.
    """
    b, pred_k, target_k = pair_dice.shape
    matched_target = torch.full(
        (b, pred_k), -1, device=pair_dice.device, dtype=torch.long
    )
    matched = torch.zeros((b, pred_k), device=pair_dice.device, dtype=torch.bool)
    detached = pair_dice.detach()
    for sample in range(b):
        available = set(range(pred_k))
        targets = [
            index for index in range(target_k)
            if bool(target_valid[sample, index].item())
        ]
        targets.sort(
            key=lambda index: (
                float(
                    target_priority[sample, index].item()
                    if target_priority is not None else 0.0
                ),
                float(target_area[sample, index].item()),
            ),
            reverse=True,
        )
        for target_index in targets:
            if not available:
                break
            best_pred = max(
                available,
                key=lambda pred_index: float(
                    detached[sample, pred_index, target_index].item()
                ),
            )
            available.remove(best_pred)
            matched_target[sample, best_pred] = int(target_index)
            matched[sample, best_pred] = True
    return matched_target, matched


def _optimal_unique_match(
    pair_dice: torch.Tensor,
    target_valid: torch.Tensor,
    target_area: torch.Tensor,
    target_priority: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact maximum-total-Dice assignment for the small component set.

    BUSI uses at most eight predicted slots.  Dynamic programming over the
    used-slot bitmask is therefore exact and inexpensive.  High-utility/large
    targets are selected first only when valid targets outnumber slots; after
    that, assignment is globally optimal and independent of slot order.
    """
    b, pred_k, target_k = pair_dice.shape
    matched_target = torch.full(
        (b, pred_k), -1, device=pair_dice.device, dtype=torch.long
    )
    matched = torch.zeros((b, pred_k), device=pair_dice.device, dtype=torch.bool)
    detached = pair_dice.detach()
    for sample in range(b):
        targets = [
            index for index in range(target_k)
            if bool(target_valid[sample, index].item())
        ]
        targets.sort(
            key=lambda index: (
                float(
                    target_priority[sample, index].item()
                    if target_priority is not None else 0.0
                ),
                float(target_area[sample, index].item()),
                -int(index),
            ),
            reverse=True,
        )
        targets = targets[:pred_k]
        # state: used-slot mask -> (total Dice, assigned slots in target order)
        states = {0: (0.0, tuple())}
        for target_index in targets:
            next_states = {}
            for used_mask, (score, assignment) in states.items():
                for pred_index in range(pred_k):
                    if used_mask & (1 << pred_index):
                        continue
                    new_mask = used_mask | (1 << pred_index)
                    new_score = score + float(
                        detached[sample, pred_index, target_index].item()
                    )
                    new_assignment = assignment + (pred_index,)
                    previous = next_states.get(new_mask)
                    if (
                        previous is None
                        or new_score > previous[0] + 1.0e-12
                        or (
                            abs(new_score - previous[0]) <= 1.0e-12
                            and new_assignment < previous[1]
                        )
                    ):
                        next_states[new_mask] = (new_score, new_assignment)
            states = next_states
        if not targets:
            continue
        _, (_, best_assignment) = max(
            states.items(),
            key=lambda item: (item[1][0], tuple(-x for x in item[1][1])),
        )
        for target_index, pred_index in zip(targets, best_assignment):
            matched_target[sample, pred_index] = int(target_index)
            matched[sample, pred_index] = True
    return matched_target, matched


def _v565_build_center_heatmap(
    teacher_masks: torch.Tensor,
    teacher_valid: torch.Tensor,
    *,
    sigma_min_px: float = 1.5,
    sigma_max_px: float = 4.0,
):
    """One inference-aligned interior Gaussian center per factual component."""
    b, t, h, w = teacher_masks.shape
    dtype, device = teacher_masks.dtype, teacher_masks.device
    yy = torch.arange(h, device=device, dtype=dtype)[:, None].expand(h, w)
    xx = torch.arange(w, device=device, dtype=dtype)[None, :].expand(h, w)
    target = teacher_masks.new_zeros((b, 1, h, w))
    center_yx = torch.full((b, t, 2), -1, device=device, dtype=torch.long)
    for bi in range(b):
        for ti in range(t):
            if not bool(teacher_valid[bi, ti].item()):
                continue
            coords = (teacher_masks[bi, ti] > 0.5).nonzero(as_tuple=False)
            if coords.numel() == 0:
                continue
            centroid = coords.to(dtype).mean(dim=0)
            chosen = coords[(coords.to(dtype) - centroid[None]).square().sum(dim=1).argmin()]
            cy, cx = int(chosen[0].item()), int(chosen[1].item())
            center_yx[bi, ti] = torch.tensor([cy, cx], device=device)
            sigma = 0.35 * math.sqrt(max(float(coords.shape[0]), 1.0) / math.pi)
            sigma = min(max(sigma, float(sigma_min_px)), float(sigma_max_px))
            gaussian = torch.exp(-((yy - float(cy)).square() + (xx - float(cx)).square()) / (2.0 * sigma * sigma))
            target[bi, 0] = torch.maximum(target[bi, 0], gaussian)
    return target, center_yx


def _v564_strict_anchor_ownership_match(
    teacher_masks: torch.Tensor,
    teacher_valid: torch.Tensor,
    anchor_xy: torch.Tensor,
    anchor_score: torch.Tensor,
    target_priority: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Strict one-seed-one-component ownership used by V564.

    A slot may own a teacher component only when its *predicted* anchor pixel
    lies inside that factual residual component.  Duplicate seeds for one
    component are resolved by the highest proposal score (then the lowest slot
    index).  There is deliberately no fallback Hungarian assignment: an anchor
    without a physical owner is a no-object slot, and a teacher without a seed
    is a proposal miss.  This function is training-only; Native inference never
    sees teacher masks.
    """
    b, target_k, h, w = teacher_masks.shape
    pred_k = anchor_xy.shape[1]
    matched_target = torch.full(
        (b, pred_k), -1, device=teacher_masks.device, dtype=torch.long
    )
    matched = torch.zeros((b, pred_k), device=teacher_masks.device, dtype=torch.bool)
    owner_raw = torch.full_like(matched_target, -1)
    valid_slot_count = b * pred_k
    inside_count = 0
    duplicate_extra = 0
    covered_teacher = 0
    valid_teacher_total = int(teacher_valid.sum().item())

    anchor_detached = anchor_xy.detach()
    score_detached = anchor_score.detach()
    priority_detached = target_priority.detach() if isinstance(target_priority, torch.Tensor) else None
    masks_detached = teacher_masks.detach()

    for sample in range(b):
        teacher_to_candidates = {}
        for pred_index in range(pred_k):
            cx = float(anchor_detached[sample, pred_index, 0].item())
            cy = float(anchor_detached[sample, pred_index, 1].item())
            px = min(max(int(math.floor(cx * float(w))), 0), w - 1)
            py = min(max(int(math.floor(cy * float(h))), 0), h - 1)
            owners = []
            for target_index in range(target_k):
                if not bool(teacher_valid[sample, target_index].item()):
                    continue
                if float(masks_detached[sample, target_index, py, px].item()) >= 0.5:
                    priority = (
                        float(priority_detached[sample, target_index].item())
                        if priority_detached is not None else 0.0
                    )
                    owners.append((priority, -target_index, target_index))
            if not owners:
                continue
            owner = max(owners)[2]
            owner_raw[sample, pred_index] = owner
            inside_count += 1
            teacher_to_candidates.setdefault(owner, []).append(pred_index)

        for target_index, candidates in teacher_to_candidates.items():
            covered_teacher += 1
            if len(candidates) > 1:
                duplicate_extra += len(candidates) - 1
            best_pred = max(
                candidates,
                key=lambda pred_index: (
                    float(score_detached[sample, pred_index].item()),
                    -int(pred_index),
                ),
            )
            matched_target[sample, best_pred] = int(target_index)
            matched[sample, best_pred] = True

    zero = teacher_masks.new_zeros(())
    diagnostics = {
        "anchor_inside_teacher_rate": (
            teacher_masks.new_tensor(float(inside_count) / float(max(valid_slot_count, 1)))
        ),
        "teacher_seed_coverage": (
            teacher_masks.new_tensor(float(covered_teacher) / float(valid_teacher_total))
            if valid_teacher_total > 0 else zero
        ),
        "duplicate_owner_rate": (
            teacher_masks.new_tensor(float(duplicate_extra) / float(max(inside_count, 1)))
        ),
        "seed_without_teacher_rate": (
            teacher_masks.new_tensor(float(valid_slot_count - inside_count) / float(max(valid_slot_count, 1)))
        ),
        "unmatched_teacher_rate": (
            teacher_masks.new_tensor(float(valid_teacher_total - covered_teacher) / float(valid_teacher_total))
            if valid_teacher_total > 0 else zero
        ),
        "owner_raw": owner_raw,
    }
    return matched_target, matched, diagnostics


def _r4208_seed_consistent_partial_match(
    pair_score: torch.Tensor,
    target_valid: torch.Tensor,
    target_area: torch.Tensor,
    seed_owner: torch.Tensor,
    seed_score: torch.Tensor,
    target_priority: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Parameter-free partial ownership locking for R4.20.8 A4.

    A seed/Teacher relation is considered exact only when the selected seed
    pixel lies inside a valid Teacher component.  If several predicted slots
    seed the same Teacher, the highest R4.17 seed probability owns it; all
    other slots remain free.  Locked pairs are assigned first and the remaining
    slots/Teachers use exact maximum-score matching.  No spatial radius, cost
    weight, or confidence threshold is introduced.  This function is used only
    by the training loss; Native inference never sees Teacher masks.
    """
    b, pred_k, target_k = pair_score.shape
    matched_target = torch.full(
        (b, pred_k), -1, device=pair_score.device, dtype=torch.long
    )
    matched = torch.zeros((b, pred_k), device=pair_score.device, dtype=torch.bool)
    locked = torch.zeros((b, pred_k), device=pair_score.device, dtype=torch.bool)
    detached = pair_score.detach()
    owner_detached = seed_owner.detach()
    score_detached = seed_score.detach()

    for sample in range(b):
        # One deterministic owner slot per Teacher: maximum seed probability,
        # then lowest slot index for an exact tie.
        teacher_to_slot = {}
        for pred_index in range(pred_k):
            target_index = int(owner_detached[sample, pred_index].item())
            if target_index < 0 or target_index >= target_k:
                continue
            if not bool(target_valid[sample, target_index].item()):
                continue
            value = float(score_detached[sample, pred_index].item())
            previous = teacher_to_slot.get(target_index)
            if previous is None or value > previous[0] + 1.0e-12 or (
                abs(value - previous[0]) <= 1.0e-12 and pred_index < previous[1]
            ):
                teacher_to_slot[target_index] = (value, pred_index)

        used_pred = set()
        used_target = set()
        for target_index in sorted(teacher_to_slot):
            _, pred_index = teacher_to_slot[target_index]
            if pred_index in used_pred:
                continue
            matched_target[sample, pred_index] = target_index
            matched[sample, pred_index] = True
            locked[sample, pred_index] = True
            used_pred.add(pred_index)
            used_target.add(target_index)

        remaining_pred = [i for i in range(pred_k) if i not in used_pred]
        remaining_target = [
            j for j in range(target_k)
            if bool(target_valid[sample, j].item()) and j not in used_target
        ]
        remaining_target.sort(
            key=lambda j: (
                float(target_priority[sample, j].item()) if target_priority is not None else 0.0,
                float(target_area[sample, j].item()),
                -int(j),
            ),
            reverse=True,
        )
        remaining_target = remaining_target[:len(remaining_pred)]
        if not remaining_target:
            continue

        # Exact DP on the remaining slot indices.
        states = {0: (0.0, tuple())}
        for target_index in remaining_target:
            next_states = {}
            for used_mask, (total, assignment) in states.items():
                for local_pred, pred_index in enumerate(remaining_pred):
                    if used_mask & (1 << local_pred):
                        continue
                    new_mask = used_mask | (1 << local_pred)
                    new_total = total + float(
                        detached[sample, pred_index, target_index].item()
                    )
                    new_assignment = assignment + (pred_index,)
                    previous = next_states.get(new_mask)
                    if (
                        previous is None
                        or new_total > previous[0] + 1.0e-12
                        or (
                            abs(new_total - previous[0]) <= 1.0e-12
                            and new_assignment < previous[1]
                        )
                    ):
                        next_states[new_mask] = (new_total, new_assignment)
            states = next_states
        _, (_, best_assignment) = max(
            states.items(),
            key=lambda item: (item[1][0], tuple(-x for x in item[1][1])),
        )
        for target_index, pred_index in zip(remaining_target, best_assignment):
            matched_target[sample, pred_index] = int(target_index)
            matched[sample, pred_index] = True

    return matched_target, matched, locked


def _r414_center_type_extent_match(
    *,
    proposal_anchor: torch.Tensor,
    proposal_type: torch.Tensor,
    proposal_valid: torch.Tensor,
    teacher_geometry: torch.Tensor,
    teacher_actions: torch.Tensor,
    teacher_valid: torch.Tensor,
    teacher_area: torch.Tensor,
    teacher_utility: torch.Tensor,
    type_weight: float,
    require_type_match: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unique extent assignment using center + type only.

    V552-R4.14 intentionally excludes predicted width/height and masks from
    the target assignment.  This removes the circular supervision path
    predicted-size -> matching -> size target -> predicted-size.
    """
    center_l1 = (
        proposal_anchor[:, :, None, :2] - teacher_geometry[:, None, :, :2]
    ).abs().sum(dim=3)
    type_mismatch = (
        proposal_type[:, :, None].long() != teacher_actions[:, None, :].long()
    )
    score = -center_l1 - max(float(type_weight), 0.0) * type_mismatch.to(center_l1.dtype)
    valid_slot = proposal_valid.bool()[:, :, None]
    score = score.masked_fill(~valid_slot, -1.0e4)
    if require_type_match:
        score = score.masked_fill(type_mismatch, -1.0e4)
    matched_target, matched = _optimal_unique_match(
        score, teacher_valid, teacher_area, target_priority=teacher_utility
    )
    safe = matched_target.clamp_min(0)
    matched_teacher_type = teacher_actions.gather(1, safe)
    matched = matched & proposal_valid.bool()
    if require_type_match:
        matched = matched & proposal_type.long().eq(matched_teacher_type.long())
    return matched_target, matched


def _r415_identity_preserving_extent_match(
    *,
    proposal_anchor: torch.Tensor,
    proposal_type: torch.Tensor,
    proposal_valid: torch.Tensor,
    teacher_geometry: torch.Tensor,
    teacher_actions: torch.Tensor,
    teacher_valid: torch.Tensor,
    image_height: int,
    image_width: int,
    min_center_gate_px: float,
    max_center_gate_px: float,
    center_gate_diag_ratio: float,
    type_mismatch_penalty: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Identity-preserving extent assignment with explicit rejection.

    R4.15 fixes the R4.14 failure mode where every valid Teacher component was
    forced onto some proposal slot even when the nearest proposal was tens of
    pixels away.  Geometry identity is determined only from center proximity
    and typed-action compatibility.  Width/height, masks, utility and candidate
    gains are intentionally absent from the assignment.

    Each Teacher may remain unmatched and each proposal may remain unused.
    A pair is eligible only inside a physically scaled center gate.  The DP
    maximizes total positive identity quality over one-to-one pairs, with a
    zero-score skip transition for every Teacher.
    """
    if proposal_anchor.ndim != 3 or proposal_anchor.shape[-1] != 4:
        raise ValueError("R4.15 proposal_anchor must be [B,K,4]")
    b, pred_k, _ = proposal_anchor.shape
    target_k = teacher_geometry.shape[1]
    h = float(max(int(image_height), 1))
    w = float(max(int(image_width), 1))

    dx = (proposal_anchor[:, :, None, 0] - teacher_geometry[:, None, :, 0]) * w
    dy = (proposal_anchor[:, :, None, 1] - teacher_geometry[:, None, :, 1]) * h
    center_distance_px = torch.sqrt(dx.square() + dy.square() + EPS)

    teacher_w_px = teacher_geometry[:, :, 2].clamp_min(0.0) * w
    teacher_h_px = teacher_geometry[:, :, 3].clamp_min(0.0) * h
    teacher_diag_px = torch.sqrt(teacher_w_px.square() + teacher_h_px.square() + EPS)
    min_gate = max(float(min_center_gate_px), 0.5)
    max_gate = max(float(max_center_gate_px), min_gate)
    ratio = max(float(center_gate_diag_ratio), 0.0)
    teacher_gate_px = (ratio * teacher_diag_px).clamp(min=min_gate, max=max_gate)
    pair_gate_px = teacher_gate_px[:, None, :].expand(b, pred_k, target_k)

    type_mismatch = proposal_type[:, :, None].long() != teacher_actions[:, None, :].long()
    slot_valid = proposal_valid.bool()[:, :, None]
    target_valid = teacher_valid.bool()[:, None, :]
    pair_valid = slot_valid & target_valid & (center_distance_px <= pair_gate_px)

    normalized_distance = center_distance_px / pair_gate_px.clamp_min(1.0e-6)
    quality = 1.0 - normalized_distance
    penalty = min(max(float(type_mismatch_penalty), 0.0), 0.95)
    quality = quality - penalty * type_mismatch.to(quality.dtype)
    # Zero is the score of leaving a Teacher unmatched.  Therefore only a
    # strictly positive eligible identity pair may beat rejection.
    pair_valid = pair_valid & (quality > 0.0)
    quality = quality.masked_fill(~pair_valid, -1.0e6)

    matched_target = torch.full(
        (b, pred_k), -1, device=proposal_anchor.device, dtype=torch.long
    )
    matched = torch.zeros((b, pred_k), device=proposal_anchor.device, dtype=torch.bool)
    detached_quality = quality.detach()
    detached_valid = pair_valid.detach()

    for sample in range(b):
        targets = [
            j for j in range(target_k)
            if bool(teacher_valid[sample, j].item())
        ]
        # state: used-slot mask -> (score, match_count, assignment per processed target)
        states = {0: (0.0, 0, tuple())}
        for target_index in targets:
            next_states = {}
            for used_mask, (score, count, assignment) in states.items():
                # Explicit rejection: this Teacher may remain unmatched.
                skipped = (score, count, assignment + (-1,))
                previous = next_states.get(used_mask)
                if previous is None or (skipped[0], skipped[1], tuple(-x for x in skipped[2])) > (
                    previous[0], previous[1], tuple(-x for x in previous[2])
                ):
                    next_states[used_mask] = skipped

                for pred_index in range(pred_k):
                    if used_mask & (1 << pred_index):
                        continue
                    if not bool(detached_valid[sample, pred_index, target_index].item()):
                        continue
                    pair_score = float(detached_quality[sample, pred_index, target_index].item())
                    if pair_score <= 0.0:
                        continue
                    new_mask = used_mask | (1 << pred_index)
                    candidate = (score + pair_score, count + 1, assignment + (pred_index,))
                    previous = next_states.get(new_mask)
                    if previous is None or (candidate[0], candidate[1], tuple(-x for x in candidate[2])) > (
                        previous[0], previous[1], tuple(-x for x in previous[2])
                    ):
                        next_states[new_mask] = candidate
            states = next_states

        if not targets or not states:
            continue
        _, (_, _, best_assignment) = max(
            states.items(),
            key=lambda item: (item[1][0], item[1][1], tuple(-x for x in item[1][2])),
        )
        for target_index, pred_index in zip(targets, best_assignment):
            if pred_index < 0:
                continue
            matched_target[sample, pred_index] = int(target_index)
            matched[sample, pred_index] = True

    return matched_target, matched, center_distance_px, pair_gate_px, type_mismatch


def _matching_mean_dice(
    pair_dice: torch.Tensor,
    matched_target: torch.Tensor,
    matched: torch.Tensor,
) -> torch.Tensor:
    if not bool(matched.any().item()):
        return pair_dice.new_zeros(())
    safe = matched_target.clamp_min(0)
    selected = pair_dice.gather(2, safe[:, :, None])[:, :, 0]
    return selected[matched].mean()


def _balanced_mean(values: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    terms = []
    if bool(positive.any().item()):
        terms.append(values[positive].mean())
    if bool(negative.any().item()):
        terms.append(values[negative].mean())
    neutral = ~(positive | negative)
    if bool(neutral.any().item()):
        terms.append(values[neutral].mean())
    return torch.stack(terms).mean() if terms else values.sum() * 0.0


def _v548_weighted_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    """Weighted BCE on one observed semantic class."""
    if not bool(mask.any().item()):
        return logits.sum() * 0.0
    element = F.binary_cross_entropy_with_logits(
        logits[mask], target[mask].to(logits.dtype), reduction="none"
    )
    weight = sample_weight[mask]
    return (element * weight).sum() / weight.sum().clamp_min(EPS)


def _v548_factorized_direction_loss(
    *,
    direction_logits: torch.Tensor,
    positive_slot: torch.Tensor,
    negative_slot: torch.Tensor,
    sample_weight: torch.Tensor,
    require_both_classes: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Return a semantically balanced Benefit/Harm direction objective.

    When class-complete mode is enabled, Benefit-only and Harm-only mini-batches
    are intentionally not allowed to move the global Direction boundary.  This
    is the V545 contract restored for the V547 factorized head.
    """
    has_benefit = bool(positive_slot.any().item())
    has_harm = bool(negative_slot.any().item())
    active = (has_benefit and has_harm) if require_both_classes else (
        has_benefit or has_harm
    )
    target = positive_slot.to(direction_logits.dtype)
    benefit_loss = _v548_weighted_bce(
        direction_logits, target, positive_slot, sample_weight
    )
    harm_loss = _v548_weighted_bce(
        direction_logits, target, negative_slot, sample_weight
    )
    if not active:
        return direction_logits.sum() * 0.0, benefit_loss, harm_loss, False
    terms = []
    if has_benefit:
        terms.append(benefit_loss)
    if has_harm:
        terms.append(harm_loss)
    return torch.stack(terms).mean(), benefit_loss, harm_loss, True


def _v548_balanced_gain_sign_loss(
    *,
    gain_normalized: torch.Tensor,
    positive_slot: torch.Tensor,
    negative_slot: torch.Tensor,
    normalized_margin: float,
    require_both_classes: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Calibrate the exact signed Gain used by deployment."""
    has_benefit = bool(positive_slot.any().item())
    has_harm = bool(negative_slot.any().item())
    active = (has_benefit and has_harm) if require_both_classes else (
        has_benefit or has_harm
    )
    benefit_loss = (
        F.relu(float(normalized_margin) - gain_normalized[positive_slot]).mean()
        if has_benefit else gain_normalized.sum() * 0.0
    )
    harm_loss = (
        F.relu(float(normalized_margin) + gain_normalized[negative_slot]).mean()
        if has_harm else gain_normalized.sum() * 0.0
    )
    if not active:
        return gain_normalized.sum() * 0.0, benefit_loss, harm_loss, False
    terms = []
    if has_benefit:
        terms.append(benefit_loss)
    if has_harm:
        terms.append(harm_loss)
    combined = (
        torch.stack(terms).mean()
        if require_both_classes
        else torch.stack(terms).sum()
    )
    return combined, benefit_loss, harm_loss, True



def _v549_take_balanced_features(
    current: torch.Tensor,
    queued: torch.Tensor,
    count: int,
) -> torch.Tensor:
    """Prefer current features (trunk gradients), then fill from FIFO memory."""
    count = max(int(count), 0)
    if count == 0:
        return current[:0]
    current_take = min(int(current.shape[0]), count)
    pieces = [current[:current_take]] if current_take > 0 else []
    remaining = count - current_take
    if remaining > 0 and int(queued.shape[0]) > 0:
        pieces.append(queued[-remaining:])
    if not pieces:
        return current[:0]
    return torch.cat(pieces, dim=0)


def _v549_cross_batch_balanced_losses(
    *,
    cfg: Any,
    epoch: int,
    selector_features: torch.Tensor,
    positive_slot: torch.Tensor,
    negative_slot: torch.Tensor,
    direction_weight: torch.Tensor,
    direction_bias: torch.Tensor,
    editability_weight: torch.Tensor,
    editability_bias: torch.Tensor,
    gain_weight: torch.Tensor,
    gain_bias: torch.Tensor,
    normalized_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, Dict[str, torch.Tensor]]:
    """Balanced cross-batch Direction and exact signed-Gain supervision.

    Current-class features remain connected to the selector trunk.  Missing
    counterpart classes are supplied by detached, within-epoch FIFO features
    and re-evaluated with the *current* factorized head parameters.  Therefore
    no single-class batch can move the boundary alone, while its useful samples
    are no longer discarded.
    """
    run_tag = str(_m1(cfg, "RUN_TAG", "V549_DEFAULT"))
    max_queue = max(int(_m1(cfg, "V549_QUEUE_SIZE_PER_CLASS", 256)), 1)
    max_per_class = max(int(_m1(cfg, "V549_QUEUE_MAX_SAMPLES_PER_CLASS", 32)), 1)
    state = _V549_BALANCED_QUEUE_STATE.get(run_tag)
    if state is None or int(state.get("epoch", -1)) != int(epoch):
        state = {
            "epoch": int(epoch),
            "benefit": torch.empty((0, selector_features.shape[-1]), dtype=torch.float32),
            "harm": torch.empty((0, selector_features.shape[-1]), dtype=torch.float32),
        }
        _V549_BALANCED_QUEUE_STATE[run_tag] = state

    current_benefit = selector_features[positive_slot]
    current_harm = selector_features[negative_slot]
    queue_benefit = state["benefit"].to(
        device=selector_features.device, dtype=selector_features.dtype
    )
    queue_harm = state["harm"].to(
        device=selector_features.device, dtype=selector_features.dtype
    )
    available_benefit = int(current_benefit.shape[0] + queue_benefit.shape[0])
    available_harm = int(current_harm.shape[0] + queue_harm.shape[0])
    balanced_count = min(available_benefit, available_harm, max_per_class)
    active = balanced_count > 0

    zero = selector_features.sum() * 0.0
    benefit_direction_loss = zero
    harm_direction_loss = zero
    benefit_sign_loss = zero
    harm_sign_loss = zero
    direction_loss = zero
    sign_loss = zero

    if active:
        benefit_feature = _v549_take_balanced_features(
            current_benefit, queue_benefit, balanced_count
        )
        harm_feature = _v549_take_balanced_features(
            current_harm, queue_harm, balanced_count
        )

        benefit_direction_logit = F.linear(
            benefit_feature, direction_weight, direction_bias
        ).squeeze(-1)
        harm_direction_logit = F.linear(
            harm_feature, direction_weight, direction_bias
        ).squeeze(-1)
        benefit_direction_loss = F.binary_cross_entropy_with_logits(
            benefit_direction_logit, torch.ones_like(benefit_direction_logit)
        )
        harm_direction_loss = F.binary_cross_entropy_with_logits(
            harm_direction_logit, torch.zeros_like(harm_direction_logit)
        )
        direction_loss = 0.5 * (
            benefit_direction_loss + harm_direction_loss
        )

        def normalized_gain(feature: torch.Tensor) -> torch.Tensor:
            editability = torch.sigmoid(
                F.linear(feature, editability_weight, editability_bias).squeeze(-1)
            )
            direction = torch.sigmoid(
                F.linear(feature, direction_weight, direction_bias).squeeze(-1)
            )
            magnitude = F.softplus(
                F.linear(feature, gain_weight, gain_bias).squeeze(-1)
            )
            return editability * (2.0 * direction - 1.0) * magnitude

        benefit_gain = normalized_gain(benefit_feature)
        harm_gain = normalized_gain(harm_feature)
        benefit_sign_loss = F.relu(
            float(normalized_margin) - benefit_gain
        ).mean()
        harm_sign_loss = F.relu(
            float(normalized_margin) + harm_gain
        ).mean()
        sign_loss = 0.5 * (benefit_sign_loss + harm_sign_loss)

    # Queue update happens after constructing the differentiable losses.  Only
    # training selector features are inserted; labels come from exact candidate
    # Dice gain and are never inferred from model predictions.
    with torch.no_grad():
        for name, current in (("benefit", current_benefit), ("harm", current_harm)):
            if int(current.shape[0]) == 0:
                continue
            merged = torch.cat(
                [state[name], current.detach().to("cpu", torch.float32)], dim=0
            )
            state[name] = merged[-max_queue:].contiguous()

    diagnostics = {
        "v549_queue_active": selector_features.new_tensor(1.0 if active else 0.0),
        "v549_queue_balanced_count": selector_features.new_tensor(float(balanced_count)),
        "v549_queue_benefit_count": selector_features.new_tensor(float(state["benefit"].shape[0])),
        "v549_queue_harm_count": selector_features.new_tensor(float(state["harm"].shape[0])),
        "v549_current_benefit_count": selector_features.new_tensor(float(current_benefit.shape[0])),
        "v549_current_harm_count": selector_features.new_tensor(float(current_harm.shape[0])),
    }
    return (
        direction_loss,
        benefit_direction_loss,
        harm_direction_loss,
        sign_loss,
        benefit_sign_loss,
        harm_sign_loss,
        active,
        diagnostics,
    )


def _v550_current_semantic_balanced_losses(
    *,
    cfg: Any,
    epoch: int,
    direction_logits: torch.Tensor,
    gain_normalized: torch.Tensor,
    positive_slot: torch.Tensor,
    negative_slot: torch.Tensor,
    sample_weight: torch.Tensor,
    normalized_margin: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    bool,
    Dict[str, torch.Tensor],
]:
    """Current-semantic streaming balance for Direction and signed Gain.

    V549 balanced detached historical features at the head while the selector
    trunk still received the imbalanced current-batch gradient.  It also paired
    current heads with stale representations and stale labels.  V550 stores
    only cumulative class counts.  Every loss below is evaluated on current
    features and current exact hard candidate gain, so trunk and heads receive
    the same class-balanced, semantically current objective.

    Once both classes have been observed in the current epoch, inverse-frequency
    weights implement a 50/50 class prior over the stream.  Weights are bounded
    and renormalized to unit expected sample weight to avoid numerical spikes.
    """
    run_tag = str(_m1(cfg, "RUN_TAG", "V550_DEFAULT"))
    state = _V550_STREAMING_BALANCE_STATE.get(run_tag)
    if state is None or int(state.get("epoch", -1)) != int(epoch):
        state = {"epoch": int(epoch), "benefit": 0.0, "harm": 0.0}
        _V550_STREAMING_BALANCE_STATE[run_tag] = state

    current_benefit = float(positive_slot.detach().sum().cpu())
    current_harm = float(negative_slot.detach().sum().cpu())
    prior_benefit = float(state["benefit"])
    prior_harm = float(state["harm"])
    state["benefit"] += current_benefit
    state["harm"] += current_harm
    cumulative_benefit = float(state["benefit"])
    cumulative_harm = float(state["harm"])
    # Do not let the first observed class, or the first batch containing the
    # missing counterpart, move the boundary alone.  Once both classes have
    # existed in earlier current-semantic batches, all later current samples
    # contribute with inverse-frequency stream weights.
    active = prior_benefit > 0.0 and prior_harm > 0.0

    zero = direction_logits.sum() * 0.0
    benefit_direction_loss = zero
    harm_direction_loss = zero
    benefit_sign_loss = zero
    harm_sign_loss = zero
    direction_loss = zero
    sign_loss = zero

    min_weight = max(float(_m1(cfg, "V550_MIN_CLASS_WEIGHT", 0.25)), 1.0e-3)
    max_weight = max(float(_m1(cfg, "V550_MAX_CLASS_WEIGHT", 8.0)), min_weight)
    benefit_weight = 0.0
    harm_weight = 0.0

    if active:
        total = cumulative_benefit + cumulative_harm
        p_benefit = cumulative_benefit / max(total, 1.0)
        p_harm = cumulative_harm / max(total, 1.0)
        benefit_weight = min(max(total / (2.0 * cumulative_benefit), min_weight), max_weight)
        harm_weight = min(max(total / (2.0 * cumulative_harm), min_weight), max_weight)
        expected = p_benefit * benefit_weight + p_harm * harm_weight
        benefit_weight /= max(expected, EPS)
        harm_weight /= max(expected, EPS)

        def _current_class_terms(mask: torch.Tensor, target_value: float):
            if not bool(mask.any().item()):
                return zero, zero
            weights = sample_weight[mask]
            weights = weights / weights.mean().detach().clamp_min(EPS)
            direction_element = F.binary_cross_entropy_with_logits(
                direction_logits[mask],
                torch.full_like(direction_logits[mask], target_value),
                reduction="none",
            )
            signed_gain = gain_normalized[mask]
            if target_value > 0.5:
                sign_element = F.relu(float(normalized_margin) - signed_gain)
            else:
                sign_element = F.relu(float(normalized_margin) + signed_gain)
            return (
                (direction_element * weights).sum(),
                (sign_element * weights).sum(),
            )

        benefit_direction_sum, benefit_sign_sum = _current_class_terms(
            positive_slot, 1.0
        )
        harm_direction_sum, harm_sign_sum = _current_class_terms(
            negative_slot, 0.0
        )
        current_total = max(current_benefit + current_harm, 1.0)
        benefit_direction_loss = benefit_direction_sum / max(current_benefit, 1.0)
        harm_direction_loss = harm_direction_sum / max(current_harm, 1.0)
        benefit_sign_loss = benefit_sign_sum / max(current_benefit, 1.0)
        harm_sign_loss = harm_sign_sum / max(current_harm, 1.0)
        direction_loss = (
            float(benefit_weight) * benefit_direction_sum
            + float(harm_weight) * harm_direction_sum
        ) / current_total
        sign_loss = (
            float(benefit_weight) * benefit_sign_sum
            + float(harm_weight) * harm_sign_sum
        ) / current_total

    diagnostics = {
        "v550_current_semantic_balance_enabled": direction_logits.new_tensor(1.0),
        "v550_streaming_balance_active": direction_logits.new_tensor(1.0 if active else 0.0),
        "v550_cumulative_benefit_count": direction_logits.new_tensor(cumulative_benefit),
        "v550_cumulative_harm_count": direction_logits.new_tensor(cumulative_harm),
        "v550_current_benefit_count": direction_logits.new_tensor(current_benefit),
        "v550_current_harm_count": direction_logits.new_tensor(current_harm),
        "v550_benefit_class_weight": direction_logits.new_tensor(benefit_weight),
        "v550_harm_class_weight": direction_logits.new_tensor(harm_weight),
        "v550_current_benefit_gradient_mass": direction_logits.new_tensor(
            benefit_weight * current_benefit
        ),
        "v550_current_harm_gradient_mass": direction_logits.new_tensor(
            harm_weight * current_harm
        ),
    }
    return (
        direction_loss,
        benefit_direction_loss,
        harm_direction_loss,
        sign_loss,
        benefit_sign_loss,
        harm_sign_loss,
        active,
        diagnostics,
    )


def _v550_deployed_gain_order_losses(
    *,
    gain_normalized: torch.Tensor,
    pair_target: torch.Tensor,
    target_index: torch.Tensor,
    case_weight: torch.Tensor,
    slot_valid: torch.Tensor,
    pair_margin: float,
    listwise_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Supervise the exact utility score used by the V550 composer.

    Returns pairwise loss, listwise Preserve-vs-slot loss, pair accuracy and
    top-choice accuracy.  No auxiliary Rank head participates.
    """
    gain_difference = gain_normalized[:, :, None] - gain_normalized[:, None, :]
    if bool(pair_target.any().item()):
        pairwise_loss = F.softplus(
            float(pair_margin) - gain_difference[pair_target]
        ).mean()
        pair_accuracy = (gain_difference[pair_target] > 0.0).float().mean()
    else:
        pairwise_loss = gain_normalized.sum() * 0.0
        pair_accuracy = gain_normalized.new_zeros(())

    b = gain_normalized.shape[0]
    all_scores = torch.cat(
        [
            gain_normalized.new_zeros((b, 1)),
            gain_normalized.masked_fill(~slot_valid, -1.0e4),
        ],
        dim=1,
    )
    listwise_case = F.cross_entropy(
        all_scores / max(float(listwise_temperature), 1.0e-4),
        target_index,
        reduction="none",
    )
    listwise_loss = (
        listwise_case * case_weight
    ).sum() / case_weight.sum().clamp_min(1.0)
    top_choice_accuracy = (
        all_scores.argmax(dim=1) == target_index
    ).float().mean()
    return pairwise_loss, listwise_loss, pair_accuracy, top_choice_accuracy



def _r412_cxcywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    center = box[..., :2]
    half = 0.5 * box[..., 2:].clamp_min(1.0e-6)
    return torch.cat([center - half, center + half], dim=-1)


def _r412_pairwise_box_giou(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pairwise IoU/GIoU for normalized cx,cy,w,h boxes [B,K,4]/[B,J,4]."""
    p = _r412_cxcywh_to_xyxy(pred)[:, :, None, :]
    t = _r412_cxcywh_to_xyxy(target)[:, None, :, :]
    inter_lt = torch.maximum(p[..., :2], t[..., :2])
    inter_rb = torch.minimum(p[..., 2:], t[..., 2:])
    inter_wh = (inter_rb - inter_lt).clamp_min(0.0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    p_wh = (p[..., 2:] - p[..., :2]).clamp_min(0.0)
    t_wh = (t[..., 2:] - t[..., :2]).clamp_min(0.0)
    p_area = p_wh[..., 0] * p_wh[..., 1]
    t_area = t_wh[..., 0] * t_wh[..., 1]
    union = p_area + t_area - inter
    iou = inter / union.clamp_min(EPS)
    enc_lt = torch.minimum(p[..., :2], t[..., :2])
    enc_rb = torch.maximum(p[..., 2:], t[..., 2:])
    enc_wh = (enc_rb - enc_lt).clamp_min(0.0)
    enc_area = enc_wh[..., 0] * enc_wh[..., 1]
    giou = iou - (enc_area - union) / enc_area.clamp_min(EPS)
    return iou, giou


def _r412_aligned_box_iou_giou(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pair_iou, pair_giou = _r412_pairwise_box_giou(pred, target)
    if pred.shape[1] != target.shape[1]:
        raise ValueError("R4.12 aligned box metric requires equal slot counts")
    idx = torch.arange(pred.shape[1], device=pred.device)
    return pair_iou[:, idx, idx], pair_giou[:, idx, idx]


def _r412_canonical_crop(
    value: torch.Tensor,
    box: torch.Tensor,
    size: int,
    *,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Crop [B,K,H,W] tensors into normalized per-box canonical coordinates."""
    b, k, h, w = value.shape
    r = max(int(size), 8)
    axis = torch.linspace(-1.0, 1.0, r, device=value.device, dtype=value.dtype)
    gy, gx = torch.meshgrid(axis, axis, indexing="ij")
    local = torch.stack([gx, gy], dim=-1)[None, None].expand(b, k, -1, -1, -1)
    center = box[:, :, None, None, :2]
    extent = box[:, :, None, None, 2:].clamp_min(1.0e-5)
    xy = center + 0.5 * extent * local
    grid = (2.0 * xy - 1.0).reshape(b * k, r, r, 2)
    source = value.reshape(b * k, 1, h, w)
    out = F.grid_sample(
        source,
        grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=True,
    )
    return out.reshape(b, k, r, r)


def _r412_balanced_mask_map(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    elem = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pos = target.sum(dim=(-2, -1))
    total = float(target.shape[-2] * target.shape[-1])
    neg = total - pos
    pos_loss = (elem * target).sum(dim=(-2, -1)) / pos.clamp_min(1.0)
    neg_loss = (elem * (1.0 - target)).sum(dim=(-2, -1)) / neg.clamp_min(1.0)
    bce = 0.5 * (pos_loss + neg_loss)
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(-2, -1))
    dice = (2.0 * inter + EPS) / (
        prob.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1)) + EPS
    )
    purity = (inter + EPS) / (prob.sum(dim=(-2, -1)) + EPS)
    return bce + (1.0 - dice), dice, purity, inter


def compute_v538_online_component_loss(
    *,
    cfg: Any,
    masks: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    epoch: int,
    base_probability: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Return M1 component loss, V541 M2 risk loss, and diagnostics.

    V541 deliberately separates raw gain regression from dimensionless ranking,
    trains Benefit/Harm/Preserve on every valid candidate batch, and computes
    labels from the exact hard single-slot candidate used by deployment.
    """
    gt = (_as_b1hw(masks) >= 0.5).float()
    r46_enabled = bool(_m1(cfg, "V552R46_ROOTFIX_ENABLED", False))
    r47_enabled = bool(_m1(cfg, "V552R47_ROOTFIX_ENABLED", False))
    r48_enabled = bool(_m1(cfg, "V552R48_ROOTFIX_ENABLED", False))
    r49_enabled = bool(_m1(cfg, "V552R49_ROOTFIX_ENABLED", False))
    r410_enabled = bool(_m1(cfg, "V552R410_ROOTFIX_ENABLED", False))
    r411_enabled = bool(_m1(cfg, "V552R411_ROOTFIX_ENABLED", False))
    r412_enabled = bool(_m1(cfg, "V552R412_ROOTFIX_ENABLED", False))
    r413_enabled = bool(_m1(cfg, "V552R413_ROOTFIX_ENABLED", False))
    r414_enabled = bool(_m1(cfg, "V552R414_ROOTFIX_ENABLED", False))
    r415_enabled = bool(_m1(cfg, "V552R415_ROOTFIX_ENABLED", False))
    r416_enabled = bool(_m1(cfg, "V552R416_ROOTFIX_ENABLED", False))
    r416_ltrb_enabled = r416_enabled and bool(
        _m1(cfg, "V552R416_ASYMMETRIC_LTRB_ENABLED", False)
    )
    r417_enabled = bool(_m1(cfg, "V552R417_ROOTFIX_ENABLED", False))
    r417_shared_offset_enabled = r417_enabled and bool(
        _m1(cfg, "V552R417_SHARED_OFFSET_ENABLED", True)
    )
    r418_enabled = bool(_m1(cfg, "V552R418_ROOTFIX_ENABLED", False))
    r418_paired_enabled = r418_enabled and bool(
        _m1(cfg, "V552R418_PAIRED_STABLE_TEACHER_ENABLED", True)
    )
    r419_enabled = r418_enabled and bool(
        _m1(cfg, "V552R419_ROOTFIX_ENABLED", False)
    )
    r420_enabled = r418_enabled and bool(
        _m1(cfg, "V552R420_ROOTFIX_ENABLED", False)
    )
    # R4.20.1 is a clean ownership/training contract shared by the R4.20
    # point-conditioned control and the R4.20.3 dense-set replacement.
    r4201_enabled = bool(_m1(cfg, "V552R4201_ROOTFIX_ENABLED", False))
    r4203_enabled = r418_enabled and bool(
        _m1(cfg, "V552R4203_ROOTFIX_ENABLED", False)
    )
    r4204_enabled = r4203_enabled and bool(
        _m1(cfg, "V552R4204_ROOTFIX_ENABLED", False)
    )
    r4205_enabled = r4204_enabled and bool(
        _m1(cfg, "V552R4205_ROOTFIX_ENABLED", False)
    )
    r4206_requested = bool(_m1(cfg, "V552R4206_ROOTFIX_ENABLED", False))
    r4206_enabled = r4205_enabled and r4206_requested
    r4207_requested = bool(_m1(cfg, "V552R4207_ROOTFIX_ENABLED", False))
    r4207_enabled = r4205_enabled and r4207_requested
    r4208_requested = bool(_m1(cfg, "V552R4208_ROOTFIX_ENABLED", False))
    r4208_normalized = r4208_requested and bool(
        _m1(cfg, "V552R4208_NORMALIZED_FUSION_ENABLED", False)
    )
    r4208_persistent = r4208_requested and bool(
        _m1(cfg, "V552R4208_PERSISTENT_IDENTITY_ENABLED", False)
    )
    r4208_seed_matching = r4208_requested and bool(
        _m1(cfg, "V552R4208_SEED_CONSISTENT_MATCHING_ENABLED", False)
    )
    r4209_requested = bool(_m1(cfg, "V552R4209_ROOTFIX_ENABLED", False))
    r4209_peak_focal = r4209_requested and bool(
        _m1(cfg, "V552R4209_CENTER_PEAK_FOCAL_ENABLED", True)
    )
    r4209_balanced_overflow = r4209_requested and bool(
        _m1(cfg, "V552R4209_BALANCED_OVERFLOW_ENABLED", True)
    )
    r4210_requested = bool(_m1(cfg, "V552R4210_ROOTFIX_ENABLED", False))
    r4210_interior_anchor = r4210_requested and bool(
        _m1(cfg, "V552R4210_INTERIOR_ANCHOR_ENABLED", False)
    )
    r4210_variable_seed = r4210_requested and bool(
        _m1(cfg, "V552R4210_VARIABLE_CARDINALITY_SEEDS_ENABLED", False)
    )
    r4210_independent_overflow = r4210_requested and bool(
        _m1(cfg, "V552R4210_INDEPENDENT_OVERFLOW_GATE_ENABLED", False)
    )
    r4210_m1_native_alignment = r4210_requested and bool(
        _m1(cfg, "V552R4210_M1_NATIVE_ALIGNMENT_ENABLED", False)
    )
    r4211_requested = bool(_m1(cfg, "V552R4211_ROOTFIX_ENABLED", False))
    r4211_proposal_existence = r4211_requested and bool(
        _m1(cfg, "V552R4211_PROPOSAL_EXISTENCE_DECOUPLING_ENABLED", False)
    )
    r4211_geometry_overflow = r4211_requested and bool(
        _m1(cfg, "V552R4211_GEOMETRY_OVERFLOW_DECOUPLING_ENABLED", False)
    )
    r4212_requested = bool(_m1(cfg, "V552R4212_ROOTFIX_ENABLED", False))
    r4212_independent_set = r4212_requested and bool(
        _m1(cfg, "V552R4212_INDEPENDENT_CANDIDATE_SET_ENABLED", False)
    )
    r4212_existence_no_object = r4212_requested and bool(
        _m1(cfg, "V552R4212_EXISTENCE_NO_OBJECT_ENABLED", False)
    )
    r4212_candidate_alignment = r4212_requested and bool(
        _m1(cfg, "V552R4212_CANDIDATE_ALIGNMENT_ENABLED", False)
    )
    # V560 Clean Core is a deliberately small scientific path.  It reuses the
    # existing Teacher/Hungarian machinery but removes historical probability
    # ownership from the physical candidate mask.
    m1_protocol = str(_m1(cfg, "PROTOCOL", "")).strip().lower()
    tc_drcs = m1_protocol == "tc_drcs"
    clean_dynamic_component_set = m1_protocol in {"clean_dynamic_component_set", "tc_drcs"}
    v560_clean_core = clean_dynamic_component_set or bool(_m1(cfg, "V560_CLEAN_CORE_ENABLED", False))
    v561_bcrs = clean_dynamic_component_set or bool(_m1(cfg, "V561_BCRS_ENABLED", False))
    v562_rootfix = clean_dynamic_component_set or (
        v561_bcrs and bool(_m1(cfg, "V562_ROOTFIX_ENABLED", False))
    )
    # CLEAN deliberately stops before the historical fixed-window V563/V564/V565
    # geometry contracts.  Strict factual ownership is enabled explicitly below.
    v563_rootfix = (not clean_dynamic_component_set) and v562_rootfix and bool(_m1(cfg, "V563_ROOTFIX_ENABLED", False))
    v564_rootfix = (not clean_dynamic_component_set) and v563_rootfix and bool(_m1(cfg, "V564_ROOTFIX_ENABLED", False))
    v565_rootfix = (not clean_dynamic_component_set) and v564_rootfix and bool(_m1(cfg, "V565_ROOTFIX_ENABLED", False))
    if v560_clean_core and not v561_bcrs:
        # Historical V560R2 kept whole-image candidate alignment.  V561 removes
        # it from geometry ownership because sparse corrections can minimize the
        # full-image objective by simply copying Base.
        r4212_candidate_alignment = True
    if v561_bcrs:
        r4212_candidate_alignment = False
    if r4211_requested:
        if not r4210_requested or not r4210_interior_anchor:
            raise RuntimeError("V552-R4.21.1 requires R4.21.0 interior-anchor scaffold")
        if r4211_proposal_existence and r4210_variable_seed:
            raise RuntimeError("V552-R4.21.1 proposal/existence decoupling forbids R4.21.0 hard seed validity")
        if r4211_geometry_overflow and not r4210_independent_overflow:
            raise RuntimeError("V552-R4.21.1 geometry/overflow decoupling requires independent overflow")
    if r4210_requested:
        if not r4209_requested or not r4209_peak_focal:
            raise RuntimeError("V552-R4.21.0 requires the R4.20.9 CenterNet-style peak scaffold")
        if not r4208_requested or not r4208_persistent:
            raise RuntimeError("V552-R4.21.0 requires normalized persistent R4.20.8 binding")
        if r4208_seed_matching:
            raise RuntimeError("V552-R4.21.0 forbids hard seed-consistent matching; optimal mask matching owns geometry")
        if r4209_balanced_overflow:
            raise RuntimeError("V552-R4.21.0 forbids the shared-simplex R4.20.9 balanced overflow objective")
        if r4210_independent_overflow and not r4205_enabled:
            raise RuntimeError("V552-R4.21.0 independent overflow requires the R4.20.5 capacity scaffold")
    if r4209_requested and not r4208_requested:
        raise RuntimeError("V552-R4.20.9 requires the R4.20.8 normalized visual binding contract")
    if r4209_requested and not r4208_persistent:
        raise RuntimeError("V552-R4.20.9 requires persistent instance identity")
    if r4209_requested and not r417_enabled:
        raise RuntimeError("V552-R4.20.9 requires R4.17 inference-visible location evidence")
    if r4209_requested and r4206_requested:
        raise RuntimeError("V552-R4.20.9 forbids R4.20.6 global K+1 CE; balanced overflow owns capacity supervision")
    if r4208_requested and not r4205_enabled:
        raise RuntimeError(
            "V552-R4.20.8 requires the R4.20.5 capacity-consistent factorization"
        )
    if r4208_requested and not r4208_normalized:
        raise RuntimeError(
            "V552-R4.20.8 rootfix requires V552R4208_NORMALIZED_FUSION_ENABLED=true"
        )
    if r4208_persistent and not r4208_normalized:
        raise RuntimeError("R4.20.8 persistent identity requires normalized fusion")
    if r4208_seed_matching and not r4208_persistent:
        raise RuntimeError(
            "R4.20.8 seed-consistent matching is an A4 intervention and requires persistent identity"
        )
    if r4208_requested and r4207_requested:
        raise RuntimeError("R4.20.7 and R4.20.8 query interventions are mutually exclusive")
    if r4208_requested and r4206_requested:
        raise RuntimeError(
            "R4.20.8 causal ablation requires V552R4206_ROOTFIX_ENABLED=false"
        )
    if r4207_requested and not r4205_enabled:
        raise RuntimeError(
            "V552-R4.20.7 dynamic visual instance binding requires the "
            "R4.20.5 capacity-consistent factorized residual set"
        )
    if r4207_requested and r4206_requested:
        raise RuntimeError(
            "V552-R4.20.7 root-cause ablation requires V552R4206_ROOTFIX_ENABLED=false; "
            "do not confound query binding with the conditional-CE objective"
        )
    if r4206_requested and not r4205_enabled:
        raise RuntimeError(
            "V552-R4.20.6 conditional identity supervision requires the "
            "R4.20.5 capacity-consistent K+overflow probability model"
        )
    # Keep the R4.20.6 causal intervention single-purpose.  Canonical/paired
    # alternative mask objectives would re-introduce independent pixel BCE
    # terms and confound the requested CE+Dice factorization.
    if r4206_requested and r412_enabled:
        raise RuntimeError("V552-R4.20.6 requires V552R412_ROOTFIX_ENABLED=false")
    if r4206_requested and r418_paired_enabled:
        raise RuntimeError(
            "V552-R4.20.6 requires V552R418_PAIRED_STABLE_TEACHER_ENABLED=false"
        )
    if r418_enabled:
        # The failed R4.17 shared offset is not part of the box-free contract.
        r417_shared_offset_enabled = False
    paired_replay_enabled = bool(
        _m1(cfg, "V547_PAIRED_RESIDUAL_REPLAY_ENABLED", False)
    ) and (not r46_enabled)
    original_base = _as_b1hw(base_probability).detach().clamp(EPS, 1.0 - EPS)
    paired_anchor = aux.get("v547_refiner_anchor_prob")
    base = (
        _as_b1hw(paired_anchor).detach().clamp(EPS, 1.0 - EPS)
        if paired_replay_enabled and isinstance(paired_anchor, torch.Tensor)
        else original_base
    )
    # R4.6 single-native contract: teacher labels and student candidates are
    # both defined on the exact current Native Base state.  Historical EMA /
    # synthetic anchors are intentionally forbidden for the main contract.
    teacher_probability = (
        base
        if r46_enabled
        else _as_b1hw(aux.get("v538_teacher_base_prob", base))
    ).detach().clamp(EPS, 1.0 - EPS)

    slot_mask_logits = aux["v538_slot_mask_logits"]
    slot_masks = aux["v538_slot_masks"].clamp(EPS, 1.0 - EPS)
    action_logits = aux["v538_slot_action_logits"]
    polarity_logits = aux.get("v538_slot_polarity_logits")
    if polarity_logits is None:
        polarity_logits = torch.stack(
            [
                torch.logsumexp(action_logits[:, :, (0, 2)], dim=2),
                torch.logsumexp(action_logits[:, :, (1, 3)], dim=2),
            ],
            dim=2,
        )
    slot_doses = aux.get("v538_slot_doses")
    if slot_doses is None:
        slot_doses = action_logits.new_full(
            action_logits.shape[:2], float(_m1(cfg, "V540_INITIAL_DOSE", 1.0))
        )
    gain_scores = aux["v538_slot_gain_scores"]
    gain_normalized = aux.get(
        "v541_slot_gain_normalized",
        gain_scores * float(_m1(cfg, "V541_GAIN_SCALE", 1000.0)),
    )
    gain_magnitude_normalized = aux.get(
        "v543_slot_gain_magnitude_normalized",
        gain_normalized.abs(),
    )
    outcome_logits = aux.get("v543_slot_outcome_logits")
    gain_logvar = aux.get(
        "v541_slot_gain_logvar", torch.zeros_like(gain_scores)
    )
    rank_scores = aux.get("v541_slot_rank_scores", gain_scores)
    decision_scores = aux.get("v541_slot_decision_scores", rank_scores)
    benefit_logits = aux.get("v541_slot_benefit_logits", gain_scores)
    harm_logits = aux.get("v541_slot_harm_logits", -gain_scores)
    benefit_probs = aux.get(
        "v541_slot_benefit_probs", torch.sigmoid(benefit_logits)
    ).clamp(EPS, 1.0 - EPS)
    harm_probs = aux.get(
        "v541_slot_harm_probs", torch.sigmoid(harm_logits)
    ).clamp(EPS, 1.0 - EPS)
    editability_logits = aux.get(
        "v547_slot_editability_logits",
        torch.logsumexp(torch.stack([benefit_logits, harm_logits], dim=-1), dim=-1),
    )
    direction_logits = aux.get(
        "v547_slot_direction_logits", benefit_logits - harm_logits
    )
    editability_probs = aux.get(
        "v547_slot_editability_probs", torch.sigmoid(editability_logits)
    ).clamp(EPS, 1.0 - EPS)
    direction_probs = aux.get(
        "v547_slot_direction_probs", torch.sigmoid(direction_logits)
    ).clamp(EPS, 1.0 - EPS)
    factorized_outcome_enabled = bool(
        _m1(cfg, "V547_FACTORIZED_OUTCOME_ENABLED", False)
    )
    gain_lcb = aux.get("v541_slot_gain_lcb", gain_scores)
    presence_logits = aux["v538_slot_presence_logits"]
    presence_probs = aux["v538_slot_presence_probs"].clamp(EPS, 1.0 - EPS)
    slot_candidates = aux["v538_slot_candidate_probs"].clamp(EPS, 1.0 - EPS)
    v541_selector_enabled = bool(
        _m1(cfg, "V541_CANDIDATE_OUTCOME_SELECTOR_ENABLED", False)
    )
    exact_slot_candidates = (
        aux.get("v541_slot_exact_candidate_probs", slot_candidates.detach())
        if v541_selector_enabled else slot_candidates.detach()
    ).detach().clamp(EPS, 1.0 - EPS)
    exact_slot_candidates_st = aux.get(
        "v545_slot_exact_candidate_st_probs", slot_candidates
    ).clamp(EPS, 1.0 - EPS)
    hard_masks_st = aux.get("v545_slot_hard_masks_st", slot_masks)
    mask_contrast_active = aux.get(
        "v545_slot_mask_contrast_active", torch.ones_like(gain_scores)
    ) > 0.5
    slot_valid = aux["v538_slot_valid"].bool()
    r47_anchor_params = aux.get(
        "v552r47_anchor_params",
        slot_masks.new_zeros((*slot_masks.shape[:2], 4)),
    ).clamp(0.0, 1.0)
    deploy_slot_valid = aux.get("v552_deploy_slot_valid", slot_valid).bool()
    b, k, h, w = slot_masks.shape

    forward_teacher_signal = aux.get(
        "v552r48_teacher_built",
        aux.get("v552r4201_forward_teacher_built"),
    )
    forward_teacher_built = bool(
        isinstance(forward_teacher_signal, torch.Tensor)
        and float(forward_teacher_signal.detach().mean().item()) > 0.5
    )
    use_r48_forward_teacher = (
        r48_enabled
        and forward_teacher_built
        and isinstance(aux.get("v552r48_teacher_masks"), torch.Tensor)
        and aux["v552r48_teacher_masks"].shape == slot_masks.shape
        and isinstance(aux.get("v552r48_teacher_valid"), torch.Tensor)
    )
    if use_r48_forward_teacher:
        # Reuse only a forward teacher bank that was actually built.  R4.20.1
        # forbids treating shape-correct all-zero placeholders as supervision.
        teacher_masks = aux["v552r48_teacher_masks"].detach().to(slot_masks.dtype)
        heuristic_teacher_actions = aux["v552r48_teacher_actions"].detach().long()
        teacher_valid = aux["v552r48_teacher_valid"].detach().bool()
        teacher_area = aux["v552r48_teacher_area"].detach().to(slot_masks.dtype)
        teacher_raw_component_count = teacher_valid.to(slot_masks.dtype).sum(dim=1)
        replay_case = torch.zeros((b,), device=slot_masks.device, dtype=torch.bool)
        effective_teacher = _as_b1hw(aux.get("v552r48_teacher_effective", teacher_probability)).detach().to(slot_masks.dtype)
        effective_teacher_error = _as_b1hw(aux.get("v552r48_teacher_error", (effective_teacher >= 0.5) != (gt >= 0.5))).detach().to(slot_masks.dtype)
    else:
        (
            teacher_masks,
            heuristic_teacher_actions,
            teacher_valid,
            teacher_area,
            replay_case,
            effective_teacher,
            effective_teacher_error,
            teacher_raw_component_count,
        ) = _build_teacher_components(
            teacher_probability=teacher_probability,
            gt=gt,
            num_slots=k,
            min_pixels=int(_m1(cfg, "V538_TEACHER_MIN_PIXELS", 4)),
            current_epoch=int(epoch),
            replay_enabled=(
                bool(_m1(cfg, "V538_RESIDUAL_REPLAY_ENABLED", True))
                and not paired_replay_enabled
                and not v560_clean_core
            ),
            replay_error_floor=float(_m1(cfg, "V538_REPLAY_ERROR_FLOOR", 0.004)),
            replay_radius=int(_m1(cfg, "V538_REPLAY_RADIUS", 2)),
            utility_rank_enabled=(
                v560_clean_core
                or (
                    r411_enabled
                    and bool(_m1(cfg, "V552R411_UTILITY_AWARE_TEACHER_TOPK", True))
                )
            ),
        )
    if paired_replay_enabled and isinstance(aux.get("v547_replay_case"), torch.Tensor):
        replay_case = aux["v547_replay_case"].detach().bool()
    teacher_error = effective_teacher_error.to(slot_masks.dtype)
    v560_factual_residual_fraction = (
        ((teacher_probability >= 0.5) != (gt >= 0.5))
        .to(slot_masks.dtype).mean().detach()
        if v560_clean_core else slot_masks.new_zeros(())
    )

    # V562 residualness proposal supervision.  Standard balanced BCE + soft
    # Dice on the factual current-Base residual.  GT never enters the forward
    # proposal/anchor path; it is used only here as a training target.
    v562_residual_proposal_bce = slot_masks.sum() * 0.0
    v562_residual_proposal_dice_loss = slot_masks.sum() * 0.0
    v562_residual_proposal_loss = slot_masks.sum() * 0.0
    v562_residual_proposal_soft_dice = slot_masks.new_zeros(())
    v564_component_seed_bce = slot_masks.sum() * 0.0
    v565_seed_heatmap_bce = slot_masks.sum() * 0.0
    v565_seed_heatmap_dice_loss = slot_masks.sum() * 0.0
    v565_seed_heatmap_loss = slot_masks.sum() * 0.0
    v565_seed_rank_loss = slot_masks.sum() * 0.0
    v565_seed_target_fraction = slot_masks.new_zeros(())
    v565_seed_soft_dice = slot_masks.new_zeros(())
    v565_true_center_probability = slot_masks.new_zeros(())
    v565_false_peak_probability = slot_masks.new_zeros(())
    v565_center_rank_accuracy = slot_masks.new_zeros(())
    v562_residual_target_fraction = slot_masks.new_zeros(())
    v562_residual_probability_mean = aux.get(
        "v562_residual_probability_mean", slot_masks.new_zeros(())
    ).detach()
    if v562_rootfix:
        residual_logits_v562 = aux.get("v562_residual_logits")
        expected_shape_v562 = (b, 1, h, w)
        if not isinstance(residual_logits_v562, torch.Tensor) or residual_logits_v562.shape != expected_shape_v562:
            raise RuntimeError(
                "V562 requires attached v562_residual_logits with shape "
                f"{expected_shape_v562}, got "
                + str(None if not isinstance(residual_logits_v562, torch.Tensor) else tuple(residual_logits_v562.shape))
            )
        if clean_dynamic_component_set:
            # Clean-family residualness is a continuous current-error field.
            residual_target_v562 = (teacher_probability - gt).abs().detach().to(slot_masks.dtype)
            residual_bce_element_v562 = F.binary_cross_entropy_with_logits(
                residual_logits_v562, residual_target_v562, reduction="none"
            )
            if tc_drcs:
                # Soft region balancing prevents the sparse residual field from
                # winning BCE by predicting background everywhere.  Target mass
                # itself defines positive/negative contributions; no pos_weight
                # hyperparameter is introduced.
                pos_mass = residual_target_v562.flatten(1).sum(dim=1)
                neg_mass = (1.0 - residual_target_v562).flatten(1).sum(dim=1)
                pos_term = (
                    residual_bce_element_v562 * residual_target_v562
                ).flatten(1).sum(dim=1) / pos_mass.clamp_min(1.0)
                neg_term = (
                    residual_bce_element_v562 * (1.0 - residual_target_v562)
                ).flatten(1).sum(dim=1) / neg_mass.clamp_min(1.0)
                v562_residual_proposal_bce = torch.where(
                    pos_mass > 0, 0.5 * (pos_term + neg_term), neg_term
                ).mean()
            else:
                v562_residual_proposal_bce = residual_bce_element_v562.mean()
        else:
            residual_target_v562 = (
                (teacher_probability >= 0.5) != (gt >= 0.5)
            ).to(slot_masks.dtype)
            residual_bce_element_v562 = F.binary_cross_entropy_with_logits(
                residual_logits_v562, residual_target_v562, reduction="none"
            )
            pos_v562 = residual_target_v562.flatten(1).sum(dim=1)
            neg_v562 = (1.0 - residual_target_v562).flatten(1).sum(dim=1)
            pos_bce_v562 = (
                residual_bce_element_v562 * residual_target_v562
            ).flatten(1).sum(dim=1) / pos_v562.clamp_min(1.0)
            neg_bce_v562 = (
                residual_bce_element_v562 * (1.0 - residual_target_v562)
            ).flatten(1).sum(dim=1) / neg_v562.clamp_min(1.0)
            bce_case_v562 = torch.where(
                pos_v562 > 0, 0.5 * (pos_bce_v562 + neg_bce_v562), neg_bce_v562
            )
            v562_residual_proposal_bce = bce_case_v562.mean()
        residual_prob_v562 = torch.sigmoid(residual_logits_v562)
        inter_v562 = (residual_prob_v562 * residual_target_v562).flatten(1).sum(dim=1)
        den_v562 = (
            residual_prob_v562.flatten(1).sum(dim=1)
            + residual_target_v562.flatten(1).sum(dim=1)
        )
        dice_v562 = (2.0 * inter_v562 + EPS) / (den_v562 + EPS)
        v562_residual_proposal_dice_loss = (1.0 - dice_v562).mean()
        v562_residual_proposal_soft_dice = dice_v562.mean().detach()
        v562_residual_target_fraction = residual_target_v562.mean().detach()
        v562_residual_probability_mean = residual_prob_v562.mean().detach()
        v562_residual_proposal_loss = (
            v562_residual_proposal_bce + v562_residual_proposal_dice_loss
        )
        if v565_rootfix:
            # Independent center head: train exactly the heatmap consumed by
            # inference NMS.  Standard region-balanced BCE + soft Dice shape the
            # center map; standard MarginRankingLoss forces every factual center
            # above the strongest false center peak.
            seed_logits_v565 = aux.get("v565_seed_logits")
            if not isinstance(seed_logits_v565, torch.Tensor) or seed_logits_v565.shape != (b, 1, h, w):
                raise RuntimeError("V565 requires independent v565_seed_logits [B,1,H,W]")
            seed_target_v565, center_yx_v565 = _v565_build_center_heatmap(
                teacher_masks, teacher_valid,
                sigma_min_px=float(_m1(cfg, "V565_CENTER_SIGMA_MIN_PX", 1.5)),
                sigma_max_px=float(_m1(cfg, "V565_CENTER_SIGMA_MAX_PX", 4.0)),
            )
            seed_bce_pixel_v565 = F.binary_cross_entropy_with_logits(seed_logits_v565, seed_target_v565, reduction="none")
            positive_region_v565 = (seed_target_v565 >= 0.10).to(seed_bce_pixel_v565.dtype)
            negative_region_v565 = 1.0 - positive_region_v565
            pos_loss_v565 = (seed_bce_pixel_v565 * positive_region_v565).sum() / positive_region_v565.sum().clamp_min(1.0)
            neg_loss_v565 = (seed_bce_pixel_v565 * negative_region_v565).sum() / negative_region_v565.sum().clamp_min(1.0)
            v565_seed_heatmap_bce = 0.5 * (pos_loss_v565 + neg_loss_v565)
            seed_prob_v565 = torch.sigmoid(seed_logits_v565)
            seed_intersection_v565 = (seed_prob_v565 * seed_target_v565).sum(dim=(-2, -1))
            seed_denom_v565 = seed_prob_v565.sum(dim=(-2, -1)) + seed_target_v565.sum(dim=(-2, -1))
            seed_dice_v565 = (2.0 * seed_intersection_v565 + EPS) / (seed_denom_v565 + EPS)
            v565_seed_heatmap_dice_loss = 1.0 - seed_dice_v565.mean()
            v565_seed_soft_dice = seed_dice_v565.mean().detach()
            v565_seed_heatmap_loss = v565_seed_heatmap_bce + v565_seed_heatmap_dice_loss
            v565_seed_target_fraction = (seed_target_v565 >= 0.10).to(seed_prob_v565.dtype).mean().detach()

            positive_logits_v565, negative_logits_v565, rank_correct_v565 = [], [], []
            false_region_v565 = seed_target_v565[:, 0] < 0.05
            for bi in range(b):
                hard_neg = (
                    seed_logits_v565[bi, 0].masked_fill(~false_region_v565[bi], -1.0e4).max()
                    if bool(false_region_v565[bi].any().item())
                    else seed_logits_v565[bi, 0].min().detach()
                )
                for ti in range(teacher_masks.shape[1]):
                    cy, cx = (int(center_yx_v565[bi, ti, 0].item()), int(center_yx_v565[bi, ti, 1].item()))
                    if cy < 0 or cx < 0:
                        continue
                    pos = seed_logits_v565[bi, 0, cy, cx]
                    positive_logits_v565.append(pos)
                    negative_logits_v565.append(hard_neg)
                    rank_correct_v565.append((pos.detach() > hard_neg.detach()).to(seed_prob_v565.dtype))
            if positive_logits_v565:
                pos_stack_v565 = torch.stack(positive_logits_v565)
                neg_stack_v565 = torch.stack(negative_logits_v565)
                v565_seed_rank_loss = F.margin_ranking_loss(
                    pos_stack_v565, neg_stack_v565, torch.ones_like(pos_stack_v565),
                    margin=float(_m1(cfg, "V565_SEED_RANK_MARGIN", 0.50)), reduction="mean"
                )
                v565_true_center_probability = torch.sigmoid(pos_stack_v565).mean().detach()
                v565_false_peak_probability = torch.sigmoid(neg_stack_v565).mean().detach()
                v565_center_rank_accuracy = torch.stack(rank_correct_v565).mean().detach()
            v562_residual_proposal_loss = (
                v562_residual_proposal_loss
                + float(_m1(cfg, "V565_SEED_HEATMAP_WEIGHT", 1.0)) * v565_seed_heatmap_loss
                + float(_m1(cfg, "V565_SEED_RANK_WEIGHT", 0.25)) * v565_seed_rank_loss
            )
        elif v564_rootfix:
            # Historical V564 compatibility path.
            proposal_logits_expanded_v564 = residual_logits_v562[:, :1].expand(-1, teacher_masks.shape[1], -1, -1)
            valid_component_mask_v564 = teacher_masks > 0.5
            masked_logits_v564 = proposal_logits_expanded_v564.masked_fill(~valid_component_mask_v564, -1.0e4)
            component_peak_logits_v564 = masked_logits_v564.flatten(2).amax(dim=2)
            if bool(teacher_valid.any().item()):
                v564_component_seed_bce = F.binary_cross_entropy_with_logits(
                    component_peak_logits_v564[teacher_valid], torch.ones_like(component_peak_logits_v564[teacher_valid])
                )
            else:
                v564_component_seed_bce = v562_residual_proposal_loss * 0.0
            v562_residual_proposal_loss = v562_residual_proposal_loss + float(
                _m1(cfg, "V564_COMPONENT_SEED_WEIGHT", 0.50)
            ) * v564_component_seed_bce

    # V552-R4.20.4 fixes a target/representation mismatch in R4.20.3.
    # R4.17 location logits are trained on Gaussian component centres and must
    # never be interpreted as dense residual occupancy.  The dedicated R4.20.4
    # occupancy head is supervised by the union of the retained exact Teacher
    # component masks using only standard region-balanced BCE + soft Dice.
    r4204_occupancy_bce_loss = slot_masks.sum() * 0.0
    r4204_occupancy_dice_loss = slot_masks.sum() * 0.0
    r4204_occupancy_loss = slot_masks.sum() * 0.0
    r4204_occupancy_target_fraction = slot_masks.new_zeros(())
    r4204_occupancy_probability_mean = slot_masks.new_zeros(())
    r4204_occupancy_soft_dice = slot_masks.new_zeros(())
    r4204_retained_teacher_residual_fraction = slot_masks.new_zeros(())
    r4204_unretained_teacher_residual_fraction = slot_masks.new_zeros(())
    r4204_occupancy_target_teacher_error_mae = slot_masks.new_zeros(())
    r4205_overflow_target_fraction = slot_masks.new_zeros(())
    r4205_overflow_target_residual_fraction = slot_masks.new_zeros(())
    r4205_overflow_probability_mean = slot_masks.new_zeros(())
    r4205_overflow_predicted_residual_fraction = slot_masks.new_zeros(())
    r4205_overflow_soft_dice = slot_masks.new_zeros(())
    r4205_overflow_soft_precision = slot_masks.new_zeros(())
    r4205_overflow_soft_coverage = slot_masks.new_zeros(())
    r4205_editable_on_overflow_leakage = slot_masks.new_zeros(())
    r4205_overflow_on_retained_leakage = slot_masks.new_zeros(())
    r4205_target_decomposition_error = slot_masks.new_zeros(())
    r4205_prediction_decomposition_error = slot_masks.new_zeros(())
    r4205_final_logits_finite_fraction = slot_masks.new_ones(())
    # R4.20.6 aligns the training likelihood with the R4.20.5 model
    # factorization: P(residual) * P(identity | residual).  No new loss-weight
    # hyperparameter is introduced; the old per-slot BCE term is replaced by
    # one standard residual-masked K+1 categorical CE, while Dice remains the
    # matched-component shape term.
    r4206_conditional_identity_loss = slot_masks.sum() * 0.0
    r4206_shape_dice_loss = slot_masks.sum() * 0.0
    r4206_conditional_accuracy = slot_masks.new_zeros(())
    r4206_editable_accuracy = slot_masks.new_zeros(())
    r4206_overflow_recall = slot_masks.new_zeros(())
    r4206_overflow_precision = slot_masks.new_zeros(())
    r4206_overflow_target_residual_fraction = slot_masks.new_zeros(())
    r4206_overflow_predicted_residual_fraction = slot_masks.new_zeros(())
    r4206_true_class_probability = slot_masks.new_zeros(())
    r4206_target_decomposition_error = slot_masks.new_zeros(())
    r4206_teacher_overlap_rate = slot_masks.new_zeros(())
    r4206_supervised_residual_fraction = slot_masks.new_zeros(())
    r4206_unsupervised_residual_pixels = slot_masks.new_zeros(())
    r4206_empty_residual_batch = slot_masks.new_zeros(())
    r4209_balanced_overflow_loss = slot_masks.sum() * 0.0
    r4209_overflow_positive_loss = slot_masks.sum() * 0.0
    r4209_overflow_negative_loss = slot_masks.sum() * 0.0
    r4209_overflow_target_present = slot_masks.new_zeros(())
    r4210_independent_overflow_loss = slot_masks.sum() * 0.0
    r4210_overflow_positive_loss = slot_masks.sum() * 0.0
    r4210_overflow_negative_loss = slot_masks.sum() * 0.0
    r4210_overflow_target_present = slot_masks.new_zeros(())
    if r4204_enabled:
        occupancy_logits = aux.get("v552r4204_occupancy_logits")
        if not isinstance(occupancy_logits, torch.Tensor) or occupancy_logits.shape != (b, 1, h, w):
            raise RuntimeError(
                "V552-R4.20.4 requires v552r4204_occupancy_logits [B,1,H,W], got "
                + str(None if not isinstance(occupancy_logits, torch.Tensor) else tuple(occupancy_logits.shape))
            )
        # IMPORTANT: residual existence must be independent of the number of
        # component slots K.  ``teacher_masks`` contains at most K retained
        # components, so using their union as the occupancy target would make
        # P(residual) implicitly depend on K and on Teacher top-K selection.
        # The exact dense target is the full effective Base-vs-GT error map
        # (after the existing deterministic replay policy, if active).
        occupancy_target = teacher_error.detach().clamp(0.0, 1.0)

        # Keep the retained-component union only as a diagnostic of capacity:
        # it tells us how much of the full residual field is representable by
        # the configured component set, but it never supervises occupancy.
        valid_teacher_mask = teacher_valid[:, :, None, None].to(teacher_masks.dtype)
        retained_occupancy_target = (
            teacher_masks * valid_teacher_mask
        ).amax(dim=1, keepdim=True).detach()

        occupancy_element = F.binary_cross_entropy_with_logits(
            occupancy_logits, occupancy_target, reduction="none"
        )
        occ_pos = occupancy_target.sum().clamp_min(1.0)
        occ_neg = (1.0 - occupancy_target).sum().clamp_min(1.0)
        r4204_occupancy_bce_loss = 0.5 * (
            (occupancy_element * occupancy_target).sum() / occ_pos
            + (occupancy_element * (1.0 - occupancy_target)).sum() / occ_neg
        )
        occupancy_probability = torch.sigmoid(occupancy_logits).clamp(EPS, 1.0 - EPS)
        inter = (occupancy_probability * occupancy_target).flatten(1).sum(dim=1)
        denom = (
            occupancy_probability.flatten(1).sum(dim=1)
            + occupancy_target.flatten(1).sum(dim=1)
        )
        occupancy_soft_dice_case = (2.0 * inter + EPS) / (denom + EPS)
        r4204_occupancy_dice_loss = 1.0 - occupancy_soft_dice_case.mean()
        r4204_occupancy_loss = 0.5 * (
            r4204_occupancy_bce_loss + r4204_occupancy_dice_loss
        )
        r4204_occupancy_target_fraction = occupancy_target.mean().detach()
        r4204_occupancy_probability_mean = occupancy_probability.mean().detach()
        r4204_occupancy_soft_dice = occupancy_soft_dice_case.mean().detach()

        full_residual_mass = occupancy_target.flatten(1).sum(dim=1)
        retained_residual_mass = (
            retained_occupancy_target * occupancy_target
        ).flatten(1).sum(dim=1)
        nonempty_full = full_residual_mass > 0
        retained_fraction_case = torch.where(
            nonempty_full,
            retained_residual_mass / full_residual_mass.clamp_min(1.0),
            torch.ones_like(full_residual_mass),
        )
        r4204_retained_teacher_residual_fraction = retained_fraction_case.mean().detach()
        r4204_unretained_teacher_residual_fraction = (
            1.0 - retained_fraction_case
        ).mean().detach()
        r4204_occupancy_target_teacher_error_mae = (
            occupancy_target - teacher_error.detach().clamp(0.0, 1.0)
        ).abs().mean().detach()

        if r4205_enabled:
            overflow_probability = aux.get("v552r4205_overflow_probability")
            editable_probability_sum = aux.get("v552r4205_editable_probability_sum")
            residual_probability_native = torch.sigmoid(occupancy_logits).clamp(EPS, 1.0 - EPS)
            expected_shape = (b, 1, h, w)
            if (
                not isinstance(overflow_probability, torch.Tensor)
                or overflow_probability.shape != expected_shape
            ):
                raise RuntimeError(
                    "V552-R4.20.5 requires v552r4205_overflow_probability [B,1,H,W], got "
                    + str(None if not isinstance(overflow_probability, torch.Tensor) else tuple(overflow_probability.shape))
                )
            if (
                not isinstance(editable_probability_sum, torch.Tensor)
                or editable_probability_sum.shape != expected_shape
            ):
                raise RuntimeError(
                    "V552-R4.20.5 requires v552r4205_editable_probability_sum [B,1,H,W], got "
                    + str(None if not isinstance(editable_probability_sum, torch.Tensor) else tuple(editable_probability_sum.shape))
                )

            # Capacity-consistent target decomposition.  The retained union is
            # the portion representable by K editable Teacher components; all
            # remaining full residual pixels are an explicit reject/overflow
            # target.  No new loss is introduced in R4.20.5: existing matched
            # component suppression plus full occupancy supervision provides the
            # learning signal for the reference category.
            retained_target = (
                occupancy_target * retained_occupancy_target
            ).clamp(0.0, 1.0)
            overflow_target = (
                occupancy_target * (1.0 - retained_occupancy_target)
            ).clamp(0.0, 1.0)
            target_reconstruction = retained_target + overflow_target
            r4205_target_decomposition_error = (
                target_reconstruction - occupancy_target
            ).abs().mean().detach()

            prediction_reconstruction = editable_probability_sum + overflow_probability
            r4205_prediction_decomposition_error = (
                prediction_reconstruction - residual_probability_native
            ).abs().mean().detach()

            overflow_target_mass = overflow_target.sum()
            retained_target_mass = retained_target.sum()
            full_target_mass = occupancy_target.sum().clamp_min(EPS)
            overflow_pred_mass = overflow_probability.sum()
            residual_pred_mass = residual_probability_native.sum().clamp_min(EPS)
            overflow_intersection = (overflow_probability * overflow_target).sum()

            if r4209_balanced_overflow:
                # Capacity supervision is binary *conditional on residual*: a
                # residual pixel is either retained/editable or overflow.  This
                # avoids R4.20.6's pixel-count-dominated K+1 CE, which improved
                # the dustbin while collapsing small/high-utility components.
                # Positive and negative residual classes are averaged separately
                # using standard BCE, so no class-frequency or loss-weight knob
                # is introduced.
                conditional_overflow = (
                    overflow_probability / residual_probability_native.clamp_min(EPS)
                ).clamp(EPS, 1.0 - EPS)
                residual_mask = occupancy_target > 0.5
                overflow_pos = (overflow_target > 0.5) & residual_mask
                retained_neg = (retained_target > 0.5) & residual_mask
                if bool(overflow_pos.any().item()):
                    r4209_overflow_positive_loss = (
                        -torch.log(conditional_overflow[overflow_pos])
                    ).mean()
                    r4209_overflow_target_present = conditional_overflow.new_ones(())
                if bool(retained_neg.any().item()):
                    r4209_overflow_negative_loss = (
                        -torch.log1p(-conditional_overflow[retained_neg])
                    ).mean()
                class_terms = []
                if bool(overflow_pos.any().item()):
                    class_terms.append(r4209_overflow_positive_loss)
                if bool(retained_neg.any().item()):
                    class_terms.append(r4209_overflow_negative_loss)
                if class_terms:
                    r4209_balanced_overflow_loss = torch.stack(class_terms).mean()

            if r4210_independent_overflow:
                # R4.21.0 factorizes capacity from editable identity.  The
                # overflow head predicts P(overflow | residual) independently
                # from the K-way editable-slot softmax.  Its standard binary
                # BCE therefore cannot directly steal relative probability
                # from any editable slot.  Positive/negative residual classes
                # are mean-normalized separately; no tunable class weight is
                # introduced.
                overflow_gate_logits = aux.get("v552r4210_overflow_gate_logits")
                if (
                    not isinstance(overflow_gate_logits, torch.Tensor)
                    or overflow_gate_logits.shape != (b, 1, h, w)
                ):
                    raise RuntimeError(
                        "V552-R4.21.0 independent overflow requires "
                        "v552r4210_overflow_gate_logits [B,1,H,W]"
                    )
                residual_mask = occupancy_target > 0.5
                overflow_pos = (overflow_target > 0.5) & residual_mask
                retained_neg = (retained_target > 0.5) & residual_mask
                independent_terms = []
                if bool(overflow_pos.any().item()):
                    r4210_overflow_positive_loss = F.softplus(
                        -overflow_gate_logits[overflow_pos]
                    ).mean()
                    r4210_overflow_target_present = overflow_gate_logits.new_ones(())
                    independent_terms.append(r4210_overflow_positive_loss)
                if bool(retained_neg.any().item()):
                    r4210_overflow_negative_loss = F.softplus(
                        overflow_gate_logits[retained_neg]
                    ).mean()
                    independent_terms.append(r4210_overflow_negative_loss)
                if independent_terms:
                    r4210_independent_overflow_loss = torch.stack(independent_terms).mean()

            r4205_overflow_target_fraction = overflow_target.mean().detach()
            r4205_overflow_target_residual_fraction = (
                overflow_target_mass / full_target_mass
            ).detach()
            r4205_overflow_probability_mean = overflow_probability.mean().detach()
            r4205_overflow_predicted_residual_fraction = (
                overflow_pred_mass / residual_pred_mass
            ).detach()
            r4205_overflow_soft_dice = (
                (2.0 * overflow_intersection + EPS)
                / (overflow_pred_mass + overflow_target_mass + EPS)
            ).detach()
            r4205_overflow_soft_precision = (
                (overflow_intersection + EPS) / (overflow_pred_mass + EPS)
            ).detach()
            r4205_overflow_soft_coverage = (
                (overflow_intersection + EPS) / (overflow_target_mass + EPS)
            ).detach()
            r4205_editable_on_overflow_leakage = torch.where(
                overflow_target_mass > 0,
                (editable_probability_sum * overflow_target).sum()
                / overflow_target_mass.clamp_min(EPS),
                editable_probability_sum.new_zeros(()),
            ).detach()
            r4205_overflow_on_retained_leakage = torch.where(
                retained_target_mass > 0,
                (overflow_probability * retained_target).sum()
                / retained_target_mass.clamp_min(EPS),
                overflow_probability.new_zeros(()),
            ).detach()
            r4205_final_logits_finite_fraction = aux.get(
                "v552r4205_final_logits_finite_fraction", slot_masks.new_ones(())
            ).detach()

    action_candidates = aux["v532_action_candidate_probs"].detach().clamp(
        EPS, 1.0 - EPS
    )
    if action_candidates.ndim != 4 or action_candidates.shape[1] != 4:
        raise ValueError(
            "V538 requires v532_action_candidate_probs=[B,4,H,W], got "
            f"{tuple(action_candidates.shape)}"
        )
    if action_candidates.shape[-2:] != (h, w):
        action_candidates = F.interpolate(
            action_candidates, size=(h, w), mode="bilinear", align_corners=False
        )

    teacher_polarity = (heuristic_teacher_actions % 2).long()
    continuous_dose_enabled = bool(_m1(cfg, "V540_CONTINUOUS_DOSE_ENABLED", True))
    with torch.no_grad():
        hard_base_for_teacher = (base >= 0.5).to(gt.dtype)
        hard_base_dice_for_teacher = _dice_many(
            hard_base_for_teacher[:, 0][:, None], gt
        )[:, 0]
        teacher_mask = teacher_masks.to(base.dtype)
        if continuous_dose_enabled:
            doses = _dose_grid(cfg, base)
            base_logit = torch.logit(base.clamp(EPS, 1.0 - EPS))[:, 0][:, None]
            polarity_sign = torch.where(
                teacher_polarity == 0,
                teacher_masks.new_full(teacher_polarity.shape, -1.0),
                teacher_masks.new_ones(teacher_polarity.shape),
            )
            dose_gain_terms = []
            for dose in doses:
                signed_delta = polarity_sign[:, :, None, None] * dose * teacher_mask
                candidate_probability = torch.sigmoid(base_logit + signed_delta)
                candidate_dice = _dice_binary_many(candidate_probability >= 0.5, gt)
                dose_gain_terms.append(
                    candidate_dice - hard_base_dice_for_teacher[:, None]
                )
            teacher_dose_gain = torch.stack(dose_gain_terms, dim=2).masked_fill(
                ~teacher_valid[:, :, None], -1.0e4
            )
            best_teacher_action_gain, best_dose_index = teacher_dose_gain.max(dim=2)
            best_teacher_dose = doses[best_dose_index]
            best_teacher_action_gain = torch.where(
                teacher_valid,
                best_teacher_action_gain,
                torch.zeros_like(best_teacher_action_gain),
            )
            teacher_actions = heuristic_teacher_actions
        else:
            hard_action_candidates = (action_candidates >= 0.5).to(gt.dtype)
            action_gain_terms = []
            teacher_mask_bool = teacher_masks.bool()
            base_expand = hard_base_for_teacher[:, 0][:, None].expand(-1, k, -1, -1)
            for action_index in range(4):
                action_expand = hard_action_candidates[:, action_index][:, None].expand(
                    -1, k, -1, -1
                )
                candidate = torch.where(
                    teacher_mask_bool, action_expand.bool(), base_expand.bool()
                ).to(gt.dtype)
                candidate_dice = _dice_binary_many(candidate, gt)
                action_gain_terms.append(
                    candidate_dice - hard_base_dice_for_teacher[:, None]
                )
            teacher_action_gain = torch.stack(action_gain_terms, dim=2)
            negative_component = teacher_polarity == 0
            allowed_action = torch.zeros_like(teacher_action_gain, dtype=torch.bool)
            allowed_action[:, :, 0] = negative_component
            allowed_action[:, :, 2] = negative_component
            allowed_action[:, :, 1] = ~negative_component
            allowed_action[:, :, 3] = ~negative_component
            allowed_action &= teacher_valid[:, :, None]
            allowed_gain = teacher_action_gain.masked_fill(~allowed_action, -1.0e4)
            best_teacher_action_gain, best_teacher_action = allowed_gain.max(dim=2)
            teacher_actions = torch.where(
                teacher_valid, best_teacher_action, heuristic_teacher_actions
            )
            best_teacher_dose = slot_doses.new_full(
                teacher_valid.shape, float(_m1(cfg, "V540_INITIAL_DOSE", 1.0))
            )
            best_teacher_action_gain = torch.where(
                teacher_valid,
                best_teacher_action_gain,
                torch.zeros_like(best_teacher_action_gain),
            )

    teacher_oracle_case_for_weight = best_teacher_action_gain.clamp_min(0.0).max(
        dim=1
    ).values
    teacher_utility = (
        best_teacher_action_gain.clamp_min(0.0)
        / teacher_oracle_case_for_weight[:, None].clamp_min(5.0e-4)
    ).clamp(0.0, 1.0)

    # R4.7 explicit component geometry.  This is derived from the exact
    # training teacher only after forward, never fed into the model.  It is
    # used both as an auxiliary localization target and as a mild hybrid
    # matching term, following Mask DINO's mask+box matching principle.
    y_coord = torch.linspace(0.0, 1.0, h, device=teacher_masks.device, dtype=teacher_masks.dtype)
    x_coord = torch.linspace(0.0, 1.0, w, device=teacher_masks.device, dtype=teacher_masks.dtype)
    teacher_bool = teacher_masks > 0.5
    x_map = x_coord[None, None, None, :].expand(b, k, h, w)
    y_map = y_coord[None, None, :, None].expand(b, k, h, w)
    teacher_xmin = torch.where(teacher_bool, x_map, x_map.new_ones(())).amin(dim=(-2, -1))
    teacher_xmax = torch.where(teacher_bool, x_map, x_map.new_zeros(())).amax(dim=(-2, -1))
    teacher_ymin = torch.where(teacher_bool, y_map, y_map.new_ones(())).amin(dim=(-2, -1))
    teacher_ymax = torch.where(teacher_bool, y_map, y_map.new_zeros(())).amax(dim=(-2, -1))
    teacher_cx = 0.5 * (teacher_xmin + teacher_xmax)
    teacher_cy = 0.5 * (teacher_ymin + teacher_ymax)
    teacher_w = (teacher_xmax - teacher_xmin + 1.0 / float(max(w, 1))).clamp(0.0, 1.0)
    teacher_h = (teacher_ymax - teacher_ymin + 1.0 / float(max(h, 1))).clamp(0.0, 1.0)
    teacher_geometry = torch.stack([teacher_cx, teacher_cy, teacher_w, teacher_h], dim=2)
    teacher_geometry = torch.where(
        teacher_valid[:, :, None], teacher_geometry, teacher_geometry.new_zeros(teacher_geometry.shape)
    )

    # V552-R4.21.0: residual components are often crescent/ring/boundary-band
    # shapes whose bounding-box centre may lie outside the component.  Build a
    # deterministic guaranteed-inside anchor from the exact Teacher mask and
    # keep box geometry only for legacy extent diagnostics.
    teacher_interior_anchor_xy, teacher_interior_anchor_index, r4210_interior_anchor_inside_rate = (
        _r4210_component_interior_anchor(teacher_masks, teacher_valid)
    )
    batch_grid_r4210 = torch.arange(b, device=teacher_masks.device)[:, None].expand(b, k)
    slot_grid_r4210 = torch.arange(k, device=teacher_masks.device)[None, :].expand(b, k)
    bbox_px_r4210 = (teacher_cx * float(max(w - 1, 1))).round().long().clamp(0, w - 1)
    bbox_py_r4210 = (teacher_cy * float(max(h - 1, 1))).round().long().clamp(0, h - 1)
    bbox_inside_r4210 = teacher_bool[
        batch_grid_r4210, slot_grid_r4210, bbox_py_r4210, bbox_px_r4210
    ]
    if bool(teacher_valid.any().item()):
        r4210_bbox_center_inside_teacher_rate = (
            bbox_inside_r4210[teacher_valid].to(teacher_masks.dtype).mean().detach()
        )
        anchor_py_r4210 = torch.div(teacher_interior_anchor_index, w, rounding_mode="floor")
        anchor_px_r4210 = teacher_interior_anchor_index.remainder(w)
        r4210_bbox_to_interior_anchor_distance_px = torch.sqrt(
            (bbox_px_r4210.to(teacher_masks.dtype) - anchor_px_r4210.to(teacher_masks.dtype)).square()
            + (bbox_py_r4210.to(teacher_masks.dtype) - anchor_py_r4210.to(teacher_masks.dtype)).square()
            + 1.0e-12
        )[teacher_valid].mean().detach()
    else:
        r4210_bbox_center_inside_teacher_rate = teacher_masks.new_ones(())
        r4210_bbox_to_interior_anchor_distance_px = teacher_masks.new_zeros(())

    # C0 preserves the historical bbox-centre target exactly.  C1+ switches
    # only the instance anchor to a guaranteed-inside residual pixel; box
    # geometry remains available for extent/size supervision.
    teacher_seed_xy = (
        teacher_interior_anchor_xy
        if r4210_interior_anchor
        else teacher_geometry[:, :, :2]
    )

    # V552-R4.11 typed residual proposal supervision.  This directly trains
    # localization and preserves Delete/Fill/Trim/Expand identity instead of
    # deriving proposals from a detached class-agnostic cause maximum.
    r411_center_loss = slot_masks.sum() * 0.0
    r411_size_loss = slot_masks.sum() * 0.0
    r411_offset_loss = slot_masks.sum() * 0.0
    r411_center_recall = slot_masks.new_zeros(())
    r411_teacher_center_probability = slot_masks.new_zeros(())
    r416_pre_topk_peak_recall = slot_masks.new_zeros(())
    r417_location_loss = slot_masks.sum() * 0.0
    r417_location_target_mass = slot_masks.new_zeros(())
    r417_location_center_recall = slot_masks.new_zeros(())
    r417_pre_topk_peak_recall = slot_masks.new_zeros(())
    r417_selected_spatial_coverage = slot_masks.new_zeros(())
    r417_location_offset_loss = slot_masks.sum() * 0.0
    r417_location_offset_mae_px = slot_masks.new_zeros(())
    if r411_enabled:
        proposal_center_logits = aux.get("v552r411_center_logits")
        proposal_size_map = aux.get("v552r411_size_map")
        proposal_offset_map = aux.get("v552r411_offset_map")
        valid_shapes = (
            isinstance(proposal_center_logits, torch.Tensor)
            and proposal_center_logits.shape == (b, 4, h, w)
            and isinstance(proposal_size_map, torch.Tensor)
            and proposal_size_map.shape == (b, 4, 2, h, w)
            and isinstance(proposal_offset_map, torch.Tensor)
            and proposal_offset_map.shape == (b, 4, 2, h, w)
        )
        if not valid_shapes:
            raise RuntimeError(
                "V552-R4.11 requires typed proposal center/size/offset maps with "
                f"shapes [B,4,H,W]/[B,4,2,H,W]; got "
                f"{None if not isinstance(proposal_center_logits, torch.Tensor) else tuple(proposal_center_logits.shape)}"
            )

        # Gaussian center targets follow the standard center-based detector
        # principle; scale is derived from each exact residual box.
        gy = torch.arange(h, device=slot_masks.device, dtype=slot_masks.dtype)[None, None, :, None]
        gx = torch.arange(w, device=slot_masks.device, dtype=slot_masks.dtype)[None, None, None, :]
        center_target = torch.zeros_like(proposal_center_logits)
        for teacher_index in range(k):
            valid_j = teacher_valid[:, teacher_index]
            if not bool(valid_j.any().item()):
                continue
            geom = teacher_geometry[:, teacher_index]
            center_xy = teacher_seed_xy[:, teacher_index]
            action_j = teacher_actions[:, teacher_index].clamp(0, 3)
            if r4210_interior_anchor:
                cx_pix = center_xy[:, 0] * float(w) - 0.5
                cy_pix = center_xy[:, 1] * float(h) - 0.5
            else:
                cx_pix = geom[:, 0] * float(max(w - 1, 1))
                cy_pix = geom[:, 1] * float(max(h - 1, 1))
            sigma_x = (0.25 * geom[:, 2] * float(w)).clamp_min(1.0)
            sigma_y = (0.25 * geom[:, 3] * float(h)).clamp_min(1.0)
            gaussian = torch.exp(
                -0.5 * (
                    ((gx - cx_pix[:, None, None, None]) / sigma_x[:, None, None, None]) ** 2
                    + ((gy - cy_pix[:, None, None, None]) / sigma_y[:, None, None, None]) ** 2
                )
            )[:, 0]
            for action_index in range(4):
                active = valid_j & (action_j == action_index)
                if bool(active.any().item()):
                    center_target[active, action_index] = torch.maximum(
                        center_target[active, action_index], gaussian[active]
                    )

        center_bce = F.binary_cross_entropy_with_logits(
            proposal_center_logits, center_target, reduction="none"
        )
        positive_mass = center_target.sum().clamp_min(1.0)
        negative_mass = (1.0 - center_target).sum().clamp_min(1.0)
        positive_center_loss = (center_bce * center_target).sum() / positive_mass
        negative_center_loss = (center_bce * (1.0 - center_target)).sum() / negative_mass
        r411_center_loss = 0.5 * (positive_center_loss + negative_center_loss)

        # R4.17: explicit location-first target.  Location is the union of all
        # typed Gaussian center targets; action identity cannot suppress a
        # physical residual location.  This is standard balanced BCE, not a
        # new task-specific loss family.
        if r417_enabled:
            location_logits = aux.get("v552r417_location_logits")
            location_offset_map = aux.get("v552r417_location_offset_map")
            if not isinstance(location_logits, torch.Tensor) or location_logits.shape != (b, 1, h, w):
                raise RuntimeError("V552-R4.17 requires v552r417_location_logits [B,1,H,W]")
            if not isinstance(location_offset_map, torch.Tensor) or location_offset_map.shape != (b, 2, h, w):
                raise RuntimeError("V552-R4.17 requires v552r417_location_offset_map [B,2,H,W]")
            location_target = center_target.max(dim=1, keepdim=True).values
            if r4209_peak_focal:
                # R4.17's balanced BCE learned a broad "residual is nearby"
                # field (high centre recall) but did not guarantee one strong
                # physical maximum per Teacher component.  R4.20.9 keeps the
                # same geometry-derived Gaussian target but forces each exact
                # Teacher center pixel to one and applies the canonical
                # CenterNet modified focal heatmap objective.
                location_target = location_target.clone()
                batch_index_peak = torch.arange(b, device=slot_masks.device)
                for teacher_index in range(k):
                    valid_j = teacher_valid[:, teacher_index]
                    if not bool(valid_j.any().item()):
                        continue
                    if r4210_interior_anchor:
                        anchor_index_j = teacher_interior_anchor_index[:, teacher_index]
                        py_j = torch.div(anchor_index_j, w, rounding_mode="floor")
                        px_j = anchor_index_j.remainder(w)
                    else:
                        geom_j = teacher_geometry[:, teacher_index]
                        px_j = (geom_j[:, 0] * float(max(w - 1, 1))).round().long().clamp(0, w - 1)
                        py_j = (geom_j[:, 1] * float(max(h - 1, 1))).round().long().clamp(0, h - 1)
                    active_batch = batch_index_peak[valid_j]
                    location_target[active_batch, 0, py_j[valid_j], px_j[valid_j]] = 1.0
                r417_location_loss = _r4209_centernet_focal_loss(
                    location_logits, location_target.detach()
                )
            else:
                location_bce = F.binary_cross_entropy_with_logits(
                    location_logits, location_target, reduction="none"
                )
                loc_pos = location_target.sum().clamp_min(1.0)
                loc_neg = (1.0 - location_target).sum().clamp_min(1.0)
                r417_location_loss = 0.5 * (
                    (location_bce * location_target).sum() / loc_pos
                    + (location_bce * (1.0 - location_target)).sum() / loc_neg
                )
            r417_location_target_mass = location_target.sum().detach()

        valid_count = teacher_valid.float().sum().clamp_min(1.0)
        size_terms = []
        offset_terms = []
        center_prob_terms = []
        center_hit_terms = []
        r417_location_hit_terms = []
        r417_offset_terms = []
        r417_offset_mae_terms = []
        center_prob = torch.sigmoid(proposal_center_logits)
        if r417_enabled:
            r417_location_prob = torch.sigmoid(aux["v552r417_location_logits"])
            r417_local_location_prob = F.max_pool2d(r417_location_prob, 7, stride=1, padding=3)
        else:
            r417_location_prob = None
            r417_local_location_prob = None
        local_center_prob = F.max_pool2d(center_prob, 7, stride=1, padding=3)
        proposal_threshold = float(_m1(cfg, "V552R411_PROPOSAL_SCORE_THRESHOLD", 0.01))
        batch_index_all = torch.arange(b, device=slot_masks.device)
        for teacher_index in range(k):
            valid_j = teacher_valid[:, teacher_index]
            if not bool(valid_j.any().item()):
                continue
            geom = teacher_geometry[:, teacher_index]
            center_xy = teacher_seed_xy[:, teacher_index]
            action_j = teacher_actions[:, teacher_index].clamp(0, 3)
            px_float = center_xy[:, 0] * float(w) - 0.5
            py_float = center_xy[:, 1] * float(h) - 0.5
            px = px_float.round().long().clamp(0, w - 1)
            py = py_float.round().long().clamp(0, h - 1)
            pred_size = proposal_size_map[batch_index_all, action_j, :, py, px]
            pred_offset = proposal_offset_map[batch_index_all, action_j, :, py, px]
            target_offset = torch.stack(
                [
                    center_xy[:, 0] * float(w) - (px.to(center_xy.dtype) + 0.5),
                    center_xy[:, 1] * float(h) - (py.to(center_xy.dtype) + 0.5),
                ],
                dim=1,
            ).clamp(-0.5, 0.5)
            size_map_loss = F.smooth_l1_loss(
                pred_size, geom[:, 2:].detach(), reduction="none", beta=0.02
            ).mean(dim=1)
            offset_map_loss = F.smooth_l1_loss(
                pred_offset, target_offset.detach(), reduction="none", beta=0.10
            ).mean(dim=1)
            if not r413_enabled:
                size_terms.append(size_map_loss[valid_j])
            offset_terms.append(offset_map_loss[valid_j])
            p_center = local_center_prob[batch_index_all, action_j, py, px]
            center_prob_terms.append(p_center[valid_j])
            center_hit_terms.append((p_center[valid_j] >= proposal_threshold).to(slot_masks.dtype))
            if r417_enabled:
                p_location = r417_local_location_prob[:, 0][batch_index_all, py, px]
                r417_location_hit_terms.append(
                    (p_location[valid_j] >= proposal_threshold).to(slot_masks.dtype)
                )
                if r417_shared_offset_enabled:
                    pred_shared_offset = aux["v552r417_location_offset_map"].permute(0, 2, 3, 1)[
                        batch_index_all, py, px
                    ]
                    shared_map_loss = F.smooth_l1_loss(
                        pred_shared_offset, target_offset.detach(), reduction="none", beta=0.10
                    ).mean(dim=1)
                    r417_offset_terms.append(shared_map_loss[valid_j])
                    scale = pred_shared_offset.new_tensor([float(w), float(h)])
                    r417_offset_mae_terms.append(
                        ((pred_shared_offset - target_offset.detach()).abs() * scale).mean(dim=1)[valid_j]
                    )

        if size_terms:
            r411_size_loss = torch.cat(size_terms).mean()
        if offset_terms:
            r411_offset_loss = torch.cat(offset_terms).mean()
        if center_prob_terms:
            r411_teacher_center_probability = torch.cat(center_prob_terms).mean().detach()
        if center_hit_terms:
            r411_center_recall = torch.cat(center_hit_terms).mean().detach()
        if r417_location_hit_terms:
            r417_location_center_recall = torch.cat(r417_location_hit_terms).mean().detach()
        if r417_offset_terms:
            r417_location_offset_loss = torch.cat(r417_offset_terms).mean()
        if r417_offset_mae_terms:
            r417_location_offset_mae_px = torch.cat(r417_offset_mae_terms).mean().detach()

        # R4.17 proposal-recovery diagnostic: measure a local maximum from the
        # dedicated location map before Top-K.  Unlike R4.16, action-channel
        # competition is absent by construction.
        if r417_enabled:
            nms_kernel417 = max(int(_m1(cfg, "V552R417_LOCATION_NMS_KERNEL", 3)), 1)
            if nms_kernel417 % 2 == 0:
                nms_kernel417 += 1
            radius417 = nms_kernel417 // 2
            loc_prob = torch.sigmoid(aux["v552r417_location_logits"])
            loc_max = F.max_pool2d(loc_prob, nms_kernel417, stride=1, padding=radius417)
            loc_peak = loc_prob * (loc_prob >= (loc_max - 1.0e-6)).to(loc_prob.dtype)
            local_loc_peak = F.max_pool2d(loc_peak, 7, stride=1, padding=3)[:, 0]
            r417_peak_terms = []
            for teacher_index in range(k):
                valid_j = teacher_valid[:, teacher_index]
                if not bool(valid_j.any().item()):
                    continue
                center_xy = teacher_seed_xy[:, teacher_index]
                px = (center_xy[:, 0] * float(w) - 0.5).round().long().clamp(0, w - 1)
                py = (center_xy[:, 1] * float(h) - 0.5).round().long().clamp(0, h - 1)
                p = local_loc_peak[torch.arange(b, device=slot_masks.device), py, px]
                r417_peak_terms.append((p[valid_j] >= proposal_threshold).to(slot_masks.dtype))
            if r417_peak_terms:
                r417_pre_topk_peak_recall = torch.cat(r417_peak_terms).mean().detach()

        # R4.16 causal diagnostic: measure whether a *physical*, cross-type
        # NMS peak exists around each Teacher before the hard Top-K capacity
        # bottleneck.  Comparing this with post-TopK trusted identity coverage
        # directly exposes slot loss caused by typed duplicate competition.
        if r416_enabled:
            spatial_prob = center_prob.max(dim=1, keepdim=True).values
            nms_radius = max(int(round(float(_m1(cfg, "V552R416_CROSS_TYPE_NMS_RADIUS_PX", 4.0)))), 1)
            nms_kernel = 2 * nms_radius + 1
            spatial_max = F.max_pool2d(spatial_prob, nms_kernel, stride=1, padding=nms_radius)
            spatial_peak = spatial_prob * (spatial_prob >= (spatial_max - 1.0e-6)).to(spatial_prob.dtype)
            local_peak = F.max_pool2d(spatial_peak, 7, stride=1, padding=3)[:, 0]
            hit_terms = []
            for teacher_index in range(k):
                valid_j = teacher_valid[:, teacher_index]
                if not bool(valid_j.any().item()):
                    continue
                center_xy = teacher_seed_xy[:, teacher_index]
                px = (center_xy[:, 0] * float(w) - 0.5).round().long().clamp(0, w - 1)
                py = (center_xy[:, 1] * float(h) - 0.5).round().long().clamp(0, h - 1)
                p = local_peak[torch.arange(b, device=slot_masks.device), py, px]
                hit_terms.append((p[valid_j] >= proposal_threshold).to(slot_masks.dtype))
            if hit_terms:
                r416_pre_topk_peak_recall = torch.cat(hit_terms).mean().detach()

    pred_flat = slot_masks.flatten(2)
    target_flat = teacher_masks.flatten(2)
    intersection = torch.einsum("bkp,bjp->bkj", pred_flat, target_flat)
    denominator = pred_flat.sum(dim=2)[:, :, None] + target_flat.sum(dim=2)[:, None, :]
    pair_dice = (2.0 * intersection + EPS) / (denominator + EPS)

    # V552-R4.20.7 binding diagnostics (diagnostic only; never enter loss).
    # For every valid Teacher component, ask which Native slot has the highest
    # mask Dice before Hungarian matching.  If several Teachers independently
    # choose the same slot, the set has a many-to-one merge/collision.  The
    # best-vs-second margin quantifies whether the ownership is decisive or
    # diffuse.  These are threshold-free set diagnostics.
    r4207_teacher_best_slot_collision_rate = pair_dice.new_zeros(())
    r4207_teacher_best_slot_margin = pair_dice.new_zeros(())
    r4207_teacher_best_slot_dice = pair_dice.new_zeros(())
    if r4205_enabled and bool(teacher_valid.any().item()):
        teacher_to_slot = pair_dice.detach().transpose(1, 2)  # [B,Teacher,PredSlot]
        top_count = min(2, teacher_to_slot.shape[2])
        top_value, top_index = torch.topk(
            teacher_to_slot, k=top_count, dim=2, largest=True, sorted=True
        )
        best_value = top_value[:, :, 0]
        second_value = (
            top_value[:, :, 1] if top_count > 1 else torch.zeros_like(best_value)
        )
        r4207_teacher_best_slot_dice = best_value[teacher_valid].mean().detach()
        r4207_teacher_best_slot_margin = (
            best_value[teacher_valid] - second_value[teacher_valid]
        ).mean().detach()
        duplicate_count = 0
        valid_count = 0
        best_slot = top_index[:, :, 0]
        for batch_index in range(b):
            valid_j = teacher_valid[batch_index]
            n_valid = int(valid_j.sum().item())
            if n_valid <= 0:
                continue
            chosen = best_slot[batch_index, valid_j]
            duplicate_count += n_valid - int(torch.unique(chosen).numel())
            valid_count += n_valid
        if valid_count > 0:
            r4207_teacher_best_slot_collision_rate = pair_dice.new_tensor(
                float(duplicate_count) / float(valid_count)
            )

    # R4.20.8 D1: GT-visible diagnostics only.  These values diagnose whether
    # R4.17 peaks are actually a usable one-instance-one-seed set.  They never
    # enter Native forward, seed selection, or any loss.
    r4208_seed_teacher_precision = pair_dice.new_zeros(())
    r4208_teacher_seed_coverage = pair_dice.new_zeros(())
    r4208_seed_duplicate_teacher_rate = pair_dice.new_zeros(())
    r4208_teacher_multi_seed_rate = pair_dice.new_zeros(())
    r4208_seed_without_teacher_rate = pair_dice.new_zeros(())
    r4208_seed_occupancy_probability = pair_dice.new_zeros(())
    r4208_seed_owner = torch.full((b, k), -1, device=slot_masks.device, dtype=torch.long)
    r4208_seed_score = aux.get("v552r4207_seed_score")
    r4208_seed_center = aux.get("v552r4207_seed_center_xy")
    r4208_seed_valid = aux.get("v552r4207_seed_valid")
    r4210_teacher_component_count = teacher_valid.to(slot_masks.dtype).sum(dim=1).mean().detach()
    r4210_valid_seed_count = slot_masks.new_zeros(())
    r4210_valid_seed_minus_teacher_count = slot_masks.new_zeros(())
    r4210_true_seed_score_mean = slot_masks.new_zeros(())
    r4210_false_seed_score_mean = slot_masks.new_zeros(())
    r4210_true_seed_logit_mean = slot_masks.new_zeros(())
    r4210_false_seed_logit_mean = slot_masks.new_zeros(())
    r4211_proposal_seed_count = aux.get("v552r4211_proposal_seed_count", slot_masks.new_zeros(())).detach()
    r4211_proposal_confidence_mean = aux.get("v552r4211_proposal_confidence_mean", slot_masks.new_zeros(())).detach()
    r4211_presence_expected_count = aux.get("v552r4211_presence_expected_count", slot_masks.new_zeros(())).detach()
    r4211_presence_hard_count = aux.get("v552r4211_presence_hard_count", slot_masks.new_zeros(())).detach()
    r4211_presence_minus_teacher_count = (r4211_presence_expected_count - r4210_teacher_component_count).detach()
    r4211_geometry_effective_l1 = aux.get("v552r4211_geometry_effective_l1", slot_masks.new_zeros(())).detach()
    if r4208_requested:
        if not isinstance(r4208_seed_score, torch.Tensor) or r4208_seed_score.shape != (b, k):
            raise RuntimeError("V552-R4.20.8 requires seed scores [B,K]")
        if not isinstance(r4208_seed_center, torch.Tensor) or r4208_seed_center.shape != (b, k, 2):
            raise RuntimeError("V552-R4.20.8 requires seed centers [B,K,2]")
        if not isinstance(r4208_seed_valid, torch.Tensor) or r4208_seed_valid.shape != (b, k):
            raise RuntimeError("V552-R4.20.8 requires seed validity [B,K]")
        valid_seed = r4208_seed_valid > 0.5
        seed_inside_count = 0
        selected_seed_count = int(valid_seed.sum().item())
        covered_teacher_count = 0
        duplicate_extra_count = 0
        multi_seed_teacher_count = 0
        valid_teacher_count = int(teacher_valid.sum().item())
        occupancy_terms = []
        occupancy_logits_r4208 = aux.get("v552r4204_occupancy_logits")
        for sample in range(b):
            teacher_seed_counts = [0 for _ in range(teacher_masks.shape[1])]
            for pred_index in range(k):
                if not bool(valid_seed[sample, pred_index].item()):
                    continue
                cx = float(r4208_seed_center[sample, pred_index, 0].detach().item())
                cy = float(r4208_seed_center[sample, pred_index, 1].detach().item())
                px = min(max(int(math.floor(cx * float(w))), 0), w - 1)
                py = min(max(int(math.floor(cy * float(h))), 0), h - 1)
                owner = -1
                for target_index in range(teacher_masks.shape[1]):
                    if not bool(teacher_valid[sample, target_index].item()):
                        continue
                    if float(teacher_masks[sample, target_index, py, px].detach().item()) >= 0.5:
                        owner = target_index
                        break
                r4208_seed_owner[sample, pred_index] = owner
                if owner >= 0:
                    seed_inside_count += 1
                    teacher_seed_counts[owner] += 1
                if isinstance(occupancy_logits_r4208, torch.Tensor) and occupancy_logits_r4208.shape == (b, 1, h, w):
                    occupancy_terms.append(
                        torch.sigmoid(occupancy_logits_r4208[sample, 0, py, px].detach())
                    )
            for target_index, count in enumerate(teacher_seed_counts):
                if not bool(teacher_valid[sample, target_index].item()):
                    continue
                if count > 0:
                    covered_teacher_count += 1
                if count > 1:
                    duplicate_extra_count += count - 1
                    multi_seed_teacher_count += 1
        if selected_seed_count > 0:
            r4208_seed_teacher_precision = pair_dice.new_tensor(
                float(seed_inside_count) / float(selected_seed_count)
            )
            r4208_seed_without_teacher_rate = pair_dice.new_tensor(
                float(selected_seed_count - seed_inside_count) / float(selected_seed_count)
            )
        if valid_teacher_count > 0:
            r4208_teacher_seed_coverage = pair_dice.new_tensor(
                float(covered_teacher_count) / float(valid_teacher_count)
            )
            r4208_teacher_multi_seed_rate = pair_dice.new_tensor(
                float(multi_seed_teacher_count) / float(valid_teacher_count)
            )
        if seed_inside_count > 0:
            r4208_seed_duplicate_teacher_rate = pair_dice.new_tensor(
                float(duplicate_extra_count) / float(seed_inside_count)
            )
        if occupancy_terms:
            r4208_seed_occupancy_probability = torch.stack(occupancy_terms).mean().to(pair_dice.dtype)

        if r4210_requested:
            r4210_valid_seed_count = valid_seed.to(slot_masks.dtype).sum(dim=1).mean().detach()
            r4210_valid_seed_minus_teacher_count = (
                valid_seed.to(slot_masks.dtype).sum(dim=1)
                - teacher_valid.to(slot_masks.dtype).sum(dim=1)
            ).mean().detach()
            seed_true = valid_seed & (r4208_seed_owner >= 0)
            seed_false = valid_seed & (r4208_seed_owner < 0)
            if bool(seed_true.any().item()):
                r4210_true_seed_score_mean = r4208_seed_score[seed_true].mean().detach()
            if bool(seed_false.any().item()):
                r4210_false_seed_score_mean = r4208_seed_score[seed_false].mean().detach()
            seed_logit_r4210 = aux.get("v552r4210_seed_logit")
            if isinstance(seed_logit_r4210, torch.Tensor) and seed_logit_r4210.shape == (b, k):
                if bool(seed_true.any().item()):
                    r4210_true_seed_logit_mean = seed_logit_r4210[seed_true].mean().detach()
                if bool(seed_false.any().item()):
                    r4210_false_seed_logit_mean = seed_logit_r4210[seed_false].mean().detach()

    r47_anchor_pair_cost = (
        (r47_anchor_params[:, :, None, :2] - teacher_geometry[:, None, :, :2]).abs().sum(dim=3)
        + 0.5 * (r47_anchor_params[:, :, None, 2:] - teacher_geometry[:, None, :, 2:]).abs().sum(dim=3)
    )
    r47_match_anchor_weight = (
        max(float(_m1(cfg, "V552R47_MATCH_ANCHOR_WEIGHT", 0.25)), 0.0)
        if r47_enabled and bool(_m1(cfg, "V552R47_HYBRID_MATCHING_ENABLED", False))
        else 0.0
    )
    if r412_enabled:
        # R4.12 removes the circular dependency exposed by R4.11: a bad tiny
        # mask must not decide which teacher component supervises that mask.
        # Geometry/type dominate assignment; mask overlap is only a tie-break.
        pair_iou_r412, pair_giou_r412 = _r412_pairwise_box_giou(
            r47_anchor_params, teacher_geometry
        )
        pair_center_l1_r412 = (
            r47_anchor_params[:, :, None, :2] - teacher_geometry[:, None, :, :2]
        ).abs().sum(dim=3)
        pair_box_l1_r412 = (
            r47_anchor_params[:, :, None, :] - teacher_geometry[:, None, :, :]
        ).abs().mean(dim=3)
        pair_match_score = (
            float(_m1(cfg, "V552R412_MATCH_MASK_WEIGHT", 0.25)) * pair_dice
            + float(_m1(cfg, "V552R412_MATCH_GIOU_WEIGHT", 1.0)) * pair_giou_r412
            - float(_m1(cfg, "V552R412_MATCH_CENTER_WEIGHT", 1.5)) * pair_center_l1_r412
            - float(_m1(cfg, "V552R412_MATCH_BOX_WEIGHT", 1.0)) * pair_box_l1_r412
        )
    else:
        pair_iou_r412 = pair_dice.new_zeros(pair_dice.shape)
        pair_giou_r412 = pair_dice.new_zeros(pair_dice.shape)
        pair_match_score = pair_dice - r47_match_anchor_weight * r47_anchor_pair_cost
    if v561_bcrs:
        action_probability_v561 = F.softmax(action_logits.detach(), dim=2)
        teacher_action_index_v561 = teacher_actions[:, None, :, None].expand(
            b, k, teacher_actions.shape[1], 1
        )
        pair_action_probability_v561 = action_probability_v561[:, :, None, :].expand(
            b, k, teacher_actions.shape[1], 4
        ).gather(3, teacher_action_index_v561).squeeze(3)
        if v562_rootfix:
            # V562 identity is geometry-first.  Proposal anchors are predicted
            # independently of mask shape, so same-type queries cannot match a
            # component solely because their action class is similar.
            center_distance_v562 = (
                r47_anchor_params[:, :, None, :2]
                - teacher_geometry[:, None, :, :2]
            ).abs().sum(dim=3)
            center_weight_v562 = max(float(_m1(cfg, "V562_MATCH_CENTER_WEIGHT", 0.50)), 0.0)
            action_weight_v562 = max(float(_m1(cfg, "V562_MATCH_ACTION_WEIGHT", 0.10)), 0.0)
            pair_match_score = (
                pair_dice
                - center_weight_v562 * center_distance_v562
                + action_weight_v562 * pair_action_probability_v561
            )
        else:
            pair_match_score = pair_match_score + pair_action_probability_v561
    else:
        pair_action_probability_v561 = pair_dice.new_zeros(pair_dice.shape)

    # R4.11/R4.12 typed matching.  Spatial overlap alone is not a
    # sufficient identity signal for residual corrections because Delete/Fill/
    # Trim/Expand may overlap around the same boundary.  The proposal type is
    # therefore part of the matching cost, while the dense typed proposal head
    # itself is trained independently above.
    r411_match_type_weight = (
        max(float(_m1(cfg, "V552R411_MATCH_TYPE_WEIGHT", 0.5)), 0.0)
        if r411_enabled
        else 0.0
    )
    r411_proposal_type = aux.get("v552r411_proposal_type")
    if r411_enabled:
        if not isinstance(r411_proposal_type, torch.Tensor) or r411_proposal_type.shape != (b, k):
            raise RuntimeError(
                "V552-R4.11 requires v552r411_proposal_type with shape [B,K] "
                f"for typed component matching; got {None if not isinstance(r411_proposal_type, torch.Tensor) else tuple(r411_proposal_type.shape)}"
            )
        r411_type_pair_cost = (
            r411_proposal_type[:, :, None].long()
            != teacher_actions[:, None, :].long()
        ).to(pair_match_score.dtype)
        # Before a typed center becomes confident the slot falls back to its
        # learned spatial reference.  Do not impose a random hard class cost on
        # that fallback slot.  Once the dense proposal crosses its calibrated
        # confidence threshold, type identity participates in matching.
        r411_proposal_valid = aux.get("v552r411_proposal_valid")
        if not isinstance(r411_proposal_valid, torch.Tensor) or r411_proposal_valid.shape != (b, k):
            raise RuntimeError(
                "V552-R4.11 requires v552r411_proposal_valid with shape [B,K]; "
                f"got {None if not isinstance(r411_proposal_valid, torch.Tensor) else tuple(r411_proposal_valid.shape)}"
            )
        type_confidence = r411_proposal_valid.to(pair_match_score.dtype)[:, :, None]
        if not r418_enabled:
            pair_match_score = pair_match_score - (
                r411_match_type_weight * type_confidence * r411_type_pair_cost
            )
        else:
            # R4.18 identity is mask-defined.  Action/type is supervised only
            # after one-to-one mask assignment, preventing a wrong early type
            # prediction from choosing the component target.
            pair_match_score = pair_dice
    else:
        r411_type_pair_cost = pair_match_score.new_zeros(pair_match_score.shape)
    if tc_drcs:
        # TC-DRCS identity is defined only by final mask geometry.  Action,
        # center, utility and hard proposal ownership cannot choose a teacher.
        pair_match_score = pair_dice

    greedy_matched_target, greedy_matched = _greedy_unique_match(
        pair_match_score,
        teacher_valid,
        teacher_area,
        target_priority=(None if tc_drcs else teacher_utility),
    )
    optimal_matched_target, optimal_matched = _optimal_unique_match(
        pair_match_score,
        teacher_valid,
        teacher_area,
        target_priority=(None if tc_drcs else teacher_utility),
    )
    use_optimal_matching = bool(
        _m1(cfg, "V546_OPTIMAL_COMPONENT_MATCHING_ENABLED", False)
    )
    r4208_seed_locked_fraction = pair_match_score.new_zeros(())
    v564_anchor_inside_teacher_rate = pair_match_score.new_zeros(())
    v564_teacher_seed_coverage = pair_match_score.new_zeros(())
    v564_duplicate_owner_rate = pair_match_score.new_zeros(())
    v564_seed_without_teacher_rate = pair_match_score.new_zeros(())
    v564_unmatched_teacher_rate = pair_match_score.new_zeros(())
    v564_owner_raw = torch.full((b, k), -1, device=slot_masks.device, dtype=torch.long)
    if v564_rootfix or (clean_dynamic_component_set and not tc_drcs):
        proposal_anchor_v564 = aux.get("v562_proposal_anchor_xy")
        residual_logits_v564 = aux.get("v562_residual_logits")
        if not isinstance(proposal_anchor_v564, torch.Tensor) or proposal_anchor_v564.shape != (b, k, 2):
            raise RuntimeError("V564 requires proposal-owned anchors [B,K,2]")
        if not isinstance(residual_logits_v564, torch.Tensor) or residual_logits_v564.shape != (b, 1, h, w):
            raise RuntimeError("V564 requires residual proposal logits [B,1,H,W]")
        owner_score_logits = residual_logits_v564
        if v565_rootfix:
            owner_score_logits = aux.get("v565_seed_logits")
            if not isinstance(owner_score_logits, torch.Tensor) or owner_score_logits.shape != (b, 1, h, w):
                raise RuntimeError("V565 strict ownership requires center-head confidence")
        with torch.no_grad():
            anchor_px_v564 = (proposal_anchor_v564[:, :, 0] * float(w)).floor().long().clamp(0, w - 1)
            anchor_py_v564 = (proposal_anchor_v564[:, :, 1] * float(h)).floor().long().clamp(0, h - 1)
            batch_v564 = torch.arange(b, device=slot_masks.device)[:, None].expand(b, k)
            anchor_score_v564 = torch.sigmoid(
                owner_score_logits[:, 0][batch_v564, anchor_py_v564, anchor_px_v564]
            )
        matched_target, matched, v564_match_diag = _v564_strict_anchor_ownership_match(
            teacher_masks, teacher_valid, proposal_anchor_v564, anchor_score_v564,
            target_priority=teacher_utility,
        )
        v564_anchor_inside_teacher_rate = v564_match_diag["anchor_inside_teacher_rate"].detach()
        v564_teacher_seed_coverage = v564_match_diag["teacher_seed_coverage"].detach()
        v564_duplicate_owner_rate = v564_match_diag["duplicate_owner_rate"].detach()
        v564_seed_without_teacher_rate = v564_match_diag["seed_without_teacher_rate"].detach()
        v564_unmatched_teacher_rate = v564_match_diag["unmatched_teacher_rate"].detach()
        v564_owner_raw = v564_match_diag["owner_raw"]
    elif r4208_seed_matching:
        matched_target, matched, r4208_seed_locked = _r4208_seed_consistent_partial_match(
            pair_match_score,
            teacher_valid,
            teacher_area,
            r4208_seed_owner,
            r4208_seed_score,
            target_priority=teacher_utility,
        )
        r4208_seed_locked_fraction = (
            r4208_seed_locked.to(pair_match_score.dtype).sum()
            / teacher_valid.to(pair_match_score.dtype).sum().clamp_min(1.0)
        ).detach()
    elif tc_drcs or use_optimal_matching:
        matched_target, matched = optimal_matched_target, optimal_matched
    else:
        matched_target, matched = greedy_matched_target, greedy_matched
    r411_matched_type_accuracy = pair_match_score.new_zeros(())
    if r411_enabled and bool(matched.any().item()):
        matched_teacher_type = teacher_actions.gather(1, matched_target.clamp_min(0))
        r411_matched_type_accuracy = (
            r411_proposal_type.long().eq(matched_teacher_type.long())[matched].float().mean()
        ).detach()

    # V552-R4.14 extent supervision has its own identity assignment.  The
    # assignment is allowed to see only center and typed action identity; size
    # and mask quality are deliberately excluded so extent does not decide its
    # own target through the main candidate matching loop.
    r414_extent_matched_target = matched_target
    r414_extent_matched = matched
    r414_extent_teacher_geometry = teacher_geometry.gather(
        1, matched_target.clamp_min(0)[:, :, None].expand(b, k, 4)
    )
    r414_extent_match_rate = pair_match_score.new_zeros(())
    if r414_enabled:
        proposal_anchor_r414 = aux.get("v552r413_proposal_anchor_params")
        if not isinstance(proposal_anchor_r414, torch.Tensor) or proposal_anchor_r414.shape != r47_anchor_params.shape:
            raise RuntimeError(
                "V552-R4.14 requires v552r413_proposal_anchor_params aligned with Native atoms"
            )
        if not isinstance(r411_proposal_type, torch.Tensor) or r411_proposal_type.shape != (b, k):
            raise RuntimeError("V552-R4.14 requires typed proposal identities [B,K]")
        if not isinstance(r411_proposal_valid, torch.Tensor) or r411_proposal_valid.shape != (b, k):
            raise RuntimeError("V552-R4.14 requires typed proposal validity [B,K]")
        if r415_enabled:
            (
                r414_extent_matched_target,
                r414_extent_matched,
                r415_pair_center_distance_px,
                r415_pair_center_gate_px,
                r415_pair_type_mismatch,
            ) = _r415_identity_preserving_extent_match(
                proposal_anchor=proposal_anchor_r414,
                proposal_type=r411_proposal_type,
                proposal_valid=r411_proposal_valid,
                teacher_geometry=teacher_geometry,
                teacher_actions=teacher_actions,
                teacher_valid=teacher_valid,
                image_height=h,
                image_width=w,
                min_center_gate_px=float(_m1(cfg, "V552R415_MIN_CENTER_GATE_PX", 4.0)),
                max_center_gate_px=float(_m1(cfg, "V552R415_MAX_CENTER_GATE_PX", 12.0)),
                center_gate_diag_ratio=float(_m1(cfg, "V552R415_CENTER_GATE_DIAG_RATIO", 0.75)),
                type_mismatch_penalty=float(_m1(cfg, "V552R415_TYPE_MISMATCH_PENALTY", 0.35)),
            )
            if r417_enabled:
                # Spatial coverage ignores action type intentionally: this asks
                # whether location-first proposal extraction retained a usable
                # physical point for each Teacher before type binding quality.
                spatial_pair = (
                    (r411_proposal_valid[:, :, None] > 0.5)
                    & teacher_valid[:, None, :]
                    & (r415_pair_center_distance_px <= r415_pair_center_gate_px)
                )
                teacher_spatial_hit = spatial_pair.any(dim=1)
                r417_selected_spatial_coverage = (
                    teacher_spatial_hit[teacher_valid].float().mean().detach()
                    if bool(teacher_valid.any().item()) else slot_masks.new_zeros(())
                )
        else:
            r414_extent_matched_target, r414_extent_matched = _r414_center_type_extent_match(
                proposal_anchor=proposal_anchor_r414,
                proposal_type=r411_proposal_type,
                proposal_valid=r411_proposal_valid,
                teacher_geometry=teacher_geometry,
                teacher_actions=teacher_actions,
                teacher_valid=teacher_valid,
                teacher_area=teacher_area,
                teacher_utility=teacher_utility,
                type_weight=float(_m1(cfg, "V552R414_EXTENT_MATCH_TYPE_WEIGHT", 0.2)),
                require_type_match=bool(_m1(cfg, "V552R414_EXTENT_REQUIRE_TYPE_MATCH", True)),
            )
        safe_r414_extent = r414_extent_matched_target.clamp_min(0)
        r414_extent_teacher_geometry = teacher_geometry.gather(
            1, safe_r414_extent[:, :, None].expand(b, k, 4)
        )
        r414_extent_match_rate = (
            r414_extent_matched.float().sum() / teacher_valid.float().sum().clamp_min(1.0)
        ).detach()

    greedy_matching_dice = _matching_mean_dice(
        pair_dice.detach(), greedy_matched_target, greedy_matched
    )
    optimal_matching_dice = _matching_mean_dice(
        pair_dice.detach(), optimal_matched_target, optimal_matched
    )
    safe_target = matched_target.clamp_min(0)
    gathered_masks = teacher_masks.gather(
        1, safe_target[:, :, None, None].expand(b, k, h, w)
    )
    gathered_actions = teacher_actions.gather(1, safe_target)
    gathered_polarity = teacher_polarity.gather(1, safe_target)
    gathered_dose = best_teacher_dose.gather(1, safe_target)
    gathered_teacher_gain = best_teacher_action_gain.gather(1, safe_target)
    gathered_utility = teacher_utility.gather(1, safe_target)
    gathered_teacher_geometry = teacher_geometry.gather(
        1, safe_target[:, :, None].expand(b, k, 4)
    )
    gathered_masks = gathered_masks * matched[:, :, None, None].to(gathered_masks.dtype)
    utility_scale = max(float(_m1(cfg, "V541_M1_UTILITY_WEIGHT_SCALE", 2.0)), 0.0)
    replay_case_weight = min(
        max(float(_m1(cfg, "V544_REPLAY_CASE_WEIGHT", 1.0)), 0.0), 1.0
    )
    case_reliability = torch.where(
        replay_case[:, None],
        gathered_utility.new_full(gathered_utility.shape, replay_case_weight),
        torch.ones_like(gathered_utility),
    )
    matched_weight = (
        (1.0 + utility_scale * gathered_utility) * case_reliability
    ).detach()

    def weighted_matched_mean(value: torch.Tensor) -> torch.Tensor:
        if not bool(matched.any().item()):
            return value.sum() * 0.0
        weight = matched_weight[matched]
        return (value[matched] * weight).sum() / weight.sum().clamp_min(1.0)

    def equal_matched_mean(value: torch.Tensor) -> torch.Tensor:
        """Equal component mean used by V560 physical set supervision.

        A component's geometry is not a utility-weighted quantity.  Utility is
        deliberately delegated to M2, so every matched Teacher component owns
        one equal geometry target here.
        """
        if not bool(matched.any().item()):
            return value.sum() * 0.0
        return value[matched].mean()

    def r414_extent_mean(value: torch.Tensor) -> torch.Tensor:
        # Equal component weighting is intentional here.  Extent is a physical
        # geometry target; its supervision must not be weakened by downstream
        # utility calibration or by the main candidate matching weights.
        if not bool(r414_extent_matched.any().item()):
            return value.sum() * 0.0
        return value[r414_extent_matched].mean()

    # V544: an explicit positive/negative region mean is invariant to component
    # area.  Unlike a capped pos_weight, a 0.01% component and a 10% component
    # both contribute one half of the BCE.  The old V543B loss remains available
    # as an exact ablation through the config flag.
    mask_bce_element = F.binary_cross_entropy_with_logits(
        slot_mask_logits, gathered_masks, reduction="none"
    )
    positive_pixels = gathered_masks.sum(dim=(-2, -1))
    total_pixels = float(gathered_masks.shape[-2] * gathered_masks.shape[-1])
    negative_pixels = total_pixels - positive_pixels
    mask_pos_weight = (
        negative_pixels / positive_pixels.clamp_min(1.0)
    ).clamp(
        1.0,
        float(_m1(cfg, "V543_MASK_POS_WEIGHT_CAP", 64.0)),
    )
    use_region_balanced_mask = tc_drcs or bool(
        _m1(cfg, "V544_REGION_BALANCED_MASK_ENABLED", False)
    )
    if use_region_balanced_mask:
        positive_mask = gathered_masks
        negative_mask = 1.0 - gathered_masks
        positive_bce = (
            mask_bce_element * positive_mask
        ).sum(dim=(-2, -1)) / positive_pixels.clamp_min(1.0)
        negative_bce = (
            mask_bce_element * negative_mask
        ).sum(dim=(-2, -1)) / negative_pixels.clamp_min(1.0)
        mask_bce_map = 0.5 * (positive_bce + negative_bce)
    else:
        mask_pixel_weight = torch.where(
            gathered_masks > 0.5,
            mask_pos_weight[:, :, None, None],
            torch.ones_like(gathered_masks),
        )
        mask_bce_map = (
            mask_bce_element * mask_pixel_weight
        ).sum(dim=(-2, -1)) / mask_pixel_weight.sum(dim=(-2, -1)).clamp_min(1.0)

    pred_target_intersection = (slot_masks * gathered_masks).sum(dim=(-2, -1))
    predicted_area = slot_masks.sum(dim=(-2, -1))
    target_area = gathered_masks.sum(dim=(-2, -1))
    pred_target_denominator = predicted_area + target_area
    mask_dice_loss_map = 1.0 - (
        2.0 * pred_target_intersection + EPS
    ) / (pred_target_denominator + EPS)
    mask_soft_purity_map = (
        pred_target_intersection + EPS
    ) / (predicted_area + EPS)
    mask_soft_coverage_map = (
        pred_target_intersection + EPS
    ) / (target_area + EPS)
    v560_standard_mask_bce = slot_masks.sum() * 0.0
    v560_standard_mask_dice_loss = slot_masks.sum() * 0.0
    v563_local_mask_bce = slot_masks.sum() * 0.0
    v563_local_mask_dice_loss = slot_masks.sum() * 0.0
    v563_outside_mask_loss = slot_masks.sum() * 0.0
    v563_local_target_coverage = slot_masks.new_zeros(())
    if v560_clean_core:
        if clean_dynamic_component_set:
            if tc_drcs:
                # TC-DRCS makes sparse positive and negative regions contribute
                # equally, removing the all-background BCE shortcut without a
                # tuned positive-class weight.
                v560_standard_mask_bce = equal_matched_mean(mask_bce_map)
            else:
                # Legacy CLEAN retained for exact reproducibility.
                clean_mask_bce_map = mask_bce_element.mean(dim=(-2, -1))
                v560_standard_mask_bce = equal_matched_mean(clean_mask_bce_map)
            v560_standard_mask_dice_loss = equal_matched_mean(mask_dice_loss_map)
        elif v563_rootfix:
            # V563 no longer gives every positive residual pixel half of the
            # *whole-image* BCE mass.  Geometry is supervised only inside the
            # query's predicted local window, where the positive/negative ratio
            # is naturally well-conditioned.  This keeps the anti-collapse Dice
            # term while removing the V562 incentive to inflate masks globally.
            local_window_v563 = aux.get("v563_mask_window")
            raw_mask_probability_v563 = aux.get("v563_raw_mask_probability")
            if (
                not isinstance(local_window_v563, torch.Tensor)
                or local_window_v563.shape != slot_masks.shape
            ):
                raise RuntimeError(
                    "V563 requires GT-free v563_mask_window with shape "
                    f"{tuple(slot_masks.shape)}"
                )
            if (
                not isinstance(raw_mask_probability_v563, torch.Tensor)
                or raw_mask_probability_v563.shape != slot_masks.shape
            ):
                raise RuntimeError(
                    "V563 requires v563_raw_mask_probability before hard extent gating"
                )
            local_window_v563 = local_window_v563.to(slot_masks.dtype)
            local_pixels_v563 = local_window_v563.sum(dim=(-2, -1)).clamp_min(1.0)
            v563_bce_map = (
                mask_bce_element * local_window_v563
            ).sum(dim=(-2, -1)) / local_pixels_v563
            v563_local_mask_bce = equal_matched_mean(v563_bce_map)

            local_target_v563 = gathered_masks * local_window_v563
            local_intersection_v563 = (slot_masks * local_target_v563).sum(dim=(-2, -1))
            local_pred_area_v563 = (slot_masks * local_window_v563).sum(dim=(-2, -1))
            local_target_area_v563 = local_target_v563.sum(dim=(-2, -1))
            local_dice_v563 = (
                2.0 * local_intersection_v563 + EPS
            ) / (local_pred_area_v563 + local_target_area_v563 + EPS)
            v563_local_mask_dice_loss = equal_matched_mean(1.0 - local_dice_v563)
            v560_standard_mask_bce = v563_local_mask_bce
            v560_standard_mask_dice_loss = v563_local_mask_dice_loss

            # Standard L1/BCE-style outside suppression on the *raw*, pre-gate
            # mask probability teaches the representation itself to stay local;
            # the hard forward gate is therefore a safety contract, not a crutch.
            outside_v563 = 1.0 - local_window_v563
            outside_pixels_v563 = outside_v563.sum(dim=(-2, -1)).clamp_min(1.0)
            outside_map_v563 = (
                raw_mask_probability_v563 * outside_v563
            ).sum(dim=(-2, -1)) / outside_pixels_v563
            v563_outside_mask_loss = equal_matched_mean(outside_map_v563)
            teacher_area_full_v563 = gathered_masks.sum(dim=(-2, -1)).clamp_min(1.0)
            teacher_area_local_v563 = local_target_v563.sum(dim=(-2, -1))
            v563_local_target_coverage = equal_matched_mean(
                teacher_area_local_v563 / teacher_area_full_v563
            ).detach()
        elif v562_rootfix:
            # Sparse-component root fix: equal positive/negative BCE mass per
            # matched component prevents the background-only shortcut.
            positive_mask_v562 = gathered_masks
            negative_mask_v562 = 1.0 - gathered_masks
            positive_pixels_v562 = positive_mask_v562.sum(dim=(-2, -1))
            negative_pixels_v562 = negative_mask_v562.sum(dim=(-2, -1))
            positive_bce_v562 = (
                mask_bce_element * positive_mask_v562
            ).sum(dim=(-2, -1)) / positive_pixels_v562.clamp_min(1.0)
            negative_bce_v562 = (
                mask_bce_element * negative_mask_v562
            ).sum(dim=(-2, -1)) / negative_pixels_v562.clamp_min(1.0)
            balanced_bce_map_v562 = 0.5 * (positive_bce_v562 + negative_bce_v562)
            v560_standard_mask_bce = equal_matched_mean(balanced_bce_map_v562)
        else:
            # Historical V560/V561 objective retained for reproducibility.
            v560_bce_map = F.binary_cross_entropy_with_logits(
                slot_mask_logits, gathered_masks, reduction="none"
            ).mean(dim=(-2, -1))
            v560_standard_mask_bce = equal_matched_mean(v560_bce_map)
        if not v563_rootfix:
            if not clean_dynamic_component_set:
                v560_standard_mask_dice_loss = equal_matched_mean(mask_dice_loss_map)
        global_mask_loss = v560_standard_mask_bce + v560_standard_mask_dice_loss
        if v563_rootfix:
            global_mask_loss = global_mask_loss + float(
                _m1(cfg, "V563_OUTSIDE_MASK_WEIGHT", 1.0)
            ) * v563_outside_mask_loss
        mask_soft_purity = equal_matched_mean(mask_soft_purity_map)
        mask_soft_coverage = equal_matched_mean(mask_soft_coverage_map)
    else:
        global_mask_loss = weighted_matched_mean(mask_bce_map + mask_dice_loss_map)
        mask_soft_purity = weighted_matched_mean(mask_soft_purity_map)
        mask_soft_coverage = weighted_matched_mean(mask_soft_coverage_map)

    # V562 matched-local execution closure.  Supervise the actual candidate on
    # its assigned factual residual and on a narrow safe ring where Base is
    # already correct.  This avoids reintroducing whole-image Base-copy loss.
    v562_local_execution_component_bce = slot_masks.sum() * 0.0
    v562_local_execution_ring_bce = slot_masks.sum() * 0.0
    v562_local_execution_loss = slot_masks.sum() * 0.0
    v562_matched_add_flip_rate = slot_masks.new_zeros(())
    v562_matched_remove_flip_rate = slot_masks.new_zeros(())
    v564_infeasible_teacher_fraction = slot_masks.new_zeros(())
    if v564_rootfix:
        legal_support_diag_v564 = aux.get("v563_mask_window")
        if isinstance(legal_support_diag_v564, torch.Tensor) and legal_support_diag_v564.shape == gathered_masks.shape:
            full_area_v564 = gathered_masks.sum(dim=(-2, -1)).clamp_min(1.0)
            local_area_v564 = (gathered_masks * legal_support_diag_v564).sum(dim=(-2, -1))
            v564_infeasible_teacher_fraction = equal_matched_mean(
                1.0 - local_area_v564 / full_area_v564
            ).detach()
    if v562_rootfix:
        candidate_v562 = exact_slot_candidates_st.clamp(EPS, 1.0 - EPS)
        gt_slot_v562 = gt[:, 0][:, None].expand_as(candidate_v562)
        component_bce_pixel_v562 = F.binary_cross_entropy(
            candidate_v562, gt_slot_v562, reduction="none"
        )
        execution_teacher_v562 = gathered_masks
        if v564_rootfix:
            legal_support_v564 = aux.get("v563_mask_window")
            if not isinstance(legal_support_v564, torch.Tensor) or legal_support_v564.shape != gathered_masks.shape:
                raise RuntimeError("V564 requires legal support for feasible execution supervision")
            execution_teacher_v562 = gathered_masks * legal_support_v564.to(gathered_masks.dtype)
        component_area_v562 = execution_teacher_v562.sum(dim=(-2, -1)).clamp_min(1.0)
        component_bce_map_v562 = (
            component_bce_pixel_v562 * execution_teacher_v562
        ).sum(dim=(-2, -1)) / component_area_v562
        v562_local_execution_component_bce = equal_matched_mean(component_bce_map_v562)

        flat_teacher_v562 = execution_teacher_v562.reshape(b * k, 1, h, w)
        dilated_teacher_v562 = F.max_pool2d(
            flat_teacher_v562, 5, stride=1, padding=2
        ).reshape(b, k, h, w)
        ring_v562 = (dilated_teacher_v562 - execution_teacher_v562).clamp(0.0, 1.0)
        base_correct_v562 = (
            (base[:, 0][:, None] >= 0.5) == (gt[:, 0][:, None] >= 0.5)
        ).to(slot_masks.dtype)
        safe_ring_v562 = ring_v562 * base_correct_v562
        base_slot_v562 = base[:, 0][:, None].expand_as(candidate_v562).detach()
        ring_bce_pixel_v562 = F.binary_cross_entropy(
            candidate_v562, base_slot_v562, reduction="none"
        )
        ring_area_v562 = safe_ring_v562.sum(dim=(-2, -1)).clamp_min(1.0)
        ring_bce_map_v562 = (
            ring_bce_pixel_v562 * safe_ring_v562
        ).sum(dim=(-2, -1)) / ring_area_v562
        v562_local_execution_ring_bce = equal_matched_mean(ring_bce_map_v562)
        v562_local_execution_loss = (
            v562_local_execution_component_bce
            + float(_m1(cfg, "V562_SAFE_RING_WEIGHT", 0.25)) * v562_local_execution_ring_bce
        )

        with torch.no_grad():
            pred_binary_v562 = candidate_v562 >= 0.5
            component_bool_v562 = gathered_masks > 0.5
            matched_expand_v562 = matched[:, :, None, None]
            add_slot_v562 = (gathered_actions % 2 == 1)[:, :, None, None]
            add_pixels_v562 = component_bool_v562 & matched_expand_v562 & add_slot_v562
            remove_pixels_v562 = component_bool_v562 & matched_expand_v562 & (~add_slot_v562)
            if bool(add_pixels_v562.any().item()):
                v562_matched_add_flip_rate = pred_binary_v562[add_pixels_v562].float().mean()
            if bool(remove_pixels_v562.any().item()):
                v562_matched_remove_flip_rate = (~pred_binary_v562[remove_pixels_v562]).float().mean()

    if r4206_enabled:
        conditional_logits = aux.get("v552r4205_conditional_identity_logits")
        expected_conditional_shape = (b, k + 1, h, w)
        if (
            not isinstance(conditional_logits, torch.Tensor)
            or conditional_logits.shape != expected_conditional_shape
        ):
            raise RuntimeError(
                "V552-R4.20.6 requires v552r4205_conditional_identity_logits "
                f"with shape {expected_conditional_shape}, got "
                + str(
                    None
                    if not isinstance(conditional_logits, torch.Tensor)
                    else tuple(conditional_logits.shape)
                )
            )
        if not bool(torch.isfinite(conditional_logits).all().item()):
            raise FloatingPointError(
                "V552-R4.20.6 received non-finite conditional identity logits"
            )

        # ``gathered_masks`` is already permutation-aligned by the existing
        # globally optimal one-to-one set matching.  Connected residual
        # components are disjoint, so every residual pixel has exactly one
        # categorical target: its matched editable slot, or K=overflow when no
        # matched editable Teacher owns that pixel.  Background is ignored.
        with torch.no_grad():
            residual_pixel = occupancy_target[:, 0] > 0.5
            aligned_teacher = gathered_masks.detach() > 0.5
            aligned_count = aligned_teacher.sum(dim=1)
            aligned_editable_pixel = aligned_count > 0
            overflow_pixel = residual_pixel & (~aligned_editable_pixel)

            conditional_target = torch.full(
                (b, h, w), -100, device=slot_masks.device, dtype=torch.long
            )
            aligned_slot_index = aligned_teacher.to(torch.long).argmax(dim=1)
            conditional_target[aligned_editable_pixel] = aligned_slot_index[
                aligned_editable_pixel
            ]
            conditional_target[overflow_pixel] = k

            # Keep the *raw* residual count separate from the safe denominator.
            # An empty-residual batch has no conditional labels to predict; by
            # definition its supervision coverage is vacuously complete (1.0),
            # not 0.0.  The previous clamp-before-ratio implementation turned
            # 0 / clamp(0, 1) into 0 and made epoch averaging falsely report
            # incomplete supervision whenever even one batch contained no Base
            # error pixels.  That was a diagnostic/contract bug, not a target
            # decomposition failure.
            residual_count_raw = residual_pixel.sum()
            residual_count_safe = residual_count_raw.clamp_min(1)
            supervised_count = (conditional_target >= 0).sum()
            overflow_count = overflow_pixel.sum()
            reconstructed_residual = aligned_editable_pixel | overflow_pixel
            r4206_target_decomposition_error = (
                reconstructed_residual.to(slot_masks.dtype)
                - residual_pixel.to(slot_masks.dtype)
            ).abs().sum() / residual_count_safe.to(slot_masks.dtype)
            r4206_teacher_overlap_rate = (
                ((aligned_count > 1) & residual_pixel).to(slot_masks.dtype).sum()
                / residual_count_safe.to(slot_masks.dtype)
            )
            r4206_overflow_target_residual_fraction = torch.where(
                residual_count_raw > 0,
                overflow_count.to(slot_masks.dtype)
                / residual_count_safe.to(slot_masks.dtype),
                slot_masks.new_zeros(()),
            )
            r4206_supervised_residual_fraction = torch.where(
                residual_count_raw > 0,
                supervised_count.to(slot_masks.dtype)
                / residual_count_safe.to(slot_masks.dtype),
                slot_masks.new_ones(()),
            )
            # Exact count-based companion contract.  Unlike a batch-average
            # ratio, this remains unambiguous in the presence of empty-residual
            # batches: any genuinely unlabelled residual pixel makes it > 0.
            r4206_unsupervised_residual_pixels = (
                residual_count_raw - supervised_count
            ).clamp_min(0).to(slot_masks.dtype)
            r4206_empty_residual_batch = (
                residual_count_raw == 0
            ).to(slot_masks.dtype)

        valid_conditional = conditional_target >= 0
        if bool(valid_conditional.any().item()):
            logits_pixel = conditional_logits.permute(0, 2, 3, 1)
            r4206_conditional_identity_loss = F.cross_entropy(
                logits_pixel[valid_conditional],
                conditional_target[valid_conditional],
            )
            conditional_probability = torch.softmax(conditional_logits, dim=1)
            conditional_prediction = conditional_logits.argmax(dim=1)
            correct = conditional_prediction.eq(conditional_target)
            r4206_conditional_accuracy = correct[valid_conditional].float().mean().detach()
            editable_target = valid_conditional & (conditional_target < k)
            overflow_target_bool = valid_conditional & (conditional_target == k)
            predicted_overflow = valid_conditional & (conditional_prediction == k)
            r4206_editable_accuracy = torch.where(
                editable_target.any(),
                correct[editable_target].float().mean(),
                correct.new_zeros((), dtype=slot_masks.dtype),
            ).detach()
            r4206_overflow_recall = torch.where(
                overflow_target_bool.any(),
                (conditional_prediction[overflow_target_bool] == k).float().mean(),
                slot_masks.new_zeros(()),
            ).detach()
            r4206_overflow_precision = torch.where(
                predicted_overflow.any(),
                (conditional_target[predicted_overflow] == k).float().mean(),
                slot_masks.new_zeros(()),
            ).detach()
            r4206_overflow_predicted_residual_fraction = (
                predicted_overflow.to(slot_masks.dtype).sum()
                / valid_conditional.to(slot_masks.dtype).sum().clamp_min(1.0)
            ).detach()
            safe_conditional_target = conditional_target.clamp_min(0)
            true_class_probability = conditional_probability.gather(
                1, safe_conditional_target[:, None]
            )[:, 0]
            r4206_true_class_probability = (
                true_class_probability[valid_conditional].mean().detach()
            )
        else:
            r4206_conditional_identity_loss = conditional_logits.sum() * 0.0

        # Shape supervision remains a standard matched soft-Dice term.  The
        # previous independent per-slot BCE is deliberately not optimized in
        # R4.20.6 because categorical CE now owns component/overflow identity.
        r4206_shape_dice_loss = weighted_matched_mean(mask_dice_loss_map)

    r412_native_canonical_loss = slot_masks.sum() * 0.0
    r412_native_canonical_dice = slot_masks.new_zeros(())
    r412_native_canonical_purity = slot_masks.new_zeros(())
    if r412_enabled:
        canonical_size = max(int(_m1(cfg, "V552R412_CANONICAL_ROI_SIZE", 64)), 32)
        # Stop shape loss from moving boxes to hide shape errors. Geometry is
        # trained explicitly below; canonical shape is trained in its own frame.
        canonical_pred_logits = _r412_canonical_crop(
            slot_mask_logits, r47_anchor_params.detach(), canonical_size
        )
        canonical_target = _r412_canonical_crop(
            gathered_masks, gathered_teacher_geometry.detach(), canonical_size
        ).clamp(0.0, 1.0)
        canonical_map, canonical_dice_map, canonical_purity_map, _ = _r412_balanced_mask_map(
            canonical_pred_logits, canonical_target
        )
        r412_native_canonical_loss = weighted_matched_mean(canonical_map)
        r412_native_canonical_dice = weighted_matched_mean(canonical_dice_map).detach()
        r412_native_canonical_purity = weighted_matched_mean(canonical_purity_map).detach()
        mask_loss = (
            float(_m1(cfg, "V552R412_CANONICAL_MASK_WEIGHT", 1.0)) * r412_native_canonical_loss
            + float(_m1(cfg, "V552R412_GLOBAL_MASK_WEIGHT", 0.20)) * global_mask_loss
        )
    else:
        mask_loss = global_mask_loss

    tc_stage0_mask_loss = slot_masks.sum() * 0.0
    tc_pilot_mask_loss = slot_masks.sum() * 0.0
    tc_pilot_action_loss = slot_masks.sum() * 0.0
    tc_teacher_supervision_coverage = slot_masks.new_ones(())
    tc_retained_teacher_count = teacher_valid.to(slot_masks.dtype).sum(dim=1).mean().detach()
    tc_teacher_raw_count_mean = teacher_raw_component_count.mean().detach()
    tc_teacher_overflow_rate = (teacher_raw_component_count > float(k)).to(slot_masks.dtype).mean().detach()
    tc_pilot_teacher_consistency = slot_masks.new_ones(())
    if tc_drcs:
        def _tc_balanced_mask_map(logits_tc: torch.Tensor, target_tc: torch.Tensor):
            bce_tc = F.binary_cross_entropy_with_logits(logits_tc, target_tc, reduction="none")
            pos_tc = target_tc.sum(dim=(-2, -1))
            neg_tc = (1.0 - target_tc).sum(dim=(-2, -1))
            pos_bce_tc = (bce_tc * target_tc).sum(dim=(-2, -1)) / pos_tc.clamp_min(1.0)
            neg_bce_tc = (bce_tc * (1.0 - target_tc)).sum(dim=(-2, -1)) / neg_tc.clamp_min(1.0)
            balanced_tc = torch.where(
                pos_tc > 0, 0.5 * (pos_bce_tc + neg_bce_tc), neg_bce_tc
            )
            prob_tc = torch.sigmoid(logits_tc)
            inter_tc = (prob_tc * target_tc).sum(dim=(-2, -1))
            dice_tc = 1.0 - (2.0 * inter_tc + EPS) / (
                prob_tc.sum(dim=(-2, -1)) + target_tc.sum(dim=(-2, -1)) + EPS
            )
            return balanced_tc + dice_tc, balanced_tc, dice_tc

        stage0_logits_tc = aux.get("tc_stage0_logits")
        if not isinstance(stage0_logits_tc, torch.Tensor) or stage0_logits_tc.shape != slot_mask_logits.shape:
            raise RuntimeError("TC-DRCS requires attached tc_stage0_logits [B,K,H,W]")
        stage0_map_tc, _, _ = _tc_balanced_mask_map(stage0_logits_tc, gathered_masks)

        pilot_logits_tc = aux.get("tc_pilot_logits")
        pilot_valid_tc = aux.get("tc_pilot_valid")
        pilot_teacher_masks_tc = aux.get("tc_pilot_teacher_masks")
        pilot_teacher_actions_tc = aux.get("tc_pilot_teacher_actions")
        pilot_action_logits_tc = aux.get("tc_pilot_action_logits")
        if not isinstance(pilot_logits_tc, torch.Tensor) or pilot_logits_tc.shape != slot_mask_logits.shape:
            raise RuntimeError("TC-DRCS requires tc_pilot_logits [B,K,H,W]")
        if not isinstance(pilot_valid_tc, torch.Tensor) or pilot_valid_tc.shape != teacher_valid.shape:
            raise RuntimeError("TC-DRCS requires tc_pilot_valid [B,K]")
        if not isinstance(pilot_teacher_masks_tc, torch.Tensor) or pilot_teacher_masks_tc.shape != teacher_masks.shape:
            raise RuntimeError("TC-DRCS requires tc_pilot_teacher_masks [B,K,H,W]")
        if not isinstance(pilot_teacher_actions_tc, torch.Tensor) or pilot_teacher_actions_tc.shape != teacher_actions.shape:
            raise RuntimeError("TC-DRCS requires tc_pilot_teacher_actions [B,K]")
        if not isinstance(pilot_action_logits_tc, torch.Tensor) or pilot_action_logits_tc.shape != action_logits.shape:
            raise RuntimeError("TC-DRCS requires tc_pilot_action_logits [B,K,4]")
        pilot_valid_tc = pilot_valid_tc.bool()
        # Pilot and loss-side teacher banks must be identical; otherwise the
        # training-only localization guide would optimize a different task.
        with torch.no_grad():
            valid_equal = torch.equal(pilot_valid_tc, teacher_valid)
            mask_equal = torch.equal(
                pilot_teacher_masks_tc.detach().to(teacher_masks.dtype), teacher_masks.detach()
            )
            action_equal = torch.equal(
                pilot_teacher_actions_tc.detach().long(), heuristic_teacher_actions.detach().long()
            )
        if not (valid_equal and mask_equal and action_equal):
            raise RuntimeError("TC-DRCS pilot teacher bank diverged from loss-side factual teacher bank")
        pilot_map_tc, _, _ = _tc_balanced_mask_map(
            pilot_logits_tc, pilot_teacher_masks_tc.to(pilot_logits_tc.dtype)
        )

        main_final_map_tc = mask_bce_map + mask_dice_loss_map
        mask_instances = []
        if bool(matched.any().item()):
            mask_instances.extend([main_final_map_tc[matched], stage0_map_tc[matched]])
            tc_stage0_mask_loss = stage0_map_tc[matched].mean()
        if bool(pilot_valid_tc.any().item()):
            mask_instances.append(pilot_map_tc[pilot_valid_tc])
            tc_pilot_mask_loss = pilot_map_tc[pilot_valid_tc].mean()
        if mask_instances:
            mask_loss = torch.cat([x.reshape(-1) for x in mask_instances], dim=0).mean()
        else:
            mask_loss = slot_mask_logits.sum() * 0.0

        retained_teacher_total = teacher_valid.to(slot_masks.dtype).sum()
        matched_teacher_total = matched.to(slot_masks.dtype).sum()
        tc_teacher_supervision_coverage = torch.where(
            retained_teacher_total > 0,
            matched_teacher_total / retained_teacher_total.clamp_min(1.0),
            retained_teacher_total.new_ones(()),
        ).detach()
        # With N<=K and exact bipartite matching, retained-teacher supervision
        # completeness is a code invariant, not a soft readiness threshold.
        if abs(float(tc_teacher_supervision_coverage.item()) - 1.0) > 1.0e-7:
            raise RuntimeError(
                f"TC-DRCS teacher supervision incomplete: {float(tc_teacher_supervision_coverage):.8f}"
            )
        tc_pilot_teacher_consistency = slot_masks.new_ones(())

    if r4206_enabled:
        # Replace (do not stack on top of) the legacy independent mask BCE.
        # This preserves the existing outer V538_COMPONENT_MASK_WEIGHT and
        # introduces no new relative loss-weight hyperparameter.
        mask_loss = r4206_conditional_identity_loss + r4206_shape_dice_loss

    # V552-R4.18 paired stable residual-mask supervision.  Native set loss and
    # paired set loss are averaged, not stacked with an arbitrary new weight.
    # The auxiliary coarse mask itself is the input that generated its target.
    r418_paired_mask_loss = slot_masks.sum() * 0.0
    r418_paired_mask_dice = slot_masks.new_zeros(())
    r418_paired_valid_fraction = slot_masks.new_zeros(())
    r418_paired_target_consistency = slot_masks.new_zeros(())
    r418_paired_error_coverage = slot_masks.new_zeros(())
    if r418_paired_enabled:
        paired_logits = aux.get("v552r418_paired_mask_logits")
        paired_target = aux.get("v552r418_paired_teacher_masks")
        paired_valid = aux.get("v552r418_paired_teacher_valid")
        paired_coarse = aux.get("v552r418_paired_coarse")
        if not isinstance(paired_logits, torch.Tensor) or paired_logits.shape != slot_mask_logits.shape:
            raise RuntimeError("V552-R4.18 requires paired mask logits aligned [B,K,H,W]")
        if not isinstance(paired_target, torch.Tensor) or paired_target.shape != slot_masks.shape:
            raise RuntimeError("V552-R4.18 requires paired teacher masks aligned [B,K,H,W]")
        if not isinstance(paired_valid, torch.Tensor) or paired_valid.shape != (b, k):
            raise RuntimeError("V552-R4.18 requires paired teacher validity [B,K]")
        if not isinstance(paired_coarse, torch.Tensor) or paired_coarse.shape[-2:] != (h, w):
            raise RuntimeError("V552-R4.18 requires the exact paired coarse input")
        paired_target = paired_target.to(slot_masks.dtype)
        paired_valid = paired_valid.bool()
        paired_elem = F.binary_cross_entropy_with_logits(
            paired_logits, paired_target, reduction="none"
        )
        paired_pos = paired_target.sum(dim=(-2, -1))
        paired_total = float(h * w)
        paired_neg = paired_total - paired_pos
        paired_pos_bce = (paired_elem * paired_target).sum(dim=(-2, -1)) / paired_pos.clamp_min(1.0)
        paired_neg_bce = (paired_elem * (1.0 - paired_target)).sum(dim=(-2, -1)) / paired_neg.clamp_min(1.0)
        paired_bce = 0.5 * (paired_pos_bce + paired_neg_bce)
        paired_prob = torch.sigmoid(paired_logits)
        paired_inter = (paired_prob * paired_target).sum(dim=(-2, -1))
        paired_dice = (2.0 * paired_inter + EPS) / (
            paired_prob.sum(dim=(-2, -1)) + paired_target.sum(dim=(-2, -1)) + EPS
        )
        paired_loss_map = paired_bce + (1.0 - paired_dice)
        if bool(paired_valid.any().item()):
            r418_paired_mask_loss = paired_loss_map[paired_valid].mean()
            r418_paired_mask_dice = paired_dice[paired_valid].mean().detach()
            r418_paired_valid_fraction = paired_valid.float().mean().detach()
            mask_loss = 0.5 * (mask_loss + r418_paired_mask_loss)
        paired_union = (paired_target * paired_valid[:, :, None, None].to(paired_target.dtype)).amax(dim=1, keepdim=True)
        paired_error = (
            (paired_coarse[:, :1].detach() >= 0.5)
            != (gt >= 0.5)
        ).to(paired_target.dtype)
        represented = paired_union.sum().clamp_min(1.0)
        outside_error = (paired_union * (1.0 - paired_error)).sum()
        r418_paired_target_consistency = (1.0 - outside_error / represented).clamp(0.0, 1.0).detach()
        r418_paired_error_coverage = (
            (paired_union * paired_error).sum() / paired_error.sum().clamp_min(1.0)
        ).clamp(0.0, 1.0).detach()

    r47_anchor_reg_loss = slot_masks.sum() * 0.0
    r47_anchor_l1 = slot_masks.new_zeros(())
    r412_box_iou = slot_masks.new_zeros(())
    r412_box_giou_loss = slot_masks.sum() * 0.0
    r412_width_mae = slot_masks.new_zeros(())
    r412_height_mae = slot_masks.new_zeros(())
    r413_proposal_box_iou = slot_masks.new_zeros(())
    r413_box_drift_l1 = slot_masks.new_zeros(())
    r413_box_drift_center = slot_masks.new_zeros(())
    r413_box_drift_size = slot_masks.new_zeros(())
    r413_log_size_loss = slot_masks.sum() * 0.0
    r414_oracle_center_pred_size_iou = slot_masks.new_zeros(())
    r414_pred_center_oracle_size_iou = slot_masks.new_zeros(())
    r414_center_error_px_mean = slot_masks.new_zeros(())
    r414_center_error_px_median = slot_masks.new_zeros(())
    r414_center_error_px_p90 = slot_masks.new_zeros(())
    r415_identity_match_rate = slot_masks.new_zeros(())
    r415_same_type_match_rate = slot_masks.new_zeros(())
    r415_tight_center_match_rate = slot_masks.new_zeros(())
    r415_center_error_px_mean = slot_masks.new_zeros(())
    r415_center_error_px_median = slot_masks.new_zeros(())
    r415_center_error_px_p90 = slot_masks.new_zeros(())
    r415_center_gate_utilization_mean = slot_masks.new_zeros(())
    r415_unmatched_teacher_rate = slot_masks.new_zeros(())
    r415_unmatched_proposal_rate = slot_masks.new_zeros(())
    r415_oracle_center_pred_size_iou = slot_masks.new_zeros(())
    r415_pred_center_oracle_size_iou = slot_masks.new_zeros(())
    r416_edge_offset_loss = slot_masks.sum() * 0.0
    r416_edge_offset_mae_px = slot_masks.new_zeros(())
    r416_post_topk_identity_coverage = slot_masks.new_zeros(())
    r416_legacy_topk_unique_fraction = aux.get(
        "v552r416_legacy_topk_unique_fraction", slot_masks.new_zeros(())
    ).to(slot_masks.dtype)
    r47_point_mask_loss = slot_masks.sum() * 0.0
    r47_point_bce = slot_masks.new_zeros(())
    r47_point_dice = slot_masks.new_zeros(())
    if r47_enabled:
        # R4.14 geometry losses use the center+type-only extent assignment.
        # Historical protocols retain the main candidate matching target.
        geometry_target = (
            r414_extent_teacher_geometry.detach() if r414_enabled
            else gathered_teacher_geometry.detach()
        )
        geometry_mean = r414_extent_mean if r414_enabled else weighted_matched_mean

        anchor_l1_map = F.smooth_l1_loss(
            r47_anchor_params, geometry_target, reduction="none", beta=0.05
        ).mean(dim=2)
        r47_anchor_reg_loss = geometry_mean(anchor_l1_map)
        r47_anchor_l1 = geometry_mean(
            (r47_anchor_params - geometry_target).abs().mean(dim=2)
        )

        if r412_enabled:
            aligned_iou, aligned_giou = _r412_aligned_box_iou_giou(
                r47_anchor_params, geometry_target
            )
            r412_box_iou = geometry_mean(aligned_iou).detach()
            r412_box_giou_loss = geometry_mean(1.0 - aligned_giou)
            r412_width_mae = geometry_mean(
                (r47_anchor_params[:, :, 2] - geometry_target[:, :, 2]).abs()
            ).detach()
            r412_height_mae = geometry_mean(
                (r47_anchor_params[:, :, 3] - geometry_target[:, :, 3]).abs()
            ).detach()

            if r413_enabled:
                proposal_anchor = aux.get("v552r413_proposal_anchor_params")
                if not isinstance(proposal_anchor, torch.Tensor) or proposal_anchor.shape != r47_anchor_params.shape:
                    raise RuntimeError(
                        "V552-R4.13 requires v552r413_proposal_anchor_params aligned with Native atoms"
                    )
                proposal_iou, _ = _r412_aligned_box_iou_giou(
                    proposal_anchor, geometry_target
                )
                r413_proposal_box_iou = geometry_mean(proposal_iou).detach()
                drift = (r47_anchor_params - proposal_anchor).abs()
                r413_box_drift_l1 = geometry_mean(drift.mean(dim=2)).detach()
                r413_box_drift_center = geometry_mean(drift[:, :, :2].mean(dim=2)).detach()
                r413_box_drift_size = geometry_mean(drift[:, :, 2:].mean(dim=2)).detach()
                pred_log_size = torch.log(r47_anchor_params[:, :, 2:].clamp_min(1.0e-6))
                target_log_size = torch.log(geometry_target[:, :, 2:].clamp_min(1.0e-6))
                log_size_map = F.smooth_l1_loss(
                    pred_log_size, target_log_size, reduction="none", beta=0.10
                ).mean(dim=2)
                r413_log_size_loss = geometry_mean(log_size_map)

                if r414_enabled:
                    oracle_center_pred_size = torch.cat(
                        [geometry_target[:, :, :2], proposal_anchor[:, :, 2:]], dim=2
                    )
                    pred_center_oracle_size = torch.cat(
                        [proposal_anchor[:, :, :2], geometry_target[:, :, 2:]], dim=2
                    )
                    ocps_iou, _ = _r412_aligned_box_iou_giou(
                        oracle_center_pred_size, geometry_target
                    )
                    pcos_iou, _ = _r412_aligned_box_iou_giou(
                        pred_center_oracle_size, geometry_target
                    )
                    r414_oracle_center_pred_size_iou = geometry_mean(ocps_iou).detach()
                    r414_pred_center_oracle_size_iou = geometry_mean(pcos_iou).detach()
                    dx_px = (proposal_anchor[:, :, 0] - geometry_target[:, :, 0]) * float(w)
                    dy_px = (proposal_anchor[:, :, 1] - geometry_target[:, :, 1]) * float(h)
                    center_error_px = torch.sqrt(dx_px.square() + dy_px.square() + EPS)
                    if bool(r414_extent_matched.any().item()):
                        center_values = center_error_px[r414_extent_matched]
                        r414_center_error_px_mean = center_values.mean().detach()
                        r414_center_error_px_median = center_values.median().detach()
                        r414_center_error_px_p90 = torch.quantile(center_values, 0.90).detach()

                    if r415_enabled:
                        teacher_count = teacher_valid.float().sum().clamp_min(1.0)
                        valid_proposal_count = r411_proposal_valid.float().sum().clamp_min(1.0)
                        match_count = r414_extent_matched.float().sum()
                        r415_identity_match_rate = (match_count / teacher_count).detach()
                        r415_unmatched_teacher_rate = (1.0 - match_count / teacher_count).clamp(0.0, 1.0).detach()
                        r415_unmatched_proposal_rate = (
                            1.0 - match_count / valid_proposal_count
                        ).clamp(0.0, 1.0).detach()
                        r415_oracle_center_pred_size_iou = r414_oracle_center_pred_size_iou
                        r415_pred_center_oracle_size_iou = r414_pred_center_oracle_size_iou
                        r415_center_error_px_mean = r414_center_error_px_mean
                        r415_center_error_px_median = r414_center_error_px_median
                        r415_center_error_px_p90 = r414_center_error_px_p90
                        if bool(r414_extent_matched.any().item()):
                            safe_id = r414_extent_matched_target.clamp_min(0)
                            selected_dist = r415_pair_center_distance_px.gather(
                                2, safe_id[:, :, None]
                            )[:, :, 0]
                            selected_gate = r415_pair_center_gate_px.gather(
                                2, safe_id[:, :, None]
                            )[:, :, 0]
                            selected_type_mismatch = r415_pair_type_mismatch.gather(
                                2, safe_id[:, :, None]
                            )[:, :, 0]
                            r415_same_type_match_rate = (
                                (~selected_type_mismatch[r414_extent_matched]).float().mean()
                            ).detach()
                            utilization = selected_dist / selected_gate.clamp_min(1.0e-6)
                            r415_center_gate_utilization_mean = utilization[r414_extent_matched].mean().detach()
                            r415_tight_center_match_rate = (
                                utilization[r414_extent_matched] <= 0.5
                            ).float().mean().detach()

                if r416_enabled:
                    r416_post_topk_identity_coverage = r415_identity_match_rate.detach() if r415_enabled else r414_extent_match_rate.detach()
                    if r416_ltrb_enabled:
                        proposal_point = aux.get("v552r416_proposal_point_xy")
                        pred_edge = aux.get("v552r416_edge_offsets")
                        if not isinstance(proposal_point, torch.Tensor) or proposal_point.shape != (b, k, 2):
                            raise RuntimeError("V552-R4.16 LTRB requires v552r416_proposal_point_xy [B,K,2]")
                        if not isinstance(pred_edge, torch.Tensor) or pred_edge.shape != (b, k, 4):
                            raise RuntimeError("V552-R4.16 LTRB requires v552r416_edge_offsets [B,K,4]")
                        gt_cx, gt_cy, gt_w, gt_h = geometry_target.unbind(dim=2)
                        gt_edges = torch.stack([
                            gt_cx - 0.5 * gt_w, gt_cy - 0.5 * gt_h,
                            gt_cx + 0.5 * gt_w, gt_cy + 0.5 * gt_h,
                        ], dim=2)
                        point_ref = torch.stack([
                            proposal_point[:, :, 0], proposal_point[:, :, 1],
                            proposal_point[:, :, 0], proposal_point[:, :, 1],
                        ], dim=2)
                        target_edge = gt_edges - point_ref.detach()
                        edge_map = F.smooth_l1_loss(
                            pred_edge, target_edge.detach(), reduction="none", beta=0.02
                        ).mean(dim=2)
                        r416_edge_offset_loss = geometry_mean(edge_map)
                        scale_px = pred_edge.new_tensor([float(w), float(h), float(w), float(h)])
                        edge_mae_map = ((pred_edge - target_edge).abs() * scale_px).mean(dim=2)
                        r416_edge_offset_mae_px = geometry_mean(edge_mae_map).detach()

        if bool(_m1(cfg, "V552R47_POINT_MASK_SUPERVISION_ENABLED", False)):
            probability = torch.sigmoid(slot_mask_logits)
            uncertainty = 1.0 - (2.0 * (probability - 0.5).abs()).clamp(0.0, 1.0)
            target_flat4 = gathered_masks.reshape(b * k, 1, h, w)
            dilated = F.max_pool2d(target_flat4, 3, stride=1, padding=1)
            eroded = 1.0 - F.max_pool2d(1.0 - target_flat4, 3, stride=1, padding=1)
            target_boundary = (dilated - eroded).clamp(0.0, 1.0).reshape(b, k, h, w)
            score = (
                float(_m1(cfg, "V552R47_POINT_UNCERTAINTY_WEIGHT", 1.0)) * uncertainty.detach()
                + float(_m1(cfg, "V552R47_POINT_POSITIVE_WEIGHT", 1.0)) * gathered_masks
                + float(_m1(cfg, "V552R47_POINT_BOUNDARY_WEIGHT", 1.0)) * target_boundary
            )
            num_points = min(
                max(int(_m1(cfg, "V552R47_POINT_COUNT", 1024)), 64), h * w
            )
            point_index = score.flatten(2).topk(num_points, dim=2).indices
            point_logits = slot_mask_logits.flatten(2).gather(2, point_index)
            point_target = gathered_masks.flatten(2).gather(2, point_index)
            point_bce_element = F.binary_cross_entropy_with_logits(
                point_logits, point_target, reduction="none"
            )
            point_pos = point_target.sum(dim=2)
            point_neg = float(num_points) - point_pos
            point_pos_loss = (point_bce_element * point_target).sum(dim=2) / point_pos.clamp_min(1.0)
            point_neg_loss = (point_bce_element * (1.0 - point_target)).sum(dim=2) / point_neg.clamp_min(1.0)
            point_bce_map = 0.5 * (point_pos_loss + point_neg_loss)
            point_probability = torch.sigmoid(point_logits)
            point_intersection = (point_probability * point_target).sum(dim=2)
            point_dice_map = 1.0 - (
                2.0 * point_intersection + EPS
            ) / (point_probability.sum(dim=2) + point_target.sum(dim=2) + EPS)
            r47_point_bce = weighted_matched_mean(point_bce_map)
            r47_point_dice = weighted_matched_mean(point_dice_map)
            r47_point_mask_loss = r47_point_bce + r47_point_dice

    # R4.8 consistent-assignment deep supervision.  The final Hungarian
    # assignment is reused at every decoder layer, preventing intermediate
    # layers from receiving a different query/component identity target.
    r48_deep_mask_loss = slot_masks.sum() * 0.0
    r48_deep_anchor_loss = slot_masks.sum() * 0.0
    r48_stage0_purity = slot_masks.new_zeros(())
    r48_stage_last_purity = mask_soft_purity.detach()
    r48_dn_mask_loss = slot_masks.sum() * 0.0
    r48_dn_anchor_loss = slot_masks.sum() * 0.0
    r48_dn_reconstruction_dice = slot_masks.new_zeros(())
    r48_dn_active_fraction = slot_masks.new_zeros(())
    r412_oracle_box_canonical_loss = slot_masks.sum() * 0.0
    r412_oracle_box_canonical_dice = slot_masks.new_zeros(())
    r48_attention_entropy = aux.get("v552r48_local_attention_entropy", slot_masks.new_zeros(())).to(slot_masks.dtype)
    r49_attention_max_weight = aux.get("v552r49_attention_max_weight", slot_masks.new_zeros(())).to(slot_masks.dtype)
    r49_dn_noise_scale = aux.get("v552r49_dn_noise_scale", slot_masks.new_zeros(())).to(slot_masks.dtype)
    r49_attention_logit_scale = aux.get("v552r49_attention_logit_scale_mean", slot_masks.new_zeros(())).to(slot_masks.dtype)
    local_points = max(int(_m1(cfg, "V552R48_LOCAL_GRID_SIZE", 3)), 1) ** 2
    max_entropy = math.log(float(max(local_points, 2)))
    r49_attention_entropy_ratio = r48_attention_entropy / max(max_entropy, EPS)

    def _r48_balanced_mask_maps(logits: torch.Tensor, target: torch.Tensor):
        elem = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pos = target.sum(dim=(-2, -1))
        total = float(target.shape[-2] * target.shape[-1])
        neg = total - pos
        pos_loss = (elem * target).sum(dim=(-2, -1)) / pos.clamp_min(1.0)
        neg_loss = (elem * (1.0 - target)).sum(dim=(-2, -1)) / neg.clamp_min(1.0)
        bce = 0.5 * (pos_loss + neg_loss)
        prob = torch.sigmoid(logits)
        inter = (prob * target).sum(dim=(-2, -1))
        dice_loss = 1.0 - (2.0 * inter + EPS) / (
            prob.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1)) + EPS
        )
        purity = (inter + EPS) / (prob.sum(dim=(-2, -1)) + EPS)
        return bce + dice_loss, purity

    if r48_enabled and bool(_m1(cfg, "V552R48_DEEP_SUPERVISION_ENABLED", False)):
        stage_logits = aux.get("v552r48_stage_mask_logits")
        stage_anchors = aux.get("v552r48_stage_anchor_params")
        if isinstance(stage_logits, torch.Tensor) and stage_logits.ndim == 5 and stage_logits.shape[2] == k:
            losses = []
            anchors = []
            purities = []
            # Final stage already receives the primary mask/anchor objective;
            # deep supervision is applied to preceding stages only.
            for stage_index in range(max(int(stage_logits.shape[1]) - 1, 0)):
                stage_map, stage_purity_map = _r48_balanced_mask_maps(
                    stage_logits[:, stage_index], gathered_masks
                )
                losses.append(weighted_matched_mean(stage_map))
                purities.append(weighted_matched_mean(stage_purity_map).detach())
                if (not r413_enabled) and isinstance(stage_anchors, torch.Tensor) and stage_anchors.ndim == 4:
                    anchor_map = F.smooth_l1_loss(
                        stage_anchors[:, stage_index], gathered_teacher_geometry.detach(),
                        reduction="none", beta=0.05,
                    ).mean(dim=2)
                    anchors.append(weighted_matched_mean(anchor_map))
            if losses:
                r48_deep_mask_loss = torch.stack(losses).mean()
                r48_stage0_purity = purities[0]
            if anchors:
                r48_deep_anchor_loss = torch.stack(anchors).mean()

    if r48_enabled and bool(_m1(cfg, "V552R48_DN_COMPONENT_QUERY_ENABLED", False)):
        dn_logits = aux.get("v552r48_dn_stage_mask_logits")
        dn_anchors = aux.get("v552r48_dn_stage_anchor_params")
        dn_target_index = aux.get("v552r48_dn_target_index")
        dn_valid = aux.get("v552r48_dn_valid")
        if (
            isinstance(dn_logits, torch.Tensor) and dn_logits.ndim == 5
            and isinstance(dn_target_index, torch.Tensor)
            and isinstance(dn_valid, torch.Tensor)
            and dn_target_index.ndim == 2 and dn_valid.ndim == 2
            and dn_logits.shape[0] == b and dn_logits.shape[2] == dn_target_index.shape[1]
        ):
            dn_index = dn_target_index.long().clamp(0, teacher_masks.shape[1] - 1)
            batch_index = torch.arange(b, device=slot_masks.device)[:, None]
            dn_target_masks = teacher_masks[batch_index, dn_index]
            dn_target_geometry = teacher_geometry[batch_index, dn_index]
            valid = dn_valid.bool() & teacher_valid[batch_index, dn_index]
            dn_mask_terms = []
            dn_anchor_terms = []
            dn_final_dice = []
            r412_oracle_terms = []
            r412_oracle_dice_terms = []
            for stage_index in range(dn_logits.shape[1]):
                dn_map, _ = _r48_balanced_mask_maps(dn_logits[:, stage_index], dn_target_masks)
                if bool(valid.any().item()):
                    dn_mask_terms.append(dn_map[valid].mean())
                    if r412_enabled:
                        canonical_size = max(int(_m1(cfg, "V552R412_CANONICAL_ROI_SIZE", 64)), 32)
                        oracle_logits = _r412_canonical_crop(
                            dn_logits[:, stage_index], dn_target_geometry.detach(), canonical_size
                        )
                        oracle_target = _r412_canonical_crop(
                            dn_target_masks, dn_target_geometry.detach(), canonical_size
                        ).clamp(0.0, 1.0)
                        oracle_map, oracle_dice_map, _, _ = _r412_balanced_mask_map(
                            oracle_logits, oracle_target
                        )
                        r412_oracle_terms.append(oracle_map[valid].mean())
                        if stage_index == dn_logits.shape[1] - 1:
                            r412_oracle_dice_terms.append(oracle_dice_map[valid].mean())
                    prob = torch.sigmoid(dn_logits[:, stage_index])
                    inter = (prob * dn_target_masks).sum(dim=(-2, -1))
                    dice = (2.0 * inter + EPS) / (
                        prob.sum(dim=(-2, -1)) + dn_target_masks.sum(dim=(-2, -1)) + EPS
                    )
                    if stage_index == dn_logits.shape[1] - 1:
                        dn_final_dice.append(dice[valid].mean())
                    if isinstance(dn_anchors, torch.Tensor) and dn_anchors.ndim == 4:
                        amap = F.smooth_l1_loss(
                            dn_anchors[:, stage_index], dn_target_geometry.detach(),
                            reduction="none", beta=0.05,
                        ).mean(dim=2)
                        dn_anchor_terms.append(amap[valid].mean())
            if dn_mask_terms:
                r48_dn_mask_loss = torch.stack(dn_mask_terms).mean()
            if dn_anchor_terms:
                r48_dn_anchor_loss = torch.stack(dn_anchor_terms).mean()
            if dn_final_dice:
                r48_dn_reconstruction_dice = torch.stack(dn_final_dice).mean().detach()
            if r412_oracle_terms:
                r412_oracle_box_canonical_loss = torch.stack(r412_oracle_terms).mean()
            if r412_oracle_dice_terms:
                r412_oracle_box_canonical_dice = torch.stack(r412_oracle_dice_terms).mean().detach()
            r48_dn_active_fraction = valid.to(slot_masks.dtype).mean().detach()

    unmatched = ~matched
    unmatched_suppression = (
        slot_masks[unmatched].mean()
        if bool(unmatched.any().item()) else slot_masks.sum() * 0.0
    )

    action_map = F.cross_entropy(
        action_logits.reshape(b * k, 4), gathered_actions.reshape(b * k), reduction="none"
    ).reshape(b, k)
    polarity_map = F.cross_entropy(
        polarity_logits.reshape(b * k, 2), gathered_polarity.reshape(b * k), reduction="none"
    ).reshape(b, k)
    dose_map = F.smooth_l1_loss(
        slot_doses,
        gathered_dose.detach(),
        beta=max(float(_m1(cfg, "V540_DOSE_HUBER_BETA", 0.5)), 1.0e-6),
        reduction="none",
    )
    if v560_clean_core:
        action_loss = equal_matched_mean(action_map)
        polarity_loss = equal_matched_mean(polarity_map)
        dose_loss = equal_matched_mean(dose_map)
    else:
        action_loss = weighted_matched_mean(action_map)
        polarity_loss = weighted_matched_mean(polarity_map)
        dose_loss = weighted_matched_mean(dose_map)

    if tc_drcs:
        pilot_action_logits_tc = aux.get("tc_pilot_action_logits")
        pilot_valid_tc = aux.get("tc_pilot_valid")
        pilot_teacher_actions_tc = aux.get("tc_pilot_teacher_actions")
        if not isinstance(pilot_action_logits_tc, torch.Tensor) or pilot_action_logits_tc.shape != action_logits.shape:
            raise RuntimeError("TC-DRCS requires tc_pilot_action_logits [B,K,4]")
        if not isinstance(pilot_valid_tc, torch.Tensor) or pilot_valid_tc.shape != matched.shape:
            raise RuntimeError("TC-DRCS requires tc_pilot_valid [B,K]")
        if not isinstance(pilot_teacher_actions_tc, torch.Tensor) or pilot_teacher_actions_tc.shape != gathered_actions.shape:
            raise RuntimeError("TC-DRCS requires tc_pilot_teacher_actions [B,K]")
        pilot_action_map_tc = F.cross_entropy(
            pilot_action_logits_tc.reshape(b * k, 4),
            pilot_teacher_actions_tc.reshape(b * k).long(),
            reduction="none",
        ).reshape(b, k)
        action_instances = []
        if bool(matched.any().item()):
            action_instances.append(action_map[matched])
        pilot_valid_tc = pilot_valid_tc.bool()
        if bool(pilot_valid_tc.any().item()):
            action_instances.append(pilot_action_map_tc[pilot_valid_tc])
            tc_pilot_action_loss = pilot_action_map_tc[pilot_valid_tc].mean()
        if action_instances:
            action_loss = torch.cat([x.reshape(-1) for x in action_instances], dim=0).mean()

    m1_positive_epsilon = max(float(_m1(cfg, "V538_GAIN_EPSILON", 1.0e-4)), 0.0)
    useful_match = matched & (gathered_teacher_gain > m1_positive_epsilon)
    # R4.21.2 makes presence literal set existence/no-object.  Utility is a
    # separate M2 responsibility; harmful or neutral *real* components remain
    # valid candidates and are valuable counterexamples for Safety/Utility.
    presence_owner = matched if r4212_existence_no_object else useful_match
    if v560_clean_core:
        presence_owner = matched
    presence_target = presence_owner.to(presence_logits.dtype)
    positive_count = presence_target.sum()
    negative_count = presence_target.numel() - positive_count
    positive_weight = (negative_count / positive_count.clamp_min(1.0)).clamp(1.0, 8.0)
    presence_element = F.binary_cross_entropy_with_logits(
        presence_logits, presence_target, reduction="none"
    )
    presence_weight = torch.where(
        presence_target > 0.5,
        positive_weight * (1.0 + utility_scale * gathered_utility.detach()),
        torch.ones_like(presence_target),
    )
    if v560_clean_core:
        # K is small (six by default); ordinary BCE is sufficient and keeps
        # existence logits physically interpretable as no-object probabilities.
        presence_loss = presence_element.mean()
    else:
        presence_loss = (
            presence_element * presence_weight
        ).sum() / presence_weight.sum().clamp_min(1.0)

    use_exact_utility_alignment = bool(
        _m1(cfg, "V545_EXACT_UTILITY_ALIGNMENT_ENABLED", False)
    )
    use_matched_locality = bool(
        _m1(cfg, "V545_MATCHED_LOCALITY_ENABLED", False)
    )
    utility_candidates = (
        exact_slot_candidates_st if use_exact_utility_alignment else slot_candidates
    )
    utility_slot_dice = _dice_many(utility_candidates, gt)
    utility_base_dice = _dice_many(base[:, 0][:, None], gt)[:, 0]
    utility_gain = utility_slot_dice - utility_base_dice[:, None]
    aligned_useful = useful_match & (
        mask_contrast_active if use_exact_utility_alignment else torch.ones_like(useful_match)
    )
    if bool(aligned_useful.any().item()):
        aligned_weight = matched_weight[aligned_useful]
        regret_numerator = (
            F.relu(
                gathered_teacher_gain.detach()[aligned_useful]
                - utility_gain[aligned_useful]
            )
            * aligned_weight
        ).sum()
        regret_denominator = (
            gathered_teacher_gain.detach()[aligned_useful].clamp_min(0.0)
            * aligned_weight
        ).sum().clamp_min(EPS)
        soft_gain_loss = regret_numerator / regret_denominator
    else:
        soft_gain_loss = utility_gain.sum() * 0.0
    harm_loss = (
        presence_probs * F.relu(-utility_gain)
    ).sum() / presence_probs.sum().clamp_min(1.0)
    if use_matched_locality:
        locality_area = hard_masks_st.sum(dim=(-2, -1)).clamp_min(1.0)
        matched_locality_map = (
            hard_masks_st * (1.0 - gathered_masks)
        ).sum(dim=(-2, -1)) / locality_area
        outside_damage = weighted_matched_mean(matched_locality_map)
    else:
        outside_damage = (
            slot_masks * (1.0 - teacher_error[:, 0][:, None])
        ).sum() / slot_masks.sum().clamp_min(1.0)

    mask_intersection = torch.einsum("bkp,bjp->bkj", pred_flat, pred_flat)
    mask_union = (
        pred_flat.sum(dim=2)[:, :, None]
        + pred_flat.sum(dim=2)[:, None, :]
        - mask_intersection
    ).clamp_min(EPS)
    pair_iou = mask_intersection / mask_union
    off_diagonal = ~torch.eye(k, device=slot_masks.device, dtype=torch.bool)[None]
    diversity_loss = pair_iou[off_diagonal.expand_as(pair_iou)].mean()

    with torch.no_grad():
        exact_hard_candidates = exact_slot_candidates >= 0.5
        hard_slot_dice = _dice_binary_many(exact_hard_candidates, gt)
        hard_base = base >= 0.5
        hard_base_dice = _dice_many(hard_base[:, 0][:, None], gt)[:, 0]
        hard_gain = hard_slot_dice - hard_base_dice[:, None]

        base_bool = hard_base[:, 0].bool()[:, None].expand(-1, k, -1, -1)
        gt_bool = gt[:, 0].bool()[:, None].expand(-1, k, -1, -1)
        ideal_teacher_candidate = torch.where(
            teacher_masks.bool(), gt_bool, base_bool
        ).to(gt.dtype)
        ideal_teacher_dice = _dice_binary_many(ideal_teacher_candidate, gt)
        ideal_teacher_gain = torch.where(
            teacher_valid,
            ideal_teacher_dice - hard_base_dice[:, None],
            torch.zeros_like(best_teacher_action_gain),
        )
        teacher_gain = best_teacher_action_gain.to(hard_gain.dtype)

        # --------------------------------------------------------------
        # V561 set-level upper bounds.
        # --------------------------------------------------------------
        # Teacher components are factual Base-error regions and therefore
        # mutually consistent with GT.  Correcting the union of all retained
        # Teacher components is the exact K-capacity Teacher Set Oracle.
        if v561_bcrs:
            teacher_union_v561 = (
                (teacher_masks > 0.5)
                & teacher_valid[:, :, None, None]
            ).any(dim=1)
            base_bool_case_v561 = hard_base[:, 0].bool()
            gt_bool_case_v561 = gt[:, 0].bool()
            teacher_set_candidate_v561 = torch.where(
                teacher_union_v561, gt_bool_case_v561, base_bool_case_v561
            )
            teacher_set_inter_v561 = (
                teacher_set_candidate_v561 & gt_bool_case_v561
            ).flatten(1).sum(dim=1).to(base.dtype)
            teacher_set_den_v561 = (
                teacher_set_candidate_v561.flatten(1).sum(dim=1)
                + gt_bool_case_v561.flatten(1).sum(dim=1)
            ).to(base.dtype)
            teacher_set_dice_v561 = (
                2.0 * teacher_set_inter_v561 + EPS
            ) / (teacher_set_den_v561 + EPS)
            teacher_set_oracle_gain_v561 = (
                teacher_set_dice_v561 - hard_base_dice
            ).mean()

            # Student Set Oracle enumerates every subset of the <=6 executable
            # predicted candidates.  Composition itself is GT-free: candidate
            # changes are applied relative to Base, while pixels on which two
            # selected candidates request contradictory binary states revert to
            # Base.  GT is used only to choose the best subset, hence this is a
            # diagnostic upper bound and never a deployment path.
            if k <= 8:
                candidate_bool_v561 = exact_hard_candidates.bool()
                base_expand_v561 = base_bool_case_v561[:, None]
                add_change_v561 = candidate_bool_v561 & (~base_expand_v561)
                remove_change_v561 = (~candidate_bool_v561) & base_expand_v561
                best_set_dice_v561 = hard_base_dice.clone()
                subset_ids_v561 = torch.arange(
                    1, 1 << k, device=slot_masks.device, dtype=torch.long
                )
                bit_ids_v561 = torch.arange(k, device=slot_masks.device, dtype=torch.long)
                for start_v561 in range(0, int(subset_ids_v561.numel()), 8):
                    subset_chunk_v561 = subset_ids_v561[start_v561:start_v561 + 8]
                    choose_v561 = (
                        (subset_chunk_v561[:, None] >> bit_ids_v561[None, :]) & 1
                    ).bool()
                    selected_v561 = (
                        choose_v561[None, :, :, None, None]
                        & slot_valid[:, None, :, None, None]
                    )
                    add_any_v561 = (
                        selected_v561 & add_change_v561[:, None]
                    ).any(dim=2)
                    remove_any_v561 = (
                        selected_v561 & remove_change_v561[:, None]
                    ).any(dim=2)
                    conflict_v561 = add_any_v561 & remove_any_v561
                    composed_v561 = base_bool_case_v561[:, None].expand(
                        -1, subset_chunk_v561.numel(), -1, -1
                    ).clone()
                    composed_v561 = torch.where(
                        add_any_v561 & (~conflict_v561),
                        torch.ones_like(composed_v561),
                        composed_v561,
                    )
                    composed_v561 = torch.where(
                        remove_any_v561 & (~conflict_v561),
                        torch.zeros_like(composed_v561),
                        composed_v561,
                    )
                    gt_expand_v561 = gt_bool_case_v561[:, None]
                    inter_v561 = (composed_v561 & gt_expand_v561).flatten(2).sum(dim=2).to(base.dtype)
                    den_v561 = (
                        composed_v561.flatten(2).sum(dim=2)
                        + gt_expand_v561.expand_as(composed_v561).flatten(2).sum(dim=2)
                    ).to(base.dtype)
                    dice_v561 = (2.0 * inter_v561 + EPS) / (den_v561 + EPS)
                    best_set_dice_v561 = torch.maximum(
                        best_set_dice_v561, dice_v561.max(dim=1).values
                    )
                student_set_oracle_gain_v561 = (
                    best_set_dice_v561 - hard_base_dice
                ).mean()
            else:
                # Formal V561 uses K=6.  Keep a safe fallback if a future
                # diagnostic changes capacity beyond exact subset enumeration.
                student_set_oracle_gain_v561 = hard_gain.clamp_min(0.0).max(dim=1).values.mean()
        else:
            teacher_set_oracle_gain_v561 = hard_gain.new_zeros(())
            student_set_oracle_gain_v561 = hard_gain.new_zeros(())

        # V544 M1 oracle ladder.  Each level replaces exactly one predicted
        # factor, making mask/polarity/dose bottlenecks separately testable.
        decomposition_valid = matched & slot_valid
        predicted_hard_masks = (hard_masks_st.detach() >= 0.5).to(base.dtype)
        base_logit_for_decomposition = torch.logit(
            base.clamp(EPS, 1.0 - EPS)
        )[:, 0][:, None]
        teacher_sign_for_slot = torch.where(
            gathered_polarity == 0,
            -torch.ones_like(gathered_dose),
            torch.ones_like(gathered_dose),
        )
        predicted_polarity_index = polarity_logits.detach().argmax(dim=2)
        predicted_sign_for_slot = torch.where(
            predicted_polarity_index == 0,
            -torch.ones_like(slot_doses),
            torch.ones_like(slot_doses),
        )
        mask_oracle_probability = torch.sigmoid(
            base_logit_for_decomposition
            + predicted_hard_masks
            * teacher_sign_for_slot[:, :, None, None]
            * gathered_dose.detach()[:, :, None, None]
        )
        polarity_oracle_probability = torch.sigmoid(
            base_logit_for_decomposition
            + predicted_hard_masks
            * predicted_sign_for_slot[:, :, None, None]
            * gathered_dose.detach()[:, :, None, None]
        )
        mask_oracle_gain_slots = (
            _dice_binary_many(mask_oracle_probability >= 0.5, gt)
            - hard_base_dice[:, None]
        )
        polarity_oracle_gain_slots = (
            _dice_binary_many(polarity_oracle_probability >= 0.5, gt)
            - hard_base_dice[:, None]
        )
        full_oracle_gain_slots = hard_gain
        def _oracle_case_mean(slot_gain: torch.Tensor) -> torch.Tensor:
            return torch.where(
                decomposition_valid,
                slot_gain.clamp_min(0.0),
                torch.zeros_like(slot_gain),
            ).max(dim=1).values.mean()
        v544_mask_oracle_gain = _oracle_case_mean(mask_oracle_gain_slots)
        v544_polarity_oracle_gain = _oracle_case_mean(
            polarity_oracle_gain_slots
        )
        v544_full_oracle_gain = _oracle_case_mean(full_oracle_gain_slots)

    gain_epsilon = max(float(_m1(cfg, "V538_GAIN_EPSILON", 1.0e-4)), 0.0)
    teacher_oracle_case = teacher_gain.clamp_min(0.0).max(dim=1).values
    relative_margin = max(float(_m1(cfg, "V541_M2_RELATIVE_GAIN_MARGIN", 0.05)), 0.0)
    absolute_margin = max(float(_m1(cfg, "V541_M2_ABSOLUTE_GAIN_MARGIN", gain_epsilon)), 0.0)
    gain_margin_case = torch.maximum(
        teacher_oracle_case * relative_margin,
        hard_gain.new_full((b,), absolute_margin),
    )
    positive_slot = slot_valid & (hard_gain > gain_margin_case[:, None])
    negative_slot = slot_valid & (hard_gain < -gain_margin_case[:, None])
    neutral_slot = slot_valid & (~positive_slot) & (~negative_slot)

    # V543B: one mutually exclusive outcome and one mean loss per class.
    # Class frequency therefore cannot make the 7%-level Benefit class vanish.
    if not isinstance(outcome_logits, torch.Tensor):
        outcome_logits = torch.stack(
            [torch.zeros_like(benefit_logits), benefit_logits, harm_logits],
            dim=-1,
        )
    outcome_target = torch.zeros_like(benefit_logits, dtype=torch.long)
    outcome_target = torch.where(
        positive_slot, torch.ones_like(outcome_target), outcome_target
    )
    outcome_target = torch.where(
        negative_slot, torch.full_like(outcome_target, 2), outcome_target
    )

    candidate_case_weight = torch.where(
        replay_case[:, None],
        gain_scores.new_full(gain_scores.shape, replay_case_weight),
        torch.ones_like(gain_scores),
    )

    streaming_balanced_softmax = bool(
        _m1(cfg, "V546_STREAMING_BALANCED_SOFTMAX_ENABLED", False)
    )
    batch_outcome_counts = torch.stack(
        [
            neutral_slot.float().sum(),
            positive_slot.float().sum(),
            negative_slot.float().sum(),
        ]
    )
    if streaming_balanced_softmax:
        streaming_outcome_counts = _v546_streaming_class_counts(
            cfg, int(epoch), batch_outcome_counts
        )
        # Balanced Softmax uses empirical class counts in the training
        # denominator.  clamp_min(1) is a mathematical zero-count guard, not a
        # tuned class weight.  Raw logits are retained for inference.
        adjusted_outcome_logits = outcome_logits + streaming_outcome_counts.clamp_min(
            1.0
        ).log().view(1, 1, 3)
        if bool(slot_valid.any().item()):
            balanced_element = F.cross_entropy(
                adjusted_outcome_logits[slot_valid],
                outcome_target[slot_valid],
                reduction="none",
            )
            balanced_weight = candidate_case_weight[slot_valid]
            streaming_balanced_softmax_loss = (
                balanced_element * balanced_weight
            ).sum() / balanced_weight.sum().clamp_min(EPS)
        else:
            streaming_balanced_softmax_loss = outcome_logits.sum() * 0.0
    else:
        streaming_outcome_counts = batch_outcome_counts.detach()
        streaming_balanced_softmax_loss = outcome_logits.sum() * 0.0
    streaming_outcome_prior = streaming_outcome_counts / streaming_outcome_counts.sum().clamp_min(
        1.0
    )

    def outcome_group_ce(mask: torch.Tensor, class_index: int) -> torch.Tensor:
        if not bool(mask.any().item()):
            return outcome_logits.sum() * 0.0
        target = torch.full(
            (int(mask.sum().item()),),
            int(class_index),
            device=outcome_logits.device,
            dtype=torch.long,
        )
        element = F.cross_entropy(outcome_logits[mask], target, reduction="none")
        weight = candidate_case_weight[mask]
        return (element * weight).sum() / weight.sum().clamp_min(EPS)

    neutral_outcome_loss = outcome_group_ce(neutral_slot, 0)
    benefit_loss = outcome_group_ce(positive_slot, 1)
    harm_classification_loss = outcome_group_ce(negative_slot, 2)
    outcome_classification_loss = (
        float(
            _m1(
                cfg,
                "V543_M2_NEUTRAL_OUTCOME_WEIGHT",
                _m1(cfg, "V543_NEUTRAL_OUTCOME_WEIGHT", 0.50),
            )
        )
        * neutral_outcome_loss
        + float(
            _m1(
                cfg,
                "V543_M2_BENEFIT_OUTCOME_WEIGHT",
                _m1(cfg, "V543_BENEFIT_OUTCOME_WEIGHT", 1.00),
            )
        )
        * benefit_loss
        + float(
            _m1(
                cfg,
                "V543_M2_HARM_OUTCOME_WEIGHT",
                _m1(cfg, "V543_HARM_OUTCOME_WEIGHT", 1.00),
            )
        )
        * harm_classification_loss
    )

    # Legacy V541 static contract tokens retained for source-level compatibility:
    # V541_M2_BENEFIT_WEIGHT / V541_M2_HARM_CLASS_WEIGHT are superseded by
    # V547_EDITABILITY_WEIGHT and V547_DIRECTION_WEIGHT when factorisation is on.
    # V547: explicit factorisation.  Editability is trained on every valid slot;
    # direction is trained only on actual Benefit/Harm slots.  A Harm-only batch
    # cannot suppress Benefit through a three-class denominator because Neutral
    # and sign are separate Bernoulli decisions.
    class_complete_outcome = bool(
        _m1(cfg, "V545_CLASS_COMPLETE_OUTCOME_ENABLED", False)
    )
    nonneutral_slot = positive_slot | negative_slot

    def _weighted_bce(
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return _v548_weighted_bce(
            logits, target, mask, candidate_case_weight
        )

    # V548 root fix: restore the class-complete Direction contract.  The
    # previous V547 hotfix let Harm-only batches move the conditional boundary,
    # which was sufficient to collapse P(Benefit | Editable) to zero at batch=2.
    class_complete_direction = bool(
        _m1(cfg, "V548_CLASS_COMPLETE_DIRECTION_UPDATE", False)
    )
    direction_update_active = False
    benefit_direction_loss = direction_logits.sum() * 0.0
    harm_direction_loss = direction_logits.sum() * 0.0
    v549_queue_enabled = bool(
        _m1(cfg, "V549_CROSS_BATCH_BALANCED_QUEUE_ENABLED", False)
    )
    v549_queue_diag: Dict[str, torch.Tensor] = {}
    v549_queue_sign_loss = direction_logits.sum() * 0.0
    v549_queue_benefit_sign_loss = direction_logits.sum() * 0.0
    v549_queue_harm_sign_loss = direction_logits.sum() * 0.0
    v549_queue_sign_active = False
    v550_streaming_enabled = bool(
        _m1(cfg, "V550_CURRENT_SEMANTIC_BALANCE_ENABLED", False)
    )
    v550_streaming_diag: Dict[str, torch.Tensor] = {}
    v550_streaming_sign_loss = direction_logits.sum() * 0.0
    v550_streaming_benefit_sign_loss = direction_logits.sum() * 0.0
    v550_streaming_harm_sign_loss = direction_logits.sum() * 0.0
    v550_streaming_sign_active = False

    if factorized_outcome_enabled:
        edit_target = nonneutral_slot.to(editability_logits.dtype)
        # Equal class means when both classes are present; otherwise update only
        # the observed class without inventing a pseudo-count.
        edit_parts = []
        if bool(nonneutral_slot.any().item()):
            edit_parts.append(
                _weighted_bce(
                    editability_logits, edit_target, nonneutral_slot
                )
            )
        if bool(neutral_slot.any().item()):
            edit_parts.append(
                _weighted_bce(editability_logits, edit_target, neutral_slot)
            )
        editability_loss = (
            torch.stack(edit_parts).mean()
            if edit_parts else editability_logits.sum() * 0.0
        )
        if v550_streaming_enabled:
            (
                direction_loss,
                benefit_direction_loss,
                harm_direction_loss,
                v550_streaming_sign_loss,
                v550_streaming_benefit_sign_loss,
                v550_streaming_harm_sign_loss,
                direction_update_active,
                v550_streaming_diag,
            ) = _v550_current_semantic_balanced_losses(
                cfg=cfg,
                epoch=int(epoch),
                direction_logits=direction_logits,
                gain_normalized=gain_normalized,
                positive_slot=positive_slot,
                negative_slot=negative_slot,
                sample_weight=candidate_case_weight,
                normalized_margin=float(
                    max(float(_m1(cfg, "V542_GAIN_SIGN_MARGIN", 1.0e-4)), 0.0)
                    * max(float(_m1(cfg, "V541_GAIN_SCALE", 1000.0)), 1.0)
                ),
            )
            v550_streaming_sign_active = direction_update_active
        elif v549_queue_enabled:
            selector_features = aux["v549_slot_selector_features"]
            (
                direction_loss,
                benefit_direction_loss,
                harm_direction_loss,
                v549_queue_sign_loss,
                v549_queue_benefit_sign_loss,
                v549_queue_harm_sign_loss,
                direction_update_active,
                v549_queue_diag,
            ) = _v549_cross_batch_balanced_losses(
                cfg=cfg,
                epoch=int(epoch),
                selector_features=selector_features,
                positive_slot=positive_slot,
                negative_slot=negative_slot,
                direction_weight=aux["v549_direction_head_weight"],
                direction_bias=aux["v549_direction_head_bias"],
                editability_weight=aux["v549_editability_head_weight"],
                editability_bias=aux["v549_editability_head_bias"],
                gain_weight=aux["v549_gain_head_weight"],
                gain_bias=aux["v549_gain_head_bias"],
                normalized_margin=float(
                    max(float(_m1(cfg, "V542_GAIN_SIGN_MARGIN", 1.0e-4)), 0.0)
                    * max(float(_m1(cfg, "V541_GAIN_SCALE", 1000.0)), 1.0)
                ),
            )
            v549_queue_sign_active = direction_update_active
        else:
            (
                direction_loss,
                benefit_direction_loss,
                harm_direction_loss,
                direction_update_active,
            ) = _v548_factorized_direction_loss(
                direction_logits=direction_logits,
                positive_slot=positive_slot,
                negative_slot=negative_slot,
                sample_weight=candidate_case_weight,
                require_both_classes=class_complete_direction,
            )
        outcome_classification_loss = (
            float(_m1(cfg, "V547_EDITABILITY_WEIGHT", 1.0))
            * editability_loss
            + float(_m1(cfg, "V547_DIRECTION_WEIGHT", 1.0))
            * direction_loss
        )
    else:
        nonneutral_logit = torch.logsumexp(
            torch.stack([benefit_logits, harm_logits], dim=-1), dim=-1
        )

        def _binary_group_mean(mask: torch.Tensor, target_value: float) -> torch.Tensor:
            if not bool(mask.any().item()):
                return nonneutral_logit.sum() * 0.0
            selected_logit = nonneutral_logit[mask]
            target = torch.full_like(selected_logit, target_value)
            element = F.binary_cross_entropy_with_logits(
                selected_logit, target, reduction="none"
            )
            weight = candidate_case_weight[mask]
            return (element * weight).sum() / weight.sum().clamp_min(EPS)

        edit_positive_loss = _binary_group_mean(nonneutral_slot, 1.0)
        edit_neutral_loss = _binary_group_mean(neutral_slot, 0.0)
        editability_parts = []
        if bool(nonneutral_slot.any().item()):
            editability_parts.append(edit_positive_loss)
        if bool(neutral_slot.any().item()):
            editability_parts.append(edit_neutral_loss)
        editability_loss = (
            torch.stack(editability_parts).mean()
            if editability_parts else nonneutral_logit.sum() * 0.0
        )
        direction_update_active = bool(
            positive_slot.any().item() and negative_slot.any().item()
        )
        benefit_direction_loss = nonneutral_logit.sum() * 0.0
        harm_direction_loss = nonneutral_logit.sum() * 0.0
        if direction_update_active:
            legacy_direction_logits = torch.stack(
                [benefit_logits, harm_logits], dim=-1
            )
            benefit_direction = F.cross_entropy(
                legacy_direction_logits[positive_slot],
                torch.zeros(
                    int(positive_slot.sum().item()),
                    device=legacy_direction_logits.device,
                    dtype=torch.long,
                ),
                reduction="none",
            )
            harm_direction = F.cross_entropy(
                legacy_direction_logits[negative_slot],
                torch.ones(
                    int(negative_slot.sum().item()),
                    device=legacy_direction_logits.device,
                    dtype=torch.long,
                ),
                reduction="none",
            )
            benefit_direction_loss = (
                benefit_direction * candidate_case_weight[positive_slot]
            ).sum() / candidate_case_weight[positive_slot].sum().clamp_min(EPS)
            harm_direction_loss = (
                harm_direction * candidate_case_weight[negative_slot]
            ).sum() / candidate_case_weight[negative_slot].sum().clamp_min(EPS)
            direction_loss = 0.5 * (
                benefit_direction_loss + harm_direction_loss
            )
        else:
            direction_loss = outcome_logits.sum() * 0.0
        if class_complete_outcome:
            outcome_classification_loss = editability_loss + direction_loss
        if streaming_balanced_softmax:
            outcome_classification_loss = streaming_balanced_softmax_loss

    # ================================================================
    # V542: deterministic, class-balanced normalized Gain regression.
    # The gain head predicts normalized_gain = raw_dice_gain * gain_scale.
    # Regressing in normalized space avoids the 1 / gain_scale gradient shrink.
    # ================================================================
    gain_scale = max(float(_m1(cfg, "V541_GAIN_SCALE", 1000.0)), 1.0)
    normalized_target = (hard_gain.detach() * gain_scale).clamp(-8.0, 8.0)
    magnitude_target = normalized_target.abs()
    huber_beta = max(float(_m1(cfg, "V542_GAIN_HUBER_BETA", 0.25)), 1.0e-6)

    def masked_huber(
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if not bool(mask.any().item()):
            return prediction.sum() * 0.0
        element = F.smooth_l1_loss(
            prediction[mask],
            target[mask],
            beta=huber_beta,
            reduction="none",
        )
        weight = candidate_case_weight[mask]
        return (element * weight).sum() / weight.sum().clamp_min(EPS)

    gain_huber_benefit = masked_huber(
        gain_magnitude_normalized, magnitude_target, positive_slot
    )
    gain_huber_harm = masked_huber(
        gain_magnitude_normalized, magnitude_target, negative_slot
    )
    gain_huber_neutral = masked_huber(
        gain_magnitude_normalized,
        torch.zeros_like(magnitude_target),
        neutral_slot,
    )

    gain_regression = (
        float(_m1(cfg, "V542_GAIN_BENEFIT_WEIGHT", 0.50))
        * gain_huber_benefit
        + float(_m1(cfg, "V542_GAIN_HARM_WEIGHT", 0.40))
        * gain_huber_harm
        + float(_m1(cfg, "V542_GAIN_NEUTRAL_WEIGHT", 0.10))
        * gain_huber_neutral
    )

    # Explicit sign calibration is also computed in normalized space.
    # A raw margin of 1e-4 becomes 0.1 when V541_GAIN_SCALE == 1000.
    raw_sign_margin = max(
        float(_m1(cfg, "V542_GAIN_SIGN_MARGIN", 1.0e-4)), 0.0
    )
    normalized_sign_margin = raw_sign_margin * gain_scale

    explicit_sign_loss_enabled = bool(
        _m1(cfg, "V548_EXPLICIT_SIGN_LOSS_ENABLED", False)
    )
    sign_loss_requires_both = bool(
        _m1(cfg, "V548_SIGN_LOSS_REQUIRES_BOTH_CLASSES", False)
    )
    if v550_streaming_enabled and factorized_outcome_enabled:
        gain_sign_loss = v550_streaming_sign_loss
        benefit_sign_loss = v550_streaming_benefit_sign_loss
        harm_sign_loss = v550_streaming_harm_sign_loss
        sign_loss_active = v550_streaming_sign_active
    elif v549_queue_enabled and factorized_outcome_enabled:
        gain_sign_loss = v549_queue_sign_loss
        benefit_sign_loss = v549_queue_benefit_sign_loss
        harm_sign_loss = v549_queue_harm_sign_loss
        sign_loss_active = v549_queue_sign_active
    else:
        (
            gain_sign_loss,
            benefit_sign_loss,
            harm_sign_loss,
            sign_loss_active,
        ) = _v548_balanced_gain_sign_loss(
            gain_normalized=gain_normalized,
            positive_slot=positive_slot,
            negative_slot=negative_slot,
            normalized_margin=normalized_sign_margin,
            require_both_classes=(
                sign_loss_requires_both and factorized_outcome_enabled
            ),
        )

    # Legacy V541 compatibility: when deployment explicitly uses an LCB
    # (beta>0), train the log-variance with a normalized Gaussian NLL.  V547
    # sets beta=0, so its deterministic factorized objective is unchanged.
    learned_uncertainty_active = float(
        _m1(cfg, "V541_DEPLOY_LCB_BETA", 0.0)
    ) > 0.0
    if learned_uncertainty_active and bool(slot_valid.any().item()):
        squared_error = (gain_normalized - normalized_target).pow(2)
        uncertainty_element = 0.5 * (
            torch.exp(-gain_logvar) * squared_error + gain_logvar
        )
        uncertainty_weight = candidate_case_weight[slot_valid]
        gain_uncertainty_nll = (
            uncertainty_element[slot_valid] * uncertainty_weight
        ).sum() / uncertainty_weight.sum().clamp_min(EPS)
    else:
        gain_uncertainty_nll = gain_logvar.sum() * 0.0
    gain_nll_shift = gain_uncertainty_nll.detach()

    gain_difference = hard_gain[:, :, None] - hard_gain[:, None, :]
    pair_valid = slot_valid[:, :, None] & slot_valid[:, None, :]
    pair_target = pair_valid & (gain_difference > gain_margin_case[:, None, None])
    score_difference = rank_scores[:, :, None] - rank_scores[:, None, :]
    rank_margin = max(float(_m1(cfg, "V541_M2_RANK_MARGIN", 0.5)), 0.0)
    pairwise_rank_loss = (
        F.softplus(rank_margin - score_difference[pair_target]).mean()
        if bool(pair_target.any().item()) else rank_scores.sum() * 0.0
    )
    preserve_terms = []
    if bool(positive_slot.any().item()):
        preserve_terms.append(F.softplus(rank_margin - rank_scores[positive_slot]).mean())
    if bool(negative_slot.any().item()):
        preserve_terms.append(F.softplus(rank_margin + rank_scores[negative_slot]).mean())
    if bool(neutral_slot.any().item()):
        preserve_terms.append(
            float(_m1(cfg, "V541_M2_NEUTRAL_PRESERVE_WEIGHT", 0.25))
            * rank_scores[neutral_slot].abs().mean()
        )
    preserve_loss = (
        torch.stack(preserve_terms).mean()
        if preserve_terms else rank_scores.sum() * 0.0
    )

    masked_gain = hard_gain.detach().masked_fill(~slot_valid, -1.0e4)
    best_gain, best_index = masked_gain.max(dim=1)
    positive_case = slot_valid.any(dim=1) & (best_gain > gain_margin_case)
    target_index = torch.where(positive_case, best_index + 1, torch.zeros_like(best_index))
    all_rank_scores = torch.cat(
        [rank_scores.new_zeros((b, 1)), rank_scores.masked_fill(~slot_valid, -1.0e4)],
        dim=1,
    )
    temperature = max(float(_m1(cfg, "V541_M2_LISTWISE_TEMPERATURE", 1.0)), 1.0e-4)
    listwise_case = F.cross_entropy(
        all_rank_scores / temperature, target_index, reduction="none"
    )
    pos_cases = positive_case.float().sum()
    neg_cases = float(b) - pos_cases
    positive_case_weight = (
        neg_cases / pos_cases.clamp_min(1.0)
    ).clamp(1.0, float(_m1(cfg, "V538_M2_MAX_POSITIVE_WEIGHT", 8.0)))
    case_weight = torch.where(
        positive_case, positive_case_weight, torch.ones_like(listwise_case)
    )
    listwise_loss = (
        listwise_case * case_weight
    ).sum() / case_weight.sum().clamp_min(1.0)
    (
        v550_gain_pairwise_loss,
        v550_gain_listwise_loss,
        v550_gain_pair_accuracy,
        v550_gain_top_choice_accuracy,
    ) = _v550_deployed_gain_order_losses(
        gain_normalized=gain_normalized,
        pair_target=pair_target,
        target_index=target_index,
        case_weight=case_weight,
        slot_valid=slot_valid,
        pair_margin=max(
            float(_m1(cfg, "V550_GAIN_PAIRWISE_MARGIN", normalized_sign_margin)),
            0.0,
        ),
        listwise_temperature=max(
            float(_m1(cfg, "V550_GAIN_LISTWISE_TEMPERATURE", 1.0)), 1.0e-4
        ),
    )

    deploy_presence = float(_m1(cfg, "V538_DEPLOY_PRESENCE_THRESHOLD", 0.50))
    deploy_benefit = float(_m1(cfg, "V541_DEPLOY_BENEFIT_THRESHOLD", 0.70))
    deploy_harm = float(_m1(cfg, "V541_DEPLOY_HARM_THRESHOLD", 0.10))
    deploy_gain = float(_m1(cfg, "V538_COMPOSER_MIN_GAIN", 0.001))
    factorized_deployment = bool(
        _m1(cfg, "V549_FACTORIZED_DEPLOYMENT_ENABLED", False)
    ) and factorized_outcome_enabled
    unified_gate = bool(_m1(cfg, "V552_UNIFIED_DEPLOYMENT_GATE", False))
    benefit_harm_margin = max(
        float(_m1(cfg, "V552_DEPLOY_BENEFIT_HARM_MARGIN", 0.10)), 0.0
    )
    eligible = deploy_slot_valid & (presence_probs.detach() >= deploy_presence)
    if factorized_deployment:
        deploy_editability = float(
            _m1(cfg, "V549_DEPLOY_EDITABILITY_THRESHOLD", 0.50)
        )
        deploy_direction = float(
            _m1(cfg, "V549_DEPLOY_DIRECTION_THRESHOLD", 0.50)
        )
        eligible = eligible & (
            editability_probs.detach() >= deploy_editability
        ) & (direction_probs.detach() >= deploy_direction)
    if unified_gate or not factorized_deployment:
        eligible = eligible & (
            benefit_probs.detach() >= deploy_benefit
        ) & (harm_probs.detach() <= deploy_harm)
        if unified_gate:
            eligible = eligible & (
                benefit_probs.detach()
                >= harm_probs.detach() + benefit_harm_margin
            )
    eligible = eligible & (gain_lcb.detach() > deploy_gain)
    masked_decision = decision_scores.detach().masked_fill(~eligible, -1.0e4)
    selected_slot = masked_decision.argmax(dim=1)
    predicted_execute = eligible.any(dim=1)
    selected_gain = hard_gain.gather(1, selected_slot[:, None])[:, 0]
    selected_gain = torch.where(
        predicted_execute, selected_gain, torch.zeros_like(selected_gain)
    )
    selected_positive = predicted_execute & (selected_gain > gain_margin_case)
    selected_harmful = predicted_execute & (selected_gain < -gain_margin_case)
    tp = selected_positive.float().sum()
    fp = (predicted_execute & (~selected_positive)).float().sum()
    fn = ((~predicted_execute) & positive_case).float().sum()
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)

    shadow_probability = aux.get(
        "v541_shadow_selected_final_probability", base
    ).detach().clamp(EPS, 1.0 - EPS)
    shadow_execute = aux.get(
        "v541_shadow_predicted_execute",
        torch.zeros((b,), device=base.device, dtype=torch.bool),
    ).detach().bool()
    shadow_dice = _dice_many(
        (shadow_probability >= 0.5).to(gt.dtype)[:, 0][:, None], gt
    )[:, 0]
    shadow_gain = shadow_dice - hard_base_dice
    shadow_positive = shadow_execute & (shadow_gain > gain_margin_case)
    shadow_harmful = shadow_execute & (shadow_gain < -gain_margin_case)
    shadow_tp = shadow_positive.float().sum()
    shadow_fp = (shadow_execute & (~shadow_positive)).float().sum()
    shadow_precision = shadow_tp / (shadow_tp + shadow_fp).clamp_min(1.0)
    shadow_improved_rate = shadow_positive.float().mean()
    shadow_harmful_rate = shadow_harmful.float().mean()
    shadow_selected_gain = torch.where(
        shadow_execute, shadow_gain, torch.zeros_like(shadow_gain)
    ).mean()

    # R4.4 independent audit evidence.  This trace is forced on the highest
    # scoring physically valid candidate and is never used as the deployed
    # prediction.  It exists solely to estimate real train-set improvement and
    # harm without requiring the persistent quality gate to be open first.
    audit_probability = aux.get(
        "v552r44_audit_selected_final_probability", base
    ).detach().clamp(EPS, 1.0 - EPS)
    audit_execute = aux.get(
        "v552r44_audit_predicted_execute",
        torch.zeros((b,), device=base.device, dtype=torch.bool),
    ).detach().bool()
    audit_dice = _dice_many(
        (audit_probability >= 0.5).to(gt.dtype)[:, 0][:, None], gt
    )[:, 0]
    audit_gain = audit_dice - hard_base_dice
    audit_positive = audit_execute & (audit_gain > gain_margin_case)
    audit_harmful = audit_execute & (audit_gain < -gain_margin_case)
    audit_tp = audit_positive.float().sum()
    audit_fp = (audit_execute & (~audit_positive)).float().sum()
    audit_precision = audit_tp / (audit_tp + audit_fp).clamp_min(1.0)
    audit_selected_gain = torch.where(
        audit_execute, audit_gain, torch.zeros_like(audit_gain)
    ).mean()

    # Policy audit must be identity-equal to formal deployment except for the
    # persistent latch.  R4.6 therefore audits the latch-bypass *gated shadow*
    # rather than the historical ungated Composer trace.
    policy_audit_probability = aux.get(
        "v541_shadow_selected_final_probability" if r46_enabled
        else "v552_composer_ungated_final_probability",
        base,
    ).detach().clamp(EPS, 1.0 - EPS)
    policy_audit_steps = aux.get(
        "v541_shadow_accepted_count" if r46_enabled
        else "v552_composer_ungated_step_count",
        torch.zeros((b,), device=base.device, dtype=base.dtype),
    ).detach().float()
    policy_audit_execute = (
        aux.get(
            "v541_shadow_predicted_execute",
            policy_audit_steps > 0,
        ).detach().bool()
        if r46_enabled else policy_audit_steps > 0
    )
    policy_audit_dice = _dice_many(
        (policy_audit_probability >= 0.5).to(gt.dtype)[:, 0][:, None], gt
    )[:, 0]
    policy_audit_gain = policy_audit_dice - hard_base_dice
    policy_audit_positive = policy_audit_execute & (
        policy_audit_gain > gain_margin_case
    )
    policy_audit_harmful = policy_audit_execute & (
        policy_audit_gain < -gain_margin_case
    )
    policy_audit_tp = policy_audit_positive.float().sum()
    policy_audit_fp = (
        policy_audit_execute & (~policy_audit_positive)
    ).float().sum()
    policy_audit_precision = policy_audit_tp / (
        policy_audit_tp + policy_audit_fp
    ).clamp_min(1.0)
    policy_audit_selected_gain = torch.where(
        policy_audit_execute,
        policy_audit_gain,
        torch.zeros_like(policy_audit_gain),
    ).mean()

    component_oracle_gain = torch.where(
        slot_valid, hard_gain.clamp_min(0.0), torch.zeros_like(hard_gain)
    ).max(dim=1).values.mean()
    ideal_teacher_oracle_case = ideal_teacher_gain.clamp_min(0.0).max(dim=1).values
    teacher_oracle_gain = teacher_oracle_case.mean()
    teacher_realization_ratio = (
        component_oracle_gain / teacher_oracle_gain.clamp_min(EPS)
    ).clamp(0.0, 10.0)
    ideal_teacher_oracle_gain = ideal_teacher_oracle_case.mean()
    real_case = ~replay_case
    real_teacher_oracle_gain = (
        teacher_oracle_case[real_case].mean()
        if bool(real_case.any().item()) else teacher_oracle_case.new_zeros(())
    )
    replay_teacher_oracle_gain = (
        teacher_oracle_case[replay_case].mean()
        if bool(replay_case.any().item()) else teacher_oracle_case.new_zeros(())
    )
    teacher_positive = teacher_valid & (teacher_gain > gain_epsilon)
    teacher_positive_rate = (
        teacher_positive.float().sum() / teacher_valid.float().sum().clamp_min(1.0)
    )
    positive_rate = positive_slot.float().sum() / slot_valid.float().sum().clamp_min(1.0)
    slot_purity = (
        slot_masks * teacher_error[:, 0][:, None]
    ).sum(dim=(-2, -1)) / slot_masks.sum(dim=(-2, -1)).clamp_min(EPS)
    mean_purity = (
        slot_purity[slot_valid].mean()
        if bool(slot_valid.any().item()) else slot_purity.new_zeros(())
    )
    sample_purity = (
        (slot_purity * slot_valid.to(slot_purity.dtype)).sum(dim=1)
        / slot_valid.float().sum(dim=1).clamp_min(1.0)
    )
    real_component_purity = (
        sample_purity[real_case].mean()
        if bool(real_case.any().item()) else sample_purity.new_zeros(())
    )
    replay_component_purity = (
        sample_purity[replay_case].mean()
        if bool(replay_case.any().item()) else sample_purity.new_zeros(())
    )
    m1_oracle_case = torch.where(
        slot_valid, hard_gain.clamp_min(0.0), torch.zeros_like(hard_gain)
    ).max(dim=1).values
    captured_gain_mass = torch.minimum(m1_oracle_case, teacher_oracle_case).sum()
    teacher_gain_mass = teacher_oracle_case.sum()
    capture_ratio = torch.where(
        teacher_gain_mass > EPS,
        captured_gain_mass / teacher_gain_mass.clamp_min(EPS),
        teacher_gain_mass.new_zeros(()),
    ).clamp(0.0, 1.0)

    teacher_floor = max(float(_m1(cfg, "V540_M2_MIN_TEACHER_ORACLE_GAIN", 0.0)), 0.0)
    absolute_floor = max(float(_m1(
        cfg,
        "V540_M2_MIN_ABSOLUTE_ORACLE_GAIN",
        _m1(cfg, "V538_M2_MIN_COMPONENT_ORACLE_GAIN", 0.005),
    )), 0.0)
    teacher_fraction = max(float(_m1(cfg, "V540_M2_MIN_TEACHER_CAPTURE_FRACTION", 0.0)), 0.0)
    adaptive_oracle_threshold = max(
        absolute_floor, teacher_fraction * float(teacher_oracle_gain.detach().cpu())
    )
    teacher_ready = float(teacher_oracle_gain.detach().cpu()) >= teacher_floor
    oracle_ready = float(component_oracle_gain.detach().cpu()) >= adaptive_oracle_threshold
    capture_ready = float(capture_ratio.detach().cpu()) >= float(
        _m1(cfg, "V540_M2_MIN_CAPTURE_RATIO", 0.0)
    )
    selector_precision_ready = (
        not v541_selector_enabled
        or float(shadow_precision.detach().cpu()) >= float(
            _m1(cfg, "V541_QUALITY_MIN_EXECUTE_PRECISION", 0.50)
        )
    )
    selector_gain_ready = (
        not v541_selector_enabled
        or float(shadow_selected_gain.detach().cpu()) > float(
            _m1(cfg, "V541_QUALITY_MIN_COMPOSER_GAIN", 0.0)
        )
    )
    selector_harm_ready = (
        not v541_selector_enabled
        or float(shadow_harmful_rate.detach().cpu()) <= float(
            _m1(cfg, "V541_QUALITY_MAX_HARMFUL_CASE_RATE", 0.10)
        )
    )
    selector_balance_ready = (
        not v541_selector_enabled
        or float(shadow_improved_rate.detach().cpu())
        > float(shadow_harmful_rate.detach().cpu())
    )
    ready = bool(
        int(epoch) >= int(_m1(cfg, "V538_M2_MIN_START_EPOCH", 15))
        and teacher_ready
        and oracle_ready
        and capture_ready
        and float(positive_rate.detach().cpu())
            >= float(_m1(cfg, "V538_M2_MIN_POSITIVE_COMPONENT_RATE", 0.10))
        and float(mean_purity.detach().cpu())
            >= float(_m1(cfg, "V538_M2_MIN_COMPONENT_PURITY", 0.15))
        and selector_precision_ready
        and selector_gain_ready
        and selector_harm_ready
        and selector_balance_ready
    )
    ready_scale = gain_scores.new_tensor(1.0 if ready else 0.0)

    def _curriculum_scale(start_key: str, ramp_key: str, default_start: int) -> float:
        start_epoch = int(_m1(cfg, start_key, default_start))
        ramp_epochs = max(int(_m1(cfg, ramp_key, 1)), 1)
        if int(epoch) < start_epoch:
            return 0.0
        if ramp_epochs <= 1:
            return 1.0
        return min(1.0, max(0.0, float(int(epoch) - start_epoch + 1) / float(ramp_epochs)))

    m1_train_scale = gain_scores.new_tensor(
        _curriculum_scale("V538_M1_ROUTE_START_EPOCH", "V538_M1_ROUTE_RAMP_EPOCHS", 0)
    )
    m2_curriculum_scale = _curriculum_scale(
        "V538_M2_ROUTE_START_EPOCH",
        "V538_M2_ROUTE_RAMP_EPOCHS",
        int(_m1(cfg, "V538_M2_MIN_START_EPOCH", 15)),
    )
    v550_quality_routing_enabled = bool(
        _m1(cfg, "V550_CANDIDATE_QUALITY_ROUTING_ENABLED", False)
    )
    v550_candidate_quality_score = 1.0
    v550_quality_route_multiplier = 1.0
    if v550_quality_routing_enabled:
        target_positive = max(
            float(_m1(cfg, "V550_QUALITY_TARGET_POSITIVE_RATE", 0.03)), EPS
        )
        target_purity = max(
            float(_m1(cfg, "V550_QUALITY_TARGET_PURITY", 0.10)), EPS
        )
        target_capture = max(
            float(_m1(cfg, "V550_QUALITY_TARGET_CAPTURE", 0.20)), EPS
        )
        supply_score = min(
            1.0, max(0.0, float(positive_rate.detach().cpu()) / target_positive)
        )
        purity_score = min(
            1.0, max(0.0, float(mean_purity.detach().cpu()) / target_purity)
        )
        capture_score = min(
            1.0, max(0.0, float(capture_ratio.detach().cpu()) / target_capture)
        )
        v550_candidate_quality_score = (
            max(supply_score * purity_score * capture_score, 0.0) ** (1.0 / 3.0)
        )
        quality_floor = min(
            1.0,
            max(0.0, float(_m1(cfg, "V550_MIN_QUALITY_TRAIN_SCALE", 0.25))),
        )
        v550_quality_route_multiplier = quality_floor + (
            1.0 - quality_floor
        ) * v550_candidate_quality_score
    # V552 separates learning from execution.  The calibrated selector keeps
    # receiving standard supervised gradients throughout its curriculum, while
    # deployment remains controlled by the strict quality latch.
    m2_supervision_scale = gain_scores.new_tensor(m2_curriculum_scale)
    m2_execution_scale = gain_scores.new_tensor(
        m2_curriculum_scale
        * v550_quality_route_multiplier
        * (1.0 if ready else 0.0)
    )
    m2_train_scale = m2_supervision_scale

    # V552-R4.21.0 C4: directly align the deployable M1-only output with the
    # segmentation target using only standard BCE + Dice.  The model builds
    # this probability with the exact inference-visible hard winner in forward
    # and a straight-through score distribution in backward.  Base logits are
    # detached in the model, so this objective cannot improve Base by shortcut.
    r4210_m1_native_bce_loss = slot_masks.sum() * 0.0
    r4210_m1_native_dice_loss = slot_masks.sum() * 0.0
    r4210_m1_native_alignment_loss = slot_masks.sum() * 0.0
    r4210_m1_native_soft_dice = slot_masks.new_zeros(())
    if r4210_m1_native_alignment:
        native_train_probability = aux.get("v552r4210_m1_native_train_probability")
        if (
            not isinstance(native_train_probability, torch.Tensor)
            or native_train_probability.shape != (b, 1, h, w)
        ):
            raise RuntimeError(
                "V552-R4.21.0 M1Native alignment requires "
                "v552r4210_m1_native_train_probability [B,1,H,W]"
            )
        native_train_probability = native_train_probability.clamp(EPS, 1.0 - EPS)
        native_target = gt.detach().to(native_train_probability.dtype).clamp(0.0, 1.0)
        r4210_m1_native_bce_loss = F.binary_cross_entropy(
            native_train_probability, native_target
        )
        native_intersection = (native_train_probability * native_target).flatten(1).sum(dim=1)
        native_denominator = (
            native_train_probability.flatten(1).sum(dim=1)
            + native_target.flatten(1).sum(dim=1)
        )
        native_dice_case = (2.0 * native_intersection + EPS) / (native_denominator + EPS)
        r4210_m1_native_soft_dice = native_dice_case.mean().detach()
        r4210_m1_native_dice_loss = 1.0 - native_dice_case.mean()
        r4210_m1_native_alignment_loss = 0.5 * (
            r4210_m1_native_bce_loss + r4210_m1_native_dice_loss
        )

    # R4.21.2 candidate alignment: train exactly the M1 intervention that M2
    # consumes.  A matched Teacher component defines the corrected candidate as
    # Base outside the component and GT inside it.  The objective is standard
    # BCE + soft Dice, with no project-specific margin or gain normalization.
    r4212_candidate_alignment_bce = slot_masks.sum() * 0.0
    r4212_candidate_alignment_dice_loss = slot_masks.sum() * 0.0
    r4212_candidate_alignment_loss = slot_masks.sum() * 0.0
    r4212_candidate_alignment_soft_dice = slot_masks.new_zeros(())
    if r4212_candidate_alignment:
        teacher_candidate_r4212 = (
            base[:, 0][:, None].detach() * (1.0 - gathered_masks.detach())
            + gt[:, 0][:, None].detach() * gathered_masks.detach()
        ).clamp(EPS, 1.0 - EPS)
        pred_candidate_r4212 = exact_slot_candidates_st.clamp(EPS, 1.0 - EPS)
        bce_map_r4212 = F.binary_cross_entropy(
            pred_candidate_r4212, teacher_candidate_r4212, reduction="none"
        ).mean(dim=(-2, -1))
        inter_r4212 = (pred_candidate_r4212 * teacher_candidate_r4212).sum(dim=(-2, -1))
        denom_r4212 = (
            pred_candidate_r4212.sum(dim=(-2, -1))
            + teacher_candidate_r4212.sum(dim=(-2, -1))
        )
        dice_r4212 = (2.0 * inter_r4212 + EPS) / (denom_r4212 + EPS)
        if v560_clean_core:
            r4212_candidate_alignment_bce = equal_matched_mean(bce_map_r4212)
            r4212_candidate_alignment_dice_loss = equal_matched_mean(1.0 - dice_r4212)
            r4212_candidate_alignment_soft_dice = equal_matched_mean(dice_r4212).detach()
        else:
            r4212_candidate_alignment_bce = weighted_matched_mean(bce_map_r4212)
            r4212_candidate_alignment_dice_loss = weighted_matched_mean(1.0 - dice_r4212)
            r4212_candidate_alignment_soft_dice = weighted_matched_mean(dice_r4212).detach()
        r4212_candidate_alignment_loss = 0.5 * (
            r4212_candidate_alignment_bce + r4212_candidate_alignment_dice_loss
        )

    m1_objective = (
        float(_m1(cfg, "V538_COMPONENT_MASK_WEIGHT", 1.0)) * mask_loss
        + (r4212_candidate_alignment_loss if r4212_candidate_alignment else r4212_candidate_alignment_loss * 0.0)
        + float(_m1(cfg, "V538_UNMATCHED_SLOT_WEIGHT", 0.25)) * unmatched_suppression
        + float(_m1(cfg, "V538_ACTION_WEIGHT", 0.25)) * action_loss
        + float(_m1(cfg, "V540_POLARITY_WEIGHT", 1.0)) * polarity_loss
        + float(_m1(cfg, "V540_DOSE_WEIGHT", 0.25)) * dose_loss
        + float(_m1(cfg, "V538_PRESENCE_WEIGHT", 0.25)) * presence_loss
        + float(_m1(cfg, "V538_SOFT_GAIN_WEIGHT", 1.0)) * soft_gain_loss
        + float(_m1(cfg, "V538_HARM_WEIGHT", 2.0)) * harm_loss
        + float(_m1(cfg, "V538_CORRECT_DAMAGE_WEIGHT", 1.0)) * outside_damage
        + float(_m1(cfg, "V538_DIVERSITY_WEIGHT", 0.10)) * diversity_loss
        + (
            float(_m1(cfg, "V552R47_POINT_MASK_WEIGHT", 2.0)) * r47_point_mask_loss
            + float(_m1(cfg, "V552R47_ANCHOR_REG_WEIGHT", 0.5)) * r47_anchor_reg_loss
            if r47_enabled
            else r47_point_mask_loss * 0.0
        )
        + (
            float(_m1(cfg, "V552R48_DEEP_MASK_WEIGHT", 1.0)) * r48_deep_mask_loss
            + float(_m1(cfg, "V552R48_DEEP_ANCHOR_WEIGHT", 0.25)) * r48_deep_anchor_loss
            + float(_m1(cfg, "V552R48_DN_MASK_WEIGHT", 1.0)) * r48_dn_mask_loss
            + float(_m1(cfg, "V552R48_DN_ANCHOR_WEIGHT", 0.5)) * r48_dn_anchor_loss
            if r48_enabled
            else r48_deep_mask_loss * 0.0
        )
        + (
            float(_m1(cfg, "V552R411_PROPOSAL_CENTER_WEIGHT", 1.0)) * (
                0.5 * (r411_center_loss + r417_location_loss) if r417_enabled else r411_center_loss
            )
            + (0.0 if r413_enabled else float(_m1(cfg, "V552R411_PROPOSAL_SIZE_WEIGHT", 0.5))) * r411_size_loss
            + float(_m1(cfg, "V552R411_PROPOSAL_OFFSET_WEIGHT", 0.25)) * (
                r417_location_offset_loss if r417_shared_offset_enabled else r411_offset_loss
            )
            if r411_enabled
            else r411_center_loss * 0.0
        )
        + (
            r4204_occupancy_loss
            if r4204_enabled
            else r4204_occupancy_loss * 0.0
        )
        + (
            r4209_balanced_overflow_loss
            if r4209_balanced_overflow
            else r4209_balanced_overflow_loss * 0.0
        )
        + (
            r4210_independent_overflow_loss
            if r4210_independent_overflow
            else r4210_independent_overflow_loss * 0.0
        )
        + (
            r4210_m1_native_alignment_loss
            if r4210_m1_native_alignment
            else r4210_m1_native_alignment_loss * 0.0
        )
        + (
            float(_m1(cfg, "V552R412_BOX_GIOU_WEIGHT", 0.5)) * r412_box_giou_loss
            + float(_m1(cfg, "V552R412_ORACLE_SHAPE_WEIGHT", 1.0)) * r412_oracle_box_canonical_loss
            + (
                (
                    float(_m1(cfg, "V552R416_LTRB_WEIGHT", 1.0)) * r416_edge_offset_loss
                    if r416_ltrb_enabled
                    else float(_m1(cfg, "V552R413_LOG_SIZE_WEIGHT", 1.0)) * r413_log_size_loss
                )
                if r413_enabled else r413_log_size_loss * 0.0
            )
            if r412_enabled
            else r412_box_giou_loss * 0.0
        )
    )
    if v560_clean_core:
        # Minimal V560 M1 contract.
        m1_objective = (
            mask_loss
            + presence_loss
            + action_loss
            + polarity_loss
            + dose_loss
            + r4212_candidate_alignment_loss
        )
    if v561_bcrs:
        # V561 correction instances have exactly three learned responsibilities:
        # geometry, existence/no-object and four-way executable action.
        m1_objective = mask_loss + presence_loss + action_loss
    if v562_rootfix:
        # V562 closes the missing proposal->instance and action->execution links.
        # All added terms are standard BCE/Dice-family supervision.
        m1_objective = (
            mask_loss
            + presence_loss
            + action_loss
            + float(_m1(cfg, "V562_RESIDUAL_PROPOSAL_WEIGHT", 1.0)) * v562_residual_proposal_loss
            + float(_m1(cfg, "V562_LOCAL_EXECUTION_WEIGHT", 1.0)) * v562_local_execution_loss
        )
    if v563_rootfix:
        # V563 keeps the V562 factual proposal/executor closure but replaces the
        # unstable global-balanced geometry owner with persistent local binding.
        m1_objective = (
            mask_loss
            + presence_loss
            + action_loss
            + float(_m1(cfg, "V562_RESIDUAL_PROPOSAL_WEIGHT", 1.0)) * v562_residual_proposal_loss
            + float(_m1(cfg, "V562_LOCAL_EXECUTION_WEIGHT", 1.0)) * v562_local_execution_loss
        )

    clean_uncertainty_weight_mask = slot_masks.new_zeros(())
    clean_uncertainty_weight_presence = slot_masks.new_zeros(())
    clean_uncertainty_weight_action = slot_masks.new_zeros(())
    clean_uncertainty_weight_residual = slot_masks.new_zeros(())
    if clean_dynamic_component_set:
        # Four identifiable M1 responsibilities, all standard losses:
        # residual BCE+Dice, component mask BCE+Dice, no-object BCE, action CE.
        # Relative weights are learned by homoscedastic uncertainty rather than
        # hand-tuned constants: exp(-s_i) L_i + s_i.
        clean_log_vars = aux.get("clean_loss_log_vars")
        if not isinstance(clean_log_vars, torch.Tensor) or clean_log_vars.numel() != 4:
            raise RuntimeError("CLEAN requires four attached learnable loss log-variances")
        clean_log_vars = clean_log_vars.reshape(4)
        clean_losses = torch.stack([
            mask_loss,
            presence_loss,
            action_loss,
            v562_residual_proposal_loss,
        ])
        has_match = matched.any().to(clean_losses.dtype)
        clean_active = torch.stack([
            has_match,
            clean_losses.new_ones(()),
            has_match,
            clean_losses.new_ones(()),
        ])
        clean_terms = torch.exp(-clean_log_vars) * clean_losses + clean_log_vars
        m1_objective = (clean_terms * clean_active).sum() / clean_active.sum().clamp_min(1.0)
        clean_weights = torch.exp(-clean_log_vars).detach()
        clean_uncertainty_weight_mask = clean_weights[0]
        clean_uncertainty_weight_presence = clean_weights[1]
        clean_uncertainty_weight_action = clean_weights[2]
        clean_uncertainty_weight_residual = clean_weights[3]

    # V544 rigorous minimal M2: only two identifiable targets are required.
    # Outcome owns sign and magnitude owns |Gain|.  Rank and Preserve remain
    # diagnostics/ablations, but cannot dominate the deployed utility unless the
    # legacy objective is explicitly requested.
    minimal_m2_objective = bool(
        _m1(cfg, "V544_MINIMAL_M2_OBJECTIVE_ENABLED", False)
    )
    uncertainty_objective = (
        float(_m1(cfg, "V541_M2_GAIN_WEIGHT", 0.50))
        * gain_uncertainty_nll
    )
    explicit_sign_objective = (
        float(
            _m1(
                cfg,
                "V550_M2_SIGN_WEIGHT",
                _m1(cfg, "V549_M2_SIGN_WEIGHT",
                    _m1(cfg, "V542_M2_GAIN_SIGN_WEIGHT",
                        _m1(cfg, "V543_M2_SIGN_MARGIN_WEIGHT", 0.50))),
            )
        )
        * gain_sign_loss
        if explicit_sign_loss_enabled
        else gain_sign_loss * 0.0
    )
    if minimal_m2_objective:
        m2_objective = (
            float(_m1(cfg, "V543_M2_OUTCOME_WEIGHT", 1.00))
            * outcome_classification_loss
            + float(_m1(cfg, "V543_M2_MAGNITUDE_WEIGHT", 1.00))
            * gain_regression
            + explicit_sign_objective
            + float(_m1(cfg, "V550_M2_GAIN_PAIRWISE_WEIGHT", 0.0))
            * v550_gain_pairwise_loss
            + float(_m1(cfg, "V550_M2_GAIN_LISTWISE_WEIGHT", 0.0))
            * v550_gain_listwise_loss
            + uncertainty_objective
        )
    else:
        m2_objective = (
            float(_m1(cfg, "V543_M2_OUTCOME_WEIGHT", 1.00))
            * outcome_classification_loss
            + float(_m1(cfg, "V543_M2_MAGNITUDE_WEIGHT", 1.00))
            * gain_regression
            + float(
                _m1(
                    cfg,
                    "V550_M2_SIGN_WEIGHT",
                    _m1(
                        cfg,
                        "V549_M2_SIGN_WEIGHT",
                        _m1(
                            cfg,
                            "V542_M2_GAIN_SIGN_WEIGHT",
                            _m1(cfg, "V543_M2_SIGN_MARGIN_WEIGHT", 0.50),
                        ),
                    ),
                )
            )
            * gain_sign_loss
            + float(_m1(cfg, "V542_M2_RANK_WEIGHT", 1.00))
            * pairwise_rank_loss
            + float(_m1(cfg, "V542_M2_PRESERVE_WEIGHT", 0.50))
            * preserve_loss
            + uncertainty_objective
        )

    # V552-R2 owns the single consistent Outcome/Gain objective.  Keeping the
    # historical Benefit/Harm/Direction/Sign objectives active would recreate
    # the exact semantic conflict diagnosed in R1.
    if bool(_m1(cfg, "V552R2_TEACHER_DECOUPLED_ENABLED", False)):
        m2_objective = m2_objective * 0.0

    v551_m1_extra, v551_m2_extra, v551_diagnostics = (
        compute_v551_multiscale_editor_loss(
            cfg=cfg,
            masks=masks,
            aux=aux,
            epoch=epoch,
            base_probability=base,
        )
    )
    # R4.6: V538 Hungarian action-realizable teacher is the *only* M1
    # objective.  R43/R45 student-current usefulness/EPR losses remain
    # observable diagnostics but may not redefine slot existence.
    if r46_enabled:
        v551_m1_extra = v551_m1_extra * 0.0
    if clean_dynamic_component_set:
        # CLEAN M2 has one literal target: the exact signed DSC gain of each
        # executable candidate. No hand class thresholds, outcome weights, rank
        # weights, or auxiliary historical editor losses enter the clean route.
        m1_objective = m1_objective
        if bool(slot_valid.any().item()):
            clean_m2_signed_gain_mse = F.mse_loss(
                gain_scores[slot_valid], hard_gain.detach()[slot_valid], reduction="mean"
            )
        else:
            clean_m2_signed_gain_mse = gain_scores.sum() * 0.0
        m2_objective = clean_m2_signed_gain_mse
    else:
        clean_m2_signed_gain_mse = gain_scores.sum() * 0.0
        m1_objective = m1_objective + v551_m1_extra
        m2_objective = m2_objective + v551_m2_extra

    benefit_gain_positive_rate = (
        (gain_scores[positive_slot] > 0.0).float().mean()
        if bool(positive_slot.any().item())
        else gain_scores.new_zeros(())
    )
    harm_gain_negative_rate = (
        (gain_scores[negative_slot] < 0.0).float().mean()
        if bool(negative_slot.any().item())
        else gain_scores.new_zeros(())
    )
    signed_slot = positive_slot | negative_slot
    if bool(signed_slot.any().item()):
        sign_correct = (
            (positive_slot & (gain_scores > 0.0))
            | (negative_slot & (gain_scores < 0.0))
        )
        gain_sign_accuracy = sign_correct[signed_slot].float().mean()
    else:
        gain_sign_accuracy = gain_scores.new_zeros(())

    outcome_prediction = outcome_logits.argmax(dim=-1)
    outcome_balanced_terms = []
    for outcome_class, outcome_mask in (
        (0, neutral_slot),
        (1, positive_slot),
        (2, negative_slot),
    ):
        if bool(outcome_mask.any().item()):
            outcome_balanced_terms.append(
                (
                    outcome_prediction[outcome_mask]
                    == int(outcome_class)
                ).float().mean()
            )
    v543_outcome_balanced_accuracy = (
        torch.stack(outcome_balanced_terms).mean()
        if outcome_balanced_terms
        else gain_scores.new_zeros(())
    )


    # V541.1:
    # 部署阈值统计只能说明候选是否通过安全门，不能说明分类器
    # 是否形成了正负样本分离。这里增加常规 0.5 阈值、条件概率、
    # pairwise rank accuracy 和条件 Gain 统计。
    valid_count = slot_valid.float().sum().clamp_min(1.0)

    nonbenefit_valid = slot_valid & (~positive_slot)
    nonharm_valid = slot_valid & (~negative_slot)

    benefit_probability_mean = (
        benefit_probs[slot_valid].mean()
        if bool(slot_valid.any().item())
        else benefit_probs.new_zeros(())
    )

    benefit_probability_positive_mean = (
        benefit_probs[positive_slot].mean()
        if bool(positive_slot.any().item())
        else benefit_probs.new_zeros(())
    )

    benefit_probability_negative_mean = (
        benefit_probs[nonbenefit_valid].mean()
        if bool(nonbenefit_valid.any().item())
        else benefit_probs.new_zeros(())
    )

    harm_probability_mean = (
        harm_probs[slot_valid].mean()
        if bool(slot_valid.any().item())
        else harm_probs.new_zeros(())
    )

    harm_probability_positive_mean = (
        harm_probs[negative_slot].mean()
        if bool(negative_slot.any().item())
        else harm_probs.new_zeros(())
    )

    harm_probability_negative_mean = (
        harm_probs[nonharm_valid].mean()
        if bool(nonharm_valid.any().item())
        else harm_probs.new_zeros(())
    )

    rank_pair_accuracy = (
        (score_difference[pair_target] > 0.0).float().mean()
        if bool(pair_target.any().item())
        else rank_scores.new_zeros(())
    )

    gain_mean_on_benefit = (
        gain_scores[positive_slot].mean()
        if bool(positive_slot.any().item())
        else gain_scores.new_zeros(())
    )

    gain_lcb_on_benefit = (
        gain_lcb[positive_slot].mean()
        if bool(positive_slot.any().item())
        else gain_lcb.new_zeros(())
    )

    gain_mean_on_harm = (
        gain_scores[negative_slot].mean()
        if bool(negative_slot.any().item())
        else gain_scores.new_zeros(())
    )

    # V544 global-count diagnostics.  train.py aggregates numerators and
    # denominators over the full epoch before forming any ratio.
    benefit_total_count = positive_slot.float().sum()
    harm_total_count = negative_slot.float().sum()
    neutral_total_count = neutral_slot.float().sum()
    benefit_gain_positive_count = (
        positive_slot & (gain_scores > 0.0)
    ).float().sum()
    harm_gain_negative_count = (
        negative_slot & (gain_scores < 0.0)
    ).float().sum()
    neutral_outcome_correct_count = (
        neutral_slot & (outcome_prediction == 0)
    ).float().sum()
    benefit_outcome_correct_count = (
        positive_slot & (outcome_prediction == 1)
    ).float().sum()
    harm_outcome_correct_count = (
        negative_slot & (outcome_prediction == 2)
    ).float().sum()
    benefit_probability_sum = benefit_probs[positive_slot].sum()
    nonbenefit_probability_sum = benefit_probs[nonbenefit_valid].sum()
    nonbenefit_total_count = nonbenefit_valid.float().sum()
    signed_outcome = aux.get(
        "v543_slot_signed_outcome", benefit_probs - harm_probs
    )
    signed_outcome_benefit_sum = signed_outcome[positive_slot].sum()
    signed_outcome_harm_sum = signed_outcome[negative_slot].sum()
    zero_benefit_batch = gain_scores.new_tensor(
        0.0 if bool(positive_slot.any().item()) else 1.0
    )

    diagnostics = {
        "v538_m1_component_objective": m1_objective.detach(),
        "v538_m2_listwise_objective": m2_objective.detach(),
        **v551_diagnostics,
        "v547_paired_replay_enabled": gain_scores.new_tensor(
            1.0 if paired_replay_enabled else 0.0
        ),
        "v547_replay_case_rate": replay_case.float().mean().detach(),
        "v547_factorized_outcome_enabled": gain_scores.new_tensor(
            1.0 if factorized_outcome_enabled else 0.0
        ),
        "v547_editability_loss": editability_loss.detach(),
        "v547_direction_loss": direction_loss.detach(),
        "v548_factorized_direction_zero_init_enabled": aux.get(
            "v548_factorized_direction_zero_init_enabled",
            gain_scores.new_zeros(()),
        ).detach(),
        "v548_class_complete_direction_enabled": gain_scores.new_tensor(
            1.0 if class_complete_direction else 0.0
        ),
        "v548_both_direction_classes_present": gain_scores.new_tensor(
            1.0 if bool(positive_slot.any().item() and negative_slot.any().item())
            else 0.0
        ),
        "v548_benefit_direction_loss": benefit_direction_loss.detach(),
        "v548_harm_direction_loss": harm_direction_loss.detach(),
        "v548_explicit_sign_loss_enabled": gain_scores.new_tensor(
            1.0 if explicit_sign_loss_enabled else 0.0
        ),
        "v548_sign_loss_active": gain_scores.new_tensor(
            1.0 if sign_loss_active else 0.0
        ),
        "v548_explicit_sign_objective": explicit_sign_objective.detach(),
        "v549_cross_batch_queue_enabled": gain_scores.new_tensor(
            1.0 if v549_queue_enabled else 0.0
        ),
        **v549_queue_diag,
        **v550_streaming_diag,
        "v550_current_semantic_balance_enabled": gain_scores.new_tensor(
            1.0 if v550_streaming_enabled else 0.0
        ),
        "v550_gain_pairwise_loss": v550_gain_pairwise_loss.detach(),
        "v550_gain_listwise_loss": v550_gain_listwise_loss.detach(),
        "v550_gain_pair_accuracy": v550_gain_pair_accuracy.detach(),
        "v550_gain_top_choice_accuracy": v550_gain_top_choice_accuracy.detach(),
        "v538_component_mask_loss": mask_loss.detach(),
        "v544_region_balanced_mask_enabled": gain_scores.new_tensor(
            1.0 if use_region_balanced_mask else 0.0
        ),
        "v544_mask_soft_purity": mask_soft_purity.detach(),
        "v544_mask_soft_coverage": mask_soft_coverage.detach(),
        "v552r47_rootfix_enabled": mask_loss.new_tensor(1.0 if r47_enabled else 0.0),
        "v552r47_hybrid_matching_enabled": mask_loss.new_tensor(
            1.0 if r47_enabled and r47_match_anchor_weight > 0.0 else 0.0
        ),
        "v552r47_point_mask_loss": r47_point_mask_loss.detach(),
        "v552r47_point_bce": r47_point_bce.detach(),
        "v552r47_point_dice_loss": r47_point_dice.detach(),
        "v552r47_anchor_reg_loss": r47_anchor_reg_loss.detach(),
        "v552r47_anchor_l1": r47_anchor_l1.detach(),
        "v552r47_match_anchor_cost": r47_anchor_l1.detach(),
        "v552r48_rootfix_enabled": mask_loss.new_tensor(1.0 if r48_enabled else 0.0),
        "v552r48_remove_coarse_mask_bias_enabled": aux.get(
            "v552r48_remove_coarse_mask_bias_enabled", mask_loss.new_zeros(())
        ).detach(),
        "v552r48_deep_mask_loss": r48_deep_mask_loss.detach(),
        "v552r48_deep_anchor_loss": r48_deep_anchor_loss.detach(),
        "v552r48_stage0_mask_purity": r48_stage0_purity.detach(),
        "v552r48_stage_last_mask_purity": r48_stage_last_purity.detach(),
        "v552r48_dn_mask_loss": r48_dn_mask_loss.detach(),
        "v552r48_dn_anchor_loss": r48_dn_anchor_loss.detach(),
        "v552r48_dn_reconstruction_dice": r48_dn_reconstruction_dice.detach(),
        "v552r48_dn_active_fraction": r48_dn_active_fraction.detach(),
        "v552r48_local_attention_entropy": r48_attention_entropy.detach(),
        "v552r49_rootfix_enabled": mask_loss.new_tensor(1.0 if r49_enabled else 0.0),
        "v552r49_anchor_sampling_only_enabled": aux.get(
            "v552r49_anchor_sampling_only_enabled", mask_loss.new_zeros(())
        ).detach(),
        "v552r49_attention_entropy_ratio": r49_attention_entropy_ratio.detach(),
        "v552r49_attention_max_weight": r49_attention_max_weight.detach(),
        "v552r49_attention_logit_scale_mean": r49_attention_logit_scale.detach(),
        "v552r49_dn_noise_scale": r49_dn_noise_scale.detach(),
        "v552r410_rootfix_enabled": mask_loss.new_tensor(1.0 if r410_enabled else 0.0),
        "v552r410_evidence_proposal_enabled": aux.get(
            "v552r410_evidence_proposal_enabled", mask_loss.new_zeros(())
        ).detach(),
        "v552r410_support_only_local_readout_enabled": aux.get(
            "v552r410_support_only_local_readout_enabled", mask_loss.new_zeros(())
        ).detach(),
        "v552r410_proposal_score_mean": aux.get(
            "v552r410_proposal_score_mean", mask_loss.new_zeros(())
        ).detach(),
        "v552r410_proposal_valid_fraction": aux.get(
            "v552r410_proposal_valid_fraction", mask_loss.new_zeros(())
        ).detach(),
        "v552r410_dn_clean_curriculum_enabled": aux.get(
            "v552r410_dn_clean_curriculum_enabled", mask_loss.new_zeros(())
        ).detach(),
        "v552r411_rootfix_enabled": mask_loss.new_tensor(1.0 if r411_enabled else 0.0),
        "v552r411_typed_proposal_enabled": aux.get(
            "v552r411_typed_proposal_enabled", mask_loss.new_zeros(())
        ).detach(),
        "v552r411_local_roi_decoder_enabled": aux.get(
            "v552r411_local_roi_decoder_enabled", mask_loss.new_zeros(())
        ).detach(),
        "v552r411_raw_native_mask_enabled": aux.get(
            "v552r411_raw_native_mask_enabled", mask_loss.new_zeros(())
        ).detach(),
        "v552r411_proposal_center_loss": r411_center_loss.detach(),
        "v552r411_proposal_size_loss": r411_size_loss.detach(),
        "v552r411_proposal_offset_loss": r411_offset_loss.detach(),
        "v552r411_matched_type_accuracy": r411_matched_type_accuracy.detach(),
        "v552r411_proposal_center_recall": r411_center_recall.detach(),
        "v552r411_teacher_center_probability": r411_teacher_center_probability.detach(),
        "v552r412_rootfix_enabled": mask_loss.new_tensor(1.0 if r412_enabled else 0.0),
        "v552r412_canonical_renderer_enabled": aux.get(
            "v552r412_canonical_renderer_enabled", mask_loss.new_zeros(())
        ),
        "v552r412_action_support_enabled": aux.get(
            "v552r412_action_support_enabled", mask_loss.new_zeros(())
        ),
        "v552r412_global_mask_loss": global_mask_loss.detach(),
        "v552r412_native_canonical_loss": r412_native_canonical_loss.detach(),
        "v552r412_native_box_canonical_dice": r412_native_canonical_dice.detach(),
        "v552r412_native_box_canonical_purity": r412_native_canonical_purity.detach(),
        "v552r412_oracle_box_canonical_loss": r412_oracle_box_canonical_loss.detach(),
        "v552r412_oracle_box_canonical_dice": r412_oracle_box_canonical_dice.detach(),
        "v552r412_box_iou": r412_box_iou.detach(),
        "v552r412_box_giou_loss": r412_box_giou_loss.detach(),
        "v552r412_width_mae": r412_width_mae.detach(),
        "v552r412_height_mae": r412_height_mae.detach(),
        "v552r413_rootfix_enabled": slot_masks.new_tensor(1.0 if r413_enabled else 0.0),
        "v552r413_proposal_box_iou": r413_proposal_box_iou.detach(),
        "v552r413_final_box_iou": r412_box_iou.detach() if r413_enabled else slot_masks.new_zeros(()),
        "v552r413_box_drift_l1": r413_box_drift_l1.detach(),
        "v552r413_box_drift_center": r413_box_drift_center.detach(),
        "v552r413_box_drift_size": r413_box_drift_size.detach(),
        "v552r413_log_size_loss": r413_log_size_loss.detach(),
        "v552r414_rootfix_enabled": slot_masks.new_tensor(1.0 if r414_enabled else 0.0),
        "v552r414_contextual_extent_enabled": aux.get(
            "v552r414_rootfix_enabled", slot_masks.new_zeros(())
        ).detach(),
        "v552r414_extent_match_rate": r414_extent_match_rate.detach(),
        "v552r414_oracle_center_pred_size_iou": r414_oracle_center_pred_size_iou.detach(),
        "v552r414_pred_center_oracle_size_iou": r414_pred_center_oracle_size_iou.detach(),
        "v552r414_center_error_px_mean": r414_center_error_px_mean.detach(),
        "v552r414_center_error_px_median": r414_center_error_px_median.detach(),
        "v552r414_center_error_px_p90": r414_center_error_px_p90.detach(),
        "v552r414_log_size_loss": r413_log_size_loss.detach() if r414_enabled else slot_masks.new_zeros(()),
        "v552r415_rootfix_enabled": slot_masks.new_tensor(1.0 if r415_enabled else 0.0),
        "v552r415_identity_match_rate": r415_identity_match_rate.detach(),
        "v552r415_same_type_match_rate": r415_same_type_match_rate.detach(),
        "v552r415_tight_center_match_rate": r415_tight_center_match_rate.detach(),
        "v552r415_center_error_px_mean": r415_center_error_px_mean.detach(),
        "v552r415_center_error_px_median": r415_center_error_px_median.detach(),
        "v552r415_center_error_px_p90": r415_center_error_px_p90.detach(),
        "v552r415_center_gate_utilization_mean": r415_center_gate_utilization_mean.detach(),
        "v552r415_unmatched_teacher_rate": r415_unmatched_teacher_rate.detach(),
        "v552r415_unmatched_proposal_rate": r415_unmatched_proposal_rate.detach(),
        "v552r415_oracle_center_pred_size_iou": r415_oracle_center_pred_size_iou.detach(),
        "v552r415_pred_center_oracle_size_iou": r415_pred_center_oracle_size_iou.detach(),
        "v552r415_log_size_loss": r413_log_size_loss.detach() if r415_enabled else slot_masks.new_zeros(()),
        "v552r416_rootfix_enabled": slot_masks.new_tensor(1.0 if r416_enabled else 0.0),
        "v552r416_unique_point_enabled": aux.get("v552r416_unique_point_enabled", slot_masks.new_zeros(())).detach(),
        "v552r416_asymmetric_ltrb_enabled": aux.get("v552r416_asymmetric_ltrb_enabled", slot_masks.new_zeros(())).detach(),
        "v552r416_context_radius": aux.get("v552r414_context_radius", slot_masks.new_zeros(())).detach(),
        "v552r416_legacy_topk_unique_fraction": r416_legacy_topk_unique_fraction.detach(),
        "v552r416_pre_topk_peak_recall": r416_pre_topk_peak_recall.detach(),
        "v552r416_post_topk_identity_coverage": r416_post_topk_identity_coverage.detach(),
        "v552r416_edge_offset_loss": r416_edge_offset_loss.detach(),
        "v552r416_edge_offset_mae_px": r416_edge_offset_mae_px.detach(),
        "v552r416_proposal_box_iou": r413_proposal_box_iou.detach() if r416_enabled else slot_masks.new_zeros(()),
        "v552r417_rootfix_enabled": slot_masks.new_tensor(1.0 if r417_enabled else 0.0),
        "v552r417_location_first_enabled": aux.get("v552r417_location_first_enabled", slot_masks.new_zeros(())).detach(),
        "v552r417_shared_offset_enabled": aux.get("v552r417_shared_offset_enabled", slot_masks.new_zeros(())).detach(),
        "v552r417_location_loss": r417_location_loss.detach(),
        "v552r4201_location_target_mass": r417_location_target_mass.detach(),
        "v552r417_location_center_recall": r417_location_center_recall.detach(),
        "v552r417_pre_topk_peak_recall": r417_pre_topk_peak_recall.detach(),
        "v552r417_selected_spatial_coverage": r417_selected_spatial_coverage.detach(),
        "v552r417_location_offset_loss": r417_location_offset_loss.detach(),
        "v552r417_location_offset_mae_px": r417_location_offset_mae_px.detach(),
        "v552r417_location_nms_kernel": aux.get("v552r417_location_nms_kernel", slot_masks.new_zeros(())).detach(),
        "v552r417_location_dedup_radius_px": aux.get("v552r417_location_dedup_radius_px", slot_masks.new_zeros(())).detach(),
        "v552r418_rootfix_enabled": slot_masks.new_tensor(1.0 if r418_enabled else 0.0),
        "v552r418_box_free_mask_set_enabled": aux.get("v552r418_box_free_mask_set_enabled", slot_masks.new_zeros(())).detach(),
        "v552r418_paired_stable_teacher_enabled": aux.get("v552r418_paired_stable_teacher_enabled", slot_masks.new_zeros(())).detach(),
        "v552r418_paired_mask_loss": r418_paired_mask_loss.detach(),
        "v552r418_paired_mask_dice": r418_paired_mask_dice.detach(),
        "v552r418_paired_valid_fraction": r418_paired_valid_fraction.detach(),
        "v552r418_paired_target_consistency": r418_paired_target_consistency.detach(),
        "v552r418_paired_error_coverage": r418_paired_error_coverage.detach(),
        "v552r418_native_mask_matching_dice": optimal_matching_dice.detach(),
        "v552r419_rootfix_enabled": slot_masks.new_tensor(1.0 if r419_enabled else 0.0),
        "v552r419_seeded_masked_attention_enabled": aux.get("v552r419_seeded_masked_attention_enabled", slot_masks.new_zeros(())).detach(),
        "v552r419_seed_support_fraction": aux.get("v552r419_seed_support_fraction", slot_masks.new_zeros(())).detach(),
        "v552r419_final_support_fraction": aux.get("v552r419_final_support_fraction", slot_masks.new_zeros(())).detach(),
        "v552r419_outside_mask_probability": aux.get("v552r419_outside_mask_probability", slot_masks.new_zeros(())).detach(),
        "v552r419_native_mask_matching_dice": optimal_matching_dice.detach(),
        "v552r419_native_mask_soft_purity": mask_soft_purity.detach(),
        "v552r419_native_mask_soft_coverage": mask_soft_coverage.detach(),
        "v552r420_rootfix_enabled": slot_masks.new_tensor(1.0 if r420_enabled else 0.0),
        "v552r420_dynamic_mask_enabled": aux.get("v552r420_dynamic_mask_enabled", slot_masks.new_zeros(())).detach(),
        "v552r420_type_decoupled_mask_enabled": aux.get("v552r420_type_decoupled_mask_enabled", slot_masks.new_zeros(())).detach(),
        "v552r420_dynamic_channels": aux.get("v552r420_dynamic_channels", slot_masks.new_zeros(())).detach(),
        "v552r420_relative_coord_mean_abs": aux.get("v552r420_relative_coord_mean_abs", slot_masks.new_zeros(())).detach(),
        "v552r420_native_mask_matching_dice": optimal_matching_dice.detach(),
        "v552r420_native_mask_soft_purity": mask_soft_purity.detach(),
        "v552r420_native_mask_soft_coverage": mask_soft_coverage.detach(),
        "v552r4203_rootfix_enabled": slot_masks.new_tensor(1.0 if r4203_enabled else 0.0),
        "v552r4203_dense_competitive_set_enabled": aux.get(
            "v552r4203_dense_competitive_set_enabled", slot_masks.new_zeros(())
        ).detach(),
        "v552r4203_ownership_sum_error": aux.get(
            "v552r4203_ownership_sum_error", slot_masks.new_zeros(())
        ).detach(),
        "v552r4203_assignment_entropy": aux.get(
            "v552r4203_assignment_entropy", slot_masks.new_zeros(())
        ).detach(),
        "v552r4203_background_fraction": aux.get(
            "v552r4203_background_fraction", slot_masks.new_zeros(())
        ).detach(),
        "v552r4203_max_slot_ownership": aux.get(
            "v552r4203_max_slot_ownership", slot_masks.new_zeros(())
        ).detach(),
        "v552r4203_slot_mass_cv": aux.get(
            "v552r4203_slot_mass_cv", slot_masks.new_zeros(())
        ).detach(),
        "v552r4203_point_bottleneck_used": aux.get(
            "v552r4203_point_bottleneck_used", slot_masks.new_ones(())
        ).detach(),
        "v552r4203_native_mask_matching_dice": optimal_matching_dice.detach(),
        "v552r4203_native_mask_soft_purity": mask_soft_purity.detach(),
        "v552r4203_native_mask_soft_coverage": mask_soft_coverage.detach(),
        "v552r4204_rootfix_enabled": slot_masks.new_tensor(1.0 if r4204_enabled else 0.0),
        "v552r4204_spatial_identity_enabled": aux.get(
            "v552r4204_spatial_identity_enabled", slot_masks.new_zeros(())
        ).detach(),
        "v552r4204_location_as_occupancy_used": aux.get(
            "v552r4204_location_as_occupancy_used", slot_masks.new_zeros(())
        ).detach(),
        "v552r4204_residual_mass_conservation_error": aux.get(
            "v552r4204_residual_mass_conservation_error", slot_masks.new_zeros(())
        ).detach(),
        "v552r4204_conditional_slot_entropy": aux.get(
            "v552r4204_conditional_slot_entropy", slot_masks.new_zeros(())
        ).detach(),
        "v552r4204_conditional_max_slot_probability": aux.get(
            "v552r4204_conditional_max_slot_probability", slot_masks.new_zeros(())
        ).detach(),
        "v552r4204_residual_existence_mean": aux.get(
            "v552r4204_residual_existence_mean", slot_masks.new_zeros(())
        ).detach(),
        "v552r4204_centroid_separation": aux.get(
            "v552r4204_centroid_separation", slot_masks.new_zeros(())
        ).detach(),
        "v552r4204_spatial_variance_mean": aux.get(
            "v552r4204_spatial_variance_mean", slot_masks.new_zeros(())
        ).detach(),
        "v552r4204_occupancy_bce_loss": r4204_occupancy_bce_loss.detach(),
        "v552r4204_occupancy_dice_loss": r4204_occupancy_dice_loss.detach(),
        "v552r4204_occupancy_loss": r4204_occupancy_loss.detach(),
        "v552r4204_occupancy_target_fraction": r4204_occupancy_target_fraction.detach(),
        "v552r4204_occupancy_probability_mean": r4204_occupancy_probability_mean.detach(),
        "v552r4204_occupancy_soft_dice": r4204_occupancy_soft_dice.detach(),
        "v552r4204_retained_teacher_residual_fraction": r4204_retained_teacher_residual_fraction.detach(),
        "v552r4204_unretained_teacher_residual_fraction": r4204_unretained_teacher_residual_fraction.detach(),
        "v552r4204_occupancy_target_teacher_error_mae": r4204_occupancy_target_teacher_error_mae.detach(),
        "v552r4204_native_mask_matching_dice": optimal_matching_dice.detach(),
        "v552r4204_native_mask_soft_purity": mask_soft_purity.detach(),
        "v552r4204_native_mask_soft_coverage": mask_soft_coverage.detach(),
        "v552r4205_rootfix_enabled": slot_masks.new_tensor(1.0 if r4205_enabled else 0.0),
        "v552r4205_overflow_target_fraction": r4205_overflow_target_fraction.detach(),
        "v552r4205_overflow_target_residual_fraction": r4205_overflow_target_residual_fraction.detach(),
        "v552r4205_overflow_probability_mean": r4205_overflow_probability_mean.detach(),
        "v552r4205_overflow_predicted_residual_fraction": r4205_overflow_predicted_residual_fraction.detach(),
        "v552r4205_overflow_soft_dice": r4205_overflow_soft_dice.detach(),
        "v552r4205_overflow_soft_precision": r4205_overflow_soft_precision.detach(),
        "v552r4205_overflow_soft_coverage": r4205_overflow_soft_coverage.detach(),
        "v552r4205_editable_on_overflow_leakage": r4205_editable_on_overflow_leakage.detach(),
        "v552r4205_overflow_on_retained_leakage": r4205_overflow_on_retained_leakage.detach(),
        "v552r4205_target_decomposition_error": r4205_target_decomposition_error.detach(),
        "v552r4205_prediction_decomposition_error": r4205_prediction_decomposition_error.detach(),
        "v552r4205_final_logits_finite_fraction": r4205_final_logits_finite_fraction.detach(),
        "v552r4206_rootfix_enabled": slot_masks.new_tensor(1.0 if r4206_enabled else 0.0),
        "v552r4206_conditional_identity_loss": r4206_conditional_identity_loss.detach(),
        "v552r4206_shape_dice_loss": r4206_shape_dice_loss.detach(),
        "v552r4206_conditional_accuracy": r4206_conditional_accuracy.detach(),
        "v552r4206_editable_accuracy": r4206_editable_accuracy.detach(),
        "v552r4206_overflow_recall": r4206_overflow_recall.detach(),
        "v552r4206_overflow_precision": r4206_overflow_precision.detach(),
        "v552r4206_overflow_target_residual_fraction": r4206_overflow_target_residual_fraction.detach(),
        "v552r4206_overflow_predicted_residual_fraction": r4206_overflow_predicted_residual_fraction.detach(),
        "v552r4206_true_class_probability": r4206_true_class_probability.detach(),
        "v552r4206_target_decomposition_error": r4206_target_decomposition_error.detach(),
        "v552r4206_teacher_overlap_rate": r4206_teacher_overlap_rate.detach(),
        "v552r4206_supervised_residual_fraction": r4206_supervised_residual_fraction.detach(),
        "v552r4206_unsupervised_residual_pixels": r4206_unsupervised_residual_pixels.detach(),
        "v552r4206_empty_residual_batch": r4206_empty_residual_batch.detach(),
        "v552r4206_legacy_independent_mask_bce_active": slot_masks.new_tensor(0.0 if r4206_enabled else 1.0),
        "v552r4207_rootfix_enabled": aux.get(
            "v552r4207_rootfix_enabled", slot_masks.new_zeros(())
        ).detach(),
        "v552r4207_seed_feature_finite_fraction": aux.get(
            "v552r4207_seed_feature_finite_fraction", slot_masks.new_ones(())
        ).detach(),
        "v552r4207_seed_valid_fraction": aux.get(
            "v552r4207_seed_valid_fraction", slot_masks.new_zeros(())
        ).detach(),
        "v552r4207_seed_score_mean": aux.get(
            "v552r4207_seed_score_mean", slot_masks.new_zeros(())
        ).detach(),
        "v552r4207_seed_pairwise_distance_px": aux.get(
            "v552r4207_seed_pairwise_distance_px", slot_masks.new_zeros(())
        ).detach(),
        "v552r4207_q0_pairwise_cosine": aux.get(
            "v552r4207_q0_pairwise_cosine", slot_masks.new_zeros(())
        ).detach(),
        "v552r4207_q1_pairwise_cosine": aux.get(
            "v552r4207_q1_pairwise_cosine", slot_masks.new_zeros(())
        ).detach(),
        "v552r4207_seed_to_slot_centroid_drift_px": aux.get(
            "v552r4207_seed_to_slot_centroid_drift_px", slot_masks.new_zeros(())
        ).detach(),
        "v552r4207_full_image_assignment_enabled": aux.get(
            "v552r4207_full_image_assignment_enabled", slot_masks.new_zeros(())
        ).detach(),
        "v552r4207_hard_spatial_support_used": aux.get(
            "v552r4207_hard_spatial_support_used", slot_masks.new_zeros(())
        ).detach(),
        "v552r4207_teacher_best_slot_collision_rate": r4207_teacher_best_slot_collision_rate.detach(),
        "v552r4207_teacher_best_slot_margin": r4207_teacher_best_slot_margin.detach(),
        "v552r4207_teacher_best_slot_dice": r4207_teacher_best_slot_dice.detach(),
        "v552r4208_rootfix_enabled": slot_masks.new_tensor(1.0 if r4208_requested else 0.0),
        "v552r4208_normalized_fusion_enabled": aux.get(
            "v552r4208_normalized_fusion_enabled", slot_masks.new_zeros(())
        ).detach(),
        "v552r4208_persistent_identity_enabled": aux.get(
            "v552r4208_persistent_identity_enabled", slot_masks.new_zeros(())
        ).detach(),
        "v552r4208_seed_consistent_matching_enabled": slot_masks.new_tensor(1.0 if r4208_seed_matching else 0.0),
        "v552r4208_learned_query_norm": aux.get("v552r4208_learned_query_norm", slot_masks.new_zeros(())).detach(),
        "v552r4208_seed_feature_norm": aux.get("v552r4208_seed_feature_norm", slot_masks.new_zeros(())).detach(),
        "v552r4208_seed_to_learned_norm_ratio": aux.get("v552r4208_seed_to_learned_norm_ratio", slot_masks.new_zeros(())).detach(),
        "v552r4208_seed_feature_pairwise_cosine": aux.get("v552r4208_seed_feature_pairwise_cosine", slot_masks.new_zeros(())).detach(),
        "v552r4208_q0_seed_identity_cosine": aux.get("v552r4208_q0_seed_identity_cosine", slot_masks.new_zeros(())).detach(),
        "v552r4208_q1_seed_identity_cosine": aux.get("v552r4208_q1_seed_identity_cosine", slot_masks.new_zeros(())).detach(),
        "v552r4208_identity_retention_delta": aux.get("v552r4208_identity_retention_delta", slot_masks.new_zeros(())).detach(),
        "v552r4208_seed_teacher_precision": r4208_seed_teacher_precision.detach(),
        "v552r4208_teacher_seed_coverage": r4208_teacher_seed_coverage.detach(),
        "v552r4208_seed_duplicate_teacher_rate": r4208_seed_duplicate_teacher_rate.detach(),
        "v552r4208_teacher_multi_seed_rate": r4208_teacher_multi_seed_rate.detach(),
        "v552r4208_seed_without_teacher_rate": r4208_seed_without_teacher_rate.detach(),
        "v552r4208_seed_occupancy_probability": r4208_seed_occupancy_probability.detach(),
        "v552r4208_seed_locked_fraction": r4208_seed_locked_fraction.detach(),
        "v552r4209_rootfix_enabled": slot_masks.new_tensor(1.0 if r4209_requested else 0.0),
        "v552r4209_center_peak_focal_enabled": slot_masks.new_tensor(1.0 if r4209_peak_focal else 0.0),
        "v552r4209_balanced_overflow_enabled": slot_masks.new_tensor(1.0 if r4209_balanced_overflow else 0.0),
        "v552r4209_balanced_overflow_loss": r4209_balanced_overflow_loss.detach(),
        "v552r4209_overflow_positive_loss": r4209_overflow_positive_loss.detach(),
        "v552r4209_overflow_negative_loss": r4209_overflow_negative_loss.detach(),
        "v552r4209_overflow_target_present": r4209_overflow_target_present.detach(),
        "v552r4209_m1_native_valid_fraction": aux.get(
            "v552r4209_m1_native_valid_fraction", slot_masks.new_zeros(())
        ).detach(),
        "v552r4209_m1_native_presence_mean": aux.get(
            "v552r4209_m1_native_presence_mean", slot_masks.new_zeros(())
        ).detach(),
        "v552r4210_rootfix_enabled": slot_masks.new_tensor(1.0 if r4210_requested else 0.0),
        "v552r4210_interior_anchor_enabled": slot_masks.new_tensor(1.0 if r4210_interior_anchor else 0.0),
        "v552r4210_variable_cardinality_seed_enabled": slot_masks.new_tensor(1.0 if r4210_variable_seed else 0.0),
        "v552r4210_independent_overflow_enabled": slot_masks.new_tensor(1.0 if r4210_independent_overflow else 0.0),
        "v552r4210_m1_native_alignment_enabled": slot_masks.new_tensor(1.0 if r4210_m1_native_alignment else 0.0),
        "v552r4210_bbox_center_inside_teacher_rate": r4210_bbox_center_inside_teacher_rate.detach(),
        "v552r4210_interior_anchor_inside_teacher_rate": r4210_interior_anchor_inside_rate.detach(),
        "v552r4210_bbox_to_interior_anchor_distance_px": r4210_bbox_to_interior_anchor_distance_px.detach(),
        "v552r4210_teacher_component_count": r4210_teacher_component_count.detach(),
        "v552r4210_valid_seed_count": r4210_valid_seed_count.detach(),
        "v552r4210_valid_seed_minus_teacher_count": r4210_valid_seed_minus_teacher_count.detach(),
        "v552r4210_true_seed_score_mean": r4210_true_seed_score_mean.detach(),
        "v552r4210_false_seed_score_mean": r4210_false_seed_score_mean.detach(),
        "v552r4210_true_seed_logit_mean": r4210_true_seed_logit_mean.detach(),
        "v552r4210_false_seed_logit_mean": r4210_false_seed_logit_mean.detach(),
        "v552r4210_independent_overflow_loss": r4210_independent_overflow_loss.detach(),
        "v552r4210_overflow_positive_loss": r4210_overflow_positive_loss.detach(),
        "v552r4210_overflow_negative_loss": r4210_overflow_negative_loss.detach(),
        "v552r4210_overflow_target_present": r4210_overflow_target_present.detach(),
        "v552r4210_overflow_conditional_mean": aux.get(
            "v552r4210_overflow_conditional_mean", slot_masks.new_zeros(())
        ).detach(),
        "v552r4210_overflow_logit_mean": aux.get(
            "v552r4210_overflow_logit_mean", slot_masks.new_zeros(())
        ).detach(),
        "v552r4210_m1_native_alignment_active": aux.get(
            "v552r4210_m1_native_alignment_active", slot_masks.new_zeros(())
        ).detach(),
        "v552r4210_m1_native_bce_loss": r4210_m1_native_bce_loss.detach(),
        "v552r4210_m1_native_dice_loss": r4210_m1_native_dice_loss.detach(),
        "v552r4210_m1_native_alignment_loss": r4210_m1_native_alignment_loss.detach(),
        "v552r4210_m1_native_soft_dice": r4210_m1_native_soft_dice.detach(),
        "v552r4211_rootfix_enabled": slot_masks.new_tensor(1.0 if r4211_requested else 0.0),
        "v552r4211_proposal_existence_decoupling_enabled": slot_masks.new_tensor(1.0 if r4211_proposal_existence else 0.0),
        "v552r4211_geometry_overflow_decoupling_enabled": slot_masks.new_tensor(1.0 if r4211_geometry_overflow else 0.0),
        "v552r4211_proposal_seed_count": r4211_proposal_seed_count,
        "v552r4211_proposal_confidence_mean": r4211_proposal_confidence_mean,
        "v552r4211_presence_expected_count": r4211_presence_expected_count,
        "v552r4211_presence_hard_count": r4211_presence_hard_count,
        "v552r4211_presence_minus_teacher_count": r4211_presence_minus_teacher_count,
        "v552r4211_geometry_effective_l1": r4211_geometry_effective_l1,
        "v552r4201_rootfix_enabled": slot_masks.new_tensor(1.0 if r4201_enabled else 0.0),
        "v552r4201_forward_teacher_built": slot_masks.new_tensor(1.0 if forward_teacher_built else 0.0),
        "v552r4201_teacher_valid_count": teacher_valid.to(slot_masks.dtype).sum().detach(),
        "v552r4201_teacher_error_fraction": effective_teacher_error.to(slot_masks.dtype).mean().detach(),
        "v538_unmatched_slot_loss": unmatched_suppression.detach(),
        "v538_action_loss": action_loss.detach(),
        "v540_polarity_loss": polarity_loss.detach(),
        "v540_dose_loss": dose_loss.detach(),
        "v540_mean_predicted_dose": slot_doses.mean().detach(),
        "v540_mean_teacher_dose": (
            best_teacher_dose[teacher_valid].mean().detach()
            if bool(teacher_valid.any().item()) else best_teacher_dose.new_zeros(())
        ),
        "v538_presence_loss": presence_loss.detach(),
        "v552r4212_rootfix_enabled": slot_masks.new_tensor(1.0 if r4212_requested else 0.0),
        "v552r4212_independent_candidate_set_enabled": slot_masks.new_tensor(1.0 if r4212_independent_set else 0.0),
        "v552r4212_existence_no_object_enabled": slot_masks.new_tensor(1.0 if r4212_existence_no_object else 0.0),
        # Forward-owned runtime diagnostics.  These three fields were omitted in
        # R4212 and caused E2-E5 to be falsely killed by the epoch contract.
        "v552r4212_visual_seed_identity_disabled": aux.get(
            "v552r4212_visual_seed_identity_disabled", slot_masks.new_zeros(())
        ).detach(),
        "v552r4212_deployment_candidate_count": aux.get(
            "v552r4212_deployment_candidate_count", slot_masks.new_zeros(())
        ).detach(),
        "v552r4212_independent_soft_overlap_mass": aux.get(
            "v552r4212_independent_soft_overlap_mass", slot_masks.new_zeros(())
        ).detach(),
        "v552r4212_presence_target_count": presence_target.detach().sum(dim=1).mean(),
        "clean_dynamic_component_set_enabled": slot_masks.new_tensor(1.0 if clean_dynamic_component_set else 0.0),
        "tc_drcs_enabled": slot_masks.new_tensor(1.0 if tc_drcs else 0.0),
        "tc_teacher_supervision_coverage": tc_teacher_supervision_coverage.detach(),
        "tc_retained_teacher_count": tc_retained_teacher_count.detach(),
        "tc_teacher_raw_count_mean": tc_teacher_raw_count_mean.detach(),
        "tc_teacher_overflow_rate": tc_teacher_overflow_rate.detach(),
        "tc_stage0_mask_loss": tc_stage0_mask_loss.detach(),
        "tc_pilot_mask_loss": tc_pilot_mask_loss.detach(),
        "tc_pilot_action_loss": tc_pilot_action_loss.detach(),
        "tc_pilot_teacher_consistency": tc_pilot_teacher_consistency.detach(),
        "tc_q0_pairwise_cosine": aux.get("v561_q0_pairwise_cosine", slot_masks.new_zeros(())).detach(),
        "tc_q1_pairwise_cosine": aux.get("v561_q1_pairwise_cosine", slot_masks.new_zeros(())).detach(),
        "tc_qfinal_pairwise_cosine": aux.get("v561_q2_pairwise_cosine", slot_masks.new_zeros(())).detach(),
        "clean_residual_target_mean": v562_residual_target_fraction.detach(),
        "clean_residual_probability_mean": v562_residual_probability_mean.detach(),
        "clean_residual_soft_dice": v562_residual_proposal_soft_dice.detach(),
        "clean_mask_bce": v560_standard_mask_bce.detach(),
        "clean_mask_dice_loss": v560_standard_mask_dice_loss.detach(),
        "clean_presence_bce": presence_loss.detach(),
        "clean_action_ce": action_loss.detach(),
        "clean_m2_signed_gain_mse": clean_m2_signed_gain_mse.detach(),
        "clean_uncertainty_weight_mask": clean_uncertainty_weight_mask.detach(),
        "clean_uncertainty_weight_presence": clean_uncertainty_weight_presence.detach(),
        "clean_uncertainty_weight_action": clean_uncertainty_weight_action.detach(),
        "clean_uncertainty_weight_residual": clean_uncertainty_weight_residual.detach(),
        "clean_anchor_inside_teacher_rate": v564_anchor_inside_teacher_rate.detach(),
        "clean_teacher_seed_coverage": v564_teacher_seed_coverage.detach(),
        "clean_duplicate_owner_rate": v564_duplicate_owner_rate.detach(),
        "clean_seed_without_teacher_rate": v564_seed_without_teacher_rate.detach(),
        "clean_unmatched_teacher_rate": v564_unmatched_teacher_rate.detach(),
        "clean_attention_precision_mean": aux.get("clean_attention_precision_mean", slot_masks.new_zeros(())).detach(),
        "clean_mask_precision_mean": aux.get("clean_mask_precision_mean", slot_masks.new_zeros(())).detach(),
        "clean_student_set_oracle_gain": student_set_oracle_gain_v561.detach(),
        "clean_component_oracle_gain": component_oracle_gain.detach(),
        "clean_component_capture_ratio": capture_ratio.detach(),
        "clean_component_purity": mean_purity.detach(),
        "clean_q2_pairwise_cosine": aux.get("v561_q2_pairwise_cosine", slot_masks.new_zeros(())).detach(),
        "v560_clean_core_enabled": slot_masks.new_tensor(1.0 if v560_clean_core else 0.0),
        "v560_forward_teacher_reused": slot_masks.new_tensor(1.0 if (v560_clean_core and use_r48_forward_teacher) else 0.0),
        "v560_factual_residual_fraction": v560_factual_residual_fraction.detach(),
        "v560_standard_mask_bce": v560_standard_mask_bce.detach(),
        "v560_standard_mask_dice_loss": v560_standard_mask_dice_loss.detach(),
        "v560_direct_mask_probability_mean": aux.get(
            "v560_direct_mask_probability_mean", slot_masks.new_zeros(())
        ).detach(),
        "v560_independent_soft_overlap_mass": aux.get(
            "v560_independent_soft_overlap_mass", slot_masks.new_zeros(())
        ).detach(),
        "v560_q0_pairwise_cosine": aux.get(
            "v560_q0_pairwise_cosine", slot_masks.new_zeros(())
        ).detach(),
        "v560_q1_pairwise_cosine": aux.get(
            "v560_q1_pairwise_cosine", slot_masks.new_zeros(())
        ).detach(),
        "v560_mask_bias_mean": aux.get(
            "v560_mask_bias_mean", slot_masks.new_zeros(())
        ).detach(),
        "v561_bcrs_enabled": aux.get("v561_bcrs_enabled", slot_masks.new_zeros(())).detach(),
        "v561_geometry_owner": aux.get("v561_geometry_owner", slot_masks.new_zeros(())).detach(),
        "v561_variant_static": aux.get("v561_variant_static", slot_masks.new_zeros(())).detach(),
        "v561_variant_image": aux.get("v561_variant_image", slot_masks.new_zeros(())).detach(),
        "v561_variant_typed": aux.get("v561_variant_typed", slot_masks.new_zeros(())).detach(),
        "v561_mask_probability_mean": aux.get("v561_mask_probability_mean", slot_masks.new_zeros(())).detach(),
        "v561_soft_overlap_mass": aux.get("v561_soft_overlap_mass", slot_masks.new_zeros(())).detach(),
        "v561_q0_pairwise_cosine": aux.get("v561_q0_pairwise_cosine", slot_masks.new_zeros(())).detach(),
        "v561_q1_pairwise_cosine": aux.get("v561_q1_pairwise_cosine", slot_masks.new_zeros(())).detach(),
        "v561_q2_pairwise_cosine": aux.get("v561_q2_pairwise_cosine", slot_masks.new_zeros(())).detach(),
        "v561_stage1_query_delta_norm": aux.get("v561_stage1_query_delta_norm", slot_masks.new_zeros(())).detach(),
        "v561_stage2_query_delta_norm": aux.get("v561_stage2_query_delta_norm", slot_masks.new_zeros(())).detach(),
        "v561_typed_support_mean": aux.get("v561_typed_support_mean", slot_masks.new_zeros(())).detach(),
        "v561_typed_support_std": aux.get("v561_typed_support_std", slot_masks.new_zeros(())).detach(),
        "v561_typed_support_neutrality_error": aux.get("v561_typed_support_neutrality_error", slot_masks.new_zeros(())).detach(),
        "v561_stage1_attention_entropy_ratio": aux.get("v561_stage1_attention_entropy_ratio", slot_masks.new_zeros(())).detach(),
        "v561_stage2_attention_entropy_ratio": aux.get("v561_stage2_attention_entropy_ratio", slot_masks.new_zeros(())).detach(),
        "v561_mask_bias_mean": aux.get("v561_mask_bias_mean", slot_masks.new_zeros(())).detach(),
        "v562_rootfix_enabled": slot_masks.new_tensor(1.0 if v562_rootfix else 0.0),
        "v562_query_owned_presence": aux.get("v562_query_owned_presence", slot_masks.new_zeros(())).detach(),
        "v562_direct_binary_executor": aux.get("v562_direct_binary_executor", slot_masks.new_zeros(())).detach(),
        "v562_residual_target_fraction": v562_residual_target_fraction.detach(),
        "v562_residual_probability_mean": v562_residual_probability_mean.detach(),
        "v562_residual_proposal_bce": v562_residual_proposal_bce.detach(),
        "v562_residual_proposal_dice_loss": v562_residual_proposal_dice_loss.detach(),
        "v562_residual_proposal_soft_dice": v562_residual_proposal_soft_dice.detach(),
        "v562_residual_proposal_loss": v562_residual_proposal_loss.detach(),
        "v562_local_execution_component_bce": v562_local_execution_component_bce.detach(),
        "v562_local_execution_ring_bce": v562_local_execution_ring_bce.detach(),
        "v562_local_execution_loss": v562_local_execution_loss.detach(),
        "v562_matched_add_flip_rate": v562_matched_add_flip_rate.detach(),
        "v562_matched_remove_flip_rate": v562_matched_remove_flip_rate.detach(),
        "v563_rootfix_enabled": slot_masks.new_tensor(1.0 if v563_rootfix else 0.0),
        "v564_rootfix_enabled": slot_masks.new_tensor(1.0 if v564_rootfix else 0.0),
        "v564_anchor_inside_teacher_rate": v564_anchor_inside_teacher_rate,
        "v564_teacher_seed_coverage": v564_teacher_seed_coverage,
        "v564_duplicate_owner_rate": v564_duplicate_owner_rate,
        "v564_seed_without_teacher_rate": v564_seed_without_teacher_rate,
        "v564_unmatched_teacher_rate": v564_unmatched_teacher_rate,
        "v564_infeasible_teacher_fraction": v564_infeasible_teacher_fraction,
        "v564_component_seed_bce": v564_component_seed_bce.detach(),
        "v565_rootfix_enabled": slot_masks.new_tensor(1.0 if v565_rootfix else 0.0),
        "v565_seed_heatmap_bce": v565_seed_heatmap_bce.detach(),
        "v565_seed_heatmap_dice_loss": v565_seed_heatmap_dice_loss.detach(),
        "v565_seed_heatmap_loss": v565_seed_heatmap_loss.detach(),
        "v565_seed_rank_loss": v565_seed_rank_loss.detach(),
        "v565_seed_target_fraction": v565_seed_target_fraction.detach(),
        "v565_seed_soft_dice": v565_seed_soft_dice.detach(),
        "v565_true_center_probability": v565_true_center_probability.detach(),
        "v565_false_peak_probability": v565_false_peak_probability.detach(),
        "v565_center_rank_accuracy": v565_center_rank_accuracy.detach(),
        "v565_seed_probability_mean": aux.get("v565_seed_probability_mean", slot_masks.new_zeros(())).detach(),
        "v565_relative_support_mean": aux.get("v565_relative_support_mean", slot_masks.new_zeros(())).detach(),
        "v565_attention_radius_mean": aux.get("v565_attention_radius_mean", slot_masks.new_zeros(())).detach(),
        "v565_peak_to_background_contrast_mean": aux.get("v565_peak_to_background_contrast_mean", slot_masks.new_zeros(())).detach(),
        "v565_shape_condition_abs_mean": aux.get("v565_shape_condition_abs_mean", slot_masks.new_zeros(())).detach(),
        "v564_proposal_radius_mean": aux.get("v564_proposal_radius_mean", slot_masks.new_zeros(())).detach(),
        "v564_proposal_shape_prior_abs_mean": aux.get("v564_proposal_shape_prior_abs_mean", slot_masks.new_zeros(())).detach(),
        "v564_dual_stream_identity_enabled": aux.get("v564_dual_stream_identity_enabled", slot_masks.new_zeros(())).detach(),
        "v564_typed_spatial_feedback_disabled": aux.get("v564_typed_spatial_feedback_disabled", slot_masks.new_zeros(())).detach(),
        "v563_local_mask_bce": v563_local_mask_bce.detach(),
        "v563_local_mask_dice_loss": v563_local_mask_dice_loss.detach(),
        "v563_outside_mask_loss": v563_outside_mask_loss.detach(),
        "v563_local_target_coverage": v563_local_target_coverage.detach(),
        "v563_pre_gate_mask_probability_mean": aux.get(
            "v563_pre_gate_mask_probability_mean", slot_masks.new_zeros(())
        ).detach(),
        "v563_outside_mask_probability": aux.get(
            "v563_outside_mask_probability", slot_masks.new_zeros(())
        ).detach(),
        "v563_attention_window_fraction": aux.get(
            "v563_attention_window_fraction", slot_masks.new_zeros(())
        ).detach(),
        "v563_mask_window_fraction": aux.get(
            "v563_mask_window_fraction", slot_masks.new_zeros(())
        ).detach(),
        "v563_identity_retention_q1": aux.get(
            "v563_identity_retention_q1", slot_masks.new_zeros(())
        ).detach(),
        "v563_identity_retention_q2": aux.get(
            "v563_identity_retention_q2", slot_masks.new_zeros(())
        ).detach(),
        "v561_teacher_add_count": (
            (teacher_valid & ((teacher_actions % 2) == 1)).to(slot_masks.dtype).sum(dim=1).mean().detach()
            if v561_bcrs else slot_masks.new_zeros(())
        ),
        "v561_teacher_remove_count": (
            (teacher_valid & ((teacher_actions % 2) == 0)).to(slot_masks.dtype).sum(dim=1).mean().detach()
            if v561_bcrs else slot_masks.new_zeros(())
        ),
        "v561_pair_action_probability": (
            pair_action_probability_v561[teacher_valid[:, None, :].expand_as(pair_action_probability_v561)].mean().detach()
            if (v561_bcrs and bool(teacher_valid.any().item())) else slot_masks.new_zeros(())
        ),
        "v552r4212_candidate_alignment_enabled": slot_masks.new_tensor(1.0 if r4212_candidate_alignment else 0.0),
        "v552r4212_candidate_alignment_bce": r4212_candidate_alignment_bce.detach(),
        "v552r4212_candidate_alignment_dice_loss": r4212_candidate_alignment_dice_loss.detach(),
        "v552r4212_candidate_alignment_soft_dice": r4212_candidate_alignment_soft_dice.detach(),
        "v552r4212_candidate_alignment_loss": r4212_candidate_alignment_loss.detach(),
        "v538_soft_gain_loss": soft_gain_loss.detach(),
        "v538_harm_loss": harm_loss.detach(),
        "v538_correct_damage_loss": outside_damage.detach(),
        "v545_exact_utility_alignment_enabled": gain_scores.new_tensor(
            1.0 if use_exact_utility_alignment else 0.0
        ),
        "v545_matched_locality_enabled": gain_scores.new_tensor(
            1.0 if use_matched_locality else 0.0
        ),
        "v545_utility_gain_mean": (
            utility_gain[matched].mean().detach()
            if bool(matched.any().item()) else utility_gain.new_zeros(())
        ),
        "v545_exact_regret_loss": soft_gain_loss.detach(),
        "v545_hard_mask_area_fraction": (
            hard_masks_st.detach().mean(dim=(-2, -1))[matched].mean()
            if bool(matched.any().item()) else hard_masks_st.new_zeros(())
        ),
        "v545_mask_contrast_active_rate": mask_contrast_active.float().mean().detach(),
        "v538_diversity_loss": diversity_loss.detach(),
        "v538_gain_regression_loss": gain_regression.detach(),
        "v538_listwise_loss": listwise_loss.detach(),
        "v538_preserve_loss": preserve_loss.detach(),
        "v541_benefit_loss": benefit_loss.detach(),
        "v541_harm_classification_loss": harm_classification_loss.detach(),
        "v541_gain_nll_loss": gain_regression.detach(),
        "v541_pairwise_rank_loss": pairwise_rank_loss.detach(),
        "v541_preserve_decision_loss": preserve_loss.detach(),
        "v542_gain_huber_benefit": gain_huber_benefit.detach(),
        "v542_gain_huber_harm": gain_huber_harm.detach(),
        "v542_gain_huber_neutral": gain_huber_neutral.detach(),
        "v542_gain_regression_loss": gain_regression.detach(),
        "v542_benefit_sign_loss": benefit_sign_loss.detach(),
        "v542_harm_sign_loss": harm_sign_loss.detach(),
        "v542_gain_sign_loss": gain_sign_loss.detach(),
        "v542_benefit_gain_positive_rate": benefit_gain_positive_rate.detach(),
        "v542_harm_gain_negative_rate": harm_gain_negative_rate.detach(),
        "v542_gain_sign_accuracy": gain_sign_accuracy.detach(),
        "v543_factorized_gain_enabled": gain_scores.new_ones(()),
        "v544_minimal_m2_objective_enabled": gain_scores.new_tensor(
            1.0 if minimal_m2_objective else 0.0
        ),
        "v545_class_complete_outcome_enabled": gain_scores.new_tensor(
            1.0 if class_complete_outcome else 0.0
        ),
        "v546_optimal_matching_enabled": gain_scores.new_tensor(
            1.0 if use_optimal_matching else 0.0
        ),
        "v546_greedy_matching_mean_dice": greedy_matching_dice.detach(),
        "v546_optimal_matching_mean_dice": optimal_matching_dice.detach(),
        "v546_matching_gain": (
            optimal_matching_dice - greedy_matching_dice
        ).detach(),
        "v546_slot_competition_enabled": aux.get(
            "v546_slot_competition_enabled",
            aux.get("v538_slot_competition_enabled", gain_scores.new_zeros(())),
        ).detach(),
        "v546_raw_slot_overlap_mass": aux.get(
            "v546_raw_slot_overlap_mass",
            aux.get("v538_slot_raw_overlap_mass", gain_scores.new_zeros(())),
        ).detach(),
        "v546_competition_overlap_mass": aux.get(
            "v546_competition_overlap_mass",
            aux.get("v538_slot_competition_overlap_mass", gain_scores.new_zeros(())),
        ).detach(),
        "v546_streaming_balanced_softmax_enabled": gain_scores.new_tensor(
            1.0 if streaming_balanced_softmax else 0.0
        ),
        "v546_streaming_balanced_softmax_loss": (
            streaming_balanced_softmax_loss.detach()
        ),
        "v546_streaming_neutral_prior": streaming_outcome_prior[0].detach(),
        "v546_streaming_benefit_prior": streaming_outcome_prior[1].detach(),
        "v546_streaming_harm_prior": streaming_outcome_prior[2].detach(),
        "v545_editability_loss": editability_loss.detach(),
        "v545_direction_loss": direction_loss.detach(),
        "v545_direction_update_active": gain_scores.new_tensor(
            1.0 if direction_update_active else 0.0
        ),
        "v544_benefit_total_count": benefit_total_count.detach(),
        "v544_harm_total_count": harm_total_count.detach(),
        "v544_neutral_total_count": neutral_total_count.detach(),
        "v544_benefit_gain_positive_count": (
            benefit_gain_positive_count.detach()
        ),
        "v544_harm_gain_negative_count": (
            harm_gain_negative_count.detach()
        ),
        "v544_neutral_outcome_correct_count": (
            neutral_outcome_correct_count.detach()
        ),
        "v544_benefit_outcome_correct_count": (
            benefit_outcome_correct_count.detach()
        ),
        "v544_harm_outcome_correct_count": (
            harm_outcome_correct_count.detach()
        ),
        "v544_benefit_probability_sum": benefit_probability_sum.detach(),
        "v544_nonbenefit_probability_sum": (
            nonbenefit_probability_sum.detach()
        ),
        "v544_nonbenefit_total_count": nonbenefit_total_count.detach(),
        "v544_signed_outcome_benefit_sum": (
            signed_outcome_benefit_sum.detach()
        ),
        "v544_signed_outcome_harm_sum": (
            signed_outcome_harm_sum.detach()
        ),
        "v544_zero_benefit_batch": zero_benefit_batch.detach(),
        "v544_replay_case_weight": gain_scores.new_tensor(
            replay_case_weight
        ),
        "v544_real_benefit_count": (
            positive_slot & (~replay_case[:, None])
        ).float().sum().detach(),
        "v544_replay_benefit_count": (
            positive_slot & replay_case[:, None]
        ).float().sum().detach(),
        "v544_mask_oracle_gain": v544_mask_oracle_gain.detach(),
        "v544_polarity_oracle_gain": (
            v544_polarity_oracle_gain.detach()
        ),
        "v544_full_oracle_gain": v544_full_oracle_gain.detach(),
        "v543_neutral_outcome_loss": neutral_outcome_loss.detach(),
        "v543_outcome_classification_loss": (
            outcome_classification_loss.detach()
        ),
        "v543_balanced_sign_accuracy": (
            0.5 * (benefit_gain_positive_rate + harm_gain_negative_rate)
        ).detach(),
        "v543_true_benefit_separation": (
            benefit_probability_positive_mean
            - benefit_probability_negative_mean
        ).detach(),
        "v543_outcome_balanced_accuracy": (
            v543_outcome_balanced_accuracy.detach()
        ),
        "v543_semantic_consistency_rate": (
            (
                gain_scores
                * aux.get(
                    "v543_slot_signed_outcome",
                    benefit_probs - harm_probs,
                )
                >= -1.0e-12
            )[slot_valid]
            .float()
            .mean()
            if bool(slot_valid.any().item())
            else gain_scores.new_ones(())
        ).detach(),
        "v543_gain_magnitude_mean": (
            gain_magnitude_normalized[slot_valid].mean()
            if bool(slot_valid.any().item())
            else gain_magnitude_normalized.new_zeros(())
        ).detach(),
        "v543_mask_pos_weight_mean": (
            mask_pos_weight[matched].mean()
            if bool(matched.any().item())
            else mask_pos_weight.new_zeros(())
        ).detach(),
        "v542_learned_uncertainty_enabled": gain_scores.new_tensor(
            1.0 if bool(_m1(cfg, "V542_USE_LEARNED_UNCERTAINTY", False)) else 0.0
        ),
        "v541_decision_supervision_active": gain_scores.new_tensor(1.0),
        "v541_candidate_outcome_selector_enabled": gain_scores.new_tensor(
            1.0 if v541_selector_enabled else 0.0
        ),
        "v541_exact_candidate_gain_mean": hard_gain.mean().detach(),
        "v541_gain_margin_mean": gain_margin_case.mean().detach(),
        "v541_benefit_target_rate": positive_slot.float().sum().detach()
            / slot_valid.float().sum().clamp_min(1.0),
        "v541_harm_target_rate": negative_slot.float().sum().detach()
            / slot_valid.float().sum().clamp_min(1.0),
        "v541_neutral_target_rate": neutral_slot.float().sum().detach()
            / slot_valid.float().sum().clamp_min(1.0),
        # 下面两项仍表示“通过部署阈值”的比例。
        "v541_predicted_benefit_rate": (
            (benefit_probs >= deploy_benefit) & slot_valid
        ).float().sum().detach() / valid_count,

        "v541_predicted_harm_rate": (
            (harm_probs >= deploy_harm) & slot_valid
        ).float().sum().detach() / valid_count,

        # 常规分类与条件分离诊断。
        "v541_benefit_probability_mean": (
            benefit_probability_mean.detach()
        ),
        "v541_benefit_probability_positive_mean": (
            benefit_probability_positive_mean.detach()
        ),
        "v541_benefit_probability_negative_mean": (
            benefit_probability_negative_mean.detach()
        ),
        "v541_benefit_prediction_rate_at_05": (
            ((benefit_probs >= 0.5) & slot_valid)
            .float()
            .sum()
            .detach()
            / valid_count
        ),

        "v541_harm_probability_mean": (
            harm_probability_mean.detach()
        ),
        "v541_harm_probability_positive_mean": (
            harm_probability_positive_mean.detach()
        ),
        "v541_harm_probability_negative_mean": (
            harm_probability_negative_mean.detach()
        ),
        "v541_harm_prediction_rate_at_05": (
            ((harm_probs >= 0.5) & slot_valid)
            .float()
            .sum()
            .detach()
            / valid_count
        ),

        "v541_rank_pair_accuracy": rank_pair_accuracy.detach(),
        "v541_gain_mean_on_benefit": gain_mean_on_benefit.detach(),
        "v541_gain_lcb_on_benefit": gain_lcb_on_benefit.detach(),
        "v541_gain_mean_on_harm": gain_mean_on_harm.detach(),
        "v541_gain_nll_shift": gain_nll_shift.detach(),

        "v541_gain_lcb_mean": (
            gain_lcb[slot_valid].mean().detach()
            if bool(slot_valid.any().item())
            else gain_lcb.new_zeros(())
        ),
        "v541_shadow_composer_gain": shadow_selected_gain.detach(),
        "v541_shadow_composer_execute_rate": shadow_execute.float().mean().detach(),
        "v541_shadow_composer_precision": shadow_precision.detach(),
        "v541_shadow_composer_improved_case_rate": shadow_improved_rate.detach(),
        "v541_shadow_composer_harmful_case_rate": shadow_harmful_rate.detach(),
        # V549 epoch-global evidence.  Counts are summed by train.py rather than
        # averaging per-batch booleans, so zero execution cannot masquerade as
        # safety and the persistent gate is based on actual edited cases.
        "v549_shadow_execute_count": shadow_execute.float().sum().detach(),
        "v549_shadow_improved_count": shadow_positive.float().sum().detach(),
        "v549_shadow_harmful_count": shadow_harmful.float().sum().detach(),
        "v549_shadow_selected_gain_sum": torch.where(
            shadow_execute, shadow_gain, torch.zeros_like(shadow_gain)
        ).sum().detach(),
        "v549_shadow_case_count": shadow_execute.new_tensor(
            float(shadow_execute.numel()), dtype=gain_scores.dtype
        ),
        "v552r44_audit_execute_count": audit_execute.float().sum().detach(),
        "v552r44_audit_improved_count": audit_positive.float().sum().detach(),
        "v552r44_audit_harmful_count": audit_harmful.float().sum().detach(),
        "v552r44_audit_selected_gain_sum": torch.where(
            audit_execute, audit_gain, torch.zeros_like(audit_gain)
        ).sum().detach(),
        "v552r44_audit_case_count": audit_execute.new_tensor(
            float(audit_execute.numel()), dtype=gain_scores.dtype
        ),
        "v552r44_audit_execute_rate": audit_execute.float().mean().detach(),
        "v552r44_audit_precision": audit_precision.detach(),
        "v552r44_audit_harm_rate": audit_harmful.float().sum().detach()
            / audit_execute.float().sum().clamp_min(1.0),
        "v552r44_audit_mean_gain": audit_selected_gain.detach(),
        "v552r44_policy_audit_execute_count": (
            policy_audit_execute.float().sum().detach()
        ),
        "v552r44_policy_audit_improved_count": (
            policy_audit_positive.float().sum().detach()
        ),
        "v552r44_policy_audit_harmful_count": (
            policy_audit_harmful.float().sum().detach()
        ),
        "v552r44_policy_audit_selected_gain_sum": torch.where(
            policy_audit_execute,
            policy_audit_gain,
            torch.zeros_like(policy_audit_gain),
        ).sum().detach(),
        "v552r44_policy_audit_case_count": policy_audit_execute.new_tensor(
            float(policy_audit_execute.numel()), dtype=gain_scores.dtype
        ),
        "v552r44_policy_audit_execute_rate": (
            policy_audit_execute.float().mean().detach()
        ),
        "v552r44_policy_audit_precision": policy_audit_precision.detach(),
        "v552r44_policy_audit_harm_rate": (
            policy_audit_harmful.float().sum().detach()
            / policy_audit_execute.float().sum().clamp_min(1.0)
        ),
        "v552r44_policy_audit_mean_gain": (
            policy_audit_selected_gain.detach()
        ),
        "v552r46_formal_policy_audit_execute_count": (
            policy_audit_execute.float().sum().detach()
        ),
        "v552r46_formal_policy_audit_improved_count": (
            policy_audit_positive.float().sum().detach()
        ),
        "v552r46_formal_policy_audit_harmful_count": (
            policy_audit_harmful.float().sum().detach()
        ),
        "v552r46_formal_policy_audit_selected_gain_sum": torch.where(
            policy_audit_execute, policy_audit_gain, torch.zeros_like(policy_audit_gain)
        ).sum().detach(),
        "v552r46_formal_policy_audit_case_count": policy_audit_execute.new_tensor(
            float(policy_audit_execute.numel()), dtype=gain_scores.dtype
        ),
        "v552r46_formal_policy_audit_execute_rate": (
            policy_audit_execute.float().mean().detach()
        ),
        "v552r46_formal_policy_audit_precision": policy_audit_precision.detach(),
        "v552r46_formal_policy_audit_harm_rate": (
            policy_audit_harmful.float().sum().detach()
            / policy_audit_execute.float().sum().clamp_min(1.0)
        ),
        "v552r46_formal_policy_audit_mean_gain": policy_audit_selected_gain.detach(),
        "v552r46_teacher_realization_ratio": teacher_realization_ratio.detach(),
        "v552r46_native_state_only_enabled": gain_scores.new_tensor(
            1.0 if r46_enabled else 0.0
        ),
        "v549_factorized_deployment_enabled": gain_scores.new_tensor(
            1.0 if factorized_deployment else 0.0
        ),
        "v549_presence_gate_pass_rate": (
            ((presence_probs.detach() >= deploy_presence) & deploy_slot_valid).float().sum()
            / deploy_slot_valid.float().sum().clamp_min(1.0)
        ).detach(),
        "v549_editability_gate_pass_rate": (
            ((editability_probs.detach() >= float(
                _m1(cfg, "V549_DEPLOY_EDITABILITY_THRESHOLD", 0.50)
            )) & deploy_slot_valid).float().sum()
            / deploy_slot_valid.float().sum().clamp_min(1.0)
        ).detach(),
        "v549_direction_gate_pass_rate": (
            ((direction_probs.detach() >= float(
                _m1(cfg, "V549_DEPLOY_DIRECTION_THRESHOLD", 0.50)
            )) & deploy_slot_valid).float().sum()
            / deploy_slot_valid.float().sum().clamp_min(1.0)
        ).detach(),
        "v549_positive_gain_gate_pass_rate": (
            ((gain_lcb.detach() > deploy_gain) & slot_valid).float().sum()
            / slot_valid.float().sum().clamp_min(1.0)
        ).detach(),
        "v549_exact_eligible_slot_rate": (
            eligible.float().sum() / deploy_slot_valid.float().sum().clamp_min(1.0)
        ).detach(),
        "v541_selector_precision_ready": gain_scores.new_tensor(
            1.0 if selector_precision_ready else 0.0
        ),
        "v541_selector_gain_ready": gain_scores.new_tensor(
            1.0 if selector_gain_ready else 0.0
        ),
        "v541_selector_harm_ready": gain_scores.new_tensor(
            1.0 if selector_harm_ready else 0.0
        ),
        "v541_selector_balance_ready": gain_scores.new_tensor(
            1.0 if selector_balance_ready else 0.0
        ),
        "v538_teacher_component_count": teacher_valid.float().sum(dim=1).mean().detach(),
        "v538_matched_slot_rate": matched.float().mean().detach(),
        "v538_replay_case_rate": replay_case.float().mean().detach(),
        "v538_soft_gain_mean": utility_gain.mean().detach(),
        "v538_hard_gain_mean": hard_gain.mean().detach(),
        "v538_component_oracle_gain": component_oracle_gain.detach(),
        "v561_teacher_set_oracle_gain": teacher_set_oracle_gain_v561.detach(),
        "v561_student_set_oracle_gain": student_set_oracle_gain_v561.detach(),
        "v561_set_oracle_realization_ratio": (
            student_set_oracle_gain_v561 / teacher_set_oracle_gain_v561.clamp_min(EPS)
        ).clamp(0.0, 10.0).detach() if v561_bcrs else slot_masks.new_zeros(()),
        "v538_teacher_component_oracle_gain": teacher_oracle_gain.detach(),
        "v538_action_realizable_teacher_oracle_gain": teacher_oracle_gain.detach(),
        "v538_ideal_mask_teacher_oracle_gain": ideal_teacher_oracle_gain.detach(),
        "v538_teacher_action_realizable_rate": teacher_positive_rate.detach(),
        "v538_real_teacher_oracle_gain": real_teacher_oracle_gain.detach(),
        "v538_replay_teacher_oracle_gain": replay_teacher_oracle_gain.detach(),
        "v538_real_component_purity": real_component_purity.detach(),
        "v538_replay_component_purity": replay_component_purity.detach(),
        "v538_effective_teacher_error_fraction": teacher_error.float().mean().detach(),
        "v538_component_capture_ratio": capture_ratio.detach(),
        "v540_adaptive_oracle_threshold": gain_scores.new_tensor(adaptive_oracle_threshold),
        "v540_teacher_oracle_ready": gain_scores.new_tensor(1.0 if teacher_ready else 0.0),
        "v540_component_oracle_ready": gain_scores.new_tensor(1.0 if oracle_ready else 0.0),
        "v540_capture_ready": gain_scores.new_tensor(1.0 if capture_ready else 0.0),
        "v540_continuous_dose_enabled": gain_scores.new_tensor(
            1.0 if continuous_dose_enabled else 0.0
        ),
        "v538_positive_component_rate": positive_rate.detach(),
        "v538_component_purity": mean_purity.detach(),
        "v538_target_execute_case_rate": positive_case.float().mean().detach(),
        "v538_predicted_execute_case_rate": predicted_execute.float().mean().detach(),
        "v538_execute_precision": precision.detach(),
        "v538_execute_recall": recall.detach(),
        "v538_selected_gain": selected_gain.mean().detach(),
        "v538_improved_case_rate": selected_positive.float().mean().detach(),
        "v538_harmful_case_rate": selected_harmful.float().mean().detach(),
        "v538_m2_ready": ready_scale.detach(),
        "v538_m1_train_scale": m1_train_scale.detach(),
        "v538_m2_train_scale": m2_train_scale.detach(),
        "v552_m2_supervision_scale": m2_supervision_scale.detach(),
        "v552_m2_execution_scale": m2_execution_scale.detach(),
        "v552_unified_deployment_gate": gain_scores.new_tensor(
            1.0 if unified_gate else 0.0
        ),
        "v552_benefit_gate_pass_rate": (
            ((benefit_probs.detach() >= deploy_benefit) & deploy_slot_valid).float().sum()
            / deploy_slot_valid.float().sum().clamp_min(1.0)
        ).detach(),
        "v552_harm_gate_pass_rate": (
            ((harm_probs.detach() <= deploy_harm) & deploy_slot_valid).float().sum()
            / deploy_slot_valid.float().sum().clamp_min(1.0)
        ).detach(),
        "v552_benefit_dominance_pass_rate": (
            ((benefit_probs.detach() >= harm_probs.detach() + benefit_harm_margin) & deploy_slot_valid).float().sum()
            / deploy_slot_valid.float().sum().clamp_min(1.0)
        ).detach(),
        "v550_candidate_quality_routing_enabled": gain_scores.new_tensor(
            1.0 if v550_quality_routing_enabled else 0.0
        ),
        "v550_candidate_quality_score": gain_scores.new_tensor(
            float(v550_candidate_quality_score)
        ),
        "v550_quality_route_multiplier": gain_scores.new_tensor(
            float(v550_quality_route_multiplier)
        ),
        "v538_positive_case_weight": positive_case_weight.detach(),
    }
    return m1_objective, m2_objective, diagnostics
