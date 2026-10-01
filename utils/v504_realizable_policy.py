"""V504 candidate-realizable potential-outcome supervision.

The factual error geometry is still defined only by detached Base and GT.  A
candidate is allowed to become an action target only when its *actual deployed
pixel value* repairs that factual error.  Candidate predictions therefore do
not define where they are evaluated; they only define the potential outcome of
an already-defined intervention.
"""
from __future__ import annotations

from typing import Dict

import torch

from utils.v503_factual_atomic_causal import EPS, _as_b1hw, build_factual_cause_targets


@torch.no_grad()
def build_realizable_action_targets(
    c0_prob: torch.Tensor,
    candidate_probs: torch.Tensor,
    gt: torch.Tensor,
    *,
    boundary_radius: int = 2,
    failure_dice: float = 0.55,
) -> Dict[str, torch.Tensor]:
    """Return a safe action teacher grounded in realized candidate outcomes."""
    c0 = _as_b1hw(c0_prob).detach().clamp(EPS, 1.0 - EPS)
    y = (_as_b1hw(gt) >= 0.5).float()
    if candidate_probs.ndim != 4:
        raise ValueError(
            f"candidate_probs must be [B,K,H,W], got {tuple(candidate_probs.shape)}"
        )
    candidates = candidate_probs.detach().clamp(EPS, 1.0 - EPS)
    if candidates.shape[-2:] != c0.shape[-2:]:
        raise ValueError("candidate_probs and c0_prob must share spatial size")

    factual = build_factual_cause_targets(
        c0,
        y,
        boundary_radius=boundary_radius,
        failure_dice=failure_dice,
        include_global_action=False,
    )
    causes = factual["cause_targets"]
    factual_error = factual["factual_error"]
    failure = factual["failure_target"][:, :, None, None]

    batch, count, height, width = candidates.shape
    compatibility = candidates.new_zeros((batch, count, height, width))
    local_count = min(4, count)
    compatibility[:, :local_count] = causes[:, :local_count]
    if count > 4:
        compatibility[:, 4:] = factual_error * failure

    c0_hard = c0 >= 0.5
    cand_hard = candidates >= 0.5
    y_hard = y >= 0.5
    base_wrong = c0_hard != y_hard
    base_right = ~base_wrong

    repairs = (
        base_wrong.expand(-1, count, -1, -1)
        & (cand_hard == y_hard.expand(-1, count, -1, -1))
    ).float()
    harms = (
        base_right.expand(-1, count, -1, -1)
        & (cand_hard != y_hard.expand(-1, count, -1, -1))
    ).float()

    base_abs_error = (c0 - y).abs()
    candidate_abs_error = (candidates - y.expand(-1, count, -1, -1)).abs()
    signed_gain = base_abs_error.expand(-1, count, -1, -1) - candidate_abs_error
    positive_soft_gain = signed_gain.clamp_min(0.0)

    executable = compatibility * repairs
    ranking_score = positive_soft_gain.masked_fill(executable <= 0.5, -1.0e6)
    _, best_index = ranking_score.max(dim=1)
    has_action = executable.amax(dim=1) > 0.5
    action_target = torch.where(
        has_action,
        best_index + 1,
        torch.zeros_like(best_index),
    ).long()

    selected_repair = torch.gather(repairs, 1, best_index[:, None])[:, 0] * has_action.float()
    selected_gain = torch.gather(positive_soft_gain, 1, best_index[:, None])[:, 0] * has_action.float()

    return {
        **factual,
        "compatibility_masks": compatibility.detach(),
        "candidate_repair_targets": repairs.detach(),
        "candidate_harm_targets": harms.detach(),
        "candidate_signed_gain_targets": signed_gain.detach(),
        "candidate_positive_gain_targets": positive_soft_gain.detach(),
        "candidate_executable_targets": executable.detach(),
        "action_target": action_target.detach(),
        "has_action_target": has_action.float()[:, None].detach(),
        "selected_repair_target": selected_repair[:, None].detach(),
        "selected_gain_target": selected_gain[:, None].detach(),
        "realizable_error_fraction": (
            executable.amax(dim=1, keepdim=True) * factual_error
        ).sum() / factual_error.sum().clamp_min(1.0),
        "candidate_harm_fraction": (
            harms.sum()
            / (cand_hard != c0_hard.expand(-1, count, -1, -1))
            .float()
            .sum()
            .clamp_min(1.0)
        ),
    }
