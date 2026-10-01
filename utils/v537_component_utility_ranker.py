"""V537 candidate-level component utility ranking.

The deployment unit is one typed connected component.  All components produced
by the four monotone M1 actions are embedded independently and compared with an
explicit Preserve candidate whose utility is exactly zero.

Connected-component extraction is deliberately detached.  The ranker is
trained by explicit component-level Dice-gain supervision, so no straight-
through surrogate is used for the discrete component operation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy import ndimage
except Exception as exc:  # pragma: no cover
    ndimage = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None

EPS = 1.0e-6


@dataclass(frozen=True)
class ComponentBankStats:
    raw_components: int
    retained_components: int


def _masked_mean(feature: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pool [B,C,H,W] with [B,K,H,W] masks -> [B,K,C]."""
    numerator = (feature[:, None] * mask[:, :, None]).sum(dim=(-2, -1))
    denominator = mask.sum(dim=(-2, -1)).unsqueeze(-1).clamp_min(1.0)
    return numerator / denominator


def _masked_scalar_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pool [B,K,H,W] values with [B,K,H,W] masks -> [B,K,1]."""
    numerator = (value * mask).sum(dim=(-2, -1)).unsqueeze(-1)
    denominator = mask.sum(dim=(-2, -1)).unsqueeze(-1).clamp_min(1.0)
    return numerator / denominator


def _masked_scalar_max(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = value.masked_fill(mask <= 0.5, -1.0e4)
    maximum = masked.amax(dim=(-2, -1)).unsqueeze(-1)
    return torch.where(maximum < -1.0e3, torch.zeros_like(maximum), maximum)


def extract_typed_component_bank(
    *,
    action_candidates: torch.Tensor,
    base_probability: torch.Tensor,
    cause_probability: torch.Tensor,
    min_pixels: int,
    max_components: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, ComponentBankStats]:
    """Extract a fixed-width bank of typed hard-change components.

    Returns:
      masks: [B,K,H,W] detached float masks;
      actions: [B,K] zero-based action indices;
      valid: [B,K] validity mask;
      support: [B,K] detached M1 cause support used only for truncation;
      stats: aggregate extraction counts.
    """
    if ndimage is None:
        raise RuntimeError(
            "V537 component ranker requires scipy.ndimage; original error: "
            + repr(_SCIPY_IMPORT_ERROR)
        )
    if action_candidates.ndim != 4 or action_candidates.shape[1] != 4:
        raise ValueError(
            f"action_candidates must be [B,4,H,W], got {tuple(action_candidates.shape)}"
        )
    if cause_probability.shape != action_candidates.shape:
        raise ValueError(
            "cause_probability must match action_candidates, got "
            f"{tuple(cause_probability.shape)} vs {tuple(action_candidates.shape)}"
        )

    b, _, h, w = action_candidates.shape
    k = max(int(max_components), 1)
    min_pixels = max(int(min_pixels), 1)
    device = action_candidates.device
    dtype = action_candidates.dtype

    masks = torch.zeros((b, k, h, w), device=device, dtype=dtype)
    actions = torch.zeros((b, k), device=device, dtype=torch.long)
    valid = torch.zeros((b, k), device=device, dtype=torch.bool)
    support = torch.zeros((b, k), device=device, dtype=dtype)

    base_np = (base_probability.detach().float().cpu().numpy()[:, 0] >= 0.5)
    candidate_np = action_candidates.detach().float().cpu().numpy() >= 0.5
    cause_np = cause_probability.detach().float().cpu().numpy()
    structure = np.ones((3, 3), dtype=np.uint8)
    raw_total = 0
    retained_total = 0

    for sample in range(b):
        entries = []
        for action in range(4):
            changed = np.logical_xor(candidate_np[sample, action], base_np[sample])
            labels, count = ndimage.label(changed, structure=structure)
            for label_id in range(1, count + 1):
                component = labels == label_id
                area = int(component.sum())
                if area < min_pixels:
                    continue
                raw_total += 1
                values = cause_np[sample, action][component]
                cause_mean = float(values.mean()) if values.size else 0.0
                cause_max = float(values.max()) if values.size else 0.0
                # Retain small but strongly supported components as well as large
                # coherent components.  This score is only a truncation guard;
                # the learnable ranker decides deployment.
                prefilter = cause_mean + 0.25 * cause_max + 0.01 * np.log1p(area)
                entries.append((prefilter, area, action, component, cause_mean))

        entries.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for slot, (_, _, action, component, cause_mean) in enumerate(entries[:k]):
            masks[sample, slot] = torch.from_numpy(component.astype(np.float32)).to(
                device=device, dtype=dtype
            )
            actions[sample, slot] = int(action)
            valid[sample, slot] = True
            support[sample, slot] = float(cause_mean)
            retained_total += 1

    return masks.detach(), actions.detach(), valid.detach(), support.detach(), ComponentBankStats(
        raw_components=raw_total,
        retained_components=retained_total,
    )


class V537ComponentUtilityRanker(nn.Module):
    """Predict case-level Dice gain for every typed connected component."""

    def __init__(
        self,
        *,
        feature_channels: int,
        hidden_dim: int = 128,
        dropout: float = 0.10,
        min_pixels: int = 4,
        max_components: int = 48,
        initial_score_bias: float = -0.002,
    ) -> None:
        super().__init__()
        self.min_pixels = max(int(min_pixels), 1)
        self.max_components = max(int(max_components), 1)
        feature_channels = int(feature_channels)
        hidden_dim = int(hidden_dim)
        # inside, ring, cause mean/max, alpha mean/max, area, entropy,
        # boundary support, Base probability and four-dimensional action type.
        input_dim = 2 * feature_channels + 12
        self.scorer = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.constant_(self.scorer[-1].bias, float(initial_score_bias))

    def forward(
        self,
        *,
        feature: torch.Tensor,
        action_candidates: torch.Tensor,
        base_probability: torch.Tensor,
        cause_probability: torch.Tensor,
        action_alpha: torch.Tensor,
        entropy: torch.Tensor,
        boundary: torch.Tensor,
        deploy_enabled: bool,
    ) -> Dict[str, torch.Tensor]:
        masks, actions, valid, prefilter_support, bank_stats = extract_typed_component_bank(
            action_candidates=action_candidates,
            base_probability=base_probability,
            cause_probability=cause_probability,
            min_pixels=self.min_pixels,
            max_components=self.max_components,
        )
        b, k, h, w = masks.shape
        flat_masks = masks.reshape(b * k, 1, h, w)
        dilated = F.max_pool2d(flat_masks, kernel_size=3, stride=1, padding=1)
        rings = ((dilated > 0.5) & (flat_masks <= 0.5)).to(masks.dtype).reshape(b, k, h, w)

        inside_feature = _masked_mean(feature, masks)
        ring_feature = _masked_mean(feature, rings)

        safe_actions = actions.clamp(0, 3)
        gather_index = safe_actions[:, :, None, None, None].expand(b, k, 1, h, w)
        expanded_cause = cause_probability[:, None].expand(b, k, 4, h, w)
        expanded_alpha = action_alpha[:, None].expand(b, k, 4, h, w)
        selected_cause = expanded_cause.gather(2, gather_index).squeeze(2)
        selected_alpha = expanded_alpha.gather(2, gather_index).squeeze(2)

        cause_mean = _masked_scalar_mean(selected_cause, masks)
        cause_max = _masked_scalar_max(selected_cause, masks)
        alpha_mean = _masked_scalar_mean(selected_alpha, masks)
        alpha_max = _masked_scalar_max(selected_alpha, masks)
        area_fraction = masks.mean(dim=(-2, -1)).unsqueeze(-1)
        entropy_map = entropy[:, 0][:, None].expand(b, k, h, w)
        boundary_map = boundary[:, 0][:, None].expand(b, k, h, w)
        base_map = base_probability[:, 0][:, None].expand(b, k, h, w)
        entropy_mean = _masked_scalar_mean(entropy_map, masks)
        boundary_mean = _masked_scalar_mean(boundary_map, masks)
        base_mean = _masked_scalar_mean(base_map, masks)
        action_one_hot = F.one_hot(safe_actions, num_classes=4).to(feature.dtype)

        candidate_feature = torch.cat(
            [
                inside_feature,
                ring_feature,
                cause_mean,
                cause_max,
                alpha_mean,
                alpha_max,
                area_fraction,
                entropy_mean,
                boundary_mean,
                base_mean,
                action_one_hot,
            ],
            dim=2,
        )
        raw_scores = self.scorer(candidate_feature).squeeze(-1)
        scores = torch.where(valid, raw_scores, raw_scores.new_full(raw_scores.shape, -1.0e4))

        best_score, best_index = scores.max(dim=1)
        has_candidate = valid.any(dim=1)
        execute = has_candidate & (best_score > 0.0)
        if not bool(deploy_enabled):
            execute = torch.zeros_like(execute)

        selected_mask = masks.gather(
            1, best_index[:, None, None, None].expand(b, 1, h, w)
        )
        selected_action = actions.gather(1, best_index[:, None])[:, 0]
        selected_mask = selected_mask * execute[:, None, None, None].to(masks.dtype)
        selected_action = torch.where(execute, selected_action, torch.zeros_like(selected_action))
        selected_score = torch.where(execute, best_score, torch.zeros_like(best_score))

        action_max_scores = raw_scores.new_full((b, 4), -20.0)
        for action in range(4):
            action_valid = valid & (actions == action)
            action_max_scores[:, action] = torch.where(
                action_valid.any(dim=1),
                raw_scores.masked_fill(~action_valid, -1.0e4).max(dim=1).values,
                action_max_scores[:, action],
            )

        return {
            "candidate_masks": masks,
            "candidate_actions": actions,
            "candidate_valid": valid,
            "candidate_scores": raw_scores,
            "candidate_prefilter_support": prefilter_support,
            "selected_index": best_index,
            "selected_mask": selected_mask,
            "selected_action": selected_action,
            "selected_score": selected_score,
            "predicted_execute": execute,
            "action_max_scores": action_max_scores,
            "raw_component_count": raw_scores.new_full((b,), float(bank_stats.raw_components) / max(b, 1)),
            "retained_component_count": valid.sum(dim=1).to(raw_scores.dtype),
        }
