"""V531 typed sparse refinement for medical image segmentation.

The module replaces candidate-bank ranking with one spatially sparse, monotone
refinement pass.  It consumes the factual error maps and four atomic action
strengths already produced by V503/V518:

    Preserve / Delete / Fill / Trim / Expand

A five-way spatial router chooses one action per pixel.  Delete/Trim can only
remove foreground; Fill/Expand can only add foreground.  A calibrated action
correctness head provides an explicit Preserve-first risk gate.  The resulting
composition is bounded in [0, 1] by construction and can execute different
actions in different regions of the same case.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1.0e-6
ACTION_NAMES: Tuple[str, ...] = (
    "preserve",
    "delete",
    "fill",
    "trim",
    "expand",
)


def _as_b1hw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        return x[:, :1]
    if x.ndim == 3:
        return x[:, None]
    raise ValueError(f"Expected [B,H,W] or [B,1,H,W], got {tuple(x.shape)}")


def _entropy(prob: torch.Tensor) -> torch.Tensor:
    p = prob.clamp(EPS, 1.0 - EPS)
    return (
        -(p * p.log() + (1.0 - p) * (1.0 - p).log()) / math.log(2.0)
    ).clamp(0.0, 1.0)


def _soft_boundary(prob: torch.Tensor, radius: int = 1) -> torch.Tensor:
    radius = max(int(radius), 1)
    kernel = 2 * radius + 1
    dilated = F.max_pool2d(prob, kernel, stride=1, padding=radius)
    eroded = -F.max_pool2d(-prob, kernel, stride=1, padding=radius)
    return (dilated - eroded).clamp(0.0, 1.0)


def _gray_edge(
    image: torch.Tensor,
    size: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    gray = image.mean(dim=1, keepdim=True)
    if gray.shape[-2:] != size:
        gray = F.interpolate(
            gray,
            size=size,
            mode="bilinear",
            align_corners=False,
        )
    lo = gray.amin(dim=(-2, -1), keepdim=True)
    hi = gray.amax(dim=(-2, -1), keepdim=True)
    gray = (gray - lo) / (hi - lo).clamp_min(EPS)
    gx = F.pad(
        (gray[..., :, 1:] - gray[..., :, :-1]).abs(),
        (0, 1, 0, 0),
    )
    gy = F.pad(
        (gray[..., 1:, :] - gray[..., :-1, :]).abs(),
        (0, 0, 0, 1),
    )
    return gray, (gx + gy).clamp(0.0, 1.0)


def _group_count(channels: int) -> int:
    groups = min(8, int(channels))
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


def _safe_logit(prob: torch.Tensor) -> torch.Tensor:
    return torch.logit(prob.clamp(EPS, 1.0 - EPS))


class V531TypedSparseRefiner(nn.Module):
    """Preserve-first spatial routing over four monotone repair actions.

    Inputs
    ------
    c0_prob:
        Base probability map [B,1,H,W].
    cause_map_probs:
        V503/V518 typed factual error probabilities in
        Delete, Fill, Trim, Expand order [B,4,H,W].
    action_alpha:
        Four monotone action strengths produced by AtomicLocalRepairGenerator
        [B,4,H,W].

    Notes
    -----
    * At each pixel the route is a five-way distribution, so action weights sum
      to at most one and opposite edits cannot both be fully active.
    * The output is bounded without clamp:
        P = P0 * (1 - A_minus) + (1 - P0) * A_plus.
    * The correctness probability partitions every proposed edit into predicted
      fix and predicted harm mass exactly; no independent heads can violate the
      conservation identity.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        dropout: float = 0.10,
        semantic_channels: int = 512,
        semantic_dim: int = 32,
        use_semantic: bool = True,
        route_temperature: float = 0.50,
        risk_temperature: float = 0.25,
        harm_penalty: float = 1.50,
        edit_penalty: float = 0.02,
        utility_threshold: float = 0.0,
        preserve_bias: float = 1.25,
        action_bias: float = 0.0,
        initial_correctness: float = 0.50,
        hard_inference: bool = True,
        straight_through_train: bool = False,
        detach_m1_inputs: bool = True,
        max_action_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        semantic_dim = int(semantic_dim)
        self.use_semantic = bool(use_semantic)
        self.semantic_dim = semantic_dim
        self.route_temperature = max(float(route_temperature), 1.0e-4)
        self.risk_temperature = max(float(risk_temperature), 1.0e-4)
        self.harm_penalty = max(float(harm_penalty), 0.0)
        self.edit_penalty = max(float(edit_penalty), 0.0)
        self.utility_threshold = float(utility_threshold)
        self.hard_inference = bool(hard_inference)
        self.straight_through_train = bool(straight_through_train)
        self.detach_m1_inputs = bool(detach_m1_inputs)
        self.max_action_alpha = min(max(float(max_action_alpha), 0.0), 1.0)

        groups = _group_count(hidden_dim)
        # c0, 1-c0, entropy, boundary, gray, edge, four causes, four alphas.
        self.context_encoder = nn.Sequential(
            nn.Conv2d(14, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            nn.Dropout2d(float(dropout)),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
        )

        if self.use_semantic:
            semantic_groups = _group_count(semantic_dim)
            self.semantic_proj = nn.Sequential(
                nn.Conv2d(
                    int(semantic_channels),
                    semantic_dim,
                    1,
                    bias=False,
                ),
                nn.GroupNorm(semantic_groups, semantic_dim),
                nn.GELU(),
            )
            self.semantic_fuse = nn.Sequential(
                nn.Conv2d(
                    hidden_dim + semantic_dim,
                    hidden_dim,
                    3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(groups, hidden_dim),
                nn.GELU(),
            )
        else:
            self.semantic_proj = None
            self.semantic_fuse = None

        self.route_head = nn.Conv2d(hidden_dim, 5, 1)
        self.correctness_head = nn.Conv2d(hidden_dim, 4, 1)
        self.uncertainty_head = nn.Conv2d(hidden_dim, 4, 1)

        nn.init.zeros_(self.route_head.weight)
        nn.init.constant_(self.route_head.bias, float(action_bias))
        with torch.no_grad():
            self.route_head.bias[0] = float(preserve_bias)

        rate = min(max(float(initial_correctness), 1.0e-4), 1.0 - 1.0e-4)
        correctness_bias = math.log(rate / (1.0 - rate))
        nn.init.zeros_(self.correctness_head.weight)
        nn.init.constant_(self.correctness_head.bias, correctness_bias)
        nn.init.zeros_(self.uncertainty_head.weight)
        nn.init.constant_(self.uncertainty_head.bias, -2.0)

    def set_epoch(self, epoch: int) -> None:
        # Kept for the common pipeline contract and future curriculum schedules.
        self.current_epoch = int(epoch)

    def _fuse_semantic(
        self,
        feature: torch.Tensor,
        semantic_map: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.use_semantic:
            return feature
        if not isinstance(semantic_map, torch.Tensor):
            semantic = feature.new_zeros(
                feature.shape[0],
                self.semantic_dim,
                *feature.shape[-2:],
            )
        else:
            semantic = semantic_map
            if semantic.shape[-2:] != feature.shape[-2:]:
                semantic = F.interpolate(
                    semantic,
                    size=feature.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            semantic = self.semantic_proj(semantic)
        return self.semantic_fuse(torch.cat([feature, semantic], dim=1))

    @staticmethod
    def _hard_route_straight_through(
        route_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        index = route_probs.argmax(dim=1)
        hard = F.one_hot(index, num_classes=5).permute(0, 3, 1, 2)
        hard = hard.to(dtype=route_probs.dtype)
        return hard + route_probs - route_probs.detach(), index

    def forward(
        self,
        *,
        image: torch.Tensor,
        c0_prob: torch.Tensor,
        cause_map_probs: torch.Tensor,
        action_alpha: torch.Tensor,
        semantic_map: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        c0 = _as_b1hw(c0_prob).clamp(EPS, 1.0 - EPS)
        size = c0.shape[-2:]

        if cause_map_probs.ndim != 4 or cause_map_probs.shape[1] != 4:
            raise ValueError("cause_map_probs must have shape [B,4,H,W]")
        if action_alpha.ndim != 4 or action_alpha.shape[1] != 4:
            raise ValueError("action_alpha must have shape [B,4,H,W]")
        if cause_map_probs.shape[-2:] != size:
            cause_map_probs = F.interpolate(
                cause_map_probs,
                size=size,
                mode="bilinear",
                align_corners=False,
            )
        if action_alpha.shape[-2:] != size:
            action_alpha = F.interpolate(
                action_alpha,
                size=size,
                mode="bilinear",
                align_corners=False,
            )

        causes = cause_map_probs.clamp(0.0, 1.0)
        alphas = action_alpha.clamp(0.0, self.max_action_alpha)
        if self.detach_m1_inputs:
            causes_for_router = causes.detach()
            alphas_for_router = alphas.detach()
        else:
            causes_for_router = causes
            alphas_for_router = alphas

        entropy = _entropy(c0)
        boundary = _soft_boundary(c0)
        gray, edge = _gray_edge(image, size)
        feature = self.context_encoder(
            torch.cat(
                [
                    c0,
                    1.0 - c0,
                    entropy,
                    boundary,
                    gray,
                    edge,
                    causes_for_router,
                    alphas_for_router,
                ],
                dim=1,
            )
        )
        feature = self._fuse_semantic(feature, semantic_map)

        route_logits = self.route_head(feature)
        soft_route = F.softmax(
            route_logits / self.route_temperature,
            dim=1,
        )
        if (not self.training and self.hard_inference) or (
            self.training and self.straight_through_train
        ):
            route_probs, selected_action = self._hard_route_straight_through(
                soft_route
            )
        else:
            route_probs = soft_route
            selected_action = route_probs.argmax(dim=1)

        correctness_logits = self.correctness_head(feature)
        correctness_probs = torch.sigmoid(correctness_logits)
        harm_probs = 1.0 - correctness_probs
        log_variance = self.uncertainty_head(feature).clamp(-8.0, 4.0)
        predicted_sigma = torch.exp(0.5 * log_variance)

        action_utility = (
            correctness_probs
            - self.harm_penalty * harm_probs
            - self.edit_penalty * alphas_for_router
        )
        risk_gate = torch.sigmoid(
            (action_utility - self.utility_threshold) / self.risk_temperature
        )

        action_route = route_probs[:, 1:]
        action_weight = action_route * risk_gate
        effective_alpha = (action_weight * alphas).clamp(0.0, 1.0)

        negative_alpha = (
            effective_alpha[:, 0:1] + effective_alpha[:, 2:3]
        ).clamp(0.0, 1.0)
        positive_alpha = (
            effective_alpha[:, 1:2] + effective_alpha[:, 3:4]
        ).clamp(0.0, 1.0)

        # Because route probabilities form one simplex, negative_alpha and
        # positive_alpha cannot jointly exceed one.  This monotone composition
        # is therefore bounded in [0, 1] without an output clamp.
        final_prob = c0 * (1.0 - negative_alpha) + (1.0 - c0) * positive_alpha

        proposal_action_weight = action_route
        proposal_effective_alpha = (
            proposal_action_weight * alphas
        ).clamp(0.0, 1.0)
        proposal_negative = (
            proposal_effective_alpha[:, 0:1]
            + proposal_effective_alpha[:, 2:3]
        ).clamp(0.0, 1.0)
        proposal_positive = (
            proposal_effective_alpha[:, 1:2]
            + proposal_effective_alpha[:, 3:4]
        ).clamp(0.0, 1.0)
        proposal_prob = (
            c0 * (1.0 - proposal_negative)
            + (1.0 - c0) * proposal_positive
        )

        # Actual probability mass moved by each action.  For negative actions
        # the removable mass is c0; for positive actions it is 1-c0.
        per_action_edit_mass = torch.cat(
            [
                c0 * effective_alpha[:, 0:1],
                (1.0 - c0) * effective_alpha[:, 1:2],
                c0 * effective_alpha[:, 2:3],
                (1.0 - c0) * effective_alpha[:, 3:4],
            ],
            dim=1,
        )
        predicted_fix_mass_map = per_action_edit_mass * correctness_probs
        predicted_harm_mass_map = per_action_edit_mass * harm_probs
        conservation_error = (
            predicted_fix_mass_map
            + predicted_harm_mass_map
            - per_action_edit_mass
        ).abs().amax(dim=(1, 2, 3))

        hard_route = F.one_hot(
            selected_action,
            num_classes=5,
        ).permute(0, 3, 1, 2).to(route_probs.dtype)

        return {
            "m2_route_logits": route_logits,
            "m2_route_probs": route_probs,
            "m2_route_soft_probs": soft_route,
            "m2_route_hard": hard_route,
            "m2_selected_action": selected_action,
            "m2_action_correctness_logits": correctness_logits,
            "m2_action_correctness_probs": correctness_probs,
            "m2_action_harm_probs": harm_probs,
            "m2_action_log_variance": log_variance,
            "m2_action_sigma": predicted_sigma,
            "m2_action_utility": action_utility,
            "m2_action_risk_gate": risk_gate,
            "m2_action_weight": action_weight,
            "m2_effective_action_alpha": effective_alpha,
            "m2_negative_alpha": negative_alpha,
            "m2_positive_alpha": positive_alpha,
            "m2_proposal_probs": proposal_prob[:, 0],
            "m2_convex_probs": final_prob[:, 0],
            "m2_training_probs": final_prob[:, 0],
            "m2_fused_probs": final_prob[:, 0],
            "m2_residual_map": (final_prob - c0)[:, 0],
            "m2_edit_gate_prob": action_weight.sum(dim=1),
            "m2_edit_gate_logit": _safe_logit(
                action_weight.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
            )[:, 0],
            "v531_predicted_fix_mass_map": predicted_fix_mass_map,
            "v531_predicted_harm_mass_map": predicted_harm_mass_map,
            "v531_per_action_edit_mass": per_action_edit_mass,
            "v531_outcome_contract_error": conservation_error,
            "v531_preserve_probability": route_probs[:, 0],
            "v531_feature_map": feature,
        }
