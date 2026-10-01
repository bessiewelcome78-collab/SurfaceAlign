"""V505 interaction-driven region-level net-potential-outcome policy.

This module adapts the *iterative correction* idea from interactive medical
segmentation to a fully automatic setting without leaking GT at deployment:

* M1 emits typed, cause-gated atomic candidates.
* M2 performs K consecutive box-like correction interactions.  Every step sees
  the previous M2 mask and an interaction-memory map, exactly as an interactive
  decoder sees the previous mask and prior prompts.
* M2 never has an independent "edit presence" head.  Preserve is the zero-score
  reference and an edit exists iff at least one proposal has predicted positive
  net outcome.
* M3 replays the selected interactions and accepts/rejects each *whole region*.
* GT is used only by the loss-side potential-outcome teacher, never by forward.

The proposal unit is a typed action (Delete/Fill/Trim/Expand) combined with a
spatial box.  Candidate effects remain cause-gated, so a box does not grant an
action permission to alter unrelated pixels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.v503_factual_atomic_causal import EPS, _as_b1hw, _entropy, _gray_edge, _soft_boundary


OUTCOME_NEUTRAL = 0
OUTCOME_BENEFIT = 1
OUTCOME_HARM = 2


def _groups(hidden_dim: int) -> int:
    groups = min(8, int(hidden_dim))
    while hidden_dim % groups != 0 and groups > 1:
        groups -= 1
    return groups


def make_box_interactions(
    height: int,
    width: int,
    grid_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    include_full_box: bool = True,
    pad_fraction: float = 0.125,
) -> torch.Tensor:
    """Create deterministic full-image + overlapping grid box prompts [Q,H,W]."""
    grid = max(int(grid_size), 1)
    masks: List[torch.Tensor] = []
    if include_full_box:
        masks.append(torch.ones((height, width), device=device, dtype=dtype))

    cell_h = float(height) / float(grid)
    cell_w = float(width) / float(grid)
    pad_h = int(round(cell_h * max(float(pad_fraction), 0.0)))
    pad_w = int(round(cell_w * max(float(pad_fraction), 0.0)))
    for row in range(grid):
        for col in range(grid):
            y0 = max(0, int(round(row * cell_h)) - pad_h)
            y1 = min(height, int(round((row + 1) * cell_h)) + pad_h)
            x0 = max(0, int(round(col * cell_w)) - pad_w)
            x1 = min(width, int(round((col + 1) * cell_w)) + pad_w)
            mask = torch.zeros((height, width), device=device, dtype=dtype)
            mask[y0:y1, x0:x1] = 1.0
            masks.append(mask)
    return torch.stack(masks, dim=0)


def expand_typed_boxes(boxes: torch.Tensor, num_actions: int = 4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return action-major proposal masks, action ids and box ids."""
    if boxes.ndim != 3:
        raise ValueError(f"boxes must be [Q,H,W], got {tuple(boxes.shape)}")
    q = boxes.shape[0]
    masks = boxes.repeat(int(num_actions), 1, 1)
    action_ids = torch.arange(num_actions, device=boxes.device, dtype=torch.long).repeat_interleave(q)
    box_ids = torch.arange(q, device=boxes.device, dtype=torch.long).repeat(num_actions)
    return masks, action_ids, box_ids


def _masked_pool(feature: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Pool [B,C,H,W] over shared [P,H,W] masks -> [B,P,C]."""
    denominator = masks.flatten(1).sum(dim=1).clamp_min(1.0)
    pooled = torch.einsum("bchw,phw->bpc", feature, masks)
    return pooled / denominator[None, :, None]


def _typed_box_pool(values: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Pool action maps [B,A,H,W] over boxes -> flattened [B,A*Q]."""
    denominator = boxes.flatten(1).sum(dim=1).clamp_min(1.0)
    pooled = torch.einsum("bahw,qhw->baq", values, boxes)
    pooled = pooled / denominator[None, None]
    return pooled.reshape(values.shape[0], -1)


def _gather_action_map(values: torch.Tensor, action_ids: torch.Tensor) -> torch.Tensor:
    """Gather [B,A,H,W] with proposal action ids [P] -> [B,P,H,W]."""
    return values[:, action_ids]


def _gather_proposal_map(
    action_values: torch.Tensor,
    proposal_masks: torch.Tensor,
    action_ids: torch.Tensor,
    proposal_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather selected candidate, box and action for each batch item."""
    batch = action_values.shape[0]
    safe_index = proposal_index.clamp_min(0)
    selected_action = action_ids[safe_index]
    selected_candidate = action_values[
        torch.arange(batch, device=action_values.device), selected_action
    ][:, None]
    selected_mask = proposal_masks[safe_index][:, None]
    selected_mask = selected_mask * (proposal_index >= 0)[:, None, None, None].to(selected_mask.dtype)
    return selected_candidate, selected_mask, selected_action


def _proposal_score(
    outcome_logits: torch.Tensor,
    delta_dice: torch.Tensor,
    delta_surface: torch.Tensor,
    surface_weight: float,
) -> torch.Tensor:
    probability = torch.softmax(outcome_logits, dim=-1)
    return (
        probability[..., OUTCOME_BENEFIT]
        - probability[..., OUTCOME_HARM]
        + torch.tanh(delta_dice)
        + float(surface_weight) * torch.tanh(delta_surface)
    )


def _selection_floor(score: torch.Tensor) -> float:
    """Finite floor used only inside argmax/softmax-style selection paths.

    The learned proposal score must remain finite and unmodified for training.
    Invalid/used proposals are masked only at the point where a discrete choice
    is made, preventing sentinel values from leaking into regression or ranking
    losses.
    """
    if score.dtype in (torch.float16, torch.bfloat16):
        return -1.0e4
    return -1.0e9


class _RegionOutcomeHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(hidden_dim, 3)
        self.delta_head = nn.Linear(hidden_dim, 2)
        # Neutral by default; Preserve remains the zero-score reference.
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        with torch.no_grad():
            self.classifier.bias[OUTCOME_NEUTRAL] = 1.0
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, feature: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.trunk(feature)
        delta = self.delta_head(hidden)
        return {
            "embedding": hidden,
            "outcome_logits": self.classifier(hidden),
            "delta_dice": delta[..., 0],
            "delta_surface": delta[..., 1],
        }


class V505InteractiveRegionPolicy(nn.Module):
    """M2: iterative typed-box interaction policy with explicit Preserve."""

    def __init__(
        self,
        hidden_dim: int = 64,
        proposal_hidden_dim: int = 96,
        dropout: float = 0.10,
        grid_size: int = 4,
        steps: int = 4,
        surface_weight: float = 0.5,
        pad_fraction: float = 0.125,
        min_edit_mass: float = 1.0e-5,
        selection_margin: float = 0.05,
        include_full_box: bool = True,
        strict_deploy_valid: bool = False,
    ) -> None:
        super().__init__()
        self.grid_size = max(int(grid_size), 1)
        self.steps = max(int(steps), 1)
        self.surface_weight = float(surface_weight)
        self.pad_fraction = float(pad_fraction)
        self.min_edit_mass = float(min_edit_mass)
        self.selection_margin = max(float(selection_margin), 0.0)
        self.include_full_box = bool(include_full_box)
        self.strict_deploy_valid = bool(strict_deploy_valid)
        groups = _groups(hidden_dim)
        # current/base/entropy/boundary/gray/edge + cause4 + delta4 + support4 + memory4
        input_channels = 22
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
        )
        action_dim = 8
        self.action_embedding = nn.Embedding(4, action_dim)
        # pooled visual + action embedding + area/edit/support/cause statistics
        self.outcome_head = _RegionOutcomeHead(
            hidden_dim + action_dim + 4,
            proposal_hidden_dim,
            dropout,
        )

    def _encode_proposals(
        self,
        image: torch.Tensor,
        base: torch.Tensor,
        current: torch.Tensor,
        action_candidates: torch.Tensor,
        action_supports: torch.Tensor,
        cause_maps: torch.Tensor,
        interaction_memory: torch.Tensor,
        boxes: torch.Tensor,
        proposal_masks: torch.Tensor,
        action_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        entropy = _entropy(current)
        boundary = _soft_boundary(current)
        gray, edge = _gray_edge(image, current.shape[-2:])
        delta = action_candidates - current
        feature = self.encoder(
            torch.cat(
                [
                    current,
                    base,
                    entropy,
                    boundary,
                    gray,
                    edge,
                    cause_maps,
                    delta,
                    action_supports,
                    interaction_memory,
                ],
                dim=1,
            )
        )
        pooled = _masked_pool(feature, proposal_masks)
        q = boxes.shape[0]
        action_embed = self.action_embedding(action_ids)[None].expand(current.shape[0], -1, -1)

        edit_magnitude = delta.abs()
        edit_mass = _typed_box_pool(edit_magnitude, boxes)
        support_mass = _typed_box_pool(action_supports, boxes)
        cause_mass = _typed_box_pool(cause_maps, boxes)
        hard_change = (action_candidates >= 0.5) != (current >= 0.5)
        hard_edit_fraction = _typed_box_pool(hard_change.float(), boxes)
        box_area = boxes.flatten(1).mean(dim=1).repeat(4)[None].expand(current.shape[0], -1)
        stats = torch.stack([box_area, edit_mass, support_mass, cause_mass], dim=-1)
        head = self.outcome_head(torch.cat([pooled, action_embed, stats], dim=-1))
        score = _proposal_score(
            head["outcome_logits"],
            head["delta_dice"],
            head["delta_surface"],
            self.surface_weight,
        )

        # Separate learning support from deployment executability.  M2 may learn
        # an outcome model from a typed region as soon as it has cause/support
        # evidence, but it may deploy that region only after the candidate
        # actually changes at least one binary pixel.  This prevents tiny soft
        # residuals from being counted as real interactions.
        soft_valid = (
            (support_mass > self.min_edit_mass)
            & (cause_mass > self.min_edit_mass)
        )
        deploy_valid = (
            (edit_mass > self.min_edit_mass)
            & (hard_edit_fraction > 0.0)
        )
        if self.strict_deploy_valid:
            deploy_valid = deploy_valid & soft_valid
            # V516: outcome, ranking and deployment see one executable
            # proposal distribution.  This removes train_valid=1.0 versus
            # deploy_valid≈0.05 distribution shift.
            train_valid = deploy_valid
        else:
            train_valid = soft_valid
        return {
            **head,
            "proposal_score": score,
            "proposal_valid": train_valid,
            "proposal_deploy_valid": deploy_valid,
            "proposal_edit_mass": edit_mass,
            "proposal_hard_edit_fraction": hard_edit_fraction,
            "proposal_support_mass": support_mass,
            "proposal_cause_mass": cause_mass,
        }

    def forward(
        self,
        *,
        image: torch.Tensor,
        c0_prob: torch.Tensor,
        candidate_probs: torch.Tensor,
        supports: torch.Tensor,
        cause_map_probs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        base = _as_b1hw(c0_prob).detach().clamp(EPS, 1.0 - EPS)
        if candidate_probs.ndim != 4 or candidate_probs.shape[1] < 5:
            raise ValueError("V505 requires Preserve + four typed local candidates")
        actions = candidate_probs[:, 1:5].detach().clamp(EPS, 1.0 - EPS)
        action_supports = supports[:, 1:5].detach().clamp(0.0, 1.0)
        causes = cause_map_probs.detach().clamp(0.0, 1.0)
        if causes.shape[-2:] != base.shape[-2:]:
            causes = F.interpolate(causes, size=base.shape[-2:], mode="bilinear", align_corners=False)

        boxes = make_box_interactions(
            base.shape[-2],
            base.shape[-1],
            self.grid_size,
            device=base.device,
            dtype=base.dtype,
            include_full_box=self.include_full_box,
            pad_fraction=self.pad_fraction,
        )
        proposal_masks, action_ids, box_ids = expand_typed_boxes(boxes, 4)

        current = base
        memory = torch.zeros(
            (base.shape[0], 4, base.shape[-2], base.shape[-1]),
            device=base.device,
            dtype=base.dtype,
        )
        used = torch.zeros(
            (base.shape[0], proposal_masks.shape[0]),
            device=base.device,
            dtype=torch.bool,
        )

        step_current: List[torch.Tensor] = []
        step_proposed: List[torch.Tensor] = []
        step_logits: List[torch.Tensor] = []
        step_delta_dice: List[torch.Tensor] = []
        step_delta_surface: List[torch.Tensor] = []
        step_score: List[torch.Tensor] = []
        step_selected: List[torch.Tensor] = []
        step_selected_mask: List[torch.Tensor] = []
        step_embedding: List[torch.Tensor] = []
        step_valid: List[torch.Tensor] = []
        step_available: List[torch.Tensor] = []

        active = torch.ones((base.shape[0],), device=base.device, dtype=torch.bool)
        for _ in range(self.steps):
            encoded = self._encode_proposals(
                image,
                base,
                current,
                actions,
                action_supports,
                causes,
                memory,
                boxes,
                proposal_masks,
                action_ids,
            )
            raw_score = encoded["proposal_score"]
            available = encoded["proposal_deploy_valid"] & (~used) & active[:, None]
            selection_score = raw_score.masked_fill(
                ~available, _selection_floor(raw_score)
            )
            best_score, best_index = selection_score.max(dim=1)
            selected = torch.where(
                active
                & available.any(dim=1)
                & (best_score > self.selection_margin),
                best_index,
                torch.full_like(best_index, -1),
            )
            selected_candidate, selected_mask, selected_action = _gather_proposal_map(
                actions, proposal_masks, action_ids, selected
            )
            proposed = (
                current * (1.0 - selected_mask) + selected_candidate * selected_mask
            ).clamp(EPS, 1.0 - EPS)

            step_current.append(current)
            step_proposed.append(proposed)
            step_logits.append(encoded["outcome_logits"])
            step_delta_dice.append(encoded["delta_dice"])
            step_delta_surface.append(encoded["delta_surface"])
            # Store the finite learned score. The selection-only mask is stored
            # separately so the loss can ignore unavailable proposals without
            # ever seeing a large negative sentinel.
            step_score.append(raw_score)
            step_selected.append(selected)
            step_selected_mask.append(selected_mask[:, 0])
            step_embedding.append(encoded["embedding"])
            step_valid.append(encoded["proposal_valid"])
            step_available.append(available)

            valid_selected = selected >= 0
            if valid_selected.any():
                used[
                    torch.arange(base.shape[0], device=base.device)[valid_selected],
                    selected[valid_selected],
                ] = True
                for action in range(4):
                    action_mask = valid_selected & (selected_action == action)
                    if action_mask.any():
                        memory[action_mask, action] = torch.maximum(
                            memory[action_mask, action], selected_mask[action_mask, 0]
                        )
            current = proposed
            active = active & valid_selected

        return {
            "m2_fused_probs": current[:, 0],
            "v505_m2_step_current_probs": torch.stack(step_current, dim=1)[:, :, 0],
            "v505_m2_step_proposed_probs": torch.stack(step_proposed, dim=1)[:, :, 0],
            "v505_m2_outcome_logits": torch.stack(step_logits, dim=1),
            "v505_m2_delta_dice_pred": torch.stack(step_delta_dice, dim=1),
            "v505_m2_delta_surface_pred": torch.stack(step_delta_surface, dim=1),
            "v505_m2_proposal_score": torch.stack(step_score, dim=1),
            "v505_m2_selected_proposal": torch.stack(step_selected, dim=1),
            "v505_m2_selected_region": torch.stack(step_selected_mask, dim=1),
            "v505_m2_proposal_embedding": torch.stack(step_embedding, dim=1),
            "v505_m2_proposal_valid": torch.stack(step_valid, dim=1),
            "v505_m2_proposal_train_valid": torch.stack(step_valid, dim=1),
            "v505_m2_proposal_deploy_valid": torch.stack(step_available, dim=1),
            "v505_m2_proposal_available": torch.stack(step_available, dim=1),
            "v505_interaction_boxes": boxes,
            "v505_proposal_masks": proposal_masks,
            "v505_proposal_action_ids": action_ids,
            "v505_proposal_box_ids": box_ids,
            "v505_num_interactions": (torch.stack(step_selected, dim=1) >= 0).float().sum(dim=1),
            "m2_edit_gate_prob": (current - base).abs().clamp(0.0, 1.0),
            "m2_residual_map": current - base,
        }


class V505RegionVerifier(nn.Module):
    """M3: region-level verifier that replays whole M2 interactions."""

    def __init__(
        self,
        hidden_dim: int = 64,
        proposal_hidden_dim: int = 96,
        dropout: float = 0.10,
        surface_weight: float = 0.5,
        min_edit_mass: float = 1.0e-5,
        selection_margin: float = 0.05,
        strict_deploy_valid: bool = False,
    ) -> None:
        super().__init__()
        self.surface_weight = float(surface_weight)
        self.min_edit_mass = float(min_edit_mass)
        self.selection_margin = max(float(selection_margin), 0.0)
        self.strict_deploy_valid = bool(strict_deploy_valid)
        groups = _groups(hidden_dim)
        # current/base/entropy/boundary/gray/edge + cause4 + delta4 + support4 + accepted-memory4
        self.encoder = nn.Sequential(
            nn.Conv2d(22, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
        )
        action_dim = 8
        self.action_embedding = nn.Embedding(4, action_dim)
        # pooled feature + action + four stats + detached M2 class/deltas/score (6)
        self.outcome_head = _RegionOutcomeHead(
            hidden_dim + action_dim + 4 + 6,
            proposal_hidden_dim,
            dropout,
        )

    def _score_all(
        self,
        image: torch.Tensor,
        base: torch.Tensor,
        current: torch.Tensor,
        actions: torch.Tensor,
        supports: torch.Tensor,
        causes: torch.Tensor,
        accepted_memory: torch.Tensor,
        boxes: torch.Tensor,
        proposal_masks: torch.Tensor,
        action_ids: torch.Tensor,
        m2_logits: torch.Tensor,
        m2_delta_dice: torch.Tensor,
        m2_delta_surface: torch.Tensor,
        m2_score: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        entropy = _entropy(current)
        boundary = _soft_boundary(current)
        gray, edge = _gray_edge(image, current.shape[-2:])
        delta = actions - current
        feature = self.encoder(
            torch.cat(
                [
                    current,
                    base,
                    entropy,
                    boundary,
                    gray,
                    edge,
                    causes,
                    delta,
                    supports,
                    accepted_memory,
                ],
                dim=1,
            )
        )
        pooled = _masked_pool(feature, proposal_masks)
        q = boxes.shape[0]
        action_embed = self.action_embedding(action_ids)[None].expand(current.shape[0], -1, -1)
        edit_mass = _typed_box_pool(delta.abs(), boxes)
        support_mass = _typed_box_pool(supports, boxes)
        cause_mass = _typed_box_pool(causes, boxes)
        hard_change = (actions >= 0.5) != (current >= 0.5)
        hard_edit_fraction = _typed_box_pool(hard_change.float(), boxes)
        box_area = boxes.flatten(1).mean(dim=1).repeat(4)[None].expand(current.shape[0], -1)
        stats = torch.stack([box_area, edit_mass, support_mass, cause_mass], dim=-1)
        m2_probability = torch.softmax(m2_logits.detach(), dim=-1)
        m2_evidence = torch.cat(
            [
                m2_probability,
                m2_delta_dice.detach()[..., None],
                m2_delta_surface.detach()[..., None],
                m2_score.detach()[..., None],
            ],
            dim=-1,
        )
        head = self.outcome_head(torch.cat([pooled, action_embed, stats, m2_evidence], dim=-1))
        score = _proposal_score(
            head["outcome_logits"],
            head["delta_dice"],
            head["delta_surface"],
            self.surface_weight,
        )
        soft_valid = (
            (support_mass > self.min_edit_mass)
            & (cause_mass > self.min_edit_mass)
        )
        deploy_valid = (
            (edit_mass > self.min_edit_mass)
            & (hard_edit_fraction > 0.0)
        )
        if self.strict_deploy_valid:
            deploy_valid = deploy_valid & soft_valid
            # V516: outcome, ranking and deployment see one executable
            # proposal distribution.  This removes train_valid=1.0 versus
            # deploy_valid≈0.05 distribution shift.
            train_valid = deploy_valid
        else:
            train_valid = soft_valid
        return {
            **head,
            "proposal_score": score,
            "proposal_valid": train_valid,
            "proposal_deploy_valid": deploy_valid,
            "proposal_hard_edit_fraction": hard_edit_fraction,
        }

    def forward(
        self,
        *,
        image: torch.Tensor,
        c0_prob: torch.Tensor,
        candidate_probs: torch.Tensor,
        supports: torch.Tensor,
        cause_map_probs: torch.Tensor,
        m2_output: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        base = _as_b1hw(c0_prob).detach().clamp(EPS, 1.0 - EPS)
        actions = candidate_probs[:, 1:5].detach().clamp(EPS, 1.0 - EPS)
        action_supports = supports[:, 1:5].detach().clamp(0.0, 1.0)
        causes = cause_map_probs.detach().clamp(0.0, 1.0)
        boxes = m2_output["v505_interaction_boxes"]
        proposal_masks = m2_output["v505_proposal_masks"]
        action_ids = m2_output["v505_proposal_action_ids"]
        selected_sequence = m2_output["v505_m2_selected_proposal"]

        current = base
        memory = torch.zeros(
            (base.shape[0], 4, base.shape[-2], base.shape[-1]),
            device=base.device,
            dtype=base.dtype,
        )
        step_current: List[torch.Tensor] = []
        step_proposed: List[torch.Tensor] = []
        step_final: List[torch.Tensor] = []
        step_logits: List[torch.Tensor] = []
        step_delta_dice: List[torch.Tensor] = []
        step_delta_surface: List[torch.Tensor] = []
        step_score: List[torch.Tensor] = []
        step_accept: List[torch.Tensor] = []
        step_accept_mask: List[torch.Tensor] = []
        step_selected_benefit_probability: List[torch.Tensor] = []
        step_valid: List[torch.Tensor] = []
        step_deploy_valid: List[torch.Tensor] = []

        for step in range(selected_sequence.shape[1]):
            scored = self._score_all(
                image,
                base,
                current,
                actions,
                action_supports,
                causes,
                memory,
                boxes,
                proposal_masks,
                action_ids,
                m2_output["v505_m2_outcome_logits"][:, step],
                m2_output["v505_m2_delta_dice_pred"][:, step],
                m2_output["v505_m2_delta_surface_pred"][:, step],
                m2_output["v505_m2_proposal_score"][:, step],
            )
            selected = selected_sequence[:, step]
            selected_candidate, selected_mask, selected_action = _gather_proposal_map(
                actions, proposal_masks, action_ids, selected
            )
            proposed = (
                current * (1.0 - selected_mask) + selected_candidate * selected_mask
            ).clamp(EPS, 1.0 - EPS)
            safe_index = selected.clamp_min(0)
            selected_score = scored["proposal_score"].gather(1, safe_index[:, None])[:, 0]
            selected_valid = scored["proposal_deploy_valid"].gather(
                1, safe_index[:, None]
            )[:, 0]
            accept = (
                (selected >= 0)
                & selected_valid
                & (selected_score > self.selection_margin)
            )
            accept_mask = accept[:, None, None, None].to(current.dtype)
            final = current * (1.0 - accept_mask) + proposed * accept_mask

            step_current.append(current)
            step_proposed.append(proposed)
            step_final.append(final)
            step_logits.append(scored["outcome_logits"])
            step_delta_dice.append(scored["delta_dice"])
            step_delta_surface.append(scored["delta_surface"])
            step_score.append(scored["proposal_score"])
            step_accept.append(accept.float())
            step_accept_mask.append(selected_mask * accept_mask)
            step_valid.append(scored["proposal_valid"])
            step_deploy_valid.append(scored["proposal_deploy_valid"])
            benefit_probability = torch.softmax(scored["outcome_logits"], dim=-1)[..., OUTCOME_BENEFIT]
            selected_benefit_probability = benefit_probability.gather(1, safe_index[:, None])[:, 0]
            step_selected_benefit_probability.append(
                torch.where(selected >= 0, selected_benefit_probability, torch.zeros_like(selected_benefit_probability))
            )

            if accept.any():
                for action in range(4):
                    action_mask = accept & (selected_action == action)
                    if action_mask.any():
                        memory[action_mask, action] = torch.maximum(
                            memory[action_mask, action], selected_mask[action_mask, 0]
                        )
            current = final

        # Compatibility aliases keep the existing validation/evaluation code intact.
        expert_probs = torch.cat([base, _as_b1hw(m2_output["m2_fused_probs"])], dim=1)
        changed_map = ((current - base).abs() > 1.0e-6).to(base.dtype)
        changed_rate = changed_map.flatten(1).mean(dim=1)
        selection_rate = torch.stack([1.0 - changed_rate, changed_rate], dim=1)
        accepted_region_union = torch.stack(step_accept_mask, dim=1).amax(dim=1)
        selected_benefit_probability = torch.stack(
            step_selected_benefit_probability, dim=1
        ).mean(dim=1)[:, None, None, None].expand_as(base)
        return {
            "fused_probs": current[:, 0],
            "final_probs": current[:, 0],
            "m3_expert_probs": expert_probs,
            "m3_pixel_weights": torch.cat([1.0 - changed_map, changed_map], dim=1),
            "m3_expert_selection_rate": selection_rate,
            "m3_selected_index": changed_map.long(),
            "m3_gate_prob": accepted_region_union,
            "m3_accept_probability": selected_benefit_probability,
            "m3_edit_relevance": (current - base).abs(),
            "m3_intervention_mask": changed_map,
            "m3_predicted_risk_map": torch.zeros_like(base),
            "m3_predicted_log_variance_map": torch.zeros_like(base),
            "v505_m3_step_current_probs": torch.stack(step_current, dim=1)[:, :, 0],
            "v505_m3_step_proposed_probs": torch.stack(step_proposed, dim=1)[:, :, 0],
            "v505_m3_step_final_probs": torch.stack(step_final, dim=1)[:, :, 0],
            "v505_m3_outcome_logits": torch.stack(step_logits, dim=1),
            "v505_m3_delta_dice_pred": torch.stack(step_delta_dice, dim=1),
            "v505_m3_delta_surface_pred": torch.stack(step_delta_surface, dim=1),
            "v505_m3_proposal_score": torch.stack(step_score, dim=1),
            "v505_m3_proposal_valid": torch.stack(step_valid, dim=1),
            "v505_m3_proposal_train_valid": torch.stack(step_valid, dim=1),
            "v505_m3_proposal_deploy_valid": torch.stack(step_deploy_valid, dim=1),
            "v505_m3_accept": torch.stack(step_accept, dim=1),
        }


@torch.no_grad()
def _surface_score(mask: torch.Tensor, gt: torch.Tensor, radius: int) -> torch.Tensor:
    """Tolerance-aware symmetric boundary score, a vectorized NSD surrogate."""
    boundary_mask = (_soft_boundary(mask.float(), radius=1) > 0.0).float()
    boundary_gt = (_soft_boundary(gt.float(), radius=1) > 0.0).float()
    kernel = 2 * max(int(radius), 1) + 1
    gt_tolerance = F.max_pool2d(boundary_gt, kernel, stride=1, padding=kernel // 2)
    mask_tolerance = F.max_pool2d(boundary_mask, kernel, stride=1, padding=kernel // 2)
    precision = (boundary_mask * gt_tolerance).flatten(1).sum(dim=1) / boundary_mask.flatten(1).sum(dim=1).clamp_min(1.0)
    recall = (boundary_gt * mask_tolerance).flatten(1).sum(dim=1) / boundary_gt.flatten(1).sum(dim=1).clamp_min(1.0)
    return 0.5 * (precision + recall)


@torch.no_grad()
def build_region_potential_outcomes(
    current_probs: torch.Tensor,
    candidate_probs: torch.Tensor,
    proposal_masks: torch.Tensor,
    action_ids: torch.Tensor,
    gt: torch.Tensor,
    *,
    surface_radius: int = 2,
    chunk_size: int = 8,
    teacher_size: int | None = None,
) -> Dict[str, torch.Tensor]:
    """Evaluate complete region interventions relative to the current mask.

    The teacher evaluates the exact deployment unit: one typed candidate inside
    one interaction box, with the current mask preserved elsewhere.
    """
    current = _as_b1hw(current_probs).detach().clamp(EPS, 1.0 - EPS)
    actions = candidate_probs[:, 1:5].detach().clamp(EPS, 1.0 - EPS)
    y = (_as_b1hw(gt) >= 0.5).float()
    if teacher_size is not None and int(teacher_size) > 0:
        target_size = min(int(teacher_size), current.shape[-2], current.shape[-1])
        if current.shape[-2:] != (target_size, target_size):
            current = F.interpolate(current, size=(target_size, target_size), mode="bilinear", align_corners=False)
            actions = F.interpolate(actions, size=(target_size, target_size), mode="bilinear", align_corners=False)
            y = F.interpolate(y, size=(target_size, target_size), mode="nearest")
            proposal_masks = F.interpolate(
                proposal_masks[:, None], size=(target_size, target_size), mode="nearest"
            )[:, 0]
    current_hard = current >= 0.5
    base_inter = (current_hard.float() * y).flatten(1).sum(dim=1)
    base_den = current_hard.float().flatten(1).sum(dim=1) + y.flatten(1).sum(dim=1)
    base_dice = (2.0 * base_inter + EPS) / (base_den + EPS)
    base_surface = _surface_score(current_hard.float(), y, surface_radius)

    delta_dice_chunks: List[torch.Tensor] = []
    delta_surface_chunks: List[torch.Tensor] = []
    benefit_fraction_chunks: List[torch.Tensor] = []
    harm_fraction_chunks: List[torch.Tensor] = []
    changed_fraction_chunks: List[torch.Tensor] = []

    proposal_count = proposal_masks.shape[0]
    for start in range(0, proposal_count, max(int(chunk_size), 1)):
        end = min(start + max(int(chunk_size), 1), proposal_count)
        ids = action_ids[start:end]
        masks = proposal_masks[start:end][None, :, None]
        selected_actions = actions[:, ids]
        cf = current[:, None] * (1.0 - masks) + selected_actions[:, :, None] * masks
        cf = cf[:, :, 0]
        hard = cf >= 0.5
        gt_expand = y[:, None, 0].bool().expand_as(hard)
        current_expand = current_hard[:, None, 0].expand_as(hard)
        inter = (hard.float() * gt_expand.float()).flatten(2).sum(dim=2)
        den = hard.float().flatten(2).sum(dim=2) + gt_expand.float().flatten(2).sum(dim=2)
        dice = (2.0 * inter + EPS) / (den + EPS)

        flat_hard = hard.reshape(-1, 1, hard.shape[-2], hard.shape[-1]).float()
        flat_gt = gt_expand.reshape(-1, 1, hard.shape[-2], hard.shape[-1]).float()
        surface = _surface_score(flat_hard, flat_gt, surface_radius).reshape(current.shape[0], -1)

        changed = hard != current_expand
        beneficial = changed & (current_expand != gt_expand) & (hard == gt_expand)
        harmful = changed & (current_expand == gt_expand) & (hard != gt_expand)
        changed_count = changed.float().flatten(2).sum(dim=2)
        benefit_count = beneficial.float().flatten(2).sum(dim=2)
        harm_count = harmful.float().flatten(2).sum(dim=2)

        delta_dice_chunks.append(dice - base_dice[:, None])
        delta_surface_chunks.append(surface - base_surface[:, None])
        benefit_fraction_chunks.append(benefit_count / changed_count.clamp_min(1.0))
        harm_fraction_chunks.append(harm_count / changed_count.clamp_min(1.0))
        changed_fraction_chunks.append(changed.float().flatten(2).mean(dim=2))

    delta_dice = torch.cat(delta_dice_chunks, dim=1)
    delta_surface = torch.cat(delta_surface_chunks, dim=1)
    benefit_fraction = torch.cat(benefit_fraction_chunks, dim=1)
    harm_fraction = torch.cat(harm_fraction_chunks, dim=1)
    changed_fraction = torch.cat(changed_fraction_chunks, dim=1)

    changed = changed_fraction > 0.0
    benefit = (
        changed
        & (delta_dice > 0.0)
        & (delta_surface >= 0.0)
        & (benefit_fraction > harm_fraction)
    )
    harm = changed & (
        (delta_dice < 0.0)
        | (delta_surface < 0.0)
        | (harm_fraction >= benefit_fraction)
    )
    outcome = torch.full_like(delta_dice, OUTCOME_NEUTRAL, dtype=torch.long)
    outcome[harm] = OUTCOME_HARM
    outcome[benefit] = OUTCOME_BENEFIT
    return {
        "outcome_class": outcome,
        "delta_dice": delta_dice,
        "delta_surface": delta_surface,
        "benefit_fraction": benefit_fraction,
        "harm_fraction": harm_fraction,
        "changed_fraction": changed_fraction,
        "benefit_mask": benefit.float(),
        "harm_mask": harm.float(),
    }


class V519FamilyAwareRegionComposer(nn.Module):
    """Family/dose/radius-aware dense region composer for a frozen V518 M1 bank.

    The V518 audit established two simultaneous facts:
      * the 42-candidate bank has a useful image-wise Oracle and a much larger
        pixel-wise Oracle;
      * many strong candidates are beneficial only on a small subset of regions.

    V519 therefore never makes one image-wise top-1 decision.  It predicts a
    Base-relative treatment effect, uncertainty, harm probability and
    eligibility for every candidate at every pixel.  Spatially smoothed lower
    confidence bound scores choose a different candidate in each local region,
    with Preserve represented by the fixed score zero.

    M1/Base inputs are observations.  A2 passes detached tensors and trains only
    this composer, so the high-ceiling V518 candidate bank cannot be destroyed
    by the selector objective.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 32,
        metadata_dim: int = 16,
        semantic_channels: int = 512,
        max_candidates: int = 64,
        temperature: float = 0.35,
        lcb_kappa: float = 1.0,
        harm_penalty: float = 0.02,
        edit_penalty: float = 0.002,
        utility_margin: float = 0.001,
        support_floor: float = 1.0e-4,
        edit_epsilon: float = 1.0e-4,
        region_radius: int = 2,
        dropout: float = 0.10,
        hard_inference: bool = True,
        straight_through_train: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_candidates = int(max_candidates)
        self.temperature = max(float(temperature), 1.0e-3)
        self.lcb_kappa = float(lcb_kappa)
        self.harm_penalty = float(harm_penalty)
        self.edit_penalty = float(edit_penalty)
        self.utility_margin = float(utility_margin)
        self.support_floor = max(float(support_floor), 0.0)
        self.edit_epsilon = max(float(edit_epsilon), 0.0)
        self.region_radius = max(int(region_radius), 0)
        self.dropout = max(float(dropout), 0.0)
        self.hard_inference = bool(hard_inference)
        self.straight_through_train = bool(straight_through_train)

        groups = _groups(self.hidden_dim)
        self.context_encoder = nn.Sequential(
            nn.Conv2d(6, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
        )
        self.semantic_proj = nn.Sequential(
            nn.Conv2d(int(semantic_channels), self.hidden_dim, 1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Conv2d(7, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
        )

        self.family_embedding = nn.Embedding(5, int(metadata_dim))
        self.action_embedding = nn.Embedding(10, int(metadata_dim))
        self.metadata_proj = nn.Sequential(
            nn.Linear(int(metadata_dim) * 2 + 3, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
            nn.Dropout2d(self.dropout),
        )
        self.effect_head = nn.Conv2d(self.hidden_dim, 1, 1)
        self.log_sigma_head = nn.Conv2d(self.hidden_dim, 1, 1)
        self.harm_head = nn.Conv2d(self.hidden_dim, 1, 1)
        self.eligibility_head = nn.Conv2d(self.hidden_dim, 1, 1)

        # Conservative start: zero effect, moderate uncertainty, low edit
        # eligibility. Preserve therefore wins until supervision is learned.
        for head in (self.effect_head, self.harm_head, self.eligibility_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.log_sigma_head.weight)
        nn.init.constant_(self.log_sigma_head.bias, -1.5)
        nn.init.constant_(self.eligibility_head.bias, -1.0)

    def _semantic_map(
        self,
        semantic_map: torch.Tensor | None,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(semantic_map, torch.Tensor):
            return reference.new_zeros(
                reference.shape[0], self.hidden_dim, *reference.shape[-2:]
            )
        semantic = semantic_map
        if semantic.shape[-2:] != reference.shape[-2:]:
            semantic = F.interpolate(
                semantic,
                size=reference.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return self.semantic_proj(semantic)

    def _smooth_regions(self, score: torch.Tensor) -> torch.Tensor:
        if self.region_radius <= 0:
            return score
        kernel = 2 * self.region_radius + 1
        return F.avg_pool2d(
            score,
            kernel_size=kernel,
            stride=1,
            padding=self.region_radius,
        )

    def forward(
        self,
        *,
        image: torch.Tensor,
        c0_prob: torch.Tensor,
        candidate_probs: torch.Tensor,
        supports: torch.Tensor,
        candidate_causes: torch.Tensor,
        family_ids: torch.Tensor,
        action_ids: torch.Tensor,
        dose_values: torch.Tensor,
        radius_values: torch.Tensor,
        deploy_mask: torch.Tensor,
        semantic_map: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        c0 = _as_b1hw(c0_prob).clamp(EPS, 1.0 - EPS)
        if candidate_probs.ndim != 4:
            raise ValueError("candidate_probs must be [B,K,H,W]")
        b, k, h, w = candidate_probs.shape
        if k < 2 or k > self.max_candidates:
            raise ValueError(
                f"V519 requires 2..{self.max_candidates} candidates, got {k}"
            )
        for name, tensor in (
            ("supports", supports),
            ("candidate_causes", candidate_causes),
        ):
            if tensor.shape != candidate_probs.shape:
                raise ValueError(
                    f"{name} must match candidate_probs; got "
                    f"{tuple(tensor.shape)} vs {tuple(candidate_probs.shape)}"
                )

        metadata_tensors = (family_ids, action_ids, dose_values, radius_values, deploy_mask)
        if any(tensor.ndim != 1 or tensor.numel() != k for tensor in metadata_tensors):
            raise ValueError("V519 candidate metadata must be one-dimensional with K entries")

        ent = _entropy(c0)
        bnd = _soft_boundary(c0)
        gray, edge = _gray_edge(image, (h, w))
        shared = self.context_encoder(
            torch.cat([c0, 1.0 - c0, ent, bnd, gray, edge], dim=1)
        )
        shared = shared + self._semantic_map(semantic_map, c0)

        nonbase = candidate_probs[:, 1:].clamp(EPS, 1.0 - EPS)
        n = nonbase.shape[1]
        c0_bank = c0.expand(-1, n, -1, -1)
        signed = nonbase - c0_bank
        abs_edit = signed.abs()
        support = supports[:, 1:].clamp(0.0, 1.0)
        cause = candidate_causes[:, 1:].clamp(0.0, 1.0)
        ent_bank = ent.expand(-1, n, -1, -1)
        bnd_bank = bnd.expand(-1, n, -1, -1)

        candidate_input = torch.stack(
            [nonbase, signed, abs_edit, support, cause, ent_bank, bnd_bank],
            dim=2,
        ).reshape(b * n, 7, h, w)
        candidate_feature = self.candidate_encoder(candidate_input)

        family = family_ids[1:].to(device=c0.device, dtype=torch.long).clamp(0, 4)
        action = action_ids[1:].to(device=c0.device, dtype=torch.long).clamp(0, 9)
        dose = dose_values[1:].to(device=c0.device, dtype=c0.dtype)
        radius = radius_values[1:].to(device=c0.device, dtype=c0.dtype)
        # Three explicit structural scalars: log-dose, normalized radius and
        # image-wise edit fraction. The latter is candidate-specific per image.
        log_dose = torch.log2(dose.clamp_min(1.0e-3)) / 3.0
        radius_norm = radius / radius.max().clamp_min(1.0)
        edit_fraction = abs_edit.flatten(2).mean(dim=2)

        family_emb = self.family_embedding(family)
        action_emb = self.action_embedding(action)
        fixed_meta = torch.cat(
            [family_emb, action_emb, log_dose[:, None], radius_norm[:, None]],
            dim=1,
        )
        fixed_meta = fixed_meta[None].expand(b, -1, -1)
        meta_input = torch.cat([fixed_meta, edit_fraction[..., None]], dim=2)
        meta = self.metadata_proj(meta_input.reshape(b * n, -1))
        meta = meta[:, :, None, None]

        shared_bank = shared[:, None].expand(-1, n, -1, -1, -1)
        shared_bank = shared_bank.reshape(b * n, self.hidden_dim, h, w)
        fused = self.fuse(candidate_feature + shared_bank + meta)

        effect_mean = self.effect_head(fused).reshape(b, n, h, w)
        log_sigma = self.log_sigma_head(fused).reshape(b, n, h, w).clamp(-6.0, 3.0)
        sigma = (F.softplus(log_sigma) + 1.0e-4).clamp_max(2.0)
        harm_logit = self.harm_head(fused).reshape(b, n, h, w)
        harm_prob = torch.sigmoid(harm_logit)
        eligibility_logit = self.eligibility_head(fused).reshape(b, n, h, w)
        eligibility_prob = torch.sigmoid(eligibility_logit)

        score = (
            effect_mean
            - self.lcb_kappa * sigma
            - self.harm_penalty * harm_prob
            - self.edit_penalty * abs_edit
            + 0.05 * eligibility_logit
            - self.utility_margin
        )
        score = self._smooth_regions(score)

        candidate_deploy = deploy_mask[1:].to(device=c0.device, dtype=torch.bool)
        # Region support/executability: a candidate is selectable throughout a
        # local part when it has evidence and a real edit somewhere inside that
        # neighbourhood.  The selected candidate still equals Base outside its
        # own edit pixels, so this widens the decision region without inventing
        # new edits.
        if self.region_radius > 0:
            kernel = 2 * self.region_radius + 1
            region_support = F.max_pool2d(
                support.reshape(b * n, 1, h, w),
                kernel_size=kernel,
                stride=1,
                padding=self.region_radius,
            ).reshape(b, n, h, w)
            region_edit = F.max_pool2d(
                abs_edit.reshape(b * n, 1, h, w),
                kernel_size=kernel,
                stride=1,
                padding=self.region_radius,
            ).reshape(b, n, h, w)
        else:
            region_support = support
            region_edit = abs_edit
        executable = (
            candidate_deploy[None, :, None, None]
            & (region_support > self.support_floor)
            & (region_edit > self.edit_epsilon)
        )
        floor = -1.0e4 if score.dtype in (torch.float16, torch.bfloat16) else -1.0e9
        score = score.masked_fill(~executable, floor)

        preserve_score = score.new_zeros((b, 1, h, w))
        route_logits = torch.cat([preserve_score, score], dim=1)
        soft_weights = F.softmax(route_logits / self.temperature, dim=1)
        selected_index = route_logits.argmax(dim=1, keepdim=True)
        hard_weights = torch.zeros_like(soft_weights).scatter_(1, selected_index, 1.0)

        if self.training:
            route_weights = (
                hard_weights + soft_weights - soft_weights.detach()
                if self.straight_through_train
                else soft_weights
            )
        else:
            route_weights = hard_weights if self.hard_inference else soft_weights

        fused_prob = (route_weights * candidate_probs).sum(dim=1, keepdim=True)
        fused_prob = fused_prob.clamp(EPS, 1.0 - EPS)
        selected_score = route_logits.gather(1, selected_index)
        selected_nonbase = selected_index > 0

        return {
            "m2_fused_probs": fused_prob[:, 0],
            "m2_training_probs": (
                (soft_weights * candidate_probs).sum(dim=1, keepdim=True)
                .clamp(EPS, 1.0 - EPS)[:, 0]
            ),
            "m2_proposal_probs": fused_prob[:, 0],
            "m2_convex_probs": fused_prob[:, 0],
            "m2_edit_gate_prob": (1.0 - route_weights[:, :1]).clamp(0.0, 1.0),
            "m2_residual_map": fused_prob - c0,
            "v519_route_logits": route_logits,
            "v519_route_weights": route_weights,
            "v519_soft_route_weights": soft_weights,
            "v519_selected_index": selected_index[:, 0],
            "v519_selected_score": selected_score[:, 0],
            "v519_effect_mean": effect_mean,
            "v519_log_sigma": log_sigma,
            "v519_sigma": sigma,
            "v519_harm_logit": harm_logit,
            "v519_harm_prob": harm_prob,
            "v519_eligibility_logit": eligibility_logit,
            "v519_eligibility_prob": eligibility_prob,
            "v519_executable_mask": executable,
            "v519_region_support": region_support,
            "v519_region_edit": region_edit,
            "v519_abs_edit": abs_edit,
            "v519_nonbase_route_rate": selected_nonbase.float().mean(dim=(-2, -1)),
            "v519_candidate_edit_fraction": edit_fraction,
        }


class V520GateSelectorRegionComposer(nn.Module):
    """Preserve-safe two-stage regional composer for a V518 candidate bank.

    The old V519 policy put Preserve and every intervention in one flat
    candidate softmax.  With dozens of candidates that changes the prior as the
    bank grows and creates a train/deploy mismatch when training uses a dense
    convex mixture while inference uses one hard candidate.

    V520 separates the decision into two conditional questions:

    1. ``edit_gate``: should this local region leave Preserve at all?
    2. ``selector``: conditioned on editing, which executable M1 candidate
       should supply the actual Base-relative delta?

    Training uses straight-through hard gate and hard candidate selection.  The
    forward value is therefore exactly the deployed mask, while gradients use
    the corresponding sigmoid/softmax relaxations.  Candidate observations are
    expected to be detached by the caller, so M2 can never alter its own M1
    counterfactual labels.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 48,
        metadata_dim: int = 16,
        semantic_channels: int = 512,
        max_candidates: int = 64,
        selector_temperature: float = 0.35,
        gate_temperature: float = 1.0,
        harm_penalty: float = 0.50,
        edit_penalty: float = 0.02,
        utility_margin: float = 0.0,
        support_floor: float = 1.0e-4,
        edit_epsilon: float = 1.0e-4,
        region_radius: int = 2,
        dropout: float = 0.10,
        initial_edit_probability: float = 0.02,
        hard_inference: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_candidates = int(max_candidates)
        self.selector_temperature = max(float(selector_temperature), 1.0e-3)
        self.gate_temperature = max(float(gate_temperature), 1.0e-3)
        self.harm_penalty = max(float(harm_penalty), 0.0)
        self.edit_penalty = max(float(edit_penalty), 0.0)
        self.utility_margin = float(utility_margin)
        self.support_floor = max(float(support_floor), 0.0)
        self.edit_epsilon = max(float(edit_epsilon), 0.0)
        self.region_radius = max(int(region_radius), 0)
        self.dropout = max(float(dropout), 0.0)
        self.hard_inference = bool(hard_inference)

        groups = _groups(self.hidden_dim)
        self.context_encoder = nn.Sequential(
            nn.Conv2d(6, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
        )
        self.semantic_proj = nn.Sequential(
            nn.Conv2d(int(semantic_channels), self.hidden_dim, 1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Conv2d(7, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
        )
        self.family_embedding = nn.Embedding(5, int(metadata_dim))
        self.action_embedding = nn.Embedding(10, int(metadata_dim))
        self.metadata_proj = nn.Sequential(
            nn.Linear(int(metadata_dim) * 2 + 3, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.candidate_fuse = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
            nn.Dropout2d(self.dropout),
        )
        self.utility_head = nn.Conv2d(self.hidden_dim, 1, 1)
        self.harm_head = nn.Conv2d(self.hidden_dim, 1, 1)

        # The gate is independent of candidate count.  It observes the shared
        # image/Base context, the strongest candidate feature, and three bank
        # summary maps (support, actual edit and cause evidence).
        self.gate_fuse = nn.Sequential(
            nn.Conv2d(2 * self.hidden_dim + 3, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
            nn.Dropout2d(self.dropout),
            nn.Conv2d(self.hidden_dim, 1, 1),
        )

        nn.init.zeros_(self.utility_head.weight)
        nn.init.zeros_(self.utility_head.bias)
        nn.init.zeros_(self.harm_head.weight)
        # A mildly optimistic "not harmful" prior avoids the old 0.5 harm
        # penalty while still requiring learned positive utility to edit.
        nn.init.constant_(self.harm_head.bias, -2.0)
        nn.init.zeros_(self.gate_fuse[-1].weight)
        p0 = min(max(float(initial_edit_probability), 1.0e-4), 1.0 - 1.0e-4)
        nn.init.constant_(self.gate_fuse[-1].bias, float(torch.log(torch.tensor(p0 / (1.0 - p0)))))

    def _semantic_map(
        self,
        semantic_map: torch.Tensor | None,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(semantic_map, torch.Tensor):
            return reference.new_zeros(
                reference.shape[0], self.hidden_dim, *reference.shape[-2:]
            )
        semantic = semantic_map
        if semantic.shape[-2:] != reference.shape[-2:]:
            semantic = F.interpolate(
                semantic,
                size=reference.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return self.semantic_proj(semantic)

    def _smooth(self, value: torch.Tensor) -> torch.Tensor:
        if self.region_radius <= 0:
            return value
        kernel = 2 * self.region_radius + 1
        return F.avg_pool2d(
            value,
            kernel_size=kernel,
            stride=1,
            padding=self.region_radius,
        )

    def _region_max(self, value: torch.Tensor) -> torch.Tensor:
        if self.region_radius <= 0:
            return value
        b, n, h, w = value.shape
        kernel = 2 * self.region_radius + 1
        return F.max_pool2d(
            value.reshape(b * n, 1, h, w),
            kernel_size=kernel,
            stride=1,
            padding=self.region_radius,
        ).reshape(b, n, h, w)

    def forward(
        self,
        *,
        image: torch.Tensor,
        c0_prob: torch.Tensor,
        candidate_probs: torch.Tensor,
        supports: torch.Tensor,
        candidate_causes: torch.Tensor,
        family_ids: torch.Tensor,
        action_ids: torch.Tensor,
        dose_values: torch.Tensor,
        radius_values: torch.Tensor,
        deploy_mask: torch.Tensor,
        semantic_map: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        c0 = _as_b1hw(c0_prob).clamp(EPS, 1.0 - EPS)
        if candidate_probs.ndim != 4:
            raise ValueError("candidate_probs must be [B,K,H,W]")
        b, k, h, w = candidate_probs.shape
        if k < 2 or k > self.max_candidates:
            raise ValueError(
                f"V520 requires 2..{self.max_candidates} candidates, got {k}"
            )
        if supports.shape != candidate_probs.shape:
            raise ValueError("supports must match candidate_probs")
        if candidate_causes.shape != candidate_probs.shape:
            raise ValueError("candidate_causes must match candidate_probs")
        metadata_tensors = (
            family_ids,
            action_ids,
            dose_values,
            radius_values,
            deploy_mask,
        )
        if any(t.ndim != 1 or t.numel() != k for t in metadata_tensors):
            raise ValueError("V520 candidate metadata must have K entries")

        ent = _entropy(c0)
        bnd = _soft_boundary(c0)
        gray, edge = _gray_edge(image, (h, w))
        shared = self.context_encoder(
            torch.cat([c0, 1.0 - c0, ent, bnd, gray, edge], dim=1)
        )
        shared = shared + self._semantic_map(semantic_map, c0)

        nonbase = candidate_probs[:, 1:].clamp(EPS, 1.0 - EPS)
        n = nonbase.shape[1]
        c0_bank = c0.expand(-1, n, -1, -1)
        signed_delta = nonbase - c0_bank
        abs_edit = signed_delta.abs()
        support = supports[:, 1:].clamp(0.0, 1.0)
        cause = candidate_causes[:, 1:].clamp(0.0, 1.0)
        ent_bank = ent.expand(-1, n, -1, -1)
        bnd_bank = bnd.expand(-1, n, -1, -1)

        candidate_input = torch.stack(
            [nonbase, signed_delta, abs_edit, support, cause, ent_bank, bnd_bank],
            dim=2,
        ).reshape(b * n, 7, h, w)
        candidate_feature = self.candidate_encoder(candidate_input)

        family = family_ids[1:].to(device=c0.device, dtype=torch.long).clamp(0, 4)
        action = action_ids[1:].to(device=c0.device, dtype=torch.long).clamp(0, 9)
        dose = dose_values[1:].to(device=c0.device, dtype=c0.dtype)
        radius = radius_values[1:].to(device=c0.device, dtype=c0.dtype)
        log_dose = torch.log2(dose.clamp_min(1.0e-3)) / 3.0
        radius_norm = radius / radius.max().clamp_min(1.0)
        edit_fraction = abs_edit.flatten(2).mean(dim=2)
        fixed_meta = torch.cat(
            [
                self.family_embedding(family),
                self.action_embedding(action),
                log_dose[:, None],
                radius_norm[:, None],
            ],
            dim=1,
        )[None].expand(b, -1, -1)
        meta_input = torch.cat([fixed_meta, edit_fraction[..., None]], dim=2)
        meta = self.metadata_proj(meta_input.reshape(b * n, -1))[:, :, None, None]

        shared_bank = shared[:, None].expand(-1, n, -1, -1, -1)
        shared_bank = shared_bank.reshape(b * n, self.hidden_dim, h, w)
        fused = self.candidate_fuse(candidate_feature + shared_bank + meta)
        fused_bank = fused.reshape(b, n, self.hidden_dim, h, w)

        utility = self.utility_head(fused).reshape(b, n, h, w)
        harm_logit = self.harm_head(fused).reshape(b, n, h, w)
        harm_prob = torch.sigmoid(harm_logit)

        region_support = self._region_max(support)
        region_edit = self._region_max(abs_edit)
        region_cause = self._region_max(cause)
        candidate_deploy = deploy_mask[1:].to(device=c0.device, dtype=torch.bool)
        executable = (
            candidate_deploy[None, :, None, None]
            & (region_support > self.support_floor)
            & (region_edit > self.edit_epsilon)
        )

        selector_logits = (
            utility
            - self.harm_penalty * harm_prob
            - self.edit_penalty * abs_edit
            - self.utility_margin
        )
        selector_logits = self._smooth(selector_logits)
        floor = -1.0e4 if selector_logits.dtype in (torch.float16, torch.bfloat16) else -1.0e9
        selector_logits = selector_logits.masked_fill(~executable, floor)
        any_executable = executable.any(dim=1, keepdim=True)

        # Aggregate only real candidate information.  Non-executable features
        # are replaced by a large negative value before max pooling and the
        # aggregate is zeroed when the entire bank is unavailable.
        feature_mask = executable[:, :, None]
        feature_floor = torch.finfo(fused_bank.dtype).min
        strongest_feature = fused_bank.masked_fill(~feature_mask, feature_floor).amax(dim=1)
        strongest_feature = torch.where(
            any_executable.expand(-1, self.hidden_dim, -1, -1),
            strongest_feature,
            torch.zeros_like(strongest_feature),
        )
        gate_input = torch.cat(
            [
                shared,
                strongest_feature,
                region_support.amax(dim=1, keepdim=True),
                region_edit.amax(dim=1, keepdim=True),
                region_cause.amax(dim=1, keepdim=True),
            ],
            dim=1,
        )
        gate_logit = self._smooth(self.gate_fuse(gate_input))
        gate_logit = gate_logit.masked_fill(~any_executable, -20.0)
        soft_gate = torch.sigmoid(gate_logit / self.gate_temperature)
        hard_gate = (gate_logit > 0.0).to(dtype=c0.dtype)

        soft_selector = F.softmax(
            selector_logits / self.selector_temperature,
            dim=1,
        )
        selected_nonbase = selector_logits.argmax(dim=1, keepdim=True)
        hard_selector = torch.zeros_like(soft_selector).scatter_(
            1, selected_nonbase, 1.0
        )
        # No executable candidate means the conditional selector has no effect.
        soft_selector = soft_selector * any_executable.to(dtype=soft_selector.dtype)
        hard_selector = hard_selector * any_executable.to(dtype=hard_selector.dtype)

        soft_route = torch.cat(
            [1.0 - soft_gate, soft_gate * soft_selector],
            dim=1,
        )
        hard_route = torch.cat(
            [1.0 - hard_gate, hard_gate * hard_selector],
            dim=1,
        )
        if self.training:
            route_weights = hard_route + soft_route - soft_route.detach()
        else:
            route_weights = hard_route if self.hard_inference else soft_route

        fused_prob = (route_weights * candidate_probs).sum(dim=1, keepdim=True)
        fused_prob = fused_prob.clamp(EPS, 1.0 - EPS)
        selected_score = selector_logits.gather(1, selected_nonbase)
        selected_index = torch.where(
            hard_gate > 0.5,
            selected_nonbase + 1,
            torch.zeros_like(selected_nonbase),
        )

        return {
            "m2_fused_probs": fused_prob[:, 0],
            # ST forward value is exactly the hard deployed result.  This alias
            # intentionally equals m2_fused_probs in both train and eval.
            "m2_training_probs": fused_prob[:, 0],
            "m2_proposal_probs": fused_prob[:, 0],
            "m2_convex_probs": fused_prob[:, 0],
            "m2_edit_gate_logit": gate_logit,
            "m2_edit_gate_prob": soft_gate,
            "m2_residual_map": fused_prob - c0,
            "v520_gate_logit": gate_logit,
            "v520_soft_gate": soft_gate,
            "v520_hard_gate": hard_gate,
            "v520_selector_logits": selector_logits,
            "v520_soft_selector_weights": soft_selector,
            "v520_hard_selector_weights": hard_selector,
            "v520_soft_route_weights": soft_route,
            "v520_hard_route_weights": hard_route,
            "v520_route_weights": route_weights,
            "v520_selected_nonbase_index": selected_nonbase[:, 0],
            "v520_selected_index": selected_index[:, 0],
            "v520_selected_score": selected_score[:, 0],
            "v520_utility_pred": utility,
            "v520_harm_logit": harm_logit,
            "v520_harm_prob": harm_prob,
            "v520_executable_mask": executable,
            "v520_any_executable": any_executable,
            "v520_region_support": region_support,
            "v520_region_edit": region_edit,
            "v520_region_cause": region_cause,
            "v520_abs_edit": abs_edit,
            "v520_signed_delta": signed_delta,
            "v520_candidate_edit_fraction": edit_fraction,
            "v520_hard_nonbase_rate": hard_gate.mean(dim=(-2, -1)),
            "v520_soft_nonbase_rate": soft_gate.mean(dim=(-2, -1)),
        }
