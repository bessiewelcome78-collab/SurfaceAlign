"""V521 candidate-conditional region utility composer.

V521 removes the independent edit gate and flat K-way pixel selector used by
V520.  It predicts Base-relative utility and harm for every candidate inside a
small set of deterministic, non-overlapping regions.  Preserve has fixed score
zero; an edit can be deployed only when the selected candidate's conservative
score is positive.

The forward value during training is exactly the hard deployed output.  A
straight-through soft route supplies gradients without creating a separate
soft-mixture training path.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.v503_factual_atomic_causal import EPS, _as_b1hw, _entropy, _gray_edge, _soft_boundary


def _groups(hidden_dim: int) -> int:
    groups = min(8, int(hidden_dim))
    while hidden_dim % groups != 0 and groups > 1:
        groups -= 1
    return groups


def _grid_masks(
    height: int,
    width: int,
    grid_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a non-overlapping grid partition [R,H,W]."""
    grid = max(int(grid_size), 1)
    masks = []
    for row in range(grid):
        y0 = int(round(row * height / grid))
        y1 = int(round((row + 1) * height / grid))
        for col in range(grid):
            x0 = int(round(col * width / grid))
            x1 = int(round((col + 1) * width / grid))
            mask = torch.zeros((height, width), device=device, dtype=dtype)
            mask[y0:y1, x0:x1] = 1.0
            masks.append(mask)
    return torch.stack(masks, dim=0)


def _masked_region_pool(feature: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Pool [B,C,H,W] into [B,R,C] with [B,R,H,W] masks."""
    denom = masks.flatten(2).sum(dim=2).clamp_min(1.0)
    pooled = torch.einsum("bchw,brhw->brc", feature, masks)
    return pooled / denom[..., None]


def _masked_candidate_region_pool(
    feature: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """Pool [B,N,C,H,W] into [B,R,N,C]."""
    denom = masks.flatten(2).sum(dim=2).clamp_min(1.0)
    pooled = torch.einsum("bnchw,brhw->brnc", feature, masks)
    return pooled / denom[:, :, None, None]


class V521CandidateConditionalRegionUtilityComposer(nn.Module):
    """One-action-per-region candidate utility composer.

    Each region/candidate pair receives a continuous utility prediction and a
    harm prediction.  Candidate scores are conservative Base-relative values;
    Preserve is represented by the exact zero score.  Consequently the edit
    decision and candidate selection are the same decision.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        metadata_dim: int = 16,
        semantic_channels: int = 512,
        max_candidates: int = 64,
        grid_size: int = 4,
        support_floor: float = 1.0e-4,
        edit_epsilon: float = 1.0e-4,
        region_dilation_radius: int = 1,
        active_region_threshold: float = 1.0e-4,
        student_temperature: float = 0.10,
        harm_penalty: float = 1.0,
        edit_penalty: float = 0.02,
        base_utility_threshold: float = 0.002,
        harm_prior_weight: float = 0.02,
        edit_prior_weight: float = 0.01,
        max_selected_regions: int = 4,
        deploy_start_epoch: int = 5,
        full_deploy_epoch: int = 20,
        dropout: float = 0.10,
        hard_inference: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_candidates = int(max_candidates)
        self.grid_size = max(int(grid_size), 1)
        self.support_floor = max(float(support_floor), 0.0)
        self.edit_epsilon = max(float(edit_epsilon), 0.0)
        self.region_dilation_radius = max(int(region_dilation_radius), 0)
        self.active_region_threshold = max(float(active_region_threshold), 0.0)
        self.student_temperature = max(float(student_temperature), 1.0e-3)
        self.harm_penalty = max(float(harm_penalty), 0.0)
        self.edit_penalty = max(float(edit_penalty), 0.0)
        self.base_utility_threshold = float(base_utility_threshold)
        self.harm_prior_weight = max(float(harm_prior_weight), 0.0)
        self.edit_prior_weight = max(float(edit_prior_weight), 0.0)
        self.max_selected_regions = max(int(max_selected_regions), 1)
        self.deploy_start_epoch = max(int(deploy_start_epoch), 0)
        self.full_deploy_epoch = max(int(full_deploy_epoch), self.deploy_start_epoch)
        self.dropout = max(float(dropout), 0.0)
        self.hard_inference = bool(hard_inference)
        self.current_epoch = 0

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
            nn.Linear(int(metadata_dim) * 2 + 4, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.region_fuse = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.utility_head = nn.Linear(self.hidden_dim, 1)
        self.harm_head = nn.Linear(self.hidden_dim, 1)
        self.validity_head = nn.Linear(self.hidden_dim, 1)

        nn.init.zeros_(self.utility_head.weight)
        nn.init.zeros_(self.utility_head.bias)
        nn.init.zeros_(self.harm_head.weight)
        nn.init.constant_(self.harm_head.bias, -2.0)
        nn.init.zeros_(self.validity_head.weight)
        nn.init.constant_(self.validity_head.bias, 1.0)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

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

    def _dilate(self, value: torch.Tensor) -> torch.Tensor:
        if self.region_dilation_radius <= 0:
            return value
        kernel = 2 * self.region_dilation_radius + 1
        return F.max_pool2d(
            value,
            kernel_size=kernel,
            stride=1,
            padding=self.region_dilation_radius,
        )

    @staticmethod
    def _static_risk_prior(
        family: torch.Tensor,
        action: torch.Tensor,
        radius_norm: torch.Tensor,
        dose_norm: torch.Tensor,
    ) -> torch.Tensor:
        # Pair/global and aggressive morphology have higher deployment risk.
        family_table = torch.tensor(
            [0.0, 0.05, 0.15, 0.10, 0.20],
            device=family.device,
            dtype=radius_norm.dtype,
        )
        action_table = torch.tensor(
            [0.0, 0.10, 0.08, 0.10, 0.10, 0.30, 0.25, 0.08, 0.00, 0.20],
            device=action.device,
            dtype=radius_norm.dtype,
        )
        return (
            family_table[family]
            + action_table[action]
            + 0.20 * radius_norm
            + 0.10 * dose_norm
        )

    def _curriculum_mask(
        self,
        *,
        family: torch.Tensor,
        action: torch.Tensor,
        dose: torch.Tensor,
        radius: torch.Tensor,
        deploy: torch.Tensor,
    ) -> torch.Tensor:
        if not self.training or self.current_epoch >= self.full_deploy_epoch:
            return deploy
        # Utility warm-up still scores every deployable candidate; hard output
        # is separately forced to Preserve before deploy_start_epoch.
        if self.current_epoch < self.deploy_start_epoch:
            return deploy
        learned_x1 = (family == 1) & (dose <= 1.01)
        safe_morph = (family == 3) & ((action == 7) | (action == 8)) & (radius <= 2.0)
        safe = learned_x1 | safe_morph
        return deploy & safe

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
            raise ValueError(f"V521 requires 2..{self.max_candidates} candidates, got {k}")
        if supports.shape != candidate_probs.shape:
            raise ValueError("supports must match candidate_probs")
        if candidate_causes.shape != candidate_probs.shape:
            raise ValueError("candidate_causes must match candidate_probs")
        for tensor in (family_ids, action_ids, dose_values, radius_values, deploy_mask):
            if tensor.ndim != 1 or tensor.numel() != k:
                raise ValueError("V521 candidate metadata must have K entries")

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
        candidate_feature = candidate_feature.reshape(b, n, self.hidden_dim, h, w)

        union = torch.maximum(
            support.amax(dim=1, keepdim=True),
            cause.amax(dim=1, keepdim=True),
        )
        union = torch.maximum(union, abs_edit.amax(dim=1, keepdim=True))
        union = self._dilate(union)
        base_masks = _grid_masks(
            h, w, self.grid_size, device=c0.device, dtype=c0.dtype
        )
        region_masks = base_masks[None].expand(b, -1, -1, -1)
        region_signal = torch.einsum("bchw,rhw->br", union, base_masks)
        region_area = base_masks.flatten(1).sum(dim=1).clamp_min(1.0)
        region_signal = region_signal / region_area[None]
        region_active = region_signal > self.active_region_threshold
        region_masks = region_masks * region_active[:, :, None, None].to(c0.dtype)
        r = region_masks.shape[1]

        shared_region = _masked_region_pool(shared, region_masks)
        candidate_region = _masked_candidate_region_pool(candidate_feature, region_masks)

        family = family_ids[1:].to(device=c0.device, dtype=torch.long).clamp(0, 4)
        action = action_ids[1:].to(device=c0.device, dtype=torch.long).clamp(0, 9)
        dose = dose_values[1:].to(device=c0.device, dtype=c0.dtype)
        radius = radius_values[1:].to(device=c0.device, dtype=c0.dtype)
        log_dose = torch.log2(dose.clamp_min(1.0e-3)) / 3.0
        radius_norm = radius / radius.max().clamp_min(1.0)
        dose_norm = dose / dose.max().clamp_min(1.0)
        global_edit_fraction = abs_edit.flatten(2).mean(dim=2)
        fixed_meta = torch.cat(
            [
                self.family_embedding(family),
                self.action_embedding(action),
                log_dose[:, None],
                radius_norm[:, None],
                dose_norm[:, None],
            ],
            dim=1,
        )[None].expand(b, -1, -1)
        meta_input = torch.cat(
            [fixed_meta, global_edit_fraction[..., None]], dim=2
        )
        meta = self.metadata_proj(meta_input.reshape(b * n, -1)).reshape(
            b, 1, n, self.hidden_dim
        ).expand(-1, r, -1, -1)

        shared_bank = shared_region[:, :, None, :].expand(-1, -1, n, -1)
        fused = self.region_fuse(
            torch.cat([shared_bank, candidate_region, meta], dim=-1)
        )
        utility = self.utility_head(fused)[..., 0]
        harm_logit = self.harm_head(fused)[..., 0]
        harm_prob = torch.sigmoid(harm_logit)
        validity_logit = self.validity_head(fused)[..., 0]
        validity_prob = torch.sigmoid(validity_logit)

        candidate_deploy = deploy_mask[1:].to(device=c0.device, dtype=torch.bool)
        candidate_support = torch.einsum("bnhw,brhw->brn", support, region_masks)
        candidate_edit = torch.einsum("bnhw,brhw->brn", abs_edit, region_masks)
        region_den = region_masks.flatten(2).sum(dim=2).clamp_min(1.0)
        candidate_support = candidate_support / region_den[:, :, None]
        candidate_edit = candidate_edit / region_den[:, :, None]
        region_candidate_valid = (
            region_active[:, :, None]
            & candidate_deploy[None, None, :]
            & (candidate_support > self.support_floor)
            & (candidate_edit > self.edit_epsilon)
        )

        risk_prior = self._static_risk_prior(
            family, action, radius_norm, dose_norm
        )
        candidate_threshold = (
            self.base_utility_threshold
            + self.harm_prior_weight * risk_prior[None, None, :]
            + self.edit_prior_weight * candidate_edit
        )
        nonbase_score = (
            utility
            - self.harm_penalty * harm_prob
            - self.edit_penalty * candidate_edit
            - candidate_threshold
        )
        floor = -1.0e4 if nonbase_score.dtype in (torch.float16, torch.bfloat16) else -1.0e9
        score_all = torch.cat(
            [torch.zeros((b, r, 1), device=c0.device, dtype=c0.dtype), nonbase_score],
            dim=2,
        )
        valid_all = torch.cat(
            [torch.ones((b, r, 1), device=c0.device, dtype=torch.bool), region_candidate_valid],
            dim=2,
        )
        score_all = score_all.masked_fill(~valid_all, floor)
        soft_route = F.softmax(score_all / self.student_temperature, dim=2)

        curriculum = self._curriculum_mask(
            family=family,
            action=action,
            dose=dose,
            radius=radius,
            deploy=candidate_deploy,
        )
        hard_valid = region_candidate_valid & curriculum[None, None, :]
        hard_score = nonbase_score.masked_fill(~hard_valid, floor)
        best_nonbase_score, best_nonbase_index = hard_score.max(dim=2)
        selected_index = best_nonbase_index + 1
        edit_region = region_active & (best_nonbase_score > 0.0)

        if self.training and self.current_epoch < self.deploy_start_epoch:
            edit_region = torch.zeros_like(edit_region)

        if self.max_selected_regions < r:
            ranking = best_nonbase_score.masked_fill(~edit_region, floor)
            topk = min(self.max_selected_regions, r)
            top_indices = ranking.topk(topk, dim=1).indices
            keep = torch.zeros_like(edit_region)
            keep.scatter_(1, top_indices, True)
            edit_region = edit_region & keep

        selected_index = torch.where(
            edit_region,
            selected_index,
            torch.zeros_like(selected_index),
        )
        hard_route = torch.zeros_like(score_all).scatter_(
            2, selected_index[..., None], 1.0
        )
        if self.training:
            route_weights = hard_route + soft_route - soft_route.detach()
        else:
            route_weights = hard_route if self.hard_inference else soft_route

        region_candidate_prob = torch.einsum(
            "brk,bkhw->brhw", route_weights, candidate_probs
        )
        region_delta = (region_candidate_prob - c0) * region_masks
        fused_prob = (c0 + region_delta.sum(dim=1, keepdim=True)).clamp(
            EPS, 1.0 - EPS
        )

        pixel_selected_index = torch.zeros((b, h, w), device=c0.device, dtype=torch.long)
        for region_id in range(r):
            mask = region_masks[:, region_id] > 0.5
            pixel_selected_index = torch.where(
                mask,
                selected_index[:, region_id, None, None].expand(-1, h, w),
                pixel_selected_index,
            )

        selected_nonbase_score = nonbase_score.gather(
            2, best_nonbase_index[..., None]
        )[..., 0]
        selected_score = torch.where(
            edit_region, selected_nonbase_score, torch.zeros_like(selected_nonbase_score)
        )

        return {
            "m2_fused_probs": fused_prob[:, 0],
            "m2_training_probs": fused_prob[:, 0],
            "m2_proposal_probs": fused_prob[:, 0],
            "m2_convex_probs": fused_prob[:, 0],
            "m2_edit_gate_prob": edit_region.float(),
            "m2_residual_map": fused_prob - c0,
            "v521_region_masks": region_masks,
            "v521_region_active": region_active,
            "v521_region_signal": region_signal,
            "v521_region_candidate_valid": region_candidate_valid,
            "v521_candidate_support": candidate_support,
            "v521_candidate_edit_fraction": candidate_edit,
            "v521_utility_pred": utility,
            "v521_harm_logit": harm_logit,
            "v521_harm_prob": harm_prob,
            "v521_validity_logit": validity_logit,
            "v521_validity_prob": validity_prob,
            "v521_candidate_threshold": candidate_threshold,
            "v521_nonbase_score": nonbase_score,
            "v521_score_all": score_all,
            "v521_soft_route_weights": soft_route,
            "v521_hard_route_weights": hard_route,
            "v521_route_weights": route_weights,
            "v521_selected_region_index": selected_index,
            "v521_selected_region_score": selected_score,
            "v521_selected_pixel_index": pixel_selected_index,
            "v521_edit_region_mask": edit_region,
            "v521_curriculum_candidate_mask": curriculum,
            "v521_abs_edit": abs_edit,
            "v521_signed_delta": signed_delta,
            "v521_deploy_phase": c0.new_full((b,), float(
                0 if self.current_epoch < self.deploy_start_epoch
                else 1 if self.current_epoch < self.full_deploy_epoch
                else 2
            )),
            "v521_selected_region_rate": edit_region.float().mean(dim=1),
            "v521_changed_pixel_rate": ((fused_prob >= 0.5) != (c0 >= 0.5)).float().flatten(1).mean(dim=1),
        }
