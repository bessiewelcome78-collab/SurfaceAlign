"""V522 PWO-distilled realizable sequential local composer.

The module keeps M1 unchanged and turns its candidate bank into a deployable
local correction policy.  It factorises M2 into two questions:

1. Gate: is the current mask wrong at this candidate-supported pixel?
2. Source: which *realizable* M1 candidate should provide the replacement?

The hard forward path is Preserve-first and candidate-realizable.  Training
uses straight-through gates/routes so the tensor seen by downstream losses is
exactly the hard deployed tensor while gradients still reach the gate and
source networks.  A dynamic connected component around the most confident
remaining error is edited at each step; edited pixels are locked before the
next step.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.v503_factual_atomic_causal import EPS, _as_b1hw, _entropy, _gray_edge, _soft_boundary


def _groups(channels: int) -> int:
    groups = min(8, int(channels))
    while groups > 1 and int(channels) % groups != 0:
        groups -= 1
    return groups


def _resize_4d(value: torch.Tensor, size: Tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
    if value.shape[-2:] == size:
        return value
    if mode == "nearest":
        return F.interpolate(value, size=size, mode=mode)
    return F.interpolate(value, size=size, mode=mode, align_corners=False)


def _resize_5d_candidates(value: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Resize [B,N,C,H,W] candidate tensors without a Python candidate loop."""
    b, n, c, h, w = value.shape
    flat = value.reshape(b * n, c, h, w)
    flat = _resize_4d(flat, size, mode="bilinear")
    return flat.reshape(b, n, c, *size)


class V522PWODistilledSequentialLocalComposer(nn.Module):
    """Candidate-supported PWO gate + multi-positive source + local replay.

    Preserve is not represented by another learned candidate score.  A pixel is
    editable only when all of the following are true in the hard forward path:

    * at least one deployable M1 candidate differs from the current mask;
    * the pixel belongs to the selected dynamic connected component;
    * the predicted repair gate exceeds its threshold;
    * the best candidate utility lower-confidence bound is positive.

    At every accepted step exactly one connected component is exposed, but the
    candidate source can vary inside that component.  This is the closest
    deployable approximation to PWO that remains fully realizable by the M1
    candidate bank.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 48,
        metadata_dim: int = 16,
        semantic_channels: int = 512,
        max_candidates: int = 64,
        score_stride: int = 4,
        max_steps: int = 3,
        support_floor: float = 1.0e-4,
        edit_epsilon: float = 1.0e-4,
        source_temperature: float = 0.20,
        lcb_beta: float = 1.0,
        min_candidate_lcb: float = 0.0,
        gate_threshold: float = 0.55,
        component_threshold: float = 0.30,
        component_growth_steps: int = 48,
        component_dilation_radius: int = 2,
        min_component_pixels: int = 4,
        initial_edit_probability: float = 0.02,
        deploy_start_epoch: int = 10,
        full_deploy_epoch: int = 30,
        dropout: float = 0.10,
        hard_inference: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_candidates = int(max_candidates)
        self.score_stride = max(int(score_stride), 1)
        self.max_steps = max(int(max_steps), 1)
        self.support_floor = max(float(support_floor), 0.0)
        self.edit_epsilon = max(float(edit_epsilon), 0.0)
        self.source_temperature = max(float(source_temperature), 1.0e-3)
        self.lcb_beta = max(float(lcb_beta), 0.0)
        self.min_candidate_lcb = float(min_candidate_lcb)
        self.gate_threshold = min(max(float(gate_threshold), 1.0e-4), 1.0 - 1.0e-4)
        self.component_threshold = min(max(float(component_threshold), 0.0), 1.0)
        self.component_growth_steps = max(int(component_growth_steps), 1)
        self.component_dilation_radius = max(int(component_dilation_radius), 0)
        self.min_component_pixels = max(int(min_component_pixels), 1)
        self.deploy_start_epoch = max(int(deploy_start_epoch), 0)
        self.full_deploy_epoch = max(int(full_deploy_epoch), self.deploy_start_epoch)
        self.dropout = max(float(dropout), 0.0)
        self.hard_inference = bool(hard_inference)
        self.current_epoch = 0

        groups = _groups(self.hidden_dim)
        self.context_encoder = nn.Sequential(
            nn.Conv2d(12, self.hidden_dim, 3, padding=1, bias=False),
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
        self.gate_decoder = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
            nn.Dropout2d(self.dropout),
        )
        self.gate_head = nn.Conv2d(self.hidden_dim, 1, 1)

        self.candidate_encoder = nn.Sequential(
            nn.Conv2d(8, self.hidden_dim, 3, padding=1, bias=False),
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
        self.candidate_fuse = nn.Sequential(
            nn.Conv2d(3 * self.hidden_dim, self.hidden_dim, 1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
            nn.Dropout2d(self.dropout),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
        )
        self.utility_mean_head = nn.Conv2d(self.hidden_dim, 1, 1)
        self.utility_logvar_head = nn.Conv2d(self.hidden_dim, 1, 1)

        nn.init.zeros_(self.gate_head.weight)
        initial = min(max(float(initial_edit_probability), 1.0e-4), 1.0 - 1.0e-4)
        nn.init.constant_(self.gate_head.bias, float(torch.logit(torch.tensor(initial))))
        nn.init.zeros_(self.utility_mean_head.weight)
        nn.init.zeros_(self.utility_mean_head.bias)
        nn.init.zeros_(self.utility_logvar_head.weight)
        nn.init.constant_(self.utility_logvar_head.bias, -2.0)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def _semantic_map(
        self,
        semantic_map: torch.Tensor | None,
        *,
        batch: int,
        size: Tuple[int, int],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(semantic_map, torch.Tensor):
            return reference.new_zeros(batch, self.hidden_dim, *size)
        semantic = _resize_4d(semantic_map, size, mode="bilinear")
        return self.semantic_proj(semantic)

    def _active_steps(self) -> int:
        if not self.training:
            return self.max_steps
        if self.current_epoch < self.full_deploy_epoch:
            return 1
        return self.max_steps

    @staticmethod
    def _candidate_metadata(
        *,
        family_ids: torch.Tensor,
        action_ids: torch.Tensor,
        dose_values: torch.Tensor,
        radius_values: torch.Tensor,
        global_edit_fraction: torch.Tensor,
        family_embedding: nn.Embedding,
        action_embedding: nn.Embedding,
        metadata_proj: nn.Module,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        family = family_ids[1:].long().clamp(0, 4)
        action = action_ids[1:].long().clamp(0, 9)
        dose = dose_values[1:].to(dtype=dtype)
        radius = radius_values[1:].to(dtype=dtype)
        dose_norm = dose / dose.max().clamp_min(1.0)
        radius_norm = radius / radius.max().clamp_min(1.0)
        log_dose = torch.log2(dose.clamp_min(1.0e-3)) / 3.0
        fixed = torch.cat(
            [
                family_embedding(family),
                action_embedding(action),
                log_dose[:, None],
                radius_norm[:, None],
                dose_norm[:, None],
            ],
            dim=1,
        )
        b = global_edit_fraction.shape[0]
        fixed = fixed[None].expand(b, -1, -1)
        features = torch.cat([fixed, global_edit_fraction[..., None]], dim=2)
        return metadata_proj(features.reshape(b * features.shape[1], -1)).reshape(
            b, features.shape[1], -1
        )

    def _connected_component_from_peak(
        self,
        *,
        heat: torch.Tensor,
        valid_union: torch.Tensor,
        proposal_prior: torch.Tensor,
    ) -> torch.Tensor:
        """Extract one 8-connected component around the highest remaining seed.

        Region construction is deliberately stop-gradient.  Gate/source losses
        supervise every realizable pixel, so no learning signal depends on a
        derivative through connected-components.  During training only, an
        empty predicted component falls back to a candidate-supported component;
        the hard gate is still Preserve before it becomes confident.
        """
        heat_detached = heat.detach().clamp(0.0, 1.0)
        valid = valid_union.detach().bool()
        active = (heat_detached >= self.component_threshold) & valid
        has_active = active.flatten(1).any(dim=1)
        if self.training:
            fallback = valid & (proposal_prior.detach() > self.support_floor)
            support = torch.where(has_active[:, None, None, None], active, fallback)
            seed_score = torch.where(
                has_active[:, None, None, None],
                heat_detached,
                proposal_prior.detach() * fallback.to(heat_detached.dtype),
            )
        else:
            support = active
            seed_score = heat_detached

        b, _, h, w = support.shape
        flat_score = seed_score.flatten(1).masked_fill(~support.flatten(1), -1.0)
        seed_index = flat_score.argmax(dim=1)
        has_support = support.flatten(1).any(dim=1)
        seed = torch.zeros((b, h * w), device=heat.device, dtype=heat.dtype)
        seed.scatter_(1, seed_index[:, None], 1.0)
        seed = seed.reshape(b, 1, h, w) * has_support[:, None, None, None].to(heat.dtype)

        component = seed
        support_float = support.to(heat.dtype)
        for _ in range(self.component_growth_steps):
            component = F.max_pool2d(component, kernel_size=3, stride=1, padding=1)
            component = component * support_float

        if self.component_dilation_radius > 0:
            kernel = 2 * self.component_dilation_radius + 1
            component = F.max_pool2d(
                component,
                kernel_size=kernel,
                stride=1,
                padding=self.component_dilation_radius,
            )
            component = component * valid.to(heat.dtype)

        enough = component.flatten(1).sum(dim=1) >= float(self.min_component_pixels)
        return component * enough[:, None, None, None].to(component.dtype)

    def _score_step(
        self,
        *,
        image: torch.Tensor,
        c0: torch.Tensor,
        current: torch.Tensor,
        nonbase: torch.Tensor,
        support: torch.Tensor,
        cause: torch.Tensor,
        candidate_deploy: torch.Tensor,
        metadata: torch.Tensor,
        semantic_map: torch.Tensor | None,
        lock_map: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        b, n, h, w = nonbase.shape
        low_h = max(1, (h + self.score_stride - 1) // self.score_stride)
        low_w = max(1, (w + self.score_stride - 1) // self.score_stride)
        low_size = (low_h, low_w)

        c0_low = _resize_4d(c0, low_size)
        current_low = _resize_4d(current, low_size)
        nonbase_low = _resize_4d(nonbase.reshape(b * n, 1, h, w), low_size).reshape(
            b, n, low_h, low_w
        )
        support_low = _resize_4d(support.reshape(b * n, 1, h, w), low_size).reshape(
            b, n, low_h, low_w
        )
        cause_low = _resize_4d(cause.reshape(b * n, 1, h, w), low_size).reshape(
            b, n, low_h, low_w
        )
        gray, edge = _gray_edge(image, low_size)
        entropy = _entropy(current_low)
        boundary = _soft_boundary(current_low)
        signed = nonbase_low - current_low.expand(-1, n, -1, -1)
        abs_edit = signed.abs()
        support_union = support_low.amax(dim=1, keepdim=True)
        cause_union = cause_low.amax(dim=1, keepdim=True)
        max_edit = abs_edit.amax(dim=1, keepdim=True)
        mean_edit = abs_edit.mean(dim=1, keepdim=True)
        lock_low = _resize_4d(lock_map, low_size, mode="nearest")

        context_input = torch.cat(
            [
                c0_low,
                current_low,
                (current_low - c0_low).abs(),
                entropy,
                boundary,
                gray,
                edge,
                max_edit,
                mean_edit,
                support_union,
                cause_union,
                lock_low,
            ],
            dim=1,
        )
        shared = self.context_encoder(context_input)
        shared = shared + self._semantic_map(
            semantic_map,
            batch=b,
            size=low_size,
            reference=current,
        )
        gate_logit_low = self.gate_head(self.gate_decoder(shared))

        current_bank = current_low[:, None].expand(-1, n, -1, -1, -1)
        boundary_bank = boundary[:, None].expand(-1, n, -1, -1, -1)
        entropy_bank = entropy[:, None].expand(-1, n, -1, -1, -1)
        candidate_input = torch.stack(
            [
                nonbase_low,
                current_low.expand(-1, n, -1, -1),
                signed,
                abs_edit,
                support_low,
                cause_low,
                boundary.expand(-1, n, -1, -1),
                entropy.expand(-1, n, -1, -1),
            ],
            dim=2,
        ).reshape(b * n, 8, low_h, low_w)
        candidate_feature = self.candidate_encoder(candidate_input)
        shared_bank = shared[:, None].expand(-1, n, -1, -1, -1).reshape(
            b * n, self.hidden_dim, low_h, low_w
        )
        metadata_bank = metadata.reshape(b * n, self.hidden_dim, 1, 1).expand(
            -1, -1, low_h, low_w
        )
        fused = self.candidate_fuse(
            torch.cat([shared_bank, candidate_feature, metadata_bank], dim=1)
        )
        utility_mean_low = self.utility_mean_head(fused).reshape(b, n, low_h, low_w)
        utility_logvar_low = self.utility_logvar_head(fused).reshape(b, n, low_h, low_w).clamp(-6.0, 4.0)

        gate_logit = _resize_4d(gate_logit_low, (h, w))
        utility_mean = _resize_4d(
            utility_mean_low.reshape(b * n, 1, low_h, low_w), (h, w)
        ).reshape(b, n, h, w)
        utility_logvar = _resize_4d(
            utility_logvar_low.reshape(b * n, 1, low_h, low_w), (h, w)
        ).reshape(b, n, h, w).clamp(-6.0, 4.0)
        utility_std = torch.sqrt(torch.exp(utility_logvar).clamp_min(EPS))
        lcb = utility_mean - self.lcb_beta * utility_std

        current_bank_full = current.expand(-1, n, -1, -1)
        abs_edit_full = (nonbase - current_bank_full).abs()
        valid = (
            candidate_deploy[None, :, None, None]
            & (support > self.support_floor)
            & (abs_edit_full > self.edit_epsilon)
            & (lock_map < 0.5)
        )
        floor = -1.0e4 if lcb.dtype in (torch.float16, torch.bfloat16) else -1.0e9
        source_logits = (lcb / self.source_temperature).masked_fill(~valid, floor)
        source_soft = F.softmax(source_logits, dim=1)
        any_valid = valid.any(dim=1, keepdim=True)
        source_soft = source_soft * any_valid.to(source_soft.dtype)
        best_index = source_logits.argmax(dim=1)
        source_hard = torch.zeros_like(source_soft).scatter_(1, best_index[:, None], 1.0)
        source_hard = source_hard * any_valid.to(source_hard.dtype)
        if self.training:
            source_weights = source_hard + source_soft - source_soft.detach()
        else:
            source_weights = source_hard if self.hard_inference else source_soft

        selected_candidate = (source_weights * nonbase).sum(dim=1, keepdim=True)
        max_lcb = lcb.masked_fill(~valid, floor).max(dim=1, keepdim=True).values
        max_lcb = torch.where(any_valid, max_lcb, torch.full_like(max_lcb, floor))
        gate_prob = torch.sigmoid(gate_logit)
        support_union_full = support.amax(dim=1, keepdim=True)
        cause_union_full = cause.amax(dim=1, keepdim=True)
        edit_union = valid.any(dim=1, keepdim=True)
        confidence = torch.sigmoid(max_lcb / self.source_temperature)
        heat = gate_prob * confidence * edit_union.to(gate_prob.dtype) * (1.0 - lock_map)
        proposal_prior = torch.maximum(support_union_full, cause_union_full)
        proposal_prior = torch.maximum(
            proposal_prior,
            abs_edit_full.amax(dim=1, keepdim=True),
        )
        roi = self._connected_component_from_peak(
            heat=heat,
            valid_union=edit_union & (lock_map < 0.5),
            proposal_prior=proposal_prior,
        )

        hard_gate = (
            (gate_prob >= self.gate_threshold)
            & (max_lcb > self.min_candidate_lcb)
            & (roi > 0.5)
            & edit_union
            & (lock_map < 0.5)
        )
        if self.training and self.current_epoch < self.deploy_start_epoch:
            hard_gate = torch.zeros_like(hard_gate)
        soft_gate = gate_prob * roi * edit_union.to(gate_prob.dtype) * (1.0 - lock_map)
        if self.training:
            gate_weight = hard_gate.to(gate_prob.dtype) + soft_gate - soft_gate.detach()
        else:
            gate_weight = hard_gate.to(gate_prob.dtype) if self.hard_inference else soft_gate

        next_current = (
            current + gate_weight * (selected_candidate - current)
        ).clamp(EPS, 1.0 - EPS)
        selected_index = best_index + 1
        selected_index = torch.where(
            hard_gate[:, 0],
            selected_index,
            torch.zeros_like(selected_index),
        )
        return {
            "current_before": current,
            "next_current": next_current,
            "gate_logit": gate_logit,
            "gate_prob": gate_prob,
            "gate_weight": gate_weight,
            "hard_gate": hard_gate.to(gate_prob.dtype),
            "roi": roi,
            "source_logits": source_logits,
            "source_soft": source_soft,
            "source_hard": source_hard,
            "source_weights": source_weights,
            "utility_mean": utility_mean,
            "utility_logvar": utility_logvar,
            "utility_std": utility_std,
            "lcb": lcb,
            "candidate_valid": valid,
            "selected_index": selected_index,
            "max_lcb": max_lcb,
            "edit_union": edit_union.to(gate_prob.dtype),
            "proposal_prior": proposal_prior,
            "selected_candidate": selected_candidate,
        }

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
            raise ValueError(f"V522 requires 2..{self.max_candidates} candidates, got {k}")
        if supports.shape != candidate_probs.shape:
            raise ValueError("supports must match candidate_probs")
        if candidate_causes.shape != candidate_probs.shape:
            raise ValueError("candidate_causes must match candidate_probs")
        for tensor in (family_ids, action_ids, dose_values, radius_values, deploy_mask):
            if tensor.ndim != 1 or tensor.numel() != k:
                raise ValueError("V522 candidate metadata must have K entries")

        nonbase = candidate_probs[:, 1:].clamp(EPS, 1.0 - EPS)
        support = supports[:, 1:].clamp(0.0, 1.0)
        cause = candidate_causes[:, 1:].clamp(0.0, 1.0)
        n = nonbase.shape[1]
        candidate_deploy = deploy_mask[1:].to(device=c0.device, dtype=torch.bool)
        global_edit_fraction = (nonbase - c0.expand(-1, n, -1, -1)).abs().flatten(2).mean(dim=2)
        metadata = self._candidate_metadata(
            family_ids=family_ids.to(c0.device),
            action_ids=action_ids.to(c0.device),
            dose_values=dose_values.to(c0.device),
            radius_values=radius_values.to(c0.device),
            global_edit_fraction=global_edit_fraction,
            family_embedding=self.family_embedding,
            action_embedding=self.action_embedding,
            metadata_proj=self.metadata_proj,
            dtype=c0.dtype,
        )

        current = c0
        lock_map = torch.zeros_like(c0)
        steps: List[Dict[str, torch.Tensor]] = []
        for _ in range(self._active_steps()):
            step = self._score_step(
                image=image,
                c0=c0,
                current=current,
                nonbase=nonbase,
                support=support,
                cause=cause,
                candidate_deploy=candidate_deploy,
                metadata=metadata,
                semantic_map=semantic_map,
                lock_map=lock_map,
            )
            steps.append(step)
            current = step["next_current"]
            # Hard forward lock: later actions cannot overwrite accepted edits.
            lock_map = torch.maximum(lock_map, step["hard_gate"].detach())

        def stack(name: str, dim: int = 1) -> torch.Tensor:
            return torch.stack([step[name] for step in steps], dim=dim)

        final_prob = current.clamp(EPS, 1.0 - EPS)
        hard_edit_union = stack("hard_gate").amax(dim=1)
        selected_pixel_index = stack("selected_index")
        return {
            "m2_fused_probs": final_prob[:, 0],
            "m2_training_probs": final_prob[:, 0],
            "m2_proposal_probs": final_prob[:, 0],
            "m2_convex_probs": final_prob[:, 0],
            "m2_edit_gate_prob": hard_edit_union,
            "m2_residual_map": final_prob - c0,
            "v522_step_current_before": stack("current_before"),
            "v522_step_gate_logit": stack("gate_logit"),
            "v522_step_gate_prob": stack("gate_prob"),
            "v522_step_gate_weight": stack("gate_weight"),
            "v522_step_hard_edit_mask": stack("hard_gate"),
            "v522_step_roi_mask": stack("roi"),
            "v522_step_source_logits": stack("source_logits"),
            "v522_step_source_soft": stack("source_soft"),
            "v522_step_source_hard": stack("source_hard"),
            "v522_step_source_weights": stack("source_weights"),
            "v522_step_utility_mean": stack("utility_mean"),
            "v522_step_utility_logvar": stack("utility_logvar"),
            "v522_step_utility_std": stack("utility_std"),
            "v522_step_candidate_lcb": stack("lcb"),
            "v522_step_candidate_valid": stack("candidate_valid"),
            "v522_step_selected_pixel_index": selected_pixel_index,
            "v522_step_max_lcb": stack("max_lcb"),
            "v522_step_edit_union": stack("edit_union"),
            "v522_step_proposal_prior": stack("proposal_prior"),
            "v522_step_selected_candidate": stack("selected_candidate"),
            "v522_lock_map": lock_map,
            "v522_deploy_mask": deploy_mask,
            "v522_candidate_support": support,
            "v522_candidate_causes": cause,
            "v522_initial_abs_edit": (nonbase - c0.expand(-1, n, -1, -1)).abs(),
            "v522_num_steps": c0.new_full((b,), float(len(steps))),
            "v522_deploy_phase": c0.new_full(
                (b,),
                float(
                    0
                    if self.current_epoch < self.deploy_start_epoch
                    else 1
                    if self.current_epoch < self.full_deploy_epoch
                    else 2
                ),
            ),
            "v522_changed_pixel_rate": (
                (final_prob >= 0.5) != (c0 >= 0.5)
            ).float().flatten(1).mean(dim=1),
            "v522_selected_step_rate": stack("hard_gate").flatten(2).any(dim=2).float().mean(dim=1),
        }
