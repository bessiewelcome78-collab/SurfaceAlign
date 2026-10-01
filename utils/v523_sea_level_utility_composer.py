"""V523 sea-level utility-volume candidate composer.

This module keeps the validated V518 M1 candidate bank unchanged.  M2 treats
Preserve/Base as a zero-height sea level and predicts the *relative utility*
of every deployable candidate at every pixel.  All candidates and Preserve are
ranked in one action space; there is no separate Gate -> Source -> Utility
cascade.

The design deliberately combines mechanisms that have been validated in prior
segmentation/refinement and routing work:

* dense teacher supervision for all routes (teacher-guided sparse routing);
* shared candidate/expert encoding and dynamic spatial routing;
* preserve-first sparse refinement of candidate-supported error-prone pixels;
* straight-through hard deployment so train and inference tensors agree.

Hard deployment is candidate-realizable: every changed pixel is copied from a
real M1 candidate and Preserve is selected whenever no candidate is safely
above sea level.  GT is never an input to ``forward``; privileged PWO utility
is built only inside the training loss.
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




class _StraightThroughHardRoute(torch.autograd.Function):
    """Return the exact hard route while backpropagating through soft route."""

    @staticmethod
    def forward(ctx, hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
        del ctx, soft
        return hard

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        del ctx
        return None, grad_output


def _resize_4d(value: torch.Tensor, size: Tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
    if value.shape[-2:] == size:
        return value
    if mode == "nearest":
        return F.interpolate(value, size=size, mode=mode)
    return F.interpolate(value, size=size, mode=mode, align_corners=False)


class V523SeaLevelUtilityComposer(nn.Module):
    """Dense candidate utility terrain with an explicit zero Preserve plane.

    For every sequential state the network predicts one relative utility score
    and one harm logit for each non-Base candidate and pixel.  Preserve has a
    fixed score of zero.  A hard candidate action is accepted only when:

    * the candidate is deployable, supported and differs from the current mask;
    * its score is above the sea-level margin;
    * the top-1/top-2 score gap is sufficiently large;
    * predicted harm is below the risk threshold;
    * local candidate evidence is spatially coherent.

    During training, the hard route is used in the forward value while a dense
    soft route supplies gradients (straight-through estimator).  Thus the
    deployed tensor is always one of Base/candidate values at each pixel.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        metadata_dim: int = 16,
        semantic_channels: int = 512,
        max_candidates: int = 64,
        score_stride: int = 4,
        max_steps: int = 1,
        support_floor: float = 1.0e-4,
        edit_epsilon: float = 1.0e-4,
        student_temperature: float = 0.20,
        sea_level_margin: float = 0.05,
        top_gap_margin: float = 0.02,
        risk_threshold: float = 0.10,
        benefit_threshold: float = 0.65,
        risk_penalty: float = 2.0,
        hard_change_only: bool = True,
        coherence_kernel: int = 5,
        coherence_threshold: float = 0.40,
        coherence_alpha: float = 0.25,
        initial_edit_probability: float = 0.01,
        deploy_start_epoch: int = 5,
        full_deploy_epoch: int = 40,
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
        self.student_temperature = max(float(student_temperature), 1.0e-3)
        self.sea_level_margin = float(sea_level_margin)
        self.top_gap_margin = max(float(top_gap_margin), 0.0)
        self.risk_threshold = min(max(float(risk_threshold), 1.0e-4), 1.0 - 1.0e-4)
        self.benefit_threshold = min(max(float(benefit_threshold), 1.0e-4), 1.0 - 1.0e-4)
        self.risk_penalty = max(float(risk_penalty), 0.0)
        self.hard_change_only = bool(hard_change_only)
        kernel = max(int(coherence_kernel), 1)
        self.coherence_kernel = kernel if kernel % 2 == 1 else kernel + 1
        self.coherence_threshold = min(max(float(coherence_threshold), 0.0), 1.0)
        self.coherence_alpha = min(max(float(coherence_alpha), 0.0), 1.0)
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
        self.candidate_encoder = nn.Sequential(
            nn.Conv2d(10, self.hidden_dim, 3, padding=1, bias=False),
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
        self.utility_head = nn.Conv2d(self.hidden_dim, 1, 1)
        self.benefit_head = nn.Conv2d(self.hidden_dim, 1, 1)
        self.harm_head = nn.Conv2d(self.hidden_dim, 1, 1)

        # Preserve-first initialization.  Utility starts neutral, exact-repair
        # probability starts very low, and harm starts conservatively high.
        nn.init.zeros_(self.utility_head.weight)
        nn.init.zeros_(self.utility_head.bias)
        initial = min(max(float(initial_edit_probability), 1.0e-4), 1.0 - 1.0e-4)
        nn.init.zeros_(self.benefit_head.weight)
        nn.init.constant_(self.benefit_head.bias, float(torch.logit(torch.tensor(initial))))
        nn.init.zeros_(self.harm_head.weight)
        nn.init.constant_(self.harm_head.bias, 1.0)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def _active_steps(self) -> int:
        if not self.training:
            return self.max_steps
        if self.current_epoch < self.full_deploy_epoch:
            return 1
        return self.max_steps

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

    @staticmethod
    def oracle_compose(
        *,
        current_prob: torch.Tensor,
        candidate_probs: torch.Tensor,
        candidate_valid: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Training/debug-only exact candidate-realizable pixel oracle.

        This helper is never called by deployment ``forward``.  It proves that
        the V523 action space can express the candidate PWO: Preserve is action
        zero and every edited pixel copies one actual candidate value.
        """
        current = _as_b1hw(current_prob).clamp(EPS, 1.0 - EPS)
        target = _as_b1hw(target).clamp(0.0, 1.0)
        if candidate_probs.ndim != 4:
            raise ValueError("candidate_probs must be [B,N,H,W]")
        if candidate_valid.shape != candidate_probs.shape:
            raise ValueError("candidate_valid must match candidate_probs")
        n = candidate_probs.shape[1]
        current_bank = current.expand(-1, n, -1, -1)
        target_bank = target.expand(-1, n, -1, -1)
        current_loss = F.binary_cross_entropy(current_bank, target_bank, reduction="none")
        candidate_loss = F.binary_cross_entropy(
            candidate_probs.clamp(EPS, 1.0 - EPS), target_bank, reduction="none"
        )
        utility = (current_loss - candidate_loss).masked_fill(~candidate_valid.bool(), -1.0e9)
        all_scores = torch.cat([torch.zeros_like(current), utility], dim=1)
        best_index = all_scores.argmax(dim=1)
        bank = torch.cat([current, candidate_probs], dim=1)
        fused = bank.gather(1, best_index[:, None]).clamp(EPS, 1.0 - EPS)
        return {
            "fused_prob": fused,
            "selected_index": best_index,
            "utility": utility,
            "best_utility": all_scores.max(dim=1, keepdim=True).values,
        }

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
                gray.expand(-1, n, -1, -1),
                edge.expand(-1, n, -1, -1),
            ],
            dim=2,
        ).reshape(b * n, 10, low_h, low_w)
        candidate_feature = self.candidate_encoder(candidate_input)
        shared_bank = shared[:, None].expand(-1, n, -1, -1, -1).reshape(
            b * n, self.hidden_dim, low_h, low_w
        )
        metadata_bank = metadata.reshape(b * n, self.hidden_dim, 1, 1).expand(
            -1, -1, low_h, low_w
        )
        fused_feature = self.candidate_fuse(
            torch.cat([shared_bank, candidate_feature, metadata_bank], dim=1)
        )
        raw_utility_low = self.utility_head(fused_feature).reshape(b, n, low_h, low_w)
        benefit_logit_low = self.benefit_head(fused_feature).reshape(b, n, low_h, low_w)
        harm_logit_low = self.harm_head(fused_feature).reshape(b, n, low_h, low_w)

        # Edge-aware local consensus.  In homogeneous regions candidate utility
        # should be spatially coherent; strong image edges reduce smoothing so
        # legitimate boundary transitions remain possible.
        pad = self.coherence_kernel // 2
        score_flat = raw_utility_low.reshape(b * n, 1, low_h, low_w)
        local_score = F.avg_pool2d(
            score_flat,
            kernel_size=self.coherence_kernel,
            stride=1,
            padding=pad,
        ).reshape(b, n, low_h, low_w)
        edge_weight = (1.0 - edge).clamp(0.0, 1.0)
        utility_low = raw_utility_low + self.coherence_alpha * edge_weight * (
            local_score - raw_utility_low
        )

        utility = _resize_4d(utility_low.reshape(b * n, 1, low_h, low_w), (h, w)).reshape(
            b, n, h, w
        )
        benefit_logit = _resize_4d(
            benefit_logit_low.reshape(b * n, 1, low_h, low_w), (h, w)
        ).reshape(b, n, h, w)
        harm_logit = _resize_4d(
            harm_logit_low.reshape(b * n, 1, low_h, low_w), (h, w)
        ).reshape(b, n, h, w)
        edge_full = _resize_4d(edge, (h, w))

        current_bank = current.expand(-1, n, -1, -1)
        abs_edit_full = (nonbase - current_bank).abs()
        hard_changed = (nonbase >= 0.5) != (current_bank >= 0.5)
        valid = (
            candidate_deploy[None, :, None, None]
            & (support > self.support_floor)
            & (abs_edit_full > self.edit_epsilon)
            & (lock_map < 0.5)
        )
        if self.hard_change_only:
            # A deployed A2 action is meaningful only when the actual candidate
            # value flips the binary segmentation label.  This makes accepted
            # action pixels identical to changed output pixels and removes the
            # previous soft-only pseudo-edits.
            valid = valid & hard_changed

        benefit_prob = torch.sigmoid(benefit_logit)
        harm_prob = torch.sigmoid(harm_logit)
        # One coupled action score avoids the old failure where a candidate won
        # the utility ranking while a separately trained risk head was ignored.
        safe_utility = (
            benefit_logit
            + 0.25 * utility
            - self.risk_penalty * F.softplus(harm_logit)
        )
        floor = -1.0e4 if safe_utility.dtype in (torch.float16, torch.bfloat16) else -1.0e9
        candidate_score = safe_utility.masked_fill(~valid, floor)
        preserve_score = torch.zeros_like(current)
        action_score = torch.cat([preserve_score, candidate_score], dim=1)
        action_logits = action_score / self.student_temperature
        soft_route = F.softmax(action_logits, dim=1)

        top_values, top_indices = torch.topk(action_score, k=2, dim=1)
        best_score = top_values[:, :1]
        best_index = top_indices[:, 0]

        # Preserve/Base is a real zero-score competitor.  When Preserve wins
        # because no candidate is valid, the runner-up is the masked sentinel
        # (-1e9 in fp32).  Subtracting that sentinel created a meaningless
        # ~1e9 confidence gap in diagnostics.  The edit margin is defined only
        # for an actual candidate winner; Preserve decisions therefore have
        # exactly zero top-gap.  When a candidate wins, top-2 already compares
        # it against the strongest alternative, including Base=0.
        candidate_wins = best_index[:, None] > 0
        raw_top_gap = top_values[:, :1] - top_values[:, 1:2]
        top_gap = torch.where(
            candidate_wins,
            torch.nan_to_num(raw_top_gap, nan=0.0, posinf=0.0, neginf=0.0),
            torch.zeros_like(raw_top_gap),
        )

        candidate_positive = (benefit_prob * (1.0 - harm_prob)).clamp(0.0, 1.0)
        valid_float = valid.to(candidate_positive.dtype)
        coherence_num = F.avg_pool2d(
            (candidate_positive * valid_float).reshape(b * n, 1, h, w),
            kernel_size=self.coherence_kernel,
            stride=1,
            padding=pad,
        ).reshape(b, n, h, w)
        coherence_den = F.avg_pool2d(
            valid_float.reshape(b * n, 1, h, w),
            kernel_size=self.coherence_kernel,
            stride=1,
            padding=pad,
        ).reshape(b, n, h, w).clamp_min(EPS)
        candidate_coherence = coherence_num / coherence_den

        selected_candidate_index = (best_index - 1).clamp_min(0)
        gathered_risk = harm_prob.gather(1, selected_candidate_index[:, None])
        gathered_benefit = benefit_prob.gather(1, selected_candidate_index[:, None])
        gathered_coherence = candidate_coherence.gather(
            1, selected_candidate_index[:, None]
        )
        selected_valid = valid.gather(1, selected_candidate_index[:, None])

        # A Preserve decision has no selected candidate.  Return zero-valued
        # candidate diagnostics instead of accidentally reporting candidate 0's
        # risk/coherence after the clamped gather index.  This does not relax the
        # safety rule because safe_edit still explicitly requires candidate_wins.
        selected_risk = torch.where(
            candidate_wins, gathered_risk, torch.zeros_like(gathered_risk)
        )
        selected_benefit = torch.where(
            candidate_wins, gathered_benefit, torch.zeros_like(gathered_benefit)
        )
        selected_coherence = torch.where(
            candidate_wins, gathered_coherence, torch.zeros_like(gathered_coherence)
        )

        safe_edit = (
            candidate_wins
            & selected_valid
            & (best_score > self.sea_level_margin)
            & (top_gap > self.top_gap_margin)
            & (selected_benefit >= self.benefit_threshold)
            & (selected_risk <= self.risk_threshold)
            & (selected_coherence >= self.coherence_threshold)
            & (lock_map < 0.5)
        )
        if self.current_epoch < self.deploy_start_epoch:
            # Curriculum is part of deployment state, not only a training trick.
            # Checkpoint evaluation restores the saved epoch, so a pre-deploy or
            # explicit Preserve fallback checkpoint cannot unexpectedly activate M2.
            safe_edit = torch.zeros_like(safe_edit)

        hard_index = torch.where(
            safe_edit[:, 0],
            best_index,
            torch.zeros_like(best_index),
        )
        hard_route = torch.zeros_like(soft_route).scatter_(1, hard_index[:, None], 1.0)
        if self.training:
            route = _StraightThroughHardRoute.apply(hard_route, soft_route)
        else:
            route = hard_route if self.hard_inference else soft_route

        candidate_bank = torch.cat([current, nonbase], dim=1)
        next_current = (route * candidate_bank).sum(dim=1, keepdim=True).clamp(
            EPS, 1.0 - EPS
        )
        edit_prob = soft_route[:, 1:].sum(dim=1, keepdim=True)
        return {
            "current_before": current,
            "next_current": next_current,
            "candidate_score": candidate_score,
            "candidate_utility": utility,
            "candidate_benefit_logit": benefit_logit,
            "candidate_benefit_prob": benefit_prob,
            "candidate_harm_logit": harm_logit,
            "candidate_valid": valid,
            "action_score": action_score,
            "soft_route": soft_route,
            "hard_route": hard_route,
            "selected_index": hard_index,
            "hard_edit": safe_edit.to(current.dtype),
            "edit_prob": edit_prob,
            "best_score": best_score,
            "top_gap": top_gap,
            "selected_risk": selected_risk,
            "selected_benefit": selected_benefit,
            "selected_coherence": selected_coherence,
            "candidate_coherence": candidate_coherence,
            "edge_map": edge_full,
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
        b, k, _, _ = candidate_probs.shape
        if k < 2 or k > self.max_candidates:
            raise ValueError(f"V523 requires 2..{self.max_candidates} candidates, got {k}")
        if supports.shape != candidate_probs.shape:
            raise ValueError("supports must match candidate_probs")
        if candidate_causes.shape != candidate_probs.shape:
            raise ValueError("candidate_causes must match candidate_probs")
        for tensor in (family_ids, action_ids, dose_values, radius_values, deploy_mask):
            if tensor.ndim != 1 or tensor.numel() != k:
                raise ValueError("V523 candidate metadata must have K entries")

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
            lock_map = torch.maximum(lock_map, step["hard_edit"].detach())

        def stack(name: str) -> torch.Tensor:
            return torch.stack([step[name] for step in steps], dim=1)

        final_prob = current.clamp(EPS, 1.0 - EPS)
        hard_edit_union = stack("hard_edit").amax(dim=1)
        soft_edit_union = stack("edit_prob").amax(dim=1)
        return {
            "m2_fused_probs": final_prob[:, 0],
            "m2_training_probs": final_prob[:, 0],
            "m2_proposal_probs": final_prob[:, 0],
            "m2_convex_probs": final_prob[:, 0],
            "m2_edit_gate_prob": soft_edit_union,
            "m2_residual_map": final_prob - c0,
            "v523_step_current_before": stack("current_before"),
            "v523_step_candidate_score": stack("candidate_score"),
            "v523_step_candidate_utility": stack("candidate_utility"),
            "v523_step_candidate_benefit_logit": stack("candidate_benefit_logit"),
            "v523_step_candidate_benefit_prob": stack("candidate_benefit_prob"),
            "v523_step_candidate_harm_logit": stack("candidate_harm_logit"),
            "v523_step_candidate_valid": stack("candidate_valid"),
            "v523_step_action_score": stack("action_score"),
            "v523_step_soft_route": stack("soft_route"),
            "v523_step_hard_route": stack("hard_route"),
            "v523_step_selected_index": stack("selected_index"),
            "v523_step_hard_edit_mask": stack("hard_edit"),
            "v523_step_edit_prob": stack("edit_prob"),
            "v523_step_best_score": stack("best_score"),
            "v523_step_top_gap": stack("top_gap"),
            "v523_step_selected_risk": stack("selected_risk"),
            "v523_step_selected_benefit": stack("selected_benefit"),
            "v523_step_selected_coherence": stack("selected_coherence"),
            "v523_step_candidate_coherence": stack("candidate_coherence"),
            "v523_step_edge_map": stack("edge_map"),
            "v523_lock_map": lock_map,
            "v523_deploy_mask": deploy_mask,
            "v523_candidate_support": support,
            "v523_candidate_causes": cause,
            "v523_num_steps": c0.new_full((b,), float(len(steps))),
            "v523_deploy_phase": c0.new_full(
                (b,),
                float(
                    0
                    if self.current_epoch < self.deploy_start_epoch
                    else 1
                    if self.current_epoch < self.full_deploy_epoch
                    else 2
                ),
            ),
            "v523_changed_pixel_rate": (
                (final_prob >= 0.5) != (c0 >= 0.5)
            ).float().flatten(1).mean(dim=1),
            "v523_selected_step_rate": stack("hard_edit").flatten(2).any(dim=2).float().mean(dim=1),
        }
