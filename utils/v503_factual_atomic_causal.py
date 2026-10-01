"""V503 factual-error / atomic-intervention causal components.

This module repairs the causal-chain break in V502:

* error causes are independent pixel maps rather than one image-level softmax;
* boundary errors have direction (trim vs expand);
* M1 emits typed atomic interventions, not full-mask experts;
* the factual action teacher is a deterministic function of detached C0 and GT;
* candidate predictions never define their own supervision geometry.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1.0e-6


def _as_b1hw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        return x[:, :1]
    if x.ndim == 3:
        return x[:, None]
    raise ValueError(f"Expected [B,H,W] or [B,1,H,W], got {tuple(x.shape)}")


def _entropy(prob: torch.Tensor) -> torch.Tensor:
    p = prob.clamp(EPS, 1.0 - EPS)
    return (-(p * p.log() + (1.0 - p) * (1.0 - p).log()) / math.log(2.0)).clamp(0.0, 1.0)


def _soft_boundary(prob: torch.Tensor, radius: int = 1) -> torch.Tensor:
    radius = max(int(radius), 1)
    kernel = 2 * radius + 1
    dilated = F.max_pool2d(prob, kernel, stride=1, padding=radius)
    eroded = -F.max_pool2d(-prob, kernel, stride=1, padding=radius)
    return (dilated - eroded).clamp(0.0, 1.0)


def _gray_edge(image: torch.Tensor, size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    gray = image.mean(dim=1, keepdim=True)
    if gray.shape[-2:] != size:
        gray = F.interpolate(gray, size=size, mode="bilinear", align_corners=False)
    lo = gray.amin(dim=(-2, -1), keepdim=True)
    hi = gray.amax(dim=(-2, -1), keepdim=True)
    gray = (gray - lo) / (hi - lo).clamp_min(EPS)
    gx = F.pad((gray[..., :, 1:] - gray[..., :, :-1]).abs(), (0, 1, 0, 0))
    gy = F.pad((gray[..., 1:, :] - gray[..., :-1, :]).abs(), (0, 0, 0, 1))
    return gray, (gx + gy).clamp(0.0, 1.0)


@torch.no_grad()
def build_factual_cause_targets(
    c0_prob: torch.Tensor,
    gt: torch.Tensor,
    boundary_radius: int = 2,
    failure_dice: float = 0.55,
    include_global_action: bool = False,
) -> Dict[str, torch.Tensor]:
    """Build candidate-independent factual error and action targets.

    Action ids are fixed and auditable:
      0 Preserve, 1 Delete, 2 Fill, 3 Trim, 4 Expand, 5 GlobalRepair.

    Every factual binary error pixel receives exactly one action. Base-correct
    pixels are always Preserve. Severe failure optionally redirects factual
    error pixels to GlobalRepair without using any candidate prediction.
    """
    c0 = (_as_b1hw(c0_prob) >= 0.5).float()
    y = (_as_b1hw(gt) >= 0.5).float()

    fp = c0 * (1.0 - y)
    fn = (1.0 - c0) * y

    radius = max(int(boundary_radius), 1)
    gt_band = (_soft_boundary(y, radius=radius) > 0.0).float()
    c0_band = (_soft_boundary(c0, radius=radius) > 0.0).float()

    # Directional boundary causes.  Trim removes over-segmentation adjacent to
    # the GT surface; Expand adds under-segmentation adjacent to the C0 surface.
    trim = fp * gt_band
    expand = fn * c0_band
    delete = (fp - trim).clamp(0.0, 1.0)
    fill = (fn - expand).clamp(0.0, 1.0)
    causes = torch.cat([delete, fill, trim, expand], dim=1)

    inter = (c0 * y).flatten(1).sum(dim=1)
    den = c0.flatten(1).sum(dim=1) + y.flatten(1).sum(dim=1)
    dice = (2.0 * inter + EPS) / (den + EPS)
    c0_empty = c0.flatten(1).sum(dim=1) < 1.0
    y_nonempty = y.flatten(1).sum(dim=1) >= 1.0
    failure = ((dice < float(failure_dice)) | (c0_empty & y_nonempty)).float()[:, None]

    action = torch.zeros(c0.shape[0], c0.shape[-2], c0.shape[-1], dtype=torch.long, device=c0.device)
    for index in range(4):
        action[causes[:, index] > 0.5] = index + 1
    factual_error = (c0 != y).float()
    if include_global_action:
        global_region = factual_error * failure[:, :, None, None]
        action[global_region[:, 0] > 0.5] = 5
    preserve = (action == 0).float()[:, None]

    return {
        "cause_targets": causes.detach(),
        "failure_target": failure.detach(),
        "action_target": action.detach(),
        "preserve_target": preserve.detach(),
        "factual_error": factual_error.detach(),
        "base_correct": (1.0 - factual_error).detach(),
        "base_hard": c0.detach(),
        "gt_hard": y.detach(),
        "base_dice": dice.detach(),
    }


class PixelCausalErrorStateHead(nn.Module):
    """Predict hierarchical FP/FN error maps and typed atomic causes.

    V517 predicted four extremely sparse maps independently from low-level
    Base geometry.  The ceiling decomposition showed that the factual target
    covered all Base errors while the learned support was anti-correlated with
    the target.  V518 therefore shares evidence between the two dense error
    polarities (FP/FN), conditions Delete/Trim and Fill/Expand on that shared
    evidence, and optionally fuses frozen ViT/PVL semantic tokens.

    The public outputs remain exactly compatible with V503/V505:
      error_cause_map_logits/probs are [B,4,H,W] in
      Delete, Fill, Trim, Expand order.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        initial_rate: float = 0.02,
        *,
        semantic_channels: int = 512,
        semantic_dim: int = 32,
        use_semantic: bool = False,
        hierarchical: bool = False,
    ) -> None:
        super().__init__()
        groups = min(8, int(hidden_dim))
        while hidden_dim % groups != 0 and groups > 1:
            groups -= 1
        self.use_semantic = bool(use_semantic)
        self.hierarchical = bool(hierarchical)
        self.semantic_dim = int(semantic_dim)

        # Keep the historical six-channel encoder unchanged for old configs and
        # checkpoints.  Semantic evidence is fused after this encoder.
        self.encoder = nn.Sequential(
            nn.Conv2d(6, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
        )
        if self.use_semantic:
            semantic_groups = min(8, self.semantic_dim)
            while self.semantic_dim % semantic_groups != 0 and semantic_groups > 1:
                semantic_groups -= 1
            self.semantic_proj = nn.Sequential(
                nn.Conv2d(int(semantic_channels), self.semantic_dim, 1, bias=False),
                nn.GroupNorm(semantic_groups, self.semantic_dim),
                nn.GELU(),
            )
            self.semantic_fuse = nn.Sequential(
                nn.Conv2d(hidden_dim + self.semantic_dim, hidden_dim, 3, padding=1, bias=False),
                nn.GroupNorm(groups, hidden_dim),
                nn.GELU(),
            )
        else:
            self.semantic_proj = None
            self.semantic_fuse = None

        rate = min(max(float(initial_rate), 1.0e-4), 1.0 - 1.0e-4)
        bias = math.log(rate / (1.0 - rate))
        if self.hierarchical:
            # Independent FP/FN probabilities allow both error polarities to be
            # present in one image.  Subtype softmaxes partition each polarity.
            self.polarity_head = nn.Conv2d(hidden_dim, 2, 1)
            self.fp_subtype_head = nn.Conv2d(hidden_dim, 2, 1)
            self.fn_subtype_head = nn.Conv2d(hidden_dim, 2, 1)
            nn.init.zeros_(self.polarity_head.weight)
            nn.init.constant_(self.polarity_head.bias, bias)
            nn.init.zeros_(self.fp_subtype_head.weight)
            nn.init.zeros_(self.fp_subtype_head.bias)
            nn.init.zeros_(self.fn_subtype_head.weight)
            nn.init.zeros_(self.fn_subtype_head.bias)
            self.cause_head = None
        else:
            self.cause_head = nn.Conv2d(hidden_dim, 4, 1)
            nn.init.zeros_(self.cause_head.weight)
            nn.init.constant_(self.cause_head.bias, bias)
            self.polarity_head = None
            self.fp_subtype_head = None
            self.fn_subtype_head = None

        self.failure_head = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.failure_head[-1].weight)
        nn.init.constant_(self.failure_head[-1].bias, bias)

    def _fuse_semantic(
        self,
        feature: torch.Tensor,
        semantic_map: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.use_semantic:
            return feature
        if not isinstance(semantic_map, torch.Tensor):
            semantic = feature.new_zeros(
                feature.shape[0], self.semantic_dim, *feature.shape[-2:]
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

    def forward(
        self,
        image: torch.Tensor,
        c0_prob: torch.Tensor,
        semantic_map: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        c0 = _as_b1hw(c0_prob).clamp(EPS, 1.0 - EPS)
        size = c0.shape[-2:]
        ent = _entropy(c0)
        bnd = _soft_boundary(c0)
        gray, edge = _gray_edge(image, size)
        feature = self.encoder(
            torch.cat([c0, 1.0 - c0, ent, bnd, gray, edge], dim=1)
        )
        feature = self._fuse_semantic(feature, semantic_map)

        extra: Dict[str, torch.Tensor] = {}
        if self.hierarchical:
            polarity_logits = self.polarity_head(feature)
            polarity_probs = torch.sigmoid(polarity_logits)
            fp_subtype_logits = self.fp_subtype_head(feature)
            fn_subtype_logits = self.fn_subtype_head(feature)
            fp_subtype_probs = F.softmax(fp_subtype_logits, dim=1)
            fn_subtype_probs = F.softmax(fn_subtype_logits, dim=1)

            fp_prob = polarity_probs[:, 0:1]
            fn_prob = polarity_probs[:, 1:2]
            delete_prob = fp_prob * fp_subtype_probs[:, 0:1]
            trim_prob = fp_prob * fp_subtype_probs[:, 1:2]
            fill_prob = fn_prob * fn_subtype_probs[:, 0:1]
            expand_prob = fn_prob * fn_subtype_probs[:, 1:2]
            cause_probs = torch.cat(
                [delete_prob, fill_prob, trim_prob, expand_prob], dim=1
            ).clamp(EPS, 1.0 - EPS)
            cause_logits = torch.logit(cause_probs)
            extra = {
                "v518_error_polarity_logits": polarity_logits,
                "v518_error_polarity_probs": polarity_probs,
                "v518_fp_subtype_logits": fp_subtype_logits,
                "v518_fn_subtype_logits": fn_subtype_logits,
                "v518_fp_subtype_probs": fp_subtype_probs,
                "v518_fn_subtype_probs": fn_subtype_probs,
            }
        else:
            cause_logits = self.cause_head(feature)
            cause_probs = torch.sigmoid(cause_logits)

        fg_area = c0.flatten(1).mean(dim=1, keepdim=True)
        unc_mass = ent.flatten(1).mean(dim=1, keepdim=True)
        bnd_mass = bnd.flatten(1).mean(dim=1, keepdim=True)
        empty_flag = (c0.flatten(1).max(dim=1, keepdim=True).values < 0.5).float()
        pooled = F.adaptive_avg_pool2d(feature, 1).flatten(1)
        failure_logit = self.failure_head(
            torch.cat([pooled, fg_area, unc_mass, bnd_mass, empty_flag], dim=1)
        )
        failure_prob = torch.sigmoid(failure_logit)

        pooled_causes = cause_probs.flatten(2).mean(dim=2)
        boundary_prob = torch.maximum(pooled_causes[:, 2], pooled_causes[:, 3])
        any_error = torch.maximum(pooled_causes.amax(dim=1), failure_prob[:, 0])
        no_edit = (1.0 - any_error).clamp(EPS, 1.0 - EPS)
        state_probs = torch.stack(
            [
                no_edit,
                pooled_causes[:, 0],
                pooled_causes[:, 1],
                boundary_prob,
                failure_prob[:, 0],
            ],
            dim=1,
        ).clamp(EPS, 1.0 - EPS)
        state_logits = torch.logit(state_probs)
        return {
            "error_state_logits": state_logits,
            "error_state_probs": state_probs,
            "error_cause_map_logits": cause_logits,
            "error_cause_map_probs": cause_probs,
            "failure_state_logit": failure_logit,
            "failure_state_prob": failure_prob,
            "p_no_edit": state_probs[:, 0],
            "p_fp": state_probs[:, 1],
            "p_fn": state_probs[:, 2],
            "p_boundary": state_probs[:, 3],
            "p_failure": state_probs[:, 4],
            **extra,
        }


class AtomicLocalRepairGenerator(nn.Module):
    """Generate typed local interventions with semantic support and dose bank.

    The first four output slots are always the historical deployment contract:
    Delete, Fill, Trim, Expand at dose 1.  Additional doses are appended, so
    V505 M2/M3 continue to read ``candidate_probs[:, 1:5]`` unchanged while A1
    Oracle/PWO can evaluate the complete candidate bank.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        max_atom_delta: float = 3.0,
        delta_bias: float = -1.5,
        support_bias: float = -2.0,
        cause_gated: bool = False,
        min_action_strength: float = 0.0,
        *,
        semantic_channels: int = 512,
        semantic_dim: int = 32,
        use_semantic: bool = False,
        dose_values: Tuple[float, ...] = (1.0,),
        candidate_to_cause_grad_scale: float = 0.0,
    ) -> None:
        super().__init__()
        groups = min(8, int(hidden_dim))
        while hidden_dim % groups != 0 and groups > 1:
            groups -= 1
        self.max_atom_delta = float(max_atom_delta)
        self.delta_bias = float(delta_bias)
        self.cause_gated = bool(cause_gated)
        self.min_action_strength = min(max(float(min_action_strength), 0.0), 1.0)
        self.use_semantic = bool(use_semantic)
        self.semantic_dim = int(semantic_dim)
        self.candidate_to_cause_grad_scale = min(
            max(float(candidate_to_cause_grad_scale), 0.0), 1.0
        )
        doses = []
        for value in dose_values:
            value = float(value)
            if value <= 0:
                continue
            if all(abs(value - old) > 1.0e-8 for old in doses):
                doses.append(value)
        if not doses:
            doses = [1.0]
        if all(abs(value - 1.0) > 1.0e-8 for value in doses):
            doses.insert(0, 1.0)
        # Primary dose must be first for the fixed V505 typed-action contract.
        doses = [1.0] + [value for value in doses if abs(value - 1.0) > 1.0e-8]
        self.dose_values = tuple(doses)

        self.trunk = nn.Sequential(
            nn.Conv2d(10, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
        )
        if self.use_semantic:
            semantic_groups = min(8, self.semantic_dim)
            while self.semantic_dim % semantic_groups != 0 and semantic_groups > 1:
                semantic_groups -= 1
            self.semantic_proj = nn.Sequential(
                nn.Conv2d(int(semantic_channels), self.semantic_dim, 1, bias=False),
                nn.GroupNorm(semantic_groups, self.semantic_dim),
                nn.GELU(),
            )
            self.semantic_fuse = nn.Sequential(
                nn.Conv2d(hidden_dim + self.semantic_dim, hidden_dim, 3, padding=1, bias=False),
                nn.GroupNorm(groups, hidden_dim),
                nn.GELU(),
            )
        else:
            self.semantic_proj = None
            self.semantic_fuse = None

        self.support_heads = nn.ModuleList(
            [nn.Conv2d(hidden_dim, 1, 1) for _ in range(4)]
        )
        self.delta_heads = nn.ModuleList(
            [nn.Conv2d(hidden_dim, 1, 1) for _ in range(4)]
        )
        self.register_buffer(
            "typed_signs",
            torch.tensor([-1.0, 1.0, -1.0, 1.0]),
            persistent=False,
        )
        for head in self.support_heads:
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, float(support_bias))
        for head in self.delta_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @staticmethod
    def _conjunctive_support(
        raw_support: torch.Tensor,
        cause_gate: torch.Tensor,
    ) -> torch.Tensor:
        product = (
            raw_support.clamp(0.0, 1.0) * cause_gate.clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)
        return (torch.sqrt(product + EPS) - math.sqrt(EPS)).clamp(0.0, 1.0)

    def _cause_with_scaled_gradient(self, cause: torch.Tensor) -> torch.Tensor:
        scale = self.candidate_to_cause_grad_scale
        if scale <= 0.0:
            return cause.detach()
        if scale >= 1.0:
            return cause
        return cause.detach() + scale * (cause - cause.detach())

    def _fuse_semantic(
        self,
        feature: torch.Tensor,
        semantic_map: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.use_semantic:
            return feature
        if not isinstance(semantic_map, torch.Tensor):
            semantic = feature.new_zeros(
                feature.shape[0], self.semantic_dim, *feature.shape[-2:]
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
    def _candidate_from_alpha(
        c0: torch.Tensor,
        alpha: torch.Tensor,
        negative_action: bool,
    ) -> torch.Tensor:
        if negative_action:
            return c0 * (1.0 - alpha)
        return c0 + (1.0 - c0) * alpha

    def forward(
        self,
        image: torch.Tensor,
        c0_prob: torch.Tensor,
        cause_map_probs: torch.Tensor,
        semantic_map: torch.Tensor | None = None,
        teacher_cause_targets: torch.Tensor | None = None,
        teacher_weight: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        c0 = _as_b1hw(c0_prob).clamp(EPS, 1.0 - EPS)
        if cause_map_probs.ndim != 4 or cause_map_probs.shape[1] != 4:
            raise ValueError("cause_map_probs must be [B,4,H,W]")
        if cause_map_probs.shape[-2:] != c0.shape[-2:]:
            cause_map_probs = F.interpolate(
                cause_map_probs,
                size=c0.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        predicted_cause = self._cause_with_scaled_gradient(
            cause_map_probs.clamp(0.0, 1.0)
        )
        teacher_weight = min(max(float(teacher_weight), 0.0), 1.0)
        teacher = None
        if isinstance(teacher_cause_targets, torch.Tensor):
            teacher = teacher_cause_targets.to(
                device=c0.device, dtype=c0.dtype
            )
            if teacher.shape[-2:] != c0.shape[-2:]:
                teacher = F.interpolate(
                    teacher,
                    size=c0.shape[-2:],
                    mode="nearest",
                )
            teacher = teacher.clamp(0.0, 1.0)
        if teacher is not None and teacher_weight > 0.0:
            cause_evidence = (
                teacher_weight * teacher
                + (1.0 - teacher_weight) * predicted_cause
            )
        else:
            cause_evidence = predicted_cause

        ent = _entropy(c0)
        bnd = _soft_boundary(c0)
        gray, edge = _gray_edge(image, c0.shape[-2:])
        feature = self.trunk(
            torch.cat(
                [c0, 1.0 - c0, ent, bnd, gray, edge, cause_evidence], dim=1
            )
        )
        feature = self._fuse_semantic(feature, semantic_map)

        c0_logits = torch.logit(c0)
        primary_logits = []
        primary_supports = []
        primary_raw_supports = []
        primary_deltas = []
        primary_magnitudes = []
        primary_alphas = []

        for index in range(4):
            raw_support = torch.sigmoid(self.support_heads[index](feature))
            cause_gate = cause_evidence[:, index:index + 1]
            predicted_support = (
                self._conjunctive_support(raw_support, cause_gate)
                if self.cause_gated
                else raw_support
            )
            if teacher is not None and teacher_weight > 0.0:
                action_support = (
                    teacher_weight * teacher[:, index:index + 1]
                    + (1.0 - teacher_weight) * predicted_support
                )
            else:
                action_support = predicted_support
            magnitude = torch.sigmoid(
                self.delta_heads[index](feature) + self.delta_bias
            )
            conditional_strength = (
                self.min_action_strength
                + (1.0 - self.min_action_strength) * magnitude
            )
            alpha = (action_support * conditional_strength).clamp(0.0, 1.0)
            negative_action = index in (0, 2)
            candidate_prob = self._candidate_from_alpha(
                c0, alpha, negative_action
            ).clamp(EPS, 1.0 - EPS)
            candidate_logit = torch.logit(candidate_prob)

            primary_logits.append(candidate_logit)
            primary_supports.append(action_support)
            primary_raw_supports.append(raw_support)
            primary_deltas.append(candidate_logit - c0_logits)
            primary_magnitudes.append(magnitude)
            primary_alphas.append(alpha)

        # Keep the four primary slots first.  Append stronger/weaker dose copies.
        logits = list(primary_logits)
        supports = list(primary_supports)
        raw_supports = list(primary_raw_supports)
        deltas = list(primary_deltas)
        magnitudes = list(primary_magnitudes)
        alphas = list(primary_alphas)
        action_ids = [0, 1, 2, 3]
        dose_ids = [1.0, 1.0, 1.0, 1.0]

        for dose in self.dose_values[1:]:
            for index in range(4):
                alpha = (primary_alphas[index] * float(dose)).clamp(0.0, 1.0)
                candidate_prob = self._candidate_from_alpha(
                    c0, alpha, index in (0, 2)
                ).clamp(EPS, 1.0 - EPS)
                candidate_logit = torch.logit(candidate_prob)
                logits.append(candidate_logit)
                supports.append(primary_supports[index])
                raw_supports.append(primary_raw_supports[index])
                deltas.append(candidate_logit - c0_logits)
                magnitudes.append(primary_magnitudes[index])
                alphas.append(alpha)
                action_ids.append(index)
                dose_ids.append(float(dose))

        spatial_gate_mean = predicted_cause.detach().flatten(2).mean(dim=2)
        action_id_tensor = torch.tensor(
            action_ids, device=c0.device, dtype=torch.long
        )
        dose_tensor = c0.new_tensor(dose_ids)
        return {
            "local_candidate_logits": torch.cat(logits, dim=1),
            "local_primary_candidate_logits": torch.cat(primary_logits, dim=1),
            "local_supports": torch.cat(supports, dim=1),
            "local_primary_supports": torch.cat(primary_supports, dim=1),
            "local_raw_supports": torch.cat(raw_supports, dim=1),
            "local_primary_raw_supports": torch.cat(primary_raw_supports, dim=1),
            "local_deltas": torch.cat(deltas, dim=1),
            "local_raw_magnitudes": torch.cat(magnitudes, dim=1),
            "local_action_alpha": torch.cat(alphas, dim=1),
            "local_primary_action_alpha": torch.cat(primary_alphas, dim=1),
            "local_candidate_action_ids": action_id_tensor,
            "local_candidate_dose_values": dose_tensor,
            "local_soft_gates": spatial_gate_mean,
            "local_active_mask": spatial_gate_mean > 0.01,
            "local_cause_map_probs": predicted_cause.detach(),
            "v518_teacher_weight": c0.new_full(
                (c0.shape[0],), teacher_weight
            ),
        }
