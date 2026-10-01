"""/home/tsz-25/MedCLIPSeg-pristine/utils/v484_error_state_causal.py
V485 error-state gated causal intervention pipeline.

This file replaces the earlier V484 prototype with a deterministic, auditable
candidate source.  The invariants are:

1. C0/Preserve is read-only for proposal losses.  The pipeline receives the
   online Base logits, immediately detaches them, and uses the detached C0 as
   the factual anchor for all candidates.
2. The pipeline returns a standard dictionary, never a tuple.  The same keys are
   consumed by train.py, validation, test.py, and diagnosis tools.
3. Local candidates are bounded residual interventions instead of full-mask
   generators:
       Ck_logit = C0_logit + gate_k * support_k * signed_magnitude_k
4. The initial edit is small but non-zero so the generator is not trapped in a
   no-op state.  Safety is enforced by support supervision, edit-budget, and
   no-harm losses, not by zeroing the residual branch.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.v503_factual_atomic_causal import (
    AtomicLocalRepairGenerator,
    PixelCausalErrorStateHead,
    build_factual_cause_targets,
)
from utils.v531_typed_sparse_refiner import V531TypedSparseRefiner
from utils.v532_unified_sparse_refiner import V532UnifiedSparseRefiner
from utils.v547_paired_anchor import build_v547_paired_anchor
from utils.v505_interactive_region_causal import (
    V505InteractiveRegionPolicy,
    V505RegionVerifier,
    V519FamilyAwareRegionComposer,
    V520GateSelectorRegionComposer,
)
from utils.v521_candidate_conditional_region import (
    V521CandidateConditionalRegionUtilityComposer,
)
from utils.v522_pwo_distilled_sequential_local import (
    V522PWODistilledSequentialLocalComposer,
)
from utils.v523_sea_level_utility_composer import (
    V523SeaLevelUtilityComposer,
)
from utils.v524_counterfactual_prompted_region import (
    V524CounterfactualPromptedRegionComposer,
)

EPS = 1.0e-6


def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def _m1(cfg: Any, key: str, default: Any = None) -> Any:
    return _cfg_get(_cfg_get(cfg, "M1", None), key, default)


def _as_b1hw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        if x.shape[1] == 1:
            return x
        return x[:, :1]
    if x.ndim == 3:
        return x[:, None]
    raise ValueError(f"Expected [B,H,W] or [B,1,H,W], got {tuple(x.shape)}")


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


def _entropy(prob: torch.Tensor) -> torch.Tensor:
    p = prob.clamp(EPS, 1.0 - EPS)
    return (-(p * p.log() + (1.0 - p) * (1.0 - p).log()) / math.log(2.0)).clamp(0.0, 1.0)


def _soft_boundary(prob: torch.Tensor, radius: int = 1) -> torch.Tensor:
    radius = int(radius)
    if radius <= 0:
        return torch.zeros_like(prob)
    k = 2 * radius + 1
    dilated = F.max_pool2d(prob, k, 1, radius)
    eroded = -F.max_pool2d(-prob, k, 1, radius)
    return (dilated - eroded).clamp(0.0, 1.0)



def _v518_float_tuple(value: Any, default: Tuple[float, ...]) -> Tuple[float, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, (list, tuple)):
        raw = value
    else:
        raw = str(value).replace("[", "").replace("]", "").split(",")
    out = []
    for item in raw:
        try:
            number = float(item)
        except (TypeError, ValueError):
            continue
        if number > 0 and all(abs(number - old) > 1.0e-8 for old in out):
            out.append(number)
    return tuple(out) if out else tuple(default)


def _v518_int_tuple(value: Any, default: Tuple[int, ...]) -> Tuple[int, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, (list, tuple)):
        raw = value
    else:
        raw = str(value).replace("[", "").replace("]", "").split(",")
    out = []
    for item in raw:
        try:
            number = int(item)
        except (TypeError, ValueError):
            continue
        if number > 0 and number not in out:
            out.append(number)
    return tuple(out) if out else tuple(default)


def _v518_hard_dilate(prob: torch.Tensor, radius: int) -> torch.Tensor:
    radius = max(int(radius), 1)
    hard = (prob >= 0.5).to(prob.dtype)
    kernel = 2 * radius + 1
    return F.max_pool2d(hard, kernel, stride=1, padding=radius)


def _v518_hard_erode(prob: torch.Tensor, radius: int) -> torch.Tensor:
    radius = max(int(radius), 1)
    hard = (prob >= 0.5).to(prob.dtype)
    kernel = 2 * radius + 1
    return -F.max_pool2d(-hard, kernel, stride=1, padding=radius)


def _v518_morphology_bank(
    c0_prob: torch.Tensor,
    radii: Tuple[int, ...],
    *,
    include_erode: bool,
    include_dilate: bool,
    include_open: bool,
    include_close: bool,
) -> Tuple[torch.Tensor, Tuple[str, ...]]:
    """Deterministic multiscale candidates validated by the V517 ceiling audit."""
    outputs = []
    names = []
    factual = c0_prob.detach()
    for radius in radii:
        eroded = _v518_hard_erode(factual, radius)
        dilated = _v518_hard_dilate(factual, radius)
        opened = _v518_hard_dilate(eroded, radius)
        closed = _v518_hard_erode(dilated, radius)
        for enabled, name, value in (
            (include_erode, f"erode_r{radius}", eroded),
            (include_dilate, f"dilate_r{radius}", dilated),
            (include_open, f"open_r{radius}", opened),
            (include_close, f"close_r{radius}", closed),
        ):
            if enabled:
                outputs.append(value.clamp(EPS, 1.0 - EPS))
                names.append(name)
    if not outputs:
        b, _, h, w = c0_prob.shape
        return c0_prob.new_empty((b, 0, h, w)), tuple()
    return torch.cat(outputs, dim=1), tuple(names)


def _v518_pair_bank(
    c0_prob: torch.Tensor,
    primary_local_probs: torch.Tensor,
) -> Tuple[torch.Tensor, Tuple[str, ...]]:
    """Compose one negative and one positive atomic edit in the same mask."""
    if primary_local_probs.ndim != 4 or primary_local_probs.shape[1] < 4:
        raise ValueError("V518 pair bank requires Delete/Fill/Trim/Expand")
    delete, fill, trim, expand = [
        primary_local_probs[:, index:index + 1] for index in range(4)
    ]
    pairs = []
    names = []
    for negative_name, negative in (("delete", delete), ("trim", trim)):
        for positive_name, positive in (("fill", fill), ("expand", expand)):
            candidate = (
                c0_prob + (negative - c0_prob) + (positive - c0_prob)
            ).clamp(EPS, 1.0 - EPS)
            pairs.append(candidate)
            names.append(f"{negative_name}+{positive_name}")
    return torch.cat(pairs, dim=1), tuple(names)



def _v519_candidate_metadata(
    *,
    c0_prob: torch.Tensor,
    local_action_ids: torch.Tensor,
    local_dose_values: torch.Tensor,
    pair_count: int,
    morph_names: Tuple[str, ...],
    global_count: int,
    exclude_erode_r6: bool = True,
    deploy_pair_candidates: bool = False,
    deploy_global_candidates: bool = False,
) -> Dict[str, torch.Tensor]:
    """Build explicit structural metadata for Preserve + V518 candidates.

    Families: Preserve=0, learned=1, pair=2, morphology=3, global=4.
    Actions: Preserve=0, Delete=1, Fill=2, Trim=3, Expand=4,
             Erode=5, Dilate=6, Open=7, Close=8, Global=9.

    Pair and global candidates remain available to A1 Oracle audits but are
    deliberately excluded from the V519 deployment bank because their audit
    contribution was negligible and GlobalRepair collapsed to duplicate masks.
    """
    device = c0_prob.device
    dtype = c0_prob.dtype
    family = [0]
    action = [0]
    dose = [0.0]
    radius = [0.0]
    deploy = [True]

    for index in range(int(local_action_ids.numel())):
        family.append(1)
        action.append(int(local_action_ids[index].item()) + 1)
        dose.append(float(local_dose_values[index].item()))
        radius.append(0.0)
        deploy.append(True)

    for _ in range(int(pair_count)):
        family.append(2)
        action.append(0)
        dose.append(1.0)
        radius.append(0.0)
        deploy.append(bool(deploy_pair_candidates))

    action_map = {"erode": 5, "dilate": 6, "open": 7, "close": 8}
    for name in morph_names:
        lower = str(name).lower()
        prefix, _, suffix = lower.partition("_r")
        try:
            r = int(suffix)
        except ValueError:
            r = 0
        family.append(3)
        action.append(action_map.get(prefix, 0))
        dose.append(1.0)
        radius.append(float(r))
        unsafe = bool(exclude_erode_r6 and prefix == "erode" and r >= 6)
        deploy.append(not unsafe)

    for _ in range(int(global_count)):
        family.append(4)
        action.append(9)
        dose.append(1.0)
        radius.append(0.0)
        deploy.append(bool(deploy_global_candidates))

    return {
        "family_ids": torch.tensor(family, device=device, dtype=torch.long),
        "action_ids": torch.tensor(action, device=device, dtype=torch.long),
        "dose_values": torch.tensor(dose, device=device, dtype=dtype),
        "radius_values": torch.tensor(radius, device=device, dtype=dtype),
        "deploy_mask": torch.tensor(deploy, device=device, dtype=torch.bool),
    }


def _v519_candidate_cause_stack(
    *,
    c0_prob: torch.Tensor,
    cause_maps: torch.Tensor,
    local_action_ids: torch.Tensor,
    pair_count: int,
    morph_names: Tuple[str, ...],
    global_count: int,
    failure_prob: torch.Tensor,
) -> torch.Tensor:
    """Map every candidate to the factual error evidence it is meant to fix."""
    cause = cause_maps.clamp(0.0, 1.0)
    outputs = [torch.zeros_like(c0_prob)]
    for action_id in local_action_ids.tolist():
        outputs.append(cause[:, int(action_id):int(action_id) + 1])

    # V518 pair order: Delete+Fill, Delete+Expand, Trim+Fill, Trim+Expand.
    pair_actions = ((0, 1), (0, 3), (2, 1), (2, 3))
    for left, right in pair_actions[:int(pair_count)]:
        outputs.append(torch.maximum(cause[:, left:left + 1], cause[:, right:right + 1]))

    fp = torch.maximum(cause[:, 0:1], cause[:, 2:3])
    fn = torch.maximum(cause[:, 1:2], cause[:, 3:4])
    for name in morph_names:
        lower = str(name).lower()
        outputs.append(fp if lower.startswith(("erode_", "open_")) else fn)

    failure = failure_prob
    if failure.ndim == 2:
        failure = failure[:, :, None, None]
    failure = failure[:, :1].expand_as(c0_prob).clamp(0.0, 1.0)
    outputs.extend([failure for _ in range(int(global_count))])
    return torch.cat(outputs, dim=1)


def _soft_dice_probs(probs: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    if probs.ndim == 3:
        probs = probs[:, None]
    if gt.ndim == 3:
        gt = gt[:, None]
    if probs.ndim != 4 or gt.ndim != 4:
        raise ValueError(f"Expected probs/gt rank 4, got {probs.ndim}/{gt.ndim}")
    if gt.shape[1] == 1 and probs.shape[1] != 1:
        gt = gt.expand(-1, probs.shape[1], -1, -1)
    inter = (probs * gt).flatten(2).sum(dim=-1)
    den = probs.flatten(2).sum(dim=-1) + gt.flatten(2).sum(dim=-1)
    return (2.0 * inter + 1e-7) / (den + 1e-7)


def _scale_gradient(value: torch.Tensor, scale: float) -> torch.Tensor:
    """Keep the forward value unchanged while bounding upstream gradients.

    ``scale=1`` is ordinary end-to-end learning, ``scale=0`` is stop-gradient,
    and values in ``(0, 1)`` keep every active module trainable while preventing
    a downstream verifier from numerically taking over its upstream generator.
    """
    if not isinstance(value, torch.Tensor):
        return value
    scale = float(scale)
    if scale >= 1.0:
        return value
    if scale <= 0.0:
        return value.detach()
    return value.detach() + scale * (value - value.detach())


class ErrorStateHead(nn.Module):
    """Predict no_edit/fp/fn/boundary/failure from image and factual C0."""

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        g = min(8, int(hidden_dim))
        while hidden_dim % g != 0 and g > 1:
            g -= 1
        self.encoder = nn.Sequential(
            nn.Conv2d(6, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(g, hidden_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(g, hidden_dim),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 5),
        )
        nn.init.zeros_(self.classifier[-1].weight)
        nn.init.zeros_(self.classifier[-1].bias)
        with torch.no_grad():
            self.classifier[-1].bias[0] = 0.8
            self.classifier[-1].bias[1:] = -0.2

    def forward(self, image: torch.Tensor, c0_prob: torch.Tensor) -> Dict[str, torch.Tensor]:
        c0_prob = _as_b1hw(c0_prob).clamp(EPS, 1.0 - EPS)
        size = c0_prob.shape[-2:]
        ent = _entropy(c0_prob)
        bnd = _soft_boundary(c0_prob)
        gray, edge = _gray_edge(image, size)
        x = torch.cat([c0_prob, 1.0 - c0_prob, ent, bnd, gray, edge], dim=1)
        feat = self.encoder(x)
        pooled = self.pool(feat).flatten(1)
        fg_area = c0_prob.flatten(1).mean(dim=1, keepdim=True)
        unc_mass = ent.flatten(1).mean(dim=1, keepdim=True)
        bnd_mass = bnd.flatten(1).mean(dim=1, keepdim=True)
        empty_flag = (c0_prob.flatten(1).max(dim=1, keepdim=True).values < 0.5).float()
        logits = self.classifier(torch.cat([pooled, fg_area, unc_mass, bnd_mass, empty_flag], dim=1))
        probs = F.softmax(logits, dim=-1)
        return {
            "error_state_logits": logits,
            "error_state_probs": probs,
            "p_no_edit": probs[:, 0],
            "p_fp": probs[:, 1],
            "p_fn": probs[:, 2],
            "p_boundary": probs[:, 3],
            "p_failure": probs[:, 4],
        }

    @staticmethod
    @torch.no_grad()
    def make_training_labels(
        c0_prob: torch.Tensor,
        gt: torch.Tensor,
        fp_threshold: float = 0.001,
        fn_threshold: float = 0.001,
        boundary_threshold: float = 0.003,
        failure_dice: float = 0.55,
    ) -> torch.Tensor:
        c0 = (_as_b1hw(c0_prob) >= 0.5).float()
        y = (_as_b1hw(gt) >= 0.5).float()
        inter = (c0 * y).flatten(1).sum(dim=1)
        den = c0.flatten(1).sum(dim=1) + y.flatten(1).sum(dim=1)
        dice = (2.0 * inter + 1e-7) / (den + 1e-7)
        fp_area = (c0 * (1.0 - y)).flatten(1).mean(dim=1)
        fn_area = ((1.0 - c0) * y).flatten(1).mean(dim=1)
        c0_b = _soft_boundary(c0, radius=1)
        y_b = _soft_boundary(y, radius=1)
        bnd_error = (c0_b - y_b).abs().flatten(1).mean(dim=1)
        c0_empty = c0.flatten(1).sum(dim=1) < 1.0
        y_nonempty = y.flatten(1).sum(dim=1) >= 1.0
        failure = (dice < failure_dice) | (c0_empty & y_nonempty)
        labels = torch.zeros(c0.shape[0], dtype=torch.long, device=c0.device)
        labels[failure] = 4
        rem = ~failure
        fp_case = rem & (fp_area > fp_threshold) & (fp_area >= fn_area)
        fn_case = rem & (~fp_case) & (fn_area > fn_threshold)
        bd_case = rem & (~fp_case) & (~fn_case) & (bnd_error > boundary_threshold)
        labels[fp_case] = 1
        labels[fn_case] = 2
        labels[bd_case] = 3
        return labels


class GatedLocalRepairGenerator(nn.Module):
    """Four typed local residual candidates: delete/fill/trim/expand.

    V486 changes the candidate generator from a nearly-no-op residual to a
    stronger but still local intervention.  The important contract is that C0
    is detached by the outer pipeline, so increasing edit strength here cannot
    send proposal-loss gradients back into the Base path.
    """

    def __init__(self, hidden_dim: int = 64, max_atom_delta: float = 2.0, delta_bias: float = -2.0, support_bias: float = -2.0) -> None:
        super().__init__()
        g = min(8, int(hidden_dim))
        while hidden_dim % g != 0 and g > 1:
            g -= 1
        self.max_atom_delta = float(max_atom_delta)
        self.delta_bias = float(delta_bias)
        self.trunk = nn.Sequential(
            nn.Conv2d(6, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(g, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(g, hidden_dim),
            nn.GELU(),
        )
        self.support_heads = nn.ModuleList([nn.Conv2d(hidden_dim, 1, 1) for _ in range(4)])
        self.delta_heads = nn.ModuleList([nn.Conv2d(hidden_dim, 1, 1) for _ in range(4)])
        self.register_buffer("typed_signs", torch.tensor([-1.0, 1.0, -1.0, 1.0]), persistent=False)
        for h in self.support_heads:
            nn.init.zeros_(h.weight)
            nn.init.constant_(h.bias, float(support_bias))
        for h in self.delta_heads:
            nn.init.zeros_(h.weight)
            nn.init.constant_(h.bias, 0.0)

    def forward(self, image: torch.Tensor, c0_prob: torch.Tensor, error_state_probs: torch.Tensor) -> Dict[str, torch.Tensor]:
        c0_prob = _as_b1hw(c0_prob).clamp(EPS, 1.0 - EPS)
        size = c0_prob.shape[-2:]
        ent = _entropy(c0_prob)
        bnd = _soft_boundary(c0_prob)
        gray, edge = _gray_edge(image, size)
        feat = self.trunk(torch.cat([c0_prob, 1.0 - c0_prob, ent, bnd, gray, edge], dim=1))
        gates = torch.stack(
            [error_state_probs[:, 1], error_state_probs[:, 2], error_state_probs[:, 3], error_state_probs[:, 3]],
            dim=1,
        )
        c0_logits = torch.logit(c0_prob)
        logits, supports, deltas, raw_magnitudes = [], [], [], []
        for k in range(4):
            support = torch.sigmoid(self.support_heads[k](feat))
            raw = self.delta_heads[k](feat)
            magnitude = torch.sigmoid(raw + self.delta_bias)
            sign = self.typed_signs[k].to(raw.device, raw.dtype)
            delta = sign * self.max_atom_delta * support * magnitude
            g = gates[:, k].view(-1, 1, 1, 1).clamp(0.0, 1.0)
            candidate = c0_logits + g * delta
            supports.append(support)
            deltas.append(g * delta)
            raw_magnitudes.append(magnitude)
            logits.append(candidate)
        return {
            "local_candidate_logits": torch.cat(logits, dim=1),
            "local_supports": torch.cat(supports, dim=1),
            "local_deltas": torch.cat(deltas, dim=1),
            "local_raw_magnitudes": torch.cat(raw_magnitudes, dim=1),
            "local_soft_gates": gates,
            "local_active_mask": gates > 0.15,
        }


class GatedGlobalRediscoveryGenerator(nn.Module):
    """Failure-gated global rediscovery candidate.

    The historical branch predicted only a bounded residual around C0.  That
    construction cannot recover an empty/fully missed Base mask because the
    factual logit may be far below zero.  V501 optionally predicts an
    independent segmentation probability from image/text semantic evidence and
    blends it with C0 only through the learned failure probability:

        C_global = (1 - p_failure) * C0 + p_failure * C_rediscovery

    This keeps normal cases close to Preserve while giving catastrophic cases a
    candidate that is not constrained by the factual logit magnitude.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_discovery: int = 0,
        max_delta: float = 4.0,
        semantic_channels: int = 512,
        direct_rediscovery: bool = False,
        initial_foreground_rate: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_discovery = int(num_discovery)
        self.max_delta = float(max_delta)
        self.semantic_channels = int(semantic_channels)
        self.direct_rediscovery = bool(direct_rediscovery)
        g = min(8, int(hidden_dim))
        while hidden_dim % g != 0 and g > 1:
            g -= 1
        self.encoder = nn.Sequential(
            nn.Conv2d(6, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(g, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(g, hidden_dim),
            nn.GELU(),
        )
        self.image_feature_proj = nn.Conv2d(
            self.semantic_channels, hidden_dim, 1, bias=False
        )
        self.text_feature_proj = nn.Linear(
            self.semantic_channels, hidden_dim, bias=False
        )
        self.semantic_fuse = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(g, hidden_dim),
            nn.GELU(),
        )
        self.heads = nn.ModuleList(
            [nn.Conv2d(hidden_dim, 1, 1) for _ in range(max(self.num_discovery, 1))]
        )
        initial_foreground_rate = min(
            max(float(initial_foreground_rate), 1.0e-4), 1.0 - 1.0e-4
        )
        initial_bias = math.log(
            initial_foreground_rate / (1.0 - initial_foreground_rate)
        )
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.constant_(
                head.bias,
                initial_bias if self.direct_rediscovery else -4.0,
            )

    def _semantic_image_map(
        self,
        image_features: Optional[torch.Tensor],
        size: Tuple[int, int],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(image_features, torch.Tensor) or image_features.ndim != 3:
            return reference.new_zeros((reference.shape[0], self.image_feature_proj.out_channels, *size))
        tokens = image_features
        patch_tokens = tokens[:, 1:]
        count = int(patch_tokens.shape[1])
        side = int(round(math.sqrt(max(count, 1))))
        if side * side != count:
            patch_tokens = tokens
            count = int(patch_tokens.shape[1])
            side = int(round(math.sqrt(max(count, 1))))
        if side * side != count or patch_tokens.shape[-1] != self.semantic_channels:
            return reference.new_zeros((reference.shape[0], self.image_feature_proj.out_channels, *size))
        fmap = patch_tokens.transpose(1, 2).reshape(
            tokens.shape[0], self.semantic_channels, side, side
        )
        fmap = self.image_feature_proj(fmap.to(dtype=reference.dtype))
        return F.interpolate(fmap, size=size, mode="bilinear", align_corners=False)

    def _text_context(
        self,
        text_features: Optional[torch.Tensor],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(text_features, torch.Tensor):
            return reference.new_zeros((reference.shape[0], self.text_feature_proj.out_features, 1, 1))
        text = text_features
        while text.ndim > 2:
            text = text.mean(dim=1)
        if text.ndim != 2 or text.shape[-1] != self.semantic_channels:
            return reference.new_zeros((reference.shape[0], self.text_feature_proj.out_features, 1, 1))
        return self.text_feature_proj(text.to(dtype=reference.dtype))[:, :, None, None]

    def forward(
        self,
        image: torch.Tensor,
        c0_prob: torch.Tensor,
        error_state_probs: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        c0_prob = _as_b1hw(c0_prob).clamp(EPS, 1.0 - EPS)
        batch, _, height, width = c0_prob.shape
        c0_logits = torch.logit(c0_prob)
        if self.num_discovery <= 0:
            return {
                "global_candidate_logits": c0_logits.new_empty((batch, 0, height, width)),
                "global_candidate_probs": c0_prob.new_empty((batch, 0, height, width)),
                "global_direct_probs": c0_prob.new_empty((batch, 0, height, width)),
                "global_active_mask": torch.zeros(batch, 0, dtype=torch.bool, device=c0_prob.device),
                "global_soft_gates": torch.zeros(batch, 0, device=c0_prob.device, dtype=c0_prob.dtype),
            }
        entropy = _entropy(c0_prob)
        boundary = _soft_boundary(c0_prob)
        gray, edge = _gray_edge(image, (height, width))
        feature = self.encoder(
            torch.cat([c0_prob, 1.0 - c0_prob, entropy, boundary, gray, edge], dim=1)
        )
        feature = feature + self._semantic_image_map(
            image_features, (height, width), c0_prob
        )
        feature = feature + self._text_context(text_features, c0_prob)
        feature = self.semantic_fuse(feature)
        p_failure = error_state_probs[:, 4].view(batch, 1, 1, 1).clamp(0.0, 1.0)
        candidate_logits = []
        candidate_probs = []
        direct_probs = []
        for index in range(self.num_discovery):
            raw = self.heads[index](feature)
            if self.direct_rediscovery:
                direct = torch.sigmoid(raw).clamp(EPS, 1.0 - EPS)
                candidate = (
                    (1.0 - p_failure) * c0_prob + p_failure * direct
                ).clamp(EPS, 1.0 - EPS)
                logits = torch.logit(candidate)
            else:
                direct = torch.sigmoid(raw).clamp(EPS, 1.0 - EPS)
                delta = self.max_delta * torch.tanh(raw)
                logits = c0_logits + p_failure * delta
                candidate = torch.sigmoid(logits).clamp(EPS, 1.0 - EPS)
            candidate_logits.append(logits)
            candidate_probs.append(candidate)
            direct_probs.append(direct)
        global_logits = torch.cat(candidate_logits, dim=1)
        global_probs = torch.cat(candidate_probs, dim=1)
        direct_probs_tensor = torch.cat(direct_probs, dim=1)
        soft_gates = error_state_probs[:, 4:5].expand(-1, self.num_discovery)
        return {
            "global_candidate_logits": global_logits,
            "global_candidate_probs": global_probs,
            "global_direct_probs": direct_probs_tensor,
            "global_active_mask": soft_gates > 0.30,
            "global_soft_gates": soft_gates,
        }


class LocalEffectVerifier(nn.Module):
    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        g = min(8, int(hidden_dim))
        while hidden_dim % g != 0 and g > 1:
            g -= 1
        self.encoder = nn.Sequential(
            nn.Conv2d(8, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(g, hidden_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(g, hidden_dim),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Linear(hidden_dim + 6, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 4))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, image: torch.Tensor, c0_prob: torch.Tensor, candidate_probs: torch.Tensor, supports: torch.Tensor) -> Dict[str, torch.Tensor]:
        c0_prob = _as_b1hw(c0_prob).clamp(EPS, 1.0 - EPS)
        b, k, h, w = candidate_probs.shape
        c0 = c0_prob.expand(-1, k, -1, -1)
        signed = candidate_probs - c0
        ent = _entropy(c0_prob).expand(-1, k, -1, -1)
        bnd = _soft_boundary(c0_prob).expand(-1, k, -1, -1)
        gray, edge = _gray_edge(image, (h, w))
        gray = gray.expand(-1, k, -1, -1)
        edge = edge.expand(-1, k, -1, -1)
        x = torch.stack([c0, candidate_probs, signed, supports, ent, bnd, gray, edge], dim=2).reshape(b * k, 8, h, w)
        feat = self.encoder(x)
        pooled = self.pool(feat).flatten(1).reshape(b, k, -1)
        denom = supports.flatten(2).sum(dim=-1).clamp_min(EPS)
        edit_area = supports.flatten(2).mean(dim=-1)
        edit_mag = (signed.abs() * supports).flatten(2).sum(dim=-1) / denom
        unc = (ent * supports).flatten(2).sum(dim=-1) / denom
        bnd_s = (bnd * supports).flatten(2).sum(dim=-1) / denom
        fg = c0_prob.flatten(1).mean(dim=1, keepdim=True).expand(-1, k)
        c0_unc = _entropy(c0_prob).flatten(1).mean(dim=1, keepdim=True).expand(-1, k)
        scalars = torch.stack([edit_area, edit_mag, unc, bnd_s, fg, c0_unc], dim=-1)
        out = self.head(torch.cat([pooled, scalars], dim=-1).reshape(b * k, -1)).reshape(b, k, 4)
        return {
            "local_pred_delta_dsc": out[:, :, 0],
            "local_pred_delta_nsd": out[:, :, 1],
            "local_pred_harm_logit": out[:, :, 2],
            "local_pred_harm_prob": torch.sigmoid(out[:, :, 2]),
            "local_pred_support_precision_logit": out[:, :, 3],
            "local_pred_support_precision": torch.sigmoid(out[:, :, 3]),
        }


class PixelCounterfactualComposer(nn.Module):
    """V488 M2: pixel-wise counterfactual intervention composer.

    The module is deliberately constrained: its output is a convex combination
    of the frozen M1 candidate probabilities at every pixel.  It cannot invent a
    free segmentation mask.  Therefore it may exceed the *case-wise* M1 oracle
    by combining complementary candidates, but it cannot exceed the pixel-wise
    candidate oracle (PWO) except for soft-threshold tie effects.
    """

    def __init__(self, cfg: Any, hidden_dim: int = 64, semantic_channels: int = 512) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.semantic_channels = int(semantic_channels)
        self.max_candidates = int(_m1(cfg, "V488_MAX_CANDIDATES", 8))
        self.dropout = float(_m1(cfg, "V488_M2_DROPOUT", 0.15))
        self.temperature = max(float(_m1(cfg, "V488_M2_TEMPERATURE", 0.35)), 1.0e-3)
        self.evidence_weight = float(_m1(cfg, "V488_M2_EVIDENCE_WEIGHT", 1.0))
        self.uncertainty_penalty = float(_m1(cfg, "V488_M2_UNCERTAINTY_PENALTY", 0.25))
        self.preserve_bias = float(_m1(cfg, "V488_M2_PRESERVE_BIAS", 0.25))

        groups = min(8, self.hidden_dim)
        while self.hidden_dim % groups != 0 and groups > 1:
            groups -= 1

        # Shared factual context: C0, entropy/boundary, image intensity/edge,
        # and the five M1 error-state probabilities broadcast spatially.
        self.context_encoder = nn.Sequential(
            nn.Conv2d(11, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
        )
        self.image_feature_proj = nn.Conv2d(self.semantic_channels, self.hidden_dim, 1, bias=False)
        self.text_feature_proj = nn.Linear(self.semantic_channels, self.hidden_dim, bias=False)

        # Candidate-specific information is only the candidate prediction and
        # its signed/absolute intervention relative to C0 plus M1 support.
        self.candidate_encoder = nn.Sequential(
            nn.Conv2d(4, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
        )
        self.type_embedding = nn.Embedding(self.max_candidates, self.hidden_dim)
        self.joint_conv1 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False)
        self.joint_norm1 = nn.GroupNorm(groups, self.hidden_dim)
        self.joint_conv2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False)
        self.joint_norm2 = nn.GroupNorm(groups, self.hidden_dim)

        self.effect_head = nn.Conv2d(self.hidden_dim, 1, 1)
        self.evidence_head = nn.Conv2d(self.hidden_dim, 1, 1)
        self.log_variance_head = nn.Conv2d(self.hidden_dim, 1, 1)

        nn.init.zeros_(self.type_embedding.weight)
        nn.init.zeros_(self.effect_head.weight)
        nn.init.zeros_(self.effect_head.bias)
        nn.init.zeros_(self.evidence_head.weight)
        nn.init.zeros_(self.evidence_head.bias)
        nn.init.zeros_(self.log_variance_head.weight)
        nn.init.constant_(self.log_variance_head.bias, -2.0)

    def _semantic_image_map(
        self,
        image_features: Optional[torch.Tensor],
        size: Tuple[int, int],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(image_features, torch.Tensor) or image_features.ndim != 3:
            return reference.new_zeros((reference.shape[0], self.hidden_dim, *size))
        tokens = image_features
        # UniMedCLIP ViT output includes one CLS token.  Fall back to all tokens
        # only when the remaining count is not a square.
        patch_tokens = tokens[:, 1:]
        count = int(patch_tokens.shape[1])
        side = int(round(math.sqrt(max(count, 1))))
        if side * side != count:
            patch_tokens = tokens
            count = int(patch_tokens.shape[1])
            side = int(round(math.sqrt(max(count, 1))))
        if side * side != count or patch_tokens.shape[-1] != self.semantic_channels:
            return reference.new_zeros((reference.shape[0], self.hidden_dim, *size))
        fmap = patch_tokens.transpose(1, 2).reshape(tokens.shape[0], self.semantic_channels, side, side)
        fmap = self.image_feature_proj(fmap.to(dtype=reference.dtype))
        return F.interpolate(fmap, size=size, mode="bilinear", align_corners=False)

    def _text_context(
        self,
        text_features: Optional[torch.Tensor],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(text_features, torch.Tensor):
            return reference.new_zeros((reference.shape[0], self.hidden_dim, 1, 1))
        text = text_features
        while text.ndim > 2:
            text = text.mean(dim=1)
        if text.ndim != 2 or text.shape[-1] != self.semantic_channels:
            return reference.new_zeros((reference.shape[0], self.hidden_dim, 1, 1))
        return self.text_feature_proj(text.to(dtype=reference.dtype))[:, :, None, None]

    def forward(
        self,
        image: torch.Tensor,
        c0_prob: torch.Tensor,
        candidate_probs: torch.Tensor,
        supports: torch.Tensor,
        error_state_probs: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        stochastic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        c0_prob = _as_b1hw(c0_prob).clamp(EPS, 1.0 - EPS)
        if candidate_probs.ndim != 4:
            raise ValueError(f"candidate_probs must be [B,K,H,W], got {tuple(candidate_probs.shape)}")
        b, k, h, w = candidate_probs.shape
        if k > self.max_candidates:
            raise ValueError(f"V488 supports at most {self.max_candidates} candidates, got {k}")

        c0 = c0_prob.expand(-1, k, -1, -1)
        signed = candidate_probs - c0
        abs_edit = signed.abs()

        supports = supports.to(device=candidate_probs.device, dtype=candidate_probs.dtype)
        if supports.ndim == 3:
            supports = supports[:, None]
        if supports.shape[-2:] != (h, w):
            supports = F.interpolate(supports, size=(h, w), mode="bilinear", align_corners=False)
        if supports.shape[1] < k:
            pad = abs_edit[:, supports.shape[1]:k]
            supports = torch.cat([supports, pad], dim=1)
        supports = supports[:, :k].clamp(0.0, 1.0)

        ent = _entropy(c0_prob)
        bnd = _soft_boundary(c0_prob)
        gray, edge = _gray_edge(image, (h, w))
        state = error_state_probs[:, :, None, None].expand(-1, -1, h, w)
        context = self.context_encoder(torch.cat([c0_prob, 1.0 - c0_prob, ent, bnd, gray, edge, state], dim=1))
        context = context + self._semantic_image_map(image_features, (h, w), c0_prob)
        context = context + self._text_context(text_features, c0_prob)

        cand_input = torch.stack([candidate_probs, signed, abs_edit, supports], dim=2).reshape(b * k, 4, h, w)
        cand_feat = self.candidate_encoder(cand_input)
        type_ids = torch.arange(k, device=candidate_probs.device).clamp_max(self.max_candidates - 1)
        type_bias = self.type_embedding(type_ids).to(dtype=candidate_probs.dtype)
        type_bias = type_bias[None].expand(b, -1, -1).reshape(b * k, self.hidden_dim, 1, 1)
        joint = cand_feat + context[:, None].expand(-1, k, -1, -1, -1).reshape(b * k, self.hidden_dim, h, w) + type_bias
        joint = F.gelu(self.joint_norm1(self.joint_conv1(joint)))
        joint = F.dropout2d(joint, p=self.dropout, training=bool(self.training or stochastic))
        joint = F.gelu(self.joint_norm2(self.joint_conv2(joint)))
        joint = F.dropout2d(joint, p=self.dropout, training=bool(self.training or stochastic))

        effect = self.effect_head(joint).reshape(b, k, h, w)
        evidence_logit = self.evidence_head(joint).reshape(b, k, h, w)
        log_variance = self.log_variance_head(joint).reshape(b, k, h, w).clamp(-6.0, 4.0)
        uncertainty = torch.sqrt(F.softplus(log_variance) + EPS)

        policy_logits = (
            effect
            + self.evidence_weight * F.logsigmoid(evidence_logit)
            - self.uncertainty_penalty * uncertainty
        ) / self.temperature
        policy_logits[:, 0] = policy_logits[:, 0] + self.preserve_bias
        weights = F.softmax(policy_logits, dim=1)
        fused = (weights * candidate_probs).sum(dim=1).clamp(EPS, 1.0 - EPS)
        expected_effect = (weights * effect).sum(dim=1, keepdim=True)
        expected_evidence = (weights * torch.sigmoid(evidence_logit)).sum(dim=1, keepdim=True)

        return {
            "m2_pixel_effect": effect,
            "m2_pixel_evidence_logit": evidence_logit,
            "m2_pixel_evidence": torch.sigmoid(evidence_logit),
            "m2_pixel_log_variance": log_variance,
            "m2_pixel_uncertainty": uncertainty,
            "m2_pixel_policy_logits": policy_logits,
            "m2_pixel_weights": weights,
            "m2_expected_effect": expected_effect,
            "m2_expected_evidence": expected_evidence,
            "m2_fused_probs": fused,
        }


class UncertaintyCalibratedSafeDeployer(nn.Module):
    """V488 M3: pixel-wise uncertainty/effect calibrated rollback to C0."""

    def __init__(self, cfg: Any, hidden_dim: int = 32) -> None:
        super().__init__()
        self.threshold = float(_m1(cfg, "V488_M3_GATE_THRESHOLD", 0.50))
        self.hard_inference = bool(_m1(cfg, "V488_M3_HARD_INFERENCE", True))
        self.straight_through = bool(_m1(cfg, "V488_M3_STRAIGHT_THROUGH", True))
        self.margin_scale = float(_m1(cfg, "V488_M3_MARGIN_SCALE", 6.0))
        self.evidence_scale = float(_m1(cfg, "V488_M3_EVIDENCE_SCALE", 2.0))
        beta = max(float(_m1(cfg, "V488_M3_UNCERTAINTY_BETA", 1.0)), 1.0e-4)
        edit_penalty = max(float(_m1(cfg, "V488_M3_EDIT_PENALTY", 0.10)), 1.0e-4)
        self.raw_beta = nn.Parameter(torch.log(torch.expm1(torch.tensor(beta))))
        self.raw_edit_penalty = nn.Parameter(torch.log(torch.expm1(torch.tensor(edit_penalty))))

        groups = min(8, int(hidden_dim))
        while hidden_dim % groups != 0 and groups > 1:
            groups -= 1
        self.gate_net = nn.Sequential(
            nn.Conv2d(8, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1),
        )
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, -1.0)

    def forward(
        self,
        c0_prob: torch.Tensor,
        m2_mean: torch.Tensor,
        m2_variance: torch.Tensor,
        expected_effect: torch.Tensor,
        expected_evidence: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        c0 = _as_b1hw(c0_prob).clamp(EPS, 1.0 - EPS)
        mean = _as_b1hw(m2_mean).clamp(EPS, 1.0 - EPS)
        variance = _as_b1hw(m2_variance).clamp_min(0.0)
        effect = _as_b1hw(expected_effect)
        evidence = _as_b1hw(expected_evidence).clamp(0.0, 1.0)
        uncertainty = torch.sqrt(variance + EPS)
        abs_edit = (mean - c0).abs()
        entropy = _entropy(mean)
        boundary = _soft_boundary(c0)
        beta = F.softplus(self.raw_beta)
        edit_penalty = F.softplus(self.raw_edit_penalty)
        risk_adjusted_margin = effect - beta * uncertainty - edit_penalty * abs_edit

        gate_features = torch.cat([
            c0,
            mean,
            abs_edit,
            uncertainty,
            entropy,
            boundary,
            effect,
            evidence,
        ], dim=1)
        learned_logit = self.gate_net(gate_features)
        gate_logit = learned_logit + self.margin_scale * risk_adjusted_margin + self.evidence_scale * (evidence - 0.5)
        gate_prob = torch.sigmoid(gate_logit)
        hard_gate = (gate_prob >= self.threshold).to(gate_prob.dtype)
        if self.training and self.straight_through:
            deploy_gate = hard_gate.detach() - gate_prob.detach() + gate_prob
        elif (not self.training) and self.hard_inference:
            deploy_gate = hard_gate
        else:
            deploy_gate = gate_prob

        final = (deploy_gate * mean + (1.0 - deploy_gate) * c0).clamp(EPS, 1.0 - EPS)
        return {
            "m3_gate_logit": gate_logit,
            "m3_gate_prob": gate_prob,
            "m3_deploy_gate": deploy_gate,
            "m3_hard_gate": hard_gate,
            "m3_uncertainty": uncertainty,
            "m3_risk_adjusted_margin": risk_adjusted_margin,
            "m3_beta": beta.expand(c0.shape[0]),
            "m3_edit_penalty_value": edit_penalty.expand(c0.shape[0]),
            "fused_probs": final[:, 0],
            "final_probs": final[:, 0],
        }

class AsymmetricSparseCounterfactualComposer(nn.Module):
    """V489 M2: asymmetric sparse edit gate + conditional candidate router.

    The module separates two decisions that have very different class balance:

    1. ``edit_gate`` predicts whether Preserve/C0 should be changed at a pixel.
    2. ``route_probs`` predicts which non-base intervention should be used,
       conditioned on opening the edit gate.

    The primary output is a convex combination of C0 and the M1 candidates.
    An optional, strongly bounded residual refinement is restricted to the
    candidate-disagreement support, so it cannot become a free full-image
    segmentation decoder.  The convex-only output is always returned
    separately because PWO is a strict diagnostic upper bound only for the
    candidate-constrained path.
    """

    def __init__(self, cfg: Any, hidden_dim: int = 64, semantic_channels: int = 512) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.semantic_channels = int(semantic_channels)
        self.max_candidates = int(_m1(cfg, "V489_MAX_CANDIDATES", 8))
        self.dropout = float(_m1(cfg, "V489_M2_DROPOUT", 0.15))
        self.route_temperature = max(float(_m1(cfg, "V489_M2_ROUTE_TEMPERATURE", 0.70)), 1.0e-3)
        self.effect_weight = float(_m1(cfg, "V489_M2_ROUTE_EFFECT_WEIGHT", 0.50))
        self.evidence_weight = float(_m1(cfg, "V489_M2_ROUTE_EVIDENCE_WEIGHT", 0.25))
        self.uncertainty_penalty = float(_m1(cfg, "V489_M2_ROUTE_UNCERTAINTY_PENALTY", 0.10))
        self.residual_enabled = bool(_m1(cfg, "V489_M2_RESIDUAL_ENABLED", True))
        self.residual_scale = max(float(_m1(cfg, "V489_M2_RESIDUAL_SCALE", 0.03)), 0.0)
        self.residual_support_floor = max(float(_m1(cfg, "V489_M2_RESIDUAL_SUPPORT_FLOOR", 0.02)), EPS)

        initial_edit_rate = float(_m1(cfg, "V489_M2_INITIAL_EDIT_RATE", 0.02))
        initial_edit_rate = min(max(initial_edit_rate, 1.0e-4), 1.0 - 1.0e-4)
        # A 2% sigmoid probability requires a negative bias: logit(0.02)≈-3.89.
        self.initial_gate_bias = math.log(initial_edit_rate / (1.0 - initial_edit_rate))

        groups = min(8, self.hidden_dim)
        while self.hidden_dim % groups != 0 and groups > 1:
            groups -= 1

        # 11 factual channels from V488 + 3 explicit candidate-disagreement
        # summaries.  The two decoders only share this lightweight context.
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(14, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
        )
        self.image_feature_proj = nn.Conv2d(self.semantic_channels, self.hidden_dim, 1, bias=False)
        self.text_feature_proj = nn.Linear(self.semantic_channels, self.hidden_dim, bias=False)

        # Gate decoder: independent final layers so the sparse binary decision
        # is not dominated by the four-way route classification gradients.
        self.gate_decoder = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
        )
        self.gate_head = nn.Conv2d(self.hidden_dim, 1, 1)

        # Route decoder receives each signed intervention explicitly.  Its
        # final three layers are independent from the gate decoder.
        self.route_candidate_encoder = nn.Sequential(
            nn.Conv2d(4, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
        )
        self.route_type_embedding = nn.Embedding(max(self.max_candidates - 1, 1), self.hidden_dim)
        self.route_decoder = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
        )
        self.route_head = nn.Conv2d(self.hidden_dim, 1, 1)
        self.effect_head = nn.Conv2d(self.hidden_dim, 1, 1)
        self.evidence_head = nn.Conv2d(self.hidden_dim, 1, 1)
        self.log_variance_head = nn.Conv2d(self.hidden_dim, 1, 1)

        # The refinement starts as an exact no-op and is tightly bounded.
        # V490 uses a strict convex M2.  Do not leave a disabled residual head
        # as trainable dead parameters with no gradient.
        self.residual_head = (
            nn.Conv2d(self.hidden_dim, 1, 1)
            if self.residual_enabled and self.residual_scale > 0.0
            else None
        )

        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, self.initial_gate_bias)
        nn.init.zeros_(self.route_head.weight)
        nn.init.zeros_(self.route_head.bias)
        nn.init.zeros_(self.effect_head.weight)
        nn.init.zeros_(self.effect_head.bias)
        nn.init.zeros_(self.evidence_head.weight)
        nn.init.zeros_(self.evidence_head.bias)
        nn.init.zeros_(self.log_variance_head.weight)
        nn.init.constant_(self.log_variance_head.bias, -2.0)
        if self.residual_head is not None:
            nn.init.zeros_(self.residual_head.weight)
            nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.route_type_embedding.weight)

    def _semantic_image_map(
        self,
        image_features: Optional[torch.Tensor],
        size: Tuple[int, int],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(image_features, torch.Tensor) or image_features.ndim != 3:
            return reference.new_zeros((reference.shape[0], self.hidden_dim, *size))
        tokens = image_features
        patch_tokens = tokens[:, 1:]
        count = int(patch_tokens.shape[1])
        side = int(round(math.sqrt(max(count, 1))))
        if side * side != count:
            patch_tokens = tokens
            count = int(patch_tokens.shape[1])
            side = int(round(math.sqrt(max(count, 1))))
        if side * side != count or patch_tokens.shape[-1] != self.semantic_channels:
            return reference.new_zeros((reference.shape[0], self.hidden_dim, *size))
        fmap = patch_tokens.transpose(1, 2).reshape(tokens.shape[0], self.semantic_channels, side, side)
        fmap = self.image_feature_proj(fmap.to(dtype=reference.dtype))
        return F.interpolate(fmap, size=size, mode="bilinear", align_corners=False)

    def _text_context(
        self,
        text_features: Optional[torch.Tensor],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(text_features, torch.Tensor):
            return reference.new_zeros((reference.shape[0], self.hidden_dim, 1, 1))
        text = text_features
        while text.ndim > 2:
            text = text.mean(dim=1)
        if text.ndim != 2 or text.shape[-1] != self.semantic_channels:
            return reference.new_zeros((reference.shape[0], self.hidden_dim, 1, 1))
        return self.text_feature_proj(text.to(dtype=reference.dtype))[:, :, None, None]

    def forward(
        self,
        image: torch.Tensor,
        c0_prob: torch.Tensor,
        candidate_probs: torch.Tensor,
        supports: torch.Tensor,
        error_state_probs: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        stochastic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        c0_prob = _as_b1hw(c0_prob).clamp(EPS, 1.0 - EPS)
        if candidate_probs.ndim != 4:
            raise ValueError(f"candidate_probs must be [B,K,H,W], got {tuple(candidate_probs.shape)}")
        b, k, h, w = candidate_probs.shape
        if k < 2:
            raise ValueError("V489 M2 requires Preserve plus at least one intervention candidate.")
        if k > self.max_candidates:
            raise ValueError(f"V489 supports at most {self.max_candidates} candidates, got {k}")

        nonbase = candidate_probs[:, 1:]
        n = nonbase.shape[1]
        c0_nonbase = c0_prob.expand(-1, n, -1, -1)
        signed = nonbase - c0_nonbase
        abs_edit = signed.abs()

        supports = supports.to(device=candidate_probs.device, dtype=candidate_probs.dtype)
        if supports.ndim == 3:
            supports = supports[:, None]
        if supports.shape[-2:] != (h, w):
            supports = F.interpolate(supports, size=(h, w), mode="bilinear", align_corners=False)
        # The pipeline passes Preserve support in slot 0.
        if supports.shape[1] == k:
            nonbase_supports = supports[:, 1:]
        else:
            nonbase_supports = supports[:, :n]
        if nonbase_supports.shape[1] < n:
            nonbase_supports = torch.cat(
                [nonbase_supports, abs_edit[:, nonbase_supports.shape[1]:]], dim=1
            )
        nonbase_supports = nonbase_supports[:, :n].clamp(0.0, 1.0)

        ent = _entropy(c0_prob)
        bnd = _soft_boundary(c0_prob)
        gray, edge = _gray_edge(image, (h, w))
        state = error_state_probs[:, :, None, None].expand(-1, -1, h, w)
        max_abs_edit = abs_edit.max(dim=1, keepdim=True).values
        mean_abs_edit = abs_edit.mean(dim=1, keepdim=True)
        support_union = nonbase_supports.max(dim=1, keepdim=True).values
        shared_input = torch.cat(
            [
                c0_prob,
                1.0 - c0_prob,
                ent,
                bnd,
                gray,
                edge,
                state,
                max_abs_edit,
                mean_abs_edit,
                support_union,
            ],
            dim=1,
        )
        shared = self.shared_encoder(shared_input)
        shared = shared + self._semantic_image_map(image_features, (h, w), c0_prob)
        shared = shared + self._text_context(text_features, c0_prob)

        gate_feat = self.gate_decoder(shared)
        gate_feat = F.dropout2d(gate_feat, p=self.dropout, training=bool(self.training or stochastic))
        gate_logit = self.gate_head(gate_feat)
        edit_gate = torch.sigmoid(gate_logit)

        route_input = torch.stack([nonbase, signed, abs_edit, nonbase_supports], dim=2)
        route_input = route_input.reshape(b * n, 4, h, w)
        route_feat = self.route_candidate_encoder(route_input)
        shared_rep = shared[:, None].expand(-1, n, -1, -1, -1).reshape(b * n, self.hidden_dim, h, w)
        type_ids = torch.arange(n, device=nonbase.device).clamp_max(self.route_type_embedding.num_embeddings - 1)
        type_bias = self.route_type_embedding(type_ids).to(dtype=nonbase.dtype)
        type_bias = type_bias[None].expand(b, -1, -1).reshape(b * n, self.hidden_dim, 1, 1)
        route_feat = self.route_decoder(route_feat + shared_rep + type_bias)
        route_feat = F.dropout2d(route_feat, p=self.dropout, training=bool(self.training or stochastic))

        raw_route = self.route_head(route_feat).reshape(b, n, h, w)
        effect = self.effect_head(route_feat).reshape(b, n, h, w)
        evidence_logit = self.evidence_head(route_feat).reshape(b, n, h, w)
        log_variance = self.log_variance_head(route_feat).reshape(b, n, h, w).clamp(-6.0, 4.0)
        uncertainty = torch.sqrt(F.softplus(log_variance) + EPS)
        route_logits = (
            raw_route
            + self.effect_weight * effect
            + self.evidence_weight * F.logsigmoid(evidence_logit)
            - self.uncertainty_penalty * uncertainty
        ) / self.route_temperature
        route_probs = F.softmax(route_logits, dim=1)

        preserve_weight = 1.0 - edit_gate
        nonbase_weights = edit_gate * route_probs
        pixel_weights = torch.cat([preserve_weight, nonbase_weights], dim=1)
        convex = (pixel_weights * candidate_probs).sum(dim=1, keepdim=True).clamp(EPS, 1.0 - EPS)

        if self.residual_head is not None:
            disagreement_support = torch.maximum(
                support_union,
                (max_abs_edit / self.residual_support_floor).clamp(0.0, 1.0),
            )
            bounded_residual = (
                self.residual_scale
                * torch.tanh(self.residual_head(gate_feat))
                * edit_gate
                * disagreement_support
            )
        else:
            bounded_residual = torch.zeros_like(convex)
        refined = (convex + bounded_residual).clamp(EPS, 1.0 - EPS)

        preserve_effect = torch.zeros_like(c0_prob)
        preserve_evidence = torch.ones_like(c0_prob)
        preserve_logvar = torch.full_like(c0_prob, -6.0)
        all_effect = torch.cat([preserve_effect, effect], dim=1)
        all_evidence_logit = torch.cat([torch.full_like(c0_prob, 6.0), evidence_logit], dim=1)
        all_log_variance = torch.cat([preserve_logvar, log_variance], dim=1)
        all_uncertainty = torch.sqrt(F.softplus(all_log_variance) + EPS)
        expected_effect = (pixel_weights * all_effect).sum(dim=1, keepdim=True)
        expected_evidence = (
            pixel_weights * torch.cat([preserve_evidence, torch.sigmoid(evidence_logit)], dim=1)
        ).sum(dim=1, keepdim=True)

        return {
            "m2_edit_gate_logit": gate_logit,
            "m2_edit_gate_prob": edit_gate,
            "m2_route_logits": route_logits,
            "m2_route_probs": route_probs,
            "m2_route_raw_logits": raw_route,
            "m2_pixel_effect": all_effect,
            "m2_pixel_evidence_logit": all_evidence_logit,
            "m2_pixel_evidence": torch.sigmoid(all_evidence_logit),
            "m2_pixel_log_variance": all_log_variance,
            "m2_pixel_uncertainty": all_uncertainty,
            "m2_pixel_policy_logits": torch.log(pixel_weights.clamp_min(EPS)),
            "m2_pixel_weights": pixel_weights,
            "m2_expected_effect": expected_effect,
            "m2_expected_evidence": expected_evidence,
            "m2_candidate_disagreement": max_abs_edit,
            "m2_support_union": support_union,
            "m2_convex_probs": convex[:, 0],
            "m2_residual_map": bounded_residual,
            "m2_fused_probs": refined[:, 0],
        }


class CausalLocalPixelEditor(nn.Module):
    """V492-V494 M2: support-constrained candidate-conditioned causal editor.

    M1 candidates are treated as explicit interventions relative to factual C0.
    The editor separates four roles that were entangled in the V491 gate:

    1. candidate support: where an M1 intervention is structurally defined;
    2. benefit probability: whether editing is expected to improve C0;
    3. edit amplitude: how far to move from C0 toward the selected candidate;
    4. sparse route: which intervention should be applied.

    The forward path cannot invent an unconstrained mask. Every non-zero edit
    is a convex movement from C0 toward one M1 candidate. V493 adds a dense,
    candidate-conditioned causal-utility map. Route, Benefit and Amplitude are
    aligned to the same selected intervention, which prevents the V492 failure
    where a globally-positive Benefit label could execute a different harmful
    candidate. V494 removes the redundant learned amplitude attenuation. For a
    binary segmentation target and a candidate constrained to [0,1], every
    candidate that improves absolute pixel error has an exact convex optimum at
    full candidate dose (alpha=1); alpha=0 is optimal for a non-improving
    candidate. The candidate-conditioned utility therefore becomes the sole
    causal accept/reject gate and is deployed with a straight-through binary
    decision. A straight-through top-k route preserves sparse deployment and
    differentiable training.
    """

    def __init__(self, cfg: Any, hidden_dim: int = 64, semantic_channels: int = 512) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.semantic_channels = int(semantic_channels)
        self.max_candidates = int(_m1(cfg, "V489_MAX_CANDIDATES", 8))
        self.dropout = float(_m1(cfg, "V492_M2_DROPOUT", 0.10))
        self.route_temperature = max(
            float(_m1(cfg, "V492_M2_ROUTE_TEMPERATURE", 0.70)), 1.0e-3
        )
        self.route_topk = max(1, int(_m1(cfg, "V492_M2_ROUTE_TOPK", 1)))
        self.utility_weight = float(_m1(cfg, "V492_M2_UTILITY_LOGIT_WEIGHT", 0.50))
        self.v493_enabled = bool(
            _m1(cfg, "V493_CANDIDATE_CONDITIONED_CAUSAL_ENABLED", False)
        )
        self.v494_enabled = bool(
            _m1(cfg, "V494_DIRECT_CAUSAL_DOSE_ENABLED", False)
        ) and self.v493_enabled
        self.v495_enabled = bool(
            _m1(cfg, "V495_SINGLE_DECISION_ENABLED", False)
        ) and self.v494_enabled
        self.v495_mc_logit_aggregation = bool(
            _m1(cfg, "V495_MC_LOGIT_AGGREGATION_ENABLED", True)
        ) and self.v495_enabled
        # V498 removes the remaining train/deploy mismatch without giving M2
        # an irreversible Preserve-vs-edit decision.  Preserve is represented
        # as a zero-logit reference action for supervision and M3 evidence,
        # while the deployed M2 proposal remains the complete hard top-1
        # non-Base intervention.  M3 is still the only module allowed to accept
        # or reject that proposal.
        self.v498_enabled = bool(
            _m1(cfg, "V498_CONSISTENT_FULL_PROPOSAL_ENABLED", False)
        ) and self.v495_enabled
        # V500 factorises the all-action reference into two calibrated tasks:
        # (a) does any candidate deserve consideration, and (b) which
        # candidate is the least-regret intervention.  The binary presence
        # reference is supervision/evidence only; M2 still emits one complete
        # non-Base proposal and M3 remains the sole irreversible accept/reject
        # decision.
        self.v500_enabled = bool(
            _m1(cfg, "V500_HIERARCHICAL_SAFE_ROUTE_ENABLED", False)
        ) and self.v498_enabled
        self.v501_enabled = bool(
            _m1(cfg, "V501_BASE_ANCHORED_SELECTIVE_REPAIR_ENABLED", False)
        ) and self.v500_enabled
        self.v501_gate_threshold = min(
            max(float(_m1(cfg, "V501_M2_GATE_THRESHOLD", 0.05)), 1.0e-4),
            1.0 - 1.0e-4,
        )
        # V502 replaces the inconsistent hard Preserve/route/gate chain with a
        # single hierarchical soft action distribution:
        #   P(Preserve)=1-g, P(C_k)=g*q_k.
        # It is intentionally built on the V501 detached-anchor contract so M2
        # can learn aggressively without sending gradients into Base or M1.
        self.v502_enabled = bool(
            _m1(cfg, "V502_HIERARCHICAL_UTILITY_SOFT_ROUTER_ENABLED", False)
        ) and self.v501_enabled
        self.v504_enabled = bool(
            _m1(cfg, "V504_REALIZABLE_POTENTIAL_OUTCOME_ENABLED", False)
        ) and self.v502_enabled
        self.v503_enabled = (
            bool(_m1(cfg, "V503_FACTUAL_ATOMIC_CAUSAL_ENABLED", False))
            or self.v504_enabled
        ) and self.v502_enabled
        if self.v502_enabled:
            self.route_temperature = max(
                float(_m1(cfg, "V502_M2_ROUTE_TEMPERATURE", 1.0)), 1.0e-3
            )
        self.pixel_utility_weight = float(
            _m1(cfg, "V493_M2_PIXEL_UTILITY_LOGIT_WEIGHT", 1.0)
        )
        self.selected_utility_weight = float(
            _m1(cfg, "V493_M2_SELECTED_UTILITY_BENEFIT_WEIGHT", 1.0)
        )

        initial_benefit = float(_m1(cfg, "V492_M2_INITIAL_BENEFIT_RATE", 0.02))
        initial_benefit = min(max(initial_benefit, 1.0e-4), 1.0 - 1.0e-4)
        benefit_bias = math.log(initial_benefit / (1.0 - initial_benefit))
        initial_amplitude = float(_m1(cfg, "V492_M2_INITIAL_AMPLITUDE", 0.25))
        initial_amplitude = min(max(initial_amplitude, 1.0e-4), 1.0 - 1.0e-4)
        amplitude_bias = math.log(initial_amplitude / (1.0 - initial_amplitude))

        groups = min(8, self.hidden_dim)
        while self.hidden_dim % groups != 0 and groups > 1:
            groups -= 1

        # C0, complement, entropy, boundary, image gray/edge, five error-state
        # probabilities, and three candidate-disagreement summaries.
        self.context_stem = nn.Sequential(
            nn.Conv2d(14, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
        )
        # Multi-scale local evidence.  This borrows the refinement principle of
        # combining fine local detail with coarser context, but uses only the
        # existing frozen image/text encoders and trainable task modules.
        self.fine_branch = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
        )
        self.medium_branch = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
        )
        self.coarse_branch = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
        )
        self.context_fuse = nn.Sequential(
            nn.Conv2d(self.hidden_dim * 3, self.hidden_dim, 1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
        )
        self.image_feature_proj = nn.Conv2d(
            self.semantic_channels, self.hidden_dim, 1, bias=False
        )
        self.text_feature_proj = nn.Linear(
            self.semantic_channels, self.hidden_dim, bias=False
        )

        # Candidate route branch.  Each intervention is encoded independently,
        # then receives both pixel-wise and image-level utility logits.
        self.candidate_encoder = nn.Sequential(
            nn.Conv2d(4, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
        )
        self.route_type_embedding = nn.Embedding(
            max(self.max_candidates - 1, 1), self.hidden_dim
        )
        self.route_decoder = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
        )
        self.route_head = nn.Conv2d(self.hidden_dim, 1, 1)
        # Per-candidate, per-pixel probability of a positive realizable
        # treatment effect. It is directly supervised from the exact convex
        # counterfactual teacher in utils/v484_loss.py.
        self.candidate_utility_head = nn.Conv2d(self.hidden_dim, 1, 1)
        self.utility_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.hidden_dim, 1),
        )

        # Benefit and amplitude are deliberately independent.  Benefit answers
        # whether to edit; amplitude estimates the optimal convex dose.
        self.benefit_decoder = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
            nn.Dropout2d(self.dropout),
        )
        self.amplitude_decoder = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
            nn.Dropout2d(self.dropout),
        )
        self.benefit_head = nn.Conv2d(self.hidden_dim, 1, 1)
        self.amplitude_head = nn.Conv2d(self.hidden_dim, 1, 1)
        self.candidate_presence_head = (
            nn.Conv2d(self.hidden_dim, 1, 1) if self.v500_enabled else None
        )

        nn.init.zeros_(self.route_type_embedding.weight)
        nn.init.zeros_(self.route_head.weight)
        nn.init.zeros_(self.route_head.bias)
        nn.init.zeros_(self.candidate_utility_head.weight)
        nn.init.constant_(
            self.candidate_utility_head.bias,
            benefit_bias if self.v494_enabled else 0.0,
        )
        nn.init.zeros_(self.utility_head[-1].weight)
        nn.init.zeros_(self.utility_head[-1].bias)
        nn.init.zeros_(self.benefit_head.weight)
        nn.init.constant_(self.benefit_head.bias, benefit_bias)
        nn.init.zeros_(self.amplitude_head.weight)
        nn.init.constant_(self.amplitude_head.bias, amplitude_bias)
        if self.candidate_presence_head is not None:
            initial_candidate_rate = float(
                _m1(
                    cfg,
                    "V502_M2_INITIAL_EDIT_RATE" if self.v502_enabled
                    else "V500_M2_INITIAL_CANDIDATE_RATE",
                    0.005 if self.v502_enabled else 0.02,
                )
            )
            initial_candidate_rate = min(
                max(initial_candidate_rate, 1.0e-4), 1.0 - 1.0e-4
            )
            initial_presence_bias = math.log(
                initial_candidate_rate / (1.0 - initial_candidate_rate)
            )
            nn.init.zeros_(self.candidate_presence_head.weight)
            nn.init.constant_(
                self.candidate_presence_head.bias, initial_presence_bias
            )

    def _semantic_image_map(
        self,
        image_features: Optional[torch.Tensor],
        size: Tuple[int, int],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(image_features, torch.Tensor) or image_features.ndim != 3:
            return reference.new_zeros((reference.shape[0], self.hidden_dim, *size))
        tokens = image_features
        patch_tokens = tokens[:, 1:]
        count = int(patch_tokens.shape[1])
        side = int(round(math.sqrt(max(count, 1))))
        if side * side != count:
            patch_tokens = tokens
            count = int(patch_tokens.shape[1])
            side = int(round(math.sqrt(max(count, 1))))
        if side * side != count or patch_tokens.shape[-1] != self.semantic_channels:
            return reference.new_zeros((reference.shape[0], self.hidden_dim, *size))
        fmap = patch_tokens.transpose(1, 2).reshape(
            tokens.shape[0], self.semantic_channels, side, side
        )
        fmap = self.image_feature_proj(fmap.to(dtype=reference.dtype))
        return F.interpolate(fmap, size=size, mode="bilinear", align_corners=False)

    def _text_context(
        self,
        text_features: Optional[torch.Tensor],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(text_features, torch.Tensor):
            return reference.new_zeros((reference.shape[0], self.hidden_dim, 1, 1))
        text = text_features
        while text.ndim > 2:
            text = text.mean(dim=1)
        if text.ndim != 2 or text.shape[-1] != self.semantic_channels:
            return reference.new_zeros((reference.shape[0], self.hidden_dim, 1, 1))
        return self.text_feature_proj(text.to(dtype=reference.dtype))[:, :, None, None]

    @staticmethod
    def _resize_supports(
        supports: torch.Tensor,
        candidate_probs: torch.Tensor,
        c0_prob: torch.Tensor,
    ) -> torch.Tensor:
        b, k, h, w = candidate_probs.shape
        n = k - 1
        supports = supports.to(
            device=candidate_probs.device, dtype=candidate_probs.dtype
        )
        if supports.ndim == 3:
            supports = supports[:, None]
        if supports.shape[-2:] != (h, w):
            supports = F.interpolate(
                supports, size=(h, w), mode="bilinear", align_corners=False
            )
        if supports.shape[1] == k:
            supports = supports[:, 1:]
        else:
            supports = supports[:, :n]
        if supports.shape[1] < n:
            fallback = (
                candidate_probs[:, 1 + supports.shape[1]:] - c0_prob
            ).abs()
            supports = torch.cat([supports, fallback], dim=1)
        return supports[:, :n].clamp(0.0, 1.0)

    def _multiscale_context(self, stem: torch.Tensor) -> torch.Tensor:
        h, w = stem.shape[-2:]
        fine = self.fine_branch(stem)
        medium_in = F.avg_pool2d(stem, kernel_size=2, stride=2, ceil_mode=True)
        medium = self.medium_branch(medium_in)
        medium = F.interpolate(
            medium, size=(h, w), mode="bilinear", align_corners=False
        )
        coarse_in = F.avg_pool2d(stem, kernel_size=4, stride=4, ceil_mode=True)
        coarse = self.coarse_branch(coarse_in)
        coarse = F.interpolate(
            coarse, size=(h, w), mode="bilinear", align_corners=False
        )
        return self.context_fuse(torch.cat([fine, medium, coarse], dim=1))

    def _sparse_route(self, route_probs: torch.Tensor) -> torch.Tensor:
        n = route_probs.shape[1]
        topk = min(self.route_topk, n)
        if topk == n:
            return route_probs
        values, indices = route_probs.topk(topk, dim=1)
        hard = torch.zeros_like(route_probs).scatter_(1, indices, values)
        hard = hard / hard.sum(dim=1, keepdim=True).clamp_min(EPS)
        if self.training:
            return hard + route_probs - route_probs.detach()
        return hard

    def _action_reference_distribution(
        self,
        route_logits: torch.Tensor,
        route_probs: torch.Tensor,
        candidate_presence_logit: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build a Preserve/candidate reference without candidate-count bias.

        V498 concatenated one fixed Preserve logit with N candidate logits and
        applied one softmax.  At equal logits Preserve therefore received only
        1/(N+1) probability, even when the teacher was almost entirely
        Preserve.  V500 uses a hierarchical probability factorisation:

            P(Preserve) = 1 - P(any candidate)
            P(C_k)      = P(any candidate) * P(k | candidate)

        This distribution is never used to attenuate the full M2 proposal.
        """
        if self.v500_enabled:
            if not isinstance(candidate_presence_logit, torch.Tensor):
                raise RuntimeError(
                    "V500 requires candidate_presence_logit for the action reference."
                )
            presence_prob = torch.sigmoid(candidate_presence_logit)
            action_probs = torch.cat(
                [1.0 - presence_prob, presence_prob * route_probs], dim=1
            ).clamp_min(EPS)
            action_probs = action_probs / action_probs.sum(
                dim=1, keepdim=True
            ).clamp_min(EPS)
            action_logits = torch.log(action_probs)
            best_candidate = route_logits.argmax(dim=1, keepdim=True) + 1
            presence_threshold_logit = math.log(
                self.v501_gate_threshold / (1.0 - self.v501_gate_threshold)
            ) if self.v501_enabled else 0.0
            action_index = torch.where(
                candidate_presence_logit > presence_threshold_logit,
                best_candidate,
                torch.zeros_like(best_candidate),
            )
        else:
            preserve_reference_logit = torch.zeros_like(route_logits[:, :1])
            action_logits = torch.cat(
                [preserve_reference_logit, route_logits], dim=1
            )
            action_probs = F.softmax(action_logits, dim=1)
            action_index = action_logits.argmax(dim=1, keepdim=True)
            presence_prob = 1.0 - action_probs[:, :1]

        action_hard = torch.zeros_like(action_probs).scatter_(
            1, action_index, 1.0
        )
        if self.v502_enabled:
            # V502 never inserts a hidden hard decision inside M2.  The action
            # policy is the same continuous distribution used to compose the
            # proposal; M3 remains the sole final hard accept/reject module.
            action_policy = action_probs
        else:
            action_policy = (
                action_hard + action_probs - action_probs.detach()
                if self.training else action_hard
            )
        return {
            "candidate_presence_logit": (
                candidate_presence_logit
                if isinstance(candidate_presence_logit, torch.Tensor)
                else torch.logit(presence_prob.clamp(EPS, 1.0 - EPS))
            ),
            "candidate_presence_prob": presence_prob,
            "action_logits": action_logits,
            "action_probs": action_probs,
            "action_index": action_index,
            "action_policy": action_policy,
        }

    def _compose_v502_soft_action(
        self,
        c0_prob: torch.Tensor,
        candidate_probs: torch.Tensor,
        edit_logit: torch.Tensor,
        route_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compose the only V502 M2 action distribution.

        The function is deliberately shared by deterministic forward and MC
        aggregation.  It contains no argmax, top-k or threshold, which makes
        train, validation and deployment semantics identical up to M3.
        """
        c0 = _as_b1hw(c0_prob).clamp(EPS, 1.0 - EPS)
        if candidate_probs.ndim != 4:
            raise ValueError(
                f"candidate_probs must be [B,N,H,W], got {tuple(candidate_probs.shape)}"
            )
        if route_logits.shape != candidate_probs.shape:
            raise ValueError(
                "route_logits and candidate_probs must have identical shape; "
                f"got {tuple(route_logits.shape)} vs {tuple(candidate_probs.shape)}"
            )
        edit_logit = _as_b1hw(edit_logit)
        edit_prob = torch.sigmoid(edit_logit)
        route_probs = F.softmax(route_logits, dim=1)
        candidate_mix = (
            route_probs * candidate_probs.clamp(EPS, 1.0 - EPS)
        ).sum(dim=1, keepdim=True)
        m2_probs = (
            c0 + edit_prob * (candidate_mix - c0)
        ).clamp(EPS, 1.0 - EPS)
        action_probs = torch.cat(
            [1.0 - edit_prob, edit_prob * route_probs], dim=1
        )
        action_probs = action_probs / action_probs.sum(
            dim=1, keepdim=True
        ).clamp_min(EPS)
        route_entropy = -(
            route_probs * torch.log(route_probs.clamp_min(EPS))
        ).sum(dim=1, keepdim=True)
        if route_probs.shape[1] > 1:
            top2 = route_probs.topk(2, dim=1).values
            route_margin = top2[:, :1] - top2[:, 1:2]
        else:
            route_margin = torch.ones_like(edit_prob)
        candidate_variance = candidate_probs.var(
            dim=1, keepdim=True, unbiased=False
        )
        return {
            "edit_prob": edit_prob,
            "route_probs": route_probs,
            "candidate_mix": candidate_mix,
            "m2_probs": m2_probs,
            "action_probs": action_probs,
            "route_entropy": route_entropy,
            "route_margin": route_margin,
            "candidate_variance": candidate_variance,
        }

    def _compose_v503_atomic_action(
        self,
        c0_prob: torch.Tensor,
        candidate_probs: torch.Tensor,
        edit_logit: torch.Tensor,
        route_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compose one mutually-exclusive atomic action per pixel.

        Preserve and all typed interventions share one action distribution.
        The forward pass is hard in both training and evaluation; training uses
        a straight-through gradient through the same probability distribution.
        This prevents Delete/Fill or Trim/Expand from being averaged together.
        """
        c0 = _as_b1hw(c0_prob).clamp(EPS, 1.0 - EPS)
        edit_logit = _as_b1hw(edit_logit)
        route_probs = F.softmax(route_logits, dim=1)
        edit_prob = torch.sigmoid(edit_logit)
        action_probs = torch.cat(
            [1.0 - edit_prob, edit_prob * route_probs], dim=1
        ).clamp_min(EPS)
        action_probs = action_probs / action_probs.sum(dim=1, keepdim=True).clamp_min(EPS)
        action_index = action_probs.argmax(dim=1, keepdim=True)
        hard_policy = torch.zeros_like(action_probs).scatter_(1, action_index, 1.0)
        action_policy = (
            hard_policy + action_probs - action_probs.detach()
            if self.training else hard_policy
        )
        candidate_policy = action_policy[:, 1:]
        signed_edits = candidate_probs.clamp(EPS, 1.0 - EPS) - c0
        m2_probs = (
            c0 + (candidate_policy * signed_edits).sum(dim=1, keepdim=True)
        ).clamp(EPS, 1.0 - EPS)
        conditional_mix = (route_probs * candidate_probs).sum(dim=1, keepdim=True)
        route_entropy = -(route_probs * torch.log(route_probs.clamp_min(EPS))).sum(dim=1, keepdim=True)
        if route_probs.shape[1] > 1:
            top2 = route_probs.topk(2, dim=1).values
            route_margin = top2[:, :1] - top2[:, 1:2]
        else:
            route_margin = torch.ones_like(edit_prob)
        return {
            "edit_prob": edit_prob,
            "route_probs": route_probs,
            "candidate_mix": conditional_mix,
            "m2_probs": m2_probs,
            "action_probs": action_probs,
            "action_policy": action_policy,
            "action_index": action_index,
            "route_entropy": route_entropy,
            "route_margin": route_margin,
            "candidate_variance": candidate_probs.var(dim=1, keepdim=True, unbiased=False),
        }

    def aggregate_mc_outputs(
        self,
        runs: list[Dict[str, torch.Tensor]],
        c0_prob: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Aggregate stochastic evidence before making a single route decision.

        V494 averaged already-deployed masks from multiple stochastic passes.
        That silently converted a full-dose binary intervention into a fractional
        dose. V495 instead averages route/utility logits, chooses one sparse route
        once, and constructs one complete proposal. MC disagreement is retained
        separately as epistemic variance for M3.
        """
        if not self.v495_enabled or not self.v495_mc_logit_aggregation:
            raise RuntimeError(
                "aggregate_mc_outputs requires V495 MC-logit aggregation."
            )
        if not runs:
            raise ValueError("V495 MC aggregation requires at least one run.")

        common_keys = set(runs[0])
        for run in runs[1:]:
            common_keys.intersection_update(run)
        aggregated: Dict[str, torch.Tensor] = {}
        for key in common_keys:
            values = [run[key] for run in runs]
            if not isinstance(values[0], torch.Tensor):
                continue
            if all(value.shape == values[0].shape for value in values):
                # MC evidence aggregation is defined for continuous tensors.
                # Discrete indices (for example V500 action_index) are rebuilt
                # after logit aggregation and must never be arithmetically
                # averaged.
                if values[0].is_floating_point() or values[0].is_complex():
                    aggregated[key] = torch.stack(values, dim=0).mean(dim=0)

        required = (
            "m2_route_logits",
            "m2_candidate_utility_map_logits",
            "m2_effective_candidates",
            "m2_effective_supports",
        )
        missing = [key for key in required if key not in aggregated]
        if missing:
            raise RuntimeError("V495 MC aggregation missing tensors: " + str(missing))

        c0 = _as_b1hw(c0_prob).clamp(EPS, 1.0 - EPS)
        route_logits = aggregated["m2_route_logits"]
        route_probs = F.softmax(route_logits, dim=1)
        route_policy = (
            route_probs if self.v502_enabled else self._sparse_route(route_probs)
        )
        candidate_presence_logit = aggregated.get(
            "m2_candidate_presence_logit"
        )
        action_reference = self._action_reference_distribution(
            route_logits,
            route_probs,
            candidate_presence_logit=candidate_presence_logit,
        )
        action_logits = action_reference["action_logits"]
        action_probs = action_reference["action_probs"]
        action_index = action_reference["action_index"]
        action_policy = action_reference["action_policy"]
        effective_nonbase = aggregated["m2_effective_candidates"].clamp(
            EPS, 1.0 - EPS
        )
        if self.v502_enabled:
            if not isinstance(candidate_presence_logit, torch.Tensor):
                raise RuntimeError(
                    "V502 MC aggregation requires m2_candidate_presence_logit."
                )
            composed = (
                self._compose_v503_atomic_action(
                    c0, effective_nonbase, candidate_presence_logit, route_logits
                )
                if self.v503_enabled
                else self._compose_v502_soft_action(
                    c0, effective_nonbase, candidate_presence_logit, route_logits
                )
            )
            route_probs = composed["route_probs"]
            candidate_mix = composed["candidate_mix"]
            m2_probs = composed["m2_probs"]
            action_probs = composed["action_probs"]
            utility_map_logits = aggregated["m2_candidate_utility_map_logits"]
            selected_utility_logit = (
                route_probs.detach() * utility_map_logits
            ).sum(dim=1, keepdim=True)
            effective_supports = aggregated["m2_effective_supports"].clamp(0.0, 1.0)
            structural_support = (
                route_probs.detach() * effective_supports
            ).sum(dim=1, keepdim=True).clamp(0.0, 1.0)
            action_index = action_probs.argmax(dim=1, keepdim=True)
            aggregated.update({
                "m2_benefit_logit": candidate_presence_logit,
                "m2_benefit_residual_logit": torch.zeros_like(candidate_presence_logit),
                "m2_benefit_prob": composed["edit_prob"],
                "m2_selected_utility_logit": selected_utility_logit,
                "m2_amplitude_logit": torch.full_like(candidate_presence_logit, 20.0),
                "m2_amplitude_prob": torch.ones_like(candidate_presence_logit),
                "m2_direct_causal_accept": composed["edit_prob"],
                "m2_edit_gate_logit": candidate_presence_logit,
                "m2_edit_gate_prob": composed["edit_prob"],
                "m2_route_logits": route_logits,
                "m2_route_probs": route_probs,
                "m2_route_policy": route_probs,
                "m2_route_policy_with_preserve": action_probs,
                "m2_action_logits": torch.log(action_probs.clamp_min(EPS)),
                "m2_action_probs": action_probs,
                "m2_action_policy": composed.get("action_policy", action_probs),
                "m2_action_index": composed.get("action_index", action_index),
                "m2_candidate_presence_logit": candidate_presence_logit,
                "m2_candidate_presence_prob": composed["edit_prob"],
                "m2_preserve_reference_prob": action_probs[:, :1],
                "m2_pixel_policy_logits": torch.log(action_probs.clamp_min(EPS)),
                "m2_pixel_weights": action_probs,
                "m2_routed_candidate": candidate_mix,
                "m2_full_proposal_probs": candidate_mix[:, 0],
                "m2_proposal_probs": m2_probs[:, 0],
                "m2_training_probs": m2_probs[:, 0],
                "m2_convex_probs": m2_probs[:, 0],
                "m2_residual_map": torch.zeros_like(m2_probs),
                "m2_fused_probs": m2_probs[:, 0],
                "m2_noop_gate_prob": composed["edit_prob"],
                "m2_noop_gate": composed["edit_prob"],
                "m2_single_decision_enabled": m2_probs.new_ones((m2_probs.shape[0],)),
                "m2_mc_logit_aggregation_enabled": m2_probs.new_ones((m2_probs.shape[0],)),
                "m2_structural_support": structural_support,
                "m2_route_entropy": composed["route_entropy"],
                "m2_route_margin": composed["route_margin"],
                "m2_candidate_variance": composed["candidate_variance"],
                "m2_soft_router_enabled": m2_probs.new_full(
                    (m2_probs.shape[0],), float(not self.v503_enabled)
                ),
                "m2_atomic_policy_enabled": m2_probs.new_full(
                    (m2_probs.shape[0],), float(self.v503_enabled)
                ),
            })
            return aggregated

        full_proposal = (
            route_policy * effective_nonbase
        ).sum(dim=1, keepdim=True).clamp(EPS, 1.0 - EPS)

        utility_map_logits = aggregated["m2_candidate_utility_map_logits"]
        selected_utility_logit = (
            route_policy.detach() * utility_map_logits
        ).sum(dim=1, keepdim=True)
        benefit_prob = torch.sigmoid(selected_utility_logit)
        benefit_scaled = (
            c0 + benefit_prob * (full_proposal - c0)
        ).clamp(EPS, 1.0 - EPS)
        presence_prob = action_reference["candidate_presence_prob"]
        if self.v501_enabled:
            hard_gate = (presence_prob >= self.v501_gate_threshold).to(c0.dtype)
            deploy_gate = (
                hard_gate + presence_prob - presence_prob.detach()
                if self.training else hard_gate
            )
            gated_proposal = (
                c0 + deploy_gate * (full_proposal - c0)
            ).clamp(EPS, 1.0 - EPS)
            training_probs = gated_proposal
            deployed_probs = gated_proposal
        else:
            deploy_gate = benefit_prob
            training_probs = full_proposal if self.v498_enabled else benefit_scaled
            deployed_probs = full_proposal
        effective_supports = aggregated["m2_effective_supports"].clamp(0.0, 1.0)
        structural_support = (
            route_policy.detach() * effective_supports
        ).sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        pixel_weights = (
            action_probs if self.v498_enabled
            else torch.cat(
                [1.0 - benefit_prob, benefit_prob * route_policy], dim=1
            )
        )

        aggregated.update({
            "m2_benefit_logit": selected_utility_logit,
            "m2_benefit_residual_logit": torch.zeros_like(selected_utility_logit),
            "m2_benefit_prob": benefit_prob,
            "m2_selected_utility_logit": selected_utility_logit,
            "m2_amplitude_logit": torch.full_like(selected_utility_logit, 20.0),
            "m2_amplitude_prob": torch.ones_like(selected_utility_logit),
            "m2_direct_causal_accept": deploy_gate,
            "m2_edit_gate_logit": (
                action_reference["candidate_presence_logit"]
                if self.v501_enabled else selected_utility_logit
            ),
            "m2_edit_gate_prob": (
                presence_prob if self.v501_enabled else benefit_prob
            ),
            "m2_route_logits": route_logits,
            "m2_route_probs": route_probs,
            "m2_route_policy": route_policy,
            "m2_action_logits": action_logits,
            "m2_action_probs": action_probs,
            "m2_action_policy": action_policy,
            "m2_action_index": action_index,
            "m2_candidate_presence_logit": action_reference[
                "candidate_presence_logit"
            ],
            "m2_candidate_presence_prob": action_reference[
                "candidate_presence_prob"
            ],
            "m2_preserve_reference_prob": action_probs[:, :1],
            "m2_pixel_policy_logits": torch.log(pixel_weights.clamp_min(EPS)),
            "m2_pixel_weights": pixel_weights,
            "m2_routed_candidate": full_proposal,
            "m2_full_proposal_probs": full_proposal[:, 0],
            "m2_proposal_probs": deployed_probs[:, 0],
            "m2_training_probs": training_probs[:, 0],
            "m2_convex_probs": training_probs[:, 0],
            "m2_residual_map": torch.zeros_like(full_proposal),
            "m2_fused_probs": deployed_probs[:, 0],
            "m2_noop_gate_prob": presence_prob,
            "m2_noop_gate": deploy_gate,
            "m2_single_decision_enabled": full_proposal.new_ones((full_proposal.shape[0],)),
            "m2_mc_logit_aggregation_enabled": full_proposal.new_ones((full_proposal.shape[0],)),
            "m2_structural_support": structural_support,
        })
        return aggregated

    def forward(
        self,
        image: torch.Tensor,
        c0_prob: torch.Tensor,
        candidate_probs: torch.Tensor,
        supports: torch.Tensor,
        error_state_probs: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        stochastic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        c0_prob = _as_b1hw(c0_prob).clamp(EPS, 1.0 - EPS)
        if candidate_probs.ndim != 4:
            raise ValueError(
                f"candidate_probs must be [B,K,H,W], got {tuple(candidate_probs.shape)}"
            )
        b, k, h, w = candidate_probs.shape
        if k < 2:
            raise ValueError("V492 requires Preserve plus at least one intervention.")
        if k > self.max_candidates:
            raise ValueError(
                f"V492 supports at most {self.max_candidates} candidates, got {k}"
            )

        raw_nonbase = candidate_probs[:, 1:]
        n = raw_nonbase.shape[1]
        support = self._resize_supports(supports, candidate_probs, c0_prob)
        c0_expand = c0_prob.expand(-1, n, -1, -1)
        raw_signed = raw_nonbase - c0_expand

        # Locality contract: the M1 generator has already multiplied every
        # intervention by its learned support.  M2 therefore keeps the candidate
        # value unchanged (avoiding an erroneous second support multiplication)
        # and uses detached support only as evidence for benefit/amplitude.
        support_geometry = support.detach()
        effective_signed = raw_signed
        effective_nonbase = raw_nonbase.clamp(EPS, 1.0 - EPS)
        abs_edit = effective_signed.abs()

        ent = _entropy(c0_prob)
        bnd = _soft_boundary(c0_prob)
        gray, edge = _gray_edge(image, (h, w))
        state = error_state_probs[:, :, None, None].expand(-1, -1, h, w)
        max_abs_edit = abs_edit.max(dim=1, keepdim=True).values
        mean_abs_edit = abs_edit.mean(dim=1, keepdim=True)
        support_union = support_geometry.max(dim=1, keepdim=True).values
        shared_input = torch.cat(
            [
                c0_prob,
                1.0 - c0_prob,
                ent,
                bnd,
                gray,
                edge,
                state,
                max_abs_edit,
                mean_abs_edit,
                support_union,
            ],
            dim=1,
        )
        stem = self.context_stem(shared_input)
        shared = self._multiscale_context(stem)
        shared = shared + self._semantic_image_map(
            image_features, (h, w), c0_prob
        )
        shared = shared + self._text_context(text_features, c0_prob)

        route_input = torch.stack(
            [effective_nonbase, effective_signed, abs_edit, support_geometry],
            dim=2,
        ).reshape(b * n, 4, h, w)
        route_feat = self.candidate_encoder(route_input)
        shared_rep = shared[:, None].expand(
            -1, n, -1, -1, -1
        ).reshape(b * n, self.hidden_dim, h, w)
        type_ids = torch.arange(n, device=candidate_probs.device).clamp_max(
            self.route_type_embedding.num_embeddings - 1
        )
        type_bias = self.route_type_embedding(type_ids).to(candidate_probs.dtype)
        type_bias = type_bias[None].expand(b, -1, -1).reshape(
            b * n, self.hidden_dim, 1, 1
        )
        route_feat = self.route_decoder(route_feat + shared_rep + type_bias)
        route_feat = F.dropout2d(
            route_feat,
            p=self.dropout,
            training=bool(self.training or stochastic),
        )
        pixel_route_logits = self.route_head(route_feat).reshape(b, n, h, w)
        route_feature_bank = route_feat.reshape(
            b, n, self.hidden_dim, h, w
        )
        candidate_utility_logits = self.candidate_utility_head(route_feat).reshape(
            b, n, h, w
        )
        utility_logits = self.utility_head(route_feat).reshape(b, n)
        route_logits = (
            pixel_route_logits
            + self.utility_weight * utility_logits[:, :, None, None]
            + (
                self.pixel_utility_weight * candidate_utility_logits
                if self.v493_enabled
                else 0.0
            )
        ) / self.route_temperature
        # V502 uses candidate-only soft routing.  Preserve is represented only
        # by the independent eligibility probability and is never inserted as a
        # second, conflicting route decision.  Legacy versions retain their
        # historical sparse route for reproducibility.
        candidate_route_probs = F.softmax(route_logits, dim=1)
        candidate_route_policy = (
            candidate_route_probs
            if self.v502_enabled
            else self._sparse_route(candidate_route_probs)
        )

        presence_context = (
            shared + route_feature_bank.mean(dim=1)
            if self.v502_enabled else shared
        )
        candidate_presence_logit = (
            self.candidate_presence_head(presence_context)
            if self.candidate_presence_head is not None
            else None
        )
        action_reference = self._action_reference_distribution(
            route_logits,
            candidate_route_probs,
            candidate_presence_logit=candidate_presence_logit,
        )
        action_logits = action_reference["action_logits"]
        action_probs = action_reference["action_probs"]
        action_index = action_reference["action_index"]
        action_policy = action_reference["action_policy"]

        routed_candidate = (
            candidate_route_policy * effective_nonbase
        ).sum(dim=1, keepdim=True)
        # Compatibility/audit view only.  Preserve is represented by the
        # hierarchical action distribution, never by a second route softmax.
        route_policy_with_preserve = action_probs
        routed_support = (
            candidate_route_policy.detach() * support_geometry
        ).sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        route_context = (
            candidate_route_probs[:, :, None] * route_feature_bank
        ).sum(dim=1)
        local_context = shared + route_context

        # Benefit is conditioned on the intervention that will actually be
        # deployed. Route is detached only for this conditioning operation so
        # Benefit cannot manipulate Route; Route has its own exact supervision.
        selected_utility_logit = (
            candidate_route_policy.detach() * candidate_utility_logits
        ).sum(dim=1, keepdim=True)

        if self.v494_enabled:
            # The selected candidate's utility is retained as calibrated causal
            # evidence. V494 used it as an early hard gate. V495 removes that
            # irreversible decision: M2 always exposes the complete selected
            # proposal, while Benefit remains continuous evidence for training
            # and for the final Preserve-vs-proposal risk decision in M3.
            benefit_residual_logit = torch.zeros_like(selected_utility_logit)
            benefit_logit = selected_utility_logit
            benefit_prob = torch.sigmoid(benefit_logit)
            if self.v495_enabled:
                deployed_benefit = benefit_prob
            else:
                hard_benefit = (benefit_prob >= 0.5).to(benefit_prob.dtype)
                deployed_benefit = (
                    hard_benefit + benefit_prob - benefit_prob.detach()
                    if self.training
                    else hard_benefit
                )
            amplitude_prob = torch.ones_like(benefit_prob)
            amplitude_logit = torch.full_like(benefit_prob, 20.0)
            edit_gate = deployed_benefit
        else:
            benefit_feat = self.benefit_decoder(local_context)
            amplitude_feat = self.amplitude_decoder(local_context)
            benefit_residual_logit = self.benefit_head(benefit_feat)
            benefit_logit = benefit_residual_logit + (
                self.selected_utility_weight * selected_utility_logit
                if self.v493_enabled
                else 0.0
            )
            amplitude_logit = self.amplitude_head(amplitude_feat)
            benefit_prob = torch.sigmoid(benefit_logit)
            amplitude_prob = torch.sigmoid(amplitude_logit)
            edit_gate = (benefit_prob * amplitude_prob).clamp(0.0, 1.0)

        # ``proposal`` is the complete hard top-1 intervention.  V495
        # introduced a Benefit-scaled training surrogate, but that created the
        # observed mismatch: losses saw a discounted edit while validation and
        # M3 saw the full edit.  V498 uses exactly the same full proposal for
        # training and deployment.  Benefit remains calibrated evidence only.
        full_proposal = routed_candidate.clamp(EPS, 1.0 - EPS)
        benefit_scaled = (
            c0_prob + edit_gate * (full_proposal - c0_prob)
        ).clamp(EPS, 1.0 - EPS)
        presence_prob = action_reference["candidate_presence_prob"]
        if self.v502_enabled:
            composed = (
                self._compose_v503_atomic_action(
                    c0_prob,
                    effective_nonbase,
                    action_reference["candidate_presence_logit"],
                    route_logits,
                )
                if self.v503_enabled
                else self._compose_v502_soft_action(
                    c0_prob,
                    effective_nonbase,
                    action_reference["candidate_presence_logit"],
                    route_logits,
                )
            )
            # Reuse the exact shared composition contract.  ``full_proposal``
            # remains the conditional candidate mixture for M3 diagnostics,
            # while every deployed/training M2 output is the Base-anchored soft
            # action.
            candidate_route_probs = composed["route_probs"]
            candidate_route_policy = candidate_route_probs
            full_proposal = composed["candidate_mix"]
            training_probs = composed["m2_probs"]
            deployed_m2 = composed["m2_probs"]
            deploy_noop_gate = composed["edit_prob"]
            effective_edit_gate = composed["edit_prob"]
            effective_edit_logit = action_reference["candidate_presence_logit"]
            action_probs = composed["action_probs"]
            action_logits = torch.log(action_probs.clamp_min(EPS))
            action_policy = composed.get("action_policy", action_probs)
            action_index = composed.get(
                "action_index", action_probs.argmax(dim=1, keepdim=True)
            )
            route_policy_with_preserve = action_probs
        elif self.v501_enabled:
            hard_noop_gate = (
                presence_prob >= self.v501_gate_threshold
            ).to(c0_prob.dtype)
            deploy_noop_gate = (
                hard_noop_gate + presence_prob - presence_prob.detach()
                if self.training else hard_noop_gate
            )
            gated_proposal = (
                c0_prob + deploy_noop_gate * (full_proposal - c0_prob)
            ).clamp(EPS, 1.0 - EPS)
            training_probs = gated_proposal
            deployed_m2 = gated_proposal
            effective_edit_gate = presence_prob
            effective_edit_logit = action_reference["candidate_presence_logit"]
        else:
            deploy_noop_gate = edit_gate
            training_probs = full_proposal if self.v498_enabled else benefit_scaled
            deployed_m2 = full_proposal if self.v495_enabled else benefit_scaled
            effective_edit_gate = edit_gate
            effective_edit_logit = benefit_logit
        residual = torch.zeros_like(full_proposal)
        pixel_weights = (
            action_probs if (self.v498_enabled or self.v502_enabled)
            else torch.cat(
                [1.0 - edit_gate, edit_gate * candidate_route_policy], dim=1
            )
        )
        if self.v502_enabled:
            route_entropy = composed["route_entropy"]
            route_margin = composed["route_margin"]
            candidate_variance = composed["candidate_variance"]
        else:
            route_entropy = -(
                candidate_route_probs
                * torch.log(candidate_route_probs.clamp_min(EPS))
            ).sum(dim=1, keepdim=True)
            if candidate_route_probs.shape[1] > 1:
                top2 = candidate_route_probs.topk(2, dim=1).values
                route_margin = top2[:, :1] - top2[:, 1:2]
            else:
                route_margin = torch.ones_like(effective_edit_gate)
            candidate_variance = effective_nonbase.var(
                dim=1, keepdim=True, unbiased=False
            )

        return {
            "m2_benefit_logit": benefit_logit,
            "m2_benefit_residual_logit": benefit_residual_logit,
            "m2_benefit_prob": benefit_prob,
            "m2_candidate_utility_map_logits": candidate_utility_logits,
            "m2_candidate_utility_map_prob": torch.sigmoid(candidate_utility_logits),
            "m2_selected_utility_logit": selected_utility_logit,
            "m2_amplitude_logit": amplitude_logit,
            "m2_amplitude_prob": amplitude_prob,
            "m2_direct_causal_accept": deploy_noop_gate,
            "m2_structural_support": routed_support,
            "m2_effective_candidates": effective_nonbase,
            "m2_effective_supports": support_geometry,
            "m2_routed_candidate": routed_candidate,
            "m2_full_proposal_probs": full_proposal[:, 0],
            "m2_proposal_probs": deployed_m2[:, 0],
            "m2_training_probs": training_probs[:, 0],
            "m2_single_decision_enabled": full_proposal.new_full(
                (full_proposal.shape[0],), float(self.v495_enabled)
            ),
            "m2_mc_logit_aggregation_enabled": full_proposal.new_zeros(
                (full_proposal.shape[0],)
            ),
            "m2_candidate_utility_logits": utility_logits,
            "m2_edit_gate_logit": effective_edit_logit,
            "m2_edit_gate_prob": effective_edit_gate,
            "m2_route_logits": route_logits,
            "m2_route_probs": candidate_route_probs,
            "m2_route_policy": candidate_route_policy,
            "m2_route_policy_with_preserve": route_policy_with_preserve,
            "m2_route_raw_logits": pixel_route_logits,
            "m2_action_logits": action_logits,
            "m2_action_probs": action_probs,
            "m2_action_policy": action_policy,
            "m2_action_index": action_index,
            "m2_candidate_presence_logit": action_reference[
                "candidate_presence_logit"
            ],
            "m2_candidate_presence_prob": action_reference[
                "candidate_presence_prob"
            ],
            "m2_preserve_reference_prob": action_probs[:, :1],
            "m2_pixel_policy_logits": torch.log(pixel_weights.clamp_min(EPS)),
            "m2_pixel_weights": pixel_weights,
            "m2_candidate_disagreement": max_abs_edit,
            "m2_support_union": support_union,
            "m2_convex_probs": training_probs[:, 0],
            "m2_residual_map": residual,
            "m2_fused_probs": deployed_m2[:, 0],
            "m2_noop_gate_prob": effective_edit_gate,
            "m2_noop_gate": deploy_noop_gate,
            "m2_route_entropy": route_entropy,
            "m2_route_margin": route_margin,
            "m2_candidate_variance": candidate_variance,
            "m2_soft_router_enabled": full_proposal.new_full(
                (full_proposal.shape[0],), float(self.v502_enabled and not self.v503_enabled)
            ),
            "m2_atomic_policy_enabled": full_proposal.new_full(
                (full_proposal.shape[0],), float(self.v503_enabled)
            ),
        }


class RegionBestExpertSelector(nn.Module):
    """V489 M3: region-wise best-expert selector.

    Experts are Preserve/C0, all M1 candidates, and the M2 composition.  The
    selector predicts one utility logit per expert on a low-resolution region
    grid.  Training uses soft region weights for stable end-to-end gradients;
    inference uses hard argmax.  The early M2-expert margin is annealed rather
    than freezing M2 or detaching the M2→M3 path.
    """

    def __init__(self, cfg: Any, hidden_dim: int = 48) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_experts = int(_m1(cfg, "V489_M3_MAX_EXPERTS", 9))
        self.region_scale = max(1, int(_m1(cfg, "V489_M3_REGION_SCALE", 8)))
        self.dropout = float(_m1(cfg, "V489_M3_DROPOUT", 0.10))
        self.hard_inference = bool(_m1(cfg, "V489_M3_HARD_INFERENCE", True))
        self.straight_through_train = bool(_m1(cfg, "V489_M3_STRAIGHT_THROUGH_TRAIN", False))
        self.initial_m2_margin = max(float(_m1(cfg, "V489_M3_M2_INITIAL_MARGIN", 2.0)), 0.0)
        self.margin_release_epochs = max(1, int(_m1(cfg, "V489_M3_M2_MARGIN_RELEASE_EPOCHS", 10)))
        self.min_temperature = max(float(_m1(cfg, "V489_M3_MIN_TEMPERATURE", 0.20)), 1.0e-3)
        initial_temperature = max(float(_m1(cfg, "V489_M3_INITIAL_TEMPERATURE", 1.0)), self.min_temperature)
        self.raw_temperature = nn.Parameter(
            torch.log(torch.expm1(torch.tensor(initial_temperature - self.min_temperature + 1.0e-4)))
        )
        self.current_epoch = 0

        groups = min(8, self.hidden_dim)
        while self.hidden_dim % groups != 0 and groups > 1:
            groups -= 1
        self.expert_encoder = nn.Sequential(
            nn.Conv2d(8, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
        )
        self.type_embedding = nn.Embedding(self.max_experts, self.hidden_dim)
        self.expert_bias = nn.Parameter(torch.zeros(self.max_experts))
        self.region_head = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
            nn.Dropout2d(self.dropout),
            nn.Conv2d(self.hidden_dim, 1, 1),
        )
        nn.init.zeros_(self.type_embedding.weight)
        nn.init.zeros_(self.region_head[-1].weight)
        nn.init.zeros_(self.region_head[-1].bias)
        # Preserve has a modest initial advantage; it is not forced after training.
        with torch.no_grad():
            self.expert_bias.zero_()
            self.expert_bias[0] = 0.50

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def _m2_margin(self) -> float:
        if not self.training:
            return 0.0
        progress = min(1.0, max(0.0, float(self.current_epoch) / float(self.margin_release_epochs)))
        return self.initial_m2_margin * (1.0 - progress)

    def forward(
        self,
        c0_prob: torch.Tensor,
        candidate_probs: torch.Tensor,
        m2_prob: torch.Tensor,
        m2_variance: torch.Tensor,
        m2_edit_gate: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        c0 = _as_b1hw(c0_prob).clamp(EPS, 1.0 - EPS)
        if candidate_probs.ndim != 4:
            raise ValueError("candidate_probs must be [B,K,H,W].")
        m2 = _as_b1hw(m2_prob).clamp(EPS, 1.0 - EPS)
        variance = _as_b1hw(m2_variance).clamp_min(0.0)
        edit_gate = _as_b1hw(m2_edit_gate).clamp(0.0, 1.0)
        experts = torch.cat([candidate_probs, m2], dim=1)
        b, e, h, w = experts.shape
        if e > self.max_experts:
            raise ValueError(f"V489 M3 supports at most {self.max_experts} experts, got {e}")

        c0e = c0.expand(-1, e, -1, -1)
        signed = experts - c0e
        abs_edit = signed.abs()
        entropy = _entropy(experts)
        boundary = _soft_boundary(experts.reshape(b * e, 1, h, w)).reshape(b, e, h, w)
        var_e = variance.expand(-1, e, -1, -1)
        gate_e = edit_gate.expand(-1, e, -1, -1)
        x = torch.stack(
            [experts, c0e, signed, abs_edit, entropy, boundary, var_e, gate_e], dim=2
        ).reshape(b * e, 8, h, w)
        feat = self.expert_encoder(x)
        type_ids = torch.arange(e, device=experts.device).clamp_max(self.max_experts - 1)
        type_bias = self.type_embedding(type_ids).to(dtype=experts.dtype)
        type_bias = type_bias[None].expand(b, -1, -1).reshape(b * e, self.hidden_dim, 1, 1)
        feat = feat + type_bias
        region_feat = F.avg_pool2d(
            feat,
            kernel_size=self.region_scale,
            stride=self.region_scale,
            ceil_mode=True,
        )
        region_logits = self.region_head(region_feat).reshape(
            b, e, region_feat.shape[-2], region_feat.shape[-1]
        )
        region_logits = region_logits + self.expert_bias[:e].to(region_logits.dtype)[None, :, None, None]

        m2_margin = self._m2_margin()
        if m2_margin > 0.0:
            # M2 is always the last expert by construction.
            margin_tensor = region_logits.new_zeros(region_logits.shape)
            margin_tensor[:, -1:] = float(m2_margin)
            region_logits = region_logits - margin_tensor

        temperature = F.softplus(self.raw_temperature) + self.min_temperature
        region_probs = F.softmax(region_logits / temperature, dim=1)
        pixel_logits = F.interpolate(region_logits, size=(h, w), mode="bilinear", align_corners=False)
        pixel_probs = F.softmax(pixel_logits / temperature, dim=1)
        selected_index = pixel_logits.argmax(dim=1)
        hard_weights = F.one_hot(selected_index, num_classes=e).permute(0, 3, 1, 2).to(experts.dtype)

        if self.training:
            if self.straight_through_train:
                deploy_weights = hard_weights.detach() - pixel_probs.detach() + pixel_probs
            else:
                deploy_weights = pixel_probs
        elif self.hard_inference:
            deploy_weights = hard_weights
        else:
            deploy_weights = pixel_probs

        final = (deploy_weights * experts).sum(dim=1).clamp(EPS, 1.0 - EPS)
        selection_rate = hard_weights.flatten(2).mean(dim=-1)
        nonbase_prob = 1.0 - pixel_probs[:, :1]
        nonbase_hard = (selected_index != 0).to(experts.dtype)[:, None]
        return {
            "m3_expert_probs": experts,
            "m3_region_logits": region_logits,
            "m3_region_probs": region_probs,
            "m3_pixel_logits": pixel_logits,
            "m3_pixel_weights": pixel_probs,
            "m3_deploy_weights": deploy_weights,
            "m3_selected_index": selected_index,
            "m3_expert_selection_rate": selection_rate,
            "m3_temperature": temperature.expand(b),
            "m3_m2_margin": experts.new_full((b,), float(m2_margin)),
            # Compatibility names used by existing validation/evaluation code.
            "m3_gate_prob": nonbase_prob,
            "m3_hard_gate": nonbase_hard,
            "m3_deploy_gate": nonbase_prob,
            "fused_probs": final,
            "final_probs": final,
        }


class PreserveFirstBinaryRiskDeployer(nn.Module):
    """V491 M3: a role-separated Preserve-vs-M2 safety verifier.

    M1 generates counterfactual candidates and M2 composes them.  M3 must not
    repeat M2's routing decision by selecting raw candidates again.  It predicts
    only the relative risk of the already-composed M2 proposal against factual
    Preserve/C0 and accepts M2 only when its one-standard-deviation upper risk
    bound is below Preserve's exact zero relative risk.

    The final output is therefore an exact no-op at initialization and outside
    genuine M2 edits.  During training the hard forward decision is paired with
    a straight-through soft gate, so train/validation/test use the same deployed
    rule while the risk head remains differentiable.
    """

    def __init__(self, cfg: Any, hidden_dim: int = 48) -> None:
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = int(hidden_dim)
        self.v493_potential_relevance = bool(
            _m1(cfg, "V493_M3_POTENTIAL_RELEVANCE_ENABLED", False)
        )
        self.v494_expected_risk_deployment = bool(
            _m1(cfg, "V494_M3_EXPECTED_RISK_DEPLOYMENT", False)
        )
        self.v495_enabled = bool(
            _m1(cfg, "V495_SINGLE_DECISION_ENABLED", False)
        ) and bool(_m1(cfg, "V494_DIRECT_CAUSAL_DOSE_ENABLED", False))
        self.dropout = float(_m1(cfg, "V491_M3_DROPOUT", 0.10))
        self.min_temperature = max(
            float(_m1(cfg, "V491_M3_MIN_TEMPERATURE", 0.10)), 1.0e-3
        )
        initial_temperature = max(
            float(_m1(cfg, "V491_M3_INITIAL_TEMPERATURE", 0.50)),
            self.min_temperature,
        )
        initial_raw = max(initial_temperature - self.min_temperature, 1.0e-4)
        self.raw_temperature = nn.Parameter(
            torch.log(torch.expm1(torch.tensor(initial_raw)))
        )
        self.ucb_std_weight = max(
            float(_m1(cfg, "V491_M3_UCB_STD_WEIGHT", 1.0)), 0.0
        )
        initial_risk_bias = float(
            _m1(cfg, "V491_M3_INITIAL_RELATIVE_RISK", 0.25)
        )
        initial_log_variance = float(
            _m1(cfg, "V491_M3_INITIAL_LOG_VARIANCE", -2.0)
        )

        groups = min(8, self.hidden_dim)
        while self.hidden_dim % groups != 0 and groups > 1:
            groups -= 1

        # factual probability, M2 probability, signed/absolute edit, factual and
        # proposal entropy/boundary, MC variance, M2 edit gate and M1 support.
        m3_input_channels = 12 if self.v495_enabled else 11
        self.encoder = nn.Sequential(
            nn.Conv2d(m3_input_channels, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
            nn.Dropout2d(self.dropout),
        )
        self.risk_mean_head = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, 1, 1),
        )
        self.risk_log_variance_head = nn.Conv2d(self.hidden_dim, 1, 1)

        # Preserve-first initialization: the proposal starts with positive
        # relative risk, hence the deployed output is exactly C0 before learning.
        nn.init.zeros_(self.risk_mean_head[-1].weight)
        nn.init.constant_(self.risk_mean_head[-1].bias, initial_risk_bias)
        nn.init.zeros_(self.risk_log_variance_head.weight)
        nn.init.constant_(
            self.risk_log_variance_head.bias, initial_log_variance
        )

    @staticmethod
    def _support_union(
        candidate_supports: torch.Tensor,
        size: Tuple[int, int],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        supports = candidate_supports.to(
            device=reference.device, dtype=reference.dtype
        )
        if supports.ndim == 3:
            supports = supports[:, None]
        if supports.ndim != 4:
            return torch.zeros_like(reference)
        if supports.shape[-2:] != size:
            supports = F.interpolate(
                supports, size=size, mode="bilinear", align_corners=False
            )
        # V495/V496 passes one channel: the support of the exact top-1
        # proposal selected by M2.  Older paths pass [Preserve, C1, ...], in
        # which case slot zero is removed before taking the union.
        if supports.shape[1] == 1:
            return supports.clamp(0.0, 1.0)
        if supports.shape[1] > 1:
            supports = supports[:, 1:]
        return supports.amax(dim=1, keepdim=True).clamp(0.0, 1.0)

    def forward(
        self,
        c0_prob: torch.Tensor,
        candidate_probs: torch.Tensor,
        candidate_supports: torch.Tensor,
        m2_prob: torch.Tensor,
        m2_variance: torch.Tensor,
        m2_edit_gate: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        c0 = _as_b1hw(c0_prob).clamp(EPS, 1.0 - EPS)
        m2 = _as_b1hw(m2_prob).clamp(EPS, 1.0 - EPS)
        variance = _as_b1hw(m2_variance).clamp_min(0.0)
        edit_gate = _as_b1hw(m2_edit_gate).clamp(0.0, 1.0)
        if c0.shape != m2.shape:
            raise ValueError(
                f"V491 M3 requires matching C0/M2 shapes, got "
                f"{tuple(c0.shape)} and {tuple(m2.shape)}"
            )

        height, width = c0.shape[-2:]
        signed_edit = m2 - c0
        abs_edit = signed_edit.abs()
        support_union = self._support_union(
            candidate_supports, (height, width), c0
        )
        factual_entropy = _entropy(c0)
        proposal_entropy = _entropy(m2)
        factual_boundary = _soft_boundary(c0)
        proposal_boundary = _soft_boundary(m2)
        boundary_change = (proposal_boundary - factual_boundary).abs()

        evidence_channels = (
            [edit_gate, support_union]
            if self.v495_enabled
            else [torch.maximum(edit_gate, support_union)]
        )
        model_input = torch.cat(
            [
                c0,
                m2,
                signed_edit,
                abs_edit,
                factual_entropy,
                proposal_entropy,
                factual_boundary,
                proposal_boundary,
                boundary_change,
                variance,
                *evidence_channels,
            ],
            dim=1,
        )
        feature = self.encoder(model_input)
        relative_risk_mean = self.risk_mean_head(feature)
        relative_log_variance = self.risk_log_variance_head(feature).clamp(
            -6.0, 4.0
        )
        relative_risk_std = torch.sqrt(
            F.softplus(relative_log_variance) + EPS
        )
        upper_confidence_risk = (
            relative_risk_mean
            + self.ucb_std_weight * relative_risk_std
        )
        # V494 uses the Bayes action for relative risk: choose M2 exactly when
        # its posterior expected loss is lower than Preserve's zero risk.  The
        # heteroscedastic variance is still trained and reported for calibration,
        # but is no longer added as a fixed one-standard-deviation penalty that
        # caused the V493 soft/hard deployment mismatch and near-total rollback.
        deployment_risk = (
            relative_risk_mean
            if self.v494_expected_risk_deployment
            else upper_confidence_risk
        )

        # V493 expands risk supervision to the full *potential* intervention
        # support, not only pixels already opened by a conservative M2 gate.
        # This avoids the V492 feedback loop: tiny M2 recall -> almost no M3
        # supervision -> permanent rollback. Geometry is detached and cannot be
        # manipulated to reduce the risk loss.
        edit_scale = abs_edit.flatten(1).amax(dim=1).view(-1, 1, 1, 1)
        actual_edit_availability = (
            abs_edit / edit_scale.clamp_min(EPS)
        ).detach().clamp(0.0, 1.0)
        candidate_delta = (
            candidate_probs[:, 1:] - c0
        ).abs().amax(dim=1, keepdim=True)
        potential_scale = candidate_delta.flatten(1).amax(dim=1).view(
            -1, 1, 1, 1
        )
        potential_availability = (
            candidate_delta / potential_scale.clamp_min(EPS)
        ).detach().clamp(0.0, 1.0) * support_union.detach()
        # V495 receives the complete selected proposal, so its actual edit
        # geometry is no longer starved by an upstream gate. Risk supervision is
        # therefore aligned to the selected intervention instead of the union of
        # unrelated candidates. Older V493/V494 configurations keep the fallback.
        edit_availability = (
            actual_edit_availability
            if self.v495_enabled
            else (
                torch.maximum(actual_edit_availability, potential_availability)
                if self.v493_potential_relevance else actual_edit_availability
            )
        )
        has_real_edit = abs_edit.detach() > EPS

        temperature = F.softplus(self.raw_temperature) + self.min_temperature
        accept_probability = torch.sigmoid(-deployment_risk / temperature)
        soft_accept = accept_probability * edit_availability
        hard_accept = (
            (deployment_risk < 0.0) & has_real_edit
        ).to(c0.dtype)
        deploy_accept = (
            hard_accept + soft_accept - soft_accept.detach()
            if self.training
            else hard_accept
        )

        final = (
            c0 + deploy_accept * (m2 - c0)
        ).clamp(EPS, 1.0 - EPS)
        experts = torch.cat([c0, m2], dim=1)
        soft_weights = torch.cat([1.0 - soft_accept, soft_accept], dim=1)
        deploy_weights = torch.cat(
            [1.0 - deploy_accept, deploy_accept], dim=1
        )
        selected_index = hard_accept[:, 0].long()
        selection_rate = torch.stack(
            [
                (selected_index == 0).flatten(1).float().mean(dim=1),
                (selected_index == 1).flatten(1).float().mean(dim=1),
            ],
            dim=1,
        ).to(c0.dtype)

        zero_risk = torch.zeros_like(relative_risk_mean)
        preserve_logvar = torch.full_like(relative_log_variance, -6.0)
        predicted_risk = torch.cat(
            [zero_risk, relative_risk_mean], dim=1
        )
        decision_risk = torch.cat(
            [zero_risk, deployment_risk], dim=1
        )
        predicted_log_variance = torch.cat(
            [preserve_logvar, relative_log_variance], dim=1
        )

        return {
            "m3_expert_probs": experts,
            "m3_predicted_relative_risk_map": predicted_risk,
            "m3_predicted_risk_map": predicted_risk,
            "m3_aggregated_risk_map": predicted_risk,
            "m3_normalized_risk_map": decision_risk,
            "m3_predicted_log_variance_map": predicted_log_variance,
            "m3_predicted_risk_std": relative_risk_std,
            "m3_upper_confidence_risk": upper_confidence_risk,
            "m3_deployment_risk": deployment_risk,
            "m3_accept_probability": accept_probability,
            "m3_risk_scale": torch.ones_like(relative_risk_mean),
            "m3_edit_relevance": edit_availability,
            "m3_actual_edit_relevance": actual_edit_availability,
            "m3_potential_edit_relevance": potential_availability,
            "m3_intervention_mask": edit_availability,
            "m3_context_mass": edit_availability,
            "m3_pixel_weights": soft_weights,
            "m3_deploy_weights": deploy_weights,
            "m3_selected_index": selected_index,
            "m3_expert_selection_rate": selection_rate,
            "m3_temperature": temperature.expand(c0.shape[0]),
            "m3_m2_margin": deployment_risk.flatten(1).mean(dim=1),
            "m3_gate_prob": soft_accept,
            "m3_hard_gate": hard_accept,
            "m3_deploy_gate": deploy_accept,
            "m3_support_union": support_union,
            "m3_selected_support": support_union,
            "fused_probs": final[:, 0],
            "final_probs": final[:, 0],
        }


class DenseCoherentRelativeRiskSelector(nn.Module):
    """V490.4 M3: dense multi-scale scale-invariant relative-risk selection.

    This selector removes the unstable connected-component routing used by
    V490/V490.1.  The earlier implementation binarised candidate probabilities
    at 0.5, split the resulting threshold noise into connected components, and
    truncated the component list.  At random task initialisation this produced
    more than one thousand fragments per image and discarded over half of the
    intervention area.

    V490.2 keeps the same expert evidence encoder, but predicts a dense risk map
    for every expert and centres all risks on Preserve/C0.  Candidate edit
    magnitude supplies a continuous, stop-gradient relevance field used only to
    aggregate local context at several spatial scales.  No connected components,
    region cap, or hard intervention threshold are used.

    Forward deployment is a hard expert choice at every pixel.  During training
    the same hard choice is used in the forward pass with a straight-through
    soft-risk gradient, so training, validation, and test execute the same
    decision rule.
    """

    def __init__(self, cfg: Any, hidden_dim: int = 48) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_experts = int(_m1(cfg, "V490_M3_MAX_EXPERTS", 9))
        self.dropout = float(_m1(cfg, "V490_M3_DROPOUT", 0.10))
        self.scale_invariant_risk = bool(
            _m1(cfg, "V490_SCALE_INVARIANT_RISK_ENABLED", False)
        )
        self.risk_scale_floor = max(
            float(_m1(cfg, "V490_M3_RISK_SCALE_FLOOR", 1.0e-3)), EPS
        )

        raw_scales = _m1(cfg, "V490_M3_CONTEXT_SCALES", [1, 3, 7, 15])
        if isinstance(raw_scales, str):
            raw_scales = [int(part.strip()) for part in raw_scales.split(",") if part.strip()]
        self.context_scales = tuple(
            sorted({max(1, int(scale)) | 1 for scale in raw_scales})
        )
        if not self.context_scales:
            self.context_scales = (1, 3, 7, 15)

        self.min_temperature = max(
            float(_m1(cfg, "V490_M3_MIN_TEMPERATURE", 0.10)), 1.0e-3
        )
        initial_temperature = max(
            float(_m1(cfg, "V490_M3_INITIAL_TEMPERATURE", 0.50)),
            self.min_temperature,
        )
        initial_raw = max(initial_temperature - self.min_temperature, 1.0e-4)
        self.raw_temperature = nn.Parameter(
            torch.log(torch.expm1(torch.tensor(initial_raw)))
        )

        groups = min(8, self.hidden_dim)
        while self.hidden_dim % groups != 0 and groups > 1:
            groups -= 1

        # Per-expert evidence: expert probability, factual probability, signed
        # and absolute intervention, entropy, boundary, M2 variance, M2 edit
        # gate, and the expert's intervention support.
        self.expert_encoder = nn.Sequential(
            nn.Conv2d(9, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
            nn.Dropout2d(self.dropout),
        )
        self.type_embedding = nn.Embedding(self.max_experts, self.hidden_dim)
        self.risk_head = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, 1, 1),
        )

        # Equal zero risk at initialisation is intentional: argmin then returns
        # Preserve/C0 and the new selector starts as an exact no-op.
        nn.init.zeros_(self.type_embedding.weight)
        nn.init.zeros_(self.risk_head[-1].weight)
        nn.init.zeros_(self.risk_head[-1].bias)

    @staticmethod
    def _resize_supports(
        supports: torch.Tensor,
        batch: int,
        channels: int,
        height: int,
        width: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        supports = supports.to(device=reference.device, dtype=reference.dtype)
        if supports.ndim == 3:
            supports = supports[:, None]
        if supports.ndim != 4:
            raise ValueError(
                f"candidate_supports must be [B,K,H,W], got {tuple(supports.shape)}"
            )
        if supports.shape[-2:] != (height, width):
            supports = F.interpolate(
                supports,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        if supports.shape[1] < channels:
            supports = torch.cat(
                [
                    supports,
                    supports.new_zeros(
                        batch,
                        channels - supports.shape[1],
                        height,
                        width,
                    ),
                ],
                dim=1,
            )
        return supports[:, :channels].clamp(0.0, 1.0)

    def aggregate_relative_risk(
        self,
        relative_risk: torch.Tensor,
        relevance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate risk with identical continuous geometry at every scale.

        ``relevance`` is detached before this method is called.  It defines the
        contextual averaging support but cannot be manipulated by a candidate
        to move its own decision label.  The expert probabilities still enter
        the risk encoder and final selected prediction without detachment.
        """
        if relative_risk.ndim != 4 or relevance.ndim != 4:
            raise ValueError("relative_risk and relevance must be rank-4 tensors")
        if relevance.shape[1] != 1:
            raise ValueError("relevance must have one channel")
        relevance = relevance.to(relative_risk.dtype).clamp(0.0, 1.0)
        aggregated = []
        masses = []
        for scale in self.context_scales:
            padding = scale // 2
            mass = F.avg_pool2d(
                relevance,
                kernel_size=scale,
                stride=1,
                padding=padding,
            )
            numerator = F.avg_pool2d(
                relative_risk * relevance,
                kernel_size=scale,
                stride=1,
                padding=padding,
            )
            local = torch.where(
                mass > EPS,
                numerator / mass.clamp_min(EPS),
                torch.zeros_like(numerator),
            )
            aggregated.append(local)
            masses.append(mass)
        result = torch.stack(aggregated, dim=0).mean(dim=0)
        # Preserve is the exact reference risk, not a learned offset.
        result = result - result[:, :1]
        context_mass = torch.stack(masses, dim=0).mean(dim=0)
        return result, context_mass

    def forward(
        self,
        c0_prob: torch.Tensor,
        candidate_probs: torch.Tensor,
        candidate_supports: torch.Tensor,
        m2_prob: torch.Tensor,
        m2_variance: torch.Tensor,
        m2_edit_gate: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        c0 = _as_b1hw(c0_prob).clamp(EPS, 1.0 - EPS)
        if candidate_probs.ndim != 4:
            raise ValueError("candidate_probs must be [B,K,H,W].")
        m2 = _as_b1hw(m2_prob).clamp(EPS, 1.0 - EPS)
        variance = _as_b1hw(m2_variance).clamp_min(0.0)
        edit_gate = _as_b1hw(m2_edit_gate).clamp(0.0, 1.0)
        experts = torch.cat([candidate_probs, m2], dim=1)
        batch, expert_count, height, width = experts.shape
        if expert_count > self.max_experts:
            raise ValueError(
                f"V490.2 M3 supports at most {self.max_experts} experts, "
                f"got {expert_count}"
            )

        supports = self._resize_supports(
            candidate_supports,
            batch,
            candidate_probs.shape[1],
            height,
            width,
            experts,
        )
        m2_support = torch.maximum(edit_gate, (m2 - c0).abs())
        expert_supports = torch.cat([supports, m2_support], dim=1)

        c0_expand = c0.expand(-1, expert_count, -1, -1)
        signed = experts - c0_expand
        abs_edit = signed.abs()
        entropy = _entropy(experts)
        boundary = _soft_boundary(
            experts.reshape(batch * expert_count, 1, height, width)
        ).reshape(batch, expert_count, height, width)
        variance_expand = variance.expand(-1, expert_count, -1, -1)
        gate_expand = edit_gate.expand(-1, expert_count, -1, -1)
        model_input = torch.stack(
            [
                experts,
                c0_expand,
                signed,
                abs_edit,
                entropy,
                boundary,
                variance_expand,
                gate_expand,
                expert_supports,
            ],
            dim=2,
        ).reshape(batch * expert_count, 9, height, width)
        feature = self.expert_encoder(model_input)
        type_ids = torch.arange(expert_count, device=experts.device).clamp_max(
            self.max_experts - 1
        )
        type_feature = self.type_embedding(type_ids).to(dtype=experts.dtype)
        feature = feature + type_feature[None].expand(batch, -1, -1).reshape(
            batch * expert_count, self.hidden_dim, 1, 1
        )

        raw_risk = self.risk_head(feature).reshape(
            batch, expert_count, height, width
        )
        relative_risk = raw_risk - raw_risk[:, :1]

        # Continuous edit relevance replaces thresholded binary components.
        # It is detached only as routing geometry; expert predictions and risk
        # features remain online and differentiable.
        relevance = abs_edit[:, 1:].amax(dim=1, keepdim=True).detach().clamp(0.0, 1.0)
        aggregated_risk, context_mass = self.aggregate_relative_risk(
            relative_risk,
            relevance,
        )

        if self.scale_invariant_risk:
            risk_scale = aggregated_risk[:, 1:].abs().mean(
                dim=1, keepdim=True
            ).detach().clamp_min(self.risk_scale_floor)
            decision_risk = aggregated_risk / risk_scale
            decision_risk = decision_risk - decision_risk[:, :1]
        else:
            risk_scale = aggregated_risk.new_ones(
                (batch, 1, height, width)
            )
            decision_risk = aggregated_risk

        temperature = F.softplus(self.raw_temperature) + self.min_temperature
        soft_weights = F.softmax(-decision_risk / temperature, dim=1)
        # Positive scaling leaves the hard expert argmin unchanged.
        selected_index = decision_risk.argmin(dim=1)
        hard_weights = F.one_hot(
            selected_index,
            num_classes=expert_count,
        ).permute(0, 3, 1, 2).to(experts.dtype)
        deploy_weights = (
            hard_weights + soft_weights - soft_weights.detach()
            if self.training
            else hard_weights
        )

        final = (deploy_weights * experts).sum(dim=1).clamp(EPS, 1.0 - EPS)
        selection_rate = hard_weights.flatten(2).mean(dim=-1)
        nonbase_soft = 1.0 - soft_weights[:, :1]
        nonbase_hard = (selected_index != 0).to(experts.dtype)[:, None]
        return {
            "m3_expert_probs": experts,
            "m3_predicted_relative_risk_map": relative_risk,
            # Compatibility name consumed by the loss.  This is the deployed,
            # multi-scale relative risk field, not an absolute non-negative risk.
            "m3_predicted_risk_map": aggregated_risk,
            "m3_aggregated_risk_map": aggregated_risk,
            "m3_normalized_risk_map": decision_risk,
            "m3_risk_scale": risk_scale,
            "m3_edit_relevance": relevance,
            "m3_intervention_mask": relevance,
            "m3_context_mass": context_mass,
            "m3_pixel_weights": soft_weights,
            "m3_deploy_weights": deploy_weights,
            "m3_selected_index": selected_index,
            "m3_expert_selection_rate": selection_rate,
            "m3_temperature": temperature.expand(batch),
            "m3_m2_margin": experts.new_zeros((batch,)),
            "m3_gate_prob": nonbase_soft,
            "m3_hard_gate": nonbase_hard,
            "m3_deploy_gate": nonbase_hard,
            "fused_probs": final,
            "final_probs": final,
        }


class CalibratedM3Rejector(nn.Module):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.force_preserve = bool(_m1(cfg, "CEM_M3_FORCE_PRESERVE", True))
        # V487: conservative deployment gate.  This gate is intentionally
        # separate from the generator gate: the generator may explore repair
        # candidates, but the final prediction is allowed to leave Preserve/C0
        # only when the verifier predicts positive gain, low harm, sufficient
        # support precision and bounded edit mass.
        self.local_margin = float(_m1(cfg, "V487_DEPLOY_GAIN_MARGIN", _m1(cfg, "V484_LOCAL_MARGIN", 0.004)))
        self.local_harm_threshold = float(_m1(cfg, "V487_DEPLOY_HARM_THRESHOLD", _m1(cfg, "V484_LOCAL_HARM_THRESHOLD", 0.15)))
        self.local_edit_budget = float(_m1(cfg, "V487_DEPLOY_EDIT_BUDGET", _m1(cfg, "V484_LOCAL_EDIT_BUDGET", 0.025)))
        self.min_support_precision = float(_m1(cfg, "V487_DEPLOY_MIN_SUPPORT_PRECISION", 0.55))
        self.harm_penalty = float(_m1(cfg, "V487_DEPLOY_HARM_PENALTY", 0.75))
        self.edit_penalty = float(_m1(cfg, "V487_DEPLOY_EDIT_PENALTY", 0.10))
        self.support_bonus = float(_m1(cfg, "V487_DEPLOY_SUPPORT_BONUS", 0.10))

    def forward(self, c0_prob: torch.Tensor, candidate_probs: torch.Tensor, local_m2: Dict[str, torch.Tensor], local_active: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, total, h, w = candidate_probs.shape
        scores = torch.zeros(b, total, device=candidate_probs.device, dtype=candidate_probs.dtype)
        eligible = torch.zeros(b, total, dtype=torch.bool, device=candidate_probs.device)
        if total > 1:
            k_local = min(4, total - 1)
            pred = local_m2["local_pred_delta_dsc"][:, :k_local]
            harm = local_m2["local_pred_harm_prob"][:, :k_local]
            support_precision = local_m2.get(
                "local_pred_support_precision",
                torch.ones_like(pred),
            )[:, :k_local]
            edit = (candidate_probs[:, 1:1 + k_local] - c0_prob).abs().flatten(2).mean(dim=-1)
            loc_eligible = (
                local_active[:, :k_local]
                & (pred > self.local_margin)
                & (harm < self.local_harm_threshold)
                & (support_precision > self.min_support_precision)
                & (edit <= self.local_edit_budget)
            )
            eligible[:, 1:1 + k_local] = loc_eligible
            scores[:, 1:1 + k_local] = (
                pred
                - self.harm_penalty * harm
                - self.edit_penalty * edit
                + self.support_bonus * (support_precision - 0.5)
            )
        scores = scores.masked_fill(~eligible, -1e4)
        scores[:, 0] = 0.0
        selected = scores.argmax(dim=1)
        if self.force_preserve:
            selected = torch.zeros_like(selected)
        gather = selected[:, None, None, None].expand(-1, 1, h, w)
        final = candidate_probs.gather(1, gather)[:, 0]
        return {
            "selected_index": selected,
            "accepted": (selected > 0).float(),
            "all_scores": scores,
            "fused_probs": final,
            "final_probs": final,
            "m3_eligible": eligible,
        }


def _v502_apply_ablation_output_stage(
    stage: str,
    c0_prob: torch.Tensor,
    m2_prob: torch.Tensor,
    m3_output: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Return the deployable output for a clean V502 module ablation.

    ``base`` and ``m1`` both deploy the factual Base prediction.  M1 is a
    candidate generator and has no learned deployer of its own, so reporting a
    GT-selected candidate as its final prediction would leak test labels.
    ``m2`` deploys the complete V502 soft M2 proposal and bypasses M3.
    ``full`` keeps the learned M3 output unchanged.

    Only the two public deployment aliases are replaced.  The original M3
    evidence tensors remain available for diagnostics, but its objective is
    disabled and its parameters are frozen by the matching training config.
    """
    normalized = str(stage or "full").strip().lower()
    aliases = {
        "a0": "base",
        "base_only": "base",
        "a1": "m1",
        "m1_only": "m1",
        "m1_generator": "m1",
        "a2": "m2",
        "m1_m2": "m2",
        "m2_only": "m2",
        "control": "full",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"base", "m1", "m2", "full"}:
        raise ValueError(
            "M1.V502_ABLATION_OUTPUT_STAGE must be one of "
            "{base,m1,m2,full}; got " + repr(stage)
        )

    updated = dict(m3_output)
    if normalized in {"base", "m1"}:
        deployed = _as_b1hw(c0_prob)[:, 0].clamp(EPS, 1.0 - EPS)
    elif normalized == "m2":
        deployed = _as_b1hw(m2_prob)[:, 0].clamp(EPS, 1.0 - EPS)
    else:
        return updated

    updated["fused_probs"] = deployed
    updated["final_probs"] = deployed
    return updated


class V484ErrorStateCausalPipeline(nn.Module):
    """Historical pipeline name with V487/V488/V489 compatible branches."""

    def __init__(self, cfg: Any, semantic_channels: int = 512) -> None:
        super().__init__()
        self.cfg = cfg
        self.current_epoch = 0
        self.m1_protocol = str(_m1(cfg, "PROTOCOL", "")).strip().lower()
        self.tc_drcs_enabled = self.m1_protocol == "tc_drcs"
        self.clean_dynamic_component_set_enabled = self.m1_protocol in {
            "clean_dynamic_component_set", "tc_drcs"
        }
        self.v532_enabled = bool(
            _m1(cfg, "V532_UNIFIED_SPARSE_REFINER_ENABLED", False)
        ) or self.clean_dynamic_component_set_enabled
        self.v533_detach_coarse_anchor = bool(
            _m1(cfg, "V533_DETACH_COARSE_ANCHOR", self.v532_enabled)
        )
        self.v531_enabled = bool(
            _m1(cfg, "V531_TYPED_SPARSE_REFINER_ENABLED", False)
        ) and not self.v532_enabled
        self.v518_enabled = bool(_m1(cfg, "V518_ENABLED", False))
        self.v518_use_semantic = bool(
            _m1(cfg, "V518_USE_SEMANTIC_FEATURES", self.v518_enabled)
        )
        self.v518_hierarchical_error = bool(
            _m1(cfg, "V518_HIERARCHICAL_ERROR_HEAD", self.v518_enabled)
        )
        self.v518_dose_values = _v518_float_tuple(
            _m1(cfg, "V518_DOSE_VALUES", (1.0,)), (1.0,)
        )
        self.v518_morph_radii = _v518_int_tuple(
            _m1(cfg, "V518_MORPH_RADII", ()), tuple()
        )
        self.v518_include_pairs = bool(
            _m1(cfg, "V518_PAIR_CANDIDATES", False)
        )
        self.v518_include_erode = bool(_m1(cfg, "V518_INCLUDE_ERODE", True))
        self.v518_include_dilate = bool(_m1(cfg, "V518_INCLUDE_DILATE", True))
        self.v518_include_open = bool(_m1(cfg, "V518_INCLUDE_OPEN", True))
        self.v518_include_close = bool(_m1(cfg, "V518_INCLUDE_CLOSE", True))
        self.v524_counterfactual_prompted_region = bool(
            _m1(cfg, "V524_COUNTERFACTUAL_PROMPTED_REGION_ENABLED", False)
        ) and not self.v531_enabled and not self.v532_enabled
        self.v523_sea_level_utility = bool(
            _m1(cfg, "V523_SEA_LEVEL_UTILITY_COMPOSER_ENABLED", False)
        ) and not self.v524_counterfactual_prompted_region
        self.v522_pwo_distilled_sequential = bool(
            _m1(cfg, "V522_PWO_DISTILLED_SEQUENTIAL_LOCAL_ENABLED", False)
        ) and not self.v523_sea_level_utility and not self.v524_counterfactual_prompted_region
        self.v521_candidate_conditional_region = bool(
            _m1(cfg, "V521_CANDIDATE_CONDITIONAL_REGION_COMPOSER_ENABLED", False)
        ) and not self.v522_pwo_distilled_sequential and not self.v523_sea_level_utility and not self.v524_counterfactual_prompted_region
        self.v520_gate_selector = bool(
            _m1(cfg, "V520_GATE_SELECTOR_REGION_COMPOSER_ENABLED", False)
        )
        self.v519_region_composer = (
            bool(_m1(cfg, "V519_FAMILY_RISK_REGION_COMPOSER_ENABLED", False))
            or self.v520_gate_selector
            or self.v521_candidate_conditional_region
            or self.v522_pwo_distilled_sequential
            or self.v523_sea_level_utility
            or self.v524_counterfactual_prompted_region
        )
        self.v519_exclude_erode_r6 = bool(
            _m1(
                cfg,
                "V524_EXCLUDE_ERODE_R6",
                _m1(
                    cfg,
                    "V523_EXCLUDE_ERODE_R6",
                    _m1(
                        cfg,
                        "V522_EXCLUDE_ERODE_R6",
                        _m1(
                            cfg,
                            "V521_EXCLUDE_ERODE_R6",
                            _m1(
                                cfg,
                                "V520_EXCLUDE_ERODE_R6",
                                _m1(cfg, "V519_EXCLUDE_ERODE_R6", True),
                            ),
                        ),
                    ),
                ),
            )
        )
        self.v519_deploy_pair_candidates = bool(
            _m1(
                cfg,
                "V524_DEPLOY_PAIR_CANDIDATES",
                _m1(
                    cfg,
                    "V523_DEPLOY_PAIR_CANDIDATES",
                    _m1(
                        cfg,
                        "V522_DEPLOY_PAIR_CANDIDATES",
                        _m1(
                            cfg,
                            "V521_DEPLOY_PAIR_CANDIDATES",
                            _m1(
                                cfg,
                                "V520_DEPLOY_PAIR_CANDIDATES",
                                _m1(cfg, "V519_DEPLOY_PAIR_CANDIDATES", False),
                            ),
                        ),
                    ),
                ),
            )
        )
        self.v519_deploy_global_candidates = bool(
            _m1(
                cfg,
                "V524_DEPLOY_GLOBAL_CANDIDATES",
                _m1(
                    cfg,
                    "V523_DEPLOY_GLOBAL_CANDIDATES",
                    _m1(
                        cfg,
                        "V522_DEPLOY_GLOBAL_CANDIDATES",
                        _m1(
                            cfg,
                            "V521_DEPLOY_GLOBAL_CANDIDATES",
                            _m1(
                                cfg,
                                "V520_DEPLOY_GLOBAL_CANDIDATES",
                                _m1(cfg, "V519_DEPLOY_GLOBAL_CANDIDATES", False),
                            ),
                        ),
                    ),
                ),
            )
        )
        hidden = int(_m1(cfg, "V484_HIDDEN_DIM", _m1(cfg, "V485_HIDDEN_DIM", 64)))
        num_discovery = int(_m1(cfg, "V484_NUM_DISCOVERY", _m1(cfg, "V485_NUM_DISCOVERY", 0)))
        self.v490_enabled = bool(_m1(cfg, "V490_ROOT_CAUSE_ENABLED", False))
        self.v491_enabled = bool(
            _m1(cfg, "V491_PRESERVE_FIRST_ENABLED", False)
        ) and self.v490_enabled
        self.v492_enabled = bool(
            _m1(cfg, "V492_CAUSAL_LOCAL_EDITOR_ENABLED", False)
        ) and self.v491_enabled
        self.v493_enabled = bool(
            _m1(cfg, "V493_CANDIDATE_CONDITIONED_CAUSAL_ENABLED", False)
        ) and self.v492_enabled
        self.v494_enabled = bool(
            _m1(cfg, "V494_DIRECT_CAUSAL_DOSE_ENABLED", False)
        ) and self.v493_enabled
        self.v495_enabled = bool(
            _m1(cfg, "V495_SINGLE_DECISION_ENABLED", False)
        ) and self.v494_enabled
        self.v498_enabled = bool(
            _m1(cfg, "V498_CONSISTENT_FULL_PROPOSAL_ENABLED", False)
        ) and self.v495_enabled
        self.v499_enabled = bool(
            _m1(cfg, "V499_CAUSAL_PRESERVE_REFERENCE_ENABLED", False)
        ) and self.v498_enabled
        self.v500_enabled = bool(
            _m1(cfg, "V500_HIERARCHICAL_SAFE_ROUTE_ENABLED", False)
        ) and self.v499_enabled
        self.v501_enabled = bool(
            _m1(cfg, "V501_BASE_ANCHORED_SELECTIVE_REPAIR_ENABLED", False)
        ) and self.v500_enabled
        self.v502_enabled = bool(
            _m1(cfg, "V502_HIERARCHICAL_UTILITY_SOFT_ROUTER_ENABLED", False)
        ) and self.v501_enabled
        self.v504_enabled = bool(
            _m1(cfg, "V504_REALIZABLE_POTENTIAL_OUTCOME_ENABLED", False)
        ) and self.v502_enabled
        self.v505_enabled = bool(
            _m1(cfg, "V505_INTERACTIVE_REGION_CAUSAL_ENABLED", False)
        ) and self.v502_enabled
        self.v503_enabled = (
            bool(_m1(cfg, "V503_FACTUAL_ATOMIC_CAUSAL_ENABLED", False))
            or self.v504_enabled
            or self.v505_enabled
            or self.v531_enabled
            or self.v532_enabled
        ) and (self.v502_enabled or self.v531_enabled or self.v532_enabled)
        self.v502_ablation_output_stage = str(
            _m1(cfg, "V502_ABLATION_OUTPUT_STAGE", "full")
        ).strip().lower()
        # Validate the requested stage at construction time so a misspelled
        # ablation cannot silently train or test the wrong deployment path.
        _v502_apply_ablation_output_stage(
            self.v502_ablation_output_stage,
            torch.full((1, 1, 1, 1), 0.5),
            torch.full((1, 1, 1, 1), 0.5),
            {"fused_probs": torch.full((1, 1, 1), 0.5),
             "final_probs": torch.full((1, 1, 1), 0.5)},
        )
        self.v495_mc_logit_aggregation = bool(
            _m1(cfg, "V495_MC_LOGIT_AGGREGATION_ENABLED", True)
        ) and self.v495_enabled
        if self.v503_enabled:
            semantic_dim = int(_m1(cfg, "V518_SEMANTIC_DIM", 32))
            self.error_state_head = PixelCausalErrorStateHead(
                hidden_dim=hidden,
                dropout=float(_m1(cfg, "V503_CAUSE_DROPOUT", 0.10)),
                initial_rate=float(_m1(cfg, "V503_INITIAL_CAUSE_RATE", 0.02)),
                semantic_channels=int(semantic_channels),
                semantic_dim=semantic_dim,
                use_semantic=self.v518_use_semantic,
                hierarchical=self.v518_hierarchical_error,
            )
            self.local_generator = AtomicLocalRepairGenerator(
                hidden_dim=hidden,
                max_atom_delta=float(_m1(cfg, "V503_MAX_ATOM_LOGIT_DELTA", 3.0)),
                delta_bias=float(_m1(cfg, "V503_DELTA_BIAS", -1.5)),
                support_bias=float(_m1(cfg, "V503_SUPPORT_BIAS", -2.0)),
                cause_gated=bool(
                    _m1(cfg, "V516_CAUSE_GATED_SUPPORT", self.v505_enabled)
                ),
                min_action_strength=float(
                    _m1(cfg, "V516_MIN_ACTION_STRENGTH", 0.0)
                ),
                semantic_channels=int(semantic_channels),
                semantic_dim=semantic_dim,
                use_semantic=self.v518_use_semantic,
                dose_values=self.v518_dose_values,
                candidate_to_cause_grad_scale=float(
                    _m1(cfg, "V518_CANDIDATE_TO_CAUSE_GRAD_SCALE", 0.0)
                ),
            )
        else:
            self.error_state_head = ErrorStateHead(hidden_dim=hidden)
            self.local_generator = GatedLocalRepairGenerator(
                hidden_dim=hidden,
                max_atom_delta=float(
                    _m1(
                        cfg,
                        "V486_MAX_EDIT_LOGIT_DELTA",
                        _m1(cfg, "V484_LOCAL_DELTA_CAP", _m1(cfg, "V485_LOCAL_DELTA_CAP", 2.0)),
                    )
                ),
                delta_bias=float(_m1(cfg, "V486_DELTA_BIAS", -2.0)),
                support_bias=float(_m1(cfg, "V486_SUPPORT_BIAS", -2.0)),
            )
        self.global_generator = (
            GatedGlobalRediscoveryGenerator(
                hidden_dim=hidden,
                num_discovery=num_discovery,
                max_delta=float(
                    _m1(
                        cfg,
                        "V484_GLOBAL_DELTA_CAP",
                        _m1(cfg, "V485_GLOBAL_DELTA_CAP", 4.0),
                    )
                ),
                semantic_channels=int(semantic_channels),
                direct_rediscovery=self.v501_enabled,
                initial_foreground_rate=float(
                    _m1(cfg, "V501_GLOBAL_INITIAL_FOREGROUND_RATE", 0.05)
                ),
            )
            if num_discovery > 0
            else None
        )
        # V490/V491 remove the obsolete V487 local verifier/rejector branch from the
        # active architecture instead of carrying trainable-but-unused tensors.
        self.local_verifier = (
            None if (self.v490_enabled or self.v531_enabled or self.v532_enabled) else LocalEffectVerifier(hidden_dim=hidden)
        )
        self.rejector = (
            None if (self.v490_enabled or self.v531_enabled or self.v532_enabled) else CalibratedM3Rejector(cfg)
        )

        self.v489_enabled = bool(_m1(cfg, "V489_END_TO_END_ENABLED", False)) and not self.v490_enabled
        self.v488_enabled = (
            bool(_m1(cfg, "V488_PIXEL_COMPOSER_ENABLED", False))
            and not self.v489_enabled
            and not self.v490_enabled
        )
        if self.v532_enabled:
            self.pixel_composer = V532UnifiedSparseRefiner(
                hidden_dim=int(_m1(cfg, "V532_HIDDEN_DIM", hidden)),
                dropout=float(_m1(cfg, "V532_DROPOUT", 0.10)),
                semantic_channels=int(semantic_channels),
                semantic_dim=int(_m1(cfg, "V532_SEMANTIC_DIM", 32)),
                use_semantic=bool(
                    _m1(cfg, "V532_USE_SEMANTIC_FEATURES", self.v518_use_semantic)
                ),
                edit_temperature=float(_m1(cfg, "V532_EDIT_TEMPERATURE", 1.0)),
                action_temperature=float(_m1(cfg, "V532_ACTION_TEMPERATURE", 0.70)),
                outcome_temperature=float(_m1(cfg, "V532_OUTCOME_TEMPERATURE", 1.0)),
                risk_temperature=float(_m1(cfg, "V532_RISK_TEMPERATURE", 0.20)),
                harm_penalty=float(_m1(cfg, "V532_HARM_PENALTY", 2.0)),
                edit_penalty=float(_m1(cfg, "V532_EDIT_PENALTY", 0.03)),
                utility_threshold=float(_m1(cfg, "V532_UTILITY_THRESHOLD", 0.0)),
                initial_edit_rate=float(_m1(cfg, "V532_INITIAL_EDIT_RATE", 0.02)),
                initial_outcome_neutral=float(_m1(cfg, "V532_INITIAL_OUTCOME_NEUTRAL", 0.80)),
                initial_outcome_benefit=float(_m1(cfg, "V532_INITIAL_OUTCOME_BENEFIT", 0.10)),
                initial_outcome_harm=float(_m1(cfg, "V532_INITIAL_OUTCOME_HARM", 0.10)),
                max_action_alpha=float(_m1(cfg, "V532_MAX_ACTION_ALPHA", 1.0)),
                deployment_hard_route=bool(
                    _m1(cfg, "V533_EXACT_DEPLOYMENT_ROUTE", True)
                ),
                execute_threshold=float(
                    _m1(cfg, "V533_EXECUTE_THRESHOLD", 0.50)
                ),
                adaptive_utility_policy=bool(
                    _m1(cfg, "V535_ADAPTIVE_UTILITY_POLICY_ENABLED", False)
                ),
                clean_dynamic_component_set=self.clean_dynamic_component_set_enabled,
                tc_drcs=self.tc_drcs_enabled,
                policy_temperature=float(
                    _m1(cfg, "V535_POLICY_TEMPERATURE", 1.0)
                ),
                prior_aligned_case_component=bool(
                    _m1(cfg, "V536_PRIOR_ALIGNED_CASE_COMPONENT_ENABLED", False)
                ),
                component_utility_ranking=bool(
                    _m1(cfg, "V537_COMPONENT_UTILITY_RANKER_ENABLED", False)
                ),
                component_min_pixels=int(
                    _m1(cfg, "V537_COMPONENT_MIN_PIXELS",
                        _m1(cfg, "V536_COMPONENT_MIN_PIXELS", 4))
                ),
                max_components=int(_m1(cfg, "V537_MAX_COMPONENTS", 48)),
                component_ranker_hidden_dim=int(
                    _m1(cfg, "V537_RANKER_HIDDEN_DIM", 128)
                ),
                component_ranker_dropout=float(
                    _m1(cfg, "V537_RANKER_DROPOUT", 0.10)
                ),
                component_initial_score_bias=float(
                    _m1(cfg, "V538_INITIAL_GAIN_BIAS",
                        _m1(cfg, "V537_INITIAL_SCORE_BIAS", -0.002))
                ),
                online_component_refinement=(
                    self.clean_dynamic_component_set_enabled
                    or bool(_m1(cfg, "V538_ONLINE_COMPONENT_REFINER_ENABLED", False))
                ),
                component_num_slots=int(
                    _m1(cfg, "NUM_COMPONENT_SLOTS", 6) if self.clean_dynamic_component_set_enabled
                    else _m1(cfg, "V538_NUM_COMPONENT_SLOTS", 8)
                ),
                component_slot_hidden_dim=int(
                    _m1(cfg, "V538_COMPONENT_HIDDEN_DIM", 128)
                ),
                component_slot_dropout=float(
                    _m1(cfg, "V538_COMPONENT_DROPOUT", 0.10)
                ),
                component_mask_temperature=float(
                    _m1(cfg, "V538_MASK_TEMPERATURE", 1.0)
                ),
                component_action_temperature=float(
                    _m1(cfg, "V538_ACTION_TEMPERATURE", 0.70)
                ),
                component_min_area_fraction=float(
                    _m1(cfg, "V538_MIN_COMPONENT_AREA_FRACTION", 1.0e-4)
                ),
                component_max_area_fraction=float(
                    _m1(cfg, "V538_MAX_COMPONENT_AREA_FRACTION", 0.035)
                ),
                component_initial_presence_rate=float(
                    _m1(cfg, "V538_INITIAL_PRESENCE_RATE", 0.10)
                ),
                component_deploy_mask_threshold=float(
                    _m1(cfg, "V538_DEPLOY_MASK_THRESHOLD", 0.50)
                ),
                component_deploy_presence_threshold=float(
                    _m1(cfg, "V538_DEPLOY_PRESENCE_THRESHOLD", 0.50)
                ),
                component_deploy_min_gain=float(
                    _m1(cfg, "V538_COMPOSER_MIN_GAIN", 0.001)
                ),
                component_max_steps=(
                    1 if self.clean_dynamic_component_set_enabled
                    else int(_m1(cfg, "V538_COMPOSER_MAX_STEPS", 3))
                ),
                component_max_overlap=float(
                    _m1(cfg, "V538_COMPOSER_MAX_OVERLAP", 0.20)
                ),
                component_max_total_edit_fraction=float(
                    _m1(cfg, "V538_COMPOSER_MAX_EDIT_FRACTION", 0.035)
                ),
                component_continuous_dose_enabled=bool(
                    _m1(cfg, "V540_CONTINUOUS_DOSE_ENABLED", True)
                ),
                component_polarity_temperature=float(
                    _m1(cfg, "V540_POLARITY_TEMPERATURE", 0.70)
                ),
                component_minimum_dose=float(
                    _m1(cfg, "V540_MINIMUM_DOSE", 0.25)
                ),
                component_maximum_dose=float(
                    _m1(cfg, "V540_MAXIMUM_DOSE", 16.0)
                ),
                component_initial_dose=float(
                    _m1(cfg, "V540_INITIAL_DOSE", 1.0)
                ),
                component_candidate_outcome_selector_enabled=bool(
                    _m1(cfg, "V541_CANDIDATE_OUTCOME_SELECTOR_ENABLED", True)
                ),
                component_selector_spatial_size=int(
                    _m1(cfg, "V541_SELECTOR_SPATIAL_SIZE", 16)
                ),
                component_selector_hidden_dim=int(
                    _m1(cfg, "V541_SELECTOR_HIDDEN_DIM", 128)
                ),
                component_selector_dropout=float(
                    _m1(cfg, "V541_SELECTOR_DROPOUT", 0.10)
                ),
                component_selector_gain_scale=float(
                    _m1(cfg, "V541_GAIN_SCALE", 1000.0)
                ),
                component_selector_benefit_threshold=float(
                    _m1(cfg, "V541_DEPLOY_BENEFIT_THRESHOLD", 0.70)
                ),
                component_selector_harm_threshold=float(
                    _m1(cfg, "V541_DEPLOY_HARM_THRESHOLD", 0.10)
                ),
                component_selector_lcb_beta=float(
                    _m1(cfg, "V541_DEPLOY_LCB_BETA", 1.0)
                ),
                component_selector_logvar_min=float(
                    _m1(cfg, "V541_LOGVAR_MIN", -8.0)
                ),
                component_selector_logvar_max=float(
                    _m1(cfg, "V541_LOGVAR_MAX", 4.0)
                ),
                component_selector_initial_benefit_rate=float(
                    _m1(cfg, "V541_INITIAL_BENEFIT_RATE", 0.10)
                ),
                component_selector_initial_harm_rate=float(
                    _m1(cfg, "V541_INITIAL_HARM_RATE", 0.50)
                ),
                component_use_gain_as_decision_score=bool(
                    _m1(cfg, "V544_USE_GAIN_AS_DECISION_SCORE", False)
                ),
                component_adaptive_cardinality_hard_mask=bool(
                    _m1(cfg, "V545_ADAPTIVE_HARD_MASK_ENABLED", False)
                ),
                component_prior_free_outcome_init=bool(
                    _m1(cfg, "V545_CLASS_COMPLETE_OUTCOME_ENABLED", False)
                    or _m1(cfg, "V546_STREAMING_BALANCED_SOFTMAX_ENABLED", False)
                ),
                component_slot_competition_enabled=bool(
                    _m1(cfg, "V546_SLOT_COMPETITION_ENABLED", False)
                ),
                component_gain_sign_shadow_deploy_enabled=bool(
                    _m1(cfg, "V546_GAIN_SIGN_SHADOW_DEPLOY_ENABLED", False)
                ),
                component_factorized_outcome_enabled=bool(
                    _m1(cfg, "V547_FACTORIZED_OUTCOME_ENABLED", False)
                ),
                component_factorized_direction_zero_init=bool(
                    _m1(cfg, "V548_FACTORIZED_DIRECTION_ZERO_INIT", False)
                ),
                component_factorized_deployment_enabled=bool(
                    _m1(cfg, "V549_FACTORIZED_DEPLOYMENT_ENABLED", False)
                ),
                component_deploy_editability_threshold=float(
                    _m1(cfg, "V549_DEPLOY_EDITABILITY_THRESHOLD", 0.50)
                ),
                component_deploy_direction_threshold=float(
                    _m1(cfg, "V549_DEPLOY_DIRECTION_THRESHOLD", 0.50)
                ),
                multiscale_typed_editor=(
                    self.clean_dynamic_component_set_enabled
                    or bool(_m1(cfg, "V551_MULTISCALE_TYPED_EDITOR_ENABLED", False))
                ),
                component_pyramid_scales=tuple(
                    _m1(cfg, "V551_PYRAMID_SCALES", [1, 2, 4, 8])
                ),
                component_scale_temperature=float(
                    _m1(cfg, "V551_SCALE_TEMPERATURE", 0.70)
                ),
                component_max_atoms_per_slot=int(
                    _m1(cfg, "V551_MAX_ATOMS_PER_SLOT", 3)
                ),
                component_atomization_start_epoch=int(
                    _m1(cfg, "V551_ATOMIZATION_START_EPOCH", 1)
                ),
                component_atom_min_pixels=int(
                    _m1(cfg, "V551_ATOM_MIN_PIXELS", 4)
                ),
                component_atom_dedup_iou=float(
                    _m1(cfg, "V551_ATOM_DEDUP_IOU", 0.85)
                ),
                component_gpu_atomizer_enabled=bool(
                    _m1(cfg, "V551_GPU_ATOMIZER_ENABLED", True)
                ),
                component_max_active_atoms=int(
                    _m1(cfg, "V551_MAX_ACTIVE_ATOMS", 8)
                ),
                component_atom_partition_temperature=float(
                    _m1(cfg, "V551_ATOM_PARTITION_TEMPERATURE", 0.35)
                ),
                component_atom_partition_extent=float(
                    _m1(cfg, "V551_ATOM_PARTITION_EXTENT", 0.85)
                ),
                component_atom_presence_threshold=float(
                    _m1(cfg, "V552_ATOM_PRESENCE_THRESHOLD", 0.35)
                ),
                component_atom_quality_threshold=float(
                    _m1(cfg, "V552_ATOM_QUALITY_THRESHOLD", 0.20)
                ),
                component_atom_quality_gate_start_epoch=int(
                    _m1(cfg, "V552_ATOM_QUALITY_GATE_START_EPOCH", 4)
                ),
                component_single_pass_editor=bool(
                    _m1(cfg, "V551_SINGLE_PASS_EDITOR", True)
                ),
                component_boundary_residual_enabled=bool(
                    _m1(cfg, "V551_BOUNDARY_RESIDUAL_ENABLED", True)
                ),
                component_boundary_residual_cap=float(
                    _m1(cfg, "V551_BOUNDARY_RESIDUAL_CAP", 0.75)
                ),
                component_boundary_band_radii=tuple(
                    _m1(cfg, "V551_BOUNDARY_BAND_RADII", [1, 1, 2, 3])
                ),
                component_editor_enabled=bool(
                    _m1(cfg, "V551_REGION_EDITOR_ENABLED", True)
                ),
                component_editor_start_epoch=int(
                    _m1(cfg, "V551_EDITOR_START_EPOCH", 2)
                ),
                component_editor_ramp_epochs=int(
                    _m1(cfg, "V551_EDITOR_RAMP_EPOCHS", 8)
                ),
                component_editor_route_temperature=float(
                    _m1(cfg, "V551_EDITOR_ROUTE_TEMPERATURE", 0.70)
                ),
                component_editor_preserve_bias=float(
                    _m1(cfg, "V551_EDITOR_PRESERVE_BIAS", 1.25)
                ),
                component_editor_dose_adjust_min=float(
                    _m1(cfg, "V551_EDITOR_DOSE_ADJUST_MIN", 0.50)
                ),
                component_editor_dose_adjust_max=float(
                    _m1(cfg, "V551_EDITOR_DOSE_ADJUST_MAX", 1.50)
                ),
                component_editor_local_residual_cap=float(
                    _m1(cfg, "V551_EDITOR_LOCAL_RESIDUAL_CAP", 0.75)
                ),
                component_editor_total_logit_delta_cap=float(
                    _m1(cfg, "V552_EDITOR_TOTAL_LOGIT_DELTA_CAP", 1.0)
                ),
                component_editor_region_radii=tuple(
                    _m1(cfg, "V551_EDITOR_REGION_RADII", [1, 2, 4, 7])
                ),
                component_editor_residual_dropout=float(
                    _m1(cfg, "V551_EDITOR_RESIDUAL_DROPOUT", 0.10)
                ),
                component_unified_deployment_gate=bool(
                    _m1(cfg, "V552_UNIFIED_DEPLOYMENT_GATE", True)
                ),
                component_deployment_benefit_harm_margin=float(
                    _m1(cfg, "V552_DEPLOY_BENEFIT_HARM_MARGIN", 0.10)
                ),
                component_multicandidate_composer_enabled=bool(
                    _m1(cfg, "V552_MULTICANDIDATE_COMPOSER_ENABLED", True)
                ),
                component_composer_stop_bias=float(
                    _m1(cfg, "V552_COMPOSER_STOP_BIAS", 0.0)
                ),
                component_composer_marginal_correction_cap=float(
                    _m1(cfg, "V552_COMPOSER_MARGINAL_CORRECTION_CAP", 0.02)
                ),
                component_composer_overlap_penalty=float(
                    _m1(cfg, "V552_COMPOSER_OVERLAP_PENALTY", 0.50)
                ),
                component_composer_conflict_penalty=float(
                    _m1(cfg, "V552_COMPOSER_CONFLICT_PENALTY", 1.00)
                ),
                component_composer_budget_penalty=float(
                    _m1(cfg, "V552_COMPOSER_BUDGET_PENALTY", 0.50)
                ),
                component_teacher_decoupled_r2_enabled=bool(
                    _m1(cfg, "V552R2_TEACHER_DECOUPLED_ENABLED", False)
                ),
                component_composer_teacher_pool_size=int(
                    _m1(cfg, "V552_COMPOSER_TEACHER_POOL_SIZE", 4)
                ),
                component_composer_teacher_stop_margin=float(
                    _m1(cfg, "V552_COMPOSER_TEACHER_STOP_MARGIN", 0.0)
                ),
                component_composer_deploy_pool_size=int(
                    _m1(cfg, "V552_COMPOSER_DEPLOY_POOL_SIZE", 3)
                ),
                component_shadow_evidence_start_epoch=int(
                    _m1(cfg, "V552_SHADOW_EVIDENCE_START_EPOCH", 14)
                ),
                component_unified_reference_r4_enabled=bool(
                    _m1(cfg, "V552R4_UNIFIED_REFERENCE_CONTRACT_ENABLED", False)
                ),
                component_editor_relative_margin=float(
                    _m1(cfg, "V552_EDITOR_RELATIVE_MARGIN", 1.0e-3)
                ),
                component_editor_safety_benefit_threshold=float(
                    _m1(cfg, "V552R4_EDITOR_SAFETY_BENEFIT_THRESHOLD", 0.45)
                ),
                component_editor_safety_harm_threshold=float(
                    _m1(cfg, "V552R4_EDITOR_SAFETY_HARM_THRESHOLD", 0.35)
                ),
                component_editor_incremental_gain_threshold=float(
                    _m1(cfg, "V552R4_EDITOR_INCREMENTAL_GAIN_THRESHOLD", 0.0)
                ),
                component_critic_gain_cap=float(
                    _m1(cfg, "V552R4_CRITIC_GAIN_CAP", 0.05)
                ),
                component_critic_queue_capacity=int(
                    _m1(cfg, "V552R4_CRITIC_QUEUE_CAPACITY", 256)
                ),
                component_decoupled_critic_r42_enabled=bool(
                    _m1(cfg, "V552R42_DECOUPLED_CRITIC_ENABLED", False)
                ),
                component_spatial_evidence_r43_enabled=bool(
                    _m1(cfg, "V552R43_SPATIAL_ROUTE_EVIDENCE_ENABLED", False)
                ),
                component_audit_gate_r44_enabled=bool(
                    _m1(cfg, "V552R44_AUDIT_GATE_ROOTFIX_ENABLED", False)
                ),
                component_class_value_decoupling_r44_enabled=bool(
                    _m1(cfg, "V552R44_CLASS_VALUE_DECOUPLING_ENABLED", False)
                ),
                component_semantic_deployment_r44_enabled=bool(
                    _m1(cfg, "V552R44_SEMANTIC_DEPLOYMENT_ENABLED", False)
                ),
                component_audit_shadow_start_epoch=int(
                    _m1(cfg, "V552R44_AUDIT_SHADOW_START_EPOCH", 12)
                ),
                component_audit_shadow_topk=int(
                    _m1(cfg, "V552R44_AUDIT_SHADOW_TOPK", 1)
                ),
                component_error_aware_r45_enabled=bool(
                    _m1(cfg, "V552R45_ERROR_AWARE_ENABLED", False)
                ),
                component_factorized_safety_r45_enabled=bool(
                    _m1(cfg, "V552R45_FACTORIZED_SAFETY_ENABLED", False)
                ),
                component_direct_signed_utility_r45_enabled=bool(
                    _m1(cfg, "V552R45_DIRECT_SIGNED_UTILITY_ENABLED", False)
                ),
                component_factorized_composer_r45_enabled=bool(
                    _m1(cfg, "V552R45_FACTORIZED_COMPOSER_ENABLED", False)
                ),
                component_native_contract_r46_enabled=bool(
                    _m1(cfg, "V552R46_ROOTFIX_ENABLED", False)
                ),
                component_spatial_realization_r47_enabled=bool(
                    _m1(cfg, "V552R47_SPATIALLY_ANCHORED_QUERY_MASK_ENABLED", False)
                ),
                component_r47_mask_dim=int(
                    _m1(cfg, "V552R47_MASK_DIM", 64)
                ),
                component_r47_query_logit_scale=float(
                    _m1(cfg, "V552R47_QUERY_LOGIT_SCALE", 4.0)
                ),
                component_r47_anchor_prior_scale=float(
                    _m1(cfg, "V552R47_ANCHOR_PRIOR_SCALE", 1.5)
                ),
                component_r47_anchor_min_size=float(
                    _m1(cfg, "V552R47_ANCHOR_MIN_SIZE", 0.04)
                ),
                component_r47_anchor_max_size=float(
                    _m1(cfg, "V552R47_ANCHOR_MAX_SIZE", 0.55)
                ),
                component_r47_direct_slot_components=bool(
                    _m1(cfg, "V552R47_DIRECT_SLOT_COMPONENTS_ENABLED", False)
                ),
                component_iterative_binding_r48_enabled=bool(
                    _m1(cfg, "V552R48_ITERATIVE_BINDING_ENABLED", False)
                ),
                component_r48_decoder_layers=int(_m1(cfg, "V552R48_DECODER_LAYERS", 3)),
                component_r48_local_grid_size=int(_m1(cfg, "V552R48_LOCAL_GRID_SIZE", 3)),
                component_r48_anchor_delta_scale=float(_m1(cfg, "V552R48_ANCHOR_DELTA_SCALE", 0.75)),
                component_r48_window_prior_scale=float(_m1(cfg, "V552R48_WINDOW_PRIOR_SCALE", 2.0)),
                component_r48_window_temperature=float(_m1(cfg, "V552R48_WINDOW_TEMPERATURE", 0.025)),
                component_r48_remove_coarse_mask_bias=bool(_m1(cfg, "V552R48_REMOVE_COARSE_MASK_BIAS_ENABLED", True)),
                component_r48_deep_supervision_enabled=bool(_m1(cfg, "V552R48_DEEP_SUPERVISION_ENABLED", False)),
                component_r48_dn_enabled=bool(_m1(cfg, "V552R48_DN_COMPONENT_QUERY_ENABLED", False)),
                component_r48_dn_groups=int(_m1(cfg, "V552R48_DN_GROUPS", 2)),
                component_r48_dn_noise_scale=float(_m1(cfg, "V552R48_DN_NOISE_SCALE", 0.35)),
                component_r48_teacher_min_pixels=int(_m1(cfg, "V538_TEACHER_MIN_PIXELS", 4)),
                component_content_selective_r49_enabled=bool(
                    _m1(cfg, "V552R49_CONTENT_SELECTIVE_ATTENTION_ENABLED", False)
                ),
                component_r49_anchor_sampling_only=bool(
                    _m1(cfg, "V552R49_ANCHOR_SAMPLING_ONLY_ENABLED", True)
                ),
                component_r49_attention_logit_scale_init=float(
                    _m1(cfg, "V552R49_ATTENTION_LOGIT_SCALE_INIT", 10.0)
                ),
                component_r49_attention_logit_scale_max=float(
                    _m1(cfg, "V552R49_ATTENTION_LOGIT_SCALE_MAX", 100.0)
                ),
                component_r49_dn_curriculum_enabled=bool(
                    _m1(cfg, "V552R49_DN_CURRICULUM_ENABLED", False)
                ),
                component_r49_dn_noise_start=float(
                    _m1(cfg, "V552R49_DN_NOISE_START", 0.05)
                ),
                component_r49_dn_noise_final=float(
                    _m1(cfg, "V552R49_DN_NOISE_FINAL", 0.30)
                ),
                component_r49_dn_noise_ramp_epochs=int(
                    _m1(cfg, "V552R49_DN_NOISE_RAMP_EPOCHS", 15)
                ),
                component_evidence_proposal_r410_enabled=bool(
                    _m1(cfg, "V552R410_ROOTFIX_ENABLED", False)
                ),
                component_r410_use_evidence_proposals=bool(
                    _m1(cfg, "V552R410_EVIDENCE_PROPOSAL_ENABLED", True)
                ),
                component_r410_support_only_local_readout=bool(
                    _m1(cfg, "V552R410_SUPPORT_ONLY_LOCAL_READOUT_ENABLED", False)
                ),
                component_r410_proposal_nms_kernel=int(
                    _m1(cfg, "V552R410_PROPOSAL_NMS_KERNEL", 17)
                ),
                component_r410_proposal_score_threshold=float(
                    _m1(cfg, "V552R410_PROPOSAL_SCORE_THRESHOLD", 0.05)
                ),
                component_r410_support_expand=float(
                    _m1(cfg, "V552R410_SUPPORT_EXPAND", 1.75)
                ),
                component_r410_support_temperature=float(
                    _m1(cfg, "V552R410_SUPPORT_TEMPERATURE", 0.02)
                ),
                component_r410_support_max_penalty=float(
                    _m1(cfg, "V552R410_SUPPORT_MAX_PENALTY", 8.0)
                ),
                component_r410_dn_clean_curriculum_enabled=bool(
                    _m1(cfg, "V552R410_DN_CLEAN_CURRICULUM_ENABLED", True)
                ),
                component_r410_dn_clean_epochs=int(
                    _m1(cfg, "V552R410_DN_CLEAN_EPOCHS", 10)
                ),
                component_r410_dn_noise_final=float(
                    _m1(cfg, "V552R410_DN_NOISE_FINAL", 0.20)
                ),
                component_r410_dn_noise_ramp_epochs=int(
                    _m1(cfg, "V552R410_DN_NOISE_RAMP_EPOCHS", 50)
                ),
                component_native_residual_set_r411_enabled=bool(
                    _m1(cfg, "V552R411_ROOTFIX_ENABLED", False)
                ),
                component_r411_typed_proposal_enabled=bool(
                    _m1(cfg, "V552R411_TYPED_PROPOSAL_ENABLED", True)
                ),
                component_r411_local_roi_decoder_enabled=bool(
                    _m1(cfg, "V552R411_LOCAL_ROI_DECODER_ENABLED", True)
                ),
                component_r411_use_raw_native_masks=bool(
                    _m1(cfg, "V552R411_USE_RAW_NATIVE_MASKS", True)
                ),
                component_r411_proposal_nms_kernel=int(
                    _m1(cfg, "V552R411_PROPOSAL_NMS_KERNEL", 9)
                ),
                component_r411_proposal_score_threshold=float(
                    _m1(cfg, "V552R411_PROPOSAL_SCORE_THRESHOLD", 0.01)
                ),
                component_r411_initial_box_size=float(
                    _m1(cfg, "V552R411_INITIAL_BOX_SIZE", 0.08)
                ),
                component_r411_roi_size=int(
                    _m1(cfg, "V552R411_ROI_SIZE", 32)
                ),
                component_r411_roi_expand=float(
                    _m1(cfg, "V552R411_ROI_EXPAND", 1.50)
                ),
                component_r411_outside_penalty=float(
                    _m1(cfg, "V552R411_OUTSIDE_PENALTY", 8.0)
                ),
                component_canonical_shape_r412_enabled=bool(
                    _m1(cfg, "V552R412_ROOTFIX_ENABLED", False)
                ),
                component_r412_roi_size=int(
                    _m1(cfg, "V552R412_CANONICAL_ROI_SIZE", 64)
                ),
                component_r412_action_support_strength=float(
                    _m1(cfg, "V552R412_ACTION_SUPPORT_STRENGTH", 0.75)
                ),
                component_r412_action_support_floor=float(
                    _m1(cfg, "V552R412_ACTION_SUPPORT_FLOOR", 0.05)
                ),
                component_r412_boundary_band_kernel=int(
                    _m1(cfg, "V552R412_BOUNDARY_BAND_KERNEL", 7)
                ),
                component_geometry_lock_r413_enabled=bool(
                    _m1(cfg, "V552R413_ROOTFIX_ENABLED", False)
                ),
                component_r413_query_extent_enabled=bool(
                    _m1(cfg, "V552R413_QUERY_EXTENT_ENABLED", True)
                ),
                component_geometry_context_r414_enabled=bool(
                    _m1(cfg, "V552R414_ROOTFIX_ENABLED", False)
                ),
                component_r414_context_grid_size=int(
                    _m1(cfg, "V552R414_CONTEXT_GRID_SIZE", 17)
                ),
                component_r414_context_radius=float(
                    _m1(cfg, "V552R414_CONTEXT_RADIUS", 0.20)
                ),
                component_unique_point_r416_enabled=bool(
                    _m1(cfg, "V552R416_UNIQUE_POINT_TOPK_ENABLED", False)
                ),
                component_r416_cross_type_nms_radius_px=float(
                    _m1(cfg, "V552R416_CROSS_TYPE_NMS_RADIUS_PX", 4.0)
                ),
                component_r416_asymmetric_ltrb_enabled=bool(
                    _m1(cfg, "V552R416_ASYMMETRIC_LTRB_ENABLED", False)
                ),
                component_proposal_recovery_r417_enabled=bool(
                    _m1(cfg, "V552R417_ROOTFIX_ENABLED", False)
                ),
                component_r417_location_nms_kernel=int(
                    _m1(cfg, "V552R417_LOCATION_NMS_KERNEL", 3)
                ),
                component_r417_location_oversample_factor=int(
                    _m1(cfg, "V552R417_LOCATION_OVERSAMPLE_FACTOR", 4)
                ),
                component_r417_location_dedup_radius_px=float(
                    _m1(cfg, "V552R417_LOCATION_DEDUP_RADIUS_PX", 2.0)
                ),
                component_r417_shared_offset_enabled=bool(
                    _m1(cfg, "V552R417_SHARED_OFFSET_ENABLED", True)
                ),
                component_box_free_mask_set_r418_enabled=bool(
                    _m1(cfg, "V552R418_ROOTFIX_ENABLED", False)
                ),
                component_r418_paired_stable_teacher_enabled=bool(
                    _m1(cfg, "V552R418_PAIRED_STABLE_TEACHER_ENABLED", True)
                ),
                component_r418_paired_min_pixels=int(
                    _m1(cfg, "V552R418_PAIRED_MIN_PIXELS", 4)
                ),
                component_seeded_masked_attention_r419_enabled=bool(
                    _m1(cfg, "V552R419_ROOTFIX_ENABLED", False)
                ),
                component_r419_seed_radius_px=float(
                    _m1(cfg, "V552R419_SEED_RADIUS_PX", 12.0)
                ),
                component_r419_support_dilate_kernel=int(
                    _m1(cfg, "V552R419_SUPPORT_DILATE_KERNEL", 9)
                ),
                component_r419_outside_logit_penalty=float(
                    _m1(cfg, "V552R419_OUTSIDE_LOGIT_PENALTY", 8.0)
                ),
                component_r419_mask_threshold=float(
                    _m1(cfg, "V552R419_MASK_THRESHOLD", 0.5)
                ),
                component_dynamic_residual_mask_r420_enabled=bool(
                    _m1(cfg, "V552R420_ROOTFIX_ENABLED", False)
                ),
                component_r420_dynamic_channels=int(
                    _m1(cfg, "V552R420_DYNAMIC_CHANNELS", 8)
                ),
                component_r420_type_decoupled_mask_enabled=bool(
                    _m1(cfg, "V552R420_TYPE_DECOUPLED_MASK_ENABLED", False)
                ),
                component_r4201_clean_rootfix_enabled=bool(
                    _m1(cfg, "V552R4201_ROOTFIX_ENABLED", False)
                ),
                component_dense_competitive_residual_set_r4203_enabled=bool(
                    _m1(cfg, "V552R4203_ROOTFIX_ENABLED", False)
                ),
                component_factorized_residual_existence_identity_r4204_enabled=bool(
                    _m1(cfg, "V552R4204_ROOTFIX_ENABLED", False)
                ),
                component_r4204_spatial_identity_enabled=bool(
                    _m1(cfg, "V552R4204_SPATIAL_IDENTITY_ENABLED", False)
                ),
                component_capacity_consistent_overflow_r4205_enabled=bool(
                    _m1(cfg, "V552R4205_ROOTFIX_ENABLED", False)
                ),
                component_dynamic_visual_instance_binding_r4207_enabled=bool(
                    _m1(cfg, "V552R4207_ROOTFIX_ENABLED", False)
                ),
                component_normalized_visual_instance_binding_r4208_enabled=bool(
                    _m1(cfg, "V552R4208_NORMALIZED_FUSION_ENABLED", False)
                ),
                component_persistent_identity_r4208_enabled=bool(
                    _m1(cfg, "V552R4208_PERSISTENT_IDENTITY_ENABLED", False)
                ),
                component_instance_valid_factorization_r4210_enabled=bool(
                    _m1(cfg, "V552R4210_ROOTFIX_ENABLED", False)
                ),
                component_variable_cardinality_seeds_r4210_enabled=bool(
                    _m1(cfg, "V552R4210_VARIABLE_CARDINALITY_SEEDS_ENABLED", False)
                ),
                component_independent_overflow_gate_r4210_enabled=bool(
                    _m1(cfg, "V552R4210_INDEPENDENT_OVERFLOW_GATE_ENABLED", False)
                ),
                component_m1_native_alignment_r4210_enabled=bool(
                    _m1(cfg, "V552R4210_M1_NATIVE_ALIGNMENT_ENABLED", False)
                ),
                component_instance_valid_decoupling_r4211_enabled=bool(
                    _m1(cfg, "V552R4211_ROOTFIX_ENABLED", False)
                ),
                component_proposal_existence_decoupling_r4211_enabled=bool(
                    _m1(cfg, "V552R4211_PROPOSAL_EXISTENCE_DECOUPLING_ENABLED", False)
                ),
                component_geometry_overflow_decoupling_r4211_enabled=bool(
                    _m1(cfg, "V552R4211_GEOMETRY_OVERFLOW_DECOUPLING_ENABLED", False)
                ),
                component_independent_candidate_set_r4212_enabled=bool(
                    _m1(cfg, "V552R4212_INDEPENDENT_CANDIDATE_SET_ENABLED", False)
                ),
                component_disable_visual_seed_identity_r4212_enabled=bool(
                    _m1(cfg, "V552R4212_DISABLE_VISUAL_SEED_IDENTITY_ENABLED", False)
                ),
                component_existence_no_object_r4212_enabled=bool(
                    _m1(cfg, "V552R4212_EXISTENCE_NO_OBJECT_ENABLED", False)
                ),
                component_candidate_alignment_r4212_enabled=bool(
                    _m1(cfg, "V552R4212_CANDIDATE_ALIGNMENT_ENABLED", False)
                ),
                component_direct_delta_utility_r4212_enabled=bool(
                    _m1(cfg, "V552R4212_DIRECT_DELTA_UTILITY_ENABLED", False)
                ),
                component_zero_stop_one_step_r4212_enabled=bool(
                    _m1(cfg, "V552R4212_ZERO_STOP_ONE_STEP_ENABLED", False)
                ),
                component_clean_core_v560_enabled=(
                    self.clean_dynamic_component_set_enabled
                    or bool(_m1(cfg, "V560_CLEAN_CORE_ENABLED", False))
                ),
                component_base_conditioned_residual_set_v561_enabled=(
                    self.clean_dynamic_component_set_enabled
                    or bool(_m1(cfg, "V561_BCRS_ENABLED", False))
                ),
                component_bcrs_v561_variant=(
                    "persistent" if self.clean_dynamic_component_set_enabled
                    else str(_m1(cfg, "V561_BCRS_VARIANT", "typed"))
                ),
                component_v563_attention_radius=float(
                    _m1(cfg, "V563_ATTENTION_RADIUS", 0.24)
                ),
                component_v563_mask_radius=float(
                    _m1(cfg, "V563_MASK_RADIUS", 0.20)
                ),
                component_v563_identity_mix=float(
                    _m1(cfg, "V563_IDENTITY_MIX", 0.60)
                ),
                component_v563_query_residual_scale=float(
                    _m1(cfg, "V563_QUERY_RESIDUAL_SCALE", 0.15)
                ),
                component_v563_outside_logit_penalty=float(
                    _m1(cfg, "V563_OUTSIDE_LOGIT_PENALTY", 12.0)
                ),
                component_v564_rootfix_enabled=(
                    False if self.clean_dynamic_component_set_enabled
                    else bool(_m1(cfg, "V564_ROOTFIX_ENABLED", False))
                ),
                component_v564_attention_identity_scale=float(
                    _m1(cfg, "V564_ATTENTION_IDENTITY_SCALE", 0.35)
                ),
                component_v564_proposal_shape_scale=float(
                    _m1(cfg, "V564_PROPOSAL_SHAPE_SCALE", 0.50)
                ),
                component_v564_min_mask_radius=float(
                    _m1(cfg, "V564_MIN_MASK_RADIUS", 0.05)
                ),
                component_v564_max_mask_radius=float(
                    _m1(cfg, "V564_MAX_MASK_RADIUS", 0.20)
                ),
                component_v565_rootfix_enabled=(
                    False if self.clean_dynamic_component_set_enabled
                    else bool(_m1(cfg, "V565_ROOTFIX_ENABLED", False))
                ),
                component_v565_seed_nms_radius=float(
                    _m1(cfg, "V565_SEED_NMS_RADIUS", 0.025)
                ),
                component_v565_support_relative_threshold=float(
                    _m1(cfg, "V565_SUPPORT_RELATIVE_THRESHOLD", 0.35)
                ),
                component_v565_extent_quantile=float(
                    _m1(cfg, "V565_EXTENT_QUANTILE", 0.90)
                ),
                component_v565_attention_radius_scale=float(
                    _m1(cfg, "V565_ATTENTION_RADIUS_SCALE", 1.35)
                ),
                component_v565_max_attention_radius=float(
                    _m1(cfg, "V565_MAX_ATTENTION_RADIUS", 0.20)
                ),
                component_v565_shape_scale=float(
                    _m1(cfg, "V565_SHAPE_SCALE", 0.75)
                ),
                component_clean_dynamic_set_enabled=self.clean_dynamic_component_set_enabled,
                component_tc_drcs_enabled=self.tc_drcs_enabled,
                deploy_start_epoch=int(
                    _m1(
                        cfg,
                        "V538_DEPLOY_START_EPOCH",
                        _m1(
                            cfg,
                            "V537_RANKER_START_EPOCH",
                            _m1(
                                cfg,
                                "V536_DEPLOY_START_EPOCH",
                                int(_m1(cfg, "V532_M1_START_EPOCH", 0))
                                + int(_m1(cfg, "V532_M1_RAMP_EPOCHS", 10)),
                            ),
                        ),
                    )
                ),
                m1_grad_start_epoch=int(_m1(cfg, "V532_M1_GRAD_START_EPOCH", 8)),
                m1_grad_ramp_epochs=int(_m1(cfg, "V532_M1_GRAD_RAMP_EPOCHS", 12)),
                m1_grad_final_scale=float(_m1(cfg, "V532_M1_GRAD_FINAL_SCALE", 0.25)),
            )
            self.safe_deployer = None
        elif self.v531_enabled:
            self.pixel_composer = V531TypedSparseRefiner(
                hidden_dim=int(_m1(cfg, "V531_HIDDEN_DIM", hidden)),
                dropout=float(_m1(cfg, "V531_DROPOUT", 0.10)),
                semantic_channels=int(semantic_channels),
                semantic_dim=int(_m1(cfg, "V531_SEMANTIC_DIM", 32)),
                use_semantic=bool(
                    _m1(cfg, "V531_USE_SEMANTIC_FEATURES", self.v518_use_semantic)
                ),
                route_temperature=float(
                    _m1(cfg, "V531_ROUTE_TEMPERATURE", 0.50)
                ),
                risk_temperature=float(
                    _m1(cfg, "V531_RISK_TEMPERATURE", 0.25)
                ),
                harm_penalty=float(_m1(cfg, "V531_HARM_PENALTY", 1.50)),
                edit_penalty=float(_m1(cfg, "V531_EDIT_PENALTY", 0.02)),
                utility_threshold=float(
                    _m1(cfg, "V531_UTILITY_THRESHOLD", 0.0)
                ),
                preserve_bias=float(_m1(cfg, "V531_PRESERVE_BIAS", 1.25)),
                action_bias=float(_m1(cfg, "V531_ACTION_BIAS", 0.0)),
                initial_correctness=float(
                    _m1(cfg, "V531_INITIAL_CORRECTNESS", 0.50)
                ),
                hard_inference=bool(_m1(cfg, "V531_HARD_INFERENCE", True)),
                straight_through_train=bool(
                    _m1(cfg, "V531_STRAIGHT_THROUGH_TRAIN", False)
                ),
                detach_m1_inputs=bool(
                    _m1(cfg, "V531_DETACH_M1_INPUTS", True)
                ),
                max_action_alpha=float(
                    _m1(cfg, "V531_MAX_ACTION_ALPHA", 1.0)
                ),
            )
            self.safe_deployer = None
        elif self.v490_enabled:
            if self.v505_enabled:
                requested_steps = int(_m1(cfg, "V505_INTERACTION_STEPS", 1))
                single_step_safe = bool(
                    _m1(cfg, "V506_SINGLE_STEP_SAFE_MODE", True)
                )
                if single_step_safe and requested_steps != 1:
                    raise ValueError(
                        "V506_SINGLE_STEP_SAFE_MODE requires "
                        "M1.V505_INTERACTION_STEPS=1.  The previous four-step "
                        "M2-then-M3 replay used different current-mask states "
                        "after an M3 rejection and is intentionally disabled."
                    )
                selection_margin = float(
                    _m1(cfg, "V506_SELECTION_MARGIN", 0.05)
                )
                if self.v524_counterfactual_prompted_region:
                    self.pixel_composer = V524CounterfactualPromptedRegionComposer(
                        hidden_dim=int(_m1(cfg, "V524_M2_HIDDEN_DIM", 64)),
                        metadata_dim=int(_m1(cfg, "V524_METADATA_DIM", 16)),
                        semantic_channels=int(semantic_channels),
                        max_candidates=int(_m1(cfg, "V524_MAX_CANDIDATES", 64)),
                        family_count=int(_m1(cfg, "V524_FAMILY_COUNT", 5)),
                        action_count=int(_m1(cfg, "V524_ACTION_COUNT", 10)),
                        score_stride=int(_m1(cfg, "V524_SCORE_STRIDE", 2)),
                        region_grid_size=int(_m1(cfg, "V524_REGION_GRID_SIZE", 7)),
                        set_layers=int(_m1(cfg, "V524_SET_LAYERS", 2)),
                        set_heads=int(_m1(cfg, "V524_SET_HEADS", 4)),
                        set_ffn_dim=int(_m1(cfg, "V524_SET_FFN_DIM", 192)),
                        top_family_count=int(_m1(cfg, "V524_TOP_FAMILY_COUNT", 3)),
                        support_floor=float(_m1(cfg, "V524_SUPPORT_FLOOR", 1.0e-4)),
                        edit_epsilon=float(_m1(cfg, "V524_EDIT_EPS", 1.0e-4)),
                        route_temperature=float(_m1(cfg, "V524_ROUTE_TEMPERATURE", 0.50)),
                        lcb_beta=float(_m1(cfg, "V524_LCB_BETA", 0.50)),
                        harm_penalty=float(_m1(cfg, "V524_HARM_PENALTY", 1.50)),
                        family_log_weight=float(_m1(cfg, "V524_FAMILY_LOG_WEIGHT", 0.25)),
                        prompt_weight=float(_m1(cfg, "V524_PROMPT_WEIGHT", 0.50)),
                        region_accept_threshold=float(_m1(cfg, "V524_REGION_ACCEPT_THRESHOLD", 0.05)),
                        prompt_accept_threshold=float(_m1(cfg, "V524_PROMPT_ACCEPT_THRESHOLD", 0.35)),
                        refiner_gate_threshold=float(_m1(cfg, "V524_REFINER_GATE_THRESHOLD", 0.55)),
                        full_region_execution=bool(_m1(cfg, "V524_FULL_REGION_EXECUTION", True)),
                        case_guard_enabled=bool(_m1(cfg, "V524_CASE_GUARD_ENABLED", True)),
                        case_accept_threshold=float(_m1(cfg, "V524_CASE_ACCEPT_THRESHOLD", 0.0)),
                        case_harm_threshold=float(_m1(cfg, "V524_CASE_HARM_THRESHOLD", 0.10)),
                        deploy_start_epoch=int(_m1(cfg, "V524_M2_DEPLOY_START_EPOCH", 30)),
                        dropout=float(_m1(cfg, "V524_DROPOUT", 0.10)),
                        hard_inference=bool(_m1(cfg, "V524_HARD_INFERENCE", True)),
                        utility_aligned_score=bool(_m1(cfg, "V526_UTILITY_ALIGNED_SCORE", False)),
                        route_residual_scale=float(_m1(cfg, "V526_ROUTE_RESIDUAL_SCALE", 0.01)),
                        causal_hard_mask=bool(_m1(cfg, "V526_CAUSAL_HARD_MASK", False)),
                        causal_mask_threshold=float(_m1(cfg, "V526_CAUSAL_MASK_THRESHOLD", 0.35)),
                        causal_mask_start_epoch=int(_m1(cfg, "V526_CAUSAL_MASK_START_EPOCH", 0)),
                        force_preserve_output=bool(_m1(cfg, "V526_FORCE_PRESERVE_OUTPUT", False)),
                        oracle_preserving_validity=bool(
                            _m1(cfg, "V527_ORACLE_PRESERVING_VALIDITY", False)
                        ),
                        factorized_ranker=bool(
                            _m1(cfg, "V527_FACTORIZED_RANKER_ENABLED", False)
                        ),
                        editability_threshold=float(
                            _m1(cfg, "V527_EDITABILITY_THRESHOLD", 0.50)
                        ),
                        conditional_temperature=float(
                            _m1(cfg, "V527_CONDITIONAL_TEMPERATURE", 0.25)
                        ),
                        validation_expose_proposal=bool(
                            _m1(cfg, "V527_VALIDATION_USE_PROPOSAL", False)
                        ),
                        outcome_composer=bool(
                            _m1(cfg, "V528_OUTCOME_COMPOSER_ENABLED", False)
                        ),
                        outcome_temperature=float(
                            _m1(cfg, "V528_OUTCOME_TEMPERATURE", 0.10)
                        ),
                        outcome_lcb_beta=float(
                            _m1(cfg, "V528_OUTCOME_LCB_BETA", 0.50)
                        ),
                        outcome_min_gain=float(
                            _m1(cfg, "V528_MIN_PREDICTED_GAIN", 0.0)
                        ),
                        outcome_max_harm=float(
                            _m1(cfg, "V528_MAX_PREDICTED_HARM", 0.45)
                        ),
                        outcome_exploration_epochs=int(
                            _m1(cfg, "V528_EXPLORATION_EPOCHS", 3)
                        ),
                        v529_calibration_first_outcome=bool(
                            _m1(cfg, "V529_CALIBRATION_FIRST_OUTCOME_ENABLED", False)
                        ),
                        v529_selector_enabled=bool(
                            _m1(cfg, "V529_SELECTOR_ENABLED", False)
                        ),
                        v529_uncertainty_enabled=bool(
                            _m1(cfg, "V529_UNCERTAINTY_ENABLED", False)
                        ),
                        v529_execution_enabled=bool(
                            _m1(cfg, "V529_EXECUTION_ENABLED", False)
                        ),
                        v529_route_temperature=float(
                            _m1(cfg, "V529_ROUTE_TEMPERATURE", 1.0)
                        ),
                        v529_utility_scale=float(
                            _m1(cfg, "V529_UTILITY_SCALE", 0.005)
                        ),
                        v529_lcb_beta=float(
                            _m1(cfg, "V529_LCB_BETA", 0.0)
                        ),
                        v529_min_gain=float(
                            _m1(cfg, "V529_MIN_PREDICTED_GAIN", 0.0)
                        ),
                        v529_max_harm=float(
                            _m1(cfg, "V529_MAX_PREDICTED_HARM", 0.45)
                        ),
                        v530_probability_calibrated_outcome=bool(
                            _m1(cfg, "V530_PROBABILITY_CALIBRATED_OUTCOME_ENABLED", False)
                        ),
                        v530_selector_enabled=bool(
                            _m1(cfg, "V530_SELECTOR_ENABLED", False)
                        ),
                        v530_uncertainty_enabled=bool(
                            _m1(cfg, "V530_UNCERTAINTY_ENABLED", False)
                        ),
                        v530_execution_enabled=bool(
                            _m1(cfg, "V530_EXECUTION_ENABLED", False)
                        ),
                        v530_route_temperature=float(
                            _m1(cfg, "V530_ROUTE_TEMPERATURE", 1.0)
                        ),
                        v530_utility_scale=float(
                            _m1(cfg, "V530_UTILITY_SCALE", 0.005)
                        ),
                        v530_lcb_beta=float(
                            _m1(cfg, "V530_LCB_BETA", 0.0)
                        ),
                        v530_min_gain=float(
                            _m1(cfg, "V530_MIN_PREDICTED_GAIN", 0.0)
                        ),
                        v530_max_harm=float(
                            _m1(cfg, "V530_MAX_PREDICTED_HARM", 0.45)
                        ),
                        v530_base_fp_prior=float(
                            _m1(cfg, "V530_BASE_FP_PRIOR", 0.10)
                        ),
                        v530_base_fn_prior=float(
                            _m1(cfg, "V530_BASE_FN_PRIOR", 0.01)
                        ),
                        v530_add_fix_prior=float(
                            _m1(cfg, "V530_ADD_FIX_PRIOR", 0.35)
                        ),
                        v530_remove_fix_prior=float(
                            _m1(cfg, "V530_REMOVE_FIX_PRIOR", 0.25)
                        ),
                        logvar_min=float(_m1(cfg, "V526_LOGVAR_MIN", -14.0)),
                        logvar_max=float(_m1(cfg, "V526_LOGVAR_MAX", 2.0)),
                        logvar_init=float(_m1(cfg, "V526_LOGVAR_INIT", -10.0)),
                    )
                elif self.v523_sea_level_utility:
                    self.pixel_composer = V523SeaLevelUtilityComposer(
                        hidden_dim=int(_m1(cfg, "V523_M2_HIDDEN_DIM", 64)),
                        metadata_dim=int(_m1(cfg, "V523_METADATA_DIM", 16)),
                        semantic_channels=int(semantic_channels),
                        max_candidates=int(_m1(cfg, "V523_MAX_CANDIDATES", 64)),
                        score_stride=int(_m1(cfg, "V523_SCORE_STRIDE", 4)),
                        max_steps=int(_m1(cfg, "V523_MAX_STEPS", 1)),
                        support_floor=float(_m1(cfg, "V523_SUPPORT_FLOOR", 1.0e-4)),
                        edit_epsilon=float(_m1(cfg, "V523_EDIT_EPS", 1.0e-4)),
                        student_temperature=float(_m1(cfg, "V523_STUDENT_TEMPERATURE", 0.20)),
                        sea_level_margin=float(_m1(cfg, "V523_SEA_LEVEL_MARGIN", 0.05)),
                        top_gap_margin=float(_m1(cfg, "V523_TOP_GAP_MARGIN", 0.02)),
                        risk_threshold=float(_m1(cfg, "V523_RISK_THRESHOLD", 0.10)),
                        benefit_threshold=float(_m1(cfg, "V523_BENEFIT_THRESHOLD", 0.65)),
                        risk_penalty=float(_m1(cfg, "V523_RISK_PENALTY", 2.0)),
                        hard_change_only=bool(_m1(cfg, "V523_HARD_CHANGE_ONLY", True)),
                        coherence_kernel=int(_m1(cfg, "V523_COHERENCE_KERNEL", 5)),
                        coherence_threshold=float(_m1(cfg, "V523_COHERENCE_THRESHOLD", 0.40)),
                        coherence_alpha=float(_m1(cfg, "V523_COHERENCE_ALPHA", 0.25)),
                        initial_edit_probability=float(_m1(cfg, "V523_INITIAL_EDIT_PROBABILITY", 0.01)),
                        deploy_start_epoch=int(_m1(cfg, "V523_M2_DEPLOY_START_EPOCH", 5)),
                        full_deploy_epoch=int(_m1(cfg, "V523_M2_FULL_DEPLOY_EPOCH", 40)),
                        dropout=float(_m1(cfg, "V523_DROPOUT", 0.10)),
                        hard_inference=bool(_m1(cfg, "V523_HARD_INFERENCE", True)),
                    )
                elif self.v522_pwo_distilled_sequential:
                    self.pixel_composer = V522PWODistilledSequentialLocalComposer(
                        hidden_dim=int(_m1(cfg, "V522_M2_HIDDEN_DIM", 48)),
                        metadata_dim=int(_m1(cfg, "V522_METADATA_DIM", 16)),
                        semantic_channels=int(semantic_channels),
                        max_candidates=int(_m1(cfg, "V522_MAX_CANDIDATES", 64)),
                        score_stride=int(_m1(cfg, "V522_SCORE_STRIDE", 4)),
                        max_steps=int(_m1(cfg, "V522_MAX_STEPS", 3)),
                        support_floor=float(_m1(cfg, "V522_SUPPORT_FLOOR", 1.0e-4)),
                        edit_epsilon=float(_m1(cfg, "V522_EDIT_EPS", 1.0e-4)),
                        source_temperature=float(_m1(cfg, "V522_SOURCE_TEMPERATURE", 0.20)),
                        lcb_beta=float(_m1(cfg, "V522_LCB_BETA", 1.0)),
                        min_candidate_lcb=float(_m1(cfg, "V522_MIN_CANDIDATE_LCB", 0.0)),
                        gate_threshold=float(_m1(cfg, "V522_GATE_THRESHOLD", 0.55)),
                        component_threshold=float(_m1(cfg, "V522_COMPONENT_THRESHOLD", 0.30)),
                        component_growth_steps=int(_m1(cfg, "V522_COMPONENT_GROWTH_STEPS", 48)),
                        component_dilation_radius=int(_m1(cfg, "V522_COMPONENT_DILATION_RADIUS", 2)),
                        min_component_pixels=int(_m1(cfg, "V522_MIN_COMPONENT_PIXELS", 4)),
                        initial_edit_probability=float(_m1(cfg, "V522_INITIAL_EDIT_PROBABILITY", 0.02)),
                        deploy_start_epoch=int(_m1(cfg, "V522_M2_DEPLOY_START_EPOCH", 10)),
                        full_deploy_epoch=int(_m1(cfg, "V522_M2_FULL_DEPLOY_EPOCH", 30)),
                        dropout=float(_m1(cfg, "V522_DROPOUT", 0.10)),
                        hard_inference=bool(_m1(cfg, "V522_HARD_INFERENCE", True)),
                    )
                elif self.v521_candidate_conditional_region:
                    self.pixel_composer = V521CandidateConditionalRegionUtilityComposer(
                        hidden_dim=int(_m1(cfg, "V521_M2_HIDDEN_DIM", 64)),
                        metadata_dim=int(_m1(cfg, "V521_METADATA_DIM", 16)),
                        semantic_channels=int(semantic_channels),
                        max_candidates=int(_m1(cfg, "V521_MAX_CANDIDATES", 64)),
                        grid_size=int(_m1(cfg, "V521_REGION_GRID_SIZE", 4)),
                        support_floor=float(_m1(cfg, "V521_SUPPORT_FLOOR", 1.0e-4)),
                        edit_epsilon=float(_m1(cfg, "V521_EDIT_EPS", 1.0e-4)),
                        region_dilation_radius=int(_m1(cfg, "V521_REGION_DILATION_RADIUS", 1)),
                        active_region_threshold=float(_m1(cfg, "V521_ACTIVE_REGION_THRESHOLD", 1.0e-4)),
                        student_temperature=float(_m1(cfg, "V521_STUDENT_TEMPERATURE", 0.10)),
                        harm_penalty=float(_m1(cfg, "V521_HARM_PENALTY", 1.0)),
                        edit_penalty=float(_m1(cfg, "V521_EDIT_PENALTY", 0.02)),
                        base_utility_threshold=float(_m1(cfg, "V521_BASE_UTILITY_THRESHOLD", 0.002)),
                        harm_prior_weight=float(_m1(cfg, "V521_HARM_PRIOR_WEIGHT", 0.02)),
                        edit_prior_weight=float(_m1(cfg, "V521_EDIT_PRIOR_WEIGHT", 0.01)),
                        max_selected_regions=int(_m1(cfg, "V521_MAX_SELECTED_REGIONS", 4)),
                        deploy_start_epoch=int(_m1(cfg, "V521_M2_DEPLOY_START_EPOCH", 5)),
                        full_deploy_epoch=int(_m1(cfg, "V521_M2_FULL_DEPLOY_EPOCH", 20)),
                        dropout=float(_m1(cfg, "V521_DROPOUT", 0.10)),
                        hard_inference=bool(_m1(cfg, "V521_HARD_INFERENCE", True)),
                    )
                elif self.v520_gate_selector:
                    self.pixel_composer = V520GateSelectorRegionComposer(
                        hidden_dim=int(_m1(cfg, "V520_M2_HIDDEN_DIM", 48)),
                        metadata_dim=int(_m1(cfg, "V520_METADATA_DIM", 16)),
                        semantic_channels=int(semantic_channels),
                        max_candidates=int(_m1(cfg, "V520_MAX_CANDIDATES", 64)),
                        selector_temperature=float(
                            _m1(cfg, "V520_SELECTOR_TEMPERATURE", 0.35)
                        ),
                        gate_temperature=float(
                            _m1(cfg, "V520_GATE_TEMPERATURE", 1.0)
                        ),
                        harm_penalty=float(_m1(cfg, "V520_HARM_PENALTY", 0.50)),
                        edit_penalty=float(_m1(cfg, "V520_EDIT_PENALTY", 0.02)),
                        utility_margin=float(_m1(cfg, "V520_UTILITY_MARGIN", 0.0)),
                        support_floor=float(_m1(cfg, "V520_SUPPORT_FLOOR", 1.0e-4)),
                        edit_epsilon=float(_m1(cfg, "V520_EDIT_EPS", 1.0e-4)),
                        region_radius=int(_m1(cfg, "V520_REGION_RADIUS", 2)),
                        dropout=float(_m1(cfg, "V520_DROPOUT", 0.10)),
                        initial_edit_probability=float(
                            _m1(cfg, "V520_INITIAL_EDIT_PROBABILITY", 0.02)
                        ),
                        hard_inference=bool(_m1(cfg, "V520_HARD_INFERENCE", True)),
                    )
                elif self.v519_region_composer:
                    self.pixel_composer = V519FamilyAwareRegionComposer(
                        hidden_dim=int(_m1(cfg, "V519_M2_HIDDEN_DIM", 32)),
                        metadata_dim=int(_m1(cfg, "V519_METADATA_DIM", 16)),
                        semantic_channels=int(semantic_channels),
                        max_candidates=int(_m1(cfg, "V519_MAX_CANDIDATES", 64)),
                        temperature=float(_m1(cfg, "V519_ROUTE_TEMPERATURE", 0.35)),
                        lcb_kappa=float(_m1(cfg, "V519_LCB_KAPPA", 1.0)),
                        harm_penalty=float(_m1(cfg, "V519_HARM_PENALTY", 0.02)),
                        edit_penalty=float(_m1(cfg, "V519_EDIT_PENALTY", 0.002)),
                        utility_margin=float(_m1(cfg, "V519_UTILITY_MARGIN", 0.001)),
                        support_floor=float(_m1(cfg, "V519_SUPPORT_FLOOR", 1.0e-4)),
                        edit_epsilon=float(_m1(cfg, "V519_EDIT_EPS", 1.0e-4)),
                        region_radius=int(_m1(cfg, "V519_REGION_RADIUS", 2)),
                        dropout=float(_m1(cfg, "V519_DROPOUT", 0.10)),
                        hard_inference=bool(_m1(cfg, "V519_HARD_INFERENCE", True)),
                        straight_through_train=bool(
                            _m1(cfg, "V519_STRAIGHT_THROUGH_TRAIN", False)
                        ),
                    )
                else:
                    self.pixel_composer = V505InteractiveRegionPolicy(
                        hidden_dim=int(_m1(cfg, "V505_M2_HIDDEN_DIM", hidden)),
                        proposal_hidden_dim=int(_m1(cfg, "V505_M2_PROPOSAL_HIDDEN_DIM", 96)),
                        dropout=float(_m1(cfg, "V505_DROPOUT", 0.10)),
                        grid_size=int(_m1(cfg, "V505_INTERACTION_GRID_SIZE", 4)),
                        steps=requested_steps,
                        surface_weight=float(_m1(cfg, "V505_SURFACE_SCORE_WEIGHT", 0.5)),
                        pad_fraction=float(_m1(cfg, "V505_BOX_PAD_FRACTION", 0.125)),
                        min_edit_mass=float(_m1(cfg, "V505_MIN_EDIT_MASS", 1.0e-5)),
                        selection_margin=selection_margin,
                        include_full_box=bool(_m1(cfg, "V507_INCLUDE_FULL_BOX", True)),
                        strict_deploy_valid=bool(
                            _m1(cfg, "V507_STRICT_DEPLOY_VALID", False)
                        ),
                    )
                self.safe_deployer = V505RegionVerifier(
                    hidden_dim=int(_m1(cfg, "V505_M3_HIDDEN_DIM", max(hidden // 2, 32))),
                    proposal_hidden_dim=int(_m1(cfg, "V505_M3_PROPOSAL_HIDDEN_DIM", 96)),
                    dropout=float(_m1(cfg, "V505_DROPOUT", 0.10)),
                    surface_weight=float(_m1(cfg, "V505_SURFACE_SCORE_WEIGHT", 0.5)),
                    min_edit_mass=float(_m1(cfg, "V505_MIN_EDIT_MASS", 1.0e-5)),
                    selection_margin=selection_margin,
                    strict_deploy_valid=bool(
                        _m1(cfg, "V507_STRICT_DEPLOY_VALID", False)
                    ),
                )
            else:
                if self.v492_enabled:
                    self.pixel_composer = CausalLocalPixelEditor(
                        cfg,
                        hidden_dim=int(_m1(cfg, "V492_M2_HIDDEN_DIM", _m1(cfg, "V489_M2_HIDDEN_DIM", hidden))),
                        semantic_channels=int(semantic_channels),
                    )
                else:
                    self.pixel_composer = AsymmetricSparseCounterfactualComposer(
                        cfg,
                        hidden_dim=int(_m1(cfg, "V489_M2_HIDDEN_DIM", hidden)),
                        semantic_channels=int(semantic_channels),
                    )
                if self.v491_enabled:
                    self.safe_deployer = PreserveFirstBinaryRiskDeployer(
                        cfg,
                        hidden_dim=int(
                            _m1(
                                cfg,
                                "V491_M3_HIDDEN_DIM",
                                _m1(cfg, "V490_M3_HIDDEN_DIM", max(hidden // 2, 32)),
                            )
                        ),
                    )
                else:
                    self.safe_deployer = DenseCoherentRelativeRiskSelector(
                        cfg,
                        hidden_dim=int(_m1(cfg, "V490_M3_HIDDEN_DIM", max(hidden // 2, 32))),
                    )
        elif self.v489_enabled:
            self.pixel_composer = AsymmetricSparseCounterfactualComposer(
                cfg,
                hidden_dim=int(_m1(cfg, "V489_M2_HIDDEN_DIM", hidden)),
                semantic_channels=int(semantic_channels),
            )
            self.safe_deployer = RegionBestExpertSelector(
                cfg,
                hidden_dim=int(_m1(cfg, "V489_M3_HIDDEN_DIM", max(hidden // 2, 32))),
            )
        elif self.v488_enabled:
            self.pixel_composer = PixelCounterfactualComposer(
                cfg,
                hidden_dim=int(_m1(cfg, "V488_M2_HIDDEN_DIM", hidden)),
                semantic_channels=int(semantic_channels),
            )
            self.safe_deployer = UncertaintyCalibratedSafeDeployer(
                cfg,
                hidden_dim=int(_m1(cfg, "V488_M3_HIDDEN_DIM", max(hidden // 2, 16))),
            )
        else:
            self.pixel_composer = None
            self.safe_deployer = None

    def _forward_v532(
        self,
        *,
        image: torch.Tensor,
        c0_logits: torch.Tensor,
        c0_prob: torch.Tensor,
        err: Dict[str, torch.Tensor],
        loc: Dict[str, torch.Tensor],
        semantic_map: Optional[torch.Tensor],
        supervision_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.pixel_composer is None:
            raise RuntimeError("V532 is enabled but pixel_composer is absent.")

        primary_logits = loc.get(
            "local_primary_candidate_logits",
            loc["local_candidate_logits"][:, :4],
        )
        primary_probs = torch.sigmoid(primary_logits).clamp(EPS, 1.0 - EPS)
        primary_alpha = loc.get("local_primary_action_alpha")
        if not isinstance(primary_alpha, torch.Tensor):
            neg_delete = (1.0 - primary_probs[:, 0:1] / c0_prob).clamp(0.0, 1.0)
            pos_fill = ((primary_probs[:, 1:2] - c0_prob) / (1.0 - c0_prob)).clamp(0.0, 1.0)
            neg_trim = (1.0 - primary_probs[:, 2:3] / c0_prob).clamp(0.0, 1.0)
            pos_expand = ((primary_probs[:, 3:4] - c0_prob) / (1.0 - c0_prob)).clamp(0.0, 1.0)
            primary_alpha = torch.cat(
                [neg_delete, pos_fill, neg_trim, pos_expand], dim=1
            )

        refined = self.pixel_composer(
            image=image,
            c0_prob=c0_prob,
            cause_map_probs=err["error_cause_map_probs"],
            action_alpha=primary_alpha,
            semantic_map=semantic_map,
            supervision_masks=supervision_masks,
        )

        action_probs = refined["v532_action_candidate_probs"]
        candidate_probs = torch.cat([c0_prob, action_probs], dim=1)
        candidate_logits = torch.logit(candidate_probs.clamp(EPS, 1.0 - EPS))
        b, slots = candidate_probs.shape[:2]
        route_case = F.adaptive_avg_pool2d(
            refined["m2_route_probs"], 1
        ).flatten(1)
        hard_case = F.one_hot(
            route_case.argmax(dim=1), num_classes=5
        ).to(route_case.dtype)
        deployed = refined["m2_fused_probs"]
        trainable_candidate_mask = torch.ones(
            slots, dtype=torch.bool, device=candidate_probs.device
        )
        trainable_candidate_mask[0] = False
        aux: Dict[str, torch.Tensor] = {
            **err,
            **loc,
            **refined,
            "candidates": candidate_logits,
            "candidate_logits": candidate_logits,
            "candidate_probs": candidate_probs,
            "c0_prob_detached": c0_prob,
            "base_probs_detached": c0_prob,
            "c0_prob_online": c0_prob,
            "candidate_family": torch.arange(
                slots, device=candidate_probs.device
            ).view(1, -1).expand(b, -1),
            "candidate_source_v485": candidate_probs.new_ones((b,)),
            "v484_enabled": candidate_probs.new_ones((b,)),
            "v485_enabled": candidate_probs.new_ones((b,)),
            "v20_action_types": torch.arange(
                4, device=candidate_probs.device, dtype=torch.long
            ),
            "v20_selector_hard": hard_case[:, 1:],
            "m1_selector_soft": route_case,
            "m1_selector_hard": hard_case,
            "m1_choice_logits": route_case.clamp_min(EPS).log(),
            "v20_hard_fused_probs": deployed,
            "m1_hard_fused_probs": deployed,
            "direct_fused_probs": deployed,
            "router_fused_probs": deployed,
            "text_verifier_fused_probs": deployed,
            "fused_probs": deployed,
            "final_probs": deployed,
            "v488_enabled": candidate_probs.new_zeros((b,)),
            "v489_enabled": candidate_probs.new_zeros((b,)),
            "v492_enabled": candidate_probs.new_zeros((b,)),
            "v493_enabled": candidate_probs.new_zeros((b,)),
            "v494_enabled": candidate_probs.new_zeros((b,)),
            "v495_enabled": candidate_probs.new_zeros((b,)),
            "v501_enabled": candidate_probs.new_zeros((b,)),
            "v502_enabled": candidate_probs.new_zeros((b,)),
            "v503_enabled": candidate_probs.new_ones((b,)),
            "v505_enabled": candidate_probs.new_zeros((b,)),
            "v518_enabled": candidate_probs.new_full((b,), float(self.v518_enabled)),
            "v519_enabled": candidate_probs.new_zeros((b,)),
            "v521_enabled": candidate_probs.new_zeros((b,)),
            "v522_enabled": candidate_probs.new_zeros((b,)),
            "v523_enabled": candidate_probs.new_zeros((b,)),
            "v524_enabled": candidate_probs.new_zeros((b,)),
            "v531_enabled": candidate_probs.new_zeros((b,)),
            "v532_enabled": candidate_probs.new_ones((b,)),
            "v518_trainable_candidate_mask": trainable_candidate_mask,
            "v518_local_candidate_count": candidate_probs.new_full((b,), 4.0),
            "v518_pair_candidate_count": candidate_probs.new_zeros((b,)),
            "v518_morph_candidate_count": candidate_probs.new_zeros((b,)),
            "v518_global_candidate_count": candidate_probs.new_zeros((b,)),
            "v518_candidate_supports": torch.cat(
                [torch.zeros_like(c0_prob), primary_alpha], dim=1
            ),
            "m2_mc_variance": torch.zeros_like(c0_prob),
            "m2_mc_std": torch.zeros_like(c0_prob),
            "m2_mc_samples": candidate_probs.new_ones((b,)),
        }
        aux["v518_pair_candidate_names"] = tuple()
        aux["v518_morph_candidate_names"] = tuple()
        return aux

    def _forward_v531(
        self,
        *,
        image: torch.Tensor,
        c0_logits: torch.Tensor,
        c0_prob: torch.Tensor,
        err: Dict[str, torch.Tensor],
        loc: Dict[str, torch.Tensor],
        semantic_map: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if self.pixel_composer is None:
            raise RuntimeError("V531 is enabled but pixel_composer is absent.")

        primary_logits = loc.get(
            "local_primary_candidate_logits",
            loc["local_candidate_logits"][:, :4],
        )
        primary_probs = torch.sigmoid(primary_logits).clamp(EPS, 1.0 - EPS)
        primary_alpha = loc.get("local_primary_action_alpha")
        if not isinstance(primary_alpha, torch.Tensor):
            # Exact monotone alpha recovery from the four primary candidates.
            neg_delete = (1.0 - primary_probs[:, 0:1] / c0_prob).clamp(0.0, 1.0)
            pos_fill = ((primary_probs[:, 1:2] - c0_prob) / (1.0 - c0_prob)).clamp(0.0, 1.0)
            neg_trim = (1.0 - primary_probs[:, 2:3] / c0_prob).clamp(0.0, 1.0)
            pos_expand = ((primary_probs[:, 3:4] - c0_prob) / (1.0 - c0_prob)).clamp(0.0, 1.0)
            primary_alpha = torch.cat(
                [neg_delete, pos_fill, neg_trim, pos_expand], dim=1
            )

        m2 = self.pixel_composer(
            image=image,
            c0_prob=c0_prob,
            cause_map_probs=err["error_cause_map_probs"],
            action_alpha=primary_alpha,
            semantic_map=(
                semantic_map.detach()
                if isinstance(semantic_map, torch.Tensor)
                else semantic_map
            ),
        )

        candidate_logits = torch.cat([c0_logits, primary_logits], dim=1)
        candidate_probs = torch.sigmoid(candidate_logits).clamp(EPS, 1.0 - EPS)
        b, slots = candidate_probs.shape[:2]
        preserve_case = candidate_probs.new_zeros((b, slots))
        preserve_case[:, 0] = 1.0
        route_case = F.adaptive_avg_pool2d(
            m2["m2_route_probs"], 1
        ).flatten(1)
        hard_case = F.one_hot(
            route_case.argmax(dim=1), num_classes=5
        ).to(route_case.dtype)
        deployed = m2["m2_fused_probs"]
        action_count = max(slots - 1, 0)
        trainable_candidate_mask = torch.ones(
            slots, dtype=torch.bool, device=candidate_probs.device
        )
        trainable_candidate_mask[0] = False
        aux: Dict[str, torch.Tensor] = {
            **err,
            **loc,
            **m2,
            "candidates": candidate_logits,
            "candidate_logits": candidate_logits,
            "candidate_probs": candidate_probs,
            "c0_prob_detached": c0_prob,
            "base_probs_detached": c0_prob,
            "c0_prob_online": c0_prob,
            "candidate_family": torch.arange(
                slots, device=candidate_probs.device
            ).view(1, -1).expand(b, -1),
            "candidate_source_v485": candidate_probs.new_ones((b,)),
            "v484_enabled": candidate_probs.new_ones((b,)),
            "v485_enabled": candidate_probs.new_ones((b,)),
            "v20_action_types": torch.arange(
                action_count,
                device=candidate_probs.device,
                dtype=torch.long,
            ),
            "v20_selector_hard": hard_case[:, 1:],
            "m1_selector_soft": route_case,
            "m1_selector_hard": hard_case,
            "m1_choice_logits": route_case.clamp_min(EPS).log(),
            "v20_hard_fused_probs": deployed,
            "m1_hard_fused_probs": deployed,
            "direct_fused_probs": deployed,
            "router_fused_probs": deployed,
            "text_verifier_fused_probs": deployed,
            "fused_probs": deployed,
            "final_probs": deployed,
            "v488_enabled": candidate_probs.new_zeros((b,)),
            "v489_enabled": candidate_probs.new_full((b,), float(self.v489_enabled)),
            "v492_enabled": candidate_probs.new_full((b,), float(self.v492_enabled)),
            "v493_enabled": candidate_probs.new_full((b,), float(self.v493_enabled)),
            "v494_enabled": candidate_probs.new_full((b,), float(self.v494_enabled)),
            "v495_enabled": candidate_probs.new_full((b,), float(self.v495_enabled)),
            "v501_enabled": candidate_probs.new_full((b,), float(self.v501_enabled)),
            "v502_enabled": candidate_probs.new_full((b,), float(self.v502_enabled)),
            "v503_enabled": candidate_probs.new_full((b,), float(self.v503_enabled)),
            "v505_enabled": candidate_probs.new_zeros((b,)),
            "v518_enabled": candidate_probs.new_full((b,), float(self.v518_enabled)),
            "v519_enabled": candidate_probs.new_zeros((b,)),
            "v521_enabled": candidate_probs.new_zeros((b,)),
            "v522_enabled": candidate_probs.new_zeros((b,)),
            "v523_enabled": candidate_probs.new_zeros((b,)),
            "v524_enabled": candidate_probs.new_zeros((b,)),
            "v531_enabled": candidate_probs.new_ones((b,)),
            "v518_trainable_candidate_mask": trainable_candidate_mask,
            "v518_local_candidate_count": candidate_probs.new_full((b,), 4.0),
            "v518_pair_candidate_count": candidate_probs.new_zeros((b,)),
            "v518_morph_candidate_count": candidate_probs.new_zeros((b,)),
            "v518_global_candidate_count": candidate_probs.new_zeros((b,)),
            "v518_candidate_supports": torch.cat(
                [torch.zeros_like(c0_prob), primary_alpha], dim=1
            ),
            "m2_mc_variance": torch.zeros_like(c0_prob),
            "m2_mc_std": torch.zeros_like(c0_prob),
            "m2_mc_samples": candidate_probs.new_ones((b,)),
            "v531_preserve_case_prior": preserve_case,
        }
        aux["v518_pair_candidate_names"] = tuple()
        aux["v518_morph_candidate_names"] = tuple()
        return aux

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)
        if self.pixel_composer is not None and hasattr(self.pixel_composer, "set_epoch"):
            self.pixel_composer.set_epoch(epoch)
        if self.safe_deployer is not None and hasattr(self.safe_deployer, "set_epoch"):
            self.safe_deployer.set_epoch(epoch)

    def _v518_teacher_weight(self) -> float:
        # M2-only must observe exactly the same frozen M1 candidate bank during
        # training and deployment. Ground-truth cause injection is forbidden.
        if (
            bool(_m1(self.cfg, "V519_M2_ONLY", False))
            or bool(_m1(self.cfg, "V524_M2_ONLY", False))
            or bool(_m1(self.cfg, "V526_DISABLE_GT_TEACHER", False))
        ):
            return 0.0
        if not self.v518_enabled or not self.training:
            return 0.0
        full_epochs = max(int(_m1(self.cfg, "V518_TEACHER_FULL_EPOCHS", 5)), 0)
        end_epoch = max(
            int(_m1(self.cfg, "V518_TEACHER_END_EPOCH", 18)),
            full_epochs + 1,
        )
        epoch = int(self.current_epoch)
        if epoch < full_epochs:
            return 1.0
        if epoch >= end_epoch:
            return 0.0
        return float(end_epoch - epoch) / float(end_epoch - full_epochs)

    def forward(
        self,
        image: torch.Tensor,
        base_logits: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        semantic_map: Optional[torch.Tensor] = None,
        supervision_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # V489/V490 are truly end-to-end from Final→M3→M2→M1→Base/PVL.
        # Older V488 runs retain their detached factual-anchor contract.
        c0_logits_online = _as_b1hw(base_logits)
        c0_logits = (
            c0_logits_online.detach()
            if (
                self.v501_enabled
                or (self.v532_enabled and self.v533_detach_coarse_anchor)
            )
            else (
                c0_logits_online
                if (self.v489_enabled or self.v490_enabled or self.v532_enabled)
                else c0_logits_online.detach()
            )
        )
        c0_prob = torch.sigmoid(c0_logits).clamp(EPS, 1.0 - EPS)
        # V547: M1/M2 must observe the same factual anchor that is later used
        # to build component teachers and exact gain labels.  Replay is training
        # only; validation and Test always use the unmodified Base probability.
        refiner_c0_prob = c0_prob
        v547_replay = None
        if (
            self.v532_enabled
            and bool(_m1(self.cfg, "V538_ONLINE_COMPONENT_REFINER_ENABLED", False))
            and self.training
            and isinstance(supervision_masks, torch.Tensor)
            and bool(_m1(self.cfg, "V547_PAIRED_RESIDUAL_REPLAY_ENABLED", False))
        ):
            v547_replay = build_v547_paired_anchor(
                base_probability=c0_prob,
                target=supervision_masks,
                epoch=int(self.current_epoch),
                residual_floor=float(
                    _m1(self.cfg, "V547_REPLAY_RESIDUAL_FLOOR", 0.0025)
                ),
                radius_min=int(_m1(self.cfg, "V547_REPLAY_RADIUS_MIN", 1)),
                radius_max=int(_m1(self.cfg, "V547_REPLAY_RADIUS_MAX", 3)),
                force_confidence=float(
                    _m1(self.cfg, "V547_REPLAY_FORCE_CONFIDENCE", 0.90)
                ),
            )
            refiner_c0_prob = v547_replay.probability

        semantic_for_m1 = (
            semantic_map.detach()
            if self.v518_use_semantic and isinstance(semantic_map, torch.Tensor)
            else semantic_map
        )
        if self.v503_enabled:
            err = self.error_state_head(
                image, refiner_c0_prob, semantic_map=semantic_for_m1
            )
        else:
            err = self.error_state_head(image, refiner_c0_prob)

        teacher_targets = None
        teacher_weight = self._v518_teacher_weight()
        if (
            self.v518_enabled
            and self.training
            and isinstance(supervision_masks, torch.Tensor)
            and teacher_weight > 0.0
        ):
            teacher_targets = build_factual_cause_targets(
                refiner_c0_prob.detach(),
                supervision_masks.detach(),
                boundary_radius=int(
                    _m1(self.cfg, "V503_FACTUAL_BOUNDARY_RADIUS", 2)
                ),
                failure_dice=float(_m1(self.cfg, "V503_FAILURE_DICE", 0.55)),
                include_global_action=False,
            )["cause_targets"]

        if self.v503_enabled:
            loc = self.local_generator(
                image,
                refiner_c0_prob,
                err["error_cause_map_probs"],
                semantic_map=semantic_for_m1,
                teacher_cause_targets=teacher_targets,
                teacher_weight=teacher_weight,
            )
        else:
            loc = self.local_generator(
                image, refiner_c0_prob, err["error_state_probs"]
            )
        if self.v532_enabled:
            output = self._forward_v532(
                image=image,
                c0_logits=torch.logit(refiner_c0_prob.clamp(EPS, 1.0 - EPS)),
                c0_prob=refiner_c0_prob,
                err=err,
                loc=loc,
                semantic_map=semantic_map,
                supervision_masks=supervision_masks,
            )
            batch_size = refiner_c0_prob.shape[0]
            output["v547_original_base_prob"] = c0_prob.detach()
            output["v547_refiner_anchor_prob"] = refiner_c0_prob.detach()
            if v547_replay is None:
                output["v547_replay_case"] = torch.zeros(
                    batch_size, device=c0_prob.device, dtype=torch.bool
                )
                output["v547_replay_type"] = torch.full(
                    (batch_size,), -1, device=c0_prob.device, dtype=torch.long
                )
                output["v547_replay_radius"] = torch.zeros(
                    batch_size, device=c0_prob.device, dtype=torch.long
                )
                hard_base = (c0_prob.detach() >= 0.5).to(c0_prob.dtype)
                gt = (_as_b1hw(supervision_masks).detach() >= 0.5).to(c0_prob.dtype) \
                    if isinstance(supervision_masks, torch.Tensor) else hard_base
                residual = (hard_base != gt).to(c0_prob.dtype).flatten(1).mean(dim=1)
                output["v547_real_residual_fraction"] = residual
                output["v547_synthetic_residual_fraction"] = residual
            else:
                output["v547_replay_case"] = v547_replay.replay_case
                output["v547_replay_type"] = v547_replay.replay_type
                output["v547_replay_radius"] = v547_replay.replay_radius
                output["v547_real_residual_fraction"] = (
                    v547_replay.real_residual_fraction
                )
                output["v547_synthetic_residual_fraction"] = (
                    v547_replay.synthetic_residual_fraction
                )
            return output
        if self.v531_enabled:
            return self._forward_v531(
                image=image,
                c0_logits=c0_logits,
                c0_prob=c0_prob,
                err=err,
                loc=loc,
                semantic_map=semantic_for_m1,
            )
        if self.global_generator is not None:
            glob = self.global_generator(
                image,
                c0_prob,
                err["error_state_probs"],
                image_features=(
                    image_features.detach()
                    if self.v501_enabled and isinstance(image_features, torch.Tensor)
                    else image_features
                ),
                text_features=(
                    text_features.detach()
                    if self.v501_enabled and isinstance(text_features, torch.Tensor)
                    else text_features
                ),
            )
        else:
            batch, _, height, width = c0_prob.shape
            glob = {
                "global_candidate_logits": c0_prob.new_empty(
                    (batch, 0, height, width)
                ),
                "global_candidate_probs": c0_prob.new_empty(
                    (batch, 0, height, width)
                ),
                "global_direct_probs": c0_prob.new_empty(
                    (batch, 0, height, width)
                ),
                "global_active_mask": torch.zeros(
                    batch,
                    0,
                    dtype=torch.bool,
                    device=c0_prob.device,
                ),
                "global_soft_gates": c0_prob.new_empty((batch, 0)),
            }
        local_logits = loc["local_candidate_logits"]
        primary_local_logits = loc.get(
            "local_primary_candidate_logits", local_logits[:, :4]
        )
        primary_local_probs = torch.sigmoid(primary_local_logits).clamp(
            EPS, 1.0 - EPS
        )

        pair_probs = c0_prob.new_empty(
            (c0_prob.shape[0], 0, *c0_prob.shape[-2:])
        )
        pair_names: Tuple[str, ...] = tuple()
        if self.v518_enabled and self.v518_include_pairs:
            pair_probs, pair_names = _v518_pair_bank(
                c0_prob, primary_local_probs
            )
        pair_logits = torch.logit(pair_probs.clamp(EPS, 1.0 - EPS))

        morph_probs = c0_prob.new_empty(
            (c0_prob.shape[0], 0, *c0_prob.shape[-2:])
        )
        morph_names: Tuple[str, ...] = tuple()
        if self.v518_enabled and self.v518_morph_radii:
            morph_probs, morph_names = _v518_morphology_bank(
                c0_prob,
                self.v518_morph_radii,
                include_erode=self.v518_include_erode,
                include_dilate=self.v518_include_dilate,
                include_open=self.v518_include_open,
                include_close=self.v518_include_close,
            )
        morph_logits = torch.logit(morph_probs.clamp(EPS, 1.0 - EPS))

        global_logits = glob["global_candidate_logits"]
        candidate_logits = torch.cat(
            [c0_logits, local_logits, pair_logits, morph_logits, global_logits],
            dim=1,
        )
        candidate_probs = torch.sigmoid(candidate_logits).clamp(EPS, 1.0 - EPS)
        k_local = min(4, candidate_probs.shape[1] - 1)
        local_probs = candidate_probs[:, 1:1 + k_local]

        local_count = int(local_logits.shape[1])
        pair_count = int(pair_logits.shape[1])
        morph_count = int(morph_logits.shape[1])
        global_count = int(global_logits.shape[1])
        trainable_candidate_mask = torch.zeros(
            candidate_probs.shape[1],
            dtype=torch.bool,
            device=candidate_probs.device,
        )
        # Preserve is not an intervention. Learned local/pair/global candidates
        # are differentiable; deterministic morphology candidates are not.
        trainable_candidate_mask[1:1 + local_count + pair_count] = True
        if global_count > 0:
            trainable_candidate_mask[-global_count:] = True
        if self.local_verifier is not None:
            local_m2 = self.local_verifier(
                image,
                c0_prob,
                local_probs,
                loc["local_supports"][:, :k_local],
            )
        else:
            local_m2 = {}

        if self.v505_enabled:
            if self.pixel_composer is None or self.safe_deployer is None:
                raise RuntimeError("V505 is enabled but its M2/M3 modules are absent.")
            preserve_support = torch.zeros_like(c0_prob)
            learned_supports = loc["local_supports"][:, :local_count]
            support_stack = torch.cat(
                [preserve_support, learned_supports], dim=1
            )
            if support_stack.shape[1] < candidate_probs.shape[1]:
                extra_support = (
                    candidate_probs[:, support_stack.shape[1]:] - c0_prob
                ).abs()
                support_stack = torch.cat([support_stack, extra_support], dim=1)

            # Base/M1 tensors are observations for the interaction policy.  M2/M3
            # objectives cannot change their own counterfactual labels by moving
            # Base or the candidates.
            if self.v519_region_composer:
                metadata = _v519_candidate_metadata(
                    c0_prob=c0_prob,
                    local_action_ids=loc["local_candidate_action_ids"],
                    local_dose_values=loc["local_candidate_dose_values"],
                    pair_count=pair_count,
                    morph_names=morph_names,
                    global_count=global_count,
                    exclude_erode_r6=self.v519_exclude_erode_r6,
                    deploy_pair_candidates=self.v519_deploy_pair_candidates,
                    deploy_global_candidates=self.v519_deploy_global_candidates,
                )
                cause_stack = _v519_candidate_cause_stack(
                    c0_prob=c0_prob,
                    cause_maps=err["error_cause_map_probs"].detach(),
                    local_action_ids=loc["local_candidate_action_ids"],
                    pair_count=pair_count,
                    morph_names=morph_names,
                    global_count=global_count,
                    failure_prob=err["failure_state_prob"].detach(),
                )
                m2_mean_dict = self.pixel_composer(
                    image=image,
                    c0_prob=c0_prob.detach(),
                    candidate_probs=candidate_probs.detach(),
                    supports=support_stack.detach(),
                    candidate_causes=cause_stack.detach(),
                    family_ids=metadata["family_ids"],
                    action_ids=metadata["action_ids"],
                    dose_values=metadata["dose_values"],
                    radius_values=metadata["radius_values"],
                    deploy_mask=metadata["deploy_mask"],
                    semantic_map=(
                        semantic_map.detach()
                        if isinstance(semantic_map, torch.Tensor)
                        else semantic_map
                    ),
                )
                # A2 deploys M2 directly. M3 remains a future safety stage and
                # is not allowed to alter the A2 result.
                deployed = _as_b1hw(m2_mean_dict["m2_fused_probs"])[:, 0]
                m3 = {
                    "fused_probs": deployed,
                    "final_probs": deployed,
                    "v519_m3_bypassed": candidate_probs.new_ones(
                        (candidate_probs.shape[0],)
                    ),
                }
                metadata_prefix = (
                    "v524" if self.v524_counterfactual_prompted_region
                    else "v523" if self.v523_sea_level_utility
                    else "v522" if self.v522_pwo_distilled_sequential
                    else "v521" if self.v521_candidate_conditional_region
                    else "v520" if self.v520_gate_selector
                    else "v519"
                )
                m2_mean_dict.update({
                    f"{metadata_prefix}_family_ids": metadata["family_ids"],
                    f"{metadata_prefix}_action_ids": metadata["action_ids"],
                    f"{metadata_prefix}_dose_values": metadata["dose_values"],
                    f"{metadata_prefix}_radius_values": metadata["radius_values"],
                    f"{metadata_prefix}_deploy_mask": metadata["deploy_mask"],
                    f"{metadata_prefix}_candidate_causes": cause_stack,
                })
            else:
                m2_mean_dict = self.pixel_composer(
                    image=image,
                    c0_prob=c0_prob.detach(),
                    candidate_probs=candidate_probs.detach(),
                    supports=support_stack.detach(),
                    cause_map_probs=err["error_cause_map_probs"].detach(),
                )
                m3 = self.safe_deployer(
                    image=image,
                    c0_prob=c0_prob.detach(),
                    candidate_probs=candidate_probs.detach(),
                    supports=support_stack.detach(),
                    cause_map_probs=err["error_cause_map_probs"].detach(),
                    m2_output=m2_mean_dict,
                )
                m2_mean_dict["m2_convex_probs"] = m2_mean_dict["m2_fused_probs"]
                m2_mean_dict["m2_proposal_probs"] = m2_mean_dict["m2_fused_probs"]
                m2_mean_dict["m2_training_probs"] = m2_mean_dict["m2_fused_probs"]
            m2_mean_dict["m2_mc_variance"] = torch.zeros_like(c0_prob)
            m2_mean_dict["m2_mc_std"] = torch.zeros_like(c0_prob)
            m2_mean_dict["m2_mc_samples"] = candidate_probs.new_ones(
                (candidate_probs.shape[0],)
            )
        elif self.v490_enabled or self.v489_enabled or self.v488_enabled:
            if self.pixel_composer is None or self.safe_deployer is None:
                raise RuntimeError("V488/V489 is enabled but M2/M3 modules are absent.")
            preserve_support = torch.zeros_like(c0_prob)
            support_stack = torch.cat([preserve_support, loc["local_supports"][:, :k_local]], dim=1)
            if support_stack.shape[1] < candidate_probs.shape[1]:
                extra_support = (candidate_probs[:, support_stack.shape[1]:] - c0_prob).abs()
                support_stack = torch.cat([support_stack, extra_support], dim=1)

            if self.v490_enabled or self.v489_enabled:
                train_mc = max(1, int(_m1(self.cfg, "V489_M2_TRAIN_MC_SAMPLES", 1)))
                eval_mc = max(1, int(_m1(self.cfg, "V489_M2_EVAL_MC_SAMPLES", 4)))
            else:
                train_mc = max(1, int(_m1(self.cfg, "V488_M2_TRAIN_MC_SAMPLES", 2)))
                eval_mc = max(1, int(_m1(self.cfg, "V488_M3_MC_SAMPLES", 8)))
            mc_samples = train_mc if self.training else eval_mc

            composer_runs = []
            for _ in range(mc_samples):
                if self.v490_enabled or self.v489_enabled:
                    # V501 assigns one owner to every gradient path. M2 observes
                    # the current M1 candidates and frozen semantic features but
                    # cannot update Base/PVL or M1 through its own objective.
                    if self.v501_enabled:
                        cand_in = candidate_probs.detach()
                        support_in = support_stack.detach()
                        state_in = err["error_state_probs"].detach()
                        image_in = image_features.detach() if isinstance(image_features, torch.Tensor) else image_features
                        text_in = text_features.detach() if isinstance(text_features, torch.Tensor) else text_features
                    elif self.v491_enabled:
                        m2_to_m1 = float(
                            _m1(self.cfg, "V491_M2_TO_M1_GRAD_SCALE", 0.10)
                        )
                        cand_in = _scale_gradient(candidate_probs, m2_to_m1)
                        support_in = _scale_gradient(support_stack, m2_to_m1)
                        state_in = _scale_gradient(
                            err["error_state_probs"], m2_to_m1
                        )
                        image_in = image_features
                        text_in = text_features
                    else:
                        cand_in = candidate_probs
                        support_in = support_stack
                        state_in = err["error_state_probs"]
                        image_in = image_features
                        text_in = text_features
                else:
                    cand_in = candidate_probs.detach()
                    support_in = support_stack.detach()
                    state_in = err["error_state_probs"].detach()
                    image_in = image_features.detach() if isinstance(image_features, torch.Tensor) else image_features
                    text_in = text_features.detach() if isinstance(text_features, torch.Tensor) else text_features
                composer_runs.append(
                    self.pixel_composer(
                        image=image,
                        c0_prob=c0_prob,
                        candidate_probs=cand_in,
                        supports=support_in,
                        error_state_probs=state_in,
                        image_features=image_in,
                        text_features=text_in,
                        stochastic=mc_samples > 1,
                    )
                )

            if self.v495_enabled:
                if not self.v495_mc_logit_aggregation:
                    raise RuntimeError(
                        "V495 requires V495_MC_LOGIT_AGGREGATION_ENABLED=true."
                    )
                if not isinstance(self.pixel_composer, CausalLocalPixelEditor):
                    raise RuntimeError("V495 requires CausalLocalPixelEditor.")
                # Aggregate stochastic route/utility evidence first, then make
                # one sparse route and one complete proposal. Never average
                # already-deployed masks.
                m2_mean_dict = self.pixel_composer.aggregate_mc_outputs(
                    composer_runs, c0_prob
                )
                proposal_stack = torch.stack(
                    [run["m2_proposal_probs"] for run in composer_runs], dim=0
                )
                m2_mean = m2_mean_dict["m2_fused_probs"]
                m2_variance = proposal_stack.var(dim=0, unbiased=False)
            else:
                fused_stack = torch.stack(
                    [run["m2_fused_probs"] for run in composer_runs], dim=0
                )
                m2_mean = fused_stack.mean(dim=0)
                m2_variance = fused_stack.var(dim=0, unbiased=False)

                common_keys = set(composer_runs[0])
                for run in composer_runs[1:]:
                    common_keys.intersection_update(run)
                meanable_keys = [
                    key
                    for key in common_keys
                    if key != "m2_fused_probs"
                    and isinstance(composer_runs[0][key], torch.Tensor)
                ]
                m2_mean_dict = {}
                for key in meanable_keys:
                    values = [run[key] for run in composer_runs]
                    if all(value.shape == values[0].shape for value in values):
                        stacked = torch.stack(values, dim=0)
                        if stacked.is_floating_point() or stacked.is_complex():
                            m2_mean_dict[key] = stacked.mean(dim=0)
                        else:
                            # Discrete audit tensors (for example action index)
                            # cannot be averaged.  They are recomputed from the
                            # aggregated logits in V495; older branches keep the
                            # first deterministic value for compatibility.
                            m2_mean_dict[key] = values[0]
                m2_mean_dict["m2_fused_probs"] = m2_mean

            m2_mean_dict["m2_mc_variance"] = m2_variance[:, None]
            m2_mean_dict["m2_mc_std"] = torch.sqrt(m2_variance[:, None] + EPS)
            m2_mean_dict["m2_mc_samples"] = candidate_probs.new_full(
                (candidate_probs.shape[0],), float(mc_samples)
            )

            if self.v490_enabled:
                if self.v501_enabled:
                    m3_source_prob = (
                        m2_mean_dict["m2_proposal_probs"]
                        if self.v495_enabled else m2_mean
                    )
                    m3_source_gate = (
                        m2_mean_dict["m2_edit_gate_prob"]
                    )
                    m3_m2_prob = m3_source_prob.detach()
                    m3_variance = m2_variance.detach()
                    m3_edit_gate = m3_source_gate.detach()
                    selected_support = m2_mean_dict.get("m2_structural_support")
                    if not isinstance(selected_support, torch.Tensor):
                        raise RuntimeError(
                            "V501 requires proposal-aligned m2_structural_support."
                        )
                    m3_supports = selected_support.detach()
                    m3_candidates = candidate_probs.detach()
                elif self.v491_enabled:
                    m3_to_m2 = float(
                        _m1(self.cfg, "V491_M3_TO_M2_GRAD_SCALE", 0.10)
                    )
                    m3_source_prob = (
                        m2_mean_dict["m2_proposal_probs"]
                        if self.v495_enabled else m2_mean
                    )
                    m3_source_gate = (
                        m2_mean_dict["m2_benefit_prob"]
                        if self.v495_enabled
                        else m2_mean_dict["m2_edit_gate_prob"]
                    )
                    m3_m2_prob = _scale_gradient(m3_source_prob, m3_to_m2)
                    m3_variance = _scale_gradient(m2_variance, m3_to_m2)
                    m3_edit_gate = _scale_gradient(
                        m3_source_gate, m3_to_m2
                    )
                    if self.v495_enabled:
                        # The M3 proposal, support and Benefit must describe the
                        # same routed top-1 intervention.  Passing support_stack
                        # here mixes unrelated candidates and makes the risk
                        # selector spatially ambiguous.
                        selected_support = m2_mean_dict.get(
                            "m2_structural_support"
                        )
                        if not isinstance(selected_support, torch.Tensor):
                            raise RuntimeError(
                                "V495/V496 requires m2_structural_support for "
                                "proposal-aligned M3 risk estimation."
                            )
                        m3_supports = selected_support.detach()
                    else:
                        m3_supports = support_stack.detach()
                    m3_candidates = candidate_probs.detach()
                else:
                    m3_m2_prob = (
                        m2_mean_dict["m2_proposal_probs"]
                        if self.v495_enabled else m2_mean
                    )
                    m3_variance = m2_variance
                    m3_edit_gate = (
                        m2_mean_dict["m2_benefit_prob"]
                        if self.v495_enabled
                        else m2_mean_dict["m2_edit_gate_prob"]
                    )
                    m3_supports = (
                        m2_mean_dict["m2_structural_support"]
                        if self.v495_enabled else support_stack
                    )
                    m3_candidates = candidate_probs
                m3 = self.safe_deployer(
                    c0_prob=c0_prob,
                    candidate_probs=m3_candidates,
                    candidate_supports=m3_supports,
                    m2_prob=m3_m2_prob,
                    m2_variance=m3_variance,
                    m2_edit_gate=m3_edit_gate,
                )
            elif self.v489_enabled:
                m3 = self.safe_deployer(
                    c0_prob=c0_prob,
                    candidate_probs=candidate_probs,
                    m2_prob=m2_mean,
                    m2_variance=m2_variance,
                    m2_edit_gate=m2_mean_dict["m2_edit_gate_prob"],
                )
            else:
                m3 = self.safe_deployer(
                    c0_prob=c0_prob,
                    m2_mean=m2_mean,
                    m2_variance=m2_variance,
                    expected_effect=m2_mean_dict["m2_expected_effect"],
                    expected_evidence=m2_mean_dict["m2_expected_evidence"],
                )
        else:
            m2_mean_dict = {}
            m3 = self.rejector(c0_prob, candidate_probs, local_m2, loc["local_active_mask"][:, :k_local])

        if self.v502_enabled:
            m2_deploy_prob = m2_mean_dict.get("m2_fused_probs", c0_prob[:, 0])
            m3 = _v502_apply_ablation_output_stage(
                self.v502_ablation_output_stage,
                c0_prob,
                m2_deploy_prob,
                m3,
            )
            stage_ids = {"base": 0.0, "m1": 1.0, "m2": 2.0, "full": 3.0}
            canonical_stage = {
                "a0": "base", "base_only": "base",
                "a1": "m1", "m1_only": "m1", "m1_generator": "m1",
                "a2": "m2", "m1_m2": "m2", "m2_only": "m2",
                "control": "full",
            }.get(self.v502_ablation_output_stage, self.v502_ablation_output_stage)
            m2_mean_dict["v502_ablation_output_stage_id"] = candidate_probs.new_full(
                (candidate_probs.shape[0],), stage_ids[canonical_stage]
            )

        b, slots = candidate_probs.shape[:2]
        action_count = max(slots - 1, 0)
        candidate_family = torch.arange(slots, device=candidate_probs.device).view(1, -1).expand(b, -1)
        preserve = torch.zeros(b, slots, device=candidate_probs.device, dtype=candidate_probs.dtype)
        preserve[:, 0] = 1.0
        aux: Dict[str, torch.Tensor] = {
            **err,
            **loc,
            **glob,
            **{f"local_m2_{key}": value for key, value in local_m2.items()},
            **m2_mean_dict,
            **m3,
            "candidates": candidate_logits,
            "candidate_logits": candidate_logits,
            "candidate_probs": candidate_probs,
            "local_candidate_logits": local_logits,
            # Historical aliases are retained.  In V489 these tensors are
            # online and differentiable despite the old key name.
            "c0_prob_detached": c0_prob,
            "base_probs_detached": c0_prob,
            "c0_prob_online": c0_prob,
            "candidate_family": candidate_family,
            "candidate_source_v485": candidate_logits.new_ones((b,)),
            "v484_enabled": candidate_logits.new_ones((b,)),
            "v485_enabled": candidate_logits.new_ones((b,)),
            "v20_action_types": torch.arange(action_count, device=candidate_probs.device, dtype=torch.long),
            "v20_selector_hard": torch.zeros(b, action_count, device=candidate_probs.device, dtype=candidate_probs.dtype),
            "m1_selector_soft": preserve,
            "m1_selector_hard": preserve,
            "m1_choice_logits": torch.zeros_like(preserve),
            "v20_hard_fused_probs": m3["fused_probs"],
            "m1_hard_fused_probs": m3["fused_probs"],
            "direct_fused_probs": m2_mean_dict.get("m2_fused_probs", c0_prob[:, 0]),
            "router_fused_probs": m3["fused_probs"],
            "text_verifier_fused_probs": m3["fused_probs"],
            "v488_enabled": candidate_logits.new_full((b,), float(self.v488_enabled)),
            "v489_enabled": candidate_logits.new_full((b,), float(self.v489_enabled)),
            "v492_enabled": candidate_logits.new_full((b,), float(self.v492_enabled)),
            "v493_enabled": candidate_logits.new_full((b,), float(self.v493_enabled)),
            "v494_enabled": candidate_logits.new_full((b,), float(self.v494_enabled)),
            "v495_enabled": candidate_logits.new_full((b,), float(self.v495_enabled)),
            "v501_enabled": candidate_logits.new_full((b,), float(self.v501_enabled)),
            "v502_enabled": candidate_logits.new_full((b,), float(self.v502_enabled)),
            "v503_enabled": candidate_logits.new_full((b,), float(self.v503_enabled)),
            "v505_enabled": candidate_logits.new_full((b,), float(self.v505_enabled)),
            "v518_enabled": candidate_logits.new_full((b,), float(self.v518_enabled)),
            "v519_enabled": candidate_logits.new_full((b,), float(self.v519_region_composer)),
            "v521_enabled": candidate_logits.new_full((b,), float(self.v521_candidate_conditional_region)),
            "v522_enabled": candidate_logits.new_full((b,), float(self.v522_pwo_distilled_sequential)),
            "v523_enabled": candidate_logits.new_full((b,), float(self.v523_sea_level_utility)),
            "v524_enabled": candidate_logits.new_full((b,), float(self.v524_counterfactual_prompted_region)),
            "v518_trainable_candidate_mask": trainable_candidate_mask,
            "v518_local_candidate_count": candidate_logits.new_full((b,), float(local_count)),
            "v518_pair_candidate_count": candidate_logits.new_full((b,), float(pair_count)),
            "v518_morph_candidate_count": candidate_logits.new_full((b,), float(morph_count)),
            "v518_global_candidate_count": candidate_logits.new_full((b,), float(global_count)),
            "v518_candidate_supports": (
                support_stack if self.v505_enabled else torch.cat(
                    [
                        torch.zeros_like(c0_prob),
                        loc["local_supports"][:, :local_count],
                        (candidate_probs[:, 1 + local_count:] - c0_prob).abs(),
                    ],
                    dim=1,
                )
            ),
        }
        # Python metadata is intentionally kept outside tensor-only diagnostics.
        aux["v518_pair_candidate_names"] = pair_names
        aux["v518_morph_candidate_names"] = morph_names
        return aux

