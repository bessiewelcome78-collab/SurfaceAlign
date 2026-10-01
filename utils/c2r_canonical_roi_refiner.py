"""Canonical ROI Counterfactual-Consensus Regional Reconstruction (C2R-v2).

This module implements the corrected Stage-2 contract used by the canonical
C2R-v2 experiments.

Scientific contract
-------------------
1. WHERE: select deterministic, non-overlapping uncertainty ROIs with true
   greedy suppression.  ROI selection is inspection only.
2. WHAT: crop the selected ROIs and reconstruct them with a shared 2-D local
   decoder.  The decoder sees factual visual/semantic/text evidence plus one of
   three GT-free counterfactual coarse-mask conditions (eroded/factual/dilated).
3. CANONICAL REFERENCE: every counterfactual view predicts a residual around
   the same factual deployment anchor P0.  Counterfactual morphology is a
   conditioning variable, never the output reference.
4. CAUSAL TRACE: A3 uses one factual Base->Transport trace for all views;
   morphology counterfactuals never masquerade as Geometry outputs.
5. VERIFY/COMMIT: candidate hard changes are partitioned into 4-connected
   components inside each ROI.  An entire component is committed atomically
   only if the counterfactual reconstructions jointly satisfy a deterministic
   agreement/spread/confidence certificate and the component intersects the
   uncertainty seed.  Otherwise the component is kept as the factual anchor.
6. SAFETY/OWNERSHIP: outside committed components output is exact identity;
   Stage-2 inputs are detached so C2R gradients cannot alter Base/Transport.
7. TRAIN/TEST MATCH: erode/factual/dilate construction and the ROI decoder are
   identical at train and inference.  GT is never an input to this forward.

The module deliberately contains no synthetic corruption labels, learned
benefit/harm gate, error-type classifier, stochastic Stage-2 state, or
straight-through deployment gate.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1.0e-4


def _groups(channels: int) -> int:
    groups = min(8, max(1, int(channels)))
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


class _ConvNormGELU(nn.Sequential):
    def __init__(self, cin: int, cout: int, kernel_size: int = 3, dilation: int = 1) -> None:
        padding = dilation * (kernel_size // 2)
        super().__init__(
            nn.Conv2d(cin, cout, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.GroupNorm(_groups(cout), cout),
            nn.GELU(),
        )


class CanonicalCounterfactualROIRefiner(nn.Module):
    """True ROI-conditioned canonical counterfactual refiner."""

    def __init__(
        self,
        hidden_dim: int,
        semantic_channels: int,
        text_dim: int,
        *,
        fine_feature_channels: int = 512,
        num_regions: int = 4,
        region_size: int = 33,
        counterfactual_radius: int = 1,
        selection_score: str = "margin",
        flow_scale_px: float = 8.0,
        min_center_distance: Optional[int] = None,
        seed_radius: int = 2,
        component_agreement_threshold: float = 0.90,
        component_spread_threshold: float = 0.12,
        component_confidence_threshold: float = 0.10,
        component_min_area: int = 2,
        canonical_residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.semantic_channels = int(semantic_channels)
        self.text_dim = int(text_dim)
        self.fine_feature_channels = int(fine_feature_channels)
        self.num_regions = max(1, int(num_regions))
        self.region_size = int(region_size)
        if self.region_size < 3 or self.region_size % 2 == 0:
            raise ValueError("GEOTR_C2R_REGION_SIZE must be an odd integer >= 3")
        self.counterfactual_radius = max(1, int(counterfactual_radius))
        self.selection_score = str(selection_score).strip().lower()
        if self.selection_score not in {
            "margin", "mc_std", "mc_disagreement", "entropy", "hybrid_max"
        }:
            raise ValueError(
                "GEOTR_C2R_SELECTION_SCORE must be one of "
                "margin/mc_std/mc_disagreement/entropy/hybrid_max"
            )
        self.flow_scale_px = float(max(flow_scale_px, 1.0e-3))
        self.min_center_distance = int(
            self.region_size if min_center_distance is None else min_center_distance
        )
        if self.min_center_distance < self.region_size:
            raise ValueError(
                "GEOTR_C2R_MIN_CENTER_DISTANCE must be >= GEOTR_C2R_REGION_SIZE "
                "to guarantee non-overlapping square ROIs"
            )
        self.seed_radius = max(0, int(seed_radius))
        self.component_agreement_threshold = float(component_agreement_threshold)
        self.component_spread_threshold = float(component_spread_threshold)
        self.component_confidence_threshold = float(component_confidence_threshold)
        self.component_min_area = max(1, int(component_min_area))
        self.canonical_residual_scale = float(canonical_residual_scale)
        if self.canonical_residual_scale <= 0.0:
            raise ValueError("GEOTR_C2R_CANONICAL_RESIDUAL_SCALE must be > 0")
        if not (0.0 <= self.component_agreement_threshold <= 1.0):
            raise ValueError("GEOTR_C2R_COMPONENT_AGREEMENT_THRESHOLD must be in [0,1]")
        if self.component_spread_threshold < 0.0:
            raise ValueError("GEOTR_C2R_COMPONENT_SPREAD_THRESHOLD must be >= 0")
        if not (0.0 <= self.component_confidence_threshold <= 0.5):
            raise ValueError("GEOTR_C2R_COMPONENT_CONFIDENCE_THRESHOLD must be in [0,0.5]")

        # C2R owns trainable projections over detached cached evidence.
        self.image_stem = nn.Sequential(
            _ConvNormGELU(3, hidden_dim),
            _ConvNormGELU(hidden_dim, hidden_dim),
        )
        self.semantic_proj = nn.Sequential(
            nn.Conv2d(semantic_channels, hidden_dim, 1, bias=False),
            nn.GroupNorm(_groups(hidden_dim), hidden_dim),
            nn.GELU(),
        )
        self.fine_proj = nn.Sequential(
            nn.Conv2d(self.fine_feature_channels, hidden_dim, 1, bias=False),
            nn.GroupNorm(_groups(hidden_dim), hidden_dim),
            nn.GELU(),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.visual_fuse = nn.Sequential(
            _ConvNormGELU(3 * hidden_dim, hidden_dim),
            _ConvNormGELU(hidden_dim, hidden_dim),
        )
        self.context_film = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.context_gate = nn.Parameter(torch.zeros(()))

        # Canonical ROI decoder inputs:
        # common visual H
        # factual anchor P0
        # counterfactual condition Pv
        # Pv-P0
        # margin uncertainty
        # transport-aligned MC std
        # factual 6-channel Base->Anchor trace
        # 2-D ROI-relative coordinates
        decoder_in = hidden_dim + 1 + 1 + 1 + 1 + 1 + 6 + 2
        self.region_decoder = nn.Sequential(
            _ConvNormGELU(decoder_in, hidden_dim, 3, dilation=1),
            _ConvNormGELU(hidden_dim, hidden_dim, 3, dilation=2),
            _ConvNormGELU(hidden_dim, hidden_dim, 3, dilation=1),
        )
        self.region_residual_out = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        # Exact factual-anchor identity at initialization for every CF view.
        nn.init.zeros_(self.region_residual_out.weight)
        nn.init.zeros_(self.region_residual_out.bias)

    @staticmethod
    def _resize(x: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] == hw:
            return x
        return F.interpolate(x, size=hw, mode="bilinear", align_corners=False)

    @staticmethod
    def _margin_uncertainty(prob: torch.Tensor) -> torch.Tensor:
        return (4.0 * prob * (1.0 - prob)).clamp(0.0, 1.0)

    @staticmethod
    def _entropy(prob: torch.Tensor) -> torch.Tensor:
        p = prob.clamp(EPS, 1.0 - EPS)
        h = -(p * p.log() + (1.0 - p) * (1.0 - p).log())
        return (h / math.log(2.0)).clamp(0.0, 1.0)

    @staticmethod
    def _fit_evidence(
        x: Optional[torch.Tensor], anchor: torch.Tensor, scale: float = 1.0
    ) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            return torch.zeros_like(anchor)
        x = x.detach().to(anchor)
        if x.ndim == 3:
            x = x[:, None]
        if x.shape[-2:] != anchor.shape[-2:]:
            x = F.interpolate(x, size=anchor.shape[-2:], mode="bilinear", align_corners=False)
        return (float(scale) * x).clamp(0.0, 1.0)

    def _factual_trace(
        self,
        base_prob: torch.Tensor,
        factual_anchor: torch.Tensor,
        flow_px: torch.Tensor,
    ) -> torch.Tensor:
        """One causal Base->factual-anchor trace shared by all CF views."""
        base = base_prob.detach().clamp(EPS, 1.0 - EPS)
        anchor = factual_anchor.detach().clamp(EPS, 1.0 - EPS)
        flow = flow_px.detach().to(anchor) / self.flow_scale_px
        if flow.shape[-2:] != anchor.shape[-2:]:
            flow = self._resize(flow, tuple(anchor.shape[-2:]))
        flow = flow.clamp(-1.0, 1.0)
        mag = torch.sqrt(
            flow[:, 0:1].square() + flow[:, 1:2].square() + 1.0e-12
        ).clamp(0.0, 1.0)
        return torch.cat(
            [base, anchor, anchor - base, flow[:, 0:1], flow[:, 1:2], mag], dim=1
        )

    def _counterfactual_views(
        self, anchor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k = 2 * self.counterfactual_radius + 1
        eroded = -F.max_pool2d(
            -anchor, kernel_size=k, stride=1, padding=self.counterfactual_radius
        )
        dilated = F.max_pool2d(
            anchor, kernel_size=k, stride=1, padding=self.counterfactual_radius
        )
        return (
            eroded.clamp(EPS, 1.0 - EPS),
            anchor,
            dilated.clamp(EPS, 1.0 - EPS),
        )

    def _selection_score(self, evidence: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.selection_score == "hybrid_max":
            return torch.maximum(
                torch.maximum(evidence["margin"], evidence["mc_std"]),
                torch.maximum(evidence["mc_disagreement"], evidence["entropy"]),
            )
        return evidence[self.selection_score]

    def _greedy_centers(
        self, score: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Deterministic greedy non-overlap selection.

        Returns
        -------
        center_mask: [B,1,H,W] bool
        center_yx:   [B,K,2] long (y,x)
        center_valid:[B,K] bool
        """
        b, _, h, w = score.shape
        k = self.num_regions
        ranked = score[:, 0].detach().clone()
        # deterministic tie break smaller than any meaningful fp32 score change
        order = torch.arange(h * w, device=score.device, dtype=score.dtype).view(h, w)
        ranked = ranked + order / max(float(h * w), 1.0) * 1.0e-8
        valid = torch.ones((b, h, w), dtype=torch.bool, device=score.device)
        yy = torch.arange(h, device=score.device).view(1, h, 1)
        xx = torch.arange(w, device=score.device).view(1, 1, w)
        coords = torch.zeros((b, k, 2), dtype=torch.long, device=score.device)
        coord_valid = torch.zeros((b, k), dtype=torch.bool, device=score.device)
        centers = torch.zeros((b, 1, h, w), dtype=torch.bool, device=score.device)
        batch_idx = torch.arange(b, device=score.device)

        for j in range(k):
            masked = ranked.masked_fill(~valid, float("-inf"))
            flat = masked.flatten(1)
            best_val, best_idx = flat.max(dim=1)
            is_valid = torch.isfinite(best_val)
            cy = torch.div(best_idx, w, rounding_mode="floor")
            cx = best_idx % w
            coords[:, j, 0] = cy
            coords[:, j, 1] = cx
            coord_valid[:, j] = is_valid
            centers[batch_idx[is_valid], 0, cy[is_valid], cx[is_valid]] = True

            # Suppress centers whose square ROIs would overlap.  For odd R,
            # center Chebyshev distance >= R guarantees disjoint R x R windows.
            dy = (yy - cy[:, None, None]).abs()
            dx = (xx - cx[:, None, None]).abs()
            suppress = (dy < self.min_center_distance) & (dx < self.min_center_distance)
            suppress = suppress & is_valid[:, None, None]
            valid = valid & (~suppress)

        return centers, coords, coord_valid

    def _patch_geometry(
        self,
        center_yx: torch.Tensor,
        center_valid: torch.Tensor,
        h: int,
        w: int,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build integer patch indices, sampling grid, valid mask, and rel coords."""
        b, k, _ = center_yx.shape
        r = self.region_size
        half = r // 2
        off = torch.arange(-half, half + 1, device=center_yx.device)
        oy, ox = torch.meshgrid(off, off, indexing="ij")
        ay = center_yx[:, :, 0, None, None] + oy[None, None]
        ax = center_yx[:, :, 1, None, None] + ox[None, None]
        valid = (
            (ay >= 0) & (ay < h) & (ax >= 0) & (ax < w)
            & center_valid[:, :, None, None]
        )
        ay_clamped = ay.clamp(0, max(h - 1, 0))
        ax_clamped = ax.clamp(0, max(w - 1, 0))
        if h > 1:
            gy = 2.0 * ay_clamped.to(dtype) / float(h - 1) - 1.0
        else:
            gy = ay_clamped.to(dtype) * 0.0
        if w > 1:
            gx = 2.0 * ax_clamped.to(dtype) / float(w - 1) - 1.0
        else:
            gx = ax_clamped.to(dtype) * 0.0
        grid = torch.stack([gx, gy], dim=-1).reshape(b * k, r, r, 2)
        rel_y = oy.to(dtype) / max(float(half), 1.0)
        rel_x = ox.to(dtype) / max(float(half), 1.0)
        rel = torch.stack([rel_x, rel_y], dim=0)[None, None].expand(b, k, -1, -1, -1)
        return ay_clamped, ax_clamped, grid, valid[:, :, None], rel

    def _crop(
        self, x: torch.Tensor, grid: torch.Tensor, k: int
    ) -> torch.Tensor:
        """Sample integer-centered ROIs as [B,K,C,R,R]."""
        b, c, _, _ = x.shape
        expanded = x[:, None].expand(b, k, c, *x.shape[-2:]).reshape(b * k, c, *x.shape[-2:])
        patch = F.grid_sample(
            expanded, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        return patch.view(b, k, c, self.region_size, self.region_size)

    @staticmethod
    def _scatter_patch_values(
        patch: torch.Tensor,
        anchor: torch.Tensor,
        ay: torch.Tensor,
        ax: torch.Tensor,
        valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Scatter non-overlapping patches to full image.

        patch: [B,K,V,R,R]
        anchor:[B,1,H,W]
        valid: [B,K,1,R,R]
        Returns full [B,V,H,W], union mask [B,1,H,W], overlap count scalar.
        """
        b, k, v, r, _ = patch.shape
        h, w = anchor.shape[-2:]
        idx = (ay * w + ax).reshape(b, -1)
        val = valid[:, :, 0].reshape(b, -1).to(anchor)
        vals = patch.permute(0, 2, 1, 3, 4).reshape(b, v, -1) * val[:, None]
        sums = anchor.new_zeros((b, v, h * w))
        counts = anchor.new_zeros((b, 1, h * w))
        sums.scatter_add_(2, idx[:, None].expand(-1, v, -1), vals)
        counts.scatter_add_(2, idx[:, None], val[:, None])
        anchor_v = anchor.expand(-1, v, -1, -1).reshape(b, v, h * w)
        full = torch.where(
            counts > 0,
            sums / counts.clamp_min(1.0),
            anchor_v,
        ).view(b, v, h, w)
        union = (counts > 0).view(b, 1, h, w)
        overlap = (counts > 1.0 + 1.0e-6).float().sum()
        return full, union, overlap

    @staticmethod
    def _connected_labels(mask: torch.Tensor) -> torch.Tensor:
        """4-connected label propagation for small ROI masks, entirely in torch."""
        # mask [N,R,R] bool
        n, r, _ = mask.shape
        ids = torch.arange(1, r * r + 1, device=mask.device, dtype=torch.long).view(1, r, r)
        labels = ids.expand(n, -1, -1) * mask.long()
        # Max-label propagation.  2*(R-1) is the maximum Manhattan diameter.
        for _ in range(2 * (r - 1)):
            up = F.pad(labels[:, :-1, :], (0, 0, 1, 0))
            down = F.pad(labels[:, 1:, :], (0, 0, 0, 1))
            left = F.pad(labels[:, :, :-1], (1, 0, 0, 0))
            right = F.pad(labels[:, :, 1:], (0, 1, 0, 0))
            labels = torch.where(
                mask,
                torch.maximum(
                    torch.maximum(labels, up),
                    torch.maximum(torch.maximum(down, left), right),
                ),
                torch.zeros_like(labels),
            )
        return labels

    def _component_commit(
        self,
        view_probs: torch.Tensor,
        anchor_patch: torch.Tensor,
        valid: torch.Tensor,
        rel: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return atomic component commit mask [B,K,1,R,R] and diagnostics."""
        # view_probs [B,K,V,R,R], anchor_patch [B,K,1,R,R]
        b, k, v, r, _ = view_probs.shape
        with torch.no_grad():
            mean = view_probs.mean(dim=2)
            anchor_h = anchor_patch[:, :, 0] >= 0.5
            mean_h = mean >= 0.5
            candidate = (mean_h != anchor_h) & valid[:, :, 0]
            hard = view_probs >= 0.5
            unanimous = (hard == hard[:, :, :1]).all(dim=2)
            spread = view_probs.max(dim=2).values - view_probs.min(dim=2).values
            confidence = (mean - 0.5).abs()
            seed = (
                rel[:, :, 0].abs() <= (self.seed_radius / max(float(self.region_size // 2), 1.0) + 1e-8)
            ) & (
                rel[:, :, 1].abs() <= (self.seed_radius / max(float(self.region_size // 2), 1.0) + 1e-8)
            )
            seed = seed & valid[:, :, 0]

            labels = self._connected_labels(candidate.reshape(b * k, r, r))
            commit = torch.zeros_like(candidate)
            candidate_components = anchor_patch.new_zeros(())
            committed_components = anchor_patch.new_zeros(())
            candidate_area = anchor_patch.new_zeros(())
            committed_area = anchor_patch.new_zeros(())
            agreement_sum = anchor_patch.new_zeros(())
            spread_sum = anchor_patch.new_zeros(())
            confidence_sum = anchor_patch.new_zeros(())

            labels = labels.view(b, k, r, r)
            for bi in range(b):
                for ki in range(k):
                    ids = torch.unique(labels[bi, ki])
                    ids = ids[ids > 0]
                    for cid in ids:
                        comp = labels[bi, ki] == cid
                        area = comp.float().sum()
                        if int(area.item()) < self.component_min_area:
                            continue
                        candidate_components += 1.0
                        candidate_area += area
                        vals_agree = unanimous[bi, ki][comp].float()
                        vals_spread = spread[bi, ki][comp]
                        vals_conf = confidence[bi, ki][comp]
                        agreement = vals_agree.mean()
                        spread_q90 = torch.quantile(vals_spread, 0.90)
                        conf_q10 = torch.quantile(vals_conf, 0.10)
                        agreement_sum += agreement
                        spread_sum += spread_q90
                        confidence_sum += conf_q10
                        seed_hit = bool((seed[bi, ki] & comp).any().item())
                        passed = (
                            seed_hit
                            and float(agreement.item()) >= self.component_agreement_threshold
                            and float(spread_q90.item()) <= self.component_spread_threshold
                            and float(conf_q10.item()) >= self.component_confidence_threshold
                        )
                        if passed:
                            commit[bi, ki] |= comp
                            committed_components += 1.0
                            committed_area += area

            denom = candidate_components.clamp_min(1.0)
            stats = {
                "candidate_mask": candidate[:, :, None].to(anchor_patch),
                "candidate_component_count": candidate_components,
                "committed_component_count": committed_components,
                "candidate_component_area_mean": candidate_area / denom,
                "committed_component_area_mean": committed_area / committed_components.clamp_min(1.0),
                "component_agreement_mean": agreement_sum / denom,
                "component_spread_q90_mean": spread_sum / denom,
                "component_confidence_q10_mean": confidence_sum / denom,
            }
        return commit[:, :, None], stats

    def _common_visual(
        self,
        image: torch.Tensor,
        semantic_map: torch.Tensor,
        text_features: torch.Tensor,
        fine_feature_map: Optional[torch.Tensor],
        hw: Tuple[int, int],
    ) -> torch.Tensor:
        img = self.image_stem(self._resize(image.detach(), hw))
        sem = self.semantic_proj(self._resize(semantic_map.detach(), hw))
        if isinstance(fine_feature_map, torch.Tensor):
            fine = fine_feature_map.detach()
            if fine.ndim != 4:
                raise ValueError(f"fine_feature_map must be BCHW, got {tuple(fine.shape)}")
            fine = self.fine_proj(self._resize(fine.to(img), hw))
        else:
            fine = torch.zeros_like(img)
        visual = self.visual_fuse(torch.cat([img, sem, fine], dim=1))
        text = self.text_proj(text_features.detach().float()).to(visual)
        gamma, beta = self.context_film(text).chunk(2, dim=1)
        gate = torch.tanh(self.context_gate)
        return visual + gate * (
            torch.tanh(gamma)[:, :, None, None] * visual + beta[:, :, None, None]
        )

    def forward(
        self,
        anchor_prob: torch.Tensor,
        image: torch.Tensor,
        semantic_map: torch.Tensor,
        text_features: torch.Tensor,
        *,
        base_prob: Optional[torch.Tensor] = None,
        flow_px: Optional[torch.Tensor] = None,
        mc_std_map: Optional[torch.Tensor] = None,
        mc_disagreement_map: Optional[torch.Tensor] = None,
        fine_feature_map: Optional[torch.Tensor] = None,
        posterior_probability_samples: Optional[torch.Tensor] = None,
        supervision_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # API compatibility only.  GT never enters deployable C2R-v2 forward.
        del posterior_probability_samples
        del supervision_masks
        anchor = anchor_prob.detach().clamp(EPS, 1.0 - EPS)
        if base_prob is None:
            base_prob = anchor
        base = base_prob.detach().clamp(EPS, 1.0 - EPS)
        if flow_px is None:
            flow_px = anchor.new_zeros((anchor.shape[0], 2, *anchor.shape[-2:]))
        flow = flow_px.detach().to(anchor)
        b, _, h, w = anchor.shape
        hw = (h, w)

        margin = self._margin_uncertainty(anchor)
        mc_std = self._fit_evidence(mc_std_map, anchor, scale=2.0)
        mc_dis = self._fit_evidence(mc_disagreement_map, anchor, scale=1.0)
        entropy = self._entropy(anchor)
        evidence = {
            "margin": margin,
            "mc_std": mc_std,
            "mc_disagreement": mc_dis,
            "entropy": entropy,
        }
        score = self._selection_score(evidence)
        centers, center_yx, center_valid = self._greedy_centers(score)
        ay, ax, grid, valid_patch, rel = self._patch_geometry(
            center_yx, center_valid, h, w, anchor.dtype
        )
        k = self.num_regions

        common = self._common_visual(image, semantic_map, text_features, fine_feature_map, hw)
        trace = self._factual_trace(base, anchor, flow)
        views = self._counterfactual_views(anchor)

        common_p = self._crop(common, grid, k)
        anchor_p = self._crop(anchor, grid, k)
        margin_p = self._crop(margin, grid, k)
        mc_std_p = self._crop(mc_std, grid, k)
        trace_p = self._crop(trace, grid, k)
        view_p = torch.stack([self._crop(vv, grid, k)[:, :, 0] for vv in views], dim=2)
        # [B,K,V,R,R]
        inputs = []
        for vi in range(3):
            vp = view_p[:, :, vi:vi+1]
            inp = torch.cat(
                [
                    common_p,
                    anchor_p,
                    vp,
                    vp - anchor_p,
                    margin_p,
                    mc_std_p,
                    trace_p,
                    rel,
                ],
                dim=2,
            )
            inputs.append(inp)
        x = torch.stack(inputs, dim=2)  # [B,K,V,C,R,R]
        cin = x.shape[3]
        x = x.reshape(b * k * 3, cin, self.region_size, self.region_size)
        residual = self.region_residual_out(self.region_decoder(x))
        residual = residual.view(b, k, 3, self.region_size, self.region_size)

        # All CF conditions reconstruct the same factual anchor reference.
        canonical_anchor = anchor_p[:, :, 0:1]
        patch_probs = (
            canonical_anchor
            + self.canonical_residual_scale * torch.tanh(residual)
        ).clamp(EPS, 1.0 - EPS)
        patch_logits = torch.logit(patch_probs)
        patch_mean = patch_probs.mean(dim=2)

        full_view_probs, region, overlap = self._scatter_patch_values(
            patch_probs, anchor, ay, ax, valid_patch
        )
        full_view_logits = torch.logit(full_view_probs.clamp(EPS, 1.0 - EPS))
        mean_prob = full_view_probs.mean(dim=1, keepdim=True)

        commit_patch, comp_stats = self._component_commit(
            patch_probs, anchor_p, valid_patch, rel
        )
        candidate_full, _, _ = self._scatter_patch_values(
            comp_stats["candidate_mask"][:, :, 0:1],
            torch.zeros_like(anchor), ay, ax, valid_patch,
        )
        candidate_full = candidate_full[:, 0:1] > 0.5
        commit_full, _, _ = self._scatter_patch_values(
            commit_patch[:, :, 0:1],
            torch.zeros_like(anchor), ay, ax, valid_patch,
        )
        commit_full = commit_full[:, 0:1] > 0.5
        # Hard deployment is atomic at component level.  KEEP/outside-commit is
        # copied directly from the factual anchor, giving bitwise identity at
        # initialization and on every rejected component.
        final_prob = torch.where(commit_full, mean_prob, anchor).clamp(EPS, 1.0 - EPS)
        final_logits = torch.logit(final_prob)
        delta = final_logits - torch.logit(anchor)

        hard_views = full_view_probs >= 0.5
        unanimous = ((hard_views == hard_views[:, :1]).all(dim=1, keepdim=True)) & region

        # ROI geometry diagnostics.
        center_float = centers.to(anchor)
        # Because greedy separation forbids overlap, overlap must be exactly 0.
        roi_pixel_count = region.float().sum()
        # Pairwise minimum Chebyshev center distance across valid pairs.
        min_dist = anchor.new_tensor(float(max(h, w)))
        if k > 1:
            dvals = []
            for i in range(k):
                for j in range(i + 1, k):
                    pair_valid = center_valid[:, i] & center_valid[:, j]
                    if pair_valid.any():
                        dy = (center_yx[pair_valid, i, 0] - center_yx[pair_valid, j, 0]).abs()
                        dx = (center_yx[pair_valid, i, 1] - center_yx[pair_valid, j, 1]).abs()
                        dvals.append(torch.maximum(dy, dx).to(anchor.dtype))
            if dvals:
                min_dist = torch.cat(dvals).min()

        z = torch.zeros_like(anchor)
        op = torch.full((anchor.shape[0],), -1, dtype=torch.long, device=anchor.device)
        return {
            "logits": final_logits,
            "prob": final_prob,
            "selection_mask": region.to(anchor),
            "selection_score": score,
            "refined_logits": torch.logit(mean_prob.clamp(EPS, 1.0 - EPS)),
            "refined_prob": mean_prob,
            "delta_logit": delta,
            "margin_uncertainty": margin,
            "mc_std_map": mc_std,
            "mc_disagreement_map": mc_dis,
            "entropy_map": entropy,
            "trace": trace,
            "fine_feature_map": common,
            # Legacy compatibility fields: no synthetic R4 branch exists.
            "dn_anchor_prob": anchor,
            "dn_corruption_mask": z,
            "dn_selection_mask": z,
            "dn_refined_logits": torch.logit(anchor),
            "dn_refined_prob": anchor,
            "dn_final_prob": anchor,
            "dn_delta_logit": z,
            "dn_op_id": op,
            "r4_flip_logits": z,
            "r4_flip_prob": z,
            "r4_flip_mask": z,
            "r4_synth_flip_logits": z,
            "r4_synth_flip_prob": z,
            "r4_synth_flip_mask": z,
            "r4_synth_target": z,
            "r2_enabled": anchor.new_zeros((anchor.shape[0],)),
            "r3_enabled": anchor.new_zeros((anchor.shape[0],)),
            "r4_enabled": anchor.new_zeros((anchor.shape[0],)),
            "r41_enabled": anchor.new_zeros((anchor.shape[0],)),
            # C2R shared compatibility fields.
            "c2r_center_mask": center_float,
            "c2r_region_mask": region.to(anchor),
            "c2r_view_logits": full_view_logits,
            "c2r_view_probs": full_view_probs,
            "c2r_mean_prob": mean_prob,
            "c2r_consensus_mask": unanimous.to(anchor),
            "c2r_edit_mask": commit_full.to(anchor),
            "c2r_eroded_anchor": views[0],
            "c2r_factual_anchor": views[1],
            "c2r_dilated_anchor": views[2],
            # Canonical v2-native fields.
            "c2r_v2_enabled": anchor.new_ones((anchor.shape[0],)),
            "c2r_center_yx": center_yx,
            "c2r_center_valid": center_valid.to(anchor),
            "c2r_roi_overlap_pixel_count": overlap,
            "c2r_roi_unique_pixel_count": roi_pixel_count,
            "c2r_center_min_chebyshev_distance": min_dist,
            "c2r_candidate_mask": candidate_full.to(anchor),
            "c2r_commit_mask": commit_full.to(anchor),
            "c2r_candidate_component_count": comp_stats["candidate_component_count"],
            "c2r_committed_component_count": comp_stats["committed_component_count"],
            "c2r_candidate_component_area_mean": comp_stats["candidate_component_area_mean"],
            "c2r_committed_component_area_mean": comp_stats["committed_component_area_mean"],
            "c2r_component_agreement_mean": comp_stats["component_agreement_mean"],
            "c2r_component_spread_q90_mean": comp_stats["component_spread_q90_mean"],
            "c2r_component_confidence_q10_mean": comp_stats["component_confidence_q10_mean"],
            "c2r_patch_view_logits": patch_logits,
            "c2r_patch_view_probs": patch_probs,
            "c2r_patch_mean_prob": patch_mean,
            "c2r_patch_valid_mask": valid_patch.to(anchor),
        }
