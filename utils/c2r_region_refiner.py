"""Counterfactual-Consensus Regional Reconstruction (C2R).

This is the deployable GEOTR Stage-2 used by the C2R experiments.

Design contract
---------------
1. WHERE is inspection only: uncertain locations define a small union of local ROIs.
2. HOW is direct regional segmentation, not error/action classification.
3. Train and inference use the same three GT-free anchor views:
      soft-eroded / factual / soft-dilated.
4. All three views share one 2-D local reconstruction decoder and are supervised
   against the same GT regional segmentation target during training.
5. Deployment is fail-closed without a learned safety gate: an anchor pixel is
   changed only when the three counterfactual reconstructions unanimously agree
   on the opposite hard label.  Outside the inspection ROIs the output is exact
   identity.
6. Stage-2 never back-propagates into Base/Transport anchors or upstream cached
   evidence; all inputs are detached before entering the C2R evidence encoder.

The module intentionally contains no GT-dependent corruption generator, no
FLIP/KEEP classifier, no benefit/harm oracle, and no stochastic Stage-2 state.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
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
            nn.Conv2d(
                cin, cout, kernel_size,
                padding=padding, dilation=dilation, bias=False,
            ),
            nn.GroupNorm(_groups(cout), cout),
            nn.GELU(),
        )


class CounterfactualConsensusRegionRefiner(nn.Module):
    """C2R regional refiner.

    The implementation computes a dense feature lattice once, but *supervision
    and deployment are restricted to the deterministic union of selected local
    ROIs*. This avoids fragile crop/scatter bookkeeping while preserving the
    exact local intervention contract.
    """

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

        # Upstream tensors are detached, but C2R owns trainable projections that
        # adapt the cached image/semantic/fine/text evidence for local recovery.
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

        # View-specific evidence: common visual H + anchor view + factual margin
        # + transport-aligned MC std + 6-channel causal trace + x/y coordinates.
        decoder_in = hidden_dim + 1 + 1 + 1 + 6 + 2
        self.region_decoder = nn.Sequential(
            _ConvNormGELU(decoder_in, hidden_dim, 3, dilation=1),
            _ConvNormGELU(hidden_dim, hidden_dim, 3, dilation=2),
            _ConvNormGELU(hidden_dim, hidden_dim, 3, dilation=1),
        )
        self.region_residual_out = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        # Exact identity initialization in logit space for every anchor view.
        nn.init.zeros_(self.region_residual_out.weight)
        nn.init.zeros_(self.region_residual_out.bias)

    @staticmethod
    def _resize(x: torch.Tensor, hw: Tuple[int, int], *, nearest: bool = False) -> torch.Tensor:
        if x.shape[-2:] == hw:
            return x
        if nearest:
            return F.interpolate(x, size=hw, mode="nearest")
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

    def _trace(
        self,
        base_prob: torch.Tensor,
        view_prob: torch.Tensor,
        flow_px: torch.Tensor,
    ) -> torch.Tensor:
        base = base_prob.detach().clamp(EPS, 1.0 - EPS)
        view = view_prob.detach().clamp(EPS, 1.0 - EPS)
        flow = flow_px.detach().to(view) / self.flow_scale_px
        if flow.shape[-2:] != view.shape[-2:]:
            flow = self._resize(flow, tuple(view.shape[-2:]))
        flow = flow.clamp(-1.0, 1.0)
        mag = torch.sqrt(
            flow[:, 0:1].square() + flow[:, 1:2].square() + 1.0e-12
        ).clamp(0.0, 1.0)
        return torch.cat(
            [base, view, view - base, flow[:, 0:1], flow[:, 1:2], mag], dim=1
        )

    def _counterfactual_views(self, anchor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k = 2 * self.counterfactual_radius + 1
        # max-pool padding uses -inf internally.  For probabilities this gives
        # a valid dilation; applying the same operation to -p yields a min-pool.
        eroded = -F.max_pool2d(-anchor, kernel_size=k, stride=1, padding=self.counterfactual_radius)
        dilated = F.max_pool2d(anchor, kernel_size=k, stride=1, padding=self.counterfactual_radius)
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

    def _region_support(self, score: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select NMS-separated uncertainty peaks, then expand them to local ROIs."""
        b, _, h, w = score.shape
        n = h * w
        k = min(self.num_regions, n)
        # Region-size NMS deliberately separates inspection windows.  A small
        # deterministic coordinate tie-break prevents plateau-dependent topk ties.
        nms_kernel = self.region_size
        pooled = F.max_pool2d(score.detach(), nms_kernel, stride=1, padding=nms_kernel // 2)
        maxima = score.detach() >= (pooled - 1.0e-12)
        yy = torch.arange(h, device=score.device, dtype=score.dtype).view(1, 1, h, 1)
        xx = torch.arange(w, device=score.device, dtype=score.dtype).view(1, 1, 1, w)
        tie = (yy * w + xx) / max(float(n), 1.0) * 1.0e-7
        ranked = (score.detach() + tie).masked_fill(~maxima, float("-inf"))
        idx = ranked.flatten(1).topk(k, dim=1, largest=True, sorted=False).indices
        center_flat = torch.zeros((b, n), dtype=torch.bool, device=score.device)
        center_flat.scatter_(1, idx, True)
        centers = center_flat.view(b, 1, h, w)
        region = F.max_pool2d(
            centers.to(score), self.region_size, stride=1, padding=self.region_size // 2
        ) > 0.5
        return centers, region

    @staticmethod
    def _coords(anchor: torch.Tensor) -> torch.Tensor:
        b, _, h, w = anchor.shape
        yy = torch.linspace(-1.0, 1.0, h, device=anchor.device, dtype=anchor.dtype)
        xx = torch.linspace(-1.0, 1.0, w, device=anchor.device, dtype=anchor.dtype)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        return torch.stack([gx, gy], dim=0)[None].expand(b, -1, -1, -1)

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
        visual = visual + gate * (
            torch.tanh(gamma)[:, :, None, None] * visual + beta[:, :, None, None]
        )
        return visual

    def _decode_views(
        self,
        views: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        common_visual: torch.Tensor,
        base_prob: torch.Tensor,
        flow_px: torch.Tensor,
        margin: torch.Tensor,
        mc_std: torch.Tensor,
        coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = views[0].shape
        inputs = []
        for view in views:
            trace = self._trace(base_prob, view, flow_px)
            inputs.append(torch.cat([common_visual, view, margin, mc_std, trace, coords], dim=1))
        x = torch.cat(inputs, dim=0)
        residual = self.region_residual_out(self.region_decoder(x))
        # Probability-residual parameterization gives *bitwise* identity at
        # initialization (tanh(0)=0 and p+0==p) while still allowing a view to
        # move to either class after learning.  This avoids tiny sigmoid(logit(p))
        # round-trip errors crossing the 0.5 commit boundary at step zero.
        base_prob_cat = torch.cat(list(views), dim=0)
        probs = (base_prob_cat + torch.tanh(residual)).clamp(EPS, 1.0 - EPS)
        logits = torch.logit(probs)
        logits = logits.view(3, b, 1, h, w).permute(1, 0, 2, 3, 4)[:, :, 0]
        probs = probs.view(3, b, 1, h, w).permute(1, 0, 2, 3, 4)[:, :, 0]
        return logits, probs

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
        # ``supervision_masks`` is accepted only for wrapper API compatibility.
        # It is deliberately ignored: GT never enters the deployable C2R forward.
        del posterior_probability_samples
        del supervision_masks
        anchor = anchor_prob.detach().clamp(EPS, 1.0 - EPS)
        if base_prob is None:
            base_prob = anchor
        base = base_prob.detach().clamp(EPS, 1.0 - EPS)
        if flow_px is None:
            flow_px = anchor.new_zeros((anchor.shape[0], 2, *anchor.shape[-2:]))
        flow = flow_px.detach().to(anchor)
        hw = tuple(anchor.shape[-2:])

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
        centers, region = self._region_support(score)
        views = self._counterfactual_views(anchor)
        common = self._common_visual(image, semantic_map, text_features, fine_feature_map, hw)
        coords = self._coords(anchor)
        view_logits, view_probs = self._decode_views(
            views, common, base, flow, margin, mc_std, coords
        )

        hard_views = view_probs >= 0.5
        unanimous_fg = hard_views.all(dim=1, keepdim=True)
        unanimous_bg = (~hard_views).all(dim=1, keepdim=True)
        consensus = (unanimous_fg | unanimous_bg) & region
        consensus_label = unanimous_fg
        anchor_hard = anchor >= 0.5
        edit = consensus & (consensus_label != anchor_hard)
        mean_prob = view_probs.mean(dim=1, keepdim=True)
        # ``edit`` is a non-learned commit decision.  Detach it so gradients, if
        # final segmentation loss is enabled, flow only through the agreed local
        # reconstruction values and never through a hidden straight-through gate.
        edit_detached = edit.detach()
        final_prob = torch.where(edit_detached, mean_prob, anchor).clamp(EPS, 1.0 - EPS)
        final_logits = torch.logit(final_prob)
        delta = final_logits - torch.logit(anchor)

        # Compatibility fields keep the surrounding GEOTR diagnostic/export code
        # stable while making it explicit that R4/R4.1 synthetic action heads are
        # absent from the C2R graph.
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
            "trace": self._trace(base, anchor, flow),
            "fine_feature_map": common,
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
            # C2R-native fields.
            "c2r_center_mask": centers.to(anchor),
            "c2r_region_mask": region.to(anchor),
            "c2r_view_logits": view_logits,
            "c2r_view_probs": view_probs,
            "c2r_mean_prob": mean_prob,
            "c2r_consensus_mask": consensus.to(anchor),
            "c2r_edit_mask": edit.to(anchor),
            "c2r_eroded_anchor": views[0],
            "c2r_factual_anchor": views[1],
            "c2r_dilated_anchor": views[2],
        }
