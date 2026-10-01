"""PC2R-v3/v3.1: Posterior-Consistent Canonical Residual Refinement.

This module is the root-causal successor of Canonical C2R-v2.  It keeps the
validated WHERE contract (deterministic non-overlapping uncertainty ROIs) but
changes WHAT/VERIFY so the certificate is no longer self-confirming.

Scientific contract
-------------------
1. WHERE is unchanged: deterministic non-overlapping uncertainty ROIs.
2. Counterfactual views are *actual on-policy MC posterior hypotheses* rather
   than hand-made erosion/dilation masks.  A3 receives the same posterior
   samples after applying the deployed mean Geometry flow to each sample.
3. For every ROI, three posterior hypotheses are chosen deterministically by a
   farthest-point rule.  This preserves real posterior diversity without RNG.
4. The local decoder sees the factual anchor P0 once, the view-specific logit
   deviation logit(Pv)-logit(P0), visual/semantic/text evidence, uncertainty,
   Geometry flow-only evidence, and ROI-relative coordinates.  Base/P0 is not
   redundantly repeated in a pseudo-transport trace.
5. PC2R-v3 predicts each view around its own posterior coordinate.  The v3.1
   root fix instead interprets every view output as a *canonical-coordinate*
   correction of the same factual anchor: Qv=sigmoid(logit(P0)+Cv).  Posterior
   hypotheses remain distinct conditioning evidence through Zv-Z0, but the
   residual target/reference is shared across views.
6. Deployment aggregates the same-coordinate corrections around the factual
   anchor: Qcan=sigmoid(logit(P0)+mean_v Cv).  At zero-init Qcan=P0 exactly.
   This removes the v3 reference mismatch between view losses (Zv+Dzv) and
   canonical deployment (Z0+mean Dzv).
7. VERIFY certifies *correction direction/strength consistency* across posterior
   views, not merely equality of final hard labels.  Ownership is tied to local
   risk support rather than a fixed center seed.
8. Connected components are committed atomically.  Rejected components and all
   pixels outside committed components are exact factual-anchor identity.
9. GT never enters forward.  Base/Transport/posterior samples are detached, so
   Stage-2 gradients cannot alter the upstream anchor or Geometry trajectory.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .c2r_canonical_roi_refiner import (
    CanonicalCounterfactualROIRefiner,
    _ConvNormGELU,
    _groups,
    EPS,
)


class PosteriorConsistentCanonicalROIRefiner(CanonicalCounterfactualROIRefiner):
    """True posterior-conditioned local logit-residual refiner."""

    def __init__(
        self,
        hidden_dim: int,
        semantic_channels: int,
        text_dim: int,
        *,
        fine_feature_channels: int = 512,
        num_regions: int = 4,
        region_size: int = 33,
        selection_score: str = "margin",
        flow_scale_px: float = 8.0,
        min_center_distance: Optional[int] = None,
        component_min_area: int = 2,
        residual_logit_scale: float = 2.0,
        num_posterior_views: int = 3,
        risk_top_fraction: float = 0.20,
        require_risk_overlap: bool = True,
        direction_agreement_threshold: float = 0.80,
        residual_spread_threshold: float = 1.00,
        residual_strength_threshold: float = 0.05,
        raw_correction_threshold: float = 0.02,
        reliance_diagnostics: bool = True,
        canonical_coordinate: bool = False,
    ) -> None:
        # Build/reuse the robust ROI/image/text machinery from v2.  The old
        # morphology/certificate decoder is replaced immediately below.
        super().__init__(
            hidden_dim,
            semantic_channels,
            text_dim,
            fine_feature_channels=fine_feature_channels,
            num_regions=num_regions,
            region_size=region_size,
            counterfactual_radius=1,
            selection_score=selection_score,
            flow_scale_px=flow_scale_px,
            min_center_distance=min_center_distance,
            seed_radius=0,
            component_agreement_threshold=0.0,
            component_spread_threshold=1.0e9,
            component_confidence_threshold=0.0,
            component_min_area=component_min_area,
            canonical_residual_scale=1.0,
        )
        self.num_posterior_views = int(num_posterior_views)
        if self.num_posterior_views != 3:
            raise ValueError("PC2R-v3 root contract currently requires exactly 3 posterior views")
        self.residual_logit_scale = float(residual_logit_scale)
        if self.residual_logit_scale <= 0.0:
            raise ValueError("GEOTR_PC2R_RESIDUAL_LOGIT_SCALE must be > 0")
        self.risk_top_fraction = float(risk_top_fraction)
        if not (0.0 < self.risk_top_fraction <= 1.0):
            raise ValueError("GEOTR_PC2R_RISK_TOP_FRACTION must be in (0,1]")
        self.require_risk_overlap = bool(require_risk_overlap)
        self.direction_agreement_threshold = float(direction_agreement_threshold)
        self.residual_spread_threshold = float(residual_spread_threshold)
        self.residual_strength_threshold = float(residual_strength_threshold)
        self.raw_correction_threshold = float(raw_correction_threshold)
        self.reliance_diagnostics = bool(reliance_diagnostics)
        self.canonical_coordinate = bool(canonical_coordinate)
        if not (0.0 <= self.direction_agreement_threshold <= 1.0):
            raise ValueError("GEOTR_PC2R_DIRECTION_AGREEMENT_THRESHOLD must be in [0,1]")
        if self.residual_spread_threshold < 0.0:
            raise ValueError("GEOTR_PC2R_RESIDUAL_SPREAD_THRESHOLD must be >=0")
        if self.residual_strength_threshold < 0.0:
            raise ValueError("GEOTR_PC2R_RESIDUAL_STRENGTH_THRESHOLD must be >=0")
        if self.raw_correction_threshold < 0.0:
            raise ValueError("GEOTR_PC2R_RAW_CORRECTION_THRESHOLD must be >=0")

        # Factorized decoder input:
        # common visual H
        # factual anchor P0 exactly once
        # view-specific logit deviation Zv-Z0
        # margin uncertainty
        # posterior MC std
        # factual Geometry flow only: dx,dy,|flow|
        # ROI-relative coordinates
        decoder_in = hidden_dim + 1 + 1 + 1 + 1 + 3 + 2
        self.region_decoder = nn.Sequential(
            _ConvNormGELU(decoder_in, hidden_dim, 3, dilation=1),
            _ConvNormGELU(hidden_dim, hidden_dim, 3, dilation=2),
            _ConvNormGELU(hidden_dim, hidden_dim, 3, dilation=1),
        )
        self.region_residual_out = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        # Exact canonical identity and no hard candidate at initialization.
        nn.init.zeros_(self.region_residual_out.weight)
        nn.init.zeros_(self.region_residual_out.bias)

    @staticmethod
    def _normalize_samples(samples: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        s = samples.detach().to(anchor)
        if s.ndim == 4:  # [S,B,H,W]
            s = s[:, :, None]
        if s.ndim != 5 or s.shape[2] != 1:
            raise ValueError(
                "posterior_probability_samples must be [S,B,H,W] or [S,B,1,H,W], got "
                f"{tuple(s.shape)}"
            )
        if s.shape[1] != anchor.shape[0]:
            raise ValueError(
                f"posterior batch mismatch samples={s.shape[1]} anchor={anchor.shape[0]}"
            )
        if tuple(s.shape[-2:]) != tuple(anchor.shape[-2:]):
            ss, bb = s.shape[:2]
            s = F.interpolate(
                s.reshape(ss * bb, 1, *s.shape[-2:]),
                size=anchor.shape[-2:], mode="bilinear", align_corners=False,
            ).reshape(ss, bb, 1, *anchor.shape[-2:])
        if s.shape[0] < 3:
            raise ValueError("PC2R-v3 requires at least 3 MC posterior samples")
        return s.clamp(EPS, 1.0 - EPS)

    def _flow_evidence(self, flow_px: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        flow = flow_px.detach().to(anchor)
        if tuple(flow.shape[-2:]) != tuple(anchor.shape[-2:]):
            flow = self._resize(flow, tuple(anchor.shape[-2:]))
        flow = (flow / self.flow_scale_px).clamp(-1.0, 1.0)
        mag = torch.sqrt(flow[:, 0:1].square() + flow[:, 1:2].square() + 1.0e-12).clamp(0.0, 1.0)
        return torch.cat([flow[:, 0:1], flow[:, 1:2], mag], dim=1)

    def _crop_posterior_samples(
        self, samples: torch.Tensor, grid: torch.Tensor, k: int
    ) -> torch.Tensor:
        # [S,B,1,H,W] -> [B,K,S,1,R,R]
        s, b = samples.shape[:2]
        flat = samples.permute(1, 0, 2, 3, 4).reshape(b * s, 1, *samples.shape[-2:])
        # Reuse each ROI grid for all S posterior samples.
        grid_bk = grid.view(b, k, self.region_size, self.region_size, 2)
        grid_bks = grid_bk[:, :, None].expand(b, k, s, -1, -1, -1).reshape(
            b * k * s, self.region_size, self.region_size, 2
        )
        flat_bks = samples.permute(1, 0, 2, 3, 4)[:, None].expand(
            b, k, s, 1, *samples.shape[-2:]
        ).reshape(b * k * s, 1, *samples.shape[-2:])
        patch = F.grid_sample(
            flat_bks, grid_bks, mode="bilinear", padding_mode="border", align_corners=True
        )
        return patch.view(b, k, s, 1, self.region_size, self.region_size)

    @staticmethod
    def _masked_distance(a: torch.Tensor, b: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # a,b [...,R,R], valid [R,R]
        w = valid.to(a)
        return ((a - b).abs() * w).sum(dim=(-2, -1)) / w.sum().clamp_min(1.0)

    def _select_diverse_views(
        self,
        posterior_patch: torch.Tensor,
        anchor_patch: torch.Tensor,
        valid_patch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Deterministic farthest-point selection of 3 actual posterior views.

        Returns selected probabilities [B,K,3,R,R], indices [B,K,3], and mean
        pairwise selected diversity scalar.
        """
        b, k, s, _, r, _ = posterior_patch.shape
        selected = posterior_patch.new_empty((b, k, 3, r, r))
        indices = torch.zeros((b, k, 3), dtype=torch.long, device=posterior_patch.device)
        diversity_sum = posterior_patch.new_zeros(())
        diversity_n = posterior_patch.new_zeros(())
        with torch.no_grad():
            for bi in range(b):
                for ki in range(k):
                    valid = valid_patch[bi, ki, 0]
                    if not bool(valid.any().item()):
                        selected[bi, ki] = anchor_patch[bi, ki, 0][None].expand(3, -1, -1)
                        continue
                    cand = posterior_patch[bi, ki, :, 0]
                    ref = anchor_patch[bi, ki, 0]
                    d0 = torch.stack([self._masked_distance(cand[j], ref, valid) for j in range(s)])
                    i0 = int(torch.argmax(d0).item())
                    d1 = torch.stack([self._masked_distance(cand[j], cand[i0], valid) for j in range(s)])
                    i1 = int(torch.argmax(d1).item())
                    d_to_0 = d1
                    d_to_1 = torch.stack([self._masked_distance(cand[j], cand[i1], valid) for j in range(s)])
                    min_d = torch.minimum(d_to_0, d_to_1)
                    min_d[i0] = -1.0
                    min_d[i1] = -1.0
                    i2 = int(torch.argmax(min_d).item())
                    ids = [i0, i1, i2]
                    indices[bi, ki] = torch.tensor(ids, device=indices.device)
                    selected[bi, ki] = cand[ids]
                    for aa, bb in ((0, 1), (0, 2), (1, 2)):
                        diversity_sum += self._masked_distance(cand[ids[aa]], cand[ids[bb]], valid)
                        diversity_n += 1.0
        return selected, indices, diversity_sum / diversity_n.clamp_min(1.0)

    def _risk_support(self, risk_patch: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Per-ROI top-risk ownership support [B,K,R,R] bool."""
        b, k, _, r, _ = risk_patch.shape
        out = torch.zeros((b, k, r, r), dtype=torch.bool, device=risk_patch.device)
        with torch.no_grad():
            for bi in range(b):
                for ki in range(k):
                    m = valid[bi, ki, 0]
                    n = int(m.sum().item())
                    if n <= 0:
                        continue
                    count = max(1, int(round(self.risk_top_fraction * n)))
                    vals = risk_patch[bi, ki, 0][m]
                    kth = torch.topk(vals, k=min(count, vals.numel()), largest=True).values[-1]
                    out[bi, ki] = m & (risk_patch[bi, ki, 0] >= kth)
        return out

    def _component_commit_v3(
        self,
        canonical_patch: torch.Tensor,   # [B,K,1,R,R]
        anchor_patch: torch.Tensor,      # [B,K,1,R,R]
        delta_views: torch.Tensor,       # [B,K,3,R,R]
        valid: torch.Tensor,             # [B,K,1,R,R]
        risk_patch: torch.Tensor,        # [B,K,1,R,R]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        b, k, _, r, _ = canonical_patch.shape
        with torch.no_grad():
            mean_delta = delta_views.mean(dim=2)
            anchor_h = anchor_patch[:, :, 0] >= 0.5
            can_h = canonical_patch[:, :, 0] >= 0.5
            candidate = (can_h != anchor_h) & valid[:, :, 0]
            raw_correction = (mean_delta.abs() >= self.raw_correction_threshold) & valid[:, :, 0]

            pos_all = (delta_views > 0).all(dim=2)
            neg_all = (delta_views < 0).all(dim=2)
            direction_agree = pos_all | neg_all
            delta_spread = delta_views.max(dim=2).values - delta_views.min(dim=2).values
            delta_strength = mean_delta.abs()
            risk_support = self._risk_support(risk_patch, valid)

            labels = self._connected_labels(candidate.reshape(b * k, r, r)).view(b, k, r, r)
            commit = torch.zeros_like(candidate)
            # PC2R-v3.2 root-autopsy masks.  These are cumulative deployment
            # stages, not alternative policies: candidate -> area -> risk ->
            # direction -> spread -> strength(final commit).  Exposing them lets
            # validation measure the exact native metric change introduced by
            # each non-differentiable deployment operator.
            area_stage = torch.zeros_like(candidate)
            risk_stage = torch.zeros_like(candidate)
            direction_stage = torch.zeros_like(candidate)
            spread_stage = torch.zeros_like(candidate)
            strength_stage = torch.zeros_like(candidate)

            # Global totals.  These are totals (not batch means), so train.py can
            # aggregate them without the old zero-batch area bias.
            z = anchor_patch.new_zeros(())
            raw_components = z.clone()
            area_pass = z.clone()
            risk_pass = z.clone()
            direction_pass = z.clone()
            spread_pass = z.clone()
            strength_pass = z.clone()
            all_pass = z.clone()
            candidate_area_total = z.clone()
            committed_area_total = z.clone()
            direction_sum = z.clone()
            spread_sum = z.clone()
            strength_sum = z.clone()
            risk_overlap_sum = z.clone()

            for bi in range(b):
                for ki in range(k):
                    ids = torch.unique(labels[bi, ki])
                    ids = ids[ids > 0]
                    for cid in ids:
                        comp = labels[bi, ki] == cid
                        raw_components += 1.0
                        area = comp.float().sum()
                        candidate_area_total += area
                        pass_area = int(area.item()) >= self.component_min_area
                        if not pass_area:
                            continue
                        area_pass += 1.0
                        area_stage[bi, ki] |= comp

                        risk_hit = bool((risk_support[bi, ki] & comp).any().item())
                        pass_risk = (not self.require_risk_overlap) or risk_hit
                        if pass_risk:
                            risk_pass += 1.0
                            risk_stage[bi, ki] |= comp

                        d_agree = direction_agree[bi, ki][comp].float().mean()
                        sp_q90 = torch.quantile(delta_spread[bi, ki][comp], 0.90)
                        st_q10 = torch.quantile(delta_strength[bi, ki][comp], 0.10)
                        risk_frac = (risk_support[bi, ki] & comp).float().sum() / area.clamp_min(1.0)
                        direction_sum += d_agree
                        spread_sum += sp_q90
                        strength_sum += st_q10
                        risk_overlap_sum += risk_frac

                        pass_dir = float(d_agree.item()) >= self.direction_agreement_threshold
                        pass_spread = float(sp_q90.item()) <= self.residual_spread_threshold
                        pass_strength = float(st_q10.item()) >= self.residual_strength_threshold
                        if pass_dir:
                            direction_pass += 1.0
                        if pass_spread:
                            spread_pass += 1.0
                        if pass_strength:
                            strength_pass += 1.0

                        # Cumulative masks mirror the actual deployment order.
                        if pass_risk and pass_dir:
                            direction_stage[bi, ki] |= comp
                        if pass_risk and pass_dir and pass_spread:
                            spread_stage[bi, ki] |= comp
                        if pass_risk and pass_dir and pass_spread and pass_strength:
                            strength_stage[bi, ki] |= comp

                        passed = pass_risk and pass_dir and pass_spread and pass_strength
                        if passed:
                            commit[bi, ki] |= comp
                            all_pass += 1.0
                            committed_area_total += area

            denom = area_pass.clamp_min(1.0)
            stats = {
                "candidate_mask": candidate[:, :, None].to(anchor_patch),
                "raw_correction_mask": raw_correction[:, :, None].to(anchor_patch),
                "risk_support_mask": risk_support[:, :, None].to(anchor_patch),
                "area_stage_mask": area_stage[:, :, None].to(anchor_patch),
                "risk_stage_mask": risk_stage[:, :, None].to(anchor_patch),
                "direction_stage_mask": direction_stage[:, :, None].to(anchor_patch),
                "spread_stage_mask": spread_stage[:, :, None].to(anchor_patch),
                "strength_stage_mask": strength_stage[:, :, None].to(anchor_patch),
                "candidate_component_count": area_pass,
                "committed_component_count": all_pass,
                "candidate_component_area_total": candidate_area_total,
                "committed_component_area_total": committed_area_total,
                "candidate_component_area_mean": candidate_area_total / area_pass.clamp_min(1.0),
                "committed_component_area_mean": committed_area_total / all_pass.clamp_min(1.0),
                # Compatibility aliases reinterpret agreement/spread/confidence
                # as correction-direction agreement, residual spread, strength.
                "component_agreement_mean": direction_sum / denom,
                "component_spread_q90_mean": spread_sum / denom,
                "component_confidence_q10_mean": strength_sum / denom,
                "component_risk_overlap_mean": risk_overlap_sum / denom,
                "component_raw_count": raw_components,
                "component_area_pass_count": area_pass,
                "component_risk_pass_count": risk_pass,
                "component_direction_pass_count": direction_pass,
                "component_spread_pass_count": spread_pass,
                "component_strength_pass_count": strength_pass,
                "component_all_pass_count": all_pass,
            }
        return commit[:, :, None], stats

    def _decode_from_view_delta(
        self,
        common_p: torch.Tensor,
        anchor_p: torch.Tensor,
        view_delta_logit: torch.Tensor,
        margin_p: torch.Tensor,
        mc_std_p: torch.Tensor,
        flow_p: torch.Tensor,
        rel: torch.Tensor,
    ) -> torch.Tensor:
        # Shapes [B,K,V?,C,R,R].  Caller supplies one view axis in delta.
        inp = torch.cat(
            [common_p, anchor_p, view_delta_logit, margin_p, mc_std_p, flow_p, rel],
            dim=2,
        )
        b, k, c, r, _ = inp.shape
        raw = self.region_residual_out(self.region_decoder(inp.reshape(b * k, c, r, r)))
        return self.residual_logit_scale * torch.tanh(raw.view(b, k, r, r))

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
        del supervision_masks
        anchor = anchor_prob.detach()
        if anchor.ndim == 3:
            anchor = anchor[:, None]
        anchor = anchor.clamp(EPS, 1.0 - EPS)
        if base_prob is None:
            base_prob = anchor
        base = base_prob.detach()
        if base.ndim == 3:
            base = base[:, None]
        base = base.clamp(EPS, 1.0 - EPS)
        if flow_px is None:
            flow_px = anchor.new_zeros((anchor.shape[0], 2, *anchor.shape[-2:]))
        flow = flow_px.detach().to(anchor)
        if not isinstance(posterior_probability_samples, torch.Tensor):
            raise RuntimeError(
                "PC2R-v3 requires actual mc_probability_samples; morphology fallback is intentionally disabled"
            )
        posterior = self._normalize_samples(posterior_probability_samples, anchor)

        b, _, h, w = anchor.shape
        hw = (h, w)
        margin = self._margin_uncertainty(anchor)
        mc_std = self._fit_evidence(mc_std_map, anchor, scale=2.0)
        mc_dis = self._fit_evidence(mc_disagreement_map, anchor, scale=1.0)
        entropy = self._entropy(anchor)
        evidence = {"margin": margin, "mc_std": mc_std, "mc_disagreement": mc_dis, "entropy": entropy}
        score = self._selection_score(evidence)
        centers, center_yx, center_valid = self._greedy_centers(score)
        ay, ax, grid, valid_patch, rel = self._patch_geometry(center_yx, center_valid, h, w, anchor.dtype)
        k = self.num_regions

        common = self._common_visual(image, semantic_map, text_features, fine_feature_map, hw)
        flow_evidence = self._flow_evidence(flow, anchor)
        common_p = self._crop(common, grid, k)
        anchor_p = self._crop(anchor, grid, k)
        margin_p = self._crop(margin, grid, k)
        mc_std_p = self._crop(mc_std, grid, k)
        flow_p = self._crop(flow_evidence, grid, k)
        posterior_p = self._crop_posterior_samples(posterior, grid, k)
        selected_p, selected_idx, selected_diversity = self._select_diverse_views(
            posterior_p, anchor_p, valid_patch
        )  # [B,K,3,R,R]

        z0 = torch.logit(anchor_p[:, :, 0].clamp(EPS, 1.0 - EPS))
        selected_logits = torch.logit(selected_p.clamp(EPS, 1.0 - EPS))
        selected_center_bias_map = selected_logits.mean(dim=2) - z0
        valid_center = valid_patch[:, :, 0].to(selected_center_bias_map)
        center_den = valid_center.sum().clamp_min(1.0)
        selected_center_bias_abs = (selected_center_bias_map.abs() * valid_center).sum() / center_den
        selected_center_bias_signed = (selected_center_bias_map * valid_center).sum() / center_den

        # Root-Audit++: compare the selected farthest-3 posterior center against
        # the full transported posterior center in BOTH probability and logit
        # coordinates.  These are forward-only diagnostics and cannot affect the
        # deployed prediction.
        all_p = posterior_p[:, :, :, 0].clamp(EPS, 1.0 - EPS)
        all_z = torch.logit(all_p)
        all_logit_bias_map = all_z.mean(dim=2) - z0
        all_prob_bias_map = all_p.mean(dim=2) - anchor_p[:, :, 0]
        selected_prob_bias_map = selected_p.mean(dim=2) - anchor_p[:, :, 0]
        selected_vs_all_logit_map = selected_logits.mean(dim=2) - all_z.mean(dim=2)
        all_center_bias_logit_abs = (all_logit_bias_map.abs() * valid_center).sum() / center_den
        all_center_bias_logit_signed = (all_logit_bias_map * valid_center).sum() / center_den
        all_center_bias_prob_abs = (all_prob_bias_map.abs() * valid_center).sum() / center_den
        selected_center_bias_prob_abs = (selected_prob_bias_map.abs() * valid_center).sum() / center_den
        selected_vs_all_logit_bias_abs = (selected_vs_all_logit_map.abs() * valid_center).sum() / center_den

        delta_views = []
        q_views = []
        for vi in range(3):
            pv = selected_p[:, :, vi].clamp(EPS, 1.0 - EPS)
            zv = selected_logits[:, :, vi]
            dev = (zv - z0).clamp(-8.0, 8.0)[:, :, None]
            dz = self._decode_from_view_delta(
                common_p, anchor_p, dev, margin_p, mc_std_p, flow_p, rel
            )
            # v3.1 root fix: every posterior hypothesis estimates the correction
            # of the *same* factual anchor.  v3 is retained by the false branch
            # for exact paper ablation/backward compatibility.
            view_reference = z0 if self.canonical_coordinate else zv
            qv = torch.sigmoid(view_reference + dz).clamp(EPS, 1.0 - EPS)
            delta_views.append(dz)
            q_views.append(qv)
        delta_views = torch.stack(delta_views, dim=2)  # [B,K,3,R,R]
        patch_probs = torch.stack(q_views, dim=2)      # [B,K,3,R,R]
        patch_view_logits = torch.logit(patch_probs)

        mean_delta = delta_views.mean(dim=2)
        _valid3 = valid_patch[:, :, 0][:, :, None].to(delta_views)
        _den3 = (_valid3.sum() * float(delta_views.shape[2])).clamp_min(1.0)
        branch_deviation_rms = torch.sqrt(
            (((delta_views - mean_delta[:, :, None]).square()) * _valid3).sum() / _den3 + 1.0e-12
        )
        branch_direction_unanimity = ((((delta_views > 0).all(dim=2) | (delta_views < 0).all(dim=2)).to(delta_views) * valid_center).sum() / center_den)
        canonical_patch = torch.sigmoid(z0 + mean_delta).clamp(EPS, 1.0 - EPS)[:, :, None]

        full_view_probs, region, overlap = self._scatter_patch_values(
            patch_probs, anchor, ay, ax, valid_patch
        )
        full_view_logits = torch.logit(full_view_probs.clamp(EPS, 1.0 - EPS))
        canonical_full3, _, _ = self._scatter_patch_values(
            canonical_patch, anchor, ay, ax, valid_patch
        )
        canonical_prob = canonical_full3[:, 0:1].clamp(EPS, 1.0 - EPS)

        commit_patch, comp_stats = self._component_commit_v3(
            canonical_patch, anchor_p, delta_views, valid_patch, margin_p
        )
        candidate_full, _, _ = self._scatter_patch_values(
            comp_stats["candidate_mask"], torch.zeros_like(anchor), ay, ax, valid_patch
        )
        candidate_full = candidate_full[:, 0:1] > 0.5
        raw_full, _, _ = self._scatter_patch_values(
            comp_stats["raw_correction_mask"], torch.zeros_like(anchor), ay, ax, valid_patch
        )
        raw_full = raw_full[:, 0:1] > 0.5
        risk_full, _, _ = self._scatter_patch_values(
            comp_stats["risk_support_mask"], torch.zeros_like(anchor), ay, ax, valid_patch
        )
        risk_full = risk_full[:, 0:1] > 0.5
        commit_full, _, _ = self._scatter_patch_values(
            commit_patch, torch.zeros_like(anchor), ay, ax, valid_patch
        )
        commit_full = commit_full[:, 0:1] > 0.5

        def _scatter_stage(mask_patch: torch.Tensor) -> torch.Tensor:
            full, _, _ = self._scatter_patch_values(
                mask_patch, torch.zeros_like(anchor), ay, ax, valid_patch
            )
            return full[:, 0:1] > 0.5

        area_full = _scatter_stage(comp_stats["area_stage_mask"])
        risk_stage_full = _scatter_stage(comp_stats["risk_stage_mask"])
        direction_stage_full = _scatter_stage(comp_stats["direction_stage_mask"])
        spread_stage_full = _scatter_stage(comp_stats["spread_stage_mask"])
        strength_stage_full = _scatter_stage(comp_stats["strength_stage_mask"])

        # Exact operator-decomposition probabilities.  Each stage writes the
        # same canonical proposal but with progressively stricter deterministic
        # support; these are diagnostics and do not alter the deployable Final.
        stage_candidate_prob = torch.where(candidate_full, canonical_prob, anchor).clamp(EPS, 1.0 - EPS)
        stage_area_prob = torch.where(area_full, canonical_prob, anchor).clamp(EPS, 1.0 - EPS)
        stage_risk_prob = torch.where(risk_stage_full, canonical_prob, anchor).clamp(EPS, 1.0 - EPS)
        stage_direction_prob = torch.where(direction_stage_full, canonical_prob, anchor).clamp(EPS, 1.0 - EPS)
        stage_spread_prob = torch.where(spread_stage_full, canonical_prob, anchor).clamp(EPS, 1.0 - EPS)
        stage_strength_prob = torch.where(strength_stage_full, canonical_prob, anchor).clamp(EPS, 1.0 - EPS)

        final_prob = torch.where(commit_full, canonical_prob, anchor).clamp(EPS, 1.0 - EPS)
        final_logits = torch.logit(final_prob)
        delta_logit = final_logits - torch.logit(anchor)

        # Correction-direction consensus map for compatibility/diagnostics.
        dpos = (delta_views > 0).all(dim=2)
        dneg = (delta_views < 0).all(dim=2)
        dir_cons_patch = (dpos | dneg)[:, :, None].to(anchor)
        dir_cons_full, _, _ = self._scatter_patch_values(
            dir_cons_patch, torch.zeros_like(anchor), ay, ax, valid_patch
        )
        dir_cons_full = dir_cons_full[:, 0:1] > 0.5

        # Eval-only counterfactual reliance probes.  These do not affect deployable
        # output or training RNG/gradient.  factualized means view deviation=0;
        # shuffled swaps posterior deviation between separated ROIs.
        reliance_fact = anchor.new_zeros(())
        reliance_shuffle = anchor.new_zeros(())
        if (not self.training) and self.reliance_diagnostics:
            with torch.no_grad():
                factual_deltas = []
                shuffled_deltas = []
                for vi in range(3):
                    zero_dev = torch.zeros_like(anchor_p)
                    dz0 = self._decode_from_view_delta(
                        common_p, anchor_p, zero_dev, margin_p, mc_std_p, flow_p, rel
                    )
                    factual_deltas.append(dz0)
                    pv = selected_p[:, :, vi].roll(shifts=1, dims=1).clamp(EPS, 1.0-EPS)
                    shuf_dev = (torch.logit(pv) - z0).clamp(-8.0, 8.0)[:, :, None]
                    dzs = self._decode_from_view_delta(
                        common_p, anchor_p, shuf_dev, margin_p, mc_std_p, flow_p, rel
                    )
                    shuffled_deltas.append(dzs)
                fd = torch.stack(factual_deltas, dim=2)
                sd = torch.stack(shuffled_deltas, dim=2)
                vm = valid_patch[:, :, 0][:, :, None].to(delta_views)
                den = vm.sum().clamp_min(1.0) * 3.0
                reliance_fact = ((delta_views - fd).abs() * vm).sum() / den
                reliance_shuffle = ((delta_views - sd).abs() * vm).sum() / den

        # ROI geometry diagnostics.
        roi_pixel_count = region.float().sum()
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
        # Compatibility keys keep v2 loss/validation code functional.  The
        # eroded/factual/dilated names now map to the three selected posterior
        # hypotheses and are not used scientifically by v3.
        selected_full, _, _ = self._scatter_patch_values(selected_p, anchor, ay, ax, valid_patch)
        return {
            "logits": final_logits,
            "prob": final_prob,
            "selection_mask": region.to(anchor),
            "selection_score": score,
            "refined_logits": torch.logit(canonical_prob),
            "refined_prob": canonical_prob,
            "delta_logit": delta_logit,
            "margin_uncertainty": margin,
            "mc_std_map": mc_std,
            "mc_disagreement_map": mc_dis,
            "entropy_map": entropy,
            "trace": flow_evidence,
            "fine_feature_map": common,
            "dn_anchor_prob": anchor,
            "dn_corruption_mask": z,
            "dn_selection_mask": z,
            "dn_refined_logits": torch.logit(anchor),
            "dn_refined_prob": anchor,
            "dn_final_prob": anchor,
            "dn_delta_logit": z,
            "dn_op_id": op,
            "r4_flip_logits": z, "r4_flip_prob": z, "r4_flip_mask": z,
            "r4_synth_flip_logits": z, "r4_synth_flip_prob": z,
            "r4_synth_flip_mask": z, "r4_synth_target": z,
            "r2_enabled": anchor.new_zeros((anchor.shape[0],)),
            "r3_enabled": anchor.new_zeros((anchor.shape[0],)),
            "r4_enabled": anchor.new_zeros((anchor.shape[0],)),
            "r41_enabled": anchor.new_zeros((anchor.shape[0],)),
            "c2r_center_mask": centers.to(anchor),
            "c2r_region_mask": region.to(anchor),
            "c2r_view_logits": full_view_logits,
            "c2r_view_probs": full_view_probs,
            "c2r_mean_prob": canonical_prob,
            "c2r_consensus_mask": dir_cons_full.to(anchor),
            "c2r_edit_mask": commit_full.to(anchor),
            "c2r_eroded_anchor": selected_full[:, 0:1],
            "c2r_factual_anchor": selected_full[:, 1:2],
            "c2r_dilated_anchor": selected_full[:, 2:3],
            "c2r_v2_enabled": anchor.new_ones((anchor.shape[0],)),
            "pc2r_v3_enabled": anchor.new_ones((anchor.shape[0],)),
            "pc2r_v31_enabled": anchor.new_full((anchor.shape[0],), 1.0 if self.canonical_coordinate else 0.0),
            "c2r_center_yx": center_yx,
            "c2r_center_valid": center_valid.to(anchor),
            "c2r_roi_overlap_pixel_count": overlap,
            "c2r_roi_unique_pixel_count": roi_pixel_count,
            "c2r_center_min_chebyshev_distance": min_dist,
            "c2r_candidate_mask": candidate_full.to(anchor),
            "c2r_raw_correction_mask": raw_full.to(anchor),
            "c2r_risk_support_mask": risk_full.to(anchor),
            "c2r_commit_mask": commit_full.to(anchor),
            "pc2r_stage_candidate_mask": candidate_full.to(anchor),
            "pc2r_stage_area_mask": area_full.to(anchor),
            "pc2r_stage_risk_mask": risk_stage_full.to(anchor),
            "pc2r_stage_direction_mask": direction_stage_full.to(anchor),
            "pc2r_stage_spread_mask": spread_stage_full.to(anchor),
            "pc2r_stage_strength_mask": strength_stage_full.to(anchor),
            "pc2r_stage_candidate_prob": stage_candidate_prob,
            "pc2r_stage_area_prob": stage_area_prob,
            "pc2r_stage_risk_prob": stage_risk_prob,
            "pc2r_stage_direction_prob": stage_direction_prob,
            "pc2r_stage_spread_prob": stage_spread_prob,
            "pc2r_stage_strength_prob": stage_strength_prob,
            "c2r_candidate_component_count": comp_stats["candidate_component_count"],
            "c2r_committed_component_count": comp_stats["committed_component_count"],
            "c2r_candidate_component_area_total": comp_stats["candidate_component_area_total"],
            "c2r_committed_component_area_total": comp_stats["committed_component_area_total"],
            "c2r_candidate_component_area_mean": comp_stats["candidate_component_area_mean"],
            "c2r_committed_component_area_mean": comp_stats["committed_component_area_mean"],
            "c2r_component_agreement_mean": comp_stats["component_agreement_mean"],
            "c2r_component_spread_q90_mean": comp_stats["component_spread_q90_mean"],
            "c2r_component_confidence_q10_mean": comp_stats["component_confidence_q10_mean"],
            "pc2r_component_risk_overlap_mean": comp_stats["component_risk_overlap_mean"],
            "pc2r_component_raw_count": comp_stats["component_raw_count"],
            "pc2r_component_area_pass_count": comp_stats["component_area_pass_count"],
            "pc2r_component_risk_pass_count": comp_stats["component_risk_pass_count"],
            "pc2r_component_direction_pass_count": comp_stats["component_direction_pass_count"],
            "pc2r_component_spread_pass_count": comp_stats["component_spread_pass_count"],
            "pc2r_component_strength_pass_count": comp_stats["component_strength_pass_count"],
            "pc2r_component_all_pass_count": comp_stats["component_all_pass_count"],
            "pc2r_posterior_view_indices": selected_idx,
            "pc2r_selected_posterior_diversity": selected_diversity,
            "pc2r_selected_center_bias_abs": selected_center_bias_abs,
            "pc2r_selected_center_bias_signed": selected_center_bias_signed,
            "pc2r_all_center_bias_logit_abs": all_center_bias_logit_abs,
            "pc2r_all_center_bias_logit_signed": all_center_bias_logit_signed,
            "pc2r_all_center_bias_prob_abs": all_center_bias_prob_abs,
            "pc2r_selected_center_bias_prob_abs": selected_center_bias_prob_abs,
            "pc2r_selected_vs_all_logit_bias_abs": selected_vs_all_logit_bias_abs,
            "pc2r_branch_deviation_rms": branch_deviation_rms,
            "pc2r_branch_direction_unanimity": branch_direction_unanimity,
            "pc2r_patch_delta_logits": delta_views,
            "pc2r_mean_abs_delta_logit": mean_delta.abs().mean(),
            "pc2r_reliance_factualized": reliance_fact,
            "pc2r_reliance_shuffled": reliance_shuffle,
            "c2r_patch_view_logits": patch_view_logits,
            "c2r_patch_view_probs": patch_probs,
            "c2r_patch_canonical_prob": canonical_patch,
        }


class CanonicalCoordinatePosteriorROIRefiner(PosteriorConsistentCanonicalROIRefiner):
    """PC2R-v3.1: posterior evidence, factual/canonical residual coordinates.

    Posterior hypotheses remain actual on-policy conditions, but each decoder
    branch estimates the correction of the same factual anchor Z0.  Therefore
    per-view supervision and canonical aggregation share one residual reference.
    """
    def __init__(self, *args, **kwargs) -> None:
        kwargs["canonical_coordinate"] = True
        super().__init__(*args, **kwargs)
