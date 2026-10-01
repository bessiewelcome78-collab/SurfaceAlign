"""Utility-consistent dual-space residual transition refiner (UCDRT).

This module is the Stage-2 successor of SLR 2.3.  It intentionally reuses the
validated SLR infrastructure (residual proposal map, true-HR crops, WOLA and
paired train-only supervision) while replacing the generic local segmentation
residual with an explicit residual-decision problem:

    WHERE  : selector proposes factual residual regions.
    VALUE  : each proposed region predicts signed executable utility; KEEP has
             fixed utility zero and therefore competes in the same action set.
    STATE  : every local pixel predicts KEEP / MOVE / ADD / REMOVE.
    ACTION : MOVE is an exact SDF-normal local warp; ADD/REMOVE are monotone
             signed logit actions.  Training and deployment use the same operator.

At initialization the state argmax is KEEP and candidate utility is exactly zero,
therefore the deployed output is exact Geometry identity even though gradients can
reach the state/action heads through straight-through estimators and direct losses.
GT is used only by training targets / paired corruption and validation diagnostics;
GT never changes inference proposals or deployment actions.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .c2r_canonical_roi_refiner import EPS, _ConvNormGELU
from .slr_local_rerenderer import GeometryConditionedSparseLocalRerenderer


class UtilityConsistentDualSpaceResidualTransitionRefiner(
    GeometryConditionedSparseLocalRerenderer
):
    """Sparse local refiner with explicit residual states and executable utility."""

    STAGE_ID = 10.0
    STATE_KEEP = 0
    STATE_MOVE = 1
    STATE_ADD = 2
    STATE_REMOVE = 3
    NUM_STATES = 4

    def __init__(
        self,
        hidden_dim: int,
        semantic_channels: int,
        text_dim: int,
        *,
        fine_feature_channels: int = 512,
        num_regions: int = 8,
        region_size: int = 33,
        flow_scale_px: float = 8.0,
        min_center_distance: Optional[int] = None,
        residual_logit_scale: float = 4.0,
        sdf_radius_px: int = 8,
        train_oracle_regions: Optional[int] = None,
        train_positive_regions: Optional[int] = None,
        train_clean_regions: int = 4,
        clean_max_error_fraction: float = 0.02,
        clean_max_sdf_discrepancy: float = 0.05,
        blend_taper_px: int = 4,
        context_size: int = 55,
        hr_size: int = 448,
        heaviside_tau_px: float = 1.0,
        require_true_hr: bool = True,
        action_margin_prob: float = 0.05,
        boundary_radius_px: int = 5,
        max_boundary_displacement_px: float = 4.0,
        max_interior_logit_step: float = 4.0,
        utility_temperature: float = 0.25,
        state_keep_bias: float = 2.0,
        interior_magnitude_init_fraction: float = 0.10,
        paired_stable_enabled: bool = True,
        paired_regions: int = 4,
    ) -> None:
        super().__init__(
            hidden_dim,
            semantic_channels,
            text_dim,
            fine_feature_channels=fine_feature_channels,
            num_regions=num_regions,
            region_size=region_size,
            flow_scale_px=flow_scale_px,
            min_center_distance=min_center_distance,
            residual_logit_scale=residual_logit_scale,
            sdf_radius_px=sdf_radius_px,
            train_oracle_regions=train_oracle_regions,
            train_positive_regions=train_positive_regions,
            train_clean_regions=train_clean_regions,
            clean_max_error_fraction=clean_max_error_fraction,
            clean_max_sdf_discrepancy=clean_max_sdf_discrepancy,
            blend_taper_px=blend_taper_px,
            context_size=context_size,
            hr_size=hr_size,
            heaviside_tau_px=heaviside_tau_px,
            require_true_hr=require_true_hr,
            action_margin_prob=action_margin_prob,
        )

        self.boundary_radius_px = max(1, int(boundary_radius_px))
        self.max_boundary_displacement_px = float(max_boundary_displacement_px)
        self.max_interior_logit_step = float(max_interior_logit_step)
        self.utility_temperature = max(float(utility_temperature), 1.0e-3)
        self.paired_stable_enabled = bool(paired_stable_enabled)
        self.paired_regions = max(1, int(paired_regions))
        if self.max_boundary_displacement_px <= 0:
            raise ValueError("GEOTR_SLR_UCDRT_MAX_BOUNDARY_DISPLACEMENT_PX must be >0")
        if self.max_interior_logit_step <= 0:
            raise ValueError("GEOTR_SLR_UCDRT_MAX_INTERIOR_LOGIT_STEP must be >0")

        # The inherited generic segmentation and non-causal absolute-SDF heads are
        # deliberately retired in UCDRT.  The shared HR decoder is retained.
        for inherited_actor_head in ("region_action_state_out", "region_action_dose_out"):
            if hasattr(self, inherited_actor_head):
                delattr(self, inherited_actor_head)
        if hasattr(self, "region_sdf_out"):
            del self.region_sdf_out

        self.transition_state_out = nn.Conv2d(hidden_dim, self.NUM_STATES, 1)
        nn.init.zeros_(self.transition_state_out.weight)
        nn.init.zeros_(self.transition_state_out.bias)
        with torch.no_grad():
            self.transition_state_out.bias[self.STATE_KEEP] = float(state_keep_bias)

        # MOVE predicts signed displacement in pixels.  Zero init => zero physical
        # motion.  The displacement is applied along the anchor-SDF normal.
        self.transition_move_out = nn.Conv2d(hidden_dim, 1, 1)
        nn.init.zeros_(self.transition_move_out.weight)
        nn.init.zeros_(self.transition_move_out.bias)

        # ADD/REMOVE share a positive magnitude; direction is supplied by STATE.
        self.transition_interior_magnitude_out = nn.Conv2d(hidden_dim, 1, 1)
        nn.init.zeros_(self.transition_interior_magnitude_out.weight)
        f = min(max(float(interior_magnitude_init_fraction), 1.0e-4), 1.0 - 1.0e-4)
        nn.init.constant_(
            self.transition_interior_magnitude_out.bias,
            float(torch.logit(torch.tensor(f)).item()),
        )

        # Candidate utility is signed. KEEP has fixed utility 0.  A zero-initialized
        # value head therefore makes every candidate a tie with KEEP and the strict
        # hard rule (>0) executes none at initialization.
        self.transition_utility_out = nn.Conv2d(hidden_dim, 1, 1)
        nn.init.zeros_(self.transition_utility_out.weight)
        nn.init.zeros_(self.transition_utility_out.bias)

    @staticmethod
    def _gradient_xy(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Central-difference x/y derivatives for [N,1,H,W]."""
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"gradient input must be [N,1,H,W], got {tuple(x.shape)}")
        kx = x.new_tensor([[0.0, 0.0, 0.0], [-0.5, 0.0, 0.5], [0.0, 0.0, 0.0]]).view(1,1,3,3)
        ky = x.new_tensor([[0.0, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]]).view(1,1,3,3)
        xp = F.pad(x, (1,1,1,1), mode="replicate")
        return F.conv2d(xp, kx), F.conv2d(xp, ky)

    @staticmethod
    def _warp_patch_logits_px(logits: torch.Tensor, flow_px: torch.Tensor) -> torch.Tensor:
        """Exact backward warp in pixel units, align_corners=False."""
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError(f"logits must be [N,1,H,W], got {tuple(logits.shape)}")
        if flow_px.ndim != 4 or flow_px.shape[1] != 2:
            raise ValueError(f"flow must be [N,2,H,W], got {tuple(flow_px.shape)}")
        n, _, h, w = logits.shape
        yy = torch.arange(h, device=logits.device, dtype=logits.dtype) + 0.5
        xx = torch.arange(w, device=logits.device, dtype=logits.dtype) + 0.5
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        x = gx[None] + flow_px[:, 0]
        y = gy[None] + flow_px[:, 1]
        grid = torch.stack([2.0 * x / float(w) - 1.0, 2.0 * y / float(h) - 1.0], dim=-1)
        return F.grid_sample(logits, grid, mode="bilinear", padding_mode="border", align_corners=False)

    def _straight_through_state(self, state_prob: torch.Tensor) -> torch.Tensor:
        hard_idx = state_prob.detach().argmax(dim=2, keepdim=True)
        hard = torch.zeros_like(state_prob).scatter_(2, hard_idx, 1.0)
        if self.training:
            return hard + state_prob - state_prob.detach()
        return hard

    def _straight_through_commit(self, utility_logit: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        utility_value = torch.tanh(utility_logit)
        commit_prob = torch.sigmoid(utility_logit / self.utility_temperature)
        hard = (utility_value.detach() > 0.0).to(commit_prob)
        commit = hard + commit_prob - commit_prob.detach() if self.training else hard
        return utility_value, commit_prob, commit

    def _decode_centers(
        self,
        center_yx,
        center_valid,
        *,
        common,
        dense_ev,
        anchor,
        anchor_sdf,
        hr_image,
    ):
        b, _, h, w = anchor.shape
        if hr_image is None:
            if self.require_true_hr:
                raise RuntimeError("GEOTR-SLR-UCDRT requires true paired HR image")
            hr_image = F.interpolate(common[:, :3], scale_factor=2, mode="bilinear", align_corners=False)
        if hr_image.ndim != 4 or hr_image.shape[0] != b or hr_image.shape[1] != 3:
            raise ValueError("UCDRT hr_image must be [B,3,Hh,Wh]")
        if hr_image.shape[-2] != h * 2 or hr_image.shape[-1] != w * 2:
            raise ValueError(
                f"UCDRT true-HR contract requires exactly 2x LR, got LR={(h,w)} HR={tuple(hr_image.shape[-2:])}"
            )

        k = center_yx.shape[1]
        _, _, cgrid, _ = self._grid_for_centers(
            center_yx, center_valid, h, w, self.context_size, anchor.dtype
        )
        oay, oax, ogrid, ovalid = self._grid_for_centers(
            center_yx, center_valid, h, w, self.region_size, anchor.dtype
        )
        common_c = self._crop_grid(common, cgrid, k)
        ev_c = self._crop_grid(dense_ev, cgrid, k)
        asdf_ctx = self._crop_grid(anchor_sdf, cgrid, k)
        asdf_c = asdf_ctx / float(self.sdf_radius_px)

        hr_grid = self._dense_normalized_hr_grid(
            center_yx,
            center_valid,
            h,
            w,
            hr_image.shape[-2],
            hr_image.shape[-1],
            self.context_size,
            2,
            hr_image.dtype,
        )
        hr_expand = hr_image[:, None].expand(-1,k,-1,-1,-1).reshape(b*k,3,*hr_image.shape[-2:])
        hr_crop = F.grid_sample(
            hr_expand, hr_grid, mode="bilinear", padding_mode="border", align_corners=False
        )
        hr_feat = self.hr_encoder(hr_crop).view(b,k,-1,self.context_size,self.context_size)
        x = torch.cat([common_c, ev_c, asdf_c, hr_feat], dim=2)
        hidden = self.region_decoder(x.reshape(b*k, x.shape[2], self.context_size, self.context_size))

        # ---------- STATE ----------
        state_logits_ctx = self.transition_state_out(hidden).view(
            b, k, self.NUM_STATES, self.context_size, self.context_size
        )
        state_prob_ctx = torch.softmax(state_logits_ctx, dim=2)
        state_st_ctx = self._straight_through_state(state_prob_ctx)

        # ---------- MOVE: causal SDF-normal warp ----------
        move_raw_ctx = self.transition_move_out(hidden).view(
            b, k, 1, self.context_size, self.context_size
        )
        move_px_ctx = self.max_boundary_displacement_px * torch.tanh(move_raw_ctx)

        asdf_bk = asdf_ctx.reshape(b*k,1,self.context_size,self.context_size)
        gx, gy = self._gradient_xy(asdf_bk)
        norm = torch.sqrt(gx.square() + gy.square()).clamp_min(1.0e-6)
        nx = gx / norm
        ny = gy / norm
        move_bk = move_px_ctx.reshape(b*k,1,self.context_size,self.context_size)
        flow_bk = torch.cat([nx * move_bk, ny * move_bk], dim=1)

        anchor_logits_full = self._logit(anchor)
        anchor_logit_ctx = self._crop_grid(anchor_logits_full, cgrid, k)
        warped_ctx = self._warp_patch_logits_px(
            anchor_logit_ctx.reshape(b*k,1,self.context_size,self.context_size), flow_bk
        ).view(b,k,1,self.context_size,self.context_size)

        anchor_prob_patch = self._crop_grid(anchor, ogrid, k)
        anchor_logit_patch = self._logit(anchor_prob_patch)
        anchor_sdf_patch = self._crop_grid(anchor_sdf, ogrid, k)
        boundary_patch = (anchor_sdf_patch.detach().abs() <= float(self.boundary_radius_px)).to(anchor)
        move_logits = self._center_crop_patch(warped_ctx, self.region_size)
        # MOVE is physically owned only by the deterministic anchor boundary band.
        move_delta = (move_logits - anchor_logit_patch) * ovalid.to(anchor) * boundary_patch
        move_px = self._center_crop_patch(move_px_ctx, self.region_size) * ovalid.to(anchor)

        # ---------- ADD / REMOVE ----------
        mag_raw_ctx = self.transition_interior_magnitude_out(hidden).view(
            b, k, 1, self.context_size, self.context_size
        )
        interior_mag_ctx = self.max_interior_logit_step * torch.sigmoid(mag_raw_ctx)
        interior_mag = self._center_crop_patch(interior_mag_ctx, self.region_size) * ovalid.to(anchor)

        state_prob = self._center_crop_patch(state_prob_ctx, self.region_size)
        state_st = self._center_crop_patch(state_st_ctx, self.region_size)
        state_logits = self._center_crop_patch(state_logits_ctx, self.region_size)

        interior_patch = (1.0 - boundary_patch) * ovalid.to(anchor)
        soft_delta = (
            state_prob[:, :, self.STATE_MOVE:self.STATE_MOVE+1] * move_delta
            + state_prob[:, :, self.STATE_ADD:self.STATE_ADD+1] * interior_mag * interior_patch
            - state_prob[:, :, self.STATE_REMOVE:self.STATE_REMOVE+1] * interior_mag * interior_patch
        ) * ovalid.to(anchor)
        candidate_delta = (
            state_st[:, :, self.STATE_MOVE:self.STATE_MOVE+1] * move_delta
            + state_st[:, :, self.STATE_ADD:self.STATE_ADD+1] * interior_mag * interior_patch
            - state_st[:, :, self.STATE_REMOVE:self.STATE_REMOVE+1] * interior_mag * interior_patch
        ) * ovalid.to(anchor)

        soft_candidate_logits = anchor_logit_patch + soft_delta
        soft_candidate_probs = torch.sigmoid(soft_candidate_logits).clamp(EPS, 1.0-EPS)
        candidate_logits = anchor_logit_patch + candidate_delta
        candidate_probs = torch.sigmoid(candidate_logits).clamp(EPS, 1.0-EPS)

        # ---------- VALUE / KEEP ----------
        utility_map_ctx = self.transition_utility_out(hidden).view(
            b, k, 1, self.context_size, self.context_size
        )
        utility_map = self._center_crop_patch(utility_map_ctx, self.region_size)
        validf = ovalid.to(anchor)
        utility_logit = (utility_map * validf).sum(dim=(-1,-2)) / validf.sum(dim=(-1,-2)).clamp_min(1.0)
        utility_logit = utility_logit[:, :, :, None, None]  # [B,K,1,1,1]
        utility_value, utility_prob, commit = self._straight_through_commit(utility_logit)
        commit = commit * center_valid[:, :, None, None, None].to(anchor)

        deploy_delta_patch = commit * candidate_delta
        patch_logits = anchor_logit_patch + deploy_delta_patch
        patch_mapped = torch.sigmoid(patch_logits).clamp(EPS, 1.0-EPS)
        patch_probs = torch.where(
            deploy_delta_patch.detach().abs() <= 1.0e-12, anchor_prob_patch, patch_mapped
        )

        full_delta, region, overlap, weight_sum = self._weighted_overlap_add(
            deploy_delta_patch, torch.zeros_like(anchor), oay, oax, ovalid
        )
        full_logits = anchor_logits_full + full_delta
        full_mapped = torch.sigmoid(full_logits).clamp(EPS, 1.0-EPS)
        full_prob = torch.where(full_delta.detach().abs() <= 1.0e-12, anchor, full_mapped)
        full_logits = self._logit(full_prob)

        # Physical action diagnostics use the exact same hard/ST action
        # representation as deployment.  Invalid cross-space states are no-ops.
        move_state = state_st[:, :, self.STATE_MOVE:self.STATE_MOVE+1]
        add_state = state_st[:, :, self.STATE_ADD:self.STATE_ADD+1]
        remove_state = state_st[:, :, self.STATE_REMOVE:self.STATE_REMOVE+1]
        committed_move = commit * move_state * move_px * boundary_patch
        committed_interior = commit * (add_state - remove_state) * interior_mag * interior_patch
        committed_action_support = commit * (
            move_state * boundary_patch + (add_state + remove_state) * interior_patch
        ) * ovalid.to(anchor)
        full_move, _, _, _ = self._weighted_overlap_add(
            committed_move, torch.zeros_like(anchor), oay, oax, ovalid
        )
        full_interior, _, _, _ = self._weighted_overlap_add(
            committed_interior, torch.zeros_like(anchor), oay, oax, ovalid
        )
        full_action_support, _, _, _ = self._weighted_overlap_add(
            committed_action_support.detach(), torch.zeros_like(anchor), oay, oax, ovalid
        )
        full_boundary_mask, _, _, _ = self._weighted_overlap_add(
            (commit * move_state * boundary_patch).detach(), torch.zeros_like(anchor), oay, oax, ovalid
        )

        # Causal geometry diagnostic: the exported SDF changes iff a MOVE action
        # is actually committed.  The same signed displacement drives the warp.
        pred_sdf = anchor_sdf_patch + committed_move
        full_sdf, _, _, _ = self._weighted_overlap_add(pred_sdf, anchor_sdf, oay, oax, ovalid)
        seg_overlap_dis = self._weighted_overlap_disagreement(
            deploy_delta_patch, torch.zeros_like(anchor), oay, oax, ovalid
        )
        sdf_overlap_dis = self._weighted_overlap_disagreement(
            pred_sdf, anchor_sdf, oay, oax, ovalid
        )

        return {
            "center_yx": center_yx,
            "center_valid": center_valid,
            "ay": oay,
            "ax": oax,
            "grid": ogrid,
            "valid_patch": ovalid,
            "patch_logits": patch_logits,
            "patch_probs": patch_probs,
            "raw_patch_logits": candidate_logits,
            "raw_patch_probs": candidate_probs,
            "soft_candidate_logits": soft_candidate_logits,
            "soft_candidate_probs": soft_candidate_probs,
            "patch_action_support": ovalid.to(anchor),
            "patch_seg_delta_logit": candidate_delta,
            "patch_deploy_delta_logit": deploy_delta_patch,
            "patch_boundary_weight": torch.ones_like(deploy_delta_patch),
            "patch_sdf_absolute": pred_sdf,
            "patch_sdf_anchor": anchor_sdf_patch,
            "patch_sdf_correction": move_px,
            "patch_state_logits": state_logits,
            "patch_state_probs": state_prob,
            "patch_move_px": move_px,
            "patch_move_delta_logit": move_delta,
            "patch_interior_magnitude": interior_mag,
            "patch_boundary_mask": boundary_patch,
            "candidate_utility_logit": utility_logit.squeeze(-1).squeeze(-1).squeeze(-1),
            "candidate_utility_value": utility_value.squeeze(-1).squeeze(-1).squeeze(-1),
            "candidate_commit_prob": utility_prob.squeeze(-1).squeeze(-1).squeeze(-1),
            "candidate_commit": commit.squeeze(-1).squeeze(-1).squeeze(-1),
            "full_logits": full_logits,
            "full_prob": full_prob,
            "full_sdf_absolute": full_sdf,
            "full_boundary_displacement_px": full_move,
            "full_boundary_mask": (full_boundary_mask > 0).to(anchor),
            "full_interior_signed_action": full_interior,
            "full_action_support": (full_action_support > 0).to(anchor),
            "region": region,
            "overlap": overlap,
            "blend_weight_sum": weight_sum,
            "overlap_disagreement": seg_overlap_dis,
            "overlap_sdf_disagreement": sdf_overlap_dis,
            "hr_crop": hr_crop,
            "context_size": anchor.new_tensor(float(self.context_size)),
        }

    def _paired_corrupt(self, target4: torch.Tensor) -> torch.Tensor:
        """Paired boundary + topology corruption for stable residual training.

        Each mini-batch contains both missing-foreground and false-positive
        synthetic states whenever the label permits it.  The topology block is
        deliberately wider than 2*boundary_radius so its centre contains genuine
        ADD/REMOVE states rather than being explainable entirely by MOVE.  This is
        training-only; validation/deployment always consume the factual M1 anchor.
        """
        hard = (target4.detach() >= 0.5).float()
        dil = F.max_pool2d(hard, 3, stride=1, padding=1)
        ero = 1.0 - F.max_pool2d(1.0 - hard, 3, stride=1, padding=1)
        b, _, h, w = hard.shape
        out = hard.clone()
        # Ensure an odd topology block with an interior beyond the MOVE radius.
        block = max(3, 2 * int(self.boundary_radius_px) + 3)
        block = min(block, h if h % 2 == 1 else h - 1, w if w % 2 == 1 else w - 1)
        block = max(block, 3)
        half = block // 2

        for bi in range(b):
            fg = hard[bi,0] > 0.5
            # Even samples emphasize FN/ADD; odd samples emphasize FP/REMOVE.
            want_missing = (bi % 2 == 0) and bool(fg.any())
            if want_missing:
                base = ero[bi:bi+1].clone()
                yy, xx = torch.where(fg)
                cy = int(torch.round(yy.float().mean()).item())
                cx = int(torch.round(xx.float().mean()).item())
                y0, y1 = max(0, cy-half), min(h, cy+half+1)
                x0, x1 = max(0, cx-half), min(w, cx+half+1)
                base[:, :, y0:y1, x0:x1] = 0.0
                out[bi:bi+1] = base
            else:
                base = dil[bi:bi+1].clone()
                # Pick the corner block with least GT foreground so the injected
                # component is deployment-relevant FP rather than label overwrite.
                candidates = [
                    (0, 0),
                    (0, max(0, w-block)),
                    (max(0, h-block), 0),
                    (max(0, h-block), max(0, w-block)),
                ]
                masses = [
                    float(hard[bi,0,y:y+block,x:x+block].sum().item())
                    for y,x in candidates
                ]
                y0,x0 = candidates[min(range(len(candidates)), key=lambda i: masses[i])]
                base[:, :, y0:y0+block, x0:x0+block] = 1.0
                out[bi:bi+1] = base
        return out.clamp(EPS, 1.0-EPS)

    def forward(
        self,
        anchor_prob,
        image,
        semantic_map,
        text_features,
        *,
        base_prob=None,
        flow_px=None,
        mc_std_map=None,
        mc_disagreement_map=None,
        fine_feature_map=None,
        posterior_probability_samples=None,
        supervision_masks=None,
        hr_image=None,
    ) -> Dict[str, torch.Tensor]:
        del posterior_probability_samples
        anchor = anchor_prob.detach().clamp(EPS, 1.0-EPS)
        base = anchor if base_prob is None else base_prob.detach().to(anchor).clamp(EPS,1.0-EPS)
        if flow_px is None:
            flow_px = anchor.new_zeros((anchor.shape[0],2,*anchor.shape[-2:]))
        flow = flow_px.detach().to(anchor)
        b, _, h, w = anchor.shape
        hw = (h,w)

        margin = self._margin_uncertainty(anchor)
        mc_std = self._fit_evidence(mc_std_map, anchor, scale=2.0)
        mc_dis = self._fit_evidence(mc_disagreement_map, anchor, scale=1.0)
        entropy = self._entropy(anchor)
        common = self._common_visual(image, semantic_map, text_features, fine_feature_map, hw)
        dense_ev = self._dense_evidence(anchor, base, flow, margin, mc_std, mc_dis, entropy)
        selector_features = torch.cat([common, dense_ev], dim=1)
        selector_logits = self.slr_selector_out(self.slr_selector(selector_features))
        selector_prob = torch.sigmoid(selector_logits)

        proposal_k = self.num_regions + (self.train_positive_regions if self.training else 0)
        proposal_centers, proposal_yx, proposal_valid = self._slr_greedy_centers(
            selector_prob, k=proposal_k
        )
        center_yx = proposal_yx[:, :self.num_regions]
        center_valid = proposal_valid[:, :self.num_regions]
        centers = torch.zeros_like(selector_prob, dtype=torch.bool)
        for bi in range(b):
            vv = center_valid[bi]
            if bool(vv.any()):
                yy = center_yx[bi,vv,0]
                xx = center_yx[bi,vv,1]
                centers[bi,0,yy,xx] = True

        anchor_sdf = self._truncated_signed_distance(
            (anchor >= 0.5).to(anchor), self.sdf_radius_px
        ).detach()
        pred = self._decode_centers(
            center_yx, center_valid, common=common, dense_ev=dense_ev,
            anchor=anchor, anchor_sdf=anchor_sdf, hr_image=hr_image
        )
        final_logits, final_prob, region = pred["full_logits"], pred["full_prob"], pred["region"]
        delta = final_logits - self._logit(anchor)

        zfull = torch.zeros_like(anchor)
        train_selector_target = zfull
        train_sdf_discrepancy = zfull
        positive_score = zfull
        # Dual-space executor can modify every valid pixel inside a selected patch:
        # MOVE owns the boundary band and ADD/REMOVE own its complement.
        action_weight_full = torch.ones_like(anchor)
        action_support_full = torch.ones_like(anchor)
        positive_from_selector_fraction = anchor.new_zeros(())

        pred_target_patch = torch.zeros_like(pred["patch_probs"])
        pred_sdf_target_patch = torch.zeros_like(pred["patch_sdf_absolute"])
        pred_sdf_support_patch = pred["valid_patch"].to(anchor)

        def empty_pack(k):
            q = anchor.new_zeros((b,k,1,self.region_size,self.region_size))
            return q, torch.sigmoid(q), torch.zeros_like(q), torch.zeros_like(q), torch.zeros_like(q), torch.zeros_like(q), torch.zeros_like(q), torch.zeros_like(anchor)

        positive_patch_logits, positive_patch_probs, positive_patch_valid, positive_target_patch, positive_sdf_abs, positive_sdf_target, positive_sdf_support, positive_region = empty_pack(self.train_positive_regions)
        positive_sdf_anchor = torch.zeros_like(positive_sdf_abs)
        clean_patch_logits, clean_patch_probs, clean_patch_valid, clean_target_patch, clean_sdf_abs, clean_sdf_target, clean_sdf_support, clean_region = empty_pack(self.train_clean_regions)
        clean_anchor_patch = torch.zeros_like(clean_patch_logits)
        clean_sdf_anchor = torch.zeros_like(clean_patch_logits)
        pos = None
        clean = None
        paired = None
        paired_anchor = None
        paired_target_patch = None
        paired_sdf_target = None

        if self.training and isinstance(supervision_masks, torch.Tensor):
            target4 = supervision_masks.detach().to(anchor)
            if target4.ndim == 3:
                target4 = target4[:,None]
            if target4.shape[-2:] != hw:
                target4 = F.interpolate(target4.float(), size=hw, mode="nearest").to(anchor)
            target4 = (target4 >= 0.5).to(anchor)
            err4 = ((anchor >= 0.5) != (target4 >= 0.5)).to(anchor)
            gt_sdf = self._truncated_signed_distance(target4, self.sdf_radius_px).detach()
            sdf_discrepancy_pixel = ((anchor_sdf - gt_sdf).abs() / float(self.sdf_radius_px)).clamp(0,1)
            train_sdf_discrepancy = self._patch_error_density(
                sdf_discrepancy_pixel, self.region_size
            ).detach()

            # WHERE is now factual residual density over the *full dual-space action
            # family*, not an SLR boundary-only support proxy.
            train_selector_target = self._patch_error_density(err4, self.region_size).detach()
            positive_score = train_selector_target
            pred_target_patch = self._crop_grid(target4, pred["grid"], self.num_regions).detach()
            pred_sdf_target_patch = self._crop_grid(gt_sdf, pred["grid"], self.num_regions).detach()
            pred_sdf_support_patch = pred["valid_patch"].to(anchor)

            extra_yx = proposal_yx[:, self.num_regions:self.num_regions+self.train_positive_regions].clone()
            extra_valid = proposal_valid[:, self.num_regions:self.num_regions+self.train_positive_regions].clone()
            if extra_yx.shape[1] > 0:
                bi = torch.arange(b, device=anchor.device)[:,None].expand(-1, extra_yx.shape[1])
                candidate_value = train_selector_target[:,0][bi,extra_yx[:,:,0],extra_yx[:,:,1]]
                extra_valid = extra_valid & (candidate_value > 0)
                order = torch.argsort(
                    candidate_value.masked_fill(~extra_valid, -1.0), dim=1, descending=True
                )
                extra_yx = torch.gather(extra_yx,1,order[:,:,None].expand(-1,-1,2))
                extra_valid = torch.gather(extra_valid,1,order)
            pyx, pvalid = extra_yx, extra_valid
            positive_from_selector_fraction = (
                pvalid.float().sum() / pvalid.numel() if pvalid.numel() else anchor.new_zeros(())
            )
            pos = self._decode_centers(
                pyx,pvalid,common=common,dense_ev=dense_ev,anchor=anchor,
                anchor_sdf=anchor_sdf,hr_image=hr_image
            )
            positive_patch_logits = pos["raw_patch_logits"]
            positive_patch_probs = pos["raw_patch_probs"]
            positive_patch_valid = pos["valid_patch"].to(anchor)
            positive_sdf_anchor = pos["patch_sdf_anchor"].detach()
            positive_target_patch = self._crop_grid(target4,pos["grid"],pyx.shape[1]).detach()
            positive_sdf_abs = pos["patch_sdf_absolute"]
            positive_sdf_target = self._crop_grid(gt_sdf,pos["grid"],pyx.shape[1]).detach()
            positive_sdf_support = pos["valid_patch"].to(anchor)
            positive_region = pos["region"].to(anchor)

            boundary_hardness = (1.0 - anchor_sdf.abs()/float(self.sdf_radius_px)).clamp(0,1)
            uncertainty_hardness = (margin + mc_std + mc_dis + entropy) / 4.0
            clean_score = 0.5 * boundary_hardness + 0.5 * uncertainty_hardness
            clean_valid_map = (
                (train_selector_target <= self.clean_max_error_fraction)
                & (train_sdf_discrepancy <= self.clean_max_sdf_discrepancy)
            )
            _, cyx, cvalid = self._slr_greedy_centers(
                clean_score, k=self.train_clean_regions, valid_map=clean_valid_map
            )
            clean = self._decode_centers(
                cyx,cvalid,common=common,dense_ev=dense_ev,anchor=anchor,
                anchor_sdf=anchor_sdf,hr_image=hr_image
            )
            clean_patch_logits = clean["raw_patch_logits"]
            clean_patch_probs = clean["raw_patch_probs"]
            clean_patch_valid = clean["valid_patch"].to(anchor)
            clean_target_patch = self._crop_grid(target4,clean["grid"],cyx.shape[1]).detach()
            clean_anchor_patch = self._crop_grid(anchor,clean["grid"],cyx.shape[1]).detach()
            clean_sdf_abs = clean["patch_sdf_absolute"]
            clean_sdf_anchor = clean["patch_sdf_anchor"].detach()
            clean_sdf_target = self._crop_grid(gt_sdf,clean["grid"],cyx.shape[1]).detach()
            clean_sdf_support = clean["valid_patch"].to(anchor)
            clean_region = clean["region"].to(anchor)

            # Paired stable residual training: the same current image/features and
            # same refiner receive an explicitly paired synthetic coarse state.
            # The synthetic state is *input* to the branch whose target is GT.
            if self.paired_stable_enabled:
                paired_anchor = self._paired_corrupt(target4)
                paired_sdf = self._truncated_signed_distance(
                    (paired_anchor >= 0.5).to(anchor), self.sdf_radius_px
                ).detach()
                p_margin = self._margin_uncertainty(paired_anchor)
                p_entropy = self._entropy(paired_anchor)
                p_dense_ev = self._dense_evidence(
                    paired_anchor, paired_anchor, torch.zeros_like(flow),
                    p_margin, mc_std, mc_dis, p_entropy
                )
                p_selector_features = torch.cat([common, p_dense_ev], dim=1)
                p_selector_prob = torch.sigmoid(
                    self.slr_selector_out(self.slr_selector(p_selector_features))
                )
                # Causal paired mining: GT may rank *selector-proposed* candidates
                # during training, but it never creates a center that deployment
                # could not have proposed.  Oversampling the factual proposal pool
                # prevents the paired branch from degenerating into mostly KEEP
                # patches during the selector warm-up.
                pair_pool_k = max(self.paired_regions * 3, self.paired_regions)
                _, pair_pool_yx, pair_pool_valid = self._slr_greedy_centers(
                    p_selector_prob, k=pair_pool_k
                )
                paired_err = ((paired_anchor >= 0.5) != (target4 >= 0.5)).to(anchor)
                paired_value = self._patch_error_density(paired_err, self.region_size)
                bi = torch.arange(b, device=anchor.device)[:,None].expand(-1,pair_pool_k)
                pvscore = paired_value[:,0][bi,pair_pool_yx[:,:,0],pair_pool_yx[:,:,1]]
                pvscore = pvscore.masked_fill(~pair_pool_valid, -1.0)
                order = torch.argsort(pvscore, dim=1, descending=True)[:, :self.paired_regions]
                pair_yx = torch.gather(pair_pool_yx, 1, order[:,:,None].expand(-1,-1,2))
                pair_valid = torch.gather(pair_pool_valid, 1, order)
                paired = self._decode_centers(
                    pair_yx,pair_valid,common=common,dense_ev=p_dense_ev,
                    anchor=paired_anchor,anchor_sdf=paired_sdf,hr_image=hr_image
                )
                paired_target_patch = self._crop_grid(
                    target4, paired["grid"], self.paired_regions
                ).detach()
                paired_sdf_target = self._crop_grid(
                    gt_sdf, paired["grid"], self.paired_regions
                ).detach()

        action_region = region
        hard_change = ((final_prob >= 0.5) != (anchor >= 0.5)) & action_region
        center_float = centers.to(anchor)
        zero = torch.zeros_like(anchor)
        op = torch.full((b,), -1, dtype=torch.long, device=anchor.device)
        view_probs = final_prob.expand(-1,3,-1,-1)
        view_logits = final_logits.expand(-1,3,-1,-1)

        out: Dict[str, torch.Tensor] = {
            "logits": final_logits,
            "prob": final_prob,
            "selection_mask": region.to(anchor),
            "selection_score": selector_prob,
            "refined_logits": final_logits,
            "refined_prob": final_prob,
            "delta_logit": delta,
            "margin_uncertainty": margin,
            "mc_std_map": mc_std,
            "mc_disagreement_map": mc_dis,
            "entropy_map": entropy,
            "trace": dense_ev,
            "fine_feature_map": common,
            "dn_anchor_prob": anchor,
            "dn_corruption_mask": zero,
            "dn_selection_mask": zero,
            "dn_refined_logits": self._logit(anchor),
            "dn_refined_prob": anchor,
            "dn_final_prob": anchor,
            "dn_delta_logit": zero,
            "dn_op_id": op,
            "r4_flip_logits": zero,
            "r4_flip_prob": zero,
            "r4_flip_mask": zero,
            "r4_synth_flip_logits": zero,
            "r4_synth_flip_prob": zero,
            "r4_synth_flip_mask": zero,
            "r4_synth_target": zero,
            "r2_enabled": anchor.new_zeros((b,)),
            "r3_enabled": anchor.new_zeros((b,)),
            "r4_enabled": anchor.new_zeros((b,)),
            "r41_enabled": anchor.new_zeros((b,)),
            "c2r_center_mask": center_float,
            "c2r_region_mask": region.to(anchor),
            "c2r_view_logits": view_logits,
            "c2r_view_probs": view_probs,
            "c2r_mean_prob": final_prob,
            "c2r_consensus_mask": region.to(anchor),
            "c2r_edit_mask": hard_change.to(anchor),
            "c2r_candidate_mask": hard_change.to(anchor),
            "c2r_commit_mask": hard_change.to(anchor),
            "c2r_roi_overlap_pixel_count": pred["overlap"],
            "c2r_roi_unique_pixel_count": region.float().sum(),
            "c2r_center_min_chebyshev_distance": anchor.new_tensor(float(self.slr_min_center_distance)),
            "aefr_enabled": anchor.new_ones((b,)),
            "aefr_stage_id": anchor.new_full((b,), self.STAGE_ID),
            "aefr_joint_geometry_grad": anchor.new_zeros((b,)),
            "aefr_transition_aware": anchor.new_ones((b,)),
            "aefr_raw_flow_evidence_enabled": anchor.new_ones((b,)),
            "aefr_transition_delta_logit": torch.tanh((self._logit(anchor)-self._logit(base))/2),
            "aefr_transition_abs_mean": (anchor-base).abs().mean(),
            "aefr_transition_active_fraction": ((anchor-base).abs()>1e-4).float().mean(),
            "aefr_transition_flip_fraction": ((anchor>=.5)!=(base>=.5)).float().mean(),
            "aefr_action_delta_logit": delta,
            "aefr_boundary_mask": pred["full_boundary_mask"],
            "aefr_boundary_displacement_px": pred["full_boundary_displacement_px"],
            "aefr_interior_delta_logit": pred["full_interior_signed_action"],
            "aefr_error_localizer_logit": selector_logits,
            "aefr_error_localizer_prob": selector_prob,
            "aefr_error_localizer_prior": zero,
            "aefr_edit_logit": zero,
            "aefr_edit_prob": hard_change.to(anchor),
            "aefr_direction_logit": zero,
            "aefr_direction_prob": zero,
            "aefr_interior_magnitude_logit": zero,
            "aefr_signed_action": delta,
            "aefr_commit_mask": hard_change.to(anchor),
            "aefr_boundary_magnitude_px": pred["full_boundary_displacement_px"].abs(),
            "aefr_signed_boundary_action": pred["full_boundary_displacement_px"],
            "aefr_signed_interior_action": pred["full_interior_signed_action"],
            "aefr_posterior_stability_support": anchor.new_zeros(()),
            "aefr_posterior_stability_improvement": anchor.new_zeros(()),
            "aefr_posterior_disagreement_pre": anchor.new_zeros(()),
            "aefr_posterior_disagreement_post": anchor.new_zeros(()),
            "aefr_action_support_fraction": region.float().mean(),
            "aefr_posterior_diversity_all": anchor.new_zeros(()),
            "aefr_posterior_center_bias_abs": anchor.new_zeros(()),
            "aefr_posterior_center_bias_signed": anchor.new_zeros(()),
            "slr_selector_logits": selector_logits,
            "slr_selector_prob": selector_prob,
            "slr_selector_target": train_selector_target,
            "slr_pred_patch_logits": pred["patch_logits"],
            "slr_pred_patch_probs": pred["patch_probs"],
            "slr_raw_patch_logits": pred["raw_patch_logits"],
            "slr_raw_patch_probs": pred["raw_patch_probs"],
            "slr_pred_patch_grid": pred["grid"],
            "slr_pred_patch_center_valid": pred["center_valid"].to(anchor),
            "slr_pred_patch_valid": pred["valid_patch"].to(anchor),
            "slr_pred_patch_target": pred_target_patch,
            "slr_pred_patch_anchor_prob": self._crop_grid(anchor,pred["grid"],self.num_regions),
            "slr_pred_patch_delta_logit": pred["patch_deploy_delta_logit"],
            "slr_pred_patch_raw_delta_logit": pred["patch_seg_delta_logit"],
            "slr_pred_patch_boundary_weight": pred["patch_boundary_weight"],
            "slr_pred_patch_action_support": pred["patch_action_support"],
            "slr_pred_sdf_absolute": pred["patch_sdf_absolute"],
            "slr_pred_sdf_anchor": pred["patch_sdf_anchor"],
            "slr_pred_sdf_delta": pred["patch_sdf_correction"],
            "slr_pred_sdf_target": pred_sdf_target_patch,
            "slr_pred_sdf_support": pred_sdf_support_patch,
            "slr_sdf_absolute_full": pred["full_sdf_absolute"],
            "slr_sdf_delta_full": pred["full_sdf_absolute"]-anchor_sdf,
            "slr_anchor_sdf_full": anchor_sdf,
            "slr_blend_weight_sum": pred["blend_weight_sum"],
            "slr_overlap_disagreement": pred["overlap_disagreement"],
            "slr_overlap_sdf_disagreement": pred["overlap_sdf_disagreement"],
            "slr_boundary_weight_full": torch.ones_like(anchor),
            "slr_sdf_discrepancy_map": train_sdf_discrepancy,
            "slr_positive_training_score": positive_score,
            "slr_sdf_support_full": action_support_full,
            "slr_action_weight_full": action_weight_full,
            "slr_action_support_full": action_support_full,
            "slr_action_region_mask": action_region.to(anchor),
            "slr_positive_from_selector_fraction": positive_from_selector_fraction,
            "slr_true_hr_active": anchor.new_tensor(float(hr_image is not None)),
            "slr_context_size": anchor.new_tensor(float(self.context_size)),
            "slr_context_halo": anchor.new_tensor(float(self.context_halo)),
            "slr_positive_patch_logits": positive_patch_logits,
            "slr_positive_patch_probs": positive_patch_probs,
            "slr_positive_patch_valid": positive_patch_valid,
            "slr_positive_patch_target": positive_target_patch,
            "slr_positive_sdf_absolute": positive_sdf_abs,
            "slr_positive_sdf_target": positive_sdf_target,
            "slr_positive_sdf_support": positive_sdf_support,
            "slr_positive_region_mask": positive_region,
            "slr_positive_sdf_anchor": positive_sdf_anchor,
            "slr_positive_sdf_delta": positive_sdf_abs-positive_sdf_anchor,
            "slr_clean_patch_logits": clean_patch_logits,
            "slr_clean_patch_probs": clean_patch_probs,
            "slr_clean_patch_valid": clean_patch_valid,
            "slr_clean_patch_target": clean_target_patch,
            "slr_clean_patch_anchor_prob": clean_anchor_patch,
            "slr_clean_patch_delta_logit": clean_patch_logits-self._logit(clean_anchor_patch.clamp(EPS,1-EPS)),
            "slr_clean_sdf_absolute": clean_sdf_abs,
            "slr_clean_sdf_anchor": clean_sdf_anchor,
            "slr_clean_sdf_delta": clean_sdf_abs-clean_sdf_anchor,
            "slr_clean_sdf_target": clean_sdf_target,
            "slr_clean_sdf_support": clean_sdf_support,
            "slr_clean_region_mask": clean_region,
            "slr_oracle_patch_logits": positive_patch_logits,
            "slr_oracle_patch_probs": positive_patch_probs,
            "slr_oracle_patch_valid": positive_patch_valid,
            "slr_oracle_patch_target": positive_target_patch,
            "slr_oracle_sdf_delta": positive_sdf_abs-positive_sdf_anchor,
            "slr_oracle_sdf_target": positive_sdf_target,
            "slr_oracle_sdf_support": positive_sdf_support,
            "slr_oracle_region_mask": positive_region,
            "slr_patch_window": self.slr_patch_window.to(anchor),
            # UCDRT-native outputs.
            "slr_ucdrt_enabled": anchor.new_ones(()),
            "slr_ucdrt_state_logits": pred["patch_state_logits"],
            "slr_ucdrt_state_probs": pred["patch_state_probs"],
            "slr_ucdrt_move_px": pred["patch_move_px"],
            "slr_ucdrt_interior_magnitude": pred["patch_interior_magnitude"],
            "slr_ucdrt_boundary_mask": pred["patch_boundary_mask"],
            "slr_ucdrt_candidate_soft_probs": pred["soft_candidate_probs"],
            "slr_ucdrt_utility_logit": pred["candidate_utility_logit"],
            "slr_ucdrt_utility_value": pred["candidate_utility_value"],
            "slr_ucdrt_commit_prob": pred["candidate_commit_prob"],
            "slr_ucdrt_commit": pred["candidate_commit"],
            "slr_ucdrt_boundary_displacement_full": pred["full_boundary_displacement_px"],
            "slr_ucdrt_interior_action_full": pred["full_interior_signed_action"],
            "slr_ucdrt_action_mask_full": pred["full_action_support"],
        }
        # UCDRT-R2 is implemented as a subclass and reuses this factual/positive/
        # paired proposal pipeline.  Expose its native actor/critic tensors only
        # when the subclass decoder provides them; R1 behavior remains unchanged.
        if "patch_edit_logits" in pred:
            out.update({
                "slr_ucdrt_r2_enabled": anchor.new_ones(()),
                "slr_ucdrt_r2_edit_logits": pred["patch_edit_logits"],
                "slr_ucdrt_r2_edit_probs": pred["patch_edit_probs"],
                "slr_ucdrt_r2_type_logits": pred["patch_type_logits"],
                "slr_ucdrt_r2_type_probs": pred["patch_type_probs"],
                "slr_ucdrt_r2_move_bin_logits": pred["patch_move_bin_logits"],
                "slr_ucdrt_r2_move_bin_probs": pred["patch_move_bin_probs"],
                "slr_ucdrt_r2_move_dictionary_probs": pred["patch_move_dictionary_probs"],
                "slr_ucdrt_r2_move_bins_px": pred["patch_move_bins_px"],
                "slr_ucdrt_r2_add_dose": pred["patch_add_dose"],
                "slr_ucdrt_r2_remove_dose": pred["patch_remove_dose"],
                "slr_ucdrt_r2_hard_candidate_probs": pred["candidate_hard_probs"],
                "slr_ucdrt_r2_soft_candidate_probs": pred["candidate_soft_probs_r2"],
                "slr_ucdrt_r2_critic_step_logits": pred["critic_step_logits"],
                "slr_ucdrt_r2_critic_step_values": pred["critic_step_values"],
                "slr_ucdrt_r2_critic_step_available": pred["critic_step_available"].to(anchor),
                "slr_ucdrt_r2_selection_step_index": pred["selection_step_index"],
                "slr_ucdrt_r2_candidate_full_num": pred["candidate_full_num"],
                "slr_ucdrt_r2_candidate_full_den": pred["candidate_full_den"],
                "slr_ucdrt_r2_selected_candidate_mask": pred["selected_candidate_mask"],
            })

        if pos is not None:
            out.update({
                "slr_ucdrt_positive_state_logits": pos["patch_state_logits"],
                "slr_ucdrt_positive_move_px": pos["patch_move_px"],
                "slr_ucdrt_positive_interior_magnitude": pos["patch_interior_magnitude"],
                "slr_ucdrt_positive_boundary_mask": pos["patch_boundary_mask"],
                "slr_ucdrt_positive_soft_probs": pos["soft_candidate_probs"],
                "slr_ucdrt_positive_utility_value": pos["candidate_utility_value"],
                "slr_ucdrt_positive_anchor_prob": self._crop_grid(anchor,pos["grid"],pos["center_yx"].shape[1]),
                "slr_ucdrt_positive_anchor_sdf": pos["patch_sdf_anchor"],
            })
            if "patch_edit_logits" in pos:
                out.update({
                    "slr_ucdrt_r2_positive_edit_logits": pos["patch_edit_logits"],
                    "slr_ucdrt_r2_positive_type_logits": pos["patch_type_logits"],
                    "slr_ucdrt_r2_positive_move_bin_logits": pos["patch_move_bin_logits"],
                    "slr_ucdrt_r2_positive_move_dictionary_probs": pos["patch_move_dictionary_probs"],
                    "slr_ucdrt_r2_positive_add_dose": pos["patch_add_dose"],
                    "slr_ucdrt_r2_positive_remove_dose": pos["patch_remove_dose"],
                    "slr_ucdrt_r2_positive_hard_candidate_probs": pos["candidate_hard_probs"],
                })
        if clean is not None:
            out.update({
                "slr_ucdrt_clean_state_logits": clean["patch_state_logits"],
                "slr_ucdrt_clean_move_px": clean["patch_move_px"],
                "slr_ucdrt_clean_interior_magnitude": clean["patch_interior_magnitude"],
                "slr_ucdrt_clean_boundary_mask": clean["patch_boundary_mask"],
                "slr_ucdrt_clean_soft_probs": clean["soft_candidate_probs"],
                "slr_ucdrt_clean_utility_value": clean["candidate_utility_value"],
                "slr_ucdrt_clean_anchor_sdf": clean["patch_sdf_anchor"],
            })
        if paired is not None and paired_anchor is not None:
            out.update({
                "slr_ucdrt_paired_state_logits": paired["patch_state_logits"],
                "slr_ucdrt_paired_move_px": paired["patch_move_px"],
                "slr_ucdrt_paired_interior_magnitude": paired["patch_interior_magnitude"],
                "slr_ucdrt_paired_boundary_mask": paired["patch_boundary_mask"],
                "slr_ucdrt_paired_soft_probs": paired["soft_candidate_probs"],
                "slr_ucdrt_paired_valid": paired["valid_patch"].to(anchor),
                "slr_ucdrt_paired_target": paired_target_patch,
                "slr_ucdrt_paired_anchor_prob": self._crop_grid(paired_anchor,paired["grid"],self.paired_regions),
                "slr_ucdrt_paired_anchor_sdf": paired["patch_sdf_anchor"],
                "slr_ucdrt_paired_target_sdf": paired_sdf_target,
            })
            if "patch_edit_logits" in paired:
                out.update({
                    "slr_ucdrt_r2_paired_edit_logits": paired["patch_edit_logits"],
                    "slr_ucdrt_r2_paired_type_logits": paired["patch_type_logits"],
                    "slr_ucdrt_r2_paired_move_bin_logits": paired["patch_move_bin_logits"],
                    "slr_ucdrt_r2_paired_move_dictionary_probs": paired["patch_move_dictionary_probs"],
                    "slr_ucdrt_r2_paired_add_dose": paired["patch_add_dose"],
                    "slr_ucdrt_r2_paired_remove_dose": paired["patch_remove_dose"],
                    "slr_ucdrt_r2_paired_hard_candidate_probs": paired["candidate_hard_probs"],
                    "slr_ucdrt_r2_paired_soft_candidate_probs": paired["candidate_soft_probs_r2"],
                })
        return out
