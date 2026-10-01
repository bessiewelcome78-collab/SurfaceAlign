"""UCDRT-R2: operator-consistent selective residual transition refinement.

R2 keeps the validated SLR proposal/true-HR/WOLA infrastructure but fixes five
contracts exposed by the first UCDRT diagnostic:

1) EDIT and TYPE are factorized.  EDIT is calibrated on the factual sparse
   residual prior; MOVE/ADD/REMOVE are conditional types.
2) ADD/REMOVE are probability-space monotone doses, therefore every binary FN/FP
   is representationally crossable with a finite dose in [0,1].
3) MOVE is a discrete dictionary of the *actual SDF-normal warp operator*.  The
   loss chooses targets by evaluating this dictionary, not by treating an SDF
   difference as a displacement label.
4) The actor exposes ungated hard/soft executable candidates.  The critic is
   action-conditioned and uses a detached actor descriptor, so critic learning
   cannot rewrite the actor it is evaluating.
5) Deployment is a sequential STOP-vs-candidate policy.  Selected candidate
   deltas are accumulated with the exact same WOLA numerator/denominator as the
   final compositor.  STOP has fixed value zero; no hand-written safety threshold
   is used.

GT is never used to create inference proposals/actions.  It is used only by
training losses and diagnostics after selector-proposed candidates exist.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .c2r_canonical_roi_refiner import EPS
from .slr_transition_refiner import UtilityConsistentDualSpaceResidualTransitionRefiner


class OperatorConsistentSelectiveTransitionRefiner(
    UtilityConsistentDualSpaceResidualTransitionRefiner
):
    """UCDRT-R2 executable actor + action-conditioned marginal critic."""

    STAGE_ID = 11.0
    TYPE_MOVE = 0
    TYPE_ADD = 1
    TYPE_REMOVE = 2
    NUM_TYPES = 3

    def __init__(
        self,
        hidden_dim: int,
        semantic_channels: int,
        text_dim: int,
        *,
        move_bins: int = 21,
        edit_prior: float = 0.15,
        dose_init_fraction: float = 0.10,
        utility_gain_scale: float = 20.0,
        utility_boundary_weight: float = 0.25,
        **kwargs,
    ) -> None:
        # Build the R1 trunk/infrastructure, then retire the R1 action/value heads.
        super().__init__(
            hidden_dim,
            semantic_channels,
            text_dim,
            interior_magnitude_init_fraction=dose_init_fraction,
            **kwargs,
        )
        for name in (
            "transition_state_out",
            "transition_move_out",
            "transition_interior_magnitude_out",
            "transition_utility_out",
        ):
            if hasattr(self, name):
                delattr(self, name)

        self.r2_hidden_dim = int(hidden_dim)
        self.move_bin_count = max(int(move_bins), 3)
        if self.move_bin_count % 2 == 0:
            self.move_bin_count += 1
        self.utility_gain_scale = max(float(utility_gain_scale), 1.0)
        self.utility_boundary_weight = min(max(float(utility_boundary_weight), 0.0), 0.75)

        # Factorized sparse-state actor.
        self.transition_edit_out = nn.Conv2d(hidden_dim, 1, 1)
        nn.init.zeros_(self.transition_edit_out.weight)
        p0 = min(max(float(edit_prior), 1.0e-3), 1.0 - 1.0e-3)
        nn.init.constant_(self.transition_edit_out.bias, float(torch.logit(torch.tensor(p0)).item()))

        self.transition_type_out = nn.Conv2d(hidden_dim, self.NUM_TYPES, 1)
        nn.init.zeros_(self.transition_type_out.weight)
        nn.init.zeros_(self.transition_type_out.bias)

        self.transition_move_bin_out = nn.Conv2d(hidden_dim, self.move_bin_count, 1)
        nn.init.zeros_(self.transition_move_bin_out.weight)
        nn.init.zeros_(self.transition_move_bin_out.bias)
        with torch.no_grad():
            self.transition_move_bin_out.bias[self.move_bin_count // 2] = 1.0

        self.transition_add_dose_out = nn.Conv2d(hidden_dim, 1, 1)
        self.transition_remove_dose_out = nn.Conv2d(hidden_dim, 1, 1)
        for head in (self.transition_add_dose_out, self.transition_remove_dose_out):
            nn.init.zeros_(head.weight)
            f = min(max(float(dose_init_fraction), 1.0e-4), 1.0 - 1.0e-4)
            nn.init.constant_(head.bias, float(torch.logit(torch.tensor(f)).item()))

        bins = torch.linspace(
            -float(self.max_boundary_displacement_px),
            float(self.max_boundary_displacement_px),
            self.move_bin_count,
        )
        self.register_buffer("transition_move_bins_px", bins, persistent=True)

        # Candidate descriptor: edit(1)+type(3)+move(1)+add/remove(2)+
        # signed/abs candidate change(2)+boundary fraction(1) = 10.
        # Current-set descriptor: signed/abs/max current WOLA delta + coverage = 4.
        critic_in = hidden_dim + 14
        critic_hidden = max(hidden_dim, 64)
        self.transition_critic = nn.Sequential(
            nn.Linear(critic_in, critic_hidden),
            nn.GELU(),
            nn.Linear(critic_hidden, critic_hidden),
            nn.GELU(),
            nn.Linear(critic_hidden, 1),
        )
        nn.init.zeros_(self.transition_critic[-1].weight)
        nn.init.zeros_(self.transition_critic[-1].bias)

    @staticmethod
    def _st_binary(prob: torch.Tensor) -> torch.Tensor:
        hard = (prob.detach() >= 0.5).to(prob)
        return hard + prob - prob.detach() if prob.requires_grad else hard

    @staticmethod
    def _st_categorical(prob: torch.Tensor, dim: int) -> torch.Tensor:
        idx = prob.detach().argmax(dim=dim, keepdim=True)
        hard = torch.zeros_like(prob).scatter_(dim, idx, 1.0)
        return hard + prob - prob.detach() if prob.requires_grad else hard

    @staticmethod
    def _masked_mean_patch(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """[B,K,C,R,R] -> [B,K,C] with [B,K,1,R,R] validity."""
        v = valid.to(x)
        return (x * v).sum(dim=(-1, -2)) / v.sum(dim=(-1, -2)).clamp_min(1.0)

    def _scatter_candidate_weighted(
        self,
        patch_delta: torch.Tensor,
        ay: torch.Tensor,
        ax: torch.Tensor,
        valid: torch.Tensor,
        h: int,
        w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-candidate WOLA numerator/denominator full maps.

        Returns [B,K,1,H,W] tensors.  Their sum over any selected subset followed
        by numerator/denominator division is exactly the inherited WOLA compositor.
        """
        b, k, _, r, _ = patch_delta.shape
        idx = (ay * w + ax).reshape(b * k, -1)
        win = self.slr_patch_window.to(patch_delta).expand(b, k, 1, r, r)
        wt = (win * valid.to(win)).reshape(b * k, -1)
        val = patch_delta.reshape(b * k, -1)
        num = patch_delta.new_zeros((b * k, h * w))
        den = patch_delta.new_zeros((b * k, h * w))
        num.scatter_add_(1, idx, val * wt)
        den.scatter_add_(1, idx, wt)
        return (
            num.view(b, k, 1, h, w),
            den.view(b, k, 1, h, w),
        )

    def _critic_scores(
        self,
        candidate_feat: torch.Tensor,
        action_desc: torch.Tensor,
        current_delta: torch.Tensor,
        current_den: torch.Tensor,
        grid: torch.Tensor,
        valid: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Critic may observe the current selected-set state, but its input is
        # detached so critic losses do not rewrite the actor being evaluated.
        cur_patch = self._crop_grid(current_delta.detach(), grid, k)
        cov_patch = self._crop_grid((current_den.detach() > 1.0e-8).to(current_delta), grid, k)
        cur_mean = self._masked_mean_patch(cur_patch, valid)
        cur_abs = self._masked_mean_patch(cur_patch.abs(), valid)
        masked_abs = cur_patch.abs() * valid.to(cur_patch)
        cur_max = masked_abs.flatten(3).amax(dim=-1)
        cur_cov = self._masked_mean_patch(cov_patch, valid)
        current_desc = torch.cat([cur_mean, cur_abs, cur_max, cur_cov], dim=2)
        critic_in = torch.cat(
            [candidate_feat.detach(), action_desc.detach(), current_desc.detach()], dim=2
        )
        raw = self.transition_critic(critic_in).squeeze(-1)
        return raw, torch.tanh(raw)

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
                raise RuntimeError("GEOTR-UCDRT-R2 requires true paired HR image")
            hr_image = F.interpolate(common[:, :3], scale_factor=2, mode="bilinear", align_corners=False)
        if hr_image.ndim != 4 or hr_image.shape[0] != b or hr_image.shape[1] != 3:
            raise ValueError("UCDRT-R2 hr_image must be [B,3,Hh,Wh]")
        if hr_image.shape[-2] != h * 2 or hr_image.shape[-1] != w * 2:
            raise ValueError(
                f"UCDRT-R2 true-HR contract requires exactly 2x LR, got LR={(h,w)} "
                f"HR={tuple(hr_image.shape[-2:])}"
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
        hr_expand = hr_image[:, None].expand(-1, k, -1, -1, -1).reshape(
            b * k, 3, *hr_image.shape[-2:]
        )
        hr_crop = F.grid_sample(
            hr_expand, hr_grid, mode="bilinear", padding_mode="border", align_corners=False
        )
        hr_feat = self.hr_encoder(hr_crop).view(
            b, k, -1, self.context_size, self.context_size
        )
        x = torch.cat([common_c, ev_c, asdf_c, hr_feat], dim=2)
        hidden_bk = self.region_decoder(
            x.reshape(b * k, x.shape[2], self.context_size, self.context_size)
        )
        hidden = hidden_bk.view(
            b, k, self.r2_hidden_dim, self.context_size, self.context_size
        )
        hidden_patch = self._center_crop_patch(hidden, self.region_size)

        anchor_logits_full = self._logit(anchor)
        anchor_prob_patch = self._crop_grid(anchor, ogrid, k)
        anchor_logit_patch = self._logit(anchor_prob_patch)
        anchor_sdf_patch = self._crop_grid(anchor_sdf, ogrid, k)
        validf = ovalid.to(anchor)
        boundary_patch = (
            anchor_sdf_patch.detach().abs() <= float(self.boundary_radius_px)
        ).to(anchor) * validf

        # ---------------- factorized EDIT / TYPE ----------------
        edit_logits_ctx = self.transition_edit_out(hidden_bk).view(
            b, k, 1, self.context_size, self.context_size
        )
        type_logits_ctx = self.transition_type_out(hidden_bk).view(
            b, k, self.NUM_TYPES, self.context_size, self.context_size
        )
        edit_logits = self._center_crop_patch(edit_logits_ctx, self.region_size)
        type_logits = self._center_crop_patch(type_logits_ctx, self.region_size)
        edit_prob = torch.sigmoid(edit_logits)
        type_prob = torch.softmax(type_logits, dim=2)
        edit_st = self._st_binary(edit_prob)
        type_st = self._st_categorical(type_prob, dim=2)

        # ---------------- MOVE dictionary: exact warp operator ----------------
        move_bin_logits_ctx = self.transition_move_bin_out(hidden_bk).view(
            b, k, self.move_bin_count, self.context_size, self.context_size
        )
        move_bin_logits = self._center_crop_patch(move_bin_logits_ctx, self.region_size)
        move_bin_prob = torch.softmax(move_bin_logits, dim=2)
        move_bin_st = self._st_categorical(move_bin_prob, dim=2)

        asdf_bk = asdf_ctx.reshape(b * k, 1, self.context_size, self.context_size)
        gx, gy = self._gradient_xy(asdf_bk)
        norm = torch.sqrt(gx.square() + gy.square()).clamp_min(1.0e-6)
        nx, ny = gx / norm, gy / norm
        anchor_logit_ctx = self._crop_grid(anchor_logits_full, cgrid, k).reshape(
            b * k, 1, self.context_size, self.context_size
        )
        d = self.move_bin_count
        bins = self.transition_move_bins_px.to(anchor_logit_ctx)
        nx_d = nx[:, None].expand(-1, d, -1, -1, -1).reshape(
            b * k * d, 1, self.context_size, self.context_size
        )
        ny_d = ny[:, None].expand(-1, d, -1, -1, -1).reshape_as(nx_d)
        disp = bins.view(1, d, 1, 1, 1).expand(
            b * k, -1, 1, self.context_size, self.context_size
        ).reshape(b * k * d, 1, self.context_size, self.context_size)
        flow = torch.cat([nx_d * disp, ny_d * disp], dim=1)
        src = anchor_logit_ctx[:, None].expand(
            -1, d, -1, -1, -1
        ).reshape(b * k * d, 1, self.context_size, self.context_size)
        warped = self._warp_patch_logits_px(src, flow).view(
            b, k, d, 1, self.context_size, self.context_size
        )
        move_dictionary_logits = self._center_crop_patch(warped, self.region_size)
        move_dictionary_probs = torch.sigmoid(move_dictionary_logits).clamp(EPS, 1.0 - EPS)
        move_hard_logits = (move_bin_st[:, :, :, None] * move_dictionary_logits).sum(dim=2)
        move_soft_logits = (move_bin_prob[:, :, :, None] * move_dictionary_logits).sum(dim=2)
        move_hard_prob = torch.sigmoid(move_hard_logits).clamp(EPS, 1.0 - EPS)
        move_soft_prob = torch.sigmoid(move_soft_logits).clamp(EPS, 1.0 - EPS)
        move_px = (
            move_bin_st * bins.view(1, 1, d, 1, 1)
        ).sum(dim=2, keepdim=True) * validf
        move_expected_px = (
            move_bin_prob * bins.view(1, 1, d, 1, 1)
        ).sum(dim=2, keepdim=True) * validf

        # ---------------- ADD / REMOVE: finite probability doses ----------------
        add_raw_ctx = self.transition_add_dose_out(hidden_bk).view(
            b, k, 1, self.context_size, self.context_size
        )
        rem_raw_ctx = self.transition_remove_dose_out(hidden_bk).view(
            b, k, 1, self.context_size, self.context_size
        )
        add_dose = torch.sigmoid(self._center_crop_patch(add_raw_ctx, self.region_size)) * validf
        remove_dose = torch.sigmoid(self._center_crop_patch(rem_raw_ctx, self.region_size)) * validf
        add_prob = anchor_prob_patch + add_dose * (1.0 - anchor_prob_patch)
        remove_prob = anchor_prob_patch * (1.0 - remove_dose)
        add_prob = add_prob.clamp(EPS, 1.0 - EPS)
        remove_prob = remove_prob.clamp(EPS, 1.0 - EPS)

        # MOVE is geometrically owned by the anchor boundary band.  If TYPE=MOVE
        # elsewhere it becomes identity; executable targets therefore never assign
        # such an invalid state.
        move_hard_prob = torch.where(boundary_patch > 0, move_hard_prob, anchor_prob_patch)
        move_soft_prob = torch.where(boundary_patch > 0, move_soft_prob, anchor_prob_patch)

        hard_operator_prob = (
            type_st[:, :, self.TYPE_MOVE:self.TYPE_MOVE + 1] * move_hard_prob
            + type_st[:, :, self.TYPE_ADD:self.TYPE_ADD + 1] * add_prob
            + type_st[:, :, self.TYPE_REMOVE:self.TYPE_REMOVE + 1] * remove_prob
        )
        soft_operator_prob = (
            type_prob[:, :, self.TYPE_MOVE:self.TYPE_MOVE + 1] * move_soft_prob
            + type_prob[:, :, self.TYPE_ADD:self.TYPE_ADD + 1] * add_prob
            + type_prob[:, :, self.TYPE_REMOVE:self.TYPE_REMOVE + 1] * remove_prob
        )
        candidate_probs = (
            (1.0 - edit_st) * anchor_prob_patch + edit_st * hard_operator_prob
        ).clamp(EPS, 1.0 - EPS)
        soft_candidate_probs = (
            (1.0 - edit_prob) * anchor_prob_patch + edit_prob * soft_operator_prob
        ).clamp(EPS, 1.0 - EPS)
        candidate_probs = torch.where(validf > 0, candidate_probs, anchor_prob_patch)
        soft_candidate_probs = torch.where(validf > 0, soft_candidate_probs, anchor_prob_patch)
        candidate_logits = self._logit(candidate_probs)
        soft_candidate_logits = self._logit(soft_candidate_probs)
        candidate_delta = (candidate_logits - anchor_logit_patch) * validf
        soft_delta = (soft_candidate_logits - anchor_logit_patch) * validf

        # R1-compatible four-state probabilities are diagnostics only in R2.
        state_prob = torch.cat(
            [
                1.0 - edit_prob,
                edit_prob * type_prob[:, :, self.TYPE_MOVE:self.TYPE_MOVE + 1],
                edit_prob * type_prob[:, :, self.TYPE_ADD:self.TYPE_ADD + 1],
                edit_prob * type_prob[:, :, self.TYPE_REMOVE:self.TYPE_REMOVE + 1],
            ],
            dim=2,
        ).clamp_min(1.0e-8)
        state_logits = state_prob.log()

        # ---------------- action-conditioned marginal critic ----------------
        candidate_feat = self._masked_mean_patch(hidden_patch, validf)
        change = candidate_probs - anchor_prob_patch
        action_desc = torch.cat(
            [
                self._masked_mean_patch(edit_prob, validf),
                self._masked_mean_patch(type_prob, validf),
                self._masked_mean_patch(
                    move_expected_px / max(float(self.max_boundary_displacement_px), 1.0e-6), validf
                ),
                self._masked_mean_patch(add_dose, validf),
                self._masked_mean_patch(remove_dose, validf),
                self._masked_mean_patch(change, validf),
                self._masked_mean_patch(change.abs(), validf),
                self._masked_mean_patch(boundary_patch, validf),
            ],
            dim=2,
        )
        cand_num, cand_den = self._scatter_candidate_weighted(
            candidate_delta, oay, oax, ovalid, h, w
        )
        accum_num = anchor.new_zeros((b, 1, h, w))
        accum_den = anchor.new_zeros((b, 1, h, w))
        selected = torch.zeros((b, k), dtype=torch.bool, device=anchor.device)
        active = center_valid.any(dim=1)
        step_logits, step_values, step_available, step_selected = [], [], [], []

        for _step in range(k):
            current_delta = torch.where(
                accum_den > 1.0e-8,
                accum_num / accum_den.clamp_min(1.0e-8),
                torch.zeros_like(accum_num),
            )
            raw_score, value_score = self._critic_scores(
                candidate_feat, action_desc, current_delta, accum_den,
                ogrid, validf, k,
            )
            available = center_valid & (~selected) & active[:, None]
            masked_value = value_score.masked_fill(~available, -1.0e4)
            best_value, best_idx = masked_value.max(dim=1)
            execute = active & available.any(dim=1) & (best_value.detach() > 0.0)
            onehot = F.one_hot(best_idx, num_classes=k).to(anchor) * execute[:, None].to(anchor)
            step_logits.append(raw_score)
            step_values.append(value_score)
            step_available.append(available)
            step_selected.append(torch.where(execute, best_idx, torch.full_like(best_idx, -1)))

            selected = selected | (onehot > 0.5)
            accum_num = accum_num + (cand_num * onehot[:, :, None, None, None]).sum(dim=1)
            accum_den = accum_den + (cand_den * onehot[:, :, None, None, None]).sum(dim=1)
            active = active & execute

        critic_step_logits = torch.stack(step_logits, dim=1)
        critic_step_values = torch.stack(step_values, dim=1)
        critic_step_available = torch.stack(step_available, dim=1)
        selection_step_index = torch.stack(step_selected, dim=1)
        commit = selected.to(anchor)

        full_delta = torch.where(
            accum_den > 1.0e-8,
            accum_num / accum_den.clamp_min(1.0e-8),
            torch.zeros_like(accum_num),
        )
        full_logits = anchor_logits_full + full_delta
        full_mapped = torch.sigmoid(full_logits).clamp(EPS, 1.0 - EPS)
        full_prob = torch.where(full_delta.detach().abs() <= 1.0e-12, anchor, full_mapped)
        full_logits = self._logit(full_prob)
        # Proposal ROI remains observable even when the learned policy selects STOP.
        # This preserves WHERE supervision/diagnostics and lets factual residuals
        # train the final deploy objective without conflating proposal with commit.
        _zpatch = torch.zeros_like(candidate_delta)
        _, region, proposal_overlap, proposal_weight_sum = self._weighted_overlap_add(
            _zpatch, torch.zeros_like(anchor), oay, oax, ovalid
        )

        commit5 = commit[:, :, None, None, None]
        deploy_delta_patch = commit5 * candidate_delta
        patch_logits = anchor_logit_patch + deploy_delta_patch
        patch_probs = torch.where(
            commit5 > 0,
            candidate_probs,
            anchor_prob_patch,
        )

        move_type_st = type_st[:, :, self.TYPE_MOVE:self.TYPE_MOVE + 1]
        add_type_st = type_st[:, :, self.TYPE_ADD:self.TYPE_ADD + 1]
        rem_type_st = type_st[:, :, self.TYPE_REMOVE:self.TYPE_REMOVE + 1]
        committed_move = commit5 * edit_st * move_type_st * move_px * boundary_patch
        committed_interior = commit5 * edit_st * (
            add_type_st * add_dose - rem_type_st * remove_dose
        ) * validf
        full_move, _, _, _ = self._weighted_overlap_add(
            committed_move, torch.zeros_like(anchor), oay, oax, ovalid
        )
        full_interior, _, _, _ = self._weighted_overlap_add(
            committed_interior, torch.zeros_like(anchor), oay, oax, ovalid
        )
        committed_action_support = commit5 * edit_st * validf
        full_action_support, _, _, _ = self._weighted_overlap_add(
            committed_action_support.detach(), torch.zeros_like(anchor), oay, oax, ovalid
        )
        full_boundary_mask, _, _, _ = self._weighted_overlap_add(
            (commit5 * edit_st * move_type_st * boundary_patch).detach(),
            torch.zeros_like(anchor), oay, oax, ovalid,
        )

        pred_sdf = anchor_sdf_patch + committed_move
        full_sdf, _, _, _ = self._weighted_overlap_add(
            pred_sdf, anchor_sdf, oay, oax, ovalid
        )
        seg_overlap_dis = self._weighted_overlap_disagreement(
            deploy_delta_patch, torch.zeros_like(anchor), oay, oax, ovalid
        )
        sdf_overlap_dis = self._weighted_overlap_disagreement(
            pred_sdf, anchor_sdf, oay, oax, ovalid
        )

        first_raw = critic_step_logits[:, 0] if k > 0 else anchor.new_zeros((b, 0))
        first_value = critic_step_values[:, 0] if k > 0 else anchor.new_zeros((b, 0))
        first_prob = torch.sigmoid(first_raw / self.utility_temperature)

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
            "patch_action_support": validf,
            "patch_seg_delta_logit": candidate_delta,
            "patch_deploy_delta_logit": deploy_delta_patch,
            "patch_boundary_weight": torch.ones_like(deploy_delta_patch),
            "patch_sdf_absolute": pred_sdf,
            "patch_sdf_anchor": anchor_sdf_patch,
            "patch_sdf_correction": move_px * boundary_patch,
            # compatibility aliases
            "patch_state_logits": state_logits,
            "patch_state_probs": state_prob,
            "patch_move_px": move_px,
            "patch_interior_magnitude": torch.maximum(add_dose, remove_dose),
            "patch_boundary_mask": boundary_patch,
            "candidate_utility_logit": first_raw,
            "candidate_utility_value": first_value,
            "candidate_commit_prob": first_prob,
            "candidate_commit": commit,
            # R2 actor
            "patch_edit_logits": edit_logits,
            "patch_edit_probs": edit_prob,
            "patch_type_logits": type_logits,
            "patch_type_probs": type_prob,
            "patch_move_bin_logits": move_bin_logits,
            "patch_move_bin_probs": move_bin_prob,
            "patch_move_dictionary_probs": move_dictionary_probs,
            "patch_move_bins_px": bins,
            "patch_add_dose": add_dose,
            "patch_remove_dose": remove_dose,
            "candidate_hard_probs": candidate_probs,
            "candidate_soft_probs_r2": soft_candidate_probs,
            # R2 critic / exact set compositor
            "critic_step_logits": critic_step_logits,
            "critic_step_values": critic_step_values,
            "critic_step_available": critic_step_available,
            "selection_step_index": selection_step_index,
            "candidate_full_num": cand_num,
            "candidate_full_den": cand_den,
            "selected_candidate_mask": commit,
            "full_logits": full_logits,
            "full_prob": full_prob,
            "full_sdf_absolute": full_sdf,
            "full_boundary_displacement_px": full_move,
            "full_boundary_mask": (full_boundary_mask > 0).to(anchor),
            "full_interior_signed_action": full_interior,
            "full_action_support": (full_action_support > 0).to(anchor),
            "region": region,
            "overlap": proposal_overlap,
            "blend_weight_sum": proposal_weight_sum,
            "overlap_disagreement": seg_overlap_dis,
            "overlap_sdf_disagreement": sdf_overlap_dis,
            "hr_crop": hr_crop,
            "context_size": anchor.new_tensor(float(self.context_size)),
        }

    def _paired_corrupt(self, target4: torch.Tensor) -> torch.Tensor:
        """Operator-closed paired corruption.

        Start from a calibrated target-like state (0.9/0.1), apply one real MOVE
        warp in the boundary band, a probability REMOVE corruption in foreground,
        and a probability ADD corruption in background.  Every FN/FP corruption is
        exactly crossable by an R2 dose in [0,1]; MOVE supervision is later chosen
        by searching the same warp dictionary used by deployment.
        """
        hard = (target4.detach() >= 0.5).to(target4)
        base = hard * 0.90 + (1.0 - hard) * 0.10
        b, _, h, w = base.shape
        sdf = self._truncated_signed_distance(hard, self.sdf_radius_px).detach()
        gx, gy = self._gradient_xy(sdf)
        norm = torch.sqrt(gx.square() + gy.square()).clamp_min(1.0e-6)
        nx, ny = gx / norm, gy / norm
        # Exactly one dictionary-sized boundary corruption (nearest non-zero bin).
        bins = self.transition_move_bins_px.to(base)
        nz = bins[bins.abs() > 1.0e-6]
        d = nz[nz.abs().argmin()] if nz.numel() else base.new_tensor(1.0)
        flow = torch.cat([nx * d, ny * d], dim=1)
        moved = torch.sigmoid(self._warp_patch_logits_px(self._logit(base), flow))
        boundary = sdf.abs() <= float(self.boundary_radius_px)
        out = torch.where(boundary, moved, base)

        # Topology corruptions use the inverse probability-dose family.
        remove_corrupt_rho = 0.75
        add_corrupt_rho = 0.75
        deep_fg = hard.bool() & (sdf > min(2.0, float(self.boundary_radius_px)))
        deep_bg = (~hard.bool()) & (sdf < -min(2.0, float(self.boundary_radius_px)))
        block = max(3, 2 * int(self.boundary_radius_px) + 3)
        block = min(block, h if h % 2 else h - 1, w if w % 2 else w - 1)
        block = max(block, 3)
        half = block // 2

        for bi in range(b):
            if bool(deep_fg[bi, 0].any()):
                yy, xx = torch.where(deep_fg[bi, 0])
                cy = int(torch.round(yy.float().mean()).item())
                cx = int(torch.round(xx.float().mean()).item())
                y0, y1 = max(0, cy-half), min(h, cy+half+1)
                x0, x1 = max(0, cx-half), min(w, cx+half+1)
                m = deep_fg[bi:bi+1, :, y0:y1, x0:x1]
                q = out[bi:bi+1, :, y0:y1, x0:x1]
                corr = q * (1.0 - remove_corrupt_rho)
                out[bi:bi+1, :, y0:y1, x0:x1] = torch.where(m, corr, q)
            if bool(deep_bg[bi, 0].any()):
                yy, xx = torch.where(deep_bg[bi, 0])
                # choose a stable background location away from the lesion centroid
                pick = int((yy.numel() - 1) // 3)
                cy, cx = int(yy[pick].item()), int(xx[pick].item())
                y0, y1 = max(0, cy-half), min(h, cy+half+1)
                x0, x1 = max(0, cx-half), min(w, cx+half+1)
                m = deep_bg[bi:bi+1, :, y0:y1, x0:x1]
                q = out[bi:bi+1, :, y0:y1, x0:x1]
                corr = q + add_corrupt_rho * (1.0 - q)
                out[bi:bi+1, :, y0:y1, x0:x1] = torch.where(m, corr, q)
        return out.clamp(EPS, 1.0 - EPS)
