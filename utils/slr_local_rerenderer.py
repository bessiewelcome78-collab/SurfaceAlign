"""Operator-consistent sparse high-resolution residual actor (SLR 2.4 / OCRA).

Scientific contract
-------------------
* Base/Geometry and selector stay on the canonical low-resolution grid.
* The local refiner receives a true paired high-resolution image view.
* A 55x55 context halo is decoded and only the central 33x33 patch deploys.
* ROI-relative XY coordinates remain absent.
* The local actor predicts a typed KEEP/ADD/REMOVE action and a non-negative dose.
  Patch supervision and deployment consume the exact same signed logit action.
  Geometry boundary weight is evidence only and never attenuates the actuator.
* Signed distance is an auxiliary geometry task, predicted directly as absolute
  GT SDF with a stationary target.  SDF is never mapped into deployment logits.
* Patch actor supervision covers the actual WOLA-deployable support. Selector
  supervision is residual error density, and training-only positives are
  drawn from selector-proposed non-deployed candidates rather than a GT-only map.
* Overlapping deployed logit residuals are fused by normalized weighted
  overlap-add.  GT is used only for training targets/samplers and validation
  diagnostics, never for eval routing or deployment.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .c2r_canonical_roi_refiner import (
    CanonicalCounterfactualROIRefiner,
    _ConvNormGELU,
    EPS,
)


class GeometryConditionedSparseLocalRerenderer(CanonicalCounterfactualROIRefiner):
    # Keep the established sparse-local stage identifier. OCRA is a corrected
    # actor contract inside the same stage, not a new downstream M3 stage.
    STAGE_ID = 9.0

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
        residual_logit_scale: float = 4.0,  # legacy parser compatibility only
        sdf_radius_px: int = 8,
        train_oracle_regions: Optional[int] = None,
        train_positive_regions: Optional[int] = None,
        train_clean_regions: int = 4,
        clean_max_error_fraction: float = 0.02,
        clean_max_sdf_discrepancy: float = 0.05,
        blend_taper_px: int = 4,
        context_size: int = 55,
        hr_size: int = 448,
        heaviside_tau_px: float = 1.0,  # legacy config compatibility; not used in deploy
        require_true_hr: bool = True,
        action_margin_prob: float = 0.05,
    ) -> None:
        requested_min_center_distance = int(
            max(1, region_size // 2) if min_center_distance is None else min_center_distance
        )
        super().__init__(
            hidden_dim,
            semantic_channels,
            text_dim,
            fine_feature_channels=fine_feature_channels,
            num_regions=num_regions,
            region_size=region_size,
            counterfactual_radius=1,
            selection_score="margin",
            flow_scale_px=flow_scale_px,
            min_center_distance=region_size,
            seed_radius=0,
            component_agreement_threshold=0.0,
            component_spread_threshold=1.0e9,
            component_confidence_threshold=0.0,
            component_min_area=1,
            canonical_residual_scale=1.0,
        )
        # Remove the inherited C2R residual-logit action head. SLR uses its own
        # direct segmentation residual head plus an auxiliary absolute-SDF head.
        if hasattr(self, "region_residual_out"):
            del self.region_residual_out
        self.residual_logit_scale = float(residual_logit_scale)  # unused
        self.slr_min_center_distance = requested_min_center_distance
        self.sdf_radius_px = max(1, int(sdf_radius_px))
        self.blend_taper_px = max(0, int(blend_taper_px))
        self.context_size = int(context_size)
        self.hr_size = int(hr_size)
        self.heaviside_tau_px = max(float(heaviside_tau_px), 1.0e-3)  # legacy only
        self.require_true_hr = bool(require_true_hr)
        self.action_margin_prob = float(action_margin_prob)
        if not (0.0 < self.action_margin_prob < 0.5):
            raise ValueError("GEOTR_SLR_ACTION_MARGIN_PROB must be in (0,0.5)")
        if self.context_size < self.region_size or (self.context_size - self.region_size) % 2:
            raise ValueError("GEOTR_SLR_CONTEXT_SIZE must be >= REGION_SIZE with an even difference")
        self.context_halo = (self.context_size - self.region_size) // 2
        if self.context_halo < 11:
            raise ValueError("SLR context halo must be >=11 pixels for the configured decoder contract")
        if self.blend_taper_px > self.region_size // 2:
            raise ValueError("GEOTR_SLR_BLEND_TAPER_PX must be <= REGION_SIZE//2")
        if train_positive_regions is None:
            train_positive_regions = num_regions if train_oracle_regions is None else train_oracle_regions
        self.train_positive_regions = max(1, int(train_positive_regions))
        self.train_oracle_regions = self.train_positive_regions
        self.train_clean_regions = max(1, int(train_clean_regions))
        self.clean_max_error_fraction = float(clean_max_error_fraction)
        self.clean_max_sdf_discrepancy = float(clean_max_sdf_discrepancy)
        if not (0.0 <= self.clean_max_error_fraction < 1.0):
            raise ValueError("GEOTR_SLR_CLEAN_MAX_ERROR_FRACTION must be in [0,1)")
        if not (0.0 <= self.clean_max_sdf_discrepancy < 1.0):
            raise ValueError("GEOTR_SLR_CLEAN_MAX_SDF_DISCREPANCY must be in [0,1)")

        selector_in = hidden_dim + 11
        self.slr_selector = nn.Sequential(
            _ConvNormGELU(selector_in, hidden_dim, 3, dilation=1),
            _ConvNormGELU(hidden_dim, hidden_dim, 3, dilation=2),
            _ConvNormGELU(hidden_dim, hidden_dim, 3, dilation=1),
        )
        self.slr_selector_out = nn.Conv2d(hidden_dim, 1, 1)
        nn.init.normal_(self.slr_selector_out.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.slr_selector_out.bias)

        # True-HR local encoder: a 110x110 crop becomes a 55x55 feature map.
        hr_mid = max(hidden_dim // 2, 16)
        self.hr_encoder = nn.Sequential(
            _ConvNormGELU(3, hr_mid, 3, dilation=1),
            nn.Conv2d(hr_mid, hidden_dim, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(max(1, min(8, hidden_dim)), hidden_dim),
            nn.GELU(),
            _ConvNormGELU(hidden_dim, hidden_dim, 3, dilation=1),
        )
        # common H + evidence 11 + anchor SDF 1 + HR H.  No ROI-relative XY.
        decoder_in = 2 * hidden_dim + 12
        self.region_decoder = nn.Sequential(
            _ConvNormGELU(decoder_in, hidden_dim, 3, dilation=1),
            _ConvNormGELU(hidden_dim, hidden_dim, 3, dilation=2),
            _ConvNormGELU(hidden_dim, hidden_dim, 3, dilation=3),
            _ConvNormGELU(hidden_dim, hidden_dim, 3, dilation=1),
        )
        # Operator-consistent typed residual actor. State order is
        # KEEP/ADD/REMOVE. At zero initialization p(ADD)==p(REMOVE), hence the
        # signed action cancels exactly without a saturating residual cap.
        self.region_action_state_out = nn.Conv2d(hidden_dim, 3, 1)
        self.region_action_dose_out = nn.Conv2d(hidden_dim, 1, 1)
        nn.init.zeros_(self.region_action_state_out.weight)
        nn.init.zeros_(self.region_action_state_out.bias)
        nn.init.zeros_(self.region_action_dose_out.weight)
        nn.init.zeros_(self.region_action_dose_out.bias)

        # Auxiliary geometry task: direct *absolute* normalized SDF prediction.
        # There is deliberately no anchor-SDF skip connection here: the target is
        # stationary GT geometry even while the jointly-trained Geometry branch
        # changes over epochs.
        self.region_sdf_out = nn.Conv2d(hidden_dim, 1, 1)
        nn.init.zeros_(self.region_sdf_out.weight)
        nn.init.zeros_(self.region_sdf_out.bias)

        w1 = torch.ones(self.region_size, dtype=torch.float32)
        t = self.blend_taper_px
        if t > 0:
            ramp = 0.5 - 0.5 * torch.cos(torch.linspace(0.0, torch.pi, t + 1))
            w1[: t + 1] = ramp
            w1[-(t + 1) :] = torch.flip(ramp, dims=[0])
        self.register_buffer(
            "slr_patch_window", torch.outer(w1, w1)[None, None, None], persistent=False
        )

    @staticmethod
    def _logit(prob: torch.Tensor) -> torch.Tensor:
        return torch.logit(prob.clamp(EPS, 1.0 - EPS))

    def _flow_evidence(self, flow_px: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        flow = flow_px.detach().to(anchor)
        if tuple(flow.shape[-2:]) != tuple(anchor.shape[-2:]):
            flow = self._resize(flow, tuple(anchor.shape[-2:]))
        flow = (flow / max(self.flow_scale_px, 1.0e-6)).clamp(-1.0, 1.0)
        mag = torch.sqrt(flow[:, 0:1].square() + flow[:, 1:2].square() + 1.0e-12).clamp(0, 1)
        return torch.cat([flow[:, 0:1], flow[:, 1:2], mag], 1)

    @staticmethod
    def _patch_error_density(error4: torch.Tensor, region_size: int) -> torch.Tensor:
        return F.avg_pool2d(
            error4.float(), kernel_size=region_size, stride=1, padding=region_size // 2
        ).clamp(0, 1)

    _normalized_patch_error_value = _patch_error_density

    def _slr_greedy_centers(self, score, k=None, valid_map=None):
        if score.ndim != 4 or score.shape[1] != 1:
            raise ValueError("SLR selector score must be [B,1,H,W]")
        b, _, h, w = score.shape
        k = self.num_regions if k is None else int(k)
        coords = torch.zeros(b, k, 2, dtype=torch.long, device=score.device)
        coord_valid = torch.zeros(b, k, dtype=torch.bool, device=score.device)
        centers = torch.zeros_like(score, dtype=torch.bool)
        valid = torch.ones(b, h, w, dtype=torch.bool, device=score.device)
        if valid_map is not None:
            vm = valid_map[:, 0] if valid_map.ndim == 4 else valid_map
            valid &= vm.bool()
        yy = torch.arange(h, device=score.device)[None, :, None]
        xx = torch.arange(w, device=score.device)[None, None, :]
        flat_score = score[:, 0]
        dmin = max(1, int(self.slr_min_center_distance))
        batch_idx = torch.arange(b, device=score.device)
        for j in range(k):
            ranked = flat_score.masked_fill(~valid, float("-inf")).reshape(b, -1)
            best_val, best_idx = ranked.max(dim=1)
            is_valid = torch.isfinite(best_val)
            cy = torch.div(best_idx, w, rounding_mode="floor")
            cx = best_idx % w
            coords[:, j, 0], coords[:, j, 1] = cy, cx
            coord_valid[:, j] = is_valid
            centers[batch_idx[is_valid], 0, cy[is_valid], cx[is_valid]] = True
            dy, dx = (yy - cy[:, None, None]).abs(), (xx - cx[:, None, None]).abs()
            valid &= ~((dy < dmin) & (dx < dmin) & is_valid[:, None, None])
        return centers, coords, coord_valid

    @staticmethod
    def _grid_for_centers(center_yx, center_valid, h, w, size, dtype):
        b, k, _ = center_yx.shape
        half = size // 2
        offs = torch.arange(-half, half + 1, device=center_yx.device, dtype=torch.long)
        yy = center_yx[:, :, 0, None, None] + offs[None, None, :, None]
        xx = center_yx[:, :, 1, None, None] + offs[None, None, None, :]
        yy = yy.expand(-1, -1, size, size)
        xx = xx.expand(-1, -1, size, size)
        valid = (
            center_valid[:, :, None, None]
            & (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        )
        yyc = yy.clamp(0, h - 1)
        xxc = xx.clamp(0, w - 1)
        gy = (2.0 * yyc.to(dtype) / max(h - 1, 1) - 1.0)
        gx = (2.0 * xxc.to(dtype) / max(w - 1, 1) - 1.0)
        grid = torch.stack([gx, gy], dim=-1).reshape(b * k, size, size, 2)
        return yyc, xxc, grid, valid[:, :, None]

    @staticmethod
    def _dense_normalized_hr_grid(
        center_yx, center_valid, h, w, hr_h, hr_w, context_size, scale, dtype
    ):
        """Pixel-centre aligned LR->HR sampling grid.

        For an integer scale s, LR pixel x corresponds to the HR cell containing
        pixel centres s*x ... s*x+s-1.  A context of C LR pixels therefore maps
        to exactly C*s native HR pixel centres.  ``align_corners=False`` is used
        consistently with the half-pixel normalization.
        """
        del center_valid  # spatial padding is handled by grid_sample(border)
        b, k, _ = center_yx.shape
        out = int(context_size) * int(scale)
        half = int(context_size) // 2
        start_y = int(scale) * (center_yx[:, :, 0].to(dtype) - float(half))
        start_x = int(scale) * (center_yx[:, :, 1].to(dtype) - float(half))
        offs = torch.arange(out, device=center_yx.device, dtype=dtype)
        yy = start_y[:, :, None, None] + offs[None, None, :, None]
        xx = start_x[:, :, None, None] + offs[None, None, None, :]
        yy = yy.expand(-1, -1, out, out)
        xx = xx.expand(-1, -1, out, out)
        gy = 2.0 * (yy + 0.5) / float(hr_h) - 1.0
        gx = 2.0 * (xx + 0.5) / float(hr_w) - 1.0
        return torch.stack([gx, gy], dim=-1).reshape(b * k, out, out, 2)

    @staticmethod
    def _crop_grid(tensor, grid, k):
        b, c, _, _ = tensor.shape
        return F.grid_sample(
            tensor[:, None].expand(-1, k, -1, -1, -1).reshape(b * k, c, *tensor.shape[-2:]),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).view(b, k, c, grid.shape[1], grid.shape[2])

    @staticmethod
    def _center_crop_patch(x: torch.Tensor, out_size: int) -> torch.Tensor:
        s = (x.shape[-1] - out_size) // 2
        return x[..., s : s + out_size, s : s + out_size]

    def _weighted_overlap_add(self, patch, anchor, ay, ax, valid):
        b, k, v, r, _ = patch.shape
        h, w = anchor.shape[-2:]
        idx = (ay * w + ax).reshape(b, -1)
        win = self.slr_patch_window.to(patch).expand(b, k, 1, r, r)
        weights = (win[:, :, 0] * valid[:, :, 0].to(win)).reshape(b, -1)
        vals = patch.permute(0, 2, 1, 3, 4).reshape(b, v, -1)
        num = anchor.new_zeros((b, v, h * w))
        den = anchor.new_zeros((b, 1, h * w))
        counts = anchor.new_zeros((b, 1, h * w))
        num.scatter_add_(2, idx[:, None].expand(-1, v, -1), vals * weights[:, None])
        den.scatter_add_(2, idx[:, None], weights[:, None])
        counts.scatter_add_(2, idx[:, None], valid[:, :, 0].reshape(b, -1).to(anchor)[:, None])
        anchor_v = anchor.expand(-1, v, -1, -1).reshape(b, v, h * w)
        full = torch.where(den > 1e-8, num / den.clamp_min(1e-8), anchor_v).view(b, v, h, w)
        return full, (den > 1e-8).view(b, 1, h, w), (counts > 1.0).float().sum(), den.view(b, 1, h, w)

    def _weighted_overlap_disagreement(self, patch, anchor, ay, ax, valid):
        b, k, v, r, _ = patch.shape
        h, w = anchor.shape[-2:]
        idx = (ay * w + ax).reshape(b, -1)
        win = self.slr_patch_window.to(patch).expand(b, k, 1, r, r)
        weights = (win[:, :, 0] * valid[:, :, 0].to(win)).reshape(b, -1)
        vals = patch.permute(0, 2, 1, 3, 4).reshape(b, v, -1)
        num = anchor.new_zeros((b, v, h * w)); num2 = torch.zeros_like(num)
        den = anchor.new_zeros((b, 1, h * w)); counts = torch.zeros_like(den)
        num.scatter_add_(2, idx[:, None].expand(-1, v, -1), vals * weights[:, None])
        num2.scatter_add_(2, idx[:, None].expand(-1, v, -1), vals.square() * weights[:, None])
        den.scatter_add_(2, idx[:, None], weights[:, None])
        counts.scatter_add_(2, idx[:, None], valid[:, :, 0].reshape(b, -1).to(anchor)[:, None])
        mean = num / den.clamp_min(1e-8)
        var = (num2 / den.clamp_min(1e-8) - mean.square()).clamp_min(0)
        ov = (counts > 1.0).expand(-1, v, -1)
        return torch.where(ov, torch.sqrt(var), torch.zeros_like(var)).sum() / ov.float().sum().clamp_min(1)

    @staticmethod
    def _truncated_signed_distance(mask4: torch.Tensor, radius: int) -> torch.Tensor:
        """Label-faithful pixel-centred truncated signed distance.

        Foreground pixel centres are strictly positive and background centres
        strictly negative.  The continuous interface is placed half a pixel
        between opposite-label neighbouring pixel centres, so the nearest
        foreground/background centres receive +0.5/-0.5 rather than a shared
        ambiguous zero target.  An 8-neighbour chamfer propagation approximates
        Euclidean distance while keeping the target pure Torch/GPU friendly.
        """
        radius = max(int(radius), 1)
        m = mask4 > 0.5
        inf = float(radius) + 2.0

        def shift(x, dy, dx):
            h, w = x.shape[-2:]
            pl, pr, pt, pb = max(dx,0), max(-dx,0), max(dy,0), max(-dy,0)
            y = F.pad(x, (pl,pr,pt,pb), value=inf)
            return y[..., pb:pb+h, pr:pr+w]

        def distance_to(seed_mask: torch.Tensor) -> torch.Tensor:
            dist = torch.where(
                seed_mask,
                mask4.new_zeros(()),
                mask4.new_full((), inf),
            )
            diag = 2.0 ** 0.5
            # radius+1 is enough to resolve every unsaturated value <= R+0.5.
            for _ in range(radius + 1):
                cand = [dist]
                for dy,dx,cost in (
                    (-1,0,1.),(1,0,1.),(0,-1,1.),(0,1,1.),
                    (-1,-1,diag),(-1,1,diag),(1,-1,diag),(1,1,diag),
                ):
                    cand.append(shift(dist,dy,dx)+cost)
                dist = torch.stack(cand, dim=0).amin(dim=0)
            return dist

        d_to_bg = distance_to(~m)
        d_to_fg = distance_to(m)
        inside = (d_to_bg - 0.5).clamp(min=0.5, max=float(radius))
        outside = (d_to_fg - 0.5).clamp(min=0.5, max=float(radius))
        return torch.where(m, inside, -outside).to(mask4)

    def _boundary_weight_from_sdf(self, anchor_sdf: torch.Tensor) -> torch.Tensor:
        """Deterministic boundary-local deployment support in [0,1].

        The support is derived only from deployed Geometry.  It is strongest at
        the zero-level neighbourhood and decays linearly to zero at the existing
        truncated radius R.  Therefore saturated far-interior/exterior pixels
        are exact Geometry identity without a learned safety/gating controller.
        """
        return (1.0 - anchor_sdf.detach().abs() / float(self.sdf_radius_px)).clamp(0.0, 1.0)

    def _dense_evidence(self, anchor, base, flow, margin, mc_std, mc_dis, entropy):
        a_logit, b_logit = self._logit(anchor), self._logit(base)
        transition_logit = torch.tanh((a_logit - b_logit) / 2.0)
        return torch.cat([anchor, base, anchor-base, transition_logit, margin, mc_std, mc_dis, entropy, self._flow_evidence(flow, anchor)], 1)

    def _typed_action_targets(self, anchor_patch: torch.Tensor, target_patch: torch.Tensor):
        """KEEP/ADD/REMOVE labels and minimum threshold-crossing logit dose."""
        anchor_patch = anchor_patch.detach().clamp(EPS, 1.0 - EPS)
        target_hard = target_patch.detach() >= 0.5
        anchor_hard = anchor_patch >= 0.5
        add = target_hard & (~anchor_hard)
        remove = (~target_hard) & anchor_hard
        state = torch.zeros_like(anchor_patch, dtype=torch.long)
        state = torch.where(add, torch.ones_like(state), state)
        state = torch.where(remove, torch.full_like(state, 2), state)
        z = self._logit(anchor_patch)
        margin = anchor_patch.new_tensor(self.action_margin_prob)
        z_pos = torch.log((0.5 + margin) / (0.5 - margin))
        z_neg = -z_pos
        add_dose = (z_pos - z).clamp_min(0.0)
        remove_dose = (z - z_neg).clamp_min(0.0)
        dose = torch.where(add, add_dose, torch.where(remove, remove_dose, torch.zeros_like(z)))
        signed = torch.where(add, dose, torch.where(remove, -dose, torch.zeros_like(dose)))
        return state, dose, signed

    def _decode_centers(self, center_yx, center_valid, *, common, dense_ev, anchor, anchor_sdf, hr_image):
        b, _, h, w = anchor.shape
        if hr_image is None:
            if self.require_true_hr:
                raise RuntimeError("GEOTR-SLR true-HR contract requires slr_hr_image; LR upsampling is forbidden")
            hr_image = F.interpolate(common[:, :3], scale_factor=2, mode="bilinear", align_corners=False)
        if hr_image.ndim != 4 or hr_image.shape[0] != b:
            raise ValueError("SLR hr_image must be [B,3,Hh,Wh]")
        if hr_image.shape[1] != 3:
            raise ValueError("SLR hr_image must have 3 channels")
        sy = int(hr_image.shape[-2]) // h; sx = int(hr_image.shape[-1]) // w
        if sy != 2 or sx != 2 or hr_image.shape[-2] != h*2 or hr_image.shape[-1] != w*2:
            raise ValueError(f"SLR true-HR contract requires exactly 2x LR spatial size, got LR={(h,w)} HR={tuple(hr_image.shape[-2:])}")
        k = center_yx.shape[1]
        _,_,cgrid,_ = self._grid_for_centers(center_yx,center_valid,h,w,self.context_size,anchor.dtype)
        oay,oax,ogrid,ovalid = self._grid_for_centers(center_yx,center_valid,h,w,self.region_size,anchor.dtype)
        common_c = self._crop_grid(common,cgrid,k)
        ev_c = self._crop_grid(dense_ev,cgrid,k)
        asdf_c = self._crop_grid(anchor_sdf,cgrid,k) / float(self.sdf_radius_px)
        hr_grid = self._dense_normalized_hr_grid(
            center_yx, center_valid, h, w, hr_image.shape[-2], hr_image.shape[-1],
            self.context_size, 2, hr_image.dtype
        )
        hr_expand = hr_image[:,None].expand(-1,k,-1,-1,-1).reshape(b*k,3,*hr_image.shape[-2:])
        hr_crop = F.grid_sample(hr_expand, hr_grid, mode="bilinear", padding_mode="border", align_corners=False)
        hr_feat = self.hr_encoder(hr_crop).view(b,k,-1,self.context_size,self.context_size)
        x = torch.cat([common_c, ev_c, asdf_c, hr_feat], 2)
        hidden = self.region_decoder(x.reshape(b*k,x.shape[2],self.context_size,self.context_size))

        # ------------------------------------------------------------------
        # Primary deploy path: typed residual actor. Patch supervision and
        # deployment consume the same signed action tensor. Geometry boundary
        # weight remains a diagnostic/evidence prior only.
        # ------------------------------------------------------------------
        state_logits_ctx = self.region_action_state_out(hidden)
        state_probs_ctx = torch.softmax(state_logits_ctx.float(), dim=1).to(hidden)
        dose_ctx = F.softplus(self.region_action_dose_out(hidden))
        signed_action_ctx = (state_probs_ctx[:,1:2] - state_probs_ctx[:,2:3]) * dose_ctx
        action_state_logits = self._center_crop_patch(
            state_logits_ctx.view(b,k,3,self.context_size,self.context_size), self.region_size
        )
        action_state_probs = self._center_crop_patch(
            state_probs_ctx.view(b,k,3,self.context_size,self.context_size), self.region_size
        )
        action_dose = self._center_crop_patch(
            dose_ctx.view(b,k,1,self.context_size,self.context_size), self.region_size
        )
        seg_delta = self._center_crop_patch(
            signed_action_ctx.view(b,k,1,self.context_size,self.context_size), self.region_size
        )
        anchor_prob_patch = self._crop_grid(anchor, ogrid, k)
        anchor_logit_patch = self._logit(anchor_prob_patch)
        anchor_sdf_patch = self._crop_grid(anchor_sdf, ogrid, k)
        boundary_weight_patch = self._boundary_weight_from_sdf(anchor_sdf_patch) * ovalid.to(anchor)
        # WOLA has exactly zero window weight only on the patch edge. Excluding
        # those pixels from both actor supervision and deployment gives an exact
        # executable support contract without SDF-dependent dose attenuation.
        action_support_patch = (self.slr_patch_window.to(anchor) > 0).to(anchor) * ovalid.to(anchor)
        deploy_delta_patch = seg_delta * action_support_patch
        raw_patch_logits = anchor_logit_patch + deploy_delta_patch
        raw_patch_probs = torch.sigmoid(raw_patch_logits).clamp(EPS, 1.0-EPS)
        patch_logits = raw_patch_logits
        patch_probs = raw_patch_probs
        full_delta, region, overlap, weight_sum = self._weighted_overlap_add(
            deploy_delta_patch, torch.zeros_like(anchor), oay, oax, ovalid
        )
        anchor_logits = self._logit(anchor)
        full_logits = anchor_logits + full_delta
        full_mapped = torch.sigmoid(full_logits).clamp(EPS, 1.0-EPS)
        full_prob = torch.where(
            full_delta.detach().abs() <= 1.0e-12, anchor, full_mapped
        )
        # Keep logits exactly synchronized with the probability actually deployed.
        full_logits = self._logit(full_prob)
        seg_overlap_dis = self._weighted_overlap_disagreement(
            deploy_delta_patch, torch.zeros_like(anchor), oay, oax, ovalid
        )

        # ------------------------------------------------------------------
        # Auxiliary geometry path: direct absolute SDF.  No anchor-SDF residual
        # parameterization is used, so the learning target is stationary GT SDF.
        # The SDF is not used to generate deploy probabilities.
        # ------------------------------------------------------------------
        # Prevent the auxiliary SDF objective from rotating the shared actor
        # gradient. The SDF head still learns on actor features, but its gradient
        # stops at the shared local decoder.
        sdf_norm_ctx = self.region_sdf_out(hidden.detach()).view(b,k,1,self.context_size,self.context_size)
        pred_sdf_ctx = float(self.sdf_radius_px) * sdf_norm_ctx
        pred_sdf = self._center_crop_patch(pred_sdf_ctx, self.region_size) * ovalid.to(anchor)
        patch_sdf_correction = pred_sdf - anchor_sdf_patch
        full_sdf, _, _, _ = self._weighted_overlap_add(
            pred_sdf, anchor_sdf, oay, oax, ovalid
        )
        sdf_overlap_dis = self._weighted_overlap_disagreement(
            pred_sdf, anchor_sdf, oay, oax, ovalid
        )
        return {
            "center_yx":center_yx,"center_valid":center_valid,"ay":oay,"ax":oax,"grid":ogrid,"valid_patch":ovalid,
            "patch_logits":patch_logits,"patch_probs":patch_probs,
            "raw_patch_logits":raw_patch_logits,"raw_patch_probs":raw_patch_probs,
            "patch_action_support":action_support_patch,"patch_seg_delta_logit":deploy_delta_patch,
            "patch_residual_magnitude":torch.abs(deploy_delta_patch),
            "patch_deploy_delta_logit":deploy_delta_patch,"patch_boundary_weight":boundary_weight_patch,
            "patch_action_state_logits":action_state_logits,"patch_action_state_probs":action_state_probs,
            "patch_action_dose":action_dose,"patch_signed_action":deploy_delta_patch,
            "patch_sdf_absolute":pred_sdf,"patch_sdf_anchor":anchor_sdf_patch,"patch_sdf_correction":patch_sdf_correction,
            "full_logits":full_logits,"full_prob":full_prob,"full_sdf_absolute":full_sdf,
            "region":region,"overlap":overlap,"blend_weight_sum":weight_sum,
            "overlap_disagreement":seg_overlap_dis,"overlap_sdf_disagreement":sdf_overlap_dis,
            "hr_crop":hr_crop,"context_size":anchor.new_tensor(float(self.context_size)),
        }

    def forward(
        self, anchor_prob, image, semantic_map, text_features, *, base_prob=None, flow_px=None,
        mc_std_map=None, mc_disagreement_map=None, fine_feature_map=None,
        posterior_probability_samples=None, supervision_masks=None, hr_image=None,
    ) -> Dict[str,torch.Tensor]:
        del posterior_probability_samples
        anchor=anchor_prob.detach().clamp(EPS,1-EPS)
        base=anchor if base_prob is None else base_prob.detach().to(anchor).clamp(EPS,1-EPS)
        if flow_px is None: flow_px=anchor.new_zeros((anchor.shape[0],2,*anchor.shape[-2:]))
        flow=flow_px.detach().to(anchor); b,_,h,w=anchor.shape; hw=(h,w)
        margin=self._margin_uncertainty(anchor); mc_std=self._fit_evidence(mc_std_map,anchor,scale=2.0)
        mc_dis=self._fit_evidence(mc_disagreement_map,anchor,scale=1.0); entropy=self._entropy(anchor)
        common=self._common_visual(image,semantic_map,text_features,fine_feature_map,hw)
        dense_ev=self._dense_evidence(anchor,base,flow,margin,mc_std,mc_dis,entropy)
        selector_features=torch.cat([common,dense_ev],1)
        selector_logits=self.slr_selector_out(self.slr_selector(selector_features)); selector_prob=torch.sigmoid(selector_logits)
        # Proposal contract: inference deploys the first K selector proposals. During
        # training the same selector proposes K+K_pos centres; the extra proposals
        # are the *only* pool from which GT may select positive training examples.
        proposal_k = self.num_regions + (self.train_positive_regions if self.training else 0)
        proposal_centers,proposal_yx,proposal_valid=self._slr_greedy_centers(selector_prob,k=proposal_k)
        center_yx=proposal_yx[:, :self.num_regions]
        center_valid=proposal_valid[:, :self.num_regions]
        centers=torch.zeros_like(selector_prob,dtype=torch.bool)
        for bi in range(b):
            vv=center_valid[bi]
            if bool(vv.any()):
                yy=center_yx[bi,vv,0]; xx=center_yx[bi,vv,1]
                centers[bi,0,yy,xx]=True
        anchor_sdf=self._truncated_signed_distance((anchor>=0.5).to(anchor),self.sdf_radius_px).detach()
        pred=self._decode_centers(center_yx,center_valid,common=common,dense_ev=dense_ev,anchor=anchor,anchor_sdf=anchor_sdf,hr_image=hr_image)
        final_logits,final_prob,region=pred["full_logits"],pred["full_prob"],pred["region"]
        delta=final_logits-self._logit(anchor)

        zfull=torch.zeros_like(anchor); train_selector_target=zfull
        train_sdf_discrepancy=zfull; positive_score=zfull; sdf_support_full=zfull
        boundary_weight_full=self._boundary_weight_from_sdf(anchor_sdf).detach()
        # OCRA can represent an action anywhere inside a selected WOLA patch.
        # Boundary geometry remains evidence, not an actuator.
        action_weight_full=torch.ones_like(anchor)
        action_support_full=torch.ones_like(anchor)
        positive_from_selector_fraction=anchor.new_zeros(())
        shape_pred=pred["patch_probs"].shape
        pred_target_patch=torch.zeros_like(pred["patch_probs"]); pred_sdf_target_patch=torch.zeros_like(pred["patch_sdf_absolute"]); pred_sdf_support_patch=torch.zeros_like(pred["patch_sdf_absolute"])
        pred_action_type_target=torch.zeros_like(pred["patch_action_dose"],dtype=torch.long)
        pred_action_dose_target=torch.zeros_like(pred["patch_action_dose"])
        pred_signed_action_target=torch.zeros_like(pred["patch_signed_action"])
        def empty_pack(k):
            q=anchor.new_zeros((b,k,1,self.region_size,self.region_size)); return q,torch.sigmoid(q),torch.zeros_like(q),torch.zeros_like(q),torch.zeros_like(q),torch.zeros_like(q),torch.zeros_like(q),torch.zeros_like(anchor)
        positive_patch_logits,positive_patch_probs,positive_patch_valid,positive_target_patch,positive_sdf_abs,positive_sdf_target,positive_sdf_support,positive_region=empty_pack(self.train_positive_regions)
        positive_sdf_anchor=torch.zeros_like(positive_sdf_abs)
        positive_action_state_logits=anchor.new_zeros((b,self.train_positive_regions,3,self.region_size,self.region_size))
        positive_action_state_probs=torch.softmax(positive_action_state_logits,dim=2)
        positive_action_dose=torch.zeros_like(positive_patch_logits); positive_signed_action=torch.zeros_like(positive_patch_logits)
        positive_action_support=torch.zeros_like(positive_patch_logits)
        positive_action_type_target=torch.zeros_like(positive_patch_logits,dtype=torch.long)
        positive_action_dose_target=torch.zeros_like(positive_patch_logits); positive_signed_action_target=torch.zeros_like(positive_patch_logits)
        clean_patch_logits,clean_patch_probs,clean_patch_valid,clean_target_patch,clean_sdf_abs,clean_sdf_target,clean_sdf_support,clean_region=empty_pack(self.train_clean_regions)
        clean_anchor_patch=torch.zeros_like(clean_patch_logits); clean_sdf_anchor=torch.zeros_like(clean_patch_logits)
        clean_action_state_logits=anchor.new_zeros((b,self.train_clean_regions,3,self.region_size,self.region_size))
        clean_action_state_probs=torch.softmax(clean_action_state_logits,dim=2)
        clean_action_dose=torch.zeros_like(clean_patch_logits); clean_signed_action=torch.zeros_like(clean_patch_logits)
        clean_action_support=torch.zeros_like(clean_patch_logits)
        clean_action_type_target=torch.zeros_like(clean_patch_logits,dtype=torch.long)
        clean_action_dose_target=torch.zeros_like(clean_patch_logits); clean_signed_action_target=torch.zeros_like(clean_patch_logits)

        if self.training and isinstance(supervision_masks,torch.Tensor):
            target4=supervision_masks.detach().to(anchor)
            if target4.ndim==3: target4=target4[:,None]
            if target4.shape[-2:]!=hw: target4=F.interpolate(target4.float(),size=hw,mode="nearest").to(anchor)
            target4=(target4>=0.5).to(anchor)
            err4=((anchor>=0.5)!=(target4>=0.5)).to(anchor)
            gt_sdf=self._truncated_signed_distance(target4,self.sdf_radius_px).detach()
            sdf_discrepancy_pixel=((anchor_sdf-gt_sdf).abs()/float(self.sdf_radius_px)).clamp(0,1)
            train_sdf_discrepancy=self._patch_error_density(sdf_discrepancy_pixel,self.region_size).detach()

            # Operator-aligned WHERE: every residual inside a selected WOLA patch
            # is executable, so selector supervision is factual error density.
            actionable_error=err4.detach()
            train_selector_target=self._patch_error_density(actionable_error,self.region_size).detach()

            # Segmentation supervision is restricted to action-reachable pixels.
            # SDF regression remains full-valid and stationary in the loss.
            sdf_support_full=action_support_full
            pred_target_patch=self._crop_grid(target4,pred["grid"],self.num_regions).detach()
            pred_sdf_target_patch=self._crop_grid(gt_sdf,pred["grid"],self.num_regions).detach()
            pred_sdf_support_patch=self._crop_grid(action_support_full,pred["grid"],self.num_regions).detach()
            pred_anchor_patch=self._crop_grid(anchor,pred["grid"],self.num_regions).detach()
            pred_action_type_target,pred_action_dose_target,pred_signed_action_target=self._typed_action_targets(pred_anchor_patch,pred_target_patch)

            # Selector-conditioned positive mining.  GT never proposes an arbitrary
            # new centre: it only validates/ranks the selector's extra K_pos proposals.
            positive_score=train_selector_target.detach()
            extra_yx=proposal_yx[:, self.num_regions:self.num_regions+self.train_positive_regions].clone()
            extra_valid=proposal_valid[:, self.num_regions:self.num_regions+self.train_positive_regions].clone()
            if extra_yx.shape[1] > 0:
                bi=torch.arange(b,device=anchor.device)[:,None].expand(-1,extra_yx.shape[1])
                candidate_value=train_selector_target[:,0][bi,extra_yx[:,:,0],extra_yx[:,:,1]]
                extra_valid = extra_valid & (candidate_value>0)
                # Stable descending GT-actionability order within the selector pool.
                order=torch.argsort(candidate_value.masked_fill(~extra_valid,-1.0),dim=1,descending=True)
                extra_yx=torch.gather(extra_yx,1,order[:,:,None].expand(-1,-1,2))
                extra_valid=torch.gather(extra_valid,1,order)
            pyx,pvalid=extra_yx,extra_valid
            positive_from_selector_fraction = pvalid.float().sum() / pvalid.numel() if pvalid.numel() else anchor.new_zeros(())
            pos=self._decode_centers(pyx,pvalid,common=common,dense_ev=dense_ev,anchor=anchor,anchor_sdf=anchor_sdf,hr_image=hr_image)
            positive_patch_logits=pos["raw_patch_logits"]; positive_patch_probs=pos["raw_patch_probs"]; positive_patch_valid=pos["valid_patch"].to(anchor)
            positive_action_state_logits=pos["patch_action_state_logits"]; positive_action_state_probs=pos["patch_action_state_probs"]
            positive_action_dose=pos["patch_action_dose"]; positive_signed_action=pos["patch_signed_action"]
            positive_action_support=pos["patch_action_support"]
            positive_sdf_anchor=pos["patch_sdf_anchor"].detach()
            positive_target_patch=self._crop_grid(target4,pos["grid"],pyx.shape[1]).detach(); positive_sdf_abs=pos["patch_sdf_absolute"]
            positive_sdf_target=self._crop_grid(gt_sdf,pos["grid"],pyx.shape[1]).detach(); positive_sdf_support=self._crop_grid(action_support_full,pos["grid"],pyx.shape[1]).detach(); positive_region=pos["region"].to(anchor)
            positive_anchor_patch=self._crop_grid(anchor,pos["grid"],pyx.shape[1]).detach()
            positive_action_type_target,positive_action_dose_target,positive_signed_action_target=self._typed_action_targets(positive_anchor_patch,positive_target_patch)

            boundary_hardness=(1-anchor_sdf.abs()/float(self.sdf_radius_px)).clamp(0,1); uncertainty_hardness=(margin+mc_std+mc_dis+entropy)/4
            clean_score=.5*boundary_hardness+.5*uncertainty_hardness
            clean_valid_map=((train_selector_target<=self.clean_max_error_fraction) &
                             (train_sdf_discrepancy<=self.clean_max_sdf_discrepancy))
            _,cyx,cvalid=self._slr_greedy_centers(clean_score,k=self.train_clean_regions,valid_map=clean_valid_map)
            clean=self._decode_centers(cyx,cvalid,common=common,dense_ev=dense_ev,anchor=anchor,anchor_sdf=anchor_sdf,hr_image=hr_image)
            clean_patch_logits=clean["patch_logits"]; clean_patch_probs=clean["patch_probs"]; clean_patch_valid=clean["valid_patch"].to(anchor)
            clean_action_state_logits=clean["patch_action_state_logits"]; clean_action_state_probs=clean["patch_action_state_probs"]
            clean_action_dose=clean["patch_action_dose"]; clean_signed_action=clean["patch_signed_action"]
            clean_action_support=clean["patch_action_support"]
            clean_target_patch=self._crop_grid(target4,clean["grid"],cyx.shape[1]).detach(); clean_anchor_patch=self._crop_grid(anchor,clean["grid"],cyx.shape[1]).detach()
            clean_sdf_abs=clean["patch_sdf_absolute"]; clean_sdf_anchor=clean["patch_sdf_anchor"].detach(); clean_sdf_target=self._crop_grid(gt_sdf,clean["grid"],cyx.shape[1]).detach(); clean_sdf_support=self._crop_grid(sdf_support_full,clean["grid"],cyx.shape[1]).detach(); clean_region=clean["region"].to(anchor)
            clean_action_type_target,clean_action_dose_target,clean_signed_action_target=self._typed_action_targets(clean_anchor_patch,clean_target_patch)

        action_region = region & (action_support_full > 0)
        hard_change=((final_prob>=.5)!=(anchor>=.5))&action_region; center_float=centers.to(anchor); zero=torch.zeros_like(anchor); op=torch.full((b,),-1,dtype=torch.long,device=anchor.device)
        # There is one factual deterministic view. Do not fabricate consensus by
        # expanding it three times; downstream diagnostics must report N/A/zero
        # spread for a genuine single-view actor.
        view_probs=final_prob; view_logits=final_logits
        pred_anchor_patch=pred["patch_sdf_anchor"]
        out={
            "logits":final_logits,"prob":final_prob,"selection_mask":region.to(anchor),"selection_score":selector_prob,"refined_logits":final_logits,"refined_prob":final_prob,"delta_logit":delta,
            "margin_uncertainty":margin,"mc_std_map":mc_std,"mc_disagreement_map":mc_dis,"entropy_map":entropy,"trace":dense_ev,"fine_feature_map":common,
            "dn_anchor_prob":anchor,"dn_corruption_mask":zero,"dn_selection_mask":zero,"dn_refined_logits":self._logit(anchor),"dn_refined_prob":anchor,"dn_final_prob":anchor,"dn_delta_logit":zero,"dn_op_id":op,
            "r4_flip_logits":zero,"r4_flip_prob":zero,"r4_flip_mask":zero,"r4_synth_flip_logits":zero,"r4_synth_flip_prob":zero,"r4_synth_flip_mask":zero,"r4_synth_target":zero,
            "r2_enabled":anchor.new_zeros((b,)),"r3_enabled":anchor.new_zeros((b,)),"r4_enabled":anchor.new_zeros((b,)),"r41_enabled":anchor.new_zeros((b,)),
            "c2r_center_mask":center_float,"c2r_region_mask":region.to(anchor),"c2r_view_logits":view_logits,"c2r_view_probs":view_probs,"c2r_mean_prob":final_prob,"c2r_consensus_mask":region.to(anchor),"c2r_edit_mask":hard_change.to(anchor),"c2r_candidate_mask":hard_change.to(anchor),"c2r_commit_mask":hard_change.to(anchor),"c2r_roi_overlap_pixel_count":pred["overlap"],"c2r_roi_unique_pixel_count":region.float().sum(),"c2r_center_min_chebyshev_distance":anchor.new_tensor(float(self.slr_min_center_distance)),
            "aefr_enabled":anchor.new_ones((b,)),"aefr_stage_id":anchor.new_full((b,),self.STAGE_ID),"aefr_joint_geometry_grad":anchor.new_zeros((b,)),"aefr_transition_aware":anchor.new_ones((b,)),"aefr_raw_flow_evidence_enabled":anchor.new_ones((b,)),"aefr_transition_delta_logit":torch.tanh((self._logit(anchor)-self._logit(base))/2),"aefr_transition_abs_mean":(anchor-base).abs().mean(),"aefr_transition_active_fraction":((anchor-base).abs()>1e-4).float().mean(),"aefr_transition_flip_fraction":((anchor>=.5)!=(base>=.5)).float().mean(),"aefr_action_delta_logit":delta,"aefr_boundary_mask":zero,"aefr_boundary_displacement_px":zero,"aefr_interior_delta_logit":delta,"aefr_error_localizer_logit":selector_logits,"aefr_error_localizer_prob":selector_prob,"aefr_error_localizer_prior":zero,"aefr_edit_logit":zero,"aefr_edit_prob":hard_change.to(anchor),"aefr_direction_logit":zero,"aefr_direction_prob":zero,"aefr_interior_magnitude_logit":zero,"aefr_signed_action":zero,"aefr_commit_mask":zero,"aefr_boundary_magnitude_px":zero,"aefr_signed_boundary_action":zero,"aefr_signed_interior_action":zero,"aefr_posterior_stability_support":anchor.new_zeros(()),"aefr_posterior_stability_improvement":anchor.new_zeros(()),"aefr_posterior_disagreement_pre":anchor.new_zeros(()),"aefr_posterior_disagreement_post":anchor.new_zeros(()),"aefr_action_support_fraction":region.float().mean(),"aefr_posterior_diversity_all":anchor.new_zeros(()),"aefr_posterior_center_bias_abs":anchor.new_zeros(()),"aefr_posterior_center_bias_signed":anchor.new_zeros(()),
            "slr_selector_logits":selector_logits,"slr_selector_prob":selector_prob,"slr_selector_target":train_selector_target,
            # Compatibility pred_patch keys remain the deployed patch; explicit raw
            # keys expose predictor quality independently of the boundary actuator.
            "slr_pred_patch_logits":pred["patch_logits"],"slr_pred_patch_probs":pred["patch_probs"],
            "slr_raw_patch_logits":pred["raw_patch_logits"],"slr_raw_patch_probs":pred["raw_patch_probs"],
            "slr_pred_patch_grid":pred["grid"],"slr_pred_patch_center_valid":pred["center_valid"].to(anchor),"slr_pred_patch_valid":pred["valid_patch"].to(anchor),"slr_pred_patch_target":pred_target_patch,"slr_pred_patch_anchor_prob":self._crop_grid(anchor,pred["grid"],self.num_regions),
            "slr_pred_patch_delta_logit":pred["patch_deploy_delta_logit"],"slr_pred_patch_raw_delta_logit":pred["patch_seg_delta_logit"],"slr_pred_patch_boundary_weight":pred["patch_boundary_weight"],"slr_pred_patch_action_support":pred["patch_action_support"],
            "slr_pred_action_state_logits":pred["patch_action_state_logits"],"slr_pred_action_state_probs":pred["patch_action_state_probs"],"slr_pred_action_dose":pred["patch_action_dose"],"slr_pred_signed_action":pred["patch_signed_action"],"slr_pred_action_type_target":pred_action_type_target,"slr_pred_action_dose_target":pred_action_dose_target,"slr_pred_signed_action_target":pred_signed_action_target,
            "slr_pred_sdf_absolute":pred["patch_sdf_absolute"],"slr_pred_sdf_anchor":pred_anchor_patch,"slr_pred_sdf_delta":pred["patch_sdf_correction"],"slr_pred_sdf_target":pred_sdf_target_patch,"slr_pred_sdf_support":pred_sdf_support_patch,
            "slr_sdf_absolute_full":pred["full_sdf_absolute"],"slr_sdf_delta_full":pred["full_sdf_absolute"]-anchor_sdf,"slr_anchor_sdf_full":anchor_sdf,"slr_blend_weight_sum":pred["blend_weight_sum"],"slr_overlap_disagreement":pred["overlap_disagreement"],"slr_overlap_sdf_disagreement":pred["overlap_sdf_disagreement"],"slr_boundary_weight_full":boundary_weight_full,"slr_action_delta_full":delta,
            "slr_sdf_discrepancy_map":train_sdf_discrepancy,"slr_positive_training_score":positive_score,"slr_sdf_support_full":sdf_support_full,
            "slr_action_weight_full":action_weight_full,"slr_action_support_full":action_support_full,"slr_action_region_mask":action_region.to(anchor),
            "slr_positive_from_selector_fraction":positive_from_selector_fraction,
            "slr_true_hr_active":anchor.new_tensor(float(hr_image is not None)),"slr_context_size":anchor.new_tensor(float(self.context_size)),"slr_context_halo":anchor.new_tensor(float(self.context_halo)),
            "slr_positive_patch_logits":positive_patch_logits,"slr_positive_patch_probs":positive_patch_probs,"slr_positive_patch_valid":positive_patch_valid,"slr_positive_patch_target":positive_target_patch,"slr_positive_sdf_absolute":positive_sdf_abs,"slr_positive_sdf_target":positive_sdf_target,"slr_positive_sdf_support":positive_sdf_support,"slr_positive_region_mask":positive_region,
            "slr_positive_action_state_logits":positive_action_state_logits,"slr_positive_action_state_probs":positive_action_state_probs,"slr_positive_action_dose":positive_action_dose,"slr_positive_signed_action":positive_signed_action,"slr_positive_action_support":positive_action_support,"slr_positive_action_type_target":positive_action_type_target,"slr_positive_action_dose_target":positive_action_dose_target,"slr_positive_signed_action_target":positive_signed_action_target,
            "slr_positive_sdf_anchor":positive_sdf_anchor,"slr_positive_sdf_delta":positive_sdf_abs-positive_sdf_anchor,
            "slr_clean_patch_logits":clean_patch_logits,"slr_clean_patch_probs":clean_patch_probs,"slr_clean_patch_valid":clean_patch_valid,"slr_clean_patch_target":clean_target_patch,"slr_clean_patch_anchor_prob":clean_anchor_patch,"slr_clean_patch_delta_logit":clean_patch_logits-self._logit(clean_anchor_patch.clamp(EPS,1-EPS)),"slr_clean_sdf_absolute":clean_sdf_abs,"slr_clean_sdf_anchor":clean_sdf_anchor,"slr_clean_sdf_delta":clean_sdf_abs-clean_sdf_anchor,"slr_clean_sdf_target":clean_sdf_target,"slr_clean_sdf_support":clean_sdf_support,"slr_clean_region_mask":clean_region,
            "slr_clean_action_state_logits":clean_action_state_logits,"slr_clean_action_state_probs":clean_action_state_probs,"slr_clean_action_dose":clean_action_dose,"slr_clean_signed_action":clean_signed_action,"slr_clean_action_support":clean_action_support,"slr_clean_action_type_target":clean_action_type_target,"slr_clean_action_dose_target":clean_action_dose_target,"slr_clean_signed_action_target":clean_signed_action_target,
            "slr_oracle_patch_logits":positive_patch_logits,"slr_oracle_patch_probs":positive_patch_probs,"slr_oracle_patch_valid":positive_patch_valid,"slr_oracle_patch_target":positive_target_patch,"slr_oracle_sdf_delta":positive_sdf_abs-positive_sdf_anchor,"slr_oracle_sdf_target":positive_sdf_target,"slr_oracle_sdf_support":positive_sdf_support,"slr_oracle_region_mask":positive_region,"slr_patch_window":self.slr_patch_window.to(anchor),
        }
        return out


def adaptive_local_fusion(
        geometry_prob,
        local_prob,
        confidence):

    """
    Geometry provides global structure.
    Local branch reconstructs uncertain regions.
    """

    return (
        confidence *
        local_prob
        +
        (1-confidence)
        *
        geometry_prob
    )
