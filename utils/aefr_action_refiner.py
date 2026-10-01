"""Action-Evidence Factorized Refinement (AEFR).

This module is an ablation-safe successor to PC2R-v3.2.  It is intentionally
implemented as a separate refiner so the historical PC2R path remains exactly
reproducible.

Scientific contracts
--------------------
E1 (single_atomic)
    One factual correction is predicted from factual visual/semantic/text,
    anchor, uncertainty and Geometry evidence.  MC posterior samples do NOT
    generate separate actions.  The historical atomic component deployment is
    retained only to isolate the effect of action/evidence factorization.

E2 (single_continuous)
    The same single factual correction is deployed continuously inside the
    deterministic non-overlapping ROIs.  There is no hard candidate/CC/gate in
    the train/deploy path, so the deployed output has non-zero action gradient
    from step zero.

E3 (hybrid_geometry)
    The factual action is factorized by residual geometry:
      * boundary band: signed normal-coordinate displacement d, converted to a
        first-order level-set/logit transport -d*|grad z|;
      * ROI interior: anchor-adaptive logit residual
        (|z|+rho)*tanh(r), which has no fixed-confidence unreachable ceiling.
    Both actions are zero-initialized and the deployed output is continuous.

SRO-Exact (soft_ownership_exact)
    Historical soft KEEP/ADD/REMOVE ownership control retained for ablation.

IFR (intervention_factorized)
    Historical factorized residual-refinement control.

SMI (selective_minimal_intervention)
    Selective Minimal Intervention. A single calibrated hard-error posterior is used
    for both WHERE and COMMIT, so a cost-sensitive classification score is never
    interpreted as physical correction amplitude. Deployment uses a hard forward
    commit with a straight-through gradient; direction remains independently
    supervised on true residual errors. Interior correction is the deterministic
    minimum logit crossing, while boundary displacement magnitude is learned in
    pixel units with an operator-aligned target in the loss. Context ROI and action
    support are therefore explicitly separated.

E4 (posterior action stability, diagnostic-only by default)
    The *same factual action* is applied to every actual MC posterior state.
    Posterior samples never create their own learned actions.  We report how
    strongly the posterior states support the factual hard change and whether
    disagreement decreases.  No GT and no stability gate enter forward.

Gradient contract
-----------------
The Base anchor remains detached by the surrounding GEOTR generator.  For A3,
``joint_geometry_grad=True`` optionally allows Final loss to backpropagate
through the Geometry anchor into the unchanged Stage-1 flow.  Evidence features
(image/text/fine/flow channels) remain detached, so this cooperative gradient
has a single interpretable path: Final -> factual Geometry prediction -> flow.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .c2r_canonical_roi_refiner import _ConvNormGELU, EPS
from .pc2r_posterior_refiner import PosteriorConsistentCanonicalROIRefiner


class ActionEvidenceFactorizedROIRefiner(PosteriorConsistentCanonicalROIRefiner):
    VALID_STAGES = {
        "single_atomic", "single_continuous", "hybrid_geometry",
        "typed_state_taylor", "typed_state_exact", "soft_ownership_exact",
        "intervention_factorized", "selective_minimal_intervention",
    }

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
        risk_top_fraction: float = 0.20,
        require_risk_overlap: bool = True,
        raw_correction_threshold: float = 0.02,
        stage: str = "single_atomic",
        joint_geometry_grad: bool = False,
        posterior_stability: bool = True,
        boundary_radius_px: int = 5,
        boundary_max_displacement_px: float = 4.0,
        interior_crossing_margin: float = 0.50,
        transition_aware: bool = False,
        transition_logit_scale: float = 2.0,
        use_raw_flow_evidence: bool = True,
        intervention_error_prior: float = 0.03,
        intervention_edit_prior: float = 0.03,
        smi_commit_threshold: float = 0.50,
        smi_direction_confidence_threshold: float = 0.50,
    ) -> None:
        # Parent supplies deterministic ROI geometry, posterior normalization and
        # the historical atomic component routine used only by E1/diagnostics.
        super().__init__(
            hidden_dim,
            semantic_channels,
            text_dim,
            fine_feature_channels=fine_feature_channels,
            num_regions=num_regions,
            region_size=region_size,
            selection_score=selection_score,
            flow_scale_px=flow_scale_px,
            min_center_distance=min_center_distance,
            component_min_area=component_min_area,
            residual_logit_scale=residual_logit_scale,
            num_posterior_views=3,
            risk_top_fraction=risk_top_fraction,
            require_risk_overlap=require_risk_overlap,
            direction_agreement_threshold=0.0,  # repeated factual action => diagnostic only
            residual_spread_threshold=1.0e9,
            residual_strength_threshold=0.0,
            raw_correction_threshold=raw_correction_threshold,
            reliance_diagnostics=False,
            canonical_coordinate=True,
        )
        self.stage = str(stage).strip().lower()
        if self.stage not in self.VALID_STAGES:
            raise ValueError(f"AEFR stage must be one of {sorted(self.VALID_STAGES)}, got {stage!r}")
        self.joint_geometry_grad = bool(joint_geometry_grad)
        self.posterior_stability = bool(posterior_stability)
        self.boundary_radius_px = int(boundary_radius_px)
        self.boundary_max_displacement_px = float(boundary_max_displacement_px)
        self.interior_crossing_margin = float(interior_crossing_margin)
        self.transition_aware = bool(transition_aware)
        self.transition_logit_scale = float(transition_logit_scale)
        self.use_raw_flow_evidence = bool(use_raw_flow_evidence)
        self.intervention_error_prior = float(intervention_error_prior)
        self.intervention_edit_prior = float(intervention_edit_prior)
        self.smi_commit_threshold = float(smi_commit_threshold)
        self.smi_direction_confidence_threshold = float(smi_direction_confidence_threshold)
        if self.boundary_radius_px < 1:
            raise ValueError("GEOTR_AEFR_BOUNDARY_RADIUS_PX must be >= 1")
        if self.boundary_max_displacement_px <= 0.0:
            raise ValueError("GEOTR_AEFR_BOUNDARY_MAX_DISPLACEMENT_PX must be > 0")
        if self.interior_crossing_margin <= 0.0:
            raise ValueError("GEOTR_AEFR_INTERIOR_CROSSING_MARGIN must be > 0")
        if self.transition_logit_scale <= 0.0:
            raise ValueError("GEOTR_AEFR_TRANSITION_LOGIT_SCALE must be > 0")
        if not (0.0 < self.intervention_error_prior < 1.0):
            raise ValueError("GEOTR_AEFR_INTERVENTION_ERROR_PRIOR must be in (0,1)")
        if not (0.0 < self.intervention_edit_prior < 1.0):
            raise ValueError("GEOTR_AEFR_INTERVENTION_EDIT_PRIOR must be in (0,1)")
        if not (0.0 < self.smi_commit_threshold < 1.0):
            raise ValueError("GEOTR_AEFR_SMI_COMMIT_THRESHOLD must be in (0,1)")
        if not (0.0 <= self.smi_direction_confidence_threshold < 1.0):
            raise ValueError("GEOTR_AEFR_SMI_DIRECTION_CONFIDENCE_THRESHOLD must be in [0,1)")

        # Single factual action trunk.  Posterior samples are deliberately absent
        # from this input: Action and Evidence are factorized by construction.
        # Input = common visual H + anchor P + margin + MC std + flow(dx,dy,mag)
        #         + ROI-relative (x,y).
        decoder_in = hidden_dim + 1 + 1 + 1 + 3 + 2
        self.region_decoder = nn.Sequential(
            _ConvNormGELU(decoder_in, hidden_dim, 3, dilation=1),
            _ConvNormGELU(hidden_dim, hidden_dim, 3, dilation=2),
            _ConvNormGELU(hidden_dim, hidden_dim, 3, dilation=1),
        )
        self.region_residual_out = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        nn.init.zeros_(self.region_residual_out.weight)
        nn.init.zeros_(self.region_residual_out.bias)

        # E3 heads.  Zero init makes the factual Geometry anchor an exact
        # fixed-point at initialization while preserving non-zero first-order
        # gradients wrt both action heads.
        self.boundary_displacement_out = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.interior_residual_out = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        nn.init.zeros_(self.boundary_displacement_out.weight)
        nn.init.zeros_(self.boundary_displacement_out.bias)
        nn.init.zeros_(self.interior_residual_out.weight)
        nn.init.zeros_(self.interior_residual_out.bias)

        # R1/R2 typed residual-state heads.  The five states are not a learned
        # safety gate; they are the supervised residual action alphabet itself:
        #   0 KEEP
        #   1 BOUNDARY_ADD
        #   2 BOUNDARY_REMOVE
        #   3 INTERIOR_ADD
        #   4 INTERIOR_REMOVE
        # A zero-initialized state head gives a uniform distribution, therefore
        # q(B_ADD)-q(B_REMOVE) == q(I_ADD)-q(I_REMOVE) == 0 and the deployed
        # operator is exact anchor identity at initialization.
        self.typed_state_enabled = self.stage in {"typed_state_taylor", "typed_state_exact"}
        if self.typed_state_enabled:
            self.residual_state_out = nn.Conv2d(hidden_dim, 5, kernel_size=1)
            nn.init.zeros_(self.residual_state_out.weight)
            nn.init.zeros_(self.residual_state_out.bias)
            self.boundary_magnitude_out = nn.Conv2d(hidden_dim, 1, kernel_size=1)
            nn.init.zeros_(self.boundary_magnitude_out.weight)
            # Start from one quarter of d_max once a non-zero state action is
            # learned.  Identity does not depend on this bias because signed
            # state action is exactly zero at initialization.
            init_mag_frac = 0.25
            init_mag_logit = float(torch.logit(torch.tensor(init_mag_frac)).item())
            nn.init.constant_(self.boundary_magnitude_out.bias, init_mag_logit)
        else:
            self.residual_state_out = None
            self.boundary_magnitude_out = None

        # SRO-Exact: soft KEEP/ADD/REMOVE residual ownership.  Boundary vs
        # interior is *not* re-classified by the network because the anchor
        # geometry already deterministically supplies that routing variable.
        # Zero logits imply q_add == q_remove, hence signed action is exactly
        # zero and the deployed operator is an exact anchor fixed point.
        self.soft_ownership_enabled = self.stage == "soft_ownership_exact"
        if self.soft_ownership_enabled:
            self.residual_ownership_out = nn.Conv2d(hidden_dim, 3, kernel_size=1)
            nn.init.zeros_(self.residual_ownership_out.weight)
            nn.init.zeros_(self.residual_ownership_out.bias)
            self.soft_boundary_magnitude_out = nn.Conv2d(hidden_dim, 1, kernel_size=1)
            nn.init.zeros_(self.soft_boundary_magnitude_out.weight)
            init_mag_frac = 0.25
            init_mag_logit = float(torch.logit(torch.tensor(init_mag_frac)).item())
            nn.init.constant_(self.soft_boundary_magnitude_out.bias, init_mag_logit)
        else:
            self.residual_ownership_out = None
            self.soft_boundary_magnitude_out = None

        # IFR: hard-intervention-aligned WHERE / WHETHER / DIRECTION / MAGNITUDE.
        # The dense localizer is a learned residual correction on top of margin
        # uncertainty.  Its final conv is zero-initialized, so the initial ROI
        # trajectory matches the historical margin selector while the target is
        # actual post-Geometry hard error.  No GT enters this forward path.
        self.intervention_enabled = self.stage in {"intervention_factorized", "selective_minimal_intervention"}
        self.smi_enabled = self.stage == "selective_minimal_intervention"
        if self.intervention_enabled:
            loc_in = self.hidden_dim + 10  # common + anchor/margin/std/dis/ent + transition(2) + flow(3)
            self.intervention_error_localizer = nn.Sequential(
                _ConvNormGELU(loc_in, self.hidden_dim, 3, dilation=1),
                _ConvNormGELU(self.hidden_dim, self.hidden_dim, 3, dilation=2),
                nn.Conv2d(self.hidden_dim, 1, kernel_size=1),
            )
            nn.init.zeros_(self.intervention_error_localizer[-1].weight)
            nn.init.zeros_(self.intervention_error_localizer[-1].bias)

            # SMI uses a proper, trainable logistic posterior.  The margin term
            # is only an *initialization of a learnable evidence coefficient*,
            # not a fixed additive prior.  At initialization margin=0 maps to
            # the declared residual prior and margin=1 maps exactly to p=0.5;
            # unweighted BCE can subsequently calibrate both bias and scale.
            if self.smi_enabled:
                prior_logit = float(torch.logit(torch.tensor(self.intervention_error_prior)).item())
                self.smi_error_bias = nn.Parameter(torch.tensor(prior_logit, dtype=torch.float32))
                self.smi_margin_scale = nn.Parameter(torch.tensor(-prior_logit, dtype=torch.float32))
            else:
                self.register_parameter("smi_error_bias", None)
                self.register_parameter("smi_margin_scale", None)

            # Additional action evidence that the historical trunk omitted:
            # MC disagreement, entropy, realized Geometry transition and the
            # learned residual-error probability.
            self.intervention_context_encoder = nn.Sequential(
                _ConvNormGELU(5, self.hidden_dim, 3, dilation=1),
                nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1, bias=False),
            )
            nn.init.zeros_(self.intervention_context_encoder[-1].weight)

            self.intervention_direction_out = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)
            self.intervention_boundary_magnitude_out = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)
            nn.init.zeros_(self.intervention_direction_out.weight)
            nn.init.zeros_(self.intervention_direction_out.bias)
            nn.init.zeros_(self.intervention_boundary_magnitude_out.weight)
            init_mag_logit = float(torch.logit(torch.tensor(0.25)).item())
            nn.init.constant_(self.intervention_boundary_magnitude_out.bias, init_mag_logit)

            if self.smi_enabled:
                # SMI has no duplicate editness head and no free interior-dose
                # head by construction. Their roles are respectively the
                # calibrated error posterior and deterministic minimum crossing.
                self.intervention_edit_out = None
                self.intervention_interior_magnitude_out = None
            else:
                self.intervention_edit_out = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)
                self.intervention_interior_magnitude_out = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)
                nn.init.zeros_(self.intervention_edit_out.weight)
                nn.init.zeros_(self.intervention_edit_out.bias)
                nn.init.constant_(
                    self.intervention_edit_out.bias,
                    float(torch.logit(torch.tensor(self.intervention_edit_prior)).item()),
                )
                nn.init.zeros_(self.intervention_interior_magnitude_out.weight)
                nn.init.constant_(self.intervention_interior_magnitude_out.bias, init_mag_logit)
        else:
            self.intervention_error_localizer = None
            self.intervention_context_encoder = None
            self.intervention_edit_out = None
            self.register_parameter("smi_error_bias", None)
            self.register_parameter("smi_margin_scale", None)
            self.intervention_direction_out = None
            self.intervention_boundary_magnitude_out = None
            self.intervention_interior_magnitude_out = None

        # Transition-aware adapter (T1/T2).  It is deliberately residual and
        # zero-initialized so enabling the branch changes neither the historical
        # E3 output nor the A2 zero-transition control at initialization.  The
        # adapter is multiplied by realized-transition activity, hence A2
        # (anchor == Base) remains an exact no-op throughout training.
        if self.transition_aware:
            transition_hidden = max(16, self.hidden_dim // 2)
            self.transition_encoder = nn.Sequential(
                _ConvNormGELU(2, transition_hidden, 3, dilation=1),
                nn.Conv2d(transition_hidden, self.hidden_dim, kernel_size=1, bias=False),
            )
            nn.init.zeros_(self.transition_encoder[-1].weight)
        else:
            self.transition_encoder = None

    def _flow_evidence_aefr(self, flow_px: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        # Evidence is detached even in joint-Geometry mode.  The only M2->M1
        # gradient path is through the factual Geometry prediction itself.
        flow = flow_px.detach().to(anchor)
        if tuple(flow.shape[-2:]) != tuple(anchor.shape[-2:]):
            flow = self._resize(flow, tuple(anchor.shape[-2:]))
        flow = (flow / self.flow_scale_px).clamp(-1.0, 1.0)
        # flow is detached evidence; use an exact Euclidean norm so the A2
        # zero-flow control is exactly zero (the historical +1e-12 under sqrt
        # created an artificial 1e-6 magnitude channel).
        mag = torch.sqrt(flow[:, 0:1].square() + flow[:, 1:2].square()).clamp(0.0, 1.0)
        evidence = torch.cat([flow[:, 0:1], flow[:, 1:2], mag], dim=1)
        if not self.use_raw_flow_evidence:
            evidence = torch.zeros_like(evidence)
        return evidence

    def _transition_evidence(
        self,
        anchor_evidence: torch.Tensor,
        base: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode the *realized* upstream Geometry transition.

        ``flow_px`` is an action parameter, not the actual prediction change.
        The downstream refiner therefore receives the factual state transition
        in logit coordinates: ΔG = logit(P_anchor) - logit(P_base).

        Returns
        -------
        transition_input : [B,2,H,W]
            [tanh(ΔG/s), tanh(|ΔG|/s)].
        activity : [B,1,H,W]
            Exact zero when anchor == Base; used to guarantee the A2 no-op.
        delta_logit : [B,1,H,W]
            Raw detached realized transition for diagnostics.
        flip_mask : [B,1,H,W]
            Hard class change induced by the upstream Geometry stage.
        """
        za = torch.logit(anchor_evidence.detach().clamp(EPS, 1.0 - EPS))
        zb = torch.logit(base.detach().clamp(EPS, 1.0 - EPS))
        delta = za - zb
        signed = torch.tanh(delta / self.transition_logit_scale)
        activity = torch.tanh(delta.abs() / self.transition_logit_scale)
        transition_input = torch.cat([signed, activity], dim=1)
        with torch.no_grad():
            flip = ((anchor_evidence >= 0.5) != (base >= 0.5)).to(anchor_evidence)
        return transition_input, activity, delta, flip

    @staticmethod
    def _logit_gradient_components(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Central-difference derivative in logit / pixel.  Replicate padding
        # avoids artificial zero borders and keeps dimensions exact.
        kx = z.new_tensor([[0.0, 0.0, 0.0], [-0.5, 0.0, 0.5], [0.0, 0.0, 0.0]]).view(1, 1, 3, 3)
        ky = z.new_tensor([[0.0, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]]).view(1, 1, 3, 3)
        zp = F.pad(z, (1, 1, 1, 1), mode="replicate")
        return F.conv2d(zp, kx), F.conv2d(zp, ky)

    @classmethod
    def _logit_gradient_magnitude(cls, z: torch.Tensor) -> torch.Tensor:
        gx, gy = cls._logit_gradient_components(z)
        return torch.sqrt(gx.square() + gy.square() + 1.0e-12)

    @staticmethod
    def _warp_logits_px(logits: torch.Tensor, flow_px: torch.Tensor) -> torch.Tensor:
        """Exact backward warp using the same pixel convention as Stage-1.

        Output at destination x samples the factual anchor at x + flow(x).
        This deliberately matches MultiHypothesisCompositionalSegmenter._warp_logits
        so global and local geometry use one operator family rather than a
        Taylor surrogate in the second stage.
        """
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError(f"logits must be [B,1,H,W], got {tuple(logits.shape)}")
        if flow_px.ndim != 4 or flow_px.shape[1] != 2:
            raise ValueError(f"flow_px must be [B,2,H,W], got {tuple(flow_px.shape)}")
        _, _, h, w = logits.shape
        dtype, device = logits.dtype, logits.device
        yy = torch.arange(h, device=device, dtype=dtype) + 0.5
        xx = torch.arange(w, device=device, dtype=dtype) + 0.5
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        x = gx[None] + flow_px[:, 0]
        y = gy[None] + flow_px[:, 1]
        x_norm = 2.0 * x / float(w) - 1.0
        y_norm = 2.0 * y / float(h) - 1.0
        grid = torch.stack([x_norm, y_norm], dim=-1)
        return F.grid_sample(
            logits, grid, mode="bilinear", padding_mode="border", align_corners=False
        )

    def _boundary_band(self, anchor: torch.Tensor) -> torch.Tensor:
        # Deterministic action partition only; it is not a learned gate.
        with torch.no_grad():
            hard = (anchor >= 0.5).float()
            dil = F.max_pool2d(hard, 3, stride=1, padding=1)
            ero = -F.max_pool2d(-hard, 3, stride=1, padding=1)
            boundary = (dil != ero).float()
            r = self.boundary_radius_px
            band = F.max_pool2d(boundary, 2 * r + 1, stride=1, padding=r) > 0.5
        return band

    def _decode_trunk(
        self,
        common_p: torch.Tensor,
        anchor_p: torch.Tensor,
        margin_p: torch.Tensor,
        mc_std_p: torch.Tensor,
        flow_p: torch.Tensor,
        rel: torch.Tensor,
    ) -> torch.Tensor:
        inp = torch.cat([common_p, anchor_p, margin_p, mc_std_p, flow_p, rel], dim=2)
        b, k, c, r, _ = inp.shape
        return self.region_decoder(inp.reshape(b * k, c, r, r)).view(b, k, -1, r, r)

    @staticmethod
    def _pairwise_posterior_diversity(posterior_p: torch.Tensor, valid_patch: torch.Tensor) -> torch.Tensor:
        # posterior_p [B,K,S,1,R,R]
        b, k, s = posterior_p.shape[:3]
        if s < 2:
            return posterior_p.new_zeros(())
        valid = valid_patch[:, :, 0].to(posterior_p)  # [B,K,R,R]
        den = valid.sum().clamp_min(1.0)
        total = posterior_p.new_zeros(())
        n = 0
        p = posterior_p[:, :, :, 0]
        for i in range(s):
            for j in range(i + 1, s):
                total = total + ((p[:, :, i] - p[:, :, j]).abs() * valid).sum() / den
                n += 1
        return total / max(n, 1)

    def _posterior_action_probe(
        self,
        posterior_p: Optional[torch.Tensor],  # [B,K,S,1,R,R]
        anchor_p: torch.Tensor,
        final_patch: torch.Tensor,            # [B,K,1,R,R]
        valid_patch: torch.Tensor,
        *,
        delta_patch: Optional[torch.Tensor] = None,
        displacement_patch: Optional[torch.Tensor] = None,
        interior_raw_patch: Optional[torch.Tensor] = None,
        boundary_patch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        z = anchor_p.new_zeros(())
        if (not self.posterior_stability) or posterior_p is None or self.typed_state_enabled:
            return {
                "stability_support": z, "stability_improvement": z,
                "disagreement_pre": z, "disagreement_post": z,
                "action_support_fraction": z, "posterior_diversity": z,
                "posterior_center_bias_abs": z, "posterior_center_bias_signed": z,
            }
        with torch.no_grad():
            pp = posterior_p[:, :, :, 0].clamp(EPS, 1.0 - EPS)  # [B,K,S,R,R]
            zp = torch.logit(pp)
            za = torch.logit(anchor_p[:, :, 0].clamp(EPS, 1.0 - EPS))
            valid = valid_patch[:, :, 0].bool()
            factual_h = final_patch[:, :, 0] >= 0.5
            anchor_h = anchor_p[:, :, 0] >= 0.5
            action_support = (factual_h != anchor_h) & valid

            if self.stage in {"single_atomic", "single_continuous"}:
                assert delta_patch is not None
                post_z = zp + delta_patch[:, :, 0, None]
            else:
                assert displacement_patch is not None and interior_raw_patch is not None and boundary_patch is not None
                bs = boundary_patch[:, :, 0].to(zp)[:, :, None]
                # Same factual action parameters are probed in every posterior
                # state; state-specific gradients only convert displacement to
                # the local logit coordinate.
                bs_flat = zp.reshape(-1, 1, zp.shape[-2], zp.shape[-1])
                gm = self._logit_gradient_magnitude(bs_flat).reshape_as(zp)
                d = displacement_patch[:, :, 0, None]
                raw_i = interior_raw_patch[:, :, 0, None]
                scale_i = zp.abs() + self.interior_crossing_margin
                post_z = zp - bs * d * gm + (1.0 - bs) * scale_i * torch.tanh(raw_i)

            pre_h = pp >= 0.5
            post_h = post_z >= 0.0
            target_h = factual_h[:, :, None]
            support_pre = (pre_h == target_h).float().mean(dim=2)
            support_post = (post_h == target_h).float().mean(dim=2)
            fg_pre = pre_h.float().mean(dim=2)
            fg_post = post_h.float().mean(dim=2)
            dis_pre = 4.0 * fg_pre * (1.0 - fg_pre)
            dis_post = 4.0 * fg_post * (1.0 - fg_post)
            m = action_support.to(support_post)
            den = m.sum().clamp_min(1.0)

            all_mean_logit = zp.mean(dim=2)
            bias = all_mean_logit - za
            v = valid.to(bias)
            vden = v.sum().clamp_min(1.0)
            return {
                "stability_support": (support_post * m).sum() / den,
                "stability_improvement": ((support_post - support_pre) * m).sum() / den,
                "disagreement_pre": (dis_pre * m).sum() / den,
                "disagreement_post": (dis_post * m).sum() / den,
                "action_support_fraction": m.sum() / v.sum().clamp_min(1.0),
                "posterior_diversity": self._pairwise_posterior_diversity(posterior_p, valid_patch),
                "posterior_center_bias_abs": (bias.abs() * v).sum() / vden,
                "posterior_center_bias_signed": (bias * v).sum() / vden,
            }

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
        anchor_in = anchor_prob[:, None] if anchor_prob.ndim == 3 else anchor_prob
        anchor = anchor_in if self.joint_geometry_grad else anchor_in.detach()
        anchor = anchor.clamp(EPS, 1.0 - EPS)
        if base_prob is None:
            base_prob = anchor.detach()
        base = base_prob[:, None] if base_prob.ndim == 3 else base_prob
        base = base.detach().clamp(EPS, 1.0 - EPS)
        if flow_px is None:
            flow_px = anchor.new_zeros((anchor.shape[0], 2, *anchor.shape[-2:]))
        flow = flow_px.to(anchor)

        posterior = None
        if isinstance(posterior_probability_samples, torch.Tensor):
            posterior = self._normalize_samples(posterior_probability_samples, anchor.detach())
        elif self.posterior_stability:
            raise RuntimeError("AEFR posterior-stability diagnostics require actual mc_probability_samples")

        b, _, h, w = anchor.shape
        # Evidence channels are detached.  In joint-Geometry mode the Final->M1
        # gradient is intentionally restricted to application of the factual
        # action on the Geometry anchor, not through the action-prediction inputs.
        anchor_evidence = anchor.detach()
        transition_input, transition_activity, transition_delta_logit, transition_flip = self._transition_evidence(
            anchor_evidence, base
        )
        margin = self._margin_uncertainty(anchor_evidence)
        mc_std = self._fit_evidence(mc_std_map, anchor_evidence, scale=2.0)
        mc_dis = self._fit_evidence(mc_disagreement_map, anchor_evidence, scale=1.0)
        entropy = self._entropy(anchor_evidence)
        common = self._common_visual(image, semantic_map, text_features, fine_feature_map, (h, w))
        flow_evidence = self._flow_evidence_aefr(flow, anchor.detach())

        # Residual-error posterior used for WHERE. Historical IFR retains its
        # fixed uncertainty prior for exact reproducibility. SMI instead uses a
        # proper learnable logistic posterior; margin is an ordinary learnable
        # evidence term, so the deployed commit probability is not contaminated
        # by a permanent hand-written uncertainty posterior.
        intervention_error_prior = (
            self.intervention_error_prior
            + (1.0 - 2.0 * self.intervention_error_prior) * margin.detach()
        ).clamp(EPS, 1.0 - EPS)
        intervention_error_logit = torch.logit(intervention_error_prior)
        intervention_error_prob = intervention_error_prior
        if self.intervention_enabled:
            assert self.intervention_error_localizer is not None
            loc_features = torch.cat([
                common, anchor_evidence, margin.detach(), mc_std, mc_dis, entropy.detach(),
                transition_input, flow_evidence,
            ], dim=1)
            residual_error_logit = self.intervention_error_localizer(loc_features)
            if self.smi_enabled:
                assert self.smi_error_bias is not None and self.smi_margin_scale is not None
                calibrated_prior_logit = (
                    self.smi_error_bias.to(anchor)
                    + self.smi_margin_scale.to(anchor) * margin.detach()
                )
                intervention_error_prior = torch.sigmoid(calibrated_prior_logit).clamp(EPS, 1.0 - EPS)
                intervention_error_logit = calibrated_prior_logit + residual_error_logit
            else:
                intervention_error_logit = torch.logit(
                    intervention_error_prior.clamp(EPS, 1.0 - EPS)
                ) + residual_error_logit
            intervention_error_prob = torch.sigmoid(intervention_error_logit)
            score = intervention_error_prob.detach()
        else:
            score = self._selection_score({
                "margin": margin.detach(), "mc_std": mc_std,
                "mc_disagreement": mc_dis, "entropy": entropy.detach(),
            })
        centers, center_yx, center_valid = self._greedy_centers(score)
        ay, ax, grid, valid_patch, rel = self._patch_geometry(center_yx, center_valid, h, w, anchor.dtype)
        k = self.num_regions
        common_p = self._crop(common, grid, k)
        anchor_p = self._crop(anchor, grid, k)
        anchor_evidence_p = self._crop(anchor_evidence, grid, k)
        margin_p = self._crop(margin, grid, k)
        mc_std_p = self._crop(mc_std, grid, k)
        mc_dis_p = self._crop(mc_dis, grid, k)
        entropy_p = self._crop(entropy, grid, k)
        error_prob_p = self._crop(intervention_error_prob, grid, k)
        flow_p = self._crop(flow_evidence, grid, k)
        transition_p = self._crop(transition_input, grid, k)
        transition_activity_p = self._crop(transition_activity, grid, k)
        posterior_p = self._crop_posterior_samples(posterior, grid, k) if posterior is not None else None

        hidden = self._decode_trunk(common_p, anchor_evidence_p, margin_p, mc_std_p, flow_p, rel)
        if self.transition_encoder is not None:
            tp = transition_p.reshape(b * k, 2, self.region_size, self.region_size)
            ap = transition_activity_p.reshape(b * k, 1, self.region_size, self.region_size)
            transition_hidden = self.transition_encoder(tp) * ap
            hidden = hidden + transition_hidden.view(b, k, -1, self.region_size, self.region_size)
        if self.intervention_enabled:
            assert self.intervention_context_encoder is not None
            extra = torch.cat([mc_dis_p, entropy_p, transition_p, error_prob_p], dim=2)
            eb, ek, ec, er, _ = extra.shape
            extra_h = self.intervention_context_encoder(extra.reshape(eb * ek, ec, er, er))
            hidden = hidden + extra_h.view(eb, ek, -1, er, er)
        bk = b * k
        r = self.region_size
        hidden_flat = hidden.reshape(bk, hidden.shape[2], r, r)
        z0_patch = torch.logit(anchor_p.clamp(EPS, 1.0 - EPS))

        displacement_patch = None
        interior_raw_patch = None
        boundary_patch = None
        state_logits_full = None
        state_probs_full = None
        state_logits_patch = None
        ownership_logits_full = None
        ownership_probs_full = None
        ownership_logits_patch = None
        intervention_edit_logit_full = torch.zeros_like(anchor)
        intervention_edit_prob_full = torch.zeros_like(anchor)
        intervention_direction_logit_full = torch.zeros_like(anchor)
        intervention_direction_prob_full = torch.zeros_like(anchor)
        intervention_interior_magnitude_full = torch.zeros_like(anchor)
        intervention_signed_action_full = torch.zeros_like(anchor)
        intervention_commit_full = torch.zeros_like(anchor, dtype=torch.bool)
        boundary_magnitude_full = torch.zeros_like(anchor)
        signed_boundary_full = torch.zeros_like(anchor)
        signed_interior_full = torch.zeros_like(anchor)

        if self.stage in {"single_atomic", "single_continuous"}:
            raw = self.region_residual_out(hidden_flat).view(b, k, 1, r, r)
            delta_patch = self.residual_logit_scale * torch.tanh(raw)
            action_logits_patch = z0_patch + delta_patch
            # Scatter ACTION, not cropped anchor probability.
            delta_full, region, overlap = self._scatter_patch_values(
                delta_patch, torch.zeros_like(anchor), ay, ax, valid_patch
            )
            z_anchor_full = torch.logit(anchor.clamp(EPS, 1.0 - EPS))
            full_proposal = torch.sigmoid(z_anchor_full + delta_full)
            full_baseline = torch.sigmoid(z_anchor_full)
            canonical_prob = (anchor + (full_proposal - full_baseline)).clamp(EPS, 1.0 - EPS)
            canonical_patch = self._crop(canonical_prob, grid, k)

        elif self.stage == "hybrid_geometry":
            raw_d = self.boundary_displacement_out(hidden_flat).view(b, k, 1, r, r)
            interior_raw_patch = self.interior_residual_out(hidden_flat).view(b, k, 1, r, r)
            displacement_patch = self.boundary_max_displacement_px * torch.tanh(raw_d)
            boundary_full = self._boundary_band(anchor.detach())
            boundary_patch = self._crop(boundary_full.to(anchor), grid, k) > 0.5
            grad_mag_patch = self._logit_gradient_magnitude(
                z0_patch.reshape(bk, 1, r, r)
            ).view(b, k, 1, r, r)
            interior_scale = z0_patch.detach().abs() + self.interior_crossing_margin
            boundary_delta = -boundary_patch.to(anchor) * displacement_patch * grad_mag_patch
            interior_delta = (~boundary_patch).to(anchor) * interior_scale * torch.tanh(interior_raw_patch)
            delta_patch = boundary_delta + interior_delta
            delta_full, region, overlap = self._scatter_patch_values(
                delta_patch, torch.zeros_like(anchor), ay, ax, valid_patch
            )
            z_anchor_full = torch.logit(anchor.clamp(EPS, 1.0 - EPS))
            full_proposal = torch.sigmoid(z_anchor_full + delta_full)
            full_baseline = torch.sigmoid(z_anchor_full)
            canonical_prob = (anchor + (full_proposal - full_baseline)).clamp(EPS, 1.0 - EPS)
            canonical_patch = self._crop(canonical_prob, grid, k)

        elif self.typed_state_enabled:
            # Historical R1/R2 five-state controls are kept exactly reproducible.
            assert self.residual_state_out is not None and self.boundary_magnitude_out is not None
            state_logits_patch = self.residual_state_out(hidden_flat).view(b, k, 5, r, r)
            state_probs_patch = torch.softmax(state_logits_patch, dim=2)
            q_badd = state_probs_patch[:, :, 1:2]
            q_bremove = state_probs_patch[:, :, 2:3]
            q_iadd = state_probs_patch[:, :, 3:4]
            q_iremove = state_probs_patch[:, :, 4:5]
            signed_boundary_patch = q_badd - q_bremove
            signed_interior_patch = q_iadd - q_iremove
            mag_raw_patch = self.boundary_magnitude_out(hidden_flat).view(b, k, 1, r, r)
            magnitude_patch = self.boundary_max_displacement_px * torch.sigmoid(mag_raw_patch)

            zero_anchor = torch.zeros_like(anchor)
            state_logits_full, region, overlap = self._scatter_patch_values(
                state_logits_patch, zero_anchor, ay, ax, valid_patch
            )
            state_probs_full = torch.softmax(state_logits_full, dim=1)
            signed_boundary_full, _, _ = self._scatter_patch_values(
                signed_boundary_patch, zero_anchor, ay, ax, valid_patch
            )
            signed_interior_full, _, _ = self._scatter_patch_values(
                signed_interior_patch, zero_anchor, ay, ax, valid_patch
            )
            boundary_magnitude_full, _, _ = self._scatter_patch_values(
                magnitude_patch, zero_anchor, ay, ax, valid_patch
            )

            z_anchor_full = torch.logit(anchor.clamp(EPS, 1.0 - EPS))
            boundary_full = self._boundary_band(anchor.detach())
            bnd_mask = boundary_full & region
            int_mask = (~boundary_full) & region
            signed_disp_full = bnd_mask.to(anchor) * signed_boundary_full * boundary_magnitude_full
            gx, gy = self._logit_gradient_components(z_anchor_full)
            grad_norm = torch.sqrt(gx.square() + gy.square() + 1.0e-12)
            nx = gx / grad_norm
            ny = gy / grad_norm

            if self.stage == "typed_state_taylor":
                boundary_delta_full = bnd_mask.to(anchor) * signed_disp_full * grad_norm
            else:
                residual_flow = torch.cat([signed_disp_full * nx, signed_disp_full * ny], dim=1)
                warped_logits = self._warp_logits_px(z_anchor_full, residual_flow)
                nonzero_flow = signed_disp_full.abs() > 0.0
                exact_warp_delta = torch.where(
                    nonzero_flow, warped_logits - z_anchor_full, torch.zeros_like(z_anchor_full)
                )
                boundary_delta_full = bnd_mask.to(anchor) * exact_warp_delta

            interior_scale_full = z_anchor_full.detach().abs() + self.interior_crossing_margin
            interior_delta_full = int_mask.to(anchor) * signed_interior_full * interior_scale_full
            delta_full = boundary_delta_full + interior_delta_full
            full_proposal = torch.sigmoid(z_anchor_full + delta_full)
            full_baseline = torch.sigmoid(z_anchor_full)
            canonical_prob = (anchor + (full_proposal - full_baseline)).clamp(EPS, 1.0 - EPS)
            canonical_patch = self._crop(canonical_prob, grid, k)
            delta_patch = self._crop(delta_full, grid, k)
            displacement_patch = self._crop(signed_disp_full, grid, k)
            boundary_patch = self._crop(bnd_mask.to(anchor), grid, k) > 0.5

        elif self.soft_ownership_enabled:
            # Historical SRO-Exact control.
            assert self.soft_ownership_enabled
            assert self.residual_ownership_out is not None and self.soft_boundary_magnitude_out is not None
            ownership_logits_patch = self.residual_ownership_out(hidden_flat).view(b, k, 3, r, r)
            ownership_probs_patch = torch.softmax(ownership_logits_patch, dim=2)
            q_add = ownership_probs_patch[:, :, 1:2]
            q_remove = ownership_probs_patch[:, :, 2:3]
            signed_action_patch = q_add - q_remove
            mag_raw_patch = self.soft_boundary_magnitude_out(hidden_flat).view(b, k, 1, r, r)
            magnitude_patch = self.boundary_max_displacement_px * torch.sigmoid(mag_raw_patch)

            zero_anchor = torch.zeros_like(anchor)
            ownership_logits_full, region, overlap = self._scatter_patch_values(
                ownership_logits_patch, zero_anchor, ay, ax, valid_patch
            )
            ownership_probs_full = torch.softmax(ownership_logits_full, dim=1)
            signed_action_full, _, _ = self._scatter_patch_values(
                signed_action_patch, zero_anchor, ay, ax, valid_patch
            )
            boundary_magnitude_full, _, _ = self._scatter_patch_values(
                magnitude_patch, zero_anchor, ay, ax, valid_patch
            )

            z_anchor_full = torch.logit(anchor.clamp(EPS, 1.0 - EPS))
            boundary_full = self._boundary_band(anchor.detach())
            bnd_mask = boundary_full & region
            int_mask = (~boundary_full) & region
            signed_boundary_full = bnd_mask.to(anchor) * signed_action_full
            signed_interior_full = int_mask.to(anchor) * signed_action_full
            signed_disp_full = signed_boundary_full * boundary_magnitude_full

            gx, gy = self._logit_gradient_components(z_anchor_full)
            grad_norm = torch.sqrt(gx.square() + gy.square() + 1.0e-12)
            nx = gx / grad_norm
            ny = gy / grad_norm
            residual_flow = torch.cat([signed_disp_full * nx, signed_disp_full * ny], dim=1)
            warped_logits = self._warp_logits_px(z_anchor_full, residual_flow)
            nonzero_flow = signed_disp_full.abs() > 0.0
            exact_warp_delta = torch.where(
                nonzero_flow, warped_logits - z_anchor_full, torch.zeros_like(z_anchor_full)
            )
            boundary_delta_full = bnd_mask.to(anchor) * exact_warp_delta

            interior_scale_full = z_anchor_full.detach().abs() + self.interior_crossing_margin
            interior_delta_full = signed_interior_full * interior_scale_full
            delta_full = boundary_delta_full + interior_delta_full
            full_proposal = torch.sigmoid(z_anchor_full + delta_full)
            full_baseline = torch.sigmoid(z_anchor_full)
            canonical_prob = (anchor + (full_proposal - full_baseline)).clamp(EPS, 1.0 - EPS)
            canonical_patch = self._crop(canonical_prob, grid, k)
            delta_patch = self._crop(delta_full, grid, k)
            displacement_patch = self._crop(signed_disp_full, grid, k)
            boundary_patch = self._crop(bnd_mask.to(anchor), grid, k) > 0.5

        elif self.smi_enabled:
            # SMI: one calibrated residual-error posterior performs BOTH WHERE
            # and COMMIT.  Forward deployment is hard/selective; the straight-
            # through value supplies a localizer gradient without interpreting
            # confidence as correction dose.  Direction is independent.
            assert self.intervention_enabled
            assert self.intervention_direction_out is not None
            assert self.intervention_boundary_magnitude_out is not None
            direction_logit_patch = self.intervention_direction_out(hidden_flat).view(b, k, 1, r, r)
            direction_prob_patch = torch.sigmoid(direction_logit_patch)
            signed_direction_soft = 2.0 * direction_prob_patch - 1.0
            direction_confidence_patch = signed_direction_soft.abs()
            direction_ready_patch = direction_confidence_patch > self.smi_direction_confidence_threshold
            direction_hard_patch = torch.where(
                direction_prob_patch >= 0.5,
                torch.ones_like(direction_prob_patch),
                -torch.ones_like(direction_prob_patch),
            )
            # Hard sign in forward, soft derivative in backward. Direction
            # confidence never scales physical dose.
            signed_direction_patch = (
                direction_hard_patch
                + signed_direction_soft - signed_direction_soft.detach()
            )
            commit_prob_patch = error_prob_p
            commit_hard_patch = (
                (commit_prob_patch > self.smi_commit_threshold) & direction_ready_patch
            )
            commit_gate_patch = (
                commit_hard_patch.to(commit_prob_patch)
                + commit_prob_patch - commit_prob_patch.detach()
            )
            signed_action_patch = commit_gate_patch * signed_direction_patch

            bmag_raw_patch = self.intervention_boundary_magnitude_out(hidden_flat).view(b, k, 1, r, r)
            boundary_mag_patch = self.boundary_max_displacement_px * torch.sigmoid(bmag_raw_patch)

            zero_anchor = torch.zeros_like(anchor)
            intervention_edit_logit_full = intervention_error_logit
            intervention_edit_prob_full = intervention_error_prob
            intervention_direction_logit_full, region, overlap = self._scatter_patch_values(
                direction_logit_patch, zero_anchor, ay, ax, valid_patch
            )
            intervention_direction_prob_full, _, _ = self._scatter_patch_values(
                direction_prob_patch, zero_anchor, ay, ax, valid_patch
            )
            intervention_signed_action_full, _, _ = self._scatter_patch_values(
                signed_action_patch, zero_anchor, ay, ax, valid_patch
            )
            boundary_magnitude_full, _, _ = self._scatter_patch_values(
                boundary_mag_patch, zero_anchor, ay, ax, valid_patch
            )
            commit_full_float, _, _ = self._scatter_patch_values(
                commit_hard_patch.to(anchor), zero_anchor, ay, ax, valid_patch
            )
            intervention_commit_full = commit_full_float > 0.5

            z_anchor_full = torch.logit(anchor.clamp(EPS, 1.0 - EPS))
            boundary_full = self._boundary_band(anchor.detach())
            bnd_mask = boundary_full & region & intervention_commit_full
            int_mask = (~boundary_full) & region & intervention_commit_full
            signed_boundary_full = bnd_mask.to(anchor) * intervention_signed_action_full
            signed_interior_full = int_mask.to(anchor) * intervention_signed_action_full
            signed_disp_full = signed_boundary_full * boundary_magnitude_full

            gx, gy = self._logit_gradient_components(z_anchor_full)
            grad_norm = torch.sqrt(gx.square() + gy.square() + 1.0e-12)
            nx = gx / grad_norm
            ny = gy / grad_norm
            residual_flow = torch.cat([signed_disp_full * nx, signed_disp_full * ny], dim=1)
            warped_logits = self._warp_logits_px(z_anchor_full, residual_flow)
            nonzero_flow = signed_disp_full.abs() > 0.0
            exact_warp_delta = torch.where(
                nonzero_flow, warped_logits - z_anchor_full, torch.zeros_like(z_anchor_full)
            )
            boundary_delta_full = bnd_mask.to(anchor) * exact_warp_delta

            # Interior HOW-MUCH is not learned freely. The operator performs the
            # smallest configured logit crossing: |z|+rho moves a committed FN
            # or FP just across the current decision boundary.
            interior_scale_full = z_anchor_full.detach().abs() + self.interior_crossing_margin
            intervention_interior_magnitude_full = int_mask.to(anchor) * interior_scale_full
            interior_delta_full = signed_interior_full * interior_scale_full

            delta_full = boundary_delta_full + interior_delta_full
            full_proposal = torch.sigmoid(z_anchor_full + delta_full)
            full_baseline = torch.sigmoid(z_anchor_full)
            canonical_prob = (anchor + (full_proposal - full_baseline)).clamp(EPS, 1.0 - EPS)
            canonical_patch = self._crop(canonical_prob, grid, k)
            delta_patch = self._crop(delta_full, grid, k)
            displacement_patch = self._crop(signed_disp_full, grid, k)
            boundary_patch = self._crop(bnd_mask.to(anchor), grid, k) > 0.5

        else:
            # Historical IFR: WHETHER and DIRECTION are independent sigmoid heads.
            assert self.intervention_enabled
            assert self.intervention_edit_out is not None
            assert self.intervention_direction_out is not None
            assert self.intervention_boundary_magnitude_out is not None
            assert self.intervention_interior_magnitude_out is not None
            edit_logit_patch = self.intervention_edit_out(hidden_flat).view(b, k, 1, r, r)
            direction_logit_patch = self.intervention_direction_out(hidden_flat).view(b, k, 1, r, r)
            edit_prob_patch = torch.sigmoid(edit_logit_patch)
            direction_prob_patch = torch.sigmoid(direction_logit_patch)
            signed_direction_patch = 2.0 * direction_prob_patch - 1.0
            signed_action_patch = edit_prob_patch * signed_direction_patch

            bmag_raw_patch = self.intervention_boundary_magnitude_out(hidden_flat).view(b, k, 1, r, r)
            imag_raw_patch = self.intervention_interior_magnitude_out(hidden_flat).view(b, k, 1, r, r)
            boundary_mag_patch = self.boundary_max_displacement_px * torch.sigmoid(bmag_raw_patch)

            zero_anchor = torch.zeros_like(anchor)
            intervention_edit_logit_full, region, overlap = self._scatter_patch_values(
                edit_logit_patch, zero_anchor, ay, ax, valid_patch
            )
            intervention_edit_prob_full, _, _ = self._scatter_patch_values(
                edit_prob_patch, zero_anchor, ay, ax, valid_patch
            )
            intervention_direction_logit_full, _, _ = self._scatter_patch_values(
                direction_logit_patch, zero_anchor, ay, ax, valid_patch
            )
            intervention_direction_prob_full, _, _ = self._scatter_patch_values(
                direction_prob_patch, zero_anchor, ay, ax, valid_patch
            )
            intervention_signed_action_full, _, _ = self._scatter_patch_values(
                signed_action_patch, zero_anchor, ay, ax, valid_patch
            )
            intervention_commit_full = (intervention_edit_prob_full > 0.5) & region
            boundary_magnitude_full, _, _ = self._scatter_patch_values(
                boundary_mag_patch, zero_anchor, ay, ax, valid_patch
            )

            z_anchor_full = torch.logit(anchor.clamp(EPS, 1.0 - EPS))
            boundary_full = self._boundary_band(anchor.detach())
            bnd_mask = boundary_full & region
            int_mask = (~boundary_full) & region
            signed_boundary_full = bnd_mask.to(anchor) * intervention_signed_action_full
            signed_interior_full = int_mask.to(anchor) * intervention_signed_action_full
            signed_disp_full = signed_boundary_full * boundary_magnitude_full

            gx, gy = self._logit_gradient_components(z_anchor_full)
            grad_norm = torch.sqrt(gx.square() + gy.square() + 1.0e-12)
            nx = gx / grad_norm
            ny = gy / grad_norm
            residual_flow = torch.cat([signed_disp_full * nx, signed_disp_full * ny], dim=1)
            warped_logits = self._warp_logits_px(z_anchor_full, residual_flow)
            nonzero_flow = signed_disp_full.abs() > 0.0
            exact_warp_delta = torch.where(
                nonzero_flow, warped_logits - z_anchor_full, torch.zeros_like(z_anchor_full)
            )
            boundary_delta_full = bnd_mask.to(anchor) * exact_warp_delta

            # Interior magnitude is explicitly in logit units and is conditioned
            # on the factual anchor confidence.  2*sigmoid spans (0,2), so an
            # edit can cross any finite anchor logit without a fixed global cap.
            interior_mag_patch = (z0_patch.detach().abs() + self.interior_crossing_margin) * (
                2.0 * torch.sigmoid(imag_raw_patch)
            )
            intervention_interior_magnitude_full, _, _ = self._scatter_patch_values(
                interior_mag_patch, zero_anchor, ay, ax, valid_patch
            )
            interior_delta_full = signed_interior_full * intervention_interior_magnitude_full

            delta_full = boundary_delta_full + interior_delta_full
            full_proposal = torch.sigmoid(z_anchor_full + delta_full)
            full_baseline = torch.sigmoid(z_anchor_full)
            canonical_prob = (anchor + (full_proposal - full_baseline)).clamp(EPS, 1.0 - EPS)
            canonical_patch = self._crop(canonical_prob, grid, k)
            delta_patch = self._crop(delta_full, grid, k)
            displacement_patch = self._crop(signed_disp_full, grid, k)
            boundary_patch = self._crop(bnd_mask.to(anchor), grid, k) > 0.5

        # Diagnostic hard-change support.  E2/E3 do not use this support in the
        # deployed forward; it exists only for native correction accounting.
        with torch.no_grad():
            candidate_patch = ((canonical_patch >= 0.5) != (anchor_p >= 0.5)) & valid_patch.bool()
        delta_repeat = delta_patch[:, :, 0][:, :, None].expand(-1, -1, 3, -1, -1)
        commit_patch, comp_stats = self._component_commit_v3(
            canonical_patch, anchor_p, delta_repeat, valid_patch, margin_p.detach()
        )

        def scatter_mask(patch_mask: torch.Tensor) -> torch.Tensor:
            full, _, _ = self._scatter_patch_values(
                patch_mask.to(anchor), torch.zeros_like(anchor), ay, ax, valid_patch
            )
            return full[:, 0:1] > 0.5

        candidate_full = scatter_mask(candidate_patch)
        atomic_commit_full = scatter_mask(commit_patch)
        area_full = scatter_mask(comp_stats["area_stage_mask"])
        risk_full = scatter_mask(comp_stats["risk_stage_mask"])
        direction_full = scatter_mask(comp_stats["direction_stage_mask"])
        spread_full = scatter_mask(comp_stats["spread_stage_mask"])
        strength_full = scatter_mask(comp_stats["strength_stage_mask"])
        raw_full = scatter_mask(comp_stats["raw_correction_mask"])
        risk_support_full = scatter_mask(comp_stats["risk_support_mask"])

        if self.stage == "single_atomic":
            final_prob = torch.where(atomic_commit_full, canonical_prob, anchor).clamp(EPS, 1.0 - EPS)
            deploy_support = atomic_commit_full
        elif self.smi_enabled:
            # SMI explicitly separates context ROI from executable support.
            final_prob = torch.where(intervention_commit_full, canonical_prob, anchor).clamp(EPS, 1.0 - EPS)
            deploy_support = intervention_commit_full
        else:
            # Historical continuous deploy stages edit every selected ROI pixel.
            final_prob = torch.where(region, canonical_prob, anchor).clamp(EPS, 1.0 - EPS)
            deploy_support = region
        final_logits = torch.logit(final_prob)
        delta_logit = final_logits - torch.logit(anchor)

        # E4 action-stability probe uses all posterior states and the SAME action.
        probe = self._posterior_action_probe(
            posterior_p, anchor_p, canonical_patch, valid_patch,
            delta_patch=delta_patch,
            displacement_patch=displacement_patch,
            interior_raw_patch=interior_raw_patch,
            boundary_patch=boundary_patch,
        )

        # Full-image hybrid/typed diagnostics.
        if self.stage == "hybrid_geometry":
            disp_full, _, _ = self._scatter_patch_values(
                displacement_patch, torch.zeros_like(anchor), ay, ax, valid_patch
            )
            int_delta_patch = (~boundary_patch).to(anchor) * (z0_patch.detach().abs() + self.interior_crossing_margin) * torch.tanh(interior_raw_patch)
            int_full, _, _ = self._scatter_patch_values(
                int_delta_patch, torch.zeros_like(anchor), ay, ax, valid_patch
            )
            bnd_full = self._boundary_band(anchor.detach()) & region
        elif self.typed_state_enabled or self.soft_ownership_enabled or self.intervention_enabled:
            disp_full = signed_boundary_full * boundary_magnitude_full
            bnd_full = self._boundary_band(anchor.detach()) & region
            int_full = ((~bnd_full) & region).to(anchor) * signed_interior_full * (
                torch.logit(anchor.detach().clamp(EPS, 1.0 - EPS)).abs() + self.interior_crossing_margin
            )
        else:
            disp_full = torch.zeros_like(anchor)
            int_full = torch.zeros_like(anchor)
            bnd_full = torch.zeros_like(region)

        # Compatibility view tensors: AEFR has one factual action, therefore the
        # three historical slots are exact replicas and are diagnostic-only.
        view_probs = final_prob.expand(-1, 3, -1, -1)
        view_logits = final_logits.expand(-1, 3, -1, -1)
        hard_change = ((final_prob >= 0.5) != (anchor >= 0.5)) & region

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
            "dn_anchor_prob": anchor.detach(), "dn_corruption_mask": z,
            "dn_selection_mask": z, "dn_refined_logits": torch.logit(anchor),
            "dn_refined_prob": anchor, "dn_final_prob": anchor, "dn_delta_logit": z,
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
            "c2r_view_logits": view_logits,
            "c2r_view_probs": view_probs,
            "c2r_mean_prob": canonical_prob,
            "c2r_consensus_mask": hard_change.to(anchor),
            "c2r_edit_mask": hard_change.to(anchor),
            "c2r_eroded_anchor": anchor, "c2r_factual_anchor": anchor, "c2r_dilated_anchor": anchor,
            "c2r_v2_enabled": anchor.new_ones((anchor.shape[0],)),
            "pc2r_v3_enabled": anchor.new_zeros((anchor.shape[0],)),
            "pc2r_v31_enabled": anchor.new_zeros((anchor.shape[0],)),
            "aefr_enabled": anchor.new_ones((anchor.shape[0],)),
            "aefr_stage_id": anchor.new_full((anchor.shape[0],), {"single_atomic": 1.0, "single_continuous": 2.0, "hybrid_geometry": 3.0, "typed_state_taylor": 4.0, "typed_state_exact": 5.0, "soft_ownership_exact": 6.0, "intervention_factorized": 7.0, "selective_minimal_intervention": 8.0}[self.stage]),
            "aefr_joint_geometry_grad": anchor.new_full((anchor.shape[0],), 1.0 if self.joint_geometry_grad else 0.0),
            "aefr_transition_aware": anchor.new_full((anchor.shape[0],), 1.0 if self.transition_aware else 0.0),
            "aefr_raw_flow_evidence_enabled": anchor.new_full((anchor.shape[0],), 1.0 if self.use_raw_flow_evidence else 0.0),
            "aefr_transition_delta_logit": transition_delta_logit.detach(),
            "aefr_transition_abs_mean": transition_delta_logit.detach().abs().mean(),
            "aefr_transition_active_fraction": (transition_activity.detach() > 1.0e-6).float().mean(),
            "aefr_transition_flip_fraction": transition_flip.detach().float().mean(),
            "c2r_center_yx": center_yx, "c2r_center_valid": center_valid.to(anchor),
            "c2r_roi_overlap_pixel_count": overlap,
            "c2r_roi_unique_pixel_count": roi_pixel_count,
            "c2r_center_min_chebyshev_distance": min_dist,
            "c2r_candidate_mask": candidate_full.to(anchor),
            "c2r_raw_correction_mask": raw_full.to(anchor),
            "c2r_risk_support_mask": risk_support_full.to(anchor),
            "c2r_commit_mask": deploy_support.to(anchor),
            "pc2r_stage_candidate_mask": candidate_full.to(anchor),
            "pc2r_stage_area_mask": area_full.to(anchor),
            "pc2r_stage_risk_mask": risk_full.to(anchor),
            "pc2r_stage_direction_mask": direction_full.to(anchor),
            "pc2r_stage_spread_mask": spread_full.to(anchor),
            "pc2r_stage_strength_mask": strength_full.to(anchor),
            "pc2r_stage_candidate_prob": torch.where(candidate_full, canonical_prob, anchor),
            "pc2r_stage_area_prob": torch.where(area_full, canonical_prob, anchor),
            "pc2r_stage_risk_prob": torch.where(risk_full, canonical_prob, anchor),
            "pc2r_stage_direction_prob": torch.where(direction_full, canonical_prob, anchor),
            "pc2r_stage_spread_prob": torch.where(spread_full, canonical_prob, anchor),
            "pc2r_stage_strength_prob": torch.where(strength_full, canonical_prob, anchor),
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
            "pc2r_selected_posterior_diversity": probe["posterior_diversity"],
            "pc2r_selected_center_bias_abs": probe["posterior_center_bias_abs"],
            "pc2r_selected_center_bias_signed": probe["posterior_center_bias_signed"],
            "pc2r_mean_abs_delta_logit": delta_patch.detach().abs().mean(),
            "pc2r_reliance_factualized": z.new_zeros(()),
            "pc2r_reliance_shuffled": z.new_zeros(()),
            "aefr_action_delta_logit": delta_logit,
            "aefr_boundary_mask": bnd_full.to(anchor),
            "aefr_boundary_displacement_px": disp_full,
            "aefr_interior_delta_logit": int_full,
            "aefr_state_logits": (state_logits_full if state_logits_full is not None else anchor.new_zeros((anchor.shape[0], 5, h, w))),
            "aefr_state_probs": (state_probs_full if state_probs_full is not None else anchor.new_zeros((anchor.shape[0], 5, h, w))),
            "aefr_ownership_logits": (ownership_logits_full if ownership_logits_full is not None else anchor.new_zeros((anchor.shape[0], 3, h, w))),
            "aefr_ownership_probs": (ownership_probs_full if ownership_probs_full is not None else anchor.new_zeros((anchor.shape[0], 3, h, w))),
            "aefr_error_localizer_logit": intervention_error_logit,
            "aefr_error_localizer_prob": intervention_error_prob,
            "aefr_error_localizer_prior": intervention_error_prior,
            "aefr_edit_logit": intervention_edit_logit_full,
            "aefr_edit_prob": intervention_edit_prob_full,
            "aefr_direction_logit": intervention_direction_logit_full,
            "aefr_direction_prob": intervention_direction_prob_full,
            "aefr_interior_magnitude_logit": intervention_interior_magnitude_full,
            "aefr_signed_action": intervention_signed_action_full,
            "aefr_commit_mask": intervention_commit_full.to(anchor),
            "aefr_boundary_magnitude_px": boundary_magnitude_full,
            "aefr_signed_boundary_action": signed_boundary_full,
            "aefr_signed_interior_action": signed_interior_full,
            "aefr_posterior_stability_support": probe["stability_support"],
            "aefr_posterior_stability_improvement": probe["stability_improvement"],
            "aefr_posterior_disagreement_pre": probe["disagreement_pre"],
            "aefr_posterior_disagreement_post": probe["disagreement_post"],
            "aefr_action_support_fraction": probe["action_support_fraction"],
            "aefr_posterior_diversity_all": probe["posterior_diversity"],
            "aefr_posterior_center_bias_abs": probe["posterior_center_bias_abs"],
            "aefr_posterior_center_bias_signed": probe["posterior_center_bias_signed"],
            "c2r_patch_view_logits": torch.logit(canonical_patch).expand(-1, -1, 3, -1, -1),
            "c2r_patch_view_probs": canonical_patch.expand(-1, -1, 3, -1, -1),
            "c2r_patch_canonical_prob": canonical_patch,
        }
