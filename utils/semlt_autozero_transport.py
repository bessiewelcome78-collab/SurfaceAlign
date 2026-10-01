"""SemLT-LST v3: calibrated rare-edit gating with memory-exact local transport.

This implementation is mathematically equivalent to the original dense local-set
operator, but it never materializes [B, K, H, W] tensors for all K=(2r+1)^2
sources at once. Source logits and local evidence are processed in exact chunks;
chunk summaries are recombined with log-sum-exp identities, and training uses
activation checkpointing for the chunk summaries.

The chunk size is an implementation detail only. It does not alter the model,
loss, source radius, or attention normalization.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from utils.semlt_sdf_geometry import (
    binary_signed_distance,
    foreground_contour_owner,
    gather_owner_field,
    nearest_owner_map,
    sdf_normal,
)

EPS = 1.0e-6


def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def _groups(channels: int) -> int:
    groups = min(8, max(1, int(channels)))
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


class ConvNormGELU(nn.Sequential):
    def __init__(self, cin: int, cout: int, kernel_size: int = 3) -> None:
        super().__init__(
            nn.Conv2d(cin, cout, kernel_size, padding=kernel_size // 2, bias=False),
            nn.GroupNorm(_groups(cout), cout),
            nn.GELU(),
        )


class AutoZeroSemanticTransportSegmenter(nn.Module):
    """Local set-valued transport with exact memory-bounded source aggregation."""

    use_semantic_feature = True
    use_semantic_conditioning = True
    use_text_conditioning = False
    use_anchor_cues = True
    unified_m1_safe_fusion_enabled = True
    mhcs_root_complete = True
    mhcs_geometry_topology_refinement = True
    exact_geotr_m1_only = True
    autozero_semantic_transport = True
    local_set_transport_v2 = True
    memory_exact_local_set_v1 = True
    calibrated_rare_edit_gate_v3 = True
    selective_teacher_rootfix_v31 = True

    def __init__(self, cfg) -> None:
        super().__init__()
        model_cfg = _cfg_get(cfg, "MODEL", None)
        m1_cfg = _cfg_get(cfg, "M1", None)
        self.hidden_dim = int(_cfg_get(model_cfg, "ADAPTER_DIM", 256))
        if self.hidden_dim <= 0:
            raise ValueError("MODEL.ADAPTER_DIM must be positive")

        backbone = str(_cfg_get(model_cfg, "BACKBONE", "ViT-B/16"))
        channels = {"ViT-B/16": 512, "ViT-L/14": 768}
        if backbone not in channels:
            raise ValueError(f"Unsupported backbone for SemLT-LST v3: {backbone!r}")
        self.semantic_channels = channels[backbone]

        self.local_radius = int(_cfg_get(m1_cfg, "SEMLT_LOCAL_RADIUS_PX", 8))
        if not 1 <= self.local_radius <= 16:
            raise ValueError("M1.SEMLT_LOCAL_RADIUS_PX must be in [1,16]")
        self.boundary_normal_warp = bool(
            _cfg_get(m1_cfg, "SEMLT_BOUNDARY_NORMAL_WARP", False)
        )
        self.sdf_operator_matched_warp = bool(
            _cfg_get(m1_cfg, "SEMLT_SDF_OPERATOR_MATCHED_WARP", False)
        )
        self.posterior_stable_operator_warp = bool(
            _cfg_get(m1_cfg, "SEMLT_POSTERIOR_STABLE_OPERATOR_WARP", False)
        )
        self.uc_fnrt = bool(_cfg_get(m1_cfg, "SEMLT_UC_FNRT", False))
        # SEMLT_DS_UC_FNRT_STAGEA_V2
        # Domain-stable evidence only: no new trainable modules/parameters.
        self.ds_uc_fnrt = bool(_cfg_get(m1_cfg, "SEMLT_DS_UC_FNRT", False))
        self.ds_relative_uncertainty = bool(
            _cfg_get(m1_cfg, "SEMLT_DS_RELATIVE_UNCERTAINTY", self.ds_uc_fnrt)
        )
        self.ds_normalized_ray = bool(
            _cfg_get(m1_cfg, "SEMLT_DS_NORMALIZED_RAY", self.ds_uc_fnrt)
        )
        self.ds_ray_agreement = bool(
            _cfg_get(m1_cfg, "SEMLT_DS_RAY_AGREEMENT", self.ds_uc_fnrt)
        )
        # ROOTCAUSE-A2: decouple *measuring* multi-scale ray agreement from
        # *multiplying* the physical displacement by it.  For backwards
        # compatibility, legacy configs that do not specify the new flag keep
        # the archived Stage-A behaviour.  New A2/root-cause configs set this
        # to false so agreement is diagnostic-only.
        self.ds_ray_agreement_apply = bool(
            _cfg_get(m1_cfg, "SEMLT_DS_RAY_AGREEMENT_APPLY", self.ds_ray_agreement)
        )
        if self.ds_ray_agreement_apply and not self.ds_ray_agreement:
            raise ValueError(
                "SEMLT_DS_RAY_AGREEMENT_APPLY=true requires "
                "SEMLT_DS_RAY_AGREEMENT=true"
            )

        # DNR-RootFix: represent the *joint signed offset* as an ordered
        # distribution over {-r,...,0,...,+r}.  A matched inactive head can be
        # registered in the control run so D0/D1 have identical state-dict and
        # optimizer structure.  Existing UC-FNRT configs remain fully backward
        # compatible because registration defaults to the enabled flag.
        self.uc_offset_distribution = bool(
            _cfg_get(m1_cfg, "SEMLT_UC_OFFSET_DISTRIBUTION", False)
        )
        self.uc_offset_head_registered = bool(
            _cfg_get(
                m1_cfg,
                "SEMLT_UC_OFFSET_HEAD_REGISTERED",
                self.uc_offset_distribution,
            )
        )
        if self.uc_offset_distribution and not self.uc_offset_head_registered:
            raise ValueError(
                "SEMLT_UC_OFFSET_DISTRIBUTION=true requires "
                "SEMLT_UC_OFFSET_HEAD_REGISTERED=true"
            )
        if self.uc_offset_head_registered and not self.uc_fnrt:
            raise ValueError(
                "SEMLT_UC_OFFSET_HEAD_REGISTERED is defined only for UC-FNRT"
            )

        # HRCV-RootFix: Hypothesis-conditioned 1-D normal-ray cost volume.
        # The scorer is shared across all signed offset hypotheses.  The strict
        # H0/H1 causal pair registers and trains the same modules; the only
        # intervention is whether local evidence is sampled at x (H0) or at
        # the candidate x + k*n(x) (H1).
        self.uc_hrcv = bool(
            _cfg_get(m1_cfg, "SEMLT_UC_HRCV", False)
        )
        self.uc_hrcv_registered = bool(
            _cfg_get(m1_cfg, "SEMLT_UC_HRCV_REGISTERED", self.uc_hrcv)
        )
        self.uc_hrcv_candidate_conditioned = bool(
            _cfg_get(m1_cfg, "SEMLT_UC_HRCV_CANDIDATE_CONDITIONED", self.uc_hrcv)
        )
        self.uc_hrcv_delta_px = float(
            _cfg_get(m1_cfg, "SEMLT_UC_HRCV_DELTA_PX", 1.0)
        )
        if self.uc_hrcv and not self.uc_offset_distribution:
            raise ValueError(
                "SEMLT_UC_HRCV=true requires SEMLT_UC_OFFSET_DISTRIBUTION=true"
            )
        if self.uc_hrcv and not self.uc_hrcv_registered:
            raise ValueError(
                "SEMLT_UC_HRCV=true requires SEMLT_UC_HRCV_REGISTERED=true"
            )
        if self.uc_hrcv_registered and not self.uc_fnrt:
            raise ValueError(
                "SEMLT_UC_HRCV_REGISTERED is defined only for UC-FNRT"
            )
        if self.uc_hrcv_candidate_conditioned and not self.uc_hrcv:
            raise ValueError(
                "SEMLT_UC_HRCV_CANDIDATE_CONDITIONED=true requires SEMLT_UC_HRCV=true"
            )
        if not (0.0 < self.uc_hrcv_delta_px <= 2.0):
            raise ValueError("SEMLT_UC_HRCV_DELTA_PX must lie in (0,2]")

        # MRM-RootFix: Mode-Resolved Ray Matching.  HRCV H1 proved that each
        # signed hypothesis must observe x+k*n(x), but the current HRCV scorer
        # is still unary and the DNR decoder still takes the global posterior
        # mean.  The MRM branch therefore separates three causal interventions:
        #   (1) relational query<->candidate costs w.r.t. the current contour,
        #   (2) ordered 1-D aggregation across neighbouring signed hypotheses,
        #   (3) dominant-mode decoding instead of full-distribution averaging.
        # All modules can be registered in every control so parameter/state-dict
        # structure remains matched even when a particular intervention is off.
        self.uc_mrm_registered = bool(
            _cfg_get(m1_cfg, "SEMLT_UC_MRM_REGISTERED", False)
        )
        self.uc_mrm_relational_cost = bool(
            _cfg_get(m1_cfg, "SEMLT_UC_MRM_RELATIONAL_COST", False)
        )
        self.uc_mrm_ordered_aggregation = bool(
            _cfg_get(m1_cfg, "SEMLT_UC_MRM_ORDERED_AGGREGATION", False)
        )
        self.uc_mrm_dominant_mode = bool(
            _cfg_get(m1_cfg, "SEMLT_UC_MRM_DOMINANT_MODE", False)
        )
        self.uc_mrm_mode_radius = int(
            _cfg_get(m1_cfg, "SEMLT_UC_MRM_MODE_RADIUS", 1)
        )
        # OACD root fix.  ``owner_scalar`` is an inverse-warp sampling offset:
        # a boundary motion of +m along the Base foreground normal requires a
        # sampling displacement d=-m.  Candidate evidence for bin d must
        # therefore be observed at x-d*n, not at x+d*n.  The archived HRCV/MRM
        # route used the latter and paired every supervised bin with the mirror
        # point on the opposite side of the Base contour.
        self.uc_operator_aligned_candidates = bool(
            _cfg_get(
                m1_cfg,
                "SEMLT_UC_OPERATOR_ALIGNED_CANDIDATES",
                False,
            )
        )
        # OACD-R3: a flat posterior mean is not an action-safe decoder for a
        # signed displacement.  Positive and negative modes cancel even when
        # both have learned a plausible magnitude.  R3 first makes the
        # three-way decision NEGATIVE / KEEP / POSITIVE, then decodes the
        # magnitude *conditional on the winning sign*.  A normalized margin
        # between the best and second-best sign groups is the physical trust
        # dose.  The radius-dependent KEEP prior below makes equal evidence an
        # exact three-way tie and therefore an exact AutoZero identity.
        self.uc_hierarchical_confidence_decoder = bool(
            _cfg_get(
                m1_cfg,
                "SEMLT_UC_HIERARCHICAL_CONFIDENCE_DECODER",
                False,
            )
        )
        self.uc_hierarchical_execution = str(
            _cfg_get(
                m1_cfg,
                "SEMLT_UC_HIERARCHICAL_EXECUTION",
                "margin_dose",
            )
            or "margin_dose"
        ).strip().lower()
        if self.uc_hierarchical_execution not in {"margin_dose", "full_conditional"}:
            raise ValueError(
                "SEMLT_UC_HIERARCHICAL_EXECUTION must be margin_dose or full_conditional"
            )
        self.uc_radius_balanced_keep_prior = bool(
            _cfg_get(
                m1_cfg,
                "SEMLT_UC_RADIUS_BALANCED_KEEP_PRIOR",
                self.uc_hierarchical_confidence_decoder,
            )
        )
        if self.uc_mrm_registered and not self.uc_hrcv_registered:
            raise ValueError(
                "SEMLT_UC_MRM_REGISTERED=true requires SEMLT_UC_HRCV_REGISTERED=true"
            )
        if (self.uc_mrm_relational_cost or self.uc_mrm_ordered_aggregation) and not self.uc_mrm_registered:
            raise ValueError(
                "MRM relational/aggregation interventions require SEMLT_UC_MRM_REGISTERED=true"
            )
        if self.uc_mrm_ordered_aggregation and not self.uc_mrm_relational_cost:
            raise ValueError(
                "SEMLT_UC_MRM_ORDERED_AGGREGATION=true requires relational cost"
            )
        if self.uc_mrm_dominant_mode and not self.uc_offset_distribution:
            raise ValueError(
                "SEMLT_UC_MRM_DOMINANT_MODE=true requires distributional offset transport"
            )
        if (self.uc_mrm_relational_cost or self.uc_mrm_ordered_aggregation) and not self.uc_hrcv_candidate_conditioned:
            raise ValueError(
                "MRM matching requires candidate-conditioned HRCV evidence"
            )
        if self.uc_mrm_mode_radius < 1 or self.uc_mrm_mode_radius > 2:
            raise ValueError("SEMLT_UC_MRM_MODE_RADIUS must be 1 or 2")
        if self.uc_operator_aligned_candidates and not self.uc_hrcv_candidate_conditioned:
            raise ValueError(
                "SEMLT_UC_OPERATOR_ALIGNED_CANDIDATES requires "
                "SEMLT_UC_HRCV_CANDIDATE_CONDITIONED=true"
            )
        if self.uc_hierarchical_confidence_decoder and not self.uc_offset_distribution:
            raise ValueError(
                "SEMLT_UC_HIERARCHICAL_CONFIDENCE_DECODER requires "
                "SEMLT_UC_OFFSET_DISTRIBUTION=true"
            )
        if self.uc_radius_balanced_keep_prior and not self.uc_offset_distribution:
            raise ValueError(
                "SEMLT_UC_RADIUS_BALANCED_KEEP_PRIOR requires "
                "SEMLT_UC_OFFSET_DISTRIBUTION=true"
            )

        if self.ds_uc_fnrt and not self.uc_fnrt:
            raise ValueError("SEMLT_DS_UC_FNRT requires SEMLT_UC_FNRT=true")
        # Causal ablation is a SINGLE categorical intervention, never a stack
        # of optional modules. ``full`` preserves the validated UC-FNRT route;
        # every other value removes exactly one scientific factor while leaving
        # the host, MC budget, warp operator, radius and optimizer unchanged.
        self.uc_ablation = str(
            _cfg_get(m1_cfg, "SEMLT_UC_FNRT_ABLATION", "full") or "full"
        ).strip().lower()
        allowed_uc_ablations = {
            "full", "no_posterior_uncertainty", "no_normal_ray_evidence",
            "direct_signed", "segmentation_only",
        }
        if self.uc_ablation not in allowed_uc_ablations:
            raise ValueError(
                "M1.SEMLT_UC_FNRT_ABLATION must be one of "
                + ", ".join(sorted(allowed_uc_ablations))
            )
        if self.uc_ablation != "full" and not self.uc_fnrt:
            raise ValueError(
                "SEMLT_UC_FNRT_ABLATION is only defined when SEMLT_UC_FNRT=true"
            )
        if self.uc_offset_distribution and self.uc_ablation != "full":
            raise ValueError(
                "Distributional signed-offset transport is a root-fix branch and "
                "must use SEMLT_UC_FNRT_ABLATION=full."
            )
        if self.uc_offset_distribution and (
            self.ds_relative_uncertainty
            or self.ds_normalized_ray
            or self.ds_ray_agreement_apply
        ):
            raise ValueError(
                "DNR root-fix must be isolated from relative-U, normalized-ray, "
                "and multiplicative agreement interventions."
            )
        if sum(int(v) for v in (
            self.boundary_normal_warp,
            self.sdf_operator_matched_warp,
            self.posterior_stable_operator_warp,
            self.uc_fnrt,
        )) > 1:
            raise ValueError(
                "SEMLT_BOUNDARY_NORMAL_WARP, SEMLT_SDF_OPERATOR_MATCHED_WARP, "
                "SEMLT_POSTERIOR_STABLE_OPERATOR_WARP and SEMLT_UC_FNRT are mutually exclusive"
            )
        self.kernel_size = 2 * self.local_radius + 1
        self.num_sources = self.kernel_size * self.kernel_size
        self.detach_conditioners = bool(
            _cfg_get(m1_cfg, "SEMLT_AUTOZERO_DETACH_CONDITIONERS", True)
        )
        if not self.detach_conditioners:
            raise ValueError("SemLT-LST v3 requires detached Base/semantic conditioners")

        self.image_stem = nn.Sequential(
            ConvNormGELU(3, self.hidden_dim),
            ConvNormGELU(self.hidden_dim, self.hidden_dim),
        )
        self.semantic_proj = nn.Sequential(
            nn.Conv2d(self.semantic_channels, self.hidden_dim, 1, bias=False),
            nn.GroupNorm(_groups(self.hidden_dim), self.hidden_dim),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            ConvNormGELU(2 * self.hidden_dim + 3, self.hidden_dim),
            ConvNormGELU(self.hidden_dim, self.hidden_dim),
            ConvNormGELU(self.hidden_dim, self.hidden_dim),
        )
        if self.uc_fnrt:
            # UC-FNRT keeps the archived PS-OMW representation untouched.  The
            # new route explicitly conditions on posterior structural uncertainty
            # and on bilateral evidence sampled along the Base SDF normal.
            self.uc_context_fuse = nn.Sequential(
                ConvNormGELU(2 * self.hidden_dim + 5, self.hidden_dim),
                ConvNormGELU(self.hidden_dim, self.hidden_dim),
            )
            ray_dim = max(16, self.hidden_dim // 8)
            cue_dim = max(8, self.hidden_dim // 16)
            self.uc_ray_image_proj = nn.Sequential(
                nn.Conv2d(self.hidden_dim, ray_dim, 1, bias=False),
                nn.GroupNorm(_groups(ray_dim), ray_dim), nn.GELU(),
            )
            self.uc_ray_semantic_proj = nn.Sequential(
                nn.Conv2d(self.hidden_dim, ray_dim, 1, bias=False),
                nn.GroupNorm(_groups(ray_dim), ray_dim), nn.GELU(),
            )
            self.uc_ray_cue_proj = nn.Sequential(
                nn.Conv2d(3, cue_dim, 1, bias=False),
                nn.GroupNorm(_groups(cue_dim), cue_dim), nn.GELU(),
            )
            ray_channels = 4 * (2 * ray_dim + cue_dim)
            self.uc_ray_fuse = nn.Sequential(
                ConvNormGELU(self.hidden_dim + ray_channels, self.hidden_dim),
                ConvNormGELU(self.hidden_dim, self.hidden_dim),
            )
            self.direction_controller = nn.Conv2d(self.hidden_dim, 1, kernel_size=3, padding=1)
            self.magnitude_controller = nn.Conv2d(self.hidden_dim, 1, kernel_size=3, padding=1)
            nn.init.zeros_(self.direction_controller.weight)
            nn.init.zeros_(self.direction_controller.bias)
            nn.init.zeros_(self.magnitude_controller.weight)
            nn.init.zeros_(self.magnitude_controller.bias)

            if self.uc_offset_head_registered:
                # Preserve the global CPU RNG state so merely registering the
                # matched D0/D1 head cannot perturb any module constructed later.
                # Conv2d's default random initialization is immediately replaced
                # by exact zeros; restoring RNG makes registration causally inert.
                _cpu_rng_state = torch.random.get_rng_state()
                self.offset_controller = nn.Conv2d(
                    self.hidden_dim,
                    2 * self.local_radius + 1,
                    kernel_size=3,
                    padding=1,
                )
                torch.random.set_rng_state(_cpu_rng_state)
                nn.init.zeros_(self.offset_controller.weight)
                nn.init.zeros_(self.offset_controller.bias)
                self.register_buffer(
                    "uc_offset_values_px",
                    torch.arange(
                        -self.local_radius,
                        self.local_radius + 1,
                        dtype=torch.float32,
                    ),
                    persistent=False,
                )
            else:
                self.offset_controller = None

            if self.uc_hrcv_registered:
                # Register the matched H0/H1 cost-volume modules without changing
                # construction RNG for any later module.  Only the final shared
                # unary scorer is zero-initialized, yielding equal logits for all
                # k and therefore exact zero expected offset at initialization.
                _cpu_rng_state_hrcv = torch.random.get_rng_state()
                cost_dim = max(16, self.hidden_dim // 8)
                cue_dim_hrcv = max(8, self.hidden_dim // 16)
                self.uc_hrcv_cost_dim = int(cost_dim)
                self.uc_hrcv_cue_dim = int(cue_dim_hrcv)
                self.uc_hrcv_context = ConvNormGELU(
                    2 * self.hidden_dim + 5, cost_dim, kernel_size=1
                )
                self.uc_hrcv_cue_proj = nn.Sequential(
                    nn.Conv2d(5, cue_dim_hrcv, 1, bias=False),
                    nn.GroupNorm(_groups(cue_dim_hrcv), cue_dim_hrcv),
                    nn.GELU(),
                )
                # Candidate descriptor:
                #   compact center context
                #   image center + local normal transition
                #   semantic center + local normal transition
                #   dense cue center + local normal transition
                #   signed coordinate k/r
                hrcv_in = (
                    cost_dim
                    + 2 * ray_dim
                    + 2 * ray_dim
                    + 2 * cue_dim_hrcv
                    + 1
                )
                self.uc_hrcv_scorer = nn.Sequential(
                    nn.Linear(hrcv_in, cost_dim, bias=True),
                    nn.GELU(),
                    nn.Linear(cost_dim, 1, bias=True),
                )
                torch.random.set_rng_state(_cpu_rng_state_hrcv)
                nn.init.zeros_(self.uc_hrcv_scorer[-1].weight)
                nn.init.zeros_(self.uc_hrcv_scorer[-1].bias)
            else:
                self.uc_hrcv_context = None
                self.uc_hrcv_cue_proj = None
                self.uc_hrcv_scorer = None
                self.uc_hrcv_cost_dim = 0
                self.uc_hrcv_cue_dim = 0

            if self.uc_mrm_registered:
                # Register every MRM component for all matched controls.  RNG is
                # restored after construction so merely registering the root-fix
                # cannot change any later module initialization.
                _cpu_rng_state_mrm = torch.random.get_rng_state()
                cost_dim_mrm = int(self.uc_hrcv_cost_dim)
                cue_dim_mrm = int(self.uc_hrcv_cue_dim)
                # Relational descriptor = compact owner context + candidate-owner
                # center differences + candidate local transitions for image,
                # semantic and dense cues + two cosine similarities + k/r.
                mrm_in = (
                    cost_dim_mrm
                    + 4 * ray_dim
                    + 2 * cue_dim_mrm
                    + 3
                )
                self.uc_mrm_rel_embed = nn.Sequential(
                    nn.Linear(mrm_in, cost_dim_mrm, bias=True),
                    nn.GELU(),
                    nn.Linear(cost_dim_mrm, cost_dim_mrm, bias=True),
                    nn.GELU(),
                )
                self.uc_mrm_unary_head = nn.Linear(cost_dim_mrm, 1, bias=True)
                self.uc_mrm_aggregate = nn.Sequential(
                    nn.Conv1d(cost_dim_mrm, cost_dim_mrm, 3, padding=1, bias=False),
                    nn.GroupNorm(_groups(cost_dim_mrm), cost_dim_mrm),
                    nn.GELU(),
                    nn.Conv1d(cost_dim_mrm, cost_dim_mrm, 3, padding=1, bias=True),
                )
                self.uc_mrm_head = nn.Conv1d(cost_dim_mrm, 1, 1, bias=True)
                torch.random.set_rng_state(_cpu_rng_state_mrm)
                # Zero final heads preserve exact equal-logit AutoZero identity.
                nn.init.zeros_(self.uc_mrm_unary_head.weight)
                nn.init.zeros_(self.uc_mrm_unary_head.bias)
                nn.init.zeros_(self.uc_mrm_head.weight)
                nn.init.zeros_(self.uc_mrm_head.bias)
            else:
                self.uc_mrm_rel_embed = None
                self.uc_mrm_unary_head = None
                self.uc_mrm_aggregate = None
                self.uc_mrm_head = None
        # v3 deliberately separates representation balancing from deployment:
        # [calibrated EDIT, auxiliary balanced LOCATOR, ADD/REMOVE type,
        #  local source scores...].  Only EDIT may gate the physical output.
        # This prevents a class-balanced discriminator from being misread as a
        # posterior when correctable pixels are extremely rare.
        if self.uc_fnrt:
            # Compatibility name only; the physical UC-FNRT route uses the two
            # factorized heads above.  No extra trainable legacy controller is
            # registered.
            self.controller = self.direction_controller
        elif self.boundary_normal_warp or self.sdf_operator_matched_warp or self.posterior_stable_operator_warp:
            # Geometry-warp routes use one signed scalar field.  In the new SDF
            # route only the one-sided contour-owner values are physical; they
            # are extended to the swept band by nearest-contour ownership.
            # Positive/negative values are inverse-warp sampling offsets along the
            # Base foreground normal; zero is exact KEEP.  Zero initialization is
            # therefore a structural identity, not a learned gate heuristic.
            self.controller = nn.Conv2d(
                self.hidden_dim, 1, kernel_size=3, padding=1
            )
            nn.init.zeros_(self.controller.weight)
            nn.init.zeros_(self.controller.bias)
        else:
            self.controller = nn.Conv2d(
                self.hidden_dim, 4 + self.num_sources, kernel_size=3, padding=1
            )
            nn.init.normal_(self.controller.weight, mean=0.0, std=1.0e-3)
            nn.init.zeros_(self.controller.bias)
            with torch.no_grad():
                # Safe identity-biased initialization for archived sparse-gate routes.
                self.controller.bias[0] = -6.0
        self.force_preserve = False

        # V3.1 root fix: selection and correction have different responsibilities.
        # The deployment gate is a sparse *decision*, not a continuous edit dose.
        # Outside a realizable local FG/BG transition band, transport is physically
        # impossible and the output is exact Base.  These switches are explicit so
        # archived v3 configs remain reproducible when SEMLT_LST_V31_ROOTFIX=false.
        self.v31_rootfix = bool(_cfg_get(m1_cfg, "SEMLT_LST_V31_ROOTFIX", False))
        self.action_value_policy = bool(
            _cfg_get(m1_cfg, "SEMLT_ACTION_CONSISTENT_VALUE_POLICY", False)
        )
        self.transition_band_enabled = bool(
            _cfg_get(m1_cfg, "SEMLT_TRANSITION_BAND_ENABLED", self.v31_rootfix)
        )
        self.hard_deploy = bool(_cfg_get(m1_cfg, "SEMLT_HARD_DEPLOY", self.v31_rootfix))
        self.deploy_threshold = float(_cfg_get(m1_cfg, "SEMLT_DEPLOY_THRESHOLD", 0.5))
        if self.v31_rootfix and not self.hard_deploy:
            raise ValueError("SemLT-LST v3.1 requires SEMLT_HARD_DEPLOY=true")
        if not 0.0 < self.deploy_threshold < 1.0:
            raise ValueError("M1.SEMLT_DEPLOY_THRESHOLD must be in (0,1)")

        offsets = [
            (dx, dy)
            for dy in range(-self.local_radius, self.local_radius + 1)
            for dx in range(-self.local_radius, self.local_radius + 1)
        ]
        self.register_buffer(
            "source_offsets_px", torch.tensor(offsets, dtype=torch.float32), persistent=False
        )

    @staticmethod
    def _resize(x: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] == hw:
            return x
        return F.interpolate(x, size=hw, mode="bilinear", align_corners=False)


    def _semantic_project_lowmem(
        self,
        semantic_map: torch.Tensor,
        hw: Tuple[int, int],
    ) -> torch.Tensor:
        """Memory-equivalent semantic projection for AutoZero / UC-FNRT.

        Historical order:
            interpolate(C_in=semantic_channels) -> Conv1x1 -> GN -> GELU

        LOWMEM4 order:
            Conv1x1 at patch grid -> interpolate(C_out=hidden_dim) -> GN -> GELU

        The Conv2d is kernel_size=1 and bias=False. Bilinear resize is a
        channel-independent linear operator, therefore Conv1x1 and resize
        commute in exact arithmetic. GroupNorm and GELU remain after resize.

        No parameter, loss, radius, batch size, MC budget, or architecture
        contract is changed.
        """
        if semantic_map.ndim != 4:
            raise ValueError(
                f"Expected semantic_map [B,C,H,W], got {tuple(semantic_map.shape)}"
            )
        if semantic_map.shape[1] != self.semantic_channels:
            raise ValueError(
                "Semantic channel mismatch in LOWMEM4: "
                f"expected {self.semantic_channels}, "
                f"got {semantic_map.shape[1]}"
            )

        x = semantic_map.detach()

        conv = self.semantic_proj[0]
        norm = self.semantic_proj[1]
        act  = self.semantic_proj[2]

        # 512 -> hidden_dim while still on the small ViT patch grid.
        x = conv(x)

        # Only hidden_dim channels are now expanded spatially.
        x = self._resize(x, hw)

        # Preserve historical ordering of all nonlinear operations.
        x = norm(x)
        x = act(x)
        return x

    @staticmethod
    def _base_boundary(probability: torch.Tensor) -> torch.Tensor:
        dilated = F.max_pool2d(probability, 3, stride=1, padding=1)
        eroded = -F.max_pool2d(-probability, 3, stride=1, padding=1)
        return (dilated - eroded).clamp(0.0, 1.0)

    @staticmethod
    def _flow_jacobian_stats(flow_px: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if flow_px.shape[-2] < 2 or flow_px.shape[-1] < 2:
            b = flow_px.shape[0]
            return flow_px.new_ones(b), flow_px.new_zeros(b)
        ux, uy = flow_px[:, 0], flow_px[:, 1]
        dux_dx = ux[:, :-1, 1:] - ux[:, :-1, :-1]
        dux_dy = ux[:, 1:, :-1] - ux[:, :-1, :-1]
        duy_dx = uy[:, :-1, 1:] - uy[:, :-1, :-1]
        duy_dy = uy[:, 1:, :-1] - uy[:, :-1, :-1]
        det = (1.0 + dux_dx) * (1.0 + duy_dy) - dux_dy * duy_dx
        return det.mean(dim=(1, 2)), (det <= 0.0).float().mean(dim=(1, 2))

    @staticmethod
    def _auto_source_chunk_size(reference: torch.Tensor, num_sources: int) -> int:
        """Choose an implementation-only chunk size from the current tensor shape.

        This is not a method hyperparameter: every chunking choice evaluates the
        same exact source softmax. The target limits temporary source-channel
        working sets while keeping enough channels per convolution for throughput.
        """
        b, _, h, w = reference.shape
        bytes_per_channel = max(1, b * h * w * reference.element_size())
        # Approximate temporary footprint per source channel inside the chunk:
        # scores, local logits/probs, add/remove energies and a few work buffers.
        target_working_bytes = 512 * 1024 * 1024
        estimated_live_buffers = 8
        chunk = target_working_bytes // (bytes_per_channel * estimated_live_buffers)
        chunk = max(4, int(chunk))
        chunk = min(32, chunk, int(num_sources))
        return max(1, chunk)

    def _local_chunk(
        self,
        padded_logits: torch.Tensor,
        padded_valid: torch.Tensor,
        start: int,
        end: int,
        h: int,
        w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r = self.local_radius
        local_logits: List[torch.Tensor] = []
        valid: List[torch.Tensor] = []
        for idx in range(start, end):
            dx = int(self.source_offsets_px[idx, 0].item())
            dy = int(self.source_offsets_px[idx, 1].item())
            y0 = r + dy
            x0 = r + dx
            local_logits.append(padded_logits[:, :, y0:y0 + h, x0:x0 + w])
            valid.append(padded_valid[:, :, y0:y0 + h, x0:x0 + w])
        # [B, Cchunk, H, W]
        return torch.cat(local_logits, dim=1), torch.cat(valid, dim=1)

    def _chunk_summary(
        self,
        hidden: torch.Tensor,
        padded_logits: torch.Tensor,
        padded_valid: torch.Tensor,
        start: int,
        end: int,
        h: int,
        w: int,
    ) -> Tuple[torch.Tensor, ...]:
        weight = self.controller.weight[4 + start:4 + end]
        bias = self.controller.bias[4 + start:4 + end]
        scores = F.conv2d(hidden, weight, bias, stride=1, padding=1)
        local_logits, valid = self._local_chunk(
            padded_logits, padded_valid, start, end, h, w
        )
        local_prob = torch.sigmoid(local_logits).clamp(EPS, 1.0 - EPS)

        add_energy = (scores + torch.log(local_prob)).masked_fill(valid <= 0.5, -torch.inf)
        remove_energy = (scores + torch.log1p(-local_prob)).masked_fill(valid <= 0.5, -torch.inf)

        add_lse = torch.logsumexp(add_energy, dim=1, keepdim=True)
        remove_lse = torch.logsumexp(remove_energy, dim=1, keepdim=True)

        add_safe_lse = torch.where(torch.isfinite(add_lse), add_lse, torch.zeros_like(add_lse))
        remove_safe_lse = torch.where(
            torch.isfinite(remove_lse), remove_lse, torch.zeros_like(remove_lse)
        )
        add_w = torch.where(
            valid > 0.5,
            torch.exp(add_energy - add_safe_lse),
            torch.zeros_like(add_energy),
        )
        remove_w = torch.where(
            valid > 0.5,
            torch.exp(remove_energy - remove_safe_lse),
            torch.zeros_like(remove_energy),
        )

        offsets = self.source_offsets_px[start:end].to(local_logits)
        dx = offsets[:, 0][None, :, None, None]
        dy = offsets[:, 1][None, :, None, None]
        local_hard_fg = (local_prob >= 0.5).to(local_prob) * valid
        local_hard_bg = (local_prob < 0.5).to(local_prob) * valid

        # All moments are conditional on this chunk. Global combination uses
        # exp(chunk_lse - global_lse), which is exactly the full softmax mass.
        return (
            add_lse,
            (add_w * local_logits).sum(1, keepdim=True),
            (add_w * dx).sum(1, keepdim=True),
            (add_w * dy).sum(1, keepdim=True),
            (add_w * local_hard_fg).sum(1, keepdim=True),
            remove_lse,
            (remove_w * local_logits).sum(1, keepdim=True),
            (remove_w * dx).sum(1, keepdim=True),
            (remove_w * dy).sum(1, keepdim=True),
            (remove_w * local_hard_bg).sum(1, keepdim=True),
        )

    def _memory_exact_local_aggregate(
        self,
        hidden: torch.Tensor,
        factual_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        b, _, h, w = factual_logits.shape
        r = self.local_radius
        padded_logits = F.pad(factual_logits, (r, r, r, r), mode="constant", value=0.0)
        padded_valid = F.pad(
            torch.ones_like(factual_logits), (r, r, r, r), mode="constant", value=0.0
        )
        chunk_size = self._auto_source_chunk_size(factual_logits, self.num_sources)

        chunks: List[Tuple[torch.Tensor, ...]] = []
        for start in range(0, self.num_sources, chunk_size):
            end = min(self.num_sources, start + chunk_size)
            if self.training and torch.is_grad_enabled() and hidden.requires_grad:
                # Checkpoint prevents source-channel work buffers from being kept
                # alive for backward; only compact [B,1,H,W] summaries survive.
                out = checkpoint(
                    lambda h_, pl_, pv_, s=start, e=end: self._chunk_summary(
                        h_, pl_, pv_, s, e, h, w
                    ),
                    hidden,
                    padded_logits,
                    padded_valid,
                    use_reentrant=False,
                )
            else:
                out = self._chunk_summary(
                    hidden, padded_logits, padded_valid, start, end, h, w
                )
            chunks.append(out)

        def combine(lse_index: int, value_index: int) -> torch.Tensor:
            lses = torch.cat([c[lse_index] for c in chunks], dim=1)
            chunk_mass = torch.softmax(lses, dim=1)
            values = torch.cat([c[value_index] for c in chunks], dim=1)
            return (chunk_mass * values).sum(1, keepdim=True)

        add_logits = combine(0, 1)
        add_dx = combine(0, 2)
        add_dy = combine(0, 3)
        add_fg_mass = combine(0, 4).clamp(0.0, 1.0)
        remove_logits = combine(5, 6)
        remove_dx = combine(5, 7)
        remove_dy = combine(5, 8)
        remove_bg_mass = combine(5, 9).clamp(0.0, 1.0)

        base_hard = (torch.sigmoid(factual_logits) >= 0.5).to(factual_logits)
        add_source_exists = (
            F.max_pool2d(base_hard, self.kernel_size, stride=1, padding=r) > 0.5
        )
        remove_source_exists = (
            F.max_pool2d(1.0 - base_hard, self.kernel_size, stride=1, padding=r) > 0.5
        )

        return {
            "add_logits": add_logits,
            "remove_logits": remove_logits,
            "add_dx": add_dx,
            "add_dy": add_dy,
            "remove_dx": remove_dx,
            "remove_dy": remove_dy,
            "add_valid_source_mass": add_fg_mass,
            "remove_valid_source_mass": remove_bg_mass,
            "add_source_exists": add_source_exists,
            "remove_source_exists": remove_source_exists,
            "source_chunk_size": factual_logits.new_tensor(float(chunk_size)),
            "source_chunk_count": factual_logits.new_tensor(
                float((self.num_sources + chunk_size - 1) // chunk_size)
            ),
        }


    @staticmethod
    def _hard_boundary(mask: torch.Tensor) -> torch.Tensor:
        """Two-sided 1-pixel morphological boundary of a binary mask."""
        x = mask.float()
        dilated = F.max_pool2d(x, 3, stride=1, padding=1)
        eroded = -F.max_pool2d(-x, 3, stride=1, padding=1)
        return (dilated - eroded) > 0.5

    @staticmethod
    def _foreground_normal(probability: torch.Tensor) -> torch.Tensor:
        """Unit normal pointing toward increasing foreground probability."""
        kx = probability.new_tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
        )[:, None]
        ky = probability.new_tensor(
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]
        )[:, None]
        gx = F.conv2d(probability, kx, padding=1)
        gy = F.conv2d(probability, ky, padding=1)
        magnitude = torch.sqrt(gx.square() + gy.square() + 1.0e-12)
        valid = magnitude > 1.0e-5
        nx = torch.where(valid, gx / magnitude, torch.zeros_like(gx))
        ny = torch.where(valid, gy / magnitude, torch.zeros_like(gy))
        return torch.cat([nx, ny], dim=1)

    @staticmethod
    def _identity_grid(reference: torch.Tensor) -> torch.Tensor:
        b, _, h, w = reference.shape
        ys = torch.linspace(-1.0, 1.0, h, device=reference.device, dtype=reference.dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=reference.device, dtype=reference.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gx, gy], dim=-1)[None].expand(b, h, w, 2)

    def _generate_sdf_operator_matched_warp(
        self,
        factual_logits: torch.Tensor,
        base_prob: torch.Tensor,
        hidden: torch.Tensor,
        soft_boundary: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """SDF-guided operator-matched contour warp.

        A one-sided Base contour owns one signed inverse-warp displacement.  The
        owner displacement and SDF normal are extended by nearest-contour
        ownership, and only the actually swept narrow band is physically warped.
        This removes the old dense-band independent-flow ambiguity.
        """
        b, _, h, w = factual_logits.shape
        base_hard = base_prob >= 0.5

        # Continuous positive-inside SDF: the zero level lies half a pixel between
        # opposite labels, so there is no two-sided morphological-shell target.
        base_sdf = binary_signed_distance(base_hard, dtype=factual_logits.dtype)
        base_normal, normal_valid = sdf_normal(base_sdf)
        contour_owner = foreground_contour_owner(base_hard) & normal_valid

        owner_distance, owner_flat_index, owner_case_valid = nearest_owner_map(
            contour_owner, dtype=factual_logits.dtype
        )

        raw_scalar = self.controller(hidden)
        owner_scalar = float(self.local_radius) * torch.tanh(raw_scalar)
        if self.force_preserve:
            owner_scalar = torch.zeros_like(owner_scalar)

        # Only contour-owner values are physical.  Nearest-owner gathering is
        # differentiable w.r.t. those owner values, so dense support gradients
        # return to the contour rather than creating independent band actions.
        extended_offset = gather_owner_field(owner_scalar, owner_flat_index, owner_case_valid)
        extended_normal = gather_owner_field(base_normal.detach(), owner_flat_index, owner_case_valid)
        extended_normal_mag = torch.linalg.vector_norm(extended_normal, dim=1, keepdim=True).clamp_min(EPS)
        extended_normal = torch.where(
            extended_normal_mag > 1.0e-5,
            extended_normal / extended_normal_mag,
            torch.zeros_like(extended_normal),
        )

        # Physical support is the contour swept by the predicted displacement,
        # plus exactly one pixel for bilinear interpolation footprint.
        swept_radius = extended_offset.detach().abs()[:, 0] + 1.0
        support = (
            owner_case_valid[:, None, None]
            & torch.isfinite(owner_distance)
            & (owner_distance <= swept_radius)
        )
        sample_offset_px = extended_offset * support[:, None].to(extended_offset)
        flow_px = sample_offset_px * extended_normal

        grid = self._identity_grid(factual_logits)
        if w > 1:
            grid_x = grid[..., 0] + 2.0 * flow_px[:, 0] / float(w - 1)
        else:
            grid_x = grid[..., 0]
        if h > 1:
            grid_y = grid[..., 1] + 2.0 * flow_px[:, 1] / float(h - 1)
        else:
            grid_y = grid[..., 1]
        sampling_grid = torch.stack([grid_x, grid_y], dim=-1)
        warped_logits = F.grid_sample(
            factual_logits, sampling_grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        identity_sample = F.grid_sample(
            factual_logits, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        warp_residual = warped_logits - identity_sample
        final_logits = factual_logits + support[:, None].to(factual_logits) * warp_residual
        final_prob = torch.sigmoid(final_logits).clamp(EPS, 1.0 - EPS)

        candidate_logits = torch.cat([factual_logits, final_logits], dim=1)
        candidate_probs = torch.cat([base_prob, final_prob], dim=1)
        flow_mag = torch.linalg.vector_norm(flow_px, dim=1)
        jac_mean, folding = self._flow_jacobian_stats(flow_px)
        change = (final_prob - base_prob).abs().mean(dim=(1, 2, 3))
        zero, one = base_prob.new_zeros(b), base_prob.new_ones(b)
        quality = base_prob.new_zeros((b, 2)); quality[:, 1] = 1.0
        base_min = factual_logits.flatten(1).min(1).values[:, None, None, None]
        base_max = factual_logits.flatten(1).max(1).values[:, None, None, None]
        overshoot = F.relu(final_logits - base_max) + F.relu(base_min - final_logits)
        support_fraction = support.float().mean((1, 2))
        owner_fraction = contour_owner.float().mean((1, 2))
        physical_abs = sample_offset_px.abs()

        aux: Dict[str, torch.Tensor] = {
            "candidates": candidate_logits,
            "candidate_probs": candidate_probs,
            "mhcs_final_logits": final_logits,
            "mhcs_final_probs": final_prob,
            "mhcs_local_probs": final_prob,
            "mhcs_surface_hard_probs": final_prob,
            "mhcs_global_selected_probs": final_prob,
            "mhcs_quality_probs": quality,
            "mhcs_quality_logits": torch.log(quality.clamp_min(EPS)),
            "mhcs_quality_pred": quality,
            "mhcs_global_weights": quality,
            "mhcs_gate_alpha": change,
            "mhcs_effective_rank": torch.where(change > EPS, one * 2.0, one),
            "mhcs_surface_nonbase_mass": change,
            "mhcs_m1_distribution_log_var": base_prob.new_zeros(()),
            "v20_selector_hard": base_prob.new_ones((b, 1)),
            "direct_fused_probs": final_prob,
            "router_fused_probs": final_prob,
            "v20_fused_probs": final_prob,
            "v20_hard_fused_probs": final_prob,
            "v20_fused_logits": final_logits,
            "geotopo_base_logits": factual_logits,
            "geotopo_base_probs": base_prob,
            "geotopo_geometry_flow_px": flow_px,
            "geotopo_geometry_logits": final_logits,
            "geotopo_geometry_probs": final_prob,
            "geotopo_residual_only_logits": factual_logits,
            "geotopo_residual_only_probs": base_prob,
            "geotopo_reconstruction_after_geometry_logits": final_logits,
            "geotopo_reconstruction_after_geometry_probs": final_prob,
            "geotopo_final_logits": final_logits,
            "geotopo_final_probs": final_prob,
            "geotopo_flow_rms_px": flow_mag.square().mean((1, 2)).sqrt(),
            "geotopo_flow_mean_px": flow_mag.mean((1, 2)),
            "geotopo_flow_max_px": flow_mag.flatten(1).max(1).values,
            "geotopo_flow_jacobian_mean": jac_mean,
            "geotopo_flow_folding_fraction": folding,
            "geotopo_geometry_abs_change": change,
            "geotopo_mode_id": one * 4.0,
            "geotr_m1_base_logits": factual_logits,
            "geotr_m1_final_logits": final_logits,
            "geotr_m1_final_probs": final_prob,
            "geotr_m1_flow_px": flow_px,
            "geotr_m1_boundary_normal": extended_normal,
            "geotr_m1_contour_owner_normal": base_normal,
            "geotr_m1_base_sdf": base_sdf,
            "geotr_m1_contour_owner_mask": contour_owner,
            "geotr_m1_base_boundary_mask": contour_owner,
            "geotr_m1_owner_distance_px": owner_distance,
            "geotr_m1_warp_support": support,
            "geotr_m1_predicted_owner_sample_offset_px": owner_scalar[:, 0],
            "geotr_m1_owner_scalar_field": owner_scalar,
            "geotr_m1_predicted_sample_offset_px": owner_scalar[:, 0],
            "geotr_m1_extended_sample_offset_px": extended_offset[:, 0],
            "geotr_m1_warped_logits": warped_logits,
            "geotr_m1_evidence_band": soft_boundary,
            "geotr_m1_scalar_field": extended_offset,
            "geotr_m1_regularized_field": flow_px,
            "geotr_m1_normal_field": extended_normal,
            "geotr_m1_rewrite_residual": final_logits - factual_logits,
            "geotr_m1_flow_scale_px": flow_mag.square().mean((1, 2)).sqrt().detach(),
            "geotr_m1_context_gate": zero,
            "geotr_m1_context_film_abs": zero,
            "geotr_m1_text_latent_abs": zero,
            "geotr_m1_anchor_boundary_fraction": owner_fraction,
            "geotr_m1_warp_support_fraction": support_fraction,
            "geotr_m1_conditioning_abs": zero,
            "geotr_m1_boundary_normal_warp": zero,
            "geotr_m1_sdf_operator_matched_warp": one,
            "geotr_m1_operator_id": one * 4.0,
            "geotr_m1_gate_mode_id": zero,
            "geotr_m1_is_normal_1d": one,
            "geotr_m1_deadzone_fraction": (physical_abs <= 1.0e-6).float().mean((1, 2, 3)),
            "geotr_m1_exact_identity_error": (final_logits - factual_logits).abs().mean((1, 2, 3)),
            "geotr_m1_tangent_energy_ratio": zero,
            "geotr_m1_range_violation_fraction": (overshoot > 2e-6).float().mean((1, 2, 3)),
            "geotr_m1_range_violation_max": overshoot.flatten(1).max(1).values,
            "geotr_m1_autozero": one,
            "geotr_m1_autozero_trust_mean": (physical_abs / float(self.local_radius)).mean((1, 2, 3)).detach(),
            "geotr_m1_autozero_trust_std": (physical_abs / float(self.local_radius)).flatten(1).std(1, unbiased=False).detach(),
            "geotr_m1_edit_probability_mean": (physical_abs / float(self.local_radius)).mean((1, 2, 3)).detach(),
            "geotr_m1_transition_band_fraction": support_fraction.detach(),
            "geotr_m1_local_radius_px": base_prob.new_tensor(float(self.local_radius)),
            "geotr_m1_memory_exact": one,
        }
        return candidate_logits, aux

    def _sample_along_normal(
        self,
        field: torch.Tensor,
        normal: torch.Tensor,
        offset_px: float,
    ) -> torch.Tensor:
        """Bilinearly sample a dense field at x + offset * n(x)."""
        b, _, h, w = field.shape
        grid = self._identity_grid(field)
        if w > 1:
            gx = grid[..., 0] + 2.0 * float(offset_px) * normal[:, 0] / float(w - 1)
        else:
            gx = grid[..., 0]
        if h > 1:
            gy = grid[..., 1] + 2.0 * float(offset_px) * normal[:, 1] / float(h - 1)
        else:
            gy = grid[..., 1]
        return F.grid_sample(
            field, torch.stack([gx, gy], dim=-1), mode="bilinear",
            padding_mode="border", align_corners=True,
        )

    @staticmethod
    def _uc_piecewise_convnormgelu(
        block: nn.Module,
        pieces: List[torch.Tensor],
    ) -> torch.Tensor:
        """Evaluate one ConvNormGELU without materialising channel concatenation.

        Conv2d is linear in its input-channel blocks:
            Conv(cat(x_i), W) == sum_i Conv(x_i, W_i).
        The GroupNorm and GELU are applied once after the exact channel-block
        accumulation, exactly where they appear in ConvNormGELU.  This removes
        multi-gigabyte [B,C,H,W] concatenation tensors at physical batch=24.
        """
        conv = block[0]
        norm = block[1]
        act = block[2]
        if not isinstance(conv, nn.Conv2d):
            raise TypeError("UC-FNRT piecewise fusion expects ConvNormGELU[0]=Conv2d")
        if int(conv.groups) != 1:
            raise ValueError("UC-FNRT piecewise fusion requires groups=1")

        offset = 0
        fused = None
        for piece in pieces:
            channels = int(piece.shape[1])
            if channels <= 0:
                continue
            end = offset + channels
            if end > int(conv.in_channels):
                raise RuntimeError(
                    f"UC-FNRT piecewise fusion channel overflow: {end}>{conv.in_channels}"
                )
            weight = conv.weight[:, offset:end]
            term = F.conv2d(
                piece,
                weight,
                bias=None,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=1,
            )
            fused = term if fused is None else fused + term
            offset = end

        if fused is None or offset != int(conv.in_channels):
            raise RuntimeError(
                "UC-FNRT piecewise fusion did not consume all input channels: "
                f"consumed={offset}, expected={conv.in_channels}"
            )
        if conv.bias is not None:
            fused = fused + conv.bias[None, :, None, None]
        return act(norm(fused))

    @staticmethod
    def _ds_channel_rms_normalize(x: torch.Tensor) -> torch.Tensor:
        """Per-pixel channel RMS normalization; invariant to positive feature scale."""
        # Compute the scale in FP32 for AMP stability, then cast back.  The
        # conditioner is detached by the UC-FNRT contract, so this does not add
        # a gradient path to the frozen semantic host.
        scale = x.float().square().mean(dim=1, keepdim=True).add(EPS).sqrt().to(x)
        return x / scale

    @staticmethod
    def _ds_symmetric_contrast(
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """Dimensionless bilateral contrast in [-1,1] up to EPS."""
        return (positive - negative) / (positive.abs() + negative.abs() + EPS)

    @staticmethod
    def _ds_relative_uncertainty_map(
        x: torch.Tensor,
        contour_support: torch.Tensor,
    ) -> torch.Tensor:
        """Case-relative robust posterior dispersion on the current Base contour.

        Center = median. Scale = 1.4826*MAD; if MAD degenerates (common for
        sparse disagreement maps), the per-case standard deviation is used as
        a parameter-free fallback.  The final z-score is smoothly bounded by
        z/(1+|z|), so no hand-tuned clipping threshold is introduced.

        Only the current prediction is used. No GT, target-domain aggregate
        statistic, validation statistic, or test-time adaptation is involved.
        """
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"relative uncertainty expects [B,1,H,W], got {tuple(x.shape)}")
        if contour_support.ndim == 3:
            contour_support = contour_support[:, None]
        if contour_support.shape != x.shape:
            raise ValueError(
                "contour support must match uncertainty map: "
                f"support={tuple(contour_support.shape)} x={tuple(x.shape)}"
            )
        outputs = []
        with torch.no_grad():
            for bi in range(x.shape[0]):
                values = x[bi, 0][contour_support[bi, 0].bool()]
                # Degenerate/empty contour cases are handled case-locally.  This
                # is important for BUSI/BTMRI empty predictions and never mixes
                # statistics across patients/cases.
                if values.numel() < 8:
                    values = x[bi, 0].reshape(-1)
                values32 = values.float()
                med = values32.median()
                mad_scale = 1.4826 * (values32 - med).abs().median()
                std_scale = values32.std(unbiased=False)
                scale = torch.maximum(mad_scale, std_scale).clamp_min(EPS)
                z = (x[bi:bi + 1].float() - med) / scale
                bounded = z / (1.0 + z.abs())
                outputs.append(bounded.to(x))
        return torch.cat(outputs, dim=0)

    def _ds_ray_scale_agreement(
        self,
        base_prob: torch.Tensor,
        base_normal: torch.Tensor,
    ) -> torch.Tensor:
        """Parameter-free multi-scale ray coherence in [0,1].

        At each already-declared radius (1,2,4,r), use a symmetric probability
        contrast. The magnitude of the mean contrast jointly reflects directional
        agreement and evidence strength: contradictory/weak scales contract the
        final offset toward exact KEEP; consistent strong scales leave it active.
        """
        contrasts = []
        for radius in (1.0, 2.0, 4.0, float(self.local_radius)):
            positive = self._sample_along_normal(base_prob, base_normal, radius)
            negative = self._sample_along_normal(base_prob, base_normal, -radius)
            contrasts.append(self._ds_symmetric_contrast(positive, negative))
        coherence = torch.stack(contrasts, dim=0).mean(dim=0).abs().clamp(0.0, 1.0)
        return coherence.detach()

    @staticmethod
    def _hrcv_gather_points(field: torch.Tensor, flat_index: torch.Tensor) -> torch.Tensor:
        """Gather dense [B,C,H,W] at padded flat owner indices -> [B,N,C]."""
        flat = field.flatten(2)
        idx = flat_index[:, None].expand(-1, flat.shape[1], -1)
        return flat.gather(2, idx).transpose(1, 2)

    def _hrcv_sample_owner_points(
        self,
        field: torch.Tensor,
        owner_flat_index: torch.Tensor,
        owner_valid: torch.Tensor,
        owner_normal: torch.Tensor,
        offset_px: float,
    ) -> torch.Tensor:
        """Bilinearly sample a dense field at sparse owner x+offset*n(x)."""
        b, _, h, w = field.shape
        idx = owner_flat_index
        y = torch.div(idx, w, rounding_mode="floor").to(field.dtype)
        x = (idx % w).to(field.dtype)
        if w > 1:
            gx = 2.0 * x / float(w - 1) - 1.0
            gx = gx + 2.0 * float(offset_px) * owner_normal[..., 0] / float(w - 1)
        else:
            gx = torch.zeros_like(x)
        if h > 1:
            gy = 2.0 * y / float(h - 1) - 1.0
            gy = gy + 2.0 * float(offset_px) * owner_normal[..., 1] / float(h - 1)
        else:
            gy = torch.zeros_like(y)
        grid = torch.stack([gx, gy], dim=-1)[:, :, None, :]
        sampled = F.grid_sample(
            field, grid, mode="bilinear", padding_mode="border", align_corners=True
        )[:, :, :, 0].transpose(1, 2)
        return sampled * owner_valid[..., None].to(sampled)

    def _uc_candidate_evidence_offset_px(self, transport_offset_px: float) -> float:
        """Map an inverse-warp displacement bin to its image evidence location.

        Let the true boundary move by ``m`` along the Base normal.  The deployed
        grid sampler uses ``d=-m`` because the output at x reads the old logit at
        x+d*n.  Consequently, the image/semantic boundary that explains bin d is
        located at ``m=-d``.  Keeping this conversion explicit prevents the two
        coordinate systems from being silently conflated again.
        """
        d = float(transport_offset_px)
        return -d if self.uc_operator_aligned_candidates else d

    def _uc_hrcv_logits_memory_bounded(
        self,
        image_feature: torch.Tensor,
        semantic_feature: torch.Tensor,
        base_prob: torch.Tensor,
        margin_uncertainty: torch.Tensor,
        mc_std_map: torch.Tensor,
        mc_disagreement_map: torch.Tensor,
        soft_boundary: torch.Tensor,
        base_normal: torch.Tensor,
        contour_owner_float: torch.Tensor,
    ) -> torch.Tensor:
        """Sparse ordered normal-ray matching volume.

        Legacy HRCV scores each signed candidate independently.  MRM keeps the
        same sparse owner-only memory contract but can replace the unary score
        with a true relational cost: candidate evidence is expressed relative
        to the current Base-contour evidence, then neighbouring signed offsets
        are aggregated along the physically ordered k axis before scoring.

        H0/H1 and MRM controls share the same registered modules.  When MRM is
        disabled this function is mathematically identical to the audited HRCV
        unary scorer apart from the presence of inactive registered parameters.
        """
        if not self.uc_hrcv_registered:
            raise RuntimeError("HRCV logits requested without registered modules")
        if self.uc_hrcv_context is None or self.uc_hrcv_cue_proj is None or self.uc_hrcv_scorer is None:
            raise RuntimeError("HRCV module registration is incomplete")
        if self.uc_mrm_registered and (
            self.uc_mrm_rel_embed is None
            or self.uc_mrm_unary_head is None
            or self.uc_mrm_aggregate is None
            or self.uc_mrm_head is None
        ):
            raise RuntimeError("MRM module registration is incomplete")

        owner = contour_owner_float[:, 0] > 0.5
        b, h, w = owner.shape
        counts = owner.flatten(1).sum(dim=1).long()
        max_n = int(counts.max().item()) if counts.numel() else 0
        k_count = 2 * int(self.local_radius) + 1
        if max_n <= 0:
            return base_prob.new_zeros((b, k_count, h, w))

        owner_flat_index = torch.zeros((b, max_n), device=owner.device, dtype=torch.long)
        owner_valid = torch.zeros((b, max_n), device=owner.device, dtype=torch.bool)
        for bi in range(b):
            ids = owner[bi].reshape(-1).nonzero(as_tuple=False).flatten()
            n = int(ids.numel())
            if n > 0:
                owner_flat_index[bi, :n] = ids
                owner_valid[bi, :n] = True

        normal_points = self._hrcv_gather_points(base_normal, owner_flat_index)
        normal_points = normal_points * owner_valid[..., None].to(normal_points)

        context = self._uc_piecewise_convnormgelu(
            self.uc_hrcv_context,
            [
                image_feature, semantic_feature, base_prob, margin_uncertainty,
                mc_std_map, mc_disagreement_map, soft_boundary,
            ],
        )
        context_points = self._hrcv_gather_points(context, owner_flat_index)
        context_points = context_points * owner_valid[..., None].to(context_points)

        ray_image = self.uc_ray_image_proj(image_feature)
        ray_semantic = self.uc_ray_semantic_proj(semantic_feature)
        cue_source = torch.cat(
            [base_prob, margin_uncertainty, mc_std_map, mc_disagreement_map, soft_boundary],
            dim=1,
        )
        ray_cue = self.uc_hrcv_cue_proj(cue_source)

        radius = int(self.local_radius)
        delta = float(self.uc_hrcv_delta_px)

        # Current contour acts as the query/anchor for relational matching.
        # These are computed once, not independently re-learned for each k.
        if self.uc_mrm_relational_cost:
            anchor_lo = max(-float(radius), -delta)
            anchor_hi = min(float(radius), +delta)
            anchor_span = max(anchor_hi - anchor_lo, EPS)
            img_a = self._hrcv_sample_owner_points(
                ray_image, owner_flat_index, owner_valid, normal_points, 0.0
            )
            img_a_lo = self._hrcv_sample_owner_points(
                ray_image, owner_flat_index, owner_valid, normal_points, anchor_lo
            )
            img_a_hi = self._hrcv_sample_owner_points(
                ray_image, owner_flat_index, owner_valid, normal_points, anchor_hi
            )
            img_a_d = (img_a_hi - img_a_lo) / float(anchor_span)

            sem_a = self._hrcv_sample_owner_points(
                ray_semantic, owner_flat_index, owner_valid, normal_points, 0.0
            )
            sem_a_lo = self._hrcv_sample_owner_points(
                ray_semantic, owner_flat_index, owner_valid, normal_points, anchor_lo
            )
            sem_a_hi = self._hrcv_sample_owner_points(
                ray_semantic, owner_flat_index, owner_valid, normal_points, anchor_hi
            )
            sem_a_d = (sem_a_hi - sem_a_lo) / float(anchor_span)

            cue_a = self._hrcv_sample_owner_points(
                ray_cue, owner_flat_index, owner_valid, normal_points, 0.0
            )

        score_bank = []
        embed_bank = []
        for k in range(-radius, radius + 1):
            transport_k = float(k)
            evidence_k = (
                self._uc_candidate_evidence_offset_px(transport_k)
                if self.uc_hrcv_candidate_conditioned
                else 0.0
            )
            lo = max(-float(radius), evidence_k - delta)
            hi = min(float(radius), evidence_k + delta)
            span = max(hi - lo, EPS)

            img_c = self._hrcv_sample_owner_points(
                ray_image, owner_flat_index, owner_valid, normal_points, evidence_k
            )
            img_lo = self._hrcv_sample_owner_points(
                ray_image, owner_flat_index, owner_valid, normal_points, lo
            )
            img_hi = self._hrcv_sample_owner_points(
                ray_image, owner_flat_index, owner_valid, normal_points, hi
            )
            img_d = (img_hi - img_lo) / float(span)

            sem_c = self._hrcv_sample_owner_points(
                ray_semantic, owner_flat_index, owner_valid, normal_points, evidence_k
            )
            sem_lo = self._hrcv_sample_owner_points(
                ray_semantic, owner_flat_index, owner_valid, normal_points, lo
            )
            sem_hi = self._hrcv_sample_owner_points(
                ray_semantic, owner_flat_index, owner_valid, normal_points, hi
            )
            sem_d = (sem_hi - sem_lo) / float(span)

            cue_c = self._hrcv_sample_owner_points(
                ray_cue, owner_flat_index, owner_valid, normal_points, evidence_k
            )
            cue_lo = self._hrcv_sample_owner_points(
                ray_cue, owner_flat_index, owner_valid, normal_points, lo
            )
            cue_hi = self._hrcv_sample_owner_points(
                ray_cue, owner_flat_index, owner_valid, normal_points, hi
            )
            cue_d = (cue_hi - cue_lo) / float(span)

            coord = base_prob.new_full(
                (b, max_n, 1), transport_k / float(max(radius, 1))
            )

            if self.uc_mrm_relational_cost:
                # Explicit query<->candidate relations turn the old unary
                # "does k look boundary-like?" classifier into a matching cost.
                img_delta = img_c - img_a
                sem_delta = sem_c - sem_a
                cue_delta = cue_c - cue_a
                img_corr = F.cosine_similarity(img_a, img_c, dim=-1, eps=EPS)[..., None]
                sem_corr = F.cosine_similarity(sem_a, sem_c, dim=-1, eps=EPS)[..., None]
                descriptor = torch.cat(
                    [
                        context_points,
                        img_delta, img_d - img_a_d,
                        sem_delta, sem_d - sem_a_d,
                        cue_delta, cue_d,
                        img_corr, sem_corr, coord,
                    ],
                    dim=-1,
                )
                embed = self.uc_mrm_rel_embed(descriptor)
                embed = embed * owner_valid[..., None].to(embed)
                embed_bank.append(embed)
            else:
                descriptor = torch.cat(
                    [context_points, img_c, img_d, sem_c, sem_d, cue_c, cue_d, coord],
                    dim=-1,
                )
                score = self.uc_hrcv_scorer(descriptor)[..., 0]
                score = score * owner_valid.to(score)
                score_bank.append(score)

        if self.uc_mrm_relational_cost:
            # [B,N,K,D] -> every contour owner is one independent 1-D ordered
            # matching problem.  Convolution only mixes neighbouring k values;
            # it never mixes owners/cases or spatially unrelated pixels.
            owner_embed = torch.stack(embed_bank, dim=2)
            if self.uc_mrm_ordered_aggregation:
                bn = b * max_n
                x = owner_embed.permute(0, 1, 3, 2).reshape(
                    bn, int(self.uc_hrcv_cost_dim), k_count
                )
                x = x + self.uc_mrm_aggregate(x)
                score = self.uc_mrm_head(x)[:, 0]
                owner_scores = score.reshape(b, max_n, k_count).permute(0, 2, 1)
            else:
                score = self.uc_mrm_unary_head(owner_embed)[..., 0]
                owner_scores = score.permute(0, 2, 1)
            owner_scores = owner_scores * owner_valid[:, None].to(owner_scores)
        else:
            owner_scores = torch.stack(score_bank, dim=1)  # [B,K,N]

        # Owner indices are unique inside each case.  Use index_copy on only the
        # valid sparse owners rather than scatter_add over padded duplicate index
        # zero.  Besides being mathematically exact, this avoids CUDA atomic-add
        # nondeterminism in strict causal-parity runs.
        dense_cases = []
        for bi in range(b):
            valid_b = owner_valid[bi]
            ids_b = owner_flat_index[bi, valid_b]
            values_b = owner_scores[bi, :, valid_b]
            dense_b = base_prob.new_zeros((k_count, h * w))
            if int(ids_b.numel()) > 0:
                dense_b = dense_b.index_copy(1, ids_b, values_b)
            dense_cases.append(dense_b)
        return torch.stack(dense_cases, dim=0).view(b, k_count, h, w)

    def _uc_dominant_mode_decode(
        self,
        offset_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode the locally dominant ordered mode without global cancellation.

        The full posterior mean is biased toward zero when positive and negative
        ray modes coexist.  We locate the mode with the largest *local average
        mass* and then take a differentiable weighted average only inside that
        contiguous mode.  A tiny deterministic center preference resolves exact
        ties in favour of KEEP; uniform zero-init logits therefore decode to
        exactly zero, preserving AutoZero identity.
        """
        if offset_probs.ndim != 4:
            raise ValueError("offset_probs must be [B,K,H,W]")
        b, k, h, w = offset_probs.shape
        if k != 2 * int(self.local_radius) + 1:
            raise ValueError("dominant-mode decoder received an unexpected bin count")
        mode_radius = int(self.uc_mrm_mode_radius)
        flat = offset_probs.permute(0, 2, 3, 1).reshape(-1, 1, k)
        kernel = 2 * mode_radius + 1
        # DME chooses the mode with maximum *cumulative* probability mass.
        # Use a fixed ones kernel rather than avg_pool1d: count-normalized
        # pooling would give edge windows an artificial advantage because they
        # contain fewer valid bins.  This operator is parameter-free.
        local_sum = F.conv1d(
            flat,
            flat.new_ones((1, 1, kernel)),
            stride=1,
            padding=mode_radius,
        )
        idx = torch.arange(k, device=offset_probs.device, dtype=offset_probs.dtype)
        center = float(self.local_radius)
        tie_bias = (1.0 - (idx - center).abs() / float(self.local_radius + 1)) * 1.0e-7
        winner = (local_sum[:, 0] + tie_bias[None]).argmax(dim=1)

        distance = (idx[None, :] - winner[:, None].to(idx.dtype)).abs()
        modal_mask = (distance <= float(mode_radius)).to(flat)
        modal_prob = flat[:, 0] * modal_mask
        modal_mass = modal_prob.sum(dim=1).clamp_min(EPS)
        values = idx - center
        modal_offset = (modal_prob * values[None]).sum(dim=1) / modal_mass
        winner_offset = values[winner]

        modal_offset = modal_offset.view(b, h, w)[:, None]
        winner_offset = winner_offset.view(b, h, w)[:, None]
        modal_mass = modal_mass.view(b, h, w)[:, None]
        return modal_offset, winner_offset, modal_mass

    def _uc_hierarchical_confidence_decode(
        self,
        offset_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode sign first and magnitude conditional on that sign.

        A global expectation ``E[D]`` is the Bayes action for squared offset
        error, but it is a poor physical action when the posterior has modes on
        both sides of the contour: the two valid-looking motions erase each
        other.  R3 instead uses the Bayes class decision over three mutually
        exclusive actions (negative, KEEP, positive) and the conditional mean
        magnitude inside the selected non-KEEP class.

        The forward decision is deliberately conservative.  Its dose is the
        normalized winning margin ``(p1-p2)/(1-p2)``.  Equal or near-equal sign
        evidence produces zero or a small edit, while a decisive sign can use
        the full learned conditional magnitude.  This removes cross-sign
        cancellation without turning a tiny noisy bin-level peak into a full
        displacement.

        Returns
        -------
        deployed_offset_px : confidence-gated signed conditional magnitude
        conditional_offset : ungated signed conditional magnitude
        winning_sign_mass  : posterior mass of the chosen sign group
        sign_confidence    : normalized top-two sign margin in [0,1]
        """
        if offset_probs.ndim != 4:
            raise ValueError("offset_probs must be [B,K,H,W]")
        b, k, h, w = offset_probs.shape
        radius = int(self.local_radius)
        if k != 2 * radius + 1:
            raise ValueError(
                "hierarchical confidence decoder received an unexpected bin count"
            )

        negative = torch.flip(offset_probs[:, :radius], dims=(1,))
        keep = offset_probs[:, radius : radius + 1]
        positive = offset_probs[:, radius + 1 :]
        negative_mass = negative.sum(dim=1, keepdim=True)
        positive_mass = positive.sum(dim=1, keepdim=True)
        sign_probs = torch.cat([negative_mass, keep, positive_mass], dim=1)

        magnitude_values = torch.arange(
            1,
            radius + 1,
            device=offset_probs.device,
            dtype=offset_probs.dtype,
        ).view(1, radius, 1, 1)
        negative_mean = (
            (negative * magnitude_values).sum(dim=1, keepdim=True)
            / negative_mass.clamp_min(EPS)
        )
        positive_mean = (
            (positive * magnitude_values).sum(dim=1, keepdim=True)
            / positive_mass.clamp_min(EPS)
        )

        # Exact ties choose KEEP.  The bias is used only by argmax and is far
        # below the probability resolution that affects the reported scores.
        tie_bias = sign_probs.new_tensor([0.0, 2.0e-7, 1.0e-7]).view(1, 3, 1, 1)
        winner = (sign_probs + tie_bias).argmax(dim=1, keepdim=True)
        winner_onehot = F.one_hot(winner[:, 0], num_classes=3).permute(0, 3, 1, 2)
        winner_onehot = winner_onehot.to(offset_probs)

        conditional_offset = (
            -winner_onehot[:, 0:1] * negative_mean
            + winner_onehot[:, 2:3] * positive_mean
        )
        top2 = sign_probs.topk(k=2, dim=1).values
        sign_confidence = (
            (top2[:, 0:1] - top2[:, 1:2])
            / (1.0 - top2[:, 1:2]).clamp_min(EPS)
        ).clamp(0.0, 1.0)
        winning_sign_mass = (winner_onehot * sign_probs).sum(dim=1, keepdim=True)
        if self.uc_hierarchical_execution == "full_conditional":
            # Diagnostic / R4 execution: confidence decides the sign class but
            # is not reinterpreted as a physical distance scale.  KEEP remains
            # exact because conditional_offset is exactly zero when KEEP wins.
            deployed = conditional_offset
        else:
            # R3 reference execution: normalized sign margin attenuates dose.
            deployed = conditional_offset * sign_confidence
        return deployed, conditional_offset, winning_sign_mass, sign_confidence

    def _uc_factor_raw_memory_exact(
        self,
        image_feature: torch.Tensor,
        semantic_feature: torch.Tensor,
        base_prob: torch.Tensor,
        margin_uncertainty: torch.Tensor,
        mc_std_map: torch.Tensor,
        mc_disagreement_map: torch.Tensor,
        soft_boundary: torch.Tensor,
        base_normal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Memory-bounded UC-FNRT predictor with the same mathematical operator.

        The old implementation materialised:
          * a 517-channel context concatenation, and
          * a 576-channel context+ray concatenation.
        At B=24,H=W=224 those tensors alone are multi-GB FP32 allocations.

        R2 exploits the linearity of the first convolution in each fusion block:
        input-channel blocks are convolved against the corresponding weight
        slices and accumulated before the unchanged GroupNorm/GELU.  Ray terms
        are produced and consumed one at a time, so the 12 dense ray tensors
        are never simultaneously resident.
        """
        context = self._uc_piecewise_convnormgelu(
            self.uc_context_fuse[0],
            [
                image_feature,
                semantic_feature,
                base_prob,
                margin_uncertainty,
                mc_std_map,
                mc_disagreement_map,
                soft_boundary,
            ],
        )
        context = self.uc_context_fuse[1](context)

        cue_source = torch.cat(
            [base_prob, mc_std_map, mc_disagreement_map], dim=1
        )
        ray_image = self.uc_ray_image_proj(image_feature)
        ray_semantic = self.uc_ray_semantic_proj(semantic_feature)
        ray_cue = self.uc_ray_cue_proj(cue_source)
        if self.ds_normalized_ray:
            # Normalize only the compact ray evidence; the archived context
            # representation and all learned layers remain unchanged.
            ray_image = self._ds_channel_rms_normalize(ray_image)
            ray_semantic = self._ds_channel_rms_normalize(ray_semantic)
            ray_cue = self._ds_channel_rms_normalize(ray_cue)

        first = self.uc_ray_fuse[0]
        conv = first[0]
        norm = first[1]
        act = first[2]
        if not isinstance(conv, nn.Conv2d) or int(conv.groups) != 1:
            raise RuntimeError("UC-FNRT ray fusion requires a dense Conv2d")

        # Consume context first, then each bilateral ray term in the exact
        # channel order used by torch.cat([context] + ray_terms, dim=1).
        offset = 0
        c = int(context.shape[1])
        hidden_pre = F.conv2d(
            context,
            conv.weight[:, offset:offset + c],
            bias=None,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=1,
        )
        offset += c

        for radius in (1.0, 2.0, 4.0, float(self.local_radius)):
            for source in (ray_image, ray_semantic, ray_cue):
                positive = self._sample_along_normal(source, base_normal, radius)
                negative = self._sample_along_normal(source, base_normal, -radius)
                if self.ds_normalized_ray:
                    # Dimensionless relative evidence is substantially less
                    # sensitive to scanner/illumination/feature amplitude shift.
                    delta = self._ds_symmetric_contrast(positive, negative)
                else:
                    delta = positive - negative
                # R-control: zero only bilateral normal-ray contrast while
                # preserving channel layout, parameter count and compute path.
                if self.uc_ablation == "no_normal_ray_evidence":
                    delta = delta * 0.0
                c = int(delta.shape[1])
                hidden_pre = hidden_pre + F.conv2d(
                    delta,
                    conv.weight[:, offset:offset + c],
                    bias=None,
                    stride=conv.stride,
                    padding=conv.padding,
                    dilation=conv.dilation,
                    groups=1,
                )
                offset += c
                del delta

        if offset != int(conv.in_channels):
            raise RuntimeError(
                "UC-FNRT ray-fusion channel contract failed: "
                f"consumed={offset}, expected={conv.in_channels}"
            )
        if conv.bias is not None:
            hidden_pre = hidden_pre + conv.bias[None, :, None, None]

        hidden = act(norm(hidden_pre))
        hidden = self.uc_ray_fuse[1](hidden)
        direction_raw = self.direction_controller(hidden)
        magnitude_raw = self.magnitude_controller(hidden)
        if self.uc_offset_head_registered:
            assert self.offset_controller is not None
            offset_logits = self.offset_controller(hidden)
        else:
            # One-channel zero dummy keeps the checkpoint return signature
            # tensor-only without allocating a 17-channel inactive volume.
            offset_logits = hidden[:, :1] * 0.0
        return direction_raw, magnitude_raw, offset_logits

    def _generate_uc_fnrt(
        self,
        factual_logits: torch.Tensor,
        base_prob: torch.Tensor,
        image_feature: torch.Tensor,
        semantic_feature: torch.Tensor,
        soft_boundary: torch.Tensor,
        margin_uncertainty: torch.Tensor,
        mc_std_map: torch.Tensor,
        mc_disagreement_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Uncertainty-conditioned factorized normal-ray transport (UC-FNRT).

        Geometry remains the verified PS-OMW operator.  The predictor is changed:
        posterior uncertainty is retained, bilateral semantic evidence is sampled
        on the Base normal ray, and signed displacement is factorized into a
        direction confidence and a non-negative magnitude.  Zero direction
        confidence is an exact structural identity.
        """
        b, _, h, w = factual_logits.shape
        base_hard = base_prob >= 0.5
        base_sdf = binary_signed_distance(base_hard, dtype=factual_logits.dtype)
        base_normal, normal_valid = sdf_normal(base_sdf)
        contour_owner = foreground_contour_owner(base_hard) & normal_valid
        owner_distance, owner_flat_index, owner_case_valid = nearest_owner_map(
            contour_owner, dtype=factual_logits.dtype
        )

        # U-control: retain the same MC posterior mean and MC10/MC30 compute,
        # but mask posterior dispersion channels seen by the predictor.  The
        # deterministic margin uncertainty 4p(1-p) remains available.
        effective_mc_std_map = mc_std_map
        effective_mc_disagreement_map = mc_disagreement_map
        if self.uc_ablation == "no_posterior_uncertainty":
            effective_mc_std_map = mc_std_map * 0.0
            effective_mc_disagreement_map = mc_disagreement_map * 0.0
        elif self.ds_relative_uncertainty:
            contour_support = contour_owner[:, None].to(dtype=torch.bool)
            effective_mc_std_map = self._ds_relative_uncertainty_map(
                mc_std_map, contour_support
            )
            effective_mc_disagreement_map = self._ds_relative_uncertainty_map(
                mc_disagreement_map, contour_support
            )

        # R2 memory-exact predictor.  The whole dense UC predictor is
        # activation-checkpointed, and the two wide channel concatenations are
        # evaluated by exact channel-block convolution accumulation.
        if self.training and torch.is_grad_enabled():
            direction_raw, magnitude_raw, offset_logits = checkpoint(
                self._uc_factor_raw_memory_exact,
                image_feature,
                semantic_feature,
                base_prob,
                margin_uncertainty,
                effective_mc_std_map,
                effective_mc_disagreement_map,
                soft_boundary,
                base_normal,
                use_reentrant=False,
            )
        else:
            direction_raw, magnitude_raw, offset_logits = self._uc_factor_raw_memory_exact(
                image_feature,
                semantic_feature,
                base_prob,
                margin_uncertainty,
                effective_mc_std_map,
                effective_mc_disagreement_map,
                soft_boundary,
                base_normal,
            )
        global_expectation_px = None
        dominant_center_px = None
        dominant_mass = None
        hierarchical_conditional_offset_px = None
        hierarchical_sign_confidence = None
        hierarchical_winning_sign_mass = None
        if self.uc_offset_distribution:
            if self.offset_controller is None:
                raise RuntimeError(
                    "Distributional offset transport is enabled without an offset head."
                )
            if self.uc_hrcv:
                # HRCV is separately checkpointed so the 17 candidate descriptors
                # are not retained for backward at physical batch=24.  The output
                # is only [B,17,H,W].  H0/H1 differ only in candidate sampling.
                if self.training and torch.is_grad_enabled():
                    offset_logits = checkpoint(
                        self._uc_hrcv_logits_memory_bounded,
                        image_feature, semantic_feature, base_prob, margin_uncertainty,
                        effective_mc_std_map, effective_mc_disagreement_map,
                        soft_boundary, base_normal, contour_owner[:, None].to(base_prob),
                        use_reentrant=False,
                    )
                else:
                    offset_logits = self._uc_hrcv_logits_memory_bounded(
                        image_feature, semantic_feature, base_prob, margin_uncertainty,
                        effective_mc_std_map, effective_mc_disagreement_map,
                        soft_boundary, base_normal, contour_owner[:, None].to(base_prob),
                    )
            expected_channels = 2 * self.local_radius + 1
            if int(offset_logits.shape[1]) != expected_channels:
                raise RuntimeError(
                    "DNR offset-logit channel mismatch: "
                    f"got={int(offset_logits.shape[1])}, expected={expected_channels}"
                )
            if self.uc_radius_balanced_keep_prior:
                # A zero-initialized flat K-bin head otherwise assigns only
                # 1/(2R+1) probability to KEEP but R/(2R+1) to each direction.
                # That radius-dependent implicit edit prior gets worse when R
                # is enlarged.  Adding log(R) to the single KEEP bin makes the
                # three sign groups exactly equiprobable under equal evidence:
                #   negative mass = R, KEEP mass = R, positive mass = R.
                # This is a structural prior, not a learned/dataset threshold.
                keep_prior = offset_logits.new_zeros((1, expected_channels, 1, 1))
                keep_prior[:, self.local_radius] = math.log(float(self.local_radius))
                offset_logits = offset_logits + keep_prior
            offset_probs = F.softmax(offset_logits.float(), dim=1).to(offset_logits)
            offset_values = self.uc_offset_values_px.to(
                device=offset_probs.device, dtype=offset_probs.dtype
            ).view(1, expected_channels, 1, 1)

            # One distribution owns sign, distance and KEEP jointly.  Keep the
            # legacy full posterior mean as an explicit diagnostic because it
            # quantifies positive/negative mode cancellation.
            pos_probs = offset_probs[:, self.local_radius + 1 :]
            neg_probs = torch.flip(
                offset_probs[:, : self.local_radius], dims=(1,)
            )
            positive_values = torch.arange(
                1,
                self.local_radius + 1,
                device=offset_probs.device,
                dtype=offset_probs.dtype,
            ).view(1, self.local_radius, 1, 1)
            global_expectation_px = (
                (pos_probs - neg_probs) * positive_values
            ).sum(dim=1, keepdim=True)
            magnitude_px = (
                (pos_probs + neg_probs) * positive_values
            ).sum(dim=1, keepdim=True)
            p_keep = offset_probs[:, self.local_radius : self.local_radius + 1]
            p_pos = pos_probs.sum(dim=1, keepdim=True)
            p_neg = neg_probs.sum(dim=1, keepdim=True)
            direction_conf = (p_pos - p_neg).clamp(-1.0, 1.0)

            if self.uc_hierarchical_confidence_decoder:
                (
                    owner_scalar,
                    hierarchical_conditional_offset_px,
                    hierarchical_winning_sign_mass,
                    hierarchical_sign_confidence,
                ) = self._uc_hierarchical_confidence_decode(offset_probs)
                dominant_center_px = hierarchical_conditional_offset_px
                dominant_mass = hierarchical_winning_sign_mass
            elif self.uc_mrm_dominant_mode:
                owner_scalar, dominant_center_px, dominant_mass = (
                    self._uc_dominant_mode_decode(offset_probs)
                )
            else:
                owner_scalar = global_expectation_px
                dominant_center_px = torch.zeros_like(owner_scalar)
                dominant_mass = torch.ones_like(owner_scalar)

            # Diagnostic log-odds only.  Deployment is either E[d] (controls)
            # or the dominant contiguous mode (MRM-DME).
            direction_raw_effective = torch.log(
                (p_pos + 0.5 * p_keep).clamp_min(EPS)
            ) - torch.log((p_neg + 0.5 * p_keep).clamp_min(EPS))
        else:
            offset_probs = None
            p_keep = None
            direction_conf = torch.tanh(direction_raw)
            direction_raw_effective = direction_raw
            if self.uc_ablation == "direct_signed":
                # F-control: remove direction×magnitude factorization.  The same
                # signed normal-ray operator is driven by d=r*tanh(raw).  The
                # magnitude head stays registered so construction/RNG/parameter
                # count are identical, but it has no causal path to deployment.
                owner_scalar = float(self.local_radius) * direction_conf
                magnitude_px = owner_scalar.abs()
            else:
                magnitude_px = float(self.local_radius) * torch.sigmoid(magnitude_raw)
                owner_scalar = direction_conf * magnitude_px
        # Keep the pre-agreement physical prediction explicit.  This makes the
        # full realization chain auditable:
        # target -> magnitude -> direction*distance -> optional agreement dose.
        # The A2 candidate does NOT apply agreement as a physical dose; it only
        # measures it.  The archived Stage-A behaviour remains reproducible by
        # setting SEMLT_DS_RAY_AGREEMENT_APPLY=true.
        preagreement_owner_scalar = owner_scalar
        if self.ds_ray_agreement and self.uc_ablation != "no_normal_ray_evidence":
            ray_agreement = self._ds_ray_scale_agreement(base_prob, base_normal)
        else:
            ray_agreement = torch.ones_like(owner_scalar)
        if self.ds_ray_agreement_apply and self.uc_ablation != "no_normal_ray_evidence":
            owner_scalar = preagreement_owner_scalar * ray_agreement
        else:
            owner_scalar = preagreement_owner_scalar
        if self.force_preserve:
            owner_scalar = torch.zeros_like(owner_scalar)

        extended_offset = gather_owner_field(owner_scalar, owner_flat_index, owner_case_valid)
        extended_normal = gather_owner_field(base_normal.detach(), owner_flat_index, owner_case_valid)
        nmag = torch.linalg.vector_norm(extended_normal, dim=1, keepdim=True).clamp_min(EPS)
        extended_normal = torch.where(nmag > 1.0e-5, extended_normal / nmag, torch.zeros_like(extended_normal))

        band_radius = float(self.local_radius + 1)
        finite_band = (
            owner_case_valid[:, None, None]
            & torch.isfinite(owner_distance)
            & (owner_distance <= band_radius)
        )
        phase = (owner_distance / max(band_radius, 1.0)).clamp(0.0, 1.0)
        band_weight = 0.5 * (1.0 + torch.cos(torch.pi * phase))
        band_weight = torch.where(finite_band, band_weight, torch.zeros_like(band_weight))

        sample_offset_px = extended_offset * band_weight[:, None]
        flow_px = sample_offset_px * extended_normal
        grid = self._identity_grid(factual_logits)
        gx = grid[..., 0] + (2.0 * flow_px[:, 0] / float(w - 1) if w > 1 else 0.0)
        gy = grid[..., 1] + (2.0 * flow_px[:, 1] / float(h - 1) if h > 1 else 0.0)
        sampling_grid = torch.stack([gx, gy], dim=-1)
        warped_logits = F.grid_sample(
            factual_logits, sampling_grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        identity_sample = F.grid_sample(
            factual_logits, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        final_logits = factual_logits + (warped_logits - identity_sample)
        final_prob = torch.sigmoid(final_logits).clamp(EPS, 1.0 - EPS)

        candidate_logits = torch.cat([factual_logits, final_logits], dim=1)
        candidate_probs = torch.cat([base_prob, final_prob], dim=1)
        flow_mag = torch.linalg.vector_norm(flow_px, dim=1)
        jac_mean, folding = self._flow_jacobian_stats(flow_px)
        change = (final_prob - base_prob).abs().mean(dim=(1, 2, 3))
        zero, one = base_prob.new_zeros(b), base_prob.new_ones(b)
        quality = base_prob.new_zeros((b, 2)); quality[:, 1] = 1.0
        base_min = factual_logits.flatten(1).min(1).values[:, None, None, None]
        base_max = factual_logits.flatten(1).max(1).values[:, None, None, None]
        overshoot = F.relu(final_logits - base_max) + F.relu(base_min - final_logits)
        band_fraction = finite_band.float().mean((1, 2))
        weighted_band_fraction = band_weight.mean((1, 2))
        owner_fraction = contour_owner.float().mean((1, 2))
        physical_abs = sample_offset_px.abs()

        aux: Dict[str, torch.Tensor] = {
            "candidates": candidate_logits,
            "candidate_probs": candidate_probs,
            "mhcs_final_logits": final_logits,
            "mhcs_final_probs": final_prob,
            "mhcs_local_probs": final_prob,
            "mhcs_surface_hard_probs": final_prob,
            "mhcs_global_selected_probs": final_prob,
            "mhcs_quality_probs": quality,
            "mhcs_quality_logits": torch.log(quality.clamp_min(EPS)),
            "mhcs_quality_pred": quality,
            "mhcs_global_weights": quality,
            "mhcs_gate_alpha": change,
            "mhcs_effective_rank": torch.where(change > EPS, one * 2.0, one),
            "mhcs_surface_nonbase_mass": change,
            "mhcs_m1_distribution_log_var": base_prob.new_zeros(()),
            "v20_selector_hard": base_prob.new_ones((b, 1)),
            "direct_fused_probs": final_prob,
            "router_fused_probs": final_prob,
            "v20_fused_probs": final_prob,
            "v20_hard_fused_probs": final_prob,
            "v20_fused_logits": final_logits,
            "geotopo_base_logits": factual_logits,
            "geotopo_base_probs": base_prob,
            "geotopo_geometry_flow_px": flow_px,
            "geotopo_geometry_logits": final_logits,
            "geotopo_geometry_probs": final_prob,
            "geotopo_residual_only_logits": factual_logits,
            "geotopo_residual_only_probs": base_prob,
            "geotopo_reconstruction_after_geometry_logits": final_logits,
            "geotopo_reconstruction_after_geometry_probs": final_prob,
            "geotopo_final_logits": final_logits,
            "geotopo_final_probs": final_prob,
            "geotopo_flow_rms_px": flow_mag.square().mean((1, 2)).sqrt(),
            "geotopo_flow_mean_px": flow_mag.mean((1, 2)),
            "geotopo_flow_max_px": flow_mag.flatten(1).max(1).values,
            "geotopo_flow_jacobian_mean": jac_mean,
            "geotopo_flow_folding_fraction": folding,
            "geotopo_geometry_abs_change": change,
            "geotopo_mode_id": one * 6.0,
            "geotr_m1_base_logits": factual_logits,
            "geotr_m1_final_logits": final_logits,
            "geotr_m1_final_probs": final_prob,
            "geotr_m1_flow_px": flow_px,
            "geotr_m1_boundary_normal": extended_normal,
            "geotr_m1_contour_owner_normal": base_normal,
            "geotr_m1_base_sdf": base_sdf,
            "geotr_m1_contour_owner_mask": contour_owner,
            "geotr_m1_base_boundary_mask": contour_owner,
            "geotr_m1_owner_distance_px": owner_distance,
            "geotr_m1_owner_flat_index": owner_flat_index,
            "geotr_m1_owner_case_valid": owner_case_valid,
            "geotr_m1_warp_support": finite_band,
            "geotr_m1_band_weight": band_weight,
            "geotr_m1_predicted_owner_sample_offset_px": owner_scalar[:, 0],
            "geotr_m1_owner_scalar_field": owner_scalar,
            # ROOTCAUSE-A2: retain the factorized/direct-signed prediction before
            # any optional Stage-A agreement shrinkage, plus the agreement map
            # itself.  These are diagnostics only and do not introduce trainable
            # parameters or new gradient paths.
            "geotr_m1_preagreement_owner_sample_offset_px": preagreement_owner_scalar[:, 0],
            "geotr_m1_ray_agreement_map": ray_agreement[:, 0],
            "geotr_m1_direction_raw": direction_raw_effective,
            "geotr_m1_direction_confidence": direction_conf,
            "geotr_m1_magnitude_raw": magnitude_raw,
            "geotr_m1_magnitude_px": magnitude_px,
            "geotr_m1_offset_distribution_enabled": one * float(
                self.uc_offset_distribution
            ),
            "geotr_m1_predicted_sample_offset_px": owner_scalar[:, 0],
            "geotr_m1_extended_sample_offset_px": extended_offset[:, 0],
            "geotr_m1_physical_sample_offset_px": sample_offset_px[:, 0],
            "geotr_m1_warped_logits": warped_logits,
            "geotr_m1_evidence_band": soft_boundary,
            "geotr_m1_scalar_field": extended_offset,
            "geotr_m1_regularized_field": flow_px,
            "geotr_m1_normal_field": extended_normal,
            "geotr_m1_rewrite_residual": final_logits - factual_logits,
            "geotr_m1_flow_scale_px": flow_mag.square().mean((1, 2)).sqrt().detach(),
            "geotr_m1_context_gate": zero,
            "geotr_m1_context_film_abs": zero,
            "geotr_m1_text_latent_abs": zero,
            "geotr_m1_anchor_boundary_fraction": owner_fraction,
            "geotr_m1_warp_support_fraction": band_fraction,
            "geotr_m1_weighted_band_fraction": weighted_band_fraction,
            "geotr_m1_conditioning_abs": zero,
            "geotr_m1_boundary_normal_warp": zero,
            "geotr_m1_sdf_operator_matched_warp": zero,
            "geotr_m1_posterior_stable_operator_warp": zero,
            "geotr_m1_uc_fnrt": one,
            "geotr_m1_operator_id": one * 6.0,
            "geotr_m1_gate_mode_id": zero,
            "geotr_m1_is_normal_1d": one,
            "geotr_m1_deadzone_fraction": (physical_abs <= 1.0e-6).float().mean((1, 2, 3)),
            "geotr_m1_exact_identity_error": (final_logits - factual_logits).abs().mean((1, 2, 3)),
            "geotr_m1_tangent_energy_ratio": zero,
            "geotr_m1_range_violation_fraction": (overshoot > 2e-6).float().mean((1, 2, 3)),
            "geotr_m1_range_violation_max": overshoot.flatten(1).max(1).values,
            "geotr_m1_autozero": one,
            "geotr_m1_autozero_trust_mean": (physical_abs / float(self.local_radius)).mean((1, 2, 3)).detach(),
            "geotr_m1_autozero_trust_std": (physical_abs / float(self.local_radius)).flatten(1).std(1, unbiased=False).detach(),
            "geotr_m1_edit_probability_mean": (
                (1.0 - p_keep).mean((1, 2, 3)).detach()
                if self.uc_offset_distribution and p_keep is not None
                else direction_conf.abs().mean((1, 2, 3)).detach()
            ),
            "geotr_m1_transition_band_fraction": band_fraction.detach(),
            "geotr_m1_local_radius_px": base_prob.new_tensor(float(self.local_radius)),
            "geotr_m1_memory_exact": one,
            # Keep raw posterior statistics for auditing and separately log
            # the effective statistics consumed by the transport predictor.
            "geotr_m1_mc_std_map": mc_std_map,
            "geotr_m1_mc_disagreement_map": mc_disagreement_map,
            "geotr_m1_mc_std_mean": mc_std_map.mean((1, 2, 3)).detach(),
            "geotr_m1_mc_disagreement_mean": mc_disagreement_map.mean((1, 2, 3)).detach(),
            "geotr_m1_effective_mc_std_mean": effective_mc_std_map.mean((1, 2, 3)).detach(),
            "geotr_m1_effective_mc_disagreement_mean": effective_mc_disagreement_map.mean((1, 2, 3)).detach(),
            "geotr_m1_ds_uc_fnrt": one * float(self.ds_uc_fnrt),
            "geotr_m1_ds_relative_uncertainty": one * float(self.ds_relative_uncertainty),
            "geotr_m1_ds_normalized_ray": one * float(self.ds_normalized_ray),
            "geotr_m1_ds_ray_agreement": one * float(self.ds_ray_agreement),
            "geotr_m1_ds_ray_agreement_apply": one * float(self.ds_ray_agreement_apply),
            "geotr_m1_ds_relative_mc_std_abs_mean": effective_mc_std_map.abs().mean((1, 2, 3)).detach(),
            "geotr_m1_ds_relative_mc_disagreement_abs_mean": effective_mc_disagreement_map.abs().mean((1, 2, 3)).detach(),
            "geotr_m1_ds_ray_agreement_mean": ray_agreement.mean((1, 2, 3)).detach(),
            "geotr_m1_ds_ray_agreement_low_fraction": (ray_agreement < 0.25).float().mean((1, 2, 3)).detach(),
            "geotr_m1_ablation_no_posterior": one * float(self.uc_ablation == "no_posterior_uncertainty"),
            "geotr_m1_ablation_no_ray": one * float(self.uc_ablation == "no_normal_ray_evidence"),
            "geotr_m1_ablation_direct_signed": one * float(self.uc_ablation == "direct_signed"),
            "geotr_m1_ablation_segmentation_only": one * float(self.uc_ablation == "segmentation_only"),
        }
        if self.uc_offset_distribution:
            assert offset_probs is not None and p_keep is not None
            k = int(offset_probs.shape[1])
            log_k = float(torch.log(base_prob.new_tensor(float(k))).item())
            offset_entropy = -(
                offset_probs.clamp_min(EPS).log() * offset_probs
            ).sum(dim=1, keepdim=True) / max(log_k, EPS)
            aux.update(
                {
                    "geotr_m1_offset_logits": offset_logits,
                    "geotr_m1_offset_probs": offset_probs,
                    "geotr_m1_offset_keep_prob": p_keep,
                    "geotr_m1_offset_entropy": offset_entropy,
                    "geotr_m1_offset_expected_abs_px": magnitude_px,
                    "geotr_m1_offset_global_expectation_px": (
                        global_expectation_px
                        if isinstance(global_expectation_px, torch.Tensor)
                        else owner_scalar
                    ),
                    "geotr_m1_offset_dominant_mode_px": owner_scalar,
                    "geotr_m1_offset_dominant_center_px": (
                        dominant_center_px
                        if isinstance(dominant_center_px, torch.Tensor)
                        else torch.zeros_like(owner_scalar)
                    ),
                    "geotr_m1_offset_dominant_mass": (
                        dominant_mass
                        if isinstance(dominant_mass, torch.Tensor)
                        else torch.ones_like(owner_scalar)
                    ),
                    "geotr_m1_hrcv_enabled": one * float(self.uc_hrcv),
                    "geotr_m1_hrcv_candidate_conditioned": one * float(
                        self.uc_hrcv and self.uc_hrcv_candidate_conditioned
                    ),
                    "geotr_m1_hrcv_delta_px": one * float(self.uc_hrcv_delta_px),
                    "geotr_m1_mrm_registered": one * float(self.uc_mrm_registered),
                    "geotr_m1_mrm_relational_cost": one * float(self.uc_mrm_relational_cost),
                    "geotr_m1_mrm_ordered_aggregation": one * float(self.uc_mrm_ordered_aggregation),
                    "geotr_m1_mrm_dominant_mode": one * float(self.uc_mrm_dominant_mode),
                    "geotr_m1_mrm_mode_radius": one * float(self.uc_mrm_mode_radius),
                    "geotr_m1_hierarchical_confidence_decoder": one * float(
                        self.uc_hierarchical_confidence_decoder
                    ),
                    "geotr_m1_hierarchical_full_conditional_execution": one * float(
                        self.uc_hierarchical_execution == "full_conditional"
                    ),
                    "geotr_m1_radius_balanced_keep_prior": one * float(
                        self.uc_radius_balanced_keep_prior
                    ),
                    "geotr_m1_hierarchical_conditional_offset_px": (
                        hierarchical_conditional_offset_px
                        if isinstance(hierarchical_conditional_offset_px, torch.Tensor)
                        else torch.zeros_like(owner_scalar)
                    ),
                    "geotr_m1_hierarchical_sign_confidence": (
                        hierarchical_sign_confidence
                        if isinstance(hierarchical_sign_confidence, torch.Tensor)
                        else torch.zeros_like(owner_scalar)
                    ),
                    "geotr_m1_hierarchical_winning_sign_mass": (
                        hierarchical_winning_sign_mass
                        if isinstance(hierarchical_winning_sign_mass, torch.Tensor)
                        else torch.zeros_like(owner_scalar)
                    ),
                    "geotr_m1_operator_aligned_candidates": one * float(
                        self.uc_operator_aligned_candidates
                    ),
                }
            )
        return candidate_logits, aux

    def _generate_posterior_stable_operator_warp(
        self,
        factual_logits: torch.Tensor,
        base_prob: torch.Tensor,
        hidden: torch.Tensor,
        soft_boundary: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Posterior-stable operator-matched contour warp.

        Compared with the archived SDF-OMW route, the physical velocity field is
        defined on a *fixed*, prediction-independent narrow band.  A contour owner
        predicts one signed inverse-warp displacement; nearest-owner extension and
        a cosine distance taper convert that contour velocity into a continuous
        narrow-band field.  Therefore the task loss is continuous in displacement:
        there are no integer-radius support jumps and no detached support decision.
        """
        b, _, h, w = factual_logits.shape
        base_hard = base_prob >= 0.5
        base_sdf = binary_signed_distance(base_hard, dtype=factual_logits.dtype)
        base_normal, normal_valid = sdf_normal(base_sdf)
        contour_owner = foreground_contour_owner(base_hard) & normal_valid

        owner_distance, owner_flat_index, owner_case_valid = nearest_owner_map(
            contour_owner, dtype=factual_logits.dtype
        )

        raw_scalar = self.controller(hidden)
        owner_scalar = float(self.local_radius) * torch.tanh(raw_scalar)
        if self.force_preserve:
            owner_scalar = torch.zeros_like(owner_scalar)

        extended_offset = gather_owner_field(owner_scalar, owner_flat_index, owner_case_valid)
        extended_normal = gather_owner_field(base_normal.detach(), owner_flat_index, owner_case_valid)
        extended_normal_mag = torch.linalg.vector_norm(extended_normal, dim=1, keepdim=True).clamp_min(EPS)
        extended_normal = torch.where(
            extended_normal_mag > 1.0e-5,
            extended_normal / extended_normal_mag,
            torch.zeros_like(extended_normal),
        )

        # Fixed, prediction-independent narrow band.  The +1 pixel is only the
        # bilinear interpolation footprint.  The cosine taper is C1 at both ends
        # and contains no dataset-tuned parameter.
        band_radius = float(self.local_radius + 1)
        finite_band = (
            owner_case_valid[:, None, None]
            & torch.isfinite(owner_distance)
            & (owner_distance <= band_radius)
        )
        phase = (owner_distance / max(band_radius, 1.0)).clamp(0.0, 1.0)
        band_weight = 0.5 * (1.0 + torch.cos(torch.pi * phase))
        band_weight = torch.where(finite_band, band_weight, torch.zeros_like(band_weight))

        sample_offset_px = extended_offset * band_weight[:, None]
        flow_px = sample_offset_px * extended_normal

        grid = self._identity_grid(factual_logits)
        if w > 1:
            grid_x = grid[..., 0] + 2.0 * flow_px[:, 0] / float(w - 1)
        else:
            grid_x = grid[..., 0]
        if h > 1:
            grid_y = grid[..., 1] + 2.0 * flow_px[:, 1] / float(h - 1)
        else:
            grid_y = grid[..., 1]
        sampling_grid = torch.stack([grid_x, grid_y], dim=-1)
        warped_logits = F.grid_sample(
            factual_logits, sampling_grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        identity_sample = F.grid_sample(
            factual_logits, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        # Subtracting the same identity sample makes d=0 an exact structural
        # identity even on platforms where grid_sample(identity) differs from the
        # input by a few ulps.
        warp_residual = warped_logits - identity_sample
        final_logits = factual_logits + warp_residual
        final_prob = torch.sigmoid(final_logits).clamp(EPS, 1.0 - EPS)

        candidate_logits = torch.cat([factual_logits, final_logits], dim=1)
        candidate_probs = torch.cat([base_prob, final_prob], dim=1)
        flow_mag = torch.linalg.vector_norm(flow_px, dim=1)
        jac_mean, folding = self._flow_jacobian_stats(flow_px)
        change = (final_prob - base_prob).abs().mean(dim=(1, 2, 3))
        zero, one = base_prob.new_zeros(b), base_prob.new_ones(b)
        quality = base_prob.new_zeros((b, 2)); quality[:, 1] = 1.0
        base_min = factual_logits.flatten(1).min(1).values[:, None, None, None]
        base_max = factual_logits.flatten(1).max(1).values[:, None, None, None]
        overshoot = F.relu(final_logits - base_max) + F.relu(base_min - final_logits)
        band_fraction = finite_band.float().mean((1, 2))
        weighted_band_fraction = band_weight.mean((1, 2))
        owner_fraction = contour_owner.float().mean((1, 2))
        physical_abs = sample_offset_px.abs()

        aux: Dict[str, torch.Tensor] = {
            "candidates": candidate_logits,
            "candidate_probs": candidate_probs,
            "mhcs_final_logits": final_logits,
            "mhcs_final_probs": final_prob,
            "mhcs_local_probs": final_prob,
            "mhcs_surface_hard_probs": final_prob,
            "mhcs_global_selected_probs": final_prob,
            "mhcs_quality_probs": quality,
            "mhcs_quality_logits": torch.log(quality.clamp_min(EPS)),
            "mhcs_quality_pred": quality,
            "mhcs_global_weights": quality,
            "mhcs_gate_alpha": change,
            "mhcs_effective_rank": torch.where(change > EPS, one * 2.0, one),
            "mhcs_surface_nonbase_mass": change,
            "mhcs_m1_distribution_log_var": base_prob.new_zeros(()),
            "v20_selector_hard": base_prob.new_ones((b, 1)),
            "direct_fused_probs": final_prob,
            "router_fused_probs": final_prob,
            "v20_fused_probs": final_prob,
            "v20_hard_fused_probs": final_prob,
            "v20_fused_logits": final_logits,
            "geotopo_base_logits": factual_logits,
            "geotopo_base_probs": base_prob,
            "geotopo_geometry_flow_px": flow_px,
            "geotopo_geometry_logits": final_logits,
            "geotopo_geometry_probs": final_prob,
            "geotopo_residual_only_logits": factual_logits,
            "geotopo_residual_only_probs": base_prob,
            "geotopo_reconstruction_after_geometry_logits": final_logits,
            "geotopo_reconstruction_after_geometry_probs": final_prob,
            "geotopo_final_logits": final_logits,
            "geotopo_final_probs": final_prob,
            "geotopo_flow_rms_px": flow_mag.square().mean((1, 2)).sqrt(),
            "geotopo_flow_mean_px": flow_mag.mean((1, 2)),
            "geotopo_flow_max_px": flow_mag.flatten(1).max(1).values,
            "geotopo_flow_jacobian_mean": jac_mean,
            "geotopo_flow_folding_fraction": folding,
            "geotopo_geometry_abs_change": change,
            "geotopo_mode_id": one * 5.0,
            "geotr_m1_base_logits": factual_logits,
            "geotr_m1_final_logits": final_logits,
            "geotr_m1_final_probs": final_prob,
            "geotr_m1_flow_px": flow_px,
            "geotr_m1_boundary_normal": extended_normal,
            "geotr_m1_contour_owner_normal": base_normal,
            "geotr_m1_base_sdf": base_sdf,
            "geotr_m1_contour_owner_mask": contour_owner,
            "geotr_m1_base_boundary_mask": contour_owner,
            "geotr_m1_owner_distance_px": owner_distance,
            "geotr_m1_owner_flat_index": owner_flat_index,
            "geotr_m1_owner_case_valid": owner_case_valid,
            "geotr_m1_warp_support": finite_band,
            "geotr_m1_band_weight": band_weight,
            "geotr_m1_predicted_owner_sample_offset_px": owner_scalar[:, 0],
            "geotr_m1_owner_scalar_field": owner_scalar,
            "geotr_m1_predicted_sample_offset_px": owner_scalar[:, 0],
            "geotr_m1_extended_sample_offset_px": extended_offset[:, 0],
            "geotr_m1_physical_sample_offset_px": sample_offset_px[:, 0],
            "geotr_m1_warped_logits": warped_logits,
            "geotr_m1_evidence_band": soft_boundary,
            "geotr_m1_scalar_field": extended_offset,
            "geotr_m1_regularized_field": flow_px,
            "geotr_m1_normal_field": extended_normal,
            "geotr_m1_rewrite_residual": final_logits - factual_logits,
            "geotr_m1_flow_scale_px": flow_mag.square().mean((1, 2)).sqrt().detach(),
            "geotr_m1_context_gate": zero,
            "geotr_m1_context_film_abs": zero,
            "geotr_m1_text_latent_abs": zero,
            "geotr_m1_anchor_boundary_fraction": owner_fraction,
            "geotr_m1_warp_support_fraction": band_fraction,
            "geotr_m1_weighted_band_fraction": weighted_band_fraction,
            "geotr_m1_conditioning_abs": zero,
            "geotr_m1_boundary_normal_warp": zero,
            "geotr_m1_sdf_operator_matched_warp": zero,
            "geotr_m1_posterior_stable_operator_warp": one,
            "geotr_m1_operator_id": one * 5.0,
            "geotr_m1_gate_mode_id": zero,
            "geotr_m1_is_normal_1d": one,
            "geotr_m1_deadzone_fraction": (physical_abs <= 1.0e-6).float().mean((1, 2, 3)),
            "geotr_m1_exact_identity_error": (final_logits - factual_logits).abs().mean((1, 2, 3)),
            "geotr_m1_tangent_energy_ratio": zero,
            "geotr_m1_range_violation_fraction": (overshoot > 2e-6).float().mean((1, 2, 3)),
            "geotr_m1_range_violation_max": overshoot.flatten(1).max(1).values,
            "geotr_m1_autozero": one,
            "geotr_m1_autozero_trust_mean": (physical_abs / float(self.local_radius)).mean((1, 2, 3)).detach(),
            "geotr_m1_autozero_trust_std": (physical_abs / float(self.local_radius)).flatten(1).std(1, unbiased=False).detach(),
            "geotr_m1_edit_probability_mean": (physical_abs / float(self.local_radius)).mean((1, 2, 3)).detach(),
            "geotr_m1_transition_band_fraction": band_fraction.detach(),
            "geotr_m1_local_radius_px": base_prob.new_tensor(float(self.local_radius)),
            "geotr_m1_memory_exact": one,
        }
        return candidate_logits, aux

    def _generate_boundary_normal_warp(
        self,
        factual_logits: torch.Tensor,
        base_prob: torch.Tensor,
        hidden: torch.Tensor,
        soft_boundary: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """True spatial logit warp constrained to the Base boundary normal.

        ``sample_offset_px`` is the *inverse-warp sampling* displacement.  If the
        predicted contour should move by +d along the foreground normal, the
        output samples the factual logit field at -d along that normal.  This is
        exactly the convention used by ``grid_sample``.
        """
        b, _, h, w = factual_logits.shape
        base_hard = base_prob >= 0.5
        base_boundary = self._hard_boundary(base_hard)
        support = F.max_pool2d(
            base_boundary.float(),
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.local_radius,
        ) > 0.5

        normal = self._foreground_normal(base_prob.detach())
        raw_scalar = self.controller(hidden)
        sample_offset_px = float(self.local_radius) * torch.tanh(raw_scalar)
        sample_offset_px = sample_offset_px * support.to(sample_offset_px)
        if self.force_preserve:
            sample_offset_px = torch.zeros_like(sample_offset_px)

        flow_px = sample_offset_px * normal
        grid = self._identity_grid(factual_logits)
        if w > 1:
            grid_x = grid[..., 0] + 2.0 * flow_px[:, 0] / float(w - 1)
        else:
            grid_x = grid[..., 0]
        if h > 1:
            grid_y = grid[..., 1] + 2.0 * flow_px[:, 1] / float(h - 1)
        else:
            grid_y = grid[..., 1]
        sampling_grid = torch.stack([grid_x, grid_y], dim=-1)
        warped_logits = F.grid_sample(
            factual_logits,
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        # ``grid_sample`` can have a few-ulp identity-grid interpolation error.
        # Subtract the *same operator* at zero flow so d=0 is bitwise exact Base
        # while preserving the derivative of the physical warped sample.
        identity_sample = F.grid_sample(
            factual_logits,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        warp_residual = warped_logits - identity_sample

        # Exact Base outside support and exact AutoZero identity at d=0.
        final_logits = factual_logits + support.to(factual_logits) * warp_residual
        final_prob = torch.sigmoid(final_logits).clamp(EPS, 1.0 - EPS)
        candidate_logits = torch.cat([factual_logits, final_logits], dim=1)
        candidate_probs = torch.cat([base_prob, final_prob], dim=1)

        flow_mag = torch.linalg.vector_norm(flow_px, dim=1)
        jac_mean, folding = self._flow_jacobian_stats(flow_px)
        change = (final_prob - base_prob).abs().mean(dim=(1, 2, 3))
        zero, one = base_prob.new_zeros(b), base_prob.new_ones(b)
        quality = base_prob.new_zeros((b, 2)); quality[:, 1] = 1.0
        base_min = factual_logits.flatten(1).min(1).values[:, None, None, None]
        base_max = factual_logits.flatten(1).max(1).values[:, None, None, None]
        overshoot = F.relu(final_logits - base_max) + F.relu(base_min - final_logits)
        support_fraction = support.float().mean((1, 2, 3))
        boundary_fraction = base_boundary.float().mean((1, 2, 3))
        offset_abs = sample_offset_px.abs()

        aux: Dict[str, torch.Tensor] = {
            "candidates": candidate_logits,
            "candidate_probs": candidate_probs,
            "mhcs_final_logits": final_logits,
            "mhcs_final_probs": final_prob,
            "mhcs_local_probs": final_prob,
            "mhcs_surface_hard_probs": final_prob,
            "mhcs_global_selected_probs": final_prob,
            "mhcs_quality_probs": quality,
            "mhcs_quality_logits": torch.log(quality.clamp_min(EPS)),
            "mhcs_quality_pred": quality,
            "mhcs_global_weights": quality,
            "mhcs_gate_alpha": change,
            "mhcs_effective_rank": torch.where(change > EPS, one * 2.0, one),
            "mhcs_surface_nonbase_mass": change,
            "mhcs_m1_distribution_log_var": base_prob.new_zeros(()),
            "v20_selector_hard": base_prob.new_ones((b, 1)),
            "direct_fused_probs": final_prob,
            "router_fused_probs": final_prob,
            "v20_fused_probs": final_prob,
            "v20_hard_fused_probs": final_prob,
            "v20_fused_logits": final_logits,
            "geotopo_base_logits": factual_logits,
            "geotopo_base_probs": base_prob,
            "geotopo_geometry_flow_px": flow_px,
            "geotopo_geometry_logits": final_logits,
            "geotopo_geometry_probs": final_prob,
            "geotopo_residual_only_logits": factual_logits,
            "geotopo_residual_only_probs": base_prob,
            "geotopo_reconstruction_after_geometry_logits": final_logits,
            "geotopo_reconstruction_after_geometry_probs": final_prob,
            "geotopo_final_logits": final_logits,
            "geotopo_final_probs": final_prob,
            "geotopo_flow_rms_px": flow_mag.square().mean((1, 2)).sqrt(),
            "geotopo_flow_mean_px": flow_mag.mean((1, 2)),
            "geotopo_flow_max_px": flow_mag.flatten(1).max(1).values,
            "geotopo_flow_jacobian_mean": jac_mean,
            "geotopo_flow_folding_fraction": folding,
            "geotopo_geometry_abs_change": change,
            "geotopo_mode_id": one * 3.0,
            "geotr_m1_base_logits": factual_logits,
            "geotr_m1_final_logits": final_logits,
            "geotr_m1_final_probs": final_prob,
            "geotr_m1_flow_px": flow_px,
            "geotr_m1_boundary_normal": normal,
            "geotr_m1_base_boundary_mask": base_boundary[:, 0],
            "geotr_m1_warp_support": support[:, 0],
            "geotr_m1_predicted_sample_offset_px": sample_offset_px[:, 0],
            "geotr_m1_warped_logits": warped_logits,
            "geotr_m1_evidence_band": soft_boundary,
            "geotr_m1_scalar_field": sample_offset_px,
            "geotr_m1_regularized_field": flow_px,
            "geotr_m1_normal_field": normal,
            "geotr_m1_rewrite_residual": final_logits - factual_logits,
            "geotr_m1_flow_scale_px": flow_mag.square().mean((1, 2)).sqrt().detach(),
            "geotr_m1_context_gate": zero,
            "geotr_m1_context_film_abs": zero,
            "geotr_m1_text_latent_abs": zero,
            "geotr_m1_anchor_boundary_fraction": boundary_fraction,
            "geotr_m1_warp_support_fraction": support_fraction,
            "geotr_m1_conditioners_detached": one,
            "geotr_m1_has_m2": zero,
            "geotr_m1_uses_semantic": one,
            "geotr_m1_uses_text": zero,
            "geotr_m1_uses_anchor_cues": one,
            "geotr_m1_logit_transport": one,
            "geotr_m1_boundary_normal_warp": one,
            "geotr_m1_operator_id": one * 3.0,
            "geotr_m1_gate_mode_id": zero,
            "geotr_m1_is_normal_1d": one,
            "geotr_m1_deadzone_fraction": (offset_abs <= 1.0e-6).float().mean((1, 2, 3)),
            "geotr_m1_exact_identity_error": (final_logits - factual_logits).abs().mean((1, 2, 3)),
            "geotr_m1_tangent_energy_ratio": zero,
            "geotr_m1_range_violation_fraction": (overshoot > 2e-6).float().mean((1, 2, 3)),
            "geotr_m1_range_violation_max": overshoot.flatten(1).max(1).values,
            "geotr_m1_autozero": one,
            "geotr_m1_autozero_trust_mean": (offset_abs / float(self.local_radius)).mean((1, 2, 3)).detach(),
            "geotr_m1_autozero_trust_std": (offset_abs / float(self.local_radius)).flatten(1).std(1, unbiased=False).detach(),
            "geotr_m1_edit_probability_mean": (offset_abs / float(self.local_radius)).mean((1, 2, 3)).detach(),
            "geotr_m1_transition_band_fraction": support_fraction.detach(),
            "geotr_m1_local_radius_px": base_prob.new_tensor(float(self.local_radius)),
            "geotr_m1_memory_exact": one,
        }
        return candidate_logits, aux

    def generate(
        self,
        base_logits: torch.Tensor,
        image: torch.Tensor,
        semantic_map: Optional[torch.Tensor],
        text_features: Optional[torch.Tensor],
        negative_text_features: Optional[torch.Tensor] = None,
        mc_std_map: Optional[torch.Tensor] = None,
        mc_disagreement_map: Optional[torch.Tensor] = None,
        mc_pairwise_disagreement: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del text_features, negative_text_features, mc_pairwise_disagreement, kwargs
        if base_logits.ndim == 3:
            base_logits = base_logits[:, None]
        if base_logits.ndim != 4 or base_logits.shape[1] != 1:
            raise ValueError(f"Expected Base logits [B,1,H,W], got {tuple(base_logits.shape)}")
        if semantic_map is None:
            raise RuntimeError("SemLT-LST v3 requires the UniMedCLIP spatial semantic map")
        if semantic_map.ndim != 4 or semantic_map.shape[1] != self.semantic_channels:
            raise ValueError(
                f"Expected semantic_map [B,{self.semantic_channels},H,W], got {tuple(semantic_map.shape)}"
            )

        factual_logits = base_logits.detach()
        base_prob = torch.sigmoid(factual_logits).clamp(EPS, 1.0 - EPS)
        hw = tuple(factual_logits.shape[-2:])
        uncertainty = (4.0 * base_prob * (1.0 - base_prob)).clamp(0.0, 1.0)
        boundary = self._base_boundary(base_prob)
        if self.uc_fnrt and self.training and torch.is_grad_enabled():
            # R2 memory-exact implementation: keep only the compact inputs and
            # recompute dense UC-FNRT feature activations during backward.
            # This changes memory/computation only, not the mathematical forward.
            image_input = self._resize(image, hw)
            image_feature = checkpoint(
                lambda x: self.image_stem(x),
                image_input,
                use_reentrant=False,
            )
            semantic_input = semantic_map.detach()
            semantic_feature = checkpoint(
                lambda x: self._semantic_project_lowmem(x, hw),
                semantic_input,
                use_reentrant=False,
            )
        else:
            image_feature = self.image_stem(self._resize(image, hw))
            semantic_feature = self._semantic_project_lowmem(semantic_map, hw)

        if mc_std_map is None:
            mc_std = torch.zeros_like(base_prob)
        else:
            mc_std = self._resize(mc_std_map.detach().to(base_prob), hw)
            if mc_std.shape[1] != 1:
                mc_std = mc_std.mean(dim=1, keepdim=True)
        if mc_disagreement_map is None:
            mc_disagreement = torch.zeros_like(base_prob)
        else:
            mc_disagreement = self._resize(mc_disagreement_map.detach().to(base_prob), hw)
            if mc_disagreement.shape[1] != 1:
                mc_disagreement = mc_disagreement.mean(dim=1, keepdim=True)
        mc_std = mc_std.clamp_min(0.0)
        mc_disagreement = mc_disagreement.clamp(0.0, 1.0)

        if self.uc_fnrt:
            return self._generate_uc_fnrt(
                factual_logits=factual_logits,
                base_prob=base_prob,
                image_feature=image_feature,
                semantic_feature=semantic_feature,
                soft_boundary=boundary,
                margin_uncertainty=uncertainty,
                mc_std_map=mc_std,
                mc_disagreement_map=mc_disagreement,
            )

        hidden = self.fuse(
            torch.cat(
                [
                    image_feature,
                    semantic_feature,
                    base_prob,
                    uncertainty,
                    boundary,
                ],
                dim=1,
            )
        )

        if self.posterior_stable_operator_warp:
            return self._generate_posterior_stable_operator_warp(
                factual_logits=factual_logits,
                base_prob=base_prob,
                hidden=hidden,
                soft_boundary=boundary,
            )

        if self.sdf_operator_matched_warp:
            return self._generate_sdf_operator_matched_warp(
                factual_logits=factual_logits,
                base_prob=base_prob,
                hidden=hidden,
                soft_boundary=boundary,
            )

        if self.boundary_normal_warp:
            return self._generate_boundary_normal_warp(
                factual_logits=factual_logits,
                base_prob=base_prob,
                hidden=hidden,
                soft_boundary=boundary,
            )

        # Do not materialize the 4+K output. Gate/type and source rows use the
        # same convolution but are evaluated separately for bounded memory.
        if self.action_value_policy:
            # Decision isolation.  The edit-policy head consumes the same rich
            # representation, but its loss is not allowed to reshape the editor's
            # shared hidden features.
            edit_logit = F.conv2d(
                hidden.detach(),
                self.controller.weight[0:1],
                self.controller.bias[0:1],
                stride=1,
                padding=1,
            )
            structure_logits = F.conv2d(
                hidden,
                self.controller.weight[1:4],
                self.controller.bias[1:4],
                stride=1,
                padding=1,
            )
            locator_logit = structure_logits[:, 0:1]
            type_logits = structure_logits[:, 1:3]
        else:
            # Frozen compatibility route for archived v3/v3.1/eligible-gate runs.
            decision_logits = F.conv2d(
                hidden, self.controller.weight[:4], self.controller.bias[:4],
                stride=1, padding=1,
            )
            edit_logit = decision_logits[:, 0:1]
            locator_logit = decision_logits[:, 1:2]
            type_logits = decision_logits[:, 2:4]
        raw_edit_prob = torch.sigmoid(edit_logit)
        type_prob = torch.softmax(type_logits, dim=1)
        type_index = type_logits.argmax(dim=1, keepdim=True)
        hard_type = torch.zeros_like(type_prob).scatter_(1, type_index, 1.0)
        local = self._memory_exact_local_aggregate(hidden, factual_logits)

        add_logits = local["add_logits"]
        remove_logits = local["remove_logits"]
        soft_edited_logits = (
            type_prob[:, 0:1] * add_logits + type_prob[:, 1:2] * remove_logits
        )
        if self.action_value_policy:
            # ADD and REMOVE are mutually exclusive semantic actions.  A soft
            # convex blend is not a physical action; value-policy deployment
            # executes the exact argmax action.  Type CE trains the discrete
            # action head and no STE is used.
            edited_logits = (
                hard_type[:, 0:1] * add_logits + hard_type[:, 1:2] * remove_logits
            )
        else:
            edited_logits = soft_edited_logits

        # A realizable local transport must have both states somewhere in the
        # local set.  This gives a parameter-free physical eligibility mask.
        transition_band = local["add_source_exists"] & local["remove_source_exists"]
        if self.v31_rootfix and self.transition_band_enabled:
            eligible_edit_prob = raw_edit_prob * transition_band.to(raw_edit_prob)
        else:
            eligible_edit_prob = raw_edit_prob

        if self.force_preserve:
            physical_edit_gate = torch.zeros_like(eligible_edit_prob)
        elif self.v31_rootfix and self.hard_deploy:
            # Exact identity for KEEP.  We deliberately do NOT use an STE here:
            # v3.1 loss trains edit_logit only through its own supervised gate
            # objective, while correction losses train the editor through a GT-
            # derived teacher route.  This is the causal gradient separation.
            physical_edit_gate = (eligible_edit_prob >= self.deploy_threshold).to(eligible_edit_prob)
        else:
            physical_edit_gate = eligible_edit_prob

        final_logits = factual_logits + physical_edit_gate * (edited_logits - factual_logits)
        final_prob = torch.sigmoid(final_logits).clamp(EPS, 1.0 - EPS)

        # Algebraically identical to the dense effective-attention flow:
        # edit_prob * E[offset | edited] = p_ADD E_ADD + p_REMOVE E_REMOVE.
        flow_x = (
            physical_edit_gate * type_prob[:, 0:1] * local["add_dx"]
            + physical_edit_gate * type_prob[:, 1:2] * local["remove_dx"]
        )[:, 0]
        flow_y = (
            physical_edit_gate * type_prob[:, 0:1] * local["add_dy"]
            + physical_edit_gate * type_prob[:, 1:2] * local["remove_dy"]
        )[:, 0]
        flow_px = torch.stack([flow_x, flow_y], dim=1)

        candidate_logits = torch.cat([factual_logits, final_logits], dim=1)
        candidate_probs = torch.cat([base_prob, final_prob], dim=1)
        flow_mag = torch.linalg.vector_norm(flow_px, dim=1)
        jac_mean, folding = self._flow_jacobian_stats(flow_px)
        change = (final_prob - base_prob).abs().mean(dim=(1, 2, 3))
        b = factual_logits.shape[0]
        zero, one = base_prob.new_zeros(b), base_prob.new_ones(b)
        quality = base_prob.new_zeros((b, 2)); quality[:, 1] = 1.0
        base_min = factual_logits.flatten(1).min(1).values[:, None, None, None]
        base_max = factual_logits.flatten(1).max(1).values[:, None, None, None]
        overshoot = F.relu(final_logits - base_max) + F.relu(base_min - final_logits)

        aux: Dict[str, torch.Tensor] = {
            "candidates": candidate_logits, "candidate_probs": candidate_probs,
            "mhcs_final_logits": final_logits, "mhcs_final_probs": final_prob,
            "mhcs_local_probs": final_prob, "mhcs_surface_hard_probs": final_prob,
            "mhcs_global_selected_probs": final_prob, "mhcs_quality_probs": quality,
            "mhcs_quality_logits": torch.log(quality.clamp_min(EPS)),
            "mhcs_quality_pred": quality, "mhcs_global_weights": quality,
            "mhcs_gate_alpha": change,
            "mhcs_effective_rank": torch.where(change > EPS, one * 2.0, one),
            "mhcs_surface_nonbase_mass": change,
            "mhcs_m1_distribution_log_var": base_prob.new_zeros(()),
            "v20_selector_hard": base_prob.new_ones((b, 1)),
            "direct_fused_probs": final_prob, "router_fused_probs": final_prob,
            "v20_fused_probs": final_prob, "v20_hard_fused_probs": final_prob,
            "v20_fused_logits": final_logits,
            "geotopo_base_logits": factual_logits, "geotopo_base_probs": base_prob,
            "geotopo_geometry_flow_px": flow_px, "geotopo_geometry_logits": final_logits,
            "geotopo_geometry_probs": final_prob,
            "geotopo_residual_only_logits": factual_logits,
            "geotopo_residual_only_probs": base_prob,
            "geotopo_reconstruction_after_geometry_logits": final_logits,
            "geotopo_reconstruction_after_geometry_probs": final_prob,
            "geotopo_final_logits": final_logits, "geotopo_final_probs": final_prob,
            "geotopo_flow_rms_px": flow_mag.square().mean((1, 2)).sqrt(),
            "geotopo_flow_mean_px": flow_mag.mean((1, 2)),
            "geotopo_flow_max_px": flow_mag.flatten(1).max(1).values,
            "geotopo_flow_jacobian_mean": jac_mean,
            "geotopo_flow_folding_fraction": folding,
            "geotopo_geometry_abs_change": change, "geotopo_mode_id": one * 2.0,
            "geotr_m1_base_logits": factual_logits, "geotr_m1_final_logits": final_logits,
            "geotr_m1_final_probs": final_prob, "geotr_m1_flow_px": flow_px,
            "geotr_m1_edit_logit": edit_logit[:, 0],
            "geotr_m1_locator_logit": locator_logit[:, 0],
            "geotr_m1_type_logits": type_logits,
            "geotr_m1_type_probs": type_prob,
            "geotr_m1_hard_type": hard_type,
            # Compatibility views are diagnostics only; the loss does not use
            # these as deployment probabilities.
            "geotr_m1_state_logits": torch.cat(
                [-edit_logit, edit_logit + type_logits[:, 0:1],
                 edit_logit + type_logits[:, 1:2]], dim=1
            ),
            "geotr_m1_state_probs": torch.cat(
                [1.0 - eligible_edit_prob, eligible_edit_prob * type_prob], dim=1
            ),
            # Compact sufficient statistics replace dense K-channel attention banks.
            "geotr_m1_add_valid_source_mass": local["add_valid_source_mass"][:, 0],
            "geotr_m1_remove_valid_source_mass": local["remove_valid_source_mass"][:, 0],
            "geotr_m1_add_source_exists": local["add_source_exists"][:, 0],
            "geotr_m1_remove_source_exists": local["remove_source_exists"][:, 0],
            "geotr_m1_source_chunk_size": local["source_chunk_size"],
            "geotr_m1_source_chunk_count": local["source_chunk_count"],
            "geotr_m1_local_radius_px": base_prob.new_tensor(float(self.local_radius)),
            "geotr_m1_change_gate": physical_edit_gate,
            "geotr_m1_raw_edit_probability": raw_edit_prob,
            "geotr_m1_masked_edit_probability": eligible_edit_prob,
            "geotr_m1_transition_band": transition_band[:, 0],
            "geotr_m1_add_logits": add_logits,
            "geotr_m1_remove_logits": remove_logits,
            "geotr_m1_edited_logits": edited_logits,
            "geotr_m1_soft_edited_logits": soft_edited_logits,
            "geotr_m1_evidence_band": boundary,
            "geotr_m1_scalar_field": eligible_edit_prob, "geotr_m1_regularized_field": flow_px,
            "geotr_m1_normal_field": flow_px.new_zeros(flow_px.shape),
            "geotr_m1_rewrite_residual": factual_logits.new_zeros(factual_logits.shape),
            "geotr_m1_flow_scale_px": flow_mag.square().mean((1, 2)).sqrt().detach(),
            "geotr_m1_context_gate": zero, "geotr_m1_context_film_abs": zero,
            "geotr_m1_text_latent_abs": zero,
            "geotr_m1_anchor_boundary_fraction": boundary.mean((1, 2, 3)).detach(),
            "geotr_m1_conditioners_detached": one, "geotr_m1_has_m2": zero,
            "geotr_m1_uses_semantic": one, "geotr_m1_uses_text": zero,
            "geotr_m1_uses_anchor_cues": one, "geotr_m1_logit_transport": one,
            "geotr_m1_operator_id": one * 2.0, "geotr_m1_gate_mode_id": one * 2.0,
            "geotr_m1_is_normal_1d": zero,
            "geotr_m1_deadzone_fraction": (physical_edit_gate <= 0.0).float().mean((1, 2, 3)),
            "geotr_m1_exact_identity_error": (final_logits - factual_logits).abs().mean((1, 2, 3)),
            "geotr_m1_tangent_energy_ratio": zero,
            "geotr_m1_range_violation_fraction": (overshoot > 2e-6).float().mean((1, 2, 3)),
            "geotr_m1_range_violation_max": overshoot.flatten(1).max(1).values,
            "geotr_m1_autozero": one,
            "geotr_m1_autozero_trust_mean": physical_edit_gate.detach().mean((1, 2, 3)),
            "geotr_m1_autozero_trust_std": eligible_edit_prob.detach().flatten(1).std(1, unbiased=False),
            "geotr_m1_edit_probability_mean": eligible_edit_prob.detach().mean((1, 2, 3)),
            "geotr_m1_transition_band_fraction": transition_band.float().detach().mean((1, 2, 3)),
            "geotr_m1_memory_exact": one,
        }
        return candidate_logits, aux
