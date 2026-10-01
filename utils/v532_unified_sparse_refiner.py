"""V532 unified typed error-guided sparse refinement.

V532 is a one-pass refinement head for medical image segmentation.  It keeps an
internally supervised coarse prediction ``P0`` and converts its residual errors
into four monotone actions:

    Delete / Fill / Trim / Expand

The design intentionally removes the historical candidate-bank ranking and the
separate M3 veto.  Instead it factorises the decision into three auditable
spatial predictions:

1. edit existence: Preserve versus Edit;
2. conditional action type among four typed actions;
3. actual action outcome: Neutral / Benefit / Harm.

V535 optionally replaces the factorised probability product with a direct
utility policy.  Preserve has a fixed utility of zero and the four actions have
learned signed utilities.  A single five-way hard-forward/soft-backward argmax
therefore implements both action selection and Execute/Preserve without a
manual execution threshold.  The four action candidates are monotone by
construction and the final probability is bounded in [0, 1].
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.v536_component_rejector import select_top1_component
from utils.v537_component_utility_ranker import V537ComponentUtilityRanker
from utils.v538_online_component_refiner import V538OnlineComponentRefiner
from utils.v551_multiscale_component_editor import V551MultiscaleTypedComposerEditor

EPS = 1.0e-6
OUTCOME_NEUTRAL = 0
OUTCOME_BENEFIT = 1
OUTCOME_HARM = 2
ACTION_NAMES: Tuple[str, ...] = (
    "preserve",
    "delete",
    "fill",
    "trim",
    "expand",
)


def _as_b1hw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        return x[:, :1]
    if x.ndim == 3:
        return x[:, None]
    raise ValueError(f"Expected [B,H,W] or [B,1,H,W], got {tuple(x.shape)}")


def _entropy(prob: torch.Tensor) -> torch.Tensor:
    p = prob.clamp(EPS, 1.0 - EPS)
    return (
        -(p * p.log() + (1.0 - p) * (1.0 - p).log()) / math.log(2.0)
    ).clamp(0.0, 1.0)


def _soft_boundary(prob: torch.Tensor, radius: int = 1) -> torch.Tensor:
    radius = max(int(radius), 1)
    kernel = 2 * radius + 1
    dilated = F.max_pool2d(prob, kernel, stride=1, padding=radius)
    eroded = -F.max_pool2d(-prob, kernel, stride=1, padding=radius)
    return (dilated - eroded).clamp(0.0, 1.0)


def _gray_edge(
    image: torch.Tensor,
    size: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    gray = image.mean(dim=1, keepdim=True)
    if gray.shape[-2:] != size:
        gray = F.interpolate(
            gray,
            size=size,
            mode="bilinear",
            align_corners=False,
        )
    lo = gray.amin(dim=(-2, -1), keepdim=True)
    hi = gray.amax(dim=(-2, -1), keepdim=True)
    gray = (gray - lo) / (hi - lo).clamp_min(EPS)
    gx = F.pad(
        (gray[..., :, 1:] - gray[..., :, :-1]).abs(),
        (0, 1, 0, 0),
    )
    gy = F.pad(
        (gray[..., 1:, :] - gray[..., :-1, :]).abs(),
        (0, 0, 0, 1),
    )
    return gray, (gx + gy).clamp(0.0, 1.0)


def _group_count(channels: int) -> int:
    groups = min(8, int(channels))
    while groups > 1 and int(channels) % groups != 0:
        groups -= 1
    return groups


def _scale_gradient(value: torch.Tensor, scale: float) -> torch.Tensor:
    """Keep the forward value unchanged while scaling upstream gradients."""
    if not isinstance(value, torch.Tensor):
        return value
    scale = float(scale)
    if scale <= 0.0:
        return value.detach()
    if scale >= 1.0:
        return value
    return value.detach() + scale * (value - value.detach())


def _ste_binary(probability: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Binary hard-forward / soft-backward gate.

    Forward values are exactly 0 or 1, which gives physical Preserve semantics.
    Backward gradients are those of the supplied probability.
    """
    hard = (probability >= float(threshold)).to(probability.dtype)
    return hard.detach() - probability.detach() + probability


def _ste_one_hot(logits: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return hard-forward one-hot actions and their soft probabilities."""
    soft = F.softmax(logits / max(float(temperature), 1.0e-4), dim=1)
    index = soft.argmax(dim=1, keepdim=True)
    hard = torch.zeros_like(soft).scatter_(1, index, 1.0)
    straight_through = hard.detach() - soft.detach() + soft
    return straight_through, soft


def _initial_outcome_bias(
    neutral: float,
    benefit: float,
    harm: float,
) -> torch.Tensor:
    values = torch.tensor([neutral, benefit, harm], dtype=torch.float32)
    values = values.clamp_min(1.0e-5)
    values = values / values.sum()
    return values.log()


class V532UnifiedSparseRefiner(nn.Module):
    """One-pass typed sparse refiner with explicit Preserve and outcome risk.

    The route is factorised as:

        P(edit | context) * P(action | edit, context)

    This avoids forcing an extremely imbalanced five-way softmax to learn both
    Preserve and rare error subtypes.  A three-way outcome head predicts
    Neutral / Benefit / Harm for every actual action candidate.  The risk gate
    is derived from these calibrated probabilities, not from a generic FP/FN
    correctness label.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        dropout: float = 0.10,
        semantic_channels: int = 512,
        semantic_dim: int = 32,
        use_semantic: bool = True,
        edit_temperature: float = 1.0,
        action_temperature: float = 0.70,
        outcome_temperature: float = 1.0,
        risk_temperature: float = 0.20,
        harm_penalty: float = 2.0,
        edit_penalty: float = 0.03,
        utility_threshold: float = 0.0,
        initial_edit_rate: float = 0.02,
        initial_outcome_neutral: float = 0.80,
        initial_outcome_benefit: float = 0.10,
        initial_outcome_harm: float = 0.10,
        max_action_alpha: float = 1.0,
        deployment_hard_route: bool = True,
        execute_threshold: float = 0.50,
        adaptive_utility_policy: bool = False,
        policy_temperature: float = 1.0,
        prior_aligned_case_component: bool = False,
        component_utility_ranking: bool = False,
        component_min_pixels: int = 4,
        max_components: int = 48,
        component_ranker_hidden_dim: int = 128,
        component_ranker_dropout: float = 0.10,
        component_initial_score_bias: float = -0.002,
        online_component_refinement: bool = False,
        component_num_slots: int = 8,
        component_slot_hidden_dim: int = 128,
        component_slot_dropout: float = 0.10,
        component_mask_temperature: float = 1.0,
        component_action_temperature: float = 0.70,
        component_min_area_fraction: float = 1.0e-4,
        component_max_area_fraction: float = 0.035,
        component_initial_presence_rate: float = 0.10,
        component_deploy_mask_threshold: float = 0.50,
        component_deploy_presence_threshold: float = 0.50,
        component_deploy_min_gain: float = 0.001,
        component_max_steps: int = 3,
        component_max_overlap: float = 0.20,
        component_max_total_edit_fraction: float = 0.035,
        component_continuous_dose_enabled: bool = True,
        component_polarity_temperature: float = 0.70,
        component_minimum_dose: float = 0.25,
        component_maximum_dose: float = 16.0,
        component_initial_dose: float = 1.0,
        component_candidate_outcome_selector_enabled: bool = True,
        component_selector_spatial_size: int = 16,
        component_selector_hidden_dim: int = 128,
        component_selector_dropout: float = 0.10,
        component_selector_gain_scale: float = 1000.0,
        component_selector_benefit_threshold: float = 0.70,
        component_selector_harm_threshold: float = 0.10,
        component_selector_lcb_beta: float = 1.0,
        component_selector_logvar_min: float = -8.0,
        component_selector_logvar_max: float = 4.0,
        component_selector_initial_benefit_rate: float = 0.10,
        component_selector_initial_harm_rate: float = 0.50,
        component_use_gain_as_decision_score: bool = False,
        component_adaptive_cardinality_hard_mask: bool = False,
        component_prior_free_outcome_init: bool = False,
        component_slot_competition_enabled: bool = False,
        component_gain_sign_shadow_deploy_enabled: bool = False,
        component_factorized_outcome_enabled: bool = False,
        component_factorized_direction_zero_init: bool = False,
        component_factorized_deployment_enabled: bool = False,
        component_deploy_editability_threshold: float = 0.50,
        component_deploy_direction_threshold: float = 0.50,
        multiscale_typed_editor: bool = False,
        component_pyramid_scales: Tuple[int, ...] = (1, 2, 4, 8),
        component_scale_temperature: float = 0.70,
        component_max_atoms_per_slot: int = 3,
        component_atomization_start_epoch: int = 1,
        component_atom_min_pixels: int = 4,
        component_atom_dedup_iou: float = 0.85,
        component_gpu_atomizer_enabled: bool = True,
        component_max_active_atoms: int = 8,
        component_atom_partition_temperature: float = 0.35,
        component_atom_partition_extent: float = 0.85,
        component_atom_presence_threshold: float = 0.35,
        component_atom_quality_threshold: float = 0.20,
        component_atom_quality_gate_start_epoch: int = 4,
        component_single_pass_editor: bool = True,
        component_boundary_residual_enabled: bool = True,
        component_boundary_residual_cap: float = 0.75,
        component_boundary_band_radii: Tuple[int, ...] = (1, 1, 2, 3),
        component_editor_enabled: bool = True,
        component_editor_start_epoch: int = 2,
        component_editor_ramp_epochs: int = 8,
        component_editor_route_temperature: float = 0.70,
        component_editor_preserve_bias: float = 1.25,
        component_editor_dose_adjust_min: float = 0.50,
        component_editor_dose_adjust_max: float = 1.50,
        component_editor_local_residual_cap: float = 0.75,
        component_editor_total_logit_delta_cap: float = 1.00,
        component_editor_region_radii: Tuple[int, ...] = (1, 2, 4, 7),
        component_editor_residual_dropout: float = 0.10,
        component_unified_deployment_gate: bool = True,
        component_deployment_benefit_harm_margin: float = 0.10,
        component_multicandidate_composer_enabled: bool = True,
        component_composer_stop_bias: float = 0.0,
        component_composer_marginal_correction_cap: float = 0.02,
        component_composer_overlap_penalty: float = 0.50,
        component_composer_conflict_penalty: float = 1.00,
        component_composer_budget_penalty: float = 0.50,
        component_teacher_decoupled_r2_enabled: bool = False,
        component_composer_teacher_pool_size: int = 4,
        component_composer_teacher_stop_margin: float = 0.0,
        component_composer_deploy_pool_size: int = 3,
        component_shadow_evidence_start_epoch: int = 14,
        component_unified_reference_r4_enabled: bool = False,
        component_editor_relative_margin: float = 1.0e-3,
        component_editor_safety_benefit_threshold: float = 0.45,
        component_editor_safety_harm_threshold: float = 0.35,
        component_editor_incremental_gain_threshold: float = 0.0,
        component_critic_gain_cap: float = 0.05,
        component_critic_queue_capacity: int = 256,
        component_decoupled_critic_r42_enabled: bool = False,
        component_spatial_evidence_r43_enabled: bool = False,
        component_audit_gate_r44_enabled: bool = False,
        component_class_value_decoupling_r44_enabled: bool = False,
        component_semantic_deployment_r44_enabled: bool = False,
        component_audit_shadow_start_epoch: int = 12,
        component_audit_shadow_topk: int = 1,
        component_error_aware_r45_enabled: bool = False,
        component_factorized_safety_r45_enabled: bool = False,
        component_direct_signed_utility_r45_enabled: bool = False,
        component_factorized_composer_r45_enabled: bool = False,
        component_native_contract_r46_enabled: bool = False,
        component_spatial_realization_r47_enabled: bool = False,
        component_r47_mask_dim: int = 64,
        component_r47_query_logit_scale: float = 4.0,
        component_r47_anchor_prior_scale: float = 1.5,
        component_r47_anchor_min_size: float = 0.04,
        component_r47_anchor_max_size: float = 0.55,
        component_r47_direct_slot_components: bool = False,
        component_iterative_binding_r48_enabled: bool = False,
        component_r48_decoder_layers: int = 3,
        component_r48_local_grid_size: int = 3,
        component_r48_anchor_delta_scale: float = 0.75,
        component_r48_window_prior_scale: float = 2.0,
        component_r48_window_temperature: float = 0.025,
        component_r48_remove_coarse_mask_bias: bool = True,
        component_r48_deep_supervision_enabled: bool = False,
        component_r48_dn_enabled: bool = False,
        component_r48_dn_groups: int = 2,
        component_r48_dn_noise_scale: float = 0.35,
        component_r48_teacher_min_pixels: int = 4,
        component_content_selective_r49_enabled: bool = False,
        component_r49_anchor_sampling_only: bool = True,
        component_r49_attention_logit_scale_init: float = 10.0,
        component_r49_attention_logit_scale_max: float = 100.0,
        component_r49_dn_curriculum_enabled: bool = False,
        component_r49_dn_noise_start: float = 0.05,
        component_r49_dn_noise_final: float = 0.30,
        component_r49_dn_noise_ramp_epochs: int = 15,
        component_evidence_proposal_r410_enabled: bool = False,
        component_r410_use_evidence_proposals: bool = True,
        component_r410_support_only_local_readout: bool = False,
        component_r410_proposal_nms_kernel: int = 17,
        component_r410_proposal_score_threshold: float = 0.05,
        component_r410_support_expand: float = 1.75,
        component_r410_support_temperature: float = 0.02,
        component_r410_support_max_penalty: float = 8.0,
        component_r410_dn_clean_curriculum_enabled: bool = True,
        component_r410_dn_clean_epochs: int = 10,
        component_r410_dn_noise_final: float = 0.20,
        component_r410_dn_noise_ramp_epochs: int = 50,
        component_native_residual_set_r411_enabled: bool = False,
        component_r411_typed_proposal_enabled: bool = True,
        component_r411_local_roi_decoder_enabled: bool = True,
        component_r411_use_raw_native_masks: bool = True,
        component_r411_proposal_nms_kernel: int = 9,
        component_r411_proposal_score_threshold: float = 0.01,
        component_r411_initial_box_size: float = 0.08,
        component_r411_roi_size: int = 32,
        component_r411_roi_expand: float = 1.50,
        component_r411_outside_penalty: float = 8.0,
        component_canonical_shape_r412_enabled: bool = False,
        component_r412_roi_size: int = 64,
        component_r412_action_support_strength: float = 0.75,
        component_r412_action_support_floor: float = 0.05,
        component_r412_boundary_band_kernel: int = 7,
        component_geometry_lock_r413_enabled: bool = False,
        component_r413_query_extent_enabled: bool = True,
        component_geometry_context_r414_enabled: bool = False,
        component_r414_context_grid_size: int = 17,
        component_r414_context_radius: float = 0.20,
        component_unique_point_r416_enabled: bool = False,
        component_r416_cross_type_nms_radius_px: float = 4.0,
        component_r416_asymmetric_ltrb_enabled: bool = False,
        component_proposal_recovery_r417_enabled: bool = False,
        component_r417_location_nms_kernel: int = 3,
        component_r417_location_oversample_factor: int = 4,
        component_r417_location_dedup_radius_px: float = 2.0,
        component_r417_shared_offset_enabled: bool = True,
        component_box_free_mask_set_r418_enabled: bool = False,
        component_r418_paired_stable_teacher_enabled: bool = True,
        component_r418_paired_min_pixels: int = 4,
        component_seeded_masked_attention_r419_enabled: bool = False,
        component_r419_seed_radius_px: float = 12.0,
        component_r419_support_dilate_kernel: int = 9,
        component_r419_outside_logit_penalty: float = 8.0,
        component_r419_mask_threshold: float = 0.5,
        component_dynamic_residual_mask_r420_enabled: bool = False,
        component_r420_dynamic_channels: int = 8,
        component_r420_type_decoupled_mask_enabled: bool = False,
        component_r4201_clean_rootfix_enabled: bool = False,
        component_dense_competitive_residual_set_r4203_enabled: bool = False,
        component_factorized_residual_existence_identity_r4204_enabled: bool = False,
        component_r4204_spatial_identity_enabled: bool = False,
        component_capacity_consistent_overflow_r4205_enabled: bool = False,
        component_dynamic_visual_instance_binding_r4207_enabled: bool = False,
        component_normalized_visual_instance_binding_r4208_enabled: bool = False,
        component_persistent_identity_r4208_enabled: bool = False,
        component_instance_valid_factorization_r4210_enabled: bool = False,
        component_variable_cardinality_seeds_r4210_enabled: bool = False,
        component_independent_overflow_gate_r4210_enabled: bool = False,
        component_m1_native_alignment_r4210_enabled: bool = False,
        component_instance_valid_decoupling_r4211_enabled: bool = False,
        component_proposal_existence_decoupling_r4211_enabled: bool = False,
        component_geometry_overflow_decoupling_r4211_enabled: bool = False,
        component_independent_candidate_set_r4212_enabled: bool = False,
        component_disable_visual_seed_identity_r4212_enabled: bool = False,
        component_existence_no_object_r4212_enabled: bool = False,
        component_candidate_alignment_r4212_enabled: bool = False,
        component_direct_delta_utility_r4212_enabled: bool = False,
        component_zero_stop_one_step_r4212_enabled: bool = False,
        component_clean_core_v560_enabled: bool = False,
        component_base_conditioned_residual_set_v561_enabled: bool = False,
        component_bcrs_v561_variant: str = "typed",
        component_v563_attention_radius: float = 0.24,
        component_v563_mask_radius: float = 0.20,
        component_v563_identity_mix: float = 0.60,
        component_v563_query_residual_scale: float = 0.15,
        component_v563_outside_logit_penalty: float = 12.0,
        component_v564_rootfix_enabled: bool = False,
        component_v564_attention_identity_scale: float = 0.35,
        component_v564_proposal_shape_scale: float = 0.50,
        component_v564_min_mask_radius: float = 0.05,
        component_v564_max_mask_radius: float = 0.20,
        component_v565_rootfix_enabled: bool = False,
        component_v565_seed_nms_radius: float = 0.025,
        component_v565_support_relative_threshold: float = 0.35,
        component_v565_extent_quantile: float = 0.90,
        component_v565_attention_radius_scale: float = 1.35,
        component_v565_max_attention_radius: float = 0.20,
        component_v565_shape_scale: float = 0.75,
        clean_dynamic_component_set: bool = False,
        tc_drcs: bool = False,
        component_clean_dynamic_set_enabled: bool = False,
        component_tc_drcs_enabled: bool = False,
        deploy_start_epoch: int = 3,
        m1_grad_start_epoch: int = 8,
        m1_grad_ramp_epochs: int = 12,
        m1_grad_final_scale: float = 0.25,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        semantic_dim = int(semantic_dim)
        self.use_semantic = bool(use_semantic)
        self.semantic_dim = semantic_dim
        self.edit_temperature = max(float(edit_temperature), 1.0e-4)
        self.action_temperature = max(float(action_temperature), 1.0e-4)
        self.outcome_temperature = max(float(outcome_temperature), 1.0e-4)
        self.risk_temperature = max(float(risk_temperature), 1.0e-4)
        self.harm_penalty = max(float(harm_penalty), 0.0)
        self.edit_penalty = max(float(edit_penalty), 0.0)
        self.utility_threshold = float(utility_threshold)
        self.max_action_alpha = min(max(float(max_action_alpha), 0.0), 1.0)
        self.deployment_hard_route = bool(deployment_hard_route)
        self.execute_threshold = min(max(float(execute_threshold), 0.0), 1.0)
        # CLEAN is a first-class protocol, not a YAML reconstruction of V535.
        # The shared V532 shell historically reaches the online component
        # generator through the adaptive-policy branch.  Make that an internal
        # construction invariant so the formal CLEAN YAML does not need to
        # re-introduce V535_ADAPTIVE_UTILITY_POLICY_ENABLED.
        self.clean_dynamic_component_set = bool(clean_dynamic_component_set)
        self.tc_drcs = bool(tc_drcs)
        self.adaptive_utility_policy = bool(
            adaptive_utility_policy
            or (self.clean_dynamic_component_set and bool(online_component_refinement))
        )
        self.policy_temperature = max(float(policy_temperature), 1.0e-4)
        self.prior_aligned_case_component = bool(prior_aligned_case_component)
        self.component_utility_ranking = bool(component_utility_ranking)
        self.online_component_refinement = bool(online_component_refinement)
        self.multiscale_typed_editor = bool(multiscale_typed_editor)
        self.component_min_pixels = max(int(component_min_pixels), 1)
        self.max_components = max(int(max_components), 1)
        self.deploy_start_epoch = max(int(deploy_start_epoch), 0)
        self.m1_grad_start_epoch = max(int(m1_grad_start_epoch), 0)
        self.m1_grad_ramp_epochs = max(int(m1_grad_ramp_epochs), 1)
        self.m1_grad_final_scale = min(
            max(float(m1_grad_final_scale), 0.0), 1.0
        )
        self.current_epoch = 0
        self.register_buffer(
            "v538_quality_ready",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )
        self.register_buffer(
            "v538_quality_ready_streak",
            torch.tensor(0, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "v552_quality_bad_streak",
            torch.tensor(0, dtype=torch.long),
            persistent=True,
        )
        if (self.component_utility_ranking or self.online_component_refinement) and not self.adaptive_utility_policy:
            raise ValueError(
                "Legacy V537/V538 component refinement requires adaptive_utility_policy=True. "
                "CLEAN satisfies this compatibility invariant internally and must not expose "
                "the historical V535 switch in its formal YAML."
            )
        if self.component_utility_ranking and self.online_component_refinement:
            raise ValueError("Enable either V537 or V538 component refinement, not both.")

        groups = _group_count(hidden_dim)
        # P0, 1-P0, entropy, boundary, gray, edge, four causes, four alphas.
        self.context_encoder = nn.Sequential(
            nn.Conv2d(14, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            nn.Dropout2d(float(dropout)),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
        )

        if self.use_semantic:
            semantic_groups = _group_count(semantic_dim)
            self.semantic_proj = nn.Sequential(
                nn.Conv2d(
                    int(semantic_channels),
                    semantic_dim,
                    kernel_size=1,
                    bias=False,
                ),
                nn.GroupNorm(semantic_groups, semantic_dim),
                nn.GELU(),
            )
            self.semantic_fuse = nn.Sequential(
                nn.Conv2d(
                    hidden_dim + semantic_dim,
                    hidden_dim,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(groups, hidden_dim),
                nn.GELU(),
            )
        else:
            self.semantic_proj = None
            self.semantic_fuse = None

        # V535 uses one four-channel signed-utility head.  Preserve is a
        # parameter-free zero-utility reference, so one argmax jointly decides
        # Preserve versus Delete/Fill/Trim/Expand.  This replaces three heads
        # and removes the probability-product calibration failure.
        if self.adaptive_utility_policy:
            self.utility_policy_head = nn.Conv2d(
                hidden_dim, 4, kernel_size=1
            )
            # Preserve has fixed logit zero.  The four action biases jointly
            # realise the configured total edit prior, so Soft and Hard begin
            # from the same Preserve-first policy instead of V535's 80% soft
            # execute / 0% hard execute contradiction.
            nn.init.zeros_(self.utility_policy_head.weight)
            if self.prior_aligned_case_component or self.component_utility_ranking or self.online_component_refinement:
                initial_edit_rate = min(
                    max(float(initial_edit_rate), 1.0e-6), 1.0 - 1.0e-6
                )
                action_prior_bias = math.log(
                    initial_edit_rate
                    / (4.0 * (1.0 - initial_edit_rate))
                )
                nn.init.constant_(self.utility_policy_head.bias, action_prior_bias)
                if self.prior_aligned_case_component:
                    self.case_policy_head = nn.Linear(hidden_dim, 4)
                    nn.init.zeros_(self.case_policy_head.weight)
                    nn.init.constant_(self.case_policy_head.bias, action_prior_bias)
                else:
                    self.case_policy_head = None
            else:
                nn.init.zeros_(self.utility_policy_head.bias)
                self.case_policy_head = None
            self.component_ranker = (
                V537ComponentUtilityRanker(
                    feature_channels=hidden_dim,
                    hidden_dim=int(component_ranker_hidden_dim),
                    dropout=float(component_ranker_dropout),
                    min_pixels=self.component_min_pixels,
                    max_components=self.max_components,
                    initial_score_bias=float(component_initial_score_bias),
                )
                if self.component_utility_ranking
                else None
            )
            component_refiner_class = (
                V551MultiscaleTypedComposerEditor
                if self.multiscale_typed_editor
                else V538OnlineComponentRefiner
            )
            component_refiner_extra = (
                dict(
                    pyramid_scales=tuple(component_pyramid_scales),
                    scale_temperature=float(component_scale_temperature),
                    max_atoms_per_slot=int(component_max_atoms_per_slot),
                    atomization_start_epoch=int(component_atomization_start_epoch),
                    atom_min_pixels=int(component_atom_min_pixels),
                    atom_dedup_iou=float(component_atom_dedup_iou),
                    gpu_atomizer_enabled=bool(component_gpu_atomizer_enabled),
                    max_active_atoms=int(component_max_active_atoms),
                    atom_partition_temperature=float(component_atom_partition_temperature),
                    atom_partition_extent=float(component_atom_partition_extent),
                    atom_presence_threshold=float(component_atom_presence_threshold),
                    atom_quality_threshold=float(component_atom_quality_threshold),
                    atom_quality_gate_start_epoch=int(component_atom_quality_gate_start_epoch),
                    single_pass_editor=bool(component_single_pass_editor),
                    boundary_residual_enabled=bool(component_boundary_residual_enabled),
                    boundary_residual_cap=float(component_boundary_residual_cap),
                    boundary_band_radii=tuple(component_boundary_band_radii),
                    editor_enabled=bool(component_editor_enabled),
                    editor_start_epoch=int(component_editor_start_epoch),
                    editor_ramp_epochs=int(component_editor_ramp_epochs),
                    editor_route_temperature=float(component_editor_route_temperature),
                    editor_preserve_bias=float(component_editor_preserve_bias),
                    editor_dose_adjust_min=float(component_editor_dose_adjust_min),
                    editor_dose_adjust_max=float(component_editor_dose_adjust_max),
                    editor_local_residual_cap=float(component_editor_local_residual_cap),
                    editor_total_logit_delta_cap=float(component_editor_total_logit_delta_cap),
                    editor_region_radii=tuple(component_editor_region_radii),
                    editor_residual_dropout=float(component_editor_residual_dropout),
                    unified_deployment_gate=bool(component_unified_deployment_gate),
                    deployment_benefit_harm_margin=float(component_deployment_benefit_harm_margin),
                    multicandidate_composer_enabled=bool(
                        component_multicandidate_composer_enabled
                    ),
                    composer_stop_bias=float(component_composer_stop_bias),
                    composer_marginal_correction_cap=float(
                        component_composer_marginal_correction_cap
                    ),
                    composer_overlap_penalty=float(
                        component_composer_overlap_penalty
                    ),
                    composer_conflict_penalty=float(
                        component_composer_conflict_penalty
                    ),
                    composer_budget_penalty=float(
                        component_composer_budget_penalty
                    ),
                    teacher_decoupled_r2_enabled=bool(
                        component_teacher_decoupled_r2_enabled
                    ),
                    composer_teacher_pool_size=int(
                        component_composer_teacher_pool_size
                    ),
                    composer_teacher_stop_margin=float(
                        component_composer_teacher_stop_margin
                    ),
                    composer_deploy_pool_size=int(
                        component_composer_deploy_pool_size
                    ),
                    shadow_evidence_start_epoch=int(
                        component_shadow_evidence_start_epoch
                    ),
                    unified_reference_r4_enabled=bool(
                        component_unified_reference_r4_enabled
                    ),
                    editor_relative_margin=float(
                        component_editor_relative_margin
                    ),
                    editor_safety_benefit_threshold=float(
                        component_editor_safety_benefit_threshold
                    ),
                    editor_safety_harm_threshold=float(
                        component_editor_safety_harm_threshold
                    ),
                    editor_incremental_gain_threshold=float(
                        component_editor_incremental_gain_threshold
                    ),
                    critic_gain_cap=float(component_critic_gain_cap),
                    critic_queue_capacity=int(component_critic_queue_capacity),
                    decoupled_critic_r42_enabled=bool(
                        component_decoupled_critic_r42_enabled
                    ),
                    spatial_evidence_r43_enabled=bool(
                        component_spatial_evidence_r43_enabled
                    ),
                    audit_gate_r44_enabled=bool(
                        component_audit_gate_r44_enabled
                    ),
                    class_value_decoupling_r44_enabled=bool(
                        component_class_value_decoupling_r44_enabled
                    ),
                    semantic_deployment_r44_enabled=bool(
                        component_semantic_deployment_r44_enabled
                    ),
                    audit_shadow_start_epoch=int(
                        component_audit_shadow_start_epoch
                    ),
                    audit_shadow_topk=int(component_audit_shadow_topk),
                    error_aware_r45_enabled=bool(
                        component_error_aware_r45_enabled
                    ),
                    factorized_safety_r45_enabled=bool(
                        component_factorized_safety_r45_enabled
                    ),
                    direct_signed_utility_r45_enabled=bool(
                        component_direct_signed_utility_r45_enabled
                    ),
                    factorized_composer_r45_enabled=bool(
                        component_factorized_composer_r45_enabled
                    ),
                    native_contract_r46_enabled=bool(
                        component_native_contract_r46_enabled
                    ),
                    spatial_realization_r47_enabled=bool(
                        component_spatial_realization_r47_enabled
                    ),
                    r47_mask_dim=int(component_r47_mask_dim),
                    r47_query_logit_scale=float(component_r47_query_logit_scale),
                    r47_anchor_prior_scale=float(component_r47_anchor_prior_scale),
                    r47_anchor_min_size=float(component_r47_anchor_min_size),
                    r47_anchor_max_size=float(component_r47_anchor_max_size),
                    r47_direct_slot_components=bool(component_r47_direct_slot_components),
                    iterative_binding_r48_enabled=bool(component_iterative_binding_r48_enabled),
                    r48_decoder_layers=int(component_r48_decoder_layers),
                    r48_local_grid_size=int(component_r48_local_grid_size),
                    r48_anchor_delta_scale=float(component_r48_anchor_delta_scale),
                    r48_window_prior_scale=float(component_r48_window_prior_scale),
                    r48_window_temperature=float(component_r48_window_temperature),
                    r48_remove_coarse_mask_bias=bool(component_r48_remove_coarse_mask_bias),
                    r48_deep_supervision_enabled=bool(component_r48_deep_supervision_enabled),
                    r48_dn_enabled=bool(component_r48_dn_enabled),
                    r48_dn_groups=int(component_r48_dn_groups),
                    r48_dn_noise_scale=float(component_r48_dn_noise_scale),
                    r48_teacher_min_pixels=int(component_r48_teacher_min_pixels),
                    content_selective_r49_enabled=bool(component_content_selective_r49_enabled),
                    r49_anchor_sampling_only=bool(component_r49_anchor_sampling_only),
                    r49_attention_logit_scale_init=float(component_r49_attention_logit_scale_init),
                    r49_attention_logit_scale_max=float(component_r49_attention_logit_scale_max),
                    r49_dn_curriculum_enabled=bool(component_r49_dn_curriculum_enabled),
                    r49_dn_noise_start=float(component_r49_dn_noise_start),
                    r49_dn_noise_final=float(component_r49_dn_noise_final),
                    r49_dn_noise_ramp_epochs=int(component_r49_dn_noise_ramp_epochs),
                    evidence_proposal_r410_enabled=bool(component_evidence_proposal_r410_enabled),
                    r410_use_evidence_proposals=bool(component_r410_use_evidence_proposals),
                    r410_support_only_local_readout=bool(component_r410_support_only_local_readout),
                    r410_proposal_nms_kernel=int(component_r410_proposal_nms_kernel),
                    r410_proposal_score_threshold=float(component_r410_proposal_score_threshold),
                    r410_support_expand=float(component_r410_support_expand),
                    r410_support_temperature=float(component_r410_support_temperature),
                    r410_support_max_penalty=float(component_r410_support_max_penalty),
                    r410_dn_clean_curriculum_enabled=bool(component_r410_dn_clean_curriculum_enabled),
                    r410_dn_clean_epochs=int(component_r410_dn_clean_epochs),
                    r410_dn_noise_final=float(component_r410_dn_noise_final),
                    r410_dn_noise_ramp_epochs=int(component_r410_dn_noise_ramp_epochs),
                    native_residual_set_r411_enabled=bool(component_native_residual_set_r411_enabled),
                    r411_typed_proposal_enabled=bool(component_r411_typed_proposal_enabled),
                    r411_local_roi_decoder_enabled=bool(component_r411_local_roi_decoder_enabled),
                    r411_use_raw_native_masks=bool(component_r411_use_raw_native_masks),
                    r411_proposal_nms_kernel=int(component_r411_proposal_nms_kernel),
                    r411_proposal_score_threshold=float(component_r411_proposal_score_threshold),
                    r411_initial_box_size=float(component_r411_initial_box_size),
                    r411_roi_size=int(component_r411_roi_size),
                    r411_roi_expand=float(component_r411_roi_expand),
                    r411_outside_penalty=float(component_r411_outside_penalty),
                    canonical_shape_r412_enabled=bool(component_canonical_shape_r412_enabled),
                    r412_roi_size=int(component_r412_roi_size),
                    r412_action_support_strength=float(component_r412_action_support_strength),
                    r412_action_support_floor=float(component_r412_action_support_floor),
                    r412_boundary_band_kernel=int(component_r412_boundary_band_kernel),
                    geometry_lock_r413_enabled=bool(component_geometry_lock_r413_enabled),
                    r413_query_extent_enabled=bool(component_r413_query_extent_enabled),
                    geometry_context_r414_enabled=bool(component_geometry_context_r414_enabled),
                    r414_context_grid_size=int(component_r414_context_grid_size),
                    r414_context_radius=float(component_r414_context_radius),
                    unique_point_r416_enabled=bool(component_unique_point_r416_enabled),
                    r416_cross_type_nms_radius_px=float(component_r416_cross_type_nms_radius_px),
                    r416_asymmetric_ltrb_enabled=bool(component_r416_asymmetric_ltrb_enabled),
                    proposal_recovery_r417_enabled=bool(component_proposal_recovery_r417_enabled),
                    r417_location_nms_kernel=int(component_r417_location_nms_kernel),
                    r417_location_oversample_factor=int(component_r417_location_oversample_factor),
                    r417_location_dedup_radius_px=float(component_r417_location_dedup_radius_px),
                    r417_shared_offset_enabled=bool(component_r417_shared_offset_enabled),
                    box_free_mask_set_r418_enabled=bool(component_box_free_mask_set_r418_enabled),
                    r418_paired_stable_teacher_enabled=bool(component_r418_paired_stable_teacher_enabled),
                    r418_paired_min_pixels=int(component_r418_paired_min_pixels),
                    seeded_masked_attention_r419_enabled=bool(component_seeded_masked_attention_r419_enabled),
                    r419_seed_radius_px=float(component_r419_seed_radius_px),
                    r419_support_dilate_kernel=int(component_r419_support_dilate_kernel),
                    r419_outside_logit_penalty=float(component_r419_outside_logit_penalty),
                    r419_mask_threshold=float(component_r419_mask_threshold),
                    dynamic_residual_mask_r420_enabled=bool(component_dynamic_residual_mask_r420_enabled),
                    r420_dynamic_channels=int(component_r420_dynamic_channels),
                    r420_type_decoupled_mask_enabled=bool(component_r420_type_decoupled_mask_enabled),
                    r4201_clean_rootfix_enabled=bool(component_r4201_clean_rootfix_enabled),
                    dense_competitive_residual_set_r4203_enabled=bool(
                        component_dense_competitive_residual_set_r4203_enabled
                    ),
                    factorized_residual_existence_identity_r4204_enabled=bool(
                        component_factorized_residual_existence_identity_r4204_enabled
                    ),
                    r4204_spatial_identity_enabled=bool(
                        component_r4204_spatial_identity_enabled
                    ),
                    capacity_consistent_overflow_r4205_enabled=bool(
                        component_capacity_consistent_overflow_r4205_enabled
                    ),
                    dynamic_visual_instance_binding_r4207_enabled=bool(
                        component_dynamic_visual_instance_binding_r4207_enabled
                    ),
                    normalized_visual_instance_binding_r4208_enabled=bool(
                        component_normalized_visual_instance_binding_r4208_enabled
                    ),
                    persistent_identity_r4208_enabled=bool(
                        component_persistent_identity_r4208_enabled
                    ),
                    instance_valid_factorization_r4210_enabled=bool(
                        component_instance_valid_factorization_r4210_enabled
                    ),
                    variable_cardinality_seeds_r4210_enabled=bool(
                        component_variable_cardinality_seeds_r4210_enabled
                    ),
                    independent_overflow_gate_r4210_enabled=bool(
                        component_independent_overflow_gate_r4210_enabled
                    ),
                    m1_native_alignment_r4210_enabled=bool(
                        component_m1_native_alignment_r4210_enabled
                    ),
                    instance_valid_decoupling_r4211_enabled=bool(
                        component_instance_valid_decoupling_r4211_enabled
                    ),
                    proposal_existence_decoupling_r4211_enabled=bool(
                        component_proposal_existence_decoupling_r4211_enabled
                    ),
                    geometry_overflow_decoupling_r4211_enabled=bool(
                        component_geometry_overflow_decoupling_r4211_enabled
                    ),
                    independent_candidate_set_r4212_enabled=bool(
                        component_independent_candidate_set_r4212_enabled
                    ),
                    disable_visual_seed_identity_r4212_enabled=bool(
                        component_disable_visual_seed_identity_r4212_enabled
                    ),
                    existence_no_object_r4212_enabled=bool(
                        component_existence_no_object_r4212_enabled
                    ),
                    candidate_alignment_r4212_enabled=bool(
                        component_candidate_alignment_r4212_enabled
                    ),
                    direct_delta_utility_r4212_enabled=bool(
                        component_direct_delta_utility_r4212_enabled
                    ),
                    zero_stop_one_step_r4212_enabled=bool(
                        component_zero_stop_one_step_r4212_enabled
                    ),
                    clean_core_v560_enabled=bool(component_clean_core_v560_enabled),
                    base_conditioned_residual_set_v561_enabled=bool(
                        component_base_conditioned_residual_set_v561_enabled
                    ),
                    bcrs_v561_variant=str(component_bcrs_v561_variant),
                    v563_attention_radius=float(component_v563_attention_radius),
                    v563_mask_radius=float(component_v563_mask_radius),
                    v563_identity_mix=float(component_v563_identity_mix),
                    v563_query_residual_scale=float(component_v563_query_residual_scale),
                    v563_outside_logit_penalty=float(component_v563_outside_logit_penalty),
                    v564_rootfix_enabled=bool(component_v564_rootfix_enabled),
                    v564_attention_identity_scale=float(component_v564_attention_identity_scale),
                    v564_proposal_shape_scale=float(component_v564_proposal_shape_scale),
                    v564_min_mask_radius=float(component_v564_min_mask_radius),
                    v564_max_mask_radius=float(component_v564_max_mask_radius),
                    v565_rootfix_enabled=bool(component_v565_rootfix_enabled),
                    v565_seed_nms_radius=float(component_v565_seed_nms_radius),
                    v565_support_relative_threshold=float(component_v565_support_relative_threshold),
                    v565_extent_quantile=float(component_v565_extent_quantile),
                    v565_attention_radius_scale=float(component_v565_attention_radius_scale),
                    v565_max_attention_radius=float(component_v565_max_attention_radius),
                    v565_shape_scale=float(component_v565_shape_scale),
                    clean_dynamic_component_set_enabled=bool(component_clean_dynamic_set_enabled),
                    tc_drcs_enabled=bool(component_tc_drcs_enabled),
                )
                if self.multiscale_typed_editor
                else {}
            )
            self.component_slot_generator = (
                component_refiner_class(
                    feature_channels=hidden_dim,
                    **component_refiner_extra,
                    hidden_dim=int(component_slot_hidden_dim),
                    num_slots=int(component_num_slots),
                    dropout=float(component_slot_dropout),
                    mask_temperature=float(component_mask_temperature),
                    action_temperature=float(component_action_temperature),
                    min_area_fraction=float(component_min_area_fraction),
                    max_component_area_fraction=float(component_max_area_fraction),
                    initial_presence_rate=float(component_initial_presence_rate),
                    initial_gain_bias=float(component_initial_score_bias),
                    deployment_mask_threshold=float(component_deploy_mask_threshold),
                    deployment_presence_threshold=float(component_deploy_presence_threshold),
                    deployment_min_gain=float(component_deploy_min_gain),
                    max_steps=int(component_max_steps),
                    max_overlap=float(component_max_overlap),
                    max_total_edit_fraction=float(component_max_total_edit_fraction),
                    continuous_dose_enabled=bool(component_continuous_dose_enabled),
                    polarity_temperature=float(component_polarity_temperature),
                    minimum_dose=float(component_minimum_dose),
                    maximum_dose=float(component_maximum_dose),
                    initial_dose=float(component_initial_dose),
                    candidate_outcome_selector_enabled=bool(
                        component_candidate_outcome_selector_enabled
                    ),
                    selector_spatial_size=int(component_selector_spatial_size),
                    selector_hidden_dim=int(component_selector_hidden_dim),
                    selector_dropout=float(component_selector_dropout),
                    selector_gain_scale=float(component_selector_gain_scale),
                    selector_benefit_threshold=float(
                        component_selector_benefit_threshold
                    ),
                    selector_harm_threshold=float(
                        component_selector_harm_threshold
                    ),
                    selector_lcb_beta=float(component_selector_lcb_beta),
                    selector_logvar_min=float(component_selector_logvar_min),
                    selector_logvar_max=float(component_selector_logvar_max),
                    selector_initial_benefit_rate=float(
                        component_selector_initial_benefit_rate
                    ),
                    selector_initial_harm_rate=float(
                        component_selector_initial_harm_rate
                    ),
                    use_gain_as_decision_score=bool(
                        component_use_gain_as_decision_score
                    ),
                    adaptive_cardinality_hard_mask=bool(
                        component_adaptive_cardinality_hard_mask
                    ),
                    prior_free_outcome_init=bool(
                        component_prior_free_outcome_init
                    ),
                    slot_competition_enabled=bool(
                        component_slot_competition_enabled
                    ),
                    gain_sign_shadow_deploy_enabled=bool(
                        component_gain_sign_shadow_deploy_enabled
                    ),
                    factorized_outcome_enabled=bool(
                        component_factorized_outcome_enabled
                    ),
                    factorized_direction_zero_init=bool(
                        component_factorized_direction_zero_init
                    ),
                    factorized_deployment_enabled=bool(
                        component_factorized_deployment_enabled
                    ),
                    deployment_editability_threshold=float(
                        component_deploy_editability_threshold
                    ),
                    deployment_direction_threshold=float(
                        component_deploy_direction_threshold
                    ),
                )
                if self.online_component_refinement
                else None
            )
            if self.clean_dynamic_component_set:
                # This 1x1 policy head is only a structural bridge in the old
                # V532 shell.  CLEAN M1/M2 losses are owned by the component
                # generator, so keeping this historical head trainable would
                # reintroduce dead optimizer parameters without any consumer.
                for parameter in self.utility_policy_head.parameters():
                    parameter.requires_grad_(False)
                if self.case_policy_head is not None:
                    for parameter in self.case_policy_head.parameters():
                        parameter.requires_grad_(False)
            self.edit_head = None
            self.action_head = None
            self.outcome_head = None
        else:
            self.utility_policy_head = None
            self.case_policy_head = None
            self.component_ranker = None
            self.component_slot_generator = None
            self.edit_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
            self.action_head = nn.Conv2d(hidden_dim, 4, kernel_size=1)
            self.outcome_head = nn.Conv2d(hidden_dim, 4 * 3, kernel_size=1)

            nn.init.zeros_(self.edit_head.weight)
            initial_edit_rate = min(
                max(float(initial_edit_rate), 1.0e-4), 1.0 - 1.0e-4
            )
            nn.init.constant_(
                self.edit_head.bias,
                math.log(initial_edit_rate / (1.0 - initial_edit_rate)),
            )
            nn.init.zeros_(self.action_head.weight)
            nn.init.zeros_(self.action_head.bias)
            nn.init.zeros_(self.outcome_head.weight)
            outcome_bias = _initial_outcome_bias(
                initial_outcome_neutral,
                initial_outcome_benefit,
                initial_outcome_harm,
            )
            with torch.no_grad():
                self.outcome_head.bias.copy_(outcome_bias.repeat(4))

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = max(int(epoch), 0)
        generator = getattr(self, "component_slot_generator", None)
        if generator is not None and hasattr(generator, "set_epoch"):
            generator.set_epoch(self.current_epoch)

    @torch.no_grad()
    def update_v538_quality_ready(
        self,
        epoch_ready: bool,
        patience: int = 3,
        close_patience: int = 1,
    ) -> bool:
        """Reversible safety latch for deployment.

        V551 opened the latch permanently.  V552 closes it again after
        consecutive unsafe epochs, preventing a once-good selector from
        remaining deployed after calibration drift.
        """
        patience = max(int(patience), 1)
        close_patience = max(int(close_patience), 1)
        if not self.online_component_refinement:
            return False
        if bool(epoch_ready):
            self.v538_quality_ready_streak.add_(1)
            self.v552_quality_bad_streak.zero_()
            if int(self.v538_quality_ready_streak.item()) >= patience:
                self.v538_quality_ready.fill_(True)
        else:
            self.v538_quality_ready_streak.zero_()
            self.v552_quality_bad_streak.add_(1)
            if (
                bool(self.v538_quality_ready.item())
                and int(self.v552_quality_bad_streak.item()) >= close_patience
            ):
                self.v538_quality_ready.fill_(False)
        return bool(self.v538_quality_ready.item())

    def _m1_gradient_scale(self) -> float:
        if self.current_epoch < self.m1_grad_start_epoch:
            return 0.0
        if self.m1_grad_ramp_epochs <= 1:
            return self.m1_grad_final_scale
        progress = min(
            1.0,
            max(
                0.0,
                float(self.current_epoch - self.m1_grad_start_epoch)
                / float(self.m1_grad_ramp_epochs - 1),
            ),
        )
        return progress * self.m1_grad_final_scale

    def _fuse_semantic(
        self,
        feature: torch.Tensor,
        semantic_map: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.use_semantic:
            return feature
        if not isinstance(semantic_map, torch.Tensor):
            semantic = feature.new_zeros(
                feature.shape[0],
                self.semantic_dim,
                *feature.shape[-2:],
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
    def _action_candidates(
        c0: torch.Tensor,
        alphas: torch.Tensor,
    ) -> torch.Tensor:
        delete = c0 * (1.0 - alphas[:, 0:1])
        fill = c0 + (1.0 - c0) * alphas[:, 1:2]
        trim = c0 * (1.0 - alphas[:, 2:3])
        expand = c0 + (1.0 - c0) * alphas[:, 3:4]
        return torch.cat([delete, fill, trim, expand], dim=1)

    def forward(
        self,
        *,
        image: torch.Tensor,
        c0_prob: torch.Tensor,
        cause_map_probs: torch.Tensor,
        action_alpha: torch.Tensor,
        semantic_map: Optional[torch.Tensor] = None,
        supervision_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        c0 = _as_b1hw(c0_prob).clamp(EPS, 1.0 - EPS)
        if self.online_component_refinement:
            # V538 treats Base as a causal factual anchor.  Candidate and
            # selector losses may update M1/M2, never the online Base.
            c0 = c0.detach()
        size = c0.shape[-2:]

        if cause_map_probs.ndim != 4 or cause_map_probs.shape[1] != 4:
            raise ValueError("cause_map_probs must have shape [B,4,H,W]")
        if action_alpha.ndim != 4 or action_alpha.shape[1] != 4:
            raise ValueError("action_alpha must have shape [B,4,H,W]")
        if cause_map_probs.shape[-2:] != size:
            cause_map_probs = F.interpolate(
                cause_map_probs,
                size=size,
                mode="bilinear",
                align_corners=False,
            )
        if action_alpha.shape[-2:] != size:
            action_alpha = F.interpolate(
                action_alpha,
                size=size,
                mode="bilinear",
                align_corners=False,
            )

        causes = cause_map_probs.clamp(0.0, 1.0)
        alphas = action_alpha.clamp(0.0, self.max_action_alpha)
        m1_grad_scale = self._m1_gradient_scale()
        causes_for_refiner = _scale_gradient(causes, m1_grad_scale)
        alphas_for_refiner = _scale_gradient(alphas, m1_grad_scale)

        entropy = _entropy(c0)
        boundary = _soft_boundary(c0)
        gray, edge = _gray_edge(image, size)
        if self.clean_dynamic_component_set:
            # The formal CLEAN component geometry may depend only on current
            # image/Base evidence.  Historical cause/alpha heads remain in the
            # legacy compatibility dictionary, but cannot steer the shared
            # feature map consumed by the clean component generator.
            causes_for_context = torch.zeros_like(causes_for_refiner)
            alphas_for_context = torch.zeros_like(alphas_for_refiner)
        else:
            causes_for_context = causes_for_refiner
            alphas_for_context = alphas_for_refiner
        feature = self.context_encoder(
            torch.cat(
                [
                    c0,
                    1.0 - c0,
                    entropy,
                    boundary,
                    gray,
                    edge,
                    causes_for_context,
                    alphas_for_context,
                ],
                dim=1,
            )
        )
        feature = self._fuse_semantic(feature, semantic_map)
        action_candidates = self._action_candidates(c0, alphas)

        if self.adaptive_utility_policy:
            # V536: explicit hard deployment with separately supervised soft
            # probabilities.  No straight-through selector is used.
            action_utility_logits = self.utility_policy_head(feature)
            preserve_logit = action_utility_logits.new_zeros(
                action_utility_logits.shape[0],
                1,
                *action_utility_logits.shape[-2:],
            )
            policy_logits = torch.cat(
                [preserve_logit, action_utility_logits], dim=1
            )
            policy_soft = F.softmax(
                policy_logits / self.policy_temperature, dim=1
            )
            policy_index = policy_soft.argmax(dim=1, keepdim=True)
            policy_hard = torch.zeros_like(policy_soft).scatter_(
                1, policy_index, 1.0
            ).detach()

            v537_ranker_output = None
            v538_refiner_output = None
            if self.online_component_refinement:
                v538_refiner_output = self.component_slot_generator(
                    feature=feature,
                    action_candidates=action_candidates,
                    base_probability=c0,
                    cause_probability=causes,
                    action_alpha=alphas_for_refiner,
                    entropy=entropy,
                    boundary=boundary,
                    deploy_enabled=(
                        self.current_epoch >= self.deploy_start_epoch
                        and (
                            bool(self.v538_quality_ready.item())
                            or bool(getattr(
                                self.component_slot_generator,
                                "zero_stop_one_step_r4212_enabled",
                                False,
                            ))
                            or bool(getattr(
                                self.component_slot_generator,
                                "clean_core_v560_enabled",
                                False,
                            ))
                        )
                    ),
                    supervision_masks=supervision_masks,
                )
                # Reuse the V537 compatibility fields while exposing all
                # differentiable V538 slot tensors under explicit names.
                v537_ranker_output = v538_refiner_output
                safe_action_weight = v538_refiner_output["selected_action_weight"].detach()
                execute_gate = safe_action_weight.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
                preserve_route = 1.0 - execute_gate
                route_probs = torch.cat([preserve_route, safe_action_weight], dim=1)
                route_soft_probs = policy_soft
                case_policy_logits = torch.cat(
                    [v538_refiner_output["action_max_scores"].new_zeros(
                        v538_refiner_output["action_max_scores"].shape[0], 1
                    ), v538_refiner_output["action_max_scores"]], dim=1
                )
                case_policy_soft = F.softmax(
                    case_policy_logits / self.policy_temperature, dim=1
                )
                selected_action_zero = v538_refiner_output["candidate_actions"].gather(
                    1, v538_refiner_output["selected_index"][:, None]
                )[:, 0]
                predicted_execute = v538_refiner_output["predicted_execute"]
                case_policy_index = torch.where(
                    predicted_execute,
                    selected_action_zero + 1,
                    torch.zeros_like(selected_action_zero),
                )
                execute_probability = (
                    torch.sigmoid(v538_refiner_output["selected_score"] / self.policy_temperature)
                    [:, None, None, None] * execute_gate
                ).clamp(0.0, 1.0)
                component_accept = predicted_execute.to(policy_soft.dtype)
                component_area_fraction = v538_refiner_output["changed_fraction"]
                component_score = v538_refiner_output["selected_score"]
            elif self.component_utility_ranking:
                v537_ranker_output = self.component_ranker(
                    feature=feature,
                    action_candidates=action_candidates,
                    base_probability=c0,
                    cause_probability=causes,
                    action_alpha=alphas_for_refiner,
                    entropy=entropy,
                    boundary=boundary,
                    deploy_enabled=self.current_epoch >= self.deploy_start_epoch,
                )
                selected_mask = v537_ranker_output["selected_mask"]
                selected_action_zero = v537_ranker_output["selected_action"]
                predicted_execute = v537_ranker_output["predicted_execute"]
                selected_action_one_hot = F.one_hot(
                    selected_action_zero.clamp(0, 3), num_classes=4
                ).to(policy_soft.dtype)[:, :, None, None]
                safe_action_weight = (
                    selected_action_one_hot
                    * selected_mask
                    * predicted_execute[:, None, None, None].to(policy_soft.dtype)
                ).detach()
                execute_gate = safe_action_weight.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
                preserve_route = 1.0 - execute_gate
                route_probs = torch.cat([preserve_route, safe_action_weight], dim=1)
                route_soft_probs = policy_soft
                case_policy_logits = torch.cat(
                    [v537_ranker_output["action_max_scores"].new_zeros(
                        v537_ranker_output["action_max_scores"].shape[0], 1
                    ), v537_ranker_output["action_max_scores"]], dim=1
                )
                case_policy_soft = F.softmax(
                    case_policy_logits / self.policy_temperature, dim=1
                )
                case_policy_index = torch.where(
                    predicted_execute,
                    selected_action_zero + 1,
                    torch.zeros_like(selected_action_zero),
                )
                execute_probability = (
                    torch.sigmoid(v537_ranker_output["selected_score"] / self.policy_temperature)
                    [:, None, None, None] * selected_mask
                ).clamp(0.0, 1.0)
                component_accept = predicted_execute.to(policy_soft.dtype)
                component_area_fraction = selected_mask.flatten(1).mean(dim=1)
                component_score = v537_ranker_output["selected_score"]
            elif self.prior_aligned_case_component:
                pooled_feature = F.adaptive_avg_pool2d(feature, 1).flatten(1)
                case_action_logits = self.case_policy_head(pooled_feature)
                case_policy_logits = torch.cat(
                    [case_action_logits.new_zeros(case_action_logits.shape[0], 1),
                     case_action_logits],
                    dim=1,
                )
                case_policy_soft = F.softmax(
                    case_policy_logits / self.policy_temperature, dim=1
                )
                case_policy_index = case_policy_soft.argmax(dim=1)

                if self.current_epoch < self.deploy_start_epoch:
                    policy_index = torch.zeros_like(policy_index)
                    policy_hard = torch.zeros_like(policy_soft)
                    policy_hard[:, 0:1] = 1.0
                    case_policy_index = torch.zeros_like(case_policy_index)

                component_mask, component_decisions = select_top1_component(
                    policy_soft=policy_soft,
                    policy_hard_index=policy_index,
                    case_action_index=case_policy_index,
                    action_candidates=action_candidates,
                    base_probability=c0,
                    cause_probability=causes,
                    min_pixels=self.component_min_pixels,
                )
                case_action_one_hot = F.one_hot(
                    case_policy_index.clamp(0, 4), num_classes=5
                ).to(policy_soft.dtype)[:, 1:, None, None]
                safe_action_weight = (
                    policy_hard[:, 1:]
                    * case_action_one_hot
                    * component_mask
                ).detach()
                execute_gate = safe_action_weight.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
                preserve_route = 1.0 - execute_gate
                route_probs = torch.cat([preserve_route, safe_action_weight], dim=1)
                route_soft_probs = policy_soft
                case_execute_probability = (
                    1.0 - case_policy_soft[:, 0:1]
                )[:, :, None, None]
                execute_probability = (
                    (1.0 - policy_soft[:, 0:1]) * case_execute_probability
                ).clamp(0.0, 1.0)
                component_accept = execute_gate.flatten(1).amax(dim=1)
                component_area_fraction = execute_gate.flatten(1).mean(dim=1)
                component_score = action_utility_logits.new_zeros(
                    action_utility_logits.shape[0]
                )
                for item_index, decision in enumerate(component_decisions):
                    component_score[item_index] = float(decision.score)
            else:
                # Exact historical V535 path for backward compatibility.
                policy_st = (
                    policy_hard.detach() - policy_soft.detach() + policy_soft
                )
                route_probs = policy_st
                route_soft_probs = policy_soft
                preserve_route = policy_st[:, 0:1]
                safe_action_weight = policy_st[:, 1:]
                execute_gate = 1.0 - preserve_route
                execute_probability = 1.0 - policy_soft[:, 0:1]
                case_policy_logits = policy_logits.mean(dim=(2, 3))
                case_policy_soft = route_soft_probs.mean(dim=(2, 3))
                case_policy_index = case_policy_soft.argmax(dim=1)
                component_accept = execute_gate.flatten(1).amax(dim=1)
                component_area_fraction = execute_gate.flatten(1).mean(dim=1)
                component_score = execute_gate.flatten(1).mean(dim=1)

            # Compatibility fields shared by the V535 and V536 adaptive
            # policy paths.  They are derived from the same action-utility
            # logits and are used by the existing diagnostics/loss plumbing.
            edit_logit = torch.logsumexp(
                action_utility_logits / self.policy_temperature,
                dim=1,
                keepdim=True,
            )
            edit_probability = execute_probability
            action_logits = action_utility_logits
            action_conditional = F.softmax(
                action_utility_logits / self.policy_temperature, dim=1
            )
            action_hard_st = policy_hard[:, 1:]
            action_route_soft = policy_soft[:, 1:]

            neutral_logit = -action_utility_logits.abs()
            outcome_logits = torch.stack(
                [
                    neutral_logit,
                    action_utility_logits,
                    -action_utility_logits,
                ],
                dim=2,
            )
            outcome_probs = F.softmax(outcome_logits, dim=2)
            neutral_probability = outcome_probs[:, :, OUTCOME_NEUTRAL]
            benefit_probability = outcome_probs[:, :, OUTCOME_BENEFIT]
            harm_probability = outcome_probs[:, :, OUTCOME_HARM]
            utility = action_utility_logits
            safety_probability = torch.sigmoid(action_utility_logits)

        else:
            edit_logit = self.edit_head(feature)
            edit_probability = torch.sigmoid(
                edit_logit / self.edit_temperature
            )
            action_logits = self.action_head(feature)
            action_hard_st, action_conditional = _ste_one_hot(
                action_logits, self.action_temperature
            )
            action_route_soft = edit_probability * action_conditional
            preserve_route_soft = 1.0 - edit_probability
            route_soft_probs = torch.cat(
                [preserve_route_soft, action_route_soft], dim=1
            )

            raw_outcome = self.outcome_head(feature)
            b, _, h, w = raw_outcome.shape
            outcome_logits = raw_outcome.view(b, 4, 3, h, w)
            outcome_probs = F.softmax(
                outcome_logits / self.outcome_temperature,
                dim=2,
            )
            neutral_probability = outcome_probs[:, :, OUTCOME_NEUTRAL]
            benefit_probability = outcome_probs[:, :, OUTCOME_BENEFIT]
            harm_probability = outcome_probs[:, :, OUTCOME_HARM]

            utility = (
                benefit_probability
                - self.harm_penalty * harm_probability
                - self.edit_penalty * alphas_for_refiner
            )
            safety_probability = torch.sigmoid(
                (utility - self.utility_threshold) / self.risk_temperature
            )

            selected_safety_probability = (
                action_hard_st * safety_probability
            ).sum(dim=1, keepdim=True)
            execute_probability = (
                edit_probability * selected_safety_probability
            ).clamp(0.0, 1.0)
            if self.deployment_hard_route:
                execute_gate = _ste_binary(
                    execute_probability, threshold=self.execute_threshold
                )
                safe_action_weight = execute_gate * action_hard_st
            else:
                execute_gate = execute_probability
                safe_action_weight = action_route_soft * safety_probability

            preserve_route = 1.0 - execute_gate
            route_probs = torch.cat(
                [preserve_route, safe_action_weight], dim=1
            )
            policy_logits = torch.cat(
                [torch.zeros_like(edit_logit), action_logits], dim=1
            )
        # Exactly one action or exact Preserve in the hard-forward mode.
        effective_alpha = (
            safe_action_weight * alphas_for_refiner
        ).clamp(0.0, 1.0)
        negative_alpha = (
            effective_alpha[:, 0:1] + effective_alpha[:, 2:3]
        ).clamp(0.0, 1.0)
        positive_alpha = (
            effective_alpha[:, 1:2] + effective_alpha[:, 3:4]
        ).clamp(0.0, 1.0)
        final_prob = (
            c0 * (1.0 - negative_alpha)
            + (1.0 - c0) * positive_alpha
        )
        if (
            self.online_component_refinement
            and v538_refiner_output is not None
            and "selected_final_probability" in v538_refiner_output
        ):
            # V540 deployment must use the same signed continuous logit dose
            # that defines the differentiable slot candidate.  The legacy
            # action-weight path remains only for compatibility diagnostics.
            final_prob = v538_refiner_output[
                "selected_final_probability"
            ].detach().clamp(EPS, 1.0 - EPS)

        per_action_edit_mass = torch.cat(
            [
                c0 * safe_action_weight[:, 0:1] * alphas_for_refiner[:, 0:1],
                (1.0 - c0) * safe_action_weight[:, 1:2] * alphas_for_refiner[:, 1:2],
                c0 * safe_action_weight[:, 2:3] * alphas_for_refiner[:, 2:3],
                (1.0 - c0) * safe_action_weight[:, 3:4] * alphas_for_refiner[:, 3:4],
            ],
            dim=1,
        )
        predicted_neutral_mass = per_action_edit_mass * neutral_probability
        predicted_benefit_mass = per_action_edit_mass * benefit_probability
        predicted_harm_mass = per_action_edit_mass * harm_probability
        partition_error = (
            predicted_neutral_mass
            + predicted_benefit_mass
            + predicted_harm_mass
            - per_action_edit_mass
        ).abs().amax(dim=(1, 2, 3))

        selected_action = route_probs.argmax(dim=1)
        deployed_preserve_probability = preserve_route.clamp(0.0, 1.0)

        result = {
            "m2_edit_logit": edit_logit[:, 0],
            "m2_edit_probability": edit_probability[:, 0],
            "m2_action_logits": action_logits,
            "m2_action_conditional_probs": action_conditional,
            "m2_route_probs": route_probs,
            "m2_route_soft_probs": route_soft_probs,
            "m2_selected_action": selected_action,
            "m2_action_outcome_logits": outcome_logits,
            "m2_action_outcome_probs": outcome_probs,
            "m2_action_neutral_probs": neutral_probability,
            "m2_action_benefit_probs": benefit_probability,
            "m2_action_harm_probs": harm_probability,
            "m2_action_utility": utility,
            "m2_action_risk_gate": safety_probability,
            "m2_execute_probability": execute_probability[:, 0],
            "m2_execute_gate": execute_gate[:, 0],
            "m2_action_hard_st": action_hard_st,
            "m2_action_weight": safe_action_weight,
            "m2_effective_action_alpha": effective_alpha,
            "m2_negative_alpha": negative_alpha,
            "m2_positive_alpha": positive_alpha,
            "m2_proposal_probs": (
                c0 * (1.0 - (action_route_soft[:, 0:1] * alphas[:, 0:1]
                             + action_route_soft[:, 2:3] * alphas[:, 2:3]).clamp(0.0, 1.0))
                + (1.0 - c0) * (
                    action_route_soft[:, 1:2] * alphas[:, 1:2]
                    + action_route_soft[:, 3:4] * alphas[:, 3:4]
                ).clamp(0.0, 1.0)
            )[:, 0],
            "m2_convex_probs": final_prob[:, 0],
            "m2_training_probs": final_prob[:, 0],
            "m2_fused_probs": final_prob[:, 0],
            "m2_residual_map": (final_prob - c0)[:, 0],
            "m2_edit_gate_prob": safe_action_weight.sum(dim=1),
            "v535_policy_logits": policy_logits,
            "v535_policy_soft_probs": route_soft_probs,
            "v535_action_utility_logits": utility,
            "v536_case_policy_logits": (
                case_policy_logits if self.adaptive_utility_policy
                else policy_logits.mean(dim=(2, 3))
            ),
            "v536_case_policy_soft": (
                case_policy_soft if self.adaptive_utility_policy
                else route_soft_probs.mean(dim=(2, 3))
            ),
            "v536_case_policy_index": (
                case_policy_index if self.adaptive_utility_policy
                else route_probs.mean(dim=(2, 3)).argmax(dim=1)
            ),
            "v536_component_accept": (
                component_accept if self.adaptive_utility_policy
                else execute_gate.flatten(1).amax(dim=1)
            ),
            "v536_component_area_fraction": (
                component_area_fraction if self.adaptive_utility_policy
                else execute_gate.flatten(1).mean(dim=1)
            ),
            "v536_component_score": (
                component_score if self.adaptive_utility_policy
                else execute_gate.flatten(1).mean(dim=1)
            ),
            "v538_slot_mask_logits": (
                v538_refiner_output["slot_mask_logits"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:]))
            ),
            "v538_slot_masks": (
                v538_refiner_output["slot_masks"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:]))
            ),
            "v551_enabled": (
                v538_refiner_output.get("v551_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v551_fast_parent_atoms_active": (
                v538_refiner_output.get(
                    "v551_fast_parent_atoms_active", final_prob.new_zeros(())
                ) if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v551_editor_active": (
                v538_refiner_output.get(
                    "v551_editor_active", final_prob.new_zeros(())
                ) if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v551_gpu_atomizer_enabled": (
                v538_refiner_output.get(
                    "v551_gpu_atomizer_enabled", final_prob.new_zeros(())
                ) if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v551_single_pass_editor_enabled": (
                v538_refiner_output.get(
                    "v551_single_pass_editor_enabled", final_prob.new_zeros(())
                ) if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v551_parent_mask_logits": (
                v538_refiner_output.get("v551_parent_mask_logits", final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:])))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:]))
            ),
            "v551_parent_masks": (
                v538_refiner_output.get("v551_parent_masks", final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:])))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:]))
            ),
            "v552r47_rootfix_enabled": (
                v538_refiner_output.get("v552r47_rootfix_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r47_direct_slot_components_enabled": (
                v538_refiner_output.get("v552r47_direct_slot_components_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r47_anchor_params": (
                v538_refiner_output.get("v552r47_anchor_params", final_prob.new_zeros((final_prob.shape[0], 1, 4)))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0], 1, 4))
            ),
            "v552r47_query_mask_logits": (
                v538_refiner_output.get("v552r47_query_mask_logits", final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:])))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:]))
            ),
            "v552r48_rootfix_enabled": (
                v538_refiner_output.get("v552r48_rootfix_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r48_remove_coarse_mask_bias_enabled": (
                v538_refiner_output.get("v552r48_remove_coarse_mask_bias_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r48_stage_mask_logits": (
                v538_refiner_output.get("v552r48_stage_mask_logits", final_prob.new_zeros((final_prob.shape[0],1,1,*final_prob.shape[-2:])))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0],1,1,*final_prob.shape[-2:]))
            ),
            "v552r48_stage_anchor_params": (
                v538_refiner_output.get("v552r48_stage_anchor_params", final_prob.new_zeros((final_prob.shape[0],1,1,4)))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0],1,1,4))
            ),
            "v552r48_local_attention_entropy": (
                v538_refiner_output.get("v552r48_local_attention_entropy", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r49_rootfix_enabled": (
                v538_refiner_output.get("v552r49_rootfix_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r49_anchor_sampling_only_enabled": (
                v538_refiner_output.get("v552r49_anchor_sampling_only_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r49_dn_curriculum_enabled": (
                v538_refiner_output.get("v552r49_dn_curriculum_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r49_attention_max_weight": (
                v538_refiner_output.get("v552r49_attention_max_weight", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r49_dn_noise_scale": (
                v538_refiner_output.get("v552r49_dn_noise_scale", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r49_attention_logit_scale_mean": (
                v538_refiner_output.get("v552r49_attention_logit_scale_mean", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r410_rootfix_enabled": (
                v538_refiner_output.get("v552r410_rootfix_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r410_evidence_proposal_enabled": (
                v538_refiner_output.get("v552r410_evidence_proposal_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r410_support_only_local_readout_enabled": (
                v538_refiner_output.get("v552r410_support_only_local_readout_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r410_proposal_score_mean": (
                v538_refiner_output.get("v552r410_proposal_score_mean", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r410_proposal_valid_fraction": (
                v538_refiner_output.get("v552r410_proposal_valid_fraction", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r410_dn_clean_curriculum_enabled": (
                v538_refiner_output.get("v552r410_dn_clean_curriculum_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r411_rootfix_enabled": (
                v538_refiner_output.get("v552r411_rootfix_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r411_typed_proposal_enabled": (
                v538_refiner_output.get("v552r411_typed_proposal_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r411_local_roi_decoder_enabled": (
                v538_refiner_output.get("v552r411_local_roi_decoder_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r411_raw_native_mask_enabled": (
                v538_refiner_output.get("v552r411_raw_native_mask_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r412_rootfix_enabled": (
                v538_refiner_output.get("v552r412_rootfix_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r412_canonical_renderer_enabled": (
                v538_refiner_output.get("v552r412_canonical_renderer_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r412_action_support_enabled": (
                v538_refiner_output.get("v552r412_action_support_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r413_rootfix_enabled": (
                v538_refiner_output.get("v552r413_rootfix_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r413_geometry_locked_enabled": (
                v538_refiner_output.get("v552r413_geometry_locked_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r413_query_extent_enabled": (
                v538_refiner_output.get("v552r413_query_extent_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r414_rootfix_enabled": (
                v538_refiner_output.get("v552r414_rootfix_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r414_context_grid_size": (
                v538_refiner_output.get("v552r414_context_grid_size", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r414_context_radius": (
                v538_refiner_output.get("v552r414_context_radius", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r416_rootfix_enabled": (
                v538_refiner_output.get("v552r416_rootfix_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r416_unique_point_enabled": (
                v538_refiner_output.get("v552r416_unique_point_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r416_asymmetric_ltrb_enabled": (
                v538_refiner_output.get("v552r416_asymmetric_ltrb_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r416_cross_type_nms_radius_px": (
                v538_refiner_output.get("v552r416_cross_type_nms_radius_px", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r416_legacy_topk_unique_fraction": (
                v538_refiner_output.get("v552r416_legacy_topk_unique_fraction", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r416_proposal_point_xy": (
                v538_refiner_output.get("v552r416_proposal_point_xy", final_prob.new_zeros((final_prob.shape[0],1,2)))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0],1,2))
            ),
            "v552r416_edge_offsets": (
                v538_refiner_output.get("v552r416_edge_offsets", final_prob.new_zeros((final_prob.shape[0],1,4)))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0],1,4))
            ),
            "v552r417_rootfix_enabled": (
                v538_refiner_output.get("v552r417_rootfix_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r417_location_first_enabled": (
                v538_refiner_output.get("v552r417_location_first_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r417_shared_offset_enabled": (
                v538_refiner_output.get("v552r417_shared_offset_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r417_location_nms_kernel": (
                v538_refiner_output.get("v552r417_location_nms_kernel", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r417_location_dedup_radius_px": (
                v538_refiner_output.get("v552r417_location_dedup_radius_px", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r417_location_logits": (
                v538_refiner_output.get("v552r417_location_logits", final_prob.new_zeros((final_prob.shape[0],1,*final_prob.shape[-2:])))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0],1,*final_prob.shape[-2:]))
            ),
            "v552r417_location_offset_map": (
                v538_refiner_output.get("v552r417_location_offset_map", final_prob.new_zeros((final_prob.shape[0],2,*final_prob.shape[-2:])))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0],2,*final_prob.shape[-2:]))
            ),
            "v552r417_proposal_point_xy": (
                v538_refiner_output.get("v552r417_proposal_point_xy", final_prob.new_zeros((final_prob.shape[0],1,2)))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0],1,2))
            ),
            "v552r417_parent_proposal_point_xy": (
                v538_refiner_output.get("v552r417_parent_proposal_point_xy", final_prob.new_zeros((final_prob.shape[0],1,2)))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0],1,2))
            ),
            "v552r418_rootfix_enabled": (
                v538_refiner_output.get("v552r418_rootfix_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r418_box_free_mask_set_enabled": (
                v538_refiner_output.get("v552r418_box_free_mask_set_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r418_paired_stable_teacher_enabled": (
                v538_refiner_output.get("v552r418_paired_stable_teacher_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r418_paired_mask_logits": (
                v538_refiner_output.get("v552r418_paired_mask_logits", final_prob.new_zeros((final_prob.shape[0],1,*final_prob.shape[-2:])))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0],1,*final_prob.shape[-2:]))
            ),
            "v552r418_paired_teacher_masks": (
                v538_refiner_output.get("v552r418_paired_teacher_masks", final_prob.new_zeros((final_prob.shape[0],1,*final_prob.shape[-2:])))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0],1,*final_prob.shape[-2:]))
            ),
            "v552r418_paired_teacher_valid": (
                v538_refiner_output.get("v552r418_paired_teacher_valid", torch.zeros((final_prob.shape[0],1), device=final_prob.device, dtype=torch.bool))
                if self.online_component_refinement else torch.zeros((final_prob.shape[0],1), device=final_prob.device, dtype=torch.bool)
            ),
            "v552r418_paired_coarse": (
                v538_refiner_output.get("v552r418_paired_coarse", final_prob.detach())
                if self.online_component_refinement else final_prob.detach()
            ),
            "v552r419_rootfix_enabled": (
                v538_refiner_output.get("v552r419_rootfix_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r419_seeded_masked_attention_enabled": (
                v538_refiner_output.get("v552r419_seeded_masked_attention_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r419_seed_radius_px": (
                v538_refiner_output.get("v552r419_seed_radius_px", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r419_support_dilate_kernel": (
                v538_refiner_output.get("v552r419_support_dilate_kernel", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r419_seed_support_fraction": (
                v538_refiner_output.get("v552r419_seed_support_fraction", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r419_final_support_fraction": (
                v538_refiner_output.get("v552r419_final_support_fraction", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r419_outside_mask_probability": (
                v538_refiner_output.get("v552r419_outside_mask_probability", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r420_rootfix_enabled": (
                v538_refiner_output.get("v552r420_rootfix_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r420_dynamic_mask_enabled": (
                v538_refiner_output.get("v552r420_dynamic_mask_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r420_type_decoupled_mask_enabled": (
                v538_refiner_output.get("v552r420_type_decoupled_mask_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r420_dynamic_channels": (
                v538_refiner_output.get("v552r420_dynamic_channels", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r420_relative_coord_mean_abs": (
                v538_refiner_output.get("v552r420_relative_coord_mean_abs", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4201_rootfix_enabled": (
                v538_refiner_output.get("v552r4201_clean_rootfix_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4201_forward_teacher_built": (
                v538_refiner_output.get("v552r48_teacher_built", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4201_teacher_valid_count": (
                v538_refiner_output.get("v552r48_teacher_valid_count", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4201_teacher_error_fraction": (
                v538_refiner_output.get(
                    "v552r48_teacher_error", final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:]))
                ).to(final_prob.dtype).mean()
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4203_rootfix_enabled": (
                v538_refiner_output.get("v552r4203_rootfix_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4203_dense_competitive_set_enabled": (
                v538_refiner_output.get("v552r4203_dense_competitive_set_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4203_ownership_sum_error": (
                v538_refiner_output.get("v552r4203_ownership_sum_error", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4203_assignment_entropy": (
                v538_refiner_output.get("v552r4203_assignment_entropy", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4203_background_fraction": (
                v538_refiner_output.get("v552r4203_background_fraction", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4203_max_slot_ownership": (
                v538_refiner_output.get("v552r4203_max_slot_ownership", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4203_slot_mass_cv": (
                v538_refiner_output.get("v552r4203_slot_mass_cv", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4203_point_bottleneck_used": (
                v538_refiner_output.get("v552r4203_point_bottleneck_used", final_prob.new_ones(()))
                if self.online_component_refinement else final_prob.new_ones(())
            ),
            "v552r4204_rootfix_enabled": (
                v538_refiner_output.get("v552r4204_rootfix_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4204_occupancy_logits": (
                v538_refiner_output.get(
                    "v552r4204_occupancy_logits",
                    final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:])),
                )
                if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:]))
            ),
            "v552r4204_residual_mass_conservation_error": (
                v538_refiner_output.get("v552r4204_residual_mass_conservation_error", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4204_conditional_slot_entropy": (
                v538_refiner_output.get("v552r4204_conditional_slot_entropy", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4204_conditional_max_slot_probability": (
                v538_refiner_output.get("v552r4204_conditional_max_slot_probability", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4204_residual_existence_mean": (
                v538_refiner_output.get("v552r4204_residual_existence_mean", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4204_centroid_separation": (
                v538_refiner_output.get("v552r4204_centroid_separation", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4204_spatial_variance_mean": (
                v538_refiner_output.get("v552r4204_spatial_variance_mean", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4204_spatial_identity_enabled": (
                v538_refiner_output.get("v552r4204_spatial_identity_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4204_location_as_occupancy_used": (
                v538_refiner_output.get("v552r4204_location_as_occupancy_used", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4205_rootfix_enabled": (
                v538_refiner_output.get("v552r4205_rootfix_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4205_overflow_probability": (
                v538_refiner_output.get(
                    "v552r4205_overflow_probability",
                    final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:])),
                )
                if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:]))
            ),
            "v552r4205_overflow_identity_probability": (
                v538_refiner_output.get(
                    "v552r4205_overflow_identity_probability",
                    final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:])),
                )
                if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:]))
            ),
            "v552r4205_conditional_identity_logits": (
                v538_refiner_output.get(
                    "v552r4205_conditional_identity_logits",
                    final_prob.new_zeros((final_prob.shape[0], self.component_slot_generator.num_slots + 1, *final_prob.shape[-2:])),
                )
                if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], self.component_slot_generator.num_slots + 1, *final_prob.shape[-2:]))
            ),
            "v552r4205_editable_probability_sum": (
                v538_refiner_output.get(
                    "v552r4205_editable_probability_sum",
                    final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:])),
                )
                if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:]))
            ),
            "v552r4205_overflow_probability_mean": (
                v538_refiner_output.get("v552r4205_overflow_probability_mean", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4205_overflow_conditional_mean": (
                v538_refiner_output.get("v552r4205_overflow_conditional_mean", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4205_editable_probability_mean": (
                v538_refiner_output.get("v552r4205_editable_probability_mean", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4205_final_logits_finite_fraction": (
                v538_refiner_output.get("v552r4205_final_logits_finite_fraction", final_prob.new_ones(()))
                if self.online_component_refinement else final_prob.new_ones(())
            ),
            "v552r4207_rootfix_enabled": (
                v538_refiner_output.get("v552r4207_rootfix_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4207_seed_center_xy": (
                v538_refiner_output.get("v552r4207_seed_center_xy", final_prob.new_zeros((final_prob.shape[0], self.component_slot_generator.num_slots, 2)))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0], 1, 2))
            ),
            "v552r4207_seed_score": (
                v538_refiner_output.get("v552r4207_seed_score", final_prob.new_zeros((final_prob.shape[0], self.component_slot_generator.num_slots)))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0], 1))
            ),
            "v552r4207_seed_valid": (
                v538_refiner_output.get("v552r4207_seed_valid", final_prob.new_zeros((final_prob.shape[0], self.component_slot_generator.num_slots)))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0], 1))
            ),
            "v552r4207_seed_feature_finite_fraction": (
                v538_refiner_output.get("v552r4207_seed_feature_finite_fraction", final_prob.new_ones(()))
                if self.online_component_refinement else final_prob.new_ones(())
            ),
            "v552r4207_seed_valid_fraction": (
                v538_refiner_output.get("v552r4207_seed_valid_fraction", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4207_seed_score_mean": (
                v538_refiner_output.get("v552r4207_seed_score_mean", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4207_seed_pairwise_distance_px": (
                v538_refiner_output.get("v552r4207_seed_pairwise_distance_px", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4207_q0_pairwise_cosine": (
                v538_refiner_output.get("v552r4207_q0_pairwise_cosine", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4207_q1_pairwise_cosine": (
                v538_refiner_output.get("v552r4207_q1_pairwise_cosine", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4207_seed_to_slot_centroid_drift_px": (
                v538_refiner_output.get("v552r4207_seed_to_slot_centroid_drift_px", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4207_full_image_assignment_enabled": (
                v538_refiner_output.get("v552r4207_full_image_assignment_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4207_hard_spatial_support_used": (
                v538_refiner_output.get("v552r4207_hard_spatial_support_used", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4208_rootfix_enabled": (
                v538_refiner_output.get("v552r4208_rootfix_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4208_normalized_fusion_enabled": (
                v538_refiner_output.get("v552r4208_normalized_fusion_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4208_persistent_identity_enabled": (
                v538_refiner_output.get("v552r4208_persistent_identity_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4208_learned_query_norm": (
                v538_refiner_output.get("v552r4208_learned_query_norm", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4208_seed_feature_norm": (
                v538_refiner_output.get("v552r4208_seed_feature_norm", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4208_seed_to_learned_norm_ratio": (
                v538_refiner_output.get("v552r4208_seed_to_learned_norm_ratio", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4208_seed_feature_pairwise_cosine": (
                v538_refiner_output.get("v552r4208_seed_feature_pairwise_cosine", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4208_q0_seed_identity_cosine": (
                v538_refiner_output.get("v552r4208_q0_seed_identity_cosine", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4208_q1_seed_identity_cosine": (
                v538_refiner_output.get("v552r4208_q1_seed_identity_cosine", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4208_identity_retention_delta": (
                v538_refiner_output.get("v552r4208_identity_retention_delta", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r413_proposal_anchor_params": (
                v538_refiner_output.get("v552r413_proposal_anchor_params", final_prob.new_zeros((final_prob.shape[0], 1, 4)))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0], 1, 4))
            ),
            "v552r413_parent_proposal_anchor_params": (
                v538_refiner_output.get("v552r413_parent_proposal_anchor_params", final_prob.new_zeros((final_prob.shape[0], 1, 4)))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0], 1, 4))
            ),
            "v552r411_proposal_type": (
                v538_refiner_output.get("v552r411_proposal_type", torch.zeros((final_prob.shape[0],1), device=final_prob.device, dtype=torch.long))
                if self.online_component_refinement else torch.zeros((final_prob.shape[0],1), device=final_prob.device, dtype=torch.long)
            ),
            "v552r411_proposal_score": (
                v538_refiner_output.get("v552r411_proposal_score", final_prob.new_zeros((final_prob.shape[0],1)))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0],1))
            ),
            "v552r411_proposal_valid": (
                v538_refiner_output.get("v552r411_proposal_valid", final_prob.new_zeros((final_prob.shape[0],1)))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0],1))
            ),
            "v552r411_center_logits": (
                v538_refiner_output.get("v552r411_center_logits", final_prob.new_zeros((final_prob.shape[0],4,*final_prob.shape[-2:])))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0],4,*final_prob.shape[-2:]))
            ),
            "v552r411_size_map": (
                v538_refiner_output.get("v552r411_size_map", final_prob.new_zeros((final_prob.shape[0],4,2,*final_prob.shape[-2:])))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0],4,2,*final_prob.shape[-2:]))
            ),
            "v552r411_offset_map": (
                v538_refiner_output.get("v552r411_offset_map", final_prob.new_zeros((final_prob.shape[0],4,2,*final_prob.shape[-2:])))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0],4,2,*final_prob.shape[-2:]))
            ),
            "v552r48_dn_enabled": (
                v538_refiner_output.get("v552r48_dn_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r48_dn_stage_mask_logits": (
                v538_refiner_output.get("v552r48_dn_stage_mask_logits", final_prob.new_zeros((final_prob.shape[0],1,1,*final_prob.shape[-2:])))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0],1,1,*final_prob.shape[-2:]))
            ),
            "v552r48_dn_stage_anchor_params": (
                v538_refiner_output.get("v552r48_dn_stage_anchor_params", final_prob.new_zeros((final_prob.shape[0],1,1,4)))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0],1,1,4))
            ),
            "v552r48_dn_target_index": (
                v538_refiner_output.get("v552r48_dn_target_index", torch.zeros((final_prob.shape[0],1), device=final_prob.device, dtype=torch.long))
                if self.online_component_refinement else torch.zeros((final_prob.shape[0],1), device=final_prob.device, dtype=torch.long)
            ),
            "v552r48_dn_valid": (
                v538_refiner_output.get("v552r48_dn_valid", torch.zeros((final_prob.shape[0],1), device=final_prob.device, dtype=torch.bool))
                if self.online_component_refinement else torch.zeros((final_prob.shape[0],1), device=final_prob.device, dtype=torch.bool)
            ),
            "v552r48_teacher_masks": (v538_refiner_output.get("v552r48_teacher_masks", final_prob.new_zeros((final_prob.shape[0],1,*final_prob.shape[-2:]))) if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0],1,*final_prob.shape[-2:]))),
            "v552r48_teacher_actions": (v538_refiner_output.get("v552r48_teacher_actions", torch.zeros((final_prob.shape[0],1), device=final_prob.device, dtype=torch.long)) if self.online_component_refinement else torch.zeros((final_prob.shape[0],1), device=final_prob.device, dtype=torch.long)),
            "v552r48_teacher_valid": (v538_refiner_output.get("v552r48_teacher_valid", torch.zeros((final_prob.shape[0],1), device=final_prob.device, dtype=torch.bool)) if self.online_component_refinement else torch.zeros((final_prob.shape[0],1), device=final_prob.device, dtype=torch.bool)),
            "v552r48_teacher_area": (v538_refiner_output.get("v552r48_teacher_area", final_prob.new_zeros((final_prob.shape[0],1))) if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0],1))),
            "v552r48_teacher_geometry": (v538_refiner_output.get("v552r48_teacher_geometry", final_prob.new_zeros((final_prob.shape[0],1,4))) if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0],1,4))),
            "v552r48_teacher_effective": (v538_refiner_output.get("v552r48_teacher_effective", final_prob.detach()) if self.online_component_refinement else final_prob.detach()),
            "v552r48_teacher_error": (v538_refiner_output.get("v552r48_teacher_error", final_prob.new_zeros(final_prob.shape)) if self.online_component_refinement else final_prob.new_zeros(final_prob.shape)),
            "v551_atom_parent_index": (
                v538_refiner_output.get("v551_atom_parent_index", torch.zeros((final_prob.shape[0], 1), device=final_prob.device, dtype=torch.long))
                if self.online_component_refinement else torch.zeros((final_prob.shape[0], 1), device=final_prob.device, dtype=torch.long)
            ),
            "v551_atom_valid": (
                v538_refiner_output.get("v551_atom_valid", torch.zeros((final_prob.shape[0], 1), device=final_prob.device, dtype=torch.bool))
                if self.online_component_refinement else torch.zeros((final_prob.shape[0], 1), device=final_prob.device, dtype=torch.bool)
            ),
            "v552_physical_valid": (
                v538_refiner_output.get("v552_physical_valid", torch.zeros((final_prob.shape[0], 1), device=final_prob.device, dtype=torch.bool))
                if self.online_component_refinement else torch.zeros((final_prob.shape[0], 1), device=final_prob.device, dtype=torch.bool)
            ),
            "v552_m1_supervision_valid": (
                v538_refiner_output.get("v552_m1_supervision_valid", torch.zeros((final_prob.shape[0], 1), device=final_prob.device, dtype=torch.bool))
                if self.online_component_refinement else torch.zeros((final_prob.shape[0], 1), device=final_prob.device, dtype=torch.bool)
            ),
            "v552_composer_teacher_valid": (
                v538_refiner_output.get("v552_composer_teacher_valid", torch.zeros((final_prob.shape[0], 1), device=final_prob.device, dtype=torch.bool))
                if self.online_component_refinement else torch.zeros((final_prob.shape[0], 1), device=final_prob.device, dtype=torch.bool)
            ),
            "v552_deployment_valid": (
                v538_refiner_output.get("v552_deployment_valid", torch.zeros((final_prob.shape[0], 1), device=final_prob.device, dtype=torch.bool))
                if self.online_component_refinement else torch.zeros((final_prob.shape[0], 1), device=final_prob.device, dtype=torch.bool)
            ),
            "v552_atom_quality_logits": (
                v538_refiner_output.get("v552_atom_quality_logits", final_prob.new_zeros((final_prob.shape[0], 1)))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0], 1))
            ),
            "v552_atom_quality_probs": (
                v538_refiner_output.get("v552_atom_quality_probs", final_prob.new_zeros((final_prob.shape[0], 1)))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0], 1))
            ),
            "v552_editor_relative_gain_pred": (
                v538_refiner_output.get("v552_editor_relative_gain_pred", final_prob.new_zeros((final_prob.shape[0], 1)))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0], 1))
            ),
            "v552r2_unified_outcome_enabled": (
                v538_refiner_output.get("v552r2_unified_outcome_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r2_benefit_magnitude": (
                v538_refiner_output.get("v552r2_benefit_magnitude", final_prob.new_zeros((final_prob.shape[0], 1)))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0], 1))
            ),
            "v552r2_harm_magnitude": (
                v538_refiner_output.get("v552r2_harm_magnitude", final_prob.new_zeros((final_prob.shape[0], 1)))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0], 1))
            ),
            "v552r2_benefit_contribution": (
                v538_refiner_output.get("v552r2_benefit_contribution", final_prob.new_zeros((final_prob.shape[0], 1)))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0], 1))
            ),
            "v552r2_harm_contribution": (
                v538_refiner_output.get("v552r2_harm_contribution", final_prob.new_zeros((final_prob.shape[0], 1)))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0], 1))
            ),
            "v552_composer_teacher_pool_count": (
                v538_refiner_output.get("v552_composer_teacher_pool_count", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552_composer_deploy_pool_count": (
                v538_refiner_output.get("v552_composer_deploy_pool_count", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552_shadow_evidence_enabled": (
                v538_refiner_output.get("v552_shadow_evidence_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552_deploy_slot_valid": (
                v538_refiner_output.get("v552_deploy_slot_valid", torch.zeros((final_prob.shape[0], 1), device=final_prob.device, dtype=torch.bool))
                if self.online_component_refinement else torch.zeros((final_prob.shape[0], 1), device=final_prob.device, dtype=torch.bool)
            ),
            "v552_safe_calibrated_execution_enabled": (
                v538_refiner_output.get("v552_safe_calibrated_execution_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552_atom_quality_gate_active": (
                v538_refiner_output.get("v552_atom_quality_gate_active", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552_atom_quality_score": (
                v538_refiner_output.get("v552_atom_quality_score", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552_editor_strength": (
                v538_refiner_output.get("v552_editor_strength", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552_multicandidate_composer_enabled": (
                v538_refiner_output.get("v552_multicandidate_composer_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552_composer_support": (
                v538_refiner_output.get("v552_composer_support", final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:])))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:]))
            ),
            "v552_composer_step_logits": (
                v538_refiner_output.get("v552_composer_step_logits", final_prob.new_zeros((final_prob.shape[0], 1, 2)))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0], 1, 2))
            ),
            "v552_composer_step_candidate_scores": (
                v538_refiner_output.get("v552_composer_step_candidate_scores", final_prob.new_zeros((final_prob.shape[0], 1, 1)))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0], 1, 1))
            ),
            "v552_composer_step_state_probs": (
                v538_refiner_output.get("v552_composer_step_state_probs", final_prob[:, None])
                if self.online_component_refinement else final_prob[:, None]
            ),
            "v552_composer_step_eligible": (
                v538_refiner_output.get("v552_composer_step_eligible", torch.zeros((final_prob.shape[0], 1, 1), device=final_prob.device, dtype=torch.bool))
                if self.online_component_refinement else torch.zeros((final_prob.shape[0], 1, 1), device=final_prob.device, dtype=torch.bool)
            ),
            "v552_composer_step_active": (
                v538_refiner_output.get("v552_composer_step_active", torch.zeros((final_prob.shape[0], 1), device=final_prob.device, dtype=torch.bool))
                if self.online_component_refinement else torch.zeros((final_prob.shape[0], 1), device=final_prob.device, dtype=torch.bool)
            ),
            "v552_composer_step_selected_index": (
                v538_refiner_output.get("v552_composer_step_selected_index", torch.ones((final_prob.shape[0], 1), device=final_prob.device, dtype=torch.long))
                if self.online_component_refinement else torch.ones((final_prob.shape[0], 1), device=final_prob.device, dtype=torch.long)
            ),
            "v552_composer_predicted_step_count": (
                v538_refiner_output.get("v552_composer_predicted_step_count", final_prob.new_zeros((final_prob.shape[0],)))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0],))
            ),
            "v552_composer_ungated_step_count": (
                v538_refiner_output.get("v552_composer_ungated_step_count", final_prob.new_zeros((final_prob.shape[0],)))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0],))
            ),
            "v552_composer_ungated_final_probability": (
                v538_refiner_output.get("v552_composer_ungated_final_probability", final_prob)
                if self.online_component_refinement else final_prob
            ),
            "v552_composer_ungated_selected_index": (
                v538_refiner_output.get(
                    "v552_composer_ungated_selected_index",
                    torch.ones((final_prob.shape[0], 1), device=final_prob.device, dtype=torch.long),
                ) if self.online_component_refinement
                else torch.ones((final_prob.shape[0], 1), device=final_prob.device, dtype=torch.long)
            ),
            "v552_composer_teacher_forced_active": (
                v538_refiner_output.get("v552_composer_teacher_forced_active", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552_composer_forced_target_indices": (
                v538_refiner_output.get(
                    "v552_composer_forced_target_indices",
                    torch.ones((final_prob.shape[0], 1), device=final_prob.device, dtype=torch.long),
                ) if self.online_component_refinement
                else torch.ones((final_prob.shape[0], 1), device=final_prob.device, dtype=torch.long)
            ),
            "v552r41_teacher_target_indices": (
                v538_refiner_output.get(
                    "v552r41_teacher_target_indices",
                    torch.ones((final_prob.shape[0], 1), device=final_prob.device, dtype=torch.long),
                ) if self.online_component_refinement
                else torch.ones((final_prob.shape[0], 1), device=final_prob.device, dtype=torch.long)
            ),
            "v552r41_teacher_target_marginal_gains": (
                v538_refiner_output.get(
                    "v552r41_teacher_target_marginal_gains",
                    final_prob.new_zeros((final_prob.shape[0], 1, 1)),
                ) if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1, 1))
            ),
            "v552r41_teacher_target_eligible": (
                v538_refiner_output.get(
                    "v552r41_teacher_target_eligible",
                    torch.zeros((final_prob.shape[0], 1, 1), device=final_prob.device, dtype=torch.bool),
                ) if self.online_component_refinement
                else torch.zeros((final_prob.shape[0], 1, 1), device=final_prob.device, dtype=torch.bool)
            ),
            "v552r41_teacher_target_active": (
                v538_refiner_output.get(
                    "v552r41_teacher_target_active",
                    torch.zeros((final_prob.shape[0], 1), device=final_prob.device, dtype=torch.bool),
                ) if self.online_component_refinement
                else torch.zeros((final_prob.shape[0], 1), device=final_prob.device, dtype=torch.bool)
            ),
            "v552r41_teacher_target_step_gains": (
                v538_refiner_output.get(
                    "v552r41_teacher_target_step_gains",
                    final_prob.new_zeros((final_prob.shape[0], 1)),
                ) if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1))
            ),
            "v552r41_teacher_state_probs": (
                v538_refiner_output.get(
                    "v552r41_teacher_state_probs",
                    final_prob[:, None],
                ) if self.online_component_refinement
                else final_prob[:, None]
            ),
            "v552r41_teacher_final_probs": (
                v538_refiner_output.get(
                    "v552r41_teacher_final_probs", final_prob
                ) if self.online_component_refinement
                else final_prob
            ),
            "v552r41_single_teacher_contract_enabled": (
                v538_refiner_output.get(
                    "v552r41_single_teacher_contract_enabled", final_prob.new_zeros(())
                ) if self.online_component_refinement
                else final_prob.new_zeros(())
            ),
            "v552_composer_conflict_reject_rate": (
                v538_refiner_output.get("v552_composer_conflict_reject_rate", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552_composer_budget_reject_rate": (
                v538_refiner_output.get("v552_composer_budget_reject_rate", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v551_atom_dense_capacity": (
                v538_refiner_output.get("v551_atom_dense_capacity", final_prob.new_tensor(1.0))
                if self.online_component_refinement else final_prob.new_tensor(1.0)
            ),
            "v551_atom_valid_count": (
                v538_refiner_output.get("v551_atom_valid_count", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v551_scale_logits": (
                v538_refiner_output.get("v551_scale_logits", final_prob.new_zeros((final_prob.shape[0], 1, 1)))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0], 1, 1))
            ),
            "v551_scale_probs": (
                v538_refiner_output.get("v551_scale_probs", final_prob.new_ones((final_prob.shape[0], 1, 1)))
                if self.online_component_refinement else final_prob.new_ones((final_prob.shape[0], 1, 1))
            ),
            "v551_scale_index": (
                v538_refiner_output.get("v551_scale_index", torch.zeros((final_prob.shape[0], 1), device=final_prob.device, dtype=torch.long))
                if self.online_component_refinement else torch.zeros((final_prob.shape[0], 1), device=final_prob.device, dtype=torch.long)
            ),
            "v551_boundary_region": (
                v538_refiner_output.get("v551_boundary_region", final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:])))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:]))
            ),
            "v551_boundary_residual": (
                v538_refiner_output.get("v551_boundary_residual", final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:])))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:]))
            ),
            "v551_m1_logit_delta": (
                v538_refiner_output.get("v551_m1_logit_delta", final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:])))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:]))
            ),
            "v551_m1_exact_candidate_st": (
                v538_refiner_output.get("v551_m1_exact_candidate_st", final_prob[:, 0][:, None])
                if self.online_component_refinement else final_prob[:, 0][:, None]
            ),
            "v552r4209_m1_native_probability": (
                v538_refiner_output.get("v552r4209_m1_native_probability", final_prob)
                if self.online_component_refinement else final_prob
            ),
            "v552r4209_m1_native_hard_probability": (
                v538_refiner_output.get("v552r4209_m1_native_hard_probability", final_prob)
                if self.online_component_refinement else final_prob
            ),
            "v552r4209_m1_native_valid_fraction": (
                v538_refiner_output.get("v552r4209_m1_native_valid_fraction", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4209_m1_native_presence_mean": (
                v538_refiner_output.get("v552r4209_m1_native_presence_mean", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4210_rootfix_enabled": (
                v538_refiner_output.get("v552r4210_rootfix_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4210_variable_cardinality_seed_enabled": (
                v538_refiner_output.get("v552r4210_variable_cardinality_seed_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4210_independent_overflow_enabled": (
                v538_refiner_output.get("v552r4210_independent_overflow_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4210_valid_seed_count": (
                v538_refiner_output.get("v552r4210_valid_seed_count", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4210_seed_logit": (
                v538_refiner_output.get("v552r4210_seed_logit", final_prob.new_zeros((final_prob.shape[0], self.component_slot_generator.num_slots)))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0], 1))
            ),
            "v552r4210_seed_logit_mean": (
                v538_refiner_output.get("v552r4210_seed_logit_mean", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4210_overflow_gate_logits": (
                v538_refiner_output.get("v552r4210_overflow_gate_logits", final_prob.new_zeros(final_prob.shape))
                if self.online_component_refinement else final_prob.new_zeros(final_prob.shape)
            ),
            "v552r4210_overflow_conditional_probability": (
                v538_refiner_output.get("v552r4210_overflow_conditional_probability", final_prob.new_zeros(final_prob.shape))
                if self.online_component_refinement else final_prob.new_zeros(final_prob.shape)
            ),
            "v552r4210_overflow_logit_mean": (
                v538_refiner_output.get("v552r4210_overflow_logit_mean", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4210_overflow_conditional_mean": (
                v538_refiner_output.get("v552r4210_overflow_conditional_mean", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4210_m1_native_probability": (
                v538_refiner_output.get("v552r4210_m1_native_probability", v538_refiner_output.get("v552r4209_m1_native_probability", final_prob))
                if self.online_component_refinement else final_prob
            ),
            "v552r4210_m1_native_hard_probability": (
                v538_refiner_output.get("v552r4210_m1_native_hard_probability", v538_refiner_output.get("v552r4209_m1_native_hard_probability", final_prob))
                if self.online_component_refinement else final_prob
            ),
            "v552r4210_m1_native_train_probability": (
                v538_refiner_output.get("v552r4210_m1_native_train_probability", final_prob)
                if self.online_component_refinement else final_prob
            ),
            "v552r4210_m1_native_alignment_active": (
                v538_refiner_output.get("v552r4210_m1_native_alignment_active", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4211_rootfix_enabled": (
                v538_refiner_output.get("v552r4211_rootfix_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4211_proposal_existence_decoupling_enabled": (
                v538_refiner_output.get("v552r4211_proposal_existence_decoupling_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4211_geometry_overflow_decoupling_enabled": (
                v538_refiner_output.get("v552r4211_geometry_overflow_decoupling_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4211_proposal_seed_count": (
                v538_refiner_output.get("v552r4211_proposal_seed_count", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4211_proposal_confidence_mean": (
                v538_refiner_output.get("v552r4211_proposal_confidence_mean", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4211_presence_expected_count": (
                v538_refiner_output.get("v552r4211_presence_expected_count", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4211_presence_hard_count": (
                v538_refiner_output.get("v552r4211_presence_hard_count", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4211_geometry_effective_l1": (
                v538_refiner_output.get("v552r4211_geometry_effective_l1", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4211_effective_slot_masks": (
                v538_refiner_output.get("v552r4211_effective_slot_masks", final_prob.new_zeros((final_prob.shape[0], self.component_slot_generator.num_slots, *final_prob.shape[-2:])))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:]))
            ),
            "v552r4211_m1_native_probability": (
                v538_refiner_output.get("v552r4211_m1_native_probability", v538_refiner_output.get("v552r4210_m1_native_probability", final_prob))
                if self.online_component_refinement else final_prob
            ),
            "v552r4211_m1_native_hard_probability": (
                v538_refiner_output.get("v552r4211_m1_native_hard_probability", v538_refiner_output.get("v552r4210_m1_native_hard_probability", final_prob))
                if self.online_component_refinement else final_prob
            ),
            "v552r4211_m1_native_train_probability": (
                v538_refiner_output.get("v552r4211_m1_native_train_probability", v538_refiner_output.get("v552r4210_m1_native_train_probability", final_prob))
                if self.online_component_refinement else final_prob
            ),
            "v552r4212_rootfix_enabled": (
                v538_refiner_output.get("v552r4212_rootfix_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4212_independent_candidate_set_enabled": (
                v538_refiner_output.get("v552r4212_independent_candidate_set_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4212_visual_seed_identity_disabled": (
                v538_refiner_output.get("v552r4212_visual_seed_identity_disabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4212_existence_no_object_enabled": (
                v538_refiner_output.get("v552r4212_existence_no_object_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4212_candidate_alignment_enabled": (
                v538_refiner_output.get("v552r4212_candidate_alignment_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4212_direct_delta_utility_enabled": (
                v538_refiner_output.get("v552r4212_direct_delta_utility_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4212_zero_stop_one_step_enabled": (
                v538_refiner_output.get("v552r4212_zero_stop_one_step_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4212_parent_existence_count": (
                v538_refiner_output.get("v552r4212_parent_existence_count", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4212_deployment_candidate_count": (
                v538_refiner_output.get("v552r4212_deployment_candidate_count", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4212_independent_soft_overlap_mass": (
                v538_refiner_output.get("v552r4212_independent_soft_overlap_mass", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v552r4212_m1_native_probability": (
                v538_refiner_output.get("v552r4211_m1_native_probability", v538_refiner_output.get("v552r4210_m1_native_probability", final_prob))
                if self.online_component_refinement else final_prob
            ),
            "v552r4212_m1_native_hard_probability": (
                v538_refiner_output.get("v552r4211_m1_native_hard_probability", v538_refiner_output.get("v552r4210_m1_native_hard_probability", final_prob))
                if self.online_component_refinement else final_prob
            ),
            "v560_clean_core_enabled": (
                v538_refiner_output.get("v560_clean_core_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v560_direct_mask_probability_mean": (
                v538_refiner_output.get("v560_direct_mask_probability_mean", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v560_independent_soft_overlap_mass": (
                v538_refiner_output.get("v560_independent_soft_overlap_mass", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v560_direct_mask_overlap_mass": (
                v538_refiner_output.get(
                    "v560_independent_soft_overlap_mass",
                    v538_refiner_output.get("v560_direct_mask_overlap_mass", final_prob.new_zeros(())),
                )
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v560_q0_pairwise_cosine": (
                v538_refiner_output.get("v560_q0_pairwise_cosine", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v560_q1_pairwise_cosine": (
                v538_refiner_output.get("v560_q1_pairwise_cosine", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v560_mask_bias_mean": (
                v538_refiner_output.get("v560_mask_bias_mean", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v561_bcrs_enabled": (
                v538_refiner_output.get("v561_bcrs_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v561_geometry_owner": (
                v538_refiner_output.get("v561_geometry_owner", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v561_variant_static": (
                v538_refiner_output.get("v561_variant_static", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v561_variant_image": (
                v538_refiner_output.get("v561_variant_image", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v561_variant_typed": (
                v538_refiner_output.get("v561_variant_typed", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v561_mask_probability_mean": (
                v538_refiner_output.get("v561_mask_probability_mean", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v561_soft_overlap_mass": (
                v538_refiner_output.get("v561_soft_overlap_mass", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v561_q0_pairwise_cosine": (
                v538_refiner_output.get("v561_q0_pairwise_cosine", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v561_q1_pairwise_cosine": (
                v538_refiner_output.get("v561_q1_pairwise_cosine", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v561_q2_pairwise_cosine": (
                v538_refiner_output.get("v561_q2_pairwise_cosine", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v561_stage1_query_delta_norm": (
                v538_refiner_output.get("v561_stage1_query_delta_norm", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v561_stage2_query_delta_norm": (
                v538_refiner_output.get("v561_stage2_query_delta_norm", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v561_typed_support_mean": (
                v538_refiner_output.get("v561_typed_support_mean", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v561_typed_support_std": (
                v538_refiner_output.get("v561_typed_support_std", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v561_typed_support_neutrality_error": (
                v538_refiner_output.get("v561_typed_support_neutrality_error", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v561_stage1_attention_entropy_ratio": (
                v538_refiner_output.get("v561_stage1_attention_entropy_ratio", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v561_stage2_attention_entropy_ratio": (
                v538_refiner_output.get("v561_stage2_attention_entropy_ratio", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v561_mask_bias_mean": (
                v538_refiner_output.get("v561_mask_bias_mean", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v562_rootfix_enabled": (
                v538_refiner_output.get("v562_rootfix_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v562_residual_logits": (
                v538_refiner_output.get("v562_residual_logits", final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:])))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:]))
            ),
            "v562_residual_probability_mean": (
                v538_refiner_output.get("v562_residual_probability_mean", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v562_proposal_anchor_xy": (
                v538_refiner_output.get("v562_proposal_anchor_xy", final_prob.new_zeros((final_prob.shape[0], 1, 2)))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0], 1, 2))
            ),
            "v562_query_owned_presence": (
                v538_refiner_output.get("v562_query_owned_presence", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v562_direct_binary_executor": (
                v538_refiner_output.get("v562_direct_binary_executor", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v563_rootfix_enabled": (
                v538_refiner_output.get("v563_rootfix_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v564_rootfix_enabled": (
                v538_refiner_output.get("v564_rootfix_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v564_proposal_radius": (
                v538_refiner_output.get("v564_proposal_radius", final_prob.new_zeros((final_prob.shape[0], 1)))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0], 1))
            ),
            "v564_proposal_radius_mean": (
                v538_refiner_output.get("v564_proposal_radius_mean", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v564_proposal_shape_prior_abs_mean": (
                v538_refiner_output.get("v564_proposal_shape_prior_abs_mean", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v564_dual_stream_identity_enabled": (
                v538_refiner_output.get("v564_dual_stream_identity_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v564_typed_spatial_feedback_disabled": (
                v538_refiner_output.get("v564_typed_spatial_feedback_disabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v565_rootfix_enabled": (
                v538_refiner_output.get("v565_rootfix_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v565_seed_logits": (
                v538_refiner_output.get("v565_seed_logits", final_prob.new_zeros((final_prob.shape[0],1,*final_prob.shape[-2:])))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0],1,*final_prob.shape[-2:]))
            ),
            "v565_seed_probability_mean": v538_refiner_output.get("v565_seed_probability_mean", final_prob.new_zeros(())),
            "v565_relative_support_mean": v538_refiner_output.get("v565_relative_support_mean", final_prob.new_zeros(())),
            "v565_attention_radius_mean": v538_refiner_output.get("v565_attention_radius_mean", final_prob.new_zeros(())),
            "v565_peak_to_background_contrast_mean": v538_refiner_output.get("v565_peak_to_background_contrast_mean", final_prob.new_zeros(())),
            "v565_shape_condition_abs_mean": v538_refiner_output.get("v565_shape_condition_abs_mean", final_prob.new_zeros(())),
            "clean_dynamic_component_set_enabled": v538_refiner_output.get("clean_dynamic_component_set_enabled", final_prob.new_zeros(())),
            "tc_drcs_enabled": v538_refiner_output.get("tc_drcs_enabled", final_prob.new_zeros(())),
            "tc_stage0_logits": v538_refiner_output.get("tc_stage0_logits", torch.zeros_like(v538_refiner_output["slot_mask_logits"])),
            "tc_pilot_logits": v538_refiner_output.get("tc_pilot_logits", torch.zeros_like(v538_refiner_output["slot_mask_logits"])),
            "tc_pilot_action_logits": v538_refiner_output.get("tc_pilot_action_logits", final_prob.new_zeros((*v538_refiner_output["slot_action_logits"].shape,))),
            "tc_pilot_valid": v538_refiner_output.get("tc_pilot_valid", torch.zeros_like(v538_refiner_output["slot_valid"], dtype=torch.bool)),
            "tc_pilot_teacher_masks": v538_refiner_output.get("tc_pilot_teacher_masks", torch.zeros_like(v538_refiner_output["slot_masks"])),
            "tc_pilot_teacher_actions": v538_refiner_output.get("tc_pilot_teacher_actions", torch.zeros_like(v538_refiner_output["slot_action_logits"][..., 0], dtype=torch.long)),
            "tc_teacher_raw_count": v538_refiner_output.get("tc_teacher_raw_count", final_prob.new_zeros((final_prob.shape[0],))),
            "clean_attention_precision_mean": v538_refiner_output.get("clean_attention_precision_mean", final_prob.new_zeros(())),
            "clean_mask_precision_mean": v538_refiner_output.get("clean_mask_precision_mean", final_prob.new_zeros(())),
            "clean_loss_log_vars": v538_refiner_output.get("clean_loss_log_vars", final_prob.new_zeros((4,))),
            "v563_mask_window": (
                (v538_refiner_output["v563_mask_window"] if "v563_mask_window" in v538_refiner_output else torch.ones_like(v538_refiner_output["slot_masks"]))
                if self.online_component_refinement else torch.ones_like(v538_refiner_output["slot_masks"])
            ),
            "v563_attention_window": (
                v538_refiner_output.get("v563_attention_window", torch.ones_like(v538_refiner_output["slot_masks"]))
                if self.online_component_refinement else torch.ones_like(v538_refiner_output["slot_masks"])
            ),
            "v563_raw_mask_probability": (
                (v538_refiner_output["v563_raw_mask_probability"] if "v563_raw_mask_probability" in v538_refiner_output else torch.zeros_like(v538_refiner_output["slot_masks"]))
                if self.online_component_refinement else torch.zeros_like(v538_refiner_output["slot_masks"])
            ),
            "v563_pre_gate_mask_probability_mean": (
                v538_refiner_output.get("v563_pre_gate_mask_probability_mean", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v563_outside_mask_probability": (
                v538_refiner_output.get("v563_outside_mask_probability", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v563_attention_window_fraction": (
                v538_refiner_output.get("v563_attention_window_fraction", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v563_mask_window_fraction": (
                v538_refiner_output.get("v563_mask_window_fraction", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v563_identity_retention_q1": (
                v538_refiner_output.get("v563_identity_retention_q1", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v563_identity_retention_q2": (
                v538_refiner_output.get("v563_identity_retention_q2", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v551_first_selector_features": (
                v538_refiner_output.get("v551_first_selector_features", final_prob.new_zeros((final_prob.shape[0], 1, 1)))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0], 1, 1))
            ),
            "v551_editor_route_logits": (
                v538_refiner_output.get("v551_editor_route_logits", final_prob.new_zeros((final_prob.shape[0], 1, 5)))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0], 1, 5))
            ),
            "v551_editor_route_probs": (
                v538_refiner_output.get("v551_editor_route_probs", final_prob.new_zeros((final_prob.shape[0], 1, 5)))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0], 1, 5))
            ),
            "v551_editor_route_index": (
                v538_refiner_output.get("v551_editor_route_index", torch.zeros((final_prob.shape[0], 1), device=final_prob.device, dtype=torch.long))
                if self.online_component_refinement else torch.zeros((final_prob.shape[0], 1), device=final_prob.device, dtype=torch.long)
            ),
            "v551_editor_dose_adjust": (
                v538_refiner_output.get("v551_editor_dose_adjust", final_prob.new_ones((final_prob.shape[0], 1)))
                if self.online_component_refinement else final_prob.new_ones((final_prob.shape[0], 1))
            ),
            "v551_editor_region": (
                v538_refiner_output.get("v551_editor_region", final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:])))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:]))
            ),
            "v551_local_residual": (
                v538_refiner_output.get("v551_local_residual", final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:])))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:]))
            ),
            "v551_editor_logit_delta": (
                v538_refiner_output.get("v551_editor_logit_delta", final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:])))
                if self.online_component_refinement else final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:]))
            ),
            "v551_editor_exact_candidate_st": (
                v538_refiner_output.get("v551_editor_exact_candidate_st", final_prob[:, 0][:, None])
                if self.online_component_refinement else final_prob[:, 0][:, None]
            ),
            "v551_editor_enabled": (
                v538_refiner_output.get("v551_editor_enabled", final_prob.new_zeros(()))
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v546_slot_raw_masks": (
                v538_refiner_output["slot_raw_masks"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:]))
            ),
            "v546_slot_competition_enabled": (
                v538_refiner_output["slot_competition_enabled"] if self.online_component_refinement
                else final_prob.new_zeros(())
            ),
            "v546_gain_sign_shadow_deploy_enabled": (
                v538_refiner_output["gain_sign_shadow_deploy_enabled"]
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v546_raw_slot_overlap_mass": (
                v538_refiner_output["slot_raw_overlap_mass"] if self.online_component_refinement
                else final_prob.new_zeros(())
            ),
            "v546_competition_overlap_mass": (
                v538_refiner_output["slot_competition_overlap_mass"] if self.online_component_refinement
                else final_prob.new_zeros(())
            ),
            "v545_slot_hard_masks": (
                v538_refiner_output["slot_hard_masks"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:]))
            ),
            "v545_slot_hard_masks_st": (
                v538_refiner_output["slot_hard_masks_st"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:]))
            ),
            "v545_slot_mask_contrast_active": (
                v538_refiner_output["slot_mask_contrast_active"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1))
            ),
            "v545_adaptive_cardinality_hard_mask_enabled": (
                v538_refiner_output["adaptive_cardinality_hard_mask_enabled"]
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v538_slot_action_logits": (
                v538_refiner_output["slot_action_logits"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1, 4))
            ),
            "v538_slot_action_probs": (
                v538_refiner_output["slot_action_probs"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1, 4))
            ),
            "v538_slot_polarity_logits": (
                v538_refiner_output["slot_polarity_logits"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1, 2))
            ),
            "v538_slot_polarity_probs": (
                v538_refiner_output["slot_polarity_probs"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1, 2))
            ),
            "v538_slot_dose_logits": (
                v538_refiner_output["slot_dose_logits"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1))
            ),
            "v538_slot_doses": (
                v538_refiner_output["slot_doses"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1))
            ),
            "v538_slot_signed_dose": (
                v538_refiner_output["slot_signed_dose"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1))
            ),
            "v538_slot_gain_scores": (
                v538_refiner_output["slot_gain_scores"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1))
            ),
            "v541_slot_gain_normalized": (
                v538_refiner_output["slot_gain_normalized"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1))
            ),
            "v543_slot_gain_magnitude_normalized": (
                v538_refiner_output["slot_gain_magnitude_normalized"]
                if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1))
            ),
            "v543_slot_signed_outcome": (
                v538_refiner_output["slot_signed_outcome"]
                if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1))
            ),
            "v543_slot_outcome_logits": (
                v538_refiner_output["slot_outcome_logits"]
                if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1, 3))
            ),
            "v543_slot_outcome_probs": (
                v538_refiner_output["slot_outcome_probs"]
                if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1, 3))
            ),
            "v543_slot_neutral_probs": (
                v538_refiner_output["slot_neutral_probs"]
                if self.online_component_refinement
                else final_prob.new_ones((final_prob.shape[0], 1))
            ),
            "v541_slot_gain_logvar": (
                v538_refiner_output["slot_gain_logvar"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1))
            ),
            "v541_slot_gain_std": (
                v538_refiner_output["slot_gain_std"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1))
            ),
            "v541_slot_gain_lcb": (
                v538_refiner_output["slot_gain_lcb"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1))
            ),
            "v541_slot_rank_scores": (
                v538_refiner_output["slot_rank_scores"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1))
            ),
            "v541_slot_decision_scores": (
                v538_refiner_output["slot_decision_scores"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1))
            ),
            "v541_slot_benefit_logits": (
                v538_refiner_output["slot_benefit_logits"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1))
            ),
            "v541_slot_benefit_probs": (
                v538_refiner_output["slot_benefit_probs"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1))
            ),
            "v541_slot_harm_logits": (
                v538_refiner_output["slot_harm_logits"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1))
            ),
            "v541_slot_harm_probs": (
                v538_refiner_output["slot_harm_probs"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1))
            ),
            "v547_slot_editability_logits": (
                v538_refiner_output["slot_editability_logits"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1))
            ),
            "v547_slot_editability_probs": (
                v538_refiner_output["slot_editability_probs"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1))
            ),
            "v547_slot_direction_logits": (
                v538_refiner_output["slot_direction_logits"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1))
            ),
            "v547_slot_direction_probs": (
                v538_refiner_output["slot_direction_probs"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1))
            ),
            "v549_slot_selector_features": (
                v538_refiner_output["slot_selector_features"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1, 1))
            ),
            "v549_direction_head_weight": (
                v538_refiner_output["direction_head_weight"] if self.online_component_refinement
                else final_prob.new_zeros((1, 1))
            ),
            "v549_direction_head_bias": (
                v538_refiner_output["direction_head_bias"] if self.online_component_refinement
                else final_prob.new_zeros((1,))
            ),
            "v549_editability_head_weight": (
                v538_refiner_output["editability_head_weight"] if self.online_component_refinement
                else final_prob.new_zeros((1, 1))
            ),
            "v549_editability_head_bias": (
                v538_refiner_output["editability_head_bias"] if self.online_component_refinement
                else final_prob.new_zeros((1,))
            ),
            "v549_gain_head_weight": (
                v538_refiner_output["gain_head_weight"] if self.online_component_refinement
                else final_prob.new_zeros((1, 1))
            ),
            "v549_gain_head_bias": (
                v538_refiner_output["gain_head_bias"] if self.online_component_refinement
                else final_prob.new_zeros((1,))
            ),
            "v549_factorized_deployment_enabled": (
                v538_refiner_output["factorized_deployment_enabled"]
                if self.online_component_refinement else final_prob.new_zeros(())
            ),
            "v547_factorized_outcome_enabled": (
                v538_refiner_output["factorized_outcome_enabled"] if self.online_component_refinement
                else final_prob.new_zeros(())
            ),
            "v548_factorized_direction_zero_init_enabled": (
                v538_refiner_output[
                    "factorized_direction_zero_init_enabled"
                ] if self.online_component_refinement
                else final_prob.new_zeros(())
            ),
            "v538_slot_presence_logits": (
                v538_refiner_output["slot_presence_logits"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1))
            ),
            "v538_slot_presence_probs": (
                v538_refiner_output["slot_presence_probs"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1))
            ),
            "v538_slot_candidate_probs": (
                v538_refiner_output["slot_candidate_probs"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:]))
            ),
            "v541_slot_exact_candidate_probs": (
                v538_refiner_output["slot_exact_candidate_probs"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:]))
            ),
            "v545_slot_exact_candidate_st_probs": (
                v538_refiner_output["slot_exact_candidate_st_probs"]
                if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:]))
            ),
            "v538_slot_discrete_candidate_probs": (
                v538_refiner_output["slot_discrete_candidate_probs"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:]))
            ),
            "v538_slot_continuous_candidate_probs": (
                v538_refiner_output["slot_continuous_candidate_probs"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:]))
            ),
            "v538_selected_signed_delta": (
                v538_refiner_output["selected_signed_delta"] if self.online_component_refinement
                else final_prob.new_zeros(final_prob.shape)
            ),
            "v538_selected_final_probability": (
                v538_refiner_output["selected_final_probability"] if self.online_component_refinement
                else final_prob
            ),
            "v538_slot_valid": (
                v538_refiner_output["slot_valid"] if self.online_component_refinement
                else torch.zeros((final_prob.shape[0], 1), device=final_prob.device, dtype=torch.bool)
            ),
            "v538_slot_area_fraction": (
                v538_refiner_output["slot_area_fraction"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0], 1))
            ),
            "v538_accepted_slots": (
                v538_refiner_output["accepted_slots"] if self.online_component_refinement
                else torch.zeros((final_prob.shape[0], 1), device=final_prob.device, dtype=torch.bool)
            ),
            "v538_accepted_count": (
                v538_refiner_output["accepted_count"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0],))
            ),
            "v538_changed_fraction": (
                v538_refiner_output["changed_fraction"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0],))
            ),
            "v541_shadow_selected_final_probability": (
                v538_refiner_output["shadow_selected_final_probability"] if self.online_component_refinement
                else final_prob
            ),
            "v541_shadow_accepted_slots": (
                v538_refiner_output["shadow_accepted_slots"] if self.online_component_refinement
                else torch.zeros((final_prob.shape[0], 1), device=final_prob.device, dtype=torch.bool)
            ),
            "v541_shadow_predicted_execute": (
                v538_refiner_output["shadow_predicted_execute"] if self.online_component_refinement
                else torch.zeros((final_prob.shape[0],), device=final_prob.device, dtype=torch.bool)
            ),
            "v541_shadow_accepted_count": (
                v538_refiner_output["shadow_accepted_count"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0],))
            ),
            "v541_shadow_changed_fraction": (
                v538_refiner_output["shadow_changed_fraction"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0],))
            ),
            "v552r44_audit_selected_final_probability": (
                v538_refiner_output["audit_selected_final_probability"] if self.online_component_refinement
                else final_prob
            ),
            "v552r44_audit_accepted_slots": (
                v538_refiner_output["audit_accepted_slots"] if self.online_component_refinement
                else torch.zeros((final_prob.shape[0], 1), device=final_prob.device, dtype=torch.bool)
            ),
            "v552r44_audit_predicted_execute": (
                v538_refiner_output["audit_predicted_execute"] if self.online_component_refinement
                else torch.zeros((final_prob.shape[0],), device=final_prob.device, dtype=torch.bool)
            ),
            "v552r44_audit_accepted_count": (
                v538_refiner_output["audit_accepted_count"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0],))
            ),
            "v552r44_audit_changed_fraction": (
                v538_refiner_output["audit_changed_fraction"] if self.online_component_refinement
                else final_prob.new_zeros((final_prob.shape[0],))
            ),
            "v538_quality_ready": final_prob.new_full(
                (final_prob.shape[0],),
                1.0 if bool(self.v538_quality_ready.item()) else 0.0,
            ),
            "v538_quality_ready_streak": final_prob.new_full(
                (final_prob.shape[0],),
                float(self.v538_quality_ready_streak.item()),
            ),
            "v552_quality_bad_streak": final_prob.new_full(
                (final_prob.shape[0],),
                float(self.v552_quality_bad_streak.item()),
            ),
            "v538_online_component_refinement_enabled": final_prob.new_full(
                (final_prob.shape[0],), 1.0 if self.online_component_refinement else 0.0
            ),
            "v537_candidate_masks": (
                v537_ranker_output["candidate_masks"] if self.component_utility_ranking
                else final_prob.new_zeros((final_prob.shape[0], 1, *final_prob.shape[-2:]))
            ),
            "v537_candidate_actions": (
                v537_ranker_output["candidate_actions"] if self.component_utility_ranking
                else torch.zeros((final_prob.shape[0], 1), device=final_prob.device, dtype=torch.long)
            ),
            "v537_candidate_valid": (
                v537_ranker_output["candidate_valid"] if self.component_utility_ranking
                else torch.zeros((final_prob.shape[0], 1), device=final_prob.device, dtype=torch.bool)
            ),
            "v537_candidate_scores": (
                v537_ranker_output["candidate_scores"] if self.component_utility_ranking
                else final_prob.new_zeros((final_prob.shape[0], 1))
            ),
            "v537_selected_index": (
                v537_ranker_output["selected_index"] if self.component_utility_ranking
                else torch.zeros((final_prob.shape[0],), device=final_prob.device, dtype=torch.long)
            ),
            "v537_selected_score": (
                v537_ranker_output["selected_score"] if self.component_utility_ranking
                else final_prob.new_zeros((final_prob.shape[0],))
            ),
            "v537_predicted_execute": (
                v537_ranker_output["predicted_execute"] if self.component_utility_ranking
                else torch.zeros((final_prob.shape[0],), device=final_prob.device, dtype=torch.bool)
            ),
            "v537_retained_component_count": (
                v537_ranker_output["retained_component_count"] if self.component_utility_ranking
                else final_prob.new_zeros((final_prob.shape[0],))
            ),
            "v537_raw_component_count": (
                v537_ranker_output["raw_component_count"] if self.component_utility_ranking
                else final_prob.new_zeros((final_prob.shape[0],))
            ),
            "v537_component_utility_ranking_enabled": final_prob.new_full(
                (final_prob.shape[0],), 1.0 if self.component_utility_ranking else 0.0
            ),
            "v535_adaptive_policy_enabled": final_prob.new_full(
                (final_prob.shape[0],),
                1.0 if self.adaptive_utility_policy else 0.0,
            ),
            "v532_action_candidate_probs": action_candidates,
            "v532_per_action_edit_mass": per_action_edit_mass,
            "v532_predicted_neutral_mass_map": predicted_neutral_mass,
            "v532_predicted_benefit_mass_map": predicted_benefit_mass,
            "v532_predicted_harm_mass_map": predicted_harm_mass,
            "v532_outcome_partition_error": partition_error,
            "v532_route_preserve_probability": preserve_route[:, 0],
            "v532_deployed_preserve_probability": deployed_preserve_probability[:, 0],
            "v532_m1_gradient_scale": c0.new_full(
                (c0.shape[0],), float(m1_grad_scale)
            ),
            "v532_feature_map": feature,
        }
        if self.online_component_refinement:
            # Propagate every native R4 tensor without maintaining another
            # brittle hand-written compatibility list.
            result.update(
                {
                    key: value
                    for key, value in v538_refiner_output.items()
                    if str(key).startswith(("v552r4_", "v552r42_", "v552r43_", "v552r44_", "v552r45_"))
                }
            )
        return result
