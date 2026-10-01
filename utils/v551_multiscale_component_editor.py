"""V551 instance-adaptive multiscale typed intervention composer-editor.

The module is a drop-in replacement for :class:`V538OnlineComponentRefiner`.
It preserves every compatibility output used by the V538--V550 training code,
while changing the executable unit from a whole single-scale slot to a
multiscale local atom.

Design
------
1. Build a learned feature pyramid from the fused V532 context feature.
2. Predict per-slot scale routing, support, typed action, continuous dose,
   boundary residual, and presence.
3. Convert exact hard supports into connected local atoms and remove duplicates.
4. Evaluate a first exact M1 intervention to obtain candidate-conditioned
   features.
5. Let M2 predict Preserve/Edit, action correction, dose adjustment, and a
   bounded local residual for every M1 atom.
6. Evaluate every edited atom with the inherited factorized Benefit/Harm/Gain
   critic without a second dense spatial encoding.
7. Sequentially estimate context-conditioned marginal utility, select Stop or
   another compatible edited atom, and compose two to three complementary
   local edits into one refined candidate.
8. Evaluate the composed candidate under one unified safety contract and fall
   back to the Base prediction when no safe positive-marginal composition exists.

Ground-truth is never used by deployment or evaluation.  During training only,
a detached exact-Teacher trace may be constructed to align Composer states with
its step targets; all executable Student/Shadow/deployment paths remain GT-free.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.v551_loss import build_v552r2_exact_composer_teacher
from utils.v538_online_component_refiner import (
    EPS,
    V538OnlineComponentRefiner,
    _group_count,
    _masked_mean,
    _masked_scalar,
    _safe_logit,
)


def _as_tuple(values: Sequence[int] | int) -> Tuple[int, ...]:
    if isinstance(values, int):
        values = (values,)
    result = tuple(max(int(v), 1) for v in values)
    if not result:
        raise ValueError("V551 requires at least one pyramid scale")
    if result[0] != 1:
        result = (1,) + result
    return tuple(dict.fromkeys(result))




class _V561ResidualQueryDecoderStage(nn.Module):
    """Lightweight pre-norm query decoder stage used by V561 BCRS-M1.

    The block deliberately uses standard attention/FFN primitives only.  A
    single attention head removes a head-count hyperparameter while retaining
    the essential DETR-style image-conditioning mechanism.  ``attn_bias`` is
    an additive log-prior with shape ``[B,K,N]``; for the typed V561 stage it is
    derived only from inference-visible Base probability and the query's
    predicted correction type.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.cross_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, 1, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, 1, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )

    def forward(
        self,
        query: torch.Tensor,
        pixels: torch.Tensor,
        *,
        attn_bias: Optional[torch.Tensor] = None,
        residual_scale: float = 1.0,
        normalize_output: bool = False,
        persistent_identity: Optional[torch.Tensor] = None,
        identity_mix: float = 0.0,
        disable_self_attention: bool = False,
        attention_identity: Optional[torch.Tensor] = None,
        attention_identity_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # V562 can keep a persistent query identity by limiting how much one
        # attention block is allowed to overwrite the slot content.  Historical
        # V561 calls retain residual_scale=1 and normalize_output=False exactly.
        scale = max(float(residual_scale), 0.0)
        mix = min(max(float(identity_mix), 0.0), 1.0)

        def restore_identity(value: torch.Tensor) -> torch.Tensor:
            if persistent_identity is None or mix <= 0.0:
                return value
            # V563 keeps identity as a *persistent state component* rather than
            # an initialization hint.  Convex mixing is deliberately bounded:
            # no attention/FFN update can erase the anchor identity in one step.
            value = (1.0 - mix) * value + mix * persistent_identity
            return F.normalize(value, dim=2)

        identity_scale = max(float(attention_identity_scale), 0.0)

        def attention_query(value: torch.Tensor) -> torch.Tensor:
            if attention_identity is None or identity_scale <= 0.0:
                return value
            # V564 keeps identity in a parallel read-only stream.  Identity can
            # steer attention without repeatedly overwriting adaptive content.
            return value + identity_scale * attention_identity

        if not disable_self_attention:
            qn = self.self_norm(attention_query(query))
            self_update, _ = self.self_attn(qn, qn, qn, need_weights=False)
            query = query + scale * self_update
            query = restore_identity(query)

        qn = self.cross_norm(attention_query(query))
        attn_mask = None
        if attn_bias is not None:
            if attn_bias.ndim != 3:
                raise ValueError(
                    f"V561 attention bias must be [B,K,N], got {tuple(attn_bias.shape)}"
                )
            # MultiheadAttention expects [B * num_heads, K, N].  V561 uses one
            # head by construction, so the batch-shaped prior is already exact.
            attn_mask = attn_bias.to(dtype=qn.dtype)
        cross_update, weights = self.cross_attn(
            qn, pixels, pixels, attn_mask=attn_mask,
            need_weights=True, average_attn_weights=True,
        )
        query = query + scale * cross_update
        query = restore_identity(query)
        query = query + scale * self.ffn(self.ffn_norm(query))
        query = restore_identity(query)
        if normalize_output:
            query = F.normalize(query, dim=2)
        return query, weights


class _R412ResidualBlock(nn.Module):
    """Small local residual block used only by the canonical R4.12 renderer."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.gelu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.gelu(x + y)


def _boundary_band(mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """Return a binary/soft boundary band for ``[B,K,H,W]`` masks."""
    radius = max(int(radius), 1)
    kernel = 2 * radius + 1
    b, k, h, w = mask.shape
    flat = mask.reshape(b * k, 1, h, w)
    dilated = F.max_pool2d(flat, kernel, stride=1, padding=radius)
    eroded = -F.max_pool2d(-flat, kernel, stride=1, padding=radius)
    return (dilated - eroded).clamp(0.0, 1.0).reshape(b, k, h, w)


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if int(radius) <= 0:
        return mask
    radius = int(radius)
    kernel = 2 * radius + 1
    b, k, h, w = mask.shape
    flat = mask.reshape(b * k, 1, h, w)
    return F.max_pool2d(flat, kernel, stride=1, padding=radius).reshape(
        b, k, h, w
    )


def _hard_dice_route_bank(
    probability: torch.Tensor,
    gt: torch.Tensor,
) -> torch.Tensor:
    """Hard Dice for ``probability=[B,N,R,H,W]`` and ``gt=[B,1,H,W]``."""
    if probability.ndim != 5:
        raise ValueError(f"Expected [B,N,R,H,W], got {tuple(probability.shape)}")
    b, n, r, h, w = probability.shape
    flat = (probability.detach() >= 0.5).to(gt.dtype).reshape(b, n * r, h, w)
    gt_expand = gt.expand(-1, n * r, -1, -1)
    intersection = (flat * gt_expand).flatten(2).sum(dim=2)
    denominator = flat.flatten(2).sum(dim=2) + gt_expand.flatten(2).sum(dim=2)
    return ((2.0 * intersection + EPS) / (denominator + EPS)).reshape(b, n, r)


def _repeat_by_parent(value: torch.Tensor, parent_index: torch.Tensor) -> torch.Tensor:
    """Gather slot tensors ``[B,K,...]`` with ``parent_index=[B,N]``."""
    if value.ndim < 2:
        raise ValueError(f"Expected [B,K,...], got {tuple(value.shape)}")
    index = parent_index
    for _ in range(value.ndim - 2):
        index = index.unsqueeze(-1)
    expand_shape = list(parent_index.shape) + list(value.shape[2:])
    index = index.expand(*expand_shape)
    return value.gather(1, index)


class _PyramidBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Dropout2d(float(dropout)),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class V551MultiscaleTypedComposerEditor(V538OnlineComponentRefiner):
    """V538-compatible multiscale M1 plus region-wise M2 composer-editor."""

    def __init__(
        self,
        *,
        feature_channels: int,
        pyramid_scales: Sequence[int] = (1, 2, 4, 8),
        scale_temperature: float = 0.70,
        max_atoms_per_slot: int = 3,
        atomization_start_epoch: int = 1,
        atom_min_pixels: int = 4,
        atom_dedup_iou: float = 0.85,
        gpu_atomizer_enabled: bool = True,
        max_active_atoms: int = 8,
        atom_partition_temperature: float = 0.35,
        atom_partition_extent: float = 0.85,
        atom_presence_threshold: float = 0.35,
        atom_quality_threshold: float = 0.20,
        atom_quality_gate_start_epoch: int = 4,
        single_pass_editor: bool = True,
        boundary_residual_enabled: bool = True,
        boundary_residual_cap: float = 0.75,
        boundary_band_radii: Sequence[int] = (1, 1, 2, 3),
        editor_enabled: bool = True,
        editor_start_epoch: int = 2,
        editor_ramp_epochs: int = 8,
        editor_route_temperature: float = 0.70,
        editor_preserve_bias: float = 1.25,
        editor_dose_adjust_min: float = 0.50,
        editor_dose_adjust_max: float = 1.50,
        editor_local_residual_cap: float = 0.75,
        editor_total_logit_delta_cap: float = 1.00,
        editor_region_radii: Sequence[int] = (1, 2, 4, 7),
        editor_residual_dropout: float = 0.10,
        unified_deployment_gate: bool = True,
        deployment_benefit_harm_margin: float = 0.10,
        multicandidate_composer_enabled: bool = True,
        composer_stop_bias: float = 0.0,
        composer_marginal_correction_cap: float = 0.02,
        composer_overlap_penalty: float = 0.50,
        composer_conflict_penalty: float = 1.00,
        composer_budget_penalty: float = 0.50,
        teacher_decoupled_r2_enabled: bool = False,
        composer_teacher_pool_size: int = 4,
        composer_teacher_stop_margin: float = 0.0,
        composer_deploy_pool_size: int = 3,
        shadow_evidence_start_epoch: int = 14,
        unified_reference_r4_enabled: bool = False,
        editor_relative_margin: float = 1.0e-3,
        editor_safety_benefit_threshold: float = 0.45,
        editor_safety_harm_threshold: float = 0.35,
        editor_incremental_gain_threshold: float = 0.0,
        critic_gain_cap: float = 0.05,
        critic_queue_capacity: int = 256,
        decoupled_critic_r42_enabled: bool = False,
        spatial_evidence_r43_enabled: bool = False,
        audit_gate_r44_enabled: bool = False,
        class_value_decoupling_r44_enabled: bool = False,
        semantic_deployment_r44_enabled: bool = False,
        audit_shadow_start_epoch: int = 12,
        audit_shadow_topk: int = 1,
        error_aware_r45_enabled: bool = False,
        factorized_safety_r45_enabled: bool = False,
        direct_signed_utility_r45_enabled: bool = False,
        factorized_composer_r45_enabled: bool = False,
        native_contract_r46_enabled: bool = False,
        spatial_realization_r47_enabled: bool = False,
        r47_mask_dim: int = 64,
        r47_query_logit_scale: float = 4.0,
        r47_anchor_prior_scale: float = 1.5,
        r47_anchor_min_size: float = 0.04,
        r47_anchor_max_size: float = 0.55,
        r47_direct_slot_components: bool = False,
        iterative_binding_r48_enabled: bool = False,
        r48_decoder_layers: int = 3,
        r48_local_grid_size: int = 3,
        r48_anchor_delta_scale: float = 0.75,
        r48_window_prior_scale: float = 2.0,
        r48_window_temperature: float = 0.025,
        r48_remove_coarse_mask_bias: bool = True,
        r48_deep_supervision_enabled: bool = False,
        r48_dn_enabled: bool = False,
        r48_dn_groups: int = 2,
        r48_dn_noise_scale: float = 0.35,
        r48_teacher_min_pixels: int = 4,
        content_selective_r49_enabled: bool = False,
        r49_anchor_sampling_only: bool = True,
        r49_attention_logit_scale_init: float = 10.0,
        r49_attention_logit_scale_max: float = 100.0,
        r49_dn_curriculum_enabled: bool = False,
        r49_dn_noise_start: float = 0.05,
        r49_dn_noise_final: float = 0.30,
        r49_dn_noise_ramp_epochs: int = 15,
        evidence_proposal_r410_enabled: bool = False,
        r410_use_evidence_proposals: bool = True,
        r410_support_only_local_readout: bool = False,
        r410_proposal_nms_kernel: int = 17,
        r410_proposal_score_threshold: float = 0.05,
        r410_support_expand: float = 1.75,
        r410_support_temperature: float = 0.02,
        r410_support_max_penalty: float = 8.0,
        r410_dn_clean_curriculum_enabled: bool = True,
        r410_dn_clean_epochs: int = 10,
        r410_dn_noise_final: float = 0.20,
        r410_dn_noise_ramp_epochs: int = 50,
        native_residual_set_r411_enabled: bool = False,
        r411_typed_proposal_enabled: bool = True,
        r411_local_roi_decoder_enabled: bool = True,
        r411_use_raw_native_masks: bool = True,
        r411_proposal_nms_kernel: int = 9,
        r411_proposal_score_threshold: float = 0.05,
        r411_initial_box_size: float = 0.08,
        r411_roi_size: int = 32,
        r411_roi_expand: float = 1.50,
        r411_outside_penalty: float = 8.0,
        canonical_shape_r412_enabled: bool = False,
        r412_roi_size: int = 64,
        r412_action_support_strength: float = 0.75,
        r412_action_support_floor: float = 0.05,
        r412_boundary_band_kernel: int = 7,
        geometry_lock_r413_enabled: bool = False,
        r413_query_extent_enabled: bool = True,
        geometry_context_r414_enabled: bool = False,
        r414_context_grid_size: int = 17,
        r414_context_radius: float = 0.20,
        unique_point_r416_enabled: bool = False,
        r416_cross_type_nms_radius_px: float = 4.0,
        r416_asymmetric_ltrb_enabled: bool = False,
        proposal_recovery_r417_enabled: bool = False,
        r417_location_nms_kernel: int = 3,
        r417_location_oversample_factor: int = 4,
        r417_location_dedup_radius_px: float = 2.0,
        r417_shared_offset_enabled: bool = True,
        box_free_mask_set_r418_enabled: bool = False,
        r418_paired_stable_teacher_enabled: bool = True,
        r418_paired_min_pixels: int = 4,
        seeded_masked_attention_r419_enabled: bool = False,
        r419_seed_radius_px: float = 12.0,
        r419_support_dilate_kernel: int = 9,
        r419_outside_logit_penalty: float = 8.0,
        r419_mask_threshold: float = 0.5,
        dynamic_residual_mask_r420_enabled: bool = False,
        r420_dynamic_channels: int = 8,
        r420_type_decoupled_mask_enabled: bool = False,
        r4201_clean_rootfix_enabled: bool = False,
        dense_competitive_residual_set_r4203_enabled: bool = False,
        factorized_residual_existence_identity_r4204_enabled: bool = False,
        r4204_spatial_identity_enabled: bool = False,
        capacity_consistent_overflow_r4205_enabled: bool = False,
        dynamic_visual_instance_binding_r4207_enabled: bool = False,
        normalized_visual_instance_binding_r4208_enabled: bool = False,
        persistent_identity_r4208_enabled: bool = False,
        instance_valid_factorization_r4210_enabled: bool = False,
        variable_cardinality_seeds_r4210_enabled: bool = False,
        independent_overflow_gate_r4210_enabled: bool = False,
        m1_native_alignment_r4210_enabled: bool = False,
        instance_valid_decoupling_r4211_enabled: bool = False,
        proposal_existence_decoupling_r4211_enabled: bool = False,
        geometry_overflow_decoupling_r4211_enabled: bool = False,
        # V552-R4.21.2: candidate-set / utility-interface root repair.
        independent_candidate_set_r4212_enabled: bool = False,
        disable_visual_seed_identity_r4212_enabled: bool = False,
        existence_no_object_r4212_enabled: bool = False,
        candidate_alignment_r4212_enabled: bool = False,
        direct_delta_utility_r4212_enabled: bool = False,
        zero_stop_one_step_r4212_enabled: bool = False,
        # V560: clean scientific core.  This is deliberately a single root
        # switch rather than another cascade of historical sub-features.
        # It keeps the one-step physical signed-DeltaDice M2 contract.
        clean_core_v560_enabled: bool = False,
        # V561: Base-Conditioned Residual Correction Set (BCRS) M1.  One root
        # plus one controlled variant replaces the failed static/legacy mask
        # geometry owner.  Supported variants are: static -> image -> typed.
        base_conditioned_residual_set_v561_enabled: bool = False,
        bcrs_v561_variant: str = "typed",
        # V563: persistent anchor identity + hard local binding.  These values
        # are inference-visible geometry constants, not GT-derived quantities.
        v563_attention_radius: float = 0.24,
        v563_mask_radius: float = 0.20,
        v563_identity_mix: float = 0.60,
        v563_query_residual_scale: float = 0.15,
        v563_outside_logit_penalty: float = 12.0,
        # V564: ownership-consistent complete root fix.  Identity is a parallel
        # attention key, residual heatmap shape conditions the component mask,
        # and proposal-derived extent replaces one fixed suppression radius.
        v564_rootfix_enabled: bool = False,
        v564_attention_identity_scale: float = 0.35,
        v564_proposal_shape_scale: float = 0.50,
        v564_min_mask_radius: float = 0.05,
        v564_max_mask_radius: float = 0.20,
        # V565: decouple dense residual occupancy from instance-center seed
        # detection.  The dedicated seed head is trained on one interior center
        # per factual residual component and is the only owner of inference NMS.
        # Extent is estimated from peak-relative connected residual support, and
        # that support is fed directly into the component mask geometry.
        v565_rootfix_enabled: bool = False,
        v565_seed_nms_radius: float = 0.025,
        v565_support_relative_threshold: float = 0.35,
        v565_extent_quantile: float = 0.90,
        v565_attention_radius_scale: float = 1.35,
        v565_max_attention_radius: float = 0.20,
        v565_shape_scale: float = 0.75,
        clean_dynamic_component_set_enabled: bool = False,
        tc_drcs_enabled: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(feature_channels=feature_channels, **kwargs)
        self.pyramid_scales = _as_tuple(pyramid_scales)
        self.num_scales = len(self.pyramid_scales)
        self.scale_temperature = max(float(scale_temperature), 1.0e-4)
        self.max_atoms_per_slot = max(int(max_atoms_per_slot), 1)
        self.atomization_start_epoch = max(int(atomization_start_epoch), 0)
        self.current_epoch = 0
        self.atom_min_pixels = max(int(atom_min_pixels), 1)
        # V551 root fix: training/inference must never perform CUDA->CPU->SciPy
        # connected-component round trips.  Local atoms are generated by a
        # vectorized principal-axis partition on GPU and compacted to a bounded
        # active set before any candidate encoding.
        self.gpu_atomizer_enabled = bool(gpu_atomizer_enabled)
        self.max_active_atoms = max(int(max_active_atoms), 2)
        self.atom_partition_temperature = max(float(atom_partition_temperature), 1.0e-4)
        self.atom_partition_extent = max(float(atom_partition_extent), 0.05)
        self.atom_presence_threshold = min(max(float(atom_presence_threshold), 0.0), 1.0)
        self.atom_quality_threshold = min(max(float(atom_quality_threshold), 0.0), 1.0)
        self.atom_quality_gate_start_epoch = max(int(atom_quality_gate_start_epoch), 0)
        self.single_pass_editor = bool(single_pass_editor)
        self.atom_dedup_iou = min(max(float(atom_dedup_iou), 0.0), 1.0)
        self.boundary_residual_enabled = bool(boundary_residual_enabled)
        self.boundary_residual_cap = max(float(boundary_residual_cap), 0.0)
        self.editor_enabled = bool(editor_enabled)
        self.editor_start_epoch = max(int(editor_start_epoch), 0)
        self.editor_ramp_epochs = max(int(editor_ramp_epochs), 1)
        self.editor_preserve_bias = float(editor_preserve_bias)
        self.editor_route_temperature = max(float(editor_route_temperature), 1.0e-4)
        self.editor_dose_adjust_min = max(float(editor_dose_adjust_min), 0.0)
        self.editor_dose_adjust_max = max(
            float(editor_dose_adjust_max), self.editor_dose_adjust_min + EPS
        )
        self.editor_local_residual_cap = max(float(editor_local_residual_cap), 0.0)
        self.editor_total_logit_delta_cap = max(float(editor_total_logit_delta_cap), 0.0)
        self.unified_deployment_gate = bool(unified_deployment_gate)
        self.deployment_benefit_harm_margin = max(float(deployment_benefit_harm_margin), 0.0)
        # V552-R1: M2 is an Editor + Outcome Critic + Set Composer.  The
        # composer estimates the marginal utility of adding another edited atom
        # conditioned on the already composed state and explicitly predicts Stop.
        self.multicandidate_composer_enabled = bool(multicandidate_composer_enabled)
        self.composer_stop_bias = float(composer_stop_bias)
        self.composer_marginal_correction_cap = max(
            float(composer_marginal_correction_cap), 0.0
        )
        self.composer_overlap_penalty = max(float(composer_overlap_penalty), 0.0)
        self.composer_conflict_penalty = max(float(composer_conflict_penalty), 0.0)
        self.composer_budget_penalty = max(float(composer_budget_penalty), 0.0)
        # V552-R2 separates all supervision paths from deployment.  This flag is
        # intentionally explicit so historical V551/V552-R1 experiments remain
        # reproducible when old configuration files are loaded.
        self.teacher_decoupled_r2_enabled = bool(teacher_decoupled_r2_enabled)
        self.composer_teacher_pool_size = max(int(composer_teacher_pool_size), 1)
        self.composer_teacher_stop_margin = max(
            float(composer_teacher_stop_margin), 0.0
        )
        self.composer_deploy_pool_size = max(int(composer_deploy_pool_size), 1)
        self.shadow_evidence_start_epoch = max(int(shadow_evidence_start_epoch), 0)
        self.unified_reference_r4_enabled = bool(unified_reference_r4_enabled or clean_dynamic_component_set_enabled)
        self.editor_relative_margin = max(float(editor_relative_margin), 0.0)
        self.editor_safety_benefit_threshold = min(
            max(float(editor_safety_benefit_threshold), 0.0), 1.0
        )
        self.editor_safety_harm_threshold = min(
            max(float(editor_safety_harm_threshold), 0.0), 1.0
        )
        self.editor_incremental_gain_threshold = float(
            editor_incremental_gain_threshold
        )
        self.critic_gain_cap = max(float(critic_gain_cap), 1.0e-4)
        self.critic_queue_capacity = max(int(critic_queue_capacity), 0)
        # V552-R4.2 root fix.  Safety (relative-to-M1) and Utility
        # (absolute-to-Base) are related, but they are not the same prediction
        # problem.  R4.1 let both critics share one route adapter and let gain /
        # composer gradients flow back into the Outcome logits.  Once the
        # composer curriculum became active, the easiest way to reduce the
        # signed-gain error was to move both critics toward Harm.  R4.2 gives
        # each critic its own route representation and predicts positive and
        # negative conditional magnitudes with separate non-negative heads.
        self.decoupled_critic_r42_enabled = bool(decoupled_critic_r42_enabled)
        # R4.3: route utility is spatial. Scalar edit area/dose statistics are
        # insufficient to distinguish an edit aligned with a lesion error from
        # an equally sized edit on already-correct tissue. Each exact route now
        # receives masked visual evidence pooled over its positive and negative
        # changed pixels. This remains GT-free at inference.
        self.spatial_evidence_r43_enabled = bool(spatial_evidence_r43_enabled)
        # V552-R4.4: deployment readiness must be estimated by an independent
        # audit trace.  The audit trace never changes the deployed output and
        # is used only during training to collect exact outcome evidence.
        self.audit_gate_r44_enabled = bool(audit_gate_r44_enabled)
        # Classification and continuous value regression have different
        # feature requirements.  R4.4 gives them separate task adapters so
        # magnitude/expected-gain gradients cannot rotate the Outcome feature.
        self.class_value_decoupling_r44_enabled = bool(
            class_value_decoupling_r44_enabled
        )
        self.semantic_deployment_r44_enabled = bool(
            semantic_deployment_r44_enabled
        )
        self.audit_shadow_start_epoch = max(int(audit_shadow_start_epoch), 0)
        self.audit_shadow_topk = max(int(audit_shadow_topk), 1)
        # V552-R4.5: error-aware localization + factorized safety + direct
        # signed utility + explicit execute/choice/value Composer training.
        self.error_aware_r45_enabled = bool(error_aware_r45_enabled)
        self.factorized_safety_r45_enabled = bool(factorized_safety_r45_enabled)
        self.direct_signed_utility_r45_enabled = bool(
            direct_signed_utility_r45_enabled
        )
        self.factorized_composer_r45_enabled = bool(
            factorized_composer_r45_enabled
        )
        self.native_contract_r46_enabled = bool(native_contract_r46_enabled)
        # V552-R4.7: one query predicts one deployable residual component.
        # The design follows the set-prediction principle of DETR/Mask2Former
        # and the explicit spatial-anchor principle of DAB-DETR/Mask DINO.
        # It is deliberately GT-free in forward/inference; teacher masks are
        # consumed only by the loss after Hungarian assignment.
        self.spatial_realization_r47_enabled = bool(spatial_realization_r47_enabled)
        self.r47_mask_dim = max(int(r47_mask_dim), 16)
        self.r47_query_logit_scale = max(float(r47_query_logit_scale), 0.1)
        self.r47_anchor_prior_scale = max(float(r47_anchor_prior_scale), 0.0)
        self.r47_anchor_min_size = min(max(float(r47_anchor_min_size), 1.0e-4), 0.5)
        self.r47_anchor_max_size = min(max(float(r47_anchor_max_size), self.r47_anchor_min_size + 0.01), 1.0)
        self.r47_direct_slot_components = bool(r47_direct_slot_components)
        # V552-R4.8: iterative local query/component binding.  The normal
        # inference path is GT-free.  Training-only DN queries are initialized
        # from noisy teacher component boxes and reconstructed with the same
        # shared decoder, following DN-DETR/DINO-style denoising supervision.
        self.iterative_binding_r48_enabled = bool(iterative_binding_r48_enabled)
        self.r48_decoder_layers = max(int(r48_decoder_layers), 1)
        self.r48_local_grid_size = max(int(r48_local_grid_size), 3)
        if self.r48_local_grid_size % 2 == 0:
            self.r48_local_grid_size += 1
        self.r48_anchor_delta_scale = max(float(r48_anchor_delta_scale), 0.0)
        self.r48_window_prior_scale = max(float(r48_window_prior_scale), 0.0)
        self.r48_window_temperature = max(float(r48_window_temperature), 1.0e-3)
        self.r48_remove_coarse_mask_bias = bool(r48_remove_coarse_mask_bias)
        self.r48_deep_supervision_enabled = bool(r48_deep_supervision_enabled)
        self.r48_dn_enabled = bool(r48_dn_enabled)
        self.r48_dn_groups = max(int(r48_dn_groups), 1)
        self.r48_dn_noise_scale = max(float(r48_dn_noise_scale), 0.0)
        self.r48_teacher_min_pixels = max(int(r48_teacher_min_pixels), 1)
        # V552-R4.9: R4.8 exposed two structural faults rather than a lack of
        # decoder depth: normalized q/k were divided by sqrt(d), collapsing the
        # 3x3 local softmax to an almost uniform average; and the reference box
        # was added directly to mask logits, turning "where to sample" into
        # "what to segment".  R4.9 keeps the same single decoder but fixes those
        # two contracts.  Anchors guide sampling/geometry only.  Local tokens are
        # selected by scaled cosine compatibility with a learnable temperature.
        self.content_selective_r49_enabled = bool(content_selective_r49_enabled)
        self.r49_anchor_sampling_only = bool(r49_anchor_sampling_only)
        self.r49_attention_logit_scale_max = max(float(r49_attention_logit_scale_max), 1.0)
        self.r49_dn_curriculum_enabled = bool(r49_dn_curriculum_enabled)
        init_scale = min(
            max(float(r49_attention_logit_scale_init), 1.0e-3),
            self.r49_attention_logit_scale_max,
        )
        self.r49_dn_noise_start = max(float(r49_dn_noise_start), 0.0)
        self.r49_dn_noise_final = max(float(r49_dn_noise_final), self.r49_dn_noise_start)
        self.r49_dn_noise_ramp_epochs = max(int(r49_dn_noise_ramp_epochs), 1)
        # V552-R4.10: R4.9 proved that content selection and the downstream
        # policy are no longer the bottleneck.  The remaining failure is a
        # reconstruction mismatch: the reference anchor is accurate enough
        # (DN anchor loss is tiny), but the final mask is still read globally,
        # so visually similar pixels far away from the residual component are
        # activated.  R4.10 therefore uses existing dense typed-error evidence
        # to initialize image-conditioned reference centers, and lets an anchor
        # define only a *negative support domain*: it may suppress pixels far
        # outside the local component, but it can never add foreground logit.
        self.evidence_proposal_r410_enabled = bool(evidence_proposal_r410_enabled)
        self.r410_use_evidence_proposals = bool(r410_use_evidence_proposals)
        self.r410_support_only_local_readout = bool(r410_support_only_local_readout)
        nms = max(int(r410_proposal_nms_kernel), 3)
        self.r410_proposal_nms_kernel = nms if nms % 2 == 1 else nms + 1
        self.r410_proposal_score_threshold = min(max(float(r410_proposal_score_threshold), 0.0), 1.0)
        self.r410_support_expand = max(float(r410_support_expand), 1.0)
        self.r410_support_temperature = max(float(r410_support_temperature), 1.0e-4)
        self.r410_support_max_penalty = max(float(r410_support_max_penalty), 0.0)
        self.r410_dn_clean_curriculum_enabled = bool(r410_dn_clean_curriculum_enabled)
        self.r410_dn_clean_epochs = max(int(r410_dn_clean_epochs), 0)
        self.r410_dn_noise_final = min(max(float(r410_dn_noise_final), 0.0), 0.50)
        self.r410_dn_noise_ramp_epochs = max(int(r410_dn_noise_ramp_epochs), 1)

        # V552-R4.11 root-causal redesign.  R4.10 improved DN reconstruction
        # without improving Native candidate utility.  R4.11 therefore makes
        # residual instance localization an explicitly supervised typed task
        # and decodes every residual only inside a differentiable local ROI.
        # No GT enters Native inference; teacher geometry is consumed only by
        # the loss and the training-only DN branch.
        self.native_residual_set_r411_enabled = bool(native_residual_set_r411_enabled)
        self.r411_typed_proposal_enabled = bool(r411_typed_proposal_enabled)
        self.r411_local_roi_decoder_enabled = bool(r411_local_roi_decoder_enabled)
        self.r411_use_raw_native_masks = bool(r411_use_raw_native_masks)
        nms411 = max(int(r411_proposal_nms_kernel), 3)
        self.r411_proposal_nms_kernel = nms411 if nms411 % 2 == 1 else nms411 + 1
        self.r411_proposal_score_threshold = min(max(float(r411_proposal_score_threshold), 0.0), 1.0)
        self.r411_initial_box_size = min(max(float(r411_initial_box_size), self.r47_anchor_min_size), self.r47_anchor_max_size)
        self.r411_roi_size = max(int(r411_roi_size), 16)
        self.r411_roi_expand = max(float(r411_roi_expand), 1.0)
        self.r411_outside_penalty = max(float(r411_outside_penalty), 1.0)

        # V552-R4.12: keep the successful R4.11 typed locator, but replace the
        # post-localization mask stage with a canonical local renderer.  Shape
        # is learned in normalized ROI coordinates; Base-conditioned action
        # support is inference-safe and prevents physically impossible edits.
        self.canonical_shape_r412_enabled = bool(canonical_shape_r412_enabled)
        self.r412_roi_size = max(int(r412_roi_size), 32)
        self.r412_action_support_strength = max(float(r412_action_support_strength), 0.0)
        self.r412_action_support_floor = min(max(float(r412_action_support_floor), 1.0e-4), 1.0)
        band = max(int(r412_boundary_band_kernel), 3)
        self.r412_boundary_band_kernel = band if band % 2 == 1 else band + 1

        # V552-R4.13: one and only one module owns residual geometry.
        self.geometry_lock_r413_enabled = bool(geometry_lock_r413_enabled)
        self.r413_query_extent_enabled = bool(r413_query_extent_enabled)
        if self.geometry_lock_r413_enabled and not self.canonical_shape_r412_enabled:
            raise ValueError("V552-R4.13 requires V552-R4.12 canonical shape")

        # V552-R4.14: keep the single R4.13 geometry owner, but make residual
        # extent identifiable from local spatial evidence rather than a single
        # center pixel + arbitrary slot identity.  The sampling grid is fixed
        # and inference-safe; its coordinates are detached so size loss cannot
        # move the already separately-supervised center.
        self.geometry_context_r414_enabled = bool(geometry_context_r414_enabled)
        if self.geometry_context_r414_enabled and not self.geometry_lock_r413_enabled:
            raise ValueError("V552-R4.14 requires V552-R4.13 geometry lock")
        grid414 = max(int(r414_context_grid_size), 5)
        self.r414_context_grid_size = grid414 if grid414 % 2 == 1 else grid414 + 1
        self.r414_context_radius = min(max(float(r414_context_radius), 0.01), 0.25)

        # V552-R4.16: keep the same single geometry owner, but fix the two
        # structural failures exposed by R4.15.  First, one physical residual
        # point may occupy at most one query slot across all action channels.
        # Second, optional asymmetric edge offsets replace center-symmetric
        # width/height regression, so a proposal point can be a reliable point
        # inside/near a tiny residual rather than an unrealistically exact box
        # centroid.  No second geometry decoder is introduced.
        self.unique_point_r416_enabled = bool(unique_point_r416_enabled)
        self.r416_cross_type_nms_radius_px = max(float(r416_cross_type_nms_radius_px), 1.0)
        self.r416_asymmetric_ltrb_enabled = bool(r416_asymmetric_ltrb_enabled)
        if (self.unique_point_r416_enabled or self.r416_asymmetric_ltrb_enabled) and not self.geometry_context_r414_enabled:
            raise ValueError("V552-R4.16 requires the R4.14 contextual extent path")

        # V552-R4.17: proposal recovery is explicitly location-first.  A
        # dedicated one-channel heatmap answers only "where is a residual?";
        # the historical four typed maps are retained only for action binding.
        # This prevents action competition/local-max suppression from erasing a
        # spatially valid residual before Top-K.  A shared sub-pixel offset also
        # removes the remaining dependency of geometric localization on a
        # potentially wrong action argmax.
        self.proposal_recovery_r417_enabled = bool(proposal_recovery_r417_enabled)
        nms417 = max(int(r417_location_nms_kernel), 1)
        self.r417_location_nms_kernel = nms417 if nms417 % 2 == 1 else nms417 + 1
        self.r417_location_oversample_factor = max(int(r417_location_oversample_factor), 1)
        self.r417_location_dedup_radius_px = max(float(r417_location_dedup_radius_px), 0.0)
        self.box_free_mask_set_r418_enabled = bool(box_free_mask_set_r418_enabled)
        self.r418_paired_stable_teacher_enabled = bool(r418_paired_stable_teacher_enabled)
        self.r418_paired_min_pixels = max(int(r418_paired_min_pixels), 1)
        self.seeded_masked_attention_r419_enabled = bool(seeded_masked_attention_r419_enabled)
        self.r419_seed_radius_px = max(float(r419_seed_radius_px), 1.0)
        k419 = max(int(r419_support_dilate_kernel), 1)
        self.r419_support_dilate_kernel = k419 if k419 % 2 == 1 else k419 + 1
        self.r419_outside_logit_penalty = max(float(r419_outside_logit_penalty), 0.0)
        self.r419_mask_threshold = min(max(float(r419_mask_threshold), 0.05), 0.95)
        self.dynamic_residual_mask_r420_enabled = bool(dynamic_residual_mask_r420_enabled)
        self.r420_dynamic_channels = max(int(r420_dynamic_channels), 4)
        self.r420_type_decoupled_mask_enabled = bool(r420_type_decoupled_mask_enabled)
        self.r4201_clean_rootfix_enabled = bool(r4201_clean_rootfix_enabled)
        self.dense_competitive_residual_set_r4203_enabled = bool(
            dense_competitive_residual_set_r4203_enabled
        )
        self.factorized_residual_existence_identity_r4204_enabled = bool(
            factorized_residual_existence_identity_r4204_enabled
        )
        self.r4204_spatial_identity_enabled = bool(r4204_spatial_identity_enabled)
        self.capacity_consistent_overflow_r4205_enabled = bool(
            capacity_consistent_overflow_r4205_enabled
        )
        # V552-R4.20.7: dynamic visual instance binding.  The R4.17 location
        # field chooses K image-conditioned residual seeds; only the visual
        # feature sampled at each seed is injected into the corresponding
        # learned slot query.  The seed never gates/crops the mask: every slot
        # still competes over the full HxW residual field.  No parameter, loss,
        # radius support, box, or new threshold is introduced.
        self.dynamic_visual_instance_binding_r4207_enabled = bool(
            dynamic_visual_instance_binding_r4207_enabled
        )
        # V552-R4.20.8 separates image-conditioned identity from adaptive
        # content.  A2 normalizes both query branches and uses the already
        # calibrated R4.17 peak probability as a continuous seed strength.
        # A3 keeps that identity key persistent across the self-pooled content
        # refinement.  Neither switch adds parameters, spatial support, or loss
        # weights.
        self.normalized_visual_instance_binding_r4208_enabled = bool(
            normalized_visual_instance_binding_r4208_enabled
        )
        self.persistent_identity_r4208_enabled = bool(
            persistent_identity_r4208_enabled
        )
        # V552-R4.21.0 fixes five definition-level contradictions without
        # changing the full-image editable mask decoder.  K remains a maximum
        # capacity, not a required number of visual seeds; overflow obtains its
        # own conditional gate instead of competing in the K-slot identity
        # simplex; and the deployable M1Native path can optionally remain in the
        # graph for standard segmentation alignment.
        self.instance_valid_factorization_r4210_enabled = bool(
            instance_valid_factorization_r4210_enabled
        )
        self.variable_cardinality_seeds_r4210_enabled = bool(
            variable_cardinality_seeds_r4210_enabled
        )
        self.independent_overflow_gate_r4210_enabled = bool(
            independent_overflow_gate_r4210_enabled
        )
        self.m1_native_alignment_r4210_enabled = bool(
            m1_native_alignment_r4210_enabled
        )
        # V552-R4.21.1 repairs the two contradictions exposed by the R4.21.0
        # preflight: spatial proposal confidence must not own instance existence,
        # and overflow rejection must not participate in query/mask geometry.
        self.instance_valid_decoupling_r4211_enabled = bool(
            instance_valid_decoupling_r4211_enabled
        )
        self.proposal_existence_decoupling_r4211_enabled = bool(
            proposal_existence_decoupling_r4211_enabled
        )
        self.geometry_overflow_decoupling_r4211_enabled = bool(
            geometry_overflow_decoupling_r4211_enabled
        )
        # V552-R4.21.2 closes the remaining M1->M2 contract mismatch.  M1 is
        # now allowed to be a true variable-cardinality *set* predictor rather
        # than a per-pixel K-way partition.  Existence/no-object owns whether a
        # slot is executable; optional seed evidence no longer owns identity;
        # Utility may predict raw signed DeltaDice in physical units; and the
        # one-step composer can compare those utilities against Stop=0.
        self.independent_candidate_set_r4212_enabled = bool(
            independent_candidate_set_r4212_enabled
        )
        self.disable_visual_seed_identity_r4212_enabled = bool(
            disable_visual_seed_identity_r4212_enabled
        )
        self.existence_no_object_r4212_enabled = bool(
            existence_no_object_r4212_enabled
        )
        self.candidate_alignment_r4212_enabled = bool(
            candidate_alignment_r4212_enabled
        )
        self.direct_delta_utility_r4212_enabled = bool(
            direct_delta_utility_r4212_enabled
        )
        self.zero_stop_one_step_r4212_enabled = bool(
            zero_stop_one_step_r4212_enabled
        )
        self.clean_core_v560_enabled = bool(clean_core_v560_enabled)
        self.base_conditioned_residual_set_v561_enabled = bool(
            base_conditioned_residual_set_v561_enabled
        )
        self.bcrs_v561_variant = str(bcrs_v561_variant).strip().lower()
        if self.bcrs_v561_variant not in {"static", "image", "typed", "rootfix", "persistent"}:
            raise ValueError(
                "V561_BCRS_VARIANT must be one of: static, image, typed, rootfix, persistent"
            )
        self.v563_attention_radius = min(max(float(v563_attention_radius), 0.02), 0.75)
        self.v563_mask_radius = min(max(float(v563_mask_radius), 0.02), self.v563_attention_radius)
        self.v563_identity_mix = min(max(float(v563_identity_mix), 0.0), 0.95)
        self.v563_query_residual_scale = min(max(float(v563_query_residual_scale), 0.0), 1.0)
        self.v563_outside_logit_penalty = max(float(v563_outside_logit_penalty), 0.0)
        self.v564_rootfix_enabled = bool(v564_rootfix_enabled)
        self.v564_attention_identity_scale = min(max(float(v564_attention_identity_scale), 0.0), 1.0)
        self.v564_proposal_shape_scale = min(max(float(v564_proposal_shape_scale), 0.0), 4.0)
        self.v564_min_mask_radius = min(max(float(v564_min_mask_radius), 0.02), self.v563_mask_radius)
        self.v564_max_mask_radius = min(
            max(float(v564_max_mask_radius), self.v564_min_mask_radius),
            self.v563_attention_radius,
        )
        if self.v564_rootfix_enabled and self.bcrs_v561_variant != "persistent":
            raise ValueError("V564 requires V561_BCRS_VARIANT=persistent")
        self.v565_rootfix_enabled = bool(v565_rootfix_enabled)
        self.v565_seed_nms_radius = min(max(float(v565_seed_nms_radius), 0.005), 0.10)
        self.v565_support_relative_threshold = min(max(float(v565_support_relative_threshold), 0.10), 0.90)
        self.v565_extent_quantile = min(max(float(v565_extent_quantile), 0.50), 0.99)
        self.v565_attention_radius_scale = min(max(float(v565_attention_radius_scale), 1.0), 3.0)
        self.v565_max_attention_radius = min(
            max(float(v565_max_attention_radius), self.v564_min_mask_radius),
            self.v563_attention_radius,
        )
        self.v565_shape_scale = min(max(float(v565_shape_scale), 0.0), 4.0)
        self.clean_dynamic_component_set_enabled = bool(clean_dynamic_component_set_enabled)
        self.tc_drcs_enabled = bool(tc_drcs_enabled)
        if self.tc_drcs_enabled and not self.clean_dynamic_component_set_enabled:
            raise ValueError("TC-DRCS requires the shared clean component-set scaffold")
        if self.clean_dynamic_component_set_enabled:
            # Clean formal path owns geometry directly. Historical V564/V565
            # inference constants are bypassed rather than silently inherited.
            self.v565_rootfix_enabled = False
        if self.v565_rootfix_enabled and not self.v564_rootfix_enabled:
            raise ValueError("V565 extends V564 strict ownership and requires V564_ROOTFIX_ENABLED=true")
        if self.base_conditioned_residual_set_v561_enabled and not self.clean_core_v560_enabled:
            raise ValueError(
                "V561 BCRS-M1 requires V560 clean-core utility/deployment contract"
            )
        # V560/V561 own one physical candidate per slot and one one-step M2 choice.
        # Reuse the existing direct-slot plumbing but do not mutate user-visible
        # config values or instantiate a parallel model.
        if self.clean_core_v560_enabled:
            self.r47_direct_slot_components = True
        if (self.zero_stop_one_step_r4212_enabled or self.clean_core_v560_enabled) and int(self.max_steps) != 1:
            raise ValueError("V552-R4.21.2 zero-stop policy requires one-step composer")
        if self.existence_no_object_r4212_enabled and not self.independent_candidate_set_r4212_enabled:
            raise ValueError("V552-R4.21.2 no-object execution requires independent candidate masks")
        if (
            self.proposal_existence_decoupling_r4211_enabled
            or self.geometry_overflow_decoupling_r4211_enabled
        ) and not self.instance_valid_decoupling_r4211_enabled:
            raise ValueError("V552-R4.21.1 subfeatures require the R4.21.1 root contract")
        if self.instance_valid_decoupling_r4211_enabled and not self.instance_valid_factorization_r4210_enabled:
            raise ValueError("V552-R4.21.1 requires the R4.21.0 instance-valid scaffold")
        if self.proposal_existence_decoupling_r4211_enabled and self.variable_cardinality_seeds_r4210_enabled:
            raise ValueError(
                "V552-R4.21.1 proposal/existence decoupling supersedes the failed R4.21.0 hard seed-validity gate"
            )
        if self.geometry_overflow_decoupling_r4211_enabled and not self.independent_overflow_gate_r4210_enabled:
            raise ValueError("V552-R4.21.1 geometry/overflow decoupling requires the independent R4.21.0 overflow gate")
        if (
            self.variable_cardinality_seeds_r4210_enabled
            or self.independent_overflow_gate_r4210_enabled
            or self.m1_native_alignment_r4210_enabled
        ) and not self.instance_valid_factorization_r4210_enabled:
            raise ValueError("V552-R4.21.0 subfeatures require the R4.21.0 root contract")
        if self.variable_cardinality_seeds_r4210_enabled and not self.normalized_visual_instance_binding_r4208_enabled:
            raise ValueError("V552-R4.21.0 variable-cardinality seeds require R4.20.8 normalized visual binding")
        if self.independent_overflow_gate_r4210_enabled and not self.capacity_consistent_overflow_r4205_enabled:
            raise ValueError("V552-R4.21.0 independent overflow requires the R4.20.5 residual-capacity scaffold")
        if self.persistent_identity_r4208_enabled and not self.normalized_visual_instance_binding_r4208_enabled:
            raise ValueError(
                "V552-R4.20.8 persistent identity requires normalized visual binding"
            )
        if (
            self.dynamic_visual_instance_binding_r4207_enabled
            and self.normalized_visual_instance_binding_r4208_enabled
        ):
            raise ValueError(
                "V552-R4.20.7 and R4.20.8 query interventions are mutually exclusive"
            )
        if self.normalized_visual_instance_binding_r4208_enabled:
            if not self.capacity_consistent_overflow_r4205_enabled:
                raise ValueError(
                    "V552-R4.20.8 requires the R4.20.5 capacity-consistent factorization"
                )
            if not self.proposal_recovery_r417_enabled:
                raise ValueError(
                    "V552-R4.20.8 requires the independently supervised R4.17 location field"
                )
            if self.r4204_spatial_identity_enabled:
                raise ValueError(
                    "V552-R4.20.8 forbids self-derived spatial identity feedback"
                )
        self._dense_set_family_enabled = bool(
            self.dense_competitive_residual_set_r4203_enabled
            or self.factorized_residual_existence_identity_r4204_enabled
        )
        if self.dynamic_visual_instance_binding_r4207_enabled:
            if not self.capacity_consistent_overflow_r4205_enabled:
                raise ValueError(
                    "V552-R4.20.7 requires the R4.20.5 capacity-consistent K+overflow factorization"
                )
            if not self.proposal_recovery_r417_enabled:
                raise ValueError(
                    "V552-R4.20.7 requires the independently supervised R4.17 location field"
                )
            if self.r4204_spatial_identity_enabled:
                raise ValueError(
                    "V552-R4.20.7 forbids self-derived centroid/variance identity feedback; "
                    "only exogenous R4.17 visual seeds may break slot symmetry"
                )
        if self.capacity_consistent_overflow_r4205_enabled:
            if not self.factorized_residual_existence_identity_r4204_enabled:
                raise ValueError(
                    "V552-R4.20.5 requires the R4.20.4 factorized residual-existence scaffold"
                )
            if self.r4204_spatial_identity_enabled:
                raise ValueError(
                    "V552-R4.20.5 forbids the failed R4.20.4 self-derived spatial identity; "
                    "overflow capacity must be isolated before any external positional ablation"
                )
        if self.factorized_residual_existence_identity_r4204_enabled:
            if not self.dense_competitive_residual_set_r4203_enabled:
                raise ValueError(
                    "V552-R4.20.4 uses the R4.20.3 clean dense-set scaffold; "
                    "set V552R4203_ROOTFIX_ENABLED=true as the control scaffold"
                )
            if not self.r4201_clean_rootfix_enabled:
                raise ValueError("V552-R4.20.4 requires the R4.20.1 clean ownership contract")
            if self.dynamic_residual_mask_r420_enabled or self.seeded_masked_attention_r419_enabled:
                raise ValueError("V552-R4.20.4 forbids R4.19/R4.20 geometry owners")
        if self.dense_competitive_residual_set_r4203_enabled:
            if not self.r4201_clean_rootfix_enabled:
                raise ValueError("V552-R4.20.3 requires the R4.20.1 clean ownership contract")
            if not self.box_free_mask_set_r418_enabled:
                raise ValueError("V552-R4.20.3 requires the R4.18 box-free residual-set scaffold")
            if not self.proposal_recovery_r417_enabled:
                raise ValueError("V552-R4.20.3 requires the dense R4.17 residual-location field")
            if self.dynamic_residual_mask_r420_enabled:
                raise ValueError("V552-R4.20.3 replaces the point-conditioned R4.20 dynamic mask path")
            if self.seeded_masked_attention_r419_enabled:
                raise ValueError("V552-R4.20.3 forbids the R4.19 hard spatial-support path")
        if self.seeded_masked_attention_r419_enabled and not self.box_free_mask_set_r418_enabled:
            raise ValueError("V552-R4.19 requires the R4.18 box-free residual-mask set path")
        if self.dynamic_residual_mask_r420_enabled and not self.box_free_mask_set_r418_enabled:
            raise ValueError("V552-R4.20 requires the R4.18 box-free residual-mask set path")
        if self.dynamic_residual_mask_r420_enabled and self.seeded_masked_attention_r419_enabled:
            raise ValueError("V552-R4.20 replaces, rather than stacks on, the R4.19 hard-frontier mask decoder")
        if self.r420_type_decoupled_mask_enabled and not self.dynamic_residual_mask_r420_enabled:
            raise ValueError("V552-R4.20 type/shape decoupling requires the R4.20 dynamic residual-mask head")
        # R4.18 uses R4.17 only as a location seed.  The failed shared-offset
        # and center->extent geometry chain must not own the mask any more.
        self.r417_shared_offset_enabled = bool(r417_shared_offset_enabled) and not self.box_free_mask_set_r418_enabled
        if self.proposal_recovery_r417_enabled and (not self.box_free_mask_set_r418_enabled) and not self.geometry_context_r414_enabled:
            raise ValueError("V552-R4.17 requires the R4.14 contextual extent path unless R4.18 box-free mask-set mode is enabled")
        if self.box_free_mask_set_r418_enabled and not self.iterative_binding_r48_enabled:
            raise ValueError("V552-R4.18 requires the R4.8 iterative query-mask decoder")
        if self.box_free_mask_set_r418_enabled and not self.proposal_recovery_r417_enabled:
            raise ValueError("V552-R4.18 requires the R4.17 independent location head as a query seed")

        boundary_band_radii = tuple(max(int(v), 1) for v in boundary_band_radii)
        editor_region_radii = tuple(max(int(v), 0) for v in editor_region_radii)
        if len(boundary_band_radii) != self.num_scales:
            raise ValueError(
                "V551_BOUNDARY_BAND_RADII must match V551_PYRAMID_SCALES"
            )
        if len(editor_region_radii) != self.num_scales:
            raise ValueError(
                "V551_EDITOR_REGION_RADII must match V551_PYRAMID_SCALES"
            )
        self.boundary_band_radii = boundary_band_radii
        self.editor_region_radii = editor_region_radii

        channels = int(feature_channels)
        # R4.7 query-conditioned high-resolution mask branch.  Pixel features
        # include typed cause/action evidence, Base uncertainty, and normalized XY
        # coordinates; learned slot queries dot-product this map as in Mask
        # DINO.  A mild grid-anchor prior gives each query a stable spatial
        # identity, then a mask-derived soft anchor refines that location.
        r47_groups = _group_count(self.r47_mask_dim)
        self.r47_pixel_encoder = nn.Sequential(
            nn.Conv2d(channels + 13, self.r47_mask_dim, 1, bias=False),
            nn.GroupNorm(r47_groups, self.r47_mask_dim),
            nn.GELU(),
            nn.Conv2d(self.r47_mask_dim, self.r47_mask_dim, 3, padding=1, bias=False),
            nn.GroupNorm(r47_groups, self.r47_mask_dim),
            nn.GELU(),
        )
        self.r47_slot_queries = nn.Embedding(self.num_slots, self.r47_mask_dim)
        if self.clean_core_v560_enabled:
            self.v560_mask_bias = nn.Parameter(torch.zeros(self.num_slots))
        else:
            self.register_parameter("v560_mask_bias", None)

        # V561 BCRS-M1: all A0/A1/A2 variants instantiate the exact same
        # topology so same-seed ablations do not receive an initialization/RNG
        # advantage.  The variant changes forward ownership only.
        if self.base_conditioned_residual_set_v561_enabled:
            v561_dim = self.r47_mask_dim
            self.v561_global_decoder = _V561ResidualQueryDecoderStage(v561_dim)
            self.v561_typed_decoder = _V561ResidualQueryDecoderStage(v561_dim)
            self.v561_query_action_head = nn.Linear(v561_dim, 4)
            self.v561_mask_embed = nn.Sequential(
                nn.Linear(v561_dim, v561_dim),
                nn.GELU(),
                nn.Linear(v561_dim, v561_dim),
            )
            self.v561_mask_bias = nn.Parameter(torch.zeros(self.num_slots))
            # Uniform action probabilities make the typed support exactly 0.5
            # at initialization for every Base probability: the prior is
            # therefore neutral before evidence/action learning begins.
            nn.init.zeros_(self.v561_query_action_head.weight)
            nn.init.zeros_(self.v561_query_action_head.bias)
            if self.bcrs_v561_variant in {"rootfix", "persistent"}:
                # V562 root repair / V563 persistent-binding extension.  A dense residualness heatmap proposes K
                # inference-visible anchors.  Each query receives its own
                # sampled visual feature + positional embedding before local
                # image binding.  Presence is query-owned rather than pooled
                # from a mask that may still be wrong early in training.
                v562_groups = _group_count(v561_dim)
                self.v562_residual_head = nn.Sequential(
                    nn.Conv2d(v561_dim, v561_dim, 3, padding=1, bias=False),
                    nn.GroupNorm(v562_groups, v561_dim),
                    nn.GELU(),
                    nn.Conv2d(v561_dim, 1, 1),
                )
                self.v562_anchor_feature_proj = nn.Linear(v561_dim, v561_dim)
                self.v562_anchor_pos_mlp = nn.Sequential(
                    nn.Linear(2, v561_dim),
                    nn.GELU(),
                    nn.Linear(v561_dim, v561_dim),
                )
                self.v562_query_presence_head = nn.Linear(v561_dim, 1)
                # V565 has a second, independent inference-visible head whose
                # semantics are *instance center*, not residual occupancy.  This
                # removes the V564 mismatch where a union-segmentation map was
                # later interpreted as a set of object centers.
                if self.v565_rootfix_enabled:
                    self.v565_seed_head = nn.Sequential(
                        nn.Conv2d(v561_dim, v561_dim, 3, padding=1, bias=False),
                        nn.GroupNorm(v562_groups, v561_dim),
                        nn.GELU(),
                        nn.Conv2d(v561_dim, 1, 1),
                    )
                    # Dense residual occupancy and center confidence become
                    # first-class pixel inputs to the mask geometry rather than
                    # a tiny post-hoc logit perturbation.
                    self.v565_mask_pixel_fuse = nn.Conv2d(v561_dim + 2, v561_dim, 1, bias=False)
                    # Start exactly from the V564 visual pixel embedding and
                    # expose the two proposal channels as a small learnable
                    # perturbation.  This avoids a random new pixel encoder from
                    # destroying candidate geometry at epoch 1.
                    with torch.no_grad():
                        self.v565_mask_pixel_fuse.weight.zero_()
                        eye = torch.eye(v561_dim, dtype=self.v565_mask_pixel_fuse.weight.dtype)
                        self.v565_mask_pixel_fuse.weight[:, :v561_dim, 0, 0].copy_(eye)
                        nn.init.normal_(self.v565_mask_pixel_fuse.weight[:, v561_dim:, 0, 0], mean=0.0, std=0.01)
                    nn.init.constant_(self.v565_seed_head[-1].bias, -3.0)
                else:
                    self.v565_seed_head = None
                    self.v565_mask_pixel_fuse = None
                if self.clean_dynamic_component_set_enabled:
                    # No Gaussian center head, NMS radius, fixed attention radius,
                    # support threshold, extent quantile or shape-scale constant.
                    # Spatial concentration is learned from the candidate mask task.
                    self.clean_mask_pixel_fuse = nn.Conv2d(v561_dim + 1, v561_dim, 1, bias=False)
                    self.clean_attention_precision_head = nn.Linear(v561_dim, 1)
                    self.clean_mask_precision_head = nn.Linear(v561_dim, 1)
                    self.clean_loss_log_vars = nn.Parameter(torch.zeros(4))
                    # Minimal CLEAN M2: one direct signed DeltaDice regressor.
                    # It consumes the compact candidate-conditioned selector feature
                    # and replaces all Benefit/Harm/Rank/LCB parameterizations.
                    self.clean_gain_head = nn.Linear(self.gain_head.in_features, 1)
                    # TC-DRCS replaces hard peak/ownership binding with a direct
                    # differentiable mask-set path.  The residual channel remains
                    # inference-visible evidence; localization is carried by soft
                    # mask-guided cross-attention.  Pilot queries are training-only.
                    self.tc_mask_pixel_fuse = nn.Conv2d(v561_dim + 1, v561_dim, 1, bias=False)
                    self.tc_pilot_query_proj = nn.Linear(v561_dim, v561_dim, bias=False)
                    with torch.no_grad():
                        self.clean_mask_pixel_fuse.weight.zero_()
                        eye = torch.eye(v561_dim, dtype=self.clean_mask_pixel_fuse.weight.dtype)
                        self.clean_mask_pixel_fuse.weight[:, :v561_dim, 0, 0].copy_(eye)
                        self.tc_mask_pixel_fuse.weight.zero_()
                        self.tc_mask_pixel_fuse.weight[:, :v561_dim, 0, 0].copy_(eye)
                        self.tc_pilot_query_proj.weight.copy_(eye)
                    nn.init.zeros_(self.clean_attention_precision_head.weight)
                    nn.init.zeros_(self.clean_attention_precision_head.bias)
                    nn.init.zeros_(self.clean_mask_precision_head.weight)
                    nn.init.zeros_(self.clean_mask_precision_head.bias)
                    nn.init.normal_(self.clean_gain_head.weight, mean=0.0, std=1.0e-3)
                    nn.init.zeros_(self.clean_gain_head.bias)
                else:
                    self.clean_mask_pixel_fuse = None
                    self.clean_attention_precision_head = None
                    self.clean_mask_precision_head = None
                    self.clean_gain_head = None
                    self.tc_mask_pixel_fuse = None
                    self.tc_pilot_query_proj = None
                    self.register_parameter("clean_loss_log_vars", None)
                nn.init.constant_(self.v562_residual_head[-1].bias, -2.0)
                nn.init.zeros_(self.v562_query_presence_head.weight)
                nn.init.zeros_(self.v562_query_presence_head.bias)
            else:
                self.v562_residual_head = None
                self.v562_anchor_feature_proj = None
                self.v562_anchor_pos_mlp = None
                self.v562_query_presence_head = None
                self.v565_seed_head = None
                self.v565_mask_pixel_fuse = None
                self.clean_mask_pixel_fuse = None
                self.clean_attention_precision_head = None
                self.clean_mask_precision_head = None
                self.clean_gain_head = None
                self.register_parameter("clean_loss_log_vars", None)
        else:
            self.v561_global_decoder = None
            self.v561_typed_decoder = None
            self.v561_query_action_head = None
            self.v561_mask_embed = None
            self.v562_residual_head = None
            self.v562_anchor_feature_proj = None
            self.v562_anchor_pos_mlp = None
            self.v562_query_presence_head = None
            self.v565_seed_head = None
            self.v565_mask_pixel_fuse = None
            self.clean_mask_pixel_fuse = None
            self.clean_attention_precision_head = None
            self.clean_mask_precision_head = None
            self.clean_gain_head = None
            self.register_parameter("clean_loss_log_vars", None)
            self.register_parameter("v561_mask_bias", None)
        rows = max(int(math.floor(math.sqrt(self.num_slots))), 1)
        cols = int(math.ceil(float(self.num_slots) / float(rows)))
        anchors = []
        for index in range(self.num_slots):
            row = index // cols
            col = index % cols
            cx = (float(col) + 0.5) / float(cols)
            cy = (float(row) + 0.5) / float(rows)
            size = min(max(1.5 / float(max(rows, cols)), self.r47_anchor_min_size), self.r47_anchor_max_size)
            anchors.append([cx, cy, size, size])
        anchor_tensor = torch.tensor(anchors, dtype=torch.float32).clamp(1.0e-4, 1.0 - 1.0e-4)
        # Centers live in sigmoid space; sizes are normalized to the configured
        # min/max interval before conversion to logits.
        size_unit = ((anchor_tensor[:, 2:] - self.r47_anchor_min_size) /
                     max(self.r47_anchor_max_size - self.r47_anchor_min_size, 1.0e-4)).clamp(1.0e-4, 1.0 - 1.0e-4)
        base_anchor_logits = torch.cat([
            torch.logit(anchor_tensor[:, :2]),
            torch.logit(size_unit),
        ], dim=1)
        if self.r4201_clean_rootfix_enabled and (
            self.dynamic_residual_mask_r420_enabled or self._dense_set_family_enabled
        ):
            # Clean dynamic-mask path has no learned box/anchor owner.  Keep a
            # fixed compatibility grid only as a tensor fallback for code that
            # expects an anchor-shaped object; Native R4.17 peaks replace its
            # center before mask realization.
            self.register_parameter("r47_base_anchor_logits", None)
            self.register_buffer("r4201_fixed_anchor", anchor_tensor, persistent=False)
        else:
            self.r47_base_anchor_logits = nn.Parameter(base_anchor_logits)
            self.register_buffer("r4201_fixed_anchor", anchor_tensor, persistent=False)

        # R4.8 shared iterative decoder.  Every layer samples only a small
        # anchor-relative grid from the high-resolution pixel embedding, uses
        # query-to-local-token attention, updates the query, predicts a residual
        # reference-anchor delta, and re-renders the mask.  This removes the
        # broad inherited K-channel coarse-mask bias from the R4.8 final path.
        dim = self.r47_mask_dim
        # R4.21.0 keeps identical module topology across its C0-C4 ablations.
        # The overflow head is therefore constructed whenever the R4.21.0 root
        # is active, even if C0/C1/C2 leave it unused.  RNG is forked so adding
        # the dormant owner cannot shift any shared parameter initialization.
        if self.instance_valid_factorization_r4210_enabled:
            with torch.random.fork_rng(devices=[], enabled=True):
                self.r4210_overflow_gate_head = nn.Conv2d(dim, 1, 1)
            nn.init.zeros_(self.r4210_overflow_gate_head.weight)
            if self.instance_valid_decoupling_r4211_enabled:
                # R4.21.1 treats overflow as an independent Bernoulli rejection
                # variable, not a (K+1)-th symmetric identity.  Zero log-odds is
                # therefore the canonical parameter-free Bernoulli reference.
                # Geometry is fully isolated from this initial uncertainty below.
                nn.init.zeros_(self.r4210_overflow_gate_head.bias)
            else:
                # Historical R4.21.0 initialization retained for exact regression.
                nn.init.constant_(
                    self.r4210_overflow_gate_head.bias,
                    -math.log(float(max(self.num_slots, 1))),
                )
        else:
            self.r4210_overflow_gate_head = None
        if self.r4201_clean_rootfix_enabled and (
            self.dynamic_residual_mask_r420_enabled or self._dense_set_family_enabled
        ):
            # R4.20.1 uses R4.8 only as the outer set/teacher plumbing wrapper.
            # The iterative local decoder, anchor updater and DN query encoder
            # have no consumer when dynamic mask is active and DN is disabled;
            # do not instantiate dead trainable parameters.
            self.r48_context_projs = nn.ModuleList()
            self.r48_query_norm1 = nn.ModuleList()
            self.r48_query_norm2 = nn.ModuleList()
            self.r48_ffns = nn.ModuleList()
            self.r48_offset_heads = nn.ModuleList()
            self.r48_anchor_heads = nn.ModuleList()
            self.r48_mask_query_proj = None
            self.r48_dn_query_encoder = None
        else:
            self.r48_context_projs = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.r48_decoder_layers)])
            self.r48_query_norm1 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(self.r48_decoder_layers)])
            self.r48_query_norm2 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(self.r48_decoder_layers)])
            self.r48_ffns = nn.ModuleList([
                nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
                for _ in range(self.r48_decoder_layers)
            ])
            points = self.r48_local_grid_size * self.r48_local_grid_size
            self.r48_offset_heads = nn.ModuleList([nn.Linear(dim, points * 2) for _ in range(self.r48_decoder_layers)])
            self.r48_anchor_heads = nn.ModuleList([nn.Linear(dim, 4) for _ in range(self.r48_decoder_layers)])
            self.r48_mask_query_proj = nn.Linear(dim, dim, bias=False)
            self.r48_dn_query_encoder = nn.Sequential(
                nn.Linear(8, dim), nn.GELU(), nn.Linear(dim, dim)
            )

        # V552-R4.20: CondInst-style location-conditioned dynamic residual mask.
        # The residual seed supplies relative coordinates; query-conditioned
        # dynamic 1x1 convolutions generate an arbitrary full-image residual
        # mask.  There is no width/height box, ROI crop, hard support threshold,
        # or detached frontier-growth topology in this mask path.
        if self.dynamic_residual_mask_r420_enabled:
            hidden420 = self.r420_dynamic_channels
            self.r420_mask_feature_proj = nn.Sequential(
                nn.Conv2d(dim, hidden420, 1, bias=False),
                nn.GroupNorm(_group_count(hidden420), hidden420),
                nn.GELU(),
            )
            input420 = hidden420 + 2
            self.r420_dynamic_param_sizes = (
                hidden420 * input420, hidden420,
                hidden420 * hidden420, hidden420,
                hidden420, 1,
            )
            total420 = sum(self.r420_dynamic_param_sizes)
            self.r420_dynamic_controller = nn.Linear(dim, total420)
            # Stable common mask head at initialization; query-specific behavior
            # is learned through controller.weight from the first update onward.
            # Small non-zero controller weights are essential: with a zero
            # controller matrix the first mask-loss update cannot train the
            # location-conditioned query representation itself.
            nn.init.normal_(self.r420_dynamic_controller.weight, mean=0.0, std=0.01)
            with torch.no_grad():
                bias = self.r420_dynamic_controller.bias
                offset = 0
                w1n, b1n, w2n, b2n, w3n, b3n = self.r420_dynamic_param_sizes
                w1 = torch.empty(hidden420, input420, device=bias.device, dtype=bias.dtype)
                nn.init.kaiming_uniform_(w1, a=math.sqrt(5.0))
                bias[offset:offset+w1n].copy_(w1.reshape(-1)); offset += w1n
                bias[offset:offset+b1n].zero_(); offset += b1n
                w2 = torch.empty(hidden420, hidden420, device=bias.device, dtype=bias.dtype)
                nn.init.kaiming_uniform_(w2, a=math.sqrt(5.0))
                bias[offset:offset+w2n].copy_(w2.reshape(-1)); offset += w2n
                bias[offset:offset+b2n].zero_(); offset += b2n
                w3 = torch.empty(1, hidden420, device=bias.device, dtype=bias.dtype)
                nn.init.kaiming_uniform_(w3, a=math.sqrt(5.0))
                bias[offset:offset+w3n].copy_(w3.reshape(-1)); offset += w3n
                bias[offset:offset+b3n].fill_(-2.0)
        else:
            self.r420_mask_feature_proj = None
            self.r420_dynamic_controller = None
            self.r420_dynamic_param_sizes = ()

        # Scaled-cosine local attention (Swin-V2 principle) on top of the same
        # deformable samples.  Separate q/k/v projections are deliberately the
        # only new trainable attention parameters; no extra decoder branch is
        # introduced.
        if self.content_selective_r49_enabled:
            self.r49_query_key_projs = nn.ModuleList(
                [nn.Linear(dim, dim, bias=False) for _ in range(self.r48_decoder_layers)]
            )
            self.r49_sample_key_projs = nn.ModuleList(
                [nn.Linear(dim, dim, bias=False) for _ in range(self.r48_decoder_layers)]
            )
            self.r49_sample_value_projs = nn.ModuleList(
                [nn.Linear(dim, dim, bias=False) for _ in range(self.r48_decoder_layers)]
            )
            self.r49_logit_scale = nn.Parameter(
                torch.full((self.r48_decoder_layers,), math.log(init_scale), dtype=torch.float32)
            )
        else:
            # Preserve historical R4.8 parameter ownership exactly: no dormant
            # trainable R4.9 tensors are inserted into older optimizer groups.
            self.r49_query_key_projs = nn.ModuleList()
            self.r49_sample_key_projs = nn.ModuleList()
            self.r49_sample_value_projs = nn.ModuleList()
            self.register_parameter("r49_logit_scale", None)

        if self.native_residual_set_r411_enabled:
            proposal_groups = _group_count(dim)
            self.r411_proposal_stem = nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1, bias=False),
                nn.GroupNorm(proposal_groups, dim),
                nn.GELU(),
            )
            self.r411_center_head = nn.Conv2d(dim, 4, 1)
            clean_boxfree = self.r4201_clean_rootfix_enabled and self.box_free_mask_set_r418_enabled
            self.r411_size_head = None if clean_boxfree else nn.Conv2d(dim, 8, 1)
            self.r411_offset_head = None if clean_boxfree else nn.Conv2d(dim, 8, 1)
            if self.proposal_recovery_r417_enabled:
                self.r417_location_head = nn.Conv2d(dim, 1, 1)
                self.r417_location_offset_head = None if clean_boxfree else nn.Conv2d(dim, 2, 1)
            else:
                self.r417_location_head = None
                self.r417_location_offset_head = None
            # V552-R4.20.4: R4.17 location logits are trained against Gaussian
            # component *centres*, not dense residual occupancy.  R4.20.3 used
            # that centre heatmap as a residual-vs-background prior, creating a
            # target/representation mismatch.  R4.20.4 adds one dedicated dense
            # residual-occupancy head supervised by the union of the exact
            # residual-component Teacher masks.  This head answers only
            # "is this pixel residual?"; slots answer the separate conditional
            # identity question.
            if self.factorized_residual_existence_identity_r4204_enabled:
                # Construct the new owner under an RNG fork so adding R4.20.4
                # cannot shift initialization of any later shared M1/M2 module
                # in same-seed A0/A1/A2 ablations.  Use PyTorch's standard
                # Conv2d initialization; there is no hand-tuned prior bias.
                with torch.random.fork_rng(devices=[], enabled=True):
                    self.r4204_residual_occupancy_head = nn.Conv2d(dim, 1, 1)
            else:
                self.r4204_residual_occupancy_head = None
            self.r411_type_embedding = (
                None
                if (
                    clean_boxfree
                    and (
                        self.r420_type_decoupled_mask_enabled
                        or self._dense_set_family_enabled
                    )
                )
                else nn.Embedding(4, dim)
            )
            self.r411_box_query_mlp = None if clean_boxfree else nn.Sequential(
                nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim)
            )
            roi_hidden = max(dim // 2, 16)
            self.r411_roi_mask_head = None if clean_boxfree else nn.Sequential(
                nn.Conv2d(dim * 2, dim, 3, padding=1, bias=False),
                nn.GroupNorm(proposal_groups, dim),
                nn.GELU(),
                nn.Conv2d(dim, roi_hidden, 3, padding=1, bias=False),
                nn.GroupNorm(_group_count(roi_hidden), roi_hidden),
                nn.GELU(),
                nn.Conv2d(roi_hidden, 1, 1),
            )
            nn.init.normal_(self.r411_center_head.weight, mean=0.0, std=0.01)
            nn.init.constant_(self.r411_center_head.bias, -3.0)
            if self.r411_size_head is not None:
                nn.init.zeros_(self.r411_size_head.weight)
                size_unit = (self.r411_initial_box_size - self.r47_anchor_min_size) / max(
                    self.r47_anchor_max_size - self.r47_anchor_min_size, 1.0e-4
                )
                size_unit = min(max(size_unit, 1.0e-4), 1.0 - 1.0e-4)
                nn.init.constant_(self.r411_size_head.bias, float(torch.logit(torch.tensor(size_unit)).item()))
            if self.r411_offset_head is not None:
                nn.init.zeros_(self.r411_offset_head.weight)
                nn.init.zeros_(self.r411_offset_head.bias)
            if self.r417_location_head is not None:
                nn.init.normal_(self.r417_location_head.weight, mean=0.0, std=0.01)
                nn.init.constant_(self.r417_location_head.bias, -3.0)
                if self.r417_location_offset_head is not None:
                    nn.init.zeros_(self.r417_location_offset_head.weight)
                    nn.init.zeros_(self.r417_location_offset_head.bias)
            if self.r411_type_embedding is not None:
                nn.init.normal_(self.r411_type_embedding.weight, mean=0.0, std=0.02)
            if self.r411_roi_mask_head is not None:
                nn.init.zeros_(self.r411_roi_mask_head[-1].weight)
                nn.init.constant_(self.r411_roi_mask_head[-1].bias, -2.0)
            if self.box_free_mask_set_r418_enabled:
                # R4.18 has one geometry owner: the predicted mask itself.
                # Freeze every historical box/offset/ROI parameter so no
                # optimizer step can silently re-introduce the failed path.
                for module in (
                    self.r411_size_head,
                    self.r411_offset_head,
                    self.r417_location_offset_head,
                    self.r411_box_query_mlp,
                    self.r411_roi_mask_head,
                ):
                    if module is not None:
                        for parameter in module.parameters():
                            parameter.requires_grad_(False)
                if self.r47_base_anchor_logits is not None:
                    self.r47_base_anchor_logits.requires_grad_(False)
                for head in self.r48_anchor_heads:
                    for parameter in head.parameters():
                        parameter.requires_grad_(False)
        else:
            self.r411_proposal_stem = None
            self.r411_center_head = None
            self.r411_size_head = None
            self.r411_offset_head = None
            self.r417_location_head = None
            self.r417_location_offset_head = None
            self.r4204_residual_occupancy_head = None
            self.r411_type_embedding = None
            self.r411_box_query_mlp = None
            self.r411_roi_mask_head = None

        if self.geometry_lock_r413_enabled:
            if not self.native_residual_set_r411_enabled:
                raise ValueError("V552-R4.13 requires the R4.11 typed locator")
            # R4.13 used [center feature, type, slot query].  R4.14 removes
            # slot-rank identity and replaces the single center sample with
            # directional local evidence: center/left/right/top/bottom + type.
            extent_input_dim = dim * (6 if self.geometry_context_r414_enabled else 3)
            extent_dim = 4 if self.r416_asymmetric_ltrb_enabled else 2
            self.r413_extent_head = nn.Sequential(
                nn.Linear(extent_input_dim, dim), nn.GELU(), nn.Linear(dim, extent_dim)
            )
            nn.init.normal_(self.r413_extent_head[-1].weight, mean=0.0, std=0.01)
            if self.r416_asymmetric_ltrb_enabled:
                # Signed edge offsets relative to the selected physical point:
                # [x1-px, y1-py, x2-px, y2-py].  Initial prediction is the
                # historical symmetric initial box, but subsequent regression
                # is free to be asymmetric and exactly compensate point error.
                edge_scale = max(self.r47_anchor_max_size, 1.0e-3)
                half = 0.5 * self.r411_initial_box_size
                unit = min(max(half / edge_scale, 1.0e-4), 0.95)
                raw = float(torch.atanh(torch.tensor(unit)).item())
                bias = torch.tensor([-raw, -raw, raw, raw], dtype=self.r413_extent_head[-1].bias.dtype)
                with torch.no_grad():
                    self.r413_extent_head[-1].bias.copy_(bias)
            else:
                log_min = math.log(max(self.r47_anchor_min_size, 1.0e-6))
                log_max = math.log(max(self.r47_anchor_max_size, self.r47_anchor_min_size + 1.0e-6))
                init_log = math.log(max(self.r411_initial_box_size, self.r47_anchor_min_size))
                init_unit = (init_log - log_min) / max(log_max - log_min, 1.0e-6)
                init_unit = min(max(init_unit, 1.0e-4), 1.0 - 1.0e-4)
                nn.init.constant_(self.r413_extent_head[-1].bias, float(torch.logit(torch.tensor(init_unit)).item()))
            if self.r411_size_head is not None:
                for parameter in self.r411_size_head.parameters():
                    parameter.requires_grad_(False)
            for head in self.r48_anchor_heads:
                for parameter in head.parameters():
                    parameter.requires_grad_(False)
        else:
            self.r413_extent_head = None

        if self.canonical_shape_r412_enabled:
            if not self.native_residual_set_r411_enabled:
                raise ValueError("V552-R4.12 requires the R4.11 typed Native-residual locator")
            # Inputs: sampled pixel embedding, query map, canonical (u,v), and
            # one Base-derived action-admissibility support channel.
            r412_in = dim * 2 + 3
            self.r412_roi_stem = nn.Sequential(
                nn.Conv2d(r412_in, dim, 3, padding=1, bias=False),
                nn.GroupNorm(_group_count(dim), dim),
                nn.GELU(),
            )
            self.r412_roi_blocks = nn.Sequential(
                _R412ResidualBlock(dim),
                _R412ResidualBlock(dim),
                _R412ResidualBlock(dim),
            )
            self.r412_coarse_head = nn.Conv2d(dim, 1, 1)
            self.r412_refine_head = nn.Sequential(
                nn.Conv2d(dim + 1, dim, 3, padding=1, bias=False),
                nn.GroupNorm(_group_count(dim), dim),
                nn.GELU(),
                nn.Conv2d(dim, 1, 1),
            )
            # Small non-zero output weights are deliberate: zeroing the final
            # conv would block all first-step gradients into the canonical
            # feature stem/blocks, recreating the early capability bottleneck.
            nn.init.normal_(self.r412_coarse_head.weight, mean=0.0, std=0.01)
            nn.init.constant_(self.r412_coarse_head.bias, -1.5)
            nn.init.normal_(self.r412_refine_head[-1].weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.r412_refine_head[-1].bias)
        else:
            self.r412_roi_stem = None
            self.r412_roi_blocks = None
            self.r412_coarse_head = None
            self.r412_refine_head = None

        grid_axis = torch.linspace(-1.0, 1.0, self.r48_local_grid_size)
        gy, gx = torch.meshgrid(grid_axis, grid_axis, indexing="ij")
        self.register_buffer(
            "r48_base_sample_offsets",
            torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1),
            persistent=False,
        )
        if self.geometry_context_r414_enabled:
            axis414 = torch.linspace(
                -self.r414_context_radius,
                self.r414_context_radius,
                self.r414_context_grid_size,
            )
            gy414, gx414 = torch.meshgrid(axis414, axis414, indexing="ij")
            self.register_buffer(
                "r414_context_offsets",
                torch.stack([gx414, gy414], dim=-1),
                persistent=False,
            )
        else:
            self.register_buffer(
                "r414_context_offsets",
                torch.zeros(1, 1, 2),
                persistent=False,
            )

        self.pyramid_blocks = nn.ModuleList(
            [_PyramidBlock(channels, float(kwargs.get("dropout", 0.10))) for _ in self.pyramid_scales]
        )
        if self.r4201_clean_rootfix_enabled and self.dynamic_residual_mask_r420_enabled:
            # The R4.20 dynamic head owns residual masks.  The inherited V538
            # parent-mask renderers and V551 scale-mask heads are not read by
            # the clean forward graph, so remove their parameters instead of
            # carrying dead optimizer/state-dict baggage.
            self.mask_encoder = None
            self.mask_head = None
            self.scale_mask_heads = nn.ModuleList()
        else:
            # Historical path retained unchanged for reproducibility.
            self.scale_mask_heads = nn.ModuleList(
                [nn.Conv2d(channels, self.num_slots, 1) for _ in self.pyramid_scales[1:]]
            )
        self.scale_router = nn.Linear(channels, self.num_slots * self.num_scales)
        self.pyramid_fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.GELU(),
        )

        self.boundary_residual_head = nn.Conv2d(channels, self.num_slots, 1)
        self.local_residual_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.GELU(),
            nn.Dropout2d(float(editor_residual_dropout)),
            nn.Conv2d(channels, self.num_slots, 1),
        )

        # R4.5 Error-Prone Region head.  It consumes only inference-visible
        # evidence: fused image/semantic features, Base probability, entropy,
        # boundary evidence and aggregate causal evidence.  GT is used only in
        # the loss to supervise this map.
        error_hidden = max(channels // 2, 16)
        self.error_prone_head = nn.Sequential(
            nn.Conv2d(channels + 4, error_hidden, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(error_hidden), error_hidden),
            nn.GELU(),
            nn.Conv2d(error_hidden, 1, 1),
        )

        selector_dim = int(self.m2_selector_trunk[-2].out_features)
        # V552-R4 does not add M1 logits directly to the Editor decision. M1
        # action probabilities/dose/presence are context features, while one
        # standalone route head owns the final Preserve/Action decision.
        self.editor_route_context_adapter = nn.Sequential(
            nn.Linear(selector_dim + 6, selector_dim),
            nn.GELU(),
            nn.Linear(selector_dim, selector_dim),
        )
        self.editor_route_head = nn.Linear(selector_dim, 5)
        self.editor_dose_adjust_head = nn.Linear(selector_dim, 1)
        self.editor_relative_gain_head = nn.Linear(selector_dim, 1)
        self.editor_safety_outcome_head = nn.Linear(selector_dim, 3)
        # R4.5 factorizes Safety into two independent questions instead of a
        # competing Neutral/Benefit/Harm softmax: is this route beneficial,
        # and is this route harmful?
        self.editor_safety_benefit_head = nn.Linear(selector_dim, 1)
        self.editor_safety_harm_head = nn.Linear(selector_dim, 1)
        # Every exact route receives its own feature token. The same token is
        # consumed by a relative-to-M1 Safety Critic and an absolute-to-Base
        # Utility Critic, eliminating the R3 reference-frame contradiction.
        self.route_feature_adapter = nn.Sequential(
            nn.Linear(selector_dim + 11, selector_dim),
            nn.GELU(),
            nn.Linear(selector_dim, selector_dim),
        )
        self.safety_route_feature_adapter = nn.Sequential(
            nn.Linear(selector_dim + 11, selector_dim),
            nn.GELU(),
            nn.Linear(selector_dim, selector_dim),
        )
        self.utility_route_feature_adapter = nn.Sequential(
            nn.Linear(selector_dim + 11, selector_dim),
            nn.GELU(),
            nn.Linear(selector_dim, selector_dim),
        )
        route_visual_dim = channels * 2 + 4
        self.safety_route_visual_adapter = nn.Sequential(
            nn.Linear(route_visual_dim, selector_dim),
            nn.GELU(),
            nn.Linear(selector_dim, selector_dim),
        )
        self.utility_route_visual_adapter = nn.Sequential(
            nn.Linear(route_visual_dim, selector_dim),
            nn.GELU(),
            nn.Linear(selector_dim, selector_dim),
        )
        # R4.4 value-only routes are intentionally separate from the class
        # routes.  Their inputs are detached from the class representation.
        self.safety_value_feature_adapter = nn.Sequential(
            nn.Linear(selector_dim + 11, selector_dim),
            nn.GELU(),
            nn.Linear(selector_dim, selector_dim),
        )
        self.utility_value_feature_adapter = nn.Sequential(
            nn.Linear(selector_dim + 11, selector_dim),
            nn.GELU(),
            nn.Linear(selector_dim, selector_dim),
        )
        self.safety_value_visual_adapter = nn.Sequential(
            nn.Linear(route_visual_dim, selector_dim),
            nn.GELU(),
            nn.Linear(selector_dim, selector_dim),
        )
        self.utility_value_visual_adapter = nn.Sequential(
            nn.Linear(route_visual_dim, selector_dim),
            nn.GELU(),
            nn.Linear(selector_dim, selector_dim),
        )
        self.candidate_utility_outcome_head = nn.Linear(selector_dim, 3)
        self.candidate_absolute_gain_head = nn.Linear(selector_dim, 1)
        self.editor_benefit_magnitude_head = nn.Linear(selector_dim, 1)
        self.editor_harm_magnitude_head = nn.Linear(selector_dim, 1)
        self.candidate_benefit_magnitude_head = nn.Linear(selector_dim, 1)
        self.candidate_harm_magnitude_head = nn.Linear(selector_dim, 1)
        # Learned atom quality is a deployment-only estimate. Physical and M1
        # supervision validity never depend on this head.
        self.atom_quality_head = nn.Linear(selector_dim, 1)
        # Historical R2 heads are retained for old configurations/checkpoints.
        self.r2_outcome_head = nn.Linear(selector_dim, 3)
        self.r2_benefit_magnitude_head = nn.Linear(selector_dim, 1)
        self.r2_harm_magnitude_head = nn.Linear(selector_dim, 1)
        # Single-pass outcome refinement.  The old implementation spatially
        # encoded every edited dense candidate a second time.  This residual
        # adapter conditions the same calibrated outcome heads on the actual
        # edit route/dose/region statistics while exact final candidates remain
        # supervised by V551 loss.
        self.editor_feature_adapter = nn.Sequential(
            nn.Linear(selector_dim + 10, selector_dim),
            nn.GELU(),
            nn.Linear(selector_dim, selector_dim),
        )
        # Context-conditioned set composer.  Ten scalar statistics describe
        # the interaction between an edited atom and the current composition:
        # overlap, action conflict, new area, current local confidence, absolute
        # and signed delta, conservative gain, Benefit-Harm margin, presence,
        # and the current composition step.
        state_hidden = max(selector_dim // 2, 16)
        self.composer_state_encoder = nn.Sequential(
            nn.Conv2d(4, state_hidden, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(state_hidden), state_hidden),
            nn.GELU(),
            nn.Conv2d(state_hidden, selector_dim, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(selector_dim), selector_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.composer_candidate_adapter = nn.Sequential(
            nn.Linear(selector_dim * 2 + 10, selector_dim),
            nn.GELU(),
            nn.Linear(selector_dim, selector_dim),
        )
        self.composer_marginal_head = nn.Linear(selector_dim, 1)
        stop_hidden = max(selector_dim // 2, 16)
        self.composer_stop_head = nn.Sequential(
            nn.Linear(selector_dim * 2 + 4, stop_hidden),
            nn.GELU(),
            nn.Linear(stop_hidden, 1),
        )

        queue_shape = (3, self.critic_queue_capacity, selector_dim)
        gain_shape = (3, self.critic_queue_capacity)
        count_shape = (3,)
        self.register_buffer(
            "_v552r4_safety_queue_features",
            torch.zeros(queue_shape),
            persistent=False,
        )
        self.register_buffer(
            "_v552r4_safety_queue_gains",
            torch.zeros(gain_shape),
            persistent=False,
        )
        self.register_buffer(
            "_v552r4_safety_queue_count",
            torch.zeros(count_shape, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_v552r4_safety_queue_ptr",
            torch.zeros(count_shape, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_v552r4_utility_queue_features",
            torch.zeros(queue_shape),
            persistent=False,
        )
        self.register_buffer(
            "_v552r4_utility_queue_gains",
            torch.zeros(gain_shape),
            persistent=False,
        )
        self.register_buffer(
            "_v552r4_utility_queue_count",
            torch.zeros(count_shape, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_v552r4_utility_queue_ptr",
            torch.zeros(count_shape, dtype=torch.long),
            persistent=False,
        )

        for head in self.scale_mask_heads:
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, -4.0)
        nn.init.zeros_(self.scale_router.weight)
        nn.init.zeros_(self.scale_router.bias)
        nn.init.zeros_(self.boundary_residual_head.weight)
        nn.init.zeros_(self.boundary_residual_head.bias)
        nn.init.zeros_(self.local_residual_head[-1].weight)
        nn.init.zeros_(self.local_residual_head[-1].bias)
        nn.init.zeros_(self.editor_route_context_adapter[-1].weight)
        nn.init.zeros_(self.editor_route_context_adapter[-1].bias)
        nn.init.zeros_(self.editor_route_head.weight)
        if self.teacher_decoupled_r2_enabled:
            nn.init.constant_(
                self.editor_route_head.bias, -float(editor_preserve_bias)
            )
        else:
            nn.init.zeros_(self.editor_route_head.bias)
        nn.init.constant_(self.editor_route_head.bias[0], float(editor_preserve_bias))
        nn.init.zeros_(self.editor_dose_adjust_head.weight)
        nn.init.zeros_(self.editor_relative_gain_head.weight)
        nn.init.zeros_(self.editor_relative_gain_head.bias)
        nn.init.zeros_(self.editor_safety_outcome_head.weight)
        nn.init.zeros_(self.editor_safety_outcome_head.bias)
        nn.init.zeros_(self.editor_safety_benefit_head.weight)
        nn.init.zeros_(self.editor_safety_benefit_head.bias)
        nn.init.zeros_(self.editor_safety_harm_head.weight)
        nn.init.zeros_(self.editor_safety_harm_head.bias)
        nn.init.zeros_(self.error_prone_head[-1].weight)
        nn.init.zeros_(self.error_prone_head[-1].bias)
        nn.init.zeros_(self.route_feature_adapter[-1].weight)
        nn.init.zeros_(self.route_feature_adapter[-1].bias)
        nn.init.zeros_(self.safety_route_feature_adapter[-1].weight)
        nn.init.zeros_(self.safety_route_feature_adapter[-1].bias)
        nn.init.zeros_(self.utility_route_feature_adapter[-1].weight)
        nn.init.zeros_(self.utility_route_feature_adapter[-1].bias)
        for adapter in (
            self.safety_route_visual_adapter,
            self.utility_route_visual_adapter,
            self.safety_value_feature_adapter,
            self.utility_value_feature_adapter,
            self.safety_value_visual_adapter,
            self.utility_value_visual_adapter,
        ):
            nn.init.zeros_(adapter[-1].weight)
            nn.init.zeros_(adapter[-1].bias)
        nn.init.zeros_(self.candidate_utility_outcome_head.weight)
        nn.init.zeros_(self.candidate_utility_outcome_head.bias)
        nn.init.zeros_(self.candidate_absolute_gain_head.weight)
        nn.init.zeros_(self.candidate_absolute_gain_head.bias)
        if self.direct_signed_utility_r45_enabled:
            nn.init.normal_(self.candidate_absolute_gain_head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.candidate_absolute_gain_head.bias)
        if self.spatial_evidence_r43_enabled:
            # Exact-zero classifier and adapter matrices cause a two-step cold
            # start: the classifier initially sees identical route tokens and its
            # zero weight blocks gradients into the route adapter. A small,
            # symmetric random initialization opens route-specific gradients
            # without introducing a Benefit/Harm prior.
            modules = [
                self.editor_safety_outcome_head,
                self.candidate_utility_outcome_head,
                self.safety_route_feature_adapter[-1],
                self.utility_route_feature_adapter[-1],
                self.safety_route_visual_adapter[-1],
                self.utility_route_visual_adapter[-1],
            ]
            if self.factorized_safety_r45_enabled:
                modules.extend([
                    self.editor_safety_benefit_head,
                    self.editor_safety_harm_head,
                ])
            if self.error_aware_r45_enabled and not self.native_contract_r46_enabled:
                modules.extend([
                    self.error_prone_head[-1],
                ])
            if self.class_value_decoupling_r44_enabled:
                modules.extend([
                    self.safety_value_feature_adapter[-1],
                    self.utility_value_feature_adapter[-1],
                    self.safety_value_visual_adapter[-1],
                    self.utility_value_visual_adapter[-1],
                ])
            for module in modules:
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        for magnitude_head in (
            self.editor_benefit_magnitude_head,
            self.editor_harm_magnitude_head,
            self.candidate_benefit_magnitude_head,
            self.candidate_harm_magnitude_head,
        ):
            if self.class_value_decoupling_r44_enabled:
                # The value adapter is independent from the class path.  A
                # zero magnitude projection would block its first-step
                # gradient, recreating the two-step cold start fixed for the
                # class adapter in R4.3.  Symmetric small weights open value
                # gradients without preferring Benefit or Harm.
                nn.init.normal_(magnitude_head.weight, mean=0.0, std=0.01)
            else:
                nn.init.zeros_(magnitude_head.weight)
            # sigmoid(-4) * 0.05 ~= 9e-4, close to the minimum clinically
            # meaningful Dice gain instead of R4.1's 0.025 initial magnitude.
            nn.init.constant_(magnitude_head.bias, -4.0)
        nn.init.zeros_(self.atom_quality_head.weight)
        nn.init.zeros_(self.atom_quality_head.bias)
        nn.init.zeros_(self.r2_outcome_head.weight)
        nn.init.zeros_(self.r2_outcome_head.bias)
        nn.init.zeros_(self.r2_benefit_magnitude_head.weight)
        nn.init.constant_(self.r2_benefit_magnitude_head.bias, -4.0)
        nn.init.zeros_(self.r2_harm_magnitude_head.weight)
        nn.init.constant_(self.r2_harm_magnitude_head.bias, -4.0)
        nn.init.zeros_(self.editor_feature_adapter[-1].weight)
        nn.init.zeros_(self.editor_feature_adapter[-1].bias)
        # Start from the calibrated single-atom critic.  The learned marginal
        # correction is initially zero and Stop has a small conservative bias.
        nn.init.zeros_(self.composer_candidate_adapter[-1].weight)
        nn.init.zeros_(self.composer_candidate_adapter[-1].bias)
        nn.init.zeros_(self.composer_marginal_head.weight)
        nn.init.zeros_(self.composer_marginal_head.bias)
        nn.init.zeros_(self.composer_stop_head[-1].weight)
        nn.init.zeros_(self.composer_stop_head[-1].bias)
        initial_adjust = 1.0
        adjust_fraction = (initial_adjust - self.editor_dose_adjust_min) / (
            self.editor_dose_adjust_max - self.editor_dose_adjust_min
        )
        adjust_fraction = min(max(adjust_fraction, 1.0e-5), 1.0 - 1.0e-5)
        nn.init.constant_(
            self.editor_dose_adjust_head.bias,
            float(torch.logit(torch.tensor(adjust_fraction)).item()),
        )

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = max(int(epoch), 0)

    def _build_pyramid(
        self, feature: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        h, w = feature.shape[-2:]
        native: List[torch.Tensor] = []
        upsampled: List[torch.Tensor] = []
        for factor, block in zip(self.pyramid_scales, self.pyramid_blocks):
            if factor == 1:
                level = feature
            else:
                target = (max(h // factor, 1), max(w // factor, 1))
                level = F.adaptive_avg_pool2d(feature, target)
            level = block(level)
            native.append(level)
            upsampled.append(
                level
                if level.shape[-2:] == (h, w)
                else F.interpolate(level, size=(h, w), mode="bilinear", align_corners=False)
            )
        return native, upsampled

    def _multiscale_masks(
        self, feature: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, _, h, w = feature.shape
        native, upsampled = self._build_pyramid(feature)
        pooled = F.adaptive_avg_pool2d(feature, 1).flatten(1)
        scale_logits = self.scale_router(pooled).reshape(
            b, self.num_slots, self.num_scales
        )
        scale_probs = F.softmax(scale_logits / self.scale_temperature, dim=2)

        mean_scale_route = scale_probs.mean(dim=1)  # [B,L]
        pyramid_context = sum(
            level * mean_scale_route[:, idx, None, None, None]
            for idx, level in enumerate(upsampled)
        )
        fused_feature = self.pyramid_fuse(torch.cat([feature, pyramid_context], dim=1))

        if self.r4201_clean_rootfix_enabled and self.dynamic_residual_mask_r420_enabled:
            # The historical parent mask is overwritten by R4.20 before any
            # downstream descriptor/candidate computation.  Keep multiscale
            # feature routing/fusion, but do not execute dead parent mask
            # renderers in the clean graph.
            combined_logits = feature.new_zeros((b, self.num_slots, h, w))
            smooth_masks = feature.new_zeros((b, self.num_slots, h, w))
            return combined_logits, smooth_masks, scale_logits, scale_probs, fused_feature

        scale_mask_logits = []
        for index, level in enumerate(native):
            if index == 0:
                logits = self.mask_head(self.mask_encoder(feature))
            else:
                logits = self.scale_mask_heads[index - 1](level)
            if logits.shape[-2:] != (h, w):
                logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
            scale_mask_logits.append(logits)
        stacked_logits = torch.stack(scale_mask_logits, dim=2)  # [B,K,L,H,W]
        hard_index = scale_probs.detach().argmax(dim=2)
        hard_one_hot = F.one_hot(hard_index, num_classes=self.num_scales).to(
            scale_probs.dtype
        )
        scale_st = hard_one_hot + scale_probs - scale_probs.detach()
        combined_logits = (stacked_logits * scale_st[..., None, None]).sum(dim=2)
        raw_masks = torch.sigmoid(combined_logits / self.mask_temperature)
        smooth_masks = F.avg_pool2d(
            raw_masks.reshape(b * self.num_slots, 1, h, w),
            kernel_size=5,
            stride=1,
            padding=2,
        ).reshape(b, self.num_slots, h, w)
        return combined_logits, smooth_masks, scale_logits, scale_probs, fused_feature

    def _r47_query_conditioned_masks(
        self,
        *,
        coarse_logits: torch.Tensor,
        fused_feature: torch.Tensor,
        base_probability: torch.Tensor,
        cause_probability: torch.Tensor,
        action_alpha: torch.Tensor,
        entropy: torch.Tensor,
        boundary: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """R4.7 Mask-DINO/PointRend-inspired spatial realization branch.

        Forward uses no GT.  Each learned slot query predicts one high-
        resolution residual-component mask by dot-product with a pixel
        embedding map.  A trainable grid anchor stabilizes query identity; a
        differentiable center/extent derived from the provisional mask then
        supplies a second spatial prior.
        """
        b, _, h, w = coarse_logits.shape
        cause = cause_probability
        if cause.shape[1] < 4:
            cause = F.pad(cause, (0, 0, 0, 0, 0, 4 - cause.shape[1]))
        cause = cause[:, :4].detach()
        alpha = action_alpha
        if alpha.shape[1] < 4:
            alpha = F.pad(alpha, (0, 0, 0, 0, 0, 4 - alpha.shape[1]))
        alpha = alpha[:, :4].detach()
        yy = torch.linspace(0.0, 1.0, h, device=coarse_logits.device, dtype=coarse_logits.dtype)
        xx = torch.linspace(0.0, 1.0, w, device=coarse_logits.device, dtype=coarse_logits.dtype)
        y_grid = yy[:, None].expand(h, w)
        x_grid = xx[None, :].expand(h, w)
        xy = torch.stack([x_grid, y_grid], dim=0)[None].expand(b, -1, -1, -1)
        pixel_input = torch.cat(
            [
                fused_feature,
                base_probability[:, :1].detach(),
                entropy[:, :1].detach(),
                boundary[:, :1].detach(),
                cause,
                alpha,
                xy,
            ],
            dim=1,
        )
        pixel_embedding = F.normalize(self.r47_pixel_encoder(pixel_input), dim=1)
        queries = F.normalize(self.r47_slot_queries.weight, dim=1)
        query_logits = torch.einsum("kd,bdhw->bkhw", queries, pixel_embedding)
        query_logits = query_logits * self.r47_query_logit_scale

        base_anchor_raw = torch.sigmoid(self.r47_base_anchor_logits)
        base_center = base_anchor_raw[:, :2]
        base_size = self.r47_anchor_min_size + (
            self.r47_anchor_max_size - self.r47_anchor_min_size
        ) * base_anchor_raw[:, 2:]
        base_anchor = torch.cat([base_center, base_size], dim=1)[None].expand(b, -1, -1)

        def anchor_prior(anchor: torch.Tensor) -> torch.Tensor:
            cx = anchor[:, :, 0, None, None]
            cy = anchor[:, :, 1, None, None]
            sx = (0.5 * anchor[:, :, 2, None, None]).clamp_min(0.02)
            sy = (0.5 * anchor[:, :, 3, None, None]).clamp_min(0.02)
            distance = ((x_grid[None, None] - cx) / sx).square() + (
                (y_grid[None, None] - cy) / sy
            ).square()
            return self.r47_anchor_prior_scale * (1.0 - 0.5 * distance).clamp(-4.0, 1.0)

        provisional_logits = coarse_logits + query_logits + anchor_prior(base_anchor)
        provisional = torch.sigmoid(provisional_logits / self.mask_temperature)
        mass = provisional.sum(dim=(-2, -1)).clamp_min(EPS)
        dynamic_cx = (provisional * x_grid[None, None]).sum(dim=(-2, -1)) / mass
        dynamic_cy = (provisional * y_grid[None, None]).sum(dim=(-2, -1)) / mass
        var_x = (
            provisional * (x_grid[None, None] - dynamic_cx[:, :, None, None]).square()
        ).sum(dim=(-2, -1)) / mass
        var_y = (
            provisional * (y_grid[None, None] - dynamic_cy[:, :, None, None]).square()
        ).sum(dim=(-2, -1)) / mass
        dynamic_w = (4.0 * torch.sqrt(var_x.clamp_min(EPS))).clamp(
            self.r47_anchor_min_size, self.r47_anchor_max_size
        )
        dynamic_h = (4.0 * torch.sqrt(var_y.clamp_min(EPS))).clamp(
            self.r47_anchor_min_size, self.r47_anchor_max_size
        )
        dynamic_anchor = torch.stack(
            [dynamic_cx, dynamic_cy, dynamic_w, dynamic_h], dim=2
        )
        final_logits = provisional_logits + 0.5 * anchor_prior(dynamic_anchor)
        raw_masks = torch.sigmoid(final_logits / self.mask_temperature)
        smooth_masks = F.avg_pool2d(
            raw_masks.reshape(b * self.num_slots, 1, h, w),
            kernel_size=3,
            stride=1,
            padding=1,
        ).reshape(b, self.num_slots, h, w)
        return final_logits, smooth_masks, dynamic_anchor, query_logits

    @staticmethod
    def _r48_geometry_from_masks(masks: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Return normalized cx,cy,w,h for binary/soft component masks."""
        b, k, h, w = masks.shape
        yy = torch.linspace(0.0, 1.0, h, device=masks.device, dtype=masks.dtype)
        xx = torch.linspace(0.0, 1.0, w, device=masks.device, dtype=masks.dtype)
        mask = masks > 0.5
        x_map = xx[None, None, None, :].expand(b, k, h, w)
        y_map = yy[None, None, :, None].expand(b, k, h, w)
        xmin = torch.where(mask, x_map, x_map.new_ones(())).amin(dim=(-2, -1))
        xmax = torch.where(mask, x_map, x_map.new_zeros(())).amax(dim=(-2, -1))
        ymin = torch.where(mask, y_map, y_map.new_ones(())).amin(dim=(-2, -1))
        ymax = torch.where(mask, y_map, y_map.new_zeros(())).amax(dim=(-2, -1))
        geometry = torch.stack(
            [
                0.5 * (xmin + xmax),
                0.5 * (ymin + ymax),
                (xmax - xmin + 1.0 / float(max(w, 1))).clamp(0.0, 1.0),
                (ymax - ymin + 1.0 / float(max(h, 1))).clamp(0.0, 1.0),
            ],
            dim=2,
        )
        return torch.where(valid[:, :, None], geometry, geometry.new_zeros(geometry.shape))

    def _r48_window_prior(
        self,
        anchor: torch.Tensor,
        x_grid: torch.Tensor,
        y_grid: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable rectangular locality prior around each reference box."""
        cx = anchor[:, :, 0, None, None]
        cy = anchor[:, :, 1, None, None]
        half_w = (0.5 * anchor[:, :, 2, None, None]).clamp_min(0.01)
        half_h = (0.5 * anchor[:, :, 3, None, None]).clamp_min(0.01)
        tau = self.r48_window_temperature
        inside_x = torch.sigmoid((half_w - (x_grid[None, None] - cx).abs()) / tau)
        inside_y = torch.sigmoid((half_h - (y_grid[None, None] - cy).abs()) / tau)
        window = (inside_x * inside_y).clamp(1.0e-4, 1.0 - 1.0e-4)
        return self.r48_window_prior_scale * torch.logit(window).clamp(-8.0, 4.0)

    def _r410_support_penalty(
        self,
        anchor: torch.Tensor,
        x_grid: torch.Tensor,
        y_grid: torch.Tensor,
    ) -> torch.Tensor:
        """Support-only locality: zero inside, negative outside.

        Unlike R4.8's rectangular foreground prior this term can never make a
        pixel more positive.  The reference box only declares a generously
        expanded valid reconstruction domain; component shape is still decided
        entirely by image/residual content.
        """
        if self.r410_support_max_penalty <= 0.0:
            return anchor.new_zeros((anchor.shape[0], anchor.shape[1], x_grid.shape[0], x_grid.shape[1]))
        cx = anchor[:, :, 0, None, None]
        cy = anchor[:, :, 1, None, None]
        half_w = (0.5 * self.r410_support_expand * anchor[:, :, 2, None, None]).clamp_min(0.02)
        half_h = (0.5 * self.r410_support_expand * anchor[:, :, 3, None, None]).clamp_min(0.02)
        tau = self.r410_support_temperature
        in_x = torch.sigmoid((half_w - (x_grid[None, None] - cx).abs()) / tau)
        in_y = torch.sigmoid((half_h - (y_grid[None, None] - cy).abs()) / tau)
        support = (in_x * in_y).clamp_min(math.exp(-self.r410_support_max_penalty))
        # log(support) <= 0: locality can only suppress, never hallucinate FG.
        return support.log().clamp(min=-self.r410_support_max_penalty, max=0.0)

    def _r410_proposal_initial_state(
        self,
        *,
        pixel_embedding: torch.Tensor,
        cause: torch.Tensor,
        alpha: torch.Tensor,
        learned_query: torch.Tensor,
        learned_anchor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialize queries from inference-visible typed residual peaks.

        This is a deterministic two-stage proposal mechanism, not a new
        predictor.  It reuses the already-supervised Delete/Fill/Trim/Expand
        maps, performs GPU NMS, and uses the selected pixel embedding to bind
        each learned query to image content.  No GT is read here.
        """
        b, d, h, w = pixel_embedding.shape
        k = learned_query.shape[1]
        evidence = (cause.clamp(0.0, 1.0) * (0.5 + 0.5 * alpha.clamp(0.0, 1.0))).amax(dim=1)
        work = evidence.detach().clone()
        centers = []
        scores = []
        radius = self.r410_proposal_nms_kernel // 2
        yy = torch.arange(h, device=work.device)[None, :, None]
        xx = torch.arange(w, device=work.device)[None, None, :]
        for _ in range(k):
            flat = work.flatten(1)
            value, index = flat.max(dim=1)
            py = torch.div(index, w, rounding_mode='floor')
            px = index.remainder(w)
            centers.append(torch.stack([
                (px.to(work.dtype) + 0.5) / float(max(w, 1)),
                (py.to(work.dtype) + 0.5) / float(max(h, 1)),
            ], dim=1))
            scores.append(value)
            suppress = (
                ((yy - py[:, None, None]).abs() <= radius)
                & ((xx - px[:, None, None]).abs() <= radius)
            )
            work = work.masked_fill(suppress, -1.0)
        proposal_center = torch.stack(centers, dim=1)
        proposal_score = torch.stack(scores, dim=1)
        proposal_valid = proposal_score >= self.r410_proposal_score_threshold
        fallback_center = learned_anchor[:, :, :2]
        center = torch.where(proposal_valid[:, :, None], proposal_center, fallback_center)
        anchor = torch.cat([center, learned_anchor[:, :, 2:]], dim=2)

        grid = (2.0 * center - 1.0).reshape(b, k, 1, 2)
        center_feature = F.grid_sample(
            pixel_embedding, grid, mode='bilinear', padding_mode='border', align_corners=True
        )[:, :, :, 0].transpose(1, 2)
        query = F.normalize(learned_query + center_feature, dim=2)
        return query, anchor, proposal_score, proposal_valid.to(pixel_embedding.dtype)

    def _r414_contextual_extent_features(
        self,
        *,
        pixel_embedding: torch.Tensor,
        proposal_center: torch.Tensor,
        type_feature: torch.Tensor,
    ) -> torch.Tensor:
        """Return center + directional local context + type [B,K,6D].

        The grid is centered on the detached predicted center, so extent loss
        cannot improve itself by moving the center.  Left/right/top/bottom
        pooled evidence preserves the axis information needed to distinguish
        width from height while keeping a single lightweight extent MLP.
        """
        if not self.geometry_context_r414_enabled:
            raise RuntimeError("R4.14 contextual extent requested while disabled")
        b, d, _, _ = pixel_embedding.shape
        k = proposal_center.shape[1]
        g = self.r414_context_grid_size
        center = proposal_center.detach()[:, :, None, None, :]
        sample_coord = (center + self.r414_context_offsets[None, None]).clamp(0.0, 1.0)
        grid = (2.0 * sample_coord - 1.0).reshape(b, k * g, g, 2)
        sampled = F.grid_sample(
            pixel_embedding,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).reshape(b, d, k, g, g)
        mid = g // 2
        center_feature = sampled[:, :, :, mid, mid].permute(0, 2, 1)
        left = sampled[:, :, :, :, :mid].mean(dim=(3, 4)).permute(0, 2, 1)
        right = sampled[:, :, :, :, mid + 1 :].mean(dim=(3, 4)).permute(0, 2, 1)
        top = sampled[:, :, :, :mid, :].mean(dim=(3, 4)).permute(0, 2, 1)
        bottom = sampled[:, :, :, mid + 1 :, :].mean(dim=(3, 4)).permute(0, 2, 1)
        return torch.cat([center_feature, left, right, top, bottom, type_feature], dim=2)

    def _r411_typed_proposal_initial_state(
        self,
        *,
        pixel_embedding: torch.Tensor,
        learned_query: torch.Tensor,
        learned_anchor: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Predict typed residual instances with explicit center/box supervision.

        The hard Top-K operation is used only to instantiate a finite query set.
        Unlike R4.10, the locator itself is trainable because its dense center,
        size, and offset maps receive direct teacher supervision in v538_loss.
        Type identity is preserved all the way into the query embedding.
        """
        if self.r411_proposal_stem is None:
            raise RuntimeError("R4.11 typed proposal requested but modules are absent")
        b, d, h, w = pixel_embedding.shape
        k = learned_query.shape[1]
        proposal_feature = self.r411_proposal_stem(pixel_embedding)
        center_logits = self.r411_center_head(proposal_feature)
        if self.r4201_clean_rootfix_enabled and self.box_free_mask_set_r418_enabled:
            # R4.20.1 clean ownership: box/offset geometry is not part of a
            # location-seeded box-free mask. Do not evaluate historical heads.
            size_logits = center_logits.new_zeros((b, 4, 2, h, w))
            offset_logits = center_logits.new_zeros((b, 4, 2, h, w))
        else:
            size_logits = self.r411_size_head(proposal_feature).reshape(b, 4, 2, h, w)
            offset_logits = self.r411_offset_head(proposal_feature).reshape(b, 4, 2, h, w)
        if self.proposal_recovery_r417_enabled:
            location_logits = self.r417_location_head(proposal_feature)
            if self.r4201_clean_rootfix_enabled and self.box_free_mask_set_r418_enabled:
                location_offset_logits = center_logits.new_zeros((b, 2, h, w))
            else:
                location_offset_logits = self.r417_location_offset_head(proposal_feature)
        else:
            location_logits = center_logits.new_zeros((b, 1, h, w))
            location_offset_logits = center_logits.new_zeros((b, 2, h, w))

        center_prob = torch.sigmoid(center_logits)
        spatial = h * w

        # Diagnostic of the legacy typed Top-K competition: if several action
        # channels peak at one physical location they consume several slots.
        legacy_radius = self.r411_proposal_nms_kernel // 2
        legacy_local_max = F.max_pool2d(
            center_prob, self.r411_proposal_nms_kernel, stride=1, padding=legacy_radius
        )
        legacy_peak_prob = center_prob * (
            center_prob >= (legacy_local_max - 1.0e-6)
        ).to(center_prob.dtype)
        legacy_flat = legacy_peak_prob.reshape(b, -1)
        legacy_k = min(k, legacy_flat.shape[1])
        _, legacy_index = torch.topk(legacy_flat, k=legacy_k, dim=1, largest=True, sorted=True)
        legacy_pixel = legacy_index.remainder(spatial)
        legacy_first = torch.ones_like(legacy_pixel, dtype=torch.bool)
        for slot_i in range(legacy_k):
            if slot_i > 0:
                legacy_first[:, slot_i] = ~legacy_pixel[:, :slot_i].eq(legacy_pixel[:, slot_i:slot_i+1]).any(dim=1)
        legacy_unique_fraction = legacy_first.to(center_prob.dtype).mean() if legacy_k > 0 else center_prob.new_zeros(())

        if self.proposal_recovery_r417_enabled:
            # R4.17 recall-first location extraction.  The location map is
            # trained independently from action labels.  Use only a minimal
            # local-max kernel, oversample candidate peaks, then apply a small
            # greedy spatial deduplication.  This keeps two nearby physical
            # residuals when they are genuinely distinct while preventing a
            # broad plateau from filling all K slots with adjacent pixels.
            location_prob = torch.sigmoid(location_logits)
            nms_radius = self.r417_location_nms_kernel // 2
            location_max = F.max_pool2d(
                location_prob, self.r417_location_nms_kernel, stride=1, padding=nms_radius
            )
            location_peak = location_prob * (
                location_prob >= (location_max - 1.0e-6)
            ).to(location_prob.dtype)
            flat_location = location_peak[:, 0].reshape(b, -1)
            candidate_k = min(
                flat_location.shape[1], max(k, k * self.r417_location_oversample_factor)
            )
            candidate_score, candidate_index = torch.topk(
                flat_location, k=candidate_k, dim=1, largest=True, sorted=True
            )
            score = flat_location.new_zeros((b, k))
            pixel_index = torch.zeros((b, k), device=pixel_embedding.device, dtype=torch.long)
            dedup2 = self.r417_location_dedup_radius_px ** 2
            for batch_index in range(b):
                chosen = []
                for candidate_rank in range(candidate_k):
                    idx = int(candidate_index[batch_index, candidate_rank].item())
                    yy = idx // w
                    xx = idx % w
                    if dedup2 > 0.0 and chosen:
                        too_close = any(
                            float((xx - cx) * (xx - cx) + (yy - cy) * (yy - cy)) <= dedup2
                            for cy, cx, _ in chosen
                        )
                        if too_close:
                            continue
                    chosen.append((yy, xx, candidate_rank))
                    slot = len(chosen) - 1
                    pixel_index[batch_index, slot] = idx
                    score[batch_index, slot] = candidate_score[batch_index, candidate_rank]
                    if len(chosen) >= k:
                        break
            py = torch.div(pixel_index, w, rounding_mode="floor")
            px = pixel_index.remainder(w)
            batch_for_type = torch.arange(b, device=pixel_embedding.device)[:, None].expand(b, k)
            typed_at_point = center_logits.permute(0, 2, 3, 1)[batch_for_type, py, px]
            type_index = typed_at_point.argmax(dim=2)
        elif self.unique_point_r416_enabled:
            # One physical location owns at most one slot.  Collapse action
            # channels first, perform spatial NMS once, then bind the winning
            # action type at each retained point.  This removes cross-type slot
            # duplication without adding a new locator network.
            spatial_score, _ = center_prob.max(dim=1, keepdim=True)
            nms_radius = max(int(round(self.r416_cross_type_nms_radius_px)), 1)
            nms_kernel = 2 * nms_radius + 1
            spatial_max = F.max_pool2d(
                spatial_score, nms_kernel, stride=1, padding=nms_radius
            )
            spatial_peak = spatial_score * (
                spatial_score >= (spatial_max - 1.0e-6)
            ).to(spatial_score.dtype)
            flat_spatial = spatial_peak[:, 0].reshape(b, -1)
            topk = min(k, flat_spatial.shape[1])
            score, pixel_index = torch.topk(
                flat_spatial, k=topk, dim=1, largest=True, sorted=True
            )
            if topk < k:
                pad = k - topk
                score = F.pad(score, (0, pad), value=0.0)
                pixel_index = F.pad(pixel_index, (0, pad), value=0)
            py = torch.div(pixel_index, w, rounding_mode="floor")
            px = pixel_index.remainder(w)
            batch_for_type = torch.arange(b, device=pixel_embedding.device)[:, None].expand(b, k)
            typed_at_point = center_prob.permute(0, 2, 3, 1)[batch_for_type, py, px]
            type_index = typed_at_point.argmax(dim=2)
        else:
            flat = legacy_peak_prob.reshape(b, -1)
            topk = min(k, flat.shape[1])
            score, flat_index = torch.topk(flat, k=topk, dim=1, largest=True, sorted=True)
            if topk < k:
                pad = k - topk
                score = F.pad(score, (0, pad), value=0.0)
                flat_index = F.pad(flat_index, (0, pad), value=0)
            type_index = torch.div(flat_index, spatial, rounding_mode="floor").clamp(0, 3)
            pixel_index = flat_index.remainder(spatial)
            py = torch.div(pixel_index, w, rounding_mode="floor")
            px = pixel_index.remainder(w)
        batch = torch.arange(b, device=pixel_embedding.device)[:, None].expand(b, k)

        size_map = self.r47_anchor_min_size + (
            self.r47_anchor_max_size - self.r47_anchor_min_size
        ) * torch.sigmoid(size_logits)
        offset_map = 0.5 * torch.tanh(offset_logits)
        if self.r4201_clean_rootfix_enabled and self.box_free_mask_set_r418_enabled:
            # The selected physical grid peak is the sole center owner.
            shared_offset_map = location_offset_logits.new_zeros((b, 2, h, w))
            pred_offset = location_offset_logits.new_zeros((b, k, 2))
        elif self.proposal_recovery_r417_enabled and self.r417_shared_offset_enabled:
            shared_offset_map = 0.5 * torch.tanh(location_offset_logits)
            pred_offset = shared_offset_map.permute(0, 2, 3, 1)[batch, py, px]
        else:
            shared_offset_map = location_offset_logits.new_zeros((b, 2, h, w))
            pred_offset = offset_map[batch, type_index, :, py, px]

        center_x = ((px.to(pixel_embedding.dtype) + 0.5 + pred_offset[:, :, 0]) / float(max(w, 1))).clamp(0.0, 1.0)
        center_y = ((py.to(pixel_embedding.dtype) + 0.5 + pred_offset[:, :, 1]) / float(max(h, 1))).clamp(0.0, 1.0)
        proposal_center = torch.stack([center_x, center_y], dim=2)
        center_grid = (2.0 * proposal_center - 1.0).reshape(b, k, 1, 2)
        center_feature = F.grid_sample(
            pixel_embedding, center_grid, mode="bilinear", padding_mode="border", align_corners=True,
        )[:, :, :, 0].transpose(1, 2)
        type_feature = (
            self.r411_type_embedding(type_index)
            if self.r411_type_embedding is not None
            else center_feature.new_zeros(center_feature.shape)
        )
        if self.box_free_mask_set_r418_enabled:
            pred_size = proposal_center.new_full((b, k, 2), float(self.r411_initial_box_size))
            pred_edge_offsets = proposal_center.new_zeros((b, k, 4))
            proposal_anchor = torch.cat([proposal_center, pred_size], dim=2)
        elif self.geometry_lock_r413_enabled and self.r413_query_extent_enabled:
            if self.geometry_context_r414_enabled:
                extent_feature = self._r414_contextual_extent_features(
                    pixel_embedding=pixel_embedding,
                    proposal_center=proposal_center,
                    type_feature=type_feature,
                )
            else:
                extent_feature = torch.cat([center_feature, type_feature, learned_query], dim=2)
            extent_raw = self.r413_extent_head(extent_feature)
            if self.r416_asymmetric_ltrb_enabled:
                edge_scale = max(self.r47_anchor_max_size, 1.0e-3)
                pred_edge_offsets = edge_scale * torch.tanh(extent_raw)
                x1 = (proposal_center[:, :, 0] + pred_edge_offsets[:, :, 0]).clamp(0.0, 1.0)
                y1 = (proposal_center[:, :, 1] + pred_edge_offsets[:, :, 1]).clamp(0.0, 1.0)
                x2 = (proposal_center[:, :, 0] + pred_edge_offsets[:, :, 2]).clamp(0.0, 1.0)
                y2 = (proposal_center[:, :, 1] + pred_edge_offsets[:, :, 3]).clamp(0.0, 1.0)
                left, right = torch.minimum(x1, x2), torch.maximum(x1, x2)
                top, bottom = torch.minimum(y1, y2), torch.maximum(y1, y2)
                pred_w = (right - left).clamp(self.r47_anchor_min_size, self.r47_anchor_max_size)
                pred_h = (bottom - top).clamp(self.r47_anchor_min_size, self.r47_anchor_max_size)
                pred_cx = (0.5 * (left + right)).clamp(0.0, 1.0)
                pred_cy = (0.5 * (top + bottom)).clamp(0.0, 1.0)
                # Keep the box inside the image after min/max-size clamping.
                pred_cx = torch.maximum(torch.minimum(pred_cx, 1.0 - 0.5 * pred_w), 0.5 * pred_w)
                pred_cy = torch.maximum(torch.minimum(pred_cy, 1.0 - 0.5 * pred_h), 0.5 * pred_h)
                proposal_anchor = torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=2)
            else:
                log_min = math.log(max(self.r47_anchor_min_size, 1.0e-6))
                log_max = math.log(max(self.r47_anchor_max_size, self.r47_anchor_min_size + 1.0e-6))
                log_size = log_min + (log_max - log_min) * torch.sigmoid(extent_raw)
                pred_size = torch.exp(log_size)
                pred_edge_offsets = proposal_center.new_zeros((b, k, 4))
                proposal_anchor = torch.cat([proposal_center, pred_size], dim=2)
        else:
            pred_size = size_map[batch, type_index, :, py, px]
            pred_edge_offsets = proposal_center.new_zeros((b, k, 4))
            proposal_anchor = torch.cat([proposal_center, pred_size], dim=2)
        valid = score >= self.r411_proposal_score_threshold
        if self.r4201_clean_rootfix_enabled and self.box_free_mask_set_r418_enabled:
            # Confidence gates presence/execution, not spatial identity.  Even
            # an early low-confidence Top-K peak remains the query's reference
            # point so the mask objective is never conditioned on an unrelated
            # learned fallback anchor.
            anchor = proposal_anchor
        else:
            anchor = torch.where(valid[:, :, None], proposal_anchor, learned_anchor)
        if self.box_free_mask_set_r418_enabled:
            # Location is only a positional seed.  R4.20-A2 additionally keeps
            # action semantics out of the mask-shape query, so a temporary type
            # mistake cannot rotate the residual geometry representation.
            if self.dynamic_residual_mask_r420_enabled and self.r420_type_decoupled_mask_enabled:
                proposal_query = learned_query + center_feature
            else:
                proposal_query = learned_query + center_feature + type_feature
        else:
            box_feature = self.r411_box_query_mlp(anchor)
            proposal_query = learned_query + center_feature + type_feature + box_feature
        if self.r4201_clean_rootfix_enabled and self.box_free_mask_set_r418_enabled:
            query = F.normalize(proposal_query, dim=2)
        else:
            query = F.normalize(
                torch.where(valid[:, :, None], proposal_query, learned_query), dim=2
            )
        return (
            query,
            anchor,
            score,
            valid.to(pixel_embedding.dtype),
            type_index,
            center_logits,
            size_map,
            offset_map,
            proposal_center,
            pred_edge_offsets,
            legacy_unique_fraction,
            location_logits,
            shared_offset_map,
        )

    def _r412_action_support(
        self,
        *,
        base_probability: torch.Tensor,
        boundary: torch.Tensor,
        query_type: torch.Tensor,
    ) -> torch.Tensor:
        """Return GT-free action-admissibility support [B,K,H,W].

        Delete/Trim may only remove Native foreground; Fill/Expand may only add
        Native background.  Boundary actions receive a soft, dilated Base
        boundary prior rather than a hard band, so legitimate corrections are
        never made mathematically impossible.
        """
        p = base_probability[:, :1].clamp(0.0, 1.0)
        bd = boundary[:, :1].clamp(0.0, 1.0)
        radius = self.r412_boundary_band_kernel // 2
        bd = F.max_pool2d(bd, self.r412_boundary_band_kernel, stride=1, padding=radius)
        boundary_soft = 0.20 + 0.80 * bd
        support = torch.cat(
            [
                p,
                1.0 - p,
                p * boundary_soft,
                (1.0 - p) * boundary_soft,
            ],
            dim=1,
        )
        b, _, h, w = support.shape
        k = query_type.shape[1]
        batch = torch.arange(b, device=support.device)[:, None].expand(b, k)
        chosen = support[batch, query_type.long().clamp(0, 3)]
        return chosen.clamp(self.r412_action_support_floor, 1.0)

    def _r411_render_local_masks(
        self,
        *,
        pixel_embedding: torch.Tensor,
        query: torch.Tensor,
        anchor: torch.Tensor,
        query_type: Optional[torch.Tensor] = None,
        base_probability: Optional[torch.Tensor] = None,
        boundary: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode a residual mask inside a differentiable query-specific ROI.

        R4.12 keeps R4.11's GT-free local rendering contract but changes the
        learning geometry: the local renderer works on a canonical high-
        resolution ROI, receives explicit canonical coordinates and a Native
        Base action-support channel, and performs uncertainty-focused residual
        refinement before the result is pasted back to image coordinates.
        """
        if self.r411_roi_mask_head is None:
            raise RuntimeError("R4.11/R4.12 local ROI decoder requested but modules are absent")
        b, d, h, w = pixel_embedding.shape
        k = query.shape[1]
        r = self.r412_roi_size if self.canonical_shape_r412_enabled else self.r411_roi_size
        axis = torch.linspace(-1.0, 1.0, r, device=pixel_embedding.device, dtype=pixel_embedding.dtype)
        gy, gx = torch.meshgrid(axis, axis, indexing="ij")
        local_grid = torch.stack([gx, gy], dim=2)[None, None].expand(b, k, -1, -1, -1)
        center = anchor[:, :, None, None, :2]
        size = (self.r411_roi_expand * anchor[:, :, None, None, 2:]).clamp_min(1.0e-4)
        sample_xy = center + 0.5 * size * local_grid
        sample_grid = (2.0 * sample_xy - 1.0).reshape(b * k, r, r, 2)
        source = pixel_embedding[:, None].expand(-1, k, -1, -1, -1).reshape(b * k, d, h, w)
        roi_feature = F.grid_sample(
            source, sample_grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        q_map = query.reshape(b * k, d, 1, 1).expand(-1, -1, r, r)

        action_support_full = None
        if (
            self.canonical_shape_r412_enabled
            and isinstance(query_type, torch.Tensor)
            and isinstance(base_probability, torch.Tensor)
            and isinstance(boundary, torch.Tensor)
        ):
            action_support_full = self._r412_action_support(
                base_probability=base_probability, boundary=boundary, query_type=query_type
            )
            support_source = action_support_full.reshape(b * k, 1, h, w)
            support_roi = F.grid_sample(
                support_source, sample_grid, mode="bilinear", padding_mode="border", align_corners=True
            )
        else:
            support_roi = roi_feature.new_ones((b * k, 1, r, r))

        if self.canonical_shape_r412_enabled:
            coord = local_grid.permute(0, 1, 4, 2, 3).reshape(b * k, 2, r, r)
            local_feature = self.r412_roi_stem(
                torch.cat([roi_feature, q_map, coord, support_roi], dim=1)
            )
            local_feature = self.r412_roi_blocks(local_feature)
            coarse = self.r412_coarse_head(local_feature)
            probability = torch.sigmoid(coarse)
            uncertainty = 1.0 - (2.0 * (probability - 0.5).abs()).clamp(0.0, 1.0)
            refine = self.r412_refine_head(torch.cat([local_feature, uncertainty], dim=1))
            roi_logits = coarse + uncertainty.detach() * refine
        else:
            roi_logits = self.r411_roi_mask_head(torch.cat([roi_feature, q_map], dim=1))

        yy = torch.linspace(0.0, 1.0, h, device=pixel_embedding.device, dtype=pixel_embedding.dtype)
        xx = torch.linspace(0.0, 1.0, w, device=pixel_embedding.device, dtype=pixel_embedding.dtype)
        y_map = yy[:, None].expand(h, w)[None, None]
        x_map = xx[None, :].expand(h, w)[None, None]
        cx = anchor[:, :, 0, None, None]
        cy = anchor[:, :, 1, None, None]
        roi_w = (self.r411_roi_expand * anchor[:, :, 2, None, None]).clamp_min(1.0e-4)
        roi_h = (self.r411_roi_expand * anchor[:, :, 3, None, None]).clamp_min(1.0e-4)
        u = 2.0 * (x_map - cx) / roi_w
        v = 2.0 * (y_map - cy) / roi_h
        paste_grid = torch.stack([u, v], dim=-1).reshape(b * k, h, w, 2)
        pasted = F.grid_sample(
            roi_logits, paste_grid, mode="bilinear", padding_mode="zeros", align_corners=True
        ).reshape(b, k, h, w)
        tau = 0.10
        support_x = torch.sigmoid((1.0 - u.abs()) / tau)
        support_y = torch.sigmoid((1.0 - v.abs()) / tau)
        geometric_support = (support_x * support_y).clamp_min(math.exp(-self.r411_outside_penalty))
        result = pasted + geometric_support.log().clamp(min=-self.r411_outside_penalty, max=0.0)

        if action_support_full is not None and self.r412_action_support_strength > 0.0:
            result = result + self.r412_action_support_strength * action_support_full.log()
        return result

    def _r4203_dense_competitive_residual_set(
        self,
        *,
        pixel_embedding: torch.Tensor,
        location_logits: torch.Tensor,
        typed_center_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Predict a residual-component set directly from the dense residual field.

        R4.20.3 intentionally removes the Dense->TopK->Point->Mask information
        bottleneck from the Native path.  Learned residual slots compete for
        every pixel jointly with an explicit background state.  The dense R4.17
        location logit acts as a differentiable residual-vs-background prior;
        no radius, box, hard crop, NMS point, offset range, sampling count, or
        dynamic-mask channel width owns the residual geometry.

        The first competition softly pools full-image evidence into each slot.
        The pooled evidence updates the slot once, after which a second
        competition produces the final mutually-exclusive soft ownership.  No
        new trainable projection is required: the existing R4.7 pixel encoder
        and learned slot embeddings own the representation.
        """
        if not self.dense_competitive_residual_set_r4203_enabled:
            raise RuntimeError("V552-R4.20.3 dense residual set requested while disabled")
        if pixel_embedding.ndim != 4:
            raise ValueError(f"pixel_embedding must be BCHW, got {tuple(pixel_embedding.shape)}")
        b, d, h, w = pixel_embedding.shape
        if location_logits.shape != (b, 1, h, w):
            raise ValueError(
                "V552-R4.20.3 location logits must be [B,1,H,W], got "
                f"{tuple(location_logits.shape)}"
            )
        if typed_center_logits.shape != (b, 4, h, w):
            raise ValueError(
                "V552-R4.20.3 typed center logits must be [B,4,H,W], got "
                f"{tuple(typed_center_logits.shape)}"
            )

        k = self.num_slots
        dtype = pixel_embedding.dtype
        eps = max(float(EPS), 1.0e-6)
        # Pixel/query vectors are L2-normalized.  Multiplying cosine scores by
        # sqrt(d) restores the dimensional scale of ordinary dot-product
        # attention without introducing a tunable temperature.
        scale = math.sqrt(float(max(d, 1)))

        # Existing learned residual slots are the only instance-capacity owner.
        # No hard location is used to instantiate a slot.
        q0 = F.normalize(self.r47_slot_queries.weight, dim=1)[None].expand(b, -1, -1)
        pix = F.normalize(pixel_embedding, dim=1)
        loc = location_logits[:, 0]

        def compete(query: torch.Tensor) -> torch.Tensor:
            slot_score = torch.einsum("bkd,bdhw->bkhw", query, pix) * scale
            # Dense location evidence is a soft prior, not a hard gate.  A high
            # residual logit simultaneously favors residual slots and disfavors
            # the background; every pixel remains differentiable.
            residual_score = slot_score + loc[:, None]
            background_score = -loc[:, None]
            return torch.softmax(
                torch.cat([background_score, residual_score], dim=1), dim=1
            )

        ownership0 = compete(q0)
        slot0 = ownership0[:, 1:]
        mass0 = slot0.flatten(2).sum(dim=2).clamp_min(eps)
        evidence = torch.einsum("bkhw,bdhw->bkd", slot0, pix) / mass0[:, :, None]
        q1 = F.normalize(q0 + evidence, dim=2)

        ownership = compete(q1)
        background = ownership[:, :1]
        slot_prob = ownership[:, 1:].clamp(eps, 1.0 - eps)
        # Downstream code applies sigmoid(logits / mask_temperature).  Scale the
        # inverse-sigmoid so that the resulting Native probability is exactly
        # the competitive ownership probability.
        final_logits = float(self.mask_temperature) * torch.logit(slot_prob, eps=eps)

        # Soft centroids are compatibility diagnostics only.  They never enter
        # mask generation and therefore cannot recreate the old point bottleneck.
        yy = torch.linspace(0.0, 1.0, h, device=pix.device, dtype=dtype)
        xx = torch.linspace(0.0, 1.0, w, device=pix.device, dtype=dtype)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        mass = slot_prob.flatten(2).sum(dim=2).clamp_min(eps)
        cx = (slot_prob * gx[None, None]).flatten(2).sum(dim=2) / mass
        cy = (slot_prob * gy[None, None]).flatten(2).sum(dim=2) / mass
        zero_extent = torch.zeros_like(cx)
        soft_anchor = torch.stack([cx, cy, zero_extent, zero_extent], dim=2)
        soft_point = torch.stack([cx, cy], dim=2)

        # Type is read from the already-supervised dense typed field after the
        # spatial set has formed; type never owns mask geometry.
        type_score = torch.einsum(
            "bkhw,bchw->bkc", slot_prob, typed_center_logits
        ) / mass[:, :, None]
        proposal_type = type_score.argmax(dim=2)

        # Presence is learned downstream from candidate utility.  All slots are
        # eligible for one-to-one mask matching; unmatched suppression/no-object
        # supervision decides which slots become inactive.  No score threshold
        # participates in Native mask construction.
        proposal_score = (
            slot_prob * torch.sigmoid(location_logits)
        ).flatten(2).sum(dim=2) / mass
        proposal_valid = torch.ones_like(proposal_score)

        normalized_entropy = -(
            ownership.clamp_min(eps) * ownership.clamp_min(eps).log()
        ).sum(dim=1).mean() / math.log(float(k + 1))
        ownership_sum_error = (ownership.sum(dim=1) - 1.0).abs().mean()
        slot_mass_fraction = slot_prob.flatten(2).mean(dim=2)
        slot_mass_mean = slot_mass_fraction.mean(dim=1, keepdim=True).clamp_min(eps)
        slot_mass_cv = (
            slot_mass_fraction.std(dim=1, unbiased=False) / slot_mass_mean[:, 0]
        ).mean()
        max_slot_ownership = slot_prob.max(dim=1).values.mean()

        return {
            "final_logits": final_logits,
            "slot_probability": slot_prob,
            "background_probability": background,
            "refined_query": q1,
            "soft_anchor": soft_anchor,
            "soft_point": soft_point,
            "proposal_type": proposal_type,
            "proposal_score": proposal_score,
            "proposal_valid": proposal_valid,
            "ownership_sum_error": ownership_sum_error.detach(),
            "assignment_entropy": normalized_entropy.detach(),
            "background_fraction": background.mean().detach(),
            "max_slot_ownership": max_slot_ownership.detach(),
            "slot_mass_cv": slot_mass_cv.detach(),
        }

    def _r4207_visual_instance_seeds(
        self,
        *,
        pixel_embedding: torch.Tensor,
        location_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Return deterministic R4.17 visual seeds for K instance queries.

        The extraction intentionally reuses the established R4.17 policy:
        local maxima -> oversampled candidates -> greedy spatial deduplication.
        A seed only initializes query *identity*.  It never defines a support
        window, ROI, box, Gaussian prior, or any other mask-geometry owner.

        Selection indices are non-differentiable by design; R4.17 keeps its own
        dense centre supervision.  The sampled visual feature remains connected
        to ``pixel_embedding`` so the Native M1 representation learns how to
        encode an instance at a correctly localized seed.
        """
        if not (
            self.dynamic_visual_instance_binding_r4207_enabled
            or self.normalized_visual_instance_binding_r4208_enabled
        ):
            raise RuntimeError("R4.20.7/8 visual seed extraction requested while disabled")
        b, d, h, w = pixel_embedding.shape
        k = self.num_slots
        if location_logits.shape != (b, 1, h, w):
            raise RuntimeError(
                "V552-R4.20.7 requires R4.17 location logits [B,1,H,W]; "
                f"got {tuple(location_logits.shape)}"
            )

        location_probability = torch.sigmoid(location_logits)
        nms_radius = self.r417_location_nms_kernel // 2
        location_max = F.max_pool2d(
            location_probability,
            self.r417_location_nms_kernel,
            stride=1,
            padding=nms_radius,
        )
        location_peak = location_probability * (
            location_probability >= (location_max - 1.0e-6)
        ).to(location_probability.dtype)
        flat_location = location_peak[:, 0].reshape(b, -1)
        candidate_k = min(
            flat_location.shape[1],
            max(k, k * self.r417_location_oversample_factor),
        )
        candidate_score, candidate_index = torch.topk(
            flat_location,
            k=candidate_k,
            dim=1,
            largest=True,
            sorted=True,
        )
        # R4.21.0 separates maximum capacity from actual instance cardinality.
        # Candidate ranking remains the established monotonic R4.17 probability
        # ranking, but validity is decided in raw-logit space at the canonical
        # binary boundary 0 (equivalently p=0.5).  This avoids a tunable
        # probability threshold and, unlike sigmoid(score)>0, does not force all
        # K slots to carry visual seeds.
        flat_logit = location_logits[:, 0].reshape(b, -1)

        seed_score = flat_location.new_zeros((b, k))
        seed_logit = flat_location.new_full((b, k), -float("inf"))
        seed_index = torch.zeros((b, k), device=pixel_embedding.device, dtype=torch.long)
        seed_chosen = torch.zeros((b, k), device=pixel_embedding.device, dtype=torch.bool)
        dedup2 = self.r417_location_dedup_radius_px ** 2
        # K and candidate_k are small (K=6 in the formal protocol).  Reusing
        # the existing deterministic greedy R4.17 rule is preferable to adding
        # a second seed-definition mechanism solely for this ablation.
        for batch_index in range(b):
            chosen = []
            for candidate_rank in range(candidate_k):
                idx = int(candidate_index[batch_index, candidate_rank].item())
                yy = idx // w
                xx = idx % w
                if dedup2 > 0.0 and chosen:
                    too_close = any(
                        float((xx - cx) * (xx - cx) + (yy - cy) * (yy - cy)) <= dedup2
                        for cy, cx, _ in chosen
                    )
                    if too_close:
                        continue
                chosen.append((yy, xx, candidate_rank))
                slot = len(chosen) - 1
                seed_index[batch_index, slot] = idx
                seed_score[batch_index, slot] = candidate_score[batch_index, candidate_rank]
                seed_logit[batch_index, slot] = flat_logit[batch_index, idx]
                seed_chosen[batch_index, slot] = True
                if len(chosen) >= k:
                    break

        if self.proposal_existence_decoupling_r4211_enabled:
            # R4.21.1: location peaks are *proposals*, never existence decisions.
            # All finite chosen proposals may contribute a visual identity with
            # continuous strength `seed_score`; the already-supervised slot
            # presence head owns actual variable cardinality downstream.  This
            # removes both the historical always-valid sigmoid>0 rule and the
            # R4.21.0 over-correction logit>0 hard gate.
            seed_valid = seed_chosen & torch.isfinite(seed_logit)
        elif self.variable_cardinality_seeds_r4210_enabled:
            # Historical R4.21.0 hard-gate control retained for causal comparison.
            seed_valid = seed_chosen & torch.isfinite(seed_logit) & (seed_logit > 0.0)
        else:
            # Historical R4.20.7/8 behavior retained bit-for-bit for controls.
            seed_valid = seed_chosen & (seed_score > 0)
        py = torch.div(seed_index, w, rounding_mode="floor")
        px = seed_index.remainder(w)
        center_x = (px.to(pixel_embedding.dtype) + 0.5) / float(max(w, 1))
        center_y = (py.to(pixel_embedding.dtype) + 0.5) / float(max(h, 1))
        seed_center = torch.stack([center_x, center_y], dim=2)
        grid = (2.0 * seed_center - 1.0).reshape(b, k, 1, 2)
        seed_feature = F.grid_sample(
            pixel_embedding,
            grid,
            mode="bilinear",
            padding_mode="border",
            # The centre convention above is (pixel+0.5)/size, which is the
            # exact align_corners=False convention.  Keep R4.20.7 bitwise
            # backward-compatible, but use the mathematically consistent
            # convention for R4.20.8.
            align_corners=(False if self.normalized_visual_instance_binding_r4208_enabled else True),
        )[:, :, :, 0].transpose(1, 2)
        seed_feature = torch.where(
            seed_valid[:, :, None], seed_feature, torch.zeros_like(seed_feature)
        )

        finite_fraction = torch.isfinite(seed_feature).to(pixel_embedding.dtype).mean()
        valid_float = seed_valid.to(pixel_embedding.dtype)
        valid_count = valid_float.sum().clamp_min(1.0)
        score_mean = (seed_score * valid_float).sum() / valid_count
        valid_fraction = valid_float.mean()
        if k > 1:
            dx = (seed_center[:, :, None, 0] - seed_center[:, None, :, 0]) * float(max(w, 1))
            dy = (seed_center[:, :, None, 1] - seed_center[:, None, :, 1]) * float(max(h, 1))
            distance = torch.sqrt(dx.square() + dy.square() + 1.0e-12)
            eye = torch.eye(k, device=pixel_embedding.device, dtype=torch.bool)[None]
            pair_valid = seed_valid[:, :, None] & seed_valid[:, None, :] & (~eye)
            pairwise_distance_px = torch.where(
                pair_valid.any(), distance[pair_valid].mean(), distance.new_zeros(())
            )
        else:
            pairwise_distance_px = pixel_embedding.new_zeros(())

        return {
            "center": seed_center,
            "score": seed_score,
            "logit": seed_logit,
            "chosen": seed_chosen,
            "valid": seed_valid,
            "feature": seed_feature,
            "feature_finite_fraction": finite_fraction.detach(),
            "valid_fraction": valid_fraction.detach(),
            "score_mean": score_mean.detach(),
            "pairwise_distance_px": pairwise_distance_px.detach(),
        }

    def _v560_clean_query_set(
        self,
        *,
        pixel_embedding: torch.Tensor,
        cause_probability: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Direct variable-cardinality correction set used by V560.

        Each learned query predicts a complete Bernoulli component mask directly:

            M_k(x) = sigmoid(q_k^T F(x)).

        There is intentionally no shared occupancy multiplier, no K-way softmax,
        no seed ownership, no overflow/dustbin and no box/ROI geometry owner.
        The query receives one image-conditioned evidence update using its own
        soft mask.  Existence/no-object is predicted later by the existing
        presence head and therefore owns cardinality rather than mask geometry.
        """
        if pixel_embedding.ndim != 4:
            raise ValueError(
                f"V560 pixel_embedding must be BCHW, got {tuple(pixel_embedding.shape)}"
            )
        b, d, h, w = pixel_embedding.shape
        k = self.num_slots
        eps = max(float(EPS), 1.0e-6)
        pix = F.normalize(pixel_embedding, dim=1)
        scale = math.sqrt(float(max(d, 1)))
        q0 = F.normalize(self.r47_slot_queries.weight, dim=1)[None].expand(b, -1, -1)
        # V560R2 removes the untrained mask->global-evidence->query feedback
        # that collapsed all six BUSI masks toward the same full-image support.
        q1 = q0
        final_logits = torch.einsum("bkd,bdhw->bkhw", q1, pix) * scale
        if self.v560_mask_bias is None:
            raise RuntimeError("V560 mask bias is missing")
        final_logits = final_logits + self.v560_mask_bias[None, :, None, None]
        slot_probability = torch.sigmoid(final_logits).clamp(eps, 1.0 - eps)

        yy = torch.linspace(0.0, 1.0, h, device=pix.device, dtype=pix.dtype)
        xx = torch.linspace(0.0, 1.0, w, device=pix.device, dtype=pix.dtype)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        x = gx[None, None]
        y = gy[None, None]
        mass = slot_probability.flatten(2).sum(dim=2).clamp_min(eps)
        cx = (slot_probability * x).flatten(2).sum(dim=2) / mass
        cy = (slot_probability * y).flatten(2).sum(dim=2) / mass
        soft_anchor = torch.stack([cx, cy, torch.zeros_like(cx), torch.zeros_like(cy)], dim=2)
        soft_point = torch.stack([cx, cy], dim=2)

        typed = cause_probability[:, :4].detach()
        type_score = torch.einsum("bkhw,bchw->bkc", slot_probability, typed) / mass[:, :, None]
        proposal_type = type_score.argmax(dim=2)
        proposal_score = slot_probability.flatten(2).amax(dim=2)
        proposal_valid = torch.ones_like(proposal_score)

        if k > 1:
            sim0 = torch.einsum("bkd,bjd->bkj", q0, q0)
            sim1 = torch.einsum("bkd,bjd->bkj", q1, q1)
            eye = torch.eye(k, device=pix.device, dtype=torch.bool)[None]
            q0_cos = sim0.masked_select(~eye).mean()
            q1_cos = sim1.masked_select(~eye).mean()
        else:
            q0_cos = final_logits.new_zeros(())
            q1_cos = final_logits.new_zeros(())
        overlap_mass = F.relu(slot_probability.sum(dim=1) - 1.0).mean()
        return {
            "final_logits": final_logits,
            "slot_probability": slot_probability,
            "refined_query": q1,
            "soft_anchor": soft_anchor,
            "soft_point": soft_point,
            "proposal_type": proposal_type,
            "proposal_score": proposal_score,
            "proposal_valid": proposal_valid,
            "q0_pairwise_cosine": q0_cos.detach(),
            "q1_pairwise_cosine": q1_cos.detach(),
            "soft_overlap_mass": overlap_mass.detach(),
            "mask_probability_mean": slot_probability.mean().detach(),
            "mask_bias_mean": self.v560_mask_bias.mean().detach(),
        }

    def _tc_drcs_query_set(
        self,
        *,
        pixel_embedding: torch.Tensor,
        pixels: torch.Tensor,
        base_probability: torch.Tensor,
        supervision_masks: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Teacher-complete differentiable residual component set.

        This path deliberately has no hard proposal owner.  Residual probability
        is an auxiliary inference-visible evidence channel only.  Learned queries
        first predict a complete mask set; those *soft* masks provide additive
        log-attention priors for one shared cross-attention/FFN refinement stage.
        During training, factual residual components create mask-piloted queries
        that use the same decoder and mask renderer.  Pilot tensors never enter
        Native inference/deployment.
        """
        if not self.tc_drcs_enabled:
            raise RuntimeError("TC-DRCS helper called while protocol is disabled")
        if self.tc_mask_pixel_fuse is None or self.tc_pilot_query_proj is None:
            raise RuntimeError("TC-DRCS modules are not constructed")
        b, d, h, w = pixel_embedding.shape
        eps = max(float(EPS), 1.0e-6)
        scale = math.sqrt(float(max(self.r47_mask_dim, 1)))

        # Residual remains a continuously supervised evidence field.  No detach,
        # argmax, Top-K, NMS, hard Voronoi or seed identity is used downstream.
        residual_logits = self.v562_residual_head(pixel_embedding)
        residual_probability = torch.sigmoid(residual_logits).clamp(eps, 1.0 - eps)
        mask_pixel = F.normalize(
            self.tc_mask_pixel_fuse(torch.cat([pixel_embedding, residual_probability], dim=1)),
            dim=1,
        )

        def render(query: torch.Tensor) -> torch.Tensor:
            mask_query = F.normalize(self.v561_mask_embed(query), dim=2)
            logits = torch.einsum("bkd,bdhw->bkhw", mask_query, mask_pixel) * scale
            return logits + self.v561_mask_bias[None, :, None, None]

        q0 = F.normalize(self.r47_slot_queries.weight, dim=1)[None].expand(b, -1, -1)
        stage0_logits = render(q0)
        stage0_probability = torch.sigmoid(stage0_logits).clamp(eps, 1.0 - eps)
        # Soft localization remains differentiable all the way back to the mask
        # renderer and residual/pixel evidence.  log(p) is a standard additive
        # attention prior and does not exclude any pixel from receiving gradient.
        stage0_bias = stage0_probability.log().flatten(2)
        q1, stage1_attn = self.v561_global_decoder(
            q0,
            pixels,
            attn_bias=stage0_bias,
            residual_scale=1.0,
            normalize_output=True,
            disable_self_attention=True,
        )
        final_logits = render(q1)
        slot_probability = torch.sigmoid(final_logits).clamp(eps, 1.0 - eps)
        action_logits = self.v561_query_action_head(q1)
        action_probs = F.softmax(action_logits / self.action_temperature, dim=2)
        presence_logits = self.v562_query_presence_head(q1).squeeze(-1)

        # Soft centroid is diagnostic/localization state only; it never decides
        # matching or whether a teacher receives supervision.
        yy = torch.linspace(0.0, 1.0, h, device=pixel_embedding.device, dtype=pixel_embedding.dtype)
        xx = torch.linspace(0.0, 1.0, w, device=pixel_embedding.device, dtype=pixel_embedding.dtype)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        mass = slot_probability.flatten(2).sum(dim=2).clamp_min(eps)
        cx = (slot_probability * gx[None, None]).flatten(2).sum(dim=2) / mass
        cy = (slot_probability * gy[None, None]).flatten(2).sum(dim=2) / mass
        soft_anchor = torch.stack([cx, cy, torch.zeros_like(cx), torch.zeros_like(cy)], dim=2)

        pilot_logits = final_logits.new_zeros(final_logits.shape)
        pilot_action_logits = action_logits.new_zeros(action_logits.shape)
        pilot_valid = torch.zeros((b, self.num_slots), device=final_logits.device, dtype=torch.bool)
        pilot_teacher_masks = final_logits.new_zeros(final_logits.shape)
        pilot_teacher_actions = torch.zeros((b, self.num_slots), device=final_logits.device, dtype=torch.long)
        teacher_raw_count = final_logits.new_zeros((b,))
        if self.training and isinstance(supervision_masks, torch.Tensor):
            # Reuse exactly the factual component builder consumed by the loss.
            # The pilot branch is therefore teacher-complete for every retained
            # component and has no independently invented target semantics.
            from utils.v538_loss import _build_teacher_components
            (
                pilot_teacher_masks,
                pilot_teacher_actions,
                pilot_valid,
                _pilot_area,
                _pilot_replay,
                _pilot_effective,
                _pilot_error,
                teacher_raw_count,
            ) = _build_teacher_components(
                teacher_probability=base_probability[:, :1].detach(),
                gt=supervision_masks,
                num_slots=self.num_slots,
                min_pixels=self.r48_teacher_min_pixels,
                current_epoch=int(self.current_epoch),
                replay_enabled=False,
                replay_error_floor=0.0,
                replay_radius=1,
                utility_rank_enabled=True,
            )
            teacher = pilot_teacher_masks.to(pixel_embedding.dtype)
            teacher_mass = teacher.flatten(2).sum(dim=2).clamp_min(1.0)
            # A GT-mask-guided pilot query is a differentiable masked visual
            # prototype.  The projection starts at identity, so early training
            # receives a stable spatially localized decoder signal.
            pooled = torch.einsum("bkhw,bdhw->bkd", teacher, pixel_embedding)
            pooled = pooled / teacher_mass[:, :, None]
            pilot_q0 = F.normalize(self.tc_pilot_query_proj(pooled), dim=2)
            # Outside pixels keep a tiny finite probability rather than -inf;
            # gradients/attention kernels therefore remain numerically stable.
            pilot_guidance = teacher * (1.0 - 2.0 * eps) + eps
            pilot_bias = pilot_guidance.clamp(eps, 1.0 - eps).log().flatten(2)
            pilot_q1, _ = self.v561_global_decoder(
                pilot_q0,
                pixels,
                attn_bias=pilot_bias,
                residual_scale=1.0,
                normalize_output=True,
                disable_self_attention=True,
            )
            pilot_logits = render(pilot_q1)
            pilot_action_logits = self.v561_query_action_head(pilot_q1)

        def pairwise_cosine(q: torch.Tensor) -> torch.Tensor:
            if self.num_slots <= 1:
                return q.new_zeros(())
            qn = F.normalize(q, dim=2)
            sim = torch.einsum("bkd,bjd->bkj", qn, qn)
            eye = torch.eye(self.num_slots, device=q.device, dtype=torch.bool)[None]
            return sim.masked_select(~eye).mean()

        attn_entropy = -(
            stage1_attn.clamp_min(eps) * stage1_attn.clamp_min(eps).log()
        ).sum(dim=2) / math.log(float(max(h * w, 2)))
        overlap_mass = F.relu(slot_probability.sum(dim=1) - 1.0).mean()
        return {
            "final_logits": final_logits,
            "slot_probability": slot_probability,
            "refined_query": q1,
            "stage1_query": q1,
            "stage1_action_logits": action_logits,
            "action_logits": action_logits,
            "action_probs": action_probs,
            "presence_logits": presence_logits,
            "typed_support": slot_probability.new_full(slot_probability.shape, 0.5),
            "soft_anchor": soft_anchor,
            "v562_residual_logits": residual_logits,
            "v562_residual_probability": residual_probability,
            "v562_proposal_anchor_xy": soft_anchor[:, :, :2],
            "v563_attention_window": torch.ones_like(slot_probability),
            "v563_mask_window": torch.ones_like(slot_probability),
            "v563_raw_mask_probability": slot_probability,
            "v563_persistent_identity": q0,
            "v564_rootfix_enabled": final_logits.new_zeros(()),
            "v564_proposal_radius": final_logits.new_zeros((b, self.num_slots)),
            "v564_proposal_radius_mean": final_logits.new_zeros(()),
            "v564_proposal_shape_prior": torch.zeros_like(final_logits),
            "v564_proposal_shape_prior_abs_mean": final_logits.new_zeros(()),
            "v564_dual_stream_identity_enabled": final_logits.new_zeros(()),
            "v564_typed_spatial_feedback_disabled": final_logits.new_ones(()),
            "v565_rootfix_enabled": final_logits.new_zeros(()),
            "v565_seed_logits": final_logits.new_zeros((b, 1, h, w)),
            "v565_seed_probability": final_logits.new_zeros((b, 1, h, w)),
            "v565_seed_probability_mean": final_logits.new_zeros(()),
            "v565_relative_support": torch.zeros_like(slot_probability),
            "v565_relative_support_mean": final_logits.new_zeros(()),
            "v565_attention_radius": final_logits.new_zeros((b, self.num_slots)),
            "v565_attention_radius_mean": final_logits.new_zeros(()),
            "v565_peak_to_background_contrast": final_logits.new_zeros((b, self.num_slots)),
            "v565_peak_to_background_contrast_mean": final_logits.new_zeros(()),
            "v565_shape_condition_abs_mean": final_logits.new_zeros(()),
            "clean_dynamic_component_set_enabled": final_logits.new_ones(()),
            "tc_drcs_enabled": final_logits.new_ones(()),
            "clean_attention_precision_mean": final_logits.new_zeros(()),
            "clean_mask_precision_mean": final_logits.new_zeros(()),
            "clean_loss_log_vars": self.clean_loss_log_vars,
            "v563_pre_gate_mask_probability_mean": slot_probability.mean().detach(),
            "v563_outside_mask_probability": final_logits.new_zeros(()),
            "v563_attention_window_fraction": final_logits.new_ones(()),
            "v563_mask_window_fraction": final_logits.new_ones(()),
            "v563_identity_retention_q1": F.cosine_similarity(q1, q0, dim=2).mean().detach(),
            "v563_identity_retention_q2": F.cosine_similarity(q1, q0, dim=2).mean().detach(),
            "mask_probability_mean": slot_probability.mean().detach(),
            "soft_overlap_mass": overlap_mass.detach(),
            "q0_pairwise_cosine": pairwise_cosine(q0).detach(),
            "q1_pairwise_cosine": pairwise_cosine(q1).detach(),
            "q2_pairwise_cosine": pairwise_cosine(q1).detach(),
            "stage1_query_delta_norm": (q1 - q0).norm(dim=2).mean().detach(),
            "stage2_query_delta_norm": final_logits.new_zeros(()),
            "typed_support_mean": final_logits.new_tensor(0.5),
            "typed_support_std": final_logits.new_zeros(()),
            "typed_support_neutrality_error": final_logits.new_zeros(()),
            "stage1_attention_entropy_ratio": attn_entropy.mean().detach(),
            "stage2_attention_entropy_ratio": final_logits.new_zeros(()),
            "mask_bias_mean": self.v561_mask_bias.mean().detach(),
            # Attached TC tensors consumed by the loss.  Final-stage Hungarian
            # assignment is reused for stage0; pilot teacher index is explicit.
            "tc_stage0_logits": stage0_logits,
            "tc_pilot_logits": pilot_logits,
            "tc_pilot_action_logits": pilot_action_logits,
            "tc_pilot_valid": pilot_valid,
            "tc_pilot_teacher_masks": pilot_teacher_masks,
            "tc_pilot_teacher_actions": pilot_teacher_actions,
            "tc_teacher_raw_count": teacher_raw_count,
        }

    def _v561_base_conditioned_query_set(
        self,
        *,
        fused_feature: torch.Tensor,
        base_probability: torch.Tensor,
        cause_probability: torch.Tensor,
        action_alpha: torch.Tensor,
        entropy: torch.Tensor,
        boundary: torch.Tensor,
        supervision_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Base-conditioned residual correction set used by V561 BCRS-M1.

        A0/static uses the learned correction slots without image conditioning.
        A1/image performs one DETR-style full-image query update.
        A2/typed adds a second query update with a *parameter-free* physical
        support prior derived from the predicted correction action and the
        current Base foreground probability:

            S_k(x) = p_add,k * (1 - P_B(x)) + p_remove,k * P_B(x).

        The prior is soft and threshold-free.  Because the action head is
        initialized uniformly, S_k(x)=0.5 everywhere at initialization, so the
        typed stage starts neutral rather than hard-coding a spatial bias.
        """
        if not self.base_conditioned_residual_set_v561_enabled:
            raise RuntimeError("V561 BCRS query set called while V561 is disabled")
        b, _, h, w = fused_feature.shape
        cause = cause_probability
        if cause.shape[1] < 4:
            cause = F.pad(cause, (0, 0, 0, 0, 0, 4 - cause.shape[1]))
        cause = cause[:, :4].detach()
        alpha = action_alpha
        if alpha.shape[1] < 4:
            alpha = F.pad(alpha, (0, 0, 0, 0, 0, 4 - alpha.shape[1]))
        alpha = alpha[:, :4].detach()
        if self.clean_dynamic_component_set_enabled:
            # CLEAN geometry is conditioned only on inference-visible image/Base
            # evidence. Historical cause/alpha heads remain available to legacy
            # M2 compatibility code but cannot steer the clean M1 geometry path.
            cause = torch.zeros_like(cause)
            alpha = torch.zeros_like(alpha)

        yy = torch.linspace(0.0, 1.0, h, device=fused_feature.device, dtype=fused_feature.dtype)
        xx = torch.linspace(0.0, 1.0, w, device=fused_feature.device, dtype=fused_feature.dtype)
        y_grid = yy[:, None].expand(h, w)
        x_grid = xx[None, :].expand(h, w)
        xy = torch.stack([x_grid, y_grid], dim=0)[None].expand(b, -1, -1, -1)
        pixel_input = torch.cat(
            [
                fused_feature,
                base_probability[:, :1].detach(),
                entropy[:, :1].detach(),
                boundary[:, :1].detach(),
                cause,
                alpha,
                xy,
            ],
            dim=1,
        )
        pixel_embedding = F.normalize(self.r47_pixel_encoder(pixel_input), dim=1)
        pixels = pixel_embedding.flatten(2).transpose(1, 2)  # [B,HW,D]
        learned_q = F.normalize(self.r47_slot_queries.weight, dim=1)[None].expand(b, -1, -1)

        if self.tc_drcs_enabled:
            return self._tc_drcs_query_set(
                pixel_embedding=pixel_embedding,
                pixels=pixels,
                base_probability=base_probability,
                supervision_masks=supervision_masks,
            )

        # V562: predict residual existence from inference-visible features, take
        # spatially distinct local maxima, and bind one query to each peak.  GT
        # supervises the heatmap only in the loss; it never enters this path.
        residual_logits = pixel_embedding.new_zeros((b, 1, h, w))
        residual_probability = torch.sigmoid(residual_logits)
        v565_seed_logits = pixel_embedding.new_zeros((b, 1, h, w))
        v565_seed_probability = torch.sigmoid(v565_seed_logits)
        v565_relative_support = pixel_embedding.new_zeros((b, self.num_slots, h, w))
        v565_attention_radius = pixel_embedding.new_full((b, self.num_slots), self.v563_attention_radius)
        v565_peak_to_background_contrast = pixel_embedding.new_zeros((b, self.num_slots))
        clean_attention_precision = pixel_embedding.new_zeros((b, self.num_slots))
        proposal_anchor_xy = pixel_embedding.new_zeros((b, self.num_slots, 2))
        local_bias = pixel_embedding.new_zeros((b, self.num_slots, h * w))
        v563_attention_window = pixel_embedding.new_ones((b, self.num_slots, h, w))
        v563_mask_window = pixel_embedding.new_ones((b, self.num_slots, h, w))
        v563_persistent_identity = learned_q
        v564_proposal_radius = pixel_embedding.new_full((b, self.num_slots), self.v563_mask_radius)
        q0 = learned_q
        if self.bcrs_v561_variant in {"rootfix", "persistent"}:
            residual_logits = self.v562_residual_head(pixel_embedding)
            residual_probability = torch.sigmoid(residual_logits)
            if self.bcrs_v561_variant == "persistent":
                if self.clean_dynamic_component_set_enabled:
                    # CLEAN: anchors come directly from local maxima of the
                    # continuously supervised residual-error field.  3x3 max
                    # pooling is the 8-neighbourhood definition of a local
                    # maximum, not a tunable NMS radius.
                    score = residual_probability[:, 0].detach()
                    pooled = F.max_pool2d(score[:, None], 3, stride=1, padding=1)[:, 0]
                    local_max = score >= (pooled - 1.0e-12)
                    peak_score = score.masked_fill(~local_max, -1.0)
                    # Parameter-free diversity readout.  Plain Top-K can spend
                    # several slots on adjacent peaks of the same residual blob,
                    # which is exactly what DuplicateOwner/UnmatchedTeacher audit.
                    # First choose the strongest local maximum; each subsequent
                    # seed maximizes residual evidence times distance to the
                    # nearest already-selected seed.  No NMS radius or scale is
                    # introduced.
                    selected = []
                    available = local_max.clone()
                    min_distance2 = score.new_full((b, h, w), float("inf"))
                    batch_index = torch.arange(b, device=score.device)
                    for slot_index in range(self.num_slots):
                        if slot_index == 0:
                            diversity_score = peak_score
                        else:
                            diversity_score = (
                                score.clamp_min(0.0)
                                * torch.sqrt(min_distance2.clamp_min(0.0) + EPS)
                            ).masked_fill(~available, -1.0)
                        idx = diversity_score.flatten(1).argmax(dim=1)
                        selected.append(idx)
                        py = torch.div(idx, w, rounding_mode="floor")
                        px = idx.remainder(w)
                        available[batch_index, py, px] = False
                        ax = (px.to(pixel_embedding.dtype) + 0.5) / float(w)
                        ay = (py.to(pixel_embedding.dtype) + 0.5) / float(h)
                        dist2_selected = (x_grid[None] - ax[:, None, None]).square() + (
                            y_grid[None] - ay[:, None, None]
                        ).square()
                        min_distance2 = torch.minimum(min_distance2, dist2_selected)
                    peak_index = torch.stack(selected, dim=1)
                    # No fixed radius exists on the clean path.
                    v564_proposal_radius = residual_probability.new_zeros((b, self.num_slots))
                else:
                    # Historical V563/V564/V565 compatibility path.
                    if self.v565_rootfix_enabled:
                        v565_seed_logits = self.v565_seed_head(pixel_embedding)
                        v565_seed_probability = torch.sigmoid(v565_seed_logits)
                        score = v565_seed_probability[:, 0].detach().clone()
                    else:
                        score = residual_probability[:, 0].detach().clone()
                    original_score = residual_probability[:, 0].detach().clone()
                    selected = []
                    selected_radius = []
                    for _ in range(self.num_slots):
                        idx = score.flatten(1).argmax(dim=1)
                        selected.append(idx)
                        py = torch.div(idx, w, rounding_mode="floor")
                        px = idx.remainder(w)
                        ax = (px.to(pixel_embedding.dtype) + 0.5) / float(w)
                        ay = (py.to(pixel_embedding.dtype) + 0.5) / float(h)
                        ddx = x_grid[None] - ax[:, None, None]
                        ddy = y_grid[None] - ay[:, None, None]
                        dist2_seed = ddx.square() + ddy.square()
                        if self.v565_rootfix_enabled:
                            radius = original_score.new_full((b,), self.v564_min_mask_radius)
                            suppress_radius = original_score.new_full((b,), self.v565_seed_nms_radius)
                        elif self.v564_rootfix_enabled:
                            probe_sigma = max(self.v564_max_mask_radius, 0.05)
                            local_weight = original_score.clamp_min(EPS) * torch.exp(
                                -dist2_seed / (2.0 * probe_sigma * probe_sigma)
                            )
                            denom = local_weight.sum(dim=(-2, -1)).clamp_min(EPS)
                            second_moment = (local_weight * dist2_seed).sum(dim=(-2, -1)) / denom
                            radius = (1.50 * torch.sqrt(second_moment.clamp_min(EPS))).clamp(
                                self.v564_min_mask_radius, self.v564_max_mask_radius
                            )
                            suppress_radius = (0.75 * radius).clamp_min(0.03)
                        else:
                            radius = original_score.new_full((b,), self.v563_mask_radius)
                            suppress_radius = original_score.new_full(
                                (b,), max(self.v563_mask_radius * 0.75, 0.04)
                            )
                        selected_radius.append(radius)
                        suppress = dist2_seed <= suppress_radius[:, None, None].square()
                        score = score.masked_fill(suppress, -1.0)
                    peak_index = torch.stack(selected, dim=1)
                    v564_proposal_radius = torch.stack(selected_radius, dim=1)
            else:
                nms_kernel = 11
                pooled = F.max_pool2d(
                    residual_probability, nms_kernel, stride=1, padding=nms_kernel // 2
                )
                peak_score = residual_probability[:, 0].masked_fill(
                    residual_probability[:, 0] < pooled[:, 0], -1.0
                )
                topk = min(self.num_slots, h * w)
                _, peak_index = peak_score.flatten(1).topk(topk, dim=1)
                if topk < self.num_slots:
                    pad = peak_index[:, -1:].expand(-1, self.num_slots - topk)
                    peak_index = torch.cat([peak_index, pad], dim=1)
            peak_y = torch.div(peak_index, w, rounding_mode="floor")
            peak_x = peak_index.remainder(w)
            anchor_x = (peak_x.to(pixel_embedding.dtype) + 0.5) / float(w)
            anchor_y = (peak_y.to(pixel_embedding.dtype) + 0.5) / float(h)
            proposal_anchor_xy = torch.stack([anchor_x, anchor_y], dim=2)

            pixel_flat = pixel_embedding.flatten(2).transpose(1, 2)
            gather_index = peak_index[:, :, None].expand(-1, -1, pixel_flat.shape[2])
            anchor_feature = pixel_flat.gather(1, gather_index)
            anchor_feature_condition = self.v562_anchor_feature_proj(anchor_feature)
            anchor_position_condition = self.v562_anchor_pos_mlp(proposal_anchor_xy)
            anchor_condition_v562 = anchor_feature_condition + anchor_position_condition
            dx = x_grid[None, None] - proposal_anchor_xy[:, :, 0, None, None]
            dy = y_grid[None, None] - proposal_anchor_xy[:, :, 1, None, None]
            distance2 = dx.square() + dy.square()
            if self.clean_dynamic_component_set_enabled:
                # CLEAN: equal-weight semantic/visual/positional evidence and a
                # learned positive spatial precision. No fixed Gaussian sigma,
                # hard attention radius, identity-mix coefficient or mask window.
                q0 = F.normalize(
                    learned_q + anchor_feature_condition + anchor_position_condition, dim=2
                )
                v563_persistent_identity = F.normalize(
                    learned_q + anchor_position_condition, dim=2
                )
                clean_attention_precision = F.softplus(
                    self.clean_attention_precision_head(q0).squeeze(-1)
                )
                # Parameter-free Voronoi ownership: every image pixel is read by
                # exactly its nearest anchor.  This prevents all queries from
                # re-reading the same global content without introducing a fixed
                # attention radius. Learned precision only shapes attention inside
                # each slot's own territory.
                nearest_slot = distance2.argmin(dim=1, keepdim=True)
                slot_ids = torch.arange(
                    self.num_slots, device=distance2.device
                ).view(1, self.num_slots, 1, 1)
                voronoi = nearest_slot.eq(slot_ids)
                local_bias_map = -clean_attention_precision[:, :, None, None] * distance2
                local_bias_map = local_bias_map.masked_fill(~voronoi, -1.0e4)
                local_bias = local_bias_map.flatten(2)
                v563_attention_window = voronoi.to(distance2.dtype)
                v563_mask_window = torch.ones_like(distance2)
            else:
                # Historical V562--V565 compatibility path.
                q0 = F.normalize(learned_q + 0.25 * anchor_condition_v562, dim=2)
                sigma = 0.15
                local_bias = (-(distance2) / (2.0 * sigma * sigma)).flatten(2)
                if self.bcrs_v561_variant == "persistent":
                    v563_persistent_identity = F.normalize(
                        learned_q + 0.50 * anchor_position_condition, dim=2
                    )
                    if self.v564_rootfix_enabled:
                        q0 = F.normalize(
                            anchor_feature_condition + 0.75 * learned_q, dim=2
                        )
                    else:
                        q0 = F.normalize(
                            v563_persistent_identity + 0.25 * anchor_feature_condition, dim=2
                        )
                    if self.v565_rootfix_enabled:
                        occupancy = residual_probability[:, 0].detach()[:, None].expand(b, self.num_slots, h, w)
                        probe = distance2 <= self.v564_max_mask_radius ** 2
                        probe_count = probe.flatten(2).sum(dim=2).clamp_min(1).to(occupancy.dtype)
                        probe_mean = (occupancy * probe.to(occupancy.dtype)).flatten(2).sum(dim=2) / probe_count
                        flat_occ = residual_probability[:, 0].detach().flatten(1)
                        peak_occ = flat_occ.gather(1, peak_index)
                        contrast = (peak_occ - probe_mean).clamp_min(0.0)
                        threshold = probe_mean + self.v565_support_relative_threshold * contrast
                        support = probe & (occupancy >= threshold[:, :, None, None])
                        seed_onehot = occupancy.new_zeros((b, self.num_slots, h * w))
                        seed_onehot.scatter_(2, peak_index[:, :, None], 1.0)
                        connected = seed_onehot.reshape(b * self.num_slots, 1, h, w)
                        support_flat = support.reshape(b * self.num_slots, 1, h, w).to(connected.dtype)
                        connected = connected * support_flat
                        for _flood in range(24):
                            connected = F.max_pool2d(connected, 3, stride=1, padding=1) * support_flat
                        connected = connected.reshape(b, self.num_slots, h, w)
                        radii = []
                        for bi in range(b):
                            row = []
                            for ki in range(self.num_slots):
                                mask_bk = connected[bi, ki] > 0.5
                                if int(mask_bk.sum().item()) >= 2:
                                    d_bk = torch.sqrt(distance2[bi, ki][mask_bk].clamp_min(0.0))
                                    r_bk = torch.quantile(d_bk, self.v565_extent_quantile)
                                else:
                                    r_bk = occupancy.new_tensor(self.v564_min_mask_radius)
                                row.append(r_bk.clamp(self.v564_min_mask_radius, self.v564_max_mask_radius))
                            radii.append(torch.stack(row))
                        v564_proposal_radius = torch.stack(radii, dim=0)
                        denom = (peak_occ - probe_mean).clamp_min(0.02)[:, :, None, None]
                        v565_relative_support = ((occupancy - probe_mean[:, :, None, None]) / denom).clamp(0.0, 1.0)
                        v565_relative_support = v565_relative_support * connected.detach()
                        v565_peak_to_background_contrast = contrast
                        v565_attention_radius = (self.v565_attention_radius_scale * v564_proposal_radius).clamp(
                            min=max(self.v564_min_mask_radius * 1.25, 0.06), max=self.v565_max_attention_radius
                        )
                    attention_radius_for_slot = (
                        v565_attention_radius if self.v565_rootfix_enabled
                        else distance2.new_full((b, self.num_slots), self.v563_attention_radius)
                    )
                    v563_attention_window = (
                        distance2 <= attention_radius_for_slot[:, :, None, None].square()
                    ).to(pixel_embedding.dtype)
                    radius_for_mask = (
                        v564_proposal_radius if self.v564_rootfix_enabled
                        else distance2.new_full((b, self.num_slots), self.v563_mask_radius)
                    )
                    v563_mask_window = (
                        distance2 <= radius_for_mask[:, :, None, None].square()
                    ).to(pixel_embedding.dtype)
                    gaussian = -distance2 / (2.0 * max(self.v563_mask_radius, 0.02) ** 2)
                    local_bias = torch.where(
                        v563_attention_window > 0.5,
                        gaussian,
                        gaussian.new_full(gaussian.shape, -1.0e4),
                    ).flatten(2)

        q1 = q0
        stage1_attn = pixel_embedding.new_zeros((b, self.num_slots, h * w))
        if self.bcrs_v561_variant in {"image", "typed", "rootfix", "persistent"}:
            is_v563 = self.bcrs_v561_variant == "persistent"
            q1, stage1_attn = self.v561_global_decoder(
                q0, pixels,
                attn_bias=(local_bias if self.bcrs_v561_variant in {"rootfix", "persistent"} else None),
                residual_scale=(
                    1.0 if self.clean_dynamic_component_set_enabled
                    else (self.v563_query_residual_scale if is_v563
                          else (0.25 if self.bcrs_v561_variant == "rootfix" else 1.0))
                ),
                normalize_output=(self.bcrs_v561_variant in {"rootfix", "persistent"}),
                persistent_identity=(
                    v563_persistent_identity if (is_v563 and not self.v564_rootfix_enabled and not self.clean_dynamic_component_set_enabled) else None
                ),
                identity_mix=(
                    self.v563_identity_mix if (is_v563 and not self.v564_rootfix_enabled and not self.clean_dynamic_component_set_enabled) else 0.0
                ),
                disable_self_attention=is_v563,
                attention_identity=(
                    v563_persistent_identity if (is_v563 and self.v564_rootfix_enabled and not self.clean_dynamic_component_set_enabled) else None
                ),
                attention_identity_scale=(
                    self.v564_attention_identity_scale if (is_v563 and self.v564_rootfix_enabled and not self.clean_dynamic_component_set_enabled) else 0.0
                ),
            )

        stage1_action_logits = self.v561_query_action_head(q1)
        stage1_action_probs = F.softmax(
            stage1_action_logits / self.action_temperature, dim=2
        )
        remove_probability = (
            stage1_action_probs[:, :, 0] + stage1_action_probs[:, :, 2]
        ).clamp(0.0, 1.0)
        add_probability = (
            stage1_action_probs[:, :, 1] + stage1_action_probs[:, :, 3]
        ).clamp(0.0, 1.0)
        base = base_probability[:, :1].detach().clamp(0.0, 1.0)
        typed_support = (
            add_probability[:, :, None, None] * (1.0 - base)
            + remove_probability[:, :, None, None] * base
        ).clamp(EPS, 1.0)

        q2 = q1
        stage2_attn = stage1_attn.new_zeros(stage1_attn.shape)
        if self.bcrs_v561_variant in {"typed", "rootfix", "persistent"} and not self.clean_dynamic_component_set_enabled:
            typed_bias = typed_support.clamp_min(EPS).log().flatten(2)
            if self.bcrs_v561_variant in {"rootfix", "persistent"}:
                # V564 does not let an untrained action prediction steer spatial
                # identity.  Action is supervised only *after* ownership is known.
                typed_bias = local_bias if self.v564_rootfix_enabled else typed_bias + local_bias
            is_v563 = self.bcrs_v561_variant == "persistent"
            q2, stage2_attn = self.v561_typed_decoder(
                q1, pixels, attn_bias=typed_bias,
                residual_scale=(
                    self.v563_query_residual_scale if is_v563
                    else (0.25 if self.bcrs_v561_variant == "rootfix" else 1.0)
                ),
                normalize_output=(self.bcrs_v561_variant in {"rootfix", "persistent"}),
                persistent_identity=(
                    v563_persistent_identity if (is_v563 and not self.v564_rootfix_enabled) else None
                ),
                identity_mix=(
                    self.v563_identity_mix if (is_v563 and not self.v564_rootfix_enabled) else 0.0
                ),
                disable_self_attention=is_v563,
                attention_identity=(
                    v563_persistent_identity if (is_v563 and self.v564_rootfix_enabled) else None
                ),
                attention_identity_scale=(
                    self.v564_attention_identity_scale if (is_v563 and self.v564_rootfix_enabled) else 0.0
                ),
            )

        q_final = q2
        q_for_geometry = (
            F.normalize(q_final + self.v564_attention_identity_scale * v563_persistent_identity, dim=2)
            if (self.bcrs_v561_variant == "persistent" and self.v564_rootfix_enabled and not self.clean_dynamic_component_set_enabled)
            else q_final
        )
        mask_query = F.normalize(self.v561_mask_embed(q_for_geometry), dim=2)
        if self.clean_dynamic_component_set_enabled:
            mask_pixel_input_clean = torch.cat([pixel_embedding, residual_probability], dim=1)
            mask_pixel = F.normalize(self.clean_mask_pixel_fuse(mask_pixel_input_clean), dim=1)
        elif self.bcrs_v561_variant == "persistent" and self.v565_rootfix_enabled:
            mask_pixel_input_v565 = torch.cat(
                [pixel_embedding, residual_probability, v565_seed_probability], dim=1
            )
            mask_pixel = F.normalize(self.v565_mask_pixel_fuse(mask_pixel_input_v565), dim=1)
        else:
            mask_pixel = F.normalize(pixel_embedding, dim=1)
        scale = math.sqrt(float(max(self.r47_mask_dim, 1)))
        raw_final_logits = torch.einsum("bkd,bdhw->bkhw", mask_query, mask_pixel) * scale
        raw_final_logits = raw_final_logits + self.v561_mask_bias[None, :, None, None]
        clean_mask_precision = raw_final_logits.new_zeros((b, self.num_slots))
        if self.clean_dynamic_component_set_enabled:
            clean_mask_precision = F.softplus(
                self.clean_mask_precision_head(q_final).squeeze(-1)
            )
            raw_final_logits = raw_final_logits - clean_mask_precision[:, :, None, None] * distance2
        v564_proposal_shape_prior = raw_final_logits.new_zeros(raw_final_logits.shape)
        v565_shape_condition_logits = raw_final_logits.new_zeros(raw_final_logits.shape)
        if self.bcrs_v561_variant == "persistent" and self.v565_rootfix_enabled:
            # Peak-relative connected residual support supplies a bounded,
            # per-slot geometry bias.  Unlike V564's ~0.02-logit prior this is
            # intentionally O(1), but remains bounded in [-shape_scale,+shape_scale].
            # Dense occupancy + center heatmap have already entered mask_pixel
            # through v565_mask_pixel_fuse above.
            v565_shape_condition_logits = (2.0 * v565_relative_support - 1.0) * v563_mask_window
            v564_proposal_shape_prior = v565_shape_condition_logits
            raw_final_logits = raw_final_logits + self.v565_shape_scale * v565_shape_condition_logits
        elif self.bcrs_v561_variant == "persistent" and self.v564_rootfix_enabled:
            residual_map = residual_probability[:, :1, :, :].clamp_min(EPS)
            local_residual = residual_map * v563_mask_window
            local_mean = local_residual.sum(dim=(-2, -1), keepdim=True) / (
                v563_mask_window.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
            )
            relative_log = (
                residual_map.log() - local_mean.clamp_min(EPS).log()
            ).clamp(-4.0, 4.0)
            v564_proposal_shape_prior = relative_log * v563_mask_window
            raw_final_logits = raw_final_logits + (
                self.v564_proposal_shape_scale * v564_proposal_shape_prior
            )
        final_logits = raw_final_logits
        if self.bcrs_v561_variant == "persistent" and not self.clean_dynamic_component_set_enabled:
            # V563 gives the anchor a physical geometry contract: a correction
            # mask cannot expand outside its predicted local neighborhood.
            # This is GT-free and therefore identical in train/val/test forward.
            final_logits = torch.where(
                v563_mask_window > 0.5,
                raw_final_logits,
                raw_final_logits.new_full(
                    raw_final_logits.shape, -self.v563_outside_logit_penalty
                ),
            )
        slot_probability = torch.sigmoid(final_logits).clamp(EPS, 1.0 - EPS)

        # The final typed action prediction is the executable action owner.
        final_action_logits = self.v561_query_action_head(q_final)
        final_action_probs = F.softmax(
            final_action_logits / self.action_temperature, dim=2
        )

        mass = slot_probability.flatten(2).sum(dim=2).clamp_min(EPS)
        x = x_grid[None, None]
        y = y_grid[None, None]
        cx = (slot_probability * x).flatten(2).sum(dim=2) / mass
        cy = (slot_probability * y).flatten(2).sum(dim=2) / mass
        if self.bcrs_v561_variant in {"rootfix", "persistent"}:
            # Matching identity is proposal-owned, not decided by the mask it is
            # trying to learn.  Width/height are intentionally unused.
            soft_anchor = torch.cat(
                [proposal_anchor_xy, proposal_anchor_xy.new_zeros((b, self.num_slots, 2))],
                dim=2,
            )
            presence_logits = self.v562_query_presence_head(q_final).squeeze(-1)
        else:
            soft_anchor = torch.stack([cx, cy, torch.zeros_like(cx), torch.zeros_like(cy)], dim=2)
            presence_logits = q_final.new_zeros((b, self.num_slots))

        def pairwise_cosine(q: torch.Tensor) -> torch.Tensor:
            if self.num_slots <= 1:
                return q.new_zeros(())
            qn = F.normalize(q, dim=2)
            sim = torch.einsum("bkd,bjd->bkj", qn, qn)
            eye = torch.eye(self.num_slots, device=q.device, dtype=torch.bool)[None]
            return sim.masked_select(~eye).mean()

        def normalized_entropy(attn: torch.Tensor) -> torch.Tensor:
            if attn.numel() == 0:
                return final_logits.new_zeros(())
            denom = math.log(float(max(h * w, 2)))
            return (
                -(attn.clamp_min(EPS) * attn.clamp_min(EPS).log()).sum(dim=2)
                / denom
            ).mean()

        # At exact zero action logits this is identically zero, independently
        # of the Base mask.  It is a direct regression guard for neutral typed
        # support initialization, not an optimization target.
        uniform_support = typed_support.new_full(typed_support.shape, 0.5)
        typed_neutrality_error = (typed_support - uniform_support).abs().mean()
        overlap_mass = F.relu(slot_probability.sum(dim=1) - 1.0).mean()
        return {
            "final_logits": final_logits,
            "slot_probability": slot_probability,
            "refined_query": q_final,
            "stage1_query": q1,
            "stage1_action_logits": stage1_action_logits,
            "action_logits": final_action_logits,
            "action_probs": final_action_probs,
            "presence_logits": presence_logits,
            "typed_support": typed_support,
            "soft_anchor": soft_anchor,
            "v562_residual_logits": residual_logits,
            "v562_residual_probability": residual_probability,
            "v562_proposal_anchor_xy": proposal_anchor_xy,
            "v563_attention_window": v563_attention_window,
            "v563_mask_window": v563_mask_window,
            "v563_raw_mask_probability": torch.sigmoid(raw_final_logits),
            "v563_persistent_identity": v563_persistent_identity,
            "v564_rootfix_enabled": final_logits.new_tensor(1.0 if self.v564_rootfix_enabled else 0.0),
            "v564_proposal_radius": v564_proposal_radius,
            "v564_proposal_radius_mean": v564_proposal_radius.mean().detach(),
            "v564_proposal_shape_prior": v564_proposal_shape_prior,
            "v564_proposal_shape_prior_abs_mean": v564_proposal_shape_prior.abs().mean().detach(),
            "v564_dual_stream_identity_enabled": final_logits.new_tensor(1.0 if self.v564_rootfix_enabled else 0.0),
            "v564_typed_spatial_feedback_disabled": final_logits.new_tensor(1.0 if self.v564_rootfix_enabled else 0.0),
            "v565_rootfix_enabled": final_logits.new_tensor(1.0 if self.v565_rootfix_enabled else 0.0),
            "v565_seed_logits": v565_seed_logits,
            "v565_seed_probability": v565_seed_probability,
            "v565_seed_probability_mean": v565_seed_probability.mean().detach(),
            "v565_relative_support": v565_relative_support,
            "v565_relative_support_mean": v565_relative_support.mean().detach(),
            "v565_attention_radius": v565_attention_radius,
            "v565_attention_radius_mean": v565_attention_radius.mean().detach(),
            "v565_peak_to_background_contrast": v565_peak_to_background_contrast,
            "v565_peak_to_background_contrast_mean": v565_peak_to_background_contrast.mean().detach(),
            "v565_shape_condition_abs_mean": v565_shape_condition_logits.abs().mean().detach(),
            "clean_dynamic_component_set_enabled": final_logits.new_tensor(1.0 if self.clean_dynamic_component_set_enabled else 0.0),
            "clean_attention_precision_mean": clean_attention_precision.mean().detach(),
            "clean_mask_precision_mean": clean_mask_precision.mean().detach(),
            "clean_loss_log_vars": (
                self.clean_loss_log_vars if self.clean_dynamic_component_set_enabled
                else final_logits.new_zeros((4,))
            ),
            "v563_pre_gate_mask_probability_mean": torch.sigmoid(raw_final_logits).mean().detach(),
            "v563_outside_mask_probability": (
                slot_probability * (1.0 - v563_mask_window)
            ).mean().detach(),
            "v563_attention_window_fraction": v563_attention_window.mean().detach(),
            "v563_mask_window_fraction": v563_mask_window.mean().detach(),
            "v563_identity_retention_q1": F.cosine_similarity(
                F.normalize(q1, dim=2), F.normalize(v563_persistent_identity, dim=2), dim=2
            ).mean().detach(),
            "v563_identity_retention_q2": F.cosine_similarity(
                F.normalize(q2, dim=2), F.normalize(v563_persistent_identity, dim=2), dim=2
            ).mean().detach(),
            "mask_probability_mean": slot_probability.mean().detach(),
            "soft_overlap_mass": overlap_mass.detach(),
            "q0_pairwise_cosine": pairwise_cosine(q0).detach(),
            "q1_pairwise_cosine": pairwise_cosine(q1).detach(),
            "q2_pairwise_cosine": pairwise_cosine(q2).detach(),
            "stage1_query_delta_norm": (q1 - q0).norm(dim=2).mean().detach(),
            "stage2_query_delta_norm": (q2 - q1).norm(dim=2).mean().detach(),
            "typed_support_mean": typed_support.mean().detach(),
            "typed_support_std": typed_support.std(unbiased=False).detach(),
            "typed_support_neutrality_error": typed_neutrality_error.detach(),
            "stage1_attention_entropy_ratio": normalized_entropy(stage1_attn).detach(),
            "stage2_attention_entropy_ratio": normalized_entropy(stage2_attn).detach(),
            "mask_bias_mean": self.v561_mask_bias.mean().detach(),
        }

    def _r4204_factorized_residual_existence_identity_set(
        self,
        *,
        pixel_embedding: torch.Tensor,
        occupancy_logits: torch.Tensor,
        typed_center_logits: torch.Tensor,
        location_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Factor dense residual existence from conditional component identity.

        R4.20.4 separates dense residual existence from component identity:

            r(p)        = sigmoid(occupancy_logit(p))
            pi_k(p)     = P(editable slot k | p, residual)
            M_k(p)      = r(p) * pi_k(p)
            M_bg(p)     = 1 - r(p)

        R4.20.5 fixes the remaining *capacity contradiction*.  The full residual
        field is not guaranteed to contain at most K connected components, while
        the Teacher intentionally retains at most K editable components.  R4.20.4
        nevertheless forced all residual mass into those K slots.  When R4.20.5
        is enabled we add one non-editable reference/dustbin category:

            [pi_1,...,pi_K,pi_o] = softmax([s_1,...,s_K,0])
            O(p)                 = r(p) * pi_o(p)
            sum_k M_k(p) + O(p)  = r(p)

        The dustbin logit is fixed to zero.  This is not a capacity restriction:
        softmax is invariant to a common additive offset, so one category may be
        chosen as the reference logit without losing any representable
        categorical distribution.  It introduces no tunable threshold, no new
        random initialization, and no extra parameter.  Crucially, ``O`` never
        enters the editable K-slot output consumed by M2.

        R4.20.5 explicitly forbids the failed R4.20.4-A2 self-derived spatial
        feedback.  Any future positional identity must be an independent
        ablation using exogenous position evidence, not a centroid/variance prior
        recursively derived from the slot prediction itself.
        """
        if not self.factorized_residual_existence_identity_r4204_enabled:
            raise RuntimeError("V552-R4.20.4 factorized residual set requested while disabled")
        if self.capacity_consistent_overflow_r4205_enabled and self.r4204_spatial_identity_enabled:
            raise RuntimeError("V552-R4.20.5 forbids self-derived R4.20.4 spatial identity")
        if pixel_embedding.ndim != 4:
            raise ValueError(f"pixel_embedding must be BCHW, got {tuple(pixel_embedding.shape)}")
        b, d, h, w = pixel_embedding.shape
        if occupancy_logits.shape != (b, 1, h, w):
            raise ValueError(
                "V552-R4.20.4 occupancy logits must be [B,1,H,W], got "
                f"{tuple(occupancy_logits.shape)}"
            )
        if typed_center_logits.shape != (b, 4, h, w):
            raise ValueError(
                "V552-R4.20.4 typed center logits must be [B,4,H,W], got "
                f"{tuple(typed_center_logits.shape)}"
            )

        k = self.num_slots
        dtype = pixel_embedding.dtype
        eps = max(float(EPS), 1.0e-6)
        scale = math.sqrt(float(max(d, 1)))

        pix = F.normalize(pixel_embedding, dim=1)
        # Preserve the historical R4.20.5 control arithmetic exactly when the
        # treatment is disabled.  The raw expanded embedding is needed only to
        # add image-conditioned seed features in R4.20.7.
        q0_control = F.normalize(self.r47_slot_queries.weight, dim=1)[None].expand(b, -1, -1)
        learned_query = self.r47_slot_queries.weight[None].expand(b, -1, -1)
        r4207_seed_center = pixel_embedding.new_zeros((b, k, 2))
        r4207_seed_score = pixel_embedding.new_zeros((b, k))
        r4207_seed_valid = torch.zeros((b, k), device=pixel_embedding.device, dtype=torch.bool)
        r4207_seed_feature_finite_fraction = pixel_embedding.new_ones(())
        r4207_seed_valid_fraction = pixel_embedding.new_zeros(())
        r4207_seed_score_mean = pixel_embedding.new_zeros(())
        r4207_seed_pairwise_distance_px = pixel_embedding.new_zeros(())
        r4208_enabled = (
            self.normalized_visual_instance_binding_r4208_enabled
            and not self.disable_visual_seed_identity_r4212_enabled
        )
        r4208_learned_query_norm = pixel_embedding.new_zeros(())
        r4208_seed_feature_norm = pixel_embedding.new_zeros(())
        r4208_seed_to_learned_norm_ratio = pixel_embedding.new_zeros(())
        r4208_seed_feature_pairwise_cosine = pixel_embedding.new_zeros(())
        r4208_q0_seed_identity_cosine = pixel_embedding.new_zeros(())
        r4208_q1_seed_identity_cosine = pixel_embedding.new_zeros(())
        r4208_identity_retention_delta = pixel_embedding.new_zeros(())
        r4210_enabled = self.instance_valid_factorization_r4210_enabled
        r4210_variable_seed_enabled = self.variable_cardinality_seeds_r4210_enabled
        r4210_independent_overflow_enabled = self.independent_overflow_gate_r4210_enabled
        r4210_valid_seed_count = pixel_embedding.new_zeros(())
        r4210_seed_logit = pixel_embedding.new_zeros((b, k))
        r4210_seed_logit_mean = pixel_embedding.new_zeros(())
        r4210_overflow_logit_mean = pixel_embedding.new_zeros(())
        r4210_overflow_conditional_mean = pixel_embedding.new_zeros(())
        r4211_enabled = self.instance_valid_decoupling_r4211_enabled
        r4211_proposal_existence_enabled = self.proposal_existence_decoupling_r4211_enabled
        r4211_geometry_overflow_enabled = self.geometry_overflow_decoupling_r4211_enabled
        r4211_proposal_seed_count = pixel_embedding.new_zeros(())
        r4211_proposal_confidence_mean = pixel_embedding.new_zeros(())
        r4211_geometry_effective_l1 = pixel_embedding.new_zeros(())
        r4211_geometry_slot_probability = pixel_embedding.new_zeros((b, k, h, w))
        r4211_effective_slot_probability = pixel_embedding.new_zeros((b, k, h, w))
        identity_key = pixel_embedding.new_zeros((b, k, d))
        seed_unit = pixel_embedding.new_zeros((b, k, d))
        content0 = q0_control
        seed = None
        if self.dynamic_visual_instance_binding_r4207_enabled or r4208_enabled:
            if location_logits is None:
                raise RuntimeError(
                    "V552-R4.20.7/8 requires the inference-visible R4.17 location logits"
                )
            seed = self._r4207_visual_instance_seeds(
                pixel_embedding=pixel_embedding,
                location_logits=location_logits,
            )
            r4207_seed_center = seed["center"]
            r4207_seed_score = seed["score"]
            r4207_seed_valid = seed["valid"]
            r4207_seed_feature_finite_fraction = seed["feature_finite_fraction"]
            r4207_seed_valid_fraction = seed["valid_fraction"]
            r4207_seed_score_mean = seed["score_mean"]
            r4207_seed_pairwise_distance_px = seed["pairwise_distance_px"]
            if r4211_enabled:
                chosen = seed["chosen"]
                r4211_proposal_seed_count = chosen.to(pixel_embedding.dtype).sum(dim=1).mean().detach()
                chosen_count = chosen.to(pixel_embedding.dtype).sum().clamp_min(1.0)
                r4211_proposal_confidence_mean = (
                    seed["score"] * chosen.to(pixel_embedding.dtype)
                ).sum().div(chosen_count).detach()
            if r4210_enabled:
                r4210_seed_logit = torch.where(
                    r4207_seed_valid, seed["logit"], torch.zeros_like(seed["logit"])
                )
                r4210_valid_seed_count = r4207_seed_valid.to(pixel_embedding.dtype).sum(dim=1).mean().detach()
                if bool(r4207_seed_valid.any().item()):
                    r4210_seed_logit_mean = seed["logit"][r4207_seed_valid].mean().detach()

        if self.dynamic_visual_instance_binding_r4207_enabled and not self.disable_visual_seed_identity_r4212_enabled:
            # Historical R4.20.7 treatment kept exactly for causal comparison.
            seeded_query = F.normalize(learned_query + seed["feature"], dim=2)
            q0 = torch.where(r4207_seed_valid[:, :, None], seeded_query, q0_control)
        elif r4208_enabled:
            # A2: normalize both branches before fusion.  The existing R4.17
            # peak probability is the only strength signal; no alpha/threshold
            # is introduced.  Invalid/missing seeds reduce exactly to the
            # R4.20.5 learned-query control.
            learned_unit = q0_control
            raw_learned_norm = learned_query.norm(dim=2)
            raw_seed_norm = seed["feature"].norm(dim=2)
            seed_unit = F.normalize(seed["feature"], dim=2)
            seed_confidence = torch.where(
                r4207_seed_valid,
                r4207_seed_score.clamp(0.0, 1.0),
                torch.zeros_like(r4207_seed_score),
            )
            identity_key = seed_confidence[:, :, None] * seed_unit
            q0 = F.normalize(learned_unit + identity_key, dim=2)
            q0 = torch.where(r4207_seed_valid[:, :, None], q0, learned_unit)

            valid = r4207_seed_valid
            if bool(valid.any().item()):
                r4208_learned_query_norm = raw_learned_norm[valid].mean().detach()
                r4208_seed_feature_norm = raw_seed_norm[valid].mean().detach()
                r4208_seed_to_learned_norm_ratio = (
                    raw_seed_norm[valid] / raw_learned_norm[valid].clamp_min(eps)
                ).mean().detach()
                r4208_q0_seed_identity_cosine = (
                    q0 * seed_unit
                ).sum(dim=2)[valid].mean().detach()
            else:
                r4208_learned_query_norm = raw_learned_norm.mean().detach()
            if k > 1:
                seed_sim = torch.einsum("bkd,bjd->bkj", seed_unit, seed_unit)
                eye = torch.eye(k, device=pixel_embedding.device, dtype=torch.bool)[None]
                pair_valid = valid[:, :, None] & valid[:, None, :] & (~eye)
                if bool(pair_valid.any().item()):
                    r4208_seed_feature_pairwise_cosine = seed_sim[pair_valid].mean().detach()
        else:
            # Bit-identical R4.20.5 control: no seed extraction and no changed
            # query arithmetic when both query treatments are disabled.
            q0 = q0_control
        residual_probability = torch.sigmoid(occupancy_logits[:, 0]).clamp(eps, 1.0 - eps)

        # R4.21.0 independent capacity factorization.  Identity among editable
        # slots is a K-way simplex.  Overflow is a separate Bernoulli variable
        # conditioned on residual, so improving overflow can no longer steal
        # relative probability *within* the K-slot identity simplex.
        overflow_gate_logits = pixel_embedding.new_zeros((b, 1, h, w))
        overflow_gate_probability = pixel_embedding.new_zeros((b, 1, h, w))
        if r4210_independent_overflow_enabled:
            if self.r4210_overflow_gate_head is None:
                raise RuntimeError("V552-R4.21.0 independent overflow head is missing")
            overflow_gate_logits = self.r4210_overflow_gate_head(pixel_embedding)
            overflow_gate_probability = torch.sigmoid(overflow_gate_logits).clamp(eps, 1.0 - eps)
            r4210_overflow_logit_mean = overflow_gate_logits.mean().detach()
            r4210_overflow_conditional_mean = overflow_gate_probability.mean().detach()

        def appearance_assignment(
            query: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            editable_scores = torch.einsum("bkd,bdhw->bkhw", query, pix) * scale
            if self.independent_candidate_set_r4212_enabled:
                # R4.21.2: each query predicts an independent residual mask.
                # There is deliberately no normalization across slots: a false
                # or duplicate query can no longer steal probability mass from
                # a correct query.  Existence/no-object is handled downstream
                # by the slot presence logit, not by forcing every pixel to
                # belong to one of K identities.
                editable_probability = torch.sigmoid(editable_scores).clamp(eps, 1.0 - eps)
                overflow_probability = editable_scores.new_zeros((b, 1, h, w))
                all_probability = editable_probability
            elif r4210_independent_overflow_enabled:
                # pi_k sums to one over editable identities only.  The complete
                # conditional residual assignment is [(1-o)pi_1..(1-o)pi_K,o].
                editable_probability = torch.softmax(editable_scores, dim=1)
                overflow_probability = overflow_gate_probability
                all_probability = torch.cat(
                    [
                        (1.0 - overflow_probability) * editable_probability,
                        overflow_probability,
                    ],
                    dim=1,
                )
            elif self.capacity_consistent_overflow_r4205_enabled:
                # Historical R4.20.5 K+1 shared simplex retained for controls.
                dustbin_score = editable_scores.new_zeros((b, 1, h, w))
                all_scores = torch.cat([editable_scores, dustbin_score], dim=1)
                all_probability = torch.softmax(all_scores, dim=1)
                editable_probability = all_probability[:, :k]
                overflow_probability = all_probability[:, k:k + 1]
            else:
                editable_probability = torch.softmax(editable_scores, dim=1)
                overflow_probability = editable_scores.new_zeros((b, 1, h, w))
                all_probability = editable_probability
            return (
                editable_scores,
                editable_probability,
                overflow_probability,
                all_probability,
            )

        score0, identity0, overflow_identity0, assignment0 = appearance_assignment(q0)
        geometry_slot0 = residual_probability[:, None] * identity0
        if r4210_independent_overflow_enabled:
            effective_slot0 = geometry_slot0 * (1.0 - overflow_identity0)
        else:
            effective_slot0 = geometry_slot0
        # R4.21.1 binding/mask geometry is independent of capacity rejection.
        # Overflow can change effective deployment mass, but its gradients and
        # current prediction cannot change which pixels are pooled into q1.
        slot0 = geometry_slot0 if r4211_geometry_overflow_enabled else effective_slot0
        overflow0 = residual_probability[:, None] * overflow_identity0
        mass0 = slot0.flatten(2).sum(dim=2).clamp_min(eps)
        evidence = torch.einsum("bkhw,bdhw->bkd", slot0, pix) / mass0[:, :, None]
        if self.persistent_identity_r4208_enabled:
            # A3: update only adaptive content, then re-inject the immutable
            # image-conditioned identity key before every full-image assignment.
            content1 = F.normalize(content0 + evidence, dim=2)
            q1 = F.normalize(content1 + identity_key, dim=2)
        else:
            q1 = F.normalize(q0 + evidence, dim=2)

        if r4208_enabled and bool(r4207_seed_valid.any().item()):
            r4208_q1_seed_identity_cosine = (
                q1 * seed_unit
            ).sum(dim=2)[r4207_seed_valid].mean().detach()
            r4208_identity_retention_delta = (
                r4208_q1_seed_identity_cosine - r4208_q0_seed_identity_cosine
            ).detach()

        def offdiag_query_cosine(query: torch.Tensor) -> torch.Tensor:
            if k <= 1:
                return query.new_zeros(())
            sim = torch.einsum("bkd,bjd->bkj", query, query)
            eye = torch.eye(k, device=query.device, dtype=torch.bool)[None]
            return sim.masked_select(~eye).mean()

        r4207_q0_pairwise_cosine = offdiag_query_cosine(q0).detach()
        r4207_q1_pairwise_cosine = offdiag_query_cosine(q1).detach()

        score1, appearance_probability, overflow_identity_probability, full_assignment = (
            appearance_assignment(q1)
        )
        # R4.20.6 supervises the same conditional categorical variable that
        # R4.20.5 predicts.  Expose its raw K+1 logits so the loss can apply a
        # standard residual-masked cross entropy after permutation matching.
        # The overflow/dustbin remains a fixed zero reference logit; exposing
        # this tensor adds no parameter and consumes no RNG.
        if r4210_independent_overflow_enabled:
            # Expose exact log-probabilities only as a compatibility diagnostic.
            # R4.20.6 CE is forbidden by the R4.21.0 protocol.
            r4205_conditional_identity_logits = full_assignment.clamp_min(eps).log()
        else:
            r4205_conditional_identity_logits = (
                torch.cat([score1, score1.new_zeros((b, 1, h, w))], dim=1)
                if self.capacity_consistent_overflow_r4205_enabled
                else score1
            )

        yy = torch.linspace(0.0, 1.0, h, device=pix.device, dtype=dtype)
        xx = torch.linspace(0.0, 1.0, w, device=pix.device, dtype=dtype)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        x = gx[None, None]
        y = gy[None, None]

        # Moments remain diagnostics for the legacy R4.20.4 control.  They are
        # not used as a Native prior in R4.20.5.
        cx0 = (slot0 * x).flatten(2).sum(dim=2) / mass0
        cy0 = (slot0 * y).flatten(2).sum(dim=2) / mass0
        dx0 = x - cx0[:, :, None, None]
        dy0 = y - cy0[:, :, None, None]
        min_var_x = 1.0 / float(max(w - 1, 1) ** 2)
        min_var_y = 1.0 / float(max(h - 1, 1) ** 2)
        var_x = (
            (slot0 * dx0.square()).flatten(2).sum(dim=2) / mass0
        ).clamp_min(min_var_x)
        var_y = (
            (slot0 * dy0.square()).flatten(2).sum(dim=2) / mass0
        ).clamp_min(min_var_y)

        if self.r4204_spatial_identity_enabled:
            spatial_log_likelihood = -0.5 * (
                dx0.square() / var_x[:, :, None, None]
                + dy0.square() / var_y[:, :, None, None]
                + var_x[:, :, None, None].log()
                + var_y[:, :, None, None].log()
            )
            combined_log_probability = (
                F.log_softmax(score1, dim=1)
                + F.log_softmax(spatial_log_likelihood, dim=1)
            )
            identity_probability = torch.softmax(combined_log_probability, dim=1)
            overflow_identity_probability = score1.new_zeros((b, 1, h, w))
            full_assignment = identity_probability
        else:
            identity_probability = appearance_probability

        # R4.20.5 leaves residual mass not claimed by editable components in a
        # non-editable overflow category.  In the R4.20.4 control this tensor is
        # exactly zero, preserving the original formulation.
        geometry_slot_prob = residual_probability[:, None] * identity_probability
        if r4210_independent_overflow_enabled:
            effective_slot_prob = geometry_slot_prob * (1.0 - overflow_identity_probability)
        else:
            effective_slot_prob = geometry_slot_prob
        slot_prob = effective_slot_prob
        r4211_geometry_slot_probability = geometry_slot_prob
        r4211_effective_slot_probability = effective_slot_prob
        r4211_geometry_effective_l1 = (geometry_slot_prob - effective_slot_prob).abs().mean().detach()
        overflow_probability = (
            residual_probability[:, None] * overflow_identity_probability
        )
        background = (1.0 - residual_probability)[:, None]
        ownership = (
            torch.cat([background, slot_prob, overflow_probability], dim=1)
            if self.capacity_consistent_overflow_r4205_enabled
            else torch.cat([background, slot_prob], dim=1)
        )

        ownership_sum_error = (ownership.sum(dim=1) - 1.0).abs().mean()
        residual_mass_conservation_error = (
            slot_prob.sum(dim=1)
            + overflow_probability[:, 0]
            - residual_probability
        ).abs().mean()

        # R4.21.1 introduces two explicitly different objects:
        #   geometry_owner = G_k = r*pi_k, used by binding and mask matching;
        #   effective_slot = (1-o)G_k, used by deployable correction.
        # This removes the remaining path by which overflow prediction changed
        # instance geometry.  Historical paths retain effective masks as logits.
        geometry_owner = geometry_slot_prob if r4211_geometry_overflow_enabled else slot_prob
        geometry_owner_for_logits = geometry_owner.clamp(eps, 1.0 - eps)
        final_logits = float(self.mask_temperature) * torch.logit(geometry_owner_for_logits)
        final_logits_finite_fraction = torch.isfinite(final_logits).to(dtype).mean()

        mass = geometry_owner.flatten(2).sum(dim=2).clamp_min(eps)
        cx = (geometry_owner * x).flatten(2).sum(dim=2) / mass
        cy = (geometry_owner * y).flatten(2).sum(dim=2) / mass
        zero_extent = torch.zeros_like(cx)
        soft_anchor = torch.stack([cx, cy, zero_extent, zero_extent], dim=2)
        soft_point = torch.stack([cx, cy], dim=2)
        if self.dynamic_visual_instance_binding_r4207_enabled:
            drift_x = (soft_point[:, :, 0] - r4207_seed_center[:, :, 0]) * float(max(w, 1))
            drift_y = (soft_point[:, :, 1] - r4207_seed_center[:, :, 1]) * float(max(h, 1))
            drift_px = torch.sqrt(drift_x.square() + drift_y.square() + 1.0e-12)
            r4207_seed_to_slot_centroid_drift_px = torch.where(
                r4207_seed_valid.any(),
                drift_px[r4207_seed_valid].mean(),
                drift_px.new_zeros(()),
            ).detach()
        else:
            r4207_seed_to_slot_centroid_drift_px = pixel_embedding.new_zeros(())

        type_score = torch.einsum(
            "bkhw,bchw->bkc", geometry_owner, typed_center_logits
        ) / mass[:, :, None]
        proposal_type = type_score.argmax(dim=2)
        proposal_score = (
            geometry_owner * residual_probability[:, None]
        ).flatten(2).sum(dim=2) / mass
        proposal_valid = torch.ones_like(proposal_score)

        # Entropy is defined over the complete conditional assignment in R4.20.5
        # (K editable categories + dustbin) and over K slots in the R4.20.4
        # control.  This quantity remains diagnostic only.
        assignment_categories = k + (1 if self.capacity_consistent_overflow_r4205_enabled else 0)
        conditional_entropy_pixel = -(
            full_assignment.clamp_min(eps)
            * full_assignment.clamp_min(eps).log()
        ).sum(dim=1) / math.log(float(max(assignment_categories, 2)))
        residual_mass_total = residual_probability.sum().clamp_min(eps)
        conditional_slot_entropy = (
            conditional_entropy_pixel * residual_probability
        ).sum() / residual_mass_total
        conditional_max_slot_probability = (
            full_assignment.max(dim=1).values * residual_probability
        ).sum() / residual_mass_total

        slot_mass_fraction = slot_prob.flatten(2).mean(dim=2)
        slot_mass_mean = slot_mass_fraction.mean(dim=1, keepdim=True).clamp_min(eps)
        slot_mass_cv = (
            slot_mass_fraction.std(dim=1, unbiased=False) / slot_mass_mean[:, 0]
        ).mean()
        max_slot_ownership = slot_prob.max(dim=1).values.mean()

        centroid = torch.stack([cx, cy], dim=2)
        pair_distance = torch.cdist(centroid, centroid, p=2)
        if k > 1:
            eye = torch.eye(k, device=pair_distance.device, dtype=torch.bool)[None]
            centroid_separation = pair_distance.masked_select(~eye).mean()
        else:
            centroid_separation = pair_distance.sum() * 0.0
        spatial_variance_mean = 0.5 * (var_x.mean() + var_y.mean())

        editable_probability_sum = slot_prob.sum(dim=1, keepdim=True)
        return {
            "final_logits": final_logits,
            "slot_probability": slot_prob,
            "background_probability": background,
            "residual_probability": residual_probability[:, None],
            "overflow_probability": overflow_probability,
            "overflow_identity_probability": overflow_identity_probability,
            "r4205_conditional_identity_logits": r4205_conditional_identity_logits,
            "editable_probability_sum": editable_probability_sum,
            "occupancy_logits": occupancy_logits,
            "refined_query": q1,
            "soft_anchor": soft_anchor,
            "soft_point": soft_point,
            "proposal_type": proposal_type,
            "proposal_score": proposal_score,
            "proposal_valid": proposal_valid,
            "ownership_sum_error": ownership_sum_error.detach(),
            "residual_mass_conservation_error": residual_mass_conservation_error.detach(),
            "assignment_entropy": conditional_slot_entropy.detach(),
            "conditional_slot_entropy": conditional_slot_entropy.detach(),
            "conditional_max_slot_probability": conditional_max_slot_probability.detach(),
            "background_fraction": background.mean().detach(),
            "residual_existence_mean": residual_probability.mean().detach(),
            "max_slot_ownership": max_slot_ownership.detach(),
            "slot_mass_cv": slot_mass_cv.detach(),
            "centroid_separation": centroid_separation.detach(),
            "spatial_variance_mean": spatial_variance_mean.detach(),
            "spatial_identity_enabled": final_logits.new_tensor(
                1.0 if self.r4204_spatial_identity_enabled else 0.0
            ),
            "location_as_occupancy_used": final_logits.new_zeros(()),
            "r4205_enabled": final_logits.new_tensor(
                1.0 if self.capacity_consistent_overflow_r4205_enabled else 0.0
            ),
            "r4205_overflow_probability_mean": overflow_probability.mean().detach(),
            "r4205_overflow_conditional_mean": overflow_identity_probability.mean().detach(),
            "r4205_editable_probability_mean": editable_probability_sum.mean().detach(),
            "r4205_final_logits_finite_fraction": final_logits_finite_fraction.detach(),
            "r4207_enabled": final_logits.new_tensor(
                1.0 if self.dynamic_visual_instance_binding_r4207_enabled else 0.0
            ),
            "r4207_seed_center_xy": r4207_seed_center.detach(),
            "r4207_seed_score": r4207_seed_score.detach(),
            "r4207_seed_valid": r4207_seed_valid.to(final_logits.dtype).detach(),
            "r4207_seed_feature_finite_fraction": r4207_seed_feature_finite_fraction,
            "r4207_seed_valid_fraction": r4207_seed_valid_fraction,
            "r4207_seed_score_mean": r4207_seed_score_mean,
            "r4207_seed_pairwise_distance_px": r4207_seed_pairwise_distance_px,
            "r4207_q0_pairwise_cosine": r4207_q0_pairwise_cosine,
            "r4207_q1_pairwise_cosine": r4207_q1_pairwise_cosine,
            "r4207_seed_to_slot_centroid_drift_px": r4207_seed_to_slot_centroid_drift_px,
            "r4207_full_image_assignment_enabled": final_logits.new_ones(()),
            "r4207_hard_spatial_support_used": final_logits.new_zeros(()),
            "r4208_enabled": final_logits.new_tensor(1.0 if r4208_enabled else 0.0),
            "r4208_normalized_fusion_enabled": final_logits.new_tensor(1.0 if self.normalized_visual_instance_binding_r4208_enabled else 0.0),
            "r4208_persistent_identity_enabled": final_logits.new_tensor(1.0 if self.persistent_identity_r4208_enabled else 0.0),
            "r4208_learned_query_norm": r4208_learned_query_norm,
            "r4208_seed_feature_norm": r4208_seed_feature_norm,
            "r4208_seed_to_learned_norm_ratio": r4208_seed_to_learned_norm_ratio,
            "r4208_seed_feature_pairwise_cosine": r4208_seed_feature_pairwise_cosine,
            "r4208_q0_seed_identity_cosine": r4208_q0_seed_identity_cosine,
            "r4208_q1_seed_identity_cosine": r4208_q1_seed_identity_cosine,
            "r4208_identity_retention_delta": r4208_identity_retention_delta,
            "r4210_enabled": final_logits.new_tensor(1.0 if r4210_enabled else 0.0),
            "r4210_variable_seed_enabled": final_logits.new_tensor(1.0 if r4210_variable_seed_enabled else 0.0),
            "r4210_independent_overflow_enabled": final_logits.new_tensor(1.0 if r4210_independent_overflow_enabled else 0.0),
            "r4210_valid_seed_count": r4210_valid_seed_count,
            "r4210_seed_logit": r4210_seed_logit,
            "r4210_seed_logit_mean": r4210_seed_logit_mean,
            "r4210_overflow_gate_logits": overflow_gate_logits,
            "r4210_overflow_conditional_probability": overflow_gate_probability,
            "r4210_overflow_logit_mean": r4210_overflow_logit_mean,
            "r4210_overflow_conditional_mean": r4210_overflow_conditional_mean,
            "r4211_enabled": final_logits.new_tensor(1.0 if r4211_enabled else 0.0),
            "r4211_proposal_existence_enabled": final_logits.new_tensor(1.0 if r4211_proposal_existence_enabled else 0.0),
            "r4211_geometry_overflow_enabled": final_logits.new_tensor(1.0 if r4211_geometry_overflow_enabled else 0.0),
            "r4211_proposal_seed_count": r4211_proposal_seed_count,
            "r4211_proposal_confidence_mean": r4211_proposal_confidence_mean,
            "r4211_geometry_effective_l1": r4211_geometry_effective_l1,
            "r4211_geometry_slot_probability": r4211_geometry_slot_probability,
            "r4211_effective_slot_probability": r4211_effective_slot_probability,
        }

    def _r420_dynamic_mask_logits(
        self,
        *,
        pixel_embedding: torch.Tensor,
        query: torch.Tensor,
        anchor: torch.Tensor,
    ) -> torch.Tensor:
        """Generate one box-free residual mask per location-conditioned query.

        This follows the core CondInst principle: a shared mask feature is
        augmented with coordinates relative to each instance/reference point,
        while a controller emits query-specific 1x1-convolution parameters.
        Every image pixel participates differentiably in the mask objective;
        there is no hard seed support or detached frontier-growth decision.
        Width/height fields in ``anchor`` are intentionally ignored.
        """
        if self.r420_mask_feature_proj is None or self.r420_dynamic_controller is None:
            raise RuntimeError("V552-R4.20 dynamic mask head requested but modules are absent")
        b, _, h, w = pixel_embedding.shape
        k = query.shape[1]
        feature = self.r420_mask_feature_proj(pixel_embedding)
        c = feature.shape[1]

        yy = torch.linspace(0.0, 1.0, h, device=feature.device, dtype=feature.dtype)
        xx = torch.linspace(0.0, 1.0, w, device=feature.device, dtype=feature.dtype)
        y_grid = yy[:, None].expand(h, w)[None, None]
        x_grid = xx[None, :].expand(h, w)[None, None]
        cx = anchor[:, :, 0, None, None].to(feature.dtype)
        cy = anchor[:, :, 1, None, None].to(feature.dtype)
        rel_x = (x_grid - cx).unsqueeze(2)
        rel_y = (y_grid - cy).unsqueeze(2)
        shared = feature[:, None].expand(-1, k, -1, -1, -1)
        x = torch.cat([shared, rel_x, rel_y], dim=2)

        params = self.r420_dynamic_controller(query)
        parts = torch.split(params, self.r420_dynamic_param_sizes, dim=2)
        hidden = self.r420_dynamic_channels
        input_channels = c + 2
        w1 = parts[0].reshape(b, k, hidden, input_channels)
        b1 = parts[1].reshape(b, k, hidden, 1, 1)
        w2 = parts[2].reshape(b, k, hidden, hidden)
        b2 = parts[3].reshape(b, k, hidden, 1, 1)
        w3 = parts[4].reshape(b, k, 1, hidden)
        b3 = parts[5].reshape(b, k, 1, 1, 1)

        x = F.relu(torch.einsum("bkoc,bkchw->bkohw", w1, x) + b1, inplace=False)
        x = F.relu(torch.einsum("bkoc,bkchw->bkohw", w2, x) + b2, inplace=False)
        logits = torch.einsum("bkoc,bkchw->bkohw", w3, x) + b3
        return logits[:, :, 0]

    def _r48_decode_queries(
        self,
        *,
        pixel_embedding: torch.Tensor,
        query: torch.Tensor,
        anchor: torch.Tensor,
        coarse_logits: Optional[torch.Tensor],
        query_type: Optional[torch.Tensor] = None,
        base_probability: Optional[torch.Tensor] = None,
        boundary: Optional[torch.Tensor] = None,
        freeze_anchor: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Iteratively bind queries to local residual components.

        R4.9 keeps R4.8's sparse anchor-relative sampling but fixes two
        structural errors exposed by the SMOKE20 log:
        1) normalized q/k are *not* divided by sqrt(d) again; scaled cosine
           compatibility with a learnable bounded temperature selects content;
        2) reference anchors guide sampling/geometry only and are not added
           directly to segmentation-mask logits.
        """
        b, d, h, w = pixel_embedding.shape
        k = query.shape[1]
        yy = torch.linspace(0.0, 1.0, h, device=pixel_embedding.device, dtype=pixel_embedding.dtype)
        xx = torch.linspace(0.0, 1.0, w, device=pixel_embedding.device, dtype=pixel_embedding.dtype)
        y_grid = yy[:, None].expand(h, w)
        x_grid = xx[None, :].expand(h, w)

        def render(
            q: torch.Tensor,
            a: torch.Tensor,
            spatial_support: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            if (
                (not self.box_free_mask_set_r418_enabled)
                and self.native_residual_set_r411_enabled
                and self.r411_local_roi_decoder_enabled
            ):
                return self._r411_render_local_masks(
                    pixel_embedding=pixel_embedding, query=q, anchor=a,
                    query_type=query_type, base_probability=base_probability, boundary=boundary,
                )
            mask_query = F.normalize(self.r48_mask_query_proj(q), dim=2)
            logits = torch.einsum("bkd,bdhw->bkhw", mask_query, pixel_embedding)
            logits = logits * self.r47_query_logit_scale
            if (
                coarse_logits is not None
                and not self.r48_remove_coarse_mask_bias
                and coarse_logits.shape[1] == logits.shape[1]
            ):
                logits = logits + coarse_logits
            if (
                (not self.box_free_mask_set_r418_enabled)
                and not (self.content_selective_r49_enabled and self.r49_anchor_sampling_only)
            ):
                logits = logits + self._r48_window_prior(a, x_grid, y_grid)
            if self.evidence_proposal_r410_enabled and self.r410_support_only_local_readout:
                logits = logits + self._r410_support_penalty(a, x_grid, y_grid)
            if self.seeded_masked_attention_r419_enabled and spatial_support is not None:
                # R4.19 keeps the mask representation box-free but prevents the
                # R4.18 whole-image similarity readout from activating remote
                # ultrasound texture.  Support is owned by a location seed and
                # then by the previous predicted mask, never by width/height.
                logits = logits - self.r419_outside_logit_penalty * (
                    1.0 - spatial_support.to(logits.dtype)
                )
            return logits

        r419_seed_support = pixel_embedding.new_ones((b, k, h, w))
        r419_support = r419_seed_support
        r419_seed_fraction = pixel_embedding.new_zeros(())
        r419_final_fraction = pixel_embedding.new_zeros(())
        r419_outside_probability = pixel_embedding.new_zeros(())
        if self.seeded_masked_attention_r419_enabled:
            # The R4.17 point is a *reference point*, not a geometry box.  A
            # circular physical seed gives the first query update a local search
            # region.  Subsequent supports can only grow where the preceding mask
            # predicts foreground, implementing Mask2Former-style masked attention
            # specialized for tiny residuals.
            cx = anchor[:, :, 0, None, None]
            cy = anchor[:, :, 1, None, None]
            dx = (x_grid[None, None] - cx) * float(max(w, 1))
            dy = (y_grid[None, None] - cy) * float(max(h, 1))
            r2 = dx.square() + dy.square()
            r419_seed_support = (r2 <= self.r419_seed_radius_px ** 2).to(pixel_embedding.dtype)
            r419_support = r419_seed_support
            r419_seed_fraction = r419_seed_support.mean().detach()

        stage_logits = [render(query, anchor, r419_support if self.seeded_masked_attention_r419_enabled else None)]
        stage_anchors = [anchor]
        entropy_terms = []
        max_weight_terms = []
        source = pixel_embedding[:, None].expand(-1, k, -1, -1, -1).reshape(b * k, d, h, w)
        base_offsets = self.r48_base_sample_offsets.to(pixel_embedding.dtype)
        p = base_offsets.shape[0]
        for layer in range(self.r48_decoder_layers):
            if self.box_free_mask_set_r418_enabled:
                current_logits = stage_logits[-1]
                if self.seeded_masked_attention_r419_enabled:
                    # Use the current residual probability only *inside* the
                    # allowed support.  A small uniform term inside the support
                    # prevents zero-context collapse before the mask is calibrated.
                    current_prob = torch.sigmoid(current_logits.detach())
                    weighted_support = r419_support * (0.05 + 0.95 * current_prob)
                    normalizer = weighted_support.flatten(2).sum(dim=2, keepdim=True).clamp_min(EPS)
                    mask_attention = weighted_support.flatten(2) / normalizer
                else:
                    # R4.18 control: unrestricted full-image mask pooling.
                    mask_attention = F.softmax(current_logits.flatten(2), dim=2)
                pixel_flat = pixel_embedding.flatten(2)
                context = torch.einsum("bkp,bdp->bkd", mask_attention, pixel_flat)
                entropy_terms.append(
                    (-(mask_attention * mask_attention.clamp_min(EPS).log()).sum(dim=2)).mean()
                )
                max_weight_terms.append(mask_attention.max(dim=2).values.mean())
                query = self.r48_query_norm1[layer](query + self.r48_context_projs[layer](context))
                query = self.r48_query_norm2[layer](query + self.r48_ffns[layer](query))
                if self.seeded_masked_attention_r419_enabled:
                    # Expansion is mask-driven rather than box-driven.  Evaluate
                    # the updated query *without* the support penalty only on the
                    # one-step dilated frontier, then admit positive pixels into
                    # the next support. Remote locations can never jump across
                    # the frontier, while irregular residuals can grow outward.
                    raw_next_logits = render(query, anchor, None)
                    predicted = (
                        torch.sigmoid(raw_next_logits.detach()) >= self.r419_mask_threshold
                    ).to(pixel_embedding.dtype)
                    dilated_support = F.max_pool2d(
                        r419_support.reshape(b * k, 1, h, w),
                        self.r419_support_dilate_kernel,
                        stride=1,
                        padding=self.r419_support_dilate_kernel // 2,
                    ).reshape(b, k, h, w)
                    frontier = (dilated_support - r419_support).clamp(0.0, 1.0)
                    frontier_positive = predicted * frontier
                    r419_support = torch.maximum(r419_support, frontier_positive)
                    next_logits = render(query, anchor, r419_support)
                else:
                    next_logits = render(query, anchor)
                stage_logits.append(next_logits)
                stage_anchors.append(anchor)
                continue
            predicted_offset = torch.tanh(self.r48_offset_heads[layer](query)).reshape(b, k, p, 2)
            relative = (base_offsets[None, None] + 0.5 * predicted_offset).clamp(-1.5, 1.5)
            sample_xy = anchor[:, :, None, :2] + 0.5 * anchor[:, :, None, 2:] * relative
            sample_xy = sample_xy.clamp(0.0, 1.0)
            sample_grid = (2.0 * sample_xy - 1.0).reshape(b * k, p, 1, 2)
            sampled = F.grid_sample(
                source,
                sample_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )[:, :, :, 0].transpose(1, 2).reshape(b, k, p, d)

            if self.content_selective_r49_enabled and self.r49_dn_curriculum_enabled:
                q_key = F.normalize(self.r49_query_key_projs[layer](query), dim=2)
                k_key = F.normalize(self.r49_sample_key_projs[layer](sampled), dim=3)
                logit_scale = self.r49_logit_scale[layer].exp().clamp(
                    max=self.r49_attention_logit_scale_max
                )
                attention_logits = (q_key[:, :, None, :] * k_key).sum(dim=3) * logit_scale
                value = self.r49_sample_value_projs[layer](sampled)
            else:
                attention_logits = (
                    sampled * F.normalize(query, dim=2)[:, :, None]
                ).sum(dim=3) / math.sqrt(float(d))
                value = sampled

            attention = F.softmax(attention_logits, dim=2)
            entropy_terms.append((-(attention * attention.clamp_min(EPS).log()).sum(dim=2)).mean())
            max_weight_terms.append(attention.max(dim=2).values.mean())
            context = (attention[:, :, :, None] * value).sum(dim=2)
            query = self.r48_query_norm1[layer](query + self.r48_context_projs[layer](context))
            query = self.r48_query_norm2[layer](query + self.r48_ffns[layer](query))

            if not freeze_anchor:
                delta = torch.tanh(self.r48_anchor_heads[layer](query)) * self.r48_anchor_delta_scale
                center = torch.sigmoid(
                    torch.logit(anchor[:, :, :2].clamp(1.0e-4, 1.0 - 1.0e-4))
                    + delta[:, :, :2]
                )
                size_unit = ((anchor[:, :, 2:] - self.r47_anchor_min_size) / max(
                    self.r47_anchor_max_size - self.r47_anchor_min_size, 1.0e-4
                )).clamp(1.0e-4, 1.0 - 1.0e-4)
                size_unit = torch.sigmoid(torch.logit(size_unit) + delta[:, :, 2:])
                size = self.r47_anchor_min_size + (
                    self.r47_anchor_max_size - self.r47_anchor_min_size
                ) * size_unit
                anchor = torch.cat([center, size], dim=2)
            stage_logits.append(render(query, anchor))
            stage_anchors.append(anchor)

        mean_attention_entropy = (
            torch.stack(entropy_terms).mean() if entropy_terms else pixel_embedding.new_zeros(())
        )
        mean_attention_max_weight = (
            torch.stack(max_weight_terms).mean() if max_weight_terms else pixel_embedding.new_zeros(())
        )
        if self.seeded_masked_attention_r419_enabled:
            r419_final_fraction = r419_support.mean().detach()
            final_probability = torch.sigmoid(stage_logits[-1].detach())
            outside = 1.0 - r419_support
            r419_outside_probability = (
                (final_probability * outside).sum() / outside.sum().clamp_min(1.0)
            ).detach()
        return (
            stage_logits[-1],
            stage_anchors[-1],
            torch.stack(stage_logits, dim=1),
            torch.stack(stage_anchors, dim=1),
            mean_attention_entropy,
            mean_attention_max_weight,
            r419_seed_fraction,
            r419_final_fraction,
            r419_outside_probability,
        )

    @staticmethod
    def _r418_zero_shift(mask: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
        """Translate a binary mask without circular wrap-around."""
        out = torch.zeros_like(mask)
        h, w = mask.shape[-2:]
        src_x0 = max(-dx, 0); src_x1 = min(w - dx, w)
        dst_x0 = max(dx, 0);  dst_x1 = min(w + dx, w)
        src_y0 = max(-dy, 0); src_y1 = min(h - dy, h)
        dst_y0 = max(dy, 0);  dst_y1 = min(h + dy, h)
        if src_x1 > src_x0 and src_y1 > src_y0:
            out[..., dst_y0:dst_y1, dst_x0:dst_x1] = mask[..., src_y0:src_y1, src_x0:src_x1]
        return out

    def _r418_build_paired_coarse(self, supervision_masks: torch.Tensor) -> torch.Tensor:
        """Build a training-only coarse mask whose residual target is exact.

        This is ordinary mask corruption (erosion/dilation/one-pixel shifts),
        not a second teacher network.  Crucially, the corrupted mask itself is
        fed to the same refiner that is supervised on its residual, eliminating
        the historical input/teacher mismatch.  The corruption mode rotates by
        epoch and sample index, i.e. it is augmentation of a stationary
        conditional task rather than a persistent slot identity.
        """
        # R4.18.1: normalize real DataLoader masks to BCHW.
        # BUSI training may provide [B,H,W], while unit contracts previously
        # exercised only [B,1,H,W].
        if supervision_masks.ndim == 3:
            gt_input = supervision_masks.unsqueeze(1)
        elif supervision_masks.ndim == 4:
            gt_input = supervision_masks[:, :1]
        else:
            raise RuntimeError(
                "V552-R4.18 paired teacher requires supervision_masks "
                f"with shape [B,H,W] or [B,1,H,W], got "
                f"{tuple(supervision_masks.shape)}"
            )

        gt = (gt_input >= 0.5).to(supervision_masks.dtype)

        radius = 1 + ((int(self.current_epoch) // 2) % 2)
        kernel = 2 * radius + 1
        dilated = F.max_pool2d(gt, kernel, stride=1, padding=radius)
        eroded = 1.0 - F.max_pool2d(1.0 - gt, kernel, stride=1, padding=radius)
        shift_x = self._r418_zero_shift(gt, dx=radius, dy=0)
        shift_y = self._r418_zero_shift(gt, dx=0, dy=radius)
        variants = torch.stack([eroded, dilated, shift_x, shift_y], dim=1)
        b = gt.shape[0]
        batch = torch.arange(b, device=gt.device)
        mode = (batch + int(self.current_epoch)) % variants.shape[1]
        paired = variants[batch, mode]
        return paired.clamp(0.0, 1.0)

    def _r48_iterative_query_masks(
        self,
        *,
        coarse_logits: torch.Tensor,
        fused_feature: torch.Tensor,
        base_probability: torch.Tensor,
        cause_probability: torch.Tensor,
        action_alpha: torch.Tensor,
        entropy: torch.Tensor,
        boundary: torch.Tensor,
        supervision_masks: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        b, _, h, w = coarse_logits.shape
        cause = cause_probability[:, :4].detach()
        alpha = action_alpha[:, :4].detach()
        yy = torch.linspace(0.0, 1.0, h, device=coarse_logits.device, dtype=coarse_logits.dtype)
        xx = torch.linspace(0.0, 1.0, w, device=coarse_logits.device, dtype=coarse_logits.dtype)
        y_grid = yy[:, None].expand(h, w)
        x_grid = xx[None, :].expand(h, w)
        xy = torch.stack([x_grid, y_grid], dim=0)[None].expand(b, -1, -1, -1)
        pixel_input = torch.cat(
            [
                fused_feature,
                base_probability[:, :1].detach(),
                entropy[:, :1].detach(),
                boundary[:, :1].detach(),
                cause,
                alpha,
                xy,
            ],
            dim=1,
        )
        pixel_embedding = F.normalize(self.r47_pixel_encoder(pixel_input), dim=1)
        query = F.normalize(self.r47_slot_queries.weight, dim=1)[None].expand(b, -1, -1)
        if self.r4201_clean_rootfix_enabled and (
            self.dynamic_residual_mask_r420_enabled or self._dense_set_family_enabled
        ):
            anchor = self.r4201_fixed_anchor.to(
                device=pixel_embedding.device, dtype=pixel_embedding.dtype
            )[None].expand(b, -1, -1)
        else:
            raw = torch.sigmoid(self.r47_base_anchor_logits)
            center = raw[:, :2]
            size = self.r47_anchor_min_size + (
                self.r47_anchor_max_size - self.r47_anchor_min_size
            ) * raw[:, 2:]
            anchor = torch.cat([center, size], dim=1)[None].expand(b, -1, -1)
        proposal_score = pixel_embedding.new_zeros((b, self.num_slots))
        proposal_valid = pixel_embedding.new_zeros((b, self.num_slots))
        proposal_type = torch.zeros((b, self.num_slots), device=pixel_embedding.device, dtype=torch.long)
        r411_center_logits = pixel_embedding.new_zeros((b, 4, h, w))
        r411_size_map = pixel_embedding.new_zeros((b, 4, 2, h, w))
        r411_offset_map = pixel_embedding.new_zeros((b, 4, 2, h, w))
        r416_proposal_point = pixel_embedding.new_zeros((b, self.num_slots, 2))
        r416_edge_offsets = pixel_embedding.new_zeros((b, self.num_slots, 4))
        r416_legacy_unique_fraction = pixel_embedding.new_zeros(())
        r417_location_logits = pixel_embedding.new_zeros((b, 1, h, w))
        r417_location_offset_map = pixel_embedding.new_zeros((b, 2, h, w))
        r418_paired_logits = pixel_embedding.new_zeros((b, self.num_slots, h, w))
        r418_paired_teacher_masks = pixel_embedding.new_zeros((b, self.num_slots, h, w))
        r418_paired_teacher_valid = torch.zeros((b, self.num_slots), device=pixel_embedding.device, dtype=torch.bool)
        r418_paired_coarse = base_probability[:, :1].detach()
        r419_seed_support_fraction = pixel_embedding.new_zeros(())
        r419_final_support_fraction = pixel_embedding.new_zeros(())
        r419_outside_mask_probability = pixel_embedding.new_zeros(())
        r420_relative_coord_mean_abs = pixel_embedding.new_zeros(())
        r4203_ownership_sum_error = pixel_embedding.new_zeros(())
        r4203_assignment_entropy = pixel_embedding.new_zeros(())
        r4203_background_fraction = pixel_embedding.new_zeros(())
        r4203_max_slot_ownership = pixel_embedding.new_zeros(())
        r4203_slot_mass_cv = pixel_embedding.new_zeros(())
        r4204_occupancy_logits = pixel_embedding.new_zeros((b, 1, h, w))
        r4204_residual_mass_conservation_error = pixel_embedding.new_zeros(())
        r4204_conditional_slot_entropy = pixel_embedding.new_zeros(())
        r4204_conditional_max_slot_probability = pixel_embedding.new_zeros(())
        r4204_residual_existence_mean = pixel_embedding.new_zeros(())
        r4204_centroid_separation = pixel_embedding.new_zeros(())
        r4204_spatial_variance_mean = pixel_embedding.new_zeros(())
        r4204_spatial_identity_enabled = pixel_embedding.new_zeros(())
        r4204_location_as_occupancy_used = pixel_embedding.new_zeros(())
        r4205_enabled = pixel_embedding.new_zeros(())
        r4205_overflow_probability = pixel_embedding.new_zeros((b, 1, h, w))
        r4205_overflow_identity_probability = pixel_embedding.new_zeros((b, 1, h, w))
        r4205_conditional_identity_logits = pixel_embedding.new_zeros((b, self.num_slots + 1, h, w))
        r4205_editable_probability_sum = pixel_embedding.new_zeros((b, 1, h, w))
        r4205_overflow_probability_mean = pixel_embedding.new_zeros(())
        r4205_overflow_conditional_mean = pixel_embedding.new_zeros(())
        r4205_editable_probability_mean = pixel_embedding.new_zeros(())
        r4205_final_logits_finite_fraction = pixel_embedding.new_ones(())
        r4207_enabled = pixel_embedding.new_zeros(())
        r4207_seed_center_xy = pixel_embedding.new_zeros((b, self.num_slots, 2))
        r4207_seed_score = pixel_embedding.new_zeros((b, self.num_slots))
        r4207_seed_valid = pixel_embedding.new_zeros((b, self.num_slots))
        r4207_seed_feature_finite_fraction = pixel_embedding.new_ones(())
        r4207_seed_valid_fraction = pixel_embedding.new_zeros(())
        r4207_seed_score_mean = pixel_embedding.new_zeros(())
        r4207_seed_pairwise_distance_px = pixel_embedding.new_zeros(())
        r4207_q0_pairwise_cosine = pixel_embedding.new_zeros(())
        r4207_q1_pairwise_cosine = pixel_embedding.new_zeros(())
        r4207_seed_to_slot_centroid_drift_px = pixel_embedding.new_zeros(())
        r4207_full_image_assignment_enabled = pixel_embedding.new_zeros(())
        r4207_hard_spatial_support_used = pixel_embedding.new_zeros(())
        r4208_enabled = pixel_embedding.new_zeros(())
        r4208_normalized_fusion_enabled = pixel_embedding.new_zeros(())
        r4208_persistent_identity_enabled = pixel_embedding.new_zeros(())
        r4208_learned_query_norm = pixel_embedding.new_zeros(())
        r4208_seed_feature_norm = pixel_embedding.new_zeros(())
        r4208_seed_to_learned_norm_ratio = pixel_embedding.new_zeros(())
        r4208_seed_feature_pairwise_cosine = pixel_embedding.new_zeros(())
        r4208_q0_seed_identity_cosine = pixel_embedding.new_zeros(())
        r4208_q1_seed_identity_cosine = pixel_embedding.new_zeros(())
        r4208_identity_retention_delta = pixel_embedding.new_zeros(())
        r4210_enabled = pixel_embedding.new_zeros(())
        r4210_variable_seed_enabled = pixel_embedding.new_zeros(())
        r4210_independent_overflow_enabled = pixel_embedding.new_zeros(())
        r4210_valid_seed_count = pixel_embedding.new_zeros(())
        r4210_seed_logit = pixel_embedding.new_zeros((b, self.num_slots))
        r4210_seed_logit_mean = pixel_embedding.new_zeros(())
        r4210_overflow_gate_logits = pixel_embedding.new_zeros((b, 1, h, w))
        r4210_overflow_conditional_probability = pixel_embedding.new_zeros((b, 1, h, w))
        r4210_overflow_logit_mean = pixel_embedding.new_zeros(())
        r4210_overflow_conditional_mean = pixel_embedding.new_zeros(())
        r4211_enabled = pixel_embedding.new_zeros(())
        r4211_proposal_existence_enabled = pixel_embedding.new_zeros(())
        r4211_geometry_overflow_enabled = pixel_embedding.new_zeros(())
        r4211_proposal_seed_count = pixel_embedding.new_zeros(())
        r4211_proposal_confidence_mean = pixel_embedding.new_zeros(())
        r4211_geometry_effective_l1 = pixel_embedding.new_zeros(())
        r4211_geometry_slot_probability = pixel_embedding.new_zeros((b, self.num_slots, h, w))
        r4211_effective_slot_probability = pixel_embedding.new_zeros((b, self.num_slots, h, w))
        v560_clean = None
        if self.clean_core_v560_enabled:
            v560_clean = self._v560_clean_query_set(
                pixel_embedding=pixel_embedding,
                cause_probability=cause,
            )
            query = v560_clean["refined_query"]
            anchor = v560_clean["soft_anchor"]
            proposal_score = v560_clean["proposal_score"]
            proposal_valid = v560_clean["proposal_valid"]
            proposal_type = v560_clean["proposal_type"]
            r416_proposal_point = v560_clean["soft_point"]
        elif self.dense_competitive_residual_set_r4203_enabled:
            if self.r411_proposal_stem is None or self.r411_center_head is None or self.r417_location_head is None:
                raise RuntimeError("V552-R4.20.3/4 requires live dense R4.11/R4.17 locator heads")
            proposal_feature = self.r411_proposal_stem(pixel_embedding)
            r411_center_logits = self.r411_center_head(proposal_feature)
            r417_location_logits = self.r417_location_head(proposal_feature)
            r411_size_map = pixel_embedding.new_zeros((b, 4, 2, h, w))
            r411_offset_map = pixel_embedding.new_zeros((b, 4, 2, h, w))
            r417_location_offset_map = pixel_embedding.new_zeros((b, 2, h, w))
            if self.factorized_residual_existence_identity_r4204_enabled:
                if self.r4204_residual_occupancy_head is None:
                    raise RuntimeError("V552-R4.20.4 residual occupancy head is missing")
                r4204_occupancy_logits = self.r4204_residual_occupancy_head(proposal_feature)
                dense_set = self._r4204_factorized_residual_existence_identity_set(
                    pixel_embedding=pixel_embedding,
                    occupancy_logits=r4204_occupancy_logits,
                    typed_center_logits=r411_center_logits,
                    location_logits=r417_location_logits,
                )
            else:
                dense_set = self._r4203_dense_competitive_residual_set(
                    pixel_embedding=pixel_embedding,
                    location_logits=r417_location_logits,
                    typed_center_logits=r411_center_logits,
                )
            query = dense_set["refined_query"]
            anchor = dense_set["soft_anchor"]
            proposal_score = dense_set["proposal_score"]
            proposal_valid = dense_set["proposal_valid"]
            proposal_type = dense_set["proposal_type"]
            r416_proposal_point = dense_set["soft_point"]
            r4203_ownership_sum_error = dense_set["ownership_sum_error"]
            r4203_assignment_entropy = dense_set["assignment_entropy"]
            r4203_background_fraction = dense_set["background_fraction"]
            r4203_max_slot_ownership = dense_set["max_slot_ownership"]
            r4203_slot_mass_cv = dense_set["slot_mass_cv"]
            if self.factorized_residual_existence_identity_r4204_enabled:
                r4204_residual_mass_conservation_error = dense_set["residual_mass_conservation_error"]
                r4204_conditional_slot_entropy = dense_set["conditional_slot_entropy"]
                r4204_conditional_max_slot_probability = dense_set["conditional_max_slot_probability"]
                r4204_residual_existence_mean = dense_set["residual_existence_mean"]
                r4204_centroid_separation = dense_set["centroid_separation"]
                r4204_spatial_variance_mean = dense_set["spatial_variance_mean"]
                r4204_spatial_identity_enabled = dense_set["spatial_identity_enabled"]
                r4204_location_as_occupancy_used = dense_set["location_as_occupancy_used"]
                r4205_enabled = dense_set["r4205_enabled"]
                r4205_overflow_probability = dense_set["overflow_probability"]
                r4205_overflow_identity_probability = dense_set["overflow_identity_probability"]
                r4205_conditional_identity_logits = dense_set["r4205_conditional_identity_logits"]
                r4205_editable_probability_sum = dense_set["editable_probability_sum"]
                r4205_overflow_probability_mean = dense_set["r4205_overflow_probability_mean"]
                r4205_overflow_conditional_mean = dense_set["r4205_overflow_conditional_mean"]
                r4205_editable_probability_mean = dense_set["r4205_editable_probability_mean"]
                r4205_final_logits_finite_fraction = dense_set["r4205_final_logits_finite_fraction"]
                r4207_enabled = dense_set["r4207_enabled"]
                r4207_seed_center_xy = dense_set["r4207_seed_center_xy"]
                r4207_seed_score = dense_set["r4207_seed_score"]
                r4207_seed_valid = dense_set["r4207_seed_valid"]
                r4207_seed_feature_finite_fraction = dense_set["r4207_seed_feature_finite_fraction"]
                r4207_seed_valid_fraction = dense_set["r4207_seed_valid_fraction"]
                r4207_seed_score_mean = dense_set["r4207_seed_score_mean"]
                r4207_seed_pairwise_distance_px = dense_set["r4207_seed_pairwise_distance_px"]
                r4207_q0_pairwise_cosine = dense_set["r4207_q0_pairwise_cosine"]
                r4207_q1_pairwise_cosine = dense_set["r4207_q1_pairwise_cosine"]
                r4207_seed_to_slot_centroid_drift_px = dense_set["r4207_seed_to_slot_centroid_drift_px"]
                r4207_full_image_assignment_enabled = dense_set["r4207_full_image_assignment_enabled"]
                r4207_hard_spatial_support_used = dense_set["r4207_hard_spatial_support_used"]
                r4208_enabled = dense_set["r4208_enabled"]
                r4208_normalized_fusion_enabled = dense_set["r4208_normalized_fusion_enabled"]
                r4208_persistent_identity_enabled = dense_set["r4208_persistent_identity_enabled"]
                r4208_learned_query_norm = dense_set["r4208_learned_query_norm"]
                r4208_seed_feature_norm = dense_set["r4208_seed_feature_norm"]
                r4208_seed_to_learned_norm_ratio = dense_set["r4208_seed_to_learned_norm_ratio"]
                r4208_seed_feature_pairwise_cosine = dense_set["r4208_seed_feature_pairwise_cosine"]
                r4208_q0_seed_identity_cosine = dense_set["r4208_q0_seed_identity_cosine"]
                r4208_q1_seed_identity_cosine = dense_set["r4208_q1_seed_identity_cosine"]
                r4208_identity_retention_delta = dense_set["r4208_identity_retention_delta"]
                r4210_enabled = dense_set["r4210_enabled"]
                r4210_variable_seed_enabled = dense_set["r4210_variable_seed_enabled"]
                r4210_independent_overflow_enabled = dense_set["r4210_independent_overflow_enabled"]
                r4210_valid_seed_count = dense_set["r4210_valid_seed_count"]
                r4210_seed_logit = dense_set["r4210_seed_logit"]
                r4210_seed_logit_mean = dense_set["r4210_seed_logit_mean"]
                r4210_overflow_gate_logits = dense_set["r4210_overflow_gate_logits"]
                r4210_overflow_conditional_probability = dense_set["r4210_overflow_conditional_probability"]
                r4210_overflow_logit_mean = dense_set["r4210_overflow_logit_mean"]
                r4210_overflow_conditional_mean = dense_set["r4210_overflow_conditional_mean"]
                r4211_enabled = dense_set["r4211_enabled"]
                r4211_proposal_existence_enabled = dense_set["r4211_proposal_existence_enabled"]
                r4211_geometry_overflow_enabled = dense_set["r4211_geometry_overflow_enabled"]
                r4211_proposal_seed_count = dense_set["r4211_proposal_seed_count"]
                r4211_proposal_confidence_mean = dense_set["r4211_proposal_confidence_mean"]
                r4211_geometry_effective_l1 = dense_set["r4211_geometry_effective_l1"]
                r4211_geometry_slot_probability = dense_set["r4211_geometry_slot_probability"]
                r4211_effective_slot_probability = dense_set["r4211_effective_slot_probability"]
        elif self.native_residual_set_r411_enabled and self.r411_typed_proposal_enabled:
            (
                query, anchor, proposal_score, proposal_valid, proposal_type,
                r411_center_logits, r411_size_map, r411_offset_map,
                r416_proposal_point, r416_edge_offsets, r416_legacy_unique_fraction,
                r417_location_logits, r417_location_offset_map,
            ) = self._r411_typed_proposal_initial_state(
                pixel_embedding=pixel_embedding,
                learned_query=query,
                learned_anchor=anchor,
            )
        elif self.evidence_proposal_r410_enabled and self.r410_use_evidence_proposals:
            query, anchor, proposal_score, proposal_valid = self._r410_proposal_initial_state(
                pixel_embedding=pixel_embedding,
                cause=cause,
                alpha=alpha,
                learned_query=query,
                learned_anchor=anchor,
            )
        initial_anchor = anchor
        if self.clean_core_v560_enabled:
            final_logits = v560_clean["final_logits"]
            final_anchor = v560_clean["soft_anchor"]
            stage_logits = final_logits[:, None]
            stage_anchors = final_anchor[:, None]
            attn_entropy = final_logits.new_zeros(())
            attn_max_weight = final_logits.new_zeros(())
        elif self.dense_competitive_residual_set_r4203_enabled:
            final_logits = dense_set["final_logits"]
            final_anchor = dense_set["soft_anchor"]
            stage_logits = final_logits[:, None]
            stage_anchors = final_anchor[:, None]
            attn_entropy = r4203_assignment_entropy
            attn_max_weight = r4203_max_slot_ownership
        elif self.dynamic_residual_mask_r420_enabled:
            final_logits = self._r420_dynamic_mask_logits(
                pixel_embedding=pixel_embedding, query=query, anchor=anchor
            )
            final_anchor = anchor
            stage_logits = final_logits[:, None]
            stage_anchors = final_anchor[:, None]
            attn_entropy = pixel_embedding.new_zeros(())
            attn_max_weight = pixel_embedding.new_zeros(())
            # Diagnostic only: average normalized distance from each seed over
            # the image.  It verifies finite location-relative coordinates.
            yy420 = torch.linspace(0.0, 1.0, h, device=anchor.device, dtype=anchor.dtype)
            xx420 = torch.linspace(0.0, 1.0, w, device=anchor.device, dtype=anchor.dtype)
            gy420, gx420 = torch.meshgrid(yy420, xx420, indexing="ij")
            dx420 = gx420[None, None] - anchor[:, :, 0, None, None]
            dy420 = gy420[None, None] - anchor[:, :, 1, None, None]
            r420_relative_coord_mean_abs = (0.5 * (dx420.abs() + dy420.abs())).mean().detach()
        else:
            (
                final_logits,
                final_anchor,
                stage_logits,
                stage_anchors,
                attn_entropy,
                attn_max_weight,
                r419_seed_support_fraction,
                r419_final_support_fraction,
                r419_outside_mask_probability,
            ) = self._r48_decode_queries(
                pixel_embedding=pixel_embedding,
                query=query,
                anchor=anchor,
                coarse_logits=coarse_logits,
                query_type=proposal_type,
                base_probability=base_probability,
                boundary=boundary,
                freeze_anchor=self.geometry_lock_r413_enabled,
            )
        raw_masks = torch.sigmoid(final_logits / self.mask_temperature)
        smooth_masks = F.avg_pool2d(
            raw_masks.reshape(b * self.num_slots, 1, h, w), 3, stride=1, padding=1
        ).reshape(b, self.num_slots, h, w)

        # Training-only DN-DETR-style component denoising.  Native Base is used
        # to construct exactly the same residual-component teacher as V538 loss.
        dn_stage_logits = final_logits.new_zeros((b, 1, 1, h, w))
        dn_stage_anchors = final_anchor.new_zeros((b, 1, 1, 4))
        dn_target_index = torch.zeros((b, 1), device=final_logits.device, dtype=torch.long)
        dn_valid = torch.zeros((b, 1), device=final_logits.device, dtype=torch.bool)
        teacher_masks = final_logits.new_zeros((b, self.num_slots, h, w))
        teacher_actions = torch.zeros((b, self.num_slots), device=final_logits.device, dtype=torch.long)
        teacher_valid = torch.zeros((b, self.num_slots), device=final_logits.device, dtype=torch.bool)
        teacher_area = final_logits.new_zeros((b, self.num_slots))
        teacher_geometry = final_anchor.new_zeros((b, self.num_slots, 4))
        teacher_effective = base_probability[:, :1].detach()
        teacher_error = final_logits.new_zeros((b, 1, h, w))
        dn_noise_scale = 0.0
        teacher_built = False
        build_forward_teacher = (
            self.training
            and isinstance(supervision_masks, torch.Tensor)
            and (self.r48_dn_enabled or self.r4201_clean_rootfix_enabled)
        )
        if build_forward_teacher:
            # Teacher extraction and DN auxiliary decoding have separate owners.
            # Disabling DN must never erase Native/Hungarian/location supervision.
            from utils.v538_loss import _build_teacher_components
            (
                teacher_masks,
                teacher_actions,
                teacher_valid,
                teacher_area,
                _replay_case,
                teacher_effective,
                teacher_error,
                _teacher_raw_count,
            ) = _build_teacher_components(
                teacher_probability=base_probability.detach(),
                gt=supervision_masks,
                num_slots=self.num_slots,
                min_pixels=self.r48_teacher_min_pixels,
                current_epoch=int(self.current_epoch),
                replay_enabled=False,
                replay_error_floor=0.0,
                replay_radius=1,
                utility_rank_enabled=self.native_residual_set_r411_enabled,
            )
            teacher_geometry = self._r48_geometry_from_masks(teacher_masks, teacher_valid)
            teacher_built = True

        # DN is only an optional auxiliary consumer of that teacher bank.
        if build_forward_teacher and self.r48_dn_enabled:
            g = self.r48_dn_groups
            n = self.num_slots * g
            target_index = torch.arange(self.num_slots, device=final_logits.device)[None, :, None].expand(b, -1, g).reshape(b, n)
            batch_index = torch.arange(b, device=final_logits.device)[:, None]
            target_geometry = teacher_geometry[batch_index, target_index]
            target_actions = teacher_actions[batch_index, target_index]
            dn_valid = teacher_valid[batch_index, target_index]
            noise = torch.rand_like(target_geometry) * 2.0 - 1.0
            if self.evidence_proposal_r410_enabled and self.r410_dn_clean_curriculum_enabled:
                # Clean reconstruction is learned first.  Difficulty increases
                # only after a fixed clean period, while every model component
                # still trains jointly from epoch 1.
                if int(self.current_epoch) < self.r410_dn_clean_epochs:
                    dn_noise_scale = 0.0
                else:
                    progress = min(
                        max(
                            float(int(self.current_epoch) - self.r410_dn_clean_epochs)
                            / float(self.r410_dn_noise_ramp_epochs),
                            0.0,
                        ),
                        1.0,
                    )
                    dn_noise_scale = self.r410_dn_noise_final * progress
            elif self.content_selective_r49_enabled:
                progress = min(
                    max(float(self.current_epoch) / float(self.r49_dn_noise_ramp_epochs), 0.0),
                    1.0,
                )
                dn_noise_scale = self.r49_dn_noise_start + (
                    self.r49_dn_noise_final - self.r49_dn_noise_start
                ) * progress
            else:
                dn_noise_scale = self.r48_dn_noise_scale
            # R4.12 uses the DN branch as an explicit oracle-geometry shape
            # teacher.  It must answer "can the shared renderer draw the right
            # component when geometry/type are correct?"; noisy-box robustness
            # is therefore not mixed into this causal capability probe.
            if self.canonical_shape_r412_enabled:
                dn_noise_scale = 0.0
            noisy_center = target_geometry[:, :, :2] + (
                noise[:, :, :2] * dn_noise_scale * target_geometry[:, :, 2:].clamp_min(0.03)
            )
            noisy_size = target_geometry[:, :, 2:] * (
                1.0 + dn_noise_scale * noise[:, :, 2:]
            )
            noisy_anchor = torch.cat(
                [
                    noisy_center.clamp(0.0, 1.0),
                    noisy_size.clamp(self.r47_anchor_min_size, self.r47_anchor_max_size),
                ],
                dim=2,
            )
            action_one_hot = F.one_hot(target_actions.clamp(0, 3), num_classes=4).to(final_logits.dtype)
            dn_query = self.r48_dn_query_encoder(torch.cat([noisy_anchor, action_one_hot], dim=2))
            if self.native_residual_set_r411_enabled:
                dn_query = dn_query + self.r411_type_embedding(target_actions.clamp(0, 3))
                dn_query = dn_query + self.r411_box_query_mlp(noisy_anchor)
            _, _, dn_stage_logits, dn_stage_anchors, _, _, _, _, _ = self._r48_decode_queries(
                pixel_embedding=pixel_embedding,
                query=dn_query,
                anchor=noisy_anchor,
                coarse_logits=None,
                query_type=target_actions,
                base_probability=base_probability,
                boundary=boundary,
                freeze_anchor=self.canonical_shape_r412_enabled,
            )
            dn_target_index = target_index

        # V552-R4.18 training-only paired stable mask-set branch.  It uses
        # the exact same image features and decoder parameters as Native, but
        # constructs a controlled coarse mask C' and supervises only the exact
        # residual C' xor GT.  No box/extent target is used.
        if (
            self.training
            and self.box_free_mask_set_r418_enabled
            and self.r418_paired_stable_teacher_enabled
            and (not self.dynamic_residual_mask_r420_enabled)
            and (not self.dense_competitive_residual_set_r4203_enabled)
            and isinstance(supervision_masks, torch.Tensor)
        ):
            from utils.v538_loss import _build_teacher_components
            r418_paired_coarse = self._r418_build_paired_coarse(supervision_masks).detach()
            p = r418_paired_coarse.clamp(EPS, 1.0 - EPS)
            paired_entropy = -(p * p.log() + (1.0 - p) * (1.0 - p).log())
            paired_dilate = F.max_pool2d(r418_paired_coarse, 3, stride=1, padding=1)
            paired_erode = 1.0 - F.max_pool2d(1.0 - r418_paired_coarse, 3, stride=1, padding=1)
            paired_boundary = (paired_dilate - paired_erode).clamp(0.0, 1.0)
            paired_zero4 = r418_paired_coarse.new_zeros((b, 4, h, w))
            paired_pixel_input = torch.cat(
                [
                    fused_feature,
                    r418_paired_coarse,
                    paired_entropy,
                    paired_boundary,
                    paired_zero4,
                    paired_zero4,
                    xy,
                ],
                dim=1,
            )
            paired_pixel_embedding = F.normalize(self.r47_pixel_encoder(paired_pixel_input), dim=1)
            (
                r418_paired_teacher_masks,
                r418_paired_teacher_actions,
                r418_paired_teacher_valid,
                _paired_area,
                _paired_replay,
                _paired_effective,
                _paired_error,
                _paired_raw_count,
            ) = _build_teacher_components(
                teacher_probability=r418_paired_coarse,
                gt=supervision_masks,
                num_slots=self.num_slots,
                min_pixels=self.r418_paired_min_pixels,
                current_epoch=int(self.current_epoch),
                replay_enabled=False,
                replay_error_floor=0.0,
                replay_radius=1,
                utility_rank_enabled=False,
            )
            paired_geometry = self._r48_geometry_from_masks(
                r418_paired_teacher_masks, r418_paired_teacher_valid
            )
            paired_center_grid = (2.0 * paired_geometry[:, :, :2] - 1.0).reshape(
                b, self.num_slots, 1, 2
            )
            paired_center_feature = F.grid_sample(
                paired_pixel_embedding,
                paired_center_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )[:, :, :, 0].transpose(1, 2)
            paired_learned = F.normalize(self.r47_slot_queries.weight, dim=1)[None].expand(b, -1, -1)
            paired_type_feature = self.r411_type_embedding(r418_paired_teacher_actions.clamp(0, 3))
            paired_query = F.normalize(
                paired_learned + paired_center_feature + paired_type_feature, dim=2
            )
            r418_paired_logits, _, _, _, _, _, _, _, _ = self._r48_decode_queries(
                pixel_embedding=paired_pixel_embedding,
                query=paired_query,
                anchor=paired_geometry,
                coarse_logits=None,
                query_type=r418_paired_teacher_actions,
                base_probability=r418_paired_coarse,
                boundary=paired_boundary,
                freeze_anchor=True,
            )

        return {
            "final_logits": final_logits,
            "raw_masks": raw_masks,
            "smooth_masks": smooth_masks,
            "proposal_type": proposal_type,
            "r411_center_logits": r411_center_logits,
            "r411_size_map": r411_size_map,
            "r411_offset_map": r411_offset_map,
            "r416_proposal_point": r416_proposal_point,
            "r416_edge_offsets": r416_edge_offsets,
            "r416_legacy_unique_fraction": r416_legacy_unique_fraction.detach(),
            "r417_location_logits": r417_location_logits,
            "r417_location_offset_map": r417_location_offset_map,
            "r418_paired_logits": r418_paired_logits,
            "r418_paired_teacher_masks": r418_paired_teacher_masks.detach(),
            "r418_paired_teacher_valid": r418_paired_teacher_valid.detach(),
            "r418_paired_coarse": r418_paired_coarse.detach(),
            "r419_seed_support_fraction": r419_seed_support_fraction.detach(),
            "r419_final_support_fraction": r419_final_support_fraction.detach(),
            "r419_outside_mask_probability": r419_outside_mask_probability.detach(),
            "r420_dynamic_mask_enabled": final_logits.new_tensor(1.0 if self.dynamic_residual_mask_r420_enabled else 0.0),
            "r420_type_decoupled_mask_enabled": final_logits.new_tensor(1.0 if self.r420_type_decoupled_mask_enabled else 0.0),
            "r420_dynamic_channels": final_logits.new_tensor(float(self.r420_dynamic_channels if self.dynamic_residual_mask_r420_enabled else 0)),
            "r420_relative_coord_mean_abs": r420_relative_coord_mean_abs.detach(),
            "r4201_clean_rootfix_enabled": final_logits.new_tensor(1.0 if self.r4201_clean_rootfix_enabled else 0.0),
            "r4203_dense_set_enabled": final_logits.new_tensor(1.0 if self.dense_competitive_residual_set_r4203_enabled else 0.0),
            "r4203_ownership_sum_error": r4203_ownership_sum_error.detach(),
            "r4203_assignment_entropy": r4203_assignment_entropy.detach(),
            "r4203_background_fraction": r4203_background_fraction.detach(),
            "r4203_max_slot_ownership": r4203_max_slot_ownership.detach(),
            "r4203_slot_mass_cv": r4203_slot_mass_cv.detach(),
            "r4203_point_bottleneck_used": final_logits.new_zeros(()),
            "r4204_enabled": final_logits.new_tensor(1.0 if self.factorized_residual_existence_identity_r4204_enabled else 0.0),
            "r4204_occupancy_logits": r4204_occupancy_logits,
            "r4204_residual_mass_conservation_error": r4204_residual_mass_conservation_error.detach(),
            "r4204_conditional_slot_entropy": r4204_conditional_slot_entropy.detach(),
            "r4204_conditional_max_slot_probability": r4204_conditional_max_slot_probability.detach(),
            "r4204_residual_existence_mean": r4204_residual_existence_mean.detach(),
            "r4204_centroid_separation": r4204_centroid_separation.detach(),
            "r4204_spatial_variance_mean": r4204_spatial_variance_mean.detach(),
            "r4204_spatial_identity_enabled": r4204_spatial_identity_enabled.detach(),
            "r4204_location_as_occupancy_used": r4204_location_as_occupancy_used.detach(),
            "r4205_enabled": r4205_enabled.detach(),
            "r4205_overflow_probability": r4205_overflow_probability,
            "r4205_overflow_identity_probability": r4205_overflow_identity_probability,
            "r4205_conditional_identity_logits": r4205_conditional_identity_logits,
            "r4205_editable_probability_sum": r4205_editable_probability_sum,
            "r4205_overflow_probability_mean": r4205_overflow_probability_mean.detach(),
            "r4205_overflow_conditional_mean": r4205_overflow_conditional_mean.detach(),
            "r4205_editable_probability_mean": r4205_editable_probability_mean.detach(),
            "r4205_final_logits_finite_fraction": r4205_final_logits_finite_fraction.detach(),
            "r4207_enabled": r4207_enabled.detach(),
            "r4207_seed_center_xy": r4207_seed_center_xy.detach(),
            "r4207_seed_score": r4207_seed_score.detach(),
            "r4207_seed_valid": r4207_seed_valid.detach(),
            "r4207_seed_feature_finite_fraction": r4207_seed_feature_finite_fraction.detach(),
            "r4207_seed_valid_fraction": r4207_seed_valid_fraction.detach(),
            "r4207_seed_score_mean": r4207_seed_score_mean.detach(),
            "r4207_seed_pairwise_distance_px": r4207_seed_pairwise_distance_px.detach(),
            "r4207_q0_pairwise_cosine": r4207_q0_pairwise_cosine.detach(),
            "r4207_q1_pairwise_cosine": r4207_q1_pairwise_cosine.detach(),
            "r4207_seed_to_slot_centroid_drift_px": r4207_seed_to_slot_centroid_drift_px.detach(),
            "r4207_full_image_assignment_enabled": r4207_full_image_assignment_enabled.detach(),
            "r4207_hard_spatial_support_used": r4207_hard_spatial_support_used.detach(),
            "r4208_enabled": r4208_enabled.detach(),
            "r4208_normalized_fusion_enabled": r4208_normalized_fusion_enabled.detach(),
            "r4208_persistent_identity_enabled": r4208_persistent_identity_enabled.detach(),
            "r4208_learned_query_norm": r4208_learned_query_norm.detach(),
            "r4208_seed_feature_norm": r4208_seed_feature_norm.detach(),
            "r4208_seed_to_learned_norm_ratio": r4208_seed_to_learned_norm_ratio.detach(),
            "r4208_seed_feature_pairwise_cosine": r4208_seed_feature_pairwise_cosine.detach(),
            "r4208_q0_seed_identity_cosine": r4208_q0_seed_identity_cosine.detach(),
            "r4208_q1_seed_identity_cosine": r4208_q1_seed_identity_cosine.detach(),
            "r4208_identity_retention_delta": r4208_identity_retention_delta.detach(),
            "r4210_enabled": r4210_enabled.detach(),
            "r4210_variable_seed_enabled": r4210_variable_seed_enabled.detach(),
            "r4210_independent_overflow_enabled": r4210_independent_overflow_enabled.detach(),
            "r4210_valid_seed_count": r4210_valid_seed_count.detach(),
            "r4210_seed_logit": r4210_seed_logit.detach(),
            "r4210_seed_logit_mean": r4210_seed_logit_mean.detach(),
            "r4210_overflow_gate_logits": r4210_overflow_gate_logits,
            "r4210_overflow_conditional_probability": r4210_overflow_conditional_probability,
            "r4210_overflow_logit_mean": r4210_overflow_logit_mean.detach(),
            "r4210_overflow_conditional_mean": r4210_overflow_conditional_mean.detach(),
            "r4211_enabled": r4211_enabled.detach(),
            "r4211_proposal_existence_enabled": r4211_proposal_existence_enabled.detach(),
            "r4211_geometry_overflow_enabled": r4211_geometry_overflow_enabled.detach(),
            "r4211_proposal_seed_count": r4211_proposal_seed_count.detach(),
            "r4211_proposal_confidence_mean": r4211_proposal_confidence_mean.detach(),
            "r4211_geometry_effective_l1": r4211_geometry_effective_l1.detach(),
            "r4211_geometry_slot_probability": r4211_geometry_slot_probability,
            "r4211_effective_slot_probability": r4211_effective_slot_probability,
            "initial_anchor": initial_anchor,
            "final_anchor": final_anchor,
            "v560_direct_mask_probability_mean": (
                v560_clean["mask_probability_mean"] if v560_clean is not None else final_logits.new_zeros(())
            ),
            "v560_independent_soft_overlap_mass": (
                v560_clean["soft_overlap_mass"] if v560_clean is not None else final_logits.new_zeros(())
            ),
            "v560_q0_pairwise_cosine": (
                v560_clean["q0_pairwise_cosine"] if v560_clean is not None else final_logits.new_zeros(())
            ),
            "v560_q1_pairwise_cosine": (
                v560_clean["q1_pairwise_cosine"] if v560_clean is not None else final_logits.new_zeros(())
            ),
            "v560_mask_bias_mean": (
                v560_clean["mask_bias_mean"] if v560_clean is not None else final_logits.new_zeros(())
            ),
            "stage_logits": stage_logits,
            "stage_anchors": stage_anchors,
            "attention_entropy": attn_entropy,
            "attention_max_weight": attn_max_weight,
            "proposal_score": proposal_score,
            "proposal_valid": proposal_valid,
            "proposal_score_mean": proposal_score.clamp_min(0.0).mean().detach(),
            "proposal_valid_fraction": proposal_valid.mean().detach(),
            "dn_noise_scale": final_logits.new_tensor(float(dn_noise_scale)),
            "dn_stage_logits": dn_stage_logits,
            "dn_stage_anchors": dn_stage_anchors,
            "dn_target_index": dn_target_index,
            "dn_valid": dn_valid,
            "teacher_masks": teacher_masks.detach(),
            "teacher_actions": teacher_actions.detach(),
            "teacher_valid": teacher_valid.detach(),
            "teacher_area": teacher_area.detach(),
            "teacher_geometry": teacher_geometry.detach(),
            "teacher_effective": teacher_effective.detach(),
            "teacher_error": teacher_error.detach(),
            "teacher_built": final_logits.new_tensor(1.0 if teacher_built else 0.0),
            "teacher_valid_count": teacher_valid.to(final_logits.dtype).sum().detach(),
        }

    @torch.no_grad()
    def _gpu_atomize(
        self,
        *,
        hard_masks: torch.Tensor,
        soft_masks: torch.Tensor,
        contrast_active: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized principal-axis atomization entirely on the input device.

        Each parent support is partitioned into disjoint longitudinal atoms
        along its principal spatial axis.  The procedure has no CPU transfer,
        no SciPy dependency, no per-component Python loop, and a fixed bounded
        output capacity selected by confidence.
        """
        b, k, h, w = hard_masks.shape
        m = self.max_atoms_per_slot
        dtype = soft_masks.dtype
        device = hard_masks.device
        hard = hard_masks.to(dtype)
        weight = (soft_masks.detach() * hard).clamp_min(0.0)
        denom = weight.flatten(2).sum(dim=2).clamp_min(1.0)

        y = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype).view(1, 1, h, 1)
        x = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype).view(1, 1, 1, w)
        cx = (weight * x).flatten(2).sum(dim=2) / denom
        cy = (weight * y).flatten(2).sum(dim=2) / denom
        dx = x - cx[:, :, None, None]
        dy = y - cy[:, :, None, None]
        cxx = (weight * dx.square()).flatten(2).sum(dim=2) / denom
        cyy = (weight * dy.square()).flatten(2).sum(dim=2) / denom
        cxy = (weight * dx * dy).flatten(2).sum(dim=2) / denom
        theta = 0.5 * torch.atan2(2.0 * cxy, cxx - cyy + EPS)
        axis_x = torch.cos(theta)[:, :, None, None]
        axis_y = torch.sin(theta)[:, :, None, None]
        projection = dx * axis_x + dy * axis_y
        spread = torch.sqrt(
            (weight * projection.square()).flatten(2).sum(dim=2) / denom + EPS
        )
        normalized = projection / (2.0 * spread[:, :, None, None] + EPS)

        if m == 1:
            assignment_soft = hard[:, :, None]
            assignment_hard = hard_masks[:, :, None]
        else:
            centers = torch.linspace(
                -self.atom_partition_extent,
                self.atom_partition_extent,
                m,
                device=device,
                dtype=dtype,
            ).view(1, 1, m, 1, 1)
            logits = -(normalized[:, :, None] - centers).square() / self.atom_partition_temperature
            logits = logits.masked_fill(~hard_masks[:, :, None], -1.0e4)
            assignment_soft = F.softmax(logits, dim=2) * hard[:, :, None]
            assignment_index = logits.argmax(dim=2)
            assignment_hard = (
                F.one_hot(assignment_index, num_classes=m)
                .permute(0, 1, 4, 2, 3)
                .to(torch.bool)
                & hard_masks[:, :, None]
            )

        atom_hard_all = assignment_hard.reshape(b, k * m, h, w)
        atom_soft_all = (soft_masks[:, :, None] * assignment_soft).reshape(b, k * m, h, w)
        parent_all = torch.arange(k, device=device, dtype=torch.long).view(1, k, 1).expand(b, k, m).reshape(b, k * m)
        area_pixels = atom_hard_all.flatten(2).sum(dim=2)
        parent_contrast = contrast_active[:, :, None].expand(b, k, m).reshape(b, k * m)
        valid_all = parent_contrast & (area_pixels >= self.atom_min_pixels)
        confidence_all = (atom_soft_all * atom_hard_all.to(dtype)).flatten(2).sum(dim=2) / area_pixels.clamp_min(1).to(dtype)
        # Prefer high-confidence, non-trivial atoms while keeping the capacity
        # independent of noisy connected-component counts.
        area_fraction = area_pixels.to(dtype) / float(max(h * w, 1))
        selection_score = confidence_all + 0.05 * torch.log1p(area_fraction * float(h * w))
        selection_score = selection_score.masked_fill(~valid_all, -1.0e4)
        capacity = min(self.max_active_atoms, k * m)
        top_score, top_index = torch.topk(selection_score, k=capacity, dim=1)
        gather_spatial = top_index[:, :, None, None].expand(-1, -1, h, w)
        atom_hard = atom_hard_all.gather(1, gather_spatial)
        atom_soft = atom_soft_all.gather(1, gather_spatial)
        parent_index = parent_all.gather(1, top_index)
        atom_valid = valid_all.gather(1, top_index) & (top_score > -1.0e3)
        atom_confidence = confidence_all.gather(1, top_index)
        return atom_hard, atom_soft, parent_index, atom_valid, atom_confidence

    def _scale_aware_region(
        self,
        mask: torch.Tensor,
        scale_st: torch.Tensor,
        radii: Sequence[int],
    ) -> torch.Tensor:
        choices = []
        for radius in radii:
            choices.append(_dilate(mask, int(radius)))
        stacked = torch.stack(choices, dim=2)
        return (stacked * scale_st[..., None, None]).sum(dim=2).clamp(0.0, 1.0)

    def _selector_from_feature(self, selector_feature: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply the calibrated outcome critic without a second spatial encoder.

        V552-R2 uses exactly one Neutral/Benefit/Harm head.  Its signed utility is
        formed from two non-negative magnitude heads, which removes the old
        Benefit/Harm/Direction/Gain sign contradiction while preserving all
        compatibility tensors expected by V538--V550 diagnostics.
        """
        rank_scores = self.rank_head(selector_feature).squeeze(-1)
        if self.teacher_decoupled_r2_enabled:
            outcome_logits = self.r2_outcome_head(selector_feature)
            outcome_probs = F.softmax(outcome_logits, dim=-1)
            neutral_probs, benefit_probs, harm_probs = outcome_probs.unbind(dim=-1)
            editability_probs = (benefit_probs + harm_probs).clamp(EPS, 1.0 - EPS)
            direction_probs = benefit_probs / editability_probs.clamp_min(EPS)
            editability_logits = torch.logit(editability_probs)
            direction_logits = torch.logit(direction_probs.clamp(EPS, 1.0 - EPS))
            benefit_logits = torch.logit(benefit_probs.clamp(EPS, 1.0 - EPS))
            harm_logits = torch.logit(harm_probs.clamp(EPS, 1.0 - EPS))

            benefit_magnitude_normalized = F.softplus(
                self.r2_benefit_magnitude_head(selector_feature).squeeze(-1)
            )
            harm_magnitude_normalized = F.softplus(
                self.r2_harm_magnitude_head(selector_feature).squeeze(-1)
            )
            benefit_contribution = benefit_probs * benefit_magnitude_normalized
            harm_contribution = -harm_probs * harm_magnitude_normalized
            normalized_gain = benefit_contribution + harm_contribution
            gain_magnitude_normalized = (
                benefit_contribution.abs() + harm_contribution.abs()
            )
            signed_outcome = benefit_probs - harm_probs
            factorized_flag = True
        elif self.factorized_outcome_enabled:
            editability_logits = self.editability_head(selector_feature).squeeze(-1)
            direction_logits = self.direction_head(selector_feature).squeeze(-1)
            editability_probs = torch.sigmoid(editability_logits)
            direction_probs = torch.sigmoid(direction_logits)
            neutral_probs = 1.0 - editability_probs
            benefit_probs = editability_probs * direction_probs
            harm_probs = editability_probs * (1.0 - direction_probs)
            outcome_probs = torch.stack(
                [neutral_probs, benefit_probs, harm_probs], dim=-1
            ).clamp(EPS, 1.0)
            outcome_probs = outcome_probs / outcome_probs.sum(
                dim=-1, keepdim=True
            ).clamp_min(EPS)
            outcome_logits = outcome_probs.clamp_min(EPS).log()
            benefit_logits = torch.logit(benefit_probs.clamp(EPS, 1.0 - EPS))
            harm_logits = torch.logit(harm_probs.clamp(EPS, 1.0 - EPS))
            signed_outcome = editability_probs * (2.0 * direction_probs - 1.0)
            gain_magnitude_normalized = F.softplus(
                self.gain_head(selector_feature).squeeze(-1)
            )
            normalized_gain = signed_outcome * gain_magnitude_normalized
            benefit_magnitude_normalized = gain_magnitude_normalized
            harm_magnitude_normalized = gain_magnitude_normalized
            benefit_contribution = benefit_probs * gain_magnitude_normalized
            harm_contribution = -harm_probs * gain_magnitude_normalized
            factorized_flag = True
        else:
            benefit_logits = self.benefit_head(selector_feature).squeeze(-1)
            harm_logits = self.harm_head(selector_feature).squeeze(-1)
            editability_logits = torch.logsumexp(
                torch.stack([benefit_logits, harm_logits], dim=-1), dim=-1
            )
            direction_logits = benefit_logits - harm_logits
            outcome_logits = torch.stack(
                [torch.zeros_like(benefit_logits), benefit_logits, harm_logits], dim=-1
            )
            outcome_probs = torch.softmax(outcome_logits, dim=-1)
            neutral_probs, benefit_probs, harm_probs = outcome_probs.unbind(dim=-1)
            editability_probs = benefit_probs + harm_probs
            direction_probs = benefit_probs / editability_probs.clamp_min(EPS)
            signed_outcome = benefit_probs - harm_probs
            gain_magnitude_normalized = F.softplus(
                self.gain_head(selector_feature).squeeze(-1)
            )
            normalized_gain = signed_outcome * gain_magnitude_normalized
            benefit_magnitude_normalized = gain_magnitude_normalized
            harm_magnitude_normalized = gain_magnitude_normalized
            benefit_contribution = benefit_probs * gain_magnitude_normalized
            harm_contribution = -harm_probs * gain_magnitude_normalized
            factorized_flag = False

        gain_scores = normalized_gain / self.selector_gain_scale
        if self.selector_lcb_beta > 0.0:
            gain_logvar = self.gain_logvar_head(selector_feature).squeeze(-1).clamp(
                self.selector_logvar_min, self.selector_logvar_max
            )
            gain_std = torch.exp(0.5 * gain_logvar) / self.selector_gain_scale
            gain_lcb = gain_scores - self.selector_lcb_beta * gain_std
        else:
            gain_logvar = torch.zeros_like(gain_scores)
            gain_std = torch.zeros_like(gain_scores)
            gain_lcb = gain_scores
        decision_scores = (
            gain_lcb
            if self.use_gain_as_decision_score
            else rank_scores + benefit_logits - harm_logits
        )
        return {
            "benefit_logits": benefit_logits,
            "benefit_probs": benefit_probs,
            "harm_logits": harm_logits,
            "harm_probs": harm_probs,
            "editability_logits": editability_logits,
            "editability_probs": editability_probs,
            "direction_logits": direction_logits,
            "direction_probs": direction_probs,
            "factorized_outcome_enabled": gain_scores.new_tensor(
                1.0 if factorized_flag else 0.0
            ),
            "factorized_direction_zero_init_enabled": gain_scores.new_tensor(
                1.0 if self.factorized_direction_zero_init else 0.0
            ),
            "neutral_probs": neutral_probs,
            "outcome_logits": outcome_logits,
            "outcome_probs": outcome_probs,
            "signed_outcome": signed_outcome,
            "gain_magnitude_normalized": gain_magnitude_normalized,
            "benefit_magnitude_normalized": benefit_magnitude_normalized,
            "harm_magnitude_normalized": harm_magnitude_normalized,
            "benefit_contribution": benefit_contribution,
            "harm_contribution": harm_contribution,
            "use_gain_as_decision_score": gain_scores.new_tensor(
                1.0 if self.use_gain_as_decision_score else 0.0
            ),
            "rank_scores": rank_scores,
            "normalized_gain": normalized_gain,
            "gain_scores": gain_scores,
            "gain_logvar": gain_logvar,
            "gain_std": gain_std,
            "gain_lcb": gain_lcb,
            "decision_scores": decision_scores,
            "selector_features": selector_feature,
            "direction_head_weight": self.direction_head.weight,
            "direction_head_bias": self.direction_head.bias,
            "editability_head_weight": self.editability_head.weight,
            "editability_head_bias": self.editability_head.bias,
            "gain_head_weight": self.gain_head.weight,
            "gain_head_bias": self.gain_head.bias,
            "factorized_deployment_enabled": gain_scores.new_tensor(
                1.0 if self.factorized_deployment_enabled else 0.0
            ),
            "v552r2_unified_outcome_enabled": gain_scores.new_tensor(
                1.0 if self.teacher_decoupled_r2_enabled else 0.0
            ),
        }

    def _r4_coupled_signed_gain(
        self,
        *,
        outcome_logits: torch.Tensor,
        raw_magnitude: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a bounded signed gain whose sign is tied to Outcome semantics.

        Benefit probability contributes positively, Harm probability contributes
        negatively, and Neutral contributes zero.  The scalar head predicts only
        a non-negative magnitude, so Outcome and Gain cannot contradict each
        other by construction.
        """
        outcome_probs = F.softmax(outcome_logits, dim=-1)
        signed_direction = outcome_probs[..., 1] - outcome_probs[..., 2]
        magnitude = torch.sigmoid(raw_magnitude) * self.critic_gain_cap
        return signed_direction * magnitude, magnitude

    def _r42_decoupled_expected_gain(
        self,
        *,
        outcome_logits: torch.Tensor,
        raw_benefit_magnitude: torch.Tensor,
        raw_harm_magnitude: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Class-conditional expected gain used by the R4.2 critics.

        Outcome probabilities express *which state* is likely, while the two
        scalar heads express the conditional magnitude of a beneficial or
        harmful outcome.  The heads are non-negative by construction.  Losses
        supervise Outcome and magnitudes through separate paths; this method is
        used to form the inference-time expected utility only.
        """
        outcome_probs = F.softmax(outcome_logits, dim=-1)
        benefit_magnitude = (
            torch.sigmoid(raw_benefit_magnitude) * self.critic_gain_cap
        )
        harm_magnitude = torch.sigmoid(raw_harm_magnitude) * self.critic_gain_cap
        expected_gain = (
            outcome_probs[..., 1] * benefit_magnitude
            - outcome_probs[..., 2] * harm_magnitude
        )
        return expected_gain, benefit_magnitude, harm_magnitude

    def _r4_selector_from_feature(
        self,
        selector_feature: torch.Tensor,
        *,
        outcome_logits: Optional[torch.Tensor] = None,
        gain_scores: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build one absolute-to-Base utility selector from R4 route features."""
        if outcome_logits is None:
            outcome_logits = self.candidate_utility_outcome_head(selector_feature)
        outcome_probs = F.softmax(outcome_logits, dim=-1)
        neutral_probs, benefit_probs, harm_probs = outcome_probs.unbind(dim=-1)
        if gain_scores is None:
            gain_scores, magnitude = self._r4_coupled_signed_gain(
                outcome_logits=outcome_logits,
                raw_magnitude=self.candidate_absolute_gain_head(
                    selector_feature
                ).squeeze(-1),
            )
        else:
            magnitude = gain_scores.abs()
        editability_probs = (benefit_probs + harm_probs).clamp(EPS, 1.0 - EPS)
        direction_probs = (
            benefit_probs / editability_probs.clamp_min(EPS)
        ).clamp(EPS, 1.0 - EPS)
        benefit_logits = torch.logit(benefit_probs.clamp(EPS, 1.0 - EPS))
        harm_logits = torch.logit(harm_probs.clamp(EPS, 1.0 - EPS))
        editability_logits = torch.logit(editability_probs)
        direction_logits = torch.logit(direction_probs)
        signed_outcome = benefit_probs - harm_probs
        if self.clean_dynamic_component_set_enabled:
            # CLEAN does not learn an LCB, rank score, or probability calibration
            # on top of DeltaDice. Zero is the physical Preserve boundary.
            rank_scores = gain_scores
            gain_logvar = torch.zeros_like(gain_scores)
            gain_std = torch.zeros_like(gain_scores)
            gain_lcb = gain_scores
        else:
            rank_scores = self.rank_head(selector_feature).squeeze(-1)
            if self.selector_lcb_beta > 0.0:
                gain_logvar = self.gain_logvar_head(selector_feature).squeeze(-1).clamp(
                    self.selector_logvar_min, self.selector_logvar_max
                )
                gain_std = torch.exp(0.5 * gain_logvar) / self.selector_gain_scale
                gain_lcb = gain_scores - self.selector_lcb_beta * gain_std
            else:
                gain_logvar = torch.zeros_like(gain_scores)
                gain_std = torch.zeros_like(gain_scores)
                gain_lcb = gain_scores

        # R4.1 root fix: one physical unit only.  Outcome probabilities supervise
        # the sign through the coupled gain above; they are not added again as a
        # dimensionless probability term to a Dice-gain score.
        decision_scores = gain_lcb
        normalized_magnitude = magnitude * self.selector_gain_scale
        return {
            "benefit_logits": benefit_logits,
            "benefit_probs": benefit_probs,
            "harm_logits": harm_logits,
            "harm_probs": harm_probs,
            "editability_logits": editability_logits,
            "editability_probs": editability_probs,
            "direction_logits": direction_logits,
            "direction_probs": direction_probs,
            "factorized_outcome_enabled": gain_scores.new_zeros(()),
            "factorized_direction_zero_init_enabled": gain_scores.new_zeros(()),
            "neutral_probs": neutral_probs,
            "outcome_logits": outcome_logits,
            "outcome_probs": outcome_probs,
            "signed_outcome": signed_outcome,
            "gain_magnitude_normalized": normalized_magnitude,
            "benefit_magnitude_normalized": normalized_magnitude,
            "harm_magnitude_normalized": normalized_magnitude,
            "benefit_contribution": benefit_probs * normalized_magnitude,
            "harm_contribution": -harm_probs * normalized_magnitude,
            "use_gain_as_decision_score": gain_scores.new_ones(()),
            "rank_scores": rank_scores,
            "normalized_gain": gain_scores * self.selector_gain_scale,
            "gain_scores": gain_scores,
            "gain_logvar": gain_logvar,
            "gain_std": gain_std,
            "gain_lcb": gain_lcb,
            "decision_scores": decision_scores,
            "selector_features": selector_feature,
            "direction_head_weight": self.direction_head.weight,
            "direction_head_bias": self.direction_head.bias,
            "editability_head_weight": self.editability_head.weight,
            "editability_head_bias": self.editability_head.bias,
            "gain_head_weight": self.gain_head.weight,
            "gain_head_bias": self.gain_head.bias,
            "factorized_deployment_enabled": gain_scores.new_zeros(()),
            "v552r2_unified_outcome_enabled": gain_scores.new_ones(()),
            "v552r41_gain_outcome_coupled": gain_scores.new_ones(()),
            "v552r41_decision_score_gain_only": gain_scores.new_ones(()),
        }

    @torch.no_grad()
    def _enqueue_r4_critic(
        self,
        *,
        task: str,
        features: torch.Tensor,
        gains: torch.Tensor,
        labels: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        """Store detached train-only route features in class-complete queues."""
        if self.critic_queue_capacity <= 0:
            return
        if task not in {"safety", "utility"}:
            raise ValueError(f"Unknown R4 queue task: {task}")
        feature_queue = getattr(self, f"_v552r4_{task}_queue_features")
        gain_queue = getattr(self, f"_v552r4_{task}_queue_gains")
        count = getattr(self, f"_v552r4_{task}_queue_count")
        pointer = getattr(self, f"_v552r4_{task}_queue_ptr")
        flat_feature = features.detach().reshape(-1, features.shape[-1])
        flat_gain = gains.detach().reshape(-1)
        flat_label = labels.detach().reshape(-1)
        flat_valid = valid.detach().reshape(-1)
        for cls in range(3):
            selected = flat_valid & (flat_label == cls)
            values = flat_feature[selected]
            gain_values = flat_gain[selected]
            if values.numel() == 0:
                continue
            if values.shape[0] > self.critic_queue_capacity:
                values = values[-self.critic_queue_capacity :]
                gain_values = gain_values[-self.critic_queue_capacity :]
            start = int(pointer[cls].item())
            length = int(values.shape[0])
            first = min(length, self.critic_queue_capacity - start)
            feature_queue[cls, start : start + first].copy_(values[:first])
            gain_queue[cls, start : start + first].copy_(gain_values[:first])
            remaining = length - first
            if remaining > 0:
                feature_queue[cls, :remaining].copy_(values[first:])
                gain_queue[cls, :remaining].copy_(gain_values[first:])
            pointer[cls] = (start + length) % self.critic_queue_capacity
            count[cls] = min(
                self.critic_queue_capacity,
                int(count[cls].item()) + length,
            )

    def _r4_queue_predictions(
        self,
        task: str,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Replay detached features through current critic heads."""
        if task not in {"safety", "utility"}:
            raise ValueError(f"Unknown R4 queue task: {task}")
        feature_queue = getattr(self, f"_v552r4_{task}_queue_features")
        gain_queue = getattr(self, f"_v552r4_{task}_queue_gains")
        count = getattr(self, f"_v552r4_{task}_queue_count")
        pointer = getattr(self, f"_v552r4_{task}_queue_ptr")
        features = []
        targets = []
        gains = []
        for cls in range(3):
            cls_count = int(count[cls].item())
            if cls_count <= 0:
                continue
            if self.spatial_evidence_r43_enabled and cls_count >= self.critic_queue_capacity:
                # Ring-buffer pointer marks the next write position, hence also
                # the oldest sample. Reconstruct oldest->newest order so quota
                # replay's trailing slice really selects recent evidence.
                start = int(pointer[cls].item())
                cls_features = torch.cat(
                    [feature_queue[cls, start:], feature_queue[cls, :start]], dim=0
                )
                cls_gains = torch.cat(
                    [gain_queue[cls, start:], gain_queue[cls, :start]], dim=0
                )
            else:
                cls_features = feature_queue[cls, :cls_count]
                cls_gains = gain_queue[cls, :cls_count]
            features.append(cls_features.detach())
            targets.append(
                torch.full(
                    (cls_count,),
                    cls,
                    dtype=torch.long,
                    device=feature_queue.device,
                )
            )
            gains.append(cls_gains.detach())
        if not features:
            empty_feature = feature_queue.new_zeros((0, feature_queue.shape[-1]))
            return (
                empty_feature.new_zeros((0, 3)),
                empty_feature.new_zeros((0,)),
                torch.zeros((0,), dtype=torch.long, device=feature_queue.device),
                empty_feature.new_zeros((0,)),
                empty_feature.new_zeros((0,)),
                empty_feature.new_zeros((0,)),
            )
        feature = torch.cat(features, dim=0)
        target = torch.cat(targets, dim=0)
        replay_gain_value = torch.cat(gains, dim=0)
        if task == "safety":
            logits = self.editor_safety_outcome_head(feature)
            if self.decoupled_critic_r42_enabled:
                gain_pred, benefit_magnitude, harm_magnitude = (
                    self._r42_decoupled_expected_gain(
                        outcome_logits=logits,
                        raw_benefit_magnitude=self.editor_benefit_magnitude_head(
                            feature
                        ).squeeze(-1),
                        raw_harm_magnitude=self.editor_harm_magnitude_head(
                            feature
                        ).squeeze(-1),
                    )
                )
            else:
                gain_pred, shared_magnitude = self._r4_coupled_signed_gain(
                    outcome_logits=logits,
                    raw_magnitude=self.editor_relative_gain_head(feature).squeeze(-1),
                )
                benefit_magnitude = shared_magnitude
                harm_magnitude = shared_magnitude
        else:
            logits = self.candidate_utility_outcome_head(feature)
            if self.decoupled_critic_r42_enabled:
                gain_pred, benefit_magnitude, harm_magnitude = (
                    self._r42_decoupled_expected_gain(
                        outcome_logits=logits,
                        raw_benefit_magnitude=self.candidate_benefit_magnitude_head(
                            feature
                        ).squeeze(-1),
                        raw_harm_magnitude=self.candidate_harm_magnitude_head(
                            feature
                        ).squeeze(-1),
                    )
                )
            else:
                gain_pred, shared_magnitude = self._r4_coupled_signed_gain(
                    outcome_logits=logits,
                    raw_magnitude=self.candidate_absolute_gain_head(feature).squeeze(-1),
                )
                benefit_magnitude = shared_magnitude
                harm_magnitude = shared_magnitude
        return (
            logits,
            gain_pred,
            target,
            replay_gain_value,
            benefit_magnitude,
            harm_magnitude,
        )

    @staticmethod
    def _route_spatial_evidence(
        *,
        fused_feature: torch.Tensor,
        route_delta: torch.Tensor,
        base_probability: torch.Tensor,
        entropy: torch.Tensor,
        boundary: torch.Tensor,
        cause_probability: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized GT-free evidence for every exact route.

        Positive and negative logit changes are pooled separately because Fill
        and Delete actions with identical absolute area have opposite semantics.
        Four scalar maps summarize confidence/error context over the changed
        support. Shapes: feature [B,C,H,W], delta [B,N,R,H,W], output
        [B,N,R,2C+4].
        """
        feature = fused_feature.detach()
        delta = route_delta.detach()
        positive = F.relu(delta)
        negative = F.relu(-delta)

        def pool(weights: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
            denominator = weights.sum(dim=(-2, -1), keepdim=True).clamp_min(EPS)
            normalized = weights / denominator
            return torch.einsum("bnrhw,bchw->bnrc", normalized, values)

        positive_feature = pool(positive, feature)
        negative_feature = pool(negative, feature)
        support = delta.abs()
        support_mass = support.mean(dim=(-2, -1), keepdim=False)[..., None]
        base_context = pool(support, base_probability.detach())
        entropy_context = pool(support, entropy.detach())
        boundary_context = pool(support, boundary.detach())
        cause_scalar = cause_probability.detach().mean(dim=1, keepdim=True)
        cause_context = pool(support, cause_scalar)
        # Empty/Preserve relative routes naturally return zero pooled evidence.
        return torch.cat(
            [
                positive_feature,
                negative_feature,
                base_context,
                entropy_context,
                boundary_context,
                cause_context,
            ],
            dim=-1,
        )

    def _compose_exact_deltas(
        self,
        *,
        base_probability: torch.Tensor,
        atom_masks: torch.Tensor,
        exact_logit_delta: torch.Tensor,
        action_index: torch.Tensor,
        scores: torch.Tensor,
        presence: torch.Tensor,
        valid: torch.Tensor,
        benefit: torch.Tensor,
        harm: torch.Tensor,
        gain_lcb: torch.Tensor,
        editability: torch.Tensor,
        direction: torch.Tensor,
        deploy_enabled: bool,
        gain_sign_only: bool = False,
        selector_features: Optional[torch.Tensor] = None,
        apply_safety_gates: bool = True,
        force_all_steps_active: bool = False,
        forced_choices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Sequentially edit and compose a compatible candidate subset.

        Unlike the V552 single-choice safety fallback, this routine treats M2
        as a true set composer.  At every step it conditions each remaining
        edited atom on the current composed probability, estimates the marginal
        utility of adding that atom, and competes all candidates against an
        explicit Stop token.  The loop is bounded by ``max_steps`` (normally
        two or three) and all candidates/cases remain vectorized.
        """
        b, n, h, w = atom_masks.shape
        dtype, device = exact_logit_delta.dtype, exact_logit_delta.device
        if selector_features is None:
            selector_dim = int(self.m2_selector_trunk[-2].out_features)
            selector_features = exact_logit_delta.new_zeros((b, n, selector_dim))
        base = base_probability[:, :1].detach().clamp(EPS, 1.0 - EPS)
        base_logit = _safe_logit(base)
        max_steps = min(max(int(self.max_steps), 1), n)
        if forced_choices is not None:
            if forced_choices.ndim != 2 or forced_choices.shape[0] != b:
                raise ValueError("forced_choices must have shape [B,steps]")
            if forced_choices.shape[1] < max_steps:
                raise ValueError("forced_choices has fewer steps than composer")
            forced_choices = forced_choices[:, :max_steps].detach().long()

        accepted = torch.zeros((b, n), dtype=torch.bool, device=device)
        action_weight = torch.zeros((b, 4, h, w), dtype=dtype, device=device)
        polarity_weight = torch.zeros((b, 2, h, w), dtype=dtype, device=device)
        signed_delta = torch.zeros((b, 1, h, w), dtype=dtype, device=device)
        occupied = torch.zeros((b, h, w), dtype=torch.bool, device=device)
        occupied_sign = torch.zeros((b, h, w), dtype=dtype, device=device)
        selected_index = torch.zeros((b,), dtype=torch.long, device=device)
        selected_score = scores.new_zeros((b,))
        composition_score = scores.new_zeros((b,))
        accepted_count = scores.new_zeros((b,))
        first_set = torch.zeros((b,), dtype=torch.bool, device=device)
        active = torch.full(
            (b,), bool(deploy_enabled or force_all_steps_active),
            dtype=torch.bool,
            device=device,
        )
        batch = torch.arange(b, device=device)
        max_pixels = max(int(round(self.max_total_edit_fraction * h * w)), 1)
        mask_bool_all = atom_masks.bool()
        mask_float_all = mask_bool_all.to(dtype)
        area_all = mask_bool_all.flatten(2).sum(dim=2)
        area_fraction_all = area_all.to(dtype) / float(max(h * w, 1))
        action_oh_all = F.one_hot(action_index.clamp(0, 3), num_classes=4).to(dtype)
        action_sign_all = (
            action_oh_all[:, :, 1] + action_oh_all[:, :, 3]
            - action_oh_all[:, :, 0] - action_oh_all[:, :, 2]
        )

        # The training trace is intentionally allowed to see every physically
        # valid atom.  Only Shadow/real deployment apply calibration gates.
        safe = valid.clone()
        if self.clean_core_v560_enabled:
            apply_safety_gates = False
        if apply_safety_gates and (not self.native_contract_r46_enabled):
            safe = safe & (presence >= self.deployment_presence_threshold)
            if (self.factorized_outcome_enabled or self.teacher_decoupled_r2_enabled) and self.factorized_deployment_enabled:
                safe = safe & (
                    editability >= self.deployment_editability_threshold
                ) & (direction >= self.deployment_direction_threshold)
            if self.semantic_deployment_r44_enabled:
                neutral = (1.0 - benefit - harm).clamp(0.0, 1.0)
                semantic_benefit = (benefit >= neutral) & (benefit >= harm)
                safe = safe & semantic_benefit
                safe = safe & (
                    benefit >= harm + self.deployment_benefit_harm_margin
                )
            elif self.unified_deployment_gate:
                safe = safe & (
                    benefit >= self.selector_benefit_threshold
                ) & (harm <= self.selector_harm_threshold)
                safe = safe & (
                    benefit >= harm + self.deployment_benefit_harm_margin
                )
            elif not ((self.factorized_outcome_enabled or self.teacher_decoupled_r2_enabled) and self.factorized_deployment_enabled):
                safe = safe & (
                    benefit >= self.selector_benefit_threshold
                ) & (harm <= self.selector_harm_threshold)
            minimum_gain = 0.0 if gain_sign_only else self.deployment_min_gain
            safe = safe & (gain_lcb > minimum_gain)

        step_logits = []
        step_candidate_scores = []
        step_states = []
        step_eligible = []
        step_active = []
        step_selected = []

        for step in range(max_steps):
            current_probability = torch.sigmoid(base_logit + signed_delta).clamp(
                EPS, 1.0 - EPS
            )
            step_states.append(current_probability)
            step_active.append(active)

            overlap_pixels = (
                mask_bool_all & occupied[:, None]
            ).flatten(2).sum(dim=2)
            overlap = overlap_pixels.to(dtype) / area_all.clamp_min(1).to(dtype)
            sign_product = (
                occupied_sign[:, None]
                * action_sign_all[:, :, None, None]
            )
            conflict_pixels = (
                mask_bool_all & occupied[:, None] & (sign_product < 0.0)
            ).flatten(2).sum(dim=2)
            conflict = conflict_pixels.to(dtype) / area_all.clamp_min(1).to(dtype)
            new_mask = mask_bool_all & (~occupied[:, None])
            new_area = new_mask.flatten(2).sum(dim=2)
            current_pixels = occupied.flatten(1).sum(dim=1)
            budget_excess = (
                current_pixels[:, None] + new_area - max_pixels
            ).clamp_min(0).to(dtype) / float(max_pixels)

            if self.clean_core_v560_enabled:
                # In a one-step alternative-candidate selector, candidates do
                # not compose with each other.  Overlap/budget heuristics are
                # therefore not valid eligibility criteria; existence and
                # physical non-empty support are sufficient.
                dynamic_eligible = (
                    safe & (~accepted) & active[:, None] & (area_all > 0)
                )
            else:
                dynamic_eligible = (
                    safe
                    & (~accepted)
                    & active[:, None]
                    & (area_all > 0)
                    & (new_area > 0)
                    & (overlap <= self.max_overlap)
                    & ((current_pixels[:, None] + new_area) <= max_pixels)
                )
            step_eligible.append(dynamic_eligible)

            area_denom = area_all.clamp_min(1).to(dtype)
            current_region_mean = (
                current_probability[:, 0][:, None] * mask_float_all
            ).flatten(2).sum(dim=2) / area_denom
            delta_abs_mean = (
                exact_logit_delta.abs() * mask_float_all
            ).flatten(2).sum(dim=2) / area_denom
            delta_signed_mean = (
                exact_logit_delta * mask_float_all
            ).flatten(2).sum(dim=2) / area_denom
            step_fraction = scores.new_full(
                (b, n), float(step) / float(max(max_steps - 1, 1))
            )
            interaction_stats = torch.stack(
                [
                    overlap,
                    conflict,
                    area_fraction_all,
                    current_region_mean,
                    delta_abs_mean,
                    delta_signed_mean,
                    gain_lcb,
                    (
                        torch.zeros_like(benefit)
                        if self.native_contract_r46_enabled else benefit - harm
                    ),
                    presence,
                    step_fraction,
                ],
                dim=2,
            )
            state_inputs = torch.cat(
                [
                    base,
                    current_probability,
                    current_probability - base,
                    occupied[:, None].to(dtype),
                ],
                dim=1,
            )
            state_feature = self.composer_state_encoder(state_inputs).flatten(1)
            state_many = state_feature[:, None].expand(-1, n, -1)
            conditioned_feature = selector_features + self.composer_candidate_adapter(
                torch.cat(
                    [selector_features, state_many, interaction_stats], dim=2
                )
            )
            correction = torch.tanh(
                self.composer_marginal_head(conditioned_feature).squeeze(-1)
            ) * self.composer_marginal_correction_cap
            candidate_score = (
                scores
                + correction
                - self.composer_overlap_penalty * overlap
                - self.composer_conflict_penalty * conflict
                - self.composer_budget_penalty * budget_excess
            )
            if self.zero_stop_one_step_r4212_enabled or self.clean_core_v560_enabled:
                # One-step M2 has only one meaningful decision: preserve Base
                # (physical utility 0) or execute the candidate with maximum
                # predicted signed DeltaDice.  Learned marginal corrections and
                # arbitrary Stop logits would mix incompatible units.
                candidate_score = scores
            elif not self.multicandidate_composer_enabled:
                candidate_score = scores
            step_candidate_scores.append(candidate_score)

            candidate_logits = candidate_score.masked_fill(
                ~dynamic_eligible, -1.0e4
            )
            eligible_weight = dynamic_eligible.to(dtype)
            global_feature = (
                conditioned_feature * eligible_weight[:, :, None]
            ).sum(dim=1) / eligible_weight.sum(dim=1, keepdim=True).clamp_min(1.0)
            finite_max = candidate_logits.max(dim=1).values
            finite_max = torch.where(
                dynamic_eligible.any(dim=1), finite_max, torch.zeros_like(finite_max)
            )
            stop_stats = torch.stack(
                [
                    accepted_count / float(max_steps),
                    occupied.to(dtype).mean(dim=(-2, -1)),
                    finite_max,
                    dynamic_eligible.to(dtype).mean(dim=1),
                ],
                dim=1,
            )
            if self.zero_stop_one_step_r4212_enabled or self.clean_core_v560_enabled:
                stop_score = candidate_score.new_zeros((b,))
            else:
                stop_score = self.composer_stop_head(
                    torch.cat([global_feature, state_feature, stop_stats], dim=1)
                ).squeeze(-1) + self.composer_stop_bias
            # No remaining safe candidate means Stop is mandatory.
            stop_score = torch.where(
                dynamic_eligible.any(dim=1),
                stop_score,
                stop_score.new_full(stop_score.shape, 1.0e4),
            )
            logits = torch.cat([candidate_logits, stop_score[:, None]], dim=1)
            step_logits.append(logits)

            if forced_choices is None:
                choice = logits.detach().argmax(dim=1)
            else:
                # V552-R3: advance the training state with the exact teacher
                # choice, never with the student's current argmax. Student
                # logits remain differentiable and are used only by loss.
                choice = forced_choices[:, step]
            chosen_index = choice.clamp(max=n - 1)
            chosen_is_candidate = (choice < n) & active
            chosen_is_eligible = dynamic_eligible[batch, chosen_index]
            accept = chosen_is_candidate & chosen_is_eligible
            step_selected.append(torch.where(accept, chosen_index, choice.new_full(choice.shape, n)))

            chosen_mask = mask_bool_all[batch, chosen_index]
            use_mask = chosen_mask & (~occupied) & accept[:, None, None]
            chosen_delta = exact_logit_delta[batch, chosen_index]
            signed_delta[:, 0] = signed_delta[:, 0] + chosen_delta * use_mask.to(dtype)

            chosen_action = action_index[batch, chosen_index]
            chosen_action_oh = F.one_hot(chosen_action.clamp(0, 3), num_classes=4).to(dtype)
            action_weight = torch.maximum(
                action_weight,
                chosen_action_oh[:, :, None, None] * use_mask[:, None].to(dtype),
            )
            remove = chosen_action_oh[:, 0] + chosen_action_oh[:, 2]
            add = chosen_action_oh[:, 1] + chosen_action_oh[:, 3]
            chosen_polarity = torch.stack([remove, add], dim=1)
            polarity_weight = torch.maximum(
                polarity_weight,
                chosen_polarity[:, :, None, None] * use_mask[:, None].to(dtype),
            )
            accepted[batch, chosen_index] |= accept
            chosen_score = candidate_score[batch, chosen_index]
            newly_first = accept & (~first_set)
            selected_index = torch.where(newly_first, chosen_index, selected_index)
            selected_score = torch.where(newly_first, chosen_score, selected_score)
            first_set |= accept
            composition_score += torch.where(
                accept, chosen_score, torch.zeros_like(chosen_score)
            )
            accepted_count += accept.to(dtype)
            occupied |= use_mask
            chosen_sign = action_sign_all[batch, chosen_index]
            occupied_sign = torch.where(
                use_mask,
                chosen_sign[:, None, None].expand_as(occupied_sign),
                occupied_sign,
            )
            # Teacher-decoupled traces remain active even when the current model
            # predicts Stop.  This guarantees full step-2/step-3 supervision.
            if not force_all_steps_active:
                active = active & accept

        final_probability = torch.sigmoid(base_logit + signed_delta).clamp(
            EPS, 1.0 - EPS
        )
        changed_fraction = occupied.to(dtype).mean(dim=(-2, -1))
        return {
            "selected_action_weight": action_weight,
            "selected_polarity_weight": polarity_weight,
            "selected_signed_delta": signed_delta,
            "selected_final_probability": final_probability,
            "accepted_slots": accepted,
            "selected_index": selected_index,
            "selected_score": selected_score,
            "composition_score": composition_score,
            "predicted_execute": accepted.any(dim=1),
            "accepted_count": accepted_count,
            "changed_fraction": changed_fraction,
            "composer_step_logits": torch.stack(step_logits, dim=1),
            "composer_step_candidate_scores": torch.stack(
                step_candidate_scores, dim=1
            ),
            "composer_step_state_probs": torch.stack(step_states, dim=1),
            "composer_step_eligible": torch.stack(step_eligible, dim=1),
            "composer_step_active": torch.stack(step_active, dim=1),
            "composer_step_selected_index": torch.stack(step_selected, dim=1),
            "composer_conflict_reject_rate": torch.stack(
                [
                    ((conflict > 0.0) & safe).to(dtype).mean()
                    for _ in range(1)
                ]
            ).mean(),
            "composer_budget_reject_rate": torch.stack(
                [
                    ((budget_excess > 0.0) & safe).to(dtype).mean()
                    for _ in range(1)
                ]
            ).mean(),
        }

    def forward(
        self,
        *,
        feature: torch.Tensor,
        action_candidates: torch.Tensor,
        base_probability: torch.Tensor,
        cause_probability: torch.Tensor,
        action_alpha: torch.Tensor,
        entropy: torch.Tensor,
        boundary: torch.Tensor,
        deploy_enabled: bool,
        supervision_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if action_candidates.ndim != 4 or action_candidates.shape[1] != 4:
            raise ValueError(
                "V551 action_candidates must be [B,4,H,W], got "
                f"{tuple(action_candidates.shape)}"
            )
        b, _, h, w = action_candidates.shape
        (
            parent_mask_logits,
            parent_masks,
            parent_scale_logits,
            parent_scale_probs,
            fused_feature,
        ) = self._multiscale_masks(feature)
        parent_r47_anchor = parent_masks.new_zeros((b, self.num_slots, 4))
        parent_r413_proposal_anchor = parent_r47_anchor.clone()
        parent_r47_query_logits = parent_mask_logits.new_zeros(parent_mask_logits.shape)
        r48_stage_logits = parent_mask_logits[:, None]
        r48_stage_anchors = parent_r47_anchor[:, None]
        r48_dn_stage_logits = parent_mask_logits.new_zeros((b, 1, 1, h, w))
        r48_dn_stage_anchors = parent_r47_anchor.new_zeros((b, 1, 1, 4))
        r48_dn_target_index = torch.zeros((b, 1), device=parent_masks.device, dtype=torch.long)
        r48_dn_valid = torch.zeros((b, 1), device=parent_masks.device, dtype=torch.bool)
        r48_teacher_masks = parent_masks.new_zeros((b, self.num_slots, h, w))
        r48_teacher_actions = torch.zeros((b, self.num_slots), device=parent_masks.device, dtype=torch.long)
        r48_teacher_valid = torch.zeros((b, self.num_slots), device=parent_masks.device, dtype=torch.bool)
        r48_teacher_area = parent_masks.new_zeros((b, self.num_slots))
        r48_teacher_geometry = parent_r47_anchor.new_zeros((b, self.num_slots, 4))
        r48_teacher_effective = base_probability[:, :1].detach()
        r48_teacher_error = parent_masks.new_zeros((b, 1, h, w))
        r48_attention_entropy = parent_masks.new_zeros(())
        v561_bcrs = None
        v561_geometry_owner = parent_masks.new_zeros(())
        v561_mask_probability_mean = parent_masks.new_zeros(())
        v561_soft_overlap_mass = parent_masks.new_zeros(())
        v561_q0_pairwise_cosine = parent_masks.new_zeros(())
        v561_q1_pairwise_cosine = parent_masks.new_zeros(())
        v561_q2_pairwise_cosine = parent_masks.new_zeros(())
        v561_stage1_query_delta_norm = parent_masks.new_zeros(())
        v561_stage2_query_delta_norm = parent_masks.new_zeros(())
        v561_typed_support_mean = parent_masks.new_zeros(())
        v561_typed_support_std = parent_masks.new_zeros(())
        v561_typed_support_neutrality_error = parent_masks.new_zeros(())
        v561_stage1_attention_entropy_ratio = parent_masks.new_zeros(())
        v561_stage2_attention_entropy_ratio = parent_masks.new_zeros(())
        v561_mask_bias_mean = parent_masks.new_zeros(())
        v562_residual_logits = parent_masks.new_zeros((b, 1, h, w))
        v562_residual_probability_mean = parent_masks.new_zeros(())
        v562_proposal_anchor_xy = parent_masks.new_zeros((b, self.num_slots, 2))
        v563_mask_window = parent_masks.new_ones((b, self.num_slots, h, w))
        v563_attention_window = parent_masks.new_ones((b, self.num_slots, h, w))
        v563_raw_mask_probability = parent_masks.new_zeros((b, self.num_slots, h, w))
        v563_pre_gate_mask_probability_mean = parent_masks.new_zeros(())
        v563_outside_mask_probability = parent_masks.new_zeros(())
        v563_attention_window_fraction = parent_masks.new_zeros(())
        v563_mask_window_fraction = parent_masks.new_zeros(())
        v563_identity_retention_q1 = parent_masks.new_zeros(())
        v563_identity_retention_q2 = parent_masks.new_zeros(())
        v564_rootfix_enabled = parent_masks.new_zeros(())
        v564_proposal_radius = parent_masks.new_zeros((b, self.num_slots))
        v564_proposal_radius_mean = parent_masks.new_zeros(())
        v564_proposal_shape_prior_abs_mean = parent_masks.new_zeros(())
        v564_dual_stream_identity_enabled = parent_masks.new_zeros(())
        v564_typed_spatial_feedback_disabled = parent_masks.new_zeros(())
        v565_rootfix_enabled = parent_masks.new_zeros(())
        v565_seed_logits = parent_masks.new_zeros((b, 1, h, w))
        v565_seed_probability_mean = parent_masks.new_zeros(())
        v565_relative_support_mean = parent_masks.new_zeros(())
        v565_attention_radius_mean = parent_masks.new_zeros(())
        v565_peak_to_background_contrast_mean = parent_masks.new_zeros(())
        v565_shape_condition_abs_mean = parent_masks.new_zeros(())
        clean_attention_precision_mean = parent_masks.new_zeros(())
        clean_mask_precision_mean = parent_masks.new_zeros(())
        clean_loss_log_vars = parent_masks.new_zeros((4,))
        tc_stage0_logits = parent_mask_logits.new_zeros((b, self.num_slots, h, w))
        tc_pilot_logits = parent_mask_logits.new_zeros((b, self.num_slots, h, w))
        tc_pilot_action_logits = parent_masks.new_zeros((b, self.num_slots, 4))
        tc_pilot_valid = torch.zeros((b, self.num_slots), device=parent_masks.device, dtype=torch.bool)
        tc_pilot_teacher_masks = parent_masks.new_zeros((b, self.num_slots, h, w))
        tc_pilot_teacher_actions = torch.zeros((b, self.num_slots), device=parent_masks.device, dtype=torch.long)
        tc_teacher_raw_count = parent_masks.new_zeros((b,))
        r49_attention_max_weight = parent_masks.new_zeros(())
        r49_dn_noise_scale = parent_masks.new_zeros(())
        r410_proposal_score_mean = parent_masks.new_zeros(())
        r410_proposal_valid_fraction = parent_masks.new_zeros(())
        r411_proposal_type = torch.zeros((b, self.num_slots), device=parent_masks.device, dtype=torch.long)
        r411_proposal_score = parent_masks.new_zeros((b, self.num_slots))
        r411_proposal_valid = parent_masks.new_zeros((b, self.num_slots))
        r411_center_logits = parent_masks.new_zeros((b, 4, h, w))
        r411_size_map = parent_masks.new_zeros((b, 4, 2, h, w))
        r411_offset_map = parent_masks.new_zeros((b, 4, 2, h, w))
        parent_r416_proposal_point = parent_masks.new_zeros((b, self.num_slots, 2))
        parent_r416_edge_offsets = parent_masks.new_zeros((b, self.num_slots, 4))
        r416_legacy_unique_fraction = parent_masks.new_zeros(())
        r417_location_logits = parent_masks.new_zeros((b, 1, h, w))
        r417_location_offset_map = parent_masks.new_zeros((b, 2, h, w))
        r418_paired_logits = parent_masks.new_zeros((b, self.num_slots, h, w))
        r418_paired_teacher_masks = parent_masks.new_zeros((b, self.num_slots, h, w))
        r418_paired_teacher_valid = torch.zeros((b, self.num_slots), device=parent_masks.device, dtype=torch.bool)
        r418_paired_coarse = base_probability[:, :1].detach()
        r419_seed_support_fraction = parent_masks.new_zeros(())
        r419_final_support_fraction = parent_masks.new_zeros(())
        r419_outside_mask_probability = parent_masks.new_zeros(())
        r420_dynamic_mask_enabled = parent_masks.new_zeros(())
        r420_type_decoupled_mask_enabled = parent_masks.new_zeros(())
        r420_dynamic_channels = parent_masks.new_zeros(())
        r420_relative_coord_mean_abs = parent_masks.new_zeros(())
        r4201_clean_rootfix_enabled = parent_masks.new_zeros(())
        r4203_dense_set_enabled = parent_masks.new_zeros(())
        r4203_ownership_sum_error = parent_masks.new_zeros(())
        r4203_assignment_entropy = parent_masks.new_zeros(())
        r4203_background_fraction = parent_masks.new_zeros(())
        r4203_max_slot_ownership = parent_masks.new_zeros(())
        r4203_slot_mass_cv = parent_masks.new_zeros(())
        r4203_point_bottleneck_used = parent_masks.new_zeros(())
        r4204_occupancy_logits = parent_masks.new_zeros((b, 1, h, w))
        r4204_residual_mass_conservation_error = parent_masks.new_zeros(())
        r4204_conditional_slot_entropy = parent_masks.new_zeros(())
        r4204_conditional_max_slot_probability = parent_masks.new_zeros(())
        r4204_residual_existence_mean = parent_masks.new_zeros(())
        r4204_centroid_separation = parent_masks.new_zeros(())
        r4204_spatial_variance_mean = parent_masks.new_zeros(())
        r4204_spatial_identity_enabled = parent_masks.new_zeros(())
        r4204_location_as_occupancy_used = parent_masks.new_zeros(())
        r4205_enabled = parent_masks.new_zeros(())
        r4205_overflow_probability = parent_masks.new_zeros((b, 1, h, w))
        r4205_overflow_identity_probability = parent_masks.new_zeros((b, 1, h, w))
        r4205_conditional_identity_logits = parent_masks.new_zeros((b, self.num_slots + 1, h, w))
        r4205_editable_probability_sum = parent_masks.new_zeros((b, 1, h, w))
        r4205_overflow_probability_mean = parent_masks.new_zeros(())
        r4205_overflow_conditional_mean = parent_masks.new_zeros(())
        r4205_editable_probability_mean = parent_masks.new_zeros(())
        r4205_final_logits_finite_fraction = parent_masks.new_ones(())
        r4207_enabled = parent_masks.new_zeros(())
        r4207_seed_center_xy = parent_masks.new_zeros((b, self.num_slots, 2))
        r4207_seed_score = parent_masks.new_zeros((b, self.num_slots))
        r4207_seed_valid = parent_masks.new_zeros((b, self.num_slots))
        r4207_seed_feature_finite_fraction = parent_masks.new_ones(())
        r4207_seed_valid_fraction = parent_masks.new_zeros(())
        r4207_seed_score_mean = parent_masks.new_zeros(())
        r4207_seed_pairwise_distance_px = parent_masks.new_zeros(())
        r4207_q0_pairwise_cosine = parent_masks.new_zeros(())
        r4207_q1_pairwise_cosine = parent_masks.new_zeros(())
        r4207_seed_to_slot_centroid_drift_px = parent_masks.new_zeros(())
        r4207_full_image_assignment_enabled = parent_masks.new_zeros(())
        r4207_hard_spatial_support_used = parent_masks.new_zeros(())
        r4208_enabled = parent_masks.new_zeros(())
        r4208_normalized_fusion_enabled = parent_masks.new_zeros(())
        r4208_persistent_identity_enabled = parent_masks.new_zeros(())
        r4208_learned_query_norm = parent_masks.new_zeros(())
        r4208_seed_feature_norm = parent_masks.new_zeros(())
        r4208_seed_to_learned_norm_ratio = parent_masks.new_zeros(())
        r4208_seed_feature_pairwise_cosine = parent_masks.new_zeros(())
        r4208_q0_seed_identity_cosine = parent_masks.new_zeros(())
        r4208_q1_seed_identity_cosine = parent_masks.new_zeros(())
        r4208_identity_retention_delta = parent_masks.new_zeros(())
        r4210_enabled = parent_masks.new_zeros(())
        r4210_variable_seed_enabled = parent_masks.new_zeros(())
        r4210_independent_overflow_enabled = parent_masks.new_zeros(())
        r4210_valid_seed_count = parent_masks.new_zeros(())
        r4210_seed_logit = parent_masks.new_zeros((b, self.num_slots))
        r4210_seed_logit_mean = parent_masks.new_zeros(())
        r4210_overflow_gate_logits = parent_masks.new_zeros((b, 1, h, w))
        r4210_overflow_conditional_probability = parent_masks.new_zeros((b, 1, h, w))
        r4210_overflow_logit_mean = parent_masks.new_zeros(())
        r4210_overflow_conditional_mean = parent_masks.new_zeros(())
        r4211_enabled = parent_masks.new_zeros(())
        r4211_proposal_existence_enabled = parent_masks.new_zeros(())
        r4211_geometry_overflow_enabled = parent_masks.new_zeros(())
        r4211_proposal_seed_count = parent_masks.new_zeros(())
        r4211_proposal_confidence_mean = parent_masks.new_zeros(())
        r4211_geometry_effective_l1 = parent_masks.new_zeros(())
        r4211_geometry_slot_probability = parent_masks.new_zeros((b, self.num_slots, h, w))
        r4211_effective_slot_probability = parent_masks.new_zeros((b, self.num_slots, h, w))
        r48_teacher_built = parent_masks.new_zeros(())
        r48_teacher_valid_count = parent_masks.new_zeros(())
        v560_direct_mask_probability_mean = parent_masks.new_zeros(())
        v560_independent_soft_overlap_mass = parent_masks.new_zeros(())
        v560_q0_pairwise_cosine = parent_masks.new_zeros(())
        v560_q1_pairwise_cosine = parent_masks.new_zeros(())
        v560_mask_bias_mean = parent_masks.new_zeros(())
        if self.base_conditioned_residual_set_v561_enabled:
            # V561 is a first-class geometry owner.  Unlike the historical V560
            # helper, it is NOT hidden behind the R4.8 iterative-binding flag.
            # This closes the real V560R2 plumbing bug where the direct-query
            # function existed but the formal config never called it.
            v561_bcrs = self._v561_base_conditioned_query_set(
                fused_feature=fused_feature,
                base_probability=base_probability,
                cause_probability=cause_probability,
                action_alpha=action_alpha,
                entropy=entropy,
                boundary=boundary,
                supervision_masks=supervision_masks,
            )
            parent_mask_logits = v561_bcrs["final_logits"]
            parent_masks = v561_bcrs["slot_probability"]
            parent_r47_anchor = v561_bcrs["soft_anchor"]
            parent_r413_proposal_anchor = parent_r47_anchor
            parent_r47_query_logits = parent_mask_logits
            v561_geometry_owner = parent_masks.new_ones(())
            v561_mask_probability_mean = v561_bcrs["mask_probability_mean"]
            v561_soft_overlap_mass = v561_bcrs["soft_overlap_mass"]
            v561_q0_pairwise_cosine = v561_bcrs["q0_pairwise_cosine"]
            v561_q1_pairwise_cosine = v561_bcrs["q1_pairwise_cosine"]
            v561_q2_pairwise_cosine = v561_bcrs["q2_pairwise_cosine"]
            v561_stage1_query_delta_norm = v561_bcrs["stage1_query_delta_norm"]
            v561_stage2_query_delta_norm = v561_bcrs["stage2_query_delta_norm"]
            v561_typed_support_mean = v561_bcrs["typed_support_mean"]
            v561_typed_support_std = v561_bcrs["typed_support_std"]
            v561_typed_support_neutrality_error = v561_bcrs["typed_support_neutrality_error"]
            v561_stage1_attention_entropy_ratio = v561_bcrs["stage1_attention_entropy_ratio"]
            v561_stage2_attention_entropy_ratio = v561_bcrs["stage2_attention_entropy_ratio"]
            v561_mask_bias_mean = v561_bcrs["mask_bias_mean"]
            if self.bcrs_v561_variant in {"rootfix", "persistent"}:
                v562_residual_logits = v561_bcrs["v562_residual_logits"]
                v562_residual_probability_mean = v561_bcrs["v562_residual_probability"].mean().detach()
                v562_proposal_anchor_xy = v561_bcrs["v562_proposal_anchor_xy"]
                clean_attention_precision_mean = v561_bcrs.get("clean_attention_precision_mean", clean_attention_precision_mean)
                clean_mask_precision_mean = v561_bcrs.get("clean_mask_precision_mean", clean_mask_precision_mean)
                clean_loss_log_vars = v561_bcrs.get("clean_loss_log_vars", clean_loss_log_vars)
                tc_stage0_logits = v561_bcrs.get("tc_stage0_logits", tc_stage0_logits)
                tc_pilot_logits = v561_bcrs.get("tc_pilot_logits", tc_pilot_logits)
                tc_pilot_action_logits = v561_bcrs.get("tc_pilot_action_logits", tc_pilot_action_logits)
                tc_pilot_valid = v561_bcrs.get("tc_pilot_valid", tc_pilot_valid)
                tc_pilot_teacher_masks = v561_bcrs.get("tc_pilot_teacher_masks", tc_pilot_teacher_masks)
                tc_pilot_teacher_actions = v561_bcrs.get("tc_pilot_teacher_actions", tc_pilot_teacher_actions)
                tc_teacher_raw_count = v561_bcrs.get("tc_teacher_raw_count", tc_teacher_raw_count)
            if self.bcrs_v561_variant == "persistent":
                v563_mask_window = v561_bcrs["v563_mask_window"]
                v563_attention_window = v561_bcrs["v563_attention_window"]
                v563_raw_mask_probability = v561_bcrs["v563_raw_mask_probability"]
                v563_pre_gate_mask_probability_mean = v561_bcrs["v563_pre_gate_mask_probability_mean"]
                v563_outside_mask_probability = v561_bcrs["v563_outside_mask_probability"]
                v563_attention_window_fraction = v561_bcrs["v563_attention_window_fraction"]
                v563_mask_window_fraction = v561_bcrs["v563_mask_window_fraction"]
                v563_identity_retention_q1 = v561_bcrs["v563_identity_retention_q1"]
                v563_identity_retention_q2 = v561_bcrs["v563_identity_retention_q2"]
                if self.v564_rootfix_enabled:
                    v564_rootfix_enabled = v561_bcrs["v564_rootfix_enabled"]
                    v564_proposal_radius = v561_bcrs["v564_proposal_radius"]
                    v564_proposal_radius_mean = v561_bcrs["v564_proposal_radius_mean"]
                    v564_proposal_shape_prior_abs_mean = v561_bcrs["v564_proposal_shape_prior_abs_mean"]
                    v564_dual_stream_identity_enabled = v561_bcrs["v564_dual_stream_identity_enabled"]
                    v564_typed_spatial_feedback_disabled = v561_bcrs["v564_typed_spatial_feedback_disabled"]
                    if self.v565_rootfix_enabled:
                        v565_rootfix_enabled = v561_bcrs["v565_rootfix_enabled"]
                        v565_seed_logits = v561_bcrs["v565_seed_logits"]
                        v565_seed_probability_mean = v561_bcrs["v565_seed_probability_mean"]
                        v565_relative_support_mean = v561_bcrs["v565_relative_support_mean"]
                        v565_attention_radius_mean = v561_bcrs["v565_attention_radius_mean"]
                        v565_peak_to_background_contrast_mean = v561_bcrs["v565_peak_to_background_contrast_mean"]
                        v565_shape_condition_abs_mean = v561_bcrs["v565_shape_condition_abs_mean"]
        elif self.iterative_binding_r48_enabled:
            r48 = self._r48_iterative_query_masks(
                coarse_logits=parent_mask_logits,
                fused_feature=fused_feature,
                base_probability=base_probability,
                cause_probability=cause_probability,
                action_alpha=action_alpha,
                entropy=entropy,
                boundary=boundary,
                supervision_masks=supervision_masks,
            )
            parent_mask_logits = r48["final_logits"]
            parent_masks = (
                r48["raw_masks"]
                if self.native_residual_set_r411_enabled and self.r411_use_raw_native_masks
                else r48["smooth_masks"]
            )
            parent_r47_anchor = r48["final_anchor"]
            parent_r413_proposal_anchor = r48.get("initial_anchor", r48["final_anchor"])
            parent_r47_query_logits = parent_mask_logits
            r48_stage_logits = r48["stage_logits"]
            r48_stage_anchors = r48["stage_anchors"]
            r48_dn_stage_logits = r48["dn_stage_logits"]
            r48_dn_stage_anchors = r48["dn_stage_anchors"]
            r48_dn_target_index = r48["dn_target_index"]
            r48_dn_valid = r48["dn_valid"]
            r48_teacher_masks = r48["teacher_masks"]
            r48_teacher_actions = r48["teacher_actions"]
            r48_teacher_valid = r48["teacher_valid"]
            r48_teacher_area = r48["teacher_area"]
            r48_teacher_geometry = r48["teacher_geometry"]
            r48_teacher_effective = r48["teacher_effective"]
            r48_teacher_error = r48["teacher_error"]
            r48_attention_entropy = r48["attention_entropy"]
            r49_attention_max_weight = r48["attention_max_weight"]
            r49_dn_noise_scale = r48["dn_noise_scale"]
            v560_direct_mask_probability_mean = r48.get(
                "v560_direct_mask_probability_mean", v560_direct_mask_probability_mean
            )
            v560_independent_soft_overlap_mass = r48.get(
                "v560_independent_soft_overlap_mass", v560_independent_soft_overlap_mass
            )
            v560_q0_pairwise_cosine = r48.get(
                "v560_q0_pairwise_cosine", v560_q0_pairwise_cosine
            )
            v560_q1_pairwise_cosine = r48.get(
                "v560_q1_pairwise_cosine", v560_q1_pairwise_cosine
            )
            v560_mask_bias_mean = r48.get(
                "v560_mask_bias_mean", v560_mask_bias_mean
            )
            r410_proposal_score_mean = r48["proposal_score_mean"]
            r410_proposal_valid_fraction = r48["proposal_valid_fraction"]
            r411_proposal_score = r48["proposal_score"]
            r411_proposal_valid = r48["proposal_valid"]
            r411_proposal_type = r48["proposal_type"]
            r411_center_logits = r48["r411_center_logits"]
            r411_size_map = r48["r411_size_map"]
            r411_offset_map = r48["r411_offset_map"]
            parent_r416_proposal_point = r48.get("r416_proposal_point", parent_r416_proposal_point)
            parent_r416_edge_offsets = r48.get("r416_edge_offsets", parent_r416_edge_offsets)
            r416_legacy_unique_fraction = r48.get("r416_legacy_unique_fraction", r416_legacy_unique_fraction)
            r417_location_logits = r48.get("r417_location_logits", r417_location_logits)
            r417_location_offset_map = r48.get("r417_location_offset_map", r417_location_offset_map)
            r418_paired_logits = r48.get("r418_paired_logits", r418_paired_logits)
            r418_paired_teacher_masks = r48.get("r418_paired_teacher_masks", r418_paired_teacher_masks)
            r418_paired_teacher_valid = r48.get("r418_paired_teacher_valid", r418_paired_teacher_valid)
            r418_paired_coarse = r48.get("r418_paired_coarse", r418_paired_coarse)
            r419_seed_support_fraction = r48.get("r419_seed_support_fraction", r419_seed_support_fraction)
            r419_final_support_fraction = r48.get("r419_final_support_fraction", r419_final_support_fraction)
            r419_outside_mask_probability = r48.get("r419_outside_mask_probability", r419_outside_mask_probability)
            r420_dynamic_mask_enabled = r48.get("r420_dynamic_mask_enabled", r420_dynamic_mask_enabled)
            r420_type_decoupled_mask_enabled = r48.get("r420_type_decoupled_mask_enabled", r420_type_decoupled_mask_enabled)
            r420_dynamic_channels = r48.get("r420_dynamic_channels", r420_dynamic_channels)
            r420_relative_coord_mean_abs = r48.get("r420_relative_coord_mean_abs", r420_relative_coord_mean_abs)
            r4201_clean_rootfix_enabled = r48.get("r4201_clean_rootfix_enabled", r4201_clean_rootfix_enabled)
            r4203_dense_set_enabled = r48.get("r4203_dense_set_enabled", r4203_dense_set_enabled)
            r4203_ownership_sum_error = r48.get("r4203_ownership_sum_error", r4203_ownership_sum_error)
            r4203_assignment_entropy = r48.get("r4203_assignment_entropy", r4203_assignment_entropy)
            r4203_background_fraction = r48.get("r4203_background_fraction", r4203_background_fraction)
            r4203_max_slot_ownership = r48.get("r4203_max_slot_ownership", r4203_max_slot_ownership)
            r4203_slot_mass_cv = r48.get("r4203_slot_mass_cv", r4203_slot_mass_cv)
            r4203_point_bottleneck_used = r48.get("r4203_point_bottleneck_used", r4203_point_bottleneck_used)
            r4204_occupancy_logits = r48.get("r4204_occupancy_logits", r4204_occupancy_logits)
            r4204_residual_mass_conservation_error = r48.get("r4204_residual_mass_conservation_error", r4204_residual_mass_conservation_error)
            r4204_conditional_slot_entropy = r48.get("r4204_conditional_slot_entropy", r4204_conditional_slot_entropy)
            r4204_conditional_max_slot_probability = r48.get("r4204_conditional_max_slot_probability", r4204_conditional_max_slot_probability)
            r4204_residual_existence_mean = r48.get("r4204_residual_existence_mean", r4204_residual_existence_mean)
            r4204_centroid_separation = r48.get("r4204_centroid_separation", r4204_centroid_separation)
            r4204_spatial_variance_mean = r48.get("r4204_spatial_variance_mean", r4204_spatial_variance_mean)
            r4204_spatial_identity_enabled = r48.get("r4204_spatial_identity_enabled", r4204_spatial_identity_enabled)
            r4204_location_as_occupancy_used = r48.get("r4204_location_as_occupancy_used", r4204_location_as_occupancy_used)
            r4205_enabled = r48.get("r4205_enabled", r4205_enabled)
            r4205_overflow_probability = r48.get("r4205_overflow_probability", r4205_overflow_probability)
            r4205_overflow_identity_probability = r48.get("r4205_overflow_identity_probability", r4205_overflow_identity_probability)
            r4205_conditional_identity_logits = r48.get("r4205_conditional_identity_logits", r4205_conditional_identity_logits)
            r4205_editable_probability_sum = r48.get("r4205_editable_probability_sum", r4205_editable_probability_sum)
            r4205_overflow_probability_mean = r48.get("r4205_overflow_probability_mean", r4205_overflow_probability_mean)
            r4205_overflow_conditional_mean = r48.get("r4205_overflow_conditional_mean", r4205_overflow_conditional_mean)
            r4205_editable_probability_mean = r48.get("r4205_editable_probability_mean", r4205_editable_probability_mean)
            r4205_final_logits_finite_fraction = r48.get("r4205_final_logits_finite_fraction", r4205_final_logits_finite_fraction)
            r4207_enabled = r48.get("r4207_enabled", r4207_enabled)
            r4207_seed_center_xy = r48.get("r4207_seed_center_xy", r4207_seed_center_xy)
            r4207_seed_score = r48.get("r4207_seed_score", r4207_seed_score)
            r4207_seed_valid = r48.get("r4207_seed_valid", r4207_seed_valid)
            r4207_seed_feature_finite_fraction = r48.get("r4207_seed_feature_finite_fraction", r4207_seed_feature_finite_fraction)
            r4207_seed_valid_fraction = r48.get("r4207_seed_valid_fraction", r4207_seed_valid_fraction)
            r4207_seed_score_mean = r48.get("r4207_seed_score_mean", r4207_seed_score_mean)
            r4207_seed_pairwise_distance_px = r48.get("r4207_seed_pairwise_distance_px", r4207_seed_pairwise_distance_px)
            r4207_q0_pairwise_cosine = r48.get("r4207_q0_pairwise_cosine", r4207_q0_pairwise_cosine)
            r4207_q1_pairwise_cosine = r48.get("r4207_q1_pairwise_cosine", r4207_q1_pairwise_cosine)
            r4207_seed_to_slot_centroid_drift_px = r48.get("r4207_seed_to_slot_centroid_drift_px", r4207_seed_to_slot_centroid_drift_px)
            r4207_full_image_assignment_enabled = r48.get("r4207_full_image_assignment_enabled", r4207_full_image_assignment_enabled)
            r4207_hard_spatial_support_used = r48.get("r4207_hard_spatial_support_used", r4207_hard_spatial_support_used)
            r4208_enabled = r48.get("r4208_enabled", r4208_enabled)
            r4208_normalized_fusion_enabled = r48.get("r4208_normalized_fusion_enabled", r4208_normalized_fusion_enabled)
            r4208_persistent_identity_enabled = r48.get("r4208_persistent_identity_enabled", r4208_persistent_identity_enabled)
            r4208_learned_query_norm = r48.get("r4208_learned_query_norm", r4208_learned_query_norm)
            r4208_seed_feature_norm = r48.get("r4208_seed_feature_norm", r4208_seed_feature_norm)
            r4208_seed_to_learned_norm_ratio = r48.get("r4208_seed_to_learned_norm_ratio", r4208_seed_to_learned_norm_ratio)
            r4208_seed_feature_pairwise_cosine = r48.get("r4208_seed_feature_pairwise_cosine", r4208_seed_feature_pairwise_cosine)
            r4208_q0_seed_identity_cosine = r48.get("r4208_q0_seed_identity_cosine", r4208_q0_seed_identity_cosine)
            r4208_q1_seed_identity_cosine = r48.get("r4208_q1_seed_identity_cosine", r4208_q1_seed_identity_cosine)
            r4208_identity_retention_delta = r48.get("r4208_identity_retention_delta", r4208_identity_retention_delta)
            r4210_enabled = r48.get("r4210_enabled", r4210_enabled)
            r4210_variable_seed_enabled = r48.get("r4210_variable_seed_enabled", r4210_variable_seed_enabled)
            r4210_independent_overflow_enabled = r48.get("r4210_independent_overflow_enabled", r4210_independent_overflow_enabled)
            r4210_valid_seed_count = r48.get("r4210_valid_seed_count", r4210_valid_seed_count)
            r4210_seed_logit = r48.get("r4210_seed_logit", r4210_seed_logit)
            r4210_seed_logit_mean = r48.get("r4210_seed_logit_mean", r4210_seed_logit_mean)
            r4210_overflow_gate_logits = r48.get("r4210_overflow_gate_logits", r4210_overflow_gate_logits)
            r4210_overflow_conditional_probability = r48.get("r4210_overflow_conditional_probability", r4210_overflow_conditional_probability)
            r4210_overflow_logit_mean = r48.get("r4210_overflow_logit_mean", r4210_overflow_logit_mean)
            r4210_overflow_conditional_mean = r48.get("r4210_overflow_conditional_mean", r4210_overflow_conditional_mean)
            r4211_enabled = r48.get("r4211_enabled", r4211_enabled)
            r4211_proposal_existence_enabled = r48.get("r4211_proposal_existence_enabled", r4211_proposal_existence_enabled)
            r4211_geometry_overflow_enabled = r48.get("r4211_geometry_overflow_enabled", r4211_geometry_overflow_enabled)
            r4211_proposal_seed_count = r48.get("r4211_proposal_seed_count", r4211_proposal_seed_count)
            r4211_proposal_confidence_mean = r48.get("r4211_proposal_confidence_mean", r4211_proposal_confidence_mean)
            r4211_geometry_effective_l1 = r48.get("r4211_geometry_effective_l1", r4211_geometry_effective_l1)
            r4211_geometry_slot_probability = r48.get("r4211_geometry_slot_probability", r4211_geometry_slot_probability)
            r4211_effective_slot_probability = r48.get("r4211_effective_slot_probability", r4211_effective_slot_probability)
            r48_teacher_built = r48.get("teacher_built", r48_teacher_built)
            r48_teacher_valid_count = r48.get("teacher_valid_count", r48_teacher_valid_count)
        elif self.spatial_realization_r47_enabled:
            (
                parent_mask_logits,
                parent_masks,
                parent_r47_anchor,
                parent_r47_query_logits,
            ) = self._r47_query_conditioned_masks(
                coarse_logits=parent_mask_logits,
                fused_feature=fused_feature,
                base_probability=base_probability,
                cause_probability=cause_probability,
                action_alpha=action_alpha,
                entropy=entropy,
                boundary=boundary,
            )

        if self.error_aware_r45_enabled and not self.native_contract_r46_enabled:
            cause_summary = cause_probability.mean(dim=1, keepdim=True)
            error_input = torch.cat(
                [
                    fused_feature,
                    base_probability[:, :1].detach(),
                    entropy[:, :1].detach(),
                    boundary[:, :1].detach(),
                    cause_summary.detach(),
                ],
                dim=1,
            )
            error_prone_logits = self.error_prone_head(error_input)
            error_prone_probs = torch.sigmoid(error_prone_logits)
        else:
            error_prone_logits = fused_feature.new_zeros((b, 1, h, w))
            error_prone_probs = torch.sigmoid(error_prone_logits)

        if self.error_aware_r45_enabled and not self.native_contract_r46_enabled:
            # Soft, neutral-at-init spatial conditioning.  p(error)=0.5 gives
            # multiplier 1.0; likely-error pixels are amplified up to 1.25 and
            # likely-correct pixels suppressed down to 0.75.  This is not a hard
            # gate and never uses GT at inference.
            error_guidance = 0.75 + 0.50 * error_prone_probs.detach()
            parent_masks = (parent_masks * error_guidance).clamp(EPS, 1.0 - EPS)
            parent_mask_logits = torch.logit(parent_masks)

        # Region descriptors are computed on the dynamically routed multiscale
        # support, while the Base/cause/action statistics remain identical to
        # the V538 causal anchor contract.
        flat_masks = parent_masks.reshape(b * self.num_slots, 1, h, w)
        dilated = F.max_pool2d(flat_masks, 3, stride=1, padding=1)
        ring_masks = (dilated - flat_masks).clamp(0.0, 1.0).reshape(
            b, self.num_slots, h, w
        )
        inside_feature = _masked_mean(fused_feature, parent_masks)
        ring_feature = _masked_mean(fused_feature, ring_masks)
        cause_mean = _masked_scalar(cause_probability, parent_masks)
        alpha_mean = _masked_scalar(action_alpha, parent_masks)
        base_mean = _masked_scalar(base_probability, parent_masks)
        entropy_mean = _masked_scalar(entropy, parent_masks)
        boundary_mean = _masked_scalar(boundary, parent_masks)
        parent_area = parent_masks.mean(dim=(-2, -1)).unsqueeze(-1)
        descriptor = torch.cat(
            [
                inside_feature,
                ring_feature,
                cause_mean,
                alpha_mean,
                base_mean,
                entropy_mean,
                boundary_mean,
                parent_area,
            ],
            dim=2,
        )
        parent_feature = self.slot_trunk(descriptor)
        if self.base_conditioned_residual_set_v561_enabled:
            if v561_bcrs is None:
                raise RuntimeError("V561 geometry owner did not produce query outputs")
            # The same query action head owns both typed spatial support and the
            # executable four-way correction type.  This prevents a second
            # descriptor head from disagreeing with the action that conditioned
            # query-to-image binding.
            parent_action_logits = v561_bcrs["action_logits"]
        else:
            parent_action_logits = self.action_head(parent_feature)
        parent_action_probs = F.softmax(
            parent_action_logits / self.action_temperature, dim=2
        )
        parent_polarity_logits = self.polarity_head(parent_feature)
        parent_polarity_probs = F.softmax(
            parent_polarity_logits / self.polarity_temperature, dim=2
        )
        parent_dose_logits = self.dose_head(parent_feature).squeeze(-1)
        parent_raw_doses = self.minimum_dose + (
            self.maximum_dose - self.minimum_dose
        ) * torch.sigmoid(parent_dose_logits)
        remove_alpha = 0.5 * (alpha_mean[:, :, 0] + alpha_mean[:, :, 2])
        add_alpha = 0.5 * (alpha_mean[:, :, 1] + alpha_mean[:, :, 3])
        polarity_alpha = (
            parent_polarity_probs[:, :, 0] * remove_alpha
            + parent_polarity_probs[:, :, 1] * add_alpha
        ).clamp(0.0, 1.0)
        parent_dose_gate = 0.25 + 0.75 * polarity_alpha
        parent_doses = (parent_raw_doses * parent_dose_gate).clamp(
            self.minimum_dose * 0.25, self.maximum_dose
        )
        if self.base_conditioned_residual_set_v561_enabled and self.bcrs_v561_variant in {"rootfix", "persistent"}:
            if v561_bcrs is None:
                raise RuntimeError("V562 query-owned presence requires V561/V562 query outputs")
            parent_presence_logits = v561_bcrs["presence_logits"]
        else:
            parent_presence_logits = self.presence_head(parent_feature).squeeze(-1)
        parent_presence_probs = torch.sigmoid(parent_presence_logits)

        if self.independent_candidate_set_r4212_enabled or self.clean_core_v560_enabled:
            # R4.21.2/V560 executable masks use the canonical Bernoulli decision
            # boundary and a true no-object/existence decision.  Crucially,
            # independent candidates are NOT winner-take-all resolved against
            # each other: overlap is allowed because each slot is a complete
            # alternative correction candidate for M2, not a partition cell.
            parent_hard = parent_mask_logits.detach() > 0.0
            # Presence/no-object owns *execution*, not mask geometry.  Keeping
            # the independent mask proposal alive during training lets Hungarian
            # matching and Utility receive supervision even before the presence
            # logit crosses zero.  The deployment_valid contract below removes
            # absent slots from M2 exactly.
            parent_contrast = (
                parent_masks.flatten(2).amax(dim=2)
                - parent_masks.flatten(2).amin(dim=2)
            ) > torch.finfo(parent_masks.dtype).eps
        else:
            if self.adaptive_cardinality_hard_mask:
                parent_hard, _, parent_contrast = self._adaptive_hard_mask(parent_masks)
            else:
                parent_hard = parent_masks.detach() >= self.deployment_mask_threshold
                parent_contrast = torch.ones(
                    parent_masks.shape[:2], dtype=torch.bool, device=parent_masks.device
                )
            parent_hard = self._resolve_hard_slot_overlap(
                mask_logits=parent_mask_logits, hard_masks=parent_hard
            )

        # Epoch-aligned fast warm-up.  Before atomization is trainable/useful,
        # use each exact parent slot as one atom entirely on GPU.  This avoids
        # the CUDA->CPU->SciPy->CUDA round trip on the first physical epoch.
        use_fast_parent_atoms = self.clean_core_v560_enabled or self.r47_direct_slot_components or (
            self.training and self.current_epoch < self.atomization_start_epoch
        )
        if use_fast_parent_atoms:
            atom_masks = parent_hard
            atom_soft_masks = parent_masks
            parent_index = torch.arange(
                self.num_slots, device=parent_hard.device, dtype=torch.long
            )[None].expand(b, -1)
            atom_valid = parent_contrast & parent_hard.flatten(2).any(dim=2)
            numerator = (parent_masks * parent_hard.to(parent_masks.dtype)).flatten(2).sum(dim=2)
            denominator = parent_hard.flatten(2).sum(dim=2).clamp_min(1)
            atom_confidence = numerator / denominator
        else:
            atom_masks, atom_soft_masks, parent_index, atom_valid, atom_confidence = self._gpu_atomize(
                hard_masks=parent_hard, soft_masks=parent_masks, contrast_active=parent_contrast
            )
        n = atom_masks.shape[1]
        atom_r47_anchor = _repeat_by_parent(parent_r47_anchor, parent_index)
        atom_r413_proposal_anchor = _repeat_by_parent(parent_r413_proposal_anchor, parent_index)
        atom_r416_proposal_point = _repeat_by_parent(parent_r416_proposal_point, parent_index)
        atom_r416_edge_offsets = _repeat_by_parent(parent_r416_edge_offsets, parent_index)
        atom_r47_query_logits = _repeat_by_parent(parent_r47_query_logits, parent_index)
        atom_parent_soft = _repeat_by_parent(parent_masks, parent_index)
        atom_learning_region = _dilate(atom_masks.to(parent_masks.dtype), max(self.editor_region_radii))
        atom_soft_masks = torch.maximum(atom_soft_masks, atom_parent_soft * atom_learning_region * 0.25)
        atom_mask_logits = torch.logit(atom_soft_masks.clamp(EPS, 1.0 - EPS))
        atom_masks_st = (
            atom_masks.to(parent_masks.dtype)
            + atom_soft_masks
            - atom_soft_masks.detach()
        )
        if self.geometry_overflow_decoupling_r4211_enabled:
            # Geometry masks remain the owners of matching/shape.  Capacity
            # rejection is applied only when constructing an actual correction.
            overflow_keep_r4211 = (1.0 - r4210_overflow_conditional_probability).clamp(0.0, 1.0)
            atom_effective_masks_st_r4211 = atom_masks_st * overflow_keep_r4211
        else:
            overflow_keep_r4211 = parent_masks.new_ones((b, 1, h, w))
            atom_effective_masks_st_r4211 = atom_masks_st

        atom_feature = _repeat_by_parent(parent_feature, parent_index)
        atom_action_logits = _repeat_by_parent(parent_action_logits, parent_index)
        atom_action_probs = _repeat_by_parent(parent_action_probs, parent_index)
        atom_polarity_logits = _repeat_by_parent(parent_polarity_logits, parent_index)
        atom_polarity_probs = _repeat_by_parent(parent_polarity_probs, parent_index)
        atom_dose_logits = _repeat_by_parent(parent_dose_logits, parent_index)
        atom_raw_doses = _repeat_by_parent(parent_raw_doses, parent_index)
        atom_dose_gate = _repeat_by_parent(parent_dose_gate, parent_index)
        atom_doses = _repeat_by_parent(parent_doses, parent_index)
        atom_presence_logits = _repeat_by_parent(parent_presence_logits, parent_index)
        atom_presence_probs = _repeat_by_parent(parent_presence_probs, parent_index)
        atom_area_pixels = atom_masks.flatten(2).sum(dim=2)
        atom_quality_gate_active = (
            (not self.training)
            or self.current_epoch >= self.atom_quality_gate_start_epoch
        )
        # R2 physical validity contains only finite/geometric facts.  It is the
        # root mask for M1 and exact Teacher supervision, and therefore must not
        # depend on any learned safety/calibration prediction.
        finite_atom = (
            torch.isfinite(atom_soft_masks).flatten(2).all(dim=2)
            & torch.isfinite(atom_confidence)
            & torch.isfinite(atom_presence_probs)
        )
        atom_valid = atom_valid & finite_atom & (atom_area_pixels >= self.atom_min_pixels)
        atom_scale_logits = _repeat_by_parent(parent_scale_logits, parent_index)
        atom_scale_probs = _repeat_by_parent(parent_scale_probs, parent_index)
        atom_scale_index = atom_scale_probs.detach().argmax(dim=2)
        atom_scale_hard = F.one_hot(
            atom_scale_index, num_classes=self.num_scales
        ).to(atom_scale_probs.dtype)
        atom_scale_st = atom_scale_hard + atom_scale_probs - atom_scale_probs.detach()
        atom_contrast = _repeat_by_parent(parent_contrast, parent_index) & atom_valid

        base = base_probability[:, :1].detach().clamp(EPS, 1.0 - EPS)
        base_logit = _safe_logit(base)[:, 0][:, None]
        action_delta_bank = (
            _safe_logit(action_candidates.clamp(EPS, 1.0 - EPS))
            - _safe_logit(base).expand(-1, 4, -1, -1)
        )

        # M1 typed exact intervention.
        hard_m1_action = atom_action_probs.detach().argmax(dim=2)
        m1_action_onehot = F.one_hot(hard_m1_action, num_classes=4).to(
            atom_action_probs.dtype
        )
        m1_action_st = m1_action_onehot + atom_action_probs - atom_action_probs.detach()
        m1_action_delta = torch.einsum(
            "bna,bahw->bnhw", m1_action_st, action_delta_bank
        )
        # Existing continuous dose is interpreted as a bounded relative scale
        # around the historical initial dose of one.  V561 deliberately removes
        # dose/polarity ownership from M1: a correction instance is exactly
        # (mask, existence, four-way action), so Apply(Base, mask, action) has no
        # hidden amplitude hyperparameter.
        m1_dose_factor = atom_doses.clamp(0.25, 2.0)
        if self.base_conditioned_residual_set_v561_enabled:
            m1_dose_factor = torch.ones_like(m1_dose_factor)

        parent_boundary_residual = torch.tanh(
            self.boundary_residual_head(fused_feature)
        ) * self.boundary_residual_cap
        atom_boundary_residual = _repeat_by_parent(
            parent_boundary_residual, parent_index
        )
        boundary_regions = []
        for radius in self.boundary_band_radii:
            boundary_regions.append(_boundary_band(atom_masks_st, radius))
        boundary_region = (
            torch.stack(boundary_regions, dim=2)
            * atom_scale_st[..., None, None]
        ).sum(dim=2)
        if (
            not self.boundary_residual_enabled
            or self.base_conditioned_residual_set_v561_enabled
        ):
            atom_boundary_residual = atom_boundary_residual * 0.0
        m1_boundary_delta = atom_boundary_residual * boundary_region
        if self.geometry_overflow_decoupling_r4211_enabled:
            m1_boundary_delta = m1_boundary_delta * overflow_keep_r4211
        if self.base_conditioned_residual_set_v561_enabled and self.bcrs_v561_variant in {"rootfix", "persistent"}:
            # V562 exact physical executor.  The four-way action owns the target
            # binary state directly: Delete/Trim -> 0, Fill/Expand -> 1.  No old
            # action-alpha/dose magnitude can prevent a correct action from
            # crossing the 0.5 decision boundary.
            add_probability_v562 = (
                atom_action_probs[:, :, 1] + atom_action_probs[:, :, 3]
            ).clamp(0.0, 1.0)
            hard_add_v562 = (
                (hard_m1_action == 1) | (hard_m1_action == 3)
            ).to(atom_action_probs.dtype)
            target_add_st_v562 = (
                hard_add_v562
                + add_probability_v562
                - add_probability_v562.detach()
            )
            target_probability_v562 = (
                EPS + (1.0 - 2.0 * EPS) * target_add_st_v562
            )[:, :, None, None]
            execution_mask_v562 = atom_effective_masks_st_r4211.clamp(0.0, 1.0)
            base_probability_v562 = base[:, 0][:, None]
            m1_exact_st = (
                base_probability_v562 * (1.0 - execution_mask_v562)
                + target_probability_v562 * execution_mask_v562
            ).clamp(EPS, 1.0 - EPS)
            m1_logit_delta = _safe_logit(m1_exact_st) - base_logit
        else:
            m1_logit_delta = (
                atom_effective_masks_st_r4211
                * m1_action_delta
                * m1_dose_factor[:, :, None, None]
                + m1_boundary_delta
            )
            m1_exact_st = torch.sigmoid(base_logit + m1_logit_delta).clamp(
                EPS, 1.0 - EPS
            )
        m1_exact = m1_exact_st.detach()

        # V552-R4.20.9/4.21.0 deployable M1-only path.
        # R4.20.9 measured the inference-visible M1 result under no_grad.
        # R4.21.0 C4 keeps *the same hard winner in the forward pass* but uses
        # a straight-through normalized score distribution in backward, so the
        # exact deployable intervention can receive a standard segmentation
        # objective without introducing M2 or GT into Native forward.
        m1_diag_area = atom_masks.to(parent_masks.dtype).mean(dim=(-2, -1))
        m1_diag_valid = (
            atom_valid
            & atom_contrast
            & (m1_diag_area >= self.min_area_fraction)
            & (m1_diag_area <= self.max_component_area_fraction)
            & torch.isfinite(m1_logit_delta).flatten(2).all(dim=2)
        )
        m1_diag_support = atom_masks | (m1_logit_delta.detach().abs() > EPS)
        m1_diag_support_valid = m1_diag_valid[:, :, None, None] & m1_diag_support

        if self.m1_native_alignment_r4210_enabled:
            # The hard forward score is numerically identical to the R4.20.9
            # deployment rule: presence * component confidence.  Only backward
            # uses the proportional soft distribution.  log(score) followed by
            # softmax has no temperature hyperparameter and is exactly score /
            # sum(score) over eligible atoms.
            m1_score_base = (
                atom_presence_probs.clamp(0.0, 1.0)
                * atom_confidence.clamp_min(0.0)
            )[:, :, None, None].expand_as(m1_logit_delta)
            hard_score = m1_score_base.detach().masked_fill(~m1_diag_support_valid, -1.0e4)
            m1_diag_has_support = m1_diag_support_valid.any(dim=1, keepdim=True)
            m1_diag_winner = hard_score.argmax(dim=1, keepdim=True)
            hard_weight = F.one_hot(
                m1_diag_winner[:, 0], num_classes=n
            ).permute(0, 3, 1, 2).to(m1_logit_delta.dtype)
            soft_logit = torch.log(m1_score_base.clamp_min(EPS)).masked_fill(
                ~m1_diag_support_valid, -1.0e4
            )
            soft_weight = torch.softmax(soft_logit, dim=1)
            soft_weight = torch.where(
                m1_diag_has_support, soft_weight, torch.zeros_like(soft_weight)
            )
            winner_st = hard_weight + soft_weight - soft_weight.detach()
            winner_st = torch.where(
                m1_diag_has_support, winner_st, torch.zeros_like(winner_st)
            )
            m1_diag_delta = (winner_st * m1_logit_delta).sum(dim=1, keepdim=True)
            m1_diag_presence = (
                winner_st
                * atom_presence_probs[:, :, None, None].expand_as(m1_logit_delta)
            ).sum(dim=1, keepdim=True)
            m1_diag_expected_delta = m1_diag_delta * m1_diag_presence.clamp(0.0, 1.0)
            m1_native_probability_r4209 = torch.sigmoid(
                _safe_logit(base).detach() + m1_diag_expected_delta
            ).clamp(EPS, 1.0 - EPS)
            # Secondary hard-posterior diagnostic; it is never the alignment
            # owner and therefore remains detached.
            with torch.no_grad():
                hard_delta = m1_logit_delta.detach().gather(1, m1_diag_winner)
                hard_presence = (
                    atom_presence_probs.detach()[:, :, None, None]
                    .expand_as(m1_logit_delta)
                    .gather(1, m1_diag_winner)
                )
                hard_delta = torch.where(
                    m1_diag_has_support & (hard_presence >= 0.5),
                    hard_delta,
                    torch.zeros_like(hard_delta),
                )
                m1_native_hard_probability_r4209 = torch.sigmoid(
                    _safe_logit(base).detach() + hard_delta
                ).clamp(EPS, 1.0 - EPS)
            m1_native_train_probability_r4210 = m1_native_probability_r4209
            m1_native_alignment_active_r4210 = parent_masks.new_ones(())
        else:
            # Exact historical R4.20.9 audit behavior for C0-C3 and regressions.
            with torch.no_grad():
                m1_diag_score = (
                    atom_presence_probs.detach().clamp(0.0, 1.0)
                    * atom_confidence.detach().clamp_min(0.0)
                )[:, :, None, None].expand_as(m1_logit_delta)
                m1_diag_score = m1_diag_score.masked_fill(
                    ~m1_diag_support_valid, -1.0e4
                )
                m1_diag_has_support = (m1_diag_score > -1.0e3).any(dim=1, keepdim=True)
                m1_diag_winner = m1_diag_score.argmax(dim=1, keepdim=True)
                m1_diag_delta = m1_logit_delta.detach().gather(1, m1_diag_winner)
                m1_diag_presence = (
                    atom_presence_probs.detach()[:, :, None, None]
                    .expand_as(m1_logit_delta)
                    .gather(1, m1_diag_winner)
                )
                m1_diag_expected_delta = torch.where(
                    m1_diag_has_support,
                    m1_diag_delta * m1_diag_presence.clamp(0.0, 1.0),
                    torch.zeros_like(m1_diag_delta),
                )
                m1_diag_hard_delta = torch.where(
                    m1_diag_has_support & (m1_diag_presence >= 0.5),
                    m1_diag_delta,
                    torch.zeros_like(m1_diag_delta),
                )
                m1_native_probability_r4209 = torch.sigmoid(
                    _safe_logit(base).detach() + m1_diag_expected_delta
                ).clamp(EPS, 1.0 - EPS)
                m1_native_hard_probability_r4209 = torch.sigmoid(
                    _safe_logit(base).detach() + m1_diag_hard_delta
                ).clamp(EPS, 1.0 - EPS)
            m1_native_train_probability_r4210 = m1_native_probability_r4209.detach()
            m1_native_alignment_active_r4210 = parent_masks.new_zeros(())

        m1_native_valid_fraction_r4209 = m1_diag_valid.to(parent_masks.dtype).mean()
        r4211_presence_expected_count = atom_presence_probs.sum(dim=1).mean().detach()
        r4211_presence_hard_count = (atom_presence_probs >= 0.5).to(parent_masks.dtype).sum(dim=1).mean().detach()
        if bool(m1_diag_valid.any().item()):
            m1_native_presence_mean_r4209 = atom_presence_probs.detach()[m1_diag_valid].mean()
        else:
            m1_native_presence_mean_r4209 = parent_masks.new_zeros(())

        first_selector = self._candidate_selector(
            slot_feature=atom_feature,
            hard_masks=atom_masks,
            exact_candidate=m1_exact,
            base=base,
            entropy=entropy,
            boundary=boundary,
            presence_probs=atom_presence_probs,
            slot_doses=atom_doses,
            polarity_probs=atom_polarity_probs,
        )
        atom_quality_logits = self.atom_quality_head(
            first_selector["selector_features"]
        ).squeeze(-1)
        atom_quality_probs = torch.sigmoid(atom_quality_logits)
        editor_relative_gain_pred = self.editor_relative_gain_head(
            first_selector["selector_features"]
        ).squeeze(-1)
        # Compatibility diagnostic: unlike R1's confidence*presence heuristic,
        # this score is learned from exact repair purity and used only by deploy.
        atom_quality_score = atom_quality_probs

        # M2 region-wise action correction and bounded local editing.
        # The editor is not part of the live objective before editor_start_epoch,
        # so do not execute its second dense residual/selector pass during warm-up.
        editor_active = self.editor_enabled and (
            (not self.training) or self.current_epoch >= self.editor_start_epoch
        )
        if not self.training:
            editor_strength = 1.0
        elif self.current_epoch < self.editor_start_epoch:
            editor_strength = 0.0
        else:
            editor_strength = min(
                1.0,
                float(self.current_epoch - self.editor_start_epoch + 1)
                / float(self.editor_ramp_epochs),
            )
        # Route context is explicit; M1 action logits are never added to the
        # final Editor logits.  This prevents a moving detached M1 bias from
        # changing the Editor decision boundary.
        editor_selector_feature = (
            first_selector["selector_features"].detach()
            if self.unified_reference_r4_enabled
            else first_selector["selector_features"]
        )
        editor_route_context = torch.cat(
            [
                editor_selector_feature,
                atom_action_probs.detach(),
                atom_doses.detach()[:, :, None],
                atom_presence_probs.detach()[:, :, None],
            ],
            dim=2,
        )
        editor_route_feature = editor_selector_feature + (
            self.editor_route_context_adapter(editor_route_context)
        )

        if editor_active:
            editor_route_logits = self.editor_route_head(editor_route_feature)
            editor_dose_adjust_logits = self.editor_dose_adjust_head(
                editor_route_feature
            ).squeeze(-1)
            editor_dose_adjust = self.editor_dose_adjust_min + (
                self.editor_dose_adjust_max - self.editor_dose_adjust_min
            ) * torch.sigmoid(editor_dose_adjust_logits)
        else:
            editor_route_logits = editor_route_feature.new_full((b, n, 5), -6.0)
            editor_route_logits[:, :, 0] = float(self.editor_preserve_bias)
            editor_dose_adjust_logits = torch.zeros_like(atom_doses)
            editor_dose_adjust = torch.ones_like(atom_doses)

        editor_route_probs = F.softmax(
            editor_route_logits / self.editor_route_temperature, dim=2
        )
        editor_route_index = editor_route_probs.detach().argmax(dim=2)
        editor_route_hard = F.one_hot(editor_route_index, num_classes=5).to(
            editor_route_probs.dtype
        )
        editor_route_st = (
            editor_route_hard + editor_route_probs - editor_route_probs.detach()
        )
        editor_edit_gate = (
            1.0 - editor_route_st[:, :, 0]
        ) * float(editor_strength)
        editor_action_st = editor_route_st[:, :, 1:]
        editor_action_index = (editor_route_index - 1).clamp(0, 3)
        final_doses = (
            atom_doses * editor_dose_adjust
        ).clamp(self.minimum_dose * 0.25, self.maximum_dose)

        editor_dense_feature = (
            fused_feature.detach()
            if self.unified_reference_r4_enabled
            else fused_feature
        )
        parent_local_residual = torch.tanh(
            self.local_residual_head(editor_dense_feature)
        )
        atom_local_residual = _repeat_by_parent(parent_local_residual, parent_index)
        editor_region = self._scale_aware_region(
            (
                atom_masks_st.detach()
                if self.unified_reference_r4_enabled
                else atom_masks_st
            ),
            (
                atom_scale_st.detach()
                if self.unified_reference_r4_enabled
                else atom_scale_st
            ),
            self.editor_region_radii,
        )
        raw_local_residual = (
            atom_local_residual
            * self.editor_local_residual_cap
            * editor_region
        )
        if not editor_active:
            raw_local_residual = raw_local_residual * 0.0

        # ------------------------------------------------------------------
        # V552-R4 shared exact executor.
        # Preserve and all four typed actions are generated by this one route
        # bank.  Training Teachers, Student safety handoff, Shadow and real
        # deployment all consume these exact same deltas.
        # ------------------------------------------------------------------
        if self.unified_reference_r4_enabled:
            action_route_delta = (
                atom_masks_st.detach()[:, :, None]
                * action_delta_bank.detach()[:, None]
                * final_doses[:, :, None, None, None].clamp(0.25, 2.0)
                + m1_boundary_delta.detach()[:, :, None]
                + raw_local_residual[:, :, None]
            )
            if self.editor_total_logit_delta_cap > 0.0:
                action_route_delta = action_route_delta.clamp(
                    -self.editor_total_logit_delta_cap,
                    self.editor_total_logit_delta_cap,
                )
            # The ramp is part of the executor, so Teacher and Student see the
            # same partially activated route during curriculum.
            action_route_delta = (
                m1_logit_delta.detach()[:, :, None]
                + float(editor_strength)
                * (action_route_delta - m1_logit_delta.detach()[:, :, None])
            )
            route_delta_bank = torch.cat(
                [m1_logit_delta.detach()[:, :, None], action_route_delta],
                dim=2,
            )
            route_candidate_bank = torch.sigmoid(
                base_logit[:, :, None] + route_delta_bank
            ).clamp(EPS, 1.0 - EPS)

            editor_logit_delta = (
                editor_route_st[:, :, :, None, None] * route_delta_bank
            ).sum(dim=2)
            editor_exact_st = torch.sigmoid(
                base_logit + editor_logit_delta
            ).clamp(EPS, 1.0 - EPS)
            editor_exact = editor_exact_st.detach()
            local_residual = raw_local_residual * editor_edit_gate[:, :, None, None]

            route_identity = F.one_hot(
                torch.arange(5, device=feature.device), num_classes=5
            ).to(parent_masks.dtype)
            route_identity = route_identity.view(1, 1, 5, 5).expand(b, n, -1, -1)
            relative_delta_bank = (
                route_delta_bank - m1_logit_delta.detach()[:, :, None]
            )
            route_region = editor_region[:, :, None]
            route_area = route_region.mean(dim=(-2, -1)).expand(-1, -1, 5)
            incremental_abs = relative_delta_bank.abs().mean(dim=(-2, -1))
            incremental_signed = relative_delta_bank.mean(dim=(-2, -1))
            absolute_abs = route_delta_bank.abs().mean(dim=(-2, -1))
            absolute_signed = route_delta_bank.mean(dim=(-2, -1))
            route_dose = torch.cat(
                [
                    atom_doses.detach()[:, :, None],
                    final_doses[:, :, None].expand(-1, -1, 4),
                ],
                dim=2,
            )
            route_stats = torch.cat(
                [
                    route_identity,
                    incremental_abs[:, :, :, None],
                    incremental_signed[:, :, :, None],
                    absolute_abs[:, :, :, None],
                    absolute_signed[:, :, :, None],
                    route_area[:, :, :, None],
                    route_dose[:, :, :, None],
                ],
                dim=3,
            )
            base_route_feature = editor_selector_feature[:, :, None].expand(
                -1, -1, 5, -1
            )
            safety_route_base = base_route_feature
            utility_route_base = base_route_feature
            safety_visual = None
            utility_visual = None
            if self.spatial_evidence_r43_enabled:
                safety_visual = self._route_spatial_evidence(
                    fused_feature=fused_feature,
                    route_delta=relative_delta_bank,
                    base_probability=base,
                    entropy=entropy,
                    boundary=boundary,
                    cause_probability=cause_probability,
                )
                utility_visual = self._route_spatial_evidence(
                    fused_feature=fused_feature,
                    route_delta=route_delta_bank,
                    base_probability=base,
                    entropy=entropy,
                    boundary=boundary,
                    cause_probability=cause_probability,
                )
                safety_route_base = (
                    safety_route_base
                    + self.safety_route_visual_adapter(safety_visual)
                )
                utility_route_base = (
                    utility_route_base
                    + self.utility_route_visual_adapter(utility_visual)
                )
            safety_route_feature_input = torch.cat(
                [safety_route_base, route_stats.detach()], dim=3
            )
            utility_route_feature_input = torch.cat(
                [utility_route_base, route_stats.detach()], dim=3
            )
            if self.decoupled_critic_r42_enabled:
                safety_route_feature_bank = (
                    safety_route_base
                    + self.safety_route_feature_adapter(safety_route_feature_input)
                )
                utility_route_feature_bank = (
                    utility_route_base
                    + self.utility_route_feature_adapter(utility_route_feature_input)
                )
                # Compatibility tensors and Composer context use the absolute
                # Utility classification representation.
                route_feature_bank = utility_route_feature_bank
            else:
                route_feature_bank = base_route_feature + self.route_feature_adapter(
                    utility_route_feature_input
                )
                safety_route_feature_bank = route_feature_bank
                utility_route_feature_bank = route_feature_bank

            if self.class_value_decoupling_r44_enabled:
                # Value regressors receive a separately parameterized, detached
                # representation.  Their losses cannot rotate the class feature
                # or the upstream M1 selector representation.
                safety_value_base = base_route_feature.detach()
                utility_value_base = base_route_feature.detach()
                if safety_visual is not None:
                    safety_value_base = (
                        safety_value_base
                        + self.safety_value_visual_adapter(safety_visual.detach())
                    )
                    utility_value_base = (
                        utility_value_base
                        + self.utility_value_visual_adapter(utility_visual.detach())
                    )
                safety_value_input = torch.cat(
                    [safety_value_base, route_stats.detach()], dim=3
                )
                utility_value_input = torch.cat(
                    [utility_value_base, route_stats.detach()], dim=3
                )
                safety_value_feature_bank = (
                    safety_value_base
                    + self.safety_value_feature_adapter(safety_value_input)
                )
                utility_value_feature_bank = (
                    utility_value_base
                    + self.utility_value_feature_adapter(utility_value_input)
                )
            else:
                safety_value_feature_bank = safety_route_feature_bank
                utility_value_feature_bank = utility_route_feature_bank

            editor_safety_logits_bank = self.editor_safety_outcome_head(
                safety_route_feature_bank
            )
            candidate_utility_logits_bank = self.candidate_utility_outcome_head(
                utility_route_feature_bank
            )
            if self.factorized_safety_r45_enabled:
                editor_safety_benefit_logits_bank = self.editor_safety_benefit_head(
                    safety_route_feature_bank
                ).squeeze(-1)
                editor_safety_harm_logits_bank = self.editor_safety_harm_head(
                    safety_route_feature_bank
                ).squeeze(-1)
            else:
                editor_safety_benefit_logits_bank = editor_safety_logits_bank[..., 1]
                editor_safety_harm_logits_bank = editor_safety_logits_bank[..., 2]

            # R4.5 Utility value is a direct signed ΔDice estimate.  Its sign is
            # no longer generated from class probabilities, so a Harm sample
            # cannot become positive merely because the class head is
            # miscalibrated.
            if self.clean_dynamic_component_set_enabled:
                # CLEAN has one M2 quantity only: signed DeltaDice for the exact
                # M1 candidate.  The first candidate-conditioned selector already
                # predicts that quantity directly; route-specific historical critic
                # features are therefore not part of the formal path.
                clean_signed_gain = first_selector["gain_scores"]
                direct_signed_utility_bank = clean_signed_gain[:, :, None].expand(-1, -1, 5)
            elif self.direct_delta_utility_r4212_enabled or self.clean_core_v560_enabled:
                # R4.21.2/V560 predicts the physical quantity consumed by the
                # one-step selector: raw signed DeltaDice.  No tanh cap, no
                # gain-unit normalization and no class-derived sign.
                direct_utility_feature = (
                    utility_value_feature_bank.detach()
                    if self.clean_core_v560_enabled
                    else utility_value_feature_bank
                )
                direct_signed_utility_bank = self.candidate_absolute_gain_head(
                    direct_utility_feature
                ).squeeze(-1)
            else:
                direct_signed_utility_bank = torch.tanh(
                    self.candidate_absolute_gain_head(
                        utility_value_feature_bank
                    ).squeeze(-1)
                ) * self.critic_gain_cap

            if self.decoupled_critic_r42_enabled:
                (
                    editor_incremental_gain_bank,
                    editor_benefit_magnitude_bank,
                    editor_harm_magnitude_bank,
                ) = self._r42_decoupled_expected_gain(
                    outcome_logits=editor_safety_logits_bank,
                    raw_benefit_magnitude=self.editor_benefit_magnitude_head(
                        safety_value_feature_bank
                    ).squeeze(-1),
                    raw_harm_magnitude=self.editor_harm_magnitude_head(
                        safety_value_feature_bank
                    ).squeeze(-1),
                )
                (
                    candidate_absolute_gain_bank,
                    candidate_benefit_magnitude_bank,
                    candidate_harm_magnitude_bank,
                ) = self._r42_decoupled_expected_gain(
                    outcome_logits=candidate_utility_logits_bank,
                    raw_benefit_magnitude=self.candidate_benefit_magnitude_head(
                        utility_value_feature_bank
                    ).squeeze(-1),
                    raw_harm_magnitude=self.candidate_harm_magnitude_head(
                        utility_value_feature_bank
                    ).squeeze(-1),
                )
            else:
                editor_incremental_gain_bank, editor_shared_magnitude = (
                    self._r4_coupled_signed_gain(
                        outcome_logits=editor_safety_logits_bank,
                        raw_magnitude=self.editor_relative_gain_head(
                            safety_route_feature_bank
                        ).squeeze(-1),
                    )
                )
                candidate_absolute_gain_bank, utility_shared_magnitude = (
                    self._r4_coupled_signed_gain(
                        outcome_logits=candidate_utility_logits_bank,
                        raw_magnitude=self.candidate_absolute_gain_head(
                            utility_route_feature_bank
                        ).squeeze(-1),
                    )
                )
                editor_benefit_magnitude_bank = editor_shared_magnitude
                editor_harm_magnitude_bank = editor_shared_magnitude
                candidate_benefit_magnitude_bank = utility_shared_magnitude
                candidate_harm_magnitude_bank = utility_shared_magnitude

            expected_utility_gain_bank = candidate_absolute_gain_bank
            if self.direct_signed_utility_r45_enabled:
                candidate_absolute_gain_bank = direct_signed_utility_bank

            if self.clean_core_v560_enabled:
                # V560 removes the learned Editor/Safety sub-policy from the
                # scientific main path.  The M1 candidate itself (route 0) is
                # the object scored by M2.  Safety is represented by signed
                # physical utility: a harmful candidate simply has DeltaDice<0.
                student_safe_route_index = torch.zeros_like(editor_route_index)
                teacher_safe_route_index = torch.zeros_like(editor_route_index)
                student_safe_delta = m1_logit_delta
                teacher_safe_delta = m1_logit_delta.detach()
                student_safe_feature = route_feature_bank[:, :, 0]
                teacher_safe_feature = route_feature_bank[:, :, 0].detach()
                student_absolute_gain = direct_signed_utility_bank[:, :, 0]
                teacher_absolute_gain_pred = direct_signed_utility_bank[:, :, 0].detach()
                if self.clean_dynamic_component_set_enabled:
                    # Compatibility outcome is a parameter-free sign encoding;
                    # the decision score itself is exactly the signed gain.
                    sign_positive = (student_absolute_gain >= 0.0).to(student_absolute_gain.dtype)
                    sign_negative = 1.0 - sign_positive
                    student_utility_logits = torch.stack(
                        [torch.full_like(student_absolute_gain, -20.0),
                         torch.where(sign_positive > 0.5, torch.zeros_like(student_absolute_gain), torch.full_like(student_absolute_gain, -20.0)),
                         torch.where(sign_negative > 0.5, torch.zeros_like(student_absolute_gain), torch.full_like(student_absolute_gain, -20.0))],
                        dim=-1,
                    )
                    teacher_utility_logits = student_utility_logits.detach()
                else:
                    student_utility_logits = candidate_utility_logits_bank[:, :, 0]
                    teacher_utility_logits = candidate_utility_logits_bank[:, :, 0].detach()
                selector = self._r4_selector_from_feature(
                    student_safe_feature,
                    outcome_logits=student_utility_logits,
                    gain_scores=student_absolute_gain,
                )
                student_safe_action_index = hard_m1_action
                editor_action_index = hard_m1_action
                predicted_editor_safe = torch.ones_like(editor_route_index, dtype=torch.bool)
                editor_relative_gain_pred = student_absolute_gain
            else:
                selected_safety_logits = (
                    editor_route_st[:, :, :, None] * editor_safety_logits_bank
                ).sum(dim=2)
                selected_incremental_gain = (
                    editor_route_st * editor_incremental_gain_bank
                ).sum(dim=2)
                selected_safety_probs = F.softmax(selected_safety_logits, dim=2)
                selected_is_preserve = editor_route_index == 0
                selected_safety_benefit_logit = (
                    editor_route_st * editor_safety_benefit_logits_bank
                ).sum(dim=2)
                selected_safety_harm_logit = (
                    editor_route_st * editor_safety_harm_logits_bank
                ).sum(dim=2)
                selected_safety_benefit_prob = torch.sigmoid(
                    selected_safety_benefit_logit
                )
                selected_safety_harm_prob = torch.sigmoid(
                    selected_safety_harm_logit
                )
                if self.factorized_safety_r45_enabled:
                    predicted_editor_safe = selected_is_preserve | (
                        (selected_safety_benefit_prob >= 0.5)
                        & (selected_safety_harm_prob < 0.5)
                    )
                elif self.semantic_deployment_r44_enabled:
                    selected_class = selected_safety_probs.argmax(dim=2)
                    semantic_editor_benefit = (selected_class == 1) & (
                        selected_safety_probs[:, :, 1]
                        >= selected_safety_probs[:, :, 2]
                        + self.deployment_benefit_harm_margin
                    )
                    predicted_editor_safe = selected_is_preserve | (
                        semantic_editor_benefit
                        & (selected_incremental_gain > self.editor_incremental_gain_threshold)
                    )
                else:
                    predicted_editor_safe = selected_is_preserve | (
                        (selected_safety_probs[:, :, 1] >= self.editor_safety_benefit_threshold)
                        & (selected_safety_probs[:, :, 2] <= self.editor_safety_harm_threshold)
                        & (
                            selected_incremental_gain
                            > self.editor_incremental_gain_threshold
                        )
                    )
                student_safe_route_index = torch.where(
                    predicted_editor_safe,
                    editor_route_index,
                    torch.zeros_like(editor_route_index),
                )
                student_safe_one_hot = F.one_hot(
                    student_safe_route_index, num_classes=5
                ).to(parent_masks.dtype)
                student_safe_delta = (
                    student_safe_one_hot[:, :, :, None, None] * route_delta_bank
                ).sum(dim=2)
                student_safe_feature = (
                    student_safe_one_hot[:, :, :, None] * route_feature_bank
                ).sum(dim=2)
                student_utility_logits = (
                    student_safe_one_hot[:, :, :, None]
                    * candidate_utility_logits_bank
                ).sum(dim=2)
                student_absolute_gain = (
                    student_safe_one_hot * candidate_absolute_gain_bank
                ).sum(dim=2)
                selector = self._r4_selector_from_feature(
                    student_safe_feature,
                    outcome_logits=student_utility_logits,
                    gain_scores=student_absolute_gain,
                )
                editor_relative_gain_pred = selected_incremental_gain
            student_safe_action_index = torch.where(
                student_safe_route_index > 0,
                student_safe_route_index - 1,
                hard_m1_action,
            )
            editor_action_index = student_safe_action_index

            teacher_safe_route_index = torch.zeros_like(editor_route_index)
            teacher_safe_delta = m1_logit_delta.detach()
            teacher_safe_feature = route_feature_bank[:, :, 0]
            teacher_utility_logits = candidate_utility_logits_bank[:, :, 0]
            teacher_absolute_gain_pred = candidate_absolute_gain_bank[:, :, 0]
            exact_incremental_gain_bank = route_delta_bank.new_zeros((b, n, 5))
            exact_absolute_gain_bank = route_delta_bank.new_zeros((b, n, 5))
            safety_target_bank = torch.zeros(
                (b, n, 5), dtype=torch.long, device=feature.device
            )
            utility_target_bank = torch.zeros_like(safety_target_bank)

            if (
                self.training
                and isinstance(supervision_masks, torch.Tensor)
            ):
                gt_r4 = supervision_masks.detach()
                if gt_r4.ndim == 3:
                    gt_r4 = gt_r4[:, None]
                gt_r4 = (gt_r4 >= 0.5).to(base.dtype)
                if gt_r4.shape[-2:] != (h, w):
                    gt_r4 = F.interpolate(gt_r4, size=(h, w), mode="nearest")
                route_dice = _hard_dice_route_bank(route_candidate_bank, gt_r4)
                base_dice_r4 = _hard_dice_route_bank(
                    base[:, 0][:, None, None], gt_r4
                )[:, 0, 0]
                exact_incremental_gain_bank = (
                    route_dice - route_dice[:, :, :1]
                )
                exact_absolute_gain_bank = (
                    route_dice - base_dice_r4[:, None, None]
                )
                best_action_relative_gain, best_action_offset = (
                    exact_incremental_gain_bank[:, :, 1:].max(dim=2)
                )
                teacher_safe_route_index = torch.where(
                    best_action_relative_gain > self.editor_relative_margin,
                    best_action_offset + 1,
                    torch.zeros_like(best_action_offset),
                )
                teacher_safe_one_hot = F.one_hot(
                    teacher_safe_route_index, num_classes=5
                ).to(parent_masks.dtype)
                teacher_safe_delta = (
                    teacher_safe_one_hot[:, :, :, None, None]
                    * route_delta_bank.detach()
                ).sum(dim=2)
                teacher_safe_feature = (
                    teacher_safe_one_hot[:, :, :, None]
                    * route_feature_bank
                ).sum(dim=2)
                teacher_utility_logits = (
                    teacher_safe_one_hot[:, :, :, None]
                    * candidate_utility_logits_bank
                ).sum(dim=2)
                teacher_absolute_gain_pred = (
                    teacher_safe_one_hot * candidate_absolute_gain_bank
                ).sum(dim=2)
                relative_margin = max(self.editor_relative_margin, EPS)
                absolute_margin = max(
                    self.editor_incremental_gain_threshold,
                    1.0e-4,
                )
                safety_target_bank = torch.where(
                    exact_incremental_gain_bank > relative_margin,
                    torch.ones_like(safety_target_bank),
                    safety_target_bank,
                )
                safety_target_bank = torch.where(
                    exact_incremental_gain_bank < -relative_margin,
                    torch.full_like(safety_target_bank, 2),
                    safety_target_bank,
                )
                utility_target_bank = torch.where(
                    exact_absolute_gain_bank > absolute_margin,
                    torch.ones_like(utility_target_bank),
                    utility_target_bank,
                )
                utility_target_bank = torch.where(
                    exact_absolute_gain_bank < -absolute_margin,
                    torch.full_like(utility_target_bank, 2),
                    utility_target_bank,
                )
        else:
            # Historical R2/R3 execution is preserved for old configurations.
            editor_action_delta = torch.einsum(
                "bna,bahw->bnhw", editor_action_st, action_delta_bank
            )
            modified_logit_delta = (
                atom_masks_st
                * editor_action_delta
                * final_doses[:, :, None, None].clamp(0.25, 2.0)
                + m1_boundary_delta
                + raw_local_residual
            )
            editor_logit_delta = (
                m1_logit_delta
                + editor_edit_gate[:, :, None, None]
                * (modified_logit_delta - m1_logit_delta)
            )
            local_residual = raw_local_residual * editor_edit_gate[:, :, None, None]
            if self.editor_total_logit_delta_cap > 0.0:
                editor_logit_delta = editor_logit_delta.clamp(
                    -self.editor_total_logit_delta_cap,
                    self.editor_total_logit_delta_cap,
                )
            editor_exact_st = torch.sigmoid(base_logit + editor_logit_delta).clamp(
                EPS, 1.0 - EPS
            )
            editor_exact = editor_exact_st.detach()
            route_delta_bank = torch.cat(
                [
                    m1_logit_delta[:, :, None],
                    editor_logit_delta[:, :, None].expand(-1, -1, 4, -1, -1),
                ],
                dim=2,
            )
            route_candidate_bank = torch.sigmoid(
                base_logit[:, :, None] + route_delta_bank
            ).clamp(EPS, 1.0 - EPS)
            route_feature_bank = first_selector["selector_features"][:, :, None].expand(
                -1, -1, 5, -1
            )
            safety_route_feature_bank = route_feature_bank
            utility_route_feature_bank = route_feature_bank
            safety_value_feature_bank = route_feature_bank
            utility_value_feature_bank = route_feature_bank
            editor_safety_logits_bank = route_delta_bank.new_zeros((b, n, 5, 3))
            editor_safety_benefit_logits_bank = route_delta_bank.new_zeros((b, n, 5))
            editor_safety_harm_logits_bank = route_delta_bank.new_zeros((b, n, 5))
            editor_incremental_gain_bank = route_delta_bank.new_zeros((b, n, 5))
            editor_benefit_magnitude_bank = route_delta_bank.new_zeros((b, n, 5))
            editor_harm_magnitude_bank = route_delta_bank.new_zeros((b, n, 5))
            candidate_utility_logits_bank = route_delta_bank.new_zeros((b, n, 5, 3))
            candidate_absolute_gain_bank = route_delta_bank.new_zeros((b, n, 5))
            expected_utility_gain_bank = candidate_absolute_gain_bank
            direct_signed_utility_bank = candidate_absolute_gain_bank
            candidate_benefit_magnitude_bank = route_delta_bank.new_zeros((b, n, 5))
            candidate_harm_magnitude_bank = route_delta_bank.new_zeros((b, n, 5))
            teacher_safe_route_index = torch.zeros_like(editor_route_index)
            teacher_safe_delta = editor_logit_delta.detach()
            teacher_safe_feature = first_selector["selector_features"]
            teacher_utility_logits = route_delta_bank.new_zeros((b, n, 3))
            teacher_absolute_gain_pred = route_delta_bank.new_zeros((b, n))
            exact_incremental_gain_bank = route_delta_bank.new_zeros((b, n, 5))
            exact_absolute_gain_bank = route_delta_bank.new_zeros((b, n, 5))
            safety_target_bank = torch.zeros(
                (b, n, 5), dtype=torch.long, device=feature.device
            )
            utility_target_bank = torch.zeros_like(safety_target_bank)
            student_safe_route_index = editor_route_index
            student_safe_delta = editor_logit_delta
            student_safe_feature = first_selector["selector_features"]
            student_safe_action_index = editor_action_index
            predicted_editor_safe = editor_route_index == 0
            editor_relative_gain_pred = self.editor_relative_gain_head(
                first_selector["selector_features"]
            ).squeeze(-1)
            if self.single_pass_editor:
                region_area = editor_region.mean(dim=(-2, -1))
                residual_abs = local_residual.abs().mean(dim=(-2, -1))
                residual_signed = local_residual.mean(dim=(-2, -1))
                edit_delta_abs = editor_logit_delta.abs().mean(dim=(-2, -1))
                route_stats = torch.cat(
                    [
                        editor_route_probs,
                        editor_dose_adjust[:, :, None],
                        region_area[:, :, None],
                        residual_abs[:, :, None],
                        residual_signed[:, :, None],
                        edit_delta_abs[:, :, None],
                    ],
                    dim=2,
                )
                refined_feature = first_selector["selector_features"] + (
                    self.editor_feature_adapter(
                        torch.cat(
                            [first_selector["selector_features"], route_stats],
                            dim=2,
                        )
                    )
                )
                selector = self._selector_from_feature(refined_feature)
            else:
                selector = self._candidate_selector(
                    slot_feature=atom_feature,
                    hard_masks=atom_masks,
                    exact_candidate=editor_exact,
                    base=base,
                    entropy=entropy,
                    boundary=boundary,
                    presence_probs=atom_presence_probs,
                    slot_doses=final_doses,
                    polarity_probs=atom_polarity_probs,
                )

        remove_probability = (
            editor_route_probs[:, :, 1] + editor_route_probs[:, :, 3]
        ) * editor_edit_gate
        add_probability = (
            editor_route_probs[:, :, 2] + editor_route_probs[:, :, 4]
        ) * editor_edit_gate
        editor_polarity = torch.stack(
            [remove_probability, add_probability], dim=2
        )
        no_polarity = editor_polarity.sum(dim=2, keepdim=True) <= EPS
        editor_polarity = editor_polarity / editor_polarity.sum(
            dim=2, keepdim=True
        ).clamp_min(EPS)
        editor_polarity = torch.where(
            no_polarity.expand_as(editor_polarity),
            editor_polarity.new_full(editor_polarity.shape, 0.5),
            editor_polarity,
        )

        atom_area = atom_masks.to(parent_masks.dtype).mean(dim=(-2, -1))
        physical_valid = (
            atom_valid
            & atom_contrast
            & (atom_area >= self.min_area_fraction)
            & (atom_area <= self.max_component_area_fraction)
            & torch.isfinite(editor_logit_delta).flatten(2).all(dim=2)
        )
        m1_supervision_valid = physical_valid
        route_valid = physical_valid[:, :, None].expand(-1, -1, 5)

        # Replay predictions are formed before the current batch is enqueued.
        # This prevents a scarce example from being duplicated immediately as
        # both a live sample and a replay sample in the same optimizer step.
        (
            safety_queue_logits,
            safety_queue_gain_pred,
            safety_queue_target,
            safety_queue_gain_value,
            safety_queue_benefit_magnitude,
            safety_queue_harm_magnitude,
        ) = self._r4_queue_predictions("safety")
        (
            utility_queue_logits,
            utility_queue_gain_pred,
            utility_queue_target,
            utility_queue_gain_value,
            utility_queue_benefit_magnitude,
            utility_queue_harm_magnitude,
        ) = self._r4_queue_predictions("utility")

        # Class-complete replay is updated only from Train GT and stores detached
        # task-specific route features. Validation/Test remain entirely GT-free.
        if (
            self.unified_reference_r4_enabled
            and self.training
            and isinstance(supervision_masks, torch.Tensor)
        ):
            self._enqueue_r4_critic(
                task="safety",
                features=safety_route_feature_bank,
                gains=exact_incremental_gain_bank,
                labels=safety_target_bank,
                valid=route_valid,
            )
            self._enqueue_r4_critic(
                task="utility",
                features=utility_route_feature_bank,
                gains=exact_absolute_gain_bank,
                labels=utility_target_bank,
                valid=route_valid,
            )

        if self.unified_reference_r4_enabled:
            student_delta_for_compose = student_safe_delta.detach()
            student_selector = selector
            student_action_index = student_safe_action_index
            if self.native_contract_r46_enabled:
                # Exact same student-safe candidate set for teacher labels and
                # deployment. GT may score these candidates, never replace them.
                training_delta_for_compose = student_delta_for_compose
                training_selector = student_selector
                training_action_index = student_action_index
            else:
                training_delta_for_compose = teacher_safe_delta.detach()
                training_selector = self._r4_selector_from_feature(
                    teacher_safe_feature,
                    outcome_logits=teacher_utility_logits,
                    gain_scores=teacher_absolute_gain_pred,
                )
                training_action_index = torch.where(
                    teacher_safe_route_index > 0,
                    teacher_safe_route_index - 1,
                    hard_m1_action,
                )
        else:
            training_delta_for_compose = editor_logit_delta.detach()
            student_delta_for_compose = editor_logit_delta.detach()
            training_selector = selector
            student_selector = selector
            training_action_index = editor_action_index
            student_action_index = editor_action_index

        if self.clean_core_v560_enabled:
            # One immutable M1->M2 interface: both Teacher scoring and Student
            # deployment see exactly the same current M1 candidates.
            training_delta_for_compose = m1_logit_delta.detach()
            student_delta_for_compose = m1_logit_delta.detach()
            training_selector = selector
            student_selector = selector
            training_action_index = hard_m1_action
            student_action_index = hard_m1_action

        training_support = atom_masks | (training_delta_for_compose.abs() > EPS)
        composer_support = atom_masks | (student_delta_for_compose.abs() > EPS)
        composer_teacher_valid = (
            physical_valid & training_support.flatten(2).any(dim=2)
        )

        # Deployment uses only Student-safe candidates. Top-k is predicted from
        # inference-visible presence, quality and absolute utility evidence.
        deployment_valid = (
            physical_valid & composer_support.flatten(2).any(dim=2)
        )
        if self.existence_no_object_r4212_enabled or self.clean_core_v560_enabled:
            # Presence is literal instance existence/no-object, not utility.
            # Logit zero is the canonical Bernoulli boundary and introduces no
            # tuned deployment threshold.  Absent slots never enter M2.
            deployment_valid = deployment_valid & (atom_presence_logits.detach() > 0.0)
        if self.clean_dynamic_component_set_enabled:
            # Presence owns existence; signed gain owns utility.  No historical
            # quality/editability probability product is allowed to pre-filter
            # CLEAN candidates.
            deploy_pool_score = student_selector["gain_scores"].detach()
        else:
            deploy_pool_score = (
                atom_presence_probs.clamp(0.0, 1.0)
                if self.native_contract_r46_enabled
                else (
                    atom_presence_probs.clamp(0.0, 1.0)
                    * atom_quality_probs.clamp(0.0, 1.0)
                    * student_selector["editability_probs"].clamp(0.0, 1.0)
                )
            )
        if atom_quality_gate_active and (not self.native_contract_r46_enabled) and (not self.clean_dynamic_component_set_enabled):
            deployment_valid = deployment_valid & (
                atom_presence_probs >= self.atom_presence_threshold
            ) & (atom_quality_probs >= self.atom_quality_threshold)
        pool_k = min(self.composer_deploy_pool_size, n)
        if pool_k < n:
            pool_score = deploy_pool_score.masked_fill(~deployment_valid, -1.0e4)
            topk_index = pool_score.topk(pool_k, dim=1).indices
            topk_mask = torch.zeros_like(deployment_valid)
            topk_mask.scatter_(1, topk_index, True)
            deployment_valid = deployment_valid & topk_mask

        if self.decoupled_critic_r42_enabled:
            # Composer learns its own state-conditioned correction and Stop
            # decision, but it must not redefine the critic's Outcome boundary
            # or M1 presence semantics.  Detaching all upstream evidence makes
            # the task ownership explicit while keeping one end-to-end forward
            # graph and one optimizer step.
            training_scores = training_selector["decision_scores"].detach()
            training_presence = atom_presence_probs.detach()
            training_benefit = training_selector["benefit_probs"].detach()
            training_harm = training_selector["harm_probs"].detach()
            training_gain_lcb = training_selector["gain_lcb"].detach()
            training_editability = training_selector["editability_probs"].detach()
            training_direction = training_selector["direction_probs"].detach()
            training_selector_features = training_selector[
                "selector_features"
            ].detach()
        else:
            training_scores = training_selector["decision_scores"]
            training_presence = atom_presence_probs
            training_benefit = training_selector["benefit_probs"]
            training_harm = training_selector["harm_probs"]
            training_gain_lcb = training_selector["gain_lcb"]
            training_editability = training_selector["editability_probs"]
            training_direction = training_selector["direction_probs"]
            training_selector_features = training_selector["selector_features"]

        training_compose_kwargs = dict(
            base_probability=base_probability,
            atom_masks=training_support,
            exact_logit_delta=training_delta_for_compose,
            action_index=training_action_index,
            scores=training_scores,
            presence=training_presence,
            benefit=training_benefit,
            harm=training_harm,
            gain_lcb=training_gain_lcb,
            editability=training_editability,
            direction=training_direction,
            selector_features=training_selector_features,
        )
        student_compose_kwargs = dict(
            base_probability=base_probability,
            atom_masks=composer_support,
            exact_logit_delta=student_delta_for_compose,
            action_index=student_action_index,
            scores=student_selector["decision_scores"],
            presence=atom_presence_probs,
            benefit=student_selector["benefit_probs"],
            harm=student_selector["harm_probs"],
            gain_lcb=student_selector["gain_lcb"],
            editability=student_selector["editability_probs"],
            direction=student_selector["direction_probs"],
            selector_features=student_selector["selector_features"],
        )

        # Strict teacher forcing uses the M1-safe handoff. If an Editor route is
        # not exactly better than M1, the exact teacher receives M1 unchanged.
        forced_teacher = None
        teacher_trace_valid = (
            deployment_valid if self.native_contract_r46_enabled else composer_teacher_valid
        )
        if (
            self.training
            and self.teacher_decoupled_r2_enabled
            and isinstance(supervision_masks, torch.Tensor)
        ):
            gt_teacher = supervision_masks.detach()
            if gt_teacher.ndim == 3:
                gt_teacher = gt_teacher[:, None]
            gt_teacher = (gt_teacher >= 0.5).to(base_probability.dtype)
            if gt_teacher.shape[-2:] != (h, w):
                gt_teacher = F.interpolate(gt_teacher, size=(h, w), mode="nearest")
            base_hard = base_probability.detach() >= 0.5
            error = (base_hard != (gt_teacher >= 0.5))[:, 0]
            correct_mass = (
                training_support & error[:, None]
            ).flatten(2).sum(dim=2).to(base_probability.dtype)
            support_area = training_support.flatten(2).sum(dim=2).clamp_min(1).to(
                base_probability.dtype
            )
            exact_quality = correct_mass / support_area
            teacher_pool_size = min(max(int(self.composer_teacher_pool_size), 1), n)
            if teacher_pool_size < n and (not self.native_contract_r46_enabled):
                exact_pool_score = exact_quality + 0.25 * (
                    correct_mass > 0
                ).to(exact_quality.dtype)
                exact_pool_score = exact_pool_score.masked_fill(
                    ~teacher_trace_valid, -1.0e4
                )
                topk_index = exact_pool_score.topk(teacher_pool_size, dim=1).indices
                topk_mask = torch.zeros_like(teacher_trace_valid)
                topk_mask.scatter_(1, topk_index, True)
                teacher_trace_valid = teacher_trace_valid & topk_mask
            forced_teacher = build_v552r2_exact_composer_teacher(
                base=base_probability,
                editor_logit_delta=training_delta_for_compose,
                masks=training_support,
                valid=teacher_trace_valid,
                gt=gt_teacher,
                max_steps=self.max_steps,
                stop_margin=self.composer_teacher_stop_margin,
                max_overlap=self.max_overlap,
                max_total_edit_fraction=self.max_total_edit_fraction,
            )

        training_trace = self._compose_exact_deltas(
            **training_compose_kwargs,
            valid=teacher_trace_valid,
            deploy_enabled=True,
            gain_sign_only=False,
            apply_safety_gates=False,
            force_all_steps_active=(forced_teacher is None),
            forced_choices=(
                forced_teacher["target_indices"]
                if forced_teacher is not None else None
            ),
        )
        ungated_student = self._compose_exact_deltas(
            **student_compose_kwargs,
            valid=physical_valid & composer_support.flatten(2).any(dim=2),
            deploy_enabled=True,
            gain_sign_only=False,
            apply_safety_gates=False,
            force_all_steps_active=False,
        )
        shadow_enabled = (
            (not self.training) or self.current_epoch >= self.shadow_evidence_start_epoch
        )
        # R4.4 audit trace: force exactly the highest-scoring physically valid
        # candidate as a detached, training-only intervention.  This breaks the
        # circular dependency in which the quality gate required executions
        # while the shadow path itself required the unopened quality gate.
        audit_enabled = (
            self.audit_gate_r44_enabled
            and self.training
            and self.current_epoch >= self.audit_shadow_start_epoch
        )
        audit_valid = physical_valid & composer_support.flatten(2).any(dim=2)
        audit_score = student_selector["decision_scores"].detach().masked_fill(
            ~audit_valid, -1.0e4
        )
        audit_top_index = audit_score.argmax(dim=1)
        audit_has_candidate = audit_valid.any(dim=1)
        audit_choices = torch.full(
            (b, min(max(int(self.max_steps), 1), n)),
            n,
            dtype=torch.long,
            device=feature.device,
        )
        audit_choices[:, 0] = torch.where(
            audit_has_candidate,
            audit_top_index,
            audit_top_index.new_full(audit_top_index.shape, n),
        )
        audit_shadow = self._compose_exact_deltas(
            **student_compose_kwargs,
            valid=audit_valid,
            deploy_enabled=audit_enabled,
            gain_sign_only=False,
            apply_safety_gates=False,
            force_all_steps_active=False,
            forced_choices=audit_choices,
        )
        shadow = self._compose_exact_deltas(
            **student_compose_kwargs,
            valid=deployment_valid,
            deploy_enabled=shadow_enabled,
            gain_sign_only=self.gain_sign_shadow_deploy_enabled,
            apply_safety_gates=True,
        )
        deployment = self._compose_exact_deltas(
            **student_compose_kwargs,
            valid=deployment_valid,
            deploy_enabled=deploy_enabled,
            gain_sign_only=False,
            apply_safety_gates=True,
        )
        slot_valid = m1_supervision_valid
        deploy_slot_valid = deployment_valid

        hard_masks_float = atom_masks.to(parent_masks.dtype)
        action_max_scores = selector["decision_scores"].new_full((b, 4), -20.0)
        for action in range(4):
            action_valid = slot_valid & (editor_action_index == action)
            action_max_scores[:, action] = torch.where(
                action_valid.any(dim=1),
                selector["decision_scores"].masked_fill(
                    ~action_valid, -1.0e4
                ).max(dim=1).values,
                action_max_scores[:, action],
            )

        signed_dose = (
            editor_polarity[:, :, 1] - editor_polarity[:, :, 0]
        ) * final_doses
        student_safe_probability = torch.sigmoid(
            base_logit + student_safe_delta
        ).clamp(EPS, 1.0 - EPS)
        teacher_safe_probability = torch.sigmoid(
            base_logit + teacher_safe_delta
        ).clamp(EPS, 1.0 - EPS)
        return {
            # V538/V550 compatibility tensors now operate on atoms.
            "v551_fast_parent_atoms_active": parent_masks.new_tensor(
                1.0 if use_fast_parent_atoms else 0.0
            ),
            "v551_editor_active": parent_masks.new_tensor(
                1.0 if editor_active else 0.0
            ),
            "slot_mask_logits": atom_mask_logits,
            "slot_masks": atom_soft_masks.clamp(EPS, 1.0 - EPS),
            "slot_raw_masks": atom_soft_masks,
            "v552r4211_effective_slot_masks": atom_effective_masks_st_r4211.clamp(EPS, 1.0 - EPS),
            "slot_competition_enabled": parent_masks.new_tensor(
                1.0 if self.slot_competition_enabled else 0.0
            ),
            "gain_sign_shadow_deploy_enabled": parent_masks.new_tensor(
                1.0 if self.gain_sign_shadow_deploy_enabled else 0.0
            ),
            "slot_raw_overlap_mass": F.relu(atom_soft_masks.sum(dim=1) - 1.0).mean(),
            "slot_competition_overlap_mass": F.relu(
                hard_masks_float.sum(dim=1) - 1.0
            ).mean(),
            "slot_hard_masks": hard_masks_float,
            "slot_hard_masks_st": atom_masks_st,
            "slot_mask_contrast_active": atom_contrast.to(parent_masks.dtype),
            "adaptive_cardinality_hard_mask_enabled": parent_masks.new_tensor(
                1.0 if self.adaptive_cardinality_hard_mask else 0.0
            ),
            "slot_action_logits": atom_action_logits,
            "slot_action_probs": atom_action_probs,
            "slot_polarity_logits": atom_polarity_logits,
            "slot_polarity_probs": atom_polarity_probs,
            "slot_dose_logits": atom_dose_logits,
            "slot_raw_doses": atom_raw_doses,
            "slot_dose_gate": atom_dose_gate,
            "slot_doses": atom_doses,
            "slot_signed_dose": signed_dose,
            "slot_gain_scores": selector["gain_scores"],
            "slot_gain_normalized": selector["normalized_gain"],
            "slot_gain_magnitude_normalized": selector[
                "gain_magnitude_normalized"
            ],
            "slot_signed_outcome": selector["signed_outcome"],
            "slot_outcome_logits": selector["outcome_logits"],
            "slot_outcome_probs": selector["outcome_probs"],
            "slot_neutral_probs": selector["neutral_probs"],
            "slot_gain_logvar": selector["gain_logvar"],
            "slot_gain_std": selector["gain_std"],
            "slot_gain_lcb": selector["gain_lcb"],
            "slot_rank_scores": selector["rank_scores"],
            "slot_decision_scores": selector["decision_scores"],
            "slot_benefit_logits": selector["benefit_logits"],
            "slot_benefit_probs": selector["benefit_probs"],
            "slot_harm_logits": selector["harm_logits"],
            "slot_harm_probs": selector["harm_probs"],
            "slot_editability_logits": selector["editability_logits"],
            "slot_editability_probs": selector["editability_probs"],
            "slot_direction_logits": selector["direction_logits"],
            "slot_direction_probs": selector["direction_probs"],
            "slot_selector_features": selector["selector_features"],
            "direction_head_weight": selector["direction_head_weight"],
            "direction_head_bias": selector["direction_head_bias"],
            "editability_head_weight": selector["editability_head_weight"],
            "editability_head_bias": selector["editability_head_bias"],
            "gain_head_weight": selector["gain_head_weight"],
            "gain_head_bias": selector["gain_head_bias"],
            "factorized_deployment_enabled": selector[
                "factorized_deployment_enabled"
            ],
            "factorized_outcome_enabled": selector["factorized_outcome_enabled"],
            "factorized_direction_zero_init_enabled": selector[
                "factorized_direction_zero_init_enabled"
            ],
            "slot_presence_logits": atom_presence_logits,
            "slot_presence_probs": atom_presence_probs,
            "slot_candidate_probs": m1_exact_st,
            # R4 labels/deployment use the M1-safe handoff, never a raw
            # Editor candidate that failed the relative-to-M1 safety critic.
            "slot_exact_candidate_probs": student_safe_probability.detach(),
            # M1 exact-alignment loss uses the pre-M2 intervention.
            "slot_exact_candidate_st_probs": m1_exact_st,
            "slot_discrete_candidate_probs": m1_exact_st,
            "slot_continuous_candidate_probs": m1_exact_st,
            "slot_valid": slot_valid,
            "slot_area_fraction": atom_area,
            "selected_action_weight": deployment["selected_action_weight"],
            "selected_polarity_weight": deployment["selected_polarity_weight"],
            "selected_signed_delta": deployment["selected_signed_delta"],
            "selected_final_probability": deployment["selected_final_probability"],
            "accepted_slots": deployment["accepted_slots"],
            "selected_index": deployment["selected_index"],
            "selected_score": deployment["selected_score"],
            "predicted_execute": deployment["predicted_execute"],
            "accepted_count": deployment["accepted_count"],
            "changed_fraction": deployment["changed_fraction"],
            "composition_score": deployment["composition_score"],
            "shadow_selected_final_probability": shadow["selected_final_probability"],
            "shadow_accepted_slots": shadow["accepted_slots"],
            "shadow_selected_index": shadow["selected_index"],
            "shadow_selected_score": shadow["selected_score"],
            "shadow_predicted_execute": shadow["predicted_execute"],
            "shadow_accepted_count": shadow["accepted_count"],
            "shadow_changed_fraction": shadow["changed_fraction"],
            "shadow_composition_score": shadow["composition_score"],
            "audit_selected_final_probability": audit_shadow["selected_final_probability"],
            "audit_accepted_slots": audit_shadow["accepted_slots"],
            "audit_selected_index": audit_shadow["selected_index"],
            "audit_selected_score": audit_shadow["selected_score"],
            "audit_predicted_execute": audit_shadow["predicted_execute"],
            "audit_accepted_count": audit_shadow["accepted_count"],
            "audit_changed_fraction": audit_shadow["changed_fraction"],
            "audit_composition_score": audit_shadow["composition_score"],
            "v552_multicandidate_composer_enabled": parent_masks.new_tensor(
                1.0 if self.multicandidate_composer_enabled else 0.0
            ),
            "v552_composer_support": composer_support.to(parent_masks.dtype),
            "v552r4_teacher_composer_support": training_support.to(parent_masks.dtype),
            "v552r4_m1_gradient_firewall": parent_masks.new_tensor(
                1.0 if self.unified_reference_r4_enabled else 0.0
            ),
            "v552_composer_step_logits": training_trace["composer_step_logits"],
            "v552_composer_step_candidate_scores": training_trace[
                "composer_step_candidate_scores"
            ],
            "v552_composer_step_state_probs": training_trace[
                "composer_step_state_probs"
            ],
            "v552_composer_step_eligible": training_trace["composer_step_eligible"],
            "v552_composer_step_active": training_trace["composer_step_active"],
            "v552_composer_step_selected_index": training_trace[
                "composer_step_selected_index"
            ],
            # Keep safety Shadow and ungated Student evidence separate.
            "v552_composer_predicted_step_count": shadow["accepted_count"],
            "v552_composer_ungated_step_count": ungated_student["accepted_count"],
            "v552_composer_ungated_final_probability": ungated_student[
                "selected_final_probability"
            ],
            "v552_composer_ungated_selected_index": ungated_student[
                "composer_step_selected_index"
            ],
            "v552_composer_teacher_forced_active": parent_masks.new_tensor(
                1.0 if forced_teacher is not None else 0.0
            ),
            "v552_composer_forced_target_indices": (
                forced_teacher["target_indices"]
                if forced_teacher is not None
                else training_trace["composer_step_selected_index"].detach()
            ),
            # R4.1 single-source teacher contract.  Loss must consume these
            # exact tensors and is forbidden from rebuilding another teacher.
            "v552r41_teacher_target_indices": (
                forced_teacher["target_indices"]
                if forced_teacher is not None
                else training_trace["composer_step_selected_index"].detach()
            ),
            "v552r41_teacher_target_marginal_gains": (
                forced_teacher["target_marginal_gains"]
                if forced_teacher is not None
                else training_trace["composer_step_candidate_scores"].detach()
            ),
            "v552r41_teacher_target_eligible": (
                forced_teacher["target_eligible"]
                if forced_teacher is not None
                else training_trace["composer_step_eligible"].detach()
            ),
            "v552r41_teacher_target_active": (
                forced_teacher["target_active"]
                if forced_teacher is not None
                else training_trace["composer_step_active"].detach()
            ),
            "v552r41_teacher_target_step_gains": (
                forced_teacher["target_step_gains"]
                if forced_teacher is not None
                else training_trace["composer_step_candidate_scores"].detach().new_zeros(
                    training_trace["composer_step_candidate_scores"].shape[:2]
                )
            ),
            "v552r41_teacher_state_probs": (
                forced_teacher["teacher_state_probs"]
                if forced_teacher is not None
                else training_trace["composer_step_state_probs"].detach()
            ),
            "v552r41_teacher_final_probs": (
                forced_teacher["teacher_final_probs"]
                if forced_teacher is not None
                else training_trace["selected_final_probability"].detach()
            ),
            "v552r41_single_teacher_contract_enabled": parent_masks.new_tensor(
                1.0 if forced_teacher is not None else 0.0
            ),
            "v552_composer_conflict_reject_rate": training_trace[
                "composer_conflict_reject_rate"
            ],
            "v552_composer_budget_reject_rate": training_trace[
                "composer_budget_reject_rate"
            ],
            "action_max_scores": action_max_scores,
            "candidate_masks": hard_masks_float,
            "candidate_actions": editor_action_index.detach(),
            "candidate_valid": slot_valid.detach(),
            "candidate_scores": selector["decision_scores"],
            "retained_component_count": slot_valid.float().sum(dim=1),
            "raw_component_count": selector["decision_scores"].new_full(
                (b,), float(n)
            ),
            # Native V551 tensors.
            "v551_enabled": parent_masks.new_ones(()),
            "v551_gpu_atomizer_enabled": parent_masks.new_tensor(1.0 if self.gpu_atomizer_enabled else 0.0),
            "v551_single_pass_editor_enabled": parent_masks.new_tensor(1.0 if self.single_pass_editor else 0.0),
            "v551_parent_mask_logits": parent_mask_logits,
            "v551_parent_masks": parent_masks,
            "v551_parent_hard_masks": parent_hard.to(parent_masks.dtype),
            "v551_atom_parent_index": parent_index,
            "v551_atom_valid": atom_valid,
            "v552_physical_valid": physical_valid,
            "v552_m1_supervision_valid": m1_supervision_valid,
            "v552_composer_teacher_valid": composer_teacher_valid,
            "v552_deployment_valid": deployment_valid,
            "v552_deploy_slot_valid": deploy_slot_valid,
            "v552_atom_quality_logits": atom_quality_logits,
            "v552_atom_quality_probs": atom_quality_probs,
            "v552_editor_relative_gain_pred": editor_relative_gain_pred,
            "v552r2_unified_outcome_enabled": parent_masks.new_tensor(
                1.0 if self.teacher_decoupled_r2_enabled else 0.0
            ),
            "v552r2_benefit_magnitude": selector.get(
                "benefit_magnitude_normalized",
                selector["gain_magnitude_normalized"],
            ),
            "v552r2_harm_magnitude": selector.get(
                "harm_magnitude_normalized",
                selector["gain_magnitude_normalized"],
            ),
            "v552r2_benefit_contribution": selector.get(
                "benefit_contribution",
                selector["benefit_probs"] * selector["gain_magnitude_normalized"],
            ),
            "v552r2_harm_contribution": selector.get(
                "harm_contribution",
                -selector["harm_probs"] * selector["gain_magnitude_normalized"],
            ),
            "v552_composer_teacher_pool_count": composer_teacher_valid.to(parent_masks.dtype).sum(dim=1).mean(),
            "v552_composer_deploy_pool_count": deployment_valid.to(parent_masks.dtype).sum(dim=1).mean(),
            "v552_shadow_evidence_enabled": parent_masks.new_tensor(1.0 if shadow_enabled else 0.0),
            "v551_atom_confidence": atom_confidence,
            "v551_atom_dense_capacity": parent_masks.new_tensor(float(n)),
            "v551_atom_valid_count": atom_valid.to(parent_masks.dtype).sum(dim=1).mean(),
            "v552_safe_calibrated_execution_enabled": parent_masks.new_tensor(1.0),
            "v552_atom_quality_gate_active": parent_masks.new_tensor(
                1.0 if atom_quality_gate_active else 0.0
            ),
            "v552_atom_quality_score": (
                atom_quality_score[atom_valid].mean()
                if bool(atom_valid.any().item())
                else atom_quality_score.new_zeros(())
            ),
            "v552_editor_strength": parent_masks.new_tensor(float(editor_strength)),
            "v551_scale_logits": atom_scale_logits,
            "v551_scale_probs": atom_scale_probs,
            "v551_scale_index": atom_scale_index,
            "v551_boundary_region": boundary_region,
            "v551_boundary_residual": atom_boundary_residual,
            "v551_m1_logit_delta": m1_logit_delta,
            "v551_m1_exact_candidate_st": m1_exact_st,
            "v552r4209_m1_native_probability": m1_native_probability_r4209,
            "v552r4209_m1_native_hard_probability": m1_native_hard_probability_r4209,
            "v552r4209_m1_native_valid_fraction": m1_native_valid_fraction_r4209.detach(),
            "v552r4209_m1_native_presence_mean": m1_native_presence_mean_r4209.detach(),
            "v552r4210_m1_native_probability": m1_native_probability_r4209,
            "v552r4210_m1_native_hard_probability": m1_native_hard_probability_r4209,
            "v552r4210_m1_native_train_probability": m1_native_train_probability_r4210,
            "v552r4210_m1_native_alignment_active": m1_native_alignment_active_r4210,
            # R4211 keeps the exact same deployable Native definition; only proposal/existence
            # and geometry/overflow responsibilities are decoupled upstream.  Expose explicit
            # aliases so train/test tooling cannot accidentally fall back to an older contract.
            "v552r4211_m1_native_probability": m1_native_probability_r4209,
            "v552r4211_m1_native_hard_probability": m1_native_hard_probability_r4209,
            "v552r4211_m1_native_train_probability": m1_native_train_probability_r4210,
            "v552r4211_rootfix_enabled": parent_masks.new_tensor(1.0 if self.instance_valid_decoupling_r4211_enabled else 0.0),
            "v552r4211_proposal_existence_decoupling_enabled": parent_masks.new_tensor(1.0 if self.proposal_existence_decoupling_r4211_enabled else 0.0),
            "v552r4211_geometry_overflow_decoupling_enabled": parent_masks.new_tensor(1.0 if self.geometry_overflow_decoupling_r4211_enabled else 0.0),
            "v552r4211_proposal_seed_count": r4211_proposal_seed_count,
            "v552r4211_proposal_confidence_mean": r4211_proposal_confidence_mean,
            "v552r4211_presence_expected_count": r4211_presence_expected_count,
            "v552r4211_presence_hard_count": r4211_presence_hard_count,
            "v552r4211_geometry_effective_l1": r4211_geometry_effective_l1,
            "v552r4211_geometry_slot_probability": r4211_geometry_slot_probability,
            "v552r4211_effective_slot_probability": r4211_effective_slot_probability,
            # V552-R4.21.2 contract/audit fields.
            "v552r4212_rootfix_enabled": parent_masks.new_tensor(
                1.0 if (
                    self.independent_candidate_set_r4212_enabled
                    or self.disable_visual_seed_identity_r4212_enabled
                    or self.existence_no_object_r4212_enabled
                    or self.candidate_alignment_r4212_enabled
                    or self.direct_delta_utility_r4212_enabled
                    or self.zero_stop_one_step_r4212_enabled
                ) else 0.0
            ),
            "v552r4212_independent_candidate_set_enabled": parent_masks.new_tensor(1.0 if self.independent_candidate_set_r4212_enabled else 0.0),
            "v552r4212_visual_seed_identity_disabled": parent_masks.new_tensor(1.0 if self.disable_visual_seed_identity_r4212_enabled else 0.0),
            "v552r4212_existence_no_object_enabled": parent_masks.new_tensor(1.0 if self.existence_no_object_r4212_enabled else 0.0),
            "v552r4212_candidate_alignment_enabled": parent_masks.new_tensor(1.0 if self.candidate_alignment_r4212_enabled else 0.0),
            "v552r4212_direct_delta_utility_enabled": parent_masks.new_tensor(1.0 if self.direct_delta_utility_r4212_enabled else 0.0),
            "v552r4212_zero_stop_one_step_enabled": parent_masks.new_tensor(1.0 if self.zero_stop_one_step_r4212_enabled else 0.0),
            "v552r4212_parent_existence_count": (parent_presence_logits.detach() > 0.0).to(parent_masks.dtype).sum(dim=1).mean(),
            "v552r4212_deployment_candidate_count": deployment_valid.to(parent_masks.dtype).sum(dim=1).mean(),
            "v552r4212_independent_soft_overlap_mass": F.relu(parent_masks.sum(dim=1) - 1.0).mean().detach(),
            "v560_clean_core_enabled": parent_masks.new_tensor(1.0 if self.clean_core_v560_enabled else 0.0),
            "v560_direct_mask_probability_mean": v560_direct_mask_probability_mean.detach(),
            "v560_independent_soft_overlap_mass": v560_independent_soft_overlap_mass.detach(),
            "v560_q0_pairwise_cosine": v560_q0_pairwise_cosine.detach(),
            "v560_q1_pairwise_cosine": v560_q1_pairwise_cosine.detach(),
            "v560_mask_bias_mean": v560_mask_bias_mean.detach(),
            # V561 BCRS-M1 ownership/learning diagnostics.
            "v561_bcrs_enabled": parent_masks.new_tensor(1.0 if self.base_conditioned_residual_set_v561_enabled else 0.0),
            "v561_geometry_owner": v561_geometry_owner.detach(),
            "v561_variant_static": parent_masks.new_tensor(1.0 if (self.base_conditioned_residual_set_v561_enabled and self.bcrs_v561_variant == "static") else 0.0),
            "v561_variant_image": parent_masks.new_tensor(1.0 if (self.base_conditioned_residual_set_v561_enabled and self.bcrs_v561_variant == "image") else 0.0),
            "v561_variant_typed": parent_masks.new_tensor(1.0 if (self.base_conditioned_residual_set_v561_enabled and self.bcrs_v561_variant == "typed") else 0.0),
            "v561_mask_probability_mean": v561_mask_probability_mean.detach(),
            "v561_soft_overlap_mass": v561_soft_overlap_mass.detach(),
            "v561_q0_pairwise_cosine": v561_q0_pairwise_cosine.detach(),
            "v561_q1_pairwise_cosine": v561_q1_pairwise_cosine.detach(),
            "v561_q2_pairwise_cosine": v561_q2_pairwise_cosine.detach(),
            "v561_stage1_query_delta_norm": v561_stage1_query_delta_norm.detach(),
            "v561_stage2_query_delta_norm": v561_stage2_query_delta_norm.detach(),
            "v561_typed_support_mean": v561_typed_support_mean.detach(),
            "v561_typed_support_std": v561_typed_support_std.detach(),
            "v561_typed_support_neutrality_error": v561_typed_support_neutrality_error.detach(),
            "v561_stage1_attention_entropy_ratio": v561_stage1_attention_entropy_ratio.detach(),
            "v561_stage2_attention_entropy_ratio": v561_stage2_attention_entropy_ratio.detach(),
            "v561_mask_bias_mean": v561_mask_bias_mean.detach(),
            # V562 root-repair learning tensors/diagnostics.  The logits remain
            # attached so v538_loss can supervise the proposal head; all other
            # audit fields are detached.
            "v562_rootfix_enabled": parent_masks.new_tensor(1.0 if (self.base_conditioned_residual_set_v561_enabled and self.bcrs_v561_variant in {"rootfix", "persistent"}) else 0.0),
            "v562_residual_logits": v562_residual_logits,
            "v562_residual_probability_mean": v562_residual_probability_mean.detach(),
            "v562_proposal_anchor_xy": v562_proposal_anchor_xy.detach(),
            "v562_query_owned_presence": parent_masks.new_tensor(1.0 if (self.base_conditioned_residual_set_v561_enabled and self.bcrs_v561_variant in {"rootfix", "persistent"}) else 0.0),
            "v562_direct_binary_executor": parent_masks.new_tensor(1.0 if (self.base_conditioned_residual_set_v561_enabled and self.bcrs_v561_variant in {"rootfix", "persistent"}) else 0.0),
            # V563 persistent local-binding tensors.  Window/raw probability stay
            # attached because the local geometry loss consumes them.
            "v563_rootfix_enabled": parent_masks.new_tensor(1.0 if (self.base_conditioned_residual_set_v561_enabled and self.bcrs_v561_variant == "persistent") else 0.0),
            "v563_mask_window": v563_mask_window,
            "v563_attention_window": v563_attention_window,
            "v563_raw_mask_probability": v563_raw_mask_probability,
            "v563_pre_gate_mask_probability_mean": v563_pre_gate_mask_probability_mean.detach(),
            "v563_outside_mask_probability": v563_outside_mask_probability.detach(),
            "v563_attention_window_fraction": v563_attention_window_fraction.detach(),
            "v563_mask_window_fraction": v563_mask_window_fraction.detach(),
            "v563_identity_retention_q1": v563_identity_retention_q1.detach(),
            "v563_identity_retention_q2": v563_identity_retention_q2.detach(),
            "v564_rootfix_enabled": v564_rootfix_enabled.detach(),
            "v564_proposal_radius": v564_proposal_radius.detach(),
            "v564_proposal_radius_mean": v564_proposal_radius_mean.detach(),
            "v564_proposal_shape_prior_abs_mean": v564_proposal_shape_prior_abs_mean.detach(),
            "v564_dual_stream_identity_enabled": v564_dual_stream_identity_enabled.detach(),
            "v564_typed_spatial_feedback_disabled": v564_typed_spatial_feedback_disabled.detach(),
            "v565_rootfix_enabled": v565_rootfix_enabled.detach(),
            "v565_seed_logits": v565_seed_logits,
            "v565_seed_probability_mean": v565_seed_probability_mean.detach(),
            "v565_relative_support_mean": v565_relative_support_mean.detach(),
            "v565_attention_radius_mean": v565_attention_radius_mean.detach(),
            "v565_peak_to_background_contrast_mean": v565_peak_to_background_contrast_mean.detach(),
            "v565_shape_condition_abs_mean": v565_shape_condition_abs_mean.detach(),
            "clean_dynamic_component_set_enabled": parent_masks.new_tensor(1.0 if self.clean_dynamic_component_set_enabled else 0.0),
            "tc_drcs_enabled": parent_masks.new_tensor(1.0 if self.tc_drcs_enabled else 0.0),
            "clean_attention_precision_mean": clean_attention_precision_mean.detach(),
            "clean_mask_precision_mean": clean_mask_precision_mean.detach(),
            "clean_loss_log_vars": clean_loss_log_vars,
            "tc_stage0_logits": tc_stage0_logits,
            "tc_pilot_logits": tc_pilot_logits,
            "tc_pilot_action_logits": tc_pilot_action_logits,
            "tc_pilot_valid": tc_pilot_valid,
            "tc_pilot_teacher_masks": tc_pilot_teacher_masks,
            "tc_pilot_teacher_actions": tc_pilot_teacher_actions,
            "tc_teacher_raw_count": tc_teacher_raw_count.detach(),
            "v551_first_selector_features": first_selector["selector_features"],
            "v551_editor_route_logits": editor_route_logits,
            "v551_editor_route_probs": editor_route_probs,
            "v551_editor_route_index": editor_route_index,
            "v551_editor_dose_adjust_logits": editor_dose_adjust_logits,
            "v551_editor_dose_adjust": editor_dose_adjust,
            "v551_editor_region": editor_region,
            "v551_local_residual": local_residual,
            "v551_editor_logit_delta": editor_logit_delta,
            "v551_editor_exact_candidate_st": editor_exact_st,
            "v552r4_enabled": parent_masks.new_tensor(
                1.0 if self.unified_reference_r4_enabled else 0.0
            ),
            "v552r4_route_delta_bank": route_delta_bank,
            "v552r4_route_candidate_bank": route_candidate_bank,
            "v552r4_route_feature_bank": route_feature_bank,
            "v552r42_safety_route_feature_bank": safety_route_feature_bank,
            "v552r42_utility_route_feature_bank": utility_route_feature_bank,
            "v552r44_safety_value_feature_bank": safety_value_feature_bank,
            "v552r44_utility_value_feature_bank": utility_value_feature_bank,
            "v552r45_error_prone_logits": error_prone_logits,
            "v552r45_error_prone_probs": error_prone_probs,
            "v552r45_error_aware_enabled": parent_masks.new_tensor(
                1.0 if self.error_aware_r45_enabled else 0.0
            ),
            "v552r45_factorized_safety_enabled": parent_masks.new_tensor(
                1.0 if self.factorized_safety_r45_enabled else 0.0
            ),
            "v552r45_direct_signed_utility_enabled": parent_masks.new_tensor(
                1.0 if self.direct_signed_utility_r45_enabled else 0.0
            ),
            "v552r45_factorized_composer_enabled": parent_masks.new_tensor(
                1.0 if self.factorized_composer_r45_enabled else 0.0
            ),
            "v552r46_rootfix_enabled": parent_masks.new_tensor(
                1.0 if self.native_contract_r46_enabled else 0.0
            ),
            "v552r46_native_state_only_enabled": parent_masks.new_tensor(
                1.0 if self.native_contract_r46_enabled else 0.0
            ),
            "v552r46_same_candidate_contract_enabled": parent_masks.new_tensor(
                1.0 if self.native_contract_r46_enabled else 0.0
            ),
            "v552r46_formal_policy_audit_enabled": parent_masks.new_tensor(
                1.0 if self.native_contract_r46_enabled else 0.0
            ),
            "v552r47_rootfix_enabled": parent_masks.new_tensor(
                1.0 if self.spatial_realization_r47_enabled else 0.0
            ),
            "v552r47_direct_slot_components_enabled": parent_masks.new_tensor(
                1.0 if self.r47_direct_slot_components else 0.0
            ),
            "v552r47_anchor_params": atom_r47_anchor,
            "v552r47_query_mask_logits": atom_r47_query_logits,
            "v552r48_rootfix_enabled": parent_masks.new_tensor(
                1.0 if self.iterative_binding_r48_enabled else 0.0
            ),
            "v552r48_remove_coarse_mask_bias_enabled": parent_masks.new_tensor(
                1.0 if self.r48_remove_coarse_mask_bias else 0.0
            ),
            "v552r48_stage_mask_logits": r48_stage_logits,
            "v552r48_stage_anchor_params": r48_stage_anchors,
            "v552r48_local_attention_entropy": r48_attention_entropy,
            "v552r49_rootfix_enabled": parent_masks.new_tensor(
                1.0 if self.content_selective_r49_enabled else 0.0
            ),
            "v552r49_anchor_sampling_only_enabled": parent_masks.new_tensor(
                1.0 if (self.content_selective_r49_enabled and self.r49_anchor_sampling_only) else 0.0
            ),
            "v552r49_dn_curriculum_enabled": parent_masks.new_tensor(
                1.0 if (self.content_selective_r49_enabled and self.r49_dn_curriculum_enabled) else 0.0
            ),
            "v552r49_attention_max_weight": r49_attention_max_weight,
            "v552r49_dn_noise_scale": r49_dn_noise_scale,
            "v552r49_attention_logit_scale_mean": (
                self.r49_logit_scale.exp().clamp(max=self.r49_attention_logit_scale_max).mean().to(parent_masks.dtype)
                if self.r49_logit_scale is not None
                else parent_masks.new_zeros(())
            ),
            "v552r410_rootfix_enabled": parent_masks.new_tensor(
                1.0 if self.evidence_proposal_r410_enabled else 0.0
            ),
            "v552r410_evidence_proposal_enabled": parent_masks.new_tensor(
                1.0 if (self.evidence_proposal_r410_enabled and self.r410_use_evidence_proposals) else 0.0
            ),
            "v552r410_support_only_local_readout_enabled": parent_masks.new_tensor(
                1.0 if (self.evidence_proposal_r410_enabled and self.r410_support_only_local_readout) else 0.0
            ),
            "v552r410_proposal_score_mean": r410_proposal_score_mean,
            "v552r410_proposal_valid_fraction": r410_proposal_valid_fraction,
            "v552r410_dn_clean_curriculum_enabled": parent_masks.new_tensor(
                1.0 if (self.evidence_proposal_r410_enabled and self.r410_dn_clean_curriculum_enabled) else 0.0
            ),
            "v552r411_rootfix_enabled": parent_masks.new_tensor(
                1.0 if self.native_residual_set_r411_enabled else 0.0
            ),
            "v552r411_typed_proposal_enabled": parent_masks.new_tensor(
                1.0 if (self.native_residual_set_r411_enabled and self.r411_typed_proposal_enabled) else 0.0
            ),
            "v552r411_local_roi_decoder_enabled": parent_masks.new_tensor(
                1.0 if (self.native_residual_set_r411_enabled and self.r411_local_roi_decoder_enabled) else 0.0
            ),
            "v552r411_raw_native_mask_enabled": parent_masks.new_tensor(
                1.0 if (self.native_residual_set_r411_enabled and self.r411_use_raw_native_masks) else 0.0
            ),
            "v552r412_rootfix_enabled": parent_masks.new_tensor(
                1.0 if self.canonical_shape_r412_enabled else 0.0
            ),
            "v552r412_canonical_renderer_enabled": parent_masks.new_tensor(
                1.0 if (self.canonical_shape_r412_enabled and self.r412_roi_stem is not None) else 0.0
            ),
            "v552r412_action_support_enabled": parent_masks.new_tensor(
                1.0 if (self.canonical_shape_r412_enabled and self.r412_action_support_strength > 0.0) else 0.0
            ),
            "v552r413_rootfix_enabled": parent_masks.new_tensor(1.0 if self.geometry_lock_r413_enabled else 0.0),
            "v552r413_geometry_locked_enabled": parent_masks.new_tensor(1.0 if self.geometry_lock_r413_enabled else 0.0),
            "v552r413_query_extent_enabled": parent_masks.new_tensor(1.0 if (self.geometry_lock_r413_enabled and self.r413_query_extent_enabled) else 0.0),
            "v552r414_rootfix_enabled": parent_masks.new_tensor(1.0 if self.geometry_context_r414_enabled else 0.0),
            "v552r414_context_grid_size": parent_masks.new_tensor(float(self.r414_context_grid_size if self.geometry_context_r414_enabled else 0)),
            "v552r414_context_radius": parent_masks.new_tensor(float(self.r414_context_radius if self.geometry_context_r414_enabled else 0.0)),
            "v552r416_rootfix_enabled": parent_masks.new_tensor(1.0 if (self.unique_point_r416_enabled or self.r416_asymmetric_ltrb_enabled) else 0.0),
            "v552r416_unique_point_enabled": parent_masks.new_tensor(1.0 if self.unique_point_r416_enabled else 0.0),
            "v552r416_asymmetric_ltrb_enabled": parent_masks.new_tensor(1.0 if self.r416_asymmetric_ltrb_enabled else 0.0),
            "v552r416_cross_type_nms_radius_px": parent_masks.new_tensor(float(self.r416_cross_type_nms_radius_px)),
            "v552r416_legacy_topk_unique_fraction": r416_legacy_unique_fraction.to(parent_masks.dtype),
            "v552r416_proposal_point_xy": atom_r416_proposal_point,
            "v552r416_parent_proposal_point_xy": parent_r416_proposal_point,
            "v552r416_edge_offsets": atom_r416_edge_offsets,
            "v552r416_parent_edge_offsets": parent_r416_edge_offsets,
            "v552r417_rootfix_enabled": parent_masks.new_tensor(1.0 if self.proposal_recovery_r417_enabled else 0.0),
            "v552r417_location_first_enabled": parent_masks.new_tensor(1.0 if self.proposal_recovery_r417_enabled else 0.0),
            "v552r417_shared_offset_enabled": parent_masks.new_tensor(1.0 if (self.proposal_recovery_r417_enabled and self.r417_shared_offset_enabled) else 0.0),
            "v552r417_location_nms_kernel": parent_masks.new_tensor(float(self.r417_location_nms_kernel)),
            "v552r417_location_dedup_radius_px": parent_masks.new_tensor(float(self.r417_location_dedup_radius_px)),
            "v552r417_location_logits": r417_location_logits,
            "v552r417_location_offset_map": r417_location_offset_map,
            "v552r417_proposal_point_xy": atom_r416_proposal_point,
            "v552r417_parent_proposal_point_xy": parent_r416_proposal_point,
            "v552r418_rootfix_enabled": parent_masks.new_tensor(1.0 if self.box_free_mask_set_r418_enabled else 0.0),
            "v552r418_box_free_mask_set_enabled": parent_masks.new_tensor(1.0 if self.box_free_mask_set_r418_enabled else 0.0),
            "v552r418_paired_stable_teacher_enabled": parent_masks.new_tensor(
                1.0 if (self.box_free_mask_set_r418_enabled and self.r418_paired_stable_teacher_enabled) else 0.0
            ),
            "v552r418_paired_mask_logits": r418_paired_logits,
            "v552r418_paired_teacher_masks": r418_paired_teacher_masks,
            "v552r418_paired_teacher_valid": r418_paired_teacher_valid,
            "v552r418_paired_coarse": r418_paired_coarse,
            "v552r419_rootfix_enabled": parent_masks.new_tensor(1.0 if self.seeded_masked_attention_r419_enabled else 0.0),
            "v552r419_seeded_masked_attention_enabled": parent_masks.new_tensor(1.0 if self.seeded_masked_attention_r419_enabled else 0.0),
            "v552r419_seed_radius_px": parent_masks.new_tensor(float(self.r419_seed_radius_px if self.seeded_masked_attention_r419_enabled else 0.0)),
            "v552r419_support_dilate_kernel": parent_masks.new_tensor(float(self.r419_support_dilate_kernel if self.seeded_masked_attention_r419_enabled else 0)),
            "v552r419_seed_support_fraction": r419_seed_support_fraction,
            "v552r419_final_support_fraction": r419_final_support_fraction,
            "v552r419_outside_mask_probability": r419_outside_mask_probability,
            "v552r420_rootfix_enabled": parent_masks.new_tensor(1.0 if self.dynamic_residual_mask_r420_enabled else 0.0),
            "v552r420_dynamic_mask_enabled": r420_dynamic_mask_enabled,
            "v552r420_type_decoupled_mask_enabled": r420_type_decoupled_mask_enabled,
            "v552r420_dynamic_channels": r420_dynamic_channels,
            "v552r420_relative_coord_mean_abs": r420_relative_coord_mean_abs,
            "v552r4201_clean_rootfix_enabled": r4201_clean_rootfix_enabled,
            "v552r4203_rootfix_enabled": parent_masks.new_tensor(1.0 if self.dense_competitive_residual_set_r4203_enabled else 0.0),
            "v552r4203_dense_competitive_set_enabled": r4203_dense_set_enabled,
            "v552r4203_ownership_sum_error": r4203_ownership_sum_error,
            "v552r4203_assignment_entropy": r4203_assignment_entropy,
            "v552r4203_background_fraction": r4203_background_fraction,
            "v552r4203_max_slot_ownership": r4203_max_slot_ownership,
            "v552r4203_slot_mass_cv": r4203_slot_mass_cv,
            "v552r4203_point_bottleneck_used": r4203_point_bottleneck_used,
            "v552r4204_rootfix_enabled": parent_masks.new_tensor(1.0 if self.factorized_residual_existence_identity_r4204_enabled else 0.0),
            "v552r4204_occupancy_logits": r4204_occupancy_logits,
            "v552r4204_residual_mass_conservation_error": r4204_residual_mass_conservation_error,
            "v552r4204_conditional_slot_entropy": r4204_conditional_slot_entropy,
            "v552r4204_conditional_max_slot_probability": r4204_conditional_max_slot_probability,
            "v552r4204_residual_existence_mean": r4204_residual_existence_mean,
            "v552r4204_centroid_separation": r4204_centroid_separation,
            "v552r4204_spatial_variance_mean": r4204_spatial_variance_mean,
            "v552r4204_spatial_identity_enabled": r4204_spatial_identity_enabled,
            "v552r4204_location_as_occupancy_used": r4204_location_as_occupancy_used,
            "v552r4205_rootfix_enabled": r4205_enabled,
            "v552r4205_overflow_probability": r4205_overflow_probability,
            "v552r4205_overflow_identity_probability": r4205_overflow_identity_probability,
            "v552r4205_conditional_identity_logits": r4205_conditional_identity_logits,
            "v552r4205_editable_probability_sum": r4205_editable_probability_sum,
            "v552r4205_overflow_probability_mean": r4205_overflow_probability_mean,
            "v552r4205_overflow_conditional_mean": r4205_overflow_conditional_mean,
            "v552r4205_editable_probability_mean": r4205_editable_probability_mean,
            "v552r4205_final_logits_finite_fraction": r4205_final_logits_finite_fraction,
            "v552r4207_rootfix_enabled": r4207_enabled,
            "v552r4207_seed_center_xy": r4207_seed_center_xy,
            "v552r4207_seed_score": r4207_seed_score,
            "v552r4207_seed_valid": r4207_seed_valid,
            "v552r4207_seed_feature_finite_fraction": r4207_seed_feature_finite_fraction,
            "v552r4207_seed_valid_fraction": r4207_seed_valid_fraction,
            "v552r4207_seed_score_mean": r4207_seed_score_mean,
            "v552r4207_seed_pairwise_distance_px": r4207_seed_pairwise_distance_px,
            "v552r4207_q0_pairwise_cosine": r4207_q0_pairwise_cosine,
            "v552r4207_q1_pairwise_cosine": r4207_q1_pairwise_cosine,
            "v552r4207_seed_to_slot_centroid_drift_px": r4207_seed_to_slot_centroid_drift_px,
            "v552r4207_full_image_assignment_enabled": r4207_full_image_assignment_enabled,
            "v552r4207_hard_spatial_support_used": r4207_hard_spatial_support_used,
            "v552r4208_rootfix_enabled": r4208_enabled,
            "v552r4208_normalized_fusion_enabled": r4208_normalized_fusion_enabled,
            "v552r4208_persistent_identity_enabled": r4208_persistent_identity_enabled,
            "v552r4208_learned_query_norm": r4208_learned_query_norm,
            "v552r4208_seed_feature_norm": r4208_seed_feature_norm,
            "v552r4208_seed_to_learned_norm_ratio": r4208_seed_to_learned_norm_ratio,
            "v552r4208_seed_feature_pairwise_cosine": r4208_seed_feature_pairwise_cosine,
            "v552r4208_q0_seed_identity_cosine": r4208_q0_seed_identity_cosine,
            "v552r4208_q1_seed_identity_cosine": r4208_q1_seed_identity_cosine,
            "v552r4208_identity_retention_delta": r4208_identity_retention_delta,
            "v552r4210_rootfix_enabled": r4210_enabled,
            "v552r4210_variable_cardinality_seed_enabled": r4210_variable_seed_enabled,
            "v552r4210_independent_overflow_enabled": r4210_independent_overflow_enabled,
            "v552r4210_valid_seed_count": r4210_valid_seed_count,
            "v552r4210_seed_logit": r4210_seed_logit,
            "v552r4210_seed_logit_mean": r4210_seed_logit_mean,
            "v552r4210_overflow_gate_logits": r4210_overflow_gate_logits,
            "v552r4210_overflow_conditional_probability": r4210_overflow_conditional_probability,
            "v552r4210_overflow_logit_mean": r4210_overflow_logit_mean,
            "v552r4210_overflow_conditional_mean": r4210_overflow_conditional_mean,
            "v552r4211_dense_rootfix_enabled": r4211_enabled,
            "v552r4211_dense_proposal_existence_enabled": r4211_proposal_existence_enabled,
            "v552r4211_dense_geometry_overflow_enabled": r4211_geometry_overflow_enabled,
            "v552r48_teacher_built": r48_teacher_built,
            "v552r48_teacher_valid_count": r48_teacher_valid_count,
            "v552r413_proposal_anchor_params": atom_r413_proposal_anchor,
            "v552r413_parent_proposal_anchor_params": parent_r413_proposal_anchor,
            "v552r411_proposal_type": r411_proposal_type,
            "v552r411_proposal_score": r411_proposal_score,
            "v552r411_proposal_valid": r411_proposal_valid,
            "v552r411_center_logits": r411_center_logits,
            "v552r411_size_map": r411_size_map,
            "v552r411_offset_map": r411_offset_map,
            "v552r48_dn_enabled": parent_masks.new_tensor(1.0 if self.r48_dn_enabled else 0.0),
            "v552r48_dn_stage_mask_logits": r48_dn_stage_logits,
            "v552r48_dn_stage_anchor_params": r48_dn_stage_anchors,
            "v552r48_dn_target_index": r48_dn_target_index,
            "v552r48_dn_valid": r48_dn_valid,
            "v552r48_teacher_masks": r48_teacher_masks,
            "v552r48_teacher_actions": r48_teacher_actions,
            "v552r48_teacher_valid": r48_teacher_valid,
            "v552r48_teacher_area": r48_teacher_area,
            "v552r48_teacher_geometry": r48_teacher_geometry,
            "v552r48_teacher_effective": r48_teacher_effective,
            "v552r48_teacher_error": r48_teacher_error,
            "v552r4_route_valid": route_valid,
            "v552r4_editor_safety_logits_bank": editor_safety_logits_bank,
            "v552r45_editor_safety_benefit_logits_bank": (
                editor_safety_benefit_logits_bank
            ),
            "v552r45_editor_safety_harm_logits_bank": (
                editor_safety_harm_logits_bank
            ),
            "v552r4_editor_incremental_gain_bank": editor_incremental_gain_bank,
            "v552r42_editor_benefit_magnitude_bank": (
                editor_benefit_magnitude_bank
            ),
            "v552r42_editor_harm_magnitude_bank": editor_harm_magnitude_bank,
            "v552r4_candidate_utility_logits_bank": candidate_utility_logits_bank,
            "v552r4_candidate_absolute_gain_bank": candidate_absolute_gain_bank,
            "v552r45_candidate_direct_signed_gain_bank": direct_signed_utility_bank,
            "v552r45_candidate_expected_gain_bank": expected_utility_gain_bank,
            "v552r42_candidate_benefit_magnitude_bank": (
                candidate_benefit_magnitude_bank
            ),
            "v552r42_candidate_harm_magnitude_bank": (
                candidate_harm_magnitude_bank
            ),
            "v552r4_exact_incremental_gain_bank": exact_incremental_gain_bank,
            "v552r4_exact_absolute_gain_bank": exact_absolute_gain_bank,
            "v552r4_safety_target_bank": safety_target_bank,
            "v552r4_utility_target_bank": utility_target_bank,
            "v552r4_teacher_safe_route_index": teacher_safe_route_index,
            "v552r4_teacher_safe_delta": teacher_safe_delta,
            "v552r4_teacher_safe_candidate": teacher_safe_probability,
            "v552r4_student_safe_route_index": student_safe_route_index,
            "v552r4_student_safe_delta": student_safe_delta,
            "v552r4_student_safe_candidate": student_safe_probability,
            "v552r4_predicted_editor_safe": predicted_editor_safe,
            "v552r4_safety_queue_logits": safety_queue_logits,
            "v552r4_safety_queue_gain_pred": safety_queue_gain_pred,
            "v552r4_safety_queue_target": safety_queue_target,
            "v552r4_safety_queue_gain_value": safety_queue_gain_value,
            "v552r42_safety_queue_benefit_magnitude": (
                safety_queue_benefit_magnitude
            ),
            "v552r42_safety_queue_harm_magnitude": (
                safety_queue_harm_magnitude
            ),
            "v552r4_utility_queue_logits": utility_queue_logits,
            "v552r4_utility_queue_gain_pred": utility_queue_gain_pred,
            "v552r4_utility_queue_target": utility_queue_target,
            "v552r4_utility_queue_gain_value": utility_queue_gain_value,
            "v552r42_utility_queue_benefit_magnitude": (
                utility_queue_benefit_magnitude
            ),
            "v552r42_utility_queue_harm_magnitude": (
                utility_queue_harm_magnitude
            ),
            "v552r4_safety_queue_count": self._v552r4_safety_queue_count.detach().clone(),
            "v552r4_utility_queue_count": self._v552r4_utility_queue_count.detach().clone(),
            "v552r42_decoupled_critic_enabled": parent_masks.new_tensor(
                1.0 if self.decoupled_critic_r42_enabled else 0.0
            ),
            "v552r42_task_specific_route_features_enabled": parent_masks.new_tensor(
                1.0 if self.decoupled_critic_r42_enabled else 0.0
            ),
            "v552r42_composer_critic_grad_isolated": parent_masks.new_tensor(
                1.0 if self.decoupled_critic_r42_enabled else 0.0
            ),
            "v552r43_spatial_route_evidence_enabled": parent_masks.new_tensor(
                1.0 if self.spatial_evidence_r43_enabled else 0.0
            ),
            "v552r44_rootfix_enabled": parent_masks.new_tensor(
                1.0 if self.audit_gate_r44_enabled else 0.0
            ),
            "v552r44_audit_shadow_enabled": parent_masks.new_tensor(
                1.0 if self.audit_gate_r44_enabled else 0.0
            ),
            "v552r44_class_value_decoupling_enabled": parent_masks.new_tensor(
                1.0 if self.class_value_decoupling_r44_enabled else 0.0
            ),
            "v552r44_semantic_deployment_enabled": parent_masks.new_tensor(
                1.0 if self.semantic_deployment_r44_enabled else 0.0
            ),
            "v552r44_audit_trace_active": parent_masks.new_tensor(
                1.0 if audit_enabled else 0.0
            ),
            "v552r45_rootfix_enabled": parent_masks.new_tensor(
                1.0
                if (
                    (self.error_aware_r45_enabled or self.native_contract_r46_enabled)
                    and self.factorized_safety_r45_enabled
                    and self.direct_signed_utility_r45_enabled
                    and self.factorized_composer_r45_enabled
                )
                else 0.0
            ),
            "v551_editor_enabled": parent_masks.new_tensor(
                1.0 if self.editor_enabled else 0.0
            ),
        }
