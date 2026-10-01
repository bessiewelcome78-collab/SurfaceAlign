"""V524 counterfactual prompted selective region composer.

V524 keeps the validated M1 candidate bank as an explicit action space and
replaces the V523 per-pixel winner-take-all router with four coupled decisions:

1. dynamic candidate-family activation;
2. candidate-relative repair/harm/uncertainty guidance;
3. prompt-guided candidate-region construction on an auditable region grid;
4. set-context candidate routing with complete candidate-region execution, a
   case-level outcome verifier and an explicit Preserve/reject action.

The forward path never receives GT and is candidate-realizable.  Every deployed
pixel is either the factual Base value or the value of one real M1 candidate.
Privileged region utilities are constructed only by ``v484_loss.py``.
"""
from __future__ import annotations

from typing import Dict, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.v503_factual_atomic_causal import EPS, _as_b1hw, _entropy, _gray_edge, _soft_boundary


def _groups(channels: int) -> int:
    groups = min(8, int(channels))
    while groups > 1 and int(channels) % groups != 0:
        groups -= 1
    return groups


def _resize(value: torch.Tensor, size: Tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
    if value.shape[-2:] == size:
        return value
    if mode == "nearest":
        return F.interpolate(value, size=size, mode=mode)
    return F.interpolate(value, size=size, mode=mode, align_corners=False)


class _StraightThroughOneHot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
        del ctx, soft
        return hard

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        del ctx
        return None, grad_output


class V524CounterfactualPromptedRegionComposer(nn.Module):
    """Prompt-guided region-wise candidate composition with safe rejection.

    A regular region grid is intentionally used in the first auditable V524
    implementation.  Candidate supports and learned positive/negative/boundary
    prompt maps decide which grid regions are active.  Candidate-axis attention
    then compares all candidate actions that overlap the same region.  Each accepted action executes the selected candidate on its complete valid
    support inside the selected region.  No learned pixel gate may fragment the
    action.  A case-level verifier can reject the entire composed proposal.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        metadata_dim: int = 16,
        semantic_channels: int = 512,
        max_candidates: int = 64,
        family_count: int = 5,
        action_count: int = 10,
        score_stride: int = 2,
        region_grid_size: int = 7,
        set_layers: int = 2,
        set_heads: int = 4,
        set_ffn_dim: int = 192,
        top_family_count: int = 3,
        support_floor: float = 1.0e-4,
        edit_epsilon: float = 1.0e-4,
        route_temperature: float = 0.50,
        lcb_beta: float = 0.50,
        harm_penalty: float = 1.50,
        family_log_weight: float = 0.25,
        prompt_weight: float = 0.50,
        region_accept_threshold: float = 0.05,
        prompt_accept_threshold: float = 0.35,
        refiner_gate_threshold: float = 0.55,
        full_region_execution: bool = True,
        case_guard_enabled: bool = True,
        case_accept_threshold: float = 0.0,
        case_harm_threshold: float = 0.10,
        deploy_start_epoch: int = 30,
        dropout: float = 0.10,
        hard_inference: bool = True,
        utility_aligned_score: bool = False,
        route_residual_scale: float = 0.01,
        causal_hard_mask: bool = False,
        causal_mask_threshold: float = 0.35,
        causal_mask_start_epoch: int = 0,
        force_preserve_output: bool = False,
        oracle_preserving_validity: bool = False,
        factorized_ranker: bool = False,
        editability_threshold: float = 0.50,
        conditional_temperature: float = 0.25,
        validation_expose_proposal: bool = False,
        outcome_composer: bool = False,
        outcome_temperature: float = 0.10,
        outcome_lcb_beta: float = 0.50,
        outcome_min_gain: float = 0.0,
        outcome_max_harm: float = 0.45,
        outcome_exploration_epochs: int = 3,
        v529_calibration_first_outcome: bool = False,
        v529_selector_enabled: bool = False,
        v529_uncertainty_enabled: bool = False,
        v529_execution_enabled: bool = False,
        v529_route_temperature: float = 1.0,
        v529_utility_scale: float = 0.005,
        v529_lcb_beta: float = 0.0,
        v529_min_gain: float = 0.0,
        v529_max_harm: float = 0.45,
        v530_probability_calibrated_outcome: bool = False,
        v530_selector_enabled: bool = False,
        v530_uncertainty_enabled: bool = False,
        v530_execution_enabled: bool = False,
        v530_route_temperature: float = 1.0,
        v530_utility_scale: float = 0.005,
        v530_lcb_beta: float = 0.0,
        v530_min_gain: float = 0.0,
        v530_max_harm: float = 0.45,
        v530_base_fp_prior: float = 0.10,
        v530_base_fn_prior: float = 0.01,
        v530_add_fix_prior: float = 0.35,
        v530_remove_fix_prior: float = 0.25,
        logvar_min: float = -6.0,
        logvar_max: float = 3.0,
        logvar_init: float = -1.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_candidates = int(max_candidates)
        self.family_count = int(family_count)
        self.action_count = int(action_count)
        self.score_stride = max(int(score_stride), 1)
        self.region_grid_size = max(int(region_grid_size), 2)
        self.top_family_count = min(max(int(top_family_count), 1), self.family_count)
        self.support_floor = max(float(support_floor), 0.0)
        self.edit_epsilon = max(float(edit_epsilon), 0.0)
        self.route_temperature = max(float(route_temperature), 1.0e-3)
        self.lcb_beta = max(float(lcb_beta), 0.0)
        self.harm_penalty = max(float(harm_penalty), 0.0)
        self.family_log_weight = max(float(family_log_weight), 0.0)
        self.prompt_weight = max(float(prompt_weight), 0.0)
        self.region_accept_threshold = float(region_accept_threshold)
        self.prompt_accept_threshold = min(max(float(prompt_accept_threshold), 0.0), 1.0)
        self.refiner_gate_threshold = min(max(float(refiner_gate_threshold), 0.0), 1.0)
        self.full_region_execution = bool(full_region_execution)
        self.case_guard_enabled = bool(case_guard_enabled)
        self.case_accept_threshold = float(case_accept_threshold)
        self.case_harm_threshold = min(max(float(case_harm_threshold), 1.0e-4), 1.0 - 1.0e-4)
        self.deploy_start_epoch = max(int(deploy_start_epoch), 0)
        self.dropout = max(float(dropout), 0.0)
        self.hard_inference = bool(hard_inference)
        # V526: all terms in the deployed score must live in counterfactual
        # utility units.  Family/prompt predictions are used as candidate
        # filters/context, while the free route head is only a bounded residual.
        self.utility_aligned_score = bool(utility_aligned_score)
        self.route_residual_scale = max(float(route_residual_scale), 0.0)
        self.causal_hard_mask = bool(causal_hard_mask)
        self.causal_mask_threshold = min(max(float(causal_mask_threshold), 0.0), 1.0)
        self.causal_mask_start_epoch = max(int(causal_mask_start_epoch), 0)
        self.force_preserve_output = bool(force_preserve_output)
        # V527: the ranker is not allowed to define its own teacher action set.
        # Only deterministic geometric realizability may remove an action.
        self.oracle_preserving_validity = bool(oracle_preserving_validity)
        # V527 factorizes Preserve-vs-Edit from the conditional candidate rank.
        self.factorized_ranker = bool(factorized_ranker)
        self.editability_threshold = min(max(float(editability_threshold), 0.0), 1.0)
        self.conditional_temperature = max(float(conditional_temperature), 1.0e-4)
        self.validation_expose_proposal = bool(validation_expose_proposal)
        # V528 predicts physically conserved counterfactual outcomes.  Add pixels
        # split into FN-fix/TN-harm and remove pixels split into FP-fix/TP-harm.
        # Candidate selection is based on the analytically reconstructed Dice gain.
        self.outcome_composer = bool(outcome_composer)
        self.outcome_temperature = max(float(outcome_temperature), 1.0e-4)
        self.outcome_lcb_beta = max(float(outcome_lcb_beta), 0.0)
        self.outcome_min_gain = float(outcome_min_gain)
        self.outcome_max_harm = min(max(float(outcome_max_harm), 0.0), 1.0)
        self.outcome_exploration_epochs = max(int(outcome_exploration_epochs), 0)
        # V529 separates outcome calibration, selector learning and uncertainty
        # calibration.  Its ranking score is the analytically reconstructed Dice
        # gain in normalized utility units; uncertainty is never allowed to
        # change the mean prediction during calibration.
        self.v529_calibration_first_outcome = bool(v529_calibration_first_outcome)
        self.v529_selector_enabled = bool(v529_selector_enabled)
        self.v529_uncertainty_enabled = bool(v529_uncertainty_enabled)
        self.v529_execution_enabled = bool(v529_execution_enabled)
        self.v529_route_temperature = max(float(v529_route_temperature), 1.0e-4)
        self.v529_utility_scale = max(float(v529_utility_scale), 1.0e-6)
        self.v529_lcb_beta = max(float(v529_lcb_beta), 0.0)
        self.v529_min_gain = float(v529_min_gain)
        self.v529_max_harm = min(max(float(v529_max_harm), 0.0), 1.0)
        # V530 fixes two calibration failures observed in V529: zero-bias rare
        # error initialization and class-balanced BCE scores being integrated as
        # probabilities.  V530 predicts calibrated full-resolution FP/FN and
        # candidate Add/Remove correctness maps, then preserves the same exact
        # conserved outcome and analytical Dice reconstruction.
        self.v530_probability_calibrated_outcome = bool(v530_probability_calibrated_outcome)
        self.v530_selector_enabled = bool(v530_selector_enabled)
        self.v530_uncertainty_enabled = bool(v530_uncertainty_enabled)
        self.v530_execution_enabled = bool(v530_execution_enabled)
        self.v530_route_temperature = max(float(v530_route_temperature), 1.0e-4)
        self.v530_utility_scale = max(float(v530_utility_scale), 1.0e-6)
        self.v530_lcb_beta = max(float(v530_lcb_beta), 0.0)
        self.v530_min_gain = float(v530_min_gain)
        self.v530_max_harm = min(max(float(v530_max_harm), 0.0), 1.0)
        self.v530_base_fp_prior = min(max(float(v530_base_fp_prior), 1.0e-4), 1.0 - 1.0e-4)
        self.v530_base_fn_prior = min(max(float(v530_base_fn_prior), 1.0e-4), 1.0 - 1.0e-4)
        self.v530_add_fix_prior = min(max(float(v530_add_fix_prior), 1.0e-4), 1.0 - 1.0e-4)
        self.v530_remove_fix_prior = min(max(float(v530_remove_fix_prior), 1.0e-4), 1.0 - 1.0e-4)
        self.logvar_min = min(float(logvar_min), float(logvar_max))
        self.logvar_max = max(float(logvar_min), float(logvar_max))
        self.logvar_init = min(max(float(logvar_init), self.logvar_min), self.logvar_max)
        self.current_epoch = 0

        groups = _groups(self.hidden_dim)
        self.context_encoder = nn.Sequential(
            nn.Conv2d(12, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
        )
        self.semantic_proj = nn.Sequential(
            nn.Conv2d(int(semantic_channels), self.hidden_dim, 1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Conv2d(10, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
        )

        self.family_embedding = nn.Embedding(self.family_count, int(metadata_dim))
        self.action_embedding = nn.Embedding(self.action_count, int(metadata_dim))
        self.metadata_proj = nn.Sequential(
            nn.Linear(2 * int(metadata_dim) + 4, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.candidate_fuse = nn.Sequential(
            nn.Conv2d(3 * self.hidden_dim, self.hidden_dim, 1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
            nn.Dropout2d(self.dropout),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
        )

        # Dense counterfactual guidance used by the prompt explorer.
        self.dense_utility_head = nn.Conv2d(self.hidden_dim, 1, 1)
        self.dense_logvar_head = nn.Conv2d(self.hidden_dim, 1, 1)
        self.dense_harm_head = nn.Conv2d(self.hidden_dim, 1, 1)
        self.prompt_head = nn.Sequential(
            nn.Conv2d(self.hidden_dim + 5, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, 3, 1),
        )

        # Dynamic family selector.  It observes both global image/Base context
        # and pooled candidate evidence for each family.
        self.family_selector = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(set_heads),
            dim_feedforward=int(set_ffn_dim),
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(layer, num_layers=int(set_layers))
        self.preserve_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.region_context_proj = nn.Sequential(
            nn.Conv2d(self.hidden_dim + 3, self.hidden_dim, 1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
        )
        self.route_head = nn.Linear(self.hidden_dim, 1)
        # V527 separates whether a region should be edited from which candidate
        # should be used when editing.  The conditional score is the only
        # candidate-ranking score in the V527 path.
        self.editability_head = nn.Linear(self.hidden_dim, 1)
        self.conditional_score_head = nn.Linear(self.hidden_dim, 1)
        self.region_utility_head = nn.Linear(self.hidden_dim, 1)
        self.region_logvar_head = nn.Linear(self.hidden_dim, 1)
        self.region_harm_head = nn.Linear(self.hidden_dim, 1)

        # V528 outcome heads.  The global Base confusion distribution is a
        # four-way simplex (TP/FP/FN/TN).  The two action logits predict the
        # correctness of actual Add and Remove pixels; the four outcome masses
        # are then derived by conservation rather than freely regressed.
        self.v528_base_confusion_head = nn.Linear(self.hidden_dim, 4)
        self.v528_outcome_correctness_head = nn.Linear(self.hidden_dim, 2)
        self.v528_outcome_logvar_head = nn.Linear(self.hidden_dim, 1)

        # V529 calibration-first heads.  Base foreground/background masses are
        # observable from the factual mask, so only FP-within-foreground and
        # FN-within-background rates are predicted.  Candidate outcome quality
        # is predicted densely on the actual Add/Remove support before exact
        # region-wise conserved masses and Dice gains are reconstructed.
        self.v529_base_error_head = nn.Linear(self.hidden_dim, 2)
        self.v529_outcome_pixel_head = nn.Conv2d(self.hidden_dim, 2, 1)
        self.v529_outcome_logvar_head = nn.Linear(self.hidden_dim, 1)

        # V530 probability-calibrated dense heads.  The Base head receives one
        # shared full-resolution feature map plus factual geometry.  Candidate
        # correctness uses a coarse semantic logit followed by a lightweight
        # full-resolution residual that can resolve thin Remove boundaries.
        v530_base_hidden = max(16, self.hidden_dim // 2)
        self.v530_base_error_pixel_head = nn.Sequential(
            nn.Conv2d(self.hidden_dim + 8, v530_base_hidden, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(v530_base_hidden), v530_base_hidden),
            nn.GELU(),
            nn.Conv2d(v530_base_hidden, 2, 1),
        )
        self.v530_outcome_coarse_head = nn.Conv2d(self.hidden_dim, 2, 1)
        self.v530_outcome_refine_head = nn.Sequential(
            nn.Conv2d(14, 16, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(16), 16),
            nn.GELU(),
            nn.Conv2d(16, 8, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(8), 8),
            nn.GELU(),
            nn.Conv2d(8, 2, 1),
        )
        self.v530_outcome_logvar_head = nn.Linear(self.hidden_dim, 1)

        # One full-resolution refiner is used after region routing.  Its output
        # is a Base-vs-selected-candidate gate, not a free segmentation mask.
        self.refiner = nn.Sequential(
            nn.Conv2d(12, 32, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(32), 32),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(32), 32),
            nn.GELU(),
            nn.Conv2d(32, 1, 1),
        )

        # The case verifier predicts the outcome of the complete composed mask.
        # It sees global image/Base context and aggregate region-action evidence.
        self.case_verifier = nn.Sequential(
            nn.Linear(self.hidden_dim + 8, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 3),
        )

        nn.init.zeros_(self.dense_utility_head.weight)
        nn.init.zeros_(self.dense_utility_head.bias)
        nn.init.zeros_(self.dense_logvar_head.weight)
        nn.init.constant_(self.dense_logvar_head.bias, -1.0)
        nn.init.zeros_(self.dense_harm_head.weight)
        nn.init.zeros_(self.dense_harm_head.bias)
        nn.init.zeros_(self.route_head.weight)
        nn.init.constant_(self.route_head.bias, -0.10)
        nn.init.zeros_(self.editability_head.weight)
        # Preserve-first initialization.  sigmoid(-2.2) is about 0.10.
        nn.init.constant_(self.editability_head.bias, -2.20)
        nn.init.zeros_(self.conditional_score_head.weight)
        nn.init.zeros_(self.conditional_score_head.bias)
        nn.init.zeros_(self.region_utility_head.weight)
        nn.init.zeros_(self.region_utility_head.bias)
        nn.init.zeros_(self.region_logvar_head.weight)
        nn.init.constant_(self.region_logvar_head.bias, self.logvar_init)
        nn.init.zeros_(self.region_harm_head.weight)
        nn.init.zeros_(self.region_harm_head.bias)
        nn.init.zeros_(self.v528_base_confusion_head.weight)
        with torch.no_grad():
            self.v528_base_confusion_head.bias.copy_(
                torch.tensor([-2.0, -5.0, -5.0, 0.0])
            )
        nn.init.zeros_(self.v528_outcome_correctness_head.weight)
        nn.init.zeros_(self.v528_outcome_correctness_head.bias)
        nn.init.zeros_(self.v528_outcome_logvar_head.weight)
        nn.init.constant_(self.v528_outcome_logvar_head.bias, self.logvar_init)
        nn.init.zeros_(self.v529_base_error_head.weight)
        nn.init.zeros_(self.v529_base_error_head.bias)
        nn.init.zeros_(self.v529_outcome_pixel_head.weight)
        nn.init.zeros_(self.v529_outcome_pixel_head.bias)
        nn.init.zeros_(self.v529_outcome_logvar_head.weight)
        nn.init.constant_(self.v529_outcome_logvar_head.bias, self.logvar_init)

        def _prior_logit(probability: float) -> float:
            return math.log(probability / (1.0 - probability))

        # Prior-aware initialization: V529 started every rare error at 50%.
        # V530 starts close to factual BUSI-scale rates and lets training
        # calibrate them without a huge first-epoch bias.
        nn.init.zeros_(self.v530_base_error_pixel_head[-1].weight)
        with torch.no_grad():
            self.v530_base_error_pixel_head[-1].bias.copy_(torch.tensor([
                _prior_logit(self.v530_base_fp_prior),
                _prior_logit(self.v530_base_fn_prior),
            ]))
        nn.init.zeros_(self.v530_outcome_coarse_head.weight)
        with torch.no_grad():
            self.v530_outcome_coarse_head.bias.copy_(torch.tensor([
                _prior_logit(self.v530_add_fix_prior),
                _prior_logit(self.v530_remove_fix_prior),
            ]))
        nn.init.zeros_(self.v530_outcome_refine_head[-1].weight)
        nn.init.zeros_(self.v530_outcome_refine_head[-1].bias)
        nn.init.zeros_(self.v530_outcome_logvar_head.weight)
        nn.init.constant_(self.v530_outcome_logvar_head.bias, self.logvar_init)
        nn.init.zeros_(self.refiner[-1].weight)
        nn.init.constant_(self.refiner[-1].bias, -1.0)
        nn.init.zeros_(self.case_verifier[-1].weight)
        with torch.no_grad():
            # Preserve-first initialization: predicted gain is zero, uncertainty
            # starts on the same ~1e-3/1e-2 scale as regional Dice utility, and
            # harm begins neutral instead of forcing every case to reject.
            case_bias = (
                torch.tensor([0.0, self.logvar_init, 0.0])
                if self.utility_aligned_score
                else torch.tensor([0.0, -1.0, 1.0])
            )
            self.case_verifier[-1].bias.copy_(case_bias)

        if (
            self.factorized_ranker
            or self.outcome_composer
            or self.v529_calibration_first_outcome
            or self.v530_probability_calibrated_outcome
        ):
            # Stage-A trains only M2.  The case verifier belongs to a later M3
            # stage and must remain untouched by the M2 checkpoint.
            for parameter in self.case_verifier.parameters():
                parameter.requires_grad_(False)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def _semantic_map(
        self,
        semantic_map: torch.Tensor | None,
        *,
        size: Tuple[int, int],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(semantic_map, torch.Tensor):
            return reference.new_zeros(reference.shape[0], self.hidden_dim, *size)
        semantic = _resize(semantic_map, size)
        return self.semantic_proj(semantic.to(dtype=reference.dtype))

    def _metadata(
        self,
        *,
        family_ids: torch.Tensor,
        action_ids: torch.Tensor,
        dose_values: torch.Tensor,
        radius_values: torch.Tensor,
        edit_fraction: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        family = family_ids[1:].long().clamp(0, self.family_count - 1)
        action = action_ids[1:].long().clamp(0, self.action_count - 1)
        dose = dose_values[1:].to(dtype=dtype)
        radius = radius_values[1:].to(dtype=dtype)
        dose_norm = dose / dose.max().clamp_min(1.0)
        radius_norm = radius / radius.max().clamp_min(1.0)
        log_dose = torch.log2(dose.clamp_min(1.0e-3)) / 3.0
        fixed = torch.cat(
            [
                self.family_embedding(family),
                self.action_embedding(action),
                log_dose[:, None],
                dose_norm[:, None],
                radius_norm[:, None],
            ],
            dim=1,
        )
        b = edit_fraction.shape[0]
        fixed = fixed[None].expand(b, -1, -1)
        features = torch.cat([fixed, edit_fraction[..., None]], dim=2)
        return self.metadata_proj(features)

    def forward(
        self,
        *,
        image: torch.Tensor,
        c0_prob: torch.Tensor,
        candidate_probs: torch.Tensor,
        supports: torch.Tensor,
        candidate_causes: torch.Tensor,
        family_ids: torch.Tensor,
        action_ids: torch.Tensor,
        dose_values: torch.Tensor,
        radius_values: torch.Tensor,
        deploy_mask: torch.Tensor,
        semantic_map: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        c0 = _as_b1hw(c0_prob).clamp(EPS, 1.0 - EPS)
        if candidate_probs.ndim != 4:
            raise ValueError("candidate_probs must be [B,K,H,W]")
        b, k, h, w = candidate_probs.shape
        if k < 2 or k > self.max_candidates:
            raise ValueError(f"V524 requires 2..{self.max_candidates} candidates, got {k}")
        if supports.shape != candidate_probs.shape or candidate_causes.shape != candidate_probs.shape:
            raise ValueError("V524 supports/causes must match candidate_probs")
        for tensor in (family_ids, action_ids, dose_values, radius_values, deploy_mask):
            if tensor.ndim != 1 or tensor.numel() != k:
                raise ValueError("V524 candidate metadata must have K entries")

        nonbase = candidate_probs[:, 1:].clamp(EPS, 1.0 - EPS)
        support = supports[:, 1:].clamp(0.0, 1.0)
        cause = candidate_causes[:, 1:].clamp(0.0, 1.0)
        n = nonbase.shape[1]
        family_nonbase = family_ids[1:].to(c0.device).long().clamp(0, self.family_count - 1)
        deploy_nonbase = deploy_mask[1:].to(c0.device, dtype=torch.bool)
        current_bank = c0.expand(-1, n, -1, -1)
        signed = nonbase - current_bank
        abs_edit = signed.abs()
        hard_changed = (nonbase >= 0.5) != (current_bank >= 0.5)
        valid = (
            deploy_nonbase[None, :, None, None]
            & (support > self.support_floor)
            & (abs_edit > self.edit_epsilon)
            & hard_changed
        )

        low_size = (
            max(1, (h + self.score_stride - 1) // self.score_stride),
            max(1, (w + self.score_stride - 1) // self.score_stride),
        )
        c0_low = _resize(c0, low_size)
        nonbase_low = _resize(nonbase.reshape(b * n, 1, h, w), low_size).reshape(b, n, *low_size)
        support_low = _resize(support.reshape(b * n, 1, h, w), low_size).reshape(b, n, *low_size)
        cause_low = _resize(cause.reshape(b * n, 1, h, w), low_size).reshape(b, n, *low_size)
        signed_low = nonbase_low - c0_low.expand(-1, n, -1, -1)
        abs_low = signed_low.abs()
        gray_low, edge_low = _gray_edge(image, low_size)
        entropy_low = _entropy(c0_low)
        boundary_low = _soft_boundary(c0_low)
        support_union = support_low.amax(dim=1, keepdim=True)
        cause_union = cause_low.amax(dim=1, keepdim=True)
        max_edit = abs_low.amax(dim=1, keepdim=True)
        mean_edit = abs_low.mean(dim=1, keepdim=True)
        candidate_disagreement = (nonbase_low >= 0.5).float().var(dim=1, keepdim=True, unbiased=False)

        context_input = torch.cat(
            [
                c0_low,
                1.0 - c0_low,
                entropy_low,
                boundary_low,
                gray_low,
                edge_low,
                max_edit,
                mean_edit,
                support_union,
                cause_union,
                candidate_disagreement,
                (max_edit * candidate_disagreement).clamp(0.0, 1.0),
            ],
            dim=1,
        )
        shared = self.context_encoder(context_input)
        shared = shared + self._semantic_map(semantic_map, size=low_size, reference=c0)

        candidate_input = torch.stack(
            [
                nonbase_low,
                c0_low.expand(-1, n, -1, -1),
                signed_low,
                abs_low,
                support_low,
                cause_low,
                boundary_low.expand(-1, n, -1, -1),
                entropy_low.expand(-1, n, -1, -1),
                gray_low.expand(-1, n, -1, -1),
                edge_low.expand(-1, n, -1, -1),
            ],
            dim=2,
        ).reshape(b * n, 10, *low_size)
        candidate_feature = self.candidate_encoder(candidate_input)
        edit_fraction = abs_edit.flatten(2).mean(dim=2)
        metadata = self._metadata(
            family_ids=family_ids.to(c0.device),
            action_ids=action_ids.to(c0.device),
            dose_values=dose_values.to(c0.device),
            radius_values=radius_values.to(c0.device),
            edit_fraction=edit_fraction,
            dtype=c0.dtype,
        )
        shared_bank = shared[:, None].expand(-1, n, -1, -1, -1).reshape(b * n, self.hidden_dim, *low_size)
        metadata_bank = metadata.reshape(b * n, self.hidden_dim, 1, 1).expand(-1, -1, *low_size)
        fused = self.candidate_fuse(torch.cat([shared_bank, candidate_feature, metadata_bank], dim=1))
        fused_bn = fused.reshape(b, n, self.hidden_dim, *low_size)

        dense_mu_low = self.dense_utility_head(fused).reshape(b, n, *low_size)
        dense_logvar_low = self.dense_logvar_head(fused).reshape(b, n, *low_size).clamp(-6.0, 3.0)
        dense_harm_low = self.dense_harm_head(fused).reshape(b, n, *low_size)
        dense_sigma_low = torch.exp(0.5 * dense_logvar_low)
        dense_lcb_low = (
            dense_mu_low
            - self.lcb_beta * dense_sigma_low
            - self.harm_penalty * torch.sigmoid(dense_harm_low)
        )
        valid_low = _resize(valid.float().reshape(b * n, 1, h, w), low_size, mode="nearest").reshape(b, n, *low_size) > 0.5
        floor = -1.0e4 if c0.dtype in (torch.float16, torch.bfloat16) else -1.0e9
        masked_lcb = dense_lcb_low.masked_fill(~valid_low, floor)

        add_low = valid_low & (c0_low.expand(-1, n, -1, -1) < 0.5) & (nonbase_low >= 0.5)
        remove_low = valid_low & (c0_low.expand(-1, n, -1, -1) >= 0.5) & (nonbase_low < 0.5)
        add_map = masked_lcb.masked_fill(~add_low, floor).amax(dim=1, keepdim=True)
        remove_map = masked_lcb.masked_fill(~remove_low, floor).amax(dim=1, keepdim=True)
        add_map = torch.where(torch.isfinite(add_map) & (add_map > floor / 2), add_map, torch.zeros_like(add_map))
        remove_map = torch.where(torch.isfinite(remove_map) & (remove_map > floor / 2), remove_map, torch.zeros_like(remove_map))
        prompt_input = torch.cat(
            [
                shared,
                add_map,
                remove_map,
                candidate_disagreement,
                edge_low,
                boundary_low,
            ],
            dim=1,
        )
        prompt_logits_low = self.prompt_head(prompt_input)
        prompt_logits = _resize(prompt_logits_low, (h, w))
        prompt_prob = torch.sigmoid(prompt_logits)

        # Dynamic family evidence.  Every family receives a token even if no
        # candidate of that family is currently present; absent families are
        # masked before softmax.
        global_context = F.adaptive_avg_pool2d(shared, 1).flatten(1)
        candidate_global = F.adaptive_avg_pool2d(fused, 1).reshape(b, n, self.hidden_dim)
        family_tokens = c0.new_zeros(b, self.family_count, self.hidden_dim)
        family_counts = c0.new_zeros(b, self.family_count, 1)
        for family_index in range(self.family_count):
            mask = (family_nonbase == family_index).to(c0.dtype)[None, :, None]
            family_tokens[:, family_index] = (candidate_global * mask).sum(dim=1)
            count = mask.sum(dim=1).clamp_min(1.0)
            family_tokens[:, family_index] = family_tokens[:, family_index] / count
            family_counts[:, family_index] = mask.sum(dim=1)
        family_context = global_context[:, None].expand(-1, self.family_count, -1)
        family_logits = self.family_selector(torch.cat([family_context, family_tokens], dim=2)).squeeze(-1)
        family_present = family_counts[:, :, 0] > 0
        family_logits = family_logits.masked_fill(~family_present, floor)
        family_prob = F.softmax(family_logits, dim=1)
        family_candidate_prob = family_prob[:, family_nonbase].clamp_min(EPS)

        grid = self.region_grid_size
        region_prompt_context = (
            torch.zeros_like(prompt_logits_low)
            if (
                self.outcome_composer
                or self.v529_calibration_first_outcome
                or self.v530_probability_calibrated_outcome
            )
            else torch.sigmoid(prompt_logits_low)
        )
        region_context = F.adaptive_avg_pool2d(
            self.region_context_proj(torch.cat([shared, region_prompt_context], dim=1)),
            (grid, grid),
        )
        region_context_tokens = region_context.flatten(2).transpose(1, 2)  # [B,R,C]
        region_candidate = F.adaptive_avg_pool2d(fused, (grid, grid)).reshape(
            b, n, self.hidden_dim, grid, grid
        ).permute(0, 3, 4, 1, 2).reshape(b, grid * grid, n, self.hidden_dim)
        region_candidate = region_candidate + region_context_tokens[:, :, None, :] + metadata[:, None]
        preserve = self.preserve_token.expand(b * grid * grid, -1, -1)
        set_input = region_candidate.reshape(b * grid * grid, n, self.hidden_dim)
        set_input = torch.cat([preserve, set_input], dim=1)
        set_output = self.set_encoder(set_input).reshape(b, grid * grid, n + 1, self.hidden_dim)

        route_raw = self.route_head(set_output).squeeze(-1)
        editability_logit = self.editability_head(set_output[:, :, 0]).squeeze(-1)
        conditional_score = self.conditional_score_head(
            set_output[:, :, 1:]
        ).squeeze(-1)
        region_mu = self.region_utility_head(set_output[:, :, 1:]).squeeze(-1)
        region_logvar = self.region_logvar_head(set_output[:, :, 1:]).squeeze(-1).clamp(self.logvar_min, self.logvar_max)
        region_sigma = torch.exp(0.5 * region_logvar)
        region_harm_logit = self.region_harm_head(set_output[:, :, 1:]).squeeze(-1)

        v528_base_confusion_logit = self.v528_base_confusion_head(global_context)
        v528_base_confusion_prob = F.softmax(v528_base_confusion_logit, dim=1)
        v528_outcome_correctness_logit = self.v528_outcome_correctness_head(
            set_output[:, :, 1:]
        )
        v528_outcome_logvar = self.v528_outcome_logvar_head(
            set_output[:, :, 1:]
        ).squeeze(-1).clamp(self.logvar_min, self.logvar_max)
        v529_base_error_logit = self.v529_base_error_head(global_context)
        v529_outcome_pixel_logit_low = self.v529_outcome_pixel_head(fused).reshape(
            b, n, 2, *low_size
        )
        v529_outcome_logvar = self.v529_outcome_logvar_head(
            set_output[:, :, 1:]
        ).squeeze(-1).clamp(self.logvar_min, self.logvar_max)

        # V530 full-resolution factual and candidate-specific calibration maps.
        gray_full_outcome, edge_full_outcome = _gray_edge(image, (h, w))
        entropy_full_outcome = _entropy(c0)
        boundary_full_outcome = _soft_boundary(c0)
        shared_full_outcome = _resize(shared, (h, w))
        support_union_full = support.amax(dim=1, keepdim=True)
        cause_union_full = cause.amax(dim=1, keepdim=True)
        v530_base_error_input = torch.cat(
            [
                shared_full_outcome,
                c0,
                1.0 - c0,
                entropy_full_outcome,
                boundary_full_outcome,
                gray_full_outcome,
                edge_full_outcome,
                support_union_full,
                cause_union_full,
            ],
            dim=1,
        )
        v530_base_error_pixel_logit = self.v530_base_error_pixel_head(
            v530_base_error_input
        )
        v530_outcome_coarse_low = self.v530_outcome_coarse_head(fused).reshape(
            b, n, 2, *low_size
        )
        v530_outcome_coarse_full = _resize(
            v530_outcome_coarse_low.reshape(b * n * 2, 1, *low_size),
            (h, w),
        ).reshape(b, n, 2, h, w)
        add_full_geometry = valid & (current_bank < 0.5) & (nonbase >= 0.5)
        remove_full_geometry = valid & (current_bank >= 0.5) & (nonbase < 0.5)
        v530_refine_input = torch.cat(
            [
                v530_outcome_coarse_full.reshape(b * n, 2, h, w),
                nonbase.reshape(b * n, 1, h, w),
                current_bank.reshape(b * n, 1, h, w),
                signed.reshape(b * n, 1, h, w),
                abs_edit.reshape(b * n, 1, h, w),
                support.reshape(b * n, 1, h, w),
                cause.reshape(b * n, 1, h, w),
                boundary_full_outcome[:, None].expand(-1, n, -1, -1, -1).reshape(b * n, 1, h, w),
                edge_full_outcome[:, None].expand(-1, n, -1, -1, -1).reshape(b * n, 1, h, w),
                gray_full_outcome[:, None].expand(-1, n, -1, -1, -1).reshape(b * n, 1, h, w),
                entropy_full_outcome[:, None].expand(-1, n, -1, -1, -1).reshape(b * n, 1, h, w),
                add_full_geometry.reshape(b * n, 1, h, w).to(c0.dtype),
                remove_full_geometry.reshape(b * n, 1, h, w).to(c0.dtype),
            ],
            dim=1,
        )
        v530_outcome_pixel_logit = (
            v530_outcome_coarse_full.reshape(b * n, 2, h, w)
            + self.v530_outcome_refine_head(v530_refine_input)
        ).reshape(b, n, 2, h, w)
        v530_outcome_logvar = self.v530_outcome_logvar_head(
            set_output[:, :, 1:]
        ).squeeze(-1).clamp(self.logvar_min, self.logvar_max)

        region_valid = F.adaptive_max_pool2d(valid.float().reshape(b * n, 1, h, w), (grid, grid)).reshape(
            b, n, grid, grid
        ).permute(0, 2, 3, 1).reshape(b, grid * grid, n) > 0.5
        prompt_grid = F.adaptive_avg_pool2d(prompt_prob, (grid, grid)).flatten(2).transpose(1, 2)
        add_fraction = F.adaptive_avg_pool2d(add_low.float().reshape(b * n, 1, *low_size), (grid, grid)).reshape(
            b, n, grid, grid
        ).permute(0, 2, 3, 1).reshape(b, grid * grid, n)
        remove_fraction = F.adaptive_avg_pool2d(remove_low.float().reshape(b * n, 1, *low_size), (grid, grid)).reshape(
            b, n, grid, grid
        ).permute(0, 2, 3, 1).reshape(b, grid * grid, n)
        direction_prompt = (
            add_fraction * prompt_grid[:, :, 0:1]
            + remove_fraction * prompt_grid[:, :, 1:2]
            + (add_fraction + remove_fraction).clamp(0.0, 1.0) * prompt_grid[:, :, 2:3]
        ) / (add_fraction + remove_fraction).clamp_min(EPS)
        direction_prompt = direction_prompt.clamp(0.0, 1.0)

        geometric_region_valid = region_valid.clone()
        family_term = self.family_log_weight * torch.log(
            family_candidate_prob[:, None].clamp_min(EPS)
        )
        route_residual = self.route_residual_scale * torch.tanh(
            route_raw[:, :, 1:]
        )

        # At inference legacy V524/V526 may sparsify families.  V527 never lets
        # a learned family or prompt prediction delete a realizable action.
        family_active = torch.ones_like(family_prob, dtype=torch.bool)
        if (
            not self.oracle_preserving_validity
            and not self.training
            and self.top_family_count < self.family_count
        ):
            top_family = torch.topk(family_prob, self.top_family_count, dim=1).indices
            family_active = torch.zeros_like(family_prob, dtype=torch.bool).scatter_(1, top_family, True)
        candidate_family_active = family_active[:, family_nonbase][:, None]
        if not self.oracle_preserving_validity:
            region_valid = region_valid & candidate_family_active

        causal_valid = torch.ones_like(region_valid)
        if (
            not self.oracle_preserving_validity
            and self.causal_hard_mask
            and self.current_epoch >= self.causal_mask_start_epoch
        ):
            causal_valid = direction_prompt >= self.causal_mask_threshold
            region_valid = region_valid & causal_valid

        # Full-resolution action geometry for V528.  These masses are available
        # at deployment because they depend only on Base and the real candidate.
        region_ids = F.interpolate(
            torch.arange(grid * grid, device=c0.device, dtype=c0.dtype).reshape(1, 1, grid, grid),
            size=(h, w),
            mode="nearest",
        )[0, 0].long()
        add_full = add_full_geometry
        remove_full = remove_full_geometry
        v528_add_fraction = c0.new_zeros(b, grid * grid, n)
        v528_remove_fraction = c0.new_zeros(b, grid * grid, n)
        pixel_count = float(max(h * w, 1))
        for region_index in range(grid * grid):
            spatial = (region_ids == region_index)[None, None]
            v528_add_fraction[:, region_index] = (
                add_full & spatial
            ).flatten(2).sum(dim=2).to(c0.dtype) / pixel_count
            v528_remove_fraction[:, region_index] = (
                remove_full & spatial
            ).flatten(2).sum(dim=2).to(c0.dtype) / pixel_count

        v528_correctness_prob = torch.sigmoid(v528_outcome_correctness_logit)
        v528_add_correct_prob = v528_correctness_prob[..., 0]
        v528_remove_correct_prob = v528_correctness_prob[..., 1]
        v528_fn_fix = v528_add_fraction * v528_add_correct_prob
        v528_tn_harm = v528_add_fraction * (1.0 - v528_add_correct_prob)
        v528_fp_fix = v528_remove_fraction * v528_remove_correct_prob
        v528_tp_harm = v528_remove_fraction * (1.0 - v528_remove_correct_prob)

        v528_base_tp = v528_base_confusion_prob[:, 0, None, None]
        v528_base_fp = v528_base_confusion_prob[:, 1, None, None]
        v528_base_fn = v528_base_confusion_prob[:, 2, None, None]
        v528_base_dice = (
            2.0 * v528_base_tp + EPS
        ) / (2.0 * v528_base_tp + v528_base_fp + v528_base_fn + EPS)
        v528_new_tp = (v528_base_tp + v528_fn_fix - v528_tp_harm).clamp_min(0.0)
        v528_new_fp = (v528_base_fp + v528_tn_harm - v528_fp_fix).clamp_min(0.0)
        v528_new_fn = (v528_base_fn - v528_fn_fix + v528_tp_harm).clamp_min(0.0)
        v528_candidate_dice = (
            2.0 * v528_new_tp + EPS
        ) / (2.0 * v528_new_tp + v528_new_fp + v528_new_fn + EPS)
        v528_predicted_gain = v528_candidate_dice - v528_base_dice
        v528_sigma = torch.exp(0.5 * v528_outcome_logvar)
        v528_gain_lcb = v528_predicted_gain - self.outcome_lcb_beta * v528_sigma
        v528_fix_mass = v528_fn_fix + v528_fp_fix
        v528_harm_mass = v528_tn_harm + v528_tp_harm
        v528_harm_fraction = v528_harm_mass / (
            v528_fix_mass + v528_harm_mass
        ).clamp_min(EPS)

        base_hard_full = c0 >= 0.5
        base_fg_fraction = base_hard_full.flatten(1).float().mean(dim=1).to(c0.dtype)
        base_bg_fraction = 1.0 - base_fg_fraction

        # V530 calibrated dense conserved outcomes.  Base error probabilities
        # are integrated only over their factual foreground/background domains;
        # candidate correctness is integrated only over actual changed pixels.
        v530_base_error_prob = torch.sigmoid(v530_base_error_pixel_logit)
        v530_base_fp = (
            v530_base_error_prob[:, 0:1] * base_hard_full.to(c0.dtype)
        ).flatten(1).mean(dim=1)
        v530_base_fn = (
            v530_base_error_prob[:, 1:2] * (~base_hard_full).to(c0.dtype)
        ).flatten(1).mean(dim=1)
        v530_base_tp = (base_fg_fraction - v530_base_fp).clamp_min(0.0)
        v530_base_dice = (
            2.0 * v530_base_tp + EPS
        ) / (2.0 * v530_base_tp + v530_base_fp + v530_base_fn + EPS)
        v530_pixel_prob = torch.sigmoid(v530_outcome_pixel_logit)
        v530_add_fix_prob = v530_pixel_prob[:, :, 0]
        v530_remove_fix_prob = v530_pixel_prob[:, :, 1]
        v530_fn_fix = c0.new_zeros(b, grid * grid, n)
        v530_fp_fix = c0.new_zeros(b, grid * grid, n)
        v530_tn_harm = c0.new_zeros(b, grid * grid, n)
        v530_tp_harm = c0.new_zeros(b, grid * grid, n)
        for region_index in range(grid * grid):
            spatial = (region_ids == region_index)[None, None]
            add_region = add_full & spatial
            remove_region = remove_full & spatial
            add_mass = add_region.flatten(2).sum(dim=2).to(c0.dtype) / pixel_count
            remove_mass = remove_region.flatten(2).sum(dim=2).to(c0.dtype) / pixel_count
            fn_fix = (
                v530_add_fix_prob * add_region.to(c0.dtype)
            ).flatten(2).sum(dim=2) / pixel_count
            fp_fix = (
                v530_remove_fix_prob * remove_region.to(c0.dtype)
            ).flatten(2).sum(dim=2) / pixel_count
            v530_fn_fix[:, region_index] = fn_fix
            v530_tn_harm[:, region_index] = (add_mass - fn_fix).clamp_min(0.0)
            v530_fp_fix[:, region_index] = fp_fix
            v530_tp_harm[:, region_index] = (remove_mass - fp_fix).clamp_min(0.0)
        v530_base_tp_ = v530_base_tp[:, None, None]
        v530_base_fp_ = v530_base_fp[:, None, None]
        v530_base_fn_ = v530_base_fn[:, None, None]
        v530_new_tp = (v530_base_tp_ + v530_fn_fix - v530_tp_harm).clamp_min(0.0)
        v530_new_fp = (v530_base_fp_ + v530_tn_harm - v530_fp_fix).clamp_min(0.0)
        v530_new_fn = (v530_base_fn_ - v530_fn_fix + v530_tp_harm).clamp_min(0.0)
        v530_candidate_dice = (
            2.0 * v530_new_tp + EPS
        ) / (2.0 * v530_new_tp + v530_new_fp + v530_new_fn + EPS)
        v530_predicted_gain = v530_candidate_dice - v530_base_dice[:, None, None]
        v530_sigma_normalized = torch.exp(0.5 * v530_outcome_logvar)
        v530_sigma = self.v530_utility_scale * v530_sigma_normalized
        v530_normalized_score = v530_predicted_gain / self.v530_utility_scale
        if self.v530_uncertainty_enabled:
            v530_normalized_score = (
                v530_normalized_score
                - self.v530_lcb_beta * v530_sigma_normalized.detach()
            )
        v530_fix_mass = v530_fn_fix + v530_fp_fix
        v530_harm_mass = v530_tn_harm + v530_tp_harm
        v530_harm_fraction = v530_harm_mass / (
            v530_fix_mass + v530_harm_mass
        ).clamp_min(EPS)

        # V529 dense conserved outcomes.  The two pixel probabilities are only
        # integrated over pixels that the real candidate actually changes.
        v529_pixel_logit = _resize(
            v529_outcome_pixel_logit_low.reshape(b * n * 2, 1, *low_size),
            (h, w),
        ).reshape(b, n, 2, h, w)
        v529_pixel_prob = torch.sigmoid(v529_pixel_logit)
        v529_add_fix_prob = v529_pixel_prob[:, :, 0]
        v529_remove_fix_prob = v529_pixel_prob[:, :, 1]

        base_hard_full = c0 >= 0.5
        base_fg_fraction = base_hard_full.flatten(1).float().mean(dim=1).to(c0.dtype)
        base_bg_fraction = 1.0 - base_fg_fraction
        v529_base_error_prob = torch.sigmoid(v529_base_error_logit)
        v529_base_fp = base_fg_fraction * v529_base_error_prob[:, 0]
        v529_base_tp = (base_fg_fraction - v529_base_fp).clamp_min(0.0)
        v529_base_fn = base_bg_fraction * v529_base_error_prob[:, 1]
        v529_base_dice = (
            2.0 * v529_base_tp + EPS
        ) / (2.0 * v529_base_tp + v529_base_fp + v529_base_fn + EPS)

        v529_fn_fix = c0.new_zeros(b, grid * grid, n)
        v529_fp_fix = c0.new_zeros(b, grid * grid, n)
        v529_tn_harm = c0.new_zeros(b, grid * grid, n)
        v529_tp_harm = c0.new_zeros(b, grid * grid, n)
        for region_index in range(grid * grid):
            spatial = (region_ids == region_index)[None, None]
            add_region = add_full & spatial
            remove_region = remove_full & spatial
            add_mass = add_region.flatten(2).sum(dim=2).to(c0.dtype) / pixel_count
            remove_mass = remove_region.flatten(2).sum(dim=2).to(c0.dtype) / pixel_count
            fn_fix = (
                v529_add_fix_prob * add_region.to(c0.dtype)
            ).flatten(2).sum(dim=2) / pixel_count
            fp_fix = (
                v529_remove_fix_prob * remove_region.to(c0.dtype)
            ).flatten(2).sum(dim=2) / pixel_count
            v529_fn_fix[:, region_index] = fn_fix
            v529_tn_harm[:, region_index] = (add_mass - fn_fix).clamp_min(0.0)
            v529_fp_fix[:, region_index] = fp_fix
            v529_tp_harm[:, region_index] = (remove_mass - fp_fix).clamp_min(0.0)

        v529_base_tp_ = v529_base_tp[:, None, None]
        v529_base_fp_ = v529_base_fp[:, None, None]
        v529_base_fn_ = v529_base_fn[:, None, None]
        v529_new_tp = (v529_base_tp_ + v529_fn_fix - v529_tp_harm).clamp_min(0.0)
        v529_new_fp = (v529_base_fp_ + v529_tn_harm - v529_fp_fix).clamp_min(0.0)
        v529_new_fn = (v529_base_fn_ - v529_fn_fix + v529_tp_harm).clamp_min(0.0)
        v529_candidate_dice = (
            2.0 * v529_new_tp + EPS
        ) / (2.0 * v529_new_tp + v529_new_fp + v529_new_fn + EPS)
        v529_predicted_gain = v529_candidate_dice - v529_base_dice[:, None, None]
        v529_sigma_normalized = torch.exp(0.5 * v529_outcome_logvar)
        v529_sigma = self.v529_utility_scale * v529_sigma_normalized
        v529_normalized_score = v529_predicted_gain / self.v529_utility_scale
        if self.v529_uncertainty_enabled:
            # Stop-gradient prevents selector objectives from manipulating the
            # variance head instead of learning the counterfactual mean.
            v529_normalized_score = (
                v529_normalized_score
                - self.v529_lcb_beta * v529_sigma_normalized.detach()
            )
        v529_fix_mass = v529_fn_fix + v529_fp_fix
        v529_harm_mass = v529_tn_harm + v529_tp_harm
        v529_harm_fraction = v529_harm_mass / (
            v529_fix_mass + v529_harm_mass
        ).clamp_min(EPS)

        if self.v530_probability_calibrated_outcome:
            candidate_score = v530_normalized_score.masked_fill(~region_valid, floor)
            flat_score = candidate_score.reshape(b, -1)
            preserve_logit = flat_score.new_zeros(b, 1)
            global_soft = F.softmax(
                torch.cat([preserve_logit, flat_score], dim=1)
                / self.v530_route_temperature,
                dim=1,
            )
            soft_candidate = global_soft[:, 1:].reshape(b, grid * grid, n)
            soft_preserve = (1.0 - soft_candidate.sum(dim=2)).clamp(0.0, 1.0)
            soft_route = torch.cat([soft_preserve[..., None], soft_candidate], dim=2)
            action_score = torch.log(soft_route.clamp_min(EPS))
            flat_best_score, flat_best_index = flat_score.max(dim=1)
            selected_region_index = flat_best_index // n
            selected_candidate_index = flat_best_index % n
            selected_gain = v530_predicted_gain.reshape(b, -1).gather(
                1, flat_best_index[:, None]
            )[:, 0]
            selected_harm = v530_harm_fraction.reshape(b, -1).gather(
                1, flat_best_index[:, None]
            )[:, 0]
            selected_valid_action = region_valid.reshape(b, -1).gather(
                1, flat_best_index[:, None]
            )[:, 0]
            enable_execution = self.v530_selector_enabled and self.v530_execution_enabled
            if enable_execution:
                case_accept = (
                    selected_valid_action
                    & (selected_gain > self.v530_min_gain)
                    & (selected_harm <= self.v530_max_harm)
                )
            else:
                case_accept = torch.zeros_like(selected_valid_action)
            best_index = torch.zeros(b, grid * grid, device=c0.device, dtype=torch.long)
            accept = torch.zeros(b, grid * grid, device=c0.device, dtype=torch.bool)
            batch_ids = torch.arange(b, device=c0.device)
            best_index[batch_ids, selected_region_index] = selected_candidate_index + 1
            accept[batch_ids, selected_region_index] = case_accept
            best_index = torch.where(accept, best_index, torch.zeros_like(best_index))
            best_score = torch.zeros(b, grid * grid, device=c0.device, dtype=c0.dtype)
            best_score[batch_ids, selected_region_index] = torch.where(
                case_accept, flat_best_score, torch.zeros_like(flat_best_score)
            )
            selected_prompt = torch.zeros_like(best_score)
            hard_route = torch.zeros_like(soft_route)
            hard_route[:, :, 0] = 1.0
            hard_route[batch_ids, selected_region_index, 0] = torch.where(
                case_accept,
                torch.zeros_like(case_accept, dtype=c0.dtype),
                torch.ones_like(case_accept, dtype=c0.dtype),
            )
            hard_route[
                batch_ids, selected_region_index, selected_candidate_index + 1
            ] = case_accept.to(c0.dtype)
            region_route = (
                _StraightThroughOneHot.apply(hard_route, soft_route)
                if self.training else hard_route
            )
            candidate_prob = soft_candidate / soft_candidate.sum(
                dim=2, keepdim=True
            ).clamp_min(EPS)
        elif self.v529_calibration_first_outcome:
            candidate_score = v529_normalized_score.masked_fill(~region_valid, floor)
            flat_score = candidate_score.reshape(b, -1)
            preserve_logit = flat_score.new_zeros(b, 1)
            global_soft = F.softmax(
                torch.cat([preserve_logit, flat_score], dim=1)
                / self.v529_route_temperature,
                dim=1,
            )
            soft_candidate = global_soft[:, 1:].reshape(b, grid * grid, n)
            soft_preserve = (1.0 - soft_candidate.sum(dim=2)).clamp(0.0, 1.0)
            soft_route = torch.cat([soft_preserve[..., None], soft_candidate], dim=2)
            action_score = torch.log(soft_route.clamp_min(EPS))

            flat_best_score, flat_best_index = flat_score.max(dim=1)
            selected_region_index = flat_best_index // n
            selected_candidate_index = flat_best_index % n
            selected_gain = v529_predicted_gain.reshape(b, -1).gather(
                1, flat_best_index[:, None]
            )[:, 0]
            selected_harm = v529_harm_fraction.reshape(b, -1).gather(
                1, flat_best_index[:, None]
            )[:, 0]
            selected_valid_action = region_valid.reshape(b, -1).gather(
                1, flat_best_index[:, None]
            )[:, 0]
            case_accept = (
                self.v529_selector_enabled
                and self.v529_execution_enabled
            )
            if case_accept:
                case_accept = (
                    selected_valid_action
                    & (selected_gain > self.v529_min_gain)
                    & (selected_harm <= self.v529_max_harm)
                )
            else:
                case_accept = torch.zeros_like(selected_valid_action)

            best_index = torch.zeros(b, grid * grid, device=c0.device, dtype=torch.long)
            accept = torch.zeros(b, grid * grid, device=c0.device, dtype=torch.bool)
            batch_ids = torch.arange(b, device=c0.device)
            best_index[batch_ids, selected_region_index] = selected_candidate_index + 1
            accept[batch_ids, selected_region_index] = case_accept
            best_index = torch.where(accept, best_index, torch.zeros_like(best_index))
            best_score = torch.zeros(b, grid * grid, device=c0.device, dtype=c0.dtype)
            best_score[batch_ids, selected_region_index] = torch.where(
                case_accept, flat_best_score, torch.zeros_like(flat_best_score)
            )
            selected_prompt = torch.zeros_like(best_score)
            hard_route = torch.zeros_like(soft_route)
            hard_route[:, :, 0] = 1.0
            hard_route[batch_ids, selected_region_index, 0] = torch.where(
                case_accept,
                torch.zeros_like(case_accept, dtype=c0.dtype),
                torch.ones_like(case_accept, dtype=c0.dtype),
            )
            hard_route[
                batch_ids, selected_region_index, selected_candidate_index + 1
            ] = case_accept.to(c0.dtype)
            region_route = (
                _StraightThroughOneHot.apply(hard_route, soft_route)
                if self.training else hard_route
            )
            candidate_prob = soft_candidate / soft_candidate.sum(
                dim=2, keepdim=True
            ).clamp_min(EPS)
        elif self.outcome_composer:
            # One auditable action per case.  Preserve is the explicit zero-gain
            # action; no independent Editability head can authorize a harmful
            # candidate merely because some other candidate is beneficial.
            candidate_score = v528_gain_lcb.masked_fill(~region_valid, floor)
            flat_score = candidate_score.reshape(b, -1)
            preserve_logit = flat_score.new_zeros(b, 1)
            global_soft = F.softmax(
                torch.cat([preserve_logit, flat_score], dim=1)
                / self.outcome_temperature,
                dim=1,
            )
            soft_candidate = global_soft[:, 1:].reshape(b, grid * grid, n)
            soft_preserve = (1.0 - soft_candidate.sum(dim=2)).clamp(0.0, 1.0)
            soft_route = torch.cat([soft_preserve[..., None], soft_candidate], dim=2)
            action_score = torch.log(soft_route.clamp_min(EPS))

            flat_best_score, flat_best_index = flat_score.max(dim=1)
            selected_region_index = flat_best_index // n
            selected_candidate_index = flat_best_index % n
            selected_harm = v528_harm_fraction.reshape(b, -1).gather(
                1, flat_best_index[:, None]
            )[:, 0]
            selected_valid_action = region_valid.reshape(b, -1).gather(
                1, flat_best_index[:, None]
            )[:, 0]
            exploration = self.training and (
                self.current_epoch < self.outcome_exploration_epochs
            )
            active_min_gain = (
                -1.0 if exploration else self.outcome_min_gain
            )
            active_max_harm = (
                1.0 if exploration else self.outcome_max_harm
            )
            case_accept = (
                selected_valid_action
                & (flat_best_score > active_min_gain)
                & (selected_harm <= active_max_harm)
            )

            best_index = torch.zeros(b, grid * grid, device=c0.device, dtype=torch.long)
            accept = torch.zeros(b, grid * grid, device=c0.device, dtype=torch.bool)
            batch_ids = torch.arange(b, device=c0.device)
            best_index[batch_ids, selected_region_index] = selected_candidate_index + 1
            accept[batch_ids, selected_region_index] = case_accept
            best_index = torch.where(accept, best_index, torch.zeros_like(best_index))
            best_score = torch.zeros(b, grid * grid, device=c0.device, dtype=c0.dtype)
            best_score[batch_ids, selected_region_index] = torch.where(
                case_accept, flat_best_score, torch.zeros_like(flat_best_score)
            )
            selected_prompt = torch.zeros_like(best_score)
            hard_route = torch.zeros_like(soft_route)
            hard_route[:, :, 0] = 1.0
            hard_route[batch_ids, selected_region_index, 0] = torch.where(
                case_accept,
                torch.zeros_like(case_accept, dtype=c0.dtype),
                torch.ones_like(case_accept, dtype=c0.dtype),
            )
            hard_route[
                batch_ids, selected_region_index, selected_candidate_index + 1
            ] = case_accept.to(c0.dtype)
            if self.training:
                region_route = _StraightThroughOneHot.apply(hard_route, soft_route)
            else:
                region_route = hard_route
            candidate_prob = soft_candidate / soft_candidate.sum(
                dim=2, keepdim=True
            ).clamp_min(EPS)
        elif self.factorized_ranker:
            # V527: the conditional candidate score is the sole ranking score.
            # Prompt/family/harm predictions remain learnable features and audit
            # heads, but are not hand-added to the deployed score.
            candidate_score = conditional_score.masked_fill(~region_valid, floor)
            candidate_prob = F.softmax(
                candidate_score / self.conditional_temperature, dim=2
            )
            candidate_prob = candidate_prob * region_valid.to(candidate_prob.dtype)
            candidate_prob = candidate_prob / candidate_prob.sum(dim=2, keepdim=True).clamp_min(EPS)
            edit_prob = torch.sigmoid(editability_logit)
            has_action = region_valid.any(dim=2)
            edit_prob = edit_prob * has_action.to(edit_prob.dtype)
            preserve_prob = 1.0 - edit_prob
            soft_route = torch.cat(
                [preserve_prob[..., None], edit_prob[..., None] * candidate_prob],
                dim=2,
            )
            action_score = torch.log(soft_route.clamp_min(EPS))
            best_candidate_score, best_candidate_index = candidate_score.max(dim=2)
            best_score = best_candidate_score
            best_index = best_candidate_index + 1
            candidate_wins = has_action
            selected_prompt = direction_prompt.gather(
                2, best_candidate_index[..., None]
            ).squeeze(-1)
            accept = candidate_wins & (edit_prob >= self.editability_threshold)
        else:
            if self.utility_aligned_score:
                candidate_score = (
                    region_mu
                    - self.lcb_beta * region_sigma
                    - self.harm_penalty * torch.sigmoid(region_harm_logit)
                    + route_residual
                )
            else:
                candidate_score = (
                    route_raw[:, :, 1:]
                    + region_mu
                    - self.lcb_beta * region_sigma
                    - self.harm_penalty * torch.sigmoid(region_harm_logit)
                    + family_term
                    + self.prompt_weight * direction_prompt
                )
            candidate_score = candidate_score.masked_fill(~region_valid, floor)
            action_score = torch.cat([torch.zeros_like(route_raw[:, :, :1]), candidate_score], dim=2)
            soft_route = F.softmax(action_score / self.route_temperature, dim=2)
            best_score, best_index = action_score.max(dim=2)
            candidate_wins = best_index > 0
            selected_prompt = direction_prompt.gather(2, (best_index - 1).clamp_min(0)[..., None]).squeeze(-1)
            accept = (
                candidate_wins
                & (best_score > self.region_accept_threshold)
                & (selected_prompt >= self.prompt_accept_threshold)
            )

        if self.current_epoch < self.deploy_start_epoch:
            accept = torch.zeros_like(accept)
        hard_index = torch.where(accept, best_index, torch.zeros_like(best_index))
        if not (
            self.outcome_composer
            or self.v529_calibration_first_outcome
            or self.v530_probability_calibrated_outcome
        ):
            hard_route = torch.zeros_like(soft_route).scatter_(2, hard_index[..., None], 1.0)
            if self.training:
                region_route = _StraightThroughOneHot.apply(hard_route, soft_route)
            else:
                region_route = hard_route if self.hard_inference else soft_route
        else:
            # Rebuild after deploy warmup so a pre-deploy epoch is guaranteed
            # Preserve even though the global outcome selector found an action.
            hard_route = torch.zeros_like(soft_route).scatter_(2, hard_index[..., None], 1.0)
            if self.training:
                region_route = _StraightThroughOneHot.apply(hard_route, soft_route)
            else:
                region_route = hard_route

        route_grid = region_route.transpose(1, 2).reshape(b, n + 1, grid, grid)
        route_pixel = _resize(route_grid, (h, w), mode="nearest")
        candidate_route_pixel = route_pixel[:, 1:]
        selected_candidate = (
            route_pixel[:, :1] * c0
            + (candidate_route_pixel * nonbase).sum(dim=1, keepdim=True)
        )
        selected_valid_soft = (candidate_route_pixel * valid.to(c0.dtype)).sum(dim=1, keepdim=True)
        selected_region_accept = candidate_route_pixel.sum(dim=1, keepdim=True)
        selected_hard_changed = (selected_candidate >= 0.5) != (c0 >= 0.5)
        selected_valid = (selected_valid_soft > 0.5) & selected_hard_changed & (selected_region_accept > 0.5)

        dense_lcb = _resize(dense_lcb_low.reshape(b * n, 1, *low_size), (h, w)).reshape(b, n, h, w)
        selected_lcb = (candidate_route_pixel * dense_lcb).sum(dim=1, keepdim=True)
        gray_full, edge_full = _gray_edge(image, (h, w))
        boundary_full = _soft_boundary(c0)
        refiner_input = torch.cat(
            [
                gray_full,
                edge_full,
                boundary_full,
                c0,
                selected_candidate,
                selected_candidate - c0,
                (selected_candidate - c0).abs(),
                selected_valid_soft.clamp(0.0, 1.0),
                selected_lcb,
                prompt_prob,
            ],
            dim=1,
        )
        refiner_gate_logit = self.refiner(refiner_input)
        refiner_gate_prob = torch.sigmoid(refiner_gate_logit)

        # Region-level complete action: the selector may choose one real M1
        # candidate per region, but it may not fragment that candidate with a
        # learned pixel-wise gate.  The legacy refiner remains available only
        # behind an explicit compatibility switch.
        complete_region_gate = selected_valid & (selected_region_accept > 0.5)
        if self.full_region_execution:
            pre_guard_gate = complete_region_gate.to(c0.dtype)
        else:
            hard_refiner = (refiner_gate_prob >= self.refiner_gate_threshold) & complete_region_gate
            hard_refiner_float = hard_refiner.to(c0.dtype)
            soft_refiner = refiner_gate_prob * selected_valid_soft.clamp(0.0, 1.0) * selected_region_accept
            pre_guard_gate = (
                hard_refiner_float + soft_refiner - soft_refiner.detach()
                if self.training else
                (hard_refiner_float if self.hard_inference else soft_refiner)
            )

        pre_guard_prob = (c0 + pre_guard_gate * (selected_candidate - c0)).clamp(EPS, 1.0 - EPS)

        selected_idx = (best_index - 1).clamp_min(0)
        selected_mu = region_mu.gather(2, selected_idx[..., None]).squeeze(-1)
        selected_sigma = region_sigma.gather(2, selected_idx[..., None]).squeeze(-1)
        selected_harm_prob = torch.sigmoid(region_harm_logit).gather(2, selected_idx[..., None]).squeeze(-1)
        accepted_float = accept.to(c0.dtype)
        accepted_mass = accepted_float.sum(dim=1).clamp_min(1.0)
        aggregate = torch.stack(
            [
                accepted_float.mean(dim=1),
                (selected_mu * accepted_float).sum(dim=1) / accepted_mass,
                (selected_sigma * accepted_float).sum(dim=1) / accepted_mass,
                (selected_harm_prob * accepted_float).sum(dim=1) / accepted_mass,
                (selected_prompt * accepted_float).sum(dim=1) / accepted_mass,
                best_score.masked_fill(~accept, 0.0).sum(dim=1) / accepted_mass,
                selected_valid_soft.flatten(1).mean(dim=1),
                (pre_guard_prob - c0).abs().flatten(1).mean(dim=1),
            ],
            dim=1,
        )
        case_raw = self.case_verifier(torch.cat([global_context, aggregate], dim=1))
        case_gain_mu = case_raw[:, 0]
        case_logvar = case_raw[:, 1].clamp(self.logvar_min, self.logvar_max)
        case_sigma = torch.exp(0.5 * case_logvar)
        case_harm_logit = case_raw[:, 2]
        case_lcb = (
            case_gain_mu
            - self.lcb_beta * case_sigma
            - self.harm_penalty * torch.sigmoid(case_harm_logit)
        )
        case_accept = (
            (case_lcb > self.case_accept_threshold)
            & (torch.sigmoid(case_harm_logit) <= self.case_harm_threshold)
            & accept.any(dim=1)
        )
        if not self.case_guard_enabled:
            case_accept = accept.any(dim=1)
        if self.current_epoch < self.deploy_start_epoch:
            case_accept = torch.zeros_like(case_accept)
        case_gate = case_accept[:, None, None, None].to(c0.dtype)
        deploy_gate = pre_guard_gate * case_gate
        final_prob = (c0 + deploy_gate * (selected_candidate - c0)).clamp(EPS, 1.0 - EPS)
        if self.force_preserve_output:
            final_prob = c0
            deploy_gate = torch.zeros_like(deploy_gate)
            case_accept = torch.zeros_like(case_accept)
        selected_index_grid = hard_index.reshape(b, grid, grid)
        selected_index_map = _resize(selected_index_grid[:, None].float(), (h, w), mode="nearest")[:, 0].long()
        changed = (final_prob >= 0.5) != (c0 >= 0.5)

        return {
            "m2_fused_probs": final_prob[:, 0],
            # V527 exposes the pre-guard candidate-realizable proposal while
            # preserving the legacy V524/V526 output contract unchanged.
            "m2_training_probs": (
                pre_guard_prob[:, 0] if (self.factorized_ranker or self.outcome_composer or self.v529_calibration_first_outcome or self.v530_probability_calibrated_outcome) else final_prob[:, 0]
            ),
            "m2_proposal_probs": (
                pre_guard_prob[:, 0] if (self.factorized_ranker or self.outcome_composer or self.v529_calibration_first_outcome or self.v530_probability_calibrated_outcome) else final_prob[:, 0]
            ),
            "m2_convex_probs": (
                pre_guard_prob[:, 0] if (self.factorized_ranker or self.outcome_composer or self.v529_calibration_first_outcome or self.v530_probability_calibrated_outcome) else final_prob[:, 0]
            ),
            "m2_edit_gate_prob": pre_guard_gate,
            "m2_residual_map": final_prob - c0,
            "v524_candidate_valid": valid,
            "v524_dense_utility_mu": _resize(dense_mu_low.reshape(b * n, 1, *low_size), (h, w)).reshape(b, n, h, w),
            "v524_dense_logvar": _resize(dense_logvar_low.reshape(b * n, 1, *low_size), (h, w)).reshape(b, n, h, w),
            "v524_dense_harm_logit": _resize(dense_harm_low.reshape(b * n, 1, *low_size), (h, w)).reshape(b, n, h, w),
            "v524_dense_lcb": dense_lcb,
            "v524_prompt_logits": prompt_logits,
            "v524_prompt_prob": prompt_prob,
            "v524_family_logits": family_logits,
            "v524_family_prob": family_prob,
            "v524_region_action_score": action_score,
            "v524_region_soft_route": soft_route,
            "v524_region_hard_route": hard_route,
            "v524_region_selected_index": hard_index,
            "v524_region_candidate_valid": region_valid,
            "v526_region_geometric_valid": geometric_region_valid,
            "v524_region_utility_mu": region_mu,
            "v524_region_logvar": region_logvar,
            "v524_region_harm_logit": region_harm_logit,
            "v524_region_prompt_support": direction_prompt,
            "v526_region_causal_valid": causal_valid,
            "v526_route_residual": route_residual,
            "v527_oracle_preserving_validity": c0.new_full(
                (b,), float(self.oracle_preserving_validity)
            ),
            "v527_factorized_ranker": c0.new_full(
                (b,), float(self.factorized_ranker)
            ),
            "v527_region_editability_logit": editability_logit,
            "v527_region_editability_prob": torch.sigmoid(editability_logit),
            "v527_region_conditional_score": conditional_score.masked_fill(
                ~region_valid, floor
            ),
            "v527_region_conditional_prob": (
                candidate_prob
                if (self.factorized_ranker or self.outcome_composer or self.v529_calibration_first_outcome or self.v530_probability_calibrated_outcome)
                else F.softmax(candidate_score / self.route_temperature, dim=2)
            ),
            "v528_outcome_composer": c0.new_full((b,), float(self.outcome_composer)),
            "v528_base_confusion_logit": v528_base_confusion_logit,
            "v528_base_confusion_prob": v528_base_confusion_prob,
            "v528_outcome_correctness_logit": v528_outcome_correctness_logit,
            "v528_outcome_logvar": v528_outcome_logvar,
            "v528_add_fraction": v528_add_fraction,
            "v528_remove_fraction": v528_remove_fraction,
            "v528_pred_fn_fix": v528_fn_fix,
            "v528_pred_fp_fix": v528_fp_fix,
            "v528_pred_tn_harm": v528_tn_harm,
            "v528_pred_tp_harm": v528_tp_harm,
            "v528_predicted_gain": v528_predicted_gain,
            "v528_gain_lcb": v528_gain_lcb,
            "v528_predicted_harm_fraction": v528_harm_fraction,
            "v529_calibration_first_outcome": c0.new_full(
                (b,), float(self.v529_calibration_first_outcome)
            ),
            "v529_base_error_logit": v529_base_error_logit,
            "v529_base_error_prob": v529_base_error_prob,
            "v529_pred_base_tp": v529_base_tp,
            "v529_pred_base_fp": v529_base_fp,
            "v529_pred_base_fn": v529_base_fn,
            "v529_pred_base_dice": v529_base_dice,
            "v529_outcome_pixel_logit": v529_pixel_logit,
            "v529_outcome_logvar": v529_outcome_logvar,
            "v529_pred_fn_fix": v529_fn_fix,
            "v529_pred_fp_fix": v529_fp_fix,
            "v529_pred_tn_harm": v529_tn_harm,
            "v529_pred_tp_harm": v529_tp_harm,
            "v529_predicted_gain": v529_predicted_gain,
            "v529_normalized_score": v529_normalized_score,
            "v529_predicted_sigma": v529_sigma,
            "v529_predicted_harm_fraction": v529_harm_fraction,
            "v530_probability_calibrated_outcome": c0.new_full(
                (b,), float(self.v530_probability_calibrated_outcome)
            ),
            "v530_base_error_pixel_logit": v530_base_error_pixel_logit,
            "v530_base_error_pixel_prob": v530_base_error_prob,
            "v530_pred_base_tp": v530_base_tp,
            "v530_pred_base_fp": v530_base_fp,
            "v530_pred_base_fn": v530_base_fn,
            "v530_pred_base_dice": v530_base_dice,
            "v530_outcome_pixel_logit": v530_outcome_pixel_logit,
            "v530_outcome_logvar": v530_outcome_logvar,
            "v530_pred_fn_fix": v530_fn_fix,
            "v530_pred_fp_fix": v530_fp_fix,
            "v530_pred_tn_harm": v530_tn_harm,
            "v530_pred_tp_harm": v530_tp_harm,
            "v530_predicted_gain": v530_predicted_gain,
            "v530_normalized_score": v530_normalized_score,
            "v530_predicted_sigma": v530_sigma,
            "v530_predicted_harm_fraction": v530_harm_fraction,
            "v526_utility_aligned_score": c0.new_full((b,), float(self.utility_aligned_score)),
            "v524_region_accept": accept.to(c0.dtype),
            "v524_selected_index_map": selected_index_map,
            "v524_selected_candidate_prob": selected_candidate,
            "v524_selected_valid": selected_valid.to(c0.dtype),
            "v524_refiner_gate_logit": refiner_gate_logit,
            "v524_refiner_gate_prob": refiner_gate_prob,
            "v524_pre_guard_prob": pre_guard_prob[:, 0],
            "v524_case_gain_mu": case_gain_mu,
            "v524_case_logvar": case_logvar,
            "v524_case_harm_logit": case_harm_logit,
            "v524_case_lcb": case_lcb,
            "v524_case_accept": case_accept.to(c0.dtype),
            "v524_deploy_gate": deploy_gate,
            "v524_region_grid_size": c0.new_full((b,), float(grid)),
            "v524_deploy_phase": c0.new_full((b,), float(self.current_epoch >= self.deploy_start_epoch)),
            "v524_changed_pixel_rate": changed.float().flatten(1).mean(dim=1),
            "v524_selected_region_rate": accept.float().mean(dim=1),
        }
