"""V538/V540/V541 online component refinement.

V541 keeps the V540 continuous signed-logit component editor, but replaces the
under-specified single linear gain reader with a candidate-conditional risk
selector.  M1 still predicts component masks, polarity, dose, and presence.
M2 receives *detached* Base and exact hard single-slot candidate outcomes and
learns separate Benefit, Harm, rank, calibrated gain, and uncertainty heads.

The separation is deliberate:

* M1 gradients never receive selector shortcuts;
* rank logits are dimensionless and are no longer forced to share the Dice-gain
  scale;
* deployment uses Benefit/Harm risk checks and a lower-confidence-bound gain;
* training diagnostics can evaluate the exact greedy composer even while the
  persistent deployment gate remains closed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1.0e-6


def _group_count(channels: int) -> int:
    groups = min(8, int(channels))
    while groups > 1 and int(channels) % groups != 0:
        groups -= 1
    return groups


def _masked_mean(feature: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Pool ``[B,C,H,W]`` features with ``[B,K,H,W]`` soft masks."""
    numerator = (feature[:, None] * masks[:, :, None]).sum(dim=(-2, -1))
    denominator = masks.sum(dim=(-2, -1)).unsqueeze(-1).clamp_min(EPS)
    return numerator / denominator


def _masked_scalar(value: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Pool ``[B,C,H,W]`` values to ``[B,K,C]`` with slot masks."""
    numerator = (value[:, None] * masks[:, :, None]).sum(dim=(-2, -1))
    denominator = masks.sum(dim=(-2, -1)).unsqueeze(-1).clamp_min(EPS)
    return numerator / denominator


def _safe_logit(probability: torch.Tensor) -> torch.Tensor:
    probability = probability.clamp(EPS, 1.0 - EPS)
    return torch.log(probability) - torch.log1p(-probability)


@dataclass(frozen=True)
class V538DeploymentStats:
    accepted_components: int
    changed_pixels: int


class V538OnlineComponentRefiner(nn.Module):
    """Differentiable M1 component slots and candidate-conditional M2 risk."""

    def __init__(
        self,
        *,
        feature_channels: int,
        hidden_dim: int = 128,
        num_slots: int = 8,
        dropout: float = 0.10,
        mask_temperature: float = 1.0,
        action_temperature: float = 0.70,
        min_area_fraction: float = 1.0e-4,
        max_component_area_fraction: float = 0.035,
        initial_presence_rate: float = 0.10,
        initial_gain_bias: float = -0.002,
        deployment_mask_threshold: float = 0.50,
        deployment_presence_threshold: float = 0.50,
        deployment_min_gain: float = 0.001,
        max_steps: int = 3,
        max_overlap: float = 0.20,
        max_total_edit_fraction: float = 0.035,
        continuous_dose_enabled: bool = True,
        polarity_temperature: float = 0.70,
        minimum_dose: float = 0.25,
        maximum_dose: float = 16.0,
        initial_dose: float = 1.0,
        candidate_outcome_selector_enabled: bool = True,
        selector_spatial_size: int = 16,
        selector_hidden_dim: int = 128,
        selector_dropout: float = 0.10,
        selector_gain_scale: float = 1000.0,
        selector_benefit_threshold: float = 0.70,
        selector_harm_threshold: float = 0.10,
        selector_lcb_beta: float = 1.0,
        selector_logvar_min: float = -8.0,
        selector_logvar_max: float = 4.0,
        selector_initial_benefit_rate: float = 0.10,
        selector_initial_harm_rate: float = 0.50,
        use_gain_as_decision_score: bool = False,
        adaptive_cardinality_hard_mask: bool = False,
        prior_free_outcome_init: bool = False,
        slot_competition_enabled: bool = False,
        gain_sign_shadow_deploy_enabled: bool = False,
        factorized_outcome_enabled: bool = False,
        factorized_direction_zero_init: bool = False,
        factorized_deployment_enabled: bool = False,
        deployment_editability_threshold: float = 0.50,
        deployment_direction_threshold: float = 0.50,
    ) -> None:
        super().__init__()
        feature_channels = int(feature_channels)
        hidden_dim = int(hidden_dim)
        selector_hidden_dim = int(selector_hidden_dim)
        self.num_slots = max(int(num_slots), 1)
        self.mask_temperature = max(float(mask_temperature), 1.0e-4)
        self.action_temperature = max(float(action_temperature), 1.0e-4)
        self.polarity_temperature = max(float(polarity_temperature), 1.0e-4)
        self.min_area_fraction = max(float(min_area_fraction), 0.0)
        self.max_component_area_fraction = min(
            max(float(max_component_area_fraction), self.min_area_fraction + EPS),
            1.0,
        )
        self.deployment_mask_threshold = min(
            max(float(deployment_mask_threshold), 0.0), 1.0
        )
        self.deployment_presence_threshold = min(
            max(float(deployment_presence_threshold), 0.0), 1.0
        )
        self.deployment_min_gain = float(deployment_min_gain)
        self.max_steps = max(int(max_steps), 1)
        self.max_overlap = min(max(float(max_overlap), 0.0), 1.0)
        self.max_total_edit_fraction = min(
            max(float(max_total_edit_fraction), 0.0), 1.0
        )
        self.continuous_dose_enabled = bool(continuous_dose_enabled)
        self.minimum_dose = max(float(minimum_dose), 0.0)
        self.maximum_dose = max(float(maximum_dose), self.minimum_dose + EPS)

        self.candidate_outcome_selector_enabled = bool(
            candidate_outcome_selector_enabled
        )
        self.selector_spatial_size = max(int(selector_spatial_size), 4)
        self.selector_gain_scale = max(float(selector_gain_scale), 1.0)
        self.selector_benefit_threshold = min(
            max(float(selector_benefit_threshold), 0.0), 1.0
        )
        self.selector_harm_threshold = min(
            max(float(selector_harm_threshold), 0.0), 1.0
        )
        self.selector_lcb_beta = max(float(selector_lcb_beta), 0.0)
        self.selector_logvar_min = float(selector_logvar_min)
        self.selector_logvar_max = max(
            float(selector_logvar_max), self.selector_logvar_min + EPS
        )
        self.use_gain_as_decision_score = bool(use_gain_as_decision_score)
        self.adaptive_cardinality_hard_mask = bool(
            adaptive_cardinality_hard_mask
        )
        self.prior_free_outcome_init = bool(prior_free_outcome_init)
        self.slot_competition_enabled = bool(slot_competition_enabled)
        self.gain_sign_shadow_deploy_enabled = bool(
            gain_sign_shadow_deploy_enabled
        )
        self.factorized_outcome_enabled = bool(factorized_outcome_enabled)
        self.factorized_direction_zero_init = bool(
            factorized_direction_zero_init
        )
        self.factorized_deployment_enabled = bool(
            factorized_deployment_enabled
        )
        self.deployment_editability_threshold = min(
            max(float(deployment_editability_threshold), 0.0), 1.0
        )
        self.deployment_direction_threshold = min(
            max(float(deployment_direction_threshold), 0.0), 1.0
        )

        groups = _group_count(hidden_dim)
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            nn.Dropout2d(float(dropout)),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
        )
        self.mask_head = nn.Conv2d(hidden_dim, self.num_slots, 1)

        # inside/ring feature, four cause means, four alpha means, base,
        # entropy, boundary and area.
        descriptor_dim = 2 * feature_channels + 12
        self.slot_trunk = nn.Sequential(
            nn.LayerNorm(descriptor_dim),
            nn.Linear(descriptor_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.action_head = nn.Linear(hidden_dim, 4)
        self.polarity_head = nn.Linear(hidden_dim, 2)
        self.dose_head = nn.Linear(hidden_dim, 1)
        self.presence_head = nn.Linear(hidden_dim, 1)

        # V541 M2.  Seven spatial channels are Base, exact candidate, signed
        # delta, absolute delta, hard mask, entropy, and boundary.
        outcome_channels = 64
        outcome_groups_1 = _group_count(32)
        outcome_groups_2 = _group_count(outcome_channels)
        self.m2_outcome_encoder = nn.Sequential(
            nn.Conv2d(7, 32, 3, padding=1, bias=False),
            nn.GroupNorm(outcome_groups_1, 32),
            nn.GELU(),
            nn.Conv2d(32, outcome_channels, 3, padding=1, bias=False),
            nn.GroupNorm(outcome_groups_2, outcome_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        selector_scalar_dim = 12
        selector_input_dim = hidden_dim + outcome_channels * 4 + selector_scalar_dim
        self.m2_selector_trunk = nn.Sequential(
            nn.LayerNorm(selector_input_dim),
            nn.Linear(selector_input_dim, selector_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(selector_dropout)),
            nn.Linear(selector_hidden_dim, selector_hidden_dim),
            nn.GELU(),
        )
        self.benefit_head = nn.Linear(selector_hidden_dim, 1)
        self.harm_head = nn.Linear(selector_hidden_dim, 1)
        # V547 explicitly factorises Neutral-vs-Editable and Benefit-vs-Harm.
        self.editability_head = nn.Linear(selector_hidden_dim, 1)
        self.direction_head = nn.Linear(selector_hidden_dim, 1)
        self.rank_head = nn.Linear(selector_hidden_dim, 1)
        # Compatibility name retained.  Its output is normalized gain and is
        # converted back to raw Dice-gain units before export/deployment.
        self.gain_head = nn.Linear(selector_hidden_dim, 1)
        self.gain_logvar_head = nn.Linear(selector_hidden_dim, 1)

        nn.init.zeros_(self.mask_head.weight)
        nn.init.constant_(self.mask_head.bias, -4.0)
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)
        nn.init.zeros_(self.polarity_head.weight)
        nn.init.zeros_(self.polarity_head.bias)
        nn.init.zeros_(self.dose_head.weight)
        initial_dose = min(max(float(initial_dose), self.minimum_dose), self.maximum_dose)
        dose_fraction = (initial_dose - self.minimum_dose) / (
            self.maximum_dose - self.minimum_dose
        )
        dose_fraction = min(max(dose_fraction, 1.0e-5), 1.0 - 1.0e-5)
        nn.init.constant_(
            self.dose_head.bias,
            float(torch.logit(torch.tensor(dose_fraction)).item()),
        )
        nn.init.zeros_(self.presence_head.weight)
        initial_presence_rate = min(
            max(float(initial_presence_rate), 1.0e-5), 1.0 - 1.0e-5
        )
        nn.init.constant_(
            self.presence_head.bias,
            float(torch.logit(torch.tensor(initial_presence_rate)).item()),
        )

        for head in (
            self.benefit_head,
            self.harm_head,
            self.editability_head,
            self.direction_head,
            self.rank_head,
            self.gain_head,
            self.gain_logvar_head,
        ):
            nn.init.normal_(head.weight, mean=0.0, std=0.01)
        # V543B: the two learned logits are Benefit/Harm log-ratios against
        # an explicit Neutral reference logit of zero.  This initializes the
        # three-class outcome distribution to the configured priors instead of
        # treating Benefit and Harm as two independent Bernoulli variables.
        benefit_rate = min(max(float(selector_initial_benefit_rate), 1.0e-5), 1.0 - 1.0e-5)
        harm_rate = min(max(float(selector_initial_harm_rate), 1.0e-5), 1.0 - 1.0e-5)
        neutral_rate = max(1.0 - benefit_rate - harm_rate, 1.0e-5)
        prior_total = benefit_rate + harm_rate + neutral_rate
        benefit_rate /= prior_total
        harm_rate /= prior_total
        neutral_rate /= prior_total
        if self.prior_free_outcome_init:
            # V545: maximum-entropy initialization.  A fixed 10/50/40 prior
            # created a negative basin while most mini-batches contained no
            # Benefit examples.  Zero logits add no tunable prior.
            nn.init.zeros_(self.benefit_head.bias)
            nn.init.zeros_(self.harm_head.bias)
        else:
            nn.init.constant_(
                self.benefit_head.bias,
                float(math.log(benefit_rate / neutral_rate)),
            )
            nn.init.constant_(
                self.harm_head.bias,
                float(math.log(harm_rate / neutral_rate)),
            )
        initial_editable = min(
            max(benefit_rate + harm_rate, 1.0e-5), 1.0 - 1.0e-5
        )
        initial_direction = min(
            max(benefit_rate / max(benefit_rate + harm_rate, 1.0e-5), 1.0e-5),
            1.0 - 1.0e-5,
        )
        if self.prior_free_outcome_init:
            nn.init.zeros_(self.editability_head.bias)
        else:
            nn.init.constant_(
                self.editability_head.bias,
                float(math.log(initial_editable / (1.0 - initial_editable))),
            )

        # V548 root fix: Direction represents P(Benefit | Editable).  Its
        # previous prior-derived initialization was negative whenever Harm was
        # more frequent than Benefit and put every candidate in a negative-Gain
        # basin before seeing data.  Keep the Editability prior, but initialize
        # the conditional direction at maximum entropy (p=0.5).
        if self.prior_free_outcome_init or self.factorized_direction_zero_init:
            nn.init.zeros_(self.direction_head.bias)
        else:
            nn.init.constant_(
                self.direction_head.bias,
                float(math.log(initial_direction / (1.0 - initial_direction))),
            )
        nn.init.zeros_(self.rank_head.bias)

        # The former signed Gain head is now a non-negative magnitude head.
        # Convert the historical raw Dice-gain magnitude into normalized space
        # and initialize the pre-Softplus bias accordingly.
        initial_magnitude = max(
            abs(float(initial_gain_bias)) * self.selector_gain_scale,
            1.0e-3,
        )
        inverse_softplus = math.log(math.expm1(initial_magnitude))
        nn.init.constant_(self.gain_head.bias, float(inverse_softplus))
        nn.init.zeros_(self.gain_logvar_head.bias)

    def _adaptive_hard_mask(
        self, slot_masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return exact-cardinality hard masks and a straight-through view.

        The hard area is the rounded expected mask mass, clipped only by the
        existing component-area contract.  It therefore removes the fixed 0.5
        mask threshold without introducing a new tuned scalar.  ``topk`` is
        used instead of thresholding so ties cannot turn a nearly uniform mask
        into a full-image component.
        """
        b, k, h, w = slot_masks.shape
        pixels = h * w
        flat = slot_masks.flatten(2)
        min_count = max(int(math.ceil(self.min_area_fraction * pixels)), 1)
        max_count = max(
            min(int(math.floor(self.max_component_area_fraction * pixels)), pixels),
            min_count,
        )
        desired = flat.sum(dim=2).round().to(torch.long).clamp(
            min=min_count, max=max_count
        )
        hard_flat = torch.zeros_like(flat)
        # B and K are deliberately small (BUSI uses 2 x 8).  Variable-k topk
        # avoids a global sort and guarantees exact cardinality under ties.
        for sample in range(b):
            for slot in range(k):
                count = int(desired[sample, slot].item())
                index = torch.topk(
                    flat[sample, slot], k=count, largest=True, sorted=False
                ).indices
                hard_flat[sample, slot].scatter_(0, index, 1.0)
        hard = hard_flat.view(b, k, h, w)
        straight_through = hard + slot_masks - slot_masks.detach()
        contrast = (
            flat.amax(dim=2) - flat.amin(dim=2)
        ) > torch.finfo(slot_masks.dtype).eps
        return hard.bool(), straight_through, contrast

    def _resolve_hard_slot_overlap(
        self,
        *,
        mask_logits: torch.Tensor,
        hard_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Resolve only active hard-mask overlaps by winner-take-all.

        The soft sigmoid slots remain independent for descriptor pooling and
        supervision.  Competition is therefore an ownership constraint in the
        executable mask space, not a normalization that can suppress every slot
        below the deployment threshold.
        """
        hard = hard_masks.bool()
        if not self.slot_competition_enabled:
            return hard
        overlap = hard.sum(dim=1, keepdim=True) > 1
        if not bool(overlap.any().item()):
            return hard
        floor = torch.finfo(mask_logits.dtype).min
        active_logits = mask_logits.detach().masked_fill(~hard, floor)
        winner = active_logits.argmax(dim=1, keepdim=True)
        winner_mask = torch.zeros_like(hard).scatter_(1, winner, True)
        return torch.where(
            overlap.expand_as(hard),
            winner_mask & hard,
            hard,
        )

    def _hard_compose(
        self,
        *,
        base_probability: Optional[torch.Tensor] = None,
        slot_masks: torch.Tensor,
        slot_polarity_logits: Optional[torch.Tensor] = None,
        slot_doses: Optional[torch.Tensor] = None,
        slot_actions: Optional[torch.Tensor] = None,
        slot_scores: torch.Tensor,
        slot_presence: torch.Tensor,
        slot_valid: torch.Tensor,
        deploy_enabled: bool,
        slot_benefit: Optional[torch.Tensor] = None,
        slot_harm: Optional[torch.Tensor] = None,
        slot_gain_lcb: Optional[torch.Tensor] = None,
        slot_editability: Optional[torch.Tensor] = None,
        slot_direction: Optional[torch.Tensor] = None,
        gain_sign_only: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Greedy bounded composition of non-overlapping signed logit doses."""
        b, k, h, w = slot_masks.shape
        device = slot_masks.device
        dtype = slot_masks.dtype
        action_weight = torch.zeros((b, 4, h, w), device=device, dtype=dtype)
        polarity_weight = torch.zeros((b, 2, h, w), device=device, dtype=dtype)
        accepted = torch.zeros((b, k), device=device, dtype=torch.bool)
        selected_index = torch.zeros((b,), device=device, dtype=torch.long)
        selected_score = torch.zeros((b,), device=device, dtype=slot_scores.dtype)
        accepted_count = torch.zeros((b,), device=device, dtype=slot_scores.dtype)
        changed_fraction = torch.zeros((b,), device=device, dtype=slot_scores.dtype)
        signed_delta = torch.zeros((b, 1, h, w), device=device, dtype=dtype)
        if base_probability is None:
            base = slot_masks.new_full((b, 1, h, w), 0.5)
        else:
            base = base_probability[:, :1].detach().clamp(EPS, 1.0 - EPS)
        if slot_polarity_logits is None:
            if slot_actions is None:
                slot_polarity_logits = slot_masks.new_zeros((b, k, 2))
            else:
                hard_legacy = slot_actions.detach().argmax(dim=2)
                remove = ((hard_legacy == 0) | (hard_legacy == 2)).to(dtype)
                add = 1.0 - remove
                slot_polarity_logits = torch.stack([remove, add], dim=2) * 10.0
        if slot_doses is None:
            slot_doses = slot_masks.new_ones((b, k))
        if slot_benefit is None:
            slot_benefit = torch.ones_like(slot_scores)
        if slot_harm is None:
            slot_harm = torch.zeros_like(slot_scores)
        if slot_gain_lcb is None:
            slot_gain_lcb = slot_scores
        if slot_editability is None:
            slot_editability = slot_benefit + slot_harm
        if slot_direction is None:
            slot_direction = slot_benefit / slot_editability.clamp_min(EPS)

        if not bool(deploy_enabled):
            return {
                "selected_action_weight": action_weight,
                "selected_polarity_weight": polarity_weight,
                "selected_signed_delta": signed_delta,
                "selected_final_probability": base,
                "accepted_slots": accepted,
                "selected_index": selected_index,
                "selected_score": selected_score,
                "predicted_execute": torch.zeros((b,), device=device, dtype=torch.bool),
                "accepted_count": accepted_count,
                "changed_fraction": changed_fraction,
            }

        hard_masks = slot_masks.detach() >= self.deployment_mask_threshold
        hard_polarity = slot_polarity_logits.detach().argmax(dim=2)
        doses = slot_doses.detach().clamp(self.minimum_dose, self.maximum_dose)
        scores = slot_scores.detach()
        presence = slot_presence.detach()
        benefit = slot_benefit.detach()
        harm = slot_harm.detach()
        gain_lcb = slot_gain_lcb.detach()
        editability = slot_editability.detach()
        direction = slot_direction.detach()
        valid = slot_valid.detach().bool()
        max_pixels = max(int(round(self.max_total_edit_fraction * h * w)), 1)

        for sample in range(b):
            occupied = torch.zeros((h, w), device=device, dtype=torch.bool)
            order = torch.argsort(scores[sample], descending=True)
            first_index = None
            first_score = None
            steps = 0
            for index_tensor in order:
                if steps >= self.max_steps:
                    break
                index = int(index_tensor.item())
                if not bool(valid[sample, index].item()):
                    continue
                if gain_sign_only:
                    # V546 validation shadow policy: the semantic boundary for
                    # execution is predicted utility > 0.  No calibrated
                    # probability threshold or tuned margin is used.  Existing
                    # area/overlap/edit-budget contracts remain active.
                    if float(gain_lcb[sample, index].item()) <= 0.0:
                        continue
                else:
                    if float(presence[sample, index].item()) < self.deployment_presence_threshold:
                        continue
                    if self.factorized_outcome_enabled and self.factorized_deployment_enabled:
                        # V549: deployment follows the same conditional variables
                        # used by training.  The old product thresholds B=E*D and
                        # H=E*(1-D) implicitly required D close to one whenever
                        # editability E was moderate, producing a permanent
                        # zero-execution dead zone.
                        if float(editability[sample, index].item()) < self.deployment_editability_threshold:
                            continue
                        if float(direction[sample, index].item()) < self.deployment_direction_threshold:
                            continue
                    else:
                        if float(benefit[sample, index].item()) < self.selector_benefit_threshold:
                            continue
                        if float(harm[sample, index].item()) > self.selector_harm_threshold:
                            continue
                    if float(gain_lcb[sample, index].item()) <= self.deployment_min_gain:
                        continue
                mask = hard_masks[sample, index]
                area = int(mask.sum().item())
                if area <= 0:
                    continue
                overlap = int((mask & occupied).sum().item()) / float(max(area, 1))
                if overlap > self.max_overlap:
                    continue
                new_mask = mask & (~occupied)
                new_area = int(new_mask.sum().item())
                if new_area <= 0 or int(occupied.sum().item()) + new_area > max_pixels:
                    continue

                polarity = int(hard_polarity[sample, index].item())
                dose = doses[sample, index]
                sign = -1.0 if polarity == 0 else 1.0
                delta = new_mask.to(dtype) * dose * sign
                signed_delta[sample, 0] = signed_delta[sample, 0] + delta
                polarity_weight[sample, polarity] = torch.maximum(
                    polarity_weight[sample, polarity], new_mask.to(dtype)
                )
                legacy_action = 0 if polarity == 0 else 1
                action_weight[sample, legacy_action] = torch.maximum(
                    action_weight[sample, legacy_action], new_mask.to(dtype)
                )
                occupied |= new_mask
                accepted[sample, index] = True
                if first_index is None:
                    first_index = index
                    first_score = scores[sample, index]
                steps += 1

            if first_index is not None:
                selected_index[sample] = first_index
                selected_score[sample] = first_score
            accepted_count[sample] = float(steps)
            changed_fraction[sample] = occupied.float().mean()

        final_probability = torch.sigmoid(_safe_logit(base) + signed_delta).clamp(
            EPS, 1.0 - EPS
        )
        return {
            "selected_action_weight": action_weight,
            "selected_polarity_weight": polarity_weight,
            "selected_signed_delta": signed_delta,
            "selected_final_probability": final_probability,
            "accepted_slots": accepted,
            "selected_index": selected_index,
            "selected_score": selected_score,
            "predicted_execute": accepted.any(dim=1),
            "accepted_count": accepted_count,
            "changed_fraction": changed_fraction,
        }

    def _candidate_selector(
        self,
        *,
        slot_feature: torch.Tensor,
        hard_masks: torch.Tensor,
        exact_candidate: torch.Tensor,
        base: torch.Tensor,
        entropy: torch.Tensor,
        boundary: torch.Tensor,
        presence_probs: torch.Tensor,
        slot_doses: torch.Tensor,
        polarity_probs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Encode exact candidate outcomes without passing gradients to M1."""
        b, k, h, w = exact_candidate.shape
        base_expand = base.expand(-1, k, -1, -1)
        entropy_expand = entropy[:, :1].detach().expand(-1, k, -1, -1)
        boundary_expand = boundary[:, :1].detach().expand(-1, k, -1, -1)
        candidate = exact_candidate.detach()
        base_detached = base_expand.detach()
        mask = hard_masks.detach().to(candidate.dtype)
        delta = candidate - base_detached
        spatial = torch.stack(
            [
                base_detached,
                candidate,
                delta,
                delta.abs(),
                mask,
                entropy_expand,
                boundary_expand,
            ],
            dim=2,
        ).reshape(b * k, 7, h, w)
        spatial = F.adaptive_avg_pool2d(
            spatial, (self.selector_spatial_size, self.selector_spatial_size)
        )
        spatial_feature = self.m2_outcome_encoder(spatial).flatten(1).reshape(b, k, -1)

        denominator = mask.sum(dim=(-2, -1)).clamp_min(1.0)
        def inside_mean(value: torch.Tensor) -> torch.Tensor:
            return (value * mask).sum(dim=(-2, -1)) / denominator

        add_delta = delta.clamp_min(0.0)
        remove_delta = (-delta).clamp_min(0.0)
        polarity_confidence = polarity_probs.detach().max(dim=2).values
        scalar = torch.stack(
            [
                mask.mean(dim=(-2, -1)),
                delta.abs().mean(dim=(-2, -1)),
                add_delta.mean(dim=(-2, -1)),
                remove_delta.mean(dim=(-2, -1)),
                inside_mean(base_detached),
                inside_mean(candidate),
                inside_mean(entropy_expand),
                inside_mean(boundary_expand),
                presence_probs.detach(),
                (slot_doses.detach() / self.maximum_dose).clamp(0.0, 1.0),
                polarity_confidence,
                (candidate >= 0.5).to(candidate.dtype).ne(
                    (base_detached >= 0.5)
                ).to(candidate.dtype).mean(dim=(-2, -1)),
            ],
            dim=2,
        )
        selector_input = torch.cat(
            [slot_feature.detach(), spatial_feature, scalar], dim=2
        )
        selector_feature = self.m2_selector_trunk(selector_input)
        if bool(getattr(self, "clean_dynamic_component_set_enabled", False)):
            clean_gain_head = getattr(self, "clean_gain_head", None)
            if clean_gain_head is None:
                raise RuntimeError("CLEAN component set requires clean_gain_head")
            # CLEAN M2 owns exactly one physical prediction: signed DeltaDice.
            # The direct regressor and its compact candidate encoder are trained
            # by the signed-gain MSE. Historical Benefit/Harm/Rank/LCB heads are
            # compatibility-only and are not consulted by the clean path.
            gain_scores = clean_gain_head(selector_feature).squeeze(-1)
            zero = torch.zeros_like(gain_scores)
            one = torch.ones_like(gain_scores)
            positive = (gain_scores >= 0.0).to(gain_scores.dtype)
            benefit_probs = positive
            harm_probs = 1.0 - positive
            neutral_probs = zero
            outcome_probs = torch.stack([neutral_probs, benefit_probs, harm_probs], dim=-1)
            outcome_logits = torch.where(
                outcome_probs > 0.5,
                torch.zeros_like(outcome_probs),
                torch.full_like(outcome_probs, -20.0),
            )
            editability_probs = one
            direction_probs = positive
            benefit_logits = torch.where(positive > 0.5, one, -one)
            harm_logits = -benefit_logits
            editability_logits = torch.full_like(gain_scores, 20.0)
            direction_logits = benefit_logits
            gain_logvar = zero
            gain_std = zero
            gain_lcb = gain_scores
            rank_scores = gain_scores
            return {
                "benefit_logits": benefit_logits,
                "benefit_probs": benefit_probs,
                "harm_logits": harm_logits,
                "harm_probs": harm_probs,
                "editability_logits": editability_logits,
                "editability_probs": editability_probs,
                "direction_logits": direction_logits,
                "direction_probs": direction_probs,
                "factorized_outcome_enabled": zero.mean(),
                "factorized_direction_zero_init_enabled": zero.mean(),
                "neutral_probs": neutral_probs,
                "outcome_logits": outcome_logits,
                "outcome_probs": outcome_probs,
                "signed_outcome": torch.sign(gain_scores),
                "gain_magnitude_normalized": gain_scores.abs(),
                "benefit_magnitude_normalized": gain_scores.clamp_min(0.0),
                "harm_magnitude_normalized": (-gain_scores).clamp_min(0.0),
                "benefit_contribution": gain_scores.clamp_min(0.0),
                "harm_contribution": gain_scores.clamp_max(0.0),
                "use_gain_as_decision_score": one.mean(),
                "rank_scores": rank_scores,
                "normalized_gain": gain_scores,
                "gain_scores": gain_scores,
                "gain_logvar": gain_logvar,
                "gain_std": gain_std,
                "gain_lcb": gain_lcb,
                "decision_scores": gain_scores,
                "selector_features": selector_feature,
                # Compatibility tensors: clean deployment/loss never consumes them.
                "direction_head_weight": self.direction_head.weight.detach(),
                "direction_head_bias": self.direction_head.bias.detach(),
                "editability_head_weight": self.editability_head.weight.detach(),
                "editability_head_bias": self.editability_head.bias.detach(),
                "gain_head_weight": clean_gain_head.weight,
                "gain_head_bias": clean_gain_head.bias,
                "factorized_deployment_enabled": zero.mean(),
            }
        rank_scores = self.rank_head(selector_feature).squeeze(-1)
        if self.factorized_outcome_enabled:
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
        else:
            benefit_logits = self.benefit_head(selector_feature).squeeze(-1)
            harm_logits = self.harm_head(selector_feature).squeeze(-1)
            editability_logits = torch.logsumexp(
                torch.stack([benefit_logits, harm_logits], dim=-1), dim=-1
            )
            direction_logits = benefit_logits - harm_logits
            neutral_logits = torch.zeros_like(benefit_logits)
            outcome_logits = torch.stack(
                [neutral_logits, benefit_logits, harm_logits], dim=-1
            )
            outcome_probs = torch.softmax(outcome_logits, dim=-1)
            neutral_probs = outcome_probs[..., 0]
            benefit_probs = outcome_probs[..., 1]
            harm_probs = outcome_probs[..., 2]
            editability_probs = benefit_probs + harm_probs
            direction_probs = benefit_probs / editability_probs.clamp_min(EPS)
            signed_outcome = benefit_probs - harm_probs

        # V543B: magnitude is non-negative and the outcome probabilities own the
        # sign.  Benefit/Harm and Gain can therefore no longer contradict each
        # other by construction.
        gain_magnitude_normalized = F.softplus(
            self.gain_head(selector_feature).squeeze(-1)
        )
        normalized_gain = signed_outcome * gain_magnitude_normalized
        gain_scores = normalized_gain / self.selector_gain_scale

        # V542 deterministic mode is selected by LCB beta == 0.  V541
        # configurations with beta > 0 retain their learned-uncertainty path.
        if self.selector_lcb_beta > 0.0:
            gain_logvar = self.gain_logvar_head(
                selector_feature
            ).squeeze(-1).clamp(
                self.selector_logvar_min,
                self.selector_logvar_max,
            )
            gain_std = (
                torch.exp(0.5 * gain_logvar)
                / self.selector_gain_scale
            )
            gain_lcb = (
                gain_scores
                - self.selector_lcb_beta * gain_std
            )
        else:
            gain_logvar = torch.zeros_like(gain_scores)
            gain_std = torch.zeros_like(gain_scores)
            gain_lcb = gain_scores
        # V544: in the rigorous minimal objective the only deployed utility
        # is the calibrated factorized Gain itself.  This removes an
        # untrained/easy auxiliary Rank head from the deployment path.
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
                1.0 if self.factorized_outcome_enabled else 0.0
            ),
            "factorized_direction_zero_init_enabled": gain_scores.new_tensor(
                1.0 if self.factorized_direction_zero_init else 0.0
            ),
            "neutral_probs": neutral_probs,
            "outcome_logits": outcome_logits,
            "outcome_probs": outcome_probs,
            "signed_outcome": signed_outcome,
            "gain_magnitude_normalized": gain_magnitude_normalized,
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
            # V549 exposes the selector representation and the exact factorized
            # head parameters to the loss.  A bounded cross-batch queue can then
            # re-evaluate stale-within-epoch features using current head weights,
            # preserving gradients while balancing Benefit/Harm 1:1.
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
    ) -> Dict[str, torch.Tensor]:
        if action_candidates.ndim != 4 or action_candidates.shape[1] != 4:
            raise ValueError(
                "V538 action_candidates must be [B,4,H,W], got "
                f"{tuple(action_candidates.shape)}"
            )
        b, _, h, w = action_candidates.shape
        mask_feature = self.mask_encoder(feature)
        mask_logits = self.mask_head(mask_feature)
        raw_slot_masks = torch.sigmoid(mask_logits / self.mask_temperature)
        # V547: keep soft slots independent.  Hard overlap ownership is
        # resolved after thresholding, without diluting every slot probability.
        slot_masks = raw_slot_masks

        flat_masks = slot_masks.reshape(b * self.num_slots, 1, h, w)
        dilated = F.max_pool2d(flat_masks, 3, stride=1, padding=1)
        ring_masks = (dilated - flat_masks).clamp(0.0, 1.0).reshape(
            b, self.num_slots, h, w
        )

        inside_feature = _masked_mean(feature, slot_masks)
        ring_feature = _masked_mean(feature, ring_masks)
        cause_mean = _masked_scalar(cause_probability, slot_masks)
        alpha_mean = _masked_scalar(action_alpha, slot_masks)
        base_mean = _masked_scalar(base_probability, slot_masks)
        entropy_mean = _masked_scalar(entropy, slot_masks)
        boundary_mean = _masked_scalar(boundary, slot_masks)
        area_fraction = slot_masks.mean(dim=(-2, -1)).unsqueeze(-1)

        descriptor = torch.cat(
            [
                inside_feature,
                ring_feature,
                cause_mean,
                alpha_mean,
                base_mean,
                entropy_mean,
                boundary_mean,
                area_fraction,
            ],
            dim=2,
        )
        slot_feature = self.slot_trunk(descriptor)
        action_logits = self.action_head(slot_feature)
        action_probs = F.softmax(action_logits / self.action_temperature, dim=2)
        polarity_logits = self.polarity_head(slot_feature)
        polarity_probs = F.softmax(
            polarity_logits / self.polarity_temperature, dim=2
        )
        dose_logits = self.dose_head(slot_feature).squeeze(-1)
        raw_slot_doses = self.minimum_dose + (
            self.maximum_dose - self.minimum_dose
        ) * torch.sigmoid(dose_logits)
        remove_alpha = 0.5 * (alpha_mean[:, :, 0] + alpha_mean[:, :, 2])
        add_alpha = 0.5 * (alpha_mean[:, :, 1] + alpha_mean[:, :, 3])
        polarity_alpha = (
            polarity_probs[:, :, 0] * remove_alpha
            + polarity_probs[:, :, 1] * add_alpha
        ).clamp(0.0, 1.0)
        dose_gate = 0.25 + 0.75 * polarity_alpha
        slot_doses = (raw_slot_doses * dose_gate).clamp(
            self.minimum_dose * 0.25, self.maximum_dose
        )

        presence_logits = self.presence_head(slot_feature).squeeze(-1)
        presence_probs = torch.sigmoid(presence_logits)

        action_value = torch.einsum("bka,bahw->bkhw", action_probs, action_candidates)
        base = base_probability[:, 0][:, None].detach().clamp(EPS, 1.0 - EPS)
        discrete_slot_candidate_probs = (
            base * (1.0 - slot_masks) + action_value * slot_masks
        ).clamp(EPS, 1.0 - EPS)

        signed_direction = polarity_probs[:, :, 1] - polarity_probs[:, :, 0]
        signed_dose = signed_direction * slot_doses
        continuous_slot_candidate_probs = torch.sigmoid(
            _safe_logit(base) + slot_masks * signed_dose[:, :, None, None]
        ).clamp(EPS, 1.0 - EPS)
        slot_candidate_probs = (
            continuous_slot_candidate_probs
            if self.continuous_dose_enabled
            else discrete_slot_candidate_probs
        )

        if self.adaptive_cardinality_hard_mask:
            hard_masks_bool, _, mask_contrast_active = (
                self._adaptive_hard_mask(slot_masks)
            )
        else:
            hard_masks_bool = (
                slot_masks.detach() >= self.deployment_mask_threshold
            )
            mask_contrast_active = torch.ones(
                slot_masks.shape[:2], device=slot_masks.device, dtype=torch.bool
            )
        hard_masks_bool = self._resolve_hard_slot_overlap(
            mask_logits=mask_logits, hard_masks=hard_masks_bool
        )
        hard_masks_st = (
            hard_masks_bool.to(slot_masks.dtype)
            + slot_masks
            - slot_masks.detach()
        )
        hard_polarity = polarity_logits.detach().argmax(dim=2)
        hard_one_hot = F.one_hot(hard_polarity, num_classes=2).to(
            polarity_probs.dtype
        )
        polarity_st = hard_one_hot + polarity_probs - polarity_probs.detach()
        hard_sign_st = polarity_st[:, :, 1] - polarity_st[:, :, 0]
        hard_sign = hard_sign_st.detach()
        exact_candidate_st_probs = torch.sigmoid(
            _safe_logit(base)
            + hard_masks_st
            * hard_sign_st[:, :, None, None]
            * slot_doses[:, :, None, None]
        ).clamp(EPS, 1.0 - EPS)
        # M2 labels and deployment see the same hard forward value, but M2
        # still receives a detached tensor.  M1 can be trained through the
        # straight-through counterpart exported below.
        exact_candidate_probs = exact_candidate_st_probs.detach()

        selector = self._candidate_selector(
            slot_feature=slot_feature,
            hard_masks=hard_masks_bool,
            exact_candidate=exact_candidate_probs,
            base=base,
            entropy=entropy,
            boundary=boundary,
            presence_probs=presence_probs,
            slot_doses=slot_doses,
            polarity_probs=polarity_probs,
        )

        hard_area_fraction = hard_masks_bool.to(slot_masks.dtype).mean(
            dim=(-2, -1)
        )
        # Trainability is decided by the differentiable soft mass, as in V538.
        # The hard composer separately rejects zero-area masks.  Using hard area
        # here would starve the mask/gain heads before logits cross 0.5.
        validity_area = area_fraction[:, :, 0]
        slot_valid = (
            (validity_area >= self.min_area_fraction)
            & (validity_area <= self.max_component_area_fraction)
        )
        compose_kwargs = dict(
            base_probability=base_probability,
            slot_masks=hard_masks_bool.to(slot_masks.dtype),
            slot_polarity_logits=polarity_logits,
            slot_doses=slot_doses,
            slot_scores=selector["decision_scores"],
            slot_presence=presence_probs,
            slot_valid=slot_valid,
            slot_benefit=selector["benefit_probs"],
            slot_harm=selector["harm_probs"],
            slot_gain_lcb=selector["gain_lcb"],
            slot_editability=selector["editability_probs"],
            slot_direction=selector["direction_probs"],
        )
        shadow_kwargs = dict(compose_kwargs)
        if self.gain_sign_shadow_deploy_enabled:
            shadow_kwargs["slot_scores"] = selector["gain_lcb"]
        shadow_deployment = self._hard_compose(
            **shadow_kwargs,
            deploy_enabled=True,
            gain_sign_only=self.gain_sign_shadow_deploy_enabled,
        )
        deployment = self._hard_compose(
            **compose_kwargs, deploy_enabled=deploy_enabled
        )

        hard_action = torch.where(
            hard_polarity == 0,
            torch.zeros_like(hard_polarity),
            torch.ones_like(hard_polarity),
        )
        hard_masks = hard_masks_bool.to(slot_masks.dtype)
        action_max_scores = selector["decision_scores"].new_full((b, 4), -20.0)
        for action in range(4):
            if action in (0, 2):
                action_valid = slot_valid & (hard_polarity == 0)
            else:
                action_valid = slot_valid & (hard_polarity == 1)
            action_max_scores[:, action] = torch.where(
                action_valid.any(dim=1),
                selector["decision_scores"].masked_fill(~action_valid, -1.0e4).max(dim=1).values,
                action_max_scores[:, action],
            )

        return {
            "slot_mask_logits": mask_logits,
            "slot_masks": slot_masks,
            "slot_raw_masks": raw_slot_masks,
            "slot_competition_enabled": slot_masks.new_tensor(
                1.0 if self.slot_competition_enabled else 0.0
            ),
            "gain_sign_shadow_deploy_enabled": slot_masks.new_tensor(
                1.0 if self.gain_sign_shadow_deploy_enabled else 0.0
            ),
            "slot_raw_overlap_mass": F.relu(
                raw_slot_masks.sum(dim=1) - 1.0
            ).mean(),
            "slot_competition_overlap_mass": F.relu(
                hard_masks_bool.to(slot_masks.dtype).sum(dim=1) - 1.0
            ).mean(),
            "slot_hard_masks": hard_masks_bool.to(slot_masks.dtype),
            "slot_hard_masks_st": hard_masks_st,
            "slot_mask_contrast_active": mask_contrast_active.to(
                slot_masks.dtype
            ),
            "adaptive_cardinality_hard_mask_enabled": slot_masks.new_tensor(
                1.0 if self.adaptive_cardinality_hard_mask else 0.0
            ),
            "slot_action_logits": action_logits,
            "slot_action_probs": action_probs,
            "slot_polarity_logits": polarity_logits,
            "slot_polarity_probs": polarity_probs,
            "slot_dose_logits": dose_logits,
            "slot_raw_doses": raw_slot_doses,
            "slot_dose_gate": dose_gate,
            "slot_doses": slot_doses,
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
            "slot_presence_logits": presence_logits,
            "slot_presence_probs": presence_probs,
            "slot_candidate_probs": slot_candidate_probs,
            "slot_exact_candidate_probs": exact_candidate_probs,
            "slot_exact_candidate_st_probs": exact_candidate_st_probs,
            "slot_discrete_candidate_probs": discrete_slot_candidate_probs,
            "slot_continuous_candidate_probs": continuous_slot_candidate_probs,
            "slot_valid": slot_valid,
            "slot_area_fraction": area_fraction[:, :, 0],
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
            "shadow_selected_final_probability": shadow_deployment["selected_final_probability"],
            "shadow_accepted_slots": shadow_deployment["accepted_slots"],
            "shadow_selected_index": shadow_deployment["selected_index"],
            "shadow_selected_score": shadow_deployment["selected_score"],
            "shadow_predicted_execute": shadow_deployment["predicted_execute"],
            "shadow_accepted_count": shadow_deployment["accepted_count"],
            "shadow_changed_fraction": shadow_deployment["changed_fraction"],
            "action_max_scores": action_max_scores,
            "candidate_masks": hard_masks,
            "candidate_actions": hard_action.detach(),
            "candidate_valid": slot_valid.detach(),
            "candidate_scores": selector["decision_scores"],
            "retained_component_count": slot_valid.float().sum(dim=1),
            "raw_component_count": selector["decision_scores"].new_full(
                (b,), float(self.num_slots)
            ),
        }
