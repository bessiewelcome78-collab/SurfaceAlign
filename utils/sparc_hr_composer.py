"""SPARC-HR3.1: safety-constrained dense counterfactual routing.

M2 retains the useful transported-posterior/HR actors but replaces the
sample-inefficient multi-step region policy.  A shared convolutional router
chooses one source per HR grid cell.  Source zero is unchanged M1, therefore
WHERE, HOW and STOP are a single categorical decision.  HR3.1 removes the
biased straight-through gradient through the hard source mosaic: the router is
trained by explicit attainable-advantage supervision, while actors are trained
by their own segmentation/action objectives.  No-op/duplicate candidates are
masked before softmax and every edit is confined to an M1 boundary band.
Ground truth is never read here; it is only copied for the external loss.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1.0e-4


def _groups(channels: int) -> int:
    groups = min(8, max(1, int(channels)))
    while groups > 1 and channels % groups:
        groups -= 1
    return groups


class _ConvNormAct(nn.Sequential):
    def __init__(self, cin: int, cout: int, kernel_size: int = 3) -> None:
        super().__init__(
            nn.Conv2d(cin, cout, kernel_size, padding=kernel_size // 2, bias=False),
            nn.GroupNorm(_groups(cout), cout),
            nn.GELU(),
        )


class StructuredPosteriorAtomicHRComposer(nn.Module):
    """GT-free one-shot dense HR composition with M1 as reject expert."""

    def __init__(
        self,
        transport_hidden_dim: int,
        semantic_channels: int,
        text_dim: int,
        *,
        fine_feature_channels: int,
        hidden_dim: int = 48,
        max_posterior_sources: int = 6,
        require_true_hr: bool = True,
        minimum_crossing_margin: float = 0.05,
        typed_gate_init_bias: float = -2.0,
        router_grid_size: int = 112,
        router_temperature: float = 0.5,
        router_anchor_prior: float = 1.0,
        router_family_embedding_dim: int = 16,
        min_action_change_hr_pixels: int = 1,
        prune_duplicate_actions: bool = True,
        boundary_band_radius_hr: int = 12,
        router_edit_logit_margin: float = 1.5,
        router_edit_probability_threshold: float = 0.70,
        **_: object,
    ) -> None:
        super().__init__()
        del transport_hidden_dim
        self.hidden_dim = max(16, int(hidden_dim))
        self.max_posterior_sources = max(0, int(max_posterior_sources))
        self.require_true_hr = bool(require_true_hr)
        self.minimum_crossing_margin = min(0.24, max(1.0e-3, float(minimum_crossing_margin)))
        self.router_grid_size = max(28, int(router_grid_size))
        self.router_temperature = max(0.05, float(router_temperature))
        self.router_anchor_prior = max(0.0, float(router_anchor_prior))
        self.min_action_change_hr_pixels = max(1, int(min_action_change_hr_pixels))
        self.prune_duplicate_actions = bool(prune_duplicate_actions)
        self.boundary_band_radius_hr = max(1, int(boundary_band_radius_hr))
        self.router_edit_logit_margin = max(0.0, float(router_edit_logit_margin))
        self.router_edit_probability_threshold = min(
            1.0, max(0.0, float(router_edit_probability_threshold))
        )
        self.force_preserve = False

        self.image_encoder = nn.Sequential(
            _ConvNormAct(3, self.hidden_dim),
            _ConvNormAct(self.hidden_dim, self.hidden_dim),
        )
        # A cheap two-level context path increases the HR receptive field while
        # retaining the native image stream needed for speckle/boundary detail.
        self.image_context = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(self.hidden_dim), self.hidden_dim), nn.GELU(),
            _ConvNormAct(self.hidden_dim, self.hidden_dim),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(self.hidden_dim), self.hidden_dim), nn.GELU(),
            _ConvNormAct(self.hidden_dim, self.hidden_dim),
        )
        self.semantic_proj = nn.Sequential(
            nn.Conv2d(semantic_channels, self.hidden_dim, 1, bias=False),
            nn.GroupNorm(_groups(self.hidden_dim), self.hidden_dim), nn.GELU(),
        )
        self.fine_proj = nn.Sequential(
            nn.Conv2d(fine_feature_channels, self.hidden_dim, 1, bias=False),
            nn.GroupNorm(_groups(self.hidden_dim), self.hidden_dim), nn.GELU(),
        )
        self.evidence_encoder = nn.Sequential(
            _ConvNormAct(8, self.hidden_dim), _ConvNormAct(self.hidden_dim, self.hidden_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim), nn.GELU(),
        )
        self.fusion = nn.Sequential(
            _ConvNormAct(4 * self.hidden_dim, self.hidden_dim),
            _ConvNormAct(self.hidden_dim, self.hidden_dim),
        )
        # Absolute proposal: no anchor-logit reachability ceiling.
        self.proposal_head = nn.Sequential(
            _ConvNormAct(self.hidden_dim, self.hidden_dim), nn.Conv2d(self.hidden_dim, 1, 1),
        )
        self.typed_gate_head = nn.Sequential(
            _ConvNormAct(self.hidden_dim, self.hidden_dim), nn.Conv2d(self.hidden_dim, 2, 1),
        )
        self.typed_dose_head = nn.Sequential(
            _ConvNormAct(self.hidden_dim, self.hidden_dim), nn.Conv2d(self.hidden_dim, 2, 1),
        )
        family_dim = max(4, int(router_family_embedding_dim))
        self.source_family_embedding = nn.Embedding(5, family_dim)
        self.source_family_proj = nn.Linear(family_dim, self.hidden_dim)
        # context + candidate/current/delta/abs-delta/boundary-difference.
        self.router_candidate_encoder = nn.Sequential(
            _ConvNormAct(self.hidden_dim + 5, self.hidden_dim),
            _ConvNormAct(self.hidden_dim, self.hidden_dim),
        )
        self.router_score_head = nn.Conv2d(self.hidden_dim, 1, 1)

        nn.init.zeros_(self.proposal_head[-1].weight)
        nn.init.zeros_(self.proposal_head[-1].bias)
        nn.init.zeros_(self.typed_gate_head[-1].weight)
        nn.init.constant_(self.typed_gate_head[-1].bias, float(typed_gate_init_bias))
        nn.init.zeros_(self.typed_dose_head[-1].weight)
        nn.init.zeros_(self.typed_dose_head[-1].bias)
        nn.init.zeros_(self.router_score_head.weight)
        nn.init.zeros_(self.router_score_head.bias)

    @staticmethod
    def _resize(x: torch.Tensor, hw: Tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
        if tuple(x.shape[-2:]) == tuple(hw):
            return x
        if mode in {"nearest", "area"}:
            return F.interpolate(x, size=hw, mode=mode)
        return F.interpolate(x, size=hw, mode=mode, align_corners=False)

    @staticmethod
    def _boundary(prob: torch.Tensor) -> torch.Tensor:
        return (F.max_pool2d(prob, 3, 1, 1) + F.max_pool2d(-prob, 3, 1, 1)).clamp(0.0, 1.0)

    @staticmethod
    def _normalise_posterior(
        posterior: Optional[torch.Tensor], anchor: torch.Tensor
    ) -> torch.Tensor:
        if not isinstance(posterior, torch.Tensor):
            raise ValueError("SPARC-HR3 requires transported posterior_probability_samples")
        samples = posterior.detach()
        if samples.ndim == 4:
            samples = samples.unsqueeze(2)
        if samples.ndim != 5 or samples.shape[1] != anchor.shape[0] or samples.shape[2] != 1:
            raise ValueError("posterior_probability_samples must be [S,B,H,W] or [S,B,1,H,W]")
        if samples.shape[0] < 2:
            raise ValueError("SPARC-HR3 requires at least two posterior samples")
        if tuple(samples.shape[-2:]) != tuple(anchor.shape[-2:]):
            s, b = samples.shape[:2]
            samples = F.interpolate(
                samples.reshape(s * b, 1, *samples.shape[-2:]),
                size=anchor.shape[-2:], mode="bilinear", align_corners=False,
            ).reshape(s, b, 1, *anchor.shape[-2:])
        return samples.clamp(EPS, 1.0 - EPS)

    def _select_diverse_views(self, samples: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """Deterministic farthest-point subsampling; zero retains all draws."""
        s, b = samples.shape[:2]
        keep = s if self.max_posterior_sources <= 0 else min(s, self.max_posterior_sources)
        if keep == s:
            return samples.permute(1, 0, 2, 3, 4).contiguous()
        reduced = F.adaptive_avg_pool2d(samples.reshape(s * b, 1, *samples.shape[-2:]), 16)
        reduced = reduced.reshape(s, b, -1).permute(1, 0, 2)
        anchor_r = F.adaptive_avg_pool2d(anchor, 16).flatten(1)
        chosen = []
        for bi in range(b):
            first = int((reduced[bi] - anchor_r[bi]).square().mean(1).argmax().item())
            indices = [first]
            min_dist = (reduced[bi] - reduced[bi, first]).square().mean(1)
            for _ in range(1, keep):
                min_dist[indices] = -1.0
                nxt = int(min_dist.argmax().item())
                indices.append(nxt)
                min_dist = torch.minimum(min_dist, (reduced[bi] - reduced[bi, nxt]).square().mean(1))
            chosen.append(samples[indices, bi])
        return torch.stack(chosen, dim=0)

    def _candidate_validity(
        self, sources_hr: torch.Tensor, grid: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return deployment-valid, duplicate and changed-pixel maps.

        Validity is measured in the exact hard deployment space.  This is the
        implementation of the no-op/duplicate contract that earlier configs
        declared but the HR3 forward path never applied.
        """
        b, j, _, hh, hw = sources_hr.shape
        hard = sources_hr >= 0.5
        anchor = hard[:, :1]
        changed = (hard != anchor).float()
        changed_fraction = F.adaptive_avg_pool2d(
            changed.reshape(b * j, 1, hh, hw), grid
        ).reshape(b, j, *grid)
        cell_area = float(hh * hw) / float(grid[0] * grid[1])
        changed_count = changed_fraction * cell_area
        valid = changed_count >= float(self.min_action_change_hr_pixels)
        valid[:, 0] = True

        duplicate = torch.zeros_like(valid)
        if self.prune_duplicate_actions:
            for source_index in range(1, j):
                for earlier_index in range(source_index):
                    disagreement = (hard[:, source_index] != hard[:, earlier_index]).float()
                    same = F.adaptive_avg_pool2d(disagreement, grid)[:, 0] <= 0.0
                    duplicate[:, source_index] |= same
            duplicate[:, 0] = False
            valid &= ~duplicate
        return valid, duplicate, changed_count

    def _router(
        self, features_hr: torch.Tensor, sources_hr: torch.Tensor, family_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, j, _, hh, hw = sources_hr.shape
        grid = (min(self.router_grid_size, hh), min(self.router_grid_size, hw))
        context = self._resize(features_hr, grid)
        source_grid = F.adaptive_avg_pool2d(sources_hr.reshape(b * j, 1, hh, hw), grid)
        source_grid = source_grid.reshape(b, j, 1, *grid)
        anchor_grid = source_grid[:, :1].expand(-1, j, -1, -1, -1)
        delta = source_grid - anchor_grid
        source_boundary = self._boundary(source_grid.reshape(b * j, 1, *grid)).reshape(b, j, 1, *grid)
        anchor_boundary = self._boundary(anchor_grid[:, 0])[:, None]
        maps = torch.cat(
            [source_grid, anchor_grid, delta, delta.abs(), (source_boundary - anchor_boundary).abs()], dim=2
        )
        inputs = torch.cat([context[:, None].expand(-1, j, -1, -1, -1), maps], dim=2)
        encoded = self.router_candidate_encoder(inputs.reshape(b * j, self.hidden_dim + 5, *grid))
        family = self.source_family_proj(self.source_family_embedding(family_ids))[None, :, :, None, None]
        encoded = encoded.reshape(b, j, self.hidden_dim, *grid) + family
        logits = self.router_score_head(encoded.reshape(b * j, self.hidden_dim, *grid)).reshape(b, j, *grid)
        anchor_prior = logits.new_zeros((1, j, 1, 1))
        anchor_prior[:, 0] = self.router_anchor_prior
        logits = logits + anchor_prior
        valid, duplicate, changed_count = self._candidate_validity(sources_hr, grid)
        logits = logits.masked_fill(~valid, -1.0e4)
        soft = F.softmax(logits / self.router_temperature, dim=1)
        if j > 1:
            best_edit_logit, best_edit_offset = logits[:, 1:].max(dim=1)
            best_edit_index = best_edit_offset + 1
            best_edit_probability = soft.gather(1, best_edit_index[:, None])[:, 0]
            edit_margin = best_edit_logit - logits[:, 0]
            execute = (
                valid.gather(1, best_edit_index[:, None])[:, 0]
                & (edit_margin >= self.router_edit_logit_margin)
                & (best_edit_probability >= self.router_edit_probability_threshold)
            )
            selected = torch.where(execute, best_edit_index, torch.zeros_like(best_edit_index))
        else:
            selected = torch.zeros((b, *grid), device=logits.device, dtype=torch.long)
            best_edit_probability = logits.new_zeros((b, *grid))
            edit_margin = logits.new_full((b, *grid), -torch.inf)
        hard = F.one_hot(selected, num_classes=j).permute(0, 3, 1, 2).to(soft)
        # Do not use hard + soft - stopgrad(soft). Its backward value optimises a
        # soft mixture while deployment uses a discontinuous source mosaic; in
        # the observed run this produced 10^3--10^4 pre-clip router gradients.
        weights = hard
        return (
            logits, soft, weights, selected, valid, duplicate, changed_count,
            best_edit_probability, edit_margin,
        )

    def forward(
        self,
        anchor_prob: torch.Tensor,
        image: torch.Tensor,
        semantic_map: torch.Tensor,
        text_features: torch.Tensor,
        *,
        base_prob: torch.Tensor,
        flow_px: torch.Tensor,
        mc_std_map: Optional[torch.Tensor] = None,
        mc_disagreement_map: Optional[torch.Tensor] = None,
        fine_feature_map: Optional[torch.Tensor] = None,
        posterior_probability_samples: Optional[torch.Tensor] = None,
        hr_image: Optional[torch.Tensor] = None,
        hr_target: Optional[torch.Tensor] = None,
        **_: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        del image
        anchor = anchor_prob.detach().clamp(EPS, 1.0 - EPS)
        base = base_prob.detach().clamp(EPS, 1.0 - EPS)
        flow = flow_px.detach()
        samples = self._normalise_posterior(posterior_probability_samples, anchor)
        views_lr = self._select_diverse_views(samples, anchor)
        if not isinstance(hr_image, torch.Tensor):
            if self.require_true_hr:
                raise ValueError("SPARC-HR3 requires paired slr_hr_image")
            hr_image = self._resize(anchor.expand(-1, 3, -1, -1),
                                    (2 * anchor.shape[-2], 2 * anchor.shape[-1]))
        hr_image = hr_image.detach()
        lr_hw, hr_hw = tuple(anchor.shape[-2:]), tuple(hr_image.shape[-2:])
        if self.require_true_hr and hr_hw != (2 * lr_hw[0], 2 * lr_hw[1]):
            raise ValueError(f"SPARC-HR3 expects exact 2x HR registration, got LR={lr_hw}, HR={hr_hw}")
        target_view = hr_target
        if isinstance(target_view, torch.Tensor):
            if target_view.ndim == 3:
                target_view = target_view[:, None]
            if target_view.ndim != 4 or target_view.shape[:2] != (anchor.shape[0], 1):
                raise ValueError("HR target must be [B,H,W] or [B,1,H,W]")
            if tuple(target_view.shape[-2:]) != hr_hw:
                raise ValueError("HR target/image spatial mismatch")

        anchor_hr, base_hr = self._resize(anchor, hr_hw), self._resize(base, hr_hw)
        b, j_post = views_lr.shape[:2]
        views_hr = self._resize(views_lr.reshape(b * j_post, 1, *lr_hw), hr_hw)
        views_hr = views_hr.reshape(b, j_post, 1, *hr_hw).clamp(EPS, 1.0 - EPS)
        posterior_mean = views_lr.mean(dim=1)
        posterior_std = views_lr.std(dim=1, unbiased=False)
        posterior_disagreement = ((views_lr >= 0.5) != (posterior_mean[:, None] >= 0.5)).float().mean(dim=1)
        if isinstance(mc_std_map, torch.Tensor):
            posterior_std = torch.maximum(posterior_std, self._resize(mc_std_map.detach(), lr_hw))
        if isinstance(mc_disagreement_map, torch.Tensor):
            posterior_disagreement = torch.maximum(
                posterior_disagreement, self._resize(mc_disagreement_map.detach(), lr_hw)
            )
        margin = (1.0 - 2.0 * (anchor - 0.5).abs()).clamp(0.0, 1.0)
        boundary = self._boundary(anchor)
        evidence_lr = torch.cat(
            [anchor, posterior_mean, posterior_std, posterior_disagreement,
             margin, boundary, self._resize(flow, lr_hw)], dim=1
        )
        image_features = self.image_encoder(hr_image)
        image_features = image_features + self._resize(
            self.image_context(image_features), hr_hw
        )
        semantic_features = self._resize(self.semantic_proj(semantic_map.detach()), hr_hw)
        fine_features = (
            self._resize(self.fine_proj(fine_feature_map.detach()), hr_hw)
            if isinstance(fine_feature_map, torch.Tensor) else torch.zeros_like(semantic_features)
        )
        evidence_features = self._resize(self.evidence_encoder(evidence_lr), hr_hw)
        features_hr = self.fusion(torch.cat(
            [image_features, semantic_features, fine_features, evidence_features], dim=1
        )) + self.text_proj(text_features.detach())[:, :, None, None]

        proposal_logits = self.proposal_head(features_hr)
        proposal_prob = torch.sigmoid(proposal_logits).clamp(EPS, 1.0 - EPS)
        gate_logits = self.typed_gate_head(features_hr)
        gate_soft = torch.sigmoid(gate_logits)
        gate_hard = (gate_soft >= 0.5).to(gate_soft)
        gate = gate_hard + gate_soft - gate_soft.detach() if self.training else gate_hard
        dose = torch.sigmoid(self.typed_dose_head(features_hr))
        add_value = 0.5 + self.minimum_crossing_margin + (0.5 - self.minimum_crossing_margin - EPS) * dose[:, :1]
        remove_value = (0.5 - self.minimum_crossing_margin) * (1.0 - dose[:, 1:2])
        add_target = torch.maximum(anchor_hr, add_value)
        remove_target = torch.minimum(anchor_hr, remove_value)
        anchor_binary = (anchor_hr >= 0.5).to(anchor_hr)
        hard_boundary = self._boundary(anchor_binary)
        radius = min(self.boundary_band_radius_hr, max(1, min(hr_hw) // 8))
        boundary_band = F.max_pool2d(
            (hard_boundary > 0.0).to(anchor_hr), 2 * radius + 1, 1, radius
        ).clamp(0.0, 1.0)
        proposal_local = (
            anchor_hr * (1.0 - boundary_band) + proposal_prob * boundary_band
        ).clamp(EPS, 1.0 - EPS)
        add_gate = gate[:, :1] * boundary_band
        remove_gate = gate[:, 1:2] * boundary_band
        typed_add = (anchor_hr * (1.0 - add_gate) + add_target * add_gate).clamp(EPS, 1.0 - EPS)
        typed_remove = (anchor_hr * (1.0 - remove_gate) + remove_target * remove_gate).clamp(EPS, 1.0 - EPS)
        views_hr = (
            anchor_hr[:, None] * (1.0 - boundary_band[:, None])
            + views_hr * boundary_band[:, None]
        ).clamp(EPS, 1.0 - EPS)

        sources_hr = torch.cat(
            [anchor_hr[:, None], views_hr, proposal_local[:, None], typed_add[:, None], typed_remove[:, None]], dim=1
        )
        family_ids = torch.tensor([0] + [1] * j_post + [2, 3, 4], device=anchor.device, dtype=torch.long)
        (
            router_logits, router_soft, weights_grid, selected, router_valid,
            router_duplicate, router_changed_count, router_edit_probability,
            router_edit_margin,
        ) = self._router(features_hr, sources_hr, family_ids)
        if self.force_preserve:
            selected = torch.zeros_like(selected)
            weights_grid = F.one_hot(selected, num_classes=sources_hr.shape[1]).permute(0, 3, 1, 2).to(router_soft)
        weights_hr = self._resize(weights_grid, hr_hw, mode="nearest")
        final_hr = (weights_hr[:, :, None] * sources_hr).sum(dim=1).clamp(EPS, 1.0 - EPS)
        final_prob = (anchor + self._resize(final_hr - anchor_hr, lr_hw, mode="area")).clamp(EPS, 1.0 - EPS)
        final_logits = torch.logit(final_prob)
        changed_grid = selected != 0
        selection_mask = self._resize(
            self._resize(changed_grid[:, None].to(anchor), hr_hw, mode="nearest"), lr_hw, mode="area"
        )
        entropy = -(anchor * anchor.log() + (1.0 - anchor) * (1.0 - anchor).log())
        zero = torch.zeros_like(anchor)
        output: Dict[str, torch.Tensor] = {
            "logits": final_logits, "prob": final_prob,
            "selection_mask": selection_mask,
            "selection_score": self._resize(1.0 - router_soft[:, :1], lr_hw),
            "refined_logits": final_logits, "refined_prob": final_prob,
            "delta_logit": final_logits - torch.logit(anchor),
            "margin_uncertainty": margin, "mc_std_map": posterior_std,
            "mc_disagreement_map": posterior_disagreement, "entropy_map": entropy,
            "dn_anchor_prob": anchor, "dn_corruption_mask": zero, "dn_selection_mask": zero,
            "dn_refined_logits": torch.logit(anchor), "dn_refined_prob": anchor,
            "dn_final_prob": anchor, "dn_delta_logit": zero,
            "r4_flip_logits": zero, "r4_flip_prob": zero, "r4_flip_mask": zero,
            "r4_synth_flip_logits": zero, "r4_synth_flip_prob": zero,
            "r4_synth_flip_mask": zero, "r4_synth_target": zero,
            "sparc_enabled": anchor.new_ones((b,)),
            "sparc_force_preserve": anchor.new_full((b,), float(bool(self.force_preserve))),
            "sparc_protocol_version": anchor.new_full((b,), 3.1),
            "sparc_base_prob_hr": base_hr, "sparc_anchor_prob_hr": anchor_hr,
            "sparc_final_prob_hr": final_hr, "sparc_final_logits_hr": torch.logit(final_hr),
            "sparc_learned_proposal_logits_hr": proposal_logits,
            "sparc_learned_proposal_prob_hr": proposal_local,
            "sparc_unrestricted_proposal_prob_hr": proposal_prob,
            "sparc_proposal_delta_logit_hr": torch.logit(proposal_local) - torch.logit(anchor_hr),
            "sparc_typed_gate_logits_hr": gate_logits, "sparc_typed_gate_prob_hr": gate_soft,
            "sparc_typed_gate_hard_hr": gate_hard, "sparc_typed_dose_hr": dose,
            "sparc_typed_add_prob_hr": typed_add, "sparc_typed_remove_prob_hr": typed_remove,
            "sparc_posterior_views_hr": views_hr, "sparc_source_probs_hr": sources_hr,
            "sparc_source_family_ids": family_ids, "sparc_router_logits": router_logits,
            "sparc_router_soft_weights_grid": router_soft,
            "sparc_router_weights_grid": weights_grid, "sparc_router_weights_hr": weights_hr,
            "sparc_router_selected_source": selected, "sparc_router_changed_grid": changed_grid,
            "sparc_router_valid_mask": router_valid,
            "sparc_router_duplicate_mask": router_duplicate,
            "sparc_router_changed_pixel_count": router_changed_count,
            "sparc_router_edit_probability": router_edit_probability,
            "sparc_router_edit_logit_margin": router_edit_margin,
            "sparc_boundary_edit_band_hr": boundary_band,
            "sparc_execute": changed_grid.flatten(1).any(1, keepdim=True),
            "sparc_source_count": anchor.new_full((b,), float(sources_hr.shape[1])),
            "sparc_posterior_source_count": anchor.new_full((b,), float(j_post)),
        }
        if isinstance(target_view, torch.Tensor):
            output["sparc_hr_target"] = target_view.detach()
        return output
