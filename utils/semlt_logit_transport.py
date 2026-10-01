"""Semantic-conditioned Logit Transport (SemLT).

This is the complete M1 model used by the M1-only project.  It deliberately
contains no residual reconstruction, candidate router, critic, STOP policy or
other M2 parameter.  The only intervention is a bounded dense displacement
field applied once to the factual Base logits.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1.0e-4


def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def _groups(channels: int) -> int:
    groups = min(8, max(1, int(channels)))
    while groups > 1 and channels % groups:
        groups -= 1
    return groups


class ConvNormGELU(nn.Sequential):
    def __init__(self, cin: int, cout: int, kernel_size: int = 3) -> None:
        super().__init__(
            nn.Conv2d(cin, cout, kernel_size, padding=kernel_size // 2, bias=False),
            nn.GroupNorm(_groups(cout), cout),
            nn.GELU(),
        )


class BoundedFlowHead(nn.Module):
    """Predict displacement in pixels with a learned but hard-bounded scale."""

    def __init__(self, hidden_dim: int, init_scale_px: float, max_flow_px: float) -> None:
        super().__init__()
        if not (0.0 < init_scale_px <= max_flow_px):
            raise ValueError("SEMLT_FLOW_INIT_SCALE_PX must be in (0, SEMLT_MAX_FLOW_PX]")
        self.max_flow_px = float(max_flow_px)
        fraction = min(max(float(init_scale_px) / self.max_flow_px, 1.0e-4), 1.0 - 1.0e-4)
        self.raw_scale = nn.Parameter(torch.tensor(math.log(fraction / (1.0 - fraction))))
        self.body = nn.Sequential(
            ConvNormGELU(hidden_dim, hidden_dim),
            ConvNormGELU(hidden_dim, hidden_dim),
        )
        self.flow_out = nn.Conv2d(hidden_dim, 2, kernel_size=3, padding=1)
        # Exact identity at initialization.  flow_out receives a live gradient
        # immediately; the upstream encoder becomes live after its first update.
        nn.init.zeros_(self.flow_out.weight)
        nn.init.zeros_(self.flow_out.bias)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = self.max_flow_px * torch.sigmoid(self.raw_scale)
        flow = scale.to(features) * torch.tanh(self.flow_out(self.body(features)))
        return flow, scale


class SemanticLogitTransportSegmenter(nn.Module):
    """One-shot semantic-conditioned transport of detached Base logits."""

    use_semantic_feature = True
    unified_m1_safe_fusion_enabled = True
    mhcs_root_complete = True  # compatibility with the existing outer dispatcher
    mhcs_geometry_topology_refinement = True
    semlt_m1_only = True

    def __init__(self, cfg) -> None:
        super().__init__()
        m1 = _cfg_get(cfg, "M1", None)
        self.hidden_dim = max(32, int(_cfg_get(m1, "SEMLT_HIDDEN_DIM", 128)))
        self.semantic_channels = int(_cfg_get(m1, "SEMANTIC_CHANNELS", 512))
        self.text_dim = int(_cfg_get(m1, "SEMLT_TEXT_DIM", 512))
        max_flow_px = float(_cfg_get(m1, "SEMLT_MAX_FLOW_PX", 8.0))

        self.image_stem = nn.Sequential(
            ConvNormGELU(3, self.hidden_dim),
            ConvNormGELU(self.hidden_dim, self.hidden_dim),
        )
        self.semantic_proj = nn.Sequential(
            nn.Conv2d(self.semantic_channels, self.hidden_dim, 1, bias=False),
            nn.GroupNorm(_groups(self.hidden_dim), self.hidden_dim),
            nn.GELU(),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        # Image, UniMedCLIP semantic map, P0, uncertainty and boundary.
        self.pixel_fuse = nn.Sequential(
            ConvNormGELU(2 * self.hidden_dim + 3, self.hidden_dim),
            ConvNormGELU(self.hidden_dim, self.hidden_dim),
        )
        self.global_context = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.context_film = nn.Linear(self.hidden_dim, 2 * self.hidden_dim)
        self.context_gate = nn.Parameter(torch.zeros(()))
        self.distribution_trunk = nn.Sequential(
            ConvNormGELU(self.hidden_dim, self.hidden_dim),
            ConvNormGELU(self.hidden_dim, self.hidden_dim),
        )
        self.flow_head = BoundedFlowHead(
            self.hidden_dim,
            init_scale_px=float(_cfg_get(m1, "SEMLT_FLOW_INIT_SCALE_PX", 1.0)),
            max_flow_px=max_flow_px,
        )

    @staticmethod
    def _resize(value: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        if tuple(value.shape[-2:]) == tuple(hw):
            return value
        return F.interpolate(value, size=hw, mode="bilinear", align_corners=False)

    @staticmethod
    def _boundary(probability: torch.Tensor) -> torch.Tensor:
        maximum = F.max_pool2d(probability, 3, stride=1, padding=1)
        minimum = -F.max_pool2d(-probability, 3, stride=1, padding=1)
        return (maximum - minimum).clamp(0.0, 1.0)

    @staticmethod
    def _warp_logits(base_logits: torch.Tensor, flow_px: torch.Tensor) -> torch.Tensor:
        _, _, height, width = base_logits.shape
        dtype, device = base_logits.dtype, base_logits.device
        yy = torch.arange(height, device=device, dtype=dtype) + 0.5
        xx = torch.arange(width, device=device, dtype=dtype) + 0.5
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        sample_x = gx[None] + flow_px[:, 0]
        sample_y = gy[None] + flow_px[:, 1]
        grid = torch.stack(
            [2.0 * sample_x / float(width) - 1.0, 2.0 * sample_y / float(height) - 1.0],
            dim=-1,
        )
        return F.grid_sample(
            base_logits,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

    @staticmethod
    def _jacobian(flow_px: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = flow_px.shape[0]
        if min(flow_px.shape[-2:]) < 2:
            determinant = flow_px.new_ones((batch, 1, 1))
            return determinant, flow_px.new_zeros(batch)
        ux, uy = flow_px[:, 0], flow_px[:, 1]
        dux_dx = ux[:, :-1, 1:] - ux[:, :-1, :-1]
        dux_dy = ux[:, 1:, :-1] - ux[:, :-1, :-1]
        duy_dx = uy[:, :-1, 1:] - uy[:, :-1, :-1]
        duy_dy = uy[:, 1:, :-1] - uy[:, :-1, :-1]
        determinant = (1.0 + dux_dx) * (1.0 + duy_dy) - dux_dy * duy_dx
        return determinant, (determinant <= 0.0).float().mean(dim=(1, 2))

    def generate(
        self,
        base_logits: torch.Tensor,
        image: torch.Tensor,
        semantic_map: Optional[torch.Tensor],
        text_features: torch.Tensor,
        negative_text_features: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del negative_text_features, kwargs
        if semantic_map is None:
            raise RuntimeError("SemLT requires the UniMedCLIP spatial semantic map")
        if base_logits.ndim == 3:
            base_logits = base_logits[:, None]
        if base_logits.ndim != 4 or base_logits.shape[1] != 1:
            raise ValueError(f"Expected Base logits [B,1,H,W], got {tuple(base_logits.shape)}")

        # M1 learns a conditional transformation of a factual Base prediction;
        # its loss may not rewrite the Base/PVL representation.
        factual_logits = base_logits.detach()
        factual_prob = torch.sigmoid(factual_logits).clamp(EPS, 1.0 - EPS)
        hw = tuple(factual_prob.shape[-2:])
        image_latent = self.image_stem(self._resize(image.detach(), hw))
        semantic_latent = self.semantic_proj(self._resize(semantic_map.detach(), hw))
        text_latent = self.text_proj(text_features.detach().float())
        uncertainty = (4.0 * factual_prob * (1.0 - factual_prob)).clamp(0.0, 1.0)
        boundary = self._boundary(factual_prob)
        pixel = self.pixel_fuse(
            torch.cat([image_latent, semantic_latent, factual_prob, uncertainty, boundary], dim=1)
        )
        visual_global = F.adaptive_avg_pool2d(pixel, 1).flatten(1)
        context = self.global_context(torch.cat([visual_global, text_latent], dim=1))
        gamma, beta = self.context_film(context).chunk(2, dim=1)
        gate = torch.tanh(self.context_gate)
        pixel = pixel + gate * (
            torch.tanh(gamma)[:, :, None, None] * pixel + beta[:, :, None, None]
        )
        flow_px, learned_scale_px = self.flow_head(self.distribution_trunk(pixel))
        transported_logits = self._warp_logits(factual_logits, flow_px)
        transported_prob = torch.sigmoid(transported_logits).clamp(EPS, 1.0 - EPS)

        candidate_logits = torch.cat([factual_logits, transported_logits], dim=1)
        candidate_probs = torch.cat([factual_prob, transported_prob], dim=1)
        flow_magnitude = torch.linalg.vector_norm(flow_px, dim=1)
        jacobian, folding_fraction = self._jacobian(flow_px)
        batch = factual_logits.shape[0]
        zeros = factual_logits.new_zeros(batch)
        ones = factual_logits.new_ones(batch)

        # Compatibility aliases all point to the same physical M1 output.  No
        # alias owns parameters and no M2 computation exists in this module.
        aux: Dict[str, torch.Tensor] = {
            "candidate_probs": candidate_probs,
            "candidates": candidate_logits,
            "mhcs_final_logits": transported_logits,
            "mhcs_final_probs": transported_prob,
            "mhcs_local_probs": transported_prob,
            "mhcs_surface_hard_probs": transported_prob,
            "mhcs_global_selected_probs": transported_prob,
            "direct_fused_probs": transported_prob,
            "router_fused_probs": transported_prob,
            "v20_fused_probs": transported_prob,
            "v20_hard_fused_probs": transported_prob,
            "v20_fused_logits": transported_logits,
            "v20_selector_hard": factual_logits.new_ones((batch, 1)),
            "geotopo_base_logits": factual_logits,
            "geotopo_base_probs": factual_prob,
            "geotopo_geometry_logits": transported_logits,
            "geotopo_geometry_probs": transported_prob,
            "geotopo_residual_only_logits": factual_logits,
            "geotopo_residual_only_probs": factual_prob,
            "geotopo_reconstruction_after_geometry_logits": transported_logits,
            "geotopo_reconstruction_after_geometry_probs": transported_prob,
            "geotopo_final_logits": transported_logits,
            "geotopo_final_probs": transported_prob,
            "geotopo_flow_px": flow_px,
            "geotopo_flow_rms_px": flow_magnitude.square().mean(dim=(1, 2)).sqrt(),
            "geotopo_flow_mean_px": flow_magnitude.mean(dim=(1, 2)),
            "geotopo_flow_max_px": flow_magnitude.flatten(1).max(dim=1).values,
            "geotopo_flow_jacobian_mean": jacobian.mean(dim=(1, 2)),
            "geotopo_flow_folding_fraction": folding_fraction,
            "geotopo_mode_id": ones,
            "mhcs_effective_rank": ones,
            "mhcs_gate_alpha": ones,
            "mhcs_quality_probs": factual_logits.new_zeros((batch, 1)),
            "mhcs_surface_nonbase_mass": (transported_prob - factual_prob).abs().mean(dim=(1, 2, 3)),
            "semlt_base_logits": factual_logits,
            "semlt_final_logits": transported_logits,
            "semlt_final_probs": transported_prob,
            "semlt_flow_px": flow_px,
            "semlt_flow_scale_px": learned_scale_px.expand(batch),
            "semlt_jacobian_determinant": jacobian,
            "semlt_folding_fraction": folding_fraction,
            "semlt_has_m2": zeros,
        }
        return candidate_logits, aux
