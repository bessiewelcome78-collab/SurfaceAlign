"""Uncertainty-Gated Boundary Resonance Adapter (UGBRA).

A lightweight decoder-side feature adapter for MedCLIPSeg/JBT-Lite.

Design goals
------------
1. Do not introduce a second semantic branch.
2. Operate only on the existing decoder feature map.
3. Use the model's own coarse prediction to decide *where* local geometric
   detail is worth refining.
4. Recover local high-frequency evidence that ViT patchification and progressive
   upsampling cannot reconstruct on their own, including a gated raw-image edge cue.
5. Start as an exact identity mapping (ReZero-style scalar gamma = 0), so the
   pretrained/Base trajectory is not abruptly perturbed at initialization.
6. Require no new auxiliary loss. Existing Dice/CE and the early Surface
   Alignment objective supervise the adapter through the final mask logits.

The module is intentionally small: two bottleneck decoder high-pass branches
at local scales 3 and 5, one fixed raw-image edge cue with a tiny projection,
one spatial gate, and one per-case source selector.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _HighPassResidualBranch(nn.Module):
    """Bottleneck branch that transforms a local high-pass residual."""

    def __init__(self, channels: int, hidden_channels: int, pool_kernel: int) -> None:
        super().__init__()
        if pool_kernel < 3 or pool_kernel % 2 == 0:
            raise ValueError("pool_kernel must be an odd integer >= 3")
        self.pool_kernel = int(pool_kernel)
        self.reduce = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False)
        self.depthwise = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
            bias=False,
        )
        # GroupNorm is batch-size independent and stable for medical datasets.
        groups = min(8, hidden_channels)
        while groups > 1 and hidden_channels % groups != 0:
            groups -= 1
        self.norm = nn.GroupNorm(groups, hidden_channels)
        self.act = nn.GELU()
        self.expand = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def high_pass(self, x: torch.Tensor) -> torch.Tensor:
        k = self.pool_kernel
        low = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
        return x - low

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        high = self.high_pass(x)
        residual = self.expand(self.act(self.norm(self.depthwise(self.reduce(high)))))
        return residual, high


class UncertaintyGatedBoundaryResonanceAdapter(nn.Module):
    """Refine decoder features only around the model's uncertain contour.

    Parameters
    ----------
    channels:
        Decoder feature channels (512 in UniMedCLIP ViT-B/16 MedCLIPSeg).
    reduction:
        Bottleneck reduction factor. ``8`` gives a 64-channel internal branch.
    gamma_init:
        Residual LayerScale/ReZero coefficient. ``0`` makes the first forward
        exactly identical to the original decoder.
    detach_uncertainty:
        Stop gradients through the coarse uncertainty gate. This prevents a
        degenerate feedback loop in which the coarse predictor changes its
        confidence merely to open/close the refinement gate.
    gate_floor:
        Small non-zero floor multiplied into the uncertainty gate. A value of
        0.05 keeps tiny learning signal outside the exact p=0.5 contour while
        preserving strong boundary localization.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 8,
        gamma_init: float = 0.0,
        detach_uncertainty: bool = True,
        gate_floor: float = 0.05,
        use_image_edge: bool = True,
    ) -> None:
        super().__init__()
        channels = int(channels)
        reduction = max(1, int(reduction))
        hidden = max(16, channels // reduction)
        self.channels = channels
        self.hidden_channels = hidden
        self.detach_uncertainty = bool(detach_uncertainty)
        self.gate_floor = float(max(0.0, min(0.5, gate_floor)))
        self.use_image_edge = bool(use_image_edge)

        self.branch3 = _HighPassResidualBranch(channels, hidden, pool_kernel=3)
        self.branch5 = _HighPassResidualBranch(channels, hidden, pool_kernel=5)
        image_groups = min(8, hidden)
        while image_groups > 1 and hidden % image_groups != 0:
            image_groups -= 1
        self.image_edge_proj = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(image_groups, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

        # Spatial gate sees only geometry-like signals: coarse uncertainty,
        # coarse contour strength, raw-image edge strength, and decoder high-pass
        # magnitudes. It does not introduce a new text/semantic expert.
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(5, 8, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(8, 1, kernel_size=1, bias=True),
        )

        # Per-case scale selector chooses sharper (3x3) vs broader (5x5)
        # residual evidence. Two scalar descriptors -> two mixture weights.
        self.scale_selector = nn.Sequential(
            nn.Linear(3, 8),
            nn.GELU(),
            nn.Linear(8, 3),
        )
        # Start the learned spatial gate at 0.5 and the source mixture uniformly.
        # This removes arbitrary random preference before task gradients arrive.
        nn.init.zeros_(self.spatial_gate[-1].weight)
        nn.init.zeros_(self.spatial_gate[-1].bias)
        nn.init.zeros_(self.scale_selector[-1].weight)
        nn.init.zeros_(self.scale_selector[-1].bias)

        # Exact identity at initialization. At step 1 only gamma learns; once it
        # departs from zero, the residual branches receive gradients as well.
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    @staticmethod
    def _coarse_uncertainty(coarse_logits: torch.Tensor) -> torch.Tensor:
        if coarse_logits.ndim == 3:
            coarse_logits = coarse_logits[:, None]
        if coarse_logits.ndim != 4 or coarse_logits.shape[1] != 1:
            raise ValueError(
                "coarse_logits must be [B,H,W] or [B,1,H,W], got "
                f"{tuple(coarse_logits.shape)}"
            )
        p = torch.sigmoid(coarse_logits)
        # Normalized Bernoulli variance: maximum 1 at p=0.5, zero at p in {0,1}.
        return (4.0 * p * (1.0 - p)).clamp_(0.0, 1.0)

    @staticmethod
    def _coarse_boundary_strength(coarse_logits: torch.Tensor) -> torch.Tensor:
        """Fixed, differentiable-free contour strength from the coarse mask.

        A simple first-order finite difference is used instead of a trainable edge
        head, so UGBRA remains a feature module rather than a second prediction task.
        """
        if coarse_logits.ndim == 3:
            coarse_logits = coarse_logits[:, None]
        p = torch.sigmoid(coarse_logits)
        dx = F.pad((p[..., :, 1:] - p[..., :, :-1]).abs(), (0, 1, 0, 0))
        dy = F.pad((p[..., 1:, :] - p[..., :-1, :]).abs(), (0, 0, 0, 1))
        edge = dx + dy
        denom = edge.flatten(2).amax(dim=2, keepdim=True).view(-1, 1, 1, 1)
        return edge / denom.clamp_min(torch.finfo(edge.dtype).eps)

    @staticmethod
    def _raw_image_edge(image: torch.Tensor, out_size) -> torch.Tensor:
        """Extract a normalized local gradient cue from the augmented input.

        The cue is deliberately fixed/non-semantic. Per-channel absolute finite
        differences are averaged, normalized per case, clipped, and resized to
        the decoder grid. This preserves actual image detail that the 16x16 ViT
        patchification cannot recreate by interpolation alone.
        """
        if image is None or image.ndim != 4:
            raise ValueError("UGBRA image-edge path requires image [B,C,H,W]")
        dx = F.pad((image[..., :, 1:] - image[..., :, :-1]).abs(), (0, 1, 0, 0))
        dy = F.pad((image[..., 1:, :] - image[..., :-1, :]).abs(), (0, 0, 0, 1))
        edge = (dx + dy).mean(dim=1, keepdim=True)
        mean = edge.flatten(2).mean(dim=2, keepdim=True).view(-1, 1, 1, 1)
        eps = torch.finfo(edge.dtype).eps
        # Relative edge strength: 4x the case mean saturates at one.
        edge = (edge / mean.clamp_min(eps) / 4.0).clamp(0.0, 1.0)
        if tuple(edge.shape[-2:]) != tuple(out_size):
            edge = F.interpolate(edge, size=out_size, mode="bilinear", align_corners=False)
        return edge

    def forward(
        self,
        features: torch.Tensor,
        coarse_logits: torch.Tensor,
        image: torch.Tensor | None = None,
        *,
        return_diagnostics: bool = False,
    ):
        if features.ndim != 4 or features.shape[1] != self.channels:
            raise ValueError(
                f"features must be [B,{self.channels},H,W], got {tuple(features.shape)}"
            )

        uncertainty = self._coarse_uncertainty(coarse_logits)
        coarse_edge = self._coarse_boundary_strength(coarse_logits)
        if uncertainty.shape[-2:] != features.shape[-2:]:
            uncertainty = F.interpolate(
                uncertainty, size=features.shape[-2:], mode="bilinear", align_corners=False
            )
            coarse_edge = F.interpolate(
                coarse_edge, size=features.shape[-2:], mode="bilinear", align_corners=False
            )
        if self.detach_uncertainty:
            uncertainty = uncertainty.detach()
            coarse_edge = coarse_edge.detach()

        if self.use_image_edge:
            image_edge = self._raw_image_edge(image, features.shape[-2:]).detach()
        else:
            image_edge = torch.zeros_like(uncertainty)

        r3, h3 = self.branch3(features)
        r5, h5 = self.branch5(features)

        e3 = h3.abs().mean(dim=1, keepdim=True)
        e5 = h5.abs().mean(dim=1, keepdim=True)
        # Normalize per case so modality-dependent feature amplitude (ultrasound,
        # dermoscopy, endoscopy, MRI) does not dominate the gate.
        eps = torch.finfo(features.dtype).eps if features.is_floating_point() else 1e-6
        e3n = e3 / e3.flatten(2).mean(dim=2, keepdim=True).view(-1, 1, 1, 1).clamp_min(eps)
        e5n = e5 / e5.flatten(2).mean(dim=2, keepdim=True).view(-1, 1, 1, 1).clamp_min(eps)

        learned_gate = torch.sigmoid(
            self.spatial_gate(torch.cat([uncertainty, coarse_edge, image_edge, e3n, e5n], dim=1))
        )
        # Use the union of ambiguity and the current coarse contour. This keeps the
        # adapter active on both soft uncertain borders and sharp but misplaced ones.
        boundary_support = torch.maximum(torch.maximum(uncertainty, coarse_edge), image_edge * coarse_edge)
        uncertainty_gate = self.gate_floor + (1.0 - self.gate_floor) * boundary_support
        gate = learned_gate * uncertainty_gate

        image_residual = self.image_edge_proj(image_edge)
        source_desc = torch.cat(
            [
                e3.flatten(2).mean(dim=2),
                e5.flatten(2).mean(dim=2),
                image_edge.flatten(2).mean(dim=2),
            ],
            dim=1,
        )
        source_weights = torch.softmax(self.scale_selector(source_desc), dim=1)
        w3 = source_weights[:, 0].view(-1, 1, 1, 1)
        w5 = source_weights[:, 1].view(-1, 1, 1, 1)
        wi = source_weights[:, 2].view(-1, 1, 1, 1)
        mixed = w3 * r3 + w5 * r5 + wi * image_residual

        # tanh bounds the global residual amplitude and protects Base semantics.
        gamma = torch.tanh(self.gamma)
        refined = features + gamma * gate * mixed

        if not return_diagnostics:
            return refined
        diagnostics: Dict[str, torch.Tensor] = {
            "ugbra_gamma": gamma.detach(),
            "ugbra_gate_mean": gate.mean().detach(),
            "ugbra_uncertainty_mean": uncertainty.mean().detach(),
            "ugbra_coarse_edge_mean": coarse_edge.mean().detach(),
            "ugbra_image_edge_mean": image_edge.mean().detach(),
            "ugbra_source3_mean": source_weights[:, 0].mean().detach(),
            "ugbra_source5_mean": source_weights[:, 1].mean().detach(),
            "ugbra_source_image_mean": source_weights[:, 2].mean().detach(),
            "ugbra_residual_rms": mixed.square().mean().sqrt().detach(),
        }
        return refined, diagnostics
