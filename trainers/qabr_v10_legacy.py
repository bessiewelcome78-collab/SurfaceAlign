"""Query-Anchored Boundary Refiner (QABR).

A tiny high-resolution boundary refinement module for MedCLIPSeg/JBT-Lite.

Why it exists
-------------
The MedCLIPSeg semantic mask is produced at a coarse decoder resolution (56x56
for a 224x224 input) and then bilinearly resized.  The previous UGBRA module
injected an unsigned high-frequency feature residual at 56x56.  That was able
to improve surface proximity, but an unsigned edge only answers *where an edge
is*, not *which side of the current contour should gain foreground evidence*.

QABR keeps the existing text-conditioned coarse logit as the semantic anchor
and predicts only a small signed logit correction on the native 224x224 grid.
It combines:
  * uncertainty and the current predicted contour (where to refine),
  * high-resolution image gradient magnitude (where image structure is),
  * signed image contrast along the predicted mask normal (direction cue),
  * compact decoder detail and coarse-logit high-pass cues (what local evidence
    the decoder already contains).

There is no extra prediction task and no extra loss. Existing CE/Dice and the
optional early Surface Alignment objective supervise the final refined mask.
A zero-initialized ReZero/LayerScale scalar makes the first forward exactly the
same as the original bilinear Base prediction.
"""
from __future__ import annotations

from typing import Dict, Tuple
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int, max_groups: int = 8) -> int:
    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "1" if default else "0").strip().lower()
    return raw not in {"0", "false", "no", "off", ""}


class QueryAnchoredBoundaryRefiner(nn.Module):
    """Refine a text-conditioned coarse mask only inside a high-res boundary band.

    Parameters
    ----------
    decoder_channels:
        Channel count of the existing decoder feature map (512 for this project).
    detail_channels:
        Compressed decoder-detail channels used by the local refinement head.
    hidden_channels:
        Hidden width of the high-resolution local head.
    alpha_init:
        Residual coefficient before tanh. ``0`` guarantees exact Base identity.
    band_kernel:
        Odd max-pool kernel on the 224-grid used to dilate the current contour.
    max_logit_delta:
        Hard upper bound on the signed local logit correction before alpha.
    detach_support:
        Detach uncertainty/contour geometry so Base cannot game the gate.
    """

    def __init__(
        self,
        decoder_channels: int = 512,
        detail_channels: int = 8,
        hidden_channels: int = 32,
        alpha_init: float = 0.0,
        band_kernel: int = 11,
        max_logit_delta: float = 3.0,
        detach_support: bool = True,
        use_image_detail: bool = True,
    ) -> None:
        super().__init__()
        decoder_channels = int(decoder_channels)
        detail_channels = max(4, int(detail_channels))
        hidden_channels = max(16, int(hidden_channels))
        band_kernel = int(band_kernel)
        if band_kernel < 3 or band_kernel % 2 == 0:
            raise ValueError("band_kernel must be an odd integer >= 3")

        self.decoder_channels = decoder_channels
        self.detail_channels = detail_channels
        self.hidden_channels = hidden_channels
        self.band_kernel = band_kernel
        self.max_logit_delta = float(max(0.25, max_logit_delta))
        self.detach_support = bool(detach_support)
        self.use_image_detail = bool(use_image_detail)
        # Strong structural ablations; all OFF in FULL.
        self.ablate_no_query_anchor = _env_bool("QABR_ABL_NO_QUERY_ANCHOR", False)
        self.ablate_no_signed_direction = _env_bool("QABR_ABL_NO_SIGNED_DIRECTION", False)

        # ICASSP QABR-only structural ablations. All are OFF in FULL, so the
        # default/full forward is byte-for-byte equivalent at the mathematical
        # operator level. These switches remove *functional evidence groups*
        # rather than tiny optimizer safeguards.
        self.ablate_no_directional_geometry = _env_bool(
            "QABR_ABL_NO_DIRECTIONAL_GEOMETRY", False
        )
        self.ablate_no_semantic_detail = _env_bool(
            "QABR_ABL_NO_SEMANTIC_DETAIL", False
        )
        self.ablate_mask_only = _env_bool("QABR_ABL_MASK_ONLY", False)

        # Compress semantic decoder features before bringing them to 224x224.
        # The local high-pass is calculated in the 56x56 decoder grid first,
        # which is substantially cheaper than upsampling all 512 channels.
        self.detail_proj = nn.Sequential(
            nn.Conv2d(decoder_channels, detail_channels, 1, bias=False),
            nn.GroupNorm(_group_count(detail_channels), detail_channels),
            nn.GELU(),
        )

        # Inputs to local head:
        # detail_channels + p/u/coarse_edge/image_edge/signed_normal/hp3/hp5 = +7
        in_channels = detail_channels + 7
        g = _group_count(hidden_channels)
        self.refine_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(g, hidden_channels),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels, hidden_channels, 3, padding=1,
                groups=hidden_channels, bias=False,
            ),
            nn.GroupNorm(g, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, 1, bias=True),
        )

        # Keep the candidate correction unbiased at construction while avoiding
        # a dead alpha/head pair. Alpha=0 is sufficient for exact identity, so
        # the head may use ordinary Kaiming initialization and receives gradient
        # as soon as alpha leaves zero after the first optimizer step.
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    @staticmethod
    def _as_4d(logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 3:
            logits = logits[:, None]
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError(
                "coarse_logits must be [B,H,W] or [B,1,H,W], got "
                f"{tuple(logits.shape)}"
            )
        return logits

    @staticmethod
    def _finite_diff(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Centered finite differences with replicate-safe one-sided borders."""
        dx = 0.5 * F.pad(x[..., :, 2:] - x[..., :, :-2], (1, 1, 0, 0))
        dy = 0.5 * F.pad(x[..., 2:, :] - x[..., :-2, :], (0, 0, 1, 1))
        return dx, dy

    @staticmethod
    def _case_normalize(x: torch.Tensor, *, robust_scale: float = 3.0) -> torch.Tensor:
        # Mean normalization is deterministic and cheap. Clipping makes the cue
        # comparable across ultrasound, dermoscopy, endoscopy and MRI.
        eps = torch.finfo(x.dtype).eps
        scale = x.flatten(2).mean(dim=2, keepdim=True).view(-1, 1, 1, 1)
        return (x / scale.clamp_min(eps) / robust_scale).clamp(0.0, 1.0)

    def _image_geometry(
        self,
        image: torch.Tensor | None,
        mask_nx: torch.Tensor,
        mask_ny: torch.Tensor,
        out_size,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (not self.use_image_detail) or image is None:
            z = torch.zeros_like(mask_nx)
            return z, z
        if image.ndim != 4:
            raise ValueError(f"image must be [B,C,H,W], got {tuple(image.shape)}")
        if tuple(image.shape[-2:]) != tuple(out_size):
            image = F.interpolate(image, size=out_size, mode="bilinear", align_corners=False)

        # Mean over channels keeps the path fixed/non-semantic. A light local
        # average suppresses pixel noise before directional finite differences.
        gray = image.mean(dim=1, keepdim=True)
        gray = F.avg_pool2d(gray, 3, stride=1, padding=1)
        gx, gy = self._finite_diff(gray)
        eps = torch.finfo(gray.dtype).eps
        image_mag_raw = torch.sqrt(gx.square() + gy.square() + eps)
        image_edge = self._case_normalize(image_mag_raw, robust_scale=3.0)

        # Crucial difference from UGBRA: preserve the sign of contrast along the
        # *predicted mask normal*. This provides the local head with an inside/
        # outside orientation cue instead of only saying that an edge exists.
        signed_normal = (gx * mask_nx + gy * mask_ny) / image_mag_raw.clamp_min(eps)
        signed_normal = signed_normal.clamp(-1.0, 1.0)
        return image_edge, signed_normal

    def forward(
        self,
        decoder_features: torch.Tensor,
        coarse_logits: torch.Tensor,
        image: torch.Tensor | None = None,
        *,
        output_size=None,
        return_diagnostics: bool = False,
    ):
        if decoder_features.ndim != 4 or decoder_features.shape[1] != self.decoder_channels:
            raise ValueError(
                f"decoder_features must be [B,{self.decoder_channels},H,W], got "
                f"{tuple(decoder_features.shape)}"
            )
        coarse_logits = self._as_4d(coarse_logits)
        if output_size is None:
            output_size = image.shape[-2:] if image is not None else (
                coarse_logits.shape[-2] * 4, coarse_logits.shape[-1] * 4
            )
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        output_size = tuple(int(v) for v in output_size)

        base_hr = F.interpolate(
            coarse_logits, size=output_size, mode="bilinear", align_corners=False
        )
        p = torch.sigmoid(base_hr)
        uncertainty = (4.0 * p * (1.0 - p)).clamp(0.0, 1.0)

        # Predicted-mask normal and a normalized current contour strength.
        px, py = self._finite_diff(p)
        eps = torch.finfo(p.dtype).eps
        mask_mag_raw = torch.sqrt(px.square() + py.square() + eps)
        mask_nx = px / mask_mag_raw.clamp_min(eps)
        mask_ny = py / mask_mag_raw.clamp_min(eps)
        coarse_edge = self._case_normalize(mask_mag_raw, robust_scale=2.0)

        # Build a strict support band from the *hard current contour*.  This is
        # intentionally detached and prevents the soft uncertainty field from
        # turning QABR into a whole-image residual adapter.  A uniform all-BG or
        # all-FG coarse prediction has no contour, hence QABR cannot hallucinate
        # a remote lesion by itself.
        hard = (p.detach() >= 0.5).to(p.dtype)
        hard_dilate = F.max_pool2d(hard, 3, stride=1, padding=1)
        hard_erode = 1.0 - F.max_pool2d(1.0 - hard, 3, stride=1, padding=1)
        hard_boundary = (hard_dilate - hard_erode).clamp(0.0, 1.0)
        band = F.max_pool2d(
            hard_boundary,
            kernel_size=self.band_kernel,
            stride=1,
            padding=self.band_kernel // 2,
        ).clamp(0.0, 1.0)

        image_edge, signed_normal = self._image_geometry(
            image, mask_nx, mask_ny, output_size
        )

        if self.detach_support:
            support_uncertainty = uncertainty.detach()
            support_band = band.detach()
            support_edge = coarse_edge.detach()
            image_edge_for_support = image_edge.detach()
            signed_normal_input = signed_normal.detach()
        else:
            support_uncertainty = uncertainty
            support_band = band
            support_edge = coarse_edge
            image_edge_for_support = image_edge
            signed_normal_input = signed_normal

        # Compact decoder detail. High-pass before resize avoids simply feeding a
        # second copy of the semantic feature map into the refinement head.
        detail = self.detail_proj(decoder_features)
        detail = detail - F.avg_pool2d(detail, 3, stride=1, padding=1)
        detail = F.interpolate(detail, size=output_size, mode="bilinear", align_corners=False)

        # Text-query anchored scalar structure: these are derived from the
        # already text-conditioned coarse logit, not from an independent head.
        hp3 = base_hr - F.avg_pool2d(base_hr, 3, stride=1, padding=1)
        hp5 = base_hr - F.avg_pool2d(base_hr, 5, stride=1, padding=2)
        # Bounded inputs improve cross-modality stability.
        hp3 = torch.tanh(hp3)
        hp5 = torch.tanh(hp5)

        if self.ablate_no_query_anchor:
            # Historical narrow ablation: only removes explicit coarse-logit
            # channels. Kept for compatibility, but it is intentionally not the
            # recommended ICASSP main-table ablation because the residual/support
            # path still remains anchored to the Base mask.
            p_input = torch.zeros_like(p)
            hp3_input = torch.zeros_like(hp3)
            hp5_input = torch.zeros_like(hp5)
        else:
            p_input = p
            hp3_input = hp3
            hp5_input = hp5

        detail_input = detail
        image_edge_input = image_edge.detach() if self.detach_support else image_edge

        if self.ablate_no_signed_direction:
            signed_normal_input = torch.zeros_like(signed_normal_input)

        # (1) w/o Directional Geometry: remove the image-derived edge magnitude
        # and signed normal cue together. This is a structural cue-group ablation:
        # the local head still receives decoder detail and coarse-mask evidence.
        if self.ablate_no_directional_geometry:
            image_edge_input = torch.zeros_like(image_edge_input)
            signed_normal_input = torch.zeros_like(signed_normal_input)

        # (2) w/o Semantic Detail: remove the compressed decoder-detail branch
        # and explicit coarse-logit high-pass/query channels, but retain the
        # boundary/uncertainty geometry and image-direction cues.
        if self.ablate_no_semantic_detail:
            detail_input = torch.zeros_like(detail_input)
            p_input = torch.zeros_like(p_input)
            hp3_input = torch.zeros_like(hp3_input)
            hp5_input = torch.zeros_like(hp5_input)

        # (3) Mask-only QABR: strongest evidence ablation. Keep the same QABR
        # head, local support and V15 tangent operator, but remove *all* high-res
        # image/decoder evidence. The refiner can only use the current mask's
        # probability/uncertainty/contour/high-pass structure. Parameter count
        # and construction/RNG remain unchanged.
        if self.ablate_mask_only:
            detail_input = torch.zeros_like(detail_input)
            image_edge_input = torch.zeros_like(image_edge_input)
            signed_normal_input = torch.zeros_like(signed_normal_input)

        local_input = torch.cat(
            [
                detail_input,
                p_input,
                support_uncertainty,
                support_edge,
                image_edge_input,
                signed_normal_input,
                hp3_input,
                hp5_input,
            ],
            dim=1,
        )
        candidate_delta = self.max_logit_delta * torch.tanh(self.refine_head(local_input))

        # Let image evidence strengthen an existing predicted band, but never
        # create a new remote edit region by itself. The uncertainty factor keeps
        # the strongest correction on ambiguous pixels while retaining 25% of
        # capacity for sharp-but-misplaced contours.
        local_evidence = torch.maximum(support_uncertainty, image_edge_for_support * support_band)
        support = support_band * (0.25 + 0.75 * local_evidence)

        alpha = torch.tanh(self.alpha)
        refined = base_hr + alpha * support * candidate_delta

        if not return_diagnostics:
            return refined
        diagnostics: Dict[str, torch.Tensor] = {
            "qabr_alpha": alpha.detach(),
            "qabr_support_mean": support.mean().detach(),
            "qabr_support_active": (support > 0.10).float().mean().detach(),
            "qabr_uncertainty_mean": uncertainty.mean().detach(),
            "qabr_image_edge_mean": image_edge.mean().detach(),
            "qabr_signed_normal_abs_mean": signed_normal.abs().mean().detach(),
            "qabr_delta_rms": candidate_delta.square().mean().sqrt().detach(),
            "qabr_effective_delta_rms": (alpha * support * candidate_delta).square().mean().sqrt().detach(),
        }
        return refined, diagnostics
