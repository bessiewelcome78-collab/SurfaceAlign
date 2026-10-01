"""GEOTR-V4D root-fix with V4C-compatible Transport--Reconstruction refinement.

Root-cause target
-----------------
The P0 audit showed that posterior aggregation order is negligible, while the
last checkpoint strongly improves reconstruction on fixed Train cases but
collapses on Val.  The failure is therefore a Stage-2 generalization problem,
not a posterior-order or numerical-instability problem.

V4D keeps the validated Transport path unchanged and changes only Stage-2:
  * Stage 1 remains the V3 dense spatial transport operator.
  * Stage 2 no longer asks one dense head to infer both WHERE and HOW to edit.
  * A three-way typed localizer predicts Correct / False-Negative / False-Positive.
  * Its output prior is initialized to the sparse residual prevalence instead of q=0.5.
  * Separate positive Add and Remove magnitude fields implement monotonic bounded edits.
  * The final edit is q_FN.detach()*Add - q_FP.detach()*Remove. No hard gate,
    benefit/harm oracle or selector is used in deployment.
  * Stage-2 owns its own evidence encoder under ``m2_surface``.  Its gradients do
    not change Stage-1 Transport features.  This makes A1 and the Transport part
    of A3 causally comparable.

Controlled modes (M1.GEOTOPO_MODE):
  base     : A0, Base only.
  geometry : A1, Transport(Base).
  residual : A2, TypedResidualCorrection(Base).
  full     : A3, Transport(Base) -> TypedResidualCorrection(Transport).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .c2r_region_refiner import CounterfactualConsensusRegionRefiner
from .c2r_canonical_roi_refiner import CanonicalCounterfactualROIRefiner
from .pc2r_posterior_refiner import PosteriorConsistentCanonicalROIRefiner, CanonicalCoordinatePosteriorROIRefiner
from .aefr_action_refiner import ActionEvidenceFactorizedROIRefiner
from .slr_local_rerenderer import GeometryConditionedSparseLocalRerenderer
from .slr_transition_refiner import UtilityConsistentDualSpaceResidualTransitionRefiner
from .slr_transition_refiner_r2 import OperatorConsistentSelectiveTransitionRefiner
from .sparc_hr_composer import StructuredPosteriorAtomicHRComposer

EPS = 1.0e-4


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
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(cin, cout, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(_groups(cout), cout),
            nn.GELU(),
        )


class GeometryHead(nn.Module):
    """Predict a dense displacement field in pixel units."""

    def __init__(self, hidden_dim: int, init_scale_px: float = 1.0) -> None:
        super().__init__()
        self.body = nn.Sequential(
            ConvNormGELU(hidden_dim, hidden_dim),
            ConvNormGELU(hidden_dim, hidden_dim),
        )
        self.flow_out = nn.Conv2d(hidden_dim, 2, kernel_size=3, padding=1)
        init = max(float(init_scale_px), 1.0e-3)
        self.log_scale = nn.Parameter(torch.tensor(math.log(math.expm1(init))))
        nn.init.zeros_(self.flow_out.weight)
        nn.init.zeros_(self.flow_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.softplus(self.log_scale).to(dtype=x.dtype, device=x.device)
        return torch.tanh(self.flow_out(self.body(x))) * scale



class SparseDirectResidualRefiner(nn.Module):
    """GEOTR-V4G sparse direct refiner with the optional V4G-R2 root contract.

    V4G-R1 keeps the original direct-GT sparse re-segmentation path for exact
    backward compatibility.

    V4G-R2 changes the *operator contract*, not merely a loss weight:
      * selected == "inspect", not "must edit";
      * factual hard-error points are corrected toward GT;
      * factual already-correct points preserve the detached anchor posterior;
      * training selection can exactly match inference top-K selection;
      * denoising corruption has stationary, structured morphology and a known
        corruption support instead of inheriting the current Base error count;
      * a detached high-resolution decoder feature can be injected as local
        evidence;
      * the sparse correction head is evaluated only at selected points.

    The caller is responsible for supplying transport-aligned MC evidence for
    post-Geometry refinement.  Validation/test never use GT in this module.
    """

    def __init__(
        self,
        hidden_dim: int,
        semantic_channels: int,
        text_dim: int,
        *,
        coverage: float = 0.10,
        train_importance_ratio: float = 0.75,
        selection_score: str = "margin",
        flow_scale_px: float = 8.0,
        denoise_enabled: bool = True,
        r2_enabled: bool = False,
        fine_feature_channels: int = 512,
        match_inference_selection: bool = True,
        denoise_coverage: float = 0.03,
        denoise_radius_min: int = 1,
        denoise_radius_max: int = 3,
        r3_enabled: bool = False,
        r3_max_abs_delta_logit: float = 1.0,
        r4_enabled: bool = False,
        r4_patch_size: int = 5,
        r4_synth_edit_fraction: float = 0.20,
        r4_flip_threshold: float = 0.50,
        r4_target_prob_margin: float = 0.05,
        r4_corrupt_logit_min: float = 0.05,
        r4_corrupt_logit_max: float = 2.50,
        r41_enabled: bool = False,
        r41_patch_reduce_channels: int = 32,
        r41_corrupt_logit_max: float = 0.80,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.semantic_channels = int(semantic_channels)
        self.text_dim = int(text_dim)
        self.coverage = float(min(max(coverage, 1.0e-4), 1.0))
        self.train_importance_ratio = float(min(max(train_importance_ratio, 0.0), 1.0))
        self.selection_score = str(selection_score).strip().lower()
        if self.selection_score not in {"margin", "mc_std", "mc_disagreement", "entropy", "hybrid_max"}:
            raise ValueError(
                "GEOTR_V4G_SELECTION_SCORE must be one of "
                "margin/mc_std/mc_disagreement/entropy/hybrid_max, got "
                f"{self.selection_score!r}"
            )
        self.flow_scale_px = float(max(flow_scale_px, 1.0e-3))
        self.denoise_enabled = bool(denoise_enabled)
        self.r2_enabled = bool(r2_enabled)
        self.match_inference_selection = bool(match_inference_selection)
        self.denoise_coverage = float(min(max(denoise_coverage, 1.0e-4), 1.0))
        self.denoise_radius_min = max(1, int(denoise_radius_min))
        self.denoise_radius_max = max(self.denoise_radius_min, int(denoise_radius_max))
        self.r3_enabled = bool(r3_enabled)
        self.r3_max_abs_delta_logit = float(max(r3_max_abs_delta_logit, 1.0e-3))
        self.r4_enabled = bool(r4_enabled)
        self.r4_patch_size = int(r4_patch_size)
        if self.r4_patch_size < 3 or self.r4_patch_size % 2 == 0:
            raise ValueError("GEOTR-V4G-R4 patch size must be an odd integer >= 3")
        self.r4_synth_edit_fraction = float(min(max(r4_synth_edit_fraction, 0.02), 0.80))
        self.r4_flip_threshold = float(min(max(r4_flip_threshold, 0.50), 0.99))
        self.r4_target_prob_margin = float(min(max(r4_target_prob_margin, 1.0e-4), 0.49))
        self.r4_corrupt_logit_min = float(max(r4_corrupt_logit_min, 1.0e-3))
        self.r4_corrupt_logit_max = float(max(r4_corrupt_logit_max, self.r4_corrupt_logit_min + 1.0e-3))
        self.r41_enabled = bool(r41_enabled)
        self.r41_patch_reduce_channels = max(8, min(int(r41_patch_reduce_channels), self.hidden_dim))
        self.r41_corrupt_logit_max = float(
            max(r41_corrupt_logit_max, self.r4_corrupt_logit_min + 1.0e-3)
        )
        if self.r3_enabled and not self.r2_enabled:
            raise ValueError("GEOTR-V4G-R3 requires the R2 sparse point architecture as its causal base")
        if self.r4_enabled and not self.r2_enabled:
            raise ValueError("GEOTR-V4G-R4 requires the R2 sparse feature architecture as its causal base")
        if self.r41_enabled and not self.r4_enabled:
            raise ValueError("GEOTR-V4G-R4.1 requires GEOTR-V4G-R4 to be enabled")

        self.image_stem = nn.Sequential(
            ConvNormGELU(3, hidden_dim),
            ConvNormGELU(hidden_dim, hidden_dim),
        )
        self.semantic_proj = nn.Sequential(
            nn.Conv2d(semantic_channels, hidden_dim, 1, bias=False),
            nn.GroupNorm(_groups(hidden_dim), hidden_dim),
            nn.GELU(),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        if self.r2_enabled:
            self.fine_proj = nn.Sequential(
                nn.Conv2d(int(fine_feature_channels), hidden_dim, 1, bias=False),
                nn.GroupNorm(_groups(hidden_dim), hidden_dim),
                nn.GELU(),
            )
            # Active R2 uncertainty evidence is deliberately compact:
            # anchor + margin + transport-aligned MC std.  Disagreement and
            # entropy are still exported as diagnostics but are not forced into
            # the representation when validation says they add little ranking
            # information.
            fuse_in = 3 * hidden_dim + 3
        else:
            self.fine_proj = None
            # R1: anchor + margin + MC std + disagreement + entropy.
            fuse_in = 2 * hidden_dim + 5
        self.pixel_fuse = nn.Sequential(
            ConvNormGELU(fuse_in, hidden_dim),
            ConvNormGELU(hidden_dim, hidden_dim),
        )
        self.global_context = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.context_film = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.context_gate = nn.Parameter(torch.zeros(()))
        # Base, Anchor, Anchor-Base, flow_x, flow_y, |flow|.
        self.trace_fuse = nn.Sequential(
            ConvNormGELU(hidden_dim + 6, hidden_dim),
            ConvNormGELU(hidden_dim, hidden_dim),
        )

        # Historical dense R1 head.
        self.refine_body = nn.Sequential(
            ConvNormGELU(hidden_dim, hidden_dim),
            ConvNormGELU(hidden_dim, hidden_dim),
        )
        self.delta_out = nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.delta_out.weight)
        nn.init.zeros_(self.delta_out.bias)

        # R2 sparse point head.  Only K selected point features pass through
        # these layers; outside K the delta is exactly zero by construction.
        if self.r2_enabled:
            self.point_body = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            self.point_delta = nn.Linear(hidden_dim, 1)
            nn.init.zeros_(self.point_delta.weight)
            nn.init.zeros_(self.point_delta.bias)
        else:
            self.point_body = None
            self.point_delta = None

        # R4: the remaining learned question is only whether the current hard
        # label should be flipped. Direction and magnitude are deterministic in
        # binary segmentation: BG->FG is ADD, FG->BG is REMOVE, and the deployed
        # probability is moved only to a small GT-side margin. A local patch
        # encoder is used because error-vs-uncertain-correct separability is the
        # unresolved bottleneck in R3.
        if self.r4_enabled:
            # Dedicated synthetic-task RNG state.  R4 corruption must vary
            # across calls while remaining deterministic and isolated from the
            # global/DataLoader RNG used by Base/Geometry.  A non-persistent
            # step buffer makes the sequence reproducible within a run without
            # changing checkpoint compatibility.
            self.register_buffer("_r4_synth_step", torch.zeros((), dtype=torch.long), persistent=False)
            # R4.1 keeps local patch reasoning but avoids materializing the
            # full-image C*P*P unfold tensor.
            if self.r41_enabled:
                self.r41_patch_reduce = nn.Sequential(
                    nn.Conv2d(self.hidden_dim, self.r41_patch_reduce_channels, 1, bias=False),
                    nn.GroupNorm(_groups(self.r41_patch_reduce_channels), self.r41_patch_reduce_channels),
                    nn.GELU(),
                )
                patch_channels = self.r41_patch_reduce_channels
            else:
                self.r41_patch_reduce = None
                patch_channels = self.hidden_dim
            patch_dim = patch_channels * self.r4_patch_size * self.r4_patch_size
            self.r4_patch_proj = nn.Sequential(
                nn.Linear(patch_dim, 2 * self.hidden_dim),
                nn.LayerNorm(2 * self.hidden_dim),
                nn.GELU(),
                nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                nn.GELU(),
            )
            self.r4_flip_head = nn.Linear(self.hidden_dim, 1)
            if self.r41_enabled:
                # Keep the initial policy safely below the FLIP threshold while
                # allowing gradients to reach the patch encoder from step 1.
                nn.init.normal_(self.r4_flip_head.weight, mean=0.0, std=1.0e-3)
            else:
                nn.init.zeros_(self.r4_flip_head.weight)
            prior = self.r4_synth_edit_fraction
            nn.init.constant_(self.r4_flip_head.bias, math.log(prior / max(1.0 - prior, 1.0e-6)))
            # Historical R1/R2/R3 actuator heads are not part of the active R4
            # graph. Freeze them so trainable-parameter accounting is honest.
            for mod in (self.refine_body, self.delta_out, self.point_body, self.point_delta):
                if mod is not None:
                    for par in mod.parameters():
                        par.requires_grad_(False)
        else:
            self.r41_patch_reduce = None
            self.r4_patch_proj = None
            self.r4_flip_head = None

    @staticmethod
    def _resize(x: torch.Tensor, hw: Tuple[int, int], *, nearest: bool = False) -> torch.Tensor:
        if x.shape[-2:] == hw:
            return x
        if nearest:
            return F.interpolate(x, size=hw, mode="nearest")
        return F.interpolate(x, size=hw, mode="bilinear", align_corners=False)

    @staticmethod
    def _margin_uncertainty(prob: torch.Tensor) -> torch.Tensor:
        return (4.0 * prob * (1.0 - prob)).clamp(0.0, 1.0)

    @staticmethod
    def _entropy(prob: torch.Tensor) -> torch.Tensor:
        p = prob.clamp(EPS, 1.0 - EPS)
        h = -(p * p.log() + (1.0 - p) * (1.0 - p).log())
        return (h / math.log(2.0)).clamp(0.0, 1.0)

    @staticmethod
    def _fit_evidence(x: Optional[torch.Tensor], anchor: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            return torch.zeros_like(anchor)
        x = x.detach().to(anchor)
        if x.ndim == 3:
            x = x[:, None]
        if x.shape[-2:] != anchor.shape[-2:]:
            x = F.interpolate(x, size=anchor.shape[-2:], mode="bilinear", align_corners=False)
        return (float(scale) * x).clamp(0.0, 1.0)

    def _trace(
        self,
        base_prob: torch.Tensor,
        anchor_prob: torch.Tensor,
        flow_px: torch.Tensor,
    ) -> torch.Tensor:
        base = base_prob.detach().clamp(EPS, 1.0 - EPS)
        anchor = anchor_prob.detach().clamp(EPS, 1.0 - EPS)
        flow = flow_px.detach().to(anchor) / self.flow_scale_px
        if flow.shape[-2:] != anchor.shape[-2:]:
            flow = self._resize(flow, tuple(anchor.shape[-2:]))
        flow = flow.clamp(-1.0, 1.0)
        mag = torch.sqrt(flow[:, 0:1].square() + flow[:, 1:2].square() + 1.0e-12).clamp(0.0, 1.0)
        return torch.cat([base, anchor, anchor - base, flow[:, 0:1], flow[:, 1:2], mag], dim=1)

    def _features(
        self,
        image: torch.Tensor,
        semantic_map: torch.Tensor,
        text_features: torch.Tensor,
        anchor_prob: torch.Tensor,
        base_prob: torch.Tensor,
        flow_px: torch.Tensor,
        mc_std_map: Optional[torch.Tensor],
        mc_disagreement_map: Optional[torch.Tensor],
        fine_feature_map: Optional[torch.Tensor] = None,
        trace_override: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        anchor = anchor_prob.detach().clamp(EPS, 1.0 - EPS)
        hw = tuple(anchor.shape[-2:])
        image_latent = self.image_stem(self._resize(image.detach(), hw))
        semantic_latent = self.semantic_proj(self._resize(semantic_map.detach(), hw))
        text = self.text_proj(text_features.detach().float())
        margin = self._margin_uncertainty(anchor)
        # Bernoulli probability std is <= 0.5.  2*std maps the physically
        # possible range into [0,1] without validation-fitted calibration.
        mc_std = self._fit_evidence(mc_std_map, anchor, scale=2.0)
        mc_dis = self._fit_evidence(mc_disagreement_map, anchor, scale=1.0)
        entropy = self._entropy(anchor)
        if self.r2_enabled:
            if isinstance(fine_feature_map, torch.Tensor):
                fine = fine_feature_map.detach().to(anchor)
                if fine.ndim != 4:
                    raise ValueError(f"fine_feature_map must be BCHW, got {tuple(fine.shape)}")
                fine = self.fine_proj(self._resize(fine, hw))
            else:
                fine = torch.zeros_like(image_latent)
            pixel_in = torch.cat(
                [image_latent, semantic_latent, fine, anchor, margin, mc_std], dim=1
            )
        else:
            fine = torch.zeros_like(image_latent)
            pixel_in = torch.cat(
                [image_latent, semantic_latent, anchor, margin, mc_std, mc_dis, entropy], dim=1
            )
        pixel = self.pixel_fuse(pixel_in)
        global_visual = F.adaptive_avg_pool2d(pixel, 1).flatten(1)
        context = self.global_context(torch.cat([global_visual, text], dim=1))
        gamma, beta = self.context_film(context).chunk(2, dim=1)
        gate = torch.tanh(self.context_gate)
        pixel = pixel + gate * (
            torch.tanh(gamma)[:, :, None, None] * pixel + beta[:, :, None, None]
        )
        if isinstance(trace_override, torch.Tensor):
            trace = trace_override.detach().to(anchor)
            if trace.shape[-2:] != hw:
                trace = self._resize(trace, hw)
            if trace.shape[1] != 6:
                raise ValueError(f"trace_override must have 6 channels, got {tuple(trace.shape)}")
        else:
            trace = self._trace(base_prob, anchor, flow_px)
        hidden = self.trace_fuse(torch.cat([pixel, trace], dim=1))
        evidence = {
            "margin": margin,
            "mc_std": mc_std,
            "mc_disagreement": mc_dis,
            "entropy": entropy,
            "trace": trace,
            "fine_feature": fine,
        }
        return hidden, evidence

    def _selection_score(self, evidence: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.selection_score == "hybrid_max":
            return torch.maximum(
                torch.maximum(evidence["margin"], evidence["mc_std"]),
                torch.maximum(evidence["mc_disagreement"], evidence["entropy"]),
            )
        return evidence[self.selection_score]

    @staticmethod
    def _topk_mask(score: torch.Tensor, k: int, forbid: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, _, h, w = score.shape
        n = h * w
        k = int(max(0, min(int(k), n)))
        out = torch.zeros((b, n), dtype=torch.bool, device=score.device)
        if k <= 0:
            return out.view(b, 1, h, w)
        flat = score.detach().flatten(1)
        if isinstance(forbid, torch.Tensor):
            f = forbid.flatten(1).bool()
            flat = flat.masked_fill(f, float("-inf"))
        idx = flat.topk(k, dim=1, largest=True, sorted=False).indices
        out.scatter_(1, idx, True)
        if isinstance(forbid, torch.Tensor):
            out &= ~forbid.flatten(1).bool()
        return out.view(b, 1, h, w)

    @staticmethod
    def _fill_topk_with_forced(score: torch.Tensor, k: int, forced: Optional[torch.Tensor]) -> torch.Tensor:
        b, _, h, w = score.shape
        n = h * w
        k = max(1, min(int(k), n))
        forced_b = torch.zeros_like(score, dtype=torch.bool) if forced is None else forced.bool()
        out = forced_b.flatten(1).clone()
        flat = score.detach().flatten(1)
        for bi in range(b):
            used = out[bi]
            count = int(used.sum().item())
            if count > k:
                # Keep the highest-score forced points if synthetic corruption
                # ever exceeds the sparse budget.
                s = flat[bi].masked_fill(~used, float("-inf"))
                idx = s.topk(k, largest=True, sorted=False).indices
                used.zero_(); used[idx] = True
                continue
            rem = k - count
            if rem > 0:
                s = flat[bi].masked_fill(used, float("-inf"))
                idx = s.topk(rem, largest=True, sorted=False).indices
                used[idx] = True
        return out.view(b, 1, h, w)

    def _select_points(
        self,
        score: torch.Tensor,
        *,
        training: bool,
        forced: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, _, h, w = score.shape
        n = h * w
        k = max(1, min(n, int(math.ceil(self.coverage * n))))
        if self.r2_enabled and self.match_inference_selection:
            return self._fill_topk_with_forced(score, k, forced)

        forced_b = torch.zeros_like(score, dtype=torch.bool) if forced is None else forced.bool()
        selected = forced_b.clone()
        if not training:
            return self._topk_mask(score, k)

        # Historical R1 PointRend-style uncertainty+uniform sampling.
        imp_k = int(round(k * self.train_importance_ratio))
        imp_k = max(0, min(k, imp_k))
        out = selected.flatten(1)
        flat_score = score.detach().flatten(1)
        cuda_devices = [int(score.device.index)] if score.is_cuda and score.device.index is not None else []
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            rand_score = torch.rand_like(flat_score)
            for bi in range(b):
                used = out[bi]
                remaining = max(0, k - int(used.sum().item()))
                if remaining <= 0:
                    idx = torch.nonzero(used, as_tuple=False).flatten()[:k]
                    used.zero_(); used[idx] = True
                    continue
                this_imp = min(remaining, imp_k)
                s = flat_score[bi].masked_fill(used, float("-inf"))
                if this_imp > 0:
                    idx = s.topk(this_imp, largest=True, sorted=False).indices
                    used[idx] = True
                remaining = max(0, k - int(used.sum().item()))
                if remaining > 0:
                    r = rand_score[bi].masked_fill(used, float("-inf"))
                    idx = r.topk(remaining, largest=True, sorted=False).indices
                    used[idx] = True
        return out.view(b, 1, h, w)

    @staticmethod
    def _erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
        if radius <= 0:
            return mask
        k = 2 * int(radius) + 1
        inv = 1.0 - mask.float()[:, None]
        eroded = 1.0 - F.max_pool2d(inv, kernel_size=k, stride=1, padding=radius)
        return eroded[:, 0] > 0.5

    @staticmethod
    def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
        if radius <= 0:
            return mask
        k = 2 * int(radius) + 1
        return F.max_pool2d(mask.float()[:, None], kernel_size=k, stride=1, padding=radius)[:, 0] > 0.5

    @staticmethod
    def _structured_subset(candidates: torch.Tensor, count: int, radius: int) -> torch.Tensor:
        """Choose a spatially coherent random subset without using factual errors."""
        b, h, w = candidates.shape
        out = torch.zeros_like(candidates, dtype=torch.bool)
        count = max(0, int(count))
        if count <= 0:
            return out
        cuda_devices = [int(candidates.device.index)] if candidates.is_cuda and candidates.device.index is not None else []
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            noise = torch.rand((b, 1, h, w), device=candidates.device)
            k = max(3, 2 * int(radius) + 1)
            if k % 2 == 0:
                k += 1
            smooth = F.avg_pool2d(noise, kernel_size=k, stride=1, padding=k // 2)[:, 0]
            smooth = smooth.masked_fill(~candidates, float("-inf"))
            for bi in range(b):
                avail = int(candidates[bi].sum().item())
                kk = min(count, avail)
                if kk > 0:
                    idx = smooth[bi].flatten().topk(kk, largest=True, sorted=False).indices
                    out[bi].view(-1)[idx] = True
        return out

    def _make_structured_exogenous_corruption(
        self,
        anchor_prob: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stationary structured corruption independent of factual residual count.

        Four standard mask-corruption families are used: erosion-like FN,
        dilation-like FP, interior hole and local exterior island.  The *rate*
        is fixed by ``denoise_coverage`` rather than the instantaneous Base
        FN/FP count, so the denoising task does not shrink merely because Base
        becomes better during joint training.
        """
        anchor = anchor_prob.detach().clamp(EPS, 1.0 - EPS)
        gt = target.detach() >= 0.5
        pred = anchor[:, 0] >= 0.5
        b, h, w = gt.shape
        n = h * w
        budget = max(1, min(n, int(math.ceil(self.denoise_coverage * n))))
        corruption = torch.zeros_like(gt, dtype=torch.bool)
        op_id = torch.zeros((b,), dtype=torch.long, device=gt.device)
        cuda_devices = [int(gt.device.index)] if gt.is_cuda and gt.device.index is not None else []
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            radii = torch.randint(
                self.denoise_radius_min,
                self.denoise_radius_max + 1,
                (b,), device=gt.device,
            )
            ops = torch.randint(0, 4, (b,), device=gt.device)
        for bi in range(b):
            r = int(radii[bi].item())
            op = int(ops[bi].item())
            op_id[bi] = op
            gt_i = gt[bi:bi+1]
            pred_i = pred[bi:bi+1]
            correct_fg = gt_i & pred_i
            correct_bg = (~gt_i) & (~pred_i)
            eroded = self._erode(gt_i, r)
            dilated = self._dilate(gt_i, r)
            inner_band = gt_i & (~eroded)
            outer_band = dilated & (~gt_i)
            if op == 0:       # erosion-like FN
                cand = correct_fg & inner_band
                fallback = correct_fg
            elif op == 1:     # dilation-like FP
                cand = correct_bg & outer_band
                fallback = correct_bg
            elif op == 2:     # interior hole
                cand = correct_fg
                fallback = correct_fg
            else:             # local exterior island near boundary
                near = self._dilate(gt_i, max(r + 1, 2)) & (~gt_i)
                cand = correct_bg & near
                fallback = correct_bg
            if int(cand.sum().item()) == 0:
                cand = fallback
            if int(cand.sum().item()) == 0:
                # Degenerate all-one/all-zero or badly wrong anchor: choose any
                # currently correct point, still independent of factual *count*.
                cand = (pred_i == gt_i)
            chosen = self._structured_subset(cand, budget, max(r, 1))
            corruption[bi] = chosen[0]
        corrupted = torch.where(corruption[:, None], 1.0 - anchor, anchor)
        return corrupted.clamp(EPS, 1.0 - EPS), corruption[:, None], op_id

    def _make_exogenous_corruption(
        self,
        anchor_prob: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Historical R1 point corruption retained for backward compatibility."""
        anchor = anchor_prob.detach().clamp(EPS, 1.0 - EPS)
        gt = target.detach() >= 0.5
        pred = anchor[:, 0] >= 0.5
        factual_fn = gt & (~pred)
        factual_fp = (~gt) & pred
        b, h, w = gt.shape
        n = h * w
        budget = max(1, min(n, int(math.ceil(self.coverage * n))))
        fn_count = factual_fn.flatten(1).sum(1)
        fp_count = factual_fp.flatten(1).sum(1)
        total = (fn_count + fp_count).clamp_min(1)
        desired_total = torch.minimum(fn_count + fp_count, torch.full_like(fn_count, budget))
        desired_total = torch.where(desired_total > 0, desired_total, torch.ones_like(desired_total))
        desired_fn = torch.round(desired_total.float() * fn_count.float() / total.float()).long()
        desired_fp = (desired_total.long() - desired_fn).clamp_min(0)
        correct_fg = gt & pred
        correct_bg = (~gt) & (~pred)
        fn_mask = torch.zeros_like(gt, dtype=torch.bool)
        fp_mask = torch.zeros_like(gt, dtype=torch.bool)
        cuda_devices = [int(gt.device.index)] if gt.is_cuda and gt.device.index is not None else []
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            scores_fg = torch.rand_like(anchor[:,0]).masked_fill(~correct_fg, float("-inf"))
            scores_bg = torch.rand_like(anchor[:,0]).masked_fill(~correct_bg, float("-inf"))
            for bi in range(b):
                if int(desired_total[bi]) == 1 and int(fn_count[bi] + fp_count[bi]) == 0:
                    if int(correct_fg[bi].sum()) > 0:
                        desired_fn[bi] = 1; desired_fp[bi] = 0
                    elif int(correct_bg[bi].sum()) > 0:
                        desired_fn[bi] = 0; desired_fp[bi] = 1
                kf = min(int(desired_fn[bi].item()), int(correct_fg[bi].sum().item()))
                kb = min(int(desired_fp[bi].item()), int(correct_bg[bi].sum().item()))
                if kf > 0:
                    idx = scores_fg[bi].flatten().topk(kf, sorted=False).indices
                    fn_mask[bi].view(-1)[idx] = True
                if kb > 0:
                    idx = scores_bg[bi].flatten().topk(kb, sorted=False).indices
                    fp_mask[bi].view(-1)[idx] = True
        corruption = fn_mask | fp_mask
        corrupted = torch.where(corruption[:, None], 1.0 - anchor, anchor)
        return corrupted.clamp(EPS, 1.0 - EPS), corruption[:, None]

    @staticmethod
    def _r4_structured_subset(
        candidates: torch.Tensor,
        count: int,
        radius: int,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """R4-only coherent subset using a dedicated generator.

        Unlike the historical helper, this function never touches/forks the
        process-global RNG.  That prevents Stage-2 synthetic sampling from
        perturbing Base/Geometry stochastic trajectories and prevents accidental
        repetition caused by nested fork_rng restoration.
        """
        b, h, w = candidates.shape
        out = torch.zeros_like(candidates, dtype=torch.bool)
        count = max(0, int(count))
        if count <= 0:
            return out
        noise = torch.rand(
            (b, 1, h, w),
            device=candidates.device,
            generator=generator,
        )
        k = max(3, 2 * int(radius) + 1)
        if k % 2 == 0:
            k += 1
        smooth = F.avg_pool2d(noise, kernel_size=k, stride=1, padding=k // 2)[:, 0]
        smooth = smooth.masked_fill(~candidates, float("-inf"))
        for bi in range(b):
            avail = int(candidates[bi].sum().item())
            kk = min(count, avail)
            if kk > 0:
                idx = smooth[bi].flatten().topk(kk, largest=True, sorted=False).indices
                out[bi].view(-1)[idx] = True
        return out

    def _make_r4_stationary_flip_corruption(
        self,
        anchor_prob: torch.Tensor,
        target: torch.Tensor,
        inspection_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create a stationary, deployment-matched synthetic action view.

        R4.1 restricts GT cleaning and corruption to the same factual top-K
        inspection support used at inference.  This prevents training-time
        positives from being force-inserted outside the deployment domain.
        """
        anchor = anchor_prob.detach().clamp(EPS, 1.0 - EPS)
        gt = target.detach() >= 0.5
        if gt.ndim == 4:
            gt = gt[:, 0]
        b, h, w = gt.shape

        # Backward-compatibility contract:
        #   * legacy R4 (r41_enabled=False, inspection_mask=None) may corrupt
        #     anywhere in the image, but its synthetic positive budget is
        #     defined relative to the *deployment top-K inspection budget*,
        #     not the full HxW image.
        #   * R4.1 receives the factual top-K inspection mask explicitly and
        #     restricts both cleaning and corruption to that support.
        #
        # The first R4.1 implementation accidentally used H*W as inspect_k
        # when inspection_mask=None.  With coverage=5% and synth fraction=20%,
        # this created ~20% of the whole image as synthetic positives (e.g.
        # 115/576) while the sparse selector could keep only 29/576 points.
        # Consequently forced corruption was truncated by _fill_topk_with_forced
        # and the historical R4 contract corr subset dn_selection became false.
        if inspection_mask is None:
            inspection = torch.ones((b, h, w), dtype=torch.bool, device=gt.device)
            n = h * w
            inspect_k = max(1, min(n, int(math.ceil(self.coverage * n))))
        else:
            inspection = inspection_mask.detach().bool()
            if inspection.ndim == 4:
                inspection = inspection[:, 0]
            if inspection.shape != gt.shape:
                raise ValueError(
                    f"inspection_mask shape {tuple(inspection.shape)} != GT {tuple(gt.shape)}"
                )
            counts = inspection.flatten(1).sum(1)
            if int(counts.min().item()) <= 0:
                raise RuntimeError("R4.1 requires non-empty inspection support")
            if not torch.equal(counts, counts[:1].expand_as(counts)):
                raise RuntimeError(f"R4.1 requires equal K per image, got {counts.tolist()}")
            inspect_k = int(counts[0].item())

        edit_budget = max(
            1, min(inspect_k, int(round(self.r4_synth_edit_fraction * inspect_k)))
        )

        p_pos = anchor.new_tensor(0.5 + self.r4_target_prob_margin)
        p_neg = anchor.new_tensor(0.5 - self.r4_target_prob_margin)
        corrected_selected = torch.where(
            gt[:, None],
            torch.maximum(anchor, p_pos),
            torch.minimum(anchor, p_neg),
        )
        clean = torch.where(
            inspection[:, None], corrected_selected, anchor
        ).clamp(EPS, 1.0 - EPS)

        corruption = torch.zeros_like(gt, dtype=torch.bool)
        op_id = torch.zeros((b,), dtype=torch.long, device=gt.device)
        synth_step = int(self._r4_synth_step.item()) if hasattr(self, "_r4_synth_step") else 0
        seed = int((int(torch.initial_seed()) + 104729 * synth_step + 7919) % (2**63 - 1))
        gen = torch.Generator(device=gt.device)
        gen.manual_seed(seed)
        if self.training and hasattr(self, "_r4_synth_step"):
            self._r4_synth_step.add_(1)

        radii = torch.randint(
            self.denoise_radius_min,
            self.denoise_radius_max + 1,
            (b,), device=gt.device, generator=gen,
        )
        ops = torch.randint(0, 4, (b,), device=gt.device, generator=gen)
        for bi in range(b):
            r = int(radii[bi].item())
            op = int(ops[bi].item())
            op_id[bi] = op
            gt_i = gt[bi:bi+1]
            inspect_i = inspection[bi:bi+1]
            eroded = self._erode(gt_i, r)
            dilated = self._dilate(gt_i, r)
            inner_band = gt_i & (~eroded)
            outer_band = dilated & (~gt_i)
            if op == 0:
                cand, fallback = inner_band & inspect_i, gt_i & inspect_i
            elif op == 1:
                cand, fallback = outer_band & inspect_i, (~gt_i) & inspect_i
            elif op == 2:
                cand, fallback = gt_i & inspect_i, inspect_i
            else:
                near = self._dilate(gt_i, max(r + 1, 2)) & (~gt_i)
                cand, fallback = near & inspect_i, (~gt_i) & inspect_i

            if int(cand.sum().item()) == 0:
                cand = fallback
            if int(cand.sum().item()) == 0:
                cand = inspect_i

            chosen = self._r4_structured_subset(
                cand, edit_budget, max(r, 1), generator=gen
            )
            missing = edit_budget - int(chosen.sum().item())
            if missing > 0:
                fill = self._r4_structured_subset(
                    inspect_i & (~chosen), missing, max(r, 1), generator=gen
                )
                chosen = chosen | fill
            corruption[bi] = chosen[0]

        effective_max = (
            min(self.r4_corrupt_logit_max, self.r41_corrupt_logit_max)
            if self.r41_enabled else self.r4_corrupt_logit_max
        )
        mag = self.r4_corrupt_logit_min + (
            effective_max - self.r4_corrupt_logit_min
        ) * torch.rand(
            (b, 1, h, w), device=anchor.device, dtype=anchor.dtype, generator=gen
        )
        wrong_logit = torch.where(gt[:, None], -mag, mag)
        wrong_prob = torch.sigmoid(wrong_logit)
        corrupted = torch.where(corruption[:, None], wrong_prob, clean)
        return corrupted.clamp(EPS, 1.0 - EPS), corruption[:, None], op_id

    def _r41_gather_selected_patches(
        self,
        feature: torch.Tensor,
        selection: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather only selected P×P patches; never full-image F.unfold."""
        b, c, h, w = feature.shape
        flat_sel = selection.flatten(1).bool()
        counts = flat_sel.sum(dim=1)
        if not torch.equal(counts, counts[:1].expand_as(counts)):
            raise RuntimeError(f"R4.1 patch gather requires equal K, got {counts.tolist()}")
        k = int(counts[0].item())
        indices = torch.stack([
            torch.nonzero(flat_sel[bi], as_tuple=False).flatten() for bi in range(b)
        ], dim=0)
        if k <= 0:
            return feature.new_zeros(
                (b, 0, c * self.r4_patch_size * self.r4_patch_size)
            ), indices

        p = self.r4_patch_size
        pad = p // 2
        padded = F.pad(feature, (pad, pad, pad, pad))
        wp = w + 2 * pad
        y = torch.div(indices, w, rounding_mode="floor")
        x = indices.remainder(w)
        oy = torch.arange(p, device=feature.device, dtype=torch.long)
        ox = torch.arange(p, device=feature.device, dtype=torch.long)
        gy, gx = torch.meshgrid(oy, ox, indexing="ij")
        lin = (
            (y[:, :, None] + gy.reshape(1, 1, -1)) * wp
            + (x[:, :, None] + gx.reshape(1, 1, -1))
        )
        flat = padded.flatten(2)
        gather_idx = lin.flatten(1)[:, None, :].expand(-1, c, -1)
        chosen = torch.gather(flat, 2, gather_idx)
        chosen = chosen.view(b, c, k, p * p).permute(0, 2, 1, 3).contiguous()
        return chosen.flatten(2), indices

    def _r4_patch_logits(
        self,
        hidden: torch.Tensor,
        selection: torch.Tensor,
    ) -> torch.Tensor:
        """Predict FLIP/KEEP on local patches with a memory-safe R4.1 path."""
        if not self.r4_enabled or self.r4_patch_proj is None or self.r4_flip_head is None:
            raise RuntimeError("R4 patch action head requested while R4 is disabled")
        b, c, h, w = hidden.shape
        flat_sel = selection.flatten(1).bool()
        counts = flat_sel.sum(dim=1)
        if not torch.equal(counts, counts[:1].expand_as(counts)):
            raise RuntimeError(f"V4G-R4 patch head requires equal K per image, got {counts.tolist()}")
        k = int(counts[0].item())
        out = hidden.new_zeros((b, h * w))
        if k <= 0:
            return out.view(b, 1, h, w)

        if self.r41_enabled:
            if self.r41_patch_reduce is None:
                raise RuntimeError("R4.1 patch reducer is missing")
            reduced = self.r41_patch_reduce(hidden)
            chosen, indices = self._r41_gather_selected_patches(reduced, selection)
        else:
            patches = F.unfold(
                hidden,
                kernel_size=self.r4_patch_size,
                padding=self.r4_patch_size // 2,
            ).transpose(1, 2)
            indices = torch.stack([
                torch.nonzero(flat_sel[bi], as_tuple=False).flatten() for bi in range(b)
            ], dim=0)
            chosen = torch.gather(
                patches, 1, indices[:, :, None].expand(-1, -1, patches.shape[-1])
            )

        feat = self.r4_patch_proj(chosen)
        logits = self.r4_flip_head(feat)[:, :, 0]
        out.scatter_(1, indices, logits)
        return out.view(b, 1, h, w)

    def _r4_refine_from_hidden(
        self,
        anchor: torch.Tensor,
        hidden: torch.Tensor,
        selection: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Deploy a conservative FLIP/KEEP operator.

        The network never predicts a direction or magnitude. In binary
        segmentation, the current hard label determines the only admissible
        corrective direction. A confident FLIP moves the posterior only to the
        opposite side's fixed margin; KEEP is exact identity.
        """
        flip_logits = self._r4_patch_logits(hidden, selection)
        flip_prob = torch.sigmoid(flip_logits)
        selected = selection.bool()
        flip = selected & (flip_prob >= self.r4_flip_threshold)
        hard_fg = anchor >= 0.5
        add = flip & (~hard_fg)
        remove = flip & hard_fg
        p_add = anchor.new_tensor(0.5 + self.r4_target_prob_margin)
        p_remove = anchor.new_tensor(0.5 - self.r4_target_prob_margin)
        final_prob = anchor.clone()
        final_prob = torch.where(add, torch.maximum(final_prob, p_add), final_prob)
        final_prob = torch.where(remove, torch.minimum(final_prob, p_remove), final_prob)
        final_prob = final_prob.clamp(EPS, 1.0 - EPS)
        final_logits = torch.logit(final_prob)
        delta = final_logits - torch.logit(anchor.clamp(EPS, 1.0 - EPS))
        return final_logits, final_prob, delta, flip_logits, flip_prob, flip.to(anchor)

    def _sparse_delta(self, hidden: torch.Tensor, selection: torch.Tensor) -> torch.Tensor:
        """Evaluate the R2 correction MLP only on selected locations."""
        if not self.r2_enabled:
            return self.delta_out(self.refine_body(hidden))
        assert self.point_body is not None and self.point_delta is not None
        b, c, h, w = hidden.shape
        flat_hidden = hidden.flatten(2).transpose(1, 2)  # B,N,C
        flat_sel = selection.flatten(1).bool()
        counts = flat_sel.sum(dim=1)
        if not torch.equal(counts, counts[:1].expand_as(counts)):
            raise RuntimeError(f"V4G-R2 sparse head requires equal K per image, got {counts.tolist()}")
        k = int(counts[0].item())
        if k <= 0:
            return hidden.new_zeros((b, 1, h, w))
        indices = torch.stack([
            torch.nonzero(flat_sel[bi], as_tuple=False).flatten() for bi in range(b)
        ], dim=0)
        gather_idx = indices[:, :, None].expand(-1, -1, c)
        points = torch.gather(flat_hidden, 1, gather_idx)
        point_delta_raw = self.point_delta(self.point_body(points))[:, :, 0]
        # R3 turns the residual actuator into an explicit trust region.  R2 had
        # an unbounded scalar logit actuator; R3 cannot exceed the pre-declared
        # maximum even if the point MLP extrapolates.  Zero raw output remains
        # exact identity because tanh(0)=0.
        if self.r3_enabled:
            max_step = point_delta_raw.new_tensor(self.r3_max_abs_delta_logit)
            point_delta = max_step * torch.tanh(point_delta_raw / max_step)
        else:
            point_delta = point_delta_raw
        delta_flat = hidden.new_zeros((b, h * w)).scatter(1, indices, point_delta)
        return delta_flat.view(b, 1, h, w)

    def _refine_from_hidden(
        self,
        anchor: torch.Tensor,
        hidden: torch.Tensor,
        selection: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        delta = self._sparse_delta(hidden, selection)
        refined_logits = torch.logit(anchor.clamp(EPS, 1.0 - EPS)) + delta
        refined_prob = torch.sigmoid(refined_logits).clamp(EPS, 1.0 - EPS)
        final_prob = torch.where(selection, refined_prob, anchor).clamp(EPS, 1.0 - EPS)
        final_logits = torch.logit(final_prob)
        return refined_logits, refined_prob, final_logits, final_prob, delta

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
        supervision_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        anchor = anchor_prob.detach().clamp(EPS, 1.0 - EPS)
        if base_prob is None:
            base_prob = anchor
        if flow_px is None:
            flow_px = anchor.new_zeros((anchor.shape[0], 2, *anchor.shape[-2:]))
        # R4.1 has no factual Stage-2 training target.  During training its
        # factual branch is inference-only, so do not retain a useless autograd
        # graph; the synthetic branch below owns all R4.1 gradients.
        factual_no_grad = bool(self.r41_enabled and self.r4_enabled and self.training)
        factual_ctx = torch.no_grad() if factual_no_grad else torch.enable_grad()
        with factual_ctx:
            hidden, evidence = self._features(
                image, semantic_map, text_features, anchor, base_prob, flow_px,
                mc_std_map, mc_disagreement_map, fine_feature_map,
            )
            score = self._selection_score(evidence)
            selected = self._select_points(score, training=self.training)
            r4_flip_logits = torch.zeros_like(anchor)
            r4_flip_prob = torch.zeros_like(anchor)
            r4_flip_mask = torch.zeros_like(anchor)
            if self.r4_enabled:
                final_logits, final_prob, delta, r4_flip_logits, r4_flip_prob, r4_flip_mask = self._r4_refine_from_hidden(
                    anchor, hidden, selected
                )
                refined_logits = final_logits
                refined_prob = final_prob
            else:
                refined_logits, refined_prob, final_logits, final_prob, delta = self._refine_from_hidden(
                    anchor, hidden, selected
                )

        # Training-only exogenous denoising/action view. Validation/test never use GT.
        dn_anchor = anchor
        dn_corruption = torch.zeros_like(anchor, dtype=torch.bool)
        dn_selected = torch.zeros_like(anchor, dtype=torch.bool)
        dn_refined_logits = torch.logit(anchor)
        dn_refined_prob = anchor
        dn_final_prob = anchor
        dn_delta = torch.zeros_like(anchor)
        dn_op_id = torch.full((anchor.shape[0],), -1, dtype=torch.long, device=anchor.device)
        r4_synth_flip_logits = torch.zeros_like(anchor)
        r4_synth_flip_prob = torch.zeros_like(anchor)
        r4_synth_flip_mask = torch.zeros_like(anchor)
        r4_synth_target = torch.zeros_like(anchor)
        # R4 always trains from its stationary exogenous action task, even
        # though the historical V4G denoising objective remains disabled.
        use_train_view = self.training and isinstance(supervision_masks, torch.Tensor) and (self.denoise_enabled or self.r4_enabled)
        if use_train_view:
            target = supervision_masks.detach()
            if target.ndim == 3:
                target = target[:, None]
            target = self._resize(target.float(), tuple(anchor.shape[-2:]), nearest=True)
            if self.r4_enabled:
                dn_anchor, dn_corruption, dn_op_id = self._make_r4_stationary_flip_corruption(
                    anchor,
                    target[:, 0],
                    inspection_mask=selected if self.r41_enabled else None,
                )
                r4_synth_target = dn_corruption.to(anchor)
            elif self.r2_enabled:
                dn_anchor, dn_corruption, dn_op_id = self._make_structured_exogenous_corruption(
                    anchor, target[:, 0]
                )
            else:
                dn_anchor, dn_corruption = self._make_exogenous_corruption(anchor, target[:, 0])
            dn_hidden, dn_evidence = self._features(
                image, semantic_map, text_features, dn_anchor, base_prob, flow_px,
                mc_std_map, mc_disagreement_map, fine_feature_map,
                trace_override=evidence["trace"] if self.r41_enabled else None,
            )
            if self.r41_enabled:
                # Train and deploy on identical factual top-K coordinates.
                dn_selected = selected.detach().bool()
            else:
                dn_score = self._selection_score(dn_evidence)
                dn_selected = self._select_points(dn_score, training=True, forced=dn_corruption)
            if self.r4_enabled:
                dn_final_logits, dn_final_prob, dn_delta, r4_synth_flip_logits, r4_synth_flip_prob, r4_synth_flip_mask = self._r4_refine_from_hidden(
                    dn_anchor, dn_hidden, dn_selected
                )
                dn_refined_logits = dn_final_logits
                dn_refined_prob = dn_final_prob
            else:
                dn_refined_logits, dn_refined_prob, dn_final_logits, dn_final_prob, dn_delta = self._refine_from_hidden(
                    dn_anchor, dn_hidden, dn_selected
                )

        return {
            "logits": final_logits,
            "prob": final_prob,
            "selection_mask": selected.to(anchor),
            "selection_score": score,
            "refined_logits": refined_logits,
            "refined_prob": refined_prob,
            "delta_logit": delta,
            "margin_uncertainty": evidence["margin"],
            "mc_std_map": evidence["mc_std"],
            "mc_disagreement_map": evidence["mc_disagreement"],
            "entropy_map": evidence["entropy"],
            "trace": evidence["trace"],
            "fine_feature_map": evidence["fine_feature"],
            "dn_anchor_prob": dn_anchor,
            "dn_corruption_mask": dn_corruption.to(anchor),
            "dn_selection_mask": dn_selected.to(anchor),
            "dn_refined_logits": dn_refined_logits,
            "dn_refined_prob": dn_refined_prob,
            "dn_final_prob": dn_final_prob,
            "dn_delta_logit": dn_delta,
            "dn_op_id": dn_op_id,
            "r4_flip_logits": r4_flip_logits,
            "r4_flip_prob": r4_flip_prob,
            "r4_flip_mask": r4_flip_mask,
            "r4_synth_flip_logits": r4_synth_flip_logits,
            "r4_synth_flip_prob": r4_synth_flip_prob,
            "r4_synth_flip_mask": r4_synth_flip_mask,
            "r4_synth_target": r4_synth_target,
            "r2_enabled": anchor.new_full((anchor.shape[0],), float(self.r2_enabled)),
            "r3_enabled": anchor.new_full((anchor.shape[0],), float(self.r3_enabled)),
            "r4_enabled": anchor.new_full((anchor.shape[0],), float(self.r4_enabled)),
            "r41_enabled": anchor.new_full((anchor.shape[0],), float(self.r41_enabled)),
        }

class ErrorLocalizedReconstructionStage(nn.Module):
    """GEOTR Stage-2 with an optional V4D root-fix training contract.

    V4C compatibility path
    ----------------------
    When ``v4d_enabled=False`` this module reproduces the V4C operator:

        P_final = P_anchor
                  + q_FN.detach() * a_FN * (1 - P_anchor)
                  - q_FP.detach() * a_FP * P_anchor

    V4D root-fix path
    -----------------
    V4D separates three quantities that were coupled in V4C:

      * WHERE/TYPE: a diagnostic/auxiliary Correct/FN/FP classifier;
      * SEVERITY: two calibrated continuous residual-severity fields;
      * HOW: two non-negative logit-space correction magnitudes.

    The deployed V4D operator is

        Z_final = Z_anchor + S_FN * A_FN - S_FP * A_FP
        P_final = sigmoid(Z_final)

    GEOTR-V4E operator-consistent path
    ----------------------------------
    V4E removes the mathematically inconsistent ``severity * full-dose``
    composition.  A single hard typed support H determines WHERE+direction and
    A predicts the *complete* monotonic logit correction dose:

        H = one_hot(argmax p(Correct/FN/FP))
        Z_final = Z_anchor + H_FN.detach() * A_FN - H_FP.detach() * A_FP

    The GT-supported teacher uses the exact same operator with H replaced by GT
    FN/FP support.  Therefore magnitude supervision, teacher execution, and
    deployment all solve the same correction problem.  Magnitude is also
    explicitly supervised toward zero off its action support, so a false
    support prediction does not inherit an unconstrained large dose.

    GEOTR-V4F selective-intervention path
    --------------------------------------
    V4F explicitly separates three responsibilities that V4E still coupled:

      * PROPOSAL: the imbalance-aware Correct/FN/FP head is used only to rank
        sparse residual candidates; its weighted score is never interpreted as
        an execution probability.
      * EXECUTION: a separate NoEdit/Edit policy plus Add/Remove direction head
        decides whether a candidate is safe to intervene on. During training a
        soft expected action is used so the final segmentation loss directly
        optimizes the execution policy; validation/test use a hard thresholded
        action.
      * DOSE: two bounded fractional correction fields in [0,1] determine only
        how much probability mass to add/remove *conditional on editing*. The
        dose branch is supervised on true edit support and is not asked to own
        NoEdit.

    V4F also conditions residual reasoning on a detached Transport trace
    [Base, Anchor, Anchor-Base, flow_x, flow_y, |flow|]. For residual-only A2
    the trace reduces to Base=Anchor and zero flow; for A3 it tells Stage-2
    explicitly what Geometry already moved.

    All image/semantic/text/anchor/trace evidence is detached at the Stage-2 boundary;
    therefore no Stage-2 objective can change Base/PVL or Stage-1 Transport.
    """

    def __init__(
        self,
        hidden_dim: int,
        semantic_channels: int,
        text_dim: int,
        error_prior: float = 0.03,
        fn_prior_fraction: float = 0.50,
        magnitude_init: float = 0.02,
        *,
        v4d_enabled: bool = False,
        v4e_enabled: bool = False,
        v4f_enabled: bool = False,
        severity_prior: float = 0.03,
        logit_magnitude_init: float = 0.25,
        max_logit_step: float = 4.0,
        severity_grad_scale: float = 0.25,
        v4f_proposal_coverage: float = 0.10,
        v4f_execution_prior: float = 0.20,
        v4f_execution_threshold: float = 0.65,
        v4f_dose_init: float = 0.25,
        v4f_flow_scale_px: float = 8.0,
        v4f_deploy_dose_grad_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.semantic_channels = int(semantic_channels)
        self.text_dim = int(text_dim)
        self.v4d_enabled = bool(v4d_enabled)
        self.v4e_enabled = bool(v4e_enabled)
        self.v4f_enabled = bool(v4f_enabled)
        self.error_prior = float(min(max(error_prior, 1.0e-4), 0.49))
        self.fn_prior_fraction = float(min(max(fn_prior_fraction, 1.0e-3), 1.0 - 1.0e-3))
        self.magnitude_init = float(min(max(magnitude_init, 1.0e-4), 1.0 - 1.0e-4))
        self.severity_prior = float(min(max(severity_prior, 1.0e-4), 1.0 - 1.0e-4))
        self.logit_magnitude_init = float(max(logit_magnitude_init, 1.0e-4))
        self.max_logit_step = float(max(max_logit_step, 1.0e-3))
        self.severity_grad_scale = float(min(max(severity_grad_scale, 0.0), 1.0))
        self.v4f_proposal_coverage = float(min(max(v4f_proposal_coverage, 1.0e-4), 1.0))
        self.v4f_execution_prior = float(min(max(v4f_execution_prior, 1.0e-4), 1.0 - 1.0e-4))
        self.v4f_execution_threshold = float(min(max(v4f_execution_threshold, 0.0), 1.0))
        self.v4f_dose_init = float(min(max(v4f_dose_init, 1.0e-4), 1.0 - 1.0e-4))
        self.v4f_flow_scale_px = float(max(v4f_flow_scale_px, 1.0e-3))
        self.v4f_deploy_dose_grad_scale = float(min(max(v4f_deploy_dose_grad_scale, 0.0), 1.0))

        self.image_stem = nn.Sequential(
            ConvNormGELU(3, hidden_dim),
            ConvNormGELU(hidden_dim, hidden_dim),
        )
        self.semantic_proj = nn.Sequential(
            nn.Conv2d(semantic_channels, hidden_dim, 1, bias=False),
            nn.GroupNorm(_groups(hidden_dim), hidden_dim),
            nn.GELU(),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.pixel_fuse = nn.Sequential(
            ConvNormGELU(2 * hidden_dim + 3, hidden_dim),
            ConvNormGELU(hidden_dim, hidden_dim),
        )
        self.global_context = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.context_film = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.context_gate = nn.Parameter(torch.zeros(()))
        self.trunk = nn.Sequential(
            ConvNormGELU(hidden_dim, hidden_dim),
            ConvNormGELU(hidden_dim, hidden_dim),
        )

        # V4F transport trace and execution policy.  These modules exist only
        # for V4F so older checkpoints/ablations keep their exact tensor shapes.
        if self.v4f_enabled:
            self.v4f_trace_fuse = nn.Sequential(
                ConvNormGELU(hidden_dim + 6, hidden_dim),
                ConvNormGELU(hidden_dim, hidden_dim),
            )
            self.v4f_policy_body = nn.Sequential(
                ConvNormGELU(hidden_dim, hidden_dim),
                ConvNormGELU(hidden_dim, hidden_dim),
            )
            self.v4f_edit_out = nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)
            self.v4f_direction_out = nn.Conv2d(hidden_dim, 2, kernel_size=3, padding=1)
            nn.init.zeros_(self.v4f_edit_out.weight)
            nn.init.constant_(
                self.v4f_edit_out.bias,
                math.log(self.v4f_execution_prior / (1.0 - self.v4f_execution_prior)),
            )
            nn.init.zeros_(self.v4f_direction_out.weight)
            nn.init.zeros_(self.v4f_direction_out.bias)
        else:
            self.v4f_trace_fuse = None
            self.v4f_policy_body = None
            self.v4f_edit_out = None
            self.v4f_direction_out = None

        # WHERE/TYPE.  Historical owner names are retained for optimizer and
        # checkpoint compatibility.
        self.error_body = nn.Sequential(
            ConvNormGELU(hidden_dim, hidden_dim),
            ConvNormGELU(hidden_dim, hidden_dim),
        )
        self.error_out = nn.Conv2d(hidden_dim, 3, kernel_size=3, padding=1)

        # HOW branch.  V4C interprets these two channels through sigmoid in
        # probability space; V4D interprets them through softplus as non-negative
        # logit-space steps.
        self.correction_body = nn.Sequential(
            ConvNormGELU(hidden_dim, hidden_dim),
            ConvNormGELU(hidden_dim, hidden_dim),
        )
        self.correction_out = nn.Conv2d(hidden_dim, 2, kernel_size=3, padding=1)

        # V4D adds an explicit continuous severity branch.  It is deliberately
        # not the class-weighted CE probability: localization score and edit
        # amplitude are different statistical quantities.
        if self.v4d_enabled or self.v4e_enabled:
            self.severity_body = nn.Sequential(
                ConvNormGELU(hidden_dim, hidden_dim),
                ConvNormGELU(hidden_dim, hidden_dim),
            )
            self.severity_out = nn.Conv2d(hidden_dim, 2, kernel_size=3, padding=1)
        else:
            self.severity_body = None
            self.severity_out = None

        # V4E removes continuous severity from both objective and deployment.
        # Keep the submodule only for state-dict compatibility, but freeze it so
        # the optimizer/trainable count does not include dead parameters.
        if self.v4e_enabled and self.severity_body is not None and self.severity_out is not None:
            for _p in self.severity_body.parameters():
                _p.requires_grad_(False)
            for _p in self.severity_out.parameters():
                _p.requires_grad_(False)

        nn.init.zeros_(self.error_out.weight)
        p_fn = self.error_prior * self.fn_prior_fraction
        p_fp = self.error_prior * (1.0 - self.fn_prior_fraction)
        p_ok = 1.0 - p_fn - p_fp
        prior = torch.tensor([p_ok, p_fn, p_fp], dtype=self.error_out.bias.dtype)
        with torch.no_grad():
            self.error_out.bias.copy_(prior.clamp_min(1.0e-8).log())

        nn.init.zeros_(self.correction_out.weight)
        if self.v4f_enabled:
            # V4F predicts bounded fractional doses alpha in [0,1].
            dose_bias = math.log(self.v4f_dose_init / (1.0 - self.v4f_dose_init))
            nn.init.constant_(self.correction_out.bias, dose_bias)
        elif self.v4d_enabled or self.v4e_enabled:
            # softplus^{-1}(x) = log(expm1(x)); this gives a non-saturated,
            # positive initial logit step while direct magnitude supervision is
            # already available from the first update.
            mag_bias = math.log(math.expm1(self.logit_magnitude_init))
            nn.init.constant_(self.correction_out.bias, mag_bias)
            assert self.severity_out is not None
            nn.init.zeros_(self.severity_out.weight)
            sev_bias = math.log(self.severity_prior / (1.0 - self.severity_prior))
            nn.init.constant_(self.severity_out.bias, sev_bias)
        else:
            mag_bias = math.log(self.magnitude_init / (1.0 - self.magnitude_init))
            nn.init.constant_(self.correction_out.bias, mag_bias)

    @staticmethod
    def _resize(x: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] == hw:
            return x
        return F.interpolate(x, size=hw, mode="bilinear", align_corners=False)

    @staticmethod
    def _boundary(prob: torch.Tensor) -> torch.Tensor:
        maxp = F.max_pool2d(prob, 3, stride=1, padding=1)
        minp = -F.max_pool2d(-prob, 3, stride=1, padding=1)
        return (maxp - minp).clamp(0.0, 1.0)

    @staticmethod
    def _uncertainty(prob: torch.Tensor) -> torch.Tensor:
        return (4.0 * prob * (1.0 - prob)).clamp(0.0, 1.0)

    def _features(
        self,
        image: torch.Tensor,
        semantic_map: torch.Tensor,
        text_features: torch.Tensor,
        anchor_prob: torch.Tensor,
    ) -> torch.Tensor:
        hw = tuple(anchor_prob.shape[-2:])
        # Hard causal boundary: Stage-2 cannot change Base/PVL or Transport.
        image = image.detach()
        semantic_map = semantic_map.detach()
        text_features = text_features.detach()
        anchor_prob = anchor_prob.detach()

        image_latent = self.image_stem(self._resize(image, hw))
        semantic_latent = self.semantic_proj(self._resize(semantic_map, hw))
        text = self.text_proj(text_features.float())
        uncertainty = self._uncertainty(anchor_prob)
        boundary = self._boundary(anchor_prob)
        pixel = self.pixel_fuse(
            torch.cat(
                [image_latent, semantic_latent, anchor_prob, uncertainty, boundary],
                dim=1,
            )
        )
        global_visual = F.adaptive_avg_pool2d(pixel, 1).flatten(1)
        context = self.global_context(torch.cat([global_visual, text], dim=1))
        gamma, beta = self.context_film(context).chunk(2, dim=1)
        gate = torch.tanh(self.context_gate)
        pixel = pixel + gate * (
            torch.tanh(gamma)[:, :, None, None] * pixel + beta[:, :, None, None]
        )
        return self.trunk(pixel)

    @staticmethod
    def _topk_mask(score: torch.Tensor, coverage: float) -> torch.Tensor:
        """Per-image fixed-coverage proposal mask; ranking only, never a probability."""
        if score.ndim != 4 or score.shape[1] != 1:
            raise ValueError(f"Expected score [B,1,H,W], got {tuple(score.shape)}")
        b, _, h, w = score.shape
        n = h * w
        k = max(1, min(n, int(math.ceil(float(coverage) * n))))
        flat = score.detach().flatten(1)
        idx = flat.topk(k, dim=1, largest=True, sorted=False).indices
        mask = torch.zeros_like(flat, dtype=torch.bool)
        mask.scatter_(1, idx, True)
        return mask.view(b, 1, h, w)

    def _v4f_trace(
        self,
        base_prob: torch.Tensor,
        anchor_prob: torch.Tensor,
        flow_px: torch.Tensor,
    ) -> torch.Tensor:
        base = base_prob.detach().clamp(EPS, 1.0 - EPS)
        anchor = anchor_prob.detach().clamp(EPS, 1.0 - EPS)
        flow = flow_px.detach() / self.v4f_flow_scale_px
        flow = flow.clamp(-1.0, 1.0)
        mag = torch.sqrt(flow[:, 0:1].square() + flow[:, 1:2].square() + 1.0e-12).clamp(0.0, 1.0)
        return torch.cat([base, anchor, anchor - base, flow[:, 0:1], flow[:, 1:2], mag], dim=1)

    def forward(
        self,
        anchor_prob: torch.Tensor,
        image: torch.Tensor,
        semantic_map: torch.Tensor,
        text_features: torch.Tensor,
        *,
        base_prob: Optional[torch.Tensor] = None,
        flow_px: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        anchor = anchor_prob.detach().clamp(EPS, 1.0 - EPS)
        hidden = self._features(image, semantic_map, text_features, anchor)
        if self.v4f_enabled:
            if base_prob is None:
                base_prob = anchor
            if flow_px is None:
                flow_px = anchor.new_zeros((anchor.shape[0], 2, anchor.shape[-2], anchor.shape[-1]))
            if flow_px.shape[-2:] != anchor.shape[-2:]:
                flow_px = self._resize(flow_px, tuple(anchor.shape[-2:]))
            trace = self._v4f_trace(base_prob, anchor, flow_px)
            assert self.v4f_trace_fuse is not None
            hidden = self.v4f_trace_fuse(torch.cat([hidden, trace], dim=1))
        else:
            trace = anchor.new_zeros((anchor.shape[0], 6, anchor.shape[-2], anchor.shape[-1]))

        typed_logits = self.error_out(self.error_body(hidden))
        typed_prob = torch.softmax(typed_logits.float(), dim=1).to(anchor)
        q_correct = typed_prob[:, 0:1]
        q_fn = typed_prob[:, 1:2]
        q_fp = typed_prob[:, 2:3]
        q_error = (q_fn + q_fp).clamp(EPS, 1.0 - EPS)
        error_logits = torch.logit(q_error)

        raw = self.correction_out(self.correction_body(hidden))
        # V4F diagnostics default to inert tensors for backward-compatible keys.
        candidate_mask = torch.zeros_like(anchor, dtype=torch.bool)
        edit_logit = torch.full_like(anchor, -20.0)
        edit_prob = torch.zeros_like(anchor)
        direction_logits = torch.zeros(anchor.shape[0], 2, *anchor.shape[-2:], device=anchor.device, dtype=anchor.dtype)
        direction_prob = torch.full_like(direction_logits, 0.5)
        p_no = torch.ones_like(anchor)
        p_add = torch.zeros_like(anchor)
        p_remove = torch.zeros_like(anchor)
        soft_prob = anchor
        hard_prob = anchor
        hard_edit = torch.zeros_like(anchor, dtype=torch.bool)
        hard_add = torch.zeros_like(anchor, dtype=torch.bool)
        hard_remove = torch.zeros_like(anchor, dtype=torch.bool)

        if not (self.v4d_enabled or self.v4e_enabled or self.v4f_enabled):
            # Exact V4C compatibility path.
            add_magnitude = torch.sigmoid(raw[:, 0:1])
            remove_magnitude = torch.sigmoid(raw[:, 1:2])
            severity_fn = q_fn
            severity_fp = q_fp
            severity_logits = torch.cat([
                torch.logit(q_fn.clamp(EPS, 1.0 - EPS)),
                torch.logit(q_fp.clamp(EPS, 1.0 - EPS)),
            ], dim=1)
            add_delta = q_fn.detach() * add_magnitude * (1.0 - anchor)
            remove_delta = q_fp.detach() * remove_magnitude * anchor
            gated_delta = add_delta - remove_delta
            corrected = (anchor + gated_delta).clamp(EPS, 1.0 - EPS)
            logits = torch.logit(corrected)
            add_logit_delta = torch.zeros_like(add_delta)
            remove_logit_delta = torch.zeros_like(remove_delta)
        elif self.v4f_enabled:
            # V4F: proposal ranking is deliberately separated from execution.
            # The weighted proposal score only determines a fixed-coverage
            # candidate pool. A separately supervised policy decides NoEdit/Edit
            # and Add/Remove; bounded fractional dose owns only HOW MUCH.
            candidate_mask = self._topk_mask(q_error, self.v4f_proposal_coverage)
            assert self.v4f_policy_body is not None
            assert self.v4f_edit_out is not None and self.v4f_direction_out is not None
            policy_hidden = self.v4f_policy_body(hidden)
            edit_logit = self.v4f_edit_out(policy_hidden)
            edit_prob = torch.sigmoid(edit_logit.float()).to(anchor)
            direction_logits = self.v4f_direction_out(policy_hidden)
            direction_prob = torch.softmax(direction_logits.float(), dim=1).to(anchor)

            dose_add = torch.sigmoid(raw[:, 0:1])
            dose_remove = torch.sigmoid(raw[:, 1:2])
            # Responsibility contract: teacher/GT-edit support owns dose learning.
            # Deployment utility trains policy by default, not a second NoEdit
            # mechanism hidden in the dose. A small non-zero scale is available
            # only as an explicit later ablation. Forward values are unchanged.
            dg = self.v4f_deploy_dose_grad_scale
            dose_add_deploy = dose_add.detach() + dg * (dose_add - dose_add.detach())
            dose_remove_deploy = dose_remove.detach() + dg * (dose_remove - dose_remove.detach())
            p_add_action = (anchor + dose_add_deploy * (1.0 - anchor)).clamp(EPS, 1.0 - EPS)
            p_remove_action = (anchor - dose_remove_deploy * anchor).clamp(EPS, 1.0 - EPS)

            candidate_f = candidate_mask.to(anchor)
            p_edit = candidate_f * edit_prob
            p_add = p_edit * direction_prob[:, 0:1]
            p_remove = p_edit * direction_prob[:, 1:2]
            p_no = (1.0 - p_add - p_remove).clamp(0.0, 1.0)
            soft_prob = (p_no * anchor + p_add * p_add_action + p_remove * p_remove_action).clamp(EPS, 1.0 - EPS)

            hard_edit = candidate_mask & (edit_prob >= self.v4f_execution_threshold)
            hard_dir = direction_prob.detach().argmax(dim=1, keepdim=True)
            hard_add = hard_edit & (hard_dir == 0)
            hard_remove = hard_edit & (hard_dir == 1)
            hard_prob = torch.where(hard_add, p_add_action, anchor)
            hard_prob = torch.where(hard_remove, p_remove_action, hard_prob).clamp(EPS, 1.0 - EPS)

            corrected = soft_prob if self.training else hard_prob
            logits = torch.logit(corrected)
            add_magnitude = dose_add
            remove_magnitude = dose_remove
            add_delta = (corrected - anchor).clamp_min(0.0)
            remove_delta = (anchor - corrected).clamp_min(0.0)
            add_logit_delta = torch.logit(torch.maximum(anchor, corrected)) - torch.logit(anchor)
            remove_logit_delta = torch.logit(anchor) - torch.logit(torch.minimum(anchor, corrected))
            gated_delta = corrected - anchor

            # Compatibility aliases: V4F no longer uses continuous severity.
            severity_fn = p_add
            severity_fp = p_remove
            severity_logits = torch.cat([edit_logit, direction_logits[:, 0:1] - direction_logits[:, 1:2]], dim=1)
        elif self.v4e_enabled:
            # V4E: one variable owns support/type, one variable owns the full
            # correction dose.  No continuous severity multiplier is used.
            pred_cls = typed_prob.detach().argmax(dim=1, keepdim=True)
            support_fn = (pred_cls == 1).to(anchor)
            support_fp = (pred_cls == 2).to(anchor)

            add_magnitude = F.softplus(raw[:, 0:1]).clamp_max(self.max_logit_step)
            remove_magnitude = F.softplus(raw[:, 1:2]).clamp_max(self.max_logit_step)
            anchor_logits = torch.logit(anchor)
            add_logit_delta = support_fn * add_magnitude
            remove_logit_delta = support_fp * remove_magnitude
            logits = anchor_logits + add_logit_delta - remove_logit_delta
            corrected = torch.sigmoid(logits).clamp(EPS, 1.0 - EPS)

            # Compatibility diagnostics: in V4E these fields are *hard deploy
            # support*, not a continuous severity quantity.
            severity_fn = support_fn
            severity_fp = support_fp
            severity_logits = torch.cat([
                torch.where(support_fn > 0, anchor.new_tensor(8.0), anchor.new_tensor(-8.0)),
                torch.where(support_fp > 0, anchor.new_tensor(8.0), anchor.new_tensor(-8.0)),
            ], dim=1)

            add_only = torch.sigmoid(anchor_logits + add_logit_delta)
            remove_only = torch.sigmoid(anchor_logits - remove_logit_delta)
            add_delta = (add_only - anchor).clamp_min(0.0)
            remove_delta = (anchor - remove_only).clamp_min(0.0)
            gated_delta = corrected - anchor
        else:
            assert self.severity_body is not None and self.severity_out is not None
            severity_logits = self.severity_out(self.severity_body(hidden))
            severity = torch.sigmoid(severity_logits.float()).to(anchor)
            severity_fn = severity[:, 0:1]
            severity_fp = severity[:, 1:2]

            # Positive logit-space HOW.  Unlike V4C's sigmoid(0.02), softplus
            # does not start in a near-saturated low-gradient regime.  A finite
            # cap is only a numerical/safety bound; direct targets lie below it.
            add_magnitude = F.softplus(raw[:, 0:1]).clamp_max(self.max_logit_step)
            remove_magnitude = F.softplus(raw[:, 1:2]).clamp_max(self.max_logit_step)

            # Forward value equals severity exactly; only the deployment-loss
            # gradient into severity is scaled.  Direct severity supervision is
            # unaffected because it reads ``severity_fn/fp`` above.
            sg = self.severity_grad_scale
            severity_fn_deploy = severity_fn.detach() + sg * (severity_fn - severity_fn.detach())
            severity_fp_deploy = severity_fp.detach() + sg * (severity_fp - severity_fp.detach())

            anchor_logits = torch.logit(anchor)
            add_logit_delta = severity_fn_deploy * add_magnitude
            remove_logit_delta = severity_fp_deploy * remove_magnitude
            logits = anchor_logits + add_logit_delta - remove_logit_delta
            corrected = torch.sigmoid(logits).clamp(EPS, 1.0 - EPS)

            # Probability-space deltas are diagnostics only.  The deployed
            # operator itself is logit-space and monotonic by construction.
            add_only = torch.sigmoid(anchor_logits + add_logit_delta)
            remove_only = torch.sigmoid(anchor_logits - remove_logit_delta)
            add_delta = (add_only - anchor).clamp_min(0.0)
            remove_delta = (anchor - remove_only).clamp_min(0.0)
            gated_delta = corrected - anchor

        strength = add_magnitude - remove_magnitude
        return {
            "logits": logits,
            "prob": corrected,
            "typed_logits": typed_logits,
            "typed_prob": typed_prob,
            "q_correct": q_correct,
            "q_fn": q_fn,
            "q_fp": q_fp,
            "error_logits": error_logits,
            "error_prob": q_error,
            "severity_logits": severity_logits,
            "severity_fn": severity_fn,
            "severity_fp": severity_fp,
            "raw": raw,
            "strength": strength,
            "add_magnitude": add_magnitude,
            "remove_magnitude": remove_magnitude,
            "add_logit_delta": add_logit_delta,
            "remove_logit_delta": remove_logit_delta,
            "add_delta": add_delta,
            "remove_delta": remove_delta,
            "gated_delta": gated_delta,
            "abs_change": gated_delta.abs().mean(dim=(1, 2, 3)),
            "v4f_trace": trace,
            "v4f_proposal_candidate_mask": candidate_mask.to(anchor),
            "v4f_policy_edit_logit": edit_logit,
            "v4f_policy_edit_prob": edit_prob,
            "v4f_policy_direction_logits": direction_logits,
            "v4f_policy_direction_prob": direction_prob,
            "v4f_policy_p_no": p_no,
            "v4f_policy_p_add": p_add,
            "v4f_policy_p_remove": p_remove,
            "v4f_soft_prob": soft_prob,
            "v4f_hard_prob": hard_prob,
            "v4f_hard_edit": hard_edit.to(anchor),
            "v4f_hard_add": hard_add.to(anchor),
            "v4f_hard_remove": hard_remove.to(anchor),
            "v4f_dose_add": add_magnitude,
            "v4f_dose_remove": remove_magnitude,
        }

class MultiHypothesisCompositionalSegmenter(nn.Module):
    """Compatibility wrapper implementing A0--A3 GEOTR-V4C/V4D."""

    use_semantic_feature = True
    unified_m1_safe_fusion_enabled = True
    mhcs_root_complete = True
    mhcs_geometry_topology_refinement = True
    mhcs_counterfactual_safety_r52 = False

    def __init__(self, cfg) -> None:
        super().__init__()
        m1 = _cfg_get(cfg, "M1", None)
        self.mode = str(_cfg_get(m1, "GEOTOPO_MODE", "full")).strip().lower()
        if self.mode not in {"base", "geometry", "residual", "full"}:
            raise ValueError(
                f"GEOTOPO_MODE must be one of base/geometry/residual/full, got {self.mode!r}"
            )
        self.aefr_enabled = bool(_cfg_get(m1, "GEOTR_AEFR_ENABLED", False))
        self.aefr_stage = str(_cfg_get(m1, "GEOTR_AEFR_STAGE", "single_atomic")).strip().lower()
        self.aefr_joint_geometry_grad = bool(_cfg_get(m1, "GEOTR_AEFR_JOINT_GEOMETRY_GRAD_ENABLED", False))
        self.aefr_transition_aware = bool(_cfg_get(m1, "GEOTR_AEFR_TRANSITION_AWARE_ENABLED", False))
        self.aefr_use_raw_flow_evidence = bool(_cfg_get(m1, "GEOTR_AEFR_USE_RAW_FLOW_EVIDENCE", True))
        stopgrad_transport = bool(_cfg_get(m1, "GEOTR_STAGE2_STOPGRAD_TRANSPORT", True))
        if not bool(_cfg_get(m1, "GEOTR_ERROR_LOCALIZATION_ENABLED", True)):
            raise ValueError("GEOTR-V4C requires GEOTR_ERROR_LOCALIZATION_ENABLED=true")
        if self.aefr_enabled:
            if self.aefr_stage not in (ActionEvidenceFactorizedROIRefiner.VALID_STAGES | {"sparse_local_rerendering"}):
                raise ValueError(f"Unknown GEOTR_AEFR_STAGE={self.aefr_stage!r}")
            if self.aefr_joint_geometry_grad and stopgrad_transport:
                raise ValueError("AEFR joint-Geometry mode requires GEOTR_STAGE2_STOPGRAD_TRANSPORT=false")
            if (not self.aefr_joint_geometry_grad) and (not stopgrad_transport):
                raise ValueError("AEFR stop-gradient ablations require GEOTR_STAGE2_STOPGRAD_TRANSPORT=true")
        elif not stopgrad_transport:
            raise ValueError("GEOTR-V4C requires GEOTR_STAGE2_STOPGRAD_TRANSPORT=true outside AEFR joint mode")
        if not bool(_cfg_get(m1, "GEOTR_TYPED_RESIDUAL_ENABLED", True)):
            raise ValueError("GEOTR-V4C/V4D requires GEOTR_TYPED_RESIDUAL_ENABLED=true")
        self.v4g_sparse_direct_refiner = bool(_cfg_get(m1, "GEOTR_V4G_SPARSE_DIRECT_REFINER_ENABLED", False))
        self.sparc_hr_enabled = bool(_cfg_get(m1, "GEOTR_SPARC_HR_ENABLED", False))
        self.c2r_enabled = bool(_cfg_get(m1, "GEOTR_C2R_ENABLED", False))
        self.c2r_v2_enabled = bool(_cfg_get(m1, "GEOTR_C2R_CANONICAL_ROI_ENABLED", False))
        self.pc2r_v3_enabled = bool(_cfg_get(m1, "GEOTR_PC2R_POSTERIOR_V3_ENABLED", False))
        self.pc2r_v31_enabled = bool(_cfg_get(m1, "GEOTR_PC2R_CANONICAL_COORD_V31_ENABLED", False))
        self.pc2r_v32_enabled = bool(_cfg_get(m1, "GEOTR_PC2R_OPERATOR_ALIGNED_V32_ENABLED", False))
        if self.aefr_enabled and (self.pc2r_v3_enabled or self.pc2r_v31_enabled or self.pc2r_v32_enabled):
            raise ValueError("GEOTR_AEFR_ENABLED is an alternative Stage-2 contract; disable PC2R-v3/v3.1/v3.2 flags")
        if self.pc2r_v31_enabled and not self.pc2r_v3_enabled:
            raise ValueError("GEOTR_PC2R_CANONICAL_COORD_V31_ENABLED requires GEOTR_PC2R_POSTERIOR_V3_ENABLED=true")
        if self.pc2r_v32_enabled and not self.pc2r_v31_enabled:
            raise ValueError("GEOTR_PC2R_OPERATOR_ALIGNED_V32_ENABLED requires canonical-coordinate PC2R-v3.1")
        if self.c2r_v2_enabled and not self.c2r_enabled:
            raise ValueError("GEOTR_C2R_CANONICAL_ROI_ENABLED requires GEOTR_C2R_ENABLED=true")
        if self.pc2r_v3_enabled and not (self.c2r_enabled and self.c2r_v2_enabled):
            raise ValueError("GEOTR_PC2R_POSTERIOR_V3_ENABLED requires GEOTR_C2R_ENABLED=true and GEOTR_C2R_CANONICAL_ROI_ENABLED=true")
        if self.c2r_enabled and not self.v4g_sparse_direct_refiner:
            raise ValueError("GEOTR_C2R_ENABLED requires GEOTR_V4G_SPARSE_DIRECT_REFINER_ENABLED=true for optimizer/diagnostic ownership")
        if self.sparc_hr_enabled:
            if not self.v4g_sparse_direct_refiner:
                raise ValueError("GEOTR_SPARC_HR_ENABLED requires GEOTR_V4G_SPARSE_DIRECT_REFINER_ENABLED=true")
            if self.c2r_enabled or self.aefr_enabled:
                raise ValueError("SPARC-HR is a complete alternative M2; disable C2R/PC2R/AEFR/SLR flags")
        self.v4f_selective_intervention = bool(_cfg_get(m1, "GEOTR_V4F_SELECTIVE_INTERVENTION_ENABLED", False)) and not self.v4g_sparse_direct_refiner
        self.v4e_operator_consistent = bool(_cfg_get(m1, "GEOTR_V4E_OPERATOR_CONSISTENT_ENABLED", False)) and not self.v4f_selective_intervention and not self.v4g_sparse_direct_refiner
        self.v4d_root_fix = (bool(_cfg_get(m1, "GEOTR_V4D_ROOT_FIX_ENABLED", False)) or self.v4e_operator_consistent) and not self.v4f_selective_intervention and not self.v4g_sparse_direct_refiner

        self.hidden_dim = max(32, int(_cfg_get(m1, "MHCS_HIDDEN_DIM", 128)))
        self.semantic_channels = int(_cfg_get(m1, "SEMANTIC_CHANNELS", 512))
        self.text_dim = int(_cfg_get(m1, "MHCS_TEXT_DIM", 512))

        # Stage-1 Transport evidence encoder.  These historical names preserve
        # the existing mhcs_bank optimizer ownership.
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
        self.mean_head = GeometryHead(
            self.hidden_dim,
            init_scale_px=float(_cfg_get(m1, "GEOTOPO_FLOW_INIT_SCALE_PX", 1.0)),
        )

        # Kept only to preserve historical optimizer/checkpoint owner names.
        self.factor_head = nn.ModuleDict({
            "residual_context": nn.Identity()
        })
        self.diag_std_head = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)
        nn.init.zeros_(self.diag_std_head.weight)
        nn.init.zeros_(self.diag_std_head.bias)

        # All Stage-2 trainable parameters are under m2_surface.* and therefore
        # owned by the existing mhcs_m2 optimizer group.
        self.m2_surface = ErrorLocalizedReconstructionStage(
            self.hidden_dim,
            self.semantic_channels,
            self.text_dim,
            error_prior=float(_cfg_get(m1, "GEOTR_STAGE2_ERROR_PRIOR", 0.03)),
            fn_prior_fraction=float(_cfg_get(m1, "GEOTR_STAGE2_FN_PRIOR_FRACTION", 0.50)),
            magnitude_init=float(_cfg_get(m1, "GEOTR_STAGE2_MAGNITUDE_INIT", 0.02)),
            v4d_enabled=self.v4d_root_fix and not self.v4e_operator_consistent,
            v4e_enabled=self.v4e_operator_consistent,
            v4f_enabled=self.v4f_selective_intervention,
            severity_prior=float(_cfg_get(m1, "GEOTR_STAGE2_SEVERITY_PRIOR", 0.03)),
            logit_magnitude_init=float(_cfg_get(m1, "GEOTR_STAGE2_LOGIT_MAGNITUDE_INIT", 0.25)),
            max_logit_step=float(_cfg_get(m1, "GEOTR_STAGE2_MAX_LOGIT_STEP", 4.0)),
            severity_grad_scale=float(_cfg_get(m1, "GEOTR_STAGE2_DEPLOY_SEVERITY_GRAD_SCALE", 0.25)),
            v4f_proposal_coverage=float(_cfg_get(m1, "GEOTR_V4F_PROPOSAL_COVERAGE", 0.10)),
            v4f_execution_prior=float(_cfg_get(m1, "GEOTR_V4F_EXECUTION_PRIOR", 0.20)),
            v4f_execution_threshold=float(_cfg_get(m1, "GEOTR_V4F_EXECUTION_THRESHOLD", 0.65)),
            v4f_dose_init=float(_cfg_get(m1, "GEOTR_V4F_DOSE_INIT", 0.25)),
            v4f_flow_scale_px=float(_cfg_get(m1, "GEOTR_V4F_FLOW_SCALE_PX", 8.0)),
            v4f_deploy_dose_grad_scale=float(_cfg_get(m1, "GEOTR_V4F_DEPLOY_DOSE_GRAD_SCALE", 0.0)),
        )
        self.v4g_r2_correction_preserve = bool(
            _cfg_get(m1, "GEOTR_V4G_R2_CORRECTION_PRESERVE_ENABLED", False)
        ) and self.v4g_sparse_direct_refiner
        self.v4g_r3_minimal_intervention = bool(
            _cfg_get(m1, "GEOTR_V4G_R3_MINIMAL_INTERVENTION_ENABLED", False)
        ) and self.v4g_r2_correction_preserve
        self.v4g_r4_exogenous_patch_flip = bool(
            _cfg_get(m1, "GEOTR_V4G_R4_EXOGENOUS_PATCH_FLIP_ENABLED", False)
        ) and self.v4g_r2_correction_preserve
        self.v4g_r41_selection_consistent = bool(
            _cfg_get(m1, "GEOTR_V4G_R41_SELECTION_CONSISTENT_ENABLED", False)
        ) and self.v4g_r4_exogenous_patch_flip
        if self.v4g_sparse_direct_refiner:
            if self.sparc_hr_enabled:
                self.v4g_refiner = StructuredPosteriorAtomicHRComposer(
                    self.hidden_dim,
                    self.semantic_channels,
                    self.text_dim,
                    fine_feature_channels=int(_cfg_get(m1, "GEOTR_C2R_FINE_FEATURE_CHANNELS", self.semantic_channels)),
                    hidden_dim=int(_cfg_get(m1, "GEOTR_SPARC_HIDDEN_DIM", 48)),
                    num_regions=int(_cfg_get(m1, "GEOTR_SPARC_NUM_REGIONS", 4)),
                    region_size_lr=int(_cfg_get(m1, "GEOTR_SPARC_REGION_SIZE_LR", 49)),
                    min_center_distance_lr=int(_cfg_get(m1, "GEOTR_SPARC_MIN_CENTER_DISTANCE_LR", 32)),
                    max_steps=int(_cfg_get(m1, "GEOTR_SPARC_MAX_STEPS", 3)),
                    max_posterior_sources=int(_cfg_get(m1, "GEOTR_SPARC_MAX_POSTERIOR_SOURCES", 0)),
                    max_proposal_logit_step=float(_cfg_get(m1, "GEOTR_SPARC_MAX_PROPOSAL_LOGIT_STEP", 4.0)),
                    stop_margin=float(_cfg_get(m1, "GEOTR_SPARC_STOP_MARGIN", 2.0e-4)),
                    require_true_hr=bool(_cfg_get(m1, "GEOTR_SPARC_REQUIRE_TRUE_HR", True)),
                    minimum_crossing_margin=float(_cfg_get(m1, "GEOTR_SPARC_MINIMUM_CROSSING_MARGIN", 0.05)),
                    critic_grid_size=int(_cfg_get(m1, "GEOTR_SPARC_CRITIC_GRID_SIZE", 112)),
                    utility_gain_scale=float(_cfg_get(m1, "GEOTR_SPARC_UTILITY_GAIN_SCALE", 0.02)),
                    utility_class_score_scale=float(_cfg_get(m1, "GEOTR_SPARC_UTILITY_CLASS_SCORE_SCALE", 0.0)),
                    max_region_overlap=float(_cfg_get(m1, "GEOTR_SPARC_MAX_REGION_OVERLAP", 0.10)),
                    typed_gate_init_bias=float(_cfg_get(m1, "GEOTR_SPARC_TYPED_GATE_INIT_BIAS", -2.0)),
                    selector_evidence_weight=float(_cfg_get(m1, "GEOTR_SPARC_SELECTOR_EVIDENCE_WEIGHT", 0.50)),
                    neutral_utility_margin=float(_cfg_get(m1, "GEOTR_SPARC_NEUTRAL_UTILITY_MARGIN", 2.0e-4)),
                    utility_std_floor=float(_cfg_get(m1, "GEOTR_SPARC_UTILITY_STD_FLOOR", 2.0e-4)),
                    risk_lcb_z=float(_cfg_get(m1, "GEOTR_SPARC_RISK_LCB_Z", 1.0)),
                    utility_quantile_tau=float(_cfg_get(m1, "GEOTR_SPARC_UTILITY_QUANTILE_TAU", 0.10)),
                    benefit_probability_threshold=float(_cfg_get(m1, "GEOTR_SPARC_BENEFIT_PROB_THRESHOLD", 0.50)),
                    policy_margin=float(_cfg_get(m1, "GEOTR_SPARC_POLICY_MARGIN", 0.0)),
                    teacher_surface_tolerance_hr_px=int(_cfg_get(m1, "GEOTR_SPARC_SURFACE_TOLERANCE_HR_PX", 4)),
                    critic_patch_size=int(_cfg_get(m1, "GEOTR_SPARC_CRITIC_PATCH_SIZE", 12)),
                    source_embedding_dim=int(_cfg_get(m1, "GEOTR_SPARC_SOURCE_EMBED_DIM", 12)),
                    policy_set_layers=int(_cfg_get(m1, "GEOTR_SPARC_POLICY_SET_LAYERS", 2)),
                    policy_set_heads=int(_cfg_get(m1, "GEOTR_SPARC_POLICY_SET_HEADS", 4)),
                    min_action_change_hr_pixels=int(_cfg_get(m1, "GEOTR_SPARC_MIN_ACTION_CHANGE_HR_PIXELS", 4)),
                    prune_duplicate_actions=bool(_cfg_get(m1, "GEOTR_SPARC_PRUNE_DUPLICATE_ACTIONS", True)),
                    selector_candidate_evidence_weight=float(_cfg_get(m1, "GEOTR_SPARC_SELECTOR_CANDIDATE_EVIDENCE_WEIGHT", 0.50)),
                    router_grid_size=int(_cfg_get(m1, "GEOTR_SPARC_ROUTER_GRID_SIZE", 112)),
                    router_temperature=float(_cfg_get(m1, "GEOTR_SPARC_ROUTER_TEMPERATURE", 0.50)),
                    router_anchor_prior=float(_cfg_get(m1, "GEOTR_SPARC_ROUTER_ANCHOR_PRIOR", 1.0)),
                    router_family_embedding_dim=int(_cfg_get(m1, "GEOTR_SPARC_ROUTER_FAMILY_EMBED_DIM", 16)),
                    boundary_band_radius_hr=int(_cfg_get(m1, "GEOTR_SPARC_BOUNDARY_BAND_RADIUS_HR", 12)),
                    router_edit_logit_margin=float(_cfg_get(m1, "GEOTR_SPARC_ROUTER_EDIT_LOGIT_MARGIN", 1.5)),
                    router_edit_probability_threshold=float(_cfg_get(m1, "GEOTR_SPARC_ROUTER_EDIT_PROBABILITY_THRESHOLD", 0.70)),
                )
            elif self.c2r_enabled:
                if self.v4g_r3_minimal_intervention or self.v4g_r4_exogenous_patch_flip or self.v4g_r41_selection_consistent:
                    raise ValueError("C2R is an alternative Stage-2 contract; disable R3/R4/R4.1 when GEOTR_C2R_ENABLED=true")
                if self.c2r_v2_enabled:
                    if self.aefr_enabled:
                        if self.aefr_stage == "sparse_local_rerendering":
                            _slr_common = dict(
                                fine_feature_channels=int(_cfg_get(m1, "GEOTR_C2R_FINE_FEATURE_CHANNELS", self.semantic_channels)),
                                num_regions=int(_cfg_get(m1, "GEOTR_SLR_NUM_REGIONS", _cfg_get(m1, "GEOTR_C2R_NUM_REGIONS", 8))),
                                region_size=int(_cfg_get(m1, "GEOTR_SLR_REGION_SIZE", _cfg_get(m1, "GEOTR_C2R_REGION_SIZE", 33))),
                                flow_scale_px=float(_cfg_get(m1, "GEOTR_C2R_FLOW_SCALE_PX", 8.0)),
                                min_center_distance=int(_cfg_get(m1, "GEOTR_SLR_MIN_CENTER_DISTANCE", _cfg_get(m1, "GEOTR_C2R_MIN_CENTER_DISTANCE", _cfg_get(m1, "GEOTR_C2R_REGION_SIZE", 33)))),
                                residual_logit_scale=float(_cfg_get(m1, "GEOTR_SLR_RESIDUAL_LOGIT_SCALE", 4.0)),
                                sdf_radius_px=int(_cfg_get(m1, "GEOTR_SLR_SDF_RADIUS_PX", 8)),
                                train_oracle_regions=int(_cfg_get(m1, "GEOTR_SLR_TRAIN_ORACLE_REGIONS", _cfg_get(m1, "GEOTR_SLR_TRAIN_POSITIVE_REGIONS", 4))),
                                train_positive_regions=int(_cfg_get(m1, "GEOTR_SLR_TRAIN_POSITIVE_REGIONS", _cfg_get(m1, "GEOTR_SLR_TRAIN_ORACLE_REGIONS", 4))),
                                train_clean_regions=int(_cfg_get(m1, "GEOTR_SLR_TRAIN_CLEAN_REGIONS", 4)),
                                clean_max_error_fraction=float(_cfg_get(m1, "GEOTR_SLR_CLEAN_MAX_ERROR_FRACTION", 0.02)),
                                clean_max_sdf_discrepancy=float(_cfg_get(m1, "GEOTR_SLR_CLEAN_MAX_SDF_DISCREPANCY", 0.05)),
                                blend_taper_px=int(_cfg_get(m1, "GEOTR_SLR_BLEND_TAPER_PX", 4)),
                                context_size=int(_cfg_get(m1, "GEOTR_SLR_CONTEXT_SIZE", 55)),
                                hr_size=int(_cfg_get(m1, "GEOTR_SLR_HR_SIZE", 448)),
                                heaviside_tau_px=float(_cfg_get(m1, "GEOTR_SLR_HEAVISIDE_TAU_PX", 1.0)),
                                require_true_hr=bool(_cfg_get(m1, "GEOTR_SLR_REQUIRE_TRUE_HR", True)),
                                action_margin_prob=float(_cfg_get(m1, "GEOTR_SLR_ACTION_MARGIN_PROB", 0.05)),
                            )
                            if bool(_cfg_get(m1, "GEOTR_SLR_UCDRT_R2_ENABLED", False)):
                                self.v4g_refiner = OperatorConsistentSelectiveTransitionRefiner(
                                    self.hidden_dim,
                                    self.semantic_channels,
                                    self.text_dim,
                                    **_slr_common,
                                    boundary_radius_px=int(_cfg_get(m1, "GEOTR_SLR_UCDRT_BOUNDARY_RADIUS_PX", 5)),
                                    max_boundary_displacement_px=float(_cfg_get(m1, "GEOTR_SLR_UCDRT_MAX_BOUNDARY_DISPLACEMENT_PX", 5.0)),
                                    max_interior_logit_step=1.0,  # retired in R2; compatibility only
                                    utility_temperature=float(_cfg_get(m1, "GEOTR_SLR_UCDRT_UTILITY_TEMPERATURE", 0.25)),
                                    state_keep_bias=0.0,  # retired by factorized EDIT prior
                                    paired_stable_enabled=bool(_cfg_get(m1, "GEOTR_SLR_UCDRT_PAIRED_STABLE_ENABLED", True)),
                                    paired_regions=int(_cfg_get(m1, "GEOTR_SLR_UCDRT_PAIRED_REGIONS", 4)),
                                    move_bins=int(_cfg_get(m1, "GEOTR_SLR_UCDRT_R2_MOVE_BINS", 21)),
                                    edit_prior=float(_cfg_get(m1, "GEOTR_SLR_UCDRT_R2_EDIT_PRIOR", 0.15)),
                                    dose_init_fraction=float(_cfg_get(m1, "GEOTR_SLR_UCDRT_R2_DOSE_INIT_FRACTION", 0.10)),
                                    utility_gain_scale=float(_cfg_get(m1, "GEOTR_SLR_UCDRT_R2_UTILITY_GAIN_SCALE", 20.0)),
                                    utility_boundary_weight=float(_cfg_get(m1, "GEOTR_SLR_UCDRT_R2_UTILITY_BOUNDARY_WEIGHT", 0.25)),
                                )
                            elif bool(_cfg_get(m1, "GEOTR_SLR_UCDRT_ENABLED", False)):
                                self.v4g_refiner = UtilityConsistentDualSpaceResidualTransitionRefiner(
                                    self.hidden_dim,
                                    self.semantic_channels,
                                    self.text_dim,
                                    **_slr_common,
                                    boundary_radius_px=int(_cfg_get(m1, "GEOTR_SLR_UCDRT_BOUNDARY_RADIUS_PX", 5)),
                                    max_boundary_displacement_px=float(_cfg_get(m1, "GEOTR_SLR_UCDRT_MAX_BOUNDARY_DISPLACEMENT_PX", 4.0)),
                                    max_interior_logit_step=float(_cfg_get(m1, "GEOTR_SLR_UCDRT_MAX_INTERIOR_LOGIT_STEP", 4.0)),
                                    utility_temperature=float(_cfg_get(m1, "GEOTR_SLR_UCDRT_UTILITY_TEMPERATURE", 0.25)),
                                    state_keep_bias=float(_cfg_get(m1, "GEOTR_SLR_UCDRT_KEEP_BIAS", 2.0)),
                                    interior_magnitude_init_fraction=float(_cfg_get(m1, "GEOTR_SLR_UCDRT_INTERIOR_MAG_INIT_FRACTION", 0.10)),
                                    paired_stable_enabled=bool(_cfg_get(m1, "GEOTR_SLR_UCDRT_PAIRED_STABLE_ENABLED", True)),
                                    paired_regions=int(_cfg_get(m1, "GEOTR_SLR_UCDRT_PAIRED_REGIONS", 4)),
                                )
                            else:
                                self.v4g_refiner = GeometryConditionedSparseLocalRerenderer(
                                    self.hidden_dim,
                                    self.semantic_channels,
                                    self.text_dim,
                                    **_slr_common,
                                )
                        else:
                            self.v4g_refiner = ActionEvidenceFactorizedROIRefiner(
                            self.hidden_dim,
                            self.semantic_channels,
                            self.text_dim,
                            fine_feature_channels=int(_cfg_get(m1, "GEOTR_C2R_FINE_FEATURE_CHANNELS", self.semantic_channels)),
                            num_regions=int(_cfg_get(m1, "GEOTR_C2R_NUM_REGIONS", 4)),
                            region_size=int(_cfg_get(m1, "GEOTR_C2R_REGION_SIZE", 33)),
                            selection_score=str(_cfg_get(m1, "GEOTR_C2R_SELECTION_SCORE", "margin")),
                            flow_scale_px=float(_cfg_get(m1, "GEOTR_C2R_FLOW_SCALE_PX", 8.0)),
                            min_center_distance=int(_cfg_get(m1, "GEOTR_C2R_MIN_CENTER_DISTANCE", _cfg_get(m1, "GEOTR_C2R_REGION_SIZE", 33))),
                            component_min_area=int(_cfg_get(m1, "GEOTR_C2R_COMPONENT_MIN_AREA", 2)),
                            residual_logit_scale=float(_cfg_get(m1, "GEOTR_AEFR_SINGLE_RESIDUAL_LOGIT_SCALE", _cfg_get(m1, "GEOTR_PC2R_RESIDUAL_LOGIT_SCALE", 2.0))),
                            risk_top_fraction=float(_cfg_get(m1, "GEOTR_PC2R_RISK_TOP_FRACTION", 0.20)),
                            require_risk_overlap=bool(_cfg_get(m1, "GEOTR_PC2R_REQUIRE_RISK_OVERLAP", True)),
                            raw_correction_threshold=float(_cfg_get(m1, "GEOTR_PC2R_RAW_CORRECTION_THRESHOLD", 0.02)),
                            stage=self.aefr_stage,
                            joint_geometry_grad=self.aefr_joint_geometry_grad,
                            posterior_stability=bool(_cfg_get(m1, "GEOTR_AEFR_POSTERIOR_STABILITY_ENABLED", True)),
                            boundary_radius_px=int(_cfg_get(m1, "GEOTR_AEFR_BOUNDARY_RADIUS_PX", 5)),
                            boundary_max_displacement_px=float(_cfg_get(m1, "GEOTR_AEFR_BOUNDARY_MAX_DISPLACEMENT_PX", 4.0)),
                            interior_crossing_margin=float(_cfg_get(m1, "GEOTR_AEFR_INTERIOR_CROSSING_MARGIN", 0.5)),
                            transition_aware=self.aefr_transition_aware,
                            transition_logit_scale=float(_cfg_get(m1, "GEOTR_AEFR_TRANSITION_LOGIT_SCALE", 2.0)),
                            use_raw_flow_evidence=self.aefr_use_raw_flow_evidence,
                            intervention_error_prior=float(_cfg_get(m1, "GEOTR_AEFR_INTERVENTION_ERROR_PRIOR", 0.03)),
                            intervention_edit_prior=float(_cfg_get(m1, "GEOTR_AEFR_INTERVENTION_EDIT_PRIOR", 0.03)),
                            smi_commit_threshold=float(_cfg_get(m1, "GEOTR_AEFR_SMI_COMMIT_THRESHOLD", 0.50)),
                            smi_direction_confidence_threshold=float(_cfg_get(m1, "GEOTR_AEFR_SMI_DIRECTION_CONFIDENCE_THRESHOLD", 0.50)),
                        )
                    elif self.pc2r_v3_enabled:
                        pc2r_cls = CanonicalCoordinatePosteriorROIRefiner if self.pc2r_v31_enabled else PosteriorConsistentCanonicalROIRefiner
                        self.v4g_refiner = pc2r_cls(
                            self.hidden_dim,
                            self.semantic_channels,
                            self.text_dim,
                            fine_feature_channels=int(_cfg_get(m1, "GEOTR_C2R_FINE_FEATURE_CHANNELS", self.semantic_channels)),
                            num_regions=int(_cfg_get(m1, "GEOTR_C2R_NUM_REGIONS", 4)),
                            region_size=int(_cfg_get(m1, "GEOTR_C2R_REGION_SIZE", 33)),
                            selection_score=str(_cfg_get(m1, "GEOTR_C2R_SELECTION_SCORE", "margin")),
                            flow_scale_px=float(_cfg_get(m1, "GEOTR_C2R_FLOW_SCALE_PX", 8.0)),
                            min_center_distance=int(_cfg_get(m1, "GEOTR_C2R_MIN_CENTER_DISTANCE", _cfg_get(m1, "GEOTR_C2R_REGION_SIZE", 33))),
                            component_min_area=int(_cfg_get(m1, "GEOTR_C2R_COMPONENT_MIN_AREA", 2)),
                            residual_logit_scale=float(_cfg_get(m1, "GEOTR_PC2R_RESIDUAL_LOGIT_SCALE", 2.0)),
                            num_posterior_views=int(_cfg_get(m1, "GEOTR_PC2R_NUM_POSTERIOR_VIEWS", 3)),
                            risk_top_fraction=float(_cfg_get(m1, "GEOTR_PC2R_RISK_TOP_FRACTION", 0.20)),
                            require_risk_overlap=bool(_cfg_get(m1, "GEOTR_PC2R_REQUIRE_RISK_OVERLAP", True)),
                            direction_agreement_threshold=float(_cfg_get(m1, "GEOTR_PC2R_DIRECTION_AGREEMENT_THRESHOLD", 0.80)),
                            residual_spread_threshold=float(_cfg_get(m1, "GEOTR_PC2R_RESIDUAL_SPREAD_THRESHOLD", 1.0)),
                            residual_strength_threshold=float(_cfg_get(m1, "GEOTR_PC2R_RESIDUAL_STRENGTH_THRESHOLD", 0.05)),
                            raw_correction_threshold=float(_cfg_get(m1, "GEOTR_PC2R_RAW_CORRECTION_THRESHOLD", 0.02)),
                            reliance_diagnostics=bool(_cfg_get(m1, "GEOTR_PC2R_RELIANCE_DIAGNOSTICS", True)),
                        )
                    else:
                        self.v4g_refiner = CanonicalCounterfactualROIRefiner(
                            self.hidden_dim,
                            self.semantic_channels,
                            self.text_dim,
                            fine_feature_channels=int(_cfg_get(m1, "GEOTR_C2R_FINE_FEATURE_CHANNELS", self.semantic_channels)),
                            num_regions=int(_cfg_get(m1, "GEOTR_C2R_NUM_REGIONS", 4)),
                            region_size=int(_cfg_get(m1, "GEOTR_C2R_REGION_SIZE", 33)),
                            counterfactual_radius=int(_cfg_get(m1, "GEOTR_C2R_COUNTERFACTUAL_RADIUS", 1)),
                            selection_score=str(_cfg_get(m1, "GEOTR_C2R_SELECTION_SCORE", "margin")),
                            flow_scale_px=float(_cfg_get(m1, "GEOTR_C2R_FLOW_SCALE_PX", 8.0)),
                            min_center_distance=int(_cfg_get(m1, "GEOTR_C2R_MIN_CENTER_DISTANCE", _cfg_get(m1, "GEOTR_C2R_REGION_SIZE", 33))),
                            seed_radius=int(_cfg_get(m1, "GEOTR_C2R_SEED_RADIUS", 2)),
                            component_agreement_threshold=float(_cfg_get(m1, "GEOTR_C2R_COMPONENT_AGREEMENT_THRESHOLD", 0.90)),
                            component_spread_threshold=float(_cfg_get(m1, "GEOTR_C2R_COMPONENT_SPREAD_THRESHOLD", 0.12)),
                            component_confidence_threshold=float(_cfg_get(m1, "GEOTR_C2R_COMPONENT_CONFIDENCE_THRESHOLD", 0.10)),
                            component_min_area=int(_cfg_get(m1, "GEOTR_C2R_COMPONENT_MIN_AREA", 2)),
                            canonical_residual_scale=float(_cfg_get(m1, "GEOTR_C2R_CANONICAL_RESIDUAL_SCALE", 1.0)),
                        )
                else:
                    self.v4g_refiner = CounterfactualConsensusRegionRefiner(
                        self.hidden_dim,
                        self.semantic_channels,
                        self.text_dim,
                        fine_feature_channels=int(_cfg_get(m1, "GEOTR_C2R_FINE_FEATURE_CHANNELS", self.semantic_channels)),
                        num_regions=int(_cfg_get(m1, "GEOTR_C2R_NUM_REGIONS", 4)),
                        region_size=int(_cfg_get(m1, "GEOTR_C2R_REGION_SIZE", 33)),
                        counterfactual_radius=int(_cfg_get(m1, "GEOTR_C2R_COUNTERFACTUAL_RADIUS", 1)),
                        selection_score=str(_cfg_get(m1, "GEOTR_C2R_SELECTION_SCORE", "margin")),
                        flow_scale_px=float(_cfg_get(m1, "GEOTR_C2R_FLOW_SCALE_PX", 8.0)),
                    )
            else:
                self.v4g_refiner = SparseDirectResidualRefiner(
                    self.hidden_dim,
                    self.semantic_channels,
                    self.text_dim,
                    coverage=float(_cfg_get(m1, "GEOTR_V4G_REFINEMENT_COVERAGE", 0.10)),
                    train_importance_ratio=float(_cfg_get(m1, "GEOTR_V4G_TRAIN_IMPORTANCE_RATIO", 0.75)),
                    selection_score=str(_cfg_get(m1, "GEOTR_V4G_SELECTION_SCORE", "margin")),
                    flow_scale_px=float(_cfg_get(m1, "GEOTR_V4G_FLOW_SCALE_PX", 8.0)),
                    denoise_enabled=bool(_cfg_get(m1, "GEOTR_V4G_DENOISE_ENABLED", True)),
                    r2_enabled=self.v4g_r2_correction_preserve,
                    fine_feature_channels=int(_cfg_get(m1, "GEOTR_V4G_R2_FINE_FEATURE_CHANNELS", self.semantic_channels)),
                    match_inference_selection=bool(_cfg_get(m1, "GEOTR_V4G_R2_MATCH_INFERENCE_SELECTION", True)),
                    denoise_coverage=float(_cfg_get(m1, "GEOTR_V4G_R2_DENOISE_COVERAGE", 0.03)),
                    denoise_radius_min=int(_cfg_get(m1, "GEOTR_V4G_R2_DENOISE_RADIUS_MIN", 1)),
                    denoise_radius_max=int(_cfg_get(m1, "GEOTR_V4G_R2_DENOISE_RADIUS_MAX", 3)),
                    r3_enabled=self.v4g_r3_minimal_intervention,
                    r3_max_abs_delta_logit=float(_cfg_get(m1, "GEOTR_V4G_R3_MAX_ABS_DELTA_LOGIT", 1.0)),
                    r4_enabled=self.v4g_r4_exogenous_patch_flip,
                    r4_patch_size=int(_cfg_get(m1, "GEOTR_V4G_R4_PATCH_SIZE", 5)),
                    r4_synth_edit_fraction=float(_cfg_get(m1, "GEOTR_V4G_R4_SYNTH_EDIT_FRACTION", 0.20)),
                    r4_flip_threshold=float(_cfg_get(m1, "GEOTR_V4G_R4_FLIP_THRESHOLD", 0.50)),
                    r4_target_prob_margin=float(_cfg_get(m1, "GEOTR_V4G_R4_TARGET_PROB_MARGIN", 0.05)),
                    r4_corrupt_logit_min=float(_cfg_get(m1, "GEOTR_V4G_R4_CORRUPT_LOGIT_MIN", 0.05)),
                    r4_corrupt_logit_max=float(_cfg_get(m1, "GEOTR_V4G_R4_CORRUPT_LOGIT_MAX", 2.50)),
                    r41_enabled=self.v4g_r41_selection_consistent,
                    r41_patch_reduce_channels=int(_cfg_get(m1, "GEOTR_V4G_R41_PATCH_REDUCE_CHANNELS", 32)),
                    r41_corrupt_logit_max=float(_cfg_get(m1, "GEOTR_V4G_R41_CORRUPT_LOGIT_MAX", 0.80)),
                )
        else:
            self.v4g_refiner = None

        self.m1_distribution_log_var = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _resize(x: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] == hw:
            return x
        return F.interpolate(x, size=hw, mode="bilinear", align_corners=False)

    @staticmethod
    def _boundary(prob: torch.Tensor) -> torch.Tensor:
        maxp = F.max_pool2d(prob, 3, stride=1, padding=1)
        minp = -F.max_pool2d(-prob, 3, stride=1, padding=1)
        return (maxp - minp).clamp(0.0, 1.0)

    @staticmethod
    def _uncertainty(prob: torch.Tensor) -> torch.Tensor:
        return (4.0 * prob * (1.0 - prob)).clamp(0.0, 1.0)

    @staticmethod
    def _warp_logits(base_logits: torch.Tensor, flow_px: torch.Tensor) -> torch.Tensor:
        _, _, h, w = base_logits.shape
        dtype, device = base_logits.dtype, base_logits.device
        yy = torch.arange(h, device=device, dtype=dtype) + 0.5
        xx = torch.arange(w, device=device, dtype=dtype) + 0.5
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        x = gx[None] + flow_px[:, 0]
        y = gy[None] + flow_px[:, 1]
        x_norm = 2.0 * x / float(w) - 1.0
        y_norm = 2.0 * y / float(h) - 1.0
        grid = torch.stack([x_norm, y_norm], dim=-1)
        return F.grid_sample(
            base_logits,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

    @classmethod
    def _transport_mc_posterior(
        cls,
        probability_samples: Optional[torch.Tensor],
        flow_px: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Transport every MC posterior with the same deployed mean flow.

        Returns warped posterior samples plus recomputed std/disagreement.  This
        exposes the on-policy hypotheses needed by PC2R-v3 instead of discarding
        them after uncertainty calculation.
        """
        if not isinstance(probability_samples, torch.Tensor):
            return None, None, None
        samples = probability_samples.detach()
        if samples.ndim == 4:
            samples = samples.unsqueeze(2)
        if samples.ndim != 5 or samples.shape[2] != 1:
            raise ValueError(
                "mc_probability_samples must be [S,B,H,W] or [S,B,1,H,W], got "
                f"{tuple(samples.shape)}"
            )
        s, b, _, h, w = samples.shape
        if tuple(flow_px.shape[-2:]) != (h, w):
            flow = F.interpolate(flow_px.detach(), size=(h, w), mode="bilinear", align_corners=False)
        else:
            flow = flow_px.detach()
        logits = torch.logit(samples.clamp(EPS, 1.0 - EPS)).reshape(s * b, 1, h, w)
        flow_rep = flow[None].expand(s, -1, -1, -1, -1).reshape(s * b, 2, h, w)
        warped_logits = cls._warp_logits(logits, flow_rep)
        warped_prob = torch.sigmoid(warped_logits).reshape(s, b, 1, h, w).clamp(EPS, 1.0 - EPS)
        if s <= 1:
            z = torch.zeros_like(warped_prob[0])
            return warped_prob, z, z
        std_map = warped_prob.std(dim=0, unbiased=False)
        hard = warped_prob >= 0.5
        xor_maps = []
        for i in range(s):
            for j in range(i + 1, s):
                xor_maps.append(hard[i].ne(hard[j]).to(warped_prob.dtype))
        disagreement = torch.stack(xor_maps, dim=0).mean(dim=0)
        return warped_prob, std_map, disagreement

    @classmethod
    def _transport_mc_uncertainty(
        cls,
        probability_samples: Optional[torch.Tensor],
        flow_px: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        _, std_map, disagreement = cls._transport_mc_posterior(probability_samples, flow_px)
        return std_map, disagreement

    @staticmethod
    def _flow_jacobian_stats(flow_px: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if flow_px.shape[-2] < 2 or flow_px.shape[-1] < 2:
            b = flow_px.shape[0]
            return flow_px.new_ones(b), flow_px.new_zeros(b)
        ux = flow_px[:, 0]
        uy = flow_px[:, 1]
        dux_dx = ux[:, :-1, 1:] - ux[:, :-1, :-1]
        dux_dy = ux[:, 1:, :-1] - ux[:, :-1, :-1]
        duy_dx = uy[:, :-1, 1:] - uy[:, :-1, :-1]
        duy_dy = uy[:, 1:, :-1] - uy[:, :-1, :-1]
        det = (1.0 + dux_dx) * (1.0 + duy_dy) - dux_dy * duy_dx
        return det.mean(dim=(1, 2)), (det <= 0.0).float().mean(dim=(1, 2))

    def _transport_features(
        self,
        image: torch.Tensor,
        semantic_map: torch.Tensor,
        text_features: torch.Tensor,
        anchor_prob: torch.Tensor,
    ) -> torch.Tensor:
        hw = tuple(anchor_prob.shape[-2:])
        image_latent = self.image_stem(self._resize(image, hw))
        semantic_latent = self.semantic_proj(self._resize(semantic_map, hw))
        text = self.text_proj(text_features.float())
        p = anchor_prob.detach()
        uncertainty = self._uncertainty(p)
        boundary = self._boundary(p)
        pixel = self.pixel_fuse(
            torch.cat([image_latent, semantic_latent, p, uncertainty, boundary], dim=1)
        )
        global_visual = F.adaptive_avg_pool2d(pixel, 1).flatten(1)
        context = self.global_context(torch.cat([global_visual, text], dim=1))
        gamma, beta = self.context_film(context).chunk(2, dim=1)
        gate = torch.tanh(self.context_gate)
        pixel = pixel + gate * (
            torch.tanh(gamma)[:, :, None, None] * pixel + beta[:, :, None, None]
        )
        hidden = self.distribution_trunk(pixel)
        return hidden + self.diag_std_head(hidden)

    def generate(
        self,
        base_logits: torch.Tensor,
        image: torch.Tensor,
        semantic_map: Optional[torch.Tensor],
        text_features: torch.Tensor,
        negative_text_features: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del negative_text_features
        mc_std_map = kwargs.get("mc_std_map", None)
        mc_disagreement_map = kwargs.get("mc_disagreement_map", None)
        mc_probability_samples = kwargs.get("mc_probability_samples", None)
        fine_feature_map = kwargs.get("fine_feature_map", None)
        supervision_masks = kwargs.get("supervision_masks", None)
        slr_hr_image = kwargs.get("slr_hr_image", None)
        sparc_hr_mask = kwargs.get("sparc_hr_mask", None)
        if semantic_map is None:
            raise RuntimeError("GEOTR-V4C requires UniMedCLIP spatial semantic_map")
        if base_logits.ndim == 3:
            base_logits = base_logits[:, None]
        if base_logits.ndim != 4 or base_logits.shape[1] != 1:
            raise ValueError(f"Expected Base logits [B,1,H,W], got {tuple(base_logits.shape)}")

        factual_base_logits = base_logits.detach()
        base_prob = torch.sigmoid(factual_base_logits).clamp(EPS, 1.0 - EPS)

        # Stage 1: unchanged Transport operator.
        transport_hidden = self._transport_features(
            image, semantic_map, text_features, base_prob
        )
        flow_px = self.mean_head(transport_hidden)
        geo_logits = self._warp_logits(factual_base_logits, flow_px)
        geo_prob = torch.sigmoid(geo_logits).clamp(EPS, 1.0 - EPS)

        # Stage 2 owns independent evidence parameters.  Base and Transport
        # anchors are detached inside m2_surface.
        zero_flow = torch.zeros_like(flow_px)
        # Historical Stage-2 is retained only for old-version diagnostics when
        # V4G is active.  It is frozen and evaluated without autograd; the
        # deployable V4G prediction comes exclusively from the direct refiner.
        old_ctx = torch.no_grad() if self.v4g_sparse_direct_refiner else torch.enable_grad()
        with old_ctx:
            recon_base = self.m2_surface(
                base_prob, image, semantic_map, text_features,
                base_prob=base_prob, flow_px=zero_flow,
            )
            recon_geo = self.m2_surface(
                geo_prob, image, semantic_map, text_features,
                base_prob=base_prob, flow_px=flow_px,
            )

        if self.v4g_sparse_direct_refiner:
            assert self.v4g_refiner is not None
            geo_mc_std_map = mc_std_map
            geo_mc_disagreement_map = mc_disagreement_map
            geo_mc_probability_samples = None
            if (self.v4g_r2_correction_preserve or self.c2r_enabled or self.aefr_enabled or self.sparc_hr_enabled) and isinstance(mc_probability_samples, torch.Tensor):
                geo_mc_probability_samples, geo_mc_std_map, geo_mc_disagreement_map = self._transport_mc_posterior(
                    mc_probability_samples, flow_px
                )
            direct_base = direct_geo = None
            if self.sparc_hr_enabled:
                # One GT-free deployment operator.  The optional HR target is
                # only exported for the loss and is never read by the composer.
                run_base = self.mode == "residual"
                run_geo = self.mode == "full"
                if run_base:
                    direct_base = self.v4g_refiner(
                        base_prob, image, semantic_map, text_features,
                        base_prob=base_prob, flow_px=zero_flow,
                        mc_std_map=mc_std_map, mc_disagreement_map=mc_disagreement_map,
                        fine_feature_map=fine_feature_map,
                        posterior_probability_samples=mc_probability_samples,
                        hr_image=slr_hr_image, hr_target=sparc_hr_mask,
                    )
                if run_geo:
                    direct_geo = self.v4g_refiner(
                        geo_prob, image, semantic_map, text_features,
                        base_prob=base_prob, flow_px=flow_px,
                        mc_std_map=geo_mc_std_map, mc_disagreement_map=geo_mc_disagreement_map,
                        fine_feature_map=fine_feature_map,
                        posterior_probability_samples=geo_mc_probability_samples,
                        hr_image=slr_hr_image, hr_target=sparc_hr_mask,
                    )
            elif self.c2r_enabled:
                # C2R removes the historical double-forward training confound:
                # A2 trains only Base->C2R and A3 trains only Transport->C2R.
                # In eval mode both branches may be computed under the caller's
                # no_grad context for paired diagnostics; C2R has no stochastic
                # Stage-2 state, so this cannot perturb either trajectory.
                run_base = (self.mode == "residual") or ((not self.training) and self.mode in {"residual", "full"})
                run_geo = (self.mode == "full") or ((not self.training) and self.mode in {"residual", "full"})
                if run_base:
                    direct_base = self.v4g_refiner(
                        base_prob, image, semantic_map, text_features,
                        base_prob=base_prob, flow_px=zero_flow,
                        mc_std_map=mc_std_map, mc_disagreement_map=mc_disagreement_map,
                        fine_feature_map=fine_feature_map,
                        posterior_probability_samples=(mc_probability_samples if (self.pc2r_v3_enabled or self.aefr_enabled) else None),
                        supervision_masks=(supervision_masks if self.aefr_stage == "sparse_local_rerendering" else None),
                        **({"hr_image": slr_hr_image} if self.aefr_stage == "sparse_local_rerendering" else {}),
                    )
                if run_geo:
                    direct_geo = self.v4g_refiner(
                        geo_prob, image, semantic_map, text_features,
                        base_prob=base_prob, flow_px=flow_px,
                        mc_std_map=geo_mc_std_map, mc_disagreement_map=geo_mc_disagreement_map,
                        fine_feature_map=fine_feature_map,
                        posterior_probability_samples=(geo_mc_probability_samples if (self.pc2r_v3_enabled or self.aefr_enabled) else None),
                        supervision_masks=(supervision_masks if self.aefr_stage == "sparse_local_rerendering" else None),
                        **({"hr_image": slr_hr_image} if self.aefr_stage == "sparse_local_rerendering" else {}),
                    )
            else:
                direct_base = self.v4g_refiner(
                    base_prob, image, semantic_map, text_features,
                    base_prob=base_prob, flow_px=zero_flow,
                    mc_std_map=mc_std_map, mc_disagreement_map=mc_disagreement_map,
                    fine_feature_map=fine_feature_map,
                    supervision_masks=supervision_masks,
                )
                direct_geo = self.v4g_refiner(
                    geo_prob, image, semantic_map, text_features,
                    base_prob=base_prob, flow_px=flow_px,
                    mc_std_map=geo_mc_std_map, mc_disagreement_map=geo_mc_disagreement_map,
                    fine_feature_map=fine_feature_map,
                    supervision_masks=supervision_masks,
                )
        else:
            direct_base = direct_geo = None

        if self.mode == "base":
            final_logits, final_prob = factual_base_logits, base_prob
            active = recon_base
        elif self.mode == "geometry":
            final_logits, final_prob = geo_logits, geo_prob
            active = recon_geo
        elif self.mode == "residual":
            if self.v4g_sparse_direct_refiner:
                assert direct_base is not None
                final_logits, final_prob = direct_base["logits"], direct_base["prob"]
            else:
                final_logits, final_prob = recon_base["logits"], recon_base["prob"]
            active = recon_base
        else:
            if self.v4g_sparse_direct_refiner:
                assert direct_geo is not None
                final_logits, final_prob = direct_geo["logits"], direct_geo["prob"]
            else:
                final_logits, final_prob = recon_geo["logits"], recon_geo["prob"]
            active = recon_geo

        candidate_logits = torch.cat([factual_base_logits, final_logits], dim=1)
        candidate_probs = torch.cat([base_prob, final_prob], dim=1)

        flow_mag = torch.sqrt(flow_px[:, 0].square() + flow_px[:, 1].square() + 1.0e-12)
        flow_jac_mean, flow_folding = self._flow_jacobian_stats(flow_px)
        final_change = (final_prob - base_prob).abs().mean(dim=(1, 2, 3))
        geo_change = (geo_prob - base_prob).abs().mean(dim=(1, 2, 3))
        b = base_logits.shape[0]
        z = base_prob.new_zeros(b)
        ones = base_prob.new_ones(b)
        quality_probs = base_prob.new_zeros((b, 2))
        quality_probs[:, 1] = 1.0
        selector_hard = base_prob.new_ones((b, 1))
        effective_rank = torch.where(final_change > 1.0e-8, base_prob.new_full((b,), 2.0), ones)

        aux: Dict[str, torch.Tensor] = {
            "candidates": candidate_logits,
            "candidate_probs": candidate_probs,
            "mhcs_final_logits": final_logits,
            "mhcs_final_probs": final_prob,
            "mhcs_local_probs": final_prob,
            "mhcs_surface_hard_probs": final_prob,
            "mhcs_global_selected_probs": final_prob,
            "mhcs_quality_probs": quality_probs,
            "mhcs_quality_logits": torch.log(quality_probs.clamp_min(EPS)),
            "mhcs_quality_pred": quality_probs,
            "mhcs_global_weights": quality_probs,
            "mhcs_gate_alpha": final_change,
            "mhcs_effective_rank": effective_rank,
            "v20_selector_hard": selector_hard,
            "direct_fused_probs": final_prob,
            "router_fused_probs": final_prob,
            "v20_fused_probs": final_prob,
            "v20_hard_fused_probs": final_prob,

            "geotopo_base_logits": factual_base_logits,
            "geotopo_base_probs": base_prob,
            "geotopo_geometry_flow_px": flow_px,
            "geotopo_geometry_logits": geo_logits,
            "geotopo_geometry_probs": geo_prob,

            "geotopo_residual_only_logits": (direct_base["logits"] if direct_base is not None else recon_base["logits"]),
            "geotopo_residual_only_probs": (direct_base["prob"] if direct_base is not None else recon_base["prob"]),
            "geotopo_reconstruction_after_geometry_logits": (direct_geo["logits"] if direct_geo is not None else recon_geo["logits"]),
            "geotopo_reconstruction_after_geometry_probs": (direct_geo["prob"] if direct_geo is not None else recon_geo["prob"]),
            "geotopo_final_logits": final_logits,
            "geotopo_final_probs": final_prob,

            # V4C typed WHERE/WHAT + directional HOW outputs.  Binary error
            # aliases are retained so legacy trainers/loggers keep working.
            "geotopo_reconstruction_base_error_logits": recon_base["error_logits"],
            "geotopo_reconstruction_base_error_probs": recon_base["error_prob"],
            "geotopo_reconstruction_after_geometry_error_logits": recon_geo["error_logits"],
            "geotopo_reconstruction_after_geometry_error_probs": recon_geo["error_prob"],
            "geotopo_reconstruction_base_typed_logits": recon_base["typed_logits"],
            "geotopo_reconstruction_base_typed_probs": recon_base["typed_prob"],
            "geotopo_reconstruction_after_geometry_typed_logits": recon_geo["typed_logits"],
            "geotopo_reconstruction_after_geometry_typed_probs": recon_geo["typed_prob"],
            "geotopo_reconstruction_base_q_correct": recon_base["q_correct"],
            "geotopo_reconstruction_base_q_fn": recon_base["q_fn"],
            "geotopo_reconstruction_base_q_fp": recon_base["q_fp"],
            "geotopo_reconstruction_after_geometry_q_correct": recon_geo["q_correct"],
            "geotopo_reconstruction_after_geometry_q_fn": recon_geo["q_fn"],
            "geotopo_reconstruction_after_geometry_q_fp": recon_geo["q_fp"],
            # V4D deploy-severity fields.  Unlike q_FN/q_FP these are not
            # class-weighted CE posteriors and therefore can be used as edit
            # amplitudes.
            "geotopo_reconstruction_base_severity_logits": recon_base["severity_logits"],
            "geotopo_reconstruction_base_severity_fn": recon_base["severity_fn"],
            "geotopo_reconstruction_base_severity_fp": recon_base["severity_fp"],
            "geotopo_reconstruction_after_geometry_severity_logits": recon_geo["severity_logits"],
            "geotopo_reconstruction_after_geometry_severity_fn": recon_geo["severity_fn"],
            "geotopo_reconstruction_after_geometry_severity_fp": recon_geo["severity_fp"],
            "geotopo_reconstruction_base_add_magnitude": recon_base["add_magnitude"],
            "geotopo_reconstruction_base_remove_magnitude": recon_base["remove_magnitude"],
            "geotopo_reconstruction_after_geometry_add_magnitude": recon_geo["add_magnitude"],
            "geotopo_reconstruction_after_geometry_remove_magnitude": recon_geo["remove_magnitude"],
            # V4G uncertainty-guided sparse direct-refinement outputs.
            "geotopo_reconstruction_base_v4g_selection_mask": (direct_base["selection_mask"] if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_v4g_selection_mask": (direct_geo["selection_mask"] if direct_geo is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_base_v4g_selection_score": (direct_base["selection_score"] if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_v4g_selection_score": (direct_geo["selection_score"] if direct_geo is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_base_v4g_refined_logits": (direct_base["refined_logits"] if direct_base is not None else factual_base_logits),
            "geotopo_reconstruction_after_geometry_v4g_refined_logits": (direct_geo["refined_logits"] if direct_geo is not None else geo_logits),
            "geotopo_reconstruction_base_v4g_refined_prob": (direct_base["refined_prob"] if direct_base is not None else base_prob),
            "geotopo_reconstruction_after_geometry_v4g_refined_prob": (direct_geo["refined_prob"] if direct_geo is not None else geo_prob),
            "geotopo_reconstruction_base_v4g_delta_logit": (direct_base["delta_logit"] if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_v4g_delta_logit": (direct_geo["delta_logit"] if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_v4g_margin_uncertainty": (direct_base["margin_uncertainty"] if direct_base is not None else self._uncertainty(base_prob)),
            "geotopo_reconstruction_after_geometry_v4g_margin_uncertainty": (direct_geo["margin_uncertainty"] if direct_geo is not None else self._uncertainty(geo_prob)),
            "geotopo_reconstruction_base_v4g_mc_std_map": (direct_base["mc_std_map"] if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_v4g_mc_std_map": (direct_geo["mc_std_map"] if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_v4g_mc_disagreement_map": (direct_base["mc_disagreement_map"] if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_v4g_mc_disagreement_map": (direct_geo["mc_disagreement_map"] if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_v4g_entropy_map": (direct_base["entropy_map"] if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_v4g_entropy_map": (direct_geo["entropy_map"] if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_v4g_dn_anchor_prob": (direct_base["dn_anchor_prob"] if direct_base is not None else base_prob),
            "geotopo_reconstruction_after_geometry_v4g_dn_anchor_prob": (direct_geo["dn_anchor_prob"] if direct_geo is not None else geo_prob),
            "geotopo_reconstruction_base_v4g_dn_corruption_mask": (direct_base["dn_corruption_mask"] if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_v4g_dn_corruption_mask": (direct_geo["dn_corruption_mask"] if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_v4g_dn_selection_mask": (direct_base["dn_selection_mask"] if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_v4g_dn_selection_mask": (direct_geo["dn_selection_mask"] if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_v4g_dn_refined_logits": (direct_base["dn_refined_logits"] if direct_base is not None else factual_base_logits),
            "geotopo_reconstruction_after_geometry_v4g_dn_refined_logits": (direct_geo["dn_refined_logits"] if direct_geo is not None else geo_logits),
            "geotopo_reconstruction_base_v4g_dn_refined_prob": (direct_base["dn_refined_prob"] if direct_base is not None else base_prob),
            "geotopo_reconstruction_after_geometry_v4g_dn_refined_prob": (direct_geo["dn_refined_prob"] if direct_geo is not None else geo_prob),
            "geotopo_reconstruction_base_v4g_dn_final_prob": (direct_base["dn_final_prob"] if direct_base is not None else base_prob),
            "geotopo_reconstruction_after_geometry_v4g_dn_final_prob": (direct_geo["dn_final_prob"] if direct_geo is not None else geo_prob),
            "geotopo_reconstruction_base_v4g_dn_delta_logit": (direct_base["dn_delta_logit"] if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_v4g_dn_delta_logit": (direct_geo["dn_delta_logit"] if direct_geo is not None else torch.zeros_like(geo_prob)),
            # R4 exogenous local FLIP/KEEP action outputs.
            "geotopo_reconstruction_base_v4g_r4_flip_logits": (direct_base["r4_flip_logits"] if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_v4g_r4_flip_logits": (direct_geo["r4_flip_logits"] if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_v4g_r4_flip_prob": (direct_base["r4_flip_prob"] if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_v4g_r4_flip_prob": (direct_geo["r4_flip_prob"] if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_v4g_r4_flip_mask": (direct_base["r4_flip_mask"] if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_v4g_r4_flip_mask": (direct_geo["r4_flip_mask"] if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_v4g_r4_synth_flip_logits": (direct_base["r4_synth_flip_logits"] if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_v4g_r4_synth_flip_logits": (direct_geo["r4_synth_flip_logits"] if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_v4g_r4_synth_flip_prob": (direct_base["r4_synth_flip_prob"] if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_v4g_r4_synth_flip_prob": (direct_geo["r4_synth_flip_prob"] if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_v4g_r4_synth_target": (direct_base["r4_synth_target"] if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_v4g_r4_synth_target": (direct_geo["r4_synth_target"] if direct_geo is not None else torch.zeros_like(geo_prob)),
            # C2R counterfactual-consensus regional reconstruction outputs.
            "geotopo_reconstruction_base_c2r_center_mask": (direct_base.get("c2r_center_mask", torch.zeros_like(base_prob)) if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_c2r_center_mask": (direct_geo.get("c2r_center_mask", torch.zeros_like(geo_prob)) if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_c2r_region_mask": (direct_base.get("c2r_region_mask", torch.zeros_like(base_prob)) if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_c2r_region_mask": (direct_geo.get("c2r_region_mask", torch.zeros_like(geo_prob)) if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_c2r_view_logits": (direct_base.get("c2r_view_logits", torch.zeros_like(base_prob).expand(-1, 3, -1, -1)) if direct_base is not None else torch.zeros_like(base_prob).expand(-1, 3, -1, -1)),
            "geotopo_reconstruction_after_geometry_c2r_view_logits": (direct_geo.get("c2r_view_logits", torch.zeros_like(geo_prob).expand(-1, 3, -1, -1)) if direct_geo is not None else torch.zeros_like(geo_prob).expand(-1, 3, -1, -1)),
            "geotopo_reconstruction_base_c2r_view_probs": (direct_base.get("c2r_view_probs", base_prob.expand(-1, 3, -1, -1)) if direct_base is not None else base_prob.expand(-1, 3, -1, -1)),
            "geotopo_reconstruction_after_geometry_c2r_view_probs": (direct_geo.get("c2r_view_probs", geo_prob.expand(-1, 3, -1, -1)) if direct_geo is not None else geo_prob.expand(-1, 3, -1, -1)),
            "geotopo_reconstruction_base_c2r_mean_prob": (direct_base.get("c2r_mean_prob", base_prob) if direct_base is not None else base_prob),
            "geotopo_reconstruction_after_geometry_c2r_mean_prob": (direct_geo.get("c2r_mean_prob", geo_prob) if direct_geo is not None else geo_prob),
            "geotopo_reconstruction_base_c2r_consensus_mask": (direct_base.get("c2r_consensus_mask", torch.zeros_like(base_prob)) if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_c2r_consensus_mask": (direct_geo.get("c2r_consensus_mask", torch.zeros_like(geo_prob)) if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_c2r_edit_mask": (direct_base.get("c2r_edit_mask", torch.zeros_like(base_prob)) if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_c2r_edit_mask": (direct_geo.get("c2r_edit_mask", torch.zeros_like(geo_prob)) if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_c2r_eroded_anchor": (direct_base.get("c2r_eroded_anchor", base_prob) if direct_base is not None else base_prob),
            "geotopo_reconstruction_after_geometry_c2r_eroded_anchor": (direct_geo.get("c2r_eroded_anchor", geo_prob) if direct_geo is not None else geo_prob),
            "geotopo_reconstruction_base_c2r_factual_anchor": (direct_base.get("c2r_factual_anchor", base_prob) if direct_base is not None else base_prob),
            "geotopo_reconstruction_after_geometry_c2r_factual_anchor": (direct_geo.get("c2r_factual_anchor", geo_prob) if direct_geo is not None else geo_prob),
            "geotopo_reconstruction_base_c2r_dilated_anchor": (direct_base.get("c2r_dilated_anchor", base_prob) if direct_base is not None else base_prob),
            "geotopo_reconstruction_after_geometry_c2r_dilated_anchor": (direct_geo.get("c2r_dilated_anchor", geo_prob) if direct_geo is not None else geo_prob),
            # Canonical ROI C2R-v2 structural outputs.
            "geotopo_reconstruction_base_c2r_candidate_mask": (direct_base.get("c2r_candidate_mask", torch.zeros_like(base_prob)) if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_c2r_candidate_mask": (direct_geo.get("c2r_candidate_mask", torch.zeros_like(geo_prob)) if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_c2r_commit_mask": (direct_base.get("c2r_commit_mask", torch.zeros_like(base_prob)) if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_c2r_commit_mask": (direct_geo.get("c2r_commit_mask", torch.zeros_like(geo_prob)) if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_c2r_v2_enabled": (direct_base.get("c2r_v2_enabled", base_prob.new_zeros((base_prob.shape[0],))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0],))),
            "geotopo_reconstruction_after_geometry_c2r_v2_enabled": (direct_geo.get("c2r_v2_enabled", geo_prob.new_zeros((geo_prob.shape[0],))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0],))),
            "geotopo_reconstruction_base_c2r_roi_overlap_pixel_count": (direct_base.get("c2r_roi_overlap_pixel_count", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_c2r_roi_overlap_pixel_count": (direct_geo.get("c2r_roi_overlap_pixel_count", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_c2r_roi_unique_pixel_count": (direct_base.get("c2r_roi_unique_pixel_count", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_c2r_roi_unique_pixel_count": (direct_geo.get("c2r_roi_unique_pixel_count", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_c2r_center_min_chebyshev_distance": (direct_base.get("c2r_center_min_chebyshev_distance", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_c2r_center_min_chebyshev_distance": (direct_geo.get("c2r_center_min_chebyshev_distance", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_c2r_candidate_component_count": (direct_base.get("c2r_candidate_component_count", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_c2r_candidate_component_count": (direct_geo.get("c2r_candidate_component_count", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_c2r_committed_component_count": (direct_base.get("c2r_committed_component_count", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_c2r_committed_component_count": (direct_geo.get("c2r_committed_component_count", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_c2r_candidate_component_area_mean": (direct_base.get("c2r_candidate_component_area_mean", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_c2r_candidate_component_area_mean": (direct_geo.get("c2r_candidate_component_area_mean", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_c2r_committed_component_area_mean": (direct_base.get("c2r_committed_component_area_mean", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_c2r_committed_component_area_mean": (direct_geo.get("c2r_committed_component_area_mean", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_c2r_component_agreement_mean": (direct_base.get("c2r_component_agreement_mean", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_c2r_component_agreement_mean": (direct_geo.get("c2r_component_agreement_mean", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_c2r_component_spread_q90_mean": (direct_base.get("c2r_component_spread_q90_mean", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_c2r_component_spread_q90_mean": (direct_geo.get("c2r_component_spread_q90_mean", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_c2r_component_confidence_q10_mean": (direct_base.get("c2r_component_confidence_q10_mean", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_c2r_component_confidence_q10_mean": (direct_geo.get("c2r_component_confidence_q10_mean", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),

            # PC2R-v3 posterior-consistency diagnostics.
            "geotopo_reconstruction_base_pc2r_v3_enabled": (direct_base.get("pc2r_v3_enabled", base_prob.new_zeros((base_prob.shape[0],))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0],))),
            "geotopo_reconstruction_after_geometry_pc2r_v3_enabled": (direct_geo.get("pc2r_v3_enabled", geo_prob.new_zeros((geo_prob.shape[0],))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0],))),
            "geotopo_reconstruction_base_pc2r_v31_enabled": (direct_base.get("pc2r_v31_enabled", base_prob.new_zeros((base_prob.shape[0],))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0],))),
            "geotopo_reconstruction_after_geometry_pc2r_v31_enabled": (direct_geo.get("pc2r_v31_enabled", geo_prob.new_zeros((geo_prob.shape[0],))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0],))),
            "geotopo_reconstruction_base_pc2r_v32_enabled": base_prob.new_full((base_prob.shape[0],), 1.0 if self.pc2r_v32_enabled else 0.0),
            "geotopo_reconstruction_after_geometry_pc2r_v32_enabled": geo_prob.new_full((geo_prob.shape[0],), 1.0 if self.pc2r_v32_enabled else 0.0),
            "geotopo_reconstruction_base_c2r_raw_correction_mask": (direct_base.get("c2r_raw_correction_mask", torch.zeros_like(base_prob)) if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_c2r_raw_correction_mask": (direct_geo.get("c2r_raw_correction_mask", torch.zeros_like(geo_prob)) if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_c2r_risk_support_mask": (direct_base.get("c2r_risk_support_mask", torch.zeros_like(base_prob)) if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_c2r_risk_support_mask": (direct_geo.get("c2r_risk_support_mask", torch.zeros_like(geo_prob)) if direct_geo is not None else torch.zeros_like(geo_prob)),
            # PC2R-v3.2 exact deployment-stage masks/probabilities for native
            # operator decomposition.  They are diagnostics only.
            "geotopo_reconstruction_base_pc2r_stage_candidate_prob": (direct_base.get("pc2r_stage_candidate_prob", base_prob) if direct_base is not None else base_prob),
            "geotopo_reconstruction_after_geometry_pc2r_stage_candidate_prob": (direct_geo.get("pc2r_stage_candidate_prob", geo_prob) if direct_geo is not None else geo_prob),
            "geotopo_reconstruction_base_pc2r_stage_area_prob": (direct_base.get("pc2r_stage_area_prob", base_prob) if direct_base is not None else base_prob),
            "geotopo_reconstruction_after_geometry_pc2r_stage_area_prob": (direct_geo.get("pc2r_stage_area_prob", geo_prob) if direct_geo is not None else geo_prob),
            "geotopo_reconstruction_base_pc2r_stage_risk_prob": (direct_base.get("pc2r_stage_risk_prob", base_prob) if direct_base is not None else base_prob),
            "geotopo_reconstruction_after_geometry_pc2r_stage_risk_prob": (direct_geo.get("pc2r_stage_risk_prob", geo_prob) if direct_geo is not None else geo_prob),
            "geotopo_reconstruction_base_pc2r_stage_direction_prob": (direct_base.get("pc2r_stage_direction_prob", base_prob) if direct_base is not None else base_prob),
            "geotopo_reconstruction_after_geometry_pc2r_stage_direction_prob": (direct_geo.get("pc2r_stage_direction_prob", geo_prob) if direct_geo is not None else geo_prob),
            "geotopo_reconstruction_base_pc2r_stage_spread_prob": (direct_base.get("pc2r_stage_spread_prob", base_prob) if direct_base is not None else base_prob),
            "geotopo_reconstruction_after_geometry_pc2r_stage_spread_prob": (direct_geo.get("pc2r_stage_spread_prob", geo_prob) if direct_geo is not None else geo_prob),
            "geotopo_reconstruction_base_pc2r_stage_strength_prob": (direct_base.get("pc2r_stage_strength_prob", base_prob) if direct_base is not None else base_prob),
            "geotopo_reconstruction_after_geometry_pc2r_stage_strength_prob": (direct_geo.get("pc2r_stage_strength_prob", geo_prob) if direct_geo is not None else geo_prob),
            "geotopo_reconstruction_base_c2r_candidate_component_area_total": (direct_base.get("c2r_candidate_component_area_total", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_c2r_candidate_component_area_total": (direct_geo.get("c2r_candidate_component_area_total", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_c2r_committed_component_area_total": (direct_base.get("c2r_committed_component_area_total", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_c2r_committed_component_area_total": (direct_geo.get("c2r_committed_component_area_total", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_pc2r_component_risk_overlap_mean": (direct_base.get("pc2r_component_risk_overlap_mean", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_pc2r_component_risk_overlap_mean": (direct_geo.get("pc2r_component_risk_overlap_mean", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_pc2r_component_raw_count": (direct_base.get("pc2r_component_raw_count", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_pc2r_component_raw_count": (direct_geo.get("pc2r_component_raw_count", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_pc2r_component_area_pass_count": (direct_base.get("pc2r_component_area_pass_count", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_pc2r_component_area_pass_count": (direct_geo.get("pc2r_component_area_pass_count", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_pc2r_component_risk_pass_count": (direct_base.get("pc2r_component_risk_pass_count", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_pc2r_component_risk_pass_count": (direct_geo.get("pc2r_component_risk_pass_count", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_pc2r_component_direction_pass_count": (direct_base.get("pc2r_component_direction_pass_count", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_pc2r_component_direction_pass_count": (direct_geo.get("pc2r_component_direction_pass_count", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_pc2r_component_spread_pass_count": (direct_base.get("pc2r_component_spread_pass_count", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_pc2r_component_spread_pass_count": (direct_geo.get("pc2r_component_spread_pass_count", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_pc2r_component_strength_pass_count": (direct_base.get("pc2r_component_strength_pass_count", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_pc2r_component_strength_pass_count": (direct_geo.get("pc2r_component_strength_pass_count", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_pc2r_component_all_pass_count": (direct_base.get("pc2r_component_all_pass_count", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_pc2r_component_all_pass_count": (direct_geo.get("pc2r_component_all_pass_count", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_pc2r_selected_posterior_diversity": (direct_base.get("pc2r_selected_posterior_diversity", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_pc2r_selected_posterior_diversity": (direct_geo.get("pc2r_selected_posterior_diversity", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_pc2r_selected_center_bias_abs": (direct_base.get("pc2r_selected_center_bias_abs", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_pc2r_selected_center_bias_abs": (direct_geo.get("pc2r_selected_center_bias_abs", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_pc2r_selected_center_bias_signed": (direct_base.get("pc2r_selected_center_bias_signed", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_pc2r_selected_center_bias_signed": (direct_geo.get("pc2r_selected_center_bias_signed", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_pc2r_all_center_bias_logit_abs": (direct_base.get("pc2r_all_center_bias_logit_abs", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_pc2r_all_center_bias_logit_abs": (direct_geo.get("pc2r_all_center_bias_logit_abs", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_pc2r_all_center_bias_logit_signed": (direct_base.get("pc2r_all_center_bias_logit_signed", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_pc2r_all_center_bias_logit_signed": (direct_geo.get("pc2r_all_center_bias_logit_signed", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_pc2r_all_center_bias_prob_abs": (direct_base.get("pc2r_all_center_bias_prob_abs", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_pc2r_all_center_bias_prob_abs": (direct_geo.get("pc2r_all_center_bias_prob_abs", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_pc2r_selected_center_bias_prob_abs": (direct_base.get("pc2r_selected_center_bias_prob_abs", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_pc2r_selected_center_bias_prob_abs": (direct_geo.get("pc2r_selected_center_bias_prob_abs", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_pc2r_selected_vs_all_logit_bias_abs": (direct_base.get("pc2r_selected_vs_all_logit_bias_abs", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_pc2r_selected_vs_all_logit_bias_abs": (direct_geo.get("pc2r_selected_vs_all_logit_bias_abs", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_pc2r_branch_deviation_rms": (direct_base.get("pc2r_branch_deviation_rms", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_pc2r_branch_deviation_rms": (direct_geo.get("pc2r_branch_deviation_rms", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_pc2r_branch_direction_unanimity": (direct_base.get("pc2r_branch_direction_unanimity", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_pc2r_branch_direction_unanimity": (direct_geo.get("pc2r_branch_direction_unanimity", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_pc2r_mean_abs_delta_logit": (direct_base.get("pc2r_mean_abs_delta_logit", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_pc2r_mean_abs_delta_logit": (direct_geo.get("pc2r_mean_abs_delta_logit", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_pc2r_reliance_factualized": (direct_base.get("pc2r_reliance_factualized", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_pc2r_reliance_factualized": (direct_geo.get("pc2r_reliance_factualized", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_pc2r_reliance_shuffled": (direct_base.get("pc2r_reliance_shuffled", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_pc2r_reliance_shuffled": (direct_geo.get("pc2r_reliance_shuffled", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),

            # AEFR: action/evidence-factorized factual action and posterior-stability diagnostics.
            "geotopo_reconstruction_base_aefr_enabled": (direct_base.get("aefr_enabled", base_prob.new_zeros((base_prob.shape[0],))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0],))),
            "geotopo_reconstruction_after_geometry_aefr_enabled": (direct_geo.get("aefr_enabled", geo_prob.new_zeros((geo_prob.shape[0],))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0],))),
            "geotopo_reconstruction_base_aefr_stage_id": (direct_base.get("aefr_stage_id", base_prob.new_zeros((base_prob.shape[0],))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0],))),
            "geotopo_reconstruction_after_geometry_aefr_stage_id": (direct_geo.get("aefr_stage_id", geo_prob.new_zeros((geo_prob.shape[0],))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0],))),
            "geotopo_reconstruction_base_aefr_joint_geometry_grad": (direct_base.get("aefr_joint_geometry_grad", base_prob.new_zeros((base_prob.shape[0],))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0],))),
            "geotopo_reconstruction_after_geometry_aefr_joint_geometry_grad": (direct_geo.get("aefr_joint_geometry_grad", geo_prob.new_zeros((geo_prob.shape[0],))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0],))),
            "geotopo_reconstruction_base_aefr_transition_aware": (direct_base.get("aefr_transition_aware", base_prob.new_zeros((base_prob.shape[0],))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0],))),
            "geotopo_reconstruction_after_geometry_aefr_transition_aware": (direct_geo.get("aefr_transition_aware", geo_prob.new_zeros((geo_prob.shape[0],))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0],))),
            "geotopo_reconstruction_base_aefr_raw_flow_evidence_enabled": (direct_base.get("aefr_raw_flow_evidence_enabled", base_prob.new_zeros((base_prob.shape[0],))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0],))),
            "geotopo_reconstruction_after_geometry_aefr_raw_flow_evidence_enabled": (direct_geo.get("aefr_raw_flow_evidence_enabled", geo_prob.new_zeros((geo_prob.shape[0],))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0],))),
            "geotopo_reconstruction_base_aefr_transition_delta_logit": (direct_base.get("aefr_transition_delta_logit", torch.zeros_like(base_prob)) if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_aefr_transition_delta_logit": (direct_geo.get("aefr_transition_delta_logit", torch.zeros_like(geo_prob)) if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_aefr_transition_abs_mean": (direct_base.get("aefr_transition_abs_mean", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_aefr_transition_abs_mean": (direct_geo.get("aefr_transition_abs_mean", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_aefr_transition_active_fraction": (direct_base.get("aefr_transition_active_fraction", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_aefr_transition_active_fraction": (direct_geo.get("aefr_transition_active_fraction", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_aefr_transition_flip_fraction": (direct_base.get("aefr_transition_flip_fraction", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_aefr_transition_flip_fraction": (direct_geo.get("aefr_transition_flip_fraction", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_aefr_action_delta_logit": (direct_base.get("aefr_action_delta_logit", torch.zeros_like(base_prob)) if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_aefr_action_delta_logit": (direct_geo.get("aefr_action_delta_logit", torch.zeros_like(geo_prob)) if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_aefr_boundary_mask": (direct_base.get("aefr_boundary_mask", torch.zeros_like(base_prob)) if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_aefr_boundary_mask": (direct_geo.get("aefr_boundary_mask", torch.zeros_like(geo_prob)) if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_aefr_boundary_displacement_px": (direct_base.get("aefr_boundary_displacement_px", torch.zeros_like(base_prob)) if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_aefr_boundary_displacement_px": (direct_geo.get("aefr_boundary_displacement_px", torch.zeros_like(geo_prob)) if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_aefr_interior_delta_logit": (direct_base.get("aefr_interior_delta_logit", torch.zeros_like(base_prob)) if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_aefr_interior_delta_logit": (direct_geo.get("aefr_interior_delta_logit", torch.zeros_like(geo_prob)) if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_aefr_state_logits": (direct_base.get("aefr_state_logits", base_prob.new_zeros((base_prob.shape[0], 5, *base_prob.shape[-2:]))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 5, *base_prob.shape[-2:]))),
            "geotopo_reconstruction_after_geometry_aefr_state_logits": (direct_geo.get("aefr_state_logits", geo_prob.new_zeros((geo_prob.shape[0], 5, *geo_prob.shape[-2:]))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 5, *geo_prob.shape[-2:]))),
            "geotopo_reconstruction_base_aefr_state_probs": (direct_base.get("aefr_state_probs", base_prob.new_zeros((base_prob.shape[0], 5, *base_prob.shape[-2:]))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 5, *base_prob.shape[-2:]))),
            "geotopo_reconstruction_after_geometry_aefr_state_probs": (direct_geo.get("aefr_state_probs", geo_prob.new_zeros((geo_prob.shape[0], 5, *geo_prob.shape[-2:]))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 5, *geo_prob.shape[-2:]))),
            "geotopo_reconstruction_base_aefr_ownership_logits": (direct_base.get("aefr_ownership_logits", base_prob.new_zeros((base_prob.shape[0], 3, *base_prob.shape[-2:]))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 3, *base_prob.shape[-2:]))),
            "geotopo_reconstruction_after_geometry_aefr_ownership_logits": (direct_geo.get("aefr_ownership_logits", geo_prob.new_zeros((geo_prob.shape[0], 3, *geo_prob.shape[-2:]))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 3, *geo_prob.shape[-2:]))),
            "geotopo_reconstruction_base_aefr_ownership_probs": (direct_base.get("aefr_ownership_probs", base_prob.new_zeros((base_prob.shape[0], 3, *base_prob.shape[-2:]))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 3, *base_prob.shape[-2:]))),
            "geotopo_reconstruction_after_geometry_aefr_ownership_probs": (direct_geo.get("aefr_ownership_probs", geo_prob.new_zeros((geo_prob.shape[0], 3, *geo_prob.shape[-2:]))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 3, *geo_prob.shape[-2:]))),
            "geotopo_reconstruction_base_aefr_error_localizer_logit": (direct_base.get("aefr_error_localizer_logit", torch.zeros_like(base_prob)) if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_aefr_error_localizer_logit": (direct_geo.get("aefr_error_localizer_logit", torch.zeros_like(geo_prob)) if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_aefr_error_localizer_prob": (direct_base.get("aefr_error_localizer_prob", torch.zeros_like(base_prob)) if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_aefr_error_localizer_prob": (direct_geo.get("aefr_error_localizer_prob", torch.zeros_like(geo_prob)) if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_aefr_error_localizer_prior": (direct_base.get("aefr_error_localizer_prior", torch.zeros_like(base_prob)) if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_aefr_error_localizer_prior": (direct_geo.get("aefr_error_localizer_prior", torch.zeros_like(geo_prob)) if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_aefr_edit_logit": (direct_base.get("aefr_edit_logit", torch.zeros_like(base_prob)) if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_aefr_edit_logit": (direct_geo.get("aefr_edit_logit", torch.zeros_like(geo_prob)) if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_aefr_edit_prob": (direct_base.get("aefr_edit_prob", torch.zeros_like(base_prob)) if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_aefr_edit_prob": (direct_geo.get("aefr_edit_prob", torch.zeros_like(geo_prob)) if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_aefr_direction_logit": (direct_base.get("aefr_direction_logit", torch.zeros_like(base_prob)) if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_aefr_direction_logit": (direct_geo.get("aefr_direction_logit", torch.zeros_like(geo_prob)) if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_aefr_direction_prob": (direct_base.get("aefr_direction_prob", torch.zeros_like(base_prob)) if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_aefr_direction_prob": (direct_geo.get("aefr_direction_prob", torch.zeros_like(geo_prob)) if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_aefr_interior_magnitude_logit": (direct_base.get("aefr_interior_magnitude_logit", torch.zeros_like(base_prob)) if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_aefr_interior_magnitude_logit": (direct_geo.get("aefr_interior_magnitude_logit", torch.zeros_like(geo_prob)) if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_aefr_signed_action": (direct_base.get("aefr_signed_action", torch.zeros_like(base_prob)) if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_aefr_signed_action": (direct_geo.get("aefr_signed_action", torch.zeros_like(geo_prob)) if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_aefr_commit_mask": (direct_base.get("aefr_commit_mask", torch.zeros_like(base_prob)) if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_aefr_commit_mask": (direct_geo.get("aefr_commit_mask", torch.zeros_like(geo_prob)) if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_aefr_boundary_magnitude_px": (direct_base.get("aefr_boundary_magnitude_px", torch.zeros_like(base_prob)) if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_aefr_boundary_magnitude_px": (direct_geo.get("aefr_boundary_magnitude_px", torch.zeros_like(geo_prob)) if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_aefr_signed_boundary_action": (direct_base.get("aefr_signed_boundary_action", torch.zeros_like(base_prob)) if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_aefr_signed_boundary_action": (direct_geo.get("aefr_signed_boundary_action", torch.zeros_like(geo_prob)) if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_aefr_signed_interior_action": (direct_base.get("aefr_signed_interior_action", torch.zeros_like(base_prob)) if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_aefr_signed_interior_action": (direct_geo.get("aefr_signed_interior_action", torch.zeros_like(geo_prob)) if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_aefr_posterior_stability_support": (direct_base.get("aefr_posterior_stability_support", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_aefr_posterior_stability_support": (direct_geo.get("aefr_posterior_stability_support", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_aefr_posterior_stability_improvement": (direct_base.get("aefr_posterior_stability_improvement", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_aefr_posterior_stability_improvement": (direct_geo.get("aefr_posterior_stability_improvement", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_aefr_posterior_disagreement_pre": (direct_base.get("aefr_posterior_disagreement_pre", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_aefr_posterior_disagreement_pre": (direct_geo.get("aefr_posterior_disagreement_pre", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_aefr_posterior_disagreement_post": (direct_base.get("aefr_posterior_disagreement_post", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_aefr_posterior_disagreement_post": (direct_geo.get("aefr_posterior_disagreement_post", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_aefr_action_support_fraction": (direct_base.get("aefr_action_support_fraction", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_aefr_action_support_fraction": (direct_geo.get("aefr_action_support_fraction", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_aefr_posterior_diversity_all": (direct_base.get("aefr_posterior_diversity_all", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_aefr_posterior_diversity_all": (direct_geo.get("aefr_posterior_diversity_all", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_aefr_posterior_center_bias_abs": (direct_base.get("aefr_posterior_center_bias_abs", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_aefr_posterior_center_bias_abs": (direct_geo.get("aefr_posterior_center_bias_abs", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_aefr_posterior_center_bias_signed": (direct_base.get("aefr_posterior_center_bias_signed", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_aefr_posterior_center_bias_signed": (direct_geo.get("aefr_posterior_center_bias_signed", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),

            # SLR sparse compute routing + direct local re-rendering outputs.
            "geotopo_reconstruction_base_slr_selector_logits": (direct_base.get("slr_selector_logits", torch.zeros_like(base_prob)) if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_slr_selector_logits": (direct_geo.get("slr_selector_logits", torch.zeros_like(geo_prob)) if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_slr_selector_prob": (direct_base.get("slr_selector_prob", torch.zeros_like(base_prob)) if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_slr_selector_prob": (direct_geo.get("slr_selector_prob", torch.zeros_like(geo_prob)) if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_slr_selector_target": (direct_base.get("slr_selector_target", torch.zeros_like(base_prob)) if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_slr_selector_target": (direct_geo.get("slr_selector_target", torch.zeros_like(geo_prob)) if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_slr_pred_patch_logits": (direct_base.get("slr_pred_patch_logits", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_pred_patch_logits": (direct_geo.get("slr_pred_patch_logits", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_pred_patch_probs": (direct_base.get("slr_pred_patch_probs", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_pred_patch_probs": (direct_geo.get("slr_pred_patch_probs", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_pred_patch_grid": (direct_base.get("slr_pred_patch_grid", base_prob.new_zeros((1, 1, 1, 2))) if direct_base is not None else base_prob.new_zeros((1, 1, 1, 2))),
            "geotopo_reconstruction_after_geometry_slr_pred_patch_grid": (direct_geo.get("slr_pred_patch_grid", geo_prob.new_zeros((1, 1, 1, 2))) if direct_geo is not None else geo_prob.new_zeros((1, 1, 1, 2))),
            "geotopo_reconstruction_base_slr_pred_patch_center_valid": (direct_base.get("slr_pred_patch_center_valid", base_prob.new_zeros((base_prob.shape[0], 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1))),
            "geotopo_reconstruction_after_geometry_slr_pred_patch_center_valid": (direct_geo.get("slr_pred_patch_center_valid", geo_prob.new_zeros((geo_prob.shape[0], 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1))),
            "geotopo_reconstruction_base_slr_pred_patch_valid": (direct_base.get("slr_pred_patch_valid", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_pred_patch_valid": (direct_geo.get("slr_pred_patch_valid", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_pred_patch_target": (direct_base.get("slr_pred_patch_target", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_pred_patch_target": (direct_geo.get("slr_pred_patch_target", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_pred_patch_anchor_prob": (direct_base.get("slr_pred_patch_anchor_prob", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_pred_patch_anchor_prob": (direct_geo.get("slr_pred_patch_anchor_prob", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_raw_patch_logits": (direct_base.get("slr_raw_patch_logits", direct_base.get("slr_pred_patch_logits")) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_raw_patch_logits": (direct_geo.get("slr_raw_patch_logits", direct_geo.get("slr_pred_patch_logits")) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_raw_patch_probs": (direct_base.get("slr_raw_patch_probs", direct_base.get("slr_pred_patch_probs")) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_raw_patch_probs": (direct_geo.get("slr_raw_patch_probs", direct_geo.get("slr_pred_patch_probs")) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_pred_patch_action_support": (direct_base.get("slr_pred_patch_action_support", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_pred_patch_action_support": (direct_geo.get("slr_pred_patch_action_support", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_action_region_mask": (direct_base.get("slr_action_region_mask", torch.zeros_like(base_prob)) if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_slr_action_region_mask": (direct_geo.get("slr_action_region_mask", torch.zeros_like(geo_prob)) if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_slr_action_weight_full": (direct_base.get("slr_action_weight_full", torch.zeros_like(base_prob)) if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_slr_action_weight_full": (direct_geo.get("slr_action_weight_full", torch.zeros_like(geo_prob)) if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_slr_positive_from_selector_fraction": (direct_base.get("slr_positive_from_selector_fraction", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_slr_positive_from_selector_fraction": (direct_geo.get("slr_positive_from_selector_fraction", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_slr_pred_patch_delta_logit": (direct_base.get("slr_pred_patch_delta_logit", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_pred_patch_delta_logit": (direct_geo.get("slr_pred_patch_delta_logit", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_pred_sdf_delta": (direct_base.get("slr_pred_sdf_delta", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_pred_sdf_delta": (direct_geo.get("slr_pred_sdf_delta", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_pred_sdf_absolute": (direct_base.get("slr_pred_sdf_absolute", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_pred_sdf_absolute": (direct_geo.get("slr_pred_sdf_absolute", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_pred_sdf_anchor": (direct_base.get("slr_pred_sdf_anchor", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_pred_sdf_anchor": (direct_geo.get("slr_pred_sdf_anchor", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_pred_sdf_target": (direct_base.get("slr_pred_sdf_target", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_pred_sdf_target": (direct_geo.get("slr_pred_sdf_target", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_pred_sdf_support": (direct_base.get("slr_pred_sdf_support", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_pred_sdf_support": (direct_geo.get("slr_pred_sdf_support", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_blend_weight_sum": (direct_base.get("slr_blend_weight_sum", torch.zeros_like(base_prob)) if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_slr_blend_weight_sum": (direct_geo.get("slr_blend_weight_sum", torch.zeros_like(geo_prob)) if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_slr_overlap_disagreement": (direct_base.get("slr_overlap_disagreement", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_slr_overlap_disagreement": (direct_geo.get("slr_overlap_disagreement", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_slr_overlap_sdf_disagreement": (direct_base.get("slr_overlap_sdf_disagreement", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_slr_overlap_sdf_disagreement": (direct_geo.get("slr_overlap_sdf_disagreement", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_slr_sdf_delta_full": (direct_base.get("slr_sdf_delta_full", torch.zeros_like(base_prob)) if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_slr_sdf_delta_full": (direct_geo.get("slr_sdf_delta_full", torch.zeros_like(geo_prob)) if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_slr_sdf_absolute_full": (direct_base.get("slr_sdf_absolute_full", torch.zeros_like(base_prob)) if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_slr_sdf_absolute_full": (direct_geo.get("slr_sdf_absolute_full", torch.zeros_like(geo_prob)) if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_slr_anchor_sdf_full": (direct_base.get("slr_anchor_sdf_full", torch.zeros_like(base_prob)) if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_slr_anchor_sdf_full": (direct_geo.get("slr_anchor_sdf_full", torch.zeros_like(geo_prob)) if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_slr_true_hr_active": (direct_base.get("slr_true_hr_active", base_prob.new_zeros(())) if direct_base is not None else base_prob.new_zeros(())),
            "geotopo_reconstruction_after_geometry_slr_true_hr_active": (direct_geo.get("slr_true_hr_active", geo_prob.new_zeros(())) if direct_geo is not None else geo_prob.new_zeros(())),
            "geotopo_reconstruction_base_slr_sdf_discrepancy_map": (direct_base.get("slr_sdf_discrepancy_map", torch.zeros_like(base_prob)) if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_slr_sdf_discrepancy_map": (direct_geo.get("slr_sdf_discrepancy_map", torch.zeros_like(geo_prob)) if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_slr_oracle_patch_logits": (direct_base.get("slr_oracle_patch_logits", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_oracle_patch_logits": (direct_geo.get("slr_oracle_patch_logits", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_oracle_patch_probs": (direct_base.get("slr_oracle_patch_probs", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_oracle_patch_probs": (direct_geo.get("slr_oracle_patch_probs", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_oracle_patch_valid": (direct_base.get("slr_oracle_patch_valid", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_oracle_patch_valid": (direct_geo.get("slr_oracle_patch_valid", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_oracle_patch_target": (direct_base.get("slr_oracle_patch_target", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_oracle_patch_target": (direct_geo.get("slr_oracle_patch_target", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_oracle_sdf_delta": (direct_base.get("slr_oracle_sdf_delta", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_oracle_sdf_delta": (direct_geo.get("slr_oracle_sdf_delta", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_oracle_sdf_target": (direct_base.get("slr_oracle_sdf_target", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_oracle_sdf_target": (direct_geo.get("slr_oracle_sdf_target", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_oracle_sdf_support": (direct_base.get("slr_oracle_sdf_support", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_oracle_sdf_support": (direct_geo.get("slr_oracle_sdf_support", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_oracle_region_mask": (direct_base.get("slr_oracle_region_mask", torch.zeros_like(base_prob)) if direct_base is not None else torch.zeros_like(base_prob)),
            "geotopo_reconstruction_after_geometry_slr_oracle_region_mask": (direct_geo.get("slr_oracle_region_mask", torch.zeros_like(geo_prob)) if direct_geo is not None else torch.zeros_like(geo_prob)),
            "geotopo_reconstruction_base_slr_positive_patch_logits": (direct_base.get("slr_positive_patch_logits", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_positive_patch_logits": (direct_geo.get("slr_positive_patch_logits", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_positive_patch_probs": (direct_base.get("slr_positive_patch_probs", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_positive_patch_probs": (direct_geo.get("slr_positive_patch_probs", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_positive_patch_valid": (direct_base.get("slr_positive_patch_valid", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_positive_patch_valid": (direct_geo.get("slr_positive_patch_valid", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_positive_patch_target": (direct_base.get("slr_positive_patch_target", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_positive_patch_target": (direct_geo.get("slr_positive_patch_target", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_positive_sdf_delta": (direct_base.get("slr_positive_sdf_delta", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_positive_sdf_delta": (direct_geo.get("slr_positive_sdf_delta", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_positive_sdf_absolute": (direct_base.get("slr_positive_sdf_absolute", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_positive_sdf_absolute": (direct_geo.get("slr_positive_sdf_absolute", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_positive_sdf_anchor": (direct_base.get("slr_positive_sdf_anchor", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_positive_sdf_anchor": (direct_geo.get("slr_positive_sdf_anchor", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_positive_sdf_target": (direct_base.get("slr_positive_sdf_target", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_positive_sdf_target": (direct_geo.get("slr_positive_sdf_target", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_positive_sdf_support": (direct_base.get("slr_positive_sdf_support", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_positive_sdf_support": (direct_geo.get("slr_positive_sdf_support", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_clean_patch_logits": (direct_base.get("slr_clean_patch_logits", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_clean_patch_logits": (direct_geo.get("slr_clean_patch_logits", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_clean_patch_probs": (direct_base.get("slr_clean_patch_probs", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_clean_patch_probs": (direct_geo.get("slr_clean_patch_probs", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_clean_patch_valid": (direct_base.get("slr_clean_patch_valid", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_clean_patch_valid": (direct_geo.get("slr_clean_patch_valid", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_clean_patch_target": (direct_base.get("slr_clean_patch_target", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_clean_patch_target": (direct_geo.get("slr_clean_patch_target", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_clean_patch_anchor_prob": (direct_base.get("slr_clean_patch_anchor_prob", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_clean_patch_anchor_prob": (direct_geo.get("slr_clean_patch_anchor_prob", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_clean_patch_delta_logit": (direct_base.get("slr_clean_patch_delta_logit", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_clean_patch_delta_logit": (direct_geo.get("slr_clean_patch_delta_logit", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_clean_sdf_delta": (direct_base.get("slr_clean_sdf_delta", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_clean_sdf_delta": (direct_geo.get("slr_clean_sdf_delta", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_clean_sdf_absolute": (direct_base.get("slr_clean_sdf_absolute", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_clean_sdf_absolute": (direct_geo.get("slr_clean_sdf_absolute", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_clean_sdf_anchor": (direct_base.get("slr_clean_sdf_anchor", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_clean_sdf_anchor": (direct_geo.get("slr_clean_sdf_anchor", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_clean_sdf_target": (direct_base.get("slr_clean_sdf_target", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_clean_sdf_target": (direct_geo.get("slr_clean_sdf_target", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_base_slr_clean_sdf_support": (direct_base.get("slr_clean_sdf_support", base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))) if direct_base is not None else base_prob.new_zeros((base_prob.shape[0], 1, 1, 1, 1))),
            "geotopo_reconstruction_after_geometry_slr_clean_sdf_support": (direct_geo.get("slr_clean_sdf_support", geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))) if direct_geo is not None else geo_prob.new_zeros((geo_prob.shape[0], 1, 1, 1, 1))),

            # V4F proposal / selective execution / bounded dose outputs.
            "geotopo_reconstruction_base_v4f_proposal_candidate_mask": recon_base["v4f_proposal_candidate_mask"],
            "geotopo_reconstruction_after_geometry_v4f_proposal_candidate_mask": recon_geo["v4f_proposal_candidate_mask"],
            "geotopo_reconstruction_base_v4f_policy_edit_logit": recon_base["v4f_policy_edit_logit"],
            "geotopo_reconstruction_after_geometry_v4f_policy_edit_logit": recon_geo["v4f_policy_edit_logit"],
            "geotopo_reconstruction_base_v4f_policy_edit_prob": recon_base["v4f_policy_edit_prob"],
            "geotopo_reconstruction_after_geometry_v4f_policy_edit_prob": recon_geo["v4f_policy_edit_prob"],
            "geotopo_reconstruction_base_v4f_policy_direction_logits": recon_base["v4f_policy_direction_logits"],
            "geotopo_reconstruction_after_geometry_v4f_policy_direction_logits": recon_geo["v4f_policy_direction_logits"],
            "geotopo_reconstruction_base_v4f_policy_direction_prob": recon_base["v4f_policy_direction_prob"],
            "geotopo_reconstruction_after_geometry_v4f_policy_direction_prob": recon_geo["v4f_policy_direction_prob"],
            "geotopo_reconstruction_base_v4f_policy_p_add": recon_base["v4f_policy_p_add"],
            "geotopo_reconstruction_after_geometry_v4f_policy_p_add": recon_geo["v4f_policy_p_add"],
            "geotopo_reconstruction_base_v4f_policy_p_remove": recon_base["v4f_policy_p_remove"],
            "geotopo_reconstruction_after_geometry_v4f_policy_p_remove": recon_geo["v4f_policy_p_remove"],
            "geotopo_reconstruction_base_v4f_soft_prob": recon_base["v4f_soft_prob"],
            "geotopo_reconstruction_after_geometry_v4f_soft_prob": recon_geo["v4f_soft_prob"],
            "geotopo_reconstruction_base_v4f_hard_prob": recon_base["v4f_hard_prob"],
            "geotopo_reconstruction_after_geometry_v4f_hard_prob": recon_geo["v4f_hard_prob"],
            "geotopo_reconstruction_base_v4f_hard_edit": recon_base["v4f_hard_edit"],
            "geotopo_reconstruction_after_geometry_v4f_hard_edit": recon_geo["v4f_hard_edit"],
            "geotopo_reconstruction_base_v4f_hard_add": recon_base["v4f_hard_add"],
            "geotopo_reconstruction_after_geometry_v4f_hard_add": recon_geo["v4f_hard_add"],
            "geotopo_reconstruction_base_v4f_hard_remove": recon_base["v4f_hard_remove"],
            "geotopo_reconstruction_after_geometry_v4f_hard_remove": recon_geo["v4f_hard_remove"],
            "geotopo_reconstruction_base_v4f_trace": recon_base["v4f_trace"],
            "geotopo_reconstruction_after_geometry_v4f_trace": recon_geo["v4f_trace"],
            "geotopo_reconstruction_base_add_delta": recon_base["add_delta"],
            "geotopo_reconstruction_base_remove_delta": recon_base["remove_delta"],
            "geotopo_reconstruction_after_geometry_add_delta": recon_geo["add_delta"],
            "geotopo_reconstruction_after_geometry_remove_delta": recon_geo["remove_delta"],
            "geotopo_reconstruction_base_add_logit_delta": recon_base["add_logit_delta"],
            "geotopo_reconstruction_base_remove_logit_delta": recon_base["remove_logit_delta"],
            "geotopo_reconstruction_after_geometry_add_logit_delta": recon_geo["add_logit_delta"],
            "geotopo_reconstruction_after_geometry_remove_logit_delta": recon_geo["remove_logit_delta"],
            "geotopo_reconstruction_base_raw": recon_base["raw"],
            "geotopo_reconstruction_base_strength": recon_base["strength"],
            "geotopo_reconstruction_after_geometry_raw": recon_geo["raw"],
            "geotopo_reconstruction_after_geometry_strength": recon_geo["strength"],
            "geotopo_residual_logit": active["raw"],

            "geotopo_flow_rms_px": flow_mag.square().mean(dim=(1, 2)).sqrt(),
            "geotopo_flow_mean_px": flow_mag.mean(dim=(1, 2)),
            "geotopo_flow_max_px": flow_mag.flatten(1).max(dim=1).values,
            "geotopo_flow_jacobian_mean": flow_jac_mean,
            "geotopo_flow_folding_fraction": flow_folding,
            "geotopo_residual_rms": active["raw"].square().mean(dim=(1, 2, 3)).sqrt(),
            "geotopo_reconstruction_strength_mean_abs": 0.5 * (active["add_magnitude"] + active["remove_magnitude"]).mean(dim=(1, 2, 3)),
            "geotopo_reconstruction_strength_p95_proxy": (0.5 * (active["add_magnitude"] + active["remove_magnitude"])).flatten(1).quantile(0.95, dim=1),
            "geotopo_reconstruction_large_magnitude_fraction_gt_095": ((active["add_magnitude"] > 0.95) | (active["remove_magnitude"] > 0.95)).float().mean(dim=(1, 2, 3)),
            "geotopo_reconstruction_magnitude_cap_fraction": ((active["add_magnitude"] >= 0.99 * self.m2_surface.max_logit_step) | (active["remove_magnitude"] >= 0.99 * self.m2_surface.max_logit_step)).float().mean(dim=(1, 2, 3)),
            "geotopo_reconstruction_saturation_fraction": ((active["add_magnitude"] >= 0.99 * self.m2_surface.max_logit_step) | (active["remove_magnitude"] >= 0.99 * self.m2_surface.max_logit_step)).float().mean(dim=(1, 2, 3)),
            "geotopo_error_probability_mean": active["error_prob"].mean(dim=(1, 2, 3)),
            "geotopo_error_probability_p95": active["error_prob"].flatten(1).quantile(0.95, dim=1),
            "geotopo_q_correct_mean": active["q_correct"].mean(dim=(1, 2, 3)),
            "geotopo_q_fn_mean": active["q_fn"].mean(dim=(1, 2, 3)),
            "geotopo_q_fp_mean": active["q_fp"].mean(dim=(1, 2, 3)),
            "geotr_v4e_operator_consistent_enabled": active["prob"].new_full((active["prob"].shape[0],), float(self.v4e_operator_consistent)),
            "geotr_v4f_selective_intervention_enabled": active["prob"].new_full((active["prob"].shape[0],), float(self.v4f_selective_intervention)),
            "geotr_v4g_sparse_direct_refiner_enabled": active["prob"].new_full((active["prob"].shape[0],), float(self.v4g_sparse_direct_refiner)),
            "geotr_v4f_proposal_coverage": active["prob"].new_full((active["prob"].shape[0],), float(self.m2_surface.v4f_proposal_coverage if self.v4f_selective_intervention else 0.0)),
            "geotr_v4f_execution_threshold": active["prob"].new_full((active["prob"].shape[0],), float(self.m2_surface.v4f_execution_threshold if self.v4f_selective_intervention else 0.0)),
            "geotr_v4d_severity_fn_mean": active["severity_fn"].mean(dim=(1, 2, 3)),
            "geotr_v4d_severity_fp_mean": active["severity_fp"].mean(dim=(1, 2, 3)),
            "geotopo_add_magnitude_mean": active["add_magnitude"].mean(dim=(1, 2, 3)),
            "geotopo_remove_magnitude_mean": active["remove_magnitude"].mean(dim=(1, 2, 3)),
            "geotopo_geometry_abs_change": geo_change,
            "geotopo_reconstruction_base_abs_change": recon_base["abs_change"],
            "geotopo_reconstruction_after_geometry_abs_change": recon_geo["abs_change"],
            "geotopo_abs_change": final_change,
            "geotopo_mode_id": base_prob.new_full(
                (b,), {"base": 0.0, "geometry": 1.0, "residual": 2.0, "full": 3.0}[self.mode]
            ),
            "mhcs_m1_distribution_log_var": self.m1_distribution_log_var,

            # Historical compatibility diagnostics.
            "mhcs_surface_nonbase_mass": final_change,
            "mhcs_surface_surrogate_nonbase_mass": final_change,
            "mhcs_surface_switch_rate": z,
            "mhcs_surface_route_entropy": z,
            "mhcs_surface_effective_sources": effective_rank,
            "mhcs_surface_active_sources": effective_rank,
            "mhcs_candidate_disagreement_mean": final_change,
            "mhcs_route_set_std_mean": candidate_probs.std(1, unbiased=False).mean(dim=(1, 2)),
            "mhcs_route_raw_abs_disagreement_mean": final_change,
            "mhcs_candidate_probability_novelty_mean": final_change,
            "mhcs_candidate_boundary_novelty_mean": z,
            "mhcs_global_adapter_delta_rms": z,
            "mhcs_local_adapter_delta_rms": active["raw"].square().mean(dim=(1, 2, 3)).sqrt(),
            "mhcs_context_gate": torch.tanh(self.context_gate).expand_as(final_change),
            "mhcs_global_set_token_cosine": ones,
        }

        # UCDRT-native diagnostics/actions are appended explicitly instead of
        # expanding the compatibility dictionary above. OCRA owns the false
        # branch of GEOTR_SLR_UCDRT_ENABLED; UCDRT owns the true branch.
        _ucdrt_keys = (
            "slr_ucdrt_enabled",
            "slr_ucdrt_state_logits",
            "slr_ucdrt_state_probs",
            "slr_ucdrt_move_px",
            "slr_ucdrt_interior_magnitude",
            "slr_ucdrt_boundary_mask",
            "slr_ucdrt_candidate_soft_probs",
            "slr_ucdrt_utility_logit",
            "slr_ucdrt_utility_value",
            "slr_ucdrt_commit_prob",
            "slr_ucdrt_commit",
            "slr_ucdrt_boundary_displacement_full",
            "slr_ucdrt_interior_action_full",
            "slr_ucdrt_action_mask_full",
            "slr_ucdrt_positive_state_logits",
            "slr_ucdrt_positive_move_px",
            "slr_ucdrt_positive_interior_magnitude",
            "slr_ucdrt_positive_anchor_prob",
            "slr_ucdrt_positive_anchor_sdf",
            "slr_ucdrt_clean_state_logits",
            "slr_ucdrt_clean_move_px",
            "slr_ucdrt_clean_interior_magnitude",
            "slr_ucdrt_clean_anchor_prob",
            "slr_ucdrt_clean_anchor_sdf",
            "slr_ucdrt_paired_state_logits",
            "slr_ucdrt_paired_move_px",
            "slr_ucdrt_paired_interior_magnitude",
            "slr_ucdrt_paired_valid",
            "slr_ucdrt_paired_target",
            "slr_ucdrt_paired_anchor_prob",
            "slr_ucdrt_paired_anchor_sdf",
            "slr_ucdrt_paired_target_sdf",
            # UCDRT-R2 native actor/critic/set outputs.
            "slr_ucdrt_r2_enabled",
            "slr_ucdrt_r2_edit_logits",
            "slr_ucdrt_r2_edit_probs",
            "slr_ucdrt_r2_type_logits",
            "slr_ucdrt_r2_type_probs",
            "slr_ucdrt_r2_move_bin_logits",
            "slr_ucdrt_r2_move_bin_probs",
            "slr_ucdrt_r2_move_dictionary_probs",
            "slr_ucdrt_r2_move_bins_px",
            "slr_ucdrt_r2_add_dose",
            "slr_ucdrt_r2_remove_dose",
            "slr_ucdrt_r2_hard_candidate_probs",
            "slr_ucdrt_r2_soft_candidate_probs",
            "slr_ucdrt_r2_critic_step_logits",
            "slr_ucdrt_r2_critic_step_values",
            "slr_ucdrt_r2_critic_step_available",
            "slr_ucdrt_r2_selection_step_index",
            "slr_ucdrt_r2_candidate_full_num",
            "slr_ucdrt_r2_candidate_full_den",
            "slr_ucdrt_r2_selected_candidate_mask",
            "slr_ucdrt_r2_positive_edit_logits",
            "slr_ucdrt_r2_positive_type_logits",
            "slr_ucdrt_r2_positive_move_bin_logits",
            "slr_ucdrt_r2_positive_move_dictionary_probs",
            "slr_ucdrt_r2_positive_add_dose",
            "slr_ucdrt_r2_positive_remove_dose",
            "slr_ucdrt_r2_positive_hard_candidate_probs",
            "slr_ucdrt_r2_paired_edit_logits",
            "slr_ucdrt_r2_paired_type_logits",
            "slr_ucdrt_r2_paired_move_bin_logits",
            "slr_ucdrt_r2_paired_move_dictionary_probs",
            "slr_ucdrt_r2_paired_add_dose",
            "slr_ucdrt_r2_paired_remove_dose",
            "slr_ucdrt_r2_paired_hard_candidate_probs",
            "slr_ucdrt_r2_paired_soft_candidate_probs",
        )
        for _tag, _direct in (("geotopo_reconstruction_base", direct_base),
                              ("geotopo_reconstruction_after_geometry", direct_geo)):
            if isinstance(_direct, dict):
                for _key in _ucdrt_keys:
                    if _key in _direct:
                        aux[f"{_tag}_{_key}"] = _direct[_key]

        # SLR2.4/OCRA typed actor exports. Keep this explicit fail-closed list so
        # a training target can never be silently replaced by a compatibility
        # zero in the wrapper dictionary above.
        _ocra_keys = (
            "slr_pred_action_state_logits", "slr_pred_action_state_probs",
            "slr_pred_action_dose", "slr_pred_signed_action",
            "slr_pred_action_type_target", "slr_pred_action_dose_target",
            "slr_pred_signed_action_target", "slr_action_delta_full",
            "slr_positive_action_state_logits", "slr_positive_action_state_probs",
            "slr_positive_action_dose", "slr_positive_signed_action",
            "slr_positive_action_support", "slr_positive_action_type_target",
            "slr_positive_action_dose_target", "slr_positive_signed_action_target",
            "slr_clean_action_state_logits", "slr_clean_action_state_probs",
            "slr_clean_action_dose", "slr_clean_signed_action",
            "slr_clean_action_support", "slr_clean_action_type_target",
            "slr_clean_action_dose_target", "slr_clean_signed_action_target",
        )
        for _tag, _direct in (("geotopo_reconstruction_base", direct_base),
                              ("geotopo_reconstruction_after_geometry", direct_geo)):
            if isinstance(_direct, dict):
                for _key in _ocra_keys:
                    if _key in _direct:
                        aux[f"{_tag}_{_key}"] = _direct[_key]

        # SPARC-HR exports its native counterfactual bank without compatibility
        # zeros. Missing fields therefore fail loudly in the SPARC loss.
        _sparc_active = direct_base if self.mode == "residual" else direct_geo
        for _tag, _direct in (("geotopo_reconstruction_base", direct_base),
                              ("geotopo_reconstruction_after_geometry", direct_geo)):
            if isinstance(_direct, dict):
                for _key, _value in _direct.items():
                    if _key.startswith("sparc_"):
                        aux[f"{_tag}_{_key}"] = _value
        if isinstance(_sparc_active, dict):
            for _key, _value in _sparc_active.items():
                if _key.startswith("sparc_"):
                    aux[_key] = _value

        return candidate_logits, aux
