"""M1-only evidence transport with causal LeST57 operator controls.

The historical unrestricted two-dimensional transport is retained as the
default compatibility path. JBT-v5 keeps the posterior-consistent v3 branch but
adds strict Base-gradient isolation, broad adaptive proposal support, smooth
FN/FP-directed magnitude learning and risk-aware multi-strength selection.
Historical paths remain opt-in for compatibility; no M2/M3 actor or router is
constructed here.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import math
import os

import torch
from torch.utils.checkpoint import checkpoint as _activation_checkpoint
import torch.nn as nn
import torch.nn.functional as F


EPS = 1.0e-4


def _jbt_memory_safe_head_forward(
    module: nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    """Memory-only execution wrapper for dense JBT heads.

    The physical training batch is unchanged. Only the JBT dense head is
    evaluated in micro-chunks along dimension 0, and its internal activations
    are recomputed during backward.

    This is deliberately NOT gradient accumulation and does not alter the
    protected MedCLIPSeg Base optimizer protocol.
    """
    chunk_size = max(
        1,
        int(os.environ.get("JBT_V6_HEAD_CHUNK_SIZE", "4"))
    )

    if x.ndim == 0 or int(x.shape[0]) <= chunk_size:
        if torch.is_grad_enabled() and any(
            p.requires_grad for p in module.parameters()
        ):
            return _activation_checkpoint(
                module,
                x,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        return module(x)

    outputs = []
    trainable = any(p.requires_grad for p in module.parameters())

    for chunk in x.split(chunk_size, dim=0):
        if torch.is_grad_enabled() and trainable:
            out = _activation_checkpoint(
                module,
                chunk,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        else:
            out = module(chunk)
        outputs.append(out)

    return torch.cat(outputs, dim=0)



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
    """Zero-initialized displacement head with optional strict scale cap.

    ``max_flow_px <= 0`` preserves the historical SemLT/FORMAL54 numerical
    path exactly: ``scale = softplus(log_scale)`` is learned but unbounded.
    ``max_flow_px > 0`` switches to ``scale = max_flow_px * sigmoid(log_scale)``
    so every component satisfies ``|d_k(x)| < max_flow_px``.  The bounded mode
    is intended for NEW confirmatory runs after the cap has been selected from
    validation transportability/oracle profiles; it must not be retrofitted to
    an already-tested checkpoint.
    """

    def __init__(self, hidden_dim: int, init_scale_px: float = 1.0,
                 output_channels: int = 2, max_flow_px: float = 0.0) -> None:
        super().__init__()
        if int(output_channels) not in {1, 2}:
            raise ValueError("GeometryHead output_channels must be 1 or 2")
        self.max_flow_px = float(max_flow_px)
        if self.max_flow_px > 0.0 and not (0.0 < float(init_scale_px) <= self.max_flow_px):
            raise ValueError("init_scale_px must be in (0, max_flow_px] in bounded mode")
        self.body = nn.Sequential(
            ConvNormGELU(hidden_dim, hidden_dim),
            ConvNormGELU(hidden_dim, hidden_dim),
        )
        self.flow_out = nn.Conv2d(
            hidden_dim, int(output_channels), kernel_size=3, padding=1
        )
        init = max(float(init_scale_px), 1.0e-3)
        if self.max_flow_px > 0.0:
            frac = min(max(init / self.max_flow_px, 1.0e-4), 1.0 - 1.0e-4)
            raw = math.log(frac / (1.0 - frac))
        else:
            raw = math.log(math.expm1(init))
        # Keep the historical parameter NAME for checkpoint compatibility.
        self.log_scale = nn.Parameter(torch.tensor(raw))
        nn.init.zeros_(self.flow_out.weight)
        nn.init.zeros_(self.flow_out.bias)

    def current_scale(self) -> torch.Tensor:
        if self.max_flow_px > 0.0:
            return float(self.max_flow_px) * torch.sigmoid(self.log_scale)
        return F.softplus(self.log_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.current_scale().to(dtype=x.dtype, device=x.device)
        return torch.tanh(self.flow_out(self.body(x))) * scale


class ExactGeometryTransportSegmenter(nn.Module):
    """The complete validated M1 and nothing beyond M1.

    Base logits are always the factual anchor.  Historical modes detach them;
    joint JBT modes preserve a deliberately scaled gradient path into Base/PVL.
    Spatial semantics/text condition one residual branch and one learned normal
    flow; no candidate policy, critic, STOP head, router or M2 parameter exists.
    """

    use_semantic_feature = True
    unified_m1_safe_fusion_enabled = True
    mhcs_root_complete = True  # outer-dispatch compatibility only
    mhcs_geometry_topology_refinement = True
    exact_geotr_m1_only = True

    def __init__(self, cfg) -> None:
        super().__init__()
        m1 = _cfg_get(cfg, "M1", None)
        self.hidden_dim = max(32, int(_cfg_get(m1, "MHCS_HIDDEN_DIM", 128)))
        self.semantic_channels = int(_cfg_get(m1, "SEMANTIC_CHANNELS", 512))
        self.text_dim = int(_cfg_get(m1, "MHCS_TEXT_DIM", 512))

        # FORMAL55 ablations are strictly opt-in.  The defaults reproduce the
        # previously validated Full-M1 graph exactly, including operation
        # order.  Keeping the switches here (rather than deleting modules)
        # also preserves checkpoint compatibility across every ablation.
        self.use_semantic_conditioning = bool(
            _cfg_get(m1, "GEOTR_M1_USE_SEMANTIC_CONDITIONING", True)
        )
        self.use_text_conditioning = bool(
            _cfg_get(m1, "GEOTR_M1_USE_TEXT_CONDITIONING", True)
        )
        self.use_anchor_cues = bool(
            _cfg_get(m1, "GEOTR_M1_USE_ANCHOR_CUES", True)
        )
        # Causal-observer contract.  The refiner may observe features produced
        # by a trainable upstream model, but its loss must not update that
        # upstream model through a hidden conditioning path.  Detaching inside
        # the module (rather than relying only on an outer dispatcher) keeps the
        # contract true for every caller, including unit tests and future Base
        # integrations.  This changes gradients only; forward numerics and
        # historical checkpoints are unchanged.
        self.detach_conditioners = bool(
            _cfg_get(m1, "GEOTR_M1_DETACH_CONDITIONERS", True)
        )
        # JBT v1: when enabled, the transport is part of the segmentation
        # computation graph instead of a detached post-hoc observer.  The
        # historical mode is preserved by default for backward compatibility.
        self.joint_base_integration = bool(
            _cfg_get(m1, "GEOTR_M1_JOINT_BASE_INTEGRATION", False)
        )
        # JBT-v2 turns posterior uncertainty into an explicit correction signal
        # and, optionally, refines the decoder feature field *before* the final
        # pixel-text similarity is converted to logits.  All switches are opt-in
        # so historical GEOTR/SemLT checkpoints keep their old numerics.
        self.posterior_uncertainty_enabled = bool(
            _cfg_get(m1, "JBT_POSTERIOR_UNCERTAINTY_ENABLED", False)
        )
        self.posterior_uncertainty_gain = max(0.0, float(
            _cfg_get(m1, "JBT_POSTERIOR_UNCERTAINTY_GAIN", 1.0)
        ))
        self.feature_feedback_enabled = bool(
            _cfg_get(m1, "JBT_FEATURE_FEEDBACK_ENABLED", False)
        )
        self.feature_feedback_max_scale = max(1.0e-4, float(
            _cfg_get(m1, "JBT_FEATURE_FEEDBACK_MAX_SCALE", 0.25)
        ))
        self.feature_feedback_init_scale = float(
            _cfg_get(m1, "JBT_FEATURE_FEEDBACK_INIT_SCALE", 0.10)
        )
        if not 0.0 < self.feature_feedback_init_scale <= self.feature_feedback_max_scale:
            raise ValueError(
                "JBT feature feedback requires 0 < init_scale <= max_scale"
            )
        # JBT-v3: keep the feature-space interpretation but make the *induced
        # segmentation-logit residual* explicitly bounded.  v2 bounded the
        # feature norm only; because mask_head(text) can have a large norm, a
        # tiny 512-D feature perturbation could still shift logits by 4--20.
        self.logit_bounded_feature_feedback = bool(
            _cfg_get(m1, "JBT_LOGIT_BOUNDED_FEATURE_FEEDBACK_ENABLED", False)
        )
        self.feature_feedback_max_logit_delta = max(1.0e-4, float(
            _cfg_get(m1, "JBT_FEATURE_FEEDBACK_MAX_LOGIT_DELTA", 1.25)
        ))
        # Posterior evidence in the v2 code was combined in absolute units.
        # Typical MC std was only ~0.002--0.01, so it barely affected the gate.
        # Robust per-case normalization makes posterior *ranking* meaningful
        # without changing the Base posterior itself.
        self.posterior_robust_quantile = float(
            _cfg_get(m1, "JBT_POSTERIOR_ROBUST_QUANTILE", 0.95)
        )
        self.posterior_robust_quantile = min(max(self.posterior_robust_quantile, 0.50), 0.999)
        # JBT-v3 learns whether a local edit is likely FN-like (expand) or
        # FP-like (contract).  This benefit gate is trained from Base-vs-GT
        # residuals, but uses only predicted scores at inference.
        self.error_gate_enabled = bool(
            _cfg_get(m1, "JBT_ERROR_GATE_ENABLED", False)
        )
        self.error_gate_low = float(_cfg_get(m1, "JBT_ERROR_GATE_LOW", 0.10))
        self.error_gate_high = float(_cfg_get(m1, "JBT_ERROR_GATE_HIGH", 0.40))
        self.error_gate_margin = max(1.0e-4, float(
            _cfg_get(m1, "JBT_ERROR_GATE_MARGIN", 0.10)
        ))
        self.error_prior = min(max(float(
            _cfg_get(m1, "JBT_ERROR_PRIOR", 0.03)
        ), 1.0e-4), 1.0 - 1.0e-4)
        if not 0.0 <= self.error_gate_low < self.error_gate_high <= 1.0:
            raise ValueError("JBT error gate requires 0 <= low < high <= 1")

        # JBT-v4: fixed absolute error-probability cutoffs behaved differently
        # on BUSI and Kvasir.  The dynamic gate ranks predicted FN/FP evidence
        # *within the current Base boundary/uncertainty support*, while a learned
        # case utility head can still choose exact Preserve when no edit is
        # expected to improve overlap.  This is dataset-agnostic: no dataset
        # name or hand-tuned dataset-specific threshold is consulted.
        self.dynamic_error_gate_enabled = bool(
            _cfg_get(m1, "JBT_DYNAMIC_ERROR_GATE_ENABLED", False)
        )
        self.dynamic_error_gate_low_quantile = float(
            _cfg_get(m1, "JBT_DYNAMIC_ERROR_GATE_LOW_QUANTILE", 0.60)
        )
        self.dynamic_error_gate_high_quantile = float(
            _cfg_get(m1, "JBT_DYNAMIC_ERROR_GATE_HIGH_QUANTILE", 0.90)
        )
        self.dynamic_error_gate_low_quantile = min(
            max(self.dynamic_error_gate_low_quantile, 0.0), 0.98
        )
        self.dynamic_error_gate_high_quantile = min(
            max(self.dynamic_error_gate_high_quantile,
                self.dynamic_error_gate_low_quantile + 0.01), 0.999
        )
        self.error_direction_coupling = bool(
            _cfg_get(m1, "JBT_ERROR_DIRECTION_COUPLING", False)
        )
        self.case_utility_gate_enabled = bool(
            _cfg_get(m1, "JBT_CASE_UTILITY_GATE_ENABLED", False)
        )
        self.case_utility_prior = min(max(float(
            _cfg_get(m1, "JBT_CASE_UTILITY_PRIOR", 0.15)
        ), 1.0e-4), 1.0 - 1.0e-4)
        self.case_utility_threshold = float(
            _cfg_get(m1, "JBT_CASE_UTILITY_THRESHOLD", 0.05)
        )

        # JBT-v5: decouple correction *capacity* from correction *execution*.
        # The generator is allowed to search a broad, case-adaptive boundary
        # band and a larger displacement range; a risk-aware utility selector
        # decides whether any generated counterfactual should actually replace
        # Preserve.  No dataset-specific threshold is used.
        self.v5_broad_support_enabled = bool(
            _cfg_get(m1, "JBT_V5_BROAD_SUPPORT_ENABLED", False)
        )
        self.v5_support_radius_min_px = max(1, int(
            _cfg_get(m1, "JBT_V5_SUPPORT_RADIUS_MIN_PX", 2)
        ))
        self.v5_support_radius_max_px = max(
            self.v5_support_radius_min_px, int(
                _cfg_get(m1, "JBT_V5_SUPPORT_RADIUS_MAX_PX", 8)
            )
        )
        self.v5_support_floor = min(max(float(
            _cfg_get(m1, "JBT_V5_SUPPORT_FLOOR", 0.15)
        ), 0.0), 1.0)
        self.v5_direction_magnitude_gain = max(0.1, float(
            _cfg_get(m1, "JBT_V5_DIRECTION_MAGNITUDE_GAIN", 3.0)
        ))
        raw_strengths = _cfg_get(
            m1, "JBT_V5_COUNTERFACTUAL_STRENGTHS", [0.5, 1.0, 1.5]
        )
        if not isinstance(raw_strengths, (list, tuple)) or not raw_strengths:
            raise ValueError("JBT_V5_COUNTERFACTUAL_STRENGTHS must be a non-empty list")
        self.v5_counterfactual_strengths = tuple(float(x) for x in raw_strengths)
        if any(x <= 0.0 for x in self.v5_counterfactual_strengths):
            raise ValueError("JBT-v5 non-Preserve strengths must all be > 0")
        self.v5_multistrength_enabled = bool(
            _cfg_get(m1, "JBT_V5_MULTISTRENGTH_ENABLED", False)
        )
        self.v5_utility_risk_lambda = max(0.0, float(
            _cfg_get(m1, "JBT_V5_UTILITY_RISK_LAMBDA", 1.5)
        ))
        # JBT-v6: the v5 candidate family had a wide *nominal* 8-px search
        # range but the actual learned flow stayed near 0.003--0.005 px because
        # signed motion was multiplicatively bottlenecked by FN-FP probability
        # differences.  v6 restores a directly signed geometry field and uses
        # FN/FP only as evidence/regularization, not as a hard multiplicative
        # direction switch.  Safety is moved downstream into candidate-specific
        # utility/risk selection.
        self.v6_direct_signed_flow = bool(
            _cfg_get(m1, "JBT_V6_DIRECT_SIGNED_FLOW", False)
        )
        self.v6_candidate_utility_enabled = bool(
            _cfg_get(m1, "JBT_V6_CANDIDATE_UTILITY_ENABLED", False)
        )
        self.v62_enabled = bool(_cfg_get(m1, "JBT_V62_ENABLED", False))
        self.v62_support_floor = min(max(float(
            _cfg_get(m1, "JBT_V62_ACTUAL_SUPPORT_FLOOR", 0.75)
        ), 0.0), 1.0)
        self.v62_gain_scale = max(1.0e-4, float(
            _cfg_get(m1, "JBT_V62_GAIN_SCALE", 0.02)
        ))
        self.v62_lcb_lambda = max(0.0, float(
            _cfg_get(m1, "JBT_V62_LCB_LAMBDA", 1.0)
        ))
        self.v62_harm_penalty = max(0.0, float(
            _cfg_get(m1, "JBT_V62_HARM_PROB_PENALTY", 0.25)
        ))
        self.v62_preserve_margin = max(0.0, float(
            _cfg_get(m1, "JBT_V62_PRESERVE_MARGIN", 5.0e-4)
        ))
        # v6.3 replaces the uncalibrated aleatoric LCB used by v6.2 with an
        # expected-utility score.  In the observed BUSI run the predicted sigma
        # (0.0115 Dice) was almost twice the entire Case-Oracle gain (0.0062),
        # which made every edit mathematically impossible even though every
        # fixed-strength cohort improved on average.  The three-way
        # benefit/neutral/harm head is retained and contributes directly to the
        # score; sigma remains an audit output, not a hard deployment veto.
        self.v63_enabled = bool(_cfg_get(m1, "JBT_V63_ENABLED", False))
        self.v63_class_score_weight = max(0.0, float(
            _cfg_get(m1, "JBT_V63_CLASS_SCORE_WEIGHT", 0.25)
        ))
        self.v63_harm_penalty = max(0.0, float(
            _cfg_get(m1, "JBT_V63_HARM_PROB_PENALTY", 0.05)
        ))
        self.v63_accept_margin = float(
            _cfg_get(m1, "JBT_V63_ACCEPT_MARGIN", 0.0)
        )
        # A value >0 is supplied only after it has been selected on Val.  It is
        # the non-collapse fallback when the per-case scorer abstains.  Test
        # labels are never involved in this choice.
        self.v63_val_fallback_strength = float(
            _cfg_get(m1, "JBT_V63_VAL_FALLBACK_STRENGTH", 0.0)
        )
        self.v634_detach_utility_from_generator = bool(
            _cfg_get(m1, "JBT_V634_DETACH_UTILITY_FROM_GENERATOR", False)
        )
        self.v6_utility_descriptor_dim = 8
        # Global context alone cannot tell whether a *specific edited region* is
        # lesion-like or background-like.  v6 therefore pools the fused hidden
        # features over candidate ADD and REMOVE regions separately.
        self.v6_utility_context_multiplier = 3
        self.v6_utility_score_temperature = max(1.0e-3, float(
            _cfg_get(m1, "JBT_V6_UTILITY_SCORE_TEMPERATURE", 1.0)
        ))
        self.normal_smooth_kernel = max(1, int(
            _cfg_get(m1, "GEOTR_M1_NORMAL_SMOOTH_KERNEL", 1)
        ))
        if self.normal_smooth_kernel % 2 == 0:
            self.normal_smooth_kernel += 1
        self.transport_space = str(
            _cfg_get(m1, "GEOTR_M1_TRANSPORT_SPACE", "logit")
        ).strip().lower()
        if self.transport_space not in {"logit", "probability"}:
            raise ValueError(
                "M1.GEOTR_M1_TRANSPORT_SPACE must be 'logit' or 'probability', "
                f"got {self.transport_space!r}"
            )
        self.operator = str(
            _cfg_get(m1, "GEOTR_M1_OPERATOR", "free_2d")
        ).strip().lower()
        if self.operator not in {"free_2d", "normal_1d", "residual_rewrite"}:
            raise ValueError(
                "M1.GEOTR_M1_OPERATOR must be free_2d, normal_1d or "
                f"residual_rewrite, got {self.operator!r}"
            )
        self.gate_mode = str(
            _cfg_get(m1, "GEOTR_M1_GATE_MODE", "none")
        ).strip().lower()
        if self.gate_mode not in {"none", "soft", "deadzone"}:
            raise ValueError(
                "M1.GEOTR_M1_GATE_MODE must be none, soft or deadzone, "
                f"got {self.gate_mode!r}"
            )
        self.gate_low = float(_cfg_get(m1, "GEOTR_M1_GATE_LOW", 0.05))
        self.gate_high = float(_cfg_get(m1, "GEOTR_M1_GATE_HIGH", 0.25))
        self.gate_gamma = float(_cfg_get(m1, "GEOTR_M1_GATE_GAMMA", 1.0))
        self.normal_eps = float(_cfg_get(m1, "GEOTR_M1_NORMAL_EPS", 1.0e-4))
        self.max_residual_logit = float(
            _cfg_get(m1, "GEOTR_M1_MAX_RESIDUAL_LOGIT", 1.0)
        )
        if not 0.0 <= self.gate_low < self.gate_high <= 1.0:
            raise ValueError("LeST gate requires 0 <= low < high <= 1")
        if self.gate_gamma <= 0.0 or self.normal_eps <= 0.0:
            raise ValueError("LeST gate_gamma and normal_eps must be positive")
        if self.max_residual_logit <= 0.0:
            raise ValueError("GEOTR_M1_MAX_RESIDUAL_LOGIT must be positive")

        # The construction order and names match the previously successful M1.
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
        context_gate_init = float(
            _cfg_get(m1, "GEOTR_M1_CONTEXT_GATE_INIT", 0.0)
        )
        if not -0.999 < context_gate_init < 0.999:
            raise ValueError(
                "M1.GEOTR_M1_CONTEXT_GATE_INIT must be in (-0.999, 0.999)"
            )
        self.context_gate = nn.Parameter(
            torch.tensor(math.atanh(context_gate_init))
        )
        self.distribution_trunk = nn.Sequential(
            ConvNormGELU(self.hidden_dim, self.hidden_dim),
            ConvNormGELU(self.hidden_dim, self.hidden_dim),
        )
        self.mean_head = GeometryHead(
            self.hidden_dim,
            init_scale_px=float(_cfg_get(m1, "GEOTOPO_FLOW_INIT_SCALE_PX", 1.0)),
            output_channels=2 if self.operator == "free_2d" else 1,
            max_flow_px=float(_cfg_get(m1, "GEOTR_M1_MAX_FLOW_PX", 0.0)),
        )
        self.factor_head = nn.ModuleDict({"residual_context": nn.Identity()})
        self.diag_std_head = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)
        nn.init.zeros_(self.diag_std_head.weight)
        nn.init.zeros_(self.diag_std_head.bias)
        self.m1_distribution_log_var = nn.Parameter(torch.zeros(()))

        # Decoder-space boundary feedback.  The zero-initialized output keeps
        # the initial function exactly equal to the factual Base.  A bounded,
        # feature-RMS-relative residual avoids destabilising the strong Base
        # representation while still allowing the auxiliary branch to learn
        # faster than a sub-pixel logit warp alone.
        if self.feature_feedback_enabled:
            if self.logit_bounded_feature_feedback:
                # Predict one bounded logit residual.  It is converted back to
                # a text-aligned decoder-feature residual such that
                # <t, deltaF> == delta_logit by construction.
                self.feature_feedback_head = nn.Sequential(
                    ConvNormGELU(self.hidden_dim, self.hidden_dim),
                    nn.Conv2d(self.hidden_dim, 1, 1, bias=True),
                )
            else:
                self.feature_feedback_head = nn.Sequential(
                    ConvNormGELU(self.hidden_dim, self.hidden_dim),
                    nn.Conv2d(self.hidden_dim, self.semantic_channels, 1, bias=True),
                )
            nn.init.zeros_(self.feature_feedback_head[-1].weight)
            nn.init.zeros_(self.feature_feedback_head[-1].bias)
            frac = min(
                max(self.feature_feedback_init_scale / self.feature_feedback_max_scale, 1.0e-4),
                1.0 - 1.0e-4,
            )
            self.feature_feedback_logit_scale = nn.Parameter(
                torch.tensor(math.log(frac / (1.0 - frac)))
            )
        else:
            self.feature_feedback_head = None
            self.register_parameter("feature_feedback_logit_scale", None)

        if self.error_gate_enabled:
            # hidden already contains two convolutional trunk blocks; a 1x1
            # directional head is sufficient here and avoids another expensive
            # 224x224 3x3 block in every train step.
            self.error_head = nn.Sequential(
                nn.Conv2d(self.hidden_dim, 2, 1, bias=True),
            )
            nn.init.zeros_(self.error_head[-1].weight)
            prior_bias = math.log(self.error_prior / (1.0 - self.error_prior))
            nn.init.constant_(self.error_head[-1].bias, prior_bias)
        else:
            self.error_head = None

        if self.case_utility_gate_enabled:
            utility_hidden = max(16, self.hidden_dim // 2)
            if self.v5_multistrength_enabled and self.v6_candidate_utility_enabled:
                # v6 uses the shared candidate-specific scorer below.  Keep no
                # unused trainable v5 case-prior head in the optimizer.
                self.case_utility_head = nn.Identity()
            elif self.v5_multistrength_enabled:
                # For every non-Preserve strength predict three outcomes:
                # [benefit, neutral, harm].  Preserve itself has exact score 0.
                utility_out = 3 * len(self.v5_counterfactual_strengths)
                self.case_utility_head = nn.Sequential(
                    nn.Linear(self.hidden_dim, utility_hidden),
                    nn.GELU(),
                    nn.Linear(utility_hidden, utility_out),
                )
                nn.init.zeros_(self.case_utility_head[-1].weight)
                # Start from a neutral prior so deployment initially preserves
                # rather than editing every case or permanently starving the head.
                prior = torch.tensor([-1.0, 1.0, -1.0]).repeat(
                    len(self.v5_counterfactual_strengths)
                )
                with torch.no_grad():
                    self.case_utility_head[-1].bias.copy_(prior)
            else:
                self.case_utility_head = nn.Sequential(
                    nn.Linear(self.hidden_dim, utility_hidden),
                    nn.GELU(),
                    nn.Linear(utility_hidden, 1),
                )
                nn.init.zeros_(self.case_utility_head[-1].weight)
                utility_bias = math.log(
                    self.case_utility_prior / (1.0 - self.case_utility_prior)
                )
                nn.init.constant_(self.case_utility_head[-1].bias, utility_bias)
        else:
            self.case_utility_head = None

        # v6 scores every counterfactual from its *actual predicted change*.
        # v5 pooled only the pre-edit hidden state and then emitted K fixed
        # outcome triplets, so the selector could learn only a case prior for
        # each strength; it could not inspect whether a particular candidate
        # expanded/contracted the right pixels.  The shared v6 scorer receives
        # pooled semantic context plus eight candidate-specific descriptors.
        if self.case_utility_gate_enabled and self.v6_candidate_utility_enabled:
            utility_hidden = max(32, self.hidden_dim // 2)
            # v6.2 adds a direct gain mean and log-variance to the three
            # outcome logits.  The deployed score is a lower confidence bound
            # in physical Dice-gain units; Preserve remains exactly zero.
            utility_out = 5 if self.v62_enabled else 3
            self.v6_candidate_utility_head = nn.Sequential(
                nn.Linear(
                    self.v6_utility_context_multiplier * self.hidden_dim
                    + self.v6_utility_descriptor_dim,
                    utility_hidden,
                ),
                nn.GELU(),
                nn.Linear(utility_hidden, utility_hidden),
                nn.GELU(),
                nn.Linear(utility_hidden, utility_out),
            )
            nn.init.zeros_(self.v6_candidate_utility_head[-1].weight)
            with torch.no_grad():
                prior = torch.tensor([-1.0, 1.0, -1.0, 0.0, -2.0])
                self.v6_candidate_utility_head[-1].bias.copy_(prior[:utility_out])
        else:
            self.v6_candidate_utility_head = None

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

    def _normal_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Stable smoothed central-difference x/y normal; flat regions map to zero."""
        work = logits.float()
        if self.normal_smooth_kernel > 1:
            k = int(self.normal_smooth_kernel)
            work = F.avg_pool2d(work, kernel_size=k, stride=1, padding=k // 2)
        padded = F.pad(work, (1, 1, 1, 1), mode="replicate")
        grad_x = 0.5 * (padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2])
        grad_y = 0.5 * (padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1])
        norm = torch.sqrt(
            grad_x.square() + grad_y.square() + float(self.normal_eps) ** 2
        )
        return torch.cat([grad_x / norm, grad_y / norm], dim=1).to(logits)

    def _robust_unit_map(self, value: torch.Tensor) -> torch.Tensor:
        """Per-case robust [0,1] normalization for sparse MC evidence."""
        flat = value.detach().float().flatten(1)
        scale = torch.quantile(
            flat, self.posterior_robust_quantile, dim=1, keepdim=True
        ).clamp_min(1.0e-4)
        scale = scale[:, :, None, None]
        return (value.detach().float() / scale).clamp(0.0, 1.0).to(value)

    def _dynamic_support_gate(
        self, score: torch.Tensor, support: torch.Tensor
    ) -> torch.Tensor:
        """Per-case support-relative calibration for sparse error scores.

        Quantiles are computed only where the pre-error Base evidence gate is
        active, so lesion size / boundary length does not alter the meaning of
        a global whole-image percentile.  Cases with no support stay Preserve.
        """
        out = torch.zeros_like(score)
        score_det = score.detach()
        support_det = support.detach()
        for i in range(score.shape[0]):
            active = support_det[i, 0] > 1.0e-6
            values = score_det[i, 0][active]
            if values.numel() < 8:
                continue
            q_low = torch.quantile(values.float(), self.dynamic_error_gate_low_quantile).to(score)
            q_high = torch.quantile(values.float(), self.dynamic_error_gate_high_quantile).to(score)
            span = q_high - q_low
            if float(span.detach().abs().cpu()) < 1.0e-4:
                # Uniform-but-confident predictions occur while the error head
                # is still mostly bias-driven. A pure quantile gate would be
                # identically zero and starve feature/flow learning. Preserve
                # zero when the score itself is zero, otherwise fall back to an
                # absolute confidence normalization inside the Base support.
                scale = q_high.abs().clamp_min(1.0e-4)
                rel = (score[i:i+1] / scale).clamp(0.0, 1.0)
            else:
                rel = ((score[i:i+1] - q_low) / span).clamp(0.0, 1.0)
            out[i:i+1] = rel * (support[i:i+1] > 1.0e-6).to(rel)
        return out

    def _v5_broad_adaptive_support(
        self,
        probability: torch.Tensor,
        evidence: torch.Tensor,
        posterior_evidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """High-recall, case-adaptive correction search band.

        Safety is deliberately *not* enforced by making this band tiny.  The
        band only defines where counterfactual corrections may be proposed; the
        outcome selector later decides whether they are executed.  Radius is
        inferred per case from posterior difficulty around the Base boundary.
        """
        boundary = self._boundary(probability).detach()
        support = torch.zeros_like(boundary)
        radii = boundary.new_zeros(boundary.shape[0])
        r0, r1 = self.v5_support_radius_min_px, self.v5_support_radius_max_px
        for i in range(boundary.shape[0]):
            b = boundary[i:i+1]
            post = posterior_evidence[i:i+1].detach()
            local = evidence[i:i+1].detach()
            active = b > 0.05
            if bool(active.any()):
                # Posterior evidence is robust-normalized per case.  A hard case
                # receives a wider search radius, while a clean case remains local.
                denom = active.float().sum().clamp_min(1.0)
                severity = ((0.65 * post + 0.35 * local) * active).sum() / denom
                severity = severity.clamp(0.0, 1.0)
            else:
                severity = b.new_zeros(())
            radius = int(round(r0 + float(severity.cpu()) * (r1 - r0)))
            radius = min(max(radius, r0), r1)
            radii[i] = float(radius)
            band = F.max_pool2d(
                (b > 0.05).to(b), kernel_size=2 * radius + 1,
                stride=1, padding=radius,
            )
            # Keep gradients/evidence alive across the broad band.  The floor
            # prevents the old 95% dead-zone collapse while evidence still
            # concentrates the strongest corrections on difficult locations.
            weight = self.v5_support_floor + (1.0 - self.v5_support_floor) * local
            support[i:i+1] = band * weight
        return support, radii

    def _change_gate(
        self,
        probability: torch.Tensor,
        mc_std_map: Optional[torch.Tensor] = None,
        mc_disagreement_map: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local_evidence = torch.maximum(
            self._boundary(probability), self._uncertainty(probability)
        ).detach()
        posterior_evidence = torch.zeros_like(local_evidence)
        if self.posterior_uncertainty_enabled:
            if isinstance(mc_std_map, torch.Tensor):
                std = mc_std_map
                if std.ndim == 3:
                    std = std[:, None]
                std = self._resize(std, probability.shape[-2:]).detach()
                posterior_evidence = torch.maximum(
                    posterior_evidence, self._robust_unit_map(std)
                )
            if isinstance(mc_disagreement_map, torch.Tensor):
                dis = mc_disagreement_map
                if dis.ndim == 3:
                    dis = dis[:, None]
                dis = self._resize(dis, probability.shape[-2:]).detach()
                posterior_evidence = torch.maximum(
                    posterior_evidence, self._robust_unit_map(dis)
                )
        if self.posterior_uncertainty_gain > 0.0:
            evidence = torch.maximum(
                local_evidence,
                (self.posterior_uncertainty_gain * posterior_evidence).clamp(0.0, 1.0),
            )
        else:
            evidence = local_evidence
        if self.gate_mode == "none":
            gate = torch.ones_like(evidence)
        elif self.gate_mode == "soft":
            gate = evidence.pow(float(self.gate_gamma))
        else:
            gate = (
                (evidence - float(self.gate_low))
                / float(self.gate_high - self.gate_low)
            ).clamp(0.0, 1.0).pow(float(self.gate_gamma))
        return gate, evidence, posterior_evidence

    @staticmethod
    def _warp_logits(base_logits: torch.Tensor, flow_px: torch.Tensor) -> torch.Tensor:
        _, _, h, w = base_logits.shape
        dtype, device = base_logits.dtype, base_logits.device
        yy = torch.arange(h, device=device, dtype=dtype) + 0.5
        xx = torch.arange(w, device=device, dtype=dtype) + 0.5
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        x = gx[None] + flow_px[:, 0]
        y = gy[None] + flow_px[:, 1]
        grid = torch.stack(
            [2.0 * x / float(w) - 1.0, 2.0 * y / float(h) - 1.0], dim=-1
        )
        return F.grid_sample(
            base_logits,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

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

    def _transport_features(
        self,
        image: torch.Tensor,
        semantic_map: Optional[torch.Tensor],
        text_features: Optional[torch.Tensor],
        anchor_prob: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        hw = tuple(anchor_prob.shape[-2:])
        image_latent = self.image_stem(self._resize(image, hw))
        if self.use_semantic_conditioning:
            if semantic_map is None:
                raise RuntimeError(
                    "GEOTR-M1 semantic conditioning requires semantic_map"
                )
            semantic_latent = self.semantic_proj(self._resize(semantic_map, hw))
        else:
            semantic_latent = image_latent.new_zeros(image_latent.shape)

        if self.use_text_conditioning:
            if text_features is None:
                raise RuntimeError(
                    "GEOTR-M1 text conditioning requires text_features"
                )
            text = self.text_proj(text_features.float())
        else:
            text = image_latent.new_zeros(
                (image_latent.shape[0], self.hidden_dim)
            )
        p = (
            anchor_prob
            if self.joint_base_integration
            else anchor_prob.detach()
        )
        if self.use_anchor_cues:
            anchor_channels = (p, self._uncertainty(p), self._boundary(p))
        else:
            anchor_channels = tuple(torch.zeros_like(p) for _ in range(3))
        pixel = self.pixel_fuse(
            torch.cat(
                [image_latent, semantic_latent, *anchor_channels],
                dim=1,
            )
        )
        global_visual = F.adaptive_avg_pool2d(pixel, 1).flatten(1)
        context = self.global_context(torch.cat([global_visual, text], dim=1))
        gamma, beta = self.context_film(context).chunk(2, dim=1)
        gate = torch.tanh(self.context_gate)
        modulation = gate * (
            torch.tanh(gamma)[:, :, None, None] * pixel + beta[:, :, None, None]
        )
        pixel = pixel + modulation
        hidden = self.distribution_trunk(pixel)
        stats = {
            # FiLM is driven by pooled visual context and optional text, so it
            # must not be mislabeled as a text-only contribution.
            "context_film_abs": modulation.detach().abs().mean(dim=(1, 2, 3)),
            "text_latent_abs": text.detach().abs().mean(dim=1),
            "anchor_boundary_fraction": self._boundary(p).detach().mean(
                dim=(1, 2, 3)
            ),
        }
        return hidden + self.diag_std_head(hidden), stats

    def generate(
        self,
        base_logits: torch.Tensor,
        image: torch.Tensor,
        semantic_map: Optional[torch.Tensor],
        text_features: Optional[torch.Tensor],
        negative_text_features: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del negative_text_features
        mc_std_map = kwargs.pop("mc_std_map", None)
        mc_disagreement_map = kwargs.pop("mc_disagreement_map", None)
        fine_feature_map = kwargs.pop("fine_feature_map", None)
        seg_text_vector = kwargs.pop("seg_text_vector", None)
        # Ignore unrelated historical compatibility kwargs, but never silently
        # discard the JBT evidence tensors above.
        if base_logits.ndim == 3:
            base_logits = base_logits[:, None]
        if base_logits.ndim != 4 or base_logits.shape[1] != 1:
            raise ValueError(f"Expected Base logits [B,1,H,W], got {tuple(base_logits.shape)}")

        factual_logits = (
            base_logits
            if self.joint_base_integration
            else base_logits.detach()
        )
        base_prob = torch.sigmoid(factual_logits).clamp(EPS, 1.0 - EPS)
        observed_semantic = semantic_map
        observed_text = text_features
        if self.detach_conditioners and not self.joint_base_integration:
            if isinstance(observed_semantic, torch.Tensor):
                observed_semantic = observed_semantic.detach()
            if isinstance(observed_text, torch.Tensor):
                observed_text = observed_text.detach()
        hidden, conditioning_stats = self._transport_features(
            image, observed_semantic, observed_text, base_prob
        )
        gate, evidence_band, posterior_evidence = self._change_gate(
            base_prob,
            mc_std_map=mc_std_map,
            mc_disagreement_map=mc_disagreement_map,
        )
        v5_support_radius_px = factual_logits.new_zeros(factual_logits.shape[0])
        if self.v5_broad_support_enabled:
            gate, v5_support_radius_px = self._v5_broad_adaptive_support(
                base_prob, evidence_band, posterior_evidence
            )
        v5_proposal_gate = gate

        # JBT-v3/v5 benefit/error gate.  Uncertainty says *where the Base is unsure*;
        # it does not say whether moving that pixel will help.  The learned FN/FP
        # head supplies that missing utility signal and makes intervention sparse
        # on already-correct boundaries.
        error_logits = factual_logits.new_zeros(
            factual_logits.shape[0], 2, *factual_logits.shape[-2:]
        )
        error_probs = torch.zeros_like(error_logits)
        error_gate = torch.ones_like(gate)
        error_direction = torch.zeros_like(gate)
        if self.error_gate_enabled:
            error_logits = _jbt_memory_safe_head_forward(self.error_head, hidden)
            error_probs = torch.sigmoid(error_logits)
            fn_prob = error_probs[:, 0:1]
            fp_prob = error_probs[:, 1:2]
            error_conf = torch.maximum(fn_prob, fp_prob)
            error_direction = fn_prob - fp_prob
            margin_gate = (
                error_direction.abs() / float(self.error_gate_margin)
            ).clamp(0.0, 1.0)
            if self.dynamic_error_gate_enabled:
                # Rank predicted error evidence only inside the Base support.
                # The square-root confidence factor retains an absolute notion
                # of certainty, preventing pure rank normalization from forcing
                # edits in a genuinely clean case.
                raw_error_score = error_conf * margin_gate
                relative_gate = self._dynamic_support_gate(raw_error_score, gate)
                if self.v5_broad_support_enabled:
                    # v4 multiplied rank evidence by absolute error probability;
                    # as the error classifier calibrated downward this collapsed
                    # the support from ~40% to ~4%.  v5 lets the *relative* error
                    # evidence rank proposals, while the downstream risk selector
                    # owns execution safety.
                    error_gate = relative_gate
                else:
                    error_gate = relative_gate * error_conf.clamp_min(0.0).sqrt()
            else:
                confidence_gate = (
                    (error_conf - float(self.error_gate_low))
                    / float(self.error_gate_high - self.error_gate_low)
                ).clamp(0.0, 1.0)
                error_gate = confidence_gate * margin_gate
            gate = gate * error_gate

        # v6 capacity/safety separation: the learned error gate is evidence for
        # utility/risk, not the hard owner of geometric reach.  Candidate
        # generation uses the broad adaptive support so recall can grow without
        # forcing deployment; the downstream Preserve/counterfactual selector
        # still owns safety.  This prevents the v5 7--11% effective-support
        # bottleneck from nullifying an otherwise 2--8 px proposal band.
        geometry_gate = (
            v5_proposal_gate
            if (self.v6_direct_signed_flow and self.v5_broad_support_enabled)
            else gate
        )
        if self.v62_enabled and self.v6_direct_signed_flow:
            # The support decides *where* editing is legal.  It must not shrink
            # a supervised 2--3 px displacement to a tenth of a pixel through
            # another confidence product.  Retain soft localization while
            # guaranteeing a well-conditioned amplitude on active support.
            active_support = geometry_gate > 0.0
            geometry_gate = torch.where(
                active_support,
                self.v62_support_floor
                + (1.0 - self.v62_support_floor) * geometry_gate.clamp(0.0, 1.0),
                torch.zeros_like(geometry_gate),
            )

        # Feature-space boundary feedback is applied before geometry, but JBT-v3
        # adds only the *delta induced by the feature residual* to factual logits.
        # This is crucial under MC mean-then-refine: E[z] and logit(E[sigmoid(z)])
        # are not equal, so replacing the factual MC-mean logit with a logit
        # recomputed from mean features breaks the zero-residual identity.
        transport_anchor_logits = factual_logits
        feature_feedback_abs = factual_logits.new_zeros(factual_logits.shape[0])
        feature_feedback_logit_change = factual_logits.new_zeros(factual_logits.shape[0])
        feature_feedback_logit_change_max = factual_logits.new_zeros(factual_logits.shape[0])
        feature_feedback_scale = factual_logits.new_zeros(factual_logits.shape[0])
        posterior_reconstruction_mismatch = factual_logits.new_zeros(factual_logits.shape[0])
        posterior_reconstruction_prob_mismatch = factual_logits.new_zeros(factual_logits.shape[0])
        if self.feature_feedback_enabled:
            if not isinstance(fine_feature_map, torch.Tensor):
                raise RuntimeError(
                    "JBT_FEATURE_FEEDBACK_ENABLED requires fine_feature_map from the Base decoder"
                )
            if not isinstance(seg_text_vector, torch.Tensor):
                raise RuntimeError(
                    "JBT_FEATURE_FEEDBACK_ENABLED requires seg_text_vector from mask_head(text_features)"
                )
            if fine_feature_map.ndim != 4 or fine_feature_map.shape[1] != self.semantic_channels:
                raise ValueError(
                    "fine_feature_map must be [B,C,H,W] with C="
                    f"{self.semantic_channels}, got {tuple(fine_feature_map.shape)}"
                )
            if seg_text_vector.ndim != 2 or seg_text_vector.shape[1] != self.semantic_channels:
                raise ValueError(
                    "seg_text_vector must be [B,C] with C="
                    f"{self.semantic_channels}, got {tuple(seg_text_vector.shape)}"
                )
            low_hidden = self._resize(hidden, fine_feature_map.shape[-2:])
            low_gate = self._resize(gate, fine_feature_map.shape[-2:])
            reference_low_logits = torch.einsum(
                "bc,bchw->bhw", seg_text_vector, fine_feature_map
            )[:, None]
            reference_logits = F.interpolate(
                reference_low_logits, size=factual_logits.shape[-2:],
                mode="bilinear", align_corners=False,
            )
            posterior_reconstruction_mismatch = (
                reference_logits.detach() - factual_logits.detach()
            ).abs().mean(dim=(1, 2, 3))
            posterior_reconstruction_prob_mismatch = (
                torch.sigmoid(reference_logits.detach()) - base_prob.detach()
            ).abs().mean(dim=(1, 2, 3))

            if self.logit_bounded_feature_feedback:
                raw_delta = torch.tanh(_jbt_memory_safe_head_forward(self.feature_feedback_head, low_hidden))
                bounded_logit_scale = (
                    float(self.feature_feedback_max_logit_delta)
                    * torch.sigmoid(self.feature_feedback_logit_scale)
                ).to(fine_feature_map)
                delta_low = raw_delta * low_gate * bounded_logit_scale
                text_norm_sq = seg_text_vector.float().square().sum(
                    dim=1, keepdim=True
                ).clamp_min(1.0e-6).to(fine_feature_map)
                text_direction = seg_text_vector[:, :, None, None] / text_norm_sq[:, :, None, None]
                feedback = text_direction * delta_low
                refined_feature_map = fine_feature_map + feedback
                refined_low_logits = torch.einsum(
                    "bc,bchw->bhw", seg_text_vector, refined_feature_map
                )[:, None]
                feature_delta_low = refined_low_logits - reference_low_logits
                feature_delta_logits = F.interpolate(
                    feature_delta_low, size=factual_logits.shape[-2:],
                    mode="bilinear", align_corners=False,
                )
                # Exact identity anchor: only the controlled delta is added to
                # factual logits, never the mean-feature reconstruction itself.
                transport_anchor_logits = factual_logits + feature_delta_logits
                scale = bounded_logit_scale
            else:
                raw_feedback = torch.tanh(_jbt_memory_safe_head_forward(self.feature_feedback_head, low_hidden))
                feature_rms = (
                    fine_feature_map.detach().float().square().mean(dim=(1, 2, 3), keepdim=True)
                    .sqrt().clamp_min(1.0e-4).to(fine_feature_map)
                )
                scale = (
                    self.feature_feedback_max_scale
                    * torch.sigmoid(self.feature_feedback_logit_scale)
                ).to(fine_feature_map)
                feedback = raw_feedback * low_gate * feature_rms * scale
                refined_feature_map = fine_feature_map + feedback
                refined_low_logits = torch.einsum(
                    "bc,bchw->bhw", seg_text_vector, refined_feature_map
                )[:, None]
                feature_delta_logits = F.interpolate(
                    refined_low_logits - reference_low_logits,
                    size=factual_logits.shape[-2:],
                    mode="bilinear", align_corners=False,
                )
                transport_anchor_logits = factual_logits + feature_delta_logits

            feature_feedback_abs = feedback.detach().abs().mean(dim=(1, 2, 3))
            feature_feedback_logit_change = feature_delta_logits.detach().abs().mean(dim=(1, 2, 3))
            feature_feedback_logit_change_max = feature_delta_logits.detach().abs().flatten(1).max(dim=1).values
            if isinstance(scale, torch.Tensor) and scale.ndim == 0:
                feature_feedback_scale = scale.detach().expand(factual_logits.shape[0])
            else:
                feature_feedback_scale = scale.detach().reshape(-1).mean().expand(factual_logits.shape[0])

        raw_field = _jbt_memory_safe_head_forward(self.mean_head, hidden)
        normal = self._normal_from_logits(transport_anchor_logits)
        scalar_field = raw_field[:, :1]
        direction = torch.tanh(
            error_direction / float(self.error_gate_margin)
        ) if self.error_gate_enabled else torch.zeros_like(scalar_field)
        if self.error_gate_enabled and self.error_direction_coupling:
            if self.v6_direct_signed_flow:
                # v6 capacity fix: GeometryHead already predicts a bounded
                # signed displacement with tanh and has non-zero derivative at
                # zero.  Do not multiply it by the often tiny FN-FP margin.
                # Error direction remains a supervised diagnostic/prior and the
                # proposal gate still focuses the editable band.
                scalar_field = raw_field[:, :1]
            elif self.v5_multistrength_enabled:
                scale = self.mean_head.current_scale().to(raw_field)
                normalized_raw = raw_field[:, :1] / scale.clamp_min(1.0e-6)
                magnitude = scale * torch.sigmoid(
                    self.v5_direction_magnitude_gain * normalized_raw
                )
                scalar_field = magnitude * direction
            else:
                scalar_field = raw_field[:, :1].abs() * direction
        rewrite_residual = torch.zeros_like(factual_logits)
        if self.operator == "free_2d":
            flow_px = raw_field * geometry_gate
            regularized_field = flow_px
        elif self.operator == "normal_1d":
            flow_px = (scalar_field * geometry_gate) * normal
            regularized_field = scalar_field * geometry_gate
        else:
            flow_px = torch.zeros(
                factual_logits.shape[0], 2, *factual_logits.shape[-2:],
                dtype=factual_logits.dtype, device=factual_logits.device,
            )
            regularized_field = scalar_field * geometry_gate
            rewrite_residual = regularized_field * float(self.max_residual_logit)

        def _apply_strength(alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
            alpha_t = float(alpha)
            anchor_delta = alpha_t * (transport_anchor_logits - factual_logits)
            if self.logit_bounded_feature_feedback:
                anchor_delta = anchor_delta.clamp(
                    -float(self.feature_feedback_max_logit_delta),
                    float(self.feature_feedback_max_logit_delta),
                )
            anchor_logits = factual_logits + anchor_delta
            scaled_flow = alpha_t * flow_px
            # Strength explores under/nominal/over-correction but the declared
            # physical displacement cap remains absolute.  Thus strength=1.5
            # cannot silently turn an 8-px contract into a 12-px deformation.
            physical_cap = float(self.mean_head.max_flow_px)
            if physical_cap > 0.0:
                mag = torch.sqrt(scaled_flow[:, :1].square() + scaled_flow[:, 1:2].square() + 1.0e-12)
                factor = (physical_cap / mag.clamp_min(1.0e-6)).clamp(max=1.0)
                scaled_flow = scaled_flow * factor
            if self.operator == "residual_rewrite":
                out_logits = anchor_logits + alpha_t * rewrite_residual
                out_prob = torch.sigmoid(out_logits).clamp(EPS, 1.0 - EPS)
            elif self.transport_space == "logit":
                out_logits = self._warp_logits(anchor_logits, scaled_flow)
                out_prob = torch.sigmoid(out_logits).clamp(EPS, 1.0 - EPS)
            else:
                anchor_prob = torch.sigmoid(anchor_logits).clamp(EPS, 1.0 - EPS)
                out_prob = self._warp_logits(anchor_prob, scaled_flow).clamp(EPS, 1.0 - EPS)
                out_logits = torch.logit(out_prob, eps=EPS)
            # Exact Preserve outside the proposal support.  Broad-support mode
            # uses a soft nonzero band, so only truly outside-band pixels reset.
            inactive = geometry_gate <= 0.0
            out_logits = torch.where(inactive, factual_logits, out_logits)
            out_prob = torch.where(inactive, base_prob, out_prob)
            return out_logits, out_prob

        # One shared field produces several counterfactual strengths.  This
        # raises candidate recall/Oracle ceiling without adding separate editor
        # networks or dataset-specific radii.
        strength_logits = []
        strength_probs = []
        for alpha in self.v5_counterfactual_strengths if self.v5_multistrength_enabled else (1.0,):
            sl, sp = _apply_strength(alpha)
            strength_logits.append(sl)
            strength_probs.append(sp)
        strength_logits_t = torch.cat(strength_logits, dim=1)
        strength_probs_t = torch.cat(strength_probs, dim=1)
        # Nominal raw candidate is the strength closest to 1.0 and remains the
        # training-time main segmentation branch for backwards compatibility.
        nominal_index = min(
            range(len(strength_logits)),
            key=lambda j: abs((self.v5_counterfactual_strengths if self.v5_multistrength_enabled else (1.0,))[j] - 1.0),
        )
        raw_geo_logits = strength_logits_t[:, nominal_index:nominal_index+1]
        raw_geo_prob = strength_probs_t[:, nominal_index:nominal_index+1]
        geo_logits = raw_geo_logits
        geo_prob = raw_geo_prob

        case_utility_logits = factual_logits.new_zeros(factual_logits.shape[0])
        case_utility_probs = factual_logits.new_ones(factual_logits.shape[0])
        case_accept_hard = factual_logits.new_ones(factual_logits.shape[0])
        v63_selector_accept = factual_logits.new_ones(factual_logits.shape[0])
        v63_fallback_used = factual_logits.new_zeros(factual_logits.shape[0])
        selected_strength_index = torch.full(
            (factual_logits.shape[0],), nominal_index + 1,
            dtype=torch.long, device=factual_logits.device,
        )
        selected_strength = factual_logits.new_full(
            (factual_logits.shape[0],),
            float((self.v5_counterfactual_strengths if self.v5_multistrength_enabled else (1.0,))[nominal_index]),
        )
        v5_utility_outcome_probs = factual_logits.new_zeros(
            factual_logits.shape[0], len(strength_logits), 3
        )
        v5_utility_scores = factual_logits.new_zeros(
            factual_logits.shape[0], len(strength_logits)
        )
        v62_gain_mean = factual_logits.new_zeros(
            factual_logits.shape[0], len(strength_logits)
        )
        v62_gain_sigma = factual_logits.new_zeros(
            factual_logits.shape[0], len(strength_logits)
        )
        v62_lcb_scores = factual_logits.new_zeros(
            factual_logits.shape[0], len(strength_logits)
        )
        if self.case_utility_gate_enabled:
            pooled_hidden = F.adaptive_avg_pool2d(hidden, 1).flatten(1)
            raw_utility = self.case_utility_head(pooled_hidden)
            if self.v5_multistrength_enabled:
                if self.v6_candidate_utility_enabled:
                    # Candidate-specific descriptors.  The descriptors are
                    # detached so the utility loss cannot game the editor; the
                    # candidate generator is optimized by segmentation, direct
                    # displacement and oracle-capacity objectives instead.
                    cand = strength_probs_t
                    base_rep = base_prob.expand(-1, cand.shape[1], -1, -1)
                    delta = (cand - base_rep).detach()
                    abs_delta = delta.abs()
                    pos = F.relu(delta)
                    neg = F.relu(-delta)
                    post = posterior_evidence.detach().expand_as(cand)
                    band = evidence_band.detach().expand_as(cand)
                    fn = error_probs[:, 0:1].detach().expand_as(cand)
                    fp = error_probs[:, 1:2].detach().expand_as(cand)
                    descriptor = torch.stack([
                        abs_delta.mean(dim=(2,3)),
                        delta.mean(dim=(2,3)),
                        abs_delta.square().mean(dim=(2,3)).sqrt(),
                        (abs_delta * post).mean(dim=(2,3)),
                        (abs_delta * band).mean(dim=(2,3)),
                        (pos * fn + neg * fp).mean(dim=(2,3)),
                        (pos * fp + neg * fn).mean(dim=(2,3)),
                        ((cand > 0.5).float() - (base_rep > 0.5).float()).abs().mean(dim=(2,3)),
                    ], dim=-1)
                    hidden_det = hidden.detach()
                    pooled_global = F.adaptive_avg_pool2d(hidden_det, 1).flatten(1)
                    pooled_global = pooled_global[:, None, :].expand(-1, cand.shape[1], -1)

                    # Candidate-local semantic evidence. Positive delta is an
                    # expansion/add proposal; negative delta is a contraction/remove
                    # proposal.  Weighted pooling makes the selector inspect what
                    # each candidate actually changes rather than learning only a
                    # per-case/per-strength prior.
                    def _weighted_hidden_pool(weight):
                        weight = weight.detach().clamp_min(0.0)
                        numerator = torch.einsum(
                            "bkhw,bchw->bkc", weight, hidden_det
                        )
                        denominator = weight.flatten(2).sum(dim=2, keepdim=True).clamp_min(1.0e-6)
                        return numerator / denominator

                    pooled_add = _weighted_hidden_pool(pos)
                    pooled_remove = _weighted_hidden_pool(neg)
                    utility_input = torch.cat(
                        [pooled_global, pooled_add, pooled_remove, descriptor], dim=-1
                    )
                    # v6.3.4 proposal/selector ownership contract.  The BUSI
                    # diagnostic had utility-gradient norm 2.32 versus flow
                    # 0.16.  Utility supervision should train the selector, not
                    # rewrite the shared proposal representation and reverse
                    # the displacement field.  Candidate descriptors remain
                    # factual inputs, but selector gradients stop here.
                    if self.v634_detach_utility_from_generator:
                        utility_input = utility_input.detach()
                    utility_raw = self.v6_candidate_utility_head(utility_input)
                    utility_logits3 = utility_raw[..., :3]
                    if self.v62_enabled:
                        v62_gain_mean = self.v62_gain_scale * torch.tanh(
                            utility_raw[..., 3]
                        )
                        # Softplus gives a strictly positive, trainable scale.
                        # The -2 bias starts cautiously without making every
                        # non-Preserve action impossible to select forever.
                        v62_gain_sigma = self.v62_gain_scale * (
                            F.softplus(utility_raw[..., 4]) + 1.0e-3
                        )
                else:
                    utility_logits3 = raw_utility.view(
                        factual_logits.shape[0], len(strength_logits), 3
                    )
                outcome_probs = torch.softmax(utility_logits3 / self.v6_utility_score_temperature, dim=-1)
                benefit = outcome_probs[..., 0]
                harm = outcome_probs[..., 2]
                if self.v62_enabled:
                    if self.v63_enabled:
                        utility_scores = (
                            v62_gain_mean
                            + self.v63_class_score_weight * self.v62_gain_scale
                            * (benefit - harm)
                            - self.v63_harm_penalty * self.v62_gain_scale * harm
                        )
                    else:
                        utility_scores = (
                            v62_gain_mean
                            - self.v62_lcb_lambda * v62_gain_sigma
                            - self.v62_harm_penalty * self.v62_gain_scale * harm
                        )
                    v62_lcb_scores = utility_scores
                else:
                    utility_scores = benefit - self.v5_utility_risk_lambda * harm
                case_utility_logits = (
                    utility_raw if (self.v62_enabled and self.v6_candidate_utility_enabled)
                    else utility_logits3
                )
                case_utility_probs = benefit
                v5_utility_outcome_probs = outcome_probs
                v5_utility_scores = utility_scores
                if not self.training:
                    best_score, best_k = utility_scores.max(dim=1)
                    threshold = (
                        self.v63_accept_margin if self.v63_enabled
                        else self.v62_preserve_margin if self.v62_enabled
                        else float(self.case_utility_threshold)
                    )
                    accept = best_score > threshold
                    deploy = accept
                    deployed_k = best_k
                    fallback_used = torch.zeros_like(accept)
                    if self.v63_enabled and self.v63_val_fallback_strength > 0.0:
                        strengths_tensor = factual_logits.new_tensor(
                            self.v5_counterfactual_strengths
                        )
                        fallback_k_scalar = int(torch.argmin(
                            (strengths_tensor - self.v63_val_fallback_strength).abs()
                        ).item())
                        fallback_k = torch.full_like(best_k, fallback_k_scalar)
                        fallback_used = ~accept
                        deploy = torch.ones_like(accept)
                        deployed_k = torch.where(accept, best_k, fallback_k)
                    case_accept_hard = deploy.to(factual_logits)
                    selected_strength_index = torch.where(
                        deploy, deployed_k + 1, torch.zeros_like(best_k)
                    )
                    gather = deployed_k[:, None, None, None].expand(
                        -1, 1, factual_logits.shape[-2], factual_logits.shape[-1]
                    )
                    best_logits = strength_logits_t.gather(1, gather)
                    best_probs = strength_probs_t.gather(1, gather)
                    geo_logits = torch.where(
                        deploy[:, None, None, None], best_logits, factual_logits
                    )
                    geo_prob = torch.where(
                        deploy[:, None, None, None], best_probs, base_prob
                    )
                    strengths_tensor = factual_logits.new_tensor(
                        self.v5_counterfactual_strengths
                    )
                    selected_strength = torch.where(
                        deploy, strengths_tensor[deployed_k],
                        torch.zeros_like(best_score),
                    )
                    v63_selector_accept = accept.to(factual_logits)
                    v63_fallback_used = fallback_used.to(factual_logits)
            else:
                case_utility_logits = raw_utility.squeeze(1)
                case_utility_probs = torch.sigmoid(case_utility_logits)
                if not self.training:
                    case_accept_hard = (
                        case_utility_probs >= float(self.case_utility_threshold)
                    ).to(factual_logits)
                    accept4 = case_accept_hard[:, None, None, None].bool()
                    geo_logits = torch.where(accept4, raw_geo_logits, factual_logits)
                    geo_prob = torch.where(accept4, raw_geo_prob, base_prob)

        candidate_logits = torch.cat([factual_logits, geo_logits], dim=1)
        candidate_probs = torch.cat([base_prob, geo_prob], dim=1)
        flow_mag = torch.sqrt(flow_px[:, 0].square() + flow_px[:, 1].square() + 1.0e-12)
        jac_mean, folding = self._flow_jacobian_stats(flow_px)
        change = (geo_prob - base_prob).abs().mean(dim=(1, 2, 3))
        projected_normal = (flow_px * normal).sum(dim=1, keepdim=True) * normal
        tangent = flow_px - projected_normal
        tangent_energy = tangent.square().flatten(1).sum(dim=1)
        total_flow_energy = flow_px.square().flatten(1).sum(dim=1)
        tangent_energy_ratio = torch.where(
            total_flow_energy > 1.0e-12,
            tangent_energy / total_flow_energy.clamp_min(1.0e-12),
            torch.zeros_like(total_flow_energy),
        )
        deadzone = geometry_gate == 0.0
        deadzone_count = deadzone.flatten(1).sum(dim=1)
        identity_error = (
            (geo_logits - factual_logits).abs() * deadzone
        ).flatten(1).sum(dim=1) / deadzone_count.clamp_min(1)
        identity_error = torch.where(
            deadzone_count > 0, identity_error, torch.zeros_like(identity_error)
        )
        range_anchor_delta = selected_strength[:, None, None, None] * (
            transport_anchor_logits - factual_logits
        )
        if self.logit_bounded_feature_feedback:
            range_anchor_delta = range_anchor_delta.clamp(
                -float(self.feature_feedback_max_logit_delta),
                float(self.feature_feedback_max_logit_delta),
            )
        range_anchor_logits = factual_logits + range_anchor_delta
        base_min = range_anchor_logits.flatten(1).min(dim=1).values[:, None, None, None]
        base_max = range_anchor_logits.flatten(1).max(dim=1).values[:, None, None, None]
        range_overshoot = F.relu(geo_logits - base_max) + F.relu(base_min - geo_logits)
        # Dead-zone pixels are explicitly restored to factual Base and therefore
        # need not lie inside the feature-anchor range.  Range preservation is a
        # property of the active bilinear transport only.
        active = (geometry_gate > 0.0).to(range_overshoot)
        active_count = active.flatten(1).sum(dim=1).clamp_min(1.0)
        active_overshoot = range_overshoot * active
        range_violation = (
            (active_overshoot > 2.0e-6).float().flatten(1).sum(dim=1) / active_count
        )
        range_violation_max = active_overshoot.flatten(1).max(dim=1).values
        b = base_logits.shape[0]
        zero = base_prob.new_zeros(b)
        one = base_prob.new_ones(b)
        flow_scale = self.mean_head.current_scale().to(base_prob).expand(b)
        context_gate = torch.tanh(self.context_gate).to(base_prob).expand(b)
        quality = base_prob.new_zeros((b, 2))
        if isinstance(case_utility_probs, torch.Tensor) and case_utility_probs.ndim == 2:
            deploy_edit_prob = case_utility_probs.max(dim=1).values
        else:
            deploy_edit_prob = case_utility_probs.reshape(b)
        quality[:, 0] = 1.0 - deploy_edit_prob
        quality[:, 1] = deploy_edit_prob

        aux: Dict[str, torch.Tensor] = {
            "candidates": candidate_logits,
            "candidate_probs": candidate_probs,
            "mhcs_final_logits": geo_logits,
            "mhcs_final_probs": geo_prob,
            "mhcs_local_probs": geo_prob,
            "mhcs_surface_hard_probs": geo_prob,
            "mhcs_global_selected_probs": geo_prob,
            "mhcs_quality_probs": quality,
            "mhcs_quality_logits": torch.log(quality.clamp_min(EPS)),
            "mhcs_quality_pred": quality,
            "mhcs_global_weights": quality,
            "mhcs_gate_alpha": change,
            "mhcs_effective_rank": torch.where(change > 1.0e-8, one * 2.0, one),
            "mhcs_surface_nonbase_mass": change,
            "mhcs_m1_distribution_log_var": self.m1_distribution_log_var,
            "v20_selector_hard": case_accept_hard[:, None],
            "direct_fused_probs": geo_prob,
            "router_fused_probs": geo_prob,
            "v20_fused_probs": geo_prob,
            "v20_hard_fused_probs": geo_prob,
            "v20_fused_logits": geo_logits,
            "geotopo_base_logits": factual_logits,
            "geotopo_base_probs": base_prob,
            "geotopo_geometry_flow_px": flow_px,
            "geotopo_geometry_logits": geo_logits,
            "geotopo_geometry_probs": geo_prob,
            "geotopo_residual_only_logits": factual_logits,
            "geotopo_residual_only_probs": base_prob,
            "geotopo_reconstruction_after_geometry_logits": geo_logits,
            "geotopo_reconstruction_after_geometry_probs": geo_prob,
            "geotopo_final_logits": geo_logits,
            "geotopo_final_probs": geo_prob,
            "geotopo_flow_rms_px": flow_mag.square().mean(dim=(1, 2)).sqrt(),
            "geotopo_flow_mean_px": flow_mag.mean(dim=(1, 2)),
            "geotopo_flow_max_px": flow_mag.flatten(1).max(dim=1).values,
            "geotopo_flow_jacobian_mean": jac_mean,
            "geotopo_flow_folding_fraction": folding,
            "geotopo_geometry_abs_change": change,
            "geotopo_mode_id": one,
            "geotr_m1_base_logits": factual_logits,
            "geotr_m1_final_logits": geo_logits,
            "geotr_m1_final_probs": geo_prob,
            "geotr_m1_flow_px": flow_px,
            "geotr_m1_scalar_field": scalar_field,
            "geotr_m1_regularized_field": regularized_field,
            "geotr_m1_normal_field": normal,
            "geotr_m1_change_gate": geometry_gate,
            "jbt_v6_error_selected_gate": gate,
            "geotr_m1_evidence_band": evidence_band,
            "geotr_m1_posterior_evidence": posterior_evidence,
            "geotr_m1_transport_anchor_logits": transport_anchor_logits,
            "geotr_m1_error_logits": error_logits,
            "geotr_m1_error_probs": error_probs,
            "geotr_m1_error_gate": error_gate,
            "geotr_m1_error_direction": error_direction,
            "geotr_m1_error_gate_fraction": (error_gate > 0.0).float().mean(dim=(1, 2, 3)),
            "geotr_m1_posterior_reconstruction_mismatch": posterior_reconstruction_mismatch,
            "geotr_m1_posterior_reconstruction_prob_mismatch": posterior_reconstruction_prob_mismatch,
            "geotr_m1_raw_geometry_logits": raw_geo_logits,
            "geotr_m1_raw_geometry_probs": raw_geo_prob,
            "jbt_case_utility_logits": case_utility_logits,
            "jbt_case_utility_probs": case_utility_probs,
            "jbt_case_accept_hard": case_accept_hard,
            "jbt_v63_selector_accept_hard": v63_selector_accept,
            "jbt_v63_val_fallback_used": v63_fallback_used,
            "jbt_v63_val_fallback_strength": factual_logits.new_full(
                (b,), float(self.v63_val_fallback_strength)
            ),
            "jbt_v5_strength_candidate_logits": torch.cat([factual_logits, strength_logits_t], dim=1),
            "jbt_v5_strength_candidate_probs": torch.cat([base_prob, strength_probs_t], dim=1),
            "jbt_v5_utility_outcome_probs": v5_utility_outcome_probs,
            "jbt_v5_utility_scores": v5_utility_scores,
            "jbt_v62_gain_mean": v62_gain_mean,
            "jbt_v62_gain_sigma": v62_gain_sigma,
            "jbt_v62_lcb_scores": v62_lcb_scores,
            "jbt_v5_selected_strength_index": selected_strength_index.to(factual_logits),
            "jbt_v5_selected_strength": selected_strength,
            "jbt_v5_support_radius_px": v5_support_radius_px,
            "jbt_v5_broad_support_fraction": (v5_proposal_gate > 0.0).float().mean(dim=(1,2,3)),
            "jbt_v5_effective_support_fraction": (geometry_gate > 0.0).float().mean(dim=(1,2,3)),
            "jbt_v6_error_selected_support_fraction": (gate > 0.0).float().mean(dim=(1,2,3)),
            "jbt_v5_multistrength_enabled": one if self.v5_multistrength_enabled else zero,
            "jbt_v6_direct_signed_flow": one if self.v6_direct_signed_flow else zero,
            "jbt_v6_candidate_utility_enabled": one if self.v6_candidate_utility_enabled else zero,
            "jbt_v6_direction_abs_mean": direction.detach().abs().mean(dim=(1,2,3)),
            "jbt_v6_scalar_abs_mean": scalar_field.detach().abs().mean(dim=(1,2,3)),
            "jbt_v62_enabled": one if self.v62_enabled else zero,
            "jbt_v63_enabled": one if self.v63_enabled else zero,
            "jbt_v634_utility_detached_from_generator": (
                one if self.v634_detach_utility_from_generator else zero
            ),
            "jbt_v62_actual_support_mean": geometry_gate.detach().mean(dim=(1,2,3)),
            "jbt_v62_active_flow_to_scalar_ratio": (
                (flow_mag * (geometry_gate[:, 0] > 0).to(flow_mag)).sum(dim=(1,2))
                / (
                    scalar_field[:, 0].detach().abs()
                    * (geometry_gate[:, 0] > 0).to(scalar_field)
                ).sum(dim=(1,2)).clamp_min(1.0e-6)
            ),
            "jbt_case_accept_fraction": case_accept_hard.detach(),
            "geotr_m1_dynamic_error_gate_enabled": (
                one if self.dynamic_error_gate_enabled else zero
            ),
            "geotr_m1_error_direction_coupling": (
                one if self.error_direction_coupling else zero
            ),
            "geotr_m1_feature_feedback_abs": feature_feedback_abs,
            "geotr_m1_feature_feedback_logit_change": feature_feedback_logit_change,
            "geotr_m1_feature_feedback_logit_change_max": feature_feedback_logit_change_max,
            "geotr_m1_feature_feedback_scale": feature_feedback_scale,
            "geotr_m1_feature_feedback_enabled": (
                one if self.feature_feedback_enabled else zero
            ),
            "geotr_m1_logit_bounded_feature_feedback": (
                one if self.logit_bounded_feature_feedback else zero
            ),
            "geotr_m1_error_gate_enabled": (
                one if self.error_gate_enabled else zero
            ),
            "geotr_m1_posterior_uncertainty_enabled": (
                one if self.posterior_uncertainty_enabled else zero
            ),
            "geotr_m1_rewrite_residual": rewrite_residual,
            "geotr_m1_flow_scale_px": flow_scale,
            "geotr_m1_context_gate": context_gate,
            "geotr_m1_context_film_abs": conditioning_stats["context_film_abs"],
            "geotr_m1_text_latent_abs": conditioning_stats["text_latent_abs"],
            "geotr_m1_anchor_boundary_fraction": conditioning_stats[
                "anchor_boundary_fraction"
            ],
            "geotr_m1_conditioners_detached": (
                zero if self.joint_base_integration
                else (one if self.detach_conditioners else zero)
            ),
            "geotr_m1_joint_base_integration": (
                one if self.joint_base_integration else zero
            ),
            "geotr_m1_has_m2": zero,
            "geotr_m1_uses_semantic": one if self.use_semantic_conditioning else zero,
            "geotr_m1_uses_text": one if self.use_text_conditioning else zero,
            "geotr_m1_uses_anchor_cues": one if self.use_anchor_cues else zero,
            "geotr_m1_logit_transport": one if self.transport_space == "logit" else zero,
            "geotr_m1_operator_id": base_prob.new_tensor(
                {"free_2d": 1.0, "normal_1d": 2.0, "residual_rewrite": 3.0}[
                    self.operator
                ]
            ).expand(b),
            "geotr_m1_gate_mode_id": base_prob.new_tensor(
                {"none": 0.0, "soft": 1.0, "deadzone": 2.0}[self.gate_mode]
            ).expand(b),
            "geotr_m1_is_normal_1d": one if self.operator == "normal_1d" else zero,
            "geotr_m1_deadzone_fraction": deadzone.float().mean(dim=(1, 2, 3)),
            "geotr_m1_exact_identity_error": identity_error,
            "geotr_m1_tangent_energy_ratio": tangent_energy_ratio,
            "geotr_m1_range_violation_fraction": range_violation,
            "geotr_m1_range_violation_max": range_violation_max,
        }
        return candidate_logits, aux
