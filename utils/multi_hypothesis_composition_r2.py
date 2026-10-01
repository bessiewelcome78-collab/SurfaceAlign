"""MHCS-R2: Structured Stochastic Hypothesis Composition.

This is the root-cause redesign of the R1 deterministic-query bank.

R1 failure mode
---------------
R1 generated K masks from K learned queries.  All queries were conditioned by
adding the same global vector before the first mask renderer, and the global
vector was much larger than the query identity at initialization.  The queries
therefore became almost collinear before training.  Hard winner-take-all MCL
then amplified a tiny random advantage into a single winning head.

R2 design
---------
R2 removes *both* sources of artificial hypothesis identity:

  image / semantic features / text
      -> image-conditioned low-rank logistic-normal logit distribution
         mean mu(x), factors P_r(x), diagonal std d(x)
      -> deterministic principal sigma hypotheses spanning the learned
         structured uncertainty directions
      -> Base inserted only as H0
      -> the unchanged permutation-equivariant set-aware pixel composer
      -> one final segmentation.

The stochastic bank is inspired by Stochastic Segmentation Networks (SSN,
NeurIPS 2020): a low-rank plus diagonal Gaussian distribution in logit space
models spatially correlated segmentation uncertainty.  At inference we do not
randomly draw the candidate bank.  Instead, one mean hypothesis and +/- each
low-rank factor are used as deterministic sigma hypotheses.  This makes the
bank reproducible and intentionally covers the principal structured variation
learned by the distribution.

The forward path is GT-free and generated hypotheses never read Base.  Base is
only candidate H0 for the final composer.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

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
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


class ConvNormGELU(nn.Sequential):
    def __init__(self, cin: int, cout: int, kernel_size: int = 3) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(cin, cout, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(_groups(cout), cout),
            nn.GELU(),
        )


class MultiHypothesisCompositionalSegmenter(nn.Module):
    """Structured stochastic full-mask bank + unchanged set-aware composer."""

    use_semantic_feature = True
    unified_m1_safe_fusion_enabled = True

    def __init__(self, cfg) -> None:
        super().__init__()
        m1 = _cfg_get(cfg, "M1", None)
        self.num_hypotheses = max(3, int(_cfg_get(m1, "MHCS_NUM_HYPOTHESES", 7)))
        self.rank = max(1, int(_cfg_get(m1, "MHCS_SSN_RANK", 3)))
        expected = 1 + 2 * self.rank
        if self.num_hypotheses != expected:
            raise ValueError(
                "MHCS-R2 uses one distribution mean plus +/- every low-rank factor; "
                f"therefore MHCS_NUM_HYPOTHESES must equal 1+2*MHCS_SSN_RANK={expected}, "
                f"got {self.num_hypotheses}."
            )
        self.mc_samples = max(2, int(_cfg_get(m1, "MHCS_SSN_MC_SAMPLES", 4)))
        self.hidden_dim = max(32, int(_cfg_get(m1, "MHCS_HIDDEN_DIM", 128)))
        self.semantic_channels = int(_cfg_get(m1, "SEMANTIC_CHANNELS", 512))
        self.text_dim = int(_cfg_get(m1, "MHCS_TEXT_DIM", 512))
        self.set_layers = max(1, int(_cfg_get(m1, "MHCS_SET_LAYERS", 2)))
        self.set_heads = max(1, int(_cfg_get(m1, "MHCS_SET_HEADS", 4)))
        while self.hidden_dim % self.set_heads != 0 and self.set_heads > 1:
            self.set_heads -= 1
        self.ffn_dim = max(self.hidden_dim, int(_cfg_get(m1, "MHCS_FFN_DIM", 256)))
        self.dropout = float(_cfg_get(m1, "MHCS_DROPOUT", 0.10))

        # Shared visual-semantic pixel representation.  Base is deliberately not
        # an input to any module in the generated-bank path.
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
            ConvNormGELU(2 * self.hidden_dim, self.hidden_dim),
            nn.Dropout2d(self.dropout),
            ConvNormGELU(self.hidden_dim, self.hidden_dim),
        )

        # Context is injected as a gated FiLM residual on pixels, never added to
        # a hypothesis identity vector.  The gate starts at exactly zero, so the
        # image/text global context cannot erase bank identity at initialization.
        self.global_context = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.context_film = nn.Linear(self.hidden_dim, 2 * self.hidden_dim)
        self.context_gate = nn.Parameter(torch.zeros(()))

        # SSN-style distribution heads in logit space.
        self.distribution_trunk = nn.Sequential(
            ConvNormGELU(self.hidden_dim, self.hidden_dim),
            nn.Dropout2d(self.dropout),
            ConvNormGELU(self.hidden_dim, self.hidden_dim),
        )
        self.mean_head = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)
        self.factor_head = nn.Conv2d(self.hidden_dim, self.rank, kernel_size=1)
        self.diag_std_head = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)

        # Fixed deterministic sigma codes: [mean, +e1, -e1, +e2, -e2, ...].
        # Unlike learned query IDs, these cannot collapse or be monopolised.
        codes = torch.zeros(self.num_hypotheses, self.rank)
        row = 1
        for r in range(self.rank):
            codes[row, r] = 1.0
            codes[row + 1, r] = -1.0
            row += 2
        self.register_buffer("sigma_codes", codes, persistent=True)

        # The R1 set-aware spatial composer is intentionally retained.
        self.candidate_token = nn.Sequential(
            nn.Linear(2 * self.hidden_dim + 4, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.set_heads,
            dim_feedforward=self.ffn_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.set_layers,
            norm=nn.LayerNorm(self.hidden_dim),
        )
        self.composer_pixel_proj = nn.Conv2d(
            self.hidden_dim, self.hidden_dim, kernel_size=1, bias=False
        )
        self.composer_token_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.local_evidence = nn.Sequential(
            nn.Linear(2, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

        # Two principled objectives only: SSN distribution likelihood and final
        # composed segmentation.  Their relative scale is learned rather than a
        # hand-set lambda.
        self.loss_log_vars = nn.Parameter(torch.zeros(2))

        self._reset_distribution_parameters()

    def _reset_distribution_parameters(self) -> None:
        with torch.no_grad():
            # Mean starts as an ordinary segmentation head.
            nn.init.xavier_uniform_(self.mean_head.weight)
            nn.init.zeros_(self.mean_head.bias)
            # Small but non-zero correlated factors.  Every factor participates
            # in the same distribution likelihood; there is no winner head.
            nn.init.normal_(self.factor_head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.factor_head.bias)
            # softplus(-4) ~= 0.018 logit std: numerically non-zero without
            # injecting destructive pixel noise at initialization.
            nn.init.zeros_(self.diag_std_head.weight)
            nn.init.constant_(self.diag_std_head.bias, -4.0)
            nn.init.zeros_(self.local_evidence[-1].weight)
            nn.init.zeros_(self.local_evidence[-1].bias)

    @staticmethod
    def _resize(x: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] == hw:
            return x
        return F.interpolate(x, size=hw, mode="bilinear", align_corners=False)

    def _pixel_features(
        self,
        image: torch.Tensor,
        semantic_map: torch.Tensor,
        text_features: torch.Tensor,
        output_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if semantic_map is None:
            raise RuntimeError("MHCS-R2 requires the spatial UniMedCLIP semantic map.")
        if semantic_map.shape[1] != self.semantic_channels:
            raise RuntimeError(
                f"MHCS-R2 expected {self.semantic_channels} semantic channels, "
                f"got {semantic_map.shape[1]}."
            )
        image_latent = self.image_stem(self._resize(image, output_hw))
        semantic_latent = self.semantic_proj(self._resize(semantic_map, output_hw))
        pixel = self.pixel_fuse(torch.cat([image_latent, semantic_latent], dim=1))
        text = self.text_proj(text_features.float())
        global_visual = F.adaptive_avg_pool2d(pixel, 1).flatten(1)
        global_context = self.global_context(torch.cat([global_visual, text], dim=1))

        gamma, beta = self.context_film(global_context).chunk(2, dim=1)
        gate = torch.tanh(self.context_gate)
        conditioned = pixel + gate * (
            torch.tanh(gamma)[:, :, None, None] * pixel
            + torch.tanh(beta)[:, :, None, None]
        )
        return conditioned, global_context

    def _distribution(self, pixel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.distribution_trunk(pixel)
        mean_logits = self.mean_head(hidden)                     # [B,1,H,W]
        factors = self.factor_head(hidden)                       # [B,R,H,W]
        diag_std = F.softplus(self.diag_std_head(hidden)) + EPS # [B,1,H,W]
        return mean_logits, factors, diag_std

    def _principal_hypotheses(
        self, mean_logits: torch.Tensor, factors: torch.Tensor
    ) -> torch.Tensor:
        # [B,K,H,W] = mean + deterministic +/- principal low-rank directions.
        delta = torch.einsum("kr,brhw->bkhw", self.sigma_codes.to(factors), factors)
        return mean_logits + delta

    def sample_distribution(
        self,
        mean_logits: torch.Tensor,
        factors: torch.Tensor,
        diag_std: torch.Tensor,
        num_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """Reparameterised SSN logit samples [B,M,H,W].

        Antithetic sampling is used when M is even to reduce Monte-Carlo
        variance while preserving the same Gaussian distribution.
        """
        m = int(self.mc_samples if num_samples is None else num_samples)
        m = max(2, m)
        b, r, h, w = factors.shape
        half = (m + 1) // 2
        z = torch.randn(b, half, r, device=factors.device, dtype=factors.dtype)
        eps = torch.randn(b, half, h, w, device=factors.device, dtype=factors.dtype)
        if m > half:
            z = torch.cat([z, -z[:, : m - half]], dim=1)
            eps = torch.cat([eps, -eps[:, : m - half]], dim=1)
        low_rank = torch.einsum("bmr,brhw->bmhw", z, factors)
        return mean_logits[:, None, 0] + low_rank + diag_std[:, None, 0] * eps

    @staticmethod
    def _soft_boundary(prob: torch.Tensor) -> torch.Tensor:
        flat = prob.reshape(-1, 1, *prob.shape[-2:])
        dilate = F.max_pool2d(flat, 3, stride=1, padding=1)
        erode = -F.max_pool2d(-flat, 3, stride=1, padding=1)
        return (dilate - erode).reshape_as(prob).clamp(0.0, 1.0)

    def _candidate_tokens(self, pixel: torch.Tensor, candidate_probs: torch.Tensor) -> torch.Tensor:
        p = candidate_probs.clamp(EPS, 1.0 - EPS)
        fg_mass = p.flatten(2).sum(-1).clamp_min(EPS)
        bg = 1.0 - p
        bg_mass = bg.flatten(2).sum(-1).clamp_min(EPS)
        fg_pool = torch.einsum("bchw,bdhw->bcd", p, pixel) / fg_mass[:, :, None]
        bg_pool = torch.einsum("bchw,bdhw->bcd", bg, pixel) / bg_mass[:, :, None]
        area = p.mean(dim=(-2, -1))
        entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)) / math.log(2.0)
        entropy_mean = entropy.mean(dim=(-2, -1))
        boundary = self._soft_boundary(p).mean(dim=(-2, -1))
        confidence = (p - 0.5).abs().mean(dim=(-2, -1)) * 2.0
        stats = torch.stack([area, entropy_mean, boundary, confidence], dim=-1)
        return self.candidate_token(torch.cat([fg_pool, bg_pool, stats], dim=-1))

    def _compose(self, pixel: torch.Tensor, candidate_probs: torch.Tensor) -> Dict[str, torch.Tensor]:
        tokens = self._candidate_tokens(pixel, candidate_probs)
        contextual = self.set_encoder(tokens)
        pixel_key = self.composer_pixel_proj(pixel)
        candidate_key = self.composer_token_proj(contextual)
        score = torch.einsum("bcd,bdhw->bchw", candidate_key, pixel_key) / math.sqrt(self.hidden_dim)

        p = candidate_probs.clamp(EPS, 1.0 - EPS)
        ent = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)) / math.log(2.0)
        local = torch.stack([p, ent], dim=-1)
        score = score + self.local_evidence(local).squeeze(-1)
        weights = torch.softmax(score, dim=1)
        final_prob = (weights * candidate_probs).sum(dim=1).clamp(EPS, 1.0 - EPS)
        return {
            "tokens": tokens,
            "contextual_tokens": contextual,
            "scores": score,
            "weights": weights,
            "final_prob": final_prob,
        }

    @staticmethod
    def _pairwise_cosine(x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 3 or x.shape[1] <= 1:
            return x.new_zeros(x.shape[0])
        y = F.normalize(x.flatten(2) if x.ndim > 3 else x, dim=-1)
        sim = torch.matmul(y, y.transpose(1, 2))
        k = sim.shape[1]
        mask = ~torch.eye(k, dtype=torch.bool, device=sim.device)[None]
        return sim.masked_select(mask.expand_as(sim)).reshape(sim.shape[0], -1).mean(dim=1)

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
        if base_logits.ndim == 3:
            base_logits = base_logits[:, None]
        if base_logits.ndim != 4 or base_logits.shape[1] != 1:
            raise ValueError(f"MHCS-R2 expects Base logits [B,1,H,W], got {tuple(base_logits.shape)}")

        output_hw = tuple(base_logits.shape[-2:])
        pixel, global_context = self._pixel_features(
            image=image,
            semantic_map=semantic_map,
            text_features=text_features,
            output_hw=output_hw,
        )
        mean_logits, factors, diag_std = self._distribution(pixel)
        generated_logits = self._principal_hypotheses(mean_logits, factors)
        generated_probs = torch.sigmoid(generated_logits).clamp(EPS, 1.0 - EPS)

        base_prob = torch.sigmoid(base_logits).clamp(EPS, 1.0 - EPS)
        candidate_probs = torch.cat([base_prob, generated_probs], dim=1)
        candidate_logits = torch.cat([base_logits, generated_logits], dim=1)
        composition = self._compose(pixel, candidate_probs)
        final_prob = composition["final_prob"]
        final_logits = torch.logit(final_prob)

        # Reparameterised likelihood samples are consumed only by the loss.
        # They remain Base-independent and GT-free.
        likelihood_samples = self.sample_distribution(mean_logits, factors, diag_std)

        global_weights = composition["weights"].mean(dim=(-2, -1))
        generated_weight = global_weights[:, 1:]
        best_generated = generated_weight.argmax(dim=1)
        beats_base = generated_weight.max(dim=1).values > global_weights[:, 0]
        selector_hard = generated_weight.new_zeros(generated_weight.shape)
        selector_hard.scatter_(1, best_generated[:, None], beats_base[:, None].to(generated_weight.dtype))
        selector_logits = torch.log(generated_weight.clamp_min(EPS)) - torch.log(global_weights[:, :1].clamp_min(EPS))
        action_supports = (generated_probs - base_prob).abs()
        edit_fraction = action_supports.mean(dim=(-2, -1))

        factor_flat = factors.flatten(2)
        aux: Dict[str, torch.Tensor] = {
            "candidates": candidate_logits,
            "candidate_probs": candidate_probs,
            "mhcs_generated_logits": generated_logits,
            "mhcs_generated_probs": generated_probs,
            "mhcs_distribution_mean_logits": mean_logits[:, 0],
            "mhcs_distribution_factors": factors,
            "mhcs_distribution_diag_std": diag_std[:, 0],
            "mhcs_distribution_samples": likelihood_samples,
            "mhcs_sigma_codes": self.sigma_codes,
            "mhcs_context_gate": torch.tanh(self.context_gate),
            "mhcs_factor_cosine": self._pairwise_cosine(factor_flat),
            "mhcs_composer_scores": composition["scores"],
            "mhcs_composer_weights": composition["weights"],
            "mhcs_global_weights": global_weights,
            "mhcs_final_probs": final_prob,
            "mhcs_final_logits": final_logits,
            "mhcs_loss_log_vars": self.loss_log_vars,
            "mhcs_set_token_cosine": self._pairwise_cosine(composition["contextual_tokens"]),
            # Existing project interfaces: these are compatibility aliases only.
            "direct_fused_probs": final_prob,
            "router_fused_probs": final_prob,
            "v20_fused_probs": final_prob,
            "v20_hard_fused_probs": final_prob,
            "v20_fused_logits": final_logits,
            "m1_soft_fused_probs": final_prob,
            "m1_hard_fused_probs": final_prob,
            "m1_soft_fused_logits": final_logits,
            "v552r4209_m1_native_probability": final_prob,
            "v20_selector_hard": selector_hard,
            "v20_selector_logits": selector_logits,
            "v20_action_supports": action_supports,
            "v20_action_types": torch.zeros(
                self.num_hypotheses, dtype=torch.long, device=base_logits.device
            ),
            "v20_edit_fraction": edit_fraction,
        }
        return candidate_logits, aux

    def forward(self, *args, **kwargs):
        return self.generate(*args, **kwargs)
