"""Multi-Hypothesis Compositional Segmentation (MHCS).

Scientific design
-----------------
This module implements a clean hypothesis-space segmentation path:

    image / semantic features / text
        -> K complete segmentation hypotheses
        -> permutation-equivariant set interaction
        -> pixel-wise hypothesis composition
        -> one final complete segmentation.

The generator never reads the Base prediction when producing H1..HK.  Base is
inserted only as H0 in the hypothesis bank, so it is a candidate rather than the
reference frame of the method.

Mechanism references:
  * Multiple Choice Learning (Guzman-Rivera et al., NeurIPS 2012): multiple
    structured hypotheses with winner-take-all / oracle-style supervision.
  * Set Transformer (Lee et al., ICML 2019): attention over unordered sets.
  * Mask2Former (Cheng et al., CVPR 2022): mask-guided localized cross-attention.

No GT is accepted by forward/generate.  Ground truth is used only by the loss
module in ``utils/multi_hypothesis_composition_loss.py``.
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


class SoftMaskGuidedQueryDecoder(nn.Module):
    """One differentiable Mask2Former-inspired query refinement step.

    The coarse predicted mask is converted to a *soft* additive attention prior.
    There is no hard threshold, connected component, proposal gate, NMS, or GT
    pilot.  A wrong coarse mask still receives gradients from the refined mask.
    """

    def __init__(self, hidden_dim: int, ffn_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        queries: torch.Tensor,
        pixel_tokens: torch.Tensor,
        soft_mask_prior: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # queries: [B,K,D], pixel_tokens: [B,N,D], prior: [B,K,N] in [0,1]
        q = self.q_proj(queries)
        k = self.k_proj(pixel_tokens)
        v = self.v_proj(pixel_tokens)
        score = torch.einsum("bkd,bnd->bkn", q, k) / math.sqrt(self.hidden_dim)
        score = score + torch.log(soft_mask_prior.clamp_min(EPS))
        attention = torch.softmax(score, dim=-1)
        context = torch.einsum("bkn,bnd->bkd", attention, v)
        queries = self.norm1(queries + self.out_proj(context))
        queries = self.norm2(queries + self.ffn(queries))
        return queries, attention


class MultiHypothesisCompositionalSegmenter(nn.Module):
    """Generate complete masks and compose them as an unordered hypothesis set."""

    # Compatibility flags consumed by the existing MedCLIPSeg wrapper.
    use_semantic_feature = True
    unified_m1_safe_fusion_enabled = True

    def __init__(self, cfg) -> None:
        super().__init__()
        m1 = _cfg_get(cfg, "M1", None)
        self.num_hypotheses = max(2, int(_cfg_get(m1, "MHCS_NUM_HYPOTHESES", 6)))
        self.hidden_dim = max(32, int(_cfg_get(m1, "MHCS_HIDDEN_DIM", 128)))
        self.semantic_channels = int(_cfg_get(m1, "SEMANTIC_CHANNELS", 512))
        self.text_dim = int(_cfg_get(m1, "MHCS_TEXT_DIM", 512))
        self.attention_pool_size = max(7, int(_cfg_get(m1, "MHCS_ATTENTION_POOL_SIZE", 28)))
        self.set_layers = max(1, int(_cfg_get(m1, "MHCS_SET_LAYERS", 2)))
        self.set_heads = max(1, int(_cfg_get(m1, "MHCS_SET_HEADS", 4)))
        while self.hidden_dim % self.set_heads != 0 and self.set_heads > 1:
            self.set_heads -= 1
        self.ffn_dim = max(self.hidden_dim, int(_cfg_get(m1, "MHCS_FFN_DIM", 256)))
        self.dropout = float(_cfg_get(m1, "MHCS_DROPOUT", 0.10))

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
        self.global_context = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )

        self.hypothesis_queries = nn.Embedding(self.num_hypotheses, self.hidden_dim)
        self.query_norm = nn.LayerNorm(self.hidden_dim)
        self.mask_pixel_proj = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 1, bias=False),
            nn.GroupNorm(_groups(self.hidden_dim), self.hidden_dim),
            nn.GELU(),
        )
        self.mask_query_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.mask_bias = nn.Parameter(torch.zeros(self.num_hypotheses))
        self.query_decoder = SoftMaskGuidedQueryDecoder(
            hidden_dim=self.hidden_dim,
            ffn_dim=self.ffn_dim,
        )

        # Candidate tokens deliberately have no candidate-index positional
        # embedding.  Therefore the composer is permutation equivariant with
        # respect to the hypothesis bank.
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
        self.composer_token_proj = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False
        )
        # Shared evidence head: no candidate-specific bias and no special Base
        # treatment.  It reads only a candidate's local probability/entropy.
        self.local_evidence = nn.Sequential(
            nn.Linear(2, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

        # Learned homoscedastic weighting for two standard objectives:
        # final-composition segmentation and MCL bank coverage.
        self.loss_log_vars = nn.Parameter(torch.zeros(2))

        with torch.no_grad():
            nn.init.normal_(self.hypothesis_queries.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.mask_bias)
            # Keep initial candidate-local evidence neutral; set/context dot
            # products determine the first composer updates.
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
            raise RuntimeError("MHCS requires the spatial UniMedCLIP semantic map.")
        if semantic_map.shape[1] != self.semantic_channels:
            raise RuntimeError(
                f"MHCS expected {self.semantic_channels} semantic channels, "
                f"got {semantic_map.shape[1]}."
            )
        image_latent = self.image_stem(self._resize(image, output_hw))
        semantic_latent = self.semantic_proj(self._resize(semantic_map, output_hw))
        pixel = self.pixel_fuse(torch.cat([image_latent, semantic_latent], dim=1))
        text = self.text_proj(text_features.float())
        global_visual = F.adaptive_avg_pool2d(pixel, 1).flatten(1)
        global_context = self.global_context(torch.cat([global_visual, text], dim=1))
        return pixel, global_context

    def _render_masks(
        self, queries: torch.Tensor, mask_pixels: torch.Tensor
    ) -> torch.Tensor:
        mask_q = self.mask_query_proj(queries)
        logits = torch.einsum("bkd,bdhw->bkhw", mask_q, mask_pixels)
        logits = logits / math.sqrt(self.hidden_dim)
        logits = logits + self.mask_bias[None, :, None, None]
        return logits

    @staticmethod
    def _soft_boundary(prob: torch.Tensor) -> torch.Tensor:
        flat = prob.reshape(-1, 1, *prob.shape[-2:])
        dilate = F.max_pool2d(flat, 3, stride=1, padding=1)
        erode = -F.max_pool2d(-flat, 3, stride=1, padding=1)
        return (dilate - erode).reshape_as(prob).clamp(0.0, 1.0)

    def _candidate_tokens(
        self, pixel: torch.Tensor, candidate_probs: torch.Tensor
    ) -> torch.Tensor:
        # candidate_probs [B,C,H,W], pixel [B,D,H,W]
        p = candidate_probs.clamp(EPS, 1.0 - EPS)
        fg_mass = p.flatten(2).sum(-1).clamp_min(EPS)
        bg = 1.0 - p
        bg_mass = bg.flatten(2).sum(-1).clamp_min(EPS)
        fg_pool = torch.einsum("bchw,bdhw->bcd", p, pixel) / fg_mass[:, :, None]
        bg_pool = torch.einsum("bchw,bdhw->bcd", bg, pixel) / bg_mass[:, :, None]
        area = p.mean(dim=(-2, -1))
        entropy = -(
            p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)
        ) / math.log(2.0)
        entropy_mean = entropy.mean(dim=(-2, -1))
        boundary = self._soft_boundary(p).mean(dim=(-2, -1))
        confidence = (p - 0.5).abs().mean(dim=(-2, -1)) * 2.0
        stats = torch.stack([area, entropy_mean, boundary, confidence], dim=-1)
        token = torch.cat([fg_pool, bg_pool, stats], dim=-1)
        return self.candidate_token(token)

    def _compose(
        self,
        pixel: torch.Tensor,
        candidate_probs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        tokens = self._candidate_tokens(pixel, candidate_probs)
        contextual = self.set_encoder(tokens)
        pixel_key = self.composer_pixel_proj(pixel)
        candidate_key = self.composer_token_proj(contextual)
        score = torch.einsum("bcd,bdhw->bchw", candidate_key, pixel_key)
        score = score / math.sqrt(self.hidden_dim)

        p = candidate_probs.clamp(EPS, 1.0 - EPS)
        ent = -(
            p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)
        ) / math.log(2.0)
        local = torch.stack([p, ent], dim=-1)
        local_score = self.local_evidence(local).squeeze(-1)
        score = score + local_score
        weights = torch.softmax(score, dim=1)
        final_prob = (weights * candidate_probs).sum(dim=1).clamp(EPS, 1.0 - EPS)
        return {
            "tokens": tokens,
            "contextual_tokens": contextual,
            "scores": score,
            "weights": weights,
            "final_prob": final_prob,
        }

    def generate(
        self,
        base_logits: torch.Tensor,
        image: torch.Tensor,
        semantic_map: Optional[torch.Tensor],
        text_features: torch.Tensor,
        negative_text_features: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # negative_text_features and legacy kwargs are intentionally unused:
        # MHCS is a full-mask hypothesis-space model, not a signed Base-error
        # intervention bank.
        del negative_text_features, kwargs
        if base_logits.ndim == 3:
            base_logits = base_logits[:, None]
        if base_logits.ndim != 4 or base_logits.shape[1] != 1:
            raise ValueError(
                f"MHCS expects Base logits [B,1,H,W], got {tuple(base_logits.shape)}"
            )
        output_hw = tuple(base_logits.shape[-2:])
        pixel, global_context = self._pixel_features(
            image=image,
            semantic_map=semantic_map,
            text_features=text_features,
            output_hw=output_hw,
        )
        batch = base_logits.shape[0]
        queries = self.hypothesis_queries.weight[None].expand(batch, -1, -1)
        queries = self.query_norm(queries + global_context[:, None, :])
        mask_pixels = self.mask_pixel_proj(pixel)
        stage0_logits = self._render_masks(queries, mask_pixels)

        pooled_pixels = F.adaptive_avg_pool2d(
            pixel, self.attention_pool_size
        ).flatten(2).transpose(1, 2)
        coarse_prior = F.adaptive_avg_pool2d(
            torch.sigmoid(stage0_logits), self.attention_pool_size
        ).flatten(2)
        refined_queries, mask_attention = self.query_decoder(
            queries,
            pooled_pixels,
            coarse_prior,
        )
        generated_logits = self._render_masks(refined_queries, mask_pixels)
        generated_probs = torch.sigmoid(generated_logits).clamp(EPS, 1.0 - EPS)

        base_prob = torch.sigmoid(base_logits).clamp(EPS, 1.0 - EPS)
        candidate_probs = torch.cat([base_prob, generated_probs], dim=1)
        candidate_logits = torch.cat([base_logits, generated_logits], dim=1)
        composition = self._compose(pixel, candidate_probs)
        final_prob = composition["final_prob"]
        final_logits = torch.logit(final_prob)

        global_weights = composition["weights"].mean(dim=(-2, -1))
        generated_weight = global_weights[:, 1:]
        best_generated = generated_weight.argmax(dim=1)
        beats_base = generated_weight.max(dim=1).values > global_weights[:, 0]
        selector_hard = generated_weight.new_zeros(generated_weight.shape)
        selector_hard.scatter_(1, best_generated[:, None], beats_base[:, None].to(generated_weight.dtype))
        selector_logits = torch.log(generated_weight.clamp_min(EPS)) - torch.log(
            global_weights[:, :1].clamp_min(EPS)
        )
        action_supports = (generated_probs - base_prob).abs()
        edit_fraction = action_supports.mean(dim=(-2, -1))

        aux: Dict[str, torch.Tensor] = {
            "candidates": candidate_logits,
            "candidate_probs": candidate_probs,
            "mhcs_stage0_logits": stage0_logits,
            "mhcs_generated_logits": generated_logits,
            "mhcs_generated_probs": generated_probs,
            "mhcs_mask_attention": mask_attention,
            "mhcs_composer_scores": composition["scores"],
            "mhcs_composer_weights": composition["weights"],
            "mhcs_global_weights": global_weights,
            "mhcs_final_probs": final_prob,
            "mhcs_final_logits": final_logits,
            "mhcs_loss_log_vars": self.loss_log_vars,
            "mhcs_hypothesis_query_cosine": self._pairwise_cosine(refined_queries),
            "mhcs_set_token_cosine": self._pairwise_cosine(composition["contextual_tokens"]),
            # Existing project interfaces.  In MHCS these are aliases to the
            # compositional final mask, not Preserve/Edit actions.
            "direct_fused_probs": final_prob,
            "router_fused_probs": final_prob,
            "v20_fused_probs": final_prob,
            "v20_hard_fused_probs": final_prob,
            "v20_fused_logits": final_logits,
            "m1_soft_fused_probs": final_prob,
            "m1_hard_fused_probs": final_prob,
            "m1_soft_fused_logits": final_logits,
            "m1_hard_fused_logits": final_logits,
            "m1_selector_soft": global_weights,
            "m1_selector_hard": torch.cat(
                [(~beats_base)[:, None].to(generated_weight.dtype), selector_hard], dim=1
            ),
            "m1_choice_logits": torch.log(global_weights.clamp_min(EPS)),
            "v20_selector_logits": selector_logits,
            "v20_selector_probs": generated_weight,
            "v20_selector_hard": selector_hard,
            "v20_action_supports": action_supports,
            "v20_control_supports": torch.zeros_like(action_supports),
            "v20_action_types": torch.arange(
                self.num_hypotheses, device=base_logits.device, dtype=torch.long
            ),
            "v20_cf_logit": selector_logits,
            "v20_cf_signed_delta": selector_logits,
            "v20_cf_available": torch.ones_like(selector_logits),
            "v20_budget": edit_fraction,
            "local_action_area": edit_fraction,
            # Legacy selection metric can consume this tensor; it is exactly
            # the MHCS final composition, not a separate M2 model.
            "m2_fused_probs": final_prob,
        }
        return candidate_logits, aux

    @staticmethod
    def _pairwise_cosine(tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] <= 1:
            return tokens.new_zeros(tokens.shape[0])
        z = F.normalize(tokens, dim=-1, eps=1.0e-6)
        sim = torch.einsum("bkd,bjd->bkj", z, z)
        k = sim.shape[1]
        mask = ~torch.eye(k, dtype=torch.bool, device=sim.device)
        return sim[:, mask].reshape(sim.shape[0], -1).mean(dim=1)
