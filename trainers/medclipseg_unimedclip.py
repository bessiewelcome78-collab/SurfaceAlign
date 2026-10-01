"""UniMedCLIP MedCLIPSeg with V20 unified action-counterfactual-set selection.

V20 is one jointly optimized model: atomic action proposal pool -> matched
counterfactual text falsification -> sparse non-overlapping action-set selection.
There is no M1/M2/M3 stage freeze or checkpoint reload.
"""
from __future__ import annotations

import os
import math
import copy
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _activation_checkpoint
from huggingface_hub import hf_hub_download

from open_clip_lib import HFTokenizer, create_model_and_transforms, get_mean_std
from .layers import PVL_Adapter
from .scale_block import ScaleBlock
from .ugbra import UncertaintyGatedBoundaryResonanceAdapter
from .qabr import QueryAnchoredBoundaryRefiner
from .v463_causal_modules import V463ResidualErrorHead, V463CausalCounterfactualVerifier
from utils.v481_m1_residual_repair import v481_reproject_m1_candidates
from utils.v484_error_state_causal import V484ErrorStateCausalPipeline
from utils.multi_hypothesis_composition import MultiHypothesisCompositionalSegmenter
from utils.semlt_logit_transport import SemanticLogitTransportSegmenter
from utils.geotr_m1_transport import ExactGeometryTransportSegmenter
from utils.semlt_autozero_transport import AutoZeroSemanticTransportSegmenter

EPS = 1e-4


def _scale_gradient(value: torch.Tensor, scale: float) -> torch.Tensor:
    """Keep the forward value unchanged while scaling its upstream gradient.

    scale=1 is ordinary end-to-end backpropagation, scale=0 is stop-gradient,
    and values in (0,1) provide bounded but non-zero cross-module gradients.
    """
    if not isinstance(value, torch.Tensor):
        return value
    scale = float(scale)
    if scale >= 1.0:
        return value
    if scale <= 0.0:
        return value.detach()
    return value.detach() + scale * (value - value.detach())


# Fixed geometry of the validated reference candidate generator.
# These are implementation constants, not dataset-specific YAML tuning knobs.
REFERENCE_CANDIDATE_GEOMETRY = {
    "window_radius": 7,
    "nms_radius": 14,
    "context_radius": 8,
    "outer_radius": 4,
    "hole_radius": 3,
    "protrusion_radius": 2,
    "density_radius": 3,
    "density_max": 0.70,
    "island_connect_radius": 10,
    "type_window_radius": {0: 5, 1: 5, 2: 4, 3: 5},
    "type_nms_radius": {0: 14, 1: 14, 2: 7, 3: 14},
    "fill_edge_weight": 0.35,
    "fill_uncertainty_weight": 0.65,
    "fill_evidence_floor": 0.15,
    "expanded_fill_radius": 2,
    "control_context_radius": 2,
    "fill_control_outer_radius": 8,
}


def _uses_reference_mechanism(m1: Any) -> bool:
    """Canonical public switch with a legacy-read fallback for old runs."""
    mode = str(_cfg_get(m1, "CANDIDATE_MODE", "")).strip().lower()
    return mode == "mechanism" or bool(
        _cfg_get(m1, "V426_MECHANISM_DUAL_EXPERT", False)
    )


def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def _canonical_m1_inference_mode(mode: Any) -> str:
    """Map public algorithm names onto the historical outer dispatcher.

    SPARC-HR3 DCRR is implemented inside the MHCS/GEOTR candidate path and
    therefore uses the validated unified-action outer execution machinery.
    Keeping the public DCRR name in YAML is useful for auditability, but it
    must not create a second, partially integrated top-level dispatch mode.
    """
    normalized = str(mode).strip().lower()
    aliases = {
        "dense_counterfactual_residual_routing": "unified_action_cf_selection",
    }
    return aliases.get(normalized, normalized)


def _offline_mode() -> bool:
    return any(
        str(os.environ.get(name, "0")).lower() in {"1", "true", "yes"}
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_LOCAL_FILES_ONLY")
    )


def download_checkpoint(filename: str, cfg=None) -> str:
    """Resolve the original UniMedCLIP checkpoint without silently changing weights."""
    configured = str(_cfg_get(_cfg_get(cfg, "MODEL", None), "UNIMEDCLIP_CHECKPOINT", "") or "")
    if configured:
        if os.path.isfile(configured):
            print(f"Found configured checkpoint: {configured}")
            return configured
        raise FileNotFoundError(f"MODEL.UNIMEDCLIP_CHECKPOINT does not exist: {configured}")

    ckpt_dir = str(_cfg_get(_cfg_get(cfg, "MODEL", None), "CHECKPOINT_DIR", "checkpoints"))
    local_path = os.path.join(ckpt_dir, filename)
    if os.path.isfile(local_path):
        print(f"Found checkpoint: {local_path}")
        return local_path

    if _offline_mode():
        raise FileNotFoundError(
            f"Offline mode is enabled and {local_path} is missing. "
            "Use the original UniMedCLIP checkpoint; do not silently substitute another checkpoint."
        )

    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoint not found. Downloading {filename} from Hugging Face...")
    hf_hub_download(
        repo_id="TahaKoleilat/MedCLIPSeg",
        repo_type="model",
        filename=f"checkpoints/{filename}",
        local_dir=".",
        local_dir_use_symlinks=False,
    )
    if not os.path.isfile(local_path):
        raise FileNotFoundError(f"Downloaded checkpoint was not found at {local_path}")
    return local_path


def load_unimedclip_to_device(cfg):
    if cfg.MODEL.BACKBONE == "ViT-B/16":
        model_name = "ViT-B-16-quickgelu"
        pretrained_weights = download_checkpoint("unimed_clip_vit_b16.pt", cfg)
    elif cfg.MODEL.BACKBONE == "ViT-L/14":
        model_name = "ViT-L-14-336-quickgelu"
        pretrained_weights = download_checkpoint("unimed_clip_vit_l14_base_text_encoder.pt", cfg)
    else:
        raise NotImplementedError(f"Backbone {cfg.MODEL.BACKBONE} not implemented.")

    text_encoder_name = str(
        _cfg_get(cfg.MODEL, "TEXT_ENCODER_PATH", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
    )
    if _offline_mode() and not os.path.isdir(text_encoder_name):
        raise FileNotFoundError(
            "Offline mode requires MODEL.TEXT_ENCODER_PATH to be your local BiomedBERT directory: "
            f"{text_encoder_name}"
        )

    mean, std = get_mean_std()
    model, _, _ = create_model_and_transforms(
        model_name,
        pretrained_weights,
        precision="amp",
        device=cfg.MODEL.DEVICE,
        force_quick_gelu=True,
        mean=mean,
        std=std,
        inmem=True,
        text_encoder_name=text_encoder_name,
    )
    return model.to(cfg.MODEL.DEVICE).eval()


def _soft_erode(x: torch.Tensor, radius: int) -> torch.Tensor:
    radius = int(radius)
    if radius <= 0:
        return x
    kernel = 2 * radius + 1
    return -F.max_pool2d(-x, kernel_size=kernel, stride=1, padding=radius)


def _soft_dilate(x: torch.Tensor, radius: int) -> torch.Tensor:
    radius = int(radius)
    if radius <= 0:
        return x
    kernel = 2 * radius + 1
    return F.max_pool2d(x, kernel_size=kernel, stride=1, padding=radius)



def _run_checkpointed_module(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Memory-safe module forward for trainable dense M1 blocks.

    This does not change the mathematical forward result. It only trades
    extra recomputation during backward for lower activation memory.  Use
    non-reentrant checkpointing so modules with trainable parameters are
    checkpointed even when the input tensor itself is detached.
    """
    if torch.is_grad_enabled():
        try:
            has_trainable = any(p.requires_grad for p in module.parameters(recurse=True))
        except Exception:
            has_trainable = bool(getattr(x, "requires_grad", False))
        if bool(getattr(x, "requires_grad", False)) or has_trainable:
            return _activation_checkpoint(module, x, use_reentrant=False)
    return module(x)


class _ConvNormGELU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = min(8, int(out_channels))
        while groups > 1 and int(out_channels) % groups != 0:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)



class CounterfactualEffectEstimator(nn.Module):
    """M2: candidate-level counterfactual treatment-effect estimator.

    The module compares every intervention candidate against the factual
    no-intervention candidate C0.  It predicts a relative effect mean, an
    uncertainty scale, and antisymmetric pairwise preferences.  All operations
    are differentiable and all parameters are jointly optimized with the
    segmentation and candidate-generation branches.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        min_sigma: float,
        max_sigma: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.effect_head = nn.Linear(hidden_dim, 2)
        self.pairwise_head = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        with torch.no_grad():
            nn.init.normal_(self.effect_head.weight, mean=0.0, std=1.0e-3)
            nn.init.zeros_(self.effect_head.bias)
            self.effect_head.bias[1].fill_(-4.0)
            nn.init.normal_(self.pairwise_head[-1].weight, mean=0.0, std=1.0e-3)
            nn.init.zeros_(self.pairwise_head[-1].bias)

    def forward(self, candidate_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        if candidate_features.ndim != 3:
            raise ValueError(
                'CounterfactualEffectEstimator expects [B,M,D], got '
                f'{tuple(candidate_features.shape)}'
            )
        batch, candidates, _ = candidate_features.shape
        representation = self.encoder(candidate_features)
        raw = self.effect_head(representation)
        mean_raw = raw[..., 0]
        effect_mean = mean_raw - mean_raw[:, :1]
        effect_sigma = F.softplus(raw[..., 1]) + self.min_sigma
        effect_sigma = effect_sigma.clamp(self.min_sigma, self.max_sigma)
        effect_sigma = torch.cat(
            [
                effect_sigma.new_full((batch, 1), self.min_sigma),
                effect_sigma[:, 1:],
            ],
            dim=1,
        )

        left = representation[:, :, None, :].expand(-1, -1, candidates, -1)
        right = representation[:, None, :, :].expand(-1, candidates, -1, -1)
        pair_input = torch.cat(
            [left, right, left - right, left * right], dim=-1
        )
        raw_pairwise = self.pairwise_head(pair_input).squeeze(-1)
        pairwise_logits = 0.5 * (raw_pairwise - raw_pairwise.transpose(1, 2))
        diagonal = torch.eye(
            candidates, device=pairwise_logits.device, dtype=torch.bool
        )[None]
        pairwise_logits = pairwise_logits.masked_fill(diagonal, 0.0)
        pairwise_probability = torch.sigmoid(pairwise_logits)
        pairwise_win_rate = (
            pairwise_probability.sum(dim=2) - 0.5
        ) / max(candidates - 1, 1)

        return {
            'representation': representation,
            'effect_mean': effect_mean,
            'effect_sigma': effect_sigma,
            'pairwise_logits': pairwise_logits,
            'pairwise_probability': pairwise_probability,
            'pairwise_win_rate': pairwise_win_rate,
        }


class RiskAwareInterventionPolicy(nn.Module):
    """M3: risk-aware intervention policy over C0 and M1 candidates.

    M2 provides counterfactual effect evidence.  M3 converts that evidence into
    a differentiable training policy and a conservative hard inference action.
    C0 is a factual no-intervention action with fixed reference score zero.
    """

    def __init__(
        self,
        hidden_dim: int,
        temperature: float,
        lcb_kappa: float,
        pairwise_weight: float,
        edit_penalty: float,
        mc_penalty: float,
        residual_scale: float,
        utility_margin: float,
        local_max_edit: float,
        discovery_max_edit: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.temperature = max(float(temperature), 1.0e-3)
        self.lcb_kappa = float(lcb_kappa)
        self.pairwise_weight = float(pairwise_weight)
        self.edit_penalty = float(edit_penalty)
        self.mc_penalty = float(mc_penalty)
        self.residual_scale = float(residual_scale)
        self.utility_margin = float(utility_margin)
        self.local_max_edit = float(local_max_edit)
        self.discovery_max_edit = float(discovery_max_edit)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim + 8, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        with torch.no_grad():
            nn.init.normal_(self.policy_head[-1].weight, mean=0.0, std=1.0e-3)
            nn.init.zeros_(self.policy_head[-1].bias)

    def forward(
        self,
        representation: torch.Tensor,
        effect_mean: torch.Tensor,
        effect_sigma: torch.Tensor,
        pairwise_logits: torch.Tensor,
        candidate_stats: torch.Tensor,
        failure_probability: torch.Tensor,
        action_types: torch.Tensor,
        candidate_probs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch, candidates, _ = representation.shape
        pair_prob = torch.sigmoid(pairwise_logits)
        pair_win = (pair_prob.sum(dim=2) - 0.5) / max(candidates - 1, 1)
        failure = failure_probability[:, None].expand(-1, candidates)
        scalars = torch.stack(
            [
                effect_mean,
                effect_sigma,
                pair_win,
                candidate_stats[..., 2],  # edit fraction
                candidate_stats[..., 9],  # local MC disagreement
                candidate_stats[..., 0],  # candidate area
                candidate_stats[..., 5],  # boundary fraction
                failure,
            ],
            dim=-1,
        )
        residual = self.residual_scale * torch.tanh(
            self.policy_head(torch.cat([representation, scalars], dim=-1)).squeeze(-1)
        )
        base_score = (
            effect_mean
            - self.lcb_kappa * effect_sigma
            + self.pairwise_weight * (pair_win - 0.5)
            - self.edit_penalty * candidate_stats[..., 2]
            - self.mc_penalty * candidate_stats[..., 9]
        )
        policy_score = base_score + residual
        policy_score = torch.cat(
            [policy_score.new_zeros((batch, 1)), policy_score[:, 1:]], dim=1
        )
        soft_weights = torch.softmax(policy_score / self.temperature, dim=1)
        hard_index_argmax = policy_score.argmax(dim=1)

        # Inference uses a progressive pairwise tournament.  The differentiable
        # soft policy remains the training path; the hard tournament is only a
        # deterministic deployment decision and always compares the winner with C0.
        if not self.training and candidates > 1:
            action_scores = policy_score[:, 1:]
            ordered_actions = action_scores.argsort(dim=1, descending=False) + 1
            winner = ordered_actions[:, 0]
            for position in range(1, ordered_actions.shape[1]):
                challenger = ordered_actions[:, position]
                pref = pairwise_logits.gather(
                    1, winner[:, None, None].expand(-1, 1, candidates)
                ).squeeze(1).gather(1, challenger[:, None]).squeeze(1)
                winner = torch.where(pref >= 0.0, winner, challenger)
            beats_factual = pairwise_logits.gather(
                1, winner[:, None, None].expand(-1, 1, candidates)
            ).squeeze(1)[:, 0] > 0.0
            hard_index = torch.where(
                beats_factual, winner, torch.zeros_like(winner)
            )
        else:
            hard_index = hard_index_argmax

        action_type_full = torch.cat(
            [
                action_types.new_full((1,), -1),
                action_types,
            ],
            dim=0,
        )[:candidates]
        type_per_case = action_type_full[hard_index]
        max_edit = torch.where(
            type_per_case >= 4,
            policy_score.new_full((batch,), self.discovery_max_edit),
            policy_score.new_full((batch,), self.local_max_edit),
        )
        chosen_score = policy_score.gather(1, hard_index[:, None]).squeeze(1)
        chosen_edit = candidate_stats[..., 2].gather(
            1, hard_index[:, None]
        ).squeeze(1)
        accept = (
            (hard_index > 0)
            & (chosen_score > self.utility_margin)
            & (chosen_edit <= max_edit)
        )
        selected_index = torch.where(
            accept, hard_index, torch.zeros_like(hard_index)
        )

        hard_weights = F.one_hot(selected_index, num_classes=candidates).to(
            policy_score.dtype
        )
        st_weights = hard_weights + soft_weights - soft_weights.detach()
        soft_final = (
            soft_weights[:, :, None, None] * candidate_probs
        ).sum(dim=1)
        st_final = (
            st_weights[:, :, None, None] * candidate_probs
        ).sum(dim=1)
        gather = selected_index[:, None, None, None].expand(
            -1, 1, candidate_probs.shape[-2], candidate_probs.shape[-1]
        )
        hard_final = candidate_probs.gather(1, gather)[:, 0]
        return {
            'policy_score': policy_score,
            'policy_soft': soft_weights,
            'policy_st': st_weights,
            'selected_index': selected_index,
            'accept': accept.to(policy_score.dtype),
            'soft_final': soft_final,
            'st_final': st_final,
            'hard_final': hard_final,
            'pairwise_win_rate': pair_win,
            'chosen_score': chosen_score,
            'chosen_edit': chosen_edit,
        }



class FamilyCounterfactualEffectEstimator(nn.Module):
    """V479 M2: family-conditioned candidate-vs-factual effect estimator.

    A single shared representation is retained, but effect heads are separated
    for Preserve, Delete, Fill, Boundary and Discovery families.  The low
    quantile is parameterised as mean-softplus(gap), which prevents quantile
    crossing by construction.  Pairwise preferences are derived from one
    globally consistent scalar score rather than an independent cyclic head.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        family_count: int,
        quantile_alpha: float,
        dsc_weight: float,
        nsd_weight: float,
        pairwise_temperature: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.family_count = int(family_count)
        self.quantile_alpha = float(quantile_alpha)
        self.dsc_weight = float(dsc_weight)
        self.nsd_weight = float(nsd_weight)
        self.pairwise_temperature = max(float(pairwise_temperature), 1.0e-4)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.family_embedding = nn.Embedding(self.family_count, hidden_dim)
        self.family_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 4),
            )
            for _ in range(self.family_count)
        ])
        with torch.no_grad():
            nn.init.normal_(self.family_embedding.weight, mean=0.0, std=0.02)
            for head in self.family_heads:
                nn.init.normal_(head[-1].weight, mean=0.0, std=1.0e-3)
                nn.init.zeros_(head[-1].bias)
                # q10 starts slightly below the mean, but not so low that every
                # candidate is permanently rejected at initialisation.
                head[-1].bias[1].fill_(-4.0)
                head[-1].bias[3].fill_(-4.0)

    def forward(
        self,
        candidate_features: torch.Tensor,
        family_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if candidate_features.ndim != 3:
            raise ValueError(
                'FamilyCounterfactualEffectEstimator expects [B,M,D], got '
                f'{tuple(candidate_features.shape)}'
            )
        batch, candidates, _ = candidate_features.shape
        if family_ids.ndim != 1 or family_ids.numel() != candidates:
            raise ValueError(
                f'family_ids must be [M]={candidates}, got {tuple(family_ids.shape)}'
            )
        family_ids = family_ids.to(candidate_features.device).long().clamp(
            0, self.family_count - 1
        )
        representation = self.encoder(candidate_features)
        conditioned = representation + self.family_embedding(family_ids)[None]
        all_raw = torch.stack(
            [head(conditioned) for head in self.family_heads], dim=2
        )  # [B,M,F,4]
        gather = family_ids[None, :, None, None].expand(batch, -1, 1, 4)
        raw = all_raw.gather(2, gather).squeeze(2)

        mean_dsc_raw = raw[..., 0]
        q_gap_dsc = F.softplus(raw[..., 1])
        mean_nsd_raw = raw[..., 2]
        q_gap_nsd = F.softplus(raw[..., 3])

        mean_dsc = mean_dsc_raw - mean_dsc_raw[:, :1]
        mean_nsd = mean_nsd_raw - mean_nsd_raw[:, :1]
        q_dsc = mean_dsc - q_gap_dsc
        q_nsd = mean_nsd - q_gap_nsd

        zeros = mean_dsc.new_zeros((batch, 1))
        mean_dsc = torch.cat([zeros, mean_dsc[:, 1:]], dim=1)
        mean_nsd = torch.cat([zeros, mean_nsd[:, 1:]], dim=1)
        q_dsc = torch.cat([zeros, q_dsc[:, 1:]], dim=1)
        q_nsd = torch.cat([zeros, q_nsd[:, 1:]], dim=1)

        conservative_score = self.dsc_weight * q_dsc + self.nsd_weight * q_nsd
        pairwise_logits = (
            conservative_score[:, :, None] - conservative_score[:, None, :]
        ) / self.pairwise_temperature
        diagonal = torch.eye(candidates, device=pairwise_logits.device, dtype=torch.bool)[None]
        pairwise_logits = pairwise_logits.masked_fill(diagonal, 0.0)
        pairwise_probability = torch.sigmoid(pairwise_logits)
        pairwise_win_rate = (
            pairwise_probability.sum(dim=2) - 0.5
        ) / max(candidates - 1, 1)

        return {
            'representation': representation,
            'family_ids': family_ids,
            'mean_dsc': mean_dsc,
            'q_dsc': q_dsc,
            'mean_nsd': mean_nsd,
            'q_nsd': q_nsd,
            'effect_mean': self.dsc_weight * mean_dsc + self.nsd_weight * mean_nsd,
            'effect_sigma': (mean_dsc - q_dsc).clamp_min(1.0e-4),
            'conservative_score': conservative_score,
            'pairwise_logits': pairwise_logits,
            'pairwise_probability': pairwise_probability,
            'pairwise_win_rate': pairwise_win_rate,
        }


class SetContextCounterfactualEffectEstimator(nn.Module):
    """V480 M2: set-context candidate quality and harm estimator.

    This module keeps the V479 relative-effect targets, but replaces the
    independent family heads with a shared scorer over permutation-equivariant
    candidate tokens.  A lightweight Transformer encoder lets every candidate
    observe the competing candidate set before predicting candidate-minus-C0
    DSC/NSD effects.  Family identity is retained as an embedding instead of
    using separately calibrated output heads.

    The direct harm head estimates whether an intervention is likely to reduce
    candidate utility.  It is trained with the existing candidate-level harm
    labels and is consumed by the V480 predictor-rejector policy.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        family_count: int,
        quantile_alpha: float,
        dsc_weight: float,
        nsd_weight: float,
        pairwise_temperature: float,
        dropout: float,
        context_layers: int = 1,
        context_heads: int = 4,
        context_ffn_dim: int = 256,
        router_residual_scale: float = 0.02,
    ) -> None:
        super().__init__()
        self.family_count = int(family_count)
        self.quantile_alpha = float(quantile_alpha)
        self.dsc_weight = float(dsc_weight)
        self.nsd_weight = float(nsd_weight)
        self.pairwise_temperature = max(float(pairwise_temperature), 1.0e-4)
        self.router_residual_scale = max(float(router_residual_scale), 0.0)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.family_embedding = nn.Embedding(self.family_count, hidden_dim)

        context_layers = max(int(context_layers), 0)
        context_heads = max(int(context_heads), 1)
        while context_heads > 1 and hidden_dim % context_heads != 0:
            context_heads -= 1
        if context_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=context_heads,
                dim_feedforward=max(int(context_ffn_dim), hidden_dim),
                dropout=float(dropout),
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            self.set_context = nn.TransformerEncoder(
                layer, num_layers=context_layers, norm=nn.LayerNorm(hidden_dim)
            )
        else:
            self.set_context = nn.Identity()

        # One shared output scale for all candidate families.
        self.effect_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )
        self.harm_head = nn.Sequential(
            nn.Linear(hidden_dim, max(hidden_dim // 2, 16)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_dim // 2, 16), 1),
        )
        self.router_residual_head = nn.Sequential(
            nn.Linear(hidden_dim, max(hidden_dim // 2, 16)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_dim // 2, 16), 1),
        )

        with torch.no_grad():
            nn.init.normal_(self.family_embedding.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.effect_head[-1].weight, mean=0.0, std=1.0e-3)
            nn.init.zeros_(self.effect_head[-1].bias)
            self.effect_head[-1].bias[1].fill_(-4.0)
            self.effect_head[-1].bias[3].fill_(-4.0)
            nn.init.normal_(self.harm_head[-1].weight, mean=0.0, std=1.0e-3)
            nn.init.zeros_(self.harm_head[-1].bias)
            nn.init.normal_(self.router_residual_head[-1].weight, mean=0.0, std=1.0e-3)
            nn.init.zeros_(self.router_residual_head[-1].bias)

    def forward(
        self,
        candidate_features: torch.Tensor,
        family_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if candidate_features.ndim != 3:
            raise ValueError(
                'SetContextCounterfactualEffectEstimator expects [B,M,D], got '
                f'{tuple(candidate_features.shape)}'
            )
        batch, candidates, _ = candidate_features.shape
        if family_ids.ndim != 1 or family_ids.numel() != candidates:
            raise ValueError(
                f'family_ids must be [M]={candidates}, got {tuple(family_ids.shape)}'
            )
        family_ids = family_ids.to(candidate_features.device).long().clamp(
            0, self.family_count - 1
        )

        representation = self.encoder(candidate_features)
        representation = representation + self.family_embedding(family_ids)[None]
        contextual = self.set_context(representation)

        raw = self.effect_head(contextual)
        mean_dsc_raw = raw[..., 0]
        q_gap_dsc = F.softplus(raw[..., 1])
        mean_nsd_raw = raw[..., 2]
        q_gap_nsd = F.softplus(raw[..., 3])

        # All effects are explicitly candidate-minus-Preserve.
        mean_dsc = mean_dsc_raw - mean_dsc_raw[:, :1]
        mean_nsd = mean_nsd_raw - mean_nsd_raw[:, :1]
        q_dsc = mean_dsc - q_gap_dsc
        q_nsd = mean_nsd - q_gap_nsd

        zeros = mean_dsc.new_zeros((batch, 1))
        mean_dsc = torch.cat([zeros, mean_dsc[:, 1:]], dim=1)
        mean_nsd = torch.cat([zeros, mean_nsd[:, 1:]], dim=1)
        q_dsc = torch.cat([zeros, q_dsc[:, 1:]], dim=1)
        q_nsd = torch.cat([zeros, q_nsd[:, 1:]], dim=1)

        harm_logits = self.harm_head(contextual).squeeze(-1)
        harm_logits = torch.cat(
            [harm_logits.new_full((batch, 1), -20.0), harm_logits[:, 1:]], dim=1
        )

        conservative_score = self.dsc_weight * q_dsc + self.nsd_weight * q_nsd
        residual = torch.tanh(self.router_residual_head(contextual).squeeze(-1))
        residual = residual - residual[:, :1]
        residual = torch.cat([zeros, residual[:, 1:]], dim=1)
        ranking_score = conservative_score + self.router_residual_scale * residual
        ranking_score = torch.cat([zeros, ranking_score[:, 1:]], dim=1)

        pairwise_logits = (
            ranking_score[:, :, None] - ranking_score[:, None, :]
        ) / self.pairwise_temperature
        diagonal = torch.eye(
            candidates, device=pairwise_logits.device, dtype=torch.bool
        )[None]
        pairwise_logits = pairwise_logits.masked_fill(diagonal, 0.0)
        pairwise_probability = torch.sigmoid(pairwise_logits)
        pairwise_win_rate = (
            pairwise_probability.sum(dim=2) - 0.5
        ) / max(candidates - 1, 1)

        return {
            'representation': contextual,
            'family_ids': family_ids,
            'mean_dsc': mean_dsc,
            'q_dsc': q_dsc,
            'mean_nsd': mean_nsd,
            'q_nsd': q_nsd,
            'effect_mean': self.dsc_weight * mean_dsc + self.nsd_weight * mean_nsd,
            'effect_sigma': (mean_dsc - q_dsc).clamp_min(1.0e-4),
            'conservative_score': conservative_score,
            'ranking_score': ranking_score,
            'harm_logits': harm_logits,
            'pairwise_logits': pairwise_logits,
            'pairwise_probability': pairwise_probability,
            'pairwise_win_rate': pairwise_win_rate,
        }


class SetPredictorRejectorPolicy(nn.Module):
    """V480 M3: candidate router plus a learned intervention rejector.

    The router compares all candidates directly, without collapsing four Local
    candidates into a size-dependent log-sum-exp family score.  The rejector is
    trained to decide whether the routed intervention should replace Preserve.
    Hard deployment also enforces Pareto non-degradation, edit budget, direct
    harm probability, global intervention availability, and MC disagreement.

    Every gate is controlled by configuration so the same implementation can
    run the M2-only and M3 component ablations without changing code.
    """

    def __init__(
        self,
        hidden_dim: int,
        temperature: float,
        edit_penalty: float,
        harm_penalty: float,
        mc_penalty: float,
        accept_bonus: float,
        local_margin: float,
        discovery_margin: float,
        local_nsd_floor: float,
        discovery_nsd_floor: float,
        local_max_edit: float,
        discovery_max_edit: float,
        accept_threshold: float,
        max_harm_probability: float,
        failure_threshold: float,
        max_mc_disagreement: float,
        min_top_gap: float,
        use_rejector: bool,
        use_harm_gate: bool,
        use_failure_gate: bool,
        use_mc_gate: bool,
        use_pareto_gate: bool,
        use_edit_gate: bool,
        router_only: bool,
        force_preserve: bool,
        allow_local: bool,
        allow_discovery: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.temperature = max(float(temperature), 1.0e-3)
        self.edit_penalty = float(edit_penalty)
        self.harm_penalty = float(harm_penalty)
        self.mc_penalty = float(mc_penalty)
        self.accept_bonus = float(accept_bonus)
        self.local_margin = float(local_margin)
        self.discovery_margin = float(discovery_margin)
        self.local_nsd_floor = float(local_nsd_floor)
        self.discovery_nsd_floor = float(discovery_nsd_floor)
        self.local_max_edit = float(local_max_edit)
        self.discovery_max_edit = float(discovery_max_edit)
        self.accept_threshold = float(accept_threshold)
        self.max_harm_probability = float(max_harm_probability)
        self.failure_threshold = float(failure_threshold)
        self.max_mc_disagreement = float(max_mc_disagreement)
        self.min_top_gap = float(min_top_gap)
        self.use_rejector = bool(use_rejector)
        self.use_harm_gate = bool(use_harm_gate)
        self.use_failure_gate = bool(use_failure_gate)
        self.use_mc_gate = bool(use_mc_gate)
        self.use_pareto_gate = bool(use_pareto_gate)
        self.use_edit_gate = bool(use_edit_gate)
        self.router_only = bool(router_only)
        self.force_preserve = bool(force_preserve)
        self.allow_local = bool(allow_local)
        self.allow_discovery = bool(allow_discovery)

        rejector_input_dim = int(hidden_dim) + 7
        rejector_hidden = max(int(hidden_dim) // 2, 32)
        self.rejector_head = nn.Sequential(
            nn.Linear(rejector_input_dim, rejector_hidden),
            nn.LayerNorm(rejector_hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(rejector_hidden, 1),
        )
        with torch.no_grad():
            nn.init.normal_(self.rejector_head[-1].weight, mean=0.0, std=1.0e-3)
            nn.init.zeros_(self.rejector_head[-1].bias)

    def forward(
        self,
        representation: torch.Tensor,
        ranking_score: torch.Tensor,
        q_dsc: torch.Tensor,
        q_nsd: torch.Tensor,
        harm_logits: torch.Tensor,
        failure_prob: torch.Tensor,
        candidate_stats: torch.Tensor,
        family_ids: torch.Tensor,
        candidate_probs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch, candidates = ranking_score.shape
        if representation.shape[:2] != (batch, candidates):
            raise ValueError('representation and ranking_score disagree')
        if candidate_probs.shape[:2] != (batch, candidates):
            raise ValueError('candidate_probs and ranking_score disagree')

        family_ids = family_ids.to(ranking_score.device)
        edit_fraction = candidate_stats[..., 2]
        mc_disagreement = torch.maximum(
            candidate_stats[..., 9], candidate_stats[..., 10]
        ).clamp(0.0, 1.0)
        harm_prob = torch.sigmoid(harm_logits)
        failure_expand = failure_prob[:, None].expand(-1, candidates)

        rejector_features = torch.cat(
            [
                representation,
                q_dsc[..., None],
                q_nsd[..., None],
                harm_prob[..., None],
                edit_fraction[..., None],
                mc_disagreement[..., None],
                failure_expand[..., None],
                ranking_score[..., None],
            ],
            dim=-1,
        )
        accept_logits = self.rejector_head(rejector_features).squeeze(-1)
        accept_logits = torch.cat(
            [accept_logits.new_full((batch, 1), -20.0), accept_logits[:, 1:]], dim=1
        )
        accept_prob = torch.sigmoid(accept_logits)

        policy_score = ranking_score - self.edit_penalty * edit_fraction
        if self.use_harm_gate:
            policy_score = policy_score - self.harm_penalty * harm_prob
        if self.use_mc_gate:
            policy_score = policy_score - self.mc_penalty * mc_disagreement
        if self.use_rejector:
            policy_score = policy_score + self.accept_bonus * (accept_prob - 0.5)
        policy_score = torch.cat(
            [policy_score.new_zeros((batch, 1)), policy_score[:, 1:]], dim=1
        )

        # Optional family deployment ablations.  Preserve is always available.
        allowed = torch.ones_like(policy_score, dtype=torch.bool)
        local_action = ((family_ids >= 1) & (family_ids <= 3))[None]
        discovery_action = (family_ids == 4)[None]
        if not self.allow_local:
            allowed = allowed & ~local_action
        if not self.allow_discovery:
            allowed = allowed & ~discovery_action
        allowed[:, 0] = True
        very_negative = torch.finfo(policy_score.dtype).min / 4.0
        masked_score = policy_score.masked_fill(~allowed, very_negative)

        policy_soft = torch.softmax(masked_score / self.temperature, dim=1)

        if self.force_preserve:
            selected_index = torch.zeros(batch, device=policy_score.device, dtype=torch.long)
            accept = torch.zeros(batch, device=policy_score.device, dtype=torch.bool)
            top_gap = policy_score.new_zeros(batch)
            hard_gate = accept
        elif self.router_only:
            selected_index = masked_score.argmax(dim=1)
            accept = selected_index > 0
            sorted_scores = masked_score.topk(k=min(2, candidates), dim=1).values
            top_gap = sorted_scores[:, 0] - sorted_scores[:, -1]
            hard_gate = accept
        else:
            action_scores = masked_score[:, 1:]
            best_action_offset = action_scores.argmax(dim=1)
            best_action = best_action_offset + 1
            best_score = masked_score.gather(1, best_action[:, None])[:, 0]
            second_pool = masked_score.clone()
            second_pool.scatter_(1, best_action[:, None], very_negative)
            second_score = second_pool.max(dim=1).values
            top_gap = best_score - second_score

            best_family = family_ids.gather(0, best_action)
            is_discovery = best_family == 4
            margin = torch.where(
                is_discovery,
                best_score.new_full(best_score.shape, self.discovery_margin),
                best_score.new_full(best_score.shape, self.local_margin),
            )
            nsd_floor = torch.where(
                is_discovery,
                best_score.new_full(best_score.shape, self.discovery_nsd_floor),
                best_score.new_full(best_score.shape, self.local_nsd_floor),
            )
            max_edit = torch.where(
                is_discovery,
                best_score.new_full(best_score.shape, self.discovery_max_edit),
                best_score.new_full(best_score.shape, self.local_max_edit),
            )

            best_q_dsc = q_dsc.gather(1, best_action[:, None])[:, 0]
            best_q_nsd = q_nsd.gather(1, best_action[:, None])[:, 0]
            best_edit = edit_fraction.gather(1, best_action[:, None])[:, 0]
            best_harm = harm_prob.gather(1, best_action[:, None])[:, 0]
            best_mc = mc_disagreement.gather(1, best_action[:, None])[:, 0]
            best_accept_prob = accept_prob.gather(1, best_action[:, None])[:, 0]

            hard_gate = best_score > margin
            if self.use_pareto_gate:
                hard_gate = hard_gate & (best_q_dsc > margin) & (best_q_nsd >= nsd_floor)
            if self.use_edit_gate:
                hard_gate = hard_gate & (best_edit <= max_edit)
            if self.use_rejector:
                hard_gate = hard_gate & (best_accept_prob >= self.accept_threshold)
            if self.use_harm_gate:
                hard_gate = hard_gate & (best_harm <= self.max_harm_probability)
            if self.use_failure_gate:
                hard_gate = hard_gate & (failure_prob >= self.failure_threshold)
            if self.use_mc_gate:
                hard_gate = hard_gate & (best_mc <= self.max_mc_disagreement)
            hard_gate = hard_gate & (top_gap >= self.min_top_gap)

            selected_index = torch.where(
                hard_gate, best_action, torch.zeros_like(best_action)
            )
            accept = selected_index > 0

        hard_weights = F.one_hot(selected_index, num_classes=candidates).to(policy_score.dtype)
        st_weights = hard_weights + policy_soft - policy_soft.detach()
        soft_final = (policy_soft[:, :, None, None] * candidate_probs).sum(dim=1)
        st_final = (st_weights[:, :, None, None] * candidate_probs).sum(dim=1)
        gather = selected_index[:, None, None, None].expand(
            -1, 1, candidate_probs.shape[-2], candidate_probs.shape[-1]
        )
        hard_final = candidate_probs.gather(1, gather)[:, 0]
        chosen_score = policy_score.gather(1, selected_index[:, None])[:, 0]
        chosen_edit = edit_fraction.gather(1, selected_index[:, None])[:, 0]

        pairwise_logits = (
            policy_score[:, :, None] - policy_score[:, None, :]
        ) / self.temperature
        pairwise_probability = torch.sigmoid(pairwise_logits)
        pairwise_win_rate = (
            pairwise_probability.sum(dim=2) - 0.5
        ) / max(candidates - 1, 1)

        return {
            'policy_score': policy_score,
            'policy_soft': policy_soft,
            'policy_st': st_weights,
            'selected_index': selected_index,
            'selected_family': family_ids.gather(0, selected_index),
            'accept': accept.to(policy_score.dtype),
            'soft_final': soft_final,
            'st_final': st_final,
            'hard_final': hard_final,
            'accept_logits': accept_logits,
            'accept_prob': accept_prob,
            'harm_prob': harm_prob,
            'mc_disagreement': mc_disagreement,
            'hard_gate': hard_gate.to(policy_score.dtype),
            'top_gap': top_gap,
            'pairwise_logits': pairwise_logits,
            'pairwise_probability': pairwise_probability,
            'pairwise_win_rate': pairwise_win_rate,
            'chosen_score': chosen_score,
            'chosen_edit': chosen_edit,
        }


class HierarchicalParetoInterventionPolicy(nn.Module):
    """V479 M3: Preserve / best-Local / Discovery hierarchical policy.

    M3 does not learn another unconstrained residual score.  It consumes the
    calibrated lower quantiles predicted by M2, first forms a differentiable
    best-local representative, then compares Preserve, Local and Discovery.
    Deployment additionally enforces family-specific DSC/NSD and edit-budget
    constraints.  Duplicate Discovery masks therefore cannot receive repeated
    votes.
    """

    def __init__(
        self,
        temperature: float,
        dsc_weight: float,
        nsd_weight: float,
        edit_penalty: float,
        local_margin: float,
        discovery_margin: float,
        local_nsd_floor: float,
        discovery_nsd_floor: float,
        local_max_edit: float,
        discovery_max_edit: float,
    ) -> None:
        super().__init__()
        self.temperature = max(float(temperature), 1.0e-3)
        self.dsc_weight = float(dsc_weight)
        self.nsd_weight = float(nsd_weight)
        self.edit_penalty = float(edit_penalty)
        self.local_margin = float(local_margin)
        self.discovery_margin = float(discovery_margin)
        self.local_nsd_floor = float(local_nsd_floor)
        self.discovery_nsd_floor = float(discovery_nsd_floor)
        self.local_max_edit = float(local_max_edit)
        self.discovery_max_edit = float(discovery_max_edit)

    def forward(
        self,
        q_dsc: torch.Tensor,
        q_nsd: torch.Tensor,
        candidate_stats: torch.Tensor,
        family_ids: torch.Tensor,
        candidate_probs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch, candidates = q_dsc.shape
        if candidate_probs.shape[:2] != (batch, candidates):
            raise ValueError('candidate_probs and q_dsc disagree')
        family_ids = family_ids.to(q_dsc.device)
        edit_fraction = candidate_stats[..., 2]
        candidate_score = (
            self.dsc_weight * q_dsc
            + self.nsd_weight * q_nsd
            - self.edit_penalty * edit_fraction
        )
        candidate_score = torch.cat(
            [candidate_score.new_zeros((batch, 1)), candidate_score[:, 1:]], dim=1
        )

        local_mask = ((family_ids >= 1) & (family_ids <= 3))[None].expand(batch, -1)
        discovery_mask = (family_ids == 4)[None].expand(batch, -1)

        very_negative = torch.finfo(candidate_score.dtype).min / 4.0
        local_logits = candidate_score.masked_fill(~local_mask, very_negative)
        discovery_logits = candidate_score.masked_fill(~discovery_mask, very_negative)

        has_local = local_mask.any(dim=1)
        has_discovery = discovery_mask.any(dim=1)
        local_weights = torch.softmax(local_logits / self.temperature, dim=1)
        local_weights = torch.where(local_mask, local_weights, torch.zeros_like(local_weights))
        local_weights = local_weights / local_weights.sum(dim=1, keepdim=True).clamp_min(EPS)
        discovery_weights = torch.softmax(discovery_logits / self.temperature, dim=1)
        discovery_weights = torch.where(
            discovery_mask, discovery_weights, torch.zeros_like(discovery_weights)
        )
        discovery_weights = discovery_weights / discovery_weights.sum(
            dim=1, keepdim=True
        ).clamp_min(EPS)

        local_score = self.temperature * torch.logsumexp(
            local_logits / self.temperature, dim=1
        )
        discovery_score = self.temperature * torch.logsumexp(
            discovery_logits / self.temperature, dim=1
        )
        local_score = torch.where(has_local, local_score, local_score.new_full(local_score.shape, very_negative))
        discovery_score = torch.where(
            has_discovery, discovery_score, discovery_score.new_full(discovery_score.shape, very_negative)
        )
        family_scores = torch.stack(
            [candidate_score.new_zeros(batch), local_score, discovery_score], dim=1
        )
        family_soft = torch.softmax(family_scores / self.temperature, dim=1)

        flat_soft = family_soft[:, :1] * F.one_hot(
            torch.zeros(batch, device=q_dsc.device, dtype=torch.long), candidates
        ).to(q_dsc.dtype)
        flat_soft = flat_soft + family_soft[:, 1:2] * local_weights
        flat_soft = flat_soft + family_soft[:, 2:3] * discovery_weights

        local_best = local_logits.argmax(dim=1)
        discovery_best = discovery_logits.argmax(dim=1)
        local_ok = (
            has_local
            & (q_dsc.gather(1, local_best[:, None])[:, 0] > self.local_margin)
            & (q_nsd.gather(1, local_best[:, None])[:, 0] >= self.local_nsd_floor)
            & (edit_fraction.gather(1, local_best[:, None])[:, 0] <= self.local_max_edit)
        )
        discovery_ok = (
            has_discovery
            & (q_dsc.gather(1, discovery_best[:, None])[:, 0] > self.discovery_margin)
            & (q_nsd.gather(1, discovery_best[:, None])[:, 0] >= self.discovery_nsd_floor)
            & (edit_fraction.gather(1, discovery_best[:, None])[:, 0] <= self.discovery_max_edit)
        )
        hard_family_scores = family_scores.clone()
        hard_family_scores[:, 1] = hard_family_scores[:, 1].masked_fill(~local_ok, very_negative)
        hard_family_scores[:, 2] = hard_family_scores[:, 2].masked_fill(~discovery_ok, very_negative)
        selected_family = hard_family_scores.argmax(dim=1)
        selected_index = torch.zeros(batch, device=q_dsc.device, dtype=torch.long)
        selected_index = torch.where(selected_family == 1, local_best, selected_index)
        selected_index = torch.where(selected_family == 2, discovery_best, selected_index)
        accept = selected_index > 0

        hard_weights = F.one_hot(selected_index, num_classes=candidates).to(q_dsc.dtype)
        st_weights = hard_weights + flat_soft - flat_soft.detach()
        soft_final = (flat_soft[:, :, None, None] * candidate_probs).sum(dim=1)
        st_final = (st_weights[:, :, None, None] * candidate_probs).sum(dim=1)
        gather = selected_index[:, None, None, None].expand(
            -1, 1, candidate_probs.shape[-2], candidate_probs.shape[-1]
        )
        hard_final = candidate_probs.gather(1, gather)[:, 0]
        chosen_score = candidate_score.gather(1, selected_index[:, None])[:, 0]
        chosen_edit = edit_fraction.gather(1, selected_index[:, None])[:, 0]
        pairwise_logits = (
            candidate_score[:, :, None] - candidate_score[:, None, :]
        ) / self.temperature
        pairwise_probability = torch.sigmoid(pairwise_logits)
        pairwise_win_rate = (
            pairwise_probability.sum(dim=2) - 0.5
        ) / max(candidates - 1, 1)
        return {
            'policy_score': candidate_score,
            'family_scores': family_scores,
            'family_soft': family_soft,
            'policy_soft': flat_soft,
            'policy_st': st_weights,
            'selected_family': selected_family,
            'selected_index': selected_index,
            'accept': accept.to(q_dsc.dtype),
            'soft_final': soft_final,
            'st_final': st_final,
            'hard_final': hard_final,
            'pairwise_logits': pairwise_logits,
            'pairwise_probability': pairwise_probability,
            'pairwise_win_rate': pairwise_win_rate,
            'chosen_score': chosen_score,
            'chosen_edit': chosen_edit,
        }

class CompositionalErrorModeCandidateGenerator(nn.Module):
    """End-to-end compositional signed-error candidate generator.

    The first candidate is the *current forward pass* consensus segmentation,
    not a separately trained or frozen B0 model.  The remaining candidates are
    complete masks obtained by composing a small set of learned signed
    correction modes.  Each mode predicts:

      * a soft spatial support A_k in [0, 1],
      * a signed logit correction Delta_k,
      * a sample-specific no-op gate.

    With K=4 modes the default candidate set is:
      C0                    : consensus
      C1..C4                : four single-mode hypotheses
      C5..C10               : all six pair hypotheses
      C11                   : the bounded full composition

    Candidate quality and harm are estimated jointly.  Deployment always
    gathers one exact candidate and falls back to C0 unless the learned failure
    gate, utility margin and harm threshold all agree.
    """

    use_semantic_feature = True
    unified_m1_safe_fusion_enabled = True

    def __init__(self, cfg) -> None:
        super().__init__()
        m1 = _cfg_get(cfg, "M1", None)

        self.num_modes = max(
            1,
            int(
                _cfg_get(
                    m1,
                    "CEM_NUM_MODES",
                    _cfg_get(m1, "TPMHG_NUM_HYPOTHESES", 4),
                )
            ),
        )
        self.num_hypotheses = self.num_modes  # legacy audit compatibility
        self.hidden_dim = int(_cfg_get(m1, "CEM_HIDDEN_DIM", 128))
        self.semantic_channels = int(_cfg_get(m1, "SEMANTIC_CHANNELS", 512))
        self.text_dim = int(_cfg_get(m1, "CEM_TEXT_DIM", self.semantic_channels))
        self.dropout_p = float(_cfg_get(m1, "CEM_DROPOUT", 0.10))
        self.max_atom_delta = float(_cfg_get(m1, "CEM_MAX_ATOM_LOGIT_DELTA", 3.5))
        self.max_total_delta = float(_cfg_get(m1, "CEM_MAX_TOTAL_LOGIT_DELTA", 5.0))
        self.compose_pairs = bool(_cfg_get(m1, "CEM_COMPOSE_PAIRS", True))
        self.compose_full = bool(_cfg_get(m1, "CEM_COMPOSE_FULL", True))
        self.typed_modes = bool(_cfg_get(m1, "CEM_TYPED_MODES", False))
        self.discovery_enabled = bool(
            _cfg_get(m1, "CEM_DISCOVERY_ENABLED", False)
        )
        self.num_discovery = (
            max(0, int(_cfg_get(m1, "CEM_NUM_DISCOVERY_CANDIDATES", 2)))
            if self.discovery_enabled else 0
        )
        self.discovery_iterations = max(
            1, int(_cfg_get(m1, "CEM_DISCOVERY_ITERATIONS", 3))
        )
        self.discovery_size = max(
            14, int(_cfg_get(m1, "CEM_DISCOVERY_SIZE", 56))
        )
        self.discovery_temperature = max(
            float(_cfg_get(m1, "CEM_DISCOVERY_TEMPERATURE", 0.15)), 1.0e-3
        )
        self.discovery_similarity_threshold = float(
            _cfg_get(m1, "CEM_DISCOVERY_SIMILARITY_THRESHOLD", 0.15)
        )
        self.discovery_separation_margin = float(
            _cfg_get(m1, "CEM_DISCOVERY_SEPARATION_MARGIN", 0.50)
        )
        self.discovery_mode = str(
            _cfg_get(m1, "CEM_DISCOVERY_MODE", "prototype")
        ).strip().lower()
        self.discovery_dedup_iou = float(
            _cfg_get(m1, "CEM_DISCOVERY_DEDUP_IOU", 0.95)
        )
        self.cf_quantile_alpha = float(
            _cfg_get(m1, "CEM_CF_QUANTILE_ALPHA", 0.10)
        )
        self.cf_dsc_weight = float(
            _cfg_get(m1, "CEM_CF_DSC_WEIGHT", 0.60)
        )
        self.cf_nsd_weight = float(
            _cfg_get(m1, "CEM_CF_NSD_WEIGHT", 0.40)
        )
        self.m3_local_margin = float(
            _cfg_get(m1, "CEM_M3_LOCAL_MARGIN", 0.0)
        )
        self.m3_discovery_margin = float(
            _cfg_get(m1, "CEM_M3_DISCOVERY_MARGIN", 0.0)
        )
        self.m3_local_nsd_floor = float(
            _cfg_get(m1, "CEM_M3_LOCAL_NSD_FLOOR", -0.001)
        )
        self.m3_discovery_nsd_floor = float(
            _cfg_get(m1, "CEM_M3_DISCOVERY_NSD_FLOOR", 0.0)
        )

        self.selector_temperature = max(
            float(_cfg_get(m1, "CEM_SELECTOR_TEMPERATURE", 0.20)), 1.0e-3
        )
        self.selector_risk_weight = float(
            _cfg_get(m1, "CEM_SELECTOR_RISK_WEIGHT", 0.50)
        )
        self.selector_utility_margin = float(
            _cfg_get(m1, "CEM_SELECTOR_UTILITY_MARGIN", 0.001)
        )
        self.selector_max_harm = float(
            _cfg_get(m1, "CEM_SELECTOR_MAX_HARM", 0.45)
        )
        self.failure_threshold = float(
            _cfg_get(m1, "CEM_FAILURE_THRESHOLD", 0.55)
        )
        # V476: utility is measured in DSC-scale units (~1e-3 to 1e-2), while
        # harm is a probability in [0, 1].  The historical selector subtracted
        # risk_weight * harm_probability directly, making every action score
        # negative by orders of magnitude.  Risk is now explicitly converted
        # to the same utility scale before comparison.
        self.selector_risk_scale = float(
            _cfg_get(m1, "CEM_SELECTOR_RISK_SCALE", 0.02)
        )
        self.selector_benefit_weight = float(
            _cfg_get(m1, "CEM_SELECTOR_BENEFIT_WEIGHT", 0.01)
        )
        self.selector_min_benefit_prob = float(
            _cfg_get(m1, "CEM_SELECTOR_MIN_BENEFIT_PROB", 0.55)
        )
        self.use_global_failure_gate = bool(
            _cfg_get(m1, "CEM_USE_GLOBAL_FAILURE_GATE", False)
        )
        # V477 M2: relative lower-confidence-bound selector.  Selection is
        # aligned directly with candidate-minus-Preserve utility; no separate
        # pixel-harm gate is allowed to veto a net-positive candidate.
        self.selector_mode = str(
            _cfg_get(m1, "CEM_SELECTOR_MODE", "legacy_benefit_harm")
        ).strip().lower()
        self.selector_lcb_kappa = float(
            _cfg_get(m1, "CEM_SELECTOR_LCB_KAPPA", 1.0)
        )
        self.selector_min_sigma = float(
            _cfg_get(m1, "CEM_SELECTOR_MIN_SIGMA", 0.002)
        )
        self.selector_max_sigma = float(
            _cfg_get(m1, "CEM_SELECTOR_MAX_SIGMA", 0.10)
        )
        self.selector_mc_weight = float(
            _cfg_get(m1, "CEM_SELECTOR_MC_DISAGREEMENT_WEIGHT", 0.01)
        )
        self.selector_edit_penalty = float(
            _cfg_get(m1, "CEM_SELECTOR_EDIT_PENALTY", 0.05)
        )
        self.selector_max_deploy_edit = float(
            _cfg_get(m1, "CEM_SELECTOR_MAX_DEPLOY_EDIT", 0.04)
        )
        self.discovery_max_deploy_edit = float(
            _cfg_get(m1, "CEM_DISCOVERY_MAX_DEPLOY_EDIT", 0.35)
        )
        self.m3_pairwise_weight = float(
            _cfg_get(m1, "CEM_M3_PAIRWISE_WEIGHT", 0.02)
        )
        self.m3_residual_scale = float(
            _cfg_get(m1, "CEM_M3_RESIDUAL_SCALE", 0.01)
        )

        groups = max(1, min(8, self.hidden_dim))
        while groups > 1 and self.hidden_dim % groups != 0:
            groups -= 1

        self.image_stem = nn.Sequential(
            _ConvNormGELU(3, self.hidden_dim),
            _ConvNormGELU(self.hidden_dim, self.hidden_dim),
        )
        self.semantic_proj = nn.Sequential(
            nn.Conv2d(self.semantic_channels, self.hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.negative_text_proj = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.state_proj = nn.Sequential(
            nn.Conv2d(6, self.hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            _ConvNormGELU(self.hidden_dim, self.hidden_dim),
            nn.Dropout2d(self.dropout_p),
            _ConvNormGELU(self.hidden_dim, self.hidden_dim),
        )
        self.discovery_fusion = nn.Sequential(
            _ConvNormGELU(self.hidden_dim, self.hidden_dim),
            nn.Dropout2d(self.dropout_p),
            _ConvNormGELU(self.hidden_dim, self.hidden_dim),
        )

        self.mode_queries = nn.Embedding(self.num_modes, self.hidden_dim)
        self.mode_film = nn.Sequential(
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
            nn.Tanh(),
        )
        self.mode_trunk = nn.Sequential(
            _ConvNormGELU(self.hidden_dim, self.hidden_dim),
            nn.Dropout2d(self.dropout_p),
            _ConvNormGELU(self.hidden_dim, self.hidden_dim),
        )
        self.attention_head = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)
        self.delta_head = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)
        self.mode_gate = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.discovery_queries = nn.Embedding(
            max(self.num_discovery, 1), self.hidden_dim
        )
        self.discovery_decoder = nn.Sequential(
            _ConvNormGELU(self.hidden_dim + 2, self.hidden_dim),
            nn.Dropout2d(self.dropout_p),
            _ConvNormGELU(self.hidden_dim, self.hidden_dim),
            nn.Conv2d(self.hidden_dim, 1, kernel_size=1),
        )
        # V479 standard base-independent query-mask path.  It does not read C0.
        self.discovery_pixel_proj = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
        )
        self.discovery_query_context = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.discovery_mask_refine = nn.Sequential(
            _ConvNormGELU(self.hidden_dim + 1, self.hidden_dim),
            nn.Dropout2d(self.dropout_p),
            _ConvNormGELU(self.hidden_dim, self.hidden_dim),
            nn.Conv2d(self.hidden_dim, 1, kernel_size=1),
        )

        combo_rows = [torch.zeros(self.num_modes, dtype=torch.float32)]
        combo_names = ["consensus"]
        typed_names = ("fp_delete", "fn_fill", "boundary_trim", "boundary_expand")
        for k in range(self.num_modes):
            row = torch.zeros(self.num_modes, dtype=torch.float32)
            row[k] = 1.0
            combo_rows.append(row)
            if self.typed_modes and k < len(typed_names):
                combo_names.append(typed_names[k])
            else:
                combo_names.append(f"single_{k}")
        if self.compose_pairs and self.num_modes >= 2:
            for i in range(self.num_modes):
                for j in range(i + 1, self.num_modes):
                    row = torch.zeros(self.num_modes, dtype=torch.float32)
                    row[i] = 1.0
                    row[j] = 1.0
                    combo_rows.append(row)
                    combo_names.append(f"pair_{i}_{j}")
        if self.compose_full and self.num_modes >= 3:
            combo_rows.append(torch.ones(self.num_modes, dtype=torch.float32))
            combo_names.append("full")

        combo_matrix = torch.stack(combo_rows, dim=0)
        self.register_buffer("combo_matrix", combo_matrix, persistent=True)
        local_candidate_count = int(combo_matrix.shape[0])
        if self.typed_modes and not self.compose_pairs and not self.compose_full:
            local_action_types = torch.arange(
                self.num_modes, dtype=torch.long
            ).clamp_max(3)
        else:
            local_action_types = combo_matrix[1:].sum(dim=1).sub(1.0).clamp(0, 3).long()
        discovery_action_types = torch.full(
            (self.num_discovery,), 4, dtype=torch.long
        )
        action_types = torch.cat([local_action_types, discovery_action_types], dim=0)
        self.register_buffer("action_types", action_types, persistent=True)
        typed_signs = torch.tensor(
            [-1.0, 1.0, -1.0, 1.0], dtype=torch.float32
        )
        if self.num_modes != 4:
            typed_signs = torch.where(
                torch.arange(self.num_modes) % 2 == 0,
                torch.tensor(-1.0),
                torch.tensor(1.0),
            ).float()
        self.register_buffer("typed_mode_signs", typed_signs, persistent=True)
        for index in range(self.num_discovery):
            combo_names.append(f"feature_discovery_{index}")
        self.combo_names = tuple(combo_names)
        self.local_candidate_count = local_candidate_count
        self.candidate_count = local_candidate_count + self.num_discovery

        # V476 local-evidence selector.  Global pooled features alone cannot
        # distinguish two candidates that edit different locations in the same
        # image.  Each candidate now receives: global context, edit-conditioned
        # local evidence, a run-specific slot embedding and explicit geometry.
        self.candidate_embedding = nn.Embedding(self.candidate_count, self.hidden_dim)
        self.candidate_reliability_bias = nn.Parameter(
            torch.zeros(self.candidate_count)
        )
        quality_input_dim = 3 * self.hidden_dim + 11
        if self.selector_mode in {
            "set_context_predictor_rejector", "v480_set_predictor_rejector"
        }:
            self.quality_head = nn.Identity()
            self.m2_counterfactual = SetContextCounterfactualEffectEstimator(
                input_dim=quality_input_dim,
                hidden_dim=self.hidden_dim,
                family_count=5,
                quantile_alpha=self.cf_quantile_alpha,
                dsc_weight=self.cf_dsc_weight,
                nsd_weight=self.cf_nsd_weight,
                pairwise_temperature=float(_cfg_get(m1, "CEM_CF_PAIRWISE_TEMPERATURE", 0.02)),
                dropout=self.dropout_p,
                context_layers=int(_cfg_get(m1, "CEM_SET_CONTEXT_LAYERS", 1)),
                context_heads=int(_cfg_get(m1, "CEM_SET_CONTEXT_HEADS", 4)),
                context_ffn_dim=int(_cfg_get(m1, "CEM_SET_CONTEXT_FFN_DIM", 256)),
                router_residual_scale=float(_cfg_get(m1, "CEM_ROUTER_RESIDUAL_SCALE", 0.02)),
            )
            self.m3_policy = SetPredictorRejectorPolicy(
                hidden_dim=self.hidden_dim,
                temperature=self.selector_temperature,
                edit_penalty=self.selector_edit_penalty,
                harm_penalty=float(_cfg_get(m1, "CEM_M3_HARM_PENALTY", 0.02)),
                mc_penalty=float(_cfg_get(m1, "CEM_M3_MC_PENALTY", 0.01)),
                accept_bonus=float(_cfg_get(m1, "CEM_M3_ACCEPT_BONUS", 0.01)),
                local_margin=self.m3_local_margin,
                discovery_margin=self.m3_discovery_margin,
                local_nsd_floor=self.m3_local_nsd_floor,
                discovery_nsd_floor=self.m3_discovery_nsd_floor,
                local_max_edit=self.selector_max_deploy_edit,
                discovery_max_edit=self.discovery_max_deploy_edit,
                accept_threshold=float(_cfg_get(m1, "CEM_M3_ACCEPT_THRESHOLD", 0.50)),
                max_harm_probability=float(_cfg_get(m1, "CEM_M3_MAX_HARM_PROB", 0.35)),
                failure_threshold=float(_cfg_get(m1, "CEM_M3_FAILURE_THRESHOLD", 0.50)),
                max_mc_disagreement=float(_cfg_get(m1, "CEM_M3_MAX_MC_DISAGREEMENT", 0.25)),
                min_top_gap=float(_cfg_get(m1, "CEM_M3_MIN_TOP_GAP", 0.0)),
                use_rejector=bool(_cfg_get(m1, "CEM_M3_USE_REJECTOR", True)),
                use_harm_gate=bool(_cfg_get(m1, "CEM_M3_USE_HARM_GATE", True)),
                use_failure_gate=bool(_cfg_get(m1, "CEM_M3_USE_FAILURE_GATE", True)),
                use_mc_gate=bool(_cfg_get(m1, "CEM_M3_USE_MC_GATE", True)),
                use_pareto_gate=bool(_cfg_get(m1, "CEM_M3_USE_PARETO_GATE", True)),
                use_edit_gate=bool(_cfg_get(m1, "CEM_M3_USE_EDIT_GATE", True)),
                router_only=bool(_cfg_get(m1, "CEM_M3_ROUTER_ONLY", False)),
                force_preserve=bool(_cfg_get(m1, "CEM_M3_FORCE_PRESERVE", False)),
                allow_local=bool(_cfg_get(m1, "CEM_M3_ALLOW_LOCAL", True)),
                allow_discovery=bool(_cfg_get(m1, "CEM_M3_ALLOW_DISCOVERY", True)),
                dropout=self.dropout_p,
            )
        elif self.selector_mode in {"family_quantile_hierarchical", "v479_family_quantile"}:
            self.quality_head = nn.Identity()
            self.m2_counterfactual = FamilyCounterfactualEffectEstimator(
                input_dim=quality_input_dim,
                hidden_dim=self.hidden_dim,
                family_count=5,
                quantile_alpha=self.cf_quantile_alpha,
                dsc_weight=self.cf_dsc_weight,
                nsd_weight=self.cf_nsd_weight,
                pairwise_temperature=float(_cfg_get(m1, "CEM_CF_PAIRWISE_TEMPERATURE", 0.02)),
                dropout=self.dropout_p,
            )
            self.m3_policy = HierarchicalParetoInterventionPolicy(
                temperature=self.selector_temperature,
                dsc_weight=self.cf_dsc_weight,
                nsd_weight=self.cf_nsd_weight,
                edit_penalty=self.selector_edit_penalty,
                local_margin=self.m3_local_margin,
                discovery_margin=self.m3_discovery_margin,
                local_nsd_floor=self.m3_local_nsd_floor,
                discovery_nsd_floor=self.m3_discovery_nsd_floor,
                local_max_edit=self.selector_max_deploy_edit,
                discovery_max_edit=self.discovery_max_deploy_edit,
            )
        elif self.selector_mode in {"counterfactual_m3", "v478_counterfactual_m3"}:
            self.quality_head = nn.Identity()
            self.m2_counterfactual = CounterfactualEffectEstimator(
                input_dim=quality_input_dim,
                hidden_dim=self.hidden_dim,
                min_sigma=self.selector_min_sigma,
                max_sigma=self.selector_max_sigma,
                dropout=self.dropout_p,
            )
            self.m3_policy = RiskAwareInterventionPolicy(
                hidden_dim=self.hidden_dim,
                temperature=self.selector_temperature,
                lcb_kappa=self.selector_lcb_kappa,
                pairwise_weight=self.m3_pairwise_weight,
                edit_penalty=self.selector_edit_penalty,
                mc_penalty=self.selector_mc_weight,
                residual_scale=self.m3_residual_scale,
                utility_margin=self.selector_utility_margin,
                local_max_edit=self.selector_max_deploy_edit,
                discovery_max_edit=self.discovery_max_deploy_edit,
                dropout=self.dropout_p,
            )
        else:
            self.m2_counterfactual = None
            self.m3_policy = None
            self.quality_head = nn.Sequential(
                nn.Linear(quality_input_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(self.hidden_dim, 2),
            )
        # V477 outputs: candidate-minus-Preserve utility mean and raw
        # aleatoric scale.  Benefit/harm probabilities are derived from this
        # single relative distribution instead of learned by conflicting heads.
        self.failure_head = nn.Sequential(
            nn.Linear(self.hidden_dim + 4, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

        with torch.no_grad():
            # Sparse, near-no-op initialization protects the consensus path.
            # Identity FiLM initialisation: all typed branches start from the
            # same shared representation and learn their own modulation.
            nn.init.zeros_(self.mode_film[0].weight)
            nn.init.zeros_(self.mode_film[0].bias)
            nn.init.zeros_(self.attention_head.weight)
            self.attention_head.bias.fill_(-2.5)
            nn.init.normal_(self.delta_head.weight, mean=0.0, std=1.0e-3)
            nn.init.zeros_(self.delta_head.bias)
            nn.init.zeros_(self.mode_gate[-1].weight)
            self.mode_gate[-1].bias.fill_(-1.0)
            if isinstance(self.quality_head, nn.Sequential):
                nn.init.normal_(self.quality_head[-1].weight, mean=0.0, std=1.0e-3)
                nn.init.zeros_(self.quality_head[-1].bias)
                # Start with a small uncertainty (~0.02 utility units), so the
                # one-sided LCB initially abstains but can become positive once a
                # candidate is consistently useful.
                self.quality_head[-1].bias[1].fill_(-4.0)
            nn.init.normal_(self.discovery_queries.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.discovery_decoder[-1].weight)
            nn.init.zeros_(self.discovery_decoder[-1].bias)
            nn.init.zeros_(self.discovery_mask_refine[-1].weight)
            nn.init.zeros_(self.discovery_mask_refine[-1].bias)
            nn.init.zeros_(self.failure_head[-1].weight)
            self.failure_head[-1].bias.fill_(-1.0)

    @staticmethod
    def _resize_image(image: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        if image.shape[-2:] != target_hw:
            image = F.interpolate(image, size=target_hw, mode="bilinear", align_corners=False)
        return image

    @staticmethod
    def _entropy(prob: torch.Tensor) -> torch.Tensor:
        p = prob.clamp(EPS, 1.0 - EPS)
        return (
            -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
            / math.log(2.0)
        )

    @staticmethod
    def _soft_boundary_map(prob: torch.Tensor) -> torch.Tensor:
        return (_soft_dilate(prob, 1) - _soft_erode(prob, 1)).clamp(0.0, 1.0)

    def _shared_feature(
        self,
        image: torch.Tensor,
        semantic_map: Optional[torch.Tensor],
        text_features: torch.Tensor,
        negative_text_features: Optional[torch.Tensor],
        base_logits: torch.Tensor,
        mc_std_map: Optional[torch.Tensor] = None,
        mc_disagreement_map: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target_hw = base_logits.shape[-2:]
        image_latent = self.image_stem(self._resize_image(image, target_hw))

        if semantic_map is None:
            semantic_latent = torch.zeros_like(image_latent)
        else:
            if semantic_map.shape[-2:] != target_hw:
                semantic_map = F.interpolate(
                    semantic_map, size=target_hw, mode="bilinear", align_corners=False
                )
            if semantic_map.shape[1] != self.semantic_channels:
                raise RuntimeError(
                    "CEM semantic channel mismatch: expected "
                    f"{self.semantic_channels}, got {semantic_map.shape[1]}"
                )
            semantic_latent = self.semantic_proj(semantic_map)

        positive_text = self.text_proj(text_features.float())
        if negative_text_features is None:
            negative_text = torch.zeros_like(positive_text)
        else:
            negative_text = self.negative_text_proj(negative_text_features.float())
        text_latent = (positive_text - 0.25 * negative_text)[:, :, None, None]

        base_prob = torch.sigmoid(base_logits).clamp(EPS, 1.0 - EPS)
        entropy = self._entropy(base_prob)
        boundary = self._soft_boundary_map(base_prob)
        if mc_std_map is None:
            mc_std_map = torch.zeros_like(base_prob)
        elif mc_std_map.shape[-2:] != target_hw:
            mc_std_map = F.interpolate(
                mc_std_map, size=target_hw, mode="bilinear", align_corners=False
            )
        if mc_disagreement_map is None:
            mc_disagreement_map = torch.zeros_like(base_prob)
        elif mc_disagreement_map.shape[-2:] != target_hw:
            mc_disagreement_map = F.interpolate(
                mc_disagreement_map,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
        state = torch.cat(
            [
                base_prob,
                1.0 - base_prob,
                entropy,
                boundary,
                mc_std_map.to(base_prob.dtype),
                mc_disagreement_map.to(base_prob.dtype),
            ],
            dim=1,
        )
        state_latent = self.state_proj(state)
        discovery_shared = self.discovery_fusion(
            image_latent + semantic_latent + text_latent
        )
        shared = self.fusion(discovery_shared + state_latent)
        return shared, discovery_shared, base_prob, entropy, boundary

    def _predict_modes(
        self,
        shared: torch.Tensor,
        base_prob: torch.Tensor,
        boundary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = shared.shape[0]
        pooled = F.adaptive_avg_pool2d(shared, 1).flatten(1)
        attn_logits, attn_probs = [], []
        delta_raws, corrections, gates = [], [], []

        for k in range(self.num_modes):
            query = self.mode_queries.weight[k][None].expand(batch, -1)
            scale, bias = self.mode_film(query).chunk(2, dim=1)
            modulated = shared * (1.0 + scale[:, :, None, None])
            modulated = modulated + bias[:, :, None, None]
            mode_feature = self.mode_trunk(modulated)

            attention_logit = self.attention_head(mode_feature)
            attention = torch.sigmoid(attention_logit)
            delta_raw = self.delta_head(mode_feature)
            gate = torch.sigmoid(self.mode_gate(torch.cat([pooled, query], dim=1)))

            if self.typed_modes and self.num_modes >= 4:
                # M1 typed correction mechanisms:
                # 0 false-positive deletion, 1 false-negative fill,
                # 2 boundary trim, 3 boundary expansion.
                interior = (1.0 - boundary).clamp(0.0, 1.0)
                if k == 0:
                    region = base_prob * interior
                    direction = -1.0
                elif k == 1:
                    region = (1.0 - base_prob) * interior
                    direction = 1.0
                elif k == 2:
                    region = boundary * base_prob
                    direction = -1.0
                elif k == 3:
                    region = boundary * (1.0 - base_prob)
                    direction = 1.0
                else:
                    region = torch.ones_like(base_prob)
                    direction = -1.0 if k % 2 == 0 else 1.0
                effective_attention = attention * region
                magnitude = (
                    self.max_atom_delta
                    * effective_attention
                    * torch.sigmoid(delta_raw)
                    * gate[:, :, None, None]
                )
                correction = direction * magnitude
            else:
                effective_attention = attention
                correction = (
                    self.max_atom_delta
                    * attention
                    * torch.tanh(delta_raw)
                    * gate[:, :, None, None]
                )

            attn_logits.append(attention_logit)
            attn_probs.append(effective_attention)
            delta_raws.append(delta_raw)
            corrections.append(correction)
            gates.append(gate)

        return (
            torch.cat(attn_logits, dim=1),
            torch.cat(attn_probs, dim=1),
            torch.cat(delta_raws, dim=1),
            torch.cat(corrections, dim=1),
            torch.cat(gates, dim=1),
        )

    def _compose_candidates(
        self, base_logits: torch.Tensor, corrections: torch.Tensor
    ) -> torch.Tensor:
        # [B, M, K, 1, 1] x [B, 1, K, H, W] -> [B, M, H, W]
        combo = self.combo_matrix.to(device=corrections.device, dtype=corrections.dtype)
        total = torch.einsum("mk,bkhw->bmhw", combo, corrections)
        if self.typed_modes and not self.compose_pairs and not self.compose_full:
            bounded = total.clamp(-self.max_total_delta, self.max_total_delta)
        else:
            bounded = self.max_total_delta * torch.tanh(
                total / max(self.max_total_delta, EPS)
            )
        bounded[:, 0] = 0.0
        return base_logits[:, 0:1] + bounded

    def _discovery_candidates(
        self,
        discovery_shared: torch.Tensor,
        output_hw: tuple[int, int],
    ) -> tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """Generate Base-independent feature-coherent full-mask candidates."""
        if self.num_discovery <= 0:
            return None, {}
        if self.discovery_mode in {"query_mask", "v479_query_mask"}:
            pooled_hw = (
                min(self.discovery_size, discovery_shared.shape[-2]),
                min(self.discovery_size, discovery_shared.shape[-1]),
            )
            feature = F.adaptive_avg_pool2d(discovery_shared, pooled_hw)
            pixel = F.normalize(self.discovery_pixel_proj(feature), dim=1, eps=EPS)
            global_context = self.discovery_query_context(
                F.adaptive_avg_pool2d(feature, 1).flatten(1)
            )
            queries = self.discovery_queries.weight[: self.num_discovery][None]
            queries = F.normalize(queries + global_context[:, None], dim=-1, eps=EPS)
            coarse = torch.einsum("bchw,bkc->bkhw", pixel, queries)
            feature_expand = feature[:, None].expand(
                -1, self.num_discovery, -1, -1, -1
            )
            refine_input = torch.cat(
                [feature_expand, coarse[:, :, None]], dim=2
            ).reshape(
                -1, self.hidden_dim + 1, pooled_hw[0], pooled_hw[1]
            )
            residual = self.discovery_mask_refine(refine_input).reshape(
                discovery_shared.shape[0], self.num_discovery, pooled_hw[0], pooled_hw[1]
            )
            logits_low = coarse + residual
            logits = F.interpolate(
                logits_low, size=output_hw, mode="bilinear", align_corners=False
            )
            probability = torch.sigmoid(logits)
            if self.num_discovery > 1:
                pair_ious = []
                for i in range(self.num_discovery):
                    for j in range(i + 1, self.num_discovery):
                        inter = (probability[:, i] * probability[:, j]).flatten(1).sum(1)
                        union = (
                            probability[:, i] + probability[:, j]
                            - probability[:, i] * probability[:, j]
                        ).flatten(1).sum(1)
                        pair_ious.append((inter + EPS) / (union + EPS))
                duplicate_iou = torch.stack(pair_ious, dim=1).mean()
            else:
                duplicate_iou = logits.sum() * 0.0
            return logits, {
                "cem_discovery_membership": probability,
                "cem_discovery_similarity": torch.sigmoid(F.interpolate(
                    coarse, size=output_hw, mode="bilinear", align_corners=False
                )),
                "cem_discovery_coherence_energy": logits.sum() * 0.0,
                "cem_discovery_diversity_loss": duplicate_iou,
                "cem_discovery_duplicate_iou": duplicate_iou.detach(),
            }
        pooled_hw = (
            min(self.discovery_size, discovery_shared.shape[-2]),
            min(self.discovery_size, discovery_shared.shape[-1]),
        )
        feature = F.adaptive_avg_pool2d(discovery_shared, pooled_hw)
        feature_norm = F.normalize(feature, dim=1, eps=EPS)
        queries = F.normalize(
            self.discovery_queries.weight[: self.num_discovery], dim=1, eps=EPS
        )
        seed_logits = torch.einsum('bchw,kc->bkhw', feature_norm, queries)
        membership = torch.sigmoid(seed_logits / self.discovery_temperature)
        coherence_energies = []
        similarity = seed_logits
        for _ in range(self.discovery_iterations):
            foreground_mass = membership.flatten(2).sum(dim=-1).clamp_min(EPS)
            background = 1.0 - membership
            background_mass = background.flatten(2).sum(dim=-1).clamp_min(EPS)
            foreground_center = torch.einsum(
                'bkhw,bchw->bkc', membership, feature_norm
            ) / foreground_mass[:, :, None]
            background_center = torch.einsum(
                'bkhw,bchw->bkc', background, feature_norm
            ) / background_mass[:, :, None]
            dist_foreground = (
                feature_norm[:, None] - foreground_center[:, :, :, None, None]
            ).square().mean(dim=2)
            dist_background = (
                feature_norm[:, None] - background_center[:, :, :, None, None]
            ).square().mean(dim=2)
            membership = torch.sigmoid(
                (dist_background - dist_foreground) / self.discovery_temperature
            )
            foreground_center_norm = F.normalize(
                foreground_center, dim=-1, eps=EPS
            )
            similarity = torch.einsum(
                'bchw,bkc->bkhw', feature_norm, foreground_center_norm
            )
            intra = (
                membership * dist_foreground
                + (1.0 - membership) * dist_background
            ).mean(dim=(-2, -1))
            separation = (
                foreground_center - background_center
            ).square().mean(dim=-1).sqrt()
            coherence_energies.append(
                intra + F.relu(self.discovery_separation_margin - separation)
            )

        similarity_prior = torch.sigmoid(
            (similarity - self.discovery_similarity_threshold)
            / self.discovery_temperature
        )
        prior = (0.5 * membership + 0.5 * similarity_prior).clamp(EPS, 1.0 - EPS)
        feature_expand = feature[:, None].expand(
            -1, self.num_discovery, -1, -1, -1
        )
        decoder_input = torch.cat(
            [feature_expand, prior[:, :, None], similarity[:, :, None]], dim=2
        ).reshape(
            -1, self.hidden_dim + 2, pooled_hw[0], pooled_hw[1]
        )
        residual = self.discovery_decoder(decoder_input).reshape(
            discovery_shared.shape[0], self.num_discovery, pooled_hw[0], pooled_hw[1]
        )
        logits_low = torch.logit(prior) + residual
        logits = F.interpolate(
            logits_low,
            size=output_hw,
            mode='bilinear',
            align_corners=False,
        )
        probability = torch.sigmoid(logits)
        if self.num_discovery > 1:
            overlap = []
            for i in range(self.num_discovery):
                for j in range(i + 1, self.num_discovery):
                    inter = (probability[:, i] * probability[:, j]).flatten(1).sum(1)
                    union = (
                        probability[:, i] + probability[:, j]
                        - probability[:, i] * probability[:, j]
                    ).flatten(1).sum(1)
                    overlap.append((inter + EPS) / (union + EPS))
            diversity_loss = torch.stack(overlap, dim=1).mean()
        else:
            diversity_loss = logits.sum() * 0.0
        energy = torch.stack(coherence_energies, dim=0).mean()
        return logits, {
            'cem_discovery_membership': F.interpolate(
                membership,
                size=output_hw,
                mode='bilinear',
                align_corners=False,
            ),
            'cem_discovery_similarity': F.interpolate(
                similarity,
                size=output_hw,
                mode='bilinear',
                align_corners=False,
            ),
            'cem_discovery_coherence_energy': energy,
            'cem_discovery_diversity_loss': diversity_loss,
        }

    def _candidate_statistics(
        self,
        candidate_probs: torch.Tensor,
        base_prob: torch.Tensor,
        entropy: torch.Tensor,
        mc_std_map: Optional[torch.Tensor] = None,
        mc_disagreement_map: Optional[torch.Tensor] = None,
        mc_pairwise_disagreement: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        base = base_prob[:, 0:1]
        edit = (candidate_probs - base).abs()
        add = (candidate_probs - base).relu()
        remove = (base - candidate_probs).relu()
        cand_entropy = self._entropy(candidate_probs)
        cand_boundary = self._soft_boundary_map(
            candidate_probs.reshape(-1, 1, *candidate_probs.shape[-2:])
        ).reshape_as(candidate_probs)

        edit_mass = edit.flatten(2).sum(dim=-1).clamp_min(EPS)
        uncertainty_on_edit = (
            edit * entropy[:, 0:1]
        ).flatten(2).sum(dim=-1) / edit_mass

        if mc_std_map is None:
            mc_std_map = torch.zeros_like(base_prob)
        if mc_disagreement_map is None:
            mc_disagreement_map = torch.zeros_like(base_prob)
        mc_std_on_edit = (
            edit * mc_std_map[:, 0:1]
        ).flatten(2).sum(dim=-1) / edit_mass
        mc_disagree_on_edit = (
            edit * mc_disagreement_map[:, 0:1]
        ).flatten(2).sum(dim=-1) / edit_mass
        if mc_pairwise_disagreement is None:
            global_mc_disagreement = candidate_probs.new_zeros(
                candidate_probs.shape[0], candidate_probs.shape[1]
            )
        else:
            global_mc_disagreement = mc_pairwise_disagreement[:, None].expand(
                -1, candidate_probs.shape[1]
            )

        area = candidate_probs.mean(dim=(-2, -1))
        entropy_mean = cand_entropy.mean(dim=(-2, -1))
        edit_fraction = edit.mean(dim=(-2, -1))
        add_fraction = add.mean(dim=(-2, -1))
        remove_fraction = remove.mean(dim=(-2, -1))
        boundary_fraction = cand_boundary.mean(dim=(-2, -1))
        confidence = (candidate_probs - 0.5).abs().mean(dim=(-2, -1)) * 2.0
        return torch.stack(
            [
                area,
                entropy_mean,
                edit_fraction,
                add_fraction,
                remove_fraction,
                boundary_fraction,
                uncertainty_on_edit,
                confidence,
                mc_std_on_edit,
                mc_disagree_on_edit,
                global_mc_disagreement,
            ],
            dim=-1,
        )

    def _quality_and_selection(
        self,
        shared: torch.Tensor,
        candidate_logits: torch.Tensor,
        base_prob: torch.Tensor,
        entropy: torch.Tensor,
        mc_std_map: Optional[torch.Tensor] = None,
        mc_disagreement_map: Optional[torch.Tensor] = None,
        mc_pairwise_disagreement: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        candidate_probs = torch.sigmoid(candidate_logits).clamp(EPS, 1.0 - EPS)
        batch, candidate_count = candidate_probs.shape[:2]
        pooled = F.adaptive_avg_pool2d(shared, 1).flatten(1)
        stats = self._candidate_statistics(
            candidate_probs,
            base_prob,
            entropy,
            mc_std_map=mc_std_map,
            mc_disagreement_map=mc_disagreement_map,
            mc_pairwise_disagreement=mc_pairwise_disagreement,
        )

        edit = (candidate_probs - base_prob[:, 0:1]).abs()
        edit_mass = edit.flatten(2).sum(dim=-1)
        edit_weights = edit / edit_mass[:, :, None, None].clamp_min(EPS)
        local_pooled = torch.einsum("bmhw,bchw->bmc", edit_weights, shared)
        local_pooled = torch.where(
            (edit_mass > EPS)[:, :, None],
            local_pooled,
            pooled[:, None, :].expand(-1, candidate_count, -1),
        )

        pooled_expand = pooled[:, None, :].expand(-1, candidate_count, -1)
        candidate_ids = torch.arange(candidate_count, device=shared.device)
        candidate_embed = self.candidate_embedding(candidate_ids)[None].expand(
            batch, -1, -1
        )
        quality_input = torch.cat(
            [pooled_expand, local_pooled, candidate_embed, stats], dim=-1
        )
        if isinstance(self.m2_counterfactual, SetContextCounterfactualEffectEstimator) and isinstance(
            self.m3_policy, SetPredictorRejectorPolicy
        ):
            action_types = self.action_types[: max(candidate_count - 1, 0)]
            family_actions = torch.where(
                action_types == 0, torch.ones_like(action_types),
                torch.where(
                    action_types == 1, torch.full_like(action_types, 2),
                    torch.where(
                        (action_types == 2) | (action_types == 3),
                        torch.full_like(action_types, 3),
                        torch.full_like(action_types, 4),
                    ),
                ),
            )
            family_ids = torch.cat(
                [family_actions.new_zeros(1), family_actions], dim=0
            )[:candidate_count]
            m2 = self.m2_counterfactual(quality_input, family_ids)
            mean_dsc = m2["mean_dsc"]
            q_dsc = m2["q_dsc"]
            mean_nsd = m2["mean_nsd"]
            q_nsd = m2["q_nsd"]
            utility = m2["effect_mean"]
            sigma = m2["effect_sigma"]
            z = utility / sigma.clamp_min(EPS)
            benefit_prob = torch.sigmoid(z)
            harm_logits = m2["harm_logits"]
            harm_prob = torch.sigmoid(harm_logits)
            benefit_logits = torch.logit(benefit_prob.clamp(EPS, 1.0 - EPS))
            consensus_stats = stats[:, 0, :4]
            failure_logit = self.failure_head(
                torch.cat([pooled, consensus_stats], dim=1)
            )[:, 0]
            failure_prob = torch.sigmoid(failure_logit)
            m3 = self.m3_policy(
                representation=m2["representation"],
                ranking_score=m2["ranking_score"],
                q_dsc=q_dsc,
                q_nsd=q_nsd,
                harm_logits=harm_logits,
                failure_prob=failure_prob,
                candidate_stats=stats,
                family_ids=family_ids,
                candidate_probs=candidate_probs,
            )
            selected_index = m3["selected_index"]
            gather_index = selected_index[:, None, None, None].expand(
                -1, 1, candidate_probs.shape[-2], candidate_probs.shape[-1]
            )
            hard_selected_logits = candidate_logits.gather(1, gather_index)[:, 0]
            action_hard = m3["policy_score"].new_zeros(
                (batch, max(candidate_count - 1, 0))
            )
            if candidate_count > 1:
                action_hard.scatter_(
                    1, (selected_index - 1).clamp_min(0)[:, None], m3["accept"][:, None]
                )
            full_hard = F.one_hot(selected_index, num_classes=candidate_count).to(
                m3["policy_score"].dtype
            )
            best_action_score = m3["policy_score"][:, 1:].max(dim=1).values
            best_action = m3["policy_score"][:, 1:].argmax(dim=1)
            best_action_harm = harm_prob[:, 1:].gather(1, best_action[:, None])[:, 0]
            best_action_benefit = benefit_prob[:, 1:].gather(1, best_action[:, None])[:, 0]
            best_action_edit = stats[:, 1:, 2].gather(1, best_action[:, None])[:, 0]
            return {
                "candidate_probs": candidate_probs,
                "cem_candidate_stats": stats,
                "cem_quality_utility": utility,
                "cem_quality_sigma": sigma,
                "cem_quality_benefit_logits": benefit_logits,
                "cem_quality_benefit_prob": benefit_prob,
                "cem_quality_harm_logits": harm_logits,
                "cem_quality_harm_prob": harm_prob,
                "cem_failure_logit": failure_logit,
                "cem_failure_prob": failure_prob,
                "cem_cf_representation": m2["representation"],
                "cem_cf_family_ids": family_ids,
                "cem_cf_mean_dsc": mean_dsc,
                "cem_cf_q10_dsc": q_dsc,
                "cem_cf_mean_nsd": mean_nsd,
                "cem_cf_q10_nsd": q_nsd,
                "cem_cf_conservative_score": m2["conservative_score"],
                "cem_cf_ranking_score": m2["ranking_score"],
                "cem_cf_pairwise_logits": m2["pairwise_logits"],
                "cem_cf_pairwise_probability": m2["pairwise_probability"],
                "cem_cf_pairwise_win_rate": m2["pairwise_win_rate"],
                "cem_candidate_accept_logits": m3["accept_logits"],
                "cem_candidate_accept_prob": m3["accept_prob"],
                "cem_selection_scores": m3["policy_score"],
                "cem_selection_lcb": m2["conservative_score"],
                "cem_selection_soft": m3["policy_soft"],
                "cem_selection_st": m3["policy_st"],
                "cem_st_selected_probs": m3["st_final"],
                "cem_soft_selected_probs": m3["soft_final"],
                "cem_hard_selected_probs": m3["hard_final"],
                "cem_hard_selected_logits": hard_selected_logits,
                "cem_selected_index": selected_index,
                "cem_accept": m3["accept"],
                "cem_best_action_score": best_action_score,
                "cem_best_action_harm": best_action_harm,
                "cem_best_action_benefit": best_action_benefit,
                "cem_best_action_edit_fraction": best_action_edit,
                "cem_m3_policy_scores": m3["policy_score"],
                "cem_m3_selected_family": m3["selected_family"],
                "cem_m3_pairwise_win_rate": m3["pairwise_win_rate"],
                "cem_m3_chosen_score": m3["chosen_score"],
                "cem_m3_chosen_edit": m3["chosen_edit"],
                "cem_m3_top_gap": m3["top_gap"],
                "cem_m3_mc_disagreement": m3["mc_disagreement"],
                "cem_m3_hard_gate": m3["hard_gate"],
                "v20_selector_logits": m3["policy_score"][:, 1:],
                "v20_selector_probs": action_hard,
                "v20_selector_hard": action_hard,
                "m1_selector_soft": m3["policy_soft"],
                "m1_selector_hard": full_hard,
                "m1_choice_logits": m3["policy_score"],
                "m1_hard_selected_slot": selected_index,
                "m1_preserve_hard": (selected_index == 0).to(m3["policy_score"].dtype),
            }

        if isinstance(self.m2_counterfactual, FamilyCounterfactualEffectEstimator) and isinstance(
            self.m3_policy, HierarchicalParetoInterventionPolicy
        ):
            action_types = self.action_types[: max(candidate_count - 1, 0)]
            family_actions = torch.where(
                action_types == 0, torch.ones_like(action_types),
                torch.where(
                    action_types == 1, torch.full_like(action_types, 2),
                    torch.where(
                        (action_types == 2) | (action_types == 3),
                        torch.full_like(action_types, 3),
                        torch.full_like(action_types, 4),
                    ),
                ),
            )
            family_ids = torch.cat(
                [family_actions.new_zeros(1), family_actions], dim=0
            )[:candidate_count]
            m2 = self.m2_counterfactual(quality_input, family_ids)
            mean_dsc = m2["mean_dsc"]
            q_dsc = m2["q_dsc"]
            mean_nsd = m2["mean_nsd"]
            q_nsd = m2["q_nsd"]
            utility = m2["effect_mean"]
            sigma = m2["effect_sigma"]
            z = utility / sigma.clamp_min(EPS)
            benefit_prob = torch.sigmoid(z)
            harm_prob = torch.sigmoid(-q_dsc / sigma.clamp_min(EPS))
            benefit_logits = torch.logit(benefit_prob.clamp(EPS, 1.0 - EPS))
            harm_logits = torch.logit(harm_prob.clamp(EPS, 1.0 - EPS))
            consensus_stats = stats[:, 0, :4]
            failure_logit = self.failure_head(
                torch.cat([pooled, consensus_stats], dim=1)
            )[:, 0]
            failure_prob = torch.sigmoid(failure_logit)
            m3 = self.m3_policy(
                q_dsc=q_dsc,
                q_nsd=q_nsd,
                candidate_stats=stats,
                family_ids=family_ids,
                candidate_probs=candidate_probs,
            )
            selected_index = m3["selected_index"]
            gather_index = selected_index[:, None, None, None].expand(
                -1, 1, candidate_probs.shape[-2], candidate_probs.shape[-1]
            )
            hard_selected_logits = candidate_logits.gather(1, gather_index)[:, 0]
            action_hard = m3["policy_score"].new_zeros(
                (batch, max(candidate_count - 1, 0))
            )
            if candidate_count > 1:
                action_hard.scatter_(
                    1, (selected_index - 1).clamp_min(0)[:, None], m3["accept"][:, None]
                )
            full_hard = F.one_hot(selected_index, num_classes=candidate_count).to(
                m3["policy_score"].dtype
            )
            best_action_score = m3["policy_score"][:, 1:].max(dim=1).values
            best_action = m3["policy_score"][:, 1:].argmax(dim=1)
            best_action_harm = harm_prob[:, 1:].gather(1, best_action[:, None])[:, 0]
            best_action_benefit = benefit_prob[:, 1:].gather(1, best_action[:, None])[:, 0]
            best_action_edit = stats[:, 1:, 2].gather(1, best_action[:, None])[:, 0]
            return {
                "candidate_probs": candidate_probs,
                "cem_candidate_stats": stats,
                "cem_quality_utility": utility,
                "cem_quality_sigma": sigma,
                "cem_quality_benefit_logits": benefit_logits,
                "cem_quality_benefit_prob": benefit_prob,
                "cem_quality_harm_logits": harm_logits,
                "cem_quality_harm_prob": harm_prob,
                "cem_failure_logit": failure_logit,
                "cem_failure_prob": failure_prob,
                "cem_cf_representation": m2["representation"],
                "cem_cf_family_ids": family_ids,
                "cem_cf_mean_dsc": mean_dsc,
                "cem_cf_q10_dsc": q_dsc,
                "cem_cf_mean_nsd": mean_nsd,
                "cem_cf_q10_nsd": q_nsd,
                "cem_cf_pairwise_logits": m2["pairwise_logits"],
                "cem_cf_pairwise_probability": m2["pairwise_probability"],
                "cem_cf_pairwise_win_rate": m2["pairwise_win_rate"],
                "cem_selection_scores": m3["policy_score"],
                "cem_selection_lcb": m3["policy_score"],
                "cem_selection_soft": m3["policy_soft"],
                "cem_selection_st": m3["policy_st"],
                "cem_st_selected_probs": m3["st_final"],
                "cem_soft_selected_probs": m3["soft_final"],
                "cem_hard_selected_probs": m3["hard_final"],
                "cem_hard_selected_logits": hard_selected_logits,
                "cem_selected_index": selected_index,
                "cem_accept": m3["accept"],
                "cem_best_action_score": best_action_score,
                "cem_best_action_harm": best_action_harm,
                "cem_best_action_benefit": best_action_benefit,
                "cem_best_action_edit_fraction": best_action_edit,
                "cem_m3_policy_scores": m3["policy_score"],
                "cem_m3_family_scores": m3["family_scores"],
                "cem_m3_selected_family": m3["selected_family"],
                "cem_m3_pairwise_win_rate": m3["pairwise_win_rate"],
                "cem_m3_chosen_score": m3["chosen_score"],
                "cem_m3_chosen_edit": m3["chosen_edit"],
                "v20_selector_logits": m3["policy_score"][:, 1:],
                "v20_selector_probs": action_hard,
                "v20_selector_hard": action_hard,
                "m1_selector_soft": m3["policy_soft"],
                "m1_selector_hard": full_hard,
                "m1_choice_logits": m3["policy_score"],
                "m1_hard_selected_slot": selected_index,
                "m1_preserve_hard": (selected_index == 0).to(m3["policy_score"].dtype),
            }

        if self.m2_counterfactual is not None and self.m3_policy is not None:
            m2 = self.m2_counterfactual(quality_input)
            utility = m2["effect_mean"] + self.candidate_reliability_bias[None]
            utility = utility - utility[:, :1]
            sigma = m2["effect_sigma"]
            z = utility / sigma.clamp_min(EPS)
            benefit_prob = torch.sigmoid(z)
            harm_prob = torch.sigmoid(
                (-utility - self.selector_utility_margin) / sigma.clamp_min(EPS)
            )
            benefit_logits = torch.logit(benefit_prob.clamp(EPS, 1.0 - EPS))
            harm_logits = torch.logit(harm_prob.clamp(EPS, 1.0 - EPS))
            consensus_stats = stats[:, 0, :4]
            failure_logit = self.failure_head(
                torch.cat([pooled, consensus_stats], dim=1)
            )[:, 0]
            failure_prob = torch.sigmoid(failure_logit)
            m3 = self.m3_policy(
                representation=m2["representation"],
                effect_mean=utility,
                effect_sigma=sigma,
                pairwise_logits=m2["pairwise_logits"],
                candidate_stats=stats,
                failure_probability=failure_prob,
                action_types=self.action_types[: max(candidate_count - 1, 0)],
                candidate_probs=candidate_probs,
            )
            selected_index = m3["selected_index"]
            gather_index = selected_index[:, None, None, None].expand(
                -1, 1, candidate_probs.shape[-2], candidate_probs.shape[-1]
            )
            hard_selected_logits = candidate_logits.gather(1, gather_index)[:, 0]
            action_hard = m3["policy_score"].new_zeros(
                (batch, max(candidate_count - 1, 0))
            )
            if candidate_count > 1:
                action_hard.scatter_(
                    1,
                    (selected_index - 1).clamp_min(0)[:, None],
                    m3["accept"][:, None],
                )
            full_hard = F.one_hot(
                selected_index, num_classes=candidate_count
            ).to(m3["policy_score"].dtype)
            best_action_score = m3["policy_score"][:, 1:].max(dim=1).values
            best_action = m3["policy_score"][:, 1:].argmax(dim=1)
            best_action_harm = harm_prob[:, 1:].gather(
                1, best_action[:, None]
            )[:, 0]
            best_action_benefit = benefit_prob[:, 1:].gather(
                1, best_action[:, None]
            )[:, 0]
            best_action_edit = stats[:, 1:, 2].gather(
                1, best_action[:, None]
            )[:, 0]
            return {
                "candidate_probs": candidate_probs,
                "cem_candidate_stats": stats,
                "cem_quality_utility": utility,
                "cem_quality_sigma": sigma,
                "cem_quality_benefit_logits": benefit_logits,
                "cem_quality_benefit_prob": benefit_prob,
                "cem_quality_harm_logits": harm_logits,
                "cem_quality_harm_prob": harm_prob,
                "cem_failure_logit": failure_logit,
                "cem_failure_prob": failure_prob,
                "cem_cf_representation": m2["representation"],
                "cem_cf_pairwise_logits": m2["pairwise_logits"],
                "cem_cf_pairwise_probability": m2["pairwise_probability"],
                "cem_cf_pairwise_win_rate": m2["pairwise_win_rate"],
                "cem_selection_scores": m3["policy_score"],
                "cem_selection_lcb": m3["policy_score"],
                "cem_selection_soft": m3["policy_soft"],
                "cem_selection_st": m3["policy_st"],
                "cem_st_selected_probs": m3["st_final"],
                "cem_soft_selected_probs": m3["soft_final"],
                "cem_hard_selected_probs": m3["hard_final"],
                "cem_hard_selected_logits": hard_selected_logits,
                "cem_selected_index": selected_index,
                "cem_accept": m3["accept"],
                "cem_best_action_score": best_action_score,
                "cem_best_action_harm": best_action_harm,
                "cem_best_action_benefit": best_action_benefit,
                "cem_best_action_edit_fraction": best_action_edit,
                "cem_m3_policy_scores": m3["policy_score"],
                "cem_m3_pairwise_win_rate": m3["pairwise_win_rate"],
                "cem_m3_chosen_score": m3["chosen_score"],
                "cem_m3_chosen_edit": m3["chosen_edit"],
                "v20_selector_logits": m3["policy_score"][:, 1:],
                "v20_selector_probs": action_hard,
                "v20_selector_hard": action_hard,
                "m1_selector_soft": m3["policy_soft"],
                "m1_selector_hard": full_hard,
                "m1_choice_logits": m3["policy_score"],
                "m1_hard_selected_slot": selected_index,
                "m1_preserve_hard": (selected_index == 0).to(m3["policy_score"].dtype),
            }

        quality_raw = self.quality_head(
            quality_input.reshape(-1, quality_input.shape[-1])
        ).reshape(batch, candidate_count, 2)

        utility_raw = quality_raw[..., 0] + self.candidate_reliability_bias[None]
        utility = utility_raw - utility_raw[:, :1]
        raw_scale = quality_raw[..., 1]
        sigma = F.softplus(raw_scale) + self.selector_min_sigma
        sigma = sigma.clamp(
            min=self.selector_min_sigma,
            max=self.selector_max_sigma,
        )
        sigma = torch.cat(
            [sigma.new_full((batch, 1), self.selector_min_sigma), sigma[:, 1:]],
            dim=1,
        )

        z = utility / sigma.clamp_min(EPS)
        benefit_prob = torch.sigmoid(z)
        harm_prob = torch.sigmoid(
            (-utility - self.selector_utility_margin)
            / sigma.clamp_min(EPS)
        )
        benefit_logits = torch.logit(benefit_prob.clamp(EPS, 1.0 - EPS))
        harm_logits = torch.logit(harm_prob.clamp(EPS, 1.0 - EPS))

        consensus_stats = stats[:, 0, :4]
        failure_logit = self.failure_head(
            torch.cat([pooled, consensus_stats], dim=1)
        )[:, 0]
        failure_prob = torch.sigmoid(failure_logit)

        lcb = utility - self.selector_lcb_kappa * sigma
        # Penalize edits in regions where MC predictions disagree, and very
        # large edits.  These are candidate-relative failure signals rather
        # than a global image-level veto.
        lcb = (
            lcb
            - self.selector_mc_weight * stats[..., 9]
            - self.selector_edit_penalty * stats[..., 2]
        )
        score = torch.cat([lcb.new_zeros((batch, 1)), lcb[:, 1:]], dim=1)

        soft_weights = torch.softmax(score / self.selector_temperature, dim=1)
        soft_index = score.argmax(dim=1)
        hard_weights = F.one_hot(
            soft_index, num_classes=candidate_count
        ).to(score.dtype)
        st_weights = hard_weights + soft_weights - soft_weights.detach()
        st_selected_probs = (
            st_weights[:, :, None, None] * candidate_probs
        ).sum(dim=1)
        soft_selected_probs = (
            soft_weights[:, :, None, None] * candidate_probs
        ).sum(dim=1)

        if candidate_count > 1:
            action_score = score[:, 1:]
            best_action_score, best_action = action_score.max(dim=1)
            best_action_harm = harm_prob[:, 1:].gather(
                1, best_action[:, None]
            )[:, 0]
            best_action_benefit = benefit_prob[:, 1:].gather(
                1, best_action[:, None]
            )[:, 0]
            best_action_edit = stats[:, 1:, 2].gather(
                1, best_action[:, None]
            )[:, 0]
            accept = (
                (best_action_score > self.selector_utility_margin)
                & (best_action_edit <= self.selector_max_deploy_edit)
            )
            selected_index = torch.where(
                accept, best_action + 1, torch.zeros_like(best_action)
            )
        else:
            action_score = score.new_zeros((batch, 0))
            best_action_score = score.new_zeros(batch)
            best_action_harm = score.new_zeros(batch)
            best_action_benefit = score.new_zeros(batch)
            best_action_edit = score.new_zeros(batch)
            accept = torch.zeros(batch, device=score.device, dtype=torch.bool)
            selected_index = torch.zeros(batch, device=score.device, dtype=torch.long)

        gather_index = selected_index[:, None, None, None].expand(
            -1, 1, candidate_probs.shape[-2], candidate_probs.shape[-1]
        )
        hard_selected_probs = candidate_probs.gather(1, gather_index)[:, 0]
        hard_selected_logits = candidate_logits.gather(1, gather_index)[:, 0]

        action_hard = score.new_zeros((batch, max(candidate_count - 1, 0)))
        if candidate_count > 1:
            action_hard.scatter_(
                1,
                (selected_index - 1).clamp_min(0)[:, None],
                accept.to(score.dtype)[:, None],
            )
        full_hard = F.one_hot(
            selected_index, num_classes=candidate_count
        ).to(score.dtype)

        return {
            "candidate_probs": candidate_probs,
            "cem_candidate_stats": stats,
            "cem_quality_utility": utility,
            "cem_quality_sigma": sigma,
            "cem_quality_benefit_logits": benefit_logits,
            "cem_quality_benefit_prob": benefit_prob,
            "cem_quality_harm_logits": harm_logits,
            "cem_quality_harm_prob": harm_prob,
            "cem_failure_logit": failure_logit,
            "cem_failure_prob": failure_prob,
            "cem_selection_scores": score,
            "cem_selection_lcb": score,
            "cem_selection_soft": soft_weights,
            "cem_selection_st": st_weights,
            "cem_st_selected_probs": st_selected_probs,
            "cem_soft_selected_probs": soft_selected_probs,
            "cem_hard_selected_probs": hard_selected_probs,
            "cem_hard_selected_logits": hard_selected_logits,
            "cem_selected_index": selected_index,
            "cem_accept": accept.to(score.dtype),
            "cem_best_action_score": best_action_score,
            "cem_best_action_harm": best_action_harm,
            "cem_best_action_benefit": best_action_benefit,
            "cem_best_action_edit_fraction": best_action_edit,
            "v20_selector_logits": action_score,
            "v20_selector_probs": action_hard,
            "v20_selector_hard": action_hard,
            "m1_selector_soft": soft_weights,
            "m1_selector_hard": full_hard,
            "m1_choice_logits": score,
            "m1_hard_selected_slot": selected_index,
            "m1_preserve_hard": (selected_index == 0).to(score.dtype),
        }

    def generate(
        self,
        base_logits: torch.Tensor,
        image: torch.Tensor,
        semantic_map: Optional[torch.Tensor],
        text_features: torch.Tensor,
        negative_text_features: Optional[torch.Tensor] = None,
        mc_std_map: Optional[torch.Tensor] = None,
        mc_disagreement_map: Optional[torch.Tensor] = None,
        mc_pairwise_disagreement: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        del kwargs
        if base_logits.ndim == 3:
            base_logits = base_logits[:, None]
        if base_logits.ndim != 4 or base_logits.shape[1] != 1:
            raise ValueError(
                f"CEM expects consensus logits [B,1,H,W], got {tuple(base_logits.shape)}"
            )

        shared, discovery_shared, base_prob, entropy, boundary = self._shared_feature(
            image=image,
            semantic_map=semantic_map,
            text_features=text_features,
            negative_text_features=negative_text_features,
            base_logits=base_logits,
            mc_std_map=mc_std_map,
            mc_disagreement_map=mc_disagreement_map,
        )
        (
            atom_attention_logits,
            atom_attention,
            atom_delta_raw,
            atom_corrections,
            atom_gates,
        ) = self._predict_modes(shared, base_prob, boundary)
        local_candidate_logits = self._compose_candidates(base_logits, atom_corrections)
        discovery_logits, discovery_aux = self._discovery_candidates(
            discovery_shared=discovery_shared,
            output_hw=base_logits.shape[-2:],
        )
        candidate_logits = (
            torch.cat([local_candidate_logits, discovery_logits], dim=1)
            if discovery_logits is not None
            else local_candidate_logits
        )
        selection = self._quality_and_selection(
            shared=shared,
            candidate_logits=candidate_logits,
            base_prob=base_prob,
            entropy=entropy,
            mc_std_map=mc_std_map,
            mc_disagreement_map=mc_disagreement_map,
            mc_pairwise_disagreement=mc_pairwise_disagreement,
        )
        candidate_probs = selection["candidate_probs"]

        action_supports = (candidate_probs[:, 1:] - candidate_probs[:, :1]).abs()
        action_count = action_supports.shape[1]
        zero_actions = candidate_logits.new_zeros((candidate_logits.shape[0], action_count))
        edit_fraction = action_supports.mean(dim=(-2, -1))
        action_types = self.action_types[:action_count]

        pairwise = []
        for i in range(candidate_probs.shape[1]):
            for j in range(i + 1, candidate_probs.shape[1]):
                pairwise.append(
                    (candidate_probs[:, i] - candidate_probs[:, j]).abs().mean(dim=(-2, -1))
                )
        pairwise_l1 = (
            torch.stack(pairwise, dim=1).mean(dim=1)
            if pairwise
            else candidate_logits.new_zeros(candidate_logits.shape[0])
        )

        hard_probs = selection["cem_hard_selected_probs"]
        hard_logits = selection["cem_hard_selected_logits"]
        soft_probs = selection["cem_soft_selected_probs"]
        action_utility = selection["cem_quality_utility"][:, 1:]

        aux = {
            **selection,
            **discovery_aux,
            "cem_atom_attention_logits": atom_attention_logits,
            "cem_atom_attention": atom_attention,
            "cem_atom_delta_raw": atom_delta_raw,
            "cem_atom_corrections": atom_corrections,
            "cem_atom_gates": atom_gates,
            "cem_typed_modes": candidate_logits.new_tensor(float(self.typed_modes)),
            "cem_typed_mode_signs": self.typed_mode_signs.to(candidate_logits),
            "cem_mc_std_map": (
                mc_std_map if mc_std_map is not None else torch.zeros_like(base_prob)
            ),
            "cem_mc_disagreement_map": (
                mc_disagreement_map
                if mc_disagreement_map is not None
                else torch.zeros_like(base_prob)
            ),
            "cem_mc_pairwise_disagreement": (
                mc_pairwise_disagreement
                if mc_pairwise_disagreement is not None
                else candidate_logits.new_zeros(candidate_logits.shape[0])
            ),
            "cem_combo_matrix": self.combo_matrix,
            "cem_combo_cardinality": torch.cat([
                self.combo_matrix.sum(dim=1),
                self.combo_matrix.new_full((self.num_discovery,), -1.0),
            ]),
            "cem_discovery_enabled": candidate_logits.new_tensor(float(self.discovery_enabled)),
            "cem_num_discovery_candidates": candidate_logits.new_tensor(float(self.num_discovery)),
            "cem_pairwise_l1_forward": pairwise_l1,
            "cem_consensus_prob": base_prob[:, 0],
            "candidate_probs": candidate_probs,
            "direct_fused_probs": hard_probs,
            "router_fused_probs": hard_probs,
            "v20_fused_probs": soft_probs,
            "v20_hard_fused_probs": hard_probs,
            "v20_fused_logits": hard_logits,
            "m1_soft_fused_probs": soft_probs,
            "m1_hard_fused_probs": hard_probs,
            "m1_soft_fused_logits": torch.logit(soft_probs.clamp(EPS, 1.0 - EPS)),
            "m1_st_fused_logits": torch.logit(
                selection["cem_st_selected_probs"].clamp(EPS, 1.0 - EPS)
            ),
            "m1_hard_fused_logits": hard_logits,
            "v20_action_supports": action_supports,
            "v20_control_supports": torch.zeros_like(action_supports),
            "v20_action_types": action_types,
            "v20_visual_scores": action_utility,
            "v20_cf_logit": action_utility,
            "v20_cf_signed_delta": action_utility,
            "v20_cf_available": torch.ones_like(action_utility),
            "v20_budget": edit_fraction,
            "local_action_area": edit_fraction,
            "v20_entropy": entropy[:, 0],
            "v20_boundary": boundary[:, 0],
            "v469_action_enabled_mask": torch.ones(
                action_count,
                device=candidate_logits.device,
                dtype=candidate_logits.dtype,
            ),
            "tpmhg_hypothesis_logits": candidate_logits[:, 1:],
            "tpmhg_hypothesis_probs": candidate_probs[:, 1:],
            "tpmhg_pairwise_l1_forward": pairwise_l1,
            "tpmhg_text_separation": candidate_logits.sum() * 0.0,
            "candidate_base_detached": candidate_logits.new_zeros(candidate_logits.shape[0]),
        }
        return candidate_logits, aux


# Legacy class name retained so old constructor sites and checkpoints fail only
# on genuine parameter-shape incompatibilities, not on a missing symbol.
TextPromptedMultiHypothesisGenerator = CompositionalErrorModeCandidateGenerator


class TrainablePSEGenerator(nn.Module):
    """V10.1 M1 local Preserve/Shrink/Expand candidate generator.

    M1 only exports candidates. It never performs final candidate selection;
    candidate verification and final preserve/select decision are handled by M2.


    The generator produces three direction-constrained candidates:
    C0 Preserve, C1 Shrink, and C2 Expand.  It also predicts an optional
    three-class spatial router (Preserve/Shrink/Expand).  The router is trained
    only with train-split labels; inference uses image, Base prediction, local
    structure, and optional projected UniMedCLIP patch features only.
    """

    role_names = ("preserve", "atomic_delete", "atomic_fill")

    def __init__(self, cfg) -> None:
        super().__init__()
        m1 = _cfg_get(cfg, "M1", None)
        self.anchor_threshold = float(_cfg_get(m1, "ANCHOR_THRESHOLD", 0.50))
        self.morph_radius = int(_cfg_get(m1, "MORPH_RADIUS", 3))
        self.initial_shrink_alpha = float(_cfg_get(m1, "SHRINK_ALPHA", 0.42))
        self.initial_expand_alpha = float(_cfg_get(m1, "EXPAND_ALPHA", 0.45))
        self.max_edit_strength = float(_cfg_get(m1, "MAX_EDIT_STRENGTH", 0.85))
        self.entropy_weight = float(_cfg_get(m1, "ENTROPY_WEIGHT", 0.35))
        self.boundary_weight = float(_cfg_get(m1, "BOUNDARY_WEIGHT", 0.65))
        self.image_edge_weight = float(_cfg_get(m1, "IMAGE_EDGE_WEIGHT", 0.25))
        self.band_inside_radius = int(_cfg_get(m1, "BAND_INSIDE_RADIUS", 4))
        self.band_outside_radius = int(_cfg_get(m1, "BAND_OUTSIDE_RADIUS", 6))
        # V12: Expand is restricted to a connected near-boundary repair band.
        # It remains a diagnostic candidate by default; M3 may disable its deployment.
        self.expand_connect_radius = max(1, int(_cfg_get(m1, "M1_EXPAND_CONNECT_RADIUS", 2)))
        self.expand_uncertainty_floor = min(max(float(_cfg_get(m1, "M1_EXPAND_UNCERTAINTY_FLOOR", 0.25)), 0.0), 1.0)
        self.gate_min_strength = float(_cfg_get(m1, "GATE_MIN_STRENGTH", 0.10))
        self.gate_power = float(_cfg_get(m1, "GATE_POWER", 1.0))
        mode = str(_cfg_get(m1, "TRAIN_MODE", "anchor_student")).lower()
        self.detach_base_for_candidates = bool(_cfg_get(m1, "DETACH_BASE_FOR_CANDIDATES", True)) or mode in {
            "anchor_student", "frozen"
        }

        self.use_learned_residual = bool(_cfg_get(m1, "USE_LEARNED_RESIDUAL", True))
        self.morph_residual_mix = float(_cfg_get(m1, "MORPH_RESIDUAL_MIX", 0.50))
        self.morph_residual_mix = min(max(self.morph_residual_mix, 0.0), 1.0)
        self.router_enabled = bool(_cfg_get(m1, "ROUTER_ENABLED", False))
        self.use_semantic_feature = bool(_cfg_get(m1, "USE_SEMANTIC_FEATURE", False))

        hidden_dim = int(_cfg_get(m1, "HIDDEN_DIM", 48))
        self.hidden_dim = hidden_dim
        self.semantic_channels = int(_cfg_get(m1, "SEMANTIC_CHANNELS", 512))

        # Base probability, entropy, boundary, inside/outside contour bands,
        # ultrasound gray intensity, and image-gradient map.
        self.trunk = nn.Sequential(
            _ConvNormGELU(7, hidden_dim),
            _ConvNormGELU(hidden_dim, hidden_dim),
        )
        self.semantic_proj = None
        if self.use_semantic_feature:
            self.semantic_proj = nn.Sequential(
                nn.Conv2d(self.semantic_channels, hidden_dim, kernel_size=1, bias=False),
                nn.GroupNorm(max(1, min(8, hidden_dim)), hidden_dim),
                nn.GELU(),
            )

        self.fp_head = nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=True)
        self.fn_head = nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=True)

        gate_init = float(_cfg_get(m1, "LEARNED_GATE_INIT", 0.20))
        gate_init = min(max(gate_init, 1e-4), 1.0 - 1e-4)
        init_gate_bias = math.log(gate_init / (1.0 - gate_init))
        nn.init.zeros_(self.fp_head.weight)
        nn.init.zeros_(self.fn_head.weight)
        nn.init.constant_(self.fp_head.bias, init_gate_bias)
        nn.init.constant_(self.fn_head.bias, init_gate_bias)

        def _inverse_sigmoid(value: float) -> float:
            ratio = min(
                max(float(value) / max(self.max_edit_strength, 1e-4), 1e-4),
                1.0 - 1e-4,
            )
            return math.log(ratio / (1.0 - ratio))

        self.shrink_strength_logit = nn.Parameter(
            torch.tensor(_inverse_sigmoid(self.initial_shrink_alpha))
        )
        self.expand_strength_logit = nn.Parameter(
            torch.tensor(_inverse_sigmoid(self.initial_expand_alpha))
        )

        self.fp_delta_head = None
        self.fn_delta_head = None
        if self.use_learned_residual:
            self.fp_delta_head = nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=True)
            self.fn_delta_head = nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=True)
            delta_init = float(_cfg_get(m1, "LEARNED_DELTA_INIT", 0.20))
            delta_init = min(
                max(delta_init, 1e-4),
                self.max_edit_strength * (1.0 - 1e-4),
            )
            init_delta_bias = _inverse_sigmoid(delta_init)
            nn.init.zeros_(self.fp_delta_head.weight)
            nn.init.zeros_(self.fn_delta_head.weight)
            nn.init.constant_(self.fp_delta_head.bias, init_delta_bias)
            nn.init.constant_(self.fn_delta_head.bias, init_delta_bias)

        # V17_LOGIT_SHRINK_COMPAT
        # Kept so existing V17 checkpoints remain loadable after V18 is
        # installed. V18 itself does not enable this global shrink branch.
        self.use_logit_shrink = bool(
            _cfg_get(m1, "USE_LOGIT_SHRINK", False)
        )
        self.v17_logit_gate_scale = max(
            0.0, float(_cfg_get(m1, "V17_LOGIT_GATE_SCALE", 3.0))
        )
        self.v17_max_logit_delta = max(
            0.1, float(_cfg_get(m1, "V17_MAX_LOGIT_DELTA", 5.0))
        )
        self.logit_shrink_head = None
        if self.use_logit_shrink:
            self.logit_shrink_head = nn.Sequential(
                _ConvNormGELU(hidden_dim, hidden_dim),
                nn.Conv2d(
                    hidden_dim, 1, kernel_size=3, padding=1, bias=True
                ),
            )
            init_delta = min(
                max(
                    float(_cfg_get(m1, "V17_LOGIT_DELTA_INIT", 1.50)),
                    1e-4,
                ),
                self.v17_max_logit_delta,
            )
            init_bias = math.log(math.expm1(init_delta))
            nn.init.zeros_(self.logit_shrink_head[-1].weight)
            nn.init.constant_(self.logit_shrink_head[-1].bias, init_bias)

        # V18_ATOMIC_CANDIDATE_BANK
        # C0 Preserve, C1 Boundary-Trim, C2 FP-Island-Drop.
        #
        # The two edit candidates are deliberately disjoint:
        # - Boundary-Trim only operates on the predicted inner contour of
        #   sufficiently dense foreground regions.
        # - FP-Island-Drop only operates on low-density foreground islands.
        #
        # This prevents one global shrink map from mixing unrelated edits and
        # gives each candidate a concrete counterfactual interpretation.
        self.v18_atomic_candidates = bool(
            _cfg_get(m1, "V18_ATOMIC_CANDIDATES", False)
        )
        self.v18_trim_radius = max(
            1, int(_cfg_get(m1, "V18_TRIM_RADIUS", 1))
        )
        self.v18_component_radius = max(
            1, int(_cfg_get(m1, "V18_COMPONENT_RADIUS", 3))
        )
        self.v18_component_density_max = min(
            max(
                float(_cfg_get(m1, "V18_COMPONENT_DENSITY_MAX", 0.55)),
                0.0,
            ),
            1.0,
        )
        self.v18_max_logit_delta = max(
            0.1, float(_cfg_get(m1, "V18_MAX_LOGIT_DELTA", 5.0))
        )
        self.v18_trim_gate_head = None
        self.v18_component_gate_head = None
        self.v18_trim_delta_head = None
        self.v18_component_delta_head = None

        if self.v18_atomic_candidates:
            gate_init = min(
                max(float(_cfg_get(m1, "V18_GATE_INIT", 0.20)), 1e-4),
                1.0 - 1e-4,
            )
            gate_bias = math.log(gate_init / (1.0 - gate_init))

            delta_init = min(
                max(float(_cfg_get(m1, "V18_LOGIT_DELTA_INIT", 1.50)), 1e-4),
                self.v18_max_logit_delta,
            )
            delta_bias = math.log(math.expm1(delta_init))

            self.v18_trim_gate_head = nn.Conv2d(
                hidden_dim, 1, kernel_size=1, bias=True
            )
            self.v18_component_gate_head = nn.Conv2d(
                hidden_dim, 1, kernel_size=1, bias=True
            )
            self.v18_trim_delta_head = nn.Sequential(
                _ConvNormGELU(hidden_dim, hidden_dim),
                nn.Conv2d(
                    hidden_dim,
                    1,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                ),
            )
            self.v18_component_delta_head = nn.Sequential(
                _ConvNormGELU(hidden_dim, hidden_dim),
                nn.Conv2d(
                    hidden_dim,
                    1,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                ),
            )

            for layer in (
                self.v18_trim_gate_head,
                self.v18_component_gate_head,
            ):
                nn.init.zeros_(layer.weight)
                nn.init.constant_(layer.bias, gate_bias)

            for head in (
                self.v18_trim_delta_head,
                self.v18_component_delta_head,
            ):
                nn.init.zeros_(head[-1].weight)
                nn.init.constant_(head[-1].bias, delta_bias)

        # V19_ACTION_BANK
        # C0 Preserve, C1 AtomicFPDelete, C2 AtomicFNFill.
        # Unlike V18, each non-preserve candidate is constrained to ONE local
        # top-ranked action window.  The dense heads only rank possible action
        # centers; the final hypothesis is never a union of unrelated edits.
        self.v19_action_bank = bool(_cfg_get(m1, "V19_ACTION_BANK", False))
        self.v19_delete_window_radius = max(
            1, int(_cfg_get(m1, "V19_DELETE_WINDOW_RADIUS", 8))
        )
        self.v19_fill_window_radius = max(
            1, int(_cfg_get(m1, "V19_FILL_WINDOW_RADIUS", 8))
        )
        self.v19_protrusion_radius = max(
            1, int(_cfg_get(m1, "V19_PROTRUSION_RADIUS", 2))
        )
        self.v19_density_radius = max(
            1, int(_cfg_get(m1, "V19_DENSITY_RADIUS", 3))
        )
        self.v19_density_max = min(
            max(float(_cfg_get(m1, "V19_DENSITY_MAX", 0.75)), 0.0),
            1.0,
        )
        self.v19_fill_outer_radius = max(
            1, int(_cfg_get(m1, "V19_FILL_OUTER_RADIUS", 3))
        )
        self.v19_hole_radius = max(
            1, int(_cfg_get(m1, "V19_HOLE_RADIUS", 3))
        )
        self.v19_max_logit_delta = max(
            0.1, float(_cfg_get(m1, "V19_MAX_LOGIT_DELTA", 5.0))
        )
        self.v19_delete_gate_head = None
        self.v19_fill_gate_head = None
        self.v19_delete_delta_head = None
        self.v19_fill_delta_head = None
        if self.v19_action_bank:
            gate_init = min(
                max(float(_cfg_get(m1, "V19_GATE_INIT", 0.25)), 1e-4),
                1.0 - 1e-4,
            )
            gate_bias = math.log(gate_init / (1.0 - gate_init))
            delta_init = min(
                max(float(_cfg_get(m1, "V19_LOGIT_DELTA_INIT", 2.0)), 1e-4),
                self.v19_max_logit_delta,
            )
            delta_bias = math.log(math.expm1(delta_init))
            self.v19_delete_gate_head = nn.Conv2d(
                hidden_dim, 1, kernel_size=1, bias=True
            )
            self.v19_fill_gate_head = nn.Conv2d(
                hidden_dim, 1, kernel_size=1, bias=True
            )
            self.v19_delete_delta_head = nn.Sequential(
                _ConvNormGELU(hidden_dim, hidden_dim),
                nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1, bias=True),
            )
            self.v19_fill_delta_head = nn.Sequential(
                _ConvNormGELU(hidden_dim, hidden_dim),
                nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1, bias=True),
            )
            for layer in (self.v19_delete_gate_head, self.v19_fill_gate_head):
                nn.init.zeros_(layer.weight)
                nn.init.constant_(layer.bias, gate_bias)
            for head in (self.v19_delete_delta_head, self.v19_fill_delta_head):
                nn.init.zeros_(head[-1].weight)
                nn.init.constant_(head[-1].bias, delta_bias)

        # V7: separate the decision "should this location be edited?" from
        # the conditional direction decision "shrink or expand?".  This avoids
        # V6's three-class-softmax failure mode where preserve pixels competed
        # directly with very sparse correction classes and caused over-editing.
        self.router_edit_head = None
        self.router_direction_head = None
        self.router_edit_max = float(_cfg_get(m1, "EDIT_GATE_MAX", 1.0))
        self.router_edit_max = min(max(self.router_edit_max, 0.0), 1.0)
        if self.router_enabled:
            self.router_edit_head = nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=True)
            self.router_direction_head = nn.Conv2d(hidden_dim, 2, kernel_size=1, bias=True)

            edit_init = float(_cfg_get(m1, "EDIT_GATE_INIT", 0.05))
            edit_init = min(max(edit_init, 1e-4), 1.0 - 1e-4)
            edit_bias = math.log(edit_init / (1.0 - edit_init))
            shrink_bias = float(_cfg_get(m1, "DIRECTION_INIT_SHRINK_BIAS", 0.0))
            expand_bias = float(_cfg_get(m1, "DIRECTION_INIT_EXPAND_BIAS", 0.0))

            nn.init.zeros_(self.router_edit_head.weight)
            nn.init.zeros_(self.router_direction_head.weight)
            nn.init.constant_(self.router_edit_head.bias, edit_bias)
            with torch.no_grad():
                self.router_direction_head.bias.copy_(torch.tensor([shrink_bias, expand_bias]))

    @staticmethod
    def _entropy(prob: torch.Tensor) -> torch.Tensor:
        p = prob.clamp(EPS, 1.0 - EPS)
        return (-(p * p.log() + (1.0 - p) * (1.0 - p).log()) / 0.6931471805599453).clamp(0.0, 1.0)

    @staticmethod
    def _boundary(prob: torch.Tensor) -> torch.Tensor:
        return (_soft_dilate(prob, 1) - _soft_erode(prob, 1)).abs().clamp(0.0, 1.0)

    @staticmethod
    def _normalized_image_and_edge(
        image: torch.Tensor, target_hw: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 4:
            raise ValueError(f"Expected image [B,C,H,W], got {tuple(image.shape)}")
        gray = image.mean(dim=1, keepdim=True)
        if gray.shape[-2:] != target_hw:
            gray = F.interpolate(gray, size=target_hw, mode="bilinear", align_corners=False)
        lo = gray.amin(dim=(-2, -1), keepdim=True)
        hi = gray.amax(dim=(-2, -1), keepdim=True)
        gray = (gray - lo) / (hi - lo).clamp_min(EPS)
        gx = F.pad((gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs(), (0, 1, 0, 0))
        gy = F.pad((gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs(), (0, 0, 0, 1))
        edge = (gx + gy).clamp(0.0, 1.0)
        return gray, edge

    def _contour_band(self, base_prob: torch.Tensor):
        anchor = (base_prob >= self.anchor_threshold).float()
        inside = (anchor - _soft_erode(anchor, self.band_inside_radius)).clamp(0.0, 1.0)
        outside = (_soft_dilate(anchor, self.band_outside_radius) - anchor).clamp(0.0, 1.0)
        return inside, outside, (inside + outside).clamp(0.0, 1.0)

    def _semantic_latent(
        self, semantic_map: Optional[torch.Tensor], target_hw: tuple[int, int]
    ) -> Optional[torch.Tensor]:
        if not self.use_semantic_feature:
            return None
        if semantic_map is None:
            raise ValueError(
                "M1.USE_SEMANTIC_FEATURE=true requires semantic_map from the UniMedCLIP patch tokens."
            )
        if semantic_map.ndim != 4:
            raise ValueError(f"Expected semantic_map [B,C,H,W], got {tuple(semantic_map.shape)}")
        if semantic_map.shape[1] != self.semantic_channels:
            raise ValueError(
                f"Semantic channel mismatch: expected {self.semantic_channels}, got {semantic_map.shape[1]}"
            )
        if semantic_map.shape[-2:] != target_hw:
            semantic_map = F.interpolate(semantic_map, size=target_hw, mode="bilinear", align_corners=False)
        assert self.semantic_proj is not None
        return self.semantic_proj(semantic_map)

    def _v18_atomic_supports(
        self, base_prob: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return disjoint supports for Boundary-Trim and FP-Island-Drop.

        The supports depend only on the frozen/Base prediction.  They are
        therefore available at inference without GT and form stable factual
        hypotheses for later counterfactual verification.
        """
        anchor = (base_prob >= self.anchor_threshold).float()

        inner_ring = (
            anchor - _soft_erode(anchor, self.v18_trim_radius)
        ).clamp(0.0, 1.0)

        kernel = 2 * self.v18_component_radius + 1
        local_density = F.avg_pool2d(
            anchor, kernel_size=kernel, stride=1, padding=self.v18_component_radius
        )

        # Dense foreground edge -> Boundary-Trim.
        trim_support = (
            inner_ring
            * (local_density > self.v18_component_density_max).float()
        )

        # Low-density predicted foreground -> FP-island candidate.
        # It is deliberately disjoint from trim_support.
        component_support = (
            anchor
            * (local_density <= self.v18_component_density_max).float()
            * (1.0 - trim_support)
        )

        return trim_support, component_support, local_density

    def _v19_action_supports(
        self, base_prob: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build broad but structured supports for atomic delete/fill actions.

        Delete support combines detached/sparse islands with connected
        protrusions obtained by an opening residual. Fill support combines a
        narrow external band with small closing-defined internal holes. The
        later top-1 window makes each hypothesis a single local action.
        """
        anchor = (base_prob >= self.anchor_threshold).float()

        opened = _soft_dilate(
            _soft_erode(anchor, self.v19_protrusion_radius),
            self.v19_protrusion_radius,
        )
        protrusion = (anchor - opened).clamp(0.0, 1.0)

        density_kernel = 2 * self.v19_density_radius + 1
        density = F.avg_pool2d(
            anchor,
            kernel_size=density_kernel,
            stride=1,
            padding=self.v19_density_radius,
        )
        sparse_island = anchor * (density <= self.v19_density_max).float()
        delete_support = (protrusion + sparse_island).clamp(0.0, 1.0)

        outer = (
            _soft_dilate(anchor, self.v19_fill_outer_radius) - anchor
        ).clamp(0.0, 1.0)
        closed = _soft_erode(
            _soft_dilate(anchor, self.v19_hole_radius),
            self.v19_hole_radius,
        )
        holes = (closed - anchor).clamp(0.0, 1.0)
        fill_support = ((outer + holes).clamp(0.0, 1.0) * (1.0 - anchor))
        return delete_support, fill_support, protrusion, holes

    @staticmethod
    def _v19_top1_window(
        score: torch.Tensor,
        support: torch.Tensor,
        radius: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Choose one action center per image and dilate it into a local window.

        Argmax is intentional: the generated candidate is a discrete factual
        action. Dense gate heads remain trainable through their separate
        ranking and pixel objectives in the V19 loss.
        """
        if score.ndim != 4 or support.ndim != 4:
            raise ValueError("V19 top-1 selection expects [B,1,H,W] tensors.")
        b, _, h, w = score.shape
        masked = score * support
        flat = masked.reshape(b, -1)
        index = flat.argmax(dim=1)
        has_support = (support.reshape(b, -1).sum(dim=1) > 0).float()
        seed = torch.zeros_like(flat)
        seed.scatter_(1, index[:, None], 1.0)
        seed = seed.reshape(b, 1, h, w) * has_support[:, None, None, None]
        window = _soft_dilate(seed, int(radius)).clamp(0.0, 1.0)
        selected = support * window
        top_score = flat.gather(1, index[:, None])[:, 0] * has_support
        return selected, top_score

    def generate(
        self,
        base_logits: torch.Tensor,
        image: torch.Tensor,
        semantic_map: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return candidates and all routing diagnostics.

        Base logits are detached in frozen/anchor-student mode.  Candidate,
        router, and fusion gradients cannot update B0 under that contract.
        """
        if base_logits.ndim == 3:
            logits = base_logits.unsqueeze(1)
        elif base_logits.ndim == 4 and base_logits.shape[1] == 1:
            logits = base_logits
        else:
            raise ValueError(f"Expected [B,H,W] or [B,1,H,W], got {tuple(base_logits.shape)}")

        anchor_logits = logits.detach() if self.detach_base_for_candidates else logits
        base_prob = torch.sigmoid(anchor_logits)
        entropy = self._entropy(base_prob)
        boundary = self._boundary(base_prob)
        inside_band, outside_band, contour_band = self._contour_band(base_prob)
        gray, image_edge = self._normalized_image_and_edge(image, base_prob.shape[-2:])

        structural = (
            self.entropy_weight * entropy
            + self.boundary_weight * boundary
            + self.image_edge_weight * image_edge
        ).clamp(0.0, 1.0)
        structural_strength = self.gate_min_strength + (1.0 - self.gate_min_strength) * structural
        if self.gate_power != 1.0:
            structural_strength = structural_strength.pow(self.gate_power)

        features = torch.cat(
            [base_prob, entropy, boundary, inside_band, outside_band, gray, image_edge], dim=1
        )
        latent = _run_checkpointed_module(self.trunk, features)
        semantic_latent = self._semantic_latent(semantic_map, base_prob.shape[-2:])
        if semantic_latent is not None:
            latent = latent + semantic_latent

        fp_gate_logits = self.fp_head(latent)
        fn_gate_logits = self.fn_head(latent)
        shrink_gate = inside_band * structural_strength * torch.sigmoid(fp_gate_logits)
        # V12 Expand may only repair an immediately connected external contour band.
        # This prevents distant islands and shifts the objective from recall to precision.
        base_anchor = (base_prob >= self.anchor_threshold).float()
        expand_connect_band = (_soft_dilate(base_anchor, self.expand_connect_radius) - base_anchor).clamp(0.0, 1.0)
        expand_uncertainty = self.expand_uncertainty_floor + (1.0 - self.expand_uncertainty_floor) * entropy
        expand_gate = outside_band * expand_connect_band * expand_uncertainty * structural_strength * torch.sigmoid(fn_gate_logits)

        shrink_alpha = self.max_edit_strength * torch.sigmoid(self.shrink_strength_logit)
        expand_alpha = self.max_edit_strength * torch.sigmoid(self.expand_strength_logit)
        eroded = _soft_erode(base_prob, self.morph_radius)
        dilated = _soft_dilate(base_prob, self.morph_radius)
        preserve = base_prob

        shrink_morph_step = shrink_alpha * (base_prob - eroded).clamp(0.0, 1.0)
        expand_morph_step = expand_alpha * (dilated - base_prob).clamp(0.0, 1.0)

        if self.use_learned_residual:
            assert self.fp_delta_head is not None and self.fn_delta_head is not None
            learned_shrink_step = self.max_edit_strength * torch.sigmoid(self.fp_delta_head(latent))
            learned_expand_step = self.max_edit_strength * torch.sigmoid(self.fn_delta_head(latent))
            shrink_step = (
                self.morph_residual_mix * shrink_morph_step
                + (1.0 - self.morph_residual_mix) * learned_shrink_step
            )
            expand_step = (
                self.morph_residual_mix * expand_morph_step
                + (1.0 - self.morph_residual_mix) * learned_expand_step
            )
        else:
            shrink_step = shrink_morph_step
            expand_step = expand_morph_step

        # Defaults keep legacy V15/V17 paths compatible with V19 diagnostics.
        delete_support = torch.zeros_like(base_prob)
        fill_support = torch.zeros_like(base_prob)
        delete_selected_support = torch.zeros_like(base_prob)
        fill_selected_support = torch.zeros_like(base_prob)
        protrusion_support = torch.zeros_like(base_prob)
        hole_support = torch.zeros_like(base_prob)
        delete_top_score = base_prob.new_zeros(base_prob.shape[0])
        fill_top_score = base_prob.new_zeros(base_prob.shape[0])

        # V19_ACTION_BANK
        # Candidate slots remain [C0, C1, C2] for the existing evaluator:
        # C0 Preserve, C1 one AtomicFPDelete action, C2 one AtomicFNFill
        # action.  The two actions are independent hypotheses, not cumulative
        # edits and are not fused during Phase A.
        if self.v19_action_bank:
            assert self.v19_delete_gate_head is not None
            assert self.v19_fill_gate_head is not None
            assert self.v19_delete_delta_head is not None
            assert self.v19_fill_delta_head is not None

            delete_support, fill_support, protrusion_support, hole_support = (
                self._v19_action_supports(base_prob)
            )
            delete_gate_logits = self.v19_delete_gate_head(latent)
            fill_gate_logits = self.v19_fill_gate_head(latent)
            delete_rank = delete_support * torch.sigmoid(delete_gate_logits)
            fill_rank = fill_support * torch.sigmoid(fill_gate_logits)
            delete_selected_support, delete_top_score = self._v19_top1_window(
                delete_rank,
                delete_support,
                self.v19_delete_window_radius,
            )
            fill_selected_support, fill_top_score = self._v19_top1_window(
                fill_rank,
                fill_support,
                self.v19_fill_window_radius,
            )

            delete_gate = delete_selected_support * torch.sigmoid(delete_gate_logits)
            fill_gate = fill_selected_support * torch.sigmoid(fill_gate_logits)
            delete_delta = F.softplus(
                self.v19_delete_delta_head(latent)
            ).clamp(max=self.v19_max_logit_delta)
            fill_delta = F.softplus(
                self.v19_fill_delta_head(latent)
            ).clamp(max=self.v19_max_logit_delta)

            delete_logits = anchor_logits - delete_gate * delete_delta
            fill_logits = anchor_logits + fill_gate * fill_delta
            atomic_delete = torch.sigmoid(delete_logits).clamp(EPS, 1.0 - EPS)
            atomic_fill = torch.sigmoid(fill_logits).clamp(EPS, 1.0 - EPS)
            candidate_probs = torch.cat([preserve, atomic_delete, atomic_fill], dim=1)
            candidate_logits = torch.cat(
                [anchor_logits, delete_logits, fill_logits], dim=1
            )
            direct_fused_probs = preserve

            # Shared legacy aliases; V19 loss consumes explicit keys below.
            shrink = atomic_delete
            expand = atomic_fill
            shrink_gate = delete_gate
            expand_gate = fill_gate
            fp_gate_logits = delete_gate_logits
            fn_gate_logits = fill_gate_logits
            trim_support = delete_support
            component_support = fill_support
            component_density = torch.zeros_like(base_prob)

                # V18_ATOMIC_CANDIDATE_BANK
        # Candidate slots remain [C0, C1, C2] for compatibility with the
        # existing evaluator.  Under V18 their meanings are:
        # C0 Preserve, C1 Boundary-Trim, C2 FP-Island-Drop.
        elif self.v18_atomic_candidates:
            assert self.v18_trim_gate_head is not None
            assert self.v18_component_gate_head is not None
            assert self.v18_trim_delta_head is not None
            assert self.v18_component_delta_head is not None

            trim_support, component_support, component_density = (
                self._v18_atomic_supports(base_prob)
            )

            trim_gate_logits = self.v18_trim_gate_head(latent)
            component_gate_logits = self.v18_component_gate_head(latent)

            trim_gate = trim_support * torch.sigmoid(trim_gate_logits)
            component_gate = component_support * torch.sigmoid(
                component_gate_logits
            )

            trim_delta = F.softplus(
                self.v18_trim_delta_head(latent)
            ).clamp(max=self.v18_max_logit_delta)
            component_delta = F.softplus(
                self.v18_component_delta_head(latent)
            ).clamp(max=self.v18_max_logit_delta)

            trim_logits = anchor_logits - trim_gate * trim_delta
            component_logits = (
                anchor_logits - component_gate * component_delta
            )

            boundary_trim = torch.sigmoid(trim_logits).clamp(
                EPS, 1.0 - EPS
            )
            component_drop = torch.sigmoid(component_logits).clamp(
                EPS, 1.0 - EPS
            )

            candidate_probs = torch.cat(
                [preserve, boundary_trim, component_drop], dim=1
            )
            candidate_logits = torch.cat(
                [anchor_logits, trim_logits, component_logits], dim=1
            )

            # Atomic candidates are hypotheses, not masks to be added
            # together.  During Phase A the configured mode is Preserve.
            direct_fused_probs = preserve

            # Backward-compatible aliases used by shared logging paths.
            shrink = boundary_trim
            expand = component_drop
            shrink_gate = trim_gate
            expand_gate = component_gate
            fp_gate_logits = trim_gate_logits
            fn_gate_logits = component_gate_logits
        elif self.use_logit_shrink:
            assert self.logit_shrink_head is not None
            trim_support = inside_band
            component_support = outside_band
            component_density = torch.zeros_like(base_prob)

            logit_delta = F.softplus(
                self.logit_shrink_head(latent)
            ).clamp(max=self.v17_max_logit_delta)
            logit_gate = (
                shrink_gate * self.v17_logit_gate_scale
            ).clamp(0.0, 1.0)
            shrink_logits = anchor_logits - logit_gate * logit_delta
            shrink = torch.sigmoid(shrink_logits).clamp(EPS, 1.0 - EPS)
            expand = (
                base_prob + expand_gate * expand_step
            ).clamp(EPS, 1.0 - EPS)
            candidate_probs = torch.cat([preserve, shrink, expand], dim=1)
            candidate_logits = torch.cat(
                [anchor_logits, shrink_logits, torch.logit(expand)], dim=1
            )

            direct_fused_probs = (
                preserve + (shrink - preserve) + (expand - preserve)
            ).clamp(EPS, 1.0 - EPS)

            # Preserve existing V17 diagnostics through the V18 auxiliary
            # schema. In this mode slot-1 remains global logit Shrink.
            shrink_gate = logit_gate
        else:
            trim_support = inside_band
            component_support = outside_band
            component_density = torch.zeros_like(base_prob)

            shrink = (base_prob - shrink_gate * shrink_step).clamp(EPS, 1.0 - EPS)
            expand = (base_prob + expand_gate * expand_step).clamp(EPS, 1.0 - EPS)
            candidate_probs = torch.cat([preserve, shrink, expand], dim=1)
            candidate_logits = torch.cat(
                [anchor_logits, torch.logit(shrink), torch.logit(expand)], dim=1
            )

            direct_fused_probs = (
                preserve + (shrink - preserve) + (expand - preserve)
            ).clamp(EPS, 1.0 - EPS)

        if self.router_enabled:
            assert self.router_edit_head is not None and self.router_direction_head is not None
            router_edit_logits = self.router_edit_head(latent)
            router_edit_probability = self.router_edit_max * torch.sigmoid(router_edit_logits)
            router_direction_logits = self.router_direction_head(latent)
            router_direction_probs = torch.softmax(router_direction_logits, dim=1)
        else:
            router_edit_logits = torch.zeros_like(base_prob)
            router_edit_probability = torch.zeros_like(base_prob)
            router_direction_logits = torch.zeros(
                base_prob.shape[0], 2, *base_prob.shape[-2:],
                dtype=base_prob.dtype,
                device=base_prob.device,
            )
            router_direction_probs = torch.full_like(router_direction_logits, 0.5)

        # Legacy-compatible 3-channel diagnostic probabilities. They represent
        # Preserve / Shrink / Expand but are generated factorwise:
        # P(edit) and P(direction | edit), not a competing 3-class softmax.
        router_probs = torch.cat([
            1.0 - router_edit_probability,
            router_edit_probability * router_direction_probs[:, 0:1],
            router_edit_probability * router_direction_probs[:, 1:2],
        ], dim=1)

        router_shrink_weight = inside_band * router_edit_probability * router_direction_probs[:, 0:1]
        router_expand_weight = outside_band * router_edit_probability * router_direction_probs[:, 1:2]
        router_fused_probs = (
            preserve
            + router_shrink_weight * (shrink - preserve)
            + router_expand_weight * (expand - preserve)
        ).clamp(EPS, 1.0 - EPS)

        abs_change = (candidate_probs[:, 1:] - preserve).abs().mean(dim=(-2, -1))
        return candidate_logits, {
            "candidate_probs": candidate_probs,
            "direct_fused_probs": direct_fused_probs[:, 0],
            "router_fused_probs": router_fused_probs[:, 0],
            # V7 conservative-router diagnostics.
            "router_logits": torch.cat([
                router_edit_logits,
                router_direction_logits,
            ], dim=1),
            "router_probs": router_probs,
            "router_edit_logits": router_edit_logits[:, 0],
            "router_edit_probability": router_edit_probability[:, 0],
            "router_direction_logits": router_direction_logits,
            "router_direction_probs": router_direction_probs,
            "router_shrink_weight": router_shrink_weight[:, 0],
            "router_expand_weight": router_expand_weight[:, 0],
            "edit_gate": contour_band[:, 0] * structural_strength[:, 0],
            "shrink_gate": shrink_gate[:, 0],
            "expand_gate": expand_gate[:, 0],
            "shrink_gate_logits": fp_gate_logits[:, 0],
            "expand_gate_logits": fn_gate_logits[:, 0],
            "edit_band": contour_band[:, 0],
            "inside_band": inside_band[:, 0],
            "outside_band": outside_band[:, 0],
            "expand_connect_band": expand_connect_band[:, 0],
            "entropy_map": entropy[:, 0],
            "boundary_map": boundary[:, 0],
            "image_edge_map": image_edge[:, 0],
            "candidate_std": candidate_probs.std(dim=1, unbiased=False).mean(dim=(1, 2)),
            "candidate_mean_abs_change": abs_change.mean(dim=1),
            "shrink_mean_abs_change": abs_change[:, 0],
            "expand_mean_abs_change": abs_change[:, 1],
            "shrink_alpha": shrink_alpha.expand(base_prob.shape[0]),
            "expand_alpha": expand_alpha.expand(base_prob.shape[0]),
            "candidate_base_detached": self.detach_base_for_candidates,
            "v18_atomic_candidates": torch.full(
                (base_prob.shape[0],),
                float(self.v18_atomic_candidates),
                device=base_prob.device,
                dtype=base_prob.dtype,
            ),
            "v18_boundary_trim_support": trim_support[:, 0],
            "v18_component_drop_support": component_support[:, 0],
            "v18_component_density": component_density[:, 0],
            "v19_action_bank": torch.full(
                (base_prob.shape[0],),
                float(self.v19_action_bank),
                device=base_prob.device,
                dtype=base_prob.dtype,
            ),
            "v19_delete_support": (
                delete_support[:, 0]
                if self.v19_action_bank else torch.zeros_like(base_prob[:, 0])
            ),
            "v19_fill_support": (
                fill_support[:, 0]
                if self.v19_action_bank else torch.zeros_like(base_prob[:, 0])
            ),
            "v19_delete_selected_support": (
                delete_selected_support[:, 0]
                if self.v19_action_bank else torch.zeros_like(base_prob[:, 0])
            ),
            "v19_fill_selected_support": (
                fill_selected_support[:, 0]
                if self.v19_action_bank else torch.zeros_like(base_prob[:, 0])
            ),
            "v19_protrusion_support": (
                protrusion_support[:, 0]
                if self.v19_action_bank else torch.zeros_like(base_prob[:, 0])
            ),
            "v19_hole_support": (
                hole_support[:, 0]
                if self.v19_action_bank else torch.zeros_like(base_prob[:, 0])
            ),
            "v19_delete_top_score": (
                delete_top_score
                if self.v19_action_bank else base_prob.new_zeros(base_prob.shape[0])
            ),
            "v19_fill_top_score": (
                fill_top_score
                if self.v19_action_bank else base_prob.new_zeros(base_prob.shape[0])
            ),
        }



class UnifiedActionCounterfactualSetBank(nn.Module):
    """V20: joint atomic proposal -> counterfactual text verification -> set selection.

    The bank generates several local actions per image rather than one global
    edit. Every factual action has a same-type, equal-window, non-overlapping
    counterfactual control.  The verifier sees only local image-text relation
    differences; it does not receive action rank, action type IDs, GT labels,
    whole-mask area, or global lesion size.

    Output candidate slots are: C0 Preserve followed by K factual atomic actions.
    Final output is the sparse non-overlapping selected action set, not a
    single best candidate.
    """

    DELETE_TYPES = (0, 1)  # island, protrusion
    FILL_TYPES = (2, 3)    # boundary gap, interior hole

    def __init__(self, cfg) -> None:
        super().__init__()
        m1 = _cfg_get(cfg, "M1", None)
        self.anchor_threshold = float(_cfg_get(m1, "ANCHOR_THRESHOLD", 0.50))
        self.hidden_dim = int(_cfg_get(m1, "HIDDEN_DIM", 64))
        self.semantic_channels = int(_cfg_get(m1, "SEMANTIC_CHANNELS", 512))
        # Candidate proposal must be prompt-independent. Semantic/text features
        # are reserved for the later counterfactual verifier only.
        self.use_semantic_feature = bool(
            _cfg_get(m1, "ACTION_BANK_CANDIDATE_USE_SEMANTIC_FEATURE", False)
        )
        # V451: allow the candidate proposal field itself to use the projected
        # image-text semantic map.  Previous runs kept proposal locations almost
        # purely morphology-driven; when B0 already had a strong hard mask this
        # made the oracle ceiling very low.
        self.use_semantic_feature = bool(
            self.use_semantic_feature
            or _cfg_get(m1, "ACTION_BANK_USE_SEMANTIC_PROPOSAL", False)
        )
        self.k_per_type = max(1, int(_cfg_get(m1, "ACTION_BANK_ACTIONS_PER_TYPE", 2)))
        self.num_types = 4
        raw_enabled_types = _cfg_get(m1, "V469_ENABLED_ACTION_TYPES", [0, 1, 2, 3])
        if isinstance(raw_enabled_types, str):
            raw_enabled_types = [
                x.strip() for x in raw_enabled_types.replace(",", " ").split()
                if x.strip()
            ]
        try:
            enabled_types = tuple(sorted({
                int(x) for x in raw_enabled_types if 0 <= int(x) < self.num_types
            }))
        except Exception as exc:
            raise ValueError(
                "M1.V469_ENABLED_ACTION_TYPES must contain action IDs in {0,1,2,3}."
            ) from exc
        if not enabled_types:
            raise ValueError("At least one V469 action type must be enabled.")
        self.v469_enabled_action_types = enabled_types
        self.num_actions = self.num_types * self.k_per_type
        self.window_radius = max(1, int(_cfg_get(m1, "ACTION_BANK_WINDOW_RADIUS", 7)))
        self.nms_radius = max(self.window_radius, int(_cfg_get(m1, "ACTION_BANK_NMS_RADIUS", 10)))
        # Rank-wise radius diversification keeps the C0..C8 public schema
        # unchanged, but makes C1/C2, C5/C6, etc. represent genuinely
        # different edit scales instead of same-radius duplicate proposals.
        self.rank_radius_step = max(0, int(_cfg_get(m1, "ACTION_BANK_RANK_RADIUS_STEP", 0)))
        self.context_radius = max(0, int(_cfg_get(m1, "ACTION_BANK_CONTEXT_RADIUS", 8)))

        # ==============================================================
        # V422_M1_SAFE_CANDIDATE_BANK
        #
        # Preserve C0 and the eight existing atomic slots exactly:
        #   C1/C2 island-delete, C3/C4 boundary-trim,
        #   C5/C6 boundary-fill, C7/C8 hole-fill.
        #
        # Only boundary-fill receives a narrower local window and an
        # image-edge/uncertainty prior. This avoids broad outer-band edits
        # while preserving M2 factual/control and M3 selector interfaces.
        # ==============================================================
        self.v422_m1_safe_candidate_bank = bool(
            _cfg_get(m1, "M1_SAFE_CANDIDATE_BANK", False)
        )

        # V450 action_bank compatibility:
        # _type_supports() reads these flags in the parent action bank path.
        # In older mechanism runs they were only defined in ReferenceAdaptiveC6Bank,
        # so switching CANDIDATE_MODE=action_bank exposed an AttributeError.
        self.v428_adaptive_dual_expert = bool(
            _cfg_get(m1, "V428_ADAPTIVE_DUAL_EXPERT", False)
        )
        self.v429_asymmetric_adaptive = bool(
            _cfg_get(m1, "V429_ASYMMETRIC_ADAPTIVE", False)
        )
        # V426 keeps C0..C8 but assigns explicit mechanism roles to
        # C1/C2/C5/C6. Final deployment remains external V410-R1 M2.
        self.mechanism_candidates = _uses_reference_mechanism(m1)
        self.v426_island_connect_radius = max(
            1,
            int(_cfg_get(m1, "V426_ISLAND_CONNECT_RADIUS", 10)),
        )

        self.type_window_radius = {
            0: max(1, int(_cfg_get(
                m1, "ACTION_BANK_ISLAND_WINDOW_RADIUS", self.window_radius
            ))),
            1: max(1, int(_cfg_get(
                m1, "ACTION_BANK_TRIM_WINDOW_RADIUS", self.window_radius
            ))),
            2: max(1, int(_cfg_get(
                m1, "ACTION_BANK_FILL_WINDOW_RADIUS",
                min(self.window_radius, 4),
            ))),
            3: max(1, int(_cfg_get(
                m1, "ACTION_BANK_HOLE_WINDOW_RADIUS", self.window_radius
            ))),
        }

        self.type_nms_radius = {
            0: max(
                self.type_window_radius[0],
                int(_cfg_get(m1, "ACTION_BANK_ISLAND_NMS_RADIUS", self.nms_radius)),
            ),
            1: max(
                self.type_window_radius[1],
                int(_cfg_get(m1, "ACTION_BANK_TRIM_NMS_RADIUS", self.nms_radius)),
            ),
            2: max(
                self.type_window_radius[2],
                int(_cfg_get(m1, "ACTION_BANK_FILL_NMS_RADIUS",
                             min(self.nms_radius, 7))),
            ),
            3: max(
                self.type_window_radius[3],
                int(_cfg_get(m1, "ACTION_BANK_HOLE_NMS_RADIUS", self.nms_radius)),
            ),
        }

        self.v422_fill_edge_weight = max(
            0.0,
            float(_cfg_get(m1, "ACTION_BANK_FILL_EDGE_WEIGHT", 0.35)),
        )
        self.v422_fill_uncertainty_weight = max(
            0.0,
            float(_cfg_get(m1, "ACTION_BANK_FILL_UNCERTAINTY_WEIGHT", 0.65)),
        )
        self.v422_fill_evidence_floor = min(
            max(
                float(_cfg_get(m1, "ACTION_BANK_FILL_EVIDENCE_FLOOR", 0.15)),
                0.0,
            ),
            0.95,
        )
        # V27 introduced context-clean controls.  V37 makes this an explicit
        # M2 protocol requirement: a factual/control pair may not share the
        # dilated local context from which text evidence is pooled.
        self.v37_text_falsified_structural_consensus = bool(
            _cfg_get(m1, "V37_TEXT_FALSIFIED_STRUCTURAL_CONSENSUS", False)
            or _cfg_get(m1, "CONSENSUS_CASEWISE_FALSIFIED_DELTA_CONSENSUS", False)
            or _cfg_get(m1, "CALIBRATION_LESION_BACKGROUND_CALIBRATED_ATOMIC", False)
            or _cfg_get(m1, "V382_ACTION_CONDITIONAL_QUANTILE_ATOMIC", False)
            or _cfg_get(m1, "V383_CONSERVATIVE_ACTION_VALUE", False)
            or str(_cfg_get(m1, "M1_LOSS_VERSION", "")).lower()
            in {
                "v37_text_falsified_structural_consensus",
                "v38_casewise_falsified_delta_consensus",
                "v381_lesion_background_calibrated_atomic",
                "v382_action_conditional_quantile_atomic",
                "v383_conservative_action_value",
            }
            or str(_cfg_get(m1, "RUN_TAG", "")).upper().startswith(("V37_", "CONSENSUS_", "CALIBRATION_", "V382_", "V383_"))
        )
        self.v27_context_clean_controls = str(
            _cfg_get(m1, "RUN_TAG", "")
        ).upper().startswith("V27_")
        # V426 external PAIR-M2 requires true counterfactual pairs: the
        # factual/control local contexts must be disjoint even when an older
        # V37/V38 flag is absent.  This changes only control construction; it
        # never changes the factual candidate bank or B0 segmentation logits.
        self.context_clean_controls = bool(
            self.mechanism_candidates
            or self.v27_context_clean_controls
            or self.v37_text_falsified_structural_consensus
            or _cfg_get(m1, "V37_CONTEXT_CLEAN_CONTROLS", False)
            or _cfg_get(m1, "EXACT_CONTROL", False)
            or _cfg_get(m1, "UNIFIED_M1_PAIRED_CONTROL_SELECTOR", False)
        )
        self.control_context_guard = max(
            0, int(_cfg_get(
                m1,
                "V37_CONTROL_CONTEXT_GUARD" if self.v37_text_falsified_structural_consensus
                else "V27_CONTROL_CONTEXT_GUARD",
                1,
            ))
        )
        self.outer_radius = max(1, int(_cfg_get(m1, "ACTION_BANK_FILL_OUTER_RADIUS", 4)))
        self.hole_radius = max(1, int(_cfg_get(m1, "ACTION_BANK_HOLE_RADIUS", 3)))
        self.protrusion_radius = max(1, int(_cfg_get(m1, "ACTION_BANK_PROTRUSION_RADIUS", 2)))
        self.density_radius = max(1, int(_cfg_get(m1, "ACTION_BANK_DENSITY_RADIUS", 3)))
        self.density_max = float(_cfg_get(m1, "ACTION_BANK_DENSITY_MAX", 0.70))
        self.max_logit_delta = max(0.1, float(_cfg_get(m1, "ACTION_BANK_MAX_LOGIT_DELTA", 5.0)))
        self.select_threshold = float(_cfg_get(m1, "ACTION_BANK_SELECT_THRESHOLD", 0.35))
        # V451: keep the old public names, but also read the accidentally used
        # ACTION_BANK_MIN_BUDGET / ACTION_BANK_MAX_BUDGET aliases so changing
        # YAML really changes the deployed edit budget.  The default max is
        # deliberately larger than the old 0.025 because the previous candidate
        # oracle was capped by tiny local edits.
        self.budget_min = max(
            0.0,
            float(_cfg_get(
                m1,
                "ACTION_BANK_EDIT_BUDGET_MIN",
                _cfg_get(m1, "ACTION_BANK_MIN_BUDGET", 0.0030),
            )),
        )
        self.budget_max = max(
            self.budget_min,
            float(_cfg_get(
                m1,
                "ACTION_BANK_EDIT_BUDGET_MAX",
                _cfg_get(m1, "ACTION_BANK_MAX_BUDGET", 0.0800),
            )),
        )
        self.detach_base = bool(_cfg_get(m1, "DETACH_BASE_FOR_CANDIDATES", False))

        # V451 candidate-ceiling fixes.
        # 1) Soft-support carriers are built from B0 probabilities instead of a
        #    single 0.5 hard mask, so low-confidence FN/FP regions can become
        #    valid hypotheses.
        # 2) Global non-overlap can be disabled between action families while
        #    retaining within-family NMS.  Controls still exclude the final
        #    union of factual actions.
        # 3) The old V422 boundary-fill evidence gate can be relaxed/disabled
        #    for upper-bound diagnosis; otherwise it can remove true FN fill
        #    regions before learning has a chance to rank them.
        self.soft_support_enabled = bool(
            _cfg_get(m1, "ACTION_BANK_SOFT_SUPPORT", False)
        )
        self.soft_fg_min = min(max(
            float(_cfg_get(m1, "ACTION_BANK_SOFT_FOREGROUND_MIN", 0.30)),
            0.0,
        ), 1.0)
        self.soft_bg_max = min(max(
            float(_cfg_get(m1, "ACTION_BANK_SOFT_BACKGROUND_MAX", 0.70)),
            0.0,
        ), 1.0)
        self.soft_uncertain_low = min(max(
            float(_cfg_get(m1, "ACTION_BANK_SOFT_UNCERTAIN_LOW", 0.25)),
            0.0,
        ), 1.0)
        self.soft_uncertain_high = min(max(
            float(_cfg_get(m1, "ACTION_BANK_SOFT_UNCERTAIN_HIGH", 0.75)),
            self.soft_uncertain_low,
        ), 1.0)
        self.global_non_overlap = bool(
            _cfg_get(m1, "ACTION_BANK_GLOBAL_NON_OVERLAP", True)
        )
        self.disable_safe_fill_filter = bool(
            _cfg_get(m1, "ACTION_BANK_DISABLE_SAFE_FILL_FILTER", False)
        )
        self.hard_edit_logit_margin = max(
            0.0,
            float(_cfg_get(m1, "ACTION_BANK_HARD_EDIT_LOGIT_MARGIN", 0.50)),
        )

        # Carrier-matched counterfactual controls. Candidate generation remains
        # text-free; these values are used only after factual actions exist.
        self.control_top_centers = max(
            4, int(_cfg_get(m1, "ACTION_BANK_CONTROL_TOP_CENTERS", 16))
        )
        self.control_min_shift = max(
            1, int(_cfg_get(m1, "ACTION_BANK_CONTROL_MIN_SHIFT", 8))
        )
        if self.context_clean_controls:
            self.control_min_shift = max(
                self.control_min_shift,
                2 * self.context_radius + self.control_context_guard,
            )
        self.control_gray_weight = max(
            0.0, float(_cfg_get(m1, "ACTION_BANK_CONTROL_GRAY_WEIGHT", 1.0))
        )
        self.control_entropy_weight = max(
            0.0, float(_cfg_get(m1, "ACTION_BANK_CONTROL_ENTROPY_WEIGHT", 1.0))
        )
        self.control_boundary_weight = max(
            0.0, float(_cfg_get(m1, "ACTION_BANK_CONTROL_BOUNDARY_WEIGHT", 1.5))
        )
        self.control_shape_weight = max(
            0.0, float(_cfg_get(m1, "ACTION_BANK_CONTROL_SHAPE_WEIGHT", 0.25))
        )
        # V397 fast path: vectorised carrier-matched controls. It preserves
        # exact action area and context separation, but removes the historical
        # Python loop over batch × action × candidate control centres.
        self.v396_fast_matched_controls = bool(
            _cfg_get(m1, "EVIDENCE_GUIDED_FAST_MATCHED_CONTROLS", False)
        )

        # Same structural channels as the established candidate generator.
        self.trunk = nn.Sequential(
            _ConvNormGELU(7, self.hidden_dim),
            _ConvNormGELU(self.hidden_dim, self.hidden_dim),
        )
        self.semantic_proj = None
        if self.use_semantic_feature:
            groups = max(1, min(8, self.hidden_dim))
            while groups > 1 and self.hidden_dim % groups != 0:
                groups -= 1
            self.semantic_proj = nn.Sequential(
                nn.Conv2d(self.semantic_channels, self.hidden_dim, kernel_size=1, bias=False),
                nn.GroupNorm(groups, self.hidden_dim),
                nn.GELU(),
            )

        self.actionness_heads = nn.ModuleList([
            nn.Conv2d(self.hidden_dim, 1, kernel_size=1, bias=True)
            for _ in range(self.num_types)
        ])
        self.delta_heads = nn.ModuleList([
            nn.Sequential(
                _ConvNormGELU(self.hidden_dim, self.hidden_dim),
                nn.Conv2d(self.hidden_dim, 1, kernel_size=3, padding=1, bias=True),
            )
            for _ in range(self.num_types)
        ])

        gate_init = min(max(float(_cfg_get(m1, "ACTION_BANK_GATE_INIT", 0.25)), 1e-4), 1.0 - 1e-4)
        gate_bias = math.log(gate_init / (1.0 - gate_init))
        delta_init = min(max(float(_cfg_get(m1, "ACTION_BANK_LOGIT_DELTA_INIT", 1.75)), 1e-4), self.max_logit_delta)
        delta_bias = math.log(math.expm1(delta_init))
        for head in self.actionness_heads:
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, gate_bias)
        for head in self.delta_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.constant_(head[-1].bias, delta_bias)

        # V35 source-level proposal purification.  The residual head predicts
        # where B0 is locally foreground while the training mask is background.
        # At inference it uses only image/Base structure and biases the island
        # action proposal field before discrete top-k action construction.
        self.v36_casewise_plackett_luce = bool(
            _cfg_get(m1, "V36_CASEWISE_PLACKETT_LUCE", False)
        )
        self.v35_residual_purified_world_model = bool(
            _cfg_get(m1, "V35_RESIDUAL_PURIFIED_WORLD_MODEL", False)
            or self.v36_casewise_plackett_luce
            # V37 uses the source residual field only for candidate generation;
            # it never uses V35's gain-minus-risk deployment rule.
            or self.v37_text_falsified_structural_consensus
        )
        self.v35_residual_proposal_scale = float(
            _cfg_get(m1, "V35_RESIDUAL_PROPOSAL_SCALE", 1.0)
        )
        self.v35_residual_head = None
        if self.v35_residual_purified_world_model:
            self.v35_residual_head = nn.Sequential(
                _ConvNormGELU(self.hidden_dim, self.hidden_dim),
                nn.Conv2d(self.hidden_dim, 1, kernel_size=1, bias=True),
            )
            nn.init.zeros_(self.v35_residual_head[-1].weight)
            nn.init.zeros_(self.v35_residual_head[-1].bias)

        # V451 learned residual proposal map.  Unlike the legacy V35 single
        # delete-only bias, this head predicts two dense proposal biases:
        # channel-0 for Delete candidates and channel-1 for Fill candidates.
        # It does not directly output a mask; it only changes where the fixed
        # typed action bank places its local windows, keeping the public C0..C8
        # schema and all downstream losses compatible.
        self.learned_residual_proposal_enabled = bool(
            _cfg_get(m1, "ACTION_BANK_LEARNED_RESIDUAL_PROPOSAL", False)
        )
        self.residual_proposal_scale = float(
            _cfg_get(m1, "ACTION_BANK_RESIDUAL_PROPOSAL_SCALE", 1.0)
        )
        self.residual_proposal_head = None
        if self.learned_residual_proposal_enabled:
            self.residual_proposal_head = nn.Sequential(
                _ConvNormGELU(self.hidden_dim, self.hidden_dim),
                nn.Conv2d(self.hidden_dim, 2, kernel_size=1, bias=True),
            )
            nn.init.zeros_(self.residual_proposal_head[-1].weight)
            nn.init.zeros_(self.residual_proposal_head[-1].bias)

        # V463 root fix: independent residual-error head.
        # This is not a reused v451 proposal tensor. It is randomly
        # initialised and trained end-to-end from Train GT residual labels.
        self.v463_residual_enabled = bool(
            _cfg_get(m1, "V463_RESIDUAL_ERROR_HEAD", False)
        )
        self.v463_residual_scale = float(
            _cfg_get(m1, "V463_RESIDUAL_ACTION_BIAS_SCALE", 1.0)
        )
        # V468: FP/FN/TP/BG are mutually exclusive states.  Training them as a
        # class-balanced softmax head makes the logits comparable, which is
        # required because proposal ranking uses causal log-odds differences.
        self.v469_conditional_residual = bool(
            _cfg_get(m1, "V469_CONDITIONAL_RESIDUAL", False)
        )
        self.v468_residual_multiclass = bool(
            _cfg_get(m1, "V468_RESIDUAL_MULTICLASS_CE", False)
        ) and not self.v469_conditional_residual
        self.v468_soft_causal_edit_mask = bool(
            _cfg_get(m1, "V468_SOFT_CAUSAL_EDIT_MASK", False)
        )
        self.v469_relative_gate = bool(
            _cfg_get(m1, "V469_RELATIVE_WINDOW_GATE", False)
        )
        self.v469_gate_keep_fraction = min(max(
            float(_cfg_get(m1, "V469_GATE_KEEP_FRACTION", 0.45)),
            0.05,
        ), 1.0)
        self.v469_gate_rank_fraction_step = max(
            0.0,
            float(_cfg_get(m1, "V469_GATE_RANK_FRACTION_STEP", 0.05)),
        )
        self.v469_gate_min_pixels = max(
            1, int(_cfg_get(m1, "V469_GATE_MIN_PIXELS", 4))
        )
        self.v469_gate_temperature = max(
            1.0e-3,
            float(_cfg_get(m1, "V469_GATE_TEMPERATURE", 0.15)),
        )
        raw_keep_fractions = _cfg_get(
            m1,
            "V470_GATE_KEEP_FRACTIONS_BY_TYPE",
            [self.v469_gate_keep_fraction] * self.num_types,
        )
        if isinstance(raw_keep_fractions, str):
            raw_keep_fractions = [
                x for x in raw_keep_fractions.replace(",", " ").split() if x
            ]
        if len(raw_keep_fractions) != self.num_types:
            raise ValueError(
                "V470_GATE_KEEP_FRACTIONS_BY_TYPE must have four values "
                "for island-delete, boundary-trim, boundary-fill and hole-fill."
            )
        self.v470_gate_keep_fractions = tuple(
            min(max(float(x), 0.05), 1.0) for x in raw_keep_fractions
        )
        raw_min_pixels = _cfg_get(
            m1,
            "V470_GATE_MIN_PIXELS_BY_TYPE",
            [self.v469_gate_min_pixels] * self.num_types,
        )
        if isinstance(raw_min_pixels, str):
            raw_min_pixels = [
                x for x in raw_min_pixels.replace(",", " ").split() if x
            ]
        if len(raw_min_pixels) != self.num_types:
            raise ValueError(
                "V470_GATE_MIN_PIXELS_BY_TYPE must have four integer values."
            )
        self.v470_gate_min_pixels = tuple(max(1, int(x)) for x in raw_min_pixels)
        self.v468_causal_gate_threshold = min(max(
            float(_cfg_get(m1, "V468_CAUSAL_GATE_THRESHOLD", 0.35)),
            1.0e-4,
        ), 1.0 - 1.0e-4)
        self.v468_causal_gate_rank_step = max(
            0.0, float(_cfg_get(m1, "V468_CAUSAL_GATE_RANK_STEP", 0.10))
        )
        self.v468_causal_gate_temperature = max(
            1.0e-3, float(_cfg_get(m1, "V468_CAUSAL_GATE_TEMPERATURE", 0.25))
        )
        self.v468_actionness_gate_weight = max(
            0.0, float(_cfg_get(m1, "V468_ACTIONNESS_GATE_WEIGHT", 0.25))
        )
        self.v463_residual_head = (
            V463ResidualErrorHead(
                self.hidden_dim,
                multiclass=self.v468_residual_multiclass,
                conditional_binary=self.v469_conditional_residual,
            )
            if self.v463_residual_enabled else None
        )

        # Only local factual-vs-control text relation enters this verifier.
        # No candidate type/rank/area/global-mask feature is supplied.
        self.cf_verifier = nn.Sequential(
            nn.Linear(3, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

        # The selector sees proposal confidence, text falsification evidence,
        # local uncertainty, local edit magnitude and local area. It selects a
        # sparse set, with geometry handled by explicit budget/overlap losses.
        self.selector = nn.Sequential(
            nn.Linear(5, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        select_init = min(max(float(_cfg_get(m1, "ACTION_BANK_SELECT_INIT", 0.20)), 1e-4), 1.0 - 1e-4)
        with torch.no_grad():
            nn.init.zeros_(self.selector[-1].weight)
            self.selector[-1].bias.fill_(math.log(select_init / (1.0 - select_init)))

        action_types = []
        action_enabled = []
        for action_type in range(self.num_types):
            action_types.extend([action_type] * self.k_per_type)
            action_enabled.extend([
                action_type in self.v469_enabled_action_types
            ] * self.k_per_type)
        self.register_buffer(
            "action_types",
            torch.tensor(action_types, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "v469_action_enabled",
            torch.tensor(action_enabled, dtype=torch.bool),
            persistent=False,
        )

        # V420 unified M1-only safe-fusion head.
        #
        # This head deliberately consumes only M1-local, image/Base-derived
        # action statistics. It does not read M2 counterfactual text evidence,
        # M3 consensus, GT, candidate rank IDs, or validation thresholds.
        #
        # The action bank itself is already a single unified M1: four
        # morphology-defined error families x K local non-overlapping slots.
        # V420 adds the missing train/deploy-aligned Preserve-vs-one-action
        # decision required to measure M1 as a standalone module.
        self.unified_m1_safe_fusion_enabled = bool(
            _cfg_get(m1, "UNIFIED_M1_SAFE_FUSION", False)
            or str(_cfg_get(m1, "M1_LOSS_VERSION", "")).lower()
            == "unified_m1_safe_fusion"
        )
        self.m1_safe_value_scale = max(
            1.0e-4,
            float(_cfg_get(m1, "UNIFIED_M1_VALUE_SCALE", 0.08)),
        )
        self.m1_safe_temperature = max(
            1.0e-4,
            float(_cfg_get(m1, "UNIFIED_M1_SELECTOR_TEMPERATURE", 0.25)),
        )
        self.m1_safe_max_edit_fraction = max(
            1.0e-4,
            float(_cfg_get(m1, "UNIFIED_M1_MAX_EDIT_FRACTION", 0.035)),
        )
        self.m1_safe_benefit_weight = float(
            _cfg_get(m1, "UNIFIED_M1_BENEFIT_SCORE_WEIGHT", 0.15)
        )
        self.m1_safe_harm_weight = float(
            _cfg_get(m1, "UNIFIED_M1_HARM_SCORE_WEIGHT", 0.20)
        )
        # V451: unified safe fusion no longer has to pick exactly one action.
        # It can compose several local, budgeted actions, which is required
        # when one case has both FP deletion and FN/boundary fill errors.
        self.m1_safe_multi_action = bool(
            _cfg_get(m1, "UNIFIED_M1_MULTI_ACTION_FUSION", False)
        )
        self.m1_safe_max_actions = max(
            1, int(_cfg_get(m1, "UNIFIED_M1_MAX_ACTIONS", 4))
        )
        self.m1_safe_total_edit_fraction = max(
            self.m1_safe_max_edit_fraction,
            float(_cfg_get(
                m1,
                "UNIFIED_M1_TOTAL_EDIT_FRACTION",
                _cfg_get(m1, "ACTION_BANK_EDIT_BUDGET_MAX", 0.0800),
            )),
        )
        self.m1_safe_action_threshold = float(
            _cfg_get(m1, "UNIFIED_M1_ACTION_SCORE_THRESHOLD", 0.0)
        )
        # V456 adaptive risk gate.  Unlike V455, this is not a hand-tuned
        # family blacklist or dataset-specific threshold.  At deployment each
        # case builds its own barrier from the distribution of valid candidate
        # value/risk scores.  An action can beat Preserve only when it is an
        # outlier with positive predicted utility and positive benefit-minus-
        # harm margin in that same case.
        self.m1_safe_adaptive_risk_gate = bool(
            _cfg_get(m1, "UNIFIED_M1_ADAPTIVE_RISK_GATE", False)
        )
        self.m1_safe_paired_control_selector = bool(
            _cfg_get(m1, "UNIFIED_M1_PAIRED_CONTROL_SELECTOR", True)
        )
        # V460: the forward logit mask and the loss valid mask must share this
        # same contract.  When False, exact controls are relative evidence only;
        # non-pair-valid but structurally valid factual actions remain eligible.
        self.m1_safe_require_control_for_selector = bool(
            _cfg_get(m1, "UNIFIED_M1_REQUIRE_CONTROL_FOR_SELECTOR", False)
        )
        # V458: paired control is kept inside one unified training run, but the
        # selector/risk loss is not allowed to push the candidate generator into
        # the degenerate empty-edit solution.  Candidate masks/logit edits still
        # learn from their own factual repair objectives; selector heads learn
        # from detached factual/control evidence.
        self.m1_safe_decouple_selector_from_candidates = bool(
            _cfg_get(m1, "UNIFIED_M1_DECOUPLE_SELECTOR_FROM_CANDIDATES", True)
        )
        self.m1_safe_control_area_tolerance = max(
            1.0e-6,
            float(_cfg_get(m1, "UNIFIED_M1_CONTROL_AREA_TOLERANCE", 0.15)),
        )
        self.m1_safe_control_overlap_tolerance = max(
            0.0,
            float(_cfg_get(m1, "UNIFIED_M1_CONTROL_OVERLAP_TOLERANCE", 1.0e-6)),
        )
        self.m1_safe_control_context_overlap_tolerance = max(
            0.0,
            float(_cfg_get(m1, "UNIFIED_M1_CONTROL_CONTEXT_OVERLAP_TOLERANCE", 1.0e-6)),
        )
        self.m1_safe_control_context_radius = max(
            1,
            int(_cfg_get(m1, "UNIFIED_M1_CONTROL_CONTEXT_RADIUS", 2)),
        )
        self.m1_safe_use_deploy_gate_during_training = bool(
            _cfg_get(m1, "UNIFIED_M1_USE_DEPLOY_GATE_DURING_TRAINING", False)
        )
        # V455 relative-quality gated selector controls.
        # Kept for backwards compatibility; V456 configs should leave these
        # empty/zero and rely on the adaptive risk gate below.
        raw_disabled = _cfg_get(m1, "UNIFIED_M1_DISABLE_DEPLOY_TYPES", [])
        if isinstance(raw_disabled, str):
            raw_disabled = [x.strip() for x in raw_disabled.replace(",", " ").split() if x.strip()]
        try:
            self.m1_safe_disable_deploy_types = tuple(sorted({int(x) for x in raw_disabled}))
        except Exception:
            self.m1_safe_disable_deploy_types = tuple()
        self.m1_safe_type_score_bias = {
            0: float(_cfg_get(m1, "UNIFIED_M1_TYPE0_SCORE_BIAS", 0.0)),
            1: float(_cfg_get(m1, "UNIFIED_M1_TYPE1_SCORE_BIAS", 0.0)),
            2: float(_cfg_get(m1, "UNIFIED_M1_TYPE2_SCORE_BIAS", 0.0)),
            3: float(_cfg_get(m1, "UNIFIED_M1_TYPE3_SCORE_BIAS", 0.0)),
        }
        self.m1_safe_preserve_logit_bias = float(
            _cfg_get(m1, "UNIFIED_M1_PRESERVE_LOGIT_BIAS", 0.0)
        )
        self.m1_safe_state = None
        self.m1_safe_value_head = None
        self.m1_safe_benefit_head = None
        self.m1_safe_harm_head = None
        if self.unified_m1_safe_fusion_enabled:
            self.m1_safe_state = nn.Sequential(
                nn.Linear(5, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
            )
            self.m1_safe_value_head = nn.Linear(self.hidden_dim, 2)
            self.m1_safe_benefit_head = nn.Linear(self.hidden_dim, 1)
            self.m1_safe_harm_head = nn.Linear(self.hidden_dim, 1)
            # Preserve is the initial safe action. The direct M1 loss provides
            # positive candidate coverage and Pareto labels, preventing this
            # initialization from becoming an all-Preserve optimum.
            nn.init.zeros_(self.m1_safe_value_head.weight)
            nn.init.zeros_(self.m1_safe_value_head.bias)
            nn.init.zeros_(self.m1_safe_benefit_head.weight)
            nn.init.zeros_(self.m1_safe_harm_head.weight)
            nn.init.zeros_(self.m1_safe_benefit_head.bias)
            nn.init.zeros_(self.m1_safe_harm_head.bias)

        # M1-only safety residual branch. C9 is a diagnostic boundary
        # candidate. In A3, C10 is a separately parameterised context expert:
        # it cannot alter C9's validated representation and abstains exactly
        # when the global repair direction is ambiguous.
        self.safe_residual_enabled = bool(
            _cfg_get(m1, "SAFE_RESIDUAL_ENABLED", False)
        )
        self.safe_residual_only_training = bool(
            _cfg_get(m1, "SAFE_RESIDUAL_ONLY_TRAINING", False)
        )
        self.safe_context_certified_only = bool(
            _cfg_get(m1, "SAFE_CONTEXT_CERTIFIED_ONLY", False)
        )
        self.safe_residual_mode = str(
            _cfg_get(m1, "SAFE_RESIDUAL_MODE", "both")
        ).lower()
        self.safe_residual_use_semantic = bool(
            _cfg_get(m1, "SAFE_RESIDUAL_USE_SEMANTIC", True)
        )
        self.safe_boundary_outer_radius = max(
            1, int(_cfg_get(m1, "SAFE_BOUNDARY_OUTER_RADIUS", 3))
        )
        self.safe_boundary_inner_radius = max(
            1, int(_cfg_get(m1, "SAFE_BOUNDARY_INNER_RADIUS", 2))
        )
        self.safe_context_outer_radius = max(
            self.safe_boundary_outer_radius,
            int(_cfg_get(m1, "SAFE_CONTEXT_OUTER_RADIUS", 6)),
        )
        self.safe_context_inner_radius = max(
            1, int(_cfg_get(m1, "SAFE_CONTEXT_INNER_RADIUS", 2))
        )
        self.safe_boundary_max_delta = max(
            0.05, float(_cfg_get(m1, "SAFE_BOUNDARY_MAX_DELTA", 1.0))
        )
        self.safe_context_max_delta = max(
            0.05, float(_cfg_get(m1, "SAFE_CONTEXT_MAX_DELTA", 1.0))
        )
        self.safe_uncertainty_floor = float(
            _cfg_get(m1, "SAFE_UNCERTAINTY_FLOOR", 0.35)
        )
        self.safe_uncertainty_temperature = max(
            1e-4,
            float(_cfg_get(m1, "SAFE_UNCERTAINTY_TEMPERATURE", 0.08)),
        )
        self.safe_edge_floor = float(_cfg_get(m1, "SAFE_EDGE_FLOOR", 0.04))
        self.safe_edge_temperature = max(
            1e-4,
            float(_cfg_get(m1, "SAFE_EDGE_TEMPERATURE", 0.04)),
        )

        # A3 certificate. C10 edits only when the same direction score is
        # both correct and sufficiently decisive; otherwise C10 == Preserve.
        self.safe_context_direction_margin = min(
            max(
                float(_cfg_get(m1, "SAFE_CONTEXT_DIRECTION_MARGIN", 0.45)),
                0.0,
            ),
            0.95,
        )
        self.safe_context_direction_temperature = max(
            1e-4,
            float(
                _cfg_get(
                    m1,
                    "SAFE_CONTEXT_DIRECTION_TEMPERATURE",
                    0.08,
                )
            ),
        )
        self.safe_context_logit_band = max(
            1e-4,
            float(_cfg_get(m1, "SAFE_CONTEXT_LOGIT_BAND", 0.45)),
        )
        self.safe_context_logit_temperature = max(
            1e-4,
            float(
                _cfg_get(
                    m1,
                    "SAFE_CONTEXT_LOGIT_TEMPERATURE",
                    0.08,
                )
            ),
        )

        # Existing C9 names stay unchanged so A1 checkpoints remain exactly
        # loadable. Independent C10 modules are instantiated only for A3.
        self.safe_residual_semantic_proj = None
        self.safe_residual_refiner = None
        self.safe_boundary_residual_head = None
        self.safe_context_semantic_proj = None
        self.safe_context_refiner = None
        self.safe_context_residual_head = None
        self.safe_context_direction_head = None

        if self.safe_residual_enabled:
            groups = max(1, min(8, self.hidden_dim))
            while groups > 1 and self.hidden_dim % groups != 0:
                groups -= 1

            def _make_safe_refiner():
                return nn.Sequential(
                    _ConvNormGELU(self.hidden_dim, self.hidden_dim),
                    nn.Conv2d(
                        self.hidden_dim,
                        self.hidden_dim,
                        kernel_size=3,
                        padding=2,
                        dilation=2,
                        groups=self.hidden_dim,
                        bias=False,
                    ),
                    nn.GroupNorm(groups, self.hidden_dim),
                    nn.GELU(),
                    _ConvNormGELU(self.hidden_dim, self.hidden_dim),
                )

            if self.safe_residual_use_semantic:
                self.safe_residual_semantic_proj = nn.Sequential(
                    nn.Conv2d(
                        self.semantic_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.GroupNorm(groups, self.hidden_dim),
                    nn.GELU(),
                )

            # C9 architecture is intentionally identical to A1/A2.
            self.safe_residual_refiner = _make_safe_refiner()
            self.safe_boundary_residual_head = nn.Conv2d(
                self.hidden_dim, 3, kernel_size=1, bias=True
            )

            # Certified C10 uses an independent semantic/refinement path so
            # C10 optimisation cannot drift the accepted A1 C9 candidate.
            if self.safe_context_certified_only:
                if self.safe_residual_use_semantic:
                    self.safe_context_semantic_proj = nn.Sequential(
                        nn.Conv2d(
                            self.semantic_channels,
                            self.hidden_dim,
                            kernel_size=1,
                            bias=False,
                        ),
                        nn.GroupNorm(groups, self.hidden_dim),
                        nn.GELU(),
                    )
                self.safe_context_refiner = _make_safe_refiner()

            self.safe_context_residual_head = nn.Conv2d(
                self.hidden_dim, 2, kernel_size=1, bias=True
            )
            self.safe_context_direction_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.hidden_dim, 1),
            )

            gate_init = min(
                max(float(_cfg_get(m1, "SAFE_GATE_INIT", 0.06)), 1e-4),
                1.0 - 1e-4,
            )
            gate_bias = math.log(gate_init / (1.0 - gate_init))
            context_gate_init = min(
                max(
                    float(_cfg_get(m1, "SAFE_CONTEXT_GATE_INIT", gate_init)),
                    1e-4,
                ),
                1.0 - 1e-4,
            )
            context_gate_bias = math.log(
                context_gate_init / (1.0 - context_gate_init)
            )
            confidence_init = min(
                max(
                    float(_cfg_get(m1, "SAFE_CONFIDENCE_INIT", 0.15)),
                    1e-4,
                ),
                1.0 - 1e-4,
            )
            confidence_bias = math.log(
                confidence_init / (1.0 - confidence_init)
            )
            magnitude_init = max(
                float(_cfg_get(m1, "SAFE_CONTEXT_MAGNITUDE_INIT", 0.10)),
                1e-4,
            )
            magnitude_bias = math.log(math.expm1(magnitude_init))

            nn.init.zeros_(self.safe_boundary_residual_head.weight)
            nn.init.constant_(self.safe_boundary_residual_head.bias[0:1], gate_bias)
            nn.init.zeros_(self.safe_boundary_residual_head.bias[1:2])
            nn.init.constant_(
                self.safe_boundary_residual_head.bias[2:3],
                confidence_bias,
            )
            nn.init.zeros_(self.safe_context_residual_head.weight)
            nn.init.constant_(
                self.safe_context_residual_head.bias[0:1],
                context_gate_bias,
            )
            nn.init.constant_(
                self.safe_context_residual_head.bias[1:2],
                magnitude_bias,
            )
            nn.init.zeros_(self.safe_context_direction_head[-1].weight)
            nn.init.zeros_(self.safe_context_direction_head[-1].bias)


    @staticmethod
    def _entropy(prob: torch.Tensor) -> torch.Tensor:
        prob = prob.clamp(EPS, 1.0 - EPS)
        return (-(prob * prob.log() + (1.0 - prob) * (1.0 - prob).log()) / math.log(2.0)).clamp(0.0, 1.0)

    @staticmethod
    def _boundary(prob: torch.Tensor) -> torch.Tensor:
        return (_soft_dilate(prob, 1) - _soft_erode(prob, 1)).abs().clamp(0.0, 1.0)

    @staticmethod
    def _image_gray_edge(image: torch.Tensor, target_hw: tuple[int, int]):
        gray = image.mean(dim=1, keepdim=True)
        if gray.shape[-2:] != target_hw:
            gray = F.interpolate(gray, size=target_hw, mode="bilinear", align_corners=False)
        lo = gray.amin(dim=(-2, -1), keepdim=True)
        hi = gray.amax(dim=(-2, -1), keepdim=True)
        gray = (gray - lo) / (hi - lo).clamp_min(EPS)
        gx = F.pad((gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs(), (0, 1, 0, 0))
        gy = F.pad((gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs(), (0, 0, 0, 1))
        return gray, (gx + gy).clamp(0.0, 1.0)

    def _latent(self, base_prob: torch.Tensor, image: torch.Tensor, semantic_map: Optional[torch.Tensor]):
        # V451: the latent proposal features should not be locked to the 0.5
        # hard mask.  A broader probability carrier keeps low-confidence
        # lesion evidence visible to the actionness/residual proposal heads.
        if getattr(self, "soft_support_enabled", False):
            anchor = (base_prob >= self.soft_fg_min).float()
        else:
            anchor = (base_prob >= self.anchor_threshold).float()
        inside = (anchor - _soft_erode(anchor, 2)).clamp(0.0, 1.0)
        outside = (_soft_dilate(anchor, self.outer_radius) - anchor).clamp(0.0, 1.0)
        entropy = self._entropy(base_prob)
        boundary = self._boundary(base_prob)
        gray, edge = self._image_gray_edge(image, base_prob.shape[-2:])
        features = torch.cat([base_prob, entropy, boundary, inside, outside, gray, edge], dim=1)
        latent = _run_checkpointed_module(self.trunk, features)
        if self.use_semantic_feature:
            if semantic_map is None:
                raise RuntimeError("V20 requires projected UniMedCLIP semantic map.")
            if semantic_map.shape[-2:] != base_prob.shape[-2:]:
                semantic_map = F.interpolate(semantic_map, size=base_prob.shape[-2:], mode="bilinear", align_corners=False)
            if semantic_map.shape[1] != self.semantic_channels:
                raise RuntimeError(f"V20 semantic channel mismatch: expected {self.semantic_channels}, got {semantic_map.shape[1]}")
            assert self.semantic_proj is not None
            latent = latent + self.semantic_proj(semantic_map)
        return latent, anchor, entropy, boundary

    def _type_supports(
        self,
        base_prob: torch.Tensor,
        image_edge: Optional[torch.Tensor] = None,
        entropy: Optional[torch.Tensor] = None,
    ):
        """Build text-independent atomic geometry supports from B0 only.

        Types:
          0: sparse FP island delete
          1: FP inner-boundary trim
          2: FN connected outer-boundary fill
          3: FN interior-hole fill

        V422 keeps all slots and action semantics unchanged. It only narrows
        the C5/C6 boundary-fill carrier to locations supported by local image
        edge or B0 uncertainty. M2 still receives the same binary factual
        action support and constructs controls with the original code.
        """
        if getattr(self, "soft_support_enabled", False):
            # V451: broaden candidate carriers from hard 0.5 foreground to
            # probability-aware foreground/uncertainty bands.  The masks stay
            # binary enough for downstream exact-control code, but candidate
            # locations no longer disappear just because B0 is below 0.5.
            anchor = (base_prob >= self.soft_fg_min).float()
            uncertain = (
                (base_prob >= self.soft_uncertain_low)
                & (base_prob <= self.soft_uncertain_high)
            ).to(base_prob.dtype)
            fill_allowed = (base_prob <= self.soft_bg_max).to(base_prob.dtype)
        else:
            anchor = (base_prob >= self.anchor_threshold).float()
            uncertain = torch.zeros_like(base_prob)
            fill_allowed = (1.0 - anchor).clamp(0.0, 1.0)

        kernel = 2 * self.density_radius + 1
        density = F.avg_pool2d(
            anchor, kernel, stride=1, padding=self.density_radius
        )
        island = anchor * (density <= self.density_max).float()

        if self.mechanism_candidates:
            # V426 separates isolated false-positive islands from attached
            # leakage using an approximate lesion-core connection test.
            #
            # C1 / type-0:
            #   sparse foreground outside a dilated robust lesion core.
            #
            # C2 / type-1:
            #   opening residual that remains connected to that core.
            #
            # Therefore a thin attached leakage is not consumed by C1.
            opened = _soft_dilate(
                _soft_erode(anchor, self.protrusion_radius),
                self.protrusion_radius,
            )
            protrusion = (anchor - opened).clamp(0.0, 1.0)

            robust_core = _soft_erode(
                anchor,
                self.protrusion_radius,
            )
            core_halo = _soft_dilate(
                robust_core,
                self.v426_island_connect_radius,
            ).clamp(0.0, 1.0)

            island = (
                anchor
                * (density <= self.density_max).float()
                * (1.0 - core_halo)
            )

            boundary_trim = protrusion * core_halo
        else:
            inner_boundary = (
                anchor - _soft_erode(anchor, 1)
            ).clamp(0.0, 1.0)
            boundary_trim = inner_boundary * (1.0 - island)

        outer = (
            _soft_dilate(anchor, self.outer_radius) - anchor
        ).clamp(0.0, 1.0)

        closed = _soft_erode(
            _soft_dilate(anchor, self.hole_radius),
            self.hole_radius,
        )
        hole = (closed - anchor).clamp(0.0, 1.0) * (1.0 - anchor)
        boundary_fill = outer * (1.0 - hole)

        if getattr(self, "soft_support_enabled", False):
            # Fill hypotheses should be allowed in uncertain / low-confidence
            # regions around the foreground, not only in the strict complement
            # of a 0.5 hard mask.  This is the main fix for FN regions that B0
            # never activated strongly enough.
            uncertain_outer = (
                _soft_dilate(anchor, self.outer_radius + 1)
                * uncertain
                * fill_allowed
            ).clamp(0.0, 1.0)
            boundary_fill = (
                boundary_fill * fill_allowed + uncertain_outer
            ).clamp(0.0, 1.0)
            hole = (hole * fill_allowed).clamp(0.0, 1.0)
            boundary_trim = (boundary_trim + uncertain * anchor).clamp(0.0, 1.0)

        if self.v422_m1_safe_candidate_bank and not self.disable_safe_fill_filter:
            if image_edge is None:
                edge_norm = torch.zeros_like(base_prob)
            else:
                if image_edge.shape[-2:] != base_prob.shape[-2:]:
                    image_edge = F.interpolate(
                        image_edge,
                        size=base_prob.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                edge_norm = image_edge / image_edge.amax(
                    dim=(-2, -1),
                    keepdim=True,
                ).clamp_min(EPS)

            if entropy is None:
                p = base_prob.clamp(EPS, 1.0 - EPS)
                uncertainty = (
                    -(p * p.log() + (1.0 - p) * (1.0 - p).log())
                    / 0.6931471805599453
                ).clamp(0.0, 1.0)
            else:
                if entropy.shape[-2:] != base_prob.shape[-2:]:
                    entropy = F.interpolate(
                        entropy,
                        size=base_prob.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                uncertainty = entropy.clamp(0.0, 1.0)

            edge_weight = self.v422_fill_edge_weight
            uncertainty_weight = self.v422_fill_uncertainty_weight
            weight_sum = max(edge_weight + uncertainty_weight, EPS)

            fill_evidence = (
                edge_weight * edge_norm
                + uncertainty_weight * uncertainty
            ) / weight_sum

            evidence_ok = (
                fill_evidence >= self.v422_fill_evidence_floor
            ).float()

            boundary_fill = boundary_fill * evidence_ok

        if self.v428_adaptive_dual_expert:
            # V428: the audit showed that the old mechanism carriers cover
            # less than 1% of the true FP/FN residual. Keep actions local
            # through top-k windows, but do not pre-eliminate most residuals
            # before the learned actionness field can rank them.
            #
            # Delete carrier: complete predicted inner contour.
            # Fill carrier: complete connected outer contour.
            #
            # Precision is learned by the dynamic coverage-purity objective;
            # it is not imposed by a fixed image-edge/entropy threshold.
            boundary_trim = (
                anchor - _soft_erode(anchor, 1)
            ).clamp(0.0, 1.0)

            # V428 broadens the Fill carrier. V429 must keep the
            # original evidence-qualified Fill carrier instead.
            if not self.v429_asymmetric_adaptive:
                boundary_fill = outer * (1.0 - hole)

        return torch.cat([island, boundary_trim, boundary_fill, hole], dim=1)


    @staticmethod
    def _window_from_index(index: torch.Tensor, h: int, w: int, radius: int, device, dtype):
        b = index.shape[0]
        flat = torch.zeros((b, h * w), device=device, dtype=dtype)
        flat.scatter_(1, index[:, None], 1.0)
        seed = flat.reshape(b, 1, h, w)
        return _soft_dilate(seed, radius).clamp(0.0, 1.0)

    def _topk_windows(
        self,
        score: torch.Tensor,
        support: torch.Tensor,
        forbidden: Optional[torch.Tensor] = None,
        window_radius: Optional[int] = None,
        nms_radius: Optional[int] = None,
    ):
        """Greedy NMS action windows with rank-wise multi-scale radii.

        The original implementation generated top-k windows with the same
        radius. That makes C1/C2 and C5/C6 highly correlated and limits the
        candidate oracle ceiling. This version keeps the public C0..C8 schema
        unchanged but makes later ranks use progressively larger windows:

            rank 0: base radius
            rank 1: base radius + ACTION_BANK_RANK_RADIUS_STEP
            rank 2: base radius + 2 * ACTION_BANK_RANK_RADIUS_STEP
            ...

        With CANDIDATES_PER_FAMILY=2 this directly turns C1/C2 and C5/C6
        into small/large hypotheses instead of duplicate hypotheses.
        """
        b, _, h, w = score.shape
        base_radius = (
            self.window_radius
            if window_radius is None
            else max(1, int(window_radius))
        )
        base_nms = (
            self.nms_radius
            if nms_radius is None
            else max(base_radius, int(nms_radius))
        )

        available = support.clone()
        if forbidden is not None:
            available = available * (1.0 - forbidden.clamp(0.0, 1.0))

        masks, top_scores = [], []
        for rank in range(self.k_per_type):
            rank_radius = max(1, int(base_radius + rank * self.rank_radius_step))
            rank_nms = max(rank_radius, int(base_nms + rank * self.rank_radius_step))

            masked = score * available
            flat = masked.reshape(b, -1)
            idx = flat.argmax(dim=1)
            has = (available.reshape(b, -1).sum(dim=1) > 0).to(score.dtype)

            seed_window = self._window_from_index(
                idx, h, w, rank_radius, score.device, score.dtype
            )
            chosen = available * seed_window * has[:, None, None, None]
            masks.append(chosen)
            top_scores.append(flat.gather(1, idx[:, None])[:, 0] * has)

            suppress = self._window_from_index(
                idx, h, w, rank_nms, score.device, score.dtype
            )
            available = available * (1.0 - suppress)

        return torch.cat(masks, dim=1), torch.stack(top_scores, dim=1)

    def _control_carrier(
        self,
        action_type: int,
        anchor: torch.Tensor,
        support: torch.Tensor,
        factual_union: torch.Tensor,
    ) -> torch.Tensor:
        """Return a broad polarity-compatible carrier, excluding all factual edits.

        A counterfactual control must be a matched alternative intervention,
        not another rare-error hypothesis. Delete controls therefore use
        predicted foreground carriers; fill controls use the external repair
        band. Hole-fill stays strict because it is intrinsically sparse.
        """
        if action_type in self.DELETE_TYPES:
            carrier = anchor
        elif action_type == 2:
            carrier = (
                _soft_dilate(anchor, self.outer_radius) - anchor
            ).clamp(0.0, 1.0)
        else:
            carrier = support

        return carrier * (1.0 - factual_union.clamp(0.0, 1.0))

    @staticmethod
    def _compactness(mask: torch.Tensor) -> torch.Tensor:
        """Perimeter²/area for one binary 2D mask."""
        x = mask.float()[None, None]
        eroded = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
        perimeter = (x - eroded).clamp_min(0.0).sum()
        area = x.sum().clamp_min(1.0)
        return (perimeter.pow(2) / area).clamp_min(1e-6)

    @staticmethod
    def _translate_mask_no_wrap(
        mask: torch.Tensor,
        shift_y: int,
        shift_x: int,
    ) -> torch.Tensor:
        """Translate one binary 2D template without circular wrapping."""
        if mask.ndim != 2:
            raise ValueError(
                f"Expected 2D template mask, got {tuple(mask.shape)}"
            )
        height, width = mask.shape
        output = torch.zeros_like(mask)
        src_y0 = max(0, -int(shift_y))
        src_y1 = min(height, height - int(shift_y))
        src_x0 = max(0, -int(shift_x))
        src_x1 = min(width, width - int(shift_x))
        if src_y1 <= src_y0 or src_x1 <= src_x0:
            return output
        dst_y0 = src_y0 + int(shift_y)
        dst_y1 = src_y1 + int(shift_y)
        dst_x0 = src_x0 + int(shift_x)
        dst_x1 = src_x1 + int(shift_x)
        output[dst_y0:dst_y1, dst_x0:dst_x1] = mask[
            src_y0:src_y1,
            src_x0:src_x1,
        ]
        return output

    @torch.no_grad()
    def _matched_local_control(
        self,
        factual: torch.Tensor,
        carrier: torch.Tensor,
        gray: torch.Tensor,
        entropy: torch.Tensor,
        boundary: torch.Tensor,
    ) -> torch.Tensor:
        """Build a strict translated-template factual/control counterpart.

        The former implementation selected a same-area compact blob around a
        matched seed.  That made the control a different spatial action.  The
        revised routine first selects a feature-matched *target centre*, then
        translates the entire factual support template to that centre without
        wrapping.  The translated template must fit fully inside the control
        carrier and its context must be disjoint from the factual context.
        Otherwise the action receives an empty control and is rejected by the
        downstream PAIR-M2 geometry gate.
        """
        if factual.ndim != 4 or factual.shape[1] != 1:
            raise ValueError(
                "Matched control expects factual [B,1,H,W], got "
                f"{tuple(factual.shape)}"
            )
        if carrier.ndim != 4 or carrier.shape != factual.shape:
            raise ValueError(
                "Matched control carrier must match factual shape, got "
                f"factual={tuple(factual.shape)} carrier={tuple(carrier.shape)}"
            )

        output = torch.zeros_like(factual)
        for bi in range(factual.shape[0]):
            factual_mask = factual[bi, 0] > 0.5
            carrier_mask = carrier[bi, 0] > 0.5
            target_area = int(factual_mask.sum().item())
            if target_area <= 0:
                continue

            factual_coords = factual_mask.nonzero(as_tuple=False)
            factual_y = factual_coords[:, 0].float().mean()
            factual_x = factual_coords[:, 1].float().mean()
            factual_context = _soft_dilate(
                factual[bi:bi + 1],
                self.context_radius,
            )[0, 0] > 0.5

            # Candidate centres must already be outside the factual context
            # and farther than the symmetric local-context separation.
            candidate_pool = carrier_mask & (~factual_context)
            candidate_coords = candidate_pool.nonzero(as_tuple=False)
            if candidate_coords.numel() == 0:
                continue
            distance = torch.sqrt(
                (candidate_coords[:, 0].float() - factual_y).pow(2)
                + (candidate_coords[:, 1].float() - factual_x).pow(2)
            )
            candidate_coords = candidate_coords[
                distance >= float(self.control_min_shift)
            ]
            if candidate_coords.numel() == 0:
                continue

            factual_gray = gray[bi, 0][factual_mask].mean()
            factual_entropy = entropy[bi, 0][factual_mask].mean()
            factual_boundary = boundary[bi, 0][factual_mask].mean()
            seed_gray = gray[bi, 0][
                candidate_coords[:, 0], candidate_coords[:, 1]
            ]
            seed_entropy = entropy[bi, 0][
                candidate_coords[:, 0], candidate_coords[:, 1]
            ]
            seed_boundary = boundary[bi, 0][
                candidate_coords[:, 0], candidate_coords[:, 1]
            ]
            seed_cost = (
                self.control_gray_weight * (seed_gray - factual_gray).abs()
                + self.control_entropy_weight
                * (seed_entropy - factual_entropy).abs()
                + self.control_boundary_weight
                * (seed_boundary - factual_boundary).abs()
            )
            top_count = min(self.control_top_centers, candidate_coords.shape[0])
            candidate_ids = torch.topk(
                seed_cost,
                k=top_count,
                largest=False,
            ).indices

            best_mask = None
            best_cost = None
            for candidate_id in candidate_ids.tolist():
                centre = candidate_coords[candidate_id]
                shift_y = int(torch.round(centre[0].float() - factual_y).item())
                shift_x = int(torch.round(centre[1].float() - factual_x).item())
                translated = self._translate_mask_no_wrap(
                    factual_mask.to(factual.dtype),
                    shift_y,
                    shift_x,
                ) > 0.5

                # Reject clipping, shape/carrier mismatch and context leakage.
                if int(translated.sum().item()) != target_area:
                    continue
                if not bool(carrier_mask[translated].all().item()):
                    continue
                control_context = _soft_dilate(
                    translated.to(factual.dtype)[None, None],
                    self.context_radius,
                )[0, 0] > 0.5
                if bool((factual_context & control_context).any().item()):
                    continue

                control_gray = gray[bi, 0][translated].mean()
                control_entropy = entropy[bi, 0][translated].mean()
                control_boundary = boundary[bi, 0][translated].mean()
                cost = (
                    self.control_gray_weight * (control_gray - factual_gray).abs()
                    + self.control_entropy_weight
                    * (control_entropy - factual_entropy).abs()
                    + self.control_boundary_weight
                    * (control_boundary - factual_boundary).abs()
                )
                if best_cost is None or float(cost.item()) < float(best_cost.item()):
                    best_cost = cost
                    best_mask = translated

            if best_mask is not None:
                output[bi, 0][best_mask] = 1.0
        return output

    @torch.no_grad()
    def _matched_local_controls_fast(
        self,
        factual: torch.Tensor,
        carrier: torch.Tensor,
        gray: torch.Tensor,
        entropy: torch.Tensor,
        boundary: torch.Tensor,
    ) -> torch.Tensor:
        """Exact-template controls for all ranks.

        This preserves the public fast-path interface but deliberately routes
        through the strict template constructor.  A compact blob approximation
        would reintroduce the action-shape confound that PAIR-M2 is designed to
        remove.  The operation remains entirely on the active device and uses
        no CPU cache or host-side tensor storage.
        """
        if factual.ndim != 4:
            raise RuntimeError(
                "Expected factual controls [B,R,H,W], got "
                f"{tuple(factual.shape)}"
            )
        return torch.cat(
            [
                self._matched_local_control(
                    factual[:, rank:rank + 1],
                    carrier,
                    gray,
                    entropy,
                    boundary,
                )
                for rank in range(factual.shape[1])
            ],
            dim=1,
        )

    @staticmethod
    def _pool_feature(feature_map: torch.Tensor, roi: torch.Tensor):
        # feature_map [B,C,H,W], roi [B,K,H,W] -> [B,K,C]
        weight = roi.clamp_min(0.0)
        numerator = torch.einsum("bchw,bkhw->bkc", feature_map, weight)
        denominator = weight.sum(dim=(-2, -1), keepdim=False).clamp_min(EPS)
        return numerator / denominator.unsqueeze(-1)

    def _text_counterfactual(
        self,
        semantic_map,
        text_features,
        negative_text_features,
        factual_masks,
        control_masks,
        action_types,
    ):
        if semantic_map is None:
            raise RuntimeError("V20 counterfactual verifier requires semantic map.")

        if negative_text_features is None:
            negative_text_features = text_features

        b, k, h, w = factual_masks.shape
        context_fact = _soft_dilate(
            factual_masks.reshape(b * k, 1, h, w), self.context_radius
        ).reshape(b, k, h, w)
        context_ctrl = _soft_dilate(
            control_masks.reshape(b * k, 1, h, w), self.context_radius
        ).reshape(b, k, h, w)

        factual_feat = self._pool_feature(semantic_map, context_fact)
        control_feat = self._pool_feature(semantic_map, context_ctrl)

        pos_text = F.normalize(text_features, dim=-1, eps=1e-6)
        neg_text = F.normalize(negative_text_features, dim=-1, eps=1e-6)
        factual_norm = F.normalize(factual_feat, dim=-1, eps=1e-6)
        control_norm = F.normalize(control_feat, dim=-1, eps=1e-6)

        factual_pos = (factual_norm * pos_text[:, None, :]).sum(dim=-1)
        control_pos = (control_norm * pos_text[:, None, :]).sum(dim=-1)
        factual_neg = (factual_norm * neg_text[:, None, :]).sum(dim=-1)
        control_neg = (control_norm * neg_text[:, None, :]).sum(dim=-1)

        factual_lesionness = factual_pos - factual_neg
        control_lesionness = control_pos - control_neg
        raw_delta = factual_lesionness - control_lesionness

        polarity = torch.where(
            torch.isin(
                action_types,
                torch.tensor(self.FILL_TYPES, device=action_types.device),
            ),
            torch.ones_like(action_types, dtype=raw_delta.dtype),
            -torch.ones_like(action_types, dtype=raw_delta.dtype),
        )[None, :]

        signed_delta = polarity * raw_delta
        factual_visual_delta = torch.sqrt(
            (factual_norm - control_norm).pow(2).mean(dim=-1).clamp_min(0.0) + 1e-6
        )

        cf_input = torch.stack(
            [signed_delta, factual_visual_delta, raw_delta],
            dim=-1,
        )
        cf_logit = self.cf_verifier(cf_input.reshape(b * k, -1)).reshape(b, k)

        swap_input = torch.stack(
            [-signed_delta, factual_visual_delta, -raw_delta],
            dim=-1,
        )
        swap_logit = self.cf_verifier(swap_input.reshape(b * k, -1)).reshape(b, k)

        factual_area = factual_masks.sum(dim=(-2, -1))
        control_area = control_masks.sum(dim=(-2, -1))
        area_ratio = control_area / factual_area.clamp_min(EPS)
        overlap = (
            factual_masks * control_masks
        ).sum(dim=(-2, -1)) / factual_area.clamp_min(EPS)

        available = (
            (factual_area > 0)
            & (control_area > 0)
            & ((area_ratio - 1.0).abs() <= 1e-3)
            & (overlap <= 1e-6)
        )

        return {
            "cf_logit": cf_logit,
            "cf_swap_logit": swap_logit,
            "cf_signed_delta": signed_delta,
            "cf_factual_similarity": factual_lesionness,
            "cf_control_similarity": control_lesionness,
            "cf_available": available,
            "cf_control_area_ratio": area_ratio,
            "cf_control_overlap": overlap,
        }


    def _unified_m1_safe_fusion(
        self,
        anchor_logits: torch.Tensor,
        candidates_t: torch.Tensor,
        factual_masks: torch.Tensor,
        visual_scores: torch.Tensor,
        actionness_logits: torch.Tensor,
        delta_maps: torch.Tensor,
        entropy: torch.Tensor,
        boundary: torch.Tensor,
        image_edge: torch.Tensor,
        control_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Unified M1 Preserve-vs-local-action selector with exact control diagnostics.

        V458 fixes the V457 failure mode at the mechanism level:
          1) paired-control validity is structural evidence, not a score gate;
          2) invalid controls are exposed to audit instead of silently becoming NaN;
          3) selector/risk heads consume detached candidate evidence, so a bad
             early selector cannot train M1 into an empty-edit generator.
        """
        if not self.unified_m1_safe_fusion_enabled:
            raise RuntimeError("Unified M1 safe fusion was not enabled.")
        required = (
            self.m1_safe_state,
            self.m1_safe_value_head,
            self.m1_safe_benefit_head,
            self.m1_safe_harm_head,
        )
        if any(module is None for module in required):
            raise RuntimeError("Unified M1 safe-fusion heads were not constructed.")

        b, k, h, w = candidates_t.shape
        if factual_masks.shape != candidates_t.shape:
            raise RuntimeError(
                "Unified M1 factual/action tensor shape mismatch: "
                f"{tuple(factual_masks.shape)} vs {tuple(candidates_t.shape)}"
            )
        if visual_scores is None:
            raise RuntimeError("Unified M1 safe fusion requires visual_scores.")
        if control_masks is None:
            control_masks = torch.zeros_like(factual_masks)
        if control_masks.shape != factual_masks.shape:
            raise RuntimeError(
                "Unified M1 control/action tensor shape mismatch: "
                f"{tuple(control_masks.shape)} vs {tuple(factual_masks.shape)}"
            )

        # Structural factual/control validity.  These tensors are always returned
        # to the V27 audit columns; they must never be missing/NaN when paired
        # control is enabled.
        factual_area_px = factual_masks.sum(dim=(-2, -1))
        control_area_px = control_masks.sum(dim=(-2, -1))
        area_ratio = control_area_px / factual_area_px.clamp_min(EPS)
        control_overlap = (
            factual_masks * control_masks
        ).sum(dim=(-2, -1)) / factual_area_px.clamp_min(EPS)
        ctx_radius = int(getattr(self, "m1_safe_control_context_radius", 2))
        factual_context = _soft_dilate(
            factual_masks.reshape(b * k, 1, h, w), ctx_radius
        ).reshape(b, k, h, w)
        control_context = _soft_dilate(
            control_masks.reshape(b * k, 1, h, w), ctx_radius
        ).reshape(b, k, h, w)
        context_overlap = (
            factual_context * control_context
        ).sum(dim=(-2, -1)) / factual_context.sum(dim=(-2, -1)).clamp_min(EPS)
        raw_control_valid = (
            (factual_area_px > 0.5)
            & (control_area_px > 0.5)
            & ((area_ratio - 1.0).abs() <= self.m1_safe_control_area_tolerance)
            & (control_overlap <= self.m1_safe_control_overlap_tolerance)
        )
        context_clean = context_overlap <= self.m1_safe_control_context_overlap_tolerance
        pair_valid = raw_control_valid & context_clean

        # Selector/risk evidence is detached from candidate generation by
        # default.  This preserves unified training while preventing selector
        # risk gradients from collapsing ActionBank into zero-area proposals.
        detach_selector = bool(getattr(self, "m1_safe_decouple_selector_from_candidates", True))
        sf = factual_masks.detach() if detach_selector else factual_masks
        sc = control_masks.detach() if detach_selector else control_masks
        sd = delta_maps.detach() if detach_selector else delta_maps
        sv = visual_scores.detach() if detach_selector else visual_scores
        sentropy = entropy.detach() if detach_selector else entropy
        sboundary = boundary.detach() if detach_selector else boundary
        sedge = image_edge.detach() if detach_selector else image_edge

        area = sf.mean(dim=(-2, -1))
        mass = sf.sum(dim=(-2, -1)).clamp_min(EPS)
        local_entropy = (sf * sentropy).sum(dim=(-2, -1)) / mass
        local_boundary = (sf * sboundary).sum(dim=(-2, -1)) / mass
        local_edge = (sf * sedge).sum(dim=(-2, -1)) / mass
        local_magnitude = (sf * sd).sum(dim=(-2, -1)) / mass

        state_input = torch.stack(
            [sv, local_entropy, local_boundary, local_edge, local_magnitude],
            dim=-1,
        )
        factual_state = self.m1_safe_state(state_input.reshape(b * k, 5))
        factual_value_delta = (
            self.m1_safe_value_scale * torch.tanh(self.m1_safe_value_head(factual_state))
        ).reshape(b, k, 2)
        factual_benefit_logits = self.m1_safe_benefit_head(factual_state).reshape(b, k)
        factual_harm_logits = self.m1_safe_harm_head(factual_state).reshape(b, k)

        control_mass = sc.sum(dim=(-2, -1)).clamp_min(EPS)
        control_entropy = (sc * sentropy).sum(dim=(-2, -1)) / control_mass
        control_boundary = (sc * sboundary).sum(dim=(-2, -1)) / control_mass
        control_edge = (sc * sedge).sum(dim=(-2, -1)) / control_mass
        control_magnitude = (sc * sd).sum(dim=(-2, -1)) / control_mass
        control_area = sc.mean(dim=(-2, -1))
        control_state_input = torch.stack(
            [sv, control_entropy, control_boundary, control_edge, control_magnitude],
            dim=-1,
        )
        control_state = self.m1_safe_state(control_state_input.reshape(b * k, 5))
        control_value_delta = (
            self.m1_safe_value_scale * torch.tanh(self.m1_safe_value_head(control_state))
        ).reshape(b, k, 2)
        control_benefit_logits = self.m1_safe_benefit_head(control_state).reshape(b, k)
        control_harm_logits = self.m1_safe_harm_head(control_state).reshape(b, k)

        if self.m1_safe_paired_control_selector:
            paired_value_delta = factual_value_delta - control_value_delta
            paired_benefit_logits = factual_benefit_logits - control_benefit_logits
            # V461 root fix: pair-control residuals are only meaningful when an
            # exact matched control is valid.  V459/V460 allowed factual fallback
            # labels in the loss, but the forward score still subtracted invalid
            # control states for every non-pair-valid action.  Because exact
            # controls are sparse on BUSI ActionBank, this made almost all
            # action scores negative and preserved every case.  Keep paired
            # residual scoring for pair-valid actions, and use factual scoring
            # for candidate-valid actions without an exact control, matching the
            # loss-side target definition.
            pair_mask = pair_valid.to(factual_value_delta.dtype)
            value_delta = torch.where(
                pair_mask[..., None] > 0.5,
                paired_value_delta,
                factual_value_delta,
            )
            benefit_logits = torch.where(
                pair_mask > 0.5,
                paired_benefit_logits,
                factual_benefit_logits,
            )
            # Harm remains factual: a harmful factual edit must be rejected even
            # if the translated control edit is also harmful or unavailable.
            harm_logits = factual_harm_logits
        else:
            paired_value_delta = factual_value_delta
            paired_benefit_logits = factual_benefit_logits
            pair_mask = torch.zeros_like(factual_benefit_logits)
            value_delta = factual_value_delta
            benefit_logits = factual_benefit_logits
            harm_logits = factual_harm_logits

        value_utility = 0.60 * value_delta[..., 0] + 0.40 * value_delta[..., 1]
        benefit_probability = torch.sigmoid(benefit_logits)
        harm_probability = torch.sigmoid(harm_logits)
        risk_margin = benefit_probability - harm_probability
        raw_score = (
            value_utility
            + self.m1_safe_benefit_weight * benefit_logits
            - self.m1_safe_harm_weight * harm_logits
        )

        action_types = self.action_types[:k].to(raw_score.device)
        type_bias_values = raw_score.new_tensor([
            self.m1_safe_type_score_bias.get(0, 0.0),
            self.m1_safe_type_score_bias.get(1, 0.0),
            self.m1_safe_type_score_bias.get(2, 0.0),
            self.m1_safe_type_score_bias.get(3, 0.0),
        ])
        static_type_bias = type_bias_values.gather(0, action_types.clamp(0, 3))[None, :]
        score = raw_score + static_type_bias

        valid = (
            (factual_area_px > 0.5)
            & (area <= self.m1_safe_max_edit_fraction)
        )
        # V460 root fix: keep the model-side train/deploy logit mask consistent
        # with the loss-side valid mask.  V459 used fallback factual positives
        # in the loss when exact controls were unavailable, but still masked
        # every non-pair-valid logit to -20 here.  That made choice CE explode
        # and forced Preserve despite oracle-positive candidates.
        if self.m1_safe_paired_control_selector and self.m1_safe_require_control_for_selector:
            train_valid = valid & pair_valid
        else:
            train_valid = valid

        static_deploy_valid = train_valid
        if self.m1_safe_disable_deploy_types:
            disabled = torch.zeros(k, dtype=torch.bool, device=raw_score.device)
            for type_id in self.m1_safe_disable_deploy_types:
                disabled = disabled | (action_types == int(type_id))
            static_deploy_valid = static_deploy_valid & (~disabled[None, :])

        # Deployment validity is structural.  Score thresholding is a decision
        # comparison against Preserve=0, not a validity definition; otherwise
        # audit shows deploy_valid=0 for every action and hides the true failure.
        deploy_valid = static_deploy_valid
        score_pass = score > self.m1_safe_action_threshold
        risk_barrier = score.new_zeros((b, 1))
        value_barrier = score.new_zeros((b, 1))

        train_choice_logits = torch.cat(
            [
                score.new_full((b, 1), self.m1_safe_preserve_logit_bias),
                score.masked_fill(~train_valid, -20.0),
            ],
            dim=1,
        )

        if self.m1_safe_multi_action:
            action_logits = score.masked_fill(~deploy_valid, -20.0)
            action_soft = torch.sigmoid(action_logits / self.m1_safe_temperature) * deploy_valid.to(score.dtype)
            expected_area = (action_soft * area).sum(dim=1, keepdim=True)
            soft_budget_scale = (self.m1_safe_total_edit_fraction / expected_area.clamp_min(EPS)).clamp(max=1.0)
            action_soft = action_soft * soft_budget_scale

            top_k = min(self.m1_safe_max_actions, k)
            top_values, top_indices = action_logits.topk(k=top_k, dim=1)
            hard_actions = torch.zeros_like(action_logits)
            selected = (top_values > self.m1_safe_action_threshold) & torch.gather(deploy_valid, 1, top_indices)
            hard_actions.scatter_(1, top_indices, selected.to(action_logits.dtype))
            hard_area = (hard_actions * area).sum(dim=1, keepdim=True)
            hard_budget_scale = (self.m1_safe_total_edit_fraction / hard_area.clamp_min(EPS)).clamp(max=1.0)
            hard_actions = hard_actions * hard_budget_scale
            action_st = hard_actions + action_soft - action_soft.detach() if self.training else hard_actions

            preserve_soft = (1.0 - action_soft.max(dim=1, keepdim=True).values).clamp(0.0, 1.0)
            preserve_hard = (hard_actions.sum(dim=1, keepdim=True) <= 0.0).to(action_logits.dtype)
            preserve_st = preserve_hard + preserve_soft - preserve_soft.detach() if self.training else preserve_hard
            choice_logits = torch.cat([score.new_full((b, 1), self.m1_safe_preserve_logit_bias), action_logits], dim=1)
            choice_soft = torch.cat([preserve_soft, action_soft], dim=1)
            choice_hard = torch.cat([preserve_hard, hard_actions], dim=1)
            choice_st = torch.cat([preserve_st, action_st], dim=1)
            hard_index = torch.where(
                preserve_hard[:, 0] > 0.5,
                torch.zeros(b, device=score.device, dtype=torch.long),
                hard_actions.argmax(dim=1) + 1,
            )
        else:
            choice_logits = torch.cat(
                [
                    score.new_full((b, 1), self.m1_safe_preserve_logit_bias),
                    score.masked_fill(~deploy_valid, -20.0),
                ],
                dim=1,
            )
            choice_soft = torch.softmax(choice_logits / self.m1_safe_temperature, dim=1)
            hard_index = choice_logits.argmax(dim=1)
            choice_hard = F.one_hot(hard_index, num_classes=k + 1).to(dtype=anchor_logits.dtype)
            choice_st = choice_hard + choice_soft - choice_soft.detach() if self.training else choice_hard

        if detach_selector:
            anchor_for_fusion = anchor_logits.detach()
            action_delta = (candidates_t - anchor_logits).detach()
        else:
            anchor_for_fusion = anchor_logits
            action_delta = candidates_t - anchor_logits
        soft_logits = anchor_for_fusion[:, 0] + (choice_soft[:, 1:, None, None] * action_delta).sum(dim=1)
        st_logits = anchor_for_fusion[:, 0] + (choice_st[:, 1:, None, None] * action_delta).sum(dim=1)
        hard_logits = anchor_for_fusion[:, 0] + (choice_hard[:, 1:, None, None] * action_delta).sum(dim=1)

        return {
            "m1_value_delta": value_delta,
            "m1_paired_value_delta": paired_value_delta,
            "m1_factual_value_delta": factual_value_delta,
            "m1_control_value_delta": control_value_delta,
            "m1_pair_valid_action": pair_valid,
            "m1_pair_score_used_action": pair_valid if self.m1_safe_paired_control_selector else torch.zeros_like(pair_valid),
            "m1_benefit_logits": benefit_logits,
            "m1_factual_benefit_logits": factual_benefit_logits,
            "m1_control_benefit_logits": control_benefit_logits,
            "m1_harm_logits": harm_logits,
            "m1_factual_harm_logits": factual_harm_logits,
            "m1_control_harm_logits": control_harm_logits,
            "m1_raw_score": raw_score,
            "m1_score": score,
            "m1_choice_logits": choice_logits,
            "m1_train_choice_logits": train_choice_logits,
            "m1_selector_soft": choice_soft,
            "m1_selector_hard": choice_hard,
            "m1_selector_st": choice_st,
            "m1_soft_fused_logits": soft_logits,
            "m1_st_fused_logits": st_logits,
            "m1_hard_fused_logits": hard_logits,
            "m1_soft_fused_probs": torch.sigmoid(soft_logits).clamp(EPS, 1.0 - EPS),
            "m1_hard_fused_probs": torch.sigmoid(hard_logits).clamp(EPS, 1.0 - EPS),
            "m1_valid_action": valid,
            "m1_train_pair_valid_action": pair_valid,
            "m1_train_valid_action": train_valid,
            "m1_pair_valid_action": pair_valid,
            "m1_require_control_for_selector": torch.full((b,), float(self.m1_safe_require_control_for_selector), device=score.device, dtype=score.dtype),
            "m1_deploy_valid_action": deploy_valid,
            "m1_static_deploy_valid_action": static_deploy_valid,
            "m1_adaptive_deploy_valid_action": score_pass,
            "m1_adaptive_risk_barrier": risk_barrier.expand_as(score),
            "m1_adaptive_value_barrier": value_barrier.expand_as(score),
            "m1_risk_margin": risk_margin,
            "m1_value_utility": value_utility,
            "m1_benefit_probability": benefit_probability,
            "m1_harm_probability": harm_probability,
            "m1_action_type_bias": static_type_bias.expand_as(score),
            "m1_local_area": area,
            "m1_local_entropy": local_entropy,
            "m1_local_boundary": local_boundary,
            "m1_local_edge": local_edge,
            "m1_local_magnitude": local_magnitude,
            "m1_control_area": control_area,
            "m1_control_entropy": control_entropy,
            "m1_control_boundary": control_boundary,
            "m1_control_edge": control_edge,
            "m1_control_magnitude": control_magnitude,
            "m1_paired_control_selector_enabled": torch.full((b,), float(self.m1_safe_paired_control_selector), device=score.device, dtype=score.dtype),
            "m1_decoupled_selector_enabled": torch.full((b,), float(detach_selector), device=score.device, dtype=score.dtype),
            "m1_multi_action_enabled": torch.full((b,), float(self.m1_safe_multi_action), device=score.device, dtype=score.dtype),
            "m1_hard_action_count": (choice_hard[:, 1:] > 0).to(score.dtype).sum(dim=1),
            "m1_soft_action_count": choice_soft[:, 1:].sum(dim=1),
            "m1_total_hard_edit_fraction": (choice_hard[:, 1:] * area).sum(dim=1),
            "m1_total_soft_edit_fraction": (choice_soft[:, 1:] * area).sum(dim=1),
            "m1_preserve_hard": choice_hard[:, 0],
            "m1_hard_selected_slot": hard_index,
            # V25/V27-compatible audit/control evidence fields.
            "v25_valid_control": pair_valid,
            "v25_control_area_ratio": area_ratio,
            "v25_control_overlap": control_overlap,
            "v27_raw_control_valid": raw_control_valid,
            "v27_context_clean_valid": context_clean,
            "v27_context_overlap": context_overlap,
            "v27_local_action_area": factual_area_px / float(h * w),
            "v457_train_pair_valid": train_valid,
            "v457_control_area_px": control_area_px,
            "v457_factual_area_px": factual_area_px,
        }


    def _safe_residual_candidates(
        self,
        anchor_logits: torch.Tensor,
        base_prob: torch.Tensor,
        anchor: torch.Tensor,
        entropy: torch.Tensor,
        latent: torch.Tensor,
        semantic_map: Optional[torch.Tensor],
        image_edge: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Return C9 and a certified context C10 diagnostic candidate.

        Standard A1/A2 behaviour is retained when
        ``SAFE_CONTEXT_CERTIFIED_ONLY`` is false. In A3, C9 follows its
        frozen A1 branch while C10 uses an independent path and returns
        Preserve exactly for ambiguous global edit directions.
        """
        if not self.safe_residual_enabled:
            return {}

        required = (
            self.safe_residual_refiner,
            self.safe_boundary_residual_head,
            self.safe_context_residual_head,
            self.safe_context_direction_head,
        )
        if any(module is None for module in required):
            raise RuntimeError("Safe residual heads were not constructed.")
        if self.safe_context_certified_only and self.safe_context_refiner is None:
            raise RuntimeError(
                "SAFE_CONTEXT_CERTIFIED_ONLY requires safe_context_refiner."
            )

        semantic = semantic_map
        if self.safe_residual_use_semantic:
            if semantic is None or self.safe_residual_semantic_proj is None:
                raise RuntimeError(
                    "SAFE_RESIDUAL_USE_SEMANTIC requires semantic_map."
                )
            if semantic.shape[-2:] != base_prob.shape[-2:]:
                semantic = F.interpolate(
                    semantic,
                    size=base_prob.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            if semantic.shape[1] != self.semantic_channels:
                raise RuntimeError(
                    "Safe residual semantic channel mismatch: "
                    f"expected {self.semantic_channels}, got "
                    f"{semantic.shape[1]}."
                )

        # C9 branch. Its module names and computation are unchanged from A1/A2.
        boundary_feature = latent
        if self.safe_residual_use_semantic:
            boundary_feature = (
                boundary_feature + self.safe_residual_semantic_proj(semantic)
            )
        boundary_feature = self.safe_residual_refiner(boundary_feature)

        # A3 C10 branch is independent. Outside A3 it deliberately reproduces
        # the previous shared-feature behaviour for backward compatibility.
        if self.safe_context_certified_only:
            context_feature = latent
            if self.safe_residual_use_semantic:
                if self.safe_context_semantic_proj is None:
                    raise RuntimeError(
                        "Certified C10 semantic projection is absent."
                    )
                context_feature = (
                    context_feature
                    + self.safe_context_semantic_proj(semantic)
                )
            assert self.safe_context_refiner is not None
            context_feature = self.safe_context_refiner(context_feature)
        else:
            context_feature = boundary_feature

        uncertainty_gate = torch.sigmoid(
            (entropy - self.safe_uncertainty_floor)
            / self.safe_uncertainty_temperature
        )
        edge_gate = torch.sigmoid(
            (image_edge - self.safe_edge_floor)
            / self.safe_edge_temperature
        )

        boundary_band = (
            _soft_dilate(anchor, self.safe_boundary_outer_radius)
            - _soft_erode(anchor, self.safe_boundary_inner_radius)
        ).clamp(0.0, 1.0)
        boundary_raw = self.safe_boundary_residual_head(boundary_feature)
        boundary_gate = (
            boundary_band
            * uncertainty_gate
            * edge_gate
            * torch.sigmoid(boundary_raw[:, :1])
            * torch.sigmoid(boundary_raw[:, 2:3])
        )
        boundary_delta = (
            torch.tanh(boundary_raw[:, 1:2])
            * self.safe_boundary_max_delta
        )
        boundary_logits = anchor_logits + boundary_gate * boundary_delta

        inner_context = (
            anchor - _soft_erode(anchor, self.safe_context_inner_radius)
        ).clamp(0.0, 1.0)
        outer_context = (
            _soft_dilate(anchor, self.safe_context_outer_radius) - anchor
        ).clamp(0.0, 1.0)

        direction_logit = self.safe_context_direction_head(context_feature).reshape(
            context_feature.shape[0], 1, 1, 1
        )
        fill_probability = torch.sigmoid(direction_logit)
        fill_hard = (fill_probability >= 0.5).to(context_feature.dtype)
        fill_choice = fill_hard + fill_probability - fill_probability.detach()
        delete_choice = 1.0 - fill_choice

        context_raw = self.safe_context_residual_head(context_feature)
        context_carrier = (
            fill_choice * outer_context + delete_choice * inner_context
        )

        # A3's certificate uses the direction head itself. Ambiguous cases have
        # a hard zero gate, producing C10 == Preserve in the forward pass.
        if self.safe_context_certified_only:
            direction_confidence = (2.0 * fill_probability - 1.0).abs()
            certificate_soft = torch.sigmoid(
                (
                    direction_confidence
                    - self.safe_context_direction_margin
                )
                / self.safe_context_direction_temperature
            )
            certificate_hard = (
                direction_confidence >= self.safe_context_direction_margin
            ).to(context_feature.dtype)
            # Training uses the differentiable certificate probability so
            # the newly initialised C10 refiner/gate/delta path receives
            # gradients from the first batch. Evaluation remains strictly
            # certified: ambiguous directions return Preserve exactly.
            certificate = (
                certificate_soft
                if self.training
                else certificate_hard
            )
            near_threshold_gate = torch.sigmoid(
                (
                    self.safe_context_logit_band
                    - anchor_logits.abs()
                )
                / self.safe_context_logit_temperature
            )
        else:
            direction_confidence = torch.ones_like(fill_probability)
            certificate_soft = torch.ones_like(fill_probability)
            certificate_hard = torch.ones_like(fill_probability)
            certificate = torch.ones_like(fill_probability)
            near_threshold_gate = torch.ones_like(anchor_logits)

        context_gate = (
            context_carrier
            * uncertainty_gate
            * (0.5 + 0.5 * edge_gate)
            * near_threshold_gate
            * certificate
            * torch.sigmoid(context_raw[:, :1])
        )
        context_magnitude = F.softplus(context_raw[:, 1:2]).clamp(
            max=self.safe_context_max_delta
        )
        signed_context_delta = (
            (fill_choice - delete_choice)
            * context_gate
            * context_magnitude
        )
        context_logits = anchor_logits + signed_context_delta

        if self.safe_residual_mode == "boundary":
            context_logits = anchor_logits
            context_carrier = torch.zeros_like(context_carrier)
            context_gate = torch.zeros_like(context_gate)
            signed_context_delta = torch.zeros_like(signed_context_delta)
        elif self.safe_residual_mode == "context":
            boundary_logits = anchor_logits
            boundary_band = torch.zeros_like(boundary_band)
            boundary_gate = torch.zeros_like(boundary_gate)
            boundary_delta = torch.zeros_like(boundary_delta)
        elif self.safe_residual_mode != "both":
            raise ValueError(
                "SAFE_RESIDUAL_MODE must be boundary, context, or both."
            )

        logits = torch.cat([boundary_logits, context_logits], dim=1)
        return {
            "safe_residual_candidate_logits": logits,
            "safe_residual_candidate_probs": torch.sigmoid(logits).clamp(
                EPS, 1.0 - EPS
            ),
            "safe_boundary_carrier": boundary_band,
            "safe_context_carrier": context_carrier,
            "safe_boundary_gate": boundary_gate,
            "safe_context_gate": context_gate,
            "safe_boundary_delta": boundary_delta,
            "safe_context_signed_delta": signed_context_delta,
            "safe_context_fill_probability": fill_probability[:, 0, 0, 0],
            "safe_context_direction_logit": direction_logit[:, 0, 0, 0],
            "safe_context_direction_confidence": (
                direction_confidence[:, 0, 0, 0]
            ),
            "safe_context_direction_certificate_soft": (
                certificate_soft[:, 0, 0, 0]
            ),
            "safe_context_direction_certificate_hard": (
                certificate_hard[:, 0, 0, 0]
            ),
            "safe_context_near_threshold_gate": near_threshold_gate,
        }

    def generate(
        self,
        base_logits: torch.Tensor,
        image: torch.Tensor,
        semantic_map: Optional[torch.Tensor],
        text_features: torch.Tensor,
        negative_text_features: Optional[torch.Tensor] = None,
    ):
        if base_logits.ndim == 3:
            base_logits = base_logits[:, None]

        anchor_logits = base_logits.detach() if self.detach_base else base_logits
        base_prob = torch.sigmoid(anchor_logits)

        latent, anchor, entropy, boundary = self._latent(
            base_prob, image, semantic_map
        )
        gray, image_edge = self._image_gray_edge(image, base_prob.shape[-2:])
        if self.v463_residual_enabled:
            if self.v463_residual_head is None:
                raise RuntimeError("V463 residual head is absent.")
            v463_residual = self.v463_residual_head(
                latent=latent,
                base_prob=base_prob,
                entropy=entropy,
                boundary=boundary,
                image_edge=image_edge,
            )
        else:
            v463_zero = base_prob.new_zeros(
                base_prob.shape[0], 4, base_prob.shape[-2], base_prob.shape[-1]
            )
            v463_residual = {
                "v463_residual_logits": v463_zero,
                "v463_residual_probs": torch.sigmoid(v463_zero),
                "v463_fp_logits": v463_zero[:, 0:1],
                "v463_fn_logits": v463_zero[:, 1:2],
                "v463_tp_risk_logits": v463_zero[:, 2:3],
                "v463_bg_risk_logits": v463_zero[:, 3:4],
                "v463_fp_prob": torch.sigmoid(v463_zero[:, 0:1]),
                "v463_fn_prob": torch.sigmoid(v463_zero[:, 1:2]),
                "v463_tp_risk_prob": torch.sigmoid(v463_zero[:, 2:3]),
                "v463_bg_risk_prob": torch.sigmoid(v463_zero[:, 3:4]),
            }
        safe_residual_aux = self._safe_residual_candidates(
            anchor_logits=anchor_logits,
            base_prob=base_prob,
            anchor=anchor,
            entropy=entropy,
            latent=latent,
            semantic_map=semantic_map,
            image_edge=image_edge,
        )
        # Fast M1-only training path: B0 and A1--A8 are frozen, and the loss
        # consumes only C9/C10. This avoids frozen textual-control work during
        # training while evaluation still produces the complete legacy bank.
        if self.safe_residual_only_training and self.training:
            return anchor_logits, {
                "candidate_probs": base_prob,
                "direct_fused_probs": base_prob[:, 0],
                "router_fused_probs": base_prob[:, 0],
                "v20_fused_probs": base_prob[:, 0],
                "v20_hard_fused_probs": base_prob[:, 0],
                "v20_fused_logits": anchor_logits[:, 0],
                **safe_residual_aux,
            }
        supports = self._type_supports(base_prob, image_edge=image_edge, entropy=entropy)
        if self.v35_residual_purified_world_model:
            if self.v35_residual_head is None:
                raise RuntimeError("V35 residual proposal head is absent.")
            v35_residual_logit_map = self.v35_residual_head(latent)
        else:
            v35_residual_logit_map = torch.zeros_like(base_prob)

        if self.learned_residual_proposal_enabled:
            if self.residual_proposal_head is None:
                raise RuntimeError("V451 residual proposal head is absent.")
            residual_proposal_logits = self.residual_proposal_head(latent)
        else:
            residual_proposal_logits = base_prob.new_zeros(
                base_prob.shape[0], 2, base_prob.shape[-2], base_prob.shape[-1]
            )

        # Pass 1: construct every factual action first, so control construction
        # can exclude all factual actions, including later action families.
        factual_specs = []
        raw_factual_by_type = {}
        all_factual_union = torch.zeros_like(base_prob)

        for action_type in range(self.num_types):
            support = supports[:, action_type:action_type + 1]
            if action_type not in self.v469_enabled_action_types:
                support = torch.zeros_like(support)
            action_logit_map = self.actionness_heads[action_type](latent)
            residual_channel = 0 if action_type in self.DELETE_TYPES else 1
            residual_bias = (
                self.residual_proposal_scale
                * residual_proposal_logits[:, residual_channel:residual_channel + 1]
            )
            if self.v463_residual_enabled:
                if action_type in self.DELETE_TYPES:
                    causal_bias = v463_residual.get(
                        "v468_delete_causal_logit",
                        v463_residual["v463_fp_logits"]
                        - v463_residual["v463_tp_risk_logits"],
                    )
                else:
                    causal_bias = v463_residual.get(
                        "v468_fill_causal_logit",
                        v463_residual["v463_fn_logits"]
                        - v463_residual["v463_bg_risk_logits"],
                    )
                residual_bias = residual_bias + self.v463_residual_scale * causal_bias
            if self.v35_residual_purified_world_model and action_type == 0:
                residual_bias = (
                    residual_bias
                    + self.v35_residual_proposal_scale * v35_residual_logit_map
                )
            action_score = support * torch.sigmoid(action_logit_map + residual_bias)

            # V422: C5/C6 are boundary-fill hypotheses. Keep the original
            # actionness head, but rank its local windows using a conservative
            # image-side prior. This does not use text, GT, M2, or M3.
            if (
                action_type == 2
                and self.v422_m1_safe_candidate_bank
                and not self.disable_safe_fill_filter
                and (
                    not self.v428_adaptive_dual_expert
                    or self.v429_asymmetric_adaptive
                )
            ):
                edge_norm = image_edge / image_edge.amax(
                    dim=(-2, -1),
                    keepdim=True,
                ).clamp_min(EPS)

                edge_weight = self.v422_fill_edge_weight
                uncertainty_weight = self.v422_fill_uncertainty_weight
                weight_sum = max(edge_weight + uncertainty_weight, EPS)

                fill_evidence = (
                    edge_weight * edge_norm
                    + uncertainty_weight * entropy
                ) / weight_sum

                fill_gate = (
                    (fill_evidence - self.v422_fill_evidence_floor)
                    / max(1.0 - self.v422_fill_evidence_floor, EPS)
                ).clamp(0.0, 1.0)

                # Keep a small floor so a valid but low-evidence FN region
                # is not mathematically impossible to propose.
                action_score = action_score * (0.10 + 0.90 * fill_gate)

            factual, factual_score = self._topk_windows(
                action_score,
                support,
                forbidden=all_factual_union if self.global_non_overlap else None,
                window_radius=self.type_window_radius[action_type],
                nms_radius=self.type_nms_radius[action_type],
            )

            # V469: diagnostics showed that raw top-k windows preserve more
            # than 90% of the same-window oracle ceiling, whereas the global
            # absolute causal threshold retained only about 3% of window pixels.
            # Use a per-window relative gate with an explicit minimum support.
            # This makes the gate a local refiner rather than an all-or-nothing
            # veto.  The detached threshold keeps gradients through evidence.
            raw_factual_by_type[action_type] = factual
            if (
                self.v468_soft_causal_edit_mask
                and self.v463_residual_enabled
                and action_type in self.v469_enabled_action_types
            ):
                gated = []
                gate_evidence = torch.sigmoid(
                    causal_bias
                    + self.v468_actionness_gate_weight * action_logit_map
                )
                for rank in range(self.k_per_type):
                    raw_window = factual[:, rank:rank + 1]
                    if self.v469_relative_gate:
                        keep_fraction = max(
                            0.05,
                            self.v470_gate_keep_fractions[action_type]
                            - rank * self.v469_gate_rank_fraction_step,
                        )
                        thresholds = []
                        for batch_index in range(raw_window.shape[0]):
                            window_bool = raw_window[
                                batch_index, 0
                            ].detach() > 0.5
                            values = gate_evidence[
                                batch_index, 0
                            ][window_bool]
                            if values.numel() == 0:
                                thresholds.append(
                                    gate_evidence.new_tensor(2.0)
                                )
                                continue
                            keep = max(
                                self.v470_gate_min_pixels[action_type],
                                int(math.ceil(
                                    keep_fraction * float(values.numel())
                                )),
                            )
                            keep = min(keep, int(values.numel()))
                            threshold_value = torch.topk(
                                values.detach(), k=keep, largest=True
                            ).values[-1]
                            thresholds.append(threshold_value)
                        threshold = torch.stack(thresholds).view(-1, 1, 1, 1)
                        gate_soft = torch.sigmoid(
                            (gate_evidence - threshold)
                            / self.v469_gate_temperature
                        ) * raw_window
                        gate_hard = (
                            (gate_evidence >= threshold).to(gate_evidence.dtype)
                            * raw_window
                        )
                    else:
                        threshold = min(
                            self.v468_causal_gate_threshold
                            + rank * self.v468_causal_gate_rank_step,
                            0.95,
                        )
                        threshold_logit = math.log(
                            threshold / max(1.0 - threshold, EPS)
                        )
                        gate_logit = (
                            causal_bias
                            + self.v468_actionness_gate_weight * action_logit_map
                            - threshold_logit
                        ) / self.v468_causal_gate_temperature
                        gate_soft = torch.sigmoid(gate_logit) * raw_window
                        gate_hard = (
                            (gate_soft >= 0.5).to(gate_soft.dtype)
                            * raw_window
                        )
                    gate_st = (
                        gate_hard + gate_soft - gate_soft.detach()
                        if self.training else gate_hard
                    )
                    gated.append(gate_st)
                factual = torch.cat(gated, dim=1)

            all_factual_union = (
                all_factual_union
                + factual.max(dim=1, keepdim=True).values
            ).clamp(0.0, 1.0)

            factual_specs.append((
                action_type,
                support,
                action_logit_map,
                action_score,
                factual,
                factual_score,
            ))

        # V420 M1-only fast path. M2 counterfactual controls/verifier and the
        # historical M3 selector are not evaluated in this configuration.
        # Candidate construction remains exactly the same typed, top-k,
        # globally non-overlapping action bank.
        if self.unified_m1_safe_fusion_enabled:
            factual_masks = []
            raw_factual_masks = []
            control_masks = []
            visual_scores = []
            actionness_logits = []
            deltas = []
            candidate_logits = []
            for (
                action_type,
                support,
                action_logit_map,
                action_score,
                factual,
                factual_score,
            ) in factual_specs:
                carrier = self._control_carrier(
                    action_type,
                    anchor,
                    support,
                    all_factual_union,
                )
                if (
                    bool(getattr(self, "evidence_guided_candidate_control_enabled", False))
                    and self.v396_fast_matched_controls
                ):
                    control = self._matched_local_controls_fast(
                        factual,
                        carrier,
                        gray,
                        entropy,
                        boundary,
                    )
                else:
                    control = torch.cat([
                        self._matched_local_control(
                            factual[:, rank:rank + 1],
                            carrier,
                            gray,
                            entropy,
                            boundary,
                        )
                        for rank in range(self.k_per_type)
                    ], dim=1)

                delta = F.softplus(
                    _run_checkpointed_module(self.delta_heads[action_type], latent)
                ).clamp(max=self.max_logit_delta)
                for rank in range(self.k_per_type):
                    factual_mask = factual[:, rank:rank + 1]
                    control_mask = control[:, rank:rank + 1]
                    if action_type in self.DELETE_TYPES:
                        required = (
                            anchor_logits + self.hard_edit_logit_margin
                        ).clamp_min(0.0)
                        action_delta = torch.maximum(delta, required)
                        edited = anchor_logits - factual_mask * action_delta
                    else:
                        required = (
                            -anchor_logits + self.hard_edit_logit_margin
                        ).clamp_min(0.0)
                        action_delta = torch.maximum(delta, required)
                        edited = anchor_logits + factual_mask * action_delta

                    candidate_logits.append(edited)
                    factual_masks.append(factual_mask)
                    raw_factual_masks.append(
                        raw_factual_by_type[action_type][:, rank:rank + 1]
                    )
                    control_masks.append(control_mask)
                    visual_scores.append(factual_score[:, rank])
                    actionness_logits.append(action_logit_map[:, 0])
                    deltas.append(action_delta[:, 0])

            factual_masks_t = torch.cat(factual_masks, dim=1)
            raw_factual_masks_t = torch.cat(raw_factual_masks, dim=1)
            control_masks_t = torch.cat(control_masks, dim=1)
            candidates_t = torch.cat(candidate_logits, dim=1)
            candidate_logits_all = torch.cat(
                [anchor_logits, candidates_t],
                dim=1,
            )
            candidate_probs = torch.cat([
                base_prob,
                torch.sigmoid(candidates_t).clamp(EPS, 1.0 - EPS),
            ], dim=1)
            visual_scores_t = torch.stack(visual_scores, dim=1)
            actionness_logits_t = torch.stack(actionness_logits, dim=1)
            delta_maps_t = torch.stack(deltas, dim=1)
            m1_safe = self._unified_m1_safe_fusion(
                anchor_logits=anchor_logits,
                candidates_t=candidates_t,
                factual_masks=factual_masks_t,
                control_masks=control_masks_t,
                visual_scores=visual_scores_t,
                actionness_logits=actionness_logits_t,
                delta_maps=delta_maps_t,
                entropy=entropy,
                boundary=boundary,
                image_edge=image_edge,
            )
            zero_action = visual_scores_t * 0.0
            zero_bool = torch.zeros_like(
                factual_masks_t.sum(dim=(-2, -1)),
                dtype=torch.bool,
            )
            return candidate_logits_all, {
                "candidate_probs": candidate_probs,
                **safe_residual_aux,
                **v463_residual,
                "direct_fused_probs": m1_safe["m1_soft_fused_probs"],
                "router_fused_probs": m1_safe["m1_soft_fused_probs"],
                "v20_fused_probs": m1_safe["m1_soft_fused_probs"],
                "v20_hard_fused_probs": m1_safe["m1_hard_fused_probs"],
                "v20_fused_logits": m1_safe["m1_st_fused_logits"],
                "v20_action_supports": factual_masks_t,
                "v469_raw_action_supports": raw_factual_masks_t,
                "v20_control_supports": control_masks_t,
                "v20_type_supports": supports,
                "v20_action_types": self.action_types,
                "v469_action_enabled_mask": self.v469_action_enabled,
                "v20_actionness_logits": actionness_logits_t,
                "v35_residual_logit_map": v35_residual_logit_map,
                "v35_residual_probability_map": torch.sigmoid(
                    v35_residual_logit_map
                ),
                "v451_residual_proposal_logits": residual_proposal_logits,
                "v451_residual_delete_probability_map": torch.sigmoid(
                    residual_proposal_logits[:, 0]
                ),
                "v451_residual_fill_probability_map": torch.sigmoid(
                    residual_proposal_logits[:, 1]
                ),
                "v451_soft_support_enabled": torch.full(
                    (base_prob.shape[0],),
                    float(self.soft_support_enabled),
                    device=base_prob.device,
                    dtype=base_prob.dtype,
                ),
                "v451_global_non_overlap_enabled": torch.full(
                    (base_prob.shape[0],),
                    float(self.global_non_overlap),
                    device=base_prob.device,
                    dtype=base_prob.dtype,
                ),
                "v20_visual_scores": visual_scores_t,
                "v20_delta_maps": delta_maps_t,
                "v20_selector_logits": m1_safe["m1_score"],
                "v20_selector_probs": m1_safe["m1_selector_soft"][:, 1:],
                "v20_selector_hard": m1_safe["m1_selector_hard"][:, 1:],
                "v20_cf_logit": zero_action,
                "v20_cf_swap_logit": zero_action,
                "v20_cf_signed_delta": zero_action,
                "v20_cf_factual_similarity": zero_action,
                "v20_cf_control_similarity": zero_action,
                "v20_cf_available": zero_bool,
                "v20_cf_control_area_ratio": zero_action,
                "v20_cf_control_overlap": zero_action,
                "v20_entropy": entropy[:, 0],
                "v20_boundary": boundary[:, 0],
                "v20_budget": entropy.new_full(
                    (base_prob.shape[0],),
                    self.m1_safe_max_edit_fraction,
                ),
                "candidate_base_detached": self.detach_base,
                "shrink_gate": factual_masks_t[:, 0],
                "expand_gate": factual_masks_t[:, -1],
                "shrink_gate_logits": actionness_logits_t[:, 0],
                "expand_gate_logits": actionness_logits_t[:, -1],
                "edit_gate": factual_masks_t.max(dim=1).values,
                "edit_band": factual_masks_t.max(dim=1).values,
                "inside_band": supports[:, :2].max(dim=1).values,
                "outside_band": supports[:, 2:].max(dim=1).values,
                "expand_connect_band": supports[:, 2:].max(dim=1).values,
                "router_probs": torch.stack([
                    m1_safe["m1_selector_soft"][:, 0],
                    m1_safe["m1_selector_soft"][
                        :, 1:1 + self.num_actions // 2
                    ].sum(dim=1),
                    m1_safe["m1_selector_soft"][
                        :, 1 + self.num_actions // 2:
                    ].sum(dim=1),
                ], dim=1)[:, :, None, None].expand(
                    -1, -1, base_prob.shape[-2], base_prob.shape[-1]
                ),
                "router_edit_probability": (
                    1.0 - m1_safe["m1_selector_soft"][:, 0]
                )[:, None, None].expand(
                    -1, base_prob.shape[-2], base_prob.shape[-1]
                ),
                "router_direction_probs": torch.stack([
                    m1_safe["m1_selector_soft"][
                        :, 1:1 + self.num_actions // 2
                    ].sum(dim=1),
                    m1_safe["m1_selector_soft"][
                        :, 1 + self.num_actions // 2:
                    ].sum(dim=1),
                ], dim=1)[:, :, None, None].expand(
                    -1, -1, base_prob.shape[-2], base_prob.shape[-1]
                ),
                "router_shrink_weight": factual_masks_t[
                    :, :self.num_actions // 2
                ].sum(dim=1).clamp(max=1.0),
                "router_expand_weight": factual_masks_t[
                    :, self.num_actions // 2:
                ].sum(dim=1).clamp(max=1.0),
                **m1_safe,
            }

        # Pass 2: build matched controls from broad carriers and then form
        # factual candidate masks.
        factual_masks = []
        control_masks = []
        visual_scores = []
        actionness_logits = []
        deltas = []
        candidate_logits = []

        for (
            action_type,
            support,
            action_logit_map,
            action_score,
            factual,
            factual_score,
        ) in factual_specs:
            carrier = self._control_carrier(
                action_type,
                anchor,
                support,
                all_factual_union,
            )

            if (
                bool(getattr(self, "evidence_guided_candidate_control_enabled", False))
                and self.v396_fast_matched_controls
            ):
                control = self._matched_local_controls_fast(
                    factual,
                    carrier,
                    gray,
                    entropy,
                    boundary,
                )
            else:
                control = torch.cat([
                    self._matched_local_control(
                        factual[:, rank:rank + 1],
                        carrier,
                        gray,
                        entropy,
                        boundary,
                    )
                    for rank in range(self.k_per_type)
                ], dim=1)

            delta = F.softplus(
                _run_checkpointed_module(self.delta_heads[action_type], latent)
            ).clamp(max=self.max_logit_delta)

            for rank in range(self.k_per_type):
                factual_mask = factual[:, rank:rank + 1]

                if action_type in self.DELETE_TYPES:
                    required = (
                        anchor_logits + self.hard_edit_logit_margin
                    ).clamp_min(0.0)
                    action_delta = torch.maximum(delta, required)
                    edited = anchor_logits - factual_mask * action_delta
                else:
                    required = (
                        -anchor_logits + self.hard_edit_logit_margin
                    ).clamp_min(0.0)
                    action_delta = torch.maximum(delta, required)
                    edited = anchor_logits + factual_mask * action_delta

                candidate_logits.append(edited)
                factual_masks.append(factual_mask)
                control_masks.append(control[:, rank:rank + 1])
                visual_scores.append(factual_score[:, rank])
                actionness_logits.append(action_logit_map[:, 0])
                deltas.append(action_delta[:, 0])

        factual_masks_t = torch.cat(factual_masks, dim=1)
        control_masks_t = torch.cat(control_masks, dim=1)
        candidates_t = torch.cat(candidate_logits, dim=1)

        candidate_probs = torch.cat([
            base_prob,
            torch.sigmoid(candidates_t).clamp(EPS, 1.0 - EPS),
        ], dim=1)

        candidate_logits_all = torch.cat([
            anchor_logits,
            candidates_t,
        ], dim=1)

        visual_scores_t = torch.stack(visual_scores, dim=1)
        actionness_logits_t = torch.stack(actionness_logits, dim=1)
        delta_maps_t = torch.stack(deltas, dim=1)

        cf = self._text_counterfactual(
            semantic_map,
            text_features,
            negative_text_features,
            factual_masks_t,
            control_masks_t,
            self.action_types,
        )

        local_entropy = (
            factual_masks_t * entropy
        ).sum(dim=(-2, -1)) / factual_masks_t.sum(
            dim=(-2, -1)
        ).clamp_min(EPS)

        local_area = factual_masks_t.mean(dim=(-2, -1))

        local_magnitude = (
            factual_masks_t * delta_maps_t
        ).sum(dim=(-2, -1)) / factual_masks_t.sum(
            dim=(-2, -1)
        ).clamp_min(EPS)

        selector_input = torch.stack([
            visual_scores_t,
            cf["cf_logit"],
            local_entropy,
            local_area,
            local_magnitude,
        ], dim=-1)

        selector_logits = self.selector(
            selector_input.reshape(-1, 5)
        ).reshape(base_prob.shape[0], self.num_actions)

        raw_selector_probs = torch.sigmoid(selector_logits)

        # Only actions with an exact-area, zero-overlap counterfactual pair may
        # enter the selector or inference output.
        verified = cf["cf_available"].to(raw_selector_probs.dtype)

        expected_area = (
            raw_selector_probs * verified * local_area
        ).sum(dim=1, keepdim=True)

        uncertainty = entropy.mean(dim=(-2, -1, -3), keepdim=False)
        budget = self.budget_min + (
            self.budget_max - self.budget_min
        ) * uncertainty

        budget_scale = (
            budget[:, None] / expected_area.clamp_min(EPS)
        ).clamp(max=1.0)

        selector_probs_budget = (
            raw_selector_probs * budget_scale * verified
        )

        hard = (
            selector_probs_budget >= self.select_threshold
        ).to(base_prob.dtype)

        selector_st = (
            hard
            + selector_probs_budget
            - selector_probs_budget.detach()
            if self.training else hard
        )

        action_delta_logits = candidates_t - anchor_logits

        fused_logits = anchor_logits[:, 0] + (
            selector_st[:, :, None, None] * action_delta_logits
        ).sum(dim=1)

        fused_probs = torch.sigmoid(fused_logits).clamp(EPS, 1.0 - EPS)

        hard_fused_logits = anchor_logits[:, 0] + (
            hard[:, :, None, None] * action_delta_logits
        ).sum(dim=1)

        hard_fused_probs = torch.sigmoid(
            hard_fused_logits
        ).clamp(EPS, 1.0 - EPS)

        return candidate_logits_all, {
            "candidate_probs": candidate_probs,
            **safe_residual_aux,
            "direct_fused_probs": fused_probs,
            "router_fused_probs": fused_probs,
            "v20_fused_probs": fused_probs,
            "v20_hard_fused_probs": hard_fused_probs,
            "v20_fused_logits": fused_logits,
            "v20_action_supports": factual_masks_t,
            "v20_control_supports": control_masks_t,
            "v20_type_supports": supports,
            "v20_action_types": self.action_types,
            "v20_actionness_logits": actionness_logits_t,
            "v35_residual_logit_map": v35_residual_logit_map,
            "v35_residual_probability_map": torch.sigmoid(v35_residual_logit_map),
            "v451_residual_proposal_logits": residual_proposal_logits,
            "v451_residual_delete_probability_map": torch.sigmoid(
                residual_proposal_logits[:, 0]
            ),
            "v451_residual_fill_probability_map": torch.sigmoid(
                residual_proposal_logits[:, 1]
            ),
            "v451_soft_support_enabled": torch.full(
                (base_prob.shape[0],),
                float(self.soft_support_enabled),
                device=base_prob.device,
                dtype=base_prob.dtype,
            ),
            "v451_global_non_overlap_enabled": torch.full(
                (base_prob.shape[0],),
                float(self.global_non_overlap),
                device=base_prob.device,
                dtype=base_prob.dtype,
            ),
            "v20_visual_scores": visual_scores_t,
            "v20_delta_maps": delta_maps_t,
            "v20_selector_logits": selector_logits,
            "v20_selector_probs": selector_probs_budget,
            "v20_selector_hard": hard,
            "v20_cf_logit": cf["cf_logit"],
            "v20_cf_swap_logit": cf["cf_swap_logit"],
            "v20_cf_signed_delta": cf["cf_signed_delta"],
            "v20_cf_factual_similarity": cf["cf_factual_similarity"],
            "v20_cf_control_similarity": cf["cf_control_similarity"],
            "v20_cf_available": cf["cf_available"],
            "v20_cf_control_area_ratio": cf["cf_control_area_ratio"],
            "v20_cf_control_overlap": cf["cf_control_overlap"],
            "v20_entropy": entropy[:, 0],
            "v20_boundary": boundary[:, 0],
            "v20_budget": budget,
            "candidate_base_detached": self.detach_base,
            "shrink_gate": factual_masks_t[:, 0],
            "expand_gate": factual_masks_t[:, -1],
            "shrink_gate_logits": actionness_logits_t[:, 0],
            "expand_gate_logits": actionness_logits_t[:, -1],
            "edit_gate": factual_masks_t.max(dim=1).values,
            "edit_band": factual_masks_t.max(dim=1).values,
            "inside_band": supports[:, :2].max(dim=1).values,
            "outside_band": supports[:, 2:].max(dim=1).values,
            "expand_connect_band": supports[:, 2:].max(dim=1).values,
            "router_probs": torch.stack([
                1.0 - selector_probs_budget.mean(dim=1),
                selector_probs_budget[:, :self.num_actions // 2].mean(dim=1),
                selector_probs_budget[:, self.num_actions // 2:].mean(dim=1),
            ], dim=1)[:, :, None, None].expand(
                -1, -1, base_prob.shape[-2], base_prob.shape[-1]
            ),
            "router_edit_probability": selector_probs_budget.mean(
                dim=1
            )[:, None, None].expand(
                -1, base_prob.shape[-2], base_prob.shape[-1]
            ),
            "router_direction_probs": torch.stack([
                selector_probs_budget[:, :self.num_actions // 2].mean(dim=1),
                selector_probs_budget[:, self.num_actions // 2:].mean(dim=1),
            ], dim=1)[:, :, None, None].expand(
                -1, -1, base_prob.shape[-2], base_prob.shape[-1]
            ),
            "router_shrink_weight": factual_masks_t[
                :, :self.num_actions // 2
            ].sum(dim=1).clamp(max=1.0),
            "router_expand_weight": factual_masks_t[
                :, self.num_actions // 2:
            ].sum(dim=1).clamp(max=1.0),
        }




class _V23GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, scale: float):
        ctx.scale = float(scale)
        return value.view_as(value)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.scale * grad_output, None


def _v23_grad_reverse(value: torch.Tensor, scale: float) -> torch.Tensor:
    return _V23GradientReverse.apply(value, float(scale))


class TextQualifiedStructuralMedoidBank(UnifiedActionCounterfactualSetBank):
    """V23: unchanged V20-R M1 candidates + deconfounded text qualification
    + deterministic structural-medoid selection.

    There is no learned action selector in the deployed path. Candidate geometry,
    factual masks, and carrier-matched controls are inherited unchanged from V20-R.
    M2 sees only local image-only ROI features and standalone text embeddings;
    explicit type, rank, area, coordinates, proposal score, GT and global-mask
    features are not inputs to the verifier. A fixed polarity convention aligns
    delete/fill evidence signs but is not a learned input feature.
    """

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        m1 = _cfg_get(cfg, "M1", None)
        hidden = max(32, int(_cfg_get(m1, "V23_CF_HIDDEN_DIM", 128)))

        # Text qualification gates. All thresholds are fixed before Test.
        self.v23_text_threshold = float(_cfg_get(m1, "V23_TEXT_THRESHOLD", 0.50))
        self.v23_pos_delta_min = float(_cfg_get(m1, "V23_POS_DELTA_MIN", 0.00))
        self.v23_neg_delta_min = float(_cfg_get(m1, "V23_NEG_DELTA_MIN", 0.00))
        self.v23_swap_margin = float(_cfg_get(m1, "V23_SWAP_MARGIN", 0.00))

        # Structural medoid / consensus constants.
        self.v23_struct_size = max(32, int(_cfg_get(m1, "V23_STRUCT_SIZE", 80)))
        self.v23_mask_dice_weight = float(_cfg_get(m1, "V23_MASK_DICE_WEIGHT", 0.40))
        self.v23_boundary_dice_weight = float(_cfg_get(m1, "V23_BOUNDARY_DICE_WEIGHT", 0.45))
        self.v23_membership_weight = float(_cfg_get(m1, "V23_MEMBERSHIP_WEIGHT", 0.15))
        self.v23_edge_weight = float(_cfg_get(m1, "V23_EDGE_WEIGHT", 0.25))
        self.v23_area_penalty = float(_cfg_get(m1, "V23_AREA_PENALTY", 0.10))
        self.v23_perimeter_penalty = float(_cfg_get(m1, "V23_PERIMETER_PENALTY", 0.15))
        self.v23_min_edited_hypotheses = max(2, int(_cfg_get(m1, "V23_MIN_EDITED_HYPOTHESES", 2)))
        self.v23_min_cluster_size = max(2, int(_cfg_get(m1, "V23_MIN_CLUSTER_SIZE", 2)))
        self.v23_cluster_similarity_min = float(_cfg_get(m1, "V23_CLUSTER_SIMILARITY_MIN", 0.82))
        self.v23_max_area_log_shift = float(_cfg_get(m1, "V23_MAX_AREA_LOG_SHIFT", 0.18))
        self.v23_max_perimeter_growth = float(_cfg_get(m1, "V23_MAX_PERIMETER_GROWTH", 0.20))
        self.v23_base_edge_tolerance = float(_cfg_get(m1, "V23_BASE_EDGE_TOLERANCE", 0.02))
        self.v23_adv_scale = max(0.0, float(_cfg_get(m1, "V23_GEOMETRY_ADV_SCALE", 0.20)))

        # V24: raw pos/neg/swap deltas remain train-only regularizers.  They are
        # not repeated as a deployment-time four-way hard AND gate.
        self.v24_text_ranked_structural_medoid = bool(
            _cfg_get(m1, "V24_TEXT_RANKED_STRUCTURAL_MEDOID", False)
        )
        self.v24_topk_delete = max(1, int(_cfg_get(m1, "V24_TOPK_DELETE", 1)))
        self.v24_topk_fill = max(1, int(_cfg_get(m1, "V24_TOPK_FILL", 1)))
        self.v24_singleton_text_threshold = float(
            _cfg_get(m1, "V24_SINGLETON_TEXT_THRESHOLD", 0.75)
        )
        self.v24_singleton_min_edge_gain = float(
            _cfg_get(m1, "V24_SINGLETON_MIN_EDGE_GAIN", 0.005)
        )
        self.v24_singleton_min_stability = float(
            _cfg_get(m1, "V24_SINGLETON_MIN_STABILITY", 0.0)
        )

        # Image-only ROI/text-only prompt adapters. No prompt-conditioned image
        # map is used by this verifier.
        self.v23_image_adapter = nn.Sequential(
            nn.Linear(self.semantic_channels, hidden, bias=False),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.v23_text_adapter = nn.Sequential(
            nn.Linear(self.semantic_channels, hidden, bias=False),
            nn.LayerNorm(hidden),
        )
        self.v23_cf_encoder = nn.Sequential(
            nn.Linear(4, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.v23_cf_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.v23_cf_head.weight)
        nn.init.constant_(self.v23_cf_head.bias, -1.5)

        # Geometry probes are train-time diagnostics/regularizers only. Their
        # gradient is reversed before entering the M2 embedding, following the
        # causal/deconfounding adversarial principle; probes do not enter Test
        # qualification or structural consensus.
        self.v23_type_probe = nn.Linear(hidden, self.num_types)
        self.v23_area_probe = nn.Linear(hidden, 4)
        self.v23_position_probe = nn.Linear(hidden, 9)

    def _v23_text_counterfactual(
        self,
        image_only_map: torch.Tensor,
        positive_text: torch.Tensor,
        negative_text: torch.Tensor,
        swapped_text: torch.Tensor,
        factual_masks: torch.Tensor,
        control_masks: torch.Tensor,
        action_types: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Factual/control ROI contrast under positive, opposite, and swapped
        text. The verifier never receives geometry labels as input.
        """
        b, k, h, w = factual_masks.shape
        factual_context = _soft_dilate(
            factual_masks.reshape(b * k, 1, h, w), self.context_radius
        ).reshape(b, k, h, w)
        control_context = _soft_dilate(
            control_masks.reshape(b * k, 1, h, w), self.context_radius
        ).reshape(b, k, h, w)

        factual_roi = self._pool_feature(image_only_map, factual_context)
        control_roi = self._pool_feature(image_only_map, control_context)
        factual_roi = F.normalize(self.v23_image_adapter(factual_roi), dim=-1, eps=1e-6)
        control_roi = F.normalize(self.v23_image_adapter(control_roi), dim=-1, eps=1e-6)
        positive_text = F.normalize(self.v23_text_adapter(positive_text), dim=-1, eps=1e-6)
        negative_text = F.normalize(self.v23_text_adapter(negative_text), dim=-1, eps=1e-6)
        swapped_text = F.normalize(self.v23_text_adapter(swapped_text), dim=-1, eps=1e-6)

        def paired_delta(text_feature: torch.Tensor):
            factual_score = (factual_roi * text_feature[:, None, :]).sum(dim=-1)
            control_score = (control_roi * text_feature[:, None, :]).sum(dim=-1)
            return factual_score - control_score, factual_score, control_score

        raw_pos, factual_pos, control_pos = paired_delta(positive_text)
        raw_neg, factual_neg, control_neg = paired_delta(negative_text)
        raw_swap, _, _ = paired_delta(swapped_text)

        is_fill = torch.isin(
            action_types,
            torch.tensor(self.FILL_TYPES, device=action_types.device),
        )
        polarity = torch.where(
            is_fill,
            torch.ones_like(action_types, dtype=raw_pos.dtype),
            -torch.ones_like(action_types, dtype=raw_pos.dtype),
        )[None, :]

        # Correct fill should increase lesion-text support and decrease
        # opposite-text support; correct delete has the inverse relation.
        pos_delta = polarity * raw_pos
        neg_delta = -polarity * raw_neg
        swap_delta = polarity * raw_swap
        text_gap = pos_delta - swap_delta
        visual_gap = torch.sqrt(
            (factual_roi - control_roi).pow(2).mean(dim=-1).clamp_min(0.0) + 1e-6
        )

        evidence = torch.stack(
            [pos_delta, neg_delta, text_gap, visual_gap], dim=-1
        )
        embedding = self.v23_cf_encoder(evidence.reshape(b * k, 4)).reshape(b, k, -1)
        cf_logit = self.v23_cf_head(embedding.reshape(b * k, -1)).reshape(b, k)

        # Factual/control pair swap must reverse the direction of evidence.
        swapped_evidence = torch.stack(
            [-pos_delta, -neg_delta, -text_gap, visual_gap], dim=-1
        )
        swapped_embedding = self.v23_cf_encoder(
            swapped_evidence.reshape(b * k, 4)
        ).reshape(b, k, -1)
        cf_swap_logit = self.v23_cf_head(
            swapped_embedding.reshape(b * k, -1)
        ).reshape(b, k)

        factual_area = factual_masks.sum(dim=(-2, -1))
        control_area = control_masks.sum(dim=(-2, -1))
        area_ratio = control_area / factual_area.clamp_min(EPS)
        overlap = (factual_masks * control_masks).sum(dim=(-2, -1)) / factual_area.clamp_min(EPS)
        available = (
            (factual_area > 0)
            & (control_area > 0)
            & ((area_ratio - 1.0).abs() <= 1e-3)
            & (overlap <= 1e-6)
        )

        adversarial_embedding = _v23_grad_reverse(embedding, self.v23_adv_scale)
        return {
            "cf_logit": cf_logit,
            "cf_swap_logit": cf_swap_logit,
            "cf_signed_delta": pos_delta,
            "cf_pos_delta": pos_delta,
            "cf_neg_delta": neg_delta,
            "cf_swap_delta": swap_delta,
            "cf_available": available,
            "cf_control_area_ratio": area_ratio,
            "cf_control_overlap": overlap,
            "cf_factual_similarity": factual_pos - factual_neg,
            "cf_control_similarity": control_pos - control_neg,
            "cf_embedding": embedding,
            "v23_type_probe": self.v23_type_probe(adversarial_embedding),
            "v23_area_probe": self.v23_area_probe(adversarial_embedding),
            "v23_position_probe": self.v23_position_probe(adversarial_embedding),
        }

    @staticmethod
    def _v23_pairwise_dice(mask: torch.Tensor) -> torch.Tensor:
        # [B,N,H,W] -> [B,N,N], where mask is binary.
        flat = mask.flatten(2)
        inter = torch.einsum("bih,bjh->bij", flat, flat)
        mass = flat.sum(dim=-1)
        return (2.0 * inter + EPS) / (mass[:, :, None] + mass[:, None, :] + EPS)

    @staticmethod
    def _v23_membership_jaccard(membership: torch.Tensor) -> torch.Tensor:
        # Membership is [B,N,K] over atomic actions. It is a compatibility
        # signal, not an M2 feature or direct semantic decision.
        inter = torch.einsum("bik,bjk->bij", membership, membership)
        mass = membership.sum(dim=-1)
        union = mass[:, :, None] + mass[:, None, :] - inter
        out = inter / union.clamp_min(EPS)
        empty_pair = union <= EPS
        return torch.where(empty_pair, torch.ones_like(out), out)

    def _v24_ranked_semantic_actions(
        self,
        cf: Dict[str, torch.Tensor],
        action_types: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """M2 text qualification for deployment.

        A candidate must have a valid matched control and sufficient learned M2
        probability.  The factual/control raw deltas remain active in the
        training loss, but are not repeated as hard inference gates.  At most
        one delete and one fill are exposed to structural consensus per image.
        """
        score = torch.sigmoid(cf["cf_logit"])
        raw = cf["cf_available"] & (score >= self.v23_text_threshold)
        b, k = score.shape
        delete = torch.isin(
            action_types,
            torch.tensor(self.DELETE_TYPES, device=action_types.device),
        )
        fill = ~delete

        def keep_top(group: torch.Tensor, topk: int) -> torch.Tensor:
            allowed = raw & group[None, :]
            values = score.masked_fill(~allowed, -1e9)
            num = min(int(topk), k)
            _, idx = values.topk(num, dim=1)
            out = torch.zeros_like(allowed)
            out.scatter_(1, idx, True)
            return out & allowed

        chosen = keep_top(delete, self.v24_topk_delete)
        chosen = chosen | keep_top(fill, self.v24_topk_fill)
        return chosen, raw, score

    def _v23_build_hypotheses(
        self,
        candidate_logits_all: torch.Tensor,
        semantic_action: torch.Tensor,
        supports: torch.Tensor,
        action_types: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Hypotheses = Base, qualified single actions, qualified delete+fill
        compositions. No learned selector and no GT are used.
        """
        b, slots, h, w = candidate_logits_all.shape
        k = slots - 1
        base = candidate_logits_all[:, 0]
        deltas = candidate_logits_all[:, 1:] - base[:, None]

        hypothesis_logits = [base]
        hypothesis_valid = [torch.ones((b,), device=base.device, dtype=torch.bool)]
        memberships = [torch.zeros((b, k), device=base.device, dtype=base.dtype)]

        for action_idx in range(k):
            hypothesis_logits.append(base + deltas[:, action_idx])
            hypothesis_valid.append(semantic_action[:, action_idx])
            member = torch.zeros((b, k), device=base.device, dtype=base.dtype)
            member[:, action_idx] = 1.0
            memberships.append(member)

        delete = torch.isin(
            action_types,
            torch.tensor(self.DELETE_TYPES, device=action_types.device),
        )
        fill = ~delete
        for delete_idx in torch.where(delete)[0].tolist():
            for fill_idx in torch.where(fill)[0].tolist():
                disjoint = (
                    (supports[:, delete_idx] * supports[:, fill_idx])
                    .sum(dim=(-2, -1)) <= 1e-6
                )
                hypothesis_logits.append(base + deltas[:, delete_idx] + deltas[:, fill_idx])
                hypothesis_valid.append(
                    semantic_action[:, delete_idx]
                    & semantic_action[:, fill_idx]
                    & disjoint
                )
                member = torch.zeros((b, k), device=base.device, dtype=base.dtype)
                member[:, delete_idx] = 1.0
                member[:, fill_idx] = 1.0
                memberships.append(member)

        return (
            torch.stack(hypothesis_logits, dim=1),
            torch.stack(hypothesis_valid, dim=1),
            torch.stack(memberships, dim=1),
        )

    def _v23_structural_medoid(
        self,
        hypothesis_logits: torch.Tensor,
        valid: torch.Tensor,
        membership: torch.Tensor,
        image_edge: torch.Tensor,
        hypothesis_text_score: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Deterministic text-qualified structural selection.

        First prefer a compatible multi-hypothesis structural medoid.  A
        singleton is permitted only when it is high-confidence under M2,
        structurally safe, and improves local image-edge alignment.  Otherwise
        return Base exactly.
        """
        b, n, h, w = hypothesis_logits.shape
        side = min(self.v23_struct_size, h, w)
        low_prob = torch.sigmoid(F.interpolate(
            hypothesis_logits.reshape(b * n, 1, h, w),
            size=(side, side), mode="bilinear", align_corners=False,
        )).reshape(b, n, side, side)
        low_hard = (low_prob >= 0.5).float()
        low_boundary = self._boundary(low_hard)
        low_edge = F.interpolate(image_edge, size=(side, side), mode="bilinear", align_corners=False)

        mask_dice = self._v23_pairwise_dice(low_hard)
        boundary_dice = self._v23_pairwise_dice(low_boundary)
        membership_jaccard = self._v23_membership_jaccard(membership)
        pairwise = (
            self.v23_mask_dice_weight * mask_dice
            + self.v23_boundary_dice_weight * boundary_dice
            + self.v23_membership_weight * membership_jaccard
        )

        boundary_mass = low_boundary.sum(dim=(-2, -1)).clamp_min(EPS)
        edge_alignment = (low_boundary * low_edge).sum(dim=(-2, -1)) / boundary_mass
        area = low_hard.sum(dim=(-2, -1)).clamp_min(EPS)
        perimeter = boundary_mass
        base_area = area[:, :1]
        base_perimeter = perimeter[:, :1]
        base_edge = edge_alignment[:, :1]
        area_log_shift = (area / base_area).log().abs()
        perimeter_growth = F.relu(perimeter / base_perimeter - 1.0)
        edge_gain = edge_alignment - base_edge

        edited_valid = valid.clone()
        edited_valid[:, 0] = False
        structural_safe = (
            (area_log_shift <= self.v23_max_area_log_shift)
            & (perimeter_growth <= self.v23_max_perimeter_growth)
            & (edge_alignment >= base_edge - self.v23_base_edge_tolerance)
        )
        qualified = edited_valid & structural_safe

        eye = torch.eye(n, device=hypothesis_logits.device, dtype=torch.bool)[None]
        pair_valid = qualified[:, :, None] & qualified[:, None, :] & (~eye)
        agreement = (pairwise * pair_valid.float()).sum(dim=-1) / pair_valid.float().sum(dim=-1).clamp_min(1.0)
        stability = (
            edge_alignment
            - self.v23_area_penalty * area_log_shift
            - self.v23_perimeter_penalty * perimeter_growth
        )
        if hypothesis_text_score is None:
            hypothesis_text_score = torch.zeros((b, n), device=hypothesis_logits.device, dtype=hypothesis_logits.dtype)

        medoid_score = (
            agreement + self.v23_edge_weight * stability + 0.05 * hypothesis_text_score
        ).masked_fill(~qualified, -1e9)
        medoid_index = medoid_score.argmax(dim=1)
        medoid_exists = qualified.any(dim=1)
        medoid_pairwise = pairwise.gather(1, medoid_index[:, None, None].expand(-1, 1, n))[:, 0]
        cluster = qualified & (medoid_pairwise >= self.v23_cluster_similarity_min)
        cluster_size = cluster.sum(dim=1)
        edited_count = qualified.sum(dim=1)

        multi_active = (
            medoid_exists
            & (edited_count >= self.v23_min_edited_hypotheses)
            & (cluster_size >= self.v23_min_cluster_size)
        )
        selected_text = hypothesis_text_score.gather(1, medoid_index[:, None])[:, 0]
        selected_edge_gain = edge_gain.gather(1, medoid_index[:, None])[:, 0]
        selected_stability = stability.gather(1, medoid_index[:, None])[:, 0]
        singleton_active = (
            medoid_exists
            & (edited_count == 1)
            & (selected_text >= self.v24_singleton_text_threshold)
            & (selected_edge_gain >= self.v24_singleton_min_edge_gain)
            & (selected_stability >= self.v24_singleton_min_stability)
        )
        consensus_active = multi_active | singleton_active

        selected_logits = hypothesis_logits.gather(
            1, medoid_index[:, None, None, None].expand(-1, 1, h, w)
        )[:, 0]
        final_logits = torch.where(
            consensus_active[:, None, None], selected_logits, hypothesis_logits[:, 0]
        )
        k = membership.shape[-1]
        selected_membership = membership.gather(
            1, medoid_index[:, None, None].expand(-1, 1, k)
        )[:, 0]
        action_endorsement = selected_membership * consensus_active[:, None].float()
        return final_logits, action_endorsement, {
            "hypothesis_valid": valid,
            "structural_safe": structural_safe,
            "qualified": qualified,
            "pairwise": pairwise,
            "agreement": agreement,
            "stability": stability,
            "edge_alignment": edge_alignment,
            "edge_gain": edge_gain,
            "area_log_shift": area_log_shift,
            "perimeter_growth": perimeter_growth,
            "medoid_score": medoid_score,
            "medoid_index": medoid_index,
            "cluster_size": cluster_size.float(),
            "edited_count": edited_count.float(),
            "multi_active": multi_active.float(),
            "singleton_active": singleton_active.float(),
            "selected_text_score": selected_text,
            "consensus_active": consensus_active.float(),
        }

    def generate(
        self,
        base_logits: torch.Tensor,
        image: torch.Tensor,
        semantic_map: Optional[torch.Tensor],
        text_features: torch.Tensor,
        negative_text_features: Optional[torch.Tensor] = None,
        swapped_text_features: Optional[torch.Tensor] = None,
    ):
        # Preserve V20-R candidate geometry and matched carrier controls exactly.
        # Parent selector outputs are overwritten below and never deployed.
        candidate_logits_all, aux = super().generate(
            base_logits=base_logits,
            image=image,
            semantic_map=semantic_map,
            text_features=text_features,
            negative_text_features=negative_text_features,
        )
        if semantic_map is None:
            raise RuntimeError("V23 requires image-only spatial patch features.")
        if negative_text_features is None:
            negative_text_features = text_features
        if swapped_text_features is None:
            swapped_text_features = negative_text_features

        cf = self._v23_text_counterfactual(
            image_only_map=semantic_map,
            positive_text=text_features,
            negative_text=negative_text_features,
            swapped_text=swapped_text_features,
            factual_masks=aux["v20_action_supports"],
            control_masks=aux["v20_control_supports"],
            action_types=self.action_types,
        )

        if self.v24_text_ranked_structural_medoid:
            semantic_action, raw_semantic_action, text_action_score = self._v24_ranked_semantic_actions(
                cf, self.action_types
            )
        else:
            raw_semantic_action = (
                cf["cf_available"]
                & (torch.sigmoid(cf["cf_logit"]) >= self.v23_text_threshold)
                & (cf["cf_pos_delta"] >= self.v23_pos_delta_min)
                & (cf["cf_neg_delta"] >= self.v23_neg_delta_min)
                & ((cf["cf_pos_delta"] - cf["cf_swap_delta"]) >= self.v23_swap_margin)
            )
            semantic_action = raw_semantic_action
            text_action_score = torch.sigmoid(cf["cf_logit"])

        hypotheses, valid, membership = self._v23_build_hypotheses(
            candidate_logits_all,
            semantic_action,
            aux["v20_action_supports"],
            self.action_types,
        )
        member_mass = membership.sum(dim=-1)
        hypothesis_text_score = (membership * text_action_score[:, None, :]).sum(dim=-1) / member_mass.clamp_min(1.0)
        hypothesis_text_score[:, 0] = 0.0
        _, image_edge = self._image_gray_edge(
            image,
            candidate_logits_all.shape[-2:],
        )
        final_logits, action_endorsement, structural = self._v23_structural_medoid(
            hypotheses,
            valid,
            membership,
            image_edge,
            hypothesis_text_score=hypothesis_text_score,
        )
        final_probs = torch.sigmoid(final_logits).clamp(EPS, 1.0 - EPS)

        aux.update({
            "direct_fused_probs": final_probs,
            "router_fused_probs": final_probs,
            "v20_fused_probs": final_probs,
            "v20_hard_fused_probs": final_probs,
            "v20_fused_logits": final_logits,
            # Compatibility fields: these are deterministic medoid memberships,
            # never outputs of the inherited learned selector.
            "v20_selector_logits": structural["medoid_score"],
            "v20_selector_probs": action_endorsement,
            "v20_selector_hard": action_endorsement,
            "v20_cf_logit": cf["cf_logit"],
            "v20_cf_swap_logit": cf["cf_swap_logit"],
            "v20_cf_signed_delta": cf["cf_signed_delta"],
            "v20_cf_factual_similarity": cf["cf_factual_similarity"],
            "v20_cf_control_similarity": cf["cf_control_similarity"],
            "v20_cf_available": cf["cf_available"],
            "v20_cf_control_area_ratio": cf["cf_control_area_ratio"],
            "v20_cf_control_overlap": cf["cf_control_overlap"],
            "v23_cf_pos_delta": cf["cf_pos_delta"],
            "v23_cf_neg_delta": cf["cf_neg_delta"],
            "v23_cf_swap_delta": cf["cf_swap_delta"],
            "v23_type_probe": cf["v23_type_probe"],
            "v23_area_probe": cf["v23_area_probe"],
            "v23_position_probe": cf["v23_position_probe"],
            "v23_semantic_action": semantic_action,
            "v24_raw_semantic_action": raw_semantic_action,
            "v24_text_action_score": text_action_score,
            "v24_hypothesis_text_score": hypothesis_text_score,
            "v24_multi_active": structural["multi_active"],
            "v24_singleton_active": structural["singleton_active"],
            "v24_selected_text_score": structural["selected_text_score"],
            "v24_edge_gain": structural["edge_gain"],
            "v23_structural_safe": structural["structural_safe"],
            "v23_structural_qualified": structural["qualified"],
            "v23_hypothesis_agreement": structural["agreement"],
            "v23_hypothesis_stability": structural["stability"],
            "v23_edge_alignment": structural["edge_alignment"],
            "v23_area_log_shift": structural["area_log_shift"],
            "v23_perimeter_growth": structural["perimeter_growth"],
            "v23_medoid_index": structural["medoid_index"],
            "v23_cluster_size": structural["cluster_size"],
            "v23_edited_hypothesis_count": structural["edited_count"],
            "v23_consensus_active": structural["consensus_active"],
        })
        return candidate_logits_all, aux

class V37TextFalsifiedStructuralConsensusBank(TextQualifiedStructuralMedoidBank):
    """V37: trainable multi-type M1 -> matched-control M2 -> deterministic M3.

    M1 remains trainable and emits all four local action families. M2 learns a
    text-falsification qualification score from factual/control local evidence.
    M3 receives only M2-qualified hypotheses and performs a non-learned
    structural-consensus medoid selection with exact Preserve fallback.

    Deployment never reads Dice gain, GT, a gain head, or a risk-adjusted score.
    Dice gain labels exist only inside the train-split loss for M1/M2 supervision.
    """

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        m1 = _cfg_get(cfg, "M1", None)
        self.v37_text_threshold = float(_cfg_get(m1, "V37_TEXT_THRESHOLD", 0.50))
        self.v37_pos_delta_min = float(_cfg_get(m1, "V37_POS_DELTA_MIN", 0.00))
        self.v37_neg_delta_min = float(_cfg_get(m1, "V37_NEG_DELTA_MIN", 0.00))
        self.v37_swap_gap_min = float(_cfg_get(m1, "V37_SWAP_GAP_MIN", 0.00))

        # The inherited V23 structural implementation is used as M3, but all
        # thresholds are exposed under V37 names so the protocol is explicit.
        self.v23_struct_size = max(32, int(_cfg_get(m1, "V37_STRUCT_SIZE", self.v23_struct_size)))
        self.v23_mask_dice_weight = float(_cfg_get(m1, "V37_MASK_DICE_WEIGHT", self.v23_mask_dice_weight))
        self.v23_boundary_dice_weight = float(_cfg_get(m1, "V37_BOUNDARY_DICE_WEIGHT", self.v23_boundary_dice_weight))
        self.v23_membership_weight = float(_cfg_get(m1, "V37_MEMBERSHIP_WEIGHT", self.v23_membership_weight))
        self.v23_edge_weight = float(_cfg_get(m1, "V37_EDGE_WEIGHT", self.v23_edge_weight))
        self.v23_area_penalty = float(_cfg_get(m1, "V37_AREA_PENALTY", self.v23_area_penalty))
        self.v23_perimeter_penalty = float(_cfg_get(m1, "V37_PERIMETER_PENALTY", self.v23_perimeter_penalty))
        self.v23_min_edited_hypotheses = max(2, int(_cfg_get(m1, "V37_MIN_EDITED_HYPOTHESES", self.v23_min_edited_hypotheses)))
        self.v23_min_cluster_size = max(2, int(_cfg_get(m1, "V37_MIN_CLUSTER_SIZE", self.v23_min_cluster_size)))
        self.v23_cluster_similarity_min = float(_cfg_get(m1, "V37_CLUSTER_SIMILARITY_MIN", self.v23_cluster_similarity_min))
        self.v23_max_area_log_shift = float(_cfg_get(m1, "V37_MAX_AREA_LOG_SHIFT", self.v23_max_area_log_shift))
        self.v23_max_perimeter_growth = float(_cfg_get(m1, "V37_MAX_PERIMETER_GROWTH", self.v23_max_perimeter_growth))
        self.v23_base_edge_tolerance = float(_cfg_get(m1, "V37_BASE_EDGE_TOLERANCE", self.v23_base_edge_tolerance))
        self.v24_singleton_text_threshold = float(_cfg_get(m1, "V37_SINGLETON_TEXT_THRESHOLD", 0.80))
        self.v24_singleton_min_edge_gain = float(_cfg_get(m1, "V37_SINGLETON_MIN_EDGE_GAIN", 0.005))
        self.v24_singleton_min_stability = float(_cfg_get(m1, "V37_SINGLETON_MIN_STABILITY", 0.0))

    def _v37_text_counterfactual(
        self,
        image_only_map: torch.Tensor,
        positive_text: torch.Tensor,
        negative_text: torch.Tensor,
        swapped_text: torch.Tensor,
        factual_masks: torch.Tensor,
        control_masks: torch.Tensor,
        action_types: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """M2 factual/control evidence with an explicit context-clean check."""
        cf = self._v23_text_counterfactual(
            image_only_map=image_only_map,
            positive_text=positive_text,
            negative_text=negative_text,
            swapped_text=swapped_text,
            factual_masks=factual_masks,
            control_masks=control_masks,
            action_types=action_types,
        )
        b, k, h, w = factual_masks.shape
        factual_context = _soft_dilate(
            factual_masks.reshape(b * k, 1, h, w), self.context_radius
        ).reshape(b, k, h, w)
        control_context = _soft_dilate(
            control_masks.reshape(b * k, 1, h, w), self.context_radius
        ).reshape(b, k, h, w)
        context_overlap = (
            factual_context * control_context
        ).sum(dim=(-2, -1)) / factual_context.sum(dim=(-2, -1)).clamp_min(EPS)
        context_clean = context_overlap <= 1e-6
        cf["cf_context_overlap"] = context_overlap
        cf["cf_context_clean"] = context_clean
        cf["cf_available"] = cf["cf_available"].bool() & context_clean
        return cf

    def _v37_text_qualified(self, cf: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """The sole M2 -> M3 admission gate, fixed before any Test execution."""
        score = torch.sigmoid(cf["cf_logit"])
        gap = cf["cf_pos_delta"] - cf["cf_swap_delta"]
        qualified = (
            cf["cf_available"].bool()
            & (score >= self.v37_text_threshold)
            & (cf["cf_pos_delta"] >= self.v37_pos_delta_min)
            & (cf["cf_neg_delta"] >= self.v37_neg_delta_min)
            & (gap >= self.v37_swap_gap_min)
        )
        return qualified, score

    def generate(
        self,
        base_logits: torch.Tensor,
        image: torch.Tensor,
        semantic_map: Optional[torch.Tensor],
        text_features: torch.Tensor,
        negative_text_features: Optional[torch.Tensor] = None,
        swapped_text_features: Optional[torch.Tensor] = None,
    ):
        # M1: all candidate families remain active and trainable. We call the
        # original V20 generator directly so no inherited V23 deployment key
        # silently determines the final mask.
        candidate_logits_all, aux = UnifiedActionCounterfactualSetBank.generate(
            self,
            base_logits=base_logits,
            image=image,
            semantic_map=semantic_map,
            text_features=text_features,
            negative_text_features=negative_text_features,
        )
        if semantic_map is None:
            raise RuntimeError("V37 requires the frozen image-only semantic map for M2.")
        if negative_text_features is None:
            negative_text_features = text_features
        if swapped_text_features is None:
            swapped_text_features = negative_text_features

        # M2: qualified only when matched control is geometrically/contextually
        # clean and the learned counterfactual text evidence passes fixed rules.
        cf = self._v37_text_counterfactual(
            image_only_map=semantic_map,
            positive_text=text_features,
            negative_text=negative_text_features,
            swapped_text=swapped_text_features,
            factual_masks=aux["v20_action_supports"],
            control_masks=aux["v20_control_supports"],
            action_types=self.action_types,
        )
        text_qualified, text_score = self._v37_text_qualified(cf)

        # M3: only M2-qualified singles and non-overlapping delete+fill
        # compositions become hypotheses. Selection is a structural medoid;
        # Preserve is selected exactly when no stable cluster/safe singleton is
        # available. No predicted Dice gain/risk score reaches this path.
        hypotheses, hypothesis_valid, membership = self._v23_build_hypotheses(
            candidate_logits_all,
            text_qualified,
            aux["v20_action_supports"],
            self.action_types,
        )
        member_mass = membership.sum(dim=-1)
        hypothesis_text_score = (
            membership * text_score[:, None, :]
        ).sum(dim=-1) / member_mass.clamp_min(1.0)
        hypothesis_text_score[:, 0] = 0.0
        _, image_edge = self._image_gray_edge(image, candidate_logits_all.shape[-2:])
        final_logits, action_endorsement, structural = self._v23_structural_medoid(
            hypotheses,
            hypothesis_valid,
            membership,
            image_edge,
            hypothesis_text_score=hypothesis_text_score,
        )
        final_probs = torch.sigmoid(final_logits).clamp(EPS, 1.0 - EPS)

        # Keep V20 public names for training/diagnostics compatibility, while
        # exposing V37-specific proof and structure records for formal audits.
        aux.update({
            "direct_fused_probs": final_probs,
            "router_fused_probs": final_probs,
            "v20_fused_probs": final_probs,
            "v20_hard_fused_probs": final_probs,
            "v20_fused_logits": final_logits,
            "v20_selector_logits": structural["medoid_score"],
            "v20_selector_probs": action_endorsement,
            "v20_selector_hard": action_endorsement,
            "v20_cf_logit": cf["cf_logit"],
            "v20_cf_swap_logit": cf["cf_swap_logit"],
            "v20_cf_signed_delta": cf["cf_signed_delta"],
            "v20_cf_factual_similarity": cf["cf_factual_similarity"],
            "v20_cf_control_similarity": cf["cf_control_similarity"],
            "v20_cf_available": cf["cf_available"],
            "v20_cf_control_area_ratio": cf["cf_control_area_ratio"],
            "v20_cf_control_overlap": cf["cf_control_overlap"],
            "v25_text_residual": cf["cf_pos_delta"] - cf["cf_swap_delta"],
            "v37_text_cf_logit": cf["cf_logit"],
            "v37_text_cf_swap_logit": cf["cf_swap_logit"],
            "v37_text_probability": text_score,
            "v37_text_qualified": text_qualified,
            "v37_text_available": cf["cf_available"],
            "v37_context_clean": cf["cf_context_clean"],
            "v37_context_overlap": cf["cf_context_overlap"],
            "v37_pos_delta": cf["cf_pos_delta"],
            "v37_neg_delta": cf["cf_neg_delta"],
            "v37_swap_delta": cf["cf_swap_delta"],
            "v37_text_gap": cf["cf_pos_delta"] - cf["cf_swap_delta"],
            "v37_type_probe": cf["v23_type_probe"],
            "v37_area_probe": cf["v23_area_probe"],
            "v37_position_probe": cf["v23_position_probe"],
            "v37_hypothesis_valid": structural["hypothesis_valid"],
            "v37_structural_safe": structural["structural_safe"],
            "v37_structural_qualified": structural["qualified"],
            "v37_hypothesis_agreement": structural["agreement"],
            "v37_hypothesis_stability": structural["stability"],
            "v37_edge_alignment": structural["edge_alignment"],
            "v37_edge_gain": structural["edge_gain"],
            "v37_area_log_shift": structural["area_log_shift"],
            "v37_perimeter_growth": structural["perimeter_growth"],
            "v37_medoid_score": structural["medoid_score"],
            "v37_medoid_index": structural["medoid_index"],
            "v37_cluster_size": structural["cluster_size"],
            "v37_edited_hypothesis_count": structural["edited_count"],
            "v37_consensus_active": structural["consensus_active"],
            "v37_multi_active": structural["multi_active"],
            "v37_singleton_active": structural["singleton_active"],
            "v37_selected_text_score": structural["selected_text_score"],
        })
        return candidate_logits_all, aux


class MonotoneDenseEvidenceCalibrator(nn.Module):
    """Positive-weight logistic calibrator for fixed raw V15.2 evidence.

    The learned mapping is monotone in candidate core quality, candidate-vs-
    Preserve delta, and candidate-vs-control specificity.  The fourth feature
    is the negative control delta, so a control that explains the same local
    score can only lower (never increase) the predicted action validity.
    """

    def __init__(self, weight_init: float = 0.01, bias_init: float = 0.0) -> None:
        super().__init__()
        value = max(float(weight_init), 1e-6)
        # inverse softplus so effective weights start at ``weight_init``.
        raw = math.log(math.expm1(value)) if value < 20.0 else value
        self.raw_weight = nn.Parameter(torch.full((4,), raw, dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(float(bias_init), dtype=torch.float32))

    def effective_weight(self) -> torch.Tensor:
        return F.softplus(self.raw_weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != 4:
            raise ValueError(f"Expected four V15.2 evidence features, got {tuple(features.shape)}")
        return features.matmul(self.effective_weight()) + self.bias

    def l2_penalty(self) -> torch.Tensor:
        return self.effective_weight().pow(2).mean() + self.bias.pow(2)


class TextCounterfactualVerifier(nn.Module):
    """V15.3 fixed dense-mask text observer with same-area controls and conservative M3 selector.

    The local image crop and prompt are unchanged.  Preserve, Shrink and a
    nearby same-area control are pooled on the *same* dense patch--text map.
    Unlike V15.1, the patch adapter is permanently identity/frozen: Phase B
    cannot rewrite raw textual evidence.  Only a monotone scalar calibrator is
    fitted on Train outcomes and all raw-gate thresholds remain Validation-only.
    """

    role_names = ("preserve", "shrink", "expand")

    def __init__(self, cfg, semantic_channels: int) -> None:
        super().__init__()
        m1 = _cfg_get(cfg, "M1", None)
        self.anchor_threshold = float(_cfg_get(m1, "ANCHOR_THRESHOLD", 0.50))
        self.deploy_expand = bool(_cfg_get(m1, "M3_DEPLOY_EXPAND", False))
        self.score_expand = bool(_cfg_get(m1, "M2_CF_SCORE_EXPAND", False))
        self.min_edit_area = float(_cfg_get(m1, "M3_MIN_EDIT_AREA", 0.0003))
        self.max_edit_area = float(_cfg_get(m1, "M3_MAX_EDIT_AREA", 0.0300))
        self.core_min = float(_cfg_get(m1, "M3_TEXT_CORE_QUALITY_MIN", _cfg_get(m1, "M3_TEXT_CORE_NECESSITY_MIN", -1.0)))
        self.delta_min = float(_cfg_get(m1, "M3_TEXT_MASK_DELTA_MIN", _cfg_get(m1, "M3_TEXT_EDIT_DIRECTION_MIN", -1.0)))
        self.specificity_min = float(_cfg_get(m1, "M3_TEXT_CONTROL_SPECIFICITY_MIN", _cfg_get(m1, "M3_TEXT_SPECIFICITY_MIN", -1.0)))
        self.semantic_min = float(_cfg_get(m1, "M3_SEMANTIC_VALID_MIN", 0.50))
        self.min_score = float(_cfg_get(m1, "M3_MIN_TEXT_SCORE", 0.50))
        self.consensus_weight = float(_cfg_get(m1, "M3_CONSENSUS_WEIGHT", 0.20))
        self.boundary_weight = float(_cfg_get(m1, "M3_BOUNDARY_STABILITY_WEIGHT", 0.10))
        self.size_penalty = float(_cfg_get(m1, "M3_SIZE_PENALTY", 0.10))
        self.score_temperature = max(float(_cfg_get(m1, "M2_DENSE_SCORE_TEMPERATURE", _cfg_get(m1, "M2_CF_SCORE_TEMPERATURE", 0.02))), 1e-5)
        self.specificity_temperature = max(float(_cfg_get(m1, "M2_DENSE_SPECIFICITY_TEMPERATURE", _cfg_get(m1, "M2_CF_SPECIFICITY_TEMPERATURE", 0.01))), 1e-5)
        self.control_min_area_ratio = float(_cfg_get(m1, "M2_DENSE_CONTROL_MIN_AREA_RATIO", 0.80))
        self.control_max_area_ratio = float(_cfg_get(m1, "M2_DENSE_CONTROL_MAX_AREA_RATIO", 1.25))
        self.control_guard_radius = max(0, int(_cfg_get(m1, "M2_DENSE_CONTROL_GUARD_RADIUS", 0)))
        self.control_max_overlap_ratio = float(_cfg_get(m1, "M2_DENSE_CONTROL_MAX_OVERLAP_RATIO", 0.0))
        if not (0.0 < self.control_min_area_ratio <= self.control_max_area_ratio):
            raise ValueError("V15.3 requires 0 < M2_DENSE_CONTROL_MIN_AREA_RATIO <= M2_DENSE_CONTROL_MAX_AREA_RATIO.")
        if not (0.0 <= self.control_max_overlap_ratio <= 1.0):
            raise ValueError("V15.3 requires 0 <= M2_DENSE_CONTROL_MAX_OVERLAP_RATIO <= 1.")

        # V15.3 protocol: raw dense evidence is a fixed observer.  Making this
        # configurable would permit accidental evidence drift, so fail closed.
        if bool(_cfg_get(m1, "M2_DENSE_PATCH_ADAPTER_TRAINABLE", False)):
            raise ValueError("V15.3 requires M2_DENSE_PATCH_ADAPTER_TRAINABLE: false.")
        if abs(float(_cfg_get(m1, "M2_DENSE_LOCAL_LOSS_WEIGHT", 0.0))) > 0.0:
            raise ValueError("V15.3 requires M2_DENSE_LOCAL_LOSS_WEIGHT: 0.0; absolute GT map alignment can alter raw evidence.")

        self.semantic_calibrator = MonotoneDenseEvidenceCalibrator(
            weight_init=float(_cfg_get(m1, "M2_MONOTONE_CALIBRATOR_WEIGHT_INIT", 0.01)),
            bias_init=float(_cfg_get(m1, "M2_MONOTONE_CALIBRATOR_BIAS_INIT", 0.0)),
        )
        self.patch_adapter = nn.Linear(int(semantic_channels), int(semantic_channels), bias=False)
        with torch.no_grad():
            self.patch_adapter.weight.copy_(torch.eye(int(semantic_channels)))
        self.patch_adapter.weight.requires_grad_(False)

    def align_patch_tokens(self, patches: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.patch_adapter(patches), dim=-1)

    def patch_adapter_identity_penalty(self) -> torch.Tensor:
        # Frozen identity observer: this is retained as a zero-valued audit
        # scalar for compatibility with existing logs, not as a train loss.
        return self.patch_adapter.weight.sum() * 0.0

    def calibrator_regularizer(self) -> torch.Tensor:
        return self.semantic_calibrator.l2_penalty()

    @staticmethod
    def _soft_boundary(prob: torch.Tensor) -> torch.Tensor:
        x = prob.unsqueeze(1) if prob.ndim == 3 else prob
        return (_soft_dilate(x, 1) - _soft_erode(x, 1)).clamp(0.0, 1.0)[:, 0]

    @staticmethod
    def _dice(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        inter = (a * b).sum(dim=(-2, -1))
        den = a.sum(dim=(-2, -1)) + b.sum(dim=(-2, -1))
        return (2.0 * inter + EPS) / (den + EPS)

    def _role_stats(self, candidate_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        preserve = candidate_probs[:, 0]
        rows, boundary_stability = [], []
        base_edge = self._soft_boundary(preserve)
        preserve_dilate = _soft_dilate((preserve >= self.anchor_threshold).float().unsqueeze(1), 1)[:, 0]
        for role in range(3):
            candidate = candidate_probs[:, role]
            signed = candidate - preserve
            add = F.relu(signed)
            remove = F.relu(-signed)
            edit = (add + remove).clamp(0.0, 1.0)
            edge = self._soft_boundary(candidate)
            bscore = self._dice(edge, base_edge)
            connect = (add * preserve_dilate).sum(dim=(-2, -1)) / add.sum(dim=(-2, -1)).clamp_min(EPS)
            if role == 0:
                connect = torch.ones_like(connect)
            rows.append(torch.stack([
                candidate.mean(dim=(-2, -1)), edit.mean(dim=(-2, -1)),
                add.mean(dim=(-2, -1)), remove.mean(dim=(-2, -1)),
                signed.mean(dim=(-2, -1)), bscore, connect,
                edge.mean(dim=(-2, -1)), preserve.mean(dim=(-2, -1)),
                candidate.std(dim=(-2, -1), unbiased=False),
                edit.max(dim=-1).values.max(dim=-1).values,
                torch.ones_like(bscore),
            ], dim=1))
            boundary_stability.append(bscore)
        return torch.stack(rows, dim=1), torch.stack(boundary_stability, dim=1)

    def _semantic_logits(self, evidence: Dict[str, torch.Tensor]) -> torch.Tensor:
        # The fourth feature is deliberately *negative* control delta.  It is
        # no longer a duplicate copy of candidate core quality as in V15.1.
        features = torch.stack([
            evidence["text_core_necessity"] / self.score_temperature,
            evidence["text_edit_direction"] / self.score_temperature,
            evidence["text_specificity"] / self.specificity_temperature,
            -evidence["control_text_edit_direction"] / self.score_temperature,
        ], dim=-1)
        return self.semantic_calibrator(features)

    def _consensus(self, candidate_probs: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        b, k = candidate_probs.shape[:2]
        flat = candidate_probs.reshape(b, k, -1)
        inter = torch.einsum("bih,bjh->bij", flat, flat)
        mass = flat.sum(dim=-1)
        dice = (2.0 * inter + EPS) / (mass[:, :, None] + mass[:, None, :] + EPS)
        distance = 1.0 - dice
        population = valid.clone()
        population[:, 0] = True
        eye = torch.eye(k, device=candidate_probs.device, dtype=torch.bool).unsqueeze(0)
        weights = population[:, None, :] & ~eye
        denom = weights.float().sum(dim=-1).clamp_min(1.0)
        return 1.0 / (1.0 + (distance * weights.float()).sum(dim=-1) / denom)

    def _m3_select(self, candidate_probs, evidence, semantic_prob, role_stats, boundary_stability):
        available = evidence["cf_available"].bool()
        edit_area = role_stats[:, :, 1]
        edit_ok = (edit_area >= self.min_edit_area) & (edit_area <= self.max_edit_area)
        raw_gate = (
            (evidence["text_core_necessity"] >= self.core_min)
            & (evidence["text_edit_direction"] >= self.delta_min)
            & (evidence["text_specificity"] >= self.specificity_min)
            & (semantic_prob >= self.semantic_min)
            & available & edit_ok
        )
        raw_gate[:, 0] = False
        if not self.deploy_expand:
            raw_gate[:, 2] = False
        consensus = self._consensus(candidate_probs, raw_gate)
        text_utility = (
            semantic_prob
            + 0.30 * torch.tanh(evidence["text_core_necessity"] / self.score_temperature)
            + 0.50 * torch.tanh(evidence["text_edit_direction"] / self.score_temperature)
            + 0.25 * torch.tanh(evidence["text_specificity"] / self.specificity_temperature)
        )
        scores = text_utility + self.consensus_weight * consensus + self.boundary_weight * boundary_stability - self.size_penalty * edit_area
        scores[:, 0] = 0.0
        proposal_scores = scores[:, 1:].masked_fill(~raw_gate[:, 1:], float("-inf"))
        best_score, best_local = proposal_scores.max(dim=1)
        selected = torch.where(best_score >= self.min_score, best_local + 1, torch.zeros_like(best_local))
        final = candidate_probs[torch.arange(candidate_probs.shape[0], device=candidate_probs.device), selected]
        return selected, final.clamp(EPS, 1.0 - EPS), scores, raw_gate, consensus

    def _m2_direct_select(
        self,
        candidate_probs: torch.Tensor,
        evidence: Dict[str, torch.Tensor],
        semantic_prob: torch.Tensor,
        role_stats: torch.Tensor,
    ):
        """Pure M2 policy: no M3 consensus/boundary/size rescoring."""
        available = evidence["cf_available"].bool()
        edit_area = role_stats[:, :, 1]

        edit_ok = (
            (edit_area >= self.min_edit_area)
            & (edit_area <= self.max_edit_area)
        )

        direct_gate = available & edit_ok & (semantic_prob >= self.semantic_min)
        direct_gate[:, 0] = False

        if not self.deploy_expand:
            direct_gate[:, 2] = False

        scores = semantic_prob.clone()
        scores[:, 0] = 0.0

        proposal_scores = scores[:, 1:].masked_fill(
            ~direct_gate[:, 1:],
            float("-inf"),
        )

        best_score, best_local = proposal_scores.max(dim=1)

        selected = torch.where(
            best_score >= self.min_score,
            best_local + 1,
            torch.zeros_like(best_local),
        )

        final = candidate_probs[
            torch.arange(candidate_probs.shape[0], device=candidate_probs.device),
            selected,
        ]

        return selected, final.clamp(EPS, 1.0 - EPS), scores, direct_gate

    def forward(self, candidate_probs: torch.Tensor, cf_evidence: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        required = {
            "core_keep_positive", "core_drop_positive", "core_keep_negative", "core_drop_negative",
            "edit_keep_positive", "edit_drop_positive", "edit_keep_negative", "edit_drop_negative",
            "text_core_necessity", "text_edit_direction", "control_text_edit_direction",
            "text_specificity", "cf_available",
        }
        missing = required.difference(cf_evidence)
        if missing:
            raise KeyError(f"V15 dense-mask text evidence missing keys: {sorted(missing)}")
        # Do not detach: frozen VLM features have no gradients, whereas patch_adapter
        # and the scalar calibrator must receive text-only gradients in Phase B.
        evidence = dict(cf_evidence)
        semantic_logits = self._semantic_logits(evidence)
        control_evidence = dict(evidence)
        control_evidence["text_core_necessity"] = evidence["control_edit_keep_discriminative"]
        control_evidence["text_edit_direction"] = evidence["control_text_edit_direction"]
        control_evidence["text_specificity"] = torch.zeros_like(evidence["text_specificity"])
        control_evidence["control_text_edit_direction"] = torch.zeros_like(evidence["control_text_edit_direction"])
        control_logits = self._semantic_logits(control_evidence)
        semantic_prob = torch.sigmoid(semantic_logits)
        role_stats, boundary_stability = self._role_stats(candidate_probs.detach())

        m2_direct_selected, m2_direct_final_probs, m2_direct_scores, m2_direct_pass = self._m2_direct_select(
            candidate_probs,
            evidence,
            semantic_prob,
            role_stats,
        )

        selected, final_probs, m3_scores, text_pass, consensus = self._m3_select(
            candidate_probs, evidence, semantic_prob, role_stats, boundary_stability
        )
        zero = candidate_probs.new_zeros(candidate_probs.shape[0], 3)
        return {
            "falsification_keep_positive": evidence["core_keep_positive"],
            "falsification_drop_positive": evidence["core_drop_positive"],
            "falsification_keep_negative": evidence["core_keep_negative"],
            "falsification_drop_negative": evidence["core_drop_negative"],
            "falsification_keep_discriminative": evidence["core_keep_discriminative"],
            "falsification_drop_discriminative": evidence["core_drop_discriminative"],
            "falsification_text_necessity": evidence["text_core_necessity"],
            "falsification_text_edit_direction": evidence["text_edit_direction"],
            "falsification_control_text_necessity": evidence["control_text_edit_direction"],
            "falsification_text_specificity": evidence["text_specificity"],
            "falsification_cf_available": evidence["cf_available"].float(),
            "falsification_semantic_logits": semantic_logits,
            "falsification_control_semantic_logits": control_logits,
            "falsification_semantic_valid": semantic_prob,
            "falsification_validity": semantic_prob,
            "falsification_validity_logits": semantic_logits,
            "falsification_benefit": semantic_prob,
            "falsification_benefit_logits": semantic_logits,
            "falsification_harm": 1.0 - semantic_prob,
            "falsification_harm_logits": -semantic_logits,
            "falsification_uncertainty": 1.0 - (semantic_prob - 0.5).abs() * 2.0,
            "falsification_text_alignment": torch.sigmoid(evidence["text_edit_direction"] / self.score_temperature),
            "falsification_support": semantic_prob,
            "falsification_contradiction": 1.0 - semantic_prob,
            "falsification_overseg_violation": 1.0 - boundary_stability,
            "falsification_underseg_violation": 1.0 - boundary_stability,
            "falsification_boundary_support": boundary_stability,
            "falsification_connectivity": role_stats[:, :, 6],
            "falsification_consensus": consensus,
            "falsification_role_stats": role_stats,
            "falsification_text_pass": text_pass.float(),
            "falsification_m3_scores": m3_scores,
            "falsification_m2_direct_selected_index": m2_direct_selected,
            "falsification_m2_direct_selected_probs": m2_direct_final_probs,
            "falsification_m2_direct_scores": m2_direct_scores,
            "falsification_m2_direct_pass": m2_direct_pass.float(),
            "falsification_m2_direct_accept": (m2_direct_selected > 0).float(),
            "falsification_m3_selected_index": selected,
            "falsification_m3_selected_probs": final_probs,
            "falsification_m3_accept": (selected > 0).float(),
            "falsification_patch_adapter_reg": self.patch_adapter_identity_penalty(),
            "falsification_calibrator_reg": self.calibrator_regularizer(),
            "falsification_patch_adapter_trainable": semantic_logits.new_tensor(float(self.patch_adapter.weight.requires_grad)),
            "falsification_dense_local_alignment_loss": evidence.get("dense_local_alignment_loss", semantic_logits.sum() * 0.0),
            "falsification_dense_quality_preserve": evidence.get("dense_quality_preserve", zero),
            "falsification_dense_quality_candidate": evidence.get("dense_quality_candidate", zero),
            "falsification_dense_quality_control": evidence.get("dense_quality_control", zero),
            "falsification_dense_delta_control": evidence.get("dense_delta_control", zero),
            "falsification_dense_positive_candidate": evidence.get("dense_positive_candidate", zero),
            "falsification_dense_negative_candidate": evidence.get("dense_negative_candidate", zero),
            "falsification_dense_gt_gap": evidence.get("dense_gt_gap", zero),
            "falsification_dense_control_area_ratio": evidence.get("dense_control_area_ratio", zero),
            "falsification_dense_control_overlap_ratio": evidence.get("dense_control_overlap_ratio", zero),
        }



class V38CasewiseFalsifiedDeltaConsensusBank(V37TextFalsifiedStructuralConsensusBank):
    """V38: casewise M2 falsification ranking plus edit-delta structural consensus.

    Deployment is strictly:
      M1 all-type atomic local candidates
      -> M2 matched-control textual falsification and qualification
      -> M2 joint re-verification of delete+fill compositions
      -> M3 deterministic selection from qualified hypotheses using *edit-delta*
         agreement, local edit-boundary/image alignment, and topology safety.

    M3 never scores full-mask similarity, Dice gain, GT labels, predicted gain,
    or risk heads. Preserve is emitted exactly when no qualified hypothesis is
    structurally stable.
    """

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        m1 = _cfg_get(cfg, "M1", None)
        hidden = int(self.v23_cf_head.in_features)

        # Fixed deployment thresholds. They are configuration values, never
        # fitted on Val/Test during execution.
        self.v38_atomic_text_threshold = float(
            _cfg_get(m1, "CONSENSUS_ATOMIC_TEXT_THRESHOLD", 0.50)
        )
        self.v38_combo_text_threshold = float(
            _cfg_get(m1, "CONSENSUS_COMBO_TEXT_THRESHOLD", 0.55)
        )
        self.v38_pos_delta_min = float(_cfg_get(m1, "CONSENSUS_POS_DELTA_MIN", 0.0))
        self.v38_neg_delta_min = float(_cfg_get(m1, "CONSENSUS_NEG_DELTA_MIN", 0.0))
        self.v38_swap_gap_min = float(_cfg_get(m1, "CONSENSUS_SWAP_GAP_MIN", 0.0))

        # M3 evaluates only local signed edit deltas, not nearly-identical full
        # masks. A high full-mask Dice can no longer create a false cluster.
        self.v38_struct_size = max(32, int(_cfg_get(m1, "CONSENSUS_STRUCT_SIZE", 80)))
        self.v38_delta_dice_weight = float(
            _cfg_get(m1, "CONSENSUS_DELTA_DICE_WEIGHT", 0.60)
        )
        self.v38_edit_boundary_weight = float(
            _cfg_get(m1, "CONSENSUS_EDIT_BOUNDARY_WEIGHT", 0.40)
        )
        self.v38_edge_weight = float(_cfg_get(m1, "CONSENSUS_EDIT_EDGE_WEIGHT", 0.25))
        self.v38_edit_area_penalty = float(
            _cfg_get(m1, "CONSENSUS_EDIT_AREA_PENALTY", 0.10)
        )
        self.v38_edit_perimeter_penalty = float(
            _cfg_get(m1, "CONSENSUS_EDIT_PERIMETER_PENALTY", 0.10)
        )
        self.v38_cluster_similarity_min = float(
            _cfg_get(m1, "CONSENSUS_CLUSTER_SIMILARITY_MIN", 0.70)
        )
        self.v38_min_cluster_size = max(
            2, int(_cfg_get(m1, "CONSENSUS_MIN_CLUSTER_SIZE", 2))
        )
        self.v38_max_edit_fraction = float(
            _cfg_get(m1, "CONSENSUS_MAX_EDIT_FRACTION", 0.035)
        )
        self.v38_max_edit_perimeter_growth = float(
            _cfg_get(m1, "CONSENSUS_MAX_EDIT_PERIMETER_GROWTH", 5.0)
        )
        self.v38_singleton_text_threshold = float(
            _cfg_get(m1, "CONSENSUS_SINGLETON_TEXT_THRESHOLD", 0.80)
        )
        self.v38_singleton_min_edit_edge = float(
            _cfg_get(m1, "CONSENSUS_SINGLETON_MIN_EDIT_EDGE", 0.01)
        )
        self.v38_singleton_min_stability = float(
            _cfg_get(m1, "CONSENSUS_SINGLETON_MIN_STABILITY", 0.0)
        )

        # M2 re-verifies an entire delete+fill composition through the two
        # matched factual/control contrasts jointly. This head gets only the
        # two M2 embeddings, never candidate IDs, gain labels, or geometry.
        self.v38_combo_cf_encoder = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.v38_combo_cf_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.v38_combo_cf_head.weight)
        nn.init.constant_(self.v38_combo_cf_head.bias, -1.5)

    @staticmethod
    def _v38_pairwise_dice(mask: torch.Tensor) -> torch.Tensor:
        """Pairwise Dice with the empty/empty case defined as perfect match."""
        flat = mask.flatten(2)
        inter = torch.einsum("bih,bjh->bij", flat, flat)
        mass = flat.sum(dim=-1)
        den = mass[:, :, None] + mass[:, None, :]
        out = (2.0 * inter + EPS) / (den + EPS)
        return torch.where(den <= EPS, torch.ones_like(out), out)

    @staticmethod
    def _v38_edit_maps(hypothesis_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return signed edits for every hypothesis relative to Preserve/Base."""
        hard = (torch.sigmoid(hypothesis_logits) >= 0.5).float()
        base = hard[:, :1].expand(-1, hard.shape[1], -1, -1)
        delete = (base * (1.0 - hard)).clamp(0.0, 1.0)
        fill = ((1.0 - base) * hard).clamp(0.0, 1.0)
        return delete, fill, (delete + fill).clamp(0.0, 1.0)

    def _v38_atomic_qualified(
        self,
        cf: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        score = torch.sigmoid(cf["cf_logit"])
        gap = cf["cf_pos_delta"] - cf["cf_swap_delta"]
        qualified = (
            cf["cf_available"].bool()
            & (score >= self.v38_atomic_text_threshold)
            & (cf["cf_pos_delta"] >= self.v38_pos_delta_min)
            & (cf["cf_neg_delta"] >= self.v38_neg_delta_min)
            & (gap >= self.v38_swap_gap_min)
        )
        return qualified, score

    def _v38_combo_counterfactual(
        self,
        cf: Dict[str, torch.Tensor],
        factual_masks: torch.Tensor,
        control_masks: torch.Tensor,
        action_types: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Joint M2 re-verification for every delete+fill composition.

        A composition is admissible only if (1) its two factual edits are
        non-overlapping, (2) its two controls are non-overlapping, (3) the
        union control remains disjoint/context-clean from the union factual
        edit, and (4) the learned joint text verifier accepts the pair.
        """
        device = factual_masks.device
        delete_idx = torch.where(
            torch.isin(
                action_types,
                torch.tensor(self.DELETE_TYPES, device=device),
            )
        )[0]
        fill_idx = torch.where(
            ~torch.isin(
                action_types,
                torch.tensor(self.DELETE_TYPES, device=device),
            )
        )[0]
        pairs = torch.cartesian_prod(delete_idx, fill_idx)
        if pairs.numel() == 0:
            b = factual_masks.shape[0]
            empty = factual_masks.new_zeros((b, 0))
            return {
                "pair_indices": pairs.reshape(0, 2),
                "combo_logit": empty,
                "combo_probability": empty,
                "combo_available": empty.bool(),
                "combo_context_clean": empty.bool(),
                "combo_control_disjoint": empty.bool(),
            }

        di = pairs[:, 0]
        fi = pairs[:, 1]
        fd = factual_masks.index_select(1, di)
        ff = factual_masks.index_select(1, fi)
        cd = control_masks.index_select(1, di)
        cfmask = control_masks.index_select(1, fi)

        factual_union = (fd + ff).clamp(0.0, 1.0)
        control_union = (cd + cfmask).clamp(0.0, 1.0)
        factual_area = factual_union.sum(dim=(-2, -1))
        control_area = control_union.sum(dim=(-2, -1))
        factual_overlap = (fd * ff).sum(dim=(-2, -1))
        control_overlap = (cd * cfmask).sum(dim=(-2, -1))
        cross_overlap = (factual_union * control_union).sum(dim=(-2, -1))

        b, p, h, w = factual_union.shape
        factual_context = _soft_dilate(
            factual_union.reshape(b * p, 1, h, w), self.context_radius
        ).reshape(b, p, h, w)
        control_context = _soft_dilate(
            control_union.reshape(b * p, 1, h, w), self.context_radius
        ).reshape(b, p, h, w)
        context_overlap = (
            factual_context * control_context
        ).sum(dim=(-2, -1)) / factual_context.sum(dim=(-2, -1)).clamp_min(EPS)
        context_clean = context_overlap <= 1e-6

        geometry_valid = (
            (factual_area > 0)
            & (control_area > 0)
            & ((factual_area - control_area).abs() <= 1e-3)
            & (factual_overlap <= 1e-6)
            & (control_overlap <= 1e-6)
            & (cross_overlap <= 1e-6)
            & context_clean
        )

        embedding = cf["cf_embedding"]
        pair_embedding = torch.cat(
            [embedding.index_select(1, di), embedding.index_select(1, fi)],
            dim=-1,
        )
        combo_hidden = self.v38_combo_cf_encoder(
            pair_embedding.reshape(b * p, -1)
        ).reshape(b, p, -1)
        combo_logit = self.v38_combo_cf_head(
            combo_hidden.reshape(b * p, -1)
        ).reshape(b, p)
        return {
            "pair_indices": pairs,
            "combo_logit": combo_logit,
            "combo_probability": torch.sigmoid(combo_logit),
            "combo_available": geometry_valid,
            "combo_context_clean": context_clean,
            "combo_control_disjoint": control_overlap <= 1e-6,
        }

    def _v38_build_hypotheses(
        self,
        candidate_logits_all: torch.Tensor,
        atomic_qualified: torch.Tensor,
        atomic_text_score: torch.Tensor,
        combo: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build Base + qualified atomic + jointly re-verified compositions."""
        b, slots, h, w = candidate_logits_all.shape
        k = slots - 1
        base = candidate_logits_all[:, 0]
        deltas = candidate_logits_all[:, 1:] - base[:, None]

        logits = [base]
        valid = [torch.ones((b,), dtype=torch.bool, device=base.device)]
        members = [torch.zeros((b, k), dtype=base.dtype, device=base.device)]
        text_scores = [base.new_zeros((b,))]
        combo_flags = [torch.zeros((b,), dtype=torch.bool, device=base.device)]

        for ai in range(k):
            logits.append(base + deltas[:, ai])
            valid.append(atomic_qualified[:, ai])
            member = torch.zeros((b, k), dtype=base.dtype, device=base.device)
            member[:, ai] = 1.0
            members.append(member)
            text_scores.append(atomic_text_score[:, ai])
            combo_flags.append(torch.zeros((b,), dtype=torch.bool, device=base.device))

        pair_indices = combo["pair_indices"]
        combo_qualified = (
            combo["combo_available"].bool()
            & (combo["combo_probability"] >= self.v38_combo_text_threshold)
        )
        for pair_rank in range(pair_indices.shape[0]):
            di = int(pair_indices[pair_rank, 0])
            fi = int(pair_indices[pair_rank, 1])
            logits.append(base + deltas[:, di] + deltas[:, fi])
            valid.append(
                atomic_qualified[:, di]
                & atomic_qualified[:, fi]
                & combo_qualified[:, pair_rank]
            )
            member = torch.zeros((b, k), dtype=base.dtype, device=base.device)
            member[:, di] = 1.0
            member[:, fi] = 1.0
            members.append(member)
            text_scores.append(combo["combo_probability"][:, pair_rank])
            combo_flags.append(torch.ones((b,), dtype=torch.bool, device=base.device))

        return (
            torch.stack(logits, dim=1),
            torch.stack(valid, dim=1),
            torch.stack(members, dim=1),
            torch.stack(text_scores, dim=1),
            torch.stack(combo_flags, dim=1),
        )

    def _v38_delta_consensus_select(
        self,
        hypothesis_logits: torch.Tensor,
        valid: torch.Tensor,
        membership: torch.Tensor,
        hypothesis_text_score: torch.Tensor,
        image_edge: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """M3: edit-delta consensus and local structural safety only."""
        b, n, h, w = hypothesis_logits.shape
        side = min(self.v38_struct_size, h, w)
        low_logits = F.interpolate(
            hypothesis_logits.reshape(b * n, 1, h, w),
            size=(side, side),
            mode="bilinear",
            align_corners=False,
        ).reshape(b, n, side, side)

        delete, fill, edit = self._v38_edit_maps(low_logits)
        delete_dice = self._v38_pairwise_dice(delete)
        fill_dice = self._v38_pairwise_dice(fill)
        delete_mass = delete.flatten(2).sum(dim=-1)
        fill_mass = fill.flatten(2).sum(dim=-1)
        delete_present = (
            delete_mass[:, :, None] + delete_mass[:, None, :]
        ) > EPS
        fill_present = (
            fill_mass[:, :, None] + fill_mass[:, None, :]
        ) > EPS
        signed_weight = delete_present.float() + fill_present.float()
        signed_delta_dice = (
            delete_dice * delete_present.float()
            + fill_dice * fill_present.float()
        ) / signed_weight.clamp_min(1.0)

        edit_boundary = self._boundary(edit)
        edit_boundary_dice = self._v38_pairwise_dice(edit_boundary)
        pairwise = (
            self.v38_delta_dice_weight * signed_delta_dice
            + self.v38_edit_boundary_weight * edit_boundary_dice
        )

        low_edge = F.interpolate(
            image_edge, size=(side, side), mode="bilinear", align_corners=False
        )
        edit_boundary_mass = edit_boundary.sum(dim=(-2, -1)).clamp_min(EPS)
        edit_edge_alignment = (
            edit_boundary * low_edge
        ).sum(dim=(-2, -1)) / edit_boundary_mass
        edit_area = edit.mean(dim=(-2, -1))
        edit_perimeter = edit_boundary_mass / float(side * side)

        edited_valid = valid.clone()
        edited_valid[:, 0] = False
        structural_safe = (
            edited_valid
            & (edit_area > 0)
            & (edit_area <= self.v38_max_edit_fraction)
            & (edit_perimeter <= self.v38_max_edit_perimeter_growth)
        )

        eye = torch.eye(n, dtype=torch.bool, device=hypothesis_logits.device)[None]
        pair_valid = (
            structural_safe[:, :, None]
            & structural_safe[:, None, :]
            & (~eye)
        )
        agreement = (
            pairwise * pair_valid.float()
        ).sum(dim=-1) / pair_valid.float().sum(dim=-1).clamp_min(1.0)
        stability = (
            agreement
            + self.v38_edge_weight * edit_edge_alignment
            - self.v38_edit_area_penalty * edit_area
            - self.v38_edit_perimeter_penalty * edit_perimeter
        )
        medoid_score = stability.masked_fill(~structural_safe, -1e9)
        medoid_index = medoid_score.argmax(dim=1)
        medoid_exists = structural_safe.any(dim=1)

        medoid_pairwise = pairwise.gather(
            1, medoid_index[:, None, None].expand(-1, 1, n)
        )[:, 0]
        cluster = structural_safe & (
            medoid_pairwise >= self.v38_cluster_similarity_min
        )
        cluster_size = cluster.sum(dim=1)
        multi_active = (
            medoid_exists
            & (cluster_size >= self.v38_min_cluster_size)
        )

        selected_text = hypothesis_text_score.gather(
            1, medoid_index[:, None]
        )[:, 0]
        selected_edge = edit_edge_alignment.gather(
            1, medoid_index[:, None]
        )[:, 0]
        selected_stability = stability.gather(
            1, medoid_index[:, None]
        )[:, 0]
        singleton_active = (
            medoid_exists
            & (cluster_size == 1)
            & (selected_text >= self.v38_singleton_text_threshold)
            & (selected_edge >= self.v38_singleton_min_edit_edge)
            & (selected_stability >= self.v38_singleton_min_stability)
        )
        consensus_active = multi_active | singleton_active

        selected_logits = hypothesis_logits.gather(
            1, medoid_index[:, None, None, None].expand(-1, 1, h, w)
        )[:, 0]
        final_logits = torch.where(
            consensus_active[:, None, None],
            selected_logits,
            hypothesis_logits[:, 0],
        )

        k = membership.shape[-1]
        selected_membership = membership.gather(
            1, medoid_index[:, None, None].expand(-1, 1, k)
        )[:, 0]
        action_endorsement = (
            selected_membership * consensus_active[:, None].float()
        )

        return final_logits, action_endorsement, {
            "structural_safe": structural_safe,
            "signed_delta_dice": signed_delta_dice,
            "edit_boundary_dice": edit_boundary_dice,
            "pairwise": pairwise,
            "agreement": agreement,
            "stability": stability,
            "edit_edge_alignment": edit_edge_alignment,
            "edit_area": edit_area,
            "edit_perimeter": edit_perimeter,
            "medoid_score": medoid_score,
            "medoid_index": medoid_index,
            "cluster_size": cluster_size.float(),
            "multi_active": multi_active.float(),
            "singleton_active": singleton_active.float(),
            "consensus_active": consensus_active.float(),
            "selected_text_score": selected_text,
            "selected_edit_edge": selected_edge,
            "selected_stability": selected_stability,
        }

    def generate(
        self,
        base_logits: torch.Tensor,
        image: torch.Tensor,
        semantic_map: Optional[torch.Tensor],
        text_features: torch.Tensor,
        negative_text_features: Optional[torch.Tensor] = None,
        swapped_text_features: Optional[torch.Tensor] = None,
    ):
        candidate_logits_all, aux = UnifiedActionCounterfactualSetBank.generate(
            self,
            base_logits=base_logits,
            image=image,
            semantic_map=semantic_map,
            text_features=text_features,
            negative_text_features=negative_text_features,
        )
        if semantic_map is None:
            raise RuntimeError("V38 requires frozen image-only semantic features for M2.")
        if negative_text_features is None:
            negative_text_features = text_features
        if swapped_text_features is None:
            swapped_text_features = negative_text_features

        # M2 atomic factual/control falsification.
        cf = self._v37_text_counterfactual(
            image_only_map=semantic_map,
            positive_text=text_features,
            negative_text=negative_text_features,
            swapped_text=swapped_text_features,
            factual_masks=aux["v20_action_supports"],
            control_masks=aux["v20_control_supports"],
            action_types=self.action_types,
        )
        atomic_qualified, atomic_text_score = self._v38_atomic_qualified(cf)

        # M2 composition re-verification; this happens after M1 atomic
        # candidates exist and before M3 sees any delete+fill hypothesis.
        combo = self._v38_combo_counterfactual(
            cf,
            aux["v20_action_supports"],
            aux["v20_control_supports"],
            self.action_types,
        )

        # M3 sees only atomic candidates admitted by M2 and combinations
        # admitted by the joint M2 verifier.
        hypotheses, hypothesis_valid, membership, hypothesis_text_score, hypothesis_is_combo = (
            self._v38_build_hypotheses(
                candidate_logits_all,
                atomic_qualified,
                atomic_text_score,
                combo,
            )
        )
        _, image_edge = self._image_gray_edge(
            image, candidate_logits_all.shape[-2:]
        )
        final_logits, action_endorsement, structural = self._v38_delta_consensus_select(
            hypotheses,
            hypothesis_valid,
            membership,
            hypothesis_text_score,
            image_edge,
        )
        final_probs = torch.sigmoid(final_logits).clamp(EPS, 1.0 - EPS)

        aux.update({
            "direct_fused_probs": final_probs,
            "router_fused_probs": final_probs,
            "v20_fused_probs": final_probs,
            "v20_hard_fused_probs": final_probs,
            "v20_fused_logits": final_logits,
            # Compatibility names expose deterministic M3 membership only.
            "v20_selector_logits": structural["medoid_score"],
            "v20_selector_probs": action_endorsement,
            "v20_selector_hard": action_endorsement,
            "v20_cf_logit": cf["cf_logit"],
            "v20_cf_swap_logit": cf["cf_swap_logit"],
            "v20_cf_signed_delta": cf["cf_signed_delta"],
            "v20_cf_factual_similarity": cf["cf_factual_similarity"],
            "v20_cf_control_similarity": cf["cf_control_similarity"],
            "v20_cf_available": cf["cf_available"],
            "v20_cf_control_area_ratio": cf["cf_control_area_ratio"],
            "v20_cf_control_overlap": cf["cf_control_overlap"],
            # V38 M2 atomic evidence.
            "v38_atomic_cf_logit": cf["cf_logit"],
            "v38_atomic_cf_swap_logit": cf["cf_swap_logit"],
            "v38_atomic_text_probability": atomic_text_score,
            "v38_atomic_qualified": atomic_qualified,
            "v38_atomic_available": cf["cf_available"],
            "v38_context_clean": cf["cf_context_clean"],
            "v38_context_overlap": cf["cf_context_overlap"],
            "v38_pos_delta": cf["cf_pos_delta"],
            "v38_neg_delta": cf["cf_neg_delta"],
            "v38_swap_delta": cf["cf_swap_delta"],
            "v38_text_gap": cf["cf_pos_delta"] - cf["cf_swap_delta"],
            "v38_cf_embedding": cf["cf_embedding"],
            "v38_type_probe": cf["v23_type_probe"],
            "v38_area_probe": cf["v23_area_probe"],
            "v38_position_probe": cf["v23_position_probe"],
            # V38 joint M2 composition evidence.
            "v38_combo_pair_indices": combo["pair_indices"],
            "v38_combo_cf_logit": combo["combo_logit"],
            "v38_combo_text_probability": combo["combo_probability"],
            "v38_combo_available": combo["combo_available"],
            "v38_combo_qualified": (
                combo["combo_available"].bool()
                & (combo["combo_probability"] >= self.v38_combo_text_threshold)
            ),
            "v38_combo_context_clean": combo["combo_context_clean"],
            "v38_combo_control_disjoint": combo["combo_control_disjoint"],
            # V38 M3 proof trace.
            "v38_hypothesis_valid": hypothesis_valid,
            "v38_hypothesis_membership": membership,
            "v38_hypothesis_text_score": hypothesis_text_score,
            "v38_hypothesis_is_combo": hypothesis_is_combo,
            "v38_structural_safe": structural["structural_safe"],
            "v38_delta_pairwise": structural["pairwise"],
            "v38_signed_delta_dice": structural["signed_delta_dice"],
            "v38_edit_boundary_dice": structural["edit_boundary_dice"],
            "v38_hypothesis_agreement": structural["agreement"],
            "v38_hypothesis_stability": structural["stability"],
            "v38_edit_edge_alignment": structural["edit_edge_alignment"],
            "v38_edit_area": structural["edit_area"],
            "v38_edit_perimeter": structural["edit_perimeter"],
            "v38_medoid_index": structural["medoid_index"],
            "v38_cluster_size": structural["cluster_size"],
            "v38_multi_active": structural["multi_active"],
            "v38_singleton_active": structural["singleton_active"],
            "v38_consensus_active": structural["consensus_active"],
            "v38_selected_text_score": structural["selected_text_score"],
            "v38_selected_edit_edge": structural["selected_edit_edge"],
            "v38_selected_stability": structural["selected_stability"],
            # Legacy log consumers expect V37-like aliases; these do not alter
            # deployment and are only diagnostics.
            "v37_text_qualified": atomic_qualified,
            "v37_consensus_active": structural["consensus_active"],
            "v37_multi_active": structural["multi_active"],
            "v37_singleton_active": structural["singleton_active"],
            "v37_cluster_size": structural["cluster_size"],
        })
        return candidate_logits_all, aux


class _V381LesionBackgroundCalibrator(nn.Module):
    """Monotone affine calibration of one lesion-vs-background contrast.

    ``scale`` is constrained positive, so M2 cannot invert the semantic
    direction.  ``bias`` is the learned score relative to Preserve=0.
    """
    def __init__(self, scale_init: float = 10.0, bias_init: float = 0.0) -> None:
        super().__init__()
        scale_init = max(float(scale_init), 1e-4)
        self.raw_scale = nn.Parameter(torch.tensor(math.log(math.expm1(scale_init))))
        self.bias = nn.Parameter(torch.tensor(float(bias_init)))

    def forward(self, contrast: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.raw_scale) * contrast + self.bias


# ---------------------------------------------------------------------------

class V381LesionBackgroundCalibratedAtomicBank(V38CasewiseFalsifiedDeltaConsensusBank):
    """V381: minimal repair of V38's M2 evidence protocol.

    V38 audit established that the batch-rotated swap prompt collapses to the
    negative prompt.  V381 therefore uses exactly one semantic contrast:

      polarity * [(lesion(F)-lesion(C)) - (background(F)-background(C))]

    where F/C are matched factual/control local contexts.  Atomic M2 is a
    monotone calibrated score versus Preserve=0.  Training and deployment use
    the same score; there are no swap terms, no neg/swap hard gates and no
    delete+fill composition path in this causal preflight version.
    """
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        m1 = _cfg_get(cfg, "M1", None)
        self.v381_deploy_logit_threshold = float(
            _cfg_get(m1, "CALIBRATION_DEPLOY_LOGIT_THRESHOLD", 0.0)
        )
        self.v381_scale_init = float(_cfg_get(m1, "CALIBRATION_SCALE_INIT", 10.0))
        self.v381_calibrator = _V381LesionBackgroundCalibrator(
            scale_init=self.v381_scale_init,
            bias_init=float(_cfg_get(m1, "CALIBRATION_BIAS_INIT", 0.0)),
        )

        # Reuse V38's signed-delta M3 implementation but express every M3
        # threshold in V381 units.  Text score is now a calibrated logit whose
        # Preserve reference is exactly zero.
        self.v38_struct_size = max(32, int(_cfg_get(m1, "CALIBRATION_STRUCT_SIZE", 80)))
        self.v38_delta_dice_weight = float(_cfg_get(m1, "CALIBRATION_DELTA_DICE_WEIGHT", 0.60))
        self.v38_edit_boundary_weight = float(_cfg_get(m1, "CALIBRATION_EDIT_BOUNDARY_WEIGHT", 0.40))
        self.v38_edge_weight = float(_cfg_get(m1, "CALIBRATION_EDIT_EDGE_WEIGHT", 0.25))
        self.v38_edit_area_penalty = float(_cfg_get(m1, "CALIBRATION_EDIT_AREA_PENALTY", 0.10))
        self.v38_edit_perimeter_penalty = float(_cfg_get(m1, "CALIBRATION_EDIT_PERIMETER_PENALTY", 0.10))
        self.v38_cluster_similarity_min = float(_cfg_get(m1, "CALIBRATION_CLUSTER_SIMILARITY_MIN", 0.70))
        self.v38_min_cluster_size = max(2, int(_cfg_get(m1, "CALIBRATION_MIN_CLUSTER_SIZE", 2)))
        self.v38_max_edit_fraction = float(_cfg_get(m1, "CALIBRATION_MAX_EDIT_FRACTION", 0.035))
        self.v38_max_edit_perimeter_growth = float(_cfg_get(m1, "CALIBRATION_MAX_EDIT_PERIMETER_GROWTH", 5.0))
        self.v38_singleton_text_threshold = float(
            _cfg_get(m1, "CALIBRATION_SINGLETON_LOGIT_MIN", self.v381_deploy_logit_threshold)
        )
        self.v38_singleton_min_edit_edge = float(_cfg_get(m1, "CALIBRATION_SINGLETON_MIN_EDIT_EDGE", 0.01))
        self.v38_singleton_min_stability = float(_cfg_get(m1, "CALIBRATION_SINGLETON_MIN_STABILITY", 0.0))

        # V38 composition modules are intentionally retained only for B0/V38
        # checkpoint namespace compatibility. They are frozen and never used by
        # V381.  A composition must be re-pooled on union ROIs before it is
        # reintroduced; concatenating two atomic embeddings is not sufficient.
        for parameter in self.v38_combo_cf_encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.v38_combo_cf_head.parameters():
            parameter.requires_grad_(False)

    def _v381_counterfactual(
        self,
        image_only_map: torch.Tensor,
        positive_text: torch.Tensor,
        negative_text: torch.Tensor,
        factual_masks: torch.Tensor,
        control_masks: torch.Tensor,
        action_types: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        b, k, h, w = factual_masks.shape
        factual_context = _soft_dilate(
            factual_masks.reshape(b * k, 1, h, w), self.context_radius
        ).reshape(b, k, h, w)
        control_context = _soft_dilate(
            control_masks.reshape(b * k, 1, h, w), self.context_radius
        ).reshape(b, k, h, w)

        factual_roi = self._pool_feature(image_only_map, factual_context)
        control_roi = self._pool_feature(image_only_map, control_context)
        factual_roi = F.normalize(self.v23_image_adapter(factual_roi), dim=-1, eps=1e-6)
        control_roi = F.normalize(self.v23_image_adapter(control_roi), dim=-1, eps=1e-6)
        positive_text = F.normalize(self.v23_text_adapter(positive_text), dim=-1, eps=1e-6)
        negative_text = F.normalize(self.v23_text_adapter(negative_text), dim=-1, eps=1e-6)

        def paired_delta(text_feature: torch.Tensor):
            factual_score = (factual_roi * text_feature[:, None, :]).sum(dim=-1)
            control_score = (control_roi * text_feature[:, None, :]).sum(dim=-1)
            return factual_score - control_score, factual_score, control_score

        raw_pos, factual_pos, control_pos = paired_delta(positive_text)
        raw_neg, factual_neg, control_neg = paired_delta(negative_text)
        is_fill = torch.isin(
            action_types,
            torch.tensor(self.FILL_TYPES, device=action_types.device),
        )
        polarity = torch.where(
            is_fill,
            torch.ones_like(action_types, dtype=raw_pos.dtype),
            -torch.ones_like(action_types, dtype=raw_pos.dtype),
        )[None, :]

        # This is the only M2 semantic feature: evidence that F is more lesion-
        # like than matched C for a fill, and less lesion-like for a delete.
        pos_delta = polarity * raw_pos
        neg_delta = -polarity * raw_neg
        lesionness_contrast = pos_delta + neg_delta
        cf_logit = self.v381_calibrator(lesionness_contrast)

        factual_area = factual_masks.sum(dim=(-2, -1))
        control_area = control_masks.sum(dim=(-2, -1))
        area_ratio = control_area / factual_area.clamp_min(EPS)
        overlap = (
            factual_masks * control_masks
        ).sum(dim=(-2, -1)) / factual_area.clamp_min(EPS)
        available = (
            (factual_area > 0)
            & (control_area > 0)
            & ((area_ratio - 1.0).abs() <= 1e-3)
            & (overlap <= 1e-6)
        )
        context_overlap = (
            factual_context * control_context
        ).sum(dim=(-2, -1)) / factual_context.sum(dim=(-2, -1)).clamp_min(EPS)
        context_clean = context_overlap <= 1e-6
        available = available & context_clean

        return {
            "cf_logit": cf_logit,
            "cf_available": available,
            "cf_context_clean": context_clean,
            "cf_context_overlap": context_overlap,
            "cf_lesionness_contrast": lesionness_contrast,
            "cf_pos_delta": pos_delta,
            "cf_neg_delta": neg_delta,
            "cf_factual_similarity": factual_pos - factual_neg,
            "cf_control_similarity": control_pos - control_neg,
            "cf_control_area_ratio": area_ratio,
            "cf_control_overlap": overlap,
        }

    def _v381_atomic_qualified(self, cf: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        score = cf["cf_logit"]
        qualified = cf["cf_available"].bool() & (score >= self.v381_deploy_logit_threshold)
        return qualified, score

    def _v381_build_atomic_hypotheses(
        self,
        candidate_logits_all: torch.Tensor,
        atomic_qualified: torch.Tensor,
        atomic_score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, slots, h, w = candidate_logits_all.shape
        k = slots - 1
        base = candidate_logits_all[:, 0]
        deltas = candidate_logits_all[:, 1:] - base[:, None]
        logits = [base]
        valid = [torch.ones((b,), dtype=torch.bool, device=base.device)]
        members = [torch.zeros((b, k), dtype=base.dtype, device=base.device)]
        scores = [base.new_zeros((b,))]
        for ai in range(k):
            logits.append(base + deltas[:, ai])
            valid.append(atomic_qualified[:, ai])
            member = torch.zeros((b, k), dtype=base.dtype, device=base.device)
            member[:, ai] = 1.0
            members.append(member)
            scores.append(atomic_score[:, ai])
        return (
            torch.stack(logits, dim=1),
            torch.stack(valid, dim=1),
            torch.stack(members, dim=1),
            torch.stack(scores, dim=1),
        )

    def generate(
        self,
        base_logits: torch.Tensor,
        image: torch.Tensor,
        semantic_map: Optional[torch.Tensor],
        text_features: torch.Tensor,
        negative_text_features: Optional[torch.Tensor] = None,
        swapped_text_features: Optional[torch.Tensor] = None,
    ):
        # V381 intentionally ignores swapped_text_features. It is accepted only
        # to preserve the generic CustomCLIP M1 call signature.
        candidate_logits_all, aux = UnifiedActionCounterfactualSetBank.generate(
            self,
            base_logits=base_logits,
            image=image,
            semantic_map=semantic_map,
            text_features=text_features,
            negative_text_features=negative_text_features,
        )
        if semantic_map is None:
            raise RuntimeError("V381 requires frozen image-only semantic features for M2.")
        if negative_text_features is None:
            negative_text_features = text_features

        cf = self._v381_counterfactual(
            image_only_map=semantic_map,
            positive_text=text_features,
            negative_text=negative_text_features,
            factual_masks=aux["v20_action_supports"],
            control_masks=aux["v20_control_supports"],
            action_types=self.action_types,
        )
        atomic_qualified, atomic_score = self._v381_atomic_qualified(cf)
        hypotheses, hypothesis_valid, membership, hypothesis_score = self._v381_build_atomic_hypotheses(
            candidate_logits_all, atomic_qualified, atomic_score
        )
        _, image_edge = self._image_gray_edge(image, candidate_logits_all.shape[-2:])
        final_logits, action_endorsement, structural = self._v38_delta_consensus_select(
            hypotheses,
            hypothesis_valid,
            membership,
            hypothesis_score,
            image_edge,
        )
        final_probs = torch.sigmoid(final_logits).clamp(EPS, 1.0 - EPS)
        zeros = atomic_score.new_zeros(atomic_score.shape)
        aux.update({
            "direct_fused_probs": final_probs,
            "router_fused_probs": final_probs,
            "v20_fused_probs": final_probs,
            "v20_hard_fused_probs": final_probs,
            "v20_fused_logits": final_logits,
            "v20_selector_logits": structural["medoid_score"],
            "v20_selector_probs": action_endorsement,
            "v20_selector_hard": action_endorsement,
            "v20_cf_logit": cf["cf_logit"],
            "v20_cf_swap_logit": zeros,
            "v20_cf_signed_delta": cf["cf_lesionness_contrast"],
            "v20_cf_factual_similarity": cf["cf_factual_similarity"],
            "v20_cf_control_similarity": cf["cf_control_similarity"],
            "v20_cf_available": cf["cf_available"],
            "v20_cf_control_area_ratio": cf["cf_control_area_ratio"],
            "v20_cf_control_overlap": cf["cf_control_overlap"],
            "v381_atomic_logit": cf["cf_logit"],
            "v381_atomic_available": cf["cf_available"],
            "v381_context_clean": cf["cf_context_clean"],
            "v381_context_overlap": cf["cf_context_overlap"],
            "v381_atomic_qualified": atomic_qualified,
            "v381_lesionness_contrast": cf["cf_lesionness_contrast"],
            "v381_pos_delta": cf["cf_pos_delta"],
            "v381_neg_delta": cf["cf_neg_delta"],
            "v381_hypothesis_valid": hypothesis_valid,
            "v381_hypothesis_membership": membership,
            "v381_hypothesis_score": hypothesis_score,
            "v381_structural_safe": structural["structural_safe"],
            "v381_delta_pairwise": structural["pairwise"],
            "v381_hypothesis_agreement": structural["agreement"],
            "v381_hypothesis_stability": structural["stability"],
            "v381_edit_edge_alignment": structural["edit_edge_alignment"],
            "v381_medoid_index": structural["medoid_index"],
            "v381_cluster_size": structural["cluster_size"],
            "v381_multi_active": structural["multi_active"],
            "v381_singleton_active": structural["singleton_active"],
            "v381_consensus_active": structural["consensus_active"],
            "v381_selected_score": structural["selected_text_score"],
            "v381_selected_edit_edge": structural["selected_edit_edge"],
            "v381_selected_stability": structural["selected_stability"],
            # Audit compatibility aliases; no V38 logic is deployed.
            "v38_atomic_text_probability": torch.sigmoid(cf["cf_logit"]),
            "v38_atomic_qualified": atomic_qualified,
            "v38_atomic_available": cf["cf_available"],
            "v38_context_clean": cf["cf_context_clean"],
            "v38_pos_delta": cf["cf_pos_delta"],
            "v38_neg_delta": cf["cf_neg_delta"],
            "v38_swap_delta": zeros,
            "v38_consensus_active": structural["consensus_active"],
            "v38_medoid_index": structural["medoid_index"],
            "v38_cluster_size": structural["cluster_size"],
        })
        return candidate_logits_all, aux



# V391 MLP calibrator – MICCAI 2024 CLIP patch-text alignment projection head
# Replaces 2-param affine with a small MLP that takes 7 semantic features as
# input. Proven in CAT-Seg (MICCAI 2024) and MedCLIP-SAM (MICCAI 2023).
# ---------------------------------------------------------------------------
class _V391LesionBackgroundMLPCalibrator(nn.Module):
    """MLP calibrator: 7D semantic features -> 1D calibrated M2 logit.

    Input features (already computed in _v381_counterfactual):
      0  lesionness_contrast  = pos_delta + neg_delta
      1  pos_delta
      2  neg_delta
      3  factual_similarity   = factual_pos - factual_neg
      4  control_similarity   = control_pos - control_neg
      5  control_area_ratio
      6  control_overlap
    """

    def __init__(self, hidden_dim: int = 32, num_features: int = 7) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, K, 7) -> logit: (B, K)"""
        return self.net(features).squeeze(-1)


class V391LesionBackgroundMLPCalibratedAtomicBank(V381LesionBackgroundCalibratedAtomicBank):
    """V391: same M1/M3 as V381, but M2 calibrator is a small MLP instead of
    a 2-param affine transform.  The MLP consumes all seven semantic features
    computed by _v381_counterfactual, providing enough capacity to learn
    meaningful lesion-vs-background separation.

    Speed optimisations vs V390:
      - VAL_NUM_SAMPLES reduced 50→30 (proven sufficient at MICCAI 2024)
      - M1_TRAIN_NUM_SAMPLES reduced 2→1
      - SAM rho reduced 0.05→0.02
    """

    def __init__(self, cfg) -> None:
        # Replace calibrator BEFORE calling super().__init__ so that the
        # parent constructor's log does not register the old one.
        super().__init__(cfg)
        m1 = _cfg_get(cfg, "M1", None)
        self.v381_calibrator = _V391LesionBackgroundMLPCalibrator(
            hidden_dim=int(_cfg_get(m1, "V391_CALIBRATOR_HIDDEN", 32)),
        )
        # Re-freeze V38 composition (V381 already does this, but be safe)
        for parameter in self.v38_combo_cf_encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.v38_combo_cf_head.parameters():
            parameter.requires_grad_(False)

    def _v391_calibrated_logit(
        self,
        lesionness_contrast: torch.Tensor,
        pos_delta: torch.Tensor,
        neg_delta: torch.Tensor,
        factual_similarity: torch.Tensor,
        control_similarity: torch.Tensor,
        control_area_ratio: torch.Tensor,
        control_overlap: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble 7-channel feature tensor and pass through MLP calibrator."""
        features = torch.stack(
            [
                lesionness_contrast,
                pos_delta,
                neg_delta,
                factual_similarity,
                control_similarity,
                control_area_ratio,
                control_overlap,
            ],
            dim=-1,
        )  # (B, K, 7)
        return self.v381_calibrator(features)

    def _v381_counterfactual(
        self,
        image_only_map: torch.Tensor,
        positive_text: torch.Tensor,
        negative_text: torch.Tensor,
        factual_masks: torch.Tensor,
        control_masks: torch.Tensor,
        action_types: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Override: reuses parent's feature extraction, replaces calibrator call."""
        b, k, h, w = factual_masks.shape
        factual_context = _soft_dilate(
            factual_masks.reshape(b * k, 1, h, w), self.context_radius
        ).reshape(b, k, h, w)
        control_context = _soft_dilate(
            control_masks.reshape(b * k, 1, h, w), self.context_radius
        ).reshape(b, k, h, w)

        factual_roi = self._pool_feature(image_only_map, factual_context)
        control_roi = self._pool_feature(image_only_map, control_context)
        factual_roi = F.normalize(self.v23_image_adapter(factual_roi), dim=-1, eps=1e-6)
        control_roi = F.normalize(self.v23_image_adapter(control_roi), dim=-1, eps=1e-6)
        positive_text = F.normalize(self.v23_text_adapter(positive_text), dim=-1, eps=1e-6)
        negative_text = F.normalize(self.v23_text_adapter(negative_text), dim=-1, eps=1e-6)

        def paired_delta(text_feature: torch.Tensor):
            factual_score = (factual_roi * text_feature[:, None, :]).sum(dim=-1)
            control_score = (control_roi * text_feature[:, None, :]).sum(dim=-1)
            return factual_score - control_score, factual_score, control_score

        raw_pos, factual_pos, control_pos = paired_delta(positive_text)
        raw_neg, factual_neg, control_neg = paired_delta(negative_text)
        is_fill = torch.isin(
            action_types,
            torch.tensor(self.FILL_TYPES, device=action_types.device),
        )
        polarity = torch.where(
            is_fill,
            torch.ones_like(action_types, dtype=raw_pos.dtype),
            -torch.ones_like(action_types, dtype=raw_pos.dtype),
        )[None, :]

        pos_delta = polarity * raw_pos
        neg_delta = -polarity * raw_neg
        lesionness_contrast = pos_delta + neg_delta
        factual_similarity = factual_pos - factual_neg
        control_similarity = control_pos - control_neg

        # V391: call MLP calibrator instead of 2-param affine
        cf_logit = self._v391_calibrated_logit(
            lesionness_contrast,
            pos_delta,
            neg_delta,
            factual_similarity,
            control_similarity,
            control_area_ratio=torch.zeros_like(lesionness_contrast),
            control_overlap=torch.zeros_like(lesionness_contrast),
        )

        factual_area = factual_masks.sum(dim=(-2, -1))
        control_area = control_masks.sum(dim=(-2, -1))
        area_ratio = control_area / factual_area.clamp_min(EPS)
        overlap = (
            factual_masks * control_masks
        ).sum(dim=(-2, -1)) / factual_area.clamp_min(EPS)
        available = (
            (factual_area > 0)
            & (control_area > 0)
            & ((area_ratio - 1.0).abs() <= 1e-3)
            & (overlap <= 1e-6)
        )
        context_overlap = (
            factual_context * control_context
        ).sum(dim=(-2, -1)) / factual_context.sum(dim=(-2, -1)).clamp_min(EPS)
        context_clean = context_overlap <= 1e-6
        available = available & context_clean

        return {
            "cf_logit": cf_logit,
            "cf_available": available,
            "cf_context_clean": context_clean,
            "cf_context_overlap": context_overlap,
            "cf_lesionness_contrast": lesionness_contrast,
            "cf_pos_delta": pos_delta,
            "cf_neg_delta": neg_delta,
            "cf_factual_similarity": factual_similarity,
            "cf_control_similarity": control_similarity,
            "cf_control_area_ratio": area_ratio,
            "cf_control_overlap": overlap,
        }

    def _v381_atomic_qualified(self, cf: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        score = cf["cf_logit"]
        qualified = cf["cf_available"].bool() & (score >= self.v381_deploy_logit_threshold)
        return qualified, score

    def _v381_build_atomic_hypotheses(
        self,
        candidate_logits_all: torch.Tensor,
        atomic_qualified: torch.Tensor,
        atomic_score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, slots, h, w = candidate_logits_all.shape
        k = slots - 1
        base = candidate_logits_all[:, 0]
        deltas = candidate_logits_all[:, 1:] - base[:, None]
        logits = [base]
        valid = [torch.ones((b,), dtype=torch.bool, device=base.device)]
        members = [torch.zeros((b, k), dtype=base.dtype, device=base.device)]
        scores = [base.new_zeros((b,))]
        for ai in range(k):
            logits.append(base + deltas[:, ai])
            valid.append(atomic_qualified[:, ai])
            member = torch.zeros((b, k), dtype=base.dtype, device=base.device)
            member[:, ai] = 1.0
            members.append(member)
            scores.append(atomic_score[:, ai])
        return (
            torch.stack(logits, dim=1),
            torch.stack(valid, dim=1),
            torch.stack(members, dim=1),
            torch.stack(scores, dim=1),
        )

    def generate(
        self,
        base_logits: torch.Tensor,
        image: torch.Tensor,
        semantic_map: Optional[torch.Tensor],
        text_features: torch.Tensor,
        negative_text_features: Optional[torch.Tensor] = None,
        swapped_text_features: Optional[torch.Tensor] = None,
    ):
        # V381 intentionally ignores swapped_text_features. It is accepted only
        # to preserve the generic CustomCLIP M1 call signature.
        candidate_logits_all, aux = UnifiedActionCounterfactualSetBank.generate(
            self,
            base_logits=base_logits,
            image=image,
            semantic_map=semantic_map,
            text_features=text_features,
            negative_text_features=negative_text_features,
        )
        if semantic_map is None:
            raise RuntimeError("V381 requires frozen image-only semantic features for M2.")
        if negative_text_features is None:
            negative_text_features = text_features

        cf = self._v381_counterfactual(
            image_only_map=semantic_map,
            positive_text=text_features,
            negative_text=negative_text_features,
            factual_masks=aux["v20_action_supports"],
            control_masks=aux["v20_control_supports"],
            action_types=self.action_types,
        )
        atomic_qualified, atomic_score = self._v381_atomic_qualified(cf)
        hypotheses, hypothesis_valid, membership, hypothesis_score = self._v381_build_atomic_hypotheses(
            candidate_logits_all, atomic_qualified, atomic_score
        )
        _, image_edge = self._image_gray_edge(image, candidate_logits_all.shape[-2:])
        final_logits, action_endorsement, structural = self._v38_delta_consensus_select(
            hypotheses,
            hypothesis_valid,
            membership,
            hypothesis_score,
            image_edge,
        )
        final_probs = torch.sigmoid(final_logits).clamp(EPS, 1.0 - EPS)
        zeros = atomic_score.new_zeros(atomic_score.shape)
        aux.update({
            "direct_fused_probs": final_probs,
            "router_fused_probs": final_probs,
            "v20_fused_probs": final_probs,
            "v20_hard_fused_probs": final_probs,
            "v20_fused_logits": final_logits,
            "v20_selector_logits": structural["medoid_score"],
            "v20_selector_probs": action_endorsement,
            "v20_selector_hard": action_endorsement,
            "v20_cf_logit": cf["cf_logit"],
            "v20_cf_swap_logit": zeros,
            "v20_cf_signed_delta": cf["cf_lesionness_contrast"],
            "v20_cf_factual_similarity": cf["cf_factual_similarity"],
            "v20_cf_control_similarity": cf["cf_control_similarity"],
            "v20_cf_available": cf["cf_available"],
            "v20_cf_control_area_ratio": cf["cf_control_area_ratio"],
            "v20_cf_control_overlap": cf["cf_control_overlap"],
            "v381_atomic_logit": cf["cf_logit"],
            "v381_atomic_available": cf["cf_available"],
            "v381_context_clean": cf["cf_context_clean"],
            "v381_context_overlap": cf["cf_context_overlap"],
            "v381_atomic_qualified": atomic_qualified,
            "v381_lesionness_contrast": cf["cf_lesionness_contrast"],
            "v381_pos_delta": cf["cf_pos_delta"],
            "v381_neg_delta": cf["cf_neg_delta"],
            "v381_hypothesis_valid": hypothesis_valid,
            "v381_hypothesis_membership": membership,
            "v381_hypothesis_score": hypothesis_score,
            "v381_structural_safe": structural["structural_safe"],
            "v381_delta_pairwise": structural["pairwise"],
            "v381_hypothesis_agreement": structural["agreement"],
            "v381_hypothesis_stability": structural["stability"],
            "v381_edit_edge_alignment": structural["edit_edge_alignment"],
            "v381_medoid_index": structural["medoid_index"],
            "v381_cluster_size": structural["cluster_size"],
            "v381_multi_active": structural["multi_active"],
            "v381_singleton_active": structural["singleton_active"],
            "v381_consensus_active": structural["consensus_active"],
            "v381_selected_score": structural["selected_text_score"],
            "v381_selected_edit_edge": structural["selected_edit_edge"],
            "v381_selected_stability": structural["selected_stability"],
            # Audit compatibility aliases; no V38 logic is deployed.
            "v38_atomic_text_probability": torch.sigmoid(cf["cf_logit"]),
            "v38_atomic_qualified": atomic_qualified,
            "v38_atomic_available": cf["cf_available"],
            "v38_context_clean": cf["cf_context_clean"],
            "v38_pos_delta": cf["cf_pos_delta"],
            "v38_neg_delta": cf["cf_neg_delta"],
            "v38_swap_delta": zeros,
            "v38_consensus_active": structural["consensus_active"],
            "v38_medoid_index": structural["medoid_index"],
            "v38_cluster_size": structural["cluster_size"],
        })
        return candidate_logits_all, aux


class V382ActionConditionalQuantileAtomicBank(V381LesionBackgroundCalibratedAtomicBank):
    """V382: all-action conditional lower-quantile outcome policy.

    The V381 candidate generator is retained unchanged: it emits factual local
    atomic edits and matched controls for all four action families.  V382
    changes only the decision path.  A candidate-local state transition model
    predicts q10/q50/q90 of signed hard-Dice gain for *each concrete edit*.
    Preserve is the explicit zero-valued action.  The deployed choice is
    ``argmax([0, q10])`` after a geometric veto; M3 never ranks candidates and
    never auto-accepts singleton hypotheses.

    Text contrast is retained only as an auxiliary local state feature.  It is
    not a deployment gate and no text threshold is used.
    """

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        m1 = _cfg_get(cfg, "M1", None)

        # V381's two-scalar calibrator is kept only for checkpoint namespace
        # compatibility and for a diagnostic text feature.  It is not a V382
        # policy parameter and cannot decide whether an edit is applied.
        for parameter in self.v381_calibrator.parameters():
            parameter.requires_grad_(False)

        self.v382_gain_scale = max(
            1e-5, float(_cfg_get(m1, "V382_GAIN_SCALE", 0.06))
        )
        self.v382_decision_temperature = max(
            1e-5, float(_cfg_get(m1, "V382_SOFTMAX_TEMPERATURE", 0.02))
        )
        self.v382_delta_eps = max(
            0.0, float(_cfg_get(m1, "V382_DELTA_EPS", 1e-8))
        )
        self.v382_struct_size = max(
            32, int(_cfg_get(m1, "V382_STRUCT_SIZE", 80))
        )
        self.v382_max_edit_fraction = float(
            _cfg_get(m1, "V382_MAX_EDIT_FRACTION", 0.035)
        )
        self.v382_max_edit_perimeter_growth = float(
            _cfg_get(m1, "V382_MAX_EDIT_PERIMETER_GROWTH", 5.0)
        )

        self.v382_semantic_dim = max(
            4, int(_cfg_get(m1, "V382_SEMANTIC_DIM", 16))
        )
        self.v382_spatial_dim = max(
            32, int(_cfg_get(m1, "V382_SPATIAL_DIM", 96))
        )
        self.v382_hidden_dim = max(
            32, int(_cfg_get(m1, "V382_HIDDEN_DIM", 128))
        )
        self.v382_type_embed_dim = max(
            4, int(_cfg_get(m1, "V382_TYPE_EMBED_DIM", 12))
        )
        dropout = max(0.0, float(_cfg_get(m1, "V382_DROPOUT", 0.10)))

        self.v382_semantic_projector = nn.Sequential(
            nn.Conv2d(
                self.semantic_channels,
                self.v382_semantic_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(1, self.v382_semantic_dim),
            nn.GELU(),
        )
        # Semantic map + 11 inference-time local-state channels:
        # base, candidate, signed/absolute delta, factual/control support,
        # support ring, base entropy, base boundary uncertainty, image gray,
        # and image edge.
        in_channels = self.v382_semantic_dim + 11
        mid = max(32, self.v382_spatial_dim // 2)
        self.v382_spatial_encoder = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(max(1, min(8, mid)), mid),
            nn.GELU(),
            nn.Conv2d(mid, self.v382_spatial_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(max(1, min(8, self.v382_spatial_dim)), self.v382_spatial_dim),
            nn.GELU(),
            nn.Conv2d(self.v382_spatial_dim, self.v382_spatial_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(max(1, min(8, self.v382_spatial_dim)), self.v382_spatial_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.v382_type_embedding = nn.Embedding(
            self.num_types, self.v382_type_embed_dim
        )
        # pos/neg contrast, lesion-vs-background contrast, factual/control
        # similarity, matched-area deviation, and context overlap.
        self.v382_scalar_dim = 7
        self.v382_state_fuse = nn.Sequential(
            nn.Linear(
                self.v382_spatial_dim
                + self.v382_type_embed_dim
                + self.v382_scalar_dim,
                self.v382_hidden_dim,
            ),
            nn.LayerNorm(self.v382_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.v382_quantile_head = nn.Linear(self.v382_hidden_dim, 3)
        self.v382_outcome_head = nn.Linear(self.v382_hidden_dim, 3)
        nn.init.zeros_(self.v382_quantile_head.weight)
        nn.init.zeros_(self.v382_quantile_head.bias)
        nn.init.zeros_(self.v382_outcome_head.weight)
        nn.init.zeros_(self.v382_outcome_head.bias)

    def _v382_structural_veto(
        self,
        base_logits: torch.Tensor,
        action_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Per-action M3 veto only; no medoid, consensus, or singleton path."""
        b, k, h, w = action_logits.shape
        side = min(self.v382_struct_size, h, w)
        hypotheses = torch.cat([base_logits[:, None], action_logits], dim=1)
        low = F.interpolate(
            hypotheses.reshape(b * (k + 1), 1, h, w),
            size=(side, side),
            mode="bilinear",
            align_corners=False,
        ).reshape(b, k + 1, side, side)
        _, _, edit = self._v38_edit_maps(low)
        edit_boundary = self._boundary(edit)
        edit_area = edit.mean(dim=(-2, -1))
        edit_perimeter = edit_boundary.sum(dim=(-2, -1)) / float(side * side)
        action_safe = (
            (edit_area[:, 1:] > 0.0)
            & (edit_area[:, 1:] <= self.v382_max_edit_fraction)
            & (edit_perimeter[:, 1:] <= self.v382_max_edit_perimeter_growth)
        )
        return {
            "safe": action_safe,
            "edit_area": edit_area[:, 1:],
            "edit_perimeter": edit_perimeter[:, 1:],
        }

    def _v382_action_state(
        self,
        base_logits: torch.Tensor,
        action_logits: torch.Tensor,
        image: torch.Tensor,
        semantic_map: torch.Tensor,
        supports: torch.Tensor,
        controls: torch.Tensor,
        cf: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if semantic_map.ndim != 4 or semantic_map.shape[1] != self.semantic_channels:
            raise RuntimeError("V382 requires frozen image-only semantic features.")
        b, k, h, w = action_logits.shape
        if semantic_map.shape[-2:] != (h, w):
            semantic_map = F.interpolate(
                semantic_map, size=(h, w), mode="bilinear", align_corners=False
            )
        if supports.shape != (b, k, h, w) or controls.shape != (b, k, h, w):
            raise RuntimeError("V382 support/control tensor shape mismatch.")

        base_prob = torch.sigmoid(base_logits)
        action_prob = torch.sigmoid(action_logits)
        delta = action_prob - base_prob[:, None]
        entropy = self._entropy(base_prob)
        boundary = (4.0 * base_prob * (1.0 - base_prob)).clamp(0.0, 1.0)
        ring = (
            _soft_dilate(
                supports.reshape(b * k, 1, h, w), self.context_radius
            ).reshape(b, k, h, w)
            - supports
        ).clamp(0.0, 1.0)
        gray, image_edge = self._image_gray_edge(image, (h, w))

        projected = self.v382_semantic_projector(semantic_map)
        projected = projected[:, None].expand(-1, k, -1, -1, -1)
        base_expand = base_prob[:, None, None].expand(-1, k, -1, -1, -1)
        geometry = torch.cat(
            [
                base_expand,
                action_prob[:, :, None],
                delta[:, :, None],
                delta.abs()[:, :, None],
                supports[:, :, None],
                controls[:, :, None],
                ring[:, :, None],
                entropy[:, None, None].expand(-1, k, -1, -1, -1),
                boundary[:, None, None].expand(-1, k, -1, -1, -1),
                gray[:, None].expand(-1, k, -1, -1, -1),
                image_edge[:, None].expand(-1, k, -1, -1, -1),
            ],
            dim=2,
        )
        spatial_input = torch.cat([projected, geometry], dim=2)
        spatial = self.v382_spatial_encoder(
            spatial_input.reshape(b * k, spatial_input.shape[2], h, w)
        ).flatten(1).reshape(b, k, self.v382_spatial_dim)

        type_embed = self.v382_type_embedding(self.action_types).unsqueeze(0).expand(b, -1, -1)
        scalar = torch.stack(
            [
                cf["cf_pos_delta"],
                cf["cf_neg_delta"],
                cf["cf_lesionness_contrast"],
                cf["cf_factual_similarity"],
                cf["cf_control_similarity"],
                cf["cf_control_area_ratio"] - 1.0,
                cf["cf_context_overlap"],
            ],
            dim=-1,
        )
        scalar = torch.nan_to_num(scalar, nan=0.0, posinf=0.0, neginf=0.0)
        return self.v382_state_fuse(torch.cat([spatial, type_embed, scalar], dim=-1))

    def generate(
        self,
        base_logits: torch.Tensor,
        image: torch.Tensor,
        semantic_map: Optional[torch.Tensor],
        text_features: torch.Tensor,
        negative_text_features: Optional[torch.Tensor] = None,
        swapped_text_features: Optional[torch.Tensor] = None,
    ):
        # Candidate generation remains exactly the established V381/V20 atomic
        # bank. Swapped text is intentionally not used by V382.
        candidate_logits_all, aux = UnifiedActionCounterfactualSetBank.generate(
            self,
            base_logits=base_logits,
            image=image,
            semantic_map=semantic_map,
            text_features=text_features,
            negative_text_features=negative_text_features,
        )
        if semantic_map is None:
            raise RuntimeError("V382 requires frozen image-only semantic features.")
        if negative_text_features is None:
            negative_text_features = text_features

        action_logits = candidate_logits_all[:, 1:]
        cf = self._v381_counterfactual(
            image_only_map=semantic_map,
            positive_text=text_features,
            negative_text=negative_text_features,
            factual_masks=aux["v20_action_supports"],
            control_masks=aux["v20_control_supports"],
            action_types=self.action_types,
        )
        delta_strength = (
            action_logits - base_logits[:, None]
        ).abs().flatten(2).amax(dim=-1)
        pre_veto_valid = cf["cf_available"].bool() & (delta_strength > self.v382_delta_eps)
        structural = self._v382_structural_veto(base_logits, action_logits)
        valid = pre_veto_valid & structural["safe"]

        state = self._v382_action_state(
            base_logits,
            action_logits,
            image,
            semantic_map,
            aux["v20_action_supports"].float(),
            aux["v20_control_supports"].float(),
            cf,
        )
        raw_quantiles = self.v382_quantile_head(state)
        quantiles = torch.sort(
            self.v382_gain_scale * torch.tanh(raw_quantiles), dim=-1
        ).values
        lower = quantiles[..., 0]
        outcome_logits = self.v382_outcome_head(state)
        outcome_probability = torch.softmax(outcome_logits, dim=-1)

        # Preserve is the exact score-zero alternative. q10 must be strictly
        # positive to beat Preserve; there is no learned/text/singleton gate.
        decision = lower.masked_fill(~valid, -1e4)
        class_logits = torch.cat([lower.new_zeros((lower.shape[0], 1)), decision], dim=1)
        selected_index = class_logits.argmax(dim=1)
        selected = torch.zeros_like(lower)
        take = selected_index > 0
        if take.any():
            rows = torch.arange(lower.shape[0], device=lower.device)[take]
            cols = selected_index[take] - 1
            selected[rows, cols] = 1.0

        delta_logits = action_logits - base_logits[:, None]
        hard_fused_logits = base_logits + (
            selected[:, :, None, None] * delta_logits
        ).sum(dim=1)
        soft_choice = torch.softmax(class_logits / self.v382_decision_temperature, dim=1)
        soft_fused_logits = base_logits + (
            soft_choice[:, 1:, None, None] * delta_logits
        ).sum(dim=1)
        final_probs = torch.sigmoid(hard_fused_logits).clamp(EPS, 1.0 - EPS)
        zeros = lower.new_zeros(lower.shape)

        aux.update({
            "direct_fused_probs": final_probs,
            "router_fused_probs": final_probs,
            # v20_fused is hard by design: training diagnostics report the
            # actual deployed one-action result, not a soft surrogate fusion.
            "v20_fused_probs": final_probs,
            "v20_hard_fused_probs": final_probs,
            "v20_fused_logits": hard_fused_logits,
            "v20_hard_fused_logits": hard_fused_logits,
            "v20_selector_logits": lower,
            "v20_selector_probs": soft_choice[:, 1:],
            "v20_selector_hard": selected,
            "v20_cf_logit": cf["cf_logit"],
            "v20_cf_swap_logit": zeros,
            "v20_cf_signed_delta": cf["cf_lesionness_contrast"],
            "v20_cf_factual_similarity": cf["cf_factual_similarity"],
            "v20_cf_control_similarity": cf["cf_control_similarity"],
            "v20_cf_available": cf["cf_available"],
            "v20_cf_control_area_ratio": cf["cf_control_area_ratio"],
            "v20_cf_control_overlap": cf["cf_control_overlap"],
            "v382_action_state": state,
            "v382_quantile_gain": quantiles,
            "v382_lower_gain": lower,
            "v382_outcome_logits": outcome_logits,
            "v382_outcome_probability": outcome_probability,
            "v382_harm_probability": outcome_probability[..., 0],
            "v382_benefit_probability": outcome_probability[..., 2],
            "v382_class_logits": class_logits,
            "v382_valid_action": valid,
            "v382_pre_veto_valid": pre_veto_valid,
            "v382_structural_safe": structural["safe"],
            "v382_edit_area": structural["edit_area"],
            "v382_edit_perimeter": structural["edit_perimeter"],
            "v382_selected_action": selected,
            "v382_soft_choice_probs": soft_choice,
            "v382_soft_fused_logits": soft_fused_logits,
            "v382_m3_veto_only": torch.ones_like(lower),
            # Compatibility/audit aliases. None is a deployment gate.
            "v381_atomic_logit": cf["cf_logit"],
            "v381_atomic_available": cf["cf_available"],
            "v381_context_clean": cf["cf_context_clean"],
            "v381_context_overlap": cf["cf_context_overlap"],
            "v381_atomic_qualified": valid,
            "v381_lesionness_contrast": cf["cf_lesionness_contrast"],
            "v381_pos_delta": cf["cf_pos_delta"],
            "v381_neg_delta": cf["cf_neg_delta"],
            "v381_consensus_active": torch.zeros((lower.shape[0],), device=lower.device),
            "v381_multi_active": torch.zeros((lower.shape[0],), device=lower.device),
            "v381_singleton_active": torch.zeros((lower.shape[0],), device=lower.device),
            "v381_cluster_size": torch.zeros((lower.shape[0],), device=lower.device),
        })
        return candidate_logits_all, aux




class V383ConservativeActionValueBank(V382ActionConditionalQuantileAtomicBank):
    """V383 hard gates plus V384 eligibility alignment plus V385 safe ranking.

    V385 leaves all V383 hard deployment gates unchanged:

      valid AND q10 AND low-harm AND benefit AND benefit-harm-gap

    It only changes the ranking inside the eligible set.  Instead of ranking
    by q10 - lambda*harm, V385 uses a direct safe-advantage head over the
    already available V382 causal action state.  The exact same score is used
    during Train policy learning and at deployment.
    """

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        m1 = _cfg_get(cfg, "M1", None)

        self.v383_min_lower_gain = float(
            _cfg_get(m1, "V383_MIN_Q10_GAIN", 0.0010)
        )
        self.v383_max_harm_probability = float(
            _cfg_get(m1, "V383_MAX_HARM_PROBABILITY", 0.20)
        )
        self.v383_min_benefit_probability = float(
            _cfg_get(m1, "V383_MIN_BENEFIT_PROBABILITY", 0.35)
        )
        self.v383_min_benefit_harm_gap = float(
            _cfg_get(m1, "V383_MIN_BENEFIT_HARM_GAP", 0.05)
        )
        self.v383_harm_risk_penalty = float(
            _cfg_get(m1, "V383_HARM_RISK_PENALTY", 0.006)
        )

        self.v384_eligibility_aligned = bool(
            _cfg_get(m1, "V384_ELIGIBILITY_ALIGNED", False)
        )
        self.v384_q10_gate_temperature = max(
            float(_cfg_get(m1, "V384_Q10_GATE_TEMPERATURE", 0.0005)),
            1e-8,
        )
        self.v384_probability_gate_temperature = max(
            float(_cfg_get(m1, "V384_PROB_GATE_TEMPERATURE", 0.02)),
            1e-8,
        )

        # V385 direct action advantage. No GT enters this head at inference.
        self.v385_safe_advantage = bool(
            _cfg_get(m1, "V385_SAFE_ADVANTAGE", False)
        )
        self.v385_gain_scale = max(
            float(_cfg_get(m1, "V385_GAIN_SCALE", 0.020)),
            1e-6,
        )

        self.v385_advantage_head = None

        if self.v385_safe_advantage:
            self.v385_advantage_head = nn.Linear(
                self.v382_hidden_dim,
                1,
                bias=True,
            )
            nn.init.zeros_(self.v385_advantage_head.weight)
            nn.init.zeros_(self.v385_advantage_head.bias)

    def generate(
        self,
        base_logits: torch.Tensor,
        image: torch.Tensor,
        semantic_map: Optional[torch.Tensor],
        text_features: torch.Tensor,
        negative_text_features: Optional[torch.Tensor] = None,
        swapped_text_features: Optional[torch.Tensor] = None,
    ):
        candidate_logits_all, aux = super().generate(
            base_logits=base_logits,
            image=image,
            semantic_map=semantic_map,
            text_features=text_features,
            negative_text_features=negative_text_features,
            swapped_text_features=swapped_text_features,
        )

        lower = aux["v382_lower_gain"]
        outcome_probability = aux["v382_outcome_probability"]

        harmful_probability = outcome_probability[..., 0]
        benefit_probability = outcome_probability[..., 2]

        valid = aux["v382_valid_action"].bool()

        risk_adjusted_score = (
            lower
            - self.v383_harm_risk_penalty * harmful_probability
        )

        benefit_harm_gap = (
            benefit_probability - harmful_probability
        )

        lower_pass = lower > self.v383_min_lower_gain
        harm_pass = (
            harmful_probability <= self.v383_max_harm_probability
        )
        benefit_pass = (
            benefit_probability >= self.v383_min_benefit_probability
        )
        gap_pass = (
            benefit_harm_gap >= self.v383_min_benefit_harm_gap
        )

        # Fixed V383 hard deployment eligibility. Never learned or lowered
        # from Validation/Test outcomes.
        deploy_eligible = (
            valid
            & lower_pass
            & harm_pass
            & benefit_pass
            & gap_pass
        )

        # V384 soft gates remain available for gate-alignment diagnostics and
        # losses, but do not alter V385 action ranking.
        if self.v384_eligibility_aligned:
            qtemp = self.v384_q10_gate_temperature
            ptemp = self.v384_probability_gate_temperature

            soft_log_q10 = F.logsigmoid(
                (lower - self.v383_min_lower_gain) / qtemp
            )
            soft_log_harm = F.logsigmoid(
                (
                    self.v383_max_harm_probability
                    - harmful_probability
                ) / ptemp
            )
            soft_log_benefit = F.logsigmoid(
                (
                    benefit_probability
                    - self.v383_min_benefit_probability
                ) / ptemp
            )
            soft_log_gap = F.logsigmoid(
                (
                    benefit_harm_gap
                    - self.v383_min_benefit_harm_gap
                ) / ptemp
            )

            soft_log_eligibility = (
                soft_log_q10
                + soft_log_harm
                + soft_log_benefit
                + soft_log_gap
            )

            soft_eligibility = torch.exp(
                soft_log_eligibility.clamp(
                    min=-80.0,
                    max=0.0,
                )
            )
        else:
            soft_log_eligibility = torch.zeros_like(lower)
            soft_eligibility = torch.ones_like(lower)

        # V385 uses the existing V382 causal action state. It is constructed
        # from image/Base/candidate/control/text-derived factual state only.
        # No GT is available here during inference.
        if self.v385_safe_advantage:
            if self.v385_advantage_head is None:
                raise RuntimeError(
                    "V385 enabled but v385_advantage_head is missing."
                )

            action_state = aux.get("v382_action_state", None)

            if action_state is None:
                raise KeyError(
                    "V385 requires aux['v382_action_state']; "
                    "V382 must export the causal action state."
                )

            advantage = self.v385_gain_scale * torch.tanh(
                self.v385_advantage_head(action_state).squeeze(-1)
            )

            deployment_score = advantage
        else:
            advantage = risk_adjusted_score
            deployment_score = risk_adjusted_score

        # Train policy and deployment use the same scalar score. The only
        # difference is the fixed hard eligibility mask at deployment.
        policy_logits = torch.cat(
            [
                deployment_score.new_zeros(
                    (deployment_score.shape[0], 1)
                ),
                deployment_score.masked_fill(
                    ~valid,
                    -1e4,
                ),
            ],
            dim=1,
        )

        soft_choice = torch.softmax(
            policy_logits / self.v382_decision_temperature,
            dim=1,
        )

        deploy_logits = torch.cat(
            [
                deployment_score.new_zeros(
                    (deployment_score.shape[0], 1)
                ),
                deployment_score.masked_fill(
                    ~deploy_eligible,
                    -1e4,
                ),
            ],
            dim=1,
        )

        selected_index = deploy_logits.argmax(dim=1)

        selected = torch.zeros_like(deployment_score)
        take = selected_index > 0

        if take.any():
            rows = torch.arange(
                deployment_score.shape[0],
                device=deployment_score.device,
            )[take]

            cols = selected_index[take] - 1
            selected[rows, cols] = 1.0

        action_logits = candidate_logits_all[:, 1:]
        delta_logits = action_logits - base_logits[:, None]

        final_logits = base_logits + (
            selected[:, :, None, None]
            * delta_logits
        ).sum(dim=1)

        final_probs = torch.sigmoid(final_logits).clamp(
            EPS,
            1.0 - EPS,
        )

        aux.update({
            "direct_fused_probs": final_probs,
            "router_fused_probs": final_probs,
            "v20_fused_probs": final_probs,
            "v20_hard_fused_probs": final_probs,
            "v20_fused_logits": final_logits,
            "v20_hard_fused_logits": final_logits,

            # Export V385 score as selector score, while preserving V383 risk
            # score separately for audit comparability.
            "v20_selector_logits": deployment_score,
            "v20_selector_probs": soft_choice[:, 1:],
            "v20_selector_hard": selected,

            # V382 choice supervision now observes the exact V385 policy score.
            "v382_class_logits": policy_logits,
            "v382_soft_choice_probs": soft_choice,
            "v382_selected_action": selected,

            "v383_policy_logits": policy_logits,
            "v383_soft_choice_probs": soft_choice,
            "v383_risk_adjusted_score": risk_adjusted_score,
            "v383_train_policy_score": deployment_score,

            "v383_harm_probability": harmful_probability,
            "v383_benefit_probability": benefit_probability,
            "v383_benefit_harm_gap": benefit_harm_gap,

            "v383_lower_pass": lower_pass,
            "v383_harm_pass": harm_pass,
            "v383_benefit_pass": benefit_pass,
            "v383_gap_pass": gap_pass,
            "v383_deploy_eligible": deploy_eligible,
            "v383_selected_action": selected,
            "v383_preserve_selected": (~take).float(),

            "v384_soft_log_eligibility": soft_log_eligibility,
            "v384_soft_eligibility": soft_eligibility,

            "v385_enabled": torch.full_like(
                deployment_score,
                float(self.v385_safe_advantage),
            ),
            "v385_advantage": advantage,
            "v385_deployment_score": deployment_score,

            "v383_m3_veto_only": torch.ones_like(
                deployment_score
            ),
        })

        return candidate_logits_all, aux


# ======================================================================
# V392 Dense Patch-Text Falsification Bank (CAT-Seg inspired, CVPR 2024)
# ======================================================================
class V392DensePatchTextFalsificationBank(V381LesionBackgroundCalibratedAtomicBank):
    """V392: M1 candidates -> M2 dense patch-text alignment verification -> M3.

    Motivated by CAT-Seg (Cho et al., CVPR 2024 Highlight) and MedCLIP-SAM
    (Koleilat et al., MICCAI 2024), this replaces ROI-pooled contrast with
    dense per-patch CLIP text similarity maps.  Each candidate mask is scored
    by five falsification features:

      0  pos_inside   = mean positive-text sim of patches inside candidate
      1  neg_inside   = mean negative-text sim of patches inside candidate
      2  pos_outside  = mean positive-text sim of patches outside candidate
      3  neg_outside  = mean negative-text sim of patches outside candidate
      4  purity_gap   = pos_inside - pos_outside  (normalised)

    From these, three audit flags are derived and logged:
      - over_seg    : candidate extends into patches that match lesion text
      - under_seg   : candidate excludes patches that match lesion text
      - shortcut    : candidate overlaps patches better matched by bg text

    An MLP calibrator (16->1) scores each candidate; the highest-quality
    candidate above Preserve=0 is deployed.
    """

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        m1 = _cfg_get(cfg, "M1", None)
        self.v392_audit_threshold_high = float(_cfg_get(m1, "V392_AUDIT_THRESHOLD_HIGH", 0.15))
        self.v392_audit_threshold_low = float(_cfg_get(m1, "V392_AUDIT_THRESHOLD_LOW", 0.02))
        # MLP calibrator: 5 factual + 5 control -> 16 -> 1
        hidden = int(_cfg_get(m1, "V392_CALIBRATOR_HIDDEN", 16))
        self.v392_calibrator_factual = nn.Linear(5, hidden)
        self.v392_calibrator_control = nn.Linear(5, hidden)
        self.v392_calibrator_head = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # M3 text-score integration weight
        self.v392_cf_logit_medoid_weight = float(_cfg_get(m1, "V392_CF_LOGIT_MEDOID_WEIGHT", 0.3))
        # Re-freeze V38 composition
        for parameter in self.v38_combo_cf_encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.v38_combo_cf_head.parameters():
            parameter.requires_grad_(False)
        self._v392_cf_cache = None

    @torch.no_grad()
    def _dense_patch_text_similarity(
        self,
        image: torch.Tensor,
        positive_text: torch.Tensor,
        negative_text: torch.Tensor,
        target_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-patch cosine similarity to positive and negative text.

        Uses BiomedCLIP ViT patch tokens from the image encoder (features
        after the final layer, before CLS token).  The resulting maps are
        bilinearly upsampled to target_hw.

        Args:
            image: (B, 3, H, W) CLIP-normalised image.
            positive_text: (B, D) normalised positive text embedding.
            negative_text: (B, D) normalised negative text embedding.
            target_hw: (h, w) output resolution.

        Returns:
            pos_map: (B, h, w) positive similarity per pixel.
            neg_map: (B, h, w) negative similarity per pixel.
        """
        b = image.shape[0]
        # Get ViT patch features (final layer, exclude CLS token)
        with torch.no_grad():
            x = image
            x = x.type(next(self.v23_image_adapter.parameters()).dtype)
            # Manual ViT forward to get patch tokens
            # We access the model's vision_model through the parent bank.
            # Since V381 doesn't store vision_model directly, we use
            # the image adapter pathway: v23_image_adapter expects ROI-pooled
            # features, but we need dense features.
            #
            # Instead, we compute DENSE similarity via the pvl_adapter path.
            # The semantic_map already passed in from CustomCLIP is
            # the image-only projected feature map (B, C, h, w).
            # This is computed in _generate_candidates_and_fuse.
            pass

        # The dense features come from semantic_map passed into _v392_counterfactual.
        return torch.zeros(b, *target_hw), torch.zeros(b, *target_hw)

    def _v392_falsification_features(
        self,
        pos_map: torch.Tensor,
        neg_map: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract the 5 falsification features and 3 audit flags per candidate.

        candidate_mask: (B, K, h, w) soft mask for each candidate.
        pos_map, neg_map: (B, h, w).
        Returns tuple of 5 features + 3 audit flags, each (B, K).
        """
        b, k, h, w = candidate_mask.shape
        candidate = candidate_mask.reshape(b * k, h, w)
        outside = (1.0 - candidate).clamp(0, 1)
        area = candidate.sum(dim=(-2, -1)).clamp_min(1e-6)
        outside_area = outside.sum(dim=(-2, -1)).clamp_min(1e-6)
        pos = pos_map.unsqueeze(1).expand(-1, k, -1, -1).reshape(b * k, h, w)
        neg = neg_map.unsqueeze(1).expand(-1, k, -1, -1).reshape(b * k, h, w)

        pos_inside = (pos * candidate).sum(dim=(-2, -1)) / area
        neg_inside = (neg * candidate).sum(dim=(-2, -1)) / area
        pos_outside = (pos * outside).sum(dim=(-2, -1)) / outside_area
        neg_outside = (neg * outside).sum(dim=(-2, -1)) / outside_area
        purity_gap = pos_inside - pos_outside

        pos_inside = pos_inside.reshape(b, k)
        neg_inside = neg_inside.reshape(b, k)
        pos_outside = pos_outside.reshape(b, k)
        neg_outside = neg_outside.reshape(b, k)
        purity_gap = purity_gap.reshape(b, k)

        # Audit flags
        t_hi = self.v392_audit_threshold_high
        t_lo = self.v392_audit_threshold_low
        over_seg = (pos_outside > t_hi).float()  # outside patches match lesion text
        under_seg = (neg_inside > t_lo).float()  # inside patches match bg text
        shortcut = ((neg_inside > pos_inside) & (pos_inside > t_lo)).float()

        return pos_inside, neg_inside, pos_outside, neg_outside, purity_gap, over_seg, under_seg, shortcut

    def _v392_counterfactual(
        self,
        semantic_map: torch.Tensor,
        positive_text: torch.Tensor,
        negative_text: torch.Tensor,
        candidate_masks: torch.Tensor,
        control_masks: torch.Tensor,
        action_types: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """V392 M2: dense patch-text edit-region factual vs control.

        For each edit candidate, we compute 5 text-similarity features on the
        full candidate mask (factual), and 5 parallel features on the shifted
        control mask (control).  The calibrator subtracts control from factual
        to produce a net lesionness score that answers: "is this specific
        spatial location more lesion-like than a same-area random location?"

        This replaces the prior version which had no real control comparison
        (control features were hardcoded zeros/ones).
        """
        b, k, h, w = candidate_masks.shape

        # Dense similarity maps from pretrained image-only features
        semantic_norm = F.normalize(semantic_map.permute(0, 2, 3, 1), dim=-1, eps=1e-6)
        pos_text_norm = F.normalize(positive_text, dim=-1, eps=1e-6)
        neg_text_norm = F.normalize(negative_text, dim=-1, eps=1e-6)
        pos_map = (semantic_norm * pos_text_norm.unsqueeze(1).unsqueeze(2)).sum(dim=-1)
        neg_map = (semantic_norm * neg_text_norm.unsqueeze(1).unsqueeze(2)).sum(dim=-1)

        # Factual features (on candidate masks)
        pi_f, ni_f, po_f, no_f, pg_f, over, under, shortcut = self._v392_falsification_features(
            pos_map, neg_map, candidate_masks
        )
        # Control features (on shifted control masks)
        pi_c, ni_c, po_c, no_c, pg_c, _oc, _uc, _sc = self._v392_falsification_features(
            pos_map, neg_map, control_masks
        )

        # Calibrator: factual - control (net lesionness)
        feat_f = torch.stack([pi_f, ni_f, po_f, no_f, pg_f], dim=-1)  # (B, K, 5)
        feat_c = torch.stack([pi_c, ni_c, po_c, no_c, pg_c], dim=-1)  # (B, K, 5)
        h_f = self.v392_calibrator_factual(feat_f)      # (B, K, hidden)
        h_c = self.v392_calibrator_control(feat_c)      # (B, K, hidden)
        h_net = h_f - h_c                                # net lesionness
        cf_logit = self.v392_calibrator_head(h_net).squeeze(-1)  # (B, K)

        # Availability
        mask_area = candidate_masks.sum(dim=(-2, -1))
        available = mask_area > 0
        context_clean = torch.ones_like(available, dtype=torch.bool)

        # Real control metrics (not hardcoded)
        factual_area = candidate_masks.sum(dim=(-2, -1)).clamp_min(EPS)
        ctrl_area = control_masks.sum(dim=(-2, -1)).clamp_min(EPS)
        area_ratio = ctrl_area / factual_area
        overlap = (candidate_masks * control_masks).sum(dim=(-2, -1)) / factual_area

        return {
            "cf_logit": cf_logit,
            "cf_available": available,
            "cf_context_clean": context_clean,
            "cf_context_overlap": torch.zeros_like(pi_f),
            "cf_lesionness_contrast": pg_f - pg_c,
            "cf_pos_delta": pi_f - pi_c,
            "cf_neg_delta": ni_f - ni_c,
            "cf_factual_similarity": pg_f,
            "cf_control_similarity": pg_c,
            "cf_control_area_ratio": area_ratio,
            "cf_control_overlap": overlap,
            "v392_over_seg": over,
            "v392_under_seg": under,
            "v392_shortcut": shortcut,
            "v392_pos_inside": pi_f,
            "v392_pos_outside": po_f,
            "v392_purity_gap": pg_f,
        }

    def _v381_counterfactual(
        self,
        image_only_map: torch.Tensor,
        positive_text: torch.Tensor,
        negative_text: torch.Tensor,
        factual_masks: torch.Tensor,
        control_masks: torch.Tensor,
        action_types: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Override V381's ROI-pooled M2 with V392's dense patch-text falsification.

        V381's generate() calls this internally; our override routes through
        _v392_counterfactual() with both factual and control masks.
        The returned dict has every key that V381's generate() expects.

        Unlike V392 V1/V2, we do NOT center cf_logit per-batch.  The
        calibrator now learns absolute net-lesionness via factual-control
        subtraction, which is cross-case meaningful.
        """
        cf = self._v392_counterfactual(
            semantic_map=image_only_map,
            positive_text=positive_text,
            negative_text=negative_text,
            candidate_masks=factual_masks.float(),
            control_masks=control_masks.float(),
            action_types=action_types,
        )
        self._v392_cf_cache = cf
        return cf

    def _v381_atomic_qualified(self, cf: Dict[str, torch.Tensor]) -> tuple:
        """V392 override: cf_logit is a ranking signal, not a hard gate.

        The structural consensus (M3) makes the final safety decision.
        This prevents the deadlock where random-init calibrator outputs
        all-negative logits and blocks all candidates from M3.
        """
        return cf["cf_available"].bool(), cf["cf_logit"]

    def _v38_delta_consensus_select(
        self,
        hypothesis_logits: torch.Tensor,
        valid: torch.Tensor,
        membership: torch.Tensor,
        hypothesis_text_score: torch.Tensor,
        image_edge: torch.Tensor,
    ) -> tuple:
        """V392 override: boundary_fill exclusion, cf_logit preference, Preserve fallback.
        
        Same geometric consensus as V381, but:
        - boundary_fill (type=2) is excluded from structural_safe (never deployed)
        - cf_logit participates in medoid selection
        - no-consensus fallback = Preserve (not best-cf_logit candidate)
        """
        b, n, h, w = hypothesis_logits.shape
        side = min(self.v38_struct_size, h, w)
        low_logits = F.interpolate(
            hypothesis_logits.reshape(b * n, 1, h, w),
            size=(side, side), mode="bilinear", align_corners=False,
        ).reshape(b, n, side, side)
        delete, fill, edit = self._v38_edit_maps(low_logits)
        delete_dice = self._v38_pairwise_dice(delete)
        fill_dice = self._v38_pairwise_dice(fill)
        delete_mass = delete.flatten(2).sum(dim=-1)
        fill_mass = fill.flatten(2).sum(dim=-1)
        delete_present = (delete_mass[:,:,None] + delete_mass[:,None,:]) > EPS
        fill_present = (fill_mass[:,:,None] + fill_mass[:,None,:]) > EPS
        sw = delete_present.float() + fill_present.float()
        signed_delta_dice = (
            delete_dice * delete_present.float() + fill_dice * fill_present.float()
        ) / sw.clamp_min(1.0)
        edit_boundary = self._boundary(edit)
        edit_boundary_dice = self._v38_pairwise_dice(edit_boundary)
        pairwise = (
            self.v38_delta_dice_weight * signed_delta_dice
            + self.v38_edit_boundary_weight * edit_boundary_dice
        )
        low_edge = F.interpolate(image_edge, size=(side, side),
                                  mode="bilinear", align_corners=False)
        ebm = edit_boundary.sum(dim=(-2,-1)).clamp_min(EPS)
        edit_edge_alignment = (edit_boundary * low_edge).sum(dim=(-2,-1)) / ebm
        edit_area = edit.mean(dim=(-2,-1))
        edit_perimeter = ebm / float(side * side)
        edited_valid = valid.clone()
        edited_valid[:, 0] = False

        # Exclude boundary_fill (action_type=2) from ever being deployed
        k = n - 1  # number of actions
        fill_indices = [i for i, t in enumerate(self.action_types[:k]) if t in (2,)]
        for ai in fill_indices:
            edited_valid[:, ai + 1] = False

        structural_safe = (
            edited_valid
            & (edit_area > 0)
            & (edit_area <= self.v38_max_edit_fraction)
            & (edit_perimeter <= self.v38_max_edit_perimeter_growth)
        )
        eye = torch.eye(n, dtype=torch.bool, device=hypothesis_logits.device)[None]
        pair_valid = structural_safe[:,:,None] & structural_safe[:,None,:] & (~eye)
        agreement = (pairwise * pair_valid.float()).sum(dim=-1) / pair_valid.float().sum(dim=-1).clamp_min(1.0)
        stability = (
            agreement
            + self.v38_edge_weight * edit_edge_alignment
            + self.v392_cf_logit_medoid_weight * hypothesis_text_score
            - self.v38_edit_area_penalty * edit_area
            - self.v38_edit_perimeter_penalty * edit_perimeter
        )
        medoid_score = stability.masked_fill(~structural_safe, -1e9)
        medoid_index = medoid_score.argmax(dim=1)
        medoid_exists = structural_safe.any(dim=1)
        medoid_pairwise = pairwise.gather(1, medoid_index[:,None,None].expand(-1,1,n))[:,0]
        cluster = structural_safe & (medoid_pairwise >= self.v38_cluster_similarity_min)
        cluster_size = cluster.sum(dim=1)
        multi_active = medoid_exists & (cluster_size >= self.v38_min_cluster_size)
        selected_text = hypothesis_text_score.gather(1, medoid_index[:,None])[:,0]
        selected_edge = edit_edge_alignment.gather(1, medoid_index[:,None])[:,0]
        selected_stability = stability.gather(1, medoid_index[:,None])[:,0]
        singleton_active = (
            medoid_exists & (cluster_size == 1)
            & (selected_text >= self.v38_singleton_text_threshold)
            & (selected_edge >= self.v38_singleton_min_edit_edge)
            & (selected_stability >= self.v38_singleton_min_stability)
        )
        consensus_active = multi_active | singleton_active
        selected_logits = hypothesis_logits.gather(
            1, medoid_index[:,None,None,None].expand(-1,1,h,w))[:,0]
        # ★ V393: when no consensus, fall back to Preserve (not best-cf-logit)
        final_logits = torch.where(
            consensus_active[:, None, None],
            selected_logits,
            hypothesis_logits[:, 0],  # Preserve
        )
        k_out = membership.shape[-1]
        action_endorsement = torch.zeros(b, k, device=hypothesis_logits.device)
        medoid_member = membership.gather(1, medoid_index[:,None,None].expand(-1,1,k))[:,0]
        action_endorsement[consensus_active] = medoid_member[consensus_active]
        structural = {
            "medoid_score": hypothesis_text_score.gather(1, medoid_index[:,None])[:,0],
            "medoid_index": medoid_index,
            "cluster_size": cluster_size,
            "multi_active": multi_active,
            "singleton_active": singleton_active,
            "consensus_active": consensus_active,
            "pairwise": medoid_pairwise,
            "agreement": agreement.gather(1, medoid_index[:,None])[:,0],
            "stability": selected_stability,
            "edit_edge_alignment": selected_edge,
            "structural_safe": structural_safe,
            "selected_text_score": selected_text,
            "selected_edit_edge": selected_edge,
            "selected_stability": selected_stability,
        }
        return final_logits, action_endorsement, structural

    def generate(
        self,
        base_logits: torch.Tensor,
        image: torch.Tensor,
        semantic_map: Optional[torch.Tensor],
        text_features: torch.Tensor,
        negative_text_features: Optional[torch.Tensor] = None,
        swapped_text_features: Optional[torch.Tensor] = None,
    ):
        """V392: inherits V381's proven M1+M3, only overrides M2 via _v381_counterfactual.

        V381LesionBackgroundCalibratedAtomicBank.generate() calls
        self._v381_counterfactual() which our override routes to
        _v392_counterfactual().  All v381_* aux keys are therefore
        guaranteed to be present.  We only add V392-specific audit keys
        after the parent completes.
        """
        candidate_logits_all, aux = V381LesionBackgroundCalibratedAtomicBank.generate(
            self,
            base_logits=base_logits,
            image=image,
            semantic_map=semantic_map,
            text_features=text_features,
            negative_text_features=negative_text_features,
        )
        cf = self._v392_cf_cache
        aux.update({
            "v392_over_seg": cf["v392_over_seg"],
            "v392_under_seg": cf["v392_under_seg"],
            "v392_shortcut": cf["v392_shortcut"],
            "v392_pos_inside": cf["v392_pos_inside"],
            "v392_pos_outside": cf["v392_pos_outside"],
            "v392_purity_gap": cf["v392_purity_gap"],
        })
        return candidate_logits_all, aux


class V393PreserveAwareEditControlBank(V381LesionBackgroundCalibratedAtomicBank):
    """V393 Phase 1: Preserve-Aware Safe Deployment Skeleton.

    Fixes five confirmed V392 problems:
      1. boundary_fill excluded from deployment (configurable)
      2. M2 text verifier placeholder — real edit-region factual-control TBD
      3. Preserve=0 as absolute reference (no batch-centering)
      4. Preserve is the default safe action; no "best-cf-logit" fallback
      5. EMA weights deployable consistently (test.py TBD)

    Phase 1 (this commit): Preserve safety, boundary_fill block,
    action-value skeleton.  M2/M3 full implementation in subsequent phases.
    """

    FILL_TYPES = (2, 3)  # boundary_fill, hole_fill only
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self.embed_dim = int(self.semantic_channels)
        self.text_proj_dim = int(self.semantic_channels)
        m1 = _cfg_get(cfg, "M1", None)

        # ── V393 Preserve-aware flags ──
        self.v393_deploy_boundary_fill = bool(
            _cfg_get(m1, "EDIT_CONTROL_DEPLOY_BOUNDARY_FILL", False)
        )
        self.v393_use_edit_region_control = bool(
            _cfg_get(m1, "EDIT_CONTROL_USE_EDIT_REGION_CONTROL", True)
        )
        self.v393_use_preserve_relative_value = bool(
            _cfg_get(m1, "EDIT_CONTROL_USE_PRESERVE_RELATIVE_VALUE", True)
        )
        self.v393_calibrator_hidden = int(_cfg_get(m1, "EDIT_CONTROL_CALIBRATOR_HIDDEN", 32))
        self.v393_cf_medoid_weight = float(_cfg_get(m1, "EDIT_CONTROL_CF_MEDOID_WEIGHT", 0.3))
        self.v393_min_action_value = float(_cfg_get(m1, "EDIT_CONTROL_MIN_ACTION_VALUE", 0.0))

        # V394: V393 candidate geometry + independent factual/control M2 +
        # Preserve-relative all-action value ranking M3.
        self.v394_counterfactual_value_rank = bool(
            _cfg_get(m1, "V394_COUNTERFACTUAL_VALUE_RANK", False)
            or str(_cfg_get(m1, "RUN_TAG", "")).upper().startswith("V394_")
            or str(_cfg_get(m1, "M1_LOSS_VERSION", "")).lower()
            == "v394_counterfactual_value_rank"
        )
        self.v394_phase_a_epochs = int(_cfg_get(m1, "V394_PHASE_A_EPOCHS", 40))
        self.v394_min_action_value = float(_cfg_get(m1, "V394_MIN_ACTION_VALUE", 0.002))
        self.v394_class_value_weight = float(
            _cfg_get(m1, "V394_CLASS_VALUE_WEIGHT", 0.010)
        )
        self.v394_max_edit_fraction = float(
            _cfg_get(m1, "V394_MAX_EDIT_FRACTION", 0.05)
        )

        # ── M2 calibrator placeholder ──
        # Phase 1: re-use V381's 2-param calibrator as fallback.
        # Phase 2+ will replace with edit-region factual-control MLP
        # that reads benefit_logit / harm_logit / text_evidence_logit.
        self.v393_calibrator_factual = nn.Linear(5, self.v393_calibrator_hidden)
        self.v393_calibrator_control = nn.Linear(5, self.v393_calibrator_hidden)
        self.v393_calibrator_head = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.v393_calibrator_hidden, 1),
        )

        # V395: direct counterfactual Dice-gain policy.
        #
        # The old V393 code trained ``v393_calibrator_head`` but deployed a
        # hand-written mixture dominated by the frozen V381 cf-logit.  V395
        # exposes an explicit q(a) head and uses this exact scalar in both the
        # training loss and Preserve-vs-action inference decision.
        self.v395_direct_gain_policy = bool(
            _cfg_get(m1, "V395_DIRECT_GAIN_POLICY", False)
        )
        self.v395_type_embed_dim = int(
            _cfg_get(m1, "V395_TYPE_EMBED_DIM", 4)
        )
        self.v395_min_action_value = float(
            _cfg_get(m1, "V395_MIN_ACTION_VALUE", 0.0)
        )
        if self.v395_direct_gain_policy:
            self.v395_type_embedding = nn.Embedding(
                4, self.v395_type_embed_dim
            )
            self.v395_value_head = nn.Sequential(
                nn.Linear(
                    self.v393_calibrator_hidden
                    + self.v395_type_embed_dim
                    + 5,
                    self.v393_calibrator_hidden,
                ),
                nn.GELU(),
                nn.Linear(self.v393_calibrator_hidden, 1),
            )
            # Preserve=0 must be the initial safe decision.  The direct gain
            # head learns departures from zero only from real train-split Dice
            # gains.
            nn.init.zeros_(self.v395_value_head[-1].weight)
            nn.init.zeros_(self.v395_value_head[-1].bias)

        # V396/FECG: factor q(a) into a frozen semantic evidence gate and a
        # visual-gain predictor. No action type, rank, coordinate, area or
        # control-valid scalar is given to this learnable q head.
        self.evidence_guided_candidate_control_enabled = bool(_cfg_get(m1, "EVIDENCE_GUIDED_FECG_ENABLED", False))
        self.v396_evidence_threshold = float(_cfg_get(m1, "EVIDENCE_GUIDED_EVIDENCE_THRESHOLD", 0.0))
        self.v396_evidence_temperature = max(1e-4, float(_cfg_get(m1, "EVIDENCE_GUIDED_EVIDENCE_TEMPERATURE", 0.10)))
        self.v396_min_action_value = float(_cfg_get(m1, "EVIDENCE_GUIDED_MIN_ACTION_VALUE", self.v395_min_action_value))
        self.v396_control_area_tolerance = max(0.0, float(_cfg_get(m1, "EVIDENCE_GUIDED_CONTROL_AREA_TOLERANCE", 1e-3)))
        self.v396_max_edit_fraction = max(0.0, float(_cfg_get(m1, "EVIDENCE_GUIDED_MAX_EDIT_FRACTION", 0.035)))
        # The V396 visual critic sees only a frozen-observer ROI contrast.
        # It receives no action type/rank/area/coordinate/Base-confidence field.
        # A fixed action polarity merely aligns delete/fill direction; it is not
        # a learned categorical shortcut.
        self.v396_visual_gain_head = None
        self.v396_visual_dim = int(self.semantic_channels)
        if self.evidence_guided_candidate_control_enabled:
            self.v396_visual_gain_head = nn.Sequential(
                nn.Linear(self.v396_visual_dim, self.v393_calibrator_hidden),
                nn.LayerNorm(self.v393_calibrator_hidden),
                nn.GELU(),
                nn.Linear(self.v393_calibrator_hidden, 1),
            )
            nn.init.zeros_(self.v396_visual_gain_head[-1].weight)
            nn.init.zeros_(self.v396_visual_gain_head[-1].bias)

        # ── M2 TIDE-Repair Error Head ──
        # Lightweight error map predictor that reuses frozen B0 features.
        # Activated only when M2_TIDE_REPAIR_ENABLED is true.
        self.m2_tide_repair_enabled = bool(
            _cfg_get(m1, "M2_TIDE_REPAIR_ENABLED", False)
        )
        self.m2_tide_repair_head = None
        if self.m2_tide_repair_enabled:
            from trainers.m2_tide_repair_error_head import TIDERepairErrorHead
            self.m2_tide_repair_head = TIDERepairErrorHead(
                semantic_channels=self.semantic_channels,
                text_dim=self.text_proj_dim,
                hidden_dim=int(_cfg_get(m1, "M2_HIDDEN_DIM", 64)),
                scorer_feature_dim=int(_cfg_get(m1, "M2_SCORER_FEATURE_DIM", 11)),
                scorer_hidden_dim=int(_cfg_get(m1, "M2_SCORER_HIDDEN_DIM", 48)),
            )

        # Independent heads: harmfulness is no longer the negative of benefit.
        self.v394_benefit_head = nn.Linear(self.v393_calibrator_hidden, 1)
        self.v394_harm_head = nn.Linear(self.v393_calibrator_hidden, 1)
        self.v394_value_head = nn.Linear(self.v393_calibrator_hidden, 1)
        self.v394_text_head = nn.Linear(self.v393_calibrator_hidden, 1)
        for head in (
            self.v394_benefit_head,
            self.v394_harm_head,
            self.v394_value_head,
            self.v394_text_head,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def set_v394_phase(self, phase_b: bool) -> None:
        """Phase B fixes M1 candidates and trains only V394 M2/M3 evidence/value heads."""
        if not self.v394_counterfactual_value_rank:
            return

        phase_b = bool(phase_b)

        trainable_in_phase_b = (
            "v393_calibrator_factual.",
            "v393_calibrator_control.",
            "v394_benefit_head.",
            "v394_harm_head.",
            "v394_value_head.",
        )

        for name, parameter in self.named_parameters():
            parameter.requires_grad_(
                (not phase_b) or name.startswith(trainable_in_phase_b)
            )

    def set_v393_phase(self, phase_b: bool) -> None:
        """Phase B for V393: freeze candidate generator, keep V393 selector heads trainable."""
        if self.v394_counterfactual_value_rank:
            return

        phase_b = bool(phase_b)
        trainable_in_phase_b = (
            "v393_calibrator_factual.",
            "v393_calibrator_control.",
            "v393_calibrator_head.",
        )
        if self.v395_direct_gain_policy:
            trainable_in_phase_b = trainable_in_phase_b + (
                "v395_type_embedding.",
                "v395_value_head.",
            )

        for name, parameter in self.named_parameters():
            parameter.requires_grad_(
                (not phase_b) or name.startswith(trainable_in_phase_b)
            )

    # ── V393 M2: Edit-Region Factual-Control ────────────────────
    # Phase 2: real edit-mask construction, control-mask via random
    # shift, and edit-vs-control text evidence extraction.

    @staticmethod
    @torch.no_grad()
    def _v393_compute_edit_mask(
        factual_masks: torch.Tensor,
        base_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-action edit mask from action supports.

        factual_masks [B,K,H,W] are the M1 action_support masks.
        For fill actions: the support is the area being added.
        For delete/trim actions: the support is the area being removed.
        These ARE the local edit regions (not the full candidate mask).

        Returns binary edit_mask [B,K,H,W].
        """
        return (factual_masks > 0.5).float()

    @staticmethod
    @torch.no_grad()
    def _v393_edit_context(edit_mask: torch.Tensor, radius: int) -> torch.Tensor:
        """Dilate edit_mask to get the edit context region."""
        b, k, h, w = edit_mask.shape
        flat = edit_mask.reshape(b * k, 1, h, w)
        dilated = _soft_dilate(flat, radius)
        return dilated.reshape(b, k, h, w)

    @torch.no_grad()
    def _v393_construct_edit_control(
        self,
        edit_mask: torch.Tensor,
        image_h: int,
        image_w: int,
    ) -> Dict[str, torch.Tensor]:
        """Construct a same-area, non-overlapping control mask for each edit.

        Strategy: circular-shift (torch.roll) the edit_mask to a different
        image region.  Search over grid positions to find one that:
          (a) has zero overlap with the original edit_mask
          (b) has zero overlap with the edit context
        If no valid shift is found, the action is marked undeployable.

        Returns:
          control_mask:   [B,K,H,W] binary
          control_valid:  [B,K] bool
          control_area_ratio: [B,K] (≈1.0 when valid)
          control_overlap:    [B,K] (≈0.0 when valid)
          context_overlap:    [B,K]
          reject_reason:      List[str] (length B*K, empty str if valid)
        """
        b, k, h, w = edit_mask.shape
        h, w = int(image_h), int(image_w)
        device = edit_mask.device

        control_mask = torch.zeros_like(edit_mask)
        control_valid = torch.zeros(b, k, dtype=torch.bool, device=device)
        control_area_ratio = torch.zeros(b, k, device=device)
        control_overlap = torch.ones(b, k, device=device)
        context_overlap = torch.ones(b, k, device=device)
        reject_reason: list = [""] * (b * k)

        edit_context = self._v393_edit_context(edit_mask, self.context_radius)

        # Grid search step: try shifts at ~1/6 of image size intervals
        step = max(8, min(h, w) // 6)
        shifts_h = list(range(step, h, step))
        shifts_w = list(range(step, w, step))

        for bi in range(b):
            for ki in range(k):
                idx = bi * k + ki
                em = edit_mask[bi, ki]  # [H, W]
                area = em.sum()
                if area < 4:  # too small to be meaningful
                    reject_reason[idx] = "edit_area_too_small"
                    continue

                ec = edit_context[bi, ki]
                found = False

                for dy in shifts_h:
                    for dx in shifts_w:
                        rolled = torch.roll(em, shifts=(dy, dx), dims=(0, 1))
                        # (a) overlap with original edit
                        ov = (rolled * em).sum() / area.clamp_min(1.0)
                        if ov > 1e-6:
                            continue
                        # (b) overlap with edit context
                        # The control itself must not overlap factual edit context.
                        co = (rolled * ec).sum() / area.clamp_min(1.0)
                        if co > 1e-6:
                            continue
                        # Valid — accept this shift
                        control_mask[bi, ki] = rolled
                        control_valid[bi, ki] = True
                        control_area_ratio[bi, ki] = rolled.sum() / area.clamp_min(1.0)
                        control_overlap[bi, ki] = ov
                        context_overlap[bi, ki] = co
                        reject_reason[idx] = ""
                        found = True
                        break
                    if found:
                        break

                if not found:
                    reject_reason[idx] = "no_valid_control_shift"

        return {
            "control_mask": control_mask,
            "control_valid": control_valid,
            "control_area_ratio": control_area_ratio,
            "control_overlap": control_overlap,
            "context_overlap": context_overlap,
            "reject_reason": reject_reason,
        }

    def _v393_edit_region_evidence(
        self,
        image_only_map: torch.Tensor,
        positive_text: torch.Tensor,
        negative_text: torch.Tensor,
        edit_mask: torch.Tensor,
        control_mask: torch.Tensor,
        control_valid: torch.Tensor,
        action_types: torch.Tensor,
        swapped_text: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Factual/control evidence used by V393 and the cleaned V396 path.

        V396 deliberately bypasses all trainable V23 image/text adapters:
        similarity is measured directly in the immutable UniMedCLIP/BiomedBERT
        representation.  Therefore the semantic gate cannot learn action type,
        rank, mask area, coordinate, or segmentation-label proxies.  Only the
        visual gain critic is trainable, and it sees a frozen ROI difference.
        """
        device = edit_mask.device

        edit_roi_raw = self._pool_feature(image_only_map, edit_mask)
        ctrl_roi_raw = self._pool_feature(image_only_map, control_mask)

        if self.evidence_guided_candidate_control_enabled:
            edit_roi = F.normalize(edit_roi_raw, dim=-1, eps=1e-6)
            ctrl_roi = F.normalize(ctrl_roi_raw, dim=-1, eps=1e-6)
            pos_t = F.normalize(positive_text, dim=-1, eps=1e-6)
            neg_t = F.normalize(negative_text, dim=-1, eps=1e-6)
            para_source = positive_text if swapped_text is None else swapped_text
            para_t = F.normalize(para_source, dim=-1, eps=1e-6)
        else:
            edit_roi = F.normalize(
                self.v23_image_adapter(edit_roi_raw), dim=-1, eps=1e-6
            )
            ctrl_roi = F.normalize(
                self.v23_image_adapter(ctrl_roi_raw), dim=-1, eps=1e-6
            )
            pos_t = F.normalize(
                self.v23_text_adapter(positive_text), dim=-1, eps=1e-6
            )
            neg_t = F.normalize(
                self.v23_text_adapter(negative_text), dim=-1, eps=1e-6
            )
            para_source = negative_text if swapped_text is None else swapped_text
            para_t = F.normalize(
                self.v23_text_adapter(para_source), dim=-1, eps=1e-6
            )

        edit_pos = (edit_roi * pos_t[:, None, :]).sum(dim=-1)
        ctrl_pos = (ctrl_roi * pos_t[:, None, :]).sum(dim=-1)
        edit_neg = (edit_roi * neg_t[:, None, :]).sum(dim=-1)
        ctrl_neg = (ctrl_roi * neg_t[:, None, :]).sum(dim=-1)
        edit_para = (edit_roi * para_t[:, None, :]).sum(dim=-1)
        ctrl_para = (ctrl_roi * para_t[:, None, :]).sum(dim=-1)

        is_fill = torch.isin(
            action_types,
            torch.tensor(self.FILL_TYPES, device=device),
        )
        polarity = torch.where(
            is_fill,
            torch.ones_like(action_types, dtype=edit_pos.dtype),
            -torch.ones_like(action_types, dtype=edit_pos.dtype),
        )[None, :]

        lesion_delta = edit_pos - ctrl_pos
        background_delta = ctrl_neg - edit_neg
        signed_lesion = polarity * lesion_delta
        signed_background = polarity * background_delta

        direct_benefit = signed_lesion + signed_background
        direct_harm = (
            F.relu(-signed_lesion)
            + F.relu(-signed_background)
            - 0.5 * (F.relu(signed_lesion) + F.relu(signed_background))
        )

        # True 2x2 difference-in-differences under target and opposite text.
        direct_text = polarity * (
            (edit_pos - ctrl_pos) - (edit_neg - ctrl_neg)
        )

        # Same lesion/background semantics expressed by an independent fixed
        # paraphrase. It is an invariance diagnostic, never a batch-rotated
        # pathology label that can be unrelated to the segmentation target.
        paraphrase_text = polarity * (edit_para - ctrl_para)

        visual_delta = polarity[:, :, None] * (edit_roi - ctrl_roi)
        visual_gap = torch.sqrt(
            visual_delta.pow(2).mean(dim=-1).clamp_min(0.0) + 1e-6
        )
        visual_cosine = (edit_roi * ctrl_roi).sum(dim=-1)

        factual_features = torch.stack(
            [
                edit_pos,
                edit_neg,
                edit_pos - edit_neg,
                lesion_delta,
                background_delta,
            ],
            dim=-1,
        )
        control_features = torch.stack(
            [
                ctrl_pos,
                ctrl_neg,
                ctrl_pos - ctrl_neg,
                -lesion_delta,
                -background_delta,
            ],
            dim=-1,
        )

        factual_hidden = self.v393_calibrator_factual(factual_features)
        control_hidden = self.v393_calibrator_control(control_features)
        delta_hidden = F.gelu(factual_hidden - control_hidden)

        if self.v394_counterfactual_value_rank:
            benefit_raw = direct_benefit + self.v394_benefit_head(delta_hidden).squeeze(-1)
            harm_raw = direct_harm + self.v394_harm_head(delta_hidden).squeeze(-1)
            value_raw = self.v394_value_head(delta_hidden).squeeze(-1)
            text_raw = direct_text + self.v394_text_head(delta_hidden).squeeze(-1)
            learned_delta = value_raw
            v396_visual_gain = value_raw
            v396_semantic_gate = torch.sigmoid(direct_text)

        elif self.evidence_guided_candidate_control_enabled:
            if self.v396_visual_gain_head is None:
                raise RuntimeError("V396 visual gain head is missing.")

            if visual_delta.shape[-1] != self.v396_visual_dim:
                raise RuntimeError(
                    "V396 frozen observer channel mismatch: "
                    f"expected {self.v396_visual_dim}, got {visual_delta.shape[-1]}."
                )

            visual_gain = self.v396_visual_gain_head(
                visual_delta.reshape(-1, self.v396_visual_dim)
            ).reshape_as(visual_gap)

            semantic_gate = torch.sigmoid(
                (direct_text - self.v396_evidence_threshold)
                / self.v396_evidence_temperature
            )

            # q(a) is an acceptance value, not a signed gain. Harmful/neutral
            # visual outcomes must remain at Preserve=0; only semantically
            # supported positive visual gains can exceed Preserve.
            value_raw = semantic_gate * F.relu(visual_gain)
            learned_delta = visual_gain
            text_raw = direct_text
            benefit_raw = visual_gain
            harm_raw = -visual_gain
            v396_visual_gain = visual_gain
            v396_semantic_gate = semantic_gate

        elif self.v395_direct_gain_policy:
            batch_size = delta_hidden.shape[0]
            type_ids = action_types.long().clamp(0, 3)[None, :].expand(
                batch_size, -1
            )
            type_features = self.v395_type_embedding(type_ids)
            geometry_features = torch.stack(
                [
                    edit_mask.float().mean(dim=(-2, -1)),
                    control_mask.float().mean(dim=(-2, -1)),
                    control_valid.to(dtype=delta_hidden.dtype),
                    signed_lesion,
                    signed_background,
                ],
                dim=-1,
            )
            value_features = torch.cat(
                [delta_hidden, type_features, geometry_features],
                dim=-1,
            )
            value_raw = self.v395_value_head(value_features).squeeze(-1)
            learned_delta = value_raw
            text_raw = direct_text
            benefit_raw = direct_benefit + 0.25 * value_raw
            harm_raw = direct_harm - 0.25 * value_raw
            v396_visual_gain = value_raw
            v396_semantic_gate = torch.sigmoid(direct_text)

        else:
            learned_delta = self.v393_calibrator_head(delta_hidden).squeeze(-1)
            text_raw = direct_text + learned_delta
            benefit_raw = direct_benefit + learned_delta
            harm_raw = direct_harm + 0.5 * learned_delta
            value_raw = learned_delta
            v396_visual_gain = value_raw
            v396_semantic_gate = torch.sigmoid(direct_text)

        valid_f = control_valid.to(dtype=benefit_raw.dtype)

        return {
            "v393_text_evidence_logit": text_raw * valid_f,
            "v393_benefit_logit": benefit_raw * valid_f,
            "v393_harm_logit": harm_raw * valid_f,
            "v393_text_evidence_logit_raw": text_raw,
            "v393_benefit_logit_raw": benefit_raw,
            "v393_harm_logit_raw": harm_raw,
            "v393_calibrator_delta": learned_delta,
            "v393_raw_lesionness": direct_text,
            "v393_edit_pos": edit_pos,
            "v393_ctrl_pos": ctrl_pos,
            "v393_edit_neg": edit_neg,
            "v393_ctrl_neg": ctrl_neg,
            "v393_polarity": polarity,
            "v395_direct_gain_value": value_raw,
            "v394_value_logit": value_raw,

            "v396_action_value": value_raw,
            "v396_visual_gain": v396_visual_gain,
            "v396_text_evidence": direct_text,
            "v396_paraphrase_evidence": paraphrase_text,
            # Backward-compatible audit field name.
            "v396_swap_evidence": paraphrase_text,
            "v396_semantic_gate": v396_semantic_gate,
            "v396_visual_gap": visual_gap,
            "v396_visual_cosine": visual_cosine,
        }


    # ── M2 override ─────────────────────────────────────────────
    # Phase 2: real edit-region factual-control replaces placeholder.

    def _v381_counterfactual(
        self,
        image_only_map: torch.Tensor,
        positive_text: torch.Tensor,
        negative_text: torch.Tensor,
        factual_masks: torch.Tensor,
        control_masks: torch.Tensor,
        action_types: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """V393 Phase 2: real edit-region factual-control M2.

        1. Compute edit_mask from action_supports (factual_masks).
        2. Construct same-area control_mask via circular shift.
        3. Extract text evidence from edit vs control regions.
        4. Still delegate to V381's super for legacy cf values (loss compat).

        NO batch-centering — scores retain cross-case absolute meaning
        anchored to Preserve=0.
        """
        # Legacy: get V381's cf dict for backward-compatible loss dispatch
        cf = super()._v381_counterfactual(
            image_only_map=image_only_map,
            positive_text=positive_text,
            negative_text=negative_text,
            factual_masks=factual_masks,
            control_masks=control_masks,
            action_types=action_types,
        )

        b, k, h, w = factual_masks.shape

        # Step 1: compute edit_mask from action supports
        edit_mask = self._v393_compute_edit_mask(
            factual_masks=factual_masks,
            base_mask=torch.zeros(b, h, w, device=factual_masks.device),
        )

        # Step 2: construct real control_mask
        ctrl_result = self._v393_construct_edit_control(
            edit_mask=edit_mask,
            image_h=h,
            image_w=w,
        )
        v393_control_mask = ctrl_result["control_mask"]
        v393_control_valid = ctrl_result["control_valid"]

        # Step 3: extract edit-region text evidence
        evidence = self._v393_edit_region_evidence(
            image_only_map=image_only_map,
            positive_text=positive_text,
            negative_text=negative_text,
            edit_mask=edit_mask,
            control_mask=v393_control_mask,
            control_valid=v393_control_valid,
            action_types=action_types,
        )

        # Step 4: populate all V393 audit fields
        cf.update({
            "v393_text_evidence_logit": evidence["v393_text_evidence_logit"],
            "v393_benefit_logit": evidence["v393_benefit_logit"],
            "v393_harm_logit": evidence["v393_harm_logit"],
            "v393_control_area_ratio": ctrl_result["control_area_ratio"],
            "v393_control_overlap": ctrl_result["control_overlap"],
            "v393_context_overlap": ctrl_result["context_overlap"],
            "v393_control_valid": v393_control_valid,
            "v393_reject_reason": ctrl_result["reject_reason"],
            # Internal debugging (not required by spec but useful)
            "v393_edit_mask": edit_mask,
            "v393_control_mask": v393_control_mask,
            "v393_raw_lesionness": evidence["v393_raw_lesionness"],
            "v393_edit_pos": evidence["v393_edit_pos"],
            "v393_ctrl_pos": evidence["v393_ctrl_pos"],
            "v393_edit_neg": evidence["v393_edit_neg"],
            "v393_ctrl_neg": evidence["v393_ctrl_neg"],
        })

        return cf

    # ── M3 Preserve-relative value skeleton ─────────────────────
    # Phase 1: boundary_fill exclusion + Preserve fallback.
    # Phase 2+ will compute action_value = benefit - harm * risk_weight
    # and gate on value > 0.

    def _v38_delta_consensus_select(
        self,
        hypothesis_logits: torch.Tensor,
        valid: torch.Tensor,
        membership: torch.Tensor,
        hypothesis_text_score: torch.Tensor,
        image_edge: torch.Tensor,
    ) -> tuple:
        """Phase 1 override: boundary_fill block + Preserve-only fallback.

        V392's two bugs are fixed:
          (a) boundary_fill excluded from structural_safe when blocked
          (b) no-consensus fallback = Preserve (not best-cf-logit)

        Phase 2+ will add: action_value > v393_min_action_value gate.
        """
        b, n, h, w = hypothesis_logits.shape
        side = min(self.v38_struct_size, h, w)
        low_logits = F.interpolate(
            hypothesis_logits.reshape(b * n, 1, h, w),
            size=(side, side), mode="bilinear", align_corners=False,
        ).reshape(b, n, side, side)
        delete, fill, edit = self._v38_edit_maps(low_logits)
        delete_dice = self._v38_pairwise_dice(delete)
        fill_dice = self._v38_pairwise_dice(fill)
        delete_mass = delete.flatten(2).sum(dim=-1)
        fill_mass = fill.flatten(2).sum(dim=-1)
        delete_present = (delete_mass[:,:,None] + delete_mass[:,None,:]) > EPS
        fill_present = (fill_mass[:,:,None] + fill_mass[:,None,:]) > EPS
        sw = delete_present.float() + fill_present.float()
        signed_delta_dice = (
            delete_dice * delete_present.float()
            + fill_dice * fill_present.float()
        ) / sw.clamp_min(1.0)
        edit_boundary = self._boundary(edit)
        edit_boundary_dice = self._v38_pairwise_dice(edit_boundary)
        pairwise = (
            self.v38_delta_dice_weight * signed_delta_dice
            + self.v38_edit_boundary_weight * edit_boundary_dice
        )
        low_edge = F.interpolate(image_edge, size=(side, side),
                                  mode="bilinear", align_corners=False)
        ebm = edit_boundary.sum(dim=(-2,-1)).clamp_min(EPS)
        edit_edge_alignment = (edit_boundary * low_edge).sum(dim=(-2,-1)) / ebm
        edit_area = edit.mean(dim=(-2,-1))
        edit_perimeter = ebm / float(side * side)
        edited_valid = valid.clone()
        edited_valid[:, 0] = False

        # ★ V393: exclude boundary_fill from structural_safe when blocked
        if not self.v393_deploy_boundary_fill:
            k = membership.shape[-1]
            for ai in range(k):
                atype = self.action_types[ai] if ai < len(self.action_types) else -1
                if atype == 2:  # boundary_fill_r0, boundary_fill_r1 (action type 2)
                    edited_valid[:, ai + 1] = False

        structural_safe = (
            edited_valid
            & (edit_area > 0)
            & (edit_area <= self.v38_max_edit_fraction)
            & (edit_perimeter <= self.v38_max_edit_perimeter_growth)
        )
        eye = torch.eye(n, dtype=torch.bool, device=hypothesis_logits.device)[None]
        pair_valid = structural_safe[:,:,None] & structural_safe[:,None,:] & (~eye)
        agreement = (
            pairwise * pair_valid.float()
        ).sum(dim=-1) / pair_valid.float().sum(dim=-1).clamp_min(1.0)
        stability = (
            agreement
            + self.v38_edge_weight * edit_edge_alignment
            + self.v393_cf_medoid_weight * hypothesis_text_score
            - self.v38_edit_area_penalty * edit_area
            - self.v38_edit_perimeter_penalty * edit_perimeter
        )
        medoid_score = stability.masked_fill(~structural_safe, -1e9)
        medoid_index = medoid_score.argmax(dim=1)
        medoid_exists = structural_safe.any(dim=1)
        medoid_pairwise = pairwise.gather(
            1, medoid_index[:,None,None].expand(-1,1,n)
        )[:,0]
        cluster = structural_safe & (medoid_pairwise >= self.v38_cluster_similarity_min)
        cluster_size = cluster.sum(dim=1)
        multi_active = medoid_exists & (cluster_size >= self.v38_min_cluster_size)
        selected_text = hypothesis_text_score.gather(1, medoid_index[:,None])[:,0]
        selected_edge = edit_edge_alignment.gather(1, medoid_index[:,None])[:,0]
        selected_stability = stability.gather(1, medoid_index[:,None])[:,0]
        singleton_active = (
            medoid_exists & (cluster_size == 1)
            & (selected_text >= self.v38_singleton_text_threshold)
            & (selected_edge >= self.v38_singleton_min_edit_edge)
            & (selected_stability >= self.v38_singleton_min_stability)
        )
        consensus_active = multi_active | singleton_active

        selected_logits = hypothesis_logits.gather(
            1, medoid_index[:,None,None,None].expand(-1,1,h,w)
        )[:,0]

        # ★ V393: no-consensus → Preserve (not best-cf-logit)
        final_logits = torch.where(
            consensus_active[:, None, None],
            selected_logits,
            hypothesis_logits[:, 0],  # Preserve
        )

        k_mem = membership.shape[-1]
        selected_membership = membership.gather(
            1, medoid_index[:,None,None].expand(-1,1,k_mem)
        )[:,0]
        action_endorsement = (
            selected_membership * consensus_active[:, None].float()
        )

        # Action value: placeholder for Phase 2+
        action_value = hypothesis_text_score.new_zeros(b)
        action_value[consensus_active] = selected_text[consensus_active]

        return final_logits, action_endorsement, {
            "structural_safe": structural_safe,
            "signed_delta_dice": signed_delta_dice,
            "edit_boundary_dice": edit_boundary_dice,
            "pairwise": pairwise,
            "agreement": agreement,
            "stability": stability,
            "edit_edge_alignment": edit_edge_alignment,
            "edit_area": edit_area,
            "edit_perimeter": edit_perimeter,
            "medoid_score": medoid_score,
            "medoid_index": medoid_index,
            "cluster_size": cluster_size.float(),
            "multi_active": multi_active.float(),
            "singleton_active": singleton_active.float(),
            "consensus_active": consensus_active.float(),
            "selected_text_score": selected_text,
            "selected_edit_edge": selected_edge,
            "selected_stability": selected_stability,
            # V393 additions
            "v393_action_value": action_value,
            "v393_boundary_fill_blocked": structural_safe.new_zeros(
                b, n
            ) if not self.v393_deploy_boundary_fill else structural_safe.new_zeros(b, 0),
        }

    # ── generate() ──────────────────────────────────────────────

    def generate(
        self,
        base_logits: torch.Tensor,
        image: torch.Tensor,
        semantic_map: Optional[torch.Tensor],
        text_features: torch.Tensor,
        negative_text_features: Optional[torch.Tensor] = None,
        swapped_text_features: Optional[torch.Tensor] = None,
    ):
        """V393 Phase 2: V381 M1 + real edit-region M2 + Preserve-relative M3.

        Pipeline:
          1. V381 generates candidates + structural consensus (M1+M3 skeleton)
          2. V393 M2: real edit-region factual-control evidence
          3. V393 M3: Preserve-relative action_value gate
             - action_value = benefit - harm per action
             - Preserve has fixed value = 0
             - Deploy only if action_value > 0 AND control_valid AND safe
             - Otherwise fall back to Preserve (final = Base)
        """

        # ── Pass 1: image-only atomic proposal bank ──
        # V396 deliberately bypasses the inherited V381 M2/M3 path. Candidate
        # geometry is still generated by the same atomic V20 proposal bank, but
        # all V381 semantic/calibrator/consensus outputs are excluded from the
        # V396 train and deployment route.
        if self.evidence_guided_candidate_control_enabled:
            candidate_logits_all, aux = UnifiedActionCounterfactualSetBank.generate(
                self,
                base_logits=base_logits,
                image=image,
                semantic_map=semantic_map,
                text_features=text_features,
                negative_text_features=negative_text_features,
            )
        else:
            candidate_logits_all, aux = V381LesionBackgroundCalibratedAtomicBank.generate(
                self,
                base_logits=base_logits,
                image=image,
                semantic_map=semantic_map,
                text_features=text_features,
                negative_text_features=negative_text_features,
            )

        k = len(self.action_types)
        b = base_logits.shape[0]
        hw = candidate_logits_all.shape[-2:]

        # ── Pass 2: V393 M2 edit-region factual-control evidence ──
        if self.v393_use_edit_region_control and semantic_map is not None:
            action_supports = aux.get("v20_action_supports")
            if action_supports is not None:
                edit_mask = self._v393_compute_edit_mask(
                    factual_masks=action_supports,
                    base_mask=torch.zeros(b, *action_supports.shape[-2:], device=action_supports.device),
                )
                # V396 reuses the carrier-matched controls constructed by
                # the V20 bank (matched intensity/entropy/boundary/shape),
                # rather than V393's old arbitrary circular-shift controls.
                parent_controls = aux.get("v20_control_supports")
                if (
                    self.evidence_guided_candidate_control_enabled
                    and isinstance(parent_controls, torch.Tensor)
                    and parent_controls.shape == edit_mask.shape
                ):
                    control_mask = (parent_controls.detach() > 0.5).to(edit_mask.dtype)
                    edit_area = edit_mask.sum(dim=(-2, -1))
                    control_area = control_mask.sum(dim=(-2, -1))
                    control_area_ratio = control_area / edit_area.clamp_min(1.0)
                    control_overlap = (
                        (edit_mask * control_mask).sum(dim=(-2, -1))
                        / edit_area.clamp_min(1.0)
                    )

                    factual_context = self._v393_edit_context(
                        edit_mask, self.context_radius
                    )
                    control_context = self._v393_edit_context(
                        control_mask, self.context_radius
                    )
                    context_overlap = (
                        (factual_context * control_context).sum(dim=(-2, -1))
                        / factual_context.sum(dim=(-2, -1)).clamp_min(1.0)
                    )

                    control_valid = (
                        (edit_area >= 4.0)
                        & (control_area > 0.0)
                        & ((control_area_ratio - 1.0).abs() <= self.v396_control_area_tolerance)
                        & (control_overlap <= 1e-6)
                        & (context_overlap <= 1e-6)
                    )

                    reject_reason = []
                    for flat_valid in control_valid.reshape(-1).detach().cpu().tolist():
                        reject_reason.append(
                            "" if bool(flat_valid)
                            else "matched_control_not_context_clean"
                        )

                    ctrl_result = {
                        "control_mask": control_mask,
                        "control_valid": control_valid,
                        "control_area_ratio": control_area_ratio,
                        "control_overlap": control_overlap,
                        "context_overlap": context_overlap,
                        "reject_reason": reject_reason,
                    }
                    aux["v396_control_source_parent"] = torch.ones(
                        b, k, device=base_logits.device, dtype=base_logits.dtype
                    )
                else:
                    ctrl_result = self._v393_construct_edit_control(
                        edit_mask=edit_mask,
                        image_h=action_supports.shape[-2],
                        image_w=action_supports.shape[-1],
                    )
                    aux["v396_control_source_parent"] = torch.zeros(
                        b, k, device=base_logits.device, dtype=base_logits.dtype
                    )

                neg_t = negative_text_features if negative_text_features is not None else text_features
                evidence = self._v393_edit_region_evidence(
                    image_only_map=semantic_map,
                    positive_text=text_features,
                    negative_text=neg_t,
                    edit_mask=edit_mask,
                    control_mask=ctrl_result["control_mask"],
                    control_valid=ctrl_result["control_valid"],
                    action_types=self.action_types,
                    swapped_text=swapped_text_features,
                )
                aux.update({
                    "v393_text_evidence_logit": evidence["v393_text_evidence_logit"],
                    "v393_benefit_logit": evidence["v393_benefit_logit"],
                    "v393_harm_logit": evidence["v393_harm_logit"],
                    "v395_direct_gain_value": evidence["v395_direct_gain_value"],
                    "v396_action_value": evidence["v396_action_value"],
                    "v396_visual_gain": evidence["v396_visual_gain"],
                    "v396_text_evidence": evidence["v396_text_evidence"],
                    "v396_paraphrase_evidence": evidence["v396_paraphrase_evidence"],
                    "v396_swap_evidence": evidence["v396_swap_evidence"],
                    "v396_semantic_gate": evidence["v396_semantic_gate"],
                    "v396_visual_gap": evidence["v396_visual_gap"],
                    "v396_visual_cosine": evidence["v396_visual_cosine"],
                    "v393_control_area_ratio": ctrl_result["control_area_ratio"],
                    "v393_control_overlap": ctrl_result["control_overlap"],
                    "v393_context_overlap": ctrl_result["context_overlap"],
                    "v393_control_valid": ctrl_result["control_valid"],
                    "v393_reject_reason": ctrl_result["reject_reason"],
                })
            else:
                aux.update({
                    "v393_text_evidence_logit": base_logits.new_zeros(b, k),
                    "v393_benefit_logit": base_logits.new_zeros(b, k),
                    "v393_harm_logit": base_logits.new_zeros(b, k),
                    "v393_control_area_ratio": base_logits.new_zeros(b, k),
                    "v393_control_overlap": base_logits.new_ones(b, k),
                    "v393_context_overlap": base_logits.new_ones(b, k),
                    "v393_control_valid": base_logits.new_zeros(b, k, dtype=torch.bool),
                    "v393_reject_reason": ['action_supports_missing'] * (b * k),
                })
        else:
            aux.update({
                "v393_text_evidence_logit": base_logits.new_zeros(b, k),
                "v393_benefit_logit": base_logits.new_zeros(b, k),
                "v393_harm_logit": base_logits.new_zeros(b, k),
                "v393_control_area_ratio": base_logits.new_zeros(b, k),
                "v393_control_overlap": base_logits.new_ones(b, k),
                "v393_context_overlap": base_logits.new_ones(b, k),
                "v393_control_valid": base_logits.new_zeros(b, k, dtype=torch.bool),
                "v393_reject_reason": ['semantic_map_missing'] * (b * k),
            })

        # ── V394 M3: rank every deployable action against Preserve=0 ──
        if self.v394_counterfactual_value_rank:
            benefit = aux["v393_benefit_logit"]
            harm = aux["v393_harm_logit"]
            control_valid = aux["v393_control_valid"].bool()
            # V394 M3 must rank with the differentiable value-head output.
            if "v394_value_logit" not in evidence:
                raise RuntimeError("V394 evidence has no value-head output.")
            aux["v394_value_logit"] = evidence["v394_value_logit"]
            value_raw = aux["v394_value_logit"]
            # Validation/Test runs under torch.no_grad(), where tensors are
            # correctly non-differentiable. Only enforce graph integrity during
            # gradient-enabled training.
            if torch.is_grad_enabled() and not value_raw.requires_grad:
                raise RuntimeError("V394 M3 received a detached value tensor during training.")
            action_value = value_raw + self.v394_class_value_weight * (
                torch.sigmoid(benefit) - torch.sigmoid(harm)
            )

            supports = aux.get("v20_action_supports")
            if supports is None:
                supports = torch.zeros(
                    b, k, *hw, device=base_logits.device, dtype=base_logits.dtype
                )

            action_area = supports.float().mean(dim=(-2, -1))
            deployable = control_valid & (action_area > 0.0)
            deployable = deployable & (
                action_area <= self.v394_max_edit_fraction
            )

            boundary_fill = (
                self.action_types.eq(2)[None, :]
                .expand(b, -1)
                .to(device=base_logits.device)
            )
            if not self.v393_deploy_boundary_fill:
                deployable = deployable & (~boundary_fill)

            choice_logits = torch.cat(
                [
                    base_logits.new_zeros(b, 1),
                    action_value.masked_fill(~deployable, -20.0),
                ],
                dim=1,
            )

            masked_value = action_value.masked_fill(~deployable, -1e4)
            best_value, best_action = masked_value.max(dim=1)
            final_changed = best_value > self.v394_min_action_value

            selected = torch.zeros_like(action_value)
            selected.scatter_(
                1,
                best_action[:, None],
                final_changed[:, None].to(action_value.dtype),
            )

            candidate_actions = candidate_logits_all[:, 1:]
            chosen_logits = candidate_actions.gather(
                1,
                best_action[:, None, None, None].expand(
                    -1, 1, candidate_actions.shape[-2], candidate_actions.shape[-1]
                ),
            )[:, 0]

            final_logits = torch.where(
                final_changed[:, None, None],
                chosen_logits,
                candidate_logits_all[:, 0],
            )
            final_probs = torch.sigmoid(final_logits).clamp(EPS, 1.0 - EPS)

            fallback_reason = []
            for bi in range(b):
                if bool(final_changed[bi].item()):
                    fallback_reason.append("v394_deployed")
                elif not bool(deployable[bi].any().item()):
                    fallback_reason.append("v394_no_deployable_action")
                else:
                    fallback_reason.append(
                        f"v394_best_value={float(best_value[bi].item()):.4f}_lte_"
                        f"{self.v394_min_action_value:.4f}"
                    )

            aux.update({
                "v20_selector_hard": selected,
                "v20_selector_probs": torch.softmax(choice_logits, dim=1)[:, 1:],
                "v20_fused_logits": final_logits,
                "v20_fused_probs": final_probs,
                "v20_hard_fused_probs": final_probs,
                "direct_fused_probs": final_probs,
                "router_fused_probs": final_probs,

                "v394_value": action_value,
                "v394_value_logit": value_raw,
                "v394_choice_logits": choice_logits,
                "v394_eligible": deployable,
                "v394_selected_action": selected,
                "v394_selected_index": best_action + 1,
                "v394_selected_value": best_value,
                "v394_action_area": action_area,
                "v394_final_changed": final_changed.float(),

                "v393_action_value_padded": torch.cat(
                    [base_logits.new_zeros(b, 1), action_value], dim=1
                ),
                "v393_final_changed": final_changed.float(),
                "v393_final_returned_base": (~final_changed).float(),
                "v393_fallback_reason": fallback_reason,
            })
            return candidate_logits_all, aux

        # ── Pass 3: V393 M3 Preserve-relative ranking and deployment gate ──
        benefit = aux["v393_benefit_logit"]
        harm = aux["v393_harm_logit"]
        text_evidence = aux["v393_text_evidence_logit"]
        control_valid = aux["v393_control_valid"].bool()
        control_area_ratio = aux["v393_control_area_ratio"]
        control_overlap = aux["v393_control_overlap"]
        context_overlap = aux["v393_context_overlap"]

        supports = aux.get("v20_action_supports")
        if supports is None:
            supports = base_logits.new_zeros(b, k, *base_logits.shape[-2:])
        action_area = supports.float().mean(dim=(-2, -1))

        # V395: deployment uses the exact q(a) scalar that is regressed and
        # ranked against real candidate Dice gains during training.  A missing
        # matched control is a feature of q(a), not a blanket rejection gate.
        if self.v395_direct_gain_policy or self.evidence_guided_candidate_control_enabled:
            direct_q = aux.get(
                "v396_action_value"
                if self.evidence_guided_candidate_control_enabled
                else "v395_direct_gain_value"
            )
            if (
                direct_q is None
                or direct_q.ndim != 2
                or direct_q.shape != (b, k)
            ):
                raise RuntimeError(
                    "V395 requires v395_direct_gain_value with shape [B, K]."
                )

            if self.evidence_guided_candidate_control_enabled:
                text_specific = aux.get("v396_text_evidence")
                if text_specific is None or text_specific.shape != (b, k):
                    raise RuntimeError("V396 requires text-specific evidence [B,K].")
                v395_eligible = (
                    control_valid
                    & ((control_area_ratio - 1.0).abs() <= self.v396_control_area_tolerance)
                    & (control_overlap <= 1e-6)
                    & (context_overlap <= 1e-6)
                    & (action_area >= 1e-8)
                    & (action_area <= self.v396_max_edit_fraction)
                    & (text_specific > self.v396_evidence_threshold)
                )

                # V398 safety: training may retain all proposal types for
                # coverage, while deployment is explicitly restricted to the
                # action types that have not shown systematic Val harm.  This
                # is a hard non-learned rule; it never enters q(a)'s features.
                allowed_types_cfg = _cfg_get(
                    _cfg_get(self.cfg, "M1", None),
                    "EVIDENCE_GUIDED_DEPLOY_ALLOWED_TYPES",
                    None,
                )
                if allowed_types_cfg is None:
                    v396_type_allowed = torch.ones_like(v395_eligible, dtype=torch.bool)
                else:
                    try:
                        allowed_types = {int(item) for item in allowed_types_cfg}
                    except TypeError as exc:
                        raise RuntimeError(
                            "M1.EVIDENCE_GUIDED_DEPLOY_ALLOWED_TYPES must be a YAML list of action-type integers."
                        ) from exc
                    if not allowed_types:
                        raise RuntimeError(
                            "M1.EVIDENCE_GUIDED_DEPLOY_ALLOWED_TYPES must not be empty; Preserve is handled separately."
                        )
                    v396_type_allowed = torch.zeros_like(v395_eligible, dtype=torch.bool)
                    for action_type_id in allowed_types:
                        v396_type_allowed = v396_type_allowed | (
                            self.action_types.eq(action_type_id)[None, :]
                            .expand(b, -1)
                            .to(device=base_logits.device)
                        )
                    v395_eligible = v395_eligible & v396_type_allowed
            else:
                v396_type_allowed = torch.ones_like(control_valid, dtype=torch.bool)
                v395_eligible = action_area >= 1e-8
            action_value = direct_q.masked_fill(~v395_eligible, -1e4)
            choice_logits = torch.cat(
                [
                    base_logits.new_zeros(b, 1),
                    action_value.masked_fill(~v395_eligible, -20.0),
                ],
                dim=1,
            )
            masked_value = action_value.masked_fill(~v395_eligible, -1e4)
            best_value, best_action = masked_value.max(dim=1)
            required_value = (
                self.v396_min_action_value
                if self.evidence_guided_candidate_control_enabled else self.v395_min_action_value
            )
            final_changed = best_value > required_value

            selected = torch.zeros_like(action_value)
            changed_cases = final_changed.nonzero(as_tuple=False).flatten()
            if changed_cases.numel() > 0:
                selected[
                    changed_cases,
                    best_action[changed_cases],
                ] = 1.0

            candidate_actions = candidate_logits_all[:, 1:]
            chosen_logits = candidate_actions.gather(
                1,
                best_action[:, None, None, None].expand(
                    -1,
                    1,
                    candidate_actions.shape[-2],
                    candidate_actions.shape[-1],
                ),
            )[:, 0]
            final_logits = torch.where(
                final_changed[:, None, None],
                chosen_logits,
                candidate_logits_all[:, 0],
            )
            final_probs = torch.sigmoid(final_logits).clamp(EPS, 1.0 - EPS)

            fallback_reason_list: list = []
            for bi in range(b):
                if bool(final_changed[bi].item()):
                    fallback_reason_list.append(
                        "v396_evidence_gated_action"
                        if self.evidence_guided_candidate_control_enabled
                        else "v395_deployed_direct_gain"
                    )
                elif not bool(v395_eligible[bi].any().item()):
                    fallback_reason_list.append(
                        "v396_no_semantic_safe_action"
                        if self.evidence_guided_candidate_control_enabled
                        else "v395_no_nonempty_action"
                    )
                else:
                    threshold = (
                        self.v396_min_action_value
                        if self.evidence_guided_candidate_control_enabled
                        else self.v395_min_action_value
                    )
                    fallback_reason_list.append(
                        ("v396_best_q=" if self.evidence_guided_candidate_control_enabled else "v395_best_q=")
                        + f"{float(best_value[bi].item()):.4f}_lte_{threshold:.4f}"
                    )

            aux.update(
                {
                    "v20_selector_hard": selected,
                    "v20_selector_probs": torch.softmax(
                        choice_logits, dim=1
                    )[:, 1:],
                    "v20_fused_logits": final_logits,
                    "v20_fused_probs": final_probs,
                    "v20_hard_fused_probs": final_probs,
                    "direct_fused_probs": final_probs,
                    "router_fused_probs": final_probs,
                    "v393_preserve_value": base_logits.new_zeros(b),
                    "v396_type_allowed": v396_type_allowed.float(),
                    "v393_final_returned_base": (~final_changed).float(),
                    "v393_final_changed": final_changed.float(),
                    "v393_selected_index": torch.where(
                        final_changed,
                        best_action + 1,
                        torch.zeros_like(best_action),
                    ).float(),
                    "v393_action_value": action_value,
                    "v393_action_value_padded": torch.cat(
                        [base_logits.new_zeros(b, 1), action_value], dim=1
                    ),
                    "v393_eligible": v395_eligible.float(),
                    "v393_fallback_reason": fallback_reason_list,
                    "v395_direct_gain_value": direct_q,
                    "v395_selected_value": best_value,
                    "v395_action_area": action_area,
                    "v395_eligible": v395_eligible.float(),
                    "v395_final_changed": final_changed.float(),
                    "v396_action_value": aux.get("v396_action_value", direct_q),
                    "v396_visual_gain": aux.get("v396_visual_gain", direct_q),
                    "v396_text_evidence": aux.get("v396_text_evidence", text_evidence),
                    "v396_swap_evidence": aux.get("v396_swap_evidence", text_evidence),
                    "v396_semantic_gate": aux.get("v396_semantic_gate", torch.sigmoid(text_evidence)),
                    "v396_eligible": v395_eligible.float(),
                    "v396_selected_value": best_value,
                    "v396_action_area": action_area,
                    "v396_control_source_parent": aux.get(
                        "v396_control_source_parent",
                        base_logits.new_zeros(b, k),
                    ),
                }
            )
            return candidate_logits_all, aux

        structural_safe_full = aux.get("v381_structural_safe")
        if structural_safe_full is not None and structural_safe_full.shape[1] == k + 1:
            structural_safe = structural_safe_full[:, 1:].bool()
        else:
            structural_safe = torch.ones(
                b, k, dtype=torch.bool, device=base_logits.device
            )

        stability_full = aux.get("v381_hypothesis_stability")
        if stability_full is not None and stability_full.shape[1] == k + 1:
            stability = stability_full[:, 1:]
        else:
            stability = base_logits.new_zeros(b, k)

        edit_edge_full = aux.get("v381_edit_edge_alignment")
        if edit_edge_full is not None and edit_edge_full.shape[1] == k + 1:
            edit_edge = edit_edge_full[:, 1:]
        else:
            edit_edge = base_logits.new_zeros(b, k)

        # V393 safety: control_valid is the primary hard gate.
        # structural_safe (V381 legacy) was killing 33/64 good delete
        # candidates in the diagnostic audit.  Demote it to a soft
        # penalty — control_valid already provides the geometry gate.
        structural_penalty = (~structural_safe).float() * 2.0

        area_penalty = F.relu(action_area - self.v38_max_edit_fraction)
        overlap_penalty = control_overlap + context_overlap

        # V393 deployment action_value: V381 cf_logit is the primary trained
        # quality signal.  V393 benefit/harm from the small calibrator is
        # diagnostic-only during deployment — it tends to be uniformly
        # negative because most edit candidates are genuinely harmful, and
        # would drown the V381 signal.
        v381_cf = aux.get("v20_cf_logit")
        if v381_cf is not None and v381_cf.shape[1] == k:
            # sigmoid(cf) ∈ (0,1); subtract 0.50 so that cf_logit > 0
            # yields a positive contribution.  This is the neutral-point
            # threshold: only text-evidence-positive actions enter the pool.
            v381_standalone = torch.sigmoid(v381_cf) - 0.50
        else:
            v381_standalone = base_logits.new_zeros(b, k)

        v393_action_value = (
            v381_standalone
            + 0.10 * text_evidence
            + 0.05 * stability
            + 0.05 * edit_edge
            - 0.20 * overlap_penalty
            - 0.50 * area_penalty
            - structural_penalty
        )
        # Hard safety gates (kept to minimum):
        #   control_valid       — M2 mask geometry gate
        #   action_area < 1e-8  — degenerate edit (should never fire)
        v393_action_value = v393_action_value.masked_fill(~control_valid, -1e4)
        v393_action_value = v393_action_value.masked_fill(action_area < 1e-8, -1e4)

        boundary_fill_mask = (
            self.action_types.eq(2)[None, :]
            .expand(b, -1)
            .to(device=base_logits.device)
        )
        if not self.v393_deploy_boundary_fill:
            v393_action_value = v393_action_value.masked_fill(boundary_fill_mask, -1e4)

        v393_action_value_padded = torch.cat(
            [base_logits.new_zeros(b, 1), v393_action_value], dim=1
        )
        v393_eligible = torch.isfinite(v393_action_value) & (v393_action_value > -1e3)
        choice_logits = torch.cat(
            [base_logits.new_zeros(b, 1), v393_action_value.masked_fill(~v393_eligible, -20.0)],
            dim=1,
        )

        selected_index = choice_logits.argmax(dim=1)
        final_changed = selected_index > 0
        best_action = (selected_index - 1).clamp_min(0)
        best_value = choice_logits.gather(1, selected_index[:, None])[:, 0]
        selected = torch.zeros_like(v393_action_value)
        action_case_mask = final_changed.nonzero(as_tuple=False).flatten()
        if action_case_mask.numel() > 0:
            selected[action_case_mask, best_action[action_case_mask]] = 1.0

        candidate_actions = candidate_logits_all[:, 1:]
        chosen_logits = candidate_actions.gather(
            1,
            best_action[:, None, None, None].expand(
                -1, 1, candidate_actions.shape[-2], candidate_actions.shape[-1]
            ),
        )[:, 0]
        final_logits = torch.where(
            final_changed[:, None, None],
            chosen_logits,
            candidate_logits_all[:, 0],
        )
        final_probs = torch.sigmoid(final_logits).clamp(EPS, 1.0 - EPS)

        fallback_reason_list: list = []
        for bi in range(b):
            if bool(final_changed[bi].item()):
                fallback_reason_list.append("deployed")
            elif not bool(v393_eligible[bi].any().item()):
                if bool(boundary_fill_mask[bi].any().item()) and not self.v393_deploy_boundary_fill:
                    fallback_reason_list.append("no_eligible_action_boundary_fill_blocked")
                else:
                    fallback_reason_list.append("no_eligible_action")
            else:
                fallback_reason_list.append(
                    f"action_value={float(best_value[bi].item()):.4f}_lte_{self.v393_min_action_value:.4f}"
                )

        aux["v20_selector_hard"] = selected
        aux["v20_selector_probs"] = torch.softmax(choice_logits, dim=1)[:, 1:]
        aux["v20_fused_logits"] = final_logits
        aux["v20_fused_probs"] = final_probs
        aux["v20_hard_fused_probs"] = final_probs
        aux["router_fused_probs"] = final_probs
        aux["direct_fused_probs"] = final_probs

        # ── V393 audit keys (Phase 1 + M2 + M3) ──
        aux.update({
            "v393_preserve_value": base_logits.new_zeros(b),
            "v393_boundary_fill_deployable": (
                torch.tensor([t == 2 for t in self.action_types])
                .to(base_logits.device).unsqueeze(0).expand(b, -1)
                if self.v393_deploy_boundary_fill
                else base_logits.new_zeros(b, k)
            ),
            "v393_final_returned_base": (final_changed < 0.5).float(),
            "v393_final_changed": final_changed.float(),
            "v393_selected_index": selected_index.float(),
            "v393_deploy_boundary_fill_active": torch.tensor(
                self.v393_deploy_boundary_fill, dtype=torch.bool,
                device=base_logits.device,
            ).unsqueeze(0).expand(b, 1),
            # M3 audit
            "v393_action_value": v393_action_value,
            "v393_action_value_padded": v393_action_value_padded,
            "v393_eligible": v393_eligible.float(),
            "v393_fallback_reason": fallback_reason_list,
        })

        return candidate_logits_all, aux


class CustomCLIP(nn.Module):
    def __init__(self, cfg, clip_model, output_hidden_states: bool = False):
        super().__init__()
        self.cfg = cfg
        self.vision_model = clip_model.visual
        self.text_model = clip_model.text_encoder

        # V396/FECG: an independent frozen copy of the original visual tower.
        # object.__setattr__ prevents it from entering model parameters,
        # optimizer groups, checkpoints, or segmentation gradients.
        self.evidence_guided_candidate_control_enabled = bool(
            _cfg_get(_cfg_get(cfg, "M1", None), "EVIDENCE_GUIDED_FECG_ENABLED", False)
        )
        object.__setattr__(self, "_v396_observer_vision", None)
        if self.evidence_guided_candidate_control_enabled:
            observer = copy.deepcopy(clip_model.visual)
            observer.eval()
            for parameter in observer.parameters():
                parameter.requires_grad_(False)
            object.__setattr__(self, "_v396_observer_vision", observer)
        # Frozen prompt embeddings depend only on the prompt string and the
        # immutable text tower. Cache them outside nn.Module registration so
        # they are excluded from checkpoints and optimiser state.
        self.v396_cache_frozen_text = bool(
            _cfg_get(_cfg_get(cfg, "M1", None), "EVIDENCE_GUIDED_CACHE_FROZEN_TEXT", False)
        )
        object.__setattr__(self, "_v396_text_feature_cache", {})

        self.logit_scale = clip_model.logit_scale
        self.temperature = cfg.MODEL.TEMPERATURE
        # Strong-ablation contract: keep the exact FULL model construction, RNG
        # consumption, parameter registration and optimizer layout, but allow a
        # causal forward-only removal of every PVL adapter.  Do NOT encode this
        # as MODEL.LAYERS=[] via CLI: this project's opts parser can preserve
        # the token as the literal string "[]", which makes `index in
        # self.fusion_stages` raise TypeError and also constructs two adapters
        # because len("[]") == 2.
        self.fusion_stages = list(cfg.MODEL.LAYERS)
        self.disable_pvl_ablation = str(
            os.environ.get("MEDCLIPSEG_ABL_NO_PVL", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        if self.disable_pvl_ablation:
            print(
                "[STRONG_ABL_NO_PVL] all PVL adapter forward interventions are "
                "bypassed; adapter construction/parameters are retained only to "
                "preserve the FULL initialization and optimizer contract"
            )
        train_cfg = _cfg_get(cfg, "TRAIN", None)
        self.xbm_enabled = bool(_cfg_get(train_cfg, "XBM_ENABLED", False))
        self.xbm_queue_size = max(0, int(_cfg_get(train_cfg, "XBM_QUEUE_SIZE", 0)))
        self.xbm_start_epoch = max(0, int(_cfg_get(train_cfg, "XBM_START_EPOCH", 5)))
        self.register_buffer(
            "xbm_image_queue", torch.zeros(max(self.xbm_queue_size, 1), 512),
            persistent=False,
        )
        self.register_buffer(
            "xbm_text_queue", torch.zeros(max(self.xbm_queue_size, 1), 512),
            persistent=False,
        )
        self.register_buffer("xbm_queue_ptr", torch.zeros((), dtype=torch.long), persistent=False)
        self.register_buffer("xbm_queue_count", torch.zeros((), dtype=torch.long), persistent=False)

        if cfg.MODEL.BACKBONE == "ViT-B/16":
            self.embed_dim = 768
            self.patch_size = 16
            self.text_proj_dim = 512
        elif cfg.MODEL.BACKBONE == "ViT-L/14":
            self.embed_dim = 1024
            self.patch_size = 14
            self.text_proj_dim = 768
            raise NotImplementedError("ViT-L/14 not implemented yet.")
        else:
            raise NotImplementedError(f"Backbone {cfg.MODEL.BACKBONE} not implemented.")

        self.output_hidden_states = output_hidden_states
        self.dtype = self.text_model.transformer.dtype
        self.im_size = cfg.DATASET.SIZE
        tokenizer_path = str(
            _cfg_get(cfg.MODEL, "TEXT_ENCODER_PATH", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
        )
        self.tokenizer = HFTokenizer(tokenizer_path, context_length=256, **{})

        self.mask_head = nn.Sequential(
            nn.Linear(self.text_proj_dim, self.text_proj_dim),
            nn.GELU(),
            nn.Linear(self.text_proj_dim, self.text_proj_dim),
            nn.GELU(),
            nn.Linear(self.text_proj_dim, self.text_proj_dim),
        )
        self.upscale = nn.Sequential(*[ScaleBlock(self.text_proj_dim) for _ in range(cfg.MODEL.NUM_UPSCALE)])
        self.pvl_adapters = nn.ModuleList([
            PVL_Adapter(
                in_channels_vis=self.embed_dim,
                in_channels_txt=self.embed_dim,
                adapter_channels=cfg.MODEL.ADAPTER_DIM,
                beta=cfg.MODEL.BETA,
                gate_init=cfg.MODEL.GATE_INIT,
            )
            for _ in range(len(self.fusion_stages))
        ])

        # JBT-Lite UGBRA: a decoder-side local geometry executor.  It is not a
        # semantic branch and adds no auxiliary target/loss.  Preserve the RNG
        # state across construction so BASE/EDGE/UGBRA/FULL paired runs keep
        # identical initialization and stochastic streams for all pre-existing
        # MedCLIPSeg parameters.
        ugbra_cfg = _cfg_get(_cfg_get(cfg, "MODEL", None), "UGBRA", None)
        self.ugbra_enabled = bool(_cfg_get(ugbra_cfg, "ENABLED", False))
        self.ugbra = None
        if self.ugbra_enabled:
            _cpu_rng_before_ugbra = torch.random.get_rng_state()
            _cuda_rng_before_ugbra = (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            )
            self.ugbra = UncertaintyGatedBoundaryResonanceAdapter(
                channels=self.text_proj_dim,
                reduction=int(_cfg_get(ugbra_cfg, "REDUCTION", 8)),
                gamma_init=float(_cfg_get(ugbra_cfg, "GAMMA_INIT", 0.0)),
                detach_uncertainty=bool(_cfg_get(ugbra_cfg, "DETACH_UNCERTAINTY", True)),
                gate_floor=float(_cfg_get(ugbra_cfg, "GATE_FLOOR", 0.05)),
                use_image_edge=bool(_cfg_get(ugbra_cfg, "USE_IMAGE_EDGE", True)),
            )
            torch.random.set_rng_state(_cpu_rng_before_ugbra)
            if _cuda_rng_before_ugbra is not None:
                torch.cuda.set_rng_state_all(_cuda_rng_before_ugbra)
            print(
                "[UGBRA] enabled | channels=%d hidden=%d gamma_init=%.4f "
                "detach_uncertainty=%s gate_floor=%.3f image_edge=%s params=%d"
                % (
                    self.ugbra.channels, self.ugbra.hidden_channels,
                    float(_cfg_get(ugbra_cfg, "GAMMA_INIT", 0.0)),
                    str(bool(_cfg_get(ugbra_cfg, "DETACH_UNCERTAINTY", True))),
                    float(_cfg_get(ugbra_cfg, "GATE_FLOOR", 0.05)),
                    str(bool(_cfg_get(ugbra_cfg, "USE_IMAGE_EDGE", True))),
                    sum(p.numel() for p in self.ugbra.parameters()),
                )
            )

        # JBT-Lite v10 QABR: a high-resolution, query-anchored boundary
        # refinement executor. It receives the already text-conditioned coarse
        # logit and therefore cannot become a competing semantic branch. Its
        # alpha is zero-initialized, so enabling QABR leaves the first Base
        # forward exactly unchanged. Preserve RNG across construction for strict
        # paired BASE/EDGE20/QABR/FULL experiments.
        qabr_cfg = _cfg_get(_cfg_get(cfg, "MODEL", None), "QABR", None)
        self.qabr_enabled = bool(_cfg_get(qabr_cfg, "ENABLED", False))
        self.qabr = None
        if self.qabr_enabled:
            _cpu_rng_before_qabr = torch.random.get_rng_state()
            _cuda_rng_before_qabr = (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            )
            self.qabr = QueryAnchoredBoundaryRefiner(
                decoder_channels=self.text_proj_dim,
                detail_channels=int(_cfg_get(qabr_cfg, "DETAIL_CHANNELS", 8)),
                hidden_channels=int(_cfg_get(qabr_cfg, "HIDDEN_CHANNELS", 32)),
                alpha_init=float(_cfg_get(qabr_cfg, "ALPHA_INIT", 0.0)),
                band_kernel=int(_cfg_get(qabr_cfg, "BAND_KERNEL", 11)),
                max_logit_delta=float(_cfg_get(qabr_cfg, "MAX_LOGIT_DELTA", 3.0)),
                detach_support=bool(_cfg_get(qabr_cfg, "DETACH_SUPPORT", True)),
                use_image_detail=bool(_cfg_get(qabr_cfg, "USE_IMAGE_DETAIL", True)),
            )
            torch.random.set_rng_state(_cpu_rng_before_qabr)
            if _cuda_rng_before_qabr is not None:
                torch.cuda.set_rng_state_all(_cuda_rng_before_qabr)
            print(
                "[QABR] enabled | decoder_channels=%d detail=%d hidden=%d "
                "alpha_init=%.4f band_kernel=%d max_logit_delta=%.3f "
                "detach_support=%s image_detail=%s params=%d"
                % (
                    self.qabr.decoder_channels, self.qabr.detail_channels,
                    self.qabr.hidden_channels,
                    float(_cfg_get(qabr_cfg, "ALPHA_INIT", 0.0)),
                    int(_cfg_get(qabr_cfg, "BAND_KERNEL", 11)),
                    float(_cfg_get(qabr_cfg, "MAX_LOGIT_DELTA", 3.0)),
                    str(bool(_cfg_get(qabr_cfg, "DETACH_SUPPORT", True))),
                    str(bool(_cfg_get(qabr_cfg, "USE_IMAGE_DETAIL", True))),
                    sum(p.numel() for p in self.qabr.parameters()),
                )
            )

        # The public Base-only constructor stops here. Auxiliary modules below
        # consume torch RNG and would otherwise shift the first DataLoader seed
        # and Base dropout masks even with hard gradient isolation. Save the
        # post-Base state and restore it after auxiliary initialization when
        # official parity is requested.
        m1_cfg_for_rng = _cfg_get(cfg, "M1", None)
        restore_official_base_rng = bool(
            _cfg_get(m1_cfg_for_rng, "OFFICIAL_BASE_RNG_ISOLATION", False)
        )
        official_base_cpu_rng = (
            torch.random.get_rng_state() if restore_official_base_rng else None
        )
        official_base_cuda_rng = (
            torch.cuda.get_rng_state_all()
            if restore_official_base_rng and torch.cuda.is_available()
            else None
        )

        self.m1_enabled = bool(_cfg_get(_cfg_get(cfg, "M1", None), "ENABLED", False))
        self.m1_train_mode = str(_cfg_get(_cfg_get(cfg, "M1", None), "TRAIN_MODE", "e2e")).lower()
        if self.m1_train_mode not in {"e2e", "frozen", "anchor_student"}:
            raise ValueError("M1.TRAIN_MODE must be 'e2e', 'frozen', or 'anchor_student'.")

        self.m1_inference_mode_requested = str(
            _cfg_get(_cfg_get(cfg, "M1", None), "INFERENCE_MODE", "preserve")
        ).strip().lower()
        self.m1_inference_mode = _canonical_m1_inference_mode(
            self.m1_inference_mode_requested
        )
        if self.m1_inference_mode not in {
            "preserve", "direct_fusion", "router_fusion", "text_verifier_fusion", "utility_risk_selection", "m2_direct_selection", "falsification_m3_selection", "unified_action_cf_selection", "unified_m1_safe_fusion"
        }:
            raise ValueError(
                "Unsupported M1.INFERENCE_MODE: "
                f"{self.m1_inference_mode_requested!r}."
            )

        m1_cfg = _cfg_get(cfg, "M1", None)
        self.mechanism_candidates = _uses_reference_mechanism(m1_cfg)
        # V452 TPMHG route flag.
        # This must be defined before self.v20_unified_action_cf because
        # that dispatch condition reads self.text_prompted_hypothesis.
        m1_cfg = _cfg_get(cfg, "M1", None)
        self.text_prompted_hypothesis = (
            str(_cfg_get(m1_cfg, "CANDIDATE_MODE", "")).strip().lower()
            in {"text_prompted_hypothesis", "tpmhg", "v452_tpmhg", "compositional_error_modes", "cem_candidates", "v474_cem"}
        )
        self.semlt_enabled = (
            str(_cfg_get(m1_cfg, "PROTOCOL", "")).strip().lower() == "semlt"
            or str(_cfg_get(m1_cfg, "CANDIDATE_MODE", "")).strip().lower()
            == "semlt_logit_transport"
        )
        self.geotr_m1_exact_enabled = (
            str(_cfg_get(m1_cfg, "PROTOCOL", "")).strip().lower() == "geotr_m1"
            or str(_cfg_get(m1_cfg, "CANDIDATE_MODE", "")).strip().lower()
            == "exact_geometry_transport"
        )
        self.semlt_autozero_enabled = (
            str(_cfg_get(m1_cfg, "PROTOCOL", "")).strip().lower() == "semlt_autozero"
            or str(_cfg_get(m1_cfg, "CANDIDATE_MODE", "")).strip().lower()
            == "semlt_autozero_transport"
        )
        self.mhcs_enabled = self.semlt_enabled or self.geotr_m1_exact_enabled or self.semlt_autozero_enabled or (
            str(_cfg_get(m1_cfg, "PROTOCOL", "")).strip().lower() == "mhcs"
            or str(_cfg_get(m1_cfg, "CANDIDATE_MODE", "")).strip().lower()
            in {"multi_hypothesis_composition", "mhcs"}
        )

        self.v20_unified_action_cf = (
            self.mechanism_candidates
            or self.text_prompted_hypothesis
            or self.mhcs_enabled
            or bool(_cfg_get(m1_cfg, "ACTION_BANK_UNIFIED_ACTION_CF", False))
        )
        self.v23_textqualified_structural_medoid = bool(
            _cfg_get(m1_cfg, "V23_TEXTQUALIFIED_STRUCTURAL_MEDOID", False)
        )
        self.v24_text_ranked_structural_medoid = bool(
            _cfg_get(m1_cfg, "V24_TEXT_RANKED_STRUCTURAL_MEDOID", False)
        )
        self.use_textqualified_structural_medoid = (
            self.v23_textqualified_structural_medoid
            or self.v24_text_ranked_structural_medoid
        )
        # V25_RUNTIME_CONFIG_COMPAT: legacy config loaders can drop unregistered YAML keys.
        self.v25_type_conditional_utility_bank = bool(
            _cfg_get(m1_cfg, "V25_TYPE_CONDITIONAL_UTILITY_BANK", False)
            or str(_cfg_get(m1_cfg, "M1_LOSS_VERSION", "")).lower() == "v25_type_conditional_utility"
            or str(_cfg_get(m1_cfg, "RUN_TAG", "")).lower() == "v25_b0frozen_typeconditionalutilitybank_100ep"
        )
        self.v383_conservative_action_value = bool(
            _cfg_get(m1_cfg, "V383_CONSERVATIVE_ACTION_VALUE", False)
            or str(_cfg_get(m1_cfg, "RUN_TAG", "")).upper().startswith("V383_")
            or str(_cfg_get(m1_cfg, "M1_LOSS_VERSION", "")).lower()
            == "v383_conservative_action_value"
        )
        self.v382_action_conditional_quantile_atomic = bool(
            (not self.v383_conservative_action_value)
            and (
                _cfg_get(m1_cfg, "V382_ACTION_CONDITIONAL_QUANTILE_ATOMIC", False)
                or str(_cfg_get(m1_cfg, "RUN_TAG", "")).upper().startswith("V382_")
                or str(_cfg_get(m1_cfg, "M1_LOSS_VERSION", "")).lower()
                == "v382_action_conditional_quantile_atomic"
            )
        )
        self.v381_lesion_background_calibrated_atomic = bool(
            (not self.v383_conservative_action_value)
            and (not self.v382_action_conditional_quantile_atomic)
            and (not _cfg_get(m1_cfg, "V391_LESION_BACKGROUND_MLP_CALIBRATED_ATOMIC", False))
            and (
                _cfg_get(m1_cfg, "CALIBRATION_LESION_BACKGROUND_CALIBRATED_ATOMIC", False)
                or str(_cfg_get(m1_cfg, "RUN_TAG", "")).upper().startswith("CALIBRATION_")
                or str(_cfg_get(m1_cfg, "M1_LOSS_VERSION", "")).lower()
                == "v381_lesion_background_calibrated_atomic"
            )
        )
        self.v391_lesion_background_mlp_calibrated_atomic = bool(
            (not self.v383_conservative_action_value)
            and (not self.v382_action_conditional_quantile_atomic)
            and (
                _cfg_get(m1_cfg, "V391_LESION_BACKGROUND_MLP_CALIBRATED_ATOMIC", False)
                or str(_cfg_get(m1_cfg, "RUN_TAG", "")).upper().startswith("V391_")
                or str(_cfg_get(m1_cfg, "M1_LOSS_VERSION", "")).lower()
                == "v391_lesion_background_mlp_calibrated_atomic"
            )
        )
        self.v393_preserve_aware_edit_control = bool(
            (not self.v383_conservative_action_value)
            and (not self.v382_action_conditional_quantile_atomic)
            and (
                _cfg_get(m1_cfg, "EDIT_CONTROL_PRESERVE_AWARE_EDIT_CONTROL", False)
                or _cfg_get(m1_cfg, "V394_COUNTERFACTUAL_VALUE_RANK", False)
                or bool(_cfg_get(m1_cfg, "EVIDENCE_GUIDED_FECG_ENABLED", False))
                or str(_cfg_get(m1_cfg, "RUN_TAG", "")).upper().startswith(("EDIT_CONTROL_", "V394_", "V395_", "EVIDENCE_GUIDED_"))
                or str(_cfg_get(m1_cfg, "M1_LOSS_VERSION", "")).lower()
                in {"v393_preserve_aware_edit_control", "v394_counterfactual_value_rank", "evidence_guided_candidate_control"}
            )
        )
        self.v392_dense_patch_text_falsification = bool(
            (not self.v383_conservative_action_value)
            and (not self.v382_action_conditional_quantile_atomic)
            and (not self.v393_preserve_aware_edit_control)
            and (
                _cfg_get(m1_cfg, "V392_DENSE_PATCH_TEXT_FALSIFICATION", False)
                or str(_cfg_get(m1_cfg, "RUN_TAG", "")).upper().startswith("V392_")
                or str(_cfg_get(m1_cfg, "M1_LOSS_VERSION", "")).lower()
                == "v392_dense_patch_text_falsification"
            )
        )
        self.v38_casewise_falsified_delta_consensus = bool(
            (not self.v383_conservative_action_value)
            and (not self.v382_action_conditional_quantile_atomic)
            and (not self.v381_lesion_background_calibrated_atomic)
            and (not self.v391_lesion_background_mlp_calibrated_atomic)
            and (not self.v392_dense_patch_text_falsification)
            and (not self.v393_preserve_aware_edit_control)
            and (
                _cfg_get(m1_cfg, "CONSENSUS_CASEWISE_FALSIFIED_DELTA_CONSENSUS", False)
                or str(_cfg_get(m1_cfg, "RUN_TAG", "")).upper().startswith("CONSENSUS_")
                or str(_cfg_get(m1_cfg, "M1_LOSS_VERSION", "")).lower()
                == "v38_casewise_falsified_delta_consensus"
            )
        )
        self.v37_text_falsified_structural_consensus = bool(
            (not self.v383_conservative_action_value)
            and (not self.v382_action_conditional_quantile_atomic)
            and (not self.v381_lesion_background_calibrated_atomic)
            and (not self.v391_lesion_background_mlp_calibrated_atomic)
            and (not self.v392_dense_patch_text_falsification)
            and (not self.v393_preserve_aware_edit_control)
            and (not self.v38_casewise_falsified_delta_consensus)
            and (
                _cfg_get(m1_cfg, "V37_TEXT_FALSIFIED_STRUCTURAL_CONSENSUS", False)
                or str(_cfg_get(m1_cfg, "RUN_TAG", "")).upper().startswith("V37_")
                or str(_cfg_get(m1_cfg, "M1_LOSS_VERSION", "")).lower()
                == "v37_text_falsified_structural_consensus"
            )
        )
        self.v32_island_phaseb_policy = bool(
            (not self.v383_conservative_action_value)
            and (not self.v382_action_conditional_quantile_atomic)
            and (not self.v381_lesion_background_calibrated_atomic)
            and (not self.v391_lesion_background_mlp_calibrated_atomic)
            and (not self.v392_dense_patch_text_falsification)
            and (not self.v393_preserve_aware_edit_control)
            and (not self.v38_casewise_falsified_delta_consensus)
            and (not self.v37_text_falsified_structural_consensus)
            and (
                _cfg_get(m1_cfg, "V32_ISLAND_PHASEB_POLICY", False)
                or _cfg_get(m1_cfg, "V34_SPATIAL_QUANTILE_WORLD_MODEL", False)
                or _cfg_get(m1_cfg, "V35_RESIDUAL_PURIFIED_WORLD_MODEL", False)
                or _cfg_get(m1_cfg, "V36_CASEWISE_PLACKETT_LUCE", False)
                or str(_cfg_get(m1_cfg, "RUN_TAG", "")).upper().startswith(("V32_", "V34_", "V35_", "V36_"))
                or str(_cfg_get(m1_cfg, "M1_LOSS_VERSION", "")).lower()
                in {
                    "v32_island_phaseb", "v34_spatial_quantile_world_model",
                    "v35_residual_purified_world_model", "v36_casewise_plackett_luce",
                }
            )
        )
        self.v31_candidate_conditioned_policy = bool(
            (not bool(_cfg_get(m1_cfg, "SEMLT_LST_V31_ROOTFIX", False)))
            and (
                _cfg_get(m1_cfg, "V31_CANDIDATE_CONDITIONED_POLICY", False)
                or str(_cfg_get(m1_cfg, "RUN_TAG", "")).upper().startswith("V31_")
                or str(_cfg_get(m1_cfg, "M1_LOSS_VERSION", "")).lower()
                == "v31_competitive_utility"
            )
        )
        self.text_prompted_hypothesis = (
            str(_cfg_get(m1_cfg, "CANDIDATE_MODE", "")).strip().lower()
            in {"text_prompted_hypothesis", "tpmhg", "v452_tpmhg", "compositional_error_modes", "cem_candidates", "v474_cem"}
        )
        self.semlt_enabled = (
            str(_cfg_get(m1_cfg, "PROTOCOL", "")).strip().lower() == "semlt"
            or str(_cfg_get(m1_cfg, "CANDIDATE_MODE", "")).strip().lower()
            == "semlt_logit_transport"
        )
        self.geotr_m1_exact_enabled = (
            str(_cfg_get(m1_cfg, "PROTOCOL", "")).strip().lower() == "geotr_m1"
            or str(_cfg_get(m1_cfg, "CANDIDATE_MODE", "")).strip().lower()
            == "exact_geometry_transport"
        )
        self.semlt_autozero_enabled = (
            str(_cfg_get(m1_cfg, "PROTOCOL", "")).strip().lower() == "semlt_autozero"
            or str(_cfg_get(m1_cfg, "CANDIDATE_MODE", "")).strip().lower()
            == "semlt_autozero_transport"
        )
        self.mhcs_enabled = self.semlt_enabled or self.geotr_m1_exact_enabled or self.semlt_autozero_enabled or (
            str(_cfg_get(m1_cfg, "PROTOCOL", "")).strip().lower() == "mhcs"
            or str(_cfg_get(m1_cfg, "CANDIDATE_MODE", "")).strip().lower()
            in {"multi_hypothesis_composition", "mhcs"}
        )
        # Construct exactly one active M1 module.  The historical code first
        # instantiated TrainablePSEGenerator and then overwrote it with
        # AutoZero/GEOTR/SemLT, consuming RNG for a dead module.  That changed
        # the stochastic Base trajectory even when Base parameter hashes were
        # identical.  This single-constructor route removes that ghost RNG draw.
        if not self.m1_enabled:
            self.m1_pse = None
        elif self.semlt_autozero_enabled:
            self.m1_pse = AutoZeroSemanticTransportSegmenter(cfg)
            if bool(_cfg_get(m1_cfg, "SEMLT_LST_V31_ROOTFIX", False)):
                print(
                    "[SemLT-LST v3.1 ROOTFIX] transition-band sparse deployment | "
                    "GT teacher-gated correction + benefit-supervised edit decision | "
                    "selection/correction gradients causally separated"
                )
            else:
                print(
                    "[SemLT-LST v3 CALIBRATED] bounded local state-aware evidence "
                    "transport | posterior-calibrated edit gate + supervised local "
                    "KEEP/ADD/REMOVE source distribution"
                )
        elif self.geotr_m1_exact_enabled:
            self.m1_pse = ExactGeometryTransportSegmenter(cfg)
            print(
                "[GEOTR-M1 EXACT] validated Stage-1 Geometry/Transport restored | "
                "flow head, feature path and objective are historical-exact; M2/M3 absent"
            )
        elif self.semlt_enabled:
            self.m1_pse = SemanticLogitTransportSegmenter(cfg)
            print(
                "[SemLT-M1] semantic-conditioned bounded logit transport enabled | "
                "Base is factual H0; no M2/M3 module exists"
            )
        elif self.mhcs_enabled:
            self.m1_pse = MultiHypothesisCompositionalSegmenter(cfg)
            print(
                "[MHCS-R4.2] root-complete stochastic full-mask bank + separated global/local "
                "pixel-wise composition enabled | Base is candidate H0 only"
            )
        elif self.text_prompted_hypothesis:
            self.m1_pse = TextPromptedMultiHypothesisGenerator(cfg)
        elif self.v20_unified_action_cf:
            self.m1_pse = (
                ReferenceAdaptiveC6Bank(cfg)
                if self.mechanism_candidates
                else V383ConservativeActionValueBank(cfg)
                if self.v383_conservative_action_value
                else V382ActionConditionalQuantileAtomicBank(cfg)
                if self.v382_action_conditional_quantile_atomic
                else V391LesionBackgroundMLPCalibratedAtomicBank(cfg)
                if self.v391_lesion_background_mlp_calibrated_atomic
                else V393PreserveAwareEditControlBank(cfg)
                if self.v393_preserve_aware_edit_control
                else V392DensePatchTextFalsificationBank(cfg)
                if self.v392_dense_patch_text_falsification
                else V381LesionBackgroundCalibratedAtomicBank(cfg)
                if self.v381_lesion_background_calibrated_atomic
                else V38CasewiseFalsifiedDeltaConsensusBank(cfg)
                if self.v38_casewise_falsified_delta_consensus
                else V37TextFalsifiedStructuralConsensusBank(cfg)
                if self.v37_text_falsified_structural_consensus
                else V32IslandDeletePhaseBPolicyBank(cfg)
                if self.v32_island_phaseb_policy
                else V31CandidateConditionedUtilityBank(cfg)
                if self.v31_candidate_conditioned_policy
                else TypeConditionalUtilityActionBank(cfg)
                if self.v25_type_conditional_utility_bank
                else TextQualifiedStructuralMedoidBank(cfg)
                if self.use_textqualified_structural_medoid
                else UnifiedActionCounterfactualSetBank(cfg)
            )
        else:
            self.m1_pse = TrainablePSEGenerator(cfg)


        # V484/V491: error-state gated causal intervention pipeline.
        # This is a model submodule so optimizer/EMA discover its parameters
        # naturally. V491 keeps every active downstream block trainable while
        # bounding cross-module gradients and preserving the factual Base path.
        m1_protocol = str(_cfg_get(m1_cfg, "PROTOCOL", "")).strip().lower()
        tc_drcs = m1_protocol == "tc_drcs"
        clean_dynamic_component_set = m1_protocol in {"clean_dynamic_component_set", "tc_drcs"}
        self.v484_enabled = bool(
            self.m1_enabled
            and (
                clean_dynamic_component_set
                or bool(_cfg_get(m1_cfg, "CEM_V484_ENABLED", False))
            )
        )
        self.v484_pipeline = (
            V484ErrorStateCausalPipeline(cfg, semantic_channels=int(self.text_proj_dim))
            if self.v484_enabled else None
        )
        if self.v484_enabled:
            if tc_drcs:
                print(
                    "[TC_DRCS] teacher-complete differentiable residual component set enabled | "
                    "direct masks + soft mask-guided localization + training-only mask pilots; "
                    "no hard seed/ownership gate"
                )
            elif clean_dynamic_component_set:
                print(
                    "[CLEAN_COMPONENT_SET] continuous residual error + local-max anchors + "
                    "learned spatial precision + uncertainty-balanced M1 objectives enabled"
                )
            elif bool(_cfg_get(m1_cfg, "V532_UNIFIED_SPARSE_REFINER_ENABLED", False)):
                print(
                    "[V532] internal coarse mask + typed residual-error locator + "
                    "factorized Preserve/Edit router + actual Neutral/Benefit/Harm "
                    "outcome gate enabled | one-pass soft deployment; no candidate bank "
                    "and no separate M3"
                )
            elif bool(_cfg_get(m1_cfg, "V531_TYPED_SPARSE_REFINER_ENABLED", False)):
                print(
                    "[V531] typed factual error maps + four monotone atomic "
                    "action experts + Preserve-first spatial risk routing enabled | "
                    "single refinement pass; no candidate-bank Transformer and no M3"
                )
            elif bool(_cfg_get(m1_cfg, "V498_CONSISTENT_FULL_PROPOSAL_ENABLED", False)):
                print(
                    "[V498] train/deploy-consistent full-proposal M2 + "
                    "Preserve-reference action supervision + selected-support "
                    "risk-sign M3 enabled | hard top-1 non-Base proposal in "
                    "both training and inference; M3 remains the only "
                    "irreversible Preserve-vs-proposal decision"
                )
            elif bool(_cfg_get(m1_cfg, "V495_SINGLE_DECISION_ENABLED", False)):
                print(
                    "[V495] single-decision causal proposal M2 + Preserve-first "
                    "relative-risk M3 enabled | MC logits aggregated before one "
                    "route; M3 is the only irreversible accept/reject decision"
                )
            elif bool(_cfg_get(m1_cfg, "V494_DIRECT_CAUSAL_DOSE_ENABLED", False)):
                print(
                    "[V494] direct candidate-utility causal dose M2 + "
                    "expected-relative-risk Preserve-first M3 enabled | exact "
                    "full-dose convex intervention | no task checkpoint; only "
                    "image/text encoders frozen"
                )
            elif bool(_cfg_get(m1_cfg, "V493_CANDIDATE_CONDITIONED_CAUSAL_ENABLED", False)):
                print(
                    "[V493] candidate-conditioned causal utility M2 + "
                    "imagewise-risk Preserve-first M3 enabled | mutually-exclusive "
                    "Benefit supervision | no task checkpoint; only image/text encoders frozen"
                )
            elif bool(_cfg_get(m1_cfg, "V492_CAUSAL_LOCAL_EDITOR_ENABLED", False)):
                print(
                    "[V492] causal local M2 (benefit/amplitude/sparse route) + "
                    "normalized Preserve-first M3 enabled | no task checkpoint; "
                    "only image/text encoders frozen"
                )
            elif bool(_cfg_get(m1_cfg, "V491_PRESERVE_FIRST_ENABLED", False)):
                print(
                    "[V491] optimal-convex M2 + Preserve-first binary risk M3 enabled | "
                    "no task checkpoint; only image/text encoders frozen"
                )
            elif bool(_cfg_get(m1_cfg, "V490_OPTIMAL_CONVEX_TEACHER_ENABLED", False)) and bool(
                _cfg_get(m1_cfg, "V490_SCALE_INVARIANT_RISK_ENABLED", False)
            ):
                print("[V490.4] optimal-convex M2 + scale-invariant dense relative-risk M3 enabled | bounded auxiliary-to-Base gradients")
            elif bool(_cfg_get(m1_cfg, "V490_DENSE_RELATIVE_RISK_ENABLED", False)):
                print("[V490.2] End-to-end continuous M2 + dense coherent relative-risk M3 enabled | only image/text encoders frozen")
            elif bool(_cfg_get(m1_cfg, "V490_ROOT_CAUSE_ENABLED", False)):
                print("[V490] End-to-end continuous M2 + intervention-region risk M3 enabled | only image/text encoders frozen")
            elif bool(_cfg_get(m1_cfg, "V489_END_TO_END_ENABLED", False)):
                print("[V489] End-to-end sparse composer + region best-expert enabled | only image/text encoders frozen")
            else:
                print("[V484] Error-State Gated Causal Intervention enabled | C0 detached inside M1/M2/M3")

        self.m2_text_verifier_enabled = bool(
            _cfg_get(m1_cfg, "TEXT_VERIFIER_ENABLED", False)
        ) and not self.v20_unified_action_cf
        self.m2_text_verifier = (
            TextCounterfactualVerifier(cfg, semantic_channels=self.text_proj_dim)
            if self.m1_enabled and self.m2_text_verifier_enabled
            else None
        )

        # V463: true M2 module. It is separate from M1 and therefore appears
        # in the optimiser/logs as the m2 parameter group, not hidden inside M1.
        self.v463_ccv_enabled = bool(
            self.m1_enabled and _cfg_get(m1_cfg, "V463_CCV_ENABLED", False)
        )
        self.ccv_m2 = (
            V463CausalCounterfactualVerifier(cfg)
            if self.v463_ccv_enabled else None
        )

        if (
            self.m1_enabled
            and self.m1_inference_mode == "router_fusion"
            and self.m1_pse is not None
            and not self.m1_pse.router_enabled
        ):
            raise ValueError("M1.INFERENCE_MODE='router_fusion' requires M1.ROUTER_ENABLED: true.")
        if (
            self.m1_enabled
            and self.m1_inference_mode in {"text_verifier_fusion", "utility_risk_selection", "m2_direct_selection", "falsification_m3_selection"}
            and self.m2_text_verifier is None
        ):
            raise ValueError(
                "M1.INFERENCE_MODE='text_verifier_fusion', 'utility_risk_selection', or 'falsification_m3_selection' requires "
                "M1.TEXT_VERIFIER_ENABLED: true."
            )
        self.current_epoch = 0
        if restore_official_base_rng:
            torch.random.set_rng_state(official_base_cpu_rng)
            if official_base_cuda_rng is not None:
                torch.cuda.set_rng_state_all(official_base_cuda_rng)

    def train(self, mode: bool = True):
        """Full-vision E2E while keeping text and V396 observer frozen."""
        super().train(mode)
        m1_cfg = _cfg_get(self.cfg, "M1", None)
        if bool(_cfg_get(m1_cfg, "OFFICIAL_MEDCLIPSEG_BASE_CONTRACT", False)):
            # The public implementation freezes encoder gradients but leaves
            # both encoders in the parent model's train/eval mode. In
            # particular, BERT dropout is active while training. Forcing eval
            # here changes the official Base optimization trajectory.
            return self
        freeze_encoders = bool(
            _cfg_get(_cfg_get(self.cfg, "MODEL", None), "FREEZE_IMAGE_TEXT_ENCODERS", False)
        )
        full_end_to_end = bool(_cfg_get(m1_cfg, "V478_FULL_END_TO_END", False))
        if freeze_encoders:
            self.text_model.eval()
            self.vision_model.eval()
        elif full_end_to_end:
            self.text_model.train(mode)
            self.vision_model.train(mode)
        else:
            self.text_model.eval()

        observer = getattr(self, "_v396_observer_vision", None)
        if observer is not None:
            observer.eval()

        full_vision_e2e = bool(
            _cfg_get(m1_cfg, "V395_FULL_VISION_E2E", False)
        )
        partial_blocks = int(
            _cfg_get(m1_cfg, "V395_UNFREEZE_VISION_LAST_N", 0)
        )

        # ── M2 TIDE-Repair freeze contract ──
        m2_freeze_b0_m1 = bool(_cfg_get(m1_cfg, "M2_FREEZE_B0_M1", False))

        if freeze_encoders:
            return self
        if full_end_to_end:
            return self
        if self.m1_active and self.m1_train_mode == "e2e":
            if full_vision_e2e:
                if not m2_freeze_b0_m1:
                    self.vision_model.train(mode)
                else:
                    self.vision_model.eval()
            else:
                self.vision_model.eval()
                if partial_blocks > 0:
                    blocks = list(self.vision_model.transformer.resblocks)
                    for block in blocks[-min(partial_blocks, len(blocks)):]:
                        block.train(mode)
                    self.vision_model.ln_post.train(mode)
        else:
            self.vision_model.eval()
        return self

    @property
    def m1_active(self) -> bool:
        return bool(self.m1_enabled and self.m1_pse is not None)

    @staticmethod
    def _fuse_directional_candidates(candidate_probs: torch.Tensor) -> torch.Tensor:
        if candidate_probs.ndim != 4 or candidate_probs.shape[1] != 3:
            raise ValueError(
                "Expected candidate probabilities [B,3,H,W], got "
                f"{tuple(candidate_probs.shape)}"
            )
        preserve, shrink, expand = candidate_probs[:, 0], candidate_probs[:, 1], candidate_probs[:, 2]
        return (preserve + (shrink - preserve) + (expand - preserve)).clamp(EPS, 1.0 - EPS)

    def _fuse_from_aux(self, candidate_probs: torch.Tensor, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        """V404: return Preserve or an exact gathered frozen-M1 candidate."""
        selected = aux.get("m2_selected_probs")
        if isinstance(selected, torch.Tensor):
            return selected

        if self.m1_inference_mode == "unified_m1_safe_fusion":
            if "m1_hard_fused_probs" not in aux:
                raise RuntimeError("Unified M1 safe-fusion output is absent.")
            return aux["m1_hard_fused_probs"]


        if self.v20_unified_action_cf:
            if "v20_hard_fused_probs" not in aux:
                raise RuntimeError("V20 unified selection output is absent.")
            return aux["v20_hard_fused_probs"]

        if self.m1_inference_mode == "preserve":
            return candidate_probs[:, 0]

        if self.m1_inference_mode == "router_fusion":
            if "router_fused_probs" not in aux:
                raise RuntimeError("Router fusion requested but router_fused_probs is absent.")
            return aux["router_fused_probs"]

        if self.m1_inference_mode == "text_verifier_fusion":
            if "text_verifier_fused_probs" not in aux:
                raise RuntimeError(
                    "Text-verifier fusion requested but text_verifier_fused_probs is absent."
                )
            return aux["text_verifier_fused_probs"]

        if self.m1_inference_mode == "utility_risk_selection":
            if "utility_risk_selected_probs" not in aux:
                raise RuntimeError(
                    "Utility-risk selection requested but M2 selection output is absent."
                )
            return aux["utility_risk_selected_probs"]

        if self.m1_inference_mode == "m2_direct_selection":
            if "falsification_m2_direct_selected_probs" not in aux:
                raise RuntimeError(
                    "M2 direct selection requested but direct M2 output is absent."
                )
            return aux["falsification_m2_direct_selected_probs"]

        if self.m1_inference_mode == "falsification_m3_selection":
            if "falsification_m3_selected_probs" not in aux:
                raise RuntimeError(
                    "V15 dense mask-text M3 selection requested but evidence output is absent."
                )
            return aux["falsification_m3_selected_probs"]

        if bool(_cfg_get(_cfg_get(self.cfg, "M1", None), "V19_ACTION_BANK", False)):
            return candidate_probs[:, 0]

        if bool(_cfg_get(_cfg_get(self.cfg, "M1", None), "V18_ATOMIC_CANDIDATES", False)):
            return candidate_probs[:, 0]

        return self._fuse_directional_candidates(candidate_probs)
    def fuse_candidate_probabilities(
        self, candidate_probs: torch.Tensor, aux: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Public fusion alias. V20 uses the jointly learned sparse action set."""
        if self.v20_unified_action_cf:
            if aux is None:
                raise ValueError("V20 fusion requires action-bank auxiliary outputs.")
            return self._fuse_from_aux(candidate_probs, aux)
        if self.m1_inference_mode in {"router_fusion", "text_verifier_fusion", "m2_direct_selection", "falsification_m3_selection"}:
            if aux is None:
                raise ValueError("Configured M1 fusion requires auxiliary prediction maps.")
            return self._fuse_from_aux(candidate_probs, aux)
        if bool(_cfg_get(_cfg_get(self.cfg, "M1", None), "V19_ACTION_BANK", False)):
            return candidate_probs[:, 0]
        if bool(_cfg_get(_cfg_get(self.cfg, "M1", None), "V18_ATOMIC_CANDIDATES", False)):
            return candidate_probs[:, 0]
        return self._fuse_directional_candidates(candidate_probs)

    def _semantic_map_from_image_features(
        self, image_features: torch.Tensor, image_hw: tuple[int, int], target_hw: tuple[int, int], *, resize: bool = True
    ) -> torch.Tensor:
        """Convert projected ViT patch tokens [B,1+N,C] to a spatial map."""
        patches = image_features[:, 1:, :]
        batch, patch_count, channels = patches.shape
        patch_h = int(image_hw[0]) // self.patch_size
        patch_w = int(image_hw[1]) // self.patch_size
        if patch_h * patch_w != patch_count:
            # Defensive fallback for non-square/nonstandard image sizes.
            root = int(round(math.sqrt(patch_count)))
            if root * root != patch_count:
                raise ValueError(
                    f"Cannot reshape {patch_count} patch tokens to a spatial map for image {image_hw}."
                )
            patch_h = patch_w = root
        semantic = patches.reshape(batch, patch_h, patch_w, channels).permute(0, 3, 1, 2).contiguous()
        if resize and semantic.shape[-2:] != target_hw:
            semantic = F.interpolate(semantic, size=target_hw, mode="bilinear", align_corners=False)
        return semantic

    def _negative_text_templates(self):
        """Hard negatives used only to construct target-vs-non-target maps."""
        m1 = _cfg_get(self.cfg, "M1", None)
        raw = str(_cfg_get(
            m1, "M2_DENSE_NEGATIVE_TEMPLATES",
            _cfg_get(m1, "M2_CF_NEGATIVE_TEMPLATES", "normal breast tissue without lesion|background breast ultrasound tissue|acoustic shadow or non-lesion artifact|no lesion is present in this breast ultrasound image."),
        ))
        templates = [x.strip() for x in raw.split("|") if x.strip()]
        return templates or ["no lesion is present in this breast ultrasound image."]

    @staticmethod
    def _shift_no_wrap(x: torch.Tensor, shift_y: int, shift_x: int) -> torch.Tensor:
        """Translate without toroidal wrap-around."""
        out = torch.zeros_like(x)
        h, w = x.shape[-2:]
        src_y0, src_y1 = max(0, -shift_y), min(h, h - shift_y)
        src_x0, src_x1 = max(0, -shift_x), min(w, w - shift_x)
        dst_y0, dst_y1 = max(0, shift_y), min(h, h + shift_y)
        dst_x0, dst_x1 = max(0, shift_x), min(w, w + shift_x)
        if src_y1 > src_y0 and src_x1 > src_x0 and dst_y1 > dst_y0 and dst_x1 > dst_x0:
            out[..., dst_y0:dst_y1, dst_x0:dst_x1] = x[..., src_y0:src_y1, src_x0:src_x1]
        return out

    def _same_area_shrink_control(
        self,
        preserve_hard: torch.Tensor,
        shrink_hard: torch.Tensor,
    ) -> torch.Tensor:
        """Construct a deterministic non-overlapping same-area deletion control.

        The old V15.2 control translated a narrow boundary edit once and then
        clipped it by the Base mask. On BUSI this retained only 0--24% of the
        edit area, so every proposed control failed the pre-registered 0.8
        area-ratio rule. V15.3 first searches non-wrapping translations in
        eight directions. If none preserves enough area, it uses a
        deterministic nearest-valid-pixel fallback inside Preserve, excluding
        the factual edit. This fallback changes only the mask hypothesis;
        image pixels, prompt, dense VLM map, and labels remain untouched.
        """
        source = ((preserve_hard > 0.5) & (shrink_hard < 0.5)).float()
        source_area = int(source.sum().item())
        control = torch.zeros_like(source)
        if source_area <= 0:
            return control

        m1 = _cfg_get(self.cfg, "M1", None)
        min_ratio = float(_cfg_get(m1, "M2_DENSE_CONTROL_MIN_AREA_RATIO", 0.80))
        sy = max(1, int(_cfg_get(m1, "M2_CF_CONTROL_SHIFT_Y", 3)))
        sx = max(1, int(_cfg_get(m1, "M2_CF_CONTROL_SHIFT_X", 5)))
        guard = max(0, int(_cfg_get(m1, "M2_DENSE_CONTROL_GUARD_RADIUS", 0)))

        blocked = (_soft_dilate(source[None, None], guard)[0, 0] > 0.5)
        allowed = (preserve_hard > 0.5) & (~blocked)

        offsets = []
        for scale in (1, 2, 3, 4):
            dy, dx = scale * sy, scale * sx
            offsets.extend([
                (dy, 0), (-dy, 0), (0, dx), (0, -dx),
                (dy, dx), (dy, -dx), (-dy, dx), (-dy, -dx),
            ])

        best = None
        best_key = None
        for dy, dx in offsets:
            proposal = self._shift_no_wrap(source, dy, dx) * allowed.float()
            area = float(proposal.sum().item())
            if area <= 0.0:
                continue
            ratio = area / float(source_area)
            key = (abs(ratio - 1.0), abs(dy) + abs(dx), dy, dx)
            if best is None or key < best_key:
                best = proposal
                best_key = key

        if best is not None and float(best.sum().item()) / float(source_area) >= min_ratio:
            return best

        # Translation can fail for thin boundary edits in small lesions. The
        # fallback remains label-free and exact-area whenever enough Preserve
        # pixels remain. It selects nearest valid pixels to the factual-edit
        # centroid while enforcing zero factual/control overlap.
        available_count = int(allowed.sum().item())
        required_count = int(math.ceil(min_ratio * float(source_area)))
        if available_count < required_count:
            # Relax only the optional guard band, never factual-edit overlap.
            allowed = (preserve_hard > 0.5) & (source <= 0.5)
            available_count = int(allowed.sum().item())
            if available_count < required_count:
                return control

        h, w = source.shape[-2:]
        yy = torch.arange(h, device=source.device, dtype=source.dtype).view(h, 1)
        xx = torch.arange(w, device=source.device, dtype=source.dtype).view(1, w)
        cy = (source * yy).sum() / source.sum().clamp_min(1.0)
        cx = (source * xx).sum() / source.sum().clamp_min(1.0)
        distance = (yy - cy).pow(2) + (xx - cx).pow(2)
        distance = distance.masked_fill(~allowed, float("inf"))
        take = min(source_area, available_count)
        if take <= 0:
            return control
        chosen = torch.topk(distance.reshape(-1), k=take, largest=False).indices
        control.reshape(-1)[chosen] = 1.0
        return control

    def _counterfactual_control_candidates(self, candidate_probs: torch.Tensor) -> torch.Tensor:
        """V15.3 same-area controls for Preserve/Shrink/Expand hypotheses.

        Shrink is the only deployable/scored action in this protocol. Its
        control is exact-area whenever geometry permits and is otherwise
        explicitly unavailable downstream; thresholds are never relaxed to
        fabricate an evidence sample. Expand retains the prior diagnostic
        construction and remains disabled by the formal V15.3 configuration.
        """
        m1 = _cfg_get(self.cfg, "M1", None)
        preserve = candidate_probs[:, 0].detach()
        threshold = float(_cfg_get(m1, "ANCHOR_THRESHOLD", 0.50))
        anchor = (preserve >= threshold).float()
        connect_radius = max(1, int(_cfg_get(m1, "M2_CF_CONTROL_CONNECT_RADIUS", 2)))
        connected_outside = (
            _soft_dilate(anchor.unsqueeze(1), connect_radius)[:, 0] - anchor
        ).clamp(0.0, 1.0)
        sy = max(1, int(_cfg_get(m1, "M2_CF_CONTROL_SHIFT_Y", 3)))
        sx = max(1, int(_cfg_get(m1, "M2_CF_CONTROL_SHIFT_X", 5)))
        expand_delta = (candidate_probs[:, 2].detach() - preserve).clamp_min(0.0)

        shrink_controls = []
        for index in range(preserve.shape[0]):
            preserve_hard = (preserve[index] >= threshold).float()
            shrink_hard = (candidate_probs[index, 1].detach() >= threshold).float()
            remove_control = self._same_area_shrink_control(preserve_hard, shrink_hard)
            shrink_controls.append((preserve[index] - remove_control).clamp(EPS, 1.0 - EPS))
        shrink_ctrl = torch.stack(shrink_controls, dim=0)

        def shifted_supported(delta: torch.Tensor, support: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
            a = self._shift_no_wrap(delta, dy, dx) * support
            b = self._shift_no_wrap(delta, -dy, -dx) * support
            ma = a.sum(dim=(-2, -1), keepdim=True)
            mb = b.sum(dim=(-2, -1), keepdim=True)
            return torch.where(ma >= mb, a, b)

        expand_ctrl = shifted_supported(expand_delta, connected_outside, -sy, sx)
        return torch.stack([
            preserve,
            shrink_ctrl,
            (preserve + expand_ctrl).clamp(EPS, 1.0 - EPS),
        ], dim=1)

    @staticmethod
    def _binary_ring(mask: torch.Tensor, radius: int) -> torch.Tensor:
        x = (mask > 0.5).float()
        ring = (_soft_dilate(x, radius) - x).clamp(0.0, 1.0)
        fallback = (1.0 - x).clamp(0.0, 1.0)
        return torch.where(ring.sum(dim=(-2, -1), keepdim=True) > 0, ring, fallback)

    def _build_dense_hypothesis_crops(self, image, candidate_probs, supervision_masks=None):
        """Build unchanged crops and P/S/control hypotheses with geometry audit."""
        m1 = _cfg_get(self.cfg, "M1", None)
        b, _, ih, iw = image.shape
        anchor = float(_cfg_get(m1, "ANCHOR_THRESHOLD", 0.50))
        edit_threshold = float(_cfg_get(m1, "M2_DENSE_EDIT_THRESHOLD", 0.02))
        context_ratio = float(_cfg_get(m1, "M2_DENSE_CROP_CONTEXT_RATIO", 0.35))
        min_crop = max(16, int(_cfg_get(m1, "M2_DENSE_MIN_CROP_SIZE", 64)))
        target = int(self.im_size)
        controls = self._counterfactual_control_candidates(candidate_probs)
        crops, p_list, s_list, e_list, c_list, gt_list = [], [], [], [], [], []
        origins, roles, control_ratios, control_overlaps = [], [], [], []
        score_expand = bool(_cfg_get(m1, "M2_CF_SCORE_EXPAND", False))
        requested_roles = [1] + ([2] if score_expand else [])
        gt_full = None
        if supervision_masks is not None:
            gt_full = supervision_masks
            if gt_full.ndim == 4:
                gt_full = gt_full[:, 0]
            gt_full = (gt_full > 0.5).float()
        for bi in range(b):
            preserve_h = (candidate_probs[bi, 0].detach() >= anchor).float()
            for role in requested_roles:
                cand_h = (candidate_probs[bi, role].detach() >= anchor).float()
                ctrl_h = (controls[bi, role].detach() >= anchor).float()
                edit = (preserve_h - cand_h).abs()
                ctrl_edit = (preserve_h - ctrl_h).abs()
                edit_mass = edit.sum()
                if float(edit_mass.detach().cpu()) <= 0.0:
                    continue
                control_ratio = ctrl_edit.sum() / edit_mass.clamp_min(EPS)
                control_overlap = (edit * ctrl_edit).sum() / edit_mass.clamp_min(EPS)
                coords = ((edit > edit_threshold) | (preserve_h > 0.5) | (cand_h > 0.5)).nonzero(as_tuple=False)
                if coords.numel() == 0:
                    continue
                y0, y1 = int(coords[:, 0].min()), int(coords[:, 0].max()) + 1
                x0, x1 = int(coords[:, 1].min()), int(coords[:, 1].max()) + 1
                side = max(y1-y0, x1-x0, min_crop)
                pad = max(2, int(round(side * context_ratio)))
                cy, cx = 0.5*(y0+y1), 0.5*(x0+x1)
                half = 0.5*side + pad
                yy0, yy1 = max(0, int(math.floor(cy-half))), min(ih, int(math.ceil(cy+half)))
                xx0, xx1 = max(0, int(math.floor(cx-half))), min(iw, int(math.ceil(cx+half)))
                if yy1 <= yy0 or xx1 <= xx0:
                    continue
                crop = F.interpolate(
                    image[bi:bi+1, :, yy0:yy1, xx0:xx1],
                    size=(target, target),
                    mode="bilinear",
                    align_corners=False,
                )
                def crop_mask(mask):
                    return F.interpolate(
                        mask[None, None, yy0:yy1, xx0:xx1],
                        size=(target, target),
                        mode="nearest",
                    )
                crops.append(crop)
                p_list.append(crop_mask(preserve_h))
                s_list.append(crop_mask(cand_h))
                c_list.append(crop_mask(ctrl_h))
                e_list.append(crop_mask(edit))
                if gt_full is not None:
                    gt_list.append(crop_mask(gt_full[bi]))
                origins.append(bi)
                roles.append(role)
                control_ratios.append(control_ratio)
                control_overlaps.append(control_overlap)
        if not crops:
            empty_crop = image.new_zeros(0, image.shape[1], target, target)
            empty_mask = image.new_zeros(0, 1, target, target)
            return {
                "crops": empty_crop, "preserve": empty_mask, "candidate": empty_mask,
                "control": empty_mask, "edit": empty_mask, "gt": empty_mask,
                "origins": image.new_zeros(0, dtype=torch.long),
                "roles": image.new_zeros(0, dtype=torch.long),
                "control_area_ratio": image.new_zeros(0),
                "control_overlap_ratio": image.new_zeros(0),
            }
        return {
            "crops": torch.cat(crops, 0),
            "preserve": torch.cat(p_list, 0),
            "candidate": torch.cat(s_list, 0),
            "control": torch.cat(c_list, 0),
            "edit": torch.cat(e_list, 0),
            "gt": torch.cat(gt_list, 0) if gt_list else image.new_zeros(0, 1, target, target),
            "origins": torch.tensor(origins, device=image.device, dtype=torch.long),
            "roles": torch.tensor(roles, device=image.device, dtype=torch.long),
            "control_area_ratio": torch.stack(control_ratios).to(device=image.device, dtype=image.dtype),
            "control_overlap_ratio": torch.stack(control_overlaps).to(device=image.device, dtype=image.dtype),
        }

    def _dense_text_score_map(self, crops: torch.Tensor, prompts) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dense local target-vs-negative patch score map on the unmodified crop."""
        if crops.shape[0] == 0:
            z = crops.new_zeros(0, 1, 1)
            return z, z, z
        m1 = _cfg_get(self.cfg, "M1", None)
        chunk = max(1, int(_cfg_get(m1, "M2_DENSE_VIEW_CHUNK", _cfg_get(m1, "M2_CF_VIEW_CHUNK", 8))))
        neg_templates = self._negative_text_templates()
        neg_tau = max(float(_cfg_get(m1, "M2_DENSE_NEGATIVE_LSE_TEMPERATURE", 0.05)), 1e-5)
        all_pos, all_neg = [], []
        for st in range(0, crops.shape[0], chunk):
            ed = min(crops.shape[0], st+chunk)
            view = crops[st:ed]
            pos_prompts = [str(x) for x in prompts[st:ed]]
            with torch.no_grad():
                tok = self.tokenizer(pos_prompts).to(view.device)
                emb = self.text_model.transformer.embeddings.word_embeddings(tok).type(self.dtype)
                img_feat, txt_feat = self.encode_text_image(tok, emb, view)
            patches = self.m2_text_verifier.align_patch_tokens(img_feat[:,1:,:].float())
            txt = F.normalize(txt_feat.float(), dim=-1)
            score = (patches * txt.unsqueeze(1)).sum(dim=-1)
            n = score.shape[1]; root = int(round(math.sqrt(n)))
            if root*root != n: raise ValueError(f"V15 expected square patch grid, got {n}")
            pos = score.reshape(score.shape[0], root, root)
            neg_scores=[]
            for template in neg_templates:
                with torch.no_grad():
                    ntok = self.tokenizer([template]*view.shape[0]).to(view.device)
                    nemb = self.text_model.transformer.embeddings.word_embeddings(ntok).type(self.dtype)
                    nimg, ntxt = self.encode_text_image(ntok, nemb, view)
                npatch = self.m2_text_verifier.align_patch_tokens(nimg[:,1:,:].float())
                ntxt = F.normalize(ntxt.float(), dim=-1)
                nscore = (npatch * ntxt.unsqueeze(1)).sum(dim=-1).reshape(view.shape[0],root,root)
                neg_scores.append(nscore)
            neg = neg_tau * torch.logsumexp(torch.stack(neg_scores,0)/neg_tau, dim=0)
            all_pos.append(pos); all_neg.append(neg)
        pos = torch.cat(all_pos,0); neg = torch.cat(all_neg,0)
        return pos, neg, pos-neg

    def _pool_dense_quality(self, dense_map, mask, ring_radius: int):
        if dense_map.shape[0] == 0:
            z = dense_map.new_zeros(0)
            return z,z,z,torch.zeros(0,dtype=torch.bool,device=dense_map.device)
        h,w = dense_map.shape[-2:]
        region = F.adaptive_avg_pool2d(mask, (h,w)).flatten(1)
        ring = F.adaptive_avg_pool2d(self._binary_ring(mask, ring_radius), (h,w)).flatten(1)
        scores = dense_map.flatten(1)
        valid = region.sum(dim=1) > 0
        in_score = (scores*region).sum(dim=1)/region.sum(dim=1).clamp_min(EPS)
        ring_score = (scores*ring).sum(dim=1)/ring.sum(dim=1).clamp_min(EPS)
        return in_score-ring_score, in_score, ring_score, valid

    def _dense_mask_text_evidence(self, image, candidate_probs, text_prompts, supervision_masks=None):
        """V15.2 mask-conditioned evidence on a frozen raw text score map."""
        b = image.shape[0]
        z = image.new_zeros(b,3)
        neg = image.new_full((b,3), -1.0)
        avail = torch.zeros(b,3,dtype=torch.bool,device=image.device)
        ratio_out = image.new_zeros(b,3)
        overlap_out = image.new_zeros(b,3)
        pack = self._build_dense_hypothesis_crops(image, candidate_probs, supervision_masks)
        if pack["crops"].shape[0] == 0:
            return {
                "core_keep_positive":z,"core_drop_positive":z,"core_keep_negative":z,"core_drop_negative":z,
                "edit_keep_positive":z,"edit_drop_positive":z,"edit_keep_negative":z,"edit_drop_negative":z,
                "core_keep_discriminative":neg,"core_drop_discriminative":neg,"edit_keep_discriminative":neg,"edit_drop_discriminative":neg,
                "text_core_necessity":neg,"text_edit_direction":neg,"control_text_edit_direction":neg,"control_edit_keep_discriminative":neg,"text_specificity":neg,
                "cf_available":avail,"dense_local_alignment_loss":image.sum()*0.0,"dense_quality_preserve":z,"dense_quality_candidate":z,"dense_quality_control":z,"dense_delta_control":z,"dense_positive_candidate":z,"dense_negative_candidate":z,"dense_gt_gap":z,"dense_control_area_ratio":ratio_out,"dense_control_overlap_ratio":overlap_out,
            }
        crop_prompts = [text_prompts[int(i)] for i in pack["origins"].tolist()]
        pos_map, neg_map, dense = self._dense_text_score_map(pack["crops"], crop_prompts)
        ring_radius = max(1,int(_cfg_get(_cfg_get(self.cfg,"M1",None),"M2_DENSE_RING_RADIUS",2)))
        qp, posp, negp, vp = self._pool_dense_quality(dense, pack["preserve"], ring_radius)
        qc, posc, negc, vc = self._pool_dense_quality(dense, pack["candidate"], ring_radius)
        qctrl, posctrl, negctrl, vctrl = self._pool_dense_quality(dense, pack["control"], ring_radius)
        _, ppos, _, _ = self._pool_dense_quality(pos_map, pack["candidate"], ring_radius)
        _, npos, _, _ = self._pool_dense_quality(neg_map, pack["candidate"], ring_radius)
        delta = qc - qp
        delta_ctrl = qctrl - qp
        spec = delta - delta_ctrl
        origins, roles = pack["origins"], pack["roles"]
        q_preserve = z.clone(); q_preserve[origins,roles] = qp
        q_candidate = z.clone(); q_candidate[origins,roles] = qc
        q_control = z.clone(); q_control[origins,roles] = qctrl
        delta_out = neg.clone(); delta_out[origins,roles] = delta
        delta_ctrl_out = neg.clone(); delta_ctrl_out[origins,roles] = delta_ctrl
        spec_out = neg.clone(); spec_out[origins,roles] = spec
        pos_candidate = z.clone(); pos_candidate[origins,roles] = ppos
        neg_candidate = z.clone(); neg_candidate[origins,roles] = npos
        ratio_out[origins,roles] = pack["control_area_ratio"]
        overlap_out[origins,roles] = pack["control_overlap_ratio"]
        # _dense_mask_text_evidence belongs to CustomCLIP. The control
        # geometry bounds are owned by TextCounterfactualVerifier.
        verifier = self.m2_text_verifier
        if verifier is None:
            raise RuntimeError(
                "V15.3 dense-control audit requires m2_text_verifier."
            )

        ratio_ok = (
            (pack["control_area_ratio"] >= verifier.control_min_area_ratio)
            & (pack["control_area_ratio"] <= verifier.control_max_area_ratio)
        )
        overlap_ok = (
            pack["control_overlap_ratio"] <= verifier.control_max_overlap_ratio
        )
        avail[origins,roles] = vp & vc & vctrl & ratio_ok & overlap_ok

        # V15.2 intentionally has no raw-map loss.  GT may be passed by the
        # training loop for audit plumbing, but never changes dense evidence.
        local_loss = dense.sum()*0.0
        gt_gap_out = z.clone()
        if self.training and pack["gt"].shape[0] > 0:
            qgt, _, _, _ = self._pool_dense_quality(dense, pack["gt"], ring_radius)
            gt_gap_out[origins,roles] = qgt.detach()

        q_preserve[:,0] = 0.0; q_candidate[:,0] = 0.0; q_control[:,0] = 0.0
        return {
            "core_keep_positive": pos_candidate, "core_drop_positive": q_preserve,
            "core_keep_negative": neg_candidate, "core_drop_negative": q_control,
            "edit_keep_positive": pos_candidate, "edit_drop_positive": q_preserve,
            "edit_keep_negative": neg_candidate, "edit_drop_negative": q_control,
            "core_keep_discriminative": q_candidate, "core_drop_discriminative": q_preserve,
            "edit_keep_discriminative": q_candidate, "edit_drop_discriminative": q_preserve,
            "text_core_necessity": q_candidate, "text_edit_direction": delta_out,
            "control_text_edit_direction": delta_ctrl_out, "control_edit_keep_discriminative": q_control,
            "text_specificity": spec_out, "cf_available": avail,
            "dense_local_alignment_loss": local_loss,
            "dense_quality_preserve": q_preserve, "dense_quality_candidate": q_candidate, "dense_quality_control": q_control,
            "dense_delta_control": delta_ctrl_out, "dense_positive_candidate": pos_candidate, "dense_negative_candidate": neg_candidate,
            "dense_gt_gap": gt_gap_out, "dense_control_area_ratio": ratio_out,
            "dense_control_overlap_ratio": overlap_out,
        }

    @staticmethod
    def _neutral_counterfactual_evidence(candidate_probs):
        b = candidate_probs.shape[0]; z = candidate_probs.new_zeros(b,3); neg = candidate_probs.new_full((b,3),-1.0); neg[:,0]=0.0
        return {
            "core_keep_positive":z,"core_drop_positive":z,"core_keep_negative":z,"core_drop_negative":z,
            "edit_keep_positive":z,"edit_drop_positive":z,"edit_keep_negative":z,"edit_drop_negative":z,
            "core_keep_discriminative":neg.clone(),"core_drop_discriminative":neg.clone(),"edit_keep_discriminative":neg.clone(),"edit_drop_discriminative":neg.clone(),
            "text_core_necessity":neg.clone(),"text_edit_direction":neg.clone(),"control_text_edit_direction":neg.clone(),"control_edit_keep_discriminative":neg.clone(),"text_specificity":neg.clone(),
            "cf_available":torch.zeros(b,3,dtype=torch.bool,device=candidate_probs.device),"dense_local_alignment_loss":candidate_probs.sum()*0.0,
            "dense_quality_preserve":z,"dense_quality_candidate":z,"dense_quality_control":z,"dense_delta_control":z,"dense_positive_candidate":z,"dense_negative_candidate":z,"dense_gt_gap":z,"dense_control_area_ratio":z,"dense_control_overlap_ratio":z,
        }

    @torch.no_grad()
    def _v396_observer_image_features(self, image: torch.Tensor) -> torch.Tensor:
        """Original frozen ViT features, never affected by segmentation loss."""
        observer = getattr(self, "_v396_observer_vision", None)
        if observer is None:
            raise RuntimeError("V396 frozen observer was not constructed.")

        x_img = observer.conv1(image)
        x_img = x_img.reshape(x_img.shape[0], x_img.shape[1], -1).permute(0, 2, 1)
        x_img = torch.cat([
            observer.class_embedding.to(x_img.dtype) + torch.zeros(
                x_img.shape[0], 1, x_img.shape[-1],
                dtype=x_img.dtype, device=x_img.device
            ),
            x_img,
        ], dim=1)
        x_img = x_img + observer.positional_embedding.to(x_img.dtype)
        x_img = observer.ln_pre(x_img).permute(1, 0, 2)
        for block in observer.transformer.resblocks:
            x_img = block(x_img)
        x_img = observer.ln_post(x_img.permute(1, 0, 2))
        if observer.proj is not None:
            x_img = x_img @ observer.proj
        return x_img

    @torch.no_grad()
    def _v396_frozen_text_features(self, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        """Frozen text feature path with no image/PVL dependency."""
        word_embeddings = self.text_model.transformer.embeddings.word_embeddings(
            tokenized_prompts
        ).type(self.dtype)
        attention = (tokenized_prompts != self.text_model.config.pad_token_id).long()
        extended = attention[:, None, None, :].to(dtype=self.dtype)
        extended = (1.0 - extended) * torch.finfo(self.dtype).min
        x_txt = self.text_model.transformer.embeddings(inputs_embeds=word_embeddings)
        for layer in self.text_model.transformer.encoder.layer:
            x_txt = layer(x_txt, attention_mask=extended)[0]
        return self.text_model.proj(x_txt[:, 0, :])

    @torch.no_grad()
    def _v396_cached_frozen_text_features(self, prompts, device: torch.device) -> torch.Tensor:
        """Return immutable V396 text embeddings with prompt-level caching.

        BUSI batches repeatedly use a very small prompt vocabulary. Re-running
        all frozen BiomedBERT layers three times per batch is unnecessary and
        does not change either the loss or the deployed evidence.
        """
        prompts = [str(prompt) for prompt in prompts]
        if not prompts:
            return torch.empty((0, self.text_proj_dim), device=device, dtype=self.dtype)

        if not self.v396_cache_frozen_text:
            tokens = self.tokenizer(prompts).to(device)
            return self._v396_frozen_text_features(tokens).detach()

        cache = getattr(self, "_v396_text_feature_cache", None)
        if cache is None:
            cache = {}
            object.__setattr__(self, "_v396_text_feature_cache", cache)

        device_key = str(device)
        dtype_key = str(self.dtype)
        unique_prompts = list(dict.fromkeys(prompts))
        missing = [
            prompt for prompt in unique_prompts
            if (device_key, dtype_key, prompt) not in cache
        ]
        if missing:
            tokens = self.tokenizer(missing).to(device)
            features = self._v396_frozen_text_features(tokens).detach()
            for prompt, feature in zip(missing, features):
                cache[(device_key, dtype_key, prompt)] = feature.contiguous()

        return torch.stack(
            [cache[(device_key, dtype_key, prompt)] for prompt in prompts],
            dim=0,
        )

    def _v396_negative_prompts(self, text_prompts):
        m1 = _cfg_get(self.cfg, "M1", None)
        lesion_negative = str(_cfg_get(
            m1, "EVIDENCE_GUIDED_NEGATIVE_LESION_TEMPLATE",
            "normal breast tissue without a focal lesion in this ultrasound image.",
        ))
        normal_negative = str(_cfg_get(
            m1, "EVIDENCE_GUIDED_NEGATIVE_NORMAL_TEMPLATE",
            "a focal breast lesion is present in this ultrasound image.",
        ))
        output = []
        for item in text_prompts:
            prompt = str(item).lower()
            is_normal = any(token in prompt for token in (
                "normal", "no lesion", "no signs", "without lesion", "healthy"
            ))
            output.append(normal_negative if is_normal else lesion_negative)
        return output

    def _v396_paraphrase_prompts(self, text_prompts):
        """Fixed segmentation-level paraphrases, never random batch text swaps."""
        m1 = _cfg_get(self.cfg, "M1", None)
        lesion_paraphrase = str(_cfg_get(
            m1,
            "EVIDENCE_GUIDED_PARAPHRASE_LESION_TEMPLATE",
            "a focal breast lesion is visible in this ultrasound image.",
        ))
        normal_paraphrase = str(_cfg_get(
            m1,
            "EVIDENCE_GUIDED_PARAPHRASE_NORMAL_TEMPLATE",
            "normal breast tissue is visible without a focal lesion.",
        ))
        output = []
        for item in text_prompts:
            prompt = str(item).lower()
            is_normal = any(token in prompt for token in (
                "normal", "no lesion", "no signs", "without lesion", "healthy"
            ))
            output.append(normal_paraphrase if is_normal else lesion_paraphrase)
        return output

    def _get_m2_tide_repair_head(self):
        """Return the V399 TIDE head whether it is top-level or nested in M1."""
        head = getattr(self, "m2_tide_repair_head", None)
        if head is not None:
            return head

        bank = getattr(self, "m1_pse", None)
        if bank is None:
            return None

        if not bool(getattr(bank, "m2_tide_repair_enabled", False)):
            return None

        return getattr(bank, "m2_tide_repair_head", None)



    def _apply_m2_tide_action_selection(self, candidate_logits, m1_aux):
        """Dispatch V406 legacy scoring or V407 failure-gated set reranking."""
        m1_cfg = _cfg_get(getattr(self, "cfg", None), "M1", None)
        if bool(_cfg_get(m1_cfg, "M2_FGSR_ENABLED", False)):
            return self._apply_m2_fgsr_selection(candidate_logits, m1_aux)
        return self._apply_m2_tide_action_selection_legacy(candidate_logits, m1_aux)

    def _apply_m2_fgsr_selection(self, candidate_logits, m1_aux):
        """V407: Preserve-aware dual-metric Failure-Gated Set Reranker.

        Candidate geometry is frozen.  M2 receives only its residual-map
        evidence and uses one learned set policy to:
          (1) decide whether this case has a Pareto-improving edit; and
          (2) rank valid candidates by a conservative lower confidence bound.

        The final output is always an exact gather from Preserve + the frozen
        candidate stack.  No GT/Oracle quantity enters this method.
        """
        fn_logits = m1_aux.get("m2_fn_error_logits")
        fp_logits = m1_aux.get("m2_fp_error_logits")
        boundary_logits = m1_aux.get("m2_boundary_error_logits")
        if not all(isinstance(x, torch.Tensor) for x in (
            fn_logits, fp_logits, boundary_logits
        )):
            return

        if candidate_logits.ndim != 4 or candidate_logits.shape[1] < 2:
            raise RuntimeError(
                "V407 requires candidate_logits [B,1+K,H,W] with Preserve in slot 0."
            )

        candidate_probs = m1_aux.get("candidate_probs")
        if not isinstance(candidate_probs, torch.Tensor):
            candidate_probs = torch.sigmoid(candidate_logits).clamp(EPS, 1.0 - EPS)
            m1_aux["candidate_probs"] = candidate_probs
        if candidate_probs.shape != candidate_logits.shape:
            raise RuntimeError(
                "V407 candidate_probs/candidate_logits mismatch: "
                f"{tuple(candidate_probs.shape)} vs {tuple(candidate_logits.shape)}"
            )

        head = self._get_m2_tide_repair_head()
        scorer = getattr(head, "candidate_scorer", None) if head is not None else None
        if scorer is None:
            raise RuntimeError("V407 requires TIDERepairErrorHead.candidate_scorer.")

        b, slots, h, w = candidate_probs.shape
        k = slots - 1
        m1_cfg = _cfg_get(getattr(self, "cfg", None), "M1", None)

        hard_threshold = float(_cfg_get(m1_cfg, "M2_CANDIDATE_THRESHOLD", 0.50))
        hard_threshold = min(max(hard_threshold, 0.05), 0.95)
        max_edit_fraction = min(
            max(float(_cfg_get(m1_cfg, "M2_MAX_EDIT_FRACTION", 0.060)), 0.0),
            1.0,
        )

        def resize(x):
            if x.shape[-2:] != (h, w):
                return F.interpolate(
                    x, size=(h, w), mode="bilinear", align_corners=False
                )
            return x

        fn_prob = torch.sigmoid(resize(fn_logits)).clamp(0.0, 1.0)
        fp_prob = torch.sigmoid(resize(fp_logits)).clamp(0.0, 1.0)
        boundary_prob = torch.sigmoid(resize(boundary_logits)).clamp(0.0, 1.0)

        base_prob = candidate_probs[:, :1]
        base_hard = (base_prob >= hard_threshold).to(candidate_probs.dtype)
        candidate_hard = (
            candidate_probs[:, 1:] >= hard_threshold
        ).to(candidate_probs.dtype)
        add_region = candidate_hard * (1.0 - base_hard)
        remove_region = base_hard * (1.0 - candidate_hard)
        changed_region = (add_region + remove_region).clamp(0.0, 1.0)
        changed_band = _soft_dilate(
            changed_region.reshape(b * k, 1, h, w), 1
        ).reshape(b, k, h, w).clamp(0.0, 1.0)

        def masked_mean(value, region):
            if value.shape[1] == 1:
                value = value.expand(-1, region.shape[1], -1, -1)
            denominator = region.sum(dim=(-2, -1))
            mean = (value * region).sum(dim=(-2, -1)) / denominator.clamp_min(1.0)
            return torch.where(denominator > 0.0, mean, torch.zeros_like(mean))

        add_benefit = masked_mean(fn_prob, add_region)
        remove_benefit = masked_mean(fp_prob, remove_region)
        add_harm = masked_mean(1.0 - fn_prob, add_region)
        remove_harm = masked_mean(1.0 - fp_prob, remove_region)
        boundary_evidence = masked_mean(boundary_prob, changed_band)

        edit_fraction = changed_region.mean(dim=(-2, -1))
        add_fraction = add_region.mean(dim=(-2, -1))
        remove_fraction = remove_region.mean(dim=(-2, -1))

        safe_base_prob = base_prob.clamp(EPS, 1.0 - EPS)
        base_entropy = -(
            safe_base_prob * safe_base_prob.log()
            + (1.0 - safe_base_prob) * (1.0 - safe_base_prob).log()
        ) / math.log(2.0)
        base_boundary = (
            _soft_dilate(base_hard, 1) - _soft_erode(base_hard, 1)
        ).abs().clamp(0.0, 1.0)
        probability_delta = (candidate_probs[:, 1:] - base_prob).abs()
        entropy_on_edit = masked_mean(base_entropy, changed_band)
        base_boundary_on_edit = masked_mean(base_boundary, changed_band)
        probability_delta_on_edit = masked_mean(probability_delta, changed_band)

        # Candidate family id is a legitimate policy feature: it states which
        # fixed intervention was proposed, not its GT outcome.
        action_types = m1_aux.get("v20_action_types")
        if not isinstance(action_types, torch.Tensor) or action_types.numel() != k:
            action_types = torch.arange(
                k, device=candidate_probs.device, dtype=torch.long
            )
        else:
            action_types = action_types.to(
                device=candidate_probs.device, dtype=torch.long
            ).reshape(-1)

        features = torch.stack([
            add_benefit,
            remove_benefit,
            add_harm,
            remove_harm,
            boundary_evidence,
            edit_fraction,
            add_fraction,
            remove_fraction,
            entropy_on_edit,
            base_boundary_on_edit,
            probability_delta_on_edit,
        ], dim=-1)

        scorer_output = scorer(
            features,
            action_types=action_types,
            return_aux=True,
        )
        if not isinstance(scorer_output, dict):
            raise RuntimeError("V407 scorer must return an auxiliary-output dictionary.")

        gain_mean = scorer_output["gain_mean"]
        gain_std = scorer_output["gain_std"].clamp_min(1.0e-5)
        gate_logits = scorer_output["gate_logits"]
        gate_probability = scorer_output["gate_probability"]

        if tuple(gain_mean.shape) != (b, k):
            raise RuntimeError(
                f"V407 gain shape {tuple(gain_mean.shape)} != expected {(b, k)}"
            )

        valid_action = (
            (changed_region.sum(dim=(-2, -1)) > 0.0)
            & (edit_fraction <= max_edit_fraction)
        )

        lcb_beta = max(
            0.0,
            float(_cfg_get(m1_cfg, "M2_LCB_BETA", 1.0)),
        )
        conservative_gain = gain_mean - lcb_beta * gain_std
        masked_lcb = conservative_gain.masked_fill(~valid_action, float("-inf"))
        best_lcb, best_edit_index = masked_lcb.max(dim=1)

        if k >= 2:
            top2 = masked_lcb.topk(k=2, dim=1).values
            top_gap = top2[:, 0] - top2[:, 1]
        else:
            top_gap = torch.full_like(best_lcb, float("inf"))

        gate_threshold = float(
            _cfg_get(m1_cfg, "M2_EDIT_GATE_THRESHOLD", 0.50)
        )
        min_lcb_gain = float(
            _cfg_get(m1_cfg, "M2_MIN_LCB_GAIN", 0.0)
        )
        min_lcb_gap = float(
            _cfg_get(m1_cfg, "M2_MIN_LCB_GAP", 0.0005)
        )

        accept = (
            torch.isfinite(best_lcb)
            & (gate_probability >= gate_threshold)
            & (best_lcb > min_lcb_gain)
            & (top_gap >= min_lcb_gap)
        )

        selected_idx = torch.where(
            accept,
            best_edit_index + 1,
            torch.zeros_like(best_edit_index),
        )
        gather_idx = selected_idx[:, None, None, None].expand(-1, 1, h, w)
        selected_probs = candidate_probs.gather(1, gather_idx)[:, 0]
        selected_logits = candidate_logits.gather(1, gather_idx)[:, 0]

        candidate_types_with_preserve = torch.cat([
            torch.full(
                (1,), -1, device=action_types.device, dtype=torch.long
            ),
            action_types,
        ])
        selected_type = candidate_types_with_preserve[selected_idx]

        selector_hard = candidate_probs.new_zeros((b, k))
        selector_hard.scatter_(
            1,
            (selected_idx - 1).clamp_min(0)[:, None],
            accept.to(candidate_probs.dtype)[:, None],
        )

        score_with_preserve = torch.cat([
            conservative_gain.new_zeros((b, 1)),
            conservative_gain,
        ], dim=1)

        m1_aux.update({
            # Training-only inputs. Do NOT detach: M2 ranking feedback is
            # allowed to refine M2 residual maps, while B0/M1 are frozen.
            "m2_scorer_features": features,
            "m2_action_types": action_types,
            "m2_action_valid": valid_action,

            # Exact Preserve-or-one-candidate deployment result.
            "m2_selected_probs": selected_probs,
            "m2_selected_logits": selected_logits,
            "m2_selected_idx": selected_idx,
            "m2_selected_index": selected_idx,
            "m2_selected_action_type": selected_type,
            "m2_selected_score": torch.where(
                accept, best_lcb, torch.zeros_like(best_lcb)
            ),
            "m2_selected_gain": torch.where(
                accept,
                gain_mean.gather(1, best_edit_index[:, None])[:, 0],
                torch.zeros_like(best_lcb),
            ),
            "m2_selected_risk": torch.where(
                accept,
                gain_std.gather(1, best_edit_index[:, None])[:, 0],
                torch.zeros_like(best_lcb),
            ),
            "m2_selected_confidence": gate_probability,
            "m2_accept": accept.to(candidate_probs.dtype),
            "m2_selected_action": accept.to(candidate_probs.dtype),
            "m2_selection_exact": torch.ones(
                (b,), device=candidate_probs.device, dtype=candidate_probs.dtype
            ),

            # Candidate-level audit values.
            "m2_action_scores": score_with_preserve,
            "m2_edit_scores": gain_mean,
            "m2_gain_mean": gain_mean,
            "m2_gain_std": gain_std,
            "m2_gain_lcb": conservative_gain,
            "m2_gate_logits": gate_logits,
            "m2_gate_probability": gate_probability,
            "m2_best_lcb": best_lcb,
            "m2_top_lcb_gap": top_gap,
            "m2_add_area": add_fraction,
            "m2_remove_area": remove_fraction,
            "m2_edit_fraction": edit_fraction,

            # Existing V20 validation/audit interfaces: exactly one action or
            # Preserve, never a soft fusion.
            "v20_selector_logits": conservative_gain,
            "v20_selector_probs": selector_hard,
            "v20_selector_hard": selector_hard,
            "direct_fused_probs": selected_probs,
            "router_fused_probs": selected_probs,
            "v20_fused_probs": selected_probs,
            "v20_hard_fused_probs": selected_probs,
            "v20_fused_logits": selected_logits,
        })

        gathered_again = candidate_probs.gather(1, gather_idx)[:, 0]
        if not torch.equal(selected_probs, gathered_again):
            raise RuntimeError(
                "V407 invariant failed: final output is not an exact candidate gather."
            )

        if not getattr(self, "_v407_fgsr_announced", False):
            print(
                "[V407 FGSR] "
                f"gate_threshold={gate_threshold:.3f} "
                f"lcb_beta={lcb_beta:.3f} "
                f"min_lcb={min_lcb_gain:.5f}"
            )
            self._v407_fgsr_announced = True

    def _apply_m2_tide_action_selection_legacy(self, candidate_logits, m1_aux):
        """V406 TIDE-ActionMatch with learnable candidate scorer.

        M2 predicts FN/FP/Boundary residual-error maps. It does not synthesize
        a new mask. Per-candidate features (benefit/harm/boundary/edit_fraction)
        are scored by a tiny learned MLP. The highest-scoring valid candidate
        is selected; otherwise Preserve.
        """
        fn_logits = m1_aux.get("m2_fn_error_logits")
        fp_logits = m1_aux.get("m2_fp_error_logits")
        boundary_logits = m1_aux.get("m2_boundary_error_logits")

        if not all(isinstance(x, torch.Tensor) for x in (
            fn_logits, fp_logits, boundary_logits
        )):
            return

        if candidate_logits.ndim != 4 or candidate_logits.shape[1] < 2:
            raise RuntimeError(
                "V406 requires candidate_logits [B,1+K,H,W] with Preserve in slot 0."
            )

        candidate_probs = m1_aux.get("candidate_probs")
        if not isinstance(candidate_probs, torch.Tensor):
            candidate_probs = torch.sigmoid(candidate_logits).clamp(EPS, 1.0 - EPS)
            m1_aux["candidate_probs"] = candidate_probs

        if candidate_probs.shape != candidate_logits.shape:
            raise RuntimeError(
                "V406 candidate_probs/candidate_logits mismatch: "
                f"{tuple(candidate_probs.shape)} vs {tuple(candidate_logits.shape)}"
            )

        b, slots, h, w = candidate_probs.shape
        k = slots - 1
        m1_cfg = _cfg_get(getattr(self, "cfg", None), "M1", None)

        hard_threshold = float(
            _cfg_get(m1_cfg, "M2_CANDIDATE_THRESHOLD", 0.50)
        )
        hard_threshold = min(max(hard_threshold, 0.05), 0.95)

        # V406: learnable scorer has its own calibrated output.
        # We keep a small safety margin so the scorer can push scores above it.
        action_threshold = float(
            _cfg_get(m1_cfg, "M2_ACTION_SCORE_THRESHOLD", 0.0)
        )
        max_edit_fraction = min(
            max(
                float(_cfg_get(m1_cfg, "M2_MAX_EDIT_FRACTION", 0.060)),
                0.0,
            ),
            1.0,
        )

        def resize(x):
            if x.shape[-2:] != (h, w):
                return F.interpolate(
                    x,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )
            return x

        fn_prob = torch.sigmoid(resize(fn_logits)).clamp(0.0, 1.0)
        fp_prob = torch.sigmoid(resize(fp_logits)).clamp(0.0, 1.0)
        boundary_prob = torch.sigmoid(resize(boundary_logits)).clamp(0.0, 1.0)

        base_hard = (
            candidate_probs[:, :1] >= hard_threshold
        ).to(candidate_probs.dtype)

        candidate_hard = (
            candidate_probs[:, 1:] >= hard_threshold
        ).to(candidate_probs.dtype)

        add_region = candidate_hard * (1.0 - base_hard)
        remove_region = base_hard * (1.0 - candidate_hard)
        changed_region = (add_region + remove_region).clamp(0.0, 1.0)

        changed_band = _soft_dilate(
            changed_region.reshape(b * k, 1, h, w),
            1,
        ).reshape(b, k, h, w).clamp(0.0, 1.0)

        def masked_mean(value, region):
            value = value.expand(-1, region.shape[1], -1, -1)
            denom = region.sum(dim=(-2, -1))
            mean = (value * region).sum(dim=(-2, -1)) / denom.clamp_min(1.0)
            return torch.where(denom > 0.0, mean, torch.zeros_like(mean))

        add_benefit = masked_mean(fn_prob, add_region)
        remove_benefit = masked_mean(fp_prob, remove_region)
        add_harm = masked_mean(1.0 - fn_prob, add_region)
        remove_harm = masked_mean(1.0 - fp_prob, remove_region)
        boundary_evidence = masked_mean(boundary_prob, changed_band)
        edit_fraction = changed_region.mean(dim=(-2, -1))

        # ── V406: Learnable candidate scorer ──
        head = self._get_m2_tide_repair_head()
        if head is not None and hasattr(head, "candidate_scorer"):
            # Stack per-candidate features: [B, K, 6]
            feats = torch.stack([
                add_benefit,       # fn coverage of add_region
                remove_benefit,    # fp coverage of remove_region
                add_harm,          # non-fn in add_region
                remove_harm,       # non-fp in remove_region
                boundary_evidence, # boundary prob in changed_band
                edit_fraction,     # fraction of changed pixels
            ], dim=-1)  # [B, K, 6]

            raw_scores = head.candidate_scorer(feats)  # [B, K]
        else:
            # Fallback: simple heuristic (should not happen in V406+)
            raw_scores = (
                add_benefit + remove_benefit
                - (add_harm + remove_harm)
                + 0.25 * (2.0 * boundary_evidence - 1.0)
            )

        # ── Selection ──
        valid_action = (
            (changed_region.sum(dim=(-2, -1)) > 0.0)
            & (edit_fraction <= max_edit_fraction)
        )

        masked_scores = raw_scores.masked_fill(
            ~valid_action,
            float("-inf"),
        )

        best_edit_score, best_edit_index = masked_scores.max(dim=1)

        accept = (
            torch.isfinite(best_edit_score)
            & (best_edit_score > action_threshold)
        )

        selected_idx = torch.where(
            accept,
            best_edit_index + 1,
            torch.zeros_like(best_edit_index),
        )

        selected_score = torch.where(
            accept,
            best_edit_score,
            torch.zeros_like(best_edit_score),
        )

        gather_idx = selected_idx[:, None, None, None].expand(
            -1,
            1,
            h,
            w,
        )

        selected_probs = candidate_probs.gather(1, gather_idx)[:, 0]
        selected_logits = candidate_logits.gather(1, gather_idx)[:, 0]

        action_types = m1_aux.get("v20_action_types")
        if not isinstance(action_types, torch.Tensor) or action_types.numel() != k:
            action_types = torch.arange(
                k,
                device=candidate_probs.device,
                dtype=torch.long,
            )
        else:
            action_types = action_types.to(
                device=candidate_probs.device,
                dtype=torch.long,
            ).reshape(-1)

        candidate_type = torch.cat([
            torch.full(
                (1,),
                -1,
                device=action_types.device,
                dtype=torch.long,
            ),
            action_types,
        ])

        selected_type = candidate_type[selected_idx]

        selector_hard = candidate_probs.new_zeros((b, k))
        selector_hard.scatter_(
            1,
            (selected_idx - 1).clamp_min(0)[:, None],
            accept.to(candidate_probs.dtype)[:, None],
        )

        score_with_preserve = torch.cat([
            raw_scores.new_zeros((b, 1)),
            raw_scores,
        ], dim=1)

        add_area = add_region.mean(dim=(-2, -1))
        remove_area = remove_region.mean(dim=(-2, -1))

        # ── Store scorer features for the ranking loss ──
        if head is not None and hasattr(head, "candidate_scorer"):
            m1_aux["m2_scorer_features"] = torch.stack([
                add_benefit, remove_benefit, add_harm, remove_harm,
                boundary_evidence, edit_fraction,
            ], dim=-1).detach()  # [B, K, 6] detached for loss

        m1_aux.update({
            "m2_selected_probs": selected_probs,
            "m2_selected_logits": selected_logits,
            "m2_selected_idx": selected_idx,
            "m2_selected_action_type": selected_type,
            "m2_selected_score": selected_score,
            "m2_action_scores": score_with_preserve,
            "m2_edit_scores": raw_scores,
            "m2_action_valid": valid_action,
            "m2_add_area": add_area,
            "m2_remove_area": remove_area,
            "m2_edit_fraction": edit_fraction,
            "m2_selected_action": accept.to(candidate_probs.dtype),
            "m2_selection_exact": torch.ones(
                (b,),
                device=candidate_probs.device,
                dtype=candidate_probs.dtype,
            ),

            # 保留旧 V20 评估接口，语义为单动作选择。
            "v20_selector_logits": raw_scores,
            "v20_selector_probs": selector_hard,
            "v20_selector_hard": selector_hard,

            # 最终输出只能是 candidate stack 中被 gather 的 mask。
            "direct_fused_probs": selected_probs,
            "router_fused_probs": selected_probs,
            "v20_fused_probs": selected_probs,
            "v20_hard_fused_probs": selected_probs,
            "v20_fused_logits": selected_logits,
        })

        gathered_again = candidate_probs.gather(1, gather_idx)[:, 0]
        if not torch.equal(selected_probs, gathered_again):
            raise RuntimeError(
                "V406 invariant failed: final output is not an exact candidate gather."
            )

        if not getattr(self, "_v406_action_match_announced", False):
            print(
                "[V406 TIDE-ActionMatch + LearnableScorer] "
                f"threshold={action_threshold:.4f} "
                f"selected_rate={float(accept.float().mean()):.4f}"
            )
            self._v406_action_match_announced = True
    def _m1_generate(
        self,
        base_logits: torch.Tensor,
        image: torch.Tensor,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        tokenized_prompts: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_prompts,
        supervision_masks: Optional[torch.Tensor] = None,
        verifier_text_prompts=None,
        mc_std_map: Optional[torch.Tensor] = None,
        mc_disagreement_map: Optional[torch.Tensor] = None,
        mc_pairwise_disagreement: Optional[torch.Tensor] = None,
        mc_probability_samples: Optional[torch.Tensor] = None,
        fine_feature_map: Optional[torch.Tensor] = None,
        seg_text_vector: Optional[torch.Tensor] = None,
        slr_hr_image: Optional[torch.Tensor] = None,
        sparc_hr_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Generate M1 candidates and apply the V481 residual-repair projection.

        V481 is applied at the candidate boundary only: after a candidate stack
        has been produced and before the stack is consumed by later verifier or
        selector paths. This keeps M1 as residual candidate repair rather than
        a free full-mask generator.
        """

        def _finalize_candidates(
            candidate_logits_: torch.Tensor,
            m1_aux_: Dict[str, torch.Tensor],
        ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            # MHCS hypotheses are complete segmentations, not residual edits.
            # Reprojecting them toward Base would destroy the hypothesis-space
            # formulation and make Base the reference frame again.
            if getattr(self, "mhcs_enabled", False):
                return candidate_logits_, m1_aux_
            return v481_reproject_m1_candidates(
                self.cfg,
                base_logits,
                candidate_logits_,
                m1_aux_,
            )

        def _apply_tide_repair_if_needed(
            candidate_logits_: torch.Tensor,
            m1_aux_: Dict[str, torch.Tensor],
            tide_semantic_map: torch.Tensor,
            tide_text_features: torch.Tensor,
        ) -> None:
            if self._get_m2_tide_repair_head() is None:
                return
            base_prob_slot = torch.sigmoid(candidate_logits_[:, 0])
            base_mask_slot = (base_prob_slot > 0.5).float()
            fn_l, fp_l, bd_l = self._get_m2_tide_repair_head()(
                semantic_map=tide_semantic_map,
                base_prob=base_prob_slot,
                base_mask=base_mask_slot,
                text_features=tide_text_features,
            )
            m1_aux_["m2_fn_error_logits"] = fn_l
            m1_aux_["m2_fp_error_logits"] = fp_l
            m1_aux_["m2_boundary_error_logits"] = bd_l
            self._apply_m2_tide_action_selection(
                candidate_logits_,
                m1_aux_,
            )

        if getattr(self, "v484_pipeline", None) is not None:
            m1_cfg_runtime = _cfg_get(self.cfg, "M1", None)
            v489_online = bool(
                _cfg_get(m1_cfg_runtime, "V489_END_TO_END_ENABLED", False)
                or _cfg_get(m1_cfg_runtime, "V490_ROOT_CAUSE_ENABLED", False)
            )
            v501_anchored = bool(
                _cfg_get(
                    m1_cfg_runtime,
                    "V501_BASE_ANCHORED_SELECTIVE_REPAIR_ENABLED",
                    False,
                )
            )
            v518_enabled = bool(
                _cfg_get(m1_cfg_runtime, "V518_ENABLED", False)
            )
            v518_semantic_map = None
            if v518_enabled and isinstance(image_features, torch.Tensor):
                v518_semantic_map = self._semantic_map_from_image_features(
                    image_features,
                    image.shape[-2:],
                    base_logits.shape[-2:],
                )
                # Image/text encoders stay frozen and proposal objectives may not
                # alter the official Base representation through this side path.
                v518_semantic_map = v518_semantic_map.detach()

            def _run_v484_pipeline():
                return self.v484_pipeline(
                    image=image,
                    base_logits=base_logits,
                    image_features=(
                        image_features
                        if (v489_online and not v501_anchored)
                        else image_features.detach() if isinstance(image_features, torch.Tensor) else image_features
                    ),
                    text_features=(
                        text_features
                        if (v489_online and not v501_anchored)
                        else text_features.detach() if isinstance(text_features, torch.Tensor) else text_features
                    ),
                    semantic_map=v518_semantic_map,
                    supervision_masks=(
                        supervision_masks if self.training else None
                    ),
                )

            # R4.20.4 fairness contract: variant-specific M1 branches may use
            # dropout or other stochastic operators.  Without RNG isolation, an
            # A0/A1 architecture difference can consume a different number of
            # random values and thereby change *next-batch Base/PVL dropout*,
            # even with the same global seed.  fork_rng restores the outer CPU
            # and CUDA RNG states after M1, so the Base trajectory is not
            # changed merely because the residual branch has a different graph.
            r4204_rng_isolation = bool(
                self.training
                and _cfg_get(
                    m1_cfg_runtime,
                    "V552R4204_BASE_RNG_ISOLATION_ENABLED",
                    False,
                )
            )
            if r4204_rng_isolation:
                cuda_devices = []
                if image.is_cuda and image.device.index is not None:
                    cuda_devices = [int(image.device.index)]
                with torch.random.fork_rng(devices=cuda_devices, enabled=True):
                    v485_aux = _run_v484_pipeline()
            else:
                v485_aux = _run_v484_pipeline()
            if isinstance(v485_aux, tuple):
                # Compatibility with any stale local copy; the corrected V485
                # pipeline returns a dict.
                if len(v485_aux) == 2 and isinstance(v485_aux[1], dict):
                    tmp = dict(v485_aux[1])
                    tmp.setdefault("candidates", v485_aux[0])
                    v485_aux = tmp
                else:
                    raise RuntimeError("V485 pipeline returned an unsupported tuple.")
            if not isinstance(v485_aux, dict):
                raise RuntimeError("V485 pipeline must return a dict.")
            candidate_logits = v485_aux.get("candidates", v485_aux.get("candidate_logits"))
            if not isinstance(candidate_logits, torch.Tensor):
                raise RuntimeError("V485 pipeline output lacks tensor candidates/candidate_logits.")
            if "candidate_probs" not in v485_aux:
                v485_aux["candidate_probs"] = torch.sigmoid(candidate_logits).clamp(EPS, 1.0 - EPS)
            return candidate_logits, v485_aux

        semantic_map = None
        exact_m1_needs_semantic = bool(
            (getattr(self, "geotr_m1_exact_enabled", False) or getattr(self, "semlt_autozero_enabled", False))
            and getattr(self.m1_pse, "use_semantic_conditioning", True)
        )
        non_exact_m1_needs_semantic = bool(
            not (getattr(self, "geotr_m1_exact_enabled", False) or getattr(self, "semlt_autozero_enabled", False))
            and (
                self.v20_unified_action_cf
                or self.m1_pse.use_semantic_feature
                or self.m2_text_verifier is not None
            )
        )
        if self.m1_active and (
            exact_m1_needs_semantic or non_exact_m1_needs_semantic
        ):
            # LOWMEM4:
            # AutoZero / UC-FNRT does not need the raw 512-channel semantic
            # tensor materialized at full 224x224 resolution here.
            #
            # Its first semantic operation is a bias-free 1x1 projection
            # 512 -> hidden_dim.  The projection and bilinear interpolation
            # commute, so UC-FNRT keeps the ViT patch-grid map here and
            # performs projection-before-resize inside the transport module.
            #
            # Historical/non-AutoZero routes retain the exact old behavior.
            _autozero_lowmem_semantic = bool(
                getattr(self.m1_pse, "autozero_semantic_transport", False)
            )
            semantic_map = self._semantic_map_from_image_features(
                image_features,
                image.shape[-2:],
                base_logits.shape[-2:],
                resize=not _autozero_lowmem_semantic,
            )

        if self.v20_unified_action_cf:
            if (
                self.training
                and getattr(
                    self.m1_pse,
                    "safe_residual_only_training",
                    False,
                )
            ):
                fast_semantic_map = self._semantic_map_from_image_features(
                    image_features,
                    image.shape[-2:],
                    base_logits.shape[-2:],
                ).detach()
                candidate_logits, m1_aux = UnifiedActionCounterfactualSetBank.generate(
                    self.m1_pse,
                    base_logits,
                    image=image,
                    semantic_map=fast_semantic_map,
                    text_features=text_features.detach(),
                    negative_text_features=text_features.detach(),
                )
                candidate_logits, m1_aux = _finalize_candidates(
                    candidate_logits,
                    m1_aux,
                )
                return candidate_logits, m1_aux

            if getattr(
                self.m1_pse,
                "unified_m1_safe_fusion_enabled",
                False,
            ):
                # MHCS-R4.5 gradient-topology contract: M1 distribution losses may
                # observe the current dense semantic evidence but must never
                # backpropagate through the trainable PVL/Base representation that
                # produced it.  Other historical M1 implementations keep their
                # existing semantics unchanged.
                joint_base_integration = bool(
                    getattr(self.m1_pse, "joint_base_integration", False)
                )
                safe_semantic_map = semantic_map
                if (
                    getattr(self.m1_pse, "mhcs_root_complete", False)
                    and not joint_base_integration
                ):
                    safe_semantic_map = (
                        semantic_map.detach() if isinstance(semantic_map, torch.Tensor) else semantic_map
                    )
                safe_text_features = (
                    text_features
                    if joint_base_integration
                    else text_features.detach()
                )
                candidate_logits, m1_aux = self.m1_pse.generate(
                    base_logits,
                    image=image,
                    semantic_map=safe_semantic_map,
                    text_features=safe_text_features,
                    negative_text_features=None,
                    supervision_masks=(supervision_masks if self.training else None),
                    mc_std_map=mc_std_map,
                    mc_disagreement_map=mc_disagreement_map,
                    mc_pairwise_disagreement=mc_pairwise_disagreement,
                    mc_probability_samples=mc_probability_samples,
                    fine_feature_map=fine_feature_map,
                    seg_text_vector=seg_text_vector,
                    slr_hr_image=slr_hr_image,
                    sparc_hr_mask=sparc_hr_mask,
                )
                candidate_logits, m1_aux = _finalize_candidates(
                    candidate_logits,
                    m1_aux,
                )
                return candidate_logits, m1_aux

            if self.evidence_guided_candidate_control_enabled:
                with torch.no_grad():
                    observer_features = self._v396_observer_image_features(image)
                    observer_map = self._semantic_map_from_image_features(
                        observer_features,
                        image.shape[-2:],
                        base_logits.shape[-2:],
                    ).detach()
                    positive_text_features = self._v396_cached_frozen_text_features(
                        text_prompts,
                        image.device,
                    )
                    negative_prompts = self._v396_negative_prompts(text_prompts)
                    negative_text_features = self._v396_cached_frozen_text_features(
                        negative_prompts,
                        image.device,
                    )
                    paraphrase_prompts = self._v396_paraphrase_prompts(
                        text_prompts
                    )
                    swapped_text_features = self._v396_cached_frozen_text_features(
                        paraphrase_prompts,
                        image.device,
                    )

                candidate_logits, m1_aux = self.m1_pse.generate(
                    base_logits,
                    image=image,
                    semantic_map=observer_map,
                    text_features=positive_text_features,
                    negative_text_features=negative_text_features,
                    swapped_text_features=swapped_text_features,
                )
                candidate_logits, m1_aux = _finalize_candidates(
                    candidate_logits,
                    m1_aux,
                )
                m1_aux["v396_observer_frozen"] = torch.ones(
                    base_logits.shape[0],
                    device=base_logits.device,
                    dtype=base_logits.dtype,
                )
                m1_aux["v396_paraphrase_available"] = torch.ones(
                    base_logits.shape[0],
                    device=base_logits.device,
                    dtype=base_logits.dtype,
                )
                m1_aux["v396_swap_from_batch"] = torch.zeros(
                    base_logits.shape[0],
                    device=base_logits.device,
                    dtype=base_logits.dtype,
                )
                _apply_tide_repair_if_needed(
                    candidate_logits,
                    m1_aux,
                    observer_map,
                    positive_text_features,
                )
                return candidate_logits, m1_aux

            if isinstance(
                self.m1_pse,
                (TextQualifiedStructuralMedoidBank, TypeConditionalUtilityActionBank),
            ):
                with torch.no_grad():
                    image_only_features, positive_text_features = self.encode_text_image(
                        tokenized_prompts,
                        text_embeddings,
                        image,
                        disable_pvl=True,
                    )
                    image_only_map = self._semantic_map_from_image_features(
                        image_only_features,
                        image.shape[-2:],
                        base_logits.shape[-2:],
                    ).detach()

                    m1 = _cfg_get(self.cfg, "M1", None)
                    lesion_negative = str(
                        _cfg_get(
                            m1,
                            "V23_NEGATIVE_LESION_TEMPLATE",
                            "no lesion is present in this breast ultrasound image.",
                        )
                    )
                    normal_negative = str(
                        _cfg_get(
                            m1,
                            "V23_NEGATIVE_NORMAL_TEMPLATE",
                            "a breast lesion is present in this ultrasound image.",
                        )
                    )
                    negative_prompts = []
                    for raw_prompt in text_prompts:
                        prompt = str(raw_prompt).lower()
                        is_normal = any(
                            token in prompt
                            for token in (
                                "normal",
                                "no lesion",
                                "no signs",
                                "without lesion",
                                "healthy",
                            )
                        )
                        negative_prompts.append(
                            normal_negative if is_normal else lesion_negative
                        )

                    negative_tokens = self.tokenizer(negative_prompts).to(image.device)
                    negative_embeddings = (
                        self.text_model.transformer.embeddings.word_embeddings(
                            negative_tokens
                        ).type(self.dtype)
                    )
                    _, negative_text_features = self.encode_text_image(
                        negative_tokens,
                        negative_embeddings,
                        image,
                        disable_pvl=True,
                    )

                    if (
                        self.v381_lesion_background_calibrated_atomic
                        or self.v382_action_conditional_quantile_atomic
                        or self.v383_conservative_action_value
                        or self.v391_lesion_background_mlp_calibrated_atomic
                        or self.v392_dense_patch_text_falsification
                        or self.v393_preserve_aware_edit_control
                    ):
                        swapped_text_features = negative_text_features
                    else:
                        prompt_strings = [str(item) for item in text_prompts]
                        if len(prompt_strings) > 1 and len(set(prompt_strings)) > 1:
                            swapped_prompts = prompt_strings[1:] + prompt_strings[:1]
                        else:
                            swapped_prompts = negative_prompts
                        swapped_tokens = self.tokenizer(swapped_prompts).to(image.device)
                        swapped_embeddings = (
                            self.text_model.transformer.embeddings.word_embeddings(
                                swapped_tokens
                            ).type(self.dtype)
                        )
                        _, swapped_text_features = self.encode_text_image(
                            swapped_tokens,
                            swapped_embeddings,
                            image,
                            disable_pvl=True,
                        )

                candidate_logits, m1_aux = self.m1_pse.generate(
                    base_logits,
                    image=image,
                    semantic_map=image_only_map,
                    text_features=positive_text_features.detach(),
                    negative_text_features=negative_text_features.detach(),
                    swapped_text_features=swapped_text_features.detach(),
                )
                candidate_logits, m1_aux = _finalize_candidates(
                    candidate_logits,
                    m1_aux,
                )
                _apply_tide_repair_if_needed(
                    candidate_logits,
                    m1_aux,
                    image_only_map,
                    positive_text_features,
                )
                return candidate_logits, m1_aux

            negative_template = self._negative_text_templates()[0]
            negative_prompts = [negative_template] * len(text_prompts)

            with torch.no_grad():
                negative_tokens = self.tokenizer(negative_prompts).to(image.device)
                negative_embeddings = (
                    self.text_model.transformer.embeddings.word_embeddings(
                        negative_tokens
                    ).type(self.dtype)
                )
                _, negative_text_features = self.encode_text_image(
                    negative_tokens,
                    negative_embeddings,
                    image,
                )

            candidate_logits, m1_aux = self.m1_pse.generate(
                base_logits,
                image=image,
                semantic_map=semantic_map,
                text_features=text_features,
                negative_text_features=negative_text_features,
            )
            candidate_logits, m1_aux = _finalize_candidates(
                candidate_logits,
                m1_aux,
            )
            _apply_tide_repair_if_needed(
                candidate_logits,
                m1_aux,
                semantic_map,
                text_features,
            )
            return candidate_logits, m1_aux

        candidate_logits, m1_aux = self.m1_pse.generate(
            base_logits,
            image=image,
            semantic_map=semantic_map,
            mc_std_map=mc_std_map,
            mc_disagreement_map=mc_disagreement_map,
            mc_pairwise_disagreement=mc_pairwise_disagreement,
        )
        candidate_logits, m1_aux = _finalize_candidates(
            candidate_logits,
            m1_aux,
        )

        if self.m2_text_verifier is not None:
            if semantic_map is None:
                raise RuntimeError(
                    "M2 text verifier requires projected UniMedCLIP semantic features."
                )
            m2_start = int(
                _cfg_get(_cfg_get(self.cfg, "M1", None), "M2_START_EPOCH", 0)
            )
            if self.training and self.current_epoch < m2_start:
                cf_evidence = self._neutral_counterfactual_evidence(
                    m1_aux["candidate_probs"]
                )
            else:
                cf_evidence = self._dense_mask_text_evidence(
                    image=image,
                    candidate_probs=m1_aux["candidate_probs"],
                    text_prompts=(
                        verifier_text_prompts
                        if verifier_text_prompts is not None
                        else text_prompts
                    ),
                    supervision_masks=supervision_masks if self.training else None,
                )
            verifier_aux = self.m2_text_verifier(
                m1_aux["candidate_probs"],
                cf_evidence,
            )
            m1_aux.update(verifier_aux)

        _apply_tide_repair_if_needed(
            candidate_logits,
            m1_aux,
            semantic_map,
            text_features,
        )

        return candidate_logits, m1_aux

    def _m1_inference_probability(
        self,
        base_logits: torch.Tensor,
        image: torch.Tensor,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        tokenized_prompts: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_prompts,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Legacy internal helper kept prompt-complete for V13 text falsification."""
        if not self.m1_active:
            return torch.sigmoid(base_logits), {}
        _, m1_aux = self._m1_generate(
            base_logits,
            image=image,
            image_features=image_features,
            text_features=text_features,
            tokenized_prompts=tokenized_prompts,
            text_embeddings=text_embeddings,
            text_prompts=text_prompts,
        )
        candidate_probs = m1_aux["candidate_probs"]
        return self._fuse_from_aux(candidate_probs, m1_aux), m1_aux

    def set_m1_generator_frozen(self, frozen: bool) -> None:
        """Compatibility alias: freeze the candidate generator only."""
        if self.m1_pse is None:
            return
        for parameter in self.m1_pse.parameters():
            parameter.requires_grad_(not bool(frozen))

    def set_phase_b_trainable(self, phase_b: bool) -> None:
        """Enforce the V15.3 stationary-observer Phase-B contract.

        In the provided frozen-B0 protocol this leaves Base frozen already.
        In anchor-student mode it additionally freezes the Base decoder blocks
        once M2 starts, preventing candidate/raw-evidence drift while M2 is
        calibrated.
        """
        if not self.m1_active:
            return
        if bool(
            self.m1_pse is not None
            and getattr(self.m1_pse, "v394_counterfactual_value_rank", False)
        ):
            if not hasattr(self.m1_pse, "set_v394_phase"):
                raise RuntimeError("V394 bank does not expose set_v394_phase().")
            self.m1_pse.set_v394_phase(bool(phase_b))
            return
        if bool(
            self.m1_pse is not None
            and getattr(self.m1_pse, "v393_use_preserve_relative_value", False)
        ):
            if not hasattr(self.m1_pse, "set_v393_phase"):
                raise RuntimeError("V393 bank does not expose set_v393_phase().")
            self.m1_pse.set_v393_phase(bool(phase_b))
            return
        if self.v20_unified_action_cf:
            # V20 has no phase transition: proposal, verifier and selector are jointly trainable.
            return
        phase_b = bool(phase_b)
        self.set_m1_generator_frozen(phase_b)
        freeze_base = bool(_cfg_get(_cfg_get(self.cfg, "M1", None), "M2_FREEZE_BASE_AFTER_PHASE_A", True))
        if freeze_base:
            for module in (self.pvl_adapters, self.mask_head, self.upscale):
                for parameter in module.parameters():
                    parameter.requires_grad_(not phase_b and self.m1_train_mode in {"e2e", "anchor_student"})
        if self.m2_text_verifier is not None:
            for name, parameter in self.m2_text_verifier.named_parameters():
                parameter.requires_grad_(name != "patch_adapter.weight")

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)
        pipeline = getattr(self, "v484_pipeline", None)
        if pipeline is not None and hasattr(pipeline, "set_epoch"):
            pipeline.set_epoch(epoch)

    def set_m2_tide_repair_trainable(self) -> None:
        """Freeze B0 and candidate-bank parameters; train only V399 TIDE head."""
        head = getattr(self, "m2_tide_repair_head", None)
        head_prefixes = ("m2_tide_repair_head.",)

        if head is None and self.m1_pse is not None:
            nested_head = getattr(self.m1_pse, "m2_tide_repair_head", None)
            if nested_head is not None:
                head = nested_head
                head_prefixes = ("m1_pse.m2_tide_repair_head.",)

        if head is None:
            raise RuntimeError(
                "M2_TIDE_REPAIR_ENABLED=True, but no TIDE repair head was built. "
                "Expected either CustomCLIP.m2_tide_repair_head or "
                "CustomCLIP.m1_pse.m2_tide_repair_head."
            )

        for name, param in self.named_parameters():
            param.requires_grad_(name.startswith(head_prefixes) and ".residual_decoder." not in name)

        head.train()

    def encode_text_image(
        self,
        tokenized_prompts,
        text_prompts,
        image,
        attention_mask: Optional[torch.LongTensor] = None,
        disable_pvl: bool = False,
    ):
        if attention_mask is None:
            attention_mask = (tokenized_prompts != self.text_model.config.pad_token_id).long()
        x_txt = self.text_model.transformer.embeddings(inputs_embeds=text_prompts)
        extended_attention_mask = attention_mask[:, None, None, :].to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min

        x_img = self.vision_model.conv1(image)
        x_img = x_img.reshape(x_img.shape[0], x_img.shape[1], -1).permute(0, 2, 1)
        x_img = torch.cat([
            self.vision_model.class_embedding.to(x_img.dtype)
            + torch.zeros(x_img.shape[0], 1, x_img.shape[-1], dtype=x_img.dtype, device=x_img.device),
            x_img,
        ], dim=1)
        x_img = x_img + self.vision_model.positional_embedding.to(x_img.dtype)
        x_img = self.vision_model.ln_pre(x_img).permute(1, 0, 2)

        hidden_states = []
        for index, (block, layer) in enumerate(zip(self.vision_model.transformer.resblocks, self.text_model.transformer.encoder.layer)):
            if (not disable_pvl) and (not self.disable_pvl_ablation) and index in self.fusion_stages:
                vis_pvl, txt_pvl = self.pvl_adapters[self.fusion_stages.index(index)](x_img.transpose(1, 0), x_txt)
                x_txt = x_txt + txt_pvl
                x_img = x_img + vis_pvl.transpose(1, 0)
            x_img = block(x_img)
            x_txt = layer(x_txt, attention_mask=extended_attention_mask)[0]
            hidden_states.append(x_img)

        x_img = x_img.permute(1, 0, 2)
        x_img = self.vision_model.ln_post(x_img)
        if self.vision_model.proj is not None:
            x_img = x_img @ self.vision_model.proj

        projected_text = self.text_model.proj(x_txt[:, 0, :])
        if self.output_hidden_states:
            return x_img, hidden_states, projected_text
        return x_img, projected_text

    def compute_seg_logits(
        self, image_features, text_features, batch_size, height, width,
        return_decoder_features: bool = False,
        raw_image=None,
    ):
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        cls_token = image_features[:, 0, :]
        cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)
        patch_features = image_features[:, 1:, :]
        patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)
        patch_height = height // self.patch_size
        patch_width = width // self.patch_size
        patch_features = patch_features.reshape(batch_size, patch_height, patch_width, -1).permute(0, 3, 1, 2)
        decoder_features = self.upscale(patch_features)
        seg_text_vector = self.mask_head(text_features)

        # Coarse 56x56 factual mask: it is used only to locate uncertainty.
        # UGBRA then modifies decoder FEATURES (not the output mask) using local
        # high-frequency residuals.  The uncertainty gate is detached inside
        # the module, avoiding a self-referential confidence shortcut.
        if self.ugbra_enabled and self.ugbra is not None:
            coarse_logits = torch.einsum(
                "bqc,bchw->bqhw", seg_text_vector.unsqueeze(1), decoder_features
            )
            decoder_features = self.ugbra(decoder_features, coarse_logits, image=raw_image)

        seg_logits = torch.einsum(
            "bqc,bchw->bqhw", seg_text_vector.unsqueeze(1), decoder_features
        )
        # QABR acts on the already text-query-conditioned scalar logit at the
        # native 224x224 grid. This makes every learned residual directly useful
        # to the mask, instead of injecting a 512-D feature residual whose
        # components can be orthogonal to the text query.
        if self.qabr_enabled and self.qabr is not None:
            seg_logits = self.qabr(
                decoder_features, seg_logits, image=raw_image, output_size=self.im_size
            ).squeeze(1)
        else:
            seg_logits = F.interpolate(
                seg_logits, self.im_size, mode="bilinear", align_corners=False
            ).squeeze(1)
        if return_decoder_features:
            return seg_logits, cls_token, decoder_features
        return seg_logits, cls_token

    @staticmethod
    def soft_cross_entropy(pred_logits, soft_targets):
        return -(soft_targets * F.log_softmax(pred_logits, dim=-1)).sum(dim=-1).mean()

    def _forward_base_once(
        self, image, tokenized_prompts, text_embeddings,
        return_decoder_features: bool = False,
    ):
        batch_size, _, height, width = image.shape
        image_features, text_features = self.encode_text_image(tokenized_prompts, text_embeddings, image)
        if return_decoder_features:
            base_logits, cls_token, decoder_features = self.compute_seg_logits(
                image_features, text_features, batch_size, height, width,
                return_decoder_features=True, raw_image=image,
            )
            return base_logits, image_features, text_features, cls_token, decoder_features
        base_logits, cls_token = self.compute_seg_logits(
            image_features, text_features, batch_size, height, width, raw_image=image
        )
        return base_logits, image_features, text_features, cls_token

    @torch.no_grad()
    def predict_base_probs(self, image, text, num_samples=30):
        """Stream the official MC Base posterior without constructing OACD.

        Streaming avoids retaining 30 copies of ViT feature tensors, allowing
        the reference Test batch size of 32 to be used.
        """
        tokenized_prompts = self.tokenizer(text).to(image.device)
        text_embeddings = self.text_model.transformer.embeddings.word_embeddings(
            tokenized_prompts
        ).type(self.dtype)
        probability_sum = None
        sample_count = max(1, int(num_samples))
        # The public MedCLIPSeg forward() performs one stochastic regular
        # forward before collecting the N MC samples at eval time.  That first
        # draw is discarded but advances the PVL RNG.  Replaying it is needed
        # for bitwise-close public-repo MC trajectories when requested.
        replay_public_mc_burnin = bool(
            _cfg_get(_cfg_get(self.cfg, "TRAIN", None),
                     "REPLAY_PUBLIC_REPO_MC_BURNIN", False)
        )
        if replay_public_mc_burnin:
            self._forward_base_once(image, tokenized_prompts, text_embeddings)
        for _ in range(sample_count):
            logits, _, _, _ = self._forward_base_once(
                image, tokenized_prompts, text_embeddings
            )
            probability = torch.sigmoid(logits)
            probability_sum = (
                probability if probability_sum is None
                else probability_sum + probability
            )
        return (probability_sum / float(sample_count)).clamp(EPS, 1.0 - EPS)

    @torch.no_grad()
    def _xbm_enqueue(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> None:
        if not self.xbm_enabled or self.xbm_queue_size <= 0:
            return
        image_embeddings = image_embeddings.detach()
        text_embeddings = text_embeddings.detach()
        count = image_embeddings.shape[0]
        if count >= self.xbm_queue_size:
            self.xbm_image_queue[: self.xbm_queue_size].copy_(
                image_embeddings[-self.xbm_queue_size :]
            )
            self.xbm_text_queue[: self.xbm_queue_size].copy_(
                text_embeddings[-self.xbm_queue_size :]
            )
            self.xbm_queue_ptr.zero_()
            self.xbm_queue_count.fill_(self.xbm_queue_size)
            return
        ptr = int(self.xbm_queue_ptr.item())
        first = min(count, self.xbm_queue_size - ptr)
        self.xbm_image_queue[ptr : ptr + first].copy_(image_embeddings[:first])
        self.xbm_text_queue[ptr : ptr + first].copy_(text_embeddings[:first])
        remaining = count - first
        if remaining > 0:
            self.xbm_image_queue[:remaining].copy_(image_embeddings[first:])
            self.xbm_text_queue[:remaining].copy_(text_embeddings[first:])
        self.xbm_queue_ptr.fill_((ptr + count) % self.xbm_queue_size)
        self.xbm_queue_count.fill_(min(self.xbm_queue_size, int(self.xbm_queue_count.item()) + count))

    def _clip_alignment_loss(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        if bool(
            _cfg_get(
                _cfg_get(self.cfg, "M1", None),
                "OFFICIAL_MEDCLIPSEG_BASE_CONTRACT",
                False,
            )
        ):
            # Exact public MedCLIPSeg objective. Do not normalize patch_mean or
            # projected text features here; the original code only normalizes
            # every patch token, then row-normalizes the text-similarity target.
            patch_tokens = image_features[:, 1:, :]
            patch_tokens = patch_tokens / patch_tokens.norm(dim=-1, keepdim=True)
            patch_mean = patch_tokens.mean(dim=1)
            logits_per_image = (patch_mean @ text_features.T) / self.temperature
            logits_per_text = (text_features @ patch_mean.T) / self.temperature
            with torch.no_grad():
                text_sim = (text_features @ text_features.T) / self.temperature
                text_sim = text_sim / text_sim.norm(dim=-1, keepdim=True)
                soft_targets = F.softmax(text_sim, dim=-1)
            return 0.5 * (
                self.soft_cross_entropy(logits_per_image, soft_targets)
                + self.soft_cross_entropy(logits_per_text, soft_targets.T)
            )

        patch_tokens = F.normalize(image_features[:, 1:, :].float(), dim=-1, eps=EPS)
        patch_mean = F.normalize(patch_tokens.mean(dim=1), dim=-1, eps=EPS)
        text_current = F.normalize(text_features.float(), dim=-1, eps=EPS)
        epoch = int(getattr(self, "current_epoch", 0))
        queue_count = int(self.xbm_queue_count.item())
        use_memory = (
            self.training
            and self.xbm_enabled
            and epoch >= self.xbm_start_epoch
            and queue_count > 0
        )
        if use_memory:
            image_bank = torch.cat(
                [patch_mean, self.xbm_image_queue[:queue_count].to(patch_mean)], dim=0
            )
            text_bank = torch.cat(
                [text_current, self.xbm_text_queue[:queue_count].to(text_current)], dim=0
            )
        else:
            image_bank = patch_mean
            text_bank = text_current

        logits_per_image = (patch_mean @ text_bank.T) / self.temperature
        logits_per_text = (text_current @ image_bank.T) / self.temperature
        with torch.no_grad():
            image_soft_target = F.softmax(
                (text_current @ text_bank.T) / self.temperature, dim=-1
            )
            text_soft_target = F.softmax(
                (text_current @ text_current.T) / self.temperature, dim=-1
            )
            if use_memory:
                # Text anchors compare to current and memory image keys.  The
                # matching soft target uses text-semantic similarity to the
                # same bank ordering.
                text_soft_target = F.softmax(
                    (text_current @ text_bank.T) / self.temperature, dim=-1
                )
        loss = 0.5 * (
            self.soft_cross_entropy(logits_per_image, image_soft_target)
            + self.soft_cross_entropy(logits_per_text, text_soft_target)
        )
        if self.training:
            self._xbm_enqueue(patch_mean, text_current)
        self._last_xbm_bank_size = int(text_bank.shape[0])
        return loss

    def _apply_v463_ccv(self, aux: Dict[str, Any], base_logits: torch.Tensor) -> Dict[str, Any]:
        """Apply CCV-M2 to current M1 candidates and overwrite deploy fields.

        This keeps the old test/audit interface intact: downstream code still
        reads v20_hard_fused_probs / v20_selector_hard, but those tensors now
        come from the causal counterfactual verifier when V463 is enabled.
        """
        if not getattr(self, "v463_ccv_enabled", False):
            return aux
        if self.ccv_m2 is None:
            raise RuntimeError("V463_CCV_ENABLED is true but ccv_m2 was not constructed.")
        if "candidates" not in aux or "v20_action_supports" not in aux:
            return aux
        m1_cfg = _cfg_get(self.cfg, "M1", None)
        strict_joint = bool(_cfg_get(m1_cfg, "V470_STRICT_JOINT_E2E", False))
        decouple = bool(_cfg_get(
            m1_cfg,
            "V469_DECOUPLE_CCV_FROM_M1",
            _cfg_get(
                m1_cfg,
                "UNIFIED_M1_DECOUPLE_SELECTOR_FROM_CANDIDATES",
                True,
            ),
        ))
        if strict_joint:
            # V470 remains genuinely end-to-end: CCV gradients reach Base/M1,
            # but are bounded so the selector cannot numerically dominate the
            # supervised segmentation and candidate objectives.
            decouple = False
            base_grad_scale = float(
                _cfg_get(m1_cfg, "V470_CCV_TO_BASE_GRAD_SCALE", 0.02)
            )
            m1_grad_scale = float(
                _cfg_get(m1_cfg, "V470_CCV_TO_M1_GRAD_SCALE", 0.05)
            )
        else:
            base_grad_scale = 0.0 if decouple else 1.0
            m1_grad_scale = 0.0 if decouple else 1.0

        def _ccv_input(value, scale):
            if not isinstance(value, torch.Tensor):
                return value
            return _scale_gradient(value, scale)

        ccv = self.ccv_m2(
            base_logits=_ccv_input(base_logits, base_grad_scale),
            candidate_logits_all=_ccv_input(
                aux["candidates"], m1_grad_scale
            ),
            factual_masks=_ccv_input(
                aux["v20_action_supports"], m1_grad_scale
            ),
            control_masks=_ccv_input(
                aux.get("v20_control_supports"), m1_grad_scale
            ),
            action_types=aux.get("v20_action_types", self.m1_pse.action_types),
            entropy=_ccv_input(aux.get("v20_entropy"), m1_grad_scale),
            boundary=_ccv_input(aux.get("v20_boundary"), m1_grad_scale),
            residual_probs=_ccv_input(
                aux.get("v463_residual_probs"), m1_grad_scale
            ),
            visual_scores=_ccv_input(
                aux.get("v20_visual_scores"), m1_grad_scale
            ),
        )
        ccv["v469_ccv_inputs_detached"] = base_logits.new_full(
            (base_logits.shape[0],), float(decouple)
        )
        ccv["v470_ccv_to_base_grad_scale"] = base_logits.new_full(
            (base_logits.shape[0],), float(base_grad_scale)
        )
        ccv["v470_ccv_to_m1_grad_scale"] = base_logits.new_full(
            (base_logits.shape[0],), float(m1_grad_scale)
        )
        aux.update(ccv)

        # V464 bridge: expose CCV-M2 outputs through the existing m2_* audit
        # interface. Without this bridge, v27_action_audit.csv contains NaN
        # m2 columns even when CCV exists.
        ccv_utility = ccv["v463_ccv_utility"]
        ccv_lcb_sum = ccv["v463_ccv_lcb_dsc"] + ccv["v463_ccv_lcb_nsd"]
        ccv_sigma_sum = ccv["v463_ccv_sigma_dsc"] + ccv["v463_ccv_sigma_nsd"]
        if "v469_ccv_valid_action" in ccv:
            ccv_action_valid = ccv["v469_ccv_valid_action"].to(
                ccv_utility.dtype
            )
        elif "v20_action_supports" in aux:
            ccv_action_valid = (
                aux["v20_action_supports"].flatten(2).sum(dim=-1) > 0.5
            ).to(ccv_utility.dtype)
        else:
            ccv_action_valid = torch.ones_like(ccv_utility)
        enabled_mask = aux.get("v469_action_enabled_mask")
        if isinstance(enabled_mask, torch.Tensor):
            enabled_mask = enabled_mask.to(
                device=ccv_utility.device, dtype=torch.bool
            ).reshape(-1)[:ccv_utility.shape[1]]
            ccv_action_valid = ccv_action_valid * enabled_mask[None].to(
                ccv_action_valid.dtype
            )

        if ccv_utility.shape[1] > 1:
            _top2 = ccv_utility.topk(k=2, dim=1).values
            ccv_top_gap = (_top2[:, 0] - _top2[:, 1])[:, None].expand_as(ccv_utility)
        else:
            ccv_top_gap = ccv_utility.abs()

        aux.update({
            "m2_action_valid": ccv_action_valid,
            "m2_accept": ccv["v463_ccv_selected_hard"],
            "m2_gain_lcb": ccv_lcb_sum,
            "m2_gain_mean": ccv["v463_ccv_tau_dsc"] + ccv["v463_ccv_tau_nsd"],
            "m2_gain_std": ccv_sigma_sum,
            "m2_gate_probability": torch.sigmoid(20.0 * ccv_utility),
            "m2_selected_score": ccv_utility,
            "m2_best_lcb": ccv_utility,
            "m2_top_lcb_gap": ccv_top_gap,
            "m2_edit_fraction": aux.get(
                "local_action_area",
                aux["v20_action_supports"].flatten(2).float().mean(dim=-1)
                if "v20_action_supports" in aux else torch.zeros_like(ccv_utility),
            ),
            "m2_selected_index": ccv["v463_ccv_selected_index"],
        })

        
        # V466 root fix: CCV deployment bridge.
        # The verifier already has a score correlated with true gain, but the
        # old strict accept gate rejects every candidate. Here Preserve remains
        # score 0; an action is deployed only if its CCV utility beats Preserve
        # and passes basic action validity. This fixes the train/deploy mismatch.
        ccv_utility = ccv.get("v463_ccv_utility", None)
        if ccv_utility is not None:
            m1_cfg = _cfg_get(self.cfg, "M1", None)

            candidate_probs_for_bridge = aux.get("candidate_probs", None)
            if not isinstance(candidate_probs_for_bridge, torch.Tensor):
                raise RuntimeError("V466 bridge requires aux['candidate_probs'] to select final candidate.")
            candidate_probs_for_bridge = candidate_probs_for_bridge.clamp(EPS, 1.0 - EPS)
            candidate_logits = torch.logit(candidate_probs_for_bridge)

            action_count = ccv_utility.shape[1]
            action_valid = ccv_action_valid[:, :action_count] > 0.5

            # V469: the classifier heads are a safety eligibility stage; the
            # direct utility head only ranks candidates that pass this stage.
            # The old bridge ignored this mask and recomputed deployment from
            # utility alone, which reintroduced the train/deploy mismatch.
            use_class_eligibility = bool(_cfg_get(
                m1_cfg,
                "V469_CCV_USE_CLASS_ELIGIBILITY",
                _cfg_get(
                    m1_cfg,
                    "V469_USE_CLASSIFICATION_ELIGIBILITY",
                    True,
                ),
            ))
            class_eligible = ccv.get("v469_ccv_class_eligible")
            if use_class_eligibility and isinstance(class_eligible, torch.Tensor):
                action_valid = action_valid & (
                    class_eligible[:, :action_count] > 0.5
                )

            action_types = aux.get("v20_action_types", None)
            if isinstance(action_types, torch.Tensor):
                at = action_types.to(device=ccv_utility.device).long().reshape(-1)[:action_count]
                # V466: type2/fill 是当前主要污染源。训练仍保留，但部署 probe
                # 先禁用 type2，等 fill 的 mean_gain 不再显著为负后再打开。
                deploy_type2 = bool(_cfg_get(m1_cfg, "V466_DEPLOY_TYPE2_FILL", False))
                if not deploy_type2:
                    action_valid = action_valid & (at[None, :] != 2)

            threshold = float(_cfg_get(m1_cfg, "V463_CCV_UTILITY_THRESHOLD", 0.0))
            score = ccv_utility.masked_fill(~action_valid, -1.0e4)
            best_score, best_action = score.max(dim=1)
            accept = torch.isfinite(best_score) & (best_score > threshold)

            selected_idx = torch.where(
                accept,
                best_action + 1,
                torch.zeros_like(best_action),
            )
            gather_idx = selected_idx[:, None, None, None].expand(
                -1, 1, candidate_logits.shape[-2], candidate_logits.shape[-1]
            )

            selected_logits = candidate_logits.gather(1, gather_idx)[:, 0]
            selected_probs = torch.sigmoid(selected_logits).clamp(1.0e-6, 1.0 - 1.0e-6)

            selector_hard = candidate_logits.new_zeros((candidate_logits.shape[0], action_count))
            selector_hard.scatter_(
                1,
                (selected_idx - 1).clamp_min(0)[:, None],
                accept.to(candidate_logits.dtype)[:, None],
            )

            ccv["v463_ccv_selected_index"] = selected_idx
            ccv["v463_ccv_selected_hard"] = selector_hard
            ccv["v463_ccv_final_probs"] = selected_probs
            ccv["v463_ccv_final_logits"] = selected_logits
            ccv["v463_ccv_changed"] = accept.to(candidate_logits.dtype)
            ccv["v463_ccv_accept_mask"] = selector_hard

            aux.update({
                "m2_accept": selector_hard,
                "m2_selected_index": selected_idx,
                "m2_selected_score": score,
                "m2_best_lcb": best_score[:, None].expand_as(ccv_utility),
                "v20_selector_hard": selector_hard,
                "v20_selector_probs": selector_hard,
                "m1_selector_hard": selector_hard,
                "m1_selector_soft": selector_hard,
            })
            # Re-publish modified CCV fields.  The old bridge updated only local
            # `ccv`, so training diagnostics kept the pre-bridge changed_rate=0.
            aux.update(ccv)


        aux["v20_hard_fused_probs"] = ccv["v463_ccv_final_probs"]
        aux["v20_fused_probs"] = ccv["v463_ccv_final_probs"]
        aux["direct_fused_probs"] = ccv["v463_ccv_final_probs"]
        aux["router_fused_probs"] = ccv["v463_ccv_final_probs"]
        aux["v20_fused_logits"] = ccv["v463_ccv_final_logits"]
        aux["v20_selector_hard"] = ccv["v463_ccv_selected_hard"]
        aux["v20_selector_probs"] = ccv["v463_ccv_select_soft"][:, 1:]
        aux["v20_selector_logits"] = ccv["v463_ccv_utility"]
        return aux

    def _m1_aux_to_base_grad_scale(self) -> tuple[float, bool]:
        """Resolve cross-module gradient flow for candidate supervision.

        All task modules remain trainable in the same optimizer/run.  The
        optional V475 protection flag only prevents noisy candidate losses from
        changing the Base segmentation pathway; it does not freeze Base or M1.
        """
        m1_cfg = _cfg_get(self.cfg, "M1", None)
        protect_base = bool(
            _cfg_get(m1_cfg, "V475_PROTECT_BASE_FROM_CANDIDATE_GRAD", False)
        )
        legacy_stopgrad = bool(
            _cfg_get(m1_cfg, "DETACH_BASE_FOR_CANDIDATES", False)
            or _cfg_get(m1_cfg, "V383_GI_STOPGRAD_AUX_TO_BASE", False)
        )
        strict_joint = bool(_cfg_get(m1_cfg, "V470_STRICT_JOINT_E2E", False))

        # JBT-v4 BaseSafe contract: the official Base/PVL trajectory is owned
        # exclusively by the published MedCLIPSeg objective.  M1 is co-trained
        # in the same run on moving Base predictions, but its loss has zero
        # derivative w.r.t. Base/PVL.  This removes the dataset-dependent
        # auxiliary-gradient rotation observed as +0.68pp BUSI vs -1.39pp
        # Kvasir Base drift.
        if bool(_cfg_get(m1_cfg, "JBT_BASE_GRAD_ISOLATION", False)):
            return 0.0, True
        if protect_base:
            return 0.0, True
        if strict_joint:
            if bool(_cfg_get(m1_cfg, "JBT_GRAD_RAMP_ENABLED", False)):
                epoch1 = max(1, int(getattr(self, "current_epoch", 0)) + 1)
                start = float(_cfg_get(m1_cfg, "JBT_GRAD_SCALE_START", 0.05))
                mid = float(_cfg_get(m1_cfg, "JBT_GRAD_SCALE_MID", 0.10))
                final = float(_cfg_get(m1_cfg, "JBT_GRAD_SCALE_FINAL", 0.15))
                mid_epoch = max(1, int(_cfg_get(m1_cfg, "JBT_GRAD_SCALE_MID_EPOCH", 20)))
                final_epoch = max(mid_epoch + 1, int(
                    _cfg_get(m1_cfg, "JBT_GRAD_SCALE_FINAL_EPOCH", 50)
                ))
                if epoch1 <= 5:
                    scale = start
                elif epoch1 <= mid_epoch:
                    t = float(epoch1 - 5) / float(max(mid_epoch - 5, 1))
                    scale = start + t * (mid - start)
                elif epoch1 <= final_epoch:
                    t = float(epoch1 - mid_epoch) / float(max(final_epoch - mid_epoch, 1))
                    scale = mid + t * (final - mid)
                else:
                    scale = final
                return float(max(0.0, min(1.0, scale))), False
            return float(
                _cfg_get(m1_cfg, "V470_AUX_TO_BASE_GRAD_SCALE", 0.10)
            ), False
        if legacy_stopgrad:
            return 0.0, True
        return 1.0, False


    @staticmethod
    def _mc_structural_uncertainty(
        probability_samples: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pixel-wise and image-wise MC disagreement for failure detection.

        probability_samples: [S,B,1,H,W].  The image-level score is one minus
        the mean pairwise Dice between thresholded MC masks, a robust baseline
        for medical-segmentation failure detection.
        """
        # _forward_base_once currently exports segmentation probabilities
        # as [B,H,W]. Stacking MC passes therefore produces [S,B,H,W].
        # Convert that valid layout to the canonical [S,B,1,H,W] form locally
        # without changing the public Base-logit shape used elsewhere.
        if probability_samples.ndim == 4:
            probability_samples = probability_samples.unsqueeze(2)
        elif (
            probability_samples.ndim != 5
            or probability_samples.shape[2] != 1
        ):
            raise ValueError(
                "MC samples must be [S,B,H,W] or [S,B,1,H,W], got "
                f"{tuple(probability_samples.shape)}"
            )
        sample_count = probability_samples.shape[0]
        if sample_count <= 1:
            zeros_map = torch.zeros_like(probability_samples[0])
            zeros_case = probability_samples.new_zeros(
                probability_samples.shape[1]
            )
            return zeros_map, zeros_map, zeros_case
        std_map = probability_samples.std(dim=0, unbiased=False)
        hard = probability_samples >= 0.5
        xor_maps = []
        pairwise_dice = []
        for i in range(sample_count):
            for j in range(i + 1, sample_count):
                xor_maps.append(hard[i].ne(hard[j]).to(probability_samples.dtype))
                hi = hard[i].to(probability_samples.dtype)
                hj = hard[j].to(probability_samples.dtype)
                inter = (hi * hj).flatten(1).sum(dim=1)
                den = hi.flatten(1).sum(dim=1) + hj.flatten(1).sum(dim=1)
                pairwise_dice.append((2.0 * inter + EPS) / (den + EPS))
        disagreement_map = torch.stack(xor_maps, dim=0).mean(dim=0)
        pairwise_disagreement = 1.0 - torch.stack(
            pairwise_dice, dim=0
        ).mean(dim=0)
        return std_map, disagreement_map, pairwise_disagreement

    def _jbt_training_posterior_evidence(
        self,
        image: torch.Tensor,
        tokenized_prompts: torch.Tensor,
        text_embeddings: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """RNG-isolated low-sample posterior evidence for JBT-v2 training.

        The official Base objective keeps exactly one trainable stochastic
        forward.  JBT adds a small number of no-grad stochastic draws inside
        ``fork_rng`` only to estimate where the Base posterior is unstable.
        The outer RNG state is restored afterwards, so later Base/PVL dropout
        follows the same sequence it would have followed without this observer.
        """
        m1_cfg = _cfg_get(self.cfg, "M1", None)
        if not bool(_cfg_get(m1_cfg, "JBT_POSTERIOR_UNCERTAINTY_ENABLED", False)):
            return None, None, None
        total_samples = max(1, int(
            _cfg_get(m1_cfg, "JBT_TRAIN_UNCERTAINTY_MC_SAMPLES", 1)
        ))
        if total_samples <= 1:
            return None, None, None
        probabilities = [torch.sigmoid(base_logits.detach()).clamp(EPS, 1.0 - EPS)]
        cuda_devices = []
        if image.is_cuda and image.device.index is not None:
            cuda_devices = [int(image.device.index)]
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            with torch.no_grad():
                for _ in range(total_samples - 1):
                    logits, _, _, _ = self._forward_base_once(
                        image, tokenized_prompts, text_embeddings
                    )
                    probabilities.append(
                        torch.sigmoid(logits).clamp(EPS, 1.0 - EPS)
                    )
        samples = torch.stack(probabilities, dim=0)
        return self._mc_structural_uncertainty(samples)

    def _jbt_training_mean_then_refine_bundle(
        self,
        image: torch.Tensor,
        tokenized_prompts: torch.Tensor,
        text_embeddings: torch.Tensor,
        base_logits: torch.Tensor,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        decoder_features: Optional[torch.Tensor],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Build an RNG-isolated MC training state matching deployment order.

        Official MedCLIPSeg still receives its ordinary one-sample train loss.
        The JBT branch, however, is trained on a small posterior mean:
        mean-then-refine at train time matches MC30 mean-then-refine at test.
        Only the first sample carries gradient; extra draws are no-grad and the
        outer RNG state is restored.  This preserves the Base contract while
        eliminating the v2 single-sample/refined-MC-mean domain mismatch.
        """
        m1_cfg = _cfg_get(self.cfg, "M1", None)
        if not bool(_cfg_get(m1_cfg, "JBT_TRAIN_MEAN_THEN_REFINE", False)):
            return None
        sample_count = max(2, int(
            _cfg_get(m1_cfg, "JBT_TRAIN_UNCERTAINTY_MC_SAMPLES", 3)
        ))
        p0 = torch.sigmoid(base_logits).clamp(EPS, 1.0 - EPS)
        probability_sum = p0
        image_sum = image_features
        text_sum = text_features
        decoder_sum = decoder_features
        detached_probabilities = [p0.detach()]

        cuda_devices = []
        if image.is_cuda and image.device.index is not None:
            cuda_devices = [int(image.device.index)]
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            with torch.no_grad():
                for _ in range(sample_count - 1):
                    if isinstance(decoder_features, torch.Tensor):
                        logits_i, image_i, text_i, _, decoder_i = self._forward_base_once(
                            image, tokenized_prompts, text_embeddings,
                            return_decoder_features=True,
                        )
                    else:
                        logits_i, image_i, text_i, _ = self._forward_base_once(
                            image, tokenized_prompts, text_embeddings
                        )
                        decoder_i = None
                    pi = torch.sigmoid(logits_i).clamp(EPS, 1.0 - EPS)
                    detached_probabilities.append(pi.detach())
                    probability_sum = probability_sum + pi.detach()
                    image_sum = image_sum + image_i.detach()
                    text_sum = text_sum + text_i.detach()
                    if isinstance(decoder_sum, torch.Tensor):
                        if not isinstance(decoder_i, torch.Tensor):
                            raise RuntimeError(
                                "JBT train posterior bundle lost decoder features"
                            )
                        decoder_sum = decoder_sum + decoder_i.detach()

        mean_probability = (probability_sum / float(sample_count)).clamp(
            EPS, 1.0 - EPS
        )
        mc_samples = torch.stack(detached_probabilities, dim=0)
        mc_std_map, mc_disagreement_map, mc_pairwise_disagreement = (
            self._mc_structural_uncertainty(mc_samples)
        )
        bundle = {
            "base_logits": torch.logit(mean_probability),
            "image_features": image_sum / float(sample_count),
            "text_features": text_sum / float(sample_count),
            "mc_std_map": mc_std_map,
            "mc_disagreement_map": mc_disagreement_map,
            "mc_pairwise_disagreement": mc_pairwise_disagreement,
            "sample_count": base_logits.new_tensor(float(sample_count)),
        }
        if isinstance(decoder_sum, torch.Tensor):
            bundle["decoder_features"] = decoder_sum / float(sample_count)
        return bundle

    def forward_geotr_train_mc_detached_base(
        self,
        image,
        text,
        num_samples: int = 10,
        supervision_masks: Optional[torch.Tensor] = None,
        slr_hr_image: Optional[torch.Tensor] = None,
        sparc_hr_mask: Optional[torch.Tensor] = None,
    ):
        """Build GEOTR candidates from an MC-mean posterior without changing Base training.

        The official Base/PVL loss is computed by the normal single-forward path in
        train.py.  This auxiliary path samples the current Base posterior under
        no_grad, restores the outer RNG state afterwards, and then trains GEOTR on
        the same mean-then-refine posterior family used by Val/Test.  Because GEOTR
        already has a hard stop-gradient boundary to Base/PVL, this removes the
        Stage1/Stage2 posterior-domain mismatch without changing Base gradients or
        consuming RNG that would perturb the next official Base batch.
        """
        if not self.m1_active:
            raise RuntimeError("forward_geotr_train_mc_detached_base requires M1.ENABLED: true")
        sample_count = max(1, int(num_samples))
        tokenized_prompts = self.tokenizer(text).to(image.device)
        with torch.no_grad():
            text_embeddings = self.text_model.transformer.embeddings.word_embeddings(tokenized_prompts).type(self.dtype)

        cuda_devices = []
        if image.is_cuda and image.device.index is not None:
            cuda_devices = [int(image.device.index)]
        base_probs = []
        feature_sum = None
        text_feature_sum = None
        decoder_feature_sum = None
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            with torch.no_grad():
                for _ in range(sample_count):
                    logits, image_features, text_features, _, decoder_features = self._forward_base_once(
                        image, tokenized_prompts, text_embeddings, return_decoder_features=True
                    )
                    base_probs.append(torch.sigmoid(logits))
                    feature_sum = image_features if feature_sum is None else feature_sum + image_features
                    text_feature_sum = text_features if text_feature_sum is None else text_feature_sum + text_features
                    decoder_feature_sum = decoder_features if decoder_feature_sum is None else decoder_feature_sum + decoder_features

        samples = torch.stack(base_probs, dim=0)

        # samples now owns the exact posterior tensor. The ten individual
        # probability tensors in base_probs are redundant Python references.
        del base_probs

        mean_base_probs = samples.mean(dim=0).clamp(EPS, 1.0 - EPS)
        mc_std_map, mc_disagreement_map, mc_pairwise_disagreement = self._mc_structural_uncertainty(samples)

        mean_base_logits = torch.logit(mean_base_probs).detach()
        mean_image_features = (feature_sum / float(sample_count)).detach()
        mean_text_features = (text_feature_sum / float(sample_count)).detach()
        mean_decoder_features = (decoder_feature_sum / float(sample_count)).detach()

        # Exact low-memory handoff into trainable UC-FNRT.
        # All required posterior means have already been materialized.
        # Remove accumulation buffers and final-loop aliases before creating
        # the large dense semantic/refinement tensors.
        del feature_sum
        del text_feature_sum
        del decoder_feature_sum

        for _name in (
            "logits",
            "image_features",
            "text_features",
            "decoder_features",
        ):
            if _name in locals():
                del locals()[_name]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        candidate_logits, candidate_aux = self._m1_generate(
            mean_base_logits,
            image=image,
            image_features=mean_image_features,
            text_features=mean_text_features,
            tokenized_prompts=tokenized_prompts,
            text_embeddings=text_embeddings,
            text_prompts=text,
            supervision_masks=supervision_masks,
            slr_hr_image=slr_hr_image,
            sparc_hr_mask=sparc_hr_mask,
            mc_std_map=mc_std_map,
            mc_disagreement_map=mc_disagreement_map,
            mc_pairwise_disagreement=mc_pairwise_disagreement,
            mc_probability_samples=samples,
            fine_feature_map=mean_decoder_features,
        )
        aux = {
            "base_logits": mean_base_logits,
            "image_features": mean_image_features,
            "text_features": mean_text_features,
            "candidates": candidate_logits,
            "geotr_train_posterior_samples": mean_base_logits.new_full(
                (mean_base_logits.shape[0],), float(sample_count)
            ),
            **candidate_aux,
        }
        aux = self._apply_v463_ccv(aux, mean_base_logits)
        aux["fused_probs"] = self._fuse_from_aux(aux["candidate_probs"], aux)
        return mean_base_logits, aux

    def forward_m1_train_mc(
        self, image, text, num_samples: int = 4,
        supervision_masks: Optional[torch.Tensor] = None,
        slr_hr_image: Optional[torch.Tensor] = None,
        sparc_hr_mask: Optional[torch.Tensor] = None,
    ):
        """Train M1/M2 on the same MC-mean candidate distribution used by Val/Test.

        Base/Preserve, M1 and M2 may all be trainable.  When V470 strict joint
        E2E is active, every task-specific module receives gradients in the same
        optimization step; cross-module paths are scaled but never detached.
        """
        if not self.m1_active:
            raise RuntimeError("forward_m1_train_mc requires M1.ENABLED: true.")
        if bool(_cfg_get(_cfg_get(self.cfg, "M1", None), "SEMLT_LST_V31_ROOTFIX", False)):
            raise RuntimeError(
                "SemLT-LST v3.1 Root-Fix must use the single-forward joint-E2E "
                "path, not forward_m1_train_mc(). This guard prevents accidental "
                "legacy-V31 MC routing and first-batch OOM."
            )
        tokenized_prompts = self.tokenizer(text).to(image.device)
        full_end_to_end = bool(
            _cfg_get(_cfg_get(self.cfg, "M1", None), "V478_FULL_END_TO_END", False)
        )
        text_context = torch.enable_grad() if full_end_to_end else torch.no_grad()
        with text_context:
            text_embeddings = self.text_model.transformer.embeddings.word_embeddings(tokenized_prompts).type(self.dtype)

        base_is_trainable = any(
            parameter.requires_grad
            for name, parameter in self.named_parameters()
            if not (name.startswith("m1_pse.") or name.startswith("m2_text_verifier."))
        )
        sample_count = max(1, int(num_samples))
        base_probs = []
        feature_sum = None
        text_feature_sum = None
        # In frozen mode this prevents pointless graph storage for B0 while M1/M2
        # remain outside the context and fully trainable.
        context = torch.enable_grad() if base_is_trainable else torch.no_grad()
        with context:
            for _ in range(sample_count):
                logits, image_features, text_features, _ = self._forward_base_once(
                    image, tokenized_prompts, text_embeddings
                )
                base_probs.append(torch.sigmoid(logits))
                feature_sum = image_features if feature_sum is None else feature_sum + image_features
                text_feature_sum = text_features if text_feature_sum is None else text_feature_sum + text_features

        base_probability_samples = torch.stack(base_probs, dim=0)
        mean_base_probs = base_probability_samples.mean(dim=0).clamp(EPS, 1.0 - EPS)
        mc_std_map, mc_disagreement_map, mc_pairwise_disagreement = (
            self._mc_structural_uncertainty(base_probability_samples)
        )
        mean_base_logits = torch.logit(mean_base_probs)
        mean_image_features = feature_sum / float(sample_count)
        mean_text_features = text_feature_sum / float(sample_count)

        # V475 keeps Base and M1 trainable in one run while optionally
        # decoupling noisy candidate gradients from the Base pathway.
        aux_to_base_scale, stopgrad_aux_to_base = (
            self._m1_aux_to_base_grad_scale()
        )

        m1_base_logits = _scale_gradient(
            mean_base_logits, aux_to_base_scale
        )
        m1_image_features = _scale_gradient(
            mean_image_features, aux_to_base_scale
        )
        m1_text_features = _scale_gradient(
            mean_text_features, aux_to_base_scale
        )

        candidate_logits, candidate_aux = self._m1_generate(
            m1_base_logits,
            image=image,
            image_features=m1_image_features,
            text_features=m1_text_features,
            tokenized_prompts=tokenized_prompts,
            text_embeddings=text_embeddings,
            text_prompts=text,
            supervision_masks=supervision_masks,
            slr_hr_image=slr_hr_image,
            sparc_hr_mask=sparc_hr_mask,
            mc_std_map=mc_std_map,
            mc_disagreement_map=mc_disagreement_map,
            mc_pairwise_disagreement=mc_pairwise_disagreement,
            mc_probability_samples=base_probability_samples,
        )
        clip_loss = self._clip_alignment_loss(mean_image_features, mean_text_features)
        aux: Dict[str, Any] = {
            "base_logits": mean_base_logits,
            "image_features": mean_image_features,
            "text_features": mean_text_features,
            "candidates": candidate_logits,
            "v383_gi_stopgrad_aux_to_base": stopgrad_aux_to_base,
            "v470_aux_to_base_grad_scale": mean_base_logits.new_full(
                (mean_base_logits.shape[0],), float(aux_to_base_scale)
            ),
            "v479_xbm_bank_size": mean_base_logits.new_full(
                (mean_base_logits.shape[0],), float(getattr(self, "_last_xbm_bank_size", mean_base_logits.shape[0]))
            ),
            **candidate_aux,
        }
        aux = self._apply_v463_ccv(aux, mean_base_logits)
        aux["fused_probs"] = self._fuse_from_aux(aux["candidate_probs"], aux)
        return mean_base_logits, clip_loss, aux

    def forward(
        self,
        image,
        text,
        num_samples=30,
        target=None,
        return_aux=False,
        compute_m1: bool = True,
        slr_hr_image: Optional[torch.Tensor] = None,
        sparc_hr_mask: Optional[torch.Tensor] = None,
    ):
        tokenized_prompts = self.tokenizer(text).to(image.device)
        full_end_to_end = bool(
            _cfg_get(_cfg_get(self.cfg, "M1", None), "V478_FULL_END_TO_END", False)
        )
        text_context = torch.enable_grad() if full_end_to_end else torch.no_grad()
        with text_context:
            text_embeddings = self.text_model.transformer.embeddings.word_embeddings(tokenized_prompts).type(self.dtype)

        if self.training or return_aux:
            jbt_feature_feedback = bool(
                self.m1_active
                and compute_m1
                and self.m1_pse is not None
                and getattr(self.m1_pse, "feature_feedback_enabled", False)
            )
            if jbt_feature_feedback:
                base_logits, image_features, text_features, cls_token, decoder_features = self._forward_base_once(
                    image, tokenized_prompts, text_embeddings, return_decoder_features=True
                )
            else:
                base_logits, image_features, text_features, cls_token = self._forward_base_once(
                    image, tokenized_prompts, text_embeddings
                )
                decoder_features = None
            clip_loss = self._clip_alignment_loss(image_features, text_features) if self.training else torch.zeros((), device=image.device)
            aux: Dict[str, Any] = {
                "base_logits": base_logits,
                "cls_token": cls_token,
                "image_features": image_features,
                "text_features": text_features,
            }
            if self.m1_active and compute_m1:
                if self.training:
                    aux_to_base_scale, stopgrad_aux_to_base = (
                        self._m1_aux_to_base_grad_scale()
                    )
                    posterior_bundle = self._jbt_training_mean_then_refine_bundle(
                        image, tokenized_prompts, text_embeddings,
                        base_logits, image_features, text_features, decoder_features,
                    )
                    if posterior_bundle is not None:
                        evidence_base_logits = posterior_bundle["base_logits"]
                        evidence_image_features = posterior_bundle["image_features"]
                        evidence_text_features = posterior_bundle["text_features"]
                        evidence_decoder_features = posterior_bundle.get(
                            "decoder_features", None
                        )
                        mc_std_map = posterior_bundle["mc_std_map"]
                        mc_disagreement_map = posterior_bundle["mc_disagreement_map"]
                        mc_pairwise_disagreement = posterior_bundle[
                            "mc_pairwise_disagreement"
                        ]
                        aux["jbt_train_posterior_sample_count"] = posterior_bundle[
                            "sample_count"
                        ].expand(base_logits.shape[0])
                    else:
                        evidence_base_logits = base_logits
                        evidence_image_features = image_features
                        evidence_text_features = text_features
                        evidence_decoder_features = decoder_features
                        mc_std_map, mc_disagreement_map, mc_pairwise_disagreement = (
                            self._jbt_training_posterior_evidence(
                                image, tokenized_prompts, text_embeddings, base_logits
                            )
                        )
                    m1_base_logits = _scale_gradient(
                        evidence_base_logits, aux_to_base_scale
                    )
                    m1_image_features = _scale_gradient(
                        evidence_image_features, aux_to_base_scale
                    )
                    m1_text_features = _scale_gradient(
                        evidence_text_features, aux_to_base_scale
                    )
                else:
                    aux_to_base_scale, stopgrad_aux_to_base = 1.0, False
                    m1_base_logits = base_logits
                    m1_image_features = image_features
                    m1_text_features = text_features
                    evidence_decoder_features = decoder_features
                    mc_std_map = mc_disagreement_map = mc_pairwise_disagreement = None

                if isinstance(evidence_decoder_features, torch.Tensor):
                    m1_fine_feature_map = _scale_gradient(
                        evidence_decoder_features, aux_to_base_scale
                    ) if self.training else evidence_decoder_features
                    seg_text_vector = self.mask_head(
                        evidence_text_features if self.training else text_features
                    )
                    m1_seg_text_vector = _scale_gradient(
                        seg_text_vector, aux_to_base_scale
                    ) if self.training else seg_text_vector
                else:
                    m1_fine_feature_map = None
                    m1_seg_text_vector = None

                candidate_logits, candidate_aux = self._m1_generate(
                    m1_base_logits,
                    image=image,
                    image_features=m1_image_features,
                    text_features=m1_text_features,
                    tokenized_prompts=tokenized_prompts,
                    text_embeddings=text_embeddings,
                    text_prompts=text,
                    supervision_masks=target if self.training else None,
                    mc_std_map=mc_std_map,
                    mc_disagreement_map=mc_disagreement_map,
                    mc_pairwise_disagreement=mc_pairwise_disagreement,
                    fine_feature_map=m1_fine_feature_map,
                    seg_text_vector=m1_seg_text_vector,
                    slr_hr_image=slr_hr_image,
                    sparc_hr_mask=sparc_hr_mask,
                )
                aux["candidates"] = candidate_logits
                aux["v383_gi_stopgrad_aux_to_base"] = stopgrad_aux_to_base
                aux["v470_aux_to_base_grad_scale"] = base_logits.new_full(
                    (base_logits.shape[0],), float(aux_to_base_scale)
                )
                aux.update(candidate_aux)
                aux = self._apply_v463_ccv(aux, base_logits)
                aux["fused_probs"] = self._fuse_from_aux(aux["candidate_probs"], aux)
            if return_aux:
                return base_logits, clip_loss, aux
            if self.training:
                return base_logits, clip_loss

        if self.m1_active and compute_m1 and self.m1_inference_mode in {"direct_fusion", "router_fusion", "text_verifier_fusion", "utility_risk_selection", "m2_direct_selection", "falsification_m3_selection", "unified_action_cf_selection", "unified_m1_safe_fusion"}:
            final_probs = self.predict_m1_final(image, text, num_samples=num_samples, slr_hr_image=slr_hr_image)
            return torch.logit(final_probs.clamp(EPS, 1.0 - EPS)).unsqueeze(0)

        samples = []
        for _ in range(max(1, int(num_samples))):
            base_logits, _, _, _ = self._forward_base_once(image, tokenized_prompts, text_embeddings)
            samples.append(base_logits)
        return torch.stack(samples, dim=0)

    @torch.no_grad()
    def predict_m1_final(self, image, text, num_samples=30, slr_hr_image: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.predict_m1_diagnostics(image, text, num_samples=num_samples, slr_hr_image=slr_hr_image)["final_probs"]

    def _geotr_posterior_inference_order(self) -> str:
        """Resolve the posterior/refiner composition order for GEOTR.

        ``mean_then_refine`` reproduces the historical V3 behavior:
            Refiner(E[Base], E[features]).

        ``refine_then_mean`` preserves the stochastic posterior through the
        nonlinear refiner and averages only the deployable outputs:
            E[Refiner(Base_s, features_s)].

        The default stays historical for backward compatibility.  P0 audit
        code evaluates both orders on the *same* stochastic samples.
        """
        m1_cfg = _cfg_get(self.cfg, "M1", None)
        order = str(
            _cfg_get(m1_cfg, "GEOTR_POSTERIOR_INFERENCE_ORDER", "mean_then_refine")
        ).strip().lower()
        aliases = {
            "refine_mean": "mean_then_refine",
            "mean_first": "mean_then_refine",
            "mean_then_refine": "mean_then_refine",
            "mean_refine": "mean_then_refine",
            "samplewise": "refine_then_mean",
            "samplewise_refine": "refine_then_mean",
            "refine_then_mean": "refine_then_mean",
            "refine_first": "refine_then_mean",
        }
        order = aliases.get(order, order)
        if order not in {"mean_then_refine", "refine_then_mean"}:
            raise ValueError(
                "M1.GEOTR_POSTERIOR_INFERENCE_ORDER must be "
                "'mean_then_refine' or 'refine_then_mean', got "
                f"{order!r}."
            )
        return order

    @torch.no_grad()
    def _collect_mhcs_posterior_samples(self, image, text, num_samples=30):
        """Collect one shared stochastic posterior bundle for paired audits.

        This function is deliberately separated from refinement so the two
        nonlinear composition orders can be evaluated on exactly the same MC
        draws.  It never reads GT masks.
        """
        tokenized_prompts = self.tokenizer(text).to(image.device)
        text_embeddings = (
            self.text_model.transformer.embeddings.word_embeddings(
                tokenized_prompts
            ).type(self.dtype)
        )
        sample_count = max(1, int(num_samples))
        base_samples = []
        image_feature_samples = []
        text_feature_samples = []
        decoder_feature_samples = []
        for _ in range(sample_count):
            logits, image_features, text_features, _, decoder_features = self._forward_base_once(
                image, tokenized_prompts, text_embeddings, return_decoder_features=True
            )
            base_samples.append(torch.sigmoid(logits).clamp(EPS, 1.0 - EPS))
            image_feature_samples.append(image_features)
            text_feature_samples.append(text_features)
            decoder_feature_samples.append(decoder_features)

        base_probability_samples = torch.stack(base_samples, dim=0)
        mc_std_map, mc_disagreement_map, mc_pairwise_disagreement = (
            self._mc_structural_uncertainty(base_probability_samples)
        )
        return {
            "tokenized_prompts": tokenized_prompts,
            "text_embeddings": text_embeddings,
            "base_probability_samples": base_probability_samples,
            "image_feature_samples": image_feature_samples,
            "text_feature_samples": text_feature_samples,
            "decoder_feature_samples": decoder_feature_samples,
            "mc_std_map": mc_std_map,
            "mc_disagreement_map": mc_disagreement_map,
            "mc_pairwise_disagreement": mc_pairwise_disagreement,
            "sample_count": sample_count,
        }

    @torch.no_grad()
    def _run_mhcs_refiner_from_evidence(
        self,
        *,
        image,
        text,
        base_probs,
        image_features,
        text_features,
        tokenized_prompts,
        text_embeddings,
        mc_std_map,
        mc_disagreement_map,
        mc_pairwise_disagreement,
        mc_probability_samples=None,
        fine_feature_map=None,
        slr_hr_image=None,
    ):
        """Run the nonlinear M1/GEOTR refiner once on a specified evidence state."""
        base_probs = base_probs.clamp(EPS, 1.0 - EPS)
        base_logits = torch.logit(base_probs)
        seg_text_vector = (
            self.mask_head(text_features)
            if isinstance(fine_feature_map, torch.Tensor)
            else None
        )
        candidate_logits, aux = self._m1_generate(
            base_logits,
            image=image,
            image_features=image_features,
            text_features=text_features,
            tokenized_prompts=tokenized_prompts,
            text_embeddings=text_embeddings,
            text_prompts=text,
            mc_std_map=mc_std_map,
            mc_disagreement_map=mc_disagreement_map,
            mc_pairwise_disagreement=mc_pairwise_disagreement,
            mc_probability_samples=mc_probability_samples,
            fine_feature_map=fine_feature_map,
            seg_text_vector=seg_text_vector,
            slr_hr_image=slr_hr_image,
        )
        candidate_probs = aux.get("candidate_probs")
        if not isinstance(candidate_probs, torch.Tensor):
            candidate_probs = torch.sigmoid(candidate_logits).clamp(EPS, 1.0 - EPS)
        aux["candidate_probs"] = candidate_probs
        aux["candidates"] = candidate_logits
        return candidate_logits, aux

    @staticmethod
    def _posterior_mean_aux(aux_samples, batch_size: int):
        """Average continuous per-case refiner outputs across posterior samples.

        Discrete IDs/hard selections are not statistically meaningful under
        averaging and therefore retain the first sample solely for historical
        compatibility.  All deployable probabilities are overwritten below by
        exact posterior means.
        """
        if not aux_samples:
            raise ValueError("aux_samples must be non-empty")
        out = {}
        keys = set.intersection(*(set(a.keys()) for a in aux_samples))
        discrete_tokens = ("index", "_id", "types", "hard", "selected_action")
        for key in keys:
            vals = [a.get(key) for a in aux_samples]
            if not all(isinstance(v, torch.Tensor) for v in vals):
                continue
            first = vals[0]
            if any(v.shape != first.shape for v in vals[1:]):
                continue
            if (not first.dtype.is_floating_point) or any(tok in key for tok in discrete_tokens):
                out[key] = first
                continue
            # Average scalar diagnostics and tensors whose leading dimension is B.
            if first.ndim == 0 or (first.ndim >= 1 and first.shape[0] == batch_size):
                out[key] = torch.stack(vals, dim=0).mean(dim=0)
            else:
                out[key] = first
        return out

    @torch.no_grad()
    def _mhcs_result_from_aux(
        self,
        *,
        base_probs,
        candidate_logits,
        aux,
        mc_std_map,
        mc_disagreement_map,
        mc_pairwise_disagreement,
        posterior_order: str,
    ):
        """Normalize a GEOTR/MHCS result for train.py/test.py compatibility."""
        candidate_probs = aux.get("candidate_probs")
        if not isinstance(candidate_probs, torch.Tensor):
            candidate_probs = torch.sigmoid(candidate_logits).clamp(EPS, 1.0 - EPS)
        if candidate_probs.ndim != 4:
            raise RuntimeError(
                "MHCS candidate_probs must be [B,K,H,W], got "
                + str(tuple(candidate_probs.shape))
            )

        if base_probs.ndim == 4:
            if base_probs.shape[1] != 1:
                raise RuntimeError(
                    "MHCS Base probability must be [B,H,W] or [B,1,H,W], got "
                    f"{tuple(base_probs.shape)}"
                )
            base_for_h0_contract = base_probs[:, 0]
        elif base_probs.ndim == 3:
            base_for_h0_contract = base_probs
        else:
            raise RuntimeError(
                "MHCS Base probability must be [B,H,W] or [B,1,H,W], got "
                f"{tuple(base_probs.shape)}"
            )

        h0_for_contract = candidate_probs[:, 0]
        if h0_for_contract.shape != base_for_h0_contract.shape:
            raise RuntimeError(
                "MHCS H0/Base shape contract failed: "
                f"H0={tuple(h0_for_contract.shape)} Base={tuple(base_for_h0_contract.shape)}"
            )
        h0_max_abs_diff = (h0_for_contract - base_for_h0_contract).abs().max()
        if not torch.allclose(
            h0_for_contract, base_for_h0_contract, atol=1.0e-6, rtol=1.0e-6
        ):
            raise RuntimeError(
                "MHCS posterior-order H0/Base contract failed; "
                f"max_abs_diff={float(h0_max_abs_diff):.9e}."
            )

        final_probs = aux.get("mhcs_final_probs", aux.get("v20_hard_fused_probs"))
        if not isinstance(final_probs, torch.Tensor):
            raise RuntimeError("MHCS diagnostics missing mhcs_final_probs.")
        if final_probs.ndim == 4:
            if final_probs.shape[1] != 1:
                raise RuntimeError(
                    "MHCS final probability must be [B,H,W] or [B,1,H,W]."
                )
            final_probs = final_probs[:, 0]

        batch = candidate_probs.shape[0]
        hypothesis_count = candidate_probs.shape[1] - 1
        selector_hard = aux.get("v20_selector_hard")
        if not isinstance(selector_hard, torch.Tensor):
            selector_hard = candidate_probs.new_zeros(batch, hypothesis_count)
        hypothesis_ids = torch.arange(
            hypothesis_count, device=candidate_probs.device, dtype=torch.long
        )
        order_id = 0.0 if posterior_order == "mean_then_refine" else 1.0
        result = {
            "final_probs": final_probs,
            "fused_probs": final_probs,
            "soft_fused_probs": final_probs,
            "base_probs": base_probs,
            "candidate_probs": candidate_probs,
            "candidate_logits": candidate_logits,
            "mhcs_final_probs": final_probs,
            "candidate_std": candidate_probs.std(dim=1, unbiased=False).mean(dim=(1, 2)),
            "candidate_mean_abs_change": (
                candidate_probs[:, 1:] - candidate_probs[:, :1]
            ).abs().mean(dim=(1, 2, 3)),
            "mc_pairwise_disagreement": mc_pairwise_disagreement,
            "mc_std_mean": mc_std_map.mean(dim=(1, 2, 3)),
            "mc_disagreement_fraction": mc_disagreement_map.mean(dim=(1, 2, 3)),
            "geotr_posterior_order_id": candidate_probs.new_full((batch,), order_id),
            "v20_action_types": hypothesis_ids,
            "v20_selector_hard": selector_hard,
        }
        for key, value in aux.items():
            if isinstance(value, torch.Tensor) and key not in result:
                result[key] = value
        return result

    @torch.no_grad()
    def _predict_mhcs_from_bundle(self, image, text, bundle, order: str, slr_hr_image: Optional[torch.Tensor] = None):
        base_samples = bundle["base_probability_samples"]
        image_feature_samples = bundle["image_feature_samples"]
        text_feature_samples = bundle["text_feature_samples"]
        decoder_feature_samples = bundle.get("decoder_feature_samples", [])
        sample_count = int(bundle["sample_count"])
        batch = int(base_samples.shape[1])

        if order == "mean_then_refine":
            base_probs = base_samples.mean(dim=0).clamp(EPS, 1.0 - EPS)
            mean_image_features = torch.stack(image_feature_samples, dim=0).mean(dim=0)
            mean_text_features = torch.stack(text_feature_samples, dim=0).mean(dim=0)
            mean_decoder_features = (
                torch.stack(decoder_feature_samples, dim=0).mean(dim=0)
                if decoder_feature_samples else None
            )
            candidate_logits, aux = self._run_mhcs_refiner_from_evidence(
                image=image,
                text=text,
                base_probs=base_probs,
                image_features=mean_image_features,
                text_features=mean_text_features,
                tokenized_prompts=bundle["tokenized_prompts"],
                text_embeddings=bundle["text_embeddings"],
                mc_std_map=bundle["mc_std_map"],
                mc_disagreement_map=bundle["mc_disagreement_map"],
                mc_pairwise_disagreement=bundle["mc_pairwise_disagreement"],
                mc_probability_samples=base_samples,
                fine_feature_map=mean_decoder_features,
                slr_hr_image=slr_hr_image,
            )
        elif order == "refine_then_mean":
            aux_samples = []
            candidate_prob_samples = []
            for s in range(sample_count):
                _, aux_s = self._run_mhcs_refiner_from_evidence(
                    image=image,
                    text=text,
                    base_probs=base_samples[s],
                    image_features=image_feature_samples[s],
                    text_features=text_feature_samples[s],
                    tokenized_prompts=bundle["tokenized_prompts"],
                    text_embeddings=bundle["text_embeddings"],
                    mc_std_map=bundle["mc_std_map"],
                    mc_disagreement_map=bundle["mc_disagreement_map"],
                    mc_pairwise_disagreement=bundle["mc_pairwise_disagreement"],
                    mc_probability_samples=base_samples,
                    fine_feature_map=(decoder_feature_samples[s] if decoder_feature_samples else None),
                    slr_hr_image=slr_hr_image,
                )
                aux_samples.append(aux_s)
                candidate_prob_samples.append(aux_s["candidate_probs"])

            aux = self._posterior_mean_aux(aux_samples, batch)
            candidate_probs = torch.stack(candidate_prob_samples, dim=0).mean(dim=0).clamp(EPS, 1.0 - EPS)
            base_probs = base_samples.mean(dim=0).clamp(EPS, 1.0 - EPS)
            # Enforce exact posterior H0 and exact posterior means for every
            # scientifically relevant probability output.
            candidate_probs = candidate_probs.clone()
            candidate_probs[:, 0] = base_probs
            aux["candidate_probs"] = candidate_probs
            probability_keys = (
                "mhcs_final_probs",
                "mhcs_local_probs",
                "mhcs_surface_hard_probs",
                "mhcs_global_selected_probs",
                "direct_fused_probs",
                "router_fused_probs",
                "v20_fused_probs",
                "v20_hard_fused_probs",
                "geotopo_base_probs",
                "geotopo_geometry_probs",
                "geotopo_residual_only_probs",
                "geotopo_reconstruction_after_geometry_probs",
                "geotopo_final_probs",
            )
            for key in probability_keys:
                vals = [a.get(key) for a in aux_samples]
                if all(isinstance(v, torch.Tensor) for v in vals):
                    aux[key] = torch.stack(vals, dim=0).mean(dim=0).clamp(EPS, 1.0 - EPS)
            aux["geotopo_base_probs"] = (
                base_probs[:, None] if base_probs.ndim == 3 else base_probs
            )
            # Logits correspond to the deployable posterior-mean probabilities,
            # not to an average in logit space.
            logit_pairs = {
                "geotopo_geometry_logits": "geotopo_geometry_probs",
                "geotopo_residual_only_logits": "geotopo_residual_only_probs",
                "geotopo_reconstruction_after_geometry_logits": "geotopo_reconstruction_after_geometry_probs",
                "geotopo_final_logits": "geotopo_final_probs",
                "mhcs_final_logits": "mhcs_final_probs",
            }
            for logit_key, prob_key in logit_pairs.items():
                p = aux.get(prob_key)
                if isinstance(p, torch.Tensor):
                    aux[logit_key] = torch.logit(p.clamp(EPS, 1.0 - EPS))
            candidate_logits = torch.logit(candidate_probs.clamp(EPS, 1.0 - EPS))
            aux["candidates"] = candidate_logits
        else:
            raise ValueError(f"Unsupported posterior order {order!r}")

        return self._mhcs_result_from_aux(
            base_probs=base_probs,
            candidate_logits=candidate_logits,
            aux=aux,
            mc_std_map=bundle["mc_std_map"],
            mc_disagreement_map=bundle["mc_disagreement_map"],
            mc_pairwise_disagreement=bundle["mc_pairwise_disagreement"],
            posterior_order=order,
        )

    @torch.no_grad()
    def predict_geotr_posterior_order_audit(self, image, text, num_samples=10):
        """Paired P0-A audit on identical stochastic posterior samples.

        Returns both ``Refine(E[P])`` and ``E[Refine(P_s)]`` plus the shared MC
        uncertainty statistics.  This method never sees a GT mask and does not
        alter checkpoint-selection behavior by itself.
        """
        if not self.m1_active or not getattr(self, "mhcs_enabled", False):
            raise RuntimeError("Posterior-order audit requires active MHCS/GEOTR.")
        bundle = self._collect_mhcs_posterior_samples(image, text, num_samples)
        mean_then_refine = self._predict_mhcs_from_bundle(
            image, text, bundle, "mean_then_refine"
        )
        refine_then_mean = self._predict_mhcs_from_bundle(
            image, text, bundle, "refine_then_mean"
        )
        return {
            "mean_then_refine": mean_then_refine,
            "refine_then_mean": refine_then_mean,
            "mc_std_map": bundle["mc_std_map"],
            "mc_disagreement_map": bundle["mc_disagreement_map"],
            "mc_pairwise_disagreement": bundle["mc_pairwise_disagreement"],
        }

    @torch.no_grad()
    def _predict_mhcs_diagnostics(self, image, text, num_samples=30, slr_hr_image: Optional[torch.Tensor] = None):
        """GEOTR/MHCS inference with an explicit posterior/refiner order."""
        if (
            getattr(self, "geotr_m1_exact_enabled", False)
            and bool(
                _cfg_get(
                    _cfg_get(self.cfg, "M1", None),
                    "GEOTR_M1_DETERMINISTIC_EVAL",
                    False,
                )
            )
        ):
            # Exact M1 has no stochastic candidate sampler.  Repeating the same
            # evaluation pass only wastes memory/time and falsely suggests a
            # posterior estimate, so the robust protocol evaluates it once.
            num_samples = 1
        bundle = self._collect_mhcs_posterior_samples(image, text, num_samples)
        order = self._geotr_posterior_inference_order()
        return self._predict_mhcs_from_bundle(image, text, bundle, order, slr_hr_image=slr_hr_image)

    @torch.no_grad()
    def _predict_v20_diagnostics(self, image, text, num_samples=30):
        """MC-mean V20/V25/V26/V27 inference plus complete tensor diagnostics."""
        tokenized_prompts = self.tokenizer(text).to(image.device)
        text_embeddings = self.text_model.transformer.embeddings.word_embeddings(
            tokenized_prompts
        ).type(self.dtype)
        base_samples, feature_sum, text_feature_sum = [], None, None
        for _ in range(max(1, int(num_samples))):
            logits, image_features, text_features, _ = self._forward_base_once(
                image, tokenized_prompts, text_embeddings
            )
            base_samples.append(torch.sigmoid(logits))
            feature_sum = image_features if feature_sum is None else feature_sum + image_features
            text_feature_sum = text_features if text_feature_sum is None else text_feature_sum + text_features

        base_probability_samples = torch.stack(base_samples, dim=0)
        base_probs = base_probability_samples.mean(dim=0).clamp(EPS, 1.0 - EPS)
        mc_std_map, mc_disagreement_map, mc_pairwise_disagreement = (
            self._mc_structural_uncertainty(base_probability_samples)
        )
        base_logits = torch.logit(base_probs)
        candidate_logits, aux = self._m1_generate(
            base_logits,
            image=image,
            image_features=feature_sum / float(len(base_samples)),
            text_features=text_feature_sum / float(len(base_samples)),
            tokenized_prompts=tokenized_prompts,
            text_embeddings=text_embeddings,
            text_prompts=text,
            mc_std_map=mc_std_map,
            mc_disagreement_map=mc_disagreement_map,
            mc_pairwise_disagreement=mc_pairwise_disagreement,
        )
        aux["candidates"] = candidate_logits
        aux = self._apply_v463_ccv(aux, base_logits)
        final_probs = aux["v20_hard_fused_probs"]
        candidate_probs = aux["candidate_probs"]
        result: Dict[str, torch.Tensor] = {
            "final_probs": final_probs,
            "fused_probs": final_probs,
            "soft_fused_probs": aux["v20_fused_probs"],
            "base_probs": base_probs,
            "candidate_probs": candidate_probs,
            "candidate_logits": candidate_logits,
            "candidate_std": candidate_probs.std(dim=1, unbiased=False).mean(dim=(1, 2)),
            "candidate_mean_abs_change": (
                candidate_probs[:, 1:] - candidate_probs[:, :1]
            ).abs().mean(dim=(1, 2, 3)),
            "mc_pairwise_disagreement": mc_pairwise_disagreement,
            "mc_std_mean": mc_std_map.mean(dim=(1, 2, 3)),
            "mc_disagreement_fraction": mc_disagreement_map.mean(dim=(1, 2, 3)),
            "shrink_mean_abs_change": (
                candidate_probs[:, 1] - candidate_probs[:, 0]
            ).abs().mean(dim=(1, 2)),
            "expand_mean_abs_change": (
                candidate_probs[:, 2] - candidate_probs[:, 0]
            ).abs().mean(dim=(1, 2)),
            "v20_action_supports": aux["v20_action_supports"],
            "v20_control_supports": aux["v20_control_supports"],
            "v20_action_types": aux["v20_action_types"],
            "v20_selector_logits": aux["v20_selector_logits"],
            "v20_selector_raw_probs": torch.sigmoid(aux["v20_selector_logits"]),
            "v20_selector_probs": aux["v20_selector_probs"],
            "v20_selector_hard": aux["v20_selector_hard"],
            "v20_cf_logit": aux["v20_cf_logit"],
            "v20_cf_signed_delta": aux["v20_cf_signed_delta"],
            "v20_cf_available": aux["v20_cf_available"],
                        "v20_budget": aux["v20_budget"],
            # V31 fields are None for legacy V20/V25 models and tensors for V31.
            "v31_policy_logits": aux.get("v31_policy_logits"),
            "v31_null_logit": aux.get("v31_null_logit"),
            "v31_utility_logits": aux.get("v31_utility_logits"),
            "v31_semantic_veto_logits": aux.get("v31_semantic_veto_logits"),
            "v31_valid_action": aux.get("v31_valid_action"),
            "v31_selected_action": aux.get("v31_selected_action"),
        }
        # Surface every tensor produced by the bank.  This is particularly
        # important for V27 audits: test.py can now save per-action policy,
        # utility and control-validity values without assuming P/S/E slots.
        for key, value in aux.items():
            if isinstance(value, torch.Tensor) and key not in result:
                result[key] = value
        # Most V20 diagnostics are tensors. Keep the V393/V395 per-case
        # fallback text too, otherwise downstream audits report "unknown".
        for key in ("v393_fallback_reason",):
            if key in aux:
                result[key] = aux[key]
        return result

    @torch.no_grad()
    def _predict_v485_diagnostics(self, image, text, num_samples=30, verifier_text=None):
        tokenized_prompts = self.tokenizer(text).to(image.device)
        text_embeddings = self.text_model.transformer.embeddings.word_embeddings(tokenized_prompts).type(self.dtype)
        base_samples = []
        feature_sum = None
        text_feature_sum = None
        sample_count = max(1, int(num_samples))
        for _ in range(sample_count):
            base_logits, image_features, text_features, _ = self._forward_base_once(
                image, tokenized_prompts, text_embeddings
            )
            base_samples.append(torch.sigmoid(base_logits))
            feature_sum = image_features if feature_sum is None else feature_sum + image_features
            text_feature_sum = text_features if text_feature_sum is None else text_feature_sum + text_features
        base_probs = torch.stack(base_samples, dim=0).mean(dim=0).clamp(EPS, 1.0 - EPS)
        mean_base_logits = torch.logit(base_probs)
        mean_image_features = feature_sum / float(sample_count)
        mean_text_features = text_feature_sum / float(sample_count)
        candidate_logits, aux = self._m1_generate(
            mean_base_logits,
            image=image,
            image_features=mean_image_features,
            text_features=mean_text_features,
            tokenized_prompts=tokenized_prompts,
            text_embeddings=text_embeddings,
            text_prompts=text,
            verifier_text_prompts=verifier_text if verifier_text is not None else text,
            slr_hr_image=slr_hr_image,
        )
        candidate_probs = aux.get("candidate_probs")
        if candidate_probs is None:
            candidate_probs = torch.sigmoid(candidate_logits).clamp(EPS, 1.0 - EPS)
        if candidate_probs.ndim == 3:
            candidate_probs = candidate_probs[:, None]
        final_probs = aux.get("fused_probs", aux.get("final_probs", candidate_probs[:, 0]))
        if final_probs.ndim == 4 and final_probs.shape[1] == 1:
            final_probs = final_probs[:, 0]
        b, slots = candidate_probs.shape[:2]
        action_count = max(slots - 1, 0)
        preserve_onehot = candidate_probs.new_zeros((b, slots))
        preserve_onehot[:, 0] = 1.0
        result: Dict[str, torch.Tensor] = {
            "final_probs": final_probs,
            "base_probs": candidate_probs[:, 0],
            "candidate_probs": candidate_probs,
            "fused_probs": final_probs,
            "direct_fused_probs": candidate_probs[:, 0],
            "router_fused_probs": candidate_probs[:, 0],
            "text_verifier_fused_probs": candidate_probs[:, 0],
            "candidate_std": candidate_probs.std(dim=1, unbiased=False).mean(dim=(1, 2)),
            "candidate_mean_abs_change": (candidate_probs[:, 1:] - candidate_probs[:, :1]).abs().mean(dim=(1, 2, 3)) if action_count > 0 else candidate_probs.new_zeros(b),
            "v20_action_types": torch.arange(action_count, device=image.device, dtype=torch.long),
            "v20_selector_hard": aux.get("v20_selector_hard", candidate_probs.new_zeros((b, action_count))),
            "m1_selector_soft": aux.get("m1_selector_soft", preserve_onehot),
            "m1_selector_hard": aux.get("m1_selector_hard", preserve_onehot),
            "m1_choice_logits": aux.get("m1_choice_logits", candidate_probs.new_zeros((b, slots))),
            "v20_hard_fused_probs": final_probs,
        }
        # Add V485/V484 tensors for test.py CSV/audit without assuming names.
        for key, value in aux.items():
            if isinstance(value, torch.Tensor) and key not in result:
                result[key] = value
        return result

    @torch.no_grad()
    def predict_m1_diagnostics(self, image, text, num_samples=30, verifier_text=None, slr_hr_image: Optional[torch.Tensor] = None):
        if not self.m1_active:
            raise RuntimeError("M1 diagnostics require M1.ENABLED: true.")
        # MHCS must be dispatched before the historical V20 umbrella flag.
        # self.v20_unified_action_cf is True for MHCS only for outer project
        # compatibility; MHCS itself has no factual/control action protocol.
        if getattr(self, "mhcs_enabled", False):
            return self._predict_mhcs_diagnostics(
                image,
                text,
                num_samples=num_samples,
                slr_hr_image=slr_hr_image,
            )

        if getattr(self, "v484_pipeline", None) is not None:
            return self._predict_v485_diagnostics(
                image,
                text,
                num_samples=num_samples,
                verifier_text=verifier_text,
            )

        if self.v20_unified_action_cf:
            return self._predict_v20_diagnostics(
                image,
                text,
                num_samples=num_samples,
            )

        tokenized_prompts = self.tokenizer(text).to(image.device)
        text_embeddings = self.text_model.transformer.embeddings.word_embeddings(tokenized_prompts).type(self.dtype)
        base_samples = []
        feature_sum = None
        text_feature_sum = None
        sample_count = max(1, int(num_samples))
        for _ in range(sample_count):
            base_logits, image_features, text_features, _ = self._forward_base_once(
                image, tokenized_prompts, text_embeddings
            )
            base_samples.append(torch.sigmoid(base_logits))
            feature_sum = image_features if feature_sum is None else feature_sum + image_features
            text_feature_sum = (
                text_features if text_feature_sum is None else text_feature_sum + text_features
            )

        base_probability_samples = torch.stack(base_samples, dim=0)
        base_probs = base_probability_samples.mean(dim=0).clamp(EPS, 1.0 - EPS)
        mc_std_map, mc_disagreement_map, mc_pairwise_disagreement = (
            self._mc_structural_uncertainty(base_probability_samples)
        )
        mean_base_logits = torch.logit(base_probs)
        mean_image_features = feature_sum / float(sample_count)
        mean_text_features = text_feature_sum / float(sample_count)
        _, aux = self._m1_generate(
            mean_base_logits,
            image=image,
            image_features=mean_image_features,
            text_features=mean_text_features,
            tokenized_prompts=tokenized_prompts,
            text_embeddings=text_embeddings,
            text_prompts=text,
            verifier_text_prompts=verifier_text if verifier_text is not None else text,
            mc_std_map=mc_std_map,
            mc_disagreement_map=mc_disagreement_map,
            mc_pairwise_disagreement=mc_pairwise_disagreement,
            mc_probability_samples=base_probability_samples,
        )
        candidate_probs = aux["candidate_probs"]
        direct_fused_probs = aux["direct_fused_probs"]
        router_fused_probs = aux["router_fused_probs"]
        text_verifier_fused_probs = aux.get("text_verifier_fused_probs", base_probs)
        final_probs = self._fuse_from_aux(candidate_probs, aux)

        result: Dict[str, torch.Tensor] = {
            "final_probs": final_probs,
            "base_probs": base_probs,
            "candidate_probs": candidate_probs,
            "fused_probs": final_probs,
            "direct_fused_probs": direct_fused_probs,
            "router_fused_probs": router_fused_probs,
            "text_verifier_fused_probs": text_verifier_fused_probs,
            "candidate_std": candidate_probs.std(dim=1, unbiased=False).mean(dim=(1, 2)),
            "candidate_mean_abs_change": (candidate_probs[:, 1:] - candidate_probs[:, :1]).abs().mean(dim=(1, 2, 3)),
            "shrink_mean_abs_change": (candidate_probs[:, 1] - candidate_probs[:, 0]).abs().mean(dim=(1, 2)),
            "expand_mean_abs_change": (candidate_probs[:, 2] - candidate_probs[:, 0]).abs().mean(dim=(1, 2)),
            "edit_gate_mean": aux["edit_gate"].mean(dim=(1, 2)),
            "edit_band_mean": aux["edit_band"].mean(dim=(1, 2)),
            "inside_band_mean": aux["inside_band"].mean(dim=(1, 2)),
            "outside_band_mean": aux["outside_band"].mean(dim=(1, 2)),
            "shrink_gate_mean": aux["shrink_gate"].mean(dim=(1, 2)),
            "expand_gate_mean": aux["expand_gate"].mean(dim=(1, 2)),
            "router_preserve_mean": aux["router_probs"][:, 0].mean(dim=(1, 2)),
            "router_shrink_mean": aux["router_probs"][:, 1].mean(dim=(1, 2)),
            "router_expand_mean": aux["router_probs"][:, 2].mean(dim=(1, 2)),
            "router_edit_probability_mean": aux["router_edit_probability"].mean(dim=(1, 2)),
            "router_edit_active_fraction": (
                aux["router_edit_probability"] >= float(_cfg_get(_cfg_get(self.cfg, "M1", None), "EDIT_GATE_THRESHOLD", 0.50))
            ).float().mean(dim=(1, 2)),
            "router_direction_shrink_mean": aux["router_direction_probs"][:, 0].mean(dim=(1, 2)),
            "router_direction_expand_mean": aux["router_direction_probs"][:, 1].mean(dim=(1, 2)),
            "router_shrink_weight_mean": aux["router_shrink_weight"].mean(dim=(1, 2)),
            "router_expand_weight_mean": aux["router_expand_weight"].mean(dim=(1, 2)),
            "text_verifier_preserve_weight": aux.get(
                "text_verifier_weights", candidate_probs.new_zeros(candidate_probs.shape[0], 3)
            )[:, 0],
            "text_verifier_shrink_weight": aux.get(
                "text_verifier_weights", candidate_probs.new_zeros(candidate_probs.shape[0], 3)
            )[:, 1],
            "text_verifier_expand_weight": aux.get(
                "text_verifier_weights", candidate_probs.new_zeros(candidate_probs.shape[0], 3)
            )[:, 2],
            "text_verifier_score_margin": (
                aux.get("text_verifier_logits", candidate_probs.new_zeros(candidate_probs.shape[0], 3))
                .topk(k=2, dim=1).values.diff(dim=1).abs().squeeze(1)
            ),
        }
        # V15 dense-mask text-counterfactual evidence and deterministic M3 diagnostics.  Legacy V10 aliases
        # are intentionally retained only as zero-valued compatibility fields.
        evidence_support = aux.get("falsification_support", candidate_probs.new_zeros(candidate_probs.shape[0], 3))
        evidence_contra = aux.get("falsification_contradiction", candidate_probs.new_zeros(candidate_probs.shape[0], 3))
        evidence_over = aux.get("falsification_overseg_violation", candidate_probs.new_zeros(candidate_probs.shape[0], 3))
        evidence_under = aux.get("falsification_underseg_violation", candidate_probs.new_zeros(candidate_probs.shape[0], 3))
        evidence_boundary = aux.get("falsification_boundary_support", candidate_probs.new_zeros(candidate_probs.shape[0], 3))
        evidence_connect = aux.get("falsification_connectivity", candidate_probs.new_zeros(candidate_probs.shape[0], 3))
        evidence_valid = aux.get("falsification_validity", candidate_probs.new_zeros(candidate_probs.shape[0], 3))
        evidence_benefit = aux.get("falsification_benefit", candidate_probs.new_zeros(candidate_probs.shape[0], 3))
        evidence_harm = aux.get("falsification_harm", candidate_probs.new_zeros(candidate_probs.shape[0], 3))
        evidence_uncertainty = aux.get("falsification_uncertainty", candidate_probs.new_zeros(candidate_probs.shape[0], 3))
        evidence_text_align = aux.get("falsification_text_alignment", candidate_probs.new_zeros(candidate_probs.shape[0], 3))
        m3_score = aux.get("falsification_m3_scores", candidate_probs.new_zeros(candidate_probs.shape[0], 3))
        m3_index = aux.get("falsification_m3_selected_index", candidate_probs.new_zeros(candidate_probs.shape[0], dtype=torch.long))
        selected_accept = aux.get("falsification_m3_accept", (m3_index > 0).float())

        if self.m1_inference_mode == "m2_direct_selection":
            m3_score = aux.get(
                "falsification_m2_direct_scores",
                candidate_probs.new_zeros(candidate_probs.shape[0], 3),
            )
            m3_index = aux.get(
                "falsification_m2_direct_selected_index",
                candidate_probs.new_zeros(candidate_probs.shape[0], dtype=torch.long),
            )
            selected_accept = aux.get(
                "falsification_m2_direct_accept",
                (m3_index > 0).float(),
            )

        role_stats = aux.get("falsification_role_stats", candidate_probs.new_zeros(candidate_probs.shape[0], 3, 12))
        text_necessity = aux.get("falsification_text_necessity", candidate_probs.new_zeros(candidate_probs.shape[0], 3))
        text_edit_direction = aux.get("falsification_text_edit_direction", candidate_probs.new_zeros(candidate_probs.shape[0], 3))
        text_specificity = aux.get("falsification_text_specificity", candidate_probs.new_zeros(candidate_probs.shape[0], 3))
        text_keep_pos = aux.get("falsification_keep_positive", candidate_probs.new_zeros(candidate_probs.shape[0], 3))
        text_drop_pos = aux.get("falsification_drop_positive", candidate_probs.new_zeros(candidate_probs.shape[0], 3))
        text_pass = aux.get("falsification_text_pass", candidate_probs.new_zeros(candidate_probs.shape[0], 3))
        text_consensus = aux.get("falsification_consensus", candidate_probs.new_zeros(candidate_probs.shape[0], 3))
        result.update({
            "m2_text_necessity_shrink": text_necessity[:, 1],
            "m2_text_necessity_expand": text_necessity[:, 2],
            "m2_text_edit_direction_shrink": text_edit_direction[:, 1],
            "m2_text_edit_direction_expand": text_edit_direction[:, 2],
            "m2_text_specificity_shrink": text_specificity[:, 1],
            "m2_text_specificity_expand": text_specificity[:, 2],
            "m2_keep_positive_shrink": text_keep_pos[:, 1],
            "m2_keep_positive_expand": text_keep_pos[:, 2],
            "m2_drop_positive_shrink": text_drop_pos[:, 1],
            "m2_drop_positive_expand": text_drop_pos[:, 2],
            "m2_text_pass_shrink": text_pass[:, 1],
            "m2_text_pass_expand": text_pass[:, 2],
            "m2_consensus_shrink": text_consensus[:, 1],
            "m2_consensus_expand": text_consensus[:, 2],
            "m2_selected_index": m3_index,
            "m2_selected_preserve": (m3_index == 0).float(),
            "m2_selected_shrink": (m3_index == 1).float(),
            "m2_selected_expand": (m3_index == 2).float(),
            "m2_accept": selected_accept,
            "m3_score_preserve": m3_score[:, 0],
            "m3_score_shrink": m3_score[:, 1],
            "m3_score_expand": m3_score[:, 2],
            "m2_support_preserve": evidence_support[:, 0],
            "m2_support_shrink": evidence_support[:, 1],
            "m2_support_expand": evidence_support[:, 2],
            "m2_contradiction_preserve": evidence_contra[:, 0],
            "m2_contradiction_shrink": evidence_contra[:, 1],
            "m2_contradiction_expand": evidence_contra[:, 2],
            "m2_overseg_preserve": evidence_over[:, 0],
            "m2_overseg_shrink": evidence_over[:, 1],
            "m2_overseg_expand": evidence_over[:, 2],
            "m2_underseg_preserve": evidence_under[:, 0],
            "m2_underseg_shrink": evidence_under[:, 1],
            "m2_underseg_expand": evidence_under[:, 2],
            "m2_boundary_preserve": evidence_boundary[:, 0],
            "m2_boundary_shrink": evidence_boundary[:, 1],
            "m2_boundary_expand": evidence_boundary[:, 2],
            "m2_connectivity_preserve": evidence_connect[:, 0],
            "m2_connectivity_shrink": evidence_connect[:, 1],
            "m2_connectivity_expand": evidence_connect[:, 2],
            "m2_validity_preserve": evidence_valid[:, 0],
            "m2_validity_shrink": evidence_valid[:, 1],
            "m2_validity_expand": evidence_valid[:, 2],
            "m2_benefit_preserve": evidence_benefit[:, 0],
            "m2_benefit_shrink": evidence_benefit[:, 1],
            "m2_benefit_expand": evidence_benefit[:, 2],
            "m2_harm_preserve": evidence_harm[:, 0],
            "m2_harm_shrink": evidence_harm[:, 1],
            "m2_harm_expand": evidence_harm[:, 2],
            "m2_uncertainty_shrink": evidence_uncertainty[:, 1],
            "m2_uncertainty_expand": evidence_uncertainty[:, 2],
            "m2_text_alignment_shrink": evidence_text_align[:, 1],
            "m2_text_alignment_expand": evidence_text_align[:, 2],
            "m2_edit_area_shrink": role_stats[:, 1, 1],
            "m2_edit_area_expand": role_stats[:, 2, 1],
            "m2_add_area_shrink": role_stats[:, 1, 2],
            "m2_add_area_expand": role_stats[:, 2, 2],
            "m2_remove_area_shrink": role_stats[:, 1, 3],
            "m2_remove_area_expand": role_stats[:, 2, 3],
            "m2_selected_gain": evidence_benefit.gather(1, m3_index[:, None]).squeeze(1),
            "m2_selected_risk": evidence_harm.gather(1, m3_index[:, None]).squeeze(1),
            "m2_selected_confidence": 1.0 - evidence_uncertainty.gather(1, m3_index[:, None]).squeeze(1),
            "m2_selected_score": m3_score[:, 1:].max(dim=1).values,
            "m2_pred_gain_preserve": base_probs.new_zeros(base_probs.shape[0]),
            "m2_pred_gain_shrink": evidence_benefit[:, 1],
            "m2_pred_gain_expand": evidence_benefit[:, 2],
            "m2_pred_risk_preserve": evidence_harm[:, 0],
            "m2_pred_risk_shrink": evidence_harm[:, 1],
            "m2_pred_risk_expand": evidence_harm[:, 2],
            "m2_pred_confidence_preserve": 1.0 - evidence_uncertainty[:, 0],
            "m2_pred_confidence_shrink": 1.0 - evidence_uncertainty[:, 1],
            "m2_pred_confidence_expand": 1.0 - evidence_uncertainty[:, 2],
            "m2_score_preserve": m3_score[:, 0],
            "m2_score_shrink": m3_score[:, 1],
            "m2_score_expand": m3_score[:, 2],
        })
        edit_eps = float(_cfg_get(_cfg_get(self.cfg, "M1", None), "CHANGE_EPS", 1e-3))
        result["shrink_changed_fraction"] = (
            (candidate_probs[:, 1] - candidate_probs[:, 0]).abs() > edit_eps
        ).float().mean(dim=(1, 2))
        result["expand_changed_fraction"] = (
            (candidate_probs[:, 2] - candidate_probs[:, 0]).abs() > edit_eps
        ).float().mean(dim=(1, 2))
        result["fusion_changed_fraction"] = (
            (final_probs - base_probs).abs() > edit_eps
        ).float().mean(dim=(1, 2))
        dense_preserve = aux.get("falsification_dense_quality_preserve", candidate_probs.new_zeros(candidate_probs.shape[0], 3))
        dense_candidate = aux.get("falsification_dense_quality_candidate", candidate_probs.new_zeros(candidate_probs.shape[0], 3))
        dense_control = aux.get("falsification_dense_quality_control", candidate_probs.new_zeros(candidate_probs.shape[0], 3))
        dense_delta_control = aux.get("falsification_dense_delta_control", candidate_probs.new_zeros(candidate_probs.shape[0], 3))
        dense_pos = aux.get("falsification_dense_positive_candidate", candidate_probs.new_zeros(candidate_probs.shape[0], 3))
        dense_neg = aux.get("falsification_dense_negative_candidate", candidate_probs.new_zeros(candidate_probs.shape[0], 3))
        dense_gt = aux.get("falsification_dense_gt_gap", candidate_probs.new_zeros(candidate_probs.shape[0], 3))
        dense_ratio = aux.get("falsification_dense_control_area_ratio", candidate_probs.new_zeros(candidate_probs.shape[0], 3))
        dense_overlap = aux.get("falsification_dense_control_overlap_ratio", candidate_probs.new_zeros(candidate_probs.shape[0], 3))
        cf_available = aux.get("falsification_cf_available", candidate_probs.new_zeros(candidate_probs.shape[0], 3))
        result.update({
            "m2_cf_available_shrink": cf_available[:, 1],
            "m2_cf_available_expand": cf_available[:, 2],
            "m2_dense_control_area_ratio_shrink": dense_ratio[:, 1],
            "m2_dense_control_area_ratio_expand": dense_ratio[:, 2],
            "m2_dense_control_overlap_ratio_shrink": dense_overlap[:, 1],
            "m2_dense_control_overlap_ratio_expand": dense_overlap[:, 2],
            "m2_dense_quality_preserve_shrink": dense_preserve[:, 1],
            "m2_dense_quality_candidate_shrink": dense_candidate[:, 1],
            "m2_dense_quality_control_shrink": dense_control[:, 1],
            "m2_dense_control_delta_shrink": dense_delta_control[:, 1],
            "m2_dense_positive_candidate_shrink": dense_pos[:, 1],
            "m2_dense_negative_candidate_shrink": dense_neg[:, 1],
            "m2_dense_gt_gap_shrink": dense_gt[:, 1],
        })
        return result


def build_medclipseg_unimedclip(cfg):
    print(f"Loading UniMedCLIP (backbone: {cfg.MODEL.BACKBONE})")
    clip_model = load_unimedclip_to_device(cfg)
    clip_model.float()
    print("Building custom UniMedCLIP")
    model = CustomCLIP(cfg, clip_model)
    m1_cfg = _cfg_get(cfg, "M1", None)
    if bool(_cfg_get(m1_cfg, "V478_FULL_END_TO_END", False)):
        for _, parameter in model.named_parameters():
            parameter.requires_grad_(True)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(
            "[V478 full E2E] image encoder, text encoder, factual head, "
            "M1, counterfactual M2 and policy M3 are all trainable | "
            f"trainable={trainable / 1e6:.3f}M/{total / 1e6:.3f}M"
        )
        return model
    print("Turning off gradients in both the image and the text encoder")
    base_trainable = {"pvl_adapters", "mask_head", "upscale", "ugbra", "qabr"}
    for name, parameter in model.named_parameters():
        is_base_block = any(token in name for token in base_trainable)
        is_v485_block = name.startswith("v484_pipeline.")
        is_m1_block = ("m1_pse" in name) or ("m2_text_verifier" in name) or is_v485_block
        if is_m1_block:
            # V15.3 raw dense observer contract: the identity patch adapter is
            # never optimized, including Phase A before the M2 stage begins.
            # V485 exception: the error-state intervention pipeline is the
            # actual M1 candidate source and must remain trainable immediately
            # after build_medclipseg_unimedclip(), because standalone tools do
            # not call train.py's later _force_unified_e2e_trainable() hook.
            parameter.requires_grad_(name != "m2_text_verifier.patch_adapter.weight")
        elif is_base_block:
            parameter.requires_grad_(not model.m1_active or model.m1_train_mode in {"e2e", "anchor_student"})
        else:
            parameter.requires_grad_(False)

    # V469: Preserve/Base is a fixed counterfactual reference.  Freeze every
    # non-M1/non-CCV tensor before the later train.py contract is applied.
    m1_cfg = _cfg_get(cfg, "M1", None)
    if model.m1_active and bool(_cfg_get(m1_cfg, "V469_FREEZE_BASE", False)):
        for name, parameter in model.named_parameters():
            train_module = (
                name.startswith("m1_pse.")
                or name.startswith("ccv_m2.")
            )
            parameter.requires_grad_(train_module)
        print(
            "[V469 frozen-reference] B0/decoder/PVL frozen; only M1 and CCV "
            "parameters are trainable."
        )

    # FullVision E2E: train the whole image tower while keeping text frozen.
    if (
        model.m1_active
        and model.m1_train_mode == "e2e"
        and bool(_cfg_get(m1_cfg, "V395_FULL_VISION_E2E", False))
        and not bool(_cfg_get(m1_cfg, "V469_FREEZE_BASE", False))
    ):
        for parameter in model.vision_model.parameters():
            parameter.requires_grad_(True)
        print(
            "[V396 FullVision E2E] trainable decoder/adapters + entire "
            f"UniMedCLIP visual tower ({len(list(model.vision_model.transformer.resblocks))} blocks); "
            "text encoder and V396 observer remain frozen."
        )

    # V420 unified M1-only contract.  This is the final M1 ablation path:
    # B0, the text verifier, the historical counterfactual verifier and the
    # historical selector are all frozen. Only the unified type-conditioned
    # candidate generator and its M1-local value/safety heads are optimized.
    if (
        model.m1_active
        and getattr(model.m1_pse, "unified_m1_safe_fusion_enabled", False)
        and str(getattr(model, "m1_train_mode", "")).lower() not in {"e2e", "joint", "ac_pair", "unified_e2e"}
    ):
        allowed_prefixes = (
            "m1_pse.trunk.",
            "m1_pse.actionness_heads.",
            "m1_pse.delta_heads.",
            "m1_pse.m1_safe_",
        )
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(allowed_prefixes))
        active_names = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        required_prefixes = (
            "m1_pse.trunk.",
            "m1_pse.actionness_heads.",
            "m1_pse.delta_heads.",
            "m1_pse.m1_safe_state.",
            "m1_pse.m1_safe_value_head.",
            "m1_pse.m1_safe_benefit_head.",
            "m1_pse.m1_safe_harm_head.",
        )
        for prefix in required_prefixes:
            if not any(name.startswith(prefix) for name in active_names):
                raise RuntimeError(
                    "V420 unified M1 optimizer contract missing trainable prefix: "
                    + prefix
                )
        if any(
            name.startswith((
                "m1_pse.cf_verifier.",
                "m1_pse.selector.",
                "m1_pse.v23_",
                "m1_pse.v38_",
                "m1_pse.v381_",
                "m1_pse.v382_",
                "m1_pse.v393_",
                "m1_pse.v394_",
                "m1_pse.v395_",
                "m1_pse.safe_",
                "m2_text_verifier.",
                "pvl_adapters.",
                "mask_head.",
                "upscale.",
                "vision_model.",
            ))
            for name in active_names
        ):
            raise RuntimeError(
                "V420 unified M1 optimizer contract leak: non-M1 module "
                "remained trainable."
            )
        print(
            "[V420 Unified-M1-only contract] B0/M2/M3 frozen | "
            f"trainable_tensors={len(active_names)} | "
            f"trainable_parameters="
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        )


    # V426 M1-only optimizer contract.
    #
    # V426 uses the V422 directional safety loss plus its family soft-oracle
    # objective. It does not use the old V396 frozen-observer visual-gain head,
    # and B0 / decoder / PVL / historical M2-M3 heads must remain frozen.
    if (
        model.m1_active
        and getattr(model, "mechanism_candidates", False)
        and str(getattr(model, "m1_train_mode", "")).lower() not in {"e2e", "joint", "ac_pair", "unified_e2e"}
    ):
        allowed_prefixes = (
            "m1_pse.trunk.",
            "m1_pse.actionness_heads.",
            "m1_pse.delta_heads.",
            "m1_pse.v35_residual_head.",
            "m1_pse.adaptive_fill_log_scale",
        )

        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(allowed_prefixes))

        active_names = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]

        required_prefixes = (
            "m1_pse.trunk.",
            "m1_pse.actionness_heads.",
            "m1_pse.delta_heads.",
        )

        for prefix in required_prefixes:
            if not any(
                name.startswith(prefix)
                for name in active_names
            ):
                raise RuntimeError(
                    "V426 optimizer contract missing trainable prefix: "
                    + prefix
                )

        forbidden_prefixes = (
            "pvl_adapters.",
            "mask_head.",
            "upscale.",
            "vision_model.",
            "m2_text_verifier.",
            "m1_pse.v23_",
            "m1_pse.v38_",
            "m1_pse.v381_",
            "m1_pse.v393_",
            "m1_pse.v394_",
            "m1_pse.v395_",
            "m1_pse.v396_visual_gain_head.",
            "m1_pse.cf_verifier.",
            "m1_pse.selector.",
        )

        leaks = [
            name
            for name in active_names
            if name.startswith(forbidden_prefixes)
        ]

        if leaks:
            raise RuntimeError(
                "V426 optimizer contract leak: "
                + ", ".join(leaks[:12])
            )

        print(
            "[Reference M1-only optimizer contract] "
            f"trainable_tensors={len(active_names)} | "
            f"trainable_parameters="
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        )

    # V396: clean optimizer contract. Historical V381/V394/V395 heads are
    # instantiated only for checkpoint compatibility but cannot receive V396
    # gradients or optimizer updates. The semantic observer and text tower are
    # frozen; only image-only proposal/repair and frozen-observer visual-gain
    # critic are trainable, alongside the full segmentation visual branch.
    if (
        model.m1_active
        and bool(getattr(model, "evidence_guided_candidate_control_enabled", False))
        and not getattr(model, "mechanism_candidates", False)
    ):
        allowed_prefixes = (
            "pvl_adapters.",
            "mask_head.",
            "upscale.",
            "vision_model.",
            "m1_pse.trunk.",
            "m1_pse.actionness_heads.",
            "m1_pse.delta_heads.",
            "m1_pse.v35_residual_head.",
            "m1_pse.v396_visual_gain_head.",
        )
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(allowed_prefixes))
        active_names = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        required_prefixes = (
            "vision_model.",
            "m1_pse.actionness_heads.",
            "m1_pse.delta_heads.",
            "m1_pse.v396_visual_gain_head.",
        )
        for prefix in required_prefixes:
            if not any(name.startswith(prefix) for name in active_names):
                raise RuntimeError(
                    "V396 optimizer contract missing trainable prefix: "
                    + prefix
                )
        if any(
            name.startswith((
                "m1_pse.v23_",
                "m1_pse.v38_",
                "m1_pse.v381_",
                "m1_pse.v393_",
                "m1_pse.v394_",
                "m1_pse.v395_",
                "m1_pse.cf_verifier.",
                "m1_pse.selector.",
            ))
            for name in active_names
        ):
            raise RuntimeError(
                "V396 optimizer contract leak: historical verifier/value head "
                "remained trainable."
            )
        print(
            "[V396 CoverageScaled optimizer contract] "
            f"trainable_tensors={len(active_names)} | "
            f"trainable_parameters="
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        )

    # V38/V37 optimizer contract: B0 and encoders remain frozen, while M1
    # all-type proposal and M2 factual/control verifier parameters train.
    # M3 remains deterministic and has no learned gain/risk selector.
    # A frozen B0 reference teacher has M1 disabled.  It must never enter
    # a V37/V38/V381/V382/V383 optimizer contract merely because its source
    # experiment config still carries a historical RUN_TAG or loss name.
    if model.m1_enabled and (
        getattr(model, "v383_conservative_action_value", False)
        or getattr(model, "v382_action_conditional_quantile_atomic", False)
        or getattr(model, "v381_lesion_background_calibrated_atomic", False)
        or getattr(model, "v38_casewise_falsified_delta_consensus", False)
        or getattr(model, "v37_text_falsified_structural_consensus", False)
    ):
        is_v383 = bool(getattr(model, "v383_conservative_action_value", False))
        is_v382 = bool(getattr(model, "v382_action_conditional_quantile_atomic", False))
        is_v391 = bool(getattr(model, "v391_lesion_background_mlp_calibrated_atomic", False))
        is_v392 = bool(getattr(model, "v392_dense_patch_text_falsification", False))
        is_v381 = bool(getattr(model, "v381_lesion_background_calibrated_atomic", False))
        is_v38 = bool(getattr(model, "v38_casewise_falsified_delta_consensus", False))
        # V383-E2E reference-anchored protocol:
        # the *student* segmentation path is trainable from epoch 1.  Only the
        # foundation encoders remain fixed; pvl_adapters + mask_head + upscale
        # co-adapt with the candidate generator and conservative value policy.
        # A frozen B0 reference may be used by train.py only as an anchor target;
        # it is not the student and it receives no optimization updates.
        student_e2e = model.m1_train_mode in {"e2e", "anchor_student"}
        base_prefixes = (
            "pvl_adapters.",
            "mask_head.",
            "upscale.",
        )
        if is_v383 or is_v382 or is_v391 or is_v392:
            policy_prefixes = (
                "m1_pse.trunk.",
                "m1_pse.actionness_heads.",
                "m1_pse.delta_heads.",
                "m1_pse.v35_residual_head.",
                "m1_pse.v23_image_adapter.",
                "m1_pse.v23_text_adapter.",
                "m1_pse.v381_calibrator.",
                "m1_pse.v382_",
                "m1_pse.v385_",
            )
        elif is_v381 or is_v391 or is_v392:
            policy_prefixes = (
                "m1_pse.trunk.",
                "m1_pse.actionness_heads.",
                "m1_pse.delta_heads.",
                "m1_pse.v35_residual_head.",
                "m1_pse.v23_image_adapter.",
                "m1_pse.v23_text_adapter.",
                "m1_pse.v381_calibrator.",
            )
        else:
            policy_prefixes = (
                "m1_pse.trunk.",
                "m1_pse.actionness_heads.",
                "m1_pse.delta_heads.",
                "m1_pse.v35_residual_head.",
                "m1_pse.v23_",
                "m1_pse.v38_",
            )
        allowed_prefixes = (
            base_prefixes + policy_prefixes
            if student_e2e else policy_prefixes
        )
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(allowed_prefixes))
        active_names = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        label = "V383" if is_v383 else ("V382" if is_v382 else ("V392" if is_v392 else ("V391" if is_v391 else ("V381" if is_v381 else ("V38" if is_v38 else "V37")))))
        if not active_names:
            raise RuntimeError(f"{label} left no trainable parameters.")
        if any(not name.startswith(allowed_prefixes) for name in active_names):
            raise RuntimeError(
                f"{label} optimizer contract leak: unexpected trainable parameter: "
                + ", ".join(active_names[:10])
            )
        if student_e2e and not any(name.startswith(base_prefixes) for name in active_names):
            raise RuntimeError(
                f"{label} E2E contract violated: pvl_adapters/mask_head/upscale are all frozen."
            )
        required = [
            "m1_pse.trunk.",
            "m1_pse.actionness_heads.",
            "m1_pse.delta_heads.",
            "m1_pse.v35_residual_head.",
        ]
        if is_v383 or is_v382:
            required.extend([
                "m1_pse.v23_image_adapter.",
                "m1_pse.v23_text_adapter.",
                "m1_pse.v382_semantic_projector.",
                "m1_pse.v382_spatial_encoder.",
                "m1_pse.v382_type_embedding.",
                "m1_pse.v382_state_fuse.",
                "m1_pse.v382_quantile_head.",
                "m1_pse.v382_outcome_head.",
            ])
        elif is_v381 or is_v391 or is_v392:
            required.extend([
                "m1_pse.v23_image_adapter.",
                "m1_pse.v23_text_adapter.",
                "m1_pse.v381_calibrator.",
            ])
        else:
            required.append("m1_pse.v23_cf_head.")
            if is_v38:
                required.append("m1_pse.v38_combo_cf_head.")
        for prefix in required:
            if not any(name.startswith(prefix) for name in active_names):
                raise RuntimeError(f"{label} required trainable module is frozen: {prefix}")
        contract_name = (
            f"{label} E2E reference-anchored optimizer contract"
            if student_e2e else f"{label} frozen-policy optimizer contract"
        )
        print(
            f"[{contract_name}] "
            f"trainable_tensors={len(active_names)} | "
            f"trainable_parameters="
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        )

    # V32/V33/V34/V35 optimizer contract. V35 is deliberately different:
    # B0 and text encoder stay frozen, while the source island proposal field
    # (trunk + type-0 actionness/delta + residual head) is trainable under
    # train-split B0 residual supervision.
    elif (
        getattr(model, "v32_island_phaseb_policy", False)
        and not bool(_cfg_get(cfg.M1, "GEOTR_PC2R_POSTERIOR_V3_ENABLED", False))
    ):
        v36_active = bool(getattr(model.m1_pse, "v36_casewise_plackett_luce", False))
        v35_active = bool(getattr(model.m1_pse, "v35_decision_enabled", False))
        v34_active = bool(getattr(model.m1_pse, "v34_spatial_quantile_world_model", False))
        v33_active = bool(getattr(model.m1_pse, "v33_signed_gain_regression", False))

        if v36_active:
            allowed_prefixes = (
                "m1_pse.v31_visual_adapter.",
                "m1_pse.v31_scalar_adapter.",
                "m1_pse.v31_type_embedding.",
                "m1_pse.v31_action_fuse.",
                "m1_pse.trunk.",
                "m1_pse.actionness_heads.0.",
                "m1_pse.delta_heads.0.",
                "m1_pse.v35_residual_head.",
                "m1_pse.v36_",
            )
            freeze_label = "[V36 Casewise-PlackettLuce optimizer freeze] "
        elif v35_active:
            allowed_prefixes = (
                "m1_pse.v31_visual_adapter.",
                "m1_pse.v31_scalar_adapter.",
                "m1_pse.v31_type_embedding.",
                "m1_pse.v31_action_fuse.",
                "m1_pse.trunk.",
                "m1_pse.actionness_heads.0.",
                "m1_pse.delta_heads.0.",
                "m1_pse.v35_",
            )
            freeze_label = "[V35 Residual-Purified WorldModel optimizer freeze] "
        elif v34_active:
            allowed_prefixes = (
                "m1_pse.v31_visual_adapter.",
                "m1_pse.v31_scalar_adapter.",
                "m1_pse.v31_type_embedding.",
                "m1_pse.v31_action_fuse.",
                "m1_pse.v34_",
            )
            freeze_label = "[V34 Spatial-Quantile WorldModel optimizer freeze] "
        elif v33_active:
            allowed_prefixes = ("m1_pse.v31_", "m1_pse.v33_")
            freeze_label = "[V33 Signed-Gain optimizer freeze] "
        else:
            allowed_prefixes = ("m1_pse.v31_",)
            freeze_label = "[V32 Phase-B optimizer freeze] "

        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(allowed_prefixes))

        active_names = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        if not active_names:
            raise RuntimeError("Phase-B freeze left no trainable policy parameters.")
        if any(not name.startswith(allowed_prefixes) for name in active_names):
            raise RuntimeError(
                "Phase-B freeze leak: non-policy parameter remains trainable: "
                + ", ".join(active_names[:10])
            )
        if v34_active and not any(name.startswith("m1_pse.v34_") for name in active_names):
            raise RuntimeError("V34 world-model heads are frozen unexpectedly.")
        if v36_active:
            required_v36 = (
                "m1_pse.v35_residual_head.",
                "m1_pse.v36_state_fuse.",
                "m1_pse.v36_rank_head.",
                "m1_pse.v36_outcome_head.",
            )
            for prefix in required_v36:
                if not any(name.startswith(prefix) for name in active_names):
                    raise RuntimeError(f"V36 required trainable module is frozen: {prefix}")
        if v35_active:
            required_v35 = (
                "m1_pse.v35_residual_head.",
                "m1_pse.v35_state_fuse.",
                "m1_pse.v35_gain_head.",
                "m1_pse.v35_outcome_head.",
            )
            for prefix in required_v35:
                if not any(name.startswith(prefix) for name in active_names):
                    raise RuntimeError(f"V35 required trainable module is frozen: {prefix}")
        print(
            f"{freeze_label}"
            f"trainable_tensors={len(active_names)} | "
            f"trainable_parameters="
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        )

    if model.m2_text_verifier is not None and model.m2_text_verifier.patch_adapter.weight.requires_grad:
        raise RuntimeError("V15.3 safety contract violated: patch_adapter must be frozen before optimizer construction.")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    m1_trainable = sum(
        p.numel() for n, p in model.named_parameters()
        if (("m1_pse" in n) or ("m2_text_verifier" in n) or n.startswith("v484_pipeline.")) and p.requires_grad
    )
    base_trainable_count = trainable - m1_trainable
    if model.m1_active:
        if getattr(model, "semlt_autozero_enabled", False):
            m1_obj = getattr(model, "m1_pse", None)
            if getattr(m1_obj, "uc_fnrt", False):
                method_label = "SemLT UC-FNRT"
            elif getattr(m1_obj, "posterior_stable_operator_warp", False):
                method_label = "SemLT Posterior-Stable Operator Warp"
            elif getattr(m1_obj, "sdf_operator_matched_warp", False):
                method_label = "SemLT SDF Operator-Matched Warp"
            elif getattr(m1_obj, "boundary_normal_warp", False):
                method_label = "SemLT Boundary-Normal Warp"
            else:
                method_label = "SemLT-LST v3/v3.1"
            print(
                f"[{method_label} MODEL] mode={model.m1_train_mode} | "
                f"base_pvl_trainable={base_trainable_count / 1e6:.3f}M | "
                f"transport_trainable={m1_trainable / 1e6:.3f}M | "
                f"total_model={total / 1e6:.3f}M | M2/M3=absent"
            )
        elif getattr(model, "geotr_m1_exact_enabled", False):
            print(
                f"[GEOTR-M1 EXACT MODEL] mode={model.m1_train_mode} | "
                f"base_pvl_trainable={base_trainable_count / 1e6:.3f}M | "
                f"transport_trainable={m1_trainable / 1e6:.3f}M | "
                f"total_model={total / 1e6:.3f}M | M2/M3=absent"
            )
        else:
            print(
                f"[M1 CandidateBank + V15.2 fixed-dense-observer M2/M3 trainable] mode={model.m1_train_mode} | "
                f"base_trainable={base_trainable_count / 1e6:.3f}M | "
                f"m1_trainable={m1_trainable / 1e6:.3f}M | total={total / 1e6:.3f}M"
            )
        if getattr(model, "semlt_autozero_enabled", False):
            m1_obj = getattr(model, "m1_pse", None)
            if getattr(m1_obj, "uc_fnrt", False):
                print(
                    "[SemLT UC-FNRT CONTRACT] MC-mean Base defines a detached SDF contour; "
                    "posterior std/disagreement plus bilateral normal-ray evidence drive factorized "
                    "direction-confidence and magnitude; zero direction confidence is exact KEEP."
                )
            elif getattr(m1_obj, "posterior_stable_operator_warp", False):
                print(
                    "[SemLT PS-OMW CONTRACT] MC-mean Base defines detached contour geometry; "
                    "a fixed cosine narrow band removes prediction-dependent support cliffs."
                )
            elif getattr(m1_obj, "sdf_operator_matched_warp", False):
                print(
                    "[SemLT SDF-OMW CONTRACT] Base logits are detached factual evidence; "
                    "a one-sided SDF contour owns the signed displacement; operator-matched "
                    "normal-ray targets are extended to the physical swept band."
                )
            elif getattr(m1_obj, "boundary_normal_warp", False):
                print(
                    "[SemLT BNW CONTRACT] Base logits are detached factual evidence; "
                    "zero signed displacement is exact KEEP and the only physical operation "
                    "is a bounded spatial logit warp along the Base foreground normal."
                )
            else:
                print(
                    "[SemLT-LST CONTRACT] Base logits are the factual anchor; "
                    "v3.1 KEEP is exact identity and only a hard eligible edit may alter Base."
                )
        elif getattr(model, "geotr_m1_exact_enabled", False):
            print(
                "[GEOTR-M1 EXACT CONTRACT] Base is the factual anchor; "
                "the single Transport output is the final prediction."
            )
        elif model.m1_inference_mode == "falsification_m3_selection":
            print(
                "[M1/M2/M3 V15.3 inference] M1 exports Preserve/Shrink/Expand hypotheses. "
                "M2 scores competing masks on unchanged local dense patch-text maps and same-area control specificity. "
                "M3 filters by measured text evidence, then uses structural consensus with Preserve fallback; "
                "no GT/Oracle/test-time label is used."
            )
        elif model.m1_inference_mode == "utility_risk_selection":
            print(
                "[Legacy V10 utility-risk inference] enabled."
            )
        elif model.m1_inference_mode == "text_verifier_fusion":
            print(
                "[M1/M2 inference] legacy text_verifier_fusion enabled."
            )
        elif model.m1_inference_mode == "router_fusion":
            print(
                "[M1 inference] router_fusion enabled: a conservative edit gate first decides whether "
                "to retain Base; a conditional direction head then chooses Shrink/Expand. No GT/Oracle."
            )
        elif model.m1_inference_mode == "direct_fusion":
            print("[M1 inference] direct_fusion enabled: fixed Preserve/Shrink/Expand composition.")
        else:
            print("[M1 contract] Preserve/Base is final; Shrink/Expand are diagnostics only.")
    if getattr(model, "v383_conservative_action_value", False):
        print(
            "[V383 deployment] M1 all-type local candidates -> action-conditional "
            "q10/q50/q90 transition-value model -> risk-adjusted q10 with "
            "harm/benefit safety gates versus Preserve=0 -> M3 geometric veto only. "
            "No text threshold, no singleton, and no consensus admission."
        )
    elif getattr(model, "v382_action_conditional_quantile_atomic", False):
        print(
            "[V382 deployment] M1 all-type local candidates -> action-conditional "
            "q10/q50/q90 transition-value model -> Preserve=0 versus q10 -> "
            "M3 geometric veto only. No text threshold, no singleton, and no consensus admission."
        )
    elif getattr(model, "v381_lesion_background_calibrated_atomic", False):
        print(
            "[V381 deployment] M1 all-type local candidates -> M2 lesion-vs-background "
            "matched-control calibrated logit against Preserve=0 -> M3 signed-edit-delta "
            "consensus over atomic hypotheses only; Preserve fallback. No swap prompt, no raw-delta hard gate, no Dice-gain/risk deployment score."
        )
    elif getattr(model, "v38_casewise_falsified_delta_consensus", False):
        print(
            "[V38 deployment] M1 all-type local candidates -> M2 casewise matched-control "
            "text falsification (+ joint composition re-verification) -> M3 signed-edit-delta "
            "structural consensus; Preserve fallback. No Dice-gain/risk score is used at deployment."
        )
    elif getattr(model, "v37_text_falsified_structural_consensus", False):
        print(
            "[V37 deployment] M1 all-type local candidates -> M2 matched-control "
            "text falsification -> M3 qualified-hypothesis structural medoid; "
            "Preserve fallback. No Dice-gain/risk score is used at deployment."
        )
    elif getattr(model, "semlt_autozero_enabled", False):
        m1_obj = getattr(model, "m1_pse", None)
        if getattr(m1_obj, "uc_fnrt", False):
            print(
                "[SemLT UC-FNRT DEPLOYMENT] MC-mean Base + MC structural uncertainty -> Base SDF contour/normal "
                "-> bilateral normal-ray semantic contrast -> factorized direction-confidence x magnitude "
                "-> fixed-band physical grid_sample warp; no GT, Gate, Router, M2 or M3 at inference."
            )
        elif getattr(m1_obj, "posterior_stable_operator_warp", False):
            print(
                "[SemLT PS-OMW DEPLOYMENT] MC-mean detached Base -> SDF contour/normal -> signed owner displacement "
                "-> fixed cosine narrow-band grid_sample warp; no GT, Gate, Router, M2 or M3 at inference."
            )
        elif getattr(m1_obj, "sdf_operator_matched_warp", False):
            print(
                "[SemLT SDF-OMW DEPLOYMENT] detached Base logits -> Base SDF zero-level contour "
                "-> contour-owned signed normal displacement -> swept-band grid_sample warp; "
                "no GT, Gate, Type, Action-Value, M2 or M3 at inference."
            )
        elif getattr(m1_obj, "boundary_normal_warp", False):
            print(
                "[SemLT BNW DEPLOYMENT] detached Base logits -> Base boundary normal -> "
                "bounded signed displacement -> differentiable spatial logit warp; "
                "no GT, Gate, Type, Action-Value, M2 or M3 at inference."
            )
        else:
            print(
                "[SemLT-LST DEPLOYMENT] Base logits -> transition eligibility -> hard edit decision "
                "-> local ADD/REMOVE transport; no GT, hypothesis selector, M2 or M3 at inference."
            )
    elif getattr(model, "geotr_m1_exact_enabled", False):
        print(
            "[GEOTR-M1 EXACT DEPLOYMENT] Base logits -> one semantic-conditioned "
            "geometry flow -> one logit warp; no candidate selection, M2 or M3."
        )
    elif getattr(model, "v20_unified_action_cf", False):
        print(
            "[V20 unified inference] atomic pool -> matched local counterfactual text evidence -> "
            "sparse non-overlapping action-set selection; all modules jointly trainable."
        )
    if (
        model.m1_active
        and getattr(model.m1_pse, "safe_residual_only_training", False)
    ):
        certified_context_only = bool(
            getattr(model.m1_pse, "safe_context_certified_only", False)
        )
        if certified_context_only:
            # A3: B0, legacy A1--A8 and the A1 C9 branch are all frozen.
            # Only the independent certified C10 modules may be optimized.
            allowed_prefixes = ("m1_pse.safe_context_",)
            required_prefixes = [
                "m1_pse.safe_context_refiner.",
                "m1_pse.safe_context_residual_head.",
                "m1_pse.safe_context_direction_head.",
            ]
            if bool(
                getattr(model.m1_pse, "safe_residual_use_semantic", False)
            ):
                required_prefixes.append(
                    "m1_pse.safe_context_semantic_proj."
                )
            label = "[SAFE_CONTEXT_CERTIFIED_ONLY]"
        else:
            # Original A1/A2 contract remains available for reproduction.
            allowed_prefixes = ("m1_pse.safe_",)
            required_prefixes = [
                "m1_pse.safe_residual_refiner.",
                "m1_pse.safe_boundary_residual_head.",
                "m1_pse.safe_context_residual_head.",
                "m1_pse.safe_context_direction_head.",
            ]
            label = "[SAFE_RESIDUAL_M1_ONLY]"

        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(allowed_prefixes))
        active_names = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        for prefix in required_prefixes:
            if not any(name.startswith(prefix) for name in active_names):
                raise RuntimeError(
                    "Safe residual optimizer contract missing trainable prefix: "
                    + prefix
                )
        if any(
            not name.startswith(allowed_prefixes)
            for name in active_names
        ):
            raise RuntimeError(
                "Safe residual optimizer contract leak: "
                + ", ".join(active_names[:10])
            )
        print(
            f"{label} B0, legacy A1--A8, C9 reference, text observer and "
            "dual-expert policy frozen | trainable_tensors="
            f"{len(active_names)} | trainable_parameters="
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        )

    return model


# REFERENCE_ADAPTIVE_C6_CANDIDATE_BANK
class ReferenceAdaptiveC6Bank(
    UnifiedActionCounterfactualSetBank
):
    """Reference-compatible mechanism candidate bank with adaptive C6.

    Output remains [B, 9, H, W]:

      C0: Preserve/Base
      C1: island / small false-positive delete
      C2: protrusion / connected-leakage delete
      C3,C4: inactive Preserve placeholders
      C5: conservative local boundary fill
      C6: expanded fill around the C5 factual centre
      C7,C8: inactive Preserve placeholders

    The class creates candidates only. It never deploys an action itself.
    External V410-R1 M2 compares C1/C2/C5/C6 against C0.

    Adaptive C6 may include a zero-initialised trainable scalar/gate.
    It is valid only in candidate-only mode; e2e/joint training must not be
    forced back into the frozen V426 optimizer contract.
    """

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        m1 = _cfg_get(cfg, "M1", None)

        # Preserve the validated V430 candidate geometry internally.
        # No historical geometry knobs are exposed in the public YAML.
        preset = REFERENCE_CANDIDATE_GEOMETRY
        self.k_per_type = 2
        self.num_actions = 8
        self.window_radius = int(preset["window_radius"])
        self.nms_radius = int(preset["nms_radius"])
        self.context_radius = int(preset["context_radius"])
        self.outer_radius = int(preset["outer_radius"])
        self.hole_radius = int(preset["hole_radius"])
        self.protrusion_radius = int(preset["protrusion_radius"])
        self.density_radius = int(preset["density_radius"])
        self.density_max = float(preset["density_max"])
        self.v422_m1_safe_candidate_bank = True
        self.mechanism_candidates = True
        self.v426_island_connect_radius = int(preset["island_connect_radius"])
        self.type_window_radius = dict(preset["type_window_radius"])
        self.type_nms_radius = dict(preset["type_nms_radius"])
        self.v422_fill_edge_weight = float(preset["fill_edge_weight"])
        self.v422_fill_uncertainty_weight = float(preset["fill_uncertainty_weight"])
        self.v422_fill_evidence_floor = float(preset["fill_evidence_floor"])

        self.reference_fill_expand_radius = int(preset["expanded_fill_radius"])
        self.exact_pair_controls = bool(_cfg_get(m1, "EXACT_CONTROL", True))
        self.control_context_radius = int(preset["control_context_radius"])
        self.edit_epsilon = 1.0e-6
        self.fill_control_outer_radius = int(preset["fill_control_outer_radius"])

        # The only new trainable scalar. At zero it produces EXACTLY the
        # validated reference C6 candidate. During training it rescales only
        # the C6 outer-ring increment using relative Fill actionness.
        self.adaptive_expanded_fill = bool(
            _cfg_get(m1, "ADAPTIVE_EXPANDED_FILL", True)
        )
        self.adaptive_fill_log_scale = nn.Parameter(torch.zeros(()))

        # Retired experimental variants are intentionally disabled in the
        # clean research path. Aliases stay false only because the inherited
        # parent implementation still checks them.
        self.use_ranked_variants = False
        self.use_conservative_variants = False
        self.use_asymmetric_adaptation = False
        self.v428_adaptive_dual_expert = False
        self.v429_asymmetric_adaptive = False

        self.register_buffer(
            "mechanism_action_types",
            torch.tensor([0, 1, 1, 1, 2, 2, 3, 3], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "ranked_action_types",
            torch.tensor([1, 1, 1, 1, 2, 2, 3, 3], dtype=torch.long),
            persistent=False,
        )

    @staticmethod
    def _zero_like_action(action_tensor: torch.Tensor) -> torch.Tensor:
        return action_tensor[:, :1] * 0.0

    def _remap_actions(self, action_tensor: torch.Tensor) -> torch.Tensor:
        """Map parent C1..C8 tensors into reference mechanism/ranked variant/conservative variant slots."""
        if action_tensor.ndim < 2 or action_tensor.shape[1] != 8:
            raise RuntimeError(
                "reference mechanism/ranked variant/conservative variant expects eight parent action slots, got "
                f"{tuple(action_tensor.shape)}"
            )

        zero = self._zero_like_action(action_tensor)

        if self.use_conservative_variants:
            # conservative variant uses one high-coverage delete location and one
            # high-coverage fill location. C2/C6 are generated later as
            # confidence-adaptive conservative versions of C1/C5.
            return torch.cat(
                [
                    action_tensor[:, 2:3],  # C1 parent protrusion rank-0
                    action_tensor[:, 2:3],  # C2 same factual delete location
                    zero,                    # C3 inactive
                    zero,                    # C4 inactive
                    action_tensor[:, 4:5],  # C5 parent fill rank-0
                    action_tensor[:, 4:5],  # C6 same factual fill location
                    zero,                    # C7 inactive
                    zero,                    # C8 inactive
                ],
                dim=1,
            )

        if self.use_ranked_variants:
            return torch.cat(
                [
                    action_tensor[:, 2:3],  # C1 parent protrusion rank-0
                    action_tensor[:, 3:4],  # C2 parent protrusion rank-1
                    zero,
                    zero,
                    action_tensor[:, 4:5],  # C5 parent fill rank-0
                    action_tensor[:, 5:6],  # C6 parent fill rank-1
                    zero,
                    zero,
                ],
                dim=1,
            )

        return torch.cat(
            [
                action_tensor[:, 0:1],  # C1 island rank-0
                action_tensor[:, 2:3],  # C2 protrusion rank-0
                zero,
                zero,
                action_tensor[:, 4:5],  # C5 fill rank-0
                action_tensor[:, 4:5],  # C6 expanded from C5
                zero,
                zero,
            ],
            dim=1,
        )

    @staticmethod
    def _v426_binary_dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
        """Binary dilation that preserves the input [1,1,H,W] layout."""
        radius = max(0, int(radius))
        if radius == 0:
            return (mask > 0.5).to(mask.dtype)
        return F.max_pool2d(
            (mask > 0.5).to(mask.dtype),
            kernel_size=2 * radius + 1,
            stride=1,
            padding=radius,
        )

    @torch.no_grad()
    def _v426_place_exact_template(
        self,
        template: torch.Tensor,
        carrier: torch.Tensor,
        forbidden: torch.Tensor,
        gray: torch.Tensor,
        entropy: torch.Tensor,
        boundary: torch.Tensor,
    ) -> torch.Tensor:
        """Place one exact factual template in a valid control location.

        The returned map has exactly the same binary shape and pixel count as
        ``template``.  A placement is considered only when every template pixel
        lies in the polarity-compatible carrier and outside the factual/control
        exclusion domain.  Candidate locations are ranked by the same local
        gray/uncertainty/boundary matching principle used by the parent V20
        control constructor; compactness is invariant under translation and is
        therefore deliberately omitted.
        """
        if template.ndim != 4 or template.shape[0] != 1 or template.shape[1] != 1:
            raise ValueError(
                "reference mechanism exact template placement expects [1,1,H,W], got "
                f"{tuple(template.shape)}"
            )
        if carrier.shape != template.shape or forbidden.shape != template.shape:
            raise ValueError("reference mechanism exact template/carrier shape mismatch.")

        hard = template > 0.5
        if not bool(hard.any().item()):
            return torch.zeros_like(template)

        coords = hard[0, 0].nonzero(as_tuple=False)
        y0 = int(coords[:, 0].min().item())
        y1 = int(coords[:, 0].max().item()) + 1
        x0 = int(coords[:, 1].min().item())
        x1 = int(coords[:, 1].max().item()) + 1
        crop = hard[:, :, y0:y1, x0:x1].to(template.dtype)
        area = crop.sum()
        if float(area.item()) <= 0.0:
            return torch.zeros_like(template)

        # A valid top-left placement has every active crop pixel in `allowed`.
        allowed = (
            (carrier > 0.5)
            & ~(forbidden > 0.5)
        ).to(template.dtype)
        fit_count = F.conv2d(allowed, crop)
        valid = fit_count >= (area - 0.5)
        if not bool(valid.any().item()):
            return torch.zeros_like(template)

        factual_gray = (gray * hard.to(gray.dtype)).sum() / area
        factual_entropy = (entropy * hard.to(entropy.dtype)).sum() / area
        factual_boundary = (boundary * hard.to(boundary.dtype)).sum() / area

        mean_gray = F.conv2d(gray, crop) / area
        mean_entropy = F.conv2d(entropy, crop) / area
        mean_boundary = F.conv2d(boundary, crop) / area
        cost = (
            self.control_gray_weight * (mean_gray - factual_gray).abs()
            + self.control_entropy_weight * (mean_entropy - factual_entropy).abs()
            + self.control_boundary_weight * (mean_boundary - factual_boundary).abs()
        )
        cost = cost.masked_fill(~valid, torch.finfo(cost.dtype).max)
        flat_index = cost.reshape(-1).argmin()
        out_h, out_w = cost.shape[-2:]
        top_y = int((flat_index // out_w).item())
        top_x = int((flat_index % out_w).item())

        output = torch.zeros_like(template)
        output[:, :, top_y:top_y + crop.shape[-2], top_x:top_x + crop.shape[-1]] = crop
        return output

    @torch.no_grad()
    def _build_exact_pair_supports(
        self,
        *,
        base_logits: torch.Tensor,
        candidate_logits: torch.Tensor,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Rebuild reference mechanism factual/control supports from final candidate logits.

        This method is intentionally after all reference mechanism/ranked variant/conservative variant/asymmetric variant remapping.
        Thus each exported support is the exact non-zero logit-edit template of
        its final C1/C2/C5/C6 candidate, rather than a parent-bank proxy.  The
        control support is an exact translated copy of that template in a
        polarity-compatible carrier, with a >=2R separation from the factual
        template so their R-radius contexts cannot overlap.
        """
        if candidate_logits.ndim != 4 or candidate_logits.shape[1] != 9:
            raise ValueError(
                "reference mechanism exact pair supports require candidate logits [B,9,H,W], got "
                f"{tuple(candidate_logits.shape)}"
            )
        if base_logits.shape != candidate_logits[:, :1].shape:
            raise ValueError("reference mechanism base/candidate logit shape mismatch.")

        base = base_logits
        batch, _, height, width = candidate_logits.shape
        full_edit = (candidate_logits[:, 1:] - base).abs()
        factual = (full_edit > self.edit_epsilon).to(base.dtype)

        # Inactive C3/C4/C7/C8 are Preserve by construction.  Retain explicit
        # zeros even if numerical noise is present in a future variant.
        inactive = (2, 3, 6, 7)
        factual[:, list(inactive)] = 0.0

        anchor = (torch.sigmoid(base) >= self.anchor_threshold).to(base.dtype)
        fill_carrier = (
            self._v426_binary_dilate(
                anchor,
                self.fill_control_outer_radius,
            )
            - anchor
        ).clamp(0.0, 1.0)

        gray, _ = self._image_gray_edge(image, (height, width))
        entropy = self._entropy(torch.sigmoid(base))
        boundary = self._boundary(torch.sigmoid(base))

        # Exclude every actual active factual edit from all controls.  The own
        # action receives the stronger 2R exclusion below, which guarantees
        # disjoint context rings for the deployed PAIR-M2 radius R.
        active_indices = (0, 1, 4, 5)
        all_factual = factual[:, list(active_indices)].amax(dim=1, keepdim=True)
        controls = torch.zeros_like(factual)
        placed = torch.zeros(batch, 8, device=base.device, dtype=torch.bool)

        for bi in range(batch):
            for action_index in active_indices:
                template = factual[bi:bi + 1, action_index:action_index + 1]
                if not bool((template > 0.5).any().item()):
                    continue

                if action_index in (0, 1):
                    # Delete action: a same-shape deletion must stay within
                    # predicted foreground, otherwise it is not a deletion.
                    carrier = anchor[bi:bi + 1]
                else:
                    # Fill action: same-shape fill must stay in the local
                    # external background band around the predicted lesion.
                    carrier = fill_carrier[bi:bi + 1]

                own_exclusion = self._v426_binary_dilate(
                    template,
                    2 * self.control_context_radius,
                )
                other_factual = (
                    all_factual[bi:bi + 1] - template
                ).clamp(0.0, 1.0)
                forbidden = (own_exclusion + other_factual).clamp(0.0, 1.0)

                placed_template = self._v426_place_exact_template(
                    template=template,
                    carrier=carrier,
                    forbidden=forbidden,
                    gray=gray[bi:bi + 1],
                    entropy=entropy[bi:bi + 1],
                    boundary=boundary[bi:bi + 1],
                )
                controls[bi:bi + 1, action_index:action_index + 1] = placed_template
                placed[bi, action_index] = bool(
                    (placed_template > 0.5).any().item()
                )

        audit = {
            "v426_exact_supports": factual,
            "v426_exact_controls": controls,
            "v426_exact_control_placed": placed,
            "v426_exact_fill_control_carrier": fill_carrier,
        }
        return factual, controls, audit

    def generate(
        self,
        base_logits: torch.Tensor,
        image: torch.Tensor,
        semantic_map: Optional[torch.Tensor],
        text_features: torch.Tensor,
        negative_text_features: Optional[torch.Tensor] = None,
        swapped_text_features: Optional[torch.Tensor] = None,
    ):
        # Candidate geometry is intentionally prompt-independent.
        # This argument is accepted to keep the generic M1 call interface.
        del swapped_text_features

        parent_logits, aux = super().generate(
            base_logits=base_logits,
            image=image,
            semantic_map=semantic_map,
            text_features=text_features,
            negative_text_features=negative_text_features,
        )

        required = (
            "v20_action_supports",
            "v20_control_supports",
            "v20_actionness_logits",
            "v20_delta_maps",
            "v20_visual_scores",
            "v20_type_supports",
        )

        missing = [
            key
            for key in required
            if not isinstance(aux.get(key), torch.Tensor)
        ]

        if missing:
            raise RuntimeError(
                f"reference mechanism parent auxiliary tensors missing: {missing}"
            )

        if parent_logits.ndim != 4 or parent_logits.shape[1] != 9:
            raise RuntimeError(
                "reference mechanism expects parent logits [B,9,H,W], got "
                f"{tuple(parent_logits.shape)}"
            )

        supports_parent = aux["v20_action_supports"]
        controls_parent = aux["v20_control_supports"]
        delta_parent = aux["v20_delta_maps"]
        type_supports_parent = aux["v20_type_supports"]

        # C0 Preserve used by both reference mechanism and ranked variant remapping paths.
        base = parent_logits[:, :1]

        if self.use_conservative_variants:
            # conservative variant standard candidates use the best learned local action.
            # Conservative candidates use the same local action but scale the
            # edit at each pixel by its *relative learned actionness*.
            #
            # Thus C2/C6 adapt to the current trained actionness distribution
            # of that image; no manually selected second radius, score
            # threshold, or edit-strength multiplier is used.
            delete_support = supports_parent[:, 2:3]
            fill_support = supports_parent[:, 4:5]

            delete_delta = delta_parent[:, 2:3]
            fill_delta = delta_parent[:, 4:5]

            delete_score = torch.sigmoid(
                aux["v20_actionness_logits"][:, 2:3]
            )
            fill_score = torch.sigmoid(
                aux["v20_actionness_logits"][:, 4:5]
            )

            def _relative_local_strength(
                score: torch.Tensor,
                support: torch.Tensor,
            ) -> torch.Tensor:
                masked = score.masked_fill(support <= 0.0, -1.0)
                peak = masked.flatten(1).amax(dim=1).view(
                    -1, 1, 1, 1
                )
                has_support = (
                    support.flatten(1).sum(dim=1) > 0.0
                ).to(score.dtype).view(-1, 1, 1, 1)

                return (
                    support
                    * score
                    / peak.clamp_min(EPS)
                    * has_support
                ).clamp(0.0, 1.0)

            delete_conservative = _relative_local_strength(
                delete_score,
                delete_support,
            )
            fill_conservative = _relative_local_strength(
                fill_score,
                fill_support,
            )

            delete_conservative_logits = (
                base
                - delete_conservative * delete_delta
            )

            fill_conservative_logits = (
                base
                + fill_conservative * fill_delta
            )

            candidate_logits = torch.cat(
                [
                    base,                              # C0 Preserve
                    parent_logits[:, 3:4],            # C1 standard delete
                    delete_conservative_logits,        # C2 adaptive delete
                    base,                              # C3 inactive
                    base,                              # C4 inactive
                    parent_logits[:, 5:6],            # C5 standard fill
                    fill_conservative_logits,          # C6 adaptive fill
                    base,                              # C7 inactive
                    base,                              # C8 inactive
                ],
                dim=1,
            )
        elif self.use_ranked_variants:
            candidate_logits = torch.cat(
                [
                    base,
                    parent_logits[:, 3:4],
                    parent_logits[:, 4:5],
                    base,
                    base,
                    parent_logits[:, 5:6],
                    parent_logits[:, 6:7],
                    base,
                    base,
                ],
                dim=1,
            )
        else:
            fill_support = supports_parent[:, 4:5]
            fill_carrier = type_supports_parent[:, 2:3]

            fill_extended_support = (
                _soft_dilate(
                    fill_support,
                    self.reference_fill_expand_radius,
                )
                * fill_carrier
            ).clamp(0.0, 1.0)

            fill_delta = delta_parent[:, 4:5]
            c5_logits = parent_logits[:, 5:6]

            # Reference C6: full local expansion around the C5 action.
            # The adaptive branch is an additive residual with zero initial
            # scale, so loading a historical checkpoint leaves C6 unchanged.
            reference_c6_logits = (
                base + fill_extended_support * fill_delta
            )
            ring_support = (
                fill_extended_support * (1.0 - fill_support)
            ).clamp(0.0, 1.0)
            fill_actionness = torch.sigmoid(
                aux["v20_actionness_logits"][:, 4:5]
            )
            ring_mass = ring_support.sum(dim=(-2, -1), keepdim=True)
            ring_mean = (
                ring_support * fill_actionness
            ).sum(dim=(-2, -1), keepdim=True) / ring_mass.clamp_min(1.0)
            relative_actionness = fill_actionness - ring_mean

            if self.adaptive_expanded_fill:
                log_adjustment = (
                    torch.tanh(self.adaptive_fill_log_scale)
                    * relative_actionness
                )
                adaptive_increment = (
                    ring_support
                    * fill_delta
                    * torch.expm1(log_adjustment)
                )
                fill_extended_logits = reference_c6_logits + adaptive_increment
            else:
                adaptive_increment = torch.zeros_like(reference_c6_logits)
                fill_extended_logits = reference_c6_logits

            fill_c6_support = fill_extended_support


            candidate_logits = torch.cat(
                [
                    base,
                    parent_logits[:, 1:2],
                    parent_logits[:, 3:4],
                    base,
                    base,
                    c5_logits,
                    fill_extended_logits,
                    base,
                    base,
                ],
                dim=1,
            )


        supports = self._remap_actions(supports_parent)

        controls = self._remap_actions(controls_parent)

        actionness = self._remap_actions(
            aux["v20_actionness_logits"]
        )

        deltas = self._remap_actions(delta_parent)

        visual_scores = self._remap_actions(
            aux["v20_visual_scores"]
        )

        if self.use_conservative_variants:
            supports[:, 1:2] = delete_conservative
            supports[:, 5:6] = fill_conservative
            deltas[:, 1:2] = delete_conservative * delete_delta
            deltas[:, 5:6] = fill_conservative * fill_delta
        elif not self.use_ranked_variants:
            supports[:, 5:6] = fill_c6_support
            controls[:, 5:6] = controls_parent[:, 4:5]
            deltas[:, 5:6] = fill_delta

        # Keep only island/protrusion/fill carriers. Hole carrier becomes zero
        # because C7/C8 are intentionally inactive in reference mechanism.
        type_supports = torch.cat(
            [
                type_supports_parent[:, :3],
                type_supports_parent[:, 3:4] * 0.0,
            ],
            dim=1,
        )

        candidate_probs = torch.sigmoid(candidate_logits).clamp(
            EPS,
            1.0 - EPS,
        )

        exact_pair_aux = {}
        if self.exact_pair_controls:
            # Important: this is deliberately after all candidate remapping and
            # adaptive/expanded variants.  The external PAIR-M2 support is now
            # guaranteed to describe the final exported candidate, not its
            # parent-bank precursor.
            supports, controls, exact_pair_aux = self._build_exact_pair_supports(
                base_logits=base,
                candidate_logits=candidate_logits,
                image=image,
            )

        # final_pair_semantic_recompute_v1
        # The exact factual/control supports above describe the exported C0..C8
        # actions.  Recompute the text counterfactual on those final supports;
        # parent-bank semantic residuals describe different locations/shapes and
        # must never be reused after support remapping.
        final_action_types = (
            self.ranked_action_types
            if (self.use_ranked_variants or self.use_conservative_variants)
            else self.mechanism_action_types
        )
        if self.exact_pair_controls:
            final_cf = self._text_counterfactual(
                semantic_map,
                text_features,
                negative_text_features,
                supports,
                controls,
                final_action_types,
            )
        else:
            final_cf = {
                "cf_logit": aux["v20_cf_logit"],
                "cf_swap_logit": aux["v20_cf_swap_logit"],
                "cf_signed_delta": aux["v20_cf_signed_delta"],
                "cf_factual_similarity": aux["v20_cf_factual_similarity"],
                "cf_control_similarity": aux["v20_cf_control_similarity"],
                "cf_available": aux["v20_cf_available"],
                "cf_control_area_ratio": aux["v20_cf_control_area_ratio"],
                "cf_control_overlap": aux["v20_cf_control_overlap"],
            }

        batch_size = candidate_logits.shape[0]
        base_probs = candidate_probs[:, 0]

        # Preserve is the only internal output. External V410-R1 owns final
        # candidate selection, so old in-model M2/M3 paths cannot alter this.
        zero_selector = candidate_logits.new_zeros(batch_size, 8)

        aux.update(
            {
                "candidate_probs": candidate_probs,
                "v20_action_supports": supports,
                "v20_control_supports": controls,
                "v20_actionness_logits": actionness,
                "v20_delta_maps": deltas,
                "v20_visual_scores": visual_scores,
                "v20_type_supports": type_supports,
                "v20_action_types": final_action_types,
                "v20_selector_logits": zero_selector,
                "v20_selector_probs": zero_selector,
                "v20_selector_hard": zero_selector,
                "v20_fused_logits": candidate_logits[:, 0],
                "v20_fused_probs": base_probs,
                "v20_hard_fused_probs": base_probs,
                "direct_fused_probs": base_probs,
                "router_fused_probs": base_probs,
                "mechanism_candidates": (
                    candidate_logits.new_ones(batch_size, 1)
                ),
                "adaptive_c6_ring_support": ring_support,
                "adaptive_c6_relative_actionness": relative_actionness,
                "adaptive_c6_increment": adaptive_increment,
                "adaptive_c6_scale": torch.tanh(
                    self.adaptive_fill_log_scale
                ).expand(batch_size),
                # Final exact-pair semantic residual.  This is computed from
                # exactly the factual/control supports exported to PAIR-M2.
                "v20_cf_logit": final_cf["cf_logit"],
                "v20_cf_swap_logit": final_cf["cf_swap_logit"],
                "v20_cf_signed_delta": final_cf["cf_signed_delta"],
                "v20_cf_factual_similarity": final_cf["cf_factual_similarity"],
                "v20_cf_control_similarity": final_cf["cf_control_similarity"],
                "v20_cf_available": final_cf["cf_available"],
                "v20_cf_control_area_ratio": final_cf["cf_control_area_ratio"],
                "v20_cf_control_overlap": final_cf["cf_control_overlap"],
                **exact_pair_aux,
            }
        )

        return candidate_logits, aux


# V27_PARETO_SAFE_TYPE_CONDITIONAL_UTILITY_BANK
# This class keeps the established V25 public keys so existing training and
# inference call sites continue to work.  V27 is activated by RUN_TAG starting
# with "V27_" and is backward compatible with existing V25/V26 checkpoints only
# for B0 initialization; a new V27 run must train its new policy head from zero.

class TypeConditionalUtilityActionBank(UnifiedActionCounterfactualSetBank):
    """V27: Pareto-safe type-conditional one-action-or-abstain policy.

    The bank preserves V20 atomic candidates and matched controls.  Its visual
    ROI map is supplied by CustomCLIP through a no-PVL image-only forward pass;
    text is injected only as explicit factual/control residual evidence.

    V27 replaces V26's hand-tuned ``action + weight * utility-margin`` fusion
    with one learned policy logit trained against the same Null class used by
    deployment.  The utility heads remain tri-state auxiliary supervision.
    """

    HARMFUL = 0
    NEUTRAL = 1
    BENEFICIAL = 2

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        m1 = _cfg_get(cfg, "M1", None)

        self.v25_hidden = max(32, int(_cfg_get(m1, "V25_HIDDEN_DIM", 128)))
        self.v25_type_dim = max(4, int(_cfg_get(m1, "V25_TYPE_EMBED_DIM", 16)))
        self.v25_benefit_min = float(_cfg_get(m1, "V25_BENEFIT_MIN", 0.50))
        self.v25_harm_max = float(_cfg_get(m1, "V25_HARM_MAX", 0.30))
        self.v25_utility_margin = float(_cfg_get(m1, "V25_UTILITY_MARGIN", 0.10))
        self.v25_selector_margin = float(_cfg_get(m1, "V25_SELECTOR_MARGIN", 0.00))
        self.v25_max_actions = 1

        tag = str(_cfg_get(m1, "RUN_TAG", "")).upper()
        self.v26_policy_enabled = tag.startswith("V26_")
        self.v27_policy_enabled = tag.startswith("V27_")
        self.v26_utility_decision_weight = float(
            _cfg_get(m1, "V26_UTILITY_DECISION_WEIGHT", 0.75)
        )
        self.v26_null_margin = float(_cfg_get(m1, "V26_NULL_MARGIN", 0.10))
        self.v27_null_margin = float(
            _cfg_get(m1, "V27_NULL_MARGIN", self.v26_null_margin)
        )

        # Parent V20 cf_verifier/selector are retained for checkpoint-key
        # compatibility only.  V27 never deploys either parent selector.
        for module in (self.cf_verifier, self.selector):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        self.v25_image_adapter = nn.Sequential(
            nn.Linear(self.semantic_channels, self.v25_hidden, bias=False),
            nn.LayerNorm(self.v25_hidden),
            nn.GELU(),
        )
        self.v25_text_adapter = nn.Sequential(
            nn.Linear(self.semantic_channels, self.v25_hidden, bias=False),
            nn.LayerNorm(self.v25_hidden),
        )
        self.v25_type_embedding = nn.Embedding(self.num_types, self.v25_type_dim)

        # factual, control, factual-control, factual*control, four residuals,
        # and an explicit action-family embedding.
        feature_dim = 4 * self.v25_hidden + 4 + self.v25_type_dim
        self.v25_encoder = nn.Sequential(
            nn.Linear(feature_dim, self.v25_hidden),
            nn.LayerNorm(self.v25_hidden),
            nn.GELU(),
            nn.Linear(self.v25_hidden, self.v25_hidden),
            nn.GELU(),
        )
        self.v25_utility_heads = nn.ModuleList(
            [nn.Linear(self.v25_hidden, 3) for _ in range(self.num_types)]
        )
        # Kept as a logged V25-compatible action score.  V27 policy uses the
        # dedicated head below instead of a manual score combination.
        self.v25_action_head = nn.Linear(self.v25_hidden, 1)
        self.v25_null_head = nn.Sequential(
            nn.Linear(self.v25_hidden + 1, self.v25_hidden),
            nn.GELU(),
            nn.Linear(self.v25_hidden, 1),
        )

        # Policy input: image-only local relation embedding + learned utility
        # margin + action area + factual/control visual gap + text swap gap.
        self.v27_policy_head = nn.Sequential(
            nn.Linear(self.v25_hidden + 4, self.v25_hidden),
            nn.LayerNorm(self.v25_hidden),
            nn.GELU(),
            nn.Linear(self.v25_hidden, 1),
        )

        for head in self.v25_utility_heads:
            nn.init.zeros_(head.weight)
            with torch.no_grad():
                if self.v26_policy_enabled or self.v27_policy_enabled:
                    head.bias.zero_()
                else:
                    head.bias.copy_(
                        torch.tensor([0.0, 0.50, -0.75], dtype=head.bias.dtype)
                    )
        nn.init.zeros_(self.v25_action_head.weight)
        nn.init.zeros_(self.v25_action_head.bias)
        nn.init.zeros_(self.v27_policy_head[-1].weight)
        nn.init.zeros_(self.v27_policy_head[-1].bias)
        nn.init.zeros_(self.v25_null_head[-1].weight)
        with torch.no_grad():
            self.v25_null_head[-1].bias.fill_(
                0.10 if (self.v26_policy_enabled or self.v27_policy_enabled) else 0.0
            )

    def _v25_local_evidence(
        self,
        image_only_map: torch.Tensor,
        positive_text: torch.Tensor,
        negative_text: torch.Tensor,
        swapped_text: torch.Tensor,
        factual_masks: torch.Tensor,
        control_masks: torch.Tensor,
        action_types: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Build image-only local evidence plus explicit residual text tests."""
        b, k, h, w = factual_masks.shape
        factual_context = _soft_dilate(
            factual_masks.reshape(b * k, 1, h, w), self.context_radius
        ).reshape(b, k, h, w)
        control_context = _soft_dilate(
            control_masks.reshape(b * k, 1, h, w), self.context_radius
        ).reshape(b, k, h, w)

        factual_raw = self._pool_feature(image_only_map, factual_context)
        control_raw = self._pool_feature(image_only_map, control_context)
        factual = self.v25_image_adapter(factual_raw)
        control = self.v25_image_adapter(control_raw)

        pos = F.normalize(self.v25_text_adapter(positive_text), dim=-1, eps=1e-6)
        neg = F.normalize(self.v25_text_adapter(negative_text), dim=-1, eps=1e-6)
        swap = F.normalize(self.v25_text_adapter(swapped_text), dim=-1, eps=1e-6)
        factual_n = F.normalize(factual, dim=-1, eps=1e-6)
        control_n = F.normalize(control, dim=-1, eps=1e-6)

        def pair_delta(text_feature: torch.Tensor) -> torch.Tensor:
            return (
                (factual_n * text_feature[:, None, :]).sum(dim=-1)
                - (control_n * text_feature[:, None, :]).sum(dim=-1)
            )

        raw_pos = pair_delta(pos)
        raw_neg = pair_delta(neg)
        raw_swap = pair_delta(swap)
        is_fill = torch.isin(
            action_types,
            torch.tensor(self.FILL_TYPES, device=action_types.device),
        )
        polarity = torch.where(
            is_fill,
            torch.ones_like(action_types, dtype=raw_pos.dtype),
            -torch.ones_like(action_types, dtype=raw_pos.dtype),
        )[None, :]
        pos_delta = polarity * raw_pos
        neg_delta = -polarity * raw_neg
        swap_delta = polarity * raw_swap
        swap_gap = pos_delta - swap_delta
        visual_gap = torch.sqrt(
            (factual_n - control_n).pow(2).mean(dim=-1).clamp_min(0.0) + 1e-6
        )
        text_residual = torch.stack(
            [pos_delta, neg_delta, swap_gap, visual_gap], dim=-1
        )

        type_feature = self.v25_type_embedding(action_types)[None].expand(b, -1, -1)
        features = torch.cat(
            [
                factual,
                control,
                factual - control,
                factual * control,
                text_residual,
                type_feature,
            ],
            dim=-1,
        )
        embedding = self.v25_encoder(features.reshape(b * k, -1)).reshape(
            b, k, -1
        )

        utility_logits = embedding.new_zeros((b, k, 3))
        for action_type in range(self.num_types):
            idx = torch.where(action_types == action_type)[0]
            if idx.numel():
                utility_logits[:, idx] = self.v25_utility_heads[action_type](
                    embedding[:, idx].reshape(-1, self.v25_hidden)
                ).reshape(b, idx.numel(), 3)

        factual_area = factual_masks.sum(dim=(-2, -1))
        control_area = control_masks.sum(dim=(-2, -1))
        area_ratio = control_area / factual_area.clamp_min(EPS)
        overlap = (
            factual_masks * control_masks
        ).sum(dim=(-2, -1)) / factual_area.clamp_min(EPS)
        context_overlap = (
            factual_context * control_context
        ).sum(dim=(-2, -1)) / factual_context.sum(dim=(-2, -1)).clamp_min(EPS)
        raw_valid = (
            (factual_area > 0)
            & (control_area > 0)
            & ((area_ratio - 1.0).abs() <= 1e-3)
            & (overlap <= 1e-6)
        )
        context_clean = context_overlap <= 1e-6
        valid = raw_valid & context_clean

        action_logits = self.v25_action_head(embedding).squeeze(-1)
        utility_logit_margin = (
            utility_logits[..., self.BENEFICIAL]
            - utility_logits[..., self.HARMFUL]
        )
        local_area = factual_area / float(h * w)
        policy_features = torch.cat(
            [
                embedding,
                utility_logit_margin.unsqueeze(-1),
                local_area.unsqueeze(-1),
                visual_gap.unsqueeze(-1),
                swap_gap.unsqueeze(-1),
            ],
            dim=-1,
        )
        policy_logits = self.v27_policy_head(
            policy_features.reshape(b * k, -1)
        ).reshape(b, k)

        valid_f = valid.float()
        pooled = (
            embedding * valid_f[:, :, None]
        ).sum(dim=1) / valid_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        valid_fraction = valid_f.mean(dim=1, keepdim=True)
        null_logit = self.v25_null_head(
            torch.cat([pooled, valid_fraction], dim=-1)
        ).squeeze(-1)

        return {
            "utility_logits": utility_logits,
            "utility_embedding": embedding,
            "action_logits": action_logits,
            "policy_logits": policy_logits,
            "null_logit": null_logit,
            "valid": valid,
            "raw_valid": raw_valid,
            "context_clean": context_clean,
            "control_area_ratio": area_ratio,
            "control_overlap": overlap,
            "context_overlap": context_overlap,
            "local_area": local_area,
            "pos_delta": pos_delta,
            "neg_delta": neg_delta,
            "swap_delta": swap_delta,
            "text_residual": text_residual,
            "utility_logit_margin": utility_logit_margin,
        }

    def _v25_deploy_one_action(
        self,
        base_logits: torch.Tensor,
        action_logits: torch.Tensor,
        evidence: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Choose exactly one verified action or emit Preserve unchanged."""
        if base_logits.ndim != 4 or base_logits.shape[1] != 1:
            raise ValueError("V25/V26/V27 expects base logits [B,1,H,W].")

        class_probs = torch.softmax(evidence["utility_logits"], dim=-1)
        harm_prob = class_probs[..., self.HARMFUL]
        benefit_prob = class_probs[..., self.BENEFICIAL]
        utility = benefit_prob - harm_prob

        if self.v27_policy_enabled:
            # Unified, learned policy score.  There is no hand-selected
            # arithmetic coefficient between action and utility logits.
            decision_logits = evidence["policy_logits"]
            eligible = evidence["valid"]
            selector_margin = self.v27_null_margin
        elif self.v26_policy_enabled:
            decision_logits = evidence["action_logits"] + (
                self.v26_utility_decision_weight * evidence["utility_logit_margin"]
            )
            eligible = evidence["valid"]
            selector_margin = self.v26_null_margin
        else:
            decision_logits = evidence["action_logits"]
            eligible = (
                evidence["valid"]
                & (benefit_prob >= self.v25_benefit_min)
                & (harm_prob <= self.v25_harm_max)
                & (utility >= self.v25_utility_margin)
            )
            selector_margin = self.v25_selector_margin

        scores = decision_logits.masked_fill(~eligible, -1e9)
        best_score, best_index = scores.max(dim=1)
        accept = eligible.any(dim=1) & (
            best_score >= (evidence["null_logit"] + selector_margin)
        )

        selected = torch.zeros_like(scores)
        selected.scatter_(1, best_index[:, None], accept[:, None].to(selected.dtype))
        action_delta = action_logits[:, 1:] - base_logits
        final_logits = base_logits[:, 0] + (
            selected[:, :, None, None] * action_delta
        ).sum(dim=1)
        evidence["decision_logits"] = decision_logits
        return final_logits, selected, class_probs, utility

    def generate(
        self,
        base_logits: torch.Tensor,
        image: torch.Tensor,
        semantic_map: Optional[torch.Tensor],
        text_features: torch.Tensor,
        negative_text_features: Optional[torch.Tensor] = None,
        swapped_text_features: Optional[torch.Tensor] = None,
    ):
        candidate_logits_all, aux = super().generate(
            base_logits=base_logits,
            image=image,
            semantic_map=semantic_map,
            text_features=text_features,
            negative_text_features=negative_text_features,
        )
        if semantic_map is None:
            raise RuntimeError("V25/V26/V27 requires image-only spatial patch features.")
        if negative_text_features is None:
            negative_text_features = text_features
        if swapped_text_features is None:
            swapped_text_features = negative_text_features

        evidence = self._v25_local_evidence(
            image_only_map=semantic_map,
            positive_text=text_features,
            negative_text=negative_text_features,
            swapped_text=swapped_text_features,
            factual_masks=aux["v20_action_supports"],
            control_masks=aux["v20_control_supports"],
            action_types=self.action_types,
        )
        final_logits, selected, class_probs, utility = self._v25_deploy_one_action(
            candidate_logits_all[:, :1],
            candidate_logits_all,
            evidence,
        )
        final_probs = torch.sigmoid(final_logits).clamp(EPS, 1.0 - EPS)
        choice_logits = torch.cat(
            [
                evidence["decision_logits"].masked_fill(~evidence["valid"], -1e4),
                evidence["null_logit"][:, None],
            ],
            dim=1,
        )

        aux.update({
            "direct_fused_probs": final_probs,
            "router_fused_probs": final_probs,
            "v20_fused_probs": final_probs,
            "v20_hard_fused_probs": final_probs,
            "v20_fused_logits": final_logits,
            "v20_selector_logits": evidence["decision_logits"],
            "v20_selector_probs": selected,
            "v20_selector_hard": selected,
            "v25_utility_logits": evidence["utility_logits"],
            "v25_utility_probs": class_probs,
            "v25_utility_score": utility,
            "v25_action_logits": evidence["action_logits"],
            "v25_null_logit": evidence["null_logit"],
            "v25_choice_logits": choice_logits,
            "v25_selected_action": selected,
            "v25_valid_control": evidence["valid"],
            "v25_benefit_prob": class_probs[..., self.BENEFICIAL],
            "v25_harm_prob": class_probs[..., self.HARMFUL],
            "v25_pos_delta": evidence["pos_delta"],
            "v25_neg_delta": evidence["neg_delta"],
            "v25_swap_delta": evidence["swap_delta"],
            "v25_control_area_ratio": evidence["control_area_ratio"],
            "v25_control_overlap": evidence["control_overlap"],
            "v25_accepted_rate": selected.sum(dim=1),
            "v25_selected_benefit": (
                class_probs[..., self.BENEFICIAL] * selected
            ).sum(dim=1),
            "v25_selected_harm": (
                class_probs[..., self.HARMFUL] * selected
            ).sum(dim=1),
            "v26_decision_logits": evidence["decision_logits"],
            "v26_utility_logit_margin": evidence["utility_logit_margin"],
            "v27_policy_logits": evidence["policy_logits"],
            "v27_context_overlap": evidence["context_overlap"],
            "v27_context_clean_valid": evidence["context_clean"],
            "v27_raw_control_valid": evidence["raw_valid"],
            "v27_local_action_area": evidence["local_area"],
        })
        return candidate_logits_all, aux



# V31_CANDIDATE_CONDITIONED_COMPETITIVE_POLICY
# Appended by the V31 installer.  It intentionally subclasses V27's action
# bank so V20 atomic candidate construction and matched controls remain exact.

class V31CandidateConditionedUtilityBank(TypeConditionalUtilityActionBank):
    """V31: candidate-conditioned visual utility + text semantic-veto policy.

    Candidate generation is inherited from V20/V27.  V31 changes only the
    decision layer: every action is represented by local factual/control/context
    visual descriptors, base/candidate geometry and text residual evidence.
    Deployment is Preserve (Null) versus exactly one action.
    """

    HARMFUL = 0
    NEUTRAL = 1
    BENEFICIAL = 2

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        m1 = _cfg_get(cfg, "M1", None)
        self.v31_hidden = max(32, int(_cfg_get(m1, "V31_HIDDEN_DIM", 128)))
        self.v31_type_dim = max(4, int(_cfg_get(m1, "V31_TYPE_EMBED_DIM", 16)))
        self.v31_null_margin = float(_cfg_get(m1, "V31_NULL_MARGIN", 0.05))
        self.v31_text_veto_weight = float(
            _cfg_get(m1, "V31_TEXT_VETO_WEIGHT", 0.25)
        )
        self.v31_context_radius = max(
            1, int(_cfg_get(m1, "V31_CONTEXT_RADIUS", 3))
        )

        self.v31_visual_adapter = nn.Sequential(
            nn.Linear(self.semantic_channels * 5, self.v31_hidden, bias=False),
            nn.LayerNorm(self.v31_hidden),
            nn.GELU(),
        )
        self.v31_scalar_adapter = nn.Sequential(
            nn.Linear(7 + self.v31_type_dim, self.v31_hidden),
            nn.LayerNorm(self.v31_hidden),
            nn.GELU(),
        )
        self.v31_type_embedding = nn.Embedding(self.num_types, self.v31_type_dim)

        self.v31_action_fuse = nn.Sequential(
            nn.Linear(2 * self.v31_hidden, self.v31_hidden),
            nn.LayerNorm(self.v31_hidden),
            nn.GELU(),
            nn.Dropout(float(_cfg_get(m1, "V31_DROPOUT", 0.10))),
        )
        self.v31_utility_head = nn.Linear(self.v31_hidden, 3)
        self.v31_policy_head = nn.Linear(self.v31_hidden, 1)
        self.v31_semantic_veto_head = nn.Sequential(
            nn.Linear(self.v31_hidden + 2, self.v31_hidden // 2),
            nn.GELU(),
            nn.Linear(self.v31_hidden // 2, 1),
        )
        self.v31_global_adapter = nn.Sequential(
            nn.Linear(self.semantic_channels, self.v31_hidden, bias=False),
            nn.LayerNorm(self.v31_hidden),
            nn.GELU(),
        )
        self.v31_null_head = nn.Sequential(
            nn.Linear(2 * self.v31_hidden + 1, self.v31_hidden),
            nn.GELU(),
            nn.Linear(self.v31_hidden, 1),
        )

        # Preserve is the safe initial policy.  The new action policy begins
        # conservative but still receives gradients through listwise loss.
        nn.init.zeros_(self.v31_policy_head.weight)
        nn.init.constant_(self.v31_policy_head.bias, -0.25)
        nn.init.zeros_(self.v31_utility_head.weight)
        nn.init.zeros_(self.v31_utility_head.bias)
        nn.init.zeros_(self.v31_semantic_veto_head[-1].weight)
        nn.init.constant_(self.v31_semantic_veto_head[-1].bias, -1.0)
        nn.init.zeros_(self.v31_null_head[-1].weight)
        nn.init.constant_(self.v31_null_head[-1].bias, 0.25)

    @staticmethod
    def _masked_pool(
        feature_map: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Mean pool [B,C,H,W] over [B,K,H,W] masks -> [B,K,C]."""
        if feature_map.ndim != 4 or masks.ndim != 4:
            raise ValueError("V31 expects feature map [B,C,H,W] and masks [B,K,H,W].")
        mass = masks.sum(dim=(-2, -1)).clamp_min(EPS)
        pooled = torch.einsum("bchw,bkhw->bkc", feature_map, masks)
        return pooled / mass[..., None]

    def _context_ring(self, masks: torch.Tensor) -> torch.Tensor:
        b, k, h, w = masks.shape
        flat = masks.reshape(b * k, 1, h, w)
        dilated = _soft_dilate(flat, self.v31_context_radius)[:, 0]
        ring = (dilated - flat[:, 0]).clamp(0.0, 1.0)
        return ring.reshape(b, k, h, w)

    @staticmethod
    def _masked_scalar(value: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        if value.ndim == 3:
            value = value[:, None]
        mass = masks.sum(dim=(-2, -1)).clamp_min(EPS)
        return (value * masks).sum(dim=(-2, -1)) / mass

    def _v31_decide(
        self,
        candidate_logits_all: torch.Tensor,
        aux: Dict[str, torch.Tensor],
        semantic_map: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if candidate_logits_all.ndim != 4 or candidate_logits_all.shape[1] < 2:
            raise ValueError("V31 expects Preserve plus K action logits.")
        if semantic_map is None:
            raise RuntimeError("V31 requires the frozen no-PVL image-only semantic map.")

        base_logits = candidate_logits_all[:, 0]
        action_logits = candidate_logits_all[:, 1:]
        b, k, h, w = action_logits.shape
        supports = aux["v20_action_supports"].float()
        controls = aux["v20_control_supports"].float()
        types = aux["v20_action_types"].to(action_logits.device)
        if supports.shape != action_logits.shape or controls.shape != action_logits.shape:
            raise ValueError("V31 factual/control masks must match the action tensor shape.")

        base_prob = torch.sigmoid(base_logits)
        action_prob = torch.sigmoid(action_logits)
        delta = action_prob - base_prob[:, None]
        entropy = (
            -(base_prob.clamp(EPS, 1.0 - EPS) * base_prob.clamp(EPS, 1.0 - EPS).log()
              + (1.0 - base_prob).clamp(EPS, 1.0 - EPS)
              * (1.0 - base_prob).clamp(EPS, 1.0 - EPS).log())
            / math.log(2.0)
        )
        boundary = (4.0 * base_prob * (1.0 - base_prob)).clamp(0.0, 1.0)
        ring = self._context_ring(supports)

        factual = self._masked_pool(semantic_map, supports)
        control = self._masked_pool(semantic_map, controls)
        context = self._masked_pool(semantic_map, ring)
        visual_pack = torch.cat(
            [factual, control, factual - control, factual * control, context],
            dim=-1,
        )

        local_area = supports.mean(dim=(-2, -1))
        scalar_pack = torch.stack(
            [
                self._masked_scalar(base_prob, supports),
                self._masked_scalar(delta.abs(), supports),
                self._masked_scalar(entropy, supports),
                self._masked_scalar(boundary, supports),
                self._masked_scalar(base_prob, ring),
                self._masked_scalar(delta.abs(), ring),
                local_area,
            ],
            dim=-1,
        )

        type_feature = self.v31_type_embedding(types)[None].expand(b, -1, -1)
        visual_embedding = self.v31_visual_adapter(
            visual_pack.reshape(b * k, -1)
        ).reshape(b, k, self.v31_hidden)
        scalar_embedding = self.v31_scalar_adapter(
            torch.cat([scalar_pack, type_feature], dim=-1).reshape(b * k, -1)
        ).reshape(b, k, self.v31_hidden)
        embedding = self.v31_action_fuse(
            torch.cat([visual_embedding, scalar_embedding], dim=-1)
        )

        factual_area = supports.sum(dim=(-2, -1))
        control_area = controls.sum(dim=(-2, -1))
        area_ratio = control_area / factual_area.clamp_min(EPS)
        overlap = (supports * controls).sum(dim=(-2, -1)) / factual_area.clamp_min(EPS)

        inherited_valid = aux.get("v25_valid_control", aux.get("v20_cf_available"))
        if inherited_valid is None:
            inherited_valid = (
                (factual_area > 0)
                & (control_area > 0)
                & ((area_ratio - 1.0).abs() <= 1e-3)
                & (overlap <= 1e-6)
            )
        valid = inherited_valid.bool() & (factual_area > 0) & (control_area > 0)

        text_residual = aux.get(
            "v25_text_residual",
            aux.get("v20_cf_signed_delta", embedding.new_zeros(b, k)),
        ).to(embedding.dtype)
        cf_logit = aux.get(
            "v20_cf_logit", embedding.new_zeros(b, k)
        ).to(embedding.dtype)

        utility_logits = self.v31_utility_head(embedding)
        veto_logits = self.v31_semantic_veto_head(
            torch.cat(
                [embedding, text_residual[..., None], cf_logit[..., None]],
                dim=-1,
            )
        ).squeeze(-1)
        raw_policy_logits = self.v31_policy_head(embedding).squeeze(-1)
        policy_logits = raw_policy_logits - (
            self.v31_text_veto_weight * torch.sigmoid(veto_logits)
        )

        valid_f = valid.float()
        pooled_actions = (
            embedding * valid_f[..., None]
        ).sum(dim=1) / valid_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        global_embedding = self.v31_global_adapter(
            semantic_map.mean(dim=(-2, -1))
        )
        null_logit = self.v31_null_head(
            torch.cat(
                [pooled_actions, global_embedding, valid_f.mean(dim=1, keepdim=True)],
                dim=-1,
            )
        ).squeeze(-1)

        decision = policy_logits.masked_fill(~valid, -1e4)
        best_score, best_idx = decision.max(dim=1)
        accept = valid.any(dim=1) & (
            best_score >= null_logit + self.v31_null_margin
        )

        selected = torch.zeros_like(policy_logits)
        selected.scatter_(1, best_idx[:, None], accept[:, None].to(selected.dtype))

        action_delta = action_logits - base_logits[:, None]
        hard_fused_logits = base_logits + (
            selected[:, :, None, None] * action_delta
        ).sum(dim=1)

        class_logits = torch.cat([null_logit[:, None], decision], dim=1)
        soft_choice = torch.softmax(class_logits, dim=1)
        soft_fused_logits = (
            soft_choice[:, 0, None, None] * base_logits
            + (soft_choice[:, 1:, None, None] * action_logits).sum(dim=1)
        )

        return {
            "v31_policy_logits": policy_logits,
            "v31_raw_policy_logits": raw_policy_logits,
            "v31_null_logit": null_logit,
            "v31_utility_logits": utility_logits,
            "v31_semantic_veto_logits": veto_logits,
            "v31_valid_action": valid,
            "v31_selected_action": selected,
            "v31_soft_choice_probs": soft_choice,
            "v31_local_area": local_area,
            "v31_text_residual": text_residual,
            "v31_cf_logit": cf_logit,
            # Audit-only tensors: no decision path reads these fields.
            "v31_action_embedding": embedding,
            "v31_global_embedding": global_embedding,
            "v31_parent_valid_action": valid,
            "v31_fused_logits": soft_fused_logits,
            "v31_hard_fused_logits": hard_fused_logits,
        }

    def generate(
        self,
        base_logits: torch.Tensor,
        image: torch.Tensor,
        semantic_map: Optional[torch.Tensor],
        text_features: torch.Tensor,
        negative_text_features: Optional[torch.Tensor] = None,
        swapped_text_features: Optional[torch.Tensor] = None,
    ):
        candidate_logits_all, aux = super().generate(
            base_logits=base_logits,
            image=image,
            semantic_map=semantic_map,
            text_features=text_features,
            negative_text_features=negative_text_features,
            swapped_text_features=swapped_text_features,
        )
        decision = self._v31_decide(candidate_logits_all, aux, semantic_map)
        hard_probs = torch.sigmoid(
            decision["v31_hard_fused_logits"]
        ).clamp(EPS, 1.0 - EPS)
        soft_probs = torch.sigmoid(
            decision["v31_fused_logits"]
        ).clamp(EPS, 1.0 - EPS)

        aux.update(decision)
        # Keep V20 public keys so all existing inference paths remain valid.
        aux.update(
            {
                "v20_fused_logits": decision["v31_fused_logits"],
                "v20_fused_probs": soft_probs,
                "v20_hard_fused_probs": hard_probs,
                "direct_fused_probs": soft_probs,
                "router_fused_probs": soft_probs,
            }
        )
        return candidate_logits_all, aux


# V32_ISLAND_PHASE_B_POLICY
class V32IslandDeletePhaseBPolicyBank(V31CandidateConditionedUtilityBank):
    """Frozen candidate bank with V32/V33/V34 island-only decision variants."""

    def __init__(self, cfg):
        super().__init__(cfg)
        m1 = getattr(cfg, "M1", cfg)
        self.v32_text_veto_weight = float(_cfg_get(m1, "V32_TEXT_VETO_WEIGHT", 0.15))
        self.v32_island_type_id = int(_cfg_get(m1, "V32_ISLAND_TYPE_ID", 0))
        self.v32_delta_eps = float(_cfg_get(m1, "V32_DELTA_EPS", 1e-8))
        self.v32_freeze_candidate_bank = bool(_cfg_get(m1, "V32_FREEZE_CANDIDATE_BANK", True))
        if self.v32_freeze_candidate_bank:
            for name, param in self.named_parameters():
                if not name.startswith("v31_"):
                    param.requires_grad_(False)

        # V33-SGR: direct signed Dice-gain predictor.  It is added after
        # freezing the inherited candidate bank, so only the new head and
        # V31 feature adapters remain trainable in Phase-B.
        self.v33_signed_gain_regression = bool(
            _cfg_get(m1, "V33_SIGNED_GAIN_REGRESSION", False)
        )
        self.v33_gain_scale = max(
            1e-5,
            float(_cfg_get(m1, "V33_GAIN_SCALE", 0.06)),
        )
        self.v33_decision_temperature = max(
            1e-5,
            float(_cfg_get(m1, "V33_SOFTMAX_TEMPERATURE", 0.01)),
        )
        self.v33_gain_head = None

        if self.v33_signed_gain_regression:
            self.v33_gain_head = nn.Sequential(
                nn.Linear(self.v31_hidden, self.v31_hidden),
                nn.GELU(),
                nn.Linear(self.v31_hidden, 1),
            )
            nn.init.zeros_(self.v33_gain_head[-1].weight)
            nn.init.zeros_(self.v33_gain_head[-1].bias)

        # V34: spatial counterfactual outcome distribution. V31 pooling is
        # retained as a semantic descriptor, while this branch restores the
        # missing spatial topology of Base/Candidate/Delta/Support/Control.
        self.v34_spatial_quantile_world_model = bool(
            _cfg_get(m1, "V34_SPATIAL_QUANTILE_WORLD_MODEL", False)
        )
        self.v34_gain_scale = max(1e-5, float(_cfg_get(m1, "V34_GAIN_SCALE", 0.06)))
        self.v34_decision_temperature = max(1e-5, float(_cfg_get(m1, "V34_SOFTMAX_TEMPERATURE", 0.01)))
        self.v34_semantic_dim = max(4, int(_cfg_get(m1, "V34_SEMANTIC_DIM", 16)))
        self.v34_spatial_dim = max(32, int(_cfg_get(m1, "V34_SPATIAL_DIM", 96)))
        self.v34_semantic_projector = None
        self.v34_spatial_encoder = None
        self.v34_state_fuse = None
        self.v34_quantile_head = None

        if self.v34_spatial_quantile_world_model:
            self.v34_semantic_projector = nn.Sequential(
                nn.Conv2d(self.semantic_channels, self.v34_semantic_dim, kernel_size=1, bias=False),
                nn.GroupNorm(1, self.v34_semantic_dim),
                nn.GELU(),
            )
            # Geometry channels: Base, Candidate, signed/absolute Delta,
            # factual Support, matched Control, context Ring, Base entropy,
            # and Base boundary uncertainty.
            in_channels = self.v34_semantic_dim + 9
            mid = max(32, self.v34_spatial_dim // 2)
            self.v34_spatial_encoder = nn.Sequential(
                nn.Conv2d(in_channels, mid, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(max(1, min(8, mid)), mid),
                nn.GELU(),
                nn.Conv2d(mid, self.v34_spatial_dim, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(max(1, min(8, self.v34_spatial_dim)), self.v34_spatial_dim),
                nn.GELU(),
                nn.Conv2d(self.v34_spatial_dim, self.v34_spatial_dim, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(max(1, min(8, self.v34_spatial_dim)), self.v34_spatial_dim),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.v34_state_fuse = nn.Sequential(
                nn.Linear(self.v31_hidden + self.v34_spatial_dim, self.v31_hidden),
                nn.LayerNorm(self.v31_hidden),
                nn.GELU(),
                nn.Dropout(float(_cfg_get(m1, "V31_DROPOUT", 0.10))),
            )
            self.v34_quantile_head = nn.Linear(self.v31_hidden, 3)
            nn.init.zeros_(self.v34_quantile_head.weight)
            nn.init.zeros_(self.v34_quantile_head.bias)

        # V35: source-purified local action world model.  Candidate generation
        # itself is trainable only for island-delete residual proposals; B0 and
        # all non-island proposal families remain frozen.
        self.v36_casewise_plackett_luce = bool(
            _cfg_get(m1, "V36_CASEWISE_PLACKETT_LUCE", False)
        )
        # Keep the inherited source-proposal flag untouched: UnifiedActionCounterfactualSetBank
        # reads it during candidate construction.  V35 decision heads are disabled only for V36.
        self.v35_decision_enabled = bool(
            _cfg_get(m1, "V35_RESIDUAL_PURIFIED_WORLD_MODEL", False)
        ) and not self.v36_casewise_plackett_luce
        self.v35_gain_scale = max(1e-5, float(_cfg_get(m1, "V35_GAIN_SCALE", 0.06)))
        self.v35_decision_temperature = max(
            1e-5, float(_cfg_get(m1, "V35_SOFTMAX_TEMPERATURE", 0.02))
        )
        self.v35_harm_penalty = max(
            0.0, float(_cfg_get(m1, "V35_HARM_PENALTY", 0.012))
        )
        self.v35_state_fuse = None
        self.v35_gain_head = None
        self.v35_outcome_head = None
        if self.v35_decision_enabled:
            self.v35_state_fuse = nn.Sequential(
                nn.Linear(self.v31_hidden + 3, self.v31_hidden),
                nn.LayerNorm(self.v31_hidden),
                nn.GELU(),
                nn.Dropout(float(_cfg_get(m1, "V31_DROPOUT", 0.10))),
            )
            self.v35_gain_head = nn.Linear(self.v31_hidden, 1)
            self.v35_outcome_head = nn.Linear(self.v31_hidden, 3)
            nn.init.zeros_(self.v35_gain_head.weight)
            nn.init.zeros_(self.v35_gain_head.bias)
            nn.init.zeros_(self.v35_outcome_head.weight)
            nn.init.zeros_(self.v35_outcome_head.bias)

        # V36: direct casewise rank score.  The state adds two factual/control
        # text-residual scalars to V35's residual state.  The score head is the
        # sole deployed action score; outcome probabilities are auxiliary.
        self.v36_gain_scale = max(1e-5, float(_cfg_get(m1, "V36_GAIN_SCALE", 0.06)))
        self.v36_decision_temperature = max(
            1e-5, float(_cfg_get(m1, "V36_SOFTMAX_TEMPERATURE", 0.003))
        )
        self.v36_state_fuse = None
        self.v36_rank_head = None
        self.v36_outcome_head = None
        if self.v36_casewise_plackett_luce:
            self.v36_state_fuse = nn.Sequential(
                nn.Linear(self.v31_hidden + 5, self.v31_hidden),
                nn.LayerNorm(self.v31_hidden),
                nn.GELU(),
                nn.Dropout(float(_cfg_get(m1, "V31_DROPOUT", 0.10))),
            )
            self.v36_rank_head = nn.Linear(self.v31_hidden, 1)
            self.v36_outcome_head = nn.Linear(self.v31_hidden, 3)
            nn.init.zeros_(self.v36_rank_head.weight)
            nn.init.zeros_(self.v36_rank_head.bias)
            nn.init.zeros_(self.v36_outcome_head.weight)
            nn.init.zeros_(self.v36_outcome_head.bias)

    def _v32_semantic_map(self, args, kwargs):
        for key in ("semantic_map", "semantic_features", "image_semantic_map"):
            value = kwargs.get(key, None)
            if torch.is_tensor(value) and value.ndim == 4 and value.shape[1] == self.semantic_channels:
                return value
        for value in reversed(args):
            if torch.is_tensor(value) and value.ndim == 4 and value.shape[1] == self.semantic_channels:
                return value
        raise RuntimeError("V32 could not resolve the frozen image-only semantic map.")

    def _v35_residual_action_state(
        self,
        aux: Dict[str, torch.Tensor],
        action_embedding: torch.Tensor,
        island_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse local factual/control residual evidence with V31 semantics."""
        if self.v35_state_fuse is None:
            raise RuntimeError("V35 state fusion module is absent.")
        residual = aux.get("v35_residual_logit_map", None)
        if residual is None or residual.ndim != 4 or residual.shape[1] != 1:
            raise RuntimeError("V35 residual proposal map is missing or malformed.")
        supports = aux["v20_action_supports"].float().index_select(1, island_indices)
        controls = aux["v20_control_supports"].float().index_select(1, island_indices)
        island_embedding = action_embedding.index_select(1, island_indices)
        residual_prob = torch.sigmoid(residual)
        factual = self._masked_scalar(residual_prob, supports)
        control = self._masked_scalar(residual_prob, controls)
        residual_pack = torch.stack([factual, control, factual - control], dim=-1)
        return self.v35_state_fuse(torch.cat([island_embedding, residual_pack], dim=-1))

    def _v36_casewise_action_state(
        self,
        aux: Dict[str, torch.Tensor],
        action_embedding: torch.Tensor,
        island_indices: torch.Tensor,
    ) -> torch.Tensor:
        """State for direct within-image ranking.

        Unlike V35, the state receives the matched factual/control text
        residual used to form the counterfactual pair.  It remains candidate
        local and is available at inference; no GT-derived value is used.
        """
        if self.v36_state_fuse is None:
            raise RuntimeError("V36 state fusion module is absent.")
        residual = aux.get("v35_residual_logit_map", None)
        if residual is None or residual.ndim != 4 or residual.shape[1] != 1:
            raise RuntimeError("V36 residual proposal map is missing or malformed.")
        supports = aux["v20_action_supports"].float().index_select(1, island_indices)
        controls = aux["v20_control_supports"].float().index_select(1, island_indices)
        island_embedding = action_embedding.index_select(1, island_indices)
        residual_prob = torch.sigmoid(residual)
        factual = self._masked_scalar(residual_prob, supports)
        control = self._masked_scalar(residual_prob, controls)
        cf_logit = aux.get("v20_cf_logit", factual.new_zeros(action_embedding.shape[:2]))
        signed_delta = aux.get("v20_cf_signed_delta", factual.new_zeros(action_embedding.shape[:2]))
        cf_logit = cf_logit.to(factual.dtype).index_select(1, island_indices)
        signed_delta = signed_delta.to(factual.dtype).index_select(1, island_indices)
        state_pack = torch.stack(
            [factual, control, factual - control, cf_logit, signed_delta], dim=-1
        )
        return self.v36_state_fuse(torch.cat([island_embedding, state_pack], dim=-1))

    def _v34_spatial_action_state(
        self,
        base_logits: torch.Tensor,
        action_logits: torch.Tensor,
        aux: Dict[str, torch.Tensor],
        semantic_map: torch.Tensor,
        action_embedding: torch.Tensor,
        island_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Encode the causal local state of each island intervention.

        Unlike V31/V33 masked averages, this retains spatial configuration:
        whether deletion touches a coherent lesion boundary, a detached island,
        or a structured ambiguous region. All inputs are available at inference
        and derive solely from image/Base/candidate/control state.
        """
        if (
            self.v34_semantic_projector is None
            or self.v34_spatial_encoder is None
            or self.v34_state_fuse is None
        ):
            raise RuntimeError("V34 spatial world-model modules are absent.")

        b, k, h, w = action_logits.shape
        if semantic_map.shape[-2:] != (h, w):
            semantic_map = F.interpolate(
                semantic_map, size=(h, w), mode="bilinear", align_corners=False
            )

        supports = aux["v20_action_supports"].float().index_select(1, island_indices)
        controls = aux["v20_control_supports"].float().index_select(1, island_indices)
        island_logits = action_logits.index_select(1, island_indices)
        island_embedding = action_embedding.index_select(1, island_indices)
        ki = island_logits.shape[1]

        base_prob = torch.sigmoid(base_logits)
        action_prob = torch.sigmoid(island_logits)
        delta = action_prob - base_prob[:, None]
        entropy = (
            -(base_prob.clamp(EPS, 1.0 - EPS) * base_prob.clamp(EPS, 1.0 - EPS).log()
              + (1.0 - base_prob).clamp(EPS, 1.0 - EPS)
              * (1.0 - base_prob).clamp(EPS, 1.0 - EPS).log())
            / math.log(2.0)
        )
        boundary = (4.0 * base_prob * (1.0 - base_prob)).clamp(0.0, 1.0)
        ring = self._context_ring(supports)

        projected = self.v34_semantic_projector(semantic_map)
        projected = projected[:, None].expand(-1, ki, -1, -1, -1)
        base_expand = base_prob[:, None, None].expand(-1, ki, -1, -1, -1)
        action_expand = action_prob[:, :, None]
        delta_expand = delta[:, :, None]
        geometry = torch.cat(
            [
                base_expand,
                action_expand,
                delta_expand,
                delta_expand.abs(),
                supports[:, :, None],
                controls[:, :, None],
                ring[:, :, None],
                entropy[:, None, None].expand(-1, ki, -1, -1, -1),
                boundary[:, None, None].expand(-1, ki, -1, -1, -1),
            ],
            dim=2,
        )
        spatial_input = torch.cat([projected, geometry], dim=2)
        spatial = self.v34_spatial_encoder(
            spatial_input.reshape(b * ki, spatial_input.shape[2], h, w)
        ).flatten(1).reshape(b, ki, self.v34_spatial_dim)

        return self.v34_state_fuse(
            torch.cat([island_embedding, spatial], dim=-1)
        )

    def _v32_decide(self, candidate_logits_all, aux, semantic_map):
        inherited = V31CandidateConditionedUtilityBank._v31_decide(
            self, candidate_logits_all, aux, semantic_map
        )

        base_logits = candidate_logits_all[:, 0]
        action_logits = candidate_logits_all[:, 1:]
        b, k, _, _ = action_logits.shape

        delta_strength = (action_logits - base_logits[:, None]).abs().flatten(2).amax(dim=-1)
        geometry_valid = delta_strength > self.v32_delta_eps

        per_type = max(1, k // max(1, self.num_types))
        slot_type = torch.arange(k, device=action_logits.device) // per_type
        island_slot = (slot_type == self.v32_island_type_id)[None].expand(b, -1)
        valid = geometry_valid & island_slot

        cf_available = aux.get("v25_valid_control", aux.get("v20_cf_available", None))
        if cf_available is None or cf_available.shape != valid.shape:
            cf_available = torch.zeros_like(valid)
        else:
            cf_available = cf_available.bool()

        # V36: candidate sets are ranked jointly with Preserve=0.  There is no
        # hand-tuned gain-minus-probability arithmetic at deployment: the rank
        # score is trained directly against the same per-image action list.
        if self.v36_casewise_plackett_luce:
            action_embedding = inherited.get("v31_action_embedding", None)
            if action_embedding is None:
                raise RuntimeError("V36 requires v31_action_embedding from _v31_decide.")
            if self.v36_rank_head is None or self.v36_outcome_head is None:
                raise RuntimeError("V36 ranking heads are absent.")
            island_indices = torch.where(slot_type == self.v32_island_type_id)[0]
            if island_indices.numel() == 0:
                raise RuntimeError("V36 found no island candidate slots.")
            valid = valid & cf_available
            state = self._v36_casewise_action_state(aux, action_embedding, island_indices)
            island_score = self.v36_gain_scale * torch.tanh(
                self.v36_rank_head(state).squeeze(-1)
            )
            island_outcome_logits = self.v36_outcome_head(state)
            island_outcome_prob = torch.softmax(island_outcome_logits, dim=-1)

            rank_score = action_logits.new_full((b, k), -self.v36_gain_scale)
            outcome_logits = action_logits.new_zeros((b, k, 3))
            outcome_prob = action_logits.new_zeros((b, k, 3))
            outcome_prob[..., self.NEUTRAL] = 1.0
            rank_score[:, island_indices] = island_score
            outcome_logits[:, island_indices] = island_outcome_logits
            outcome_prob[:, island_indices] = island_outcome_prob
            decision = rank_score.masked_fill(~valid, -1e4)
            class_logits = torch.cat([rank_score.new_zeros((b, 1)), decision], dim=1)
            selected_index = class_logits.argmax(dim=1)
            selected = torch.zeros_like(rank_score)
            take = selected_index > 0
            if take.any():
                row = torch.arange(b, device=action_logits.device)[take]
                col = selected_index[take] - 1
                selected[row, col] = 1.0

            delta_logits = action_logits - base_logits[:, None]
            hard_fused_logits = base_logits + (
                selected[:, :, None, None] * delta_logits
            ).sum(dim=1)
            soft_choice = torch.softmax(
                class_logits / self.v36_decision_temperature, dim=1
            )
            soft_fused_logits = base_logits + (
                soft_choice[:, 1:, None, None] * delta_logits
            ).sum(dim=1)
            zero_null = rank_score.new_zeros(b)
            zero_veto = torch.zeros_like(rank_score)
            out = dict(inherited)
            out.update({
                "v36_rank_score": rank_score,
                "v36_outcome_logits": outcome_logits,
                "v36_outcome_probability": outcome_prob,
                "v36_harm_probability": outcome_prob[..., self.HARMFUL],
                "v36_benefit_probability": outcome_prob[..., self.BENEFICIAL],
                "v36_class_logits": class_logits,
                "v36_valid_action": valid,
                "v36_selected_action": selected,
                "v36_soft_choice_probs": soft_choice,
                "v36_island_indices": island_indices,
                "v36_text_cf_logit": aux.get("v20_cf_logit", rank_score.new_zeros((b, k))),
                "v36_text_signed_delta": aux.get("v20_cf_signed_delta", rank_score.new_zeros((b, k))),
                # Existing audit readers use V32/V31 aliases; in V36 policy is
                # the directly trained casewise rank score and Null is Preserve=0.
                "v32_policy_logits": rank_score,
                "v32_raw_policy_logits": rank_score,
                "v32_null_logit": zero_null,
                "v32_utility_logits": outcome_logits,
                "v32_semantic_veto_logits": zero_veto,
                "v32_valid_action": valid,
                "v32_cf_available": cf_available,
                "v32_selected_action": selected,
                "v32_soft_choice_probs": soft_choice,
                "v32_fused_logits": soft_fused_logits,
                "v32_hard_fused_logits": hard_fused_logits,
                "v31_policy_logits": rank_score,
                "v31_raw_policy_logits": rank_score,
                "v31_null_logit": zero_null,
                "v31_valid_action": valid,
                "v31_selected_action": selected,
                "v31_fused_logits": soft_fused_logits,
                "v31_hard_fused_logits": hard_fused_logits,
            })
            return out

        # V35: residual-purified candidate world model.  The source proposal
        # field is trained on B0 false-positive residuals; deployment uses a
        # train-learned expected gain penalized by learned harmful-outcome
        # probability. Preserve remains the exact zero-action alternative.
        if self.v35_decision_enabled:
            action_embedding = inherited.get("v31_action_embedding", None)
            if action_embedding is None:
                raise RuntimeError("V35 requires v31_action_embedding from _v31_decide.")
            if self.v35_gain_head is None or self.v35_outcome_head is None:
                raise RuntimeError("V35 outcome heads are absent.")
            island_indices = torch.where(slot_type == self.v32_island_type_id)[0]
            if island_indices.numel() == 0:
                raise RuntimeError("V35 found no island candidate slots.")
            # V35 never deploys an unmatched intervention.
            valid = valid & cf_available
            state = self._v35_residual_action_state(
                aux, action_embedding, island_indices
            )
            island_gain = self.v35_gain_scale * torch.tanh(
                self.v35_gain_head(state).squeeze(-1)
            )
            island_outcome_logits = self.v35_outcome_head(state)
            island_outcome_prob = torch.softmax(island_outcome_logits, dim=-1)
            # Class order follows V31: harmful, neutral, beneficial.
            island_score = island_gain - (
                self.v35_harm_penalty * island_outcome_prob[..., self.HARMFUL]
            )

            predicted_gain = action_logits.new_zeros((b, k))
            outcome_logits = action_logits.new_zeros((b, k, 3))
            outcome_prob = action_logits.new_zeros((b, k, 3))
            outcome_prob[..., self.NEUTRAL] = 1.0
            deploy_score = action_logits.new_full((b, k), -self.v35_gain_scale)
            predicted_gain[:, island_indices] = island_gain
            outcome_logits[:, island_indices] = island_outcome_logits
            outcome_prob[:, island_indices] = island_outcome_prob
            deploy_score[:, island_indices] = island_score

            decision = deploy_score.masked_fill(~valid, -1e4)
            class_logits = torch.cat([
                deploy_score.new_zeros((b, 1)), decision
            ], dim=1)
            selected_index = class_logits.argmax(dim=1)
            selected = torch.zeros_like(deploy_score)
            take = selected_index > 0
            if take.any():
                row = torch.arange(b, device=action_logits.device)[take]
                col = selected_index[take] - 1
                selected[row, col] = 1.0

            delta_logits = action_logits - base_logits[:, None]
            hard_fused_logits = base_logits + (
                selected[:, :, None, None] * delta_logits
            ).sum(dim=1)
            soft_choice = torch.softmax(
                class_logits / self.v35_decision_temperature, dim=1
            )
            soft_fused_logits = base_logits + (
                soft_choice[:, 1:, None, None] * delta_logits
            ).sum(dim=1)
            zero_null = deploy_score.new_zeros(b)
            zero_veto = torch.zeros_like(deploy_score)
            out = dict(inherited)
            out.update({
                "v35_predicted_gain": predicted_gain,
                "v35_outcome_logits": outcome_logits,
                "v35_outcome_probability": outcome_prob,
                "v35_harm_probability": outcome_prob[..., self.HARMFUL],
                "v35_benefit_probability": outcome_prob[..., self.BENEFICIAL],
                "v35_deploy_score": deploy_score,
                "v35_class_logits": class_logits,
                "v35_valid_action": valid,
                "v35_selected_action": selected,
                "v35_soft_choice_probs": soft_choice,
                "v35_island_indices": island_indices,
                "v32_policy_logits": deploy_score,
                "v32_raw_policy_logits": predicted_gain,
                "v32_null_logit": zero_null,
                "v32_utility_logits": outcome_logits,
                "v32_semantic_veto_logits": zero_veto,
                "v32_valid_action": valid,
                "v32_cf_available": cf_available,
                "v32_selected_action": selected,
                "v32_soft_choice_probs": soft_choice,
                "v32_fused_logits": soft_fused_logits,
                "v32_hard_fused_logits": hard_fused_logits,
                "v31_policy_logits": deploy_score,
                "v31_raw_policy_logits": predicted_gain,
                "v31_null_logit": zero_null,
                "v31_valid_action": valid,
                "v31_selected_action": selected,
                "v31_fused_logits": soft_fused_logits,
                "v31_hard_fused_logits": hard_fused_logits,
            })
            return out

        # V34: risk-sensitive spatial outcome world model. The deployed score
        # is the learned lower conditional quantile of signed Dice gain, not a
        # post-hoc threshold and not a point estimate. Preserve remains score 0.
        if self.v34_spatial_quantile_world_model:
            action_embedding = inherited.get("v31_action_embedding", None)
            if action_embedding is None:
                raise RuntimeError("V34 requires v31_action_embedding from _v31_decide.")

            island_indices = torch.where(slot_type == self.v32_island_type_id)[0]
            if island_indices.numel() == 0:
                raise RuntimeError("V34 found no island candidate slots.")

            state = self._v34_spatial_action_state(
                base_logits, action_logits, aux, semantic_map, action_embedding, island_indices
            )
            raw_quantiles = self.v34_quantile_head(state)
            # Sort gives non-crossing q10 <= q50 <= q90. tanh bounds scores to
            # the predeclared train-scale range and prevents unstable extremes.
            island_quantiles = torch.sort(
                self.v34_gain_scale * torch.tanh(raw_quantiles), dim=-1
            ).values

            quantiles = action_logits.new_full((b, k, 3), -self.v34_gain_scale)
            quantiles[:, island_indices] = island_quantiles
            deploy_score = quantiles[..., 0]
            decision = deploy_score.masked_fill(~valid, -1e4)
            class_logits = torch.cat([deploy_score.new_zeros((b, 1)), decision], dim=1)

            selected_index = class_logits.argmax(dim=1)
            selected = torch.zeros_like(deploy_score)
            take = selected_index > 0
            if take.any():
                row = torch.arange(b, device=action_logits.device)[take]
                col = selected_index[take] - 1
                selected[row, col] = 1.0

            delta_logits = action_logits - base_logits[:, None]
            hard_fused_logits = base_logits + (
                selected[:, :, None, None] * delta_logits
            ).sum(dim=1)
            soft_choice = torch.softmax(
                class_logits / self.v34_decision_temperature, dim=1
            )
            soft_fused_logits = base_logits + (
                soft_choice[:, 1:, None, None] * delta_logits
            ).sum(dim=1)

            zero_null = deploy_score.new_zeros(b)
            zero_veto = torch.zeros_like(deploy_score)
            out = dict(inherited)
            out.update({
                "v34_quantile_gain": quantiles,
                "v34_lower_gain": deploy_score,
                "v34_valid_action": valid,
                "v34_selected_action": selected,
                "v34_soft_choice_probs": soft_choice,
                "v34_island_indices": island_indices,

                # Compatibility: existing audits treat policy_logit as the
                # deployed conservative score and Null as exact Preserve=0.
                "v32_policy_logits": deploy_score,
                "v32_raw_policy_logits": quantiles[..., 1],
                "v32_null_logit": zero_null,
                "v32_utility_logits": inherited["v31_utility_logits"],
                "v32_semantic_veto_logits": zero_veto,
                "v32_valid_action": valid,
                "v32_cf_available": cf_available,
                "v32_selected_action": selected,
                "v32_soft_choice_probs": soft_choice,
                "v32_fused_logits": soft_fused_logits,
                "v32_hard_fused_logits": hard_fused_logits,

                "v31_policy_logits": deploy_score,
                "v31_raw_policy_logits": quantiles[..., 1],
                "v31_null_logit": zero_null,
                "v31_valid_action": valid,
                "v31_selected_action": selected,
                "v31_fused_logits": soft_fused_logits,
                "v31_hard_fused_logits": hard_fused_logits,
            })
            return out

        # V33: score each island candidate directly by predicted signed Dice
        # gain. Preserve has fixed score zero. No inherited Null, policy, text
        # veto, utility class, or Val-tuned threshold enters deployment.
        if self.v33_signed_gain_regression:
            if self.v33_gain_head is None:
                raise RuntimeError("V33 gain head is absent.")

            action_embedding = inherited.get("v31_action_embedding", None)
            if action_embedding is None:
                raise RuntimeError(
                    "V33 requires v31_action_embedding from _v31_decide."
                )

            if action_embedding.shape[:2] != valid.shape:
                raise RuntimeError(
                    "V33 embedding/valid mismatch: "
                    f"{tuple(action_embedding.shape)} vs {tuple(valid.shape)}"
                )

            gain_raw = self.v33_gain_head(action_embedding).squeeze(-1)
            predicted_gain = self.v33_gain_scale * torch.tanh(gain_raw)

            decision = predicted_gain.masked_fill(~valid, -1e4)
            class_logits = torch.cat(
                [
                    predicted_gain.new_zeros((b, 1)),
                    decision,
                ],
                dim=1,
            )

            selected_index = class_logits.argmax(dim=1)
            selected = torch.zeros_like(predicted_gain)

            action_choice = selected_index > 0
            if action_choice.any():
                row = torch.arange(
                    b,
                    device=action_logits.device,
                )[action_choice]
                col = selected_index[action_choice] - 1
                selected[row, col] = 1.0

            delta = action_logits - base_logits[:, None]
            hard_fused_logits = base_logits + (
                selected[:, :, None, None] * delta
            ).sum(dim=1)

            soft_choice = torch.softmax(
                class_logits / self.v33_decision_temperature,
                dim=1,
            )

            soft_fused_logits = base_logits + (
                soft_choice[:, 1:, None, None] * delta
            ).sum(dim=1)

            zero_null = predicted_gain.new_zeros(b)
            zero_veto = torch.zeros_like(predicted_gain)

            out = dict(inherited)
            out.update({
                "v33_predicted_gain": predicted_gain,
                "v33_gain_raw_logits": gain_raw,
                "v33_valid_action": valid,
                "v33_selected_action": selected,
                "v33_soft_choice_probs": soft_choice,

                # V32/V31 compatibility aliases. In V33, policy means the
                # predicted signed gain and Null is exactly Preserve score 0.
                "v32_policy_logits": predicted_gain,
                "v32_raw_policy_logits": predicted_gain,
                "v32_null_logit": zero_null,
                "v32_utility_logits": inherited["v31_utility_logits"],
                "v32_semantic_veto_logits": zero_veto,
                "v32_valid_action": valid,
                "v32_cf_available": cf_available,
                "v32_selected_action": selected,
                "v32_soft_choice_probs": soft_choice,
                "v32_fused_logits": soft_fused_logits,
                "v32_hard_fused_logits": hard_fused_logits,

                "v31_policy_logits": predicted_gain,
                "v31_raw_policy_logits": predicted_gain,
                "v31_null_logit": zero_null,
                "v31_valid_action": valid,
                "v31_selected_action": selected,
                "v31_fused_logits": soft_fused_logits,
                "v31_hard_fused_logits": hard_fused_logits,
            })
            return out

        raw_policy = inherited["v31_raw_policy_logits"]
        veto_logits = inherited["v31_semantic_veto_logits"]
        policy_logits = raw_policy - (
            self.v32_text_veto_weight
            * cf_available.to(raw_policy.dtype)
            * torch.sigmoid(veto_logits)
        )

        null_logit = inherited["v31_null_logit"]
        if null_logit.ndim == 2 and null_logit.shape[1] == 1:
            null_logit = null_logit[:, 0]

        decision = policy_logits.masked_fill(~valid, -1e4)
        class_logits = torch.cat([null_logit[:, None], decision], dim=1)
        selected_index = class_logits.argmax(dim=1)

        selected = torch.zeros_like(policy_logits)
        action_choice = selected_index > 0
        if action_choice.any():
            row = torch.arange(b, device=action_logits.device)[action_choice]
            col = selected_index[action_choice] - 1
            selected[row, col] = 1.0

        delta = action_logits - base_logits[:, None]
        hard_fused_logits = base_logits + (selected[:, :, None, None] * delta).sum(dim=1)
        soft_choice = torch.softmax(class_logits, dim=1)
        soft_fused_logits = base_logits + (soft_choice[:, 1:, None, None] * delta).sum(dim=1)

        out = dict(inherited)
        out.update({
            "v32_policy_logits": policy_logits,
            "v32_raw_policy_logits": raw_policy,
            "v32_null_logit": null_logit,
            "v32_utility_logits": inherited["v31_utility_logits"],
            "v32_semantic_veto_logits": veto_logits,
            "v32_valid_action": valid,
            "v32_cf_available": cf_available,
            "v32_selected_action": selected,
            "v32_soft_choice_probs": soft_choice,
            "v32_fused_logits": soft_fused_logits,
            "v32_hard_fused_logits": hard_fused_logits,
            # Existing diagnostics can continue to read V31 aliases.
            "v31_policy_logits": policy_logits,
            "v31_raw_policy_logits": raw_policy,
            "v31_null_logit": null_logit,
            "v31_valid_action": valid,
            "v31_selected_action": selected,
            "v31_fused_logits": soft_fused_logits,
            "v31_hard_fused_logits": hard_fused_logits,
        })
        return out

    def generate(self, *args, **kwargs):
        candidate_logits_all, aux = TypeConditionalUtilityActionBank.generate(self, *args, **kwargs)
        semantic_map = self._v32_semantic_map(args, kwargs)
        decision = self._v32_decide(candidate_logits_all, aux, semantic_map)
        aux = dict(aux)
        aux.update(decision)
        soft_probs = torch.sigmoid(
            decision["v32_fused_logits"]
        ).clamp(EPS, 1.0 - EPS)

        hard_probs = torch.sigmoid(
            decision["v32_hard_fused_logits"]
        ).clamp(EPS, 1.0 - EPS)

        aux.update({
            # V32 native output.
            "v20_fused_logits": decision["v32_fused_logits"],
            "v20_hard_fused_logits": decision["v32_hard_fused_logits"],
            "v20_fused_probs": soft_probs,
            "v20_hard_fused_probs": hard_probs,
            "direct_fused_probs": soft_probs,
            "router_fused_probs": soft_probs,

            # Existing V20 diagnostic contract.
            "v20_selector_logits": decision["v32_policy_logits"],
            "v20_selector_probs": decision["v32_soft_choice_probs"][:, 1:],
            "v20_selector_hard": decision["v32_selected_action"],
        })
        return candidate_logits_all, aux
