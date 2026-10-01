"""Losses and factual diagnostics for MHCS-R2 structured stochastic bank.

The bank is trained as one image-conditioned distribution rather than K heads
competing for one annotation.  The distribution objective is the Monte-Carlo
negative log likelihood of a low-rank logistic-normal segmentation model,
following the Stochastic Segmentation Networks principle.  The only second
objective is the standard BCE+Dice loss of the final compositional mask.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

EPS = 1.0e-4


def _target_3d(masks: torch.Tensor) -> torch.Tensor:
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise ValueError(f"Expected masks [B,H,W] or [B,1,H,W], got {tuple(masks.shape)}")
    return (masks > 0.5).to(dtype=masks.dtype)


def _soft_dice_loss(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target[:, None].expand_as(prob)
    inter = (prob * target).flatten(2).sum(-1)
    den = prob.flatten(2).sum(-1) + target.flatten(2).sum(-1)
    return 1.0 - (2.0 * inter + EPS) / (den + EPS)


def _seg_loss_per_case_candidate(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target4 = target[:, None].expand_as(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target4, reduction="none").flatten(2).mean(-1)
    dice = _soft_dice_loss(torch.sigmoid(logits), target)
    return 0.5 * (bce + dice)


def _joint_mc_nll(sample_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-pixel-normalised joint Bernoulli mixture NLL.

    sample_logits: [B,M,H,W].  For each MC draw we compute the joint Bernoulli
    log likelihood of the full label map, then marginalise the latent sample by
    log-mean-exp.  Division by H*W changes only scale, not the optimum.
    """
    if sample_logits.ndim != 4:
        raise RuntimeError(f"Expected SSN samples [B,M,H,W], got {tuple(sample_logits.shape)}")
    b, m, h, w = sample_logits.shape
    y = target[:, None].expand_as(sample_logits)
    neg_log_px = F.binary_cross_entropy_with_logits(sample_logits, y, reduction="none")
    log_joint = -neg_log_px.flatten(2).sum(-1)  # [B,M]
    log_marginal = torch.logsumexp(log_joint, dim=1) - math.log(float(m))
    return (-log_marginal / float(h * w)).mean()


def _hard_dice(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = (prob >= 0.5).to(target.dtype)
    gt = target[:, None].expand_as(pred)
    inter = (pred * gt).flatten(2).sum(-1)
    den = pred.flatten(2).sum(-1) + gt.flatten(2).sum(-1)
    score = (2.0 * inter + EPS) / (den + EPS)
    return torch.where(den <= EPS, torch.ones_like(score), score)


def _pixelwise_oracle(candidate_probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    hard = candidate_probs >= 0.5
    gt = target[:, None].bool()
    match = hard.eq(gt)
    repairable = match.any(dim=1)
    fallback = hard[:, 0]
    oracle_hard = torch.where(repairable, target.bool(), fallback)
    pred = oracle_hard.to(target.dtype)
    inter = (pred * target).flatten(1).sum(-1)
    den = pred.flatten(1).sum(-1) + target.flatten(1).sum(-1)
    return torch.where(den <= EPS, torch.ones_like(den), (2.0 * inter + EPS) / (den + EPS))


def _pairwise_l1(prob: torch.Tensor) -> torch.Tensor:
    if prob.shape[1] <= 1:
        return prob.new_zeros(prob.shape[0])
    vals = []
    for i in range(prob.shape[1]):
        for j in range(i + 1, prob.shape[1]):
            vals.append((prob[:, i] - prob[:, j]).abs().mean(dim=(-2, -1)))
    return torch.stack(vals, dim=1).mean(dim=1)


def _generated_unique_correct(candidate_probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Fraction of pixels uniquely correct for each generated hypothesis."""
    gen = candidate_probs[:, 1:] >= 0.5
    gt = target[:, None].bool()
    correct = gen.eq(gt)
    count = correct.sum(dim=1, keepdim=True)
    unique = correct & (count == 1)
    return unique.float().mean(dim=(-2, -1))  # [B,K]


def _loo_pwo_drop(candidate_probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """PWO loss in Dice when each generated candidate is removed."""
    full = _pixelwise_oracle(candidate_probs, target)
    drops = []
    for j in range(1, candidate_probs.shape[1]):
        keep = [i for i in range(candidate_probs.shape[1]) if i != j]
        pwo_without = _pixelwise_oracle(candidate_probs[:, keep], target)
        drops.append((full - pwo_without).clamp_min(0.0))
    return torch.stack(drops, dim=1)


def compute_multi_hypothesis_composition_loss(
    cfg,
    candidates: torch.Tensor,
    masks: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    epoch: int = 0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    del cfg, candidates, epoch
    generated_logits = aux.get("mhcs_generated_logits")
    candidate_probs = aux.get("candidate_probs")
    final_probs = aux.get("mhcs_final_probs")
    sample_logits = aux.get("mhcs_distribution_samples")
    mean_logits = aux.get("mhcs_distribution_mean_logits")
    factors = aux.get("mhcs_distribution_factors")
    diag_std = aux.get("mhcs_distribution_diag_std")
    log_vars = aux.get("mhcs_loss_log_vars")
    required = (generated_logits, candidate_probs, final_probs, sample_logits, mean_logits, factors, diag_std, log_vars)
    if not all(isinstance(x, torch.Tensor) for x in required):
        raise RuntimeError("MHCS-R2 loss requires distribution, candidate bank, final mask and log_vars tensors.")
    if candidate_probs.shape[1] != generated_logits.shape[1] + 1:
        raise RuntimeError("MHCS-R2 slot 0 must be Base followed by generated sigma hypotheses.")

    target = _target_3d(masks).to(generated_logits)

    # Principled SSN bank objective: no winner ownership and no per-head WTA.
    distribution_nll = _joint_mc_nll(sample_logits, target)

    final_logits = torch.logit(final_probs.clamp(EPS, 1.0 - EPS))[:, None]
    final_loss = _seg_loss_per_case_candidate(final_logits, target)[:, 0].mean()

    s = log_vars.reshape(-1)
    if s.numel() != 2:
        raise RuntimeError(f"MHCS-R2 loss_log_vars must contain 2 scalars, got {s.numel()}.")
    objective = 0.5 * (
        torch.exp(-s[0]) * final_loss + s[0]
        + torch.exp(-s[1]) * distribution_nll + s[1]
    )

    with torch.no_grad():
        hard_dice = _hard_dice(candidate_probs, target)
        base_dice = hard_dice[:, 0]
        best_single = hard_dice.max(dim=1).values
        generated_best = hard_dice[:, 1:].max(dim=1).values
        mean_dice = _hard_dice(torch.sigmoid(mean_logits)[:, None], target)[:, 0]
        final_dice = _hard_dice(final_probs[:, None], target)[:, 0]
        pwo = _pixelwise_oracle(candidate_probs, target)
        generated_pwo = _pixelwise_oracle(candidate_probs[:, 1:], target)
        complementarity = pwo - best_single
        generated_complementarity = generated_pwo - generated_best
        composition_gain = final_dice - best_single
        # Stable aggregate ratio is reported below from means.  Per-case ratios
        # are deliberately not averaged because near-zero denominators explode.
        pairwise_l1 = _pairwise_l1(candidate_probs[:, 1:])
        weights = aux.get("mhcs_composer_weights")
        if isinstance(weights, torch.Tensor):
            weight_entropy = -(weights.clamp_min(EPS) * torch.log(weights.clamp_min(EPS))).sum(dim=1).mean(dim=(-2, -1)) / math.log(weights.shape[1])
            base_weight = weights[:, 0].mean(dim=(-2, -1))
        else:
            weight_entropy = base_dice.new_zeros(base_dice.shape)
            base_weight = base_dice.new_zeros(base_dice.shape)
        unique = _generated_unique_correct(candidate_probs, target)
        loo_drop = _loo_pwo_drop(candidate_probs, target)
        factor_rms = factors.square().mean(dim=(-2, -1)).sqrt().mean(dim=1)
        diag_mean = diag_std.mean(dim=(-2, -1))
        factor_cos = aux.get("mhcs_factor_cosine")
        context_gate = aux.get("mhcs_context_gate")

    comp_gap_mean = complementarity.mean()
    comp_gain_mean = composition_gain.mean()
    realization = torch.where(
        comp_gap_mean.abs() > 1.0e-6,
        comp_gain_mean / comp_gap_mean.clamp_min(1.0e-6),
        comp_gap_mean.new_zeros(()),
    )

    diagnostics: Dict[str, torch.Tensor] = {
        "mhcs_objective": objective.detach(),
        "mhcs_final_loss": final_loss.detach(),
        "mhcs_distribution_nll": distribution_nll.detach(),
        "mhcs_loss_weight_final": torch.exp(-s[0]).detach(),
        "mhcs_loss_weight_distribution": torch.exp(-s[1]).detach(),
        "mhcs_base_dice": base_dice.mean(),
        "mhcs_distribution_mean_dice": mean_dice.mean(),
        "mhcs_generated_best_dice": generated_best.mean(),
        "mhcs_generated_pwo_dice": generated_pwo.mean(),
        "mhcs_generated_complementarity_gap": generated_complementarity.mean(),
        "mhcs_best_single_dice": best_single.mean(),
        "mhcs_pwo_dice": pwo.mean(),
        "mhcs_complementarity_gap": comp_gap_mean,
        "mhcs_final_dice": final_dice.mean(),
        "mhcs_composition_gain_over_best_single": comp_gain_mean,
        "mhcs_composition_realization_ratio": realization,
        "mhcs_pairwise_l1": pairwise_l1.mean(),
        "mhcs_composer_weight_entropy": weight_entropy.mean(),
        "mhcs_base_weight_mean": base_weight.mean(),
        "mhcs_factor_rms": factor_rms.mean(),
        "mhcs_diag_std_mean": diag_mean.mean(),
    }
    if isinstance(factor_cos, torch.Tensor):
        diagnostics["mhcs_factor_cosine"] = factor_cos.mean()
    if isinstance(context_gate, torch.Tensor):
        diagnostics["mhcs_context_gate"] = context_gate.detach().mean()
    tcos = aux.get("mhcs_set_token_cosine")
    if isinstance(tcos, torch.Tensor):
        diagnostics["mhcs_set_token_cosine"] = tcos.mean()
    for index in range(unique.shape[1]):
        diagnostics[f"mhcs_unique_correct_h{index+1}"] = unique[:, index].mean()
        diagnostics[f"mhcs_loo_pwo_drop_h{index+1}"] = loo_drop[:, index].mean()
    return objective, diagnostics
