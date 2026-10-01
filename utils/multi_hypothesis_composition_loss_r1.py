"""Loss and diagnostics for Multi-Hypothesis Compositional Segmentation.

Only standard segmentation losses are used:
  * BCE + soft Dice for the final composed mask.
  * Multiple-choice (best-of-K) BCE + soft Dice for the generated bank.

The relative weighting of these two standard objectives is learned with
homoscedastic uncertainty, avoiding a hand-tuned bank-vs-composer coefficient.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

EPS = 1.0e-6


def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def _target_3d(masks: torch.Tensor) -> torch.Tensor:
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise ValueError(f"Expected masks [B,H,W] or [B,1,H,W], got {tuple(masks.shape)}")
    return (masks > 0.5).to(dtype=masks.dtype)


def _soft_dice_loss(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # prob [B,K,H,W], target [B,H,W]
    target = target[:, None].expand_as(prob)
    inter = (prob * target).flatten(2).sum(-1)
    den = prob.flatten(2).sum(-1) + target.flatten(2).sum(-1)
    return 1.0 - (2.0 * inter + EPS) / (den + EPS)


def _seg_loss_per_case_candidate(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Standard equal BCE+Dice segmentation objective, with no task knob."""
    target4 = target[:, None].expand_as(logits)
    bce = F.binary_cross_entropy_with_logits(
        logits, target4, reduction="none"
    ).flatten(2).mean(-1)
    dice = _soft_dice_loss(torch.sigmoid(logits), target)
    return 0.5 * (bce + dice)


def _hard_dice(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # prob [B,K,H,W]
    pred = (prob >= 0.5).to(target.dtype)
    gt = target[:, None].expand_as(pred)
    inter = (pred * gt).flatten(2).sum(-1)
    den = pred.flatten(2).sum(-1) + gt.flatten(2).sum(-1)
    score = (2.0 * inter + EPS) / (den + EPS)
    empty = den <= EPS
    return torch.where(empty, torch.ones_like(score), score)


def _pixelwise_oracle(candidate_probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    hard = candidate_probs >= 0.5
    gt = target[:, None].bool()
    match = hard.eq(gt)
    repairable = match.any(dim=1)
    base_hard = hard[:, 0]
    oracle_hard = torch.where(repairable, target.bool(), base_hard)
    pred = oracle_hard.to(target.dtype)
    inter = (pred * target).flatten(1).sum(-1)
    den = pred.flatten(1).sum(-1) + target.flatten(1).sum(-1)
    return torch.where(
        den <= EPS,
        torch.ones_like(den),
        (2.0 * inter + EPS) / (den + EPS),
    )


def _pairwise_l1(prob: torch.Tensor) -> torch.Tensor:
    if prob.shape[1] <= 1:
        return prob.new_zeros(prob.shape[0])
    vals = []
    for i in range(prob.shape[1]):
        for j in range(i + 1, prob.shape[1]):
            vals.append((prob[:, i] - prob[:, j]).abs().mean(dim=(-2, -1)))
    return torch.stack(vals, dim=1).mean(dim=1)


def compute_multi_hypothesis_composition_loss(
    cfg,
    candidates: torch.Tensor,
    masks: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    epoch: int = 0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    del candidates, epoch
    generated_logits = aux.get("mhcs_generated_logits")
    candidate_probs = aux.get("candidate_probs")
    final_probs = aux.get("mhcs_final_probs")
    log_vars = aux.get("mhcs_loss_log_vars")
    if not all(isinstance(x, torch.Tensor) for x in (generated_logits, candidate_probs, final_probs, log_vars)):
        raise RuntimeError("MHCS loss requires generated logits, candidate bank, final probs and loss_log_vars.")
    if generated_logits.ndim != 4 or candidate_probs.ndim != 4:
        raise RuntimeError("MHCS tensors must be [B,K,H,W].")
    if candidate_probs.shape[1] != generated_logits.shape[1] + 1:
        raise RuntimeError("MHCS candidate slot 0 must be Base followed by K generated hypotheses.")

    target = _target_3d(masks).to(generated_logits)
    # MHCS deliberately does not expose bank/final BCE-vs-Dice coefficients.
    # Both use the same standard equal BCE+Dice objective; only the relative
    # final-vs-bank importance is learned through homoscedastic uncertainty.
    bank_per = _seg_loss_per_case_candidate(generated_logits, target)
    best_bank_loss, winner = bank_per.min(dim=1)
    bank_loss = best_bank_loss.mean()

    final_logits = torch.logit(final_probs.clamp(EPS, 1.0 - EPS))[:, None]
    final_per = _seg_loss_per_case_candidate(final_logits, target)[:, 0]
    final_loss = final_per.mean()

    s = log_vars.reshape(-1)
    if s.numel() != 2:
        raise RuntimeError(f"MHCS loss_log_vars must contain 2 scalars, got {s.numel()}.")
    objective = 0.5 * (
        torch.exp(-s[0]) * final_loss + s[0]
        + torch.exp(-s[1]) * bank_loss + s[1]
    )

    with torch.no_grad():
        hard_dice = _hard_dice(candidate_probs, target)
        base_dice = hard_dice[:, 0]
        best_single = hard_dice.max(dim=1).values
        generated_best = hard_dice[:, 1:].max(dim=1).values
        final_dice = _hard_dice(final_probs[:, None], target)[:, 0]
        pwo = _pixelwise_oracle(candidate_probs, target)
        complementarity = pwo - best_single
        composition_gain = final_dice - best_single
        realization = torch.where(
            complementarity > 1.0e-6,
            composition_gain / complementarity.clamp_min(1.0e-6),
            torch.zeros_like(complementarity),
        )
        generated_probs = candidate_probs[:, 1:]
        pairwise_l1 = _pairwise_l1(generated_probs)
        weights = aux.get("mhcs_composer_weights")
        if isinstance(weights, torch.Tensor):
            weight_entropy = -(
                weights.clamp_min(EPS) * torch.log(weights.clamp_min(EPS))
            ).sum(dim=1).mean(dim=(-2, -1)) / math.log(weights.shape[1])
            base_weight = weights[:, 0].mean(dim=(-2, -1))
        else:
            weight_entropy = base_dice.new_zeros(base_dice.shape)
            base_weight = base_dice.new_zeros(base_dice.shape)

        usage = F.one_hot(winner, num_classes=generated_logits.shape[1]).float().mean(dim=0)
        usage_entropy = -(
            usage.clamp_min(EPS) * torch.log(usage.clamp_min(EPS))
        ).sum() / math.log(max(int(usage.numel()), 2))

    diagnostics: Dict[str, torch.Tensor] = {
        "mhcs_objective": objective.detach(),
        "mhcs_final_loss": final_loss.detach(),
        "mhcs_bank_mcl_loss": bank_loss.detach(),
        "mhcs_loss_weight_final": torch.exp(-s[0]).detach(),
        "mhcs_loss_weight_bank": torch.exp(-s[1]).detach(),
        "mhcs_base_dice": base_dice.mean(),
        "mhcs_generated_best_dice": generated_best.mean(),
        "mhcs_best_single_dice": best_single.mean(),
        "mhcs_pwo_dice": pwo.mean(),
        "mhcs_complementarity_gap": complementarity.mean(),
        "mhcs_final_dice": final_dice.mean(),
        "mhcs_composition_gain_over_best_single": composition_gain.mean(),
        "mhcs_composition_realization_ratio": realization.mean(),
        "mhcs_pairwise_l1": pairwise_l1.mean(),
        "mhcs_composer_weight_entropy": weight_entropy.mean(),
        "mhcs_base_weight_mean": base_weight.mean(),
        "mhcs_winner_usage_entropy": usage_entropy,
    }
    qcos = aux.get("mhcs_hypothesis_query_cosine")
    tcos = aux.get("mhcs_set_token_cosine")
    if isinstance(qcos, torch.Tensor):
        diagnostics["mhcs_hypothesis_query_cosine"] = qcos.mean()
    if isinstance(tcos, torch.Tensor):
        diagnostics["mhcs_set_token_cosine"] = tcos.mean()
    for index, value in enumerate(usage):
        diagnostics[f"mhcs_winner_usage_h{index+1}"] = value
    return objective, diagnostics
