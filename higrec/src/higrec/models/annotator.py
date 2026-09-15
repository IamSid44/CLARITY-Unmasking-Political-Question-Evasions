"""C3: modelling the annotator, and being honest about what is not estimable.

The observed label is treated as a latent leaf passed through an
annotator-specific corruption. Recovering the annotator-free posterior is
useful in itself; more importantly it is what makes C4's q(c|x) computable, so
C3 stops being discarded at inference and becomes load-bearing.

WHAT IS AND IS NOT IDENTIFIABLE -- this is part of the contribution, not a caveat
------------------------------------------------------------------------------
Available supervision:

  * train, 3,448 items: ONE label each, plus the ID of the annotator who
    produced it. Items were distributed across annotators AT RANDOM, so
    differences in annotators' label MARGINALS are attributable to annotator
    effects rather than item effects.
  * dev, 308 items: three independent labels, no annotator IDs.

What follows:

  IDENTIFIABLE  a per-annotator marginal-bias vector b_a in R^9, from the train
                split, because random assignment means marginal differences are
                annotator effects. This is the PRIMARY model.
  IDENTIFIABLE  a single GLOBAL 9x9 confusion matrix, from the 308 dev triples.
  NOT IDENTIFIABLE  per-annotator confusion MATRICES. A 9x9 matrix has 72 free
                parameters per annotator; singly-labelled data with no repeated
                items per annotator cannot separate annotator confusion from the
                latent label distribution, and 308 triples without annotator IDs
                cannot attribute confusion to a specific annotator at all.

The marginal effects are large enough to be worth modelling: annotator 86
assigns Explicit 36.7% of the time against annotator 85's 24.2%, and annotator
85 assigns General 15.8% against roughly 9% for the other two.

Lineage: Dawid & Skene (1979); the crowd layer of Rodrigues & Pereira (AAAI
2018); regularized confusion estimation of Tanno et al. (CVPR 2019). We claim no
novelty for the mechanism -- only for what it is composed with in C4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from higrec.data.labels import EVASION_LABELS, N_EVASION

__all__ = [
    "MarginalBiasAnnotator",
    "LowRankConfusionAnnotator",
    "estimate_global_confusion",
    "measured_annotator_marginals",
    "validate_learned_against_measured",
]


class MarginalBiasAnnotator(nn.Module):
    """PRIMARY model: a per-annotator bias vector added to the shared logits.

        logits_observed = logits_latent + b_a

    At inference `b_a` is DROPPED, recovering the annotator-free posterior. That
    drop is the entire point: it is what "recovering a bias-free posterior"
    means operationally.

    Cheap, identifiable from singly-labelled data under random assignment, and
    exactly as expressive as the evidence supports -- no more.
    """

    def __init__(self, n_annotators: int, n_classes: int = N_EVASION) -> None:
        super().__init__()
        self.n_annotators = n_annotators
        self.n_classes = n_classes
        # Zero-initialized: the model starts as "no annotator effect" and must be
        # pushed away from it by evidence.
        self.bias = nn.Parameter(torch.zeros(n_annotators, n_classes))

    def forward(self, latent_logits: torch.Tensor, annotator_idx: torch.Tensor) -> torch.Tensor:
        """Latent logits -> the logits for the labelling annotator's observation."""
        return latent_logits + self.bias[annotator_idx]

    def clean_posterior(self, latent_logits: torch.Tensor) -> torch.Tensor:
        """Annotator-free posterior: drop b_a. Use this at inference."""
        return F.log_softmax(latent_logits, dim=-1)

    def annotator_posterior(
        self, latent_logits: torch.Tensor, annotator: int
    ) -> torch.Tensor:
        """P(annotator a labels c | x), the per-annotator term C4's q needs."""
        return F.softmax(latent_logits + self.bias[annotator], dim=-1)

    def all_annotator_posteriors(self, latent_logits: torch.Tensor) -> torch.Tensor:
        """(batch, n_annotators, n_classes), the exact input to q(c|x).

        Feeds `decision.setmembership.set_membership_from_annotators` directly.
        """
        expanded = latent_logits.unsqueeze(1) + self.bias.unsqueeze(0)
        return F.softmax(expanded, dim=-1)

    def identifiability_note(self) -> str:
        return (
            "Marginal-bias vectors are identifiable from singly-labelled data "
            "GIVEN random assignment of items to annotators. They capture how "
            "often an annotator reaches for a class, not which classes that "
            "annotator confuses for which."
        )


class LowRankConfusionAnnotator(nn.Module):
    """SECONDARY model: A_a = I + U_a V_a^T, regularized toward the identity.

    A constrained middle ground between a bias vector and a full per-annotator
    confusion matrix. The regularizer is what keeps it estimable: without a prior
    pulling toward identity, the rank-k factors are free to absorb the latent
    label distribution and the model becomes unidentifiable in exactly the way
    the module docstring warns about.

    Expose `reg_strength` as a swept hyperparameter (P5.1) and report the sweep.
    A result that only appears at one regularization value is a result about the
    regularizer.
    """

    def __init__(
        self,
        n_annotators: int,
        n_classes: int = N_EVASION,
        rank: int = 2,
        reg_strength: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_annotators = n_annotators
        self.n_classes = n_classes
        self.rank = rank
        self.reg_strength = reg_strength
        # Small init so A_a starts near the identity.
        self.u = nn.Parameter(torch.randn(n_annotators, n_classes, rank) * 0.01)
        self.v = nn.Parameter(torch.randn(n_annotators, n_classes, rank) * 0.01)

    def confusion(self, annotator_idx: torch.Tensor) -> torch.Tensor:
        """(batch, C, C) row-stochastic confusion matrices."""
        u, v = self.u[annotator_idx], self.v[annotator_idx]
        identity = torch.eye(self.n_classes, device=u.device).expand_as(u @ v.transpose(-1, -2))
        return F.softmax(identity + u @ v.transpose(-1, -2), dim=-1)

    def forward(self, latent_logits: torch.Tensor, annotator_idx: torch.Tensor) -> torch.Tensor:
        """Push the latent posterior through the annotator's confusion matrix."""
        latent = F.softmax(latent_logits, dim=-1).unsqueeze(1)  # (B, 1, C)
        observed = torch.bmm(latent, self.confusion(annotator_idx)).squeeze(1)
        return torch.log(observed.clamp_min(1e-12))

    def regularization(self) -> torch.Tensor:
        """Frobenius penalty pulling each A_a toward the identity (Tanno et al.)."""
        deviation = self.u @ self.v.transpose(-1, -2)
        return self.reg_strength * deviation.pow(2).sum(dim=(-2, -1)).mean()


def measured_annotator_marginals(
    annotator_ids: np.ndarray, leaf_labels: np.ndarray, n_classes: int = N_EVASION
) -> tuple[np.ndarray, np.ndarray]:
    """Directly measured per-annotator marginals from the training split.

    Returns
    -------
    (unique_ids, marginals)
        `marginals[a, c]` is the fraction of annotator a's labels that are c.

    This is ground truth for the C3 validation step, not a model output. P5.1
    requires comparing the LEARNED biases against this; if they disagree, the
    model is wrong and must be investigated BEFORE it is used to compute q.
    """
    annotator_ids = np.asarray(annotator_ids)
    leaf_labels = np.asarray(leaf_labels, dtype=np.int64)
    unique = np.unique(annotator_ids)

    marginals = np.zeros((unique.size, n_classes), dtype=np.float64)
    for a, annotator in enumerate(unique):
        labels = leaf_labels[(annotator_ids == annotator) & (leaf_labels >= 0)]
        if labels.size:
            marginals[a] = np.bincount(labels, minlength=n_classes) / labels.size
    return unique, marginals


def estimate_global_confusion(
    reference_sets: np.ndarray, consensus: np.ndarray, n_classes: int = N_EVASION
) -> np.ndarray:
    """A single GLOBAL 9x9 confusion matrix from the multi-reference items.

    `confusion[t, o]` is P(an annotator observes o | latent truth is t),
    estimated by treating the majority label as the latent truth and every
    annotator judgment as an observation of it.

    DELIBERATELY GLOBAL. Per-annotator matrices are not attempted: the dev split
    carries no annotator IDs, so a judgment cannot even be attributed to a
    specific annotator, and 308 items would be far too few regardless. P5.1
    requires this limit be written into the report as a stated finding rather
    than worked around.

    Items with no majority (roughly 33 of 308) have no usable latent-truth proxy
    and are skipped; the count of skipped items belongs in the report.
    """
    reference_sets = np.asarray(reference_sets, dtype=bool)
    consensus = np.asarray(consensus, dtype=np.int64)

    counts = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i, truth in enumerate(consensus):
        if truth < 0:  # no majority -> no latent-truth proxy
            continue
        for observed in np.flatnonzero(reference_sets[i]):
            counts[truth, observed] += 1.0

    row_totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, row_totals, out=np.zeros_like(counts), where=row_totals > 0)


def validate_learned_against_measured(
    learned_bias: np.ndarray,
    measured_marginals: np.ndarray,
    *,
    tolerance: float = 0.15,
) -> dict[str, object]:
    """Compare learned marginal biases against directly measured marginals.

    The P5.1 sanity check. A learned bias vector implies a marginal ordering over
    classes; that ordering should track the measured one. Reported as a rank
    correlation per annotator plus the largest single discrepancy.

    This is a check that can FAIL, and failing is informative: it would mean the
    bias vectors are absorbing something other than annotator preference.
    """
    from scipy.stats import spearmanr

    learned = np.asarray(learned_bias, dtype=np.float64)
    measured = np.asarray(measured_marginals, dtype=np.float64)
    if learned.shape != measured.shape:
        raise ValueError(f"shape mismatch: learned {learned.shape} vs measured {measured.shape}")

    correlations, worst = [], []
    for a in range(learned.shape[0]):
        rho = spearmanr(learned[a], measured[a]).statistic
        correlations.append(float(rho))
        implied = np.exp(learned[a]) / np.exp(learned[a]).sum()
        worst.append(float(np.max(np.abs(implied - measured[a]))))

    return {
        "spearman_per_annotator": correlations,
        "min_spearman": float(np.min(correlations)),
        "max_abs_marginal_error": float(np.max(worst)),
        "passes": bool(np.min(correlations) > 0.5 and np.max(worst) < tolerance),
        "class_names": list(EVASION_LABELS),
    }
