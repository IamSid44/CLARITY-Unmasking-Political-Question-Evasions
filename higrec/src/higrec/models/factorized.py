"""C1: the factorized attribute-code output head.

THE CENTRAL OBJECTION, STATED FIRST
------------------------------------
The attribute code is a DETERMINISTIC function of the leaf label. A softmax over
nine leaves and a factorized code therefore have IDENTICAL EXPRESSIVE CAPACITY
in the infinite-data limit: the code is a reparameterization, not added capacity.

It can only help under conditions the attribute-based-classification literature
identifies (Palatucci et al. 2009; Akata et al. 2013/2016): when classes are
long-tailed, supervision is limited, and attributes are more balanced than
leaves. All three hold here. The honest claim is that the code does not make the
model more expressive -- it re-allocates the sample-efficiency budget toward rare
leaves.

This module implements the mechanism. It does NOT establish that the mechanism
works; that is what the Phase 4.2 ablation and its random-code control are for.

ARCHITECTURE
------------
A shared backbone representation feeds five attribute heads of sizes
(3, 3, 2, 2, 3). The leaf posterior is the product of attribute posteriors
evaluated at that leaf's code, renormalized over the NINE VALID CODES only:

    P(leaf) ∝ Π_a P(attr_a = code_a(leaf))

The renormalization matters: the five heads range over 3*3*2*2*3 = 108
combinations, of which only 9 are valid codes. Restricting to those 9 is what
makes the factorization a classifier over leaves rather than over attribute
tuples, and it is where the structure-sharing happens.

All computation is in LOG SPACE via log_softmax and logsumexp. Products of
probabilities over five heads underflow readily in float32, and the failure is
silent -- it shows up as a mysteriously untrainable rare class, not as an error.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from higrec.data.attributes import (
    ATTRIBUTE_VALUES,
    ATTRIBUTES,
    CODE_VERSION,
    attribute_index_matrix,
)
from higrec.data.labels import N_EVASION

__all__ = [
    "FactorizedConfig",
    "FactorizedHead",
    "factorized_leaf_logits",
    "attribute_sum_loss",
]

ATTRIBUTE_SIZES: tuple[int, ...] = tuple(len(ATTRIBUTE_VALUES[a]) for a in ATTRIBUTES)


@dataclass(frozen=True)
class FactorizedConfig:
    """Configuration for the factorized head.

    `objective` selects between the two training losses specified in P4.1:

    "marginal"
        A single cross-entropy on the renormalized 9-way leaf posterior. This is
        the PRIMARY objective: it optimizes the metric actually reported and
        lets the renormalization do the work.

    "attribute_sum"
        The sum of five per-attribute cross-entropies, each head supervised by
        the deterministic attribute value. Available standalone for ablation.

    "both"
        Marginal plus `aux_weight` times the attribute-sum term.
    """

    hidden_size: int
    objective: str = "marginal"
    aux_weight: float = 0.0
    dropout: float = 0.0
    code_version: str = CODE_VERSION

    def __post_init__(self) -> None:
        if self.objective not in {"marginal", "attribute_sum", "both"}:
            raise ValueError(f"unknown objective {self.objective!r}")
        if self.objective == "both" and self.aux_weight <= 0:
            raise ValueError("objective='both' requires aux_weight > 0")


def factorized_leaf_logits(
    attribute_logits: list[torch.Tensor], code_index: torch.Tensor
) -> torch.Tensor:
    """Combine per-attribute logits into renormalized 9-way leaf logits.

    Parameters
    ----------
    attribute_logits
        Five tensors, the a-th of shape (batch, |values of attribute a|).
    code_index
        (9, 5) long tensor from `attribute_index_matrix()`: the value index each
        leaf takes on each attribute.

    Returns
    -------
    (batch, 9) tensor of UNNORMALIZED leaf log-scores. The caller applies
    log_softmax, which performs the renormalization over the nine valid codes.

    The unnormalized score for leaf l is

        s_l = Σ_a log P(attr_a = code_a(l))

    which is the log of the product of attribute probabilities. Because we then
    log_softmax over the 9 leaves, the result is exactly

        P(leaf) = Π_a P(attr_a = code_a(leaf)) / Σ_l' Π_a P(attr_a = code_a(l'))

    i.e. the product renormalized over valid codes only. The 99 invalid attribute
    combinations receive exactly zero mass by construction -- they are never
    enumerated, so they cannot leak probability.
    """
    if len(attribute_logits) != len(ATTRIBUTES):
        raise ValueError(
            f"expected {len(ATTRIBUTES)} attribute logit tensors, got {len(attribute_logits)}"
        )

    batch_size = attribute_logits[0].shape[0]
    leaf_scores = attribute_logits[0].new_zeros((batch_size, N_EVASION))

    for attr_j, logits in enumerate(attribute_logits):
        # Log-probabilities over this attribute's values.
        log_probs = F.log_softmax(logits, dim=-1)  # (batch, n_values)
        # Gather the value each leaf takes on this attribute: (9,) -> (batch, 9).
        value_of_leaf = code_index[:, attr_j]
        leaf_scores = leaf_scores + log_probs[:, value_of_leaf]

    return leaf_scores


class FactorizedHead(nn.Module):
    """Five attribute heads over a shared representation, combined into leaf logits.

    Swap-compatible with a flat `nn.Linear(hidden_size, 9)`: same input, same
    output shape. That is deliberate -- P4.1 requires the factorized and flat
    systems to differ in EXACTLY the output parameterization and nothing else,
    and a drop-in replacement is the cleanest way to guarantee it.
    """

    def __init__(self, config: FactorizedConfig) -> None:
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.heads = nn.ModuleList(
            [nn.Linear(config.hidden_size, size) for size in ATTRIBUTE_SIZES]
        )
        # (9, 5) constant; a buffer so it moves with .to(device) and is saved in
        # the state dict, making the code table part of the checkpoint.
        self.register_buffer(
            "code_index", torch.from_numpy(attribute_index_matrix()).long(), persistent=True
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """(batch, hidden) -> (batch, 9) leaf logits, renormalized over valid codes."""
        hidden = self.dropout(hidden)
        attribute_logits = [head(hidden) for head in self.heads]
        leaf_scores = factorized_leaf_logits(attribute_logits, self.code_index)
        return F.log_softmax(leaf_scores, dim=-1)

    def attribute_logits(self, hidden: torch.Tensor) -> list[torch.Tensor]:
        """Per-attribute logits, for the attribute-sum objective and for C5.

        Phase 7 feeds the span module's coverage and engagement statistics into
        the corresponding heads, and reports coverage-factor and
        engagement-factor accuracy SEPARATELY -- a decomposition no prior system
        on this task can report, and which needs these individual logits.
        """
        hidden = self.dropout(hidden)
        return [head(hidden) for head in self.heads]

    def loss(self, hidden: torch.Tensor, leaf_target: torch.Tensor) -> torch.Tensor:
        """Training loss under the configured objective."""
        objective = self.config.objective

        if objective in {"marginal", "both"}:
            log_posterior = self.forward(hidden)
            marginal = F.nll_loss(log_posterior, leaf_target)
            if objective == "marginal":
                return marginal
            aux = attribute_sum_loss(
                self.attribute_logits(hidden), leaf_target, self.code_index
            )
            return marginal + self.config.aux_weight * aux

        return attribute_sum_loss(
            self.attribute_logits(hidden), leaf_target, self.code_index
        )

    def trainable_parameter_count(self) -> int:
        """Trainable parameters in the head alone.

        P3.2 needs this number to build the parameter-matched flat capacity
        control: the matched baseline's LoRA rank is chosen so its trainable
        count equals this. Without that control, any C1 gain is attributable to
        extra parameters rather than to structure.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def attribute_sum_loss(
    attribute_logits: list[torch.Tensor],
    leaf_target: torch.Tensor,
    code_index: torch.Tensor,
) -> torch.Tensor:
    """Sum of five per-attribute cross-entropies against the leaf's true code.

    Each head is supervised by the deterministic attribute value implied by the
    gold leaf. Note this trains the heads to be individually correct, which is
    NOT the same as making the renormalized leaf posterior correct -- hence the
    marginal objective is primary and this is auxiliary. Reporting both, as P4.1
    requires, is what shows which one the gain depends on.
    """
    total = attribute_logits[0].new_zeros(())
    for attr_j, logits in enumerate(attribute_logits):
        target_value = code_index[leaf_target, attr_j]  # (batch,)
        total = total + F.cross_entropy(logits, target_value)
    return total


def random_code_index(seed: int) -> torch.Tensor:
    """A RANDOM but still-unique code assignment, for the P4.2 corruption control.

    THIS CONTROL IS THE STRONGEST EVIDENCE EITHER WAY AND MUST NOT BE SKIPPED.
    It preserves the number of distinct values per attribute and the uniqueness
    of all nine codes, discarding only the SEMANTIC content of the assignment. If
    a random code performs as well as the semantic one, the mechanism is
    regularization rather than structure, and C1 as argued is dead -- which is a
    publishable finding, reported as such rather than buried.

    Implemented as a permutation of the ROWS of the real code matrix, which
    guarantees uniqueness and preserves every per-attribute value distribution
    exactly. A freshly sampled random table would do neither, and would confound
    the control with a change in attribute balance.
    """
    generator = torch.Generator().manual_seed(seed)
    real = torch.from_numpy(attribute_index_matrix()).long()
    permutation = torch.randperm(N_EVASION, generator=generator)
    return real[permutation]
