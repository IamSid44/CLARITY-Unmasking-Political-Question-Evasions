"""C2: does a coverage-derived coarse cut raise inter-annotator agreement?

THE KEY PROPERTY OF THIS TEST: it requires NO NEW ANNOTATION. Because the
coverage attribute is a deterministic function of the existing leaf label, each
of the three annotators' existing 9-way judgments can be collapsed under two
different 3-way partitions and the resulting agreement compared. It is a
re-analysis of data we already have, it needs no GPU, and it is the
highest-value/lowest-risk result in the project.

WHAT IT CAN AND CANNOT SHOW -- state this in the paper, do not let a reader
infer it. This tests whether RELABELLING EXISTING 9-way judgments under a
different collapse raises agreement. It does NOT test whether annotators asked
DIRECTLY about coverage would agree more. That question needs the re-annotation
campaign, and the two must never be conflated.

Headline statistic is Krippendorff's alpha rather than Fleiss' kappa, because it
handles incomplete and variable-rater designs and missing judgments, which
Fleiss' kappa does not. Fleiss' kappa is reported alongside for comparability
with the organizers' published figures (kappa = 0.64 three-way, 0.48 nine-way).

Pre-registration (CLAUDE.md invariant 7): reports/06_c2_preregistration.md is
written and committed BEFORE any function here is run on real data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from higrec.data.labels import OOV_INDEX

__all__ = [
    "AgreementResult",
    "PartitionComparison",
    "krippendorff_alpha_nominal",
    "fleiss_kappa",
    "percent_unanimous",
    "pairwise_disagreement_matrix",
    "agreement_under_partition",
    "compare_partitions",
]


@dataclass(frozen=True)
class AgreementResult:
    """Agreement statistics for one labelling of the annotator matrix."""

    alpha: float
    """Krippendorff's alpha, nominal. The headline statistic."""
    kappa: float
    """Fleiss' kappa, for comparability with published figures."""
    percent_unanimous: float
    n_items: int
    n_categories: int


@dataclass(frozen=True)
class PartitionComparison:
    """The C2 result: coverage cut versus official cut, with a paired bootstrap."""

    official: AgreementResult
    coverage: AgreementResult
    delta: float
    """alpha(coverage) - alpha(official). Positive supports H1."""
    ci_low: float
    ci_high: float
    p_value: float
    n_resamples: int

    @property
    def supports_h1(self) -> bool:
        """H1 at the pre-registered threshold: delta > 0 and p < 0.05 two-sided."""
        return self.delta > 0 and self.p_value < 0.05


def _coincidence_matrix(ratings: np.ndarray, n_categories: int) -> np.ndarray:
    """Krippendorff coincidence matrix over units with >= 2 valid ratings.

    `ratings` is (n_items, n_raters) with OOV_INDEX marking a missing judgment.
    Each unit contributes its rating pairs weighted by 1/(m_u - 1), which is what
    makes alpha tolerate a variable number of raters per unit.
    """
    coincidence = np.zeros((n_categories, n_categories), dtype=np.float64)
    for row in ratings:
        valid = row[row != OOV_INDEX]
        m_u = valid.size
        if m_u < 2:
            continue  # a unit rated once carries no agreement information
        for i in range(m_u):
            for j in range(m_u):
                if i != j:
                    coincidence[valid[i], valid[j]] += 1.0 / (m_u - 1)
    return coincidence


def krippendorff_alpha_nominal(ratings: np.ndarray, n_categories: int) -> float:
    """Krippendorff's alpha with the nominal difference function.

    alpha = 1 - D_observed / D_expected, computed from the coincidence matrix so
    that units with differing numbers of raters are handled correctly.

    Returns 1.0 when there is no observed disagreement, and 0.0 when the expected
    disagreement is degenerate (a single category everywhere), rather than
    raising -- a bootstrap resample can legitimately produce either.
    """
    coincidence = _coincidence_matrix(np.asarray(ratings), n_categories)
    n_total = coincidence.sum()
    if n_total == 0:
        return float("nan")

    marginals = coincidence.sum(axis=1)
    # Observed disagreement: off-diagonal mass.
    observed = n_total - np.trace(coincidence)
    # Expected disagreement under independent assignment with the same marginals.
    expected = (n_total**2 - np.sum(marginals**2)) / (n_total - 1) if n_total > 1 else 0.0

    if expected <= 0:
        return 1.0 if observed == 0 else 0.0
    return float(1.0 - observed / expected)


def fleiss_kappa(ratings: np.ndarray, n_categories: int) -> float:
    """Fleiss' kappa. Requires a fixed number of raters per item.

    Reported only for comparability with the organizers' published values; alpha
    is the headline statistic. Items with any missing rating are dropped, which
    is precisely the limitation that motivates preferring alpha.
    """
    ratings = np.asarray(ratings)
    complete = ratings[~np.any(ratings == OOV_INDEX, axis=1)]
    n_items, n_raters = complete.shape
    if n_items == 0 or n_raters < 2:
        return float("nan")

    counts = np.zeros((n_items, n_categories), dtype=np.float64)
    for i, row in enumerate(complete):
        for value in row:
            counts[i, value] += 1

    # Per-item agreement, averaged.
    p_i = (np.sum(counts**2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = float(np.mean(p_i))
    # Chance agreement from category marginals.
    p_j = counts.sum(axis=0) / (n_items * n_raters)
    p_e = float(np.sum(p_j**2))

    if p_e >= 1.0:
        return 1.0 if p_bar >= 1.0 else 0.0
    return (p_bar - p_e) / (1.0 - p_e)


def percent_unanimous(ratings: np.ndarray) -> float:
    """Fraction of items on which all valid raters agree."""
    ratings = np.asarray(ratings)
    unanimous = 0
    counted = 0
    for row in ratings:
        valid = row[row != OOV_INDEX]
        if valid.size < 2:
            continue
        counted += 1
        unanimous += int(np.unique(valid).size == 1)
    return unanimous / counted if counted else float("nan")


def pairwise_disagreement_matrix(ratings: np.ndarray, n_categories: int) -> np.ndarray:
    """(C, C) symmetric counts of how often categories i and j co-occur in disagreement.

    Used to check whether any partition effect is driven by one class rather than
    being a general property of the cut -- a distinction the report must make.
    """
    matrix = np.zeros((n_categories, n_categories), dtype=np.int64)
    for row in np.asarray(ratings):
        valid = row[row != OOV_INDEX]
        for i in range(valid.size):
            for j in range(i + 1, valid.size):
                a, b = int(valid[i]), int(valid[j])
                if a != b:
                    matrix[a, b] += 1
                    matrix[b, a] += 1
    return matrix


def agreement_under_partition(
    leaf_ratings: np.ndarray,
    partition_map: np.ndarray,
    n_categories: int,
) -> AgreementResult:
    """Collapse each annotator's leaf label through `partition_map`, then score.

    Parameters
    ----------
    leaf_ratings
        (n_items, n_raters) leaf indices, OOV_INDEX for missing.
    partition_map
        (n_leaves,) leaf index -> coarse category index.
    """
    leaf_ratings = np.asarray(leaf_ratings, dtype=np.int64)
    collapsed = np.full(leaf_ratings.shape, OOV_INDEX, dtype=np.int64)
    known = leaf_ratings != OOV_INDEX
    collapsed[known] = np.asarray(partition_map, dtype=np.int64)[leaf_ratings[known]]

    return AgreementResult(
        alpha=krippendorff_alpha_nominal(collapsed, n_categories),
        kappa=fleiss_kappa(collapsed, n_categories),
        percent_unanimous=percent_unanimous(collapsed),
        n_items=int(leaf_ratings.shape[0]),
        n_categories=n_categories,
    )


def compare_partitions(
    leaf_ratings: np.ndarray,
    official_map: np.ndarray,
    coverage_map: np.ndarray,
    *,
    n_categories: int = 3,
    n_resamples: int = 10_000,
    seed: int = 0,
    progress: Callable[[int], None] | None = None,
) -> PartitionComparison:
    """The pre-registered C2 test: paired bootstrap over ITEMS.

    Resampling is over items, and BOTH partitions are recomputed on the SAME
    resampled items each iteration. That pairing is essential: the two alphas are
    computed from the very same annotator judgments, so an unpaired test would
    badly overstate the variance of their difference.

    The p-value is two-sided, computed by inverting the bootstrap distribution of
    the delta about zero, with the conventional +1 correction so that it can
    never be reported as exactly 0.

    Returns the point estimate, a 95% percentile interval, and the p-value,
    whichever direction the result falls in. Per the pre-registration we report a
    negative result with the same care as a positive one.
    """
    leaf_ratings = np.asarray(leaf_ratings, dtype=np.int64)
    n_items = leaf_ratings.shape[0]
    rng = np.random.default_rng(seed)

    official = agreement_under_partition(leaf_ratings, official_map, n_categories)
    coverage = agreement_under_partition(leaf_ratings, coverage_map, n_categories)
    delta = coverage.alpha - official.alpha

    deltas = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        idx = rng.integers(0, n_items, size=n_items)
        resampled = leaf_ratings[idx]
        a_off = agreement_under_partition(resampled, official_map, n_categories).alpha
        a_cov = agreement_under_partition(resampled, coverage_map, n_categories).alpha
        deltas[b] = a_cov - a_off
        if progress is not None and (b + 1) % 1000 == 0:
            progress(b + 1)

    finite = deltas[np.isfinite(deltas)]
    ci_low, ci_high = np.percentile(finite, [2.5, 97.5])

    # Two-sided p: how often does the centred bootstrap delta exceed the observed
    # magnitude? +1 in numerator and denominator is the standard correction.
    centred = finite - finite.mean()
    n_extreme = int(np.sum(np.abs(centred) >= abs(delta)))
    p_value = (n_extreme + 1) / (finite.size + 1)

    return PartitionComparison(
        official=official,
        coverage=coverage,
        delta=float(delta),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float(p_value),
        n_resamples=int(finite.size),
    )


def raw_nine_way_alpha(leaf_ratings: np.ndarray, n_leaves: int = 9) -> float:
    """Alpha on the uncollapsed 9-way labels, as a reference point.

    Both 3-way alphas should exceed this; if one does not, the collapse is not
    doing what a coarsening is supposed to do and something is wrong upstream.
    """
    return krippendorff_alpha_nominal(np.asarray(leaf_ratings), n_leaves)
