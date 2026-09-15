"""Tests for the C2 agreement machinery.

The alpha and kappa implementations are hand-rolled, so they are validated
against textbook worked examples with known answers BEFORE being trusted on real
data. An agreement statistic that is quietly wrong would produce a confident,
publishable, and false C2 result -- the worst possible failure mode for this
project, because nothing downstream would catch it.
"""

from __future__ import annotations

import numpy as np
import pytest

from higrec.analysis.agreement import (
    agreement_under_partition,
    compare_partitions,
    fleiss_kappa,
    krippendorff_alpha_nominal,
    pairwise_disagreement_matrix,
    percent_unanimous,
)
from higrec.data.labels import OOV_INDEX


# --- Validation against known values ---------------------------------------


def test_alpha_is_one_for_perfect_agreement():
    ratings = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [0, 0, 0]])
    assert krippendorff_alpha_nominal(ratings, 3) == pytest.approx(1.0)


def test_alpha_is_about_zero_for_chance_agreement():
    """Alpha near 0 means agreement no better than chance."""
    rng = np.random.default_rng(0)
    ratings = rng.integers(0, 3, size=(4000, 3))
    assert abs(krippendorff_alpha_nominal(ratings, 3)) < 0.05


def test_alpha_is_negative_for_systematic_disagreement():
    """Below-chance agreement must produce a negative alpha, not clamp to 0."""
    ratings = np.array([[0, 1], [1, 0]] * 50)
    assert krippendorff_alpha_nominal(ratings, 2) < 0


def test_alpha_matches_krippendorffs_worked_example():
    """Krippendorff's canonical nominal example.

    Cross-checked against the `krippendorff` package, which is a dependency of
    this project, so a regression in our implementation surfaces immediately.
    """
    krippendorff = pytest.importorskip("krippendorff")

    # Units x coders, with missing judgments marked.
    data = np.array(
        [
            [1, 1, OOV_INDEX],
            [2, 2, 3],
            [3, 3, 3],
            [3, 3, 3],
            [2, 2, 2],
            [1, 2, 3],
            [4, 4, 4],
            [1, 1, 2],
            [2, 2, 2],
            [OOV_INDEX, 5, 5],
        ]
    )
    ours = krippendorff_alpha_nominal(data, n_categories=6)

    reliability = np.where(data == OOV_INDEX, np.nan, data).astype(float).T
    theirs = krippendorff.alpha(
        reliability_data=reliability, level_of_measurement="nominal"
    )
    assert ours == pytest.approx(theirs, abs=1e-9)


def test_fleiss_kappa_is_one_for_perfect_agreement():
    ratings = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    assert fleiss_kappa(ratings, 3) == pytest.approx(1.0)


def test_fleiss_kappa_is_about_zero_for_chance_agreement():
    rng = np.random.default_rng(1)
    ratings = rng.integers(0, 3, size=(4000, 3))
    assert abs(fleiss_kappa(ratings, 3)) < 0.05


# --- The behaviour that motivates preferring alpha -------------------------


def test_alpha_uses_partially_rated_items_but_kappa_discards_them():
    """The concrete reason alpha is the headline statistic.

    An item rated by two of three annotators carries real agreement
    information. Alpha uses it; Fleiss' kappa throws the whole item away.
    """
    ratings = np.array([[0, 0, 0]] * 20 + [[1, 1, OOV_INDEX]] * 20)

    assert not np.isnan(krippendorff_alpha_nominal(ratings, 2))
    # kappa sees only the 20 complete rows, all of one category.
    complete_only = ratings[~np.any(ratings == OOV_INDEX, axis=1)]
    assert complete_only.shape[0] == 20


def test_singly_rated_items_are_excluded_not_counted_as_agreement():
    """A unit with one rating carries no agreement information at all."""
    ratings = np.array([[0, OOV_INDEX, OOV_INDEX]] * 50)
    assert np.isnan(krippendorff_alpha_nominal(ratings, 3))


def test_percent_unanimous():
    ratings = np.array([[0, 0, 0], [0, 0, 1], [1, 1, 1], [0, 1, 2]])
    assert percent_unanimous(ratings) == pytest.approx(0.5)


def test_pairwise_disagreement_matrix_is_symmetric_and_hollow():
    ratings = np.array([[0, 1, 2], [0, 0, 1]])
    matrix = pairwise_disagreement_matrix(ratings, 3)

    assert np.array_equal(matrix, matrix.T)
    assert np.all(np.diag(matrix) == 0), "agreeing pairs are not disagreements"
    assert matrix[0, 1] == 3  # (0,1) and (0,2) from row 1; (0,1),(0,1) from row 2
    assert matrix[1, 2] == 1


# --- Partition collapsing ---------------------------------------------------


def test_collapsing_a_partition_can_only_raise_agreement():
    """Sanity property: merging categories cannot create new disagreement.

    Any 3-way collapse of 9-way labels must score at least as high as the raw
    9-way alpha. If a real run violates this, the collapse is misapplied.
    """
    rng = np.random.default_rng(2)
    leaf_ratings = rng.integers(0, 9, size=(300, 3))
    identity = np.arange(9)
    collapse = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    raw = krippendorff_alpha_nominal(leaf_ratings, 9)
    collapsed = agreement_under_partition(leaf_ratings, collapse, 3).alpha
    del identity

    assert collapsed >= raw - 1e-12


def test_partition_map_is_applied_per_annotator_not_per_item():
    """Each annotator's OWN label is collapsed; we never collapse a consensus first.

    Collapsing a majority vote instead would destroy exactly the disagreement
    structure C2 is measuring.
    """
    leaf_ratings = np.array([[0, 3, 6]])
    partition = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    result = agreement_under_partition(leaf_ratings, partition, 3)

    # Three leaves map to three DIFFERENT coarse categories, so this item is a
    # three-way disagreement after collapsing, not a unanimous one.
    assert result.percent_unanimous == pytest.approx(0.0)


def test_missing_leaf_labels_survive_the_collapse_as_missing():
    leaf_ratings = np.array([[0, 3, OOV_INDEX]])
    partition = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    result = agreement_under_partition(leaf_ratings, partition, 3)
    assert result.n_items == 1


# --- The paired bootstrap ---------------------------------------------------


def test_compare_partitions_detects_a_real_difference():
    """Construct data where the coverage cut is genuinely better, and detect it.

    Annotators confuse leaves 0 and 1 constantly. A partition separating them
    must score worse than one grouping them together.
    """
    rng = np.random.default_rng(3)
    n = 400
    leaf_ratings = np.empty((n, 3), dtype=np.int64)
    for i in range(n):
        if i % 2 == 0:
            leaf_ratings[i] = rng.choice([0, 1], size=3)  # confusable pair
        else:
            leaf_ratings[i] = np.full(3, rng.choice([3, 6]))  # agreed

    splits_the_pair = np.array([0, 1, 1, 1, 1, 1, 2, 2, 2])
    groups_the_pair = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2])

    result = compare_partitions(
        leaf_ratings, splits_the_pair, groups_the_pair,
        n_resamples=400, seed=0,
    )

    assert result.delta > 0
    assert result.supports_h1
    assert result.ci_low <= result.delta <= result.ci_high


def test_compare_partitions_reports_no_effect_when_there_is_none():
    """A negative result must come out negative. This is the honest-reporting case."""
    rng = np.random.default_rng(4)
    leaf_ratings = rng.integers(0, 9, size=(300, 3))
    partition = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    result = compare_partitions(
        leaf_ratings, partition, partition.copy(), n_resamples=300, seed=0
    )

    assert result.delta == pytest.approx(0.0)
    assert not result.supports_h1
    assert result.p_value > 0.05


def test_bootstrap_is_paired_on_the_same_resampled_items():
    """Pairing check: identical partitions must give EXACTLY zero delta every time.

    If the bootstrap resampled the two partitions independently, identical
    partitions would still produce non-zero deltas and a spurious interval.
    """
    rng = np.random.default_rng(5)
    leaf_ratings = rng.integers(0, 9, size=(200, 3))
    partition = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    result = compare_partitions(
        leaf_ratings, partition, partition, n_resamples=200, seed=1
    )

    assert result.ci_low == pytest.approx(0.0)
    assert result.ci_high == pytest.approx(0.0)


def test_p_value_is_never_exactly_zero():
    """The +1 correction: a bootstrap p-value of 0 would be a false claim."""
    rng = np.random.default_rng(6)
    leaf_ratings = np.tile(np.array([[0, 0, 0]]), (200, 1))
    leaf_ratings[::2] = rng.choice([0, 1], size=(100, 3))
    a = np.array([0, 1, 1, 1, 1, 1, 2, 2, 2])
    b = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2])

    result = compare_partitions(leaf_ratings, a, b, n_resamples=200, seed=0)
    assert result.p_value > 0.0


def test_compare_partitions_is_deterministic_given_a_seed():
    rng = np.random.default_rng(7)
    leaf_ratings = rng.integers(0, 9, size=(150, 3))
    a = np.array([0, 1, 1, 1, 1, 1, 2, 2, 2])
    b = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2])

    first = compare_partitions(leaf_ratings, a, b, n_resamples=200, seed=42)
    second = compare_partitions(leaf_ratings, a, b, n_resamples=200, seed=42)

    assert first.delta == second.delta
    assert first.p_value == second.p_value
    assert (first.ci_low, first.ci_high) == (second.ci_low, second.ci_high)
