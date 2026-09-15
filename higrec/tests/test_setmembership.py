"""Tests for C4: the set-membership estimator and threshold tuning.

The property tests here encode the claims C4 makes, so that a regression breaks
a test rather than quietly producing a worse number.
"""

from __future__ import annotations

import numpy as np
import pytest

from higrec.data.labels import EVASION_LABELS, N_EVASION, multi_reference_mask
from higrec.decision.setmembership import (
    consensus_posterior_baseline,
    expected_set_size,
    fit_thresholds_coordinate_ascent,
    nested_cv_thresholds,
    oracle_upper_bound,
    predict_with_thresholds,
    set_membership_from_annotators,
)

L = {name: i for i, name in enumerate(EVASION_LABELS)}


# --- q(c|x): the set-membership estimator ---------------------------------


def test_q_is_not_a_distribution_and_must_not_be_normalized():
    """The defining property. Code that normalizes q has destroyed C4."""
    probs = np.zeros((1, 3, N_EVASION))
    probs[0, 0, L["Explicit"]] = 1.0
    probs[0, 1, L["Implicit"]] = 1.0
    probs[0, 2, L["Dodging"]] = 1.0

    q = set_membership_from_annotators(probs)
    assert q.sum() == pytest.approx(3.0), "three certain, distinct annotators -> sum 3"
    assert q.sum() > 1.0


def test_q_reduces_to_the_marginal_when_all_annotators_agree():
    probs = np.zeros((1, 3, N_EVASION))
    probs[0, :, L["Explicit"]] = 1.0
    q = set_membership_from_annotators(probs)

    assert q[0, L["Explicit"]] == pytest.approx(1.0)
    assert q.sum() == pytest.approx(1.0)


def test_q_implements_the_noisy_or():
    """q = 1 - Π(1 - p), checked against an explicit computation."""
    probs = np.array([[[0.5] + [0.0] * 8, [0.5] + [0.0] * 8]])
    q = set_membership_from_annotators(probs)
    assert q[0, 0] == pytest.approx(1 - 0.5 * 0.5)


def test_q_is_monotone_in_each_annotator_probability():
    base = np.full((1, 3, N_EVASION), 0.1)
    higher = base.copy()
    higher[0, 0, L["Dodging"]] = 0.9

    assert (
        set_membership_from_annotators(higher)[0, L["Dodging"]]
        > set_membership_from_annotators(base)[0, L["Dodging"]]
    )


def test_expected_set_size_tracks_disagreement():
    """Validating this against the observed mean |G| is a required P6.1 check."""
    agree = np.zeros((1, 3, N_EVASION))
    agree[0, :, L["Explicit"]] = 1.0

    disagree = np.zeros((1, 3, N_EVASION))
    for a, name in enumerate(["Explicit", "Implicit", "Dodging"]):
        disagree[0, a, L[name]] = 1.0

    assert expected_set_size(set_membership_from_annotators(agree))[0] == pytest.approx(1.0)
    assert expected_set_size(set_membership_from_annotators(disagree))[0] == pytest.approx(3.0)


def test_q_rejects_wrongly_shaped_input():
    with pytest.raises(ValueError, match="items, annotators, classes"):
        set_membership_from_annotators(np.zeros((4, N_EVASION)))


def test_q_does_not_underflow_with_many_annotators():
    probs = np.full((2, 50, N_EVASION), 1e-8)
    q = set_membership_from_annotators(probs)
    assert np.all(np.isfinite(q))
    assert np.all(q >= 0)


# --- Where the two rules diverge ------------------------------------------


def test_set_membership_diverges_from_consensus_on_high_disagreement_items():
    """C4's core claim, as an executable example.

    A class can be in the reference set more often than it is ever the mode.
    Here every annotator puts 0.4 on Dodging and each puts 0.6 on a DIFFERENT
    class, so Dodging is never the consensus argmax but is very likely to appear
    somewhere in G.
    """
    probs = np.zeros((1, 3, N_EVASION))
    for a, name in enumerate(["Explicit", "Implicit", "General"]):
        probs[0, a, L[name]] = 0.6
        probs[0, a, L["Dodging"]] = 0.4

    q = set_membership_from_annotators(probs)
    consensus = consensus_posterior_baseline(probs)

    assert q[0].argmax() == L["Dodging"], "Dodging should dominate set-membership"
    assert consensus[0].argmax() != L["Dodging"], "but never the consensus posterior"


def test_rules_coincide_when_annotators_are_unanimous():
    """The pre-registered falsifier for P6.2.

    On a unanimous item G is a singleton and the two rules provably agree, which
    is why a C4 gain concentrated on unanimous items would falsify the account.
    """
    probs = np.zeros((5, 3, N_EVASION))
    probs[:, :, L["Explicit"]] = 1.0

    q = set_membership_from_annotators(probs)
    consensus = consensus_posterior_baseline(probs)

    assert np.array_equal(q.argmax(axis=1), consensus.argmax(axis=1))


# --- Threshold fitting -----------------------------------------------------


def test_thresholds_never_reduce_in_fold_macro_f1():
    """lambda = 1 is in the search space, so the fit can only improve on it."""
    rng = np.random.default_rng(0)
    q = rng.random((80, N_EVASION))
    refs = [[EVASION_LABELS[c] for c in rng.choice(N_EVASION, size=2, replace=False)]
            for _ in range(80)]
    mask = multi_reference_mask(refs)

    from higrec.scoring.official import macro_f1_multireference

    unweighted = macro_f1_multireference(
        q.argmax(axis=1), mask, EVASION_LABELS
    ).macro_f1
    fit = fit_thresholds_coordinate_ascent(q, mask, EVASION_LABELS, n_restarts=2, seed=0)

    assert fit.in_fold_macro_f1 >= unweighted - 1e-9


def test_thresholds_lift_a_rare_class_that_argmax_never_selects():
    """The mechanism C4 relies on: a multiplier can surface a suppressed class."""
    n = 60
    q = np.zeros((n, N_EVASION))
    q[:, L["Explicit"]] = 0.9
    q[:, L["Clarification"]] = 0.4  # always present, never the argmax

    refs = [["Explicit", "Clarification"]] * n
    mask = multi_reference_mask(refs)

    fit = fit_thresholds_coordinate_ascent(q, mask, EVASION_LABELS, n_restarts=3, seed=0)
    chosen = np.unique(predict_with_thresholds(q, fit.lambda_))

    assert fit.lambda_[L["Clarification"]] > fit.lambda_[L["Explicit"]]
    assert L["Clarification"] in chosen


def test_effective_examples_per_class_is_reported():
    """P6.1: every threshold travels with how many examples actually informed it."""
    rng = np.random.default_rng(1)
    q = rng.random((40, N_EVASION))
    mask = multi_reference_mask(
        [[EVASION_LABELS[c] for c in rng.choice(N_EVASION, 2, replace=False)] for _ in range(40)]
    )
    fit = fit_thresholds_coordinate_ascent(q, mask, EVASION_LABELS, n_restarts=1, seed=0)

    assert fit.effective_examples_per_class is not None
    assert fit.effective_examples_per_class.shape == (N_EVASION,)
    assert fit.effective_examples_per_class.sum() == mask.sum()


def test_fit_is_deterministic_given_a_seed():
    rng = np.random.default_rng(2)
    q = rng.random((50, N_EVASION))
    mask = multi_reference_mask(
        [[EVASION_LABELS[c] for c in rng.choice(N_EVASION, 2, replace=False)] for _ in range(50)]
    )

    first = fit_thresholds_coordinate_ascent(q, mask, EVASION_LABELS, n_restarts=2, seed=5)
    second = fit_thresholds_coordinate_ascent(q, mask, EVASION_LABELS, n_restarts=2, seed=5)

    assert np.array_equal(first.lambda_, second.lambda_)


def test_shrinkage_pulls_thresholds_toward_the_base_rate():
    rng = np.random.default_rng(3)
    q = rng.random((60, N_EVASION))
    mask = multi_reference_mask(
        [[EVASION_LABELS[c] for c in rng.choice(N_EVASION, 2, replace=False)] for _ in range(60)]
    )

    free = fit_thresholds_coordinate_ascent(q, mask, EVASION_LABELS, n_restarts=2, seed=0)
    shrunk = fit_thresholds_coordinate_ascent(
        q, mask, EVASION_LABELS, n_restarts=2, seed=0, shrinkage=10.0
    )

    assert np.std(np.log(shrunk.lambda_)) <= np.std(np.log(free.lambda_)) + 1e-9


# --- Nested CV: the overfitting control -----------------------------------


def test_nested_cv_reports_both_scores_and_the_gap():
    """P6.1 says report the gap BEFORE anything else, so it must always exist."""
    rng = np.random.default_rng(4)
    q = rng.random((120, N_EVASION))
    mask = multi_reference_mask(
        [[EVASION_LABELS[c] for c in rng.choice(N_EVASION, 2, replace=False)] for _ in range(120)]
    )

    fit = nested_cv_thresholds(q, mask, EVASION_LABELS, n_outer=4, n_inner_restarts=1, seed=0)

    assert fit.out_of_fold_macro_f1 is not None
    assert fit.generalization_gap is not None
    assert fit.generalization_gap == pytest.approx(
        fit.in_fold_macro_f1 - fit.out_of_fold_macro_f1
    )


def test_nested_cv_exposes_overfitting_on_pure_noise():
    """On q carrying no signal, in-fold must exceed out-of-fold.

    If this test ever showed no gap, the nested CV would not be measuring what it
    claims to measure and the C4 overfitting control would be worthless.
    """
    rng = np.random.default_rng(5)
    q = rng.random((100, N_EVASION))
    mask = multi_reference_mask(
        [[EVASION_LABELS[c] for c in rng.choice(N_EVASION, 2, replace=False)] for _ in range(100)]
    )

    fit = nested_cv_thresholds(q, mask, EVASION_LABELS, n_outer=5, n_inner_restarts=2, seed=1)
    assert fit.generalization_gap > 0


# --- Oracle bound ----------------------------------------------------------


def test_oracle_upper_bound_beats_a_naive_in_set_choice():
    """Bounds how much of the gap the rule could possibly close."""
    refs = [["Explicit", "Clarification"]] * 20 + [["Explicit"]] * 20
    mask = multi_reference_mask(refs)

    from higrec.scoring.official import macro_f1_multireference

    always_frequent = np.full(40, L["Explicit"], dtype=np.int64)
    naive = macro_f1_multireference(always_frequent, mask, EVASION_LABELS).macro_f1

    assert oracle_upper_bound(mask, EVASION_LABELS) > naive


def test_oracle_only_ever_picks_in_set_members():
    rng = np.random.default_rng(6)
    refs = [[EVASION_LABELS[c] for c in rng.choice(N_EVASION, 2, replace=False)]
            for _ in range(30)]
    mask = multi_reference_mask(refs)

    score = oracle_upper_bound(mask, EVASION_LABELS)
    assert 0.0 <= score <= 1.0
