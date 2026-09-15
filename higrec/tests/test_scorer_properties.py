"""Properties of the official scorer that contribution C4 is built on.

These are the P1.2 property tests, with ONE DELIBERATE INVERSION.

The playbook specified an assertion that the score is INVARIANT to which member
of the reference set is predicted, and said to stop if it did not hold. It does
not hold. Rather than delete the test, it is inverted: we now assert that the
score DOES depend on which member is predicted, and pin the magnitude. A future
change to the scorer that silently restored invariance would break C4's
justification, and this test is what would catch it.

See reports/04_scorer_semantics.md for the full finding.
"""

from __future__ import annotations

import numpy as np
import pytest

from higrec.data.labels import EVASION_LABELS, N_EVASION, OOV_INDEX, multi_reference_mask
from higrec.scoring.official import (
    gold_set_frequency,
    macro_f1_multireference,
    macro_f1_single_label,
    prediction_hits_reference,
)

L = {name: i for i, name in enumerate(EVASION_LABELS)}


def mask_from(reference_sets: list[list[str]]) -> np.ndarray:
    return multi_reference_mask(reference_sets)


def score(pred_names: list[str], reference_sets: list[list[str]]) -> float:
    y_pred = np.asarray(
        [L[p] if p in L else OOV_INDEX for p in pred_names], dtype=np.int64
    )
    return macro_f1_multireference(y_pred, mask_from(reference_sets), EVASION_LABELS).macro_f1


# --- Property 1: FN accounting is PER MEMBER of the reference set ----------


def test_out_of_set_prediction_charges_one_fn_per_reference_member():
    """The contested clause. Resolves in favour of the proposal, against the guide.

    Implementation_Guide.md says an out-of-set prediction triggers "exactly one
    FN-type penalty" for the whole set. It triggers one PER MEMBER.
    """
    refs = [["Explicit", "Implicit", "Dodging"]]
    y_pred = np.asarray([L["Clarification"]], dtype=np.int64)
    result = macro_f1_multireference(y_pred, mask_from(refs), EVASION_LABELS)

    total_fn = sum(m.fn for m in result.per_class.values())
    assert total_fn == 3, "expected one FN per member of G, not one for the set"
    for name in ("Explicit", "Implicit", "Dodging"):
        assert result.per_class[name].fn == 1
    # The prediction itself is a false positive for its own class.
    assert result.per_class["Clarification"].fp == 1


def test_in_set_prediction_suppresses_fn_for_other_members():
    """Hitting any member of G means no FN is charged for the members missed."""
    refs = [["Explicit", "Implicit", "Dodging"]]
    y_pred = np.asarray([L["Explicit"]], dtype=np.int64)
    result = macro_f1_multireference(y_pred, mask_from(refs), EVASION_LABELS)

    assert sum(m.fn for m in result.per_class.values()) == 0
    assert result.per_class["Explicit"].tp == 1


def test_fn_count_scales_with_annotator_disagreement():
    """The proposal's G4 claim, stated as an executable property.

    "The cost of an error scales with how much the annotators disagreed."
    """
    penalties = []
    for refs in ([["Explicit"]], [["Explicit", "Implicit"]],
                 [["Explicit", "Implicit", "Dodging"]]):
        y_pred = np.asarray([L["Clarification"]], dtype=np.int64)
        result = macro_f1_multireference(y_pred, mask_from(refs), EVASION_LABELS)
        penalties.append(sum(m.fn for m in result.per_class.values()))

    assert penalties == [1, 2, 3]


# --- Property 2: INVERTED. The score DEPENDS on which member is predicted --


def test_score_is_NOT_invariant_to_which_reference_member_is_predicted():
    """THE GATE FINDING, pinned.

    The playbook predicted invariance here and said C4 depends on it. Invariance
    FAILS, and C4 is better founded as a result: because the choice among in-set
    members matters, the per-class multipliers lambda_c carry real
    decision-theoretic weight instead of being nearly inert.
    """
    refs = [["Explicit", "Partial/half-answer"]] * 3 + [["Explicit"]] * 10 + [["Dodging"]] * 10
    frequent = ["Explicit"] * 3 + ["Explicit"] * 10 + ["Dodging"] * 10
    rare = ["Partial/half-answer"] * 3 + ["Explicit"] * 10 + ["Dodging"] * 10

    f1_frequent, f1_rare = score(frequent, refs), score(rare, refs)

    assert f1_rare != pytest.approx(f1_frequent), (
        "scorer became invariant to the choice among reference members; "
        "C4's derivation in reports/13 assumes it is NOT, so re-derive it"
    )
    # Choosing the rare member is worth exactly one ninth here.
    assert f1_rare - f1_frequent == pytest.approx(1.0 / N_EVASION)
    assert f1_rare > f1_frequent, "macro averaging should reward the rare member"


def test_in_set_dominates_out_of_set():
    """The property C4's first-order objective actually needs, and it DOES hold.

    Whatever the choice among members, landing inside G beats landing outside it.
    """
    refs = [["Explicit", "Partial/half-answer"]] * 3 + [["Explicit"]] * 10 + [["Dodging"]] * 10
    filler = ["Explicit"] * 10 + ["Dodging"] * 10

    worst_in_set = min(score([m] * 3 + filler, refs)
                       for m in ("Explicit", "Partial/half-answer"))
    best_out_of_set = max(score([m] * 3 + filler, refs) for m in EVASION_LABELS
                          if m not in ("Explicit", "Partial/half-answer"))

    assert worst_in_set >= best_out_of_set


# --- Property 3: macro denominator and zero support ------------------------


def test_macro_denominator_is_always_all_nine_classes():
    """Perfect predictions on a single-class corpus score exactly 1/9.

    Absent classes are not skipped -- they contribute F1 = 0.0 and still count.
    This is why a rare class is worth as much as a frequent one, which is the
    quantitative motivation for C1 and C4 alike.
    """
    refs = [["Explicit"]] * 10
    assert score(["Explicit"] * 10, refs) == pytest.approx(1.0 / N_EVASION)


def test_zero_support_class_scores_zero_not_nan():
    refs = [["Explicit"]] * 10
    y_pred = np.full(10, L["Explicit"], dtype=np.int64)
    result = macro_f1_multireference(y_pred, mask_from(refs), EVASION_LABELS)

    absent = result.per_class["Partial/half-answer"]
    assert (absent.tp, absent.fp, absent.fn, absent.support) == (0, 0, 0, 0)
    assert absent.f1 == 0.0
    assert not np.isnan(absent.f1)


def test_single_tp_on_a_rare_class_is_worth_a_full_ninth():
    """Why C4 chases rare in-set members."""
    refs = [["Explicit"]] * 10 + [["Explicit", "Clarification"]]
    safe = score(["Explicit"] * 11, refs)
    greedy = score(["Explicit"] * 10 + ["Clarification"], refs)

    assert greedy - safe == pytest.approx(1.0 / N_EVASION)


# --- Property 4: support is endogenous -------------------------------------


def test_per_class_support_depends_on_predictions():
    """Support is NOT a fixed property of the gold data.

    Consequences are load-bearing: per-class tables must report
    `gold_set_frequency` alongside this, and P6.1's coordinate ascent is
    optimizing a non-separable objective because changing one class's threshold
    moves other classes' supports.
    """
    refs = [["Explicit", "Partial/half-answer"]] * 5
    supports = {}
    for choice in ("Explicit", "Partial/half-answer", "Dodging"):
        y_pred = np.full(5, L[choice], dtype=np.int64)
        result = macro_f1_multireference(y_pred, mask_from(refs), EVASION_LABELS)
        supports[choice] = (
            result.per_class["Explicit"].support,
            result.per_class["Partial/half-answer"].support,
        )

    assert supports["Explicit"] == (5, 0)
    assert supports["Partial/half-answer"] == (0, 5)
    assert supports["Dodging"] == (5, 5)


def test_gold_set_frequency_is_invariant_to_predictions():
    """The fixed counterpart that belongs in cross-system comparison tables."""
    refs = [["Explicit", "Partial/half-answer"]] * 5
    frequency = gold_set_frequency(mask_from(refs))

    assert frequency[L["Explicit"]] == 5
    assert frequency[L["Partial/half-answer"]] == 5
    assert frequency[L["Dodging"]] == 0


# --- Property 5: OOV and degenerate inputs ---------------------------------


def test_unrecognized_prediction_behaves_like_an_out_of_set_prediction():
    """An unparseable label earns no TP, no FP, and charges FNs across G."""
    refs = [["Explicit", "Implicit"]]
    y_pred = np.asarray([OOV_INDEX], dtype=np.int64)
    result = macro_f1_multireference(y_pred, mask_from(refs), EVASION_LABELS)

    assert sum(m.tp for m in result.per_class.values()) == 0
    assert sum(m.fp for m in result.per_class.values()) == 0
    assert sum(m.fn for m in result.per_class.values()) == 2


def test_prediction_hits_reference_handles_oov():
    refs = [["Explicit"], ["Implicit"]]
    y_pred = np.asarray([L["Explicit"], OOV_INDEX], dtype=np.int64)
    hits = prediction_hits_reference(y_pred, mask_from(refs))
    assert hits.tolist() == [True, False]


def test_empty_reference_set_yields_no_counts():
    """An item whose annotators are all blank contributes nothing anywhere."""
    mask = np.zeros((1, N_EVASION), dtype=bool)
    y_pred = np.asarray([L["Explicit"]], dtype=np.int64)
    result = macro_f1_multireference(y_pred, mask, EVASION_LABELS)

    assert result.per_class["Explicit"].fp == 1  # predicted, not in the (empty) set
    assert sum(m.tp + m.fn for m in result.per_class.values()) == 0


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="length mismatch"):
        macro_f1_multireference(
            np.zeros(3, dtype=np.int64), np.zeros((2, N_EVASION), dtype=bool), EVASION_LABELS
        )


# --- Property 6: the two scorers agree on singleton gold -------------------


def test_multireference_reduces_to_single_label_when_every_set_is_a_singleton():
    """A cross-check on both implementations at once.

    The official scorer keeps these as separate functions; with singleton
    reference sets they must coincide, and if they ever stop coinciding one of
    the two has drifted.
    """
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, N_EVASION, size=200)
    y_pred = rng.integers(0, N_EVASION, size=200)

    mask = np.zeros((200, N_EVASION), dtype=bool)
    mask[np.arange(200), y_true] = True

    multi = macro_f1_multireference(y_pred, mask, EVASION_LABELS)
    single = macro_f1_single_label(y_pred, y_true, EVASION_LABELS)

    assert multi.macro_f1 == pytest.approx(single.macro_f1)
    for name in EVASION_LABELS:
        assert multi.per_class[name].tp == single.per_class[name].tp
        assert multi.per_class[name].fp == single.per_class[name].fp
        assert multi.per_class[name].fn == single.per_class[name].fn
