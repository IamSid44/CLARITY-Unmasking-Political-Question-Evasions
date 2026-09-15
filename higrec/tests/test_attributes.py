"""Tests for the C1 attribute code table (P2.1).

These tests do three jobs: they enforce the structural properties the
factorized head relies on, they pin the facts that C2's power depends on, and
they make a silent edit to the code table impossible to land unnoticed.
"""

from __future__ import annotations

import numpy as np
import pytest

from higrec.data.attributes import (
    ATTRIBUTE_CODE,
    ATTRIBUTE_VALUES,
    ATTRIBUTES,
    CODE_VERSION,
    COVERAGE_CLARITY_OF_LEAF,
    COVERAGE_PERTURBATIONS,
    attribute_index_matrix,
    code_matrix,
    coverage_partition_map,
    leaves_moved_between_partitions,
    perturbed_coverage_partition,
)
from higrec.data.labels import (
    CLARITY_LABELS,
    EVASION_LABELS,
    N_EVASION,
    OFFICIAL_CLARITY_OF_LEAF,
    OFFICIAL_PARTITION_MAP,
)


# --- Structural requirements of the factorization --------------------------


def test_every_leaf_has_a_complete_code():
    assert set(ATTRIBUTE_CODE) == set(EVASION_LABELS)
    for leaf, code in ATTRIBUTE_CODE.items():
        assert set(code) == set(ATTRIBUTES), f"{leaf} has attributes {set(code)}"
        for attr, value in code.items():
            assert value in ATTRIBUTE_VALUES[attr], f"{leaf}.{attr} = {value!r}"


def test_all_nine_codes_are_unique():
    """Non-negotiable: a duplicate code makes two leaves indistinguishable.

    The factorized posterior renormalizes over the nine valid codes, so two
    leaves sharing a code would be assigned identical probability by
    construction and neither could ever be recovered.
    """
    codes = code_matrix()
    assert len(set(codes)) == N_EVASION, "duplicate attribute code detected"


def test_every_declared_attribute_value_is_reachable():
    """A value no leaf uses would be dead capacity in that attribute's head."""
    unreachable = []
    for attr in ATTRIBUTES:
        used = {ATTRIBUTE_CODE[leaf][attr] for leaf in EVASION_LABELS}
        for value in ATTRIBUTE_VALUES[attr]:
            if value not in used:
                unreachable.append(f"{attr}={value}")
    assert not unreachable, f"unreachable attribute values: {unreachable}"


def test_attribute_index_matrix_round_trips():
    matrix = attribute_index_matrix()
    assert matrix.shape == (N_EVASION, len(ATTRIBUTES))
    for leaf_i, leaf in enumerate(EVASION_LABELS):
        for attr_j, attr in enumerate(ATTRIBUTES):
            value = ATTRIBUTE_VALUES[attr][matrix[leaf_i, attr_j]]
            assert value == ATTRIBUTE_CODE[leaf][attr]


def test_head_sizes_are_as_specified():
    """P4.1 specifies attribute heads of sizes 3, 3, 2, 2, 3."""
    assert [len(ATTRIBUTE_VALUES[a]) for a in ATTRIBUTES] == [3, 3, 2, 2, 3]


# --- The structurally fragile distinctions ---------------------------------


def test_single_attribute_distinctions_are_exactly_the_expected_pairs():
    """Leaf pairs separated by ONE attribute are the fragile ones.

    Flagged in reports/05_code_analysis.md as a stated design tension rather
    than hidden. If this set changes, the C1 pre-registration's per-leaf
    predictions were written against different structure and must be revisited.
    """
    fragile = set()
    for i, a in enumerate(EVASION_LABELS):
        for b in EVASION_LABELS[i + 1:]:
            differing = [x for x in ATTRIBUTES if ATTRIBUTE_CODE[a][x] != ATTRIBUTE_CODE[b][x]]
            if len(differing) == 1:
                # frozenset so the assertion does not depend on EVASION_LABELS order
                fragile.add((frozenset({a, b}), differing[0]))

    assert fragile == {
        (frozenset({"Explicit", "Implicit"}), "form"),
        (frozenset({"Dodging", "Deflection"}), "form"),
        (frozenset({"General", "Partial/half-answer"}), "incompleteness"),
        (frozenset({"Declining to answer", "Claims ignorance"}), "incompleteness"),
        (frozenset({"Declining to answer", "Clarification"}), "incompleteness"),
        (frozenset({"Claims ignorance", "Clarification"}), "incompleteness"),
    }


# --- The two partitions ----------------------------------------------------


def test_official_and_coverage_partitions_are_not_identical():
    """C2 is vacuous if the two cuts coincide."""
    assert OFFICIAL_CLARITY_OF_LEAF != COVERAGE_CLARITY_OF_LEAF
    assert not np.array_equal(OFFICIAL_PARTITION_MAP, coverage_partition_map())


def test_exactly_three_leaves_move_and_they_are_the_expected_ones():
    """Pins the finding that gives the C2 test its power.

    Implicit, Dodging and Deflection together are ~45.7% of training data, so
    the partition effect is measured on a large fraction of the corpus rather
    than turning on a rare-class technicality.
    """
    moved = leaves_moved_between_partitions()
    assert set(moved) == {"Implicit", "Dodging", "Deflection"}
    assert moved["Implicit"] == ("Ambivalent", "Clear Reply")
    assert moved["Dodging"] == ("Ambivalent", "Clear Non-Reply")
    assert moved["Deflection"] == ("Ambivalent", "Clear Non-Reply")


def test_official_partition_has_the_asymmetry_that_motivates_c2():
    """Clear Reply holds ONE leaf; Ambivalent holds FIVE spanning full/partial/none.

    This asymmetry is the proposal's central empirical claim about the taxonomy.
    """
    sizes = {c: sum(1 for v in OFFICIAL_CLARITY_OF_LEAF.values() if v == c)
             for c in CLARITY_LABELS}
    assert sizes["Clear Reply"] == 1
    assert sizes["Ambivalent"] == 5
    assert sizes["Clear Non-Reply"] == 3

    ambivalent = [l for l, c in OFFICIAL_CLARITY_OF_LEAF.items() if c == "Ambivalent"]
    coverages = {ATTRIBUTE_CODE[l]["coverage"] for l in ambivalent}
    assert coverages == {"full", "partial", "none"}, (
        "Ambivalent should span all three coverage values -- that is the "
        "'cut across the wrong axis' claim, stated as a test"
    )


def test_coverage_partition_is_a_function_of_the_coverage_attribute_alone():
    mapping = {"full": "Clear Reply", "partial": "Ambivalent", "none": "Clear Non-Reply"}
    for leaf in EVASION_LABELS:
        assert COVERAGE_CLARITY_OF_LEAF[leaf] == mapping[ATTRIBUTE_CODE[leaf]["coverage"]]


# --- Sensitivity perturbations (P2.2 step 6) -------------------------------


def test_perturbations_are_declared_in_advance_and_each_changes_something():
    """Declared in the module, not improvised after seeing the alpha result."""
    assert set(COVERAGE_PERTURBATIONS) == {
        "partial_as_full", "general_as_none", "deflection_as_partial"
    }
    baseline = coverage_partition_map()
    for name in COVERAGE_PERTURBATIONS:
        assert not np.array_equal(perturbed_coverage_partition(name), baseline), name


def test_unknown_perturbation_is_rejected():
    with pytest.raises(KeyError, match="unknown perturbation"):
        perturbed_coverage_partition("whatever_makes_the_result_significant")


# --- Version stamp ---------------------------------------------------------


def test_code_version_is_stamped():
    """Every artifact records which code table produced it."""
    assert CODE_VERSION == "c1-attribute-code-v1", (
        "CODE_VERSION changed; that is correct if ATTRIBUTE_CODE changed, but "
        "results produced under the previous version are no longer comparable"
    )
