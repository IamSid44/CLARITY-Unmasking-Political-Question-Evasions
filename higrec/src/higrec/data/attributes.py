"""C1: the five-attribute code over the nine evasion leaves.

The central claim of C1, stated honestly up front because the code must not be
allowed to look like more than it is:

    Because the attribute code is a DETERMINISTIC FUNCTION of the leaf label,
    the factorized parameterization has IDENTICAL EXPRESSIVE CAPACITY to a flat
    9-way softmax in the infinite-data limit. It is a reparameterization, not a
    capacity increase. Its only possible value is sample efficiency: a rare leaf
    discriminates on top of attribute values estimated from thousands of
    examples rather than from its own handful.

That is a hypothesis this project TESTS, not a property it asserts. The decisive
experiment is the random-code control in Phase 4.2: if a randomly permuted but
still-unique code performs as well as the semantic one, the mechanism is
regularization rather than structure, and C1 as argued is dead.

The five attributes
-------------------
coverage        how much of the requested information arrived
engagement      whether the answer stays on target, pivots, or ignores
form            whether delivery is stated outright or merely implied
marker          whether there is an overt signal of non-delivery
incompleteness  whether what is missing is components or specificity

Design tension, deliberate and worth stating: Dodging and Deflection differ only
on `form`; Declining / Claims-ignorance / Clarification differ only on
`incompleteness`. Those single-attribute distinctions are structurally fragile,
and reports/05_code_analysis.md enumerates every such pair. The tension is the
point -- it forces probability mass onto shared attribute values that rare leaves
inherit from common ones.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from higrec.data.labels import CLARITY_LABELS, EVASION_LABELS, N_EVASION

__all__ = [
    "CODE_VERSION",
    "ATTRIBUTES",
    "ATTRIBUTE_VALUES",
    "ATTRIBUTE_CODE",
    "COVERAGE_CLARITY_OF_LEAF",
    "code_matrix",
    "attribute_index_matrix",
    "coverage_partition_map",
    "leaves_moved_between_partitions",
]

# Version-stamped so that any experiment artifact can record which code table
# produced it. Bump on ANY change to ATTRIBUTE_CODE; never edit silently.
CODE_VERSION: Final[str] = "c1-attribute-code-v1"

ATTRIBUTES: Final[tuple[str, ...]] = (
    "coverage",
    "engagement",
    "form",
    "marker",
    "incompleteness",
)

# Value order within each attribute fixes the column order of that attribute's
# head. Load-bearing for reproducibility; do not reorder.
ATTRIBUTE_VALUES: Final[dict[str, tuple[str, ...]]] = {
    "coverage": ("full", "partial", "none"),
    "engagement": ("on-target", "pivots", "ignores"),
    "form": ("stated", "implied"),
    "marker": ("absent", "present"),
    "incompleteness": ("none", "missing-components", "missing-specificity"),
}

# The frozen leaf -> code table. Nine leaves, five attributes, all codes unique.
ATTRIBUTE_CODE: Final[dict[str, dict[str, str]]] = {
    "Explicit": {
        "coverage": "full", "engagement": "on-target", "form": "stated",
        "marker": "absent", "incompleteness": "none",
    },
    "Implicit": {
        "coverage": "full", "engagement": "on-target", "form": "implied",
        "marker": "absent", "incompleteness": "none",
    },
    "Partial/half-answer": {
        "coverage": "partial", "engagement": "on-target", "form": "stated",
        "marker": "absent", "incompleteness": "missing-components",
    },
    "General": {
        "coverage": "partial", "engagement": "on-target", "form": "stated",
        "marker": "absent", "incompleteness": "missing-specificity",
    },
    "Dodging": {
        "coverage": "none", "engagement": "pivots", "form": "stated",
        "marker": "absent", "incompleteness": "missing-components",
    },
    "Deflection": {
        "coverage": "none", "engagement": "pivots", "form": "implied",
        "marker": "absent", "incompleteness": "missing-components",
    },
    "Declining to answer": {
        "coverage": "none", "engagement": "ignores", "form": "stated",
        "marker": "present", "incompleteness": "none",
    },
    "Claims ignorance": {
        "coverage": "none", "engagement": "ignores", "form": "stated",
        "marker": "present", "incompleteness": "missing-components",
    },
    "Clarification": {
        "coverage": "none", "engagement": "ignores", "form": "stated",
        "marker": "present", "incompleteness": "missing-specificity",
    },
}

# --- The coverage-derived 3-way partition (C2's alternative cut) -----------
#
# Read directly off the `coverage` attribute:
#     full -> Clear Reply | partial -> Ambivalent | none -> Clear Non-Reply
#
# Stored separately from the official partition in labels.py so the two can
# never be confused. C2 tests whether inter-annotator agreement is higher under
# this cut than under the official one.

_COVERAGE_TO_CLARITY: Final[dict[str, str]] = {
    "full": "Clear Reply",
    "partial": "Ambivalent",
    "none": "Clear Non-Reply",
}

COVERAGE_CLARITY_OF_LEAF: Final[dict[str, str]] = {
    leaf: _COVERAGE_TO_CLARITY[code["coverage"]] for leaf, code in ATTRIBUTE_CODE.items()
}


def attribute_index_matrix() -> np.ndarray:
    """(9, 5) integer matrix: row = leaf index, col = attribute, value = value index.

    This is the lookup the factorized head uses to gather each leaf's attribute
    log-probabilities before renormalizing over the nine valid codes.
    """
    out = np.zeros((N_EVASION, len(ATTRIBUTES)), dtype=np.int64)
    for leaf_i, leaf in enumerate(EVASION_LABELS):
        for attr_j, attr in enumerate(ATTRIBUTES):
            value = ATTRIBUTE_CODE[leaf][attr]
            out[leaf_i, attr_j] = ATTRIBUTE_VALUES[attr].index(value)
    return out


def code_matrix() -> tuple[tuple[str, ...], ...]:
    """The code table as a tuple of rows, in EVASION_LABELS index order."""
    return tuple(
        tuple(ATTRIBUTE_CODE[leaf][attr] for attr in ATTRIBUTES) for leaf in EVASION_LABELS
    )


def coverage_partition_map() -> np.ndarray:
    """(9,) leaf index -> clarity index under the COVERAGE-derived partition."""
    return np.asarray(
        [CLARITY_LABELS.index(COVERAGE_CLARITY_OF_LEAF[leaf]) for leaf in EVASION_LABELS],
        dtype=np.int64,
    )


def leaves_moved_between_partitions() -> dict[str, tuple[str, str]]:
    """Leaves assigned differently by the two partitions: leaf -> (official, coverage).

    Expected to be exactly three leaves -- Implicit, Dodging, Deflection -- which
    together are ~45.7% of the training data. That mass is what gives the C2 test
    its statistical power, and `tests/test_attributes.py` pins it so a silent
    edit to either partition cannot go unnoticed.
    """
    from higrec.data.labels import OFFICIAL_CLARITY_OF_LEAF

    moved: dict[str, tuple[str, str]] = {}
    for leaf in EVASION_LABELS:
        official = OFFICIAL_CLARITY_OF_LEAF[leaf]
        coverage = COVERAGE_CLARITY_OF_LEAF[leaf]
        if official != coverage:
            moved[leaf] = (official, coverage)
    return moved


# --- Sensitivity analysis support (P2.2 step 6) ----------------------------
#
# The coverage value is genuinely arguable for three leaves, and C2's conclusion
# must be shown to survive -- or must be reported as flipping -- under defensible
# alternatives. These are the perturbations P2.2 requires, declared here rather
# than improvised at analysis time so they cannot be chosen after seeing results.

COVERAGE_PERTURBATIONS: Final[dict[str, dict[str, str]]] = {
    # Partial/half-answer delivers SOME requested information, so one could argue
    # it belongs with the full-coverage replies rather than the partial ones.
    "partial_as_full": {"Partial/half-answer": "full"},
    # General is vague but on-target; one could argue it delivers nothing
    # specific enough to count as partial coverage.
    "general_as_none": {"General": "none"},
    # Deflection pivots but does engage the topic; one could argue it delivers
    # partial rather than zero coverage.
    "deflection_as_partial": {"Deflection": "partial"},
}


def perturbed_coverage_partition(name: str) -> np.ndarray:
    """(9,) leaf -> clarity index under a named coverage perturbation."""
    if name not in COVERAGE_PERTURBATIONS:
        raise KeyError(
            f"unknown perturbation {name!r}; expected one of "
            f"{sorted(COVERAGE_PERTURBATIONS)}"
        )
    overrides = COVERAGE_PERTURBATIONS[name]
    out = []
    for leaf in EVASION_LABELS:
        coverage = overrides.get(leaf, ATTRIBUTE_CODE[leaf]["coverage"])
        out.append(CLARITY_LABELS.index(_COVERAGE_TO_CLARITY[coverage]))
    return np.asarray(out, dtype=np.int64)
