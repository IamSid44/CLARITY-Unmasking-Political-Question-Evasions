"""Conformance of `src/higrec/scoring/official.py` against the reference scorer.

WHAT WE ARE CONFORMING TO, AND ITS LIMITS
-----------------------------------------
The reference implementation executed here is
`REF/semeval-2026-task6-camsr-cot/scripts/eval_competition.py`, which is the
1st-place team's (TeleAI's) local replica of the Codabench scorer. Its own
docstring describes it as "consistent with the Codabench scorer", and their
paper states they used the organizers' official scorer -- so this file is a
faithful replica written by the team that topped the leaderboard with it, but it
is NOT the organizers' source.

All six reference clones were searched for a second, independent implementation
of the multi-reference rule. There is none: every other participant repository
collapses multi-reference gold to a majority vote or a single annotator before
scoring. So there is exactly one implementation available to conform to, and
this limitation must be disclosed in the paper (risk R1 in
IMPLEMENTATION_PLAN.md). Obtaining the Codabench bundle would close it.

LOADING DISCIPLINE
------------------
CLAUDE.md invariant 1 forbids `src/` from importing anything in `REF/`. This test
file is not `src/`, and P1.2 explicitly requires running the official scorer for
conformance. The reference is loaded here by explicit file path via importlib --
never by putting `REF/` on the Python path -- so the dependency cannot leak into
library code. If the clone is absent, every test in this module skips rather
than failing, so a fresh checkout without `REF/` still has a green suite.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from higrec.data.labels import (
    CLARITY_LABELS,
    EVASION_LABELS,
    N_CLARITY,
    N_EVASION,
    OOV_INDEX,
)
from higrec.scoring.official import macro_f1_multireference, macro_f1_single_label

REF_SCORER_PATH = (
    Path(__file__).resolve().parents[1]
    / "REF"
    / "semeval-2026-task6-camsr-cot"
    / "scripts"
    / "eval_competition.py"
)

N_RANDOM_CONFIGURATIONS = 2000


def _load_reference_module():
    """Load the reference scorer by path, without touching sys.path."""
    spec = importlib.util.spec_from_file_location("_ref_scorer", REF_SCORER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load reference scorer from {REF_SCORER_PATH}")
    module = importlib.util.module_from_spec(spec)
    # Registration is required before exec: the module defines a dataclass, and
    # dataclasses resolves type annotations via sys.modules[cls.__module__].
    sys.modules["_ref_scorer"] = module
    spec.loader.exec_module(module)
    return module


pytestmark = pytest.mark.skipif(
    not REF_SCORER_PATH.exists(),
    reason=f"reference scorer not cloned at {REF_SCORER_PATH}; run scripts/clone_refs.sh",
)


@pytest.fixture(scope="module")
def reference():
    return _load_reference_module()


def _to_names(indices: np.ndarray, class_names: tuple[str, ...]) -> list[str]:
    """Indices to label strings; OOV becomes a string no class can match."""
    return [
        class_names[i] if i != OOV_INDEX else "__unparseable_model_output__"
        for i in indices.tolist()
    ]


def _to_sets(mask: np.ndarray, class_names: tuple[str, ...]) -> list[set[str]]:
    return [{class_names[c] for c in np.flatnonzero(row)} for row in mask]


def _random_configuration(
    rng: np.random.Generator, n_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a (predictions, reference-mask) pair, biased toward the degenerate.

    Roughly a third of configurations are drawn from deliberately awkward
    regimes -- single-class corpora, empty reference sets, OOV predictions,
    always-out-of-set predictions -- because that is where two implementations
    of a metric this fiddly actually diverge.
    """
    n_items = int(rng.integers(1, 40))
    regime = rng.integers(0, 6)

    mask = np.zeros((n_items, n_classes), dtype=bool)
    if regime == 0:
        # Every item a singleton reference set.
        mask[np.arange(n_items), rng.integers(0, n_classes, size=n_items)] = True
    elif regime == 1:
        # Every item three distinct references -- maximum disagreement.
        for i in range(n_items):
            for c in rng.choice(n_classes, size=min(3, n_classes), replace=False):
                mask[i, c] = True
    elif regime == 2:
        # Some reference sets deliberately EMPTY.
        for i in range(n_items):
            if rng.random() < 0.4:
                continue
            mask[i, rng.integers(0, n_classes)] = True
    elif regime == 3:
        # Single-class corpus: eight of nine classes have zero support.
        only = rng.integers(0, n_classes)
        mask[:, only] = True
    else:
        # General case: 1..3 references per item.
        for i in range(n_items):
            size = int(rng.integers(1, 4))
            for c in rng.choice(n_classes, size=min(size, n_classes), replace=False):
                mask[i, c] = True

    pred_regime = rng.integers(0, 5)
    if pred_regime == 0:
        y_pred = np.full(n_items, rng.integers(0, n_classes), dtype=np.int64)
    elif pred_regime == 1:
        # Always inside the reference set where one exists.
        y_pred = np.empty(n_items, dtype=np.int64)
        for i in range(n_items):
            members = np.flatnonzero(mask[i])
            y_pred[i] = rng.choice(members) if members.size else rng.integers(0, n_classes)
    elif pred_regime == 2:
        # Always OUTSIDE the reference set where possible.
        y_pred = np.empty(n_items, dtype=np.int64)
        for i in range(n_items):
            outside = np.flatnonzero(~mask[i])
            y_pred[i] = rng.choice(outside) if outside.size else rng.integers(0, n_classes)
    elif pred_regime == 3:
        # Sprinkle unparseable predictions.
        y_pred = rng.integers(0, n_classes, size=n_items).astype(np.int64)
        y_pred[rng.random(n_items) < 0.25] = OOV_INDEX
    else:
        y_pred = rng.integers(0, n_classes, size=n_items).astype(np.int64)

    return y_pred, mask


@pytest.mark.parametrize("n_classes,class_names", [(N_EVASION, EVASION_LABELS)])
def test_subtask2_conformance_randomized(reference, n_classes, class_names):
    """2000 randomized configurations must match the reference EXACTLY."""
    rng = np.random.default_rng(20260913)
    mismatches: list[str] = []

    for trial in range(N_RANDOM_CONFIGURATIONS):
        y_pred, mask = _random_configuration(rng, n_classes)

        ours = macro_f1_multireference(y_pred, mask, class_names)
        theirs_macro, theirs_per_class = reference._compute_macro_f1_multiref_official(
            classes=list(class_names),
            y_pred=_to_names(y_pred, class_names),
            y_gold_sets=_to_sets(mask, class_names),
        )

        if abs(ours.macro_f1 - theirs_macro) > 1e-12:
            mismatches.append(
                f"trial {trial}: macro {ours.macro_f1!r} != {theirs_macro!r} "
                f"(n={len(y_pred)})"
            )
            continue

        for name in class_names:
            mine, yours = ours.per_class[name], theirs_per_class[name]
            if (mine.tp, mine.fp, mine.fn) != (
                int(yours["tp"]), int(yours["fp"]), int(yours["fn"])
            ):
                mismatches.append(
                    f"trial {trial}: {name} counts "
                    f"({mine.tp},{mine.fp},{mine.fn}) != "
                    f"({int(yours['tp'])},{int(yours['fp'])},{int(yours['fn'])})"
                )

    assert not mismatches, "\n".join(mismatches[:10])


def test_subtask1_conformance_randomized(reference):
    """Subtask 1 against the reference's single-label path."""
    rng = np.random.default_rng(13092026)
    mismatches: list[str] = []

    for trial in range(N_RANDOM_CONFIGURATIONS):
        n_items = int(rng.integers(1, 40))
        y_true = rng.integers(0, N_CLARITY, size=n_items).astype(np.int64)
        y_pred = rng.integers(0, N_CLARITY, size=n_items).astype(np.int64)
        if rng.random() < 0.2:  # collapse to one class sometimes
            y_true[:] = rng.integers(0, N_CLARITY)

        ours = macro_f1_single_label(y_pred, y_true, CLARITY_LABELS)
        gold_sets = [{CLARITY_LABELS[c]} for c in y_true.tolist()]
        theirs_macro, _ = reference._compute_macro_f1_multilabel_ovr(
            classes=list(CLARITY_LABELS),
            y_pred=_to_names(y_pred, CLARITY_LABELS),
            y_gold_sets=gold_sets,
        )

        if abs(ours.macro_f1 - theirs_macro) > 1e-12:
            mismatches.append(f"trial {trial}: {ours.macro_f1!r} != {theirs_macro!r}")

    assert not mismatches, "\n".join(mismatches[:10])


def test_label_normalization_matches_reference(reference):
    """Our string canonicalization must agree with the reference's, exactly.

    A divergence here would silently turn valid predictions into false
    positives, which is the kind of bug that costs a submission several points
    and never raises an exception.
    """
    from higrec.data.labels import normalize_clarity, normalize_evasion

    probes = [
        "Explicit", "explicit", "  Explicit  ", "Explicit.", "Explicit,", "EXPLICIT",
        "Partial/half-answer", "partial/half answer", "Partial / half-answer",
        "Partial / half answer", "Partial\\/half-answer",
        "Claims ignorance", "claims   ignorance", "Declining to answer;",
        "Clarification:", "not a real label", "",
    ]
    for probe in probes:
        assert normalize_evasion(probe) == reference._normalize_task2(probe), probe

    for probe in ["Ambivalent", "ambivalent reply", "Clear Non-Reply", "clear nonreply",
                  "clear non reply", "Clear Reply.", "nonsense"]:
        assert normalize_clarity(probe) == reference._normalize_task1(probe), probe


def test_reference_agrees_that_fn_is_charged_per_member(reference):
    """Pin the contested clause directly against the reference implementation."""
    macro, per_class = reference._compute_macro_f1_multiref_official(
        classes=list(EVASION_LABELS),
        y_pred=["Clarification"],
        y_gold_sets=[{"Explicit", "Implicit", "Dodging"}],
    )
    assert sum(int(v["fn"]) for v in per_class.values()) == 3
