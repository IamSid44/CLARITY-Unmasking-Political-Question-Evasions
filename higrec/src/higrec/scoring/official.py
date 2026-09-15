"""Reimplementation of the official SemEval-2026 Task 6 scorers.

Pure functions over numpy arrays. No dependency on `REF/`; nothing here imports
from a reference clone, and this module is what every experiment scores against.

Semantics are documented and justified in reports/04_scorer_semantics.md. The
short version, because getting it wrong invalidates contribution C4:

Subtask 2 (multi-reference). For each class c, one-vs-rest over items i with
prediction p_i and reference set G_i:

    TP_c : p_i == c  and  c in G_i
    FP_c : p_i == c  and  c not in G_i
    FN_c : p_i != c  and  c in G_i  and  p_i not in G_i

Three consequences that are easy to get wrong and that this module is careful
about:

1. An out-of-set prediction is charged ONE FALSE NEGATIVE PER MEMBER of G, not
   one for the set. The `p_i not in G_i` guard is evaluated independently for
   every class, so when the prediction misses entirely it fires for all of them.
   Implementation_Guide.md asserts the opposite ("the whole set G triggers
   exactly one FN-type penalty"); the code disagrees with the guide and agrees
   with the proposal's G4. Verified empirically: |G| = 3 with an out-of-set
   prediction yields 3 FNs.

2. Macro-averaging is over ALL declared classes ALWAYS. A class absent from both
   gold and predictions contributes F1 = 0.0 and still counts in the
   denominator. Perfect predictions on a single-class corpus therefore score
   exactly 1/9. Every class is worth 1/9 of the final score no matter how rare,
   which is the quantitative case for C1 and C4 both.

3. Per-class SUPPORT (TP + FN) is ENDOGENOUS: it depends on the predictions, not
   only on the gold data. An item where the model predicts some other member of
   G_i contributes neither TP nor FN to class c and drops out of c's support
   entirely. Use `gold_set_frequency()` whenever you need the fixed,
   system-independent notion of how often a class appears in gold -- that is the
   number that belongs in a cross-system comparison table.

Subtask 1 is plain single-label macro-F1 over the three clarity classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from higrec.data.labels import N_EVASION, OOV_INDEX

__all__ = [
    "ClassMetrics",
    "ScoreResult",
    "macro_f1_multireference",
    "macro_f1_single_label",
    "gold_set_frequency",
    "prediction_hits_reference",
]


@dataclass(frozen=True)
class ClassMetrics:
    """Per-class one-vs-rest counts and derived rates."""

    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    support: int
    """TP + FN. NOTE: under the multi-reference scorer this is prediction-dependent.
    For a fixed, system-independent count use `gold_set_frequency()`."""


@dataclass(frozen=True)
class ScoreResult:
    """Macro-F1 together with the per-class breakdown it was averaged from."""

    macro_f1: float
    per_class: dict[str, ClassMetrics] = field(default_factory=dict)

    def f1_vector(self, class_names: tuple[str, ...]) -> np.ndarray:
        return np.asarray([self.per_class[c].f1 for c in class_names], dtype=np.float64)


def _safe_div(numerator: float, denominator: float) -> float:
    """Division guarded to 0.0, matching the official scorer.

    A zero denominator yields 0.0, not NaN and not a skipped class. This is the
    behaviour that makes an absent class contribute 0.0 to the macro average.
    """
    return numerator / denominator if denominator else 0.0


def _one_hot_predictions(y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """(N, C) boolean matrix; an OOV prediction produces an all-False row.

    The all-False row is exactly right: an unrecognized label matches no class,
    so it earns no TP and no FP, and it fails the reference-hit test below --
    which is precisely how the official scorer treats an unparseable prediction.
    """
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_pred.ndim != 1:
        raise ValueError(f"y_pred must be 1-D, got shape {y_pred.shape}")
    class_ids = np.arange(n_classes, dtype=np.int64)
    return y_pred[:, None] == class_ids[None, :]


def _validate(y_pred: np.ndarray, gold_mask: np.ndarray, n_classes: int) -> None:
    if gold_mask.ndim != 2:
        raise ValueError(f"gold_mask must be 2-D, got shape {gold_mask.shape}")
    if gold_mask.shape[1] != n_classes:
        raise ValueError(
            f"gold_mask has {gold_mask.shape[1]} columns, expected {n_classes}"
        )
    if len(y_pred) != gold_mask.shape[0]:
        raise ValueError(
            f"Pred/gold length mismatch: {len(y_pred)} vs {gold_mask.shape[0]}"
        )
    bad = np.asarray(y_pred)[(np.asarray(y_pred) < OOV_INDEX)]
    if bad.size:
        raise ValueError(f"y_pred contains invalid indices below {OOV_INDEX}: {bad[:5]}")


def prediction_hits_reference(y_pred: np.ndarray, gold_mask: np.ndarray) -> np.ndarray:
    """(N,) boolean: does item i's prediction land anywhere inside its reference set?

    This is the guard that makes the false-negative rule multi-reference, and it
    is the quantity contribution C4 is ultimately trying to maximize the
    probability of.
    """
    pred_onehot = _one_hot_predictions(y_pred, gold_mask.shape[1])
    return np.any(pred_onehot & gold_mask, axis=1)


def gold_set_frequency(gold_mask: np.ndarray) -> np.ndarray:
    """(C,) integer count of how often each class appears in ANY reference set.

    Unlike the scorer's `support`, this is a fixed property of the gold data and
    is therefore the count to use when comparing per-class results across
    systems. See consequence (3) in the module docstring.
    """
    return np.asarray(gold_mask, dtype=bool).sum(axis=0).astype(np.int64)


def _score_from_counts(
    tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, class_names: tuple[str, ...]
) -> ScoreResult:
    per_class: dict[str, ClassMetrics] = {}
    f1s: list[float] = []
    for i, name in enumerate(class_names):
        precision = _safe_div(float(tp[i]), float(tp[i] + fp[i]))
        recall = _safe_div(float(tp[i]), float(tp[i] + fn[i]))
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        per_class[name] = ClassMetrics(
            tp=int(tp[i]),
            fp=int(fp[i]),
            fn=int(fn[i]),
            precision=precision,
            recall=recall,
            f1=f1,
            support=int(tp[i] + fn[i]),
        )
        f1s.append(f1)
    # Macro over ALL declared classes, always -- see consequence (2).
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    return ScoreResult(macro_f1=macro_f1, per_class=per_class)


def macro_f1_multireference(
    y_pred: np.ndarray,
    gold_mask: np.ndarray,
    class_names: tuple[str, ...],
) -> ScoreResult:
    """Official Subtask 2 scorer: multi-reference macro-F1.

    Parameters
    ----------
    y_pred
        (N,) integer class indices; `OOV_INDEX` for an unrecognized prediction.
    gold_mask
        (N, C) boolean; `gold_mask[i, c]` is True iff class c is in item i's
        reference set. Build it with `labels.multi_reference_mask`, which
        deduplicates annotators by construction.
    class_names
        Class names in index order, used to key the per-class breakdown.

    Returns
    -------
    ScoreResult
    """
    n_classes = len(class_names)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    gold_mask = np.asarray(gold_mask, dtype=bool)
    _validate(y_pred, gold_mask, n_classes)

    pred_onehot = _one_hot_predictions(y_pred, n_classes)
    # (N, 1) -- broadcast across classes so one miss charges every gold member.
    hits_any = np.any(pred_onehot & gold_mask, axis=1)[:, None]

    tp = np.sum(pred_onehot & gold_mask, axis=0)
    fp = np.sum(pred_onehot & ~gold_mask, axis=0)
    # The multi-reference clause: charge a false negative only when the
    # prediction missed the reference set ENTIRELY. When it did miss, this is
    # True for every class in the set at once -- hence one FN per member.
    fn = np.sum(~pred_onehot & gold_mask & ~hits_any, axis=0)

    return _score_from_counts(tp, fp, fn, class_names)


def macro_f1_single_label(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    class_names: tuple[str, ...],
) -> ScoreResult:
    """Official Subtask 1 scorer: plain single-label macro-F1.

    Kept as a separate function rather than routed through the multi-reference
    path, mirroring the official scorer's own structure. The two agree whenever
    every reference set is a singleton -- `tests/test_scorer_conformance.py`
    asserts that equivalence, which is a useful cross-check on both.
    """
    n_classes = len(class_names)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_true = np.asarray(y_true, dtype=np.int64)
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Shape mismatch: {y_pred.shape} vs {y_true.shape}")

    pred_onehot = _one_hot_predictions(y_pred, n_classes)
    gold_onehot = _one_hot_predictions(y_true, n_classes)

    tp = np.sum(pred_onehot & gold_onehot, axis=0)
    fp = np.sum(pred_onehot & ~gold_onehot, axis=0)
    fn = np.sum(~pred_onehot & gold_onehot, axis=0)

    return _score_from_counts(tp, fp, fn, class_names)


def score_subtask2(y_pred: np.ndarray, gold_mask: np.ndarray) -> ScoreResult:
    """Convenience wrapper binding the nine evasion class names."""
    from higrec.data.labels import EVASION_LABELS

    if gold_mask.shape[1] != N_EVASION:
        raise ValueError(f"expected {N_EVASION} classes, got {gold_mask.shape[1]}")
    return macro_f1_multireference(y_pred, gold_mask, EVASION_LABELS)


def score_subtask1(y_pred: np.ndarray, y_true: np.ndarray) -> ScoreResult:
    """Convenience wrapper binding the three clarity class names."""
    from higrec.data.labels import CLARITY_LABELS

    return macro_f1_single_label(y_pred, y_true, CLARITY_LABELS)
