"""Reimplementation of the official SemEval-2026 Task 6 scorers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qevasion.labels import N_EVASION, OOV_INDEX

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
    """Division guarded to 0.0, matching the official scorer."""
    return numerator / denominator if denominator else 0.0


def _one_hot_predictions(y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """(N, C) boolean matrix; an OOV prediction produces an all-False row."""
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
    """(N,) boolean: does item i's prediction land anywhere inside its reference set?"""
    pred_onehot = _one_hot_predictions(y_pred, gold_mask.shape[1])
    return np.any(pred_onehot & gold_mask, axis=1)


def gold_set_frequency(gold_mask: np.ndarray) -> np.ndarray:
    """(C,) integer count of how often each class appears in ANY reference set."""
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
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    return ScoreResult(macro_f1=macro_f1, per_class=per_class)


def macro_f1_multireference(
    y_pred: np.ndarray,
    gold_mask: np.ndarray,
    class_names: tuple[str, ...],
) -> ScoreResult:
    """Official Subtask 2 scorer: multi-reference macro-F1."""
    n_classes = len(class_names)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    gold_mask = np.asarray(gold_mask, dtype=bool)
    _validate(y_pred, gold_mask, n_classes)

    pred_onehot = _one_hot_predictions(y_pred, n_classes)
    hits_any = np.any(pred_onehot & gold_mask, axis=1)[:, None]

    tp = np.sum(pred_onehot & gold_mask, axis=0)
    fp = np.sum(pred_onehot & ~gold_mask, axis=0)
    fn = np.sum(~pred_onehot & gold_mask & ~hits_any, axis=0)

    return _score_from_counts(tp, fp, fn, class_names)


def macro_f1_single_label(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    class_names: tuple[str, ...],
) -> ScoreResult:
    """Official Subtask 1 scorer: plain single-label macro-F1."""
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
    from qevasion.labels import EVASION_LABELS

    if gold_mask.shape[1] != N_EVASION:
        raise ValueError(f"expected {N_EVASION} classes, got {gold_mask.shape[1]}")
    return macro_f1_multireference(y_pred, gold_mask, EVASION_LABELS)


def score_subtask1(y_pred: np.ndarray, y_true: np.ndarray) -> ScoreResult:
    """Convenience wrapper binding the three clarity class names."""
    from qevasion.labels import CLARITY_LABELS

    return macro_f1_single_label(y_pred, y_true, CLARITY_LABELS)
