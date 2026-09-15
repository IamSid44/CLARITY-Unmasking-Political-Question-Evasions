"""C4: the set-membership decision rule.

WHAT THE SCORER ACTUALLY REWARDS
---------------------------------
Under the multi-reference scorer (see reports/04_scorer_semantics.md), the
per-example consequence of predicting class c is:

    c in G  ->  TP for c, and NO false negative for any other member of G
    c not in G  ->  FP for c, and ONE FALSE NEGATIVE FOR EVERY MEMBER of G

So the quantity that governs whether a prediction is safe is

    q(c|x) = P(c is in the reference set G | x)

which is NOT the consensus posterior P(c is the majority label | x). The two
diverge precisely when a class is often IN the annotator set but rarely its
mode -- which is the high-disagreement classes, which is where the field is
failing. That is the whole of contribution C4, and it is why the method predicts
in advance where its own gains must appear.

WHY q ALONE IS NOT ENOUGH
-------------------------
An earlier derivation assumed the score was invariant to WHICH member of G you
predict, leaving q as the only quantity that mattered. That assumption is FALSE
(verified: choosing the rare member of a 2-member set is worth exactly 1/9 more
macro-F1 than choosing the frequent one). Among several classes with c in G, the
macro-F1 payoff differs by class -- and that is exactly the job of the per-class
multipliers lambda_c. Because invariance fails, lambda_c carries real
decision-theoretic weight rather than being nearly inert.

    yhat = argmax_c  lambda_c * q(c|x)

Plug-in F-measure lineage: Lipton, Elkan & Narayanaswamy (arXiv:1402.1892);
Koyejo, Natarajan, Ravikumar & Dhillon (NIPS 2014); Narasimhan, Vaish & Agarwal
(NIPS 2014). Macro-F1 is not decomposable across examples, so a plug-in
thresholded rule fitted on held-out data is the consistent approach.

THE NON-SEPARABILITY THAT MOST IMPLEMENTATIONS GET WRONG
---------------------------------------------------------
Per-class SUPPORT under this scorer is ENDOGENOUS: an item where the model
predicts some other member of G_i contributes neither TP nor FN to class c and
drops out of c's support entirely. Therefore changing lambda_c alters OTHER
classes' supports, and macro-F1 is NOT separable across classes. Coordinate
ascent must recompute the FULL macro-F1 for each candidate rather than caching
per-class terms -- the caching optimization is wrong here, and wrong in a way
that silently produces a worse optimum rather than an error.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from higrec.data.labels import N_EVASION
from higrec.scoring.official import macro_f1_multireference

__all__ = [
    "ThresholdFit",
    "set_membership_from_annotators",
    "expected_set_size",
    "fit_thresholds_coordinate_ascent",
    "predict_with_thresholds",
    "nested_cv_thresholds",
]


@dataclass
class ThresholdFit:
    """Fitted per-class multipliers and the diagnostics that keep them honest."""

    lambda_: np.ndarray
    """(C,) per-class multipliers."""
    in_fold_macro_f1: float
    out_of_fold_macro_f1: float | None = None
    effective_examples_per_class: np.ndarray | None = None
    """(C,) how many tuning items actually had this class in their reference set.

    Reported NEXT TO every threshold, per P6.1. A lambda fitted on 4 examples is
    not a threshold, it is noise with a decimal point."""
    n_restarts: int = 1
    history: list[float] = field(default_factory=list)

    @property
    def generalization_gap(self) -> float | None:
        """in-fold minus out-of-fold macro-F1.

        The number to look at FIRST. A large gap means the contribution is
        threshold overfitting rather than a real decision-theoretic gain, and
        P6.1 says to report it before anything else.
        """
        if self.out_of_fold_macro_f1 is None:
            return None
        return self.in_fold_macro_f1 - self.out_of_fold_macro_f1


def set_membership_from_annotators(
    annotator_probs: np.ndarray, *, clip: float = 1e-12
) -> np.ndarray:
    """q(c|x) = 1 - Π_a (1 - P(annotator a labels c | x)).

    Parameters
    ----------
    annotator_probs
        (n_items, n_annotators, n_classes). Each [i, a, :] is annotator a's
        predicted label distribution for item i, obtained by pushing the latent
        posterior through the C3 annotator model.

    Returns
    -------
    (n_items, n_classes) array of set-membership probabilities.

    q IS NOT A DISTRIBUTION. It does not sum to 1, because the expected size of
    the reference set exceeds one whenever annotators disagree. Code that
    normalizes q has destroyed the very thing C4 exploits.

    The independence assumption -- annotators are conditionally independent given
    the latent leaf -- is a real modelling assumption and a limitation to state.
    Correlated annotator errors would make this an overestimate.
    """
    probs = np.clip(np.asarray(annotator_probs, dtype=np.float64), 0.0, 1.0)
    if probs.ndim != 3:
        raise ValueError(f"expected (items, annotators, classes), got {probs.shape}")
    # Work in log space: with many annotators the product underflows.
    log_miss = np.log(np.clip(1.0 - probs, clip, 1.0))
    return 1.0 - np.exp(log_miss.sum(axis=1))


def expected_set_size(q: np.ndarray) -> np.ndarray:
    """(n_items,) expected |G| implied by q, i.e. the row sums.

    P6.1 requires validating this against the EMPIRICALLY observed mean set size
    on the multi-reference items. A large mismatch means the annotator model is
    miscalibrated and q is not trustworthy, which must be caught before any
    threshold is fitted on it.
    """
    return np.asarray(q, dtype=np.float64).sum(axis=1)


def predict_with_thresholds(q: np.ndarray, lambda_: np.ndarray) -> np.ndarray:
    """yhat = argmax_c lambda_c * q(c|x)."""
    scores = np.asarray(q, dtype=np.float64) * np.asarray(lambda_, dtype=np.float64)[None, :]
    return np.argmax(scores, axis=1).astype(np.int64)


def _macro_f1(q: np.ndarray, lambda_: np.ndarray, gold_mask: np.ndarray,
              class_names: tuple[str, ...]) -> float:
    y_pred = predict_with_thresholds(q, lambda_)
    return macro_f1_multireference(y_pred, gold_mask, class_names).macro_f1


def _candidate_multipliers(q: np.ndarray, lambda_: np.ndarray, c: int) -> np.ndarray:
    """Candidate values of lambda_c at which the argmax can change.

    The decision for item i flips when lambda_c * q[i,c] crosses the current best
    competing score. Those crossing points are the only values worth evaluating,
    so the 1-D search is EXACT over a finite candidate set of size O(N) rather
    than a grid approximation.
    """
    q = np.asarray(q, dtype=np.float64)
    other = np.delete(np.asarray(lambda_, dtype=np.float64)[None, :] * q, c, axis=1)
    best_other = other.max(axis=1)  # (N,)

    q_c = q[:, c]
    positive = q_c > 0
    # lambda_c at which class c exactly ties the best competitor for item i.
    ties = np.full(q_c.shape, np.nan)
    ties[positive] = best_other[positive] / q_c[positive]
    ties = ties[np.isfinite(ties) & (ties > 0)]

    if ties.size == 0:
        return np.array([1.0])
    ties = np.unique(ties)
    # Evaluate just above and just below each tie so both sides of every flip are
    # reachable; a tie value itself is a measure-zero boundary.
    epsilon = 1e-9
    return np.unique(np.concatenate([ties * (1 - epsilon), ties * (1 + epsilon), [1.0]]))


def fit_thresholds_coordinate_ascent(
    q: np.ndarray,
    gold_mask: np.ndarray,
    class_names: tuple[str, ...],
    *,
    max_rounds: int = 20,
    n_restarts: int = 5,
    seed: int = 0,
    shrinkage: float = 0.0,
    base_rate: np.ndarray | None = None,
    tol: float = 1e-9,
) -> ThresholdFit:
    """Fit lambda by coordinate ascent on macro-F1.

    Each 1-D search is exact over the finite set of argmax-crossing points, so
    no grid resolution is involved. Random restarts guard against the local
    optima that non-separability makes possible.

    NOTE the non-separability: the full macro-F1 is recomputed for every
    candidate rather than updating per-class terms incrementally. That is not an
    inefficiency to optimize away -- changing lambda_c moves other classes'
    supports under this scorer, so cached per-class counts would be stale and the
    search would converge somewhere wrong.

    Parameters
    ----------
    shrinkage, base_rate
        Regularized variant required by P6.1: shrink lambda toward the value
        implied by the class base rate. `shrinkage` in [0, 1]; 0 disables it.
        With single-digit support on the rarest classes, an unregularized
        threshold is fitted on almost nothing.
    """
    q = np.asarray(q, dtype=np.float64)
    gold_mask = np.asarray(gold_mask, dtype=bool)
    n_classes = q.shape[1]
    if n_classes != len(class_names):
        raise ValueError(f"q has {n_classes} classes, class_names has {len(class_names)}")

    if shrinkage > 0 and base_rate is None:
        # Default prior: the empirical frequency with which each class appears in
        # any reference set, normalized to mean 1 so it is a neutral multiplier.
        frequency = gold_mask.sum(axis=0).astype(np.float64)
        frequency = np.where(frequency > 0, frequency, frequency[frequency > 0].min() / 2)
        base_rate = frequency.mean() / frequency

    rng = np.random.default_rng(seed)
    best_lambda = np.ones(n_classes, dtype=np.float64)
    best_score = _macro_f1(q, best_lambda, gold_mask, class_names)
    history: list[float] = [best_score]

    for restart in range(n_restarts):
        lambda_ = (
            np.ones(n_classes, dtype=np.float64)
            if restart == 0
            else np.exp(rng.normal(0.0, 0.5, size=n_classes))
        )
        score = _macro_f1(q, lambda_, gold_mask, class_names)

        for _ in range(max_rounds):
            improved = False
            for c in rng.permutation(n_classes):
                current = lambda_[c]
                for candidate in _candidate_multipliers(q, lambda_, c):
                    lambda_[c] = candidate
                    trial = _macro_f1(q, lambda_, gold_mask, class_names)
                    if shrinkage > 0 and base_rate is not None:
                        penalty = shrinkage * float(
                            np.mean((np.log(lambda_) - np.log(base_rate)) ** 2)
                        )
                        trial -= penalty
                    if trial > score + tol:
                        score, current, improved = trial, candidate, True
                lambda_[c] = current
            if not improved:
                break

        history.append(score)
        if score > best_score:
            best_score, best_lambda = score, lambda_.copy()

    effective = gold_mask.sum(axis=0).astype(np.int64)

    return ThresholdFit(
        lambda_=best_lambda,
        in_fold_macro_f1=_macro_f1(q, best_lambda, gold_mask, class_names),
        effective_examples_per_class=effective,
        n_restarts=n_restarts,
        history=history,
    )


def nested_cv_thresholds(
    q: np.ndarray,
    gold_mask: np.ndarray,
    class_names: tuple[str, ...],
    *,
    n_outer: int = 5,
    n_inner_restarts: int = 3,
    seed: int = 0,
    shrinkage: float = 0.0,
) -> ThresholdFit:
    """Nested cross-fitted thresholds. MANDATORY, not optional.

    Thresholds are fitted on inner folds and evaluated on the held-out outer
    fold, and BOTH the in-fold and out-of-fold macro-F1 are reported along with
    the gap between them.

    P6.1 is explicit that this comes before anything else: with ~308 tuning
    items and single-digit support on the rarest classes, threshold overfitting
    is the single most likely way for C4 to produce a gain that is not real. If
    the gap is large, the contribution IS the overfitting and we need to know
    immediately rather than after writing it up.
    """
    q = np.asarray(q, dtype=np.float64)
    gold_mask = np.asarray(gold_mask, dtype=bool)
    n_items = q.shape[0]

    rng = np.random.default_rng(seed)
    fold_of = rng.permutation(n_items) % n_outer

    in_fold_scores: list[float] = []
    out_fold_scores: list[float] = []
    lambdas: list[np.ndarray] = []

    for fold in range(n_outer):
        test = fold_of == fold
        train = ~test
        if not train.any() or not test.any():
            continue

        fit = fit_thresholds_coordinate_ascent(
            q[train], gold_mask[train], class_names,
            n_restarts=n_inner_restarts, seed=seed + fold, shrinkage=shrinkage,
        )
        lambdas.append(fit.lambda_)
        in_fold_scores.append(fit.in_fold_macro_f1)
        out_fold_scores.append(_macro_f1(q[test], fit.lambda_, gold_mask[test], class_names))

    # Report the geometric mean of fold lambdas: these are multiplicative
    # quantities, so an arithmetic mean would be biased upward by large values.
    mean_lambda = np.exp(np.mean(np.log(np.stack(lambdas)), axis=0))

    return ThresholdFit(
        lambda_=mean_lambda,
        in_fold_macro_f1=float(np.mean(in_fold_scores)),
        out_of_fold_macro_f1=float(np.mean(out_fold_scores)),
        effective_examples_per_class=gold_mask.sum(axis=0).astype(np.int64),
        n_restarts=n_inner_restarts,
        history=out_fold_scores,
    )


def consensus_posterior_baseline(annotator_probs: np.ndarray) -> np.ndarray:
    """The rule C4 must beat: argmax over the averaged consensus posterior.

    This is what every published system on this task does. It is the control in
    P6.2 that separates the decision-theoretic gain from the modelling gain, and
    the comparison is meaningless without it.
    """
    return np.asarray(annotator_probs, dtype=np.float64).mean(axis=1)


def oracle_upper_bound(
    gold_mask: np.ndarray, class_names: tuple[str, ...], *, seed: int = 0
) -> float:
    """Best achievable macro-F1 if q were PERFECT, bounding what the rule can close.

    With perfect knowledge of G, any member may be chosen, and by the invariance
    failure the choice matters. This greedily prefers the class that is currently
    rarest among predictions, approximating the macro-F1-optimal assignment; it
    is an achievable lower bound on the true oracle, and is reported as such.
    """
    gold_mask = np.asarray(gold_mask, dtype=bool)
    n_items, n_classes = gold_mask.shape
    chosen = np.full(n_items, -1, dtype=np.int64)
    counts = np.zeros(n_classes, dtype=np.int64)

    order = np.argsort(gold_mask.sum(axis=1))  # most-constrained items first
    for i in order:
        members = np.flatnonzero(gold_mask[i])
        if members.size == 0:
            continue
        pick = members[np.argmin(counts[members])]
        chosen[i] = pick
        counts[pick] += 1

    return macro_f1_multireference(chosen, gold_mask, class_names).macro_f1
