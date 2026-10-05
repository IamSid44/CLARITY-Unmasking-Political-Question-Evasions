#!/usr/bin/env python3
"""Regenerate every number in docs/01_scorer_geometry.md."""

from __future__ import annotations

import itertools

import numpy as np


from qevasion.labels import EVASION_LABELS, N_EVASION, multi_reference_mask
from qevasion.loader import derive_dev_consensus_leaf, dev_reference_mask, load_qevasion
from qevasion.scoring import score_subtask2

from qevasion.loader import DATA_CACHE
COLS = ("annotator1", "annotator2", "annotator3")


def _diag(pred: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
    """(in-set rate, number of distinct classes named)."""
    known = pred >= 0
    hit = np.zeros(len(pred), dtype=bool)
    hit[known] = mask[np.arange(len(pred))[known], pred[known]]
    return float(hit.mean()), len(set(pred[known].tolist()))


def rule(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def main() -> None:
    dev = load_qevasion(DATA_CACHE).dev
    mask = dev_reference_mask(dev)
    n = len(mask)

    rule("1. The majority-vote figure, and what the 0.1166 'headroom' actually measures")
    leaf, has_majority = derive_dev_consensus_leaf(dev)
    n_oov = int((leaf < 0).sum())
    print(f"items                              : {n}")
    print(f"items with a leaf majority         : {int(has_majority.sum())}")
    print(f"items left unpredicted (OOV = -1)  : {n_oov}")
    print(f"macro-F1 of that vector            : {score_subtask2(leaf, mask).macro_f1:.4f}")

    resolved = leaf.copy()
    for i in np.where(leaf < 0)[0]:
        resolved[i] = int(np.flatnonzero(mask[i])[0])
    print(
        f"same vector, {n_oov} OOV items resolved to ANY set member : "
        f"{score_subtask2(resolved, mask).macro_f1:.4f}"
    )
    print(f"  every prediction lands in G      : {bool(mask[np.arange(n), resolved].all())}")
    print(
        "\nSo the gap is the cost of ABSTAINING on the three-way splits, not the\n"
        "value of choosing among members of the reference set."
    )

    rule("2. Which member you name is worth nothing: random in-set choice")
    for trial in range(5):
        rng = np.random.default_rng(trial)
        pick = np.array([rng.choice(np.flatnonzero(mask[i])) for i in range(n)], dtype=np.int64)
        print(
            f"  trial {trial}: macro-F1 = {score_subtask2(pick, mask).macro_f1:.4f}"
            f"   distinct classes named = {len(set(pick.tolist()))}"
        )

    rule("3. The coverage identity: macro-F1 = (classes named) / 9 when in-set")
    print("   Restricting the allowed label space forces classes to go unnamed.")
    print(f"   {'allowed':>8} {'in-set rate':>12} {'named':>7} {'macro-F1':>9} {'k/9':>8}")
    for k in range(N_EVASION, 1, -1):
        allowed = set(range(k))
        pred = np.array(
            [
                next((c for c in np.flatnonzero(mask[i]) if c in allowed), -1)
                for i in range(n)
            ],
            dtype=np.int64,
        )
        hit = np.array([pred[i] >= 0 and mask[i, pred[i]] for i in range(n)])
        named = len(set(pred[pred >= 0].tolist()))
        print(
            f"   {k:>8} {hit.mean():>12.3f} {named:>7} "
            f"{score_subtask2(pred, mask).macro_f1:>9.4f} {named / 9:>8.4f}"
        )
    print(
        "\n   Why: an in-set prediction can never be a false positive, and no item\n"
        "   that hits its set can charge a false negative. So FP_c = FN_c = 0 for\n"
        "   every class, and F1_c = 1 for every class named at least once."
    )

    rule("4. The metric to optimise and report is IN-SET RATE")
    print("   Trivial baselines, to calibrate:")
    prior_pred = None
    for c, nm in enumerate(EVASION_LABELS):
        if nm in ("Explicit", "Dodging", "General", "Clarification"):
            p_ = np.full(n, c, dtype=np.int64)
            d_ = _diag(p_, mask)
            print(
                f"     always '{nm}':".ljust(34)
                + f"macro-F1={score_subtask2(p_, mask).macro_f1:.4f}"
                f"  in-set={d_[0]:.3f}  named={d_[1]}/9"
            )
    rng = np.random.default_rng(0)
    for tag, p_ in [("uniform random", rng.integers(0, N_EVASION, n))]:
        d_ = _diag(p_, mask)
        print(
            f"     {tag}:".ljust(34)
            + f"macro-F1={score_subtask2(p_, mask).macro_f1:.4f}"
            f"  in-set={d_[0]:.3f}  named={d_[1]}/9"
        )

    print("\n   macro-F1 as a function of in-set rate, for a predictor that hits G at")
    print("   rate r and otherwise names a random wrong class (all 9 classes named):")
    oracle = np.array([rng.choice(np.flatnonzero(mask[i])) for i in range(n)], dtype=np.int64)
    print(f"     {'in-set rate':>12} {'macro-F1':>10}")
    for r in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4):
        vals = []
        for t in range(20):
            rg = np.random.default_rng(t)
            p_ = oracle.copy()
            for i in np.where(rg.random(n) > r)[0]:
                p_[i] = rg.choice([c for c in range(N_EVASION) if not mask[i, c]])
            vals.append(score_subtask2(p_, mask).macro_f1)
        print(f"     {r:>12.2f} {np.mean(vals):>10.4f}")
    print("\n   Reading DEV scores off the curve (it is built on dev; test scores do not")
    print("   belong on it): our logit-adjusted ensemble (0.438) sits near an in-set rate")
    print("   of 0.59, ChulaNLP's DeBERTa (0.46) near 0.61, TeleAI (0.617) near 0.76. The curve")
    print("   is steep -- +0.10 in-set rate is worth roughly +0.10 macro-F1 in that")
    print("   range -- so in-set rate is both the right target and a sensitive one.")

    rule("5. The test set has TWO annotators, not three -- how much shrinks?")
    print(f"   {'reference sets':<24} {'|G| mean':>9} {'>1 label':>9}")
    for name, cols in [("3ann (dev as published)", COLS)] + [
        ("2ann: " + "+".join(p), p) for p in itertools.combinations(COLS, 2)
    ]:
        refs = [[r[c] for c in cols] for _, r in dev.iterrows()]
        m = multi_reference_mask(refs)
        sizes = m.sum(1)
        print(f"   {name:<24} {sizes.mean():>9.3f} {(sizes > 1).mean():>9.1%}")
    print(
        "\n   Smaller reference sets mean fewer chances to land in G, so every\n"
        "   multi-reference number measured on dev is optimistic for the test set.\n"
        "   Quantify the effect on a real model with `decide.py --drop-annotator`."
    )


if __name__ == "__main__":
    main()
