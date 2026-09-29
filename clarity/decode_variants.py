#!/usr/bin/env python3
"""Decoding variants on already-trained models (CPU): do any beat logit adjustment?

Every rule maps an ensemble's (or one model's) dev probabilities to 9-way labels. Its
parameters are fitted inside 5-fold nested CV on dev, exactly like decide.py's R1,
and the pooled held-out predictions are scored. Rules:

  LA        logit adjustment (the current rule): argmax p / prior^tau        1 param
  GATE      hierarchical Non-Reply gate: if p(NR block) > t choose within the
            NR classes, else within the other six; LA inside each branch      2 params
  OT        prior-matching decoding: per-class weights v solved by Sinkhorn so
            that predicted label mass matches a target distribution
            prior^gamma (renormalised); uses dev inputs, never dev labels     1 param
  FLOOR     LA, then every class gets at least m predictions (the m items
            with the highest p_class / p_predicted move to it)               2 params

    python clarity/decode_variants.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from decide import load_probs, logit_adjust, nested_cv
from qevasion.labels import EVASION_LABELS, N_EVASION, encode_evasion
from qevasion.loader import dev_reference_mask, load_qevasion
from qevasion.scoring import score_subtask2

RUNS = Path(__file__).resolve().parent / "runs"
sp = load_qevasion()
GOLD = dev_reference_mask(sp.dev)
y = encode_evasion(sp.train["evasion_label"].tolist())
PRIOR = np.bincount(y, minlength=N_EVASION) / len(y)
NR = np.array([EVASION_LABELS.index(c) for c in ("Declining to answer", "Claims ignorance", "Clarification")])
IS_NR = np.isin(np.arange(N_EVASION), NR)
TAUS = np.linspace(-1.0, 2.0, 31)


def f1(pred, gold):
    return score_subtask2(pred, gold).macro_f1


def la_pred(p, tau):
    return logit_adjust(p, PRIOR, tau).argmax(1)


def fit(grid, predict, p, gold, tr):
    """Best parameter tuple on the training fold."""
    best = max(grid, key=lambda g: f1(predict(p, g)[tr], gold[tr]))
    return best, f1(predict(p, best)[tr], gold[tr])


def rule_la(p, gold, tr, te):
    g, s = fit([(t,) for t in TAUS], lambda p, g: la_pred(p, g[0]), p, gold, tr)
    return la_pred(p, *g)[te], s


def gate_pred(p, t, tau):
    s = logit_adjust(p, PRIOR, tau)
    nr = p[:, IS_NR].sum(1) > t
    in_nr = np.where(IS_NR[None, :], s, -np.inf).argmax(1)
    in_ot = np.where(~IS_NR[None, :], s, -np.inf).argmax(1)
    return np.where(nr, in_nr, in_ot)


def rule_gate(p, gold, tr, te):
    grid = [(t, tau) for t in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6) for tau in TAUS[::2]]
    g, s = fit(grid, lambda p, g: gate_pred(p, *g), p, gold, tr)
    return gate_pred(p, *g)[te], s


def sinkhorn_weights(p, target, iters=200):
    """Per-class weights v with sum_i p_ij v_j / sum_k p_ik v_k ~= n * target_j."""
    v = np.ones(p.shape[1])
    n = len(p)
    for _ in range(iters):
        q = p * v
        q /= q.sum(1, keepdims=True)
        v *= (n * target) / np.maximum(q.sum(0), 1e-9)
    return v


def ot_pred(p, gamma):
    target = PRIOR ** gamma
    target /= target.sum()
    return (p * sinkhorn_weights(p, target)).argmax(1)


def rule_ot(p, gold, tr, te):
    g, s = fit([(gm,) for gm in np.linspace(0, 1, 11)], lambda p, g: ot_pred(p, g[0]), p, gold, tr)
    return ot_pred(p, *g)[te], s


def floor_pred(p, tau, m):
    pred = la_pred(p, tau)
    for c in range(N_EVASION):
        short = m - int((pred == c).sum())
        if short <= 0:
            continue
        cand = np.where(pred != c)[0]
        ratio = p[cand, c] / p[cand, pred[cand]]
        pred = pred.copy()
        pred[cand[np.argsort(-ratio)[:short]]] = c
    return pred


def rule_floor(p, gold, tr, te):
    grid = [(tau, m) for tau in TAUS[::2] for m in (0, 1, 2, 3, 5, 8)]
    g, s = fit(grid, lambda p, g: floor_pred(p, *g), p, gold, tr)
    return floor_pred(p, *g)[te], s


RULES = {"LA": rule_la, "GATE": rule_gate, "OT": rule_ot, "FLOOR": rule_floor}


def score(rule, p, splits=5):
    return float(np.mean([nested_cv(rule, p, GOLD, 5, cs)["out_of_fold_macro_f1"] for cs in range(splits)]))


def main() -> None:
    systems = {"baseline (8 ep)": ("L0_large_base", "E11_base_8ep"),
               "full question (16 ep)": ("E10_fullq_16ep", "E11_fullq_16ep")}
    print("dev S2, rule parameters fitted by 5-fold nested CV, averaged over 5 CV splits\n")
    print(f"{'':<24}" + "".join(f"{r:>8}" for r in RULES))
    for name, cfgs in systems.items():
        allp = np.concatenate([load_probs(RUNS / c)[0] for c in cfgs])
        ens = allp.mean(0)
        print(f"{name + ', 10-seed':<24}" + "".join(f"{score(fn, ens):>8.3f}" for fn in RULES.values()))
        per = {r: np.array([score(fn, p, 2) for p in allp]) for r, fn in RULES.items()}
        print(f"{name + ', per seed':<24}" + "".join(f"{per[r].mean():>8.3f}" for r in RULES))
        for r in ("GATE", "OT", "FLOOR"):
            d = per[r] - per["LA"]
            print(f"    {r} - LA per seed: {d.mean():+.3f} ± {d.std(ddof=1):.3f}, {int((d > 0).sum())}/10 seeds up")


if __name__ == "__main__":
    main()
