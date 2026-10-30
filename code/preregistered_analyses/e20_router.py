#!/usr/bin/env python3
"""E20 (as amended 2026-10-29): the value map and the learned value router, on the leak-free train rows.

    python -m preregistered_analyses.e20_router > ../docs/raw/E20_router.txt     # CPU, < 1 min

Rows: train rows whose answer text has no sibling (no other sub-question of the same answer), where cross-fitting
does not leak. M1 = the cross-fitted fold model's calibrated q (cascade_inputs.npz train_q); M3 = the VM's 27B over
the row's cross-fitted top 4 (tag qwen38_27b_k0_vm, split trainall; single order). fix = M3 right and M1 wrong,
break = the reverse (one gold label).
S1 value map: net = fixes - breaks per fifth of M1 uncertainty u = 1 - top q (Q1 = most confident), bootstrap CIs.
S4 learned router: multinomial logistic regression of {fix, break, neither} on features available before any call
(quantile-normalised top q, margin and entropy; M1's top class; log answer and question length; the E22e
specificity scores PMI_c and PMI_0), cross-fitted over 5 folds. Score = P(fix) - P(break).
Decision rule (registered, S-H3): the router replaces confidence routing only if, on the held-out folds, it beats
deferring by u in macro-F1 at >= 3 of the 4 budgets delta in {0.1, 0.2, 0.3, 0.5}.
The thinking rung (R2) was dropped (E20 amendment: past 18:00), so no multi-rung allocation is evaluated.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

from cascade.adjudicate import M1_INPUTS
from cascade.exemplars import group_keys, load_frames
from qevasion.labels import N_EVASION, encode_evasion
from qevasion.paths import RUNS
from qevasion.scoring import macro_f1_single_label
from qevasion.labels import EVASION_LABELS

BUDGETS = (0.1, 0.2, 0.3, 0.5)
rule = lambda t: print("=" * 110 + f"\n{t}\n" + "=" * 110)


def load() -> dict:
    tr = load_frames()["train"].reset_index(drop=True)
    y = encode_evasion(tr["evasion_label"].tolist())
    q = np.load(M1_INPUTS)["train_q"]
    m3 = {}
    for line in (RUNS / "M3" / "qwen38_27b_k0_vm" / "responses.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["frame"] == "train" and r["split"] == "trainall":
            m3[r["i"]] = (r["cand"], np.asarray(r["q"]))
    ans, _ = group_keys(tr)
    _, inv, cnt = np.unique(ans.astype(str), return_inverse=True, return_counts=True)
    single = cnt[inv] == 1
    spec = np.load(RUNS / "E22" / "specificity.npz")
    rows = np.array([i for i in np.where(single)[0] if i in m3])
    p1 = q.argmax(1)
    p3 = np.array([m3[i][0][int(m3[i][1].argmax())] if i in m3 else -1 for i in range(len(y))])
    srt = np.sort(q, 1)
    ent = -(q * np.log(q + 1e-12)).sum(1)
    qn = lambda v: rankdata(v) / len(v)  # quantile-normalised within the set
    X = np.column_stack([qn(srt[:, -1]), qn(srt[:, -1] - srt[:, -2]), qn(ent),
                         np.eye(N_EVASION)[p1],
                         np.log1p(tr["interview_answer"].astype(str).str.split().str.len()),
                         np.log1p(tr["question"].astype(str).str.split().str.len()),
                         spec["train_own"] - spec["train_rand"].mean(1), spec["train_own"] - spec["train_null"]])
    return {"y": y, "p1": p1, "p3": p3, "u": 1 - q.max(1), "X": X, "rows": rows, "single_n": int(single.sum())}


def value_map(d: dict) -> None:
    rule("S1. Value map on singleton train rows: M3 (VM) vs M1 (cross-fitted), by fifth of M1 uncertainty")
    r, y, p1, p3, u = d["rows"], d["y"], d["p1"], d["p3"], d["u"]
    fix = (p3[r] == y[r]) & (p1[r] != y[r])
    brk = (p1[r] == y[r]) & (p3[r] != y[r])
    edges = np.quantile(u[r], [0.2, 0.4, 0.6, 0.8])
    fifth = np.searchsorted(edges, u[r], side="right")
    rng = np.random.default_rng(20)
    print(f"   rows: {len(r)} of {d['single_n']} singleton rows have an M3 record (of 3,448 train rows)")
    print(f"   {'fifth':6} {'n':>4} {'u range':>12} {'M1 ok':>6} {'M3 ok':>6} {'fixes':>6} {'breaks':>7} {'net':>5} {'95% CI':>14}")
    for k in range(5):
        s = fifth == k
        net = int(fix[s].sum() - brk[s].sum())
        idx = np.where(s)[0]
        boot = [int(fix[b].sum() - brk[b].sum()) for b in (rng.choice(idx, len(idx)) for _ in range(2000))]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"   Q{k + 1:<5} {s.sum():>4} {u[r][s].min():5.2f}-{u[r][s].max():4.2f} {np.mean(p1[r][s] == y[r][s]):6.3f} "
              f"{np.mean(p3[r][s] == y[r][s]):6.3f} {int(fix[s].sum()):6d} {int(brk[s].sum()):7d} {net:5d} [{lo:+.0f}, {hi:+.0f}]")


def router(d: dict) -> None:
    rule("S4. Learned value router vs confidence routing, held-out folds of the singleton rows (single gold label)")
    r, y, p1, p3, u, X = d["rows"], d["y"], d["p1"], d["p3"], d["u"], d["X"]
    cls = np.where((p3 == y) & (p1 != y), 0, np.where((p1 == y) & (p3 != y), 1, 2))  # fix / break / neither
    score = np.empty(len(y))
    folds = np.array_split(np.random.default_rng(20).permutation(r), 5)
    for f in range(5):
        te = folds[f]
        trn = np.concatenate([folds[j] for j in range(5) if j != f])
        mu, sd = X[trn].mean(0), X[trn].std(0) + 1e-9
        lr = LogisticRegression(max_iter=2000, C=1.0).fit((X[trn] - mu) / sd, cls[trn])
        pr = lr.predict_proba((X[te] - mu) / sd)
        col = {c: j for j, c in enumerate(lr.classes_)}
        score[te] = pr[:, col.get(0)] - pr[:, col.get(1)]
    wins = 0
    print(f"   {'budget':>6} {'confidence: macro-F1 / acc':>28} {'router: macro-F1 / acc':>24} {'random (200)':>14}")
    rng = np.random.default_rng(21)
    for delta in BUDGETS:
        k = int(round(delta * len(r)))
        out = {}
        for name, key in (("confidence", u[r]), ("router", score[r])):
            dfr = np.zeros(len(r), bool)
            dfr[np.argsort(-key, kind="stable")[:k]] = True
            pred = np.where(dfr, p3[r], p1[r])
            out[name] = (macro_f1_single_label(pred, y[r], EVASION_LABELS).macro_f1, float(np.mean(pred == y[r])))
        rnd = []
        for _ in range(200):
            dfr = np.zeros(len(r), bool)
            dfr[rng.choice(len(r), k, replace=False)] = True
            rnd.append(macro_f1_single_label(np.where(dfr, p3[r], p1[r]), y[r], EVASION_LABELS).macro_f1)
        wins += out["router"][0] > out["confidence"][0]
        print(f"   {delta:6.1f} {out['confidence'][0]:15.3f} / {out['confidence'][1]:.3f} {out['router'][0]:15.3f} / "
              f"{out['router'][1]:.3f} {np.mean(rnd):10.3f}")
    base = macro_f1_single_label(p1[r], y[r], EVASION_LABELS).macro_f1
    print(f"   M1 alone (delta 0): macro-F1 {base:.3f}, accuracy {np.mean(p1[r] == y[r]):.3f}")
    print(f"   S-H3 / decision rule: router better at {wins} of {len(BUDGETS)} budgets -> "
          f"{'ADOPT the router' if wins >= 3 else 'confidence routing stays'}")


def main() -> None:
    d = load()
    value_map(d)
    print()
    router(d)


if __name__ == "__main__":
    main()
