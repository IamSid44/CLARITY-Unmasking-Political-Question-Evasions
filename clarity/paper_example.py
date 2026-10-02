#!/usr/bin/env python3
"""The worked example in the paper (Figure "worked example", sec:analysis): one dev item through
both 10-seed systems.

The item is dev row 17 (the Lieberman question). It was fixed in advance because the error
analysis already discusses it (reports/raw/mideval/error_examples.md, DeBERTa ensemble), before Qwen's
prediction for it was looked at.

For each system (DeBERTa-v3-large full question 16 ep, seeds 0-9; Qwen3-8B-Base + LoRA, seeds 0-9)
it prints the mean probability over seeds, the tau that the nested CV (split 0, 5 folds) fitted
on the four folds not containing the item, the logit-adjusted scores, the final 9-way label,
the derived 3-way label and whether the label is in the item's reference set. It first checks
that the pooled system scores reproduce the paper's Table 3 (0.543 and 0.405).

    python clarity/paper_example.py > clarity/reports/raw/paper_example_dev17.txt
    # needs clarity/runs/<cfg>/seed<k>/dev_probs.npy (from the HF repo) and the dev parquet
"""

from __future__ import annotations

import numpy as np

from decide import fit_tau, logit_adjust, make_rules, nested_cv
from qevasion.labels import CLARITY_LABELS, EVASION_LABELS, encode_evasion, leaf_to_official_clarity
from qevasion.loader import ANNOTATOR_COLUMNS, dev_reference_mask, load_qevasion
from e13_analysis import RUNS, deberta_cfg

ITEM = 17
SEEDS = range(10)
GRID = np.linspace(-1.0, 2.0, 61)

sp = load_qevasion()
GOLD = dev_reference_mask(sp.dev)
y = encode_evasion(sp.train["evasion_label"].tolist())
PRIOR = np.bincount(y, minlength=len(EVASION_LABELS)) / len(y)
R1 = make_rules(PRIOR)["R1_logit_adjusted"]


def mean_probs(cfg_of_seed) -> np.ndarray:
    return np.stack([np.load(RUNS / cfg_of_seed(s) / f"seed{s}" / "dev_probs.npy") for s in SEEDS]).mean(0)


def fold_tau(p: np.ndarray, item: int, seed: int = 0, n_folds: int = 5) -> float:
    """The tau nested_cv(seed=0) uses for the fold that holds `item` (same permutation and split)."""
    order = np.random.default_rng(seed).permutation(len(p))
    folds = np.array_split(order, n_folds)
    f = next(i for i, fo in enumerate(folds) if item in fo)
    tr = np.concatenate([folds[j] for j in range(n_folds) if j != f])
    return fit_tau(p[tr], GOLD[tr], PRIOR, GRID)[0]


row = sp.dev.iloc[ITEM]
print(f"dev row {ITEM}")
print(f"  sub-question : {row['question']}")
print(f"  full question: {row['interview_question']}")
print(f"  answer       : {row['interview_answer']}")
print(f"  annotators   : {', '.join(str(row[c]) for c in ANNOTATOR_COLUMNS)}")
print(f"  reference set: {', '.join(EVASION_LABELS[c] for c in np.flatnonzero(GOLD[ITEM]))}")
print(f"  gold clarity : {row['clarity_label']}\n")

expected = {"DeBERTa-v3-large (full question, 16 ep)": 0.405, "Qwen3-8B-Base + LoRA (3 ep)": 0.543}
for (name, cfg), target in zip(
    [("DeBERTa-v3-large (full question, 16 ep)", deberta_cfg), ("Qwen3-8B-Base + LoRA (3 ep)", lambda s: "Q8_fullq_lora")],
    expected.values(),
):
    p = mean_probs(cfg)
    res = nested_cv(R1, p, GOLD, 5, 0)
    score = res["out_of_fold_macro_f1"]
    assert abs(score - target) < 5e-4, f"{name}: system S2 {score:.4f} != {target}"
    tau = fold_tau(p, ITEM)
    adj = logit_adjust(p[ITEM][None], PRIOR, tau)[0]
    adj = adj / adj.sum()
    pred = int(res["pooled_predictions"][ITEM])
    assert pred == int(adj.argmax())
    clar = CLARITY_LABELS[int(leaf_to_official_clarity(np.array([pred]))[0])]
    print(f"{name}: 10-seed system S2 = {score:.3f} (reproduces the paper)")
    print(f"  ensemble argmax: {EVASION_LABELS[int(p[ITEM].argmax())]}")
    print(f"  tau for this item's CV fold: {tau:.2f}")
    print(f"  {'label':<22} {'mean p':>7} {'after LA':>9} {'train prior':>12}")
    for c in np.argsort(-p[ITEM])[:4]:
        print(f"  {EVASION_LABELS[c]:<22} {p[ITEM, c]:>7.3f} {adj[c]:>9.3f} {PRIOR[c]:>12.3f}")
    print(f"  system prediction: S2 {EVASION_LABELS[pred]} -> S1 {clar}; "
          f"in reference set: {bool(GOLD[ITEM, pred])}\n")
