#!/usr/bin/env python3
"""Qualitative comparison of the two 10-seed systems on dev (report, sec:llm and sec:analysis)."""

from __future__ import annotations

from collections import Counter

import numpy as np

from decision_rules.decide import make_rules, nested_cv
from qevasion.labels import EVASION_LABELS, encode_evasion
from qevasion.loader import ANNOTATOR_COLUMNS, dev_reference_mask, load_qevasion
from preregistered_analyses.e13_analysis import RUNS, deberta_cfg

sp = load_qevasion()
dev = sp.dev.reset_index(drop=True)
GOLD = dev_reference_mask(dev)
y = encode_evasion(sp.train["evasion_label"].tolist())
PRIOR = np.bincount(y, minlength=len(EVASION_LABELS)) / len(y)
R1 = make_rules(PRIOR)["R1_logit_adjusted"]


def system(cfg_of_seed) -> np.ndarray:
    p = np.stack([np.load(RUNS / cfg_of_seed(s) / f"seed{s}" / "dev_probs.npy") for s in range(10)]).mean(0)
    return nested_cv(R1, p, GOLD, 5, 0)["pooled_predictions"]


deb = system(deberta_cfg)
qwen = system(lambda s: "Q8_fullq_lora")
d_in = GOLD[np.arange(len(GOLD)), deb]
q_in = GOLD[np.arange(len(GOLD)), qwen]
fixed, broken = np.flatnonzero(~d_in & q_in), np.flatnonzero(d_in & ~q_in)

print(f"dev items {len(GOLD)}; in-set: DeBERTa system {d_in.sum()}, Qwen system {q_in.sum()}")
print(f"Qwen fixes {len(fixed)} items and breaks {len(broken)} (net {len(fixed) - len(broken):+d})\n")

print(f"{'class in reference set':<22} {'items':>6} {'fixed':>6} {'broken':>7}")
for c, lab in enumerate(EVASION_LABELS):
    has = GOLD[:, c]
    print(f"{lab:<22} {has.sum():>6} {(has[fixed]).sum():>6} {(has[broken]).sum():>7}")


def error_pairs(pred: np.ndarray) -> Counter:
    pairs = Counter()
    for i in np.flatnonzero(~GOLD[np.arange(len(GOLD)), pred]):
        for c in np.flatnonzero(GOLD[i]):
            pairs[(EVASION_LABELS[c], EVASION_LABELS[pred[i]])] += 1
    return pairs


for name, pred in [("DeBERTa system", deb), ("Qwen system", qwen)]:
    print(f"\n{name}: most frequent error pairs (reference label -> prediction)")
    for (r, p), n in error_pairs(pred).most_common(6):
        print(f"  {r:<20} -> {p:<20} {n}")
    print(f"  predicted counts: " + ", ".join(f"{EVASION_LABELS[c]} {n}" for c, n in
                                            sorted(Counter(pred.tolist()).items())))


def show(i: int) -> None:
    row = dev.iloc[i]
    ans = " ".join(str(row["interview_answer"]).split())
    print(f"  dev row {i}: Q: {row['question']}")
    print(f"    A ({len(ans)} chars): {ans[:400]}{'...' if len(ans) > 400 else ''}")
    print(f"    annotators: {', '.join(str(row[c]) for c in ANNOTATOR_COLUMNS)}; "
          f"DeBERTa: {EVASION_LABELS[deb[i]]}; Qwen: {EVASION_LABELS[qwen[i]]}")


alen = dev["interview_answer"].astype(str).str.len().to_numpy()
for title, items in [("FIXED by Qwen", fixed), ("BROKEN by Qwen", broken)]:
    print(f"\n=== {title}: per class, the item with the shortest answer")
    for c, lab in enumerate(EVASION_LABELS):
        cand = [i for i in items if GOLD[i, c]]
        if cand:
            print(f"[{lab}] ({len(cand)} items)")
            show(min(cand, key=lambda i: alen[i]))
