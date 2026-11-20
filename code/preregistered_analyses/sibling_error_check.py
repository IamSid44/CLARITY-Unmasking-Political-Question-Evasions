"""Preliminary check for a proposed sibling-contrast module (future work; not a registered test).

    python -m preregistered_analyses.sibling_error_check           # CPU, seconds; writes docs/raw/sibling_error_check.txt

E24 showed that sub-questions sharing one answer (siblings) get different labels far more often than annotators
disagree. If the system sometimes labels a sub-question by the part of the answer that responds to its sibling, its
errors on shared answers should land on a label that the sibling's annotators gave more often than chance. This
script asks that of the frozen system's dev labels as replayed in E25 (`runs/E25/labels/dev_own.npz`) and of M1
alone. "Correct" is the task's any-annotator rule. Chance is the share of the item's wrong labels that are a
sibling's (and not the item's own) reference labels, averaged over the errors. Written after E25 was known, dev only;
it motivates a registered test and is not evidence for a module.
"""

from __future__ import annotations

import numpy as np

from preregistered_analyses.e24_prechecks import DV, GOLD, S2, sibling_pairs
from qevasion.paths import ROOT, RUNS

RAW = ROOT / "docs" / "raw" / "sibling_error_check.txt"


def compute() -> list[str]:
    z = np.load(RUNS / "E25" / "labels" / "dev_own.npz", allow_pickle=True)
    shared = DV.groupby("interview_answer")["interview_answer"].transform("size").values > 1
    groups = [ix for ix in DV.groupby("interview_answer").indices.values() if len(ix) > 1]
    pr = np.array(sibling_pairs(DV))
    disjoint = np.array([not (GOLD[i] & GOLD[j]).any() for i, j in pr])
    L = [f"Sibling errors, preliminary check (dev, n = {len(DV)}; {shared.sum()} items share their answer with another "
         f"sub-question, {len(groups)} shared answers)", ""]
    for name, p in (("frozen system (E25 replay)", z["pred"]), ("M1 alone", z["m1_pred"])):
        ok = GOLD[np.arange(len(p)), p]
        leak, chance, n = 0, 0.0, 0
        for ix in groups:
            for i in ix:
                if ok[i]:
                    continue
                sib_only = np.any(GOLD[[j for j in ix if j != i]], axis=0) & ~GOLD[i]
                n += 1
                leak += int(sib_only[p[i]])
                chance += sib_only[~GOLD[i]].mean()
        same = np.array([p[i] == p[j] for i, j in pr])
        L += [f"{name}: S2 {S2(p):.3f}",
              f"   correct (any annotator): answer given to one sub-question {ok[~shared].mean():.3f} (n = {(~shared).sum()}), "
              f"shared answer {ok[shared].mean():.3f} (n = {shared.sum()})",
              f"   errors on shared answers: {n}; the wrong label is a sibling's reference label in {leak} "
              f"({leak / n:.1%}); chance {chance / n:.1%}",
              f"   same label on sibling pairs whose reference sets are disjoint: {same[disjoint].mean():.1%} of {disjoint.sum()}",
              ""]
    return L


def main() -> None:
    L = compute()
    print("\n".join(L))
    RAW.write_text("\n".join(L), encoding="utf8")
    print(f"\nwrote {RAW}")


if __name__ == "__main__":
    main()
