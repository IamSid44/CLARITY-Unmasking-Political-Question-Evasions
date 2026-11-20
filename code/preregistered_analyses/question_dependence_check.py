"""Preliminary check for a proposed question-dependence signal (future work; not a registered test).

    python -m preregistered_analyses.question_dependence_check     # CPU, seconds; writes docs/raw/question_dependence_check.txt

E25a ran the frozen M1 on every dev answer three ways: with its own question, with no question, and with a question
from another interview. With no question, 64% of answers read as Clear Replies; with their own question, 25%. An
answer that reads as a reply on its own but not to the question it was given is, by construction, a candidate
evasion. This script asks only whether that gap carries information beyond M1's own score, using the outputs we
already have and the same target, model and folds as `question_prior_check.py` (the M5 check): whether the
President evaded, on all items and on the 36% of items M1 is least sure of (the items the cascade defers). It also
asks whether the gap and the M5 question-side prior overlap. Written after E25 was known, dev only, 308 items; it
motivates a registered test and is not evidence for a module.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from preregistered_analyses import e25_analysis as E25
from preregistered_analyses.question_prior_check import REPS, cv_auc, logit
from qevasion.paths import RUNS, ROOT

RAW = ROOT / "docs" / "raw" / "question_dependence_check.txt"


def js(p, q):
    p, q = np.clip(p, 1e-9, 1), np.clip(q, 1e-9, 1)
    m = (p + q) / 2
    return 0.5 * (p * np.log(p / m)).sum(1) + 0.5 * (q * np.log(q / m)).sum(1)


def compute() -> list[str]:
    df = E25.llm_table()
    piv = df.pivot_table(index="src_i", columns=["condition", "model"], values="evade")
    gold = df[df.condition == "P"].groupby("src_i").pres_evade.first()
    idx = gold.index.values
    y = gold.values.astype(int)
    P = piv["P"].mean(axis=1).loc[gold.index].values
    lab = {k: np.load(RUNS / "E25" / "labels" / f"dev_{k}.npz", allow_pickle=True) for k in ("own", "null", "rand")}
    own, null = lab["own"]["pbar"][idx], lab["null"]["pbar"][idx]
    u = lab["own"]["u"][idx]
    hi = u >= np.quantile(u, 0.64)
    m1 = 1 - own[:, 0]
    gap = logit(null[:, 0]) - logit(own[:, 0])          # reads as a reply alone, not to this question
    div = js(own, null)                                  # how much the question moves the whole distribution
    L = [f"Question-dependence signal, preliminary check (dev, n = {len(y)}; the President evaded on {y.mean():.3f})", "",
         "Explicit (Clear Reply) probability, mean over items:",
         f"   own question {own[:, 0].mean():.3f}   no question {null[:, 0].mean():.3f}   "
         f"another interview's question {lab['rand']['pbar'][idx][:, 0].mean():.3f}", "",
         "President evaded, by tercile of the gap logit p_null(Explicit) - logit p_own(Explicit):"]
    cuts = np.quantile(gap, [1 / 3, 2 / 3])
    for name, m in (("low", gap <= cuts[0]), ("middle", (gap > cuts[0]) & (gap <= cuts[1])), ("high", gap > cuts[1])):
        L.append(f"   {name:6s} gap: {int(m.sum())} items, the President evaded on {y[m].mean():.3f}")
    L += ["", "AUROC for 'the President evaded' (no fitting):",
          f"   M1, 1 - p(Explicit): {roc_auc_score(y, m1):.3f}",
          f"   gap: {roc_auc_score(y, gap):.3f}     JS(own, no question): {roc_auc_score(y, div):.3f}",
          f"   correlation gap / M1: {np.corrcoef(gap, logit(m1))[0, 1]:.3f}   gap / P-role share: {np.corrcoef(gap, P)[0, 1]:.3f}", "",
          f"Cross-validated logistic regression, 5 folds, mean AUROC over {REPS} repeats:"]
    X = {"M1": np.c_[logit(m1)], "M1 + gap": np.c_[logit(m1), gap], "M1 + P-role share": np.c_[logit(m1), P],
         "M1 + gap + P-role share": np.c_[logit(m1), gap, P]}
    for nm, sel in (("all items", slice(None)), (f"least sure 36% (n = {int(hi.sum())})", hi)):
        L.append(f"   {nm}:")
        for k, x in X.items():
            L.append(f"      {k:26s} {cv_auc(x[sel], y[sel]):.3f}")
    return L


def main() -> None:
    L = compute()
    print("\n".join(L))
    RAW.write_text("\n".join(L) + "\n", encoding="utf8")
    print(f"\nwrote {RAW}")


if __name__ == "__main__":
    main()
