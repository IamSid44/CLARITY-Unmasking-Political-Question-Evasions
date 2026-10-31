"""Preliminary check for the proposed question-side evasion prior (future work; not a registered test).

    python -m preregistered_analyses.question_prior_check          # CPU, seconds; writes docs/raw/question_prior_check.txt

The role study (E25b) put the 308 dev questions to two open LLMs in the President role (P) without the President's
real answer. If the share of those P-role answers that the frozen instrument labels evasive carries information about
whether the President evaded, a question-side feature could be added to the cascade's fusion. This script asks only
whether such a signal exists in the outputs we already have: (a) how often the President evaded by the number of P-role
models that evaded, and (b) whether the feature adds to M1's evade-versus-not score under cross-validation, on all
items and on the 36% of items M1 is least sure of (the items the cascade defers). Written after the results of E25
were known, dev only, 308 items; it motivates a test and is not evidence for the module.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from preregistered_analyses import e25_analysis as E25
from qevasion.paths import RUNS, ROOT

RAW = ROOT / "docs" / "raw" / "question_prior_check.txt"
REPS = 40


def logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def cv_auc(X, y, reps=REPS):
    out = []
    for seed in range(reps):
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        p = cross_val_predict(LogisticRegression(C=1.0), X, y, cv=cv, method="predict_proba")[:, 1]
        out.append(roc_auc_score(y, p))
    return float(np.mean(out))


def compute() -> list[str]:
    df = E25.llm_table()
    piv = df.pivot_table(index="src_i", columns=["condition", "model"], values="evade")
    gold = df[df.condition == "P"].groupby("src_i").pres_evade.first()
    idx = gold.index
    y = gold.values.astype(int)
    P = piv["P"].mean(axis=1).loc[idx].values
    z = np.load(RUNS / "E25" / "labels" / "dev_own.npz", allow_pickle=True)
    m1 = 1 - z["pbar"][idx.values][:, 0]
    u = z["u"][idx.values]
    hi = u >= np.quantile(u, 0.64)
    L = [f"Question-side prior, preliminary check (dev, n = {len(y)}; the President evaded on {y.mean():.3f})", ""]
    L.append("President evaded, by the share of the two P-role LLMs whose answer the instrument labels evasive:")
    for v in sorted(set(P)):
        L.append(f"   share {v:.1f}: {int((P == v).sum())} items, the President evaded on {y[P == v].mean():.3f}")
    L += ["", "AUROC for 'the President evaded' (no fitting):",
          f"   M1, 1 - p(Explicit): {roc_auc_score(y, m1):.3f}",
          f"   P-role evasion share: {roc_auc_score(y, P):.3f}",
          f"   correlation of the two: {np.corrcoef(P, m1)[0, 1]:.3f}", "",
          f"Cross-validated logistic regression, 5 folds, mean AUROC over {REPS} repeats:"]
    X1, X2 = np.c_[logit(m1)], np.c_[logit(m1), P]
    L.append(f"   all items:            M1 {cv_auc(X1, y):.3f}   M1 + P-role share {cv_auc(X2, y):.3f}")
    L.append(f"   least sure 36% (n = {int(hi.sum())}): M1 {cv_auc(X1[hi], y[hi]):.3f}   M1 + P-role share {cv_auc(X2[hi], y[hi]):.3f}")
    return L


def main() -> None:
    L = compute()
    print("\n".join(L))
    RAW.write_text("\n".join(L) + "\n", encoding="utf8")
    print(f"\nwrote {RAW}")


if __name__ == "__main__":
    main()
