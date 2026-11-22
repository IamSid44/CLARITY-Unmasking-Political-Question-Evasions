#!/usr/bin/env python3
"""E36 (registered 2026-11-21 as exploratory, no hypotheses): what drives the role effect.

    python -m preregistered_analyses.e36_drivers      # CPU; needs the E34 confirmation table (CF labels)

On the confirmation sample (248 train interviews not in dev), CF labels, the three discovery models:
  1. ROR P/N within subgroups (interview bootstrap, 1,000 draws), Holm over the subgroups' one-sided p values:
     'you' sub-questions, multiple_questions, affirmative_questions, hypothetical wording, future wording, President.
  2. Word overlap of each answer with the President's real answer (share of the answer's distinct words of >= 5 letters
     that occur in the real answer), by role, against a random other real answer from another interview.
  3. Meta answers ("as an AI", "language model") and answer length, by role.
Writes docs/raw/E36_drivers.txt and runs/E34/e36.json.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from preregistered_analyses.e34_analysis import B_ROBUST, TRIO, fmt, ror, train_table
from qevasion.loader import DATA_CACHE, load_qevasion
from qevasion.paths import ROOT, RUNS

HYPO = re.compile(r"\b(?:if|would|suppose|supposing|hypothetical(?:ly)?)\b", re.I)
FUTURE = re.compile(r"\b(?:will|going to|plan(?:s|ning)?|next)\b", re.I)
WORD = re.compile(r"[a-z]{5,}")


def words(s: str) -> set[str]:
    return set(WORD.findall(str(s).lower()))


def main() -> None:
    train = load_qevasion(DATA_CACHE).train.reset_index(drop=True)
    d = train_table()
    d = d[(d.confirm == 1) & d.model.isin(TRIO)].reset_index(drop=True)
    src = d["src_i"].to_numpy()
    q = train["question"].astype(str).to_numpy()
    d["multi"] = train["multiple_questions"].astype(str).str.lower().eq("true").to_numpy()[src].astype(int)
    d["affirm"] = train["affirmative_questions"].astype(str).str.lower().eq("true").to_numpy()[src].astype(int)
    d["hypo"] = np.array([bool(HYPO.search(x)) for x in q])[src].astype(int)
    d["future"] = np.array([bool(FUTURE.search(x)) for x in q])[src].astype(int)
    d["president"] = train["president"].astype(str).to_numpy()[src]
    L = [f"E36 (exploratory): confirmation sample, CF, trio; {len(d)} answers", "", "1. ROR P/N within subgroups"]
    subs = {}
    for col in ("you", "multi", "affirm", "hypo", "future"):
        for v in (1, 0):
            g = d[d[col] == v]
            subs[f"{col}={v}"] = (ror(g, "P", "N", "evade_cf", n_boot=B_ROBUST), int(g.src_i.nunique()))
    for p in sorted(d.president.unique()):
        g = d[d.president == p]
        subs[f"President={p}"] = (ror(g, "P", "N", "evade_cf", n_boot=B_ROBUST), int(g.src_i.nunique()))
    ps = {k: v[0]["p_one_sided"] for k, v in subs.items()}
    order = sorted(ps, key=ps.get)
    holm, run = {}, 0.0
    for j, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - j) * ps[k]))
        holm[k] = run
    res = {"subgroups": {k: {**v[0], "n_subq": v[1], "p_holm": holm[k]} for k, v in subs.items()}}
    for k, (r, n) in subs.items():
        L.append(f"   {k:<34} n {n:>5}  ROR {fmt(r)}  p {ps[k]:.4f}  Holm {holm[k]:.4f}")
    real = train["interview_answer"].astype(str).to_numpy()
    urls = train["url"].astype(str).to_numpy()
    rng = np.random.default_rng(36)
    other = np.empty(len(train), int)
    for i in range(len(train)):
        while True:
            j = int(rng.integers(len(train)))
            if urls[j] != urls[i]:
                other[i] = j
                break
    ov, ovr = [], []
    for a, i in zip(d["raw_answer"], src):
        w = words(a)
        ov.append(len(w & words(real[i])) / max(1, len(w)))
        ovr.append(len(w & words(real[other[i]])) / max(1, len(w)))
    d["overlap"], d["overlap_rand"] = ov, ovr
    t = d.groupby("condition")[["overlap", "overlap_rand", "meta", "n_words"]].mean()
    L += ["", "2-3. By role: word overlap with the President's real answer / with a random real answer; meta-answer share; words"]
    L += ["   " + s for s in t.round(3).to_string().splitlines()]
    res["by_role"] = {c: {k: float(v) for k, v in row.items()} for c, row in t.iterrows()}
    (ROOT / "docs" / "raw" / "E36_drivers.txt").write_text("\n".join(L) + "\n", encoding="utf8")
    (RUNS / "E34" / "e36.json").write_text(json.dumps(res, indent=1, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    main()
