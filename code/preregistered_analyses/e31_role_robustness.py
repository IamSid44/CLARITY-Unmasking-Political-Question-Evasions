#!/usr/bin/env python3
"""E31 (registered 2026-11-20, docs/experiment_log.md): robustness of the E25b role effect.

    python -m preregistered_analyses.e31_role_robustness [--llm-table runs/E32/llm_table.parquet]
        # CPU, ~2 min; writes docs/raw/E31_role_robustness.txt and runs/E31/role_robustness.json

Odds ratio of an LLM evading (instrument clarity != Clear Reply) on sub-questions the President evaded (gold) vs
answered, per condition, per model and pooled (Mantel-Haenszel over models), under six specifications:
  (a) raw; (b) MH over model x tercile of the LLM answer's length; (c) logistic regression with log LLM-answer
  length, log President-answer length and model; (d) (c) + the instrument's question-only prior (qonly P(Explicit));
  (e) raw, without the answers containing the token "nan" (E25b prompts rendered the empty president/date fields as
  "nan"); (f) raw, the President's evasion as labelled by the instrument instead of gold.
Intervals: cluster bootstrap over the 191 dev interview turns (2,000 draws, e25_analysis.BOOT).
With --llm-table (E32), the same is computed on the extended table (three models) and written with suffix _e32.
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from preregistered_analyses.e25_analysis import DEV, boot_stat, llm_table, mh_or, odds_ratio
from qevasion.paths import ROOT, RUNS

PRES_WORDS = DEV["interview_answer"].astype(str).str.split().str.len().to_numpy()


def table(path: str | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path) if path else llm_table()
    df = df.reset_index(drop=True)
    df["pres_words"] = PRES_WORDS[df["src_i"].to_numpy()]
    txt = df["raw_answer"] if "raw_answer" in df else df["interview_answer"]
    df["has_nan"] = txt.astype(str).str.contains(r"\bnan\b", flags=re.I, regex=True).astype(int)
    df["len_t"] = df.groupby(["model", "condition"])["n_words"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 3, labels=False))
    df["stratum"] = df["model"] + "_" + df["len_t"].astype(str)
    return df


def logit_or(d: pd.DataFrame, extra: tuple[str, ...] = ()) -> float:
    if d["pres_evade"].nunique() < 2 or d["evade"].nunique() < 2:
        return float("nan")
    models = sorted(d["model"].unique())
    X = np.column_stack([d["pres_evade"], np.log1p(d["n_words"]), np.log1p(d["pres_words"])]
                        + [d[c] for c in extra] + [(d["model"] == m).astype(int) for m in models[1:]])
    return float(np.exp(LogisticRegression(C=1e6, max_iter=3000).fit(X, d["evade"]).coef_[0][0]))


SPECS = {
    "a raw": lambda d: mh_or(d, "evade", "pres_evade") if d["model"].nunique() > 1 else odds_ratio(d["evade"].to_numpy(), d["pres_evade"].to_numpy()),
    "b MH length terciles": lambda d: mh_or(d, "evade", "pres_evade", strata="stratum"),
    "c logistic + lengths": lambda d: logit_or(d),
    "d (c) + question prior": lambda d: logit_or(d, ("qonly_p_cr",)),
    "e raw, no 'nan' answers": lambda d: SPECS["a raw"](d[d["has_nan"] == 0]),
    "f raw, President by instrument": lambda d: mh_or(d, "evade", "pres_evade_inst") if d["model"].nunique() > 1 else odds_ratio(d["evade"].to_numpy(), d["pres_evade_inst"].to_numpy()),
}


def compute(df: pd.DataFrame, conds=("N", "P", "F", "B")) -> tuple[list[str], dict]:
    res: dict = {}
    models = sorted(df["model"].unique())
    L = [f"E31 role-effect robustness: {len(df)} LLM answers, models {models}; answers containing 'nan': "
         + ", ".join(f"{m} {int(df[df.model == m].has_nan.sum())}" for m in models), "",
         "odds ratio [95% turn-bootstrap interval]; columns: pooled (MH over models) and per model", ""]
    for c in conds:
        g = df[df["condition"] == c].reset_index(drop=True)
        L.append(f"condition {c}")
        for name, fn in SPECS.items():
            row = {}
            for who in ["pooled"] + models:
                d = g if who == "pooled" else g[g["model"] == who].reset_index(drop=True)
                v = fn(d)
                bs = np.array(boot_stat(d, fn), float)
                lo, hi = np.nanpercentile(bs, [2.5, 97.5])
                row[who] = {"or": float(v), "ci": [float(lo), float(hi)]}
            res.setdefault(c, {})[name] = row
            L.append(f"   {name:<32}" + "   ".join(f"{w[:10]} {r['or']:.2f} [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]"
                                                    for w, r in row.items()))
        L.append("")
    robust = ["b MH length terciles", "c logistic + lengths", "d (c) + question prior", "e raw, no 'nan' answers"]
    p_ok = all(res["P"][s]["pooled"]["or"] >= 1.5 and res["P"][s]["pooled"]["ci"][0] > 1 for s in robust)
    n_ok = all(res["N"][s]["pooled"]["or"] < 1.5 for s in ["a raw"] + robust)
    L.append(f"H31-1 (pooled OR(P) >= 1.5 with interval above 1 under (b)-(e), and OR(N) < 1.5 throughout): "
             f"{'HELD' if p_ok and n_ok else 'FAILED'} (P part {'held' if p_ok else 'failed'}, N part {'held' if n_ok else 'failed'})")
    res["_verdict"] = {"p": p_ok, "n": n_ok}
    return L, res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-table", default=None)
    a = ap.parse_args()
    sfx = "_e32" if a.llm_table else ""
    L, res = compute(table(a.llm_table))
    print("\n".join(L))
    raw = ROOT / "docs" / "raw" / f"E31_role_robustness{sfx}.txt"
    out = RUNS / "E31" / f"role_robustness{sfx}.json"
    raw.write_text("\n".join(L) + "\n", encoding="utf8")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {raw} and {out}")


if __name__ == "__main__":
    main()
