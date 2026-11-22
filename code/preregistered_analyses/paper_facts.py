#!/usr/bin/env python3
"""Descriptive facts the paper states (sample sizes, overlaps, rates), computed from the data and registered outputs,
so every such number has one source.

    python -m preregistered_analyses.paper_facts      # CPU, seconds; writes docs/raw/paper_facts.txt
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qevasion.labels import CLARITY_LABELS, leaf_to_official_clarity
from qevasion.loader import DATA_CACHE, load_qevasion
from qevasion.paths import ROOT, RUNS

CR = CLARITY_LABELS.index("Clear Reply")


def main() -> None:
    sp = load_qevasion(DATA_CACHE)
    tr, dv = sp.train, sp.dev
    d = pd.to_datetime(tr["date"], errors="coerce")
    dev_urls = set(dv["url"].astype(str))
    conf = ~tr["url"].astype(str).isin(dev_urls)
    turns = lambda df: (df["interview_question"].astype(str).str.split().str.join(" ") + "|||"  # noqa: E731
                        + df["interview_answer"].astype(str).str.split().str.join(" ")).nunique()
    L = ["Paper facts (descriptive)", "",
         f"train: {len(tr)} items, {tr['url'].nunique()} interviews, {turns(tr)} turns; dates {d.min():%Y-%m-%d} to {d.max():%Y-%m-%d}; "
         f"Presidents: " + ", ".join(f"{k} {v}" for k, v in tr['president'].value_counts().items()),
         f"   titles containing 'News Conference': {tr['title'].astype(str).str.contains('News Conference').mean():.3f}",
         f"   President evaded (clarity != Clear Reply): {(tr['clarity_label'] != 'Clear Reply').mean():.3f}",
         f"dev: {len(dv)} items, {dv['url'].nunique()} interviews, {turns(dv)} turns; president/date empty: "
         f"{dv['president'].isna().all() or (dv['president'].astype(str).str.strip() == '').all()}",
         f"dev interviews also in train: {len(dev_urls & set(tr['url'].astype(str)))} of {len(dev_urls)} "
         f"({int((~conf).sum())} train rows)",
         f"confirmation sample (train interviews not in dev): {tr.loc[conf, 'url'].nunique()} interviews, {int(conf.sum())} items, "
         f"{turns(tr[conf])} turns; President evaded {(tr.loc[conf, 'clarity_label'] != 'Clear Reply').mean():.3f}"]
    t = pd.read_parquet(RUNS / "E32" / "llm_table.parquet")
    t = t[t["condition"].isin(["N", "P"])]
    t["ev"] = (leaf_to_official_clarity(t["pred"].to_numpy()) != CR).astype(int)
    r = t.groupby("condition")["ev"].mean()
    rm = t.groupby(["model", "condition"])["ev"].mean().unstack()
    L += ["", "Discovery (dev, frozen instrument, protocol A): evasion rate N {:.3f}, P {:.3f}; per model:".format(r["N"], r["P"])]
    L += ["   " + s for s in rm.round(3).to_string().splitlines()]
    from audit.validity_battery import DISJ, GOLD
    from qevasion.scoring import score_subtask2

    z = np.load(RUNS / "E30" / "llm27b_zeroshot" / "dev_own" / "seed0.npy").argmax(1)
    L += ["", f"27B zero-shot on dev, unadjusted argmax (as used in E34/E35-train): S2 {score_subtask2(z, GOLD).macro_f1:.3f}, "
              f"sibling same-label {np.mean([z[i] == z[j] for i, j in DISJ]):.3f}, Clear-Reply share "
              f"{np.mean(leaf_to_official_clarity(z) == CR):.3f}"]
    p = RUNS / "E34" / "frames" / "llm.parquet"
    if p.exists():
        f = pd.read_parquet(p)
        L += ["", f"E34 answers: {len(f)}; per model x condition:"] + \
             ["   " + s for s in f.groupby(["model", "condition"]).size().unstack().to_string().splitlines()]
        L.append(f"   cut at 400 tokens: {(f['finish'] == 'length').mean():.3f}; median words "
                 f"{f.groupby('condition')['n_words'].median().to_dict()}")
    out = ROOT / "docs" / "raw" / "paper_facts.txt"
    out.write_text("\n".join(L) + "\n", encoding="utf8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
