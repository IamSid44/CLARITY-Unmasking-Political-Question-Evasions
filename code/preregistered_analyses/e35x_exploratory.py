#!/usr/bin/env python3
"""E35x (exploratory, post hoc: written 2026-11-22 after H35-1 FAILED on train). Not registered.

Why H35-1 failed is not identified by E35's design: the dev contrast used M1 (ten seeds, trained on all of train), the train
test used CF (one seed per fold, trained on four fifths of train), and the two samples define sibling pairs differently
(three annotators' disjoint sets vs one annotator's different labels). This separates the instrument from the sample:
CF's dev binding (mean of the five fold models, and each fold model alone), on E35's dev pairs and matched null.

    python -m preregistered_analyses.e35x_exploratory      # CPU, seconds; writes docs/raw/E35x_exploratory.txt
"""

from __future__ import annotations

import json

import numpy as np

from audit.validity_battery import DISJ, GOLD
from preregistered_analyses.e34_analysis import cf_labels, cf_tau
from preregistered_analyses.e35_binding import DEV, answer_id, dev_systems, fmt, null_pairs, stats
from qevasion.paths import ROOT, RUNS
from qevasion.scoring import score_subtask2


def main() -> None:
    tau = cf_tau()
    folds = [np.load(RUNS / "E33_cf_url" / f"fold{k}" / "dev_probs.npy") for k in range(5)]
    labels, s2 = dev_systems()
    keep = ("M1 (Qwen3-8B LoRA, 10 seeds)", "llm27b_zeroshot (1 seeds)")
    labels = {k: labels[k] for k in keep}
    labels["CF (mean of 5 fold models)"] = cf_labels(np.mean(folds, 0), tau)[0]
    for k, p in enumerate(folds):
        labels[f"CF fold {k} alone"] = cf_labels(p, tau)[0]
    s2 = {k: score_subtask2(y, GOLD).macro_f1 for k, y in labels.items()}
    aid = answer_id(DEV)
    nulls = null_pairs(GOLD, aid, np.arange(len(DEV)))
    st = stats(labels, np.asarray(DISJ), nulls, aid)
    L = [f"E35x (exploratory, post hoc): CF's binding on the {len(DISJ)} dev sibling pairs, E35's matched null and draws; "
         f"tau {tau:.2f} (frozen)", ""]
    for k, r in st.items():
        L.append(f"   {k:<34} S2 {s2[k]:.3f}  {fmt(r)}")
    res = {k: {kk: vv for kk, vv in r.items() if kk != "_bs"} | {"s2": s2[k]} for k, r in st.items()}
    L.append("")
    for a, b in (("CF (mean of 5 fold models)", "llm27b_zeroshot (1 seeds)"),
                 ("CF (mean of 5 fold models)", "M1 (Qwen3-8B LoRA, 10 seeds)")):
        d = st[a]["_bs"] - st[b]["_bs"]
        lo, hi = np.percentile(d, [2.5, 97.5])
        L.append(f"   {a} - {b}: {st[a]['binding'] - st[b]['binding']:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        res[f"{a} - {b}"] = {"diff": float(st[a]["binding"] - st[b]["binding"]), "ci": [float(lo), float(hi)]}
    single = [st[f"CF fold {k} alone"]["binding"] for k in range(5)]
    L.append(f"   single fold models: binding {min(single):.3f}-{max(single):.3f} (mean {np.mean(single):.3f})")
    (ROOT / "docs" / "raw" / "E35x_exploratory.txt").write_text("\n".join(L) + "\n", encoding="utf8")
    (RUNS / "E35" / "e35x.json").write_text(json.dumps(res, indent=1, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    main()
