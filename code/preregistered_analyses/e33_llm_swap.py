#!/usr/bin/env python3
"""E33 H33-1 (registered 2026-11-21): does the cross-fit instrument (CF) read the question on LLM-written answers?

    python -m preregistered_analyses.e33_llm_swap     # CPU; needs runs/E34/cf/{dev_llm,dev_llm_rand}.npz, runs/E33/cf_tau.json

The dev N and P answers of the three discovery models (runs/E34/frames/dev_llm.parquet) are scored under their own
sub-question and under the E25 dev_rand swap (another interview's sub-question and turn). Held if CF's Clear-Reply share
at least halves under the swap, separately in N and in P. Also reported: label persistence, and the same per model.
Writes docs/raw/E33_llm_swap.txt and runs/E33/llm_swap.json.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from preregistered_analyses.e34_analysis import CR, cf_labels, cf_tau
from qevasion.labels import leaf_to_official_clarity
from qevasion.paths import ROOT, RUNS


def main() -> None:
    fr = pd.read_parquet(RUNS / "E34" / "frames" / "dev_llm.parquet")
    tau = cf_tau()
    own, _ = cf_labels(np.load(RUNS / "E34" / "cf" / "dev_llm.npz")["probs"], tau)
    swp, _ = cf_labels(np.load(RUNS / "E34" / "cf" / "dev_llm_rand.npz")["probs"], tau)
    fr["cr_own"] = (leaf_to_official_clarity(own) == CR).astype(int)
    fr["cr_swap"] = (leaf_to_official_clarity(swp) == CR).astype(int)
    fr["persist"] = (own == swp).astype(int)
    L = [f"E33 H33-1: CF on the dev LLM answers (3 models x N/P, {len(fr)} answers), own question vs another interview's",
         f"(tau {tau:.2f} frozen)", ""]
    res = {}
    for c, g in fr.groupby("condition"):
        a, b = g["cr_own"].mean(), g["cr_swap"].mean()
        ok = b <= a / 2
        res[c] = {"cr_own": float(a), "cr_swap": float(b), "persist": float(g["persist"].mean()), "held": bool(ok)}
        L.append(f"   {c}: Clear-Reply share {a:.3f} -> {b:.3f}; labels kept {g['persist'].mean():.3f} -> "
                 f"{'HELD' if ok else 'FAILED'} (registered: at least halves)")
    t = fr.groupby(["model", "condition"])[["cr_own", "cr_swap", "persist"]].mean()
    L += ["", "   per model:"] + ["      " + s for s in t.round(3).to_string().splitlines()]
    held = all(r["held"] for r in res.values())
    L.append(f"\nH33-1: {'HELD' if held else 'FAILED'}")
    res["held"] = bool(held)
    (ROOT / "docs" / "raw" / "E33_llm_swap.txt").write_text("\n".join(L) + "\n", encoding="utf8")
    (RUNS / "E33").mkdir(parents=True, exist_ok=True)
    (RUNS / "E33" / "llm_swap.json").write_text(json.dumps(res, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
