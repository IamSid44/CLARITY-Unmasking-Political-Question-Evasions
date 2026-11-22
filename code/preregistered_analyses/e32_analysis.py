#!/usr/bin/env python3
"""E32 (registered 2026-11-20, docs/experiment_log.md): the role audit in a third model family (OLMo-2-7B-Instruct).

    python -m preregistered_analyses.e32_analysis       # CPU; writes docs/raw/E32_analysis.txt, runs/E32/analysis.json,
                                                        # runs/E32/llm_table.parquet (three models, protocol A, for E31)

Inputs: runs/E32 (frames/, m1/, labels/) written by audit.generate / audit.frames / audit.infer / audit.system with
CLARITY_E25_DIR=runs/E32 (code/launch/e32_label.sh), and the E25 labels for Mistral-7B and Qwen3-8B.
  A  hardware control: 200 Mistral answers (50 per condition) re-labelled on the lab RTX PRO 6000 vs their E25 (H100)
     labels.
  B  OLMo, protocol A (E25 prompts unchanged, model name olmo2_7b) and protocol B (no name/date, olmo2_7b_generic):
     evasion rate per condition; odds ratio of evading on questions the President evaded (gold), per condition, with
     2,000-draw cluster bootstrap over the 191 dev turns; pursuit.
  C  H32-1 (OLMo A: OR(P) >= 1.5, interval above 1), H32-2 (OLMo A: OR(N) <= 1.3), H32-3 (MH OR(P) over the three
     models >= 1.5, interval above 1).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from preregistered_analyses.e25_analysis import (CONDS, CR, EVASION_LABELS, GOLD_CLAR, TURN, boot_stat, clarity, ci,
                                                 lab, llm_table, mh_or, odds_ratio)
from qevasion.paths import ROOT, RUNS

E32 = RUNS / "E32"
RAW = ROOT / "docs" / "raw" / "E32_analysis.txt"


def control() -> tuple[list[str], dict]:
    p = E32 / "labels" / "ctrl_mistral.npz"
    if not p.exists():
        return ["A. hardware control: not run"], {}
    z = np.load(p)
    fr = pd.read_parquet(E32 / "frames" / "ctrl_mistral.parquet")
    e25 = lab("llm")
    rows = fr["e25_row"].to_numpy()
    agree = float(np.mean(z["pred"] == e25["pred"][rows]))
    agree_m1 = float(np.mean(z["m1_pred"] == e25["m1_pred"][rows]))
    ev_new, ev_old = clarity(z["pred"]) != CR, clarity(e25["pred"][rows]) != CR
    res = {"n": len(rows), "label_agree": agree, "m1_label_agree": agree_m1,
           "evade_agree": float(np.mean(ev_new == ev_old)), "evade_rate_new": float(ev_new.mean()),
           "evade_rate_e25": float(ev_old.mean()), "defer_new": float(z["defer"].mean()),
           "defer_agree": float(np.mean(z["defer"] == e25["defer"][rows]))}
    L = ["A. Hardware control: 200 Mistral-7B answers (50 per condition) re-labelled on the lab RTX PRO 6000 vs E25 (H100)",
         f"   instrument label agreement {agree:.3f} (M1 alone {agree_m1:.3f}); evade / not-evade agreement "
         f"{res['evade_agree']:.3f}; evasion rate {res['evade_rate_e25']:.3f} (E25) vs {res['evade_rate_new']:.3f} (lab); "
         f"routing identical on {res['defer_agree']:.3f}", "   (E25's own cross-GPU agreement on dev: 0.981)", ""]
    return L, res


def olmo_table() -> pd.DataFrame | None:
    p = E32 / "labels" / "llm.npz"
    if not p.exists():
        return None
    fr = pd.read_parquet(E32 / "frames" / "llm.parquet")
    z = np.load(p)
    fr["pred"] = z["pred"]
    fr["clar"] = clarity(z["pred"])
    fr["evade"] = (fr["clar"] != CR).astype(int)
    src = fr["src_i"].to_numpy()
    fr["turn"] = TURN[src]
    fr["pres_evade"] = (GOLD_CLAR[src] != CR).astype(int)
    own = lab("dev_own")
    fr["pres_evade_inst"] = (clarity(own["pred"])[src] != CR).astype(int)
    q = lab("dev_qonly")
    fr["qonly_evade"] = (clarity(q["pred"])[src] != CR).astype(int)
    fr["qonly_p_cr"] = q["pbar"][src][:, [i for i, e in enumerate(EVASION_LABELS) if e == "Explicit"]].sum(1)
    return fr


def or_row(g: pd.DataFrame) -> dict:
    v = odds_ratio(g["evade"].to_numpy(), g["pres_evade"].to_numpy())
    bs = np.array(boot_stat(g.reset_index(drop=True), lambda d: odds_ratio(d["evade"].to_numpy(), d["pres_evade"].to_numpy())))
    return {"or": float(v), "ci": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]}


def compute() -> tuple[list[str], dict]:
    L, ctrl = control()
    res = {"control": ctrl}
    fr = olmo_table()
    if fr is None:
        L.append("B. OLMo: not labelled")
        return L, res
    L.append(f"B. OLMo-2-7B-Instruct: {len(fr)} answers; cut at 400 tokens {(fr['finish'] == 'length').mean():.3f}; "
             f"answers containing 'nan': " + ", ".join(f"{m} {int(fr[fr.model == m].raw_answer.str.contains(r'\bnan\b', case=False, regex=True).sum())}"
                                                       for m in sorted(fr.model.unique())))
    tab = fr.groupby(["model", "condition"])["evade"].mean().unstack().reindex(columns=CONDS)
    L.append("   evasion rate (clarity != Clear Reply) by condition; Presidents (gold) 0.744\n"
             + "\n".join("   " + line for line in tab.round(3).to_string().splitlines()))
    res["rates"] = {m: {c: float(tab.loc[m, c]) for c in CONDS} for m in tab.index}
    for m in sorted(fr.model.unique()):
        res.setdefault("or", {})[m] = {}
        for c in CONDS:
            r = or_row(fr[(fr.model == m) & (fr.condition == c)])
            res["or"][m][c] = r
        L.append(f"   {m}: odds ratio (evades | President evaded), by condition: " + "  ".join(
            f"{c} {res['or'][m][c]['or']:.2f} [{res['or'][m][c]['ci'][0]:.2f}, {res['or'][m][c]['ci'][1]:.2f}]" for c in CONDS))
        piv = fr[fr.model == m].pivot_table(index="src_i", columns="condition", values="clar", aggfunc="first")
        ev = piv[piv["P"] != CR]
        res.setdefault("pursuit", {})[m] = float((ev["F"] == CR).mean())
        L.append(f"      pursuit: of P answers that evade, F gives a Clear Reply in {res['pursuit'][m]:.3f} (n = {len(ev)}); "
                 f"P - N evasion rate {tab.loc[m, 'P'] - tab.loc[m, 'N']:+.3f}")
    # three-model table (protocol A) and H32
    e25 = llm_table()
    a = fr[fr.model == "olmo2_7b"]
    three = pd.concat([e25, a[e25.columns.intersection(a.columns)]], ignore_index=True)
    three.to_parquet(E32 / "llm_table.parquet")
    gP = three[three.condition == "P"].reset_index(drop=True)
    mh = mh_or(gP, "evade", "pres_evade")
    bs = np.array(boot_stat(gP, lambda d: mh_or(d, "evade", "pres_evade")))
    lo3, hi3 = np.nanpercentile(bs, [2.5, 97.5])
    gN = three[three.condition == "N"].reset_index(drop=True)
    mhN = mh_or(gN, "evade", "pres_evade")
    bsN = np.array(boot_stat(gN, lambda d: mh_or(d, "evade", "pres_evade")))
    res["three_model"] = {"P": {"mh": float(mh), "ci": [float(lo3), float(hi3)]},
                          "N": {"mh": float(mhN), "ci": [float(np.nanpercentile(bsN, 2.5)), float(np.nanpercentile(bsN, 97.5))]}}
    oP, oN = res["or"]["olmo2_7b"]["P"], res["or"]["olmo2_7b"]["N"]
    h1 = oP["or"] >= 1.5 and oP["ci"][0] > 1
    h2 = oN["or"] <= 1.3
    h3 = mh >= 1.5 and lo3 > 1
    res["_verdict"] = {"h32_1": bool(h1), "h32_2": bool(h2), "h32_3": bool(h3)}
    L += ["", "C. Registered hypotheses (protocol A)",
          f"   H32-1 OLMo OR(P) >= 1.5, interval above 1: {oP['or']:.2f} [{oP['ci'][0]:.2f}, {oP['ci'][1]:.2f}] -> {'HELD' if h1 else 'FAILED'}",
          f"   H32-2 OLMo OR(N) <= 1.3: {oN['or']:.2f} [{oN['ci'][0]:.2f}, {oN['ci'][1]:.2f}] -> {'HELD' if h2 else 'FAILED'}",
          f"   H32-3 MH OR(P) over Mistral, Qwen, OLMo >= 1.5, interval above 1: {mh:.2f} [{lo3:.2f}, {hi3:.2f}] -> {'HELD' if h3 else 'FAILED'}"
          f"   (same pooling in N: {mhN:.2f} {ci(bsN)})",
          f"   decision: {'replicated in a third model family' if (h1 and h2) else 'the role effect stays exploratory'}"]
    return L, res


def main() -> None:
    L, res = compute()
    print("\n".join(L))
    RAW.write_text("\n".join(L) + "\n", encoding="utf8")
    (E32 / "analysis.json").write_text(json.dumps(res, indent=1))
    print(f"\nwrote {RAW} and {E32 / 'analysis.json'}")


if __name__ == "__main__":
    main()
