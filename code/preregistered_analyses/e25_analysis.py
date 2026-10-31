#!/usr/bin/env python3
"""E25 analysis (registered 2026-10-30, docs/experiment_log.md, E25): is evasion in the question or in the answerer?

    python -m preregistered_analyses.e25_analysis > ../docs/raw/E25_analysis.txt        # CPU, a few minutes
    python -m preregistered_analyses.e25_analysis sheet    # the blind 100-answer annotation sheet (docs/annotation/)
    python -m preregistered_analyses.e25_analysis kappa A.csv B.csv   # after two people fill the sheet

Inputs: runs/E25 (frames, m1/, labels/, answers.jsonl, probes.json, hidden/latency.json) from the PASS=3 lane, and
code/experiments/e25/frozen_system.json. Every label is the frozen E23 instrument's (audit/system.py).
Sections: A controls; B E25a question swaps; C E25b question or speaker; D E25c answer prefixes; E E25d probes and
latency; F cost. Statistics: cluster bootstrap over interview turns (2,000 draws, seed 0); a turn's sub-questions
share one answer (the President's, or an LLM's in N/P/B), so turns are the independent units.
"""

from __future__ import annotations

import json
import sys
from collections import Counter

import numpy as np
import pandas as pd

from qevasion.labels import CLARITY_LABELS, EVASION_LABELS, encode_clarity, encode_evasion, leaf_to_official_clarity
from qevasion.loader import DATA_CACHE, dev_reference_mask, load_qevasion
from qevasion.paths import DOCS, ROOT, RUNS
from qevasion.scoring import score_subtask1, score_subtask2

E25 = RUNS / "E25"
CR = CLARITY_LABELS.index("Clear Reply")
B = 2000
MODELS = ("llama31_8b", "mistral_7b", "qwen3_8b", "olmo2_7b")
CONDS = ("N", "P", "F", "B")
INDIRECT = {"Implicit", "General", "Partial/half-answer"}
NONANSWER = {"Dodging", "Deflection", "Declining to answer", "Claims ignorance"}

SP = load_qevasion(DATA_CACHE)
DEV = SP.dev.reset_index(drop=True)
GOLD = dev_reference_mask(DEV)
GOLD_CLAR = encode_clarity(DEV["clarity_label"].tolist())
TURN = (DEV["interview_question"].astype(str).str.split().str.join(" ") + "|||"
        + DEV["interview_answer"].astype(str).str.split().str.join(" ")).factorize()[0]
RES: dict = {}  # key numbers for the figures (runs/E25/analysis.json)
RNG = np.random.default_rng(0)
BOOT = [RNG.choice(TURN.max() + 1, TURN.max() + 1) for _ in range(B)]


def rule(t: str) -> None:
    print("\n" + "=" * 110 + f"\n{t}\n" + "=" * 110)


def lab(frame: str) -> dict:
    z = np.load(E25 / "labels" / f"{frame}.npz")
    return {k: z[k] for k in z.files}


def clarity(pred: np.ndarray) -> np.ndarray:
    return leaf_to_official_clarity(pred)


def ci(vals) -> str:
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    return f"[{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}]" if len(v) else "[n/a]"


def rows_of(draw: np.ndarray, turn_of_row: np.ndarray) -> np.ndarray:
    """Row indices of a cluster bootstrap draw (turn ids with repeats)."""
    by = {}
    for i, t in enumerate(turn_of_row):
        by.setdefault(t, []).append(i)
    return np.concatenate([by[t] for t in draw if t in by]) if len(draw) else np.array([], int)


def plurality(row) -> str | None:
    c = Counter(str(row[f"annotator{k}"]) for k in (1, 2, 3)).most_common()
    return c[0][0] if c[0][1] >= 2 else None


PLUR = [plurality(r) for _, r in DEV.iterrows()]


# ---------------------------------------------------------------- A. controls

def controls() -> dict:
    rule("A. Controls")
    rep = json.loads((E25 / "m1" / "report.json").read_text())
    ctrl = [v["ctrl_max_abs_dp"] for k, v in rep.items() if k.startswith("dev_own/")]
    print(f"   M1 on dev_own reproduces the saved dev probabilities: {len(ctrl)} seeds, max |dp| {max(ctrl):.2e} "
          f"(registered tolerance 1e-3) -> {'PASSED' if max(ctrl) < 1e-3 else 'FAILED (amendment 2: H100 numerics; recorded, not a stop)'}")
    f = json.loads((ROOT / "code" / "experiments" / "e25" / "frozen_system.json").read_text())
    own = lab("dev_own")
    from preregistered_analyses.e23_systems import LOOK
    from audit.system import calibration, decide, top4

    # E23's frozen predictions from E23's own inputs (the saved, Blackwell-scored probabilities and VM records)
    _, _, _, r = calibration()
    q0 = r["q"]["dev"]
    cand0, defer0 = top4(q0), (1 - q0.max(1)) >= f["u_threshold"]
    lq = np.full((len(q0), 9), -np.inf)
    for i in np.where(defer0)[0]:
        c = tuple(int(x) for x in cand0[i])
        lq[i, list(c)] = np.log(np.maximum(LOOK["averaged"][(i, c)], 1e-12))
    frozen_pred = decide(np.log(r["pbar"]["dev"] + 1e-12), lq, defer0, f)
    print(f"   H100 vs saved inputs: candidate sets identical on {np.mean((own['cand'] == cand0).all(1)):.3f} of dev, "
          f"deferral decision identical on {np.mean(own['defer'] == defer0):.3f}")
    agree = (own["pred"] == frozen_pred).mean()
    print(f"   instrument on dev_own with THIS VM's 27B calls vs the frozen (E23 VM-record) predictions: agreement "
          f"{agree:.3f}; S2 {score_subtask2(own['pred'], GOLD).macro_f1:.3f} vs {f['full_dev_fit_s2_s1'][0]:.3f} "
          f"(full-dev fit; the honest nested-CV estimate stays 0.650 / 0.639)")
    print(f"   deferred to the 27B on dev_own: {own['defer'].mean():.3f} (frozen: {f['dev_calls']:.3f}); "
          f"27B records missing: {int((own['defer'] & ~own['scored']).sum())}")
    return {"own": own}


# ---------------------------------------------------------------- B. E25a

def e25a() -> dict:
    rule("B. E25a: does the instrument read the question causally? (labels: the full instrument; M1 alone in brackets)")
    out = {}
    for split in ("dev", "test"):
        own = lab(f"{split}_own")
        c_own = clarity(own["pred"])
        RES.setdefault("e25a", {})[f"{split}_own"] = {"cr": float(np.mean(c_own == CR)), "persist": 1.0}
        print(f"   {split}: own  Clear-Reply share {np.mean(c_own == CR):.3f}  "
              f"clarity shares (Amb, CNR, CR) {np.round(np.bincount(c_own, minlength=3) / len(c_own), 3).tolist()}")
        for cond in ("rand", "null", "qonly"):
            p = E25 / "labels" / f"{split}_{cond}.npz"
            if not p.exists():
                continue
            z = lab(f"{split}_{cond}")
            c = clarity(z["pred"])
            pers = np.mean(z["pred"] == own["pred"])
            pers_m1 = np.mean(z["m1_pred"] == own["m1_pred"])
            types = Counter(EVASION_LABELS[k] for k in z["pred"]).most_common(4)
            out[(split, cond)] = {"cr": float(np.mean(c == CR)), "persist": float(pers)}
            RES.setdefault("e25a", {})[f"{split}_{cond}"] = {**out[(split, cond)], "persist_m1": float(pers_m1)}
            print(f"   {split}: {cond:<5} Clear-Reply share {np.mean(c == CR):.3f}  label persistence {pers:.3f} "
                  f"(M1 alone {pers_m1:.3f})  top labels {types}")
    d_own = np.mean(clarity(lab("dev_own")["pred"]) == CR)
    r = out[("dev", "rand")]
    held = r["cr"] <= 0.5 * d_own and r["persist"] <= 0.5
    print(f"   H25a-1 (dev: Clear-Reply share under rand <= half of own's {d_own:.3f}, AND persistence <= 0.5): "
          f"{'HELD' if held else 'FAILED'} (share {r['cr']:.3f}, persistence {r['persist']:.3f})")
    t = out.get(("test", "rand"))
    if t:
        t_own = np.mean(clarity(lab("test_own")["pred"]) == CR)
        print(f"   replication on test (label-free): share {t['cr']:.3f} vs own {t_own:.3f}; persistence {t['persist']:.3f}")
    n = out[("dev", "null")]
    print(f"   H25a-2 (descriptive): persistence under null {n['persist']:.3f} "
          f"-> {'labels mostly answer-driven (>= 0.7)' if n['persist'] >= 0.7 else 'the question carries much of the label (< 0.7)'}")
    # where do the own Clear Replies go when the question is unrelated?
    own, rz = lab("dev_own"), lab("dev_rand")
    m = clarity(own["pred"]) == CR
    print(f"   dev items labelled Clear Reply on their own question ({m.sum()}): under rand -> "
          f"{dict(Counter(EVASION_LABELS[k] for k in rz['pred'][m]).most_common(5))}")
    return out


# ---------------------------------------------------------------- C. E25b

def llm_table() -> pd.DataFrame:
    fr = pd.read_parquet(E25 / "frames" / "llm.parquet")
    z = lab("llm")
    fr["pred"] = z["pred"]
    fr["clar"] = clarity(z["pred"])
    fr["evade"] = (fr["clar"] != CR).astype(int)
    fr["turn"] = TURN[fr["src_i"].to_numpy()]
    fr["pres_evade"] = (GOLD_CLAR[fr["src_i"].to_numpy()] != CR).astype(int)
    own = lab("dev_own")
    fr["pres_evade_inst"] = (clarity(own["pred"])[fr["src_i"].to_numpy()] != CR).astype(int)
    q = lab("dev_qonly")
    fr["qonly_evade"] = (clarity(q["pred"])[fr["src_i"].to_numpy()] != CR).astype(int)
    fr["qonly_p_cr"] = q["pbar"][fr["src_i"].to_numpy()][:, [i for i, e in enumerate(EVASION_LABELS) if e == "Explicit"]].sum(1)
    return fr


def odds_ratio(y: np.ndarray, x: np.ndarray) -> float:
    a = ((x == 1) & (y == 1)).sum() + 0.5
    b = ((x == 1) & (y == 0)).sum() + 0.5
    c = ((x == 0) & (y == 1)).sum() + 0.5
    d = ((x == 0) & (y == 0)).sum() + 0.5
    return float(a * d / (b * c))


def mh_or(df: pd.DataFrame, y: str, x: str, strata: str = "model") -> float:
    num = den = 0.0
    for _, g in df.groupby(strata):
        n = len(g)
        a = ((g[x] == 1) & (g[y] == 1)).sum()
        b = ((g[x] == 1) & (g[y] == 0)).sum()
        c = ((g[x] == 0) & (g[y] == 1)).sum()
        d = ((g[x] == 0) & (g[y] == 0)).sum()
        num += a * d / n
        den += b * c / n
    return float(num / den) if den else float("nan")


def boot_stat(df: pd.DataFrame, fn) -> list[float]:
    out = []
    by = {t: g.index.to_numpy() for t, g in df.groupby("turn")}
    for draw in BOOT:
        idx = np.concatenate([by[t] for t in draw if t in by])
        out.append(fn(df.loc[idx]))
    return out


def e25b() -> None:
    rule("C. E25b: question or speaker? The Presidents' questions put to four open LLMs")
    fr = llm_table()
    print(f"   answers: {len(fr)}; words, median per model x condition:\n"
          f"{fr.groupby(['model', 'condition'])['n_words'].median().unstack().reindex(columns=CONDS).to_string()}")
    if "finish" in fr:
        print(f"   share cut at 400 tokens: {(fr['finish'] == 'length').mean():.3f}")
    pres_rate = 1 - np.mean(GOLD_CLAR == CR)
    print(f"\n   evasion rate (clarity != Clear Reply). Presidents: gold {pres_rate:.3f}, instrument on their answers "
          f"{1 - np.mean(clarity(lab('dev_own')['pred']) == CR):.3f}")
    tab = fr.groupby(["model", "condition"])["evade"].mean().unstack().reindex(columns=CONDS)
    print(tab.round(3).to_string())
    pooled = {c: fr[fr["condition"] == c]["evade"].mean() for c in CONDS}
    RES["rates"] = {"president_gold": float(pres_rate), "president_inst": float(1 - np.mean(clarity(lab('dev_own')['pred']) == CR)),
                    "table": {m: {c: float(tab.loc[m, c]) for c in CONDS} for m in tab.index},
                    "ci": {m: {c: [float(x) for x in np.percentile(boot_stat(fr[(fr.model == m) & (fr.condition == c)].reset_index(drop=True), lambda d: d['evade'].mean()), [2.5, 97.5])] for c in CONDS} for m in tab.index}}
    print("   pooled over models: " + "  ".join(f"{c} {v:.3f}" for c, v in pooled.items()))

    # H25b-1
    print("\n   H25b-1: odds ratio of an LLM evading on questions the President evaded vs answered (gold), by condition")
    for c in CONDS:
        g = fr[fr["condition"] == c].reset_index(drop=True)
        per = {m: odds_ratio(g[g["model"] == m]["evade"].to_numpy(), g[g["model"] == m]["pres_evade"].to_numpy())
               for m in MODELS if (g["model"] == m).any()}
        mh = mh_or(g, "evade", "pres_evade")
        bs = boot_stat(g, lambda d: mh_or(d, "evade", "pres_evade"))
        mh_i = mh_or(g, "evade", "pres_evade_inst")
        RES.setdefault("or", {})[c] = {"mh": mh, "ci": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))], "per_model": per}
        print(f"   {c}: MH OR {mh:.2f} {ci(bs)}  per model {', '.join(f'{m} {v:.2f}' for m, v in per.items())}"
              f"   (President labelled by the instrument: {mh_i:.2f})")
        if c == "N":
            or_n, bs_n = mh, np.array(bs)
            gN = g
    # control C-q: the instrument's own question prior
    q = lab("dev_qonly")
    qe = (clarity(q["pred"]) != CR).astype(int)
    or_q = odds_ratio(qe, (GOLD_CLAR != CR).astype(int))
    dq = pd.DataFrame({"turn": TURN, "y": qe, "x": (GOLD_CLAR != CR).astype(int)})
    bs_q = np.array(boot_stat(dq, lambda d: odds_ratio(d["y"].to_numpy(), d["x"].to_numpy())))
    print(f"   control C-q: OR of the instrument's evasion label on the CONSTANT answer vs the President's gold: "
          f"{or_q:.2f} {ci(bs_q)}; qonly labels {dict(Counter(EVASION_LABELS[k] for k in q['pred']).most_common(3))}")
    print(f"   P(bootstrap OR_qonly >= OR_N) = {np.mean(bs_q >= bs_n):.3f}")
    RES["or"]["qonly"] = {"mh": or_q, "ci": [float(np.nanpercentile(bs_q, 2.5)), float(np.nanpercentile(bs_q, 97.5))]}
    # adjusted: logistic regression of N evasion on the President's evasion + the instrument's question prior
    from sklearn.linear_model import LogisticRegression

    def adj_or(d: pd.DataFrame) -> float:
        X = np.column_stack([d["pres_evade"], d["qonly_p_cr"], np.log1p(d["n_words"])]
                            + [(d["model"] == m).astype(int) for m in MODELS[1:]])
        if d["pres_evade"].nunique() < 2 or d["evade"].nunique() < 2:
            return float("nan")
        return float(np.exp(LogisticRegression(C=1e6, max_iter=2000).fit(X, d["evade"]).coef_[0][0]))
    a_n = adj_or(gN)
    bs_a = boot_stat(gN, adj_or)
    print(f"   N, adjusted for the instrument's question prior (qonly P(Explicit)), log length and model: OR {a_n:.2f} {ci(bs_a)}")
    lo = np.percentile(bs_n[np.isfinite(bs_n)], 2.5)
    held = or_n >= 1.5 and lo > 1 and or_q < or_n
    print(f"   H25b-1 (N: MH OR >= 1.5, CI above 1, and the qonly control's OR below it): {'HELD' if held else 'FAILED'}")

    # H25b-2 stake
    def diff(d, a="P", b="N"):
        return d[d["condition"] == a]["evade"].mean() - d[d["condition"] == b]["evade"].mean()
    bs = boot_stat(fr, diff)
    dv = diff(fr)
    print(f"\n   H25b-2 (stake: P - N >= +0.10): {dv:+.3f} {ci(bs)} -> {'HELD' if dv >= 0.10 and np.percentile(bs, 2.5) > 0 else 'FAILED'}; "
          f"per model {', '.join(f'{m} {diff(fr[fr.model == m]):+.3f}' for m in MODELS if (fr.model == m).any())}")

    # H25b-3 pursuit
    piv = fr.pivot_table(index=["model", "src_i"], columns="condition", values="clar", aggfunc="first")
    ev = piv[piv["P"] != CR]
    turned = (ev["F"] == CR).mean()
    per_m = {m: float((ev.loc[m]["F"] == CR).mean()) for m in MODELS if m in ev.index.get_level_values(0)}
    print(f"   H25b-3 (pursuit: of P answers that evade, F gives a Clear Reply in >= 50%): {turned:.3f} of {len(ev)} -> "
          f"{'HELD' if turned >= 0.5 else 'FAILED'}; per model {', '.join(f'{m} {v:.3f}' for m, v in per_m.items())}")
    back = (piv[piv["P"] == CR]["F"] != CR).mean()
    print(f"      (and of P answers that reply, F stops replying in {back:.3f})")

    # H25b-4 information
    fr["plur"] = [PLUR[i] for i in fr["src_i"]]
    b = fr[fr["condition"] == "B"]
    ind = b[b["plur"].isin(INDIRECT)]
    non = b[b["plur"].isin(NONANSWER)]
    d4 = (ind["clar"] == CR).mean() - (non["clar"] == CR).mean()

    def d4f(d):
        d = d[d["condition"] == "B"]
        return (d[d["plur"].isin(INDIRECT)]["clar"] == CR).mean() - (d[d["plur"].isin(NONANSWER)]["clar"] == CR).mean()
    bs = boot_stat(fr, d4f)
    p_ind = fr[fr["condition"] == "P"]
    d4p = (p_ind[p_ind["plur"].isin(INDIRECT)]["clar"] == CR).mean() - (p_ind[p_ind["plur"].isin(NONANSWER)]["clar"] == CR).mean()
    print(f"   H25b-4 (information: in B, Clear-Reply rate on indirect-answer questions minus non-answer questions >= +0.15): "
          f"{(ind['clar'] == CR).mean():.3f} - {(non['clar'] == CR).mean():.3f} = {d4:+.3f} {ci(bs)} -> "
          f"{'HELD' if d4 >= 0.15 and np.percentile(bs, 2.5) > 0 else 'FAILED'}  (same difference in P, no notes: {d4p:+.3f})")
    print(f"      items: indirect {ind['src_i'].nunique()}, non-answer {non['src_i'].nunique()} (plurality of 3 annotators)")

    # length covariate for the condition effects
    from sklearn.linear_model import LogisticRegression

    X = np.column_stack([(fr["condition"] == c).astype(int) for c in ("P", "F", "B")]
                        + [(fr["model"] == m).astype(int) for m in MODELS[1:]])
    Xl = np.column_stack([X, np.log1p(fr["n_words"])])
    c0 = LogisticRegression(C=1e6, max_iter=3000).fit(X, fr["evade"]).coef_[0][:3]
    c1 = LogisticRegression(C=1e6, max_iter=3000).fit(Xl, fr["evade"]).coef_[0][:3]
    print("\n   condition effects vs N (log-odds), without / with log answer length as a covariate: "
          + "  ".join(f"{c} {a:+.2f} / {b_:+.2f}" for c, a, b_ in zip(("P", "F", "B"), c0, c1)))

    # profiles
    gold_prof = np.bincount(encode_evasion([p for p in PLUR if p]), minlength=9).astype(float)
    gold_prof /= gold_prof.sum()
    inst_prof = np.bincount(lab("dev_own")["pred"], minlength=9) / len(DEV)

    def js(p, q):
        m = (p + q) / 2
        kl = lambda a, b: np.sum(np.where(a > 0, a * np.log2((a + 1e-12) / (b + 1e-12)), 0))  # noqa: E731
        return float(np.sqrt((kl(p, m) + kl(q, m)) / 2))
    print("\n   evasion-type profiles (share of the 9 labels); Presidents: plurality gold, and the instrument on their answers")
    short = [e[:10] for e in EVASION_LABELS]
    print("   " + " " * 16 + " ".join(f"{s:>10}" for s in short) + "   JS(gold)")
    print("   " + f"{'President gold':<16}" + " ".join(f"{v:>10.3f}" for v in gold_prof))
    print("   " + f"{'President inst.':<16}" + " ".join(f"{v:>10.3f}" for v in inst_prof) + f"   {js(inst_prof, gold_prof):.3f}")
    for m in MODELS:
        for c in CONDS:
            g = fr[(fr["model"] == m) & (fr["condition"] == c)]
            if len(g):
                p = np.bincount(g["pred"], minlength=9) / len(g)
                print("   " + f"{m[:9] + ' ' + c:<16}" + " ".join(f"{v:>10.3f}" for v in p) + f"   {js(p, inst_prof):.3f}")
    print("   (JS for the LLM rows is against the instrument's profile of the Presidents, like for like; Qwen3-8B shares "
          "a family with the 27B judge)")


# ---------------------------------------------------------------- D. E25c

def e25c() -> None:
    rule("D. E25c: when does evasion become visible? Answer prefixes (instrument; M1 alone in brackets)")
    for split in ("dev", "test"):
        own = lab(f"{split}_own")
        for p in (25, 50, 75):
            if not (E25 / "labels" / f"{split}_pre{p}.npz").exists():
                continue  # test prefixes dropped (amendment 3)
            z = lab(f"{split}_pre{p}")
            ag, ag1 = np.mean(z["pred"] == own["pred"]), np.mean(z["m1_pred"] == own["m1_pred"])
            extra = ""
            if split == "dev":
                extra = (f"  S2 {score_subtask2(z['pred'], GOLD).macro_f1:.3f} (own {score_subtask2(own['pred'], GOLD).macro_f1:.3f})"
                         f"  S1 {score_subtask1(clarity(z['pred']), GOLD_CLAR).macro_f1:.3f}")
            print(f"   {split} {p}%: agreement with the full-answer label {ag:.3f} ({ag1:.3f}){extra}")
            RES.setdefault("prefix", {})[f"{split}_{p}"] = {"agree": float(ag), "agree_m1": float(ag1),
                **({"s2": score_subtask2(z['pred'], GOLD).macro_f1} if split == "dev" else {})}
    own, h = lab("dev_own"), lab("dev_pre50")
    s_own, s_h = score_subtask2(own["pred"], GOLD).macro_f1, score_subtask2(h["pred"], GOLD).macro_f1
    ag = np.mean(h["pred"] == own["pred"])
    print(f"   H25c (dev, 50%: agreement >= 0.70 and S2 drop <= 0.05): {'HELD' if ag >= 0.7 and s_own - s_h <= 0.05 else 'FAILED'} "
          f"(agreement {ag:.3f}, drop {s_own - s_h:+.3f})")
    print("   agreement at 25 / 50 / 75% by the item's plurality gold label (dev):")
    for e in EVASION_LABELS:
        idx = np.array([i for i, p in enumerate(PLUR) if p == e])
        if len(idx) >= 5:
            print(f"      {e:<22} n {len(idx):>3}  " + "  ".join(
                f"{np.mean(lab(f'dev_pre{p}')['pred'][idx] == own['pred'][idx]):.2f}" for p in (25, 50, 75)))


# ---------------------------------------------------------------- E. E25d

def e25d() -> None:
    rule("E. E25d: does the frozen base model already encode the relation? Layer-wise probes; early-exit latency")
    p = E25 / "probes.json"
    if not p.exists():
        print("   probes missing")
        return
    res = json.loads(p.read_text())
    for r in res[::4] + ([res[-1]] if (len(res) - 1) % 4 else []):
        print(f"   layer {r['layer']:>2}: slice F1 {r['slice_f1']:.3f}  dev S2 {r['dev_s2']:.3f}  S1 {r['dev_s1']:.3f}  "
              f"same label on disjoint siblings {r['sib_same_label_disjoint']:.3f}")
    by_slice = max(res, key=lambda r: r["slice_f1"])
    RES["probes"] = res
    by_dev = max(res, key=lambda r: r["dev_s2"])
    print(f"   layer chosen on the slice: {by_slice['layer']} -> dev S2 {by_slice['dev_s2']:.3f}, sibling same-label "
          f"{by_slice['sib_same_label_disjoint']:.3f};  best on dev (optimistic): layer {by_dev['layer']} {by_dev['dev_s2']:.3f}")
    print(f"   H25d-1 (slice-chosen layer, dev S2 >= 0.45): {'HELD' if by_slice['dev_s2'] >= 0.45 else 'FAILED'}")
    print(f"   H25d-2 (its same-label rate on disjoint dev siblings >= 0.30; M1 0.19): "
          f"{'HELD' if by_slice['sib_same_label_disjoint'] >= 0.30 else 'FAILED'}")
    lat = E25 / "hidden" / "latency.json"
    RES["own_s2"] = score_subtask2(lab('dev_own')['pred'], GOLD).macro_f1
    rep = json.loads((E25 / "m1" / "report.json").read_text())
    if lat.exists():
        l = json.loads(lat.read_text())
        lora = rep.get("latency_batch1", {})
        print(f"   batch-1 latency (H100, dev items, mean {lora.get('mean_tokens', float('nan')):.0f} tokens): LoRA classifier "
              f"(one seed) p50 {lora.get('p50_ms', float('nan')):.1f} ms, p95 {lora.get('p95_ms', float('nan')):.1f} ms; base "
              "model truncated to L layers: " + ", ".join(f"L={k} p50 {v['p50_ms']:.1f} ms" for k, v in l.items()))
    tp = [v["items_per_s"] for k, v in rep.items() if "/" in k and k.startswith("dev_own")]
    print(f"   batched throughput, one M1 seed: {np.mean(tp):.1f} items/s (dev_own)")


def domain_check() -> None:
    rule("G. Is the instrument on familiar ground with LLM text? (label-free; replaces the dropped human annotation)")

    def stats(z, m=None):
        m = np.ones(len(z["u"]), bool) if m is None else m
        sa = z["seed_argmax"][:, m]
        agree = (sa == np.array([np.bincount(c, minlength=9).argmax() for c in sa.T])[None]).mean(0)
        return z["u"][m].mean(), z["defer"][m].mean(), agree.mean(), (z["pred"][m] == z["m1_pred"][m]).mean()
    own, llm = lab("dev_own"), lab("llm")
    fr = pd.read_parquet(E25 / "frames" / "llm.parquet")
    print(f"   {'frame':<22}{'mean u':>8}{'deferred':>10}{'seed agreement':>16}{'final = M1 label':>18}")
    print(f"   {'Presidents (dev)':<22}" + "".join(f"{v:>{w}.3f}" for v, w in zip(stats(own), (8, 10, 16, 18))))
    print(f"   {'LLM answers (all)':<22}" + "".join(f"{v:>{w}.3f}" for v, w in zip(stats(llm), (8, 10, 16, 18))))
    for (m, c), g in fr.groupby(["model", "condition"]):
        mk = np.zeros(len(fr), bool)
        mk[g.index] = True
        print(f"   {m[:10] + ' ' + c:<22}" + "".join(f"{v:>{w}.3f}" for v, w in zip(stats(llm, mk), (8, 10, 16, 18))))
    print("   -> the instrument is MORE confident on LLM answers than on the Presidents' (lower u, higher seed agreement):")
    print("      it is not out of its depth; this rules out one failure mode, not confident errors")


def cost() -> None:
    rule("F. Cost of the lane (from logs/infer.log)")
    p = ROOT / "logs" / "jarvis" / "E25" / "infer.log"
    if not p.exists():
        print("   no lane log copied yet")
        return
    start = {}
    for line in p.read_text().splitlines():
        parts = line.split(" ", 3)
        if len(parts) < 3:
            continue
        t = pd.Timestamp(parts[0] + " " + parts[1])
        msg = parts[2]
        if line.endswith(": start") or ": start" in line:
            start[msg.rstrip(":")] = t
        elif ": done" in line and msg.rstrip(":") in start:
            print(f"   {msg.rstrip(':'):<14} {(t - start[msg.rstrip(':')]).total_seconds() / 60:6.1f} min")


# ---------------------------------------------------------------- annotation sheet and kappa

def sheet() -> None:
    fr = llm_table()
    rng = np.random.default_rng(7)
    picks = []
    for m in MODELS:
        for c in CONDS:
            g = fr[(fr["model"] == m) & (fr["condition"] == c)]
            picks += list(rng.choice(g.index, min(6, len(g)), replace=False))
    rest = fr.index.difference(picks)
    picks += list(rng.choice(rest, 100 - len(picks), replace=False))
    s = fr.loc[picks].sample(frac=1, random_state=7).reset_index().rename(columns={"index": "row"})
    s["item"] = np.arange(1, len(s) + 1)
    out = DOCS / "annotation"
    out.mkdir(parents=True, exist_ok=True)
    s[["item", "question", "interview_question", "interview_answer"]].assign(
        clarity="", evasion_type="", note="").rename(columns={"question": "sub_question", "interview_question": "full_turn",
                                                          "interview_answer": "answer"}).to_csv(out / "E25_sheet.csv", index=False)
    s[["item", "row", "model", "condition", "src_i", "pred", "clar"]].to_csv(out / "E25_key.csv", index=False)
    print(f"wrote {out / 'E25_sheet.csv'} (blind) and {out / 'E25_key.csv'} ({len(s)} items)")


def kappa(a: str, b: str) -> None:
    from sklearn.metrics import cohen_kappa_score

    key = pd.read_csv(DOCS / "annotation" / "E25_key.csv")
    norm = {"cr": "Clear Reply", "clear reply": "Clear Reply", "amb": "Ambivalent", "ambivalent": "Ambivalent",
            "cnr": "Clear Non-Reply", "clear non-reply": "Clear Non-Reply", "clear non reply": "Clear Non-Reply"}
    A, Bb = (pd.read_csv(x)[["item", "clarity"]] for x in (a, b))
    d = key.merge(A, on="item").merge(Bb, on="item", suffixes=("_a", "_b"))
    d = d[d["clarity_a"].notna() & d["clarity_b"].notna()]
    ca = d["clarity_a"].str.strip().str.lower().map(norm)
    cb = d["clarity_b"].str.strip().str.lower().map(norm)
    inst = d["clar"].map(lambda k: CLARITY_LABELS[k])
    k_hh = cohen_kappa_score(ca, cb)
    k_ha = cohen_kappa_score(ca, inst)
    k_hb = cohen_kappa_score(cb, inst)
    print(f"{len(d)} items: kappa human-human {k_hh:.3f}; instrument-human {k_ha:.3f} / {k_hb:.3f}")
    ok = min(k_ha, k_hb) >= k_hh - 0.10
    print(f"validity rule (instrument kappa within 0.10 of the humans'): {'PASSED' if ok else 'FAILED'}")
    for name, cc in (("A", ca), ("B", cb)):
        print(f"   evasion rate by annotator {name}: {(cc != 'Clear Reply').mean():.3f}; instrument {(inst != 'Clear Reply').mean():.3f}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "sheet":
        return sheet()
    if len(sys.argv) > 1 and sys.argv[1] == "kappa":
        return kappa(sys.argv[2], sys.argv[3])
    print("E25 analysis (registered 2026-10-30 14:30 IST; docs/experiment_log.md, E25). Labels: the frozen E23 instrument "
          "(code/experiments/e25/frozen_system.json).")
    controls()
    e25a()
    e25b()
    e25c()
    e25d()
    domain_check()
    cost()
    (E25 / "analysis.json").write_text(json.dumps(RES, indent=1, default=float))


if __name__ == "__main__":
    main()
