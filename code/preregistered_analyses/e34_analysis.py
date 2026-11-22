#!/usr/bin/env python3
"""E34 (registered 2026-11-21, docs/experiment_log.md): does the role change WHERE LLMs evade? The confirmatory test.

    python -m preregistered_analyses.e34_analysis dev      # discovery sample: dev, E25/E32 answers (frozen instrument,
                                                          #   and the E33 cross-fit instrument once runs/E34/cf/dev_llm.npz exists)
    python -m preregistered_analyses.e34_analysis train    # confirmation sample: the 248 train interviews not in dev

Estimand: the ratio of odds ratios ROR_{A/B} = OR_A / OR_B, where OR_c is the odds ratio of an LLM answer being
evasive (clarity level != Clear Reply, by the instrument) on sub-questions the President evaded against those he
answered (annotators), in condition c. ROR > 1 means the role A puts the LLM's evasions where the President's were more
than role B does. Anything fixed per sub-question (its difficulty, the instrument's prior for it, annotator noise in the
President's label) enters OR_A and OR_B alike.

Primary model (H34-1): logistic regression on the rows of conditions {A, B}, pooled over the models:
    evade ~ 0 + (model x condition) + (model x pres_evade) + gamma * pres_evade * [condition = A],   ROR = exp(gamma)
95% interval: 2,000-draw bootstrap over interviews (url): every row of a drawn interview, for every model and
condition, enters together. One-sided p = share of draws with gamma <= 0.

Writes docs/raw/E34_<sample>.txt and runs/E34/analysis_<sample>.json.
"""

from __future__ import annotations

import json
import re
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

from qevasion.labels import CLARITY_LABELS, EVASION_LABELS, encode_clarity, leaf_to_official_clarity
from qevasion.loader import DATA_CACHE, load_qevasion
from qevasion.paths import ROOT, RUNS

E34 = RUNS / "E34"
CR = CLARITY_LABELS.index("Clear Reply")
EXPLICIT = EVASION_LABELS.index("Explicit")
NONANSWER = [EVASION_LABELS.index(x) for x in ("Dodging", "Deflection", "Declining to answer", "Claims ignorance")]
TRIO = ("mistral_7b", "olmo2_7b", "qwen3_8b")
B = 2000
B_ROBUST = 1000
YOU = re.compile(r"\b(?:you|your|yours|yourself)\b", re.I)
META = re.compile(r"\bas an ai\b|\bai (?:language )?model\b|\blanguage model\b|\bI(?:'m| am) (?:just |only )?an ai\b", re.I)


# ----------------------------------------------------------------------------------------------------------- data
def cf_tau() -> float:
    return float(json.loads((RUNS / "E33" / "cf_tau.json").read_text())["tau"])


def cf_labels(probs: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """E33 CF instrument: logit adjustment with the frozen tau; argmax label and the soft P(evasive) after adjustment."""
    from cascade.fuse import PRIOR

    s = np.log(probs + 1e-12) - tau * np.log(PRIOR + 1e-12)
    q = np.exp(s - s.max(1, keepdims=True))
    q /= q.sum(1, keepdims=True)
    return s.argmax(1), 1.0 - q[:, EXPLICIT]


def zs_labels(tag: str, frame: str, n: int) -> np.ndarray | None:
    """Qwen3.8-27B zero-shot over the nine labels (runs/M3/<tag>/responses.jsonl, cand 0..8): argmax label per row."""
    p = RUNS / "M3" / tag / "responses.jsonl"
    if not p.exists():
        return None
    out = np.full(n, -1)
    for line in p.read_text().splitlines():
        r = json.loads(line)
        if r["frame"] == frame:
            assert r["cand"] == list(range(len(EVASION_LABELS)))
            out[r["i"]] = int(np.argmax(r["q"]))
    return out if (out >= 0).all() else None


def finish(df: pd.DataFrame, src: pd.DataFrame, gold_clar: np.ndarray) -> pd.DataFrame:
    df["pres_evade"] = (gold_clar[df["src_i"].to_numpy()] != CR).astype(int)
    df["url"] = src["url"].astype(str).to_numpy()[df["src_i"].to_numpy()]
    df["you"] = src["question"].astype(str).str.contains(YOU).to_numpy()[df["src_i"].to_numpy()].astype(int)
    df["meta"] = df["raw_answer"].astype(str).str.contains(META).astype(int) if "raw_answer" in df else 0
    df["len_t"] = df.groupby("model")["n_words"].transform(lambda x: pd.qcut(x.rank(method="first"), 3, labels=False))
    return df


def dev_table() -> pd.DataFrame:
    """Discovery: the E25/E32 dev answers (protocol A) of the three discovery models, N and P."""
    dev = load_qevasion(DATA_CACHE).dev.reset_index(drop=True)
    t = pd.read_parquet(RUNS / "E32" / "llm_table.parquet")
    t = t[t["condition"].isin(["N", "P"]) & t["model"].isin(TRIO)].reset_index(drop=True)
    df = t[["model", "condition", "src_i", "n_words", "raw_answer", "pred"]].copy()
    df["evade_frozen"] = (leaf_to_official_clarity(df["pred"].to_numpy()) != CR).astype(int)
    df["nonans_frozen"] = df["pred"].isin(NONANSWER).astype(int)
    cfp = E34 / "cf" / "dev_llm.npz"
    if cfp.exists() and (RUNS / "E33" / "cf_tau.json").exists():
        fr = pd.read_parquet(E34 / "frames" / "dev_llm.parquet")
        assert (fr["model"].to_numpy() == t["model"].to_numpy()).all() and (fr["src_i"].to_numpy() == t["src_i"].to_numpy()).all()
        lab, soft = cf_labels(np.load(cfp)["probs"], cf_tau())
        df["evade_cf"] = (leaf_to_official_clarity(lab) != CR).astype(int)
        df["soft_cf"] = soft
        df["nonans_cf"] = np.isin(lab, NONANSWER).astype(int)
    return finish(df, dev, encode_clarity(dev["clarity_label"].tolist()))


def train_table() -> pd.DataFrame:
    """Confirmation: E34 answers to train questions, labelled by CF (E33) and by the 27B zero-shot."""
    sp = load_qevasion(DATA_CACHE)
    train = sp.train.reset_index(drop=True)
    fr = pd.read_parquet(E34 / "frames" / "llm.parquet")
    assert (fr["split"] == "train").all()
    df = fr[["model", "condition", "src_i", "n_words", "raw_answer"]].copy()
    lab, soft = cf_labels(np.load(E34 / "cf" / "llm.npz")["probs"], cf_tau())
    df["pred_cf"] = lab
    df["evade_cf"] = (leaf_to_official_clarity(lab) != CR).astype(int)
    df["soft_cf"] = soft
    df["nonans_cf"] = np.isin(lab, NONANSWER).astype(int)
    zs = zs_labels("e34zs_taxonomy", "llm", len(df))
    if zs is not None:
        df["evade_zs"] = (leaf_to_official_clarity(zs) != CR).astype(int)
    df["annotator"] = train["annotator_id"].astype(str).to_numpy()[df["src_i"].to_numpy()]
    dev_urls = set(sp.dev["url"].astype(str))
    df = finish(df, train, encode_clarity(train["clarity_label"].tolist()))
    df["confirm"] = (~df["url"].isin(dev_urls)).astype(int)
    return df


# ------------------------------------------------------------------------------------------------------ estimators
def design(d: pd.DataFrame, a: str, extra: str = "") -> tuple[np.ndarray, list[str]]:
    """Columns: model x condition intercepts, model x pres_evade slopes, pres_evade x [cond = a] (last), + extras."""
    cols, names = [], []
    for m in sorted(d["model"].unique()):
        for c in sorted(d["condition"].unique()):
            cols.append(((d["model"] == m) & (d["condition"] == c)).to_numpy(float)); names.append(f"{m}:{c}")
        cols.append(((d["model"] == m).to_numpy(float) * d["pres_evade"].to_numpy())); names.append(f"{m}:pres")
    if extra == "len":
        for m in sorted(d["model"].unique()):
            for c in sorted(d["condition"].unique()):
                for t in (1, 2):
                    cols.append(((d["model"] == m) & (d["condition"] == c) & (d["len_t"] == t)).to_numpy(float))
                    names.append(f"{m}:{c}:len{t}")
    if extra == "annotator":
        anns = sorted(d["annotator"].unique())
        for an in anns[1:]:
            cols.append(((d["annotator"] == an).to_numpy(float) * d["pres_evade"].to_numpy())); names.append(f"ann{an}:pres")
            cols.append((d["annotator"] == an).to_numpy(float)); names.append(f"ann{an}")
    cols.append(d["pres_evade"].to_numpy() * (d["condition"] == a).to_numpy(float)); names.append("gamma")
    return np.column_stack(cols), names


def gamma_hat(y: np.ndarray, X: np.ndarray) -> float:
    try:
        r = sm.GLM(y, X, family=sm.families.Binomial()).fit()
        return float(r.params[-1])
    except Exception:  # noqa: BLE001  (separation in a degenerate draw)
        return float("nan")


def ror(d: pd.DataFrame, a: str, b: str, y: str, n_boot: int = B, extra: str = "", seed: int = 0) -> dict:
    d = d[d["condition"].isin([a, b])].reset_index(drop=True)
    X, _ = design(d, a, extra)
    yy = d[y].to_numpy(float)
    g = gamma_hat(yy, X)
    by = {u: ix for u, ix in d.groupby("url").indices.items()}
    urls = np.array(sorted(by))
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        idx = np.concatenate([by[u] for u in rng.choice(urls, len(urls))])
        bs.append(gamma_hat(yy[idx], X[idx]))
    bs = np.array(bs)
    bs = bs[np.isfinite(bs)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"ror": float(np.exp(g)), "ci": [float(np.exp(lo)), float(np.exp(hi))], "p_one_sided": float(np.mean(bs <= 0)),
            "p_two_sided": float(min(1.0, 2 * min(np.mean(bs <= 0), np.mean(bs >= 0)))), "n_rows": int(len(d)),
            "n_urls": int(len(urls)), "n_boot_ok": int(len(bs))}


def per_model_or(d: pd.DataFrame, y: str) -> dict:
    out = {}
    for (m, c), g in d.groupby(["model", "condition"]):
        a = ((g["pres_evade"] == 1) & (g[y] == 1)).sum() + 0.5
        b_ = ((g["pres_evade"] == 1) & (g[y] == 0)).sum() + 0.5
        c_ = ((g["pres_evade"] == 0) & (g[y] == 1)).sum() + 0.5
        d_ = ((g["pres_evade"] == 0) & (g[y] == 0)).sum() + 0.5
        out.setdefault(m, {})[c] = float(a * d_ / (b_ * c_))
    return out


def gee_ror(d: pd.DataFrame, a: str, b: str, y: str) -> dict:
    d = d[d["condition"].isin([a, b])].reset_index(drop=True)
    X, _ = design(d, a)
    r = sm.GEE(d[y].to_numpy(float), X, groups=d["url"].factorize()[0], family=sm.families.Binomial(),
               cov_struct=sm.cov_struct.Exchangeable()).fit()
    lo, hi = r.conf_int()[-1]
    return {"ror": float(np.exp(r.params[-1])), "ci": [float(np.exp(lo)), float(np.exp(hi))]}


def clogit_ror(d: pd.DataFrame, a: str, b: str, y: str) -> dict:
    """Conditional logit, strata = model x sub-question: only within-question contrasts between the roles count."""
    from statsmodels.discrete.conditional_models import ConditionalLogit

    d = d[d["condition"].isin([a, b])].reset_index(drop=True)
    isa = (d["condition"] == a).to_numpy(float)
    cols = [isa * (d["model"] == m).to_numpy(float) for m in sorted(d["model"].unique())]
    cols.append(isa * d["pres_evade"].to_numpy())
    r = ConditionalLogit(d[y].to_numpy(float), np.column_stack(cols),
                         groups=(d["model"] + "|" + d["src_i"].astype(str)).factorize()[0]).fit(disp=0)
    lo, hi = r.conf_int()[-1]
    return {"ror": float(np.exp(r.params[-1])), "ci": [float(np.exp(lo)), float(np.exp(hi))]}


def tipping(d: pd.DataFrame, a: str, b: str, y: str) -> dict:
    """How much extra label noise in condition b (symmetric flips at rate e, the same on both kinds of question) would
    make the corrected ROR_{a/b} reach 1. Pooled over models as the geometric mean of the per-model ratios."""
    rates = d.groupby(["model", "condition", "pres_evade"])[y].mean()
    ms = sorted(d["model"].unique())

    def lor(p1, p0):
        p1, p0 = np.clip([p1, p0], 1e-3, 1 - 1e-3)
        return np.log(p1 / (1 - p1)) - np.log(p0 / (1 - p0))

    def pooled(e):
        vals = []
        for m in ms:
            pa1, pa0 = rates[(m, a, 1)], rates[(m, a, 0)]
            pb1, pb0 = ((rates[(m, b, k)] - e) / (1 - 2 * e) for k in (1, 0))
            vals.append(lor(pa1, pa0) - lor(pb1, pb0))
        return float(np.exp(np.mean(vals)))

    grid = np.round(np.arange(0, 0.451, 0.01), 2)
    curve = {float(e): pooled(e) for e in grid}
    hit = [e for e in grid if curve[float(e)] <= 1]
    return {"curve": {k: round(v, 3) for k, v in curve.items() if abs(k * 20 - round(k * 20)) < 1e-9},
            "e_star": float(hit[0]) if hit else None}


def soft_ror(d: pd.DataFrame, a: str, b: str, y: str, n_boot: int = B_ROBUST) -> dict:
    """Fractional logit on the instrument's P(evasive): no argmax threshold, so no attenuation from it."""
    return ror(d, a, b, y, n_boot=n_boot, seed=1)


# ---------------------------------------------------------------------------------------------------------- report
def fmt(r: dict) -> str:
    return f"{r['ror']:.2f} [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]"


def rates_block(d: pd.DataFrame, y: str) -> list[str]:
    t = d.groupby(["model", "condition", "pres_evade"])[y].mean().unstack().rename(columns={0: "answered", 1: "evaded"})
    return ["   evasion rate by model x condition, on sub-questions the President answered / evaded:"] + \
        ["      " + s for s in t.round(3).to_string().splitlines()]


def analyse_dev() -> tuple[list[str], dict]:
    d = dev_table()
    L = [f"E34 discovery sample (dev): {len(d)} answers, models {sorted(d.model.unique())}, conditions N/P, "
         f"{d.url.nunique()} interviews; President evaded {d.drop_duplicates('src_i').pres_evade.mean():.3f}"]
    res = {"sample": "dev"}
    for inst in ("frozen", "cf"):
        y = f"evade_{inst}"
        if y not in d:
            L.append(f"   [{inst}] not available yet")
            continue
        r = ror(d, "P", "N", y)
        res[inst] = {"ror_P_N": r, "per_model_or": per_model_or(d, y),
                     "per_model_ror": {m: ror(d[d.model == m], "P", "N", y, n_boot=B_ROBUST) for m in TRIO},
                     "you": ror(d[d.you == 1], "P", "N", y, n_boot=B_ROBUST),
                     "not_you": ror(d[d.you == 0], "P", "N", y, n_boot=B_ROBUST),
                     "gee": gee_ror(d, "P", "N", y), "clogit": clogit_ror(d, "P", "N", y),
                     "tipping": tipping(d, "P", "N", y)}
        if f"soft_{inst}" in d:
            res[inst]["soft"] = soft_ror(d, "P", "N", f"soft_{inst}")
        L += ["", f"[{inst} instrument] ROR P/N (url bootstrap, {B} draws): {fmt(r)}, one-sided p {r['p_one_sided']:.4f}",
              "   per model: " + "  ".join(f"{m} {fmt(res[inst]['per_model_ror'][m])}" for m in TRIO),
              f"   'you' sub-questions {fmt(res[inst]['you'])};  others {fmt(res[inst]['not_you'])}",
              f"   GEE {fmt(res[inst]['gee'])};  conditional logit {fmt(res[inst]['clogit'])}"
              + (f";  soft labels {fmt(res[inst]['soft'])}" if "soft" in res[inst] else ""),
              f"   tipping point: extra symmetric label noise in N that brings ROR to 1: e* = {res[inst]['tipping']['e_star']}"]
        L += rates_block(d, y)
    return L, res


def analyse_train() -> tuple[list[str], dict]:
    d = train_table()
    c = d[d["confirm"] == 1]
    insts = ["cf"] + (["zs"] if "evade_zs" in d else [])
    prim = json.loads((E34 / "primary_instrument.json").read_text())["primary"] if (E34 / "primary_instrument.json").exists() else "cf"
    y = f"evade_{prim}"
    L = [f"E34 confirmation sample: {c.url.nunique()} interviews not in dev, {c.src_i.nunique()} sub-questions; "
         f"President evaded {c.drop_duplicates('src_i').pres_evade.mean():.3f}; answers {len(c)}; primary instrument {prim}"]
    res = {"sample": "train", "primary": prim}
    trio = c[c.model.isin(TRIO)]
    h1 = ror(trio, "P", "N", y)
    pm = {m: ror(trio[trio.model == m], "P", "N", y, n_boot=B_ROBUST) for m in TRIO}
    n_above = sum(pm[m]["ror"] > 1 for m in TRIO)
    held1 = h1["ci"][0] > 1 and n_above >= 2
    res["H34-1"] = {**h1, "per_model": pm, "n_models_above_1": n_above, "point_ge_1.5": h1["ror"] >= 1.5, "held": bool(held1)}
    L += ["", "H34-1 (primary): ROR P/N, pooled over the trio, lower bound > 1 and ROR > 1 in >= 2 of 3 models",
          f"   {fmt(h1)}, one-sided p {h1['p_one_sided']:.4f}; per model " + "  ".join(f"{m} {fmt(pm[m])}" for m in TRIO),
          f"   point estimate >= 1.5: {h1['ror'] >= 1.5} (reported, not a gate) -> {'HELD' if held1 else 'FAILED'}"]
    sec = {}
    if {"S", "N"} <= set(trio.condition):
        sec["H34-2 ROR S/N > 1"] = (ror(trio, "S", "N", y), "one")
        sec["H34-3 ROR P/S (two-sided)"] = (ror(trio, "P", "S", y), "two")
    if {"G", "N"} <= set(trio.condition):
        sec["H34-4 ROR G/N > 1"] = (ror(trio, "G", "N", y), "one")
    for m in sorted(set(c.model) - set(TRIO)):
        sec[f"H34-5 ROR P/N > 1, {m}"] = (ror(c[c.model == m], "P", "N", y), "one")
    sec["H34-6 ROR P/N > 1, sub-questions without 'you'"] = (ror(trio[trio.you == 0], "P", "N", y), "one")
    ps = {k: (v[0]["p_one_sided"] if v[1] == "one" else v[0]["p_two_sided"]) for k, v in sec.items()}
    order = sorted(ps, key=ps.get)
    holm, run = {}, 0.0
    for j, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - j) * ps[k]))
        holm[k] = run
    L += ["", "Secondary (Holm over the family below)"]
    res["secondary"] = {}
    for k, (r, side) in sec.items():
        ok = holm[k] < 0.05 and (side == "two" or r["ror"] > 1)
        res["secondary"][k] = {**r, "p_holm": holm[k], "held": bool(ok)}
        L.append(f"   {k}: {fmt(r)}, p {ps[k]:.4f}, Holm {holm[k]:.4f} -> {'HELD' if ok else 'FAILED'}")
    rob = {}
    for inst in insts:
        yy = f"evade_{inst}"
        rob[f"{inst}: trio, logistic"] = ror(trio, "P", "N", yy, n_boot=B_ROBUST)
        rob[f"{inst}: GEE"] = gee_ror(trio, "P", "N", yy)
        rob[f"{inst}: conditional logit"] = clogit_ror(trio, "P", "N", yy)
    rob["cf: answer-length terciles"] = ror(trio, "P", "N", "evade_cf", n_boot=B_ROBUST, extra="len")
    rob["cf: annotator-specific slopes"] = ror(trio, "P", "N", "evade_cf", n_boot=B_ROBUST, extra="annotator")
    rob["cf: without Qwen3-8B"] = ror(trio[trio.model != "qwen3_8b"], "P", "N", "evade_cf", n_boot=B_ROBUST)
    rob["cf: non-answer outcome"] = ror(trio, "P", "N", "nonans_cf", n_boot=B_ROBUST)
    rob["cf: without meta answers"] = ror(trio[trio.meta == 0], "P", "N", "evade_cf", n_boot=B_ROBUST)
    rob["cf: soft labels"] = soft_ror(trio, "P", "N", "soft_cf")
    full = d[d.model.isin(TRIO)]
    rob["cf: full train (incl. dev interviews)"] = ror(full, "P", "N", "evade_cf", n_boot=B_ROBUST)
    res["robustness"] = rob
    res["tipping"] = tipping(trio, "P", "N", y)
    L += ["", "Robustness (reported, no gate)"] + [f"   {k}: {fmt(v)}" for k, v in rob.items()]
    L.append(f"   tipping point (extra symmetric noise in N labels that brings ROR to 1): e* = {res['tipping']['e_star']}")
    if "evade_zs" in d:
        ag = trio.assign(agree=(trio.evade_cf == trio.evade_zs).astype(int)).groupby("condition")["agree"].mean()
        res["H33-2"] = {k: float(v) for k, v in ag.items()}
        diff = abs(ag.get("N", np.nan) - ag.get("P", np.nan))
        res["H33-2"]["held"] = bool(diff <= 0.10)
        L.append(f"\nH33-2 CF vs ZS evade agreement by condition: " + "  ".join(f"{k} {v:.3f}" for k, v in ag.items())
                 + f"; |N - P| = {diff:.3f} -> {'HELD' if diff <= 0.10 else 'FAILED'}")
    rate = trio.groupby(["model", "condition"])[y].mean().unstack()
    L += ["", "Evasion rate by model x condition (P - N is descriptive):"] + ["   " + s for s in rate.round(3).to_string().splitlines()]
    L += rates_block(trio, y)
    res["rates"] = {m: {k: float(v) for k, v in row.items()} for m, row in rate.iterrows()}
    return L, res


def main() -> None:
    sample = sys.argv[1] if len(sys.argv) > 1 else "dev"
    L, res = analyse_dev() if sample == "dev" else analyse_train()
    print("\n".join(L))
    raw = ROOT / "docs" / "raw" / f"E34_{'discovery_dev' if sample == 'dev' else 'confirmation_train'}.txt"
    raw.write_text("\n".join(L) + "\n", encoding="utf8")
    E34.mkdir(parents=True, exist_ok=True)
    (E34 / f"analysis_{sample}.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"\nwrote {raw} and {E34 / f'analysis_{sample}.json'}")


if __name__ == "__main__":
    main()
