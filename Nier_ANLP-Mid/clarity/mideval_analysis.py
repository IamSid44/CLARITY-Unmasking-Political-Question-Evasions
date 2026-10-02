#!/usr/bin/env python3
"""Mid-evaluation baselines and analyses, all from data and saved dev probabilities.

No GPU and no training of neural models: every number below is computed from the
QEvasion splits and the `dev_probs.npy` files the trained runs already saved. Each
section is timed, so the cost of producing the package is measured, not estimated.

    python clarity/mideval_analysis.py            # writes reports/raw/mideval/ and data/splits/

Sections
  A  data facts and annotator statistics (C3 feasibility)
  B  trivial baselines: majority, uniform random, TF-IDF + logistic regression
  C  frozen splits: train-internal slice, dev-A / dev-B
  D  main table: single models, untuned ensembles, decision rules under 2-fold dev-A/B
     and under the track's 5-fold nested CV
  E  bootstrap 95% intervals
  F  error analysis of the final system's models
  G  human ceiling and seed diversity
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from contextlib import contextmanager
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from qevasion.labels import (
    CLARITY_LABELS,
    EVASION_LABELS,
    N_EVASION,
    encode_clarity,
    encode_evasion,
    leaf_to_official_clarity,
    multi_reference_mask,
    normalize_evasion,
)
from qevasion.loader import DATA_CACHE, consensus_level, derive_dev_consensus_leaf, dev_reference_mask, load_qevasion
from qevasion.scoring import OOV_INDEX, score_subtask1, score_subtask2
from decide import fit_tau, logit_adjust, make_rules, nested_cv

ROOT = Path(__file__).resolve().parent
ANALYSIS, SPLITS = ROOT / "reports" / "raw" / "mideval", ROOT / "data" / "splits"
ANN = ("annotator1", "annotator2", "annotator3")
SEED = 2026
N_BOOT = 1000
TRIANGLE = ("Dodging", "General", "Deflection")

SYSTEMS = {
    "baseline: sub-q + answer, 8 ep": ["L0_large_base", "E11_base_8ep"],
    "16 epochs": ["E10_base_16ep", "E11_base_16ep"],
    "full question, 16 ep (FINAL)": ["E10_fullq_16ep", "E11_fullq_16ep"],
    "E12a all of train (5 seeds)": ["E12a_alldata"],
}
FINAL = "full question, 16 ep (FINAL)"
BASELINE = "baseline: sub-q + answer, 8 ep"

TIMES: dict[str, float] = {}
LOG: list[str] = []


def say(s: str = "") -> None:
    print(s)
    LOG.append(s)


@contextmanager
def section(name: str):
    say(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    t0, c0 = time.perf_counter(), time.process_time()
    yield
    TIMES[name] = (time.perf_counter() - t0, time.process_time() - c0)
    say(f"[{name.split()[0]}: {TIMES[name][0]:.1f} s wall, {TIMES[name][1]:.1f} s CPU]")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def load_seed_probs(runs: list[str]) -> np.ndarray:
    files = [f for r in runs for f in sorted((ROOT / "runs" / r).glob("seed*/dev_probs.npy"))]
    if not files:
        raise SystemExit(f"no dev_probs.npy for {runs}")
    return np.stack([np.load(f) for f in files])


def s2(pred, mask) -> float:
    return score_subtask2(np.asarray(pred), mask).macro_f1


def s1(pred, ctrue) -> float:
    return score_subtask1(leaf_to_official_clarity(np.asarray(pred)), ctrue).macro_f1


def inset(pred, mask) -> float:
    return float(mask[np.arange(len(pred)), pred].mean())


def two_annotator_masks(dev) -> list[np.ndarray]:
    return [multi_reference_mask([[r[c] for c in pair] for _, r in dev.iterrows()]) for pair in combinations(ANN, 2)]


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Single temperature minimising NLL against the majority label (brief §3.3d.1)."""
    grid = np.exp(np.linspace(np.log(0.2), np.log(5.0), 81))
    best_t, best = 1.0, np.inf
    for t in grid:
        z = logits / t
        z = z - z.max(1, keepdims=True)
        nll = -(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1))).mean()
        if nll < best:
            best_t, best = float(t), nll
    return best_t


def fit_bias(z: np.ndarray, mask: np.ndarray, passes: int = 3) -> np.ndarray:
    """Per-class additive bias, coordinate ascent, b_c in [-2, 2] step 0.1 (brief §3.3d.2)."""
    b = np.zeros(N_EVASION)
    grid = np.round(np.arange(-2.0, 2.0001, 0.1), 1)
    best = s2((z + b).argmax(1), mask)
    for _ in range(passes):
        for c in range(N_EVASION):
            keep = b[c]
            for g in grid:
                b[c] = g
                v = s2((z + b).argmax(1), mask)
                if v > best + 1e-9:
                    best, keep = v, g
            b[c] = keep
    return b


def bootstrap(fn, n: int, rng) -> tuple[float, float]:
    vals = [fn(rng.integers(0, n, n)) for _ in range(N_BOOT)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# --------------------------------------------------------------------------


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    SPLITS.mkdir(parents=True, exist_ok=True)
    t_all, c_all = time.perf_counter(), time.process_time()
    rng = np.random.default_rng(SEED)

    data = load_qevasion(DATA_CACHE)
    train, dev = data.train, data.dev
    mask = dev_reference_mask(dev)
    masks2 = two_annotator_masks(dev)
    ctrue = encode_clarity(dev["clarity_label"].tolist())
    ytr = encode_evasion(train["evasion_label"].tolist())
    prior = np.bincount(ytr, minlength=N_EVASION) / len(ytr)
    maj, has_maj = derive_dev_consensus_leaf(dev)
    lab = np.array([[EVASION_LABELS.index(normalize_evasion(str(r[c]))) for c in ANN] for _, r in dev.iterrows()])
    cons = consensus_level(dev)
    n = len(dev)

    # ------------------------------------------------------------------ A
    with section("A  data facts and annotator statistics"):
        test = pd.read_csv(ROOT / "data" / "clarity_task_evaluation_dataset.csv")
        say(f"splits: train {len(train)}, dev {n}, test {len(test)} (test has no labels locally)")
        say(f"dev majority label exists for {has_maj.sum()} items; {(~has_maj).sum()} three-way splits")
        say(f"dev consensus: unanimous {(cons == 1).sum()}, 2-1 {(cons == 2).sum()}, all different {(cons == 3).sum()}")
        say("\nlabel distribution (train single label; dev reference-set membership):")
        dist = pd.DataFrame(
            {
                "train %": 100 * prior,
                "dev items with label in G": mask.sum(0),
            },
            index=EVASION_LABELS,
        ).round(1)
        say(dist.to_string())
        ids = train["annotator_id"].astype(str)
        say(f"\ntrain annotator_id: {ids.value_counts().to_dict()} (one annotator per row; no overlap on train)")
        per_ann = pd.crosstab(ids, train["evasion_label"].map(normalize_evasion), normalize="index").reindex(columns=EVASION_LABELS)
        say("train label distribution per annotator (row %):")
        say((100 * per_ann).round(1).to_string())
        say("\ndev annotator column vs train annotator: total-variation distance of label marginals (lower = closer)")
        rows = {}
        for c in ANN:
            dv = pd.Series([normalize_evasion(str(v)) for v in dev[c]]).value_counts(normalize=True).reindex(EVASION_LABELS, fill_value=0)
            rows[c] = {a: round(0.5 * float(np.abs(dv.values - per_ann.loc[a].fillna(0).values).sum()), 3) for a in per_ann.index}
        say(pd.DataFrame(rows).T.to_string())
        say("C3 verdict: per-item annotator IDs exist on train; each train item has one annotator, so per-annotator")
        say("models are identifiable only through annotator marginals and dev's 3-way overlap -> analysis scope.")

    # ------------------------------------------------------------------ B
    with section("B  trivial baselines"):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score
        from sklearn.model_selection import StratifiedKFold

        def text(df):
            return (df["question"].astype(str) + " [SEP] " + df["interview_answer"].astype(str)).tolist()

        rows = []

        def report(name, pred):
            rows.append(
                {
                    "baseline": name,
                    "dev S2 (3 ann)": round(s2(pred, mask), 3),
                    "dev S2 (2 ann, mean)": round(float(np.mean([s2(pred, m) for m in masks2])), 3),
                    "dev S1": round(s1(pred, ctrue), 3),
                    "in-set": round(inset(pred, mask), 3),
                    "classes named": len(set(pred.tolist())),
                }
            )

        report("always majority (Explicit)", np.full(n, int(np.argmax(prior))))
        rnd = [rng.integers(0, N_EVASION, n) for _ in range(20)]
        rows.append(
            {
                "baseline": "uniform random (mean of 20)",
                "dev S2 (3 ann)": round(float(np.mean([s2(p, mask) for p in rnd])), 3),
                "dev S2 (2 ann, mean)": round(float(np.mean([s2(p, m) for p in rnd for m in masks2])), 3),
                "dev S1": round(float(np.mean([s1(p, ctrue) for p in rnd])), 3),
                "in-set": round(float(np.mean([inset(p, mask) for p in rnd])), 3),
                "classes named": 9,
            }
        )
        from joblib import Parallel, delayed
        from threadpoolctl import threadpool_limits

        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50_000)
        Xtr, Xdv = vec.fit_transform(text(train)), vec.transform(text(dev))
        folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xtr, ytr))
        grid = [(cw, C) for cw in (None, "balanced") for C in (0.3, 1.0, 3.0, 10.0)]

        def cv_fit(cw, C, tr, va):
            with threadpool_limits(1):
                m = LogisticRegression(C=C, class_weight=cw, max_iter=1000).fit(Xtr[tr], ytr[tr])
            return f1_score(ytr[va], m.predict(Xtr[va]), average="macro")

        say("TF-IDF (1-2 grams, 50k features) + logistic regression; C and class weighting chosen by 5-fold CV "
            "on TRAIN (macro-F1), 40 fits in parallel at one thread each:")
        res = Parallel(n_jobs=40)(delayed(cv_fit)(cw, C, tr, va) for cw, C in grid for tr, va in folds)
        res = np.array(res).reshape(len(grid), len(folds)).mean(1)
        for (cw, C), v in zip(grid, res):
            say(f"  class_weight={cw!s:9} C={C:<5} train-CV macro-F1 {v:.3f}")
        cw, C = grid[int(np.argmax(res))]
        with threadpool_limits(8):
            lr = LogisticRegression(C=C, class_weight=cw, max_iter=1000).fit(Xtr, ytr)
        p_lr = lr.predict_proba(Xdv)
        report(f"TF-IDF + LR (class_weight={cw}, C={C})", p_lr.argmax(1))
        tab = pd.DataFrame(rows)
        say("\n" + tab.to_string(index=False))
        tab.to_csv(ANALYSIS / "trivial_baselines.csv", index=False)

    # ------------------------------------------------------------------ C
    with section("C  frozen splits"):
        # Every 9-way run shares one slice; sub-task runs (E12b/c) hold out that slice's rows of their own
        # labels, and E12a holds out nothing, so they are excluded from the check.
        nine_way = ("L0_large_base", "E8_fullq_bal_focal", "E8b_fullq_ce", "E9_hier", "E10_base_16ep",
                    "E10_fullq_16ep", "E11_base_8ep", "E11_base_16ep", "E11_fullq_16ep")
        vi = [np.load(f) for r in nine_way for f in sorted((ROOT / "runs" / r).glob("seed*/val_index.npy"))]
        same = all(np.array_equal(np.sort(v), np.sort(vi[0])) for v in vi)
        say(f"train-internal slice: {len(vi[0])} rows, identical across all {len(vi)} 9-way runs: {same}")
        (SPLITS / "train_internal_val_index.json").write_text(json.dumps(sorted(int(i) for i in vi[0])))
        strata = np.where(has_maj, maj, -1)
        dev_a = []
        for s in np.unique(strata):
            idx = rng.permutation(np.flatnonzero(strata == s))
            dev_a += idx[: (len(idx) + rng.integers(0, 2)) // 2].tolist()
        dev_a = np.sort(np.array(dev_a))
        dev_b = np.setdiff1d(np.arange(n), dev_a)
        (SPLITS / "dev_A_B.json").write_text(
            json.dumps({"seed": SEED, "stratified_by": "majority 9-way label (no-majority items as one stratum)",
                        "index_is": "row position in the 308-item dev parquet",
                        "dev_A": dev_a.tolist(), "dev_B": dev_b.tolist()})
        )
        say(f"dev-A {len(dev_a)} items, dev-B {len(dev_b)} items, saved to clarity/data/splits/dev_A_B.json")

    # ------------------------------------------------------------------ D
    probs = {k: load_seed_probs(v) for k, v in SYSTEMS.items()}
    with section("D  main table"):
        rows = []
        rules = make_rules(prior)
        for name, P in probs.items():
            ens = P.mean(0)
            single_s2 = [s2(p.argmax(1), mask) for p in P]
            single_s1 = [s1(p.argmax(1), ctrue) for p in P]
            row = {
                "system": name,
                "seeds": len(P),
                "single S2": f"{np.mean(single_s2):.3f} ± {np.std(single_s2, ddof=1):.3f}",
                "single S1": f"{np.mean(single_s1):.3f} ± {np.std(single_s1, ddof=1):.3f}",
                "ens S2 (full dev, untuned)": round(s2(ens.argmax(1), mask), 3),
                "ens S1 (full dev, untuned)": round(s1(ens.argmax(1), ctrue), 3),
                "ens S2 dev-B untuned": round(s2(ens[dev_b].argmax(1), mask[dev_b]), 3),
            }
            # 2-fold: tune on one half, report on the other, both directions.
            for tag, (tr, te) in {"A->B": (dev_a, dev_b), "B->A": (dev_b, dev_a)}.items():
                tau, _ = fit_tau(ens[tr], mask[tr], prior, np.linspace(-1.0, 2.0, 61))
                pred = logit_adjust(ens[te], prior, tau).argmax(1)
                row[f"+LA S2 {tag}"] = round(s2(pred, mask[te]), 3)
                row[f"+LA S1 {tag}"] = round(score_subtask1(leaf_to_official_clarity(pred), ctrue[te]).macro_f1, 3)
                logits = np.log(ens + 1e-12)
                trm = tr[has_maj[tr]]
                t = fit_temperature(logits[trm], maj[trm])
                b = fit_bias(logits[tr] / t, mask[tr])
                pred = (logits[te] / t + b).argmax(1)
                row[f"+T&bias S2 {tag}"] = round(s2(pred, mask[te]), 3)
            nc = nested_cv(rules["R1_logit_adjusted"], ens, mask, 5, 0)
            row["+LA S2 nested 5-fold CV (full dev)"] = round(nc["out_of_fold_macro_f1"], 3)
            rows.append(row)
        tab = pd.DataFrame(rows).set_index("system").T
        say(tab.to_string())
        tab.to_csv(ANALYSIS / "main_table.csv")
        say("\nLA = logit adjustment (1 parameter). T&bias = temperature + 9 per-class biases (brief §3.3d).")

    # ------------------------------------------------------------------ E
    with section("E  bootstrap 95% intervals (1,000 resamples over dev items)"):
        ens_f, ens_b = probs[FINAL].mean(0), probs[BASELINE].mean(0)
        pf, pb = ens_f.argmax(1), ens_b.argmax(1)
        lo, hi = bootstrap(lambda i: s2(pf[i], mask[i]), n, rng)
        say(f"FINAL ensemble, untuned, full dev: S2 {s2(pf, mask):.3f}  [{lo:.3f}, {hi:.3f}]")
        lo, hi = bootstrap(lambda i: s1(pf[i], ctrue[i]), n, rng)
        say(f"FINAL ensemble, untuned, full dev: S1 {s1(pf, ctrue):.3f}  [{lo:.3f}, {hi:.3f}]")
        pB = pf[dev_b]
        lo, hi = bootstrap(lambda i: s2(pB[i], mask[dev_b][i]), len(dev_b), rng)
        say(f"FINAL ensemble, untuned, dev-B only: S2 {s2(pB, mask[dev_b]):.3f}  [{lo:.3f}, {hi:.3f}]  (half-dev width)")
        lo, hi = bootstrap(lambda i: s2(pf[i], mask[i]) - s2(pb[i], mask[i]), n, rng)
        say(f"FINAL minus BASELINE ensemble, untuned, full dev: S2 {s2(pf, mask) - s2(pb, mask):+.3f}  [{lo:+.3f}, {hi:+.3f}]")
        lo, hi = bootstrap(lambda i: s1(pf[i], ctrue[i]) - s1(pb[i], ctrue[i]), n, rng)
        say(f"FINAL minus BASELINE ensemble, untuned, full dev: S1 {s1(pf, ctrue) - s1(pb, ctrue):+.3f}  [{lo:+.3f}, {hi:+.3f}]")
        say("Seeds are held fixed, so these intervals cover item sampling only, not seed variance.")

    # ------------------------------------------------------------------ F
    with section("F  error analysis: FINAL 10-seed ensemble, argmax (no dev tuning)"):
        pred = probs[FINAL].mean(0).argmax(1)
        hit = mask[np.arange(n), pred]
        say(f"errors (prediction not in reference set): {(~hit).sum()} of {n} ({100 * (~hit).mean():.1f}%); in-set {hit.mean():.3f}")
        say(f"errors on unanimous items: {(~hit & (cons == 1)).sum()} (TeleAI: 47 of 112 errors)")

        r2 = score_subtask2(pred, mask)
        r1 = score_subtask1(leaf_to_official_clarity(pred), ctrue)
        pc = pd.DataFrame(
            [{"subtask": 2, "class": c, "precision": m.precision, "recall": m.recall, "f1": m.f1, "support": m.support}
             for c, m in r2.per_class.items()]
            + [{"subtask": 1, "class": c, "precision": m.precision, "recall": m.recall, "f1": m.f1, "support": m.support}
               for c, m in r1.per_class.items()]
        ).round(3)
        say("\nper-class (S2 multi-reference; support is prediction-dependent under this scorer):")
        say(pc.to_string(index=False))
        pc.to_csv(ANALYSIS / "per_class_final.csv", index=False)
        say(f"\nDodging recall {r2.per_class['Dodging'].recall:.3f} (TeleAI CAMSR-CoT: 0.179); "
            f"Dodging F1 {r2.per_class['Dodging'].f1:.3f} (TeleAI: 0.289)")

        T = [EVASION_LABELS.index(c) for c in TRIANGLE]
        err = np.flatnonzero(~hit)
        involves = np.array([pred[i] in T or mask[i, T].any() for i in err])
        inside = np.array([pred[i] in T and mask[i, T].any() for i in err])
        say(f"\nDodging-General-Deflection triangle, TeleAI's definition (error involves at least one of the three "
            f"labels, in prediction or reference set): {involves.sum()} of {len(err)} errors = {100 * involves.mean():.1f}% (TeleAI 77.7%)")
        say(f"stricter: prediction AND a reference label both in the triangle: {inside.sum()} = {100 * inside.mean():.1f}%")

        rows_lbl = [EVASION_LABELS[m] if h else "(no majority)" for m, h in zip(maj, has_maj)]
        cm = pd.crosstab(pd.Series(rows_lbl, name="majority gold"), pd.Series([EVASION_LABELS[p] for p in pred], name="predicted"))
        cm = cm.reindex(index=list(EVASION_LABELS) + ["(no majority)"], columns=EVASION_LABELS, fill_value=0)
        cm.to_csv(ANALYSIS / "confusion_final.csv")
        say("\nconfusion matrix (rows = majority gold, cols = prediction; a cell off the diagonal can still be in-set):")
        short = [c[:6] for c in EVASION_LABELS]
        say(cm.set_axis(short, axis=1).to_string())
        pairs = Counter()
        for i in err:
            for g in np.flatnonzero(mask[i]):
                pairs[(EVASION_LABELS[g], EVASION_LABELS[pred[i]])] += 1
        say("\nmost frequent (reference label -> wrong prediction) pairs among errors:")
        for (g, p_), k in pairs.most_common(8):
            say(f"  {g:22} -> {p_:22} {k}")

        say("\nin-set rate by annotator consensus:")
        for c, nm in ((1, "unanimous"), (2, "2-1 split"), (3, "all three differ")):
            say(f"  {nm:17} n={int((cons == c).sum()):3}  in-set {hit[cons == c].mean():.3f}")

        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
        ln = np.array([len(tok(q, a)["input_ids"]) for q, a in zip(dev["question"], dev["interview_answer"])])
        lf = np.array([len(tok(f"Sub-question: {q} Full question: {f}", a)["input_ids"])
                       for q, f, a in zip(dev["question"], dev["interview_question"], dev["interview_answer"])])
        say("\nin-set rate by input length (DeBERTa tokens):")
        for nm, sel in (("[sub-q; answer] <= 512", ln <= 512), ("[sub-q; answer] > 512", ln > 512),
                        ("full input <= 1024 (seen whole)", lf <= 1024), ("full input > 1024 (cut)", lf > 1024)):
            say(f"  {nm:32} n={int(sel.sum()):3}  in-set {hit[sel].mean():.3f}")
        short_half = lf <= np.median(lf)
        say(f"  shorter half of dev (test-like; test median input is 137 tokens)  n={short_half.sum()}  "
            f"S2 {s2(pred[short_half], mask[short_half]):.3f}  vs longer half {s2(pred[~short_half], mask[~short_half]):.3f}")

        ex = [i for i in err if pred[i] in T and mask[i, T].any()]
        ex = rng.choice(ex, size=min(10, len(ex)), replace=False)
        lines = ["# Ten triangle errors (FINAL 10-seed ensemble, argmax)\n",
                 "Sampled at random (seed 2026) from errors where the prediction and a reference label are both in "
                 "{Dodging, General, Deflection}. Answers are truncated to 600 characters.\n"]
        for k, i in enumerate(sorted(ex), 1):
            row = dev.iloc[i]
            lines += [f"## {k}. dev row {i}\n",
                      f"- **Sub-question:** {row['question']}",
                      f"- **Annotators:** {', '.join(EVASION_LABELS[j] for j in lab[i])}",
                      f"- **Predicted:** {EVASION_LABELS[pred[i]]} (p = {probs[FINAL].mean(0)[i, pred[i]]:.2f})",
                      f"- **Answer:** {str(row['interview_answer'])[:600].strip()}…\n"]
        (ANALYSIS / "error_examples.md").write_text("\n".join(lines))
        say(f"\n{len(ex)} triangle error examples written to clarity/reports/raw/mideval/error_examples.md")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.imshow(cm.values, cmap="Blues")
        ax.set_xticks(range(9), short, rotation=45, ha="right")
        ax.set_yticks(range(10), short + ["no maj."])
        for (r, c), v in np.ndenumerate(cm.values):
            if v:
                ax.text(c, r, v, ha="center", va="center", fontsize=8, color="white" if v > cm.values.max() / 2 else "black")
        ax.set_xlabel("predicted")
        ax.set_ylabel("majority gold")
        ax.set_title("FINAL 10-seed ensemble, dev (argmax, no dev tuning)")
        fig.tight_layout()
        fig.savefig(ANALYSIS / "confusion_final.png", dpi=150)

    # ------------------------------------------------------------------ G
    with section("G  human ceiling and seed diversity"):
        say("each annotator scored against the other two (the test set's 2-annotator regime):")
        hum = []
        for k in range(3):
            m = np.zeros((n, N_EVASION), bool)
            for j in (j for j in range(3) if j != k):
                m[np.arange(n), lab[:, j]] = True
            hum.append(s2(lab[:, k], m))
            say(f"  annotator{k + 1}: S2 {hum[-1]:.3f}  in-set {inset(lab[:, k], m):.3f}")
        say(f"  mean human S2 {np.mean(hum):.3f}")
        say(f"FINAL ensemble on the same 2-annotator sets: {[round(s2(pf, m), 3) for m in masks2]}")
        P = probs[FINAL]
        preds = P.argmax(2)
        agree = [float((a == b).mean()) for a, b in combinations(preds, 2)]
        allsame = (preds == preds[0]).all(0)
        say(f"\nFINAL seeds: pairwise agreement {np.mean(agree):.3f}; all {len(P)} agree on {allsame.mean():.3f} of items")
        say(f"  ensemble in-set where all agree {hit[allsame].mean():.3f}, where they split {hit[~allsame].mean():.3f}")
        conf = P.mean(0).max(1)
        say("  in-set rate by confidence (top ensemble probability), fifths of dev from least to most confident:")
        edges = np.quantile(conf, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        for lo_, hi_ in zip(edges[:-1], edges[1:]):
            sel = (conf >= lo_) & (conf <= hi_)
            say(f"    p in [{lo_:.2f}, {hi_:.2f}]  n={int(sel.sum()):3}  in-set {hit[sel].mean():.3f}")
        o = np.argsort(-P.mean(0), 1)
        say("  top-k contains an acceptable label: " + ", ".join(
            f"k={k} {mask[np.arange(n)[:, None], o[:, :k]].any(1).mean():.3f}" for k in (1, 2, 3, 5)))

    # ------------------------------------------------------------------ H
    with section("H  the proposal's C4 prediction: does the decision rule gain on non-unanimous items?"):
        say("Proposal §7: 'C4's gains should fall on non-unanimous items and the two highest-disagreement")
        say("classes; gains on unanimous items would falsify the account.' Tested with the rule that won (logit")
        say("adjustment, held-out predictions from 5-fold nested CV) against plain argmax, same ensemble.")
        rules = make_rules(prior)
        for name in (BASELINE, FINAL):
            ens = probs[name].mean(0)
            p0 = ens.argmax(1)
            p1 = nested_cv(rules["R1_logit_adjusted"], ens, mask, 5, 0)["pooled_predictions"]
            h0, h1 = mask[np.arange(n), p0], mask[np.arange(n), p1]
            say(f"\n{name}: S2 {s2(p0, mask):.3f} -> {s2(p1, mask):.3f}")
            for c, nm in ((1, "unanimous"), (2, "2-1 split"), (3, "all three differ")):
                sel = cons == c
                say(f"  {nm:17} n={int(sel.sum()):3}  in-set {h0[sel].mean():.3f} -> {h1[sel].mean():.3f}   "
                    f"changed {int((p0 != p1)[sel].sum()):3}  fixed {int((~h0 & h1)[sel].sum()):2}  broken {int((h0 & ~h1)[sel].sum()):2}")
            f0, f1 = score_subtask2(p0, mask).per_class, score_subtask2(p1, mask).per_class
            gain = sorted(((f1[c].f1 - f0[c].f1, c) for c in EVASION_LABELS), reverse=True)
            say("  per-class F1 change: " + ", ".join(f"{c} {d:+.3f}" for d, c in gain))
        say("\nThe two highest-disagreement classes on dev are General (90% of its reference sets non-unanimous,")
        say("experiment log Step 0) and Deflection. Read the per-class line against them.")

    wall, cpu = time.perf_counter() - t_all, time.process_time() - c_all
    say(f"\n{'=' * 78}\nCOMPUTE: total {wall:.1f} s wall, {cpu:.1f} s CPU, GPU 0 s")
    for k, (w, c) in TIMES.items():
        say(f"  {k:62} {w:6.1f} s wall  {c:6.1f} s CPU")
    (ANALYSIS / "mideval_analysis.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    sys.exit(main())
