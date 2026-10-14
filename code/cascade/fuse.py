#!/usr/bin/env python3
"""M3 + M2: fuse the LLM's q with M1 over C(x), defer by confidence, and score (E15 steps 1 and 3).

    python -m cascade.fuse screen --tags qwen38_27b_k0,qwen38_27b_k4,qwen38_27b_k10,...
    python -m cascade.fuse slice --tags <adopted tag>     # confirmation: the 30% least confident slice items
    python -m cascade.fuse dev --tag <adopted tag>

Fused score over the candidates of a deferred item: lam * log p̄ + (1 - lam) * log q - tau * log pi.
Items that are not deferred keep the M1 system's decision, argmax of log p̄ - tau_M1 * log pi, with M1's
own tau (the E4 rule); the deferred items' (lam, tau) are fitted separately.
  * screen: train rows (one gold label). The LLM alone, M1, and the fusion with (lam, tau) fitted by
    5-fold CV inside the screen rows (accuracy), plus NLL, complementarity and letter bias.
  * dev: the system. (lam, tau) are fitted by 5-fold nested CV on dev (macro-F1, 10 splits); the deferral
    threshold for a rate delta is the (1 - delta) quantile of u = 1 - top q on the TRAIN SLICE, so the
    realised rate on dev is reported too. Paired bootstrap against M1's own nested-CV system.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from cascade.adjudicate import M1_INPUTS, SCREEN_DELTA, select
from decision_rules.decide import fit_tau, make_rules, nested_cv, reference_masks
from qevasion.labels import EVASION_LABELS, N_EVASION, encode_clarity, encode_evasion, leaf_to_official_clarity
from qevasion.loader import DATA_CACHE, dev_reference_mask, load_qevasion
from qevasion.paths import RUNS
from qevasion.scoring import score_subtask1, score_subtask2

# 0, 0.1, ..., 1 plus finer steps near 1: a sharp LLM (Gemma: q ~ one-hot, E15 probe) needs lam close to 1
# before M1 can overrule it
LAMS = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 1.0])
TAUS = np.linspace(-1.0, 2.0, 61)
DELTAS = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)

sp = load_qevasion(DATA_CACHE)
Y = encode_evasion(sp.train["evasion_label"].tolist())
PRIOR = np.bincount(Y, minlength=N_EVASION) / len(Y)
LOGPI = np.log(PRIOR)
Z = np.load(M1_INPUTS)


def load_q(tag: str, split: str, n: int, rows: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(len(rows) or n, 9) log q with -inf outside C(x), and a mask of scored rows."""
    pos = {int(r): k for k, r in enumerate(rows)} if rows is not None else None
    m = len(rows) if rows is not None else n
    lq = np.full((m, N_EVASION), -np.inf)
    ok = np.zeros(m, dtype=bool)
    for line in (RUNS / "M3" / tag / "responses.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["split"] != split or (pos is not None and r["i"] not in pos):
            continue
        k = pos[r["i"]] if pos is not None else r["i"]
        lq[k, r["cand"]] = np.log(np.maximum(r["q"], 1e-12))
        ok[k] = True
    return lq, ok


def decide(lp: np.ndarray, lq: np.ndarray, defer: np.ndarray, lam: float, tau: float,
           m1: np.ndarray | None = None) -> np.ndarray:
    """Deferred items: fused argmax over C(x). Others: M1's own decision `m1` (default: lp - tau log pi).

    In the dev system `m1` is always passed, with M1's OWN tau, so items that are not deferred keep exactly
    the answer of the M1 system (report §7). Sharing one tau between the two was a bug (fixed 2026-10-14).
    """
    if m1 is None:
        m1 = (lp - tau * LOGPI).argmax(1)
    in_c = np.isfinite(lq)  # lq is -inf outside C(x); mask explicitly, since 0 * -inf is nan at lam = 1
    s = np.where(in_c, lam * lp + (1 - lam) * np.where(in_c, lq, 0.0) - tau * LOGPI, -np.inf)
    return np.where(defer, s.argmax(1), m1)


def fit_system(lp, lq, p_bar, defer, gold, tr):
    """Fit on rows tr: M1's tau by the E4 rule (as the M1 system), then (lam, tau) of the deferred items with
    M1's decisions held fixed elsewhere. Returns (predictions for all rows, in-fold S2, lam, tau, tau_m1)."""
    tau1, _ = fit_tau(p_bar[tr], gold[tr], PRIOR, TAUS)
    m1 = (lp - tau1 * LOGPI).argmax(1)
    best = (-1.0, None, None, None)
    for lam in LAMS:
        for tau in TAUS:
            pred = decide(lp, lq, defer, lam, tau, m1)
            v = score_subtask2(pred[tr], gold[tr]).macro_f1
            if v > best[0]:
                best = (v, pred, lam, tau)
    return best[1], best[0], best[2], best[3], tau1


# ---------------------------------------------------------------- screen (train rows, single gold)

def screen(tags: list[str], split: str = "screen") -> None:
    """split = screen (cross-fitted train rows), or slice: the 30% least confident slice items, with the
    10-seed ensemble's candidates and probabilities (the confirmation set of the registration)."""
    jobs = select(split, Z)
    if split == "slice":
        u = 1 - Z["val_q"].max(1)
        keep = u >= np.quantile(u, 1 - SCREEN_DELTA)
        jobs = [j for j, kp in zip(jobs, keep) if kp]
        lp_all, q_all = Z["val_pbar"][keep], Z["val_q"][keep]
    rows = np.array([j.i for j in jobs])
    y = Y[rows]
    if split == "slice":
        lp, m1_top = np.log(lp_all + 1e-12), q_all.argmax(1)
    else:
        lp = np.log(Z["train_pbar"][rows] + 1e-12)
        m1_top = Z["train_q"][rows].argmax(1)
    cand = np.zeros((len(rows), N_EVASION), dtype=bool)
    for k, j in enumerate(jobs):
        cand[k, j.cand] = True
    in_c = cand[np.arange(len(rows)), y]
    print(f"{split}: {len(rows)} rows; gold in top-4 {in_c.mean():.3f}; M1 top-1 accuracy {np.mean(m1_top == y):.3f}; "
          f"random within C {in_c.mean() / 4:.3f}")
    folds = np.array_split(np.random.default_rng(0).permutation(len(rows)), 5)
    every = np.ones(len(rows), dtype=bool)
    res = {}
    for tag in tags:
        lq, ok = load_q(tag, split, 0, rows)
        if not ok.all():
            print(f"{tag}: only {ok.sum()} of {len(rows)} rows scored -- skipped")
            continue
        llm = lq.argmax(1)
        fused = np.empty(len(rows), dtype=np.int64)
        for f in range(5):
            te = folds[f]
            tr = np.setdiff1d(np.arange(len(rows)), te)
            best = max(((np.mean(decide(lp[tr], lq[tr], every[tr], lam, tau) == y[tr]), lam, tau)
                        for lam in LAMS for tau in TAUS[::3]), key=lambda t: t[0])
            fused[te] = decide(lp[te], lq[te], every[te], best[1], best[2])
        nll = -lq[np.arange(len(rows)), y][in_c].mean()
        recs = [json.loads(line) for line in (RUNS / "M3" / tag / "responses.jsonl").read_text().splitlines()]
        recs = [r for r in recs if r["split"] == split and r["i"] in set(rows.tolist())]
        letters = np.bincount([int(np.argmax(r["q"])) for r in recs], minlength=4)
        res[tag] = {"llm": llm == y, "fused": fused == y}
        print(f"\n{tag}\n  LLM alone within C: {np.mean(llm == y):.3f}   fused (5-fold CV): {np.mean(fused == y):.3f}   "
              f"M1: {np.mean(m1_top == y):.3f}\n  NLL of gold under q (gold in C, n={in_c.sum()}): {nll:.3f}\n"
              f"  LLM right & M1 wrong {np.mean((llm == y) & (m1_top != y)):.3f};  "
              f"LLM wrong & M1 right {np.mean((llm != y) & (m1_top == y)):.3f}\n"
              f"  argmax letter A-D {letters.tolist()};  mean letter mass {np.mean([r['letter_mass'] for r in recs]):.3f};  "
              f"items with a letter missing from top-20 {np.mean([r['missing'] > 0 for r in recs]):.3f}")
        if "usd" in recs[0]:
            print(f"  cost ${sum(r['usd'] for r in recs):.4f} for {len(recs)} items; "
                  f"mean prompt tokens {np.mean([r['usage']['prompt'] for r in recs]):.0f}, "
                  f"cached {np.mean([r['usage']['cached'] for r in recs]):.0f}")
        elif "gpu_sec" in recs[0]:
            print(f"  GPU-seconds per item {np.mean([r['gpu_sec'] for r in recs]):.3f}; "
                  f"mean prompt tokens {np.mean([r['prompt_tokens'] for r in recs]):.0f}")
    boots = [np.random.default_rng(i).integers(0, len(rows), len(rows)) for i in range(2000)]
    names = list(res)
    if len(names) > 1:
        print("\npaired differences in fused accuracy (bootstrap over rows)")
        for a_ in names:
            for b_ in names:
                if a_ < b_:
                    d = res[a_]["fused"].astype(float) - res[b_]["fused"].astype(float)
                    bd = np.array([d[i].mean() for i in boots])
                    print(f"  {a_} - {b_}: {d.mean():+.3f}  95% [{np.percentile(bd, 2.5):+.3f}, "
                          f"{np.percentile(bd, 97.5):+.3f}]")


# ---------------------------------------------------------------- dev system

def dev(tag: str, splits: int) -> None:
    gold = dev_reference_mask(sp.dev)
    clar = encode_clarity(sp.dev["clarity_label"].tolist())
    lp = np.log(Z["dev_pbar"] + 1e-12)
    lq, ok = load_q(tag, "dev", len(gold))
    assert ok.all(), f"{tag}: only {ok.sum()} of {len(gold)} dev items scored"
    u_dev, u_slice = 1 - Z["dev_q"].max(1), 1 - Z["val_q"].max(1)
    s2 = lambda p, g=gold: score_subtask2(p, g).macro_f1  # noqa: E731
    s1 = lambda p: score_subtask1(leaf_to_official_clarity(p), clar).macro_f1  # noqa: E731
    r1 = make_rules(PRIOR)["R1_logit_adjusted"]
    base = [nested_cv(r1, Z["dev_pbar"], gold, 5, cs) for cs in range(splits)]
    m1_pred = base[0]["pooled_predictions"]
    two = {n: g for n, g in reference_masks(sp.dev, True).items() if n != "3ann"}
    m1_two = np.mean([nested_cv(r1, Z["dev_pbar"], g, 5, 0)["out_of_fold_macro_f1"] for g in two.values()])
    print(f"M1 system (ensemble + LA, nested CV): S2 {s2(m1_pred):.3f} ({splits} splits "
          f"{np.mean([b['out_of_fold_macro_f1'] for b in base]):.3f}), S1 {s1(m1_pred):.3f}, "
          f"2-annotator mean {m1_two:.3f}")
    boots = [np.random.default_rng(i).integers(0, len(gold), len(gold)) for i in range(2000)]
    print(f"\n{'delta':>5} {'calls/item':>10} {'S2':>6} {'S2 10sp':>7} {'S1':>6} {'2-ann':>6}  lam    tau   Δ S2 vs M1 (bootstrap)")
    s1_rows: list[str] = []
    preds: dict[float, np.ndarray] = {}
    for delta in DELTAS:
        thr = np.inf if delta == 0 else (-np.inf if delta == 1 else np.quantile(u_slice, 1 - delta))
        defer = u_dev >= thr

        def rule(_p, g, tr, te, defer=defer):
            pred_all, fit, *_ = fit_system(lp, lq, Z["dev_pbar"], defer, g, tr)
            return pred_all[te], fit

        runs = [nested_cv(rule, lp, gold, 5, cs) for cs in range(splits)]
        pred = runs[0]["pooled_predictions"]
        preds[delta] = pred
        _, _, lam_all, tau_all, _ = fit_system(lp, lq, Z["dev_pbar"], defer, gold, np.arange(len(gold)))
        full = (None, lam_all, tau_all)
        d = np.array([s2(pred[i], gold[i]) - s2(m1_pred[i], gold[i]) for i in boots])
        d1 = np.array([score_subtask1(leaf_to_official_clarity(pred[i]), clar[i]).macro_f1
                       - score_subtask1(leaf_to_official_clarity(m1_pred[i]), clar[i]).macro_f1 for i in boots])
        s1_rows.append(f"{delta:>5.1f}  Δ S1 vs M1 {d1.mean():+.3f} [{np.percentile(d1, 2.5):+.3f}, "
                       f"{np.percentile(d1, 97.5):+.3f}] P(<=0) {np.mean(d1 <= 0):.3f}")
        print(f"{delta:>5.1f} {defer.mean():>10.3f} {s2(pred):>6.3f} "
              f"{np.mean([r['out_of_fold_macro_f1'] for r in runs]):>7.3f} {s1(pred):>6.3f} "
              f"{np.mean([nested_cv(rule, lp, g, 5, 0)['out_of_fold_macro_f1'] for g in two.values()]):>6.3f}  {full[1]:.3f} {full[2]:+.2f}   "
              f"{d.mean():+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}] P(<=0) {np.mean(d <= 0):.3f}   in-set {gold[np.arange(len(gold)), pred].mean():.3f}")
    print("\nSubtask 1, paired bootstrap of the split-0 systems:\n" + "\n".join(s1_rows))
    from scipy.stats import binomtest

    idx = np.arange(len(gold))
    for delta in (0.3, 1.0):
        pr = preds[delta]
        fixed = int((gold[idx, pr] & ~gold[idx, m1_pred]).sum())
        broken = int((~gold[idx, pr] & gold[idx, m1_pred]).sum())
        a_, b_ = score_subtask2(m1_pred, gold).per_class, score_subtask2(pr, gold).per_class
        print(f"\nδ = {delta}: {int((pr != m1_pred).sum())} predictions changed; {fixed} move into the reference set, "
              f"{broken} out of it (exact McNemar p = {binomtest(fixed, max(fixed + broken, 1), 0.5).pvalue:.4f})")
        print(f"   {'class':<22}{'M1 F1':>7}{'M1+M3':>7}{'Δ':>7}{'#pred M1':>9}{'#pred M3':>9}")
        for c, name in enumerate(EVASION_LABELS):
            print(f"   {name:<22}{a_[name].f1:>7.3f}{b_[name].f1:>7.3f}{b_[name].f1 - a_[name].f1:>+7.3f}"
                  f"{int((m1_pred == c).sum()):>9}{int((pr == c).sum()):>9}")
    llm = lq.argmax(1)
    print(f"\nLLM alone within C(x), all dev items: in-set rate {gold[np.arange(len(gold)), llm].mean():.3f}, "
          f"S2 {s2(llm):.3f}; per class predicted: "
          f"{ {EVASION_LABELS[c][:10]: int(n) for c, n in enumerate(np.bincount(llm, minlength=N_EVASION)) if n} }")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=("screen", "slice", "dev"))  # dev: 2-annotator column refits the nested CV per reference set, as E14
    p.add_argument("--tags", default="")
    p.add_argument("--tag", default="")
    p.add_argument("--splits", type=int, default=10, help="nested-CV splits for dev")
    a = p.parse_args()
    if a.cmd in ("screen", "slice"):
        screen([t for t in a.tags.split(",") if t], a.cmd)
    else:
        dev(a.tag, a.splits)


if __name__ == "__main__":
    main()
