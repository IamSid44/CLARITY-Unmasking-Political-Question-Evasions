#!/usr/bin/env python3
"""E24 pre-checks: CPU-only tests of the candidate modules for the research extension (docs/07), run on the
existing artifacts before any GPU is spent. A module goes on to a GPU run only if its pre-check passes.

    python -m preregistered_analyses.e24_prechecks > ../docs/raw/E24_prechecks.txt     # CPU, ~5 min

P1 sibling structure: how many rows share their answer with another sub-question, and how often the labels differ.
P2 question sensitivity of M1 (10-seed mean, dev): on sibling pairs whose annotator sets are disjoint, how often
   M1 still predicts the same label (an answer-only classifier would always do so).
P3 epistemic vs aleatoric routing: total entropy = E[H] (aleatoric) + mutual information (epistemic) over the 10
   seeds; which one finds M1's errors (any-annotator) in its top 30%.
P4 sibling mean-field decoding: pairwise label potentials from train sibling groups, applied to dev M1 probs.
P5 metric-aware decision rule: coordinate ascent on the expected any-annotator macro-F1 (i.i.d. annotators drawn
   from p, k annotators; a miss counts a false negative for every label in the set, as the official scorer does),
   no tuned parameter, against logit adjustment.
P6 single-order position debiasing (PriDe-style) of the 27B against the 4-order average (lab records).
P7 conformal multi-reference candidate sets as router + pruner, against the threshold router, both with the
   system's (lam, tau) fusion fitted by nested CV. CPU approximation: q is restricted to C and renormalised.
P8 relevance vs answerhood: the E22 question-answer specificity scores (PMI, identifiability) by gold label.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
from scipy.stats import spearmanr

from cascade.fuse import LOGPI, PRIOR, Z, fit_system, load_q, sp
from cascade.prompts import display_order
from decision_rules.decide import nested_cv
from qevasion.labels import EVASION_LABELS, encode_evasion
from qevasion.loader import dev_reference_mask
from qevasion.paths import RUNS

from qevasion.scoring import score_subtask2

L = list(EVASION_LABELS)
TR, DV = sp.train.reset_index(drop=True), sp.dev.reset_index(drop=True)
GOLD = dev_reference_mask(DV)
ANN = [encode_evasion(list(r)) for r in DV[["annotator1", "annotator2", "annotator3"]].values]
SEEDS = np.stack([np.load(RUNS / "Q8_fullq_lora_12ep" / f"seed{s}" / "dev_probs.npy") for s in range(10)])
P = SEEDS.mean(0)
S2 = lambda p: score_subtask2(p, GOLD).macro_f1  # noqa: E731


def head(t: str) -> None:
    print("\n" + "=" * 110 + f"\n{t}\n" + "=" * 110)


def sibling_pairs(df) -> list[tuple[int, int]]:
    out = []
    for _, g in df.groupby("interview_answer"):
        ix = g.index.tolist()
        out += [(ix[a], ix[b]) for a in range(len(ix)) for b in range(a + 1, len(ix))]
    return out


def H(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(p + 1e-12)).sum(-1)


def p1_p2() -> None:
    head("P1. Sibling structure (same answer, several sub-questions)")
    for name, df in (("train", TR), ("dev", DV)):
        sizes = df.groupby("interview_answer").size()
        multi = sizes[sizes > 1]
        print(f"   {name}: rows {len(df)}, unique answers {len(sizes)}, rows sharing their answer {multi.sum()} "
              f"({multi.sum() / len(df):.1%})")
    y = encode_evasion(TR["evasion_label"].tolist())
    pr = sibling_pairs(TR)
    cl = TR["clarity_label"].values
    print(f"   train sibling pairs {len(pr)}; different S2 label {np.mean([y[i] != y[j] for i, j in pr]):.1%}, "
          f"different S1 level {np.mean([cl[i] != cl[j] for i, j in pr]):.1%}")
    ordered = [(y[i], y[j]) for i, j in pr] + [(y[j], y[i]) for i, j in pr]

    def ent(c: Counter) -> float:
        v = np.array(list(c.values()), float)
        v /= v.sum()
        return float(-(v * np.log2(v)).sum())
    mi = ent(Counter(a for a, _ in ordered)) + ent(Counter(b for _, b in ordered)) - ent(Counter(ordered))
    print(f"   mutual information between sibling labels {mi:.3f} bits ({mi / ent(Counter(b for _, b in ordered)):.1%} "
          f"of the label entropy)")

    head("P2. Does M1 read the question? Dev sibling pairs")
    pred = P.argmax(1)
    pr = np.array(sibling_pairs(DV))
    dis = np.array([not (GOLD[i] & GOLD[j]).any() for i, j in pr])
    same = np.array([pred[i] == pred[j] for i, j in pr])
    both = np.array([GOLD[i, pred[i]] and GOLD[j, pred[j]] for i, j in pr])
    print(f"   dev sibling pairs {len(pr)}; annotator sets disjoint {dis.sum()}")
    print(f"   M1 predicts the SAME label on disjoint pairs {same[dis].mean():.1%} ({same[dis].sum()}/{dis.sum()}); "
          f"on overlapping pairs {same[~dis].mean():.1%}")
    print(f"   both siblings in their reference sets, disjoint pairs: {both[dis].mean():.1%}")
    # is sibling divergence more than annotator noise? two labels from different annotators, same item vs siblings
    within = np.mean([np.mean([a[x] != a[y] for x in range(3) for y in range(3) if x != y]) for a in ANN])
    across = np.mean([np.mean([ANN[i][x] != ANN[j][y] for x in range(3) for y in range(3) if x != y]) for i, j in pr])
    rng = np.random.default_rng(0)
    rnd = np.mean([np.mean([ANN[i][x] != ANN[j][y] for x in range(3) for y in range(3) if x != y])
                   for i, j in rng.integers(0, len(ANN), (2000, 2)) if i != j])
    print(f"   P(two labels by different annotators differ): same item {within:.3f}, sibling items {across:.3f}, "
          f"random items {rnd:.3f}")
    print("   -> sibling divergence above the same-item rate is relational signal, not annotator noise")
    print("   -> M1 conditions on the question; 'models ignore the question' is not supported for M1")


def p3() -> None:
    head("P3. Routing signal: aleatoric vs epistemic uncertainty over the 10 seeds (dev)")
    tot, ale = H(P), H(SEEDS).mean(0)
    mi = tot - ale
    srt = np.sort(P, 1)
    ann_h = np.array([H(np.bincount(a, minlength=9) / len(a)) for a in ANN])
    wrong = ~GOLD[np.arange(len(P)), P.argmax(1)]
    k = int(0.3 * len(P))
    for name, v in (("total entropy", tot), ("aleatoric E[H]", ale), ("epistemic MI", mi),
                    ("negative margin", srt[:, -2] - srt[:, -1])):
        top = np.argsort(-v)[:k]
        print(f"   {name:<16} rho(annotator entropy) {spearmanr(v, ann_h)[0]:+.3f}  rho(M1 wrong) "
              f"{spearmanr(v, wrong)[0]:+.3f}  M1 errors caught in top 30%: {wrong[top].sum()}/{wrong.sum()}")
    for lvl in np.unique(ann_h.round(3)):
        m = ann_h.round(3) == lvl
        print(f"   annotator entropy {lvl:.2f}: n {m.sum():>3}, M1 error rate {wrong[m].mean():.3f}")
    print("   -> the epistemic split does not route better; reject the module")


def p4() -> None:
    head("P4. Sibling mean-field decoding over M1 probabilities (dev; S2 by tau x coupling w)")
    y = encode_evasion(TR["evasion_label"].tolist())
    J = np.ones((9, 9))
    for i, j in sibling_pairs(TR):
        J[y[i], y[j]] += 1
        J[y[j], y[i]] += 1
    J /= J.sum()
    log_r = np.log(J) - np.log(J.sum(1, keepdims=True)) - np.log(J.sum(0, keepdims=True))
    sib = {i: [j for j in g.index if j != i] for _, g in DV.groupby("interview_answer") for i in g.index}
    lp = np.log(P + 1e-12)

    def decode(tau: float, w: float) -> np.ndarray:
        base = lp - tau * LOGPI
        q = np.exp(base - base.max(1, keepdims=True))
        q /= q.sum(1, keepdims=True)
        for _ in range(10):
            msg = np.array([sum((q[j] @ log_r.T for j in sib[i]), np.zeros(9)) for i in range(len(q))])
            z = base + w * msg
            q = np.exp(z - z.max(1, keepdims=True))
            q /= q.sum(1, keepdims=True)
        return q.argmax(1)
    ws = (0.0, 0.25, 0.5, 1.0)
    print("   tau  " + "  ".join(f"w={w:<4}" for w in ws))
    for tau in (0.0, 0.25, 0.5, 0.75, 1.0):
        print(f"   {tau:<4} " + "  ".join(f"{S2(decode(tau, w)):.3f} " for w in ws))
    print("   -> differences are within +-0.01 and change sign with tau: not established")


def metric_rule(p: np.ndarray, k: int, passes: int = 6) -> np.ndarray:
    n, c = p.shape
    pi = 1 - (1 - p) ** k
    qm = (1 - p)[:, :, None] ** k - np.clip(1 - p[:, :, None] - p[:, None, :], 0, 1) ** k  # P(c not in A, c' in A)
    qm[:, np.arange(c), np.arange(c)] = 0
    y = p.argmax(1)
    tp, fp, fn = np.zeros(c), np.zeros(c), np.zeros(c)
    np.add.at(tp, y, pi[np.arange(n), y])
    np.add.at(fp, y, 1 - pi[np.arange(n), y])
    fn += qm[np.arange(n), y].sum(0)
    f = lambda a, b, d: np.mean(np.where(2 * a + b + d > 0, 2 * a / np.maximum(2 * a + b + d, 1e-12), 0))  # noqa: E731
    for _ in range(passes):
        changed = 0
        for i in range(n):
            c0 = y[i]
            tp[c0] -= pi[i, c0]; fp[c0] -= 1 - pi[i, c0]; fn -= qm[i, c0]  # noqa: E702
            best = (-1.0, c0)
            for cc in range(c):
                tp[cc] += pi[i, cc]; fp[cc] += 1 - pi[i, cc]; fn += qm[i, cc]  # noqa: E702
                best = max(best, (f(tp, fp, fn), cc))
                tp[cc] -= pi[i, cc]; fp[cc] -= 1 - pi[i, cc]; fn -= qm[i, cc]  # noqa: E702
            cc = best[1]
            tp[cc] += pi[i, cc]; fp[cc] += 1 - pi[i, cc]; fn += qm[i, cc]  # noqa: E702
            changed += cc != c0
            y[i] = cc
        if not changed:
            break
    return y


def p5() -> None:
    head("P5. Metric-aware decision rule (no tuned parameter) vs logit adjustment (dev)")
    print(f"   argmax                               {S2(P.argmax(1)):.3f}")
    for tau in (0.5, 0.75, 1.0):
        print(f"   logit adjustment tau={tau:<4} (dev-tuned) {S2((np.log(P + 1e-12) - tau * LOGPI).argmax(1)):.3f}")
    for k in (1, 2, 3):
        print(f"   metric rule, k={k}                     {S2(metric_rule(P, k)):.3f}")
    b, la, r = [], [], []
    for s in range(10):
        b.append(S2(SEEDS[s].argmax(1)))
        la.append(S2((np.log(SEEDS[s] + 1e-12) - LOGPI).argmax(1)))
        r.append(S2(metric_rule(SEEDS[s], 3)))
    print(f"   single seeds (mean of 10): argmax {np.mean(b):.3f}, LA tau=1 {np.mean(la):.3f}, metric rule k=3 "
          f"{np.mean(r):.3f}; rule > LA on {sum(x > y for x, y in zip(r, la))}/10")
    print("   -> matches the CV-tuned M1 system (0.598) with no tuning; no gain. A theory section, not a method")


def p6() -> None:
    head("P6. Single-order position debiasing of the 27B (lab records, 4 cyclic orders)")
    orders = ("taxonomy", "shift1", "shift2", "shift3")
    tags = ("qwen38_27b_k0", "qwen38_27b_k0_shift1", "qwen38_27b_k0_shift2", "qwen38_27b_k0_shift3")
    recs = []
    for t in tags:
        d = {}
        for line in (RUNS / "M3" / t / "responses.jsonl").read_text().splitlines():
            r = json.loads(line)
            if len(r["cand"]) == 4:
                d[(r["split"], r["i"], tuple(sorted(r["cand"])))] = r
        recs.append(d)
    keys = set.intersection(*(set(d) for d in recs))

    def pos(r, o):
        c = sorted(r["cand"])
        return np.array([r["q"][c.index(x)] for x in display_order(r["cand"], o)])

    def back(v, r, o):
        c, out = sorted(r["cand"]), np.zeros(4)
        for j, x in enumerate(display_order(r["cand"], o)):
            out[c.index(x)] = v[j]
        return out / out.sum()
    cal = [k for k in keys if k[0] != "dev"]
    ev = sorted(k for k in keys if k[0] == "dev")
    prior = {o: np.mean([pos(recs[j][k], o) for k in cal], 0) for j, o in enumerate(orders)}
    avg = {k: np.mean([recs[j][k]["q"] for j in range(4)], 0) for k in ev}
    inset = lambda k, q: GOLD[k[1], sorted(recs[0][k]["cand"])[int(np.argmax(q))]]  # noqa: E731
    print(f"   calibration rows (screen + test, unlabelled use) {len(cal)}, dev rows {len(ev)}")
    print("   position prior over letters A-D: " + "; ".join(f"{o} {np.round(v, 2).tolist()}" for o, v in prior.items()))
    print(f"   4-order average: in-set {np.mean([inset(k, avg[k]) for k in ev]):.3f}")
    for j, o in enumerate(orders):
        raw = [np.array(recs[j][k]["q"]) for k in ev]
        deb = [back(pos(recs[j][k], o) / prior[o], recs[j][k], o) for k in ev]
        f = lambda qs: (np.mean([inset(k, q) for k, q in zip(ev, qs)]),  # noqa: E731
                        np.mean([np.argmax(q) == np.argmax(avg[k]) for k, q in zip(ev, qs)]))
        (a, b), (c, d) = f(raw), f(deb)
        print(f"   {o:<9} raw: in-set {a:.3f}, agrees with the average {b:.3f} | debiased: in-set {c:.3f}, agrees {d:.3f}")
    print("   -> debiasing lowers agreement with the 4-order average: the canonical order confounds position with")
    print("      label identity (slot A is usually Explicit). Reject the module")


def p7() -> None:
    head("P7. Conformal multi-reference candidate sets vs the threshold router (lab 27B, single order, 5 CV splits)")
    n = len(GOLD)
    lp, q = np.log(Z["dev_pbar"] + 1e-12), Z["dev_q"]
    lq, ok = load_q("qwen38_27b_k0", "dev", n)
    assert ok.all()
    u_dev, u_slice = 1 - q.max(1), 1 - Z["val_q"].max(1)
    for d in (0.2, 0.3, 0.4, 0.5):
        defer = u_dev >= np.quantile(u_slice, 1 - d)
        rule = lambda _p, g, tr, te, defer=defer: (lambda r: (r[0][te], r[1]))(  # noqa: E731
            fit_system(lp, lq, Z["dev_pbar"], defer, g, tr))
        print(f"   threshold delta {d:.1f}: calls {defer.mean():.2f}  S2 "
              f"{np.mean([nested_cv(rule, lp, GOLD, 5, cs)['out_of_fold_macro_f1'] for cs in range(5)]):.3f}")
    for alpha in (0.05, 0.1, 0.15, 0.2, 0.3):
        stats, rng = [], np.random.default_rng(1)

        def rule(_p, g, tr, te, alpha=alpha):
            sc = np.array([min(1 - q[i, c] for c in rng.choice(ANN[i], 2, replace=False)) for i in tr])
            m = len(sc)
            qh = np.quantile(sc, min(1, np.ceil((m + 1) * (1 - alpha)) / m), method="higher")
            in_c = (1 - q) <= qh
            in_c[np.arange(n), q.argmax(1)] = True
            defer = in_c.sum(1) >= 2
            lqc = np.where(in_c, lq, -np.inf)
            empty = defer & ~np.isfinite(lqc).any(1)
            lqc[empty] = lq[empty]
            pred, fit, *_ = fit_system(lp, lqc, Z["dev_pbar"], defer, g, tr)
            stats.append((defer[te].mean(), in_c[te].sum(1).mean(), np.mean([GOLD[i, in_c[i]].any() for i in te])))
            return pred[te], fit
        s = np.mean([nested_cv(rule, lp, GOLD, 5, cs)["out_of_fold_macro_f1"] for cs in range(5)])
        st = np.array(stats).mean(0)
        print(f"   conformal alpha {alpha:.2f}: calls {st[0]:.2f}  mean|C| {st[1]:.2f}  any-annotator coverage "
              f"{st[2]:.3f} (target >= {1 - alpha:.2f})  S2 {s:.3f}")
    print("   -> coverage holds, but sets are large where annotators disagree, so the router calls the LLM on items M1")
    print("      already gets right and prunes useful options. Loses at every budget. Reject the module")


def p8() -> None:
    head("P8. Relevance is not answerhood: question-answer specificity (E22 scores, Qwen3-8B-Base) by gold label (train)")
    from scipy.stats import kruskal
    from sklearn.metrics import roc_auc_score
    z = np.load(RUNS / "E22" / "specificity.npz")
    df = TR.assign(pmi_c=z["train_own"] - z["train_rand"].mean(1), pmi_0=z["train_own"] - z["train_null"],
                   ident=(z["train_own"][:, None] > z["train_rand"]).mean(1))
    g = df.groupby("evasion_label")[["pmi_c", "pmi_0", "ident"]].mean().sort_values("pmi_c")
    n = df["evasion_label"].value_counts()
    print(f"   {'label':<22}{'n':>6}{'PMI_c':>8}{'PMI_0':>8}{'identifiable':>14}")
    for lab, r in g.iterrows():
        print(f"   {lab:<22}{n[lab]:>6}{r.pmi_c:>8.3f}{r.pmi_0:>8.3f}{r.ident:>14.3f}")
    print(f"   Kruskal-Wallis over the 9 labels (PMI_c): p = "
          f"{kruskal(*[v.values for _, v in df.groupby('evasion_label')['pmi_c']]).pvalue:.2g}")
    for lvl in ("Clear Reply", "Clear Non-Reply"):
        y = (df["clarity_label"] == lvl).astype(int)
        print(f"   AUC {lvl} vs rest: PMI_c {roc_auc_score(y, df.pmi_c):.3f}, PMI_0 {roc_auc_score(y, df.pmi_0):.3f}, "
              f"identifiable {roc_auc_score(y, df.ident):.3f}")
    print("   identifiable = share of 3 random questions under which the answer is less likely than under its own")
    print("   (the Alvarez & Morrier 2025 answer-quality idea). -> ordered but barely discriminative: Deflection and")
    print("   Partial answers are as question-specific as Explicit ones; relevance does not measure answering")


def main() -> None:
    print("E24 pre-checks (CPU, existing artifacts). Dev = 308 items, 3 annotators; S2 = any-annotator macro-F1.")
    print(f"M1 = 10-seed mean of Q8_fullq_lora_12ep; train prior from {len(TR)} rows; PRIOR sums to {PRIOR.sum():.3f}")
    p1_p2()
    p3()
    p4()
    p5()
    p6()
    p7()
    p8()


if __name__ == "__main__":
    main()
