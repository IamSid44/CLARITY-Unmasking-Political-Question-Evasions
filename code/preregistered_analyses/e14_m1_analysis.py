#!/usr/bin/env python3
"""E14 (module M1): the pre-registered analyses (docs/02_experiment_log.md, E14).

    python -m preregistered_analyses.e14_m1_analysis > ../docs/raw/E14_m1_analysis.txt

A. 12 epochs vs 3 epochs: paired per seed (H1) and as systems with a bootstrap (H2).
B. The cost lever: system S2 against the number of seeds K (H6).
C. Candidate sets C(x): coverage and size on the slice, the cross-fitted train rows and dev (H3).
D. The uncertainty score u(x): AUROC for M1's errors against 1 - top probability (H4).
E. Headroom for the LLM stage: S2 if the deferred items were decided by an oracle restricted to C(x),
   or at random within C(x), for deferral rates delta (H5).
Runs on whatever seeds have finished; the project's rule is 10 seeds before any system claim.
"""

from __future__ import annotations

import numpy as np

from cascade.candidates import build, fit_uncertainty, gold_rank, onehot
from preregistered_analyses.e13_analysis import GOLD, boot, finished, paired, probs, s1, s2, system
from qevasion.paths import RUNS
from qevasion.scoring import score_subtask2

NEW, OLD, CROSSFIT = "Q8_fullq_lora_12ep", "Q8_fullq_lora", "Q8_12ep_crossfit"
DELTAS = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)
SET_RULES = (("aps", 0.05, 3), ("aps", 0.10, 3), ("topk", 0.05, 3), ("topk", 0.10, 3),
             ("fixed", 0.0, 2), ("fixed", 0.0, 3), ("fixed", 0.0, 4))


def auroc(score: np.ndarray, err: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(err, score)) if 0 < err.sum() < len(err) else float("nan")


def section_a() -> None:
    print("A. 12 epochs (Q8_fullq_lora_12ep) vs 3 epochs (Q8_fullq_lora), same seeds -- dev\n")
    seeds = sorted(set(finished(NEW)) & set(finished(OLD)))
    print(f"   paired seeds: {seeds}")
    pn, po = probs(lambda s: NEW, seeds), probs(lambda s: OLD, seeds)
    n2, o2 = np.array([s2(p.argmax(1)) for p in pn]), np.array([s2(p.argmax(1)) for p in po])
    n1, o1 = np.array([s1(p.argmax(1)) for p in pn]), np.array([s1(p.argmax(1)) for p in po])
    print(f"   {'seed':>4} {'12ep S2':>8} {'3ep S2':>7} {'Δ':>7}   {'12ep S1':>8} {'3ep S1':>7} {'Δ':>7}")
    for i, s in enumerate(seeds):
        print(f"   {s:>4} {n2[i]:>8.3f} {o2[i]:>7.3f} {n2[i] - o2[i]:>+7.3f}   {n1[i]:>8.3f} {o1[i]:>7.3f} {n1[i] - o1[i]:>+7.3f}")
    print(f"   mean  12ep S2 {n2.mean():.3f} ± {n2.std(ddof=1):.3f}, S1 {n1.mean():.3f};  "
          f"3ep S2 {o2.mean():.3f} ± {o2.std(ddof=1):.3f}, S1 {o1.mean():.3f}")
    print(f"   paired S2: {paired(n2, o2)}")
    print(f"   paired S1: {paired(n1, o1)}")
    nl, ol = np.array([system(p)[0] for p in pn]), np.array([system(p)[0] for p in po])
    print(f"   single model + logit adjustment, S2: 12ep {nl.mean():.3f}, 3ep {ol.mean():.3f}; paired {paired(nl, ol)}")
    top = np.array([p.max(1).mean() for p in pn]), np.array([p.max(1).mean() for p in po])
    print(f"   mean top probability on dev: 12ep {top[0].mean():.3f}, 3ep {top[1].mean():.3f}")
    sn, prn = system(pn.mean(0)); sn10, _ = system(pn.mean(0), 10)
    so, pro = system(po.mean(0)); so10, _ = system(po.mean(0), 10)
    print(f"\n   {len(seeds)}-seed systems (ensemble + LA, nested CV), S2: 12ep {sn:.3f} (10 CV splits {sn10:.3f}), "
          f"3ep {so:.3f} ({so10:.3f});  S1 12ep {s1(prn):.3f}, 3ep {s1(pro):.3f}")
    print(f"   12ep - 3ep system, bootstrap over items: {boot(prn, pro)}")
    if len(seeds) < 10:
        print(f"   NOTE: {len(seeds)} seeds -- no system claim before 10")


def section_b() -> None:
    print("\nB. System S2 against the number of seeds K (20 random subsets per K, nested CV split 0)\n")
    seeds = finished(NEW)
    p = probs(lambda s: NEW, seeds)
    rng = np.random.default_rng(0)
    for k in range(1, len(seeds) + 1):
        vals = [system(p[rng.choice(len(seeds), k, replace=False)].mean(0))[0] for _ in range(20 if k < len(seeds) else 1)]
        print(f"   K={k:>2}: S2 {np.mean(vals):.3f} ± {np.std(vals):.3f}")


def section_c(results: dict) -> None:
    print("\nC. Candidate sets C(x): gold-label coverage and mean size (slice = calibration set)\n")
    print(f"   {'rule':<16} {'slice cov':>9} {'|C|':>5}   {'train OOF cov':>13} {'|C|':>5}   {'dev any-ref cov':>15} {'|C|':>5}")
    for name, r in results.items():
        y, vi = r["y"], r["val_idx"]
        c = r["cand"]
        cv = c["val"][np.arange(len(vi)), y[vi]].mean()
        cd = (c["dev"] & GOLD).any(1).mean()
        tr = "            --      --"
        if "train" in c:
            ok = ~np.isnan(r["pbar"]["train"]).any(1)
            tr = f"{c['train'][ok][np.arange(ok.sum()), y[ok]].mean():>13.3f} {c['train'][ok].sum(1).mean():>5.2f}"
        print(f"   {name:<16} {cv:>9.3f} {c['val'].sum(1).mean():>5.2f}   {tr}   {cd:>15.3f} {c['dev'].sum(1).mean():>5.2f}")
    r = next(iter(results.values()))
    m = r["meta"]
    print(f"\n   tau_C {m['tau_c']:.2f} and temperature {m['temperature']:.2f}, both fitted on the slice; seeds {m['seeds']}")
    rk = gold_rank(r["q"]["val"], onehot(r["y"][r["val_idx"]]))
    print("   slice gold-label rank distribution: " + ", ".join(f"{k}: {np.mean(rk == k):.3f}" for k in range(1, 10)))


def section_d(r: dict) -> None:
    print("\nD. Uncertainty u(x): AUROC for an error of the adjusted argmax\n")
    y, vi = r["y"], r["val_idx"]
    x, err = r["x"]["val"], (r["q"]["val"].argmax(1) != y[vi]).astype(int)
    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(len(vi)), 5)
    u_cv = np.empty(len(vi))
    for f in range(5):
        tr = np.concatenate([folds[j] for j in range(5) if j != f])
        u_cv[folds[f]] = fit_uncertainty(x[tr], err[tr]).predict_proba(x[folds[f]])[:, 1]
    err_dev = (~GOLD[np.arange(len(GOLD)), r["q"]["dev"].argmax(1)]).astype(int)
    print(f"   slice (5-fold CV inside the slice): logistic u {auroc(u_cv, err):.3f}, 1 - top prob {auroc(1 - x[:, 0], err):.3f}"
          f"   (error rate {err.mean():.3f})")
    print(f"   dev (u fitted on the whole slice):  logistic u {auroc(r['u']['dev'], err_dev):.3f}, "
          f"1 - top prob {auroc(1 - r['x']['dev'][:, 0], err_dev):.3f}   (error rate {err_dev.mean():.3f})")
    coef = dict(zip(r["meta"]["features"], np.round(r["meta"]["u_coef"], 3)))
    print(f"   coefficients (standardised features): {coef}")


def section_e(results: dict, base_pred: np.ndarray) -> None:
    print("\nE. Headroom for M3: dev S2 when the fraction delta with the highest u(x) is re-decided\n"
          "   oracle: a reference label inside C(x) if there is one (else M1's label); random: uniform in C(x)\n")
    print(f"   M1 system (ensemble + LA, nested CV) S2 {s2(base_pred):.3f}")
    rng = np.random.default_rng(0)
    for name, r in results.items():
        order = np.argsort(-r["u"]["dev"], kind="stable")
        cand, q = r["cand"]["dev"], r["q"]["dev"]
        row_o, row_r = [], []
        for d in DELTAS:
            idx = order[: int(round(d * len(order)))]
            o = base_pred.copy()
            for i in idx:
                ok = np.where(cand[i] & GOLD[i])[0]
                if len(ok):
                    o[i] = ok[np.argmax(q[i, ok])]
            row_o.append(score_subtask2(o, GOLD).macro_f1)
            vals = []
            for _ in range(20):
                rr = base_pred.copy()
                for i in idx:
                    rr[i] = rng.choice(np.where(cand[i])[0])
                vals.append(score_subtask2(rr, GOLD).macro_f1)
            row_r.append(np.mean(vals))
        print(f"   {name:<16} oracle " + "  ".join(f"δ={d:g}: {v:.3f}" for d, v in zip(DELTAS, row_o)))
        print(f"   {'':<16} random " + "  ".join(f"δ={d:g}: {v:.3f}" for d, v in zip(DELTAS, row_r)))


def main() -> None:
    folds = sorted(p.parent.name for p in RUNS.glob(f"{CROSSFIT}/fold*/metrics.json"))
    print(f"E14 (M1) analysis -- 12-epoch Qwen3-8B LoRA ({NEW}), seeds finished: {finished(NEW)}; "
          f"cross-fitted folds ({CROSSFIT}): {folds or 'none yet'}\n")
    section_a()
    section_b()
    has_cf = len(folds) == 5
    results = {f"{m} {'k=' + str(k) if m == 'fixed' else 'a=' + format(a, 'g')}":
               build(NEW, CROSSFIT if has_cf else None, a if a else 0.05, None, m, k) for m, a, k in SET_RULES}
    section_c(results)
    section_d(next(iter(results.values())))
    _, base = system(probs(lambda s: NEW, finished(NEW)).mean(0))
    section_e({k: v for k, v in results.items() if k in ("aps a=0.1", "topk a=0.1", "fixed k=3")}, base)


if __name__ == "__main__":
    main()
