#!/usr/bin/env python3
"""E27 (registered 2026-11-20, docs/experiment_log.md): counterfactual question contrast (CQC).

    python -m preregistered_analyses.e27_cqc        # CPU, a few minutes; writes docs/raw/E27_cqc.txt, runs/E27/cqc.json

M1's 10-seed mean on the E25 frames (one GPU, one run): s_alpha = log p_own - alpha * log p_ref, with p_ref the
no-question frame (dev_null; variant Q: the constant-answer frame dev_qonly), then M1's logit adjustment. alpha and tau
are chosen on the training folds of every outer fold (5 folds x 10 splits; ties go to the smaller alpha); alpha = 0
under the same protocol is the baseline. Cascade arm: softmax(s_alpha) replaces p-bar in the frozen calibration ->
q, top-4, u; the least-sure 35.7% of dev (the frozen deferral rate) is deferred; 27B records from the E23 VM
order-averaged set and the E25 dev_own set; deferred items without a record keep the CQC M1 label (lower bound).
Intervals: cluster bootstrap over the 191 dev turns (2,000 draws) of the 10-split mean difference.
"""

from __future__ import annotations

import json

import numpy as np

from audit.system import calibration, records, top4
from preregistered_analyses.e19_lib import Grid, folds_of, macro_f1_many
from preregistered_analyses.e24_prechecks import sibling_pairs
from preregistered_analyses.e25_analysis import BOOT, DEV, GOLD, GOLD_CLAR, TURN, rows_of
from qevasion.labels import N_EVASION, leaf_to_official_clarity
from qevasion.paths import ROOT, RUNS

RAW = ROOT / "docs" / "raw" / "E27_cqc.txt"
OUT = RUNS / "E27" / "cqc.json"
ALPHAS = np.round(np.arange(0, 1.01, 0.1), 2)
SPLITS = 10
DEFER_RATE = 0.357  # frozen_system.json dev_calls
PAIRS = np.array(sibling_pairs(DEV))
DISJ = PAIRS[[not (GOLD[i] & GOLD[j]).any() for i, j in PAIRS]]


def pbar(frame: str) -> np.ndarray:
    return np.stack([np.load(RUNS / "E25" / "m1" / frame / f"seed{k}.npy") for k in range(10)]).mean(0)


def log_softmax(s: np.ndarray) -> np.ndarray:
    s = s - s.max(1, keepdims=True)
    return s - np.log(np.exp(s).sum(1, keepdims=True))


def nested_select(grids: list[Grid], seed: int, m1_only: bool) -> tuple[np.ndarray, list[int]]:
    """Pooled held-out predictions; on each fold the grid (alpha) with the best training-fold S2 is used."""
    n = len(GOLD)
    pooled, chosen = np.empty(n, np.int64), []
    folds = folds_of(n, seed)
    for f in range(len(folds)):
        te = folds[f]
        tr = np.concatenate([folds[j] for j in range(len(folds)) if j != f])
        best, best_s, best_a = None, -1.0, -1
        for a, g in enumerate(grids):
            pred = g.fit_m1(GOLD, tr)[0] if m1_only else g.fit(GOLD, tr)[0]
            s = macro_f1_many(pred[tr][None], GOLD[tr])[0]
            if s > best_s + 1e-12:
                best, best_s, best_a = pred, s, a
        pooled[te] = best[te]
        chosen.append(best_a)
    return pooled, chosen


def scores(P: np.ndarray, rows=None) -> tuple[np.ndarray, np.ndarray]:
    """S2 and S1 of each row of P (k, n), optionally on bootstrap rows."""
    r = np.arange(P.shape[1]) if rows is None else rows
    s2 = macro_f1_many(P[:, r], GOLD[r])
    C = leaf_to_official_clarity(P.reshape(-1)).reshape(P.shape)
    s1 = []
    for c in C:
        y, t = c[r], GOLD_CLAR[r]
        f = []
        for k in range(3):
            tp = np.sum((y == k) & (t == k)); fp = np.sum((y == k) & (t != k)); fn = np.sum((y != k) & (t == k))
            f.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
        s1.append(np.mean(f))
    return s2, np.array(s1)


def compare(base: list[np.ndarray], new: list[np.ndarray]) -> dict:
    P = np.stack(base + new)
    s2, s1 = scores(P)
    k = len(base)
    d2, d1 = [], []
    for draw in BOOT:
        r = rows_of(draw, TURN)
        b2, b1 = scores(P, r)
        d2.append(b2[k:].mean() - b2[:k].mean())
        d1.append(b1[k:].mean() - b1[:k].mean())
    return {"base_s2": float(s2[:k].mean()), "new_s2": float(s2[k:].mean()), "base_s1": float(s1[:k].mean()),
            "new_s1": float(s1[k:].mean()), "d_s2": float(s2[k:].mean() - s2[:k].mean()),
            "d_s2_ci": [float(np.percentile(d2, 2.5)), float(np.percentile(d2, 97.5))],
            "d_s1": float(s1[k:].mean() - s1[:k].mean()),
            "d_s1_ci": [float(np.percentile(d1, 2.5)), float(np.percentile(d1, 97.5))]}


def same_disjoint(p: np.ndarray) -> float:
    return float(np.mean([p[i] == p[j] for i, j in DISJ]))


def compute() -> tuple[list[str], dict]:
    own = pbar("dev_own")
    calib, *_ = calibration()
    look = {}
    from preregistered_analyses.e23_systems import LOOK
    look.update(LOOK["averaged"])
    for k, v in records("dev_own").items():
        look.setdefault(k, v)
    L = [f"E27 counterfactual question contrast (dev, n = {len(GOLD)}; M1 = 10-seed mean on the E25 frames; "
         f"{SPLITS} nested-CV splits; {len(BOOT)} turn-bootstrap draws; {len(DISJ)} disjoint sibling pairs)", ""]
    res = {}
    for ref in ("dev_null", "dev_qonly"):
        lref = np.log(pbar(ref) + 1e-12)
        S = [np.log(own + 1e-12) - a * lref for a in ALPHAS]
        # ---- M1 level
        g_m1 = [Grid(log_softmax(s), np.full((len(s), N_EVASION), -np.inf), np.zeros(len(s), bool)) for s in S]
        base = [nested_select(g_m1[:1], sp, True)[0] for sp in range(SPLITS)]
        runs = [nested_select(g_m1, sp, True) for sp in range(SPLITS)]
        new = [r[0] for r in runs]
        al = [ALPHAS[a] for r in runs for a in r[1]]
        c = compare(base, new)
        c.update({"alpha_chosen_mean": float(np.mean(al)), "alpha_chosen_share_zero": float(np.mean(np.array(al) == 0)),
                  "same_disjoint_base": same_disjoint(base[0]), "same_disjoint_new": same_disjoint(new[0])})
        res[f"m1_{ref}"] = c
        L += [f"[{ref}] M1 level: S2 {c['base_s2']:.3f} -> {c['new_s2']:.3f}  delta {c['d_s2']:+.3f} "
              f"[{c['d_s2_ci'][0]:+.3f}, {c['d_s2_ci'][1]:+.3f}];  S1 {c['base_s1']:.3f} -> {c['new_s1']:.3f}  delta "
              f"{c['d_s1']:+.3f} [{c['d_s1_ci'][0]:+.3f}, {c['d_s1_ci'][1]:+.3f}]",
              f"      alpha chosen per fold: mean {c['alpha_chosen_mean']:.2f}, share alpha = 0 {c['alpha_chosen_share_zero']:.2f}; "
              f"same label on disjoint sibling pairs {c['same_disjoint_base']:.3f} -> {c['same_disjoint_new']:.3f}"]
        fixed = {}
        for a, g in zip(ALPHAS, g_m1):
            fixed[float(a)] = float(scores(np.stack([nested_select([g], sp, True)[0] for sp in range(SPLITS)]))[0].mean())
        res[f"m1_{ref}"]["s2_by_fixed_alpha"] = fixed
        L.append("      S2 at fixed alpha (tau nested): " + "  ".join(f"{a:.1f}:{v:.3f}" for a, v in fixed.items()))
        # ---- cascade level
        g_c, cov = [], []
        for s in S:
            lp = log_softmax(s)
            q = calib(np.exp(lp))
            cand = top4(q)
            u = 1 - q.max(1)
            defer = u >= np.quantile(u, 1 - DEFER_RATE)
            lq = np.full((len(lp), N_EVASION), -np.inf)
            hit = np.zeros(len(lp), bool)
            for i in np.where(defer)[0]:
                v = look.get((int(i), tuple(int(x) for x in cand[i])))
                if v is not None:
                    lq[i, cand[i]] = np.log(np.maximum(np.asarray(v), 1e-12))
                    hit[i] = True
            cov.append(float(hit[defer].mean()))
            g_c.append(Grid(lp, lq, defer & hit))
        base = [nested_select(g_c[:1], sp, False)[0] for sp in range(SPLITS)]
        runs = [nested_select(g_c, sp, False) for sp in range(SPLITS)]
        new = [r[0] for r in runs]
        al = [ALPHAS[a] for r in runs for a in r[1]]
        c = compare(base, new)
        c.update({"alpha_chosen_mean": float(np.mean(al)), "alpha_chosen_share_zero": float(np.mean(np.array(al) == 0)),
                  "record_coverage_by_alpha": dict(zip(map(float, ALPHAS), cov)),
                  "same_disjoint_base": same_disjoint(base[0]), "same_disjoint_new": same_disjoint(new[0])})
        res[f"cascade_{ref}"] = c
        L += [f"[{ref}] cascade: S2 {c['base_s2']:.3f} -> {c['new_s2']:.3f}  delta {c['d_s2']:+.3f} "
              f"[{c['d_s2_ci'][0]:+.3f}, {c['d_s2_ci'][1]:+.3f}];  S1 {c['base_s1']:.3f} -> {c['new_s1']:.3f}  delta "
              f"{c['d_s1']:+.3f} [{c['d_s1_ci'][0]:+.3f}, {c['d_s1_ci'][1]:+.3f}]",
              f"      alpha chosen: mean {c['alpha_chosen_mean']:.2f}, share 0 {c['alpha_chosen_share_zero']:.2f}; 27B record "
              f"coverage of deferred items: alpha 0 {cov[0]:.3f}, min {min(cov):.3f}; same label on disjoint pairs "
              f"{c['same_disjoint_base']:.3f} -> {c['same_disjoint_new']:.3f}", ""]
    # verdicts
    def ok(c):
        return c["d_s2"] >= 0.010 and c["d_s2_ci"][0] > 0
    h1 = any(ok(res[k]) for k in res)
    h2 = any((res[k]["same_disjoint_base"] - res[k]["same_disjoint_new"]) >= 0.05 and res[k]["d_s2"] >= 0 for k in res)
    L.append(f"H27-1 (S2 >= +0.010, interval above 0, in any arm): {'HELD' if h1 else 'FAILED'}")
    L.append(f"H27-2 (same-label rate on disjoint sibling pairs down >= 5 points, no S2 loss): {'HELD' if h2 else 'FAILED'}")
    L.append(f"Decision: CQC {'ADOPTED as module M0' if (h1 or h2) else 'not adopted (reported as a negative result)'}")
    res["_verdict"] = {"h1": h1, "h2": h2}
    return L, res


def main() -> None:
    L, res = compute()
    print("\n".join(L))
    RAW.write_text("\n".join(L) + "\n", encoding="utf8")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {RAW} and {OUT}")


if __name__ == "__main__":
    main()
