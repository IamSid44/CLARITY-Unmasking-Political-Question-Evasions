#!/usr/bin/env python3
"""E29 (registered 2026-11-20, docs/experiment_log.md): M5, a question-side prior fused on the deferred items.

    python -m preregistered_analyses.e29_m5          # CPU, < 1 min; writes docs/raw/E29_m5.txt, runs/E29/m5.json

s in {0, 1/2, 1} = share of the two P-role LLMs (E25b) whose answer the frozen instrument labels evasive. On the frozen
instrument's deferred dev items (E25 replay: E25 M1 inputs and 27B records), the fused score gains -beta * (s - 1/2)
on Explicit; kept items and the frozen lam, tau, tau_m1 are unchanged. beta in {0, 0.1, ..., 2} is chosen on the
training folds (5 folds x 10 splits; ties -> smaller beta); the objective is S2 (primary) and, reported separately, S1.
beta = 0 is the frozen replay itself. Intervals: turn-cluster bootstrap of the 10-split mean difference.
"""

from __future__ import annotations

import json

import numpy as np

from audit.system import prior, records
from preregistered_analyses.e19_lib import folds_of
from preregistered_analyses.e25_analysis import BOOT, GOLD, TURN, lab, llm_table, rows_of
from preregistered_analyses.e27_cqc import scores
from qevasion.labels import N_EVASION
from qevasion.paths import ROOT, RUNS

RAW = ROOT / "docs" / "raw" / "E29_m5.txt"
OUT = RUNS / "E29" / "m5.json"
BETAS = np.round(np.arange(0, 2.01, 0.1), 2)
SPLITS = 10
EXPLICIT = 0


def inputs():
    f = json.loads((ROOT / "code" / "experiments" / "e25" / "frozen_system.json").read_text())
    own = lab("dev_own")
    look = records("dev_own")
    lp = np.log(own["pbar"] + 1e-12)
    lq = np.full((len(lp), N_EVASION), -np.inf)
    for i in np.where(own["defer"])[0]:
        q = look.get((int(i), tuple(int(c) for c in own["cand"][i])))
        if q is not None:
            lq[i, own["cand"][i]] = np.log(np.maximum(q, 1e-12))
    ok = own["defer"] & np.isfinite(lq).any(1)
    df = llm_table()
    s = df[df.condition == "P"].groupby("src_i")["evade"].mean().reindex(range(len(lp))).to_numpy()
    return f, own, lp, lq, ok, s


def predict(lp, lq, ok, s, f, beta: float) -> np.ndarray:
    logpi = np.log(prior())
    m1 = (lp - f["tau_m1"] * logpi).argmax(1)
    in_c = np.isfinite(lq)
    fused = f["lam"] * lp + (1 - f["lam"]) * np.where(in_c, lq, 0.0) - f["tau"] * logpi
    fused[:, EXPLICIT] -= beta * (s - 0.5)
    fused = np.where(in_c, fused, -np.inf)
    return np.where(ok, fused.argmax(1), m1)


def nested(P: np.ndarray, seed: int, objective: int) -> tuple[np.ndarray, list[float]]:
    n = P.shape[1]
    pooled, chosen = np.empty(n, np.int64), []
    folds = folds_of(n, seed)
    for k in range(len(folds)):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(len(folds)) if j != k])
        sc = scores(P, tr)[objective]
        b = int(np.flatnonzero(sc >= sc.max() - 1e-12)[0])
        pooled[te] = P[b, te]
        chosen.append(float(BETAS[b]))
    return pooled, chosen


def compute() -> tuple[list[str], dict]:
    f, own, lp, lq, ok, s = inputs()
    P = np.stack([predict(lp, lq, ok, s, f, b) for b in BETAS])
    assert (P[0] == own["pred"]).all(), "beta = 0 does not reproduce the frozen E25 replay"
    base = P[0]
    L = [f"E29 M5 question-side prior on deferred items (dev, n = {len(base)}; deferred and scored {int(ok.sum())}; "
         f"P-role share s: " + ", ".join(f"{v:.1f}: {int((s == v).sum())}" for v in sorted(set(s))) + ")",
         f"beta = 0 reproduces the frozen replay: S2 {scores(base[None])[0][0]:.3f}, S1 {scores(base[None])[1][0]:.3f}",
         f"in-sample S2 / S1 by fixed beta: " + "  ".join(f"{b:.1f}:{a:.3f}/{c:.3f}" for b, a, c in
                                                         zip(BETAS[::2], *[x[::2] for x in scores(P)])), ""]
    res = {"n_deferred_scored": int(ok.sum()), "base_s2": float(scores(base[None])[0][0]),
           "base_s1": float(scores(base[None])[1][0])}
    for obj, name in ((0, "S2"), (1, "S1")):
        runs = [nested(P, sp, obj) for sp in range(SPLITS)]
        new = np.stack([r[0] for r in runs])
        betas = [b for r in runs for b in r[1]]
        s2n, s1n = scores(new)
        d2, d1 = [], []
        for draw in BOOT:
            r = rows_of(draw, TURN)
            b2, b1 = scores(base[None], r)
            n2, n1 = scores(new, r)
            d2.append(n2.mean() - b2[0])
            d1.append(n1.mean() - b1[0])
        out = {"s2": float(s2n.mean()), "s1": float(s1n.mean()), "d_s2": float(s2n.mean() - res["base_s2"]),
               "d_s1": float(s1n.mean() - res["base_s1"]),
               "d_s2_ci": [float(np.percentile(d2, 2.5)), float(np.percentile(d2, 97.5))],
               "d_s1_ci": [float(np.percentile(d1, 2.5)), float(np.percentile(d1, 97.5))],
               "beta_mean": float(np.mean(betas)), "beta_share_zero": float(np.mean(np.array(betas) == 0))}
        res[f"objective_{name}"] = out
        L.append(f"beta chosen by training-fold {name}: S2 {res['base_s2']:.3f} -> {out['s2']:.3f}  delta {out['d_s2']:+.3f} "
                 f"[{out['d_s2_ci'][0]:+.3f}, {out['d_s2_ci'][1]:+.3f}];  S1 {res['base_s1']:.3f} -> {out['s1']:.3f}  delta "
                 f"{out['d_s1']:+.3f} [{out['d_s1_ci'][0]:+.3f}, {out['d_s1_ci'][1]:+.3f}];  beta mean {out['beta_mean']:.2f}, "
                 f"share 0 {out['beta_share_zero']:.2f}")
    prim = res["objective_S2"]
    held = prim["d_s2_ci"][0] > 0 or prim["d_s1_ci"][0] > 0
    sec = res["objective_S1"]
    L += ["", f"H29-1 (primary, beta chosen by S2: S2 or S1 rises, interval above 0): {'HELD' if held else 'FAILED'}",
          f"   (secondary, beta chosen by S1: {'interval above 0' if (sec['d_s1_ci'][0] > 0 or sec['d_s2_ci'][0] > 0) else 'no interval above 0'})",
          f"Decision: M5 {'ADOPTED' if held else 'not adopted (reported as a negative result)'}"]
    res["_verdict"] = {"h29_1": held}
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
