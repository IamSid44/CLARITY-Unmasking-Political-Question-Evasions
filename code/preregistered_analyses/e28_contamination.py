#!/usr/bin/env python3
"""E28 (registered 2026-11-20, docs/experiment_log.md): sibling contamination -- do systems give a sub-question the
label its sibling (another sub-question answered by the same answer) deserves?

    python -m preregistered_analyses.e28_contamination        # CPU, ~1 min; writes docs/raw/E28_contamination.txt
                                                              # and runs/E28/contamination.json

On dev items whose answer is shared (siblings), among the system's errors (any-annotator rule), contamination is the
share whose wrong label is in a sibling's reference set and not in the item's own. Nulls: (primary) 1,000 permutations
replacing an item's siblings by as many items from OTHER answers in the same answer-length tercile; (secondary) items
from other answers of the same interview (url); (reference) the uniform chance of sibling_error_check.py. Intervals:
bootstrap over the shared answers (2,000 draws). Systems: DeBERTa-v3-large (E10+E11 fullq, 10 seeds), M1 (10 seeds),
the M4 student (10 seeds), each with logit adjustment fitted by nested CV (split 0); the 27B alone over M1's top-4
(E23 order-averaged VM records); the frozen cascade (E23 inputs, full-dev fit = the frozen instrument) and its E25
replay (the preliminary check's numbers, as an anchor).
"""

from __future__ import annotations

import json

import numpy as np

from preregistered_analyses.e19_lib import Grid, nested
from qevasion.loader import DATA_CACHE, dev_reference_mask, load_qevasion
from qevasion.paths import ROOT, RUNS
from qevasion.scoring import score_subtask2

RAW = ROOT / "docs" / "raw" / "E28_contamination.txt"
OUT = RUNS / "E28" / "contamination.json"
NPERM, NBOOT = 1000, 2000

DV = load_qevasion(DATA_CACHE).dev.reset_index(drop=True)
GOLD = dev_reference_mask(DV)
ANS = DV["interview_answer"].astype(str).str.split().str.join(" ").factorize()[0]
URL = DV["url"].factorize()[0]
NW = DV["interview_answer"].astype(str).str.split().str.len().to_numpy()
TERC = np.digitize(NW, np.quantile(NW, [1 / 3, 2 / 3]))
GROUPS = {a: np.where(ANS == a)[0] for a in np.unique(ANS)}
SHARED = np.array([len(GROUPS[a]) > 1 for a in ANS])


def probs(run: str, seeds) -> np.ndarray:
    return np.stack([np.load(RUNS / run / f"seed{s}" / "dev_probs.npy") for s in seeds]).mean(0)


def nested_m1(p: np.ndarray) -> np.ndarray:
    lp = np.log(p + 1e-12)
    g = Grid(lp, np.full_like(lp, -np.inf), np.zeros(len(lp), bool))
    return nested(g, GOLD, 0, m1_only=True)


def systems() -> dict[str, np.ndarray]:
    from audit.system import calibration, decide, top4
    from preregistered_analyses.e23_systems import LOOK

    deb = probs("E10_fullq_16ep", range(5)) * 5 + probs("E11_fullq_16ep", range(5, 10)) * 5
    out = {"DeBERTa-v3-large (10 seeds)": nested_m1(deb / 10)}
    _, _, _, r = calibration()
    out["M1 (10 seeds)"] = nested_m1(r["pbar"]["dev"])
    out["M4 student (10 seeds)"] = nested_m1(probs("Q8_distill_a05", range(10)))
    cand = top4(r["q"]["dev"])
    q27 = np.full((len(cand), 9), -np.inf)
    for i, c in enumerate(cand):
        q27[i, c] = LOOK["averaged"][(i, tuple(int(x) for x in c))]
    out["27B alone over M1's top-4"] = q27.argmax(1)
    f = json.loads((ROOT / "code" / "experiments" / "e25" / "frozen_system.json").read_text())
    defer = (1 - r["q"]["dev"].max(1)) >= f["u_threshold"]
    lq = np.full_like(q27, -np.inf)
    for i in np.where(defer)[0]:
        lq[i, cand[i]] = np.log(np.maximum(q27[i, cand[i]], 1e-12))
    out["frozen cascade (E23 instrument)"] = decide(np.log(r["pbar"]["dev"] + 1e-12), lq, defer, f)
    out["frozen cascade, E25 replay (anchor)"] = np.load(RUNS / "E25" / "labels" / "dev_own.npz")["pred"]
    return out


def leaks(p: np.ndarray, sib_of) -> tuple[np.ndarray, np.ndarray]:
    """(error item indices on shared answers, 0/1 leak per error) given a function i -> sibling indices."""
    ok = GOLD[np.arange(len(p)), p]
    err = np.where(SHARED & ~ok)[0]
    lk = np.array([bool((GOLD[sib_of(i)].any(0) & ~GOLD[i])[p[i]]) for i in err], int)
    return err, lk


def real_sibs(i: int) -> np.ndarray:
    g = GROUPS[ANS[i]]
    return g[g != i]


def uniform_chance(p: np.ndarray, err: np.ndarray) -> float:
    return float(np.mean([(GOLD[real_sibs(i)].any(0) & ~GOLD[i])[~GOLD[i]].mean() for i in err]))


def perm_null(p: np.ndarray, err: np.ndarray, rng, same_url: bool) -> tuple[np.ndarray, int]:
    rates, used = [], 0
    pools = {}
    for i in err:
        if same_url:
            pool = np.where((URL == URL[i]) & (ANS != ANS[i]))[0]
        else:
            pool = np.where((TERC == TERC[i]) & (ANS != ANS[i]))[0]
        pools[i] = pool
    keep = [i for i in err if len(pools[i])]
    used = len(keep)
    for _ in range(NPERM):
        hits = 0
        for i in keep:
            k = len(real_sibs(i))
            s = rng.choice(pools[i], k, replace=len(pools[i]) < k)
            hits += bool((GOLD[s].any(0) & ~GOLD[i])[p[i]])
        rates.append(hits / max(used, 1))
    return np.array(rates), used


def boot_over_answers(vals_by_item: dict[str, tuple[np.ndarray, np.ndarray]], rng) -> dict:
    """Bootstrap (over shared answers) of each system's contamination rate and of cascade - M1."""
    shared_ans = np.unique(ANS[SHARED])
    draws = [rng.choice(shared_ans, len(shared_ans)) for _ in range(NBOOT)]
    out = {}
    for name, (err, lk) in vals_by_item.items():
        a_of = ANS[err]
        by = {a: lk[a_of == a] for a in shared_ans}
        rs = []
        for d in draws:
            x = np.concatenate([by[a] for a in d])
            rs.append(x.mean() if len(x) else np.nan)
        out[name] = np.array(rs)
    return out


def compute() -> tuple[list[str], dict]:
    S = systems()
    rng = np.random.default_rng(28)
    L = [f"E28 sibling contamination (dev; {SHARED.sum()} items share their answer, {len(np.unique(ANS[SHARED]))} shared "
         f"answers; nulls: {NPERM} permutations; intervals: {NBOOT} bootstrap draws over shared answers)", "",
         "contamination = P(wrong label is a sibling's reference label, not the item's own | error on a shared answer)", ""]
    hdr = (f"{'system':<40}{'S2':>6}{'errors':>8}{'contam.':>9}{'95% CI':>17}{'uniform':>9}{'perm.':>7}{'p':>7}"
           f"{'same-url':>10}{'p':>7}")
    L.append(hdr)
    res, byitem = {}, {}
    for name, p in S.items():
        err, lk = leaks(p, real_sibs)
        byitem[name] = (err, lk)
        nl, _ = perm_null(p, err, rng, same_url=False)
        nu, used_u = perm_null(p, err, rng, same_url=True)
        rate = lk.mean()
        res[name] = {"s2": score_subtask2(p, GOLD).macro_f1, "errors": int(len(err)), "leaks": int(lk.sum()),
                     "rate": float(rate), "uniform": uniform_chance(p, err),
                     "perm_mean": float(nl.mean()), "perm_p": float((1 + (nl >= rate).sum()) / (1 + NPERM)),
                     "url_mean": float(nu.mean()), "url_p": float((1 + (nu >= rate).sum()) / (1 + NPERM)),
                     "url_items": used_u}
    B = boot_over_answers(byitem, np.random.default_rng(280))
    for name, r in res.items():
        lo, hi = np.nanpercentile(B[name], [2.5, 97.5])
        r["ci"] = [float(lo), float(hi)]
        L.append(f"{name:<40}{r['s2']:>6.3f}{r['errors']:>8}{r['rate']:>9.3f}{f'[{lo:.3f}, {hi:.3f}]':>17}"
                 f"{r['uniform']:>9.3f}{r['perm_mean']:>7.3f}{r['perm_p']:>7.3f}{r['url_mean']:>10.3f}{r['url_p']:>7.3f}")
    L += ["", "   perm. = mean of the length-tercile permutation null; same-url = siblings replaced by other answers of the "
          "same interview", ""]
    # H28-1 / H28-2
    main = [k for k in res if "anchor" not in k]
    h1 = all(res[k]["perm_p"] < 0.05 for k in main)
    L.append(f"H28-1 (above the permutation null for every system, p < 0.05): {'HELD' if h1 else 'FAILED'}  ("
             + ", ".join(f"{k.split(' (')[0]} p {res[k]['perm_p']:.3f}" for k in main) + ")")
    casc, m1 = "frozen cascade (E23 instrument)", "M1 (10 seeds)"
    d = B[casc] - B[m1]
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    top = max(main, key=lambda k: res[k]["rate"])
    h2 = top == casc and lo > 0
    L.append(f"H28-2 (highest for the cascade, cascade - M1 interval above 0): cascade - M1 "
             f"{res[casc]['rate'] - res[m1]['rate']:+.3f} [{lo:+.3f}, {hi:+.3f}]; highest: {top} -> {'HELD' if h2 else 'FAILED'}")
    res["_cascade_minus_m1"] = {"diff": float(res[casc]["rate"] - res[m1]["rate"]), "ci": [float(lo), float(hi)]}
    res["_h28"] = {"h1": h1, "h2": h2}
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
