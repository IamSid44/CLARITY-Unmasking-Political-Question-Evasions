#!/usr/bin/env python3
"""E19-E (system part), E22c and the final operating points, with ONE copy of the 27B (2026-10-30).

    python -m preregistered_analyses.e23_systems > ../docs/raw/E23_systems.txt      # CPU, a few minutes

The faith gate failed (VM vs lab server: 0.925 argmax agreement on dev), so every system here uses the VM's records
(tag qwen38_27b_k0_vm, plus the shifted orders), and the lab-record numbers are shown only as the reference they were.
E19 arm B adopted permutation-averaged q (the mean over 4 cyclic option orders), so each system is shown with
single-order q and with averaged q. Procedures are E15-E18's: M1's own tau, fused (lam, tau) for deferred items,
5-fold nested CV on dev (split 0, and the mean of 10 splits); deferral thresholds from the train slice.
S1  accuracy point: 10-seed M1 + M3, delta = 0.3
S2  cost point: one student + M3, delta = 0.5 (mean over the 10 students)
S3  E19-E: the control students (M1-taught, uniform-taught; seeds 0-2) + M3 at delta = 0.5 against the M3-taught
    students on the same seeds; H-E (system part): the M3-taught student wins by >= 0.01
S4  E22c: the escalation ladder (student -> + K M1 seeds on the e1 least confident -> M3 on the e2 least confident);
    H22c: primary configuration (e1 0.4, K 5, e2 0.3) >= 0.625 S2 (10 splits, mean over students)
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from cascade.candidates import build
from cascade.frontier import top4
from cascade.pass2_jobs import fuse
from preregistered_analyses.e19_lib import Grid, defer_mask, lq_from, nested
from qevasion.labels import encode_clarity, leaf_to_official_clarity
from qevasion.loader import DATA_CACHE, dev_reference_mask, load_qevasion
from qevasion.paths import RUNS
from qevasion.scoring import score_subtask1, score_subtask2

SP = load_qevasion(DATA_CACHE)
GOLD = dev_reference_mask(SP.dev)
CLAR = encode_clarity(SP.dev["clarity_label"].tolist())
M1, STUDENT = "Q8_fullq_lora_12ep", "Q8_distill_a05"
CONTROLS = {"M1-taught": "Q8_distill_a05_m1", "uniform-taught": "Q8_distill_a05_unif"}
ORDERS = ("qwen38_27b_k0_vm", "qwen38_27b_k0_shift1", "qwen38_27b_k0_shift2", "qwen38_27b_k0_shift3")
COST = {"m1_seed": 0.052, "m3_call": 0.151}  # E19 Tier 0 (lab card), so the frontier stays on one scale
SPLITS = 10
rule = lambda t: print("=" * 110 + f"\n{t}\n" + "=" * 110)


def dev_lookup(tag: str) -> dict:
    out = {}
    p = RUNS / "M3" / tag / "responses.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r["frame"] == "dev":
                out[(r["i"], tuple(r["cand"]))] = np.asarray(r["q"])
    return out


LOOK = {"lab": dev_lookup("qwen38_27b_k0"), "single": dev_lookup(ORDERS[0])}
_ORD = [dev_lookup(t) for t in ORDERS]
LOOK["averaged"] = {k: np.mean([o[k] for o in _ORD], 0) for k in LOOK["single"] if all(k in o for o in _ORD)}


def calibrated(run: str, seeds) -> dict:
    return _calibrated(run, tuple(int(x) for x in seeds))


@lru_cache(maxsize=None)
def _calibrated(run: str, seeds: tuple[int, ...]) -> dict:
    r = build(run, None, 0.05, list(seeds), method="fixed", k=4)
    return {"pbar": r["pbar"]["dev"], "q_dev": r["q"]["dev"], "q_val": r["q"]["val"]}


def scores_of(pred: np.ndarray) -> tuple[float, float]:
    return (score_subtask2(pred, GOLD).macro_f1, score_subtask1(leaf_to_official_clarity(pred), CLAR).macro_f1)


def evaluate(lp, cand, defer, look) -> dict | None:
    """split-0 S2/S1 and the 10-split mean S2 of one system; None if M3 lacks a deferred item's candidate set."""
    lq, ok = lq_from(cand, look)
    if not ok[defer].all():
        return None
    lq[~defer] = -np.inf
    g = Grid(lp, lq, defer)
    preds = [nested(g, GOLD, s) for s in range(SPLITS)]
    s2s = [scores_of(p)[0] for p in preds]
    s2, s1 = scores_of(preds[0])
    return {"s2": s2, "s1": s1, "s2_10": float(np.mean(s2s)), "calls": float(defer.mean())}


def single_system(run: str, seeds, delta: float, look) -> dict | None:
    m = calibrated(run, seeds)
    u_dev, u_val = 1 - m["q_dev"].max(1), 1 - m["q_val"].max(1)
    return evaluate(np.log(m["pbar"] + 1e-12), top4(m["q_dev"]), defer_mask(u_dev, u_val, delta), look)


def fmt(r: dict | None) -> str:
    return "  (M3 records missing: waiting for pass 2)" if r is None else \
        f"S2 {r['s2']:.3f}  10 splits {r['s2_10']:.3f}  S1 {r['s1']:.3f}  calls {r['calls']:.3f}"


def s1_accuracy() -> None:
    rule("S1. Accuracy point: 10-seed M1 + M3, delta = 0.3")
    for name in ("lab", "single", "averaged"):
        print(f"   {name:9} {fmt(single_system(M1, range(10), 0.3, LOOK[name]))}")
    print("   (lab = the E15/E18 records, shown as the reference: S2 0.669 / 10 splits 0.660 / S1 0.798)")


def s2_cost() -> None:
    rule("S2. Cost point: one student + M3, delta = 0.5 (mean over the 10 students)")
    for name in ("lab", "single", "averaged"):
        rs = [single_system(STUDENT, (s,), 0.5, LOOK[name]) for s in range(10)]
        if any(r is None for r in rs):
            print(f"   {name:9} {fmt(None)}")
            continue
        m = {k: np.mean([r[k] for r in rs]) for k in rs[0]}
        print(f"   {name:9} {fmt(m)}  (sd over students of split-0 S2 {np.std([r['s2'] for r in rs], ddof=1):.3f})")


def s3_controls() -> None:
    rule("S3. E19-E, system part: students + M3 at delta = 0.5 (VM records, single order), seeds 0-2")
    res = {"M3-taught": [single_system(STUDENT, (s,), 0.5, LOOK["single"]) for s in range(3)]}
    for name, run in CONTROLS.items():
        res[name] = [single_system(run, (s,), 0.5, LOOK["single"]) for s in range(3)]
    for name, rs in res.items():
        if any(r is None for r in rs):
            print(f"   {name:15} (M3 records missing: waiting for pass 2)")
            return
        per_seed = " / ".join(f"{r['s2']:.3f}" for r in rs)
        print(f"   {name:15} S2 per seed {per_seed}; mean {np.mean([r['s2'] for r in rs]):.3f}; "
              f"10-split mean {np.mean([r['s2_10'] for r in rs]):.3f}; S1 {np.mean([r['s1'] for r in rs]):.3f}")
    base = np.array([r["s2_10"] for r in res["M3-taught"]])
    for name in CONTROLS:
        d = base - np.array([r["s2_10"] for r in res[name]])
        print(f"   M3-taught minus {name}: {' / '.join(f'{x:+.3f}' for x in d)}, mean {d.mean():+.3f} -> "
              f"{'H-E (system, >= +0.01) HELD' if d.mean() >= 0.01 else 'H-E (system) FAILED'}")


def ladder(i: int, K: int, e1: float, e2: float, subs: dict, look, rng=None) -> dict | None:
    s, m = calibrated(STUDENT, (i,)), calibrated(M1, subs[f"K{K}"][i])
    u0_dev, u0_val = 1 - s["q_dev"].max(1), 1 - s["q_val"].max(1)
    esc_dev = u0_dev >= np.quantile(u0_val, 1 - e1)
    esc_val = u0_val >= np.quantile(u0_val, 1 - e1)
    if rng is not None:  # random escalation at the same rates (registered baseline), within the scored items
        esc_dev = np.zeros(len(u0_dev), bool)
        pool = np.where(u0_dev >= np.quantile(u0_val, 1 - 0.6))[0]
        esc_dev[rng.choice(pool, min(len(pool), int(round(e1 * len(u0_dev)))), replace=False)] = True
    fq_dev, fq_val = fuse(s["q_dev"], m["q_dev"]), fuse(s["q_val"], m["q_val"])
    lp = np.where(esc_dev[:, None], 0.5 * np.log(s["pbar"] + 1e-12) + 0.5 * np.log(m["pbar"] + 1e-12),
                  np.log(s["pbar"] + 1e-12))
    cand = np.where(esc_dev[:, None], top4(fq_dev), top4(s["q_dev"]))
    u1_val = np.where(esc_val, 1 - fq_val.max(1), -1.0)  # u is in [0, 1]: -1 marks items stage 1 keeps
    thr = np.quantile(u1_val, 1 - e2)
    defer = esc_dev & ((1 - fq_dev.max(1)) >= thr)
    if rng is not None:
        n_def = int(defer.sum())
        defer = np.zeros(len(defer), bool)
        defer[rng.choice(np.where(esc_dev)[0], min(n_def, int(esc_dev.sum())), replace=False)] = True
    r = evaluate(lp, cand, defer, look)
    if r is not None:
        r["gpu_s"] = COST["m1_seed"] * (1 + K * esc_dev.mean()) + COST["m3_call"] * defer.mean()
        r["e1_real"] = float(esc_dev.mean())
    return r


def s4_ladder() -> None:
    rule("S4. E22c: escalation ladder (VM records, single order); cost on the E19 lab scale")
    subs = json.loads((RUNS / "E22" / "pass2_subsets.json").read_text())
    if ladder(0, 5, 0.4, 0.3, subs, LOOK["single"]) is None:
        print("   (M3 records missing: waiting for pass 2)")
        return
    prim = [ladder(i, 5, 0.4, 0.3, subs, LOOK["single"]) for i in range(10)]
    if any(r is None for r in prim):
        print("   (M3 records missing: waiting for pass 2)")
        return
    m = {k: np.mean([r[k] for r in prim]) for k in prim[0]}
    print(f"   primary (e1 0.4, K 5, e2 0.3), mean over 10 students: {fmt(m)}  GPU-s {m['gpu_s']:.3f}  e1 realised {m['e1_real']:.2f}")
    print(f"   H22c part 1 (10-split S2 >= 0.625): {'HELD' if m['s2_10'] >= 0.625 else 'FAILED'}")
    rng = np.random.default_rng(22)
    rnd = []
    for d in range(200):
        r = ladder(d % 10, 5, 0.4, 0.3, subs, LOOK["single"], rng=rng)
        if r is not None:
            rnd.append(r["s2_10"])
    gap = m["s2_10"] - np.mean(rnd)
    print(f"   random escalation at the same rates ({len(rnd)} draws, within the items scored for e1 = 0.6): "
          f"10-split S2 {np.mean(rnd):.3f} +- {np.std(rnd):.3f}; primary minus random {gap:+.3f} -> "
          f"H22c part 2 (>= +0.01) {'HELD' if gap >= 0.01 else 'FAILED'}")
    print("   grid (mean over students): e1, K, e2 -> 10-split S2, GPU-s")
    for e1 in (0.2, 0.4, 0.6):
        for K in (3, 5):
            for e2 in (0.2, 0.3, 0.5):
                if e2 > e1:
                    continue
                rs = [ladder(i, K, e1, e2, subs, LOOK["single"]) for i in range(10)]
                if any(r is None for r in rs):
                    continue
                print(f"     {e1:.1f} {K} {e2:.1f} -> {np.mean([r['s2_10'] for r in rs]):.3f}  {np.mean([r['gpu_s'] for r in rs]):.3f}")


def main() -> None:
    print(f"E23 systems on dev, one 27B copy (VM). Records: single {len(LOOK['single'])}, averaged {len(LOOK['averaged'])}, "
          f"lab {len(LOOK['lab'])} (dev item, candidate set) pairs\n")
    for f in (s1_accuracy, s2_cost, s3_controls, s4_ladder):
        f()
        print()


if __name__ == "__main__":
    main()
