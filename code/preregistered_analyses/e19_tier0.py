#!/usr/bin/env python3
"""E19, Tier 0: CPU-only checks of the E18 headlines, from saved probabilities and M3 responses only.

    python -m preregistered_analyses.e19_tier0 > ../docs/raw/E19_tier0.txt          # CPU, ~7 min
    python -m preregistered_analyses.e19_tier0 --sections A,E                       # a subset

A. Headline re-statement: split 0 and 10 splits (mean ± sd) for S2 and S1; Δ vs M1; a paired bootstrap that
   refits every parameter (M1's tau, the fused lam and tau) inside each replicate.
B. Cost: M3's GPU-s per call from the one-pass dev scoring (frame = split = dev) instead of cost.json's mean
   over all dev-frame records; the E18 cost table recomputed; tokens; training cost and amortisation.
C. The cascade on the cross-fitted train rows (n = 3,448, one gold label), 5-fold CV inside train.
D. Value map: where M3 fixes or breaks M1, by M1-uncertainty quintile (train and dev), and the crossover.
E. Routing and fusion ablations on dev: confidence vs random vs other signals vs oracle routing; lam = 0 / 1.
F. Strata on dev: interview seen in train, answer length, annotator agreement.
G. M3's position and label bias.
H. Test-set predictions (evaluation.make_cascade_submission): M1, accuracy point, cost point (if M3 has the
   records it needs).
I. Human ceiling in the 2-annotator regime.

Same procedures as E15-E18 (cascade.fuse): M1's own tau, fused (lam, tau) for deferred items, 5-fold nested CV
on dev; deferral threshold from the train slice. Every random draw has a fixed seed.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict

import numpy as np

from cascade.frontier import top4
from cascade.fuse import DELTAS, LAMS, TAUS, Y, fit_system, sp
from cascade.prompts import LETTERS
from decision_rules.decide import nested_cv, reference_masks
from preregistered_analyses.e19_lib import (M1, STUDENT, Grid, boot_nested, defer_mask, folds_of, lookup,
                                            m1_npz, m3_records, macro_f1_many, nested, system_inputs)
from qevasion.labels import EVASION_LABELS, N_EVASION, encode_clarity, leaf_to_official_clarity
from qevasion.loader import consensus_level, dev_reference_mask
from qevasion.paths import RUNS, TEST_CSV
from qevasion.scoring import score_subtask1, score_subtask2

GOLD = dev_reference_mask(sp.dev)
CLAR = encode_clarity(sp.dev["clarity_label"].tolist())
N = len(GOLD)
IDX = np.arange(N)
COST = json.loads((RUNS / "M3" / "cost.json").read_text())
N_BOOT = 2000
SHORT = [n[:11] for n in EVASION_LABELS]
_CACHE: dict = {}


def s2(p, g=GOLD) -> float:
    return score_subtask2(p, g).macro_f1


def s1(p, c=CLAR) -> float:
    return score_subtask1(leaf_to_official_clarity(p), c).macro_f1


def inset(p, g=GOLD) -> float:
    return float(g[np.arange(len(g)), p].mean())


def ci(x: np.ndarray, d: int = 3) -> str:
    return f"{np.mean(x):+.{d}f} [{np.percentile(x, 2.5):+.{d}f}, {np.percentile(x, 97.5):+.{d}f}]"


def head(title: str) -> None:
    print(f"\n{'=' * 110}\n{title}\n{'=' * 110}\n", flush=True)


def inputs(run: str, seeds: tuple[int, ...]) -> dict:
    key = (run, seeds)
    if key not in _CACHE:
        _CACHE[key] = system_inputs(run, seeds)
    return _CACHE[key]


def grid(run: str, seeds: tuple[int, ...], delta: float, **kw) -> Grid:
    m = inputs(run, seeds)
    return Grid(m["lp_dev"], m["lq_dev"], defer_mask(m["u_dev"], m["u_val"], delta), **kw)


TEN = tuple(range(10))


# ====================================================================================================== A

def section_a(boot: bool = True) -> dict:
    head("A. Headline re-statement (dev, 308 items): split 0, 10 nested-CV splits, and a bootstrap that refits")
    # the vectorised refit reproduces cascade.fuse.fit_system exactly (checked here on split 0)
    m = inputs(M1, TEN)
    d03 = defer_mask(m["u_dev"], m["u_val"], 0.3)

    def rule(_p, g, tr, te):
        pred, fit, *_ = fit_system(m["lp_dev"], m["lq_dev"], m["pbar_dev"], d03, g, tr)
        return pred[te], fit

    ref = nested_cv(rule, m["lp_dev"], GOLD, 5, 0)["pooled_predictions"]
    acc_g = grid(M1, TEN, 0.3)
    assert np.array_equal(ref, nested(acc_g, GOLD, 0)), "e19_lib.Grid does not reproduce fit_system"
    print("   check: e19_lib's vectorised refit reproduces cascade.fuse.fit_system's split-0 predictions exactly\n")

    m1_g = grid(M1, TEN, 0.0)
    st_g = [grid(STUDENT, (s,), 0.5) for s in range(10)]
    sa_g = [grid(STUDENT, (s,), 0.0) for s in range(10)]

    def splits(g, m1_only=False):
        preds = [nested(g, GOLD, cs, m1_only=m1_only) for cs in range(10)]
        return preds, np.array([s2(p) for p in preds]), np.array([s1(p) for p in preds])

    res = {"M1": splits(m1_g, True), "acc": splits(acc_g)}
    res["student"] = [splits(g, True) for g in sa_g]
    res["cost"] = [splits(g) for g in st_g]
    calls = {"M1": 0.0, "acc": float(d03.mean()),
             "student": 0.0, "cost": float(np.mean([g.defer.mean() for g in st_g]))}

    print("   1. Point estimates. 'split 0' = the E18 numbers; '10 splits' = mean ± sd over nested-CV splits 0-9;\n"
          "      students: mean over the 10 students of each quantity (± sd over students in brackets for split 0).\n")
    print(f"   {'system':<36}{'S2 sp0':>8}{'S2 10 splits':>16}{'S1 sp0':>8}{'S1 10 splits':>16}{'in-set':>8}{'calls':>7}")
    rows = {}
    for key, name in (("M1", "M1, 10 seeds"), ("acc", "accuracy point: M1 + M3, δ = 0.3"),
                      ("student", "student alone (10 students)"), ("cost", "cost point: student + M3, δ = 0.5")):
        r = res[key]
        if isinstance(r, list):
            a2 = np.array([x[1] for x in r])  # (students, splits)
            a1 = np.array([x[2] for x in r])
            ins = np.mean([inset(x[0][0]) for x in r])
            sp0 = f"{a2[:, 0].mean():.3f}"
            tail = f" (± {a2[:, 0].std(ddof=1):.3f} over students)"
            rows[key] = (a2[:, 0].mean(), a2.mean(), a1[:, 0].mean(), a1.mean())
            print(f"   {name:<36}{sp0:>8}{a2.mean():>9.3f} ± {a2.std(1, ddof=1).mean():.3f}{a1[:, 0].mean():>8.3f}"
                  f"{a1.mean():>9.3f} ± {a1.std(1, ddof=1).mean():.3f}{ins:>8.3f}{calls[key]:>7.3f}{tail}")
        else:
            _, a2, a1 = r
            rows[key] = (a2[0], a2.mean(), a1[0], a1.mean())
            print(f"   {name:<36}{a2[0]:>8.3f}{a2.mean():>9.3f} ± {a2.std(ddof=1):.3f}{a1[0]:>8.3f}"
                  f"{a1.mean():>9.3f} ± {a1.std(ddof=1):.3f}{inset(r[0][0]):>8.3f}{calls[key]:>7.3f}")
    print("\n   Δ vs M1 (point differences)        S2 sp0   S2 10sp   S1 sp0   S1 10sp")
    for key, name in (("acc", "accuracy point"), ("student", "student alone"), ("cost", "cost point")):
        d = np.array(rows[key]) - np.array(rows["M1"])
        print(f"   {name:<34}{d[0]:>+8.3f}{d[1]:>+10.3f}{d[2]:>+9.3f}{d[3]:>+10.3f}")
    print("   (the sd over splits is the spread from the CV fold assignment alone: same data, same models)")

    out = {"m1_pred": res["M1"][0][0], "acc_pred": res["acc"][0][0], "cost_preds": [r[0][0] for r in res["cost"]],
           "student_preds": [r[0][0] for r in res["student"]], "calls": calls, "acc_grid": acc_g, "m1_grid": m1_g,
           "cost_grids": st_g}
    if not boot:
        return out
    # ---- conditional bootstrap (E18's): split-0 predictions fixed, items resampled
    boots = [np.random.default_rng(i).integers(0, N, N) for i in range(N_BOOT)]
    m1p, accp = res["M1"][0][0], res["acc"][0][0]
    costp = [r[0][0] for r in res["cost"]]
    stp = [r[0][0] for r in res["student"]]
    cond = defaultdict(list)
    for b, i in enumerate(boots):
        s = b % 10
        a, c, t = s2(m1p[i], GOLD[i]), s2(accp[i], GOLD[i]), s2(costp[s][i], GOLD[i])
        a1, c1, t1 = s1(m1p[i], CLAR[i]), s1(accp[i], CLAR[i]), s1(costp[s][i], CLAR[i])
        cond["acc-M1 S2"].append(c - a)
        cond["acc-M1 S1"].append(c1 - a1)
        cond["cost-M1 S2"].append(t - a)
        cond["cost-M1 S1"].append(t1 - a1)
        cond["cost-student S2"].append(t - s2(stp[s][i], GOLD[i]))

    # ---- full bootstrap: every replicate refits M1's tau and the fused (lam, tau) on grouped nested CV
    t0 = time.time()
    full = defaultdict(list)
    for b in range(N_BOOT):
        rng = np.random.default_rng(10_000 + b)
        idx = rng.integers(0, N, N)
        s = b % 10
        pm1, pacc, pst, pco = boot_nested([m1_g, acc_g, sa_g[s], st_g[s]], GOLD, idx, rng, m1_only=(0, 2))
        g, c = GOLD[idx], CLAR[idx]
        v2 = [s2(p, g) for p in (pm1, pacc, pst, pco)]
        v1 = [s1(p, c) for p in (pm1, pacc, pst, pco)]
        for k, n in enumerate(("M1", "acc", "student", "cost")):
            full[f"{n} S2"].append(v2[k])
            full[f"{n} S1"].append(v1[k])
        full["acc-M1 S2"].append(v2[1] - v2[0])
        full["acc-M1 S1"].append(v1[1] - v1[0])
        full["cost-M1 S2"].append(v2[3] - v2[0])
        full["cost-M1 S1"].append(v1[3] - v1[0])
        full["cost-student S2"].append(v2[3] - v2[2])
    print(f"\n   2. Paired bootstrap over items, {N_BOOT} replicates ({time.time() - t0:.0f} s CPU).\n"
          "      'conditional' = E18's: split-0 held-out predictions fixed, items resampled.\n"
          "      'refit' = each replicate resamples items, assigns 5 CV folds over its DISTINCT items (a repeated item\n"
          "      never straddles a split), and refits M1's tau and the fused (lam, tau) inside every fold; the deferral\n"
          "      set is fixed by the slice threshold, so it needs no refit. Cost point: replicate b uses student b mod 10.\n")
    print(f"   {'contrast':<26}{'conditional: mean [95% CI]':>32}{'P(<=0)':>8}{'refit: mean [95% CI]':>32}{'P(<=0)':>8}")
    for k in ("acc-M1 S2", "acc-M1 S1", "cost-M1 S2", "cost-M1 S1", "cost-student S2"):
        a, b_ = np.array(cond[k]), np.array(full[k])
        print(f"   {k:<26}{ci(a):>32}{np.mean(a <= 0):>8.3f}{ci(b_):>32}{np.mean(b_ <= 0):>8.3f}")
    print("\n   Refit bootstrap, each system's own score (mean [95% CI]; the bootstrap mean sits below the point estimate\n"
          "   because a resample has ~63% distinct items, so each fold is fitted on fewer distinct items):")
    for n in ("M1", "acc", "student", "cost"):
        print(f"   {n:<10} S2 {ci(np.array(full[f'{n} S2'])).replace('+', '')}   S1 {ci(np.array(full[f'{n} S1'])).replace('+', '')}")
    return out


# ====================================================================================================== B

def section_b(a: dict) -> None:
    head("B. Cost: M3's GPU-seconds per call, the E18 cost table recomputed, tokens, training cost")
    recs = m3_records()
    groups = defaultdict(list)
    for r in recs:
        if "gpu_sec" in r:
            groups[(r["frame"], r["split"], bool(r.get("copied")))].append(r)
    print("   1. M3 (Qwen3.8-27B-FP8, vLLM, k = 0) records by frame / split. gpu_sec = the batch's wall time / items.\n"
          "      Copied records reuse an earlier identical request and carry ITS gpu_sec; frontier records re-score\n"
          "      dev items already scored with other candidates, so vLLM's prefix cache serves the shared item text.\n")
    print(f"   {'frame':<6}{'split':<10}{'copied':>7}{'n':>6}{'GPU-s/call':>12}{'prompt tok':>12}")
    for k in sorted(groups):
        g = groups[k]
        print(f"   {k[0]:<6}{k[1]:<10}{str(k[2]):>7}{len(g):>6}{np.mean([r['gpu_sec'] for r in g]):>12.4f}"
              f"{np.mean([r['prompt_tokens'] for r in g]):>12.0f}")
    dev1 = groups[("dev", "dev", False)]
    c_new = float(np.mean([r["gpu_sec"] for r in dev1]))
    t_new = float(np.mean([r["prompt_tokens"] for r in dev1]))
    alld = [r for k, g in groups.items() if k[0] == "dev" for r in g]
    print(f"\n   cost.json m3_call_gpu_s = {COST['m3_call_gpu_s']:.4f} (mean over all dev-frame records when cost.py ran;\n"
          f"   now {np.mean([r['gpu_sec'] for r in alld]):.4f} over {len(alld)}). Corrected: frame = split = dev, one pass over the "
          f"308 distinct items,\n   no prefix reuse: {c_new:.4f} GPU-s/call ({c_new / COST['m3_call_gpu_s']:.2f}x), "
          f"{t_new:.0f} prompt tokens (cost.json: {COST['m3_prompt_tokens']:.0f}).")
    c_m1, tok_m1 = COST["m1_seed_gpu_s_per_item"], COST["m1_tokens_per_item"]
    print(f"   M1 seed: {c_m1:.4f} GPU-s/item, {tok_m1:.0f} tokens/item (cost.json; bf16, batch 16, GPU 0 alone).")

    print("\n   2. The E18 cost table with the corrected M3 call (GPU-s per dev item; tokens per item)\n")
    rows = [("M1, 10 seeds", 10, a["calls"]["M1"]), ("accuracy point: M1 + M3, δ = 0.3", 10, a["calls"]["acc"]),
            ("student alone (mean of 10)", 1, 0.0), ("cost point: student + M3, δ = 0.5", 1, a["calls"]["cost"])]
    print(f"   {'system':<36}{'calls':>7}{'GPU-s E18':>11}{'x M1':>7}{'GPU-s E19':>11}{'x M1':>7}{'in tokens':>11}{'out':>6}")
    base_old, base_new = 10 * c_m1, 10 * c_m1
    for name, k, calls in rows:
        old = k * c_m1 + calls * COST["m3_call_gpu_s"]
        new = k * c_m1 + calls * c_new
        print(f"   {name:<36}{calls:>7.3f}{old:>11.3f}{old / base_old:>6.2f}x{new:>11.3f}{new / base_new:>6.2f}x"
              f"{k * tok_m1 + calls * t_new:>11.0f}{calls:>6.2f}")
    print("   The cost point's ratio moves from 0.21x to the E19 value above; the accuracy point's overhead over M1 from\n"
          "   +7% to the E19 value. The M3 call is the same prompt whatever the candidates (only the four option names\n"
          "   change), so one per-call cost applies to the student's candidates too.")
    tst = groups.get(("test", "test", False), [])
    if tst:
        print(f"   On the official test set (shorter answers): {np.mean([r['gpu_sec'] for r in tst]):.4f} GPU-s/call, "
              f"{np.mean([r['prompt_tokens'] for r in tst]):.0f} prompt tokens (n = {len(tst)}).")

    # ---- training cost
    print("\n   3. One-off training / labelling cost (GPU-hours). 'recorded' = wall_minutes in metrics.json, which\n"
          "      includes contention (GPU 0 time-shared with lab-mates for E14 seeds 4-6, cross-fit folds) and\n"
          "      per-epoch dev/test inference; 'uncontended' = 75 min per 12-epoch run on 3,103 rows (E14 seeds 7, 8\n"
          "      on GPU 0 alone, and every student), scaled by training rows for the cross-fit folds (2,758 rows).\n"
          "      E13 seeds 0-2 ran on JarvisLabs RTX PRO 6000s (same GPU model).\n")

    def wall(pattern: str) -> list[float]:
        return [json.loads(p.read_text())["wall_minutes"] for p in sorted(RUNS.glob(pattern))]

    m1w, cfw, stw = wall(f"{M1}/seed*/metrics.json"), wall("Q8_12ep_crossfit/fold*/metrics.json"), \
        wall(f"{STUDENT}/seed*/metrics.json")
    xfit_rows = {"trainall", "train50", "screen"}
    relabel = [r for r in recs if r["frame"] == "train" and r["split"] in xfit_rows and not r.get("copied")]
    slice_r = [r for r in recs if r["frame"] == "train" and r["split"] == "slice" and not r.get("copied")]
    relabel_h = sum(r["gpu_sec"] for r in relabel) / 3600
    slice_h = sum(r["gpu_sec"] for r in slice_r) / 3600
    dev_h = N * c_new / 3600
    unc = 75 / 60
    items = [("M1: 10 seeds x 12 epochs", sum(m1w) / 60, 10 * unc),
             ("cross-fit: 5 folds x 12 epochs", sum(cfw) / 60, 5 * unc * 2758 / 3103),
             (f"M3 relabelling of train ({len(relabel)} calls)", relabel_h, relabel_h),
             ("one student x 12 epochs (mean)", np.mean(stw) / 60, unc),
             ("10 students x 12 epochs", sum(stw) / 60, 10 * unc),
             ("M3 on dev, to fit (lam, tau) (308 calls)", dev_h, dev_h),
             (f"M3 on the slice (E15 confirmation, {len(slice_r)} calls)", slice_h, slice_h)]
    print(f"   {'component':<48}{'recorded':>10}{'uncontended':>13}")
    for name, rec, u in items:
        print(f"   {name:<48}{rec:>10.2f}{u:>13.2f}")
    acc_train = 10 * unc + dev_h
    cost_train = 5 * unc * 2758 / 3103 + relabel_h + unc + dev_h
    print(f"\n   Per system (uncontended; M3 itself is an off-the-shelf model, so no M3 training):\n"
          f"     M1 alone        : 10 M1 seeds                                    = {10 * unc:6.2f} GPU-h\n"
          f"     accuracy point  : 10 M1 seeds + M3 on dev                        = {acc_train:6.2f} GPU-h\n"
          f"     cost point      : cross-fit + M3 relabel + 1 student + M3 on dev = {cost_train:6.2f} GPU-h"
          f"   (the reported mean over 10 students trained 10: {cost_train + 9 * unc:.2f})")
    inf = {"M1": 10 * c_m1, "acc": 10 * c_m1 + a["calls"]["acc"] * c_new, "cost": c_m1 + a["calls"]["cost"] * c_new}
    tr = {"M1": 10 * unc * 3600, "acc": acc_train * 3600, "cost": cost_train * 3600}
    print("\n   Amortised GPU-s per item = training / items served + inference per item:")
    print(f"   {'items served':>14}" + "".join(f"{n:>12}" for n in ("M1", "acc point", "cost point")) + f"{'cost/M1':>10}")
    for n_items in (237, 1_000, 10_000, 100_000, 1_000_000):
        v = {k: tr[k] / n_items + inf[k] for k in inf}
        print(f"   {n_items:>14,}{v['M1']:>12.3f}{v['acc']:>12.3f}{v['cost']:>12.3f}{v['cost'] / v['M1']:>9.2f}x")
    if tr["cost"] <= tr["M1"]:
        print(f"   The cost point needs less one-off GPU time than M1's 10 seeds ({tr['cost'] / 3600:.1f} vs {tr['M1'] / 3600:.1f} GPU-h) and\n"
              "   less per item, so it is cheaper at every volume; if all 10 students are counted (as in the E18 mean),\n"
              f"   it overtakes M1 after ~{(tr['cost'] + 9 * unc * 3600 - tr['M1']) / (inf['M1'] - inf['cost']):,.0f} items. Strictly, the "
              "cost point's cross-fitted candidates were\n   ranked with the 10-seed slice calibration (tau_C); refitting that on the student's own slice "
              "probabilities\n   costs nothing extra.")
    else:
        be = (tr["cost"] - tr["M1"]) / (inf["M1"] - inf["cost"])
        print(f"   The cost point's total GPU time falls below M1's after ~{be:,.0f} items.")


# ====================================================================================================== C

def train_inputs() -> dict:
    """Cross-fitted M1 on every train row, and M3's q on the same row's cross-fitted top 4."""
    if "train" in _CACHE:
        return _CACHE["train"]
    z = m1_npz()
    c4 = top4(z["train_q"])
    by = defaultdict(list)
    for r in m3_records():
        if r["frame"] == "train" and tuple(r["cand"]) == tuple(c4[r["i"]]):
            by[r["i"]].append(r)
    lq = np.full((len(Y), N_EVASION), -np.inf)
    src = Counter()
    for i, rs in by.items():
        r = sorted(rs, key=lambda r: bool(r.get("copied")))[0]  # prefer a record M3 actually scored
        lq[i, r["cand"]] = np.log(np.maximum(r["q"], 1e-12))
        src[r["split"]] += 1
    ok = np.isfinite(lq).any(1)
    gt = np.zeros((len(Y), N_EVASION), bool)
    gt[np.arange(len(Y)), Y] = True
    out = {"lp": np.log(z["train_pbar"] + 1e-12), "pbar": z["train_pbar"], "q": z["train_q"], "lq": lq, "ok": ok,
           "gt": gt, "src": src, "u": 1 - z["train_q"].max(1), "val_idx": z["val_idx"], "cand": c4}
    _CACHE["train"] = out
    return out


def section_c() -> None:
    head("C. The cascade on the cross-fitted train rows (n = 3,448, one gold label each)")
    t = train_inputs()
    assert t["ok"].all(), f"{(~t['ok']).sum()} train rows have no M3 record on their cross-fitted top 4"
    n = len(Y)
    print(f"   M3 records: every row matched on its cross-fitted top 4 ({dict(t['src'])}; copies dropped when an\n"
          f"   original exists; the slice split's own records use the 10-seed candidates and are used only where\n"
          f"   they coincide). Gold in top 4: {(t['cand'] == Y[:, None]).any(1).mean():.3f}.\n")
    folds = folds_of(n, 0)
    preds: dict[str, np.ndarray] = {}
    fits: dict[str, dict] = {}

    def cv(g: Grid, m1_only=False) -> np.ndarray:
        out = np.empty(n, np.int64)
        for f in range(5):
            te = folds[f]
            tr = np.concatenate([folds[j] for j in range(5) if j != f])
            out[te] = (g.fit_m1(t["gt"], tr)[0] if m1_only else g.fit(t["gt"], tr)[0])[te]
        return out

    preds["M1 argmax p-bar (no tau)"] = t["pbar"].argmax(1)
    g0 = Grid(t["lp"], t["lq"], np.zeros(n, bool))
    preds["M1 + logit adj. (tau by CV)"] = cv(g0, True)
    preds["M3 alone within C(x)"] = t["lq"].argmax(1)
    for d in DELTAS[1:]:
        defer = np.ones(n, bool) if d == 1 else t["u"] >= np.quantile(t["u"], 1 - d)
        g = Grid(t["lp"], t["lq"], defer)
        preds[f"cascade δ = {d:.1f}"] = cv(g)
        fits[f"cascade δ = {d:.1f}"] = g.fit(t["gt"], np.arange(n))[1]
    names = list(preds)
    P = np.stack([preds[k] for k in names])
    f1 = macro_f1_many(P, t["gt"])
    acc = (P == Y).mean(1)
    rng = np.random.default_rng(19)
    bf, ba = [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        bf.append(macro_f1_many(P[:, i], t["gt"][i]))
        ba.append((P[:, i] == Y[i]).mean(1))
    bf, ba = np.array(bf), np.array(ba)
    ref = names.index("M1 + logit adj. (tau by CV)")
    print("   δ = fraction deferred, by train's own quantile of u = 1 - top q (so the realised rate is exactly δ).\n"
          "   (lam, tau) and M1's tau fitted by 5-fold CV within train (macro-F1); CIs: 2,000 bootstrap draws of rows.\n")
    print(f"   {'system':<30}{'macro-F1 [95% CI]':>24}{'Δ vs M1+LA [95% CI]':>28}{'accuracy [95% CI]':>24}{'Δ acc':>9}  lam, tau (all train)")
    for k, name in enumerate(names):
        df, da = bf[:, k] - bf[:, ref], ba[:, k] - ba[:, ref]
        ft = fits.get(name)
        fs = f"  {ft['lam']:.2f}, {ft['tau']:+.2f}" if ft else ""
        print(f"   {name:<30}{f1[k]:>7.3f} [{np.percentile(bf[:, k], 2.5):.3f}, {np.percentile(bf[:, k], 97.5):.3f}]"
              f"{f1[k] - f1[ref]:>+12.3f} [{np.percentile(df, 2.5):+.3f}, {np.percentile(df, 97.5):+.3f}]"
              f"{acc[k]:>8.3f} [{np.percentile(ba[:, k], 2.5):.3f}, {np.percentile(ba[:, k], 97.5):.3f}]"
              f"{acc[k] - acc[ref]:>+9.3f}{fs}")
    slice_rate = {d: float((t["u"] >= np.quantile(1 - m1_npz()["val_q"].max(1), 1 - d)).mean()) for d in (0.1, 0.3, 0.5)}
    print(f"\n   For reference, the dev threshold (slice quantile of the 10-seed u) would defer "
          + ", ".join(f"{v:.2f} at δ = {d}" for d, v in slice_rate.items()) + " of train rows: the single cross-fitted\n"
          "   model is less confident than the 10-seed ensemble.")
    print("   Caveats: (1) train_pbar is ONE cross-fitted model per fold (5 folds), not the 10-seed ensemble, so M1 is\n"
          "   weaker and less calibrated here than on dev; (2) the calibration (tau_C, T) and the uncertainty model were\n"
          "   fitted on the 345-row slice, which is part of these rows; (3) M3's confusion guide was written from\n"
          "   deferred cross-fitted train rows (E15), so M3 is not blind to this population; (4) one gold label per\n"
          "   row: macro-F1 here is single-label, not the dev multi-reference score.")


# ====================================================================================================== D

def value_rows(m1_ok: np.ndarray, m3_ok: np.ndarray, u: np.ndarray, k: int, rng_seed: int) -> list[tuple]:
    order = np.argsort(u, kind="stable")  # most confident first
    out = []
    rng = np.random.default_rng(rng_seed)
    for ch in np.array_split(order, k):
        fx = m3_ok[ch] & ~m1_ok[ch]
        br = ~m3_ok[ch] & m1_ok[ch]
        net = fx.astype(int) - br.astype(int)
        bs = [net[rng.integers(0, len(ch), len(ch))].sum() for _ in range(N_BOOT)]
        out.append((len(ch), u[ch].min(), u[ch].max(), int(fx.sum()), int(br.sum()), int(net.sum()),
                    np.percentile(bs, 2.5), np.percentile(bs, 97.5), m1_ok[ch].mean(), m3_ok[ch].mean()))
    return out


def best_delta(m1_ok, m3_ok, u, rng_seed: int) -> tuple[float, float, float, float]:
    """δ* = deferral fraction (least confident first, 1% grid) maximising fixes - breaks; bootstrap 95% interval."""
    grid_d = np.linspace(0.01, 1.0, 100)

    def arg(ix):
        o = ix[np.argsort(-u[ix], kind="stable")]
        net = np.cumsum((m3_ok[o] & ~m1_ok[o]).astype(int) - (~m3_ok[o] & m1_ok[o]).astype(int))
        ks = np.maximum(1, np.round(grid_d * len(o)).astype(int)) - 1
        v = net[ks]
        return grid_d[int(np.argmax(v))], v.max()

    n = len(u)
    d0, v0 = arg(np.arange(n))
    rng = np.random.default_rng(rng_seed)
    bs = [arg(rng.integers(0, n, n))[0] for _ in range(N_BOOT)]
    return d0, v0, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def print_value(title: str, rows: list[tuple]) -> None:
    print(f"   {title}")
    print(f"   {'fifth':<7}{'n':>5}{'u range':>16}{'M1 ok':>7}{'M3 ok':>7}{'fixes':>7}{'breaks':>7}{'net':>6}{'net 95% CI':>16}")
    for q, r in enumerate(rows):
        print(f"   Q{q + 1:<6}{r[0]:>5}{f'{r[1]:.2f}-{r[2]:.2f}':>16}{r[8]:>7.3f}{r[9]:>7.3f}{r[3]:>7}{r[4]:>7}{r[5]:>+6}"
              f"{f'[{r[6]:+.0f}, {r[7]:+.0f}]':>16}")


def section_d() -> None:
    head("D. Value map: M3 vs M1 (argmax p-bar, no tau) by M1-uncertainty fifth, Q1 = most confident")
    print("   fixes = M3 right and M1 wrong; breaks = the reverse; net = fixes - breaks (95% CI: 2,000 bootstrap draws\n"
          "   of the fifth's items). M3 = argmax of q over M1's top 4 (no fusion), so this is M3's raw value.\n")
    t = train_inputs()
    m1t, m3t = t["pbar"].argmax(1) == Y, t["lq"].argmax(1) == Y
    print_value("Train (cross-fitted, n = 3,448, correct = the gold label; u = 1 - top q of the cross-fitted model):",
                value_rows(m1t, m3t, t["u"], 5, 1))
    d0, v0, lo, hi = best_delta(m1t, m3t, t["u"], 2)
    print(f"   deciles of net: {[r[5] for r in value_rows(m1t, m3t, t['u'], 10, 3)]}")
    print(f"   δ* (best fraction to defer, least confident first): {d0:.2f} (net {v0:+d}); bootstrap 95% [{lo:.2f}, {hi:.2f}]\n")
    m = inputs(M1, TEN)
    lq = m["lq_dev"]
    m1d, m3d = GOLD[IDX, m["pbar_dev"].argmax(1)], GOLD[IDX, lq.argmax(1)]
    z = m1_npz()
    for name, u in (("u = 1 - top q (the routing signal)", m["u_dev"]),
                    ("u = logistic error model (dev_u in cascade_inputs.npz)", z["dev_u"])):
        print_value(f"Dev (n = 308, correct = in the reference set), {name}:", value_rows(m1d, m3d, u, 5, 4))
        d0, v0, lo, hi = best_delta(m1d, m3d, u, 5)
        print(f"   deciles of net: {[r[5] for r in value_rows(m1d, m3d, u, 10, 6)]}")
        print(f"   δ* = {d0:.2f} (net {v0:+d}); bootstrap 95% [{lo:.2f}, {hi:.2f}]\n")
    print("   The preliminary numbers (train -41/-76/-11/+4/+31, dev -4/0/-7/+8/+8) are reproduced up to the train\n"
          "   record choice (here: the record whose candidates are the row's cross-fitted top 4) and, on dev, with the\n"
          "   logistic u; with the routing signal itself the dev fifths are as in the first dev table.")


# ====================================================================================================== E

def section_e(a: dict) -> None:
    head("E. Routing and fusion ablations on dev (10-seed M1 + M3; nested CV split 0 unless stated)")
    m = inputs(M1, TEN)
    z = m1_npz()
    x = z["dev_x"]  # top_prob, margin, entropy, seed_disagreement, set_size (10-seed ensemble)
    u = m["u_dev"]
    llm = m["lq_dev"].argmax(1)
    m1p = a["m1_pred"]
    signals = {
        "confidence 1 - top q (E15)": u,
        "margin (low first)": -x[:, 1] + 1e-9 * u,
        "entropy (high first)": x[:, 2] + 1e-9 * u,
        "seed disagreement (ties by u)": x[:, 3] + 1e-6 * u,
        "logistic error model u": z["dev_u"],
        "1 - top p-bar (uncalibrated)": 1 - m["pbar_dev"].max(1),
    }
    ok1, ok3 = GOLD[IDX, m1p], GOLD[IDX, llm]
    # gold-informed order: M3 fixes M1 > both right > both wrong > M3 breaks M1 (ties by u)
    pri = 3.0 * (~ok1 & ok3) + 2.0 * (ok1 & ok3) + 1.0 * (~ok1 & ~ok3) + 1e-6 * u

    def score_defer(defer, lams=LAMS, taus=TAUS, splits=1):
        g = Grid(m["lp_dev"], m["lq_dev"], defer, lams=np.asarray(lams), taus=np.asarray(taus))
        ps = [nested(g, GOLD, cs) for cs in range(splits)]
        return s2(ps[0]), float(np.mean([s2(p) for p in ps])), inset(ps[0]), s1(ps[0])

    def top_k(sig, k):
        d = np.zeros(N, bool)
        d[np.argsort(-sig, kind="stable")[:k]] = True
        return d

    print("   1. Routing at matched budget k = the number of items the slice threshold defers at δ (confidence routing's\n"
          "      realised count). Random: 200 draws (seeds 0-199), mean ± sd, and the share of draws >= confidence.\n"
          "      Gold-informed: defer first the items M3 alone fixes (M1's split-0 system wrong, M3's argmax right),\n"
          "      then those both get right, then those both get wrong, last those M3 breaks. It uses the gold labels\n"
          "      (a reference, not a system); with (lam, tau) refitted it is not a strict upper bound.\n"
          "      '10sp' = mean of 10 nested-CV splits; random and gold-informed: split 0.\n")
    curve = {}
    for d in (0.1, 0.2, 0.3, 0.5):
        k = int(defer_mask(u, m["u_val"], d).sum())
        print(f"   δ = {d} (k = {k}, {k / N:.3f} of dev)")
        print(f"      {'routing':<34}{'S2':>7}{'S2 10sp':>9}{'in-set':>8}{'S1':>7}")
        conf = None
        for name, sig in signals.items():
            r = score_defer(top_k(sig, k), splits=10)
            conf = r if conf is None else conf
            print(f"      {name:<34}{r[0]:>7.3f}{r[1]:>9.3f}{r[2]:>8.3f}{r[3]:>7.3f}")
        rnd = []
        for s_ in range(200):
            dd = np.zeros(N, bool)
            dd[np.random.default_rng(s_).choice(N, k, replace=False)] = True
            rnd.append(score_defer(dd))
        rnd = np.array(rnd)
        print(f"      {'random (200 draws)':<34}{rnd[:, 0].mean():>7.3f}{'':>9}{rnd[:, 2].mean():>8.3f}{rnd[:, 3].mean():>7.3f}"
              f"   sd S2 {rnd[:, 0].std(ddof=1):.3f}; P(random >= confidence) {np.mean(rnd[:, 0] >= conf[0]):.3f}")
        orc = score_defer(top_k(pri, k))
        print(f"      {'gold-informed routing':<34}{orc[0]:>7.3f}{'':>9}{orc[2]:>8.3f}{orc[3]:>7.3f}")
        curve[d] = (conf[0], rnd[:, 0].mean(), rnd[:, 0].std(ddof=1), orc[0])
    allr = score_defer(np.ones(N, bool))
    print(f"\n   δ = 1 (every item; routing is moot): S2 {allr[0]:.3f}, in-set {allr[2]:.3f}; M1 alone (δ = 0): "
          f"S2 {s2(m1p):.3f}, in-set {inset(m1p):.3f}")
    print("\n   Curve summary (S2, split 0):  δ   confidence   random mean ± sd   gold-informed")
    for d, (c, r, sd, o) in curve.items():
        print(f"                                {d:<4}{c:>9.3f}{r:>13.3f} ± {sd:.3f}{o:>10.3f}")

    print("\n   2. Fusion ablations (confidence routing, slice threshold; S2 split 0 and mean of 10 splits)\n")
    print(f"   {'δ':>5}{'calls':>7}   {'fitted lam, tau':>18}{'lam = 0 (M3 only)':>20}{'lam = 1 (M1 only)':>20}"
          f"{'M3 argmax (lam = 0, tau = 0)':>30}")
    for d in DELTAS[1:]:
        defer = defer_mask(u, m["u_val"], d)
        f = score_defer(defer, splits=10)
        l0 = score_defer(defer, lams=[0.0], splits=10)
        l1 = score_defer(defer, lams=[1.0], splits=10)
        raw = score_defer(defer, lams=[0.0], taus=[0.0], splits=10)
        cell = lambda r: f"{r[0]:.3f} / {r[1]:.3f}"  # noqa: E731
        print(f"   {d:>5.1f}{defer.mean():>7.3f}   {cell(f):>18}{cell(l0):>20}{cell(l1):>20}{cell(raw):>30}")
    print("   lam = 1 keeps M1's scores but restricts the deferred items to C(x) with their own tau: what the\n"
          "   routing + re-thresholding alone buys, without M3. lam = 0 lets M3 decide the deferred items alone.")


# ====================================================================================================== F

def section_f(a: dict) -> None:
    head("F. Strata on dev (split-0 held-out predictions; cost point = mean over the 10 students)")
    dev = sp.dev
    train_urls = set(sp.train["url"])
    seen = dev["url"].isin(train_urls).to_numpy()
    words = dev["interview_answer"].astype(str).str.split().str.len().to_numpy()
    short = np.zeros(N, bool)
    short[np.argsort(words, kind="stable")[: N // 2]] = True
    cons = consensus_level(dev)
    import pandas as pd

    test = pd.read_csv(TEST_CSV)
    tw = test["interview_answer"].astype(str).str.split().str.len().to_numpy()
    print(f"   Interviews: {dev['url'].nunique()} on dev, {int(dev['url'].isin(train_urls).groupby(dev['url']).first().sum())} of them "
          f"also in train ({seen.sum()} dev items); official test: {test['url'].isin(train_urls).sum()} of {len(test)} items\n"
          f"   from a train interview. Answer length (words): dev median {np.median(words):.0f} (shorter half <= "
          f"{words[short].max()}), test median {np.median(tw):.0f}; {np.mean(tw <= words[short].max()):.2f} of test answers are\n"
          f"   within the shorter half's range.\n")
    m1p, accp, costp = a["m1_pred"], a["acc_pred"], a["cost_preds"]
    m = inputs(M1, TEN)
    defer = defer_mask(m["u_dev"], m["u_val"], 0.3)
    strata = [("interview also in train", seen), ("interview not in train", ~seen),
              ("shorter half of answers", short), ("longer half of answers", ~short),
              ("3 annotators agree", cons == 1), ("2 agree", cons == 2), ("all 3 differ", cons == 3), ("all", np.ones(N, bool))]
    print(f"   {'stratum':<26}{'n':>4}{'defer':>7}{'M1 in-set':>10}{'acc in-set':>11}{'Δ in-set [95% CI]':>24}"
          f"{'M1 S2':>7}{'acc S2':>7}{'Δ S2':>7}{'cost in-set':>12}{'cost S2':>8}")
    rng = np.random.default_rng(7)
    for name, sel in strata:
        ix = np.where(sel)[0]
        h0, h1 = GOLD[ix, m1p[ix]].astype(int), GOLD[ix, accp[ix]].astype(int)
        bs = [np.mean(h1[j] - h0[j]) for j in (rng.integers(0, len(ix), len(ix)) for _ in range(N_BOOT))]
        a2, b2 = s2(m1p[ix], GOLD[ix]), s2(accp[ix], GOLD[ix])
        ci_ = np.mean([inset(p[ix], GOLD[ix]) for p in costp])
        cs2 = np.mean([s2(p[ix], GOLD[ix]) for p in costp])
        print(f"   {name:<26}{len(ix):>4}{defer[ix].mean():>7.2f}{h0.mean():>10.3f}{h1.mean():>11.3f}"
              f"{f'{np.mean(h1 - h0):+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]':>24}"
              f"{a2:>7.3f}{b2:>7.3f}{b2 - a2:>+7.3f}{ci_:>12.3f}{cs2:>8.3f}")
    print("   S2 within a stratum is the macro-F1 of that subset (classes absent from it score 0 for both systems), so\n"
          "   compare M1 and the system within a row, not across rows; the in-set rate is the per-item quantity.")


# ====================================================================================================== G

def section_g() -> None:
    head("G. M3's letter-position and label bias")
    print("   Candidates are shown in taxonomy (class-index) order, A = the lowest index (cascade.prompts.build_messages),\n"
          "   so position and class are confounded: A is Explicit whenever Explicit is a candidate.\n")
    m = inputs(M1, TEN)
    cand = m["cand_dev"]  # sorted top 4
    q = np.exp(m["lq_dev"])
    pick_pos = np.array([int(np.argmax(q[i, cand[i]])) for i in range(N)])
    m1_pos = np.array([int(np.argmax(m["q_dev"][i, cand[i]])) for i in range(N)])
    in_ref = GOLD[IDX[:, None], cand]  # (N, 4)
    one = in_ref.sum(1) == 1
    print("   Dev (308 items, the 10-seed top 4):")
    print(f"   {'position':<10}{'M3 picks':>9}{'M1 top':>8}{'class in ref':>14}{'only in-ref (n=' + str(one.sum()) + ')':>22}{'mean q':>8}")
    for j in range(4):
        print(f"   {LETTERS[j]:<10}{np.mean(pick_pos == j):>9.3f}{np.mean(m1_pos == j):>8.3f}{in_ref[:, j].mean():>14.3f}"
              f"{np.mean(in_ref[one].argmax(1) == j):>22.3f}{q[IDX, cand[:, j]].mean():>8.3f}")
    print("   ('class in ref' = share of items whose position-j candidate is in the reference set; 'only in-ref' = the\n"
          "   position of the single in-reference candidate on items that have exactly one)")
    t = train_inputs()
    ct = t["cand"]
    qt = np.exp(t["lq"])
    ppos = np.array([int(np.argmax(qt[i, ct[i]])) for i in range(len(Y))])
    mpos = np.array([int(np.argmax(t["q"][i, ct[i]])) for i in range(len(Y))])
    gin = (ct == Y[:, None])
    has = gin.any(1)
    print(f"\n   Train (3,448 cross-fitted rows; gold in top 4 for {has.mean():.3f}):")
    print(f"   {'position':<10}{'M3 picks':>9}{'M1 top':>8}{'gold here':>11}{'M3 right | picks j':>20}")
    for j in range(4):
        sel = ppos == j
        print(f"   {LETTERS[j]:<10}{np.mean(ppos == j):>9.3f}{np.mean(mpos == j):>8.3f}{np.mean(gin[has].argmax(1) == j):>11.3f}"
              f"{np.mean(ct[sel, j] == Y[sel]):>20.3f}")
    print("\n   Same class, different position (train): P(M3 picks class c | c is a candidate at position j). A pure\n"
          "   position bias would lower this for every class as j grows; n in brackets.")
    pick_cls = ct[np.arange(len(Y)), ppos]
    print(f"   {'class':<22}" + "".join(f"{LETTERS[j]:>14}" for j in range(4)))
    for c in range(N_EVASION):
        cells = []
        for j in range(4):
            sel = ct[:, j] == c
            cells.append(f"{np.mean(pick_cls[sel] == c):.2f} ({sel.sum()})" if sel.sum() >= 20 else "-")
        print(f"   {EVASION_LABELS[c]:<22}" + "".join(f"{x:>14}" for x in cells))
    print("\n   Per class (train): how often a class is a candidate, M3 picks it, the pick is right; gold count; recall\n"
          "   of M3 given gold in C(x); M1 (argmax p-bar) picks and precision for comparison")
    print(f"   {'class':<22}{'cand':>6}{'gold':>6}{'M3 picks':>9}{'M3 prec':>9}{'M3 rec|C':>9}{'M1 picks':>9}{'M1 prec':>9}")
    m1c = t["pbar"].argmax(1)
    for c in range(N_EVASION):
        pk = pick_cls == c
        g = (Y == c) & has
        print(f"   {EVASION_LABELS[c]:<22}{int((ct == c).any(1).sum()):>6}{int((Y == c).sum()):>6}{int(pk.sum()):>9}"
              f"{(np.mean(Y[pk] == c) if pk.any() else 0):>9.3f}{(np.mean(pick_cls[g] == c) if g.any() else 0):>9.3f}"
              f"{int((m1c == c).sum()):>9}{(np.mean(Y[m1c == c] == c) if (m1c == c).any() else 0):>9.3f}")
    print("\n   Confusion of M3's picks on train, rows = gold (rows with gold in C only), columns = M3's pick")
    cm = np.zeros((N_EVASION, N_EVASION), int)
    for yy, pp in zip(Y[has], pick_cls[has]):
        cm[yy, pp] += 1
    print(f"   {'':<14}" + "".join(f"{s[:6]:>7}" for s in SHORT))
    for c in range(N_EVASION):
        print(f"   {SHORT[c]:<14}" + "".join(f"{v:>7}" for v in cm[c]))
    dpick = cand[IDX, pick_pos]
    print("\n   Dev, per class picked by M3: n picked, share in the reference set (precision under S2's rule)")
    print("   " + ", ".join(f"{SHORT[c]} {int((dpick == c).sum())} ({GOLD[dpick == c, c].mean():.2f})"
                          for c in range(N_EVASION) if (dpick == c).any()))


# ====================================================================================================== H

def section_h() -> None:
    head("H. Test-set predictions (evaluation.make_cascade_submission; written to submissions/E19_*)")
    from evaluation.make_cascade_submission import build_all

    build_all(verbose=True)


# ====================================================================================================== I

def section_i(a: dict) -> None:
    head("I. Human ceiling: one annotator against the other two (the official test set's 2-annotator regime)")
    lab = np.stack([[EVASION_LABELS.index(v) for v in sp.dev[c]] for c in ("annotator1", "annotator2", "annotator3")], 1)
    two = {k: v for k, v in reference_masks(sp.dev, True).items() if k != "3ann"}
    st_g = a["cost_grids"]
    print("   Every 2-annotator reference set; the human row scores the held-out annotator's labels against it; the\n"
          "   systems are refitted by nested CV on that same set (split 0, as E18's 2-annotator column), and also\n"
          "   scored with their 3-annotator-fitted predictions unchanged ('fixed').\n")
    print(f"   {'reference set':<24}{'human':<12}{'human S2':>9}{'in-set':>8}{'M1':>7}{'acc refit':>10}{'acc fixed':>10}"
          f"{'cost refit':>11}{'cost fixed':>11}")
    rows = []
    for name, g in two.items():
        held = ({0, 1, 2} - {int(c[-1]) - 1 for c in name.split("+")}).pop()
        hs = s2(lab[:, held], g)
        m1 = s2(nested(a["m1_grid"], g, 0, m1_only=True), g)
        acc_r = s2(nested(a["acc_grid"], g, 0), g)
        acc_f = s2(a["acc_pred"], g)
        cr = np.mean([s2(nested(sg, g, 0), g) for sg in st_g])
        cf = np.mean([s2(p, g) for p in a["cost_preds"]])
        rows.append((hs, inset(lab[:, held], g), m1, acc_r, acc_f, cr, cf))
        print(f"   {name:<24}{'annotator' + str(held + 1):<12}{hs:>9.3f}{rows[-1][1]:>8.3f}{m1:>7.3f}{acc_r:>10.3f}"
              f"{acc_f:>10.3f}{cr:>11.3f}{cf:>11.3f}")
    r = np.mean(rows, 0)
    print(f"   {'mean':<36}{r[0]:>9.3f}{r[1]:>8.3f}{r[2]:>7.3f}{r[3]:>10.3f}{r[4]:>10.3f}{r[5]:>11.3f}{r[6]:>11.3f}")
    print(f"\n   The 0.684 'human ceiling' is the 2-annotator number (mean of the human column). The comparable system\n"
          f"   number is the accuracy point on the same sets, {r[3]:.3f} refitted ({r[4]:.3f} with its dev-fitted predictions),\n"
          f"   not the 3-annotator 0.669; M1 alone {r[2]:.3f}; the cost point {r[5]:.3f}. Gap to humans: {r[0] - r[3]:.3f} (accuracy\n"
          f"   point), {r[0] - r[2]:.3f} (M1). A human scored on a 3-annotator set would include their own label, so no\n"
          "   3-annotator human ceiling exists on dev.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sections", default="ABCDEFGHI")
    s = set(p.parse_args().sections.replace(",", "").upper())
    t0 = time.time()
    print("E19, Tier 0: CPU-only analyses of the E18 systems (dev = the 308-item QEvasion test split, 3 annotators;\n"
          "S2 = multi-reference macro-F1, S1 = clarity macro-F1). M1 = 10-seed Q8_fullq_lora_12ep; M3 = Qwen3.8-27B-FP8,\n"
          "k = 0 (tag qwen38_27b_k0); students = Q8_distill_a05; M3 candidates = each record's `cand` (the top 4 of the\n"
          "calibrated q).")
    a = section_a("A" in s) if s & set("ABEFI") else None
    if "B" in s:
        section_b(a)
    if "C" in s:
        section_c()
    if "D" in s:
        section_d()
    if "E" in s:
        section_e(a)
    if "F" in s:
        section_f(a)
    if "G" in s:
        section_g()
    if "H" in s:
        section_h()
    if "I" in s:
        section_i(a)
    print(f"\n[E19 tier 0: {time.time() - t0:.0f} s wall, CPU only]")


if __name__ == "__main__":
    main()
