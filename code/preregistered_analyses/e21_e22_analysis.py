#!/usr/bin/env python3
"""E19 A/B, E21a and E22a/b/e: the registered checks that need no new GPU work (docs/02_experiment_log.md §E19-E22).

    python -m preregistered_analyses.e21_e22_analysis > ../docs/raw/E21_E22_analysis.txt    # CPU, < 1 min

Inputs (HF: E21/, E22/, M3/qwen38_27b_k0_{vm,jarvis,jarvis_cached,shift1,shift2,shift3,reverse,all9}/):
A. E19 arm A   -- the 27B alone over all nine labels (dev, test).
B. E19 arm B / E22b -- option order: position bias per order, and the registered decision rule (permutation-averaged q
   over the 4 cyclic orders vs single-order q, on the 300 screen rows: within-C accuracy AND NLL).
C. E22a -- the cache-aligned prompt: q agreement with the original prompt and GPU-s per call, on the same VM.
D. The faithfulness gate in detail: the VM's 27B against the lab server's records, per split.
E. E21a / H21a -- transcript artifacts injected into dev: prediction changes and S2 of the 10-seed M1 system; the
   normalized test labels that change.
F. E22e / H22e -- question-answer specificity by clarity level on train (and dev).
Every M3 record set used for a comparison comes from ONE copy of the model (the VM), as the faith gate requires.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from cascade.prompts import display_order
from decision_rules.decide import logit_adjust
from qevasion.labels import CLARITY_LABELS, EVASION_LABELS, N_EVASION, encode_clarity, encode_evasion, leaf_to_official_clarity
from qevasion.loader import DATA_CACHE, dev_reference_mask, load_qevasion
from qevasion.paths import RUNS
from qevasion.scoring import score_subtask1, score_subtask2

SP = load_qevasion(DATA_CACHE)
DEV, TRAIN = SP.dev.reset_index(drop=True), SP.train.reset_index(drop=True)
G = dev_reference_mask(DEV)
Y1_DEV = encode_clarity(DEV["clarity_label"].tolist())
Y_TR = encode_evasion(TRAIN["evasion_label"].tolist())
PRIOR = np.bincount(Y_TR, minlength=N_EVASION) / len(Y_TR)
TAU_M1 = 1.15  # the 10-seed M1's tau fitted on all of dev (evaluation.make_cascade_submission, E19 Tier 0 H)
rule = lambda: print("=" * 110)


def recs(tag: str) -> dict:
    """(split, i) -> record; the last record wins (copies of identical requests are identical)."""
    out = {}
    for line in (RUNS / "M3" / tag / "responses.jsonl").read_text().splitlines():
        r = json.loads(line)
        out[(r["split"], r["i"])] = r
    return out


def scores(pred: np.ndarray) -> tuple[float, float, float]:
    s2 = score_subtask2(pred, G).macro_f1
    s1 = score_subtask1(leaf_to_official_clarity(pred), Y1_DEV).macro_f1
    return s2, s1, float(G[np.arange(len(pred)), pred].mean())


def section_a() -> None:
    rule(); print("A. E19 arm A: Qwen3.8-27B-FP8 alone over all nine labels (VM, k = 0, no thinking)"); rule()
    # one split name ("all9") covers dev and test, so records are keyed by frame here
    r = {}
    for line in (RUNS / "M3" / "qwen38_27b_k0_all9" / "responses.jsonl").read_text().splitlines():
        rec = json.loads(line)
        r[(rec["frame"], rec["i"])] = rec
    dev = np.array([r[("dev", i)]["q"] for i in range(len(DEV))])
    pred = dev.argmax(1)
    s2, s1, ins = scores(pred)
    print(f"   items scored: dev {sum(f == 'dev' for f, _ in r)}, test {sum(f == 'test' for f, _ in r)}")
    print(f"   dev: S2 {s2:.3f}  S1 {s1:.3f}  in-set {ins:.3f}   (registered expectation: below M1 alone, S2 0.598)")
    print("   dev predictions per label: " + ", ".join(f"{EVASION_LABELS[c][:10]} {n}" for c, n in
                                                     zip(*np.unique(pred, return_counts=True))))
    print(f"   expectation {'HELD' if s2 < 0.598 else 'FAILED'}: the LLM alone {'is below' if s2 < 0.598 else 'is not below'} M1 alone")


def order_table(split: str, gold_of) -> dict:
    tags = {"taxonomy": "qwen38_27b_k0_vm", "shift1": "qwen38_27b_k0_shift1", "shift2": "qwen38_27b_k0_shift2",
            "shift3": "qwen38_27b_k0_shift3", "reverse": "qwen38_27b_k0_reverse"}
    R = {o: recs(t) for o, t in tags.items()}
    keys = sorted(k for k in R["taxonomy"] if k[0] == split and all(k in R[o] for o in R))
    out = {}
    for o in R:
        q = np.array([R[o][k]["q"] for k in keys])
        cand = [R[o][k]["cand"] for k in keys]
        assert all(R[o][k]["cand"] == R["taxonomy"][k]["cand"] for k in keys), "candidate sets differ across orders"
        first = np.array([display_order(c, o)[0] for c in cand])
        out[o] = (q, cand, first)
    out["average (4 cyclic)"] = (np.mean([out[o][0] for o in ("taxonomy", "shift1", "shift2", "shift3")], 0),
                                 out["taxonomy"][1], None)
    golds = [gold_of(k[1]) for k in keys]
    return {"keys": keys, "rows": out, "golds": golds}


def section_b() -> None:
    rule(); print("B. E19 arm B / E22b: candidate order -- position bias and the permutation-averaged q"); rule()
    t = order_table("screen", lambda i: int(Y_TR[i]))
    print(f"   300 screen rows (train, one gold label): {len(t['keys'])} rows scored in all five orders")
    print(f"   {'order':20} {'picks letter A':>15} {'acc within C':>13} {'NLL (gold in C)':>16}")
    res = {}
    for o, (q, cand, first) in t["rows"].items():
        pick = np.array([c[j] for c, j in zip(cand, q.argmax(1))])
        acc = float((pick == np.array(t["golds"])).mean())
        nll = [-np.log(max(q[n][c.index(g)], 1e-12)) for n, (c, g) in enumerate(zip(cand, t["golds"])) if g in c]
        res[o] = (acc, float(np.mean(nll)))
        a = f"{(pick == first).mean():.3f}" if first is not None else "-"
        print(f"   {o:20} {a:>15} {acc:13.3f} {res[o][1]:16.3f}")
    single, avg = res["taxonomy"], res["average (4 cyclic)"]
    adopt = avg[0] > single[0] and avg[1] < single[1]
    print(f"   decision rule (registered): adopt permutation-averaged q iff it beats single-order q on BOTH -> "
          f"{'ADOPTED' if adopt else 'NOT adopted'} (acc {avg[0] - single[0]:+.3f}, NLL {avg[1] - single[1]:+.3f})")
    d = order_table("dev", lambda i: i)
    print("   dev (reported only): in-set rate of the argmax within C, per order")
    for o, (q, cand, first) in d["rows"].items():
        pick = np.array([c[j] for c, j in zip(cand, q.argmax(1))])
        a = f"; picks A {(pick == first).mean():.3f}" if first is not None else ""
        print(f"     {o:20} in-set {G[np.array([k[1] for k in d['keys']]), pick].mean():.3f}{a}")


def section_c() -> None:
    rule(); print("C. E22a: the cache-aligned prompt against the original prompt (same VM, dev, 308 items)"); rule()
    a, b = recs("qwen38_27b_k0_jarvis"), recs("qwen38_27b_k0_jarvis_cached")
    keys = sorted(k for k in a if k[0] == "dev" and k in b)
    qa, qb = np.array([a[k]["q"] for k in keys]), np.array([b[k]["q"] for k in keys])
    agree = float((qa.argmax(1) == qb.argmax(1)).mean())
    ga, gb = np.mean([a[k]["gpu_sec"] for k in keys]), np.mean([b[k]["gpu_sec"] for k in keys])
    pa, pb = np.mean([a[k]["prompt_tokens"] for k in keys]), np.mean([b[k]["prompt_tokens"] for k in keys])
    print(f"   GPU-s per call: original {ga:.3f}, cached {gb:.3f} (ratio {gb / ga:.2f}); prompt tokens {pa:.0f} vs {pb:.0f}")
    print(f"   argmax agreement {agree:.3f}; mean max |dq| {np.abs(qa - qb).max(1).mean():.4f}")
    print(f"   H22a (cost <= 0.6x AND agreement >= 0.98): {'HELD' if gb / ga <= 0.6 and agree >= 0.98 else 'FAILED'}")
    cost = json.loads((RUNS / "E22" / "cost_jarvis.json").read_text())
    print(f"   one M1 seed on the same GPU: {cost['m1_seed_gpu_s_per_item']:.4f} GPU-s per item "
          f"(max |dp| vs saved {cost['m1_max_abs_diff_vs_saved_probs']})")


def section_d() -> None:
    rule(); print("D. The faithfulness gate: the VM's 27B (qwen38_27b_k0_vm) against the lab server's records (qwen38_27b_k0)"); rule()
    lab, vm = recs("qwen38_27b_k0"), recs("qwen38_27b_k0_vm")
    for split in ("dev", "test", "screen", "slice", "trainall"):
        keys = [k for k in lab if k[0] == split and k in vm and lab[k]["cand"] == vm[k]["cand"]]
        if not keys:
            continue
        qa, qb = np.array([lab[k]["q"] for k in keys]), np.array([vm[k]["q"] for k in keys])
        print(f"   {split:9} {len(keys):5} items: argmax agreement {(qa.argmax(1) == qb.argmax(1)).mean():.3f}, "
              f"mean max |dq| {np.abs(qa - qb).max(1).mean():.4f}")
    print("   gate (>= 0.98 on dev) failed, so every new analysis uses the VM's records only (E19 amendment)")


def m1_system(variant: str, run: str = "Q8_fullq_lora_12ep") -> np.ndarray:
    p = np.mean([np.load(RUNS / "E21" / run / f"seed{k}" / f"{variant}_probs.npy") for k in range(10)], 0)
    return logit_adjust(p, PRIOR, TAU_M1).argmax(1)


def section_e() -> None:
    rule(); print("E. E21a / H21a: test's transcript artifacts injected into dev (seed 21), and normalized test"); rule()
    rep = json.loads((RUNS / "E21" / "infer_report.json").read_text())
    print(f"   control: {sum(max(r['ctrl_max_abs_dp'].values()) < 1e-3 for r in rep)}/{len(rep)} seed folders reproduce "
          f"their saved probabilities (max |dp| < 1e-3)")
    c, p = m1_system("dev_ctrl"), m1_system("dev_pert")
    sc, spt = scores(c), scores(p)
    flip = float((c != p).mean())
    print(f"   10-seed M1 (mean p, logit adjustment tau {TAU_M1}): dev predictions changed {flip:.3f}; "
          f"S2 {sc[0]:.3f} -> {spt[0]:.3f} ({spt[0] - sc[0]:+.3f}); S1 {sc[1]:.3f} -> {spt[1]:.3f}; in-set {sc[2]:.3f} -> {spt[2]:.3f}")
    held = flip >= 0.05 or spt[0] - sc[0] <= -0.01
    print(f"   H21a (>= 5% of predictions change, or S2 falls by >= 0.01): {'HELD' if held else 'FAILED'}")
    sf = [float((np.load(RUNS / 'E21' / 'Q8_distill_a05' / f'seed{k}' / 'dev_pert_probs.npy').argmax(1) !=
                 np.load(RUNS / 'E21' / 'Q8_distill_a05' / f'seed{k}' / 'dev_ctrl_probs.npy').argmax(1)).mean()) for k in range(10)]
    print(f"   single students (argmax): dev predictions changed {np.mean(sf):.3f} on average (min {min(sf):.3f}, max {max(sf):.3f})")
    tc, tn = m1_system("test_ctrl"), m1_system("test_norm")
    print(f"   official test, 10-seed M1: {int((tc != tn).sum())} of {len(tc)} labels change with normalized answers "
          f"(S1 labels: {int((leaf_to_official_clarity(tc) != leaf_to_official_clarity(tn)).sum())})")
    print(f"   decision (registered): {'rebuild the test files from normalized text' if held else 'keep the raw test text'}")


def section_f() -> None:
    rule(); print("F. E22e / H22e: question-answer specificity (Qwen3-8B-Base, mean log-prob of the first 256 answer tokens)"); rule()
    z = np.load(RUNS / "E22" / "specificity.npz")
    order = {"Clear Reply": 2, "Ambivalent": 1, "Clear Non-Reply": 0}
    print(f"   clarity labels {CLARITY_LABELS}")
    for name, df in (("train", TRAIN), ("dev", DEV)):
        pmi_c = z[f"{name}_own"] - z[f"{name}_rand"].mean(1)
        pmi_0 = z[f"{name}_own"] - z[f"{name}_null"]
        lvl = np.array([order[CLARITY_LABELS[c]] for c in encode_clarity(df["clarity_label"].tolist())])
        ok = np.isfinite(pmi_c)
        rho = spearmanr(pmi_c[ok], lvl[ok]).statistic
        rng = np.random.default_rng(22)
        idx = np.where(ok)[0]
        boot = [spearmanr(pmi_c[b], lvl[b]).statistic for b in (rng.choice(idx, len(idx)) for _ in range(2000))]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        auc = roc_auc_score(lvl[ok] == 2, pmi_c[ok])
        means = " / ".join(f"{np.mean(pmi_c[ok & (lvl == v)]):+.3f}" for v in (2, 1, 0))
        print(f"   {name:5} n {ok.sum():4}: PMI_c mean Reply / Ambivalent / Non-Reply = {means}; Spearman rho {rho:+.3f} "
              f"[{lo:+.3f}, {hi:+.3f}]; AUROC Clear Reply vs rest {auc:.3f}; PMI_0 mean {np.nanmean(pmi_0):+.3f}")
        if name == "train":
            print(f"   H22e (rho > 0 with the 95% interval above 0, on train): {'HELD' if lo > 0 else 'FAILED'}")
            y = encode_evasion(df["evasion_label"].tolist())
            print("   train PMI_c by evasion label: " + ", ".join(
                f"{EVASION_LABELS[c][:10]} {np.mean(pmi_c[ok & (y == c)]):+.3f}" for c in range(N_EVASION)))


def main() -> None:
    print("E19 A/B, E21a, E22a/b/e: registered checks on the inference-lane outputs (2026-10-29)\n")
    for f in (section_a, section_b, section_c, section_d, section_e, section_f):
        f()
        print()


if __name__ == "__main__":
    main()
