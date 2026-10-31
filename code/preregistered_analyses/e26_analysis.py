#!/usr/bin/env python3
"""E26 analysis (registered 2026-10-30, docs/02 E26): does the instrument survive reduced precision, and what does
it save? Every quantized run is compared with a 16-bit run of the same models on the SAME GPU type.

    python -m preregistered_analyses.e26_analysis > ../docs/raw/E26_analysis.txt     # CPU

Sites: rtxpro (RTX PRO 6000, Blackwell: bf16 vs native FP8, plus the 27B for the full system) and a4000 (RTX A4000,
Ampere, local: bf16 split over two cards vs NF4), whichever exist under runs/E26.
"""

from __future__ import annotations

import json

import numpy as np

from decision_rules.decide import make_rules, nested_cv
from qevasion.labels import leaf_to_official_clarity
from qevasion.loader import DATA_CACHE, dev_reference_mask, load_qevasion
from qevasion.paths import RUNS
from qevasion.scoring import score_subtask1, score_subtask2

from audit.system import FROZEN, calibration, decide, top4

E26 = RUNS / "E26"
SP = load_qevasion(DATA_CACHE)
GOLD = dev_reference_mask(SP.dev)
from qevasion.labels import encode_clarity  # noqa: E402

CLAR = encode_clarity(SP.dev["clarity_label"].tolist())
FRAMES = ("slice", "dev_own", "test_own")
M1_COST, M3_COST, CALLS = 0.522, 0.151, 0.357  # GPU-s per item (E19 Tier 0 scale) and the accuracy point's calls
F = json.loads(FROZEN.read_text())
CALIB, *_ = calibration()
PRIOR = None


def rule(t: str) -> None:
    print("\n" + "=" * 110 + f"\n{t}\n" + "=" * 110)


def seeds(site, prec, frame):
    return np.stack([np.load(E26 / site / prec / f"seed{k}" / f"{frame}.npy") for k in range(10)])


def m1_system_s2(pbar) -> tuple[float, float]:
    """The M1 system's dev S2: logit adjustment with tau fitted by nested CV (10 splits), as E14/E23 report it."""
    from cascade.fuse import PRIOR as PRI

    r1 = make_rules(PRI)["R1_logit_adjusted"]
    runs = [nested_cv(r1, pbar, GOLD, 5, cs) for cs in range(10)]
    return runs[0]["out_of_fold_macro_f1"], float(np.mean([r["out_of_fold_macro_f1"] for r in runs]))


def site_report(site: str) -> dict:
    precs = sorted(p.name for p in (E26 / site).iterdir() if p.is_dir() and (p / "seed9").exists())
    if "bf16" not in precs:
        print(f"   {site}: no bf16 reference, skipped")
        return {}
    rep = {p: json.loads((E26 / site / p / "report.json").read_text()) for p in precs}
    print(f"   {site}: GPU {rep['bf16'].get('gpu')} x{rep['bf16'].get('n_gpus')}; precisions {precs}; "
          f"torch {rep['bf16'].get('torch')}, transformers {rep['bf16'].get('transformers')}")
    out = {}
    ref = {fr: seeds(site, "bf16", fr) for fr in FRAMES}
    saved = np.stack([np.load(RUNS / "Q8_fullq_lora_12ep" / f"seed{k}" / "dev_probs.npy") for k in range(10)])
    print(f"   bf16 on {site} vs the saved (training-time) dev probabilities: argmax agreement per seed "
          f"{(ref['dev_own'].argmax(2) == saved.argmax(2)).mean():.3f}, max |dp| {np.abs(ref['dev_own'] - saved).max():.3f}")
    for prec in [p for p in precs if p != "bf16"]:
        print(f"\n   --- {prec} vs bf16 on {site} ---")
        q = {fr: seeds(site, prec, fr) for fr in FRAMES}
        res = {}
        for fr in FRAMES:
            a_seed = (q[fr].argmax(2) == ref[fr].argmax(2)).mean()
            a_ens = (q[fr].mean(0).argmax(1) == ref[fr].mean(0).argmax(1)).mean()
            dp = np.abs(q[fr] - ref[fr]).max(2).mean()
            qa, qb = CALIB(q[fr].mean(0)), CALIB(ref[fr].mean(0))
            cand = (top4(qa) == top4(qb)).all(1).mean()
            da, db = (1 - qa.max(1)) >= F["u_threshold"], (1 - qb.max(1)) >= F["u_threshold"]
            res[fr] = {"agree_seed": a_seed, "agree_ens": a_ens, "cand_same": cand, "defer_same": (da == db).mean()}
            print(f"   {fr:<9} label agreement per seed {a_seed:.3f}, 10-seed mean {a_ens:.3f}; mean max |dp| {dp:.3f}; "
                  f"top-4 identical {cand:.3f}; routing identical {(da == db).mean():.3f}; mean u {1 - qa.max(1).mean():.3f} "
                  f"(bf16 {1 - qb.max(1).mean():.3f})")
        s_b, s_b10 = m1_system_s2(ref["dev_own"].mean(0))
        s_q, s_q10 = m1_system_s2(q["dev_own"].mean(0))
        print(f"   M1 system dev S2 (nested CV): bf16 {s_b:.3f} (10 splits {s_b10:.3f}); {prec} {s_q:.3f} ({s_q10:.3f}); "
              f"difference {s_q10 - s_b10:+.3f}")
        # full system (frozen instrument) where the 27B has scored the deferred items
        lb, lq_ = (np.load(E26 / site / "labels" / f"{p}_dev_own.npz") for p in ("bf16", prec)) \
            if (E26 / site / "labels" / f"{prec}_dev_own.npz").exists() else (None, None)
        if lb is not None:
            full = lb["scored"].sum() == lb["defer"].sum() and lq_["scored"].sum() == lq_["defer"].sum()
            sb, sq = score_subtask2(lb["pred"], GOLD).macro_f1, score_subtask2(lq_["pred"], GOLD).macro_f1
            print(f"   full system dev S2 (frozen params, in-sample): bf16 {sb:.3f}, {prec} {sq:.3f} ({sq - sb:+.3f}); "
                  f"S1 {score_subtask1(leaf_to_official_clarity(lb['pred']), CLAR).macro_f1:.3f} vs "
                  f"{score_subtask1(leaf_to_official_clarity(lq_['pred']), CLAR).macro_f1:.3f}"
                  + ("" if full else f"  [27B records incomplete: {prec} {(lq_['defer'] & ~lq_['scored']).sum()} of "
                                     f"{lq_['defer'].sum()} deferred items unscored, bf16 "
                                     f"{(lb['defer'] & ~lb['scored']).sum()} of {lb['defer'].sum()}; M1 decision kept there]"))
            m1_flip = lb["m1_pred"] != lq_["m1_pred"]
            sys_flip = lb["pred"] != lq_["pred"]
            in_def = lb["defer"][m1_flip].mean() if m1_flip.any() else float("nan")
            sm_b, sm_q = (score_subtask2(z["m1_pred"], GOLD).macro_f1 for z in (lb, lq_))
            print(f"   M1 decisions alone (frozen tau_M1): bf16 {sm_b:.3f}, {prec} {sm_q:.3f} ({sm_q - sm_b:+.3f})")
            tag = "H26-4" if prec == "fp8" else "exploratory (not registered)"
            print(f"   {tag} cascade absorbs: M1 decisions flipped {m1_flip.sum()} ({m1_flip.mean():.3f}); share of them "
                  f"among items deferred to the 27B {in_def:.3f} (dev deferral {lb['defer'].mean():.3f}); system decisions "
                  f"flipped {sys_flip.sum()} ({sys_flip.mean():.3f})")
            res["absorb"] = {"m1_flips": int(m1_flip.sum()), "in_deferred": float(in_def), "sys_flips": int(sys_flip.sum())}
            res["full_s2"] = (sb, sq)
        # speed, latency, memory
        def thr(p):
            return np.mean([rep[p][f"seed{k}"][fr]["items_per_s"] for k in range(10) for fr in FRAMES
                            if f"seed{k}" in rep[p]])
        tb, tq = thr("bf16"), thr(prec)
        lb_, lq2 = rep["bf16"]["seed0"].get("latency_batch1", {}), rep[prec]["seed0"].get("latency_batch1", {})
        mb = np.mean([rep["bf16"][f"seed{k}"]["peak_mem_gb"] for k in range(10)])
        mq = np.mean([rep[prec][f"seed{k}"]["peak_mem_gb"] for k in range(10)])
        print(f"   throughput (batch {rep[prec]['batch']}): bf16 {tb:.1f} items/s, {prec} {tq:.1f} items/s -> {tq / tb:.2f}x; "
              f"batch-1 latency p50 bf16 {lb_.get('p50_ms', float('nan')):.1f} ms, {prec} {lq2.get('p50_ms', float('nan')):.1f} ms "
              f"(p95 {lb_.get('p95_ms', float('nan')):.1f} / {lq2.get('p95_ms', float('nan')):.1f}); peak memory "
              f"{mb:.1f} GB vs {mq:.1f} GB ({mq / mb:.2f}x)")
        cost = M1_COST * tb / tq + M3_COST * CALLS
        print(f"   accuracy point cost at this speed ratio: M1 {M1_COST:.3f} -> {M1_COST * tb / tq:.3f} GPU-s/item; system "
              f"{M1_COST + M3_COST * CALLS:.3f} -> {cost:.3f} ({cost / (M1_COST + M3_COST * CALLS) - 1:+.0%})")
        res.update({"s2_m1": (s_b10, s_q10), "speed": tq / tb, "mem": mq / mb, "fp8_mode": rep[prec].get("fp8_mode")})
        out[prec] = res
    return out


def verdicts(R: dict) -> None:
    rule("Registered hypotheses")
    fp8 = R.get("rtxpro", {}).get("fp8")
    if fp8:
        h1 = fp8["dev_own"]["agree_ens"] >= 0.98 and abs(fp8["s2_m1"][1] - fp8["s2_m1"][0]) <= 0.01
        print(f"   H26-1 FP8 fidelity (10-seed agreement >= 0.98 on dev AND M1 S2 within 0.01): {'HELD' if h1 else 'FAILED'} "
              f"(agreement {fp8['dev_own']['agree_ens']:.3f}, S2 {fp8['s2_m1'][1] - fp8['s2_m1'][0]:+.3f})")
        print(f"   H26-2 FP8 speed (>= 1.3x bf16 batched throughput): {'HELD' if fp8['speed'] >= 1.3 else 'FAILED'} "
              f"({fp8['speed']:.2f}x, mode {fp8['fp8_mode']})")
        if "absorb" in fp8:
            a = fp8["absorb"]
            ok = a["in_deferred"] >= 0.5 and a["sys_flips"] < a["m1_flips"]
            print(f"   H26-4 cascade absorbs FP8 error (>= 50% of M1 flips among deferred items AND fewer system flips): "
                  f"{'HELD' if ok else 'FAILED'} ({a['in_deferred']:.3f}; {a['sys_flips']} vs {a['m1_flips']})")
    nf4 = R.get("a4000", {}).get("nf4")
    if nf4:
        print(f"   H26-3 NF4 (descriptive): 10-seed agreement {nf4['dev_own']['agree_ens']:.3f} (predicted 0.93-0.97); "
              f"M1 S2 {nf4['s2_m1'][1] - nf4['s2_m1'][0]:+.3f} (predicted -0.01 to -0.03); speed {nf4['speed']:.2f}x "
              f"(predicted <= 1.0x); memory {nf4['mem']:.2f}x (predicted <= 0.4x)")
        trig = (nf4["s2_m1"][0] - nf4["s2_m1"][1]) > 0.02 or nf4["dev_own"]["agree_ens"] < 0.95
        print(f"   QLoRA trigger (NF4 costs > 0.02 S2 or flips > 5% of labels): {'MET -> QLoRA is the named fix' if trig else 'not met'}")


def main() -> None:
    print("E26 analysis (registered 2026-10-30; docs/02 E26): reduced precision of the frozen M1, each against a 16-bit "
          "run on the same GPU type.")
    R = {}
    for site in ("rtxpro", "a4000"):
        if (E26 / site).exists():
            rule(f"Site {site}")
            R[site] = site_report(site)
    verdicts(R)


if __name__ == "__main__":
    main()
