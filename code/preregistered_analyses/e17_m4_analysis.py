#!/usr/bin/env python3
"""E17 (module M4): the pre-registered analyses (docs/02_experiment_log.md, E17).

    # after the students finish: their own top-4 candidates scored by M3, then the system frontier
    python -m cascade.frontier jobs --run Q8_distill_a05 --ks 1 --seeds 0,1,2
    $VLLM_PY -m cascade.adjudicate --backend vllm --model Qwen/Qwen3.8-27B-FP8 --tag qwen38_27b_k0 --k 0 \
        --splits "" --jobs-file ../runs/M3/frontier_jobs_Q8_distill_a05.jsonl
    python -m cascade.frontier analyse --run Q8_distill_a05 --ks 1 --seeds 0,1,2
    python -m preregistered_analyses.e17_m4_analysis > ../docs/raw/E17_m4_analysis.txt

A. Single models, paired seeds: student (Q8_distill_a05) vs control (Q8_fullq_lora_12ep), dev S2 / S1 (H3).
B. Routing: mean top probability on dev and AUROC of 1 - top probability for the argmax's errors (H2).
C. Systems with M3 (from cascade.frontier, nested-CV split 0): student (K = 1) + M3 vs control (K = 1) + M3
   on the same seeds, and the midpoint to the K = 10 system (H1); the screen's decision rule.
"""

from __future__ import annotations

import json

import numpy as np

from qevasion.labels import encode_clarity, leaf_to_official_clarity
from qevasion.loader import DATA_CACHE, dev_reference_mask, load_qevasion
from qevasion.paths import RUNS
from qevasion.scoring import score_subtask1, score_subtask2

STUDENT, CONTROL = "Q8_distill_a05", "Q8_fullq_lora_12ep"
DELTAS = ("0.0", "0.1", "0.2", "0.3", "0.5", "1.0")

sp = load_qevasion(DATA_CACHE)
GOLD = dev_reference_mask(sp.dev)
CLAR = encode_clarity(sp.dev["clarity_label"].tolist())


def seeds_done(run: str) -> list[int]:
    return sorted(int(p.parent.name[4:]) for p in RUNS.glob(f"{run}/seed*/metrics.json"))


def auroc(score: np.ndarray, err: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(err, score)) if 0 < err.sum() < len(err) else float("nan")


def section_a(seeds: list[int]) -> None:
    print("A. Single models, dev, paired seeds (argmax of each model's own probabilities)\n")
    print(f"   {'seed':>4} {'student S2':>10} {'control S2':>10} {'Δ':>7}   {'student S1':>10} {'control S1':>10} {'Δ':>7}")
    d2, d1 = [], []
    for s in seeds:
        ps, pc = (np.load(RUNS / r / f"seed{s}" / "dev_probs.npy").argmax(1) for r in (STUDENT, CONTROL))
        a2, b2 = score_subtask2(ps, GOLD).macro_f1, score_subtask2(pc, GOLD).macro_f1
        a1, b1 = (score_subtask1(leaf_to_official_clarity(p), CLAR).macro_f1 for p in (ps, pc))
        d2.append(a2 - b2), d1.append(a1 - b1)
        print(f"   {s:>4} {a2:>10.3f} {b2:>10.3f} {a2 - b2:>+7.3f}   {a1:>10.3f} {b1:>10.3f} {a1 - b1:>+7.3f}")
    print(f"   mean Δ S2 {np.mean(d2):+.3f} ({sum(x > 0 for x in d2)}/{len(d2)} up), Δ S1 {np.mean(d1):+.3f} "
          f"({sum(x > 0 for x in d1)}/{len(d1)} up)\n")


def section_b(seeds: list[int]) -> None:
    print("B. Routing: confidence and how well 1 - top probability flags the argmax's errors (dev, any reference)\n")
    for run in (STUDENT, CONTROL):
        tops, aucs = [], []
        for s in seeds:
            p = np.load(RUNS / run / f"seed{s}" / "dev_probs.npy")
            err = ~GOLD[np.arange(len(p)), p.argmax(1)]
            tops.append(p.max(1).mean())
            aucs.append(auroc(1 - p.max(1), err.astype(int)))
        print(f"   {run:<20} mean top probability {np.mean(tops):.3f}; AUROC {np.mean(aucs):.3f} "
              f"(per seed {', '.join(f'{x:.3f}' for x in aucs)})")
    print()


def section_c(seeds: list[int]) -> None:
    print("C. Systems with M3 (cascade.frontier; dev S2, nested-CV split 0; K = 1 = one model's own top 4 and routing)\n")
    st = json.loads((RUNS / "M3" / f"frontier_{STUDENT}.json").read_text())["per_subset"]
    co = json.loads((RUNS / "M3" / f"frontier_{CONTROL}.json").read_text())["per_subset"]
    k10 = {d: co[f"K10_d{d}"][0] for d in DELTAS}
    pick = lambda rows: [r for r in rows if r["seeds"][0] in seeds]  # noqa: E731
    print(f"   {'δ':>4} {'student+M3':>10} {'control+M3':>10} {'Δ':>7} {'K10+M3':>7} {'midpoint':>8}   "
          f"{'student S1':>10} {'control S1':>10}   GPU-s/item (student, K10)")
    verdict = {}
    for d in DELTAS:
        s_rows, c_rows = pick(st[f"K1_d{d}"]), pick(co[f"K1_d{d}"])
        s2s, s2c = np.mean([r["s2"] for r in s_rows]), np.mean([r["s2"] for r in c_rows])
        mid = (s2c + k10[d]["s2"]) / 2
        verdict[d] = (s2s - s2c, s2s >= mid)
        print(f"   {d:>4} {s2s:>10.3f} {s2c:>10.3f} {s2s - s2c:>+7.3f} {k10[d]['s2']:>7.3f} {mid:>8.3f}   "
              f"{np.mean([r['s1'] for r in s_rows]):>10.3f} {np.mean([r['s1'] for r in c_rows]):>10.3f}   "
              f"{np.mean([r['gpu_s'] for r in s_rows]):.3f}, {k10[d]['gpu_s']:.3f}")
    gain, mid_ok = verdict["0.3"]
    if len(seeds) <= 3:
        print(f"\n   Screen rule at δ = 0.3: student + M3 beats control + M3 by {gain:+.3f} (needs ≥ +0.015) "
              f"{'or' if gain < 0.015 else 'and'} reaches the midpoint: {mid_ok}  ->  "
              f"{'run 7 more seeds' if (gain >= 0.015 or mid_ok) else 'stop: M4 is a negative result'}")
    else:
        print(f"\n   At δ = 0.3 over {len(seeds)} seeds: student + M3 - control + M3 = {gain:+.3f}; midpoint reached: {mid_ok}")


def section_d() -> None:
    """The adoption rule: student + M3 is the cost operating point if no undistilled (K, δ) point of E16 is both
    cheaper and at least as good on S2."""
    print("\nD. Adoption rule against E16's undistilled frontier (S2; GPU-s per item)\n")
    u = json.loads((RUNS / "M3" / f"frontier_{CONTROL}.json").read_text())["table"]
    s = json.loads((RUNS / "M3" / f"frontier_{STUDENT}.json").read_text())["table"]
    ok = True
    for r in s:
        dom = [(o["K"], o["delta"]) for o in u if o["gpu_s"] < r["gpu_s"] and o["s2"] >= r["s2"]]
        ok &= not dom
        print(f"   student + M3, δ = {r['delta']:.1f}: {r['gpu_s']:.3f} GPU-s, S2 {r['s2']:.3f} (± {r['s2_sd']:.3f}), "
              f"S1 {r['s1']:.3f}; dominated by: {dom or 'none'}")
    beaten = sum(any(x["gpu_s"] < o["gpu_s"] and x["s2"] >= o["s2"] for x in s) for o in u)
    allp = [dict(o, src="M1") for o in u] + [dict(o, src="student") for o in s]
    front = [r for r in allp if not any((o["gpu_s"] <= r["gpu_s"] and o["s2"] > r["s2"]) or
                                        (o["gpu_s"] < r["gpu_s"] and o["s2"] >= r["s2"]) for o in allp)]
    print(f"\n   undistilled points dominated by a student point: {beaten} of {len(u)}")
    print("   combined front (no point both cheaper and better):")
    for r in sorted(front, key=lambda r: (r["gpu_s"], -r["s2"])):
        print(f"      {r['src']:<8} K={r['K']:<2} δ={r['delta']:.1f}: {r['gpu_s']:.3f} GPU-s, S2 {r['s2']:.3f}, S1 {r['s1']:.3f}")
    print(f"\n   -> {'ADOPT student + M3 as the cost operating point' if ok else 'do not adopt: a cheaper undistilled point is as good'}")


def main() -> None:
    seeds = [s for s in seeds_done(STUDENT) if s in seeds_done(CONTROL)]
    print(f"E17 (M4) analysis -- student {STUDENT}, control {CONTROL}, paired seeds {seeds}\n")
    section_a(seeds)
    section_b(seeds)
    section_c(seeds)
    if len(seeds) > 3:
        section_d()


if __name__ == "__main__":
    main()
