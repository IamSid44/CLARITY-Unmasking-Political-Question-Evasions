#!/usr/bin/env python3
"""E16: the accuracy-vs-cost frontier of the cascade over the number of M1 seeds K and the deferral rate δ.

    python -m cascade.frontier jobs                       # runs/M3/frontier_jobs.jsonl: (dev item, top-4) pairs
    # score them with M3 (local, reuses identical pairs already scored under the same tag):
    $VLLM_PY -m cascade.adjudicate --backend vllm --model Qwen/Qwen3.8-27B-FP8 --tag qwen38_27b_k0 --k 0 \
        --splits "" --jobs-file ../runs/M3/frontier_jobs.jsonl
    python -m cascade.frontier analyse > ../docs/raw/E16_frontier.txt

Seed subsets of Q8_fullq_lora_12ep: K = 1 (each seed), K = 2, 3, 5 (10 random subsets each, seed 16), K = 10.
Each subset gets its own slice-fitted calibration, top-4 candidates and routing u = 1 - top q (as M1 in E14),
then the E15 system: M1's own tau, fused (lam, tau) for deferred items, 5-fold nested CV on dev (split 0).
Cost per item = K x (GPU-s of one M1 seed) + realised calls x (GPU-s of one M3 call), from runs/M3/cost.json
(written by cascade.cost).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from cascade.candidates import build
from cascade.fuse import DELTAS, LOGPI, PRIOR, fit_system, sp
from decision_rules.decide import fit_tau, nested_cv
from qevasion.labels import N_EVASION, encode_clarity, leaf_to_official_clarity
from qevasion.loader import dev_reference_mask
from qevasion.paths import RUNS
from qevasion.scoring import score_subtask1, score_subtask2

RUN = "Q8_fullq_lora_12ep"
KS = (1, 2, 3, 5, 10)
N_SUBSETS, SUBSET_SEED = 10, 16
JOBS = RUNS / "M3" / "frontier_jobs.jsonl"  # E16; E17 writes frontier_jobs_<run>.jsonl
TAG = "qwen38_27b_k0"
TAUS = np.linspace(-1.0, 2.0, 61)


def subsets(ks=KS, seeds=tuple(range(10))) -> list[tuple[int, ...]]:
    """K = 1: every seed; K = all: one subset; otherwise N_SUBSETS random subsets (seed SUBSET_SEED)."""
    rng = np.random.default_rng(SUBSET_SEED)
    seeds = tuple(seeds)
    out: list[tuple[int, ...]] = []
    for k in ks:
        if k == 1:
            out += [(s,) for s in seeds]
        elif k == len(seeds):
            out.append(seeds)
        else:
            seen: set[tuple[int, ...]] = set()
            while len(seen) < N_SUBSETS:
                seen.add(tuple(sorted(rng.choice(seeds, k, replace=False).tolist())))
            out += sorted(seen)
    return out


def m1_inputs(seeds: tuple[int, ...], run: str = RUN) -> dict:
    r = build(run, None, 0.05, list(seeds), method="fixed", k=4)
    return {"pbar": r["pbar"]["dev"], "q_dev": r["q"]["dev"], "q_val": r["q"]["val"]}


def top4(q: np.ndarray) -> np.ndarray:
    return np.sort(np.argsort(-q, axis=1, kind="stable")[:, :4], axis=1)


def write_jobs(run: str, subs: list[tuple[int, ...]]) -> None:
    pairs = set()
    for s in subs:
        for i, c in enumerate(top4(m1_inputs(s, run)["q_dev"])):
            pairs.add((i, tuple(int(x) for x in c)))
    path = JOBS if run == RUN else RUNS / "M3" / f"frontier_jobs_{run}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for i, c in sorted(pairs):
            f.write(json.dumps({"split": "frontier", "frame": "dev", "i": i, "cand": list(c)}) + "\n")
    print(f"{run}: {len(subs)} seed subsets; {len(pairs)} distinct (dev item, top-4) pairs -> {path}")


def m3_lookup(tag: str = TAG) -> dict:
    out = {}
    for line in (RUNS / "M3" / tag / "responses.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["frame"] == "dev":
            out[(r["i"], tuple(r["cand"]))] = r["q"]
    return out


def analyse(run: str, subs: list[tuple[int, ...]], splits: int, out: str = "") -> None:
    gold = dev_reference_mask(sp.dev)
    clar = encode_clarity(sp.dev["clarity_label"].tolist())
    n = len(gold)
    look = m3_lookup()
    cost = json.loads((RUNS / "M3" / "cost.json").read_text())
    c_m1, c_m3 = cost["m1_seed_gpu_s_per_item"], cost["m3_call_gpu_s"]
    s1 = lambda p: score_subtask1(leaf_to_official_clarity(p), clar).macro_f1  # noqa: E731
    rows = defaultdict(list)
    for s in subs:
        m = m1_inputs(s, run)
        lp = np.log(m["pbar"] + 1e-12)
        cand = top4(m["q_dev"])
        lq = np.full((n, N_EVASION), -np.inf)
        for i, c in enumerate(cand):
            q = look.get((i, tuple(int(x) for x in c)))
            if q is None:
                raise SystemExit(f"M3 has not scored dev item {i} with candidates {c.tolist()}: run the jobs first")
            lq[i, c] = np.log(np.maximum(q, 1e-12))
        u_dev, u_val = 1 - m["q_dev"].max(1), 1 - m["q_val"].max(1)

        def m1_rule(_p, g, tr, te, pbar=m["pbar"], lp=lp):
            tau, fit = fit_tau(pbar[tr], g[tr], PRIOR, TAUS)
            return (lp - tau * LOGPI).argmax(1)[te], fit

        for delta in DELTAS:
            if delta == 0:
                runs = [nested_cv(m1_rule, lp, gold, 5, cs) for cs in range(splits)]
                defer_rate = 0.0
            else:
                defer = np.ones(n, bool) if delta == 1 else u_dev >= np.quantile(u_val, 1 - delta)

                def rule(_p, g, tr, te, defer=defer, lp=lp, lq=lq, pbar=m["pbar"]):
                    pred, fit, *_ = fit_system(lp, lq, pbar, defer, g, tr)
                    return pred[te], fit

                runs = [nested_cv(rule, lp, gold, 5, cs) for cs in range(splits)]
                defer_rate = float(defer.mean())
            pred = runs[0]["pooled_predictions"]
            rows[(len(s), delta)].append({"seeds": list(s),
                "s2": float(np.mean([r["out_of_fold_macro_f1"] for r in runs])),
                "s1": s1(pred), "inset": float(gold[np.arange(n), pred].mean()), "calls": defer_rate,
                "gpu_s": len(s) * c_m1 + defer_rate * c_m3})
        print(f"  done subset {s}", flush=True)
    print(f"\n{run} frontier: M1 seed {c_m1:.3f} GPU-s/item, M3 call {c_m3:.3f} GPU-s/call (runs/M3/cost.json); "
          f"S2 = mean over seed subsets of the nested-CV score ({splits} split(s) each)\n")
    print(f"{'K':>3} {'δ':>4} {'n':>3} {'calls':>6} {'GPU-s':>7} {'S2 mean':>8} {'± sd':>6} {'S1':>6} {'in-set':>7}")
    table = []
    for (k, d), v in sorted(rows.items()):
        s2 = np.array([x["s2"] for x in v])
        rec = {"K": k, "delta": d, "n_subsets": len(v), "calls": np.mean([x["calls"] for x in v]),
               "gpu_s": np.mean([x["gpu_s"] for x in v]), "s2": s2.mean(), "s2_sd": s2.std(ddof=1) if len(v) > 1 else 0.0,
               "s1": np.mean([x["s1"] for x in v]), "inset": np.mean([x["inset"] for x in v])}
        table.append(rec)
        print(f"{k:>3} {d:>4.1f} {len(v):>3} {rec['calls']:>6.3f} {rec['gpu_s']:>7.3f} {rec['s2']:>8.3f} "
              f"{rec['s2_sd']:>6.3f} {rec['s1']:>6.3f} {rec['inset']:>7.3f}")
    pareto = [r for r in table if not any(o["gpu_s"] < r["gpu_s"] and o["s2"] >= r["s2"] for o in table)]
    print("\nPareto points (no other point is both cheaper and at least as good on S2):")
    for r in sorted(pareto, key=lambda r: r["gpu_s"]):
        print(f"   K={r['K']:<2} δ={r['delta']:.1f}: {r['gpu_s']:.3f} GPU-s/item, S2 {r['s2']:.3f}, S1 {r['s1']:.3f}")
    (RUNS / "M3" / f"frontier_{run}{out}.json").write_text(json.dumps({"table": table, "per_subset": {
        f"K{k}_d{d}": v for (k, d), v in rows.items()}}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=("jobs", "analyse"))
    p.add_argument("--splits", type=int, default=1, help="nested-CV splits per subset (K = 10 in E15 used 10)")
    p.add_argument("--run", default=RUN, help="seed ensemble to use (E17: Q8_distill_a05)")
    p.add_argument("--ks", default=",".join(map(str, KS)))
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--out", default="", help="suffix of runs/M3/frontier_<run><out>.json (E18: _ensembles)")
    a = p.parse_args()
    subs = subsets(tuple(int(k) for k in a.ks.split(",")), tuple(int(s) for s in a.seeds.split(",")))
    write_jobs(a.run, subs) if a.cmd == "jobs" else analyse(a.run, subs, a.splits, a.out)


if __name__ == "__main__":
    main()
