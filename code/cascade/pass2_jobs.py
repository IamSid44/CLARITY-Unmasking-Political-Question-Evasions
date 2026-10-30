#!/usr/bin/env python3
"""The second 27B pass (2026-10-29): the (item, candidate set) pairs that E19-E, E22c and the E21 test files need.

    python -m cascade.pass2_jobs        # CPU; writes runs/E22/jobs_p2_*.jsonl and the *_tn pseudo-runs

Every candidate set is a top 4 by the slice-calibrated q of cascade.candidates.build, as in E15-E18.
  jobs_p2_dev.jsonl       dev, scored in taxonomy order (tag qwen38_27b_k0_vm):
                          * E19-E: the control students Q8_distill_a05_{m1,unif}, seeds 0-2 (system part of H-E);
                          * E22c: the ladder's fused top 4, softmax(0.5 log q_student + 0.5 log q_M1(K seeds)), for
                            student i paired with the i-th E16 subset of K = 3 and K = 5 seeds, on the items the
                            student would escalate at the largest registered e1 (0.6; slice quantile of its u).
  jobs_p2_students.jsonl  dev, the 10 students' own top 4, for the 3 shifted orders (permutation-averaged q, E19 B).
  jobs_p2_testnorm.jsonl  official test with normalized answers (E21b), all 4 orders: the 10-seed M1's top 4,
                          student 0's top 4, and the ladder's fused top 4 (student 0, K = 5 subset 0).
Normalized test probabilities enter through pseudo-runs runs/<run>_tn/seed<k>/ (the original slice and dev
probabilities, test_probs from runs/E21), so the calibration is exactly the original one.
"""

from __future__ import annotations

import json
import shutil

import numpy as np

from cascade.candidates import build
from cascade.frontier import subsets, top4
from qevasion.paths import RUNS

M1, STUDENT = "Q8_fullq_lora_12ep", "Q8_distill_a05"
CONTROLS = ("Q8_distill_a05_m1", "Q8_distill_a05_unif")
E1_MAX = 0.6
OUT = RUNS / "E22"


def calibrated(run: str, seeds: list[int]) -> dict:
    r = build(run, None, 0.05, seeds, method="fixed", k=4)
    return {s: r["q"][s] for s in ("val", "dev", "test")}


def fuse(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    z = 0.5 * np.log(qa + 1e-12) + 0.5 * np.log(qb + 1e-12)
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def pseudo_run(run: str, seeds: list[int]) -> str:
    """runs/<run>_tn/seed<k>: the original files, with test_probs replaced by the normalized-test probabilities."""
    name = f"{run}_tn"
    for k in seeds:
        src, dst = RUNS / run / f"seed{k}", RUNS / name / f"seed{k}"
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("val_probs.npy", "val_index.npy", "dev_probs.npy", "metrics.json", "config.json"):
            shutil.copy(src / f, dst / f)
        shutil.copy(RUNS / "E21" / run / f"seed{k}" / "test_norm_probs.npy", dst / "test_probs.npy")
    return name


def write(path, jobs: list[dict]) -> None:
    seen, out = set(), []
    for j in jobs:
        key = (j["frame"], j["i"], tuple(j["cand"]))
        if key not in seen:
            seen.add(key)
            out.append(j)
    path.write_text("".join(json.dumps(j) + "\n" for j in out))
    print(f"{path.name}: {len(out)} distinct (item, candidates) pairs")


def jobs_of(split: str, frame: str, q: np.ndarray, rows=None) -> list[dict]:
    rows = range(len(q)) if rows is None else rows
    c = top4(q)
    return [{"split": split, "frame": frame, "i": int(i), "cand": [int(x) for x in c[i]]} for i in rows]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dev_jobs, stud_jobs, tn_jobs = [], [], []
    for run in CONTROLS:
        for k in range(3):
            dev_jobs += jobs_of(f"p2_{run[-4:]}_s{k}", "dev", calibrated(run, [k])["dev"])
    subs = {K: subsets(ks=(K,)) for K in (3, 5)}
    for i in range(10):
        qs = calibrated(STUDENT, [i])
        stud_jobs += jobs_of(f"stud_s{i}", "dev", qs["dev"])
        u_dev, u_val = 1 - qs["dev"].max(1), 1 - qs["val"].max(1)
        rows = np.where(u_dev >= np.quantile(u_val, 1 - E1_MAX))[0]
        for K in (3, 5):
            qm = calibrated(M1, list(subs[K][i]))
            dev_jobs += jobs_of(f"ladder_s{i}_k{K}", "dev", fuse(qs["dev"], qm["dev"]), rows)
    m1_tn = pseudo_run(M1, list(range(10)))
    st_tn = pseudo_run(STUDENT, [0])
    q_m1 = calibrated(m1_tn, list(range(10)))["test"]
    q_st = calibrated(st_tn, [0])["test"]
    q_k5 = calibrated(pseudo_run(M1, list(subs[5][0])), list(subs[5][0]))["test"]
    tn_jobs += jobs_of("tn_m1", "test", q_m1) + jobs_of("tn_stud0", "test", q_st) + jobs_of("tn_ladder", "test", fuse(q_st, q_k5))
    write(OUT / "jobs_p2_dev.jsonl", dev_jobs)
    write(OUT / "jobs_p2_students.jsonl", stud_jobs)
    write(OUT / "jobs_p2_testnorm.jsonl", tn_jobs)
    (OUT / "pass2_subsets.json").write_text(json.dumps({f"K{K}": [list(s) for s in subs[K]] for K in subs}))


if __name__ == "__main__":
    main()
