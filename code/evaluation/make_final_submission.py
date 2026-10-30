#!/usr/bin/env python3
"""E23: the final Codabench files, built as registered in E19-E22 (2026-10-30).

    python -m evaluation.make_final_submission          # CPU; writes submissions/E23_<system>/

What changed from E19's files (submissions/E19_*):
  * E21: H21a held, so the official test answers are NORMALIZED (qevasion.normalize) for M1, the student and M3;
    the test probabilities come from runs/E21 through the pseudo-runs runs/<run>_tn (cascade.pass2_jobs);
  * E19 amendment: the faith gate failed, so M3 is the VM's copy of the 27B on every split (dev and test);
  * E19 arm B: permutation-averaged q (the mean over the 4 cyclic option orders) replaces single-order q.
Systems: the 10-seed M1 alone; the accuracy point (M1 x 10 + M3, delta 0.3); the cost point (student 0 + M3,
delta 0.5). Everything else is evaluation.make_cascade_submission's: fit on all of dev, slice thresholds.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

from cascade.candidates import build
from cascade.frontier import top4
from evaluation.make_cascade_submission import dev_cv, predict, write
from evaluation.make_submission import N_TEST
from preregistered_analyses.e19_lib import lq_from
from qevasion.labels import EVASION_LABELS, leaf_to_official_clarity
from qevasion.paths import RUNS

DEV_TAGS = ("qwen38_27b_k0_vm", "qwen38_27b_k0_shift1", "qwen38_27b_k0_shift2", "qwen38_27b_k0_shift3")
TEST_TAGS = ("qwen38_27b_k0_vm_tn", "qwen38_27b_k0_vm_tn_shift1", "qwen38_27b_k0_vm_tn_shift2", "qwen38_27b_k0_vm_tn_shift3")


def averaged(tags, frame: str) -> dict:
    per = []
    for t in tags:
        d = {}
        for line in (RUNS / "M3" / t / "responses.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["frame"] == frame:
                d[(r["i"], tuple(r["cand"]))] = np.asarray(r["q"])
        per.append(d)
    return {k: np.mean([d[k] for d in per], 0) for k in per[0] if all(k in d for d in per)}


def inputs(run: str, seeds, dev_look: dict) -> dict:
    """make_cascade_submission's system inputs, from a pseudo-run whose test probabilities are the normalized ones."""
    r = build(run, None, 0.05, list(seeds), method="fixed", k=4)
    m = {"seeds": list(seeds)}
    for s in ("dev", "val", "test"):
        m[f"pbar_{s}"], m[f"q_{s}"] = r["pbar"][s], r["q"][s]
        m[f"u_{s}"] = 1 - r["q"][s].max(1)
        m[f"cand_{s}"] = top4(r["q"][s])
    m["lp_dev"] = np.log(m["pbar_dev"] + 1e-12)
    m["lq_dev"], ok = lq_from(m["cand_dev"], dev_look)
    assert ok.all(), f"{run}: M3 (averaged) lacks some dev items' top 4"
    return m


def main() -> None:
    dev_look, test_look = averaged(DEV_TAGS, "dev"), averaged(TEST_TAGS, "test")
    print(f"averaged M3 records: dev {len(dev_look)}, normalized test {len(test_look)} (item, candidates) pairs")
    common = {"m3": "Qwen3.8-27B-FP8 on the JarvisLabs VM, k = 0, q averaged over 4 cyclic option orders",
              "test_text": "official test answers normalized by qevasion.normalize (E21b)",
              "dev_fit": "all 308 dev items (3-annotator reference sets)",
              "routing": "u = 1 - top calibrated q; threshold = (1 - delta) quantile of u on the train slice"}
    m1 = inputs("Q8_fullq_lora_12ep_tn", range(10), dev_look)
    st = inputs("Q8_distill_a05_tn", (0,), dev_look)
    preds = {}
    for name, m, delta, comp in (("E23_m1_10seed", m1, 0.0, "10-seed M1 + logit adjustment"),
                                 ("E23_accuracy_point", m1, 0.3, "10-seed M1 + M3, delta 0.3"),
                                 ("E23_cost_point_student0", st, 0.5, "student Q8_distill_a05 seed 0 + M3, delta 0.5")):
        p, info, missing = predict(m, delta, test_look)
        assert not missing, f"{name}: {len(missing)} deferred test items have no averaged M3 record"
        info = {"system": comp, **common, **info, **dev_cv(m, delta), "seeds": m["seeds"]}
        write(name, p, info)
        preds[name] = p
        print(f"   {name:<26} " + ", ".join(f"{k} {info[k]:.3f}" if isinstance(info[k], float) else f"{k} {info[k]}"
                                           for k in ("tau_m1", "lam", "tau", "test_defer_rate", "test_m3_calls",
                                                     "dev_s2_nested_cv_split0", "dev_s1_nested_cv_split0") if k in info))
        print(f"      test labels {dict(Counter(EVASION_LABELS[i][:10] for i in p).most_common())}")
    for a, b in (("E23_m1_10seed", "E23_accuracy_point"), ("E23_accuracy_point", "E23_cost_point_student0")):
        x, y = preds[a], preds[b]
        print(f"   {b} vs {a}: {int((x != y).sum())} of {N_TEST} test labels differ "
              f"(S1: {int((leaf_to_official_clarity(x) != leaf_to_official_clarity(y)).sum())})")
    old = RUNS.parent / "submissions" / "E19_accuracy_point" / "subtask2_evasion" / "prediction"
    if old.exists():
        prev = [line.strip() for line in old.read_text().splitlines() if line.strip()]  # labels contain spaces
        print(f"   E23 vs E19 accuracy point (raw text, lab records, single order): "
              f"{sum(a != EVASION_LABELS[b] for a, b in zip(prev, preds['E23_accuracy_point']))} of {N_TEST} labels differ")


if __name__ == "__main__":
    main()
