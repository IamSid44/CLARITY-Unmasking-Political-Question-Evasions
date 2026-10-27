#!/usr/bin/env python3
"""Codabench files for the cascade systems (E19 H), in evaluation.make_submission's format.

    python -m evaluation.make_cascade_submission                 # CPU; writes submissions/E19_<system>/

Systems:
  * m1        -- the 10-seed M1 (Q8_fullq_lora_12ep) with logit adjustment, tau fitted on ALL of dev.
  * accuracy  -- M1 + M3 at δ = 0.3: (lam, tau, tau_M1) fitted on ALL of dev by cascade.fuse.fit_system; the
                 deferral threshold is the 0.7 quantile of u = 1 - top q on the train slice, as in cascade.fuse;
                 M3's test responses (tag qwen38_27b_k0, frame test) over the 10-seed top 4.
  * cost      -- one distilled student (Q8_distill_a05, --student, default seed 0, chosen before looking at
                 test) + M3 at δ = 0.5, fitted the same way. Needs M3 test responses for the deferred items'
                 STUDENT top 4; if any are missing, the needed (i, cand) jobs for all 10 students are written to
                 runs/M3/test_jobs_students.json and no cost-point submission is made.
Each folder holds subtask2_evasion/prediction.zip, subtask1_clarity/prediction.zip (leaf -> official clarity)
and manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter

import numpy as np

from cascade.fuse import LOGPI, PRIOR, TAUS, decide, fit_system, sp
from decision_rules.decide import fit_tau
from evaluation.make_submission import N_TEST, OUT, write_zip
from preregistered_analyses.e19_lib import M1, STUDENT, TAG, defer_mask, m1_npz, m3_records, nested, Grid, system_inputs
from qevasion.labels import EVASION_LABELS, OFFICIAL_CLARITY_OF_LEAF, encode_clarity, leaf_to_official_clarity
from qevasion.loader import dev_reference_mask
from qevasion.paths import ROOT, RUNS
from qevasion.scoring import score_subtask1, score_subtask2

GOLD = dev_reference_mask(sp.dev)
CLAR = encode_clarity(sp.dev["clarity_label"].tolist())
JOBS_OUT = RUNS / "M3" / "test_jobs_students.json"


def test_lookup() -> dict:
    return {(r["i"], tuple(r["cand"])): r["q"] for r in m3_records(TAG) if r["frame"] == "test"}


def dev_cv(m: dict, delta: float) -> dict:
    """The system's held-out dev score (nested CV split 0, as E18), for the manifest."""
    g = Grid(m["lp_dev"], m["lq_dev"], defer_mask(m["u_dev"], m["u_val"], delta))
    p = nested(g, GOLD, 0, m1_only=delta == 0)
    return {"dev_s2_nested_cv_split0": score_subtask2(p, GOLD).macro_f1,
            "dev_s1_nested_cv_split0": score_subtask1(leaf_to_official_clarity(p), CLAR).macro_f1}


def predict(m: dict, delta: float, look: dict) -> tuple[np.ndarray | None, dict, list]:
    """Fit on all of dev, predict test. Returns (predictions or None, info, missing (i, cand) pairs)."""
    lp_test = np.log(m["pbar_test"] + 1e-12)
    every = np.arange(len(GOLD))
    if delta == 0:
        tau1, fit = fit_tau(m["pbar_dev"], GOLD, PRIOR, TAUS)
        return (lp_test - tau1 * LOGPI).argmax(1), {"tau_m1": tau1, "dev_in_sample_s2": fit}, []
    d_dev = defer_mask(m["u_dev"], m["u_val"], delta)
    _, fit, lam, tau, tau1 = fit_system(m["lp_dev"], m["lq_dev"], m["pbar_dev"], d_dev, GOLD, every)
    thr = float(np.quantile(m["u_val"], 1 - delta)) if delta < 1 else -np.inf
    d_test = m["u_test"] >= thr
    lq = np.full((N_TEST, len(EVASION_LABELS)), -np.inf)
    missing = []
    for i in np.where(d_test)[0]:
        c = tuple(int(x) for x in m["cand_test"][i])
        q = look.get((int(i), c))
        if q is None:
            missing.append((int(i), list(c)))
        else:
            lq[i, list(c)] = np.log(np.maximum(q, 1e-12))
    info = {"delta": delta, "lam": float(lam), "tau": float(tau), "tau_m1": float(tau1), "dev_in_sample_s2": fit,
            "slice_threshold_u": thr, "dev_defer_rate": float(d_dev.mean()), "test_defer_rate": float(d_test.mean()),
            "test_m3_calls": int(d_test.sum())}
    if missing:
        return None, info, missing
    m1 = (lp_test - tau1 * LOGPI).argmax(1)
    return decide(lp_test, lq, d_test, lam, tau, m1), info, []


def write(tag: str, idx: np.ndarray, info: dict) -> dict:
    assert idx.shape == (N_TEST,), idx.shape
    evasion = [EVASION_LABELS[int(i)] for i in idx]
    clarity = [OFFICIAL_CLARITY_OF_LEAF[e] for e in evasion]
    base = OUT / tag
    z2 = write_zip(evasion, base / "subtask2_evasion")
    z1 = write_zip(clarity, base / "subtask1_clarity")
    manifest = {**info, "test_distribution_subtask2": dict(Counter(evasion).most_common()),
                "test_distribution_subtask1": dict(Counter(clarity).most_common()),
                "sha256": {str(z.relative_to(ROOT)): hashlib.sha256(z.read_bytes()).hexdigest()[:16] for z in (z1, z2)}}
    (base / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def build_all(student: int = 0, verbose: bool = True) -> dict:
    look = test_lookup()
    m1 = system_inputs(M1, tuple(range(10)))
    z = m1_npz()
    assert np.allclose(m1["pbar_test"], z["test_pbar"]) and np.allclose(m1["q_test"], z["test_q"]), \
        "rebuilt 10-seed test inputs differ from cascade_inputs.npz"
    out = {}
    say = print if verbose else (lambda *_a, **_k: None)
    common = {"m3_tag": TAG, "m1_run": M1, "dev_fit": "all 308 dev items (3-annotator reference sets)",
              "routing": "u = 1 - top calibrated q; threshold = (1 - delta) quantile of u on the train slice"}
    preds = {}
    for name, m, delta, comp in (("E19_m1_10seed", m1, 0.0, "10-seed M1 + logit adjustment"),
                                 ("E19_accuracy_point", m1, 0.3, "10-seed M1 + M3 (qwen38_27b_k0), delta 0.3")):
        p, info, missing = predict(m, delta, look)
        assert not missing, f"{name}: {len(missing)} deferred test items have no M3 record"
        info = {"system": comp, **common, **info, **dev_cv(m, delta), "seeds": list(m["seeds"])}
        out[name] = write(name, p, info)
        preds[name] = p
        say(f"   {name:<22} {comp}; " + ", ".join(f"{k} {info[k]:.3f}" if isinstance(info[k], float) else f"{k} {info[k]}"
                                                   for k in ("tau_m1", "lam", "tau", "test_defer_rate", "test_m3_calls",
                                                             "dev_s2_nested_cv_split0") if k in info))
        say(f"      -> submissions/{name}/ ; test S2 labels {dict(Counter(EVASION_LABELS[i][:10] for i in p).most_common())}")
    a, b = preds["E19_m1_10seed"], preds["E19_accuracy_point"]
    say(f"   accuracy point changes {int((a != b).sum())} of {N_TEST} test labels relative to M1 "
        f"(S1 labels: {int((leaf_to_official_clarity(a) != leaf_to_official_clarity(b)).sum())})")

    # ---- cost point
    need, per_student = set(), {}
    for s in range(10):
        m = system_inputs(STUDENT, (s,))
        _, info, missing = predict(m, 0.5, look)
        per_student[s] = (info, missing)
        need |= {(i, tuple(c)) for i, c in missing}
    info, missing = per_student[student]
    if not missing:
        m = system_inputs(STUDENT, (student,))
        p, info, _ = predict(m, 0.5, look)
        info = {"system": f"student {STUDENT} seed {student} + M3, delta 0.5", **common, "m1_run": STUDENT,
                **info, **dev_cv(m, 0.5), "seeds": [student]}
        name = f"E19_cost_point_student{student}"
        out[name] = write(name, p, info)
        say(f"   {name}: written; test defer rate {info['test_defer_rate']:.3f}")
    if need:
        jobs = [{"split": "test_students", "frame": "test", "i": i, "cand": list(c)} for i, c in sorted(need)]
        JOBS_OUT.write_text(json.dumps(jobs, indent=1))
        n_items = len({i for i, _ in need})
        say(f"   cost point: M3 has no test record for {len(per_student[student][1])} of student {student}'s "
            f"{per_student[student][0]['test_m3_calls']} deferred test items (its top 4 differs from the 10-seed top 4).\n"
            f"   Needed for all 10 students: {len(need)} distinct (item, top-4) pairs over {n_items} test items, written to\n"
            f"   {JOBS_OUT.relative_to(ROOT)} (a JSON list; cascade.adjudicate --jobs-file reads one JSON object per\n"
            f"   line, so convert with: python -c \"import json;[print(json.dumps(j)) for j in json.load(open('...json'))]\").\n"
            f"   Per student: " + ", ".join(f"s{s} {len(v[1])}/{v[0]['test_m3_calls']}" for s, v in per_student.items())
            + f"\n   At {0.151:.3f} GPU-s/call that is ~{len(need) * 0.151 / 60:.1f} GPU-min on GPU 0 (vLLM, Qwen3.8-27B-FP8, k = 0).")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--student", type=int, default=0, help="student seed for the cost point")
    a = p.parse_args()
    build_all(a.student)


if __name__ == "__main__":
    main()
