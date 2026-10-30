#!/usr/bin/env python3
"""E25: the frozen instrument -- the E23 accuracy point (10-seed M1 + order-averaged 27B, delta = 0.3) with every
parameter fixed in advance, applied unchanged to new frames.

    python -m audit.system freeze             # laptop, CPU: fit once on dev, write experiments/e25/frozen_system.json
    python -m audit.system jobs  'dev_*,...'  # VM, CPU: route each frame's items; 27B jobs for the deferred ones
    python -m audit.system label 'dev_*,...'  # VM/laptop, CPU: final labels from M1 + the 27B records

Frozen (no parameter is refitted on E25 data):
  * calibration of M1's ensemble (tau_C, temperature T) and the top-4 candidates: cascade.candidates on the train
    slice, method fixed, k = 4 (the E23 inputs; recomputed on the VM and asserted equal);
  * routing: defer iff u = 1 - max q >= the 70% quantile of u on the train slice (delta = 0.3);
  * decisions: M1's own tau_m1 for kept items; (lam, tau) for deferred items, fitted ONCE on all 308 dev items
    with the VM's order-averaged 27B records (qwen38_27b_k0_vm + shift1-3), as E23's averaged accuracy point.
The dev nested-CV numbers (0.650 / 0.639) remain the honest accuracy estimate; the full-dev fit only fixes the
instrument. The 27B records for E25 are written under tags e25_<order>.
"""

from __future__ import annotations

import argparse
import fnmatch
import json

import numpy as np

from qevasion.labels import N_EVASION, encode_evasion
from qevasion.loader import DATA_CACHE, load_qevasion
from qevasion.paths import ROOT, RUNS

FROZEN = ROOT / "code" / "experiments" / "e25" / "frozen_system.json"
M1 = "Q8_fullq_lora_12ep"
DELTA = 0.3
ORDERS = ("taxonomy", "shift1", "shift2", "shift3")
E25 = RUNS / "E25"


def tag_of(order: str) -> str:
    return f"e25_{order}"


def prior() -> np.ndarray:
    y = encode_evasion(load_qevasion(DATA_CACHE).train["evasion_label"].tolist())
    return np.bincount(y, minlength=N_EVASION) / len(y)


def calibration():
    """(calib function p-bar -> q, meta, u on the train slice): exactly cascade.candidates.build's."""
    from cascade.candidates import build, normalise, temperature
    from decision_rules.decide import logit_adjust

    r = build(M1, None, 0.05, list(range(10)), method="fixed", k=4)
    m, pi = r["meta"], prior()
    calib = lambda p: temperature(normalise(logit_adjust(p, pi, m["tau_c"])), m["temperature"])  # noqa: E731
    return calib, m, 1 - r["q"]["val"].max(1), r


def top4(q: np.ndarray) -> np.ndarray:
    return np.sort(np.argsort(-q, axis=1, kind="stable")[:, :4], axis=1)


def decide(lp: np.ndarray, lq: np.ndarray, defer: np.ndarray, f: dict) -> np.ndarray:
    logpi = np.log(prior())
    m1 = (lp - f["tau_m1"] * logpi).argmax(1)
    in_c = np.isfinite(lq)
    s = np.where(in_c, f["lam"] * lp + (1 - f["lam"]) * np.where(in_c, lq, 0.0) - f["tau"] * logpi, -np.inf)
    return np.where(defer, s.argmax(1), m1)


def freeze() -> None:
    from preregistered_analyses.e19_lib import Grid, nested
    from preregistered_analyses.e23_systems import GOLD, LOOK, scores_of

    calib, meta, u_val, r = calibration()
    thr = float(np.quantile(u_val, 1 - DELTA))
    lp = np.log(r["pbar"]["dev"] + 1e-12)
    cand = top4(r["q"]["dev"])
    u = 1 - r["q"]["dev"].max(1)
    defer = u >= thr
    lq = np.full((len(lp), N_EVASION), -np.inf)
    for i, c in enumerate(cand):
        lq[i, c] = np.log(np.maximum(LOOK["averaged"][(i, tuple(int(x) for x in c))], 1e-12))
    lq[~defer] = -np.inf
    g = Grid(lp, lq, defer)
    pred, par = g.fit(GOLD, np.arange(len(lp)))
    f = {"run": M1, "seeds": list(range(10)), "tau_c": meta["tau_c"], "temperature": meta["temperature"],
         "delta": DELTA, "u_threshold": thr, **par, "m3_records": "qwen38_27b_k0_vm + shift1-3, averaged",
         "full_dev_fit_s2_s1": list(scores_of(pred)),
         "nested_cv_s2_split0": scores_of(nested(g, GOLD, 0))[0],
         "nested_cv_s2_10splits": float(np.mean([scores_of(nested(g, GOLD, s))[0] for s in range(10)])),
         "dev_calls": float(defer.mean())}
    assert (decide(lp, lq, defer, f) == pred).all(), "decide() does not reproduce the Grid's full-dev predictions"
    FROZEN.parent.mkdir(parents=True, exist_ok=True)
    FROZEN.write_text(json.dumps(f, indent=2))
    print(json.dumps(f, indent=2))


def frames_of(glob: str) -> list[str]:
    return sorted({p.name for p in (E25 / "m1").iterdir() if p.is_dir()
                   and any(fnmatch.fnmatch(p.name, g) for g in glob.split(","))})


def m1_inputs(frame: str, calib) -> dict:
    P = np.stack([np.load(E25 / "m1" / frame / f"seed{k}.npy") for k in range(10)])
    pbar = P.mean(0)
    q = calib(pbar)
    return {"pbar": pbar, "q": q, "cand": top4(q), "u": 1 - q.max(1), "seed_argmax": P.argmax(2)}


def jobs(glob: str) -> None:
    f = json.loads(FROZEN.read_text())
    calib, meta, u_val, _ = calibration()
    assert abs(meta["tau_c"] - f["tau_c"]) < 1e-9 and abs(meta["temperature"] - f["temperature"]) < 1e-6, \
        "the VM's calibration differs from the frozen one"
    assert abs(float(np.quantile(u_val, 1 - DELTA)) - f["u_threshold"]) < 1e-9
    out, n = [], {}
    for fr in frames_of(glob):
        z = m1_inputs(fr, calib)
        defer = z["u"] >= f["u_threshold"]
        n[fr] = (int(defer.sum()), len(defer))
        out += [{"split": "e25", "frame": fr, "i": int(i), "cand": [int(c) for c in z["cand"][i]]}
                for i in np.where(defer)[0]]
    (E25 / "jobs_27b.jsonl").write_text("".join(json.dumps(j) + "\n" for j in out))
    print(f"27B jobs: {len(out)} deferred items; per frame (deferred, total): {n}")


def records(frame: str) -> dict:
    """(i, cand) -> q averaged over the four orders (only items scored in all four)."""
    per = []
    for o in ORDERS:
        d = {}
        p = RUNS / "M3" / tag_of(o) / "responses.jsonl"
        if p.exists():
            for line in p.read_text().splitlines():
                r = json.loads(line)
                if r["frame"] == frame:
                    d[(r["i"], tuple(r["cand"]))] = np.asarray(r["q"])
        per.append(d)
    return {k: np.mean([d[k] for d in per], 0) for k in per[0] if all(k in d for d in per)}


def label(glob: str) -> None:
    f = json.loads(FROZEN.read_text())
    calib, *_ = calibration()
    (E25 / "labels").mkdir(parents=True, exist_ok=True)
    for fr in frames_of(glob):
        z = m1_inputs(fr, calib)
        defer = z["u"] >= f["u_threshold"]
        look = records(fr)
        lq = np.full((len(defer), N_EVASION), -np.inf)
        miss = 0
        for i in np.where(defer)[0]:
            q = look.get((int(i), tuple(int(c) for c in z["cand"][i])))
            if q is None:
                miss += 1
                continue
            lq[i, z["cand"][i]] = np.log(np.maximum(q, 1e-12))
        ok = defer & np.isfinite(lq).any(1)
        pred = decide(np.log(z["pbar"] + 1e-12), lq, ok, f)
        np.savez_compressed(E25 / "labels" / f"{fr}.npz", pred=pred, m1_pred=decide(np.log(z["pbar"] + 1e-12), lq,
                            np.zeros_like(ok), f), defer=defer, scored=ok, **z)
        print(f"{fr}: {len(pred)} items, deferred {defer.sum()}, 27B records missing {miss}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=("freeze", "jobs", "label"))
    p.add_argument("frames", nargs="?", default="*")
    a = p.parse_args()
    {"freeze": lambda: freeze(), "jobs": lambda: jobs(a.frames), "label": lambda: label(a.frames)}[a.what]()


if __name__ == "__main__":
    main()
