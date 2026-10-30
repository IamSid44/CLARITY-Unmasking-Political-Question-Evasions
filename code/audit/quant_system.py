#!/usr/bin/env python3
"""E26: route and label every precision's M1 outputs through the frozen E23 instrument (audit/system.py).

    python -m audit.quant_system jobs  --site rtxpro    # 27B jobs: deferred dev/test items of every precision
    python -m audit.quant_system label --site rtxpro    # final labels; 27B records under tags e26_<site>_<order>

Calibration (tau_C, T), the routing threshold and the fusion (tau_M1, lam, tau) are the frozen E23 values
(code/experiments/e25/frozen_system.json); nothing is refitted on quantized outputs. A site without 27B records
(the local A4000s) is labelled with M1's decision wherever the 27B would have been asked, and says so.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from audit.system import FROZEN, ORDERS, calibration, decide, top4
from qevasion.labels import N_EVASION
from qevasion.paths import RUNS

E26 = RUNS / "E26"
ADJ = ("dev_own", "test_own")


def tag_of(site: str, order: str) -> str:
    return f"e26_{site}_{order}"


def inputs(site: str, precision: str, frame: str, calib) -> dict:
    P = np.stack([np.load(E26 / site / precision / f"seed{k}" / f"{frame}.npy") for k in range(10)])
    pbar = P.mean(0)
    q = calib(pbar)
    return {"pbar": pbar, "q": q, "cand": top4(q), "u": 1 - q.max(1), "seed_argmax": P.argmax(2)}


def precisions(site: str) -> list[str]:
    return sorted(p.name for p in (E26 / site).iterdir()
                  if p.is_dir() and all((p / f"seed{k}" / "dev_own.npy").exists() for k in range(10)))


def jobs(site: str) -> None:
    f = json.loads(FROZEN.read_text())
    calib, *_ = calibration()
    seen, out = set(), []
    for prec in precisions(site):
        for fr in ("dev_own",):  # the system is evaluated on dev only (test has no labels)
            z = inputs(site, prec, fr, calib)
            for i in np.where(z["u"] >= f["u_threshold"])[0]:
                key = (fr, int(i), tuple(int(c) for c in z["cand"][i]))
                if key not in seen:
                    seen.add(key)
                    out.append({"split": "e26", "frame": fr, "i": key[1], "cand": list(key[2])})
    (E26 / site / "jobs_27b.jsonl").write_text("".join(json.dumps(j) + "\n" for j in out))
    print(f"{site}: precisions {precisions(site)}; 27B jobs (deferred, distinct candidate sets): {len(out)}")


def records(site: str, frame: str) -> dict:
    per = []
    for o in ORDERS:
        d, p = {}, RUNS / "M3" / tag_of(site, o) / "responses.jsonl"
        if p.exists():
            for line in p.read_text().splitlines():
                r = json.loads(line)
                if r["frame"] == frame:
                    d[(r["i"], tuple(r["cand"]))] = np.asarray(r["q"])
        per.append(d)
    return {k: np.mean([d[k] for d in per], 0) for k in per[0] if all(k in d for d in per)}


def label(site: str) -> None:
    f = json.loads(FROZEN.read_text())
    calib, *_ = calibration()
    (E26 / site / "labels").mkdir(parents=True, exist_ok=True)
    for prec in precisions(site):
        for fr in ADJ + ("slice",):
            z = inputs(site, prec, fr, calib)
            defer = z["u"] >= f["u_threshold"]
            look = records(site, fr) if fr in ADJ else {}
            lq = np.full((len(defer), N_EVASION), -np.inf)
            for i in np.where(defer)[0]:
                q = look.get((int(i), tuple(int(c) for c in z["cand"][i])))
                if q is not None:
                    lq[i, z["cand"][i]] = np.log(np.maximum(q, 1e-12))
            scored = defer & np.isfinite(lq).any(1)
            lp = np.log(z["pbar"] + 1e-12)
            np.savez_compressed(E26 / site / "labels" / f"{prec}_{fr}.npz", pred=decide(lp, lq, scored, f),
                                m1_pred=decide(lp, lq, np.zeros_like(scored), f), defer=defer, scored=scored, **z)
            print(f"{site} {prec} {fr}: {len(defer)} items, deferred {defer.sum()}, 27B-scored {scored.sum()}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=("jobs", "label"))
    p.add_argument("--site", required=True)
    a = p.parse_args()
    jobs(a.site) if a.what == "jobs" else label(a.site)


if __name__ == "__main__":
    main()
