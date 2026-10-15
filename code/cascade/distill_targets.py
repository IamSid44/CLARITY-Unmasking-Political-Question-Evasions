#!/usr/bin/env python3
"""E17 (M4): soft targets for retraining M1, t = (1 - alpha) * onehot(gold) + alpha * q.

    python -m cascade.distill_targets [--tag qwen38_27b_k0] [--alpha 0.5]   # -> runs/M4/targets_a<alpha>.npy

q is M3's distribution over the row's cross-fitted top 4 (zero elsewhere), from the `trainall` scoring
(cascade.adjudicate --splits trainall). The slice rows get targets too, but llm_classifier never trains on
them (they stay held out to choose the epoch).
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from qevasion.labels import N_EVASION, encode_evasion
from qevasion.loader import DATA_CACHE, load_qevasion
from qevasion.paths import RUNS

CROSSFIT_SPLITS = ("trainall", "train50", "screen")  # train rows scored with cross-fitted candidates


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="qwen38_27b_k0")
    p.add_argument("--alpha", type=float, default=0.5)
    a = p.parse_args()
    y = encode_evasion(load_qevasion(DATA_CACHE).train["evasion_label"].tolist())
    q = np.full((len(y), N_EVASION), np.nan)
    for line in (RUNS / "M3" / a.tag / "responses.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["frame"] == "train" and r["split"] in CROSSFIT_SPLITS:
            row = np.zeros(N_EVASION)
            row[r["cand"]] = r["q"]
            q[r["i"]] = row
    missing = np.isnan(q).any(1)
    if missing.any():
        raise SystemExit(f"{missing.sum()} train rows have no M3 distribution: run adjudicate --splits trainall")
    onehot = np.eye(N_EVASION)[y]
    t = (1 - a.alpha) * onehot + a.alpha * q
    out = RUNS / "M4" / f"targets_a{a.alpha:g}.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, t.astype(np.float32))
    gold_in_c = q[np.arange(len(y)), y] > 0
    meta = {"tag": a.tag, "alpha": a.alpha, "n_rows": int(len(y)),
            "m3_argmax_equals_gold": float((q.argmax(1) == y).mean()),
            "gold_in_candidates": float(gold_in_c.mean()),
            "mean_target_on_gold": float(t[np.arange(len(y)), y].mean()),
            "mean_target_entropy": float(-(t * np.log(t + 1e-12)).sum(1).mean())}
    (out.with_suffix(".json")).write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}: {json.dumps(meta)}")


if __name__ == "__main__":
    main()
