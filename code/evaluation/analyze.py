#!/usr/bin/env python3
"""Full metric breakdown for a finished run, computed from saved probabilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qevasion.labels import (
    CLARITY_LABELS,
    EVASION_LABELS,
    N_EVASION,
    encode_clarity,
    leaf_to_official_clarity,
    OFFICIAL_PARTITION_MAP,
)
from qevasion.loader import dev_reference_mask, load_qevasion
from qevasion.scoring import score_subtask1, score_subtask2

from qevasion.loader import DATA_CACHE
LOW_SUPPORT = 10


def consensus(p: np.ndarray) -> np.ndarray:
    return p.mean(axis=1) if p.ndim == 3 else p


def inset(pred: np.ndarray, mask: np.ndarray) -> tuple[float, list[int]]:
    known = pred >= 0
    hit = np.zeros(len(pred), dtype=bool)
    hit[known] = mask[np.arange(len(pred))[known], pred[known]]
    return float(hit.mean()), sorted(set(pred[known].tolist()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run = Path(args.run_dir)

    dev = load_qevasion(DATA_CACHE).dev
    mask = dev_reference_mask(dev)
    clarity_true = encode_clarity(dev["clarity_label"].tolist())

    seeds = sorted(run.glob("seed*/dev_probs.npy"))
    if not seeds:
        raise SystemExit(f"no dev_probs.npy under {run}")
    probs = np.stack([np.load(f) for f in seeds])
    print(f"{run.name}: {len(seeds)} seed(s)\n")

    s2, s1, s1_level, ins, cov = [], [], [], [], []
    per_class_2 = {c: [] for c in EVASION_LABELS}
    per_class_1 = {c: [] for c in CLARITY_LABELS}
    sup_2 = {c: [] for c in EVASION_LABELS}
    sup_1 = {c: [] for c in CLARITY_LABELS}

    for p in probs:
        pred = consensus(p).argmax(1)
        r2 = score_subtask2(pred, mask)
        r1 = score_subtask1(leaf_to_official_clarity(pred), clarity_true)
        pc = consensus(p)
        level = np.stack([pc[:, OFFICIAL_PARTITION_MAP == b].sum(1) for b in range(len(CLARITY_LABELS))], 1)
        s1_level.append(score_subtask1(level.argmax(1), clarity_true).macro_f1)
        rate, named = inset(pred, mask)
        s2.append(r2.macro_f1)
        s1.append(r1.macro_f1)
        ins.append(rate)
        cov.append(len(named) / N_EVASION)
        for c, m in r2.per_class.items():
            per_class_2[c].append(m.f1)
            sup_2[c].append(m.support)
        for c, m in r1.per_class.items():
            per_class_1[c].append(m.f1)
            sup_1[c].append(m.support)

    def pm(v):
        v = np.asarray(v, dtype=float)
        return f"{v.mean():.4f}" + (f" +/- {v.std(ddof=1):.4f}" if len(v) > 1 else " (n=1)")

    print("HEADLINE")
    print(f"  Subtask 2 (9-way, multi-reference)  macro-F1 = {pm(s2)}")
    print(f"  Subtask 1 (3-way clarity, derived)  macro-F1 = {pm(s1)}")
    print(f"  Subtask 1, most probable level      macro-F1 = {pm(s1_level)}")
    print("\nDECOMPOSITION  (macro-F1 = classes-named/9 exactly, when every prediction is in-set)")
    print(f"  in-set rate                              = {pm(ins)}")
    print(f"  class coverage (classes named / 9)       = {pm(cov)}")
    print(f"  => coverage ceiling on Subtask 2         = {np.mean(cov):.4f}")

    print("\nPER-CLASS, Subtask 2      (support is from the scorer and is ENDOGENOUS)")
    print(f"  {'class':<22} {'F1':>16} {'support':>9}")
    for c in EVASION_LABELS:
        sup = np.mean(sup_2[c])
        flag = "  <- below measurement resolution" if sup < LOW_SUPPORT else ""
        print(f"  {c:<22} {pm(per_class_2[c]):>16} {sup:>9.1f}{flag}")

    print("\nPER-CLASS, Subtask 1")
    print(f"  {'class':<22} {'F1':>16} {'support':>9}")
    for c in CLARITY_LABELS:
        print(f"  {c:<22} {pm(per_class_1[c]):>16} {np.mean(sup_1[c]):>9.1f}")

    never = [c for c in EVASION_LABELS
             if np.mean(sup_2[c]) == 0 and np.mean(per_class_2[c]) == 0]
    if never:
        print(f"\n  Classes the model never predicts: {', '.join(never)}")
        print("  Each costs exactly 1/9 = 0.1111 of the Subtask-2 score.")

    out = run / "analysis.json"
    out.write_text(json.dumps({
        "n_seeds": len(seeds),
        "subtask2_macro_f1": {"mean": float(np.mean(s2)), "std": float(np.std(s2, ddof=1)) if len(s2) > 1 else 0.0},
        "subtask1_macro_f1": {"mean": float(np.mean(s1)), "std": float(np.std(s1, ddof=1)) if len(s1) > 1 else 0.0},
        "subtask1_level_macro_f1": {"mean": float(np.mean(s1_level)), "std": float(np.std(s1_level, ddof=1)) if len(s1_level) > 1 else 0.0},
        "inset_rate": float(np.mean(ins)),
        "class_coverage": float(np.mean(cov)),
        "per_class_subtask2": {c: float(np.mean(v)) for c, v in per_class_2.items()},
        "per_class_subtask1": {c: float(np.mean(v)) for c, v in per_class_1.items()},
    }, indent=2))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
