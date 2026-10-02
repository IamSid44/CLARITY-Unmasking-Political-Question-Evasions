#!/usr/bin/env python3
"""Aggregate the ablation ladder into one table. Reads only metrics.json files.

Prints mean +/- std over seeds (no single-seed number is a result; README §7),
and a paired per-seed delta against the reference configuration. The delta is
paired because every configuration shares seeds, the same internal
validation split and the same data order, so seed-to-seed variation is common to
both arms and paired differences are far tighter than the marginal spreads
suggest.

Per-class F1 travels with support (invariant 3). Classes below 10 support are
flagged inline as below measurement resolution rather than quietly reported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qevasion.labels import EVASION_LABELS

RUNS = Path(__file__).resolve().parent / "runs"
LOW_SUPPORT = 10


def load(cfg_dir: Path) -> dict[int, dict]:
    out = {}
    for seed_dir in sorted(cfg_dir.glob("seed*")):
        f = seed_dir / "metrics.json"
        if f.exists():
            out[int(seed_dir.name[4:])] = json.loads(f.read_text())
    return out


def fmt(x: np.ndarray) -> str:
    if len(x) == 0:
        return "    --     "
    if len(x) == 1:
        return f"{x[0]:.4f} (n=1)"
    return f"{x.mean():.4f}+/-{x.std(ddof=1):.4f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", default=str(RUNS))
    p.add_argument("--reference", default="A0_base")
    p.add_argument("--per-class", action="store_true")
    args = p.parse_args()

    root = Path(args.runs)
    configs = {d.name: load(d) for d in sorted(root.iterdir()) if d.is_dir()}
    configs = {k: v for k, v in configs.items() if v}
    if not configs:
        raise SystemExit(f"no completed runs under {root}")

    ref = configs.get(args.reference, {})

    print(f"{'config':<18} {'n':>2}  {'dev Subtask2':<18} {'dev Subtask1':<18} "
          f"{'in-set rate':<18} {'named':>6}  {'paired d(S2) vs ' + args.reference}")
    print("-" * 122)
    for name, seeds in configs.items():
        s2 = np.array([m["dev_subtask2_macro_f1"] for m in seeds.values()])
        s1 = np.array([m["dev_subtask1_macro_f1"] for m in seeds.values()])
        va = np.array([m["val_macro_f1"] for m in seeds.values()])
        # in-set rate and coverage: the two terms macro-F1 decomposes into when
        # every prediction lands in its reference set (reports/01_scorer_geometry.md).
        ins = np.array([m.get("dev_inset", {}).get("inset_rate", np.nan) for m in seeds.values()])
        nmd = np.array([m.get("dev_inset", {}).get("classes_named", np.nan) for m in seeds.values()])
        delta = "  --"
        if ref and name != args.reference:
            shared = sorted(set(seeds) & set(ref))
            if shared:
                d = np.array([seeds[k]["dev_subtask2_macro_f1"] - ref[k]["dev_subtask2_macro_f1"]
                              for k in shared])
                sd = f"+/-{d.std(ddof=1):.4f}" if len(d) > 1 else " (n=1)"
                delta = f"{d.mean():+.4f}{sd}  [{len(shared)} paired]"
        named = f"{np.nanmean(nmd):.1f}/9" if not np.all(np.isnan(nmd)) else "  -- "
        print(f"{name:<18} {len(s2):>2}  {fmt(s2):<18} {fmt(s1):<18} {fmt(ins):<18} "
              f"{named:>6}  {delta}")

    if args.per_class:
        print("\nPer-class Subtask-2 F1 (mean over seeds), support from the scorer.")
        print("Support is ENDOGENOUS -- an item whose prediction names a different")
        print("member of its reference set leaves this class's support entirely")
        print("so support moves between configurations.\n")
        header = f"{'config':<20}" + "".join(f"{c[:11]:>13}" for c in EVASION_LABELS)
        print(header)
        print("-" * len(header))
        for name, seeds in configs.items():
            f1 = {c: [] for c in EVASION_LABELS}
            sup = {c: [] for c in EVASION_LABELS}
            for m in seeds.values():
                for c, v in m["dev_subtask2_per_class"].items():
                    f1[c].append(v["f1"])
                    sup[c].append(v["support"])
            cells = []
            for c in EVASION_LABELS:
                s = np.mean(sup[c]) if sup[c] else 0
                flag = "*" if s < LOW_SUPPORT else " "
                cells.append(f"{np.mean(f1[c]):>9.3f}{flag}{int(round(s)):>3}")
            print(f"{name:<20}" + "".join(cells))
        print(f"\n  * support < {LOW_SUPPORT}: below measurement resolution "
              f"(README §7). Trailing integer is mean support.")


if __name__ == "__main__":
    main()
