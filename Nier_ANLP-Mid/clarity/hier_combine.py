#!/usr/bin/env python3
"""E12b/E12c: combine the gate, the specialists and the boundary experts into 9-way
predictions, and compare every combination with the flat model, seed by seed.

Everything is a 9-way probability distribution, so the usual machinery (seed
ensemble, logit adjustment by nested CV) applies unchanged and the comparison with
the flat model is like for like. With p9 the flat model (E10_fullq_16ep, same seed):

  gate        g(x) = p(Non-Reply)            flat: the sum of p9 over the 3 NR leaves
                                             dedicated: E12b_gate
  within      p(leaf | branch)               flat: p9 renormalised inside the branch
                                             specialists: E12b_nr3 / E12b_other6
  combined    p(leaf) = g * p(leaf | NR) for NR leaves, (1 - g) * p(leaf | other) else

  hard        route by g > 0.5 instead of weighting (the branch not taken gets 0)
  experts     E12c: where a pair (A, B) are an item's top two, the pair's joint mass
              is re-split by that pair's boundary expert

    python clarity/hier_combine.py            # CPU; uses whichever seeds are complete
    python clarity/hier_combine.py --focus "flat gate + specialists"   # one variant, all its seeds
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from decide import logit_adjust, make_rules, nested_cv
from qevasion.labels import (EVASION_LABELS, N_EVASION, encode_clarity, encode_evasion,
                             leaf_to_official_clarity)
from qevasion.loader import dev_reference_mask, load_qevasion
from qevasion.scoring import score_subtask1, score_subtask2

RUNS = Path(os.environ.get("CLARITY_RUNS", Path(__file__).resolve().parent / "runs"))
FLAT = "E10_fullq_16ep"
NR = [EVASION_LABELS.index(c) for c in ("Declining to answer", "Claims ignorance", "Clarification")]
OT = [i for i in range(N_EVASION) if i not in NR]
PAIRS = {"E12c_pair_impl_dodg": ("Implicit", "Dodging"),
         "E12c_pair_expl_impl": ("Explicit", "Implicit"),
         "E12c_pair_gen_defl": ("General", "Deflection")}

sp = load_qevasion()
GOLD = dev_reference_mask(sp.dev)
CLAR = encode_clarity(sp.dev["clarity_label"].tolist())
y = encode_evasion(sp.train["evasion_label"].tolist())
PRIOR = np.bincount(y, minlength=N_EVASION) / len(y)
R1 = make_rules(PRIOR)["R1_logit_adjusted"]


def load(cfg: str, seed: int, split: str = "dev") -> np.ndarray | None:
    f = RUNS / cfg / f"seed{seed}" / f"{split}_probs.npy"
    return np.load(f) if f.exists() and (f.parent / "metrics.json").exists() else None


def compose(gate: np.ndarray, nr: np.ndarray, ot: np.ndarray, hard: bool = False) -> np.ndarray:
    g = (gate > 0.5).astype(float) if hard else gate
    p = np.zeros((len(gate), N_EVASION))
    p[:, NR] = g[:, None] * nr
    p[:, OT] = (1 - g)[:, None] * ot
    return p


def within(p9: np.ndarray, idx: list[int]) -> np.ndarray:
    q = p9[:, idx]
    return q / q.sum(1, keepdims=True)


def apply_experts(p: np.ndarray, experts: dict[str, np.ndarray]) -> np.ndarray:
    out = p.copy()
    top2 = np.argsort(-p, 1)[:, :2]
    for cfg, e in experts.items():
        a, b = (EVASION_LABELS.index(next(c for c in EVASION_LABELS if c.startswith(n))) for n in PAIRS[cfg])
        hit = ((top2[:, 0] == a) & (top2[:, 1] == b)) | ((top2[:, 0] == b) & (top2[:, 1] == a))
        mass = p[hit, a] + p[hit, b]
        out[hit, a], out[hit, b] = mass * e[hit, 0], mass * e[hit, 1]
    return out


def variants(seed: int, split: str = "dev") -> dict[str, np.ndarray] | None:
    p9 = load(FLAT, seed, split)
    gate, other6, nr3 = (load(c, seed, split) for c in ("E12b_gate", "E12b_other6", "E12b_nr3"))
    if p9 is None:
        return None
    v = {"flat": p9}
    g_flat = p9[:, NR].sum(1)
    if gate is not None:
        v["dedicated gate + flat within"] = compose(gate[:, 1], within(p9, NR), within(p9, OT))
    if other6 is not None and nr3 is not None:
        v["flat gate + specialists"] = compose(g_flat, nr3, other6)
        if gate is not None:
            v["hierarchy (gate + specialists)"] = compose(gate[:, 1], nr3, other6)
            v["hierarchy, hard routing"] = compose(gate[:, 1], nr3, other6, hard=True)
    experts = {c: e for c in PAIRS if (e := load(c, seed, split)) is not None}
    if len(experts) == len(PAIRS):
        v["flat + boundary experts"] = apply_experts(p9, experts)
        if "hierarchy (gate + specialists)" in v:
            v["hierarchy + boundary experts"] = apply_experts(v["hierarchy (gate + specialists)"], experts)
    return v


def scores(p: np.ndarray) -> dict[str, float]:
    pred = p.argmax(1)
    la = np.mean([nested_cv(R1, p, GOLD, 5, cs)["out_of_fold_macro_f1"] for cs in range(5)])
    s1 = score_subtask1(leaf_to_official_clarity(pred), CLAR)
    nr_gold = GOLD[:, NR].any(1)
    nr_pred = np.isin(pred, NR)
    return {"S2": score_subtask2(pred, GOLD).macro_f1, "S2+rule": float(la), "S1": s1.macro_f1,
            "CNR F1": s1.per_class["Clear Non-Reply"].f1,
            "NR recall": float((nr_pred & nr_gold).sum() / nr_gold.sum())}


def focus(name: str) -> None:
    """One variant against the flat model on every seed that has it (e.g. after an extension)."""
    per = {s: v for s in range(10) if (v := variants(s)) is not None and name in v}
    seeds = sorted(per)
    cols = ("S2", "S2+rule", "S1", "CNR F1", "NR recall")
    tab = {s: (scores(per[s]["flat"]), scores(per[s][name])) for s in seeds}
    print(f"'{name}' vs flat ({FLAT}), seeds {seeds}\n")
    print(f"{'':<24}" + "".join(f"{c:>11}" for c in cols))
    for lab, i in (("flat, per-seed mean", 0), (name[:23] + ", mean", 1)):
        print(f"{lab:<24}" + "".join(f"{np.mean([tab[s][i][c] for s in seeds]):>11.3f}" for c in cols))
    cells = []
    for c in cols:
        d = np.array([tab[s][1][c] - tab[s][0][c] for s in seeds])
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 and d.std() > 0 else float("nan")
        cells.append(f"{d.mean():+.3f} {int((d > 0).sum())}/{len(d)}")
    print(f"{'difference, paired':<24}" + "".join(f"{x:>11}" for x in cells))
    for lab, key in (("flat", "flat"), (name[:23], name)):
        e = scores(np.mean([per[s][key] for s in seeds], 0))
        print(f"{lab + f', {len(seeds)}-seed ensemble':<24}" + "".join(f"{e[c]:>11.3f}" for c in cols))


def main() -> None:
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "--focus":
        return focus(sys.argv[2])
    per = {s: v for s in range(10) if (v := variants(s)) is not None}
    most = max(len(v) for v in per.values())
    seeds = [s for s, v in per.items() if len(v) == most]  # seeds with every component finished
    if most == 1:
        raise SystemExit("no E12 component has finished for any seed yet")
    names = list(per[seeds[0]])
    cols = ("S2", "S2+rule", "S1", "CNR F1", "NR recall")
    print(f"E12 combinations on dev, seeds {seeds} (paired with {FLAT} of the same seed).")
    print("S2+rule: logit adjustment by nested CV, mean over 5 splits. CNR = Clear Non-Reply "
          "(Subtask 1). NR recall: items with any Non-Reply label that the model calls Non-Reply.\n")
    tab = {n: {s: scores(per[s][n]) for s in seeds} for n in names}
    print(f"{'per seed, mean':<34}" + "".join(f"{c:>11}" for c in cols))
    for n in names:
        print(f"{n:<34}" + "".join(f"{np.mean([tab[n][s][c] for s in seeds]):>11.3f}" for c in cols))
    print(f"\n{'minus flat, per seed':<34}" + "".join(f"{c:>11}" for c in cols))
    for n in names[1:]:
        cells = []
        for c in cols:
            d = np.array([tab[n][s][c] - tab["flat"][s][c] for s in seeds])
            cells.append(f"{d.mean():+.3f} {int((d > 0).sum())}/{len(d)}")
        print(f"{n:<34}" + "".join(f"{x:>11}" for x in cells))
    print(f"\n{'ensemble of the seeds':<34}" + "".join(f"{c:>11}" for c in cols))
    for n in names:
        e = np.mean([per[s][n] for s in seeds], 0)
        sc = scores(e)
        print(f"{n:<34}" + "".join(f"{sc[c]:>11.3f}" for c in cols))

    ad = [s for s in seeds if (RUNS / "E12a_alldata" / f"seed{s}" / "metrics.json").exists()]
    if ad:
        print("\nE12a, all of train, last epoch, vs the flat model (single models, dev):")
        for key, lab in (("dev_subtask2_macro_f1", "S2"), ("dev_subtask1_macro_f1", "S1")):
            d = np.array([json.loads((RUNS / "E12a_alldata" / f"seed{s}" / "metrics.json").read_text())[key]
                          - json.loads((RUNS / FLAT / f"seed{s}" / "metrics.json").read_text())[key] for s in ad])
            print(f"    {lab}: {d.mean():+.3f} ± {d.std(ddof=1) if len(d) > 1 else 0:.3f}  ({int((d > 0).sum())}/{len(d)} seeds up)")


if __name__ == "__main__":
    main()
