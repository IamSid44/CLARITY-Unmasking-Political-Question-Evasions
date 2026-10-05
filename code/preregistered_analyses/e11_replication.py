#!/usr/bin/env python3
"""E11: the pre-registered replication analysis (docs/02_experiment_log.md §E11)."""

from __future__ import annotations

import itertools

import numpy as np

from decision_rules.decide import load_probs, make_rules, nested_cv
from qevasion.labels import (N_EVASION, OFFICIAL_PARTITION_MAP, encode_clarity, encode_evasion,
                             leaf_to_official_clarity)
from decision_rules.decide import reference_masks
from qevasion.loader import dev_reference_mask, load_qevasion
from qevasion.scoring import score_subtask1, score_subtask2
from qevasion.paths import RUNS

SYSTEMS = {
    "base_8ep": ("L0_large_base", "E11_base_8ep"),
    "base_16ep": ("E10_base_16ep", "E11_base_16ep"),
    "fullq_16ep": ("E10_fullq_16ep", "E11_fullq_16ep"),
}

sp = load_qevasion()
GOLD = dev_reference_mask(sp.dev)
CLAR = encode_clarity(sp.dev["clarity_label"].tolist())
N = len(GOLD)
y = encode_evasion(sp.train["evasion_label"].tolist())
PRIOR = np.bincount(y, minlength=N_EVASION) / len(y)
R1 = make_rules(PRIOR)["R1_logit_adjusted"]
P = {name: {half: load_probs(RUNS / cfg)[0] for half, cfg in zip(("old", "new"), cfgs)}
     for name, cfgs in SYSTEMS.items()}
BOOT = [np.random.default_rng(i).integers(0, N, N) for i in range(2000)]


def system(p: np.ndarray, splits: int = 1) -> tuple[float, np.ndarray]:
    """Held-out macro-F1 of ensemble p + R1; mean over `splits` CV splits, and split-0 predictions."""
    runs = [nested_cv(R1, p, GOLD, 5, cs) for cs in range(splits)]
    return float(np.mean([r["out_of_fold_macro_f1"] for r in runs])), runs[0]["pooled_predictions"]


def boot(a: np.ndarray, b: np.ndarray) -> str:
    d = np.array([score_subtask2(a[i], GOLD[i]).macro_f1 - score_subtask2(b[i], GOLD[i]).macro_f1
                  for i in BOOT])
    return f"{d.mean():+.3f}  95% [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]  P(<=0) = {np.mean(d <= 0):.3f}"


def per_seed(p: np.ndarray) -> str:
    v = [system(p[i : i + 1].mean(0))[0] for i in range(len(p))]
    return f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}"


def main() -> None:
    print("E11 replication analysis — dev Subtask 2 unless marked; system = ensemble + logit adjustment\n")

    print("Each half on its own (5 seeds each):")
    print(f"  {'system':<12}{'seeds 0-4':>12}{'seeds 5-9':>12}{'  (avg over 10 CV splits: 0-4 / 5-9)'}")
    sc = {}
    for name in SYSTEMS:
        o = system(P[name]["old"].mean(0), 10)[0]
        n10, _ = system(P[name]["new"].mean(0), 10)
        o1, _ = system(P[name]["old"].mean(0))
        n1, pred = system(P[name]["new"].mean(0))
        sc[name] = pred
        print(f"  {name:<12}{o1:>12.3f}{n1:>12.3f}      {o:.3f} / {n10:.3f}")

    print("\nH1  replication: fullq_16ep above base_8ep on seeds 5-9")
    print(f"    fullq_16ep - base_8ep (seeds 5-9): {boot(sc['fullq_16ep'], sc['base_8ep'])}")
    print(f"    fullq_16ep - base_16ep (seeds 5-9): {boot(sc['fullq_16ep'], sc['base_16ep'])}")

    print("\nH2  single models + logit adjustment, seeds 5-9 (mean ± std over seeds):")
    for name in SYSTEMS:
        print(f"    {name:<12} {per_seed(P[name]['new'])}   (seeds 0-4: {per_seed(P[name]['old'])})")

    print("\nH3  mixed inputs, 10 models each:")
    mixed = np.concatenate([P["fullq_16ep"]["new"], P["base_16ep"]["new"]])
    fq10 = np.concatenate([P["fullq_16ep"]["old"], P["fullq_16ep"]["new"]])
    m1, mp = system(mixed.mean(0)); m10, _ = system(mixed.mean(0), 10)
    f1, fp = system(fq10.mean(0)); f10, _ = system(fq10.mean(0), 10)
    print(f"    fullq_16ep 5-9 + base_16ep 5-9: {m1:.3f}  (10 splits: {m10:.3f})")
    print(f"    fullq_16ep 0-9:                 {f1:.3f}  (10 splits: {f10:.3f})")
    print(f"    mixed - fullq: {boot(mp, fp)}")

    print("\nH4  ensemble size, all 10 seeds (mean over up to 40 random subsets of each size; CV split 0):")
    rng = np.random.default_rng(0)
    for name in SYSTEMS:
        allp = np.concatenate([P[name]["old"], P[name]["new"]])
        row = []
        for k in (1, 2, 3, 5, 7, 10):
            combos = list(itertools.combinations(range(10), k))
            pick = [combos[i] for i in rng.choice(len(combos), min(40, len(combos)), replace=False)]
            row.append(np.mean([system(allp[list(c)].mean(0))[0] for c in pick]))
        print(f"    {name:<12}" + "".join(f"  k={k}: {v:.3f}" for k, v in zip((1, 2, 3, 5, 7, 10), row)))

    print("\n10-seed systems (seeds 0-9), with 2-annotator checks and Subtask 1:")
    pred10 = {}
    for name in SYSTEMS:
        allp = np.concatenate([P[name]["old"], P[name]["new"]]).mean(0)
        s1, pr = system(allp); s10, _ = system(allp, 10)
        pred10[name] = pr
        lvl = np.stack([allp[:, OFFICIAL_PARTITION_MAP == b].sum(1) for b in range(3)], 1).argmax(1)
        s1_leaf = score_subtask1(leaf_to_official_clarity(pr), CLAR).macro_f1
        print(f"    {name:<12} S2 {s1:.3f} (10 splits {s10:.3f})   S1 from final leaf {s1_leaf:.3f}, most probable level {score_subtask1(lvl, CLAR).macro_f1:.3f}")
    print(f"    fullq_16ep - base_8ep (10 seeds): {boot(pred10['fullq_16ep'], pred10['base_8ep'])}")
    masks = reference_masks(sp.dev, True)
    for name in SYSTEMS:
        allp = np.concatenate([P[name]["old"], P[name]["new"]]).mean(0)
        two = [nested_cv(R1, allp, g, 5, 0)["out_of_fold_macro_f1"] for k, g in masks.items() if k != "3ann"]
        print(f"    {name:<12} on the three 2-annotator reference sets: " + ", ".join(f"{v:.3f}" for v in two))
    print(f"    fullq_16ep - base_16ep (10 seeds): {boot(pred10['fullq_16ep'], pred10['base_16ep'])}")
    print("\nPaired by seed (same seed = same init and data order), all 10 seeds, single models:")
    import json
    def m(cfg, s, key): return json.loads((RUNS / cfg / f"seed{s}" / "metrics.json").read_text())[key]
    for a, b in (("fullq_16ep", "base_8ep"), ("fullq_16ep", "base_16ep"), ("base_16ep", "base_8ep")):
        for key, lab in (("dev_subtask2_macro_f1", "S2"), ("dev_subtask1_macro_f1", "S1")):
            d = np.array([m(SYSTEMS[a][s >= 5], s, key) - m(SYSTEMS[b][s >= 5], s, key) for s in range(10)])
            t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
            print(f"    {a} - {b} {lab}: {d.mean():+.3f} ± {d.std(ddof=1):.3f}  (t = {t:.2f}, {int((d > 0).sum())}/10 seeds positive)")
    print("\n    (bootstrap resamples dev items with seeds fixed; it understates training-seed noise)")


if __name__ == "__main__":
    main()
