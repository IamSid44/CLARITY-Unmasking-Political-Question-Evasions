#!/usr/bin/env python3
"""E13 family: the pre-registered analyses (reports/02_experiment_log.md, E13 and E13x/b/c).

Same machinery as e11_replication.py: a "system" is the mean of the seeds' probabilities
followed by logit adjustment, with tau fitted by 5-fold nested cross-validation on dev
(decide.py's R1); bootstrap over dev items with the seeds fixed. Every section prints
what exists so far and says what is missing, so it can be run while lanes are training.

  A  E13 + E13x: Qwen3-8B LoRA (Q8_fullq_lora) vs DeBERTa full-question 16 epochs
     (E10_fullq_16ep seeds 0-4 + E11_fullq_16ep seeds 5-9), paired by seed, and as systems
  B  E13b: 12 epochs (Q8_fullq_lora_12ep) -- decided on the TRAIN SLICE, dev only reported
  C  E13c: all of train, last epoch (Q8_alldata) vs E13, paired by seed
  D  E13 hypothesis 3: where the gain falls (per class; unanimous vs contested items)
  E  Qwen + DeBERTa probability mix, weight and tau both fitted inside the nested CV

    python clarity/e13_analysis.py        # CPU, about a minute; needs clarity/runs/
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from decide import fit_tau, logit_adjust, make_rules, nested_cv, reference_masks
from qevasion.labels import (EVASION_LABELS, N_EVASION, encode_clarity, encode_evasion,
                             leaf_to_official_clarity)
from qevasion.loader import consensus_level, dev_reference_mask, load_qevasion
from qevasion.scoring import score_subtask1, score_subtask2

RUNS = Path(__file__).resolve().parent / "runs"
sp = load_qevasion()
GOLD = dev_reference_mask(sp.dev)
CLAR = encode_clarity(sp.dev["clarity_label"].tolist())
N = len(GOLD)
y = encode_evasion(sp.train["evasion_label"].tolist())
PRIOR = np.bincount(y, minlength=N_EVASION) / len(y)
R1 = make_rules(PRIOR)["R1_logit_adjusted"]
BOOT = [np.random.default_rng(i).integers(0, N, N) for i in range(2000)]
E13_SLICE = {0: 0.4595, 1: 0.4440, 2: 0.4315}   # E13 slice F1 at the selected epoch (metrics.json)


def seed_dir(cfg: str, s: int) -> Path:
    return RUNS / cfg / f"seed{s}"


def deberta_cfg(s: int) -> str:
    return "E10_fullq_16ep" if s < 5 else "E11_fullq_16ep"


def finished(cfg: str) -> list[int]:
    return sorted(int(p.parent.name[4:]) for p in (RUNS / cfg).glob("seed*/metrics.json"))


def probs(cfg_of_seed, seeds) -> np.ndarray:
    return np.stack([np.load(seed_dir(cfg_of_seed(s), s) / "dev_probs.npy") for s in seeds])


def s2(pred) -> float:
    return score_subtask2(pred, GOLD).macro_f1


def s1(pred) -> float:
    return score_subtask1(leaf_to_official_clarity(pred), CLAR).macro_f1


def system(p: np.ndarray, splits: int = 1, gold: np.ndarray = GOLD) -> tuple[float, np.ndarray]:
    runs = [nested_cv(R1, p, gold, 5, cs) for cs in range(splits)]
    return float(np.mean([r["out_of_fold_macro_f1"] for r in runs])), runs[0]["pooled_predictions"]


def boot(a: np.ndarray, b: np.ndarray) -> str:
    d = np.array([score_subtask2(a[i], GOLD[i]).macro_f1 - score_subtask2(b[i], GOLD[i]).macro_f1
                  for i in BOOT])
    return f"{d.mean():+.3f}  95% [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]  P(<=0) = {np.mean(d <= 0):.3f}"


def paired(a: np.ndarray, b: np.ndarray) -> str:
    d = a - b
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    return f"{d.mean():+.3f} ± {d.std(ddof=1) if len(d) > 1 else 0:.3f} (t = {t:.2f}, {int((d > 0).sum())}/{len(d)} seeds up)"


def section_a() -> None:
    print("A. Qwen3-8B LoRA (Q8_fullq_lora) vs DeBERTa full question, 16 epochs -- dev\n")
    seeds = finished("Q8_fullq_lora")
    print(f"   finished Qwen seeds: {seeds}")
    pq, pd = probs(lambda s: "Q8_fullq_lora", seeds), probs(deberta_cfg, seeds)
    q2 = np.array([s2(p.argmax(1)) for p in pq]); d2 = np.array([s2(p.argmax(1)) for p in pd])
    q1 = np.array([s1(p.argmax(1)) for p in pq]); d1 = np.array([s1(p.argmax(1)) for p in pd])
    print(f"   {'seed':>4} {'Qwen S2':>8} {'DeBERTa S2':>10} {'Δ':>7}   {'Qwen S1':>8} {'DeBERTa S1':>10} {'Δ':>7}")
    for i, s in enumerate(seeds):
        print(f"   {s:>4} {q2[i]:>8.3f} {d2[i]:>10.3f} {q2[i] - d2[i]:>+7.3f}   {q1[i]:>8.3f} {d1[i]:>10.3f} {q1[i] - d1[i]:>+7.3f}")
    print(f"   mean  Qwen S2 {q2.mean():.3f} ± {q2.std(ddof=1):.3f}, S1 {q1.mean():.3f} ± {q1.std(ddof=1):.3f};"
          f"  DeBERTa S2 {d2.mean():.3f} ± {d2.std(ddof=1):.3f}, S1 {d1.mean():.3f} ± {d1.std(ddof=1):.3f}")
    print(f"   paired S2: {paired(q2, d2)}")
    print(f"   paired S1: {paired(q1, d1)}")
    ql = np.array([system(p)[0] for p in pq]); dl = np.array([system(p)[0] for p in pd])
    print(f"   single model + logit adjustment, S2: Qwen {ql.mean():.3f} ± {ql.std(ddof=1):.3f}, "
          f"DeBERTa {dl.mean():.3f} ± {dl.std(ddof=1):.3f}; paired {paired(ql, dl)}")
    k = len(seeds)
    sq1, prq = system(pq.mean(0)); sq10, _ = system(pq.mean(0), 10)
    sd1, prd = system(pd.mean(0)); sd10, _ = system(pd.mean(0), 10)
    print(f"\n   {k}-seed systems (ensemble + logit adjustment, nested CV), S2: Qwen {sq1:.3f} (10 CV splits {sq10:.3f}), "
          f"DeBERTa same seeds {sd1:.3f} ({sd10:.3f})")
    print(f"   ensemble argmax, S2: Qwen {s2(pq.mean(0).argmax(1)):.3f}, DeBERTa {s2(pd.mean(0).argmax(1)):.3f}")
    print(f"   system S1 (final leaf): Qwen {s1(prq):.3f}, DeBERTa {s1(prd):.3f}")
    print(f"   Qwen - DeBERTa system, bootstrap over items: {boot(prq, prd)}")
    masks = reference_masks(sp.dev, True)
    two_q = [nested_cv(R1, pq.mean(0), g, 5, 0)["out_of_fold_macro_f1"] for n, g in masks.items() if n != "3ann"]
    two_d = [nested_cv(R1, pd.mean(0), g, 5, 0)["out_of_fold_macro_f1"] for n, g in masks.items() if n != "3ann"]
    print(f"   2-annotator reference sets (test regime), systems: Qwen {', '.join(f'{v:.3f}' for v in two_q)}"
          f" (mean {np.mean(two_q):.3f}); DeBERTa {', '.join(f'{v:.3f}' for v in two_d)} (mean {np.mean(two_d):.3f})")
    if k < 10:
        print(f"   NOTE: {k} seeds -- the project's rule is 10 seeds for a system claim")


def section_b() -> None:
    print("\nB. E13b: 12 epochs (Q8_fullq_lora_12ep) -- decision on the TRAIN SLICE\n")
    seeds = finished("Q8_fullq_lora_12ep")
    if not seeds:
        print("   not finished yet"); return
    sel, ups, d = [], 0, []
    for s in seeds:
        m = json.loads((seed_dir("Q8_fullq_lora_12ep", s) / "metrics.json").read_text())
        e13 = json.loads((seed_dir("Q8_fullq_lora", s) / "metrics.json").read_text())
        h = m["history"]
        print(f"   seed {s}: selected epoch {m['selected_epoch']}, slice F1 {m['val_macro_f1']:.3f} (E13 {E13_SLICE[s]:.3f}); "
              f"dev S2 {m['dev_subtask2_macro_f1']:.3f} (E13 {e13['dev_subtask2_macro_f1']:.3f}), "
              f"S1 {m['dev_subtask1_macro_f1']:.3f} (E13 {e13['dev_subtask1_macro_f1']:.3f})")
        print("      slice F1 by epoch: " + " ".join(f"{x['val_macro_f1']:.3f}" for x in h))
        print("      train loss by epoch: " + " ".join(f"{x['train_loss']:.3f}" for x in h))
        print("      dev S2 by epoch (reported only): " + " ".join(f"{x['dev_subtask2_macro_f1']:.3f}" for x in h))
        sel.append(m["val_macro_f1"] - E13_SLICE[s]); ups += m["val_macro_f1"] > E13_SLICE[s]
        d.append(m["dev_subtask2_macro_f1"] - e13["dev_subtask2_macro_f1"])
    print(f"   slice F1 change vs E13: {np.mean(sel):+.3f} ({ups}/{len(seeds)} seeds up); "
          f"registered rule: adopt 12 epochs if >= +0.015 with >= 2/3 up -> "
          f"{'ADOPT' if np.mean(sel) >= 0.015 and ups >= 2 else 'KEEP 3 EPOCHS'}"
          f"{'' if len(seeds) == 3 else '  (incomplete: ' + str(len(seeds)) + '/3 seeds)'}")
    print(f"   dev S2 change vs E13 (reported, decides nothing): {np.mean(d):+.3f}")


def section_c() -> None:
    print("\nC. E13c: all of train, last of 3 epochs (Q8_alldata) vs E13, paired by seed\n")
    seeds = [s for s in finished("Q8_alldata") if s in finished("Q8_fullq_lora")]
    if not seeds:
        print("   not finished yet"); return
    a = probs(lambda s: "Q8_alldata", seeds); b = probs(lambda s: "Q8_fullq_lora", seeds)
    a2 = np.array([s2(p.argmax(1)) for p in a]); b2 = np.array([s2(p.argmax(1)) for p in b])
    a1 = np.array([s1(p.argmax(1)) for p in a]); b1 = np.array([s1(p.argmax(1)) for p in b])
    for i, s in enumerate(seeds):
        print(f"   seed {s}: S2 {a2[i]:.3f} vs {b2[i]:.3f} ({a2[i] - b2[i]:+.3f}), S1 {a1[i]:.3f} vs {b1[i]:.3f} ({a1[i] - b1[i]:+.3f})")
    print(f"   paired S2: {paired(a2, b2)};  S1: {paired(a1, b1)}")
    print(f"   screening bar (+0.015, >= 2/3 up): {'PASSED' if (a2 - b2).mean() >= 0.015 and (a2 > b2).sum() >= 2 else 'not passed'}"
          f"{'' if len(seeds) == 3 else '  (incomplete)'}")


def section_d() -> None:
    print("\nD. Where the gain falls (E13 hypothesis 3): Qwen vs DeBERTa, same seeds, mean over seeds\n")
    seeds = finished("Q8_fullq_lora")
    pq, pd = probs(lambda s: "Q8_fullq_lora", seeds), probs(deberta_cfg, seeds)
    fq = np.mean([score_subtask2(p.argmax(1), GOLD).f1_vector(EVASION_LABELS) for p in pq], 0)
    fd = np.mean([score_subtask2(p.argmax(1), GOLD).f1_vector(EVASION_LABELS) for p in pd], 0)
    print(f"   {'class':<22}{'Qwen F1':>9}{'DeBERTa':>9}{'Δ':>8}")
    for c, a, b in zip(EVASION_LABELS, fq, fd):
        print(f"   {c:<22}{a:>9.3f}{b:>9.3f}{a - b:>+8.3f}")
    lvl = consensus_level(sp.dev)
    print("\n   in-set rate by annotator agreement (distinct labels per item):")
    for v, name in ((1, "unanimous"), (2, "two labels"), (3, "three labels")):
        m = lvl == v
        hq = np.mean([GOLD[np.arange(N), p.argmax(1)][m].mean() for p in pq])
        hd = np.mean([GOLD[np.arange(N), p.argmax(1)][m].mean() for p in pd])
        print(f"   {name:<13} n={m.sum():>3}: Qwen {hq:.3f}, DeBERTa {hd:.3f}, Δ {hq - hd:+.3f}")


def section_e() -> None:
    print("\nE. Qwen + DeBERTa probability mix (weight w on Qwen and tau fitted inside each CV fold)\n")
    seeds = finished("Q8_fullq_lora")
    pq = probs(lambda s: "Q8_fullq_lora", seeds).mean(0)
    pd = probs(deberta_cfg, seeds).mean(0)
    grid = np.linspace(0, 1, 11)

    def rule(stack, gold, tr, te):
        best = (-1.0, 1.0, 0.0)
        for w in grid:
            p = w * stack[:, 0] + (1 - w) * stack[:, 1]
            tau, fit = fit_tau(p[tr], gold[tr], PRIOR, np.linspace(-1.0, 2.0, 61))
            if fit > best[0]:
                best = (fit, w, tau)
        fit, w, tau = best
        p = w * stack[:, 0] + (1 - w) * stack[:, 1]
        rule.ws.append(w)
        return logit_adjust(p[te], PRIOR, tau).argmax(1), fit

    rule.ws = []
    stack = np.stack([pq, pd], 1)
    mix = [nested_cv(rule, stack, GOLD, 5, cs) for cs in range(10)]
    qwen = [nested_cv(R1, pq, GOLD, 5, cs)["out_of_fold_macro_f1"] for cs in range(10)]
    print(f"   Qwen alone (+LA): {np.mean(qwen):.3f};  mix (+LA): {np.mean([m['out_of_fold_macro_f1'] for m in mix]):.3f}"
          f"  (mean over 10 CV splits, {len(seeds)} seeds each)")
    print(f"   weights chosen on the training folds: mean w = {np.mean(rule.ws):.2f} (min {min(rule.ws):.1f}, max {max(rule.ws):.1f})")
    print(f"   mix - Qwen, split 0, bootstrap: {boot(mix[0]['pooled_predictions'], nested_cv(R1, pq, GOLD, 5, 0)['pooled_predictions'])}")


if __name__ == "__main__":
    section_a()
    section_b()
    section_c()
    section_d()
    section_e()
