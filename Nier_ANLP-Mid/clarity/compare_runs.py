#!/usr/bin/env python3
"""Compare finished configurations against the baseline, seed by seed and as ensembles.

Written for E8b, whose question -- does the full journalist question help? -- needs
more than the headline table: paired seeds, a paired bootstrap for the ensemble,
the per-length split first used for E8, and an undertraining check (did the run
still improve when the epoch budget ran out?).

    python clarity/compare_runs.py L0_large_base E8b_fullq_ce E8_fullq_bal_focal

The first name is the reference; every other configuration is compared with it.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

from encoder import build_inputs
from qevasion.labels import (EVASION_LABELS, OFFICIAL_PARTITION_MAP, encode_clarity,
                             leaf_to_official_clarity)
from qevasion.loader import dev_reference_mask, load_qevasion
from qevasion.scoring import score_subtask1, score_subtask2

HERE = Path(__file__).resolve().parent
RUNS, LOGS = HERE / "runs", HERE / "logs"
EPOCH_LINE = re.compile(r"ep(\d+) loss=([0-9.]+) val=([0-9.]+) devS2=([0-9.]+)")


def seed_dirs(name: str) -> list[Path]:
    return sorted(p for p in (RUNS / name).glob("seed*") if (p / "metrics.json").exists())


def epoch_curve(name: str, seed: int) -> list[tuple[int, float, float, float]]:
    """(epoch, train loss, internal val, dev S2) per epoch; last line wins on resume."""
    rows = {}
    log = LOGS / f"{name}_seed{seed}.log"
    for line in log.read_text().splitlines() if log.exists() else []:
        m = EPOCH_LINE.search(line)
        if m:
            rows[int(m[1])] = (int(m[1]), float(m[2]), float(m[3]), float(m[4]))
    return [rows[k] for k in sorted(rows)]


def level_pred(p: np.ndarray) -> np.ndarray:
    return np.stack([p[:, OFFICIAL_PARTITION_MAP == b].sum(1) for b in range(3)], 1).argmax(1)


def main() -> None:
    names = sys.argv[1:]
    if len(names) < 2:
        raise SystemExit(__doc__)
    sp = load_qevasion()
    mask = dev_reference_mask(sp.dev)
    clar = encode_clarity(sp.dev["clarity_label"].tolist())
    n = len(mask)
    idx = np.arange(n)
    ref = names[0]

    # ---- per seed -------------------------------------------------------------
    print("PER SEED  (dev S2 / S1 at the epoch chosen on the train slice)")
    metrics = {nm: {int(d.name[4:]): json.loads((d / "metrics.json").read_text()) for d in seed_dirs(nm)}
               for nm in names}
    seeds = sorted(set.intersection(*(set(m) for m in metrics.values())))
    head = "".join(f"{nm[:22]:>26}" for nm in names)
    print(f"  seed{head}")
    for s in seeds:
        cells = "".join(f"{metrics[nm][s]['dev_subtask2_macro_f1']:>11.3f} / {metrics[nm][s]['dev_subtask1_macro_f1']:.3f} "
                        f"e{metrics[nm][s]['selected_epoch']}" for nm in names)
        print(f"  {s:>4}{cells}")
    for nm in names[1:]:
        d2 = [metrics[nm][s]["dev_subtask2_macro_f1"] - metrics[ref][s]["dev_subtask2_macro_f1"] for s in seeds]
        d1 = [metrics[nm][s]["dev_subtask1_macro_f1"] - metrics[ref][s]["dev_subtask1_macro_f1"] for s in seeds]
        print(f"  {nm} - {ref}: S2 {np.mean(d2):+.3f} +/- {np.std(d2, ddof=1):.3f}   "
              f"S1 {np.mean(d1):+.3f} +/- {np.std(d1, ddof=1):.3f}   ({len(seeds)} paired seeds)")

    # ---- undertraining check ----------------------------------------------------
    print("\nUNDERTRAINING CHECK  (train loss and internal val at the last two epochs)")
    for nm in names:
        for s in seeds:
            c = epoch_curve(nm, s)
            if len(c) < 2:
                continue
            (e1, l1, v1, _), (e2, l2, v2, _) = c[-2], c[-1]
            best = max(c, key=lambda r: r[2])[0]
            print(f"  {nm[:22]:<22} s{s}: loss {l1:.3f}->{l2:.3f}  val {v1:.3f}->{v2:.3f}  "
                  f"best val at epoch {best} of {e2}{'  <- still improving at the end' if best == e2 else ''}")

    # ---- ensembles ----------------------------------------------------------
    print("\nENSEMBLE  (mean of the seeds' probabilities, argmax)")
    ens = {nm: np.mean([np.load(RUNS / nm / f"seed{s}" / "dev_probs.npy") for s in seeds], 0) for nm in names}
    pred = {nm: p.argmax(1) for nm, p in ens.items()}
    print(f"  {'':<24}{'S2':>7}{'in-set':>8}{'named':>7}{'S1 leaf':>9}{'S1 level':>10}")
    for nm in names:
        p = pred[nm]
        print(f"  {nm[:24]:<24}{score_subtask2(p, mask).macro_f1:>7.3f}{mask[idx, p].mean():>8.3f}"
              f"{len(set(p.tolist())):>5}/9{score_subtask1(leaf_to_official_clarity(p), clar).macro_f1:>9.3f}"
              f"{score_subtask1(level_pred(ens[nm]), clar).macro_f1:>10.3f}")

    rng = np.random.default_rng(0)
    boots = [rng.integers(0, n, n) for _ in range(2000)]
    for nm in names[1:]:
        d = np.array([score_subtask2(pred[nm][b], mask[b]).macro_f1 - score_subtask2(pred[ref][b], mask[b]).macro_f1
                      for b in boots])
        print(f"  bootstrap {nm} - {ref}, S2 ensemble: {d.mean():+.3f}  95% [{np.percentile(d, 2.5):+.3f}, "
              f"{np.percentile(d, 97.5):+.3f}]  (items resampled, seeds fixed)")

    # ---- per class ----------------------------------------------------------
    print("\nPER CLASS  (mean single-model F1, from each run's analysis.json)")
    an = {nm: json.loads((RUNS / nm / "analysis.json").read_text()) for nm in names
          if (RUNS / nm / "analysis.json").exists()}
    print(f"  {'class':<22}" + "".join(f"{nm[:14]:>16}" for nm in an))
    for c in EVASION_LABELS:
        print(f"  {c:<22}" + "".join(f"{an[nm]['per_class_subtask2'][c]:>16.3f}" for nm in an))

    # ---- by input length (as in the E8 analysis) -------------------------------
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    qa_a, qa_b = build_inputs(sp.dev, "qa")
    fu_a, _ = build_inputs(sp.dev, "full")
    ln = lambda t: len(tok(t, add_special_tokens=False)["input_ids"])
    fits512 = np.array([min(ln(a), 96) + ln(b) + 3 <= 512 for a, b in zip(qa_a, qa_b)])
    fits1024 = np.array([min(ln(a), 256) + ln(b) + 3 <= 1024 for a, b in zip(fu_a, qa_b)])
    fullq_cut = np.array([ln(a) > 256 for a in fu_a])
    groups = [("short: fit in 512 tokens for E0", fits512),
              ("long: E0 had to cut the answer", ~fits512),
              ("  ... of which 1024 saw in full", ~fits512 & fits1024 & ~fullq_cut)]
    print("\nBY INPUT LENGTH  (in-set rate of the ensemble)")
    print(f"  {'dev items':<34}{'n':>5}" + "".join(f"{nm[:14]:>16}" for nm in names))
    for label, g in groups:
        print(f"  {label:<34}{g.sum():>5}" + "".join(f"{mask[idx, pred[nm]][g].mean():>16.3f}" for nm in names))


if __name__ == "__main__":
    main()
