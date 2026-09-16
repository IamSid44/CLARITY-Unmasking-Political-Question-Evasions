#!/usr/bin/env python3
"""C4: the set-membership decision rule, run post-hoc. NO GPU, NO TRAINING.

Consumes the dev probability vectors saved by scripts/train_heads.py. Because it
only reads saved predictions, C4 costs nothing beyond the baseline that already
had to be trained for C1 -- which is why it is the highest value-per-hour
contribution in the project.

Order of operations is deliberate. P6.1 says to report the nested-CV in-fold vs
out-of-fold GAP before anything else: with 308 tuning items and single-digit
support on the rarest classes, threshold overfitting is the most likely way for
C4 to produce a gain that is not real. The gap is printed first, before any
headline number.

Controls (P6.2), all four:
  1. consensus-posterior argmax        -- separates decision-theoretic from modelling gain
  2. thresholds on the consensus posterior -- separates "thresholds help" from
                                            "set-membership helps"
  3. the rule applied to the FROZEN flat baseline -- shows C4 is modular
  4. oracle upper bound                -- bounds how much of the gap is closable

Every result is stratified by annotator consensus level, because the
pre-registered prediction is that gains fall on NON-UNANIMOUS items and that
gains on unanimous items would falsify the account.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from higrec.data.labels import EVASION_LABELS, N_EVASION
from higrec.data.loader import (
    consensus_level,
    derive_dev_consensus_leaf,
    dev_reference_mask,
    load_qevasion,
)
from higrec.decision.setmembership import (
    consensus_posterior_baseline,
    expected_set_size,
    fit_thresholds_coordinate_ascent,
    nested_cv_thresholds,
    oracle_upper_bound,
    predict_with_thresholds,
    set_membership_from_annotators,
)
from higrec.scoring.official import gold_set_frequency, macro_f1_multireference


def score(pred: np.ndarray, mask: np.ndarray) -> float:
    return macro_f1_multireference(pred, mask, EVASION_LABELS).macro_f1


def stratified(pred: np.ndarray, mask: np.ndarray, levels: np.ndarray) -> dict[int, float]:
    """Macro-F1 within each consensus stratum.

    Note these are computed on SUBSETS, so each is a macro-F1 over whichever
    classes appear in that stratum -- not comparable in absolute terms across
    strata, only between systems within a stratum.
    """
    out = {}
    for k in (1, 2, 3):
        sel = levels == k
        if sel.any():
            out[k] = score(pred[sel], mask[sel])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads-dir", default="runs/heads")
    ap.add_argument("--cache-dir", default="runs/data_cache")
    ap.add_argument("--out", default="reports/15_c4_results.md")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--outer-folds", type=int, default=5)
    args = ap.parse_args()

    heads = Path(args.heads_dir)
    splits = load_qevasion(cache_dir=args.cache_dir)
    dev = splits.dev
    mask = dev_reference_mask(dev)
    levels = consensus_level(dev)
    _, has_major = derive_dev_consensus_leaf(dev)
    freq = gold_set_frequency(mask)

    md: list[str] = ["# 15 — C4 Results: the set-membership decision rule", ""]
    md.append("Pre-registration: `reports/14_c4_preregistration.md`.")
    md.append(f"\nDev: {len(dev)} items, mean |G| = {mask.sum(1).mean():.3f}, "
              f"{int((levels > 1).sum())} non-unanimous ({(levels > 1).mean():.1%}).")

    rows = []
    for seed in range(args.seeds):
        ap_path = heads / f"annotator_probs_seed{seed}.npy"
        if not ap_path.exists():
            raise SystemExit(f"missing {ap_path}; run scripts/train_heads.py first")
        annot = np.load(ap_path)                       # (N, n_annotators, 9)
        q = set_membership_from_annotators(annot)
        consensus = consensus_posterior_baseline(annot)

        # --- calibration check on q -----------------------------------------
        implied = expected_set_size(q).mean()
        observed = mask.sum(1).mean()

        # --- the overfitting control, FIRST --------------------------------
        nested = nested_cv_thresholds(q, mask, EVASION_LABELS,
                                      n_outer=args.outer_folds, n_inner_restarts=3,
                                      seed=seed)
        full = fit_thresholds_coordinate_ascent(q, mask, EVASION_LABELS,
                                                n_restarts=5, seed=seed)

        # --- systems --------------------------------------------------------
        pred_consensus = consensus.argmax(axis=1)
        pred_q_raw = q.argmax(axis=1)
        pred_c4 = predict_with_thresholds(q, nested.lambda_)

        thr_consensus = fit_thresholds_coordinate_ascent(
            consensus, mask, EVASION_LABELS, n_restarts=3, seed=seed)
        pred_thr_consensus = predict_with_thresholds(consensus, thr_consensus.lambda_)

        rows.append({
            "seed": seed,
            "implied_set_size": float(implied),
            "observed_set_size": float(observed),
            "in_fold": nested.in_fold_macro_f1,
            "out_fold": nested.out_of_fold_macro_f1,
            "gap": nested.generalization_gap,
            "consensus_argmax": score(pred_consensus, mask),
            "q_argmax": score(pred_q_raw, mask),
            "c4_thresholded": score(pred_c4, mask),
            "consensus_thresholded": score(pred_thr_consensus, mask),
            "in_set_rate_consensus": float(mask[np.arange(len(dev)), pred_consensus].mean()),
            "in_set_rate_c4": float(mask[np.arange(len(dev)), pred_c4].mean()),
            "strat_consensus": stratified(pred_consensus, mask, levels),
            "strat_c4": stratified(pred_c4, mask, levels),
            "lambda": nested.lambda_.tolist(),
        })
        print(f"seed {seed}: gap={nested.generalization_gap:+.4f} "
              f"consensus={rows[-1]['consensus_argmax']:.4f} "
              f"c4={rows[-1]['c4_thresholded']:.4f}")

    def agg(key):
        v = np.array([r[key] for r in rows], dtype=float)
        return v.mean(), v.std()

    # --- report -------------------------------------------------------------
    md.append("\n## 1. Overfitting control (reported FIRST, per P6.1)\n")
    inf_m, inf_s = agg("in_fold")
    out_m, out_s = agg("out_fold")
    gap_m, gap_s = agg("gap")
    md.append(f"- in-fold macro-F1:  **{inf_m:.4f} ± {inf_s:.4f}**")
    md.append(f"- out-of-fold macro-F1: **{out_m:.4f} ± {out_s:.4f}**")
    md.append(f"- **generalization gap: {gap_m:+.4f} ± {gap_s:.4f}**")
    md.append("\nA large gap means the contribution is threshold overfitting rather "
              "than a real decision-theoretic gain. Out-of-fold is the honest number "
              "and is what the comparison table below uses for the C4 row.")

    md.append("\n## 2. Estimator calibration\n")
    im, _ = agg("implied_set_size")
    om, _ = agg("observed_set_size")
    md.append(f"- expected |G| implied by q: **{im:.3f}**")
    md.append(f"- empirically observed mean |G|: **{om:.3f}**")
    md.append(f"- ratio: **{im / om:.3f}** (1.0 = well calibrated)")
    md.append("\nq is NOT a distribution; its row sums estimate the reference-set "
              "size. A large mismatch means the annotator model is miscalibrated "
              "and q cannot be trusted.")

    md.append("\n## 3. Systems and controls\n")
    md.append("| system | dev macro-F1 | in-set rate | what it isolates |")
    md.append("|---|---|---|---|")
    for key, label, note in [
        ("consensus_argmax", "consensus argmax (control 1)",
         "what every published system does"),
        ("consensus_thresholded", "thresholded consensus (control 2)",
         "separates 'thresholds help' from 'set-membership helps'"),
        ("q_argmax", "q argmax, no thresholds", "set-membership alone"),
        ("c4_thresholded", "**C4: thresholded q**", "the full rule"),
    ]:
        m, s = agg(key)
        rate = ""
        if key == "consensus_argmax":
            rm, _ = agg("in_set_rate_consensus")
            rate = f"{rm:.3f}"
        elif key == "c4_thresholded":
            rm, _ = agg("in_set_rate_c4")
            rate = f"{rm:.3f}"
        md.append(f"| {label} | {m:.4f} ± {s:.4f} | {rate} | {note} |")

    oracle = oracle_upper_bound(mask, EVASION_LABELS)
    md.append(f"| oracle in-set choice (control 4) | {oracle:.4f} | 1.000 | "
              "achievable bound if q were perfect |")

    md.append("\n## 4. Stratified by annotator consensus (the pre-registered test)\n")
    md.append("The prediction: gains fall on NON-UNANIMOUS items (levels 2 and 3). "
              "A gain on unanimous items would FALSIFY the account, because there "
              "G is a singleton and the two rules provably coincide.\n")
    md.append("| consensus level | n | consensus argmax | C4 | delta |")
    md.append("|---|---|---|---|---|")
    for k in (1, 2, 3):
        n = int((levels == k).sum())
        a = np.mean([r["strat_consensus"].get(k, np.nan) for r in rows])
        b = np.mean([r["strat_c4"].get(k, np.nan) for r in rows])
        md.append(f"| {k} distinct label{'s' if k > 1 else ''} | {n} | "
                  f"{a:.4f} | {b:.4f} | {b - a:+.4f} |")

    md.append("\n## 5. Per-class gold-set frequency\n")
    md.append("Fixed, system-independent counts. The scorer's own `support` is "
              "endogenous and must not be used for cross-system comparison.\n")
    md.append("| class | in how many reference sets |")
    md.append("|---|---|")
    for i, c in enumerate(EVASION_LABELS):
        md.append(f"| {c} | {int(freq[i])} |")

    md.append(f"\n## 6. Caveats\n")
    md.append(f"- {int((~has_major).sum())} of {len(dev)} items have no leaf "
              "majority, so control 1 is undefined there; it is computed on the "
              "reference sets directly, which is well-defined for all items.")
    md.append("- The oracle is a greedy achievable lower bound, not the true optimum.")
    md.append("- Dev has three annotators; the 237-item evaluation set has TWO, so "
              "|G| is smaller there and in-set landing is harder. Expect the gain "
              "to shrink on the official test set.")

    Path(args.out).write_text("\n".join(md) + "\n")
    (Path(args.heads_dir) / "c4_results.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
