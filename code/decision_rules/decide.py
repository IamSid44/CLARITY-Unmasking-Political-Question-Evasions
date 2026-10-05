#!/usr/bin/env python3
"""Post-hoc decision rules over saved probability matrices."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from qevasion.labels import N_EVASION, encode_evasion, multi_reference_mask
from qevasion.loader import DATA_CACHE, dev_reference_mask, load_qevasion
from qevasion.scoring import score_subtask2

ANNOTATOR_COLUMNS = ("annotator1", "annotator2", "annotator3")


def consensus(probs: np.ndarray) -> np.ndarray:
    return probs.mean(axis=1) if probs.ndim == 3 else probs


def set_membership(probs: np.ndarray) -> np.ndarray:
    """P(c in G | x) = 1 - prod_a (1 - p_a(c|x)), annotators independent given x."""
    if probs.ndim == 2:
        return probs
    return 1.0 - np.prod(1.0 - probs, axis=1)


def logit_adjust(p: np.ndarray, prior: np.ndarray, tau: float) -> np.ndarray:
    """log p(c|x) - tau * log pi_c, exponentiated back to a positive score."""
    return np.exp(np.log(p + 1e-12) - tau * np.log(prior + 1e-12))


def macro_f1(scores: np.ndarray, gold_mask: np.ndarray) -> float:
    return score_subtask2(scores.argmax(1), gold_mask).macro_f1


def fit_tau(p: np.ndarray, gold_mask: np.ndarray, prior: np.ndarray, grid: np.ndarray):
    vals = [macro_f1(logit_adjust(p, prior, t), gold_mask) for t in grid]
    j = int(np.argmax(vals))
    return float(grid[j]), float(vals[j])


def fit_lambda(
    scores: np.ndarray, gold_mask: np.ndarray, n_rounds: int = 8, n_grid: int = 41
) -> tuple[np.ndarray, float]:
    """Coordinate ascent on per-class multipliers against the official scorer."""
    lam = np.ones(N_EVASION)
    best = macro_f1(scores * lam, gold_mask)
    grid = np.exp(np.linspace(-2.5, 2.5, n_grid))
    for _ in range(n_rounds):
        improved = False
        for c in range(N_EVASION):
            base = lam[c]
            for g in grid:
                lam[c] = base * g
                v = macro_f1(scores * lam, gold_mask)
                if v > best + 1e-9:
                    best, base, improved = v, lam[c], True
            lam[c] = base
        if not improved:
            break
    return lam, float(best)


def nested_cv(
    score_fn, probs: np.ndarray, gold_mask: np.ndarray, n_folds: int, seed: int
) -> dict:
    """Fit the rule's parameters inside each fold, predict the held-out fold."""
    n = len(probs)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    folds = np.array_split(order, n_folds)
    pooled = np.empty(n, dtype=np.int64)
    in_fold = []
    for f in range(n_folds):
        te = folds[f]
        tr = np.concatenate([folds[j] for j in range(n_folds) if j != f])
        pred_te, fit_score = score_fn(probs, gold_mask, tr, te)
        pooled[te] = pred_te
        in_fold.append(fit_score)
    return {
        "out_of_fold_macro_f1": score_subtask2(pooled, gold_mask).macro_f1,
        "in_fold_macro_f1": float(np.mean(in_fold)),
        "generalization_gap": float(np.mean(in_fold)) - score_subtask2(pooled, gold_mask).macro_f1,
        "pooled_predictions": pooled,
    }


def make_rules(train_prior: np.ndarray):
    """Each rule is `fn(probs, gold_mask, tr_idx, te_idx) -> (pred_te, in_fold_f1)`."""

    def r0(probs, gold, tr, te):
        p = consensus(probs)
        return p[te].argmax(1), macro_f1(p[tr], gold[tr])

    def r1(probs, gold, tr, te):
        p = consensus(probs)
        tau, fit = fit_tau(p[tr], gold[tr], train_prior, np.linspace(-1.0, 2.0, 61))
        return logit_adjust(p[te], train_prior, tau).argmax(1), fit

    def r2(probs, gold, tr, te):
        s = set_membership(probs)
        return s[te].argmax(1), macro_f1(s[tr], gold[tr])

    def r3(probs, gold, tr, te):
        s = set_membership(probs)
        lam, fit = fit_lambda(s[tr], gold[tr])
        return (s[te] * lam).argmax(1), fit

    return {
        "R0_argmax_consensus": r0,
        "R1_logit_adjusted": r1,
        "R2_set_membership": r2,
        "R3_weighted_set_membership": r3,
    }


def reference_masks(dev_df, drop_annotator: bool) -> dict[str, np.ndarray]:
    """The 3-annotator gold mask, plus every 2-annotator subset if requested."""
    masks = {"3ann": dev_reference_mask(dev_df)}
    if drop_annotator:
        for pair in combinations(ANNOTATOR_COLUMNS, 2):
            refs = [[r[c] for c in pair] for _, r in dev_df.iterrows()]
            masks["+".join(pair)] = multi_reference_mask(refs)
    return masks


def load_probs(run_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Mean the per-seed probability matrices AND keep them separately."""
    seeds = sorted(run_dir.glob("seed*"))
    mats, names = [], []
    for s in seeds:
        f = s / "dev_probs.npy"
        if f.exists():
            mats.append(np.load(f))
            names.append(s.name)
    if not mats:
        raise SystemExit(f"no dev_probs.npy under {run_dir}")
    return np.stack(mats), names


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, help="runs/<config name>")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--drop-annotator", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    splits = load_qevasion(DATA_CACHE)
    dev_df = splits.dev
    col = "evasion_label" if "evasion_label" in splits.train else "label"
    y_train = encode_evasion(splits.train[col].tolist())
    prior = np.bincount(y_train, minlength=N_EVASION).astype(np.float64)
    prior /= prior.sum()

    all_probs, seed_names = load_probs(run_dir)
    ensemble = all_probs.mean(axis=0)
    rules = make_rules(prior)
    masks = reference_masks(dev_df, args.drop_annotator)

    report: dict = {"run_dir": str(run_dir), "seeds": seed_names, "results": {}}

    for mask_name, gold in masks.items():
        block: dict = {}
        for rule_name, fn in rules.items():
            per_seed = [
                nested_cv(fn, all_probs[i], gold, args.n_folds, args.seed)
                for i in range(len(all_probs))
            ]
            oof = np.array([r["out_of_fold_macro_f1"] for r in per_seed])
            gap = np.array([r["generalization_gap"] for r in per_seed])
            ens = nested_cv(fn, ensemble, gold, args.n_folds, args.seed)
            block[rule_name] = {
                "per_seed_oof_mean": float(oof.mean()),
                "per_seed_oof_std": float(oof.std(ddof=1)) if len(oof) > 1 else 0.0,
                "per_seed_gap_mean": float(gap.mean()),
                "ensemble_oof": ens["out_of_fold_macro_f1"],
                "ensemble_gap": ens["generalization_gap"],
            }
        report["results"][mask_name] = block

        print(f"\n=== reference sets: {mask_name} ===")
        print(f"{'rule':<30} {'per-seed OOF':>18} {'gap':>8} {'ens OOF':>9} {'ens gap':>8}")
        for k, v in block.items():
            print(
                f"{k:<30} {v['per_seed_oof_mean']:.4f} +/- {v['per_seed_oof_std']:.4f}"
                f" {v['per_seed_gap_mean']:>8.4f} {v['ensemble_oof']:>9.4f}"
                f" {v['ensemble_gap']:>8.4f}"
            )

    out = Path(args.out) if args.out else run_dir / "decision_rules.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
