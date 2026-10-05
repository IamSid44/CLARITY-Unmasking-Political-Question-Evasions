#!/usr/bin/env python3
"""M1, the cascade's first stage: candidate sets C(x) and an uncertainty score u(x) from a seed ensemble.

    python -m cascade.candidates --run Q8_fullq_lora_12ep [--crossfit Q8_12ep_crossfit] [--alpha 0.05]
                                 [--method aps|topk|fixed --k 3]

Inputs (saved probabilities, CPU only):
  * K seeds of one configuration: p_s(c|x) on the train slice (val_probs.npy; the 345 rows every seed
    holds out), dev and test. Their mean is p-bar.
  * optionally a cross-fitted configuration (fold0..4): out-of-fold p(c|x) for all 3,448 train rows.

Everything is fitted on the TRAIN SLICE, so dev stays untouched:
  * tau_C   -- logit-adjustment strength for ranking candidates: argmax of slice macro-F1 over [-1, 2].
  * T       -- a temperature on the adjusted scores, fitted by slice NLL. The 12-epoch models are
              overconfident (mean top probability 0.81 on the slice), and without it the mass-based
              sets below degenerate to almost all nine labels.
  * C(x)    -- --method aps (default): the smallest set of top-ranked labels whose calibrated mass
              reaches t (adaptive prediction sets); t is the split-conformal quantile of the slice
              items' "mass needed to include the gold label", so gold is in C(x) for >= 1 - alpha of
              slice items. --method topk: the same guarantee with a fixed size k (conformal on the
              gold label's rank). --method fixed: the top --k labels, no guarantee.
              The cross-fitted train rows give an independent coverage check.
  * u(x)    -- logistic regression predicting an error of the adjusted argmax from: top probability,
              margin to the second label, entropy, seed disagreement, |C(x)|. Fitted on the slice.

Writes runs/M1/<run>/cascade_inputs.npz (per split: p-bar, adjusted scores, candidate mask, u, features)
and meta.json. Dev decisions of the full system keep the nested-CV tau (decision_rules/decide.py).
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from decision_rules.decide import logit_adjust
from qevasion.labels import N_EVASION, encode_evasion
from qevasion.loader import DATA_CACHE, load_qevasion
from qevasion.paths import RUNS
from qevasion.scoring import score_subtask2

TAU_GRID = np.linspace(-1.0, 2.0, 61)
FEATURES = ("top_prob", "margin", "entropy", "seed_disagreement", "set_size")


def seed_dirs(run: str, seeds: list[int] | None = None) -> list:
    dirs = sorted(RUNS.glob(f"{run}/seed*/metrics.json"), key=lambda p: int(p.parent.name[4:]))
    dirs = [p.parent for p in dirs]
    return [d for d in dirs if seeds is None or int(d.name[4:]) in seeds]


def load_split(run: str, split: str, seeds: list[int] | None = None) -> np.ndarray:
    """(K, N, 9) per-seed probabilities of one split (val = the train slice)."""
    return np.stack([np.load(d / f"{split}_probs.npy") for d in seed_dirs(run, seeds)])


def load_crossfit(run: str, n_train: int) -> np.ndarray:
    """(N_train, 9) out-of-fold probabilities assembled from fold<k>/val_probs.npy + val_index.npy."""
    out = np.full((n_train, N_EVASION), np.nan, dtype=np.float64)
    folds = sorted(RUNS.glob(f"{run}/fold*/metrics.json"))
    for m in folds:
        d = m.parent
        out[np.load(d / "val_index.npy")] = np.load(d / "val_probs.npy")
    return out


def normalise(s: np.ndarray) -> np.ndarray:
    return s / s.sum(1, keepdims=True)


def onehot(y: np.ndarray) -> np.ndarray:
    m = np.zeros((len(y), N_EVASION), dtype=bool)
    m[np.arange(len(y)), y] = True
    return m


def fit_tau_slice(p: np.ndarray, y: np.ndarray, prior: np.ndarray) -> float:
    gold = onehot(y)
    vals = [score_subtask2(logit_adjust(p, prior, t).argmax(1), gold).macro_f1 for t in TAU_GRID]
    return float(TAU_GRID[int(np.argmax(vals))])


def mass_to_include(q: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """Per item: cumulative mass of the sorted labels up to and including the first gold label."""
    order = np.argsort(-q, axis=1, kind="stable")
    cum = np.cumsum(np.take_along_axis(q, order, 1), 1)
    hit = np.take_along_axis(gold, order, 1)
    first = hit.argmax(1)
    return cum[np.arange(len(q)), first]


def conformal_threshold(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def candidate_mask(q: np.ndarray, t: float) -> np.ndarray:
    """Smallest top-ranked set whose mass reaches t (always at least the top label)."""
    order = np.argsort(-q, axis=1, kind="stable")
    cum = np.cumsum(np.take_along_axis(q, order, 1), 1)
    k = (cum < t - 1e-12).sum(1) + 1
    mask = np.zeros_like(q, dtype=bool)
    for i in range(len(q)):
        mask[i, order[i, : k[i]]] = True
    return mask


def features(p_seeds: np.ndarray, q: np.ndarray, cand: np.ndarray, prior: np.ndarray, tau: float) -> np.ndarray:
    """(N, 5) features of FEATURES; p_seeds (K, N, 9) per-seed probabilities, q adjusted and renormalised."""
    s = np.sort(q, 1)[:, ::-1]
    ent = -(q * np.log(q + 1e-12)).sum(1)
    top = q.argmax(1)
    per_seed = np.stack([logit_adjust(p, prior, tau).argmax(1) for p in p_seeds])
    disagree = (per_seed != top[None]).mean(0)
    return np.column_stack([s[:, 0], s[:, 0] - s[:, 1], ent, disagree, cand.sum(1)])


def temperature(q: np.ndarray, T: float) -> np.ndarray:
    lg = np.log(q + 1e-12) / T
    lg -= lg.max(1, keepdims=True)
    return normalise(np.exp(lg))


def fit_temperature(q: np.ndarray, y: np.ndarray) -> float:
    from scipy.optimize import minimize_scalar

    nll = lambda T: -np.log(temperature(q, T)[np.arange(len(y)), y] + 1e-12).mean()  # noqa: E731
    return float(minimize_scalar(nll, bounds=(0.25, 20.0), method="bounded").x)


def gold_rank(q: np.ndarray, gold: np.ndarray) -> np.ndarray:
    order = np.argsort(-q, axis=1, kind="stable")
    return np.take_along_axis(gold, order, 1).argmax(1) + 1


def topk_mask(q: np.ndarray, k: int) -> np.ndarray:
    mask = np.zeros_like(q, dtype=bool)
    np.put_along_axis(mask, np.argsort(-q, axis=1, kind="stable")[:, :k], True, 1)
    return mask


def make_sets(q_cal: np.ndarray, y_cal: np.ndarray, method: str, alpha: float, k: int):
    """Calibrate on the slice; return (fn: q -> candidate mask, parameter)."""
    if method == "aps":
        t = conformal_threshold(mass_to_include(q_cal, onehot(y_cal)), alpha)
        return (lambda q: candidate_mask(q, t)), t
    if method == "topk":
        kk = int(conformal_threshold(gold_rank(q_cal, onehot(y_cal)).astype(float), alpha))
        return (lambda q: topk_mask(q, kk)), kk
    if method == "fixed":
        return (lambda q: topk_mask(q, k)), k
    raise SystemExit(f"unknown --method {method}")


def fit_uncertainty(x: np.ndarray, err: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)).fit(x, err)


def build(run: str, crossfit: str | None, alpha: float, seeds: list[int] | None = None,
          method: str = "aps", k: int = 3) -> dict:
    sp = load_qevasion(DATA_CACHE)
    y = encode_evasion(sp.train["evasion_label"].tolist())
    prior = np.bincount(y, minlength=N_EVASION) / len(y)
    dirs = seed_dirs(run, seeds)
    if not dirs:
        raise SystemExit(f"no finished seeds under {RUNS / run}")
    val_idx = np.load(dirs[0] / "val_index.npy")
    for d in dirs[1:]:
        assert np.array_equal(np.load(d / "val_index.npy"), val_idx), f"{d}: different train slice"
    y_val = y[val_idx]

    P = {sp_: load_split(run, sp_, seeds) for sp_ in ("val", "dev", "test")}
    tau = fit_tau_slice(P["val"].mean(0), y_val, prior)
    T = fit_temperature(normalise(logit_adjust(P["val"].mean(0), prior, tau)), y_val)
    calib = lambda p: temperature(normalise(logit_adjust(p, prior, tau)), T)  # noqa: E731
    Q = {s_: calib(v.mean(0)) for s_, v in P.items()}
    sets, param = make_sets(Q["val"], y_val, method, alpha, k)
    C = {s_: sets(q) for s_, q in Q.items()}
    X = {k: features(P[k], Q[k], C[k], prior, tau) for k in P}
    err_val = (Q["val"].argmax(1) != y_val).astype(int)
    model = fit_uncertainty(X["val"], err_val)
    U = {k: model.predict_proba(x)[:, 1] for k, x in X.items()}

    out = {"pbar": {k: v.mean(0) for k, v in P.items()}, "q": Q, "cand": C, "u": U, "x": X}
    if crossfit:
        oof = load_crossfit(crossfit, len(y))
        ok = ~np.isnan(oof).any(1)
        if ok.any():
            q_tr = np.full_like(oof, np.nan)
            q_tr[ok] = calib(oof[ok])
            c_tr = np.zeros_like(oof, dtype=bool)
            c_tr[ok] = sets(q_tr[ok])
            out["pbar"]["train"], out["q"]["train"], out["cand"]["train"] = oof, q_tr, c_tr
    meta = {"run": run, "seeds": [int(d.name[4:]) for d in dirs], "crossfit": crossfit, "alpha": alpha,
            "tau_c": tau, "temperature": T, "method": method, "set_parameter": param, "features": list(FEATURES),
            "u_coef": model[-1].coef_.ravel().tolist(), "u_intercept": float(model[-1].intercept_[0])}
    return {"meta": meta, "y": y, "val_idx": val_idx, **out}


def save(res: dict) -> None:
    m = res["meta"]
    outdir = RUNS / "M1" / f"{m['run']}_{m['method']}_a{m['alpha']:g}"
    outdir.mkdir(parents=True, exist_ok=True)
    arrays = {"val_idx": res["val_idx"]}
    for key in ("pbar", "q", "cand", "u", "x"):
        for split, v in res[key].items():
            arrays[f"{split}_{key}"] = v
    np.savez_compressed(outdir / "cascade_inputs.npz", **arrays)
    (outdir / "meta.json").write_text(json.dumps(res["meta"], indent=2))
    print(f"wrote {outdir}/cascade_inputs.npz and meta.json")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="Q8_fullq_lora_12ep")
    p.add_argument("--crossfit", default=None, help="cross-fitted run name (fold0..4), e.g. Q8_12ep_crossfit")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--method", choices=("aps", "topk", "fixed"), default="aps")
    p.add_argument("--k", type=int, default=3, help="set size for --method fixed")
    p.add_argument("--seeds", default=None, help="comma-separated subset, default all finished")
    a = p.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else None
    res = build(a.run, a.crossfit, a.alpha, seeds, a.method, a.k)
    m = res["meta"]
    print(f"{m['run']} seeds {m['seeds']}: tau_C {m['tau_c']:.2f}, T {m['temperature']:.2f}, "
          f"{m['method']} parameter {m['set_parameter']:.3f} (alpha {m['alpha']})")
    for split, c in res["cand"].items():
        print(f"  {split:>5}: mean |C| {c.sum(1).mean():.2f}")
    save(res)


if __name__ == "__main__":
    main()
