"""E19 helpers: the cascade's inputs per system and a vectorised copy of cascade.fuse.fit_system.

`fast_fit` searches the same (lam, tau) grid as cascade.fuse.fit_system, in the same order (lam outer, tau
inner, first maximum kept), but scores all 915 grid points in one array operation; it exists so that the
E19 bootstrap can refit the system inside every replicate. e19_tier0 checks that it reproduces
fit_system's predictions on dev before using it.
"""

from __future__ import annotations

import json

import numpy as np

from cascade.adjudicate import M1_INPUTS
from cascade.candidates import build
from cascade.frontier import m3_lookup, top4
from cascade.fuse import LAMS, LOGPI, PRIOR, TAUS
from qevasion.labels import N_EVASION
from qevasion.paths import RUNS

M1, STUDENT, TAG = "Q8_fullq_lora_12ep", "Q8_distill_a05", "qwen38_27b_k0"
LOGPI_FIT = np.log(PRIOR + 1e-12)  # decide.logit_adjust's prior term (fit_tau); decisions use LOGPI


def macro_f1_many(P: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Multi-reference macro-F1 (qevasion.scoring.macro_f1_multireference) of each row of P (k, n) vs G (n, 9)."""
    oh = P[..., None] == np.arange(N_EVASION)
    hit_c = oh & G
    tp = hit_c.sum(1)
    fp = (oh & ~G).sum(1)
    hit = hit_c.any(2)
    fn = (G[None] & ~oh & ~hit[..., None]).sum(1)
    tp, fp, fn = (x.astype(np.float64) for x in (tp, fp, fn))
    with np.errstate(invalid="ignore", divide="ignore"):
        pr = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rc = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(pr + rc > 0, 2 * pr * rc / (pr + rc), 0.0)
    return f1.mean(1)


def m1_grid(lp: np.ndarray, taus: np.ndarray = TAUS) -> tuple[np.ndarray, np.ndarray]:
    """(len(taus), n) M1 decisions argmax(lp - tau log pi): with LOGPI_FIT (for fitting) and with LOGPI (decisions)."""
    fitp = (lp[None] - taus[:, None, None] * LOGPI_FIT).argmax(2)
    dec = (lp[None] - taus[:, None, None] * LOGPI).argmax(2)
    return fitp, dec


def fused_grid(lp: np.ndarray, lq: np.ndarray, lams: np.ndarray = LAMS, taus: np.ndarray = TAUS) -> np.ndarray:
    """(len(lams) * len(taus), n) fused argmax over C(x) for every grid point, lam-major (cascade.fuse.decide)."""
    in_c = np.isfinite(lq)
    lq0 = np.where(in_c, lq, 0.0)
    out = []
    for lam in lams:
        base = np.where(in_c, lam * lp + (1 - lam) * lq0, -np.inf)
        s = np.where(in_c[None], base[None] - taus[:, None, None] * LOGPI, -np.inf)
        out.append(s.argmax(2))
    return np.concatenate(out)


class Grid:
    """Precomputed decisions of one system on every grid point; fit() is then a scoring problem only."""

    def __init__(self, lp, lq, defer, lams=LAMS, taus=TAUS):
        self.m1_fit, self.m1_dec = m1_grid(lp, taus)
        self.fused = fused_grid(lp, lq, lams, taus) if defer.any() else None
        self.defer, self.lams, self.taus = defer, np.asarray(lams), np.asarray(taus)

    def fit_m1(self, gold, rows) -> tuple[np.ndarray, int]:
        f = macro_f1_many(self.m1_fit[:, rows], gold[rows])
        j = int(np.argmax(f))
        return self.m1_dec[j], j

    def fit(self, gold, rows) -> tuple[np.ndarray, dict]:
        """Fit on `rows` (may repeat): M1's tau, then (lam, tau) of the deferred items. Predictions for all items."""
        m1, j1 = self.fit_m1(gold, rows)
        if self.fused is None:
            return m1, {"tau_m1": float(self.taus[j1])}
        P = np.where(self.defer[None], self.fused, m1[None])
        f = macro_f1_many(P[:, rows], gold[rows])
        j = int(np.argmax(f))
        lam, tau = self.lams[j // len(self.taus)], self.taus[j % len(self.taus)]
        return P[j], {"tau_m1": float(self.taus[j1]), "lam": float(lam), "tau": float(tau)}


def folds_of(n: int, seed: int, k: int = 5) -> list[np.ndarray]:
    """decision_rules.decide.nested_cv's folds."""
    return np.array_split(np.random.default_rng(seed).permutation(n), k)


def nested(grid: Grid, gold, seed: int, k: int = 5, m1_only: bool = False) -> np.ndarray:
    """Pooled held-out predictions with nested_cv's folds."""
    n = len(gold)
    pooled = np.empty(n, np.int64)
    folds = folds_of(n, seed, k)
    for f in range(k):
        te = folds[f]
        tr = np.concatenate([folds[j] for j in range(k) if j != f])
        pred = grid.fit_m1(gold, tr)[0] if m1_only else grid.fit(gold, tr)[0]
        pooled[te] = pred[te]
    return pooled


def boot_nested(grids: list[Grid], gold, idx: np.ndarray, rng, k: int = 5, m1_only=()) -> list[np.ndarray]:
    """One bootstrap replicate: items idx (with repeats); folds are formed over the DISTINCT items so a repeated
    item never sits on both sides of a split; every grid is refitted on the same folds. Returns, per grid,
    the held-out predictions of the replicate's rows (aligned with idx)."""
    uniq = np.unique(idx)
    fold_of = np.empty(len(gold), np.int64)
    fold_of[rng.permutation(uniq)] = np.arange(len(uniq)) % k
    f_idx = fold_of[idx]
    outs = [np.empty(len(idx), np.int64) for _ in grids]
    for f in range(k):
        te, tr = f_idx == f, idx[f_idx != f]
        for g, o, mo in zip(grids, outs, [i in m1_only for i in range(len(grids))]):
            pred = g.fit_m1(gold, tr)[0] if mo else g.fit(gold, tr)[0]
            o[te] = pred[idx[te]]
    return outs


# ---------------------------------------------------------------- inputs per system

def m3_records(tag: str = TAG) -> list[dict]:
    return [json.loads(line) for line in (RUNS / "M3" / tag / "responses.jsonl").read_text().splitlines()]


_LOOK: dict | None = None


def lookup() -> dict:
    global _LOOK
    if _LOOK is None:
        _LOOK = m3_lookup(TAG)
    return _LOOK


def lq_from(cand: np.ndarray, look: dict) -> tuple[np.ndarray, np.ndarray]:
    """(n, 9) log q (-inf outside C) and a mask of items M3 has scored with exactly these candidates."""
    lq = np.full((len(cand), N_EVASION), -np.inf)
    ok = np.zeros(len(cand), bool)
    for i, c in enumerate(cand):
        q = look.get((i, tuple(int(x) for x in c)))
        if q is not None:
            lq[i, c] = np.log(np.maximum(q, 1e-12))
            ok[i] = True
    return lq, ok


def system_inputs(run: str, seeds: tuple[int, ...]) -> dict:
    """p-bar, calibrated q, top-4 candidates and u = 1 - top q on dev, slice and test (as cascade.frontier)."""
    r = build(run, None, 0.05, list(seeds), method="fixed", k=4)
    out = {"seeds": seeds, "run": run}
    for s in ("dev", "val", "test"):
        out[f"pbar_{s}"], out[f"q_{s}"] = r["pbar"][s], r["q"][s]
        out[f"u_{s}"] = 1 - r["q"][s].max(1)
        out[f"cand_{s}"] = top4(r["q"][s])
        out[f"x_{s}"] = r["x"][s]
    out["lp_dev"] = np.log(out["pbar_dev"] + 1e-12)
    out["lq_dev"], ok = lq_from(out["cand_dev"], lookup())
    assert ok.all(), f"{run} {seeds}: M3 has not scored every dev item's top 4"
    return out


def defer_mask(u: np.ndarray, u_slice: np.ndarray, delta: float) -> np.ndarray:
    """cascade.fuse / frontier routing: the (1 - delta) quantile of u on the train slice."""
    if delta == 0:
        return np.zeros(len(u), bool)
    if delta == 1:
        return np.ones(len(u), bool)
    return u >= np.quantile(u_slice, 1 - delta)


def m1_npz() -> dict:
    z = np.load(M1_INPUTS)
    return {k: z[k] for k in z.files}
