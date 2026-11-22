#!/usr/bin/env python3
"""E33: fit the cross-fit instrument's logit-adjustment tau once, on the out-of-fold probabilities of the Presidents'
own train answers (runs/E34/cf/train_own.npz from audit.infer_cf), then freeze it in runs/E33/cf_tau.json. Also writes
the registered checks C33-1..3 to docs/raw/E33_checks.txt.

    python -m audit.cf_tau
"""

from __future__ import annotations

import json

import numpy as np

from cascade.fuse import PRIOR, TAUS
from decision_rules.decide import fit_tau
from qevasion.labels import N_EVASION, encode_evasion
from qevasion.loader import DATA_CACHE, load_qevasion
from qevasion.paths import ROOT, RUNS, SPLITS


def main() -> None:
    train = load_qevasion(DATA_CACHE).train.reset_index(drop=True)
    y = encode_evasion(train["evasion_label"].tolist())
    gold = np.zeros((len(train), N_EVASION), bool)
    gold[np.arange(len(train)), y] = True
    z = np.load(RUNS / "E34" / "cf" / "train_own.npz")
    probs = z["probs"]
    tau, s2 = fit_tau(probs, gold, PRIOR, TAUS)
    raw_s2 = fit_tau(probs, gold, PRIOR, np.array([0.0]))[1]
    out = RUNS / "E33"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cf_tau.json").write_text(json.dumps({"tau": tau, "oof_train_s2_at_tau": s2, "oof_train_s2_raw": raw_s2,
                                                 "grid": "cascade.fuse.TAUS", "prior": "cascade.fuse.PRIOR"}, indent=1))
    folds = json.loads((SPLITS / "train_5fold_url.json").read_text())["folds"]
    urls = train["url"].astype(str).to_numpy()
    c1 = all(not set(urls[f]) & set(urls[np.setdiff1d(np.arange(len(train)), f)]) for f in folds)
    L = ["E33 checks (registered 2026-11-21)", f"C33-1 no url in two folds: {'PASSED' if c1 else 'FAILED'}"]
    dev_s2 = []
    for k in range(5):
        m = json.loads((RUNS / "E33_cf_url" / f"fold{k}" / "metrics.json").read_text())
        dev_s2.append(m["dev_subtask2_macro_f1"])
    c2 = min(dev_s2) >= 0.45 and np.mean(dev_s2) >= 0.50
    L.append(f"C33-2 per-fold dev S2 (raw argmax) {', '.join(f'{v:.3f}' for v in dev_s2)}; min {min(dev_s2):.3f} (>= 0.45), "
             f"mean {np.mean(dev_s2):.3f} (>= 0.50): {'PASSED' if c2 else 'FAILED'}")
    rep = json.loads((RUNS / "E34" / "cf" / "report.json").read_text())
    ag = [rep[f"train_own/fold{k}"]["ctrl_argmax_agree"] for k in range(5)]
    c3 = min(ag) >= 0.97
    L.append(f"C33-3 infer_cf reproduces each fold's val_probs: argmax agreement {', '.join(f'{v:.3f}' for v in ag)} "
             f"(>= 0.97): {'PASSED' if c3 else 'FAILED'}")
    L.append(f"tau (fitted once on the out-of-fold train answers, then frozen): {tau:.2f}; out-of-fold train S2 "
             f"{raw_s2:.3f} raw -> {s2:.3f} adjusted")
    (ROOT / "docs" / "raw" / "E33_checks.txt").write_text("\n".join(L) + "\n", encoding="utf8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
