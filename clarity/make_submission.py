#!/usr/bin/env python3
"""Package test-set predictions for Codabench. One zip per subtask.

The format rules, taken from the three participant repos that ship their actual
submission code (TeleAI `scripts/make_submission.py`, KCLarity
`scripts/submissions/make_evasion_submission.py`, Duluth `evaluation/`):

  1. The upload is a .zip containing exactly ONE file, named `prediction`, with
     no extension, at the zip root (no folders).
  2. One label per line, one line per test row, 237 lines, UTF-8, "\\n" endings.
  3. Line i is the prediction for row i of the official evaluation CSV. Our copy
     (data/clarity_task_evaluation_dataset.csv) matches TeleAI's independent
     copy row by row, 237/237 on question and answer text.
  4. Labels are the canonical full strings ("Claims ignorance",
     "Partial/half-answer", ...), as TeleAI (1st place) and KCLarity submitted.
     Duluth's short forms ("Ignorance", "Partial") are NOT used: the scorer
     replica does not recognise them, and an unrecognised string scores as a
     miss against every reference label with no error raised.
  5. Subtask 1 (clarity) and Subtask 2 (evasion) are separate uploads. Subtask 1
     is derived from Subtask 2 through the fixed leaf->clarity table, which is
     what the 1st- and 2nd-place systems did.

Every rule is ASSERTED before anything is written.

    python make_submission.py --source stage1 --run L0_large_base
    python make_submission.py --source rerank --run R1_rerank
    python make_submission.py --source rerank --run R1_rerank --with-prior
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from qevasion.labels import (
    CLARITY_LABELS,
    EVASION_LABELS,
    OFFICIAL_CLARITY_OF_LEAF,
    normalize_clarity,
    normalize_evasion,
)

ROOT = Path(__file__).resolve().parent
RUNS = Path(os.environ.get("CLARITY_RUNS", ROOT / "runs"))
OUT = ROOT / "submissions"
N_TEST = 237


def consensus(p: np.ndarray) -> np.ndarray:
    return p.mean(axis=1) if p.ndim == 3 else p


# Negative tau is allowed: after Balanced-Softmax training the prior is already
# removed, and the metric may want some of it back. Same grid as decide.py.
TAU_GRID = np.round(np.arange(-1.0, 2.0001, 0.05), 2)


def fit_tau_on_dev(p_dev: np.ndarray) -> tuple[float, float, float]:
    """Post-hoc logit adjustment (Menon et al., ICLR 2021): argmax p(c|x) / pi_c^tau.

    One scalar, fitted on ALL 308 dev items against the official scorer. Under
    nested CV (reports/E0_decision_rules.txt) this rule's in-fold/out-of-fold gap
    was 0.018 on the baseline ensemble, i.e. one scalar does not overfit 308 items
    -- unlike the 9-multiplier rule R3, whose gap was 0.098.
    Returns (tau, dev macro-F1 at tau, dev macro-F1 at tau=0).
    """
    from qevasion.labels import encode_evasion
    from qevasion.loader import dev_reference_mask, load_qevasion
    from qevasion.scoring import score_subtask2

    sp = load_qevasion()
    y = encode_evasion(sp.train["evasion_label"].tolist())
    prior = np.bincount(y, minlength=len(EVASION_LABELS)) / len(y)
    mask = dev_reference_mask(sp.dev)
    score = {t: score_subtask2((np.log(p_dev + 1e-12) - t * np.log(prior)).argmax(1), mask).macro_f1
             for t in TAU_GRID}
    tau = max(score, key=score.get)
    return float(tau), score[tau], score[0.0], prior


def stage1_predictions(run: str, logit_adjust: bool = False) -> tuple[np.ndarray, dict]:
    # `run` may list several configurations, comma-separated, whose seeds form one
    # ensemble (E11's final system pools E10_fullq_16ep and E11_fullq_16ep).
    files = [f for r in run.split(",") for f in sorted((RUNS / r).glob("seed*/test_probs.npy"))]
    if not files:
        raise SystemExit(f"no test_probs.npy under {run}")
    p = np.mean([consensus(np.load(f)) for f in files], axis=0)
    dev = [json.loads((f.parent / "metrics.json").read_text())["dev_subtask2_macro_f1"] for f in files]
    info = {"seeds": [f"{f.parent.parent.name}/{f.parent.name}" for f in files], "decision": "seed-ensemble argmax",
            "dev_subtask2_per_seed": dev}
    if not logit_adjust:
        return p.argmax(1), info
    p_dev = np.mean([consensus(np.load(f.parent / "dev_probs.npy")) for f in files], axis=0)
    tau, f_tau, f_0, prior = fit_tau_on_dev(p_dev)
    info.update({"decision": f"seed-ensemble, logit-adjusted tau={tau}",
                 "tau": tau, "dev_ensemble_subtask2_at_tau": f_tau, "dev_ensemble_subtask2_argmax": f_0})
    return (np.log(p + 1e-12) - tau * np.log(prior)).argmax(1), info


def rerank_predictions(run: str, with_prior: bool) -> tuple[np.ndarray, dict]:
    dirs = sorted(d for d in (RUNS / run).glob("seed*") if (d / "test_scores.npy").exists())
    if not dirs:
        raise SystemExit(f"no test_scores.npy under {RUNS / run}")
    cands = [np.load(d / "test_cands.npy") for d in dirs]
    if any(not np.array_equal(cands[0], c) for c in cands[1:]):
        raise SystemExit("re-ranker seeds disagree on the candidate sets; cannot average scores")
    c = cands[0]
    s = np.mean([np.load(d / "test_scores.npy") for d in dirs], axis=0)
    metrics = [json.loads((d / "metrics.json").read_text()) for d in dirs]
    beta = 0.0
    if with_prior:
        beta = float(np.median([m["selected_beta"] for m in metrics]))
        cfg = metrics[0]["config"]
        s1 = sorted((RUNS / cfg["stage1_base"]).glob("seed*/test_probs.npy"))
        prior = np.mean([consensus(np.load(f)) for f in s1], axis=0)
        s = s + beta * np.log(np.take_along_axis(prior, c, axis=1) + 1e-12)
    pred = np.take_along_axis(c, s.argmax(1)[:, None], axis=1)[:, 0]
    key = "rerank_plus_prior" if with_prior else "rerank"
    dev = [m["dev"][key]["dev_subtask2_macro_f1"] for m in metrics]
    return pred, {"seeds": [d.name for d in dirs], "decision": f"seed-mean re-ranker scores, beta={beta}",
                  "dev_subtask2_per_seed": dev}


def write_zip(labels: list[str], folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    pred = folder / "prediction"
    with open(pred, "w", encoding="utf-8", newline="\n") as f:
        for lab in labels:
            f.write(lab + "\n")
    z = folder / "prediction.zip"
    if z.exists():
        z.unlink()
    with zipfile.ZipFile(z, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(pred, arcname="prediction")
    # Re-open and check what Codabench will actually see.
    with zipfile.ZipFile(z) as zf:
        assert zf.namelist() == ["prediction"], zf.namelist()
        lines = zf.read("prediction").decode("utf-8").split("\n")
    assert lines[-1] == "" and len(lines) - 1 == N_TEST, f"{len(lines) - 1} lines"
    return z


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("stage1", "rerank"), required=True)
    ap.add_argument("--run", required=True, help="configuration name; several, comma-separated, pool their seeds")
    ap.add_argument("--with-prior", action="store_true")
    ap.add_argument("--logit-adjust", action="store_true",
                    help="stage1 only: apply post-hoc logit adjustment, tau fitted on dev")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    if args.source == "stage1":
        idx, info = stage1_predictions(args.run, args.logit_adjust)
    else:
        idx, info = rerank_predictions(args.run, args.with_prior)

    assert idx.shape == (N_TEST,), idx.shape
    evasion = [EVASION_LABELS[int(i)] for i in idx]
    clarity = [OFFICIAL_CLARITY_OF_LEAF[e] for e in evasion]
    # Rule 4: every string must survive the scorer's own normalisation unchanged.
    assert all(normalize_evasion(e) == e and e in EVASION_LABELS for e in evasion)
    assert all(normalize_clarity(c) == c and c in CLARITY_LABELS for c in clarity)

    tag = args.tag or (args.run + ("_prior" if args.with_prior else "") + ("_logitadj" if args.logit_adjust else ""))
    base = OUT / tag
    z2 = write_zip(evasion, base / "subtask2_evasion")
    z1 = write_zip(clarity, base / "subtask1_clarity")

    manifest = {
        "source": args.source, "run": args.run, **info,
        "dev_subtask2_mean": float(np.mean(info["dev_subtask2_per_seed"])),
        "test_distribution_subtask2": dict(Counter(evasion).most_common()),
        "test_distribution_subtask1": dict(Counter(clarity).most_common()),
        "sha256": {str(z.relative_to(ROOT)): hashlib.sha256(z.read_bytes()).hexdigest()[:16] for z in (z1, z2)},
    }
    (base / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[submission] {tag}: {info['decision']}, {len(info['seeds'])} seed(s), "
          f"dev S2 mean {manifest['dev_subtask2_mean']:.4f}")
    print(f"  Subtask 2 (evasion) : {z2}")
    print(f"  Subtask 1 (clarity) : {z1}")
    print(f"  test distribution   : {manifest['test_distribution_subtask2']}")


if __name__ == "__main__":
    main()
