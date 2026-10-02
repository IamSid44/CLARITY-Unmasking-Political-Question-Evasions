#!/usr/bin/env python3
"""Look inside a run -- finished or still training -- without disturbing it."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

from qevasion.labels import EVASION_LABELS, N_EVASION, encode_evasion
from qevasion.loader import dev_reference_mask, load_qevasion
from qevasion.scoring import score_subtask2


def dev_probs_of(run: Path) -> tuple[np.ndarray, str]:
    if (run / "dev_probs.npy").exists():
        return np.load(run / "dev_probs.npy"), "finished run, selected epoch"
    if (run / "resume.pt").exists():
        import torch

        st = torch.load(run / "resume.pt", map_location="cpu", weights_only=False)
        ex = st["extra"]
        return ex["best_probs"][0], (f"in progress: {st['epoch'] + 1} epoch(s) done, "
                                     f"best so far = epoch {ex['best']['epoch']}")
    raise SystemExit(f"nothing to read in {run} yet (no dev_probs.npy or resume.pt)")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    run = Path(sys.argv[1])
    p, what = dev_probs_of(run)
    if p.ndim == 3:
        p = p.mean(axis=1)
    sp = load_qevasion()
    mask = dev_reference_mask(sp.dev)
    y = encode_evasion(sp.train["evasion_label"].tolist())
    prior = np.bincount(y, minlength=N_EVASION) / len(y)

    pred = p.argmax(1)
    count = Counter(pred.tolist())
    n = len(pred)
    print(f"{run}  ({what})\n")
    print(f"  {'class':<22}{'predicted':>10}{'acceptable':>12}")
    for i, name in enumerate(EVASION_LABELS):
        k, g = count.get(i, 0), int(mask[:, i].sum())
        flag = "   <- over-predicted" if k > 2 * g and k > 20 else ""
        print(f"  {name:<22}{k:>10}{g:>12}{flag}")
    print(f"\n  {'tau':>6}{'dev S2':>9}{'in-set':>9}")
    for t in (1.0, 0.5, 0.0, -0.25, -0.5, -0.75, -1.0):
        q = (np.log(p + 1e-12) - t * np.log(prior)).argmax(1)
        print(f"  {t:>+6.2f}{score_subtask2(q, mask).macro_f1:>9.4f}{mask[np.arange(n), q].mean():>9.3f}")
    print("\n  tau > 0 removes more of the class prior (favours rare classes); tau < 0 adds it back.")


if __name__ == "__main__":
    main()
