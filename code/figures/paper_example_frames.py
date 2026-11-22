"""The paper's Figure 1 example: one answer ("I already have.") given to two sub-questions of one turn (dev rows 147,
206), labelled by the frozen instrument under the E25 frames (own / question from another interview / no question /
constant answer). Prints and writes docs/raw/paper_fig1_example.txt (the figure's only source).

    python -m figures.paper_example_frames
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qevasion.labels import EVASION_LABELS
from qevasion.loader import DATA_CACHE, load_qevasion
from qevasion.paths import RAW, RUNS

ROWS = (147, 206)
FRAMES = {"own": "own sub-question", "rand": "sub-question from another interview", "null": "no question",
          "qonly": "own sub-question, constant answer"}


def main() -> None:
    d = load_qevasion(DATA_CACHE).dev.reset_index(drop=True)
    L = ["Figure 1 example (dev): one answer, two sub-questions; labels of the frozen instrument (E25 replay)",
         f"turn: {' '.join(str(d.interview_question[ROWS[0]]).split())}",
         f"answer: {d.interview_answer[ROWS[0]].strip()}", ""]
    for i in ROWS:
        L.append(f"row {i}: annotators {d.annotator1[i]}, {d.annotator2[i]}, {d.annotator3[i]}")
        for f, name in FRAMES.items():
            z = np.load(RUNS / "E25" / "labels" / f"dev_{f}.npz")
            q = pd.read_parquet(RUNS / "E25" / "frames" / f"dev_{f}.parquet").question[i]
            a = pd.read_parquet(RUNS / "E25" / "frames" / f"dev_{f}.parquet").interview_answer[i]
            L.append(f"   {name:<36} -> {EVASION_LABELS[z['pred'][i]]:<10} | Q: {' '.join(str(q).split())} | A: {' '.join(str(a).split())[:60]}")
        L.append("")
    (RAW / "paper_fig1_example.txt").write_text("\n".join(L) + "\n", encoding="utf8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
