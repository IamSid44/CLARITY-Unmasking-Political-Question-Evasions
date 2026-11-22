#!/usr/bin/env python3
"""E34/E35: jobs for the 27B zero-shot instrument (ZS), all nine labels as candidates, one per row.

    python -m audit.zs_jobs llm   runs/E34/jobs_zs_llm.jsonl     # E34: every row of <E25_DIR>/frames/llm.parquet
    python -m audit.zs_jobs train runs/E34/jobs_zs_train.jsonl   # E35: the Presidents' own train answers (frame "train")
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from qevasion.labels import N_EVASION
from qevasion.loader import DATA_CACHE, load_qevasion
from qevasion.paths import E25_DIR


def main() -> None:
    frame, out = sys.argv[1], sys.argv[2]
    n = len(load_qevasion(DATA_CACHE).train) if frame == "train" else len(pd.read_parquet(E25_DIR / "frames" / f"{frame}.parquet"))
    split = "e35zs" if frame == "train" else "e34zs"
    with open(out, "w") as f:
        for i in range(n):
            f.write(json.dumps({"split": split, "frame": frame, "i": i, "cand": list(range(N_EVASION))}) + "\n")
    print(f"{n} jobs -> {out}")


if __name__ == "__main__":
    main()
