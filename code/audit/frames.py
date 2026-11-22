#!/usr/bin/env python3
"""E25 input frames: the same (question, answer) items with one controlled change each (CPU, seconds).

    python -m audit.frames build        # writes runs/E25/frames/<split>_<condition>.parquet
    python -m audit.frames llm          # after audit.generate: runs/E25/frames/llm.parquet

E25a, question swaps (the answer is unchanged):
  own   -- the item as is (control: must reproduce the saved probabilities)
  rand  -- sub-question and interview turn of a random row from a different interview (seed 25)
  null  -- empty sub-question and interview turn
  qonly -- (dev only) the question kept, the answer replaced by one constant, contentless sentence: how much of
           the label the question alone carries (the E25b control for question-driven labelling bias)
No sibling swap: siblings share the interview turn and the answer, so a sibling's sub-question makes the row
identical to the sibling's own row (100% of dev siblings); E24 P2 compares those directly.
E25c, answer prefixes (the question is unchanged): pre25 / pre50 / pre75 -- the first 25/50/75% of the answer's
words (rounded up).
Splits: dev, and the official test with normalized answers (the text the final system scores, E21b).
Every frame keeps `src_i` (row in the source split) and the columns the classifier and the adjudicator read.

E25b, the LLM frame: one row per generated answer (audit.generate), cleaned of markdown (format control), with
the dev item's sub-question and interview turn; `model`, `condition`, `src_i` identify it.
"""

from __future__ import annotations

import argparse
import json
import math
import re

import numpy as np
import pandas as pd

from qevasion.loader import DATA_CACHE, load_qevasion
from qevasion.normalize import normalize_answer
from qevasion.paths import E25_DIR, RUNS, TEST_CSV

OUT = E25_DIR / "frames"
KEEP = ["question", "interview_question", "interview_answer", "url", "president", "date"]
SEED = 25
PREFIXES = (25, 50, 75)
CONSTANT_ANSWER = "Well, thank you for the question."


def sources() -> dict[str, pd.DataFrame]:
    dev = load_qevasion(DATA_CACHE).dev.reset_index(drop=True)
    test = pd.read_csv(TEST_CSV)
    test["interview_answer"] = [normalize_answer(a) for a in test["interview_answer"].astype(str)]
    return {"dev": dev, "test": test}


def base(df: pd.DataFrame) -> pd.DataFrame:
    out = df[[c for c in KEEP if c in df]].copy().reset_index(drop=True)
    out["src_i"] = np.arange(len(df))
    return out


def sibling_of(df: pd.DataFrame) -> np.ndarray:
    """For each row, the next sub-question (cyclically, in row order) of the same answer; -1 if none."""
    sib = np.full(len(df), -1)
    key = df["interview_answer"].astype(str).str.split().str.join(" ")
    for _, g in df.groupby(key, sort=False):
        ix = g.index.to_numpy()
        if len(ix) > 1:
            sib[ix] = np.roll(ix, -1)
    return sib


def prefix(text: str, pct: int) -> str:
    w = str(text).split()
    return " ".join(w[: max(1, math.ceil(len(w) * pct / 100))])


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    summary = {}
    for split, df in sources().items():
        b = base(df)
        frames = {"own": b.copy()}
        # no sibling-swap frame: siblings share the interview turn and the answer, so a sibling's sub-question
        # turns the row into the sibling's own row (checked: 100% on dev); E24 P2 already compares those
        if split == "dev":
            f = b.copy()
            f["interview_answer"] = CONSTANT_ANSWER
            frames["qonly"] = f
        urls = df["url"].astype(str).to_numpy()
        pick = np.empty(len(df), int)
        for i in range(len(df)):
            while True:
                j = int(rng.integers(len(df)))
                if urls[j] != urls[i]:
                    pick[i] = j
                    break
        f = b.copy()
        f["rand_i"] = pick
        f["question"] = df["question"].to_numpy()[pick]
        f["interview_question"] = df["interview_question"].to_numpy()[pick]
        frames["rand"] = f
        f = b.copy()
        f["question"], f["interview_question"] = "", ""
        frames["null"] = f
        for p in PREFIXES:
            f = b.copy()
            f["interview_answer"] = [prefix(a, p) for a in df["interview_answer"]]
            frames[f"pre{p}"] = f
        for cond, f in frames.items():
            f.reset_index(drop=True).to_parquet(OUT / f"{split}_{cond}.parquet")
            summary[f"{split}_{cond}"] = len(f)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print("frames:", summary)


MD = [(re.compile(r"<think>.*?</think>", re.S), " "), (re.compile(r"\*\*|__|`"), ""),
      (re.compile(r"^\s{0,3}#{1,6}\s*", re.M), ""), (re.compile(r"^\s*([-*•]|\d+[.)])\s+", re.M), ""),
      (re.compile(r"\s+"), " ")]


def clean(text: str) -> str:
    """Format control: LLM answers lose markdown (headers, bullets, bold) and become one paragraph of prose."""
    t = str(text)
    for pat, rep in MD:
        t = pat.sub(rep, t)
    return t.strip()


def llm() -> None:
    rows = [json.loads(line) for line in (E25_DIR / "answers.jsonl").read_text().splitlines() if line.strip()]
    srcs = {"dev": sources()["dev"], "train": load_qevasion(DATA_CACHE).train.reset_index(drop=True)}
    out = []
    for r in rows:
        src = srcs[r.get("split", "dev")].iloc[r["src_i"]]  # E34: answers to train questions carry split="train"
        out.append({"question": src["question"], "interview_question": src["interview_question"],
                    "interview_answer": clean(r["answer"]), "url": src["url"], "president": src["president"],
                    "date": src["date"], "split": r.get("split", "dev"), "src_i": r["src_i"], "model": r["model"],
                    "condition": r["condition"],
                    "raw_answer": r["answer"], "finish": r.get("finish", ""), "n_words": len(clean(r["answer"]).split())})
    df = pd.DataFrame(out)
    empty = (df["n_words"] == 0).sum()
    df.loc[df["n_words"] == 0, "interview_answer"] = "(no answer)"
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "llm.parquet")
    print(f"llm frame: {len(df)} answers ({empty} empty); words median {df['n_words'].median():.0f}; "
          f"per model x condition:\n{df.groupby(['model', 'condition']).size().unstack()}")


def e34() -> None:
    """E33/E34 frames (CLARITY_E25_DIR=runs/E34): train_own (the Presidents' train answers: the cross-fit control),
    dev_llm (the three discovery models' dev N and P answers, protocol A, from runs/E32/llm_table.parquet) and
    dev_llm_rand (the same answers under the E25 dev_rand question swap: H33-1)."""
    OUT.mkdir(parents=True, exist_ok=True)
    tr = base(load_qevasion(DATA_CACHE).train)
    tr["split"] = "train"
    tr.to_parquet(OUT / "train_own.parquet")
    t = pd.read_parquet(RUNS / "E32" / "llm_table.parquet")
    t = t[t["condition"].isin(["N", "P"])].reset_index(drop=True)
    keep = KEEP + ["src_i", "model", "condition"]
    d = t[keep].copy()
    d["split"] = "dev"
    d.to_parquet(OUT / "dev_llm.parquet")
    dev = sources()["dev"]
    rand_i = pd.read_parquet(RUNS / "E25" / "frames" / "dev_rand.parquet").set_index("src_i")["rand_i"]
    r = d.copy()
    r["rand_i"] = rand_i.loc[r["src_i"]].to_numpy()
    r["question"] = dev["question"].to_numpy()[r["rand_i"]]
    r["interview_question"] = dev["interview_question"].to_numpy()[r["rand_i"]]
    r.to_parquet(OUT / "dev_llm_rand.parquet")
    print(f"train_own {len(tr)}, dev_llm {len(d)}, dev_llm_rand {len(r)} -> {OUT}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=("build", "llm", "e34"))
    a = p.parse_args()
    {"build": build, "llm": llm, "e34": e34}[a.what]()


if __name__ == "__main__":
    main()
