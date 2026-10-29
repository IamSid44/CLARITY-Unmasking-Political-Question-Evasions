#!/usr/bin/env python3
"""E21: transcript artifacts of the official test answers -- removing them, and injecting them into dev.

    python -m qevasion.normalize check        # artifact rates per split, before and after normalization

The official test answers come from presidential transcripts and carry formatting that train and dev do not:
a leading speaker tag ("THE PRESIDENT.", "The President.", "President Bush."), footnote markers ("[25.]"),
all-caps header lines, and turns by other speakers ("The Prime Minister. ..."). `normalize_answer` removes
them, by the rules registered in docs/02_experiment_log.md §E21:
  * drop the leading speaker tag, and the tag of any later turn by the president;
  * drop footnote markers;
  * drop all-caps header lines;
  * cut the answer at the first line that opens a turn by another speaker ("Q." or "Title Name.").
`inject_artifacts` does the reverse on dev answers, at the rates measured on test (E21a's sensitivity check).
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from qevasion.paths import TEST_CSV

PRESIDENT = r"(?:THE PRESIDENT|The President|President(?: [A-Z][A-Za-z'-]+){1,2})"
LEAD_TAG = re.compile(rf"^\s*{PRESIDENT}\.\s+")
OTHER = (r"(?:Q|THE [A-Z][A-Z .'-]{2,40}|The (?:Prime Minister|Chancellor|Vice President|First Lady|Secretary|"
         r"[A-Z][a-z]+)(?: [A-Z][A-Za-z'-]+){0,3}|(?:Prime Minister|Chancellor|Secretary|Senator|Governor|General|"
         r"Mr|Mrs|Ms|Dr)\.? [A-Z][A-Za-z'-]+(?: [A-Z][A-Za-z'-]+)?)")
TURN = re.compile(rf"(?m)^\s*({PRESIDENT}|{OTHER})\.\s")
FOOTNOTE = re.compile(r"\s*\[\d+\.?\]")
HEADER = re.compile(r"(?m)^[ \t]*[A-Z][A-Z0-9 ,.'&:;/-]{5,}[ \t]*$\n?")


def normalize_answer(a: str) -> str:
    s = LEAD_TAG.sub("", str(a).strip(), count=1)
    out, pos = [], 0
    for m in TURN.finditer(s):
        if re.fullmatch(PRESIDENT, m.group(1)):  # the president resumes: keep the text, drop the tag
            out.append(s[pos:m.start()])
            pos = m.end()
        else:  # another speaker: the answer ends here
            out.append(s[pos:m.start()])
            pos = len(s)
            break
    out.append(s[pos:])
    s = "".join(out)
    s = FOOTNOTE.sub("", s)
    s = HEADER.sub("", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def artifact_flags(a: str) -> dict[str, bool]:
    s = str(a).strip()
    turns = [m.group(1) for m in TURN.finditer(s) if m.start() > 0]
    return {"lead_tag": bool(LEAD_TAG.match(s)), "footnote": bool(FOOTNOTE.search(s)),
            "header": bool(HEADER.search(s)), "other_turn": any(not re.fullmatch(PRESIDENT, t) for t in turns)}


def test_inventory() -> dict:
    """What test's artifacts look like and how often they occur (E21a injects at these rates)."""
    ans = pd.read_csv(TEST_CSV)["interview_answer"].astype(str).tolist()
    flags = pd.DataFrame([artifact_flags(a) for a in ans])
    tags = [LEAD_TAG.match(a.strip()).group(0).strip() for a in ans if LEAD_TAG.match(a.strip())]
    headers = [h.strip() for a in ans for h in HEADER.findall(str(a).strip())]
    turns = []
    for a in ans:
        s = str(a).strip()
        for m in TURN.finditer(s):
            if m.start() > 0 and not re.fullmatch(PRESIDENT, m.group(1)):
                turns.append(s[m.start():m.start() + 200].strip())
                break
    return {"rates": flags.mean().to_dict(), "tags": tags, "headers": headers, "turns": turns}


def inject_artifacts(answers: list[str], seed: int = 21, inv: dict | None = None) -> list[str]:
    """Dev answers with test's artifacts added at test's rates: a leading tag, a footnote marker after a
    sentence, a header line before a paragraph, a trailing turn by another speaker."""
    inv = inv or test_inventory()
    r, rng = inv["rates"], np.random.default_rng(seed)
    out = []
    for a in answers:
        s = str(a).strip()
        if inv["headers"] and rng.random() < r["header"]:
            paras = s.split("\n")
            k = int(rng.integers(len(paras)))
            paras.insert(k, str(rng.choice(inv["headers"])))
            s = "\n".join(paras)
        if rng.random() < r["footnote"]:
            ends = [m.end() for m in re.finditer(r"[.!?](?=\s)", s)] or [len(s)]
            k = int(rng.choice(ends))
            s = s[:k] + f" [{int(rng.integers(1, 60))}.]" + s[k:]
        if inv["turns"] and rng.random() < r["other_turn"]:
            s = s + "\n" + str(rng.choice(inv["turns"]))
        if inv["tags"] and rng.random() < r["lead_tag"]:
            s = str(rng.choice(inv["tags"])) + " " + s
        out.append(s)
    return out


def check() -> None:
    from qevasion.loader import DATA_CACHE, load_qevasion

    d = load_qevasion(DATA_CACHE)
    frames = {"train": d.train["interview_answer"], "dev": d.dev["interview_answer"],
              "test": pd.read_csv(TEST_CSV)["interview_answer"]}
    print(f"{'split':6} {'n':>5}  " + "  ".join(f"{k:>10}" for k in artifact_flags("x")) + "   changed  median words raw/norm")
    for name, col in frames.items():
        ans = col.astype(str).tolist()
        f = pd.DataFrame([artifact_flags(a) for a in ans]).mean()
        norm = [normalize_answer(a) for a in ans]
        changed = np.mean([n != a.strip() for n, a in zip(norm, ans)])
        w = np.median([len(a.split()) for a in ans]), np.median([len(n.split()) for n in norm])
        print(f"{name:6} {len(ans):>5}  " + "  ".join(f"{v:>10.3f}" for v in f) + f"   {changed:7.3f}  {w[0]:.0f}/{w[1]:.0f}")
        left = pd.DataFrame([artifact_flags(n) for n in norm]).mean()
        print(f"{'  after':6} {'':>5}  " + "  ".join(f"{v:>10.3f}" for v in left))
    print("registered check: normalization must change <= 1% of train and dev answers")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=("check",))
    p.parse_args()
    check()


if __name__ == "__main__":
    main()
