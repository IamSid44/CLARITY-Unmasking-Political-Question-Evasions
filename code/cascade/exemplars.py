#!/usr/bin/env python3
"""M3's boundary examples: embed every row once, then retrieve train neighbours whose gold label is a candidate.

    python -m cascade.exemplars embed            # GPU, ~2 min: runs/M3/embeddings/{train,dev,test}.npy

Retrieval rules (E15 registration):
  * the pool is TRAIN rows whose gold label is in C(x);
  * a row is never its own example, and rows that share the item's answer text or its interview (URL) are
    excluded for every split -- test shares neither with train, so this keeps slice/train/dev like test;
  * examples are taken round-robin over the candidates (nearest of each candidate first), so every candidate
    is illustrated and the examples do not simply vote by majority;
  * they are returned least similar first, so the most similar example sits next to the item.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from qevasion.labels import encode_evasion
from qevasion.loader import DATA_CACHE, load_qevasion
from qevasion.paths import RUNS, TEST_CSV

EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMB_DIR = RUNS / "M3" / "embeddings"


def load_frames() -> dict[str, pd.DataFrame]:
    sp = load_qevasion(DATA_CACHE)
    frames = {"train": sp.train, "dev": sp.dev, "test": pd.read_csv(TEST_CSV)}
    # E25: controlled copies of dev/test (question swaps, answer prefixes) and the LLM answers (audit.frames)
    for p in sorted((RUNS / "E25" / "frames").glob("*.parquet")):
        frames[p.stem] = pd.read_parquet(p)
    return frames


def embed_text(df: pd.DataFrame) -> list[str]:
    return [f"Sub-question: {str(q).strip()}\nAnswer: {str(a).strip()}"
            for q, a in zip(df["question"], df["interview_answer"], strict=True)]


def embed(texts: list[str], model_id: str = EMBED_MODEL, batch: int = 16, max_len: int = 1024) -> np.ndarray:
    """L2-normalised last-token embeddings (the pooling Qwen3-Embedding is trained with)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16).cuda().eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i: i + batch], padding=True, truncation=True, max_length=max_len,
                      return_tensors="pt").to("cuda")
            h = model(**enc).last_hidden_state[:, -1]
            out.append(torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def norm_answer(a) -> str:
    return " ".join(str(a).split()).lower()


def group_keys(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(answer key, interview key) per row; a missing URL never matches anything."""
    ans = np.array([norm_answer(a) for a in df["interview_answer"]], dtype=object)
    url = np.array([str(u) if isinstance(u, str) and u.strip() else f"<none-{i}>"
                    for i, u in enumerate(df["url"])], dtype=object)
    return ans, url


class Retriever:
    def __init__(self, frames: dict[str, pd.DataFrame] | None = None):
        self.frames = frames or load_frames()
        tr = self.frames["train"]
        self.y = encode_evasion(tr["evasion_label"].tolist())
        # frames without embeddings (E25) can only be scored with k = 0 examples; examples() needs them otherwise
        self.emb = {s: np.load(EMB_DIR / f"{s}.npy") for s in self.frames if (EMB_DIR / f"{s}.npy").exists()}
        self.keys = {s: group_keys(df) for s, df in self.frames.items()}
        assert "train" not in self.emb or len(self.emb["train"]) == len(tr)

    def examples(self, split: str, i: int, cand: list[int], k: int) -> list[int]:
        """Train row indices to show for row i of `split`, least similar first."""
        if k <= 0:
            return []
        ans_tr, url_tr = self.keys["train"]
        ans_i, url_i = self.keys[split][0][i], self.keys[split][1][i]
        ok = (ans_tr != ans_i) & (url_tr != url_i)
        if split == "train":
            ok[i] = False
        sim = self.emb["train"] @ self.emb[split][i]
        order = np.argsort(-sim, kind="stable")
        per = {c: order[(ok & (self.y == c))[order]] for c in cand}
        picked: list[int] = []
        rnd = 0
        while len(picked) < k and any(len(v) > rnd for v in per.values()):
            row = sorted((per[c][rnd] for c in cand if len(per[c]) > rnd), key=lambda j: -sim[j])
            picked.extend(row[: k - len(picked)])
            rnd += 1
        return [int(j) for j in sorted(picked, key=lambda j: sim[j])]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=("embed",))
    p.add_argument("--model", default=EMBED_MODEL)
    a = p.parse_args()
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    for split, df in load_frames().items():
        e = embed(embed_text(df), a.model)
        np.save(EMB_DIR / f"{split}.npy", e)
        print(f"{split}: {e.shape}")
    (EMB_DIR / "meta.json").write_text(json.dumps({"model": a.model, "text": "Sub-question + Answer",
                                                   "pooling": "last token, L2-normalised"}, indent=2))


if __name__ == "__main__":
    main()
