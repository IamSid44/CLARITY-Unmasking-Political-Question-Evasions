#!/usr/bin/env python3
"""E22e: question-answer specificity -- how much more likely the answer is after ITS question.

    python -m cascade.specificity [--model Qwen/Qwen3-8B-Base]    # one GPU, ~10 min; -> runs/E22/specificity.npz

With a frozen base LM, s(a | q) is the mean log-probability of the first 256 answer tokens after
"Question: {q}\nAnswer:". For every train, dev and test item (and test with normalized answers):
  own    -- q = the item's own sub-question
  rand   -- q = 3 sub-questions drawn from train rows of OTHER interviews (seed 22)
  null   -- an empty question
PMI_c = own - mean(rand) and PMI_0 = own - null are computed in the analysis (registered H22e:
Clear Reply > Ambivalent > Clear Non-Reply on train).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

from qevasion.loader import DATA_CACHE, load_qevasion
from qevasion.normalize import normalize_answer
from qevasion.paths import RUNS, TEST_CSV

OUT = RUNS / "E22" / "specificity.npz"
N_RAND, SEED, MAX_ANSWER, BATCH = 3, 22, 256, 16


def items() -> dict[str, pd.DataFrame]:
    d = load_qevasion(DATA_CACHE)
    test = pd.read_csv(TEST_CSV)
    norm = test.copy()
    norm["interview_answer"] = [normalize_answer(a) for a in test["interview_answer"].astype(str)]
    return {"train": d.train.reset_index(drop=True), "dev": d.dev.reset_index(drop=True), "test": test,
            "test_norm": norm}


@torch.no_grad()
def answer_logprob(model, tok, pairs: list[tuple[str, str]]) -> np.ndarray:
    """Mean log p(answer tokens | 'Question: q\\nAnswer:') for each (question, answer) pair."""
    enc = []
    for q, a in pairs:
        pre = tok(f"Question: {q.strip()}\nAnswer:", add_special_tokens=False)["input_ids"]
        ans = tok(" " + a.strip(), add_special_tokens=False)["input_ids"][:MAX_ANSWER]
        enc.append((pre, ans))
    out = np.zeros(len(enc), dtype=np.float32)
    order = np.argsort([-(len(p) + len(a)) for p, a in enc])
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for s in range(0, len(order), BATCH):
        b = order[s:s + BATCH]
        seqs = [enc[i][0] + enc[i][1] for i in b]
        L = max(map(len, seqs))
        ids = torch.full((len(b), L), pad, dtype=torch.long)
        att = torch.zeros((len(b), L), dtype=torch.long)
        for r, sq in enumerate(seqs):  # right padding: positions stay 0..len-1
            ids[r, :len(sq)] = torch.tensor(sq)
            att[r, :len(sq)] = 1
        logits = model(input_ids=ids.cuda(), attention_mask=att.cuda()).logits.float()
        lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, ids[:, 1:].cuda().unsqueeze(-1)).squeeze(-1)
        for r, i in enumerate(b):
            n_pre, n_ans = len(enc[i][0]), len(enc[i][1])
            out[i] = lp[r, n_pre - 1:n_pre - 1 + n_ans].mean().item() if n_ans else np.nan
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-8B-Base")
    p.add_argument("--limit", type=int, default=0, help="first N items per frame (smoke test)")
    a = p.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": "cuda"}).eval()
    fr = items()
    tr = fr["train"]
    rng = np.random.default_rng(SEED)
    arrays = {}
    for name, df in fr.items():
        df = df.head(a.limit) if a.limit else df
        t0 = time.time()
        ans = df["interview_answer"].astype(str).tolist()
        own = answer_logprob(model, tok, list(zip(df["question"].astype(str), ans)))
        null = answer_logprob(model, tok, [("", x) for x in ans])
        rand = np.zeros((len(df), N_RAND), dtype=np.float32)
        draws = []
        for url in df["url"].astype(str):
            pool = np.where(tr["url"].astype(str).values != url)[0]
            draws.append(rng.choice(pool, N_RAND, replace=False))
        draws = np.array(draws)
        for j in range(N_RAND):
            rand[:, j] = answer_logprob(model, tok, list(zip(tr["question"].astype(str).values[draws[:, j]], ans)))
        arrays.update({f"{name}_own": own, f"{name}_null": null, f"{name}_rand": rand, f"{name}_rand_rows": draws})
        pmi = own - rand.mean(1)
        print(f"{name}: {len(df)} items in {time.time() - t0:.0f} s; PMI_c mean {np.nanmean(pmi):+.3f}, "
              f"PMI_0 mean {np.nanmean(own - null):+.3f}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **arrays)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
