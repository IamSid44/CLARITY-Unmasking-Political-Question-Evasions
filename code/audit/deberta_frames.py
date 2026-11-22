#!/usr/bin/env python3
"""E30: the reported DeBERTa-v3-large classifier (E10/E11 fullq, 16 epochs) on the E25 frames (one GPU).

    python -m audit.deberta_frames --seeds 0-2            # writes runs/E30/deberta/<frame>/seed<k>.npy, report.json

Weights: runs/<run>/seed<k>/model.pt (downloaded from the project's Hub repo; seeds 0-4 are E10_fullq_16ep, 5-9
E11_fullq_16ep). Inputs exactly as in training (models.encoder: input "full", max_len 1024, a_budget 256, head
truncation of the answer). Control (E30 registration): on dev_own each seed must reproduce its saved dev_probs.npy
(argmax agreement >= 0.98; the max |dp| is recorded).
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.encoder import ClarityEncoder, PairDataset, build_inputs, collate, predict
from qevasion.paths import RUNS

FRAMES = RUNS / "E25" / "frames"
OUT = RUNS / "E30" / "deberta"
NAMES = ("dev_own", "dev_rand", "dev_null", "dev_qonly", "dev_pre25", "dev_pre50", "dev_pre75",
         "test_own", "test_rand", "test_null")


def run_of(seed: int) -> str:
    return "E10_fullq_16ep" if seed < 5 else "E11_fullq_16ep"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0-2")
    p.add_argument("--batch", type=int, default=16)
    a = p.parse_args()
    from transformers import AutoTokenizer

    lo, _, hi = a.seeds.partition("-")
    seeds = list(range(int(lo), int(hi or lo) + 1))
    cfgj = json.loads((RUNS / run_of(seeds[0]) / f"seed{seeds[0]}" / "config.json").read_text())
    cfg = SimpleNamespace(max_len=cfgj["max_len"], a_budget=cfgj["a_budget"], head_frac=cfgj["head_frac"])
    tok = AutoTokenizer.from_pretrained(cfgj["model"])
    data = {}
    for n in NAMES:
        df = pd.read_parquet(FRAMES / f"{n}.parquet")
        A, B = build_inputs(df, cfgj["input"])
        data[n] = PairDataset(tok, A, B, None, None, None, cfg)
    rep_p = OUT / "report.json"
    rep = json.loads(rep_p.read_text()) if rep_p.exists() else {}
    for k in seeds:
        src = RUNS / run_of(k) / f"seed{k}"
        if not (src / "model.pt").exists():
            print(f"seed {k}: no model.pt, skipped", flush=True)
            continue
        model = ClarityEncoder(cfgj["model"], cfgj["head"], cfgj["dropout"], cfgj["pooling"])
        sd = torch.load(src / "model.pt", map_location="cpu")
        missing, unexpected = model.load_state_dict({k_: v.float() for k_, v in sd.items()}, strict=False)
        assert not missing and not unexpected, (missing, unexpected)
        model = model.cuda().eval()
        for n in NAMES:
            dl = DataLoader(data[n], batch_size=a.batch, shuffle=False, collate_fn=lambda b: collate(b, tok.pad_token_id))
            with torch.no_grad():
                pr = predict(model, dl, "cuda", 3)
            (OUT / n).mkdir(parents=True, exist_ok=True)
            np.save(OUT / n / f"seed{k}.npy", pr)
            if n == "dev_own":
                saved = np.load(src / "dev_probs.npy")
                rep[f"seed{k}"] = {"max_abs_dp": float(np.abs(pr - saved).max()),
                                   "argmax_agree": float((pr.argmax(1) == saved.argmax(1)).mean())}
                print(f"seed {k} control: {rep[f'seed{k}']}", flush=True)
        del model
        torch.cuda.empty_cache()
        OUT.mkdir(parents=True, exist_ok=True)
        rep_p.write_text(json.dumps(rep, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
