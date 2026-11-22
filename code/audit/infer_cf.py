#!/usr/bin/env python3
"""E33: the url-grouped cross-fit instrument (CF) on audit frames (one GPU).

    CLARITY_E25_DIR=runs/E34 python -m audit.infer_cf --frames train_own,dev_llm,dev_llm_rand,llm

Each frame row has `split` (train/dev; default dev) and `src_i`. A train row is scored only by the fold model whose
held-out fold contains its interview (data/splits/train_5fold_url.json), so the scorer never saw that interview; a dev
row is scored by all five fold models and averaged. Writes <E25_DIR>/cf/<frame>.npz: probs (N, 9), fold (N; -1 for
dev rows). Registered control C33-3: on train_own, every fold reproduces its saved val_probs.npy (argmax agreement
>= 0.97); the numbers go to <E25_DIR>/cf/report.json.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from models.llm_classifier import Cfg, encode_df, predict
from qevasion.labels import N_EVASION
from qevasion.paths import E25_DIR, RUNS, SPLITS

RUN = RUNS / "E33_cf_url"
FRAMES = E25_DIR / "frames"
OUT = E25_DIR / "cf"


def fold_of() -> np.ndarray:
    folds = json.loads((SPLITS / "train_5fold_url.json").read_text())["folds"]
    f = np.full(sum(len(x) for x in folds), -1)
    for k, rows in enumerate(folds):
        f[np.asarray(rows)] = k
    assert (f >= 0).all()
    return f


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", required=True, help="comma-separated frame names in <E25_DIR>/frames")
    p.add_argument("--model", default="Qwen/Qwen3-8B-Base")
    p.add_argument("--batch", type=int, default=32)
    a = p.parse_args()
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    fo = fold_of()
    cfg = Cfg(name="E33", model=a.model, eval_batch_size=a.batch)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    names = [n.strip() for n in a.frames.split(",") if n.strip()]
    dfs = {n: pd.read_parquet(FRAMES / f"{n}.parquet") for n in names}
    for n, df in dfs.items():
        if "split" not in df:
            df["split"] = "dev"
    OUT.mkdir(parents=True, exist_ok=True)
    rep_path = OUT / "report.json"
    report = json.loads(rep_path.read_text()) if rep_path.exists() else {}
    probs = {n: np.full((len(df), N_EVASION), np.nan, np.float32) for n, df in dfs.items()}
    dev_sum = {n: np.zeros((len(df), N_EVASION), np.float64) for n, df in dfs.items()}
    for k in range(5):
        base = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=N_EVASION,
                                                                  dtype=torch.bfloat16, device_map={"": cfg.device})
        base.config.pad_token_id = tok.pad_token_id
        model = PeftModel.from_pretrained(base, str(RUN / f"fold{k}" / "adapter_best")).eval()
        for n, df in dfs.items():
            is_tr = (df["split"] == "train").to_numpy()
            own = is_tr & (fo[np.where(is_tr, df["src_i"].to_numpy(), 0)] == k)
            rows = np.where(own | ~is_tr)[0]
            if not len(rows):
                continue
            t0 = time.time()
            pr = predict(model, encode_df(tok, df.iloc[rows], cfg), tok.pad_token_id, cfg)
            sec = time.time() - t0
            tr_rows = rows[is_tr[rows]]
            probs[n][tr_rows] = pr[is_tr[rows]]
            dev_sum[n][rows[~is_tr[rows]]] += pr[~is_tr[rows]]
            row = {"items": int(len(rows)), "items_per_s": round(len(rows) / sec, 2)}
            if n == "train_own":
                val_idx = np.load(RUN / f"fold{k}" / "val_index.npy")
                saved = np.load(RUN / f"fold{k}" / "val_probs.npy")
                pos = {int(s): j for j, s in enumerate(df["src_i"].to_numpy())}
                mine = probs[n][[pos[int(v)] for v in val_idx]]
                row["ctrl_argmax_agree"] = float((mine.argmax(1) == saved.argmax(1)).mean())
                row["ctrl_max_abs_dp"] = float(np.abs(mine - saved).max())
            report[f"{n}/fold{k}"] = row
            print(json.dumps({"frame": n, "fold": k, **row}), flush=True)
        del model, base
        torch.cuda.empty_cache()
        rep_path.write_text(json.dumps(report, indent=1))
    for n, df in dfs.items():
        is_tr = (df["split"] == "train").to_numpy()
        probs[n][~is_tr] = (dev_sum[n][~is_tr] / 5).astype(np.float32)
        assert not np.isnan(probs[n]).any(), f"{n}: unscored rows"
        np.savez(OUT / f"{n}.npz", probs=probs[n], fold=np.where(is_tr, fo[np.where(is_tr, df["src_i"], 0)], -1))
        print(f"{n}: {len(df)} rows -> {OUT / (n + '.npz')}", flush=True)


if __name__ == "__main__":
    main()
