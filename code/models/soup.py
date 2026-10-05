#!/usr/bin/env python3
"""E12d: model soups of already-trained models (Wortsman et al., ICML 2022)."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models.encoder import (SPLIT_SEED, Cfg, ClarityEncoder, PairDataset, build_inputs, collate,
                     inset_diagnostics, load_test, predict, stratified_val_index)
from qevasion.labels import EVASION_LABELS, encode_clarity, encode_evasion, leaf_to_official_clarity
from qevasion.loader import DATA_CACHE, dev_reference_mask, load_qevasion
from qevasion.scoring import macro_f1_single_label, score_subtask1, score_subtask2

from qevasion.paths import RUNS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--members", required=True, help="comma-separated configurations whose seeds are averaged")
    ap.add_argument("--prefix", default="E12d_soup")
    args = ap.parse_args()
    t0 = time.time()

    dirs = [d for c in args.members.split(",") for d in sorted((RUNS / c).glob("seed*"))
            if (d / "model.pt").exists() and (d / "metrics.json").exists()]
    cfgs = [json.loads((d / "config.json").read_text()) for d in dirs]
    keys = ("model", "input", "max_len", "a_budget", "head_frac", "head", "pooling", "task")
    ref = {k: cfgs[0].get(k) for k in keys}
    assert all({k: c.get(k) for k in keys} == ref for c in cfgs), "members differ in architecture or input"
    cfg = Cfg(name="soup", **{k: v for k, v in ref.items() if v is not None})
    val_score = {d: json.loads((d / "metrics.json").read_text())["val_macro_f1"] for d in dirs}
    print(f"[soup] {len(dirs)} members from {args.members}", flush=True)

    sp = load_qevasion(DATA_CACHE)
    y = encode_evasion(sp.train["evasion_label"].tolist())
    val_idx = stratified_val_index(y, 0.1, SPLIT_SEED)
    for d in dirs:
        assert np.array_equal(np.load(d / "val_index.npy"), val_idx), f"{d}: different held-out slice"

    tok = AutoTokenizer.from_pretrained(cfg.model)
    pad = tok.pad_token_id
    loader = lambda a, b: DataLoader(PairDataset(tok, a, b, None, None, None, cfg), batch_size=32,
                                     collate_fn=lambda x: collate(x, pad))
    a_tr, b_tr = build_inputs(sp.train, cfg.input)
    dl_val = loader([a_tr[i] for i in val_idx], [b_tr[i] for i in val_idx])
    dl_dev = loader(*build_inputs(sp.dev, cfg.input))
    dl_te = loader(*build_inputs(load_test(), cfg.input))

    model = ClarityEncoder(cfg.model, cfg.head, cfg.dropout, cfg.pooling).to("cuda")
    states = {d: torch.load(d / "model.pt", map_location="cpu", weights_only=True) for d in dirs}

    def average(members):
        out = {}
        for k, v in states[members[0]].items():
            if v.is_floating_point():
                out[k] = sum(states[m][k].float() for m in members) / len(members)
            else:
                out[k] = v
        return out

    def val_f1(members):
        model.load_state_dict(average(members))
        p = predict(model, dl_val, "cuda", 3)
        return macro_f1_single_label(p.argmax(1), y[val_idx], tuple(EVASION_LABELS)).macro_f1, p

    order = sorted(dirs, key=lambda d: -val_score[d])
    greedy, (best, _) = [order[0]], val_f1([order[0]])
    print(f"[soup] greedy start {order[0].parent.name}/{order[0].name} val={best:.4f}", flush=True)
    for d in order[1:]:
        s, _ = val_f1(greedy + [d])
        keep = s >= best
        print(f"[soup]   + {d.parent.name}/{d.name}: val={s:.4f} {'kept' if keep else 'rejected'}", flush=True)
        if keep:
            greedy.append(d)
            best = s

    dev_mask = dev_reference_mask(sp.dev)
    dev_clar = encode_clarity(sp.dev["clarity_label"].tolist())
    for recipe, members in (("uniform", dirs), ("greedy", greedy)):
        vf1, vp = val_f1(members)
        dp = predict(model, dl_dev, "cuda", 3)
        tp = predict(model, dl_te, "cuda", 3)
        out = RUNS / f"{args.prefix}_{recipe}" / "seed0"
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "dev_probs.npy", dp)
        np.save(out / "val_probs.npy", vp)
        np.save(out / "val_index.npy", val_idx)
        np.save(out / "test_probs.npy", tp)
        (out / "config.json").write_text(json.dumps(asdict(cfg) | {"name": out.parent.name}, indent=2))
        pred = dp.argmax(1)
        s2 = score_subtask2(pred, dev_mask).macro_f1
        s1 = score_subtask1(leaf_to_official_clarity(pred), dev_clar).macro_f1
        (out / "metrics.json").write_text(json.dumps({
            "recipe": recipe, "members": [f"{m.parent.name}/{m.name}" for m in members],
            "val_macro_f1": vf1, "selected_epoch": None,
            "dev_subtask2_macro_f1": s2, "dev_subtask1_macro_f1": s1,
            "dev_inset": inset_diagnostics(pred, dev_mask),
        }, indent=2))
        print(f"[soup] {recipe}: {len(members)} members  val={vf1:.4f} devS2={s2:.4f} devS1={s1:.4f}", flush=True)
    print(f"[soup] DONE {(time.time() - t0) / 60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
