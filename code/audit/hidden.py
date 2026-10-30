#!/usr/bin/env python3
"""E25d: does the FROZEN base model already encode the question-answer relation? Layer-wise linear probes on
Qwen3-8B-Base (no adapter), and the latency of exiting early.

    python -m audit.hidden dump     # GPU, ~10 min: last-token hidden states of every layer -> runs/E25/hidden/
    python -m audit.hidden probe    # CPU: probes per layer -> runs/E25/probes.json, runs/E25/probe_probs.npz

Inputs use M1's exact text template (models.llm_classifier.encode_row), so a probe sees what M1 sees.
Splits: train (3,448), dev_own, test_own (normalized test answers).
Probe protocol, fixed in advance: per layer, PCA to 256 dimensions and multinomial logistic regression, both fitted
on train minus the 345-row slice; C in {0.01, 0.1, 1} and the logit-adjustment tau chosen on the slice (macro-F1);
nothing is chosen on dev. Reported per layer: dev S2/S1, and the same-label rate on dev sibling pairs whose
annotator sets are disjoint (the E24 P2 measure; M1: 19%).
Latency: batch-1 forward time of the base model truncated to the first L layers, L in {9, 18, 27, 36}, against the
LoRA classifier (audit.infer's measurement).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from qevasion.loader import DATA_CACHE, load_qevasion
from qevasion.paths import RUNS

OUT = RUNS / "E25" / "hidden"
FRAMES = RUNS / "E25" / "frames"
SPLITS = ("train", "dev_own", "test_own")


def frame(name: str) -> pd.DataFrame:
    if name == "train":
        return load_qevasion(DATA_CACHE).train.reset_index(drop=True)
    return pd.read_parquet(FRAMES / f"{name}.parquet")


def dump(model_id: str = "Qwen/Qwen3-8B-Base") -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    from models.llm_classifier import Cfg, collate, encode_df

    cfg = Cfg(name="E25d", model=model_id)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(model_id, dtype=torch.bfloat16, device_map={"": "cuda"}).eval()
    OUT.mkdir(parents=True, exist_ok=True)
    lat = {}
    for name in SPLITS:
        if (OUT / f"{name}.npy").exists():
            continue
        seqs = encode_df(tok, frame(name), cfg)
        order = np.argsort([-len(s) for s in seqs])
        H = None
        t0 = time.time()
        with torch.no_grad():
            for s in range(0, len(order), 16):
                b = order[s: s + 16]
                ids, att = collate([seqs[i] for i in b], tok.pad_token_id, "cuda")
                hs = model(input_ids=ids, attention_mask=att, output_hidden_states=True).hidden_states
                last = att.sum(1) - 1  # right padding: the last real token, where M1's head reads
                x = torch.stack([h[torch.arange(len(b)), last] for h in hs], 1).float().cpu().numpy().astype(np.float16)
                if H is None:
                    H = np.zeros((len(seqs), x.shape[1], x.shape[2]), np.float16)
                H[b] = x
        np.save(OUT / f"{name}.npy", H)
        print(f"{name}: {H.shape} in {time.time() - t0:.0f} s", flush=True)
    # early-exit latency: the same model truncated to its first L layers, batch 1, dev items
    seqs = encode_df(tok, frame("dev_own"), cfg)[:64]
    layers = model.layers
    for L in (9, 18, 27, len(layers)):
        model.layers = layers[:L]
        ts = []
        with torch.no_grad():
            for sq in seqs:
                ids, att = collate([sq], tok.pad_token_id, "cuda")
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(input_ids=ids, attention_mask=att)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
        ts = np.array(ts[4:]) * 1000
        lat[L] = {"p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95))}
        print(f"first {L} layers: batch-1 p50 {lat[L]['p50_ms']:.1f} ms, p95 {lat[L]['p95_ms']:.1f} ms", flush=True)
    model.layers = layers
    (OUT / "latency.json").write_text(json.dumps(lat, indent=1))


def probe() -> None:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from cascade.candidates import fit_tau_slice
    from decision_rules.decide import logit_adjust
    from qevasion.labels import encode_clarity, encode_evasion, leaf_to_official_clarity
    from qevasion.loader import dev_reference_mask
    from qevasion.scoring import macro_f1_single_label, score_subtask1, score_subtask2
    from qevasion.labels import EVASION_LABELS, N_EVASION

    sp = load_qevasion(DATA_CACHE)
    y = encode_evasion(sp.train["evasion_label"].tolist())
    prior = np.bincount(y, minlength=N_EVASION) / len(y)
    from qevasion.paths import SPLITS as SPLIT_DIR

    val = np.array(json.loads((SPLIT_DIR / "train_internal_val_index.json").read_text()))
    tr = np.setdiff1d(np.arange(len(y)), val)
    gold = dev_reference_mask(sp.dev)
    clar = encode_clarity(sp.dev["clarity_label"].tolist())
    X = {s: np.load(OUT / f"{s}.npy", mmap_mode="r") for s in SPLITS}
    dv = sp.dev.reset_index(drop=True)
    pairs = np.array([(ix[i], ix[j]) for _, g in dv.groupby("interview_answer") for ix in [g.index.tolist()]
                      for i in range(len(ix)) for j in range(i + 1, len(ix))])  # E24 P2's pairs
    disjoint = np.array([not (gold[a] & gold[b]).any() for a, b in pairs])
    res, probs = [], {s: [] for s in ("dev_own", "test_own")}
    for L in range(X["train"].shape[1]):
        t0 = time.time()
        get = lambda s: np.asarray(X[s][:, L], dtype=np.float32)  # noqa: E731
        sc = StandardScaler().fit(get("train")[tr])
        pca = PCA(256, random_state=0).fit(sc.transform(get("train")[tr]))
        Z = {s: pca.transform(sc.transform(get(s))) for s in SPLITS}
        best = None
        for C in (0.01, 0.1, 1.0):
            clf = LogisticRegression(C=C, max_iter=2000).fit(Z["train"][tr], y[tr])
            pv = clf.predict_proba(Z["train"][val])
            f = macro_f1_single_label(pv.argmax(1), y[val], EVASION_LABELS).macro_f1
            if best is None or f > best[0]:
                best = (f, C, clf, pv)
        _, C, clf, pv = best
        tau = fit_tau_slice(pv, y[val], prior)
        P = {s: clf.predict_proba(Z[s]) for s in ("dev_own", "test_own")}
        dec = {s: logit_adjust(p, prior, tau).argmax(1) for s, p in P.items()}
        d = dec["dev_own"]
        same = float((d[pairs[disjoint, 0]] == d[pairs[disjoint, 1]]).mean())
        row = {"layer": L, "C": C, "tau": tau, "slice_f1": best[0],
               "dev_s2": score_subtask2(dec["dev_own"], gold).macro_f1,
               "dev_s1": score_subtask1(leaf_to_official_clarity(dec["dev_own"]), clar).macro_f1,
               "sib_same_label_disjoint": same, "seconds": round(time.time() - t0, 1)}
        res.append(row)
        for s in probs:
            probs[s].append(P[s])
        print(json.dumps(row), flush=True)
        (RUNS / "E25" / "probes.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(RUNS / "E25" / "probe_probs.npz", **{s: np.stack(v) for s, v in probs.items()})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=("dump", "probe"))
    a = p.parse_args()
    dump() if a.what == "dump" else probe()


if __name__ == "__main__":
    main()
