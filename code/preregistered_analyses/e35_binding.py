#!/usr/bin/env python3
"""E35 (registered 2026-11-21, docs/experiment_log.md): binding with uncertainty -- is "the classifier reads the
question" more than accuracy?

    python -m preregistered_analyses.e35_binding        # CPU; writes docs/raw/E35_binding.txt, runs/E35/binding.json

Binding = share of sibling pairs (one answer, two sub-questions) whose gold label sets are disjoint and that a system
labels identically (lower = more question-bound; an answer-only system scores 1). Matched null: the same-label rate on
2,000 random pairs of items from different answers with disjoint gold sets, under the same system's labels. Binding
index = (null - sibling rate) / null (1 = fully bound; 0 = siblings labelled as alike as unrelated items). Intervals:
2,000-draw bootstrap over answers (the sibling pairs of a drawn answer enter together); system differences are paired.

Dev: the 58 E30 pairs; systems = the frozen instrument, M1, DeBERTa, the 27B zero-shot, the frozen probe at each of 10
layers, and two anchors (majority label; M1 on the no-question frame = answer-only).
Train (H35-1): sibling pairs with different gold labels in the 248 confirmation interviews; CF (E33, out-of-fold)
vs ZS (27B zero-shot over the nine labels, runs/M3/e35zs_taxonomy, frame "train").
"""

from __future__ import annotations

import json

import numpy as np

from audit.validity_battery import DISJ, GOLD, from_probs, instrument
from preregistered_analyses.e24_prechecks import sibling_pairs
from qevasion.labels import N_EVASION, encode_evasion
from qevasion.loader import DATA_CACHE, load_qevasion
from qevasion.paths import ROOT, RUNS
from qevasion.scoring import score_subtask2

RAW = ROOT / "docs" / "raw" / "E35_binding.txt"
OUT = RUNS / "E35" / "binding.json"
B = 2000
SP = load_qevasion(DATA_CACHE)
DEV = SP.dev.reset_index(drop=True)


def answer_id(df) -> np.ndarray:
    return df["interview_answer"].astype(str).str.split().str.join(" ").factorize()[0]


def null_pairs(gold: np.ndarray, aid: np.ndarray, rows: np.ndarray, n: int = 2000, seed: int = 35) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        i, j = rng.choice(rows, 2, replace=False)
        if aid[i] != aid[j] and not (gold[i] & gold[j]).any():
            out.append((i, j))
    return np.array(out)


def stats(labels: dict[str, np.ndarray], pairs: np.ndarray, nulls: np.ndarray, aid: np.ndarray) -> dict:
    """Point estimates and paired bootstrap draws over answers for every system."""
    pa = aid[pairs[:, 0]]
    by = {a: np.where(pa == a)[0] for a in np.unique(pa)}
    keys = np.array(sorted(by))
    rng = np.random.default_rng(0)
    draws = [np.concatenate([by[a] for a in rng.choice(keys, len(keys))]) for _ in range(B)]
    res = {}
    for name, y in labels.items():
        same = (y[pairs[:, 0]] == y[pairs[:, 1]]).astype(float)
        null = float(np.mean(y[nulls[:, 0]] == y[nulls[:, 1]]))
        bs = np.array([same[d].mean() for d in draws])
        idx = lambda r: (null - r) / null if null > 0 else float("nan")  # noqa: E731
        res[name] = {"binding": float(same.mean()), "ci": np.percentile(bs, [2.5, 97.5]).tolist(), "null": null,
                     "index": idx(same.mean()), "index_ci": sorted(np.percentile([idx(b) for b in bs], [2.5, 97.5]).tolist()),
                     "_bs": bs}
    return res


def fmt(r: dict) -> str:
    ratio = r["binding"] / r["null"] if r["null"] > 0 else float("nan")
    return (f"binding {r['binding']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]  null {r['null']:.3f}  "
            f"index {r['index']:.2f} [{r['index_ci'][0]:.2f}, {r['index_ci'][1]:.2f}]  ratio to null {ratio:.2f}")


def dev_systems() -> tuple[dict[str, np.ndarray], dict[str, float]]:
    from cascade.fuse import LOGPI

    labels, s2 = {}, {}
    inst = instrument()
    labels["instrument (M1 + 27B adjudicator)"] = inst["frozen instrument (M1 + 27B)"]["dev_own"]
    labels["M1 (Qwen3-8B LoRA, 10 seeds)"] = inst["M1 (Qwen3-8B LoRA, 10 seeds)"]["dev_own"]
    for name in ("deberta", "llm27b_zeroshot"):
        r = from_probs(name)
        if r:
            labels[r[0]] = r[1]["dev_own"]
    z = np.load(RUNS / "E25" / "probe_probs.npz")
    for li, r in enumerate(json.loads((RUNS / "E25" / "probes.json").read_text())):
        labels[f"frozen probe, layer {r['layer']}"] = (np.log(z["dev_own"][li] + 1e-12) - r["tau"] * LOGPI).argmax(1)
    tr = encode_evasion(SP.train["evasion_label"].tolist())
    labels["anchor: majority label"] = np.full(len(DEV), np.bincount(tr, minlength=N_EVASION).argmax())
    labels["anchor: answer-only (M1, no question)"] = inst["M1 (Qwen3-8B LoRA, 10 seeds)"]["dev_null"]
    for k, y in labels.items():
        s2[k] = score_subtask2(y, GOLD).macro_f1
    return labels, s2


def train_part(L: list[str], res: dict) -> None:
    from preregistered_analyses.e34_analysis import cf_labels, cf_tau, zs_labels

    train = SP.train.reset_index(drop=True)
    cfp = RUNS / "E34" / "cf" / "train_own.npz"
    zs = zs_labels("e35zs_taxonomy", "train", len(train))
    if not cfp.exists() or not (RUNS / "E33" / "cf_tau.json").exists() or zs is None:
        L += ["", "Train (H35-1): not run yet (needs runs/E34/cf/train_own.npz, runs/E33/cf_tau.json and the e35zs 27B run)"]
        return
    cf, _ = cf_labels(np.load(cfp)["probs"], cf_tau())
    y = encode_evasion(train["evasion_label"].tolist())
    gold = np.zeros((len(train), N_EVASION), bool)
    gold[np.arange(len(train)), y] = True
    confirm = ~train["url"].astype(str).isin(set(DEV["url"].astype(str))).to_numpy()
    rows = np.where(confirm)[0]
    pairs = np.array([(i, j) for i, j in sibling_pairs(train) if confirm[i] and confirm[j] and y[i] != y[j]])
    aid = answer_id(train)
    nulls = null_pairs(gold, aid, rows)
    st = stats({"CF (cross-fit M1, out of fold)": cf, "ZS (27B zero-shot)": zs}, pairs, nulls, aid)
    s2 = {"CF (cross-fit M1, out of fold)": score_subtask2(cf[rows], gold[rows]).macro_f1,
          "ZS (27B zero-shot)": score_subtask2(zs[rows], gold[rows]).macro_f1}
    a, b = st["CF (cross-fit M1, out of fold)"], st["ZS (27B zero-shot)"]
    diff = a["_bs"] - b["_bs"]
    lo, hi = np.percentile(diff, [2.5, 97.5])
    acc_ok = abs(s2["CF (cross-fit M1, out of fold)"] - s2["ZS (27B zero-shot)"]) <= 0.05
    held = hi < 0 and acc_ok
    L += ["", f"Train, confirmation interviews: {len(pairs)} sibling pairs with different gold labels, {len(rows)} items"]
    for k, r in st.items():
        L.append(f"   {k:<34} S2 {s2[k]:.3f}  {fmt(r)}")
    L.append(f"   H35-1 binding(CF) - binding(ZS) = {a['binding'] - b['binding']:+.3f} [{lo:+.3f}, {hi:+.3f}]; "
             f"|S2 difference| {abs(s2['CF (cross-fit M1, out of fold)'] - s2['ZS (27B zero-shot)']):.3f} (<= 0.05: {acc_ok}) "
             f"-> {'HELD' if held else ('FAILED' if acc_ok else 'DESCRIPTIVE (accuracy condition not met)')}")
    res["train"] = {k: {kk: vv for kk, vv in r.items() if kk != "_bs"} | {"s2": s2[k]} for k, r in st.items()}
    res["train"]["H35-1"] = {"diff": float(a["binding"] - b["binding"]), "ci": [float(lo), float(hi)],
                             "acc_ok": bool(acc_ok), "held": bool(held), "n_pairs": int(len(pairs))}


def main() -> None:
    labels, s2 = dev_systems()
    aid = answer_id(DEV)
    nulls = null_pairs(GOLD, aid, np.arange(len(DEV)))
    st = stats(labels, np.asarray(DISJ), nulls, aid)
    L = [f"E35 binding on dev: {len(DISJ)} sibling pairs with disjoint gold sets ({len(np.unique(aid[np.asarray(DISJ)[:, 0]]))} "
         f"answers); matched null = 2,000 random pairs from different answers with disjoint gold sets; "
         f"bootstrap over answers ({B} draws)", ""]
    for k, r in st.items():
        L.append(f"   {k:<40} S2 {s2[k]:.3f}  {fmt(r)}")
    m1, zs = st["M1 (Qwen3-8B LoRA, 10 seeds)"], st.get("llm27b_zeroshot (1 seeds)")
    res = {"dev": {k: {kk: vv for kk, vv in r.items() if kk != "_bs"} | {"s2": s2[k]} for k, r in st.items()}}
    if zs is not None:
        d = m1["_bs"] - zs["_bs"]
        lo, hi = np.percentile(d, [2.5, 97.5])
        L += ["", f"   M1 - 27B zero-shot (S2 {s2['M1 (Qwen3-8B LoRA, 10 seeds)']:.3f} vs {s2['llm27b_zeroshot (1 seeds)']:.3f}): "
                  f"binding difference {m1['binding'] - zs['binding']:+.3f} [{lo:+.3f}, {hi:+.3f}] (descriptive)"]
        res["dev"]["M1_minus_ZS"] = {"diff": float(m1["binding"] - zs["binding"]), "ci": [float(lo), float(hi)]}
    deb = st.get("deberta (3 seeds)")
    if deb is not None and zs is not None:
        d = deb["_bs"] - zs["_bs"]
        lo, hi = np.percentile(d, [2.5, 97.5])
        L.append(f"   DeBERTa - 27B zero-shot: {deb['binding'] - zs['binding']:+.3f} [{lo:+.3f}, {hi:+.3f}] (descriptive)")
    train_part(L, res)
    print("\n".join(L))
    RAW.write_text("\n".join(L) + "\n", encoding="utf8")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(f"\nwrote {RAW} and {OUT}")


if __name__ == "__main__":
    main()
