#!/usr/bin/env python3
"""E30: the counterfactual validity battery -- does a question-answer classifier read the question?

    python -m audit.validity_battery          # CPU, seconds; writes docs/raw/E30_validity_battery.txt, runs/E30/battery.json
    python -m audit.validity_battery --import-27b <M3 tag> <name>   # a 27B run over all nine labels -> runs/E30/<name>

Each system is applied to the E25 frames (audit/frames.py), which change one thing about every item:
  own    the item as it is                         rand   the sub-question and turn from a different interview
  null   no sub-question, no turn                  qonly  the question kept, the answer replaced by one constant sentence
  pre25/50/75  the first 25/50/75% of the answer's words
(test: own/rand/null on the normalised official test answers, label-free). Per frame: Clear-Reply share, label
persistence (share of items keeping their own-frame label) and the Jensen-Shannon distance between the frame's label
distribution and the own frame's. On the own frame: S2 (dev) and question binding = the share of the 58 dev sibling
pairs (same answer, disjoint annotator sets) given the same label (an answer-only classifier scores 1).

A classifier that models the question-answer relation must (registered for E25a): at least halve its Clear-Reply share
under rand, and keep at most half of its labels. Systems: the frozen instrument and its M1 (runs/E25/labels), the
reported DeBERTa-v3-large (runs/E30/deberta, logit adjustment tau fitted once on dev own and then frozen, as M1's),
and any system with per-frame probabilities in runs/E30/<name>/<frame>/seed*.npy.
To add a system: write its (n, 9) probabilities per frame there; nothing else changes.
"""

from __future__ import annotations

import json

import numpy as np

from preregistered_analyses.e19_lib import Grid
from preregistered_analyses.e24_prechecks import sibling_pairs
from qevasion.labels import CLARITY_LABELS, N_EVASION, leaf_to_official_clarity
from qevasion.loader import DATA_CACHE, dev_reference_mask, load_qevasion
from qevasion.paths import ROOT, RUNS
from qevasion.scoring import score_subtask2

RAW = ROOT / "docs" / "raw" / "E30_validity_battery.txt"
OUT = RUNS / "E30" / "battery.json"
CR = CLARITY_LABELS.index("Clear Reply")
DEV_FRAMES = ("own", "rand", "null", "qonly", "pre25", "pre50", "pre75")
TEST_FRAMES = ("own", "rand", "null")
DEV = load_qevasion(DATA_CACHE).dev.reset_index(drop=True)
GOLD = dev_reference_mask(DEV)
PAIRS = np.array(sibling_pairs(DEV))
DISJ = PAIRS[[not (GOLD[i] & GOLD[j]).any() for i, j in PAIRS]]


def js(p: np.ndarray, q: np.ndarray) -> float:
    m = (p + q) / 2
    kl = lambda a, b: np.sum(np.where(a > 0, a * np.log2((a + 1e-12) / (b + 1e-12)), 0))  # noqa: E731
    return float(np.sqrt((kl(p, m) + kl(q, m)) / 2))


def instrument() -> dict[str, dict[str, np.ndarray]]:
    out = {"M1 (Qwen3-8B LoRA, 10 seeds)": {}, "frozen instrument (M1 + 27B)": {}}
    for split, frames in (("dev", DEV_FRAMES), ("test", TEST_FRAMES)):
        for f in frames:
            p = RUNS / "E25" / "labels" / f"{split}_{f}.npz"
            if p.exists():
                z = np.load(p)
                out["M1 (Qwen3-8B LoRA, 10 seeds)"][f"{split}_{f}"] = z["m1_pred"]
                out["frozen instrument (M1 + 27B)"][f"{split}_{f}"] = z["pred"]
    return out


def from_probs(name: str) -> tuple[str, dict[str, np.ndarray], dict] | None:
    d = RUNS / "E30" / name
    if not (d / "dev_own").exists():
        return None
    seeds = sorted(p.name for p in (d / "dev_own").glob("seed*.npy"))
    mean = lambda f: np.stack([np.load(d / f / s) for s in seeds if (d / f / s).exists()]).mean(0)  # noqa: E731
    lp = np.log(mean("dev_own") + 1e-12)
    g = Grid(lp, np.full((len(lp), N_EVASION), -np.inf), np.zeros(len(lp), bool))
    _, j = g.fit_m1(GOLD, np.arange(len(lp)))
    tau = float(g.taus[j])
    from cascade.fuse import LOGPI
    labels = {}
    for split, frames in (("dev", DEV_FRAMES), ("test", TEST_FRAMES)):
        for f in frames:
            if (d / f"{split}_{f}").exists():
                labels[f"{split}_{f}"] = (np.log(mean(f"{split}_{f}") + 1e-12) - tau * LOGPI).argmax(1)
    return f"{name} ({len(seeds)} seeds)", labels, {"tau": tau, "seeds": len(seeds)}


def measure(labels: dict[str, np.ndarray]) -> dict:
    res = {}
    for split, frames in (("dev", DEV_FRAMES), ("test", TEST_FRAMES)):
        own = labels.get(f"{split}_own")
        if own is None:
            continue
        d_own = np.bincount(own, minlength=N_EVASION) / len(own)
        for f in frames:
            y = labels.get(f"{split}_{f}")
            if y is None:
                continue
            r = {"cr": float(np.mean(leaf_to_official_clarity(y) == CR)), "persist": float(np.mean(y == own)),
                 "js": js(np.bincount(y, minlength=N_EVASION) / len(y), d_own)}
            if f == "own" and split == "dev":
                r["s2"] = score_subtask2(y, GOLD).macro_f1
                r["binding_same_label"] = float(np.mean([y[i] == y[j] for i, j in DISJ]))
            res[f"{split}_{f}"] = r
    return res


def compute() -> tuple[list[str], dict]:
    systems = instrument()
    meta = {}
    for name in sorted(p.name for p in (RUNS / "E30").iterdir() if p.is_dir()) if (RUNS / "E30").exists() else []:
        r = from_probs(name)
        if r:
            systems[r[0]] = r[1]
            meta[r[0]] = r[2]
    res = {k: measure(v) for k, v in systems.items()}
    gold_cr = float(np.mean(np.array([str(x) for x in DEV["clarity_label"]]) == "Clear Reply"))
    L = [f"E30 counterfactual validity battery (dev n = {len(DEV)}, test n = 237 label-free; {len(DISJ)} disjoint "
         f"sibling pairs). Presidents' gold Clear-Reply share on dev: {gold_cr:.3f}", "",
         "Per frame: Clear-Reply share / label persistence vs own / JS distance of the label distribution from own", ""]
    cols = [f"dev_{f}" for f in DEV_FRAMES] + [f"test_{f}" for f in TEST_FRAMES]
    L.append(f"{'system':<34}" + "".join(f"{c:>20}" for c in cols))
    def cell(r: dict, c: str) -> str:
        return f"{r[c]['cr']:.3f}/{r[c]['persist']:.2f}/{r[c]['js']:.2f}" if c in r else "-"

    for k, r in res.items():
        L.append(f"{k:<34}" + "".join(f"{cell(r, c):>20}" for c in cols))
    L += ["", f"{'system':<34}{'dev S2 (own)':>14}{'binding':>10}   (binding: same label on disjoint sibling pairs; lower = more question-bound)"]
    for k, r in res.items():
        L.append(f"{k:<34}{r['dev_own']['s2']:>14.3f}{r['dev_own']['binding_same_label']:>10.3f}")
    for k, m in meta.items():
        L.append(f"   {k}: logit-adjustment tau {m['tau']:.2f} fitted once on dev own, then frozen for every frame")
    L.append("")
    verdict = {}
    for k, r in res.items():
        ok = r["dev_rand"]["cr"] <= r["dev_own"]["cr"] / 2 and r["dev_rand"]["persist"] <= 0.5
        verdict[k] = bool(ok)
        L.append(f"swap test ({k}): Clear-Reply {r['dev_own']['cr']:.3f} -> {r['dev_rand']['cr']:.3f}, persistence "
                 f"{r['dev_rand']['persist']:.3f} -> {'PASSED' if ok else 'NOT PASSED'}")
    res["_swap_test"] = verdict
    res["_meta"] = meta
    res["_gold_cr"] = gold_cr
    return L, res


def import_27b(tag: str, name: str) -> None:
    """Write a 27B zero-shot run over all nine labels (runs/M3/<tag>, cand = 0..8) as per-frame probabilities, so the
    battery treats it like any other system (one 'seed')."""
    rows = [json.loads(x) for x in (RUNS / "M3" / tag / "responses.jsonl").read_text().splitlines() if x.strip()]
    by = {}
    for r in rows:
        assert r["cand"] == list(range(N_EVASION)), "expects records over all nine labels"
        by.setdefault(r["frame"], {})[r["i"]] = np.asarray(r["q"], np.float32)
    for frame, d in by.items():
        n = max(d) + 1
        assert len(d) == n, f"{frame}: {n - len(d)} items missing"
        out = RUNS / "E30" / name / frame
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "seed0.npy", np.stack([d[i] for i in range(n)]))
        print(f"{name}/{frame}: {n} items")


def main() -> None:
    import sys
    if len(sys.argv) == 4 and sys.argv[1] == "--import-27b":
        import_27b(sys.argv[2], sys.argv[3])
        return
    L, res = compute()
    print("\n".join(L))
    RAW.write_text("\n".join(L) + "\n", encoding="utf8")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {RAW} and {OUT}")


if __name__ == "__main__":
    main()
