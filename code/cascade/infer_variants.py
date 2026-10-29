#!/usr/bin/env python3
"""E21: M1 seeds and distilled students on transformed inputs (inference only, one GPU).

    python -m cascade.infer_variants --runs Q8_fullq_lora_12ep,Q8_distill_a05 [--seeds 0-9]

For every seed folder runs/<run>/seed<k>/ (adapter_best/ plus the saved dev/test probabilities) it writes
runs/E21/<run>/seed<k>/<variant>_probs.npy for:
  dev_ctrl   -- dev as is: must reproduce the saved dev_probs.npy (registered control, max |dp| < 1e-3)
  dev_pert   -- dev answers with test's transcript artifacts injected at test's rates (seed 21)
  test_ctrl  -- test as is: must reproduce the saved test_probs.npy
  test_norm  -- test answers normalized by qevasion.normalize
The base model is loaded once; the adapters (LoRA + the trained `score` head) are swapped in turn.
Analysis of H21a is CPU work on these files; this script prints the control check and the flip rates.
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
from qevasion.loader import DATA_CACHE, load_qevasion
from qevasion.normalize import inject_artifacts, normalize_answer, test_inventory
from qevasion.paths import RUNS, TEST_CSV

OUT = RUNS / "E21"


def frames() -> dict[str, pd.DataFrame]:
    dev = load_qevasion(DATA_CACHE).dev.reset_index(drop=True)
    test = pd.read_csv(TEST_CSV)
    pert, norm = dev.copy(), test.copy()
    pert["interview_answer"] = inject_artifacts(dev["interview_answer"].astype(str).tolist(), seed=21,
                                                inv=test_inventory())
    norm["interview_answer"] = [normalize_answer(a) for a in test["interview_answer"].astype(str)]
    return {"dev_ctrl": dev, "dev_pert": pert, "test_ctrl": test, "test_norm": norm}


def seed_list(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="Q8_fullq_lora_12ep,Q8_distill_a05")
    p.add_argument("--seeds", default="0-9")
    p.add_argument("--model", default="Qwen/Qwen3-8B-Base")
    a = p.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cfg = Cfg(name="E21", model=a.model)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=N_EVASION, dtype=torch.bfloat16,
                                                              device_map={"": cfg.device})
    base.config.pad_token_id = tok.pad_token_id
    fr = frames()
    seqs = {v: encode_df(tok, df, cfg) for v, df in fr.items()}
    model, prev, report = None, None, []
    for run in a.runs.split(","):
        for k in seed_list(a.seeds):
            src = RUNS / run / f"seed{k}"
            if not (src / "adapter_best").exists():
                print(f"[skip] {src}: no adapter_best", flush=True)
                continue
            t0, name = time.time(), f"{run}_s{k}"
            if model is None:
                model = PeftModel.from_pretrained(base, str(src / "adapter_best"), adapter_name=name)
            else:
                model.load_adapter(str(src / "adapter_best"), adapter_name=name)
                model.set_adapter(name)
                model.delete_adapter(prev)
            prev = name
            dst = OUT / run / f"seed{k}"
            dst.mkdir(parents=True, exist_ok=True)
            probs = {v: predict(model, s, tok.pad_token_id, cfg) for v, s in seqs.items()}
            for v, pr in probs.items():
                np.save(dst / f"{v}_probs.npy", pr)
            ctrl = {sp: float(np.abs(probs[f"{sp}_ctrl"] - np.load(src / f"{sp}_probs.npy")).max())
                    for sp in ("dev", "test")}
            flips = {"dev_pert": float((probs["dev_pert"].argmax(1) != probs["dev_ctrl"].argmax(1)).mean()),
                     "test_norm": float((probs["test_norm"].argmax(1) != probs["test_ctrl"].argmax(1)).mean())}
            row = {"run": run, "seed": k, "ctrl_max_abs_dp": ctrl, "argmax_flip": flips,
                   "seconds": round(time.time() - t0, 1)}
            report.append(row)
            print(json.dumps(row), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "infer_report.json").write_text(json.dumps(report, indent=1))
    bad = [r for r in report if max(r["ctrl_max_abs_dp"].values()) >= 1e-3]
    print(f"control: {len(report) - len(bad)}/{len(report)} seed folders reproduce their saved probabilities"
          + (f"; FAILED: {[(r['run'], r['seed']) for r in bad]}" if bad else ""))


if __name__ == "__main__":
    main()
