#!/usr/bin/env python3
"""E16: measured inference cost per item on the lab card (GPU 0 alone), written to runs/M3/cost.json.

    CUDA_VISIBLE_DEVICES=0 python -m cascade.cost [--run Q8_fullq_lora_12ep --seed 0] [--fp8]

* M1 seed: Qwen3-8B-Base + the seed's LoRA adapter, bf16, batch 16, length-sorted -- the code path of
  models.llm_classifier.predict -- timed over the 308 dev items after one warm-up batch. The probabilities
  are checked against the run's saved dev_probs.npy.
* M3 call: the mean batched GPU time per item of the local Qwen3.8-27B-FP8 runs (vLLM; `gpu_sec` in
  runs/M3/<tag>/responses.jsonl), over the dev items.
* --fp8: also time M1 with FP8 weights (transformers FineGrainedFP8Config), if this stack supports it.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from models.llm_classifier import Cfg, collate, encode_df
from qevasion.labels import N_EVASION
from qevasion.loader import DATA_CACHE, load_qevasion
from qevasion.paths import RUNS


def load_m1(run: str, seed: int, fp8: bool):
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cfg = Cfg(**{k: v for k, v in json.loads((RUNS / run / f"seed{seed}" / "config.json").read_text()).items()
                 if k in Cfg.__dataclass_fields__})
    tok = AutoTokenizer.from_pretrained(cfg.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kw = {}
    if fp8:
        from transformers import FineGrainedFP8Config

        kw["quantization_config"] = FineGrainedFP8Config()
    base = AutoModelForSequenceClassification.from_pretrained(cfg.model, num_labels=N_EVASION, dtype=torch.bfloat16,
                                                              device_map={"": "cuda"}, **kw)
    base.config.pad_token_id = tok.pad_token_id
    model = PeftModel.from_pretrained(base, str(RUNS / run / f"seed{seed}" / "adapter_best")).eval()
    return cfg, tok, model


@torch.no_grad()
def time_predict(model, seqs, pad, bs: int) -> tuple[np.ndarray, float]:
    order = np.argsort(-np.array([len(s) for s in seqs]))
    out = np.zeros((len(seqs), N_EVASION), dtype=np.float32)
    ids, att = collate([seqs[i] for i in order[:bs]], pad, "cuda")  # warm-up
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(input_ids=ids, attention_mask=att)
    torch.cuda.synchronize()
    t0 = time.time()
    for s in range(0, len(order), bs):
        b = order[s: s + bs]
        ids, att = collate([seqs[i] for i in b], pad, "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids, attention_mask=att).logits
        out[b] = torch.softmax(logits.float(), -1).cpu().numpy()
    torch.cuda.synchronize()
    return out, (time.time() - t0) / len(seqs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="Q8_fullq_lora_12ep")
    p.add_argument("--seed", type=int, default=3)  # seeds 0-2 (E13b, Jarvis) keep only their adapter config locally
    p.add_argument("--m3-tag", default="qwen38_27b_k0")
    p.add_argument("--fp8", action="store_true")
    a = p.parse_args()
    dev = load_qevasion(DATA_CACHE).dev
    res = {"gpu": torch.cuda.get_device_name(0), "run": a.run, "seed": a.seed, "n_items": len(dev)}
    cfg, tok, model = load_m1(a.run, a.seed, False)
    seqs = encode_df(tok, dev, cfg)
    probs, sec = time_predict(model, seqs, tok.pad_token_id, cfg.eval_batch_size)
    saved = np.load(RUNS / a.run / f"seed{a.seed}" / "dev_probs.npy")
    res |= {"m1_seed_gpu_s_per_item": sec, "m1_tokens_per_item": float(np.mean([len(s) for s in seqs])),
            "m1_max_abs_diff_vs_saved_probs": float(np.abs(probs - saved).max()),
            "m1_argmax_agreement_vs_saved": float((probs.argmax(1) == saved.argmax(1)).mean())}
    print(f"M1 seed (bf16): {sec:.4f} GPU-s/item; agreement with saved dev probs {res['m1_argmax_agreement_vs_saved']:.3f}")
    del model
    torch.cuda.empty_cache()
    if a.fp8:
        try:
            _, _, m8 = load_m1(a.run, a.seed, True)
            p8, s8 = time_predict(m8, seqs, tok.pad_token_id, cfg.eval_batch_size)
            res |= {"m1_seed_fp8_gpu_s_per_item": s8,
                    "m1_fp8_argmax_agreement_vs_bf16": float((p8.argmax(1) == probs.argmax(1)).mean())}
            print(f"M1 seed (FP8): {s8:.4f} GPU-s/item; agreement with bf16 {res['m1_fp8_argmax_agreement_vs_bf16']:.3f}")
        except Exception as e:  # noqa: BLE001 -- optional measurement
            res["m1_fp8_error"] = f"{type(e).__name__}: {str(e)[:300]}"
            print(f"M1 FP8 not measured: {res['m1_fp8_error']}")
    recs = [json.loads(line) for line in (RUNS / "M3" / a.m3_tag / "responses.jsonl").read_text().splitlines()]
    dev_recs = [r for r in recs if r["frame"] == "dev" and "gpu_sec" in r]
    res |= {"m3_tag": a.m3_tag, "m3_call_gpu_s": float(np.mean([r["gpu_sec"] for r in dev_recs])),
            "m3_prompt_tokens": float(np.mean([r["prompt_tokens"] for r in dev_recs])), "m3_output_tokens": 1}
    print(f"M3 call (vLLM, batched): {res['m3_call_gpu_s']:.4f} GPU-s; {res['m3_prompt_tokens']:.0f} prompt tokens")
    (RUNS / "M3" / "cost.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
