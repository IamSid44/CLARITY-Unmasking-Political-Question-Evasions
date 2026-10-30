#!/usr/bin/env python3
"""E26: the frozen M1 (Q8_fullq_lora_12ep, 10 seeds) at reduced precision. Inference only; nothing is retrained.

    python -m audit.quant --precision bf16 --site rtxpro            # 16-bit reference, same GPU
    python -m audit.quant --precision fp8  --site rtxpro            # native FP8 (Ada / Hopper / Blackwell)
    python -m audit.quant --precision nf4  --site a4000             # 4-bit NF4 (bitsandbytes; any CUDA GPU)
    python -m audit.quant --precision bf16 --site a4000 --device-map auto --batch 8   # split over 2 x 16 GB
    python -m audit.quant --precision fp8 --smoke                   # 32 items, seed 0, layer check; no files

Items: the train slice (345 rows, where the instrument's calibration and routing threshold were fitted), dev
(dev_own) and normalized test (test_own) from runs/E25/frames: 890 items.
Precisions (the LoRA adapter is merged into the base for bf16 and fp8, so both run the same dense model):
  bf16 -- merged 16-bit model: the reference, scored on the SAME GPU as the quantized run (E25 showed GPU
          architecture alone moves probabilities by up to 0.2)
  fp8  -- merged model, every decoder Linear in W8A8 FP8 (e4m3): weights scaled per output row, activations per
          token, matmul by torch._scaled_mm (row-wise scales; tensor-wise if the GPU/torch lacks row-wise).
          Embeddings, norms and the classification head stay 16-bit.
  nf4  -- base loaded in 4-bit NF4 (bitsandbytes, bf16 compute), the adapter applied UNMERGED on top (the QLoRA
          layout; merging into 4-bit weights would quantize the LoRA delta too); the head stays 16-bit.
Writes runs/E26/<site>/<precision>/seed<k>/<frame>.npy and runs/E26/<site>/<precision>/report.json
(throughput per frame and seed, batch-1 latency of seed 0, peak memory, the FP8 mode, library versions).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from models.llm_classifier import Cfg, collate, encode_df, predict
from qevasion.labels import N_EVASION
from qevasion.loader import DATA_CACHE, load_qevasion
from qevasion.paths import RUNS, SPLITS

RUN = "Q8_fullq_lora_12ep"
FRAMES = ("slice", "dev_own", "test_own")
E4M3_MAX = 448.0


# ---------------------------------------------------------------- FP8 W8A8 linear

class FP8Linear(torch.nn.Module):
    """y = x W^T + b with W and x in float8_e4m3fn: per-output-row weight scales and per-token activation scales
    (mode "rowwise"), or one scale each (mode "tensorwise"); torch._scaled_mm accumulates in higher precision."""

    def __init__(self, lin: torch.nn.Linear, mode: str):
        super().__init__()
        W = lin.weight.data.float()
        if mode == "rowwise":
            s = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / E4M3_MAX
        else:
            s = W.abs().amax().clamp(min=1e-12).reshape(1, 1) / E4M3_MAX
        self.register_buffer("w_t", (W / s).to(torch.float8_e4m3fn).t())  # column-major, as _scaled_mm wants
        self.register_buffer("s_w", s.t().contiguous().float() if mode == "rowwise" else s.reshape(()).float())
        self.bias = None if lin.bias is None else torch.nn.Parameter(lin.bias.data, requires_grad=False)
        self.mode, self.out_features = mode, lin.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        m = x2.shape[0]
        pad = (-m) % 16
        if pad:
            x2 = F.pad(x2, (0, 0, 0, pad))
        xf = x2.float()
        if self.mode == "rowwise":
            s_x = xf.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / E4M3_MAX
        else:
            s_x = xf.abs().amax().clamp(min=1e-12).reshape(()) / E4M3_MAX
        xq = (xf / s_x).to(torch.float8_e4m3fn)
        y = torch._scaled_mm(xq, self.w_t, scale_a=s_x, scale_b=self.s_w, out_dtype=torch.bfloat16)[:m]
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shp[:-1], self.out_features)


def fp8_mode(device) -> str:
    """Row-wise scaling if this GPU/torch supports it, else tensor-wise; checked against a bf16 matmul."""
    lin = torch.nn.Linear(4096, 1024, bias=False, device=device, dtype=torch.bfloat16)
    x = torch.randn(37, 4096, device=device, dtype=torch.bfloat16)
    ref = lin(x).float()
    for mode in ("rowwise", "tensorwise"):
        try:
            y = FP8Linear(lin, mode)(x).float()
        except RuntimeError as e:
            print(f"[fp8] {mode} unavailable: {str(e)[:160]}", flush=True)
            continue
        rel = float((y - ref).norm() / ref.norm())
        print(f"[fp8] {mode}: relative error vs bf16 on a random layer {rel:.4f}", flush=True)
        if rel < 0.1:
            return mode
    raise SystemExit("no working FP8 matmul on this GPU")


def to_fp8(model, mode: str) -> int:
    n = 0
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, torch.nn.Linear) and ".layers." in f".{full}" and "score" not in full:
                setattr(mod, child_name, FP8Linear(child, mode))
                n += 1
    torch.cuda.empty_cache()
    return n


# ---------------------------------------------------------------- models

def load(precision: str, seed: int, model_id: str, device_map, fp8: str | None, pad_id: int):
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

    adapter = str(RUNS / RUN / f"seed{seed}" / "adapter_best")
    kw = {"num_labels": N_EVASION, "device_map": device_map}
    if precision == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16,
                                                       llm_int8_skip_modules=["score"])
        kw["dtype"] = torch.bfloat16
    else:
        kw["dtype"] = torch.bfloat16
    import os

    if device_map == "auto" and os.environ.get("E26_MAX_MEM_GIB"):  # leave room for activations on split models
        kw["max_memory"] = {i: f"{os.environ['E26_MAX_MEM_GIB']}GiB" for i in range(torch.cuda.device_count())}
    base = AutoModelForSequenceClassification.from_pretrained(model_id, **kw)
    base.config.pad_token_id = pad_id  # the head reads the last non-pad token (as models.llm_classifier sets it)
    model = PeftModel.from_pretrained(base, adapter)
    if precision in ("bf16", "fp8"):
        model = model.merge_and_unload()
    if precision == "fp8":
        print(f"[fp8] {to_fp8(model, fp8)} decoder Linear layers converted ({fp8})", flush=True)
    return model.eval()


def frames() -> dict[str, pd.DataFrame]:
    tr = load_qevasion(DATA_CACHE).train.reset_index(drop=True)
    val = np.array(json.loads((SPLITS / "train_internal_val_index.json").read_text()))
    out = {"slice": tr.iloc[val].reset_index(drop=True)}
    for f in ("dev_own", "test_own"):
        out[f] = pd.read_parquet(RUNS / "E25" / "frames" / f"{f}.parquet")
    return out


def latency(model, seqs, pad_id, n: int = 64) -> dict:
    ts = []
    with torch.no_grad():
        for s in seqs[:n]:
            ids, att = collate([s], pad_id, "cuda")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(input_ids=ids, attention_mask=att)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    ts = np.array(ts[8:]) * 1000
    return {"p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95)), "n": int(len(ts))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--precision", choices=("bf16", "fp8", "nf4"), required=True)
    p.add_argument("--site", default="rtxpro", help="output folder name for the GPU type")
    p.add_argument("--seeds", default="0-9")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device-map", default="single", choices=("single", "auto"))
    p.add_argument("--model", default="Qwen/Qwen3-8B-Base")
    p.add_argument("--smoke", action="store_true", help="32 items of dev, seed 0, no files written")
    a = p.parse_args()
    from transformers import AutoTokenizer

    a_, _, b_ = a.seeds.partition("-")
    seeds = [0] if a.smoke else list(range(int(a_), int(b_ or a_) + 1))
    cfg = Cfg(name="E26", model=a.model, eval_batch_size=a.batch)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    fr = frames()
    if a.smoke:
        fr = {"dev_own": fr["dev_own"].iloc[:32]}
    seqs = {n: encode_df(tok, df, cfg) for n, df in fr.items()}
    device_map = {"": 0} if a.device_map == "single" else "auto"
    fp8 = fp8_mode("cuda") if a.precision == "fp8" else None
    out = RUNS / "E26" / a.site / a.precision
    rep_path = out / "report.json"
    rep = json.loads(rep_path.read_text()) if rep_path.exists() and not a.smoke else {}
    import transformers

    rep.update({"gpu": torch.cuda.get_device_name(0), "n_gpus": torch.cuda.device_count(), "precision": a.precision,
                "fp8_mode": fp8, "batch": a.batch, "device_map": a.device_map, "torch": torch.__version__,
                "transformers": transformers.__version__})
    for k in seeds:
        if not a.smoke and all((out / f"seed{k}" / f"{n}.npy").exists() for n in seqs):
            print(f"[done] seed {k}", flush=True)
            continue
        for d in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(d)
        t0 = time.time()
        model = load(a.precision, k, a.model, device_map, fp8, tok.pad_token_id)
        load_s = time.time() - t0
        row = {"load_s": round(load_s, 1)}
        for n, s in seqs.items():
            t1 = time.time()
            pr = predict(model, s, tok.pad_token_id, cfg)
            sec = time.time() - t1
            row[n] = {"items": len(pr), "seconds": round(sec, 2), "items_per_s": round(len(pr) / sec, 2)}
            if a.smoke:
                saved = np.load(RUNS / RUN / "seed0" / "dev_probs.npy")[: len(pr)]
                print(f"[smoke] {a.precision}: max |dp| vs saved bf16 {np.abs(pr - saved).max():.3f}, argmax agreement "
                      f"{(pr.argmax(1) == saved.argmax(1)).mean():.3f}, {len(pr) / sec:.1f} items/s", flush=True)
            else:
                (out / f"seed{k}").mkdir(parents=True, exist_ok=True)
                np.save(out / f"seed{k}" / f"{n}.npy", pr)
        row["peak_mem_gb"] = round(sum(torch.cuda.max_memory_allocated(d) for d in range(torch.cuda.device_count()))
                                   / 1024**3, 2)
        if k == seeds[0] and not a.smoke:
            row["latency_batch1"] = latency(model, seqs["dev_own"], tok.pad_token_id)
        print(json.dumps({"seed": k, **row}), flush=True)
        del model
        torch.cuda.empty_cache()
        if not a.smoke:
            rep[f"seed{k}"] = row
            out.mkdir(parents=True, exist_ok=True)
            rep_path.write_text(json.dumps(rep, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
