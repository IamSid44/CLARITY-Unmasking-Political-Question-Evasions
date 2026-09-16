#!/usr/bin/env python3
"""Extract frozen backbone features once, so head variants train in seconds.

Design rationale in reasoning.md D-02. The short version: C1's entire claim is
about the OUTPUT PARAMETERIZATION, so the backbone need not be fine-tuned to test
it. Caching features makes P4.1's control requirement -- "identical backbone,
identical data order, only the output head differs" -- structurally true rather
than merely asserted by a config hash: every head variant reads the same tensor.

Cost: one forward pass per backbone instead of ~25 training runs per backbone.

Respects CLAUDE.md invariant 5: takes --max-vram-gb, logs peak allocated memory,
and fails fast rather than OOM-ing a shared card.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from higrec.data.loader import build_text_pairs, load_qevasion


def check_vram(max_gb: float) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; feature extraction needs a GPU")
    free_b, total_b = torch.cuda.mem_get_info()
    free_gb = free_b / 1024**3
    if free_gb < max_gb:
        raise SystemExit(
            f"FAIL FAST: only {free_gb:.1f} GB free, budget requires {max_gb:.1f} GB. "
            "The card is shared -- not starting rather than OOM-ing another job."
        )
    print(f"[vram] {free_gb:.1f} GB free of {total_b / 1024**3:.1f} GB total")


@torch.no_grad()
def extract(model, tokenizer, texts: list[str], max_length: int, batch_size: int,
            device: str) -> np.ndarray:
    """Mean-pooled final hidden states over non-padding tokens.

    Mean pooling rather than [CLS]: with a FROZEN encoder the [CLS] vector is
    only meaningful if it was pretrained for a sentence-level objective, whereas
    mean pooling is a reliable representation for any masked-LM encoder.
    """
    out: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length,
                        return_tensors="pt").to(device)
        hidden = model(**enc).last_hidden_state              # (B, T, H)
        mask = enc["attention_mask"].unsqueeze(-1).float()   # (B, T, 1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)
        out.append(pooled.float().cpu().numpy())
        if (start // batch_size) % 20 == 0:
            print(f"  {start + len(batch)}/{len(texts)}", flush=True)
    return np.concatenate(out, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/deberta-v3-large")
    ap.add_argument("--name", default=None, help="short name for the output dir")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-vram-gb", type=float, default=12.0)
    ap.add_argument("--out-dir", default="runs/features")
    ap.add_argument("--cache-dir", default="runs/data_cache")
    args = ap.parse_args()

    name = args.name or args.model.split("/")[-1]
    out_dir = Path(args.out_dir) / f"{name}_len{args.max_length}"
    out_dir.mkdir(parents=True, exist_ok=True)

    check_vram(args.max_vram_gb)
    device = "cuda"

    from transformers import AutoModel, AutoTokenizer

    print(f"[load] {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()

    splits = load_qevasion(cache_dir=args.cache_dir)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()

    meta = {"model": args.model, "max_length": args.max_length,
            "pooling": "mean_over_attention_mask", "dtype": "bfloat16"}

    for split_name, frame in (("train", splits.train), ("dev", splits.dev)):
        texts = build_text_pairs(frame)
        print(f"[extract] {split_name}: {len(texts)} items")
        features = extract(model, tokenizer, texts, args.max_length,
                           args.batch_size, device)
        np.save(out_dir / f"{split_name}.npy", features)
        meta[f"{split_name}_shape"] = list(features.shape)
        print(f"  -> {features.shape}")

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    meta["peak_vram_gb"] = round(peak_gb, 2)
    meta["wall_clock_s"] = round(time.time() - started, 1)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"[done] peak VRAM {peak_gb:.2f} GB, {meta['wall_clock_s']}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
