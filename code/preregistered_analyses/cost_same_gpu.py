#!/usr/bin/env python3
"""Inference cost of every reported system on ONE GPU scale (CPU, existing records; for the final report).

    python -m preregistered_analyses.cost_same_gpu > ../docs/raw/cost_same_gpu.txt

Why this file exists. Earlier cost tables mixed measurements: the M1 seed (0.052 GPU-s) was timed on the lab
card alone (E16), while the 27B call was taken either from records that reused cached prefixes (0.103, E16) or
from the E15 night run on the same lab card (0.151, E19). The E22a lane timed everything on one JarvisLabs RTX PRO
6000 Server Edition in one session, on the same 308 dev items, batched: one M1 seed (runs/E22/cost_jarvis.json)
and the 27B call in each of the four option orders (runs/M3/qwen38_27b_k0_{jarvis,shift1,shift2,shift3}). Every
cost below uses only those numbers.

A second correction: the frozen system averages q over four option orders, so each deferred item costs FOUR 27B
calls. E23's tables reported calls = deferral rate, i.e. one call per deferred item; that is right only for the
single-order variants.

Cost per item = K x c_M1 + (deferral rate) x (orders) x c_M3, plus, for the ladder, the escalated M1 seeds.
Scores are copied from docs/raw/E23_systems.txt (nested-CV split 0 / mean of 10 splits) and are unchanged here.
"""

from __future__ import annotations

import json

import numpy as np

from qevasion.paths import RUNS

ORDERS = ("jarvis", "shift1", "shift2", "shift3")  # taxonomy order first; all four timed on the E22a VM


def m3_dev_cost(tag: str) -> tuple[float, float]:
    """Mean GPU-s per call and prompt tokens over the one batched pass of the 308 distinct dev items."""
    rows = [json.loads(line) for line in (RUNS / "M3" / f"qwen38_27b_k0_{tag}" / "responses.jsonl").open(encoding="utf8")]
    rows = [r for r in rows if r["frame"] == "dev" and r["split"] == "dev" and not r.get("copied")]
    assert len(rows) == 308, (tag, len(rows))
    return float(np.mean([r["gpu_sec"] for r in rows])), float(np.mean([r["prompt_tokens"] for r in rows]))


def rule(t: str) -> None:
    print("=" * 110 + f"\n{t}\n" + "=" * 110)


def main() -> None:
    vm = json.loads((RUNS / "E22" / "cost_jarvis.json").read_text())
    c1 = vm["m1_seed_gpu_s_per_item"]
    m3 = {o: m3_dev_cost(o) for o in ORDERS}
    c3 = m3["jarvis"][0]                              # one call, taxonomy order
    c3x4 = sum(m3[o][0] for o in ORDERS)              # the four option orders of the frozen system
    tok = m3["jarvis"][1]
    cached, _ = m3_dev_cost("jarvis_cached")

    rule("A. Unit costs, all on one GPU in one session (" + vm["gpu"] + "; dev, 308 items, batched)")
    print(f"   one M1 seed (Qwen3-8B + LoRA, bf16, batch 16):   {c1:.4f} GPU-s per item ({vm['m1_tokens_per_item']:.0f} tokens)")
    for o in ORDERS:
        print(f"   one 27B call, order {o:<8}:                  {m3[o][0]:.4f} GPU-s ({m3[o][1]:.0f} prompt tokens, 1 output token)")
    print(f"   four orders together:                            {c3x4:.4f} GPU-s per deferred item")
    print(f"   cache-aligned prompt (E22a, not adopted):        {cached:.4f} GPU-s per call ({cached / c3:.2f}x)")
    print(f"   ratio 27B call / M1 seed: {c3 / c1:.2f}  (lab card, E19 mix of two sessions: {0.151 / 0.052:.2f})")

    # name, M1 seeds per item, deferral rate, orders per deferred item, extra M1 seeds per item, S2 sp0, S2 10sp, S1
    systems = [
        ("M1, 10 seeds", 10, 0.0, 0, 0.0, 0.598, 0.601, 0.743),
        ("accuracy point, 4 orders (frozen instrument)", 10, 0.357, 4, 0.0, 0.650, 0.639, 0.788),
        ("accuracy point, 1 order", 10, 0.357, 1, 0.0, 0.639, 0.637, 0.787),
        ("distilled point, 4 orders", 1, 0.574, 4, 0.0, 0.608, 0.606, 0.785),
        ("distilled point, 1 order", 1, 0.574, 1, 0.0, 0.599, 0.606, 0.784),
        ("escalation ladder (E22c primary), 1 order", 1, 0.358, 1, 0.44 * 5, None, 0.600, 0.785),
    ]
    rule("B. Systems (dev). Scores from raw/E23_systems.txt; LLM calls and tokens are 27B calls per item")
    print(f"   {'system':<46} {'S2 sp0':>6} {'S2 10sp':>7} {'S1':>6} {'calls':>6} {'27B tok in':>10} {'GPU-s':>7} {'x M1':>6}")
    base = 10 * c1
    for name, K, d, o, extra, s2, s2_10, s1 in systems:
        calls = d * o
        g = (K + extra) * c1 + calls * (c3x4 / 4 if o == 4 else c3)
        s2s = f"{s2:.3f}" if s2 is not None else "  -  "
        print(f"   {name:<46} {s2s:>6} {s2_10:>7.3f} {s1:>6.3f} {calls:>6.2f} {calls * tok:>10.0f} {g:>7.3f} {g / base:>5.2f}x")
    print("   TeleAI (1st place; DeepSeek-V3 through an API): S2 0.617, S1 0.812, 1.94 calls, ~7,300 input and ~380 output"
          "\n   tokens per item. Its GPU time is not published and its model differs, so only calls and tokens compare.")
    print(f"   share of the frozen instrument's GPU time spent in M1's ten seeds: {base / (base + 0.357 * c3x4):.2f}"
          f" (single order: {base / (base + 0.357 * c3):.2f})")

    rule("C. The E16/E17 frontier re-costed on this scale (single order, as run); calls realised per subset")
    u = json.loads((RUNS / "M3" / "frontier_Q8_fullq_lora_12ep.json").read_text())["table"]
    s = json.loads((RUNS / "M3" / "frontier_Q8_distill_a05.json").read_text())["table"]
    for t in (u, s):
        for r in t:
            r["gpu"] = r["K"] * c1 + r["calls"] * c3
    print(f"   {'':<10} {'K':>3} {'delta':>5} {'calls':>6} {'S2':>6} {'GPU-s':>7}")
    for name, t in (("M1", u), ("student", s)):
        for r in t:
            if name == "student" or r["K"] in (1, 10):
                print(f"   {name:<10} {r['K']:>3} {r['delta']:>5.1f} {r['calls']:>6.3f} {r['s2']:>6.3f} {r['gpu']:>7.3f}")
    beaten = sum(any(x["gpu"] < o["gpu"] and x["s2"] >= o["s2"] for x in s) for o in u)
    print(f"   undistilled points dominated by a student point: {beaten} of {len(u)} (E17 on the cost.json scale: 25 of 30)")
    st5 = next(r for r in s if r["delta"] == 0.5)
    print(f"   student + 27B at delta 0.5: S2 {st5['s2']:.3f} at {st5['gpu']:.3f} GPU-s = {st5['gpu'] / base:.2f}x M1 x 10 (S2 0.598)")

    rule("D. Cross-check on the lab card (E19 numbers, two sessions): the ratios to M1 move by at most 0.02")
    l1, l3 = 0.052, 0.151
    for name, K, d, o in (("accuracy point, 1 order", 10, 0.357, 1), ("distilled point, 1 order", 1, 0.574, 1)):
        print(f"   {name:<30} lab {(K * l1 + d * o * l3) / (10 * l1):.2f}x   this GPU {(K * c1 + d * o * c3) / base:.2f}x")


if __name__ == "__main__":
    main()
