#!/usr/bin/env python3
"""M3: score M1's top-4 candidates with an LLM, one request and one output token per item (E15).

    # local model, vLLM venv (GPU):
    /scratch/shlok/Temp/.venv-vllm/bin/python -m cascade.adjudicate --backend vllm \
        --model Qwen/Qwen3.8-27B-FP8 --tag qwen38_27b_k10 --k 10 --splits screen
    # JarvisLabs Model API (key JARVIS_API_KEY in code/.env); always dry-run first:
    python -m cascade.adjudicate --backend api --model <id> --tag deepseek_k10 --k 10 --splits screen --dry-run

Splits (E15): screen = 300 rows (seed 15) of the 30% least confident cross-fitted train rows, slice excluded;
slice = the 345-row train slice (10-seed candidates); dev; test; train50 = the 50% least confident
cross-fitted train rows (M4's inputs). Candidates are M1's top 4 by rank of the calibrated, adjusted q.

Writes runs/M3/<tag>/responses.jsonl, one line per item, appended as results arrive; a rerun skips
items already done, so nothing is paid for twice. q(c|x) is the softmax over the candidates' letter
log-probabilities. API spend goes through cascade.api_budget (hard cap for all of E15).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cascade.prompts import LETTERS, PROMPT_VERSION, approx_tokens, build_messages, item_from_row, letter_map
from qevasion.paths import ROOT, RUNS

M1_INPUTS = RUNS / "M1" / "Q8_fullq_lora_12ep_fixed_a0.05" / "cascade_inputs.npz"
TOP_K = 4
SCREEN_N, SCREEN_SEED, SCREEN_DELTA = 300, 15, 0.3
SPLITS = ("screen", "slice", "dev", "test", "train50")


@dataclass
class Job:
    split: str  # the E15 split name
    frame: str  # which data frame the row lives in: train / dev / test
    i: int  # row index in that frame
    cand: list[int]

    @property
    def id(self) -> str:
        return f"{self.split}:{self.i}"


def topk(q: np.ndarray, k: int = TOP_K) -> np.ndarray:
    return np.argsort(-q, axis=1, kind="stable")[:, :k]


def select(split: str, z=None) -> list[Job]:
    """The rows of one E15 split, with their candidate sets."""
    z = z if z is not None else np.load(M1_INPUTS)
    if split in ("dev", "test"):
        return [Job(split, split, i, sorted(map(int, c))) for i, c in enumerate(topk(z[f"{split}_q"]))]
    if split == "slice":
        return [Job(split, "train", int(i), sorted(map(int, c))) for i, c in zip(z["val_idx"], topk(z["val_q"]))]
    q = z["train_q"]
    u = 1.0 - q.max(1)
    c = topk(q)
    if split == "train50":
        rows = np.where(u >= np.quantile(u, 0.5))[0]
    elif split == "screen":
        pool = np.setdiff1d(np.where(u >= np.quantile(u, 1 - SCREEN_DELTA))[0], z["val_idx"])
        rows = np.sort(np.random.default_rng(SCREEN_SEED).choice(pool, SCREEN_N, replace=False))
    else:
        raise SystemExit(f"unknown split {split}")
    return [Job(split, "train", int(i), sorted(map(int, c[i]))) for i in rows]


def build_prompts(jobs: list[Job], k: int) -> list[tuple[list[dict], list[int]]]:
    from cascade.exemplars import Retriever

    r = Retriever()
    tr = r.frames["train"]
    out = []
    for j in jobs:
        ex = r.examples(j.frame, j.i, j.cand, k)
        exs = [(item_from_row(tr.iloc[e]), int(r.y[e])) for e in ex]
        msgs, keep = build_messages(item_from_row(r.frames[j.frame].iloc[j.i]), j.cand, exs)
        out.append((msgs, [ex[n] for n in keep]))
    return out


def letters_to_q(top: list[tuple[str, float]], cand: list[int]) -> dict:
    """Candidate log-probs from the first output token's top log-probs; absent letters get min(top) - 1."""
    lm = letter_map(cand)
    best = {L: max((lp for t, lp in top if t.strip() == L), default=None) for L in lm}
    floor = min(lp for _, lp in top) - 1.0 if top else -30.0
    lps = [best[L] if best[L] is not None else floor for L in lm]
    m = max(lps)
    e = [math.exp(v - m) for v in lps]
    return {"lp": lps, "q": [v / sum(e) for v in e], "missing": sum(best[L] is None for L in lm),
            "letter_mass": sum(math.exp(best[L]) for L in lm if best[L] is not None)}


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()}


def record(job: Job, ex: list[int], msgs: list[dict], top: list[tuple[str, float]], extra: dict) -> dict:
    return {"id": job.id, "split": job.split, "frame": job.frame, "i": job.i, "cand": job.cand, "examples": ex,
            **letters_to_q(top, job.cand), "top": [[t, round(lp, 4)] for t, lp in top[:8]],
            "prompt_chars": sum(len(m["content"]) for m in msgs), **extra}


# ---------------------------------------------------------------- local backend (vLLM)

def run_vllm(jobs, prompts, a, out: Path) -> None:
    from vllm import LLM, SamplingParams

    llm = LLM(model=a.model, max_model_len=a.max_model_len, enable_prefix_caching=True,
              gpu_memory_utilization=a.gpu_util, seed=0, **json.loads(a.vllm_args))
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    kw = {"chat_template_kwargs": json.loads(a.chat_template_kwargs)}
    for s in range(0, len(jobs), a.chunk):
        t0 = time.time()
        part = list(range(s, min(s + a.chunk, len(jobs))))
        outs = llm.chat([prompts[n][0] for n in part], sp, use_tqdm=True, **kw)
        sec = (time.time() - t0) / len(part)
        with out.open("a") as f:
            for n, o in zip(part, outs):
                lp = o.outputs[0].logprobs[0]
                top = sorted(((v.decoded_token or "", v.logprob) for v in lp.values()), key=lambda x: -x[1])
                f.write(json.dumps(record(jobs[n], prompts[n][1], prompts[n][0], top,
                                          {"prompt_tokens": len(o.prompt_token_ids), "gpu_sec": round(sec, 3)})) + "\n")
        print(f"{time.strftime('%T')} {s + len(part)}/{len(jobs)} scored ({sec:.2f} s/item)", flush=True)


# ---------------------------------------------------------------- API backend (OpenAI-compatible)

def api_key() -> str:
    import os

    from dotenv import dotenv_values

    key = os.environ.get("JARVIS_API_KEY") or dotenv_values(ROOT / "code" / ".env").get("JARVIS_API_KEY")
    if not key:
        raise SystemExit("JARVIS_API_KEY is not set (code/.env or the environment)")
    return key


async def run_api(jobs, prompts, a, out: Path) -> None:
    from openai import AsyncOpenAI

    from cascade.api_budget import BASE_URL, BudgetExceeded, Ledger, usage_dict

    ledger = Ledger(a.tag, a.max_usd)
    ledger.check()
    client = AsyncOpenAI(base_url=a.base_url or BASE_URL, api_key=api_key(), max_retries=6, timeout=180)
    sem = asyncio.Semaphore(a.concurrency)
    extra = json.loads(a.extra_body)
    lock = asyncio.Lock()
    stop = asyncio.Event()

    async def one(n: int) -> None:
        async with sem:
            if stop.is_set():
                return
            try:
                ledger.check()
            except BudgetExceeded as e:
                print(f"STOP: {e}", flush=True)
                stop.set()
                return
            try:
                await request(n)
            except Exception as e:  # noqa: BLE001 -- one failed item must not stop the run; a rerun retries it
                print(f"FAILED {jobs[n].id}: {type(e).__name__}: {str(e)[:300]}", flush=True)

    async def request(n: int) -> None:
        t0 = time.time()
        if a.samples == 0:
            r = await client.chat.completions.create(model=a.model, messages=prompts[n][0], max_tokens=1,
                                                     temperature=0.0, logprobs=True, top_logprobs=20,
                                                     extra_body=extra)
            lc = r.choices[0].logprobs.content if r.choices[0].logprobs else None
            top = [(t.token, t.logprob) for t in lc[0].top_logprobs] if lc else []
        else:  # fallback when the API returns no log-probabilities: letter frequencies over samples
            r = await client.chat.completions.create(model=a.model, messages=prompts[n][0], max_tokens=1,
                                                     temperature=1.0, n=a.samples, extra_body=extra)
            cnt: dict[str, int] = {}
            for ch in r.choices:
                t = (ch.message.content or "").strip()[:1]
                cnt[t] = cnt.get(t, 0) + 1
            top = sorted(((t, math.log((c + 0.5) / (a.samples + 0.5 * len(LETTERS)))) for t, c in cnt.items()),
                         key=lambda x: -x[1])
        u = usage_dict(r.usage)
        usd = ledger.add(a.model, u)
        rec = record(jobs[n], prompts[n][1], prompts[n][0], top,
                     {"usage": u, "usd": round(usd, 7), "sec": round(time.time() - t0, 2),
                      "content": r.choices[0].message.content})
        async with lock:
            with out.open("a") as f:
                f.write(json.dumps(rec) + "\n")

    tasks = [asyncio.create_task(one(n)) for n in range(len(jobs))]
    for m, t in enumerate(asyncio.as_completed(tasks), 1):
        await t
        if m % 25 == 0 or m == len(tasks):
            print(f"{time.strftime('%T')} {m}/{len(tasks)}  run ${ledger.run_spent:.4f}  E15 total ${ledger.total():.4f}",
                  flush=True)


# ---------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=("vllm", "api"), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True, help="output folder runs/M3/<tag>")
    p.add_argument("--splits", default="screen", help=f"comma-separated, from {SPLITS}")
    p.add_argument("--k", type=int, default=10, help="number of retrieved examples")
    p.add_argument("--limit", type=int, default=0, help="first N items per split only (smoke tests)")
    p.add_argument("--dry-run", action="store_true", help="build prompts, print token and cost estimates, no calls")
    # vLLM
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--gpu-util", type=float, default=0.88)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--vllm-args", default="{}", help='extra LLM(...) kwargs as JSON')
    p.add_argument("--chat-template-kwargs", default='{"enable_thinking": false}')
    # API
    p.add_argument("--base-url", default="")
    p.add_argument("--extra-body", default='{"chat_template_kwargs": {"enable_thinking": false}}',
                   help="JSON passed as extra_body; the default switches thinking off (E15 probe)")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-usd", type=float, default=1.0, help="cap for THIS run (the E15 cap applies on top)")
    p.add_argument("--samples", type=int, default=0, help="if >0: no log-probs, estimate q from N samples")
    a = p.parse_args()

    outdir = RUNS / "M3" / a.tag
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "responses.jsonl"
    z = np.load(M1_INPUTS)
    jobs = []
    for s in a.splits.split(","):
        js = select(s, z)
        jobs += js[: a.limit] if a.limit else js
    done = done_ids(out)
    jobs = [j for j in jobs if j.id not in done]
    # a row with the same candidates already scored under another split (screen rows are inside train50)
    # is copied, not re-scored
    prev = {}
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            prev[(r["frame"], r["i"], tuple(r["cand"]))] = r
    reuse = [j for j in jobs if (j.frame, j.i, tuple(j.cand)) in prev]
    if reuse:
        with out.open("a") as f:
            for j in reuse:
                r = dict(prev[(j.frame, j.i, tuple(j.cand))], id=j.id, split=j.split, copied=True)
                r.pop("usd", None)
                f.write(json.dumps(r) + "\n")
        print(f"{len(reuse)} items copied from identical earlier requests (no cost)")
        jobs = [j for j in jobs if (j.frame, j.i, tuple(j.cand)) not in prev]
    print(f"{len(jobs)} items to score ({len(done)} already done) for {a.tag}", flush=True)
    if not jobs:
        return
    prompts = build_prompts(jobs, a.k)
    toks = np.array([sum(approx_tokens(m["content"]) for m in pm) for pm, _ in prompts])
    nex = np.array([len(ex) for _, ex in prompts])
    pre = approx_tokens(prompts[0][0][0]["content"])
    print(f"prompt tokens (approx.): mean {toks.mean():.0f}, median {np.median(toks):.0f}, max {toks.max()}; "
          f"preamble {pre}; examples kept: mean {nex.mean():.1f}, min {nex.min()}", flush=True)
    (outdir / "prompt_sample.txt").write_text("\n\n=====\n\n".join(m["content"] for m in prompts[0][0]))
    cfg = {"backend": a.backend, "model": a.model, "k": a.k, "prompt_version": PROMPT_VERSION, "top_k": TOP_K,
           "m1_inputs": str(M1_INPUTS.relative_to(ROOT)), "extra_body": json.loads(a.extra_body),
           "chat_template_kwargs": json.loads(a.chat_template_kwargs), "samples": a.samples,
           "started": time.strftime("%F %T")}
    if a.backend == "api":
        from cascade.api_budget import GLOBAL_CAP_USD, Ledger, price

        pin, pcached, pout = price(a.model)
        # the preamble is billed at the cached rate after the first request; n samples share one prompt
        est = ((toks - pre).sum() * pin + pre * len(toks) * pcached + len(toks) * max(a.samples, 1) * pout) / 1e6
        print(f"projected cost ${est:.3f} (E15 spent so far ${Ledger.total():.3f} of ${GLOBAL_CAP_USD:.2f}; "
              f"run cap ${a.max_usd:.2f})", flush=True)
        if est > a.max_usd and not a.dry_run:
            raise SystemExit("projected cost is above --max-usd; raise it deliberately if intended")
    if a.dry_run:
        print(f"dry run: one prompt written to {outdir / 'prompt_sample.txt'}")
        return
    cfg_path = outdir / "config.json"
    if cfg_path.exists():  # resuming: the scoring configuration must not change under the same tag
        old = json.loads(cfg_path.read_text())
        diff = [key for key in cfg if key != "started" and old.get(key) != cfg[key]]
        if diff:
            raise SystemExit(f"{cfg_path} differs in {diff}: use a new --tag")
    else:
        cfg_path.write_text(json.dumps(cfg, indent=2))
    if a.backend == "vllm":
        run_vllm(jobs, prompts, a, out)
    else:
        asyncio.run(run_api(jobs, prompts, a, out))


if __name__ == "__main__":
    main()
