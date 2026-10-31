#!/usr/bin/env python3
"""M3: score M1's top-4 candidates with an LLM, one request and one output token per item (E15).

    # local model, vLLM venv (GPU):
    $VPY -m cascade.adjudicate --backend vllm \
        --model Qwen/Qwen3.8-27B-FP8 --tag qwen38_27b_k10 --k 10 --splits screen
    # JarvisLabs Model API (key JARVIS_API_KEY in code/.env); always dry-run first:
    python -m cascade.adjudicate --backend api --model <id> --tag deepseek_k10 --k 10 --splits screen --dry-run

Splits (E15): screen = 300 rows (seed 15) of the 30% least confident cross-fitted train rows, slice excluded;
slice = the 345-row train slice (10-seed candidates); dev; test; train50 = the 50% least confident
cross-fitted train rows (M4's inputs). Candidates are M1's top 4 by rank of the calibrated, adjusted q.

E19-E22 options (vLLM): --order shows the candidates rotated or reversed (q is still stored in sorted-candidate
order); --cache-pad aligns the shared preamble to vLLM's cache block and warms the cache; --think-budget N lets
the model think first and reads the letter after </think>.

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

from cascade.prompts import (LETTERS, ORDERS, PROMPT_VERSION, approx_tokens, build_messages, item_from_row,
                             letter_map)
from qevasion.paths import ROOT, RUNS

M1_INPUTS = RUNS / "M1" / "Q8_fullq_lora_12ep_fixed_a0.05" / "cascade_inputs.npz"
TOP_K = 4
SCREEN_N, SCREEN_SEED, SCREEN_DELTA = 300, 15, 0.3
SPLITS = ("screen", "slice", "dev", "test", "train50", "trainall")


@dataclass
class Job:
    split: str  # the E15 split name
    frame: str  # which data frame the row lives in: train / dev / test
    i: int  # row index in that frame
    cand: list[int]

    @property
    def id(self) -> str:
        # the registered splits have one candidate set per row; jobs-file items (E16/E17 frontiers) can have
        # several, so their id includes the candidates (fixed 2026-10-14: without it, a later jobs file's new
        # candidate sets for an item were skipped as "done")
        if self.split in SPLITS:
            return f"{self.split}:{self.i}"
        return f"{self.split}:{self.i}:{'-'.join(map(str, self.cand))}"


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
    elif split == "trainall":  # E17: every train row, for M4's soft targets
        rows = np.arange(len(q))
    elif split == "screen":
        pool = np.setdiff1d(np.where(u >= np.quantile(u, 1 - SCREEN_DELTA))[0], z["val_idx"])
        rows = np.sort(np.random.default_rng(SCREEN_SEED).choice(pool, SCREEN_N, replace=False))
    else:
        raise SystemExit(f"unknown split {split}")
    return [Job(split, "train", int(i), sorted(map(int, c[i]))) for i in rows]


def load_jobs(path: str) -> list[Job]:
    """Jobs written by another module (E16 frontier): one JSON per line with split, frame, i, cand."""
    return [Job(r["split"], r["frame"], int(r["i"]), sorted(map(int, r["cand"])))
            for r in map(json.loads, Path(path).read_text().splitlines()) if r]


def build_prompts(jobs: list[Job], k: int, variant: str = "full", order: str = "taxonomy",
                  test_text: str = "raw") -> list[tuple[list[dict], list[int]]]:
    from cascade.exemplars import Retriever, load_frames

    frames = load_frames()
    if test_text == "normalized":  # E21b: the official test answers without their transcript artifacts
        from qevasion.normalize import normalize_answer

        frames["test"] = frames["test"].copy()
        frames["test"]["interview_answer"] = [normalize_answer(a) for a in frames["test"]["interview_answer"]]
    r = Retriever(frames)
    tr = r.frames["train"]
    out = []
    for j in jobs:
        ex = r.examples(j.frame, j.i, j.cand, k)
        exs = [(item_from_row(tr.iloc[e]), int(r.y[e])) for e in ex]
        msgs, keep = build_messages(item_from_row(r.frames[j.frame].iloc[j.i]), j.cand, exs, variant, order)
        out.append((msgs, [ex[n] for n in keep]))
    return out


def letters_to_q(top: list[tuple[str, float]], cand: list[int], order: str = "taxonomy") -> dict:
    """Candidate log-probs from the first output token's top log-probs; absent letters get min(top) - 1.
    `lp` and `q` are in sorted-candidate order, whatever order the options were shown in."""
    lm = letter_map(cand, order)
    best = {L: max((lp for t, lp in top if t.strip() == L), default=None) for L in lm}
    floor = min(lp for _, lp in top) - 1.0 if top else -30.0
    by_label = {c: (best[L] if best[L] is not None else floor) for L, c in lm.items()}
    lps = [by_label[c] for c in sorted(cand)]
    m = max(lps)
    e = [math.exp(v - m) for v in lps]
    return {"lp": lps, "q": [v / sum(e) for v in e], "missing": sum(best[L] is None for L in lm),
            "letter_mass": sum(math.exp(best[L]) for L in lm if best[L] is not None)}


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()}


def record(job: Job, ex: list[int], msgs: list[dict], top: list[tuple[str, float]], extra: dict,
           order: str = "taxonomy") -> dict:
    return {"id": job.id, "split": job.split, "frame": job.frame, "i": job.i, "cand": job.cand, "examples": ex,
            **letters_to_q(top, job.cand, order), "order": order, "top": [[t, round(lp, 4)] for t, lp in top[:8]],
            "prompt_chars": sum(len(m["content"]) for m in msgs), **extra}


# ---------------------------------------------------------------- local backend (vLLM)

def cache_block(llm) -> int:
    """The token granularity at which vLLM can reuse a cached prefix for this model (E22a). For hybrid
    linear-attention models such as Qwen3.8 a prefix is reusable only up to a block boundary."""
    cc = llm.llm_engine.vllm_config.cache_config
    sizes = {k: getattr(cc, k, None) for k in ("block_size", "mamba_block_size")}
    print(f"[cache] {sizes}, prefix caching {getattr(cc, 'enable_prefix_caching', '?')}", flush=True)
    return int(sizes["mamba_block_size"] or sizes["block_size"])


def pad_preamble(llm, prompts, kw: dict) -> int:
    """E22a: pad the system message with ' .' units until the prefix shared by every prompt ends on a cache
    block boundary. Returns the number of units added."""
    tok, block = llm.get_tokenizer(), cache_block(llm)
    a, b = prompts[0][0], prompts[-1][0]
    base = a[0]["content"]

    def shared(sys: str) -> int:
        ids = [tok.apply_chat_template([dict(m[0], content=sys), m[1]], tokenize=True, add_generation_prompt=True,
                                       **kw["chat_template_kwargs"]) for m in (a, b)]
        ids = [list(x["input_ids"]) if hasattr(x, "keys") else list(x) for x in ids]  # transformers 5: a BatchEncoding
        return next((n for n, (x, y) in enumerate(zip(*ids)) if x != y), min(map(len, ids)))

    # " ." may add two tokens per unit, so a one-token tail is also tried, or an odd prefix could never align
    for units in range(0, 2 * block):
        for tail in ("", " ,"):
            n = shared(base + " ." * units + tail)
            if n % block == 0:
                break
        else:
            continue
        break
    else:
        raise SystemExit(f"could not align the shared prefix to the cache block ({block})")
    for msgs, _ in prompts:
        msgs[0]["content"] = base + " ." * units + tail
    print(f"[cache] preamble padded by {units} units: shared prefix {n} tokens = {n // block} blocks of {block}",
          flush=True)
    return units


def log_cache_metrics(llm) -> None:
    try:
        for m in llm.get_metrics():
            if "prefix_cache" in m.name:
                print(f"[cache] {m.name} = {getattr(m, 'value', getattr(m, 'values', '?'))}", flush=True)
    except Exception as e:  # noqa: BLE001 -- metrics are diagnostics only
        print(f"[cache] no metrics: {type(e).__name__}", flush=True)


def run_vllm(jobs, prompts, a, out: Path) -> None:
    from vllm import LLM, SamplingParams

    llm = LLM(model=a.model, max_model_len=a.max_model_len, enable_prefix_caching=True,
              gpu_memory_utilization=a.gpu_util, seed=0, **json.loads(a.vllm_args))
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    kw = {"chat_template_kwargs": json.loads(a.chat_template_kwargs)}
    extra = {}
    if a.cache_pad:
        extra["cache_pad_units"] = pad_preamble(llm, prompts, kw)
        llm.chat([prompts[0][0]], sp, use_tqdm=False, **kw)  # warm-up: the shared prefix enters the cache
    if a.think_budget:
        # E19 arm A / E20 R2: think first (Qwen's recommended sampling for thinking mode), then read the answer
        # letter's distribution at the first token after </think>
        sp_think = SamplingParams(max_tokens=a.think_budget, temperature=0.6, top_p=0.95, top_k=20, seed=0,
                                  stop=["</think>"])
        kw_think = {"chat_template_kwargs": {**kw["chat_template_kwargs"], "enable_thinking": True}}
        tok = llm.get_tokenizer()
    for s in range(0, len(jobs), a.chunk):
        t0 = time.time()
        part = list(range(s, min(s + a.chunk, len(jobs))))
        msgs = [prompts[n][0] for n in part]
        if a.think_budget:
            thoughts = llm.chat(msgs, sp_think, use_tqdm=True, **kw_think)
            texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                             **kw_think["chat_template_kwargs"]) for m in msgs]
            outs = llm.generate([t + th.outputs[0].text + "\n</think>\n\n" for t, th in zip(texts, thoughts)], sp,
                                use_tqdm=False)
        else:
            outs = llm.chat(msgs, sp, use_tqdm=True, **kw)
        sec = (time.time() - t0) / len(part)
        with out.open("a") as f:
            for j, (n, o) in enumerate(zip(part, outs)):
                lp = o.outputs[0].logprobs[0]
                top = sorted(((v.decoded_token or "", v.logprob) for v in lp.values()), key=lambda x: -x[1])
                info = {"prompt_tokens": len(o.prompt_token_ids), "gpu_sec": round(sec, 3), **extra}
                if a.think_budget:
                    th = thoughts[j].outputs[0]
                    info.update(think_tokens=len(th.token_ids), think_finished=th.finish_reason == "stop",
                                thought=th.text[-2000:])
                f.write(json.dumps(record(jobs[n], prompts[n][1], prompts[n][0], top, info, a.order)) + "\n")
        print(f"{time.strftime('%T')} {s + len(part)}/{len(jobs)} scored ({sec:.2f} s/item)", flush=True)
    if a.cache_pad:
        log_cache_metrics(llm)


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
                      "content": r.choices[0].message.content}, a.order)
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
    p.add_argument("--prompt-variant", default="full", choices=("full", "no_guide", "no_defs", "bare"))
    p.add_argument("--jobs-file", default="", help="extra jobs (jsonl: split, frame, i, cand), e.g. E16's frontier")
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
    p.add_argument("--order", default="taxonomy", choices=ORDERS, help="order the candidates are shown in (E19 arm B)")
    p.add_argument("--cache-pad", action="store_true", help="vLLM: align the shared preamble to the cache block (E22a)")
    p.add_argument("--test-text", default="raw", choices=("raw", "normalized"), help="official test answers (E21b)")
    p.add_argument("--think-budget", type=int, default=0, help="vLLM: think up to N tokens before the letter (E19 A, E20 R2)")
    a = p.parse_args()

    outdir = RUNS / "M3" / a.tag
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "responses.jsonl"
    z = np.load(M1_INPUTS)
    jobs = []
    for s in [s for s in a.splits.split(",") if s]:
        js = select(s, z)
        jobs += js[: a.limit] if a.limit else js
    if a.jobs_file:
        jobs += load_jobs(a.jobs_file)
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
    prompts = build_prompts(jobs, a.k, a.prompt_variant, a.order, a.test_text)
    toks = np.array([sum(approx_tokens(m["content"]) for m in pm) for pm, _ in prompts])
    nex = np.array([len(ex) for _, ex in prompts])
    pre = approx_tokens(prompts[0][0][0]["content"])
    print(f"prompt tokens (approx.): mean {toks.mean():.0f}, median {np.median(toks):.0f}, max {toks.max()}; "
          f"preamble {pre}; examples kept: mean {nex.mean():.1f}, min {nex.min()}", flush=True)
    (outdir / "prompt_sample.txt").write_text("\n\n=====\n\n".join(m["content"] for m in prompts[0][0]))
    cfg = {"backend": a.backend, "model": a.model, "k": a.k, "prompt_version": PROMPT_VERSION,
           "prompt_variant": a.prompt_variant, "top_k": TOP_K, "order": a.order,
           "cache_pad": a.cache_pad, "think_budget": a.think_budget, "test_text": a.test_text,
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
        defaults = {"prompt_variant": "full", "order": "taxonomy", "cache_pad": False, "think_budget": 0, "test_text": "raw"}  # added after E15
        diff = [key for key in cfg if key != "started" and old.get(key, defaults.get(key)) != cfg[key]]
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
