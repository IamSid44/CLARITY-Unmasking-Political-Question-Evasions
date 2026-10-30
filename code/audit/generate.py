#!/usr/bin/env python3
"""E25b: put the interview questions US presidents faced (dev, 308 sub-questions in 191 turns) to open LLMs.

    $VPY -m audit.generate --model llama31_8b=/path/or/hf-id [--model ...]     # vLLM venv, one GPU

Four conditions; each differs from the persona condition P in ONE factor:
  N  neutral analyst  -- the same interview turn, answered by someone who is NOT the President    (factor: stake)
  P  persona          -- answered as the President, in the interview, on the date it was asked   (reference)
  F  follow-up        -- P's conversation, then the journalist re-asks the sub-question            (factor: pursuit)
  B  briefed          -- P, with the President's own answer given as private notes                 (factor: information)
Every prompt is one user message (no system role, so every chat template accepts it), asks for spoken prose
without lists in at most 250 words, and is decoded greedily (temperature 0, <= 400 new tokens). N, P and B depend
only on the turn, so they are generated once per turn and shared by its sub-questions, as the President's one
answer is; F names the sub-question, so it is generated per sub-question. Qwen3 runs with thinking off.

Writes runs/E25/answers.jsonl: model, condition, src_i (dev row), answer, finish reason, token counts. A rerun
skips (model, condition, src_i) already written.
"""

from __future__ import annotations

import argparse
import json
import time

from qevasion.loader import DATA_CACHE, load_qevasion
from qevasion.paths import RUNS

OUT = RUNS / "E25" / "answers.jsonl"
STYLE = "Reply in spoken prose (no lists or headings), in at most 250 words"
NOTES_WORDS = 600


def turn(row) -> str:
    return " ".join(str(row["interview_question"]).split())


def prompts(row) -> dict[str, str]:
    who, when = row["president"], row["date"]
    return {
        "N": (f"The following was put to the President of the United States in a press interview on {when}. You are "
              f"not the President: answer it yourself, as a knowledgeable and neutral analyst. {STYLE}.\n\n"
              f"Journalist: {turn(row)}"),
        "P": (f"You are {who}, President of the United States. It is {when}, and you are in a press interview. "
              f"{STYLE}, as you would in the interview.\n\nJournalist: {turn(row)}"),
        "B": (f"You are {who}, President of the United States. It is {when}, and you are in a press interview. Below "
              f"are your private notes on this topic. {STYLE}, as you would in the interview, using what is in your "
              f"notes.\n\nNotes: {' '.join(str(row['interview_answer']).split()[:NOTES_WORDS])}\n\n"
              f"Journalist: {turn(row)}"),
    }


def follow_up(sub_question: str) -> str:
    return f"Journalist: With respect, you haven't answered the question. {' '.join(str(sub_question).split())}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", required=True, help="name=path-or-hf-id; repeatable")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-util", type=float, default=0.45)  # shares the GPU with the 8B scorer
    p.add_argument("--limit", type=int, default=0, help="first N dev rows only (smoke test)")
    a = p.parse_args()
    from vllm import LLM, SamplingParams

    dev = load_qevasion(DATA_CACHE).dev.reset_index(drop=True)
    if a.limit:
        dev = dev.iloc[: a.limit]
    key = dev["interview_question"].astype(str).str.split().str.join(" ") + " ||| " + \
        dev["interview_answer"].astype(str).str.split().str.join(" ")
    groups = {}
    for i, k in enumerate(key):
        groups.setdefault(k, []).append(i)
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            done.add((r["model"], r["condition"], r["src_i"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sp = SamplingParams(max_tokens=400, temperature=0.0, seed=0)
    for spec in a.model:
        name, _, path = spec.partition("=")
        if all((name, c, i) in done for c in "NPBF" for i in range(len(dev))):
            print(f"[done] {name}", flush=True)
            continue
        t0 = time.time()
        llm = LLM(model=path, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_util, seed=0,
                  enable_prefix_caching=True)
        kw = {"chat_template_kwargs": {"enable_thinking": False}} if "qwen3" in path.lower() else {}
        print(f"{name}: loaded in {time.time() - t0:.0f} s", flush=True)
        reps = [g[0] for g in groups.values()]
        conv = {c: [[{"role": "user", "content": prompts(dev.iloc[r])[c]}] for r in reps] for c in "NPB"}
        res = {}
        for c in "NPB":
            t1 = time.time()
            outs = llm.chat(conv[c], sp, use_tqdm=False, **kw)
            res[c] = outs
            print(f"{name} {c}: {len(outs)} turns in {time.time() - t1:.0f} s", flush=True)
        p_answer = {r: o.outputs[0].text for r, o in zip(reps, res["P"])}
        rep_of = {i: g[0] for g in groups.values() for i in g}
        f_conv = [conv["P"][reps.index(rep_of[i])] + [{"role": "assistant", "content": p_answer[rep_of[i]]},
                                                       {"role": "user", "content": follow_up(dev.iloc[i]["question"])}]
                  for i in range(len(dev))]
        t1 = time.time()
        f_outs = llm.chat(f_conv, sp, use_tqdm=False, **kw)
        print(f"{name} F: {len(f_outs)} sub-questions in {time.time() - t1:.0f} s", flush=True)
        with OUT.open("a") as f:
            for c in "NPB":
                for r, o in zip(reps, res[c]):
                    for i in groups[key[r]]:
                        if (name, c, i) not in done:
                            f.write(json.dumps({"model": name, "condition": c, "src_i": i, "answer": o.outputs[0].text,
                                                "finish": o.outputs[0].finish_reason,
                                                "prompt_tokens": len(o.prompt_token_ids),
                                                "gen_tokens": len(o.outputs[0].token_ids), "shared_turn": r}) + "\n")
            for i, o in enumerate(f_outs):
                if (name, "F", i) not in done:
                    f.write(json.dumps({"model": name, "condition": "F", "src_i": i, "answer": o.outputs[0].text,
                                        "finish": o.outputs[0].finish_reason, "prompt_tokens": len(o.prompt_token_ids),
                                        "gen_tokens": len(o.outputs[0].token_ids)}) + "\n")
        print(f"{name}: written ({time.time() - t0:.0f} s in all)", flush=True)
        del llm
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
        # vLLM keeps its engine process alive in some versions; a fresh process per model is safest, so the lane
        # calls this script once per model


if __name__ == "__main__":
    main()
