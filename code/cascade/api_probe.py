#!/usr/bin/env python3
"""E15 step 0: what the JarvisLabs Model API gives us, before any scoring run (cap $1).

    python -m cascade.api_probe [--models deepseek,gemma-4-31b,qwen3.8] > ../docs/raw/E15_api_probe.txt

For each model: its exact id (from /v1/models); whether chat completions return log-probabilities with
top_logprobs; which switch disables thinking (none / chat_template_kwargs / reasoning_effort / thinking);
whether a repeated long prompt is billed as cached input; whether temperature 0 is deterministic; whether
n > 1 samples are supported (the fallback if log-probs are missing); rate-limit headers. Every call's usage
goes to the E15 ledger. The key is read from code/.env (JARVIS_API_KEY) and never printed.
"""

from __future__ import annotations

import argparse
import re
import time

from cascade.adjudicate import api_key, build_prompts, select
from cascade.api_budget import BASE_URL, Ledger, cost_usd, usage_dict

THINKING_SWITCHES = {
    "none": {},
    "chat_template_kwargs": {"chat_template_kwargs": {"enable_thinking": False}},
    "reasoning_effort": {"reasoning_effort": "none"},
    "thinking_disabled": {"thinking": {"type": "disabled"}},
}
SHORT = [{"role": "system", "content": "Answer with one capital letter only."},
         {"role": "user", "content": "Which is a colour? A. Table B. Blue C. Seven D. Run\nReply with one letter."}]


def show(label: str, obj) -> None:
    print(f"  {label}: {obj}", flush=True)


def call(client, ledger, model, messages, **kw):
    ledger.check(0.05)
    t0 = time.time()
    raw = client.chat.completions.with_raw_response.create(model=model, messages=messages, **kw)
    r = raw.parse()
    u = usage_dict(r.usage)
    ledger.add(model, u)
    headers = {k: v for k, v in raw.headers.items() if "ratelimit" in k.lower() or k.lower() == "retry-after"}
    return r, u, time.time() - t0, headers


def top_of(r):
    lp = r.choices[0].logprobs
    if not lp or not lp.content:
        return None
    return [(t.token, round(t.logprob, 4)) for t in lp.content[0].top_logprobs]


def probe(client, ledger, model: str, long_msgs, switch: str, only_long: bool) -> None:
    print(f"\n=== {model}", flush=True)
    # 1. log-probs and thinking switches, on a trivial prompt
    for name, extra in ({} if only_long else THINKING_SWITCHES).items():
        try:
            r, u, sec, hdr = call(client, ledger, model, SHORT, max_tokens=8, temperature=0.0, logprobs=True,
                                  top_logprobs=20, extra_body=extra)
        except Exception as e:  # noqa: BLE001
            show(f"switch {name}", f"ERROR {type(e).__name__}: {str(e)[:200]}")
            continue
        msg = r.choices[0].message
        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
        show(f"switch {name}", f"content={msg.content!r} reasoning={'yes' if reasoning else 'no'} "
                               f"finish={r.choices[0].finish_reason} usage={u} {sec:.1f}s")
        top = top_of(r)
        show("  top_logprobs (first token)", top[:6] if top else "NONE")
        if hdr and name == "none":
            show("rate-limit headers", hdr)
    # 2. max_tokens=1 with the long, real prompt and thinking switched off: letters, cached input on the
    #    repeat, determinism at T=0
    extra = THINKING_SWITCHES[switch]
    show("long-prompt switch", f"{switch} {extra}")
    rows = []
    for rep in range(3):
        try:
            r, u, sec, _ = call(client, ledger, model, long_msgs, max_tokens=1, temperature=0.0, logprobs=True,
                                top_logprobs=20, extra_body=extra)
        except Exception as e:  # noqa: BLE001
            show(f"long prompt rep {rep}", f"ERROR {type(e).__name__}: {str(e)[:200]}")
            break
        rows.append(top_of(r))
        show(f"long prompt rep {rep}", f"content={r.choices[0].message.content!r} usage={u} "
                                       f"cost=${cost_usd(model, u):.5f} {sec:.1f}s")
        time.sleep(1)
    if len(rows) >= 2 and rows[0] and rows[1]:
        same = [a[0] for a in rows[0][:5]] == [b[0] for b in rows[1][:5]]
        dmax = max(abs(a[1] - b[1]) for a, b in zip(rows[0][:5], rows[1][:5]))
        show("deterministic at T=0 (top-5 tokens equal; max |Δ logprob|)", f"{same}; {dmax:.4f}")
        show("top-5 of rep 0", rows[0][:5])
    # 3. n > 1 (the sampling fallback)
    try:
        r, u, sec, _ = call(client, ledger, model, SHORT, max_tokens=1, temperature=1.0, n=3, extra_body=extra)
        show("n=3", f"{len(r.choices)} choices: {[c.message.content for c in r.choices]} usage={u}")
    except Exception as e:  # noqa: BLE001
        show("n=3", f"ERROR {type(e).__name__}: {str(e)[:200]}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default="deepseek-v4-flash,gemma-4-31b,qwen3.8-27b",
                   help="comma-separated patterns matched against /v1/models ids")
    p.add_argument("--base-url", default=BASE_URL)
    p.add_argument("--switch", default="chat_template_kwargs", choices=tuple(THINKING_SWITCHES),
                   help="thinking switch for the long-prompt and n>1 checks")
    p.add_argument("--only-long", action="store_true", help="skip the switch survey (already run)")
    a = p.parse_args()
    from openai import OpenAI

    client = OpenAI(base_url=a.base_url, api_key=api_key(), max_retries=3, timeout=180)
    ledger = Ledger("api_probe", run_cap_usd=1.0)
    print(f"E15 API probe {time.strftime('%F %T')}; ledger total before: ${ledger.total():.4f}")
    ids = sorted(m.id for m in client.models.list().data)
    print(f"models listed by {a.base_url}: {ids}")
    long_msgs = build_prompts(select("screen")[:1], k=10)[0][0]
    print(f"long prompt: {sum(len(m['content']) for m in long_msgs)} characters (a real screen item, k = 10)")
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())  # noqa: E731
    for pat in a.models.split(","):
        hits = [i for i in ids if norm(pat) in norm(i)]
        if not hits:
            print(f"\n=== {pat}: no listed model matches")
            continue
        probe(client, ledger, hits[0], long_msgs, a.switch, a.only_long)
    print(f"\nprobe spend ${ledger.run_spent:.4f}; E15 ledger total ${ledger.total():.4f}")


if __name__ == "__main__":
    main()
