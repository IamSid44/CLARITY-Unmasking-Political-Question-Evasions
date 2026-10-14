"""API spend for E15: prices, and a ledger that refuses any request once the hard cap is reached.

Every API response's `usage` is appended to runs/M3/api_ledger.jsonl as it arrives, so the total survives
crashes and is shared by every script (probe, screen, full runs). The cap is checked before each request.
"""

from __future__ import annotations

import json
import re
import threading
import time

from qevasion.paths import RUNS

GLOBAL_CAP_USD = 8.0  # E15 registration: all API spend for M3, ≈ ₹700
LEDGER = RUNS / "M3" / "api_ledger.jsonl"
BASE_URL = "https://models.jarvislabs.net/v1"

# USD per 1M tokens (input, cached input, output), JarvisLabs Model APIs page, read 2026-10-13.
PRICES: dict[str, tuple[float, float, float]] = {
    "deepseekv4flash": (0.13, 0.03, 0.26),
    "gemma431b": (0.10, 0.05, 0.34),
    "gemma426ba4b": (0.11, 0.05, 0.37),
    "qwen3827b": (0.30, 0.08, 2.35),
    "glm53flash": (0.14, 0.03, 0.45),
    "glm53": (1.21, 0.22, 3.86),
    "gptoss120b": (0.10, 0.05, 0.42),
}


class BudgetExceeded(RuntimeError):
    pass


def price(model: str) -> tuple[float, float, float]:
    """Price of an API model id, matched on its alphanumerics (longest key first, so glm53flash != glm53)."""
    import os

    if os.environ.get("CLARITY_API_PRICE"):  # smoke tests against a local OpenAI-compatible server: "0,0,0"
        pin, pcached, pout = (float(x) for x in os.environ["CLARITY_API_PRICE"].split(","))
        return pin, pcached, pout
    key = re.sub(r"[^a-z0-9]", "", model.lower())
    for k in sorted(PRICES, key=len, reverse=True):
        if k in key:
            return PRICES[k]
    raise SystemExit(f"no price known for API model {model!r}: add it to cascade/api_budget.PRICES first")


def usage_dict(u) -> dict:
    """prompt / cached / completion / reasoning token counts from an OpenAI-style usage object (or dict)."""
    g = (lambda o, k: (o.get(k) if isinstance(o, dict) else getattr(o, k, None)) if o is not None else None)
    ptd, ctd = g(u, "prompt_tokens_details"), g(u, "completion_tokens_details")
    return {"prompt": int(g(u, "prompt_tokens") or 0), "cached": int(g(ptd, "cached_tokens") or 0),
            "completion": int(g(u, "completion_tokens") or 0), "reasoning": int(g(ctd, "reasoning_tokens") or 0)}


def cost_usd(model: str, u: dict) -> float:
    pin, pcached, pout = price(model)
    return ((u["prompt"] - u["cached"]) * pin + u["cached"] * pcached + u["completion"] * pout) / 1e6


class Ledger:
    def __init__(self, tag: str, run_cap_usd: float):
        self.tag, self.run_cap = tag, run_cap_usd
        self.run_spent = 0.0
        self._lock = threading.Lock()
        LEDGER.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def total() -> float:
        if not LEDGER.exists():
            return 0.0
        return sum(json.loads(line)["usd"] for line in LEDGER.read_text().splitlines() if line.strip())

    def check(self, upcoming_usd: float = 0.0) -> None:
        total = self.total()
        if total + upcoming_usd > GLOBAL_CAP_USD:
            raise BudgetExceeded(f"E15 API cap: spent ${total:.3f} + ${upcoming_usd:.3f} > ${GLOBAL_CAP_USD:.2f}")
        if self.run_spent + upcoming_usd > self.run_cap:
            raise BudgetExceeded(f"run cap: this run spent ${self.run_spent:.3f} of ${self.run_cap:.2f}")

    def add(self, model: str, u: dict) -> float:
        usd = cost_usd(model, u)
        with self._lock:
            self.run_spent += usd
            with LEDGER.open("a") as f:
                f.write(json.dumps({"t": time.strftime("%F %T"), "tag": self.tag, "model": model, **u,
                                    "usd": round(usd, 7)}) + "\n")
        return usd
