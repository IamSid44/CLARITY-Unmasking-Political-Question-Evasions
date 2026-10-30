#!/usr/bin/env python3
"""E25 figures from runs/E25/analysis.json (written by preregistered_analyses.e25_analysis).

    python -m figures.e25_figures      # -> docs/figures/E25_*.pdf and report/latex/figures/E25_*.pdf

Palette: the dataviz reference categorical slots 1-4 (validated light-mode: CVD and normal-vision floors pass;
two slots are below 3:1 contrast, so every series also has its own marker shape and a legend).
"""

from __future__ import annotations

import json
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from qevasion.paths import FIGURES, REPORT_FIGURES, RUNS  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "D", "^"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
CONDS = {"N": "Neutral analyst", "P": "As the President", "F": "After a follow-up", "B": "Briefed with his answer"}
MODELS = {"llama31_8b": "Llama-3.1-8B", "mistral_7b": "Mistral-7B", "qwen3_8b": "Qwen3-8B", "olmo2_7b": "OLMo-2-7B"}

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "pdf.fonttype": 42})


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"{name}.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight", dpi=200)
    REPORT_FIGURES.mkdir(parents=True, exist_ok=True)
    shutil.copy(out, REPORT_FIGURES / out.name)
    plt.close(fig)
    print("wrote", out)


def audit(R: dict) -> None:
    """Evasion rate per model and condition (95% cluster-bootstrap CI), Presidents as the reference line;
    right: Mantel-Haenszel odds ratios (question-driven component) with the question-only control."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.3, 1]})
    fig.subplots_adjust(wspace=0.95)
    rates, ci = R["rates"]["table"], R["rates"]["ci"]
    ms = [m for m in MODELS if m in rates]
    w = 0.18
    for k, c in enumerate(CONDS):
        xs = [i + (k - 1.5) * w for i in range(len(ms))]
        ys = [rates[m][c] for m in ms]
        lo = [rates[m][c] - ci[m][c][0] for m in ms]
        hi = [ci[m][c][1] - rates[m][c] for m in ms]
        a.errorbar(xs, ys, yerr=[lo, hi], fmt=MARKERS[k], ms=4.5, color=SERIES[k], ecolor=SERIES[k], elinewidth=1,
                   capsize=0, label=CONDS[c], mec="white", mew=0.6)
    pg = R["rates"]["president_gold"]
    a.axhline(pg, color=INK, lw=1, ls="--")
    a.text(-0.45, pg + 0.015, f"Presidents (gold) {pg:.2f}", color=INK, va="bottom", ha="left", fontsize=7)
    a.set_xticks(range(len(ms)), [MODELS[m] for m in ms])
    a.set_xlim(-0.5, len(ms) - 0.5)
    a.set_ylabel("Share of answers that evade\n(clarity not Clear Reply)")
    a.set_ylim(0, 1)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=6.5, ncol=2, handletextpad=0.2, columnspacing=0.8)
    rows = [(CONDS[c], R["or"][c]) for c in CONDS if c in R["or"]] + [("Control: question only", R["or"]["qonly"])]
    for j, (name, v) in enumerate(rows):
        y = len(rows) - 1 - j
        col = SERIES[j] if j < 4 else MUTED
        b.plot(v["ci"], [y, y], color=col, lw=1.2)
        b.plot([v["mh"]], [y], MARKERS[j] if j < 4 else "x", color=col, ms=5, mec="white" if j < 4 else col, mew=0.6)
    b.axvline(1, color=INK, lw=0.8, ls=":")
    b.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    b.set_xscale("log")
    b.set_xlabel("Odds ratio (log): LLM evades\nwhere the President did")
    b.grid(axis="y", visible=False)
    save(fig, "E25_audit")  # tight bbox keeps the forest labels


def swap_prefix_probe(R: dict) -> None:
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(6.6, 2.1))
    e = R["e25a"]
    conds = [("own", "own"), ("rand", "unrelated"), ("null", "none")]
    for k, (split, mk) in enumerate((("dev", "o"), ("test", "s"))):
        ys = [e[f"{split}_{cc}"]["cr"] for cc, _ in conds]
        a.plot(range(3), ys, marker=mk, color=SERIES[k], lw=1.5, ms=4.5, label=f"{split}: Clear-Reply share",
               mec="white", mew=0.6)
        ps = [e[f"{split}_{cc}"]["persist"] for cc, _ in conds]
        a.plot(range(3), ps, marker=mk, color=SERIES[k], lw=1.2, ls="--", ms=4, label=f"{split}: label unchanged",
               mec="white", mew=0.6)
    a.set_xticks(range(3), [n for _, n in conds])
    a.set_ylim(0, 1.05)
    a.set_title("Swapping the question", fontsize=8, color=INK)
    a.set_xlabel("Question paired with the answer")
    a.legend(fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=1)
    p = R["prefix"]
    xs = [25, 50, 75, 100]
    for k, split in enumerate(s for s in ("dev", "test") if f"{s}_25" in p):
        ys = [p[f"{split}_{x}"]["agree"] for x in xs[:3]] + [1.0]
        b.plot(xs, ys, marker=MARKERS[k], color=SERIES[k], lw=1.5, ms=4.5, label=split, mec="white", mew=0.6)
    b.set_xticks(xs, [f"{x}%" for x in xs])
    b.set_xlabel("Share of the answer read")
    b.set_ylabel("Same label as the full answer")
    b.set_ylim(0, 1.05)
    b.set_title("Reading less of the answer", fontsize=8, color=INK)
    b.legend(fontsize=6.5, loc="lower right")
    pr = R["probes"]
    c.plot([r["layer"] for r in pr], [r["dev_s2"] for r in pr], color=SERIES[0], lw=1.5, label="frozen-layer probe")
    c.axhline(0.525, color=SERIES[1], lw=1, ls="--")
    c.text(1, 0.525, "one fine-tuned seed", color=INK, fontsize=6.5, va="bottom")
    c.axhline(0.598, color=SERIES[2], lw=1, ls=":")
    c.text(1, 0.598, "10-seed M1", color=INK, fontsize=6.5, va="bottom")
    c.set_xlabel("Layer of the frozen 8B")
    c.set_ylabel("Dev S2 macro-F1")
    c.set_ylim(0, 0.7)
    c.set_title("Probing the frozen model", fontsize=8, color=INK)
    fig.tight_layout(w_pad=1.2)
    save(fig, "E25_swap_prefix_probe")


def main() -> None:
    R = json.loads((RUNS / "E25" / "analysis.json").read_text())
    audit(R)
    swap_prefix_probe(R)


if __name__ == "__main__":
    main()
