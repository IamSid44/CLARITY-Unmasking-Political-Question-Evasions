#!/usr/bin/env python3
"""Mid-eval figure for E13: DeBERTa -> Qwen3-8B LoRA, per seed and as systems.

    python clarity/e13_figures.py      -> mideval/figures/e13_qwen_vs_deberta.png

Left: dev S2 of single models, one grey line per seed (paired: same seed in every
configuration), the blue marker the mean. Right: systems = mean of the seeds'
probabilities + logit adjustment, tau fitted by 5-fold nested CV (decide.py R1),
averaged over 10 CV splits, next to published dev numbers (dashed lines).
Computed from the saved dev probabilities; same style as mideval_figures.py.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from decide import make_rules, nested_cv
from qevasion.labels import N_EVASION, encode_evasion
from qevasion.loader import dev_reference_mask, load_qevasion
from qevasion.scoring import score_subtask2

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "mideval" / "figures"
CONFIGS = [  # label, run folders (seeds 0-4 and 5-9 may live in different folders)
    ("DeBERTa-v3-large\nbaseline", ["L0_large_base", "E11_base_8ep"]),
    ("DeBERTa-v3-large\n+ full question, 16 ep", ["E10_fullq_16ep", "E11_fullq_16ep"]),
    ("Qwen3-8B LoRA\n(same input, 3 ep)", ["Q8_fullq_lora"]),
    ("Qwen3-8B LoRA\n+ all of train", ["Q8_alldata"]),
]
REFS = [("TeleAI, Qwen2.5-7B fine-tuned", 0.495), ("ChulaNLP, RoBERTa top-5 + Kimi-K2", 0.52),
        ("TeleAI, 3-stage DeepSeek-V3 (1st)", 0.617), ("human annotator vs the other two", 0.684)]
SURFACE, INK, INK2, GRID, SEED, MEAN = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b1", "#2a78d6"
BAR_D, BAR_Q, REF = "#9a9a94", "#2a78d6", "#c2410c"


def load(runs: list[str]) -> dict[int, np.ndarray]:
    out = {}
    for r in runs:
        for f in sorted((ROOT / "runs" / r).glob("seed*/dev_probs.npy")):
            out[int(f.parent.name.removeprefix("seed"))] = np.load(f)
    return out


def main() -> None:
    sp = load_qevasion()
    gold = dev_reference_mask(sp.dev)
    y = encode_evasion(sp.train["evasion_label"].tolist())
    r1 = make_rules(np.bincount(y, minlength=N_EVASION) / len(y))["R1_logit_adjusted"]
    probs = [load(r) for _, r in CONFIGS]
    seeds = sorted(set.intersection(*(set(p) for p in probs)))
    s2 = [{s: score_subtask2(p[s].argmax(1), gold).macro_f1 for s in p} for p in probs]
    systems = [np.mean([nested_cv(r1, np.mean([p[s] for s in sorted(p)], 0), gold, 5, cs)["out_of_fold_macro_f1"]
                        for cs in range(10)]) for p in probs]
    n_seeds = [len(p) for p in probs]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.15, 1]})
    x = np.arange(len(CONFIGS))
    for ax in (a1, a2):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)

    for s in seeds:
        a1.plot(x, [sc[s] for sc in s2], color=SEED, lw=1.2, marker="o", ms=4, zorder=2)
    means = [np.mean([sc[s] for s in seeds]) for sc in s2]
    a1.plot(x, means, color=MEAN, lw=2, marker="o", ms=9, mec=SURFACE, mew=2, zorder=3)
    for xi, m in zip(x, means):
        a1.annotate(f"{m:.3f}", (xi, m), xytext=(10, 0), textcoords="offset points", va="center",
                    fontsize=9.5, fontweight="bold")
    a1.set_xticks(x, [c for c, _ in CONFIGS], fontsize=8.8)
    a1.set_ylabel("dev Subtask 2 macro-F1 (multi-reference)")
    a1.set_title(f"Single models, {len(seeds)} paired seeds (grey: one seed; blue: mean)", loc="left", fontsize=10)

    cols = [BAR_D, BAR_D, BAR_Q, BAR_Q]
    a2.bar(x, systems, color=cols, width=0.62, zorder=2)
    for xi, v, n in zip(x, systems, n_seeds):  # values inside the bars, clear of the reference lines
        a2.annotate(f"{v:.3f}\n{n} seeds", (xi, v), xytext=(0, -6), textcoords="offset points", ha="center",
                    va="top", fontsize=9, fontweight="bold", color="white", zorder=4)
    for i, (lab, v) in enumerate(REFS):
        a2.axhline(v, color=REF, lw=1, ls="--", zorder=3, alpha=0.8)
        below = lab.startswith("TeleAI, Qwen")      # 0.495 sits just under ChulaNLP's 0.52
        a2.annotate(f"{lab}  {v:.3f}", (-0.42, v), xytext=(0, -2 if below else 2), textcoords="offset points",
                    ha="left", va="top" if below else "bottom", fontsize=7.8, color=REF, zorder=5,
                    bbox=dict(boxstyle="square,pad=0.1", fc=SURFACE, ec="none", alpha=0.85))
    a2.set_xticks(x, [c for c, _ in CONFIGS], fontsize=8.8)
    a2.set_ylim(0, 0.75)
    a2.set_title("Systems: seed ensemble + logit adjustment (nested CV, mean of 10 splits)", loc="left", fontsize=10)

    fig.suptitle("E13: replacing the encoder with Qwen3-8B (LoRA) — dev, Subtask 2", x=0.01, ha="left",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "e13_qwen_vs_deberta.png", dpi=160, facecolor=SURFACE)
    print("single-model means:", [round(m, 3) for m in means], "over seeds", seeds)
    print("systems:", [round(v, 3) for v in systems], "seeds per config", n_seeds)
    print("wrote", OUT / "e13_qwen_vs_deberta.png")


if __name__ == "__main__":
    main()
