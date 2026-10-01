#!/usr/bin/env python3
"""Mid-eval figure: per-seed single-model scores across the three model configurations.

    python clarity/mideval_figures.py      -> mideval/figures/per_seed_progression.png

Each grey line is one seed (same initialisation and data order in every configuration);
the blue marker is the mean of the 10 seeds. Computed from the saved dev probabilities.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qevasion.labels import encode_clarity, leaf_to_official_clarity
from qevasion.loader import DATA_CACHE, dev_reference_mask, load_qevasion
from qevasion.scoring import score_subtask1, score_subtask2

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "mideval" / "figures"
CONFIGS = [
    ("sub-question + answer\n8 epochs", ["L0_large_base", "E11_base_8ep"]),
    ("sub-question + answer\n16 epochs", ["E10_base_16ep", "E11_base_16ep"]),
    ("+ full question\n16 epochs", ["E10_fullq_16ep", "E11_fullq_16ep"]),
]
SURFACE, INK, INK2, GRID, SEED, MEAN = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b1", "#2a78d6"


def per_seed(runs, mask, ctrue):
    s2, s1 = {}, {}
    for r in runs:
        for f in sorted((ROOT / "runs" / r).glob("seed*/dev_probs.npy")):
            seed = int(f.parent.name.removeprefix("seed"))
            pred = np.load(f).argmax(1)
            s2[seed] = score_subtask2(pred, mask).macro_f1
            s1[seed] = score_subtask1(leaf_to_official_clarity(pred), ctrue).macro_f1
    return s2, s1


def main() -> None:
    dev = load_qevasion(DATA_CACHE).dev
    mask, ctrue = dev_reference_mask(dev), encode_clarity(dev["clarity_label"].tolist())
    scores = [per_seed(runs, mask, ctrue) for _, runs in CONFIGS]
    seeds = sorted(scores[0][0])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), facecolor=SURFACE)
    x = np.arange(len(CONFIGS))
    for ax, k, title in ((axes[0], 0, "Subtask 2 (9 evasion types)"), (axes[1], 1, "Subtask 1 (3 clarity levels)")):
        ax.set_facecolor(SURFACE)
        for s in seeds:
            ax.plot(x, [sc[k][s] for sc in scores], color=SEED, lw=1.2, marker="o", ms=4, zorder=2)
        means = [np.mean([sc[k][s] for s in seeds]) for sc in scores]
        ax.plot(x, means, color=MEAN, lw=2, marker="o", ms=9, mec=SURFACE, mew=2, zorder=3)
        for xi, m in zip(x, means):
            ax.annotate(f"{m:.3f}", (xi, m), xytext=(10, 0), textcoords="offset points",
                        va="center", fontsize=9.5, color=INK, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none"))
        up = sum(scores[2][k][s] > scores[0][k][s] for s in seeds)
        ax.set_title(f"{title}\n{up} of {len(seeds)} seeds higher in the last configuration than the first",
                     fontsize=10, color=INK, loc="left")
        ax.set_xticks(x, [c for c, _ in CONFIGS], fontsize=9)
        ax.set_xlim(-0.3, len(CONFIGS) - 0.55)
        ax.set_ylabel("dev macro-F1, single model")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
    handles = [plt.Line2D([], [], color=SEED, lw=1.2, marker="o", ms=4, label="one seed (10 seeds, paired)"),
               plt.Line2D([], [], color=MEAN, lw=2, marker="o", ms=8, label="mean of 10 seeds")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("DeBERTa-v3-large: training longer and adding the full question improve single models",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "per_seed_progression.png", dpi=160, facecolor=SURFACE)
    print(OUT / "per_seed_progression.png")


if __name__ == "__main__":
    main()
