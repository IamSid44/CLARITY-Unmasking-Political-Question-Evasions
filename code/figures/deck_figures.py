#!/usr/bin/env python3
"""Slide-deck figures, in the publication style of paper_figures.py, sized for 16:9 slides.

Every number is copied from the file named beside it, so the figures rebuild on a CPU
without the runs/ folder:

    python -m figures.deck_figures          # from code/; writes docs/figures/deck/*.png and *.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures" / "deck"

# paper_figures.py's rcParams, scaled up so that a figure placed ~6 in wide on a slide reads at ~12 pt.
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 10.5, "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.top": False, "ytick.right": True, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300, "figure.facecolor": "white", "axes.facecolor": "white",
})
SEED_C, MEAN_C, DEB_C, QWEN_C = "0.70", "black", "0.80", "0.45"
SEED_LEGEND = [
    Line2D([], [], color=SEED_C, lw=0.8, marker="o", ms=5, mfc="white", mec=SEED_C, label="one seed (paired)"),
    Line2D([], [], color=MEAN_C, ls="none", marker="s", ms=6, label="mean $\\pm$ 1 s.d."),
]

# Per-seed single-model dev scores, seeds 0-9 (DeBERTa: runs/<config>/seed<k>/metrics.json, which
# docs/raw/E11_replication.txt summarises; Qwen: docs/raw/E13_final_analysis.txt, sections A and C).
S2 = {
    "base8": [0.3891, 0.3393, 0.2794, 0.3106, 0.3677, 0.2810, 0.2967, 0.3167, 0.2659, 0.3044],
    "base16": [0.3589, 0.3662, 0.3979, 0.3630, 0.4005, 0.3366, 0.3822, 0.3213, 0.3552, 0.3383],
    "fullq16": [0.3443, 0.3700, 0.4280, 0.4213, 0.3686, 0.3884, 0.3789, 0.3464, 0.3753, 0.4168],
    "fullq8": [0.3142, 0.2523, 0.3037, 0.2200, 0.3178],  # E8b, seeds 0-4
    "qwen": [0.405, 0.426, 0.554, 0.480, 0.512, 0.495, 0.461, 0.501, 0.474, 0.449],
    "qwen_all": [0.520, 0.443, 0.538, 0.437, 0.535, 0.552, 0.492, 0.515, 0.454, 0.438],
}
S1 = {
    "base8": [0.6417, 0.5797, 0.5323, 0.5958, 0.5786, 0.5440, 0.5601, 0.5772, 0.5655, 0.5822],
    "base16": [0.6174, 0.6121, 0.5924, 0.6091, 0.5894, 0.6013, 0.6291, 0.5994, 0.5923, 0.5722],
    "fullq16": [0.5913, 0.6268, 0.6402, 0.6406, 0.6550, 0.5836, 0.6257, 0.5826, 0.5967, 0.5935],
}


def _save(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def paired_panel(ax, series: list[list[float]], labels: list[str], title: str, ylabel: str | None):
    """paper_figures.paired_panel: open marker per seed, thin lines join a seed; square = mean +/- 1 s.d."""
    x = np.arange(len(series))
    for s in range(len(series[0])):
        ax.plot(x, [v[s] for v in series], color=SEED_C, lw=0.8, zorder=1)
        ax.plot(x, [v[s] for v in series], ls="none", marker="o", ms=5, mfc="white", mec=SEED_C, mew=0.9, zorder=2)
    mean = [np.mean(v) for v in series]
    sd = [np.std(v, ddof=1) for v in series]
    ax.errorbar(x, mean, yerr=sd, fmt="s", ms=6, color=MEAN_C, elinewidth=1.1, capsize=3.5, capthick=1.1, zorder=3)
    for xi, m in zip(x, mean):
        ax.annotate(f"{m:.3f}", (xi, m), xytext=(9, 0), textcoords="offset points", va="center", fontsize=10.5,
                    bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none"), zorder=4)
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.4, len(series) - 0.45)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=5)


def deberta_seeds() -> None:
    """Report Figure 2: the two changes that improved single DeBERTa models."""
    labels = ["Baseline\n8 ep.", "Baseline\n16 ep.", "+ Full q.\n16 ep."]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.8))
    paired_panel(axes[0], [S2[k] for k in ("base8", "base16", "fullq16")], labels, "(a) Subtask 2", "Dev macro-F1")
    paired_panel(axes[1], [S1[k] for k in ("base8", "base16", "fullq16")], labels, "(b) Subtask 1", None)
    fig.legend(handles=SEED_LEGEND, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(pad=0.3, w_pad=1.2, rect=(0, 0.07, 1, 1))
    _save(fig, "deberta_seeds")


def qwen_vs_deberta() -> None:
    """Report Figure 3: single models on 10 paired seeds, and the 10-seed systems against published dev results."""
    labels = ["DeBERTa\nbaseline", "DeBERTa\nfull q.", "Qwen-8B\nLoRA", "Qwen-8B\nall data"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 4.0), gridspec_kw={"width_ratios": [1, 1.05]})
    paired_panel(a1, [S2[k] for k in ("base8", "fullq16", "qwen", "qwen_all")], labels,
                 "(a) Single models, 10 paired seeds", "Dev Subtask 2 macro-F1")
    a1.legend(handles=SEED_LEGEND, loc="upper left", frameon=False)
    a1.set_ylim(0.25, 0.62)

    systems = [0.444, 0.412, 0.549, 0.562]  # nested CV, mean of 10 splits (E11_replication.txt, E13_final_analysis.txt)
    x = np.arange(4)
    a2.bar(x, systems, width=0.6, color=[DEB_C, DEB_C, QWEN_C, QWEN_C], edgecolor="black", linewidth=0.8,
           hatch=["", "", "////", "////"], zorder=2)
    for xi, v in zip(x, systems):
        a2.text(xi, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=10.5, zorder=4,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    refs = [("Human vs. other two", 0.684, (0, (1, 1.2))), ("TeleAI pipeline", 0.617, (0, (5, 2))),
            ("ChulaNLP hybrid", 0.520, (0, (5, 1.5, 1, 1.5))), ("TeleAI Qwen2.5-7B FT", 0.495, "-")]
    handles = []
    for lab, v, ls in refs:
        a2.axhline(v, color="black", lw=0.9, ls=ls, zorder=3)
        handles.append(Line2D([], [], color="black", lw=0.9, ls=ls, label=f"{lab} ({v:.3f})"))
    a2.legend(handles=handles, loc="upper center", ncol=2, frameon=True, framealpha=1, edgecolor="0.6",
              fancybox=False, handlelength=2.4, columnspacing=1.0, fontsize=9.5)
    a2.set_xticks(x, labels)
    a2.set_xlim(-0.55, 3.55)
    a2.set_ylim(0, 1.0)
    a2.set_yticks(np.arange(0, 0.81, 0.2))
    a2.set_ylabel("Dev Subtask 2 macro-F1")
    a2.set_title("(b) Systems: 10-seed ensemble + logit adjustment", loc="left", pad=5)
    fig.tight_layout(pad=0.3, w_pad=2.0)
    _save(fig, "qwen_vs_deberta")


def undertraining() -> None:
    """E8b -> E10: the full question looks harmful at 8 epochs and helps at 16 (seeds 0-4, paired)."""
    labels = ["8 epochs", "16 epochs"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.8), sharey=True)
    paired_panel(axes[0], [S2["base8"][:5], S2["base16"][:5]], labels, "(a) Sub-question + answer",
                 "Dev Subtask 2 macro-F1")
    paired_panel(axes[1], [S2["fullq8"], S2["fullq16"][:5]], labels, "(b) + Full question", None)
    axes[1].annotate("final train loss 1.32-1.50\nvs. 0.57-1.09 in (a)", xy=(0.0, 0.282), xytext=(0.18, 0.226),
                     fontsize=10, arrowprops=dict(arrowstyle="-", lw=0.7, color="0.35"))
    axes[0].set_ylim(0.20, 0.45)
    fig.legend(handles=SEED_LEGEND, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(pad=0.3, w_pad=1.2, rect=(0, 0.07, 1, 1))
    _save(fig, "undertraining")


def decision_rules() -> None:
    """E4: held-out score of four decision rules on the E0 5-seed ensemble (docs/02_experiment_log.md, E4)."""
    rules = ["Argmax\n(0 params)", "Set member.\n(0 params)", "Logit adj.\n(1 param)", "Per-class\n(9 params)"]
    held = [0.365, 0.365, 0.438, 0.394]
    gap = [-0.001, None, 0.018, 0.098]  # in-fold minus held-out; not reported for set membership
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.5))
    x = np.arange(4)
    hatches = ["", "", "////", ""]
    colors = [DEB_C, DEB_C, QWEN_C, DEB_C]
    a.bar(x, held, width=0.6, color=colors, hatch=hatches, edgecolor="black", linewidth=0.8, zorder=2)
    for xi, v in zip(x, held):
        a.text(xi, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=10.5)
    a.set_ylim(0.30, 0.46)
    a.set_xticks(x, rules, fontsize=10)
    a.set_ylabel("Dev S2, held out (nested CV)")
    a.set_title("(a) Score after the rule", loc="left", pad=5)

    xs = [i for i, g in enumerate(gap) if g is not None]
    b.bar(xs, [gap[i] for i in xs], width=0.6, color=[colors[i] for i in xs], hatch=[hatches[i] for i in xs],
          edgecolor="black", linewidth=0.8, zorder=2)
    for xi in xs:
        b.text(xi, max(gap[xi], 0) + 0.002, f"{gap[xi]:+.3f}", ha="center", va="bottom", fontsize=10.5)
    b.text(1, 0.003, "n/a", ha="center", va="bottom", fontsize=10.5, color="0.4")
    b.axhline(0, color="black", lw=0.8)
    b.set_ylim(-0.01, 0.115)
    b.set_xticks(x, rules, fontsize=10)
    b.set_ylabel("In-fold minus held-out")
    b.set_title("(b) Overfitting to 308 items", loc="left", pad=5)
    fig.tight_layout(pad=0.3, w_pad=2.0)
    _save(fig, "decision_rules")


def negative_results() -> None:
    """Each structural or loss variant against its own named control (docs/02_experiment_log.md E4-E12, Report 4)."""
    rows = [  # label, change in dev S2, comparison level
        ("Per-class weights (E4)", -0.044, "system"),
        ("Hard hierarchical routing (E5)", +0.004, "ensemble"),
        ("Definition re-ranker (E6)", -0.011, "ensemble"),
        ("Full q. 8 ep. + Bal. Softmax + focal (E8)", -0.059, "system"),
        ("Hierarchical output head (E9)", -0.068, "system"),
        ("Non-Reply gate (E12b)", -0.059, "per model, 3 seeds"),
        ("Branch specialists (E12b)", -0.009, "per model, 5 seeds"),
        ("Boundary pair experts (E12c)", +0.002, "per model"),
        ("Uniform weight soup (E12d)", -0.117, "vs. single model"),
        ("Mixed-input ensemble (E11)", -0.002, "system, 10 seeds"),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], height=0.6, color=DEB_C, edgecolor="black", linewidth=0.8, zorder=2)
    for yi, (_, d, lvl) in zip(y, rows):
        ax.text(0.058, yi, f"{d:+.3f}", ha="right", va="center", fontsize=10.5)
        ax.text(0.066, yi, lvl, ha="left", va="center", fontsize=10, color="0.35")
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(0.015, color="black", lw=0.9, ls=(0, (5, 2)))
    ax.text(0.018, -0.95, "screening bar +0.015", fontsize=10, va="center")
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_ylim(len(rows) - 0.4, -1.4)
    ax.set_xlim(-0.13, 0.16)
    ax.set_xticks([-0.10, -0.05, 0.0, 0.05])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Change in dev Subtask 2 macro-F1 vs. its named control")
    fig.tight_layout(pad=0.3)
    _save(fig, "negative_results")


def per_class_gain() -> None:
    """Report Table 4: per-class F1, single models averaged over the same 10 seeds."""
    classes = ["Explicit", "Implicit", "Dodging", "General", "Deflection", "Partial/half-answer",
               "Declining to answer", "Claims ignorance", "Clarification"]
    deb = [0.672, 0.403, 0.484, 0.263, 0.287, 0.000, 0.381, 0.296, 0.667]
    qwen = [0.729, 0.443, 0.587, 0.322, 0.377, 0.000, 0.601, 0.700, 0.524]
    delta = [0.057, 0.040, 0.103, 0.059, 0.089, 0.000, 0.220, 0.403, -0.144]  # as printed in the report
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    y = np.arange(len(classes))
    h = 0.38
    ax.barh(y - h / 2, deb, height=h, color=DEB_C, edgecolor="black", linewidth=0.8, label="DeBERTa-v3-large", zorder=2)
    ax.barh(y + h / 2, qwen, height=h, color=QWEN_C, edgecolor="black", linewidth=0.8, hatch="////",
            label="Qwen3-8B + LoRA", zorder=2)
    for yi, d, q, dl in zip(y, deb, qwen, delta):
        txt = "never predicted" if d == q == 0 else f"{dl:+.3f}"
        ax.text(max(d, q) + 0.02, yi, txt, va="center", fontsize=10.5,
                weight="bold" if dl >= 0.2 else "normal", color="0.35" if d == q == 0 else "black")
    for sep in (0.5, 5.5):  # clarity-level boundaries: Clear Reply | Ambivalent | Clear Non-Reply
        ax.axhline(sep, color="0.6", lw=0.6, ls=(0, (2, 2)))
    for yy, lvl in ((0, "Clear Reply"), (3, "Ambivalent"), (7, "Clear Non-Reply")):
        ax.text(0.99, yy, lvl, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=10,
                style="italic", color="0.4")
    ax.set_yticks(y, classes)
    ax.set_ylim(len(classes) - 0.45, -0.55)
    ax.set_xlim(0, 1.12)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Per-class F1, dev Subtask 2 (mean of 10 seeds)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig.tight_layout(pad=0.3)
    _save(fig, "per_class_gain")


def confusion() -> None:
    """Final DeBERTa 10-seed ensemble, argmax, rows = majority gold (docs/raw/mideval/confusion_final.csv)."""
    rows = np.genfromtxt(ROOT / "docs/raw/mideval/confusion_final.csv", delimiter=",", skip_header=1,
                         usecols=range(1, 10), dtype=int)
    short = ["Explicit", "Implicit", "Dodging", "General", "Deflection", "Partial", "Declining", "Claims ign.", "Clarif."]
    fig, ax = plt.subplots(figsize=(5.6, 4.7))
    ax.imshow(rows, cmap="Greys", vmin=0, vmax=rows.max() * 1.25, aspect="auto")
    for i in range(rows.shape[0]):
        for j in range(rows.shape[1]):
            if rows[i, j]:
                ax.text(j, i, rows[i, j], ha="center", va="center", fontsize=10.5,
                        color="white" if rows[i, j] > 0.5 * rows.max() else "black",
                        weight="bold" if i == j else "normal")
    ax.set_xticks(range(9), short, rotation=45, ha="right")
    ax.set_yticks(range(10), short + ["(no majority)"])
    ax.tick_params(length=0, right=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Majority gold label")
    fig.tight_layout(pad=0.3)
    _save(fig, "confusion")


def ranking_vs_decision() -> None:
    """Qwen seeds 0-2: top-k coverage and in-set rate by confidence (docs/raw/E13_Q8_topk_confidence.txt)."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.4))
    cov = [0.604, 0.831, 0.931, 0.991]
    a.bar(range(4), cov, width=0.6, color=QWEN_C, hatch="////", edgecolor="black", linewidth=0.8, zorder=2)
    for i, v in enumerate(cov):
        a.text(i, v + 0.015, f"{100 * v:.1f}%", ha="center", va="bottom", fontsize=10.5,
               bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    a.set_xticks(range(4), ["top 1", "top 2", "top 3", "top 5"])
    a.set_ylim(0, 1.12)
    a.set_yticks(np.arange(0, 1.01, 0.25))
    a.set_ylabel("Acceptable label within top $k$")
    a.set_title("(a) Ranking", loc="left", pad=5)

    fifths = [0.40, 0.48, 0.56, 0.70, 0.87]
    b.plot(range(5), fifths, color="black", lw=1.1, marker="s", ms=6, zorder=3)
    for i, v in enumerate(fifths):
        b.annotate(f"{v:.2f}", (i, v), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=10.5)
    b.set_xticks(range(5), ["least", "2", "3", "4", "most"])
    b.set_xlim(-0.4, 4.4)
    b.set_xlabel("Confidence fifth of dev")
    b.set_ylim(0.3, 1.0)
    b.set_ylabel("In-set rate")
    b.set_title("(b) Confidence", loc="left", pad=5)
    fig.tight_layout(pad=0.3, w_pad=2.0)
    _save(fig, "ranking_vs_decision")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.png"):
        old.unlink()
    deberta_seeds()
    qwen_vs_deberta()
    undertraining()
    decision_rules()
    negative_results()
    per_class_gain()
    confusion()
    ranking_vs_decision()


if __name__ == "__main__":
    main()
