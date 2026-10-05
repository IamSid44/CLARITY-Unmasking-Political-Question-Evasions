#!/usr/bin/env python3
"""Data figures for the report, in a conventional publication style."""

from __future__ import annotations


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from decision_rules.decide import make_rules, nested_cv
from qevasion.labels import N_EVASION, encode_clarity, encode_evasion, leaf_to_official_clarity
from qevasion.loader import dev_reference_mask, load_qevasion
from qevasion.scoring import score_subtask1, score_subtask2
from qevasion.paths import REPORT_FIGURES as OUT

from qevasion.paths import RUNS

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.8, "axes.linewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.top": False, "ytick.right": True, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})
SEED_C, MEAN_C, DEB_C, QWEN_C = "0.70", "black", "0.80", "0.45"


def load(runs: list[str]) -> dict[int, np.ndarray]:
    out = {}
    for r in runs:
        for f in sorted((RUNS / r).glob("seed*/dev_probs.npy")):
            out[int(f.parent.name.removeprefix("seed"))] = np.load(f)
    return out


def paired_panel(ax, scores: list[dict], labels: list[str], seeds: list[int], panel: str, ylabel: str | None):
    """One open marker per seed, thin lines joining the same seed; filled square = mean, bar = +/- 1 s.d."""
    x = np.arange(len(scores))
    for s in seeds:
        ax.plot(x, [sc[s] for sc in scores], color=SEED_C, lw=0.5, zorder=1)
        ax.plot(x, [sc[s] for sc in scores], ls="none", marker="o", ms=2.6, mfc="white", mec=SEED_C, mew=0.6, zorder=2)
    mean = np.array([np.mean([sc[s] for s in seeds]) for sc in scores])
    sd = np.array([np.std([sc[s] for s in seeds], ddof=1) for sc in scores])
    ax.errorbar(x, mean, yerr=sd, fmt="s", ms=3.6, color=MEAN_C, ecolor=MEAN_C, elinewidth=0.8, capsize=2.2,
                capthick=0.8, zorder=3)
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.4, len(scores) - 0.6)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(panel, loc="left", pad=3)
    return mean, sd


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sp = load_qevasion()
    gold = dev_reference_mask(sp.dev)
    clar = encode_clarity(sp.dev["clarity_label"].tolist())
    y = encode_evasion(sp.train["evasion_label"].tolist())
    r1 = make_rules(np.bincount(y, minlength=N_EVASION) / len(y))["R1_logit_adjusted"]
    seed_legend = [Line2D([], [], color=SEED_C, lw=0.5, marker="o", ms=2.6, mfc="white", mec=SEED_C, label="one seed (paired)"),
                   Line2D([], [], color=MEAN_C, ls="none", marker="s", ms=3.6, label="mean $\\pm$ 1 s.d.")]

    cfgs = [["L0_large_base", "E11_base_8ep"], ["E10_base_16ep", "E11_base_16ep"], ["E10_fullq_16ep", "E11_fullq_16ep"]]
    probs = [load(c) for c in cfgs]
    seeds = sorted(set.intersection(*(set(p) for p in probs)))
    s2 = [{s: score_subtask2(p[s].argmax(1), gold).macro_f1 for s in seeds} for p in probs]
    s1 = [{s: score_subtask1(leaf_to_official_clarity(p[s].argmax(1)), clar).macro_f1 for s in seeds} for p in probs]
    labels = ["Baseline\n8 ep.", "Baseline\n16 ep.", "+ Full q.\n16 ep."]
    fig, axes = plt.subplots(1, 2, figsize=(3.15, 2.05))
    m2, d2 = paired_panel(axes[0], s2, labels, seeds, "(a) Subtask 2", "Dev macro-F1")
    m1, d1 = paired_panel(axes[1], s1, labels, seeds, "(b) Subtask 1", None)
    fig.legend(handles=seed_legend, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01),
               handlelength=1.6, columnspacing=1.2)
    fig.tight_layout(pad=0.2, w_pad=0.6, rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "fig_deberta_seeds.pdf")
    plt.close(fig)
    print("fig 1:", len(seeds), "seeds; S2", [f"{m:.3f}+/-{d:.3f}" for m, d in zip(m2, d2)],
          "S1", [f"{m:.3f}+/-{d:.3f}" for m, d in zip(m1, d1)])

    cfgs = [("DeBERTa\nbaseline", ["L0_large_base", "E11_base_8ep"]),
            ("DeBERTa\nfull q.", ["E10_fullq_16ep", "E11_fullq_16ep"]),
            ("Qwen-8B\nLoRA", ["Q8_fullq_lora"]),
            ("Qwen-8B\nall data", ["Q8_alldata"])]
    probs = [load(r) for _, r in cfgs]
    seeds = sorted(set.intersection(*(set(p) for p in probs)))
    s2 = [{s: score_subtask2(p[s].argmax(1), gold).macro_f1 for s in seeds} for p in probs]
    systems = [np.mean([nested_cv(r1, np.mean([p[s] for s in sorted(p)], 0), gold, 5, cs)["out_of_fold_macro_f1"]
                        for cs in range(10)]) for p in probs]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.3, 2.35), gridspec_kw={"width_ratios": [1, 1]})
    mq, dq = paired_panel(a1, s2, [c for c, _ in cfgs], seeds, "(a) Single models, 10 paired seeds", "Dev Subtask 2 macro-F1")
    a1.legend(handles=seed_legend, loc="upper left", frameon=False, handlelength=1.6)
    a1.set_ylim(0.25, 0.60)

    x = np.arange(len(cfgs))
    a2.bar(x, systems, width=0.6, color=[DEB_C, DEB_C, QWEN_C, QWEN_C], edgecolor="black", linewidth=0.6,
           hatch=["", "", "////", "////"], zorder=2)
    for xi, v in zip(x, systems):
        a2.text(xi, v + 0.014, f"{v:.3f}", ha="center", va="bottom", fontsize=6.8, zorder=4,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    refs = [("Human vs. other two", 0.684, (0, (1, 1.2))),
            ("TeleAI pipeline", 0.617, (0, (5, 2))),
            ("ChulaNLP hybrid", 0.520, (0, (5, 1.5, 1, 1.5))),
            ("TeleAI Qwen2.5-7B FT", 0.495, "-")]
    handles = []
    for lab, v, ls in refs:
        a2.axhline(v, color="black", lw=0.7, ls=ls, zorder=3)
        handles.append(Line2D([], [], color="black", lw=0.7, ls=ls, label=f"{lab} ({v:.3f})"))
    a2.legend(handles=handles, loc="upper center", ncol=2, frameon=True, framealpha=1, edgecolor="0.6",
              fancybox=False, handlelength=2.4, columnspacing=1.0, borderpad=0.4, fontsize=6.3)
    a2.set_xticks(x, [c for c, _ in cfgs])
    a2.set_xlim(-0.55, len(cfgs) - 0.45)
    a2.set_ylim(0, 1.06)
    a2.set_yticks(np.arange(0, 0.81, 0.2))
    a2.set_ylabel("Dev Subtask 2 macro-F1")
    a2.set_title("(b) Systems: 10-seed ensemble + logit adjustment", loc="left", pad=3)
    fig.tight_layout(pad=0.2, w_pad=1.4)
    fig.savefig(OUT / "fig_qwen_vs_deberta.pdf")
    plt.close(fig)
    print("fig 2:", len(seeds), "seeds; single S2", [f"{m:.3f}+/-{d:.3f}" for m, d in zip(mq, dq)],
          "systems", [f"{v:.3f}" for v in systems])


if __name__ == "__main__":
    main()
