#!/usr/bin/env python3
"""Result figures of the final report, in the style of figures/paper_figures.py (grayscale, serif, inward ticks).

    python -m figures.report_final_figures      # -> report/final/figures/fig_*.pdf

Each figure replaces a results table of the report. Values come from the raw analysis outputs named in each
function's docstring; the frontier and the costs are read from the run records (CPU, seconds).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from preregistered_analyses.cost_same_gpu import m3_dev_cost  # noqa: E402
from qevasion.paths import ROOT, RUNS  # noqa: E402

OUT = ROOT / "report" / "final" / "figures"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
    "legend.fontsize": 6.5, "axes.linewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "legend.frameon": False,
})
INK, MID, LIGHT, PALE = "black", "0.45", "0.72", "0.88"
COL, FULL = 3.15, 6.45   # ACL column and text widths, inches


def title(ax, s: str) -> None:
    ax.set_title(s, loc="left", pad=3)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    if os.environ.get("FIG_PREVIEW"):
        fig.savefig(Path(os.environ["FIG_PREVIEW"]) / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=220)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def stage1() -> None:
    """Stage 1 (E0-E13). (a) every change against its named control (docs/experiment_log.md, Scoreboard; E11, E13);
    (b, c) the Qwen classifier ranks better than it decides (raw/E13_Q8_topk_confidence.txt, seeds 0-2)."""
    rows = [  # label, delta S2, kind: adopted / failed / backbone
        ("Logit adjustment vs argmax", 0.073, "adopted"),
        ("16 epochs instead of 8", 0.047, "adopted"),
        ("+ the full journalist question", 0.022, "adopted"),
        ("Pair experts (3 confused pairs)", 0.002, "failed"),
        ("Branch specialists", -0.009, "failed"),
        ("Definition re-ranker", -0.011, "failed"),
        ("Nine per-class weights", -0.044, "failed"),
        ("Balanced Softmax + focal loss", -0.059, "failed"),
        ("Non-Reply gate", -0.059, "failed"),
        ("Hierarchical output head", -0.068, "failed"),
        ("Weight soup of 10 models", -0.117, "failed"),
        ("Backbone: Qwen3-8B + LoRA", 0.092, "backbone"),
    ]
    fig = plt.figure(figsize=(FULL, 2.05))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], hspace=0.75, wspace=0.62)
    a = fig.add_subplot(gs[:, 0])
    y = np.arange(len(rows))[::-1].astype(float)
    y[-1] -= 0.0
    y[:-1] += 0.6  # a gap above the backbone row
    fill = {"adopted": INK, "failed": "white", "backbone": MID}
    for yi, (lab, d, k) in zip(y, rows):
        a.barh(yi, d, height=0.62, color=fill[k], edgecolor=INK, linewidth=0.6)
        a.text(d + (0.004 if d >= 0 else -0.004), yi, f"{d:+.3f}".replace("-", "\u2212"), va="center",
               ha="left" if d >= 0 else "right",
               fontsize=6.2)
    a.axvline(0, color=INK, lw=0.6)
    a.set_yticks(y, [r[0] for r in rows])
    a.tick_params(axis="y", length=0)
    a.set_xlim(-0.16, 0.13)
    a.set_xlabel(r"$\Delta$ dev S2 against the named control")
    title(a, "(a) One change at a time (DeBERTa-v3-large unless stated)")
    a.legend(handles=[Patch(fc=INK, ec=INK, label="adopted"), Patch(fc="white", ec=INK, label="failed"),
                      Patch(fc=MID, ec=INK, label="backbone")],
             loc="upper left", handlelength=1.0, borderaxespad=0.2)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)

    b = fig.add_subplot(gs[0, 1])
    ks, cov = ["1", "2", "3", "5"], [0.604, 0.831, 0.931, 0.991]
    b.bar(ks, cov, width=0.6, color=[INK, LIGHT, LIGHT, LIGHT], edgecolor=INK, linewidth=0.6)
    for x, v in zip(ks, cov):
        b.text(x, v + 0.03, f"{v:.2f}", ha="center", fontsize=6.2)
    b.set_ylim(0, 1.15)
    b.set_yticks([0, 0.5, 1])
    b.set_xlabel("$k$")
    b.set_ylabel("in top $k$")
    title(b, "(b) An acceptable label is in the top $k$")
    c = fig.add_subplot(gs[1, 1])
    fifths, ins = np.arange(1, 6), [0.40, 0.48, 0.56, 0.70, 0.87]
    c.plot(fifths, ins, color=INK, lw=1.0, marker="o", ms=3.2, mfc="white", mew=0.8)
    c.text(fifths[0] + 0.15, ins[0] - 0.02, f"{ins[0]:.2f}", ha="left", va="top", fontsize=6.2)
    c.text(fifths[-1], ins[-1] + 0.06, f"{ins[-1]:.2f}", ha="center", fontsize=6.2)
    c.set_ylim(0.2, 1.05)
    c.set_xticks(fifths, ["least", "2", "3", "4", "most"])
    c.set_xlabel("confidence fifth of dev")
    c.set_ylabel("in-set rate")
    title(c, "(c) Confidence tracks correctness")
    for ax in (b, c):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    save(fig, "fig_stage1")


def frontier() -> None:
    """Stage 2 (E15-E18): dev S2 against inference GPU-s per item for the deferral sweep, on one GPU
    (runs/M3/frontier_*.json for S2 and realised calls, first 27B copy, one option order; unit costs as in
    preregistered_analyses.cost_same_gpu)."""
    c1 = json.loads((RUNS / "E22" / "cost_jarvis.json").read_text())["m1_seed_gpu_s_per_item"]
    c3 = m3_dev_cost("jarvis")[0]
    u = json.loads((RUNS / "M3" / "frontier_Q8_fullq_lora_12ep.json").read_text())["table"]
    s = json.loads((RUNS / "M3" / "frontier_Q8_distill_a05.json").read_text())["table"]
    series = [("one M1 seed + 27B", [r for r in u if r["K"] == 1], dict(color=LIGHT, marker="o", mfc="white")),
              ("distilled student + 27B", s, dict(color=INK, marker="s", mfc=INK)),
              ("M1, 10 seeds + 27B", [r for r in u if r["K"] == 10], dict(color=MID, marker="o", mfc=MID))]
    fig, ax = plt.subplots(figsize=(COL, 1.8))
    ax.axhline(0.617, color=INK, lw=0.6, ls=(0, (4, 2)))
    ax.text(0.155, 0.6195, "TeleAI, 1st place", fontsize=6.0, va="bottom")
    for name, rows, kw in series:
        rows = sorted(rows, key=lambda r: r["delta"])
        x = [r["K"] * c1 + r["calls"] * c3 for r in rows]
        yv = [r["s2"] for r in rows]
        ax.plot(x, yv, lw=0.9, ms=3.4, mew=0.7, mec=kw["color"], label=name, **kw)
        if rows[0]["K"] == 1:
            ax.annotate(r"$\delta{=}0$", (x[0], yv[0]), xytext=(-3, -2), textcoords="offset points", fontsize=5.8,
                        ha="right")
            ax.annotate(r"$\delta{=}1$", (x[-1], yv[-1]), xytext=(4, -2), textcoords="offset points", fontsize=5.8)
    star = [(1 * c1 + next(r for r in s if r["delta"] == 0.5)["calls"] * c3, 0.608, "distilled point", (0, -6)),
            (10 * c1 + next(r for r in u if r["K"] == 10 and r["delta"] == 0.3)["calls"] * c3, 0.669,
             "accuracy point", (-6, 1))]
    for x, yv, lab, off in star:
        ax.plot(x, yv, marker="*", ms=8, color=INK, mec="white", mew=0.4, zorder=5)
        ax.annotate(lab, (x, yv), xytext=off, textcoords="offset points", fontsize=6.0,
                    ha="center" if off[0] == 0 else "right", va="top" if off[1] < 0 else "bottom")
    ax.set_xscale("log")
    ax.set_xlim(0.03, 0.75)
    ax.set_xticks([0.04, 0.1, 0.2, 0.4], ["0.04", "0.1", "0.2", "0.4"])
    ax.set_ylim(0.52, 0.69)
    ax.set_xlabel("inference GPU-s per item (log scale, one GPU)")
    ax.set_ylabel("dev S2 macro-F1")
    ax.legend(loc="lower right", handlelength=2.0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    save(fig, "fig_frontier")


def stage3() -> None:
    """Stage 3 (E19-E22). (a) gain over M1 (raw/E19_tier0.txt A.2, C); (b) routing at matched budgets (E.1);
    (c) net fixes of the 27B by M1-uncertainty fifth on leak-free singleton train rows (raw/E20_router.txt S1);
    (d) letter-A rate by option order on the 300 screen rows (raw/E21_E22_analysis.txt B)."""
    fig, axs = plt.subplots(1, 4, figsize=(FULL, 1.62), gridspec_kw={"width_ratios": [1.25, 1, 1, 1]})
    a, b, c, d = axs
    rows = [("dev, parameters held", 0.067, 0.009, 0.124), ("dev, parameters refit", 0.037, -0.044, 0.120),
            ("train rows, refit", 0.014, 0.002, 0.026)]
    for i, (lab, m, lo, hi) in enumerate(rows):
        yy = len(rows) - 1 - i
        a.plot([lo, hi], [yy, yy], color=INK, lw=0.9)
        a.plot(m, yy, "o", ms=3.6, color=INK if i != 1 else "white", mec=INK, mew=0.8)
    a.axvline(0, color=INK, lw=0.6, ls=":")
    a.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    a.tick_params(axis="y", length=0)
    a.set_ylim(-0.6, 2.6)
    a.set_xlabel("gain over M1 (95% CI)")
    title(a, "(a) Is the gain real?")

    dl = [0.1, 0.2, 0.3, 0.5]
    conf, rmean, rsd = [0.597, 0.635, 0.669, 0.628], [0.599, 0.601, 0.604, 0.608], [0.009, 0.014, 0.015, 0.018]
    b.fill_between(dl, np.subtract(rmean, 2 * np.array(rsd)), np.add(rmean, 2 * np.array(rsd)), color=PALE, lw=0)
    b.plot(dl, rmean, color=MID, lw=0.9, ls="--", label="random (±2 sd)")
    b.plot(dl, conf, color=INK, lw=1.0, marker="o", ms=3.2, label="confidence")
    b.axhline(0.598, color=LIGHT, lw=0.6)
    b.text(0.5, 0.592, "M1 alone", fontsize=5.8, ha="right", va="top", color=MID)
    b.set_xticks(dl, ["0.1", "0.2", "0.3", "0.5"])
    b.set_ylim(0.57, 0.725)
    b.set_xlabel(r"deferral rate $\delta$")
    b.set_ylabel("dev S2")
    b.legend(loc="upper left", fontsize=5.8, handlelength=1.6, borderaxespad=0.1, labelspacing=0.25)
    b.set_yticks([0.6, 0.65, 0.7])
    title(b, "(b) Which items to defer?")

    net, lo, hi = [0, -7, 2, 7, 12], [-7, -20, -14, -11, -5], [7, 5, 19, 24, 28]
    xs = np.arange(1, 6)
    c.bar(xs, net, width=0.6, color=[LIGHT if v <= 0 else INK for v in net], edgecolor=INK, linewidth=0.6)
    c.errorbar(xs, net, yerr=[np.subtract(net, lo), np.subtract(hi, net)], fmt="none", ecolor=MID, elinewidth=0.6,
               capsize=1.5)
    c.axhline(0, color=INK, lw=0.6)
    c.set_xticks(xs, ["most", "2", "3", "4", "least"])
    c.set_xlabel("M1 confidence fifth")
    c.set_ylabel("27B fixes $-$ breaks")
    title(c, "(c) Where does the 27B help?")

    orders, pa = ["tax.", "+1", "+2", "+3", "rev."], [0.400, 0.293, 0.247, 0.250, 0.220]
    d.bar(orders, pa, width=0.6, color=[INK] + [LIGHT] * 4, edgecolor=INK, linewidth=0.6)
    d.axhline(0.25, color=INK, lw=0.6, ls=":")
    d.text(4.4, 0.26, "chance", fontsize=5.8, ha="right", va="bottom")
    d.set_ylim(0, 0.46)
    d.set_ylabel("picks letter A")
    d.set_xlabel("option order")
    title(d, "(d) Option-order bias")
    for ax in axs:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout(w_pad=0.9)
    save(fig, "fig_stage3")


def relation() -> None:
    """Stage 4 (E24). (a) P(two annotators' labels differ) by pair type (raw/E24_prechecks.txt P2);
    (b) answer identifiability under its own question by gold label, train (P8)."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL, 1.48), gridspec_kw={"width_ratios": [1, 1.3]})
    lab, v = ["same\nitem", "sibling", "random"], [0.432, 0.716, 0.834]
    a.bar(lab, v, width=0.6, color=[LIGHT, INK, LIGHT], edgecolor=INK, linewidth=0.6)
    for x, val in zip(lab, v):
        a.text(x, val + 0.03, f"{val:.2f}", ha="center", fontsize=6.2)
    a.set_ylim(0, 1.0)
    a.set_ylabel("labels differ")
    a.tick_params(axis="x", labelsize=6.2)
    a.set_xlabel("pair of labels from")
    title(a, "(a) Labels differ")
    ident = [("Explicit", 0.817, "CR"), ("Partial", 0.835, "A"), ("Implicit", 0.827, "A"), ("Deflection", 0.798, "A"),
             ("General", 0.791, "A"), ("Dodging", 0.676, "A"), ("Claims ign.", 0.768, "N"), ("Declining", 0.634, "N"),
             ("Clarification", 0.594, "N")]
    fc = {"CR": INK, "A": LIGHT, "N": "white"}
    yy = np.arange(len(ident))[::-1]
    for y, (n, val, k) in zip(yy, ident):
        b.barh(y, val, height=0.62, color=fc[k], edgecolor=INK, linewidth=0.6)
    b.set_yticks(yy, [r[0] for r in ident])
    b.tick_params(axis="y", length=0, labelsize=6.2)
    b.set_xlim(0.5, 1.08)
    b.set_xticks([0.5, 0.7, 0.9])
    b.set_xlabel("fits own question best")
    b.legend(handles=[Patch(fc=INK, ec=INK, label="Clear Reply"), Patch(fc=LIGHT, ec=INK, label="Ambivalent"),
                      Patch(fc="white", ec=INK, label="Non-Reply")], fontsize=5.6, loc="lower right",
             handlelength=0.9, borderaxespad=0.0, bbox_to_anchor=(1.02, -0.03))
    title(b, "(b) Answer relevance")
    for ax in (a, b):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_relation")


def precision() -> None:
    """Stage 5 (E26): (a) S2 lost against 16-bit on the same GPU type, with the FROZEN parameters on both sides (so the
    estimators match); (b) FP8's flipped decisions that change reference-set membership. Numbers are recomputed from
    runs/E26 by preregistered_analyses.final_checks.precision()."""
    import contextlib
    import io

    from preregistered_analyses.final_checks import precision as flips
    with contextlib.redirect_stdout(io.StringIO()):
        R = flips()
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL, 1.5), gridspec_kw={"width_ratios": [1.2, 1]})
    x = np.arange(2)
    m1 = [R[k]["m1_s2"][1] - R[k]["m1_s2"][0] for k in ("fp8", "nf4")]
    sy = [R[k]["sys_s2"][1] - R[k]["sys_s2"][0] for k in ("fp8", "nf4")]
    a.bar(x - 0.17, m1, width=0.32, color=LIGHT, edgecolor=INK, linewidth=0.6, label="M1 alone")
    a.bar(x + 0.17, sy, width=0.32, color=INK, edgecolor=INK, linewidth=0.6, label="full instrument")
    for xi, v in zip(list(x - 0.17) + list(x + 0.17), m1 + sy):
        a.text(xi, v - 0.002, f"{v:.3f}".replace("-", "\u2212"), ha="center", va="top", fontsize=5.8)
    a.axhline(0, color=INK, lw=0.6)
    a.set_xticks(x, ["FP8", "NF4$^{\\dagger}$"])
    a.set_ylim(-0.062, 0.004)
    a.set_ylabel(r"$\Delta$ dev S2 vs 16-bit")
    a.legend(loc="lower right", fontsize=5.8, handlelength=1.0, borderaxespad=0.1)
    title(a, "(a) Accuracy lost (frozen)")
    f = R["fp8"]
    xs = np.arange(2)
    b.bar(xs - 0.17, [f["m1_lose"], f["sys_lose"]], width=0.32, color=INK, edgecolor=INK, linewidth=0.6, label="lose membership")
    b.bar(xs + 0.17, [f["m1_gain"], f["sys_gain"]], width=0.32, color="white", edgecolor=INK, linewidth=0.6, label="gain it")
    for xi, v in zip(list(xs - 0.17) + list(xs + 0.17), [f["m1_lose"], f["sys_lose"], f["m1_gain"], f["sys_gain"]]):
        b.text(xi, v + 0.3, str(v), ha="center", fontsize=6.0)
    b.set_xticks(xs, ["M1", "system"])
    b.set_ylim(0, 13.5)
    b.set_ylabel("flipped decisions")
    b.legend(loc="upper right", fontsize=5.8, handlelength=1.0, borderaxespad=0.1)
    title(b, "(b) FP8 flips that change membership")
    for ax in (a, b):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_precision")


def main() -> None:
    stage1()
    frontier()
    stage3()
    relation()
    precision()


if __name__ == "__main__":
    main()
