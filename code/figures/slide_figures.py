#!/usr/bin/env python3
"""Figures for the final presentation (report/slides_final/figures/s_*.pdf), sized for a 16:9 Beamer slide.

    python -m figures.slide_figures          # set FIG_PREVIEW=<dir> to also write PNG previews

Same style as the mid-evaluation deck (figures/deck_figures.py): STIX serif, boxed axes with inward ticks, greys for
everything, hatching or marker shape (never colour alone) for identity. The deck's navy (#182D56) marks the one
element each slide is about. Each figure is drawn at the width it occupies on its slide, so its type is not rescaled.
Numbers come from the raw analysis outputs and run records named in each function; the descriptive checks of
preregistered_analyses/final_checks.py feed the order, copy, role and precision figures.
"""

from __future__ import annotations

import contextlib
import io
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
from qevasion.loader import dev_reference_mask, load_qevasion  # noqa: E402
from qevasion.paths import ROOT, RUNS  # noqa: E402
from qevasion.scoring import score_subtask2  # noqa: E402

OUT = ROOT / "report" / "slides_final" / "figures"
NAVY, INK, MID, LIGHT, PALE = "#182D56", "black", "0.45", "0.72", "0.88"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.titlesize": 8.8, "axes.labelsize": 8.5, "xtick.labelsize": 7.8, "ytick.labelsize": 7.8,
    "legend.fontsize": 7.6, "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.major.size": 2.8, "ytick.major.size": 2.8,
    "xtick.top": False, "ytick.right": True, "pdf.fonttype": 42, "ps.fonttype": 42, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def title(ax, s: str) -> None:
    ax.set_title(s, loc="left", pad=3)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    if os.environ.get("FIG_PREVIEW"):
        fig.savefig(Path(os.environ["FIG_PREVIEW"]) / f"{name}.png", bbox_inches="tight", pad_inches=0.03, dpi=220)
    plt.close(fig)
    print("wrote", name)


def quiet(fn):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()


# ------------------------------------------------------------------------------------------------ Stage 1
def changes() -> None:
    """Each change against its named control (docs/experiment_log.md Scoreboard, E11, E13)."""
    rows = [("Logit adjustment", 0.073, "ok"), ("16 epochs instead of 8", 0.047, "ok"),
            ("+ full question", 0.022, "ok"), ("Pair experts", 0.002, "no"),
            ("Branch specialists", -0.009, "no"), ("Definition re-ranker", -0.011, "no"),
            ("9 per-class weights", -0.044, "no"), ("Rebalanced loss", -0.059, "no"),
            ("Non-Reply gate", -0.059, "no"), ("Hierarchical head", -0.068, "no"),
            ("Weight soup", -0.117, "no"), ("Backbone: Qwen3-8B", 0.092, "bb")]
    fig, ax = plt.subplots(figsize=(3.25, 2.55))
    y = np.arange(len(rows))[::-1].astype(float)
    y[:-1] += 0.7
    fc = {"ok": MID, "no": "white", "bb": NAVY}
    for yi, (lab, d, k) in zip(y, rows):
        ax.barh(yi, d, height=0.66, color=fc[k], edgecolor=NAVY if k == "bb" else INK, linewidth=0.6)
        ax.text(d + (0.004 if d >= 0 else -0.004), yi, f"{d:+.3f}".replace("-", "−"), va="center",
                ha="left" if d >= 0 else "right", fontsize=6.8, color=NAVY if k == "bb" else INK,
                fontweight="bold" if k == "bb" else "normal")
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_yticks(y, [r[0] for r in rows])
    for t, (_, _, k) in zip(ax.get_yticklabels(), rows):
        if k == "bb":
            t.set_color(NAVY)
            t.set_fontweight("bold")
    ax.tick_params(axis="y", length=0, right=False)
    ax.set_xlim(-0.185, 0.145)
    ax.set_xlabel(r"$\Delta$ dev S2 vs its control")
    ax.legend(handles=[Patch(fc=MID, ec=INK, lw=0.6, label="helped"), Patch(fc="white", ec=INK, lw=0.6, label="failed")],
              loc="upper left", handlelength=1.0, borderaxespad=0.1, fontsize=7)
    save(fig, "s_changes")


def backbone() -> None:
    """Single models on 10 paired seeds: DeBERTa (full question, 16 ep) -> Qwen3-8B LoRA 3 ep -> 12 ep."""
    gold = dev_reference_mask(load_qevasion().dev)

    def load(runs):
        out = {}
        for r in runs:
            for f in sorted((RUNS / r).glob("seed*/dev_probs.npy")):
                out[int(f.parent.name.removeprefix("seed"))] = np.load(f)
        return out
    cfgs = [("DeBERTa-v3\n(0.4B)", ["E10_fullq_16ep", "E11_fullq_16ep"]), ("Qwen3-8B\n3 epochs", ["Q8_fullq_lora"]),
            ("Qwen3-8B\n12 epochs", ["Q8_fullq_lora_12ep"])]
    probs = [load(r) for _, r in cfgs]
    seeds = sorted(set.intersection(*(set(p) for p in probs)))
    s2 = [{s: score_subtask2(p[s].argmax(1), gold).macro_f1 for s in seeds} for p in probs]
    fig, ax = plt.subplots(figsize=(2.75, 2.35))
    x = np.arange(len(cfgs))
    for s in seeds:
        ax.plot(x, [sc[s] for sc in s2], color=LIGHT, lw=0.7, zorder=1)
        ax.plot(x, [sc[s] for sc in s2], "o", ms=2.8, mfc="white", mec=MID, mew=0.6, zorder=2)
    mean = [np.mean([sc[s] for s in seeds]) for sc in s2]
    ax.plot(x, mean, color=NAVY, lw=1.6, marker="s", ms=5, zorder=3)
    for xi, m in zip(x, mean):
        ax.text(xi + 0.09, m - 0.012, f"{m:.3f}", color=NAVY, fontsize=8.2, fontweight="bold", va="center", ha="left")
    ax.set_xticks(x, [c for c, _ in cfgs])
    ax.set_xlim(-0.3, 2.6)
    ax.set_ylabel("dev S2, one model per seed")
    ax.legend(handles=[Line2D([], [], color=LIGHT, marker="o", mfc="white", mec=MID, ms=3, lw=0.7, label="one seed"),
                       Line2D([], [], color=NAVY, marker="s", ms=4.5, lw=1.6, label="mean of 10")],
              loc="lower right", handlelength=1.6, fontsize=7.2)
    save(fig, "s_backbone")


def ranking() -> None:
    """raw/E13_Q8_topk_confidence.txt, seeds 0-2."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.3, 1.7), gridspec_kw={"wspace": 0.3})
    ks, cov = ["top 1", "top 2", "top 3", "top 5"], [0.604, 0.831, 0.931, 0.991]
    a.bar(ks, cov, width=0.62, color=[MID, LIGHT, NAVY, LIGHT], edgecolor=INK, linewidth=0.5)
    for x, v in zip(ks, cov):
        a.text(x, v + 0.03, f"{v:.0%}", ha="center", fontsize=8, color=NAVY if x == "top 3" else INK,
               fontweight="bold" if x == "top 3" else "normal")
    a.set_ylim(0, 1.15)
    a.set_yticks([0, 0.5, 1], ["0", "50%", "100%"])
    title(a, "An acceptable label is in the …")
    fifths, ins = np.arange(1, 6), [0.40, 0.48, 0.56, 0.70, 0.87]
    b.plot(fifths, ins, color=NAVY, lw=1.6, marker="o", ms=4)
    b.text(1.12, 0.40 - 0.03, "40%", ha="left", va="top", fontsize=8)
    b.text(5, 0.87 + 0.05, "87%", ha="center", fontsize=8, color=NAVY, fontweight="bold")
    b.set_ylim(0.2, 1.05)
    b.set_xticks(fifths, ["least", "2", "3", "4", "most"])
    b.set_xlabel("confidence fifth of dev")
    b.set_yticks([0.25, 0.5, 0.75, 1.0], ["25%", "50%", "75%", "100%"])
    title(b, "Correct, by confidence")
    save(fig, "s_ranking")


# ------------------------------------------------------------------------------------------------ Stage 2
def frontier() -> None:
    """runs/M3/frontier_*.json (first 27B copy, one option order), unit costs on one GPU (cost_same_gpu)."""
    c1 = json.loads((RUNS / "E22" / "cost_jarvis.json").read_text())["m1_seed_gpu_s_per_item"]
    c3 = m3_dev_cost("jarvis")[0]
    u = json.loads((RUNS / "M3" / "frontier_Q8_fullq_lora_12ep.json").read_text())["table"]
    s = json.loads((RUNS / "M3" / "frontier_Q8_distill_a05.json").read_text())["table"]
    fig, ax = plt.subplots(figsize=(3.1, 2.4))
    ax.axhline(0.617, color=MID, lw=0.7, ls=(0, (4, 2)))
    ax.text(0.15, 0.6195, "TeleAI, 1st place", fontsize=7.4, va="bottom", color=MID)
    series = [("1 seed + 27B", [r for r in u if r["K"] == 1], dict(color=LIGHT, marker="o", mfc="white", mec=MID)),
              ("student + 27B", s, dict(color=MID, marker="s", mfc=MID, mec=INK)),
              ("10 seeds + 27B", [r for r in u if r["K"] == 10], dict(color=NAVY, marker="o", mfc=NAVY, mec=NAVY))]
    for name, rows, kw in series:
        rows = sorted(rows, key=lambda r: r["delta"])
        x = [r["K"] * c1 + r["calls"] * c3 for r in rows]
        yv = [r["s2"] for r in rows]
        ax.plot(x, yv, lw=1.1, ms=3.6, mew=0.7, label=name, **kw)
        if rows[0]["K"] == 10:
            ax.annotate("no LLM", (x[0], yv[0]), xytext=(-4, -2), textcoords="offset points", fontsize=7, ha="right",
                        va="center")
            ax.annotate("defer all", (x[-1], yv[-1]), xytext=(4, 0), textcoords="offset points", fontsize=7, va="center")
    xs = 10 * c1 + next(r for r in u if r["K"] == 10 and r["delta"] == 0.3)["calls"] * c3
    ax.plot(xs, 0.669, marker="*", ms=11, color=NAVY, mec="white", mew=0.4, zorder=5)
    ax.annotate("defer 30%:\n0.669", (xs, 0.669), xytext=(-8, 0), textcoords="offset points", ha="right", va="center",
                fontsize=7.8, color=NAVY, fontweight="bold")
    xd = c1 + next(r for r in s if r["delta"] == 0.5)["calls"] * c3
    ax.plot(xd, 0.608, marker="*", ms=11, color=MID, mec="white", mew=0.4, zorder=5)
    ax.annotate("student: 0.608 at\n¼ of M1's GPU time", (xd, 0.608), xytext=(0.037, 0.655), textcoords="data",
                ha="left", va="center", fontsize=7.2,
                arrowprops=dict(arrowstyle="-", color=MID, lw=0.5, shrinkB=4))
    ax.set_xscale("log")
    ax.set_xlim(0.033, 0.8)
    ax.set_xticks([0.04, 0.1, 0.2, 0.4], ["0.04", "0.1", "0.2", "0.4"])
    ax.set_ylim(0.52, 0.685)
    ax.set_xlabel("GPU-s per item (log scale)")
    ax.set_ylabel("dev S2")
    ax.legend(loc="lower right", handlelength=1.8, fontsize=7.2)
    save(fig, "s_frontier")


# ------------------------------------------------------------------------------------------------ Stage 3
def robust() -> None:
    """(a) gain over M1, held vs refitted vs train rows (raw/E19_tier0.txt A.2, C); (b) confidence vs random routing (E.1)."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.4, 1.75), gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.4})
    rows = [("dev, parameters held", 0.067, 0.009, 0.124), ("dev, refitted", 0.037, -0.044, 0.120),
            ("train rows, refitted", 0.014, 0.002, 0.026)]
    for i, (lab, m, lo, hi) in enumerate(rows):
        yy = 2 - i
        col = NAVY if i == 1 else MID
        a.plot([lo, hi], [yy, yy], color=col, lw=1.6)
        a.plot(m, yy, "o", ms=5, color=col if i != 1 else "white", mec=col, mew=1.2)
        a.text(hi + 0.006, yy, f"{m:+.3f}", va="center", fontsize=7.6, color=col, fontweight="bold" if i == 1 else "normal")
    a.axvline(0, color=INK, lw=0.7, ls=":")
    a.set_yticks([2, 1, 0], [r[0] for r in rows])
    a.tick_params(axis="y", length=0, right=False)
    a.set_ylim(-0.6, 2.6)
    a.set_xlim(-0.06, 0.17)
    a.set_xticks([0, 0.05, 0.1])
    a.set_xlabel("gain over M1 (95% CI)")
    title(a, "Is the gain real?")
    dl = [0.1, 0.2, 0.3, 0.5]
    conf, rmean, rsd = [0.597, 0.635, 0.669, 0.628], [0.599, 0.601, 0.604, 0.608], np.array([0.009, 0.014, 0.015, 0.018])
    b.fill_between(dl, np.array(rmean) - 2 * rsd, np.array(rmean) + 2 * rsd, color=PALE, lw=0)
    b.plot(dl, rmean, color=MID, lw=1.0, ls="--")
    b.plot(dl, conf, color=NAVY, lw=1.6, marker="o", ms=4)
    b.text(0.3, 0.675, "confidence routing", color=NAVY, fontsize=7.6, ha="center", va="bottom", fontweight="bold")
    b.text(0.5, 0.583, "200 random routings", color=MID, fontsize=7.2, ha="right", va="top")
    b.set_xticks(dl, ["10%", "20%", "30%", "50%"])
    b.set_ylim(0.56, 0.70)
    b.set_xlabel("share of items deferred")
    b.set_ylabel("dev S2")
    title(b, "Which items to defer?")
    save(fig, "s_robust")


def nondet() -> None:
    """Stage 3: the 27B is not a fixed function of (weights, item). (a) agreement between presentation orders of the same
    four candidates; (b) agreement between two copies of the same weights, by how decisive the lab copy was (all from
    preregistered_analyses.final_checks, sections A and B)."""
    from preregistered_analyses.final_checks import copies, order
    O, C = quiet(order), quiet(copies)
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.4, 1.85), gridspec_kw={"width_ratios": [1, 1.05], "wspace": 0.34})
    cats = ["two orders:\nsame pick", "all four orders\nagree"]
    dev = [O["dev"]["mean_pair"], O["dev"]["all4"]]
    scr = [O["screen"]["mean_pair"], O["screen"]["all4"]]
    x = np.arange(2)
    a.bar(x - 0.18, dev, width=0.34, color=NAVY, edgecolor=INK, linewidth=0.5, label="dev")
    a.bar(x + 0.18, scr, width=0.34, color="white", edgecolor=INK, linewidth=0.6, label="300 train rows")
    for xi, v in zip(list(x - 0.18) + list(x + 0.18), dev + scr):
        a.text(xi, v + 0.025, f"{v:.0%}", ha="center", fontsize=7.8)
    a.axhline(1.0, color=INK, lw=0.6, ls=":")
    a.set_xticks(x, cats)
    a.set_ylim(0, 1.38)
    a.set_yticks([0, 0.5, 1], ["0", "50%", "100%"])
    a.legend(loc="upper right", fontsize=7, handlelength=1.0, borderaxespad=0.0, ncol=2, columnspacing=0.8)
    title(a, "(a) Same weights, other option order")
    t = C["terciles"]
    lab = ["least\ndecisive", "middle", "most\ndecisive"]
    vals = [v[2] for v in t]
    b.bar(range(3), vals, width=0.6, color=[NAVY, LIGHT, LIGHT], edgecolor=INK, linewidth=0.5)
    for xi, v in zip(range(3), vals):
        b.text(xi, v + 0.025, f"{v:.0%}", ha="center", fontsize=7.8, color=NAVY if xi == 0 else INK,
               fontweight="bold" if xi == 0 else "normal")
    b.axhline(C["agree"], color=MID, lw=0.7, ls="--")
    b.text(-0.45, C["agree"] + 0.025, f"all items: {C['agree']:.1%}", ha="left", va="bottom", fontsize=7, color=MID)
    b.set_xticks(range(3), lab)
    b.set_ylim(0, 1.18)
    b.set_xlim(-0.5, 2.5)
    b.set_yticks([0, 0.5, 1], ["0", "50%", "100%"])
    b.set_xlabel("gap between the top two options", fontsize=7.6)
    title(b, "(b) Same weights, other server")
    save(fig, "s_nondet")


def robust_backup() -> None:
    """Backup: net fixes by M1 confidence (raw/E20_router.txt S1) and option-A picks by order (raw/E21_E22_analysis.txt B)."""
    fig, (c, d) = plt.subplots(1, 2, figsize=(4.8, 1.8), gridspec_kw={"wspace": 0.4})
    net, lo, hi = [0, -7, 2, 7, 12], [-7, -20, -14, -11, -5], [7, 5, 19, 24, 28]
    xs = np.arange(1, 6)
    c.bar(xs, net, width=0.62, color=[NAVY if v > 0 else LIGHT for v in net], edgecolor=INK, linewidth=0.5)
    c.errorbar(xs, net, yerr=[np.subtract(net, lo), np.subtract(hi, net)], fmt="none", ecolor=MID, elinewidth=0.6,
               capsize=1.5)
    c.axhline(0, color=INK, lw=0.6)
    c.set_xticks(xs, ["sure", "", "", "", "unsure"])
    c.set_xlabel("M1 confidence fifth")
    c.set_ylabel("27B fixes $-$ breaks")
    title(c, "Where does the 27B help?")
    orders, pa = ["tax.", "+1", "+2", "+3", "rev."], [0.400, 0.293, 0.247, 0.250, 0.220]
    d.bar(orders, pa, width=0.62, color=[NAVY] + [LIGHT] * 4, edgecolor=INK, linewidth=0.5)
    d.axhline(0.25, color=INK, lw=0.6, ls=":")
    d.text(4.45, 0.262, "chance", fontsize=7, ha="right", va="bottom")
    d.set_ylim(0, 0.46)
    d.set_yticks([0, 0.2, 0.4], ["0", "20%", "40%"])
    d.set_xlabel("order of the four options")
    d.set_ylabel("picks option A")
    title(d, "Option A, by order")
    save(fig, "s_robust_backup")


# ------------------------------------------------------------------------------------------------ Stage 4
def relation() -> None:
    """raw/E24_prechecks.txt P2 and P8."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.4, 1.9), gridspec_kw={"width_ratios": [0.9, 1.3], "wspace": 0.42})
    lab, v = ["same item", "sibling", "random"], [0.432, 0.716, 0.834]
    a.bar(lab, v, width=0.62, color=[LIGHT, NAVY, LIGHT], edgecolor=INK, linewidth=0.5)
    for x, val in zip(lab, v):
        a.text(x, val + 0.03, f"{val:.0%}", ha="center", fontsize=8, color=NAVY if x == "sibling" else INK,
               fontweight="bold" if x == "sibling" else "normal")
    a.set_ylim(0, 1.0)
    a.set_yticks([0, 0.5, 1], ["0", "50%", "100%"])
    a.set_xlabel("two annotators' labels, for …")
    title(a, "Labels differ")
    ident = [("Explicit", 0.817, "CR"), ("Partial", 0.835, "A"), ("Implicit", 0.827, "A"), ("Deflection", 0.798, "A"),
             ("General", 0.791, "A"), ("Dodging", 0.676, "A"), ("Claims ign.", 0.768, "N"), ("Declining", 0.634, "N"),
             ("Clarification", 0.594, "N")]
    fc = {"CR": NAVY, "A": LIGHT, "N": "white"}
    yy = np.arange(len(ident))[::-1]
    for y, (n, val, k) in zip(yy, ident):
        b.barh(y, val, height=0.66, color=fc[k], edgecolor=INK, linewidth=0.5)
    b.axvline(0.817, color=NAVY, lw=0.6, ls=":")
    b.set_yticks(yy, [r[0] for r in ident])
    b.tick_params(axis="y", length=0, labelsize=7.4, right=False)
    b.set_xlim(0.5, 1.08)
    b.set_xticks([0.5, 0.7, 0.9], ["50%", "70%", "90%"])
    b.set_xlabel("answer fits its own question best")
    b.legend(handles=[Patch(fc=NAVY, ec=INK, lw=0.5, label="Clear Reply"), Patch(fc=LIGHT, ec=INK, lw=0.5, label="Ambivalent"),
                      Patch(fc="white", ec=INK, lw=0.5, label="Non-Reply")], fontsize=7, loc="lower right",
             handlelength=0.9, borderaxespad=0.0, bbox_to_anchor=(1.04, -0.02))
    title(b, "Indirect answers are as on-topic as replies")
    save(fig, "s_relation")


def swap(R: dict) -> None:
    """runs/E25/analysis.json, E25a: Clear-Reply share by the question paired with the answer."""
    e = R["e25a"]
    fig, ax = plt.subplots(figsize=(2.5, 2.1))
    conds = [("own", "own\nquestion"), ("rand", "unrelated\nquestion"), ("null", "no\nquestion")]
    x = np.arange(3)
    dev = [e[f"dev_{c}"]["cr"] for c, _ in conds]
    test = [e[f"test_{c}"]["cr"] for c, _ in conds]
    ax.bar(x - 0.18, dev, width=0.34, color=[MID, NAVY, MID], edgecolor=INK, linewidth=0.5, label="dev")
    ax.bar(x + 0.18, test, width=0.34, color="white", edgecolor=INK, linewidth=0.6, label="test (no labels needed)")
    for xi, v in zip(x - 0.18, dev):
        hi = xi == 1 - 0.18
        ax.text(xi, v + 0.02, f"{v:.0%}", ha="center", fontsize=7.8, color=NAVY if hi else INK,
                fontweight="bold" if hi else "normal")
    ax.set_xticks(x, [n for _, n in conds])
    ax.set_ylim(0, 0.85)
    ax.set_yticks([0, 0.25, 0.5, 0.75], ["0", "25%", "50%", "75%"])
    ax.set_ylabel("labelled Clear Reply")
    ax.legend(loc="upper left", fontsize=6.8, handlelength=1.0, borderaxespad=0.1)
    save(fig, "s_swap")


def audit(R: dict) -> None:
    """E25b. (a) share of LLM answers that evade, by role and model (runs/E25/analysis.json); (b) the association as
    plain rates (final_checks section C): how often an LLM evades where the President evaded (filled) and where the
    President answered (open)."""
    from preregistered_analyses.final_checks import roles
    S = quiet(roles)["strata"]
    rates, ci = R["rates"]["table"], R["rates"]["ci"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.5, 1.95), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.3})
    conds = [("N", "neutral"), ("P", "as\nPresident"), ("F", "after\nfollow-up"), ("B", "briefed")]
    models = [("mistral_7b", "Mistral-7B", "o", "white"), ("qwen3_8b", "Qwen3-8B", "s", NAVY)]
    for k, (c, _) in enumerate(conds):
        for i, (m, name, mk, fc) in enumerate(models):
            xx = k + (i - 0.5) * 0.26
            yv = rates[m][c]
            a.errorbar([xx], [yv], yerr=[[yv - ci[m][c][0]], [ci[m][c][1] - yv]], fmt=mk, ms=4.6, mfc=fc, mec=NAVY,
                       mew=0.9, ecolor=MID, elinewidth=0.7, capsize=1.8, label=name if k == 0 else None)
    pg = R["rates"]["president_gold"]
    a.axhline(pg, color=INK, lw=0.7, ls="--")
    a.text(3.45, pg + 0.02, f"Presidents: {pg:.0%}", ha="right", va="bottom", fontsize=7.4)
    a.set_xticks(range(4), [n for _, n in conds])
    a.tick_params(axis="x", labelsize=7.2)
    a.set_xlim(-0.5, 3.5)
    a.set_ylim(0, 1)
    a.set_yticks([0, 0.5, 1], ["0", "50%", "100%"])
    a.set_ylabel("answers that evade")
    a.legend(loc="lower left", fontsize=7, handletextpad=0.3, borderaxespad=0.1)
    title(a, "(a) How often LLMs evade")
    for k, (c, _) in enumerate(conds):
        ev, an = S[c]
        col = NAVY if c == "P" else MID
        b.plot([k, k], [an, ev], color=col, lw=1.2)
        b.plot([k], [ev], marker="o", ms=5, mfc=col, mec=col, ls="none")
        b.plot([k], [an], marker="o", ms=5, mfc="white", mec=col, mew=1.0, ls="none")
        lab = f"OR {R['or'][c]['mh']:.2f}"
        if c == "N":
            b.text(k + 0.12, max(ev, an) + 0.03, lab, fontsize=7.2, va="bottom", ha="left", color=col)
        else:
            b.text(k + 0.2, (ev + an) / 2, lab, fontsize=7.2, va="center", color=col,
                   fontweight="bold" if c == "P" else "normal")
    b.plot([], [], marker="o", ms=5, mfc=MID, mec=MID, ls="none", label="President evaded")
    b.plot([], [], marker="o", ms=5, mfc="white", mec=MID, ls="none", label="President answered")
    b.set_xticks(range(4), ["N", "P", "F", "B"])
    b.tick_params(axis="x", labelsize=8)
    b.set_xlim(-0.4, 4.3)
    b.set_ylim(0, 0.95)
    b.set_yticks([0, 0.25, 0.5, 0.75], ["0", "25%", "50%", "75%"])
    b.legend(loc="upper left", fontsize=7, handletextpad=0.3, borderaxespad=0.1)
    title(b, "(b) Split by what the President did")
    save(fig, "s_audit")


def shortcut(R: dict) -> None:
    """runs/E25/analysis.json, E25c prefixes and E25d probes."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.8, 1.8), gridspec_kw={"wspace": 0.38})
    p = R["prefix"]
    xs = [25, 50, 75, 100]
    ys = [p[f"dev_{x}"]["agree"] for x in xs[:3]] + [1.0]
    a.plot(xs, ys, color=NAVY, lw=1.6, marker="o", ms=4)
    a.plot([50], [0.70], marker="_", ms=13, mew=1.5, color=MID)
    a.annotate("registered bar", (50, 0.70), xytext=(8, -3), textcoords="offset points", fontsize=7, color=MID, va="top")
    a.set_xticks(xs, [f"{x}%" for x in xs])
    a.set_ylim(0.4, 1.05)
    a.set_yticks([0.5, 0.75, 1], ["50%", "75%", "100%"])
    a.set_xlabel("share of the answer read")
    title(a, "Same label as the full answer")
    pr = R["probes"]
    b.plot([r["layer"] for r in pr], [r["dev_s2"] for r in pr], color=MID, lw=1.3, marker="o", ms=3)
    b.axhline(0.525, color=NAVY, lw=1.0, ls="--")
    b.text(0.5, 0.535, "one fine-tuned seed: 0.525", fontsize=7, color=NAVY, va="bottom")
    b.set_ylim(0, 0.65)
    b.set_xlabel("layer of the frozen Qwen3-8B")
    b.set_ylabel("dev S2")
    title(b, "Probing the frozen model")
    save(fig, "s_shortcut")


# ------------------------------------------------------------------------------------------------ Stage 5
def precision() -> None:
    """E26. (a) S2 lost against 16-bit on the same GPU type, with the FROZEN parameters on both sides; (b) FP8's flipped
    decisions that change reference-set membership (final_checks section D)."""
    from preregistered_analyses.final_checks import precision as flips
    R = quiet(flips)
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.4, 1.85), gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.34})
    x = np.arange(2)
    m1 = [R[k]["m1_s2"][1] - R[k]["m1_s2"][0] for k in ("fp8", "nf4")]
    sy = [R[k]["sys_s2"][1] - R[k]["sys_s2"][0] for k in ("fp8", "nf4")]
    a.bar(x - 0.18, m1, width=0.34, color=LIGHT, edgecolor=INK, linewidth=0.5, label="M1 alone")
    a.bar(x + 0.18, sy, width=0.34, color=NAVY, edgecolor=INK, linewidth=0.5, label="full instrument")
    for xi, v, col in zip(list(x - 0.18) + list(x + 0.18), m1 + sy, [INK, INK, NAVY, NAVY]):
        a.text(xi, v - 0.0015, f"{v:.3f}".replace("-", "−"), ha="center", va="top", fontsize=7.4, color=col)
    a.axhline(0, color=INK, lw=0.6)
    a.set_xticks(x, ["FP8", "NF4$^{*}$"])
    a.set_ylim(-0.062, 0.003)
    a.set_ylabel(r"$\Delta$ dev S2 vs 16-bit")
    a.legend(loc="lower right", fontsize=7, handlelength=1.0, borderaxespad=0.1)
    title(a, "(a) Accuracy lost (frozen parameters)")
    f = R["fp8"]
    xs = np.arange(2)
    b.bar(xs - 0.18, [f["m1_lose"], f["sys_lose"]], width=0.34, color=INK, edgecolor=INK, linewidth=0.5, label="lose membership")
    b.bar(xs + 0.18, [f["m1_gain"], f["sys_gain"]], width=0.34, color="white", edgecolor=INK, linewidth=0.7, label="gain it")
    for xi, v in zip(list(xs - 0.18) + list(xs + 0.18), [f["m1_lose"], f["sys_lose"], f["m1_gain"], f["sys_gain"]]):
        b.text(xi, v + 0.3, str(v), ha="center", fontsize=7.8)
    b.set_xticks(xs, ["M1", "full system"])
    b.set_ylim(0, 14)
    b.set_ylabel("flipped decisions")
    b.legend(loc="upper right", fontsize=7, handlelength=1.0, borderaxespad=0.1)
    title(b, "(b) FP8 flips, by direction")
    save(fig, "s_precision")


def main() -> None:
    R = json.loads((RUNS / "E25" / "analysis.json").read_text())
    changes()
    backbone()
    ranking()
    frontier()
    robust()
    nondet()
    robust_backup()
    relation()
    swap(R)
    audit(R)
    shortcut(R)
    precision()


if __name__ == "__main__":
    main()
