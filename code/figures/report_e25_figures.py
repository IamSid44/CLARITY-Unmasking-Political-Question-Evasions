#!/usr/bin/env python3
"""E25 figures for the final report, in the report's publication style (paper_figures.py): STIX serif,
inward ticks, grayscale, series told apart by marker shape and line style rather than colour.

    python -m figures.report_e25_figures   # -> report/final/figures/fig_e25_audit.pdf, fig_e25_swap_probe.pdf

Reads runs/E25/analysis.json (written by preregistered_analyses.e25_analysis). The colour versions for slides and
docs stay in figures/e25_figures.py.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from qevasion.paths import ROOT, RUNS  # noqa: E402

OUT = ROOT / "report" / "final" / "figures"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.8, "axes.linewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "xtick.top": False, "ytick.right": True,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})

# One marker per condition, all in black: open/filled alternate so that overlapping points stay legible.
CONDS = [("N", "Neutral analyst", "o", "white"), ("P", "As the President", "s", "black"),
         ("F", "After a follow-up", "D", "0.55"), ("B", "Briefed with the answer", "^", "white")]
MODELS = [("mistral_7b", "Mistral-7B"), ("qwen3_8b", "Qwen3-8B")]


def audit(R: dict) -> None:
    """(a) evasion rate per condition, one marker per model, 95% cluster-bootstrap CIs; (b) the plain rates behind the odds ratios. Conditions are named on the axes, so no condition legend is needed."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 1.6), gridspec_kw={"width_ratios": [1.15, 1]})
    rates, ci = R["rates"]["table"], R["rates"]["ci"]
    short = {"N": "neutral\nanalyst", "P": "as the\nPresident", "F": "after a\nfollow-up", "B": "briefed with\nthe answer"}
    model_style = {"mistral_7b": ("o", "white"), "qwen3_8b": ("s", "black")}
    for k, (c, _, _, _) in enumerate(CONDS):
        for i, (m, mname) in enumerate(MODELS):
            x = k + (i - 0.5) * 0.24
            y = rates[m][c]
            mk, fc = model_style[m]
            a.errorbar([x], [y], yerr=[[y - ci[m][c][0]], [ci[m][c][1] - y]], fmt=mk, ms=3.8, mfc=fc, mec="black",
                       mew=0.7, ecolor="black", elinewidth=0.7, capsize=1.6, capthick=0.6,
                       label=mname if k == 0 else None)
    pg = R["rates"]["president_gold"]
    a.axhline(pg, color="black", lw=0.7, ls="--")
    a.text(3.45, pg + 0.02, f"Presidents (gold) {pg:.2f}", ha="right", va="bottom", fontsize=6.8)
    a.set_xticks(range(len(CONDS)), [short[c] for c, _, _, _ in CONDS])
    a.tick_params(axis="x", labelsize=6.6)
    a.set_xlim(-0.5, len(CONDS) - 0.5)
    a.set_ylim(0, 1)
    a.set_ylabel("answers that evade")
    a.set_title("(a) Evasion rate", loc="left", pad=3)
    a.legend(loc="lower left", frameon=False, handletextpad=0.3, borderaxespad=0.2, fontsize=6.6)

    # (b) the odds ratios as plain rates: how often an LLM evades where the President evaded (filled) and where the
    # President answered (open); the gap is the association (pooled over the two models; final_checks.py section C)
    import contextlib
    import io

    from preregistered_analyses.final_checks import roles
    with contextlib.redirect_stdout(io.StringIO()):
        S = roles()["strata"]
    for k, (c, name, _, _) in enumerate(CONDS):
        ev, an = S[c]
        b.plot([k, k], [an, ev], color="black", lw=0.8)
        b.plot([k], [ev], marker="o", ms=4.2, mfc="black", mec="black", mew=0.7, ls="none")
        b.plot([k], [an], marker="o", ms=4.2, mfc="white", mec="black", mew=0.7, ls="none")
        b.text(k + 0.14, (ev + an) / 2, f"OR {R['or'][c]['mh']:.2f}", fontsize=6.0, va="center")
    b.plot([], [], marker="o", ms=4.2, mfc="black", mec="black", ls="none", label="President evaded")
    b.plot([], [], marker="o", ms=4.2, mfc="white", mec="black", ls="none", label="President answered")
    b.set_xticks(range(4), [short[c] for c, _, _, _ in CONDS])
    b.tick_params(axis="x", labelsize=6.6)
    b.set_xlim(-0.4, 3.9)
    b.set_ylim(0, 0.95)
    b.set_ylabel("answers that evade")
    b.legend(loc="upper left", frameon=False, fontsize=6.4, handletextpad=0.3, borderaxespad=0.1)
    b.set_title("(b) Evasion by whether the President evaded", loc="left", pad=3)
    fig.tight_layout(pad=0.2, w_pad=1.2)
    fig.savefig(OUT / "fig_e25_audit.pdf", bbox_inches="tight")
    plt.close(fig)


def swap_probe(R: dict) -> None:
    """(a) question swap on dev and test; (b) answer prefixes; (c) frozen-layer probes."""
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(6.3, 1.95))
    e = R["e25a"]
    frames = [("own", "own"), ("rand", "unrelated"), ("null", "none")]
    for split, ls, mk in (("dev", "-", "o"), ("test", "--", "s")):
        a.plot(range(3), [e[f"{split}_{f}"]["cr"] for f, _ in frames], ls=ls, color="black", lw=0.8, marker=mk,
               ms=3.4, mfc="black", mec="black", label=f"{split}: Clear-Reply share")
        a.plot(range(1, 3), [e[f"{split}_{f}"]["persist"] for f, _ in frames[1:]], ls=ls, color="0.55", lw=0.8,
               marker=mk, ms=3.4, mfc="white", mec="0.35", label=f"{split}: label unchanged")
    a.set_xticks(range(3), [n for _, n in frames])
    a.set_xlim(-0.25, 2.25)
    a.set_ylim(0, 1)
    a.set_xlabel("Question paired with the answer")
    a.set_title("(a) Swapping the question", loc="left", pad=3)

    p = R["prefix"]
    xs = [25, 50, 75, 100]
    b.plot(xs, [p[f"dev_{x}"]["agree"] for x in xs[:3]] + [1.0], color="black", lw=0.8, marker="o", ms=3.4)
    b.set_xticks(xs, [f"{x}%" for x in xs])
    b.set_ylim(0, 1)
    b.set_xlabel("Share of the answer read")
    b.set_title("(b) Same label as full answer", loc="left", pad=3)

    pr = R["probes"]
    c.plot([r["layer"] for r in pr], [r["dev_s2"] for r in pr], color="black", lw=0.8, marker="o", ms=2.8,
           mfc="white")
    for v, ls, lab in ((0.598, ":", "10-seed M1"), (0.525, "--", "one fine-tuned seed")):
        c.axhline(v, color="black", lw=0.7, ls=ls)
        c.text(36, v + 0.012, lab, ha="right", va="bottom", fontsize=6.5)
    c.set_ylim(0, 0.7)
    c.set_xlabel("Layer of the frozen Qwen3-8B")
    c.set_ylabel("Dev S2 macro-F1")
    c.set_title("(c) Linear probes", loc="left", pad=3)
    h, lab = a.get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02), handlelength=2.2,
               columnspacing=1.2)
    fig.tight_layout(pad=0.2, w_pad=1.0, rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "fig_e25_swap_probe.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    R = json.loads((RUNS / "E25" / "analysis.json").read_text())
    audit(R)
    swap_probe(R)
    print("wrote", OUT / "fig_e25_audit.pdf", "and", OUT / "fig_e25_swap_probe.pdf")


if __name__ == "__main__":
    main()
