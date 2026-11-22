"""Workshop-paper figures (report/paper/figures/), vector PDF; every plotted value is computed here from runs/ and
written to docs/raw/ first, so the figure and its numbers have one source.

    python -m figures.paper_figures_v2          # CPU, ~1 min

fig_audit: where LLMs evade. For each model and condition, the evasion rate (instrument clarity != Clear Reply) on
  sub-questions the Presidents evaded (gold) and on those they answered, with turn-cluster bootstrap intervals.
  Values: docs/raw/paper_audit_rates.txt. OLMo-2 (E32, protocol A) is included once runs/E32/labels/llm.npz exists.
One colour per condition across every paper and slide figure (CONDITION_COLOURS); marker shape carries the second
variable, so colour is never the only cue.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from qevasion.paths import RAW, ROOT, RUNS  # noqa: E402

OUT = ROOT / "report" / "paper" / "figures"
COL = 3.15  # ACL single column, inches
CONDS = ("N", "P", "F", "B")
CONDITION_COLOURS = {"N": "#2a78d6", "P": "#eb6834", "F": "#1baf7a", "B": "#eda100"}  # reference palette slots 1-4
CONDITION_NAMES = {"N": "neutral", "P": "President", "F": "pursued", "B": "briefed"}
MODEL_NAMES = {"mistral_7b": "Mistral-7B", "qwen3_8b": "Qwen3-8B", "olmo2_7b": "OLMo-2-7B"}
INK, MUTED, GRID = "#1a1a1a", "#5f5f5f", "#d9d9d9"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    if os.environ.get("FIG_PREVIEW"):
        fig.savefig(os.path.join(os.environ["FIG_PREVIEW"], f"{name}.png"), dpi=200)
    plt.close(fig)


def audit_table() -> pd.DataFrame:
    from preregistered_analyses.e25_analysis import llm_table

    df = llm_table()
    p = RUNS / "E32" / "llm_table.parquet"
    if p.exists():
        t = pd.read_parquet(p)
        df = pd.concat([df, t[t.model == "olmo2_7b"]], ignore_index=True)
    return df


def audit_rates(df: pd.DataFrame) -> pd.DataFrame:
    from preregistered_analyses.e25_analysis import boot_stat

    rows = []
    for m in [k for k in MODEL_NAMES if k in set(df.model)]:
        for c in CONDS:
            g = df[(df.model == m) & (df.condition == c)].reset_index(drop=True)
            for pe, name in ((1, "evaded"), (0, "answered")):
                h = g[g.pres_evade == pe].reset_index(drop=True)
                bs = np.array(boot_stat(h, lambda d: d["evade"].mean()))
                rows.append({"model": m, "condition": c, "president": name, "n": len(h), "rate": h["evade"].mean(),
                             "lo": np.nanpercentile(bs, 2.5), "hi": np.nanpercentile(bs, 97.5)})
    r = pd.DataFrame(rows)
    lines = ["LLM evasion rate (instrument clarity != Clear Reply) on dev sub-questions the Presidents evaded (gold) vs "
             "answered; 95% cluster bootstrap over 191 turns (2,000 draws)", ""]
    for _, x in r.iterrows():
        lines.append(f"{x.model:<11} {x.condition}  President {x.president:<8} n {x.n:>3}  rate {x.rate:.3f}  "
                     f"[{x.lo:.3f}, {x.hi:.3f}]")
    (RAW / "paper_audit_rates.txt").write_text("\n".join(lines) + "\n", encoding="utf8")
    return r


def fig_audit(r: pd.DataFrame) -> None:
    models = [m for m in MODEL_NAMES if m in set(r.model)]
    fig, axes = plt.subplots(1, len(models), figsize=(COL, 1.75), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        for k, c in enumerate(CONDS):
            ev = r[(r.model == m) & (r.condition == c) & (r.president == "evaded")].iloc[0]
            an = r[(r.model == m) & (r.condition == c) & (r.president == "answered")].iloc[0]
            col = CONDITION_COLOURS[c]
            ax.plot([k - 0.12, k + 0.12], [ev.rate, an.rate], color=col, lw=1.6, solid_capstyle="round", zorder=1)
            ax.errorbar(k - 0.12, ev.rate, yerr=[[ev.rate - ev.lo], [ev.hi - ev.rate]], fmt="o", ms=4.2, color=col,
                        mec="white", mew=0.8, elinewidth=0.7, capsize=0, zorder=3)
            ax.errorbar(k + 0.12, an.rate, yerr=[[an.rate - an.lo], [an.hi - an.rate]], fmt="o", ms=4.2, mfc="white",
                        color=col, mew=1.1, elinewidth=0.7, capsize=0, zorder=3)
        ax.set_xticks(range(len(CONDS)), [c for c in CONDS])
        ax.set_xlim(-0.6, len(CONDS) - 0.4)
        ax.set_title(MODEL_NAMES[m], pad=3)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("LLM evasion rate")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=MUTED, mec="white", ms=5, label="on questions the President evaded"),
         Line2D([], [], marker="o", ls="", mfc="white", color=MUTED, mew=1.1, ms=5, label="on questions the President answered")]
    fig.legend(handles=h, loc="upper center", ncol=1, frameon=False, bbox_to_anchor=(0.55, -0.02), handletextpad=0.3)
    save(fig, "fig_audit")


def main() -> None:
    r = audit_rates(audit_table())
    fig_audit(r)
    print((RAW / "paper_audit_rates.txt").read_text())


if __name__ == "__main__":
    main()
