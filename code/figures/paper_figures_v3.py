"""Paper figures v3 (audit-led paper; report/paper/figures/), vector PDF. Every plotted value is read from a registered
analysis output (runs/E35/binding.json, runs/E34/analysis_*.json), never typed in.

    python -m figures.paper_figures_v3            # CPU, seconds; FIG_PREVIEW=<dir> also writes PNG previews

fig_binding: binding vs accuracy on dev (E35). x: dev S2; y: sibling same-label rate divided by the same classifier's
  rate on unrelated pairs with disjoint gold sets (1 = siblings labelled as independently as unrelated items), with
  the 95% answer-bootstrap interval. Frozen-probe layers are small grey points.
fig_roles: the role audit (E34). (a) evasion rates by role on questions the President answered / evaded; (b) ratio of
  odds ratios against the analyst, per model and pooled, confirmation (filled) and discovery (hollow), 95% interview
  bootstrap. Values via role_numbers() -> docs/raw/paper_roles.txt.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from figures.paper_figures_v2 import COL, GRID, INK, MUTED, save  # noqa: E402  (shared style and rcParams)
from qevasion.paths import RUNS  # noqa: E402

SHORT = {"M1 (Qwen3-8B LoRA, 10 seeds)": "Qwen3-8B LoRA", "instrument (M1 + 27B adjudicator)": "shared-task instrument",
         "deberta (3 seeds)": "DeBERTa-v3", "llm27b_zeroshot (1 seeds)": "27B zero-shot",
         "anchor: answer-only (M1, no question)": "answer only"}
ACCENT = "#2a78d6"


def fig_binding() -> None:
    d = json.loads((RUNS / "E35" / "binding.json").read_text())["dev"]
    fig, ax = plt.subplots(figsize=(COL, 2.05))
    ax.axhline(1, color=GRID, lw=0.8, zorder=0)
    ax.text(0.205, 0.93, "siblings labelled as independently\nas unrelated items", color=MUTED, fontsize=5.8, va="top")
    for k, r in d.items():
        if not isinstance(r, dict) or "binding" not in r or r["null"] in (0, 1.0):
            continue
        y = r["binding"] / r["null"]
        lo, hi = 1 - r["index_ci"][1], 1 - r["index_ci"][0]
        if k.startswith("frozen probe"):
            ax.errorbar(r["s2"], y, yerr=[[y - lo], [hi - y]], fmt="o", ms=2.6, color="#9a9a9a", ecolor=GRID,
                        elinewidth=0.6, capsize=0, zorder=1)
            continue
        if k not in SHORT:
            continue
        ax.errorbar(r["s2"], y, yerr=[[y - lo], [hi - y]] if hi > lo else None, fmt="o", ms=4, color=ACCENT,
                    ecolor=ACCENT, elinewidth=0.9, capsize=0, zorder=3, mec="white", mew=0.6)
        dx, ha = (0.012, "left")
        if SHORT[k] in ("27B zero-shot",):
            dx, ha = (-0.012, "right")
        ax.text(r["s2"] + dx, y, SHORT[k], fontsize=6.3, color=INK, va="center", ha=ha)
    ax.text(0.255, 7.2, "frozen probes (10 layers)", fontsize=6, color=MUTED, ha="left", va="center")
    ax.set_xlabel("dev macro-F1 (nine labels)")
    ax.set_ylabel("sibling same-label rate\n/ rate on unrelated pairs")
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 4, 8])
    ax.set_yticklabels(["0.5", "1", "2", "4", "8"])
    ax.minorticks_off()
    ax.set_xlim(0.2, 0.8)
    ax.set_ylim(0.45, 9.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.4)
    save(fig, "fig_binding")


ROLE_COLOURS = {"N": "#2a78d6", "S": "#1baf7a", "G": "#eda100", "P": "#eb6834"}
ROLE_NAMES = {"N": "analyst", "S": "press secretary", "G": "unnamed President", "P": "named President"}
ROLE_TICKS = {"N": "analyst", "S": "press\nsecretary", "G": "unnamed\nPresident", "P": "named\nPresident"}
MODEL_NAMES = {"mistral_7b": "Mistral-7B", "qwen3_8b": "Qwen3-8B", "olmo2_7b": "OLMo-2-7B", "granite3_8b": "Granite-3.3-8B",
               "olmo2_32b": "OLMo-2-32B"}


def role_numbers(n_boot: int = 1000) -> dict:
    """Per sample (dev discovery / train confirmation, CF labels): evasion rate on President-answered and -evaded
    sub-questions by role (trio pooled), OR(evade | President evaded) per model and role, and ROR(role / analyst) per
    model, each with a 95% interview-bootstrap interval. One model's ROR is the ratio of two 2x2 odds ratios (the
    per-model logistic model is saturated), so with seed 0 and 1,000 draws it reproduces e34_analysis' per-model
    intervals. Pooled RORs are read from runs/E34/analysis_{dev,train}.json (registered). Written to
    docs/raw/paper_roles.txt and runs/E34/paper_roles.json; the figure reads only that."""
    import numpy as np

    from preregistered_analyses.e34_analysis import TRIO, dev_table, train_table
    from qevasion.paths import ROOT

    samples = [("dev", dev_table(), "evade_frozen")]
    if (RUNS / "E34" / "cf" / "llm.npz").exists():
        t = train_table()
        samples.append(("train", t[t.confirm == 1].reset_index(drop=True), "evade_cf"))
    out, L = {}, [f"Role audit numbers for the paper figure (interview bootstrap, {n_boot} draws, seed 0)"]
    for name, d, y in samples:
        urls = np.array(sorted(d.url.unique()))
        rng = np.random.default_rng(0)
        W = np.stack([np.bincount(np.searchsorted(urls, rng.choice(urls, len(urls))), minlength=len(urls))
                      for _ in range(n_boot)]).astype(float)
        ui = np.searchsorted(urls, d.url.to_numpy())
        k = (1 - d.pres_evade.to_numpy()) * 2 + (1 - d[y].to_numpy())  # 0 pe1y1, 1 pe1y0, 2 pe0y1, 3 pe0y0

        def cells(mask):
            C = np.zeros((len(urls), 4))
            np.add.at(C, (ui[mask], k[mask]), 1)
            return np.vstack([C.sum(0), W @ C])  # row 0: point estimate; rows 1..: draws

        def ci(v):
            return [float(v[0]), *np.nanpercentile(v[1:], [2.5, 97.5]).tolist()]

        res = {"rates": {}, "or": {}, "ror": {}}
        conds = [c for c in ("N", "S", "G", "P") if c in set(d.condition)]
        trio = d.model.isin(TRIO).to_numpy()
        for c in conds:
            cc = cells(trio & (d.condition == c).to_numpy())
            res["rates"][f"{c}|0"] = ci(cc[:, 2] / (cc[:, 2] + cc[:, 3]))
            res["rates"][f"{c}|1"] = ci(cc[:, 0] / (cc[:, 0] + cc[:, 1]))
        for m in sorted(d.model.unique()):
            ors = {}
            for c in conds:
                cc = cells(((d.model == m) & (d.condition == c)).to_numpy())
                ors[c] = cc[:, 0] * cc[:, 3] / (cc[:, 1] * cc[:, 2])
                res["or"][f"{m}|{c}"] = ci(ors[c])
            for c in conds:
                if c != "N":
                    res["ror"][f"{m}|{c}"] = ci(ors[c] / ors["N"])
        an = json.loads((RUNS / "E34" / f"analysis_{name}.json").read_text())
        pooled = {"P": an.get("H34-1") or an.get("frozen", {}).get("ror_P_N")}
        for key, c in (("H34-2 ROR S/N > 1", "S"), ("H34-4 ROR G/N > 1", "G")):
            if key in an.get("secondary", {}):
                pooled[c] = an["secondary"][key]
        res["pooled"] = {c: [v["ror"], *v["ci"]] for c, v in pooled.items() if v}
        out[name] = res
        L += ["", f"[{name}] evasion rate (trio pooled) on sub-questions the President answered (0) / evaded (1):"]
        L += [f"   {k_}: {v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]" for k_, v in res["rates"].items()]
        L += [f"[{name}] OR(evade | President evaded) per model and role:"]
        L += [f"   {k_}: {v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]" for k_, v in res["or"].items()]
        L += [f"[{name}] ROR(role / analyst) per model:"]
        L += [f"   {k_}: {v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]" for k_, v in res["ror"].items()]
        L += [f"[{name}] pooled trio ROR (registered analysis): " +
              "  ".join(f"{c}/N {v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]" for c, v in res["pooled"].items())]
    (ROOT / "docs" / "raw" / "paper_roles.txt").write_text("\n".join(L) + "\n", encoding="utf8")
    (RUNS / "E34").mkdir(parents=True, exist_ok=True)
    (RUNS / "E34" / "paper_roles.json").write_text(json.dumps(out, indent=1))
    return out


def fig_roles() -> None:
    """(a) confirmation sample, trio pooled: share of answers that evade, on sub-questions the President answered
    (open) and evaded (filled), by role. (b) ROR of each role against the analyst, per model and pooled over the trio:
    confirmation filled, discovery (dev, President only) hollow."""
    R = role_numbers()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(2 * COL + 0.3, 2.15), gridspec_kw={"width_ratios": [1, 1.45]})
    s = "train" if "train" in R else "dev"
    roles = [c for c in ("N", "S", "G", "P") if f"{c}|0" in R[s]["rates"]]
    for j, c in enumerate(roles):
        for pe in (0, 1):
            v = R[s]["rates"][f"{c}|{pe}"]
            x = j + (-0.13 if pe == 0 else 0.13)
            a1.errorbar(x, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="o", ms=4.2, color=ROLE_COLOURS[c],
                        mfc="white" if pe == 0 else ROLE_COLOURS[c], ecolor=ROLE_COLOURS[c], elinewidth=0.9, capsize=0)
        a1.plot([j - 0.13, j + 0.13], [R[s]["rates"][f"{c}|0"][0], R[s]["rates"][f"{c}|1"][0]], color=ROLE_COLOURS[c],
                lw=0.8, zorder=0)
    a1.set_xticks(range(len(roles)))
    a1.set_xticklabels([ROLE_TICKS[c] for c in roles], fontsize=6.3)
    a1.set_ylabel("share of answers that evade")
    a1.set_ylim(0.3, 0.7)
    a1.set_xlim(-0.5, len(roles) - 0.5)
    a1.text(0.03, 0.97, "open: the President answered\nfilled: the President evaded", transform=a1.transAxes,
            fontsize=6, color=MUTED, va="top")
    a1.set_title("(a) evasion rate by role (confirmation, three models)", fontsize=7, color=INK, loc="left")
    order = ("mistral_7b", "olmo2_7b", "qwen3_8b", "granite3_8b", "olmo2_32b")
    models = [m for m in order if f"{m}|P" in R[s]["ror"]]
    rows = models + ["pooled"]
    off = {"S": -0.27, "G": -0.09, "P": 0.09}
    for i, m in enumerate(rows):
        for c in ("S", "G", "P"):
            v = R[s]["pooled"].get(c) if m == "pooled" else R[s]["ror"].get(f"{m}|{c}")
            if v is None:
                continue
            a2.errorbar(v[0], i + off[c], xerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="o", ms=3.4, color=ROLE_COLOURS[c],
                        ecolor=ROLE_COLOURS[c], elinewidth=0.8, capsize=0, label=ROLE_NAMES[c] if i == 0 else None)
        if "dev" in R and s == "train":
            v = R["dev"]["pooled"].get("P") if m == "pooled" else R["dev"]["ror"].get(f"{m}|P")
            if v is not None:
                a2.errorbar(v[0], i + 0.27, xerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="o", ms=3.4, color=ROLE_COLOURS["P"],
                            mfc="white", ecolor=ROLE_COLOURS["P"], elinewidth=0.6, capsize=0, alpha=0.8,
                            label="named President, discovery" if i == 0 else None)
    a2.axvline(1, color=GRID, lw=0.8, zorder=0)
    a2.axhline(len(models) - 0.5, color=GRID, lw=0.5, zorder=0)
    a2.axhline(2.5, color=GRID, lw=0.5, ls=":", zorder=0)
    a2.set_xscale("log")
    a2.set_xticks([0.5, 1, 2, 4])
    a2.set_xticklabels(["0.5", "1", "2", "4"])
    a2.minorticks_off()
    a2.set_yticks(range(len(rows)))
    a2.set_yticklabels([MODEL_NAMES.get(m, "three models, pooled") for m in rows], fontsize=6.3)
    a2.set_ylim(len(rows) - 0.4, -0.5)
    a2.set_xlabel("OR(role) / OR(analyst); OR = odds of evading where the President evaded")
    a2.set_title("(b) ratio of odds ratios vs analyst", fontsize=7, color=INK, loc="left", pad=13)
    a2.legend(loc="lower left", bbox_to_anchor=(0.0, 0.995), ncol=4, fontsize=5.6, frameon=False, handletextpad=0.1,
              columnspacing=0.8, borderaxespad=0.0)
    for ax in (a1, a2):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    a1.grid(axis="y", color=GRID, lw=0.4)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_roles")


def main() -> None:
    fig_binding()
    fig_roles()
    print("wrote fig_binding, fig_roles")


if __name__ == "__main__":
    main()
