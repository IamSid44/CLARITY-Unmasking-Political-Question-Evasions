#!/usr/bin/env python3
"""E18 (freeze), part A: final numbers for both operating points, the cost table and the frontier figure.

    python -m preregistered_analyses.e18_final > ../docs/raw/E18_final.txt     # CPU, ~5 min

Accuracy point: the 10-seed M1 (Q8_fullq_lora_12ep) + M3 at δ = 0.3. Cost point: one distilled student
(Q8_distill_a05) + M3 at δ = 0.5, reported over the 10 students. Same procedures as E15-E17: M1's own tau,
fused (lam, tau) for deferred items, 5-fold nested CV on dev; 2-annotator scores refit the CV per reference set
(test regime, as E14). Needs runs/M3/{cost.json, frontier_Q8_fullq_lora_12ep.json,
frontier_Q8_distill_a05_ensembles.json} and the M3 responses of tag qwen38_27b_k0.
"""

from __future__ import annotations

import json

import numpy as np

from cascade.frontier import m1_inputs, m3_lookup, top4
from cascade.fuse import LOGPI, PRIOR, fit_system, sp
from decision_rules.decide import fit_tau, nested_cv, reference_masks
from qevasion.labels import EVASION_LABELS, N_EVASION, encode_clarity, leaf_to_official_clarity
from qevasion.loader import dev_reference_mask
from qevasion.paths import FIGURES, RUNS
from qevasion.scoring import score_subtask1, score_subtask2

M1, STUDENT = "Q8_fullq_lora_12ep", "Q8_distill_a05"
TAUS = np.linspace(-1.0, 2.0, 61)
GOLD = dev_reference_mask(sp.dev)
CLAR = encode_clarity(sp.dev["clarity_label"].tolist())
TWO = {n: g for n, g in reference_masks(sp.dev, True).items() if n != "3ann"}
N = len(GOLD)
LOOK = m3_lookup()
COST = json.loads((RUNS / "M3" / "cost.json").read_text())
BOOTS = [np.random.default_rng(i).integers(0, N, N) for i in range(2000)]
TELEAI = {"s2": 0.617, "s1": 0.812, "calls": 1.94, "in_tokens": 7300, "out_tokens": 380}  # report §7, dev


def s2(p, g=GOLD) -> float:
    return score_subtask2(p, g).macro_f1


def s1(p) -> float:
    return score_subtask1(leaf_to_official_clarity(p), CLAR).macro_f1


class System:
    """One (run, seed subset) M1 with M3 on the fraction delta of least confident items (0 = M1 alone)."""

    def __init__(self, run: str, seeds: tuple[int, ...], delta: float):
        m = m1_inputs(seeds, run)
        self.pbar, self.lp = m["pbar"], np.log(m["pbar"] + 1e-12)
        self.lq = np.full((N, N_EVASION), -np.inf)
        for i, c in enumerate(top4(m["q_dev"])):
            self.lq[i, c] = np.log(np.maximum(LOOK[(i, tuple(int(x) for x in c))], 1e-12))
        u_dev, u_val = 1 - m["q_dev"].max(1), 1 - m["q_val"].max(1)
        self.defer = np.zeros(N, bool) if delta == 0 else (
            np.ones(N, bool) if delta == 1 else u_dev >= np.quantile(u_val, 1 - delta))
        self.k = len(seeds)

    def rule(self, _p, g, tr, te):
        if not self.defer.any():
            tau, fit = fit_tau(self.pbar[tr], g[tr], PRIOR, TAUS)
            return (self.lp - tau * LOGPI).argmax(1)[te], fit
        pred, fit, *_ = fit_system(self.lp, self.lq, self.pbar, self.defer, g, tr)
        return pred[te], fit

    def score(self, splits: int = 10) -> dict:
        runs = [nested_cv(self.rule, self.lp, GOLD, 5, cs) for cs in range(splits)]
        pred = runs[0]["pooled_predictions"]
        two = [nested_cv(self.rule, self.lp, g, 5, 0)["out_of_fold_macro_f1"] for g in TWO.values()]
        calls = float(self.defer.mean())
        return {"pred": pred, "s2": s2(pred), "s2_10": float(np.mean([r["out_of_fold_macro_f1"] for r in runs])),
                "s1": s1(pred), "two": float(np.mean(two)), "two_each": two, "calls": calls,
                "inset": float(GOLD[np.arange(N), pred].mean()),
                "gpu_s": self.k * COST["m1_seed_gpu_s_per_item"] + calls * COST["m3_call_gpu_s"],
                "tokens_in": self.k * COST["m1_tokens_per_item"] + calls * COST["m3_prompt_tokens"],
                "tokens_out": calls * COST["m3_output_tokens"]}


def boot(a: np.ndarray, b: np.ndarray) -> str:
    d = np.array([s2(a[i], GOLD[i]) - s2(b[i], GOLD[i]) for i in BOOTS])
    return f"{d.mean():+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]"


def per_class(preds: list[np.ndarray]) -> np.ndarray:
    return np.mean([[score_subtask2(p, GOLD).per_class[n].f1 for n in EVASION_LABELS] for p in preds], 0)


def main() -> None:
    print("E18 (freeze), part A: final dev numbers. Nested CV: split 0 and mean of 10 splits; 2-annotator = mean over "
          "the three 2-annotator reference sets, CV refitted on each (test regime).\n")
    # ---- accuracy point and its references
    m1 = System(M1, tuple(range(10)), 0.0).score()
    acc = System(M1, tuple(range(10)), 0.3).score()
    print("1. Accuracy operating point: 10-seed M1 + M3, δ = 0.3\n")
    print(f"   {'system':<34}{'S2':>6}{'10 sp':>7}{'S1':>7}{'2-ann':>7}{'in-set':>8}{'calls':>7}{'GPU-s':>7}")
    for name, r in (("M1, 10 seeds (E14)", m1), ("M1 + M3, δ = 0.3 (E15)", acc)):
        print(f"   {name:<34}{r['s2']:>6.3f}{r['s2_10']:>7.3f}{r['s1']:>7.3f}{r['two']:>7.3f}{r['inset']:>8.3f}"
              f"{r['calls']:>7.3f}{r['gpu_s']:>7.3f}")
    print(f"   Δ S2 vs M1 (bootstrap over items): {boot(acc['pred'], m1['pred'])}; "
          f"2-annotator sets: {', '.join(f'{x:.3f}' for x in acc['two_each'])} (M1: {', '.join(f'{x:.3f}' for x in m1['two_each'])})")

    # ---- cost point: each student, and the undistilled single model on the same seed
    print("\n2. Cost operating point: one distilled student + M3, δ = 0.5 (each of the 10 students; mean ± sd)\n")
    st = [System(STUDENT, (s,), 0.5).score() for s in range(10)]
    ct = [System(M1, (s,), 0.5).score() for s in range(10)]
    sa = [System(STUDENT, (s,), 0.0).score(1) for s in range(10)]
    keys = (("s2", "S2 (split 0)"), ("s2_10", "S2 (10 splits)"), ("s1", "S1"), ("two", "2-annotator S2"),
            ("inset", "in-set rate"), ("calls", "LLM calls/item"), ("gpu_s", "GPU-s/item"))
    print(f"   {'':<20}{'student + M3':>16}{'undistilled 1 seed + M3':>26}{'paired Δ (n up)':>20}{'student alone':>16}")
    for k, lab in keys:
        a, b, c = np.array([r[k] for r in st]), np.array([r[k] for r in ct]), np.array([r[k] for r in sa])
        d = a - b
        print(f"   {lab:<20}{a.mean():>10.3f} ± {a.std(ddof=1):.3f}{b.mean():>16.3f} ± {b.std(ddof=1):.3f}"
              f"{d.mean():>+12.3f} ({int((d > 0).sum())}/10){c.mean():>12.3f} ± {c.std(ddof=1):.3f}")

    # ---- per class
    print("\n3. Per-class F1 on dev (S2 scorer), split-0 predictions; cost point averaged over the 10 students\n")
    pc = {"M1 (10 seeds)": per_class([m1["pred"]]), "accuracy point": per_class([acc["pred"]]),
          "student alone": per_class([r["pred"] for r in sa]), "cost point": per_class([r["pred"] for r in st])}
    print(f"   {'class':<22}" + "".join(f"{n:>16}" for n in pc))
    for j, name in enumerate(EVASION_LABELS):
        print(f"   {name:<22}" + "".join(f"{v[j]:>16.3f}" for v in pc.values()))

    # ---- cost table
    print("\n4. Cost per dev item (lab card: RTX PRO 6000 Max-Q, GPU 0 alone; runs/M3/cost.json)\n")
    rows = [("M1, 10 seeds", m1), ("accuracy point: M1 + M3, δ = 0.3", acc),
            ("student alone (mean of 10)", {k: np.mean([r[k] for r in sa]) for k in ("s2", "s1", "calls", "gpu_s", "tokens_in", "tokens_out")}),
            ("cost point: student + M3, δ = 0.5", {k: np.mean([r[k] for r in st]) for k in ("s2", "s1", "calls", "gpu_s", "tokens_in", "tokens_out")})]
    print(f"   {'system':<36}{'S2':>6}{'S1':>7}{'LLM calls':>10}{'GPU-s':>8}{'in tokens':>10}{'out tokens':>11}{'GPU-s vs M1':>12}")
    for name, r in rows:
        print(f"   {name:<36}{r['s2']:>6.3f}{r['s1']:>7.3f}{r['calls']:>10.2f}{r['gpu_s']:>8.3f}{r['tokens_in']:>10.0f}"
              f"{r['tokens_out']:>11.2f}{r['gpu_s'] / m1['gpu_s']:>11.2f}x")
    print(f"   {'TeleAI (1st place; DeepSeek-V3 API)':<36}{TELEAI['s2']:>6.3f}{TELEAI['s1']:>7.3f}{TELEAI['calls']:>10.2f}"
          f"{'n/a':>8}{TELEAI['in_tokens']:>10.0f}{TELEAI['out_tokens']:>11.0f}{'n/a':>12}")
    figure()


def figure() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from figures.paper_figures import plt as _style  # noqa: F401  (applies the report's rcParams)

    u = json.loads((RUNS / "M3" / f"frontier_{M1}.json").read_text())["table"]
    s = json.loads((RUNS / "M3" / f"frontier_{STUDENT}_ensembles.json").read_text())["table"]
    from matplotlib.ticker import FixedLocator, NullLocator

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    shown = (1, 3, 10)  # every K is in raw/E16_frontier.txt and raw/E18_frontier_student_ensembles.txt
    for table, mk, col, lab in ((u, "o", "0.55", "M1, K seeds + M3"), (s, "s", "black", "distilled students, K + M3")):
        for k in shown:
            pts = sorted((r for r in table if r["K"] == k), key=lambda r: r["gpu_s"])
            ax.plot([r["gpu_s"] for r in pts], [r["s2"] for r in pts], color=col, lw=0.6, zorder=1)
            ax.plot([r["gpu_s"] for r in pts], [r["s2"] for r in pts], ls="none", marker=mk, ms=2.6,
                    mfc="white" if mk == "o" else col, mec=col, mew=0.6, zorder=2, label=lab if k == 1 else None)
            right = k == 10 and mk == "o"  # M1's K = 10 at its right end; the students' K = 10 under its cluster
            end = pts[-1] if right else (min(pts, key=lambda r: r["s2"]) if k == 10 else pts[0])
            ax.annotate(f"K={k}", (end["gpu_s"], end["s2"]), xytext=(3, 0) if right else (0, -7),
                        textcoords="offset points", fontsize=5.5, color=col, ha="left" if right else "center",
                        va="center" if right else "baseline")
    acc = next(r for r in u if r["K"] == 10 and r["delta"] == 0.3)
    cost = next(r for r in s if r["K"] == 1 and r["delta"] == 0.5)
    ax.plot(acc["gpu_s"], acc["s2"], marker="*", ms=7, color="black", zorder=3)
    ax.annotate("accuracy point\n(M1 ×10 + M3, δ = 0.3)", (acc["gpu_s"], acc["s2"]), xytext=(-6, -2),
                textcoords="offset points", fontsize=5.5, ha="right", va="center")
    ax.plot(cost["gpu_s"], cost["s2"], marker="*", ms=7, color="black", zorder=3)
    ax.annotate("cost point\n(student + M3, δ = 0.5)", (cost["gpu_s"], cost["s2"]), xytext=(0.125, 0.532),
                textcoords="data", fontsize=5.5, ha="center", va="bottom",
                arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "0.2", "shrinkA": 1, "shrinkB": 4})
    ax.axhline(TELEAI["s2"], color="0.3", lw=0.6, ls="--")
    ax.text(0.048, TELEAI["s2"] + 0.003, "TeleAI, 1st place (1.94 LLM calls/item)", fontsize=5.5, color="0.3")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.2, 0.3, 0.5]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["0.05", "0.1", "0.2", "0.3", "0.5"])
    ax.set_xlim(0.045, 0.8)
    ax.set_ylim(0.52, 0.68)
    ax.set_xlabel("GPU-seconds per item (log scale)")
    ax.set_ylabel("dev Subtask 2 macro-F1")
    ax.legend(loc="lower right", frameon=False)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"E18_frontier.{ext}", bbox_inches="tight")
    print(f"\n5. Figure: {FIGURES / 'E18_frontier.pdf'} (and .png)")


if __name__ == "__main__":
    main()
