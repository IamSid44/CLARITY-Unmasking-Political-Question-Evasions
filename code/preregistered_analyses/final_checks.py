#!/usr/bin/env python3
"""Checks added for the final report and slides (CPU, existing records only; nothing is refitted).

    python -m preregistered_analyses.final_checks > ../docs/raw/final_checks.txt

These are descriptive follow-ups to E19-E26, written after the results were known, so none of them is a registered
test. They answer four questions a reader can ask of the final report:

A. Option order. How often do two presentation orders of the same four candidates give the same pick, and what is the
   "average of four orders" row of E22b (it is the score of the averaged probabilities, not the mean of the scores)?
B. Two copies of the same 27B weights. Where do the lab and cloud copies disagree (by the gap between the top two
   options), and is the apparent self-agreement of the cloud copy an independent repeat (it is not)?
C. E25b. The odds ratios of the report, as plain rates: how often does an LLM evade when the President evaded, and when
   the President answered; and do the neutral analyst and the President-role model evade on the same questions?
D. E26. Of the decisions FP8 and NF4 flip in M1, which route do they take through the instrument, and do the flips
   lose or gain reference-set membership? With the frozen parameters on both sides, what does each lose?

`compute()` returns the numbers used by figures/report_final_figures.py and figures/slide_figures.py.
"""

from __future__ import annotations

import itertools

import numpy as np

from preregistered_analyses import e25_analysis as E25
from preregistered_analyses import e26_analysis as E26A
from preregistered_analyses.e21_e22_analysis import Y_TR, order_table, recs
from qevasion.scoring import score_subtask2

CYCLIC = ("taxonomy", "shift1", "shift2", "shift3")


def rule(t: str) -> None:
    print("\n" + "=" * 110 + f"\n{t}\n" + "=" * 110)


def _picks(t: dict) -> dict:
    return {o: np.array([c[j] for c, j in zip(t["rows"][o][1], t["rows"][o][0].argmax(1))]) for o in t["rows"]}


# ------------------------------------------------------------------------------------------------ A
def order() -> dict:
    out = {}
    rule("A. Option order: agreement between presentation orders of the same four candidates")
    for split, gold_of in (("screen", lambda i: int(Y_TR[i])), ("dev", lambda i: i)):
        t = order_table(split, gold_of)
        pk = _picks(t)
        pairs = {(a, b): float((pk[a] == pk[b]).mean()) for a, b in itertools.combinations(CYCLIC, 2)}
        all4 = float(np.all([pk[o] == pk["taxonomy"] for o in CYCLIC], axis=0).mean())
        vs_avg = {o: float((pk[o] == pk["average (4 cyclic)"]).mean()) for o in CYCLIC}
        out[split] = {"pairs": pairs, "mean_pair": float(np.mean(list(pairs.values()))), "all4": all4, "vs_avg": vs_avg,
                      "n": len(t["keys"])}
        print(f"   {split} (n = {len(t['keys'])}): pairwise same pick {min(pairs.values()):.3f}-{max(pairs.values()):.3f}, "
              f"mean {out[split]['mean_pair']:.3f}; all four orders agree on {all4:.3f} of items")
        print(f"      a single order vs the average (itself one of its four members; circular): "
              + ", ".join(f"{o} {v:.3f}" for o, v in vs_avg.items()))
        if split == "screen":
            golds = np.array(t["golds"])
            acc, nll = [], []
            for o in CYCLIC:
                q, cand = t["rows"][o][0], t["rows"][o][1]
                acc.append(float((pk[o] == golds).mean()))
                nll.append(float(np.mean([-np.log(max(q[n][c.index(g)], 1e-12)) for n, (c, g) in
                                          enumerate(zip(cand, golds)) if g in c])))
            q, cand = t["rows"]["average (4 cyclic)"][0], t["rows"]["average (4 cyclic)"][1]
            a_acc = float((pk["average (4 cyclic)"] == golds).mean())
            a_nll = float(np.mean([-np.log(max(q[n][c.index(g)], 1e-12)) for n, (c, g) in
                                   enumerate(zip(cand, golds)) if g in c]))
            out["screen"].update({"mean_acc": float(np.mean(acc)), "mean_nll": float(np.mean(nll)), "avg_acc": a_acc,
                                  "avg_nll": a_nll})
            print(f"      the 'average of 4' row is the score of the AVERAGED PROBABILITIES: accuracy {a_acc:.3f}, NLL {a_nll:.3f};"
                  f" the arithmetic mean of the four single-order scores is {np.mean(acc):.3f} / {np.mean(nll):.3f}")
            se = float(np.sqrt(a_acc * (1 - a_acc) / len(golds)))
            print(f"      standard error of an accuracy on {len(golds)} rows is about {se:.3f}, so +0.017 is within noise")
    return out


# ------------------------------------------------------------------------------------------------ B
def copies() -> dict:
    rule("B. Two copies of the same 27B weights: lab server vs cloud VM, dev (308 items, identical candidate sets)")
    lab, vm, jar = recs("qwen38_27b_k0"), recs("qwen38_27b_k0_vm"), recs("qwen38_27b_k0_jarvis")
    keys = sorted(k for k in lab if k[0] == "dev" and k in vm and lab[k]["cand"] == vm[k]["cand"])
    qa = np.array([lab[k]["q"] for k in keys])
    qb = np.array([vm[k]["q"] for k in keys])
    ag = qa.argmax(1) == qb.argmax(1)
    s = np.sort(qa, 1)
    margin = s[:, -1] - s[:, -2]
    terc = np.array_split(np.argsort(margin), 3)
    res = {"n": len(keys), "agree": float(ag.mean()), "dq": float(np.abs(qa - qb).max(1).mean()), "terciles": []}
    print(f"   n = {len(keys)}: same pick on {ag.mean():.3f} ({int((~ag).sum())} items differ); mean max |dq| {res['dq']:.4f}")
    print("   agreement by the gap between the lab copy's top two options (terciles, small to large):")
    for i, t in enumerate(terc):
        res["terciles"].append((float(margin[t].min()), float(margin[t].max()), float(ag[t].mean()), len(t)))
        print(f"      tercile {i + 1}: gap {margin[t].min():.2f}-{margin[t].max():.2f}, same pick {ag[t].mean():.3f} (n = {len(t)})")
    print(f"   median gap of the differing items {np.median(margin[~ag]):.3f}; of the agreeing items {np.median(margin[ag]):.3f}")
    pt = np.array([lab[k]["prompt_tokens"] for k in keys])
    print("   by prompt length (terciles, short to long): " + ", ".join(
        f"{int(pt[t].min())}-{int(pt[t].max())} tokens {ag[t].mean():.3f}" for t in np.array_split(np.argsort(pt), 3)))
    kk = sorted(k for k in jar if k[0] == "dev" and k in vm and jar[k]["cand"] == vm[k]["cand"])
    same = float(np.mean([np.allclose(jar[k]["lp"], vm[k]["lp"]) and jar[k]["gpu_sec"] == vm[k]["gpu_sec"] for k in kk]))
    print(f"   CAUTION: the cloud lane's first run ('jarvis') and its re-score ('vm') hold the very same dev records "
          f"(identical log-probabilities and timing stamps on {same:.3f} of {len(kk)} items), so their 100% agreement is "
          f"not an independent repeat; no clean same-copy repeat exists")
    return res


# ------------------------------------------------------------------------------------------------ C
def roles() -> dict:
    rule("C. E25b as plain rates (dev; pooled over Mistral-7B and Qwen3-8B unless stated)")
    df = E25.llm_table()
    out = {"strata": {}, "per_model": {}, "nq": {}}
    print("   share of answers that evade, when the President evaded / when the President answered:")
    for c in ("N", "P", "F", "B"):
        g = df[df.condition == c]
        a, b = g[g.pres_evade == 1].evade.mean(), g[g.pres_evade == 0].evade.mean()
        out["strata"][c] = (float(a), float(b))
        print(f"      {c}: {a:.3f} / {b:.3f}  (n = {int((g.pres_evade == 1).sum())} / {int((g.pres_evade == 0).sum())})")
    for m in sorted(df.model.unique()):
        for c in ("N", "P"):
            g = df[(df.condition == c) & (df.model == m)]
            out["per_model"][(m, c)] = (float(g[g.pres_evade == 1].evade.mean()), float(g[g.pres_evade == 0].evade.mean()))
            print(f"      {m:11} {c}: {out['per_model'][(m, c)][0]:.3f} / {out['per_model'][(m, c)][1]:.3f}")
    print("   do the neutral analyst (N) and the President-role model (P) evade on the same questions?")
    for m in sorted(df.model.unique()):
        n = df[(df.condition == "N") & (df.model == m)].set_index("src_i").evade
        p = df[(df.condition == "P") & (df.model == m)].set_index("src_i").evade
        j = n.index.intersection(p.index)
        n, p = n[j], p[j]
        chance = n.mean() * p.mean() + (1 - n.mean()) * (1 - p.mean())
        out["nq"][m] = (float((n == p).mean()), float(chance))
        print(f"      {m}: same evade/not label on {(n == p).mean():.3f} of {len(j)} questions (chance given the rates {chance:.3f})")
    return out


# ------------------------------------------------------------------------------------------------ D
def precision() -> dict:
    out = {}
    rule("D. E26: which route do FP8 / NF4 flips take, and what does each lose with the frozen parameters on both sides?")
    GOLD = E26A.GOLD
    n = GOLD.shape[0]

    def inset(p):
        return GOLD[np.arange(n), p]
    for site, prec in (("rtxpro", "fp8"), ("a4000", "nf4")):
        lb = np.load(E26A.E26 / site / "labels" / "bf16_dev_own.npz")
        lq = np.load(E26A.E26 / site / "labels" / f"{prec}_dev_own.npz")
        m1f, sf = lb["m1_pred"] != lq["m1_pred"], lb["pred"] != lq["pred"]
        db, dq = lb["defer"], lq["defer"]
        both, one, none = db & dq, db ^ dq, ~db & ~dq
        m1_s2 = [score_subtask2(z["m1_pred"], GOLD).macro_f1 for z in (lb, lq)]
        sy_s2 = [score_subtask2(z["pred"], GOLD).macro_f1 for z in (lb, lq)]
        r = {"m1_flips": int(m1f.sum()), "sys_flips": int(sf.sum()), "m1_s2": m1_s2, "sys_s2": sy_s2,
             "unscored_q": int((dq & ~lq["scored"]).sum()), "unscored_b": int((db & ~lb["scored"]).sum())}
        ia, ib = inset(lb["m1_pred"]), inset(lq["m1_pred"])
        r["m1_lose"], r["m1_gain"] = int((ia & ~ib).sum()), int((~ia & ib).sum())
        ia, ib = inset(lb["pred"]), inset(lq["pred"])
        r["sys_lose"], r["sys_gain"] = int((ia & ~ib).sum()), int((~ia & ib).sum())
        r["sys_inset"] = (float(inset(lb["pred"]).mean()), float(inset(lq["pred"]).mean()))
        out[prec] = r
        print(f"   {prec} vs bf16 on {site}: M1 decisions flipped {r['m1_flips']} (on deferred items: {int((m1f & db).sum())}, "
              f"on kept items: {int((m1f & ~db).sum())}); system decisions flipped {r['sys_flips']}; the deferral set "
              f"changes on {int((db != dq).sum())} items")
        print(f"      system flips by route: deferred in both {int((sf & both).sum())} of {int(both.sum())}, "
              f"in one run only {int((sf & one).sum())} of {int(one.sum())}, kept in both {int((sf & none).sum())} of "
              f"{int(none.sum())}; flips shared with M1: {int((sf & m1f).sum())}")
        print(f"      flips that change reference-set membership: M1 loses {r['m1_lose']}, gains {r['m1_gain']}; "
              f"system loses {r['sys_lose']}, gains {r['sys_gain']}")
        print(f"      S2 with the frozen parameters (in-sample): M1 {m1_s2[0]:.3f} -> {m1_s2[1]:.3f} ({m1_s2[1] - m1_s2[0]:+.3f}); "
              f"system {sy_s2[0]:.3f} -> {sy_s2[1]:.3f} ({sy_s2[1] - sy_s2[0]:+.3f})"
              + (f"  [27B records missing for {r['unscored_q']} of {int(dq.sum())} deferred items in {prec}: lower bound]"
                 if r["unscored_q"] else ""))
    return out


def compute() -> dict:
    return {"order": order(), "copies": copies(), "roles": roles(), "precision": precision()}


def main() -> None:
    print("Final-report checks (descriptive, written after the results were known; docs/results_sources.md section 18).")
    compute()


if __name__ == "__main__":
    main()
