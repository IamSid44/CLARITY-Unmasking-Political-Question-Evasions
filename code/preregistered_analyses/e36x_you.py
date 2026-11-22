#!/usr/bin/env python3
"""E36x (exploratory, post hoc: written 2026-11-22 after E36 showed the confirmation ROR P/N on 'you' sub-questions only).
Not registered.

On the confirmation sample (CF labels, the three discovery models pooled), separately for sub-questions that address
'you' and those that do not:
  - evasion rate on sub-questions the President answered / evaded, per role;
  - OR(evade | President evaded) per role, and ROR(role / analyst), pooled 2x2 counts over the trio.
This shows whether the 'you' effect comes from the analyst evading such questions regardless of the President, or from
the roles tracking the President more closely. Interview bootstrap, 1,000 draws, seed 0. Added 2026-11-22 04:55 IST
after a review noted that no you-vs-other difference had been estimated: the ratio ROR(you) / ROR(other) per role, on
the same (paired) draws.

    python -m preregistered_analyses.e36x_you      # CPU, seconds; writes docs/raw/E36x_you.txt and runs/E34/e36x_you.json
"""

from __future__ import annotations

import json

import numpy as np

from preregistered_analyses.e34_analysis import TRIO, train_table
from qevasion.paths import ROOT, RUNS

N_BOOT = 1000


def main() -> None:
    d = train_table()
    d = d[(d.confirm == 1) & d.model.isin(TRIO)].reset_index(drop=True)
    urls = np.array(sorted(d.url.unique()))
    rng = np.random.default_rng(0)
    W = np.stack([np.bincount(np.searchsorted(urls, rng.choice(urls, len(urls))), minlength=len(urls))
                  for _ in range(N_BOOT)]).astype(float)
    ui = np.searchsorted(urls, d.url.to_numpy())
    k = (1 - d.pres_evade.to_numpy()) * 2 + (1 - d.evade_cf.to_numpy())  # 0 pe1y1, 1 pe1y0, 2 pe0y1, 3 pe0y0

    def cells(mask):
        C = np.zeros((len(urls), 4))
        np.add.at(C, (ui[mask], k[mask]), 1)
        return np.vstack([C.sum(0), W @ C])

    def ci(v):
        return [float(v[0]), *np.nanpercentile(v[1:], [2.5, 97.5]).tolist()]

    L = ["E36x (exploratory, post hoc): confirmation sample, CF, three models pooled; 'you' vs other sub-questions",
         f"interview bootstrap, {N_BOOT} draws", ""]
    res, rors = {}, {}
    for you in (1, 0):
        sub = (d.you == you).to_numpy()
        r, ors = {}, {}
        for c in ("N", "S", "G", "P"):
            cc = cells(sub & (d.condition == c).to_numpy())
            ors[c] = cc[:, 0] * cc[:, 3] / (cc[:, 1] * cc[:, 2])
            r[c] = {"rate_answered": ci(cc[:, 2] / (cc[:, 2] + cc[:, 3])), "rate_evaded": ci(cc[:, 0] / (cc[:, 0] + cc[:, 1])),
                    "or": ci(ors[c])}
            if c != "N":
                r[c]["ror_vs_N"] = ci(ors[c] / ors["N"])
                rors[(you, c)] = ors[c] / ors["N"]
        res[f"you={you}"] = r
        n_sq = int(d.loc[sub, "src_i"].nunique())
        L.append(f"you={you} ({n_sq} sub-questions)")
        L.append(f"   {'role':<5} {'rate, President answered':<26} {'rate, President evaded':<26} {'OR':<20} ROR vs analyst")
        for c, v in r.items():
            f = lambda x: f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"  # noqa: E731
            g = lambda x: f"{x[0]:.2f} [{x[1]:.2f}, {x[2]:.2f}]"  # noqa: E731
            L.append(f"   {c:<5} {f(v['rate_answered']):<26} {f(v['rate_evaded']):<26} {g(v['or']):<20} "
                     f"{g(v['ror_vs_N']) if 'ror_vs_N' in v else ''}")
        L.append("")
    L.append("ROR(you) / ROR(other), paired draws (is the role effect larger on 'you' sub-questions?)")
    res["you_over_other"] = {}
    for c in ("S", "G", "P"):
        v = rors[(1, c)] / rors[(0, c)]
        res["you_over_other"][c] = ci(v) + [float(np.mean(v[1:] <= 1))]
        x = res["you_over_other"][c]
        L.append(f"   {c}: {x[0]:.2f} [{x[1]:.2f}, {x[2]:.2f}], one-sided p {x[3]:.3f}")
    (ROOT / "docs" / "raw" / "E36x_you.txt").write_text("\n".join(L) + "\n", encoding="utf8")
    (RUNS / "E34" / "e36x_you.json").write_text(json.dumps(res, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
