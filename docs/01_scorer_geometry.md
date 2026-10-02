# 01 — What the Subtask-2 scorer actually rewards

**Status:** correction to two committed documents.
**Reproduce:** `python -m evaluation.verify_scorer_geometry` (CPU, seconds; needs
numpy, pandas and pyarrow, since it reads `data/cache/*.parquet`). Every
number in §3, §6 and the §7 table is printed by that script, which calls the
project's own scorer (`qevasion/scoring.py`) rather than a reimplementation, so it
cannot drift from it. The measured outcomes in §5 and at the end of §7 come from
`code/decision_rules/decide.py` on the baseline runs (`raw/E0_decision_rules.txt`).

**Affects:**
- `higrec/reports/03_data_audit.md` §6 — the "+0.1166 headroom" figure
- `reasoning.md` P4 — "the score is *not* invariant to which member of G is predicted"
- contribution **C4**, whose motivation is re-founded rather than removed

The two documents corrected here belong to the team's earlier `higrec` analysis
track, which has since been archived and is no longer in this repository (it
remains in the git history). The quotations below are verbatim.

---

## 1. The claim under review

`03_data_audit.md` §6 reports:

> **Headroom = +0.1166 macro-F1** over the majority-vote rule, available to a
> system that picks *which* member of the reference set to name, with no change
> to the underlying model.

and `reasoning.md` P4 supports it by measuring, on `G = {Explicit,
Partial/half-answer}`, that naming `Explicit` scores 0.2222 while naming
`Partial/half-answer` scores 0.3333 — a difference of exactly 1/9 — concluding
that the choice among set members carries real value.

## 2. What the scorer does

For class `c`, item `i`, reference set `G_i`, single prediction `pred_i`:

```
TP_c = #{ i : pred_i = c  and  c in G_i }
FP_c = #{ i : pred_i = c  and  c not in G_i }
FN_c = #{ i : c in G_i    and  pred_i not in G_i }
```

Two properties of `scoring.py` matter below: a prediction outside `G_i` charges one
false negative to **every** member of `G_i`, and the macro average is always over
all nine declared classes, so a class that is never predicted contributes
`F1_c = 0` (zero-denominator divisions return 0).

Suppose **every** prediction lands in its reference set. Then:

- `FP_c = 0` for every `c`, because a prediction of `c` is only ever made where
  `c in G_i`, which makes it a true positive.
- `FN_c = 0` for every `c`, because no item misses its set, and `FN` is charged
  only on a miss.

So `F1_c = 2·TP_c / (2·TP_c + 0 + 0) = 1` for every class predicted at least
once, and `0` for every class never predicted. Hence the identity

```
macro-F1  =  (number of distinct classes named at least once) / 9
```

**Which member of `G` is named contributes nothing.** The only thing that matters
is landing in `G`, plus naming each class somewhere.

## 3. Measurement

### 3.1 The 0.1166 is the cost of abstaining, not of member choice

| predictor | macro-F1 |
|---|---|
| `derive_dev_consensus_leaf` majority vote, 33 no-majority items left OOV | **0.8255** |
| the same vector, those 33 resolved to an arbitrary member of their set | **1.0000** |

All 308 predictions land in `G` in the second row. The gap is entirely the 33
three-way-split items being left unpredicted — an abstention penalty, not a
selection prize.

### 3.2 Member choice is worth nothing

Naming a uniformly random member of each item's reference set, five independent
draws:

```
trial 0: 1.0000   trial 1: 1.0000   trial 2: 1.0000   trial 3: 1.0000   trial 4: 1.0000
```

### 3.3 The coverage identity, exact at every point

Restricting the allowed label space forces classes to go unnamed while staying
in-set wherever possible:

| allowed classes | in-set rate | classes named | macro-F1 | k/9 |
|---|---|---|---|---|
| 9 | 1.000 | 9 | 1.0000 | 1.0000 |
| 8 | 0.987 | 8 | 0.8889 | 0.8889 |
| 7 | 0.974 | 7 | 0.7778 | 0.7778 |
| 6 | 0.951 | 6 | 0.6667 | 0.6667 |
| 5 | 0.948 | 5 | 0.5556 | 0.5556 |
| 4 | 0.912 | 4 | 0.4444 | 0.4444 |
| 3 | 0.828 | 3 | 0.3333 | 0.3333 |
| 2 | 0.607 | 2 | 0.2222 | 0.2222 |

Exact to four decimals at every row.

## 4. Why P4's test gave the answer it did

P4 is **correct on the input it was given** and the assertion it pins is true of
that input. Its two-item example has `2/9` and `3/9` as its two outcomes, and
those are coverage counts: naming `Explicit` there causes two classes to be
named, naming `Partial/half-answer` causes three. The 1/9 difference is one
class appearing or not appearing, which is exactly §3.3's identity in miniature.

What does not follow is the generalisation. On a corpus where all nine classes
get named anyway — which is every realistic prediction vector on 308 items — the
choice among set members is inert. The inverted test in
`test_scorer_properties.py` should be re-stated as a **coverage** property, or it
will keep being read as licensing a selection mechanism that does not exist.

The related P3 observation, that per-class support is endogenous, is unaffected
and remains true.

## 5. What this does to C4

**It re-founds it on something simpler and stronger.** The decision problem is
exactly:

```
maximise  P( prediction lands in G )   subject to  every class being named somewhere
```

The per-class multipliers `lambda_c` are not selecting among members of `G`.
They are buying **coverage** — pushing rare classes over the line so those
classes contribute a nonzero `F1_c` at all. That is a sharper and more
defensible statement of the mechanism than the current derivation, and it
predicts where the gain should appear: entirely in the four rare classes, and
not at all in `Explicit`.

Two consequences for how C4 must be evaluated:

1. **Report in-set rate as a first-class metric**, beside macro-F1. It is the
   quantity the scorer is a monotone function of, and no published system
   reports it.
2. **Post-hoc logit adjustment is the mandatory control.** Coverage is exactly
   what a single temperature on the class prior buys (Menon et al., ICLR 2021).
   Any `lambda`-fitting result that is not compared against a tuned `tau` is
   uninterpretable, and that comparison is the first thing a reviewer will
   demand. `code/decision_rules/decide.py` runs both under the same nested CV.

### Outcome (measured after this section was written)

Both evaluations were run on the 5-seed baseline ensemble
(`02_experiment_log.md`, E4):

- **The one-scalar control won.** Logit adjustment: 0.365 → 0.438 held-out, with
  an in-fold/held-out gap of 0.018. Nine fitted multipliers: 0.394, gap 0.098 —
  they overfit 308 items.
- **The predicted location of the gain was only partly right.** With τ = 0.85
  fitted on all of dev (an in-sample view of the per-class effect, as in the log's
  §E4), the largest gains were in rare classes (`Partial/half-answer` 0.000 →
  0.462, `Claims ignorance` +0.215, `Clarification` +0.214), but the frequent
  `Implicit` also gained (+0.118), and the rare `Declining to answer` lost (−0.144)
  as did `Dodging` (−0.065). "Entirely in the rare classes" was too strong.

## 6. How much headroom is really there, and how to read a score as an in-set rate

Trivial baselines on dev, to calibrate:

| predictor | macro-F1 | in-set rate | classes named |
|---|---|---|---|
| always `Explicit` | 0.0604 | 0.373 | 1/9 |
| always `General` | 0.0596 | 0.367 | 1/9 |
| always `Dodging` | 0.0528 | 0.312 | 1/9 |
| always `Clarification` | 0.0028 | 0.013 | 1/9 |
| uniform random over 9 | 0.1101 | 0.166 | 9/9 |

Note the shape of it: naming a single class gets you an in-set rate of 0.37 and a
macro-F1 of 0.06, while naming classes at random gets you a *worse* in-set rate
(0.17) and a *better* macro-F1 (0.11). Coverage and in-set rate trade against
each other, and the metric rewards both.

The uniform-random row is a single draw (`default_rng(0)`). The mid-evaluation
floors (`docs/raw/mideval/mideval_analysis.txt` §B) average 20 draws and report
0.129 macro-F1 and 0.196 in-set rate for the same predictor; the 0.060 "always
majority" floor there is the `Explicit` row above.

macro-F1 as a function of in-set rate, for a predictor that hits `G` at rate `r`
and otherwise names a random wrong class, with all nine classes named
(20 resamples per row):

| in-set rate | macro-F1 |
|---|---|
| 1.00 | 1.0000 |
| 0.90 | 0.8163 |
| 0.80 | 0.6654 |
| 0.70 | 0.5492 |
| 0.60 | 0.4458 |
| 0.50 | 0.3544 |
| 0.40 | 0.2720 |

Reading **dev** scores off this curve by linear interpolation between rows (it is
built on dev, so only dev scores belong on it): our logit-adjusted ensemble
(0.438) sits near an in-set rate of 0.59, ChulaNLP's fine-tuned DeBERTa (0.46) near 0.61, and TeleAI's winning pipeline
(0.617) near 0.76. The curve is steep in that range — +0.10 in-set rate is worth roughly
+0.10 macro-F1 — so in-set rate is both the right target and a sensitive one.

(The curve assumes errors name a uniformly random non-member, which is
pessimistic: real errors concentrate on confusable neighbours, which sit in `G`
more often than chance. It is a reference line, not a prediction.)

## 7. The test set has two annotators, not three

| reference sets | mean \|G\| | items with >1 distinct label |
|---|---|---|
| 3-annotator (dev as published) | 1.701 | 59.4% |
| 2-annotator: annotator1+annotator2 | 1.539 | 53.9% |
| 2-annotator: annotator1+annotator3 | 1.383 | 38.3% |
| 2-annotator: annotator2+annotator3 | 1.373 | 37.3% |

The evaluation-phase test set carries two annotators per pair (organizers'
overview, Table 1). Smaller reference sets mean fewer chances to land in `G`, so
**every multi-reference number measured on dev is optimistic for the test set**,
and the multi-reference advantage may be a third smaller than dev suggests.
**Measured** on the baseline ensemble with `decide.py --drop-annotator`
(`raw/E0_decision_rules.txt`): plain argmax drops from 0.365 (3 annotators) to
0.329–0.347 (2 annotators), and logit adjustment from 0.438 to 0.391–0.401. The
adjustment's gain survives on every 2-annotator subset (+0.055 to +0.062).
