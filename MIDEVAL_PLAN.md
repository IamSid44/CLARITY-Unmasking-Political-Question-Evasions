# The Decision-Rule Ceiling

**SemEval-2026 Task 6 (CLARITY) — Team Nier_ANLP — mid-evaluation plan**

> **Thesis.** The encoder ceiling on evasion classification is not a representation
> ceiling. The information is in the model; the argmax throws it away. This document
> re-points the project at that claim and lists the runs that test it before exams.

| | |
|---|---|
| Our baseline | **0.438** — DeBERTa-v3-large, dev Subtask 2 macro-F1 |
| Real target | **0.460** — ChulaNLP DeBERTa-large, dev Subtask 2 |
| Gap | **0.022**, not 0.07 (see §1) |
| Value of each class | **1/9 = 0.111** of macro-F1, however rare |

Everything below is *measured* from `dataset/QAEvasion.csv`, `dataset/Inter-Annotator/test_set.csv`,
and the repo's own scorer at `higrec/src/higrec/scoring/official.py`. Scorer semantics
were verified by execution, not read off prose.

---

## Contents

1. [We were aiming at the wrong number](#1-we-were-aiming-at-the-wrong-number)
2. [Does a hierarchical framework help?](#2-does-a-hierarchical-framework-help)
3. [Verdicts on the proposal's contributions](#3-verdicts-on-the-proposals-contributions)
4. [What we pitch instead](#4-what-we-pitch-instead)
5. [Run plan](#5-run-plan)
6. [Invariants — things that silently break the result](#6-invariants--things-that-silently-break-the-result)
7. [Reference numbers](#7-reference-numbers)

---

## 1. We were aiming at the wrong number

The 0.50–0.51 encoder ceiling quoted in the organizers' results paper is a **hidden-test**
figure. Our 0.438 is **dev**. These are not comparable measurements, and the direction of
the gap is not even consistent across teams.

| System | Backbone | Dev S2 | Test S2 | Δ |
|---|---|---:|---:|---:|
| **Ours** | DeBERTa-v3-large | **0.438** | — | — |
| ChulaNLP (2nd) | DeBERTa-large, fine-tuned | 0.460 | 0.430 | −0.03 |
| ChulaNLP (2nd) | RoBERTa top-5 → Kimi-K2 | 0.520 | 0.610 | +0.09 |
| TeleAI (1st) | DeepSeek-V3, CAMSR-CoT | 0.617 | 0.680 | +0.06 |
| Organizer baseline | Llama-70b, fine-tuned | — | 0.570 | — |

The mechanism is in the data: the test set has **two annotators per item**, dev has three.
Holding predictions fixed and shrinking the reference set from 3 annotators to 2 moves
macro-F1 by −0.015 to −0.082 on our own dev items, purely from the smaller target.

> **Our dev comparison point is 0.460, not 0.51.**
> We are 0.022 from the published encoder baseline, not 0.07. Every number in the mid-eval
> deck must be labelled `dev` or `test`. If anyone compares them directly, we make the
> correction first.

---

## 2. Does a hierarchical framework help?

Yes at exactly one split, and the data says where it stops.

To answer this properly we built the **human confusion matrix** from the 305 usable dev
triples — how often a second annotator picks each label given what the first picked — then
exhaustively searched all 255 binary partitions of the nine leaves.

Splits are scored by **class-balanced leakage**: the mean over classes of how much
co-annotation mass falls outside that class's group. Class-balanced, because macro-F1
weights all nine classes equally — a mass-weighted criterion would optimise the wrong thing
and would flatter rare classes for being rare.

### 2.1 Where humans actually disagree

P(a second annotator says *column* | the first says *row*). 305 dev items, row-normalised.
Diagonal = self-agreement.

| | Explic | Implic | Dodge | Genrl | Deflct | Partil | Declin | Claims | Clarif |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Explicit** | **.75** | .09 | .03 | .11 | .01 | .01 | .00 | .01 | .00 |
| **Implicit** | .13 | **.55** | .06 | .18 | .05 | .02 | .00 | .00 | .01 |
| **Dodging** | .04 | .06 | **.64** | .13 | .07 | .00 | .03 | .01 | .00 |
| **General** | .16 | .18 | .12 | **.41** | .09 | .02 | .01 | .01 | .00 |
| **Deflection** | .03 | .12 | .17 | .20 | **.42** | .02 | .03 | .01 | .00 |
| **Partial/half** | .19 | .22 | .03 | .17 | .08 | **.28** | .00 | .03 | .00 |
| **Declining** | .00 | .00 | .17 | .08 | .06 | .00 | **.64** | .06 | .00 |
| **Claims ign.** | .08 | .00 | .08 | .10 | .04 | .02 | .08 | **.60** | .00 |
| **Clarification** | .00 | .14 | .00 | .00 | .00 | .00 | .00 | .00 | **.86** |

Two clean blocks and one mush. **Clarification, Claims ignorance and Declining to answer**
barely leak — they carry overt lexical markers. **Explicit, Implicit, General, Deflection
and Dodging** form a single confusable mass where General alone confuses with four other
classes at comparable rates. **Partial/half-answer** has 0.28 self-agreement: humans agree
with each other on it less than a third of the time.

### 2.2 Best first split, exhaustive over all 255 partitions

Lower leakage is cleaner.

| Split | Leakage |
|---|---:|
| isolate Clarification | 0.017 |
| isolate Declining to answer | 0.058 |
| isolate Claims ignorance | 0.060 |
| isolate Partial/half-answer | 0.091 |
| **marker group `{Declining, Claims ignorance, Clarification}` vs the other 6** | **0.103** |
| *OFFICIAL clarity cut (published 3-way)* | *0.193* |
| *COVERAGE cut (our C2 hypothesis)* | *0.293* |
| *flat 9-way (no grouping)* | *0.429* |

### 2.3 There is no second split

Best partitions *within* the hard six, after the marker classes are removed.

| Split inside `{Explicit, Implicit, Dodging, General, Deflection, Partial}` | Leakage |
|---|---:|
| isolate Dodging | 0.123 |
| isolate Partial/half-answer | 0.133 |
| isolate Explicit | 0.137 |
| isolate Deflection | 0.146 |
| `{Dodging, Deflection}` vs rest | 0.185 |
| isolate Implicit | 0.190 |
| *flat within the six* | *0.479* |

Nothing inside the confusable six gets below 0.123, against 0.479 for flat. The structure
there is not a tree, so a deeper decision tree will spend GPU hours for nothing.

### 2.4 Three findings

**Finding 1 — the marker gate is strongly supported.**
`{Declining to answer, Claims ignorance, Clarification}` versus everything else leaks
**0.103** — roughly half the official clarity cut and a quarter of flat. These three are the
*easiest* classes in the taxonomy (human self-agreement 0.64 / 0.60 / 0.86), they carry overt
markers, and they have 356 training examples between them. They are almost certainly the ones
we are scoring near zero on, and they are worth up to 0.33 of macro-F1. TeleAI scored
0.870 / 0.818 / 1.000 on exactly these three.

**Finding 2 — deeper decomposition is unsupported, and that is the contribution.**
The organizers' results paper §5.3.3 already reports that *"systems implementing any form of
hierarchical decomposition outperformed those attempting direct nine-class inference."* So
"hierarchy helps" is the field's existing consensus, not a finding. Refining it into
**"hierarchy helps at one measurable split and the data refuses to support a second"** is the
contribution — and it partially contradicts that consensus.

**Finding 3 — C2's coverage cut looks likely to fail.**
The coverage-derived partition leaks **0.293** against the official cut's **0.193**. It is
*worse* aligned with human confusion structure. The mechanism is visible in the matrix: the
official cut buries General, Implicit, Dodging, Deflection and Partial together inside
Ambivalent so their mutual confusion is internal, while the coverage cut splits General away
from Implicit and Dodging and exposes its two largest confusions. We need this on the table
*before* `06_c2_preregistration.md` is written, not after.

---

## 3. Verdicts on the proposal's contributions

| | Contribution | Verdict | Reasoning |
|---|---|---|---|
| C1 | Five-attribute code over the leaves | **DROP** | The code is a deterministic function of the leaf label, so the factorized head has identical expressive capacity to a flat softmax — `factorized.py`'s own docstring concedes this. Keep the table only as a *hierarchy specification*. |
| C2 | Coverage axis raises agreement | **REFRAME** | Costs zero GPU, but §2 predicts it fails. Reframe as "which coarse cut matches human confusion structure?" The answer is neither published cut — it is marker-vs-rest. |
| C3 | Annotator model | **DEMOTE** | Still useful as the input to the decision rule, but its identifying assumption is broken: annotator assignment is confounded with president (Cramér's V = 0.270, p ≈ 3×10⁻¹⁰⁵). Not a headline claim. |
| C4 | Set-membership decision rule | **PROMOTE** | Zero GPU, post-hoc on saved probabilities, attacks the gap directly. This becomes the paper's spine. |
| C5 | Span / coverage module | **CUT** | Needs LLM-distilled supervision plus a shuffle control. No time before mid-eval. Revisit only if the pipeline lands early. |

---

## 4. What we pitch instead

Three claims that fit into one story: **this task is lost at the metric, not at the model.**
All three are measurable from artefacts we already have or runs we were going to do anyway.

### Claim A (primary) — the encoder ceiling is a decision-rule ceiling

Our own observation — the right answer sits in the model's top-3 — turned into the central
claim, measurable for free from saved logits:

- top-1 / top-3 / top-5 recall of the flat encoder
- macro-F1 of an oracle restricted to the model's top-3 → **decision-rule headroom**
- the gap from that oracle to 1.0 → **representation limit**
- how much of the headroom a tuned rule actually recovers

The organizers assert the 0.18 encoder–LLM gap needs *"the broader world knowledge and
generative reasoning capacity of large language models."* If top-3 recall is high, that
assertion is wrong — and no participant system tested it.

### Claim B — balanced softmax is the wrong correction for this metric

Balanced softmax and logit adjustment are Bayes-optimal for **balanced accuracy**. Macro-F1's
optimal plug-in rule is per-class thresholds tuned on held-out data (Koyejo et al. 2014;
Narasimhan et al. 2014) — already implemented in `setmembership.py`.

They are not fully redundant: balanced softmax changes *training*, so it changes calibration
as well as the boundary. That makes a clean ablation — *does training-time imbalance
correction add anything once the decision rule is tuned for the actual metric?*

The organizers report focal loss, class weighting and oversampling "proved insufficient on
their own." This explains why: they corrected the loss for the wrong functional.

### Claim C — a confusion-derived hierarchy, negative result included

The §2 analysis plus the trained gate. The honest half — no data support for a second split —
is what separates this from "we also tried hierarchy." Deliverable is the leakage table over
all 255 partitions, the marker-gate result, and the explicit statement that the confusable
six are not tree-structured.

---

## 5. Run plan

### Tier 0 — zero GPU, do before queuing anything

These decide what is worth training. Expected to move 0.438 → roughly 0.47–0.50 on their
own, which already clears the 0.460 target.

| # | Action | Output |
|---|---|---|
| 0.1 | Dump per-class P/R/F1 **and the saved probability matrix** from the finished baseline on dev | Confirms the marker classes are near zero. This one table says where the 0.438 is leaking. |
| 0.2 | Compute top-1/3/5 recall and oracle-within-top-k macro-F1 | Claim A. Decides whether the re-ranker is worth building at all. |
| 0.3 | Fit per-class thresholds with `nested_cv_thresholds` | The free score gain. Report the in-fold / out-of-fold gap *before* any headline number. |

> **Use nested CV, never the plain threshold fit.**
> On 305 dev items with Clarification at gold-set frequency 6, we measured an in-fold gain of
> +0.099 that **reversed to −0.065 out-of-fold** — a generalisation gap of +0.174, worse than
> not tuning at all. Tune on train CV folds; report dev as held-out.

### Tier 1 — the runs, in priority order

Apply threshold tuning to *every* run's output. It is free and doubles the information per
run. Use **3 seeds, not 5**, and record that change in the pre-registration before running.

| # | Run | Runs | Why this one |
|---|---|---:|---|
| 1 | **Hierarchical: marker gate → 6-way head** | 2 | Leakage 0.103. Targets the three classes costing us up to 0.33 of macro-F1. Node sizes: gate 356 vs 3,092; marker head 145 / 119 / 92. |
| 2 | Full original question added to the input | 1 | Cheap, and the confusion matrix backs the intuition: Deflection↔General at 0.20/0.09 is the largest off-diagonal pair, and "starts on topic then shifts" is only judgeable against the full question. |
| 3 | Balanced softmax | 1 | Now framed as the ablation for Claim B rather than as the fix. |
| 4 | Top-3 re-ranker *(gated)* | 2 | Run only if 0.2 shows high top-3 recall **and** 0.3 leaves headroom. If thresholding already captures it, the re-ranker is redundant — and we will have measured that, which is itself reportable. |

### Do not queue

Deeper hierarchy inside the confusable six · the C1 factorized head · the parameter-matched
capacity control (the head difference is 4,100 parameters on a 435M model — 0.0009%) ·
ModernBERT · the C5 span module.

---

## 6. Invariants — things that silently break the result

- **Dev and test are different measurements.** Dev has 3 annotators and 308 items; test has
  2 annotators and 237 items. Label every number.
- **Never tune thresholds on dev and report dev.** Clarification has gold-set frequency 6.
- **The scorer charges one false negative per member of the reference set**, not one per
  item. Miss a three-way-disagreement item and it costs 3 FNs.
- **Macro-F1 averages over all nine classes always.** A class never predicted scores F1 = 0
  and still counts in the denominator. Perfect predictions on a single-class corpus score
  exactly 1/9.
- **Per-class support under the multi-reference scorer is endogenous** — it depends on the
  predictions. Every per-class table must also report fixed gold-set frequency.
- **The `1.1` / `2.1` prefixes in the raw CSVs are not the clarity label.** They encode an
  older two-way reply/non-reply cut that groups Explicit with Implicit. Strip them before
  comparing.
- **Training items were not randomly assigned to annotators.** Assignment is by interview and
  confounded with president. Any annotator-effect claim must condition on that.

---

## 7. Reference numbers

### 7.1 Splits — what each file actually is

| Split | Source | Items | Labels |
|---|---|---:|---|
| Train | `dataset/QAEvasion.csv` | 3,448 | One `label` + `annotator_id` (85/86/89). 58 items double-annotated, 32 agreeing. |
| Dev | `dataset/Inter-Annotator/test_set.csv` | 317 → 308 | Three independent labels, no single gold for S2. 305 usable after dropping out-of-taxonomy. |
| Test | Codabench only | 237 | Two annotators per item. Not in this repo. |

### 7.2 Per-class figures used above

| Leaf | Train n | Dev gold-set freq | Human self-agreement | Official clarity |
|---|---:|---:|---:|---|
| Explicit | 1052 | 117 | 0.75 | Clear Reply |
| Dodging | 706 | 84 | 0.64 | Ambivalent |
| Implicit | 488 | 97 | 0.55 | Ambivalent |
| General | 386 | 111 | 0.41 | Ambivalent |
| Deflection | 381 | 49 | 0.42 | Ambivalent |
| **Declining to answer** | **145** | **17** | **0.64** | **Clear Non-Reply** |
| **Claims ignorance** | **119** | **14** | **0.60** | **Clear Non-Reply** |
| **Clarification** | **92** | **6** | **0.86** | **Clear Non-Reply** |
| Partial/half-answer | 79 | 14 | 0.28 | Ambivalent |

Bold rows are the marker gate's positive class. Note the inversion that drives Claim A: the
three *rarest* trainable classes have *higher* human agreement than General, Deflection and
Partial, which have 3–5× the training data.

### 7.3 Reference-set statistics

| | |
|---|---|
| dev mean \|G\| | 1.669 across 305 items — 43.0% unanimous, ~49% two-way, ~11% three-way |
| 2-annotator \|G\| | 1.348–1.515 depending on which pair — the regime the test set is in |
| same preds, 3→2 refs | macro-F1 moves −0.015 to −0.082 purely from the smaller target |
| q calibration | a 3-annotator q sums to ~2.16 against a test target of ~1.41 — compute q as a 2-annotator product at evaluation time |

### 7.4 The nine leaves and their raw codes

| Code | Leaf | Definition (organizers' Table 4) |
|---|---|---|
| 1.1 | Explicit | The information requested is explicitly stated (in the requested form) |
| 1.2 | Implicit | The information requested is given, but without being explicitly stated |
| 2.1 | Dodging | Ignoring the question altogether |
| 2.2 | Deflection | Starts on topic but shifts the focus and makes a different point than what is asked |
| 2.3 | Partial/half-answer | Offers only a specific component of the requested information |
| 2.4 | General | The information provided is too general / lacks the requested specificity |
| 2.6 | Declining to answer | Acknowledges the question but directly or indirectly refuses to answer |
| 2.7 | Claims ignorance | The answerer claims/admits not to know the answer themselves |
| 2.8 | Clarification | Does not provide the requested information and asks for clarification |

`2.5 Contradictory` and `2.9 Diffusion` appear in the raw CSVs only; they are outside the
published nine-class taxonomy and account for the 317 → 308 cleanup.
