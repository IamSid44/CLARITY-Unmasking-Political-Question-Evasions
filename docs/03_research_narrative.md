# 03 — The research narrative: from a first DeBERTa run to E13

This document tells the story of the encoder track as a chain of reasoning. Each
step covers five things: what in the data or the previous result **led to it**,
the **hypothesis**, the **setup** and how it differs from what came before, the
**result**, and the **analysis** of why it worked or failed. Most of the ideas
tested here did not work. They are kept because each failure ruled something out
and pointed to the next step.

All numbers are **dev macro-F1** (308 items, three annotators, multi-reference
scoring). The main metric is Subtask 2 (S2, 9 evasion types); Subtask 1 (S1, 3
clarity levels) is marked where it appears. "±" is the standard deviation over
seeds. The full record, with raw outputs, is in
[`02_experiment_log.md`](02_experiment_log.md) and [`raw/`](raw/).

---

## Contents

1. [The arc at a glance](#1-the-arc-at-a-glance)
2. [The task, and what the data says before any model is trained](#2-the-task-and-what-the-data-says-before-any-model-is-trained)
3. [Prologue: the analysis track that came first](#3-prologue-the-analysis-track-that-came-first)
4. [Step 0 — First-principles analysis](#4-step-0--first-principles-analysis)
5. [Steps 1–2 — Getting DeBERTa-v3-large to train](#5-steps-12--getting-deberta-v3-large-to-train)
6. [E0 — The baseline](#6-e0--the-baseline)
7. [E1–E3 — Reading the baseline](#7-e1e3--reading-the-baseline)
8. [E4 — Decision rules](#8-e4--decision-rules)
9. [E5 — Hierarchy, applied after training](#9-e5--hierarchy-applied-after-training)
10. [E6 — A second encoder as re-ranker](#10-e6--a-second-encoder-as-re-ranker)
11. [E7 — The gap to published systems](#11-e7--the-gap-to-published-systems)
12. [E8 — Full question + rebalancing loss](#12-e8--full-question--rebalancing-loss)
13. [E8b and E9 — One change at a time](#13-e8b-and-e9--one-change-at-a-time)
14. [E10 — Training length](#14-e10--training-length)
15. [E11 — Replication on new seeds](#15-e11--replication-on-new-seeds)
16. [E12 — Data-driven variants](#16-e12--data-driven-variants)
17. [E13 — Is the gap knowledge? An 8B LLM classifier](#17-e13--is-the-gap-knowledge-an-8b-llm-classifier)
18. [Threads across the story](#18-threads-across-the-story)
19. [Where it stands, and what comes next](#19-where-it-stands-and-what-comes-next)

---

## 1. The arc at a glance

| step | question asked | answer | key number (dev S2) |
|---|---|---|---|
| Step 0 | What does the metric reward, and where is the difficulty? | Landing among acceptable labels + naming every class; attribution is the hard part | — |
| Step 1 | Why does the model not learn at all? | Library loads DeBERTa-v3 in fp16; attention overflows | loss stuck at 1.887 |
| E0 | How good is a plain fine-tune? | Fine, but noisy across seeds | 0.337 ± 0.044 |
| E1 | Which classes fail? | Those humans disagree on, not the rare ones | `General` 0.12 |
| E2 | Does averaging seeds help? | Yes | 0.365 |
| E3 | Is the ranking better than the decision? | Much better: right label in top 3 for 90% | oracle 0.745 |
| E4 | Can a post-hoc rule close that gap? | One scalar helps a lot; nine overfit | **0.438** |
| E5 | Does routing by the taxonomy help? | No | 0.369 |
| E6 | Can a second encoder choose among the top 3? | No | 0.354 |
| E7 | Why is a comparable published model at 0.46? | It sees the full journalist question | — |
| E8 | Full question + Balanced Softmax + focal | Worse; first blamed on the loss | 0.379 |
| E8b | Full question alone | Worse, but undertrained | 0.343 |
| E9 | Hierarchy trained into the output head | No gain on S2 | 0.370 |
| E10 | Do 16 epochs rescue the full question? | Yes, per model; system 0.478 looked like a new best | 0.478 |
| E11 | Does it replicate on new seeds? | Per model yes (9/10 seeds); system gain within noise | final 0.405 / S1 0.648 |
| E12a | Train on all of train, keep the last epoch | Helps S2 per model (4/5 seeds) | 0.489 (5 seeds) |
| E12b | Hierarchy of specialist encoders | No | 0.390 (3 seeds) |
| E12c | Boundary experts for confused pairs | No effect | +0.002 per model |
| E12d | Model soup | Fails badly | 0.267 |

How each result led to the next:

```
Step 0: metric = in-set rate × coverage; difficulty = attribution
   │
   ├─► E0 baseline ─► E1 (fails where humans disagree) ─► E2 ensemble
   │                                                       │
   │                         E3: ranking ≫ decision ◄──────┘
   │                          │            │
   │                    E4 decision    E6 encoder re-ranker ✗
   │                    rule ✓ (0.438)
   │                          │
   │          organisers: "hierarchy helps" ─► E5 post-hoc routing ✗
   │
   └─► E7: ChulaNLP sees the full question
              │
              E8 (full q + loss) ✗ ─► E8b (full q alone) ✗ but undertrained
              │                        E9 (trained hierarchy head) ✗
              │
              E10: 16 epochs ─► full q helps per model; system 0.478?
              │
              E11: new seeds ─► per-model gain real; system gain is noise
              │
              E12 (from the error structure): all-data ✓, specialists ✗,
                                              pair experts ✗, soup ✗
```

---

## 2. The task, and what the data says before any model is trained

**The task.** SemEval-2026 Task 6 (CLARITY) uses question–answer pairs from U.S.
presidential interviews. Each item is a triple:

| field | content | median length |
|---|---|---|
| `interview_question` | the journalist's full turn, often several questions | 62 tokens |
| `question` | one sub-question split out of that turn (by GPT-3.5, checked by annotators) | 15 tokens |
| `interview_answer` | the president's entire reply | 266 tokens |

The label says how *that sub-question* was answered. Subtask 2 has 9 evasion
types; Subtask 1 has 3 clarity levels and is a fixed function of Subtask 2:

| clarity level (S1) | evasion types (S2) | share of train |
|---|---|---|
| Clear Reply | Explicit | 30.5% |
| Ambivalent | Implicit (14.2), Dodging (20.5), General (11.2), Deflection (11.0), Partial/half-answer (2.3) | 59.2% |
| Clear Non-Reply | Declining to answer (4.2), Claims ignorance (3.5), Clarification (2.7) | 10.3% |

**Splits.** Train 3,448 (one label per row) · dev 308 (three annotators per row)
· test 237 (two annotators per row; labels never released, Codabench closed).
**Every comparison in this track is therefore on dev.**

**Metric.** Macro-F1 over all classes, where a prediction counts as correct if it
matches **any** annotator's label.

**What the papers establish.**

- Annotators agree moderately on the 3-way level (Fleiss κ 0.64) and weakly on the
  9-way level (κ 0.48). The hardest pairs are `General` vs `Implicit` (κ 0.43) and
  `General` vs `Deflection` (κ 0.56) (dataset paper; task overview §3.1.4).
- The dataset paper found that predicting the 9-way label and mapping it up beats
  predicting the 3-way label directly, for most prompted and fine-tuned models.
  Its best model, a fine-tuned Llama-70B, is the official baseline: 0.82 S1 /
  0.57 S2 on test.
- In the competition, the winner (TeleAI, DeepSeek-V3, three-stage prompting) reached
  0.89 / 0.68 on test. The organisers name two strategies as the most effective:
  LLM prompting and using the taxonomy's hierarchy. Fine-tuned encoders plateaued
  around 0.50 on S2 on test, "regardless of scale, ensembling, or architectural
  augmentation".

That last point frames everything below. This track asks how far a carefully
built encoder can go, and why it stops where it does.

---

## 3. Prologue: the analysis track that came first

The team's first track (`higrec`, now archived and only in git history) began from
a proposal with four contributions, each aimed at a perceived gap:

| gap claimed | contribution |
|---|---|
| A flat 9-way output ignores the taxonomy | **C1** a factorised attribute code for the output |
| The 3-way cut is drawn along the wrong axis (clarity rather than coverage) | **C2** test which partition annotators agree on more |
| Annotator disagreement is thrown away | **C3** an annotator model |
| Everyone optimises the consensus label, but the scorer is multi-reference | **C4** a set-membership decision rule |

**C2 was pre-registered and refuted.** Annotators agree substantially more under the
official clarity cut than under the proposed coverage cut (Krippendorff α 0.623 vs
0.503; Δ −0.120, 95% CI [−0.189, −0.049], p = 0.0004), and the result held under all
three pre-declared perturbations. A data audit also put a figure on C4's
opportunity: "+0.1166 macro-F1 headroom" from choosing *which* acceptable label to
name.

That track built a frozen-backbone pipeline that trained only output heads, but
**no model was ever fine-tuned end to end**. Its own reasoning notes said a full
fine-tune "is not optional if the headline claim is to stand". This track
supplies that model. It starts from the simplest system that could work and adds
one measured change at a time. It reuses only `higrec`'s data loader, label
vocabulary and scorer replica, copied into `code/qevasion/` and checked to give
identical results.

---

## 4. Step 0 — First-principles analysis

**What led here.** Before spending GPU time, the track needed to know what the metric
rewards and where the difficulty lies. All of this step is CPU work on the data and
the scorer.

**Findings.**

1. **The metric rewards landing in the reference set, not matching the consensus.**
   If every prediction is one of the item's acceptable labels, every class predicted
   at least once has F1 = 1, so

   ```
   macro-F1 = (number of classes named at least once) / 9
   ```

   exactly. Which acceptable label is named contributes nothing. The audit's
   "+0.1166 headroom" was the cost of leaving 33 no-majority items unpredicted, an
   abstention penalty rather than a prize for choosing among labels. A uniformly
   random acceptable label scores 1.000 on five out of five draws
   ([`01_scorer_geometry.md`](01_scorer_geometry.md)).
2. **So two quantities matter: in-set rate** (does the prediction land in the
   reference set?) **and coverage** (how many classes are ever named?). They trade
   off: always predicting `Explicit` gives in-set 0.37 but macro-F1 0.06, while
   uniform random gives in-set 0.17 but macro-F1 0.11. From here on every run logs
   both.
3. **Most of the difficulty is attribution.** 69% of training rows share their
   answer with another sub-question, and 71.7% of those shared answers carry
   different labels. A model that ignores the sub-question can reach at most 0.753
   accuracy. The model has to work out which part of a long answer, if any,
   responds to *this* sub-question.
4. **`General` is where annotators disagree.** It is in 113 dev reference sets, and
   90.3% of them are non-unanimous. Long answers are more contested: the shortest
   quarter is 53.2% unanimous, the longest 26.0%.
5. **The test set is not like dev.** It has two annotators instead of three (mean
   reference-set size 1.37–1.54 vs 1.70), so fewer chances to land in-set, and its
   answers are four times shorter (median 71 tokens vs 362 on dev). Dev gains that
   come from multiple references or long answers will shrink on test.
6. **The GPT-3.5 rationales shipped with the data are too weak to learn from:**
   0.469 accuracy and 0.306 macro-F1 against the human labels.

**Consequences.** C4 was re-founded on a simpler basis: a decision rule cannot gain by
choosing among acceptable labels, only by buying **coverage**, that is, pushing
rare classes over the line so they score at all. That makes post-hoc logit
adjustment (one scalar on the class prior) the mandatory control for any
per-class rule. Dev results are treated as optimistic for test.

---

## 5. Steps 1–2 — Getting DeBERTa-v3-large to train

### Step 1 — A silent library bug

**What happened.** The first runs sat at a training loss of exactly **1.887**, the
entropy of the label distribution, and predicted `Explicit` for every item at every
learning rate.

**Diagnosis.**

| test | result | rules out |
|---|---|---|
| overfit 128 examples for 12 epochs | loss stays ~1.85 | a tuning problem |
| Hugging Face's own `DebertaV2ForSequenceClassification` | `nan` from step 1 | our code |
| same test with `roberta-base` | trains normally (2.12 → 0.88) | the data pipeline |
| dtype of loaded weights | DeBERTa-v3: **fp16**; RoBERTa: fp32 | — |

`transformers` 5.x loads a checkpoint in the dtype it was stored in (4.x always used
fp32). DeBERTa-v3's Hub weights are fp16, where its attention overflows. Gradient
clipping hid the `nan`s, so the model trained smoothly towards the label prior and
learned nothing from the input.

**Fix.** Load in fp32 and compute in bf16 under autocast. The model then overfits the
128 examples normally (2.11 → 1.03).

**Why it matters for the story.** It is the first instance of a recurring pattern: a
model that learns nothing looks the same as a model that has learned the prior.
This comes back in E8b.

### Step 2 — Learning rate

Three-epoch runs on identical seed, split and data order:

| learning rate | internal val, epochs 0 / 1 / 2 |
|---|---|
| **1e-5** (head 1e-4, layer-wise decay 0.95) | **0.094 / 0.267 / 0.285** |
| 2e-5 | 0.063 / 0.209 / — |

1e-5 is better at every epoch, consistent with DeBERTa-v3-large's known instability
at higher rates on small data. It is used for every run after this.

---

## 6. E0 — The baseline

**What led here.** The plan was the simplest system consistent with the papers:
one 9-way classifier, with Subtask 1 read off it through the fixed leaf→clarity
table. Both the dataset paper and the top two competition systems derived S1 this
way.

**Hypothesis.** A plainly fine-tuned DeBERTa-v3-large gives a usable reference point.
No novelty is expected. Its job is to be the control everything else is measured
against.

**Setup.**

| setting | value |
|---|---|
| input | `[CLS] sub-question [SEP] answer [SEP]`, 512 tokens (the full journalist question is **not** included) |
| head / loss | 9-way softmax, plain cross-entropy, deliberately no class weighting |
| optimisation | lr 1e-5, head 1e-4, layer decay 0.95, cosine schedule, 10% warmup, batch 16 |
| checkpoint | best epoch on a fixed 10% slice of **train**; dev is scored but never used to choose anything |
| decision | argmax |

**Result.** It took three attempts to get a sound baseline:

- **E0a (8 epochs, 2 seeds):** 0.371 and 0.326. Seed 0 named 2 of 9 classes at
  epoch 0 and 8 by epoch 2, so the early collapse corrected itself without class
  weighting. It peaked at epoch 3.
- **E0b (5 epochs, a mistake):** "it peaked at epoch 3, so 5 is enough". But the
  cosine schedule spans the whole run, so epoch 3 of 8 is at a learning rate a
  5-epoch run never trains at for long. Result: 0.223 ± 0.060, all seeds still
  improving at the end. Discarded.
- **E0c (8 epochs, 5 seeds), the baseline:**

| seed | 0 | 1 | 2 | 3 | 4 | mean |
|---|---|---|---|---|---|---|
| dev S2 | 0.389 | 0.339 | 0.279 | 0.311 | 0.368 | **0.337 ± 0.044** |
| dev S1 | 0.642 | 0.580 | 0.532 | 0.596 | 0.579 | **0.586 ± 0.039** |

In-set rate 0.529; coverage 8.2 of 9 classes.

**Analysis.** The seed spread (0.28 to 0.39) is large, as is typical when fine-tuning a
435M-parameter model on 3,400 examples. Any comparison needs several seeds and
paired differences. The E0b episode taught a lesson that recurs: **a model that has
not finished training can make any setting look bad.** The baseline was then frozen
and never re-tuned.

---

## 7. E1–E3 — Reading the baseline

These three steps train nothing. They read the baseline's saved probabilities to
find where the score is lost.

### E1 — Which classes fail?

**Hypothesis (the default view).** Rare classes fail, so class imbalance is the
problem, as most participants assumed.

**Result.** Per-class F1, mean over 5 seeds:

| class | F1 | dev support | share of train |
|---|---|---|---|
| Explicit | 0.679 | 102 | 30.5% |
| Dodging | 0.475 | 80 | 20.5% |
| Implicit | 0.419 | 75 | 14.2% |
| Claims ignorance | 0.400 | 12 | 3.5% |
| Declining to answer | 0.385 | 13 | 4.2% |
| Clarification | 0.345 | 4 | 2.7% |
| Deflection | 0.214 | 38 | 11.0% |
| **General** | **0.119** | 70 | 11.2% |
| **Partial/half-answer** | **0.000** | 8 | 2.3% |

**Analysis.** **Frequency does not predict failure.** `General` has the fourth-largest
support and the second-worst score, while three rare classes do comparatively well.
What predicts failure is human disagreement: the model fails where annotators do
(Step 0, finding 4). Only `Partial/half-answer` fits the imbalance story. It is
never predicted, which costs a flat 1/9 of the score through coverage.

### E2 — Ensembling seeds

**Hypothesis.** With seed spread this large, averaging seeds' probabilities should
gain for free.

**Result.** 5-seed ensemble 0.365 S2 / 0.596 S1 (+0.028 / +0.010); in-set 0.529 → 0.562.

**Analysis.** It works as expected. From here on, a "system" means an ensemble of
seeds.

### E3 — Is the ranking better than the decision?

**Result.** How often an acceptable label is in the ensemble's top k:

| k | 1 | 2 | 3 | 5 |
|---|---|---|---|---|
| hit rate | 0.562 | 0.779 | **0.896** | 0.994 |

A perfect chooser among the top 3 would score **0.745**, against 0.365 for argmax.

**Analysis.** The model nearly always has the right answer among its first few guesses;
the final choice is what fails. This is the most important number in the track. It
suggested two routes to improvement: **change how the probabilities are turned into
a decision** (E4), or **train something to choose among the top 3** (E6).

---

## 8. E4 — Decision rules

**What led here.** Two mismatches between argmax and the metric. First, macro-F1
weights every class equally, so rare classes deserve a boost (Step 0: coverage pays
directly). Second, dev's label mix differs from train's (`General` is 11.2% of train
and 18.6% of dev).

**Hypothesis.** A post-hoc rule over the saved probabilities can recover part of the
E3 gap. The theory says the per-class plug-in rule (nine multipliers, C4's
original form) is the macro-F1-optimal choice. Step 0 says one-scalar logit
adjustment is the control it must beat.

**Setup.** Applied to the 5-seed ensemble's dev probabilities. Every parameter is
fitted and scored by **nested cross-validation** on dev, so the scores reported are
held-out.

**Result.**

| rule | parameters | dev S2 (held-out) | in-fold minus held-out |
|---|---|---|---|
| R0 argmax | 0 | 0.365 | −0.001 |
| **R1 logit adjustment**, p / prior^τ (Menon et al., 2021) | 1 | **0.438** | +0.018 |
| R2 set membership | 0 | 0.365 | — |
| R3 per-class multipliers | 9 | 0.394 | **+0.098** |

On the three 2-annotator versions of dev, which approximate the test regime, R1
stays ahead (0.391–0.401 vs R0 0.329–0.347), while R3 falls *below* argmax on two
of three.

**Analysis.**

- **One scalar gives +0.073 and barely overfits.** The fitted τ = 0.85 recovers
  `Partial/half-answer` (0.000 → 0.462), `Claims ignorance` (+0.215) and
  `Clarification` (+0.214), at a cost to `Declining` (−0.144) and `Dodging` (−0.065).
- **The theoretically optimal rule loses.** Nine multipliers fitted on 308 items
  overfit (gap +0.098). At this sample size a one-parameter prior correction beats
  the metric-optimal plug-in rule, so C4 in its original form is not supported.
- **The gain is coverage, as Step 0 predicted, though not only in rare classes.**
  The frequent `Implicit` also gained (+0.118). "Entirely in the rare classes" was
  too strong.

Logit adjustment became part of every system from here on. This sets up a theme
that runs to E11: **the baseline's models lean on the class prior, and this rule
removes that lean.**

---

## 9. E5 — Hierarchy, applied after training

**What led here.** The organisers report that systems exploiting the taxonomy's
hierarchy beat flat ones, on both subtasks.

**Hypothesis.** Routing through the hierarchy (decide the clarity branch first, then
the leaf within it) improves the flat argmax.

**Setup.** No training. On the ensemble's probabilities: (a) *hard routing*: sum the
probabilities per clarity branch, pick the branch, then the best leaf within it;
(b) compare the taxonomy branch as a shortlist against the model's own top 3.

**Result.**

| | result |
|---|---|
| hard routing vs flat argmax, ensemble | 0.369 vs 0.365 (11.4% of predictions change) |
| same, single seed 0 | 0.348 vs 0.371 |
| taxonomy-branch shortlist | 3.60 labels, contains an acceptable one 81.2% of the time |
| model's top-3 shortlist | 3.00 labels, 89.6% |

**Analysis.** **No reliable effect**: +0.004 on the ensemble and −0.024 on a single
seed. The model's own shortlist is both smaller and more often right than the
taxonomy's, so routing through the taxonomy throws away information the flat
softmax already has. The published hierarchies were *prompting* pipelines, where a
smaller label set per prompt makes an LLM's job easier. A trained encoder scores
all nine classes at once and does not benefit the same way. This was the first of
three hierarchy tests (E5, E9, E12b), and all three came out the same.

---

## 10. E6 — A second encoder as re-ranker

**What led here.** E3: an acceptable label is in the top 3 for 89.6% of items, and the
oracle among them scores 0.745. A tuned rule captured only about a fifth of that
gap. Choosing among three close labels should be easier than among nine. It is
also the 2nd-place system's architecture (encoder shortlist, LLM chooses), with an
encoder in the LLM's slot.

**Hypothesis.** A cross-encoder that reads each candidate's *definition* alongside the
QA pair can choose among the top 3 better than argmax.

**Setup.**

```
sub-question + answer ──► ENCODER 1 (baseline) ──► top-3 candidates
for each candidate c:
  "Candidate: c — <one-line definition>. Question: <sub-question>" [SEP] <answer>
                         ──► ENCODER 2 (DeBERTa-v3-large) ──► one score;  highest wins
```

- The label is an *input*, not an output unit (monoBERT-style re-ranking).
- **Cross-fitted training candidates.** The full baseline has memorised its training
  rows, so its top 3 on a training item almost always starts with the gold label. A
  re-ranker trained on that would learn "pick the first". Training candidates
  therefore come from 5 copies of the baseline, each trained on 4/5 of train and
  predicting the held-out fifth.
- **Fusion variant:** re-ranker score + β · log p(baseline), with β chosen on a train
  slice from {0 … 64}. At large β it falls back to the baseline.

**Result.**

| attempt | epochs | re-ranker alone | + baseline prior |
|---|---|---|---|
| 1 | 3 | 0.242 ± 0.013 | 0.323 ± 0.008 (β hit the top of its grid) |
| **3 (final)** | **8** | **0.329 ± 0.039** | **0.354 ± 0.003** |
| *baseline ensemble, argmax* | | *0.365* | |

It also loses on a held-out train slice (0.300 vs 0.405), so the result is not a
dev artefact. It overrides the baseline on 37.7% of items, fixing 25 of its 135
errors and breaking 37 of its 173 correct answers.

**Analysis.** **Negative.** Training loss moved only 1.06 → 0.88 in 8 epochs, where 1.10
is random among three. The two models face different learning problems with the
same 3,400 examples:

- the flat classifier has one output per class, and every example trains all nine
  scores;
- the re-ranker must learn what each *definition means* and apply it by comparing
  text, which is semantic knowledge the encoder does not start with and cannot learn
  from this little data.

The 0.745 oracle shows the opportunity is real; the result shows that an encoder is
the wrong tool to take it. (Attempt 1 repeated E0b's under-training mistake, and its
fusion grid was too narrow to fall back to the baseline. Both were fixed for attempt
3.) The cross-fitted candidate set and the fusion code are kept for an LLM in the
same slot.

---

## 11. E7 — The gap to published systems

**What led here.** After E4–E6 the system stood at 0.438. Where does that sit, and what
explains the gap?

**Setup.** Comparison on **dev**, the only common ground, since test labels were never
released. Published dev numbers come from the TeleAI and ChulaNLP papers.

| system | dev S2 | dev S1 |
|---|---|---|
| TeleAI (1st), DeepSeek-V3 3-stage CoT | 0.617 | 0.812 |
| ChulaNLP (2nd), DeBERTa-large top-5 → Kimi-K2 *(the overview paper says RoBERTa; ChulaNLP's own paper says DeBERTa-large)* | 0.52 | 0.70 |
| **ChulaNLP, DeBERTa-large fine-tuned** | **0.46** | **0.65** |
| ours, 5-seed ensemble + logit adjustment | 0.438 | — |
| TeleAI, DeepSeek-V3 asked directly | 0.421 | 0.662 |
| **ours, single model** | **0.337 ± 0.044** | 0.586 |

**Analysis.** The like-for-like comparison is the single fine-tuned model: ChulaNLP's
0.46 against our 0.337. Their notebooks show four differences:

| | ChulaNLP | ours |
|---|---|---|
| input | full question + answer + sub-question | sub-question + answer |
| checkpoint choice | best step **scored on dev itself** | best epoch on a train slice |
| backbone | DeBERTa-large (v1) | DeBERTa-v3-large |
| training data | conflicting duplicates removed | all rows |

Choosing our epochs on dev, as they did, lifts our mean only to 0.342, so the
selection protocol explains little. **The biggest remaining difference is that their
model sees the full journalist question**, which is exactly the context Step 0's
attribution finding says the model needs to tell "vague answer to *this* question"
(`General`) from "answer to a *different* question" (`Dodging`).

---

## 12. E8 — Full question + rebalancing loss

**What led here.** E7 pointed to the input. E1 and E4 pointed to rare classes being
under-predicted by plain cross-entropy (`Partial` never predicted). The full
question does not fit in 512 tokens: 38% of train and 46% of dev would be cut.

**Hypothesis.** Three problems, one change each, bundled into one run:

| problem | change |
|---|---|
| the model cannot see what else was asked | add the full journalist question |
| it does not fit | 1024 tokens, with a 256-token question slot (covers the 95th percentile) |
| rare classes under-predicted; hard examples a minority of the loss | Balanced Softmax (Ren et al., 2020) + focal loss (γ = 2) |

Predicted: `General`, `Deflection` and `Partial` gain most. Balanced Softmax corrects
the prior during training, so the post-hoc τ should move toward 0 or below.

**Setup.** Identical to E0 except for the changes above:

```
[CLS] Sub-question: <sub-question> Full question: <interview_question> [SEP] <answer> [SEP]
```

The sub-question comes first on purpose: ChulaNLP placed it last, so
right-truncation removes it on exactly the long items where attribution is
hardest. 8 epochs, 5 seeds.

**Result.**

| | E0 | E8 |
|---|---|---|
| single model, S2 | 0.337 ± 0.044 | 0.322 ± **0.021** |
| single model, S1 | 0.586 | 0.539 |
| **ensemble + logit adjustment, S2** | **0.438** | **0.379** |
| in-set rate | 0.529 | 0.355 |
| classes named | 8.2 | 9 |

Per class, the loss did what it was designed to do, but far too hard: `General`
+0.148, `Partial` +0.169, `Clarification` +0.447 (4 items), while `Explicit` fell
0.679 → 0.319 and `Dodging` 0.475 → 0.081. Every seed went through an unstable phase
at epochs 1–3, swinging between single-class predictors (one seed predicted
`Partial` for 155 of 308 dev items at epoch 2).

**Analysis as first written.** "The loss over-corrects." Balanced Softmax already
removes the prior and focal loss up-weights hard, mostly rare-class examples, so
rare classes are corrected twice. The post-hoc rule could no longer help (+0.046,
best τ near 0). A length split seemed to clear the input: E8 trailed E0 by *less*
on long items.

**The design flaw, acknowledged in advance.** Three changes at once can show whether
the package helps but not which part did. That turned out to matter. **The
diagnosis above was overturned by E8b** (next section): the length split was the
wrong test, because the full question was added to *every* item, short ones
included.

---

## 13. E8b and E9 — One change at a time

**What led here.** E8 left the main question open: does the full question help on its
own? Separately, the hierarchy had only been tested post hoc (E5); a hierarchy
*trained into* the model was untested. To keep comparisons clean, each run changes
**exactly one thing relative to E0**, and the two run in parallel.

| | E0 | **E8b** | **E9** |
|---|---|---|---|
| input | sub-q + answer | **+ full question** (1024) | sub-q + answer |
| loss | CE | CE | CE |
| head | flat 9-way | flat 9-way | **hierarchical** |

### E8b — The full question, plain cross-entropy

**Hypothesis.** The full question helps, mostly in `General`, `Dodging` and
`Deflection`, and more on long items.

**Result.** Worse than the baseline, and worse than E8:

| | E0 | **E8b** | E8 |
|---|---|---|---|
| single model, S2 | 0.337 ± 0.044 | 0.282 ± 0.043 | 0.322 |
| single model, S1 | 0.586 | 0.482 | 0.539 |
| **single model + logit adjustment, S2** | 0.357 | **0.353** | 0.336 |
| ensemble + logit adjustment, S2 | 0.438 | 0.343 | 0.379 |
| final train loss | 0.57–1.09 | **1.32–1.50** | — |

The predicted gains appeared nowhere. `Dodging` was flat, `General` and `Deflection`
fell, and there was no long-item advantage.

**Analysis: this model is undertrained, not uninformed.**

- *Fit.* With the identical loss and schedule, E8b's final training loss sits where
  E0's was at epoch 3. A mid-run look had already shown it learning much more
  slowly (seed 4, internal val at epoch 3: 0.166 vs E0's 0.336).
- *What an undertrained model does.* Its seeds are less confident (mean top
  probability 0.445 vs 0.594), and their average prediction sits closer to the
  training label distribution. With less learned from the input, the class prior
  carries the decision. That is exactly what logit adjustment removes, and after it
  **a single E8b model is level with E0** (0.353 vs 0.357).
- *Why the ensemble does not recover.* E8b's seeds agree with each other more
  (pairwise 0.640 vs 0.579): seeds that have not moved far from the same start make
  the same mistakes, so averaging adds little.

**The correction to E8.** With the same full-question input, the E8 loss scores
*higher* than plain cross-entropy (+0.040 S2, +0.057 S1 per seed). So E8's deficit
came from the **input at 8 epochs**, and the loss partly *offset* it, most likely by
doing during training what logit adjustment does afterwards: undoing the
prior-leaning of an undertrained model. The per-class swing (from `Explicit` and
`Dodging` to rare classes) is still the loss's doing. The E8 section of the log is
annotated in place.

**What it changed.** The full question is not yet ruled out. It needs to be retested
at a training length where it can finish fitting (→ E10). Further loss ablations
were dropped, since the post-hoc rule does the same job for free.

### E9 — A trained hierarchical classifier

**Hypothesis.** E5 tested the hierarchy post hoc. Training it into the model, so the
coarse decision is part of every prediction, might help where post hoc did not.
Predicted effect: larger on S1 than on S2. S1 was the weakest area against
published systems (0.586 vs ChulaNLP 0.65), and a trained coarse head targets it
directly.

**Setup.** Two heads on the shared encoder, combined as

```
p(leaf) = p(level) × p(leaf | level)        (p(leaf | level): softmax over that level's leaves)
```

trained with cross-entropy on log p(leaf) = log p(level) + log p(leaf | level). Every
example trains both the 3-way decision and the choice within its level, with no
weight to tune. This was chosen over an auxiliary 3-way head, which would only
nudge the representation and be a weak test. Because a hierarchical model's natural
S1 answer is the most probable *level*, both read-outs (level of the top leaf, and
most probable level) are now reported for **every** model, the baseline included, so
the comparison cannot favour the new head.

**Result.**

| | E0 | E9 |
|---|---|---|
| single model, S2 | 0.337 ± 0.044 | 0.320 ± 0.038 |
| single model, S1 (most probable level) | 0.593 ± 0.025 | 0.607 ± **0.010** |
| ensemble, argmax, S2 | 0.365 | **0.295** (bootstrap 95% [−0.124, −0.015]) |
| **ensemble + logit adjustment, S2** | **0.438** | **0.370** |
| ensemble, S1 | 0.598 | 0.603 |

**Analysis.**

1. **S2: no gain, probably a loss.** Every S2 read-out is below E0. The ensemble's drop
   comes from where its predictions land: E9 predicts `Dodging` 42 times against
   E0's 59, and `Dodging` is acceptable on 96 dev items, so those moves mostly turn
   acceptable predictions into unacceptable ones. Single-model character
   (agreement, confidence, entropy) is unchanged.
2. **S1: steadier, not better.** The mean moves up within noise. What changes is the
   spread of the level read-out, which falls from 0.025 to 0.010: the coarse head
   makes the 3-way decision *consistent*, not more often right.
3. **A mechanism tested and rejected.** The obvious suspect was that factorising forces
   a level commitment, which should hurt on items whose acceptable labels span two
   levels (27% of dev). The loss is spread evenly instead (in-set 0.738 → 0.667 on
   cross-level items, 0.496 → 0.442 on the rest). There is no localised mechanism.
   A trained hierarchy, like the post-hoc one, simply adds nothing the flat head
   lacks.

**Scope of the conclusion.** E9 tests the weakest form of the hierarchy idea, a
factorised output layer on one encoder. The published wins came from *sequential
LLM stages*. The hierarchical head was excluded from what followed.

---

## 14. E10 — Training length

**What led here.** E8b is the one failure that could not be interpreted: the full
question might be useless, or it might need more training. The second is plausible
on its face. The input is up to twice as long, and most of the added text (the
other sub-questions) only becomes useful once the model learns to relate it to the
sub-question in front of it.

**Hypothesis.** At double the training (16 epochs), the full question beats the same
input without it, mainly in `General`, `Dodging` and `Deflection`. A control (E0 at
16 epochs) is needed so that a win cannot be credited to the longer schedule alone.

**Setup.** A 2 × 2 completed by E10's two arms:

| | 8 epochs | 16 epochs |
|---|---|---|
| sub-question + answer, 512 | E0 | **E10_base_16ep** (control) |
| + full question, 1024 | E8b | **E10_fullq_16ep** |

16 rather than 12: E8b was about four epochs behind, and 16 leaves room to see the
curve flatten. Warmup stays at 10% and the cosine schedule stretches over 16
epochs, identically in both arms.

Predictions written before the run: (1) the full-question model's final train loss
reaches E0's level (≤ 1.1); if not, slowness is a property of the input. (2)
Full question > control. (3) The control is within noise of E0.

**Result.**

| | E0 | E10_base_16ep | **E10_fullq_16ep** |
|---|---|---|---|
| single model, S2 | 0.337 ± 0.044 | 0.377 ± 0.020 | **0.387 ± 0.036** |
| single model, S1 | 0.586 | 0.604 | **0.631** |
| ensemble, argmax, S2 | 0.365 | 0.380 | **0.416** |
| **ensemble + logit adjustment, S2** | 0.438 | 0.402 | **0.478** |
| same, averaged over 10 CV splits | 0.446 | 0.403 | 0.472 |
| final train loss | 0.57–1.09 | 0.06–0.30 | 0.56–0.78 |

The 2 × 2 (ensemble + rule):

| | 8 epochs | 16 epochs |
|---|---|---|
| sub-question + answer | 0.438 | 0.402 |
| + full question | 0.343 | **0.478** |

**Analysis.**

- **The full question's effect depends on training length:** −0.095 at 8 epochs and
  +0.076 at 16. This is what "undertrained" predicted. Prediction 1 held (train loss
  0.56–0.78).
- **Longer training alone makes a better, steadier single model** (+0.040, spread
  halved), and the weakest E0 seeds gain most (seed 2: +0.119). Part of E0's seed
  variance was seeds that stopped before they had finished learning.
- **But the control does not improve the final system** (0.402 vs 0.438), and the reason
  is the central insight of this phase. The control's gains are in the rare classes
  (`General` 0.119 → 0.295, `Claims ignorance` +0.126, `Partial` 0.000 → 0.090),
  which is exactly what logit adjustment was supplying post hoc. **Longer training
  and the decision rule fix the same thing**; the rule now adds +0.022 instead of
  +0.073.
- **Why the full question then gains at the system level.** Both 16-epoch arms have
  more diverse seeds than E0. The control has nearly memorised its training set
  (loss 0.06–0.30) and is badly overconfident (top probability 0.728 vs in-set
  0.526), so averaging and the prior correction have little to work with. The
  full-question model fits less tightly (top probability 0.638 vs in-set 0.540) and
  leaves the rule more to do (+0.061). One plausible reading, not tested, is that
  the harder input acts as a regulariser.
- **Per class**, the full question's gain over the control is clearest in `Dodging`
  (0.379 → 0.478), and weak in `General` and `Deflection`. By length, the gain is as
  large on short items as on long ones the model saw in full, and absent on long
  items that were still cut.

**How solid was 0.478?** Against the 16-epoch control: +0.075, bootstrap 95%
[+0.001, +0.148], just clear of zero. Against the 8-epoch baseline: +0.042, 95%
[−0.049, +0.133], **not conclusive**. And the ensemble-size curve showed the 5-seed
value sitting well above the 4-seed average (0.471 vs 0.426), so part of it may be a
lucky draw of seeds. That called for replication rather than a new component.

---

## 15. E11 — Replication on new seeds

**What led here.** E10's headline was promising but statistically thin. In addition,
an exploratory look at E10's predictions found that a 10-model ensemble mixing both
inputs scored 0.498. That hypothesis was formed on the same dev set it would be
judged on.

**Hypotheses, fixed before the runs.**

1. **Replication:** the full-question system beats the baseline system on fresh
   seeds 5–9.
2. **Single-model ordering:** full question 16 ep > base 16 ep > base 8 ep, on the new
   seeds.
3. **Mixed inputs:** 5 full-question + 5 base-16 models beat 10 full-question models
   (same size, so the test is about mixing).
4. **Ensemble size:** each system's 10-seed ensemble is at least its 5-seed average.

**The final system was chosen by a rule written in advance:** the 10-seed
full-question ensemble with logit adjustment, unless hypothesis 3 holds.

**Setup.** The three relevant configurations re-run on **seeds 5–9**, with settings
verified identical against each run's saved config. This gives each system a second
independent 5-seed ensemble and a pooled 10-seed one.

**Result.** The same system on two sets of seeds (5-seed ensemble + rule, dev S2):

| system | seeds 0–4 | seeds 5–9 | all 10 |
|---|---|---|---|
| baseline, 8 epochs | 0.438 | **0.356** | 0.428 |
| 16 epochs | 0.402 | 0.375 | 0.400 |
| full question, 16 epochs | 0.478 | **0.434** | 0.405 |

Single models over all 10 seeds, paired by seed:

| | dev S2 | dev S1 |
|---|---|---|
| baseline, 8 epochs | 0.315 ± 0.040 | 0.576 ± 0.030 |
| 16 epochs | 0.362 ± 0.026 | 0.601 ± 0.016 |
| full question, 16 epochs | **0.384 ± 0.030** | **0.614 ± 0.027** |
| 16 epochs − baseline | +0.047 (t = 3.4; 9/10 up) | +0.026 (8/10) |
| full question − 16 epochs | +0.022 (t = 2.0; 7/10) | +0.012 (6/10) |
| **full question, 16 epochs − baseline** | **+0.069 (t = 3.6; 9/10)** | **+0.038 (9/10)** |

Hypotheses: (1) held narrowly (0.434 vs 0.356, 95% [−0.007, +0.157]); (2) held on both
seed sets; (3) **failed** (0.403 vs 0.405), so the 0.498 was a feature of those
seeds; (4) held for both baseline-input systems and **failed for the full question**.
The baseline keeps gaining with every added seed (+0.087 from 1 to 10 models), while
the full-question system peaks around 5 models (+0.016 from 1 to 10).

**Analysis.**

- **A 5-seed ensemble cannot rank systems on 308 items.** The same baseline system
  scored 0.438 and 0.356 on two seed sets. Both headline numbers of the previous week
  (0.438 and E10's 0.478) were the favourable end of that range. From here on,
  system claims use 10 seeds, and per-seed paired comparisons carry the conclusions.
- **The robust result is at the level of the single model.** The full question plus
  16 epochs improves a single fine-tuned DeBERTa on 9 of 10 seeds, on both subtasks.
  Most of it comes from the longer training (+0.047); the full question adds a
  smaller +0.022.
- **At the system level the S2 gain does not survive.** Ensembling and logit
  adjustment lift the prior-leaning baseline more than they lift better-trained
  models (+0.109 vs −0.004 from the rule on 10-seed ensembles). The leading
  explanation for the full-question curve flattening is that its models are more
  confident relative to their accuracy, and averaging confident models changes less.
  This is untested.
- **The final system** (by the pre-registered rule) is the 10-seed full-question
  ensemble with logit adjustment: **dev S2 0.405, dev S1 0.648**. The 10-seed baseline
  system scores 0.428 / 0.601. The two cannot be separated on S2
  (95% [−0.101, +0.061]); on S1 the full-question system is ahead.

---

## 16. E12 — Data-driven variants

### Groundwork: what 20 trained models say

Before proposing anything, the existing models were analysed on CPU to find out
where a gain could still come from.

1. **The decision layer is saturated.** Three alternatives to logit adjustment, under
   the same nested CV: a hierarchical Non-Reply gate with a fitted threshold (+0.003
   to +0.009 per seed), prior-matching decoding via optimal transport (−0.005 to
   −0.007), and a coverage floor (±0.002). None beats the one-parameter rule
   reliably, so **any further gain must come from the models**.
2. **No extra supervision is hiding in the data.** Train has one label per row; only 26
   QA pairs repeat with conflicting labels. The metadata flags and GPT-3.5 fields are
   empty in the test set, so they cannot be inputs.
3. **The error structure** (10-seed full-question ensemble on the train slice; dev
   untouched):
   - 85% of errors fall *within* the six non-Non-Reply classes, 24 cross the
     Non-Reply line, and 2 fall within Non-Reply.
   - Clustering the confusion matrix splits the three Non-Reply classes off first,
     the same top cut TeleAI derived. But our failure is the opposite of theirs:
     they over-predicted Non-Reply, while we **miss** it (precision 0.69, recall
     0.57). Clear Non-Reply is our weakest S1 class (F1 0.44–0.52).
   - Within the six, errors concentrate in three pairs: **Implicit ↔ Dodging**,
     **Explicit ↔ Implicit** (which also crosses the S1 line), and **General ↔
     Deflection**.
   - TeleAI's own ablation: their hierarchy moved S1 a lot (0.710 → 0.811) but S2
     barely (0.490 → 0.503). Their S2 gain came from boundary examples for exactly
     such confusable pairs.
4. **Choosing the epoch on the train slice buys nothing.** The last epoch scores at
   least as well as the selected one (S2 +0.011 ± 0.020, 7/10), and the slice costs
   10% of the training data.

**Screening protocol.** Model-level effects of useful size were visible at 5 paired
seeds in E10, so each variant was **screened on 3 seeds** (paired with the
full-question model of the same seed) and **extended to 5 only if it gained ≥ +0.015
on S1 or S2 with ≥ 2 of 3 seeds up**.

Every variant uses the best model's settings (full question, 1024 tokens, 16
epochs).

### E12a — Train on all of train, keep the last epoch

**From groundwork 4:** epoch selection adds noise, not value, and the slice it needs
withholds 10% of train (training on all rows gives 11% more data).

**Hypothesis.** S2 +0.01 to +0.02 per model, S1 about unchanged.

**Setup.** `--val-frac 0 --select last`: all 3,448 rows, fixed 16 epochs, last epoch kept.
The only change from the full-question model.

**Result.**

| seed | 0 | 1 | 2 | 3 | 4 | mean |
|---|---|---|---|---|---|---|
| dev S2 | +0.071 | +0.035 | −0.035 | +0.025 | +0.053 | **+0.029 ± 0.040 (4/5)** |
| dev S1 | +0.041 | −0.008 | −0.034 | −0.020 | −0.066 | −0.017 ± 0.039 (1/5) |

| 5 seeds | full-question model | E12a |
|---|---|---|
| single model, S2 / S1 | 0.386 / 0.631 | **0.416** / 0.613 |
| ensemble, argmax, S2 | 0.416 | **0.481** (95% [−0.004, +0.142]) |
| ensemble + rule, S2 | 0.478 | **0.489** |
| ensemble, S1 | 0.631 | 0.634 |

**Analysis.** **The one E12 variant that helps.** More data and no noisy epoch selection
lift S2 per model and as an ensemble, slightly above the prediction. The cost is on
single-model S1, which is level as an ensemble. Given E11, 0.489 is reported as a
5-seed number, not as a new best system. It needs 10 seeds.

### E12b — A hierarchy of specialist encoders

**From groundwork 3:** the data's own top cut is Non-Reply vs the rest, and missed
Non-Replies are our main failure there. This is the third hierarchy test, now with
separate trained models rather than a post-hoc rule (E5) or a factorised head (E9).

**Hypothesis.** A dedicated Non-Reply gate raises Non-Reply recall above 0.57 and lifts
S1 through Clear Non-Reply; S2 gains 0 to +0.02. If the gate does not raise recall,
the hierarchy is not the fix.

**Setup.** Three encoders: a binary **gate** (Non-Reply vs other, all rows), a 6-way
**specialist** on non-Non-Reply rows, and a 3-way **specialist** on Non-Reply rows.
They are combined softly: p(leaf) = p(branch) × p(leaf | branch). A 2 × 2 ablation
on CPU isolates each part: {flat model's implied gate, dedicated gate} × {flat model
within each branch, specialists}, plus hard vs soft routing. The combination code
was checked to reproduce the flat model exactly when fed the flat model's own parts.

**Result.** Screening seeds 0–2, per seed vs the flat model:

| | S2 + rule | S1 | Non-Reply recall |
|---|---|---|---|
| dedicated gate + flat within branches | −0.059 (0/3) | −0.030 | −0.039 |
| **flat gate + specialists** | **+0.018 (2/3)** | **+0.018 (2/3)** | +0.010 |
| full hierarchy (dedicated gate + specialists) | −0.050 (0/3) | −0.011 | −0.039 |
| full hierarchy, hard routing | −0.059 (0/3) | −0.013 | −0.039 |

The specialists passed the screening bar and were extended to 5 seeds:

| flat gate + specialists, minus flat | seeds 0–2 | **seeds 0–4** |
|---|---|---|
| S2 + rule | +0.018 (2/3) | **−0.009 (2/5)** |
| S1 | +0.018 (2/3) | **+0.004 (2/5)** |

As 5-seed ensembles with the rule: S2 0.404 vs the flat model's 0.471.

**Analysis.** **Negative on both parts.**

- **The gate hurts.** It reaches a training loss of ~0.03, memorising a 10%-vs-90% split,
  so its probabilities sit near 0 or 1. Soft routing becomes hard routing (the two
  rows are nearly identical), and its mistakes are final. The prediction failed:
  recall went *down*. The flat model's own Non-Reply probability mass is a softer,
  better-calibrated gate.
- **The specialists add nothing measurable.** Seeds 3 and 4 reversed the screening gain.
  This is the screening rule working as intended: a +0.018 on three seeds was worth
  extending, and two more seeds showed it was noise.

### E12c — Boundary experts for the confused pairs

**From groundwork 3:** errors concentrate in three pairs. This is the encoder analogue of
TeleAI's boundary examples, which drove their S2 gain.

**Hypothesis (deliberately modest).** Small or no S2 gain. A null result would agree with
E6.

**Setup.** One binary encoder per pair (Implicit/Dodging, Explicit/Implicit,
General/Deflection), trained only on that pair's rows. Where a pair are an item's
top two, the pair's joint probability mass is re-split by its expert.

**Result (3 seeds).** S2 +0.002 (1/3), S2 + rule −0.004, S1 −0.004; on top of the
hierarchy −0.051.

**Analysis.** **No effect**, as predicted. A pair expert sees the same input as the flat
model and was trained on a subset of the same labels, so it rarely changes a
decision the flat model already makes between those two labels. This is the same
lesson as E6: a second encoder choosing among the first one's candidates has no
information the first lacked. TeleAI's boundary examples worked because an LLM
brings label knowledge from outside the training set.

### E12d — Model soup

**Hypothesis.** Averaging the *weights* of the 10 full-question models (Wortsman et al.,
2022) gives one model with part of the ensemble's gain: better than a single model,
worse than the ensemble.

**Setup.** Inference only: a uniform soup of all 10, and a greedy soup that adds a model
only if the train-slice score improves.

**Result.**

| | dev S2 | dev S1 |
|---|---|---|
| single model (mean of 10) | 0.384 | 0.614 |
| uniform soup | **0.267** | 0.471 |
| greedy soup | kept only its first member (0.381) | 0.582 |

Averaging any two models already lowered the held-out score (0.41 → 0.29–0.36).

**Analysis.** **Negative, below a single model.** Soups work when fine-tuned models share
their starting point and land in one loss basin; Wortsman et al. fine-tune from a
common initialisation, *including the head*. Ours differ by seed in the randomly
initialised classification head and in data order, so their weights are not
interchangeable. Averaging *outputs* (the seed ensemble) remains the way to combine
them.

### What E12 establishes

Of four variants chosen from the error analysis, one helps: **train on every row and
keep the last epoch**. The hierarchy has now been tested in its encoder form three
ways (post-hoc routing, factorised head, specialist encoders), and none beats the
flat full-question model. Pair-specific second opinions from encoders add nothing,
for the same reason as the re-ranker.

---

## 17. E13 — Is the gap knowledge? An 8B LLM classifier

**What led here.** §16 closed the encoder-side options. The decision layer is saturated, and
every attempt to choose better among the encoder's own top candidates failed. The mid-eval
analysis added two facts:
- 53 of the final ensemble's 137 errors are on items all three annotators agreed on;
- a human annotator scored against the other two reaches 0.684, against the model's 0.38.

So the gap is in the model. Two explanations remained: missing **knowledge** (world and language
knowledge a 0.4B encoder lacks, as in the Lieberman example) or missing **label meaning** (what
each type means, which 3,400 examples may not teach).

**Hypothesis.** If knowledge is the gap, a much larger pretrained model fine-tuned under the same
protocol scores clearly higher. The prediction was S2 0.43–0.50 per model, and that the gain
would fall mostly on the commitment boundary (Explicit ↔ Implicit, General ↔ Implicit/Explicit)
and on items annotators agreed on.

**Setup. One change from the best DeBERTa model: the backbone.**
- The model is Qwen3-8B-Base with LoRA (r = 16 on every linear layer) and a new 9-way head on the
  last token, at lr 1e-4 for 3 epochs, the epoch chosen on the train slice.
- Rows, slice, input text and budgets, loss and batch size are identical.
- `code/models/llm_classifier.py`, on rented RTX PRO 6000 GPUs (JarvisLabs).
- 10 seeds paired with DeBERTa's.
- Two single-change follow-ups: **12 epochs** (E13b) and **all of train** (E13c).

**Result.**

| dev | Qwen3-8B LoRA | DeBERTa | Δ |
|---|---|---|---|
| single model S2 (10 seeds) | **0.476 ± 0.043** | 0.384 ± 0.030 | **+0.092, 10/10 seeds** |
| single model S1 | **0.709 ± 0.031** | 0.614 ± 0.027 | **+0.095, 10/10** |
| system (10-seed ensemble + LA, nested CV) S2 | **0.543** | 0.405 | +0.136 [+0.054, +0.224] |
| system S1 | **0.746** | 0.648 | |
| system on 2-annotator reference sets | 0.524 | 0.382 | |

The follow-ups:
- **E13b, 12 epochs (3 seeds):** the slice F1 rises by +0.084 on 3 of 3 seeds, and the best
  epochs are 8–11. It is adopted by the registered rule. Dev S2 for those seeds is 0.543 ± 0.015.
- **E13c, all of train (10 seeds):** +0.017 per model (6/10 seeds). The system's +0.028 has an
  interval of [−0.026, +0.092].

**Analysis.**
- **Knowledge is a large part of the gap.**
  - Every seed improves.
  - The per-model gain exceeds everything the encoder track gained (+0.069).
  - As a single pass, the system passes TeleAI's fine-tuned 7B (0.495) and ChulaNLP's
    encoder + LLM cascade (0.52).
- **The "where" prediction is refuted.**
  - The gain is broad: 7 of 9 classes, largest in Claims ignorance (+0.40), Declining (+0.22),
    Dodging (+0.10).
  - It is similar on unanimous and contested items (in-set +0.076 and +0.105).
  - What the bigger model brings is better recognition of what kind of statement the answer is,
    not one sharpened boundary.
- **Undertraining again** (the §13–14 lesson). Under a cosine schedule the last epoch is always
  favoured, so "best epoch = last" proved nothing by itself. A 12-epoch run did: the slice keeps
  improving to epochs 8–11, though training loss is near zero by then.
- **The decision rule matters more, not less.** Raw Qwen leans on frequent classes: it predicts
  General 24 times, though General is in 113 reference sets. Logit adjustment adds +0.062 per
  model, against +0.005 for DeBERTa.
- **All of train behaves as it did for DeBERTa** (E12a): a small per-model gain that the 3-seed
  screen overstated (+0.039 there, +0.017 at 10 seeds).
- **What did not move:** Partial is never predicted, and General stays at F1 0.32.
- **The headroom is in choosing, now with a stronger chooser available.** An acceptable label is
  in Qwen's top 3 for 93% of items. The least-confident fifth of items is right 40% of the time,
  the most-confident fifth 87%.

---

## 18. Threads across the story

Read as separate experiments, most of this track is a list of failures. Read as
threads, the failures explain each other.

### Thread 1 — The class prior, corrected three ways

| mechanism | where | effect |
|---|---|---|
| post-hoc logit adjustment (1 scalar) | E4 | +0.073 on the baseline ensemble; became standard |
| Balanced Softmax + focal during training | E8 | over-corrects (`Explicit` 0.68 → 0.32) and spends the rule's gain |
| per-class multipliers (9 scalars) | E4 | overfit 308 items |
| longer training | E10, E11 | the model separates rare classes itself; the rule then adds ~nothing |

All four act on the same thing: the pull towards frequent classes that a
prior-leaning model has. On S2, **any one of them is enough, and together they add
nothing**. This is why a better model did not give a better system (E10 control,
E11), and why the post-hoc rule is the cheapest way to get the effect.

### Thread 2 — Undertraining that passes for "this idea doesn't work"

E0b (5 epochs), E6 attempt 1 (3 epochs) and E8/E8b (full question at 8 epochs) all
looked like negative results and were actually budget problems. The signature is
consistent: high final training loss, low confidence, predictions sitting close to
the class prior, and a large rescue from logit adjustment. E8's failure was first
blamed on the loss; a one-change control (E8b) found the real cause. Run reports
now show fit (final train loss, best epoch) next to every score.

### Thread 3 — Three hierarchies, one answer

| form | where | result |
|---|---|---|
| post-hoc routing through the clarity branch | E5 | no effect; the model's own top 3 is a better shortlist |
| factorised output head, p(level) × p(leaf \| level) | E9 | S2 down; S1 steadier, not better |
| gate + branch specialist encoders | E12b | gate memorises and routes hard; specialists level |

The organisers' strongest claim is that hierarchical decomposition separates
strong systems from weak ones. These results do not contradict it, but they
narrow it: **the benefit belongs to LLM prompting pipelines**, where a smaller label
set per prompt is easier to reason over. A trained encoder already scores all nine
classes jointly, and splitting that decision adds no information. TeleAI's own
ablation agrees: their hierarchy mostly moved S1, and their S2 gain came from
boundary examples.

### Thread 4 — The top-3 gap that encoders cannot close

E3 found an acceptable label in the top 3 for 90% of items (oracle 0.745). The
decision rule closes about a fifth of the gap. The re-ranker (E6) and the boundary
experts (E12c) close none of it, both for the same reason: a second encoder trained
on the same 3,400 examples has no knowledge the first lacked. Closing this gap
needs label understanding from outside the training data. This is the slot the
2nd-place system filled with an LLM, and the infrastructure for it (cross-fitted
candidates, fall-back fusion) is ready.

### Thread 5 — Measurement on a small dev set

308 items, multi-reference scoring, and large seed variance make single numbers
unreliable. The track's practices came from specific mistakes:

- paired per-seed comparisons, not means of unpaired runs;
- nested CV for every fitted parameter;
- 10 seeds for any system claim (E11: the same system scored 0.438 and 0.356);
- a screening bar fixed in advance (E12b's specialists passed at 3 seeds and failed
  at 5);
- predictions and final-system selection rules written before the runs, and earlier
  conclusions annotated in place when overturned (E8's cause, E10's 0.478).

---

## 19. Where it stands, and what comes next

**Status (2026-10-02, after E13).** The best single-pass model is the LoRA-tuned **Qwen3-8B
classifier**:
- single model: S2 0.476 ± 0.043, S1 0.709 ± 0.031 (10 seeds);
- system: **S2 0.543, S1 0.746** (10-seed ensemble + logit adjustment).

12 epochs are adopted for it, at 3 seeds so far. The DeBERTa track below remains the controlled
baseline it was built to be.

**After the mid-evaluation (2026-10-06).** Phase 2 has started. The cascade (modules M1–M4), its
protocol, the compute budget and the server procedure are in
[`05_phase2_plan_and_budget.md`](05_phase2_plan_and_budget.md). Its first step, **E14 (M1)**, is
registered in the experiment log. That step is the 12-epoch Qwen at 10 seeds, cross-fitting, and
candidate sets with uncertainty. A finding made on the train slice before E14 ran changes one exit
criterion: at 12 epochs, a candidate set needs 5–7 labels to cover the gold label for 90–95% of slice
items.

**Next steps, in order (plan at the mid-evaluation)** (the timeline in the Conclusion of `Report/Report.pdf`):
1. **12 epochs at 10 seeds** → the new single-pass system. Then all of train with 12 epochs, as
   its own test.
2. **Cascade, the label-meaning test.** Uncertain items, with Qwen's top-k, go to a larger open
   LLM with definitions, a confusion guide and boundary examples. Prompts are designed on the
   train slice. We sweep the deferral rate to give an accuracy-vs-calls curve.
3. **Distil** the cascade back into the single pass. **Measure** cost per million items.

**The encoder-track summary that follows was written at the end of E12** and is kept as it was.

**What reliably improved the model** (per seed, paired):

| step | dev S2 | dev S1 |
|---|---|---|
| baseline: sub-question + answer, 8 epochs (10 seeds) | 0.315 ± 0.040 | 0.576 ± 0.030 |
| + 16 epochs | 0.362 ± 0.026 | 0.601 ± 0.016 |
| + full journalist question, 1024 tokens | **0.384 ± 0.030** | **0.614 ± 0.027** |
| + all of train, last epoch (E12a, 5 seeds) | **+0.029 per model** | −0.017 per model |

**Final system** (chosen by a pre-registered rule): the 10-seed full-question ensemble
with logit adjustment, **dev S2 0.405, dev S1 0.648**. The 10-seed baseline system
(0.428 / 0.601) cannot be separated from it on S2; on S1 the full-question system is
ahead. For reference on dev: ChulaNLP's fine-tuned DeBERTa-large scores 0.46 / 0.65
(checkpoint chosen on dev), and TeleAI's pipeline 0.617 / 0.812.

**What the evidence says about the ceiling.** The encoder now knows the right answer is
among its top three 90% of the time, the decision layer is saturated, and every
encoder-side attempt to choose better (rules beyond one scalar, re-ranker, pair
experts, hierarchies) has failed for reasons that are understood. This agrees with
the organisers' observation that encoders plateau on Subtask 2: the remaining
distinctions (`General` vs `Implicit` vs `Deflection` vs `Dodging`) are the ones
humans disagree on, and resolving them seems to need knowledge the training set
does not contain.

**Next steps as planned at the end of E12** (kept for the record; step 2 became E13 and the cascade above).

1. **E12a on 10 seeds.** The all-data model is the one remaining encoder-side gain. It
   needs the standard 10-seed system test before it replaces the final system.
2. **An LLM in the re-ranking slot.** It would take the encoder's top-3 candidates
   (cross-fitted for training) and use the fusion that can fall back to the
   encoder. This is where the 0.745 oracle gap, the hierarchy and boundary examples
   are most likely to pay off, as they did for the top two systems.
