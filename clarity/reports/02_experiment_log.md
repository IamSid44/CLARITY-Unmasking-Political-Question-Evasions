# 02 — Experiment log

The encoder track from its first run to now, in the order things happened. Each
entry says **why** it was run, **what** was run, what **came out**, and what it
**changed** about the plan. Mistakes are recorded where they happened, because
several of them changed the plan as much as the results did.

All numbers are **dev Subtask-2 macro-F1** (308 items, three annotators,
multi-reference scoring) unless marked otherwise. "± " is the standard deviation
over seeds. Raw outputs behind the headline tables are in `reports/raw/`.

| | |
|---|---|
| Period | 2026-09-19 → 2026-09-26 |
| Hardware | 1× RTX PRO 6000 (96 GB), shared with other users' jobs |
| Backbone throughout | `microsoft/deberta-v3-large` (435M parameters) |
| Best result so far | **0.438** — 5-seed ensemble + post-hoc logit adjustment (E4) |
| Latest | E10 control (E0 trained 16 epochs instead of 8): a better, steadier single model (0.377 ± 0.020 vs 0.337 ± 0.044), but no gain once the decision rule is applied (0.402 vs 0.438) |
| Running | E10's full-question half (seeds 2 and 3 of 5), due ~13:45 on 2026-09-26; it decides whether the full question helps when trained long enough |

---

## Scoreboard

Every configuration so far, one row each; the sections below explain each row in
the order it was run. Single-model numbers are the mean ± std over 5 seeds.
"+ rule" is the 5-seed ensemble followed by logit adjustment, the form a final
system takes.

| id | what changed from the baseline | single model, S2 | single model, S1 | ensemble + rule, S2 | verdict |
|---|---|---|---|---|---|
| E0 | — (DeBERTa-v3-large, sub-question + answer, 8 epochs, cross-entropy) | 0.337 ± 0.044 | 0.586 ± 0.039 | **0.438** | **best final system** |
| E4 | per-class thresholds instead of the one-number rule | | | 0.394 | overfits 308 items |
| E5 | hard hierarchical routing, post hoc | | | 0.369 *(ensemble, no rule)* | no reliable effect |
| E6 | second encoder re-ranks the baseline's top 3 | 0.329 ± 0.039 *(re-ranker alone)* | | 0.354 *(re-ranker + baseline prior)* | negative; baseline ensemble alone is 0.365 |
| E8 | + full question, Balanced Softmax + focal loss | 0.322 ± 0.021 | 0.539 ± 0.030 | 0.379 | negative; cause revised after E8b |
| E8b | + full question | 0.282 ± 0.043 | 0.482 ± 0.039 | 0.343 | undertrained at 8 epochs → E10 |
| E9 | hierarchical output head | 0.320 ± 0.038 | 0.606 ± 0.024 | 0.370 | negative for S2; S1 steadier, not better |
| E10 control | 16 epochs instead of 8 | **0.377 ± 0.020** | 0.604 ± 0.012 | 0.402 | better single model; no gain in the final system |
| E10 | + full question, 16 epochs | *running* | | | due 2026-09-26 |
| — | ChulaNLP's fine-tuned DeBERTa-large (published; checkpoint chosen on dev) | 0.46 | 0.65 | | for reference |

Packaged submissions for E0 are in `../submissions/` (Codabench is closed).

---

## Contents

- [Scoreboard](#scoreboard)
- [Starting point](#starting-point)
- [Step 0 — First-principles analysis, before training anything](#step-0--first-principles-analysis-before-training-anything)
- [Step 1 — The model would not train (a library bug)](#step-1--the-model-would-not-train-a-library-bug)
- [Step 2 — Choosing the learning rate](#step-2--choosing-the-learning-rate)
- [E0 — The baseline](#e0--the-baseline)
- [E1 — Why the baseline is only "fine": per-class analysis](#e1--why-the-baseline-is-only-fine-per-class-analysis)
- [E2 — Ensembling the seeds](#e2--ensembling-the-seeds)
- [E3 — The ranking is better than the decision](#e3--the-ranking-is-better-than-the-decision)
- [E4 — Decision rules: the free gain](#e4--decision-rules-the-free-gain)
- [E5 — Does a hierarchy help?](#e5--does-a-hierarchy-help)
- [E6 — A second encoder as re-ranker (negative result)](#e6--a-second-encoder-as-re-ranker-negative-result)
- [E7 — Where we stand against published systems](#e7--where-we-stand-against-published-systems)
- [E8 — Full question + Balanced Softmax + focal loss: negative result](#e8--full-question--balanced-softmax--focal-loss-negative-result)
- [E8b and E9 — two single-change runs in parallel](#e8b-and-e9--two-single-change-runs-in-parallel-launched-2026-09-25)
- [E10 — Training length: is the full question undertrained?](#e10--training-length-is-the-full-question-undertrained-launched-2026-09-26)
- [Deferred](#deferred)

---

## Starting point

The team's earlier `higrec` track (since archived, and no longer in this
repository; it remains in the git history) had built a data layer, a replica of the official
scorer, an agreement analysis (C2, whose hypothesis was refuted), and a
frozen-backbone pipeline that trains only output heads. **No model had been
trained end to end.** The team's own reasoning notes that a full fine-tune "is not
optional if the headline claim is to stand."

This track supplies that model. It used `higrec` as a library for data loading,
the label vocabulary and the scorer; those three modules now live in
`clarity/qevasion/`, checked to give identical results.

The plan was deliberately simple: fine-tune an encoder with a 9-way output head
for Subtask 2 and derive Subtask 1 from it by the fixed leaf→clarity table (what
the 1st- and 2nd-place systems did). Everything else is measured against that.

---

## Step 0 — First-principles analysis, before training anything

**Why.** To know what the metric rewards and where the difficulty lies before
spending GPU time. All CPU, from the data and the scorer.

**Findings.**

1. **The metric rewards landing in the reference set, not matching the consensus.**
   If every prediction is one of the item's acceptable labels, every class that is
   predicted at least once scores F1 = 1, so
   `macro-F1 = (classes named at least once) / 9` exactly. Which acceptable label
   you name is worth nothing. This corrects `higrec/reports/03_data_audit.md` §6,
   whose "+0.1166 headroom" was the cost of leaving 33 items unpredicted.
   Full derivation and reproduction: `01_scorer_geometry.md`.
2. **Two quantities matter: in-set rate** (does the prediction land in the
   reference set?) **and coverage** (how many of the nine classes are ever
   named?). Every run from here on logs both.
3. **Most of the difficulty is attribution.** 69% of training rows share their
   answer with another sub-question; 71.7% of those shared answers carry different
   labels. A model that ignores the sub-question can reach at most 0.753 accuracy.
4. **`General` is where annotators disagree.** It appears in 113 dev reference
   sets and 90.3% of them are non-unanimous. Long answers are more contested: the
   shortest quarter is 53.2% unanimous, the longest 26.0%.
5. **The test set is not like dev.** Two annotators per item instead of three
   (mean reference-set size 1.37–1.54 vs 1.70), and answers are four times shorter
   (median 71 tokens vs 266 train, 362 dev; 8% over 512 tokens vs ~30%).
6. **The GPT-3.5 rationales shipped with the data are too weak to learn from:**
   0.469 accuracy, 0.306 macro-F1 against the human labels.

**Changed the plan.** In-set rate and coverage became standard diagnostics, and
dev results are treated as optimistic for test.

---

## Step 1 — The model would not train (a library bug)

**What happened.** The first runs sat at a training loss of exactly 1.887 — the
entropy of the label distribution — and predicted `Explicit` for every item, at
every learning rate tried.

**Diagnosis**, in order:

| test | result |
|---|---|
| overfit 128 examples for 12 epochs | could not (loss stayed ~1.85) — so not a tuning problem |
| HuggingFace's own `DebertaV2ForSequenceClassification` | `nan` loss from the first step — so not our code |
| same test with `roberta-base` | trains normally (2.12 → 0.88) |
| dtype of the loaded weights | DeBERTa-v3: **float16**; RoBERTa: float32 |

`transformers` 5.x loads a checkpoint in the dtype it is stored in (4.x always
used fp32), and DeBERTa-v3's Hub weights are fp16, where its attention
overflows. Gradient clipping hid the `nan`s and left a model that trained smoothly
to the label prior.

**Fix.** Load in fp32 and let autocast use bf16 for compute. Afterwards
DeBERTa-v3-large overfits the same 128 examples normally (2.11 → 1.03).

**Changed the plan.** Worth flagging to the team: `higrec/scripts/extract_features.py`
loads in pure bf16, which avoids the overflow but has no fp32 master weights.

---

## Step 2 — Choosing the learning rate

**What.** Three-epoch runs on the real data, same seed, split and data order.

| learning rate | internal val, epochs 0 / 1 / 2 |
|---|---|
| **1e-5** (head 1e-4, layer decay 0.95) | **0.094 / 0.267 / 0.285** |
| 2e-5 | 0.063 / 0.209 / — |

1e-5 is better at every epoch, consistent with DeBERTa-v3-large's known
instability at higher rates on small data. Used for everything after this.

---

## E0 — The baseline

**What.** DeBERTa-v3-large, input `[CLS] sub-question [SEP] answer [SEP]`
truncated at 512 tokens, 9-way softmax, plain cross-entropy (no class weighting,
by design), plain argmax. The best epoch is chosen on a fixed 10% slice of
*train*; dev is scored but never used for any choice. Note that the full
journalist question (`interview_question`) is **not** in the input.

### E0a — exploratory, 8 epochs, 2 seeds

| seed | chosen epoch | dev S2 | dev S1 |
|---|---|---|---|
| 0 | 3 of 8 | 0.371 | 0.616 |
| 1 | 6 of 8 | 0.326 | 0.555 |

Seed 0's trajectory showed two things. The model named only 2 of 9 classes at
epoch 0 and 8 by epoch 2, so the early collapse fixed itself and needed no class
weighting. And it scored best at epoch 3 and overfit after.

### E0b — 5 epochs: a mistake

Reasoning that "it peaked at epoch 3, so 5 epochs is enough", the next runs used
5 epochs. That was wrong. The learning rate follows a cosine schedule over the
*whole* run, so epoch 3 of an 8-epoch run is still at a high rate, which a 5-epoch
run never reaches.

| seed | final train loss | dev S2 |
|---|---|---|
| 0 | 1.29 | 0.277 |
| 1 | 1.40 | 0.232 |
| 2 | 1.48 | 0.159 |

Mean **0.223 ± 0.060**, all still improving at the last epoch. Stopped, archived,
and rerun at 8 epochs.

### E0c — final baseline: 8 epochs, 5 seeds

| seed | chosen epoch | internal val | dev S2 | dev S1 |
|---|---|---|---|---|
| 0 | 7 | 0.441 | 0.389 | 0.642 |
| 1 | 6 | 0.363 | 0.339 | 0.580 |
| 2 | 4 | 0.350 | 0.279 | 0.532 |
| 3 | 5 | 0.358 | 0.311 | 0.596 |
| 4 | 7 | 0.431 | 0.368 | 0.579 |
| **mean** | | 0.389 ± 0.044 | **0.337 ± 0.044** | **0.586 ± 0.039** |

In-set rate 0.529 ± 0.023; coverage 8.2 of 9 classes.

**Takeaways.**

- The spread between seeds (0.28 to 0.39) is large, as is typical when fine-tuning
  a large model on ~3,400 examples. Any comparison needs several seeds; a 3-seed
  mean has an error of about ±0.025.
- Two of five seeds chose their last epoch, so 8 epochs is sufficient but not
  generous.

---

## E1 — Why the baseline is only "fine": per-class analysis

Per-class F1, averaged over the 5 seeds (`raw/E0_analysis.txt`):

| class | F1 | support | share of train |
|---|---|---|---|
| Explicit | 0.679 | 102 | 30.5% |
| Dodging | 0.475 | 80 | 20.5% |
| Implicit | 0.419 | 75 | 14.2% |
| Claims ignorance | 0.400 | 12 | 3.5% |
| Declining to answer | 0.385 | 13 | 4.2% |
| Clarification | 0.345 | 4 *(too few to measure)* | 2.7% |
| Deflection | 0.214 | 38 | 11.0% |
| **General** | **0.119** | **70** | 11.2% |
| **Partial/half-answer** | **0.000** | 8 | 2.3% |

**Takeaway: class frequency does not predict failure.** `General` has the
fourth-largest support and the second-worst score, while three rare classes do
comparatively well. What predicts failure is human disagreement: the model fails
where annotators do. `Partial/half-answer` is never predicted at all, which costs
a flat 1/9 of the score.

---

## E2 — Ensembling the seeds

Averaging the five seeds' probabilities:

| | dev S2 | dev S1 | in-set |
|---|---|---|---|
| mean single seed | 0.337 | 0.586 | 0.529 |
| **5-seed ensemble** | **0.365** | **0.596** | **0.562** |

+0.028 for free, as expected given the seed variance.

---

## E3 — The ranking is better than the decision

How often a correct label is in the ensemble's top-k:

| k | 1 | 2 | 3 | 5 |
|---|---|---|---|---|
| hit rate | 0.562 | 0.779 | **0.896** | 0.994 |

A perfect chooser among the top 3 would score **0.745**. The model almost always
has the right answer among its first few guesses; it is the final choice that
fails. This finding motivated both E4 and E6.

---

## E4 — Decision rules: the free gain

**Why.** Two mismatches between `argmax` and the metric: macro-F1 values every
class equally (so rare classes deserve a boost), and dev's label mix differs from
train's (`General`: 11.2% of train, 18.6% of dev).

**What.** Rules applied after training to the saved probabilities, with every
parameter fitted and scored by **nested cross-validation** on dev, so the scores
are held-out. `raw/E0_decision_rules.txt`.

| rule | parameters | dev S2 (held-out) | in-fold minus held-out |
|---|---|---|---|
| R0 argmax | 0 | 0.365 | −0.001 |
| **R1 logit adjustment** — divide by prior^τ (Menon et al., ICLR 2021) | 1 | **0.438** | **+0.018** |
| R2 set membership | 0 | 0.365 (identical to R0 without per-annotator heads) | — |
| R3 per-class multipliers (plug-in macro-F1 rule) | 9 | 0.394 | **+0.098** |

On the three possible 2-annotator versions of dev (the test regime):

| reference sets | R0 | R1 | R3 |
|---|---|---|---|
| annotators 1+2 | 0.347 | **0.401** | 0.325 |
| annotators 1+3 | 0.335 | **0.391** | 0.316 |
| annotators 2+3 | 0.329 | **0.391** | 0.352 |

**Takeaways.**

- **One scalar gives +0.073**, holds under 2-annotator scoring (+0.055 to +0.062),
  and barely overfits.
- **The theoretically optimal rule for macro-F1 loses.** Nine per-class
  multipliers fitted on 308 items overfit (gap +0.098) and fall below plain argmax
  on two of the three 2-annotator versions. At this sample size, a one-parameter
  prior correction beats the metric-optimal plug-in rule.
- τ = 0.85 when fitted on all of dev. Per class, it recovers
  `Partial/half-answer` (0.000 → 0.462), `Claims ignorance` (+0.215) and
  `Clarification` (+0.214), at some cost to `Declining to answer` (−0.144) and
  `Dodging` (−0.065).

---

## E5 — Does a hierarchy help?

**Why.** The organizers report that systems exploiting the taxonomy's hierarchy
beat flat ones.

**What.** On the ensemble's probabilities: (a) *hard routing* — sum probabilities
per clarity branch, pick the branch, then the best leaf within it; (b) compare
narrowing to the taxonomy branch against narrowing to the model's own top-3.

| | result |
|---|---|
| hard routing vs flat argmax | 0.369 vs 0.365 (changes 11.4% of predictions) |
| same comparison, single seed 0 (earlier) | 0.348 vs 0.371 |
| taxonomy-branch shortlist | 3.60 labels, contains a correct one 81.2% of the time |
| model's top-3 shortlist | 3.00 labels, 89.6% |

**Takeaway.** Hard routing has no reliable effect (+0.004 on the ensemble, −0.024
on one seed), and the model's own shortlist is both smaller and more often right
than the taxonomy's. The top systems' hierarchies were *prompting* pipelines,
where a smaller label set per prompt makes an LLM's job easier; a trained encoder
scores all nine at once and does not benefit the same way.

---

## E6 — A second encoder as re-ranker (negative result)

Full write-up: `03_reranker_ablation.md`. In short:

**Why.** E3: a correct label is in the top-3 for 89.6% of items, and choosing among
three should be easier than among nine.

**What.** A second DeBERTa-v3-large reads
`Candidate: <label> — <one-line definition>. Question: <sub-question> [SEP] <answer>`
and outputs one score per candidate; the highest wins. Its training candidates
come from 5 cross-fitted copies of the baseline (each trained on 4/5 of train and
predicting the fifth), because the full baseline has memorised its training rows.

| attempt | epochs | re-ranker alone | + baseline prior | note |
|---|---|---|---|---|
| 1 | 3 | 0.242 ± 0.013 | 0.323 ± 0.008 | undertrained; prior weight hit the top of its grid (β = 2) |
| 2 | 3 | 0.238 (2 seeds) | 0.349 | a script flag silently overrode the epoch fix; stopped |
| **3** | **8** | **0.329 ± 0.039** | **0.354 ± 0.003** | final |
| *baseline ensemble* | | *0.365* | | |

On the internal train split too, the re-ranker scores 0.300 ± 0.010 against 0.405
for simply keeping the baseline's top choice — so this is not a dev artefact. It
does use the candidate text (it overrides the baseline on 37.7% of items), but it
rescues 25 of the baseline's 135 errors while breaking 37 of its 173 correct
answers.

**Takeaway.** Training loss moved only 1.06 → 0.88 in 8 epochs (1.10 = random
among three). Scoring a label from its description requires learning what each
label *means*, from 3,400 examples; the flat classifier instead gets a dedicated
output per class and sees every example for every class. At this data size the
re-ranking formulation is harder than the classification it replaces. The 0.745
oracle shows the opportunity is real; an encoder is the wrong tool to take it.
The 2nd-place system put an LLM in exactly this slot.

**Mistakes.** Attempt 1 repeated E0b's error (3 epochs), and its fusion grid
stopped at β = 2, which every seed chose. The grid now runs to 64, so fusion can
fall back to the baseline's ordering. Attempt 2 was lost to `run_pipeline.sh`
passing `--epochs 3` explicitly, overriding the corrected default.

---

## E7 — Where we stand against published systems

**The comparison has to be on dev.** Codabench is closed, and the test labels are
not public: the organizers' HuggingFace dataset (`ailsntua/QEvasion`, including
its `test_set/` and `full_test_set/` folders) and GitHub repository contain only
the 3,448 train and 308 dev items — zero overlap with the 237 test items.

Published **dev** numbers (Codabench dev leaderboard, 308 items, official scorer):

| system | dev S2 | dev S1 |
|---|---|---|
| TeleAI (1st), DeepSeek-V3, 3-stage CoT pipeline | 0.617 | 0.812 |
| ChulaNLP (2nd), RoBERTa top-5 → Kimi-K2 few-shot | 0.52 | 0.70 |
| ChulaNLP, Kimi-K2 over DeBERTa top-3, few-shot | 0.50 | 0.71 |
| TeleAI, DeepSeek-V3 single-step CoT | 0.490 | 0.710 |
| **ChulaNLP, DeBERTa-large fine-tuned** | **0.46** | **0.65** |
| ChulaNLP, Kimi-K2 zero-shot, all 9 labels | 0.44 | — |
| **ours, 5-seed ensemble + logit adjustment** | **0.438** | — |
| TeleAI, DeepSeek-V3 asked directly for the 9-way label | 0.421 | 0.662 |
| ours, 5-seed ensemble | 0.365 | 0.596 |
| **ours, single seed** | **0.337 ± 0.044** | 0.586 ± 0.039 |

Sources: TeleAI paper Table 2; ChulaNLP paper Tables 2–3. Our logit-adjusted
ensemble is level with a frontier LLM prompted directly, and below every system
that adds structure (CoT, shortlisting, staged routing) on top of an LLM.

**The like-for-like comparison is the single fine-tuned model:** ChulaNLP's
0.46 against our 0.337. Their notebooks show these differences:

| | ChulaNLP | ours |
|---|---|---|
| input | `Question: <full question>  Answer: <answer>  Subquestion: <sub-question>` | sub-question + answer only |
| checkpoint choice | best of many steps, **scored on dev itself** | best epoch on a train slice; dev never used |
| backbone | DeBERTa-large (v1) | DeBERTa-v3-large |
| training data | conflicting duplicate rows removed | all rows |

Choosing our epochs on dev, as they did, only lifts our mean to 0.342, so the
selection protocol explains little of the 0.12 gap. **The biggest remaining
difference is that their model sees the full question**, which is exactly what
Step 0's attribution finding says the model needs.

Caveats that apply to every row above:

- **Published dev numbers are partly tuned on dev.** ChulaNLP chose checkpoints on
  it; TeleAI chose prompt components on it (e.g. the number of few-shot examples,
  B = 10, picked from {8, 10, 12, 14} by dev score). Ours are held-out: nothing is
  chosen on dev except τ, and τ's score above is nested-CV.
- Published numbers use the official scorer; ours use a replica (validated against
  TeleAI's copy, not the official code).
- Published fine-tuned numbers are single runs, so their seed variance is unknown;
  ours spans 0.28–0.39 across seeds.

---

## E8 — Full question + Balanced Softmax + focal loss: negative result

*The plan below was written before the run (2026-09-25 morning) and is kept as it was; results follow it.*

**Why.** Three problems, one change each:

| problem, as measured | change |
|---|---|
| the model cannot tell "vague answer to this question" (`General`) from "answered a different question" (`Dodging`) without knowing what else was asked (Step 0 §3; the 0.46 model sees the full question, ours does not — E7) | put the full journalist question in the input |
| the full question does not fit: at 512 tokens, 38% of train and 46% of dev would be cut | raise the context to 1024 tokens (10% / 7% cut; test 1.7%) and the question slot to 256 tokens (covers the 95th percentile) |
| rare classes are under-predicted by plain cross-entropy (`Partial/half-answer` never predicted), and hard examples are a minority of the loss | Balanced Softmax (Ren et al., NeurIPS 2020) with focal loss on top (Lin et al., ICCV 2017, γ = 2) |

**What.** Identical to E0 in everything else: DeBERTa-v3-large, lr 1e-5, head lr
1e-4, layer decay 0.95, 8 epochs, batch 16, epoch chosen on the same 10% train
slice, 5 seeds. The input becomes

```
[CLS] Sub-question: <sub-question> Full question: <interview_question> [SEP] <answer> [SEP]
```

The sub-question comes first on purpose: ChulaNLP placed it last, so right-truncation
removes it on exactly the long items where attribution is hardest.

Config: `experiments/E8/`. Launch: `bash clarity/start.sh E8`.

**What will be measured**, against E0 on the same seeds:

- dev Subtask 2 and 1, per seed and as a 5-seed ensemble, plus in-set rate and
  coverage;
- per-class F1 — the prediction is that `General`, `Deflection` and
  `Partial/half-answer` gain most;
- the decision rules again (`decide.py`), with the logit-adjustment search widened
  to allow negative τ. Balanced Softmax already removes the class prior during
  training, so the post-hoc rule may now want some of it put back (τ < 0), and a
  τ near 0 would itself be evidence that the training-time correction did the
  post-hoc rule's job.

**Known limitation of the design.** Three changes are made at once (four, counting
the longer context), so E8 can show whether the package helps but not which part
helped. If it helps, the natural follow-up is to remove one change at a time.

**Cost.** Measured: 11.5 GB peak memory, ~105 min per run on an idle GPU (about 3×
the 512-token baseline, mostly from the longer sequences). The 5 seeds run as three
parallel lanes — two on GPU 1, one on GPU 0 — for roughly 5–6 hours in total.

---

### E8 results (2026-09-25, 5 seeds, 11:11–16:03)

**Headline: worse than the baseline, most clearly after the decision rule.** All
numbers held-out (decision rules under nested cross-validation).

| | E0 baseline | E8 | difference |
|---|---|---|---|
| single model, dev S2 | 0.337 ± 0.044 | 0.322 ± 0.021 | −0.015 |
| single model, dev S1 | 0.586 ± 0.039 | 0.539 ± 0.030 | −0.047 |
| 5-seed ensemble, argmax, dev S2 | 0.365 | 0.333 | −0.032 |
| **ensemble + logit adjustment, dev S2** | **0.438** | **0.379** | **−0.059** |
| same, on 2-annotator reference sets | 0.391–0.401 | 0.331–0.360 | −0.03 to −0.06 |
| in-set rate (single model) | 0.529 | 0.355 | −0.174 |
| classes named | 8.2 / 9 | 9 / 9 | |

Seed variance halved (±0.021 vs ±0.044), which is the one clear improvement.

**Where the score went — per class** (mean over 5 seeds, argmax):

| class | E0 | E8 | |
|---|---|---|---|
| Explicit | 0.679 | 0.319 | −0.360 |
| Dodging | 0.475 | 0.081 | −0.394 |
| Implicit | 0.419 | 0.375 | −0.044 |
| Claims ignorance | 0.400 | 0.350 | −0.050 |
| Declining to answer | 0.385 | 0.332 | −0.053 |
| Deflection | 0.214 | 0.210 | −0.004 |
| General | 0.119 | 0.267 | +0.148 |
| Partial/half-answer | 0.000 | 0.169 | +0.169 |
| Clarification | 0.345 | 0.792 | +0.447 *(4 items; too few to measure)* |

The loss did what Balanced Softmax is designed to do — it moved predictions from
frequent classes to rare ones — but far too hard: the two largest classes,
`Explicit` and `Dodging`, lost 0.36 and 0.39. Subtask 1 shows the same thing from
above: `Clear Reply` (= `Explicit`) fell from 0.560 to 0.335.

**The post-hoc rule could not undo it.** On the baseline, logit adjustment adds
+0.073; on E8 only +0.046, with a best τ near zero. Balanced Softmax had already
applied the prior correction during training, so the free post-hoc gain was spent,
and adding the prior back (τ < 0) did not restore the frequent classes.

**Training dynamics.** Every seed went through an unstable phase at epochs 1–3,
swinging between single-class predictors (seed 2 predicted `Partial/half-answer`
for 155 of 308 dev items at epoch 2, then `General` for 239 at epoch 3) before
breaking out at epoch 3–5. The baseline shows no such phase. This is the expected
failure mode of stacking a re-weighting on Balanced Softmax: focal loss up-weights
hard examples, which are mostly rare-class ones, so rare classes are corrected
twice.

**Was the longer context the problem?** No. Split by input length (in-set rate,
5-seed ensembles):

| dev items | n | E0 | E8 | difference |
|---|---|---|---|---|
| short — fit in 512 tokens for E0 | 207 | 0.570 | 0.362 | −0.208 |
| long — E0 had to cut the answer | 101 | 0.545 | 0.406 | −0.139 |
| … of which E8 saw in full | 81 | 0.593 | 0.432 | −0.161 |

E8 trails everywhere, but by *less* on long items — where the full question and
the longer context should matter. The deficit follows the loss change, not the
input change.

**What E8 cannot tell us** is whether the full question helped on its own; the
loss change costs more than any input gain could show. That is the attribution
limitation accepted when the three changes were bundled.

> **Revised after E8b (2026-09-26).** The overall diagnosis above is wrong. E8b —
> the same input with plain cross-entropy — scores *lower* than E8 (S2 −0.040,
> S1 −0.057 per seed), so the deficit against E0 came from the full-question input
> at 8 epochs, and the loss change partly offset it. What stands is the per-class
> account: the swing from `Explicit`/`Dodging` to rare classes is the loss's doing
> (E8b keeps `Explicit` at 0.630 and `Dodging` at 0.479). The length split above was
> the wrong test for input vs loss: the full question was added to every item, so
> short and long items both received the changed input. Details under
> [E8b results](#e8b-results-2026-09-26-5-seeds-finished-0132-below-the-baseline-and-undertrained).

**Operational notes.** 1024-token runs peaked at 14.9 GB and took ~92 min alone on
a GPU, ~200 min when two shared one. At 12:57 the lanes were rebalanced (one
restart; seeds 0 and 1 resumed from their epoch-3 checkpoints, losing ~2 min),
bringing completion forward from ~18:30 to 16:03.

**Next — E8b: full question with plain cross-entropy.** Same input and context as
E8, same loss as E0. E8b vs E0 isolates the full question; E8 vs E8b isolates the
cost of Balanced Softmax + focal.

---

## E8b and E9 — two single-change runs in parallel *(launched 2026-09-25)*

*Written before either run started.*

**Why two runs, and why now.** E8 changed the input and the loss together, and the
loss change sank it, so we still don't know whether the full question helps. The
hierarchy (E9) should sit on the best input, which that question decides — but
waiting for it would leave a GPU idle for most of a day. The way out is to change
**exactly one thing relative to E0 in each run**, so both comparisons are clean and
both GPUs are used. Whatever helps is then combined (E10).

| | E0 baseline | **E8b** | **E9** |
|---|---|---|---|
| input | sub-question + answer | **+ full question** | sub-question + answer |
| context | 512 tokens | **1024** (question slot 256) | 512 |
| loss | cross-entropy | cross-entropy | cross-entropy |
| output head | flat 9-way | flat 9-way | **hierarchical** |
| everything else (backbone, lr, schedule, 8 epochs, 5 seeds, epoch selection on the train slice) | — | same | same |
| GPU | | GPU 1 (two lanes) + one seed on GPU 0 | GPU 0 |

### E8b — the full question, plain cross-entropy

**Question.** Is the full journalist question worth adding?

**Reasoning.** 69% of training rows share their answer with other sub-questions, and
a model that cannot see them cannot tell a vague answer to *this* question
(`General`) from an answer to a *different* one (`Dodging`) — the two confusions that
cost the baseline most. ChulaNLP's fine-tuned DeBERTa sees the full question and
scores 0.46 against our 0.337. In E8 the longer input was not what hurt (E8 trailed
E0 by less on long items).

**Reads.** E8b vs E0 = the value of the full question (with the context it needs).
E8 vs E8b = the cost of Balanced Softmax + focal loss.

**Prediction.** The gain, if any, shows up in `General`, `Dodging` and
`Deflection`, and more on long items than short ones.

**In-progress observation (18:15, seed 4 at epoch 3).** E8b learns markedly more
slowly than E0 with the *same* loss:

| seed 4 | epoch 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| E0 train loss (CE) | 1.929 | 1.739 | 1.517 | 1.297 |
| E8b train loss (CE) | 1.956 | 1.856 | 1.785 | 1.687 |
| E0 internal val | 0.074 | 0.286 | 0.301 | 0.336 |
| E8b internal val | 0.066 | 0.070 | 0.164 | 0.166 |

The only difference between these two runs is the input (full question, 1024
tokens), so **part of E8's slow start was the input, not only the loss** — E8's
interpretation will be revisited once E8b finishes. The risk is that at a fixed
8-epoch budget E8b ends undertrained, which would make "the full question does not
help" and "the full question needs more training" look alike. The run continues as
pre-registered; undertraining will show as seeds choosing their last epoch while
still improving, which would justify a longer follow-up.

**Operational incident (18:20–18:28).** Lab-mates needed GPU 0, so E9 and E8b
seed 4 were stopped there and moved to GPU 1. In doing so I killed what I took to be
a stale tmux client; it was the tmux *server*, which hosts every session, so E8b
seeds 0 and 1 also died. All four runs resumed from their epoch checkpoints; the
cost was about 25 minutes of training on two seeds. Nothing is lost from the
results, and every resume is recorded in `logs/E8b.log` / `logs/E9.log`.

**Also discovered: GPU 1 is split into two MIG instances** (`2g.48gb`: 48 GB and
2 of 7 compute units each), and all our "GPU 1" runs have used one of them; the
other belongs to another user. This, not just two runs sharing a card, is why
GPU-1 epochs took ~25 min against ~11.5 min on GPU 0 — relevant to the E8 timing
notes above. From 18:28 all remaining E8b and E9 runs share that one slice
(E9 8.7 GB + two E8b runs 14.4 GB each).

**Rebalanced at 22:37** when GPU 0 became available again: E8b seeds 2 and 3 moved
to GPU 0 (shared with a lab-mate's 52 GB of jobs), seed 0 resumed its last two
epochs on the slice beside E9. Only E8b's session was restarted (`kill-session`);
E9 ran through it. Seed 3 had not yet completed an epoch, so it restarted from
scratch (~15 min of slice time lost).

### E9 — a trained hierarchical classifier

**Question.** Does training the taxonomy's hierarchy into the model help, where
applying it after training (E5) did not?

**Design.** Two heads on the shared encoder: one scores the 3 clarity levels, the
other the 9 leaves, combined as `p(leaf) = p(level) × p(leaf | level)` where
`p(leaf | level)` is a softmax over only that level's leaves. Trained with
cross-entropy on `log p(leaf)`, which equals `−log p(level) − log p(leaf | level)`:
every example trains the 3-way decision and the choice within its level, with no
weight to tune. (Checked by unit test: probabilities sum to 1, a level's leaves sum
to p(level), and the loss equals the two-term form exactly.)

**Why this design over the alternatives.**

- *Over an auxiliary 3-way head* (`--head hier`, already built): that keeps the
  9-way prediction unchanged and only nudges the representation — a weak test.
  Here the coarse decision is part of every prediction.
- *Over the "marker gate"* (`MIDEVAL_PLAN.md` §2): its premise — that those three
  classes score near zero — did not hold (E4: mean F1 0.365, level with the rest).
- *The official 3-way cut* **is** Subtask 1, and Subtask 1 is our weakest area
  relative to published systems (0.586 vs ChulaNLP 0.65, TeleAI 0.81). A trained
  coarse head targets it directly; so far Subtask 1 has only been a by-product of
  the 9-way head.

**Measurement change, applied to every model.** A hierarchical model's natural
Subtask-1 answer is the most probable *level*, which is not always the level of the
most probable *leaf*. Both read-outs are now reported for every model, the baseline
included (E0: 0.586 by leaf, 0.593 by level; E8: 0.539 / 0.458), so the comparison
cannot favour the new head.

**Prediction.** A larger effect on Subtask 1 than on Subtask 2.

### E9 results (2026-09-25, 5 seeds, finished 23:04): negative for Subtask 2, neutral for Subtask 1

Raw output: `reports/raw/E9_hier_analysis.txt`, `reports/raw/E9_hier_decision_rules.txt`.

| | E0 baseline | E9 hierarchical head | difference |
|---|---|---|---|
| single model, dev S2 | 0.337 ± 0.044 | 0.320 ± 0.038 | −0.018 (paired over seeds: ± 0.038) |
| single model, dev S1 (level of the top leaf) | 0.586 ± 0.039 | 0.606 ± 0.024 | +0.020 (paired: ± 0.044) |
| single model, dev S1 (most probable level) | 0.593 ± 0.025 | 0.607 ± **0.010** | +0.014 |
| in-set rate (single model) | 0.529 | 0.501 | −0.028 |
| 5-seed ensemble, argmax, dev S2 | 0.365 | 0.295 | **−0.069** |
| **ensemble + logit adjustment, dev S2** | **0.438** | **0.370** | **−0.068** |
| same, on 2-annotator reference sets | 0.391–0.401 | 0.329–0.365 | −0.03 to −0.06 |
| ensemble, dev S1 (most probable level) | 0.598 | 0.603 | +0.005 |

Per seed (same seeds, same train slice, so the pairs are directly comparable):

| seed | S2 E0 → E9 | S1 E0 → E9 | chosen epoch |
|---|---|---|---|
| 0 | 0.389 → 0.377 | 0.642 → 0.601 | 7 |
| 1 | 0.339 → 0.309 | 0.580 → 0.588 | 5 |
| 2 | 0.279 → 0.284 | 0.532 → 0.589 | 6 |
| 3 | 0.311 → 0.335 | 0.596 → 0.605 | 6 |
| 4 | 0.368 → 0.293 | 0.579 → 0.647 | 7 |

Per class (single-model mean), the changes that exceed the seed spread are in the
classes that sit in the same level as a larger neighbour: `Dodging` 0.475 → 0.388,
`Claims ignorance` 0.400 → 0.256; the rest move by less than their standard
deviation (`Explicit` 0.672, `Implicit` 0.405, `General` 0.139, `Deflection` 0.244,
`Declining` 0.356, `Partial` still 0.000).

**Is the ensemble drop real?** The per-seed difference (−0.018 ± 0.038) is well
within noise, but the ensemble difference is larger than the single-model one, which
is unusual. Two checks:

- *Not a change in the models' character.* Seed-to-seed agreement (0.578 vs 0.579),
  mean top probability (0.598 vs 0.594) and entropy (1.097 vs 1.104) are the same as
  E0's. The ensembles simply land differently: E9's predicts `Dodging` 42 times
  against E0's 59, and `Dodging` is acceptable on 96 dev items, so those moves mostly
  turn acceptable predictions into unacceptable ones (in-set 0.503 vs 0.562).
- *Paired bootstrap over dev items* (2,000 resamples, seeds held fixed): S2 ensemble
  E9 − E0 = −0.069, 95% interval [−0.124, −0.015]; S1 ensemble = +0.005,
  [−0.053, +0.066]. Holding seeds fixed understates the uncertainty, so read the S2
  interval as "probably worse, certainly not better".

**Verdict.**

1. **Subtask 2: no gain, probably a loss.** Every S2 read-out is below E0, and the
   one that becomes the final system (ensemble + logit adjustment) is 0.068 lower.
2. **Subtask 1: no reliable gain, but more stable.** The mean moves up by 0.014–0.020,
   inside the noise; the ensemble moves by +0.005. What does change is the spread of
   the level read-out across seeds, which falls from 0.025 to 0.010 — the coarse head
   makes the 3-way decision consistent, not more often right.
3. **The prediction ("a larger effect on Subtask 1 than Subtask 2") held only in
   direction:** S1 moved slightly up and S2 moved down, but neither S1 change is
   distinguishable from zero.

**Why it did not help — a hypothesis tested and rejected.** The obvious suspect was
the level cut: factorising `p(leaf) = p(level) × p(leaf | level)` makes the model
commit to a level, which should hurt on items whose acceptable labels span two
levels. It does not explain the result. Those items are 27% of dev (84 of 308), and
E9's ensemble loses about as much in-set rate on them (0.738 → 0.667) as on the
items whose acceptable labels all sit in one level (0.496 → 0.442). The loss is
spread across the dev set rather than concentrated where the hierarchy would bite,
so we do not have a mechanism; what we have is a trained hierarchy that, like the
post-hoc one in E5, does not add anything the flat head lacks. Note what this does and
does not test. The task overview (§5.3.3) credits "hierarchical decomposition" to
systems that ran the taxonomy as *sequential stages* — predict the coarse level,
then prompt again with only that branch's 2–3 labels, or route by confidence —
almost all with LLMs. E9 tests the weakest form of the idea, a factorised output
layer on one shared encoder, and the stronger form belongs with the LLM slot
(Deferred). On Subtask 1 the same paper's main finding — derive it from the 9-way
label rather than predict it directly — is what every run here already does.

**Consequence for E10.** The hierarchical head does not go into the combined system.
Its one real effect — a steadier 3-way decision — can be had for Subtask 1 without
changing the 9-way model, by reading Subtask 1 off the summed level probabilities
(E0 by level: 0.593 vs 0.586 by leaf), which is now reported for every run.

### E8b results (2026-09-26, 5 seeds, finished 01:32): below the baseline, and undertrained

Raw output: `reports/raw/E8b_fullq_ce_analysis.txt`, `…_decision_rules.txt`, and
`reports/raw/E8b_vs_E0_E8_compare.txt` (`compare_runs.py L0_large_base E8b_fullq_ce E8_fullq_bal_focal`).

| | E0 | **E8b** | E8 (for reference) |
|---|---|---|---|
| single model, dev S2 | 0.337 ± 0.044 | 0.282 ± 0.043 (paired vs E0: −0.056 ± 0.047) | 0.322 ± 0.021 |
| single model, dev S1 (level of the top leaf) | 0.586 ± 0.039 | 0.482 ± 0.039 (paired: −0.104 ± 0.055) | 0.539 ± 0.030 |
| single model, dev S1 (most probable level) | 0.593 ± 0.025 | 0.574 ± 0.053 | 0.458 |
| single model + logit adjustment, dev S2 | 0.357 ± 0.044 | **0.353 ± 0.024** | 0.336 ± 0.014 |
| in-set rate / classes named (single model) | 0.529 / 8.2 | 0.492 / 7.4 | 0.355 / 9 |
| 5-seed ensemble, argmax, dev S2 | 0.365 | 0.257 (bootstrap −0.106, 95% [−0.188, −0.027]) | 0.333 |
| **ensemble + logit adjustment, dev S2** | **0.438** | **0.343** | 0.379 |
| same, on 2-annotator reference sets | 0.391–0.401 | 0.319–0.369 | 0.331–0.360 |
| ensemble, dev S1 (most probable level) | 0.598 | 0.620 (bootstrap +0.022, [−0.057, +0.103]) | 0.481 |

| seed | S2 E0 → E8b | S1 E0 → E8b | chosen epoch (of 0–7) | final train loss |
|---|---|---|---|---|
| 0 | 0.389 → 0.314 | 0.642 → 0.513 | 6 | 1.335 |
| 1 | 0.339 → 0.252 | 0.580 → 0.438 | 6 | 1.389 |
| 2 | 0.279 → 0.304 | 0.532 → 0.505 | 7 | 1.354 |
| 3 | 0.311 → 0.220 | 0.596 → 0.440 | 5 | 1.497 |
| 4 | 0.368 → 0.318 | 0.579 → 0.514 | 7 | 1.317 |

**The predictions failed.** The gain was supposed to appear in `General`, `Dodging`
and `Deflection`, and more on long items. Per class (single-model mean F1, E0 → E8b):
`Dodging` 0.475 → 0.479 (flat), `General` 0.119 → 0.072, `Deflection` 0.213 → 0.080,
and the losses spread further — `Implicit` 0.419 → 0.236, `Declining` 0.385 → 0.170,
`Claims ignorance` 0.400 → 0.230. By length (ensemble in-set rate), E8b trails E0 by
0.063 on short items, 0.040 on long ones and 0.074 on the long items it saw in full:
no sign that the extra context pays off where it should.

**Why this is not yet a verdict on the full question: the model is undertrained.**

- *Fit.* Final train loss 1.32–1.50, against E0's 0.57–1.09 under the identical loss
  and schedule; E8b's last epoch sits where E0 was at epoch 3.
- *What the undertrained model does.* Its seeds are less confident (mean top
  probability 0.445 vs 0.594) and their average prediction sits closer to the
  training label distribution (KL 0.013 vs 0.029): with less learned from the input,
  the class prior carries more of the decision. That is exactly what logit
  adjustment removes — and after it, **single E8b models are level with E0**
  (0.353 ± 0.024 vs 0.357 ± 0.044). The raw-argmax deficit is mostly prior-leaning,
  not missing information.
- *Where the gap remains: the ensemble.* E8b's seeds agree with each other more than
  E0's (0.640 vs 0.579 pairwise), so averaging them adds less: E8b's ensemble with
  the rule scores 0.343, *below* its own single models, while E0's gains +0.08 from
  ensembling. Seeds that have not moved far from the same starting point make the
  same mistakes.
- *A caveat on the "still improving at the end" test.* Two of five seeds chose
  their last epoch — but so did two of E0's. With cosine decay the learning rate
  reaches zero at the end, so the validation curve flattens whether or not the
  model has finished learning; the train-loss gap is the stronger evidence.

**Verdict.** At 8 epochs the full question makes the model worse on both subtasks,
and the predicted gains did not appear anywhere. But the model had not finished
fitting, and once the class prior is corrected a single E8b model matches E0. So
this does not show that the full question is useless; it shows that it needs more
training than E0's input does. That is the question E10 was launched to answer.

**Correction to the E8 reading.** E8 was attributed to its loss change ("the loss
over-corrects"; "the deficit follows the loss change, not the input change"). E8b
reverses that. With the same full-question input, the E8 loss (Balanced Softmax +
focal) scores *higher* than plain cross-entropy: +0.040 ± 0.038 on S2 and
+0.057 ± 0.066 on S1, paired over seeds. **E8's deficit against E0 came from the
input at 8 epochs, and the loss change partly offset it**, most likely by doing
during training what logit adjustment does afterwards — undoing the prior-leaning of
an undertrained model. After the post-hoc rule the two losses are mixed (per seed
E8b 0.353 vs E8 0.336; ensembles E8 0.379 vs E8b 0.343), which is what one expects
if they are doing the same job. The E8 length split could not have caught this:
the full question was added to every item, short ones included, so short-vs-long
compares two parts of the same changed input rather than the input against the
loss. The E8 section above is annotated accordingly. The reason for dropping
further loss ablations — the post-hoc rule does the same job for free — survives;
the diagnosis of E8 does not.

### After both: E10 *(the plan as written before E8b and E9 ran)*

Combine what helped (e.g. full question + hierarchical head if both do), 5 seeds,
then the 5-seed ensemble with logit adjustment as the final system.

**Dropped:** further imbalance-loss ablations (Balanced Softmax alone, focal alone).
E8 showed training-time correction costs more than it gives, while the post-hoc rule
does the same job for free (+0.073); knowing which half hurt would not change the
next step.

*What actually happened:* neither run helped, so there was nothing to combine. E9
was a clean negative; E8b could not be read at all, because it was still learning
when its epochs ran out. E10 was redefined to answer that — next section.

---

## E10 — Training length: is the full question undertrained? *(launched 2026-09-26)*

*Written before either configuration started.*

**Why.** E8b is the only one of our changes whose failure we cannot interpret. With
the same loss, learning rate and schedule as E0, it learned far more slowly and
ended its 8 epochs much less fitted:

| after 8 epochs | E0 | E8b |
|---|---|---|
| final train loss (seeds 0, 1, 4 — finished at the time of writing) | 0.57 / 1.01 / 0.71 | 1.33 / 1.39 / 1.32 |
| E8b's train loss reached E0's final level at | — | never; its epoch 7 ≈ E0's epoch 3 |

A model that has not finished fitting can make a useful input look useless, so
"the full question does not help" and "the full question needs more training" are
still both open. The second is plausible on its face: the input is up to twice as
long, and most of the added text (the other sub-questions) is signal only once the
model has learned to relate it to the sub-question in front of it.

**Design.** Two configurations at **16 epochs** (double), identical except for the
input:

| | E0 | E8b | **E10_base_16ep** | **E10_fullq_16ep** |
|---|---|---|---|---|
| input | sub-question + answer | + full question | sub-question + answer | + full question |
| context | 512 | 1024 | 512 | 1024 |
| epochs | 8 | 8 | **16** | **16** |
| everything else | — | same | same | same |

- **E10_fullq_16ep vs E10_base_16ep** is the value of the full question at equal,
  adequate training — the question E8b was meant to answer.
- **E10_base_16ep vs E0** is what longer training does by itself. It is needed as the
  control (otherwise a win for the full question could be the longer schedule), and
  it is useful on its own: E0 was still improving at its last epoch on seeds 0 and 4.
- Warmup stays at 10% of all steps and the cosine decay stretches over 16 epochs,
  identically for both. The best epoch is still chosen on the 10% train slice, so a
  model that peaks early and then overfits is still scored at its peak.

**Why 16 and not 12.** E8b's loss at epoch 7 matched E0's at epoch 3, i.e. about four
epochs behind; 12 epochs would give it roughly E0's position at epoch 7, where two
E0 seeds were still improving. At 16 there is room to see the curve flatten, which is
what separates "needs more training" from "does not help".

**Predictions, written now.**

1. E10_fullq_16ep ends with a train loss comparable to E0's (≤ 1.1) — if not, the
   slow learning is a property of the input and more epochs are not the fix.
2. If the full question carries signal, E10_fullq_16ep beats E10_base_16ep, most in
   `General`, `Dodging` and `Deflection`, and more on long items than short ones
   (the E8b predictions, carried over).
3. E10_base_16ep is within noise of E0 (±0.04 per seed): E0's selected epochs
   (4–7 of 8) suggest it was near its peak.

**Reading the outcome.** If the full question still does not help at 16 epochs, it
is dropped from the encoder track: the extra context is then either not usable by
this model at this data size or not the missing information. If it helps, it
becomes part of the final system, with the 5-seed ensemble and logit adjustment on
top.

**Compute.** 10 runs, about 22 GPU-hours on a full card. Placement, with GPU 0
shared with a lab-mate and GPU 1 limited to our MIG slice:

| lane | runs | est. finish |
|---|---|---|
| GPU 1 slice | E10_base_16ep seeds 0–4 (~1.7 h each), then E10_fullq_16ep seed 4 (~4.5 h) | ~14:30 on 2026-09-26 |
| GPU 0 | waits for E8b seeds 2 and 3 to finish (~01:45), then E10_fullq_16ep seeds 0–3 (~3 h each) | ~14:00 on 2026-09-26 |

One lane per GPU, not two: both GPUs are compute-bound (GPU 0 at 100% utilisation),
so two runs sharing one finish no sooner than one after the other, and running in
turn gives the first full-question result about three hours in instead of six.

**Re-planned at 01:58 — the one-lane reasoning above was wrong for GPU 0.** The
first full-question epoch took 17.1 min, not ~11.5, putting the finish near 20:00.
GPU 0 is not ours alone: the lab-mate's jobs (two processes by then, 33 + 19 GB)
share it, and the card's time is divided roughly per process. Measured on this
card tonight, one of our runs next to theirs managed an epoch per 17.1 min, while
two of ours side by side managed one each per 22.7 min — 11.4 min per epoch of
combined throughput, 1.5× faster. The "no sooner" argument holds only when our
runs are alone on a card. With the lab's overnight demand low, E10 was restarted
(losing ≤ 4 min of training) with:

| lane | runs |
|---|---|
| GPU 0, lanes a and b | the full-question seeds, both lanes listing all five |
| GPU 1 slice | control seeds 0–4 (~70 min each, measured), then any full-question seeds still free |

Each run now takes a lock while it trains (`run_queue.sh`), so lanes that list the
same runs share them out instead of colliding, and a faster GPU simply takes more.
Estimated finish: ~14:00 on 2026-09-26. If the lab-mate needs more of GPU 0 in the
morning, stopping lane b gives it back without losing more than a partial epoch.

**Rebalanced at 08:10.** By morning the lab-mate had more jobs on GPU 0 and a
full-question epoch there took 27 min (23 overnight), while the slice ran one in
12 min and would sit idle after seed 4 (~10:15). GPU-0 lane b was stopped two minutes
into seed 3; the slice takes seed 3 after seed 4 (its lane already listed it), for an
estimated finish around 13:30 rather than 15:20, and GPU 0 carries one of our runs
instead of two during the working day.

**Operational incident (01:24–01:33).** To add the `@after` directive I edited
`run_queue.sh` in place while E8b's two GPU-0 lanes were still running it. Bash reads
a script as it executes, so when those lanes left their main loop they read a
fragment of the new text and stopped with a syntax error. Everything that mattered
had already run: all results, post-processing and uploads are intact (both seeds'
uploads confirmed). Only the final upload-retry step was skipped, and it had nothing
left to do. `run_queue.sh` now carries a warning to replace it (`mv`), never edit it
in place while lanes are running.

### E10 interim: the control, E10_base_16ep (5 seeds, finished 06:58 on 2026-09-26)

Raw output: `reports/raw/E10_base_16ep_{analysis,decision_rules}.txt`,
`reports/raw/E10_base_vs_E0_compare.txt`. This is E0 with 16 epochs instead of 8,
nothing else changed.

| | E0 (8 epochs) | E10_base_16ep | |
|---|---|---|---|
| single model, dev S2 | 0.337 ± 0.044 | **0.377 ± 0.020** | paired +0.040 ± 0.054; 4 of 5 seeds up |
| single model, dev S1 | 0.586 ± 0.039 | 0.604 ± 0.012 | paired +0.018 ± 0.031 |
| in-set rate / classes named | 0.529 / 8.2 | 0.526 / 8.6 | |
| single model + logit adjustment | 0.357 ± 0.044 | 0.375 ± 0.027 | |
| 5-seed ensemble, argmax | 0.365 | 0.380 | bootstrap +0.014, 95% [−0.028, +0.059] |
| **ensemble + logit adjustment** | **0.438** | **0.402** | −0.036 |
| same, 2-annotator reference sets | 0.391–0.401 | 0.361–0.400 | |
| ensemble, S1 (level of top leaf / most probable level) | 0.596 / 0.598 | 0.618 / 0.619 | |

| seed | S2 E0 → E10_base | S1 E0 → E10_base | chosen epoch (of 0–15) | final train loss |
|---|---|---|---|---|
| 0 | 0.389 → 0.359 | 0.642 → 0.617 | 3 | 0.058 |
| 1 | 0.339 → 0.366 | 0.580 → 0.612 | 11 | 0.165 |
| 2 | 0.279 → 0.398 | 0.532 → 0.592 | 14 | 0.303 |
| 3 | 0.311 → 0.363 | 0.596 → 0.609 | 11 | 0.120 |
| 4 | 0.368 → 0.400 | 0.579 → 0.589 | 7 | 0.072 |

Per class (single-model mean F1, E0 → E10_base): `General` 0.119 → **0.295**,
`Claims ignorance` 0.400 → 0.526, `Partial/half-answer` 0.000 → 0.090, `Deflection`
0.213 → 0.260, `Implicit` 0.419 → 0.440; against that, `Dodging` 0.475 → 0.379.
`Explicit`, `Declining` and `Clarification` are unchanged.

**What this shows.**

1. **Longer training makes a better, steadier single model.** The mean rises by
   0.040 and the spread across seeds halves (0.044 → 0.020). The weakest E0 seeds
   gain most (seed 2: +0.119), so part of E0's seed-to-seed variance was seeds that
   stopped before they had finished learning. With 5 paired seeds neither is
   conclusive on its own: the mean gain is 1.7 standard errors, and a halving of the
   spread happens by chance about one time in thirteen (one-sided F-test, 4 and 4 d.f., p ≈ 0.08).
2. **The gain is in the rare classes** — `General`, `Claims ignorance`, `Partial` —
   which is exactly what logit adjustment was supplying post hoc. A model trained
   longer separates those classes by itself instead of leaning on the class prior.
3. **So the two gains overlap, and the final system does not improve.** Logit
   adjustment adds +0.022 on top of this ensemble against +0.073 on E0's, and the
   ensemble-plus-rule result is lower (0.402 vs 0.438). The one-number rule was
   doing much of what extra epochs now do. There is no interval on this single
   number; for scale, the rule's own in-fold-minus-held-out gap on this ensemble is
   0.028. Read it as "no gain in the final system" rather than as a firm loss.
4. **Epoch selection is noisy.** The chosen epochs range from 3 to 14. Seed 0
   picked epoch 3 on a slightly higher slice score (0.407 vs 0.38–0.40 afterwards),
   and its dev score at epoch 3 (0.359) is below what its later epochs reached. A 10%
   slice (345 single-annotator rows) is a coarse instrument for choosing among 16
   epochs.

**Prediction 3** ("E10_base_16ep within noise of E0") held for the final system
and for the ensemble; the single-model mean moved further than predicted (+0.040),
at the edge of the ±0.04 band.

**What this means for the main comparison.** The full question must now be judged
against E10_base_16ep, not against E0: 0.377 ± 0.020 single-model and 0.402 for the
ensemble with logit adjustment. The full-question seeds finish around 14:00.

---

## Deferred

- **An LLM in the re-ranking slot.** The encoder pipeline comes first; the
  cross-fitted candidates and fall-back fusion from E6 are ready for it.
- **Training on all of train.** Every run holds out 10% of train to choose its best
  epoch. Once an epoch count is settled, a final model could be retrained on
  100% with that fixed count — about 10% more data, at the cost of not being able
  to choose its checkpoint.
