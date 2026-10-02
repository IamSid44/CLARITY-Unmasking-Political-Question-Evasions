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
| Period | 2026-09-19 → 2026-10-02 |
| Hardware | 1× RTX PRO 6000 (96 GB), shared with other users' jobs; E13 on JarvisLabs VMs with the same GPU |
| Backbone | `microsoft/deberta-v3-large` (435M parameters) through E12; `Qwen/Qwen3-8B-Base` + LoRA from E13 |
| Final system (E11, by a rule fixed in advance) | full question + 16 epochs, **10-seed** ensemble + logit adjustment: dev S2 **0.405** (0.412 over CV splits), dev S1 **0.648**; the 10-seed baseline system scores 0.428 / 0.601 — not distinguishable on S2 |
| Latest | E12: training on all of train is the one variant that helps (S2 +0.029 per model, 4/5 seeds; 5-seed system 0.489); the hierarchy of specialist encoders, boundary experts and model soups do not |
| Latest | **E13: Qwen3-8B + LoRA, the backbone as the one change, 10 seeds.** Single model S2 0.476 ± 0.043 vs 0.384 (+0.092, 10/10 up); **10-seed system S2 0.543 / S1 0.746 vs DeBERTa 0.405 / 0.648** (+0.136, 95% [+0.054, +0.224]). 12 epochs adopted on the slice (3-seed screen); all of train +0.017 per model, not distinguishable |
| Running | nothing; all JarvisLabs VMs paused |

---

## Scoreboard

Every configuration so far, one row each; the sections below explain each row in
the order it was run. Single-model numbers are the mean ± std over 5 seeds.
"+ rule" is the 5-seed ensemble followed by logit adjustment, the form a final
system takes.

| id | what changed from the baseline | single model, S2 | single model, S1 | ensemble + rule, S2 | verdict |
|---|---|---|---|---|---|
| E0 | — (DeBERTa-v3-large, sub-question + answer, 8 epochs, cross-entropy) | 0.337 ± 0.044 | 0.586 ± 0.039 | 0.438 | the baseline; 0.356 on seeds 5–9 (E11) |
| E4 | per-class thresholds instead of the one-number rule | | | 0.394 | overfits 308 items |
| E5 | hard hierarchical routing, post hoc | | | 0.369 *(ensemble, no rule)* | no reliable effect |
| E6 | second encoder re-ranks the baseline's top 3 | 0.329 ± 0.039 *(re-ranker alone)* | | 0.354 *(re-ranker + baseline prior)* | negative; baseline ensemble alone is 0.365 |
| E8 | + full question, Balanced Softmax + focal loss | 0.322 ± 0.021 | 0.539 ± 0.030 | 0.379 | negative; cause revised after E8b |
| E8b | + full question | 0.282 ± 0.043 | 0.482 ± 0.039 | 0.343 | undertrained at 8 epochs → E10 |
| E9 | hierarchical output head | 0.320 ± 0.038 | 0.606 ± 0.024 | 0.370 | negative for S2; S1 steadier, not better |
| E10 control | 16 epochs instead of 8 | 0.377 ± 0.020 | 0.604 ± 0.012 | 0.402 | better single model; no gain in the final system |
| E10 | + full question, 16 epochs | 0.387 ± 0.036 | 0.631 ± 0.024 | 0.478 | best on these seeds; 0.434 on seeds 5–9 (E11) |
| **E11** | **E0, E10 control and E10 re-run on seeds 5–9** | | | | **replication and final system: table below** |
| E12a | full question, 16 epochs, **trained on all of train**, last epoch (5 seeds) | **0.416 ± 0.020** | 0.613 ± 0.017 | **0.489** | **helps S2**: +0.029 per model (4/5 up); S1 −0.017; 5-seed system only |
| E12b | hierarchy: Non-Reply gate + branch specialist encoders | 0.345 *(3 seeds)* | 0.608 | 0.390 *(3 seeds)* | negative; specialists alone level on 5 seeds |
| E12c | boundary experts for the three most-confused pairs (3 seeds) | 0.383 | 0.615 | | no effect |
| E12d | model soup of 10 trained models | 0.267 *(uniform)* | 0.471 | | negative |
| E13 + E13x | **Qwen3-8B-Base + LoRA** instead of DeBERTa (same input, rows, slice; 3 epochs; 10 seeds) | **0.476 ± 0.043** | **0.709 ± 0.031** | **0.543** (S1 0.746) | **best by far**: +0.092 S2 / +0.095 S1 per model over DeBERTa, 10/10 seeds; system +0.136 [+0.054, +0.224] |
| E13b | Qwen, 12 epochs instead of 3 (3 seeds) | 0.543 *(3 seeds)* | 0.742 | | slice F1 +0.084 (3/3) → **adopted**; best epochs 8–11; 10 seeds after the mid-eval |
| E13c | Qwen, all of train, last epoch (10 seeds) | 0.492 ± 0.045 | 0.715 | 0.575 | +0.017 per model (6/10); system +0.028 [−0.026, +0.092]: not distinguishable |
| — | ChulaNLP's fine-tuned DeBERTa-large (published; checkpoint chosen on dev) | 0.46 | 0.65 | | for reference |

**The three main systems at 10 seeds (E11)** — the numbers to quote. Single models:
mean ± std over seeds 0–9; systems: 10-seed ensemble + logit adjustment.

| system | single model, S2 | single model, S1 | system, S2 | system, S1 |
|---|---|---|---|---|
| baseline (sub-question + answer, 8 epochs) | 0.315 ± 0.040 | 0.576 ± 0.030 | **0.428** | 0.601 |
| 16 epochs | 0.362 ± 0.026 | 0.601 ± 0.016 | 0.400 | 0.620 |
| **full question, 16 epochs — final system** | **0.384 ± 0.030** | **0.614 ± 0.027** | 0.405 | **0.648** |

Per model, the full question plus 16 epochs beats the baseline on 9 of 10 seeds
(S2 +0.069, S1 +0.038). As systems, the three are within dev-set noise on S2.
Packaged submissions: the baseline (`../submissions/L0_large_base*`) and the final
system (`../submissions/FINAL_fullq_16ep_10seed_logitadj/`); Codabench is closed.

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
- [E11 — Replication on new seeds, and the final system](#e11--replication-on-new-seeds-and-the-final-system-launched-2026-09-27)
- [E12 — Groundwork, plan and runs](#e12--groundwork-plan-and-runs-launched-2026-09-28)
- [E13 — Is the encoder's gap a knowledge gap? A LoRA-tuned Qwen3-8B classifier](#e13--is-the-encoders-gap-a-knowledge-gap-a-lora-tuned-qwen3-8b-classifier-registered-2026-10-02)
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

The re-ranker design and failure analysis are retold in `03_research_narrative.md` §10. In short:

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
- *Over the "marker gate"* (the team's mid-evaluation plan, §2; since removed from the tree, in git history): its premise — that those three
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

### E10 results (2026-09-26, all 10 runs finished 13:35): the full question helps once trained long enough — a new best, 0.478

Raw output: `reports/raw/E10_fullq_16ep_{analysis,decision_rules}.txt`,
`reports/raw/E10_fullq_vs_base_compare.txt`, `reports/raw/E10_final_system_bootstrap.txt`,
`reports/raw/E10_ensemble_size.txt`.

| | E0 (8 epochs) | E10_base_16ep | **E10_fullq_16ep** |
|---|---|---|---|
| single model, dev S2 | 0.337 ± 0.044 | 0.377 ± 0.020 | **0.387 ± 0.036** |
| single model, dev S1 | 0.586 ± 0.039 | 0.604 ± 0.012 | **0.631 ± 0.024** |
| single model + logit adjustment | 0.357 ± 0.044 | 0.375 ± 0.027 | **0.398 ± 0.021** |
| in-set rate (single model) | 0.529 | 0.526 | 0.540 |
| 5-seed ensemble, argmax | 0.365 | 0.380 | **0.416** |
| **ensemble + logit adjustment** | 0.438 | 0.402 | **0.478** |
| same, averaged over 10 different CV fold splits | 0.446 | 0.403 | **0.472** |
| same, 2-annotator reference sets | 0.391–0.401 | 0.361–0.400 | **0.423–0.448** |
| ensemble, S1 (top leaf / most probable level) | 0.596 / 0.598 | 0.618 / 0.619 | **0.631 / 0.627** |
| final train loss | 0.57–1.09 | 0.06–0.30 | 0.56–0.78 |

| seed | S2: E10_base → E10_fullq | S1: E10_base → E10_fullq | epoch chosen (base / fullq) |
|---|---|---|---|
| 0 | 0.359 → 0.344 | 0.617 → 0.591 | 3 / 10 |
| 1 | 0.366 → 0.370 | 0.612 → 0.627 | 11 / 14 |
| 2 | 0.398 → 0.428 | 0.592 → 0.640 | 14 / 10 |
| 3 | 0.363 → 0.421 | 0.609 → 0.641 | 11 / 13 |
| 4 | 0.400 → 0.369 | 0.589 → 0.655 | 7 / 9 |
| paired mean | +0.009 ± 0.036 | +0.027 ± 0.035 | |

**The 2 × 2 that E8b and E10 complete.** The input and the training length were
each tested alone and together, so the effect of the full question can be read at
both training lengths (ensemble + logit adjustment; single-model mean in brackets):

| | 8 epochs | 16 epochs |
|---|---|---|
| sub-question + answer | 0.438 (0.337) — E0 | 0.402 (0.377) — E10_base |
| + full question | 0.343 (0.282) — E8b | **0.478 (0.387)** — E10_fullq |

The full question costs 0.095 at 8 epochs and gains 0.076 at 16: **its effect
depends on training length**, which is what "undertrained" predicted. Training
longer without it does not help the final system.

**Per class** (single-model mean F1, E10_base → E10_fullq): `Dodging` 0.379 →
**0.478**, `General` 0.295 → 0.323, `Deflection` 0.260 → 0.282, `Declining`
0.379 → 0.417; against that `Claims ignorance` 0.526 → 0.228 and `Partial`
0.090 → 0.000. (`Clarification` 0.350 → 0.651 rests on 4 items.) By length
(ensemble in-set rate): short items 0.560 → 0.599, long items the 1024-token model
saw in full 0.531 → 0.593, long items it still had to cut 0.505 → 0.505.

**How solid is 0.478?**

- *Not a lucky fold split.* The decision rule's τ is fitted by nested CV; over 10
  different fold splits the result is 0.472 (range 0.449–0.479), and τ is stable
  (in-fold minus held-out gap 0.000).
- *Against the same training without the full question:* +0.075, paired bootstrap
  95% [+0.001, +0.148] — just clear of zero.
- *Against the 8-epoch baseline:* +0.042, 95% [−0.049, +0.133] — **not yet
  conclusive** on 308 dev items. On the three 2-annotator versions of dev the
  margin is +0.03 to +0.06 each time, which points the same way but is not
  independent evidence.
- *Ensemble size.* Averaging over every seed subset of each size (ensemble + rule):

  | seeds in the ensemble | 1 | 2 | 3 | 4 | 5 |
  |---|---|---|---|---|---|
  | E0 | 0.361 | 0.398 | 0.415 | 0.437 | 0.449 |
  | E10_base_16ep | 0.378 | 0.405 | 0.406 | 0.406 | 0.398 |
  | E10_fullq_16ep | 0.404 | 0.430 | 0.427 | 0.426 | 0.471 |

  The full question is ahead at 1–3 seeds, but its 5-seed value is one draw that
  sits well above its 4-seed average, and E0 is still climbing at 5. Whether the
  lead holds for larger ensembles is not settled by these five seeds.

**Why the gain appears after ensembling and the rule rather than per model.** Both
16-epoch arms have more diverse seeds than E0 (pairwise agreement 0.50–0.51 vs
0.58), which is what makes ensembles gain. They differ in confidence: the control
has nearly memorised its training set (loss 0.06–0.30) and is badly overconfident
(mean top probability 0.728 against an in-set rate of 0.526), so averaging and
the prior correction have little to work with (the rule adds +0.022). The
full-question model fits less tightly (loss 0.56–0.78; top probability 0.638 vs
in-set 0.540), and the rule adds +0.061. A plausible reading, not tested here, is
that the harder input also acts as a regulariser.

**Predictions.** (1) Train loss at or below E0's: held (0.56–0.78). (2) A gain over
the control, mostly in `General`, `Dodging`, `Deflection` and on long items: held
for the final system and clearly for `Dodging`, weakly for `General` and
`Deflection`; not for the single-model mean (+0.009, noise); on length, the gain
is as large on short items as on long ones the model saw in full, and absent on
long items that were still cut. (3) The control within noise of E0: held for the
final system.

**Verdict.** The full question is worth keeping, but only with the longer
schedule; together they give the best system so far (0.478; 0.472 over fold
splits), level with ChulaNLP's fine-tuned DeBERTa (0.46, whose checkpoint was
chosen on dev). Its lead over the 8-epoch baseline is not yet distinguishable from
dev-set and seed noise, so the next step replicates on new seeds (E11).

> **Revised after E11 (2026-09-28).** 0.478 did not replicate: the same system scored
> 0.434 on seeds 5–9 and 0.405 as a 10-seed ensemble, and the baseline system's
> 0.438 likewise fell to 0.356 on new seeds. What E10 found that does hold is the
> single-model gain from the full question plus longer training (+0.069 on 9 of 10
> seeds). See [E11 results](#e11-results-2026-09-28-15-runs-finished-0643-single-models-improve-reliably-systems-are-within-noise).

---

## E11 — Replication on new seeds, and the final system *(launched 2026-09-27)*

*Written before any E11 run started.*

**Why.** E10 gave the best system so far (0.478), but two things about it are not
settled. Its lead over the 8-epoch baseline (+0.042) is inside the dev-set noise,
and its 5-seed value sits well above its own 4-seed average (0.426), so part of it
may be a lucky draw of seeds. And an exploratory look at E10's predictions — a
10-model ensemble mixing both inputs scored 0.498 — produced a hypothesis that was
formed on the same dev set it would be judged on. Both need fresh seeds, not new
ideas. With the mid-evaluation this week, a result that holds on a second set of
seeds is worth more than one more component.

**Design.** The three systems of the 2 × 2 that matter, each re-run with **seeds
5–9**, which no earlier run used. Every setting is identical to the original
(checked against each run's `config.json`); the new names keep the original
5-seed folders untouched, so each system gets a second, independent 5-seed
ensemble and a pooled 10-seed one.

| E11 configuration | identical to | seeds |
|---|---|---|
| `E11_fullq_16ep` | E10_fullq_16ep (full question, 1024 tokens, 16 epochs) | 5–9 |
| `E11_base_16ep` | E10_base_16ep (sub-question + answer, 16 epochs) | 5–9 |
| `E11_base_8ep` | E0 / `L0_large_base` (sub-question + answer, 8 epochs) | 5–9 |

**Hypotheses, fixed now.** All on dev S2, ensemble + logit adjustment unless
stated, τ by 5-fold nested CV.

1. **Replication.** E11_fullq_16ep scores above E11_base_8ep on seeds 5–9, as
   E10_fullq did over E0 on seeds 0–4.
2. **Single models.** With the decision rule applied per model, the E10 ordering
   holds on the new seeds: full question 16 epochs > base 16 epochs > base 8 epochs
   (E10: 0.398 > 0.375 > 0.357).
3. **Mixed inputs (the exploratory idea, tested once).** A 10-model ensemble of
   E11_fullq_16ep + E11_base_16ep (seeds 5–9 only) scores above the 10-model
   full-question ensemble (E10 + E11 fullq, seeds 0–9). Same number of models, so
   the comparison is about mixing, not size.
4. **Ensemble size.** Each system's 10-seed ensemble scores at least its 5-seed
   average; the curve for 1–10 seeds is reported for all three.

**The final system is chosen by this rule, now:** the 10-seed full-question
ensemble (seeds 0–9) with logit adjustment, unless hypothesis 3 holds, in which
case the mixed ensemble. If hypothesis 1 fails, the final system is the 10-seed
ensemble of whichever of the three scores highest, and the full question is
reported as not established.

**What would change the story.** If E11_fullq lands near 0.43 rather than 0.48, the
E10 number was partly luck and the honest headline is "the full question makes
single models better and steadier; the final-system gain is within noise".

**Compute.** 15 runs, about 24 GPU-hours: full-question runs ~3.1 h each and
16-epoch base runs ~1.1 h on either GPU alone, 8-epoch base runs ~0.6 h. Both GPUs
are free (GPU 0 whole, and our GPU-1 slice). One lane per GPU — alone on a card a
second lane adds no throughput — and both lanes list all 15 runs, so the per-run
locks share the work out; the long full-question runs go first. Estimated finish:
~08:00 on 2026-09-28.

**Throughput test at launch (19:25–19:40).** GPU 0 turned out to be twice as fast
alone as the estimate above assumed: a full-question epoch took **5.6 min** (the
11.6 min figure was measured with a lab-mate's jobs on the card). A second lane
was added to see whether two runs would use the idle card better. They did not:
sharing, the full-question run slowed to 12.8 min per epoch and the 512-token run
ran at ~4.5 min, against ~2.0–2.5 min alone — together at most one card's worth of
work, as expected when one run already keeps the GPU busy 100% of the time and
processes take turns. The second lane was stopped (its run resumes later from its
checkpoint). Revised finish with one lane per GPU: ~04:00 on 2026-09-28.

**GPU 0 shared again from ~20:10.** A lab-mate started three jobs on GPU 0 (22 +
19 + 19 GB); our full-question epoch there went from 5.6–7 min to 17–21 min. The
GPU-1 slice is unaffected (~12 min). Revised finish, if their jobs keep running:
late morning on 2026-09-28; the full-question seeds, which decide the replication,
come first.

At 22:04, with the user's agreement, a second GPU-0 lane was added: with GPU time
split roughly per process, two of our runs get about 2/5 of the card against the
lab-mate's three instead of 1/4. It takes the 512-token runs first (~9 GB peak), so
the card keeps ~12 GB of headroom for the neighbours' jobs; its full-question
seeds come last and start only if 18 GB is free.

### E11 results (2026-09-28, 15 runs, finished 06:43): single models improve reliably; systems are within noise

Raw output: `reports/raw/E11_replication.txt`, regenerated by
`python clarity/e11_replication.py` (CPU, seconds); per-configuration analyses in
`reports/raw/E11_*_{analysis,decision_rules}.txt`.

**The same system, two sets of seeds** (5-seed ensemble + logit adjustment, dev S2;
in brackets, averaged over 10 CV splits):

| system | seeds 0–4 (E0 / E10) | seeds 5–9 (E11) | all 10 seeds |
|---|---|---|---|
| baseline: sub-question + answer, 8 epochs | 0.438 (0.446) | 0.356 (0.348) | **0.428** (0.444) |
| 16 epochs | 0.402 (0.403) | 0.375 (0.377) | 0.400 (0.386) |
| full question, 16 epochs | 0.478 (0.472) | 0.434 (0.427) | 0.405 (0.412) |

The first thing this table shows is how noisy a 5-seed ensemble is on 308 items:
the baseline system scored 0.438 on one set of seeds and 0.356 on another. Both
headline numbers of the last week — the baseline's 0.438 and E10's 0.478 — were the
favourable end of that range.

**Single models, all 10 seeds** (mean ± std; paired differences are per seed, same
initialisation and data order):

| | dev S2 | dev S1 |
|---|---|---|
| baseline, 8 epochs | 0.315 ± 0.040 | 0.576 ± 0.030 |
| 16 epochs | 0.362 ± 0.026 | 0.601 ± 0.016 |
| full question, 16 epochs | **0.384 ± 0.030** | **0.614 ± 0.027** |
| 16 epochs − baseline | +0.047 ± 0.044 (t = 3.4; 9/10 seeds up) | +0.026 (t = 2.7; 8/10) |
| full question − 16 epochs | +0.022 ± 0.034 (t = 2.0; 7/10) | +0.012 (t = 1.3; 6/10) |
| **full question, 16 epochs − baseline** | **+0.069 ± 0.061 (t = 3.6; 9/10)** | **+0.038 (t = 2.8; 9/10)** |

**The hypotheses, as written before the runs.**

1. *Replication — held, narrowly.* On seeds 5–9 the full-question system beats the
   baseline system, 0.434 vs 0.356 (+0.077; bootstrap 95% [−0.007, +0.157]).
2. *Single-model ordering — held on both seed sets.* With the rule applied per
   model: full question 0.381 > 16 epochs 0.361 > baseline 0.325 on seeds 5–9
   (0.398 > 0.375 > 0.357 on seeds 0–4).
3. *Mixed-input ensemble — failed.* 10 mixed models 0.403 vs 10 full-question models
   0.405. The 0.498 seen in E10 was a feature of those particular seeds.
4. *Ensembles at least as good at 10 seeds as at 5 — held for both baseline-input
   systems, failed for the full question.* Averaging over subsets of the 10 seeds:

   | seeds in the ensemble | 1 | 2 | 3 | 5 | 7 | 10 |
   |---|---|---|---|---|---|---|
   | baseline, 8 epochs | 0.341 | 0.368 | 0.384 | 0.403 | 0.420 | 0.428 |
   | 16 epochs | 0.368 | 0.377 | 0.380 | 0.391 | 0.389 | 0.400 |
   | full question, 16 epochs | 0.389 | 0.407 | 0.416 | 0.418 | 0.417 | 0.405 |

   The full-question system is ahead up to about 5 models; the baseline keeps
   gaining from every added seed (+0.087 from 1 to 10, against +0.016) and draws
   level. We do not have a tested explanation. The full-question models are more
   confident relative to their accuracy (E10), and averaging confident models
   changes less; that is the leading candidate.

**The final system, by the rule fixed before the runs.** Hypothesis 1 held and
hypothesis 3 failed, so the final system is the **10-seed full-question ensemble
with logit adjustment: dev S2 0.405 (0.412 over CV splits), dev S1 0.648**. Its
submission is packaged by `make_submission.py` from the 10 runs. Reported alongside
it, as the rule did not anticipate this case: the 10-seed *baseline* ensemble scores
higher on Subtask 2 (0.428; difference −0.022, 95% [−0.101, +0.061]) and lower on
Subtask 1 (0.601), and is also a little better on the 2-annotator reference sets
(0.392–0.407 vs 0.356–0.403). On Subtask 2 the two systems cannot be told apart on
this dev set; on Subtask 1 the full-question system is ahead.

**What changes in the story.**

- **The robust result is at the level of a single model.** Longer training plus the
  full question improves a single fine-tuned DeBERTa on both subtasks, on 9 of 10
  seeds: S2 0.315 → 0.384, S1 0.576 → 0.614. Most of it is the longer training
  (+0.047); the full question adds a further, smaller +0.022.
- **At the level of the final system, the gains do not survive.** Ensembling and
  the decision rule lift the baseline more than they lift better models, and a
  308-item dev set cannot separate systems within ~0.08 of each other.
- **Method.** A single 5-seed ensemble is not enough to rank systems here; every
  system comparison from now on uses at least 10 seeds and reports the spread
  across seed sets. The E10 "new best 0.478" is annotated accordingly.

---

## E12 — Groundwork, plan and runs *(launched 2026-09-28)*

*The groundwork was run on CPU against models that already exist; the plan was
written before any E12 model was trained and approved as written on 2026-09-28.*

### Groundwork: what the existing 20 trained models say

**1. The decision layer is saturated** (`decode_variants.py`,
`reports/raw/decode_variants.txt`). Three alternatives to logit adjustment, each
fitted by the same nested CV, on the 10-seed baseline and full-question systems and
per seed:

| rule, minus logit adjustment, per seed | baseline models | full-question models |
|---|---|---|
| hierarchical Non-Reply gate with a fitted threshold | +0.009 (8/10 up) | +0.003 (5/10) |
| prior-matching decoding (per-class weights by Sinkhorn/optimal transport) | −0.007 | −0.005 |
| coverage floor (every class predicted at least m times) | +0.001 | −0.002 |

None beats the one-parameter rule reliably. Any further gain has to come from the
models, not from how their outputs are read.

**2. No extra supervision is hiding in the data.** Train has one label per row (the
`annotator1–3` columns are empty); only 26 question–answer pairs repeat with
conflicting labels (52 rows). The per-row flags (`multiple_questions`,
`affirmative_questions`, `inaudible`) and the GPT-3.5 fields are empty in the test
set, so they cannot be inputs.

**3. The error structure — the data-driven basis for a hierarchy**
(`reports/raw/E12_design_confusion.txt`; the 10-seed full-question ensemble on the
fixed 10% train slice, so dev is not touched):

- 179 errors on 345 items: **153 (85%) inside the six non-Non-Reply classes**, 24
  across the Non-Reply / other line, 2 inside the Non-Reply classes.
- Clustering the confusion matrix splits the three **Non-Reply classes off first**
  (the same top cut TeleAI derived from their confusion matrix).
- But our failure is the **opposite of TeleAI's**: they had Non-Reply false positives;
  our models *miss* Non-Replies (precision 0.69, recall 0.57). Clear Non-Reply is
  our weakest Subtask-1 class (F1 0.44–0.52), and the Non-Reply classes carry a
  third of the Subtask-1 macro-F1 and three of nine classes in Subtask 2.
- Inside the other six, errors concentrate in three pairs: **Implicit ↔ Dodging**
  (11 + 13), **Explicit ↔ Implicit** (12 + 13; this one also crosses the Subtask-1
  line between Clear Reply and Ambivalent) and **General ↔ Deflection** (5 + 8). A
  tree cut through those pairs would not help; they are the fine decisions, not the
  coarse ones.
- For context, TeleAI's own ablation (their Table 4): the confidence-gated hierarchy
  moved their Subtask 1 from 0.710 to 0.811 but Subtask 2 only from 0.490 to 0.503;
  their large Subtask-2 gain (→ 0.617) came from boundary examples for exactly such
  confusable pairs.

**4. Choosing the epoch on the train slice buys nothing**
(`reports/raw/last_vs_selected_epoch.txt`). Across the 10 full-question seeds the
last epoch scores at least as well as the selected one (S2 +0.011 ± 0.020, 7/10;
S1 −0.002). The slice costs 10% of the training data and adds noise (chosen epochs
ranged from 3 to 14).

### Why fewer seeds are enough now

Ten seeds were needed for *system* claims: two 5-seed ensembles of the same system
differed by 0.08. *Model-level* claims, paired by seed, need fewer when the effect is
of useful size: the 16-epoch gain (+0.047) was already clear at 5 paired seeds; only
the small full-question gain (+0.022) needed 10. E12 therefore **screens each variant
on 3 seeds**, paired with the existing full-question models of the same seeds
(E10_fullq_16ep, seeds 0–2), and **extends to 5 seeds only a variant that gains at
least +0.015 on S1 or S2 with at least 2 of 3 seeds up**. Three seeds reliably catch
only large effects; that is the intended filter.

### Proposed variants

All use the current best model's settings (DeBERTa-v3-large, full question, 1024
tokens, 16 epochs) unless stated.

| id | variant | why it should help, from our evidence | new code | GPU per seed |
|---|---|---|---|---|
| **E12a** | **Train on all of train**, fixed 16 epochs, keep the last epoch | +11% data; epoch selection adds nothing (groundwork 4) | small: allow `--val-frac 0` with `--select last` | ~1.5 h |
| **E12b** | **Hierarchy of specialists**: a Non-Reply gate (binary, all data), a 6-way specialist on the non-Non-Reply rows, a 3-way specialist on the Non-Reply rows; combined softly, p(leaf) = p(branch) × p(leaf \| branch) | targets our actual failure (Non-Reply recall 0.57) at the cut the data picks; specialists see a rebalanced label space | `--task gate/other6/nr3` in `encoder.py`; `hier_combine.py` | ~3 h |
| **E12c** | **Boundary experts** for the three confused pairs (Implicit vs Dodging, Explicit vs Implicit, General vs Deflection), each applied when its pair is the model's top two | 85% of errors sit inside the six, concentrated in these pairs; the encoder analogue of TeleAI's boundary examples | `--task pair:<A>,<B>`; reuses `hier_combine.py` | ~1.5 h |
| **E12d** | **Model soup**: average the weights of the 10 existing full-question models (uniform, and greedy by train-slice score; Wortsman et al., ICML 2022) | one model with part of the ensemble's gain, no training | `soup.py` (inference only) | ~20 min once |

**How E12b stands apart from TeleAI's pipeline.** Encoder-only, with trained
specialists instead of prompts; the tree comes from our models' confusion matrix;
the gate is aimed at our failure mode (missed Non-Replies) rather than theirs
(false alarms); routing is soft, so a gate error is not final; and its
contribution is isolated by a 2 × 2 ablation computed on CPU once the models exist —
{flat model's implied gate, dedicated gate} × {flat model within each branch,
specialists} — plus soft vs hard routing and a fitted gate threshold.

**Predictions** (written before launch):

- E12a: per model S2 +0.01 to +0.02, S1 about unchanged.
- E12b: S1 up through Clear Non-Reply, with Non-Reply recall above 0.57; S2 between
  0 and +0.02. If the gate does not raise recall, the hierarchy is not the fix.
- E12c: small or no S2 gain; a null result would agree with E6 (a second encoder
  choosing among the first one's candidates did not help).
- E12d: the soup scores above the average single model and below the 10-model
  ensemble.

**Order and cost.** E12d first (minutes; no training), then E12a and E12b on 3
seeds, then E12c if time allows. Estimated ~18 GPU-hours at GPU 0's unshared speed
(a full-question epoch takes 5.6 min alone), about 12 hours of wall time with GPU 0
and the GPU-1 slice, longer if lab-mates share GPU 0. Implementation and smoke tests
(uploads off) come before any launch.

**Dropped from consideration.** Further decoding rules (groundwork 1); soft-label
training (no multi-annotator data in train); metadata features (absent in test);
the LLM step (deferred by decision; §Deferred).

### E12 launch (2026-09-28, 11:29)

**What was built** (no earlier run is affected; the default path is unchanged):

- `encoder.py --task {leaf9, gate, other6, nr3, pair:<A>,<B>}` — a run learns a
  sub-problem: rows outside it are dropped, labels are remapped, the output layer
  has as many units as the task has classes, and the epoch is chosen on the
  task's own rows of the fixed train slice. `--val-frac 0 --select last` trains on
  every row and keeps the last epoch (E12a).
- `run_queue.sh` skips the 9-way post-processing for sub-task models.
- `hier_combine.py` turns gate + specialists (+ boundary experts) back into 9-way
  distributions and scores every combination against the flat model of the same
  seed (the 2 × 2, hard vs soft routing, experts on the flat model and on the
  hierarchy).
- `soup.py` builds uniform and greedy weight-averaged soups (E12d).

**Checked before launch.** Every task mode ran end to end on a tiny training set
(1 epoch, 64 rows, uploads off): output shapes 9/2/6/3/2, held-out rows per task
345/345/310/35/120. The combination code reproduces the flat model exactly when
fed the flat model's own gate and within-branch distributions (max difference
2e-7), and boundary experts keep every row a distribution.

**Layout.** tmux session `clarity-E12`: the soup runs first on the GPU-1 slice;
two lanes (GPU 0, GPU-1 slice) list the same 21 training runs — for each of seeds
0–2 the gate, the two specialists and the full-data model, then the boundary
experts — and share them through the per-run locks. GPU 0 is shared with two
other users' jobs (27 + 45 GB), so our run there gets roughly a third of the card.
Estimated finish: E12a/E12b early on 2026-09-29, E12c later that morning; sooner
if GPU 0 frees up.

**A third lane at 14:12.** GPU 1 is split into two MIG slices of 48 GB; ours is the
first, and the second — used earlier by another user's job — was idle. A lane pinned
to its UUID (`# device:` line in `lane_gpu1_b.txt`, now supported by `run_queue.sh`)
runs one model there (~14 GB, leaving ~33 GB free for its usual user; the lane is
stopped if they need it). GPU 0 had no room for a second run (~9 GB free), and a
second run on our own slice would only take turns with the first. With the extra
slice the E12a/E12b runs should finish during the night of 2026-09-28.

**Released at 15:08.** A lab-mate asked for a GPU-1 slice, so the borrowed one was
given back: lane b was stopped (its run, E12a seed 0, had finished 4 of 16 epochs and
resumes from that checkpoint in another lane) and its lane file removed. E12
continues on GPU 0 and our own slice; the finish estimate returns to early
2026-09-29 for E12a/E12b.

**A second GPU-0 lane at 15:10**, when a lab-mate's 45 GB job on GPU 0 ended (the
card then held another user's 27 GB job and ours). With GPU time split per process,
two of our runs get about 2/3 of the card instead of 1/2. It resumed E12a seed 0
from its checkpoint. The second GPU-1 slice stays with the lab-mate, and our own
slice keeps one run: a second would only take turns with the first there (measured
~12.6 min per epoch alone, ~25 min each when two share it).

### E12d result (2026-09-28, 4 min): model soups fail for these models

Raw output: `logs/E12d_soup.log`; soups saved as `runs/E12d_soup_{uniform,greedy}/`.

| | held-out slice (train) | dev S2 | dev S1 |
|---|---|---|---|
| a single full-question model (mean of 10) | 0.35–0.42 | 0.384 | 0.614 |
| **uniform soup** of all 10 | 0.248 | 0.267 | 0.471 |
| **greedy soup** | kept only its first member | 0.381 | 0.582 |

Averaging the weights of any two of the models already lowered the held-out score
(from 0.41 to 0.29–0.36), so the greedy recipe rejected all nine additions. The
prediction (between a single model and the ensemble) failed. The likely reason is
in how our models are trained, not in the idea: soups work when fine-tuned models
share their starting point and land in one basin (Wortsman et al. fine-tune from a
common initialisation, including the head); ours differ by seed in the randomly
initialised classification head and in data order, so their weights are not
interchangeable. Averaging *outputs* (the seed ensemble) remains the way to combine
them. The greedy member's dev score (0.381 vs 0.388 in training) is the cost of the
checkpoints being stored in bf16.

**GPU 1 fully released at 23:35.** A lab-mate needed GPU 1's first slice (GI 1,
ours), so its lane was stopped (E12b_nr3 seed 2, 5 of 16 epochs, resumes from its
checkpoint on GPU 0) and its lane file renamed `lane_gpu1_a.txt.released` so a
restart does not bring it back. E12 continues on GPU 0 alone (two lanes), which is
also shared with two other users' jobs. By then 8 of the 21 runs had finished,
including E12a and E12b for seeds 0 and 1.

### E12 interim (2026-09-28 23:40, seeds 0 and 1 only — not a result yet)

Raw: `reports/raw/E12_interim_seeds01.txt` (`python clarity/hier_combine.py`). Paired
with the flat full-question model of the same seed; 2 of the planned 3 seeds.

| vs the flat model, per seed | S2 | S2 + rule | S1 | Non-Reply recall |
|---|---|---|---|---|
| dedicated gate + flat within each branch | −0.016 | −0.059 | −0.012 | +0.029 |
| **flat gate + specialists** | +0.004 | **+0.021** | **+0.031 (2/2)** | +0.015 |
| full hierarchy (dedicated gate + specialists) | −0.026 | −0.048 | +0.015 | +0.029 |
| E12a, all of train | **+0.053 (2/2)** | | +0.017 | |

Early pattern, to be confirmed by seed 2: the **specialists help and the dedicated
gate hurts**. The gate ends training with a loss of ~0.03 — it has memorised the
binary split — so its probabilities sit near 0 or 1 and the soft routing
effectively becomes hard; the flat model's own, softer Non-Reply mass routes
better. Training on all of the data looks like the clearest gain so far.

### E12b result on the screening seeds (2026-09-29 01:45, seeds 0–2)

Raw: `reports/raw/E12b_hierarchy_3seeds.txt` (`python clarity/hier_combine.py`).
Paired with the flat full-question model of the same seed.

| vs the flat model, per seed | S2 | S2 + rule | S1 | Clear Non-Reply F1 | Non-Reply recall |
|---|---|---|---|---|---|
| dedicated gate + flat within each branch | −0.030 (0/3) | −0.059 (0/3) | −0.030 (0/3) | −0.082 | −0.039 |
| **flat model's gate + specialists** | +0.003 (2/3) | **+0.018 (2/3)** | **+0.018 (2/3)** | +0.005 | +0.010 |
| hierarchy: dedicated gate + specialists | −0.036 (0/3) | −0.050 (0/3) | −0.011 (1/3) | −0.082 | −0.039 |
| hierarchy, hard routing | −0.038 (0/3) | −0.059 (0/3) | −0.013 (1/3) | −0.085 | −0.039 |

As 3-seed ensembles with the rule: flat 0.418, flat gate + specialists 0.414,
hierarchy 0.390 (S2); S1 0.643 / 0.640 / 0.641.

**Against the predictions.** The gate was to raise Non-Reply recall and lift
Subtask 1 through Clear Non-Reply; it did neither (recall −0.039, Clear Non-Reply
F1 −0.082). By the rule written before launch, *the dedicated gate is not the
fix*. The reason is visible in training: the gate reaches a training loss of ~0.03,
memorising a 10%-vs-90% split, so its probabilities are near 0 or 1 — soft routing
becomes hard routing (the two rows are almost identical), and its mistakes are
final. The flat model's own Non-Reply mass is a softer, better-calibrated gate.

**What does help is the specialists.** Routed by the flat model's gate, the two
branch specialists add +0.018 on S2 with the rule and +0.018 on S1, 2 of 3 seeds
each — past the screening bar (+0.015, 2/3 seeds), so **they are extended to seeds
3 and 4** (the specialists only; the gate is dropped). Queued at 01:47 on
GPU 0, ahead of the remaining boundary experts. The 3-seed ensemble shows no gain
yet (0.414 vs 0.418); the 5-seed read decides.

**Extra GPU-0 lane at 01:46** (GPU 0 then: another user's 27 GB job + our two runs,
~37 GB free): a third lane, restarted a minute later so that the extension runs
first. E12a seed 2 finishes around 05:30; if it keeps E12a past the screening bar,
E12a is extended to seeds 3–4 the same way.

### E12a on the screening seeds (2026-09-29 06:35, seeds 0–2)

All of train, fixed 16 epochs, last epoch kept; paired with the flat full-question
model of the same seed (which held out 10% and chose its epoch):

| seed | 0 | 1 | 2 | mean |
|---|---|---|---|---|
| dev S2 | +0.070 | +0.035 | −0.035 | **+0.023** (2/3 up) |
| dev S1 | +0.041 | −0.008 | −0.034 | 0.000 |

Past the screening bar on S2, so **E12a is extended to seeds 3 and 4**, queued
first on GPU-0 lane a at 06:37 (the boundary-expert run it had just begun lost a
minute and returns to the shared queue). Its post-processing marker was cleared so
the analysis and submission are rebuilt on all five seeds. The prediction for E12a
(S2 +0.01 to +0.02, S1 unchanged) holds so far.

### E12b on 5 seeds (2026-09-29 14:30): the specialists do not hold up

Raw: `reports/raw/E12b_specialists_5seeds.txt`
(`python clarity/hier_combine.py --focus "flat gate + specialists"`).

| flat model's gate + specialists, minus flat | seeds 0–2 (screening) | **seeds 0–4** |
|---|---|---|
| S2 | +0.003 (2/3) | −0.012 (2/5) |
| S2 + rule | +0.018 (2/3) | **−0.009 (2/5)** |
| S1 | +0.018 (2/3) | **+0.004 (2/5)** |
| Clear Non-Reply F1 | +0.005 | +0.006 |
| Non-Reply recall | +0.010 | +0.024 |

As 5-seed ensembles with the rule: S2 0.404 against the flat model's 0.471; S1
0.646 against 0.631.

Seeds 3 and 4 reversed the screening gain; on five seeds the specialists are level
with the flat model on both subtasks, and as an ensemble clearly worse on
Subtask 2. **Verdict on E12b: negative.** Neither part of the hierarchy improves on
a single flat model — the dedicated gate hurts (it memorises the binary split and
routes hard), and branch specialists add nothing measurable. This is the third
hierarchy tested (after post-hoc routing, E5, and a factorised head, E9), each
with a different mechanism, and none beats the flat model trained on the full
question. It also shows the screening rule working as intended: a +0.018 on three
seeds was worth extending, and two more seeds were enough to see it was noise.

### E12 results (all 27 runs finished 2026-09-29 18:52)

Raw: `reports/raw/E12a_vs_flat_compare.txt`, `E12a_alldata_{analysis,decision_rules}.txt`,
`E12c_experts_3seeds.txt`, `E12_all_combinations_seeds012.txt`, and the E12b/E12d
files above. Every comparison is paired with the flat full-question model of the
same seed (E10_fullq_16ep).

**E12a — training on all of train (5 seeds): the one variant that helps Subtask 2.**

| seed | 0 | 1 | 2 | 3 | 4 | mean |
|---|---|---|---|---|---|---|
| dev S2 | +0.071 | +0.035 | −0.035 | +0.025 | +0.053 | **+0.029 ± 0.040 (4/5 up)** |
| dev S1 | +0.041 | −0.008 | −0.034 | −0.020 | −0.066 | −0.017 ± 0.039 (1/5 up) |

| 5 seeds | flat (E10_fullq_16ep) | E12a |
|---|---|---|
| single model, S2 / S1 | 0.386 / 0.631 | **0.416** / 0.613 |
| single model + rule, S2 | 0.398 | **0.417** |
| ensemble, argmax, S2 | 0.416 | **0.481** (bootstrap +0.066, 95% [−0.004, +0.142]) |
| ensemble + rule, S2 | 0.478 | **0.489** |
| same, 2-annotator reference sets | 0.423–0.448 | 0.417–0.483 |
| ensemble, S1 (top leaf / level) | 0.631 / 0.627 | 0.634 / 0.639 |

More data and no epoch selection lift Subtask 2 per model and as an ensemble. The
cost is on Subtask 1, which dips for single models (−0.017, 1/5 up) and is level as
an ensemble. Keeping the last epoch also removes a step the log showed to be noisy.
Two cautions apply. The comparison seeds (0–4) are the ones whose flat ensemble was
the lucky 0.478 (E11), so the gain is if anything understated at the system level;
but a single 5-seed system number is still not enough to rank systems (E11), so
0.489 is reported as a 5-seed result, not as a new best system.

**E12b — the hierarchy: negative** (above): the dedicated gate hurts, and the branch
specialists passed 3-seed screening but were level with the flat model on 5 seeds.

**E12c — boundary experts for the three confused pairs (3 seeds): no effect.**
S2 +0.002 (1/3), S2 + rule −0.004, S1 −0.004; on top of the hierarchy −0.051. The
experts rarely change a decision the flat model already makes between the same two
labels from the same input — the same lesson as the E6 re-ranker.

**E12d — model soups: negative** (above).

**Against the predictions written before launch.** E12a: S2 +0.01 to +0.02 with S1
unchanged — S2 came in a little higher (+0.029), S1 slightly lower (−0.017). E12b:
the gate was to raise Non-Reply recall — it did not; failed. E12c: small or no gain —
held (none). E12d: between a single model and the ensemble — failed (below a single
model).

**What E12 establishes.** Of four variants, one helps: **train on every row and keep
the last epoch**. The hierarchy — the idea the top systems and the organisers point
to — was tested in its encoder form three ways (post-hoc routing E5, factorised head
E9, specialist encoders E12b) and does not beat a flat model trained on the full
question; in the published systems its benefit came with LLM reasoning stages, which
this track has deferred. The next encoder step, if any, is to make the full-data
model the base and run it on 10 seeds, the standard E11 set for any system claim.

---

## E13 — Is the encoder's gap a knowledge gap? A LoRA-tuned Qwen3-8B classifier *(registered 2026-10-02)*

*Written before any E13 run started.*

**Why.** Every encoder-side attempt to choose better among the model's own top candidates has
failed: decision rules beyond one scalar (E4, E12 groundwork), a re-ranker (E6), boundary
experts (E12c) and three hierarchies (E5, E9, E12b). The mid-eval analysis
(`mideval/analysis/mideval_analysis.txt`) adds two facts:
- 53 of the final ensemble's 137 errors are on items all three annotators agreed on;
- a human annotator scored against the other two reaches S2 0.684, against the model's 0.38.

So the gap is the model, not label noise. Two explanations remain:
1. **Missing knowledge:** world and language knowledge a 0.4B encoder lacks, as in the dataset
   paper's Bernanke example.
2. **Missing label meaning:** what each evasion type means, which a classifier must infer from
   3,400 examples.

A larger pretrained classifier tests the first explanation with nothing else changed.
TeleAI report Qwen2.5-7B fine-tuned without chain-of-thought at 0.495 dev S2 (their Table 2;
one run, generative fine-tuning, their own protocol).

**Design: one change from E10_fullq_16ep, the backbone.**

| | E10_fullq_16ep | **E13 (`Q8_fullq_lora`)** |
|---|---|---|
| backbone | DeBERTa-v3-large, full fine-tune | **Qwen3-8B-Base, LoRA r=16 / alpha 32 / dropout 0.05 on all linear layers; new 9-way head on the last token, trained in full** |
| learning rate, epochs | 1e-5 (head 1e-4), 16 epochs | **1e-4, 3 epochs** (what LoRA on an 8B model needs) |
| rows, train slice, selection | train minus the fixed 345-row slice; best epoch on the slice | same (the slice is asserted identical) |
| input | sub-question + full question (≤256 tokens) + answer, 1024 tokens | same text and budgets, plus a fixed closing question so the head reads the same position on every item |
| loss, batch | cross-entropy, effective 16 | same (micro-batch 4 × accumulation 4) |
| seeds | 0–9 | **0–2 (screen)** |
| hardware | lab RTX PRO 6000 | 1× A100 80GB on JarvisLabs (`JARVISLABS_PORTING_GUIDE.md`) |

Code: `clarity/llm_classifier.py`. Smoke-tested on CPU with Qwen3-0.6B-Base, including an
interrupted epoch resumed from its checkpoint. Outputs follow `encoder.py`'s layout, so
`analyze.py` and `decide.py` read them unchanged.

**Hypotheses and predictions** (paired with E10_fullq_16ep on seeds 0–2: S2 0.344 / 0.370 /
0.428, mean 0.381; S1 0.591 / 0.627 / 0.640, mean 0.619):

1. **Screening (the E12 bar):** S2 at least +0.015 over DeBERTa with at least 2 of 3 seeds up.
   If it passes, extend to seeds 3–9 and compare 10-seed systems as in E11.
2. **Size:** if knowledge is the gap, S2 single-model mean 0.43–0.50 (+0.05 to +0.12) and S1
   +0.02 to +0.05. A gain below +0.015 is read as: knowledge is not what the encoder lacks, and
   the budget moves to the LLM-over-candidates cascade (`mideval/PLAN_REMAINING.md`, Phase B).
3. **Where:** the gain falls mostly on the commitment boundary (Explicit ↔ Implicit, General ↔
   Implicit/Explicit), the largest error pairs in the mid-eval analysis. It falls more on items
   annotators agreed on than on contested ones.
4. **Fit:** the slice selects epoch 1 or 2 of 0–2; training loss ends below 1.0.

**Cost guard.** A 20-step timing pilot aborts the run (and pauses the instance) if the 3 seeds
are projected over 5 hours. Estimated cost ₹400–650 of the ~₹5,880 credit.

### E13 results *(run 2026-10-02, 03:24–04:28 IST)*

*The 3-seed screen. Superseded at 10 seeds by §E13x/b/c results below (S2 0.476 ± 0.043; system 0.543).*

Sources: `raw/E13_Q8_fullq_lora_summary.txt` (per seed and per epoch, the pipeline log, and an
independent recomputation of every number from the saved probabilities, which matched exactly)
and `raw/E13_Q8_fullq_lora_analysis.txt` (`analyze.py`). Outputs: `clarity/runs/Q8_fullq_lora/`.

| seed | selected epoch | slice F1 | dev S2 | dev S1 | E10_fullq_16ep S2 / S1 | paired Δ S2 / S1 |
|---|---|---|---|---|---|---|
| 0 | 2 | 0.459 | 0.405 | 0.647 | 0.344 / 0.591 | +0.061 / +0.056 |
| 1 | 2 | 0.444 | 0.426 | 0.687 | 0.370 / 0.627 | +0.056 / +0.060 |
| 2 | 2 | 0.432 | 0.554 | 0.728 | 0.428 / 0.640 | +0.126 / +0.088 |
| **mean** | | | **0.462 ± 0.080** | **0.688 ± 0.040** | 0.381 / 0.619 | **+0.081 / +0.068** |

**Against the predictions:**

1. **Screening: passed.** S2 +0.081 with 3 of 3 seeds up (bar: +0.015, 2 of 3). S1 is also up on
   all three seeds.
2. **Size: S2 inside the predicted range, S1 above it.** S2 mean 0.462 (predicted 0.43–0.50). S1
   +0.068 (predicted +0.02 to +0.05). This is what the knowledge explanation predicted. Two
   cautions:
   - three seeds with a spread of 0.080: the median seed is 0.426, and seed 2 alone reaches 0.554;
   - the gain is per single model. No system claim is made until 10 seeds (E11 rule).
3. **Where: not tested yet.** The comparison needs E10_fullq_16ep's dev probabilities for seeds
   0–2. They are on the lab server and the HF repo, not on the machine that received this run.
   Per class, Qwen alone (mean of 3 seeds):
   - Explicit 0.724, Implicit 0.459, Dodging 0.550, General 0.319, Deflection 0.393;
   - Declining 0.539, Claims ignorance 0.659, Clarification 0.512 ± 0.423;
   - **Partial/half-answer 0.000 on every seed** (5 dev items).
4. **Fit: held in part.**
   - The slice selected epoch 2 on every seed, which is inside "epoch 1 or 2".
   - Training loss ended below 1.0 on only 1 of 3 seeds (1.103, 1.062, 0.979).

**What else the run shows:**

- **Three epochs may be too few.** The slice F1 rose at every epoch on every seed, and the
  selected epoch is always the last one. This is the pattern E8b → E10 showed for the encoder:
  undertraining can look like a smaller effect than the real one. A longer schedule is a separate
  experiment (one change: epochs, still selected on the slice).
- **Coverage and in-set rate.** It named 9, 8 and 8 classes on dev. Its in-set rate was
  0.604 ± 0.027 and its coverage 0.926 (`analysis` file).
- **Cost of the run.** 21 minutes per seed, with a peak of 19.3 GiB of GPU memory.

**Deviations from the registration.** None of these changes the experiment.

- **Hardware.** A JarvisLabs VM with 1× RTX PRO 6000 Blackwell 96 GB, because the A100 80GB was
  not offered for VMs. It is the same GPU family as the lab server.
- **How the weights arrived.** A network fault on the VM (path-MTU black hole) stalled the
  Hugging Face download. After the fix, the Qwen3-8B-Base shards (revision `49e3418`) were fetched
  with curl into the HF cache. Each was checked against its safetensors header. They are the same
  files `snapshot_download` fetches.
- **A setup-only restart.** The first launch took the run name "Eve" from a WSL environment
  variable. It was restarted during setup, before any training.

**Next, per the registration:** extend to seeds 3–9 and compare 10-seed systems as in E11. At
the measured 21 minutes per seed, that is about 2.5 h of GPU, roughly ₹550–650 including GST and
setup (an estimate). The epoch question above is a separate, later experiment.

---

## E13x, E13b, E13c — 10 seeds, a longer schedule, all of train *(registered 2026-10-02, 11:20 IST)*

*Written before any of these runs started.* Three runs go in parallel on four JarvisLabs VMs
(1× RTX PRO 6000 each), launched with `JPROFILE=<a|b|c|d> bash jarvis_drive.sh all`.

**Changes common to all three, none of which changes the model:**
- **Gradient checkpointing off.** The computation is the same; it is faster, using more memory.
  The pilot now measures peak memory on the longest micro-batch first.
- **W&B logging:** per step and per epoch, project `clarity-semeval26`, grouped by run name.
- **Every epoch's LoRA adapter is kept**, and each finished seed is uploaded to the HF repo
  under `<run name>/seed<k>/`.

**E13x: E13 extended to seeds 3–9 (`Q8_fullq_lora`, VM a).**
- **What:** the E13 configuration on seeds 3–9, exactly as registered for A2.
- **Comparison:** paired with E10_fullq_16ep on seeds 3–4 and E11_fullq_16ep on seeds 5–9.
- **Systems:** 10-seed ensembles with logit adjustment, scored by nested CV as in E11.
- **Predictions:**
  - the 10-seed single-model mean regresses from 0.462 towards 0.42–0.48;
  - Qwen is above DeBERTa on at least 7 of 10 seeds;
  - the 10-seed Qwen system beats DeBERTa's 0.405 by at least +0.03.
- **If it fails:** a 10-seed system at or below 0.435 would mean the 3-seed gain was mostly seed
  luck at the system level.

**E13b: 12 epochs instead of 3 (`Q8_fullq_lora_12ep`, seeds 0–2, VMs b and c).**
- **One change:** epochs 3 → 12. The cosine schedule stretches to 12 epochs, and the epoch is
  still chosen on the train slice.
- **Why:** in E13, the slice F1 rose at every epoch on every seed, and training loss ended at about
  1.0. But under cosine decay the last epoch is favoured by the schedule itself, so a longer
  schedule is the only real test.
- **Hypotheses:**
  - **H1, undertrained:** the slice selects an epoch between 4 and 9. The mean slice F1 at the
    selected epoch is at least E13's 0.445 + 0.015 (2 of 3 seeds up), and dev S2 rises paired
    with E13.
  - **H0, length is not the limit:** the slice F1 stays within ±0.015 of 0.445, and the extra
    epochs overfit (training loss near 0, slice F1 falling late).
- **Decision rule, on the slice only:** adopt the longer schedule for future Qwen runs if H1's
  slice criterion holds. Dev is reported but decides nothing.

**E13c: all of train, last epoch (`Q8_alldata`, seeds 0–2, VM d).**
- **One change from E13:** train on all 3,448 rows (+345) and keep the last of 3 epochs. E13
  selected the last epoch on every seed, so the selection-rule change is nominal.
- **Why:** this is E12a's recipe, the one encoder change that helped (+0.029 S2 per model, 4/5
  seeds).
- **Prediction:** S2 +0.01 to +0.03 per seed, paired with E13.
- **Screening bar:** +0.015 with at least 2 of 3 seeds up. It is independent of E13b, because the
  epoch count stays at 3.

**Cost guard:** each VM's pilot aborts and pauses if its seeds are projected over the
`MAX_TOTAL_HOURS` limit. The estimate is about ₹1,500–1,700 for all four VMs.

### E13c screen *(seeds 0–2, finished 12:08 IST)*

Sources: `raw/E13c_Q8_alldata_seeds012.txt` and `e13_analysis.py` §C.

| seed | E13c S2 | E13 S2 | Δ | E13c S1 | E13 S1 | Δ |
|---|---|---|---|---|---|---|
| 0 | 0.520 | 0.405 | +0.115 | 0.717 | 0.647 | +0.069 |
| 1 | 0.443 | 0.426 | +0.017 | 0.689 | 0.687 | +0.002 |
| 2 | 0.538 | 0.554 | −0.016 | 0.758 | 0.728 | +0.030 |
| **mean** | **0.500 ± 0.050** | 0.462 | **+0.039** (2/3 up) | **0.721** | 0.688 | **+0.034** (3/3 up) |

- **The screening bar is passed** (+0.015 with 2 of 3 seeds up). The S2 gain is noisy (t = 0.99),
  as E12a's was.
- **So, per the rule, it is extended to seeds 3–9**: VM d takes seeds 3–6 and VM c seeds 7–9,
  with the same configuration. The pre-registered prediction stands: +0.01 to +0.03 per seed.

### E13x, E13b, E13c results *(all runs finished 13:18 IST)*

Sources:
- `raw/E13_final_analysis.txt`: `e13_analysis.py`, sections A–E;
- `raw/E13_lane_summaries.txt`: the per-VM summaries;
- `mideval/figures/e13_qwen_vs_deberta.png`;
- every seed folder is on the HF repo under `Q8_fullq_lora/`, `Q8_fullq_lora_12ep/` and
  `Q8_alldata/`.

**E13x: Qwen3-8B LoRA at 10 seeds, against DeBERTa full question, 16 epochs (same seeds).**

| | Qwen3-8B LoRA | DeBERTa | paired |
|---|---|---|---|
| single model S2 | **0.476 ± 0.043** | 0.384 ± 0.030 | **+0.092 ± 0.041** (t = 7.1, 10/10 up) |
| single model S1 | **0.709 ± 0.031** | 0.614 ± 0.027 | **+0.095 ± 0.036** (t = 8.5, 10/10 up) |
| single model + logit adjustment, S2 | 0.538 ± 0.043 | 0.389 ± 0.019 | +0.149 (10/10 up) |
| 10-seed ensemble, argmax, S2 | 0.495 | 0.409 | |
| **10-seed system (ensemble + LA, nested CV), S2** | **0.543** (10 CV splits 0.549) | 0.405 (0.412) | **+0.136**, 95% [+0.054, +0.224] |
| 10-seed system, S1 | **0.746** | 0.648 | |
| system on the three 2-annotator reference sets | 0.498 / 0.536 / 0.537 (mean 0.524) | mean 0.382 | |

Against the predictions:
- **The single-model mean "regresses towards 0.42–0.48": held.** It is 0.476. The 3-seed 0.462
  was not an upward fluke; seeds 3–9 average 0.482.
- **"Above DeBERTa on at least 7 of 10 seeds": held, on 10 of 10,** for S2 and S1 alike.
- **"The system beats 0.405 by at least +0.03": held, by +0.138.** The bootstrap interval over
  items excludes 0.
- **Logit adjustment helps Qwen far more than DeBERTa** (+0.062 vs +0.005 per model). Qwen's raw
  predictions lean on frequent classes: across seeds 0–2 it predicted General 24 times, though
  General is in 113 reference sets (`raw/E13_Q8_topk_confidence.txt`). The one-scalar prior
  correction moves predictions into those classes.

**E13 hypothesis 3, at 10 seeds (where the gain falls).**
- **Per-class F1, Qwen − DeBERTa:**
  - large gains: Claims ignorance +0.403, Declining +0.220, Dodging +0.103, Deflection +0.089;
  - moderate gains: General +0.059, Explicit +0.057, Implicit +0.040;
  - Partial is still never predicted;
  - Clarification −0.144 (4 dev items).
- **In-set rate by annotator agreement:** unanimous +0.076, two labels +0.105, three labels
  +0.048.
- **Verdict: refuted as stated.** The gain does not fall mostly on the commitment boundary
  (Explicit ↔ Implicit, General ↔ Implicit/Explicit), and it is not larger on agreed items. It
  is broad, and largest on the Non-Reply classes, which are about recognising what kind of
  statement the answer is.

**E13b: 12 epochs (seeds 0–2), decided on the slice.**

| seed | selected epoch | slice F1 (E13) | dev S2 (E13) | dev S1 (E13) |
|---|---|---|---|---|
| 0 | 9 | 0.524 (0.460) | 0.531 (0.405) | 0.751 (0.647) |
| 1 | 11 | 0.523 (0.444) | 0.560 (0.426) | 0.740 (0.687) |
| 2 | 8 | 0.538 (0.431) | 0.539 (0.554) | 0.735 (0.728) |
| mean | | **+0.084, 3/3 up** | 0.543 (+0.082, reported only) | 0.742 (+0.055) |

- **H1 (undertrained at 3 epochs) is supported, and the registered rule says ADOPT.**
  - The slice selects epochs 8–11.
  - Training loss reaches about 0.02, yet the slice F1 does not fall: it plateaus around 0.51–0.54
    from epoch 8 on. There is no harmful overfitting within 12 epochs.
- **By the team's decision on 2026-10-02 (time), it stays a 3-seed screen for the mid-eval.** A 10-seed 12-epoch
  system is the first step afterwards.

**E13c: all of train, last of 3 epochs, at 10 seeds.**
- **Per model:** S2 +0.017 ± 0.045 (6/10 up, t = 1.2) and S1 +0.006 (6/10 up), against E13 on the
  same seeds.
- **Systems:** 0.575 (10 splits 0.562) against 0.543 (0.549). The bootstrap gives +0.028, 95%
  [−0.026, +0.092]. The system's S1 is lower: 0.711 against 0.746.
- **Verdict:** inside the predicted +0.01 to +0.03, but not distinguishable from zero at 10 seeds.
  This is the same pattern as E12a for DeBERTa: a small gain that the 3-seed screen overstated
  (+0.039).

**E (not pre-registered as a hypothesis; a check): a Qwen + DeBERTa probability mix.** The weight
and τ are fitted inside the nested CV. It gives 0.543 against Qwen alone at 0.549, and the CV puts
a mean weight of 0.79 on Qwen. **DeBERTa adds nothing to Qwen.**

**Cost.** 4 VMs, 1× RTX PRO 6000 each, from 11:10 to 13:20 IST, including the extension runs on
VMs c and d. JarvisLabs credit (API balance) went from ₹5,548.09 to ₹4,029.30, so this round cost
**₹1,518.79**. The project total is ₹1,850.70 of ₹5,880. Without gradient checkpointing a 3-epoch
seed takes 15–16 min and a 12-epoch seed about 59 min, peaking at 53 GiB.

---

## Deferred

- **An LLM in the re-ranking slot.** The encoder pipeline comes first; the
  cross-fitted candidates and fall-back fusion from E6 are ready for it.
- **Training on all of train.** Every run holds out 10% of train to choose its best
  epoch. Once an epoch count is settled, a final model could be retrained on
  100% with that fixed count — about 10% more data, at the cost of not being able
  to choose its checkpoint.
