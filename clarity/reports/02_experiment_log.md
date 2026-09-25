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
| Period | 2026-09-19 → 2026-09-25 |
| Hardware | 1× RTX PRO 6000 (96 GB), shared with other users' jobs |
| Backbone throughout | `microsoft/deberta-v3-large` (435M parameters) |
| Best result so far | **0.438** — 5-seed ensemble + post-hoc logit adjustment (E4) |
| Next | E8 (full question + Balanced Softmax + focal) — configured, waiting to launch |

---

## Contents

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
- [Summary of results](#summary-of-results)
- [E8 — Full question + Balanced Softmax + focal loss (planned)](#e8--full-question--balanced-softmax--focal-loss-planned-ready-to-launch)
- [E9 — A trained hierarchical classifier (planned)](#e9--a-trained-hierarchical-classifier-planned-after-e8)
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

## Summary of results

| id | system | dev S2 | dev S1 | status |
|---|---|---|---|---|
| E0c | DeBERTa-v3-large, single seed | 0.337 ± 0.044 | 0.586 ± 0.039 | baseline (frozen) |
| E2 | 5-seed ensemble | 0.365 | 0.596 | |
| **E4** | **ensemble + logit adjustment** | **0.438** | | **best** |
| E4 | ensemble + per-class multipliers | 0.394 | | overfits |
| E5 | ensemble + hard hierarchical routing | 0.369 | | no reliable effect |
| E6 | ensemble + encoder re-ranker | 0.354 | | negative |
| — | ChulaNLP DeBERTa-large (dev-selected) | 0.46 | 0.65 | published |

Submissions for the two main systems are packaged in `../submissions/`, although
Codabench is no longer accepting them.

---

## E8 — Full question + Balanced Softmax + focal loss *(planned; ready to launch)*

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

## E9 — A trained hierarchical classifier *(planned, after E8)*

**Why.** The organizers report that every system exploiting the taxonomy's
hierarchy outperformed flat nine-way classification.

**How it differs from E5.** E5 applied a hierarchy *after* training, by routing a
flat model's probabilities through the clarity branches, and found no reliable
effect. E9 trains the hierarchy into the model, which is a different question.

Design options, to be decided after E8:

| option | what | support in our data |
|---|---|---|
| auxiliary coarse loss | a second 3-way head trained alongside the 9-way head; prediction stays 9-way | cheapest; already implemented (`--head hier`) |
| marker-gate two-stage | first separate {Declining, Claims ignorance, Clarification} from the rest, then classify within each group | the split humans confuse least (leakage 0.103 vs 0.193 for the official cut — `MIDEVAL_PLAN.md` §2); but those three classes are not the weak ones (mean F1 0.365, level with the rest — E4) |
| official-cut two-stage | clarity level first, then the leaf within it | matches the published systems; E5's routing check found no gain on a flat model |

Whichever is chosen, it is built on E8's input and loss if E8 helps, so that E9
measures the hierarchy alone.

---

## Deferred

- **An LLM in the re-ranking slot.** The encoder pipeline comes first; the
  cross-fitted candidates and fall-back fusion from E6 are ready for it.
- **Training on all of train.** Every run holds out 10% of train to choose its best
  epoch. Once an epoch count is settled, a final model could be retrained on
  100% with that fixed count — about 10% more data, at the cost of not being able
  to choose its checkpoint.
