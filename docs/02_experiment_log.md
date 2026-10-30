# 02 — Experiment log

The encoder track from its first run to now, in the order things happened. Each
entry says **why** it was run, **what** was run, what **came out**, and what it
**changed** about the plan. Mistakes are recorded where they happened, because
several of them changed the plan as much as the results did.

All numbers are **dev Subtask-2 macro-F1** (308 items, three annotators,
multi-reference scoring) unless marked otherwise. "± " is the standard deviation
over seeds. Raw outputs behind the headline tables are in `docs/raw/`.

| | |
|---|---|
| Period | 2026-09-19 → 2026-10-02 (mid-evaluation); Phase 2 from 2026-10-06: E14 (M1) 10-06, E15 (M3) 10-14, E16 (M2) 10-14, E17 (M4) 10-15, E18 (freeze) 10-15; final push from 10-27: E19–E23 |
| Hardware | 1× RTX PRO 6000 (96 GB), shared with other users' jobs; E13 on JarvisLabs VMs with the same GPU; E14 on the lab server's GPU 0 plus two GPU-1 MIG slices (48 GB each) when free; E15–E17 on the lab server's GPU 0 (vLLM for the 27B LLM), plus the JarvisLabs Model API for the API arms ($0.26); E19–E22 on on-demand JarvisLabs RTX PRO 6000 VMs (the lab server is fully booked) |
| Backbone | `microsoft/deberta-v3-large` (435M parameters) through E12; `Qwen/Qwen3-8B-Base` + LoRA from E13 |
| Final system (E11, by a rule fixed in advance) | full question + 16 epochs, **10-seed** ensemble + logit adjustment: dev S2 **0.405** (0.412 over CV splits), dev S1 **0.648**; the 10-seed baseline system scores 0.428 / 0.601 — not distinguishable on S2 |
| Latest | E12: training on all of train is the one variant that helps (S2 +0.029 per model, 4/5 seeds; 5-seed system 0.489); the hierarchy of specialist encoders, boundary experts and model soups do not |
| Latest | **E13: Qwen3-8B + LoRA, the backbone as the one change, 10 seeds.** Single model S2 0.476 ± 0.043 vs 0.384 (+0.092, 10/10 up); **10-seed system S2 0.543 / S1 0.746 vs DeBERTa 0.405 / 0.648** (+0.136, 95% [+0.054, +0.224]). 12 epochs adopted on the slice (3-seed screen); all of train +0.017 per model, not distinguishable |
| Latest | **E14 (module M1, Phase 2 step 1) done 2026-10-06.** Qwen 12 epochs at 10 seeds: single model S2 0.525 ± 0.042 (+0.049 vs 3 epochs, 8/10 up); **10-seed system S2 0.598 / S1 0.743** (+0.057 vs 0.543, 95% [−0.003, +0.125]); 2-annotator mean 0.549. Cross-fitted train probabilities for all 3,448 rows. Candidate sets must be rank-based; u(x) = 1 − top prob; oracle headroom at δ = 0.3 is 0.727, so M3 goes ahead |
| Latest | **E15 (module M3) done 2026-10-14; dev numbers corrected the same day (a fusion bug, see E15 correction).** Local Qwen3.8-27B-FP8 over M1's top 4, no examples, fused on the 30% least confident items: **dev S2 0.669 (10 splits 0.660), +0.067 [+0.009, +0.124] over M1; S1 0.798 (+0.055); 0.36 LLM calls/item**, against TeleAI's 0.617 / 0.812 at 1.94 calls. 2-annotator mean 0.589. API spend $0.26 *E16 caveat: averaged over seed subsets, M3 adds +0.02 to +0.03; the 10-seed point is likely optimistic* |
| Latest | **E16 (M2 completion) done 2026-10-14.** Prompt ablation: `full` kept (on dev, full > no guide > no definitions > bare; within noise). Measured cost: an M1 seed 0.052 GPU-s/item, an M3 call 0.103, so the 10-seed M1 is ~93% of the cost. Frontier: M3 adds +0.02 to +0.03 (δ = 0.3) at every K ≤ 5; K = 10's +0.071 is an outlier, so **E15's 0.669 is likely optimistic**. K = 1 + M3 at δ = 1 matches the 10-seed M1 alone (0.593 vs 0.598) at 3.4× less cost |
| Latest | **E17 (M4) done 2026-10-15.** One distilled student (soft targets from M3) instead of 10 seeds: single model S2 0.573 vs 0.525 (+0.048, 9/10), S1 +0.031; **student + M3 at δ = 0.5: 0.608 / 0.790 at 0.111 GPU-s per item vs the 10-seed M1 alone at 0.598 / 0.743 and 0.522 GPU-s (4.7× cheaper)**. Adopted as the cost operating point: no undistilled point is cheaper and better |
| **Current system (2026-10-15)** | Two operating points on dev. **Accuracy:** 10-seed M1 + M3 at δ = 0.3, S2 0.669 (10 splits 0.660) / S1 0.798, 0.36 LLM calls and 0.558 GPU-s per item. **Cost:** E17 student + M3 at δ = 0.5, S2 0.608 / S1 0.790 (mean of 10 students), 0.57 calls and 0.111 GPU-s per item. TeleAI (1st place): 0.617 / 0.812 at 1.94 calls |
| Latest | **E18 (freeze numbers) done 2026-10-15.** Final dev numbers for both operating points: accuracy point (M1 × 10 + M3, δ = 0.3) S2 0.669 / 0.660 over 10 splits, S1 0.798, 2-annotator 0.589; cost point (student + M3, δ = 0.5) S2 0.608 ± 0.027, S1 0.790, 2-annotator 0.567, at 0.21× the GPU cost of M1 (0.27× after E19's cost correction). Ensembles of distilled students do not help (≤ +0.02; the students lack diversity), so there is no middle operating point. Figure: `docs/figures/E18_frontier.pdf` |
| Latest | **E19 Tier 0 done 2026-10-27** (`raw/E19_tier0.txt`). Two corrections to E15–E18. (1) **Refitting λ and τ inside each bootstrap replicate,** the accuracy point's gain over M1 is +0.037 [−0.044, +0.120] (P(≤ 0) = 0.20), no longer significant; the earlier +0.067 [+0.009, +0.124] held them fixed. (2) **M3 costs 0.151 GPU-s per call, not 0.103,** so the cost point is 0.139 GPU-s (0.27× M1, not 0.21×) and the accuracy point 0.576. What holds: confidence routing beats random routing (0.669 vs 0.604 ± 0.015 at δ = 0.3); the gain comes from M3, not from re-thresholding; and the LLM's value changes sign with M1's uncertainty |
| **Final system (2026-10-30, E23)** | One 27B copy (VM), order-averaged q, normalized test text. **Accuracy point:** M1 × 10 + M3, δ = 0.3: dev S2 **0.650** (10 splits 0.639) / S1 0.788, 0.36 calls. **Cost point:** student + M3, δ = 0.5: S2 0.608 (10 splits 0.606) / S1 0.785 (mean of 10 students), 0.139 GPU-s. Lab-record reference: 0.669 / 0.608. Test files: `submissions/E23_*` |
| Latest | **E19–E22 done 2026-10-29/30** (`raw/E21_E22_analysis.txt`, `raw/E20_router.txt`, `raw/E23_systems.txt`). The 27B on the JarvisLabs VM agrees with the lab server's copy on only 92.5% of dev (faith gate failed), so everything was re-scored on one copy. Order-averaged q adopted (E19 B). LLM alone over 9 labels 0.591 < M1 (E19 A). Distillation: the M3-taught student beats self-distillation by only +0.012 single-model (H-E fails), +0.026 as a system. On leak-free train rows the LLM's value map is 0/−7/+2/+7/+12: the big losses on confident items were sibling leakage; the learned router loses to confidence routing (E20). Test answers carry transcript artifacts that cost M1 0.048 S2 on dev when injected, so the test files use normalized text (E21). Cached prompt halves the 27B's cost but changes its answers (E22a). Ladder 0.600 at 0.221 GPU-s, not adopted (E22c). E19 D and F not run |
| Latest | **E24 pre-checks done 2026-10-30** (`raw/E24_prechecks.txt`, CPU only), for the research extension in [`07_research_extension_plan.md`](07_research_extension_plan.md). 61.5% of 3,079 train sibling pairs (same answer, different sub-question) carry different labels; sibling disagreement 72% vs 43% between annotators on one item, so evasion is relational, not annotator noise. M1 reads the question (same label on only 19% of disjoint dev sibling pairs). Question–answer relevance does not measure answering (AUC 0.566 for Clear Reply). Five candidate modules rejected: epistemic routing, sibling decoding, single-order debiasing, conformal set routing (0.564–0.617 vs 0.641–0.663), metric-optimal rule (no gain over CV-tuned logit adjustment) |
| Latest | **E25 done 2026-10-30** (`raw/E25_analysis.txt`). The instrument reads the question causally (unrelated question: Clear Reply 25% → 1%). Question or speaker: a no-stake LLM does not evade where Presidents did (OR 1.18), but in the President's role the same models evade on the same questions (OR 2.74, exploratory) at the same rate; a follow-up turns 66% of evasions into replies. Detection needs most of the answer, and only fine-tuning binds labels to questions |
| Running | nothing. **Experiments closed.** No JarvisLabs instance exists; credit left ₹230.59 |

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
| E12a | full question, 16 epochs, **trained on all of train**, last epoch (5 seeds) | **0.416 ± 0.020** | 0.614 ± 0.016 *(corrected 2026-10-02 from 0.613 ± 0.017; raw/E12a_alldata_analysis.txt gives 0.6137 ± 0.0164)* | **0.489** | **helps S2**: +0.029 per model (4/5 up); S1 −0.017; 5-seed system only |
| E12b | hierarchy: Non-Reply gate + branch specialist encoders | 0.345 *(3 seeds)* | 0.608 | 0.390 *(3 seeds)* | negative; specialists alone level on 5 seeds |
| E12c | boundary experts for the three most-confused pairs (3 seeds) | 0.383 | 0.615 | | no effect |
| E12d | model soup of 10 trained models | 0.267 *(uniform)* | 0.471 | | negative |
| E13 + E13x | **Qwen3-8B-Base + LoRA** instead of DeBERTa (same input, rows, slice; 3 epochs; 10 seeds) | **0.476 ± 0.043** | **0.709 ± 0.031** | **0.543** (S1 0.746) | **best by far**: +0.092 S2 / +0.095 S1 per model over DeBERTa, 10/10 seeds; system +0.136 [+0.054, +0.224] |
| E13b | Qwen, 12 epochs instead of 3 (3 seeds) | 0.543 *(3 seeds)* | 0.742 | | slice F1 +0.084 (3/3) → **adopted**; best epochs 8–11; 10 seeds after the mid-eval |
| E13c | Qwen, all of train, last epoch (10 seeds) | 0.492 ± 0.045 | 0.715 | 0.575 | +0.017 per model (6/10); system +0.028 [−0.026, +0.092]: not distinguishable |
| **E14** | **M1: Qwen 12 epochs at 10 seeds** (+ 5-fold cross-fitting; C(x), u(x)) | **0.525 ± 0.042** | **0.729** | **0.598** (S1 0.743) | **new best system**: +0.057 vs E13x [−0.003, +0.125]; per model +0.049 (8/10); K = 5 is not enough |
| E15 | **M3: + Qwen3.8-27B-FP8 adjudicator over M1's top 4, fused, δ = 0.3** (0.36 calls/item) | 0.603 *(LLM alone within C)* | | **0.669** (10 splits 0.660; S1 **0.798**) | **new best system**: +0.067 [+0.009, +0.124] over M1; corrected 2026-10-14 (fusion bug). *E19: +0.037 [−0.044, +0.120] with refitted parameters* |
| E17 | **M4: one Qwen3-8B student trained on soft targets 0.5·gold + 0.5·M3's q** (10 seeds) | **0.573** | **0.761** | 0.608 with M3 at δ = 0.5 (S1 0.790; 0.111 GPU-s/item) | **cost operating point**: beats the 10-seed M1 alone at 4.7× less cost. *E19: 0.139 GPU-s, 3.8× less* |
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
- [E14 — Module M1: the 12-epoch Qwen at 10 seeds, cross-fitting, candidate sets and uncertainty](#e14--module-m1-the-12-epoch-qwen-at-10-seeds-cross-fitting-candidate-sets-and-uncertainty-registered-2026-10-06)
  - [E14 results](#e14-results-all-12-runs-finished-2026-10-06-2205-ist)
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
`code/qevasion/`, checked to give identical results.

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
| ChulaNLP (2nd), DeBERTa-large top-5 → Kimi-K2 few-shot *(corrected 2026-10-02: the overview paper says RoBERTa; ChulaNLP's own paper, abstract and Table 3, says DeBERTa-large)* | 0.52 | 0.70 |
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

Config: `code/experiments/E8/`. Launch: `python -m models.encoder` with the arguments in `code/experiments/E8/`.

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

Raw output: `docs/raw/E9_hier_analysis.txt`, `docs/raw/E9_hier_decision_rules.txt`.

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

Raw output: `docs/raw/E8b_fullq_ce_analysis.txt`, `…_decision_rules.txt`, and
`docs/raw/E8b_vs_E0_E8_compare.txt` (`compare_runs.py L0_large_base E8b_fullq_ce E8_fullq_bal_focal`).

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

Raw output: `docs/raw/E10_base_16ep_{analysis,decision_rules}.txt`,
`docs/raw/E10_base_vs_E0_compare.txt`. This is E0 with 16 epochs instead of 8,
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

Raw output: `docs/raw/E10_fullq_16ep_{analysis,decision_rules}.txt`,
`docs/raw/E10_fullq_vs_base_compare.txt`, `docs/raw/E10_final_system_bootstrap.txt`,
`docs/raw/E10_ensemble_size.txt`.

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

Raw output: `docs/raw/E11_replication.txt`, regenerated by
`python -m preregistered_analyses.e11_replication` (CPU, seconds); per-configuration analyses in
`docs/raw/E11_*_{analysis,decision_rules}.txt`.

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
`docs/raw/decode_variants.txt`). Three alternatives to logit adjustment, each
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
(`docs/raw/E12_design_confusion.txt`; the 10-seed full-question ensemble on the
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
(`docs/raw/last_vs_selected_epoch.txt`). Across the 10 full-question seeds the
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

Raw: `docs/raw/E12_interim_seeds01.txt` (`python -m models.hier_combine`). Paired
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

Raw: `docs/raw/E12b_hierarchy_3seeds.txt` (`python -m models.hier_combine`).
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

Raw: `docs/raw/E12b_specialists_5seeds.txt`
(`python -m models.hier_combine --focus "flat gate + specialists"`).

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

Raw: `docs/raw/E12a_vs_flat_compare.txt`, `E12a_alldata_{analysis,decision_rules}.txt`,
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
(`docs/raw/mideval/mideval_analysis.txt`) adds two facts:
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

Code: `code/models/llm_classifier.py`. Smoke-tested on CPU with Qwen3-0.6B-Base, including an
interrupted epoch resumed from its checkpoint. Outputs follow `encoder.py`'s layout, so
`analyze.py` and `decide.py` read them unchanged.

**Hypotheses and predictions** (paired with E10_fullq_16ep on seeds 0–2: S2 0.344 / 0.370 /
0.428, mean 0.381; S1 0.591 / 0.627 / 0.640, mean 0.619):

1. **Screening (the E12 bar):** S2 at least +0.015 over DeBERTa with at least 2 of 3 seeds up.
   If it passes, extend to seeds 3–9 and compare 10-seed systems as in E11.
2. **Size:** if knowledge is the gap, S2 single-model mean 0.43–0.50 (+0.05 to +0.12) and S1
   +0.02 to +0.05. A gain below +0.015 is read as: knowledge is not what the encoder lacks, and
   the budget moves to the LLM-over-candidates cascade (the timeline in the Conclusion of `report/mid_evaluation_report.pdf`).
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
and `raw/E13_Q8_fullq_lora_analysis.txt` (`analyze.py`). Outputs: `runs/Q8_fullq_lora/`.

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
- `docs/figures/e13_qwen_vs_deberta.png`;
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

## E14 — Module M1: the 12-epoch Qwen at 10 seeds, cross-fitting, candidate sets and uncertainty *(registered 2026-10-06)*

*Written before any E14 run started.* This is the first step of the Phase 2 cascade (report §7;
[`05_phase2_plan_and_budget.md`](05_phase2_plan_and_budget.md)).

**What was already known when this was written.** The registration does not pretend otherwise.
- **E13b's seeds 0–2 dev numbers are known.** S2 0.531 / 0.560 / 0.539.
- **The candidate-set design was studied on those three seeds.** The study was done on the train
  slice; its dev numbers were printed alongside. Output: `raw/E14_m1_design_seeds012.txt`.
- **What it showed:**
  - The 12-epoch models are overconfident: mean top probability 0.81 on the slice and 0.87 on dev,
    against 0.57 / 0.55 for 3 epochs.
  - On the slice, any set rule needs 5–7 labels to cover the gold label for 90–95% of items. The gold
    label ranks 4th or lower for 22% of slice items.
  - The logistic u(x) did not beat 1 − top probability: AUROC 0.682 vs 0.696 on the slice.
  - The 3-seed 12-epoch system was 0.555 against 0.565 for 3 epochs on the same seeds.

**Why.** M1 is the cascade's base:
- **The 12-epoch schedule is adopted but not established.** It was adopted on the slice (E13b: slice
  F1 +0.084, 3/3 seeds), but only on 3 seeds. A system needs 10.
- **M3 and M4 need cross-fitted train candidates.** M3 needs them for exemplar design, and M4 for
  distillation targets.
- **C(x) and u(x) decide how much an LLM stage can add, and on which items.**

**Runs** (`code/experiments/E14/`, W&B + HF). They run on the lab server: GPU 0 (`lane_gpu0_a`), plus our GPU-1 MIG slice (`lane_gpu1_a`, same runs in reverse order) whenever `gpu_watch.sh` finds it free. From 2026-10-06 09:50, the second GPU-1 slice (`lane_gpu1_b`, GI 2) runs under the same rules. A run stopped by contention resumes from its last epoch. Launched with `bash code/launch/gpu_watch.sh E14 start`.

| run | what | change from E13b |
|---|---|---|
| `Q8_fullq_lora_12ep` seeds 3–9 | E13b's configuration | none (new seeds) |
| `Q8_12ep_crossfit` fold0–4 | train on 4/5 of all 3,448 rows, 12 epochs, last epoch; the held-out fold's probabilities are the out-of-fold predictions | rows (all of train, by fold) and `--select last`, since choosing the epoch on the held-out fold would leak |

**Fold partition.** It is `data/splits/train_5fold.json` (stratified, SPLIT_SEED 12345), identical to
the DeBERTa folds of E6 (`F0_oof`).

**Hardware.** The lab server's GPU is a 300 W Max-Q card, where JarvisLabs used a 600 W card; it is the
same architecture, with torch 2.11 / transformers 5.17 / peft 0.20. The runs go without gradient
checkpointing, as E13b did. If a lab-mate's job leaves too little memory, the queue uses gradient
checkpointing instead: the same computation, slower. Each run's `config.json` records which.

**Code.**
- `cascade/candidates.py` builds C(x) and u(x) from saved probabilities.
- `preregistered_analyses/e14_m1_analysis.py` runs sections A–E below.
- Everything is fitted on the slice: τ_C, a temperature T, the set threshold and the u(x) regression.
  Dev only reports.

**Hypotheses and predictions.**
- **H1, single models, 10 seeds.** 12 epochs gives S2 0.50–0.54 (3 epochs: 0.476 ± 0.043).
  - The paired gain is +0.03 to +0.07, with at least 7 of 10 seeds up.
  - S1 gains +0.01 to +0.04.
- **H2, systems.** The 10-seed 12-epoch system (ensemble + LA, nested CV) scores S2 0.55–0.60
  (3 epochs: 0.543).
  - The gain over 3 epochs is smaller than the per-model gain, and its bootstrap interval includes 0.
    This is the E11/E13c pattern, made stronger by overconfidence: ensembling and LA add less to
    sharper models (LA added +0.014 per 12-epoch model vs +0.066 at 3 epochs, seeds 0–2).
  - The schedule was chosen on the slice, so whatever the dev result, **M1 uses 12 epochs**. Dev is
    reported, not used to choose.
- **H3, candidate sets.** At α = 0.10 on the slice, adaptive sets after temperature scaling need
  5.5–7 labels on average.
  - The cross-fitted train rows confirm the slice coverage to within 0.03; this is an independent
    check of the calibration.
  - A fixed top-3 covers 0.75–0.82 of slice items and 0.90–0.94 of dev items (any reference).
- **H4, uncertainty.** u(x) reaches an AUROC of 0.66–0.74 on the slice (5-fold CV) and on dev.
  - If the logistic regression does not beat 1 − top probability by 0.01 on the slice, M2 uses
    1 − top probability. This is the expected outcome.
- **H5, headroom (the go/no-go for M3's budget).** An oracle restricted to the top-3 set, applied to
  the 30% most uncertain dev items, reaches S2 ≥ 0.65.
  - If it reaches less than M1's system + 0.03, M3 is de-scoped and its Jarvis budget is held back.
- **H6, cost.** K = 5 seeds lands within 0.01 S2 of K = 10 on average over random subsets.

**Decision to take with the user before M3 (not a dev choice).** How C(x) feeds M3:
- **(a) a fixed small set:** top-3 by the M1 ranking;
- **(b) all nine labels scored by the LLM** and fused with p̄, with C(x) used only to choose the
  definitions and examples in the prompt.

The slice shows that a set that is both small and high-coverage does not exist for this model.

**Exit criterion (report Table 5), restated.** "Gold in C(x) for ≥ 95% of slice items" is met only by
sets of ~7 labels, so it is replaced by H3's coverage check and H5's headroom. The reason is the
finding above, made on the slice before the runs.

**Cost.** Free (lab server). Estimated 1.2–1.6 h per run, 12 runs, ~15–19 h of wall time. A
20-step pilot measures it first.

### E14 results *(all 12 runs finished 2026-10-06 22:05 IST)*

Sources:
- `raw/E14_m1_analysis.txt`: `e14_m1_analysis.py`, sections A–E;
- every run folder is on the HF repo under `Q8_fullq_lora_12ep/seed<k>/` and `Q8_12ep_crossfit/fold<k>/`;
- curves in W&B, groups `Q8_fullq_lora_12ep` and `Q8_12ep_crossfit`.

**The runs.** In the "lane" column, *ckpt* means the run used gradient checkpointing (same computation,
slower).
- **GPU 0 alone (seeds 3, 7, 8):** 75–120 min per run.
- **GPU 0 shared with a lab-mate's vLLM (seeds 4–6):** 160–180 min.
- **A GPU-1 MIG slice:** about 200 min.
- **Seed 9:** restarted at the epoch-8 boundary with checkpointing, to leave room on GPU 0 for a lab-mate.
- **Fold 4:** stopped two minutes into its first epoch on our GI-1 slice when a lab-mate's job arrived,
  then trained on GPU 0.
- **Lost work:** none apart from those two minutes.

| run | lane(s) | selected epoch | slice / held-out fold F1 | dev S2 | dev S1 | final train loss | minutes |
|---|---|---|---|---|---|---|---|
| seed 3 | GPU 0, ckpt | 5 | 0.479 | 0.570 | 0.769 | 0.022 | 120 |
| seed 4 | GPU 0, ckpt (shared) | 8 | 0.487 | 0.530 | 0.740 | 0.025 | 178 |
| seed 5 | GPU 0, ckpt (shared) | 7 | 0.490 | 0.436 | 0.646 | 0.022 | 170 |
| seed 6 | GPU 0, ckpt | 11 | 0.490 | 0.467 | 0.664 | 0.025 | 159 |
| seed 7 | GPU 0 | 8 | 0.500 | 0.540 | 0.754 | 0.026 | 75 |
| seed 8 | GPU 0 | 10 | 0.498 | 0.557 | 0.774 | 0.022 | 75 |
| seed 9 | GPU 0, then ckpt from epoch 9 | 4 | 0.477 | 0.515 | 0.720 | 0.026 | 82 |
| fold 0 | GI-2 slice, ckpt | 11 (last) | 0.466 | 0.563 | 0.740 | 0.019 | 200 |
| fold 1 | GI-2 slice, ckpt | 11 (last) | 0.503 | 0.560 | 0.720 | 0.024 | 197 |
| fold 2 | GPU 0, ckpt | 11 (last) | 0.472 | 0.578 | 0.741 | 0.022 | 172 |
| fold 3 | GI-2 slice, ckpt | 11 (last) | 0.526 | 0.494 | 0.713 | 0.015 | 198 |
| fold 4 | GI-1 slice (stopped), GPU 0, ckpt | 11 (last) | 0.478 | 0.512 | 0.756 | 0.027 | 159 |

For a fold, the F1 column is the macro-F1 on its held-out fold of train, and its dev scores are those of
a model trained on 80% of train. Seeds 0–2 are E13b's runs (dev S2 0.531 / 0.560 / 0.539).

**Against the predictions.**

| | prediction | result | verdict |
|---|---|---|---|
| **H1** single model S2, 10 seeds | 0.50–0.54 | **0.525 ± 0.042** (3 epochs: 0.476 ± 0.043) | held |
| | paired gain +0.03 to +0.07, ≥ 7/10 up | **+0.049 ± 0.062** (t = 2.48, **8/10** up) | held |
| | S1 +0.01 to +0.04 | +0.020 (t = 1.11, 6/10 up) | held, weakly |
| **H2** 10-seed system S2 | 0.55–0.60 | **0.598** (10 CV splits 0.601); 3 epochs 0.543 (0.549) | held |
| | gain smaller than per model; CI includes 0 | +0.057, 95% [−0.003, +0.125], P(≤0) = 0.032 | gain **not** smaller; CI includes 0 (just) |
| | system S1 | 0.743 vs 0.746 | flat |
| | 2-annotator sets (test regime) | 0.540 / 0.571 / 0.536, mean **0.549** (3 epochs 0.524) | — |
| **H3** APS sets at α = 0.10 need 5.5–7 labels (slice) | 5.5–7 | 5.94 | held |
| | cross-fitted train rows confirm slice coverage within 0.03 | APS: 0.756 vs 0.977 (**no**); rank-conformal top-k: 0.896 vs 0.910 (yes) | **mixed** |
| | fixed top-3: slice 0.75–0.82, dev 0.90–0.94 | 0.812 / 0.925 (train out-of-fold 0.768) | held |
| **H4** u(x) AUROC 0.66–0.74; logistic must beat 1 − top prob by 0.01 | | logistic 0.683 (slice CV) / 0.707 (dev); 1 − top prob 0.714 / 0.712 | AUROC held; logistic does **not** beat the baseline, so **M2 uses 1 − top probability** (as expected) |
| **H5** oracle within top-3 at δ = 0.3 ≥ 0.65, and ≥ M1 + 0.03 | | **0.727** (M1 0.598; at δ = 0.1 / 0.2: 0.635 / 0.684) | held: **M3 is worth building** |
| **H6** K = 5 within 0.01 of K = 10 | | K = 5: 0.555 ± 0.029 vs K = 10: 0.598 | **failed**: keep 10 seeds |

**What the results say.**
- **Twelve epochs are confirmed.** The per-model gain holds at 10 seeds. The system gain of +0.057 is
  as large as the per-model gain, against our prediction.
  - Logit adjustment helps the overconfident 12-epoch models little: single model + LA scores 0.533 at
    12 epochs vs 0.538 at 3. Averaging seeds makes up for it.
  - Single 12-epoch models are sharp but disagree with each other, and the ensemble benefits. The
    K-curve shows this: it rises from 0.509 at K = 1 to 0.598 at K = 10. The cost lever K is therefore
    weak, since fewer seeds cost accuracy.
- **The single-pass system reaches 0.598 dev S2 with no LLM call.** That is the cascade's whole target
  (0.60), and 0.019 below TeleAI's multi-call pipeline (0.617). The bootstrap interval still includes
  the 3-epoch system, so "better than E13" is likely (P(≤ 0) = 0.03) but not established.
- **Mass-based candidate sets do not transfer between ensembles.** The temperature and threshold were
  fitted on the 10-seed ensemble's slice probabilities. The cross-fitted rows come from single fold
  models, which are sharper, so the same mass threshold gives smaller sets with 0.76 coverage.
  Rank-based sets (fixed k, or conformal k) transfer within 0.015. **So C(x) should be defined by rank.**
  M4 uses single-model cross-fitted candidates, and a fixed top-k behaves the same there as on dev.
- **There is room for an LLM stage, but it must beat a high bar.**
  - An oracle restricted to the top 3 of the 30% most uncertain items reaches 0.727 (top-5/6: 0.765).
  - Picking at random inside the top 3 *lowers* S2: 0.587 at δ = 0.3.
  - The adjudicator must therefore be clearly better than chance among candidates M1 already ranks well.

**Deviations from the registration.** None changes the computation.
- **Gradient checkpointing:** used whenever a lab-mate's job left less than 60 GB free. Each run's
  `config.json` and the queue log (`logs/E14.log`) record which.
- **A second MIG slice:** GI 2, at the user's request on 2026-10-06 09:50, under the same give-it-back
  rules.
- **Infrastructure fixes during the run:**
  - tmux session targets now use exact names;
  - the queue log reports the selected epoch's score instead of the last epoch's;
  - `QUEUE_NEED_GB` forces checkpointing on demand.

**Decisions this leaves for M3** (see `05_phase2_plan_and_budget.md` §2.1):
1. **C(x) is rank-based.** Either a fixed top-3 (dev any-reference 0.925, train out-of-fold 0.768), or
   all nine labels with p̄ fused in. The data favour top-3 or top-4 for the prompt, with fusion so the
   LLM can fall back on M1.
2. **u(x) = 1 − top probability** of the 10-seed adjusted ensemble.
3. **Deferral budget:** δ ≈ 0.2–0.3 is where the oracle gain per call is largest (+0.047 at 0.1, +0.086
   at 0.2, +0.129 at 0.3).

**Cost.** Free: lab server, ~30 GPU-hours over 20 h of wall time, from 3 lanes (GPU 0 plus two MIG
slices when they were free). On JarvisLabs the same runs would have cost roughly ₹2,300. The credit
balance is unchanged at ₹4,029.30.

---

## E15 — Module M3: a candidate-constrained LLM adjudicator, fused with M1 *(registered 2026-10-13)*

*Written before any M3 scoring run. The API probe (step 0 below) may change only the items marked
"probe-dependent"; any such change is recorded here before the screen starts.*

**Why.** E14 left M1 at dev S2 0.598 with no LLM call, and showed where an LLM can help.
- **M1 ranks better than it decides.** On the 30% least confident dev items, M1's top label is in the
  reference set for 0.45 of items, but the top 4 contain a reference label for 0.95. On the slice
  (one gold label) the figures are 0.41 and 0.84; on cross-fitted train rows, 0.31 and 0.81.
- **M1's first choice still carries information.** Within the top 3, its first choice is right about
  twice as often as its second (slice: 0.41 / 0.22 / 0.14), and picking at random inside the top 3
  lowers S2 (E14: 0.587). The LLM must therefore be *fused* with M1, not substituted for it.
- **The oracle headroom is large.** Re-deciding the 30% least confident items perfectly within the
  top 3 gives 0.727 (E14 H5).

**Two facts found while designing (2026-10-13, train only), which shape the protocol.**
- **Sibling rows.** 69% of train rows share their answer text with another row (other sub-questions of
  the same answer; 72% of those groups carry different labels). The slice and the 5 folds were split
  by row, so 67% of slice rows have their answer text in the training part. Test shares no answer and
  no interview with train; dev shares 1% of answers and 38% of interviews.
  - *Consequence 1:* exemplars are retrieved from train rows with a **different answer and a different
    interview** (URL), for every split, to match test.
  - *Consequence 2:* M1's probabilities on slice and cross-fitted rows come from a partly-seen regime,
    so the fusion weights for the reported system are fitted by **nested CV on dev**, as the report says,
    not on train rows.
- **Confusions among deferred train rows** (cross-fitted M1, 30% least confident, gold → M1):
  Explicit/Implicit 115, General/Implicit 68, Deflection/Implicit 60, Explicit/General 55,
  Dodging/Implicit 51, Deflection/General 48, Deflection/Dodging 46. These match the report's dev
  pairs, so the confusion guide covers Explicit–Implicit, Implicit–General, Explicit–General,
  Deflection vs Implicit/General, and Dodging–Deflection.

**Decisions taken with the user (2026-10-13), before any run.**
1. **Candidates C(x) = M1's top 4 by rank** (rank-based, per E14), shown to the LLM in the fixed label
   order of the taxonomy, not in M1's order, so the LLM is not anchored on M1's ranking.
2. **Scoring without reasoning.** One request per item, one output token; q(c | x) is the softmax of the
   option letters' log-probabilities over C(x). *Probe-dependent:* if an API model returns no
   log-probabilities, q is estimated from 5 samples at temperature 1 (cached input makes the repeats cheap).
3. **Fusion** over C(x): λ log p̄ + (1 − λ) log q − τ log π; items outside the deferred set and labels
   outside C(x) keep M1's decision (log p̄ − τ log π). λ ∈ {0, 0.1, …, 1} and τ on the E4 grid are
   fitted jointly by 5-fold nested CV on dev (10 splits reported), as for τ since E4.
4. **Prompt.** A fixed preamble (task, definitions of all 9 labels, the confusion guide above) that is
   identical for every item and therefore cached, then up to k retrieved examples, then the item
   (sub-question, full question, full answer), then the lettered candidates. Items are never truncated.
   - **Examples** are retrieved by cosine similarity of `Qwen/Qwen3-Embedding-0.6B` embeddings of
     "sub-question + answer", restricted to train rows whose gold label is in C(x) and which share neither
     the answer nor the interview with the item. They are taken **round-robin over the candidates**
     (nearest of each candidate first), so every candidate is illustrated and the examples do not vote by
     majority. Examples are shown in full; if a prompt would exceed 12k tokens, the least similar examples
     are dropped, never cut.
   - **Shorter prompts are not adopted.** The example count is measured instead (screen, below).
5. **Precision:** FP8 or BF16 only, from the publisher's own checkpoints; no 4-bit.
6. **Models, one or two per size class:**
   - *~30B:* `Qwen/Qwen3.8-27B-FP8`, served locally with vLLM 0.30.0 on the lab server's GPU 0 (free),
     thinking disabled; and Gemma 4 31B through the JarvisLabs Model API, screen only.
   - *large:* DeepSeek V4 Flash (304B MoE) through the JarvisLabs Model API. It is the successor of the
     model in TeleAI's pipeline.
   - Not used: GLM-5.3 (about 10× the price), self-hosted Mistral-Small-4 (dearer than the API and
     smaller), any 4-bit model.
   - **Deviation from report §7:** the report planned a locally served LLM. For an API model, cost is
     reported as calls, input/output tokens and ₹ per item (TeleAI's units); for the local model, as
     GPU-seconds per item. Every response is saved, so results are reproducible from disk even if a hosted
     model changes.
7. **Confidence gating of what gets scored.** Dev, test and slice are scored in full (so the whole
   accuracy-vs-calls curve of M2 comes for free); train only for the 50% least confident cross-fitted
   rows (M4's distillation inputs; gold labels serve elsewhere).

**Steps.**
- **Step 0, API probe** (`cascade/api_probe.py`, cap $1). For each API model: exact model id;
  log-probabilities with `top_logprobs`; how to disable thinking; cached-token accounting on a repeated
  prompt; determinism at temperature 0; rate limits; cost per request from `usage`. The user notes the
  credit balance before and after, to confirm API use is billed to the same credit.
- **Step 1, screen** (`cascade/adjudicate.py`). 300 rows drawn (seed 15) from the 30% least confident
  cross-fitted train rows, excluding the slice. Single gold label. Arms:
  - Qwen3.8-27B local with k ∈ {0, 4, 10} examples (free; the example-count measurement);
  - Gemma 4 31B (API) and DeepSeek V4 Flash (API), each at the k chosen from the Qwen arm;
  - Qwen3.8-27B through the API at the same k: a **faithfulness check** of the hosted service against our
    local copy of the same FP8 weights.
- **Step 2, full scoring** with the adopted configuration: slice, dev, test, and 50% of train. The local
  Qwen arm is run on everything in any case (free), so the paper has a size comparison on dev.
- **Step 3, fusion and the M2 curve** (`cascade/fuse.py`), on dev with nested CV; dev is scored once per
  registered configuration.

**Screen metrics (train rows, single gold).** For each arm: accuracy of argmax q within C(x) (the LLM
alone); fused accuracy, with λ and τ fitted by 5-fold CV inside the 300 rows; negative log-likelihood of
the gold label under q (rows with gold in C(x)); and complementarity: the share of rows where the LLM is
right and M1 is wrong, and the reverse.

**Hypotheses and predictions.** On the 300 screen rows, M1's top-1 accuracy is about 0.31, gold is in
the top 4 for about 0.81, and random choice within the top 4 gives about 0.20.
- **H1 (the LLM alone):** within-C accuracy is 0.35–0.45 for Qwen3.8-27B, 0.40–0.50 for DeepSeek V4
  Flash, and 0.33–0.43 for Gemma 4 31B. Every arm beats M1's 0.31.
- **H2 (examples):** for Qwen3.8-27B, k = 10 beats k = 0 by +0.03 to +0.08 in within-C accuracy, and
  k = 4 is within 0.02 of k = 10.
- **H3 (fusion):** fused accuracy beats M1 by +0.05 to +0.12 on the screen rows, and beats the LLM alone.
- **H4 (faithfulness):** API and local Qwen3.8-27B agree on argmax q for ≥ 95% of rows, with mean
  Jensen–Shannon divergence below 0.02.
- **H5 (dev system, the headline):** M1 + M3 at δ = 0.3 scores dev S2 **0.61–0.64** (M1: 0.598). The
  paired bootstrap over items excludes 0 if the gain is ≥ +0.025; this comparison is tighter than our
  seed comparisons, because 70% of items keep the same prediction in both arms.
- **H6 (the M2 curve):** most of the gain arrives by δ = 0.3; δ = 1 is not better than δ = 0.5 by more
  than 0.01.
- **H7 (Subtask 1):** S1 changes by +0.00 to +0.03; Explicit–Implicit is the one large confusion that
  crosses clarity levels.

**Decision rules (fixed now).**
- **Example count:** k = argmax of fused screen accuracy over {0, 4, 10} for Qwen3.8-27B; if another k
  is within 0.01 of the best, k = 10 (the report's design) is preferred.
- **Model:** DeepSeek V4 Flash is adopted for step 2 if its fused screen accuracy beats the better
  ~30B arm by ≥ 0.02 (6 of 300 rows); otherwise the better ~30B arm is adopted, preferring the free
  local Qwen when the two ~30B arms are within 0.01.
- **Faithfulness:** if H4 fails (agreement < 0.90), API results are reported with that caveat, and the
  adopted configuration prefers a local model when within 0.02.
- **Go/no-go:** if no arm's fused screen accuracy beats M1 by ≥ 0.03, step 2 runs only the free local
  arm, and M3 is reported as a negative result.

**Cost.** A hard cap of **$8 (≈ ₹700) of API spend for all of E15**, enforced in code by a ledger
(`runs/M3/api_ledger.jsonl`) that refuses any request beyond it. Estimate: probe < $0.5; screen ≈ $1
(Gemma $0.2, DeepSeek $0.2, Qwen through the API $0.5); DeepSeek on all ~2,600 items ≈ $1.7 (≈ ₹145).
Expected total ≈ $3–4 (≈ ₹260–350). The local Qwen arm is free.


### E15 step 0: API probe *(2026-10-13 21:45, $0.016 in total)*

Sources: `raw/E15_api_probe.txt` (switch survey) and `raw/E15_api_probe_long.txt` (the real long prompt,
thinking off). Model ids: `deepseek-v4-flash-0731`, `gemma-4-31b-it`, `qwen-3.8-27b-fp8`.

| | DeepSeek V4 Flash | Gemma 4 31B | Qwen3.8-27B (API) |
|---|---|---|---|
| log-probabilities (top 20) | yes | yes | yes |
| thinking by default | on | on | on |
| switched off by `chat_template_kwargs: {enable_thinking: false}` or `reasoning_effort: "none"` | yes | yes | yes |
| `thinking: {type: disabled}` | no effect | no effect | no effect |
| first token on the long prompt, thinking off | a letter | a letter | a letter |
| prompt tokens of the long screen item (k = 10) | 7,387 | 7,611 | 7,362 |
| repeated prompt billed as cached | **no** | yes (7,584 of 7,611) | yes (7,056 of 7,362) |
| cost per item, uncached / cached | $0.00096 / — | $0.00076 / $0.00038 | $0.00221 / $0.00066 |
| deterministic at temperature 0 | top-5 order yes; letter log-probs vary by up to 0.54 | yes (≤ 0.25) | yes (0.000) |
| n = 3 samples | supported | supported | supported |

What this settles (the probe-dependent items of the registration):
- **q is read from log-probabilities**, as registered; the sampling fallback is not needed.
- **Every API call sends `extra_body = {"chat_template_kwargs": {"enable_thinking": false}}`**, the same
  switch the local vLLM runs use.
- **DeepSeek is not bit-deterministic** (an MoE served with batching): its letter log-probabilities move by
  up to 0.54 between identical calls. Each item is scored once, as registered; this noise is part of
  what the fusion sees, and is reported.
- **Costs:** the dry run's character-based token estimate is ~30% high, so projections are conservative.
  DeepSeek on all ~2,600 items ≈ $2.5 (no caching); the screen arms ≈ $0.3 each.
- **The three models already differ on the one probe item:** DeepSeek A (−0.41) vs D (−1.41), Gemma A
  (≈ 0, very sharp), Qwen C (−0.94) vs A (−1.06).

**One change to the registration, made before any screen run.** A 3-item smoke test on Gemma
(cost $0.001) gave q ≈ 1.0 on one letter for every item, with the other letters at log-probabilities of
−15 or below. On the registered λ grid {0, 0.1, …, 1}, M1 can then never overrule Gemma short of
λ = 1, so the grid gains finer steps near 1: {…, 0.9, 0.95, 0.98, 0.99, 0.995, 1}. Nothing else changes.

**Infrastructure deviation: vLLM 0.31.0 instead of 0.30.0.** Installing 0.30.0 from PyPI failed or stalled
for over two hours: connections from the server kept timing out, and once the shared root disk's `/tmp`
filled up. vLLM 0.31.0 (with torch 2.13.0+cu130) is already installed and working for another project
under this account (`/scratch/shlok/venvs/vllm`), and its wheels are in that project's uv cache. So
`.venv-vllm` is built from that cache with 0.31.0. That environment is left untouched. Both versions
support Qwen3.8 (`Qwen3_5ForConditionalGeneration`), and scoring is unchanged: greedy, one token,
top-20 log-probs.


### E15 step 1: the screen *(2026-10-13 23:33–23:46; local arms free, API arms $0.17)*

Sources: `raw/E15_screen_local_k.txt` (example count) and `raw/E15_screen_models.txt` (models, faithfulness).
300 cross-fitted train rows from the 30% least confident (slice excluded), one gold label each. On these
rows the gold label is in the top 4 for 0.810, M1's top-1 accuracy is 0.293, and random choice within C
gives 0.203. "Fused" is λ and τ fitted by 5-fold CV inside the 300 rows.

| arm | k | LLM alone within C | fused | NLL of gold | LLM right, M1 wrong / reverse | cost or GPU time per item | prompt tokens |
|---|---|---|---|---|---|---|---|
| Qwen3.8-27B-FP8, local | 0 | 0.320 | **0.353** | 1.495 | 0.207 / 0.180 | 0.16 GPU-s | 1,145 |
| Qwen3.8-27B-FP8, local | 4 | 0.320 | 0.317 | 1.546 | 0.200 / 0.173 | 0.47 GPU-s | 3,353 |
| Qwen3.8-27B-FP8, local | 10 | 0.307 | 0.330 | 1.549 | 0.197 / 0.183 | 0.92 GPU-s | 6,303 |
| Qwen3.8-27B-FP8, API | 0 | 0.320 | 0.373 | 1.484 | 0.200 / 0.173 | $0.00034 | 1,145 |
| Gemma 4 31B, API | 0 | 0.320 | 0.363 | 8.574 | 0.203 / 0.177 | $0.00009 | 1,179 |
| DeepSeek V4 Flash, API | 0 | 0.320 | 0.317 | 2.574 | 0.227 / 0.200 | $0.00015 | 1,131 |

- The four k = 0 arms all get 96 of 300 rows right on their own, although they agree with each other on
  only 47–59% of rows (checked: a coincidence, not a bug).
- Paired fused differences (bootstrap over rows): k0 − k4 +0.037 [+0.000, +0.073], k0 − k10
  +0.023 [−0.013, +0.063]; DeepSeek − local Qwen −0.037 [−0.083, +0.007], Gemma − local Qwen
  +0.010 [−0.040, +0.053]. The API and local copies of the same Qwen differ by +0.020 [−0.010, +0.047].
  That is the screen's noise level between two nearly identical arms.
- Gemma is extremely overconfident (for 44% of rows some candidate letter is missing from its top 20).
  DeepSeek puts most of its mass on Deflection and Explicit (Deflection 104 / 300 predictions).

**Against the predictions.**
| | prediction | result | verdict |
|---|---|---|---|
| H1 LLM alone within C | Qwen 0.35–0.45, DeepSeek 0.40–0.50, Gemma 0.33–0.43; all beat M1 | 0.320 each; M1 0.293 | **failed** on the ranges; each beats M1, but only by 0.027 |
| H2 examples | k10 − k0 +0.03 to +0.08; k4 within 0.02 of k10 | k10 − k0 −0.013 (alone), −0.023 (fused) | **failed**: retrieved examples do not help this model |
| H3 fusion | fused − M1 +0.05 to +0.12; fused > LLM alone | +0.060 (local Qwen, k0); 0.353 > 0.320 | held |
| H4 faithfulness | argmax agreement ≥ 0.95, mean JS < 0.02 | 0.933, 0.0035 (max 0.027) | **half**: the distributions are close, and the disagreements are on near-ties; above the 0.90 caveat threshold |

**Decisions, by the registered rules.**
- **Example count: k = 0.** The best fused accuracy is k = 0 (0.353); k = 10 (0.330) is not within 0.01,
  so the tie-break towards k = 10 does not apply. Prompts are then ~1.1k tokens, not ~7k.
- **Model: the local Qwen3.8-27B-FP8.** DeepSeek does not beat the better ~30B arm by ≥ 0.02 (it is
  0.037 below), so it is not adopted. Gemma (0.363) and local Qwen (0.353) are within 0.01 of each
  other, so the free local Qwen is preferred.
- **Go/no-go: go.** Fused beats M1 by +0.060 (threshold 0.03).
- **Step 2** therefore runs local Qwen k = 0 on slice, dev, test and train50 (free, ~10 min).

**One addition, declared here before any dev scoring.** DeepSeek and Gemma at k = 0 are also scored
on **dev only** (≈ $0.08 together), as secondary arms. They are reported next to the adopted arm, so that
the paper shows whether a 300B model changes the system result. They are **not used for any choice**:
the adopted system is fixed above, and the screen is noisy (±0.02 between two copies of the same model).


### E15 steps 2–3: full scoring, slice confirmation, dev system and the M2 curve *(2026-10-13 23:46–00:05)*

> **Superseded in part (2026-10-14).** The dev-system numbers in this section (the dev table, H5–H7,
> "What the results say" and "Open for the user") came from a **bug in `cascade/fuse.py`**: one τ was shared between the fused
> items and the items that should keep M1's answer. See "E15 correction" below. The step-2 scoring, the
> slice confirmation and the screen are unaffected.

Sources:
- `raw/E15_dev_qwen38_27b_k0.txt`, the adopted arm;
- `raw/E15_dev_gemma4_31b_k0.txt` and `raw/E15_dev_deepseek_v4flash_k0.txt`, the secondary arms declared above;
- `raw/E15_slice_qwen_k0.txt`, the confirmation set.

**Step 2.** Local Qwen3.8-27B-FP8, k = 0, scored all 2,914 items (slice 345, dev 308, test 237,
train50 1,724; the 300 screen rows were copied, not re-scored). Wall time 7 min, 0.155 GPU-s per item,
free. Responses: `runs/M3/qwen38_27b_k0/responses.jsonl`. The train50 rows are M4's inputs.

**Slice confirmation** (the 104 slice items in the 30% least confident, 10-seed candidates, one gold label):

| | M1 (10-seed) | LLM alone within C | fused (5-fold CV within the 104) |
|---|---|---|---|
| accuracy | **0.413** | 0.337 | 0.346 |

The LLM is right where M1 is wrong on 0.096 of these items, and the reverse on 0.173. **The report's exit
criterion for M3 ("beats M1 on deferred slice items") is not met.** Caveat, registered in advance: 67%
of slice rows have their answer text in M1's training part, which flatters M1 on the slice (dev shares
1% of answers with train, test none).

**Dev system** (λ, τ by 5-fold nested CV on dev; S2 on split 0, plus the mean of 10 splits; the deferral
threshold comes from the slice, so the realised calls per item differ from δ):

| δ | calls/item | S2 | S2, 10 splits | Δ S2 vs M1, 95% | S1 | Δ S1 vs M1, 95% | 2-annotator S2 |
|---|---|---|---|---|---|---|---|
| 0 (M1) | 0 | 0.598 | 0.601 | — | 0.743 | — | 0.549 |
| 0.1 | 0.110 | 0.604 | 0.599 | +0.005 [−0.021, +0.026] | 0.752 | +0.009 [−0.027, +0.042] | 0.546 |
| 0.2 | 0.231 | 0.592 | 0.585 | −0.010 [−0.093, +0.051] | 0.756 | +0.012 [−0.042, +0.062] | 0.537 |
| **0.3** | **0.357** | **0.616** | **0.603** | +0.014 [−0.069, +0.077] | **0.781** | +0.038 [−0.006, +0.083] | 0.558 |
| 0.5 | 0.649 | 0.603 | 0.607 | +0.003 [−0.046, +0.051] | 0.793 | **+0.050 [+0.008, +0.089]** | 0.565 |
| 1 | 1.000 | 0.636 | 0.609 | +0.033 [−0.030, +0.104] | 0.798 | **+0.055 [+0.011, +0.100]** | 0.584 |

- **The LLM alone, restricted to M1's top 4, matches M1 on dev:** S2 0.603 (in-set rate 0.643), with no
  examples and one output token. Gemma alone: 0.601; DeepSeek alone: 0.476.
- **The fitted (λ, τ) jump around across δ** (λ 0.6, 0.1, 0.4, 0.5, 0.2), and the split-0 scores move by
  ±0.03 between neighbouring δ while the 10-split means stay at 0.585–0.609. The fusion objective is
  flat, and two parameters fitted on ~246 items are noisy. The 10-split means are the numbers to trust.
- **Secondary arms on dev** (declared before dev scoring; not used for any choice):
  - Gemma 4 31B: 10-split S2 0.598–0.614 for every δ, so no S2 gain; S1 +0.053 [+0.023, +0.084] at δ = 1.
  - DeepSeek V4 Flash: no gain anywhere. The fit mostly sets λ ≈ 1 (ignores it), and at δ = 0.3 S2 is
    −0.022 [−0.048, −0.001].
- **Cost of the adopted system at δ = 0.3:** 0.357 LLM calls per item, ~1.2k input tokens and 1 output
  token per call, 0.055 GPU-seconds per item on the lab card. TeleAI: 1.94 calls, ~7,300 input and ~380
  output tokens per item.

**Against the predictions.**
| | prediction | result | verdict |
|---|---|---|---|
| H5 dev S2 at δ = 0.3 | 0.61–0.64, CI excludes 0 if gain ≥ 0.025 | 0.616 on split 0, but 0.603 over 10 splits; +0.014 [−0.069, +0.077] | **not established**: no S2 gain distinguishable from M1 |
| H6 M2 curve | most gain by δ = 0.3; δ = 1 ≤ δ = 0.5 + 0.01 | 10 splits: 0.603 / 0.607 / 0.609 at δ = 0.3 / 0.5 / 1, a flat curve; split 0: δ = 1 is +0.033 over δ = 0.5 | held on the 10-split means (there is no gain to arrive) |
| H7 S1 | +0.00 to +0.03 | +0.038 at δ = 0.3; +0.050 / +0.055 at δ = 0.5 / 1 (CIs exclude 0) | **exceeded**: M3 helps Subtask 1 |
| report exit criterion | M3 beats M1 on deferred slice items | 0.346 vs 0.413 | **failed** (slice partly seen by M1, see above) |

**What the results say.**
- **A prompted 27B model, constrained to M1's top four labels, is as good as our fine-tuned 10-seed system
  on Subtask 2** (0.603 vs 0.598). It uses no examples and one output token, and its errors differ from
  M1's. Yet fusing the two does not raise S2 reliably.
- **The screen overstated the gain.** On cross-fitted rows, M1 is a single model (top-1 0.293), and
  fusion added +0.060 there. The deployed M1 is a 10-seed ensemble, which is far stronger on the same kind
  of items (slice 0.413) and leaves the LLM little to add. The screen's baseline should have been the
  deployed one; only the slice has 10-seed probabilities for train rows, and it is small and partly seen.
- **M3 does make M1 more accurate; macro-F1 hides it.** At δ = 0.3, M3 changes 62 dev predictions.
  - 26 move into the reference set and 10 move out (exact McNemar p = 0.011), so the in-set rate rises
    from **0.623 to 0.675**.
  - Per class, F1 rises for Explicit (+0.095), General (+0.120) and Declining (+0.098), and falls for
    Deflection (−0.035) and **Clarification (−0.143: one of its 4 dev items lost)**. That single item
    costs ≈ 0.016 of macro-F1, as much as most of the gain.
  - S2 differences at this size are dominated by the rarest classes. M3 also predicts Partial/half-answer
    far less often (19 → 4 predictions; F1 0.231 → 0.222).
  - Source: `raw/E15_dev_qwen38_27b_k0.txt`, end.
- **Subtask 1 gains clearly** (+0.05 at δ ≥ 0.5). The largest changes are Implicit → Explicit (11 items at
  δ = 0.3, 18 at δ = 1), which crosses from Ambivalent to Clear Reply, and Deflection → Implicit (11).
  The fusion is fitted on S2, so this gain is a by-product.
- **Examples hurt this model (H2).** Retrieved neighbours carry almost no label signal (10-NN in-set rate
  0.33 on dev, below always-Explicit's 0.37), and longer prompts made Qwen worse on the screen.
- **A bigger model is not better here.** DeepSeek V4 Flash (304B) is the weakest arm, both alone and fused.

**Cost.** API spend for all of E15: **$0.262 (≈ ₹23)**, from the ledger `runs/M3/api_ledger.jsonl`:
probe $0.016, smoke test $0.001, three screen arms $0.174, two secondary dev arms $0.071. The local runs
were free (lab server GPU 0, ~20 min in total). Cap: $8.

**Deviations.** vLLM 0.31.0 instead of 0.30.0; the λ grid gained finer steps near 1; the secondary
dev-only arms. Each was recorded above before the runs it affects. Two analysis bugs were fixed before
any reported number: λ = 1 produced NaN scores, and the 2-annotator column now refits per reference set,
as in E14.

**Open for the user (not decided overnight).**
1. **Adopt M3 for the final system?** It costs 0.36 calls per item and gains S1 but not S2. Options:
   - adopt it for S1 only;
   - adopt it at δ = 1, which gives the best point estimates (S2 0.636 split 0 / 0.609 over 10 splits,
     S1 0.798) at 1 call per item;
   - keep M1 alone for S2.
2. **M4 (distillation)** can go ahead: the train50 targets exist. The finding that the LLM alone matches
   M1 suggests its q is a useful soft target.
3. **Any further M3 variant needs a new registration:** e.g. a temperature on q fitted on cross-fitted
   rows, or an S1-targeted fusion. None was run.


### E15 correction: the fusion bug, and the corrected dev system *(2026-10-14)*

**The bug.** In `fuse.decide`, the items that are *not* deferred were decided by log p̄ − τ log π with the
**same τ** as the fused items, and that one τ was fitted jointly with λ. The fit could therefore choose a
τ that suits the LLM-decided items (it chose τ = 0.0 at δ = 0.3) and thereby switch off M1's logit
adjustment (τ ≈ 1.15) on the 64% of items that were meant to keep the M1 system's answer.
- This contradicts the design: report §7 says "items not deferred keep M1's answer", and the registration
  says "items outside the deferred set … keep M1's decision".
- It was found on 2026-10-14 while explaining the flat S2 result. The fit's τ = 0.0 at δ = 0.3 against
  M1's own 1.15 was the clue.

**The fix.** Inside each nested-CV fold, M1's τ is fitted first, exactly as for the M1 system (the E4
rule, which reproduces M1's 0.598). Then (λ, τ) of the deferred items are fitted with M1's decisions held
fixed on the other items.
- Nothing else changed: same responses, same folds, same grids and the same deferral thresholds.
- **Unaffected:** δ = 1 (no item is kept; its numbers are identical to before, which is the check),
  δ = 0, the screen and the slice (every row deferred), and therefore every decision taken from them.
- Source: `raw/E15_dev_qwen38_27b_k0.txt` (regenerated), and the same for the two secondary arms.

**Corrected dev system, adopted arm (local Qwen3.8-27B-FP8, k = 0):**

| δ | calls/item | S2 | S2, 10 splits | Δ S2 vs M1, 95% | S1 | Δ S1 vs M1, 95% | 2-annotator S2 | in-set |
|---|---|---|---|---|---|---|---|---|
| 0 (M1) | 0 | 0.598 | 0.601 | — | 0.743 | — | 0.549 | 0.623 |
| 0.1 | 0.110 | 0.597 | 0.602 | −0.001 [−0.037, +0.031] | 0.762 | +0.019 [−0.012, +0.051] | 0.549 | 0.630 |
| 0.2 | 0.231 | 0.635 | 0.640 | +0.035 [−0.013, +0.080] | 0.775 | +0.032 [−0.012, +0.075] | 0.571 | 0.662 |
| **0.3** | **0.357** | **0.669** | **0.660** | **+0.067 [+0.009, +0.124]** | **0.798** | **+0.055 [+0.004, +0.109]** | **0.589** | **0.701** |
| 0.5 | 0.649 | 0.628 | 0.628 | +0.027 [−0.023, +0.078] | 0.800 | +0.057 [+0.011, +0.105] | 0.600 | 0.679 |
| 1 | 1.000 | 0.636 | 0.609 | +0.033 [−0.030, +0.104] | 0.798 | +0.055 [+0.011, +0.100] | 0.584 | 0.669 |

- **At the registered operating point δ = 0.3**, M3 changes 69 predictions: 33 move into the reference set
  and 9 out of it (exact McNemar p = 0.0003).
  - F1 rises for 7 of 9 classes, most for Partial/half-answer (+0.169), Declining (+0.131), Dodging
    (+0.125), Explicit (+0.086) and General (+0.085).
  - Claims ignorance falls (−0.031); Clarification is unchanged.
- **Secondary arms, corrected** (not used for any choice):
  - Gemma 4 31B at δ = 0.3: S2 0.625 (10 splits 0.620), +0.026 [−0.019, +0.071].
  - DeepSeek V4 Flash: no gain at any δ (best 10-split 0.595 at δ = 0.1).
- **Against the winning system:** TeleAI's multi-call pipeline scores dev S2 0.617 / S1 0.812 with 1.94
  calls and ~7,300 input + ~380 output tokens per item. M1 + M3 scores **0.669 / 0.798 with 0.36 calls,
  ~1.2k input tokens and 1 output token per call**: higher on S2, a little lower on S1, at about a fifth
  of the calls and a tiny fraction of the tokens. This is a dev comparison on 308 items: the gain over
  M1 has a wide interval, and the 2-annotator mean (the test regime) is 0.589.

**Corrected verdicts.**
| | prediction | corrected result | verdict |
|---|---|---|---|
| H5 dev S2 at δ = 0.3 | 0.61–0.64, CI excludes 0 if gain ≥ 0.025 | **0.669** (10 splits 0.660); +0.067 [+0.009, +0.124] | **held, and above the range** |
| H6 M2 curve | most gain by δ = 0.3; δ = 1 ≤ δ = 0.5 + 0.01 | peak at δ = 0.3; 10 splits: δ = 1 0.609 < δ = 0.5 0.628 | held |
| H7 S1 | +0.00 to +0.03 | +0.055 at δ = 0.3 (CI excludes 0) | **exceeded** |
| report exit criterion | M3 beats M1 on deferred slice items | 0.346 vs 0.413 (unaffected by the bug) | failed; but the slice is 67% seen by M1, and on dev, where M1 saw 1% of answers, M3 wins clearly |

**What the corrected results say.**
- **The cascade works.** Deferring the 30% least confident items to a prompted 27B model, constrained to
  M1's top four labels and fused with M1, raises dev S2 from 0.598 to 0.669 and S1 from 0.743 to 0.798, at
  0.36 calls per item. The report's target (≥ 0.60 dev S2 at ≤ 0.3 calls per item) is met at δ = 0.2
  (0.635 at 0.23 calls). δ = 0.3 uses 0.36 calls, a little over budget, because its threshold is set on
  the slice.
- **Deferral matters, not just the LLM.** Fusing every item (δ = 1) is worse than fusing the uncertain 30%
  (10-split 0.609 vs 0.660). On confident items M1 is better than the LLM, and the router keeps them.
- **The slice failure is a property of the slice.** On the slice M1 has seen most answers in training, so
  it looks better there than it is on new interviews. Dev, with almost no answer overlap, is the honest
  comparison.
- **The screen's choices still stand** (no examples; the local 27B over DeepSeek): those analyses had all
  rows deferred, so the bug could not touch them.
- **The superseded section's claims about rare classes and Clarification were artefacts of the bug.**
  With M1's logit adjustment switched off on the kept items, rare-class predictions were lost; with it
  restored, the rare classes gain the most.

**Updated open decisions for the user.** *(Resolved 2026-10-15: M1 + M3 at δ = 0.3 is the accuracy
operating point; M4 was done as E17; no submission file is needed yet.)*
1. **Adopt M1 + M3 at δ = 0.3** as the system (the registered operating point; not chosen on dev). The
   alternative is δ = 0.2 for 0.23 calls per item. The report's target was ≤ 0.3 calls.
2. **M4 (distillation)** next; its inputs exist (`runs/M3/qwen38_27b_k0`, train50).
3. **Test predictions** of the adopted system can be written from the saved test responses (no new calls).


---

## E16 — Finishing M2: the prompt-part ablation, measured cost, and the accuracy-vs-cost frontier *(registered 2026-10-14)*

*Written before any E16 run.* Plan: `05_phase2_plan_and_budget.md` §7.2.

**Why.**
- **The report promises an ablation of each prompt part,** and E15 removed only the examples.
- **The cascade's cost has never been measured per item.** By E13's evaluation timing, M1's 10-seed
  ensemble is ≈ 2–3 GPU-s per item, while M3 at δ = 0.3 is ≈ 0.06. If that holds, the cost lever is the
  number of seeds K, not the LLM.
- **E14 showed K cannot be shrunk on its own** (K = 1 / 3 / 10: S2 0.509 / 0.564 / 0.598). Whether M3
  lets K shrink is unknown, and it is the baseline any distillation (E17) must beat.

**Runs** (all on the lab server, GPU 0; free; no API calls):
1. **Prompt ablation.** Local Qwen3.8-27B-FP8, k = 0, with prompt variants `full` (E15's), `no_guide`
   (no confusion guide), `no_defs` (label names only, no definitions) and `bare` (neither). Each is run on
   the 300 screen rows, then on dev.
2. **Cost harness.** GPU-seconds per item on dev, on GPU 0 alone:
   - one M1 seed (HF transformers, bf16, batch 16, length-sorted: the code path of `llm_classifier.predict`);
   - one M3 call (vLLM, as in E15).
   - Also input and output tokens per item.
3. **Frontier.** K ∈ {1, 2, 3, 5, 10} seeds of `Q8_fullq_lora_12ep`; 10 random seed subsets per K < 10
   (seed 16), all 10 seeds for K = 10.
   - Each subset gets its own calibration, candidate sets (top 4) and routing u = 1 − top q. These are
     fitted on the slice, as in `cascade.candidates`.
   - M3 scores every (dev item, candidate set) pair not already scored, with the same tag, so identical
     pairs are reused.
   - The system for each (subset, δ): M1's own τ, then fused (λ, τ) by nested CV, as in E15 after the
     correction. Split 0 per subset, averaged over subsets.
   - Cost of (K, δ) = K × (M1 seed cost) + realised calls × (M3 call cost).

**Hypotheses.**
- **H1, prompt parts** (fused screen accuracy; full = 0.353):
  - `no_guide` is 0.00–0.03 lower;
  - `no_defs` is 0.01–0.05 lower;
  - `bare` is the lowest.
  - Dev is reported for all four as the ablation table.
- **H2, cost:**
  - an M1 seed costs 0.15–0.35 GPU-s per item and an M3 call 0.12–0.20;
  - so M1's 10 seeds are ≥ 90% of the system's cost at δ = 0.3.
- **H3, frontier:**
  - with M3 at δ = 0.3, S2 rises with K less steeply than M1 alone;
  - K = 3 + M3 is within 0.03 of K = 10 + M3 (0.669);
  - K = 1 + M3 scores 0.59–0.63.
  - At least one (K ≤ 3, δ) point beats M1 alone with K = 10 (0.598) at under a third of its cost.

**Decision rules.**
- **The adopted M3 prompt stays `full`** unless a reduced variant beats it by ≥ 0.02 in fused screen
  accuracy. The preamble is a negligible share of the cost (H2), so no shorter prompt is adopted for cost.
- **The frontier's Pareto points are the candidates for the cost operating point** of the final system.
  The accuracy operating point stays K = 10 + M3 at δ = 0.3 (E15).

**Cost.** Free: about 30–40 min of GPU 0, plus CPU analysis.

### E16 results *(2026-10-14, 12:10–12:40; free, lab server GPU 0)*

Sources:
- `raw/E16_prompt_ablation_screen.txt` and `raw/E16_dev_qwen38_27b_k0_<variant>.txt` (the prompt ablation;
  `full` is `raw/E15_dev_qwen38_27b_k0.txt`);
- `runs/M3/cost.json` and `logs/E16_cost.log` (cost);
- `raw/E16_frontier.txt` and `runs/M3/frontier_Q8_fullq_lora_12ep.json` (frontier).

M3 also scored all train rows (E17's targets) and the frontier's 1,773 (dev item, top-4) pairs: 3,150 new
items, 2,071 reused.

**1. Prompt-part ablation** (local Qwen3.8-27B, k = 0):

| variant | screen fused | screen LLM alone | dev S2 at δ = 0.3 (10 splits) | dev S1 at δ = 0.3 | dev LLM alone S2 |
|---|---|---|---|---|---|
| full (E15) | 0.353 | 0.320 | 0.669 (0.660) | 0.798 | 0.603 |
| no confusion guide | 0.370 | 0.353 | 0.645 (0.637) | 0.793 | 0.620 |
| no definitions | 0.347 | 0.320 | 0.619 (0.628) | 0.787 | 0.563 |
| neither (bare) | 0.367 | 0.323 | 0.617 (0.617) | 0.779 | 0.520 |

- **On the screen, the variants are indistinguishable** (all paired CIs include 0). No variant beats `full`
  by 0.02, so **`full` stays**, by the rule.
- **On dev the expected order appears:** full > no guide > no definitions > bare (10 splits 0.660 / 0.637 /
  0.628 / 0.617). Each step is within the noise, but the definitions matter most for the LLM alone (0.603
  vs 0.563 without them).
- **H1:** the order of the variants held on dev; the predicted screen differences did not appear.

**2. Cost** (GPU 0 alone, RTX PRO 6000 Max-Q; `runs/M3/cost.json`):
- **One M1 seed** (HF, bf16, batch 16): **0.052 GPU-s per item**, ~490 tokens. Its dev predictions
  reproduce the saved ones exactly.
- **One M3 call** (vLLM, batched): **0.103 GPU-s**, ~1.1k prompt tokens and 1 output token.
- **FP8** was not measured: the transformers FP8 kernels package is missing, and it was not installed over
  this server's unreliable network.
- **Correction of the plan's estimate:** §7.1 of docs/05 put an M1 seed at 0.2–0.3 GPU-s, read from E13's
  epoch timings, which include evaluation overhead. The measured cost is 4–6× lower. The conclusion holds:
  the 10-seed M1 is 0.52 GPU-s per item and M3 at δ = 0.3 adds 0.04, so **M1 is ~93% of the cost**.
- **H2:** an M1 seed was predicted at 0.15–0.35, measured 0.052 (failed: cheaper); an M3 call was
  predicted at 0.12–0.20, measured 0.103 (failed, narrowly); M1's share was predicted ≥ 90%, measured 93%
  (held).

**3. The accuracy-vs-cost frontier** (dev; S2 is the mean over 10 seed subsets per K, nested-CV split 0;
K = 10 is one ensemble):

| K | M1 alone | δ = 0.3 | δ = 1 | GPU-s at δ = 0 / 0.3 / 1 |
|---|---|---|---|---|
| 1 | 0.533 ± 0.047 | 0.558 ± 0.037 | 0.593 ± 0.025 | 0.052 / 0.088 / 0.155 |
| 2 | 0.549 ± 0.036 | 0.579 ± 0.027 | 0.609 ± 0.018 | 0.104 / 0.141 / 0.207 |
| 3 | 0.562 ± 0.017 | 0.583 ± 0.014 | 0.607 ± 0.015 | 0.156 / 0.192 / 0.259 |
| 5 | 0.563 ± 0.024 | 0.594 ± 0.013 | 0.602 ± 0.021 | 0.261 / 0.298 / 0.364 |
| 10 | 0.598 | **0.669** | 0.636 | 0.522 / 0.558 / 0.624 |

- **M3 helps at every K.** With K ≤ 5, it adds +0.021 to +0.031 at δ = 0.3 and +0.039 to +0.060 at δ = 1.
  S1 rises with it.
- **The K = 10 system is an outlier.** M3 adds +0.071 there, more than twice the gain at any smaller K. It
  is a single ensemble scored on 308 items, and E14 already saw K = 10 sit above the K-curve (K = 9: 0.574
  ± 0.020). So **E15's headline (0.669, +0.067) is likely optimistic.** M3's typical gain on this dev set is
  +0.02 to +0.04 at δ = 0.3, consistent with E15's own interval [+0.009, +0.124].
- **The cost result.** With K = 1, sending every item to M3 gives **S2 0.593 / S1 0.774 at 0.155 GPU-s per
  item**. That equals the 10-seed M1 alone (0.598 / 0.743, 0.522 GPU-s) on S2, beats it on S1, and costs
  **3.4× less**. In the subset means, smaller ensembles with M3 replace seeds by LLM calls almost one for one.
- **Pareto points:** K = 1 at every δ, K = 2 at δ = 0.3 / 0.5 / 1, and K = 10 at δ = 0.2 / 0.3. No K = 3 or
  K = 5 point is on the front.
- **H3:**
  - "S2 rises with K less steeply with M3": failed. K = 1 → 10 adds +0.065 alone and +0.111 with M3 at
    δ = 0.3, driven by the K = 10 outlier; K = 1 → 5 adds +0.030 vs +0.036.
  - "K = 3 + M3 within 0.03 of K = 10 + M3": failed (0.583 vs 0.669).
  - "K = 1 + M3 at δ = 0.3 in 0.59–0.63": failed (0.558; 0.593 at δ = 1).
  - "a K ≤ 3 point beats 0.598 at under a third of the cost (0.174)": not quite. K = 1 at δ = 1 reaches
    0.593 at 0.155, and K = 2 at δ = 1 reaches 0.609 at 0.207.

**What E16 changes.**
- **The cascade's value is real but smaller than E15's single number suggests.** Report the frontier, not
  only the K = 10 point. The final report should give M3's gain as the subset-averaged +0.02 to +0.04
  alongside the K = 10 result.
- **E17's target is clear:** the student (K = 1) + M3 must beat K = 1 + M3 (0.558 at δ = 0.3, 0.593 at
  δ = 1) and move towards the K = 10 points. On seeds 0–2, the controls are those seeds' K = 1 rows
  (`runs/M3/frontier_Q8_fullq_lora_12ep.json`, per_subset).

## E17 — Module M4: distilling M3 into one M1 model, for cost *(registered 2026-10-14)*

*Written before any E17 run.* Plan: `05_phase2_plan_and_budget.md` §7.3.

**Why.** The aim is a cheaper system, not a more accurate one (the user, 2026-10-14). If a single
student model with M3 matches the 10-seed M1 with M3, inference cost drops about tenfold (E16's H2). The
report's M4 retrains M1 on soft targets from M3 (Hinton et al., 2015).

**Decisions taken with the user (2026-10-14).**
- **Targets:** t = (1 − α)·onehot(gold) + α·q, with **α = 0.5 fixed a priori**.
  - q is M3's distribution (local Qwen3.8-27B-FP8, k = 0, full prompt) over the row's cross-fitted top 4.
  - Every train row is scored: M3 is first run on the train rows not in train50. This is local and free.
  - **Deviation from the report:** α is not fitted on the slice. That would need extra training runs, and
    slice F1 from single runs is noisy.
- **Teacher:** M3's q alone, the report's design (the fused M1 + M3 distribution is not used).
- **Precision for the cost table:** bf16; an FP8 measurement of the student if transformers supports it
  on this card. **No 4-bit**, a deviation from the report.
- **Compute:** the lab server's GPU 0 first. JarvisLabs runs in parallel when that saves time. The GPU-1
  MIG slices are used only when free, under the E14 rules.

**Runs.**
- **Student:** `Q8_distill_a05`, the E14 recipe (Qwen3-8B-Base + LoRA, 12 epochs, epoch selected on the
  slice) with one change: `--soft-targets`, i.e. the cross-entropy against t instead of the gold label.
- **Screen:** seeds 0–2. Control: `Q8_fullq_lora_12ep` seeds 0–2 (E13b, dev S2 0.531 / 0.560 / 0.539).
- **Evaluated as a system:** student (K = 1) + M3.
  - The candidates are the student's own top 4, calibrated on the slice as in M1, and its own u routes.
  - M3 scores any new (item, candidate set) pairs, locally.

**Hypotheses** (on seeds 0–2, then 10 if the screen passes):
- **H1, cost (the primary question).** Student + M3 at δ = 0.3 recovers at least half of the gap between
  M1(K = 1) + M3 and M1(K = 10) + M3, from E16, on the same seeds: student + M3 ≥ (K1 + K10) / 2.
- **H2, routing.**
  - The student is less overconfident: mean top probability on dev 0.60–0.80 (M1 single model 0.839).
  - Its u separates errors at least as well (AUROC of 1 − top probability ≥ 0.70).
- **H3, accuracy (secondary).** Single student S2 vs M1 single model, paired: −0.01 to +0.03; S1 +0.00
  to +0.04.

**Decision rules.**
- **Screen:** run 7 more seeds if, on seeds 0–2, student + M3 beats M1(K = 1) + M3 by ≥ 0.015 (mean
  over the 3 seeds) or meets H1. Otherwise M4 is reported as a negative result, and the cost operating
  point comes from E16's frontier.
- **Adoption, after 10 seeds:** student + M3 becomes the **cost operating point** if no undistilled
  (K, δ) point of E16 is both cheaper and better. K = 10 + M3 stays the accuracy operating point.

**Cost.**
- **Screen:** free (GPU 0, ~1.5–2 h per run).
- **Confirmation:** split between GPU 0 and Jarvis. Jarvis ≈ ₹220 per seed, at most ≈ ₹1,550.


### E17 screen *(seeds 0–2, 2026-10-14 12:26–16:19; free, lab server GPU 0)*

Sources: `raw/E17_m4_analysis_screen_seeds012.txt` (sections A–C) and `raw/E17_frontier_students_screen_seeds012.txt`. Runs:
`Q8_distill_a05/seed0–2` (on HF; W&B group `Q8_distill_a05`); evaluation published as W&B `E17-results`.
- Each run took 76 min, and epochs 5–8 were selected on the slice (seed 2: epoch 6).
- One infrastructure pause: seed 0 was stopped 2 min in, at 12:26, so GPU 0 was idle for E16's cost
  timing, and was then restarted from scratch.
- One bug fixed before any number was read. Items from a jobs file had ids without their candidate set,
  so the students' new (item, candidates) pairs were skipped as "done". Their ids now include the
  candidates. E16 was unaffected: its pairs were all submitted in one run.

| | student `Q8_distill_a05` | control `Q8_fullq_lora_12ep` (same seeds) |
|---|---|---|
| single model, dev S2 (per seed) | 0.557 / 0.596 / 0.572, mean **0.575** | 0.531 / 0.560 / 0.539, mean 0.543 |
| single model, dev S1 | 0.749 | 0.742 |
| mean top probability on dev | **0.581** | 0.873 |
| AUROC of 1 − top probability for errors | 0.683 | 0.671 |
| + M3, δ = 0 / 0.1 / 0.2 / 0.3 / 0.5 / 1 (S2) | 0.580 / 0.581 / 0.586 / 0.589 / **0.616** / **0.626** | 0.557 / 0.564 / 0.565 / 0.574 / 0.576 / 0.597 |
| + M3, S1 at δ = 0.3 / 1 | 0.765 / 0.783 | 0.740 / 0.765 |
| GPU-s per item at δ = 0 / 0.3 / 1 | 0.052 / 0.087 / 0.155 | same |

**Against the predictions (3 seeds).**
| | prediction | result | verdict |
|---|---|---|---|
| H1 student + M3 ≥ midpoint of control + M3 and K10 + M3 | δ = 0.3: ≥ 0.621 | 0.589 at δ = 0.3; met at δ = 0 (0.580 ≥ 0.578), δ = 0.5 (0.616 ≥ 0.602) and δ = 1 (0.626 ≥ 0.617) | failed at the registered δ, held at higher deferral |
| H2 routing | top prob 0.60–0.80; AUROC ≥ 0.70 | 0.581 (less confident than predicted); 0.683 (control 0.671) | direction held, both thresholds missed |
| H3 single model | S2 −0.01 to +0.03; S1 +0.00 to +0.04 | S2 **+0.032 (3/3 up)**; S1 +0.007 | S2 at the top of the range; S1 held |

**Screen decision (the registered rule):** at δ = 0.3, student + M3 beats control + M3 by +0.0151, against
a threshold of ≥ +0.015. **This passes by a hair, so 7 more seeds (3–9) run.**
- **Placement:** by the compute rule, on the lab server's GPU 0. It is free, and there is no time pressure
  (~9 h). The GPU-1 slices are in use by a lab-mate.
- **What already looks clear on 3 seeds (the cost story):**
  - the distilled student alone (0.052 GPU-s per item) is close to the 10-seed M1 (0.598 at 0.522);
  - student + M3 on every item (0.626 / 0.783 at 0.155 GPU-s) beats the 10-seed M1 alone at 3.4× less
    cost.
  - Whether it beats the undistilled K = 1 + M3 frontier reliably is what the 10 seeds decide.


### E17 results *(10 seeds; all runs finished 2026-10-15 01:13, evaluation 01:16; free, lab server GPU 0)*

Sources:
- `raw/E17_m4_analysis.txt` (sections A–D) and `raw/E17_frontier_students.txt`;
- runs `Q8_distill_a05/seed0–9` (HF, W&B group `Q8_distill_a05`);
- the evaluation on W&B as `E17-results` (run 5gj2t9lt, with the frontier tables);
- the student frontier in `runs/M3/frontier_Q8_distill_a05.json` (on HF under `M3/`).

All 10 runs took 76 min each on GPU 0, with no failures; selected epochs 6–9. For the students' own
candidates, M3 scored 185 new (dev item, top-4) pairs; the rest were reused.

**A. Single models (dev, paired seeds).** Student S2 0.510–0.601, mean **0.573**. Control
(`Q8_fullq_lora_12ep`) 0.436–0.570, mean 0.525.
- S2: **+0.048, 9/10 seeds up**.
- S1: 0.761 vs 0.729, **+0.031, 8/10 up**.
- The student is also steadier: its worst seed is 0.510, against the control's 0.436.

**B. Routing.**
- Mean top probability 0.578 vs 0.839: far less overconfident.
- The AUROC of 1 − top probability for errors is equal, 0.686 vs 0.686.

**C/D. Systems with M3, against E16's undistilled frontier** (S2 = mean over the 10 single-seed systems,
nested-CV split 0):

| δ | student + M3 | undistilled K = 1 + M3 | Δ | K = 10 + M3 | student S1 | GPU-s per item |
|---|---|---|---|---|---|---|
| 0 (no LLM) | 0.570 ± 0.022 | 0.533 | +0.038 | 0.598 | 0.755 | 0.052 |
| 0.1 | 0.571 | 0.544 | +0.027 | 0.597 | 0.762 | 0.063 |
| 0.2 | 0.580 | 0.552 | +0.028 | 0.635 | 0.766 | 0.074 |
| 0.3 | 0.591 ± 0.016 | 0.558 | +0.032 | 0.669 | 0.776 | 0.087 |
| 0.5 | **0.608** ± 0.027 | 0.575 | +0.033 | 0.628 | **0.790** | **0.111** |
| 1 | **0.612** ± 0.022 | 0.593 | +0.019 | 0.636 | **0.792** | **0.155** |

- **Adoption rule (registered): met.** No undistilled (K, δ) point is both cheaper and at least as good as
  any student + M3 point. The students dominate 25 of the 30 undistilled points.
- **The combined accuracy-vs-cost front** is the six student + M3 points (0.052 → 0.155 GPU-s), followed
  only by the 10-seed M1 + M3 at δ = 0.2 / 0.3 (0.545 / 0.558 GPU-s).

**Against the predictions.**
| | prediction | result (10 seeds) | verdict |
|---|---|---|---|
| H1 student + M3 ≥ midpoint of K1 + M3 and K10 + M3 | at δ = 0.3: ≥ 0.614 | 0.591; met at δ = 0 (0.570 ≥ 0.565) and δ = 0.5 (0.608 ≥ 0.602), missed at δ = 1 by 0.003 | failed at the registered δ |
| H2 routing | top prob 0.60–0.80; AUROC ≥ 0.70 | 0.578; 0.686 (control 0.686) | failed: calibration is far better, but error ranking does not improve |
| H3 single model | S2 −0.01 to +0.03; S1 +0.00 to +0.04 | S2 **+0.048 (9/10)**; S1 +0.031 (8/10) | S2 **exceeded**; S1 held |
| adoption rule | student + M3 on the undistilled frontier | not dominated anywhere; dominates 25/30 | **adopted as the cost operating point** |

**What the results say.**
- **Distillation does what M4 was for: it makes the cascade cheap.**
  - **Student + M3 at δ = 0.5:** S2 0.608 / S1 0.790 at 0.111 GPU-s per item. That beats the 10-seed M1
    alone (0.598 / 0.743 at 0.522 GPU-s) at **4.7× less cost**.
  - **At δ = 0.3 (0.087 GPU-s, 6× cheaper):** about level with it on S2 (0.591), and better on S1 (0.776).
  - **At δ = 1:** 0.612 / 0.792 at 0.155 GPU-s. That is close to TeleAI's 0.617 / 0.812, with one 8B pass
    and one 27B call per item.
- **It also gives a better single model,** against the prediction: +0.048 S2 with no LLM at inference.
  - Its gain over the undistilled single model (+0.038 at δ = 0, with the E4 rule) is as large as what M3
    adds on top of the student (+0.021 / +0.038 / +0.042 at δ = 0.3 / 0.5 / 1). So the soft targets carry
    a good part of the LLM's knowledge into the 8B model, and M3 still adds about as much again.
  - It also removes the bad-seed tail: the student's worst seed is 0.510, the control's 0.436.
- **Calibration improved, error ranking did not.** The student's confidence is honest (top probability
  0.58 vs 0.84), but 1 − top probability ranks its errors no better (AUROC 0.686 for both). The routing
  gain therefore comes from a better model behind the router, not from a better router.
- **The accuracy point is unchanged.** The 10-seed M1 + M3 at δ = 0.3 (0.669) is still the most accurate
  system. The final system has two operating points:
  - **accuracy:** 10-seed M1 + M3, δ = 0.3;
  - **cost:** the student + M3, δ = 0.5 (or δ = 0.3 for the cheapest point at about the 10-seed M1's level).

**Cost.** Free: 10 × 76 min on the lab server's GPU 0, plus a few minutes of M3 scoring. No Jarvis credit
was used; the balance is ₹4,001.81.


---

## E18 — Freeze: final numbers for both operating points, and ensembles of distilled students *(registered 2026-10-15)*

*Written before any E18 computation.* The report, the slides and the submission files are left for later
(the user, 2026-10-15).

**Part A. Final numbers (CPU; no new model runs).** All computed from saved probabilities and M3 responses,
with the code paths of E15–E17, by `preregistered_analyses/e18_final.py` into `raw/E18_final.txt`.
- **Accuracy point** (10-seed M1 + M3, δ = 0.3), with the cost-point numbers next to it:
  - S2 and S1 on nested-CV split 0 and the mean of 10 splits;
  - the bootstrap against M1;
  - S2 on the three 2-annotator reference sets, with the nested CV refitted on each (the test regime, as in
    E14);
  - per-class F1.
- **Cost point** (student + M3, δ = 0.5): the same numbers per student, reported as mean ± sd over the 10
  students, plus the paired difference to the undistilled K = 1 + M3 on the same seeds.
- **Cost table:** GPU-s, LLM calls and tokens per item for M1 (10 seeds), the student alone, both operating
  points and TeleAI's published figures.
- **The accuracy-vs-cost frontier figure** (E16 + E17 + Part B), in the report's grayscale style:
  `docs/figures/E18_frontier.{pdf,png}`.

**Part B. Ensembles of distilled students (the one new measurement).**
- **Why:** between the cost point (one student, 0.111 GPU-s) and the accuracy point (ten M1 seeds, 0.558
  GPU-s) there is no measured middle option. The 10 students already exist, so ensembles of K students
  need no training.
- **Runs:** `python -m cascade.frontier` with `--run Q8_distill_a05 --ks 1,2,3,5,10`, i.e. the E16
  procedure: 10 random subsets per K < 10 (seed 16), one for K = 10. M3 (local Qwen3.8-27B-FP8, k = 0, full
  prompt) scores the new (dev item, top-4) pairs on GPU 0 (free, minutes).
- **Hypotheses:**
  - **H1:** students gain less from ensembling than M1 seeds do. Distillation has already removed much of
    the seed-to-seed variance (their spread is 0.022 vs 0.047 for single M1 seeds). Predicted: student
    K = 10 alone 0.59–0.62 (one student 0.570; M1 K = 1 → 10: +0.065).
  - **H2:** student K = 3 + M3 at δ = 0.5 scores 0.61–0.64 at ≈ 0.22 GPU-s, and lies on the combined front.
  - **H3:** student K = 10 + M3 at δ = 0.3 does not beat M1 K = 10 + M3 (0.669) by more than the noise
    (±0.03).
- **Decision rules:**
  - A student-ensemble point that is on the combined front (no point both cheaper and better) is reported
    as a **middle operating point**.
  - The **accuracy point** changes to student K = 10 + M3 only if that system beats 0.669 at δ = 0.3 by
    ≥ 0.02 (the same cost).

**Cost.** Free: CPU, plus a few minutes of the lab server's GPU 0. Everything is logged to `logs/`, HF
(`M3/`) and W&B (`E18-results`).


### E18 results *(2026-10-15, 08:40–08:50; free: CPU plus ~1 min of the lab server's GPU 0)*

Sources:
- `raw/E18_final.txt` (Part A; `preregistered_analyses/e18_final.py`);
- `raw/E18_frontier_student_ensembles.txt` and `runs/M3/frontier_Q8_distill_a05_ensembles.json` (Part B);
- the figure `docs/figures/E18_frontier.{pdf,png}`;
- W&B `E18-results` (run 0cdgfq61, with the figure and tables); HF `M3/`.
- For Part B, M3 scored 18 new (dev item, top-4) pairs; 45 were reused.
- One code change: `cascade.frontier analyse --out`, so that E18 writes its own file and E17's
  `frontier_Q8_distill_a05.json` stays as E17 used it.

**Part A. Final dev numbers for both operating points.**

| | M1, 10 seeds | **accuracy point**: M1 × 10 + M3, δ = 0.3 | student alone (10 students) | **cost point**: student + M3, δ = 0.5 (10 students) | TeleAI |
|---|---|---|---|---|---|
| S2, split 0 / 10 splits | 0.598 / 0.601 | **0.669 / 0.660** | 0.570 ± 0.022 | **0.608 ± 0.027 / 0.610 ± 0.023** | 0.617 |
| S1 | 0.743 | **0.798** | 0.755 ± 0.017 | **0.790 ± 0.019** | 0.812 |
| 2-annotator S2 (test regime) | 0.549 | **0.589** (0.538 / 0.635 / 0.593) | 0.536 ± 0.021 | **0.567 ± 0.024** | — |
| in-set rate | 0.623 | 0.701 | 0.620 | 0.658 | — |
| LLM calls per item | 0 | 0.36 | 0 | 0.57 | 1.94 |
| GPU-s per item (vs M1) | 0.522 (1×) | 0.558 (1.07×) | 0.052 (0.10×) | **0.111 (0.21×)** | n/a |
| input / output tokens per item | 4,894 / 0 | 5,284 / 0.36 | 489 / 0 | 1,117 / 0.57 | ~7,300 / ~380 |

- **The accuracy point over M1:** +0.067, 95% [+0.009, +0.124], bootstrap over items.
- **The cost point over the undistilled single model + M3, same seeds, paired over the 10 seeds:**
  - S2 +0.033 (8/10), and +0.036 over 10 splits (7/10);
  - S1 +0.035 (9/10);
  - 2-annotator S2 +0.036 (7/10).
- **Per-class F1:**
  - **Accuracy point vs M1:** it gains on 7 of 9 classes (Partial +0.169, Declining +0.131, Dodging
    +0.125), loses on Claims ignorance (−0.031), and Clarification is unchanged.
  - **Cost point:** it is close to M1 on most classes, and better on Explicit (0.743 vs 0.667), General
    (0.498 vs 0.420) and Declining (0.731 vs 0.688). It is weaker on Partial/half-answer (0.170 vs 0.231)
    and Implicit (0.545 vs 0.575).

**Part B. Ensembles of distilled students** (dev S2, mean over 10 random subsets per K; split 0):

| K students | alone | + M3, δ = 0.3 | + M3, δ = 0.5 | + M3, δ = 1 | GPU-s at δ = 0.5 |
|---|---|---|---|---|---|
| 1 | 0.570 | 0.591 | **0.608** | 0.612 | 0.111 |
| 2 | 0.591 | 0.596 | 0.603 | 0.596 | 0.164 |
| 3 | 0.582 | 0.592 | 0.600 | 0.607 | 0.217 |
| 5 | 0.587 | 0.596 | 0.601 | 0.603 | 0.322 |
| 10 | 0.575 | 0.595 | 0.630 | 0.618 | 0.583 |

**Against the predictions.**
| | prediction | result | verdict |
|---|---|---|---|
| H1 students gain less from ensembling; student K = 10 alone 0.59–0.62 | | K = 1 → 2 / 3 / 5 / 10: +0.021 / +0.012 / +0.017 / +0.005 (M1 seeds: +0.065 for K = 1 → 10); K = 10 alone 0.575 | direction **held, strongly**; range failed (0.575) |
| H2 student K = 3 + M3, δ = 0.5: 0.61–0.64, on the combined front | | 0.600; dominated by one student + M3 | failed |
| H3 student K = 10 + M3, δ = 0.3 does not beat M1 K = 10 + M3 (0.669) by more than ±0.03 | | 0.595 | held |

**Decisions, by the registered rules.**
- **No middle operating point.** No student-ensemble point lies on the combined front:
  - with K = 2–5, they cost more than one student + M3 and are no better;
  - 10 students + M3 at δ = 0.5 (0.630 at 0.583 GPU-s) is dominated by M1 × 10 + M3 at δ = 0.3 (0.669 at
    0.558).
- **The accuracy point stays M1 × 10 + M3 at δ = 0.3.**
- **The frozen system therefore has two operating points:**
  - **accuracy:** M1 × 10 + M3, δ = 0.3;
  - **cost:** one student + M3, δ = 0.5.

**What the results say.**
- **Distillation removes the diversity that ensembling feeds on.** All ten students are trained to match
  the same soft targets, so they make nearly the same predictions. Averaging them adds little (≤ +0.02),
  while averaging ten independently trained M1 seeds adds +0.065. One student keeps most of what a student
  ensemble offers, which is why the single student is the efficient point.
- **For more accuracy, use diverse members:** the independently trained M1 seeds with M3 (the accuracy
  point), at about 5× the cost of the cost point.
- **In the test regime (2-annotator reference sets):**
  - the accuracy point scores 0.589 (M1 alone 0.549);
  - the cost point scores 0.567 ± 0.024.
  - Both rankings agree with the 3-annotator numbers.

**Cost.** Free.

---

## E19 — Robustness, baselines and controls for the final submission *(registered 2026-10-27 18:55)*

*Written before any E19 run.* Deadline: all experiments done by 2026-10-30; 2026-10-31 is for the report
and slides. Compute: JarvisLabs **spot** instances (user, 2026-10-27), ≈ ₹3,400 of ₹4,001.81; if over
budget, F is cut first, then E to 2 seeds. The lab server is fully occupied by other users.

**Why.** A reviewer's read of E14–E18 (2026-10-27) raised these threats, and E19 answers them:
- **The 308 "dev" items are the only evaluation set**, and several design decisions were made on them:
  - the Qwen backbone and 10 seeds;
  - K = 10 (E14 H6);
  - the go decision for M3 (E14 H5);
  - the E17 screen bar;
  - **the cost point's δ = 0.5, chosen after the registered δ = 0.3 failed**.

  The official ranking used a new 237-item set with 2 annotators. Its labels are unpublished.
- **The headline 0.669 is a favourable draw.**
  - It is split 0; the mean over 10 splits is 0.660.
  - At K ≤ 5, M3 adds only +0.02 to +0.03 (E16).
  - The bootstrap holds both the seeds and the fitted parameters fixed.
- **The cost of an M3 call is understated.**
  - `cost.json`'s 0.103 GPU-s averages over frontier calls that ran in larger batches, plus copied records.
  - The system's own dev calls cost 0.151 GPU-s.
  - The cost point becomes ≈ 0.26× M1, not 0.21×.
- **Missing comparisons:**
  - the LLM alone, without M1;
  - random routing;
  - a candidate-order control (the letters follow class order, so A is mostly Explicit);
  - controls for distillation;
  - a larger backbone.

**Tier 0 (CPU, existing outputs; `preregistered_analyses/e19_tier0.py` → `raw/E19_tier0.txt`).**
1. Report the 10-split mean as primary, with its spread, and the bootstrap with refitted parameters.
2. Correct the cost table.
3. Evaluate the cascade on all 3,448 train rows: cross-fitted M1, single gold label.
4. Ablations on dev:
   - routing: random routing at matched δ, other router signals, and an oracle;
   - fusion: λ = 0 and λ = 1.
5. Strata on dev: interview seen in train or not, answer length, and annotator agreement.
6. Letter-position bias of M3.
7. Test-set prediction files for both operating points.
8. The comparable human ceiling (2-annotator regime).

**GPU arms (Jarvis spot).**
- **A. The LLM without M1, on dev:**
  - Qwen3.8-27B-FP8 over all 9 labels, 1 token, no thinking;
  - the same with thinking;
  - TeleAI's published pipeline (CAMSR-CoT, github ther7777/semeval-2026-task6-camsr-cot) run on the same 27B (also E20 S5);
  - DeepSeek-V4-Flash with thinking over 9 labels, through the API.

  *Expectation:* every LLM-alone arm scores below M1 alone (0.598 S2). *Read-out:* the S2 / S1 table plus calls and tokens per item.
- **B. Candidate order and set size:**
  - M3 over all dev items and the 300 screen rows, with the candidates in reverse order and two cyclic shifts.
  - **Permutation-averaged q** (the mean over 4 cyclic orders) is adopted in place of single-order q **only if** it beats single-order q on the 300 screen rows (train, single gold) in both accuracy within C(x) and NLL. The decision is not made on dev; dev is reported either way.
  - Set sizes 2, 3, 6 and 9 on dev (δ = 1): reported only.
- **D. Five new M1 seeds (10–14).** Same recipe as E14 (`Q8_fullq_lora_12ep`).
  - Draw 200 random 10-of-15 seed subsets. For each, build M1, top-4 candidates and M3 (local and de-duplicated), then the fused system at δ = 0.3.
  - **H-D:** the median accuracy point over subsets lies below 0.669, and 0.669 lies within the subsets' 5–95% range.
  - The median and the 5–95% range become the primary statement of the accuracy point.
- **E. Distillation controls.**
  - Two students, seeds 0–2, paired with `Q8_distill_a05` seeds 0–2.
  - Each uses the same candidate set per row (M3's) and the same α = 0.5. Only the distribution over C(x) changes:
    - `Q8_distill_a05_m1`: M1's own calibrated cross-fitted q (self-distillation; mean target on gold 0.739, entropy 0.56);
    - `Q8_distill_a05_unif`: uniform over C(x) (candidate-aware label smoothing; 0.606 / 1.12).
  - The M3 targets sit in between (0.713 / 0.73). Targets: `cascade/distill_targets.py --teacher m1|uniform`.
  - **H-E:**
    - the M3-taught student beats both controls on single-model S2 by ≥ 0.015, as a mean over the 3 paired seeds;
    - as a system (+ M3, δ = 0.5), by ≥ 0.01.
  - If a control comes within 0.015, the claim that the student learns "the LLM's knowledge" is dropped. The student is then described as a soft-target / candidate-smoothing effect.
- **F. Backbone scale.** `Q14_fullq_lora_12ep`: Qwen3-14B-Base + LoRA, the E14 recipe otherwise, seeds 0–2.
  - **H-F:** single-model S2 ≥ the 8B's seeds 0–2 (mean 0.543) + 0.02.
  - Reported as a third point on the encoder → 8B → 14B axis. No system-level claim (3 seeds).

## E20 — Spotlight: where should the LLM think? A value-of-compute map and a learned router *(registered 2026-10-27 18:55)*

*Written before any E20 run.*

**Why.** We have the 1-token M3 decision for every train row. Comparing it with the cross-fitted M1
(`/tmp`-scratch analysis, 2026-10-27 18:30) shows the following:
- **Overall**, the LLM is a worse classifier than M1 (accuracy 0.479 vs 0.506).
- **By M1-uncertainty quintile** (Q1 = most confident), the LLM's net effect (fixes − breaks) is:
  - train: −41, −76, −11, +4, +31;
  - dev: −4, 0, −7, +8, +8.

So the LLM's value changes sign with M1's uncertainty. A uniform-reasoning pipeline (TeleAI: CoT on
~90% of items, 1.94 calls) spends most of its compute where it cannot help. The question is where test-time
compute should go, and whether that can be learned from train.

**Rungs** (all within M1's top-4 C(x) unless stated):
- **R0:** keep M1.
- **R1:** 27B, 1 token (M3 as now).
- **R2:** 27B with thinking (local vLLM, `enable_thinking=True`). The answer letter's distribution is read at the answer token. Thinking is capped at 4,096 tokens.
- **R3:** DeepSeek-V4-Flash with thinking (Jarvis Model API). It runs on the least-confident 40% of train rows and on all dev and test rows.

Cost per rung is measured in GPU-s on the RTX PRO 6000 (R1, R2) and in tokens / USD (R3).

**Mini-experiments.**
- **S1 — value map.**
  - Net, fix and break counts per M1-u quintile, for each rung, on train (cross-fitted, n = 3,448) and dev, with bootstrap CIs.
  - Also: where the net value crosses zero.
- **S2 — reasoning.** R2 on all 3,448 train rows, 308 dev and 237 test, with a timing pilot first on the 300 screen rows.
- **S3 — frontier rung.** R3 as above; API cap $10.
- **S4 — learned value router.**
  - **Model:** for each rung r ≥ 1, a cross-fitted (5-fold, train) multinomial logistic regression predicts {fix, break, neither} relative to M1.
  - **Features**, all available before any call:
    - quantile-normalised top probability, margin and entropy (quantiles within each set, because train has single-model cross-fitted probabilities and dev has the 10-seed mean);
    - M1's top class (one-hot);
    - answer length and question length (log).
  - **Allocation:** expected net gain = P(fix) − P(break). Under a budget B (GPU-s per item), a greedy gain-per-cost allocation over the rungs.
  - **Evaluation:** on held-out train folds and on dev. Dev uses the frozen nested-CV fusion; the routed items replace "the δ least confident".
- **S5 — uniform-reasoning anchor.** TeleAI's pipeline on the same 27B (arm A). All systems go on one figure: dev S2 against GPU-s per item.

**Hypotheses.**
- **S-H1 (value map).** For R1 and R2, the net value per quintile rises with M1's uncertainty (Spearman ρ > 0 over the 5 quintiles, on train), and it is negative in Q1–Q2.
- **S-H2 (overthinking).** On train:
  - R2 beats R1 by ≥ 2 accuracy points in Q4–Q5;
  - R2 is not better than R1 in Q1–Q2.

  On dev, uniform R2 (all items) scores below the routed system at the same number of calls.
- **S-H3 (learned router).**
  - On held-out train folds, the value router beats confidence routing (deferring by u) in S2 at ≥ 3 of the 4 budgets matching δ ∈ {0.1, 0.2, 0.3, 0.5} of R1 calls.
  - The multi-rung allocation dominates every single-rung policy at matched GPU-s.

**Decision rule.** The value router replaces confidence routing in the final system **only if S-H3 holds
on held-out train folds**: selection is on train, not dev. Dev numbers are reported either way.

## E19/E20 amendments, E21 and E22 *(registered 2026-10-29, 13:30 IST, before any run they describe)*

*Written after the E19 arm-E lanes started (13:03) and before any of their results; nothing below depends
on a result.* The operational plan, budget and timetable are in
[`06_final_push_handover.md`](06_final_push_handover.md).

**What changed since 2026-10-27, and why.**
- **Compute.**
  - Jarvis spot pricing exists only for containers (the SDK refuses `is_spot` for VMs).
  - On an H200 spot container, an 8B training step took **6.83 s** (≈ 1,050 tokens/s). The GPU was
    0–43% busy while one CPU core sat at 100%, though a bare bf16 matmul reached 749 TFLOPS. The training
    loop is CPU-bound there.
  - On an on-demand RTX PRO 6000 VM (₹207/h), the same step takes **1.44 s**, about 56 min per 12-epoch
    run.
  - **Every GPU arm therefore runs on RTX PRO 6000 VMs.** The H200 attempt and the reachability probes
    cost ₹122; the balance at 13:00 was ₹3,748.
- **The laptop drives Jarvis from Windows Git Bash** (`JL_PY`, see 06 §4). WSL's network ran at about
  13 KB/s on the hotspot.
- **A review of the plan (2026-10-29)** found the issues fixed below. It also added three findings that
  E21 and E22 act on:
  - **96% of official-test answers start with a speaker tag** ("THE PRESIDENT.", "The President."),
    10.5% contain footnote markers ("[25.]") and 6.8% contain all-caps header lines. A check of train and
    dev found none of these; that check was a read-only script run on 2026-10-29 and is to be re-run by
    E21's code.
  - **The M3 call's cost grows with the whole prompt, so the shared preamble is not cached.** The E16
    prompt variants give about 1.33 × 10⁻⁴ GPU-s per prompt token, with an intercept near 0. Qwen3.8-27B
    is a hybrid linear-attention model, and vLLM reuses such prefixes only at block boundaries.
  - **Cross-fitted train rows leak through sibling answers.** On slice rows whose answer has a sibling in
    the training part, three M1 seeds predict the sibling's training label 52–62% of the time, while the
    gold label equals a sibling's label 44% of the time. Train-side value maps and routers fitted on all
    rows are therefore biased.

**E19 amendments.**
- **Arm E runs seeds 0–2, as registered.** The 2026-10-27 handover's lanes listed seeds 0 1; that was
  an error. Lanes: `Q8_distill_a05_m1` (VM 552643) and `Q8_distill_a05_unif` (VM 552645), seeds
  sequential, from 13:03.
- **Arm D is dropped.** It would cost ≈ ₹1,050. Tier 0's refit bootstrap and E16's seed subsets already
  bound the accuracy point. And 10-of-15 subsets share at least 5 seeds, so H-D's second clause would
  hold almost by construction.
- **Arm F is reinstated with 2 seeds.**
  - `Q14_fullq_lora_12ep`, Qwen3-14B-Base, seeds 0 and 1, the E14 recipe.
  - No gradient checkpointing if the pilot fits in 96 GB; otherwise checkpointing (the same
    computation).
  - **H-F, restated for 2 seeds:** the mean single-model S2 ≥ the 8B's seeds 0–1 mean (0.546) + 0.02,
    i.e. ≥ 0.566.
  - Added, exploratory: a mixed ensemble of 10 × 8B and 2 × 14B, with one per-backbone weight fitted with
    τ by nested CV, reported against 10 × 8B. With 2 seeds it is not adopted into either operating point.
- **Arms A (the 27B rungs) and B run on a local vLLM copy of Qwen3.8-27B-FP8 on a Jarvis RTX PRO 6000 VM**,
  the lab server's weights, not the Model API.
  - **Faithfulness gate:** the VM re-scores 100 existing dev (item, candidates) pairs. If argmax agreement
    with the saved records is below 0.98, every analysis that mixes saved and new records re-scores all of
    them on the VM.
  - If the server already ran A or B through the API, those results stand, labelled by source.
- **The Model API is capped at $4 for the rest of E19/E20**, for the DeepSeek-V4-Flash rungs only.

**E20 amendments.**
- **Primary rows.** The value map (S1) and the learned router (S4) are fitted on the **train rows whose
  answer text has no sibling** (≈ 31% of train), where cross-fitting does not leak. All-row versions are
  reported as well.
- **S4 features** gain the question–answer specificity score of E22e.
- **R2 (thinking) is cut** to at most 500 of those rows, stratified by M1-uncertainty quintile, plus dev
  and test, on the local vLLM. It is dropped if the inference lane is not done by 18:00 IST.
- **R3** stays within the $4 API cap.

### E21 — The official test set's shift: transcript artifacts and label-free routing *(registered 2026-10-29)*

**Why.**
- Test answers carry transcript formatting that the models never saw in training (above).
- Test answers are also shorter: a median of 59 words, against 292 on dev.
- At the slice threshold, the accuracy point defers 0.253 of test but 0.357 of dev.

None of this can be scored on test, so every rule is fixed here.

**a. Sensitivity (dev).**
- **Perturbation.** `qevasion/normalize.py` gains a function that injects the artifacts into dev answers at
  test's rates, with seed 21:
  - a leading speaker tag drawn from test's tags;
  - footnote markers;
  - one header line;
  - a trailing turn by another speaker.
- **Scoring.** The 10 M1 seeds and the 10 students score the perturbed dev (inference only).
- **H21a:** ≥ 5% of the 10-seed M1's dev predictions change, or its dev S2 falls by ≥ 0.01.
- **Control:** an unperturbed re-run reproduces the saved `dev_probs` (max |Δp| < 10⁻³).

**b. Normalization (test).**
- **The rules:**
  - drop the leading speaker tag;
  - drop footnote markers;
  - drop all-caps header lines;
  - cut the answer at a line that opens a turn by another speaker (only a well-formed "Title Name." or
    "Q." at the start of a line).
- **Check:** on train and dev the rules must change ≤ 1% of answers.
- **Decision:** if H21a holds, the test submission files are rebuilt from normalized test text. M1, the
  student and M3 (on the new candidate sets) are all re-run. Otherwise the raw text is kept. Both are
  reported.

**c. Label-free routing threshold.** The test submission defers the same fraction of test as the slice
threshold realises on dev. That is a quantile of test's own u; it uses no labels. Files with both
thresholds are written, and the number of test labels that differ is reported.

**d. An expected-macro-F1 decoder, as a control.**
- **Method.** Annotators are drawn from the temperature-scaled posterior, with the temperature fitted by
  NLL on the slice. 300 sampled 2-annotator reference sets are used, with coordinate ascent over the
  labels.
- **Scope.** It is applied to M1 and to the fused system, and compared with logit adjustment on dev:
  10-split mean and 2-annotator mean.
- **It is reported, not adopted.** Choosing between the two rules on dev would be selection on dev.

### E22 — A cheaper, debiased adjudicator and an escalation ladder *(registered 2026-10-29)*

**Why.**
- M3 calls are ~62% of the cost point's GPU time, and the preamble is not cached (above).
- M3 picks letter A for 52% of dev items (Tier 0 G).
- Extra compute helps only on uncertain items (Tier 0 D).
- Seed ensembles help through diversity, and students lack it (E18).

**a. Cost with a working prefix cache.**
- **Setup.** The fixed preamble is padded with whitespace to a multiple of the cache block that vLLM uses
  for this hybrid model, and each batch starts with a warm-up request. vLLM's prefix-cache hit rate is
  logged.
- **Measurement.** GPU-s per call is measured with the E19 protocol (one pass over the 308 distinct dev
  items) for the original and the cached prompt, together with one M1 seed's GPU-s per item, **all on the
  same VM**, so every ratio comes from one GPU.
- **H22a:** the cached prompt costs ≤ 0.6 × the original per call, and its q has argmax agreement ≥ 0.98
  with the original.
- **Use.** Accuracy numbers always use the original prompt. If H22a holds, cost tables gain a "cached"
  column.

**b. Debiasing by permutation.** This is E19 arm B as registered: 4 cyclic orders, adopted only if
permutation-averaged q beats single-order q on the 300 screen rows in both within-C accuracy and NLL.
It runs on the local vLLM.

**c. Escalation ladder.**
- **The stages:**
  1. **Stage 0:** one student (`Q8_distill_a05`, each of the 10).
  2. **Stage 1:** on the fraction e1 of items with the highest student u (threshold from the slice), add K
     M1 seeds. These are the E16 K = 5 subsets, with subset i paired with student i. Fuse with fixed
     weight 0.5: log p = 0.5 log p_student + 0.5 mean log p_M1.
  3. **Stage 2:** on the fraction e2 of all items with the highest fused u (a subset of the escalated
     items), call M3 over the fused top 4, fused as in E15 (nested CV).
- **Primary configuration, fixed now:** e1 = 0.4, K = 5, e2 = 0.3.
  - Cost = 0.052 + 0.4 × 5 × 0.052 + 0.3 × c_M3.
  - That is ≈ 0.20 GPU-s with c_M3 = 0.151, or less with E22a's cached cost.
- **Grid** e1 ∈ {0.2, 0.4, 0.6} × K ∈ {3, 5} × e2 ∈ {0.2, 0.3, 0.5}: reported as a curve, not used to
  choose.
- **H22c:**
  - the primary ladder's 10-split mean S2, averaged over the 10 students, is ≥ 0.625 (the cost point's
    0.610 + 0.015);
  - and it beats the mean of 200 random escalations at the same e1 and e2 by ≥ 0.01.
- **If H22c holds,** the ladder becomes a third operating point, with a test file (new M3 test calls on
  the VM). Otherwise it is reported on the frontier.

**d. The learned router** is E20 S4 as amended above: singleton rows, with the specificity feature.

**e. Question–answer specificity.**
- **Model and score.** Qwen3-8B-Base, frozen. For every train, dev and test item, s(a | q) is the mean log
  probability of the first 256 answer tokens after "Question: {sub-question}\nAnswer:".
- **Contrasts:**
  - with three random sub-questions from other interviews (seed 22): PMI_c = s(a | q) − mean s(a | q′);
  - with an empty question: PMI_0.
- **H22e:** on train, PMI_c is ordered Clear Reply > Ambivalent > Clear Non-Reply. The Spearman ρ between
  clarity level and PMI_c must be > 0, with a bootstrap 95% interval excluding 0.
- **Use.** It is a router feature (d) whatever the result, and an interpretive figure.

### E23 — Freeze *(was "E21" in the 2026-10-27 handover)*

Final tables, the frontier figure with every operating point under both cost accountings, test files,
docs, the HF model card and W&B summaries, from E19–E22's outputs. Nothing new is run.

### E19–E22 results *(2026-10-29 → 30)*

**Sources:**
- `raw/E21_E22_analysis.txt`: `preregistered_analyses/e21_e22_analysis.py` (E19 A/B, E21a, E22a/b/e);
- `raw/E20_router.txt`: `preregistered_analyses/e20_router.py`;
- `raw/E23_systems.txt`: `preregistered_analyses/e23_systems.py` (operating points, E19-E system part, E22c);
- `raw/E23_final_submission.txt`: `evaluation/make_final_submission.py`.

The runs are on HF:
- the arm-E students under `Q8_distill_a05_{m1,unif}/`;
- the inference lanes under `E21/`, `E22/` and `M3/qwen38_27b_k0_{vm,jarvis,jarvis_cached,shift1,shift2,shift3,reverse,all9,vm_tn*}/`;
- their logs under `logs/infer_lane/`.

**How it ran.**
- **Two inference lanes on Jarvis RTX PRO 6000 VMs:**
  - **2026-10-29:** a first lane. A reporting bug crashed it at 14:09, and with no monitor running the VM sat
    idle until 18:45 (≈ ₹950). It was then rerun.
  - **2026-10-30, 11:53–12:18:** a second pass, with the self-managing lanes of 06 §6.
- **E19 arm E:** two lanes, 6 seeds, done 16:04 on 2026-10-29.
- **Credit:** ₹3,748 at the start of 2026-10-29 → ₹785.80 at the end. Arm F (14B) was not run.

**The faithfulness gate failed.** The VM's 27B agrees with the lab server's records on 92.5% of dev
items (mean max |Δq| 0.038), against the registered 0.98. On the other splits it agrees on 95.9–97.0%.
- As registered, every record set the analyses need was re-scored on the VM: 6,497 items, tag
  `qwen38_27b_k0_vm`.
- Every new comparison uses only the VM's records.
- The E15–E18 lab numbers are kept as what they were.

**E19 arm A (the 27B alone, all nine labels, dev):** S2 **0.591**, S1 0.750, in-set 0.610. That is below
the 10-seed M1 alone (0.598), so the expectation held.

**E19 arm B / E22b (option order).**
- **Position bias by order** (300 screen rows): letter A is picked 0.40 of the time in taxonomy order
  (where A is usually Explicit), and 0.22–0.29 in the shifted and reversed orders.
- **Within-C accuracy:** 0.327 for taxonomy order alone, 0.343 for the mean over the 4 cyclic orders.
- **NLL:** 1.476 vs 1.431.
- **The registered rule adopts permutation-averaged q:** it is better on both.
- **On dev** (reported only), the in-set rate of the argmax within C is 0.643 for taxonomy order and 0.604
  for the average.

**E19 arm E (distillation controls), single-model dev S2,** seeds 0 / 1 / 2:

| Student | Seeds 0 / 1 / 2 | Mean |
|---|---|---|
| M3-taught | 0.557 / 0.596 / 0.572 | **0.575** |
| M1-taught (self-distillation) | 0.527 / 0.570 / 0.593 | 0.563 |
| uniform-taught | 0.467 / 0.528 / 0.552 | 0.516 |

- **H-E, single model:** the M3-taught student beats the uniform control by +0.059 (3/3 seeds). It beats
  self-distillation by only +0.012 (2/3), below the registered 0.015, so **the H-E single-model part
  fails**.
- **By the registered rule,** the claim that the student learns "the LLM's knowledge" is dropped. The
  student's gain is described as a soft-target / candidate-smoothing effect. Most of it is matched by M1's
  own soft targets on the same candidate sets.
- The system part is reported under E23 systems below.

**E20 (amended): value map and learned router, on the 1,067 singleton train rows.**
- **Value map:** net fixes of the VM's 27B over the cross-fitted M1, by fifth of M1 uncertainty (most
  confident first), are **0 / −7 / +2 / +7 / +12**. Every 95% interval includes 0.
- **Across all 3,448 rows** (Tier 0, lab records) the same map was −41 / −76 / −12 / +5 / +31. **Most of the
  large losses on confident items came from sibling leakage.** On leak-free rows, the LLM roughly breaks
  even where M1 is confident and helps where it is not.
- **S-H1:** net value rises with uncertainty (Spearman ρ = 0.9 over the fifths), so that part holds. "Negative
  in Q1–Q2" holds only for Q2.
- **The learned router** (multinomial logistic regression of fix / break / neither on confidence features,
  M1's class, lengths and the specificity score; 5-fold, held-out) **loses to confidence routing at all 4
  budgets.** Macro-F1 at δ = 0.1 / 0.2 / 0.3 / 0.5:
  - router 0.459 / 0.458 / 0.456 / 0.476;
  - confidence 0.469 / 0.477 / 0.479 / 0.492;
  - random 0.459 / 0.462 / 0.465 / 0.471.

  S-H3 fails, so **confidence routing stays.** The thinking rung (R2) was dropped (past 18:00), so S-H2 and
  the multi-rung allocation were not evaluated.

**E21 (test transcript shift).**
- **Normalization** changes 0.0% of train and dev answers and 97% of test answers. That confirms the
  artifacts are test-only: speaker tags 96.2%, footnotes 10.5%, headers 7.6%, other speakers' turns 6.3%.
  On test the median answer goes from 59 to 52 words.
- **Control:** 20/20 seed folders reproduce their saved probabilities exactly.
- **H21a held.** Injecting test's artifacts into dev changes **15.6%** of the 10-seed M1's dev predictions
  and lowers S2 by **0.048** (0.617 → 0.570 at a fixed τ = 1.15). A single student changes 11.8% on average.
- **Decision (registered): the test files are rebuilt from normalized text.** Normalization changes 34 of
  237 of M1's test labels.
- **Deviation:** the first lane's adapter-swapping inference did not reproduce the M1 seeds. It was replaced
  by one fresh base model per adapter, which reproduces all 20 seed folders exactly.
- **A caveat for the report:** 15 test answers lose more than half their words, because the rule cuts at
  another speaker's turn. The annotators saw the full text.

**E22a (cached prompt).**
- **Cost:** on one VM, the cache-aligned prompt costs **0.050 GPU-s per call against 0.101** (0.49×).
- **Agreement:** it moves q enough that argmax agreement is only 0.906 (mean max |Δq| 0.055), against the
  registered 0.98, so **H22a fails**. The padding is not neutral for this model.
- **Use:** caching is reported as a cost measurement only. Accuracy numbers use the original prompt.
- **One M1 seed** costs 0.040 GPU-s per item on the same GPU, and reproduces exactly.

**E22e (specificity).**
- **Training rows:** PMI_c is +0.180 / +0.099 / +0.115 for Clear Reply / Ambivalent / Clear Non-Reply.
  Spearman ρ with the clarity order is +0.095 [+0.060, +0.131], so **H22e holds by its registered
  statistic**.
- **But the signal is weak.** The AUROC for Clear Reply against the rest is 0.566, and Ambivalent and Non-Reply
  are not ordered. On dev, ρ is +0.058 [−0.065, +0.174].
- **As a router feature** it did not rescue the router (E20 above).

### E23 results: the final systems *(2026-10-30)*

**Source:** `raw/E23_systems.txt`. Dev, nested CV split 0 / mean of 10 splits / S1. Every system uses
one copy of the 27B: the VM's.

| System | Lab records (E15–E18, reference) | VM, single order | **VM, order-averaged q (final)** |
|---|---|---|---|
| Accuracy point: M1 × 10 + M3, δ = 0.3 (0.357 calls) | 0.669 / 0.660 / 0.798 | 0.639 / 0.637 / 0.787 | **0.650 / 0.639 / 0.788** |
| Cost point: student + M3, δ = 0.5 (mean of 10 students, 0.574 calls) | 0.608 / 0.610 / 0.790 | 0.599 / 0.606 / 0.784 | **0.608 / 0.606 / 0.785** |

**Reading this table.**
- **The lab and VM copies are the same FP8 weights** on a different serving stack and GPU. They disagree on 7.5%
  of dev decisions, and that alone moves the accuracy point by 0.02–0.03.
- **The lab headline (0.669) is the high end of this spread.** With one consistent copy and the adopted
  debiasing, the accuracy point is 0.650 on split 0 (0.639 over 10 splits). That is still above M1 alone
  (0.598 / 0.601) and TeleAI's dev 0.617.
- **The cost point is unchanged.**

**E19-E, system part** (students + M3 at δ = 0.5, VM records, single order; 10-split mean S2, seeds 0–2):

| Student | 10-split mean S2 |
|---|---|
| M3-taught | 0.615 |
| M1-taught | 0.589 (M3 ahead by +0.026: +0.028 / +0.032 / +0.018) |
| uniform-taught | 0.603 (M3 ahead by +0.012: +0.014 / −0.007 / +0.028) |

- **The system part of H-E holds** against both controls (≥ +0.01).
- **Taken with the single-model part,** which failed against self-distillation (+0.012 < 0.015), **H-E as
  registered is not met.**
- **So, by the registered rule,** the student is described as a soft-target / candidate-smoothing effect. In
  the cascade, though, the M3-taught student combines with M3 better than either control does.

**E22c, the escalation ladder** (VM records, single order; GPU-s on the E19 lab scale):
- **Primary configuration** (e1 0.4, K 5, e2 0.3): 10-split S2 **0.600**, S1 0.785, at 0.221 GPU-s and 0.358
  calls per item.
- **H22c part 1** (≥ 0.625) **fails**.
- **H22c part 2 holds:** primary minus random escalation at the same rates is +0.018 (random 0.582 ± 0.020).
  - **Deviation:** the random draws are made within the items scored for e1 = 0.6, not over all items, because
    the pass-2 jobs held fused candidate sets only for those.
- **Grid:** best 0.604 at e1 0.6, K 5, e2 0.3 (0.285 GPU-s). Cheapest, 0.580 at 0.118 GPU-s.
- **The ladder is not adopted.** No ladder point beats the cost point (0.606 at 0.139 GPU-s).

**Final test files** (`raw/E23_final_submission.txt`, `submissions/E23_*`). All use normalized test text, the
VM's 27B and order-averaged q:
- **M1 × 10:** dev split-0 S2 0.598.
- **Accuracy point:** dev 0.650 / S1 0.788; it defers 24.5% of test (58 calls).
- **Cost point, student 0:** dev 0.627 / S1 0.747; it defers 40.1% of test.
- **Differences:** the accuracy point changes 40 of 237 test labels relative to M1, and 31 relative to the E19
  accuracy-point file.

## E24 — Pre-checks for the research extension *(2026-10-30, CPU only)*

**Why.** The mid-evaluation asked for more than the shared task. Seven literature scouts proposed ~60 candidate
additions (doc 07). Before spending any of the remaining ₹785.80, each candidate that can be tested on existing
artifacts was tested; a module goes to the GPU only if its pre-check passes.

**What.** `python -m preregistered_analyses.e24_prechecks` (≈ 5 min) → `raw/E24_prechecks.txt`. P1 sibling
structure; P2 question sensitivity of M1 on dev sibling pairs, with an annotator-noise control; P3 epistemic vs
aleatoric routing; P4 sibling mean-field decoding; P5 a parameter-free decision rule for any-annotator macro-F1;
P6 single-order position debiasing of the 27B; P7 conformal multi-reference candidate sets as router and pruner,
with the system's nested-CV fusion; P8 question–answer relevance (E22 specificity scores) by gold label.

**Results.**

| Check | Result | Consequence |
|---|---|---|
| P1 | 69% of train rows share their answer; 61.5% of 3,079 sibling pairs differ in S2, 40.6% in S1 | a free contrast set |
| P2 | annotators disagree 43% within an item, 72% across siblings, 83% across random items; M1 repeats its label on 19% of disjoint sibling pairs; both siblings right on 40% | evasion is relational; M1 reads the question but grounding is the bottleneck |
| P3 | epistemic MI catches 49/115 errors in the top 30% vs 53 for entropy | rejected |
| P4 | ±0.01, sign depends on τ | not established |
| P5 | 0.594 untuned vs 0.598 CV-tuned logit adjustment; 5/10 seeds | theory only |
| P6 | debiasing lowers agreement with the 4-order average (0.85 → 0.79) | rejected |
| P7 | coverage valid (0.977 at α = 0.05) but S2 0.564–0.617 vs 0.641–0.663 | rejected |
| P8 | relevance ordered (p = 2e-12) but AUC 0.566 for Clear Reply; Deflection as identifiable as Explicit | relevance ≠ answerhood |

**What it changed.** The extension leads with evasion as a relation (P1, P2, P8) and an LLM evasion audit, not
with new routing or decision methods; the rejected modules become a negative-results ledger. Plan and GPU budget:
doc 07.

---

## E25 — Is evasion in the question or in the answerer? *(registered 2026-10-30, 14:30 IST, before any run)*

**Where it sits in the argument.** E13–E23 built an instrument: knowledge (an 8B LLM) beats structure, the 8B
ranks better than it decides, so a 27B adjudicates (E23 accuracy point, dev 0.650). E24 showed that what it
measures is a *relation*: the label moves with the sub-question (sibling labels differ 72% vs 43% within an item),
and question–answer relevance does not capture it (AUC 0.566). Two questions follow, in order:
(1) does the instrument respond to the question *causally* (E25a), and (2) when an answer evades, is that driven
by the **question** or by the **speaker** (E25b)? E25c/d measure what the instrument costs and when it can stop.

**One frozen instrument, no parameter changes.** Every E25 label comes from the E23 accuracy point with all
parameters fixed before the run (`audit/system.py freeze` → `code/experiments/e25/frozen_system.json`): M1 =
Q8_fullq_lora_12ep, all 10 seeds; slice calibration (τ_C 0.20, T 2.15) and top-4 candidates; defer iff
u ≥ 0.638 (the slice's 70% quantile, δ = 0.3); τ_M1 1.15 for kept items; λ 0.5, τ −0.55 for deferred items
(fitted once on all of dev with the averaged VM records; nested CV of the same procedure: 0.650 / 0.639, so the
instrument is E23's). The 27B (Qwen3.8-27B-FP8, k = 0) scores every deferred item in the 4 cyclic orders;
q is their mean. Control: `jobs` + `label` on dev with E23's records reproduce the frozen full-dev predictions
exactly (S2 0.6732; checked 2026-10-30). Text: dev answers as is, test answers normalized (E21b), LLM answers
stripped of markdown (format control). Nothing is refitted on E25 data.

**E25a — Does the instrument read the question causally?** Frames (`audit/frames.py`): each dev/test item with
its question replaced by one from a different interview (*rand*, seed 25), with no question (*null*), and (dev)
with its answer replaced by one constant, contentless sentence (*qonly*). No sibling swap: siblings share turn and
answer, so a sibling's sub-question reproduces the sibling's own row (100% of dev; E24 P2 covers it).
* **H25a-1 (relation, not style).** An answer paired with an unrelated question has, by the taxonomy's definition,
  not replied to it. If the instrument models the relation, *rand* moves labels away from Clear Reply: the
  Clear-Reply share falls by at least half (dev), and at most 50% of items keep their *own* label. If it labels
  answer style, labels persist. Not obvious: training never showed a mismatched pair.
* **H25a-2 (how much the answer alone decides; descriptive).** Persistence under *null* (share of items keeping
  their own label). Read: ≥ 0.7 = labels are mostly answer-driven.
* **Control C-q (for E25b).** *qonly* shows how much label variation the question alone creates; used below.
* Test-set replication of the label-free measures (persistence, Clear-Reply share) is reported alongside dev.

**E25b — Question or speaker? The same questions, put to speakers with no political stake.** The 308 dev
sub-questions (191 interview turns) are put to four open 7–8B instruction-tuned LLMs of four families
(Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Qwen3-8B with thinking off, OLMo-2-1124-7B-Instruct;
`audit/generate.py`), greedy decoding, ≤ 400 new tokens, one user message, "spoken prose, no lists, ≤ 250
words". Conditions, each differing from P in one factor: **N** neutral analyst ("you are not the President";
factor: stake), **P** the President, in the interview, on its date (reference), **F** P's conversation, then
"With respect, you haven't answered the question. ⟨sub-question⟩" (factor: pursuit), **B** P with the President's
own answer as private notes (factor: information). N, P, B are generated once per turn and shared by its
sub-questions, as the President's one answer is. Every answer is labelled by the frozen instrument against its
dev sub-question. *Evades* = predicted clarity ≠ Clear Reply; for the President, the dev gold clarity label (and,
like for like, the instrument's dev prediction). Statistics: cluster bootstrap over the 191 turns (2,000 draws).
* **H25b-1 (a question-driven component exists).** In N, a model with no stake evades more often on the
  questions the President evaded: odds ratio OR = odds(LLM evades | President evaded) / odds(… | President
  answered) ≥ 1.5, pooled over the four models (Mantel–Haenszel), 95% CI above 1. If evasion were purely the
  speaker's strategy, OR ≈ 1. **Control:** the same OR computed on *qonly* (the instrument's labels when the
  answer carries nothing) must be < the N OR; otherwise the instrument's question prior, not the answerers,
  explains the association.
* **H25b-2 (stake).** P evades more than N: pooled paired difference ≥ +0.10.
* **H25b-3 (pursuit).** Among P answers that evade, F turns ≥ 50% into a Clear Reply.
* **H25b-4 (information; a test of the taxonomy).** With the President's own answer as notes (B), the Clear-Reply
  rate is higher for questions whose President answer was labelled *indirect but informative* (Implicit,
  General, Partial; majority gold) than for *non-answers* (Dodging, Deflection, Declining, Claims ignorance):
  difference ≥ +0.15. If the taxonomy's split between indirect answers and non-answers is about information
  content, the notes should let the LLM answer only the former.
* **Descriptive.** Per-model evasion-type profiles against the Presidents' gold profile (Jensen–Shannon distance);
  results per model (the 27B judge is Qwen-family, so Qwen3-8B is flagged).

**E25c — When does evasion become visible?** The instrument on the first 25/50/75% of each answer's words.
**H25c:** at 50%, its label agrees with the full-answer label on ≥ 70% of dev items and dev S2 drops by ≤ 0.05
(streaming detection after half the answer). Reported per gold class.

**E25d — Does the frozen base model already encode the relation? (the fast path).** Layer-wise linear probes on
Qwen3-8B-Base without an adapter (`audit/hidden.py`; PCA-256 + logistic regression on train minus the slice;
C and τ chosen on the slice). **H25d-1:** the best layer reaches dev S2 ≥ 0.45 (one fine-tuned seed: 0.525).
**H25d-2:** the probe repeats its label on ≥ 30% of dev sibling pairs with disjoint annotator sets (M1: 19%),
i.e. fine-tuning is what binds the label to the question. Latency: batch-1 time of the base model truncated to
9/18/27/36 layers against the LoRA classifier.

**Validity checks that can be acted on** (each limitation has its test):
* *The instrument may not be valid on LLM text* → two team members label 100 LLM answers (stratified by model ×
  condition, blind to both) at the clarity level; the instrument is used for E25b conclusions only if its κ with
  the humans is within 0.10 of the humans' κ with each other. Sheet: written by the analysis script.
* *Length/format may drive labels* → markdown removed for every answer; E25b effects re-estimated with log
  answer length as a covariate.
* *The instrument's own question prior* → control C-q above.
* *Cost* → GPU time per step from the lane log; nothing in E25 changes the instrument, so its cost is E23's.

**Budget.** One RTX PRO 6000 VM (~₹207/h), self-managing lane (`infer_lane.sh`, PASS=3; watchdog, upload check,
self-destroy), MAX_HOURS 3.3 → at most ≈ ₹700 of the ₹785.80 left. Analysis: `preregistered_analyses/e25_analysis.py`
→ `docs/raw/E25_analysis.txt`.

**Amendments (2026-10-30, 14:25–14:40 IST, before any E25 result was seen).**
1. *Hardware.* No RTX PRO 6000 VM was free; the lane runs on one H100 80 GB VM (₹330/h, machine 554906), with an
   independent hard stop at 2.2 h (≤ ₹726). The lane's watchdog could not read a fractional MAX_HOURS (fixed in
   the script for later runs).
2. *Control C1 failed as registered.* On the H100, M1 seed 0 on dev_own differs from its saved dev probabilities
   (scored on Blackwell GPUs) by max |dp| 0.196, with argmax agreement 0.987 (4 of 308 labels): bf16 numerics
   differ across GPU architectures. Every E25 contrast compares frames scored in this one run on this one GPU,
   so the control is recorded for every seed and reported, not used as a stop. The instrument's dev numbers in
   E25 are this run's; the honest accuracy estimate remains E23's.
3. *Throughput.* The H100 scores ~15 items/s per seed at batch 16, which would overrun the cap; batch 64 is used
   for every E25 frame (~30 items/s), and the test-set prefix frames (a replication of a descriptive measure)
   are dropped. Dev prefixes and the test rand/null replication are kept. The subject LLMs share the GPU with the
   8B scorer (vLLM memory 0.45).
4. *Two of the four subject models failed to load (recorded 15:05 IST, before any E25 analysis):* OLMo-2-7B
   (its 4,096-token context is below the requested max_model_len of 8,192) and Llama-3.1-8B (no KV-cache memory
   left at vLLM memory 0.45 while the 8B scorer held the GPU). E25b is therefore a **two-family pilot**:
   Mistral-7B-Instruct-v0.3 and Qwen3-8B, 2,464 answers. Qwen3-8B shares a family with the 27B judge, so every
   E25b result is also reported per model. "Pooled over the four models" in H25b-1 reads "over the two".
   Re-running Llama and OLMo (with max_model_len 4096 and the GPU to themselves) is the first follow-up.
5. *The 27B did not start on the H100 (15:31 IST):* 749 Mamba cache blocks < vLLM's default max_num_seqs 1,024. Re-run
   by hand on the same VM with max_num_seqs 512 (scoring is unchanged: one token, greedy), then every frame relabelled.
6. *Probes:* at ~340 s per layer on the VM's CPU, all 37 layers would overrun the cap; every 4th layer (0, 4, …, 36),
   C in {0.01, 0.1}, max_iter 300 (`audit/hidden_fast.py`, generated from `audit/hidden.py` by the recovery script).

### E25 results *(2026-10-30, 16:10 IST; `raw/E25_analysis.txt`, figures `figures/E25_*.pdf`)*

Spend: ₹555 (credit 785.80 → 230.59); the instance destroyed itself after uploading, and the collector copied
everything into `runs/E25`. Controls: on the H100, candidate sets match the saved inputs on 96.1% of dev, routing
on 99.0%, and the instrument agrees with E23's frozen predictions on 98.1% (dev S2 0.659 vs 0.673, both in-sample).

| Hypothesis | Result | Verdict |
|---|---|---|
| H25a-1 relation, not style | Unrelated question: Clear-Reply share 0.253 → **0.013**, labels kept 0.188; 62 of 78 Clear Replies become Dodging. Test (label-free): 0.338 → 0.110, kept 0.367 | **HELD** |
| H25a-2 (descriptive) | No question: labels kept 0.354; Clear-Reply share *rises* to 0.640 (nothing to evade) | the question carries much of the label |
| H25b-1 question-driven (N) | MH OR **1.18 [0.80, 1.73]** (Mistral 0.98, Qwen 1.40); adjusted 1.16. Control C-q degenerate: the contentless answer is labelled Dodging for 98% of questions | **FAILED** |
| H25b-2 stake (P − N ≥ 0.10) | −0.031 [−0.081, +0.022] | **FAILED** |
| H25b-3 pursuit (F) | 66.1% of evasive P answers become Clear Replies (Mistral 0.63, Qwen 0.69); 5.4% go the other way | **HELD** |
| H25b-4 information (B) | +0.113 [−0.006, +0.225] (without notes, P: −0.015) | **FAILED** (direction as predicted) |
| H25c streaming at 50% | agreement 0.662, S2 −0.150; refusals visible early (Declining 0.82, Claims ignorance 0.75 at 25%), General/Dodging late | **FAILED** |
| H25d-1 frozen probe ≥ 0.45 | slice-chosen layer 28: 0.397 | **FAILED** |
| H25d-2 probe less question-bound (≥ 0.30) | probe repeats its label on 44.8% of disjoint sibling pairs (M1 19%) | **HELD** |

**Exploratory, not registered as a hypothesis (reported by condition in the registered analysis):** the President's
evasion predicts the LLM's in **P (OR 2.74 [1.87, 4.17]; Mistral 3.01, Qwen 2.46)**, F (2.19 [1.28, 4.11]) and
B (4.13 [2.68, 6.34]), but not in N (1.18). The rate does not change between N and P (H25b-2), so the role changes
*where* models evade, not *how often*. N and P are labelled by the same instrument on the same questions, so a
question prior of the instrument cannot produce the N–P difference; only the answers differ. Evasion rates overall:
Presidents 0.744 (gold), LLMs 0.52–0.55 (N, P, B) and 0.20 after a follow-up. LLMs over-use Explicit (0.41–0.83 vs
0.28) and under-use Dodging (0.02–0.09 vs 0.18); the briefed condition is the closest to the Presidents' profile.
Condition effects survive a length covariate (F −1.58 → −1.52 log-odds).

**Latency (E25d).** Base 8B, batch 1, H100: 48 ms at 36 layers, 38 ms at 27, 28 ms at 18. The 9-layer reading
(424 ms) is the first configuration measured (warm-up) and the LoRA classifier's 463 ms was measured while the GPU
was shared with the hidden-state dump: neither is reliable.

**What it changed.** The answer to "question or speaker?" is neither alone: a speaker without a stake does not
evade where Presidents did, but the same models in the President's role do, on the same questions, without
evading more often. Evasion behaves like **question × role**. The instrument reads the question causally (E25a),
and only fine-tuning binds labels to questions (E25d), so there is no frozen fast path; and detection needs most
of the answer (E25c).

**Closing decisions (2026-10-30, 16:40 IST).** The experiments end here; the report and slides use what exists.
1. *Human validation dropped* (annotation is out of scope for the remaining time). It is replaced by a label-free
   check (`raw/E25_analysis.txt` §G): on LLM answers the instrument is **more** confident than on the Presidents'
   (mean u 0.462 vs 0.540, seed agreement 0.806 vs 0.730, 29% vs 36% deferred). That rules out the instrument
   being out of its depth on LLM text; it does not rule out confident errors, which the report states.
2. *No further GPU runs.* Llama-3.1 and OLMo-2 are not re-run; the P-condition association stays **exploratory**
   (two model families), and its registered replication is future work. Credit left: ₹230.59.

---

## E26 — Reduced precision: does the instrument survive FP8 and NF4, and what does it save? *(registered 2026-10-30, 22:25 IST, before any run)*

**Where it sits in the argument.** Act 5 asks how cheaply the relation can be measured. E25d found no frozen fast
path, and the 10 M1 seeds are ~91% of the accuracy point's GPU cost, so the remaining lever is numeric precision.
The yardstick is already measured: changing GPU architecture alone flipped 1.3–1.6% of M1's labels and changed
1.9% of the instrument's decisions (E25 controls). The question is whether quantization costs more than that.

**Design.** Inference only, nothing retrained. The frozen M1 (Q8_fullq_lora_12ep, 10 seeds, `adapter_best` from
the Hub) on the train slice (345, where calibration and routing were fitted), dev (308) and normalized test (237):
890 items × 10 seeds per precision (`audit/quant.py`). **Every quantized run is compared with a 16-bit run of the
same models on the same GPU type.**
* **FP8**, on a JarvisLabs RTX PRO 6000 (Blackwell, native FP8 tensor cores): adapter merged into the base, every
  decoder Linear in W8A8 e4m3 (row-wise weight scales, per-token activation scales, `torch._scaled_mm`; tensor-wise
  if row-wise is unavailable), embeddings/norms/head in 16-bit. Against a merged bf16 run on the same VM. The 27B
  scores every dev item that either precision defers (4 orders, as E25), so the **full system** is compared.
* **NF4**, on the team's local RTX A4000s (Ampere, 16 GB; NF4 is not native on any GPU: 4-bit storage, bf16
  compute): base in NF4 (bitsandbytes), adapter unmerged on top (the QLoRA layout), head in 16-bit; the bf16
  reference split over two A4000s. M1 level only (the 27B does not fit there).
Labels and routing use the frozen instrument (`code/experiments/e25/frozen_system.json`); nothing is refitted.

**Hypotheses.**
* **H26-1 (FP8 fidelity):** on dev, the 10-seed M1 label agrees with bf16 on ≥ 98% of items and the M1 system's S2
  (nested CV, 10 splits) is within ±0.01: FP8 costs no more than changing GPU does.
* **H26-2 (FP8 speed):** batched throughput ≥ 1.3× bf16. *Not obvious:* FP8 matmuls are ~2× peak, but quantizing
  activations on the fly, small batches and the 16-bit parts eat into it.
* **H26-3 (NF4, descriptive):** agreement 0.93–0.97, M1 S2 −0.01 to −0.03, throughput ≤ 1.0× (no speed-up; it
  saves memory, ≤ 0.4×). **QLoRA trigger:** if NF4 costs > 0.02 S2 or flips > 5% of labels, QLoRA (training the
  adapter on the NF4 base) is the named fix in future work; it is not run here.
* **H26-4 (the cascade absorbs quantization error; the non-obvious one):** M1's FP8-induced decision flips sit
  where M1 is unsure, which is where the 27B re-decides: ≥ 50% of the flips fall among deferred items (dev
  deferral ~36%), and the full system flips fewer decisions than M1 alone.
* **Decision rule:** FP8 becomes the instrument's recommended serving precision if H26-1 and H26-2 both hold;
  the cost claim is then the measured speed ratio applied to the accuracy point (0.576 GPU-s/item), not a
  projection.

**Budget.** RTX PRO 6000 VM ₹207.36/h, lane PASS=4 with MAX_HOURS 0.9 (≤ ₹199 incl. storage; ₹230.59 left, no
recharge possible below ₹500). The 27B scores dev only, without CUDA graphs (faster start-up; ~100 items per
order). If the cap stops the lane before the 27B finishes, labels are computed locally with M1's decision wherever
a 27B record is missing, and the analysis says so. NF4 on the local A4000s costs nothing. Analysis:
`preregistered_analyses/e26_analysis.py` → `docs/raw/E26_analysis.txt`.

**Amendment (22:30 IST).** The first lane attempt stopped at the smoke test: `audit/quant.py` did not set the model's
padding token, so batched inference refused to run (a bug, not an FP8 issue; FP8 row-wise worked). The lane was
stopped before it paused, the loader fixed, and the same VM restarted (MAX_HOURS 0.75). The lab half runs from
`code/launch/e26_lab_run.sh` on phoenix under the team account, on GPUs 0 and 2 only (GPU 1 carries another user's
job); NF4 seeds 0–4 and 5–9 run in parallel there, then the bf16 reference split over both cards.

### E26 results, FP8 half *(2026-10-30, 22:52 IST; `raw/E26_analysis.txt`)*

Spend ₹90 (credit 230.59 → 140.18); the instance destroyed itself. Noise floor: the merged bf16 model on this VM
agrees with the training-time probabilities on 97.9% of per-seed labels (max |dp| 0.26).

| | Result | Verdict |
|---|---|---|
| H26-1 FP8 fidelity | dev labels agree with same-GPU bf16 on 91.7% per seed and **96.1%** for the 10-seed mean (bar 98%); routing identical on 96.1%, top-4 sets on 84.7%; **M1 system S2 −0.020** (0.599 → 0.579, 10 splits) | **FAILED** |
| H26-2 FP8 speed | batched throughput **0.83×** bf16 (43.1 → 35.9 items/s); batch-1 latency p50 33.6 → 30.2 ms, p95 62.7 → 49.6 ms | **FAILED** |
| H26-4 cascade absorbs | 71% of FP8's 17 M1 decision flips fall on items deferred to the 27B (dev deferral 36%); **full-system S2 0.653 → 0.649 (−0.003)** against M1's −0.020; 16 vs 17 decisions flipped | **HELD** |
| Decision rule | H26-1 and H26-2 failed | FP8 (as implemented) **not adopted** |

**Reading.** FP8 costs M1 about as much as several seeds' worth of noise (−0.020 S2, 4× the cross-GPU flip rate),
but the cascade absorbs almost all of it: the flips land where M1 is unsure, which is exactly where the 27B
re-decides, so the full system loses only −0.003. The error goes where the adjudicator already looks.

**NF4 status (23:05 IST): not run in time.** On the lab server the shared network delivers ~0.2–1 MB/s, so the 16 GB
base model needs 9+ hours; JarvisLabs refuses to start an RTX PRO 6000 below a ₹244 balance (₹164 available,
recharges start at ₹500). H26-3 is therefore unanswered in the report; the lab run continues overnight and, if it
lands, is added as a follow-up. FP8's result stands on its own (both of its comparisons are on one GPU type). FP8 is not
faster here because activations are quantized in unfused PyTorch ops (per-token scales, casts, padding) around each
FP8 matmul; a speed-up needs fused kernels (e.g. vLLM's FP8 path), which is future work. The peak-memory figure
(21.9 vs 20.9 GB) includes loading the 16-bit model before conversion, so it does not measure FP8's footprint.

---

## Deferred

- **An LLM in the re-ranking slot.** The encoder pipeline comes first; the
  cross-fitted candidates and fall-back fusion from E6 are ready for it.
- **Training on all of train.** Every run holds out 10% of train to choose its best
  epoch. Once an epoch count is settled, a final model could be retrained on
  100% with that fixed count — about 10% more data, at the cost of not being able
  to choose its checkpoint.
