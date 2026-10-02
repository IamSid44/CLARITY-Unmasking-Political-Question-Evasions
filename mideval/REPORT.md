# CLARITY mid-evaluation report — Team Nier_ANLP

SemEval-2026 Task 6: classifying how politicians answer interview questions.

Every number below is on the **308-item dev set** unless marked, and comes from a file in this
repository; [`RESULTS.md`](RESULTS.md) gives the source of each. The test labels were never
released and Codabench has closed, so dev is the only place systems can be compared.
**Done** and **planned** work are kept apart: §5–6 are done, §7 is planned.

---

## 1. The problem

Given a sub-question from a presidential interview and the president's answer, classify the
answer into one of **9 evasion types** (Subtask 2). These nest under **3 clarity levels**
(Subtask 1): Clear Reply, Ambivalent, Clear Non-Reply. Both subtasks are scored by macro-F1. In
Subtask 2 a prediction counts as correct if it matches **any** of the annotators' labels: three
annotators per item on dev, two on test.

Like the dataset paper and the top two competition systems, we predict the 9-way label and derive
the 3-way label from the fixed taxonomy.

## 2. What the field shows

| system | split | Subtask 1 | Subtask 2 | source |
|---|---|---|---|---|
| TeleAI (1st), DeepSeek-V3, 3-stage prompting | test | 0.89 | 0.68 | task overview, Table 3 |
| moswisarut / ChulaNLP (2nd), RoBERTa top-5 → Kimi-K2 | test | 0.82 | 0.61 | task overview, Table 3 |
| organisers' baseline, fine-tuned Llama-70B | test | 0.82 | 0.57 | task overview, Table 2 |
| best encoder-only systems | test | ≤ 0.81 | ≤ 0.51 | task overview §5 |
| TeleAI pipeline | dev | 0.812 | 0.617 | TeleAI, Table 2 |
| DeepSeek-V3, single-step chain-of-thought | dev | 0.710 | 0.490 | TeleAI, Table 2 |
| DeepSeek-V3, asked directly for the label | dev | 0.662 | 0.421 | TeleAI, Table 2 |
| Qwen2.5-7B, LoRA fine-tuned, no chain-of-thought | dev | 0.587 | 0.495 | TeleAI, Table 2 |
| ChulaNLP's DeBERTa-large, checkpoint chosen on dev | dev | 0.65 | 0.46 | ChulaNLP paper |

- **The top systems are multi-call LLM pipelines.** TeleAI averages 1.94 model calls and about
  7,300 input tokens per item.
- **Encoders stop around 0.50 on test Subtask 2**, whatever the scale or ensembling.
- **A fine-tuned 7B model without chain-of-thought already reaches 0.495 on dev,** level with a
  frontier model reasoning step by step (0.490).

## 3. Our question

**How much of an expensive LLM pipeline's accuracy can a single-pass model recover, and at what
cost?** The end product is an accuracy-vs-cost curve, not a leaderboard position.

The first half of the project builds the cheapest point on that curve, a fine-tuned encoder, as
rigorously as we can, and finds out *why* it stops where it does.

## 4. Re-scoping from the proposal

The proposal set out five contributions. What we did with each, and why:

| | proposed | what happened |
|---|---|---|
| **C2** | Annotators agree more under a "coverage" cut of the taxonomy than under the official one | **Tested and refuted** before any modelling, from dev's three annotations (pre-registered): Krippendorff α 0.503 for the coverage cut vs 0.623 for the official one, Δ −0.120, p = 0.0004. The re-annotation study it would have motivated is dropped. |
| **C4** | A decision rule built for the multi-reference scorer: per-class thresholds over P(class in the annotator set) | **Tested; the simpler control won.** A derivation showed that when every prediction is acceptable, macro-F1 = (classes ever predicted) / 9: the rule can only buy *coverage* of rare classes. One parameter (logit adjustment) beats nine fitted per-class weights, which overfit 308 items. Details in §5. |
| **C1** | Replace the flat output with an attribute code shared across leaves | **Not built.** It needs attribute annotations (the dropped re-annotation). The taxonomy's hierarchy was instead tested three ways (§5); none beat the flat model. |
| **C3** | Annotator model from per-item annotator IDs | **Feasibility checked:** train has annotator IDs (three annotators, one per item), so the model is identifiable only through annotator marginals. Kept as analysis only. |
| **C5** | Coverage/engagement representation of the answer | **Replaced by a simpler test of the same need**: give the model the full journalist question, so it can see what else the answer responds to (§5). |

The evaluation protocol was kept and tightened: a frozen flat baseline, every change measured
against its own control, 10 seeds for any system claim, nested cross-validation for the one fitted
parameter, bootstrap intervals, and hypotheses written down before each run.

## 5. What we built (done)

**Infrastructure.**
- A replica of the official scorer, with its geometry worked out and verified.
- Frozen splits and a fully resumable training pipeline.
- A post-hoc decision-rule module, and a one-command analysis script that reproduces every number
  in this package in 24 s of CPU time.

**Floors.**

| baseline | Subtask 2 | Subtask 1 |
|---|---|---|
| always majority | 0.060 | 0.136 |
| uniform random | 0.129 | 0.299 |
| TF-IDF + logistic regression | 0.257 | 0.445 |

**The model.** DeBERTa-v3-large fine-tuned as a 9-way classifier. Two changes improved it, each
measured against its own control on 10 paired seeds:

![per-seed progression](figures/per_seed_progression.png)

| single model, 10 seeds | Subtask 2 | Subtask 1 |
|---|---|---|
| sub-question + answer, 8 epochs (baseline) | 0.315 ± 0.040 | 0.576 ± 0.030 |
| + 16 epochs | 0.362 ± 0.026 | 0.601 ± 0.016 |
| **+ full journalist question in the input** | **0.384 ± 0.030** | **0.614 ± 0.027** |

Baseline → final: Subtask 2 +0.069 (t = 3.6) and Subtask 1 +0.038, each on 9 of 10 seeds.

**Systems** (10-seed ensembles; the decision rule scored by nested cross-validation):

| | Subtask 2 | Subtask 1 |
|---|---|---|
| final model's ensemble, no dev tuning | 0.409 [95% CI 0.306, 0.472] | 0.621 |
| **final system** (ensemble + logit adjustment, chosen by a rule fixed in advance) | **0.405** | **0.648** |
| baseline ensemble + logit adjustment | 0.428 | 0.601 |
| final minus baseline ensemble, before any rule | +0.090 [+0.009, +0.168] | +0.024 |
| training on all of train, last epoch (5 seeds only; S1 without the rule) | 0.489 | 0.634 |

**What the tests showed:**
- **The rule repairs most of what the better model fixes.** Before the decision rule, the final
  model is clearly better than the baseline. The rule then lifts the baseline by +0.109 and the
  better model by nothing, so on Subtask 2 the two systems are level. On Subtask 1 the better
  model leads.
- **5-seed system numbers are unreliable here:** the same system scored 0.438 and 0.356 on two
  sets of seeds. That is why the table uses 10.
- **What did not help** (each documented with a reason):

| tried | result |
|---|---|
| a second encoder re-ranking the top 3 | 0.354 vs 0.365 |
| Balanced Softmax + focal loss | over-shifts to rare classes |
| three forms of hierarchy: post-hoc routing, a factorised head, specialist encoders | none beats the flat model |
| boundary experts for confused label pairs | no effect |
| model soups | 0.267 |
| nine per-class decision weights | erratic: 0.29–0.46 depending on which half of dev they are fitted on |

**The proposal's C4 prediction, checked.** It predicted the rule's gains would fall on items
where annotators disagree, not on unanimous ones.
- The rule raises the baseline's Subtask 2 from 0.319 to 0.428.
- Its in-set rate is unchanged overall. It loses on unanimous items (0.536 → 0.496) and gains on
  contested ones (2–1 splits 0.513 → 0.527; three-way splits 0.727 → 0.788).
- The gain comes from rare classes being named at all (`Partial` +0.43, `Claims ignorance` +0.32
  F1), plus the two most-disputed classes (`Deflection` +0.15, `General` +0.14).
- **Verdict: partly right, and the mechanism is coverage, not set membership.**

## 6. What the errors tell us

From the final model's 10-seed ensemble, with no dev tuning.

- **The model fails where humans don't.**
  - 137 of 308 predictions (44.5%) are outside the reference set, 53 of them on items all three
    annotators agreed on.
  - Scored the test set's way (two annotators), a human annotator against the other two reaches
    0.643–0.766 on Subtask 2 (mean 0.684). The model reaches 0.38.
  - The gap is the model, not label noise.
- **The confusion is about commitment as much as topic.**
  - By TeleAI's loose definition (any error involving Dodging, General or Deflection), 86.9% of
    our errors are "triangle" errors, vs their 77.7%.
  - But only 29% have both the prediction and a gold label inside the triangle.
  - Our largest error pairs are Implicit → Explicit (25), General → Implicit (20) and
    General → Explicit (20): *how committed* an answer is, not only whether it stays on topic.
- **Dodging is a strength:** recall 0.494 and F1 0.543, against TeleAI's 0.179 and 0.289.
- **Length matters, and test is short.**
  - Inputs the model still had to cut are acceptable only 20% of the time (n = 20).
  - The shorter half of dev scores 0.400 on Subtask 2, the longer half 0.283.
  - Test inputs have a median of 137 tokens against dev's 474, so dev likely understates test
    performance on this axis. Test's two-annotator reference sets cut the other way.
- **The model usually knows when it is unsure, and the answer is usually near the top.**
  - In-set rate is 0.823 for the most confident fifth of dev and 0.43–0.47 for the least
    confident three fifths.
  - An acceptable label is in the top 3 for 93.8% of items.
- **The 10 seeds are diverse but fail together.** They agree on only 51% of items. Where all 10
  agree they are acceptable 81% of the time; where they split, 50%.

Two typical errors (more in `analysis/error_examples.md`):

> *"Would you campaign against Senator Joe Lieberman on Iraq?"* — *"I'm going to stay out of
> Connecticut."* Annotators: Dodging, Dodging, Implicit. Model: General (p = 0.37). Reading this
> as a near-answer requires knowing Lieberman is a Connecticut senator.

> *"Are people who are saying that we're moving forward to a full war wrong?"* — the answer
> discusses speculation and violence without answering. Annotators: General ×3. Model: Dodging
> (p = 0.59).

## 7. Plan to the final (planned)

Each step has a go/no-go gate. Full plan in [`PLAN_REMAINING.md`](PLAN_REMAINING.md).

1. **Is the gap knowledge?**
   - Qwen3-8B fine-tuned with LoRA as a single-pass classifier. Same input, rows, train slice and
     selection rule as the best DeBERTa model; only the backbone changes.
   - **Done (3-seed screen, 2026-10-02, one RTX PRO 6000 on JarvisLabs, 21 min per seed).**
     Dev S2 0.462 ± 0.080 and S1 0.688 ± 0.040, against DeBERTa's 0.381 / 0.619 on the same seeds.
     That is S2 +0.081 and S1 +0.068, with all 3 seeds up on both. S2 is inside the
     pre-registered range of 0.43–0.50. Source: `RESULTS.md` §9.
   - *Gate:* at least +0.015 over DeBERTa with 2 of 3 seeds up, then 10 seeds. **Passed.**
   - Next (planned): seeds 3–9 for a 10-seed system comparison. Then a longer schedule: the best
     epoch was the last of 3 on every seed.
2. **Cascade.** Keep confident items on the cheap model. Send the uncertain ones, with their top-3
   or a calibrated candidate set, to an open LLM given label definitions and boundary guidance.
   Sweep the deferral rate to trace accuracy against calls per item, with TeleAI's point on the
   same axes.
3. **One-time teacher.** LLM soft labels and short rationales for the training set, distilled into
   the single-pass student. *Gate:* the student beats its own baseline by more than the seed
   spread.
4. **Measured cost.** Quantise the models and measure throughput and latency on real hardware, as
   cost per million items.
5. **Final evaluation:** 10 seeds, two-annotator scoring, the human ceiling as a reference line.

## 8. Risks and fallbacks

| risk | fallback |
|---|---|
| the 8B classifier does not beat DeBERTa | reported as evidence that the gap is label meaning, not knowledge; budget moves to the cascade, where definitions and guidance enter through the prompt |
| GPU access: the lab GPUs are shared; we have about ₹5,880 of cloud credit | encoder work on lab GPUs; cloud only for LLM runs, with a timing pilot and automatic pause; cost per run tracked |
| dev too small to separate systems | paired per-seed comparisons, 10-seed systems, bootstrap intervals; claims stated at the resolution the data supports |
| test differs from dev (shorter answers, two annotators) | report the short-answer half and the two-annotator scores alongside the headline |

**In one sentence:** a carefully controlled single-pass encoder reaches 0.384 on Subtask 2 (0.405
as a system), fails mostly on items humans agree on, and usually has the right answer in its top
three. The second half tests whether that gap is knowledge (an 8B classifier) or label meaning (an
LLM over the top three), and maps what each costs.
