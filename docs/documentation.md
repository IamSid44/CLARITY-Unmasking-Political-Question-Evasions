# The CLARITY project, explained: every experiment, why it was run, and what it changed

*Written 2026-10-30 for the team, before the final report. It covers the work from the mid-evaluation (E13) to
the last experiment (E25), as one argument. Every number has a source: the experiment log
([`02_experiment_log.md`](02_experiment_log.md)), the raw outputs in [`raw/`](raw/), and the results map
([`04_results_sources.md`](04_results_sources.md)).*

---

## Contents

1. [The task in one page](#1-the-task-in-one-page)
2. [The one-paragraph story](#2-the-one-paragraph-story)
3. [Where we stood at the mid-evaluation](#3-where-we-stood-at-the-mid-evaluation-e0e13)
4. [What the mid-evaluation feedback said, and what we changed](#4-what-the-mid-evaluation-feedback-said-and-what-we-changed)
5. [Act 1: building the instrument (E14–E18)](#5-act-1-building-the-instrument-e14e18)
6. [Act 1b: stress-testing the instrument (E19–E23)](#6-act-1b-stress-testing-the-instrument-e19e23)
7. [The turn: our pipeline is not novel](#7-the-turn-our-pipeline-is-not-novel)
8. [Act 2: evasion is a relation (E24)](#8-act-2-evasion-is-a-relation-e24)
9. [What we tested and dropped (E24)](#9-what-we-tested-and-dropped-e24)
10. [Acts 3–5: the registered experiment E25](#10-acts-35-the-registered-experiment-e25)
11. [The connected argument, start to finish](#11-the-connected-argument-start-to-finish)
12. [What we got right, what we got wrong](#12-what-we-got-right-what-we-got-wrong)
13. [Limitations, each with the test that addresses it](#13-limitations-each-with-the-test-that-addresses-it)
14. [Glossary](#14-glossary)

---

## 1. The task in one page

**SemEval-2026 Task 6 (CLARITY).** Each item is a US presidential interview exchange:

| Field | What it is | Median length |
|---|---|---|
| interview question | the journalist's full turn, often several questions | 62 tokens |
| sub-question | one question split out of that turn | 15 tokens |
| answer | the president's whole reply | 266 tokens |

The label says how the answer responds to **that sub-question**:

| Clarity level (Subtask 1) | Evasion types (Subtask 2) |
|---|---|
| Clear Reply | Explicit |
| Ambivalent | Implicit, Dodging, General, Deflection, Partial |
| Clear Non-Reply | Declining to answer, Claims ignorance, Clarification |

**Data.** Train 3,448 rows (one label each). Dev 308 rows (three annotators each). Test 237 rows (two annotators,
labels never released). Annotators agree only moderately (κ 0.48 on the nine types).

**Metric.** Macro-F1 where a prediction counts as correct if it matches **any** annotator. Macro means the nine
classes count equally, so rare classes matter as much as frequent ones.

---

## 2. The one-paragraph story

A large pretrained model, fine-tuned simply, beats every clever architecture on a smaller one. It also *ranks*
the right label better than it *picks* it, so a larger model adjudicates the uncertain cases. That system is our
**instrument** (dev S2 0.650). It is not novel by itself: other shared-task teams built the same kind of pipeline.
So we asked what the instrument can *measure*. First, the data shows that **evasion is a relation between a
question and an answer, not a property of the answer**: one answer gets different labels for different
sub-questions, far more often than annotator noise explains, and "is the answer about the question" does not
capture it. That raises two questions in order. Does the instrument respond to the question *causally*? And when
an answer evades, is that **driven by the question or by the speaker**? We answer the second by putting the
presidents' exact questions to language models, which have no political stake, and checking whether they evade
on the same questions.

---

## 3. Where we stood at the mid-evaluation (E0–E13)

**Question we asked:** *what limits a fine-tuned classifier on this task?* There were four competing
explanations, each predicting something different:

| If the limit is… | …then this should help |
|---|---|
| label noise | nothing; models should fail where annotators disagree |
| the decision rule | a better rule for the metric |
| missing knowledge | a larger pretrained model, same protocol |
| label meaning not learnable from 3,448 examples | giving the model definitions |

**What we did (E0–E12, DeBERTa-v3-large).** We changed one thing at a time against a named control on paired seeds.

| Change | Result |
|---|---|
| Logit adjustment (a decision rule correcting for class frequency) | helped |
| Longer training (E10) | helped |
| Adding the full journalist question (E8b) | helped |
| Hierarchical heads (E9) | failed |
| Rebalanced losses (E8) | failed |
| Specialist and boundary experts (E12b/c) | failed |
| A second encoder as re-ranker (E6) | failed |
| Model soups (E12d) | failed |

The best system scored 0.405 S2 (10 seeds); a single model scored 0.384.

**The decisive test (E13).** We changed **only** the backbone, to Qwen3-8B-Base with LoRA, and kept the same data,
input, loss and selection rule:
- **per model:** +0.092 S2, up on 10 of 10 seeds;
- **10-seed system:** 0.543 S2 / 0.746 S1, against 0.405 / 0.648;
- **what it means:** knowledge beat structure. The one change gained more than all the encoder-side changes combined.

**What the mid-evaluation report claimed:** "knowledge over structure", plus a plan for a cascade.

---

## 4. What the mid-evaluation feedback said, and what we changed

The feedback, paraphrased:
1. The experiments were **disjoint** and had little narrative, not even one built afterwards.
2. Several gave the **naturally expected result**, which could have saved work.
3. Some marks were lost for **haphazardly changing experiment parameters**.
4. Some marks were lost for listing **limitations that cannot be acted upon**.
5. The direction itself was promising.

How each point changed the way we work from E24 onwards:

| Feedback | Rule we adopted |
|---|---|
| Disjoint experiments | Every experiment answers a question raised by the previous one (§11 shows the chain). |
| Expected results | Before running, write the prediction **and why the outcome is not obvious**; if the answer is predictable, do not run it. Five candidate modules were dropped this way, on CPU (§9). |
| Changing parameters | One **frozen instrument** for all new experiments: every setting is fixed before the run and stored in `code/experiments/e25/frozen_system.json`. Deviations are logged as time-stamped amendments *before* results are seen. |
| Limitations that cannot be acted on | Every limitation is paired with the test or control that addresses it (§13). |

---

## 5. Act 1: building the instrument (E14–E18)

Each step was forced by the result before it.

### E14: M1, the 8B classifier (2026-10-06)
- **Why:** E13 used 3 epochs. The encoder study had shown that under-training looks like "this idea doesn't work"
  (E10), so we tested longer training first.
- **Hypothesis:** 12 epochs beat 3 epochs per model.
- **Result:**
  - per model: 0.525 vs 0.476 (+0.049, 8 of 10 seeds up);
  - **10-seed system: 0.598 S2 / 0.743 S1.**
- **What it revealed:** an acceptable label is first for 60% of items but **in the top 3 for 93%**. The model
  *ranks* better than it *decides*.
- **What it changed:** if the right label is usually in the top few, a stronger reader only needs to choose among
  those few. That is the cascade idea.

### E15: M3, an LLM adjudicator (2026-10-14)
- **Hypothesis:** a 27B LLM (Qwen3.8-27B-FP8) choosing among M1's top 4, only for the least confident items,
  beats M1.
- **Design:** one output token (a letter), no examples. Its distribution is fused with M1's. The fusion weights
  are fitted by nested cross-validation on dev.
- **Result:** 0.669 S2 / 0.798 S1 with only 30% of items deferred (0.36 LLM calls per item). For comparison, the
  winner TeleAI used 1.94 calls per item.
- **Caveat found later:**
  - averaged over subsets of seeds, the gain is only +0.02 to +0.03 (E16);
  - with the fusion refitted inside each bootstrap, it is +0.037, with a 95% interval of [−0.044, +0.120] (E19).

### E16: prompt ablation, measured cost, the cost–accuracy frontier
- **Prompt:** the full prompt (definitions plus a guide to confusable pairs) was kept. The differences were within noise.
- **Cost:** one M1 seed costs 0.052 GPU-s per item, so the 10 seeds are about 93% of the cascade's cost.
- **What it changed:** this pointed directly at distillation. If one model could replace ten, the cost would collapse.

### E17: M4, distillation
- **Hypothesis:** one student trained on 0.5 × gold + 0.5 × the 27B's soft labels replaces the 10-seed ensemble.
- **Result:**
  - single model: 0.573 vs 0.525 (9 of 10 seeds up);
  - **student + 27B at 50% deferral: 0.608 at 0.27× M1's GPU cost.**
- This became the **cost operating point**.

### E18: freeze
Both operating points were frozen. Ensembles of students did not help: they lack diversity.

---

## 6. Act 1b: stress-testing the instrument (E19–E23)

Before submitting, we asked: *which of these numbers survive scrutiny?*

| Exp. | Question | Result | What it changed |
|---|---|---|---|
| E19 Tier 0 | Is the 27B's gain robust? | Refitting inside each bootstrap gives +0.037 [−0.044, +0.120]. The 27B also costs 0.151 GPU-s per call, not 0.103. | We report the honest interval. Confidence routing still beats random routing (0.669 vs 0.604). |
| E19 A | Is the LLM alone enough? | The 27B alone over all 9 labels scores 0.591 < M1's 0.598 | The cascade is needed. Neither model alone is as good. |
| E19 B | Does the order of the options matter? | Yes: the 27B picks option A 40% of the time in one order and 22% in another. Averaging over 4 cyclic orders improves accuracy (+0.017) and calibration. | Order-averaging adopted. |
| E19 E | Does it matter *who* teaches the student? | 27B-taught beats self-taught by only +0.012 as a single model (fails the registered bar); +0.026 as a system | Distillation's value is mostly as a system. |
| E20 | Where does the LLM help, and can a learned router beat confidence? | On leak-free rows, the LLM's net value across uncertainty quintiles is 0 / −7 / +2 / +7 / +12. The learned router **lost** at 4 of 4 budgets. | Keep simple confidence routing. The LLM only helps where M1 is uncertain. |
| E21 | Do test transcripts differ? | Test answers contain artifacts ([inaudible], crosstalk). Injected into dev, they change 15.6% of predictions and cost −0.048 S2. | Test files use normalized answers. |
| E22a | Can prompt caching halve the cost? | 0.49× cost, but only 0.906 decision agreement (needed ≥ 0.98) | Rejected: it changes the answers. |
| E22c | Does an escalation ladder (student → more seeds → 27B) win? | 0.600 at 0.221 GPU-s; target 0.625 missed | Not adopted. |
| E22e | Does "does the answer depend on the question" (PMI) predict errors? | Barely (ρ +0.095) | Kept as a signal; it returns in E24. |
| E23 | Final system, with one copy of the 27B | The lab and VM copies of the 27B agree on only 92.5% of decisions, so everything was re-scored on one copy. **Accuracy point 0.650 S2 / 0.788 S1; cost point 0.608 / 0.785.** | This is the **frozen instrument** for everything after. |

---

## 7. The turn: our pipeline is not novel

After E23 we sent seven research agents through the literature: shared-task papers, efficient inference,
reframings, label variation, interpretability, data methods, and a hostile area chair. Two facts changed the plan:

| What we built | Already done at SemEval-2026 Task 6 by |
|---|---|
| Small model proposes top labels, LLM chooses | ChulaNLP (2nd on Subtask 2) |
| Confidence routing to a bigger model | TeleAI (1st), AsymVerify |
| Distillation into a small model | YNU-HPCC |

And **none of the 30 system papers** tested whether systems read the question, evaluated outside presidential
interviews, or used the taxonomy to measure anything but presidents.

**Decision:** the pipeline becomes the **instrument**, and the contribution moves to what it can measure.

---

## 8. Act 2: evasion is a relation (E24)

All results are CPU-only, from existing outputs (`raw/E24_prechecks.txt`).

### Why look here
E22e showed that "does the answer depend on the question" barely predicts errors. That raised a deeper question:
*what is the label actually a property of?*

### Finding 1: the data is relational (P1)
- **The setup:** one answer often serves several sub-questions. We call them **siblings**.
- **The numbers:** 69% of train rows share their answer with a sibling. Of the **3,079 sibling pairs, 61.5% have
  different evasion labels** (40.6% different clarity levels).

### Finding 2: it is not annotator noise (P2 control)
- **The objection:** sibling labels might differ only because annotators are noisy.
- **The test:** on dev (three annotators), we measured how often two labels from different annotators disagree.
  - Same item: **43%**.
  - Sibling items: **72%**.
  - Random items: **83%**.
- **Reading:** siblings disagree far more than annotators do on one item. **The label moves because the question moves.**

### Finding 3: our model reads the question, but grounding is the bottleneck (P2)
- **What we expected:** M1 would be "answer-only", giving the same label to siblings.
- **What happened:** on the 58 dev sibling pairs whose annotator sets do not overlap, M1 gives the same label only
  **19%** of the time.
- **But:** both siblings are right in only **40%** of these pairs.
- We **removed the claim** "models ignore the question" from the plan. Testing first saved us from publishing a
  false headline.

### Finding 4: relevance is not answerhood (P8)
- **The measure:** political science scores answer quality by *identifiability*: can the answer be picked out from
  its own question rather than random ones (Alvarez & Morrier, 2025)? We computed exactly this with the 8B model.
- **Result:**
  - it is ordered across labels (p = 2e-12) but barely discriminative: **AUC 0.566** for Clear Reply vs rest;
  - Deflection (0.80) and Partial (0.84) answers are as identifiable as Explicit ones (0.82).
- **Reading:** **an answer can be about the question without answering it.** That is the construct our instrument
  measures and relevance does not.

---

## 9. What we tested and dropped (E24)

The scouts proposed about 60 additions. Every one that could be tested on existing outputs was tested first, for free.
A module went to the GPU only if its pre-check passed. Five failed:

| Idea | Why it looked promising | What happened | Verdict |
|---|---|---|---|
| Route on epistemic uncertainty (seed disagreement) | Annotator-ambiguous items are "free" under any-annotator scoring | Caught 49 of 115 M1 errors in the top 30%, vs 53 for plain entropy | dropped |
| Decode siblings jointly (CRF / mean field) | Sibling labels share information | They do, but only 0.24 bits; gains ±0.01 that flip sign | not established |
| A parameter-free, metric-optimal decision rule (our derivation) | It would remove the tuned τ | 0.594 untuned vs 0.598 tuned; better on 5 of 10 seeds | theory section only |
| One debiased option order instead of four (PriDe) | 4× cheaper 27B | Agreement with the 4-order average *fell* (0.85 → 0.79): position is confounded with label identity | dropped |
| Conformal candidate sets as router and pruner | A coverage guarantee aligned with the metric | Guarantee held (0.977 at α 0.05), but S2 0.564–0.617 vs 0.641–0.663: sets are large where annotators disagree | dropped |

These go into the paper as a **negative-results table**. That is honest, and it answers "known techniques" objections.

---

## 10. Acts 3–5: the registered experiment E25

Registered in the experiment log on 2026-10-30 at 14:30 IST, **before any run**.

### The frozen instrument
- **Labels:** every label is the E23 accuracy point, with all parameters fixed in advance:
  - 10 M1 seeds, slice calibration and top-4 candidates;
  - defer if uncertainty ≥ 0.638 (δ = 0.3);
  - the 27B in 4 cyclic orders, averaged;
  - fusion λ 0.5, τ −0.55.
- **Control:** applying this on dev with E23's records reproduces its predictions exactly (S2 0.6732).
- **No refitting:** nothing is refitted on E25 data.

### E25a (Act 3): does the instrument read the question causally?
- **Why:** E24 shows labels depend on the question. We need to know whether the *instrument* does, causally,
  before using it to measure anything.
- **Design:** keep each answer and replace its question with:
  - **one from a different interview** (rand);
  - **nothing** (null);
  - or (control) keep the question and replace the **answer** with one contentless sentence (qonly).
- **Why not swap siblings?** Siblings share the turn and the answer, so a sibling's sub-question simply recreates
  the sibling's own row (true for 100% of dev). E24 already compares those directly.
- **H25a-1, why it is not obvious:** an answer paired with an unrelated question has, by definition, not replied
  to it. A relation-reading instrument should move labels away from Clear Reply. A style-reading one would not.
  Training never showed a mismatched pair.
- **Decision rule:** the Clear-Reply share falls by at least half, **and** at most 50% of labels persist.

### E25b (Act 4): question or speaker?
- **Why:** if evasion is relational, either side could drive it. LLMs have **no political stake**. If they evade
  on the same questions presidents evaded, part of evasion is question-driven.
- **Design:**
  - the 308 dev sub-questions (191 interview turns) are put to open LLMs;
  - decoding is greedy, at most 250 words of spoken prose;
  - every answer is labelled by the frozen instrument.
- **Conditions,** each differing from P in one factor:
  - **N**, neutral analyst ("you are not the President"): factor = stake;
  - **P**, the President, in the interview, on its date: the reference;
  - **F**, P's conversation, then "With respect, you haven't answered the question": factor = pursuit;
  - **B**, P with the President's own answer as private notes: factor = information.
- **Hypotheses:**
  - **H25b-1 (question-driven):** in N, an LLM evades more on questions the President evaded. Bar: odds ratio
    ≥ 1.5 with a CI above 1. **Control:** the same association measured on the contentless-answer frame must be
    smaller. Otherwise the instrument's own reading of the question, not the answerers, explains it.
  - **H25b-2 (stake):** P evades more than N, by ≥ 0.10.
  - **H25b-3 (pursuit):** of P answers that evade, F turns ≥ 50% into Clear Replies.
  - **H25b-4 (information, a test of the taxonomy):** with notes, LLMs give Clear Replies more often for questions
    whose President answer was *indirect but informative* (Implicit, General, Partial) than for *non-answers*
    (Dodging, Deflection, Declining, Claims ignorance), by ≥ 0.15. If the taxonomy's split is about information
    content, the notes contain an answer only for the former.

### E25c (Act 5a): when does evasion become visible?
- **Design:** label the first 25/50/75% of each answer.
- **H25c:** at 50%, at least 70% of labels agree with the full answer, and S2 drops by ≤ 0.05. If both hold,
  streaming detection works after half the answer.

### E25d (Act 5b): does the frozen model already encode it? (the fast path)
- **Design:** linear probes on each layer of Qwen3-8B-Base, with no fine-tuning.
- **H25d-1:** the slice-chosen layer reaches ≥ 0.45 S2 (one fine-tuned seed scores 0.525).
- **H25d-2:** the probe repeats its label on ≥ 30% of disjoint sibling pairs (M1: 19%). That would mean
  fine-tuning is what binds the label to the question.
- **Latency:** measured at 9/18/27/36 layers.

### Amendments, logged before any result was seen
1. **H100 instead of RTX PRO 6000:** no RTX PRO 6000 was free. The hard stop is set to 2.2 h (≤ ₹726).
2. **Control C1 failed as registered:** on the H100, M1's dev probabilities differ from the saved ones by up to
   0.2, though labels agree 98–99%. This is a bf16 numerics difference across GPU architectures. All E25
   contrasts are within this one run, so the check is recorded per seed, not used as a stop.
3. **Batch size and dropped frames:** batch size 64 for throughput, and the test prefix frames were dropped to
   fit the cap.
4. **Two subject models failed to load,** both from our configuration:
   - **OLMo-2** has a 4,096-token context, below the 8,192 we requested;
   - **Llama-3.1** got too little memory for its cache while the scorer held the GPU.

   E25b is a **two-family pilot (Mistral-7B, Qwen3-8B; 2,464 answers)**. Qwen shares a family with the 27B
   judge, so every result is also reported per model.
5. **The 27B failed to start on the H100:** 749 hybrid-cache blocks are fewer than vLLM's default of 1,024
   parallel sequences. It was re-run with at most 512.
6. **Probes cut to every 4th layer:** on CPU, all 37 layers would have overrun the cap.

### Results (collected 2026-10-30, 16:10 IST)

Full output: [`raw/E25_analysis.txt`](raw/E25_analysis.txt). Figures: [`figures/E25_audit.pdf`](figures/E25_audit.pdf)
and [`figures/E25_swap_prefix_probe.pdf`](figures/E25_swap_prefix_probe.pdf). Cost: ₹555; the instance destroyed
itself after uploading, and ₹230.59 of credit is left.

**Controls first.** Even with the H100's numerics, the instrument's dev predictions agree with E23's frozen ones on
98.1% of items. Candidate sets match on 96.1%, routing on 99.0%.

#### Act 3 (E25a): yes, the instrument reads the question causally

| Pairing (dev) | Clear-Reply share | Labels unchanged |
|---|---|---|
| own question | 0.253 | (reference) |
| **unrelated question** | **0.013** | 0.188 |
| no question | 0.640 | 0.354 |
| question kept, contentless answer | 0.006 | 0.143 |

- **H25a-1 held.** Pair an answer with someone else's question and it stops being a reply: 62 of the 78 Clear
  Replies become *Dodging*, exactly what the taxonomy says.
- **Replication:** the test set replicates the direction (0.338 → 0.110), more weakly.
- **No question at all:** the instrument calls most answers Explicit. There is nothing to evade.

**Reading:** the instrument models the relation, not answer style. That is the precondition for using it as a
measuring device.

#### Act 4 (E25b): question or speaker? Neither alone

Evasion rate (clarity not Clear Reply):

| Speaker | N neutral | P as President | F after follow-up | B with notes |
|---|---|---|---|---|
| Presidents (gold) | 0.744 | | | |
| Mistral-7B | 0.591 | 0.516 | 0.231 | 0.614 |
| Qwen3-8B | 0.506 | 0.519 | 0.172 | 0.481 |

- **H25b-1 failed.** A model with no stake does **not** evade more on the questions presidents evaded
  (OR 1.18 [0.80, 1.73]). On its own, the question does not drive evasion.
- **H25b-2 failed.** Playing the President does **not** raise the evasion rate (−0.03).
- **Exploratory, not pre-registered** (needs the two missing models before it can be claimed): in the President's
  role, the same models evade on **the same questions presidents evaded** (OR 2.74 [1.87, 4.17]; both models:
  Mistral 3.0, Qwen 2.5).
  - **The role changes *where* models evade, not *how often*.**
  - **Why it isn't an instrument artefact:** N and P are labelled by the same instrument on the same questions,
    so any bias the instrument has toward certain questions would affect both equally. Only the answers differ.
- **H25b-3 held.** Pursuit works on LLMs: 66% of evasive answers turn into clear replies when the journalist
  presses; only 5% go the other way.
- **H25b-4 narrowly failed.** With the President's own answer as notes, LLMs answer questions whose original
  answer was *indirect* (Implicit/General/Partial) more often than *non-answers* (Dodging/Deflection/
  Declining/Ignorance): +0.113 [−0.006, +0.225], against a +0.15 bar. Without notes the difference is −0.015, so
  the direction is as predicted but not established.
- **Profiles:** LLMs over-use Explicit (0.41–0.83 vs 0.28) and under-use Dodging (0.02–0.09 vs 0.18). Given the
  President's notes, their profile comes closest to his.
- **Control that failed to be informative:** the contentless-answer control is labelled Dodging for 98% of
  questions, so its odds ratio rests on 6 items. The N-vs-P comparison above is the stronger control.

**Reading:** evasion behaves like **question × role**. A question becomes evasion-prone when it is put to someone
who holds the President's accountability.

#### Act 5 (E25c/d): how early, how cheap?

- **H25c failed.** Half the answer agrees with the full-answer label only 66% of the time, and S2 drops by 0.15.
  Outright refusals show early (Declining 0.82, Claims ignorance 0.75 at 25%), but General and Dodging need the
  end of the answer. Live detection after half an answer is not viable.
- **H25d-1 failed and H25d-2 held.** The best frozen-layer probe (layer 28, chosen on the slice) scores 0.397,
  well below one fine-tuned seed (0.525). It also gives sibling sub-questions the same label 45% of the time,
  against 19% for M1. **Fine-tuning is what binds the label to the question.** There is no frozen fast path.
- **Latency (batch 1, H100):** the base 8B takes 48 ms at full depth and 28 ms at 18 layers. The LoRA timing was
  taken while the GPU was shared, so it is not reported.

### E26 (Act 5c): does the instrument survive lower precision?

- **Why:** the 10 M1 seeds are about 91% of the accuracy point's GPU cost, and E25d ruled out a frozen fast path.
  Numeric precision is the remaining lever.
- **Yardstick:** changing GPU alone flips about 2% of labels.
- **Design:** the frozen M1 in FP8 (8-bit weights and activations) against a 16-bit run on the same RTX PRO 6000,
  on 890 items × 10 seeds, plus the 27B for the full system. Pre-registered at 22:25 IST.

| Hypothesis | Result | Verdict |
|---|---|---|
| FP8 matches 16-bit (≥ 98% agreement, M1 within ±0.01) | 96.1% agreement; M1 −0.020 | failed |
| FP8 is ≥ 1.3× faster | 0.83× (activation quantization in unfused PyTorch ops) | failed |
| **The cascade absorbs the error** | 71% of FP8's label flips fall on items the 27B re-decides; **full system −0.003** vs M1 −0.020 | **held** |

- **Reading:** quantization error lands where M1 is unsure, which is exactly where the adjudicator already looks. The
  cascade makes the system robust to a cheaper M1, even though this FP8 implementation is not itself faster.
- **Speed:** a real speed-up needs fused FP8 kernels (future work).

**NF4** (4-bit, on the lab's RTX A4000s against a 16-bit run on the same cards; finished overnight):

| | FP8 (RTX PRO 6000) | NF4 (RTX A4000) |
|---|---|---|
| Labels agreeing with 16-bit (10-seed, dev) | 96.1% | 86.7% |
| M1 accuracy change | −0.020 | −0.034 |
| Full-system change | −0.003 | −0.010 (lower bound: 60 deferred items lacked 27B records) |
| Speed | 0.83× | 0.82× |

- **Neither format saves time here.** Each format's kernels cost more than they save: FP8's activation
  quantization, NF4's dequantization.
- **The QLoRA trigger is met for NF4,** so retraining the adapter on the 4-bit base is the named fix (future work).
- **The robust finding is the cascade:** the 27B absorbs most of FP8's error and part of NF4's. A cheaper first
  stage costs the system much less than it costs M1 alone.

---

## 11. The connected argument, start to finish

```text
 Mid-eval question: what limits a classifier here?
   └─ E0–E12: structure doesn't help ─► E13: knowledge does (one change, 10/10 seeds)
        └─ E14: the 8B ranks better than it decides (top-3 93%)
             └─ E15: so a 27B chooses among the top 4, only when unsure ─► E16: the 10 seeds are 93% of cost
                  └─ E17: distil them into one student ─► E18: two operating points
                       └─ E19–E23: which numbers survive? (honest CI, order averaging, artifacts, one 27B copy)
                            = THE INSTRUMENT (0.650 / 0.608)
 Literature: the instrument is not novel ─► what can it MEASURE?
   └─ E22e hint: relevance barely predicts errors ─► E24: what is the label a property of?
        └─ the label moves with the question (72% vs 43%); relevance ≠ answerhood (AUC 0.566)
             └─ E25a: does the instrument read the question causally?  YES (unrelated question → Dodging)
                  └─ E25b: is it the QUESTION or the SPEAKER?  NEITHER ALONE:
                       no stake → no link to presidents' evasions (OR 1.2); President's role → same questions
                       evaded (OR 2.7, exploratory) at the same rate; pressing works on LLMs (66%)
                       └─ E25c/d: how early, how cheap?  It needs most of the answer, and only fine-tuning
                          binds the label to the question: no frozen fast path
```

**The thesis after E25:** *evasion is a relation between a question and an answer, and which questions get evaded
depends on the role of the person answering.* The instrument reads that relation causally. Measuring it needs the
whole answer and a fine-tuned model.

Each arrow is a result that created the next question. That is the narrative the report follows.

---

## 12. What we got right, what we got wrong

**Right:**
- The "knowledge over structure" test (E13): one change, ten paired seeds, a clear answer.
- "Ranks better than decides" became a working cascade at a fraction of the winner's LLM calls.
- Reporting the honest bootstrap interval (E19) instead of the flattering one.
- Testing ideas on CPU before spending money (E24): five modules and one false headline were caught for free.

**Wrong, and corrected:**
- Several early experiments had predictable outcomes, and parameters changed between runs. That was fixed with
  the frozen instrument and registered predictions.
- We assumed classifiers ignore the question. The data said otherwise (E24 P2), so the claim was removed.
- E25 configuration errors (OLMo context length, vLLM memory, the 27B's sequence limit on the H100). Each is
  logged, and the first follow-up is to re-run the two missing models.

---

## 13. Limitations, each with the test that addresses it

| Limitation | What addresses it |
|---|---|
| The instrument was trained on presidential speech; LLM text may differ | Human annotation was dropped (out of scope). Instead, a label-free check (§G of `raw/E25_analysis.txt`): on LLM answers the instrument is *more* confident than on presidents' (uncertainty 0.46 vs 0.54, seed agreement 0.81 vs 0.73). That rules out "out of its depth", not confident errors. E25a also shows it reads the question causally. |
| Long, structured LLM answers may look "evasive" | Markdown removed from every answer; condition effects re-estimated with answer length as a covariate. |
| The instrument may label from the question alone | The question-only control (qonly) in E25b, plus an adjusted odds ratio. |
| Qwen3-8B shares a family with the 27B judge | Results reported per model; Mistral is the independent family. |
| Only two LLM families | The role pattern is reported as **exploratory**. Its registered replication with Llama-3.1 and OLMo-2 (fixes known: context 4,096; GPU to themselves; about 20 GPU-minutes) is the first item of future work. Experiments closed 2026-10-30. |
| An LLM's "I don't know" may be honest, not evasive | Condition B gives the facts; the paper version adds a factual briefing. |
| 308 dev items, small gains | Cost claims framed as non-inferiority; cluster bootstrap over interview turns; test-set replication of label-free measures. |
| GPU numerics differ across architectures | Every E25 contrast is within one run on one GPU; the control is reported per seed. |

---

## 14. Glossary

- **Any-annotator macro-F1:** a prediction counts if any annotator gave that label; F1 is averaged over the nine
  classes.
- **Siblings:** sub-questions split from one journalist turn that share the same answer.
- **M1:** the fine-tuned Qwen3-8B classifier (10 seeds).
- **M3:** the 27B LLM that picks among M1's top 4 labels.
- **Instrument:** M1 and M3 with every setting frozen (the E23 accuracy point).
- **Deferral (δ):** the share of items sent to the 27B, chosen by M1's uncertainty (δ = 0.3).
- **Logit adjustment (τ):** a correction for class frequency at decision time.
- **Odds ratio (OR):** how much more likely an LLM is to evade where the president evaded. OR = 1 means no link.
- **Cluster bootstrap:** resampling interview turns, not items, because sub-questions of one turn share an answer.
- **Pre-registration:** predictions and decision rules written down before a run, so results cannot be
  reinterpreted afterwards.
