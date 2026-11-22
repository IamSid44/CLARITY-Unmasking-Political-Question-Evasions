# Findings: what we did, why, and what it showed

The project as one argument, from the first classifier to the last experiment. Every number has a source: the
pre-registered [`experiment_log.md`](experiment_log.md), the raw outputs in [`raw/`](raw/), and the map in
[`results_sources.md`](results_sources.md). All scores are on the official dev set (308 items).

1. [The task](#1-the-task)
2. [The story in one paragraph](#2-the-story-in-one-paragraph)
3. [Part I. Building the instrument (E0–E23)](#3-part-i-building-the-instrument-e0e23)
4. [How we worked after the mid-evaluation](#4-how-we-worked-after-the-mid-evaluation)
5. [Part II. What the instrument measures (E24–E25)](#5-part-ii-what-the-instrument-measures-e24e25)
6. [Part III. Can the instrument be made cheaper? (E26)](#6-part-iii-can-the-instrument-be-made-cheaper-e26)
   - [Part IV. Closing the loop (E27–E32)](#6b-part-iv-closing-the-loop-e27e32)
   - [Part V. The registered confirmation (E33–E36)](#6c-part-v-the-registered-confirmation-e33e36)
7. [The argument, start to finish](#7-the-argument-start-to-finish)
8. [Limitations and future work](#8-limitations-and-future-work)
9. [Glossary](#9-glossary)

---

## 1. The task

**SemEval-2026 Task 6 (CLARITY).** Each item is an exchange from a US presidential interview: the journalist's
full turn (median 62 tokens), one **sub-question** split out of it (15 tokens), and the President's whole
**answer** (266 tokens). The label says how the answer responds to *that sub-question*:

| Clarity level (Subtask 1) | Evasion types (Subtask 2) |
|---|---|
| Clear Reply | Explicit |
| Ambivalent | Implicit, Dodging, General, Deflection, Partial |
| Clear Non-Reply | Declining to answer, Claims ignorance, Clarification |

**Data.** Train 3,448 rows (one label each); dev 308 (three annotators); test 237 (two annotators, labels not
released). Annotators agree only moderately (κ 0.48 on the nine types).

**Metric.** Macro-F1 in which a prediction is correct if it matches **any** annotator; the nine classes count
equally. What this scorer rewards, and what it implies for decision rules, is worked out in
[`scorer_geometry.md`](scorer_geometry.md).

## 2. The story in one paragraph

A large pretrained model, fine-tuned simply, beats every architectural change on a smaller one. It also *ranks*
the right label better than it *picks* it, so a larger model adjudicates the uncertain cases. That cascade is
our **instrument** (dev S2 0.650). The cascade itself is not novel (other shared-task teams built similar
pipelines), so we asked what it can *measure*. The data shows that **evasion is a relation between a question
and an answer**: one answer gets different labels for different sub-questions, far more often than annotator
noise explains. The instrument reads that relation causally. Putting the Presidents' questions to language
models, which have no political stake, shows that **which questions get evaded depends on the role of the
person answering**, not on the question alone. Finally, quantizing the instrument's first stage shows that the
adjudicator absorbs most of the resulting error. A registered confirmation on 248 interviews the discovery never
saw held at a smaller size (ratio of odds ratios 1.25 against 2.27 on dev), and a press secretary and an unnamed
President show it as well: what matters is speaking for the administration, not being the President.

---

## 3. Part I. Building the instrument (E0–E23)

### What limits a fine-tuned classifier here? (E0–E13, to the mid-evaluation)

Four explanations predicted different things: label noise (nothing helps), the decision rule (a metric-aware
rule helps), missing knowledge (a larger pretrained model helps), or labels not learnable from 3,448 examples
(definitions help). On DeBERTa-v3-large we changed one thing at a time against a named control on paired seeds:

| Change | Result |
|---|---|
| Logit adjustment (correct for class frequency at decision time) | helped |
| Longer training (E10), the full journalist question (E8b → E10) | helped |
| Hierarchical heads (E9), rebalanced losses (E8), specialist and boundary experts (E12b/c), a re-ranker (E6), model soups (E12d) | failed |

Best encoder system: 0.405 S2 (10 seeds); single model 0.384.

**The decisive test (E13).** Changing **only** the backbone to Qwen3-8B-Base + LoRA, with the same data, input,
loss and selection rule, gained **+0.092 per model (10 of 10 seeds up)**: more than all the encoder-side changes
combined. 10-seed system: 0.543 S2 / 0.746 S1. *Knowledge over structure.*

### The cascade (E14–E18)

| Exp. | Question | Result | What it led to |
|---|---|---|---|
| E14 (M1) | Is the 8B under-trained at 3 epochs? | 12 epochs: +0.049 per model; 10-seed system **0.598**. An acceptable label is first for 62% of items (after logit adjustment) but **in the top 3 for 92.5%** (the 3-epoch E13 model: 60% / 93%) | The model ranks better than it decides → let a stronger reader choose among the top few |
| E15 (M3) | Does a 27B LLM choosing among M1's top 4, for the least confident 30%, beat M1? | **0.669** at 0.36 LLM calls per item (the winning team used 1.94) | The cascade works; how robust is the gain? |
| E16 | Prompt parts, measured cost | Full prompt kept (differences within noise); the 10 M1 seeds are ~93% of GPU cost on the lab card (74% of the frozen 4-order instrument's one-GPU time, `raw/cost_same_gpu.txt`) | Replace the 10 seeds |
| E17 (M4) | Can one student trained on 0.5·gold + 0.5·the 27B's labels replace them? | Single model 0.573 vs 0.525; **student + 27B: 0.608 with four option orders (0.67× M1's one-GPU time), 0.599 with one order at 0.25×** | The cost operating point |
| E18 | Freeze | Both points frozen; ensembles of students add nothing (no diversity) | |

### Which numbers survive scrutiny? (E19–E23)

| Exp. | Question | Result | What it changed |
|---|---|---|---|
| E19 | Is the 27B's gain robust? | With the fusion refitted inside each bootstrap: **+0.037 [−0.044, +0.120]**. Confidence routing still beats random routing (0.669 vs 0.604, lab records) | We report the honest interval |
| E19 | Is the LLM alone enough? | The 27B over all 9 labels: 0.591 < M1's 0.598 | Neither model alone suffices |
| E19 | Does option order matter? | The 27B picks option A 40% of the time in one order, 22% in another; averaging 4 orders helps (+0.017, within noise: SE ≈ 0.027) | Order averaging adopted |
| E20 | Can a learned router beat confidence? | It lost at 4 of 4 budgets; the LLM only helps where M1 is uncertain | Keep confidence routing |
| E21 | Do test transcripts differ? | Test answers carry artifacts ([inaudible], crosstalk); injected into dev they change 15.6% of predictions | Test files use normalized answers |
| E22 | Caching, an escalation ladder, question–answer relevance | Caching changes answers (0.906 agreement); the ladder misses its target; relevance barely predicts errors (ρ +0.095) | Relevance returns in E24 |
| E23 | The final system on one copy of the 27B (two copies agreed on only 92.5%) | **Accuracy point 0.650 / 0.788; cost point 0.608 / 0.785** | This is the **frozen instrument** |

---

## 4. How we worked after the mid-evaluation

The mid-evaluation feedback said the experiments were disjoint, some gave the expected result, parameters
changed between runs, and some limitations could not be acted on. Each point became a rule:

| Feedback | Rule adopted |
|---|---|
| Disjoint experiments | Every experiment answers a question raised by the previous one (§7 shows the chain) |
| Expected results | Before a run, write the prediction **and why the outcome is not obvious**; ideas testable on existing outputs are tested first, on CPU (five were dropped this way, §5) |
| Changing parameters | One **frozen instrument** for all later experiments, every setting fixed in `code/experiments/e25/frozen_system.json`; deviations are logged as time-stamped amendments before results are seen |
| Limitations that cannot be acted on | Every limitation is paired with the test that addresses it (§8) |

A review of the shared task's 30 system papers also showed that the cascade's parts were already in use
(candidate generation plus an LLM chooser, confidence routing, distillation), while **none** tested whether
systems read the question or used the taxonomy to measure anything. So the cascade became an **instrument**,
and the contribution moved to what it can measure.

---

## 5. Part II. What the instrument measures (E24–E25)

### E24: evasion is a relation (CPU only, existing outputs; `raw/E24_prechecks.txt`)

- **The data is relational.** One answer often serves several sub-questions (**siblings**): 69% of train rows
  share their answer. Of 3,079 sibling pairs, **61.5% carry different evasion labels**.
- **It is not annotator noise.** On dev, two labels from different annotators differ **43%** of the time on the
  same item, **72%** on sibling items and 83% on random items. The label moves because the question moves.
- **Our model reads the question.** On dev sibling pairs with disjoint gold sets, M1 repeats its label only
  **19%** of the time, so "models ignore the question" was removed from the plan before it became a headline.
- **Relevance is not answerhood.** Political science scores answers by how specific they are to their question. Computed with the
  8B model, the log-likelihood gain under the own question (PMI_c) is ordered across labels (p = 2e-12) but barely
  discriminative (**AUC 0.566** for Clear Reply; identifiability itself, in the Alvarez & Morrier sense, 0.549): Deflection and Partial answers are as identifiable as Explicit
  ones. An answer can be about the question without answering it.

**Five candidate modules were dropped on CPU** before any GPU spend:

| Idea | Result | Verdict |
|---|---|---|
| Route on epistemic uncertainty (seed disagreement) | caught 49 of 115 M1 errors vs 53 for plain entropy | dropped |
| Decode siblings jointly | 0.24 bits shared; gains ±0.01 that flip sign | not established |
| A parameter-free metric-optimal decision rule | 0.594 untuned vs 0.598 tuned | no gain |
| One debiased option order instead of four | agreement with the 4-order average fell (0.85 → 0.79) | dropped |
| Conformal candidate sets as router | coverage held (0.977) but S2 0.564–0.617 vs 0.641–0.663 | dropped |

### E25: is evasion in the question or the answerer? (registered before the run; `raw/E25_analysis.txt`)

Every label is the frozen E23 instrument's. Applied to dev with E23's records it reproduces E23 exactly; on the
H100 used for E25 it agrees with E23 on 98.1% of items. Nothing is refitted.

**E25a: does the instrument read the question causally?** Keep each answer, change only the question:

| Pairing (dev) | Clear-Reply share | Labels unchanged |
|---|---|---|
| own question | 0.253 | — |
| **question from another interview** | **0.013** | 0.188 |
| no question | 0.640 | 0.354 |
| question kept, contentless answer | 0.006 | 0.143 |

**H25a-1 held.** Paired with someone else's question, an answer stops being a reply: 62 of the 78 Clear Replies
become *Dodging*. The test set replicates the direction (0.338 → 0.110). The instrument models the relation,
not answer style, which is the precondition for using it to measure anything.

**E25b: question or speaker?** The 308 dev sub-questions were put to two open LLMs (Mistral-7B, Qwen3-8B; 2,464
answers) in four conditions, each differing from P in one factor: **N** neutral analyst (no stake), **P** the
President of the United States in an interview (a generic persona: dev has no names or dates, and the prompts
printed "nan"; see §6b), **F** P followed by "you haven't answered the question" (pursuit), **B** P with the
President's own answer as private notes (information). Evasion rate (not Clear Reply):

| Speaker | N | P | F | B |
|---|---|---|---|---|
| Presidents (gold) | 0.744 | | | |
| Mistral-7B | 0.591 | 0.516 | 0.231 | 0.614 |
| Qwen3-8B | 0.506 | 0.519 | 0.172 | 0.481 |

- **H25b-1 failed:** a model with no stake does **not** evade more on the questions the Presidents evaded
  (odds ratio 1.18 [0.80, 1.73]). The question alone does not drive evasion.
- **H25b-2 failed:** playing the President does not raise the evasion *rate* (−0.03).
- **Exploratory:** in the President's role, the same models evade on **the same questions the Presidents
  evaded** (OR 2.74 [1.87, 4.17]; Mistral 3.0, Qwen 2.5). N and P are labelled by the same instrument on the
  same questions, so an instrument bias toward certain questions would affect both equally. **The role changes
  where models evade, not how often.**
- **H25b-3 held:** pursuit works on LLMs: 66% of evasive answers become clear replies when pressed; 5% go the
  other way.
- **H25b-4 narrowly failed:** with notes, LLMs answer questions whose original answer was *indirect*
  (Implicit/General/Partial) more often than *non-answers*: +0.113 [−0.006, +0.225] against a +0.15 bar
  (−0.015 without notes).

**Reading:** evasion behaves like **question × role**. A question becomes evasion-prone when it is put to
someone who holds the President's accountability.

**E25c/d: how early, how cheap?**
- **H25c failed:** half an answer agrees with the full-answer label only 66% of the time (S2 −0.15). Refusals
  show early; General and Dodging need the end of the answer.
- **H25d-1 failed, H25d-2 held:** linear probes on the frozen base model reach 0.397 S2 (one fine-tuned seed:
  0.525) and give sibling sub-questions the same label 45% of the time (M1: 19%). **Fine-tuning is what binds
  the label to the question**; there is no frozen fast path.

**Validity check on LLM text (label-free, §G).** On LLM answers the instrument is *more* confident than on the
Presidents' (uncertainty 0.46 vs 0.54; seed agreement 0.81 vs 0.73): it is not out of its depth.

---

## 6. Part III. Can the instrument be made cheaper? (E26)

The 10 M1 seeds are ~91% of the accuracy point's GPU cost, and E25d ruled out a frozen fast path, so numeric
precision is the remaining lever. Each format was compared with a 16-bit run of the same models **on the same
GPU type** (changing GPU alone flips ~2% of labels). Registered before the run; `raw/E26_analysis.txt`.

| | FP8 W8A8 (RTX PRO 6000) | NF4 (RTX A4000) |
|---|---|---|
| Dev labels agreeing with 16-bit (10-seed) | 96.1% | 86.7% |
| M1 system S2 | −0.020 | −0.034 |
| Full system S2 | **−0.003** | −0.010 (exploratory; lower bound) |
| M1's flips on items the 27B re-decides | **71%** of 17 | 49% of 45 |
| Throughput vs 16-bit | 0.83× | 0.82× |

- **H26-1 (fidelity) and H26-2 (≥ 1.3× speed) failed; H26-4 held:** FP8's errors land where M1 is unsure,
  which is exactly where the adjudicator already looks, so the full system loses only 0.003.
- **Neither format is faster as implemented:** FP8's activations are quantized in unfused PyTorch ops, NF4
  dequantizes on the fly. NF4's speed comparison is also across batch sizes (16-bit at batch 8 split over two
  cards, NF4 at batch 16 on one).
- **NF4 meets the registered QLoRA trigger** (> 0.02 S2), so retraining the adapter on the 4-bit base is the
  named fix.

**Reading:** the adjudicator makes the system robust to a degraded first stage: a cheaper M1 costs the system
far less than it costs M1 alone.

---

## 6b. Part IV. Closing the loop (E27–E32)

Three preliminary checks written after E25 pointed at two modules and one error mode, and the role effect was still
unregistered. E27–E32 were registered together (2026-11-20, 23:19 IST) before any of them ran, and they leave the frozen
instrument unchanged. They feed the workshop paper (`report/paper/`).

| Exp. | Question | Result | Verdict |
|---|---|---|---|
| E27 | Does subtracting M1's no-question view (contrastive, `log p_own − α log p_null`) help? | S2 is highest at α = 0 and falls to 0.347 at α = 1; nested: −0.017 [−0.046, −0.002] (M1), −0.006 (cascade); sibling repetition not lowered | **failed**: the answer alone carries part of the label |
| E28 | Do systems give a sub-question its sibling's label ("contamination")? | Every system (DeBERTa, M1, student, 27B alone, cascade) is **below** a length-matched permutation null (p ≥ 0.77); the preliminary 32.9% vs 18.2% was label frequency | **failed**: the classifiers separate siblings |
| E29 | Does the P-role evasion share (M5) help on deferred items? | Weight 0 in every fold; any positive weight lowers S2/S1 even in sample | **failed**: not adopted |
| E30 | The E25a frames as a one-command validity battery, run on other models | M1 0.179 → 0.010 and the instrument 0.253 → 0.013 under a swap (both pass); question binding (same label on disjoint siblings): probe 0.448, instrument 0.224, M1 0.155; DeBERTa (3 seeds, control 99.4–99.7%) 0.312 → 0.026, binding 0.310; the 27B zero-shot over 9 labels 0.377 → 0.026, binding 0.362 | **all pass the swap** (necessary, not sufficient); binding orders probe > zero-shot 27B > DeBERTa > instrument > M1 |
| E31 | Is the P-role odds ratio robust? | 2.72 (length strata), 2.81 (+ question prior), 2.89 (without the 17 "nan" answers), 3.49 (President by the instrument); N stays at 1.15–1.18 | **held** (still exploratory) |
| E32 | A third family (OLMo-2-7B), registered as confirmatory | Hardware control 97.0% agreement. OR(P) **2.90** [1.81, 5.04] (held); three families pooled 2.81 [1.97, 4.16] (held); OR(N) 1.37 [0.83, 2.29] misses the registered ≤ 1.3 (failed). Generic wording: N 1.13, P 2.74 | **mixed**: stays exploratory |

**A defect found on the way.** The dev split has no `president` or `date` (train has both), and the E25b prompts
printed them as "nan" ("You are nan, President of the United States. It is nan, …"). The role in E25b is therefore a
generic U.S. President. Only 17 of 2,464 answers repeat the token, and excluding them leaves the effect unchanged (E31).
E32 runs the identical prompts (protocol A, confirmatory) and a corrected generic wording (protocol B, exploratory).

---

## 6c. Part V. The registered confirmation (E33–E36)

The role effect had only ever been seen on dev, the sample in which it was noticed. After an external review of the
first paper draft, it was moved to a registered test on interviews the discovery never saw. E33–E36 were registered on
2026-11-21 between 11:47 and 11:55 IST, before any answer was generated (`experiment_log.md`, E33–E36). They feed
the workshop paper (`report/paper/`, MultiDems @ EACL 2027).

**Design.**

| | Discovery (exploratory) | Confirmation (registered) |
|---|---|---|
| Sample | dev: 308 sub-questions, 56 interviews | the 248 train interviews not in dev: 2,858 sub-questions, 1,710 turns |
| President evaded | 0.744 | 0.688 (one annotator) |
| Roles | N analyst (without the President's name), P President ("nan" persona, §6b) | **N** analyst told the President's name and date, **S** White House press secretary, **G** unnamed President without a date, **P** named President on the date |
| Models | Mistral-7B, Qwen3-8B, OLMo-2-7B | the same three (registered pool) + Granite-3.3-8B and OLMo-2-32B (secondary); 57,160 answers, greedy, ≤ 400 tokens |
| Instrument | the frozen shared-task instrument | **CF**: Qwen3-8B LoRA cross-fitted in 5 folds grouped by interview (a train question is labelled by the model that never saw its interview; τ = 0.50 frozen). **ZS**: Qwen3.8-27B zero-shot over the nine labels, never trained on the task |
| Estimate | ROR P/N **2.27** [1.61, 3.22], computed before registration | registered below |

**ROR (ratio of odds ratios)** = OR(role) / OR(analyst), where OR is the odds of an LLM evading where the President
evaded against where he answered. It is the role × President-evaded interaction in a logistic model with
model × role intercepts and model × President-evaded slopes; intervals are interview bootstraps (2,000 draws for the
registered tests). Anything about a question that makes both the President and an LLM evade it cancels in the ratio,
in expectation. Noise in the President's label pulls the ratio towards 1.

### E33: the instrument (all registered checks passed)

| Check | Result | Verdict |
|---|---|---|
| C33: no interview in two folds; per-fold dev S2; `infer_cf` reproduces each fold's validation probabilities | disjoint; 0.577, 0.574, 0.564, 0.542, 0.480 (mean 0.547); argmax agreement 1.000 in every fold | **passed**: the contingency (ZS as primary) was not triggered |
| H33-1: question swap on the dev LLM answers | Clear-Reply share N 0.418 → 0.000, P 0.530 → 0.000 | **held** |
| H33-2: CF and ZS agree on "evades" equally across roles (analyst–President gap ≤ 0.10) | N 0.730, P 0.742, S 0.730, G 0.717 | **held**: no sign that labels are noisier in one role |

### E34: the confirmatory audit (`raw/E34_confirmation_train.txt`)

| Hypothesis | Result | Verdict |
|---|---|---|
| **H34-1** ROR P/N > 1, three models pooled, and > 1 in at least 2 of 3 | **1.25 [1.09, 1.43]**, p 0.001; Mistral 1.44 [1.21, 1.72], OLMo-2-7B 1.34 [1.09, 1.62], Qwen3-8B 1.01 [0.82, 1.24] | **held** (≥ 1.5 was reported, not a gate: no) |
| H34-2 ROR S/N > 1 (press secretary) | 1.34 [1.19, 1.49] | **held** |
| H34-3 ROR P/S ≠ 1 (does the named person add anything?) | 0.93 [0.85, 1.03] | failed: no difference detected |
| H34-4 ROR G/N > 1 (unnamed President, no date) | 1.22 [1.08, 1.38] | **held** |
| H34-5 ROR P/N > 1 in each added model | Granite-3.3-8B 1.19 [0.98, 1.45], OLMo-2-32B 1.08 [0.89, 1.31] | failed (both) |
| H34-6 ROR P/N > 1 on sub-questions that do not address "you" | 1.07 [0.90, 1.29] | failed |

Secondaries are Holm-corrected. **Robustness** (reported, no gate): all 13 variants give 1.22–1.39 with every lower
bound above 1, among them ZS as the instrument (1.22 [1.06, 1.42]), GEE, conditional logit (1.35), without Qwen3-8B
(1.39), soft labels, answer-length terciles, annotator-specific slopes, the non-answer outcome, without meta answers
and full train. Symmetric extra noise on 25% of the analyst's labels would be needed to bring the ratio to 1. The
analysis reruns byte-identically.

**How the role acts.** In role, models evade *less* (P 0.47–0.50, G 0.45–0.47 against N 0.56–0.62), and the drop is
larger on the questions the Presidents answered (Mistral 0.537 → 0.410) than on those they evaded (0.572 → 0.536).
The role makes models answer where the President answered more than it makes them evade where he evaded.

### E35: binding (`raw/E35_binding.txt`, `raw/E35x_exploratory.txt`)

- **Dev, with a matched null.** A classifier that ignores the question (answer-only anchor) labels disjoint siblings
  alike 4.5 times as often as unrelated pairs. Every classifier we use sits far below that: M1 1.31, the instrument
  1.66, DeBERTa 2.01, ZS 2.47 times its null. On dev, M1 and ZS differ by −0.207 [−0.379, −0.022] at the same S2
  (0.611 vs 0.604).
- **H35-1 failed** on the confirmation interviews (1,605 sibling pairs with different labels): CF 0.282, ZS 0.297,
  difference −0.015 [−0.104, +0.038], at S2 0.437 and 0.458.
- **E35x (post hoc):** CF's five single-fold models span 0.155–0.293 on dev, as wide as the spread between systems.
- **Reading:** sibling binding separates a classifier that reads the question from one that cannot. At these sample
  sizes it does not rank classifiers that all read it. The dev claim "binding is not accuracy" was dropped.

### E36: what drives it (exploratory; `raw/E36_drivers.txt`, `raw/E36x_you.txt`)

- **Not recall of the transcripts.** The unnamed President, given neither name nor date, shows the effect. Word
  overlap with the President's real answer is about 1.7 times that with another interview's answer in every role,
  the analyst's included (N 0.091 vs 0.053, P 0.129 vs 0.079). At most 0.1% of answers step out of role.
- **Questions put to "you".** ROR P/N is 1.43 [1.16, 1.75] on the 1,411 that address "you" and 1.07 on the rest
  (H34-6). On "you" questions the analyst evades whether or not the President did (62% vs 65%, OR 1.11), while every
  White House role separates them (OR 1.48–1.59). On the other questions the analyst already separates them
  (OR 1.35) and the roles do so to a similar degree. A paired test of the difference (E36x, added after the fact)
  supports it only for the named President: 1.34 [1.00, 1.76]; press secretary 1.15 [0.87, 1.49], unnamed President
  1.21 [0.92, 1.51]. **Suggestive, not established.**
- **Other subgroups** (multiple-question turns, affirmative, hypothetical, future wording, each President) give
  0.75–1.38 with wide intervals for the small ones; none is claimable.

### What it adds up to

| Claim | Status |
|---|---|
| Speaking for the White House moves LLM evasions towards the questions Presidents evaded | **confirmed** (registered, held-out interviews), ROR ≈ 1.2–1.3, two instruments |
| It is the position, not the person | **confirmed** for the press secretary and an unnamed President; the named President adds nothing detectable |
| It holds in every model | **no**: clear in Mistral-7B and OLMo-2-7B; Qwen3-8B, Granite-3.3-8B and OLMo-2-32B do not confirm it as the named President |
| The effect is as large as on dev | **no**: 2.27 on dev, 1.25 held out, about a quarter on the log scale. On dev, CF gives 2.03 [1.42, 3.17], so the change of instrument does not explain the drop; sample and prompt changes are confounded |
| Sibling binding ranks classifiers | **no** (H35-1 failed); it catches a question-blind classifier |
| The effect sits on "you" questions | **suggestive**, exploratory |

---

## 7. The argument, start to finish

```text
 What limits a classifier here?
   └─ E0–E12: structure does not help ─► E13: knowledge does (one change, 10/10 seeds)
        └─ E14: the 8B ranks better than it decides (top-3 92.5%)
             └─ E15: a 27B chooses among the top 4, only when unsure ─► E16: the 10 seeds dominate cost
                  └─ E17: distil them into one student ─► E18: two operating points
                       └─ E19–E23: which numbers survive? = THE INSTRUMENT (0.650 / 0.608)
 The cascade is not novel ─► what can it MEASURE?
   └─ E22e: relevance barely predicts errors ─► E24: what is the label a property of?
        └─ the label moves with the question (72% vs 43%); relevance ≠ answerhood (AUC 0.566 / 0.549)
             └─ E25a: does the instrument read the question causally? YES
                  └─ E25b: question or speaker? NEITHER ALONE: no stake → no link (OR 1.2);
                     the President's role → the same questions evaded (OR 2.7, exploratory)
                       └─ E25c/d: it needs the whole answer and a fine-tuned model
                            └─ E26: can M1 be cheaper? Not faster as implemented, but the
                               adjudicator absorbs most of the precision error (−0.003)
                                 └─ E27–E32: every proposed module and error mode, tested: none adopted;
                                    the battery is reusable; the role effect survives every control
                                      └─ E33–E36: registered on 248 unseen interviews: the role effect HOLDS,
                                         smaller (ROR 1.25); press secretary and unnamed President too
                                         (the position, not the person); binding catches a question-blind
                                         classifier but does not rank good ones
```

**Thesis:** evasion is not a property of an answer but of a question–answer pair under a role. Classifiers can
be checked for this with counterfactual question interventions, and one that passes shows that who is asked changes
which questions get evaded. The instrument reads that relation causally; measuring it needs the whole
answer and a fine-tuned model, and the cascade keeps the measurement stable when its first stage is degraded.

---

## 8. Limitations and future work

| Limitation | What addresses it |
|---|---|
| The instrument was trained on presidential speech; LLM text may differ | Label-free checks: the instrument is more confident on LLM answers (§5); E25a and H33-1 show it reads the question on LLM text; two instruments with no shared training on the task agree equally in every role (H33-2) and give the same effect (E34). **No human labels on LLM text** (a decision for the paper) |
| Only two LLM families in E25b; Qwen3-8B shares a family with the 27B judge; a generic persona ("nan") | E34: five models from four families, named prompts on train; the effect is clear in two of the three registered models and in neither added model |
| The role effect was found on the sample it is reported on | E34: registered confirmation on 248 held-out interviews (1.25 against 2.27) |
| One greedy answer per question and role; US Presidents only | Not addressed; sampling and other countries may change the effect |
| The instrument may label from the question alone | The question-only control and an adjusted odds ratio; the N-vs-P comparison holds the instrument fixed |
| An LLM's "I don't know" may be honest | Condition B supplies the facts |
| 308 dev items, small gains | Nested CV for every fitted parameter; cluster bootstrap over interview turns; test-set replication of label-free measures |
| GPU numerics differ across architectures | Every E25 contrast is within one run on one GPU; every E26 comparison is on one GPU type |
| Reduced precision was not faster | Fused FP8 kernels and QLoRA retraining for NF4 |

## 9. Glossary

- **Any-annotator macro-F1:** a prediction counts if any annotator gave that label; F1 averaged over classes.
- **Siblings:** sub-questions split from one journalist turn that share the same answer.
- **M1 / M3 / M4:** the fine-tuned Qwen3-8B (10 seeds) / the 27B adjudicator over M1's top 4 / the distilled
  student. **M2** is the confidence router.
- **Instrument:** M1 + M3 with every setting frozen (the E23 accuracy point).
- **Deferral δ:** the share of items sent to the 27B, the least confident first (δ = 0.3).
- **Logit adjustment τ:** dividing each class probability by its training frequency raised to τ.
- **Odds ratio (OR):** how much more likely an LLM is to evade where the President evaded; 1 means no link.
- **Ratio of odds ratios (ROR):** OR in a role divided by OR as the neutral analyst; what a role adds (E34).
- **CF / ZS:** the cross-fitted Qwen3-8B classifier (folds grouped by interview) and the 27B zero-shot classifier,
  the two instruments of E33–E36.
- **Discovery / confirmation sample:** dev (exploratory) / the 248 train interviews not in dev (registered).
- **Cluster bootstrap:** resampling interview turns, not items, because a turn's sub-questions share an answer.
- **Pre-registration:** predictions and decision rules written before a run, so results cannot be reinterpreted.
