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
adjudicator absorbs most of the resulting error.

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
| E14 (M1) | Is the 8B under-trained at 3 epochs? | 12 epochs: +0.049 per model; 10-seed system **0.598**. An acceptable label is first for 60% of items but **in the top 3 for 93%** | The model ranks better than it decides → let a stronger reader choose among the top few |
| E15 (M3) | Does a 27B LLM choosing among M1's top 4, for the least confident 30%, beat M1? | **0.669** at 0.36 LLM calls per item (the winning team used 1.94) | The cascade works; how robust is the gain? |
| E16 | Prompt parts, measured cost | Full prompt kept (differences within noise); the 10 M1 seeds are ~93% of GPU cost | Replace the 10 seeds |
| E17 (M4) | Can one student trained on 0.5·gold + 0.5·the 27B's labels replace them? | Single model 0.573 vs 0.525; **student + 27B: 0.608 at 0.27× M1's cost** | The cost operating point |
| E18 | Freeze | Both points frozen; ensembles of students add nothing (no diversity) | |

### Which numbers survive scrutiny? (E19–E23)

| Exp. | Question | Result | What it changed |
|---|---|---|---|
| E19 | Is the 27B's gain robust? | With the fusion refitted inside each bootstrap: **+0.037 [−0.044, +0.120]**. Confidence routing still beats random routing (0.669 vs 0.604) | We report the honest interval |
| E19 | Is the LLM alone enough? | The 27B over all 9 labels: 0.591 < M1's 0.598 | Neither model alone suffices |
| E19 | Does option order matter? | The 27B picks option A 40% of the time in one order, 22% in another; averaging 4 orders helps (+0.017) | Order averaging adopted |
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
- **Relevance is not answerhood.** Political science scores answers by *identifiability* (can the answer be
  matched to its own question?). Computed with the 8B model, it is ordered across labels (p = 2e-12) but barely
  discriminative (**AUC 0.566** for Clear Reply): Deflection and Partial answers are as identifiable as Explicit
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
President in the interview, **F** P followed by "you haven't answered the question" (pursuit), **B** P with the
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

## 7. The argument, start to finish

```text
 What limits a classifier here?
   └─ E0–E12: structure does not help ─► E13: knowledge does (one change, 10/10 seeds)
        └─ E14: the 8B ranks better than it decides (top-3 93%)
             └─ E15: a 27B chooses among the top 4, only when unsure ─► E16: the 10 seeds are 93% of cost
                  └─ E17: distil them into one student ─► E18: two operating points
                       └─ E19–E23: which numbers survive? = THE INSTRUMENT (0.650 / 0.608)
 The cascade is not novel ─► what can it MEASURE?
   └─ E22e: relevance barely predicts errors ─► E24: what is the label a property of?
        └─ the label moves with the question (72% vs 43%); relevance ≠ answerhood (AUC 0.566)
             └─ E25a: does the instrument read the question causally? YES
                  └─ E25b: question or speaker? NEITHER ALONE: no stake → no link (OR 1.2);
                     the President's role → the same questions evaded (OR 2.7, exploratory)
                       └─ E25c/d: it needs the whole answer and a fine-tuned model
                            └─ E26: can M1 be cheaper? Not faster as implemented, but the
                               adjudicator absorbs most of the precision error (−0.003)
```

**Thesis:** evasion is a relation between a question and an answer, and which questions get evaded depends on
the role of the person answering. The instrument reads that relation causally; measuring it needs the whole
answer and a fine-tuned model, and the cascade keeps the measurement stable when its first stage is degraded.

---

## 8. Limitations and future work

| Limitation | What addresses it |
|---|---|
| The instrument was trained on presidential speech; LLM text may differ | Label-free check: the instrument is more confident on LLM answers (§5); E25a shows it reads the question causally. Human annotation of LLM answers is future work |
| Only two LLM families in E25b; Qwen3-8B shares a family with the 27B judge | Results reported per model; the role effect is **exploratory**. Re-running Llama-3.1 and OLMo-2 (both failed on configuration: context length, GPU memory) is the first follow-up |
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
- **Cluster bootstrap:** resampling interview turns, not items, because a turn's sub-questions share an answer.
- **Pre-registration:** predictions and decision rules written before a run, so results cannot be reinterpreted.
