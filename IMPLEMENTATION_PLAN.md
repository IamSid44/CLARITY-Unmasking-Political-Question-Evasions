# HiGrEC — Understanding and Implementation Plan

Written after reading: the proposal (`Nier_ANLP-Proposal.pdf`), the QEvasion dataset paper
(Thomas et al., Findings of EMNLP 2024), the SemEval-2026 Task 6 overview
(`Task_Results_Paper.pdf`), TeleAI's 1st-place system paper, ChulaNLP/moswisarut's 2nd-place
system paper, `Implementation_Guide.md`, and `higrec-claude-code-playbook.md`; plus direct
empirical probes of the QEvasion release and the organizers' GitHub repository.

---

## 1. What the project is

SemEval-2026 Task 6 (CLARITY) classifies a political interview answer against a
sub-question on two levels:

- **Subtask 1 (clarity, 3-way):** Clear Reply / Ambivalent / Clear Non-Reply.
- **Subtask 2 (evasion, 9-way):** the leaf taxonomy of Thomas et al. (2024).

The shared task ended with Subtask 1 pushed well past baseline (0.89 vs 0.82) but Subtask 2
stalled at 0.68 — the winning system barely exceeding the organizers' fine-tuned Llama-70b
baseline of 0.57, with a 15-team band separated by less than seed noise.

HiGrEC's thesis is that this ceiling is a **measurement-design** problem, not a capacity
problem, and it makes four specific claims (G1–G4) with five contributions (C1–C5):

| Gap | Claim | Contribution |
|---|---|---|
| G1 | Flat 9-way output discards taxonomy structure; rare leaves cannot borrow strength | **C1** 5-attribute factorized code |
| G2 | The coarse cut is made across the wrong axis (clarity, not coverage) | **C2** Krippendorff α partition test; **C5** coverage/span module |
| G3 | Annotator identity and disagreement are thrown away | **C3** annotator model (marginal bias / low-rank confusion) |
| G4 | The scorer is multi-reference but everyone optimizes the consensus label | **C4** set-membership decision rule with per-class thresholds |

The unifying methodological commitment — and the thing that makes this a submission rather
than a course paper — is that **every contribution ships with a control designed to kill it**,
and every prediction is pre-registered in a committed file before the number is computed.

---

## 2. What I verified empirically (and what it changes)

I probed the actual data and repositories rather than trusting the guide's prose. Five
findings change the plan. Two of them correct `Implementation_Guide.md`.

### F1 — CORRECTION: multi-reference per-annotator labels ARE public. C3 and C4 are unblocked.

`Implementation_Guide.md` Key Finding #1 states the public QEvasion release has empty
`annotator1/2/3` columns and that C3/C4 are therefore blocked pending the official scorer
release. **This is true only of the train split.** Measured on the HuggingFace parquet files:

| split | rows | `evasion_label` | `annotator_id` | `annotator1/2/3` | `clarity_label` |
|---|---|---|---|---|---|
| train | 3,448 | 3448/3448 | 3448/3448 (3 ids: 85, 86, 89) | **0/3448** | 3448/3448 |
| test  | 308   | **0/308**   | 0/308                        | **308/308 each** | 308/308 |

The 308-row "test" split on HuggingFace is the **development/validation** split of the shared
task, and it carries three independent per-annotator leaf labels per item. Distinct-labels-
per-item is 40.6% / 48.7% / 10.7% — matching TeleAI's reported dev figures exactly, which
confirms the identification.

This is the finding the playbook says would force a replan if it came out negative
(P1.1: "if Phase 1.1 reveals that multi-reference labels are not obtainable, stop and
replan"). **It came out positive. No replan needed.**

Redundant confirmation: the organizers' dataset repo `konstantinosftw/Question-Evasion`
(MIT licensed, found via footnote 1 of the EMNLP 2024 paper) ships
`dataset/Inter-Annotator/{Annotator1,Annotator2,Annotator3}.csv` and a `test_set.csv` with
317 rows × explicit `Annotator1/2/3` columns.

### F2 — CORRECTION: the split structure is not what the guide assumes.

Per the organizers' overview paper, Table 1: **Training 3,448 / Validation 308 / Test
(Evaluation Phase) 237**. The guide repeatedly calls the 308 items "the test set". They are
the validation set. The real test set is 237 items and — critically — was annotated **with
only two annotators per pair**, "due to time and resource constraints".

Consequences: the multi-reference estimation base for C3/C4 is 308 items × 3 annotators
(the dev split), not a test set; and any claim about the evaluation set must account for
|G| ≤ 2 there rather than ≤ 3.

### F3 — The official 3-way partition is confirmed exactly from the data, not from prose.

Train `clarity_label` counts are Clear Reply 1052 / Ambivalent 2040 / Clear Non-Reply 356,
and 1052 = the Explicit count exactly, 356 = Declining(145) + Claims ignorance(119) +
Clarification(92) exactly. So:

- **Clear Reply** = {Explicit}
- **Ambivalent** = {Implicit, Partial/half-answer, General, Deflection, Dodging}
- **Clear Non-Reply** = {Declining to answer, Claims ignorance, Clarification}

Against C1's coverage-derived cut (full→Reply, partial→Ambivalent, none→Non-Reply), the
leaves that **move** are:

- **Implicit**: Ambivalent → Clear Reply
- **Dodging**: Ambivalent → Clear Non-Reply
- **Deflection**: Ambivalent → Clear Non-Reply

Those three leaves are 1,575 / 3,448 = **45.7%** of training data, so the C2 partition test
has real statistical power rather than turning on a rare-class technicality. This is the
answer P2.1 asks to "stop and report".

I also verified that dev `clarity_label` equals the majority vote of the three annotators'
leaf labels pushed through the official partition on 306/308 items, the remaining 2 being
three-way ties the expert adjudicated. The partition above is therefore the organizers'
actual mapping, not a reconstruction.

### F4 — NEW: the raw annotator files contain 11 labels, not 9.

The organizers' raw `Inter-Annotator/test_set.csv` contains `2.9 Diffusion` (9 occurrences
from Annotator2, 2 from Annotator3) and `2.5 Contradictory` (1 occurrence) on top of the
nine SemEval classes. The published HuggingFace dev split has these already cleaned to a
strict 9-class vocabulary, and 317 raw rows become 308.

Neither the guide nor the playbook mentions this. It matters because the C1 attribute table
covers exactly 9 leaves, so any pipeline that reads the GitHub CSVs rather than the HF
parquet must decide explicitly how Diffusion/Contradictory are dropped or mapped. **Plan:
use the HF parquet as the canonical source and treat the GitHub CSVs as a cross-check only,
recording the 317→308 delta in the data audit.**

### F5 — Annotator effects are large and directly measurable, so C3 is well-posed.

Per-annotator label marginals on the training split (`annotator_id` × `evasion_label`,
row-normalized) differ substantially:

| annotator | Explicit | General | Dodging | Deflection | Implicit |
|---|---|---|---|---|---|
| 85 | 0.242 | **0.158** | 0.242 | 0.069 | 0.140 |
| 86 | **0.367** | 0.094 | 0.165 | 0.126 | 0.121 |
| 89 | 0.295 | 0.086 | 0.214 | 0.134 | 0.169 |

Annotator 86 assigns Explicit 52% more often than annotator 85; annotator 85 assigns General
roughly 70% more often than either of the others. Since items were distributed at random,
these are annotator effects, not item effects — exactly the identifying assumption C3 needs.
The marginal-bias model has real signal to fit, and P5.1's validation step (learned biases
vs. measured marginals) has a concrete target to hit.

### F6 — Scorer semantics are contested between our own sources.

Three statements of the Subtask 2 metric, and they do not agree:

1. **Organizers' overview paper**, §4.2: macro-F1 over k classes, and "a prediction is
   considered correct if it matches any annotator's label." Underspecified — says nothing
   about FN accounting.
2. **TeleAI Appendix A.1**: TP for class c iff p=c and c∈G; **an FN is charged only when
   p∉G, once for the set**, not once per unpredicted member of G.
3. **Our own proposal**, §2 (G4): "one landing inside the annotator set incurs no penalty
   for the set members it did not predict, while one outside **is penalised once for every
   label in the set**."

(2) and (3) contradict each other on the FN count for an out-of-set prediction. That changes
per-class recall denominators and therefore changes the λ_c the C4 rule tunes. **C4's entire
derivation rests on resolving this from the actual scorer source, not from prose.** This is
precisely why P1.2 comes before any modelling, and it is the first blocking question of
Phase 1.

Note that the proposal's version is actually the stronger motivation for C4 (error cost
scales with disagreement), so if the code matches TeleAI's version we must rewrite the G4
argument in the paper accordingly. Either way we report what the code says.

### F7 — Environment reality check.

- GPU: **2 × NVIDIA RTX PRO 6000 Blackwell, 96 GB each**, compute capability **(12, 0) = sm_120**
  — exactly as the playbook predicts, so FA4 is out and FA2/FlashInfer is the target.
  GPU 0 has ~70 GB free; GPU 1 has ~7 GB free (a vLLM engine is resident). The card is
  genuinely shared, so the playbook's `--max-vram-gb` discipline is not theoretical.
- `/scratch/shlok/Temp/.venv`: Python **3.12.3** (playbook says 3.11 — harmless), torch
  **2.11.0+cu128** ✓, transformers **5.17.0**, datasets 5.0.1, numpy 2.5.2, pandas 3.0.5.
- **Missing and required:** `peft`, `bitsandbytes`, `accelerate`, `scikit-learn`, `scipy`,
  `krippendorff`, `statsmodels`, `pytest`, `sentencepiece`. The venv is not "everything
  installed" for this project; Phase 0 must install these.
- `transformers` is on the **5.x** line. The playbook was written against ≥4.48. v5 changed
  enough API surface that the encoder training code needs to be written against 5.x
  idioms and verified, not copied from 4.x examples.
- Disk: 909 GB free on `/scratch`.

---

## 3. The one thing that most needs saying about C1

The proposal is already honest about this and the plan must stay honest: because the
attribute code is a **deterministic function** of the leaf label, the factorized
parameterization has **identical expressive capacity** to a flat 9-way softmax in the
infinite-data limit. It is a reparameterization. Its only possible value is sample
efficiency on rare leaves, via attribute values shared with frequent leaves.

That makes two things mandatory rather than optional:

1. The **random-code control** (P4.2 step 5). If a randomly permuted but still-unique code
   performs as well as the semantic one, the mechanism is regularization, not structure, and
   C1 as argued is dead. This control is the whole ballgame.
2. The **pre-registered per-leaf direction predictions** (P4.2), committed before running.
   The account predicts gains on rare leaves with well-populated distinguishing attributes
   (Partial/half-answer at 2.3%, Clarification at 2.7%, Claims ignorance at 3.5%) and ~zero
   on the lexically-marked leaves. Gains landing on Explicit/Dodging instead would falsify it.

---

## 4. Implementation plan

Sequenced per `higrec-claude-code-playbook.md`, one stage at a time, each ending in a
stop-and-report. Deviations from the playbook are marked **[DEV]** with a reason.

### Phase 0 — Environment and references
- **P0.1** Scaffold `higrec/` (`src/higrec/{data,models,scoring,decision,annotators,analysis}`,
  `configs/`, `scripts/`, `tests/`, `REF/`, `runs/`, `reports/`); write `CLAUDE.md` with the
  six project invariants verbatim from the playbook; install the missing packages into
  `/scratch/shlok/Temp/.venv`; write and run `scripts/check_env.py`; emit
  `reports/00_environment.md`.
  **[DEV]** Use the existing Python 3.12 venv rather than creating a 3.11 one — torch
  2.11.0+cu128 is already correct for sm_120 and rebuilding risks a worse stack.
- **P0.2** Clone into `REF/` (read-only, never imported, never on the Python path):
  `konstantinosftw/Question-Evasion` (organizers' data repo — **the scorer hunt starts here**),
  `ther7777/semeval-2026-task6-camsr-cot` (TeleAI), `moswisarut/SemEval2026-Task6-moswisarut`
  (ChulaNLP — **confirmed to exist**, contrary to the guide's claim that it was unretrievable).
  Fallbacks for the official scorer, in order: the organizers' repo; the Codabench competition
  bundle; vendored copies inside participant repos (KCLarity, CLaC-Lab, Duluth and others all
  surfaced in search and commonly vendor the scorer). Write
  `reports/01_reference_inventory.md` and `reports/02_teleai_label_definitions_analysis.md`.

### Phase 1 — Data and scorer
- **P1.1** Data layer + `scripts/audit_data.py` → `reports/03_data_audit.md`. Must reproduce
  F1–F5 above as generated artifacts rather than prose, add the token-length distributions
  under both DeBERTa-v3 and ModernBERT tokenizers, and emit the frozen hashed split file.
- **P1.2** **The blocking stage.** Read the official scorer line by line; write
  `reports/04_scorer_semantics.md` resolving F6; reimplement in
  `src/higrec/scoring/official.py`; 2000+ randomized conformance tests against the official
  implementation; property tests. **Hard gate:** if the invariance property (score does not
  depend on *which* member of G is predicted) fails, C4's derivation is wrong and we stop and
  replan before touching a model.

### Phase 2 — C2 (no GPU, highest value, lowest risk)
- **P2.1** `src/higrec/data/attributes.py`: frozen version-stamped 9×5 code table; uniqueness
  and reachability tests; `reports/05_code_analysis.md` with the minimality analysis,
  attribute-value support table, and the stated capacity objection.
- **P2.2** Commit `reports/06_c2_preregistration.md` **first**, then run the Krippendorff α
  partition test on the 308 dev items (paired bootstrap, 10,000 resamples, p<0.05 two-sided),
  plus Fleiss κ, unanimity rates, raw 9-way α reference, per-class disagreement concentration,
  and the three-perturbation sensitivity analysis over the arguable coverage assignments
  (Partial/half-answer, General, Deflection). → `reports/07_c2_results.md`, negative result
  written up with equal care.

### Phase 3 — Baselines (frozen once trained)
- **P3.1** ModernBERT-large 8k (no truncation) / DeBERTa-v3-large 512 truncate / DeBERTa-v3-large
  512 drop. 5 seeds, CV-on-train primary, per-example probability vectors saved,
  peak-VRAM logged. → `reports/08_encoder_baselines.md`, answering whether removing truncation
  recovers Partial/half-answer.
- **P3.2** Qwen3-14B LoRA bf16, seq 3072, flat 9-way head over the final hidden state, plus the
  parameter-matched capacity control. → `reports/09_llm_baseline.md`. **Frozen** per CLAUDE.md.

### Phase 4 — C1
- **P4.1** Factorized head, both objectives (marginal primary, attribute-sum auxiliary),
  log-space renormalization over the 9 valid codes, config-hash assertion that nothing but the
  output head differs from the frozen baseline.
- **P4.2** Pre-register per-leaf directions → run factorized vs flat vs param-matched,
  5 seeds, 2 backbones, paired bootstrap per class, frequency-decile stratification, objective
  ablation, and **the random-code control**. → `reports/11_c1_results.md`, scored
  prediction-by-prediction.

### Phase 5 — C3
- Marginal-bias primary, low-rank confusion secondary, single **global** 9×9 confusion from the
  308 dev triples, learned-vs-measured marginal validation against F5, explicit identifiability
  section. → `reports/12_c3_results.md`.

### Phase 6 — C4
- **P6.1** Derivation from the *verified* scorer semantics (F6) →
  q(c|x) = 1 − Π_a(1 − P(annotator a labels c | x)); coordinate-ascent λ_c; **nested CV**
  with the in-fold/out-of-fold gap reported before anything else.
- **P6.2** Pre-register (gains on non-unanimous items and the two highest-disagreement classes;
  gains on unanimous items falsify), then run all four controls including the decision rule on
  the frozen Phase 3.2 baseline and the oracle upper bound, stratified by consensus level.
  **Note:** 33/308 dev items (10.7%) have no leaf majority, so the majority-vote control is
  undefined there and must be reported on the 275-item subset with the exclusion stated.

### Phase 7 — C5 (venue upgrade)
- Distilled span supervision with ≥2 model families and agreement filtering, validated against
  gold-implied coverage; sentence-relevance module; Circa transfer; **the shuffle control**;
  integration into the C1 attribute heads. Provenance logged in `reports/provenance.md`.

### Phase 8 — LLM annotator panel (venue upgrade, optional)
- Attribute code only, never leaf labels, never as a proxy for human disagreement.

### Phase 9 — Consolidation
- `scripts/run_all.py`, publication tables, reusable paired-bootstrap, label-shift importance
  weighting, `scripts/verify_claims.py`, and the findings summary with limitations.

**Course-deliverable critical path:** Phases 0, 1, 2, 3, 4, 6, 9. Phases 5, 7, 8 are the
venue upgrade.

---

## 5. Live risks

| # | Risk | Trigger | Response |
|---|---|---|---|
| R1 | Official scorer source not obtainable anywhere | P1.2 | Reimplement from the overview paper + TeleAI A.1, run the conformance suite against *both* candidate FN semantics, and report C4 under each. Disclose that the scorer was not executed. |
| R2 | FN-accounting ambiguity (F6) resolves against the proposal's version | P1.2 | Rewrite the G4 argument to match the code. C4 survives either way; only the magnitude of the predicted gain changes. |
| R3 | transformers 5.x API drift | P3.1 | Write against 5.x, verify on a 1-step smoke run before committing to 5-seed sweeps. |
| R4 | Shared GPU — GPU 1 is at 90/96 GB | P3.x | Pin to GPU 0, enforce `--max-vram-gb`, fail fast rather than OOM a shared card. |
| R5 | C2 turns on 3 leaves (Implicit, Dodging, Deflection) | P2.2 | Already mitigated by the mandated sensitivity analysis; those 3 leaves are 45.7% of train, so power is adequate. |
| R6 | Threshold overfitting on 308 items with single-digit rare-class support | P6.1 | Nested CV gap reported *before* any headline number; base-rate shrinkage; effective-tuning-examples-per-class reported next to every threshold. |
