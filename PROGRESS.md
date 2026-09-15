# HiGrEC — Progress

Living document. Updated whenever the state of the repository changes materially.
Structure follows `higrec-claude-code-playbook.md`; phase numbers below are the
playbook's phase numbers throughout.

**Last updated:** 2026-09-15
**Current stage:** Phase 2 (pre-registration), blocked on writing, not on compute
**Nothing in `src/` or `tests/` has been executed yet.**

---

## 1. Where the project stands

| Phase | Deliverable | State |
|---|---|---|
| 0.1 | Environment, scaffold, `CLAUDE.md` | **Done, verified** |
| 0.2 | Reference clones, inventory | **Done, verified** |
| 1.1 | Data layer, audit report, frozen splits | Library written; `audit_data.py` not written; **not run** |
| 1.2 | Scorer reimplementation + conformance | Library and tests written; **not run**. Semantics resolved. |
| 2.1 | Attribute code table | Library and tests written; **not run** |
| 2.2 | Krippendorff α partition test | Library and tests written; **pre-registration not written** |
| 3.1 | Encoder baselines | **No code** |
| 3.2 | LLM baseline | **No code** |
| 4.1 | Factorized head | Library and tests written; **not run** |
| 4.2 | Rare-leaf ablation | **No code** (needs 3.x first) |
| 5.1 | Annotator model | Library written; **no tests**; **not run** |
| 6.1 | Set-membership + thresholds | Library and tests written; **not run** |
| 6.2 | C4 controls | **No code** |
| 7–8 | C5 span, LLM panel | **No code** (venue upgrade) |
| 9 | Runner, tables, claim verification | **No code** |

Roughly 2,900 lines across 8 source modules and 6 test files. **Zero lines have
been executed.** Every claim about correctness below is a claim about what the
code is *intended* to do, not a verified fact. The first test run is expected to
surface real failures.

### Reports: 3 of 21 exist

Twelve missing reports are genuine outputs of running code and cannot exist
until their phase runs. **Five are not blocked at all and are the immediate
priority:**

| Report | Nature | Why it matters now |
|---|---|---|
| `02_teleai_label_definitions_analysis.md` | Reading `REF/` prompts | Independent evidence for or against C2's coverage-axis claim |
| `06_c2_preregistration.md` | Pure writing | **Gates the C2 run** |
| `10_c1_preregistration.md` | Pure writing | **Gates the C1 ablation** |
| `13_c4_derivation.md` | Pure writing | Must be redone from corrected scorer semantics |
| `14_c4_preregistration.md` | Pure writing | **Gates the C4 result** |

`04_scorer_semantics.md` exists but is partial: the semantics section is
complete, the conformance-results section awaits a test run.

**Process note, recorded because it cost time.** These five were treated as
blocked by the "do not run anything" instruction. They are not. A
pre-registration is pure prose written *before* a number exists — it is the gate
in front of the run, not an output of it. Per CLAUDE.md invariant 7, a
pre-registration written after seeing results is worse than none, so nothing may
run until 06, 10 and 14 are committed.

---

## 2. History: the findings that changed the plan

The project's direction has been altered four times by evidence, each recorded
here so the reasoning is not lost.

### 2.1 The multi-reference labels are public — C3 and C4 unblocked

`Implementation_Guide.md` Key Finding #1 states the public QEvasion release has
empty `annotator1/2/3` columns, and concludes that C3 and C4 cannot be built
without the organizers' unreleased scorer bundle. The playbook calls this the
one finding that would force a replan.

Measured directly on the HuggingFace parquet:

| split | rows | `evasion_label` | `annotator_id` | `annotator1/2/3` |
|---|---|---|---|---|
| train | 3,448 | 3448/3448 | 3448/3448 (ids 85, 86, 89) | **0/3448** |
| dev | 308 | 0/308 | 0/308 | **308/308 each** |

The guide's claim holds only for the *train* split. The 308-row split carries
three independent per-annotator labels, with distinct-labels-per-item of
40.6 / 48.7 / 10.7 % — matching TeleAI's published dev figures exactly, which
confirms the identification. **No replan needed.**

### 2.2 The splits are not what the guide assumed

Organizers' overview paper, Table 1: **train 3,448 / validation 308 / test 237**.
The guide calls the 308 items "the test set" throughout; they are the validation
set. The real test set is 237 items with only **two annotators per pair**.
`CLAUDE.md` invariant 8 pins this so no later task re-derives it wrong.

### 2.3 The scorer contradiction — resolved in the proposal's favour

Three sources disagreed about the false-negative rule. Resolved by reading the
implementation in `REF/semeval-2026-task6-camsr-cot/scripts/eval_competition.py`
and exercising it directly:

- An out-of-set prediction is charged **one FN per member of G**, not one for the
  set. Measured: `|G| = 3` out-of-set yields `TOTAL FN = 3`.
- **Our proposal's G4 was right**; the Implementation Guide's paraphrase
  ("exactly one FN-type penalty") was wrong. "The cost of an error scales with
  how much the annotators disagreed" is literally true. **G4 needs no rewrite.**

Two further properties, neither anticipated:

- **Macro-averaging is over all nine classes always.** A class absent from gold
  and predictions scores F1 = 0.0 and still counts in the denominator. Perfect
  predictions on a single-class corpus score exactly 1/9. **Every class is worth
  1/9 regardless of rarity** — the quantitative case for C1 and C4 both.
- **Per-class support is endogenous.** An item where the model predicts some
  other member of `G_i` drops out of class `c`'s support entirely, so support
  depends on predictions. Consequences: per-class tables must report fixed
  gold-set frequency alongside the scorer's support; and P6.1's coordinate ascent
  optimizes a **non-separable** objective, so it recomputes full macro-F1 per
  candidate instead of caching per-class counts.

### 2.4 The P1.2 gate fired: invariance is false

The playbook predicted the score is invariant to *which* member of `G` is
predicted, and said to stop if not. Measured:

```
G = {Explicit, Partial/half-answer}
predict 'Explicit'            -> macro_f1 = 0.222222
predict 'Partial/half-answer' -> macro_f1 = 0.333333   difference = exactly 1/9
```

**Invariance fails — and C4 is better founded as a result.** The earlier
derivation assumed that once inside `G` the choice was free, leaving `q(c|x)` as
the only quantity that mattered. `q` still governs whether you land in the set,
but among in-set classes the macro-F1 payoff differs by class, and that is
exactly what the per-class multipliers `λ_c` are for. Under true invariance `λ_c`
would have been nearly inert. The rule `argmax_c λ_c·q(c|x)` survives intact;
only `reports/13` must be written from the corrected premise.

The playbook's invariance test was **inverted rather than deleted**:
`test_score_is_NOT_invariant_to_which_reference_member_is_predicted` asserts the
dependence and pins it at 1/9, so a future change silently restoring invariance
is caught.

### 2.5 Smaller corrections to the guide

- **ChulaNLP's repo is public.** The guide reported it unretrievable and advised
  treating it as optional; it cloned without incident (13 notebooks).
- **The raw GitHub CSVs carry 11 labels**, including `2.9 Diffusion` and
  `2.5 Contradictory`, and 317 rows against 308 published. The HF parquet is
  clean. Canonical source is the parquet; the CSVs are cross-check only.
- **Prior art for C3 exists.** `SemEval2026_Task6_Duluth` contains
  `training/train_annotator_aware.py`. The proposal's G3 claim that "every
  surveyed system collapses this to a single gold label" may need qualifying.
  Phase 5 must read that file first.

---

## 3. What is happening now, and why

**Immediate task: write the five unblocked reports, starting with the three
pre-registrations.** No compute is involved and nothing may run before they are
committed.

The ordering is not arbitrary. Phase 2 is the highest-value, lowest-risk result
in the project: it needs no GPU and no annotation, because the coverage attribute
is a deterministic function of the existing leaf label, so the 308 items' three
judgments can be collapsed under two different 3-way cuts and compared. It also
de-risks the paper — if every model-based contribution disappoints, C2 still
stands.

The C2 test has real power rather than turning on a technicality. Under the
coverage cut exactly three leaves move — **Implicit** (Ambivalent → Clear Reply),
**Dodging** and **Deflection** (Ambivalent → Clear Non-Reply) — and those three
are **45.7 % of the training data**.

What C2 can and cannot show, to be stated in the paper and not left to inference:
it tests whether *relabelling existing 9-way judgments* under a different collapse
raises agreement. It does **not** test whether annotators asked *directly* about
coverage would agree more. That needs the re-annotation campaign.

---

## 4. Training estimates and the minimal plan

### 4.1 The distinction that matters

**C3 (the annotator model) is not the expensive item. Phase 3 (the baselines)
is.** C3 is a bias vector added to logits and costs essentially nothing, and
**C4 depends on it** — `q(c|x)` requires per-annotator probabilities. Cutting C3
to save time would remove C4, the contribution most likely to produce a real gain
at near-zero cost.

There is a cheaper route for C3 that costs **nothing at all**: estimate `q` from
a single posterior pushed through the **global 9×9 confusion matrix**, which is
obtained by counting the 308 dev triples. The marginal-bias model then becomes an
optional refinement rather than a prerequisite.

### 4.2 Cost of each contribution

| Item | GPU cost | Cuttable |
|---|---|---|
| C2 — α partition test | **0 h** | No — free and highest value |
| C3 — global confusion route | **0 h** (counting) | No |
| C4 — decision rule, post-hoc on saved probability vectors | **0 h** (CPU) | No — free |
| Flat baseline, DeBERTa-v3-large 512 truncate | 3.3 h | **No — it is half of C1** |
| C1 factorized, same backbone | 3.3 h | No |
| Parameter-matched capacity control | 3.3 h | No — reviewers require it |
| Random-code control | 3.3 h | No — the decisive test for C1 |
| Objective ablation (attribute-sum) | 3.3 h | Yes, at a cost |
| ModernBERT-large, 8k, no truncation | +33 h | **Yes** |
| Qwen3-14B LoRA bf16 | +75 h | **Yes** |

**Minimal complete paper: ~13–17 GPU-hours** (C1 + C2 + C3 + C4 with every
control), against the playbook's ~90.

Basis: DeBERTa-v3-large (435M) at 512 tokens, batch 16, 5 epochs over 3,448
examples ≈ 8 min/run on the measured card; 5-fold CV × 5 seeds = 25 runs per
configuration. **These are estimates, not measurements** — the dominant unknown
is achieved throughput on sm_120 under `transformers` 5.x with SDPA-only
attention, so a one-fold timed calibration run should replace this table before
any multi-day commitment.

### 4.3 Why published numbers cannot replace the flat baseline

Two distinct roles are easily conflated:

1. **Context** — "how does our system compare to published work?" Published
   numbers serve this fine, and Codabench now supplies a real one. **Zero cost.**
2. **C1's reference** — "does the attribute code help?" This must be *our* model:
   identical backbone, hyperparameters, seeds and data order, with the output
   parameterization as the only difference, asserted by config-hash diff (P4.1).

A delta against TeleAI's 0.68 measures our-backbone-versus-their-backbone,
confounded with scale, preprocessing and prompt engineering. The proposal already
commits to this: *"Published systems are context, not like-for-like
comparisons."* The endogenous-support finding sharpens it further — published
per-class numbers are not even commensurable across systems.

**The flat baseline cannot be skipped because it is half of the C1 experiment**:
the same training run, twice, once per output head. It can only be made cheap.

### 4.4 Recommended backbone change — open decision

Use **DeBERTa-v3-large at 512 tokens as the single backbone** for the flat
baseline, C1, and all controls.

- It is simultaneously the ChulaNLP parity control (playbook config 3.1b), so
  that comparison comes free.
- C1 is a claim about *output parameterization*, which is backbone-agnostic. A
  435M encoder tests it as validly as a 14B LoRA, and arguably supports the
  proposal's "gains from mechanism, not scale" argument **better**.

What this gives up, and it should be stated in limitations: the ModernBERT
comparison answering *"does removing 512-token truncation recover
Partial/half-answer?"* — a descriptive finding about prior work's limitation, not
a contribution — and C1 verified on a second backbone.

**Status: awaiting decision.** All code is written backbone-agnostic, so
deferring costs nothing.

---

## 5. Next steps, in order

1. **Write the three pre-registrations** (`06`, `10`, `14`) and commit them.
   Nothing runs before this. Also write `02` and redo `13` from the corrected
   scorer semantics.
2. **Run the test suite for the first time.** Expect failures. The most likely
   are the six fragile-pair tuples in `test_attributes.py`, which are hand
   arithmetic off the code table, and the conformance suite's 2,000 randomized
   configurations.
3. **Validate the scorer against Codabench.** Submit a trivial prediction file,
   compare the reported macro-F1 against our reimplementation's prediction for
   that same file. If they agree, risk R1 closes and our scorer is validated
   against the *real* official implementation rather than TeleAI's replica.
   **Zero training; highest value-per-effort action available.**
4. **Write and run `audit_data.py`** → `03_data_audit.md`, plus the frozen split
   artifact. Reproduces §2.1–2.2 as generated artifacts rather than prose, and
   adds token-length distributions.
5. **Run C2** → `05`, `07`. No GPU. Includes the three pre-declared sensitivity
   perturbations.
6. **Calibrate training throughput** with one timed fold before committing to any
   multi-hour sweep, replacing §4.2's estimates with measurements.
7. **Train the flat baseline**, freeze it, then C1 and its controls.
8. **Run C4 post-hoc** on the frozen baseline's saved probability vectors,
   reporting the nested-CV in-fold/out-of-fold gap *before* any headline number.

---

## 6. Open risks

| # | Risk | State |
|---|---|---|
| R1 | Scorer conformance targets TeleAI's replica; no independent implementation exists in any of the six cloned repos | **Closable now via Codabench** (step 3) |
| R2 | FN-accounting ambiguity | **Closed** — resolved in the proposal's favour (§2.3) |
| R3 | `transformers` 5.x API drift vs the playbook's 4.x assumptions | Open — smoke-test one step before any sweep |
| R4 | Shared GPU; `CUDA_VISIBLE_DEVICES=0`, 68 GB free at last check, another job resident | Open — enforce `--max-vram-gb`, fail fast |
| R5 | C2 turns on three leaves | Mitigated — they are 45.7 % of train; sensitivity perturbations pre-declared |
| R6 | Threshold overfitting on 308 items with single-digit rare-class support | Mitigated in code — nested CV, base-rate shrinkage, effective-examples-per-class reported beside every threshold |
| R7 | Code has never been executed | Open — step 2 |
| R8 | Duluth prior art may qualify the proposal's G3 novelty claim | Open — read before Phase 5 |
