# HiGrEC — Design Reasoning

Why the system is built the way it is. Each entry records the decision, the
context that forced it, the reasoning, and **what we give up** — because a design
record that only lists benefits is advocacy, not documentation.

**Division of labour with the other documents:**

- `PROGRESS.md` — *what* the state is: status, history, next steps. Changes often.
- `reasoning.md` (this file) — *why* it is that way. Changes when a decision changes.
- `IMPLEMENTATION_PLAN.md` — the phase-by-phase plan as first derived.
- `higrec/CLAUDE.md` — the invariants that follow from these decisions, in
  enforceable form.

**Last updated:** 2026-09-15

---

# Part I — What the scorer actually rewards

Everything downstream follows from three properties of the official Subtask 2
scorer. They were established by reading
`REF/semeval-2026-task6-camsr-cot/scripts/eval_competition.py` and exercising it
directly, not from any paper's prose. Full detail in
`higrec/reports/04_scorer_semantics.md`.

```
TP_c : p_i == c  and  c ∈ G_i
FP_c : p_i == c  and  c ∉ G_i
FN_c : p_i != c  and  c ∈ G_i  and  p_i ∉ G_i
```

### P1. An out-of-set prediction is charged one FN *per member* of G

The `p_i ∉ G_i` guard is evaluated independently for each class, so a prediction
that misses entirely fires it for every member of `G` at once. Measured: `|G| = 3`
with an out-of-set prediction yields 3 false negatives.

**This settles a contradiction between our own sources.** The proposal (§2, G4)
says an out-of-set prediction is "penalised once for every label in the set";
`Implementation_Guide.md` §C.4 says "the whole set G triggers exactly one FN-type
penalty". **The proposal is right and the guide is wrong.** So the proposal's
claim that *"the cost of an error scales with how much the annotators
disagreed"* is literally, arithmetically true, and G4 needs no rewrite.

### P2. Macro-averaging is over all nine classes, always

`macro_f1 = sum(f1s) / len(f1s)` with `len(classes) == 9`, unconditionally. A
class absent from both gold and predictions contributes `F1 = 0.0` and still
occupies a slot in the denominator — `0/0` is defined as `0.0`, not skipped, not
NaN. Perfect predictions on a corpus containing only `Explicit` score exactly
`1/9 = 0.111111`.

**Consequence: every class is worth 1/9 of the final score regardless of how rare
it is.** Moving a rare class from never-correctly-predicted (F1 = 0) to modestly
predicted is worth more than a large improvement on a frequent class. This is the
quantitative justification for the entire project: C1's rare-leaf focus and C4's
threshold tuning both attack the highest-leverage part of the metric.

### P3. Per-class support is endogenous

Because the FN condition carries `p_i ∉ G_i`, an item where the model predicts
*some other member* of `G_i` contributes neither TP nor FN to class `c` and drops
out of `c`'s support entirely. Measured on five items all with
`G = {Explicit, Partial/half-answer}`:

```
predict Explicit             -> support(Explicit)=5   support(Partial/half)=0
predict Partial/half-answer  -> support(Explicit)=0   support(Partial/half)=5
predict Dodging              -> support(Explicit)=5   support(Partial/half)=5
```

**Support is a function of the predictions, not a property of the gold data.**
Two consequences propagate into the code (see D-08, D-09).

### P4. The score is *not* invariant to which member of G is predicted

```
G = {Explicit, Partial/half-answer}
predict 'Explicit'            -> macro_f1 = 0.222222
predict 'Partial/half-answer' -> macro_f1 = 0.333333    difference = exactly 1/9
```

The playbook predicted invariance and said to stop if it failed, because C4
depends on it. **It fails, and C4 is better founded as a result.** The earlier
derivation assumed that once inside `G` the choice was free, leaving `q(c|x)` as
the only quantity that mattered. `q` still governs whether a prediction lands in
the set at all — but among in-set classes the macro-F1 payoff differs by class,
and that is precisely the job of the per-class multipliers `λ_c`. Under true
invariance `λ_c` would have been nearly inert. The rule
`ŷ = argmax_c λ_c · q(c|x)` survives intact; only the derivation changes.

In-set dominance *does* hold (worst in-set choice beat best out-of-set choice),
so the rule's first-order objective is sound.

---

# Part II — Experimental design

## D-01. Phase 3 is four jobs, not one

**Context.** Phase 3 as written consumes ~90 GPU-hours, the bulk of the project's
compute, and was accepted as a single indivisible block.

**Reasoning.** It is not indivisible. It bundles four jobs with different costs
and different necessity:

| # | Job | Config | Supports | Load-bearing? |
|---|---|---|---|---|
| 1 | Frozen flat baseline | 3.2, 3.1b | C1's delta; C4's control 3 | **Yes — irreducible** |
| 2 | ChulaNLP parity replication | 3.1b | Credibility | Free if 3.1b is our backbone |
| 3 | Truncation diagnostic | 3.1a vs 3.1b vs 3.1c | *Why* prior work lost Partial/half-answer | No |
| 4 | "Moderate-size LLM" framing | 3.2 | Proposal §5: mechanism not scale; single-pass cost | No |

Job 1 is not a baseline in the courtesy sense — **C1 is defined as a delta
against it**, so it is half the experiment and cannot be replaced by any
published number. Jobs 3 and 4 are the expensive ones and support no
contribution's claim.

**Status:** accepted; drives D-02.

## D-02. Freeze the trunk, train only the heads (two-tier design)

**Decision.** Tier 1: one forward pass per backbone, cache hidden
representations, train every head variant on the cached features. Tier 2: one
confirmatory full fine-tune of the single decisive comparison, on the backbone
Tier 1 identifies as mattering.

**Reasoning.**

1. **C1's claim is purely about the output parameterization.** Nothing in it
   requires the backbone to be fine-tuned.
2. **It makes P4.1's control requirement exact rather than approximate.** The
   playbook demands "identical backbone, identical hyperparameters, identical
   seeds, identical data order — only the output parameterization differs,"
   asserted by config-hash diff. Under a frozen trunk this is not asserted, it is
   *structurally true*: every variant reads the same cached tensor. The config
   hash becomes a redundant check rather than the only guarantee.
3. **The economics invert.** Fine-tuning costs ~25 training runs *per backbone*.
   Frozen-trunk costs *one forward pass per backbone*, after which each head
   variant trains in seconds. Adding a second and third backbone becomes nearly
   free — so **we do not have to choose between encoder and LLM at all.**
4. **Expensive compute is spent after we know where to point it**, not before.

**Estimated cost** (arithmetic, not measurement — see D-03):

| Step | Estimate |
|---|---|
| Feature extraction, DeBERTa-v3-large @ 512 | ~2 min |
| Feature extraction, ModernBERT-large, full length | ~15 min |
| Feature extraction, Qwen3-8B/14B last hidden state | ~30–60 min |
| All head variants × folds × seeds × 3 backbones | minutes |
| **Tier 1 total — multi-backbone C1, every control** | **~1–2 h** |
| Tier 2 — confirmatory fine-tune, flat vs factorized only | ~6–7 h |

Against ~90 h for Phase 3 as written.

**What we lose.** Stated plainly:

- **Absolute scores will be lower.** A frozen trunk is not adapted to the task.
  Tier-1 numbers are for *comparing output heads*, never for comparing against
  published systems.
- **The sample-efficiency argument is about end-to-end learning.** A reviewer can
  fairly say C1's rare-leaf claim concerns what the model *learns*, and a frozen
  trunk tests a different regime. **This is what Tier 2 exists to answer**, and
  it is why Tier 2 is not optional if the headline claim is to stand.
- **Head-only training may understate or overstate the effect.** Direction
  unknown in advance; it must be reported as a Tier-1/Tier-2 comparison rather
  than assumed to agree.

**Status:** proposed, awaiting decision.

## D-03. Training estimates are arithmetic, not measurement

Every duration in this document derives from FLOP counts and an assumed ~100–150
TFLOPS effective throughput. The dominant unknown is achieved throughput on
**sm_120 under `transformers` 5.x with SDPA-only attention** (no FlashAttention
installed). Confidence: **±2×**.

**Decision.** A one-fold timed calibration run replaces this table with measured
numbers before any multi-hour commitment is made. No schedule is promised on
these figures.

## D-04. What we lose by dropping the dedicated encoder+LLM fine-tunes

Recorded in full so the trade is visible rather than buried:

1. **The saturation objection — the most serious.** The proposal argues encoders
   are saturated on this task. A reviewer can then say: *"You showed the code
   helps a model that was already stuck; perhaps it compensates for encoder
   weakness and evaporates on a stronger backbone."* The random-code control
   partly answers this by isolating structure from regularization at any
   backbone, but not fully. **D-02 largely neutralizes this** by keeping an LLM
   backbone in Tier 1 at near-zero cost.
2. **Single-backbone generalization.** P4.2 asks for both backbones deliberately.
   Also **neutralized by D-02.**
3. **The cost-comparison claim weakens.** "Our 435M encoder is cheaper than
   DeepSeek-V3" is true but unsurprising; "a 14B single-pass model matches a
   3-stage frontier pipeline" is a real result. **Not recovered** unless Tier 2
   uses the LLM.
4. **Leaderboard competitiveness.** DeBERTa-large alone scored ~0.46 dev S2;
   ChulaNLP's full hybrid 0.61, TeleAI 0.68. An encoder-only system likely lands
   ~0.45–0.55. **Not recovered** without a fine-tuned LLM.

**Counter-consideration on (4):** C4 is precisely the mechanism that lifts a
mediocre model's score, because it optimizes what the scorer rewards rather than
the consensus label — at zero training cost. A frozen baseline *plus C4* may
submit substantially better than the baseline alone. This is worth measuring
before concluding an encoder cannot compete.

**Counter-consideration on the framing:** the proposal's logic for a moderate LLM
is "so gains are attributable to the mechanism rather than scale." By that logic
a **435M encoder is more moderate than a 14B LLM** — the argument generalizes in
our favour, not against. The framing in proposal §5 would need rewording, not
abandoning.

## D-05. C3 is cheap and load-bearing; Phase 3 is the expensive thing

**Context.** C3 was flagged as a training-cost concern.

**Reasoning.** C3 (the annotator model) is a bias vector added to logits. It is
not the expensive item — Phase 3 is. More importantly **C4 depends on C3**:
`q(c|x) = 1 − Π_a (1 − P(annotator a labels c | x))` requires per-annotator
probabilities. Cutting C3 to save time removes C4, the contribution most likely
to produce a real gain at near-zero cost.

**Cheaper route, zero training.** Estimate `q` from a single posterior pushed
through the **global 9×9 confusion matrix**, obtained by counting the 308 dev
triples. The marginal-bias model then becomes an optional refinement rather than
a prerequisite.

**Identifiability limits — part of the contribution, not a caveat.**

| Quantity | Estimable? | From what |
|---|---|---|
| Per-annotator marginal-bias vector `b_a ∈ R⁹` | **Yes** | train split; random assignment makes marginal differences annotator effects |
| Single global 9×9 confusion matrix | **Yes** | the 308 dev triples |
| Per-annotator confusion *matrices* | **No** | 72 free parameters each; dev carries no annotator IDs, so a judgment cannot even be attributed |

The effects are large enough to be worth modelling: annotator 86 assigns
`Explicit` 36.7 % of the time against annotator 85's 24.2 %; annotator 85 assigns
`General` 15.8 % against ~9 % for the others.

## D-06. Published results cannot replace the flat baseline

Two roles are easily conflated, and one genuinely is free:

1. **Context** — "how does our system compare to published work?" Published
   numbers serve this, and Codabench now supplies a real one. **Zero cost.**
2. **C1's reference** — "does the attribute code help?" Must be *our* model, same
   backbone, same seeds, same data order, only the output head differing.

A delta against TeleAI's 0.68 measures our-backbone-versus-their-backbone,
confounded with scale, preprocessing and prompt engineering. The proposal already
commits to this: *"Published systems are context, not like-for-like
comparisons."* **P3 sharpens it further** — since per-class support is
endogenous, published per-class numbers are not even commensurable across
systems.

**The flat baseline is unavoidable because it is half of the C1 experiment**: the
same training run, twice, once per output head. It can only be made cheap.

## D-07. Two further savings

**Sequence length.** "8192, no truncation" is overkill if p95 is ~1,200 tokens;
attention cost is superlinear in the long tail. Truncating at p95 rather than the
maximum captures nearly all the benefit at a fraction of the cost. **The data
audit supplies this number and has not been run** — a concrete reason to run
`audit_data.py` early.

**The 25× protocol multiplier.** 5-fold CV already yields 5 estimates; going to
25 does not grow effective sample size proportionally, because folds share data.
5 folds × 2 seeds captures most of the variance information. Spend the full
5 × 5 only on the headline flat-versus-factorized comparison.

---

# Part III — Implementation decisions

## D-08. Coordinate ascent recomputes full macro-F1 per candidate

**Forced by P3.** The obvious optimization — cache per-class TP/FP/FN and update
incrementally — is **wrong here**, because changing `λ_c` alters *other* classes'
supports. Cached counts go stale and the search converges somewhere worse. It
fails silently: no exception, just a poorer optimum.

The 1-D search is still exact: candidate `λ_c` values are the finite set of
argmax-crossing points, `O(N)` per class, not a grid.

## D-09. Per-class tables report gold-set frequency alongside scorer support

**Forced by P3.** The scorer's `support` is system-dependent and not comparable
across systems. `gold_set_frequency()` — `#{i : c ∈ G_i}` — is fixed and is the
number that belongs in any cross-system comparison. CLAUDE.md invariant 3 requires
support beside every per-class F1; this decision says *which* support.

## D-10. The invariance test was inverted, not deleted

P1.2 specified an assertion that the score is invariant to which member of `G` is
predicted. It is false (P4). Rather than remove the test,
`test_score_is_NOT_invariant_to_which_reference_member_is_predicted` asserts the
*dependence* and pins its magnitude at exactly 1/9, with a failure message
directing the reader to re-derive `reports/13`.

**Reasoning.** A future scorer change that silently restored invariance would
quietly undermine C4's justification. Deleting the test would leave nothing
watching that assumption.

## D-11. `random_code_index` permutes rows of the real table

**Context.** P4.2's code-corruption control is the decisive test for C1: if a
random but still-unique code performs as well as the semantic one, the mechanism
is regularization rather than structure, and C1 as argued is dead.

**Decision.** Permute the *rows* of the real code matrix rather than sampling a
fresh random table.

**Reasoning.** Row permutation preserves uniqueness of all nine codes *and* every
per-attribute value distribution exactly, discarding only the semantic content of
the assignment. A freshly sampled table would also change attribute balance,
confounding the control with a change in the very property (attribute-value
support) that C1's mechanism depends on.

## D-12. The factorized head is a drop-in for `nn.Linear(hidden, 9)`

Same input, same output shape. The cleanest available guarantee that the
factorized and flat systems differ in exactly the output parameterization.

**Design note on the capacity control.** The factorized head has *more*
parameters than flat: 5 heads totalling 3+3+2+2+3 = 13 outputs against flat's 9.
At `hidden = 1024` that is 13,325 versus 9,225 — a 4,100-parameter excess.
**This is exactly why the parameter-matched control exists**, and it is not
optional: without it, any C1 gain is attributable to extra capacity rather than
to shared structure.

## D-13. `OOV_INDEX = -1` with an all-False one-hot row

Not defensive tidiness — required for scorer conformance. In the reference
implementation an unrecognized prediction string matches no class, so it earns no
TP and no FP, **but it fails the "prediction hits some reference" guard and
therefore charges a false negative to every member of `G`**. An all-False one-hot
row reproduces that behaviour exactly, and `prediction_hits_reference` computed as
`any(pred_onehot & gold_mask)` handles it without a special case.

## D-14. Krippendorff's α is the headline; Fleiss' κ is reported alongside

α handles the incomplete and variable-rater design and missing judgments; κ
discards any item with a missing rating. An item rated by two of three annotators
carries real agreement information, and κ throws the whole item away. κ is
reported only for comparability with the organizers' published figures
(κ = 0.64 three-way, κ = 0.48 nine-way).

**Validation requirement.** The α implementation is hand-rolled, so it is checked
against the `krippendorff` package on the canonical worked example *before* being
trusted on real data. A quietly-wrong agreement statistic would produce a
confident, publishable, **false** C2 result, and nothing downstream would catch
it.

## D-15. The C2 bootstrap is paired on the same resampled items

Both partitions are recomputed on the *same* resampled items each iteration. The
two α values come from the very same annotator judgments, so an unpaired test
would badly overstate the variance of their difference. A test asserts that
identical partitions produce *exactly* zero CI width — if the bootstrap resampled
independently, identical partitions would still yield a spurious interval, which
is the bug that would manufacture a C2 result out of nothing.

## D-16. Sensitivity perturbations are declared in the module

The three arguable coverage assignments (`Partial/half-answer`, `General`,
`Deflection`) are declared as `COVERAGE_PERTURBATIONS` in
`data/attributes.py`, not chosen at analysis time. Perturbations selected after
seeing the α result are not a sensitivity analysis.

## D-17. Judgment calls flagged for review

- **`nested_cv_thresholds` averages fold λ geometrically.** These are
  multiplicative quantities and an arithmetic mean is biased upward by large
  values. Defensible, but a choice made without consultation.
- **`oracle_upper_bound` is greedy, not optimal.** It picks the currently-rarest
  in-set member per item, most-constrained items first — an *achievable lower
  bound* on the true oracle. The true optimum is an assignment problem not worth
  solving for a diagnostic. Must be reported as a lower bound.

---

# Part IV — Methodological discipline

## D-18. `REF/` is reference-only and never imported

HiGrEC's claim is that *structural inductive bias* drives the gain. If a
competitor's prompt scaffolding leaks into `src/`, the gain is confounded with
their prompt engineering — the first thing a reviewer will attack.

`REF/` is gitignored, absent from the Python path, and excluded from packaging
and pytest discovery. The conformance suite loads the reference scorer **by
explicit file path via `importlib`**, never by extending `sys.path`, so the
dependency cannot leak into library code; if the clone is absent the module skips
rather than fails.

The single sanctioned exception is distilling span supervision in Phase 7, which
must be logged in `reports/provenance.md` and disclosed in the paper.

## D-19. Pre-registration is binding

Where the playbook requires a pre-registration (C1, C2, C4), that file is written
and committed **before** the corresponding number is computed. A pre-registration
written after seeing results is worse than none, because it misrepresents the
process.

**Process failure recorded.** The pre-registrations were initially treated as
blocked by a "do not run anything" instruction. They are not — pre-registration
is pure prose written *before* a number exists, the gate in front of the run
rather than an output of it. Reports `06`, `10` and `14` must be committed before
anything runs.

## D-20. Canonical data source is the HuggingFace parquet

The organizers' GitHub CSVs are cross-check only. They carry **eleven** label
values — including `2.9 Diffusion` and `2.5 Contradictory`, outside the SemEval
taxonomy — and 317 rows against 308 published. The HF parquet is already cleaned.

Split naming is corrected throughout: the HF split published as `test` is the
shared task's **validation** set (308 rows, three annotators). The real test set
is 237 rows with **two** annotators per pair and is not on HuggingFace.

## D-21. Scorer conformance targets a replica — disclosed, and closable

No independent implementation of the multi-reference rule exists in any of the
six cloned repositories; every other participant collapses multi-reference gold
to a majority vote before scoring. Our conformance suite therefore targets
TeleAI's replica of the Codabench scorer — first-rate provenance, but
second-hand.

**Closable now that Codabench is open:** submit a trivial prediction file and
compare the reported macro-F1 against our reimplementation's prediction for that
same file. Agreement validates our scorer against the *real* official
implementation. Zero training; the highest value-per-effort action available.

---

# Open decisions

| # | Decision | Options | Blocking |
|---|---|---|---|
| O-1 | Adopt the two-tier frozen-trunk design (D-02) | Yes (~1–2 h + optional 6–7 h) / keep Phase 3 as written (~90 h) | Phase 3 |
| O-2 | Tier-2 backbone | DeBERTa-v3-large / ModernBERT-large / Qwen3 | Tier 2 only |
| O-3 | Is a competitive Codabench submission a goal? | If yes, argues for keeping the Qwen3 fine-tune in Tier 2 | O-2 |
| O-4 | Is proposal §5's "primary system is a LoRA-tuned LLM" framing committed with the supervisor? | Frozen-trunk changes the wording, not the argument | O-1 |
| O-5 | Sequence-length cap | p95 versus full 8k — needs the data audit | D-07 |
