# 04 — Scorer Semantics (Phase 1.2, partial: THE GATE FINDING)

**Status: STOP-AND-REPORT triggered.** P1.2 instructs: *"If that second property does not
hold, stop immediately and tell me, because contribution C4 depends on it."*

**The property does not hold.** Details below, with the empirical evidence.

Source read: `REF/semeval-2026-task6-camsr-cot/scripts/eval_competition.py`,
`_compute_macro_f1_multiref_official()` lines 329–378. Provenance caveat in `reports/01`:
this is TeleAI's replica of the Codabench scorer, the only implementation obtainable.
Probes: `prop_check.py`, `prop_check2.py` (scratchpad; to be promoted into
`tests/test_scorer_properties.py`).

---

## 1. Exact TP / FP / FN accounting

For each class `c`, one-vs-rest over items `i` with prediction `p_i` and reference set
`G_i` = deduplicated `{annotator1, annotator2, annotator3}`:

```
TP_c : p_i == c  and  c ∈ G_i
FP_c : p_i == c  and  c ∉ G_i
FN_c : p_i != c  and  c ∈ G_i  and  p_i ∉ G_i
```

Then `precision = TP/(TP+FP)`, `recall = TP/(TP+FN)`, `F1 = 2PR/(P+R)`, each guarded to
`0.0` on a zero denominator, and `macro_F1 = mean(F1_c over all 9 classes)`.

### The contested clause, resolved

`Implementation_Guide.md` §C.4 asserts: *"the whole set G triggers exactly **one** FN-type
penalty iff p∉G."*

The proposal §2 (G4) asserts the opposite: *"one outside is penalised **once for every
label** in the set."*

**The code says the proposal is right.** The FN condition is evaluated independently for
every class `c`, so when `p_i ∉ G_i` the guard `(not pred_hits_any_ref)` is true for *all*
`c ∈ G_i` simultaneously, producing `|G_i|` false negatives.

Measured, one item with `G = {Explicit, Implicit, Dodging}` and an out-of-set prediction:

```
per-class FN: Explicit=1, Implicit=1, Dodging=1
TOTAL FN = 3          -> one per member of G
```

TeleAI's Appendix A.1 prose is in fact *consistent* with the code ("a FN for a gold class
c∈G is only penalized if the prediction misses the entire reference set"). It was the
Implementation Guide's paraphrase of TeleAI that introduced the error.

**This strengthens the proposal's G4 argument rather than weakening it.** The claim that
"the cost of an error scales with how much the annotators disagreed" is literally true: miss
the set entirely on a 3-way-disagreement item and you take 3 FNs; miss on a unanimous item
and you take 1. **No rewrite of G4 is needed. Risk R2 is closed in the proposal's favour.**

---

## 2. Macro-averaging, zero support, and the 1/9 rule

- The average is over **all 9 declared classes, always** — `macro_f1 = sum(f1s)/len(f1s)`
  with `len(classes) == 9`. It is not restricted to classes present in gold or in
  predictions.
- A class with **zero support that is never predicted scores F1 = 0.0**, not NaN, and is
  **still included in the denominator**. Verified: perfect predictions on a corpus
  containing only `Explicit` yield `macro_f1 = 0.111111 = 1/9`.

**This is a large and underexploited fact.** Each class is worth 1/9 = 0.111 of the final
score regardless of how rare it is. Moving a rare class from "never correctly predicted"
(F1 = 0) to even modestly predicted is worth more than a large improvement on a frequent
class. It is the quantitative justification for the whole HiGrEC programme: C1's rare-leaf
focus and C4's threshold tuning are both attacking the highest-leverage part of the metric.

---

## 3. THE GATE: score is NOT invariant to which member of G is predicted

P1.2 predicted the score would be invariant to *which* member of `G` is predicted. **It is
not.** Three items with `G = {Explicit, Partial/half-answer}` plus filler:

```
predict 'Explicit'            on all 3 -> macro_f1 = 0.222222
predict 'Partial/half-answer' on all 3 -> macro_f1 = 0.333333
difference                              = 0.111111   (exactly 1/9)
```

Both predictions are inside `G`. Both are "correct" by the organizers' informal description
("a prediction is considered correct if it matches any annotator's label"). They differ by a
full ninth of the macro score, because the TP lands on a different class and macro-F1 weights
all nine classes equally.

### Why this does not kill C4 — it sharpens it

The playbook feared this because the Guide's derivation assumed that, given `p ∈ G`, the
choice among members was free, leaving `q(c|x) = P(c ∈ G | x)` as the only quantity that
mattered. That derivation was **under-specified**, not wrong:

- `q(c|x)` is still the right *estimand* — it is what determines whether a prediction lands
  in the set at all, which governs the FP and the `|G|`-sized FN penalty.
- But `q` alone does **not** determine the optimal choice, because among several classes with
  `c ∈ G` the macro-F1 payoff differs by class.
- **That is exactly the job of the per-class multipliers `λ_c`.** Under true invariance `λ_c`
  would have had almost nothing to do. Because invariance fails, `λ_c` is carrying real
  decision-theoretic weight, and the rule `ŷ = argmax_c λ_c · q(c|x)` is *more* motivated
  than the derivation that assumed invariance.

So the rule's **form survives intact**; what changes is the derivation in
`reports/13_c4_derivation.md`, which must now be written from the correct premise. C4 is not
blocked. It is better founded than when it was written.

Two supporting properties were also verified:

- **In-set dominance holds.** The worst in-set choice still beat the best out-of-set choice
  (0.222222 vs 0.207729). Predicting inside `G` is always at least as good, so the rule's
  first-order objective is sound.
- **Value concentrates on rare in-set members.** With `G = {Explicit, Clarification}` on one
  item in an Explicit-dominated corpus, predicting `Clarification` scored 0.222222 against
  `Explicit`'s 0.111111 — double, from a single item.

---

## 4. NEW: per-class support is endogenous

Not anticipated by the proposal, the guide, or the playbook. Because the FN condition carries
`(p_i ∉ G_i)`, an item where the model predicts *some other member* of `G_i` contributes
neither TP nor FN to class `c` — it **drops out of `c`'s support entirely**.

Five items, all with `G = {Explicit, Partial/half-answer}`:

```
predict Explicit             -> support(Explicit)=5   support(Partial/half)=0
predict Partial/half-answer  -> support(Explicit)=0   support(Partial/half)=5
predict Dodging              -> support(Explicit)=5   support(Partial/half)=5
```

**Support is a function of the predictions, not a fixed property of the gold data.** Two
consequences:

1. **Reporting.** CLAUDE.md invariant 3 requires every per-class F1 to travel with its
   support. On Subtask 2 that support figure is *system-dependent* and not comparable across
   systems. Every per-class table must therefore report gold-set frequency
   `#{i : c ∈ G_i}` — which is fixed — **alongside** the scorer's endogenous support, and
   must never present the latter as a property of the dataset.
2. **Threshold tuning (P6.1).** Coordinate ascent over `λ_c` is optimizing a metric whose
   per-class denominators move as the predictions move. Each 1-D search is still exact over
   the finite set of distinct `q`-values, but the objective is **not separable across
   classes** — changing `λ_c` alters other classes' supports. Convergence needs the
   random restarts the playbook already mandates, and the coordinate-ascent implementation
   must recompute the full macro-F1 per candidate rather than caching per-class terms.

---

## 5. Subtask 1

Plain single-label macro-F1 over `("Ambivalent", "Clear Non-Reply", "Clear Reply")`, via the
ordinary one-vs-rest path with singleton gold sets. No multi-reference behaviour.

## 6. Label-string handling

Normalization before comparison: whitespace collapsed, trailing `` .;:，, `` stripped,
lowercased for key matching, then mapped to a canonical spelling. `\/` is unescaped to `/`.
Aliases are accepted for `Partial/half-answer` spacing variants, and for
`Clear Non-Reply` / `Clear NonReply` / `Clear Non Reply` and `Ambivalent Reply`. Unrecognized
strings pass through normalized and simply never match a class — they silently become FPs
unless `--strict` is set. **Our reimplementation must replicate this exactly**, and the
conformance suite must include malformed and alias-spelled labels.

---

## Status and what P1.2 still owes

Done: scorer located and read; semantics stated; the three contested points resolved
(FN counting, macro denominator, invariance); one unanticipated property found.

Not yet done: `src/higrec/scoring/official.py` reimplementation; the 2000+ randomized
conformance tests; promotion of these probes into `tests/test_scorer_properties.py` with the
**corrected** property assertions — the invariance assertion the playbook specified must be
**inverted**: the test should assert that the score *does* depend on which member of `G` is
predicted, and pin the 1/9 magnitude, so that a future scorer change that silently restores
invariance is caught.
