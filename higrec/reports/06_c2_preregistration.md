# 06 — C2 Pre-Registration

**Written and committed BEFORE computing any α value.** Per CLAUDE.md invariant 7,
a pre-registration written after seeing results is worse than none.

**Date:** 2026-09-16
**Status at time of writing:** `src/higrec/analysis/agreement.py` implemented and
its unit tests pass (87/87 suite green), but **it has not been run on QEvasion
data**. No α value for either partition is known to the author.

---

## 1. The claim under test

The proposal (G2) argues the official coarse taxonomy is **cut across the wrong
axis**: it partitions by *clarity* when the dimension along which annotators
actually agree is *coverage* — how much of the requested information arrived.

If true, collapsing the same nine-way judgments under a coverage-derived cut
should produce **higher inter-annotator agreement** than the official cut,
because the coverage cut places its boundaries where annotators are not
conflicted.

The two partitions:

| Leaf | Official | Coverage-derived |
|---|---|---|
| Explicit | Clear Reply | Clear Reply |
| **Implicit** | **Ambivalent** | **Clear Reply** |
| Partial/half-answer | Ambivalent | Ambivalent |
| General | Ambivalent | Ambivalent |
| **Dodging** | **Ambivalent** | **Clear Non-Reply** |
| **Deflection** | **Ambivalent** | **Clear Non-Reply** |
| Declining to answer | Clear Non-Reply | Clear Non-Reply |
| Claims ignorance | Clear Non-Reply | Clear Non-Reply |
| Clarification | Clear Non-Reply | Clear Non-Reply |

Exactly **three leaves move**: Implicit, Dodging, Deflection — together ~45.7% of
the training distribution, so the test is not decided by a rare-class
technicality.

## 2. Hypothesis

**H1:** Krippendorff's α computed over the annotators' labels collapsed under the
**coverage-derived** partition is **strictly greater** than α under the
**official clarity** partition.

**H0:** α(coverage) − α(official) ≤ 0.

## 3. Test procedure, fixed in advance

- **Data:** the 308 multi-reference items, each with three independent annotator
  leaf labels. Primary and only confirmatory sample.
- **Unit of analysis:** the item. Each annotator's own nine-way label is
  collapsed **separately**; a consensus is never formed first, since collapsing a
  majority vote would destroy the disagreement being measured.
- **Statistic:** Krippendorff's α, nominal difference function. Chosen over
  Fleiss' κ because it handles incomplete and variable-rater designs and missing
  judgments; κ discards any item with a missing rating. κ is reported alongside
  for comparability with the organizers' published figures (κ = 0.64 three-way,
  κ = 0.48 nine-way) but is **not** the inferential statistic.
- **Test:** paired bootstrap over items, **10,000 resamples**. Both partitions are
  recomputed on the *same* resampled items in each iteration.
- **Significance:** **p < 0.05, two-sided.** Reported with a 95% percentile
  interval on the difference.
- **Seed:** 0.

## 4. Secondary quantities (descriptive, not tested)

Reported for interpretation, with no significance claim attached:

- α on the raw nine-way labels, as a floor. Both three-way α values must exceed
  it; if either does not, the collapse is misapplied and the result is void.
- Fleiss' κ and percent-unanimous under each partition.
- The pairwise disagreement matrix under each partition, to see whether any
  effect is driven by a single class rather than being a property of the cut.

## 5. Sensitivity analysis, specified in advance

The coverage value is genuinely arguable for three leaves. The alternatives were
declared in `src/higrec/data/attributes.py` as `COVERAGE_PERTURBATIONS` **before
any result was computed**, and the whole test is rerun under each:

| Name | Change | Rationale for it being defensible |
|---|---|---|
| `partial_as_full` | Partial/half-answer: partial → full | it does deliver some requested information |
| `general_as_none` | General: partial → none | vagueness may deliver nothing specific enough to count |
| `deflection_as_partial` | Deflection: none → partial | it pivots but does engage the topic |

**If the conclusion flips under any defensible alternative, that is a finding and
is reported prominently in the abstract-level summary, not buried in an
appendix.**

## 6. Commitments

1. **We report the result whichever way it falls.** A negative result is written
   up with the same care and the same prominence as a positive one.
2. **We will not add partitions after seeing results.** The coverage cut and the
   three perturbations above are the complete set. Any partition conceived later
   is exploratory and labelled as such.
3. **We will not switch the inferential statistic.** α is the headline; we will
   not promote κ if κ happens to be more favourable.
4. **We will not restrict the sample.** All 308 items are used. Any exclusion
   required for technical reasons is reported with its count and rationale.
5. **The stated limitation stands regardless of outcome.** This test shows
   whether *relabelling existing nine-way judgments* under a different collapse
   raises agreement. It does **not** show whether annotators asked *directly*
   about coverage would agree more. Those are different questions and the paper
   will not conflate them.

## 7. Interpretation fixed in advance

| Outcome | Reading |
|---|---|
| α(coverage) > α(official), p < 0.05, survives all three perturbations | H1 supported. The coarse taxonomy is cut across a dimension that costs agreement. |
| α(coverage) > α(official), p < 0.05, flips under a perturbation | Reported as *conditional* on the coverage assignment of the affected leaf; the perturbation is reported in the main text. |
| Difference not significant | H1 not supported. Reported as a clean negative: the two cuts are not distinguishable at this sample size. Emphasis shifts to C1 and C4. |
| α(coverage) < α(official), p < 0.05 | H1 contradicted. Reported prominently — it would be evidence *against* the proposal's G2 and must not be softened. |

## 8. Power, acknowledged in advance

308 items × 3 annotators is a small sample for a difference-of-α test. We have
**not** run a power analysis, and a null result will therefore be reported as
*inconclusive at this sample size* rather than as evidence of no effect. The
~45.7% of the distribution carried by the three moving leaves is the reason to
expect adequate sensitivity, but that is an argument, not a computation.
