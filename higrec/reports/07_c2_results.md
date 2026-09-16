# 07 — C2 Results: the partition agreement test

**Pre-registration:** `reports/06_c2_preregistration.md`, committed as `3fd751f`
**before** any α value was computed. This report is scored against it.

## Headline: H1 is REFUTED, significantly, in the opposite direction

| partition | Krippendorff α | Fleiss κ | % unanimous |
|---|---|---|---|
| **Official clarity cut** | **0.6225** | 0.6221 | 72.7% |
| **Coverage-derived cut** | **0.5028** | 0.5022 | 54.9% |
| raw 9-way labels (floor) | 0.4728 | — | 40.6% |

```
delta = α(coverage) − α(official) = −0.1198
95% CI  = [−0.1890, −0.0490]
p       = 0.0004  (two-sided, 10,000 paired bootstrap resamples)
```

H1 predicted α(coverage) **>** α(official). The observed difference is negative,
large, and significant at p = 0.0004. **The official clarity partition agrees
substantially better than the coverage-derived partition.**

Per the pre-registration's interpretation table, this is the row reading
*"H1 contradicted. Reported prominently — it would be evidence against the
proposal's G2 and must not be softened."* It is reported accordingly.

### Sensitivity analysis (three pre-declared perturbations)

| perturbation | delta | p | H1 supported |
|---|---|---|---|
| `partial_as_full` | −0.1124 | 0.0035 | No |
| `general_as_none` | −0.0736 | 0.0585 | No |
| `deflection_as_partial` | −0.1140 | 0.0030 | No |

The result is robust. Under no defensible alternative coverage assignment does
the coverage cut match, let alone beat, the official cut. The weakest case
(`general_as_none`, p = 0.0585) still shows a negative point estimate.

## Pipeline validation against the organizers' published figures

Before trusting any of the above, our agreement implementation reproduces the
organizers' own published numbers closely:

| statistic | organizers (published) | ours | 
|---|---|---|
| Fleiss κ, 3-way clarity | 0.64 | **0.6221** |
| Fleiss κ, 9-way evasion | 0.48 | **0.4728** |

Residual differences are consistent with the organizers computing over 317 raw
items with their own in-taxonomy filtering, against our 308 published items. This
is a meaningful check: a quietly-wrong α implementation would have produced a
confident and false result here, and nothing downstream would have caught it.

## Interpretation

**Stated first, and plainly: the proposal's G2 motivation, as operationalized in
this test, is not supported by the data.**

### A structural explanation — and an admission about the test's design

The two partitions differ in how they distribute nine leaves across three coarse
classes:

| partition | Clear Reply | Ambivalent | Clear Non-Reply |
|---|---|---|---|
| Official | 1 leaf | **5 leaves** | 3 leaves |
| Coverage | 2 leaves | 2 leaves | **5 leaves** |

The official partition places all five *mutually confusable* leaves — Implicit,
Partial/half-answer, General, Deflection, Dodging — into a single bucket.
Disagreement among those five is therefore **absorbed** and becomes invisible at
the coarse level. The coverage cut splits exactly that group across all three
buckets (Implicit → Clear Reply, Dodging and Deflection → Clear Non-Reply), so
the same underlying disagreements now cross coarse boundaries and are counted.

This suggests the test as designed measures something narrower than intended:
**agreement-under-collapse rewards whichever partition groups the confusable
classes together**, largely irrespective of whether the axis is conceptually
right. Any partition that absorbs the high-disagreement neighbourhood into one
bucket will score well.

**This confound was not anticipated in the pre-registration, and that is a fair
criticism of the test's design rather than of the result.** It was structurally
predictable from the leaf-per-class counts and should have been raised in advance.
Offering it now is post-hoc interpretation, and it is labelled as such: it does
not rescue H1, and H1 stands refuted.

### What this does and does not affect

| Contribution | Impact |
|---|---|
| **C2** | Clean negative result. Reported as the primary outcome. |
| **C5** (coverage/span module) | Motivation weakened — it was also justified by the coverage-axis argument. Its value must now rest on the span-selection task itself, not on the taxonomy claim. |
| **C1** (attribute code) | **Unaffected.** C1's value is sample efficiency on rare leaves through shared attribute values. That argument never depended on coverage being a better *coarse* axis. |
| **C4** (decision rule) | **Unaffected**, and independently strengthened — see `03_data_audit.md` §6. |

### What we can still say about the taxonomy

The 9-way α of 0.4728 against the 3-way 0.6225 confirms the fine-grained task is
where annotators genuinely diverge, and 59.4% of items carry more than one
distinct label. The disagreement the proposal points at is real. What this test
refutes is the specific claim that **re-cutting the coarse taxonomy along
coverage** would recover agreement.

## Limitation, restated from the pre-registration

This test shows what happens when *existing nine-way judgments* are collapsed two
different ways. It does **not** show what annotators asked *directly* about
coverage would do. That question requires the re-annotation campaign and remains
open. A negative result here does not establish that coverage is a poor axis for
*annotation*; it establishes that it is a poor axis for *collapsing these
existing labels*.
