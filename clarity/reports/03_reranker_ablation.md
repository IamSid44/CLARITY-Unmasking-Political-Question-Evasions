# 03 — Ablation: can a second encoder choose better than the first?

**Result: no.** A DeBERTa-v3-large re-ranker over the baseline's top-3 candidates
scores **0.329 ± 0.039** dev Subtask-2 macro-F1 against **0.365** for the baseline
it re-ranks, and loses on held-out training data too. The opportunity it targeted is
real — a perfect chooser would score 0.745 — but an encoder cannot exploit it at
this data size.

Code: `rerank.py`. Runs: `runs/R1_rerank/` (5 seeds). Chronology and the failed
earlier attempts: `02_experiment_log.md`, E6.

---

## 1. The opportunity

The baseline's ranking is much better than its final choice. For the 5-seed
ensemble on dev:

| a correct label is among the model's top … | 1 | 2 | 3 | 5 |
|---|---|---|---|---|
| share of items | 0.562 | 0.779 | **0.896** | 0.994 |

| decision over the same probabilities | dev S2 |
|---|---|
| argmax (take the top candidate) | 0.365 |
| best achievable by choosing among the top 3 (oracle) | **0.745** |

So the question is whether a model can learn to choose among three candidates
better than argmax does. Three-way choice between close labels should be easier
than nine-way classification. It is also the architecture of the 2nd-place system
(an encoder shortlist, with an LLM choosing).

## 2. The design

```
 sub-question + answer ──► ENCODER 1 (baseline) ──► top-3 candidates
                                                         │
 for each candidate c:  "Candidate: c — <definition of c>. Question: <sub-question>" [SEP] <answer>
                                    ──► ENCODER 2 ──► one score
                                                         │
                                        highest score = prediction
```

- **The candidate is an input, not an output slot.** Each label has a fixed
  one-sentence definition (paraphrased from the organizers' taxonomy), so one
  scoring network handles any three candidates. This is the standard retrieval
  re-ranker setup (monoBERT; Nogueira & Cho, 2019) with labels in place of documents.
- **Training:** softmax over each item's three scores, gold as the target. If the
  gold label is not among the three, it replaces the third candidate.
- **Cross-fitting.** The baseline memorises its training data (train loss ≈ 0.6),
  so its top-3 on a training item almost always starts with the gold label, and a
  re-ranker trained on that would learn "pick the first one." Training candidates
  therefore come from 5 copies of the baseline, each trained on 4/5 of train and
  predicting the held-out fifth. Dev and test candidates come from the full 5-seed
  baseline ensemble.
- **Fusion variant:** re-ranker score + β · log p(baseline), with β chosen on a
  held-out slice of train from {0, 0.25, … , 64}. At large β this reduces to the
  baseline's own ordering, so fusion can always fall back to the baseline.
- Same backbone, learning rate and schedule as the baseline (DeBERTa-v3-large,
  lr 1e-5, 8 epochs), 5 seeds.

## 3. Results

| | dev S2 | held-out train slice |
|---|---|---|
| baseline, keep its top choice | **0.365** | **0.405** |
| re-ranker alone | 0.329 ± 0.039 | 0.300 ± 0.010 |
| re-ranker + baseline prior (β = 4–16) | 0.354 ± 0.003 | — |

It loses on **both** evaluation sets, for **every** seed, so this is not an
artefact of the 308-item dev set.

It is not ignoring its input. Averaging the 5 seeds' scores:

| | |
|---|---|
| overrides the baseline's top choice | 37.7% of items |
| of the baseline's 135 errors, fixes | 25 |
| of the baseline's 173 correct answers, breaks | 37 |

It changes a lot of answers, and slightly more of those changes are wrong than
right. Learning stayed slow throughout: training loss went from 1.06 to 0.88 over
8 epochs, where 1.10 is random guessing among three.

The fusion variant behaves as designed. The held-out slice keeps asking for a large
β, i.e. to trust the baseline, and at large β the result converges to the
baseline (0.354 vs 0.365, with seed spread shrinking to ±0.003).

## 4. Why it fails

The two models face different learning problems with the same 3,400 examples:

- The **baseline** has one output unit per class. Every training example updates
  all nine class scores, so each class is learned from all 3,400 examples.
- The **re-ranker** has to learn what each *definition* means — that "too general
  to give the specific information asked for" describes one answer and not
  another — and apply it by comparing text. That is a harder, more semantic task,
  and the training signal per candidate is weaker (a three-way choice per item,
  with the easy cases already decided by the shortlist).

In short, the re-ranker needs label understanding the encoder does not start with
and cannot acquire from this much data. That is what an LLM brings: the 2nd-place
system used Kimi-K2 in this slot, and on dev it improved on their own DeBERTa by
+0.04 (0.46 → 0.50, top-3, few-shot), and by +0.18 on test.

## 5. What we keep from it

- **A measured boundary for the "decision-rule ceiling" claim.** The right answer
  is usually among the top 3, but neither a trained encoder chooser (+0.000) nor a
  tuned decision rule (0.365 → 0.438, about a fifth of the gap to 0.745) captures
  most of it. That is evidence *for* the organizers' view that this step needs
  world knowledge, not against it.
- **The infrastructure.** Cross-fitted candidates, the candidate-as-input scorer and
  the fall-back fusion are exactly what an LLM re-ranker will plug into later; only
  the scoring model changes.
- **Two procedural lessons**, both cost reruns: match the new component's training
  budget to the baseline's (the first attempt used 3 epochs), and make any fitted
  weight's search range wide enough to include "don't use the new component".
