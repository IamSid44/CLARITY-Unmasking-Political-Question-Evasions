# Slides — 8 slides, about 8 minutes

**1. The task**
- Presidential interviews: classify how a sub-question was answered.
- 9 evasion types (Subtask 2), nested under 3 clarity levels (Subtask 1).
- Macro-F1; a prediction counts if it matches any annotator (3 on dev, 2 on test).

**2. What the field shows**
- Winner: a multi-stage LLM pipeline, 0.68 on test Subtask 2, about 2 calls and 7,300 tokens per item.
- Encoders stop around 0.50 on test.
- A fine-tuned 7B without chain-of-thought reaches 0.495 on dev.
- Our question: how much of that accuracy can a single pass recover, and at what cost?

**3. Before training: what the metric rewards**
- If every prediction is acceptable, macro-F1 = (classes ever named) / 9.
- 69% of rows share their answer with another sub-question, so the hard part is attribution.
- Our coverage hypothesis (C2), pre-registered, was refuted: α 0.50 vs 0.62, p = 0.0004.

**4. Two changes that worked** *(figure: `figures/per_seed_progression.png`)*
- 16 epochs, then the full journalist question in the input.
- Single model: 0.315 → 0.384 on Subtask 2, 0.576 → 0.614 on Subtask 1, 9 of 10 seeds up on each.
- Lesson: at 8 epochs the full question looked harmful; it was undertraining.

**5. From model to system**
- One-parameter decision rule: +0.109 on the baseline ensemble, nothing on the better one.
- Final system: 0.405 / 0.648. Baseline system: 0.428 / 0.601.
- 5-seed system numbers swing by 0.08, so every system claim uses 10 seeds.
- Proposal check: the rule gains through coverage of rare classes, not on contested items.

**6. What did not help, and why**
- Re-ranker encoder: has no knowledge the first model lacks.
- Rebalancing losses: over-correct; the post-hoc rule does the same job for free.
- Three hierarchies: none beats the flat model.
- Boundary experts: no effect. Model soups: seeds don't share a basin.

**7. What the errors say**
- Human vs human 0.684, model 0.38 under the same scoring.
- 53 of 137 errors are on items all three annotators agreed on.
- Largest confusion: how committed an answer is (Implicit → Explicit, General → Implicit/Explicit).
- Confidence is informative (in-set 0.43 → 0.82 from least to most confident fifth); an
  acceptable label is in the top 3 for 94% of items.

**8. Second half: knowledge or label meaning?**
- A. Qwen3-8B LoRA single-pass classifier, the same protocol. **Done (3 seeds):** dev S2 0.462 ± 0.080 vs DeBERTa 0.381 on the same seeds (+0.081, 3/3 up); S1 0.688 vs 0.619. Passes the screen; 10 seeds next.
- B. Cascade: the LLM sees only uncertain items and their candidates; the curve of accuracy against calls per item.
- C. One-time teacher distilled into the single-pass student.
- D. Measured cost per million items.
