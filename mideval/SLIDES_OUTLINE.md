# Slides — 9 slides, about 9 minutes

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

**8. Is the gap knowledge? Swap the backbone, nothing else** *(figure: `figures/e13_qwen_vs_deberta.png`)*
- Qwen3-8B-Base + LoRA instead of DeBERTa: same rows, input, loss and epoch rule; 10 paired seeds;
  pre-registered.
- Single model: S2 0.384 → **0.476**, S1 0.614 → **0.709**, **10 of 10 seeds up** on both.
- System (10-seed ensemble + logit adjustment): S2 0.405 → **0.543** [+0.054, +0.224], S1
  0.648 → **0.746**. With 2 annotators, as on test: 0.524.
- Above TeleAI's fine-tuned 7B (0.495) and ChulaNLP's cascade (0.52) on dev; below TeleAI's
  multi-call pipeline (0.617).
- The gain is broad (7 of 9 classes, agreed and contested items alike). Our "commitment boundary"
  prediction is refuted.
- Follow-ups, one change each: 12 epochs adopted on the train slice (best epochs 8–11);
  all of train +0.017, not distinguishable.

**9. Second half: does label meaning close the rest?**
- The single-pass point is now an 8B classifier. Its top 3 holds an acceptable label for 93% of
  items, and the least-confident fifth is right only 40% of the time.
- Cascade: only uncertain items go to a larger open LLM with label definitions, a confusion guide
  and boundary examples (TeleAI's biggest lever, +0.097). Sweep the deferral rate to give an
  accuracy-vs-calls curve, with TeleAI's point on it.
- Then distil the cascade back into the single pass, and measure cost per million items.
- Still unsolved: Partial (never predicted) and General (F1 0.32).
