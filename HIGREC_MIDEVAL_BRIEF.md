# HiGrEC — Mid-Evaluation Brief for Claude Code

**Project:** Response clarity / evasion classification for SemEval-2026 Task 6 (CLARITY), QEvasion dataset.
**Team:** 4 people. **Owner of this brief:** Vidvathama.
**Hard deadline:** mid-evaluation submission, **tomorrow 23:59 (local time)**.
**Your job (Claude Code):** (1) audit what already exists in this repo, (2) finish the highest-value items that fit before the deadline, (3) package a mid-eval deliverable with an honest story for the TA, (4) write down the plan for everything after mid-eval.

---

## 0. Rules you must follow

1. **Never invent numbers.** Every number in any report must come from a results file produced in this repo. Cite the file path next to it. If a number does not exist yet, write `TBD` — never estimate it.
2. **Do not delete or overwrite existing results, checkpoints, or scripts.** Before changing anything, run `git status`, commit the current state, and work on a new branch `mideval`.
3. **Ask the user before spending money** (any paid API call) and before any run you estimate will take more than 2 hours.
4. **Change one thing at a time** relative to the current best model. Log every run.
5. **Never make decisions on reporting data.** See the split rules in §3.2.
6. **Respect the clock.** Hard stops are in §4. When a stop arrives, stop experimenting and package what exists.
7. **Planned work is labelled as planned.** Done and planned must never be blurred in the report.

---

## 1. Context (read once)

### 1.1 The task in one paragraph
Given an interview sub-question and a politician's answer, predict:
- **Subtask 1 (3 classes, "clarity"):** Clear Reply / Ambivalent / Clear Non-Reply.
- **Subtask 2 (9 classes, "evasion"):** Explicit, Implicit, General, Partial/half-answer, Dodging, Deflection, Declining to answer, Claims ignorance, Clarification.

Fixed mapping, leaf → coarse:
- Explicit → Clear Reply
- Implicit, General, Partial/half-answer, Dodging, Deflection → Ambivalent
- Declining to answer, Claims ignorance, Clarification → Clear Non-Reply

**We predict the 9-way label and derive the 3-way label from this mapping.** The competition organisers found this the single most effective strategy.

### 1.2 Data (verify every item in §2)
- Train: ~3,448 items, one human label each.
- Dev: ~308 items, three annotators each (multi-reference).
- Test: 237 items, two annotators each. This was the hidden competition test set; label availability is unknown.
- Source: QEvasion release, likely `ailsntua/QEvasion` on Hugging Face. Use whatever loader the repo already uses if one exists.

### 1.3 Scoring
- Both subtasks use **macro-F1** (every class weighted equally).
- **Subtask 2 is multi-reference.** A prediction counts as correct if it matches any annotator's label. Exact definition (from the TeleAI paper, Appendix A.1), with prediction `p` and reference set `G`:
  - **TP for class c:** if `p == c` and `c ∈ G`.
  - **FP for class p:** if `p ∉ G`.
  - **FN for each c ∈ G:** only if `p ∉ G`. If p hits any label in G, no FN is counted for the other labels in G.
  - Per-class P/R/F1 from these counts, then macro-average over the 9 classes.
- **Subtask 1:** standard single-label macro-F1 against the provided coarse gold. Document which gold column you used.
- **If the official scorer exists** in the repo or the task's GitHub, use it and unit-test your implementation against it.

### 1.4 Constraint and strategy
- We cannot run LLMs locally. **Inference stays encoder-only** (DeBERTa-v3).
- An API LLM may be used **once at training time** as a teacher (post mid-eval, with user approval).
- **Final paper story:** an accuracy-vs-cost frontier. A small encoder, optionally taught by an LLM once, with optional deferral of uncertain items to an LLM. This is compared against multi-call LLM pipelines.

### 1.5 Reference numbers (do not mix splits when comparing)

| System | Split | Subtask 1 | Subtask 2 | Source |
|---|---|---|---|---|
| TeleAI (winner), multi-stage DeepSeek-V3 | test | 0.89 | 0.68 | TeleAI paper / task paper |
| ChulaNLP (RoBERTa-large top-5 + Kimi-K2) | test | 0.82 | 0.61 | TeleAI Table 11 |
| Organiser baseline, fine-tuned Llama-70B | test | 0.82 | 0.57 | task paper Table 2 |
| Best encoder-only systems | test | ~0.81 (heavily augmented) | ~0.50 (all < 0.51) | task paper §5 |
| SG-UniBuc chunked RoBERTa-large | test | 0.80 | 0.51 | our proposal (verify) |
| TeleAI CAMSR-CoT | dev (308) | 0.812 | 0.617 | TeleAI Table 2 |
| DeepSeek-V3 single-step CoT | dev (308) | 0.710 | 0.490 | TeleAI Table 2 |
| Qwen2.5-7B LoRA, direct SFT | dev (308) | 0.587 | 0.495 | TeleAI Table 2 |
| Qwen2.5-7B LoRA + CoT distillation | dev (308) | 0.758 | 0.509 | TeleAI Table 2 |
| TeleAI Dodging recall | dev (308) | — | 0.179 | TeleAI Table 8 |

**Key framing fact:** small models, including a LoRA-tuned 7B LLM, plateau around 0.50 on dev Subtask 2. An encoder matching that is already an efficiency result: roughly 20× fewer parameters, single pass.

### 1.6 Original proposal components and how they are re-scoped
| Original | Re-scoped as | Status target for mid-eval |
|---|---|---|
| C1 coverage–move factorisation | 3-way coverage auxiliary head (move ≈ leaf, so only coverage adds sharing) | P2 stretch |
| C2 demand–coverage alignment | Sub-question-conditioned **chunk selection** for long answers, with a random-chunk control | P1 |
| C3 annotator crowd layer | Analysis section, only if per-item annotator IDs exist in train | Verify only (§2) |
| C4 multi-reference decision rule | Temperature scaling + per-class logit bias tuned for multi-ref macro-F1 | **P0** |

---

## 2. STEP 1 — Audit (first ~1 hour; do this before anything else)

Produce `mideval/AUDIT.md` containing all of the following.

### 2.1 Environment
- `nvidia-smi` output (GPU model, memory), or "CPU only".
- Python, torch, and transformers versions.
- Disk space available.

### 2.2 Repo inventory
- Tree of the repo (depth 3, excluding data blobs and checkpoints).
- For every script: one-line purpose, and whether it runs (dry-run or `--help`).
- Every results file found: path, what it contains, the numbers inside, and which split they were computed on.
- Every checkpoint found: path, size, the config that produced it (if recoverable).
- Any existing scorer implementation, and whether it implements multi-reference scoring correctly per §1.3.

### 2.3 Data facts (verify, don't assume)
- Split names and sizes. Map them to train 3,448 / dev 308 / test 237. Release split names may differ from paper names.
- Column names. Specifically check:
  - Is there a **per-item annotator ID on train**? (Needed for C3.)
  - Does dev have **all three annotators' labels** (e.g. `annotator1/2/3`)?
  - Do any test labels exist locally?
- Label distribution per split (9-way and 3-way).
- Answer length distribution in tokens (DeBERTa tokenizer): what fraction exceeds 512?

### 2.4 Status checklist
Fill this table in `AUDIT.md`:

| # | Item | Status (done / partial / missing) | Evidence (file path) | Numbers (with split) |
|---|---|---|---|---|
| 1 | Data loading pipeline | | | |
| 2 | Multi-reference scorer, unit-tested | | | |
| 3 | Frozen dev-A / dev-B split + train-internal val split | | | |
| 4 | Trivial baselines (majority, TF-IDF+LR) | | | |
| 5 | DeBERTa-v3-base 9-way → derived 3-way (B0) | | | |
| 6 | Direct 3-way control model | | | |
| 7 | Multiple seeds for B0 | | | |
| 8 | DeBERTa-v3-large attempt | | | |
| 9 | Temperature scaling + per-class bias (C4) | | | |
| 10 | Error analysis (confusion, per-class, triangle, stratified) | | | |
| 11 | Chunk selection + random control (C2) | | | |
| 12 | Coverage auxiliary head (C1) | | | |
| 13 | Annotator-ID check (C3 feasibility) | | | |
| 14 | Teacher pilot (API) | | | |
| 15 | Cascade / deferral curve | | | |
| 16 | Compression (distil / ONNX / INT8) + CPU latency | | | |

### 2.5 Report back
After writing `AUDIT.md`, print a short summary to the user covering:
- what exists
- what is broken
- the measured time for one training epoch of DeBERTa-v3-base (run one if nothing is recorded)
- a proposed list of P0/P1 items that fit in the remaining time

**Wait for confirmation** only if something in the audit contradicts this brief, for example if there is no GPU or the data is missing. Otherwise proceed to Step 2.

---

## 3. STEP 2 — Build what fits before the deadline

### 3.1 Priority tiers
- **P0 (must exist in the mid-eval package):** scorer, splits, trivial baselines, B0, decision rule, error analysis, report.
- **P1 (should, if time allows):** extra B0 seeds, chunk selection + control, direct 3-way control.
- **P2 (stretch, only if P0 and P1 are done early):** coverage head, DeBERTa-v3-large, annotator-ID feasibility analysis.
- **P3 (do not start before mid-eval; write into the plan only):** teacher labelling, cascade, compression, crowd layer.

Skip anything already marked "done" in the audit, but re-verify its numbers with the scorer.

### 3.2 Split rules (critical)
- **Train-internal validation:** hold out 10% of train, stratified by 9-way label, with a fixed seed. Use it for early stopping and checkpoint selection. **Never use dev for these.**
- **Dev split:** split dev into **dev-A** and **dev-B** (~154 each), stratified by majority 9-way label, with a fixed seed. Save the item IDs to `mideval/splits/`.
  - dev-A is for tuning post-hoc decisions (temperature, biases, chunk-selection hyperparameters).
  - dev-B is for reporting.
- **Untuned models may also be reported on full dev (308).** A model trained only with train-internal validation and no dev-based decisions can be scored on all 308 items. That number is directly comparable to the TeleAI dev numbers in §1.5. Label it clearly as "full dev, no dev tuning".
- Test: use only if labels exist, once, at the very end.

### 3.3 P0 specifications

**(a) Scorer — `src/eval/scorer.py`**
- Implement §1.3 exactly.
- Add unit tests (`tests/test_scorer.py`) with at least 5 hand-computed cases, including:
  - a multi-reference hit on a non-majority label
  - a total miss
  - a class with zero predictions
- If the official scorer is available, assert that your results match it on random predictions.

**(b) Trivial baselines**
- Always-predict-majority-class.
- TF-IDF (1–2 grams) + logistic regression on `sub-question [SEP] answer`.
- Report both on full dev and on dev-B.

**(c) B0: DeBERTa-v3-base, 9-way, derived 3-way**
- Input: `sub-question [SEP] answer`.
  - One variant prepends the full interview question. Run this only if time allows, and keep whichever is better on train-internal validation.
- Truncation: head + tail. Keep the first ~350 and last ~160 answer tokens when the total exceeds 512.
- Starting config:
  - learning rate 2e-5, warmup ratio 0.1, linear decay, weight decay 0.01
  - up to 6 epochs, batch size 16 (use gradient accumulation if memory is tight), max length 512
  - bf16 if supported, otherwise fp32 (DeBERTa-v3 can be unstable in fp16)
- Select the checkpoint by Subtask 2 macro-F1 on train-internal validation.
- Seeds: at least 1 for P0, 3 for P1. Report mean ± std when there are multiple seeds.
- Save the following for every run:
  - logits on all of dev, saved to `results/b0_seed{s}_dev_logits.npy`
  - config
  - metrics JSON

**(d) Decision rule (C4), post-hoc, minutes of compute**
1. **Temperature scaling:** fit a single temperature T on dev-A, minimising NLL against the majority-vote label.
2. **Per-class bias search:** add a bias vector b (9 values) to the logits. Run coordinate ascent: b_c ∈ [−2, 2], step 0.1, 3 passes. The objective is multi-reference Subtask 2 macro-F1 on dev-A.
3. **Report on dev-B.** Then swap the roles (tune on dev-B, report on dev-A) and report both directions. This is a 2-fold estimate of the real gain.
4. Derive Subtask 1 from the adjusted 9-way predictions and report it too.

**(e) Error analysis — `mideval/analysis/`**
- 9×9 confusion matrix (rows = majority gold, columns = prediction), saved as PNG and CSV.
- Per-class P/R/F1 for both subtasks.
- The Dodging–General–Deflection sub-matrix, and the share of all errors that fall inside it. TeleAI reported 77.7%.
- Dodging recall, compared against TeleAI's 0.179 on dev.
- Accuracy stratified by annotator consensus: unanimous / 2–1 / all three different.
- Accuracy stratified by answer length: ≤ 512 tokens vs > 512 tokens.
- 10 example errors from the triangle (sub-question, short answer excerpt, gold set, prediction) for qualitative discussion.

### 3.4 P1 specifications

**(f) Chunk selection (C2)**
1. Apply only to answers longer than 512 tokens; short answers are unchanged.
2. Split the answer into sentences, then group them into windows of ~100–150 tokens.
3. Score each window's relevance to the sub-question by cosine similarity, using a small sentence-embedding model (e.g. `sentence-transformers/all-MiniLM-L6-v2`).
4. Take the top windows in their **original order** until the 512-token budget is filled.
5. **Control:** the same number of randomly chosen windows, in original order.
6. Train B0's config with each input variant. Report:
   - overall score
   - the long-answer subset specifically
   - selection vs random control

**(g) Direct 3-way control**
- The same B0 config, trained on the 3 coarse labels directly.
- Compare its Subtask 1 against B0's derived Subtask 1.

### 3.5 P2 specifications (stretch only)

**(h) Coverage auxiliary head (C1)**
- Add a 3-way head on the same pooled representation. Mapping:
  - full: Explicit, Implicit
  - partial: General, Partial/half-answer
  - none: Dodging, Deflection, Declining to answer, Claims ignorance, Clarification
- Write this mapping into the report.
- Loss = CE_leaf + λ·CE_coverage, with λ ∈ {0.3, 0.5}. Report rare-class F1 (Partial, Clarification) specifically.

**(i) DeBERTa-v3-large**
- Learning rate 1e-5, layer-wise learning-rate decay 0.9, warmup 0.1, gradient checkpointing.
- If it collapses to one class, lower the learning rate once. Keep it only if it beats base by more than the seed std.

**(j) Annotator-ID feasibility (C3)**
- Report only: does the per-item annotator ID exist on train, how many annotators, and the label distribution per annotator. No model.

---

## 4. Clock and hard stops

Let D = deadline (tomorrow 23:59). Adjust to the actual current time when you start.

| When | What |
|---|---|
| Start → +1 h | Audit (§2). Report to user. |
| +1 h → +3 h | Scorer + tests, splits, trivial baselines. |
| +3 h → D−20 h (overnight) | B0 training, 1 seed first, then more seeds in background. Queue P1 runs to use idle GPU time overnight. |
| Morning | Decision rule (d) and error analysis (e) on B0 logits. |
| Midday → D−8 h | P1 items as time allows; P2 only if all P1 is done. |
| **D−8 h: compute freeze** | No new training runs. Only finish runs already in flight if they end before D−6 h. |
| D−8 h → D−2 h | Build the package (§5). Fill the report from results files only. |
| D−2 h → D−1 h | User review of the report. Fix only factual or clarity issues. |
| **D−1 h** | Final package zipped and handed to the user to submit. |

If B0 is not working by D−20 h, **stop all P1 and P2 work.** Spend the remaining time debugging B0 and packaging the infrastructure plus the plan. A working evaluation pipeline plus an honest plan is a valid mid-eval.

---

## 5. STEP 3 — The mid-eval package

Create `mideval/` with:

```
mideval/
  REPORT.md               # 2–4 page mid-eval report (structure in §6)
  SLIDES_OUTLINE.md       # 6–8 slides, one line per bullet, for a short presentation
  AUDIT.md                # from §2
  RESULTS.md              # all tables; every number with its source file path
  PLAN_REMAINING.md       # §7, updated with what actually got done
  REPRODUCE.md            # exact commands to regenerate every number in RESULTS.md
  splits/                 # dev-A / dev-B / train-val item IDs
  analysis/               # confusion matrices, per-class tables, stratified tables, error examples
  figures/                # PNGs used in REPORT.md
```

Requirements for `RESULTS.md`:
- **One main table:** rows = models; columns = Subtask 1 and Subtask 2 on (full dev, untuned) / dev-B / both 2-fold directions; plus parameters and seeds.
- **One comparison table** placing our numbers beside §1.5. Only same-split comparisons go in the same column.
- **Bootstrap 95% CIs** (1,000 resamples) for the main model on dev-B. Half-dev has ~154 items, so small gains may be noise; say so explicitly when a CI overlaps.

---

## 6. The story for the TA (`REPORT.md` structure)

Fill every bracketed item from actual results. Leave `TBD` if a result does not exist.

**1. Problem (3–4 sentences).** Politicians often answer without answering. The task labels each answer with one of 9 evasion types, nested under 3 clarity classes, using a multi-reference scorer.

**2. What the field shows (one short paragraph plus the §1.5 table).**
- Top systems are multi-call LLM pipelines. The winner averages ~1.94 API calls and ~7,300 input tokens per item.
- Small models plateau around 0.50 on Subtask 2, and this includes a LoRA-tuned 7B LLM (0.495–0.509 on dev).
- Residual error concentrates in the Dodging–General–Deflection triangle, where human annotators also disagree most.

**3. Our question.** How much of the accuracy of expensive LLM pipelines can a single-pass encoder recover, and at what cost? The end goal is an accuracy-vs-cost frontier rather than a leaderboard position.

**4. Re-scoping from the proposal (be explicit; TAs reward honest re-scoping).** Include the §1.6 table and one sentence of reasoning per component:
- C1: the move dimension is nearly one-to-one with the leaves, so only the coverage factor adds sharing.
- C2: sub-questions are already decomposed, so the real grounding problem is finding the relevant span of a long shared answer.
- C3: depends on annotator IDs existing on train. Found: [yes/no].
- C4: the scorer credits any annotator's label, so decisions should be tuned for that scorer.

**5. What we built (done items only).**
- [Scorer with tests], [frozen splits], [baselines].
- B0 at [X ± s] Subtask 2 on full dev (untuned), compared with Qwen2.5-7B LoRA at 0.495 and TeleAI at 0.617 on the same split.
- Decision rule: [+Y] on dev-B, [+Y'] in the swapped direction.
- [P1 results, if any.]

**6. What the errors tell us.**
- [triangle share]
- [Dodging recall vs 0.179]
- [consensus-stratified accuracy]
- [long vs short gap]
- 2–3 qualitative examples

**7. Plan to final (from §7)**, with the go/no-go gates.

**8. Risks and fallbacks (from §8).**

The **one-sentence pitch** for the TA, to be adjusted to actual numbers:
> "We built a reproducible, multi-reference evaluation pipeline and a single-pass encoder baseline at [X] on Subtask 2 — [comparable to / below] a LoRA-tuned 7B LLM at ~1/20th the parameters — and our error analysis confirms the bottleneck is the Dodging–General–Deflection boundary; the second half of the project trades a one-time LLM teacher cost and selective deferral for accuracy, and maps the full accuracy-vs-cost frontier."

---

## 7. Plan after mid-eval (write into `PLAN_REMAINING.md`; do not start now)

Rules for all phases: one change at a time, 3 seeds, mean ± std, keep only changes larger than the seed std, tune on dev-A, report on dev-B, cache all API outputs.

**Phase 2 completion (weeks 1–3 after mid-eval):** finish whatever P1/P2 items remain. Produce model B1 = B0 + surviving components, with an ablation table.

**Phase 3 — Teacher (weeks 3–6).** Requires user approval of the budget.
1. **Pilot on 50 train items.** Compare TeleAI's public pipeline against a cheaper single prompt containing their refined definitions and confusion guide. Measure tokens, cost, and agreement with gold, then extrapolate to all ~3,448 items.
2. **Measure teacher quality on dev-A** (measurement only, never training).
3. **Full labelling run.** For each item: a label sampled 3–5 times at moderate temperature (giving a soft distribution), a coverage judgement, and the quoted span the decision rests on. Use JSON outputs, cached.
4. **Targets:** α·onehot(gold) + (1−α)·teacher distribution, with α ∈ {0.5, 0.7}. Discard spans where the teacher's top label disagrees with gold.
5. **Train the student** on the soft targets. Train the chunk scorer on the teacher spans.
6. **Gate:** the student beats B1 by more than the seed std. If not, report it as a negative result and continue with B1.

**Phase 4 — Cascade (weeks 6–8).**
- Uncertainty = margin between the top-2 calibrated probabilities.
- Defer {0, 10, 20, 30, 50, 100}% of the most uncertain items to the API LLM, giving it the encoder's top 3–5 labels.
- Choose the threshold on dev-A and report on dev-B.
- Plot macro-F1 against calls per item and tokens per item, with TeleAI's point (~1.94 calls) on the same plot. **This figure is the centre of the paper.**

**Phase 5 — Compression (weeks 8–10).**
- Distil the best model into DeBERTa-v3-small or xsmall, then ONNX export and INT8 dynamic quantization.
- Check per-class F1 after every step.
- Measure CPU latency (batch 1), throughput (batch 32), size, and memory, with and without chunk selection.
- Convert to cost per 1M QA pairs. Frame as cost and throughput at scale, not as edge deployment.

**Phase 6 — Annotator analysis (parallel, optional).** Only if train annotator IDs exist.
- Crowd layer with one 9×9 confusion matrix per annotator, identity-initialised, with a penalty keeping it near identity.
- Compare learned matrices with the empirical disagreement on dev.
- Analysis section only.

**Phase 7 — Final evaluation and writing (weeks 10–13).**
- Freeze everything. Evaluate once on dev-B, and once on test if labels or a Codabench post-eval phase are available.
- Bootstrap CIs, final error analysis, write the workshop paper, release the code and cached teacher outputs.

**Workstreams for 4 people:**
1. Evaluation and baseline: scorer, splits, B0, decision rule, final evaluation.
2. Encoder improvements: chunk selection, coverage head.
3. Teacher and cascade.
4. Compression and annotator analysis.

The paper owner also owns the results log.

---

## 8. Risks and fallbacks

| Risk | Fallback |
|---|---|
| No GPU / very slow GPU | Use DeBERTa-v3-small or xsmall for the mid-eval baseline; state it; move base/large to after mid-eval. |
| DeBERTa-v3 unstable (NaNs, collapse) | Switch fp16→bf16 or fp32, lower learning rate, add warmup. |
| Large model doesn't fit | Stay on base. Nothing else depends on large. |
| No multi-annotator labels found for dev | Score single-label macro-F1, state the limitation, find the full annotations from the Thomas et al. 2024 GitHub release. |
| No train annotator IDs | Drop C3 to a one-line note. |
| API budget tight (post mid-eval) | Teacher-label a stratified subset, oversampling rare classes; use the single-prompt teacher. |
| Teacher doesn't help | Paper becomes "careful encoder baseline + accuracy/cost analysis with cascading". Still workshop-publishable. |
| Gains within noise | Report with CIs. A clean negative ablation is better than an inflated positive. |

---

## 9. First command

Start with §2 (Audit). Do not begin training until `mideval/AUDIT.md` exists and you have reported the summary to the user.
