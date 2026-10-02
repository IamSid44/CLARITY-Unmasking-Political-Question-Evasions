# Plan from mid-evaluation to the final

**Goal:** an accuracy-vs-cost curve for this task, from a single-pass encoder up to multi-call LLM
pipelines, with each point measured under one protocol.

**Rules for every phase:**
- one change at a time against a named control;
- hypotheses and predictions written before each run;
- screening on 3 paired seeds, extended to 10 only past a bar fixed in advance (≥ +0.015 with
  ≥ 2/3 seeds up);
- 10 seeds for any system claim;
- nothing tuned on dev except through nested cross-validation;
- two-annotator scores reported alongside three-annotator ones;
- every LLM output cached.

**Compute.** The lab's shared GPUs (one 96 GB card, often occupied) and about ₹5,880 of JarvisLabs
credit. Cloud runs use a timing pilot, a projected-cost cap and automatic pause.

## Phase A — Is the gap knowledge? (week of 2026-10-02)

| step | what | gate |
|---|---|---|
| A1 | **Qwen3-8B-Base + LoRA (r=16, all linear layers) as a 9-way classifier**, 3 epochs. Same rows, train slice, input and selection rule as DeBERTa E10_fullq_16ep. Seeds 0–2. Code: `clarity/llm_classifier.py`; runner: `jarvis_setup.sh` / `jarvis_drive.sh`. **Done 2026-10-02** on 1× RTX PRO 6000 (JarvisLabs VM), 21 min per seed: S2 0.462 ± 0.080, +0.081 over DeBERTa (3/3 up); S1 +0.068 (RESULTS §9). | ≥ +0.015 S2 over DeBERTa, same seeds, ≥ 2/3 up → A2. **Passed** |
| A2 | Seeds 3–9, then the 10-seed system with LA, compared with DeBERTa's 10-seed system. **Done 2026-10-02:** single S2 0.476 ± 0.043 (10/10 seeds above DeBERTa); system S2 0.543 / S1 0.746 vs 0.405 / 0.648, +0.136 [+0.054, +0.224] (RESULTS §9) | system claim at 10 seeds. **Met** |
| A2b | 12 epochs instead of 3, seeds 0–2. **Done:** train-slice F1 +0.084 (3/3), best epochs 8–11 → **adopted** by the registered rule. All of train (10 seeds): +0.017 per model, system difference not distinguishable from zero | slice rule (+0.015, ≥ 2/3) |
| A2c | **Next:** 12 epochs at seeds 3–9 → 10-seed system; then all of train with 12 epochs as its own single-change test (≈ ₹1,600 per 7 seeds on 4 parallel VMs, from E13b's 59 min per seed) | system claim at 10 seeds |
| A3 | One smaller point on the curve (Qwen3.5-4B or Gemma 4 E4B), same protocol | — |

**Interpreting A1.**
- **If it passes:** knowledge and scale are a large part of the gap, and the 8B classifier becomes
  the middle point of the curve.
- **If it fails:** the gap is about what the labels mean, which a classifier cannot learn from
  3,400 examples. Phase B, where definitions enter through the prompt, gets the budget.

## Phase B — Cascade (weeks 2–3)

1. **Candidates.** The cheap model's top-3 already contains an acceptable label for 93.8% of dev.
   Replace the fixed top-3 with conformal prediction sets calibrated on cross-fitted train
   probabilities. Each item then gets as few candidates as its uncertainty allows.
2. **Chooser.** An open LLM served with vLLM (Qwen3.6-27B dense, or a 35B-A3B mixture-of-experts
   model for throughput), given:
   - label definitions;
   - a confusion guide for the measured error pairs (Implicit/Explicit, General/Implicit,
     General/Dodging, Deflection);
   - few-shot examples retrieved from train.

   Prompt choices are made on a train slice, never on dev.
3. **The curve.** Defer the least-confident {0, 10, 20, 30, 50, 100}% of items. Confidence is
   informative here: in-set rate rises from 0.43 (least-confident fifth) to 0.82 (most). Plot
   S2 against LLM calls and tokens per item, with TeleAI (1.94 calls, ~7,300 tokens) on the same
   axes.
4. **A reasoning-budget axis** on the same model (thinking off / short / long), to measure what
   chain-of-thought adds with everything else fixed.

## Phase C — One-time teacher (weeks 3–5)

1. Teacher labels for all of train from the best Phase-B prompt: 3 samples per item (giving a
   soft label) plus a short rationale. Teacher quality is measured on a train slice first.
2. Students: DeBERTa on soft targets, and the 8B classifier with rationale generation as an
   auxiliary objective.
3. **Gate:** the student beats its own baseline by more than the seed spread. If not, it is
   reported as a negative result.

## Phase D — Cheap encoder-side items, on lab GPUs when free

| item | why |
|---|---|
| E12a (all of train, last epoch) on seeds 5–9 | the one encoder variant that helped (+0.029 per model); needs 10 seeds for a system claim |
| direct 3-way control | we derive S1 from the 9-way label, following the papers, but never ran the control ourselves |
| contrastive training across sub-questions that share an answer | 69% of rows share their answer; attribution is the core difficulty |
| per-annotator heads for P(class in set) | annotator IDs exist on train; the proposal's C3/C4 in their cheapest form |

## Phase E — Measured cost (week 5)

Quantise the classifiers (INT8/INT4, ONNX for DeBERTa). Measure throughput and latency on an L4
and on CPU, and convert to cost per million items. The cost axis is measured, not estimated from
parameter counts.

## Phase F — Final evaluation and writing (weeks 6–7)

- Freeze all systems.
- Report 10 seeds, nested CV, two-annotator scoring, the short-answer half of dev (closest to
  test) and the human ceiling (0.684) as a reference line.
- Release code and cached LLM outputs.

## Risks

| risk | response |
|---|---|
| A1 fails | Phase B gets the budget; the negative result is itself a finding about where the encoder gap lies |
| cloud budget runs short | 3-seed screens only, 2-epoch runs, a 35B-A3B mixture-of-experts teacher instead of a dense 27B |
| lab GPUs unavailable | Phase D waits; nothing in Phases A–C depends on it |
| dev too small to separate systems | paired seeds and bootstrap intervals; conclusions stated at the resolution the data supports |
