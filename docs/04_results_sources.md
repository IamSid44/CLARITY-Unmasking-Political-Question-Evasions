# 04 — Results: every number with its source

All on the 308-item dev set. "S2" = Subtask 2 (9 classes, multi-reference macro-F1);
"S1" = Subtask 1 (3 classes, derived from the 9-way prediction).

**Abbreviations.** LA = logit adjustment: divide each class's probability by its training
frequency raised to τ, the one fitted parameter. Nested CV = τ fitted and scored by 5-fold nested
cross-validation on dev, so the reported score is held-out. 2-fold = τ (or temperature + 9 biases)
fitted on one half of dev and scored on the other, both directions.

The main sources are:
- `raw/mideval/mideval_analysis.txt`, written by `python -m preregistered_analyses.mideval_analysis` (CPU, 24 s).
- `docs/02_experiment_log.md` (the log), and the raw outputs it cites in
  `docs/raw/`.
- For the Qwen classifier (§9): `docs/raw/E13_final_analysis.txt`, written by
  `python -m preregistered_analyses.e13_analysis` (CPU, about 3 min, needs the run folders from the HF repo).

## 1. Main table

Source: `raw/mideval/main_table.csv` (section D of `raw/mideval/mideval_analysis.txt`).

| | baseline (sub-q + answer, 8 ep) | 16 epochs | **full question, 16 ep (FINAL)** | all of train, last epoch |
|---|---|---|---|---|
| seeds | 10 | 10 | 10 | 5 |
| single model S2 | 0.315 ± 0.040 | 0.362 ± 0.026 | **0.384 ± 0.030** | 0.416 ± 0.020 |
| single model S1 | 0.576 ± 0.030 | 0.601 ± 0.016 | **0.614 ± 0.027** | 0.614 ± 0.016 |
| ensemble S2, full dev, no dev tuning | 0.319 | 0.377 | **0.409** | 0.481 |
| ensemble S1, full dev, no dev tuning | 0.596 | 0.622 | **0.621** | 0.634 |
| ensemble S2, dev-B only, no tuning | 0.302 | 0.357 | 0.444 | 0.501 |
| + LA, S2, 2-fold A→B / B→A | 0.375 / 0.396 | 0.367 / 0.387 | 0.395 / 0.352 | 0.511 / 0.448 |
| + LA, S1, 2-fold A→B / B→A | 0.590 / 0.589 | 0.606 / 0.617 | 0.615 / 0.644 | 0.623 / 0.671 |
| + temperature & 9 biases, S2, 2-fold A→B / B→A | 0.332 / 0.384 | 0.461 / 0.358 | 0.392 / 0.344 | 0.293 / 0.441 |
| **+ LA, S2, nested CV on full dev** | 0.428 | 0.400 | **0.405** | 0.489 |

- **The final system is the FINAL column with LA, nested CV:** S2 0.405, S1 0.648 (log, §E11).
- **The dev-A/dev-B split** is in `data/splits/dev_A_B.json` (stratified by majority label, seed 2026).
  The 2-fold rows are reported for the protocol in the team brief. Nested CV over all 308 items is
  the less noisy estimate: the 9-bias rule swings from 0.29 to 0.46 between halves.

## 2. Per-model comparisons, paired by seed

Source: the log, §E11 (`docs/raw/E11_replication.txt`).

| comparison, 10 seeds | S2 | S1 |
|---|---|---|
| 16 epochs − baseline | +0.047 (t = 3.4; 9/10 up) | +0.026 (t = 2.7; 8/10) |
| full question − 16 epochs | +0.022 (t = 2.0; 7/10) | +0.012 (t = 1.3; 6/10) |
| full question 16 ep − baseline | **+0.069 (t = 3.6; 9/10)** | **+0.038 (t = 2.8; 9/10)** |
| all of train − full question, 5 seeds (log §E12) | +0.029 (4/5) | −0.017 (1/5) |

Figure: `figures/per_seed_progression.png` (`python -m figures.mideval_figures`).

## 3. Bootstrap intervals

1,000 resamples over dev items, seeds held fixed. Source: `raw/mideval/mideval_analysis.txt` §E.

| | estimate | 95% interval |
|---|---|---|
| FINAL ensemble, no tuning, S2 | 0.409 | [0.306, 0.472] |
| FINAL ensemble, no tuning, S1 | 0.621 | [0.542, 0.695] |
| FINAL ensemble, dev-B only, S2 | 0.444 | [0.299, 0.519] |
| FINAL − baseline ensemble, no tuning, S2 | +0.090 | [+0.009, +0.168] |
| FINAL − baseline ensemble, no tuning, S1 | +0.024 | [−0.054, +0.098] |

## 4. Floors

Source: `raw/mideval/trivial_baselines.csv`.

| | S2 (3 annotators) | S2 (2 annotators, mean) | S1 | in-set rate | classes named |
|---|---|---|---|---|---|
| always majority (Explicit) | 0.060 | 0.055 | 0.136 | 0.373 | 1 |
| uniform random (mean of 20) | 0.129 | 0.116 | 0.299 | 0.196 | 9 |
| TF-IDF 1–2 grams + logistic regression | 0.257 | 0.243 | 0.445 | 0.351 | 8 |

The TF-IDF settings (class_weight=balanced, C=1) were chosen by 5-fold cross-validation on train.

## 5. Comparison with published systems

Same-split rows only.

| system | split | S1 | S2 | source |
|---|---|---|---|---|
| TeleAI pipeline | dev | 0.812 | 0.617 | TeleAI, Table 2 |
| ChulaNLP DeBERTa-large (checkpoint chosen on dev) | dev | 0.65 | 0.46 | ChulaNLP paper; log §E7 |
| Qwen2.5-7B LoRA, direct fine-tune | dev | 0.587 | 0.495 | TeleAI, Table 2 |
| DeepSeek-V3, single-step CoT | dev | 0.710 | 0.490 | TeleAI, Table 2 |
| DeepSeek-V3, direct | dev | 0.662 | 0.421 | TeleAI, Table 2 |
| **ours, Qwen3-8B LoRA, 10-seed system (ensemble + LA)** | dev | **0.746** | **0.543** | §9 |
| **ours, Qwen3-8B LoRA, single model (10 seeds)** | dev | **0.709** | **0.476** | §9 |
| ours, DeBERTa final system (E11) | dev | 0.648 | 0.405 | above |
| ours, DeBERTa single model | dev | 0.614 | 0.384 | above |
| human annotator vs the other two (2-annotator scoring) | dev | — | 0.643–0.766 (mean 0.684) | `raw/mideval/mideval_analysis.txt` §G |
| TeleAI | test | 0.89 | 0.68 | task overview, Table 3 |
| Llama-70B fine-tuned (organisers) | test | 0.82 | 0.57 | task overview, Table 2 |

## 6. Error analysis

The FINAL 10-seed ensemble, argmax, no dev tuning. Source: `raw/mideval/mideval_analysis.txt` §F–G.

| quantity | value |
|---|---|
| predictions outside the reference set | 137 / 308 (44.5%); 53 of them on unanimous items |
| Dodging recall / F1 | 0.494 / 0.543 (TeleAI: 0.179 / 0.289) |
| triangle share of errors, TeleAI's definition / strict definition | 86.9% / 29.2% (TeleAI: 77.7%) |
| largest error pairs (reference label → prediction) | Implicit → Explicit 25; General → Implicit 20; General → Explicit 20; Dodging → Implicit 17; Explicit → Implicit 17 |
| in-set rate: unanimous / 2–1 / all three differ | 0.576 / 0.507 / 0.697 (n = 125 / 150 / 33) |
| in-set rate: input ≤1024 tokens / cut (>1024) | 0.580 (n = 288) / 0.200 (n = 20) |
| S2 on the shorter / longer half of dev | 0.400 / 0.283 |
| in-set rate by confidence, least → most confident fifth | 0.435, 0.426, 0.468, 0.623, 0.823 |
| acceptable label in top 1 / 2 / 3 / 5 | 0.555 / 0.805 / 0.938 / 0.984 |
| seed agreement: pairwise / all 10 agree | 0.514 / 16.9% of items |

The per-class table is `raw/mideval/per_class_final.csv`, the confusion matrix
`raw/mideval/confusion_final.png` and `.csv`, and ten sampled errors `raw/mideval/error_examples.md`.

## 7. The proposal's C4 prediction

Source: `raw/mideval/mideval_analysis.txt` §H. LA with nested CV vs argmax, same ensemble.

| in-set rate | unanimous (125) | 2–1 split (150) | all differ (33) | S2 |
|---|---|---|---|---|
| baseline ensemble, argmax → LA | 0.536 → 0.496 | 0.513 → 0.527 | 0.727 → 0.788 | 0.319 → 0.428 |
| FINAL ensemble, argmax → LA | 0.576 → 0.536 | 0.507 → 0.493 | 0.697 → 0.727 | 0.409 → 0.405 |

Per-class F1 change on the baseline:

| class | change |
|---|---|
| Partial/half-answer | +0.429 |
| Claims ignorance | +0.316 |
| Deflection | +0.151 |
| General | +0.142 |
| Implicit | +0.122 |
| Declining to answer | +0.030 |
| Clarification | −0.036 |
| Explicit | −0.082 |
| Dodging | −0.086 |

## 8. Pre-registered hypotheses that were tested and refuted

| hypothesis | result | source |
|---|---|---|
| C2: annotators agree more under the coverage cut | α 0.503 vs 0.623 for the official cut; Δ −0.120, 95% CI [−0.189, −0.049], p = 0.0004 | team's earlier analysis track, in git history; log "Starting point"; numbers in `docs/03_research_narrative.md` §3 |
| E10's 0.478 system replicates | 0.434 on seeds 5–9; 0.405 at 10 seeds | log §E11 |
| a mixed-input ensemble beats a single-input one | 0.403 vs 0.405 | log §E11 |
| a dedicated Non-Reply gate raises recall | recall −0.039 | log §E12b |
| a model soup lands between one model and the ensemble | 0.267, below a single model | log §E12d |
| the 8B classifier's gain falls mostly on the commitment boundary and on agreed items (E13 H3) | broad: 7 of 9 classes, largest in Claims ignorance +0.40 / Declining +0.22; in-set +0.076 unanimous vs +0.105 two-label | §9; log §E13x |

## 9. Qwen3-8B LoRA classifier (E13, done 2026-10-02)

**One change from the final DeBERTa model (E10/E11 full question, 16 epochs): the backbone.**
- The model is Qwen3-8B-Base with LoRA r=16 on every linear layer and a new 9-way head on the
  last token, trained for 3 epochs, with the epoch chosen on the train slice.
- Everything else is identical: rows, train slice, input text and budgets, loss, batch size,
  selection rule.
- Seeds 0–9 are paired with DeBERTa's seeds 0–9.

Sources:
- `docs/raw/E13_final_analysis.txt`, written by `python -m preregistered_analyses.e13_analysis`
  (sections A–E);
- `docs/raw/E13_lane_summaries.txt`;
- the log, §E13–E13c;
- figure `figures/e13_qwen_vs_deberta.png` (`python -m figures.e13_figures`).

**Single models and systems, 10 seeds.**

| | Qwen3-8B LoRA | DeBERTa full question 16 ep | Qwen − DeBERTa |
|---|---|---|---|
| single model S2 | **0.476 ± 0.043** | 0.384 ± 0.030 | **+0.092** (t = 7.1, **10/10** seeds up) |
| single model S1 | **0.709 ± 0.031** | 0.614 ± 0.027 | **+0.095** (t = 8.5, **10/10**) |
| single model + LA, S2 | 0.538 ± 0.043 | 0.389 ± 0.019 | +0.149 (10/10) |
| 10-seed ensemble, no dev tuning, S2 | 0.495 | 0.409 | |
| **10-seed system (ensemble + LA, nested CV), S2** | **0.543** (mean of 10 CV splits 0.549) | 0.405 (0.412) | **+0.136**, 95% [+0.054, +0.224] |
| **10-seed system, S1** | **0.746** | 0.648 | +0.098 |
| system, three 2-annotator reference sets (test regime) | 0.498 / 0.536 / 0.537 | 0.356 / 0.403 / 0.385 | +0.142 (means) |

**Two follow-ups, each a single change from the Qwen model above.**

| | seeds | single S2 | single S1 | system S2 (+LA) | verdict |
|---|---|---|---|---|---|
| **12 epochs instead of 3** (E13b) | 0–2 | 0.543 ± 0.015 | 0.742 ± 0.008 | — | the train-slice F1 rises by +0.084 (3/3), so it is **adopted** by the registered rule; best epochs 8–11 |
| all of train, last epoch (E13c) | 0–9 | 0.492 ± 0.045 | 0.715 ± 0.027 | 0.575 (10 splits 0.562) | +0.017 per model (6/10); system +0.028 [−0.026, +0.092]: **not distinguishable** |

**Where the gain falls** (per-class S2 F1, mean over 10 seeds, Qwen − DeBERTa):
- Claims ignorance +0.403, Declining +0.220, Dodging +0.103, Deflection +0.089;
- General +0.059, Explicit +0.057, Implicit +0.040;
- Partial 0 for both (never predicted); Clarification −0.144 (4 items).

By annotator agreement, the in-set rate gains are: unanimous items +0.076, 2 labels +0.105,
3 labels +0.048.

**Mixing Qwen with DeBERTa** (weight and τ fitted inside the nested CV) gives 0.543, against Qwen
alone at 0.549 (both means over 10 CV splits). The CV puts 0.79 of the weight on Qwen.

How to read this:
- **The backbone is the largest single gain in the project.** It is larger than the full
  question, the epochs and every decision rule combined. It holds on all 10 seeds and on the
  2-annotator sets.
- **Against published dev numbers:**
  - the single-pass 10-seed Qwen system (S2 0.543) is above TeleAI's fine-tuned Qwen2.5-7B (0.495)
    and ChulaNLP's encoder + Kimi-K2 cascade (0.52);
  - it is below TeleAI's multi-call pipeline (0.617) and the human ceiling (0.684).
- **Logit adjustment matters more for Qwen** (+0.062 per model, against +0.005 for DeBERTa). Raw
  Qwen under-predicts General: on seeds 0–2, 24 predictions, though General is in 113 reference
  sets (`raw/E13_Q8_topk_confidence.txt`).
- **An acceptable label is in Qwen's top 3 for 93% of items, and confidence predicts
  correctness:** the in-set rate rises from 0.40 to 0.87 across confidence fifths. Both are
  measured on seeds 0–2. This is the headroom the planned cascade targets.
- **The 3-seed screen of E13 (S2 0.462 ± 0.080) is superseded by the 10-seed numbers.**

## 10. Phase 2, module M1: Qwen 12 epochs at 10 seeds (E14, done 2026-10-06)

These numbers are not in the mid-evaluation report. They come after it.

Source: `raw/E14_m1_analysis.txt`, written by
`python -m preregistered_analyses.e14_m1_analysis > ../docs/raw/E14_m1_analysis.txt` (CPU, ~1 min; needs
`Q8_fullq_lora`, `Q8_fullq_lora_12ep` and `Q8_12ep_crossfit` from the HF repo). Section letters refer to
that file.

| Number | Value | Section |
|---|---|---|
| single model S2, 12 epochs, seeds 0–9 | 0.525 ± 0.042 (3 epochs: 0.476 ± 0.043) | A |
| paired S2 gain over 3 epochs | +0.049 ± 0.062, t = 2.48, 8/10 seeds up | A |
| single model S1, 12 epochs | 0.729 (3 epochs: 0.709); paired +0.020, 6/10 up | A |
| 10-seed system S2 (ensemble + LA, nested CV split 0 / mean of 10 splits) | **0.598 / 0.601** (3 epochs: 0.543 / 0.549) | A |
| 10-seed system S1 | 0.743 (3 epochs: 0.746) | A |
| system difference, bootstrap over items | +0.057, 95% [−0.003, +0.125], P(≤0) = 0.032 | A |
| system on the three 2-annotator reference sets | 0.540 / 0.571 / 0.536, mean 0.549 (3 epochs: mean 0.524) | A |
| mean top probability on dev | 0.839 (3 epochs: 0.542) | A |
| system S2 vs number of seeds K | K=1 0.509, K=3 0.564, K=5 0.555, K=10 0.598 | B |
| top-3 coverage: slice / cross-fitted train / dev any-reference | 0.812 / 0.768 / 0.925 | C |
| mass-based sets (APS, α = 0.10): coverage slice / cross-fitted train, mean size | 0.977 / 0.756, 5.94 labels on the slice | C |
| u(x) AUROC: logistic regression vs 1 − top probability (slice CV / dev) | 0.683 / 0.707 vs 0.714 / 0.712 | D |
| oracle within top-3, deferral δ = 0.1 / 0.2 / 0.3 | 0.635 / 0.684 / 0.727 (M1 alone 0.598) | E |
| random within top-3 at δ = 0.3 | 0.587 | E |

Per-run numbers (selected epoch, slice F1, dev S2/S1, time) are in `runs/<config>/<seed|fold>/metrics.json`,
tabulated in the log under "E14 results".

## 11. Phase 2, module M3: the LLM adjudicator (E15, done 2026-10-14)

These numbers are not in the mid-evaluation report. The dev-system numbers are the corrected ones (see
"E15 correction" in the log).

Sources, all in `docs/raw/`, regenerated from the saved responses in `runs/M3/<tag>/responses.jsonl`
(CPU, about 1 min each, from `code/`):
- `E15_screen_local_k.txt`: `python -m cascade.fuse screen --tags qwen38_27b_k0,qwen38_27b_k4,qwen38_27b_k10`
- `E15_screen_models.txt`: the same, with `--tags qwen38_27b_k0,qwen38_27b_api_k0,gemma4_31b_k0,deepseek_v4flash_k0`;
  the faithfulness line at its end comes from the snippet logged in E15
- `E15_slice_qwen_k0.txt`: `python -m cascade.fuse slice --tags qwen38_27b_k0`
- `E15_dev_<tag>.txt`: `python -m cascade.fuse dev --tag <tag>`
- `E15_api_probe.txt`, `E15_api_probe_long.txt`: `cascade.api_probe` (makes API calls)

| Number | Value | Source |
|---|---|---|
| screen (300 cross-fitted train rows): M1 top-1 / gold in top 4 | 0.293 / 0.810 | `E15_screen_local_k.txt` |
| screen, local Qwen fused accuracy at k = 0 / 4 / 10 examples | **0.353** / 0.317 / 0.330 | `E15_screen_local_k.txt` |
| screen fused accuracy: Qwen API / Gemma 4 31B / DeepSeek V4 Flash (k = 0) | 0.373 / 0.363 / 0.317 | `E15_screen_models.txt` |
| local vs API Qwen: argmax agreement, mean JS divergence | 0.933, 0.0035 | `E15_screen_models.txt` |
| slice, 104 deferred items: M1 / LLM alone / fused accuracy | 0.413 / 0.337 / 0.346 | `E15_slice_qwen_k0.txt` |
| dev, LLM alone within top 4: S2, in-set rate | 0.603, 0.643 | `E15_dev_qwen38_27b_k0.txt` |
| **dev system, δ = 0.3 (0.357 calls/item): S2 split 0 / 10 splits** | **0.669 / 0.660** | `E15_dev_qwen38_27b_k0.txt` |
| Δ S2 vs M1, bootstrap over items | +0.067, 95% [+0.009, +0.124] | same |
| dev system, δ = 0.3: S1; Δ S1 vs M1 | **0.798**; +0.055 [+0.004, +0.109] | same |
| dev system, δ = 0.3: 2-annotator S2 mean; in-set rate | 0.589; 0.701 (M1 0.623) | same |
| δ = 0.3: items moved into / out of the reference set | 33 / 9 (exact McNemar p = 0.0003) | same, end |
| M2 curve, 10-split S2 at δ = 0.1 / 0.2 / 0.5 / 1 | 0.602 / 0.640 / 0.628 / 0.609 | same |
| secondary arms at δ = 0.3, S2 (10 splits): Gemma / DeepSeek | 0.625 (0.620) / 0.591 (0.590) | `E15_dev_gemma4_31b_k0.txt`, `E15_dev_deepseek_v4flash_k0.txt` |
| cost: local GPU time per LLM call; prompt tokens | 0.155 GPU-s; ~1.2k in, 1 out | `E15_screen_local_k.txt`, `runs/M3/qwen38_27b_k0` |
| API spend for all of E15 | $0.262 | `runs/M3/api_ledger.jsonl` |

## 12. Phase 2: M2 completion (E16) and M4 distillation (E17), done 2026-10-14/15

Sources, in `docs/raw/` unless marked:
- `E16_prompt_ablation_screen.txt`, `E16_dev_qwen38_27b_k0_<variant>.txt` (prompt ablation);
- `runs/M3/cost.json` (cost);
- `E16_frontier.txt` (`python -m cascade.frontier analyse`);
- `E17_m4_analysis.txt` (`python -m preregistered_analyses.e17_m4_analysis`);
- `E17_frontier_students.txt` (`python -m cascade.frontier analyse --run Q8_distill_a05 --ks 1`);
- the 3-seed screen versions are `*_screen_seeds012.txt`.

| Number | Value | Source |
|---|---|---|
| dev S2 at δ = 0.3, 10 splits: full / no guide / no definitions / bare prompt | 0.660 / 0.637 / 0.628 / 0.617 | `E16_dev_*`, `E15_dev_qwen38_27b_k0.txt` |
| GPU-s per item: M1 seed / M3 call (lab card, GPU 0 alone) | 0.052 / 0.103 | `runs/M3/cost.json` |
| frontier, S2 at δ = 0 / 0.3 / 1 for K = 1 (mean of 10 subsets) | 0.533 / 0.558 / 0.593 | `E16_frontier.txt` |
| frontier, S2 at δ = 0.3 for K = 2 / 3 / 5 | 0.579 / 0.583 / 0.594 | same |
| student single model S2 / S1 (10 seeds) vs control | 0.573 / 0.761 vs 0.525 / 0.729; +0.048 (9/10 up) | `E17_m4_analysis.txt` A |
| student mean top probability; AUROC | 0.578 (control 0.839); 0.686 (0.686) | same, B |
| student + M3, S2 at δ = 0 / 0.3 / 0.5 / 1 | 0.570 / 0.591 / **0.608** / 0.612 | same, C–D |
| student + M3, S1 at δ = 0.5; GPU-s per item | **0.790**; 0.111 | same, D |
| undistilled points dominated by a student point | 25 of 30 | same, D |

## 13. Freeze: final numbers for both operating points (E18, done 2026-10-15)

Sources:
- `raw/E18_final.txt`, written by `python -m preregistered_analyses.e18_final` (CPU, ~2 min);
- `raw/E18_frontier_student_ensembles.txt`, written by
  `python -m cascade.frontier analyse --run Q8_distill_a05 --ks 1,2,3,5,10 --out _ensembles`;
- the figure `docs/figures/E18_frontier.{pdf,png}`.

| Number | Value | Section of `E18_final.txt` |
|---|---|---|
| accuracy point S2 split 0 / 10 splits / S1 / 2-annotator | 0.669 / 0.660 / 0.798 / 0.589 | 1 |
| cost point S2 split 0 / 10 splits / S1 / 2-annotator (mean ± sd, 10 students) | 0.608 ± 0.027 / 0.610 ± 0.023 / 0.790 ± 0.019 / 0.567 ± 0.024 | 2 |
| cost point minus undistilled single model + M3, paired over seeds (S2, S1) | +0.033 (8/10), +0.035 (9/10) | 2 |
| GPU-s per item: M1 × 10 / accuracy point / student / cost point | 0.522 / 0.558 / 0.052 / 0.111 | 4 |
| input tokens per item: M1 × 10 / accuracy point / cost point / TeleAI | 4,894 / 5,284 / 1,117 / ~7,300 | 4 |
| student ensembles, S2 alone for K = 1 / 2 / 3 / 5 / 10 | 0.570 / 0.591 / 0.582 / 0.587 / 0.575 | `E18_frontier_student_ensembles.txt` |


## 14. Robustness and corrections (E19 Tier 0, done 2026-10-27)

Source: `raw/E19_tier0.txt`, written by `python -m preregistered_analyses.e19_tier0` (CPU, ~7 min). Where
these numbers differ from §11–13, they replace them. The earlier numbers stay above as computed at the
time.

| Number | Value | Section of `E19_tier0.txt` |
|---|---|---|
| accuracy point minus M1, S2: bootstrap with fixed / refitted parameters | +0.067 [+0.009, +0.124] / **+0.037 [−0.044, +0.120]**, P(≤ 0) 0.199 | A.2 |
| cost point minus M1, S2, refitted | +0.007 [−0.088, +0.123] | A.2 |
| M3 GPU-s per call: `cost.json` / one pass over the 308 dev items | 0.103 / **0.151** | B.1 |
| GPU-s per item: M1 × 10 / accuracy point / student / cost point | 0.522 / **0.576** / 0.052 / **0.139 (0.27× M1)** | B.2 |
| one-off training cost, GPU-h: M1 (10 seeds) / cost point | 12.5 / 7.0 | B.3 |
| cascade on the 3,448 cross-fitted train rows, δ = 0.3: macro-F1 gain over M1 + LA | +0.014 [+0.002, +0.026] | C |
| net fixes of M3 by M1-uncertainty fifth, train (Q1 = most confident) | −41 / −76 / −12 / +5 / +31 | D |
| S2 at δ = 0.3: confidence / random routing (200 draws) / gold-informed routing | 0.669 / 0.604 ± 0.015 / 0.732 | E.1 |
| fusion at δ = 0.3: fitted / λ = 0 (M3 only) / λ = 1 (M1 only) | 0.669 / 0.631 / 0.581 | E.2 |
| in-set gain of the accuracy point, interviews in train / not in train | +0.093 / +0.068 | F |
| share of M3's picks at letter A (dev) | 0.516 | G |
| human (one annotator against two) / accuracy point / M1, 2-annotator regime | 0.684 / 0.589 / 0.549 | I |

## 15. The final push: E19 arms, E20–E23 (done 2026-10-29/30)

Sources:
- `raw/E21_E22_analysis.txt` (`preregistered_analyses/e21_e22_analysis.py`);
- `raw/E20_router.txt` (`preregistered_analyses/e20_router.py`);
- `raw/E23_systems.txt` (`preregistered_analyses/e23_systems.py`);
- `raw/E23_final_submission.txt` (`evaluation/make_final_submission.py`).

All 27B numbers here come from one copy of the model (the JarvisLabs VM).

| Number | Value | Source |
|---|---|---|
| VM vs lab 27B, dev: argmax agreement, mean max \|Δq\| | 0.925, 0.038 | `E21_E22_analysis.txt` D |
| **final accuracy point** (VM, order-averaged q): S2 split 0 / 10 splits / S1 | **0.650 / 0.639 / 0.788** | `E23_systems.txt` S1 |
| accuracy point, VM single order | 0.639 / 0.637 / 0.787 | same |
| **final cost point** (mean of 10 students): S2 / 10 splits / S1 | **0.608 / 0.606 / 0.785** | `E23_systems.txt` S2 |
| LLM alone over 9 labels: dev S2 / S1 | 0.591 / 0.750 | `E21_E22_analysis.txt` A |
| letter-A pick rate, taxonomy order vs shifted orders (screen) | 0.40 vs 0.22–0.29 | same, B |
| order-averaged vs single-order q (screen): accuracy, NLL | 0.343 vs 0.327; 1.431 vs 1.476 | same, B |
| students, single model S2: M3- / M1- / uniform-taught (seeds 0–2) | 0.575 / 0.563 / 0.516 | HF `metrics.json`; log §E19–E22 results |
| students + M3 at δ = 0.5, 10-split S2: M3- / M1- / uniform-taught | 0.615 / 0.589 / 0.603 | `E23_systems.txt` S3 |
| value map, leak-free train rows (Q1 → Q5 net fixes) | 0 / −7 / +2 / +7 / +12 | `E20_router.txt` S1 |
| learned router vs confidence, macro-F1 at δ = 0.3 | 0.456 vs 0.479 | `E20_router.txt` S4 |
| test artifacts injected into dev: M1 predictions changed; S2 change | 15.6%; −0.048 | `E21_E22_analysis.txt` E |
| 27B call: original vs cache-aligned prompt (GPU-s); argmax agreement | 0.101 vs 0.050; 0.906 | same, C |
| specificity: Spearman with clarity order (train); AUROC Clear Reply | +0.095 [+0.060, +0.131]; 0.566 | same, F |
| ladder, primary: 10-split S2, GPU-s; minus random escalation | 0.600, 0.221; +0.018 | `E23_systems.txt` S4 |

## 16. After the shared task: is evasion in the question or the answerer? (E24–E26, done 2026-10-30/31)

Sources:
- `raw/E24_prechecks.txt` (`preregistered_analyses/e24_prechecks.py`, CPU);
- `raw/E25_analysis.txt` (`preregistered_analyses/e25_analysis.py`, CPU, on the outputs in `runs/E25/`);
- `raw/E26_analysis.txt` (`preregistered_analyses/e26_analysis.py`, CPU, on the outputs in `runs/E26/`).

Every E25/E26 label comes from the frozen E23 accuracy point (`code/experiments/e25/frozen_system.json`);
nothing is refitted on the new inputs. The hypotheses and their thresholds were registered before each run
(log §E24–E26). "Persistence" = share of items whose label is unchanged when the question is swapped.

| Number | Value | Source |
|---|---|---|
| train sibling pairs (same answer, different sub-question): different S2 label | 61.5% of 3,079 | `E24_prechecks.txt` P1 |
| P(two annotators' labels differ): same item / sibling items / random items | 0.432 / 0.716 / 0.834 | P2 |
| M1 predicts the same label on dev siblings with disjoint gold sets | 19.0% (11/58) | P2 |
| routing: M1 errors caught in the top 30%, total / aleatoric / epistemic uncertainty | 53 / 53 / 49 of 115 | P3 |
| question–answer specificity (PMI), AUC Clear Reply vs rest | 0.566 | P8 |
| modules rejected by their pre-checks (P3, P4, P5, P6, P7) | 5 | P3–P7 |
| M1 dev control on the H100 (registered tolerance 1e-3) | max \|dp\| 0.196: failed, amended | `E25_analysis.txt` A |
| instrument on dev: H100 27B calls vs frozen predictions, agreement | 0.981 | A |
| **H25a-1** Clear-Reply share, dev: own question / random question; persistence | **0.253 / 0.013**; 0.188 (held) | B |
| same, test (label-free replication) | 0.338 / 0.110; 0.367 | B |
| persistence with the question removed (null) / question only (qonly) | 0.354 / 0.143 | B |
| LLM answers analysed (2 models × 4 conditions × 308) | 2,464 | C |
| evasion rate, pooled: N / P / F / B; Presidents (gold) | 0.549 / 0.518 / 0.201 / 0.547; 0.744 | C |
| **H25b-1** odds ratio, N (neutral analyst) | 1.18 [0.80, 1.73] (failed) | C |
| odds ratio, P / F / B (exploratory) | 2.74 [1.87, 4.17] / 2.19 / 4.13 | C |
| H25b-2 stake, P − N | −0.031 [−0.081, +0.022] (failed) | C |
| **H25b-3** pursuit: evasive P answers that F turns into a Clear Reply | **0.661** of 319 (held) | C |
| H25b-4 information, B: indirect minus non-answer questions | +0.113 [−0.006, +0.225] (failed) | C |
| H25c answer prefixes, dev 50%: agreement with full-answer label; S2 drop | 0.662; 0.150 (failed) | D |
| H25d-1 frozen-base probe, slice-chosen layer 28, dev S2 | 0.397 (failed) | E |
| **H25d-2** its same-label rate on disjoint siblings (M1 0.19) | 0.448 (held) | E |
| label-free check: mean u / deferred, Presidents vs LLM answers | 0.540 / 0.360 vs 0.462 / 0.293 | G |
| FP8 vs 16-bit, RTX PRO 6000: dev label agreement (10 seeds) | 0.961 | `E26_analysis.txt` rtxpro |
| FP8: M1 system S2 (nested CV, 10 splits); full system S2 | 0.599 → 0.579 (−0.020); 0.653 → 0.649 (−0.003) | rtxpro |
| **H26-4** FP8's M1 flips that fall on items deferred to the 27B | **0.706** of 17; 16 system flips (held) | rtxpro |
| FP8: batched throughput; batch-1 latency p50 | 0.83×; 33.6 → 30.2 ms | rtxpro |
| H26-1 fidelity / H26-2 speed | failed / failed | Registered hypotheses |
| NF4 vs 16-bit, 2 × RTX A4000: dev label agreement (10 seeds) | 0.867 | a4000 |
| NF4: M1 system S2 (nested CV, 10 splits); throughput; peak memory | 0.598 → 0.565 (−0.034); 0.82×; 0.67× | a4000 |
| NF4 full system with the RTX PRO run's 27B records (exploratory, lower bound) | 0.624 → 0.615 (−0.010); 60 of 136 deferred items unscored | a4000 |
| NF4's M1 flips on deferred items (exploratory) | 0.489 of 45 | a4000 |
| QLoRA trigger (> 0.02 S2 or > 5% flips) | met | Registered hypotheses |
| GPU spend: E25 / E26 FP8 (JarvisLabs) | ₹555 / ₹90 | log §E25, §E26 |
