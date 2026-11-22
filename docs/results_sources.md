# Results: every number with its source

All on the 308-item dev set. "S2" = Subtask 2 (9 classes, multi-reference macro-F1);
"S1" = Subtask 1 (3 classes, derived from the 9-way prediction).

**Abbreviations.** LA = logit adjustment: divide each class's probability by its training
frequency raised to τ, the one fitted parameter. Nested CV = τ fitted and scored by 5-fold nested
cross-validation on dev, so the reported score is held-out. 2-fold = τ (or temperature + 9 biases)
fitted on one half of dev and scored on the other, both directions.

The main sources are:
- `raw/mideval/mideval_analysis.txt`, written by `python -m preregistered_analyses.mideval_analysis` (CPU, 24 s).
- `docs/experiment_log.md` (the log), and the raw outputs it cites in
  `docs/raw/`.
- For the Qwen classifier (§9): `docs/raw/E13_final_analysis.txt`, written by
  `python -m preregistered_analyses.e13_analysis` (CPU, about 3 min).

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
| NF4: M1 system S2 (nested CV, 10 splits); throughput (bf16 batch 8 on two cards, NF4 batch 16 on one); peak memory | 0.598 → 0.565 (−0.034); 0.82×; 0.67× | a4000 |
| NF4 full system with the RTX PRO run's 27B records (exploratory, lower bound) | 0.624 → 0.615 (−0.010); 60 of 136 deferred items unscored | a4000 |
| NF4's M1 flips on deferred items (exploratory) | 0.489 of 45 | a4000 |
| QLoRA trigger (> 0.02 S2 or > 5% flips) | met | Registered hypotheses |
| GPU spend: E25 / E26 FP8 (JarvisLabs) | ₹555 / ₹90 | log §E25, §E26 |

## 17. Inference cost on one GPU (final report, 2026-10-31)

Source: `raw/cost_same_gpu.txt`, written by `python -m preregistered_analyses.cost_same_gpu` (CPU, seconds). It replaces
the cost numbers of §12–§15 in the final report. Two corrections:
- **One GPU.** Every time comes from the E22a lane (one RTX PRO 6000 Server Edition, one session, one batched pass over
  the 308 dev items): M1 seed 0.0397 GPU-s per item (`runs/E22/cost_jarvis.json`), 27B call 0.1013 (taxonomy order)
  and 0.0978 (each shifted order). The E18/E19 tables mixed an M1 seed timed alone on the lab card (E16) with 27B calls
  from other sessions (0.103 with prefix reuse, or 0.151 from the E15 night run).
- **Four calls per deferred item.** The frozen system averages q over four option orders, so its 27B calls per item are
  4 x the deferral rate. E23's tables reported the deferral rate (one call per deferred item).

| Number | Value | Section |
|---|---|---|
| M1 seed / 27B call (one order) / four orders | 0.0397 / 0.1013 / 0.3947 GPU-s | A |
| M1 x 10 | 0.397 GPU-s, 0 calls | B |
| accuracy point, 4 orders (frozen instrument) | 0.650 / 0.788; 1.43 calls; ~1,591 27B input tokens; 0.538 GPU-s (1.36x M1) | B |
| accuracy point, 1 order | 0.639 / 0.787; 0.36 calls; 0.433 GPU-s (1.09x) | B |
| distilled point, 4 orders / 1 order | 0.608 / 0.599 S2 (10 splits 0.606 both); 2.30 / 0.57 calls; 0.266 / 0.098 GPU-s (0.67x / 0.25x) | B |
| escalation ladder (E22c primary) | 10-split 0.600; 0.163 GPU-s (0.41x) | B |
| M1's share of the instrument's GPU time | 0.74 (single order 0.92) | B |
| undistilled frontier points beaten by a student point on this scale | 25 of 30 (unchanged) | C |
| lab-card ratios (E19) vs this GPU | 1.10x vs 1.09x; 0.27x vs 0.25x | D |

## 18. Final-report checks (2026-10-31)

Source: `raw/final_checks.txt`, written by `python -m preregistered_analyses.final_checks` (CPU, seconds). Descriptive
checks written after the results were known; none is a registered test.

| Number | Value | Section |
|---|---|---|
| two option orders of the same four candidates give the same pick (dev / screen) | 0.720 / 0.672; all four orders 0.500 / 0.410 | A |
| averaged-q accuracy vs the mean of four single-order accuracies (screen) | 0.343 vs 0.327 | A |
| lab copy vs cloud copy of the 27B, same pick on dev | 0.925 (23 of 308 differ) | B |
| agreement by decisiveness tercile (least to most) | 0.786 / 0.990 / 1.000 | B |
| cloud-lane 'jarvis' vs 're-score' records | identical on 308 of 308, so not an independent repeat | B |
| evasion rate where the President evaded / answered: N, P, F, B | 0.559/0.519, 0.581/0.335, 0.229/0.120, 0.635/0.291 | C |
| N and P give the same evade label on the same question (Mistral / Qwen; chance) | 0.666 (0.503) / 0.734 (0.500) | C |
| FP8: M1 flips (deferred / kept), system flips | 17 (12 / 5), 16 | D |
| FP8 flips that change reference-set membership: M1, system | loses 10, gains 0; loses 5, gains 7 | D |
| FP8 S2 with frozen parameters, in sample: M1, system | 0.617 to 0.572 (-0.045); 0.653 to 0.649 (-0.003) | D |
| NF4 S2 with frozen parameters, in sample: M1, system (lower bound, 60 records missing) | -0.029; -0.010 | D |


## 19. Question-side prior, preliminary check (final report, 2026-10-31)

Source: `raw/question_prior_check.txt`, written by `python -m preregistered_analyses.question_prior_check` (CPU, seconds).
Descriptive and written after the E25 results were known; dev only (308 items). It motivates the future-work module M5
and is not a test of it.

| Number | Value | Section |
|---|---|---|
| the President evaded, by the share of the two P-role LLMs whose answer is labelled evasive (0 / 0.5 / 1) | 0.614 (n = 101) / 0.716 (95) / 0.884 (112) | A |
| AUROC for "the President evaded": M1 (1 − p(Explicit)) / P-role share; correlation | 0.880 / 0.661; 0.312 | A |
| cross-validated AUROC, all items: M1 / M1 + P-role share | 0.872 / 0.875 | A |
| cross-validated AUROC, least sure 36% (n = 111): M1 / M1 + P-role share | 0.722 / 0.746 | A |


## 20. Improvement checks (2026-11-19; not shown in the final slides)

Sources: `raw/sibling_error_check.txt` (`python -m preregistered_analyses.sibling_error_check`) and
`raw/question_dependence_check.txt` (`python -m preregistered_analyses.question_dependence_check`), CPU, seconds, from
existing E25 outputs. Descriptive and written after E25; dev only (308 items). They motivate the proposed sibling-contrast
module (M6) and the decision not to add a no-question view of M1; they are not tests of either. "Correct" is the
any-annotator rule; the system's labels are the frozen system as replayed in E25 (S2 0.659 there, 0.650 in E23).

| Number | Value | Source |
|---|---|---|
| dev items sharing their answer with another sub-question; shared answers | 203 of 308; 86 | sibling |
| frozen system correct: answer given to one sub-question / shared answer | 0.724 (n = 105) / 0.655 (n = 203) | sibling |
| system errors on shared answers whose label is a sibling's reference label; chance | 23 of 70 (32.9%); 18.2% | sibling |
| the same for M1 alone; chance | 20 of 78 (25.6%); 16.0% | sibling |
| same label on sibling pairs with disjoint reference sets: system / M1 | 22.4% / 15.5% of 58 | sibling |
| mean p(Explicit): own question / no question / another interview's question | 0.228 / 0.463 / 0.017 | dependence |
| AUROC for "the President evaded": no-question gap alone; correlation of the gap with M1 | 0.842; 0.870 | dependence |
| cross-validated AUROC, least sure 36% (n = 111): M1 / M1 + gap / M1 + P-role share / all three | 0.722 / 0.708 / 0.746 / 0.728 | dependence |


## 21. Workshop-paper checks (2026-11-20)

| Number / fact | Value | Source |
|---|---|---|
| test labels released? | no: the public HF `ailsntua/QEvasion` "test" split (308 rows, 3 annotators) is our dev set; the task paper never mentions a release of the 237-item test gold | checked 2026-11-20 (HF dataset card and rows API; `paper/Task_Results_Paper.pdf`) |
| registered teams in the shared task | 124 | `paper/Task_Results_Paper.pdf` |
| dev `president` / `date` fields | empty for all 308 rows (train: filled); the E25b prompts rendered them as "nan" | `data/cache/dev.parquet`; `code/audit/generate.py`; log E27–E32 "Defect found" |
| E25b answers containing the token "nan" | 17 of 2,464 (Mistral 13, Qwen 4) | `raw/E31_role_robustness.txt` |
| relevance (PMI_c) vs identifiability AUROC for Clear Reply (train) | 0.566 (Kruskal–Wallis p = 2.3e-12) vs 0.549 | `raw/E24_prechecks.txt` P8 (the 0.566 is PMI_c, not identifiability) |
| Figure 1 example (dev rows 147, 206): labels under own / swap / no question / contentless answer | Explicit / Dodging / Explicit / Dodging (row 147); Dodging / Dodging / Explicit / Dodging (row 206) | `raw/paper_fig1_example.txt` (`python -m figures.paper_example_frames`) |
| LLM evasion rate where the President evaded / answered, per model and condition | e.g. Mistral N 0.590 / 0.595, P 0.585 / 0.316; Qwen N 0.528 / 0.443, P 0.576 / 0.354 | `raw/paper_audit_rates.txt` (`python -m figures.paper_figures_v2`) |


## 22. E27 counterfactual question contrast (registered 2026-11-20; FAILED)

Source: `raw/E27_cqc.txt`, `runs/E27/cqc.json` (`python -m preregistered_analyses.e27_cqc`, CPU, ~30 s). M1 = 10-seed
mean on the E25 frames; α and τ nested (5 folds × 10 splits); turn-cluster bootstrap (2,000).

| Number | Value |
|---|---|
| M1 level, no-question contrast: S2 baseline → CQC (Δ, 95% interval) | 0.584 → 0.567 (−0.017 [−0.046, −0.002]) |
| M1 level, contentless-answer contrast | 0.584 → 0.579 (−0.005 [−0.011, +0.000]) |
| cascade (dev-rank deferral 35.7%), both contrasts | 0.605 → 0.599 (−0.006) |
| S2 at fixed α = 0 / 0.5 / 1 (no-question contrast, M1) | 0.584 / 0.523 / 0.347 |
| share of folds choosing α = 0 (M1 level) | 0.78 (no question), 0.88 (contentless) |
| same label on the 58 disjoint sibling pairs, M1 | 0.155 → 0.172 |


## 23. E28 sibling contamination (registered 2026-11-20; FAILED)

Source: `raw/E28_contamination.txt`, `runs/E28/contamination.json` (`python -m preregistered_analyses.e28_contamination`,
CPU, ~10 s). 1,000 length-tercile permutations; 2,000 bootstrap draws over the 86 shared answers.

| System | errors on shared answers | contamination | length-matched null (p) | uniform chance |
|---|---|---|---|---|
| DeBERTa-v3-large (10 seeds) | 96 | 0.323 | 0.387 (0.944) | 0.183 |
| M1 (10 seeds) | 79 | 0.253 | 0.339 (0.977) | 0.161 |
| M4 student (10 seeds) | 78 | 0.385 | 0.415 (0.765) | 0.174 |
| 27B alone over M1's top-4 | 88 | 0.307 | 0.359 (0.896) | 0.184 |
| frozen cascade (E23) | 68 | 0.324 | 0.426 (0.984) | 0.185 |
| frozen cascade, E25 replay (anchor = preliminary check) | 70 | 0.329 | 0.426 (0.966) | 0.182 |
| cascade − M1 | | +0.070 [−0.035, +0.179] | | |


## 24. E29 M5 question-side prior (registered 2026-11-20; FAILED)

Source: `raw/E29_m5.txt`, `runs/E29/m5.json` (`python -m preregistered_analyses.e29_m5`, CPU, ~5 s).

| Number | Value |
|---|---|
| deferred and scored dev items (E25 replay) | 111 |
| β = 0 (frozen replay) S2 / S1 | 0.659 / 0.786 |
| in-sample S2 / S1 at β = 1 | 0.650 / 0.777 |
| nested β (objective S2 or S1): share of folds β = 0; Δ S2, Δ S1 | 1.00; +0.000, +0.000 |


## 25. E31 role-effect robustness (registered 2026-11-20; HELD)

Source: `raw/E31_role_robustness.txt`, `runs/E31/role_robustness.json` (`python -m preregistered_analyses.e31_role_robustness`,
CPU, ~5 min). Pooled = Mantel–Haenszel over models; turn-cluster bootstrap (2,000).

| Specification | OR(N) pooled | OR(P) pooled [95%] | OR(P) Mistral / Qwen |
|---|---|---|---|
| (a) raw | 1.18 | 2.74 [1.87, 4.17] | 3.01 / 2.46 |
| (b) MH over model × LLM-answer-length tercile | 1.15 | 2.72 [1.85, 4.15] | 3.05 / 2.44 |
| (c) logistic + log lengths (LLM, President) + model | 1.17 | 2.79 [1.90, 4.32] | 3.03 / 2.64 |
| (d) (c) + question-only prior | 1.16 | 2.81 [1.91, 4.35] | 2.97 / 2.73 |
| (e) raw, without the 17 "nan" answers | 1.18 | 2.89 [1.98, 4.40] | 3.28 / 2.51 |
| (f) raw, President labelled by the instrument | 1.26 | 3.49 [2.34, 5.48] | 4.00 / 2.98 |


## 26. E32 hardware control (2026-11-20)

Source: `raw/E32_analysis.txt` §A (`python -m preregistered_analyses.e32_analysis`), from `runs/E32/labels/ctrl_mistral.npz`
(labelled by `code/launch/e32_label.sh ctrl_mistral e32ctrl`) against `runs/E25/labels/llm.npz`.

| Number | Value |
|---|---|
| instrument label agreement, 200 Mistral answers, lab RTX PRO 6000 vs E25 H100 (M1 alone) | 0.970 (0.990) |
| evade / not-evade agreement; evasion rate E25 vs lab; routing identical | 0.985; 0.480 vs 0.485; 0.995 |


## 27. E30 validity battery (registered 2026-11-20; HELD)

Source: `raw/E30_validity_battery.txt`, `runs/E30/battery.json` (`python -m audit.validity_battery`; DeBERTa frames
by `python -m audit.deberta_frames --seeds 0-2`, the 27B zero-shot by `cascade.adjudicate --tag e30all9_taxonomy` and
then `python -m audit.validity_battery --import-27b e30all9_taxonomy llm27b_zeroshot`). Dev, frozen parameters (S2
in sample). CR = Clear-Reply share; binding = same label on the 58 disjoint sibling pairs.

| System | S2 (own) | CR own / swap / no question / contentless | persistence under swap | binding |
|---|---|---|---|---|
| M1 (E25 labels) | 0.611 | 0.179 / 0.010 / 0.321 / 0.000 | 0.175 | 0.155 |
| frozen instrument (E25 labels) | 0.659 | 0.253 / 0.013 / 0.640 / 0.006 | 0.188 | 0.224 |
| DeBERTa-v3-large, 3 seeds (τ 0.35) | 0.444 | 0.312 / 0.026 / 0.776 / 0.003 | 0.198 | 0.310 |
| Qwen3.8-27B zero-shot, 9 labels (τ −0.20) | 0.604 | 0.377 / 0.026 / 0.289 / — | 0.133 | 0.362 |
| frozen linear probe, layer 28 (`raw/E25_analysis.txt`) | 0.397 | — | — | 0.448 |
| Presidents' gold CR (dev) | | 0.256 | | |
| DeBERTa reproduction control, argmax agreement with saved dev_probs (seeds 0/1/2) | | 0.997 / 0.994 / 0.997 | | |


## 28. E32 OLMo-2 replication (registered 2026-11-20; H32-1 and H32-3 HELD, H32-2 FAILED)

Source: `raw/E32_analysis.txt`, `runs/E32/analysis.json` (`python -m preregistered_analyses.e32_analysis`; answers
by `code/launch/e32_olmo.sh`). Odds ratio = LLM evades | President evaded (gold); turn-cluster bootstrap (2,000 draws).

| Number | Protocol A (E25 prompts) | Protocol B (generic, exploratory) |
|---|---|---|
| evasion rate N / P / F / B | 0.565 / 0.510 / 0.211 / 0.545 | 0.503 / 0.552 / 0.201 / 0.578 |
| OR(N) | 1.37 [0.83, 2.29] | 1.13 [0.69, 1.89] |
| OR(P) | 2.90 [1.81, 5.04] | 2.74 [1.64, 4.70] |
| OR(F) / OR(B) | 1.46 / 3.05 | 1.52 / 3.97 |
| pursuit: evasive P answers that F turns into Clear Replies | 0.631 (n = 157) | 0.676 (n = 170) |
| answers cut at 400 tokens; containing "nan" | 0.097 (both protocols); 3 | 0 |
| MH OR(P) over Mistral, Qwen, OLMo (A); same pooling in N | 2.81 [1.97, 4.16]; 1.24 [0.84, 1.80] | |
| deferred to the 27B; records missing | 717 of 2,464; 0 | |

| E31 on three models (`raw/E31_role_robustness_e32.txt`, `--llm-table runs/E32/llm_table.parquet`) | pooled OR(P) | pooled OR(N) |
|---|---|---|
| (a) raw / (b) length strata / (c) logistic / (d) + question prior / (e) no "nan" | 2.81 / 2.77 / 2.84 / 2.82 / 2.91 | 1.24 / 1.23 / 1.24 / 1.23 / 1.24 |
| (f) President by the instrument | 3.26 [2.22, 4.97] | 1.44 [0.96, 2.09]; OLMo alone 1.87 [1.10, 3.18] |


## 29. Paper v2 facts and the E34 discovery estimate (2026-11-21)

Sources: `raw/paper_facts.txt` (`python -m preregistered_analyses.paper_facts`), `raw/E34_discovery_dev.txt`,
`runs/E34/analysis_dev.json` (`python -m preregistered_analyses.e34_analysis dev`; frozen script, hash in the E34
registration). The discovery estimate is exploratory: the P-vs-N pattern was found after E25 and this ratio was first
computed ad hoc on 2026-11-21, before the E34 registration.

| Number | Value |
|---|---|
| train: items / interviews / turns / dates | 3,448 / 287 / 2,062 / 2006-10-25 to 2023-09-10 |
| train: President evaded (clarity ≠ Clear Reply) | 0.695 |
| dev: items / interviews / turns | 308 / 56 / 191 |
| dev interviews also in train | 39 of 56 (590 train rows) |
| confirmation sample (train interviews not in dev) | 248 interviews, 2,858 items, 1,710 turns; President evaded 0.688 |
| discovery evasion rate N / P (3 models, frozen instrument) | 0.554 / 0.515 |
| discovery ROR P/N (url bootstrap) | 2.27 [1.61, 3.22]; Mistral 3.12, OLMo 2.14, Qwen 1.76 |
| discovery ROR P/N: "you" sub-questions / others | 2.39 [1.43, 4.42] / 2.42 [1.55, 4.11] |
| discovery ROR P/N: GEE / conditional logit | 2.27 [1.61, 3.20] / 4.88 [2.47, 9.66] |
| discovery tipping point (extra symmetric label noise in N that brings ROR to 1) | 0.39 |

## 30. E35 binding with a matched null, dev (registered 2026-11-21)

Source: `raw/E35_binding.txt`, `runs/E35/binding.json` (`python -m preregistered_analyses.e35_binding`). Binding = share of
the 58 dev sibling pairs with disjoint gold sets labelled identically; null = the same system's rate on 2,000 random
pairs from different answers with disjoint gold sets; ratio = binding / null; bootstrap over the 40 answers. S2 is
in-sample (frozen or dev-fitted tau, as in E30).

| System | dev S2 | binding [95% CI] | null | ratio to null |
|---|---|---|---|---|
| Qwen3-8B LoRA (M1, 10 seeds) | 0.611 | 0.155 [0.071, 0.259] | 0.118 | 1.31 |
| shared-task instrument | 0.659 | 0.224 [0.120, 0.328] | 0.135 | 1.66 |
| DeBERTa-v3-large (3 seeds) | 0.444 | 0.310 [0.200, 0.421] | 0.154 | 2.01 |
| 27B zero-shot | 0.604 | 0.362 [0.218, 0.509] | 0.146 | 2.47 |
| frozen probe, layer 28 (layers 4–36: 0.38–0.79; layer 0 predicts one label) | 0.397 | 0.448 [0.319, 0.579] | 0.208 | 2.15 |
| answer only (M1, no question) | 0.359 | 1.000 | 0.221 | 4.52 |
| M1 − 27B zero-shot (paired) | | −0.207 [−0.379, −0.022] | | |
| DeBERTa − 27B zero-shot (paired) | | −0.052 [−0.226, +0.115] | | |

## 31. E33 instrument checks, H33-1 and E35 train binding (registered 2026-11-21; results 2026-11-22)

Sources: `raw/E33_checks.txt` (`python -m audit.cf_tau`), `raw/E33_llm_swap.txt` (`python -m preregistered_analyses.e33_llm_swap`),
`runs/E34/cf/report.json` (`python -m audit.infer_cf --batch 16`), `raw/E35_binding.txt` (`python -m preregistered_analyses.e35_binding`),
`raw/E35x_exploratory.txt` (`python -m preregistered_analyses.e35x_exploratory`, post hoc).

| Number | Value |
|---|---|
| CF fold dev S2 (raw argmax), folds 0–4 | 0.577, 0.574, 0.564, 0.542, 0.480 (mean 0.547) |
| CF reproduces each fold's held-out probabilities | argmax agreement 1.000, max abs dp 0.000 (batch 16) |
| CF τ (frozen); out-of-fold train S2 raw → adjusted | 0.50; 0.434 → 0.435 |
| CF dev S2, mean of 5 folds, τ 0.50 (exploratory) | 0.571 |
| H33-1 CF Clear-Reply share, own → swapped question (dev LLM answers), N / P | 0.418 → 0.000 / 0.530 → 0.000 (HELD) |
| train confirmation sibling pairs with different gold labels | 1,605 (2,858 items) |
| train binding, CF / ZS [95% CI]; null; ratio | 0.282 [0.251, 0.334], 0.141, 2.00 / 0.297 [0.235, 0.389], 0.169, 1.76 |
| train S2 on confirmation rows, CF / ZS | 0.437 / 0.458 |
| H35-1 binding(CF) − binding(ZS) | −0.015 [−0.104, +0.038] (FAILED) |
| dev binding, CF mean of 5 folds (exploratory) | 0.207 [0.109, 0.328], null 0.122, ratio 1.69 |
| dev binding, CF − ZS / CF − M1 (exploratory) | −0.155 [−0.339, +0.038] / +0.052 [−0.071, +0.188] |
| dev binding, single CF fold models (exploratory) | 0.155–0.293 |

## 32. E34 confirmation, E36 drivers, E36x "you" breakdown and the role figure (2026-11-22)

Sources:
- `raw/E34_confirmation_train.txt`, `runs/E34/analysis_train.json` (`python -m preregistered_analyses.e34_analysis train`; rerun byte-identical).
- `raw/E36_drivers.txt`, `runs/E34/e36.json` (`python -m preregistered_analyses.e36_drivers`, exploratory).
- `raw/E36x_you.txt`, `runs/E34/e36x_you.json` (`python -m preregistered_analyses.e36x_you`, exploratory, post hoc).
- `raw/paper_roles.txt`, `runs/E34/paper_roles.json` (`python -m figures.paper_figures_v3`, the values in `fig_roles`).

| Number | Value |
|---|---|
| confirmation answers labelled (5 models x 4 roles) | 57,160 (trio 34,296) |
| H34-1 ROR P/N, trio pooled | 1.25 [1.09, 1.43], one-sided p 0.001 (HELD) |
| H34-1 per model: Mistral / OLMo-2-7B / Qwen3-8B | 1.44 [1.21, 1.72] / 1.34 [1.09, 1.62] / 1.01 [0.82, 1.24] |
| H34-2 ROR S/N | 1.34 [1.19, 1.49] (HELD) |
| H34-3 ROR P/S | 0.93 [0.85, 1.03], Holm p 0.51 (FAILED) |
| H34-4 ROR G/N | 1.22 [1.08, 1.38] (HELD) |
| H34-5 Granite-3.3-8B / OLMo-2-32B | 1.19 [0.98, 1.45], Holm 0.19 / 1.08 [0.89, 1.31], Holm 0.51 (FAILED) |
| H34-6 no-"you" sub-questions | 1.07 [0.90, 1.29] (FAILED) |
| robustness: ZS / GEE / conditional logit / without Qwen | 1.22 [1.06, 1.42] / 1.25 [1.09, 1.43] / 1.35 [1.15, 1.59] / 1.39 [1.20, 1.60] |
| robustness: soft labels / non-answer outcome / full train | 1.26 [1.11, 1.42] / 1.24 [1.01, 1.53] / 1.23 [1.10, 1.38] |
| tipping point (confirmation) | 0.25 |
| H33-2 CF–ZS agreement N / P / S / G | 0.730 / 0.742 / 0.730 / 0.717 (HELD) |
| evasion rate trio pooled, President answered / evaded: N, S, G, P | 0.550/0.597, 0.436/0.556, 0.390/0.486, 0.418/0.521 (`paper_roles.txt`) |
| per-model ROR S/N, G/N, P/N (all 5 models; 15 values) | 1.01–1.53 (`paper_roles.txt`) |
| E36 ROR P/N "you" / other | 1.43 [1.16, 1.75] / 1.07 [0.89, 1.29] |
| E36 per President: Obama / Trump / Bush / Biden | 1.31 / 1.29 / 1.01 / 1.12 (none survives Holm) |
| E36 word overlap with the real answer / a random answer, N, S, G, P | 0.091/0.053, 0.109/0.064, 0.106/0.064, 0.129/0.079 |
| E36 meta answers | ≤ 0.001 in every role |
| E36x "you": analyst rate answered / evaded; OR | 0.621 / 0.646; 1.11 [0.93, 1.32] |
| E36x "you": OR S / G / P | 1.59 / 1.48 / 1.59 |
| E36x other: analyst OR; OR S / G / P | 1.35 [1.11, 1.63]; 1.68 / 1.49 / 1.44 |
| ln(2.27) / ln(1.25) (discovery vs confirmation, log scale; paper §7 "nearly four times") | 3.67 (arithmetic on the two registered outputs) |
| E36x "you" / other ROR ratio, paired draws: P / S / G (post hoc) | 1.34 [1.00, 1.76] / 1.15 [0.87, 1.49] / 1.21 [0.92, 1.51] (`raw/E36x_you.txt`) |
| discovery: Mistral analyst evades on President-answered / evaded questions | 0.595 / 0.590 (`raw/paper_audit_rates.txt`) |

## 33. The discovery estimate with CF, and the packaging rerun (2026-11-22)

Source: `raw/E34_discovery_dev.txt`, `runs/E34/analysis_dev.json` (`python -m preregistered_analyses.e34_analysis dev`).
The first run predated the CF labels and printed "[cf] not available yet"; the rerun adds the CF section and leaves the
frozen-instrument section byte-identical.

| Number | Value | Used in |
|---|---|---|
| discovery ROR P/N with CF (dev, mean of the five fold models, τ 0.50) | 2.03 [1.42, 3.17]; Mistral 1.92, OLMo-2-7B 1.94, Qwen3-8B 2.24 | paper §5; `findings.md` §6c |
| discovery with CF, "you" / other sub-questions | 2.00 [1.11, 4.08] / 2.22 [1.47, 4.03] | paper §6 ("Questions put to you") |
| discovery with CF: GEE / conditional logit / soft labels; tipping point | 2.05 / 4.23 / 1.99; e* = 0.33 | not cited |

