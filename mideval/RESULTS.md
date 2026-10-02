# Results — every number with its source

All on the 308-item dev set. "S2" = Subtask 2 (9 classes, multi-reference macro-F1);
"S1" = Subtask 1 (3 classes, derived from the 9-way prediction).

**Abbreviations.** LA = logit adjustment: divide each class's probability by its training
frequency raised to τ, the one fitted parameter. Nested CV = τ fitted and scored by 5-fold nested
cross-validation on dev, so the reported score is held-out. 2-fold = τ (or temperature + 9 biases)
fitted on one half of dev and scored on the other, both directions.

The main sources are:
- `analysis/mideval_analysis.txt`, written by `python clarity/mideval_analysis.py` (CPU, 24 s).
- `clarity/reports/02_experiment_log.md` (the log), and the raw outputs it cites in
  `clarity/reports/raw/`.

## 1. Main table

Source: `analysis/main_table.csv` (section D of `analysis/mideval_analysis.txt`).

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
- **The dev-A/dev-B split** is in `splits/dev_A_B.json` (stratified by majority label, seed 2026).
  The 2-fold rows are reported for the protocol in the team brief. Nested CV over all 308 items is
  the less noisy estimate: the 9-bias rule swings from 0.29 to 0.46 between halves.

## 2. Per-model comparisons, paired by seed

Source: the log, §E11 (`clarity/reports/raw/E11_replication.txt`).

| comparison, 10 seeds | S2 | S1 |
|---|---|---|
| 16 epochs − baseline | +0.047 (t = 3.4; 9/10 up) | +0.026 (t = 2.7; 8/10) |
| full question − 16 epochs | +0.022 (t = 2.0; 7/10) | +0.012 (t = 1.3; 6/10) |
| full question 16 ep − baseline | **+0.069 (t = 3.6; 9/10)** | **+0.038 (t = 2.8; 9/10)** |
| all of train − full question, 5 seeds (log §E12) | +0.029 (4/5) | −0.017 (1/5) |

Figure: `figures/per_seed_progression.png` (`python clarity/mideval_figures.py`).

## 3. Bootstrap intervals

1,000 resamples over dev items, seeds held fixed. Source: `analysis/mideval_analysis.txt` §E.

| | estimate | 95% interval |
|---|---|---|
| FINAL ensemble, no tuning, S2 | 0.409 | [0.306, 0.472] |
| FINAL ensemble, no tuning, S1 | 0.621 | [0.542, 0.695] |
| FINAL ensemble, dev-B only, S2 | 0.444 | [0.299, 0.519] |
| FINAL − baseline ensemble, no tuning, S2 | +0.090 | [+0.009, +0.168] |
| FINAL − baseline ensemble, no tuning, S1 | +0.024 | [−0.054, +0.098] |

## 4. Floors

Source: `analysis/trivial_baselines.csv`.

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
| **ours, final system** | dev | **0.648** | **0.405** | above |
| **ours, single model** | dev | **0.614** | **0.384** | above |
| human annotator vs the other two (2-annotator scoring) | dev | — | 0.643–0.766 (mean 0.684) | `analysis/mideval_analysis.txt` §G |
| TeleAI | test | 0.89 | 0.68 | task overview, Table 3 |
| Llama-70B fine-tuned (organisers) | test | 0.82 | 0.57 | task overview, Table 2 |

## 6. Error analysis

The FINAL 10-seed ensemble, argmax, no dev tuning. Source: `analysis/mideval_analysis.txt` §F–G.

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

The per-class table is `analysis/per_class_final.csv`, the confusion matrix
`analysis/confusion_final.png` and `.csv`, and ten sampled errors `analysis/error_examples.md`.

## 7. The proposal's C4 prediction

Source: `analysis/mideval_analysis.txt` §H. LA with nested CV vs argmax, same ensemble.

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
| Explicit | −0.082 |
| Dodging | −0.086 |

## 8. Pre-registered hypotheses that were tested and refuted

| hypothesis | result | source |
|---|---|---|
| C2: annotators agree more under the coverage cut | α 0.503 vs 0.623 for the official cut; Δ −0.120, 95% CI [−0.189, −0.049], p = 0.0004 | team's earlier analysis track, in git history; log "Starting point" |
| E10's 0.478 system replicates | 0.434 on seeds 5–9; 0.405 at 10 seeds | log §E11 |
| a mixed-input ensemble beats a single-input one | 0.403 vs 0.405 | log §E11 |
| a dedicated Non-Reply gate raises recall | recall −0.039 | log §E12b |
| a model soup lands between one model and the ensemble | 0.267, below a single model | log §E12d |

## 9. Qwen3-8B LoRA classifier

**Done: 3-seed screen, 2026-10-02.** The setup is identical to E10_fullq_16ep except the
backbone: Qwen3-8B-Base with LoRA r=16, 3 epochs, epoch chosen on the train slice. Single models
on dev, paired by seed with E10_fullq_16ep.

| seed | dev S2 | dev S1 | DeBERTa S2 / S1 | paired Δ S2 / S1 |
|---|---|---|---|---|
| 0 | 0.405 | 0.647 | 0.344 / 0.591 | +0.061 / +0.056 |
| 1 | 0.426 | 0.687 | 0.370 / 0.627 | +0.056 / +0.060 |
| 2 | 0.554 | 0.728 | 0.428 / 0.640 | +0.126 / +0.088 |
| **mean** | **0.462 ± 0.080** | **0.688 ± 0.040** | 0.381 / 0.619 | **+0.081 / +0.068** |

Sources:
- `clarity/reports/raw/E13_Q8_fullq_lora_summary.txt`: per seed and per epoch, with an
  independent recomputation from the saved probabilities;
- `clarity/reports/raw/E13_Q8_fullq_lora_analysis.txt`: per class;
- the experiment log, §E13 results.

How to read this:
- **The screening bar (+0.015, 2 of 3 seeds up) is passed.** S2 lands inside the pre-registered
  0.43–0.50, and S1 is above its predicted +0.02 to +0.05.
- **These are three single models, not a system.** The seed spread is 0.080. A system claim
  waits for 10 seeds.
- The selected epoch was the last one on every seed, so 3 epochs may undertrain.
