# `clarity/` — the classifiers, experiments and records

An end-to-end fine-tuned encoder for SemEval-2026 Task 6 (CLARITY), built up from
the simplest system that could work and extended one measured step at a time.

It is self-contained: `qevasion/` holds the data loader, the label vocabulary and a replica of
the official scorer.

| | |
|---|---|
| **Best system (E13)** | Qwen3-8B-Base + LoRA (`llm_classifier.py`), same input and protocol, 10-seed ensemble + logit adjustment: dev S2 **0.543**, dev S1 **0.746** |
| Best single model (E13) | Qwen3-8B LoRA: **0.476 ± 0.043** dev S2, **0.709 ± 0.031** dev S1 over 10 seeds; above DeBERTa on 10 of 10 seeds. 12 epochs adopted on the slice (3 seeds: 0.543 / 0.742) |
| DeBERTa final system (E11) | full question + 16 epochs, 10-seed DeBERTa-v3-large ensemble + logit adjustment: dev S2 0.405, dev S1 0.648 (baseline system: 0.428 / 0.601) |
| DeBERTa best single model | full question + 16 epochs: 0.384 ± 0.030 dev S2, 0.614 ± 0.027 dev S1 over 10 seeds (baseline 0.315 / 0.576); better on 9 of 10 seeds |
| The story, in order | [`reports/02_experiment_log.md`](reports/02_experiment_log.md) |
| Latest | **E13 (2026-10-02)**: the backbone swap is the largest gain in the project (+0.092 S2 / +0.095 S1 per model, 10/10 seeds; system +0.136). All of train: +0.017 per model, not distinguishable. See the log §E13–E13c and [`reports/03_research_narrative.md`](reports/03_research_narrative.md) §17 |

---

## Contents

1. [The task in one page](#1-the-task-in-one-page)
2. [Results](#2-results)
3. [How the system works](#3-how-the-system-works)
4. [Key findings](#4-key-findings)
5. [Running it](#5-running-it)
6. [Files](#6-files)
7. [Rules this track keeps](#7-rules-this-track-keeps)

---

## 1. The task in one page

Each example is a triple from a U.S. presidential interview:

| field | what it is | median length |
|---|---|---|
| `interview_question` | the journalist's full turn, often several questions at once | 62 tokens |
| `question` | **one** sub-question, split out of that turn when the dataset was built | 15 tokens |
| `interview_answer` | the politician's **entire** reply | 266 tokens |

The label says how *that sub-question* was answered.

- **Subtask 2** — 9 evasion types: Explicit, Implicit, Dodging, General,
  Deflection, Partial/half-answer, Declining to answer, Claims ignorance,
  Clarification.
- **Subtask 1** — 3 clarity levels, a **fixed function** of Subtask 2:

| clarity | evasion types | share of train |
|---|---|---|
| Clear Reply | Explicit | 30.5% |
| Ambivalent | Implicit, Dodging, General, Deflection, Partial/half-answer | 59.2% |
| Clear Non-Reply | Declining to answer, Claims ignorance, Clarification | 10.3% |

So the system is **one 9-way classifier**; Subtask 1 is a table lookup on its
output, as in the 1st- and 2nd-place systems.

**Data.** train 3,448 (one label each) · dev 308 (three annotators each) · test 237
(two annotators each; labels never released).

**Metric.** Macro-F1 over all nine classes, where a prediction counts as correct if
it matches **any** annotator's label.

---

## 2. Results

All on **dev**: Codabench is closed and the test labels are not public, so dev is
the only place systems can be compared.

| system | dev S2 | dev S1 |
|---|---|---|
| TeleAI (1st), DeepSeek-V3 3-stage CoT pipeline | 0.617 | 0.812 |
| ChulaNLP (2nd), DeBERTa-large top-5 → Kimi-K2 | 0.52 | 0.70 |
| ChulaNLP, DeBERTa-large fine-tuned *(checkpoint chosen on dev)* | 0.46 | 0.65 |
| **ours — Qwen3-8B LoRA, 10-seed ensemble + logit adjustment (E13)** | **0.543** | **0.746** |
| ours — Qwen3-8B LoRA, all of train, 10-seed system (E13c) | 0.575 | 0.711 |
| **ours — Qwen3-8B LoRA, single model (10 seeds)** | **0.476 ± 0.043** | **0.709 ± 0.031** |
| ours — baseline system: 10-seed ensemble + logit adjustment | 0.428 | 0.601 |
| TeleAI, DeepSeek-V3 asked directly for the label | 0.421 | 0.662 |
| ours — DeBERTa final system: full question, 16 epochs, 10-seed ensemble + logit adjustment | 0.405 | 0.648 |
| ours — DeBERTa, all of train, last epoch (E12a), **5-seed** ensemble + logit adjustment; single model 0.416 ± 0.020 | 0.489 | — |
| ours — DeBERTa single model, full question, 16 epochs (10 seeds) | 0.384 ± 0.030 | 0.614 ± 0.027 |
| ours — single model, 16 epochs (10 seeds) | 0.362 ± 0.026 | 0.601 ± 0.016 |
| ours — single model, baseline (10 seeds) | 0.315 ± 0.040 | 0.576 ± 0.030 |

What these numbers do and don't show is in the experiment log (§E7, §E11). In short:

- **Per model, the improvement is solid**: the full question plus 16 epochs beats
  the baseline model on 9 of 10 seeds (S2 +0.069, S1 +0.038).
- **As systems, the two are within noise on Subtask 2**: the decision rule lifts the
  baseline's prior-leaning ensemble by +0.109 and the better model's by nothing.
  The final system was chosen by a rule written before the last experiment ran.
- 5-seed system numbers are unreliable on 308 items (the baseline scored 0.438 and
  0.356 on two seed sets), so all system numbers above use 10 seeds, except
  E12a's, which is a 5-seed result and not a ranking.
- The fair external comparison is ChulaNLP's fine-tuned DeBERTa (0.46 / 0.65),
  whose checkpoint was chosen on dev; ours are held-out.

Ablations that did **not** help, all documented:

| tried | dev S2 | verdict |
|---|---|---|
| per-class decision thresholds (the textbook macro-F1 rule) | 0.394 | overfits 308 items; loses to the one-scalar rule |
| hard hierarchical routing (clarity branch first, then leaf) | 0.369 | no reliable effect |
| a second encoder re-ranking the top-3 | 0.354 | negative — [`reports/03_research_narrative.md`](reports/03_research_narrative.md) §10 |
| full question + Balanced Softmax + focal loss (E8) | 0.379 | negative — at 8 epochs the full-question input is undertrained (see E8b); experiment log §E8 |
| full question, plain cross-entropy (E8b) | 0.343 | negative at 8 epochs, but undertrained; single models match the baseline after logit adjustment — E10 retests at 16 epochs |
| trained hierarchical head, p(level) × p(leaf \| level) (E9) | 0.370 | negative for Subtask 2; Subtask 1 steadier, not better — experiment log §E9 |
| 16 epochs instead of 8 (E10 control) | 0.402 | better single model (0.377 vs 0.337), but it gains in the same rare classes the decision rule was already fixing, so the final system does not improve |
| mixing both inputs in one ensemble (E11) | 0.403 | no gain over 10 full-question models (0.405), tested on fresh seeds |
| hierarchy of specialist encoders: Non-Reply gate + branch specialists (E12b) | 0.390 *(3 seeds)* | negative — the gate routes too confidently; specialists level with the flat model on 5 seeds |
| boundary experts for the three most-confused pairs (E12c) | — | no effect (S2 +0.002, S1 −0.004 per model) |
| model soup of 10 trained models (E12d) | 0.267 | negative — seeds do not average in weight space |

---

## 3. How the system works

```
 sub-question + full question + answer ──► DeBERTa-v3-large (fine-tuned)  ──► p(9 classes) ──► logit adjustment ──► leaf ──► clarity
                                        or Qwen3-8B-Base + LoRA (E13)      (10 seeds, averaged)  argmax p / prior^τ            (table)
```

The 8B classifier (`llm_classifier.py`) keeps everything below except the backbone: LoRA rank 16 on
all linear layers, a new 9-way head on the last token, learning rate 1e-4, 3 epochs.

**The model** (`encoder.py`). Pretrained DeBERTa-v3-large, all 435M parameters
fine-tuned, with a new 9-way classification head. The final system and the
baseline differ in two settings, marked:

| setting | final system | baseline | why |
|---|---|---|---|
| input | `[CLS] Sub-question: … Full question: … [SEP] answer [SEP]`, **1024 tokens** (question slot 256) | `[CLS] sub-question [SEP] answer [SEP]`, 512 tokens (question slot 96) | the full question shows what else the answer responds to (E8b, E10); in both, the answer fills the rest and is cut from the right |
| epochs | **16** | 8 | the full-question input needs about twice the training (E8b, E10) |
| loss | plain cross-entropy | same | rebalancing losses over-correct (E8) |
| learning rate | 1e-5, head 1e-4, layer-wise decay 0.95 | same | measured: beats 2e-5 at every epoch |
| optimiser, schedule | AdamW (β 0.9/0.98, ε 1e-6, weight decay 0.01), gradient clipping 1.0; cosine, 10% warmup, batch 16 | same | |
| checkpoint | best epoch on a fixed stratified 10% slice of **train** (345 rows) | same | dev is never used to choose anything |
| weights | loaded in **fp32**, computed in bf16 | same | see §4 |

**The ensemble.** Ten seeds, probabilities averaged (+0.025 over a single model for
the final system; +0.004 for the baseline).

**The decision rule** (`decide.py`, `make_submission.py`). Divide each class's
probability by its training frequency raised to τ, then take the argmax (logit
adjustment; Menon et al., ICLR 2021). One parameter, fitted on dev and scored by
nested cross-validation. It is worth +0.109 on the baseline's ensemble, whose models
lean on the class prior, and nothing on the final system's (τ = 0.4 when fitted on
all of dev), which already predicts rare classes.

---

## 4. Key findings

Each is reproducible; the script or report is named.

1. **The metric rewards landing among the acceptable labels, not matching the
   consensus.** If every prediction is acceptable, macro-F1 = (classes ever
   predicted) / 9, exactly. So two things matter — *in-set rate* and *coverage* —
   and every run logs both. `reports/01_scorer_geometry.md`,
   `verify_scorer_geometry.py`.
2. **The difficulty is mostly attribution.** 69% of rows share their answer with
   another sub-question, and most of those shared answers carry different labels.
   Without the sub-question, accuracy is capped at 0.753.
3. **Failure follows human disagreement, not class rarity.** `General` has plenty of
   training data and the second-worst F1 (0.12); it is also the label annotators
   disagree on most.
4. **The right answer is usually in the top 3 (90%), but only an oracle can pick it**
   (0.745). A tuned rule captures about a fifth of that gap; an encoder re-ranker
   captures none.
5. **The test set differs from dev**: two annotators instead of three, and answers
   four times shorter. Dev gains from multi-reference or long-answer effects will
   shrink on test.
6. **A silent library bug.** `transformers` 5.x loads DeBERTa-v3 in its stored fp16,
   where its attention overflows; with gradient clipping the model then trains
   smoothly to the label prior and learns nothing. Load with `dtype=torch.float32`.
7. **Undertraining can pass for "this idea doesn't work".** With the full question
   in the input the model needs about twice the epochs to fit (E8b ended where the
   baseline was at epoch 3), and an undertrained model leans on the class prior.
   E8 was first blamed on its loss function; E8b showed the input was the cause.
   Trained 16 epochs, the same input gives the best single model (9 of 10 seeds, E11).
   `compare_runs.py` now reports fit (final train loss, best epoch) next to scores.
8. **Longer training and the decision rule fix the same thing.** Training 16 epochs
   lifts the rare classes (`General` 0.12 → 0.30) and steadies the seeds, which is
   what logit adjustment was doing post hoc; together they add nothing over the
   rule alone (E10 control, E11).
9. **One set of 5 seeds cannot rank systems on 308 items.** The same system scored
   0.438 and 0.356 on two seed sets (E11); every system comparison now uses 10
   seeds, and per-seed paired comparisons carry the conclusions.

---

## 5. Running it

Python 3.12 with PyTorch 2.11 and `transformers` 5.x (`peft` 0.20 for the 8B classifier). There is
no install step: scripts run from the repository root, e.g. `python clarity/analyze.py …`. The data
are downloaded on first use from the public Hugging Face dataset `ailsntua/QEvasion` and cached in
`data/cache/`.

### Credentials

Copy `.env.example` to `.env` and fill in the W&B and Hugging Face keys (`.env` is gitignored).
Without keys, W&B logs offline and Hugging Face uploads are skipped; nothing fails.

### Trained runs

Every run's per-seed probabilities, metrics and selected weights are on the public Hugging Face repo
[`siddarthg44/clarity-semeval26`](https://huggingface.co/siddarthg44/clarity-semeval26), as
`<config>/seed<k>/`. All analyses run on CPU from the probabilities alone:

```python
from huggingface_hub import snapshot_download
snapshot_download("siddarthg44/clarity-semeval26", local_dir="clarity/runs",
                  allow_patterns=["*/seed*/*.npy", "*/seed*/*.json"])
```

### Training a configuration

Each encoder experiment is a folder `experiments/<EXP>/`: `common.args` holds the flags shared by
every run, and each line of a `lane_*.txt` file is one run, `<run-name> <seed> [extra flags]`. One run
is

```bash
python clarity/encoder.py --name E11_fullq_16ep --seed 5 \
    $(grep -vE '^\s*(#|$)' clarity/experiments/E11/common.args | tr '\n' ' ') \
    --input full --max-len 1024 --a-budget 256 --epochs 16
```

An interrupted run resumes from its last finished epoch (`ckpt.py`). The 8B classifier
(`llm_classifier.py`) was run on rented single-GPU machines with `../jarvis_drive.sh` and
`../jarvis_setup.sh`; see `../JARVISLABS_PORTING_GUIDE.md`.

### Analyses

```bash
python clarity/analyze.py --run-dir clarity/runs/L0_large_base       # per-class, both subtasks
python clarity/decide.py  --run-dir clarity/runs/L0_large_base --drop-annotator  # decision rules, nested CV
python clarity/compare_runs.py L0_large_base E8b_fullq_ce             # paired seeds, bootstrap, length split
python clarity/e11_replication.py                                    # the 10-seed DeBERTa systems (E11)
python clarity/e13_analysis.py                                       # Qwen vs DeBERTa (E13, E13x/b/c)
python clarity/qualitative_examples.py                               # item-level comparison of the two systems
python clarity/verify_scorer_geometry.py                             # the scorer findings
python clarity/make_submission.py --source stage1 --run L0_large_base --logit-adjust
```

Codabench is closed, but `make_submission.py` still enforces its format: one zip per subtask with a
single extensionless file `prediction`, 237 lines in official row order, full label names.

---

## 6. Files

```
clarity/
  README.md                  this file
  encoder.py                 the DeBERTa classifier: fine-tune, predict dev + test; --task for sub-problems
                               (gate, specialists, pair experts)
  llm_classifier.py          E13: Qwen3-8B-Base + LoRA as a 9-way classifier; same rows, slice, input and
                               selection as encoder.py
  decide.py                  decision rules (logit adjustment and alternatives), nested CV, 2-annotator scoring
  analyze.py, summarize.py, peek.py, compare_runs.py    per-run and cross-run analysis
  e11_replication.py         E11: the pre-registered 10-seed DeBERTa analysis
  e13_analysis.py            E13 family: paired seeds, systems, 12 epochs, all of train, per class
  qualitative_examples.py    the two systems item by item: fixes, losses, error pairs, rule-chosen examples
  rerank.py                  E6: the second-stage re-ranker (negative result)
  hier_combine.py            E12: gate + specialists + boundary experts -> 9-way
  decode_variants.py         alternative decision rules on trained models
  soup.py                    E12d: uniform and greedy model soups
  mideval_analysis.py        the DeBERTa error analysis (reports/raw/mideval/, data/splits/)
  paper_figures.py, paper_example.py, taxonomy_counts.py   the report's figures, worked example and Table 1
  mideval_figures.py, e13_figures.py                      PNG versions of the figures (reports/figures/)
  verify_scorer_geometry.py  regenerates every number in reports/01
  make_submission.py         Codabench zips with every format rule checked
  tracking.py, ckpt.py       W&B / Hugging Face logging; resumable epoch checkpoints
  experiments/<EXP>/         experiment definitions: shared flags + one line per run
  qevasion/                  data loader, label vocabulary, official-scorer replica
  .env.example               template for API keys
  data/
    clarity_task_evaluation_dataset.csv   the 237 test items (row order matters)
    splits/                  the fixed 345-row train slice and the dev halves used by the 2-fold checks
  reports/
    01_scorer_geometry.md    what the metric rewards
    02_experiment_log.md     every experiment in order: hypothesis and prediction (written first), result
    03_research_narrative.md the same work as a story: why each experiment, what came out, why
    04_results_sources.md    every reported number with the file it comes from
    raw/                     unedited analysis outputs behind the tables
    figures/                 PNG figures
  submissions/               packaged predictions for the baseline and the final DeBERTa system
```

Not in git: `runs/` (on the Hugging Face repo), `logs/`, `wandb/`, `.env`, `data/cache/`.

---

## 7. Rules this track keeps

- The baseline is frozen once trained and never re-tuned to flatter a later change.
- Single-model numbers are means ± std over paired seeds. A variant is screened on
  3 seeds (bar: +0.015 with at least 2 of 3 up) and extended only if it passes.
  Every system claim uses 10 seeds. Per-class F1 comes with its support.
- Nothing is chosen on dev. The one fitted parameter (τ) is validated by nested
  cross-validation, and the held-out score is the one reported.
- Hypotheses and predictions are written in the experiment log before a run, and results are
  reported against them, including failures.
