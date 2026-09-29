# `clarity/` — the fine-tuned encoder track

An end-to-end fine-tuned encoder for SemEval-2026 Task 6 (CLARITY), built up from
the simplest system that could work and extended one measured step at a time.

It is self-contained. The data layer, label vocabulary and official-scorer replica
it needs (`qevasion/`) were carried over from the team's earlier `higrec` analysis
track, which is now archived; the copies were checked to give identical results.

| | |
|---|---|
| **Final system** | full question + 16 epochs, 10-seed DeBERTa-v3-large ensemble + logit adjustment: dev S2 **0.405**, dev S1 **0.648** (baseline system: 0.428 / 0.601) |
| Best single model | full question + 16 epochs: 0.384 ± 0.030 dev S2, 0.614 ± 0.027 dev S1 over 10 seeds (baseline 0.315 / 0.576); better on 9 of 10 seeds |
| The story, in order | [`reports/02_experiment_log.md`](reports/02_experiment_log.md) |
| Latest | **E12**: training on all of train (last epoch) is the one variant that helps — Subtask 2 +0.029 per model on 4 of 5 seeds, 5-seed system 0.489. The hierarchy of specialist encoders, boundary experts and model soups do not. See [`reports/04_mideval_summary.md`](reports/04_mideval_summary.md) |
| Running | nothing |

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
| ChulaNLP (2nd), RoBERTa top-5 → Kimi-K2 | 0.52 | 0.70 |
| ChulaNLP, DeBERTa-large fine-tuned *(checkpoint chosen on dev)* | 0.46 | 0.65 |
| ours — baseline system: 10-seed ensemble + logit adjustment | 0.428 | 0.601 |
| TeleAI, DeepSeek-V3 asked directly for the label | 0.421 | 0.662 |
| **ours — final system: full question, 16 epochs, 10-seed ensemble + logit adjustment** | **0.405** | **0.648** |
| **ours — single model, full question, 16 epochs (10 seeds)** | **0.384 ± 0.030** | **0.614 ± 0.027** |
| ours — single model, 16 epochs (10 seeds) | 0.362 ± 0.026 | 0.601 ± 0.016 |
| ours — single model, baseline (10 seeds) | 0.315 ± 0.040 | 0.576 ± 0.030 |

What these numbers do and don't show is in the experiment log (§E7, §E11). In short:

- **Per model, the improvement is solid**: the full question plus 16 epochs beats
  the baseline model on 9 of 10 seeds (S2 +0.069, S1 +0.038).
- **As systems, the two are within noise on Subtask 2**: the decision rule lifts the
  baseline's prior-leaning ensemble by +0.109 and the better model's by nothing.
  The final system was chosen by a rule written before the last experiment ran.
- 5-seed system numbers are unreliable on 308 items (the baseline scored 0.438 and
  0.356 on two seed sets), so all system numbers above use 10 seeds.
- The fair external comparison is ChulaNLP's fine-tuned DeBERTa (0.46 / 0.65),
  whose checkpoint was chosen on dev; ours are held-out.

Ablations that did **not** help, all documented:

| tried | dev S2 | verdict |
|---|---|---|
| per-class decision thresholds (the textbook macro-F1 rule) | 0.394 | overfits 308 items; loses to the one-scalar rule |
| hard hierarchical routing (clarity branch first, then leaf) | 0.369 | no reliable effect |
| a second encoder re-ranking the top-3 | 0.354 | negative — [`reports/03_reranker_ablation.md`](reports/03_reranker_ablation.md) |
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
 sub-question + full question + answer ──► DeBERTa-v3-large ──► p(9 classes) ──► logit adjustment ──► leaf ──► clarity
                                          (10 seeds, averaged)                   argmax p / prior^τ            (table)
```

**The model** (`encoder.py`). Pretrained DeBERTa-v3-large, all 435M parameters
fine-tuned, with a new 9-way classification head. The final system and the
baseline differ in two settings, marked:

| setting | final system | baseline | why |
|---|---|---|---|
| input | `[CLS] Sub-question: … Full question: … [SEP] answer [SEP]`, **1024 tokens** (question slot 256) | `[CLS] sub-question [SEP] answer [SEP]`, 512 tokens | the full question shows what else the answer responds to (E8b, E10) |
| epochs | **16** | 8 | the full-question input needs about twice the training (E8b, E10) |
| loss | plain cross-entropy | same | rebalancing losses over-correct (E8) |
| learning rate | 1e-5, head 1e-4, layer-wise decay 0.95 | same | measured: beats 2e-5 at every epoch |
| schedule | cosine, 10% warmup, batch 16 | same | |
| checkpoint | best epoch on a fixed 10% slice of **train** | same | dev is never used to choose anything |
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

Environment: `/scratch/shlok/Temp/.venv` (Python 3.12, torch 2.11, transformers 5.x),
no install step: scripts are run from the repo root (`python clarity/<script>.py`). Both GPUs may be used; lanes decide which (see Experiments).

### Credentials

Copy `.env.example` to `.env` and fill in the W&B and Hugging Face keys (`.env` is
gitignored). Without keys, W&B logs offline and HF uploads are skipped; nothing
fails.

### Experiments

Each experiment is a folder `experiments/<EXP>/`:

```
experiments/E8/
  common.args          flags shared by every run (model, input, loss, schedule, ...)
  lane_gpu1_a.txt      one line per run:  <run-name> <seed> [extra flags]
  lane_gpu1_b.txt      several lanes can share a GPU; the GPU comes from the file name
  lane_gpu0_a.txt      a line "@after <run-name> <seed> ..." makes the lane wait until
                         those runs finish (queueing behind another experiment)
```

Lanes may list the same runs: each run is locked while it trains, so a lane skips
whatever another lane holds, and the lanes share the work (E10 does this across
two GPU-0 lanes and the GPU-1 slice).

```bash
bash clarity/start.sh E8               # one tmux window per lane, session "clarity-E8"
tmux attach -t clarity-E8              # watch; Ctrl-b n / p switch lanes; Ctrl-b d detaches
tail -f clarity/logs/E8.log            # one line per event, all lanes
```

Each lane runs its lines in order (`run_queue.sh`). When the last seed of a
configuration finishes, whichever lane finished it writes the analysis
(`reports/raw/<name>_analysis.txt`), the decision rules
(`reports/raw/<name>_decision_rules.txt`) and a submission
(`submissions/<name>_logitadj/`).

**Giving a GPU back** (e.g. to lab-mates): stop the session
(`tmux kill-session -t clarity-E8`), move lines out of that GPU's lane files, and run
`start.sh E8` again. Finished runs are skipped and interrupted ones resume from their
last epoch, so nothing is lost.

It all survives restarts:

| level | what happens after a crash or reboot |
|---|---|
| finished run (`metrics.json` exists) | skipped |
| interrupted run (`resume.pt` exists) | continues from its last completed epoch — model, optimizer, scheduler, RNG and data order restored (`ckpt.py`) |
| W&B | the resumed run continues the same W&B run |
| HF upload | retried in the background until it succeeds |
| GPU memory taken by another job | waits 10 min and retries, up to 2 h |

After a reboot, the same `bash clarity/start.sh E8` resumes everything.
The finished E0 + E6 pipeline can be re-run with `bash clarity/start.sh pipeline`.

### Individual pieces

```bash
python clarity/encoder.py --name L0_large_base --seed 0 --epochs 8   # one baseline run
python clarity/analyze.py --run-dir clarity/runs/L0_large_base       # per-class, both subtasks
python clarity/decide.py  --run-dir clarity/runs/L0_large_base --drop-annotator  # decision rules
python clarity/summarize.py --per-class                              # all configurations
python clarity/peek.py clarity/runs/E8_fullq_bal_focal/seed0          # what a run predicts, per class
python clarity/compare_runs.py L0_large_base E8b_fullq_ce             # paired seeds, bootstrap, length split, undertraining
python clarity/verify_scorer_geometry.py                             # the scorer findings, CPU
python clarity/make_submission.py --source stage1 --run L0_large_base --logit-adjust
```

### Submission format

Codabench is closed, but the packager still enforces the format three participant
repositories used: one zip per subtask, containing a single extensionless file named
`prediction` with 237 lines in official row order, using full label names.
Packaged submissions are in `submissions/`.

---

## 6. Files

```
clarity/
  README.md                  this file
  encoder.py                 the classifier: fine-tune, predict dev + test; --fold for cross-fitting;
                               --task for a sub-problem (gate, specialists, pair experts)
  rerank.py                  the (negative-result) second-stage re-ranker
  decide.py                  post-hoc decision rules R0–R3, nested CV, 2-annotator scoring
  analyze.py                 per-class breakdown for both subtasks from saved probabilities
  peek.py                    per-class predictions of a run, even mid-training (reads its checkpoint)
  summarize.py               table across configurations, mean ± std, paired deltas
  compare_runs.py            one configuration against another: paired seeds, bootstrap,
                               per-length split, still-improving-at-the-end check
  e11_replication.py         E11's pre-registered analysis: seed sets, 10-seed systems, hypotheses
  decode_variants.py         alternative decision rules on trained models (gate, optimal transport, floor)
  hier_combine.py            E12: gate + specialists + boundary experts -> 9-way, the 2x2 ablation
  soup.py                    E12d: uniform and greedy weight-averaged model soups
  verify_scorer_geometry.py  regenerates every number in reports/01
  make_submission.py         builds Codabench zips and checks every format rule
  tracking.py                W&B and HF Hub logging; never allowed to crash a run
  ckpt.py                    atomic epoch-level resume checkpoints
  start.sh                   launches an experiment (or the original pipeline) in tmux
  run_queue.sh               runs one lane of an experiment on one GPU
  run_pipeline.sh            the E0 + E6 pipeline, kept to reproduce those results
  experiments/<EXP>/         experiment definitions: shared flags + lane files
  qevasion/                  data loader, label vocabulary, official-scorer replica
  .env.example               template for API keys
  data/
    cache/                   the organizers' train (3,448) and dev (308) splits, as parquet
    clarity_task_evaluation_dataset.csv   the 237 test items (row order matters)
  reports/
    01_scorer_geometry.md    what the metric rewards; corrects the earlier data audit
    02_experiment_log.md     every experiment in order: why, what, result, takeaway
    03_reranker_ablation.md  the re-ranker negative result, written up for presentation
    04_mideval_summary.md    the whole track on one page: progression, component effects, lessons
    raw/                     unedited analysis outputs behind the tables
  submissions/               packaged predictions for the baseline and the final system
                               (ablations write theirs locally; not in git)
```

Not in git (regenerable or large): `runs/` (probabilities and checkpoints; the
checkpoints are also on the private HF repo), `logs/`, `wandb/`, `.env`.

---

## 7. Rules this track keeps

- The baseline is frozen once trained and never re-tuned to flatter a later change.
- Every reported number is a mean over 5 seeds; per-class F1 comes with its support.
- Nothing is chosen on dev. The one fitted parameter (τ) is validated by nested
  cross-validation, and the held-out score is the one reported.
- Code is not edited while a multi-seed run is in progress, so every seed of a
  configuration runs the same code.
- Mistakes that cost a rerun are recorded in the experiment log where they
  happened.
