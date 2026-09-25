# `clarity/` — the fine-tuned encoder track

An end-to-end fine-tuned encoder for SemEval-2026 Task 6 (CLARITY), built up from
the simplest system that could work and extended one measured step at a time.

It is self-contained. The data layer, label vocabulary and official-scorer replica
it needs (`qevasion/`) were carried over from the team's earlier `higrec` analysis
track, which is now archived; the copies were checked to give identical results.

| | |
|---|---|
| **Best result** | **0.438** dev Subtask-2 macro-F1 — 5-seed DeBERTa-v3-large ensemble + post-hoc logit adjustment |
| Baseline (single model) | 0.337 ± 0.044 dev Subtask 2, 0.586 ± 0.039 dev Subtask 1 |
| The story, in order | [`reports/02_experiment_log.md`](reports/02_experiment_log.md) |
| Next | **E8** — full question + Balanced Softmax + focal loss; configured in `experiments/E8/`, not yet launched |

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
| **ours — 5-seed ensemble + logit adjustment** | **0.438** | — |
| TeleAI, DeepSeek-V3 asked directly for the label | 0.421 | 0.662 |
| ours — 5-seed ensemble | 0.365 | 0.596 |
| ours — single model (5 seeds) | 0.337 ± 0.044 | 0.586 ± 0.039 |

What these numbers do and don't show is in the experiment log, §E7. In short:

- The fair comparison for our single model is ChulaNLP's fine-tuned DeBERTa
  (0.46 vs 0.337). Their model sees the full journalist question and ours does not
  — the most likely source of the gap, and the next experiment.
- Published dev numbers are partly tuned on dev; ours are held-out.

Ablations that did **not** help, all documented:

| tried | dev S2 | verdict |
|---|---|---|
| per-class decision thresholds (the textbook macro-F1 rule) | 0.394 | overfits 308 items; loses to the one-scalar rule |
| hard hierarchical routing (clarity branch first, then leaf) | 0.369 | no reliable effect |
| a second encoder re-ranking the top-3 | 0.354 | negative — [`reports/03_reranker_ablation.md`](reports/03_reranker_ablation.md) |

---

## 3. How the system works

```
 sub-question + answer ──► DeBERTa-v3-large ──► p(9 classes) ──► logit adjustment ──► leaf ──► clarity
                          (5 seeds, averaged)                    argmax p / prior^τ            (table)
```

**The model** (`encoder.py`). Pretrained DeBERTa-v3-large, all 435M parameters
fine-tuned, with a new 9-way classification head:

| setting | value | why |
|---|---|---|
| input | `[CLS] sub-question [SEP] answer [SEP]`, 512 tokens | the minimal pair; the full question is the next experiment |
| loss | plain cross-entropy | the baseline must be a clean control |
| learning rate | 1e-5, head 1e-4, layer-wise decay 0.95 | measured: beats 2e-5 at every epoch |
| schedule | 8 epochs, cosine, 10% warmup, batch 16 | 5 epochs was measured to undertrain |
| checkpoint | best epoch on a fixed 10% slice of **train** | dev is never used to choose anything |
| weights | loaded in **fp32**, computed in bf16 | see §4 |

**The ensemble.** Five seeds, probabilities averaged. Seed-to-seed spread is large
(0.28–0.39), so averaging is worth +0.028.

**The decision rule** (`decide.py`, `make_submission.py`). Divide each class's
probability by its training frequency raised to τ, then take the argmax (logit
adjustment; Menon et al., ICLR 2021). One parameter, τ = 0.85, fitted on dev and
validated by nested cross-validation: +0.073 held-out, and +0.055 to +0.062 on the
2-annotator versions of dev that mimic the test set.

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
  lane_gpu0_a.txt
```

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
  encoder.py                 the classifier: fine-tune, predict dev + test; --fold for cross-fitting
  rerank.py                  the (negative-result) second-stage re-ranker
  decide.py                  post-hoc decision rules R0–R3, nested CV, 2-annotator scoring
  analyze.py                 per-class breakdown for both subtasks from saved probabilities
  summarize.py               table across configurations, mean ± std, paired deltas
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
    raw/                     unedited analysis outputs behind the tables
  submissions/               packaged predictions for the two main systems
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
