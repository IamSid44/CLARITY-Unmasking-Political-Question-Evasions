# Nier_ANLP: CLARITY (SemEval-2026 Task 6)

**Status (2026-10-14):** the mid-evaluation is submitted (`Report/Report.pdf`, `Slides/CLARITY_mideval.pptx`).
Phase 2 is a confidence-routed cascade towards the final evaluation. Its plan, protocol, budget and
server procedure are in [`docs/05_phase2_plan_and_budget.md`](docs/05_phase2_plan_and_budget.md).

Its first module, **M1 (experiment E14), is done.** The 12-epoch Qwen3-8B system reaches
**dev S2 0.598 / S1 0.743** with 10 seeds. E14's registration and results are in `docs/02_experiment_log.md`.
The second, **M3 (experiment E15, an LLM adjudicator for uncertain items), is done.** A local Qwen3.8-27B-FP8
chooses among M1's top four labels for the 30% least confident items, and its choice is fused with M1.
This gives **dev S2 0.669 / S1 0.798 at 0.36 LLM calls per item**, +0.067 [+0.009, +0.124] over M1; TeleAI
scores 0.617 / 0.812 at 1.94 calls. These numbers were corrected on 2026-10-14 for a fusion bug; see E15 in
the log.
E16 (2026-10-14) puts this number in context. Averaged over random seed subsets, M3 adds +0.02 to +0.03
at every smaller ensemble size, so the 10-seed point (+0.067) is probably optimistic. M3 also lets one seed
replace ten: 1 seed + M3 on every item scores 0.593 / 0.774 at 0.155 GPU-s per item, against 0.598 / 0.743
for the 10-seed M1 alone at 0.52 GPU-s.
The third, **M4 (E17, distillation), is done.** One Qwen3-8B student is trained on soft targets from M3, in place
of 10 seeds. On its own it scores dev S2 0.573 (+0.048 over a single M1 seed, 9/10 seeds). With M3 at δ = 0.5 it
reaches **0.608 / 0.790 at 0.111 GPU-s per item**: better than the 10-seed M1 alone, at 4.7× less cost. The system
has two operating points: accuracy (10-seed M1 + M3) and cost (student + M3).
Final numbers (E18): accuracy point S2 0.669 / S1 0.798 / 2-annotator 0.589; cost point 0.608 / 0.790 / 0.567
at 0.21× the GPU cost of the 10-seed M1. Figure: `docs/figures/E18_frontier.pdf`.

**Knowledge over Structure: A Controlled Study of Response Clarity Classification in Political Interviews**

SemEval-2026 Task 6 (CLARITY): given a question from a political interview and the politician's answer,
classify how clearly the answer responds to the question.

| Member | Roll number |
|---|---|
| Vidvathama R | 2024122002 |
| Siddarth Gottumukkula | 2023102040 |
| Sanjana Reddy Vonteri | 2026901007 |
| Shashikanta Sahoo | 2026900007 |

International Institute of Information Technology, Hyderabad

## Links

| Resource | Link |
|---|---|
| Code repository | https://github.com/IamSid44/CLARITY-Unmasking-Political-Question-Evasions |
| Trained runs (Hugging Face) | https://huggingface.co/siddarthg44/clarity-semeval26 (per-seed probabilities, metrics and LoRA adapters, stored as `<config>/seed<k>/`) |
| Training curves (Weights & Biases) | https://wandb.ai/iamsid44-iiit-hyderabad/clarity-semeval26 (runs named `<config>-s<seed>`, grouped by config) |
| Dataset | https://huggingface.co/datasets/ailsntua/QEvasion (public; downloaded automatically on first use) |

---

## 1. Where to start

| If you want to... | Go to |
|---|---|
| read the report | `Report/Report.pdf` (LaTeX source: `Report/latex/`) |
| see the slides | `Slides/CLARITY_mideval.pptx` (built by `code/slides/build_deck.py`) |
| see the plan for the final evaluation | `docs/05_phase2_plan_and_budget.md` |
| understand the code layout | §3 and §4 below |
| see the main models | `code/models/encoder.py` (DeBERTa), `code/models/llm_classifier.py` (Qwen3-8B + LoRA) |
| see the scoring function | `code/qevasion/scoring.py` |
| see the decision rule used in every system | `code/decision_rules/decide.py` |
| follow the research step by step | `docs/03_research_narrative.md` |
| check each experiment's hypothesis and result | `docs/02_experiment_log.md` |
| find the file behind any number in the report | `docs/04_results_sources.md` |
| reproduce something | §7 below |

## 2. The task in brief

Each item is a sub-question split out of a journalist's turn, the full journalist question, and the
politician's answer. There are two subtasks:

- **Subtask 1 (3 classes):** Clear Reply, Ambivalent, Clear Non-Reply.
- **Subtask 2 (9 classes):** Explicit, Implicit, Dodging, General, Deflection, Partial/half-answer,
  Declining to answer, Claims ignorance, Clarification.

Each Subtask 2 class belongs to exactly one Subtask 1 class, so we predict the 9-way label and map it
up to the 3-way one. Dev items have three annotators. Subtask 2 uses **multi-reference macro-F1**: a
prediction counts as correct if any annotator gave that label. Test labels were never released, so
every number here is on the 308-item dev set.

---

## 3. Folder layout

```
CLARITY-Unmasking-Political-Question-Evasions/
├── README.md                        this file
├── requirements.txt                 Python dependencies
├── Report/
│   ├── Report.pdf                   the mid-evaluation report
│   └── latex/                       its ACL-format LaTeX source (the final report extends it)
├── Slides/
│   └── CLARITY_mideval.pptx         the mid-evaluation slides (editable)
│
├── code/                            ALL source code (run every command from inside this folder)
│   ├── qevasion/                    shared library: data, labels, splits, paths, official-scorer replica
│   ├── models/                      the trainable classifiers and model-level variants
│   ├── cascade/                     Phase 2 cascade modules (M1 candidate sets and uncertainty; M3 LLM adjudicator and fusion)
│   ├── decision_rules/              turning saved probabilities into labels
│   ├── evaluation/                  metrics, run comparisons, packaging predictions
│   ├── preregistered_analyses/      the analyses behind the reported numbers
│   ├── figures/                     figures and tables used in the report and slides
│   ├── slides/                      builds the mid-evaluation deck (python-pptx)
│   ├── utils/                       checkpointing and experiment tracking
│   ├── launch/                      lab-server queue (tmux) and rented-GPU (JarvisLabs) launchers
│   ├── experiments/                 run lists of every queued experiment (E8–E12 encoder, E14 and E17 Qwen)
│   └── .env.example                 template for optional W&B / Hugging Face keys
│
├── data/                            evaluation CSV and fixed splits (train/dev download automatically)
├── docs/                            experiment log, narrative, results map, raw outputs, figures
└── submissions/                     packaged predictions of the DeBERTa systems
```

Three folders are created when you run the code and are not shipped:
- `runs/`: one folder per configuration and seed (or fold), holding probabilities, metrics and
  checkpoints. It can be downloaded from Hugging Face.
- `logs/`: training and queue logs.
- `data/cache/`: the downloaded train and dev parquet files.

Every path is defined once, in `code/qevasion/paths.py`. `CLARITY_RUNS` moves `runs/`.

---

## 4. How the code fits together

### 4.1 Architecture

```
                      ┌───────────────────────────────────────────────┐
                      │ code/qevasion/  (imported by every module)    │
                      │   loader.py   train/dev from HF, reference    │
                      │               sets, fixed internal splits     │
                      │   labels.py   9 + 3 labels, 9 → 3 mapping     │
                      │   scoring.py  official-scorer replica         │
                      └───────────────────────┬───────────────────────┘
                                              │
         ┌────────────────────────────────────┼────────────────────────────────────┐
         ▼                                    ▼                                    ▼
 ┌────────────────────┐            ┌─────────────────────┐            ┌──────────────────────┐
 │ STAGE 1: TRAIN     │            │ utils/              │            │ code/experiments/    │
 │ models/encoder.py  │◄───────────│  ckpt.py   resume   │            │  E*/common.args      │
 │ models/llm_        │            │  tracking.py W&B/HF │            │  E*/lane_*.txt       │
 │    classifier.py   │            └─────────────────────┘            │  (encoder settings)  │
 └─────────┬──────────┘                                               └──────────┬───────────┘
           │ per seed: dev_probs.npy, test_probs.npy, val_probs.npy, metrics.json │
           ▼                                                                      │
   runs/<config>/seed<k>/  ◄──────────────────────────────────────────────────────┘
           │
           ├─► models/rerank.py, models/hier_combine.py, models/soup.py   (variants built on saved runs)
           │
           ▼
 ┌──────────────────────────────┐     ┌────────────────────────────────┐     ┌──────────────────────┐
 │ STAGE 2: DECIDE              │     │ STAGE 3: EVALUATE / ANALYSE    │     │ STAGE 4: REPORT      │
 │ decision_rules/decide.py     │────►│ evaluation/*                   │────►│ figures/*            │
 │  seed ensemble + logit       │     │ preregistered_analyses/*       │     │ docs/raw/*.txt       │
 │  adjustment (τ by nested CV) │     │                                │     │ docs/*.md            │
 └──────────────────────────────┘     └───────────────┬────────────────┘     │ Report/Report.pdf    │
                                                      │                      └──────────────────────┘
                                                      ▼
                                     evaluation/make_submission.py ─► submissions/ (Codabench zips)
```

### 4.2 The flow behind a reported number

1. **Data.** `qevasion/loader.py` downloads QEvasion train (3,448 rows) and dev (308 items, three
   annotators) and builds each dev item's *reference set* (every label an annotator gave). A fixed 10%
   slice of train (`data/splits/train_internal_val_index.json`) is held out for choosing the epoch, so
   dev is never used to select anything during training.
2. **Training.** `models/encoder.py` (DeBERTa-v3-large) or `models/llm_classifier.py`
   (Qwen3-8B-Base + LoRA) trains a 9-way classifier on the input
   `sub-question + full journalist question + answer`. The two scripts use the same rows, the same
   input text and token budgets, and the same epoch-selection rule, so **only the backbone differs**.
   Each run writes per-item probabilities and metrics to `runs/<config>/seed<k>/`.
3. **Decision.** `decision_rules/decide.py` averages the seeds' probabilities (the *system*) and applies
   **logit adjustment**: each class probability is divided by its training frequency raised to τ. The
   single parameter τ is fitted and scored by nested cross-validation on dev, so the reported score is
   held out.
4. **Scoring.** `qevasion/scoring.py` computes multi-reference macro-F1 for Subtask 2 and macro-F1 for
   Subtask 1 (9-way prediction mapped to 3-way), matching the official scorer.
5. **Analysis.** The scripts in `preregistered_analyses/` test hypotheses that were written down
   *before* each run, using paired per-seed comparisons and bootstrap intervals. They print to
   `docs/raw/`.
6. **Reporting.** `figures/` regenerates the report's figures and tables, and
   `docs/04_results_sources.md` maps every number in the report to the script and output file it
   comes from.

### 4.3 Module dependencies

| Module | Imports from |
|---|---|
| everything | `qevasion` |
| `models.encoder` | `utils.ckpt`, `utils.tracking` |
| `models.llm_classifier` | `utils.tracking` |
| `models.rerank`, `models.soup` | `models.encoder` (and `utils.*`) |
| `evaluation.compare_runs` | `models.encoder` |
| `decision_rules.decode_variants`, `models.hier_combine`, `preregistered_analyses.mideval_analysis`, `preregistered_analyses.e11_replication`, `preregistered_analyses.e13_analysis`, `preregistered_analyses.qualitative_examples`, `figures.paper_figures`, `figures.paper_example`, `figures.e13_figures`, `cascade.candidates` | `decision_rules.decide` |
| `figures.paper_example`, `preregistered_analyses.qualitative_examples` | `preregistered_analyses.e13_analysis` |
| `preregistered_analyses.e14_m1_analysis` | `cascade.candidates`, `preregistered_analyses.e13_analysis` |
| `cascade.adjudicate` | `cascade.prompts` |
| `cascade.fuse` | `cascade.adjudicate`, `decision_rules.decide` |
| `cascade.frontier` | `cascade.candidates`, `cascade.fuse`, `decision_rules.decide` |
| `cascade.cost` | `models.llm_classifier` |
| `preregistered_analyses.e18_final` | `cascade.frontier`, `cascade.fuse`, `decision_rules.decide` |

---

## 5. What each file does

### `code/qevasion/`: shared library

| File | Purpose |
|---|---|
| `loader.py` | Loads train/dev from the Hugging Face parquet files (cached in `data/cache/`), builds multi-annotator reference sets, consensus levels and the fixed splits |
| `labels.py` | The 9 evasion labels, the 3 clarity labels, the official 9 → 3 mapping, and label encoders |
| `scoring.py` | Replica of the official scorer: multi-reference macro-F1 (Subtask 2) and macro-F1 (Subtask 1) |

### `code/models/`: classifiers

| File | Purpose | Experiments |
|---|---|---|
| `encoder.py` | DeBERTa-v3-large cross-encoder. Options: input form (`qa` or `full` question), token budgets, loss (CE, Balanced Softmax, focal), flat or hierarchical head, sub-task heads (`--task gate/other6/nr3/pair:A,B`), all-data training (`--val-frac 0 --select last`) | E0–E12 |
| `llm_classifier.py` | Qwen3-8B-Base with LoRA (r = 16, every linear layer) and a 9-way head on the last token; same data, input and selection rule as the encoder. `--soft-targets` trains on M4's distillation targets instead of the gold labels | E13, E13b, E13c, E14 (M1), E17 (M4 students) |
| `rerank.py` | Second-stage cross-encoder that re-ranks the first stage's top-K labels using label definitions, with cross-fitted training candidates and fusion with the first stage | E6 |
| `hier_combine.py` | Combines a Non-Reply gate, branch specialists and pair experts (all trained with `encoder.py --task ...`) into 9-way predictions; CPU only | E12b, E12c |
| `soup.py` | Uniform and greedy weight averaging of trained encoders | E12d |

### `code/decision_rules/`

| File | Purpose |
|---|---|
| `decide.py` | Decision rules over saved probabilities: argmax, logit adjustment (τ by nested CV), set-membership, per-class weights; `--drop-annotator` re-scores on 2-annotator reference sets |
| `decode_variants.py` | Further decoding variants (Non-Reply gate threshold, prior matching, coverage floor) compared against logit adjustment |

### `code/evaluation/`

| File | Purpose |
|---|---|
| `analyze.py` | Full metric breakdown for one run: per-class F1, in-set rate, coverage, confusions |
| `compare_runs.py` | Paired per-seed and ensemble comparison of configurations against the baseline |
| `summarize.py` | One table of all finished runs from their `metrics.json` |
| `verify_scorer_geometry.py` | Regenerates every number in `docs/01_scorer_geometry.md` |
| `peek.py` | Inspects a run's progress while it trains |
| `make_submission.py` | Packages test predictions as Codabench zips into `submissions/` |

### `code/preregistered_analyses/`

| File | Purpose | Output |
|---|---|---|
| `mideval_analysis.py` | Baselines, main table, bootstrap intervals, floors, error analysis, human ceiling | `docs/raw/mideval/` |
| `e11_replication.py` | E11: replication of the DeBERTa findings on seeds 5–9 | `docs/raw/E11_replication.txt` |
| `e13_analysis.py` | E13: Qwen vs DeBERTa, paired per seed and as 10-seed systems, plus E13b/E13c | `docs/raw/E13_final_analysis.txt` |
| `e13_topk.py` | Top-k coverage and confidence vs correctness for Qwen | `docs/raw/E13_Q8_topk_confidence.txt` |
| `qualitative_examples.py` | Side-by-side examples of the two 10-seed systems | `docs/raw/qualitative_examples.txt` |
| `e14_m1_analysis.py` | E14 (M1): 12 vs 3 epochs, the seed-count curve, candidate-set coverage, uncertainty AUROC, headroom for the LLM stage | `docs/raw/E14_*.txt` |
| `e17_m4_analysis.py` | E17 (M4): distilled student vs M1 single model (paired seeds), routing (top probability, AUROC), student + M3 against the undistilled frontier | `docs/raw/E17_m4_analysis.txt` |
| `e18_final.py` | E18 (freeze): final dev numbers for both operating points (nested CV, 2-annotator, per class), the cost table, and the frontier figure | `docs/raw/E18_final.txt`, `docs/figures/E18_frontier.{pdf,png}` |

### `code/cascade/`

| File | Purpose |
|---|---|
| `prompts.py`, `exemplars.py` | Module M3's prompt (definitions, confusion guide, lettered candidates) and the retrieval of train examples (excluding same-answer and same-interview rows) |
| `adjudicate.py` | Module M3: scores M1's top-4 candidates with an LLM (local vLLM or the JarvisLabs Model API), one request and one output token per item; resumable, never pays twice. Output: `runs/M3/<tag>/responses.jsonl` |
| `api_budget.py`, `api_probe.py` | API prices and the spend ledger with E15's hard cap; the probe of what the API returns |
| `fuse.py` | Fuses q with M1 over C(x), defers by confidence (the M2 curve), and scores the screen, the slice and the dev system |
| `frontier.py` | E16: the accuracy-vs-cost frontier over seed subsets (K) × deferral rate (δ), for M1 or the E17 students |
| `cost.py` | E16: measured GPU-seconds per item of an M1 seed (HF) and an M3 call (vLLM), written to `runs/M3/cost.json` |
| `distill_targets.py` | E17 (M4): soft targets (1 − α)·gold + α·q from M3, for `llm_classifier --soft-targets` |
| `publish.py` | Uploads non-training outputs (`runs/M3`, `runs/M4`) to HF and logs each experiment's results to W&B |
| `candidates.py` | Module M1. From a seed ensemble's saved probabilities it builds candidate sets C(x) (adaptive, rank-conformal or fixed top-k, calibrated on the train slice after temperature scaling) and an uncertainty score u(x) (logistic regression on the slice), for slice, dev, test and the cross-fitted train rows. Output: `runs/M1/<config>_<method>_a<alpha>/` |

### `code/figures/`

| File | Produces |
|---|---|
| `paper_figures.py` | The report's two data figures, `fig_deberta_seeds.pdf` and `fig_qwen_vs_deberta.pdf`, written to `Report/latex/figures/` |
| `deck_figures.py` | The slides' figures, in the report's style, written to `docs/figures/deck/` |
| `paper_example.py` | The report's worked example: one dev item through both 10-seed systems |
| `taxonomy_counts.py` | Per-class counts for the taxonomy table (Table 1) |
| `e13_figures.py`, `mideval_figures.py` | `docs/figures/e13_qwen_vs_deberta.png`, `docs/figures/per_seed_progression.png` |

### `code/utils/`

| File | Purpose |
|---|---|
| `ckpt.py` | Atomic checkpoint saving and resume after interruption |
| `tracking.py` | Weights & Biases logging and Hugging Face upload of metrics, probabilities and adapters; reads keys from `code/.env` and skips both when no keys are set |

### `code/launch/`

| File | Purpose |
|---|---|
| `start.sh` | Lab server: starts (or resumes) an experiment in a tmux session `clarity-<EXP>`, one window per lane file, so runs survive an ssh drop |
| `gpu_watch.sh` | Lab server: orchestrates an experiment from tmux. It finishes the model download, starts the lanes, and adds or removes a lane on our GPU-1 MIG slice when it is free or another user needs it (`start`, `status`, `release`, `allow`) |
| `run_queue.sh` | Lab server: runs one lane's list of runs in order. It skips finished runs, resumes interrupted ones, uploads each finished run to HF, and guards GPU memory on the shared card (falls back to gradient checkpointing). Usage is in the file header and `docs/05_phase2_plan_and_budget.md` §5 |
| `m3_local.sh`, `download_model.sh` | Lab server: E15's local LLM arms on GPU 0 (vLLM venv), and resumable model downloads |
| `e16_gpu.sh`, `e17_post.sh` | Lab server: E16's GPU steps (M3 scoring, prompt ablation, cost timing), and E17's evaluation of the distilled students once they finish (M3 on their candidates, frontier, analysis, HF/W&B publishing) |
| `jarvis_drive.sh` | Runs on a local machine. It packages the code, copies it to a rented GPU instance, starts training there inside tmux, copies logs and results back every 5 minutes, and pauses the instance when done |
| `jarvis_setup.sh` | Runs on the instance. It creates the Python environment, downloads the model, runs a smoke test (including resume) and a timing pilot, then trains each seed and uploads the results |

### `code/experiments/`

One folder per queued experiment: E8, E8b, E9, E10, E11 and E12 are DeBERTa; E14 (M1) and E17 (M4 students) are Qwen3-8B.
- `common.args` holds the arguments shared by the experiment, plus queue directives (`@module`,
  `@hf-push`, `@need-gb`).
- Each line of a `lane_gpu<N>_<x>.txt` file is one run: `<run name> <seed | fold<k>> <extra arguments>`.

`bash code/launch/start.sh <EXP>` runs an experiment.

### `data/`

| File | Contents |
|---|---|
| `clarity_task_evaluation_dataset.csv` | The task evaluation file (question–answer items with annotator columns) |
| `splits/train_internal_val_index.json` | The fixed 10% train slice used for epoch selection |
| `splits/dev_A_B.json` | A stratified two-way split of dev, used for the 2-fold protocol rows |
| `splits/train_5fold.json` | The stratified 5-fold partition of all 3,448 train rows used for cross-fitting (E6 DeBERTa, E14 Qwen) |

### `docs/`

| File | Contents |
|---|---|
| `01_scorer_geometry.md` | What the multi-reference scorer rewards, and what that implies for decision rules |
| `02_experiment_log.md` | Chronological log: every experiment's hypothesis and prediction (written before it ran), setup, result and analysis |
| `03_research_narrative.md` | The same work as one argument, step by step from the first DeBERTa run to the Phase 2 cascade (E14–E17) |
| `04_results_sources.md` | Every number in the report with the script and output file it comes from (§10: Phase 2 numbers) |
| `05_phase2_plan_and_budget.md` | Phase 2 (after the mid-evaluation): the cascade plan, protocol, compute ledger and budget, how to run on the lab server |
| `raw/` | Raw text and CSV outputs of every analysis script |
| `figures/` | Generated figures |

Read the documents in `docs/` in order: 01 explains the metric, 02 is the full record, 03 tells the
story, 04 is the index of numbers, and 05 is the plan from here on.

The log records the work as it happened. It mentions:
- `run_pipeline.sh`, the original E0 + E6 pipeline, which is no longer shipped;
- the team's earlier analysis track (`higrec`), since retired; it remains in the git history.

The lab-server queue (`run_queue.sh`, `start.sh`) is in `code/launch/`. Every result the report uses
is reproducible from the code here and the runs on Hugging Face.

### `submissions/`

Codabench-format prediction zips (one per subtask) for the DeBERTa baseline, the baseline with logit
adjustment, and the final DeBERTa system (`FINAL_fullq_16ep_10seed_logitadj`). Each folder has a
`manifest.json` that records the seeds and decision rule used.

---

## 6. Results so far (dev set)

| System | Subtask 2 (9 classes) | Subtask 1 (3 classes) |
|---|---|---|
| DeBERTa-v3-large baseline, single model (mean of 10 seeds) | 0.315 ± 0.040 | 0.576 ± 0.030 |
| DeBERTa-v3-large, full question + 16 epochs, single model | 0.384 ± 0.030 | 0.614 ± 0.027 |
| DeBERTa 10-seed system (ensemble + logit adjustment) | 0.405 | 0.648 |
| **Qwen3-8B-Base + LoRA, single model (mean of 10 seeds)** | **0.476 ± 0.043** | **0.709 ± 0.031** |
| **Qwen3-8B-Base + LoRA, 10-seed system** | **0.543** | **0.746** |
| Qwen3-8B-Base + LoRA, 12 epochs, single model (mean of 10 seeds; E14, after the mid-evaluation) | 0.525 ± 0.042 | 0.729 |
| **Qwen3-8B-Base + LoRA, 12 epochs, 10-seed system (E14 = cascade module M1)** | **0.598** | **0.743** |
| **M1 + M3: Qwen3.8-27B-FP8 over M1's top 4, fused, 30% deferred (0.36 calls/item; E15)** | **0.669** (10 CV splits 0.660) | **0.798** |
| M4 student, single model (one Qwen3-8B on soft targets; mean of 10 seeds; E17) | 0.573 | 0.761 |
| **Cost operating point: M4 student + M3, δ = 0.5 (0.57 calls, 0.111 GPU-s/item; E17)** | **0.608** | **0.790** |
| TeleAI (1st place), multi-call DeepSeek-V3 pipeline | 0.617 | 0.812 |
| Human annotator scored against the other two | 0.684 | — |

The plan and timeline to the final evaluation are in the Conclusion of `Report/Report.pdf`. The working
version, with the budget and the current status, is `docs/05_phase2_plan_and_budget.md`.

---

## 7. Reproducing

**Setup.** Python 3.12, then:

```bash
pip install -r requirements.txt
cd code                     # every command below runs from code/
```

The train and dev data download automatically on first use.

**Fetch the trained runs** (probabilities and metrics, without weights; needed by the analyses):

```python
from huggingface_hub import snapshot_download
snapshot_download("siddarthg44/clarity-semeval26", local_dir="../runs",
                  allow_patterns=["*/seed*/*.npy", "*/seed*/*.json"])
```

**Re-run the analyses and figures** (CPU):

```bash
python -m preregistered_analyses.e13_analysis          # Qwen vs DeBERTa, per seed and as systems
python -m preregistered_analyses.mideval_analysis      # main table, floors, error analysis
python -m preregistered_analyses.e11_replication       # DeBERTa replication
python -m decision_rules.decide --run-dir ../runs/E10_fullq_16ep
python -m evaluation.analyze   --run-dir ../runs/Q8_fullq_lora
python -m evaluation.verify_scorer_geometry
python -m figures.paper_figures                        # report figures -> Report/latex/figures/
python -m figures.taxonomy_counts                      # Table 1
```

**Train the DeBERTa encoder** (one GPU with 24 GB is enough). Each experiment line is
`<run name> <seed> <extra arguments>`. For example, the first line of `experiments/E10/lane_gpu0_a.txt`:

```bash
python -m models.encoder $(cat experiments/E10/common.args) \
    --name E10_fullq_16ep --seed 0 --input full --max-len 1024 --a-budget 256
```

`common.args` includes `--push-to-hub`. Without keys in `code/.env`, the upload is skipped and the
run still completes.

**Train the Qwen3-8B classifier** (one large GPU; we used an RTX PRO 6000 with 96 GB):

```bash
python -m models.llm_classifier --name Q8_fullq_lora --seed 0                  # E13
python -m models.llm_classifier --name Q8_fullq_lora_12ep --seed 0 --epochs 12 # E13b
python -m models.llm_classifier --name Q8_alldata --seed 0 --val-frac 0 --select last   # E13c
```

**Run the M3 cascade stage (E15).** This needs M1's inputs in `runs/M1/`, a vLLM environment for the
local LLM, and `JARVIS_API_KEY` in `code/.env` for the API arms. Details are in
`docs/05_phase2_plan_and_budget.md` §6.

```bash
python -m cascade.exemplars embed                                  # retrieval embeddings (GPU, ~2 min)
bash ../code/launch/m3_local.sh screen && bash ../code/launch/m3_local.sh full 0   # Qwen3.8-27B-FP8, GPU 0
python -m cascade.fuse dev --tag qwen38_27b_k0                     # dev system and the deferral curve
```

On the lab server, in tmux, with W&B logging and HF upload: `bash code/launch/start.sh E14` (see
`docs/05_phase2_plan_and_budget.md` §5). On a rented instance: `bash code/launch/jarvis_drive.sh`.
Run it from the repository root; with no arguments it lists its commands.

**Phase 2, module M1** (CPU, after the runs):

```bash
python -m cascade.candidates --run Q8_fullq_lora_12ep --crossfit Q8_12ep_crossfit --method aps --alpha 0.1
python -m preregistered_analyses.e14_m1_analysis > ../docs/raw/E14_m1_analysis.txt
```

The E14 runs themselves: `bash code/launch/gpu_watch.sh E14 start` on the lab server (§5 of docs/05),
or per run, for example `python -m models.llm_classifier --name Q8_fullq_lora_12ep --seed 3 --epochs 12`
and `python -m models.llm_classifier --name Q8_12ep_crossfit --seed 0 --fold 2 --select last --epochs 12`.

**Phase 2, modules M2, M4 and the freeze (E16–E18).** The full order, with the lab-server environments,
is in `docs/05_phase2_plan_and_budget.md` §6. In short:

```bash
python -m cascade.frontier jobs                                    # (dev item, top-4) pairs for the frontier
bash ../code/launch/e16_gpu.sh                                     # GPU: M3 on all train rows and frontier pairs, prompt ablation, cost timing
python -m cascade.frontier analyse > ../docs/raw/E16_frontier.txt  # E16: accuracy vs cost over K seeds x deferral
python -m cascade.distill_targets --alpha 0.5                      # E17: soft targets -> runs/M4/targets_a0.5.npy
bash ../code/launch/start.sh E17                                   # the 10 distilled students (lab-server queue)
bash ../code/launch/e17_post.sh 0,1,2,3,4,5,6,7,8,9                # M3 on the students' candidates, frontier, analysis, publishing
python -m preregistered_analyses.e18_final > ../docs/raw/E18_final.txt   # final numbers and docs/figures/E18_frontier.pdf
```

**Keys (optional).** For W&B logging or Hugging Face uploads, copy `code/.env.example` to `code/.env`
and fill it in. This submission contains no keys.
