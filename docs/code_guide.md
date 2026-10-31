# Code guide

How the code is organised, how a reported number is produced, and what each file does.
Every command runs from inside `code/`. Every path is defined once, in `code/qevasion/paths.py`.

## Contents

1. [Pipeline](#1-pipeline)
2. [Module dependencies](#2-module-dependencies)
3. [File reference](#3-file-reference)
4. [Generated folders](#4-generated-folders)

---

## 1. Pipeline

```
                      ┌───────────────────────────────────────────────┐
                      │ qevasion/  (imported by every module)         │
                      │   loader.py   train/dev from HF, reference    │
                      │               sets, fixed internal splits     │
                      │   labels.py   9 + 3 labels, 9 → 3 mapping     │
                      │   scoring.py  official-scorer replica         │
                      └───────────────────────┬───────────────────────┘
                                              │
         ┌────────────────────────────────────┼────────────────────────────────────┐
         ▼                                    ▼                                    ▼
 ┌────────────────────┐            ┌─────────────────────┐            ┌──────────────────────┐
 │ TRAIN              │            │ utils/              │            │ experiments/         │
 │ models/encoder.py  │◄───────────│  ckpt.py   resume   │            │  E*/common.args      │
 │ models/llm_        │            │  tracking.py W&B/HF │            │  E*/lane_*.txt       │
 │    classifier.py   │            └─────────────────────┘            └──────────┬───────────┘
 └─────────┬──────────┘                                                          │
           │ per seed: dev_probs.npy, test_probs.npy, val_probs.npy, metrics.json │
           ▼                                                                      │
   runs/<config>/seed<k>/  ◄──────────────────────────────────────────────────────┘
           │
           ├─► cascade/   M1 candidate sets and uncertainty → M3 LLM adjudicator → fusion → M4 distillation
           │
           ▼
 ┌──────────────────────────────┐     ┌────────────────────────────────┐     ┌──────────────────────┐
 │ DECIDE                       │     │ EVALUATE / ANALYSE             │     │ REPORT               │
 │ decision_rules/decide.py     │────►│ evaluation/*                   │────►│ figures/*            │
 │  seed ensemble + logit       │     │ preregistered_analyses/*       │     │ docs/raw/*.txt       │
 │  adjustment (τ by nested CV) │     │                                │     │ report/              │
 └──────────────────────────────┘     └────────────────────────────────┘     └──────────────────────┘
```

**The flow behind a reported number**

1. **Data.** `qevasion/loader.py` downloads QEvasion train (3,448 rows) and dev (308 items, three
   annotators) and builds each dev item's reference set (every label an annotator gave). A fixed 10%
   slice of train (`data/splits/train_internal_val_index.json`) is held out to choose the epoch, so dev
   never selects anything during training.
2. **Training.** `models/encoder.py` (DeBERTa-v3-large) or `models/llm_classifier.py`
   (Qwen3-8B-Base + LoRA) trains a 9-way classifier on `sub-question + full question + answer`. Both use
   the same rows, input text, token budgets and epoch-selection rule, so only the backbone differs. Each
   run writes per-item probabilities and metrics to `runs/<config>/seed<k>/`.
3. **Decision.** `decision_rules/decide.py` averages the seeds' probabilities and applies logit
   adjustment: each class probability is divided by its training prior raised to τ. The single parameter
   τ is fitted and scored by nested cross-validation on dev, so the reported score is held out.
4. **Cascade.** `cascade/` builds the top-4 candidate set and an uncertainty score from M1, sends the
   least confident items to an LLM that picks one candidate (M3), and fuses the two distributions.
   M4 trains a single student on soft targets from M3.
5. **Scoring.** `qevasion/scoring.py` computes multi-reference macro-F1 for Subtask 2 and macro-F1 for
   Subtask 1 (9-way prediction mapped to 3-way), matching the official scorer.
6. **Analysis.** The scripts in `preregistered_analyses/` test hypotheses registered before each run
   (in `docs/02_experiment_log.md`) with paired per-seed comparisons and bootstrap intervals, and write
   to `docs/raw/`.

## 2. Module dependencies

| Module | Imports from |
|---|---|
| everything | `qevasion` |
| `models.encoder` | `utils.ckpt`, `utils.tracking` |
| `models.llm_classifier` | `utils.tracking` |
| `models.rerank`, `models.soup`, `evaluation.compare_runs` | `models.encoder` |
| `decision_rules.decode_variants`, `models.hier_combine`, `cascade.candidates`, most of `preregistered_analyses` and `figures` | `decision_rules.decide` |
| `figures.paper_example`, `preregistered_analyses.qualitative_examples` | `preregistered_analyses.e13_analysis` |
| `preregistered_analyses.e14_m1_analysis` | `cascade.candidates`, `preregistered_analyses.e13_analysis` |
| `cascade.adjudicate` | `cascade.prompts` |
| `cascade.fuse` | `cascade.adjudicate`, `decision_rules.decide` |
| `cascade.frontier` | `cascade.candidates`, `cascade.fuse`, `decision_rules.decide` |
| `cascade.cost` | `models.llm_classifier` |
| `preregistered_analyses.e18_final` | `cascade.frontier`, `cascade.fuse`, `decision_rules.decide` |
| `preregistered_analyses.e19_tier0`, `evaluation.make_cascade_submission` | `preregistered_analyses.e19_lib`, `cascade.fuse`, `cascade.frontier`, `cascade.candidates`, `decision_rules.decide`, `evaluation.make_submission` |
| `audit.system` | `cascade.candidates`, `decision_rules.decide`, `preregistered_analyses.e19_lib`, `preregistered_analyses.e23_systems` |
| `audit.infer`, `audit.quant`, `audit.hidden` | `models.llm_classifier` (`audit.hidden` also `cascade.candidates`, `decision_rules.decide`) |
| `audit.quant_system`, `preregistered_analyses.e25_analysis`, `preregistered_analyses.e26_analysis` | `audit.system` |
| `preregistered_analyses.e24_prechecks` | `cascade.fuse`, `cascade.prompts`, `decision_rules.decide` |

## 3. File reference

### `qevasion/`: shared library

| File | Purpose |
|---|---|
| `loader.py` | Loads train/dev from the Hugging Face parquet files (cached in `data/cache/`); builds multi-annotator reference sets, consensus levels and the fixed splits |
| `labels.py` | The 9 evasion labels, the 3 clarity labels, the official 9 → 3 mapping, label encoders |
| `scoring.py` | Replica of the official scorer: multi-reference macro-F1 (Subtask 2) and macro-F1 (Subtask 1) |
| `paths.py` | Every repository path; `CLARITY_RUNS` moves `runs/` |
| `normalize.py` | **E21.** Removes the official test's transcript artifacts (speaker tags, footnotes, headers, other speakers' turns); injects them into dev at test's rates |

### `models/`: classifiers

| File | Purpose | Experiments |
|---|---|---|
| `encoder.py` | DeBERTa-v3-large cross-encoder. Options: input form (`qa` or `full`), token budgets, loss (CE, Balanced Softmax, focal), flat or hierarchical head, sub-task heads (`--task gate/other6/nr3/pair:A,B`), all-data training | E0–E12 |
| `llm_classifier.py` | Qwen3-8B-Base with LoRA (r = 16, all linear layers) and a 9-way head on the last token; same data, input and selection rule as the encoder. `--soft-targets` trains on M4's distillation targets | E13, E14 (M1), E17 (M4) |
| `rerank.py` | Second-stage cross-encoder that re-ranks the top-K labels using label definitions | E6 |
| `hier_combine.py` | Combines a Non-Reply gate, branch specialists and pair experts into 9-way predictions (CPU) | E12b, E12c |
| `soup.py` | Uniform and greedy weight averaging of trained encoders | E12d |

### `decision_rules/`

| File | Purpose |
|---|---|
| `decide.py` | Decision rules over saved probabilities: argmax, logit adjustment (τ by nested CV), set membership, per-class weights; `--drop-annotator` re-scores on 2-annotator reference sets |
| `decode_variants.py` | Further decoding variants (Non-Reply gate threshold, prior matching, coverage floor) |

### `cascade/`: the confidence-routed cascade

| File | Purpose |
|---|---|
| `candidates.py` | **M1.** From a seed ensemble's probabilities, builds candidate sets C(x) and an uncertainty score u(x), for the train slice, dev, test and cross-fitted train rows |
| `prompts.py`, `exemplars.py` | **M3** prompt (label definitions, confusion guide, lettered candidates) and retrieval of train examples |
| `adjudicate.py` | **M3.** Scores M1's top-4 candidates with an LLM (local vLLM or an OpenAI-compatible API); one request and one output token per item; resumable |
| `fuse.py` | Fuses the LLM's distribution with M1's over C(x) and defers by confidence (**M2**) |
| `frontier.py` | Accuracy-vs-cost frontier over seed count × deferral rate (E16) |
| `cost.py` | Measured GPU-seconds per item of an M1 seed and an M3 call |
| `distill_targets.py` | **M4.** Soft targets (1 − α)·gold + α·q from M3; `--teacher m1 \| uniform` gives the E19 controls |
| `infer_variants.py` | **E21.** M1 seeds and students on dev with test's transcript artifacts injected, and on normalized test text |
| `specificity.py` | **E22e.** Question–answer specificity: answer log-likelihood after its own sub-question vs random ones vs none |
| `pass2_jobs.py` | The second 27B pass's (item, candidates) jobs: E19-E controls, E22c ladder, normalized test (E21b) |
| `api_budget.py`, `api_probe.py` | API price table, spend ledger with a hard cap, and an API capability probe |
| `publish.py` | Uploads cascade outputs to Hugging Face and logs results to W&B |

### `audit/`: the frozen instrument applied to new inputs (E25–E26)

| File | Purpose |
|---|---|
| `system.py` | The frozen instrument: `freeze` fits the E23 accuracy point once and writes `experiments/e25/frozen_system.json`; `jobs` routes a frame's items and lists the 27B jobs; `label` gives final labels from M1 + the 27B records. Nothing is refitted on new inputs |
| `frames.py` | **E25a/c.** Input frames with one controlled change each: own, random and empty question, question only, answer prefixes (25/50/75%), and the LLM answers (`llm`) |
| `generate.py` | **E25b.** Puts the Presidents' dev questions to open LLMs (vLLM) under four conditions: neutral analyst (N), persona (P), follow-up (F), briefed (B) |
| `infer.py` | The frozen M1 (10 seeds) on the E25 frames, with the registered reproduction control |
| `hidden.py` | **E25d.** Layer-wise linear probes on the frozen base model's hidden states, and early-exit latency |
| `quant.py` | **E26.** M1 at reduced precision: bf16 reference, FP8 W8A8 (`torch._scaled_mm`), NF4 (bitsandbytes, adapter unmerged) |
| `quant_system.py` | **E26.** Routes and labels each precision's M1 outputs through the frozen instrument |

### `evaluation/`

| File | Purpose |
|---|---|
| `analyze.py` | Per-class F1, in-set rate, coverage and confusions for one run |
| `compare_runs.py` | Paired per-seed and ensemble comparison against a baseline |
| `summarize.py` | One table of all finished runs |
| `verify_scorer_geometry.py` | Regenerates every number in `docs/01_scorer_geometry.md` |
| `peek.py` | Inspects a run while it trains |
| `make_submission.py` | Packages test predictions as Codabench zips into `submissions/` |
| `make_cascade_submission.py` | Test files for the cascade systems (M1, accuracy point, cost point) → `submissions/E19_*/` |
| `make_final_submission.py` | **E23.** The final test files: normalized test text, the VM's 27B, order-averaged q → `submissions/E23_*/` |

### `preregistered_analyses/`

| File | Output |
|---|---|
| `mideval_analysis.py` | Baselines, main table, bootstrap intervals, error analysis, human ceiling → `docs/raw/mideval/` |
| `e11_replication.py` | Replication of the DeBERTa findings on seeds 5–9 → `docs/raw/E11_replication.txt` |
| `e13_analysis.py`, `e13_topk.py` | Qwen vs DeBERTa (paired seeds and systems); top-k coverage and confidence → `docs/raw/E13_*.txt` |
| `qualitative_examples.py` | Side-by-side examples of the two 10-seed systems |
| `e14_m1_analysis.py` | M1: epochs, seed-count curve, candidate coverage, uncertainty AUROC, LLM headroom |
| `e17_m4_analysis.py` | M4: distilled student vs M1 seeds, routing, student + M3 frontier |
| `e18_final.py` | Final numbers for both operating points, cost table, frontier figure → `docs/raw/E18_final.txt` |
| `e19_tier0.py`, `e19_lib.py` | E19 Tier 0: headlines with refitted bootstrap, corrected cost, train-row cascade, value map, routing and fusion ablations, strata, position bias, test files, human ceiling → `docs/raw/E19_tier0.txt` |
| `e21_e22_analysis.py` | E19 A/B, E21a, E22a/b/e: the LLM alone, option order, cached prompt, faithfulness, test-artifact sensitivity, specificity → `docs/raw/E21_E22_analysis.txt` |
| `e20_router.py` | E20 (amended): value map and learned router on the leak-free singleton train rows → `docs/raw/E20_router.txt` |
| `e23_systems.py` | Final operating points with one 27B copy (single and order-averaged q), E19-E system part, E22c ladder → `docs/raw/E23_systems.txt` |
| `e24_prechecks.py` | E24: CPU pre-checks P1–P8 of the candidate research modules on existing outputs → `docs/raw/E24_prechecks.txt` |
| `e25_analysis.py` | E25: controls, question swaps, question or speaker, answer prefixes, probes, label-free validity check → `docs/raw/E25_analysis.txt`, `runs/E25/analysis.json` |
| `e26_analysis.py` | E26: each precision against 16-bit on the same GPU type; registered verdicts → `docs/raw/E26_analysis.txt` |

### `figures/`

| File | Produces |
|---|---|
| `paper_figures.py` | The report's data figures, written to `report/latex/figures/` |
| `deck_figures.py` | The slides' figures, written to `docs/figures/deck/` |
| `paper_example.py` | The report's worked example (one dev item through both systems) |
| `taxonomy_counts.py` | Per-class counts for the taxonomy table |
| `e13_figures.py`, `mideval_figures.py` | Figures in `docs/figures/` |
| `e25_figures.py` | E25 figures (audit; swaps, prefixes and probes) in `docs/figures/` and `report/latex/figures/` |

### `utils/`

| File | Purpose |
|---|---|
| `ckpt.py` | Atomic checkpoint saving and resume after interruption |
| `tracking.py` | W&B logging and Hugging Face upload of metrics, probabilities and adapters; skipped when no keys are set |
| `vm_guard.py` | Lets a rented VM look after itself: copies logs to HF, uploads and checks outputs there, and pauses or destroys its own instance |

### `launch/` and `experiments/`

`launch/` holds the scripts used to run experiments on our hardware: a tmux run queue for a shared GPU
server (`start.sh`, `run_queue.sh`, `gpu_watch.sh`, `m3_local.sh`, `e16_gpu.sh`, `e17_post.sh`,
`download_model.sh`) and a driver for rented cloud GPU instances (`jarvis_drive.sh`, `jarvis_setup.sh`, and
`infer_lane.sh` for the inference lanes: E19–E22, E25 (`PASS=3`) and E26's FP8 half (`PASS=4`)).
`e25_collect.sh` and `e26_collect.sh` run on the laptop: they wait for a lane, copy its outputs from HF, make sure
the instance is destroyed, and run the analysis. E26's NF4 half ran on the lab's A4000s with `e26_lab_run.sh`
and `e26_lab_resume.sh` (`e26_local_nf4.sh` is the same run for a machine with three free 16 GB GPUs).
They are specific to that infrastructure; every run can also be started directly with `python -m`.
- **Running the cloud driver from Windows Git Bash:** set `JL_PY` to a Python that has the `jarvislabs`
  SDK installed, instead of using `uvx`.
- **Several seeds on one large GPU:** `PARALLEL=N` trains up to N seeds at once.
- **Self-managing lanes:** both pipelines copy their logs to HF and run a watchdog. At the end they destroy
  the instance if everything is on HF, and pause it otherwise (`utils/vm_guard.py`).
- **The inference lane:** `jarvis_drive.sh lane` launches it.

Usage, machine choice and costs are in
[`06_final_push_handover.md`](06_final_push_handover.md) §5–6.

`experiments/<EXP>/` lists the runs of each queued experiment. `common.args` holds the shared
arguments; each line of a `lane_gpu<N>_<x>.txt` file is one run: `<run name> <seed | fold<k>> <extra args>`.

### `slides/`

`build_deck.py` built an early PowerPoint version of the mid-evaluation deck (needs `python-pptx`).

## 4. Generated folders

These are created when the code runs:

- `runs/`: one folder per configuration and seed, with probabilities, metrics and checkpoints. The
  probabilities, metrics, configs, 27B records and labels behind the reported numbers are tracked; model
  weights and E25's hidden states are not, and are on
  [Hugging Face](https://huggingface.co/siddarthg44/clarity-semeval26) with everything else.
- `logs/`: training, queue and cloud-lane logs (tracked).
- `data/cache/`: the downloaded train and dev parquet files.
