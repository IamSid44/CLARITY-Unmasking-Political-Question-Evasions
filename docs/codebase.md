# Codebase

Where each file is, what it does, and its important functions. Every command runs from inside `code/`
(`python -m <package>.<module>`). Every path is defined once, in `code/qevasion/paths.py`.

1. [Layout](#1-layout)
2. [How a reported number is produced](#2-how-a-reported-number-is-produced)
3. [Packages and files](#3-packages-and-files)
4. [Run outputs](#4-run-outputs)

---

## 1. Layout

```text
code/
├── qevasion/                 shared library: data, labels, official scorer (imported by everything)
├── models/                   classifiers: DeBERTa encoder, Qwen3-8B + LoRA, encoder variants
├── decision_rules/           seed ensembling and logit adjustment (τ fitted by nested CV)
├── cascade/                  the cascade: M1 candidates → M2 routing → M3 LLM adjudicator → fusion; M4 distillation
├── audit/                    the frozen instrument applied to new inputs (E25), reduced precision (E26)
├── evaluation/               per-run metrics, run comparison, Codabench submission files
├── preregistered_analyses/   one script per experiment: the analysis behind each reported number → docs/raw/
├── figures/                  report and slide figures
├── utils/                    checkpoints, W&B / Hugging Face tracking, self-managing cloud VMs
├── experiments/              run lists for the training queue; frozen_system.json for E25–E26
├── launch/                   shell launchers: lab-server queue, cloud GPU lanes, collectors
└── .env.example              keys for W&B, Hugging Face and the cloud provider (copy to .env)
```

Dependencies point one way: `qevasion` ← `models`, `decision_rules` ← `cascade` ← `audit` ←
`preregistered_analyses` / `figures`. `utils` is used only by the training scripts and the launchers.

## 2. How a reported number is produced

```text
 qevasion.loader ──► models.llm_classifier ──► runs/<config>/seed<k>/{dev,test,val}_probs.npy
 (data, splits)      (one run per seed)                        │
                                                               ▼
                 decision_rules.decide  (mean of seeds, logit adjustment, τ by nested CV)  ──► M1 system
                                                               │
          cascade.candidates (top-4 C(x), uncertainty u(x)) ──► cascade.adjudicate (27B picks a letter)
                                                               │
                 cascade.fuse (route the δ least confident, fuse M1 and M3 over C(x)) ──► accuracy point
                                                               │
          preregistered_analyses.eXX (registered tests, bootstrap) ──► docs/raw/EXX_*.txt ──► report
```

1. **Data.** `qevasion/loader.py` loads QEvasion train (3,448 rows) and dev (308 items, three annotators)
   and builds each dev item's reference set. A fixed 10% train slice selects epochs and fits routing, so
   dev never selects anything during training.
2. **Training.** `models/llm_classifier.py` (or `models/encoder.py`) trains a 9-way classifier on
   `sub-question + full question + answer` and writes per-item probabilities per seed.
3. **Decision.** `decision_rules/decide.py` averages seeds and applies logit adjustment; τ is fitted and
   scored by nested cross-validation on dev, so the reported score is held out.
4. **Cascade.** `cascade/` defers the least confident items to a 27B LLM that picks among M1's top 4, and
   fuses the two distributions (fusion weights also by nested CV).
5. **Scoring.** `qevasion/scoring.py` reproduces the official scorers.
6. **Analysis.** `preregistered_analyses/eXX_*.py` tests the hypotheses registered in
   [`experiment_log.md`](experiment_log.md) and writes to `docs/raw/`. [`results_sources.md`](results_sources.md)
   maps every reported number to its script and output.

---

## 3. Packages and files

### `qevasion/`: shared library

| File | What it does | Important functions |
|---|---|---|
| `loader.py` | Loads train/dev (Hugging Face parquet, cached in `data/cache/`); reference sets, consensus levels, fixed splits | `load_qevasion` (the splits), `dev_reference_mask` (308 × 9 gold sets), `stratified_val_index` (the 345-row slice), `stratified_fold_index` (cross-fit folds), `consensus_level` |
| `labels.py` | The 9 evasion labels, the 3 clarity levels and the official 9 → 3 mapping | `encode_evasion`, `encode_clarity`, `leaf_to_official_clarity`, `multi_reference_mask` |
| `scoring.py` | Replica of the official scorers | `macro_f1_multireference` (Subtask 2: a prediction counts if any annotator gave it), `macro_f1_single_label` (Subtask 1), wrappers `score_subtask2` / `score_subtask1` |
| `normalize.py` | **E21.** Transcript artifacts of the official test answers | `normalize_answer` (removes speaker tags, footnotes, crosstalk), `inject_artifacts` (adds them to dev at test's rates), `test_inventory` |
| `paths.py` | Repository paths (`RUNS`, `LOGS`, `DATA`, `SPLITS`, `TEST_CSV`, `RAW`, `FIGURES`, `REPORT_FIGURES`); the `CLARITY_RUNS` environment variable moves `runs/` | — |

### `models/`: classifiers

| File | What it does | Important functions | Experiments |
|---|---|---|---|
| `llm_classifier.py` | Qwen3-8B-Base + LoRA (all linear layers) with a 9-way head on the last token. `--soft-targets` trains on distillation targets | `Cfg` (all hyper-parameters), `encode_row` (input text and token budgets), `build_model`, `run` (train, select epoch on the slice, save probabilities), `predict`, `padding_self_check` | E13, E14 (M1), E17 (M4) |
| `encoder.py` | DeBERTa-v3-large cross-encoder: input form, token budgets, loss (CE, Balanced Softmax, focal), flat / hierarchical / sub-task heads | `ClarityEncoder`, `build_inputs`, `make_loss`, `param_groups` (layer-wise LR decay), `run`, `predict`, `inset_diagnostics` | E0–E12 |
| `rerank.py` | Second-stage cross-encoder that re-ranks the top-K labels using label definitions | `RerankModel`, `load_stage1`, `choose` | E6 |
| `hier_combine.py` | Combines a Non-Reply gate, branch specialists and pair experts into 9-way predictions (CPU) | `compose`, `apply_experts`, `variants` | E12b/c |
| `soup.py` | Uniform and greedy weight averaging of trained encoders | `main` | E12d |

### `decision_rules/`

| File | What it does | Important functions |
|---|---|---|
| `decide.py` | Decision rules over saved probabilities, each fitted and scored by nested CV on dev | `logit_adjust` (divide by prior^τ), `fit_tau`, `nested_cv`, `make_rules` (the rule set; `R1_logit_adjusted` is used throughout), `load_probs` |
| `decode_variants.py` | Alternative decoders tested against logit adjustment: gating, optimal-transport class balancing, class floors | `rule_la`, `rule_gate`, `rule_ot`, `rule_floor` |

### `cascade/`: the confidence-routed cascade

| File | What it does | Important functions | Experiments |
|---|---|---|---|
| `candidates.py` | **M1.** From a seed ensemble: calibration on the slice, candidate sets C(x), uncertainty u(x) | `load_split`, `load_crossfit`, `fit_temperature`, `make_sets`, `build` (writes `runs/M1/<run>/cascade_inputs.npz`) | E14 |
| `prompts.py` | **M3's prompt:** task, label definitions, confusion guide, the item, lettered candidates | `build_messages`, `letter_map`, `preamble` | E15–E16 |
| `adjudicate.py` | **M3.** Scores M1's top 4 with an LLM: one request and one output token per item; local vLLM or an OpenAI-compatible API; resumable | `select` / `load_jobs` (which items), `build_prompts`, `letters_to_q` (letter log-probs → distribution over C(x)), `run_vllm`, `pad_preamble` (E22a cache alignment) | E15, E19, E22, E25 |
| `fuse.py` | **M2 + fusion.** Routes the δ least confident items, fuses M3's q with M1 over C(x), fits (λ, τ) | `load_q`, `decide`, `fit_system`, `dev` | E15 |
| `frontier.py` | Accuracy-vs-cost frontier over seed count K × deferral rate δ | `subsets`, `write_jobs`, `analyse` | E16 |
| `cost.py` | Measured GPU-seconds per item of one M1 seed and one M3 call | `time_predict` | E16 |
| `distill_targets.py` | **M4.** Soft targets (1 − α)·gold + α·q; `--teacher m1 \| uniform` for the controls | `main` | E17, E19 |
| `exemplars.py` | Embeds every row (Qwen3-Embedding) and retrieves train neighbours for few-shot prompts | `embed`, `Retriever` | E15 |
| `infer_variants.py` | M1 seeds and students on dev with test's artifacts injected, and on normalized test | `frames`, `main` | E21 |
| `specificity.py` | Question–answer specificity: answer log-likelihood after its own question vs random ones vs none | `answer_logprob` | E22e, E24 |
| `pass2_jobs.py` | The second 27B pass's (item, candidates) jobs | `jobs_of`, `pseudo_run` | E19–E23 |
| `api_budget.py`, `api_probe.py` | API price table, a spend ledger with a hard cap, and a capability probe | `Ledger`, `cost_usd`, `probe` | E15 |
| `publish.py` | Uploads cascade outputs to Hugging Face and logs result tables to W&B | `hf`, `wandb_log` | all |

### `audit/`: the frozen instrument on new inputs

| File | What it does | Important functions | Experiments |
|---|---|---|---|
| `system.py` | The **frozen instrument** (E23 accuracy point, every parameter fixed in `experiments/e25/frozen_system.json`) | `freeze` (fit once, write the JSON), `calibration`, `top4`, `decide`, `jobs` (route a frame, list 27B jobs), `records` (27B q averaged over 4 option orders), `label` | E25, E26 |
| `frames.py` | Input frames with one controlled change each: own / random / no question, question only, answer prefixes, LLM answers | `build`, `prefix`, `llm`, `clean` (strips markdown from LLM answers) | E25a/c |
| `generate.py` | Puts the Presidents' dev questions to open LLMs (vLLM) under four conditions: neutral (N), persona (P), follow-up (F), briefed (B) | `prompts`, `follow_up` | E25b |
| `infer.py` | The frozen M1 (10 seeds) on the frames, with the registered reproduction control | `main`, `batch1_latency` | E25 |
| `hidden.py` | Layer-wise linear probes on the frozen base model's hidden states; early-exit latency | `dump` (GPU), `probe` (CPU) | E25d |
| `quant.py` | M1 at reduced precision: bf16 reference, FP8 W8A8, NF4 with the adapter unmerged | `FP8Linear` (row-wise scaled `torch._scaled_mm`), `fp8_mode`, `to_fp8`, `load`, `latency` | E26 |
| `quant_system.py` | Routes and labels each precision's outputs through the frozen instrument | `inputs`, `jobs`, `records`, `label` (`--records-from` reuses another site's 27B records) | E26 |

### `evaluation/`

| File | What it does |
|---|---|
| `analyze.py` | Per-class F1, in-set rate, coverage and confusions for one run |
| `compare_runs.py` | Paired per-seed and ensemble comparison against a baseline; epoch curves |
| `summarize.py` | One table of all finished runs |
| `peek.py` | Inspects a run while it trains |
| `verify_scorer_geometry.py` | Regenerates every number in [`scorer_geometry.md`](scorer_geometry.md) |
| `make_submission.py` | Packages test predictions as Codabench zips into `submissions/` (`fit_tau_on_dev`, `write_zip`) |
| `make_cascade_submission.py` | Test files for the cascade systems → `submissions/E19_*/` (`predict`: fit on dev, predict test) |
| `make_final_submission.py` | **E23.** The final test files: normalized test text, one 27B copy, order-averaged q → `submissions/E23_*/` |

### `preregistered_analyses/`: one script per experiment

Each prints a report; the committed copy is in `docs/raw/`. Scripts are organised as `section_a`, `section_b`, …,
one per registered hypothesis, matching the sections of the output file.

| File | Output | Tests |
|---|---|---|
| `mideval_analysis.py` | `raw/mideval/` | Baselines, main table, bootstrap intervals, error analysis, human ceiling |
| `e11_replication.py` | `raw/E11_replication.txt` | DeBERTa findings on fresh seeds 5–9 |
| `e13_analysis.py`, `e13_topk.py` | `raw/E13_*.txt` | Qwen vs DeBERTa on paired seeds; top-k coverage |
| `qualitative_examples.py` | `raw/qualitative_examples.txt` | Side-by-side examples of the two 10-seed systems |
| `e14_m1_analysis.py` | `raw/E14_*.txt` | M1: epochs, seed-count curve, candidate coverage, uncertainty AUROC |
| `e17_m4_analysis.py` | `raw/E17_*.txt` | M4: students vs M1 seeds, adoption rule |
| `e18_final.py` | `raw/E18_final.txt` | Both operating points, cost table, frontier figure |
| `e19_tier0.py`, `e19_lib.py` | `raw/E19_tier0.txt` | Refitted bootstrap, corrected cost, routing and fusion ablations, human ceiling. `e19_lib` holds the vectorised fusion grid (`Grid`, `nested`, `system_inputs`) reused by later scripts |
| `e20_router.py` | `raw/E20_router.txt` | Value map and learned router on leak-free train rows |
| `e21_e22_analysis.py` | `raw/E21_E22_analysis.txt` | LLM alone, option order, cached prompt, test artifacts, specificity |
| `e23_systems.py` | `raw/E23_systems.txt` | Final operating points with one 27B copy; escalation ladder |
| `e24_prechecks.py` | `raw/E24_prechecks.txt` | P1–P8: sibling structure, does M1 read the question, five candidate modules, relevance vs answerhood |
| `e25_analysis.py` | `raw/E25_analysis.txt` | Controls, question swaps, question or speaker (`mh_or`: Mantel–Haenszel odds ratio; `rows_of`: cluster bootstrap over interview turns), prefixes, probes, label-free validity check |
| `e26_analysis.py` | `raw/E26_analysis.txt` | Each precision against 16-bit on the same GPU type; registered verdicts |

### `figures/`

| File | Produces |
|---|---|
| `paper_figures.py` | The report's data figures → `report/latex/figures/` |
| `deck_figures.py` | The slides' figures → `docs/figures/deck/` |
| `e25_figures.py` | E25 figures → `docs/figures/E25_*` and `report/latex/figures/` |
| `e13_figures.py`, `mideval_figures.py` | Mid-evaluation figures → `docs/figures/` |
| `paper_example.py`, `taxonomy_counts.py` | The report's worked example and taxonomy table |

### `utils/`

| File | What it does | Important functions |
|---|---|---|
| `ckpt.py` | Epoch-level resume checkpoints, written atomically | `save_resume`, `load_resume` |
| `tracking.py` | W&B logging and Hugging Face upload; no-ops when no keys are set | `Tracker`, `push_run`, `push_async`, `load_env` |
| `vm_guard.py` | Lets a rented VM look after itself: logs to HF, checked uploads, then pause or destroy | `heartbeat`, `upload`, `verify`, `stop` |

### `experiments/` and `launch/`

`experiments/<EXP>/` lists the runs of each queued training experiment: `common.args` holds shared flags and
directives (`@module`, `@hf-push`, `@need-gb`); each line of `lane_gpu<N>_<x>.txt` is one run
(`<name> <seed | fold<k>> [flags]`). `experiments/e25/frozen_system.json` is the frozen instrument.

| Launcher | Where it runs | What it does |
|---|---|---|
| `start.sh`, `run_queue.sh`, `gpu_watch.sh` | lab server (tmux) | Run queue: one lane per GPU, resume from the last epoch, upload finished runs; the watcher uses a shared GPU slice only while it is free |
| `download_model.sh`, `m3_local.sh`, `e16_gpu.sh`, `e17_post.sh` | lab server | Model download, local M3 scoring, the E16 and E17 GPU steps |
| `jarvis_drive.sh`, `jarvis_setup.sh` | laptop → cloud VM | Create, push code to, start and monitor a rented GPU instance; the training pipeline on it |
| `infer_lane.sh` | cloud VM | The inference lanes: E19–E22 (`PASS=1/2`), E25 (`PASS=3`), E26 FP8 (`PASS=4`); `PASS=5` (NF4 on the same GPU) was prepared but not run. Self-destroys when finished |
| `e25_collect.sh`, `e26_collect.sh` | laptop | Wait for a lane, copy its outputs from HF, confirm the instance is destroyed, run the analysis |
| `e26_lab_run.sh`, `e26_lab_resume.sh`, `e26_local_nf4.sh` | lab server | E26's NF4 half on two 16 GB GPUs |

How to run them: [`reproducing.md`](reproducing.md).

---

## 4. Run outputs

| Folder | Contents | Tracked in git |
|---|---|---|
| `runs/<config>/seed<k>/` | `dev_probs.npy`, `test_probs.npy`, `val_probs.npy`, `metrics.json`, `config.json` (+ LoRA adapter) | yes, except adapter weights |
| `runs/M1/`, `runs/M3/<tag>/`, `runs/M4/` | cascade inputs, 27B records (`responses.jsonl`), distillation targets | yes |
| `runs/E21/` … `runs/E26/` | outputs of the later experiments (frames, LLM answers, labels, precision runs) | yes, except E25 hidden states |
| `logs/` | training, queue and cloud-lane logs | yes |
| `data/` | cached train/dev, the official evaluation file, fixed splits | yes |

Model weights and E25's hidden states are on [Hugging Face](https://huggingface.co/siddarthg44/clarity-semeval26)
only; everything else is also mirrored there.
