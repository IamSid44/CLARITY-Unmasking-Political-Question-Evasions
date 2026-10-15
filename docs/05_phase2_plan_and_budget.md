# 05 — Phase 2: from the mid-evaluation to the final evaluation

This document is the working plan after the mid-evaluation (submitted 2026-10-02 with
`Report/Report.pdf`; slides `Slides/CLARITY_mideval.pptx`). It covers four things:
- what we are building, and why;
- the protocol every experiment follows;
- the compute budget and how it has been spent;
- how runs are launched.

The experiment-by-experiment record stays in [`02_experiment_log.md`](02_experiment_log.md), and every
hypothesis is registered there before its run starts.

---

## 1. Where the mid-evaluation left us

| System (dev, 308 items) | Subtask 2 | Subtask 1 |
|---|---|---|
| DeBERTa-v3-large, 10-seed system | 0.405 | 0.648 |
| Qwen3-8B-Base + LoRA (3 epochs), 10-seed system | **0.543** | **0.746** |
| Qwen, 12 epochs, single model (3-seed screen, E13b) | 0.543 ± 0.015 | 0.742 |
| **Qwen, 12 epochs, 10-seed system (E14 = M1, after the mid-evaluation)** | **0.598** | **0.743** |
| TeleAI (1st place), multi-call DeepSeek-V3 pipeline | 0.617 | 0.812 |

Two facts from the analysis decide what comes next (report §6):
- **The model ranks better than it decides.** For Qwen, an acceptable label is first for 60.4% of
  items and in the top three for 93.1%.
- **Its confidence separates easy items from hard ones.** The in-set rate is 0.87 in the most confident
  fifth and 0.40 in the least confident.

So most remaining errors are a choice among a few labels the model already ranks highly, on items it
can flag as uncertain.

## Status (2026-10-15)

| Module | State |
|---|---|
| **M1** | **done (E14).** 10-seed 12-epoch system S2 0.598 / S1 0.743; cross-fitted train probabilities for all 3,448 rows; C(x) and u(x) code; all runs on HF and W&B. Results against H1–H6 are in the log, §E14 results. |
| **M3** | **done (E15, 2026-10-14; corrected for a fusion bug the same day).** Local Qwen3.8-27B-FP8 over M1's top 4, no examples, fused on the 30% least confident items: **dev S2 0.669 (10 splits 0.660), +0.067 [+0.009, +0.124]; S1 0.798; 0.36 calls/item**. δ = 0.2 gives 0.635 at 0.23 calls. DeepSeek V4 Flash and Gemma 4 31B (API) not adopted. API spend $0.26. Next: adopt δ = 0.3, then M4. |
| **M2** | **done (E16, 2026-10-14).** Prompt-part ablation (the full prompt kept), measured cost per item (an M1 seed 0.052 GPU-s, an M3 call 0.103), and the accuracy-vs-cost frontier over seeds × deferral. |
| **M4** | **done (E17, 2026-10-15).** A distilled student replaces the 10 seeds: student + M3 at δ = 0.5 scores 0.608 / 0.790 at 0.111 GPU-s per item, the cost operating point. |
| **Freeze (E18)** | **numbers done (2026-10-15).** Final dev numbers for both operating points, incl. 2-annotator; cost table; frontier figure (`docs/figures/E18_frontier.pdf`). Ensembles of distilled students tested: no middle operating point. **Left for later (the user):** report §7, slides, submission files. |

## 2. The plan: a confidence-routed cascade (report §7, Figure 5)

**Goal:** at least **0.60 dev Subtask 2** with **at most 0.3 LLM calls per item**. The winning pipeline
makes 1.94 calls per item.

| Module | What it is | Why it should help | Exit criterion | Dates |
|---|---|---|---|---|
| **M1** candidate generator | Qwen3-8B + LoRA, 12 epochs, K seeds, logit adjustment. It outputs p̄(c\|x), a candidate set C(x) and an uncertainty score u(x). It also cross-fits the training set. | It is the strongest single-pass model we have. C(x) narrows the LLM's choice, u(x) decides what to defer, and the cross-fitted candidates feed M3/M4. | gold in C(x) for ≥ 95% of slice items (see §2.1) | Oct 5–11 |
| **M3** LLM adjudicator | An open-weight instruction LLM (vLLM) scores only the labels in C(x). Its prompt holds the candidates' definitions, a confusion guide for our three largest error pairs, and boundary examples retrieved from train. It is fused with p̄ as λ log p̄ + (1−λ) log q − τ log π. | Label meaning and world knowledge from a larger model, which the winning systems relied on. In TeleAI's final stage, definitions + confusion guide took S2 0.491 → 0.593, and boundary examples took it to 0.617. | beats M1 on the deferred slice items | Oct 12–18 |
| **M2** router | Defer the fraction δ with the highest u(x); the threshold is the (1 − δ) quantile on the slice. | It spends LLM calls only where M1 is unsure; calls per item = δ. | an accuracy-vs-calls curve | Oct 19–25 |
| **M4** distillation | M3's decisions on train become soft targets for retraining M1; 4-bit inference. | A sharper M1 needs fewer deferrals for the same accuracy. | gain over the control above the seed spread | Oct 26–29 |
| — | Freeze; final 10-seed and 2-annotator scores; release | | final submission | Oct 30–31 |

### 2.1 A finding that changes M1's exit criterion (2026-10-06, before any E14 run)

The 12-epoch screen seeds (E13b, seeds 0–2) were analysed on the **train slice only** (single gold
label). The full output is in `raw/E14_m1_design_seeds012.txt` §C.
- **Large sets are needed for 95% coverage.** The gold label ranks first for 55% of slice items, but
  4th or lower for 22%. Every set rule needs **5–7 of the 9 labels** to reach 90–95% slice coverage:
  adaptive sets, rank-conformal top-k, or temperature-scaled.
- **The 12-epoch models are overconfident.** Mean top probability is 0.81 on the slice (0.87 on dev),
  against 0.57 for 3 epochs. Without a temperature (fitted on the slice, T ≈ 3.9), mass-based sets
  degenerate to all nine labels.

So "gold in C(x) for ≥ 95% of slice items" cannot be met with a *small* C(x). The decision on how C(x)
feeds M3 is recorded in the E14 entry of the log. The options:
- a fixed top-3 (slice coverage 0.78, dev any-reference 0.92);
- letting M3 score all nine labels, fused with p̄, with C(x) choosing only which definitions and
  examples go into the prompt.

**Resolved by E14 at 10 seeds (2026-10-06).**
- **C(x) must be defined by rank, not by probability mass.** Mass thresholds fitted on the 10-seed slice
  probabilities did not transfer to the single-model cross-fitted rows: coverage 0.76 against 0.98 on
  the slice. Rank-based sets did: top-k 0.90 against 0.91.
- **Fixed top-3** covers 0.81 of slice items, 0.77 of cross-fitted train rows, and 0.925 of dev items
  (any reference).
- **u(x) is 1 − top probability.** The logistic regression did not beat it.
- **The oracle headroom is large enough to justify M3's budget.** Deciding the 30% most uncertain items
  perfectly within the top 3 would give 0.727, against M1's 0.598.

## 3. Protocol (unchanged since E8)

1. **Register first.** Every experiment's hypotheses, predicted numbers and decision rule go into
   `02_experiment_log.md` before the run starts. Results are then written against them, including
   when they are wrong.
2. **One change at a time,** each against a named control with paired seeds (the same seed numbers in
   both arms).
3. **Screen, then confirm.** A variant must first gain at least +0.015 S2 with at least 2 of 3 seeds
   up. Any system claim needs **10 seeds**: 5-seed systems swung by up to 0.08 on 308 items (E11).
4. **Nothing is chosen on dev.** Epochs, thresholds, temperatures and prompts are fitted on the 345-row
   train slice (or the cross-fitted train rows). The single decision-rule scalar τ on dev is fitted by
   5-fold nested CV, so the reported score is held out.
5. **Report the test regime too.** Dev is rescored against each 2-annotator reference set, as the test
   set has two annotators.

## 4. Compute and budget

### 4.1 Ledger

| When | What | Where | Cost |
|---|---|---|---|
| 2026-09-19 → 09-29 | E0–E12 (DeBERTa, ~120 runs) | lab server, RTX PRO 6000 (shared) | free |
| 2026-10-02 | E13 screen, setup and pilots | JarvisLabs, 1 × RTX PRO 6000 | ₹331.91 |
| 2026-10-02 | E13x / E13b / E13c (4 VMs) | JarvisLabs | ₹1,518.79 |
| 2026-10-06 | E14 (M1): 7 seeds + 5 folds, ~30 GPU-hours over 20 h of wall time (GPU 0 + two MIG slices) | lab server | free (≈ ₹2,300 at Jarvis rates) |
| 2026-10-13 | E15 (M3): API probe, screen and secondary dev arms through the JarvisLabs Model API; local Qwen3.8-27B arms on the lab server (free) | Jarvis API + lab server | $0.262 ≈ ₹23 (ledger `runs/M3/api_ledger.jsonl`) |
| **balance** | | | **₹4,001.81** (read through the JarvisLabs SDK on 2026-10-14 16:40; the API spend was charged to the same credit) |

**JarvisLabs prices on 2026-10-14** (₹/h on demand, spot in brackets; the SDK's `account.gpu_availability()`):
- RTX PRO 6000 96 GB: 207 (109);
- H100 80 GB: 330 (146);
- H200 141 GB: 435 (217);
- A100 40 GB: 84 (75);
- L4 / A30 24 GB: 41 / 39.

E17's training needs ≈ 53 GB without gradient checkpointing.
- **Per seed:** about ₹250 on either an RTX PRO 6000 (~62 min) or an H100 (est. ~35–40 min).
- **E17's confirmation (seeds 3–9)** runs on the free lab server (option A, chosen by the user). A move to
  Jarvis spot GPUs is kept open if the server becomes busy.

### 4.2 Plan for the remaining credit, and what was actually spent

| Phase | Work | Where | Planned ₹ | **Actual** |
|---|---|---|---|---|
| M1 (E14) | 12-epoch seeds 3–9, 5 cross-fit folds | lab server | 0 | **0** |
| M3 (E15) | local Qwen3.8-27B-FP8; API probe, screen and secondary arms (DeepSeek V4 Flash, Gemma 4 31B) | lab server + Jarvis Model API | 800–1,150 | **≈ ₹23 ($0.262)** |
| M2 (E16) | prompt ablation, cost timing, frontier | lab server (GPU 0) + CPU | 0 | **0** |
| M4 (E17) | M3 on all train rows; 10 distilled students | lab server (GPU 0, 10 × 76 min) | 1,750–2,100 | **0** (user chose the server over Jarvis) |
| freeze (E18) | final numbers, figures, report | CPU, ~5 min GPU at most | ~600 reserve | — |
| **balance** | | | | **₹4,001.81** (JarvisLabs SDK, 2026-10-14) |

**Rules:**
- Training that fits on the shared server runs there.
- Jarvis is kept for work that needs a whole 96 GB card (a large LLM) or that must finish under the
  deadline.
- Every Jarvis round starts with a timing pilot capped by `MAX_TOTAL_HOURS` (`code/launch/jarvis_setup.sh`).
- Each round's cost is added to the ledger above.
- **Fallback:** if the server becomes unavailable during M1, the seeds move to Jarvis (~₹1,400) and
  M4's retrain shrinks to 5 seeds.

## 5. Running on the lab server

```bash
bash code/launch/gpu_watch.sh E14 start   # watcher (tmux clarity-E14-watch): model download -> E14 lanes -> slice
bash code/launch/gpu_watch.sh E14 status  # what runs where; who else is on our GPU-1 slice
bash code/launch/gpu_watch.sh E14 release # give the slice back now (and keep off it); `allow` undoes this
tmux attach -t clarity-E14                # the lanes; Ctrl-b n/p switch windows, Ctrl-b d detach
tail -f logs/E14_watch.log logs/E14.log   # watcher events and queue events, one line each
tail -f logs/Q8_fullq_lora_12ep_seed3.log # one run's training output (loss, ETA, per-epoch scores)
```

`start.sh E14` alone starts only the normal lanes (no slice, no download step).

**Using spare capacity (`code/launch/gpu_watch.sh`).** The watcher starts the normal lane(s) once, then
looks at our GPU-1 MIG slice (GI 1, 48 GB, 2/7 of the card) every 10 minutes:
- **Starting the slice lane.** When no other user has a process on the slice and at least 30 GB is
  free, it starts the slice lane (`lane_gpu1_a.txt`, marked `# start: watcher`). That lane holds the
  same runs in reverse order.
- **Yielding the slice.** While the lane holds the slice, the watcher checks every 2 minutes. As soon as
  another user's process appears there, or on `release`, it stops the lane. The run that was training
  loses at most its current epoch.
- **Picking up orphaned runs.** Lanes re-check their lists every 10 minutes, so whichever lane is free
  next resumes the stopped run from `resume.pt`. W&B continues the same run.
- **Leaving GPU 0 alone.** The watcher never stops the GPU-0 lane. To free GPU 0, run
  `tmux kill-window -t =clarity-E14:lane_gpu0_a`; the run resumes later from its last epoch.
- **The second slice (GI 2).** It belongs to another group. From 2026-10-06 (user's request) it has its own lane, `lane_gpu1_b.txt`, under the same rules: used only while no one else is on it, and released within 2 minutes when someone is.
- **Making room on GPU 0 for a lab-mate without stopping.** Restart the GPU-0 lane with `QUEUE_NEED_GB=999`, which forces checkpointing (~21 GB instead of ~56 GB). Do it right after an epoch finishes, so no progress is lost (done 2026-10-06 16:05 for seed 9):
  `tmux kill-window -t =clarity-E14:lane_gpu0_a; tmux new-window -d -t =clarity-E14: -n lane_gpu0_a "env QUEUE_NEED_GB=999 bash $PWD/code/launch/run_queue.sh $PWD/code/experiments/E14/lane_gpu0_a.txt"`
- **Using freed GPU-0 memory.** If memory frees up on GPU 0 (the idle vLLM), the next GPU-0 run starts
  without gradient checkpointing on its own.

**The experiment folder.** An experiment is `code/experiments/<EXP>/`:
- `common.args` holds the shared flags plus directives: `@module`, `@hf-push`, `@need-gb`.
- Each `lane_gpu<N>_<x>.txt` lists runs as `<name> <seed|fold<k>> [extra flags]`.

**What the queue does** (`code/launch/run_queue.sh`):
- skips finished runs and resumes interrupted ones from their last epoch;
- locks each run so that lanes can share a list;
- uploads each finished run to the HF repo (`siddarthg44/clarity-semeval26`, under `<name>/seed<k>/`
  or `<name>/fold<k>/`);
- logs every epoch to W&B (`clarity-semeval26`, grouped by run name).

**GPU memory.** Qwen without gradient checkpointing needs ~53 GiB.
- When the shared GPU has less than `@need-gb` free, the queue waits 30 min once, then runs with
  gradient checkpointing (the same computation, about 1.35× slower). A device smaller than
  `@need-gb`, such as the 48 GB slice, always uses checkpointing.
- A run does not start at all below `@min-gb` (24 GB) free; the queue re-checks every 5 minutes.
- An out-of-memory crash is resumed from its last epoch with checkpointing.

**Server rules** (each learnt from an incident in the log):
- One lane per GPU.
- Never kill the process whose command line starts with `tmux new-session`: it is the tmux server. Use
  `tmux kill-session -t =clarity-<EXP>` (the `=` matters: without it tmux may match `clarity-<EXP>-watch` by prefix).
- Never edit a running bash script in place. Write a copy and `mv` it over the original.
- Smoke tests run without `--track`, without `@hf-push`, and into a scratch `CLARITY_RUNS`.

## 6. Running M3 (E15)

Code is in `code/cascade/`. Local runs use their own venv, `/scratch/shlok/Temp/.venv-vllm` (vLLM 0.31.0), so the training venv is untouched. API runs need `JARVIS_API_KEY` in `code/.env`, which is gitignored.

| step | command (from `code/`) | where |
|---|---|---|
| models | `bash ../code/launch/download_model.sh Qwen/Qwen3.8-27B-FP8` (also `Qwen/Qwen3-Embedding-0.6B`) | server, resumable |
| embeddings | `python -m cascade.exemplars embed` → `runs/M3/embeddings/` | server GPU, ~2 min |
| API probe | `$API_PY -m cascade.api_probe > ../docs/raw/E15_api_probe.txt` | API, cap $1 |
| local screen | `bash ../code/launch/m3_local.sh screen` (k = 0, 4, 10) | GPU 0, free |
| API screen | `$API_PY -m cascade.adjudicate --backend api --model <id> --tag <tag> --k <k> --splits screen --dry-run`, then without `--dry-run` (thinking is switched off by default) | API |
| screen analysis | `python -m cascade.fuse screen --tags <tag>,<tag>,...` | CPU |
| full scoring | `--splits slice,dev,test,train50` with the adopted arm (screen rows are copied, not re-paid) | GPU 0 / API |
| dev system and M2 curve | `python -m cascade.fuse dev --tag <tag>` | CPU |

`$VLLM_PY` is `/scratch/shlok/Temp/.venv-vllm/bin/python`: vLLM 0.31.0, a copy of `/scratch/shlok/venvs/vllm` plus pandas, pyarrow and scikit-learn. `$API_PY` is the training venv's python with the openai client on the path: `PYTHONPATH=/scratch/shlok/Temp/.pylib-openai /scratch/shlok/Temp/.venv/bin/python`. E15's API runs used this.

**Spend guard.** Every API response's usage goes to `runs/M3/api_ledger.jsonl`. Requests stop when either
limit is reached: the E15 cap of $8 (`cascade/api_budget.GLOBAL_CAP_USD`), or the run's own `--max-usd`
(default $1). A run whose dry-run projection exceeds `--max-usd` refuses to start.

**E16 and E17** (from the repository root):
- `bash code/launch/e16_gpu.sh`: M3 on all train rows and the frontier pairs, the prompt ablation, and the
  cost timing. Then `python -m cascade.frontier analyse`, from `code/`.
- `bash code/launch/start.sh E17`: the distilled students (`code/experiments/E17/`; W&B + HF via the queue).
- `bash code/launch/e17_post.sh 0,1,2`: waits for the students, then runs M3 on their candidates, the
  frontier, `preregistered_analyses/e17_m4_analysis.py` and publishing.

**Where things run, and logging (the user's rule, 2026-10-14).**
- **Placement:** if the lab server's GPU is reasonably free and there is no real time pressure, use it, to
  save credits. If the GPU is busy *and* time is short, move to JarvisLabs (parallel VMs are fine).
- **Logging:** wherever a run happens, it is logged to `logs/`, HF and W&B.
  - Training runs do this through the queue (`--track`, `@hf-push`).
  - Everything else uses `python -m cascade.publish hf M3 M4` and `python -m cascade.publish wandb <EXP>`.

**Resuming.** Outputs go to `runs/M3/<tag>/responses.jsonl`, one line per item, appended as results arrive.
A rerun skips finished items and retries failed ones. A tag's `config.json` pins the model, k, the prompt
version and the switches, so a changed configuration needs a new tag.

## 7. Next phase: plan of action after M3 *(2026-10-14; registered as E16 and E17 in the log)*

Decisions (2026-10-14): α = 0.5 fixed; M3's q as the teacher; bf16 (+ FP8 if supported), no 4-bit; the lab server's GPU 0 first, Jarvis in parallel where it saves time.

### 7.1 Where we stand, and what is left of the report's plan

| Module | Report plan | State |
|---|---|---|
| M1 | 12-epoch Qwen, 10 seeds, C(x), u(x) | done (E14): S2 0.598 / S1 0.743 |
| M3 | LLM adjudicator over C(x), fused with M1 | done (E15): **S2 0.669 / S1 0.798 at 0.36 calls/item** |
| M2 | router, δ sweep, **ablation of each prompt part**, accuracy-vs-calls curve | **done (E16, 2026-10-14)**: prompt ablation (`full` kept), measured cost, frontier over K × δ |
| M4 | distil M3 into M1; measured cost per item | **done (E17, 2026-10-15)**: student + M3 is the cost operating point (δ = 0.5: 0.608 / 0.790 at 0.111 GPU-s per item, 4.7× cheaper than the 10-seed M1 alone, and better) |
| freeze | final scores, 2-annotator scores, test predictions, release | not started |

**Where the cost is.** Per dev item, the current system runs:
- **M1, 10 seeds:** 10 forward passes of an 8B classifier. That is ≈ 0.2–0.3 GPU-s per seed with our HF
  evaluation loop (E13 timing: 3–5 min per evaluation of ~890 items), so **≈ 2–3 GPU-s per item**.
- **M3:** 0.36 calls of a 27B model at 0.155 GPU-s per call (vLLM, measured), so **≈ 0.06 GPU-s per item**.

So more than 95% of the inference cost is M1's ensemble, not the LLM. *(Measured in E16: an M1 seed costs 0.052 GPU-s per item, not 0.2–0.3, and an M3 call 0.103. M1 is still ~93% of the cost.)* E14 showed that the ensemble cannot
simply be shrunk: K = 1 / 3 / 5 / 10 seeds give S2 0.509 / 0.564 / 0.555 / 0.598. **The cost lever is K.**
That is what M4 should attack: one distilled model in place of ten.

### 7.2 E16: finish M2 (prompt ablation, and the accuracy-vs-cost frontier). Free, ~1 day.

1. **Prompt-part ablation** (report §7 promises it). Local Qwen3.8-27B, k = 0, on the 300 screen rows:
   - the full prompt;
   - without the confusion guide;
   - without the definitions (label names only);
   - without both.

   Each ablation is about 1 min of GPU. Each variant is then scored once on dev, with the fixed fusion
   code, and reported. Nothing is chosen on dev: the adopted prompt stays the full one unless the screen
   rule says otherwise.
2. **Measured cost per item, in one harness on the lab card (GPU 0):** an M1 seed pass (HF, bf16, batched)
   and an M3 call (vLLM). Reported as GPU-seconds per item, plus input/output tokens.
3. **The accuracy-vs-cost frontier without distillation.**
   - **Grid:** K ∈ {1, 2, 3, 5, 10} seeds × δ ∈ {0, 0.1, 0.2, 0.3, 0.5, 1}. For each K, the candidate
     sets are that ensemble's own top 4.
   - **New M3 scoring needed:** only for the (item, candidate set) pairs not already scored. It is local,
     free and de-duplicated.
   - **Output:** S2 / S1 against GPU-s per item; for K < 10, the mean over random seed subsets.
   - **What it gives the story:** the baseline M4 must beat. The question is how far M3 alone lets K shrink.
     For example, K = 3 + M3 at δ = 0.5 might match K = 10 + M3 at δ = 0.3 for a third of the cost.

### 7.3 E17: M4, distillation for cost

**Goal (the user's framing, 2026-10-14):** cut inference cost while keeping accuracy. An accuracy gain is
not required.

- **Teacher targets** for the 3,103 non-slice train rows: t = (1 − α)·gold + α·q, the report's design.
  - q is M3's distribution over the row's cross-fitted top 4 (zero elsewhere).
  - M3 is first run locally on the remaining ~1,700 train rows (≈ 5 min, free), so that every row has q.
- **Student:** the M1 recipe unchanged (Qwen3-8B-Base + LoRA, 12 epochs, epoch chosen on the slice), with
  the soft-target cross-entropy as the one change.
- **Control:** M1 single models on the same seeds (E13b/E14 seeds 0–2, already trained).
- **Protocol:**
  - a **3-seed screen on the lab server** (free; ~1.5–2 h per run on GPU 0, plus the GPU-1 slices when
    free);
  - if it passes, **7 more seeds**, on Jarvis in parallel (≈ 7 × 70 min × ₹190/h ≈ ₹1,550) or on the
    server (~12 h, free).
- **Evaluated as a system:** student (K = 1) + M3. The student's own top 4 are the candidates (new M3
  scoring, local and free), and its own confidence routes the items.

**Hypotheses (numbers to be fixed at registration):**
- **H1, cost:** student + M3 at K = 1 recovers at least half of the gap between M1(K = 1) + M3 and
  M1(K = 10) + M3, at a tenth of M1's cost. That puts it on or above the E16 frontier.
- **H2, routing:** soft targets make the student less overconfident (mean top probability below M1's 0.84),
  and its u(x) separates errors at least as well (AUROC ≥ 0.71). So the same accuracy may need fewer
  deferrals.
- **H3, accuracy (secondary):** the student single model gains little on S2 (TeleAI: +0.014), and more on S1.

**Decision rule (draft):** M4 is adopted as the *cost-efficient* operating point if student + M3 lies on the
E16 frontier, i.e. no (K, δ) of the undistilled system is both cheaper and better. The K = 10 + M3 system
remains the *accuracy* operating point. Both are reported.

### 7.4 E18: freeze and release (Oct 26–31)

- Final operating points: accuracy (M1 K = 10 + M3, δ = 0.3) and cost (from E16/E17), with 10-split
  nested-CV, 2-annotator and bootstrap numbers.
- Test predictions for both, from saved responses (`evaluation/make_submission.py`). No new API calls.
- Upload `runs/M3/` and the student adapters to the HF repo; update the report's §7 and the slides.

### 7.5 Decisions needed before registration

1. **α (gold vs M3 mixing weight):** fixed at 0.5 a priori (recommended: no extra runs), or fitted on the
   slice from runs at α ∈ {0.3, 0.7}. Fitting is what the report says, but it costs 2 more training runs
   and slice F1 from single runs is noisy.
2. **Teacher:** M3's q alone (the report; recommended, one change), or the fused M1 + M3 distribution.
3. **Precision for the cost table:** bf16, plus an FP8 measurement of the student (near-lossless on this
   card). **No 4-bit**, as decided for M3; this is a deviation from the report's "4-bit weights".
4. **Compute for the 7-seed confirmation:** Jarvis (≈ ₹1,550, a few hours) or the lab server (free, ~12 h).

### 7.6 Budget after this plan

E16 is free. The E17 screen is free (server). The E17 confirmation costs ≈ ₹1,550 if run on Jarvis, and
E18 is free. That leaves ≈ ₹2,450 of the ≈ ₹4,006 for reruns and the final freeze.
