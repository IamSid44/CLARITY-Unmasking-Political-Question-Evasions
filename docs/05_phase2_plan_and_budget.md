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

## Status (2026-10-06, 22:30)

| Module | State |
|---|---|
| **M1** | **done (E14).** 10-seed 12-epoch system S2 0.598 / S1 0.743; cross-fitted train probabilities for all 3,448 rows; C(x) and u(x) code; all runs on HF and W&B. Results against H1–H6 are in the log, §E14 results. |
| M3 | next, Oct 12–18. Its inputs are fixed by E14: rank-based C(x) (top-3 or top-4), u(x) = 1 − top probability, deferral δ ≈ 0.2–0.3. Model choice and prompt design are open. |
| M2, M4 | as planned |

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
| **balance** | | | **₹4,029.30 of ₹5,880** (≈ ₹190 per GPU-hour, ≈ 21 GPU-hours) |
| 2026-10-06 | E14 (M1): 7 seeds + 5 folds, ~30 GPU-hours over 20 h of wall time (GPU 0 + two MIG slices) | lab server | free (≈ ₹2,300 at Jarvis rates) |

### 4.2 Plan for the remaining credit

| Phase | Work | Where | GPU-h | ₹ (estimate) |
|---|---|---|---|---|
| M1 | 12-epoch seeds 3–9, 5 cross-fit folds | lab server | (~15–19 h wall) | 0 |
| M3 | vLLM adjudicator (~32B–70B, quantised) on slice, dev and test; prompt iterations | Jarvis, 1 VM | 4–6 | 800–1,150 |
| M2 | router, δ sweep, ablations | CPU | 0 | 0 |
| M4 | M3 on the 3,448 train rows; retrain M1 on soft targets (3-seed screen on the server, 7-seed extension on parallel Jarvis VMs if it passes) | mixed | 2–4 + 7 | 1,750–2,100 |
| reserve | reruns, final freeze | | | ~600 |
| **total** | | | | **≈ ₹3,150–3,850** |

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
