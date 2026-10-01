# CONTEXT — handover for a new Claude Code session

Written 2026-10-02 (about 03:00 IST) on the lab server, at the end of a long session, for the
session that continues on another machine. Read this whole file first, then the files it points
to.

---

## 1. Who and what

- **User:** Siddarth Gottumukkula (git user; team Nier_ANLP, 4 people). The user built the
  `clarity/` encoder track and owns its decisions. Use they/them for the user.
- **Task:** SemEval-2026 Task 6 (CLARITY). Classify how a politician's answer responds to an
  interview sub-question:
  - Subtask 2 (S2): 9 evasion types;
  - Subtask 1 (S1): 3 clarity levels, a fixed function of the 9-way label.
- **Metric:** macro-F1. On S2 a prediction counts if it matches *any* annotator (3 on dev, 2 on
  test).
- **Data:** train 3,448 (one label each) · dev 308 · test 237 (labels never released; Codabench
  closed). **All comparisons are on dev.**
- **Hard deadline: mid-evaluation submission, 2026-10-02 23:59 IST.** The submission package is
  `mideval/`, and it is ready (see §4).

## 2. Repository map and reading order

| path | what |
|---|---|
| `README.md` | top-level overview and reading order |
| `clarity/README.md` | the encoder system: task, results, how to run |
| `clarity/reports/03_research_narrative.md` | **the story of every experiment**: why, setup, result, why it worked or failed. Read this for background |
| `clarity/reports/02_experiment_log.md` | the chronological record with the scoreboard; every experiment pre-registered. **E13 is registered and not yet run** |
| `clarity/reports/01_scorer_geometry.md` | what the metric rewards (macro-F1 = classes named / 9 when every prediction is acceptable) |
| `mideval/` | the mid-eval package: REPORT, RESULTS (every number with its source), PLAN_REMAINING, SLIDES_OUTLINE, AUDIT, REPRODUCE, `analysis/`, `splits/`, `figures/` |
| `JARVISLABS_PORTING_GUIDE.md` | how to run E13 (Qwen3-8B LoRA) on JarvisLabs, end to end |
| `jarvis_drive.sh` | runs on the launch machine: SSH key, push, start, watch, sync, pause |
| `jarvis_setup.sh` | runs on the JarvisLabs instance: environment, model download, smoke, pilot, training, packaging, self-pause |
| `clarity/llm_classifier.py` | the E13 training script (decoder LLM + LoRA as a 9-way classifier) |
| `clarity/mideval_analysis.py`, `clarity/mideval_figures.py` | regenerate everything in `mideval/analysis`, `splits`, `figures` (CPU, ~25 s) |
| `HIGREC_MIDEVAL_BRIEF.md` | an earlier, partly outdated team brief for the mid-eval (if present); `mideval/` follows its structure but not its B0 plan |
| `Materials/` | task overview paper, the dataset paper, TeleAI and ChulaNLP papers, the team proposal |

## 3. Where the results stand

DeBERTa-v3-large fine-tuned as a 9-way classifier. Single models are 10 paired seeds; systems are
10-seed ensembles with the decision rule scored by nested CV.

| | dev S2 | dev S1 |
|---|---|---|
| baseline single model (sub-question + answer, 8 epochs) | 0.315 ± 0.040 | 0.576 ± 0.030 |
| + 16 epochs | 0.362 ± 0.026 | 0.601 ± 0.016 |
| **+ full journalist question in the input (best single model)** | **0.384 ± 0.030** | **0.614 ± 0.027** |
| final system: 10-seed ensemble + logit adjustment | 0.405 | 0.648 |
| baseline system, same recipe | 0.428 | 0.601 |
| all of train, last epoch (E12a), 5-seed system | 0.489 | — |
| human annotator vs the other two (2-annotator scoring) | 0.684 (0.643–0.766) | — |
| TeleAI (1st place), dev | 0.617 | 0.812 |
| ChulaNLP DeBERTa-large (checkpoint chosen on dev) | 0.46 | 0.65 |

**Key findings:**
1. The metric rewards in-set rate and coverage of rare classes.
2. The difficulty is attribution: 69% of rows share their answer with another sub-question.
3. Undertraining can look like "the idea doesn't work" (E8 vs E8b vs E10).
4. A one-parameter decision rule (logit adjustment) helps prior-leaning models; nine per-class
   weights overfit.
5. 5-seed system numbers swing by 0.08, so system claims need 10 seeds.
6. The model's top 3 contains an acceptable label for 93.8% of items, and confidence predicts
   correctness: in-set rate is 0.43 in the least confident fifth and 0.82 in the most.
7. 53 of 137 errors are on items all three annotators agreed on, so the gap is the model, not
   label noise.

**Negative results, each with a reason in the narrative:**
- a second encoder re-ranking the top 3;
- Balanced Softmax + focal loss;
- three hierarchy designs (post-hoc routing, a factorised head, specialist encoders);
- boundary experts for confused label pairs;
- model soups;
- mixed-input ensembles.

**Proposal contributions:**
- C2 (coverage-cut agreement) was pre-registered and refuted.
- C4 (multi-reference decision rule) was tested; the one-scalar control won. Its prediction
  (gains on contested items) held only in part: the gain is coverage of rare classes.
- C1, C3 and C5 were re-scoped. The table is in `mideval/REPORT.md` §4.

## 4. The mid-eval package (`mideval/`): done

- **The report is written for the TA.** Done and planned work are kept apart.
- **Every number traces to a file** (`mideval/RESULTS.md`).
- **Qwen results are marked TBD** in `REPORT.md` §7 and `RESULTS.md` §9. If E13 finishes before
  the deadline, fill those in from `clarity/logs/jarvis/summary.txt` and the run's
  `metrics.json`, and add the E13 results to the experiment log. Otherwise submit as is: the
  plan is labelled as planned.
- **Regenerate** with `python clarity/mideval_analysis.py` and `python clarity/mideval_figures.py`.
  This needs the project's Python environment (§7) **and `clarity/runs/`**, which is not in git
  (§7).

## 5. The immediate task: launch E13 on JarvisLabs from this machine

**What E13 is.** Qwen3-8B-Base fine-tuned with LoRA as a 9-way classifier, seeds 0–2, on 1×
A100 80GB. Exactly one change from the best DeBERTa model: the backbone. The rows, the fixed
345-row train slice, the input text and budgets, the loss and the epoch-selection rule are all
the same. It tests whether the encoder's gap is missing knowledge.

The pre-registration is in `clarity/reports/02_experiment_log.md` §E13:
- predictions: S2 0.43–0.50 if knowledge is the gap;
- screening bar: ≥ +0.015 over DeBERTa with ≥ 2/3 seeds up;
- DeBERTa reference on seeds 0–2: S2 0.344 / 0.370 / 0.428, S1 0.591 / 0.627 / 0.640.

**Steps** (details and troubleshooting in `JARVISLABS_PORTING_GUIDE.md`):

```bash
bash jarvis_drive.sh keygen          # print this machine's key; the user adds it in JarvisLabs -> SSH keys
# the user creates the instance AFTER adding the key: PyTorch template, 1x A100 80GB, 60 GB, on-demand
bash jarvis_drive.sh keys            # the user types the JarvisLabs API key (hidden). Never ask for it in chat
bash jarvis_drive.sh connect "<ssh command from the instance page>" <machine id>
bash jarvis_drive.sh all             # push + start (instance tmux "clarity-llm") + watch (local tmux "jarvis-watch")
```

- **Requirements** on this machine: bash, ssh, scp, tar, tmux, curl. `uv` is auto-installed.
  This machine must stay on overnight for results to be copied back.
- **On the instance**, stage by stage: setup → smoke (Qwen3-0.6B, interrupted and resumed) →
  pilot (20 steps; aborts and pauses if 3 seeds would take over `MAX_TOTAL_HOURS`=5 h) →
  seeds 0–2 → `summary.txt` → tarballs → pause. Any failure also pauses.
- **Results arrive in** `clarity/runs/Q8_fullq_lora/seed{0,1,2}/` (same file layout as encoder
  runs) and `clarity/logs/jarvis/`: `pipeline.log`, `summary.txt`, per-seed logs, tarballs.
- **Monitor** with `bash jarvis_drive.sh status`, `tail -f clarity/logs/jarvis/drive.log`, and
  `tmux attach -t jarvis-watch`.
- **Cost:** A100 80GB ₹140.94/h + 18% GST. Estimate ₹400–650 total, from the user's ~₹5,880
  credit. **Destroy the instance after the results are copied**; paused storage still bills
  (~₹19/day for 60 GB).

**Tested so far:**
- `llm_classifier.py`: CPU with Qwen3-0.6B, including resume, run from the bundle alone; the
  padding self-check passed.
- `jarvis_setup.sh`: the shell logic (pilot gate, summary, package) with stand-in data.
- `jarvis_drive.sh`: against a stand-in ssh/scp.

**Not yet tested:** a real JarvisLabs instance, a real GPU run of the 8B model, the JarvisLabs
pause API with a real key (validated at setup by `instances.get(machine_id)`). Expect to debug
the first stages. Read `clarity/logs/jarvis/pipeline.log` and the stage logs first.

**After E13 finishes:**
1. Read `summary.txt`.
2. Write the results into the log's §E13: per seed, against each prediction, with per-class
   detail (`python clarity/analyze.py --run-dir clarity/runs/Q8_fullq_lora`).
3. Update the scoreboard row, `mideval/RESULTS.md` §9 and `REPORT.md` §7.
4. If the screening bar passes, the next step is seeds 3–9: `SEEDS="3 4 5 6 7 8 9" bash
   jarvis_drive.sh all` on a resumed instance (~₹1,000–1,500, an estimate). **Ask the user
   before spending.**

## 6. Rules this project keeps (the user cares about these)

- **One change per experiment**, against a named control; the baseline is frozen.
- **Hypotheses and predictions written in the experiment log before a run.** Results reported
  against them, including failures. Earlier conclusions are corrected in place, with a note,
  when overturned.
- **Nothing is chosen on dev.** Epochs are chosen on the fixed train slice. The one decision-rule
  parameter is scored by nested CV.
- **Seeds:** screen on 3 paired seeds (bar +0.015, 2/3 up); 10 seeds for any system claim;
  report the spread.
- **Every number must trace to a file**; never estimate a result. Done and planned are never
  blurred.
- **Ask before** launching anything that costs money or takes over ~2 h of GPU, and before
  committing or pushing. The user usually pushes themselves.
- Explain decisions plainly, with the reasoning. The user values negative results written up with
  the reason they failed.

## 7. Environment and data notes

- **Python environment** (lab server, `/scratch/shlok/Temp/.venv`): Python 3.12.3, torch
  2.11.0+cu128, transformers 5.17.0, peft 0.20.0, accelerate 1.15.0, scikit-learn 1.9.1, numpy
  2.5.2, pandas 3.0.5, pyarrow 25.0.1, matplotlib. To recreate, install those versions. The
  JarvisLabs instance builds its own environment (`jarvis_setup.sh setup`); the launch machine
  needs none.
- **`clarity/runs/` is not in git** (gitignored). The trained runs' probability files (~6 MB
  without checkpoints) are on the lab server and on the private Hugging Face repo
  `siddarthg44/clarity-semeval26`, as `<config>/seed<k>/`. To fetch without checkpoints, with an
  HF token that can read it:
  ```python
  from huggingface_hub import snapshot_download
  snapshot_download("siddarthg44/clarity-semeval26", local_dir="clarity/runs", token="<HF token>",
                    allow_patterns=["*/seed*/*.npy", "*/seed*/*.json"])
  ```
  The `mideval` analyses and any comparison against DeBERTa beyond the three embedded reference
  numbers need these.
- **The transformers 5.x dtype trap:** it loads checkpoints in their stored dtype. DeBERTa-v3
  stored in fp16 silently trains to the label prior: loss stuck at 1.887, predicts only
  `Explicit`. Always pass `dtype=` explicitly (`encoder.py` and `llm_classifier.py` do).
- **Lab server facts** (only if working there):
  - GPU 0 is a full 96 GB card shared with lab-mates.
  - GPU 1 is split into two 48 GB MIG slices; ours is the first, and both have been lent out.
  - Never kill a process whose command line starts with `tmux new-session`: it is the tmux
    server.
  - Never edit `clarity/run_queue.sh` while lanes run; write a copy and `mv` it into place.
  - `run_queue.sh` and `run_pipeline.sh` hard-code `PY=/scratch/shlok/Temp/.venv/bin/python`.

## 8. Security

- **The lab server's git remote URL embeds a GitHub personal access token.** It has been printed
  in a session; the user was told to revoke it. Don't print `git remote -v` output into chat or
  files.
- **Keys never go into chat or git.** That covers the JarvisLabs API key
  (`~/.config/clarity_jarvis/keys.env`), HF and W&B tokens (`clarity/.env`, gitignored) and SSH
  private keys.

## 9. After the mid-eval (planned; details in `mideval/PLAN_REMAINING.md`)

- **A. E13** (above), and if it passes, 10 seeds and a smaller point (Qwen3.5-4B / Gemma 4 E4B).
- **B. Cascade.** The cheap model keeps confident items. Uncertain ones go, with a conformal
  candidate set, to an open LLM (Qwen3.6-27B or a 35B-A3B mixture-of-experts model via vLLM)
  given label definitions and a confusion guide. Sweep the deferral rate to get the
  accuracy-vs-calls curve, with TeleAI's point (1.94 calls, ~7,300 tokens per item) on the same
  axes. Add a reasoning-budget axis.
- **C. One-time teacher:** soft labels and rationales, distilled into single-pass students.
- **D. Cheap encoder items:** E12a seeds 5–9; a direct 3-way control (never run); contrastive
  training across sub-questions that share an answer; per-annotator heads (train has
  `annotator_id`).
- **E. Measured cost:** quantisation and latency, as cost per million items.
- **F. Final evaluation** with 10 seeds, two-annotator scoring and the human ceiling.

The paper's frame is an **accuracy-vs-cost curve**, not a leaderboard position.
