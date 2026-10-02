# CONTEXT — handover for a new Claude Code session

**Last updated 2026-10-02, about 14:00 IST** (`e13-qwen-lora` merged into `main`; VMs destroyed), at the end of the session that ran E13 at 10 seeds
and its follow-ups on JarvisLabs. It was written so that a new session on any machine can continue
without the chat history. Read this whole file first, then the files it points to.

---

## 1. Who and what

- **Team Nier_ANLP** (4 people), IIIT Hyderabad ANLP course project. Use they/them for everyone.
  - Siddarth Gottumukkula built the `clarity/` encoder track and owns its decisions.
  - E13 was run on 2026-10-02 by a teammate, from a Windows laptop with WSL Ubuntu.
- **Task:** SemEval-2026 Task 6 (CLARITY). Classify how a politician's answer responds to an
  interview sub-question:
  - Subtask 2 (S2): 9 evasion types;
  - Subtask 1 (S1): 3 clarity levels, a fixed function of the 9-way label.
- **Metric:** macro-F1. On S2 a prediction counts if it matches *any* annotator (3 on dev, 2 on
  test).
- **Data:** train 3,448 · dev 308 · test 237 (labels never released; Codabench closed).
  **All comparisons are on dev.**
- **Deadline: mid-evaluation submission, 2026-10-02 23:59 IST.** The package is `mideval/`. It is
  **complete and holds the final E13 numbers.**

## 2. Repository map and reading order

| path | what |
|---|---|
| `README.md` | top-level overview, headline table, reading order |
| `mideval/REPORT.md` | **the mid-eval report for the TA**. §7 is E13 (done); §8 is the plan (planned) |
| `mideval/RESULTS.md` | every number with its source. §9 is the Qwen classifier |
| `mideval/SLIDES_OUTLINE.md`, `AUDIT.md`, `PLAN_REMAINING.md`, `REPRODUCE.md` | slides (9), the proposal audit, the plan to the final, how to reproduce |
| `mideval/figures/` | `per_seed_progression.png` (DeBERTa), `e13_qwen_vs_deberta.png` (E13) |
| `clarity/reports/03_research_narrative.md` | the story of every experiment, E0 → E13 (§17 is E13) |
| `clarity/reports/02_experiment_log.md` | chronological record + scoreboard; every experiment pre-registered. **E13, E13x/b/c are registered and have results** |
| `clarity/reports/raw/E13_*.txt` | raw outputs cited by the E13 write-up (`E13_final_analysis.txt` is the main one) |
| `clarity/llm_classifier.py` | decoder LLM + LoRA as a 9-way classifier (`--track` → W&B; per-epoch adapters) |
| `clarity/e13_analysis.py`, `clarity/e13_figures.py` | the E13 analyses (sections A–E) and figure; CPU, about 3 min |
| `clarity/tracking.py` | W&B (online, offline fallback) + HF Hub upload (`python tracking.py push <seed dir>`) |
| `JARVISLABS_PORTING_GUIDE.md` | how the LLM runs work on JarvisLabs, **including the VM pitfalls found on 2026-10-02** |
| `jarvis_drive.sh` | launch machine (WSL): `keys`, `create`/`resume`, `connect`, `push`, `start`, `watch`, `sync`, `status`, `pause`; lanes via `JPROFILE=` |
| `jarvis_setup.sh` | on each VM: setup (+ MTU/IPv6 fix), smoke, pilot, train, HF upload per seed, summary, package, self-pause |
| `clarity/mideval_analysis.py`, `clarity/mideval_figures.py` | regenerate the DeBERTa parts of `mideval/` (needs all DeBERTa run folders) |

## 3. Where the results stand (dev)

| | S2 | S1 | source |
|---|---|---|---|
| DeBERTa baseline single model (10 seeds) | 0.315 ± 0.040 | 0.576 ± 0.030 | log §E11 |
| DeBERTa best single model: full question, 16 ep (10 seeds) | 0.384 ± 0.030 | 0.614 ± 0.027 | log §E11 |
| DeBERTa final system (10-seed ensemble + logit adjustment) | 0.405 | 0.648 | log §E11 |
| **Qwen3-8B LoRA single model, 3 epochs (E13, 10 seeds)** | **0.476 ± 0.043** | **0.709 ± 0.031** | `raw/E13_final_analysis.txt` §A |
| **Qwen3-8B LoRA 10-seed system (+ LA, nested CV)** | **0.543** (10 splits 0.549) | **0.746** | §A |
| same system, 2-annotator reference sets (test regime) | 0.524 | — | §A |
| Qwen, 12 epochs (E13b, seeds 0–2; adopted on the train slice) | 0.543 ± 0.015 | 0.742 ± 0.008 | §B |
| Qwen, all of train (E13c, 10 seeds) single / system | 0.492 ± 0.045 / 0.575 | 0.715 / 0.711 | §C |
| TeleAI (1st), DeepSeek-V3 pipeline / fine-tuned Qwen2.5-7B | 0.617 / 0.495 | 0.812 / 0.587 | TeleAI paper |
| ChulaNLP (2nd), RoBERTa top-5 → Kimi-K2 | 0.52 | 0.70 | ChulaNLP |
| human annotator vs the other two | 0.684 | — | mideval §G |

**Key findings:**
1. **The backbone swap is the largest gain in the project.** +0.092 S2 and +0.095 S1 per model,
   on **10/10 seeds**. The system difference is +0.136, 95% CI [+0.054, +0.224].
2. **The gain is broad.** It spans 7 of 9 classes, most in Claims ignorance (+0.40), Declining
   (+0.22) and Dodging (+0.10), and both agreed and contested items. The pre-registered
   "commitment boundary" prediction is **refuted**.
3. **Logit adjustment matters more for Qwen** (+0.062 per model vs +0.005). Raw Qwen
   under-predicts General.
4. **12 epochs.** Under a cosine schedule "best epoch = last" proves nothing, so we ran 12. The
   slice picks epochs 8–11, and the slice F1 rises by +0.084 (3/3 seeds). **Adopted by the
   rule.** No harmful overfitting, even at training loss ≈ 0.02.
5. **All of train:** +0.017 per model, not distinguishable at 10 seeds. The 3-seed screen
   overstated it, as with DeBERTa's E12a.
6. **A Qwen + DeBERTa mix adds nothing** (nested CV puts weight 0.79 on Qwen).
7. **Still unsolved:** Partial is never predicted; General's F1 is 0.32.
8. **Headroom for a cascade:**
   - an acceptable label is in Qwen's top 3 for 93% of items;
   - the in-set rate rises from 0.40 to 0.87 across confidence fifths (`raw/E13_Q8_topk_confidence.txt`).

## 4. State of git, machines, keys and money

- **Git.** All work since `8c80313` was done on branch `e13-qwen-lora` and **fast-forward merged
  into `main`** (pushed, with the user's OK, 2026-10-02). It holds the E13 results, the script
  changes, the analysis/figure scripts and the updated docs.
  `clarity/runs/` and `clarity/logs/` are gitignored, as is `clarity/.env`.
- **Run folders.** Every seed of `Q8_fullq_lora` (0–9), `Q8_fullq_lora_12ep` (0–2) and
  `Q8_alldata` (0–9) is on the HF repo **`siddarthg44/clarity-semeval26`**, as
  `<config>/seed<k>/`:
  - probabilities, metrics and `adapter_best/`;
  - per-epoch adapters `epochs/ep<k>_adapter/` for all of them except E13 seeds 0–2.

  The DeBERTa runs are there too. **The repo is PUBLIC** (`private=False`, checked on 2026-10-02).
  Tell the user if that is not intended.
  Fetch without weights:
  ```python
  from huggingface_hub import snapshot_download
  snapshot_download("siddarthg44/clarity-semeval26", local_dir="clarity/runs", token="<HF token>",
                    allow_patterns=["*/seed*/*.npy", "*/seed*/*.json"])
  ```
- **W&B:** entity `iamsid44-iiit-hyderabad`, project `clarity-semeval26`. Runs are named
  `<config>-s<seed>`, grouped by config. The bare username `iamsid44` is rejected as an entity.
- **Keys** (never in chat or git):
  - `clarity/.env` holds `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT`, `HF_TOKEN` (fine-grained,
    write) and `HF_REPO_ID`;
  - the JarvisLabs API key is in `~/.config/clarity_jarvis/keys.env` **on the WSL of the laptop
    that ran E13**;
  - per-VM host files are `host_<a|b|c|d>.env`;
  - on a new machine, rerun `bash jarvis_drive.sh keys` and `keygen` (add the public key in
    JarvisLabs before creating VMs).
- **JarvisLabs VMs: none.** The four used on 2026-10-02 were destroyed at about 13:50 IST, at
  the user's request, after every seed was confirmed on HF. The next run starts with
  `JPROFILE=<x> bash jarvis_drive.sh create`. Setup and the model download take about 3 min with
  the MTU fix. The old `host_<a-d>.env` files on the laptop point at destroyed machines;
  `create` overwrites them.
- **Money.** JarvisLabs credit is **₹4,026.64** (API balance after the VMs were destroyed,
  2026-10-02 ~13:50). The E13 round cost ₹1,518.79 of GPU time plus a few rupees of paused-disk
  storage; the project total is ₹1,853.36 of ₹5,880. A 1× RTX PRO 6000 VM costs
  ₹179.01/h + 18% GST ≈ ₹211/h.

## 5. What to do next (in order; ask before any spend over ~2 GPU-hours)

1. **Finish the mid-eval (deadline 2026-10-02 23:59).** The package is complete. What remains:
   - make the slides from `mideval/SLIDES_OUTLINE.md`;
   - submit.
2. **E13b at 10 seeds:** 12 epochs, seeds 3–9, as a new 10-seed single-pass system.
   - Pre-register it in the log first.
   - About 59 min per seed. 7 seeds over 4 VMs takes about 2 h wall-clock, ≈ ₹1,600.
   - Command per lane:
     `JPROFILE=b RUN_NAME=Q8_fullq_lora_12ep SEEDS="3 4" EXTRA_ARGS="--epochs 12 --no-grad-checkpoint" MAX_TOTAL_HOURS=4 bash jarvis_drive.sh all`.
   - Create each lane's VM first: `JPROFILE=b bash jarvis_drive.sh create`. If a VM is later
     paused and resumed, setup is skipped, so re-apply the MTU fix
     (`sudo ip link set dev enp1s0 mtu 1450; sudo sysctl -w net.ipv4.tcp_mtu_probing=1 net.ipv6.conf.all.disable_ipv6=1`).
3. **All of train × 12 epochs,** as its own single-change test against item 2.
4. **The cascade, the label-meaning test** (Phase B in `mideval/PLAN_REMAINING.md`):
   - uncertain items go, with Qwen's top-k, to a larger open LLM served with vLLM on a JarvisLabs
     GPU;
   - the prompt carries label definitions, a confusion guide and boundary examples from train;
   - prompts are designed on the train slice only (`val_probs.npy` exists for every slice-trained
     seed);
   - sweep the deferral fraction for an accuracy-vs-calls curve;
   - cache every LLM output.
5. **Distillation** (Phase C) is on hold until the user confirms it.
6. Phases D–F as in `mideval/PLAN_REMAINING.md`.

## 6. Rules this project keeps (the team cares about these)

- **One change per experiment**, against a named control; the baseline is frozen.
- **Hypotheses and predictions go in the experiment log before a run.** Results are reported
  against them, including failures, and overturned conclusions are corrected in place with a note.
- **Nothing is chosen on dev.** Epochs are chosen on the fixed train slice, decisions follow rules
  fixed in advance, and the one decision-rule parameter is scored by nested CV.
- **Seeds:** screen on 3 paired seeds (bar: +0.015 with ≥ 2/3 up); 10 seeds for any system claim;
  report the spread.
- **Every number must trace to a file**; never estimate a result. Done and planned are never
  blurred.
- **Ask before** launching anything that costs money or over ~2 h of GPU, before destroying VMs,
  and before committing or pushing (the user authorised pushing to `e13-qwen-lora` on
  2026-10-02). Don't push to `main` without asking.
- Explain decisions plainly, with the reasoning. The team values negative results written up with
  the reason they failed.

## 7. Environment notes

- **Laptop (Windows 11 + WSL Ubuntu).**
  - The repo lives on a OneDrive path with spaces: `/mnt/c/Users/vidva/OneDrive - International
    Institute of Information Technology/Desktop/Course_Work_4-1/ANLP/Project/Course_Project/CLARITY-Unmasking-Political-Question-Evasions`.
  - Run `jarvis_drive.sh` **from WSL**: Git Bash has no tmux, and PowerShell mangles quoting, so
    write scripts to a file and run `wsl -d Ubuntu -- bash <file>`.
  - WSL exports `NAME=<hostname>`, which is why the scripts use `RUN_NAME`.
  - Windows Python lacks pyarrow. Run analyses in WSL with uv:
    `uv run --no-project --python 3.12 --with numpy==2.5.2 --with pandas==3.0.5 --with pyarrow==25.0.1 --with scikit-learn --with matplotlib --with python-dotenv python clarity/e13_analysis.py`
    (from `clarity/`).
- **VM environment** (built by `jarvis_setup.sh`): Python 3.12, torch 2.11.0+cu128,
  transformers 5.17.0, peft 0.20.0, accelerate 1.15.0, wandb, python-dotenv, huggingface_hub.
- **Lab-server environment** (`/scratch/shlok/Temp/.venv`): the same versions, plus scikit-learn
  1.9.1, numpy 2.5.2, pandas 3.0.5, pyarrow 25.0.1. Lab facts: GPU 0 is a shared 96 GB RTX PRO
  6000; never kill the tmux server process; `run_queue.sh` hard-codes that venv.
- **The transformers 5.x dtype trap:** it loads checkpoints in their stored dtype. Always pass
  `dtype=` explicitly (`encoder.py` and `llm_classifier.py` do).

## 8. Security

- The lab server's git remote URL once embedded a GitHub token, and the user was told to revoke
  it. **Never print `git remote -v`.**
- Keys never go into chat or git: `clarity/.env`, `~/.config/clarity_jarvis/keys.env`, SSH private
  keys. `jarvis_drive.sh push` copies `clarity/.env` to each VM separately (mode 600), never inside
  the bundle.
- The HF repo is public (see §4).
