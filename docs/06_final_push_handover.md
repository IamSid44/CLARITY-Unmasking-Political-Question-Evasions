# 06 — Final push (2026-10-29 → 10-31): plan, budget and how to run it

Written 2026-10-29. It replaces the 2026-10-27 version of this document, whose plan was revised the same day
(§2, "What changed"). It has two kinds of reader: the team, and Claude Code sessions on the laptop (which
launches the JarvisLabs instances) and on the lab server (analyses). The hypotheses and decision rules are in
[`02_experiment_log.md`](02_experiment_log.md): §E19, §E20, the 2026-10-29 amendments, §E21 and §E22.
This document is the operational side.

**Deadlines.** All experiments by 2026-10-30. 2026-10-31 is for the report and slides.

---

## 1. Where we are

Dev = the 308-item QEvasion test split, 3 annotators. These are the E18 numbers with E19 Tier 0's
corrections (`raw/E19_tier0.txt`).

| System | S2, split 0 | S2, 10 splits | S1 | LLM calls / item | GPU-s / item |
|---|---|---|---|---|---|
| M1: Qwen3-8B + LoRA, 12 epochs, 10 seeds | 0.598 | 0.601 ± 0.013 | 0.743 | 0 | 0.522 |
| Accuracy point: M1 + M3 (27B), δ = 0.3 | 0.669 | 0.660 ± 0.013 | 0.798 | 0.36 | 0.576 |
| Cost point: distilled student + M3, δ = 0.5 | 0.608 | 0.610 ± 0.014 | 0.790 | 0.57 | 0.139 (0.27× M1) |
| TeleAI (1st place), dev | 0.617 | | 0.812 | 1.94 | – |
| One annotator against the other two (2-annotator regime) | 0.684 | | | | |

In the 2-annotator regime of the official test, the comparable system number is 0.589 (accuracy point) and
0.549 (M1), not 0.669.

**What Tier 0 established (2026-10-27).**
- **The accuracy point's gain over M1 is not significant once λ and τ are refitted in each bootstrap
  replicate:** +0.037 [−0.044, +0.120]. The earlier +0.067 [+0.009, +0.124] held them fixed.
- **What holds:**
  - Confidence routing beats random routing: 0.669 vs 0.604 ± 0.015 at δ = 0.3.
  - The gain comes from M3, not from re-thresholding (λ = 1 scores 0.581).
  - Every dev stratum gains in-set rate.
- **The LLM's value changes sign with M1's uncertainty.** Net fixes per M1-uncertainty fifth on train,
  most confident first: −41 / −76 / −12 / +5 / +31.
- **Test files exist** for M1 and the accuracy point (`submissions/E19_*`).

## 2. The plan

### What changed on 2026-10-29, and why

- **Compute.** Spot pricing exists only for containers, and on an H200 spot container an 8B training step
  took 6.83 s because the loop is CPU-bound there. On an on-demand RTX PRO 6000 VM it takes 1.44 s. All
  GPU work therefore runs on **RTX PRO 6000 VMs at ₹207/h**, about ₹210 per 8B seed.
- **A review found four new facts:**
  - **The official test answers carry transcript artifacts** that train and dev do not: speaker tags on
    96%, footnote markers on 10.5%, header lines on 6.8%.
  - **The 27B's shared prompt preamble is not cached,** so each call costs in proportion to the whole
    prompt.
  - **Cross-fitted train rows leak through sibling answers.**
  - **The candidate shortlist itself is not new:** ChulaNLP (2nd place) gave an LLM a DeBERTa model's top 5
    labels to choose from.
- **What no CLARITY system has done:** a learned rule for when to call the LLM, a cost-against-F1 frontier
  with significance tests, a debiased adjudicator, and a human score under the official metric. The plan
  is built around those.

### Components

| Id | What | The claim it tests | Where | Status |
|---|---|---|---|---|
| E19-E | Distillation controls: students taught by M1's own q, or by a uniform distribution over C(x); seeds 0–2 each | The student learns the LLM's knowledge, not just label smoothing | 2 VMs | **running** since 13:03 |
| E19-F | Qwen3-14B-Base backbone, seeds 0–1 | A third point on the encoder → 8B → 14B axis; a mixed 8B + 14B ensemble | 2 VMs, after E | to launch around 16:30 |
| E21 | Test transcript shift: dev sensitivity check, test normalization, label-free routing threshold, expected-F1 decoder (control) | The system is robust to how test answers are formatted | inference VM, then CPU | **running** (lane L) |
| E22-a | 27B cost with a working prefix cache | The cost point roughly halves | inference VM | **running** (lane L) |
| E22-b | Debiasing M3 by averaging 4 option orders (E19 arm B) | Part of the letter-A bias is an artifact | inference VM | **running** (lane L) |
| E19-A | LLM alone over all 9 labels: 27B with and without thinking (local), DeepSeek-V4-Flash (API) | Every LLM-alone arm scores below M1 | inference VM, API | partly server |
| E22-c | Escalation ladder: student → +5 M1 seeds on the 40% least confident → 27B on the 30% least confident | ≥ 0.625 S2 at ≈ 0.2 GPU-s, beating random escalation | CPU + one more 27B pass | after the inference lane |
| E20 + E22-d | Learned value-of-LLM router, fitted on leak-free (singleton) train rows | Beats confidence routing at matched calls | CPU | after the inference lane |
| E22-e | Question–answer specificity score (PMI under Qwen3-8B-Base) | Ordered Clear Reply > Ambivalent > Clear Non-Reply; a router feature | inference VM | **running** (lane L) |
| E23 | Freeze: tables, frontier figure, test files, docs, model card | — | CPU | Oct 30 |

**Dropped:**
- **E19-D (5 more M1 seeds, ≈ ₹1,050).** Tier 0's refit bootstrap and E16's seed subsets already bound
  the accuracy point.
- **Merging the 10 LoRA seeds into one model.** It is known to collapse when seeds start from different
  random weights (E12d; CoTo, 2025).
- **A LoRA-tuned 27B classifier.** Its hybrid linear-attention and vision architecture is too risky to
  train in a day.
- **Interview-grouped cross-fitting (≈ ₹1,050).** The singleton-row restriction removes the leak for free.
- **Synthetic rare-class data and LLMs as extra annotators.**

## 3. Budget and timetable

**Budget.**

| Item | Amount |
|---|---|
| Balance on 2026-10-29 at 13:00 | **₹3,748** |
| Already spent that day: reachability probes and the H200 attempt | (₹122, included above) |
| E19-E: 2 VMs × ≈ 3.4 h | ≈ ₹1,410 |
| Inference lane: ≈ 3 h, plus a ≈ 45 min second pass | ≈ ₹780 |
| E19-F: 2 VMs × ≈ 2.5 h | ≈ ₹1,035 |
| Model API (server), DeepSeek rungs only, hard cap $4 | ≤ ₹350 |
| **Total** | **≈ ₹3,575**, leaving ≈ ₹170 |

**Cost rules.**
- If E19-F's pilot projects more than 2.5 h per seed, run seed 0 only.
- If the balance falls below ₹600 before E19-F starts, stop and ask the user.

**Timetable (IST).**

| When | What |
|---|---|
| Oct 29, 13:03 → ≈ 16:30 | E19-E lanes |
| ≈ 14:30 → 17:30 | Inference lane |
| ≈ 16:30 → 19:30 | E19-F lanes, on fresh VMs once E's are destroyed |
| ≈ 20:00 | Second 27B pass: the ladder's candidate sets, normalized test text if E21 calls for it, and the E students' candidates |
| Night of Oct 29 | CPU analyses: E19-E, E19-F, E20, E21, E22 |
| Oct 30, by 12:00 | All results in |
| Oct 30, afternoon | E23 freeze |
| Oct 31 | Report and slides |

## 4. Who does what

- **Laptop (Claude here): every JarvisLabs instance.** That means launching, monitoring and destroying
  the E, F and inference lanes, and the inference lane's code. The lab server cannot reach Jarvis
  instances on port 22.
- **Server (Claude there): CPU analyses and the DeepSeek API rungs,** within the $4 cap.
  - If arms A or B already ran through the API, those results stand, labelled by source.
  - All further 27B scoring happens on the inference VM, so that new and saved records come from the
    same weights.
- **Results path.**
  - Every training seed is uploaded to HF `siddarthg44/clarity-semeval26` (`<run>/seed<k>/`) and logged
    to W&B.
  - The inference lane uploads its outputs to HF under `E21/` and `E22/`.
  - The server pulls from HF.

## 5. Laptop setup (Windows, Git Bash)

Git Bash has the ssh, scp, tar and curl the driver needs. Don't use WSL on the hotspot: its network ran at
about 13 KB/s. There is no tmux in Git Bash, so the driver's `watch` command is not available; use `status`
instead (§6).

```bash
# 1. a Python with the JarvisLabs SDK; the driver uses it through JL_PY instead of uvx
py -3.11 -m venv ~/jlvenv && ~/jlvenv/Scripts/python -m pip install jarvislabs
export JL_PY=~/jlvenv/Scripts/python
# 2. secrets, NEVER committed: code/.env holds WANDB_API_KEY, HF_TOKEN, HF_REPO_ID, JARVIS_API_KEY
# 3. an SSH key WITHOUT a passphrase, registered on the Jarvis account, at ~/.ssh/jarvis_ed25519
#    (the laptop's id_ed25519_jarvis, "ANLP_Midevals", has a passphrase, so batch ssh cannot use it;
#     the WSL key "clarity-to-jarvis-Eve", registered as "ANLP_Mideval_1", is the one in use)
# 4. the account key (paste JARVIS_API_KEY from code/.env; it is checked)
bash code/launch/jarvis_drive.sh keys
```

Check what is free:

```bash
. ~/.config/clarity_jarvis/keys.env; JL_API_KEY=$JL_API_KEY $JL_PY -c "
from jarvislabs import Client
for g in Client().account.gpu_availability():
    if g.num_free_devices: print(g.gpu_type, g.workload_type, g.num_free_devices, g.price_per_hour)"
```

## 6. Running the lanes

**Machine.** `GPU_TYPE=RTX-PRO6000 TEMPLATE=vm SPOT=0` for everything. Don't train on containers (see §2).

**Launching.** `create` makes the instance and connects. `push` sends the code bundle and `code/.env`.
`start` runs the pipeline on the instance: setup, smoke test, a 20-step pilot with a time projection, then
training of each seed with upload to HF.

```bash
export JL_PY=~/jlvenv/Scripts/python
# E19-E (running): one lane per control, seeds one after another
export JPROFILE=E1 GPU_TYPE=RTX-PRO6000 TEMPLATE=vm SPOT=0
bash code/launch/jarvis_drive.sh create
export RUN_NAME=Q8_distill_a05_m1 SEEDS="0 1 2" MAX_TOTAL_HOURS=4.5 \
       EXTRA_ARGS="--epochs 12 --no-grad-checkpoint --soft-targets ../runs/M4/targets_a0.5_m1.npy"
bash code/launch/jarvis_drive.sh push && bash code/launch/jarvis_drive.sh start
# lane E2: the same with JPROFILE=E2, RUN_NAME=Q8_distill_a05_unif and targets_a0.5_uniform.npy

# E19-F: one VM per seed (F0 with SEEDS=0, F1 with SEEDS=1)
export JPROFILE=F0 GPU_TYPE=RTX-PRO6000 TEMPLATE=vm SPOT=0
bash code/launch/jarvis_drive.sh create
export RUN_NAME=Q14_fullq_lora_12ep MODEL=Qwen/Qwen3-14B-Base SEEDS=0 MAX_TOTAL_HOURS=3 \
       EXTRA_ARGS="--epochs 12 --no-grad-checkpoint"
bash code/launch/jarvis_drive.sh push && bash code/launch/jarvis_drive.sh start
# if the pilot runs out of memory, start again with --grad-checkpoint instead of --no-grad-checkpoint
```

- **`PARALLEL=N`** trains up to N seeds at once on one GPU (default 1). An 8B run without checkpointing
  peaks at 53 GB, so N = 1 on a 96 GB card.
- **The inference lane** (E21, E22-a/b/e, the E19-A local rungs) uses the same `create` and `push`, then
  runs `code/launch/infer_lane.sh` on the VM:

```bash
export JPROFILE=L GPU_TYPE=RTX-PRO6000 TEMPLATE=vm SPOT=0
bash code/launch/jarvis_drive.sh create && bash code/launch/jarvis_drive.sh push
. ~/.config/clarity_jarvis/host_L.env
ssh -i ~/.ssh/jarvis_ed25519 -p $JPORT $JUSER@$JHOST "bash /home/clarity_llm/code/code/launch/infer_lane.sh start"
ssh -i ~/.ssh/jarvis_ed25519 -p $JPORT $JUSER@$JHOST "bash /home/clarity_llm/code/code/launch/infer_lane.sh status"
```

**The lane's steps.** Each step is logged to `logs/infer_<step>.log` and leaves a done-flag, so a rerun
resumes where it stopped.

| Step | What it does | Experiment |
|---|---|---|
| `fetch` | The published inputs from HF: seed probabilities, adapters, cross-fit folds, M3 records, embeddings | — |
| `m1inputs` | Rebuilds M1's cascade inputs and checks them against the candidate sets the saved M3 records used | — |
| `jobs` | Writes the job files for the all-9-labels arm and the thinking arm | E19-A, E20 R2 |
| `normalize` | `qevasion.normalize check`: artifact rates per split, before and after normalization | E21-b |
| `e21` | `cascade.infer_variants`: 10 M1 seeds and 10 students on clean, perturbed and normalized text | E21-a |
| `specificity` | `cascade.specificity`: the question–answer specificity score | E22-e |
| `vllm_dev`, `faith` | 27B re-score of dev on this VM, then the faithfulness gate | E19 amendment |
| `vllm_cached` | The same with the cache-aligned prompt | E22-a |
| `vllm_orders` | Dev, test and the 300 screen rows in 4 more option orders | E19-B, E22-b |
| `vllm_all9` | The LLM alone over all 9 labels, dev and test | E19-A |
| `vllm_think` | The thinking rung; skipped if started after 18:00 IST | E20 R2 |
| `m1cost` | One M1 seed's cost on the same GPU | E22-a |
| `upload` | Every output to HF under `E21/`, `E22/`, `M1/` and `M3/qwen38_27b_k0_*` | — |

While the 8B steps use the GPU, the 27B weights download and the vLLM 0.31.0 environment installs in the
background.

**Monitoring.** `JPROFILE=E1 bash code/launch/jarvis_drive.sh status` shows the pipeline stages, the
latest training lines and the GPU.

**Reading the pilot.** It prints a projected time for all seeds. If that is over `MAX_TOTAL_HOURS`, the
pipeline stops and pauses itself. Raise the limit only if the projection is sane.

**When a lane finishes.** The pipeline pauses the VM 30 minutes after `ALL DONE`, and a paused VM still
bills storage. So:
1. Check that each seed is on HF under `<RUN_NAME>/seed<k>`.
2. **Destroy** the VM:

```bash
P=E1; . ~/.config/clarity_jarvis/keys.env; . ~/.config/clarity_jarvis/host_$P.env
JL_API_KEY=$JL_API_KEY $JL_PY -c \
  "from jarvislabs import Client; c=Client(); print(c.instances.destroy($JL_MACHINE_ID)); print(c.account.balance())"
```

## 7. Rules that still apply

- **Claude never commits or pushes.** The user reviews and pushes.
- **Hypotheses are fixed in `02_experiment_log.md` before each run.** Nothing is chosen on dev; the
  decision rules say what is decided on train or by rule.
- **Every run is logged** to `logs/`, HF and W&B.
- **Secrets stay out of git:** `code/.env` and `~/.config/clarity_jarvis/`.
- **No idle instances.**
  - Destroy unreachable or finished instances at once.
  - Check `account.balance()` after each lane.
  - Follow the cost rules in §3.
- **The lab server's GPUs** belong to lab-mates while they are busy. Check `nvidia-smi` before any local
  GPU use.

## 8. For a Claude session

**Read first:** this document, then `02_experiment_log.md` from §E19 to the end.

**On the laptop:**
- Run the lanes in §6 and the inference lane.
- Write the E21/E22 inference code, after its registration and within it.
- Report which outputs are on HF.

**On the server:**
- Run the CPU analyses of E19-E/F, E20, E21 and E22, and E23.
- Run the DeepSeek API rungs within $4.
- Pull everything from HF.

**Either side:**
- Do not start instances outside this plan.
- Do not change a registered hypothesis after its run starts.
- Do not commit.
