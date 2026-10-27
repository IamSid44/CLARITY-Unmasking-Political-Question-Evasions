# 06 — Final push (2026-10-27 → 10-30): plan, context and how to run it

Written 2026-10-27 for two readers: the team, and a Claude Code session on the laptop that will launch
the JarvisLabs instances. The pre-registered hypotheses are in `02_experiment_log.md` §E19 and §E20; this
document is the operational side. **Deadline: all experiments by 2026-10-30; 2026-10-31 is for the report
and slides.**

---

## 1. Where we are

| System (dev, 308 items) | S2, split 0 | S2, 10 splits | S1 | GPU-s / item |
|---|---|---|---|---|
| M1: Qwen3-8B + LoRA, 12 epochs, 10 seeds (E14) | 0.598 | 0.601 ± 0.013 | 0.743 | 0.522 |
| Accuracy point: M1 + M3 (27B), δ = 0.3 (E15) | 0.669 | 0.660 ± 0.013 | 0.798 | 0.576 *(corrected)* |
| Cost point: distilled student + M3, δ = 0.5 (E17) | 0.608 | 0.610 ± 0.014 | 0.790 | 0.139 *(corrected; 0.27× M1, not 0.21×)* |
| TeleAI (1st place), dev | 0.617 | | 0.812 | |

**What the 2026-10-27 review and Tier 0 (`raw/E19_tier0.txt`) changed:**
- **The 308 "dev" items are QEvasion's test split.** That set was the task's *validation* set, and some
  design choices were made on it. The official ranking used a separate 237-item set with 2 annotators;
  its labels are unpublished. TeleAI scored 0.68 / 0.89 there.
- **The accuracy point's gain over M1 is not significant once λ and τ are refitted in each bootstrap
  replicate:** +0.037 [−0.044, +0.120], P(≤ 0) = 0.20. The earlier +0.067 [+0.009, +0.124] held them fixed.
- **On the 3,448 train rows** (cross-fitted, single gold label) the cascade adds +0.014 [+0.002, +0.026].
- **What holds up:**
  - Confidence routing beats random routing: 0.669 vs 0.604 ± 0.015.
  - The gain comes from M3, not from re-thresholding (λ = 1 scores 0.581).
  - Every dev stratum gains, including interviews not seen in train.
- **The LLM's value changes sign with M1's uncertainty.** Net fixes minus breaks per M1-uncertainty
  quintile, most confident first:
  - train: −41 / −76 / −12 / +5 / +31;
  - dev: −4 / 0 / −7 / +8 / +8.

  This is the basis of the spotlight (E20).
- **Test predictions are written** for M1 (10 seeds) and the accuracy point (`submissions/E19_*`). The
  cost point still needs about 280 M3 calls.

## 2. The locked plan

| Arm | What | Where | Status |
|---|---|---|---|
| Tier 0 | Corrected headlines and cost, train-row evaluation, routing and fusion ablations, strata, position bias, test files, human ceiling | server, CPU | **done** (`raw/E19_tier0.txt`) |
| A | LLM without M1, over all 9 labels: 27B with and without thinking, DeepSeek-V4-Flash thinking, TeleAI's pipeline on the 27B | server → Jarvis Model API | server, Oct 27–28 |
| B | M3 candidate-order control (reverse order + 2 cyclic shifts, on dev and the 300 screen rows); candidate-set sizes 2/3/6/9 | Model API | server |
| **D** | **5 new M1 seeds (10–14)**, to measure seed variance of the accuracy point | **Jarvis GPU (laptop)** | **to launch** |
| **E** | **Distillation controls**: self-distillation (M1's q) and uniform-over-C(x), 2 seeds each | **Jarvis GPU (laptop)** | **to launch** |
| F | Qwen3-14B backbone | — | **dropped** (budget rule) |
| E20 | Spotlight: value map, thinking rung, frontier-API rung, learned value router, one compute-frontier figure | Model API + CPU | server, Oct 28–29 |
| E21 | Freeze: final tables, figure, test files, docs, HF model card, W&B | server | Oct 30 |

**Budget.**

| Item | Amount |
|---|---|
| Starting balance (2026-10-14) | ₹4,001.81 |
| Three test instances on 2026-10-27, unusable because this server cannot SSH to Jarvis | −₹114 |
| **Balance at 20:00** | **₹3,887.48** |
| Model API, hard cap $12 (≈ ₹1,000) | committed |
| D + E on spot containers (≈ ₹1,350), or on-demand VMs (≈ ₹2,600) | committed |

## 3. Who does what

- **Laptop:** only JarvisLabs instance launches, monitoring and teardown, for lanes D1, D2, E1 and E2
  below. **Why:** the lab server cannot reach Jarvis instances on port 22 (VMs or containers; tested on
  2026-10-27).
- **Server:** Claude here does everything else: the Model-API runs (A, B, E20), analyses, docs, and pulling
  D/E results from HF. The laptop needs no Python environment.
- **How results get back:**
  - Each instance uploads every finished seed to HF `siddarthg44/clarity-semeval26` (`<run>/seed<k>/`) and
    logs to W&B.
  - The server pulls them from HF.
  - The laptop's `watch` also copies them into the laptop's `runs/`, which is a backup and does not need
    to be pushed.

## 4. Laptop: one-time setup

Requirements: bash, ssh, scp, tar, curl, tmux. On macOS: `brew install tmux`. On Windows, use WSL.

```bash
git pull                                   # the repo, including runs/M4 targets and data/cache
# 1. secrets — NEVER commit these. Copy code/.env from the server (it holds HF / W&B / Jarvis keys):
scp <you>@<server>:/scratch/shlok/Temp/CLARITY-Unmasking-Political-Question-Evasions/code/.env code/.env
# 2. the SSH key Jarvis instances will accept. Either reuse the laptop key already registered on the
#    account ("ANLP_Midevals" / "clarity-to-jarvis-Eve"):
cp ~/.ssh/<that key> ~/.ssh/jarvis_ed25519 && cp ~/.ssh/<that key>.pub ~/.ssh/jarvis_ed25519.pub
#    or make a new one and add the printed public key in JarvisLabs → SSH keys BEFORE creating instances:
bash code/launch/jarvis_drive.sh keygen
# 3. the account key (paste JARVIS_API_KEY from code/.env when asked; it checks the key):
bash code/launch/jarvis_drive.sh keys
```

**Reachability test first (~₹5).** Do this before any real lane. Use the cheapest spot container, probe
it, then destroy it:

```bash
JPROFILE=probe GPU_TYPE=A30 SPOT=1 TEMPLATE=pytorch bash code/launch/jarvis_drive.sh create
#  → prints "connected to ..." and nvidia-smi within ~3 min if SSH works. Then destroy it:
. ~/.config/clarity_jarvis/keys.env; . ~/.config/clarity_jarvis/host_probe.env
JL_API_KEY=$JL_API_KEY ~/.local/bin/uvx --quiet --from jarvislabs python -c \
  "from jarvislabs import Client; print(Client().instances.destroy($JL_MACHINE_ID))"
```

- **If the container fails to connect**, try the same with `TEMPLATE=vm GPU_TYPE=RTX-PRO6000 SPOT=0`.
  VMs cannot be spot.
- **If that also fails**, the problem is the key. Check JarvisLabs → SSH keys.
- **Never leave an unreachable instance running.** Destroy it at once; it bills by the hour.

## 5. Laptop: the four lanes

All four lanes use the E14/E17 recipe: Qwen3-8B-Base + LoRA, 12 epochs, no gradient checkpointing, epoch
chosen on the train slice. The pipeline adds `--track` (W&B) and uploads to HF itself.

**Instance choice, cheapest first:**
1. Spot container, `SPOT=1 TEMPLATE=pytorch`:
   - `GPU_TYPE=RTX-PRO6000`, ₹109/h, about 65–80 min per run;
   - or `A100-80GB`, ₹84/h, probably a bit slower.
2. Only if no spot container is free: an on-demand VM, `SPOT=0 TEMPLATE=vm GPU_TYPE=RTX-PRO6000`, ₹207/h.

Check what is free:

```bash
. ~/.config/clarity_jarvis/keys.env; JL_API_KEY=$JL_API_KEY ~/.local/bin/uvx --quiet --from jarvislabs python -c "
from jarvislabs import Client
for g in Client().account.gpu_availability():
    if g.spot_num_free_devices or g.num_free_devices: print(g.gpu_type, g.workload_type, 'free', g.num_free_devices, 'spot', g.spot_num_free_devices, g.price_per_hour, g.spot_price)"
```

Launch each lane in its own terminal. `create` makes the instance and connects; `all` pushes the code
bundle, starts the pipeline (setup → smoke test → 20-step pilot with a time projection → training), and
watches it from a local tmux session.

| Lane | RUN_NAME | SEEDS | Extra arguments | ≈ time |
|---|---|---|---|---|
| D1 | `Q8_fullq_lora_12ep` | `10 11 12` | — | 3.5–4 h |
| D2 | `Q8_fullq_lora_12ep` | `13 14` | — | 2.5–3 h |
| E1 | `Q8_distill_a05_m1` | `0 1` | `--soft-targets ../runs/M4/targets_a0.5_m1.npy` | 2.5–3 h |
| E2 | `Q8_distill_a05_unif` | `0 1` | `--soft-targets ../runs/M4/targets_a0.5_uniform.npy` | 2.5–3 h |

```bash
# example: lane D1 (repeat with the table's values for D2, E1, E2; one JPROFILE per lane)
export JPROFILE=D1 GPU_TYPE=RTX-PRO6000 SPOT=1 TEMPLATE=pytorch
bash code/launch/jarvis_drive.sh create
RUN_NAME=Q8_fullq_lora_12ep SEEDS="10 11 12" MAX_TOTAL_HOURS=6 \
  EXTRA_ARGS="--epochs 12 --no-grad-checkpoint" bash code/launch/jarvis_drive.sh all

# lane E1
export JPROFILE=E1 GPU_TYPE=RTX-PRO6000 SPOT=1 TEMPLATE=pytorch
bash code/launch/jarvis_drive.sh create
RUN_NAME=Q8_distill_a05_m1 SEEDS="0 1" MAX_TOTAL_HOURS=5 \
  EXTRA_ARGS="--epochs 12 --no-grad-checkpoint --soft-targets ../runs/M4/targets_a0.5_m1.npy" \
  bash code/launch/jarvis_drive.sh all
```

**If fewer GPUs are free**, run fewer lanes for longer. D1 + D2 become one lane with
`SEEDS="10 11 12 13 14" MAX_TOTAL_HOURS=8`, and the same goes for E1 + E2. The total cost is the same;
only the finish time changes. **Priority if only one lane is possible:** E1 + E2 first (they decide a
claim in the paper), then D.

**Monitoring:**
- `JPROFILE=D1 bash code/launch/jarvis_drive.sh status` shows the pipeline state on the instance.
- `tail -f logs/jarvis/D1/drive.log` shows the local watcher, which syncs every 5 min.
- `tmux attach -t jarvis-watch-D1` attaches to the watcher's session.

**Reading the pilot:** the pilot stage prints a projected time. If it is over `MAX_TOTAL_HOURS`, the
pipeline stops and pauses itself. Raise the limit only if the projection is sane.

**When a lane finishes:** the watcher pauses the instance. Then:
1. Check that each seed is on HF under `huggingface.co/siddarthg44/clarity-semeval26/tree/main/<RUN_NAME>/seed<k>`.
2. **Destroy** the instance. A paused instance still bills storage, and spot instances may not pause cleanly.

```bash
JPROFILE=D1; . ~/.config/clarity_jarvis/keys.env; . ~/.config/clarity_jarvis/host_$JPROFILE.env
JL_API_KEY=$JL_API_KEY ~/.local/bin/uvx --quiet --from jarvislabs python -c \
  "from jarvislabs import Client; c=Client(); print(c.instances.destroy($JL_MACHINE_ID)); print(c.account.balance())"
```

**If a spot instance is preempted** (the watcher reports it unreachable):
1. Destroy it.
2. Recreate the lane with only the seeds that are not on HF yet.

A seed restarts from scratch on a new instance.

**Afterwards:** tell Claude on the server which seeds landed. It pulls them from HF and runs the analyses.

## 6. Server (Claude): what runs here and when

| When | Work |
|---|---|
| Oct 27 night | Model-API code: thinking mode, candidate order, 9-way prompts, TeleAI pipeline. Pilot on the 300 screen rows, then A, B and the E20 thinking rung on dev, test and ~1,000 train rows stratified by M1 uncertainty. Cap $12 for the whole API spend. |
| Oct 28 | DeepSeek-V4-Flash rung (E20 S3), TeleAI pipeline (S5), the student test jobs (cost-point test file), value map S1. |
| Oct 29 | When D lands: M3 calls for new seed subsets (API), the 15-seed analysis. When E lands: the distillation-control analysis. E20 S4: the learned value router and the compute-frontier figure. |
| Oct 30 | E21 freeze. Final tables, figure, test files, docs/README update, HF model card, W&B summaries. Handover for the report and slides. |

## 7. Rules that still apply

- **Claude never commits or pushes.** The user reviews and pushes.
- **Hypotheses are fixed** in `02_experiment_log.md` §E19 and §E20. Nothing is chosen on dev; the decision
  rules there say what is decided on train.
- **Every run is logged** to `logs/`, HF and W&B.
- **Secrets stay out of git.** `code/.env` and `~/.config/clarity_jarvis/` are copied by hand only.
- **Before creating more than one instance, probe SSH once** (a 5-second `nc -zv <ip> 22`). Destroy
  anything unreachable immediately.
- **The lab server's GPUs** belong to lab-mates while they are busy. Check `nvidia-smi` before any
  local GPU use.

## 8. For a Claude session on the laptop

**Read first:** this document, then `02_experiment_log.md` §E19 and §E20.

**Your scope:**
- Set up the laptop (§4) with the user.
- Run the reachability test.
- Launch, monitor and destroy lanes D1, D2, E1 and E2 (§5).
- Report back which seeds are on HF.

**Not your scope:**
- Do not edit experiment code or docs beyond what a launch failure needs.
- Do not start instances other than these lanes.
- Do not commit.

**Cost discipline:**
- Never leave an instance running that is not training.
- Check `account.balance()` after each lane.
- Stop and ask the user if the balance drops below ₹1,500 before D and E are done.
