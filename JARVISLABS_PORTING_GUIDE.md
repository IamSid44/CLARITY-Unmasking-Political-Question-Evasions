# Running the Qwen3-8B LoRA classifier on JarvisLabs

> **Status (2026-10-02): done and working.** E13 (seeds 0–9), E13b (12 epochs, seeds 0–2) and
> E13c (all of train, seeds 0–9) all ran with these scripts on JarvisLabs **VMs** with 1× RTX PRO
> 6000 (96 GB, ₹179/h + GST). Four VMs ran in parallel, one per lane. Results are in
> `clarity/reports/02_experiment_log.md` §E13–E13c and `mideval/RESULTS.md` §9.
> **Read "What we learned on real VMs" below before the next launch.**

## What we learned on real VMs (2026-10-02)

| issue | symptom | fix (now in the scripts) |
|---|---|---|
| VMs log in as `ubuntu`, not root | `mkdir /home/clarity_llm` fails | `push` falls back to `sudo mkdir` + `chown` |
| **Path-MTU black hole** on the VM network | large downloads stall at 0 MB/s with SSL/read timeouts; small requests work | `jarvis_setup.sh setup` probes with a 1472-byte do-not-fragment ping and sets the MTU to 1450 (+ `tcp_mtu_probing`). With it, HF downloads run at about 75 MB/s. **The setting resets on reboot (pause/resume)**: setup re-applies it, but on a resumed VM where setup is skipped, run `sudo ip link set dev enp1s0 mtu 1450` |
| IPv6 broken on the VM | connections hang | setup disables IPv6 if `curl -6` fails |
| Hugging Face Xet transfers stall | downloads/uploads hang | `HF_HUB_DISABLE_XET=1` everywhere |
| WSL exports `NAME=<hostname>` | the run was named after the laptop | the run name is `RUN_NAME`, not `NAME` |
| a dynamic IP is reused by another VM after pause | `Host key verification failed` | `connect` runs `ssh-keygen -R <ip>` first |
| resuming a VM gives it a **new machine id and IP** | pause calls hit the old id | `resume` reconnects and stores the new id in the profile |
| W&B entity `iamsid44` rejected | `entity not found during upsertBucket` | use the team entity `iamsid44-iiit-hyderabad` (in `clarity/.env`); online failure falls back to offline logging |

**Measured on the RTX PRO 6000, without gradient checkpointing (`--no-grad-checkpoint`):**
- 1.43 s per optimizer step, 4.6 min per epoch, peak 53 GiB;
- a 3-epoch seed takes 15–16 min, a 12-epoch seed about 59 min;
- with gradient checkpointing it is 2.11 s per step and 19 GiB.

**Cost of the 2026-10-02 round:** ₹1,518.79 of GPU time, read from the API balance. Credit left
after the VMs were destroyed: ₹4,026.64 of ₹5,880.

## Several VMs in parallel (lanes)

Each lane is a profile, `JPROFILE=<name>`. A profile has its own host and machine id
(`~/.config/clarity_jarvis/host_<name>.env`), local logs (`clarity/logs/jarvis/<name>/`) and
watcher (`tmux jarvis-watch-<name>`). The JarvisLabs API key is shared
(`~/.config/clarity_jarvis/keys.env`).

```bash
# from WSL (Ubuntu) on the laptop, repo root; the laptop must stay on for the watchers
bash jarvis_drive.sh keys                         # once
JPROFILE=b bash jarvis_drive.sh create            # new 1x RTX PRO 6000 VM, 100 GB (API); or: resume
JPROFILE=b RUN_NAME=Q8_fullq_lora_12ep SEEDS="3 4" EXTRA_ARGS="--epochs 12 --no-grad-checkpoint"   MAX_TOTAL_HOURS=4 bash jarvis_drive.sh all       # push (+ clarity/.env, mode 600), start, watch
JPROFILE=b bash jarvis_drive.sh status            # progress
```

On every VM, each finished seed is uploaded to `HF_REPO_ID/<RUN_NAME>/seed<k>/` in the
background. All uploads are flushed before the pause. W&B logs per step and per epoch, grouped by
`RUN_NAME`. The laptop receives logs and probabilities, but **not the adapters**: those are on HF.
After a run, **destroy** the VMs on the website (or keep them paused at ₹1.30/h each for the
disk).

**Current VMs: none.** The four lanes (`a`–`d`) were destroyed on 2026-10-02 after every seed
was confirmed on HF. `create` makes a fresh VM per lane.

---

## Original guide (written before the first run; corrected 2026-10-02 to match the scripts)

This runs one experiment on JarvisLabs: **Qwen3-8B-Base, fine-tuned with LoRA as a 9-way
classifier, 3 seeds by default (`SEEDS`), on one GPU.** It was planned for an A100 80GB; that
GPU was not offered for VMs, so every run used a VM with 1× RTX PRO 6000 (96 GB), which is
`jarvis_drive.sh`'s default `GPU_TYPE=RTX-PRO6000`. It is registered as **E13** in
`clarity/reports/02_experiment_log.md`. Everything else (the mid-eval package and the CPU
analyses) needs no GPU.

A **launch machine**, any always-on Linux/macOS/WSL machine with a clone of this repo, drives the
instance over SSH from a tmux session, so the run doesn't depend on a laptop staying connected.
Logs and probabilities come back into that clone automatically. The launch machine's watcher
pauses the instance as soon as it has copied the final results; as a backup, the instance pauses
itself 30 minutes (`PAUSE_DELAY=1800`) after finishing or failing.

**The launch machine needs:** `bash`, `ssh`, `scp`, `tar`, `tmux`, `curl` and outbound SSH. It
needs no Python environment and no GPU. `uv`, used for the JarvisLabs SDK, is installed
automatically if missing. On Windows, use WSL.

---

## What the experiment is

**The question.** Is the encoder's gap a knowledge gap? An 8B LLM has far more world and
language knowledge than DeBERTa-v3-large (0.4B). If that is what the encoder is missing, the LLM
classifier should score clearly higher under the same protocol. TeleAI's Qwen2.5-7B, fine-tuned
without chain-of-thought, scored 0.495 on dev Subtask 2.

**One change from our best DeBERTa model (E10_fullq_16ep): the backbone.**

| | DeBERTa (E10_fullq_16ep) | this run |
|---|---|---|
| training rows | train minus the fixed 10% slice (345 rows, seed 12345) | same rows, same slice (checked against `mideval/splits/`) |
| input | sub-question + full question (≤256 tokens) + answer, 1024 tokens | same text, same budgets, plus a fixed closing question |
| loss | plain cross-entropy, effective batch 16 | same |
| epoch choice | best epoch on the train slice; dev never used | same |
| model | DeBERTa-v3-large, full fine-tune, lr 1e-5, 16 epochs | **Qwen3-8B-Base, LoRA r=16 on all linear layers + new 9-way head, lr 1e-4, 3 epochs** |

The learning rate and epoch count change because the backbone does: LoRA on an 8B model
converges in a few epochs at about 1e-4.

**Read-out.** `summary.txt` puts each seed next to DeBERTa's same seed (0.344 / 0.370 / 0.428 on
Subtask 2; the reference covers seeds 0–2 only, so other seeds show `nan`). By the screening rule from E12 (experiment log), the LLM goes on to more seeds only if it gains
at least +0.015 with at least 2 of 3 seeds up. The hypotheses and predictions are pre-registered in the
experiment log, §E13.

**Outputs** use the same files and folders as every encoder run (`clarity/runs/Q8_fullq_lora/seed<k>/`),
so `analyze.py`, `decide.py` and `mideval_analysis.py` read them unchanged.

---

## The files

| file | runs on | does |
|---|---|---|
| `clarity/llm_classifier.py` | instance | the training script (tested end to end on CPU with Qwen3-0.6B, including resume) |
| `jarvis_setup.sh` | instance | environment, model download, smoke test, timing pilot, training, packaging, self-pause |
| `clarity/tracking.py` | instance | W&B logging (`--track`; offline fallback) and the per-seed Hugging Face upload (`tracking.py push <seed dir>`) |
| `jarvis_drive.sh` | launch machine | makes the SSH key, creates/resumes VMs through the API, pushes the code and `clarity/.env`, starts the pipeline, copies logs and probabilities back every 5 min (not the LoRA adapters), pauses the instance |
| `jarvis_bundle.tgz` | built by `jarvis_drive.sh push` | the 3.4 MB bundle: training script, `tracking.py`, scorer (`qevasion/`), data (train/dev/test), train slice, `jarvis_setup.sh`, this guide |

No git clone and no GitHub token on the instance: the bundle carries everything. The model
(Qwen3-8B-Base, Apache-2.0, ungated) is downloaded on the instance without a token. The W&B key
and the Hugging Face **write** token for the per-seed uploads are in `clarity/.env`, which `push`
copies to the instance separately (mode 600), never inside the bundle. Without it the run still
trains; W&B logs offline and the upload is skipped.

---

## Tonight: about 10 minutes of your time

### 1. Make an SSH key on the launch machine and add it to JarvisLabs

```bash
bash jarvis_drive.sh keygen      # creates ~/.ssh/jarvis_ed25519 (no passphrase, for unattended use) and prints the public key
```

JarvisLabs → **SSH keys** → paste the printed line. The private half never leaves the launch
machine. Add the key **before** creating the instance; instances pick up keys when they're
created.

### 2. Create the instance

The scripted way, used for every run on 2026-10-02 (after `keys`, below):
`JPROFILE=<x> bash jarvis_drive.sh create` makes a VM through the API (`template="vm"`,
`GPU_TYPE`, 100 GB) and runs `connect` itself, so step 3's `connect` is not needed. By hand on
the website:

| setting | choose | why |
|---|---|---|
| template | **VM** (`create` uses `template="vm"`) | CUDA drivers; the script builds its own Python 3.12 environment. VMs log in as `ubuntu`, which `push` handles |
| GPU | **1 × RTX PRO 6000 (96 GB)** (the A100 80GB was not offered for VMs) | Qwen3-8B in bf16 is 16.4 GB of weights; measured peak 19.3 GiB with gradient checkpointing, 53.4 GiB without (`--no-grad-checkpoint`) |
| storage | **100 GB** (what `create` requests) | models 17 GB + environment + outputs and per-epoch adapters, with room to spare |
| pricing | **on-demand** | a spot pre-emption at night would stop the run until morning |

Note the **machine id** and the **SSH command** shown on the instance page.

### 3. On the launch machine, from the repo root

```bash
bash jarvis_drive.sh keys                     # paste the JarvisLabs API key (hidden), from jarvislabs.ai/settings/api-keys
bash jarvis_drive.sh connect "<the SSH command>" <machine id>
bash jarvis_drive.sh all
```

- `keys` stores the API key in `~/.config/clarity_jarvis/keys.env` (mode 600) and checks it.
  Never paste it into a chat.
- `connect` tests SSH and prints the instance's GPU and disk.
- `all` pushes the bundle, starts the pipeline on the instance, and starts the local watcher.

You can then disconnect. The launch machine itself must stay on for the results to be copied
back automatically. If it sleeps, the instance still pauses itself 30 minutes after finishing,
and the results stay on its disk (each finished seed is also on the HF repo). Run
`bash jarvis_drive.sh resume` (a resumed VM gets a new machine id and IP; `resume` stores them),
then `bash jarvis_drive.sh sync`, then pause or destroy it.

### What happens next, unattended

On the instance, inside tmux session `clarity-llm`:

| stage | time (measured 2026-10-02, RTX PRO 6000) | what it checks | on failure |
|---|---|---|---|
| setup | ~2 min with the MTU fix | MTU/IPv6 fix; installs torch (matched to the driver), transformers 5.17.0, peft 0.20.0; GPU bf16 test; downloads both models | pauses the instance |
| smoke | < 1 min | Qwen3-0.6B-Base on the GPU, 64 rows: padding check, 2 epochs, interrupted after epoch 0 and resumed | pauses |
| pilot | ~1 min | peak VRAM on the longest micro-batch, then 20 real optimizer steps of Qwen3-8B: minutes per epoch, peak VRAM | pauses; also aborts if all seeds × epochs are projected over `MAX_TOTAL_HOURS` (default **5 h**) |
| train | 3-epoch seed: 21 min with gradient checkpointing, 15–16 min without; 12-epoch seed: ~59 min | the seeds in turn; each epoch saves probabilities, its LoRA adapter and a resume checkpoint; each finished seed is uploaded to the HF repo in the background | a failed seed retries once from its last epoch, then pauses |
| HF flush, summary, package | ~1 min | uploads any seed still missing on HF; `summary.txt`, `results_<NAME>.tgz`, `adapters_<NAME>.tgz` (on the instance) | — |

On the launch machine, tmux session `jarvis-watch` (`jarvis-watch-<profile>` with `JPROFILE`)
copies logs, probabilities and metrics every 5 minutes, **without the LoRA adapters** (they are on
the HF repo and in `adapters_<NAME>.tgz` on the instance). When the pipeline reports ALL DONE (or
FAILED), it copies the final results and `results_<NAME>.tgz`, and **pauses the instance**. As a
backup, the instance pauses itself 30 minutes after finishing.

---

## In the morning

```bash
cat clarity/logs/jarvis/summary.txt           # per seed vs DeBERTa, paired differences, screening verdict
cat clarity/logs/jarvis/pipeline.log          # every stage, with timestamps
cat clarity/logs/jarvis/drive.log             # what the launch machine's watcher did, incl. the pause
ls  clarity/runs/Q8_fullq_lora/               # seed0..2: dev/val/test probabilities, metrics (adapters: HF repo)
```

With `JPROFILE=<x>` the logs are in `clarity/logs/jarvis/<x>/`.

Then **destroy the instance** on the website once the results are here. A paused instance still
bills storage: 100 GB ≈ ₹1.30/h.

## Watching or intervening

| command (launch machine, repo root) | shows / does |
|---|---|
| `bash jarvis_drive.sh status` | stage log, latest training lines, GPU use on the instance |
| `tail -f clarity/logs/jarvis/drive.log` | the watcher's log |
| `tmux attach -t jarvis-watch` (`jarvis-watch-<x>` with `JPROFILE=<x>`) | the watcher itself (`Ctrl-b d` to leave) |
| `bash jarvis_drive.sh ssh`, then `tmux attach -t clarity-llm` | the live training output on the instance |
| `bash jarvis_drive.sh sync` | copy results now |
| `bash jarvis_drive.sh pause` | pause the instance now |

**After a pause (yours or automatic), to continue:** run `bash jarvis_drive.sh resume` (the VM
comes back with a new machine id and IP, which `resume` stores), re-apply the MTU fix by hand
(setup is skipped on a resumed VM; see the table at the top), then run
`bash jarvis_drive.sh start && bash jarvis_drive.sh watch`. Finished stages and seeds are skipped,
and an interrupted seed continues from its last finished epoch.

## Settings

Pass these as environment variables to `jarvis_drive.sh all` (or `start`):

| variable | default | example |
|---|---|---|
| `JPROFILE` | none | `JPROFILE=b`: one profile (host, machine id, logs, watcher) per VM, for parallel lanes |
| `GPU_TYPE` | `RTX-PRO6000` | the GPU `create` asks the API for |
| `SEEDS` | `0 1 2` | `SEEDS="3 4 5 6 7 8 9"` to extend after a positive screen |
| `EXTRA_ARGS` | none | `EXTRA_ARGS="--no-grad-checkpoint"` (used for every run after the first E13 screen: same computation, faster, 53.4 GiB), `--epochs 12`, or `--batch-size 8 --grad-accum 2` (same effective batch) |
| `MAX_TOTAL_HOURS` | 5 | the pilot's abort threshold for all seeds together |
| `RUN_NAME` (sets NAME) | `Q8_fullq_lora` | use a new name for any changed configuration, so earlier results stay untouched |

## Cost

**Actual (2026-10-02):** a 1× RTX PRO 6000 VM costs ₹179.01/h + 18% GST ≈ ₹211/h; the four-VM
round (E13x, E13b, E13c) cost ₹1,518.79 (API balance before and after). The estimate below was
made before the first run, for an A100 80GB at ₹140.94/h (jarvislabs.ai/in, 2026-10-01; 18% GST
extra, so ≈ ₹166/h).

| stage | hours | ₹ incl. GST |
|---|---|---|
| setup + smoke + pilot | ~0.4 | ~70 |
| 3 seeds | ~2–3.5 | ~330–580 |
| storage until destroyed | — | ~₹19/day |
| **total** | **~2.5–4** | **≈ ₹400–650 of ₹5,880** |

The pilot's 5-hour cap bounds the worst case at roughly ₹900.

## Troubleshooting

| symptom (in `pipeline.log` or `drive.log`) | cause | fix |
|---|---|---|
| `connect`: SSH failed | key not added, or instance created before it was | `bash jarvis_drive.sh keygen`, add the key, then create the instance |
| `setup`: install failed | network, or a torch build that doesn't match the driver | `logs/setup.log`; the script picks cu128 / cu126 / cu124 from the driver version |
| `smoke`: padding self-check failed | the head isn't reading the last real token | don't train; check the transformers version in `setup.log` |
| `pilot failed` | usually out of memory | `EXTRA_ARGS="--batch-size 2 --grad-accum 8"` |
| `projected time over the limit` | the GPU is slower than estimated, or gradient checkpointing is on | raise `MAX_TOTAL_HOURS` deliberately, or use 2 seeds / 2 epochs |
| `auto-pause check FAILED` | API key or machine id wrong | the launch machine's watcher still pauses; fix with `jarvis_drive.sh keys` / `connect` |
| watcher: instance unreachable | paused already, or network | `bash jarvis_drive.sh status`; it stops trying after an hour |
