# Running the Qwen3-8B LoRA classifier on JarvisLabs

This runs one experiment on JarvisLabs: **Qwen3-8B-Base, fine-tuned with LoRA as a 9-way
classifier, 3 seeds, on one A100 80GB.** It is registered as **E13** in
`clarity/reports/02_experiment_log.md`. Everything else (the mid-eval package and the CPU
analyses) needs no GPU.

A **launch machine**, any always-on Linux/macOS/WSL machine with a clone of this repo, drives the
instance over SSH from a tmux session, so the run doesn't depend on a laptop staying connected.
Results come back into that clone automatically, and the instance pauses itself when it is done
or if anything fails.

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
Subtask 2). By the screening rule from E12 (experiment log), the LLM goes on to more seeds only if it gains
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
| `jarvis_drive.sh` | launch machine | makes the SSH key, pushes the code, starts the pipeline, copies results back every 5 min, pauses the instance |
| `jarvis_bundle.tgz` | built by `jarvis_drive.sh push` | the 3.4 MB bundle: training script, scorer, data, train slice, `jarvis_setup.sh` |

No git clone and no GitHub token on the instance: the bundle carries everything. The model
(Qwen3-8B-Base, Apache-2.0, ungated) is downloaded on the instance, so no Hugging Face token is
needed either.

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

| setting | choose | why |
|---|---|---|
| template | **PyTorch** | CUDA drivers; the script builds its own Python 3.12 environment |
| GPU | **1 × A100 80GB** | Qwen3-8B in bf16 is 16.4 GB of weights; the run should peak around 25–35 GB |
| storage | **60 GB** | model 16 GB + environment ~10 GB + outputs ~3 GB, with room to spare |
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
and the results stay on its disk. Resume the instance, run `bash jarvis_drive.sh sync`, then
pause or destroy it.

### What happens next, unattended

On the instance, inside tmux session `clarity-llm`:

| stage | time (est.) | what it checks | on failure |
|---|---|---|---|
| setup | ~10–15 min | installs torch (matched to the driver), transformers 5.17.0, peft 0.20.0; GPU bf16 test; downloads both models | pauses the instance |
| smoke | ~2–3 min | Qwen3-0.6B on the GPU: padding check, 2 epochs, interrupted after epoch 0 and resumed | pauses |
| pilot | ~5–8 min | 20 real optimizer steps of Qwen3-8B: minutes per epoch, peak VRAM | pauses; also aborts if all 3 seeds are projected over **5 h** |
| train | ~2–3.5 h (est.) | seeds 0, 1, 2 in turn; each epoch saves probabilities and a resume checkpoint | a failed seed retries once from its last epoch, then pauses |
| summary, package | 1 min | `summary.txt`, `results_Q8_fullq_lora.tgz`, `adapters_Q8_fullq_lora.tgz` | — |

On the launch machine, tmux session `jarvis-watch` copies logs and finished outputs every 5 minutes.
When the pipeline reports ALL DONE (or FAILED), it copies everything, including the LoRA adapters,
and **pauses the instance**. As a backup, the instance pauses itself 30 minutes after finishing.

---

## In the morning

```bash
cat clarity/logs/jarvis/summary.txt           # per seed vs DeBERTa, paired differences, screening verdict
cat clarity/logs/jarvis/pipeline.log          # every stage, with timestamps
cat clarity/logs/jarvis/drive.log             # what the launch machine's watcher did, incl. the pause
ls  clarity/runs/Q8_fullq_lora/               # seed0..2: dev/val/test probabilities, metrics, adapters
```

Then **destroy the instance** on the website once the results are here. A paused instance still
bills storage: 60 GB ≈ ₹0.8/h, ≈ ₹19 a day.

## Watching or intervening

| command (launch machine, repo root) | shows / does |
|---|---|
| `bash jarvis_drive.sh status` | stage log, latest training lines, GPU use on the instance |
| `tail -f clarity/logs/jarvis/drive.log` | the watcher's log |
| `tmux attach -t jarvis-watch` | the watcher itself (`Ctrl-b d` to leave) |
| `bash jarvis_drive.sh ssh`, then `tmux attach -t clarity-llm` | the live training output on the instance |
| `bash jarvis_drive.sh sync` | copy results now |
| `bash jarvis_drive.sh pause` | pause the instance now |

**After a pause (yours or automatic), to continue:** resume the instance on the website, then run
`bash jarvis_drive.sh start && bash jarvis_drive.sh watch`. Finished stages and seeds are skipped,
and an interrupted seed continues from its last finished epoch.

## Settings

Pass these as environment variables to `jarvis_drive.sh all` (or `start`):

| variable | default | example |
|---|---|---|
| `SEEDS` | `0 1 2` | `SEEDS="3 4 5 6 7 8 9"` to extend after a positive screen |
| `EXTRA_ARGS` | none | `EXTRA_ARGS="--batch-size 8 --grad-accum 2"` if the pilot shows spare memory (same effective batch) |
| `MAX_TOTAL_HOURS` | 5 | the pilot's abort threshold for all seeds together |
| `NAME` | `Q8_fullq_lora` | use a new name for any changed configuration, so earlier results stay untouched |

## Cost

At ₹140.94/h for an A100 80GB (jarvislabs.ai/in, 2026-10-01; 18% GST extra, so ≈ ₹166/h). Times
are estimates until the pilot measures them.

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
| `projected time over the limit` | the A100 is slower than estimated | raise `MAX_TOTAL_HOURS` deliberately, or use 2 seeds / 2 epochs |
| `auto-pause check FAILED` | API key or machine id wrong | the launch machine's watcher still pauses; fix with `jarvis_drive.sh keys` / `connect` |
| watcher: instance unreachable | paused already, or network | `bash jarvis_drive.sh status`; it stops trying after an hour |
