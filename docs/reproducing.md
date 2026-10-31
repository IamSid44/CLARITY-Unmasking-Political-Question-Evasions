# Reproducing the results

All commands run from `code/`. Train and dev data are in `data/cache/` (re-downloaded if missing).

1. [Setup](#1-setup)
2. [Re-run every analysis (CPU only)](#2-re-run-every-analysis-cpu-only)
3. [Train the classifiers (GPU)](#3-train-the-classifiers-gpu)
4. [The cascade (GPU)](#4-the-cascade-gpu)
5. [After the shared task: E24–E26](#5-after-the-shared-task-e24e26)
6. [Where the GPU work ran, and what it cost](#6-where-the-gpu-work-ran-and-what-it-cost)

## 1. Setup

```bash
pip install -r requirements.txt      # Python 3.11+
cd code
```

**Keys (optional).** For W&B logging, Hugging Face uploads, the M3 API arms or the cloud driver, copy
`code/.env.example` to `code/.env` and fill it in. Without keys, uploads are skipped and runs still complete.

The 27B adjudicator (Qwen3.8-27B-FP8) and E25's LLM answers use vLLM 0.31 in a separate environment
(`$VPY` below), so the training environment is untouched.

## 2. Re-run every analysis (CPU only)

The per-seed probabilities, 27B records and labels behind the reported numbers are in `runs/`. A few early
DeBERTa ablations (E8, E9, E12b–d) and all model weights are only on Hugging Face:

```python
from huggingface_hub import snapshot_download     # optional: the remaining runs, without weights
snapshot_download("siddarthg44/clarity-semeval26", local_dir="../runs",
                  allow_patterns=["*/seed*/*.npy", "*/seed*/*.json", "*/fold*/*.npy", "*/fold*/*.json"])
```

Each script prints the file committed under `docs/raw/`:

```bash
python -m preregistered_analyses.mideval_analysis      # main table, error analysis, human ceiling
python -m preregistered_analyses.e11_replication       # DeBERTa replication on fresh seeds
python -m preregistered_analyses.e13_analysis          # Qwen vs DeBERTa, per seed and as systems
python -m preregistered_analyses.e14_m1_analysis       # M1
python -m preregistered_analyses.e17_m4_analysis       # M4
python -m preregistered_analyses.e18_final             # both operating points, frontier figure
python -m preregistered_analyses.e19_tier0             # robustness checks (~7 min)
python -m preregistered_analyses.e20_router
python -m preregistered_analyses.e21_e22_analysis
python -m preregistered_analyses.e23_systems           # final operating points (~15 min)
python -m preregistered_analyses.e24_prechecks         # ~5 min
python -m preregistered_analyses.e25_analysis
python -m preregistered_analyses.e26_analysis
python -m evaluation.verify_scorer_geometry            # docs/scorer_geometry.md
python -m figures.paper_figures && python -m figures.e25_figures
```

Submission files: `python -m evaluation.make_final_submission` writes `submissions/E23_*/`.

## 3. Train the classifiers (GPU)

**DeBERTa-v3-large** (24 GB is enough), e.g. the first run of E10:

```bash
python -m models.encoder $(cat experiments/E10/common.args) \
    --name E10_fullq_16ep --seed 0 --input full --max-len 1024 --a-budget 256
```

**Qwen3-8B-Base + LoRA** (≈ 53 GB without gradient checkpointing; we used 96 GB RTX PRO 6000s):

```bash
python -m models.llm_classifier --name Q8_fullq_lora --seed 0                    # E13, 3 epochs
python -m models.llm_classifier --name Q8_fullq_lora_12ep --seed 0 --epochs 12   # E14, M1 (seeds 0-9)
python -m models.llm_classifier --name Q8_12ep_crossfit --seed 0 --fold 2 \
    --select last --epochs 12                                                    # cross-fitted train rows
```

Whole experiments run through the queue on a shared server: `bash launch/start.sh E14` reads
`experiments/E14/`, runs one lane per GPU, resumes interrupted runs from their last epoch and uploads finished
ones to Hugging Face and W&B.

## 4. The cascade (GPU)

```bash
# M1: candidate sets and uncertainty from the 10-seed ensemble (CPU)
python -m cascade.candidates --run Q8_fullq_lora_12ep --crossfit Q8_12ep_crossfit --method aps --alpha 0.1

# M3: the 27B over M1's top 4, then fusion and the deferral curve
python -m cascade.exemplars embed
bash launch/m3_local.sh screen && bash launch/m3_local.sh full 0
python -m cascade.fuse dev --tag qwen38_27b_k0

# M2: accuracy vs cost over seed count x deferral rate
python -m cascade.frontier jobs && python -m cascade.frontier analyse

# M4: soft targets, then one student per seed
python -m cascade.distill_targets --alpha 0.5
python -m models.llm_classifier --name Q8_distill_a05 --seed 0 --epochs 12 --soft-targets ../runs/M4/targets_a0.5.npy

# E19-E23: the robustness checks and the final system use the same modules; the second 27B pass's jobs:
python -m cascade.pass2_jobs
```

## 5. After the shared task: E24–E26

The frozen instrument's parameters are in `experiments/e25/frozen_system.json` (`python -m audit.system freeze`).

```bash
# E25, one GPU (we used an H100; launch/infer_lane.sh PASS=3 runs these in order)
python -m audit.frames build                               # question swaps and answer prefixes
$VPY -m audit.generate --model mistral_7b=mistralai/Mistral-7B-Instruct-v0.3 --model qwen3_8b=Qwen/Qwen3-8B
python -m audit.frames llm                                 # the LLM answers as a frame
python -m audit.infer --frames 'dev_*,test_*' && python -m audit.infer --frames llm
python -m audit.system jobs '*'                            # deferred items -> runs/E25/jobs_27b.jsonl
for o in taxonomy shift1 shift2 shift3; do                  # the 27B in four option orders
  $VPY -m cascade.adjudicate --backend vllm --model Qwen/Qwen3.8-27B-FP8 --k 0 --tag e25_$o --splits "" \
      --jobs-file ../runs/E25/jobs_27b.jsonl --order $o; done
python -m audit.system label '*'
python -m audit.hidden dump && python -m audit.hidden probe

# E26: each precision against 16-bit on the same GPU type
python -m audit.quant --precision bf16 --site rtxpro && python -m audit.quant --precision fp8 --site rtxpro
python -m audit.quant_system jobs --site rtxpro            # then the 27B; then:
python -m audit.quant_system label --site rtxpro
E26_MAX_MEM_GIB=12 python -m audit.quant --precision bf16 --site a4000 --device-map auto --batch 8  # split over two 16 GB GPUs
python -m audit.quant --precision nf4 --site a4000 --batch 16                # one 16 GB GPU (launch/e26_lab_*.sh)
python -m audit.quant_system label --site a4000 --records-from rtxpro
```

The outputs in `runs/E25/` and `runs/E26/` are what the analyses in §2 read; the E25 hidden states
(`runs/E25/hidden/*.npy`) are on Hugging Face under `E25/hidden/`.

## 6. Where the GPU work ran, and what it cost

| Work | Hardware | Cost |
|---|---|---|
| E0–E12 (DeBERTa, ~120 runs), E14 (M1), E15–E17 (local 27B, students) | lab server, shared RTX PRO 6000 | free |
| E13 (Qwen pilots and 3-epoch seeds) | JarvisLabs, RTX PRO 6000 VMs | ₹1,851 |
| E15 API arms (DeepSeek, Gemma) | JarvisLabs Model API | ₹23 ($0.26) |
| E19–E23 (controls, inference lanes, second 27B pass) | JarvisLabs, RTX PRO 6000 VMs | ₹2,962 |
| E25 | JarvisLabs, H100 | ₹555 |
| E26 FP8 / NF4 | JarvisLabs RTX PRO 6000 / lab RTX A4000s | ₹90 / free |

**Cloud lanes.** `launch/jarvis_drive.sh` drives a rented instance from a laptop (Git Bash works; set `JL_PY` to
a Python with the `jarvislabs` SDK): `create`, `push` (code bundle and `code/.env`), then `start` (training
pipeline) or `lane` (inference, `launch/infer_lane.sh` with `PASS=1…4`). Every lane looks after itself
(`utils/vm_guard.py`): it copies its logs to Hugging Face every 10 minutes, stops on a watchdog
(`IDLE_MIN`, `MAX_HOURS`), uploads and verifies its outputs, then **destroys** the instance on success or
**pauses** it on failure. `launch/e25_collect.sh` and `e26_collect.sh` wait on the laptop, copy the outputs
back and run the analysis.
