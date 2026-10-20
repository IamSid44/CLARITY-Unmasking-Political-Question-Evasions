# Reproducing the results

All commands run from `code/`. Train and dev data download automatically on first use.

## Setup

```bash
pip install -r requirements.txt      # Python 3.12
cd code
```

**Optional keys.** For W&B logging, Hugging Face uploads or the API arm of M3, copy `code/.env.example`
to `code/.env` and fill it in. Without keys, uploads are skipped and runs still complete.

## 1. Re-run the analyses from the published runs (CPU only)

Fetch the per-seed probabilities and metrics (no model weights):

```python
from huggingface_hub import snapshot_download
snapshot_download("siddarthg44/clarity-semeval26", local_dir="../runs",
                  allow_patterns=["*/seed*/*.npy", "*/seed*/*.json"])
```

Then:

```bash
python -m preregistered_analyses.e13_analysis          # Qwen vs DeBERTa, per seed and as systems
python -m preregistered_analyses.mideval_analysis      # main table, error analysis, human ceiling
python -m preregistered_analyses.e11_replication       # DeBERTa replication on fresh seeds
python -m decision_rules.decide --run-dir ../runs/E10_fullq_16ep
python -m evaluation.verify_scorer_geometry
python -m figures.paper_figures                        # report figures -> report/latex/figures/
```

The cascade analyses (E14–E18) also need the cascade outputs (`runs/M1`, `runs/M3`, `runs/M4`) from the
same Hugging Face repository.

## 2. Train the classifiers (GPU)

**DeBERTa-v3-large** (24 GB is enough). For example, the first run of experiment E10:

```bash
python -m models.encoder $(cat experiments/E10/common.args) \
    --name E10_fullq_16ep --seed 0 --input full --max-len 1024 --a-budget 256
```

**Qwen3-8B-Base + LoRA** (we used one 96 GB RTX PRO 6000):

```bash
python -m models.llm_classifier --name Q8_fullq_lora --seed 0                    # E13, 3 epochs
python -m models.llm_classifier --name Q8_fullq_lora_12ep --seed 0 --epochs 12   # E14, M1
python -m models.llm_classifier --name Q8_12ep_crossfit --seed 0 --fold 2 \
    --select last --epochs 12                                                    # cross-fitted train rows
```

## 3. The cascade

```bash
# M1: candidate sets and uncertainty from the 10-seed ensemble (CPU)
python -m cascade.candidates --run Q8_fullq_lora_12ep --crossfit Q8_12ep_crossfit --method aps --alpha 0.1

# M3: LLM adjudicator over M1's top 4 (local vLLM, Qwen3.8-27B-FP8), then fusion and the deferral curve
python -m cascade.exemplars embed
bash launch/m3_local.sh screen && bash launch/m3_local.sh full 0
python -m cascade.fuse dev --tag qwen38_27b_k0

# M2: accuracy vs cost over seed count x deferral rate
python -m cascade.frontier jobs
python -m cascade.frontier analyse

# M4: soft targets, then one student per seed
python -m cascade.distill_targets --alpha 0.5
python -m models.llm_classifier --name Q8_distill_a05 --seed 0 --epochs 12 --soft-targets ../runs/M4/targets_a0.5.npy

# Final numbers and the frontier figure
python -m preregistered_analyses.e18_final > ../docs/raw/E18_final.txt
```

The exact order used, with the server environments, is in
[`05_phase2_plan_and_budget.md`](05_phase2_plan_and_budget.md) §6, and each run list is in
`code/experiments/`.
