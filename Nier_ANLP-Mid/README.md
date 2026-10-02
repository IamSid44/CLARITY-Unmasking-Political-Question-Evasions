# Nier_ANLP — Mid-Evaluation Submission

**Knowledge over Structure: A Controlled Study of Response Clarity Classification in Political
Interviews.** SemEval-2026 Task 6 (CLARITY).

| Member | Roll number |
|---|---|
| Vidvathama R | 2024122002 |
| Siddarth Gottumukkula | 2023102040 |
| Sanjana Reddy Vonteri | 2026901007 |
| Shashikanta Sahoo | 2026900007 |

International Institute of Information Technology, Hyderabad.

## Links

| | |
|---|---|
| Code repository | https://github.com/IamSid44/CLARITY-Unmasking-Political-Question-Evasions |
| Trained runs (Hugging Face) | https://huggingface.co/siddarthg44/clarity-semeval26 — per-seed dev/test probabilities, metrics and LoRA adapters, under `<config>/seed<k>/` |
| Training curves (Weights & Biases) | https://wandb.ai/iamsid44-iiit-hyderabad/clarity-semeval26 — runs named `<config>-s<seed>`, grouped by config |

## What is in this folder

| path | contents |
|---|---|
| `Nier_ANLP-Mid-Report.pdf` | the mid-evaluation report (ACL format; about 7.5 pages including Limitations and Ethics, then references and a one-page appendix) |
| `paper/` | its LaTeX source, ready for Overleaf (compiler pdfLaTeX, main document `main.tex`) |
| `clarity/` | the system: code, experiment definitions, the experiment log, the reports behind every number |
| `jarvis_drive.sh`, `jarvis_setup.sh`, `JARVISLABS_PORTING_GUIDE.md` | how the 8B-model runs were launched on rented GPUs |

Inside `clarity/`:

| path | contents |
|---|---|
| `encoder.py` | DeBERTa-v3-large 9-way classifier (training, epoch selection on the train slice, probabilities) |
| `llm_classifier.py` | Qwen3-8B-Base + LoRA as a 9-way classifier (same input, rows and selection rule) |
| `decide.py` | decision rules: logit adjustment with τ fitted by nested cross-validation, and the alternatives we tested |
| `qevasion/` | data loading, label vocabulary, and a replica of the official scorer |
| `e11_replication.py`, `e13_analysis.py`, `mideval_analysis.py` | the analyses behind the reported numbers (CPU, from the saved per-seed probabilities) |
| `paper_figures.py`, `paper_example.py`, `taxonomy_counts.py` | the report's data figures (Figures 2–3), the worked example (Figure 4) and Table 1 |
| `experiments/` | the argument files of every experiment (E8–E12) |
| `reports/02_experiment_log.md` | the chronological log: every experiment's hypothesis and prediction, written before it ran, then its result |
| `reports/03_research_narrative.md` | the same work as a narrative, experiment by experiment |
| `reports/04_results_sources.md` | every number with the file it comes from |
| `reports/raw/` | raw outputs of the analyses |
| `submissions/` | packaged predictions of the DeBERTa systems |

## Results so far (dev set; test labels were never released)

| system | Subtask 2 (9 classes) | Subtask 1 (3 classes) |
|---|---|---|
| DeBERTa-v3-large baseline, single model (mean of 10 seeds) | 0.315 ± 0.040 | 0.576 ± 0.030 |
| DeBERTa-v3-large, full question + 16 epochs, single model | 0.384 ± 0.030 | 0.614 ± 0.027 |
| DeBERTa 10-seed system (ensemble + logit adjustment) | 0.405 | 0.648 |
| **Qwen3-8B-Base + LoRA, single model (mean of 10 seeds)** | **0.476 ± 0.043** | **0.709 ± 0.031** |
| **Qwen3-8B-Base + LoRA, 10-seed system** | **0.543** | **0.746** |
| TeleAI (1st place), multi-call DeepSeek-V3 pipeline | 0.617 | 0.812 |
| human annotator scored against the other two | 0.684 | — |

Subtask 2 is multi-reference macro-F1 (a prediction counts if any annotator gave it). The plan and
timeline to the final evaluation (31 October 2026) are in §7 of the report.

## Reproducing

- Python 3.12, PyTorch 2.11, `transformers` 5.x, `peft` 0.20; scripts run from the folder root,
  e.g. `python clarity/e13_analysis.py`.
- The analyses need the per-seed probabilities in `clarity/runs/`. Fetch them, without weights,
  from the Hugging Face repo:
  ```python
  from huggingface_hub import snapshot_download
  snapshot_download("siddarthg44/clarity-semeval26", local_dir="clarity/runs",
                    allow_patterns=["*/seed*/*.npy", "*/seed*/*.json"])
  ```
- The data are downloaded on first use from the public dataset `ailsntua/QEvasion`
  (`clarity/qevasion/loader.py`).
- Training: `bash clarity/start.sh <experiment>` for the encoder (see `clarity/README.md` §5);
  the 8B classifier through `jarvis_drive.sh` (see `JARVISLABS_PORTING_GUIDE.md`).
- API keys, if you want W&B or Hugging Face logging, go in `clarity/.env` (template
  `clarity/.env.example`).
