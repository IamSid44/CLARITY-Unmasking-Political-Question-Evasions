# Knowledge over Structure: Response Clarity Classification in Political Interviews

Team **Nier_ANLP** (IIIT Hyderabad) on **SemEval-2026 Task 6 (CLARITY)**: given a sub-question from a
U.S. presidential interview, the journalist's full question and the president's answer, classify
*how* the sub-question was answered: one of 9 evasion types (Subtask 2), which determines one of 3
clarity levels (Subtask 1).

| Member | Roll number |
|---|---|
| Vidvathama R | 2024122002 |
| Siddarth Gottumukkula | 2023102040 |
| Sanjana Reddy Vonteri | 2026901007 |
| Shashikanta Sahoo | 2026900007 |

## Links

| | |
|---|---|
| Mid-evaluation report | [`paper/main.pdf`](paper/main.pdf) (ACL format; LaTeX source in [`paper/`](paper/README.md)) |
| Trained runs (Hugging Face) | https://huggingface.co/siddarthg44/clarity-semeval26 — per-seed probabilities, metrics and LoRA adapters, as `<config>/seed<k>/` |
| Training curves (Weights & Biases) | https://wandb.ai/iamsid44-iiit-hyderabad/clarity-semeval26 — runs `<config>-s<seed>`, grouped by config |
| Code | https://github.com/IamSid44/CLARITY-Unmasking-Political-Question-Evasions (this repository) |

## Results (dev set)

Test labels were never released and the evaluation server has closed, so all numbers are on the
308-item dev set. Subtask 2 is multi-reference macro-F1 (a prediction counts if any annotator gave
it). A *system* averages 10 seeds and applies one-parameter logit adjustment, with τ chosen by nested
cross-validation.

| | Subtask 2 | Subtask 1 |
|---|---|---|
| **Qwen3-8B-Base + LoRA, 10-seed system** | **0.543** | **0.746** |
| **Qwen3-8B-Base + LoRA, single model (mean of 10 seeds)** | **0.476 ± 0.043** | **0.709 ± 0.031** |
| DeBERTa-v3-large, full question + 16 epochs, 10-seed system | 0.405 | 0.648 |
| DeBERTa-v3-large, full question + 16 epochs, single model | 0.384 ± 0.030 | 0.614 ± 0.027 |
| DeBERTa-v3-large baseline (sub-question + answer, 8 epochs), single model | 0.315 ± 0.040 | 0.576 ± 0.030 |
| TeleAI (1st place), multi-call DeepSeek-V3 pipeline | 0.617 | 0.812 |
| ChulaNLP (2nd place), DeBERTa top-5 → Kimi-K2 | 0.52 | 0.70 |
| human annotator scored against the other two | 0.684 | — |

## Where the project stands

**Done (mid-evaluation).**
- *Encoder track (E0–E12).* DeBERTa-v3-large built one measured change at a time. The full journalist
  question and 16 epochs gave +0.069 Subtask 2 per model on 9 of 10 seeds. Every decision-layer and
  structural alternative is documented as a negative result, with the reason it failed: re-ranking,
  three hierarchies, rebalancing losses, boundary experts and model soups.
- *The knowledge test (E13).* Swapping only the backbone for Qwen3-8B-Base with LoRA gave
  +0.092 Subtask 2 and +0.095 Subtask 1 per model on all 10 seeds, and lifted the system from
  0.405 to 0.543 (95% bootstrap interval of the difference [+0.054, +0.224]).

**Planned (to 31 October 2026; report §7).**
- *Structure on the stronger backbone.* The hierarchy designs that failed on DeBERTa are re-tested
  on Qwen, whose remaining errors sit inside the Ambivalent branch.
- *Accuracy against cost.* A cascade sends only Qwen's least-confident items, with its top
  candidates, to a larger LLM, tracing macro-F1 against LLM calls and cost per item.

## Repository layout

```
.
├── clarity/                the system: code, experiment definitions, records
│   ├── README.md               the task, results, method, findings, how to run (start here)
│   ├── encoder.py              DeBERTa-v3-large classifier
│   ├── llm_classifier.py       Qwen3-8B-Base + LoRA classifier
│   ├── decide.py               logit adjustment, nested cross-validation, 2-annotator scoring
│   ├── experiments/            the flags of every encoder run
│   └── reports/                scorer analysis, experiment log, research narrative,
│                                 number-to-source map, raw analysis outputs
├── paper/                  the report (official ACL template; main.tex, sections/, build.sh)
├── jarvis_drive.sh, jarvis_setup.sh, JARVISLABS_PORTING_GUIDE.md
│                           how the 8B runs were launched on rented GPUs
└── build_mid_submission.sh builds the mid-evaluation zip from this tree
```

## Reading order

1. **The report**, [`paper/main.pdf`](paper/main.pdf): the whole study in 8 pages.
2. **[`clarity/README.md`](clarity/README.md)** §1–4: the task, the system, results and key findings.
3. **[`clarity/reports/03_research_narrative.md`](clarity/reports/03_research_narrative.md)**: each
   experiment as a chain of reasoning, covering why it was run, what came out and why.
4. **[`clarity/reports/02_experiment_log.md`](clarity/reports/02_experiment_log.md)**: every
   experiment in order, with its hypothesis and prediction written before it ran.
5. **[`clarity/reports/04_results_sources.md`](clarity/reports/04_results_sources.md)**: every
   reported number with the file it comes from.

## Main findings so far

1. **The metric rewards landing among the acceptable labels and naming every class.** If every
   prediction is acceptable, macro-F1 equals the number of classes ever predicted divided by 9.
2. **The difficulty is mostly attribution.** 69% of training rows share their answer with another
   sub-question, so the model must find which part of a long answer responds to which question.
3. **Undertraining can pass for "this idea doesn't work".** The full-question input looked worse at
   8 epochs; at 16 it gives the best DeBERTa model on 9 of 10 seeds.
4. **Better training and the decision rule fix the same thing,** the pull towards frequent classes.
5. **One set of 5 seeds cannot rank systems on 308 items.** The same system scored 0.438 and 0.356
   on two seed sets, so every system claim uses 10 seeds.
6. **The encoder's gap is largely knowledge.** The 8B backbone wins on all 10 seeds. The gain is
   spread over 7 of 9 classes and is largest on the Non-Reply types. It does not sit on the
   commitment boundary, as we had predicted.
7. **The LLM needs the decision rule more.** Logit adjustment adds +0.062 per Qwen model against
   +0.005 for DeBERTa, because raw Qwen rarely predicts General.

## Reproducing

- Python 3.12, PyTorch 2.11, `transformers` 5.x, `peft` 0.20; scripts run from the repository root.
- All analyses run on CPU from the per-seed probabilities on the Hugging Face repo (download snippet
  in `clarity/README.md` §5): `python clarity/e11_replication.py`, `python clarity/e13_analysis.py`,
  `python clarity/qualitative_examples.py`.
- The report: `python clarity/paper_figures.py`, `python clarity/paper_example.py`,
  `python clarity/taxonomy_counts.py`, then `bash paper/build.sh`.
- Training: `clarity/README.md` §5 (encoder) and `JARVISLABS_PORTING_GUIDE.md` (8B classifier).
- API keys go in `clarity/.env` (template `clarity/.env.example`), which is gitignored.
