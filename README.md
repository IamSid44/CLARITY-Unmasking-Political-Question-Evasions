# CLARITY — Unmasking Political Question Evasions

Team Nier_ANLP's work on **SemEval-2026 Task 6 (CLARITY)**: given a question from a
U.S. presidential interview and the president's answer, classify *how* the question
was answered — one of 9 evasion strategies (Subtask 2), which determines one of 3
clarity levels (Subtask 1).

| | dev Subtask 2 | dev Subtask 1 |
|---|---|---|
| **Our best** — 5 fine-tuned DeBERTa-v3-large models + post-hoc logit adjustment | **0.438** | — |
| Our single fine-tuned model (mean of 5 seeds) | 0.337 ± 0.044 | 0.586 ± 0.039 |
| ChulaNLP (2nd place), fine-tuned DeBERTa-large | 0.46 | 0.65 |
| TeleAI (1st place), DeepSeek-V3 multi-stage pipeline | 0.617 | 0.812 |

All numbers are on the 308-item **dev** set. Codabench has closed and the test labels
were never released, so dev is the only common ground for comparison; see
[`clarity/reports/02_experiment_log.md`](clarity/reports/02_experiment_log.md) §E7
for what these comparisons can and cannot show.

**Status:** the next experiment, E8 (full question + Balanced Softmax + focal loss),
is configured and waiting to launch.

---

## Repository layout

```
.
├── clarity/           the system: code, experiment definitions, reports, packaged predictions
│   ├── README.md          start here
│   └── reports/           the scorer analysis, the experiment log, the re-ranker ablation
├── Materials/         project proposal and reference papers (task overview, TeleAI,
│                        ChulaNLP, the QEvasion dataset paper)
├── MIDEVAL_PLAN.md    the team's mid-evaluation plan and pitch
└── README.md
```

Everything lives in **[`clarity/`](clarity/README.md)**: an end-to-end fine-tuned
DeBERTa-v3-large, built from the simplest system that could work and extended one
measured step at a time — baseline → seed ensemble → decision rules → hierarchy check
→ second-stage re-ranker → (next) richer input and loss.

**Earlier work, archived.** The team's first track, `higrec` (a frozen-backbone
analysis pipeline, the inter-annotator agreement study, and the planning documents
that preceded this work), is no longer in the current tree. It remains in the git
history. The three modules the system still needs from it — data loading, the label
vocabulary and the official-scorer replica — were carried over into
`clarity/qevasion/` and checked to give identical results.

---

## Reading order

For someone new to the project:

1. **The task and the system** — [`clarity/README.md`](clarity/README.md).
2. **What the metric actually rewards** —
   [`clarity/reports/01_scorer_geometry.md`](clarity/reports/01_scorer_geometry.md).
   Short, and it changes how every result should be read.
3. **What we did and found, in order** —
   [`clarity/reports/02_experiment_log.md`](clarity/reports/02_experiment_log.md),
   including what runs next and why.
4. **A negative result, written up** —
   [`clarity/reports/03_reranker_ablation.md`](clarity/reports/03_reranker_ablation.md).
5. **The mid-evaluation plan** — [`MIDEVAL_PLAN.md`](MIDEVAL_PLAN.md).

---

## Main findings so far

1. **The metric rewards landing among the acceptable labels, not matching the
   consensus.** A prediction is correct if any annotator gave that label, and if
   every prediction is acceptable, macro-F1 equals the number of classes ever
   predicted divided by 9.
2. **The difficulty is mostly attribution.** Most answers respond to several
   sub-questions at once; without knowing which sub-question is being asked,
   accuracy is capped at 0.753.
3. **Failures follow human disagreement, not class rarity.** `General` has plenty of
   training data and near-worst F1; it is also the label annotators disagree on most.
4. **A one-parameter decision rule is the biggest single gain** (+0.073), and the
   textbook per-class alternative overfits the small dev set.
5. **The right answer is in the model's top 3 for 90% of items**, but a second
   encoder trained to choose among them does worse than the first — a documented
   negative result, and the slot an LLM would fill.
6. **Re-partitioning the taxonomy along "coverage" lowers annotator agreement**
   (α 0.50 vs 0.62 for the official partition, p = 0.0004) — the earlier track's
   pre-registered hypothesis, refuted.

---

## Reproducing

- Environment: Python 3.12, PyTorch 2.11, `transformers` 5.x. No package install;
  scripts run from the repo root, e.g. `python clarity/analyze.py --run-dir …`.
- Launch or resume an experiment: `bash clarity/start.sh E8` (tmux, resumable;
  `clarity/README.md` §5).
- Every number in the scorer report: `python clarity/verify_scorer_geometry.py`
  (CPU, seconds).
- Trained checkpoints are not in git; they are in a private Hugging Face repository.
- API keys go in `clarity/.env` (template: `clarity/.env.example`); never commit it.
