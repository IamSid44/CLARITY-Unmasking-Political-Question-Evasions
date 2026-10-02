# CLARITY — Unmasking Political Question Evasions

Team Nier_ANLP's work on **SemEval-2026 Task 6 (CLARITY)**: given a question from a
U.S. presidential interview and the president's answer, classify *how* the question
was answered — one of 9 evasion strategies (Subtask 2), which determines one of 3
clarity levels (Subtask 1).

| | dev Subtask 2 | dev Subtask 1 |
|---|---|---|
| **Our best system** — Qwen3-8B-Base + LoRA (E13), 10-seed ensemble + logit adjustment | **0.543** | **0.746** |
| **Our best single model** — Qwen3-8B-Base + LoRA, same input and protocol as DeBERTa (mean of 10 seeds) | **0.476 ± 0.043** | **0.709 ± 0.031** |
| Qwen3-8B LoRA, 12 epochs instead of 3 (E13b, 3 seeds; adopted on the train slice) | 0.543 ± 0.015 | 0.742 ± 0.008 |
| Best DeBERTa single model — DeBERTa-v3-large with the full question, 16 epochs (mean of 10 seeds) | 0.384 ± 0.030 | 0.614 ± 0.027 |
| Our baseline single model — sub-question + answer, 8 epochs (mean of 10 seeds) | 0.315 ± 0.040 | 0.576 ± 0.030 |
| DeBERTa final system (E11) — 10 of the best DeBERTa models + post-hoc logit adjustment | 0.405 | 0.648 |
| Our baseline system — 10 baseline models + logit adjustment | 0.428 | 0.601 |
| ChulaNLP (2nd place), fine-tuned DeBERTa-large | 0.46 | 0.65 |
| TeleAI (1st place), DeepSeek-V3 multi-stage pipeline | 0.617 | 0.812 |

All numbers are on the 308-item **dev** set. Codabench has closed and the test labels
were never released, so dev is the only common ground for comparison; see
[`clarity/reports/02_experiment_log.md`](clarity/reports/02_experiment_log.md) §E7
for what these comparisons can and cannot show.

**Status (2026-10-02, mid-evaluation).** Two tracks, one protocol:
- **The encoder track (E0–E12)** built DeBERTa-v3-large one measured change at a time:
  - full journalist question + 16 epochs: +0.069 S2 per model, on 9 of 10 seeds;
  - final system 0.405 / 0.648;
  - every decision-layer and architecture alternative is documented as a negative result, with
    the reason it failed.
- **E13 (2026-10-02)** asked whether the encoder's gap is *knowledge*. It swapped the backbone for
  Qwen3-8B-Base with LoRA and changed nothing else:
  - **+0.092 S2 and +0.095 S1 per model on all 10 seeds;**
  - **system S2 0.543 / S1 0.746**, against 0.405 / 0.648 (+0.136, 95% CI [+0.054, +0.224]);
  - 12 epochs instead of 3 is adopted by the train-slice rule (3-seed screen);
  - training on all of train gives a small gain that isn't distinguishable from zero.
- **Next: the label-meaning test.** A cascade sends only uncertain items to a larger LLM with
  label definitions and boundary examples, giving an accuracy-vs-cost curve.
  [`mideval/`](mideval/REPORT.md) is the mid-evaluation package.
- **Nothing is running.** The JarvisLabs VMs are paused; see [`CONTEXT.md`](CONTEXT.md).

---

## Repository layout

```
.
├── CONTEXT.md         handover: current state, what to do next, rules, environment (read first)
├── CLAUDE.md          standing instructions for Claude Code sessions
├── clarity/           the system: code, experiment definitions, reports, packaged predictions
│   ├── README.md          start here
│   ├── llm_classifier.py  E13: decoder LLM + LoRA as a 9-way classifier
│   ├── e13_analysis.py    E13 analyses (paired seeds, systems, 2-annotator, per class)
│   └── reports/           the scorer analysis, the experiment log, the research narrative
├── mideval/           mid-evaluation package: REPORT, RESULTS (every number + source), slides
├── jarvis_drive.sh    launch machine: VMs on JarvisLabs (create/resume, push, start, watch, pause)
├── jarvis_setup.sh    on each VM: environment, model, smoke, pilot, training, HF upload, pause
├── JARVISLABS_PORTING_GUIDE.md   how to run the LLM experiments on JarvisLabs
├── Materials/         project proposal and reference papers (task overview, TeleAI,
│                        ChulaNLP, the QEvasion dataset paper)
└── README.md
```

Everything lives in **[`clarity/`](clarity/README.md)**: an end-to-end fine-tuned
DeBERTa-v3-large, built from the simplest system that could work and extended one
measured step at a time — baseline → seed ensemble → decision rules → hierarchy check
→ second-stage re-ranker → richer input, loss and output head (E8–E9) → training
length (E10) → replication (E11) → data-driven variants (E12) → an 8B LLM classifier with
LoRA under the same protocol (E13).

**Earlier work, archived.** The team's first track, `higrec` (a frozen-backbone
analysis pipeline, the inter-annotator agreement study, and the planning documents
that preceded this work), is no longer in the current tree. It remains in the git
history. The three modules the system still needs from it — data loading, the label
vocabulary and the official-scorer replica — were carried over into
`clarity/qevasion/` and checked to give identical results.

---

## Reading order

For someone new to the project, about an hour in total:

0. **The research narrative** —
   [`clarity/reports/03_research_narrative.md`](clarity/reports/03_research_narrative.md)
   (20 min): the whole track as a chain of reasoning — for each experiment, what
   led to it, the setup, the result and why it worked or failed. Start here.
1. **This page**, then **the task and the system** —
   [`clarity/README.md`](clarity/README.md) §1–4 (15 min). What the task is, what
   the model does, the results table and the key findings.
2. **What the metric actually rewards** —
   [`clarity/reports/01_scorer_geometry.md`](clarity/reports/01_scorer_geometry.md)
   (10 min). Short, and it changes how every result should be read.
3. **The experiment log** —
   [`clarity/reports/02_experiment_log.md`](clarity/reports/02_experiment_log.md).
   It is long because every run is recorded; read it selectively (30 min):
   - the **Scoreboard** at the top — every configuration in one table;
   - **Step 0** — the first-principles analysis the whole plan rests on;
   - **E4** — the decision rule behind the best result;
   - **E8b results** — why "it didn't help" can mean "it wasn't trained enough",
     including a correction to how E8 was first read;
   - **E10** — the 2 × 2 of input and training length;
   - **E11 results** — the replication on new seeds, judged against hypotheses
     fixed in advance, and how the final system was chosen;
   - **E13, E13x/b/c** — the 8B classifier, pre-registered, with its 10-seed results.
4. **The mid-evaluation package** — [`mideval/REPORT.md`](mideval/REPORT.md) (10 min).

To run anything: `clarity/README.md` §5.

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
7. **Undertraining can pass for "this idea doesn't work".** With the full question
   in the input the model needs about twice the epochs to fit; at 8 epochs it leans
   on the class prior and looks worse than it is (0.343). Trained 16 epochs it gives
   a better model on 9 of 10 seeds.
8. **Better training and the decision rule fix the same thing.** Both remove the pull
   towards frequent classes; on Subtask 2 either is enough, and together they add
   nothing.
9. **One set of 5 seeds cannot rank systems on 308 items.** The same system scored
   0.438 and 0.356 on two seed sets; per-seed paired comparisons over 10 seeds carry
   the conclusions.
10. **The encoder's gap is largely knowledge (E13).** Swapping DeBERTa (0.4B) for a
    LoRA-tuned Qwen3-8B, with nothing else changed, wins on all 10 seeds (+0.092 S2,
    +0.095 S1). The gain is spread over 7 of 9 classes and over agreed and contested items
    alike. Our prediction that it would sit on the commitment boundary is refuted.
11. **The LLM needs the decision rule more.** Logit adjustment adds +0.062 per Qwen
    model (+0.005 for DeBERTa): raw Qwen rarely predicts `General`.

---

## Reproducing

- Environment: Python 3.12, PyTorch 2.11, `transformers` 5.x. No package install;
  scripts run from the repo root, e.g. `python clarity/analyze.py --run-dir …`.
- Launch or resume an experiment: `bash clarity/start.sh E11` (tmux, resumable;
  `clarity/README.md` §5).
- Every number in the scorer report: `python clarity/verify_scorer_geometry.py`
  (CPU, seconds).
- Trained runs are not in git. They are on the team's Hugging Face repo `siddarthg44/clarity-semeval26`,
  under `<config>/seed<k>/`: probabilities, metrics and LoRA adapters. **Note:** the repo is
  currently **public**.
- LLM runs (E13): see [`JARVISLABS_PORTING_GUIDE.md`](JARVISLABS_PORTING_GUIDE.md) and
  `mideval/REPRODUCE.md`. Analyses: `python clarity/e13_analysis.py`,
  `python clarity/e13_figures.py`.
- API keys go in `clarity/.env` (template: `clarity/.env.example`); never commit it.
