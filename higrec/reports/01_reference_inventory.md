# 01 — Reference Inventory (Phase 0.2)

All clones are **read-only**. Per CLAUDE.md invariant 1 nothing here is imported into `src/`,
nothing here is on the Python path, and no prompt text or label definition is copied out.
`REF/` is gitignored. Cloned 2026-09-12 with `scripts/clone_refs.sh` (`--depth 50`).

**6 requested, 6 obtained, 0 failures.**

| Repo | Commit | License | Why it matters |
|---|---|---|---|
| `konstantinosftw/Question-Evasion` | `ff78482` | MIT | Organizers' QEvasion dataset repo (Thomas et al. 2024). Per-annotator labels + their own agreement code. |
| `ther7777/semeval-2026-task6-camsr-cot` | `9cd8620` | see repo | TeleAI, 1st place both subtasks. **Carries the only multi-reference scorer implementation found.** |
| `moswisarut/SemEval2026-Task6-moswisarut` | `c231986` | see repo | ChulaNLP, 2nd on Subtask 2. |
| `semeval-2026-kclarity/clarity` | `f850d00` | see repo | KCLarity. Fallback scorer source. |
| `CLaC-Lab/SemEval-2026-task6-CLARITY` | `ba1349b` | see repo | CLaC. Fallback scorer source. |
| `syed0093-umn/SemEval2026_Task6_Duluth` | `7e7f70d` | see repo | Duluth. Fallback scorer source; contains annotator-aware training (C3 prior art). |

**Correction to `Implementation_Guide.md`:** the guide states the ChulaNLP repo could not be
retrieved and "may be private, renamed, or unindexed", and advises treating it as optional.
It is public and cloned without incident. It contains 13 Jupyter notebooks — one per
experimental configuration — plus `Preprocess.ipynb`.

---

## The official Subtask 2 scorer

**Location:** `REF/semeval-2026-task6-camsr-cot/scripts/eval_competition.py`
**Function:** `_compute_macro_f1_multiref_official()` (lines 329–378)
**Subtask 1 function:** `_compute_macro_f1_multilabel_ovr()` (lines 285–326)

**Important provenance caveat.** This is **TeleAI's reimplementation**, not the organizers'
source. Its module docstring describes it as "与 Codabench scorer 一致" ("consistent with the
Codabench scorer"), and their paper says they "utilize the official scorer provided by the
task organizers" — so this file is their local replica, written to match. It is a
first-rate source (the team that topped the leaderboard with it) but it is **second-hand**.

I searched all six repos for an independent implementation of the multi-reference rule.
**There is none.** Every other participant repo collapses the multi-reference gold to a
majority vote or a single annotator before scoring (e.g. Duluth's
`evaluation/predict_majority_vote.py`; ChulaNLP's paper states it used "annotator 3 as the
reference label"). So there is no second implementation available to cross-validate against.

**Consequence for P1.2:** the conformance suite can only be run against this one
implementation. Risk R1 is therefore live and must be stated in the paper: our scorer is
conformance-tested against the 1st-place team's replica of the official scorer, not against
the organizers' own code. Obtaining the Codabench bundle would close this and remains worth
attempting.

---

## Data files found in REF

### `konstantinosftw/Question-Evasion/dataset/`
- `QAEvasion.csv` — the main dataset.
- `Inter-Annotator/Annotator{1,2,3}.csv` — per-annotator label files, `[idx, Interview
  Question, Interview Answer, Question, Label]`.
- `Inter-Annotator/test_set.csv` — **317 rows × explicit `Annotator1/2/3` columns.** The raw
  multi-reference set, before the 317→308 cleanup. Contains 11 label values including
  `2.9 Diffusion` and `2.5 Contradictory` (see CLAUDE.md invariant 8).
- `Counterfactual-Summaries/Annotator{1,2,3}_countersummaries.json` — the counterfactual
  attention-check items. **Directly relevant to Phase 8**, which calls for injecting
  counterfactual items "following the original dataset protocol's check". The protocol's
  actual items are here.

### `ther7777/semeval-2026-task6-camsr-cot/data/`
- `train.jsonl`, `dev.jsonl`, **`eval_test.jsonl`** — the competition splits in JSONL form,
  including the **237-row evaluation-phase test set**, which is not on HuggingFace.
  Cross-check only; canonical source stays the HF parquet.

### Organizers' own agreement code
`Question-Evasion/scripts/datasetAnalysis.py` computes Fleiss' κ over the three annotator
files (`calculate_fleiss_kappa`, line ~131) and filters to items where all three annotators
used in-taxonomy labels (line ~112). **This is how the published κ=0.64 / κ=0.48 figures were
produced.** Phase 2 should reproduce those numbers as a sanity check before trusting our own
agreement pipeline — if we cannot reproduce the organizers' published κ from their own data
with their own filtering rule, our α computation is suspect.

---

## Prior art that C3 must be differentiated from

`SemEval2026_Task6_Duluth/.../training/train_annotator_aware.py` reads
`annotator1/2/3` during training. A team already attempted annotator-aware modelling on this
task. Phase 5 must read this file, establish exactly what it did, and state how C3's
marginal-bias / low-rank-confusion formulation differs. This was not anticipated by the
proposal, the guide, or the playbook, and the claim "every surveyed system collapses this to
a single gold label" (proposal G3) may need qualifying.

---

## Deliberately not yet read

`REF/semeval-2026-task6-camsr-cot/prompts/**` holds TeleAI's refined label definitions,
confusion guides, and few-shot anchor/boundary libraries. Per invariant 1 these are **not**
to be copied. They are read for one sanctioned purpose only — the adversarial analysis in
`reports/02_teleai_label_definitions_analysis.md`, which asks whether their decision tree
implicitly orders checks along a coverage axis. That analysis is **not yet written**.
