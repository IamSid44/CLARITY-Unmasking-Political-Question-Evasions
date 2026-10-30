# 07 — Beyond the shared task: the research extension plan

> **Superseded by the results (2026-10-30).** This is the plan as written before E24/E25 ran. What was run, what
> it showed, and the closing decisions (no human annotation, no further GPU runs, two subject LLMs) are in
> [`documentation.md`](documentation.md) and the experiment log (E24, E25). References below to annotating LLM
> answers and to four subject models describe the plan, not what was done.

*Written 2026-10-30, after the mid-evaluation feedback ("not enough innovation; the SemEval task alone is not
sufficient"). Inputs: seven literature scouts (SemEval-26 system papers and related tasks; efficient inference;
reframings; label variation and conformal prediction; interpretability and robustness; data and reasoning modules;
a hostile area-chair novelty check), and eight CPU pre-checks on our existing artifacts
([`raw/E24_prechecks.txt`](raw/E24_prechecks.txt), script `code/preregistered_analyses/e24_prechecks.py`).
Every claim below either has a number from our data or a cited prior work behind it.*

---

## 1. The problem with what we have, stated plainly

Our cascade is good engineering but not a research contribution on its own:

| What we built | Who already did it at SemEval-2026 Task 6 |
|---|---|
| Small model proposes top-k labels, an LLM chooses among them | ChulaNLP, 2nd on Subtask 2 (DeBERTa top-5 → LLM) — [2026.semeval-1.236](https://aclanthology.org/2026.semeval-1.236/) |
| Confidence routing: only uncertain items go to the big model | TeleAI, 1st on both subtasks — [2026.semeval-1.206](https://aclanthology.org/2026.semeval-1.206/); AsymVerify — [2607.20439](https://arxiv.org/abs/2607.20439) |
| Distilling a large model into a small student | YNU-HPCC (DeepSeek-V3.1 teacher) — [2026.semeval-1.45](https://aclanthology.org/2026.semeval-1.45/) |
| LLM-generated minority-class data | Duluth — [2604.20168](https://arxiv.org/pdf/2604.20168) |
| NLI reframing, multiple-instance learning over the answer | HU — [2026.semeval-1.34](https://aclanthology.org/2026.semeval-1.34/) |

Task overview (30 system papers): [2603.14027](https://arxiv.org/abs/2603.14027). Cascade prior art outside
the task: FrugalGPT ([2305.05176](https://arxiv.org/abs/2305.05176)), Conformal Cascade
([2607.25018](https://arxiv.org/abs/2607.25018)), CROQ ([2501.00555](https://arxiv.org/abs/2501.00555)).

What **nobody** in the 30 system papers did (verified by two independent scouts and the novelty check):
test whether systems read the question; evaluate outside presidential interviews; treat the evasion taxonomy as
a measurement instrument for anything other than presidents. That is where the paper goes.

---

## 2. The thesis

> **Evasion is a relation between a question and an answer, not a property of an answer.** We show the data
> encodes it as one, that relevance-based measures miss it, build a fast question-grounded detector for it,
> and use that detector to ask a new question: **do large language models evade like politicians?**

Working title: ***Answering Like a Politician? Question-Grounded Evasion Detection, and What It Reveals About
LLMs.***

Three contributions, in the order a reviewer should read them:

| | Contribution | Type | Status |
|---|---|---|---|
| **C1** | **Evasion is relational, and relevance is not answerhood.** QEvasion contains 3,079 natural contrast pairs (same answer, different sub-question); question–answer relevance, the measure political science uses, cannot separate replies from evasions | analysis / measurement validity | **evidence in hand (CPU)**; question-swap and knockout need ~25 GPU-min |
| **C2** | **A modular, latency-aware evasion detector** whose modules each earn their place by a pre-registered test; a negative-results ledger of five plausible modules that lose | system / efficiency | cascade exists; probe early-exit and answer compression need ~45 GPU-min |
| **C3** | **The first paired evasion audit of LLMs:** the exact sub-questions US presidents were asked, put to open LLMs under assistant / presidential-persona / follow-up-pressure conditions, labelled with C2 and validated by humans | new application, new resource | needs ~1.5 GPU-h + team annotation |

The novelty check rated C3 **novel** (no work applies this taxonomy, or Bull's equivocation typology, to LLM
outputs, or pairs LLM and presidential answers on identical questions) and C1 **novel for this task** (methods
are standard; the finding and the free contrast set are not). It rated our method ideas (conformal routing,
metric decision theory, shared-trunk ensembles) as **partly or fully scooped** — which our own pre-checks
independently confirmed by showing they do not help (§4). Leading with C1 + C3 is the recommended framing
for a Findings / main-track submission; C2 is the engine that makes C3 cheap and trustworthy.

---

## 3. Evidence already in hand (CPU, today)

All from [`raw/E24_prechecks.txt`](raw/E24_prechecks.txt).

**P1 — the data is relational.** 69% of train rows share their answer with another sub-question; of the 3,079
sibling pairs, **61.5% carry different evasion labels** and 40.6% different clarity levels.

**Noise control (the reviewer's sharpest objection to C1).** Two labels from different dev annotators disagree
**43%** of the time on the same item, **72%** across siblings, **83%** across random items. Sibling divergence
is far above annotator noise: the label moves because the question moves.

**P2 — our model reads the question.** On the 58 dev sibling pairs whose annotator sets are disjoint, M1 gives
the same label only 19% of the time (49.5% on overlapping pairs). We expected the opposite ("classifiers are
answer-only") and the data says no — which is why we test before we claim. Yet both siblings are right in only
**40%** of these pairs: question grounding exists but is the bottleneck.

**P8 — relevance is not answerhood.** Using our existing question–answer log-probability scores (the
"identifiability" idea of Alvarez & Morrier, *Political Analysis* 2025, the political-science measure of answer
quality), relevance is ordered across labels (Kruskal–Wallis p = 2e-12) but **barely discriminative**: AUC
**0.566** for Clear Reply vs rest and **0.479** for Clear Non-Reply. Deflection (0.80) and Partial (0.84)
answers are as identifiable from their question as Explicit ones (0.82). An answer can be *about* the question
without *answering* it — the construct our detector measures and relevance does not.

**Pre-checks that killed modules** (each is a paragraph in the paper's negative-results ledger):

| Module (source) | Pre-check | Result | Verdict |
|---|---|---|---|
| Route on epistemic uncertainty (mutual information over seeds) | P3 | catches 49/115 M1 errors in the top 30% vs 53 for entropy | reject |
| Sibling-aware joint decoding (CRF / mean field) | P4 | ±0.01, sign flips with τ (MI between siblings is only 0.24 bits) | not established |
| Parameter-free metric-optimal decision rule (our derivation) | P5 | 0.594 with no tuning vs 0.598 CV-tuned; 5/10 seeds better | theory section only |
| One debiased candidate order instead of four (PriDe) | P6 | agreement with the 4-order average *drops* (0.85→0.79); position is confounded with label identity | reject |
| Conformal multi-reference candidate sets as router + pruner | P7 | coverage holds (0.977 at α = 0.05) but S2 0.564–0.617 vs 0.641–0.663 for our router | reject |

Earlier negatives that join the ledger: learned value router (E20), cache-aligned prompt (E22a, 0.906
agreement), escalation ladder (E22c).

---

## 4. The modular architecture (C2)

The user-facing system is a pipeline of replaceable modules. Each has one job, one latency budget and one test
that decides whether it stays.

```text
 (question turn, sub-question, answer)
        │
 [A] Transcript normaliser ─────────── robustness; E21: artifacts cost −0.048 S2 when injected      KEPT
        │
 [B] Fast path: frozen-trunk probe ─── one pass of the frozen 8B, logistic probe on a middle layer;  NEW (G2)
        │  (early exit)                exits if confident. Target: ≥ 0.55 S2 at ≤ 0.6 forward passes
        │
 [C] Candidate generator ───────────── distilled student (0.608 system point) or 10-seed M1          KEPT
        │                               → top-4 labels, uncertainty u(x)
 [D] Router ────────────────────────── threshold on u(x); beat learned, conformal and epistemic       KEPT
        │                               routers in pre-registered tests (E20, P3, P7)
 [E] Adjudicator ───────────────────── 27B FP8 picks one letter in C(x), 4 cyclic orders averaged     KEPT
        │                               + question-conditioned answer compression                    NEW (G5)
 [F] Fusion + decision ─────────────── λ, τ by nested CV; metric-aware rule as the theory section    KEPT
        │
 [G] Question-grounding auditor ────── answer-only / swapped-question / null-question consistency;    NEW (G1)
        │                               flags items whose label does not depend on the question
 [H] Profiler (C3) ─────────────────── aggregates labels into per-model / per-condition evasion       NEW (G3–G4)
                                         profiles with bootstrap CIs
```

**Latency reporting.** Every module reports GPU-seconds per item *and* wall-clock p50/p95 per item at batch 1
and at full batch on one RTX PRO 6000. Known today: one 8B seed 0.04–0.05 GPU-s/item; one 27B call
0.10–0.15 GPU-s; accuracy point 0.576, cost point 0.139 GPU-s/item. The 10 seeds are ~91% of the accuracy
point's cost, so module [B] (one pass, no adapters) and the student are where latency is won.

**Why each NEW module is defensible**

- **[B] Probe early exit.** Intermediate layers of LLMs often classify as well as the last
  ([Skean et al., ICML 2025](https://proceedings.mlr.press/v267/skean25a.html)); fine-tuning is known to flatten
  the advantage ([2604.18592](https://arxiv.org/html/2604.18592)). Either outcome is a finding: if a frozen
  probe reaches the student, the fast path costs ~0.6 of one forward pass; if it lags far, the gap measures what
  fine-tuning adds to question–answer integration (connects to C1). Includes the *probe of the 27B* vs *prompt of
  the 27B* comparison at matched cost.
- **[E] Answer compression.** Answers median 266 tokens; the 27B call is prefill-bound. Keep the first N tokens
  plus the sentences most similar to the sub-question at 25/50/75% budgets
  ([LLMLingua-2](https://arxiv.org/abs/2403.12968) is the reference). Produces a per-class "where in the answer
  does evasion become visible" curve — an analysis politicians' answer structure makes interesting.
- **[G] Grounding auditor.** The C1 tests, packaged as a module: for any new domain (LLM answers in C3), it
  reports whether predictions change when the question changes. This is the validity check C3 needs.

**Considered and not built** (with the reason): batched multi-LoRA serving (engineering only; report as the fair
cost baseline if time allows); shared-trunk late-branch LoRA ensembles (scooped as a method by TreeNets/Hydra/
LoRA-Ensemble; at most an appendix ablation); GRPO with a taxonomy reward (>2 GPU-h, and RLVR is reported to hurt
disagreement modelling — [2506.19467](https://arxiv.org/pdf/2506.19467)); multi-agent debate (does not beat
self-consistency — [2311.17371](https://arxiv.org/pdf/2311.17371)); counterfactual answer rewriting for training
(LLM counterfactuals help less than human ones, and QEvasion's CC BY-NC-ND licence blocks releasing edits).

---

## 5. C3 in detail: the paired LLM evasion audit

**Design.** Sample 300 sub-questions from dev + test (stratified by president and by the gold clarity level of
the president's answer). For each, generate answers from ≥ 4 open models of different families (e.g. Qwen3-8B
instruct, Llama-3.1-8B-Instruct, Gemma-family ~9–27B, Mistral-7B; plus one API model in the paper version) under:

1. **Assistant** — the sub-question as a user question.
2. **Persona** — "You are the President of the United States, in a press interview on ⟨date⟩", with the full
   question turn.
3. **Follow-up pressure** — a second turn: "You did not answer the question. ⟨sub-question⟩".
4. *(paper version)* **Knowledge-controlled** — a short factual briefing so the model *can* answer; separates
   "cannot know" from "will not say" (the area chair's construct objection: Claims ignorance may be truthful).

Label all answers with the cascade (module [G] on), compare per-type distributions with the presidents' gold
labels on the same questions, and test condition effects with paired bootstrap.

**Validity, which decides whether C3 survives review.** Two team members label LLM answers blind to model and
condition: **100 by tomorrow, ≥ 300 for the paper**, κ reported next to QEvasion's (0.48 at 9 labels), and the
cascade's agreement with humans on LLM text reported separately. Length and format controls (strip markdown,
cap length) so "bulleted and long" is not mistaken for "evasive".

**What would be a finding beyond "LLMs hedge"** (already reported by
[Machine Bullshit](https://arxiv.org/abs/2507.07484) and [ELEPHANT](https://arxiv.org/abs/2505.13995)): which
*types* LLMs over- or under-use relative to presidents (e.g. General and Partial vs Declining), whether the
persona makes models more presidential, and whether follow-up pressure moves them toward Explicit or toward
Deflection.

**Closest work and our delta.** Herbold et al. had GPT-4 impersonate politicians and rated authenticity, not
evasion ([2407.12855](https://arxiv.org/abs/2407.12855)); Timoneda audits political engagement of assistants
without an evasion taxonomy or a human baseline ([2609.23039](https://arxiv.org/abs/2609.23039)); the refusal
taxonomy paper covers harmful prompts only ([2608.30856](https://arxiv.org/abs/2608.30856)); SpeechMap and
AbstentionBench are binary. The delta is the answerhood-relative nine-type profile, matched against humans on the
identical questions. **This space is filling month by month — speed matters.**

---

## 6. GPU plan, credit and ETA

Credit **₹785.80**. One RTX PRO 6000 VM at ~₹207/h → ~3.8 h ceiling. Plan **≈ 2.9 h ≈ ₹600**, leaving ~₹185.
Runs on the existing self-managing lane (`code/launch/infer_lane.sh` pattern: heartbeat to HF, watchdog,
upload-verify, then self-destroy), so a laptop disconnect cannot leave it billing.

| Step | What | GPU time | Feeds |
|---|---|---|---|
| G1 | Question-grounding tests: M1 (3 seeds) + student on dev + test under own / sibling-swap / random / null / answer-only question | 25 min | C1, module [G] |
| G2 | Hidden states of the frozen Qwen3-8B-Base (all layers, last token) for train + dev + test; 27B mid-layers for dev + test | 30 min | module [B] (probes fitted on CPU) |
| G3 | LLM answers: 300 sub-questions × 4 models × 3 conditions = 3,600 answers (vLLM, ≤ 300 new tokens) | 60 min | C3 |
| G4 | Label the 3,600 answers: student + 27B at δ = 0.5, plus M1 × 3 seeds as a robustness check | 35 min | C3 |
| G5 | Answer compression curve: 27B on the routed dev items at 25/50/75/100% token budgets | 15 min | module [E] |
| — | Setup, model downloads, buffer | 15 min | |

**ETA.** Code for G1–G5 and a dry run: ~3 h after approval. Lane: ~3 h unattended (overnight is fine). CPU
analysis and figures: ~2 h. Human annotation of 100 LLM answers by two people: ~2 h, in parallel. **The course
report (due 2026-10-31) can carry C1 in full, C2 with modules [B] and [E] measured, and C3 as a pilot with the
100-item validation.** The paper version needs the items marked *(paper version)* above, ≥ 300 human labels,
one API model, and roughly two more weeks.

---

## 7. What else is worth doing after the course (ranked)

1. **Interviewer pursuit as free supervision.** In conversation analysis, an interviewer re-asking or
   challenging ("you didn't answer…") marks a non-answer (Clayman & Heritage). The `url` and `question_order`
   fields let us rebuild the next turn; MediaSum and C-QUERI give millions of unlabelled exchanges. No prior work
   found. High-precision weak labels for the non-reply classes.
2. **Cross-domain transfer at the 3-level granularity.** EvasionBench (earnings calls, Apache-2.0,
   [2601.09142](https://arxiv.org/abs/2601.09142)), SubjECTive-QA (CC BY), Ferracane hearings
   ([2104.04470](https://arxiv.org/abs/2104.04470)), PolERo (Romanian, [2609.02391](https://arxiv.org/abs/2609.02391)).
   EvasionBench lists cross-domain transfer as future work; KCLarity found earnings-call pre-training *hurt*.
   Zero-shot runs cost ~10 GPU-min each.
3. **Leave-one-president-out evaluation** (2006–2023, four presidents): does the detector learn speaker style?
4. **QUD-grounded localisation:** which answer sentence (if any) resolves the sub-question; interpretable and
   targets Partial vs Implicit.
5. **Evasion red-teaming:** can LLMs be prompted to evade *undetectably*, by our detector and by humans?

---

## 8. Reviewer objections we expect, and the answers we already have

| Objection | Answer |
|---|---|
| "The cascade is ChulaNLP/TeleAI." | Agreed — it is the instrument, not the contribution (§1). |
| "Sibling labels differ because annotators are noisy." | 43% same-item vs 72% sibling vs 83% random disagreement (§3). |
| "Your classifier is invalid on LLM text." | 100 → 300 human labels with κ; module [G] consistency; length/format controls (§5). |
| "Claims ignorance by an LLM is honest, not evasive." | Knowledge-controlled condition (§5, paper version). |
| "Gains on 308 dev items are noise." | Cost claims framed as non-inferiority; refit-in-bootstrap CIs already standard in our log (E19). |
| "Conformal / decision theory is known." | We do not claim them; the pre-checks show they do not help here (§3), reported as negative results. |
| "Licence." | QEvasion is CC BY-NC-ND: we release no edited QEvasion text. LLM answers are new text; question text can be re-sourced from public-domain White House transcripts. |

---

## 9. Decisions needed from the team

1. **Approve the ~₹600 GPU plan (G1–G5)?** Nothing is launched until then.
2. **Which open models for C3** (4 families, each ≤ 27B so one GPU serves it).
3. **Who annotates** the 100 LLM answers (two people, blind to model and condition).
