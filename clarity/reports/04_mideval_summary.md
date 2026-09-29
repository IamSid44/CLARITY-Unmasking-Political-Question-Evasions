# 04 — Mid-evaluation summary

The encoder track on one page: where it started, what each change did, what did
not work and why. All numbers are on the 308-item dev set (Subtask 2 macro-F1 unless
marked) and trace to [`02_experiment_log.md`](02_experiment_log.md). Nothing was
chosen on dev except the decision rule's one parameter τ, which is always scored by
nested cross-validation.

*Updated 2026-09-28 with E11, the replication on new seeds.*

---

## 1. In three sentences

1. **We made the model reliably better:** giving it the full journalist question
   and twice the training improves a single fine-tuned DeBERTa-v3-large on **9 of
   10 seeds**, from 0.315 to 0.384 on Subtask 2 and from 0.576 to 0.614 on
   Subtask 1.
2. **At the level of the final system the gain on Subtask 2 is within noise**
   (0.405 vs 0.428 for the baseline system), because the post-hoc decision rule
   already repairs most of what the better model fixes; on Subtask 1 the final
   system is ahead (0.648 vs 0.601).
3. **A 5-seed ensemble on 308 items is not enough to rank systems:** the same
   baseline system scored 0.438 on one set of seeds and 0.356 on another, so every
   system claim here rests on 10 seeds and paired comparisons.

## 2. The progression, per model (10 seeds each)

| step | model | dev S2 | dev S1 | vs previous, per seed |
|---|---|---|---|---|
| 0 | DeBERTa-v3-large fine-tuned on sub-question + answer, 8 epochs | 0.315 ± 0.040 | 0.576 ± 0.030 | — |
| 1 | + 16 epochs instead of 8 | 0.362 ± 0.026 | 0.601 ± 0.016 | +0.047, 9/10 seeds up |
| 2 | + the full journalist question in the input (1024 tokens) | **0.384 ± 0.030** | **0.614 ± 0.027** | +0.022, 7/10 seeds up |

Step 0 → 2: +0.069 on Subtask 2 (t = 3.6) and +0.038 on Subtask 1 (t = 2.8), 9 of
10 seeds up on each.

## 3. From model to system

A system is an ensemble of seeds (probabilities averaged) followed by logit
adjustment (divide by the class prior^τ; Menon et al., ICLR 2021).

| 10 seeds, dev S2 | single model | + rule | ensemble | ensemble + rule |
|---|---|---|---|---|
| step 0 — baseline | 0.315 | 0.341 | 0.319 | **0.428** |
| step 1 — 16 epochs | 0.362 | 0.368 | 0.377 | 0.400 |
| step 2 — full question, 16 epochs | 0.384 | 0.389 | **0.409** | 0.405 |

The baseline's models lean on the class prior (they rarely predict rare classes),
and the rule fixes exactly that: +0.109 on its ensemble. The better-trained models
lean on the prior much less, so the rule has nothing left to fix (−0.004). The two
routes — train better, or correct afterwards — end in the same place on Subtask 2.
Subtask 1 favours training better: 0.648 against 0.601.

**Final system** (chosen by a rule written before E11 ran): the 10-seed
full-question ensemble with logit adjustment — dev S2 0.405 (0.412 averaged over
CV splits), dev S1 0.648; packaged in `submissions/FINAL_fullq_16ep_10seed_logitadj/`.
For reference: ChulaNLP's fine-tuned DeBERTa-large, 0.46 / 0.65 (checkpoint chosen
on dev); TeleAI's DeepSeek-V3 pipeline, 0.617 / 0.812.

## 4. What each component did

Each change was measured against a control differing only in that change.

| component | effect | verdict |
|---|---|---|
| 16 epochs instead of 8 | single model +0.047 (10 seeds, 9/10 up) | **keep** |
| full question in the input | −0.095 at 8 epochs; single model +0.022 at 16 (7/10 up) | **keep, with 16 epochs** |
| logit adjustment (1 parameter) | +0.109 on the baseline ensemble; −0.004 on the best one | keep — it is what makes the baseline competitive |
| seed ensemble | +0.004 (baseline) to +0.025 (best model) before the rule | keep |
| per-class thresholds (9 parameters) | 0.394 vs 0.438 | drop — overfits 308 items |
| Balanced Softmax + focal loss | over-shifts to rare classes (`Explicit` F1 0.68 → 0.32) | drop — the rule does this for free |
| hierarchical output head, p(level) × p(leaf \| level) | system −0.068; Subtask 1 steadier, not better | drop |
| hard hierarchical routing, post hoc | 0.369 vs 0.365 | no effect |
| second encoder re-ranking the top 3 | 0.354 vs 0.365 | drop — the slot an LLM would fill |
| mixing both inputs in one ensemble | 0.403 vs 0.405 (tested on fresh seeds) | no effect |
| **train on all of train, last epoch** (E12a) | single model S2 +0.029 (4/5 seeds up), S1 −0.017; 5-seed system 0.489 | **keep** — the one E12 variant that helps |
| hierarchy: Non-Reply gate + branch specialist encoders (E12b) | gate: S2 −0.059 with the rule; specialists: +0.018 on 3 seeds, level on 5 | drop |
| boundary experts for the three most-confused pairs (E12c) | S2 +0.002, S1 −0.004 | drop |
| model soup: averaged weights of 10 models (E12d) | S2 0.267 vs 0.384 for one model | drop — the seeds do not share a basin |

*The last five rows were measured with 5 seeds, before E11 showed how much 5-seed
system numbers move; their verdicts rest on effects larger than that noise or on
per-class behaviour, but they are the less certain part of this table.*

## 5. What we learned

1. **The metric rewards landing on any acceptable label.** A prediction counts if
   any annotator chose it, so macro-F1 depends on how often the model is acceptable
   and how many classes it ever names — which is why a one-number rule that shifts
   predictions towards rare classes is so strong (`01_scorer_geometry.md`).
2. **The hard part is attribution.** 69% of training rows share their answer with
   another sub-question, mostly with a different label; the full question shows the
   model what else the answer responds to — but it needs about twice the training
   to use it.
3. **Undertraining can look like "the idea does not work".** At 8 epochs the
   full-question model looked worse (E8b, 0.343); a loss change was first blamed
   (E8). Controls that change one thing at a time found the real cause.
4. **Better training and the decision rule fix the same thing.** Both remove the
   pull towards frequent classes; on Subtask 2 either is enough, and together they
   add nothing.
5. **The dev set is too small to rank systems from one set of seeds.** 5-seed
   system scores moved by up to 0.08 between seed sets; E10's 0.478 did not
   replicate (0.434 on new seeds). Per-seed paired comparisons were the reliable
   instrument.

## 6. How the work was done

- One change per experiment, each against its own control; paired comparisons
  over seeds; 10 seeds for every claim in §2–3.
- Hypotheses, predictions and — for the final system — the selection rule written
  in the log before each run, and results reported against them, including those
  that failed (E11: two held, one failed, one held only in part).
- Negative results kept and explained; two earlier conclusions (E8's cause, E10's
  0.478) corrected in place when later evidence overturned them.
- Everything resumable (tmux, epoch checkpoints), logged to Weights & Biases, with
  checkpoints on the Hugging Face Hub.

## 7. What E12 added, and what is next

E12 tested four variants chosen from the error analysis, each screened on 3 seeds
paired with the best model and extended to 5 only if it cleared a bar set in
advance:

- **Training on all of train, keeping the last epoch, helps Subtask 2** (+0.029 per
  model, 4/5 seeds; 5-seed system 0.489), at a small cost on Subtask 1.
- **The hierarchy does not help as an encoder architecture.** A dedicated
  Non-Reply gate memorises its split and routes too confidently; branch specialists
  passed 3-seed screening and were level on 5. With E5 and E9, that is three
  encoder hierarchies, none better than the flat full-question model. The top
  systems' hierarchies were LLM reasoning stages, which this track deferred.
- **Boundary experts and model soups do not help** (reasons in the log).

Next, within the encoder track: the full-data model on 10 seeds, the standard for
any system claim here (E11). Beyond it: the deferred LLM step, for which the
encoder's top-3 candidates (right label inside 90% of the time) are ready.
