# Viva script: Knowledge over Structure (15 minutes, 20 slides)

Speaking pace is about 140 words a minute. The times in brackets are the budget for each slide, and the running total is in the last column of the plan.

| Slide | Topic | Budget | Ends at |
|---|---|---|---|
| 1 | Title | 0:15 | 0:15 |
| 2 | Three real examples | 0:50 | 1:05 |
| 3 | The task | 0:45 | 1:50 |
| 4 | Baselines | 0:50 | 2:40 |
| 5 | How the work progressed | 0:45 | 3:25 |
| 6 | The classification pipeline | 0:50 | 4:15 |
| 7 | Phase 1 encoder | 0:45 | 5:00 |
| 8 | Encoder experiments 1/3 | 0:55 | 5:55 |
| 9 | Encoder experiments 2/3 | 0:45 | 6:40 |
| 10 | Encoder experiments 3/3 | 0:45 | 7:25 |
| 11 | Where the encoder fails | 0:50 | 8:15 |
| 12 | Change only the backbone | 0:50 | 9:05 |
| 13 | The gain is broad | 0:30 | 9:35 |
| 14 | Ranks better than it decides | 0:35 | 10:10 |
| 15 | The final system | 0:50 | 11:00 |
| 16 | The four modules | 0:55 | 11:55 |
| 17 | The cascade on dev | 0:45 | 12:40 |
| 18 | Cost and distillation | 1:00 | 13:40 |
| 19 | Final results | 0:35 | 14:15 |
| 20 | Conclusions and next steps | 0:45 | 15:00 |

---

## 1. Title (0:15)

Good morning. We are Team Nier_ANLP. Our project is SemEval-2026 Task 6, CLARITY: classifying how clearly politicians answer questions in interviews. The title, "Knowledge over Structure", is our main finding, and we will show you how we got there.

## 2. What are we classifying? (0:50)

Here are three real items from the development set.

On the left, the president is asked whether he stands by a figure of 30,000 and says "I stand by the figure". That is a **Clear Reply**, and the fine-grained label is **Explicit**.

In the middle, he is asked about the risk of a wider war and answers "I thought you were going to ask me about the pig." He ignores the question and changes the subject. That is **Ambivalent**, and the label is **Dodging**.

On the right, he says "I'm not going to talk about ongoing intelligence matters". He openly refuses, which is a **Clear Non-Reply**, labelled **Declining to answer**.

One detail matters for modelling. The "pig" answer is attached to all three sub-questions of that journalist's turn. So the model has to work out which part of one long answer responds to which sub-question.

## 3. The task (0:45)

Each item has three parts: a sub-question, the journalist's full question, and the answer.

- **Subtask 2** is the fine-grained label, one of nine evasion types.
- **Subtask 1** is the three-way clarity level, which follows directly from the nine types through this taxonomy.

The data:
- 3,448 training rows.
- A dev set of 308 items labelled by three annotators.
- A test set of 237 items whose labels were never released, so every number we show is on dev.

The metric is macro-F1 over the nine classes, and a prediction counts as correct if **any** annotator gave that label.

The task is hard for two reasons:
- 69% of rows share their answer with another sub-question.
- Humans disagree: one annotator scored against the other two reaches only **0.684**. That is effectively the ceiling.

## 4. Baselines: what the other teams did (0:50)

This is the official leaderboard from the task overview paper.

- **The organisers' baseline** is a fine-tuned Llama-70B, at 0.82 on Subtask 1 and 0.57 on Subtask 2.
- **The winner, TeleAI,** used DeepSeek-V3 with three confidence-routed chain-of-thought stages, reaching 0.68 on Subtask 2.
- **Second place, ChulaNLP,** fed the top five labels of a fine-tuned DeBERTa into a few-shot Kimi-K2.
- **CLaC and others** used prompted LLMs.
- **Encoder-based teams** like PFW, who averaged 50 DeBERTa models, stopped near 0.50 on Subtask 2.

Two patterns stand out. Only 7 of 33 teams beat the Subtask 2 baseline. And the top systems are LLM pipelines, while encoders level off around 0.50.

These are test scores, while ours are on dev, so the comparison is approximate. The question this left us with was: what actually limits a fine-tuned model here?

## 5. How the work progressed (0:45)

This slide is the map of the talk.

**Top row: how every one of our systems decides.** We train 10 seeds. Each seed gives a probability for each of the nine classes. We average them into p-bar. Then we divide by the class prior raised to a power tau, which is called logit adjustment, so rare classes get a fair chance. Tau is the only parameter fitted on dev, and it is fitted by nested cross-validation. We use ten paired seeds because a single model varies by about 0.04.

**Bottom row: the phases.**
- **Step 1:** a naive DeBERTa encoder.
- **Step 2:** we added the full question and trained longer.
- **Two approaches that failed:** adding structure (Approach A) and changing the loss or decision rule (Approach B). Neither gave a gain.
- **Step 3:** we changed the backbone to an 8-billion-parameter LLM, and that gave the jump.
- **Step 4:** we built a cascade around it, which reaches 0.669.

## 6. The classification pipeline (0:50)

This is the exact pipeline from our report, and both backbones share it.

1. **Input:** the sub-question, the full question and the answer, up to 1,024 tokens.
2. **Backbone:** either DeBERTa-v3-large, fully fine-tuned and reading the CLS token, or Qwen3-8B-Base with LoRA, reading the last token.
3. **Head:** a new linear layer gives nine logits and a softmax distribution per seed.
4. **System:** we average the 10 seeds and apply logit adjustment.
5. **Output:** the Subtask 2 label, which fixes Subtask 1.

Below the figure are the training and decision settings:
- Each seed trains on 3,103 rows with cross-entropy.
- The epoch is chosen on a 345-row slice held out from train.
- Tau is the only number fitted on dev.

Keeping the pipeline fixed is what lets us change one thing at a time.

## 7. Phase 1: the encoder (0:45)

- **Why:** our first encoder trailed ChulaNLP's 0.46, and since 69% of rows share an answer, we suspected the model needed the full question.
- **Result:** training for 16 epochs and adding the full question raised Subtask 2 from **0.315 to 0.384 per model**. That is +0.069, with 9 of 10 seeds improving. The 10-seed system reached 0.405 on Subtask 2 and 0.648 on Subtask 1.
- **Takeaway:** at 8 epochs the full question actually looked harmful, 0.282 against 0.337. But its training loss was still 1.3 to 1.5, so the model simply had not finished fitting. Since then we report the training fit next to every score.

In the plot, each grey line is one seed, paired across configurations, and the squares are the means.

## 8. Encoder experiments, 1 of 3 (0:55)

The figure shows every encoder-side change we tried, measured against its own control. The dashed line is our +0.015 screening bar. Almost everything is at or below zero.

- **E4, per-class decision weights.** *Why:* plain argmax ignores that macro-F1 weights every class equally. *Result:* logit adjustment lifted the ensemble from 0.365 to 0.438, but nine separate class weights reached only 0.394. They overfit 308 items, with an in-fold gap of +0.098. *Takeaway:* one scalar beats nine weights, and every system since uses it.
- **E5, hard routing through the clarity level.** *Why:* the organisers said using the hierarchy helps. *Result:* 0.369 against 0.365, which is no reliable effect. *Takeaway:* the model's own top 3 contains the answer 89.6% of the time, more than the taxonomy branch's 81.2%.
- **E6, a definition re-ranker.** *Why:* if the answer is in the top 3 so often, a perfect chooser would score 0.745. *Result:* a second encoder re-ranking with label definitions scored 0.354 against 0.365. It fixed 25 errors but broke 37. *Takeaway:* understanding label meaning needs knowledge that 3,400 examples cannot give. This is exactly where an LLM belongs later.

## 9. Encoder experiments, 2 of 3 (0:45)

- **E8, Balanced Softmax plus focal loss.** *Why:* rare classes were under-predicted. *Result:* the system dropped to 0.379, and Explicit F1 fell by 0.36. *Takeaway:* the loss over-corrects towards rare classes, and logit adjustment already does that job for free.
- **E9, a factorised hierarchical head.** *Why:* post-hoc routing failed, so we trained the hierarchy into the model. *Result:* Subtask 2 fell by 0.068, and Subtask 1 moved only within noise. *Takeaway:* a flat softmax already scores all nine labels jointly.
- **E11, a mixed-input ensemble.** *Why:* an exploratory mix scored 0.498 on dev. *Result:* re-run on fresh seeds 5 to 9, it scored 0.403 against 0.405. *Takeaway:* ideas found on dev must be replicated on fresh seeds, and since then we pre-register our predictions.

## 10. Encoder experiments, 3 of 3 (0:45)

- **E12d, model soups.** *Why:* we wanted the ensemble's gain in one model. *Result:* averaging the weights of 10 models gave 0.267, far below 0.384. *Takeaway:* our seeds do not share a loss basin, so you should average outputs, not weights.
- **E12b and c, specialists and pair experts.** *Why:* Non-Reply recall was 0.57, and three confused pairs held most errors. *Result:*
  - the Non-Reply gate lost 0.059;
  - the specialists passed the 3-seed screen at +0.018 but fell to −0.009 at 5 seeds;
  - the pair experts gave only +0.002.

  *Takeaway:* extra encoders re-decide what the flat model already decides from the same input.
- **E12a, all of train.** This was the only one that helped Subtask 2 (+0.029 per model), but it cost 0.017 on Subtask 1 and was measured on only 5 seeds, so we could not rank it.

The conclusion: nothing trained on the same 3,448 rows moved the system. So we changed the backbone itself.

## 11. Where the encoder fails (0:50)

**How to read it:**
- Each row is the **majority gold label**, the label at least two of the three annotators chose. When all three disagree, which happens on 33 items, the item goes to the "no majority" row.
- Each column is the encoder's prediction, and the cells are counts. The bold diagonal is correct for 121 of the 275 items that have a majority.
- Scoring itself never uses the majority. A prediction counts if any annotator gave it, so some off-diagonal cells are still scored correct.

**What it tells us:**
- The errors sit inside the Ambivalent group. Implicit is predicted as Explicit 20 times, and General is spread across five other labels, right only 8 times.
- Partial/half-answer is never predicted.

So the encoder struggles with **how committed** an answer is, which is the same boundary humans disagree on. That is a judgement that needs world knowledge, and it led to our hypothesis.

## 12. Knowledge over structure: change only the backbone (0:50)

This is the central experiment. We kept everything identical (the same rows, input, epoch-selection rule and decision rule) and only replaced DeBERTa with Qwen3-8B-Base plus LoRA.

- **Panel (a):** per model, Subtask 2 rose from 0.384 to **0.476**, and **all ten seeds improved**. That single change gained more than all our encoder work combined.
- **Panel (b):** as a system, Qwen reaches 0.549, averaged over 10 cross-validation splits. That already passes TeleAI's fine-tuned 7B model at 0.495 and ChulaNLP's hybrid at 0.520.

## 13. The gain is broad (0:30)

Where did the gain come from? Seven of nine classes improved, most in Claims ignorance (+0.403) and Declining (+0.220).

Clarification fell, but it has only 4 dev items, so its F1 swings a lot.

We had predicted the gain would sit on the Explicit/Implicit boundary, and that prediction was wrong. The gain is broad, which fits a model that simply knows more.

## 14. The model ranks better than it decides (0:35)

This observation motivated the final system.

The correct label is ranked first for 60.4% of items, but it is in the **top 3 for 93.1%**. And confidence tracks correctness: in the least confident fifth, 40% of predictions are acceptable, and in the most confident fifth, 87%.

So the model usually has the answer among a few candidates and knows when it is unsure. That suggests sending only the uncertain items to a stronger model, and only asking it to choose among those few candidates.

## 15. The final system: a confidence-routed cascade (0:50)

Now that the phases are clear, here is the final system.

1. **M1:** the Qwen3-8B classifier, trained 12 epochs with 10 seeds. It outputs its top 4 labels as a candidate set C(x) and an uncertainty u(x), which is one minus its top adjusted probability.
2. **M2:** the router. If uncertainty is low, we keep M1's answer. If it is high, the item goes to M3.
3. **M3:** a local 27-billion-parameter LLM. It picks one letter from the four candidates, and its distribution q is fused with M1's in log space before the same prior correction.
4. **M4**, the dashed box: distillation. M3 relabels the training set, and one student model replaces the ten seeds.

With delta = 0.3, the threshold set on the train slice, only **0.36 LLM calls per item** are needed. All choices are made on train or by nested cross-validation, and dev is only for reporting.

## 16. The four modules in brief (0:55)

Each module has one reason and one number.

- **M1:** Qwen ranks well but decides poorly, so we use it as a candidate generator. Alone it scores 0.598 on Subtask 2, with no LLM call.
- **M2:** confidence tracks correctness, so only the least confident items go to the LLM. At delta 0.3 that is 0.36 calls per item. Sending every item is actually worse, 0.609 against 0.660, so the router matters.
- **M3:** a larger model knows more but is costly, so it only picks among four candidates. It lifts Subtask 2 to **0.669**, +0.067 over M1, fixing 33 items and breaking 9.
- **M4:** the ten seeds are 93% of the compute, so we distil them into one student. With M3, the student reaches 0.608 at 4.7 times less compute than M1 alone.

## 17. M1 + M2 + M3: the cascade on dev (0:45)

This table sweeps delta, the share of items deferred to the LLM.

- At delta 0, M1 alone scores 0.598.
- Accuracy rises to **0.669 at delta 0.3**, then falls as we defer more: 0.628 at 0.5, and 0.636 when every item goes to the LLM.

So the LLM is better than M1 on the uncertain items but worse on the confident ones. The gain at 0.3 is +0.067, with a 95% interval from +0.009 to +0.124.

We screened the setup on training data only. A local 27B model beat DeepSeek V4 Flash and Gemma 4 31B, and the whole API screening cost 26 cents.

## 18. Cost: one distilled student replaces ten seeds (1:00)

**First, the measured cost.** On one RTX PRO 6000, an M1 seed costs 0.0522 GPU-seconds per item, and an M3 call costs 0.103.
- Ten seeds: 10 × 0.0522 = **0.522**.
- The accuracy point adds 0.357 calls × 0.103, giving **0.558**. So the ten seeds are 93% of the cost.
- The cost point uses one student plus 0.574 calls × 0.103, giving **0.111**.
- 0.522 divided by 0.111 is **4.7 times cheaper** than the ten-seed M1.

**Now the student's training target:** t(c) = 0.5 × 1[c = gold] + 0.5 × q(c).
- **q** is M3's softmax over the four candidate letters, read from a single forward pass, and zero outside the candidate set.
- **Why gold:** it keeps the student anchored to what the annotators said, especially where M3 is wrong.
- **Why q:** a one-hot label says only "this one is right and every other one is equally wrong". q says *which* other labels are plausible, and that is the LLM's knowledge we want the student to absorb. This is standard knowledge distillation.
- **Where 0.5 came from:** we fixed it in advance, and did not tune it.

**Result:**
- The student alone scores 0.573 per model, up 0.048, with 9 of 10 seeds improving.
- With M3 at delta 0.5 it reaches **0.608 on Subtask 2 and 0.790 on Subtask 1 at 0.111 GPU-seconds**. That beats the ten-seed M1 at less than a quarter of its cost.

## 19. Final results (0:35)

Here is the whole path in one table:
- DeBERTa: 0.405.
- Qwen at the mid-evaluation: 0.543.
- M1: 0.598.
- **Accuracy point: 0.669 on Subtask 2 and 0.798 on Subtask 1.**
- **Cost point: 0.608 for 0.111 GPU-seconds.**

The accuracy point is above the winning TeleAI system on Subtask 2 (0.617) with about a fifth of its LLM calls, 0.36 against 1.94, and only one output token per call. The cost point still beats the ten-seed M1 while using 4.7 times less compute.

## 20. Conclusions and what comes next (0:45)

**What we learned:**
- Knowledge, not structure: changing only the backbone gained more than every encoder-side change combined.
- A 27B model deciding only the uncertain items, among four candidates, beats the winning multi-call pipeline on Subtask 2.
- Distillation turns that accuracy into one cheap model.

**Limitations:**
- Dev has only 308 items, so the interval on the M3 gain is wide.
- Partial/half-answer and General are still our weakest classes.

**Next:**
- Let the candidate set grow on hard items instead of a fixed top 4.
- Sharpen the teacher by tuning the gold/q mix and a temperature.
- Close the small Subtask 1 gap (0.798 against 0.812).
- Add targeted examples for the weak classes.

In one line: **a small model that knows when to ask a bigger one.** Thank you, we are happy to take questions.

---

## Likely viva questions, with short answers

- **Why macro-F1 with "any annotator" scoring?** That is the official metric. A prediction is correct if any annotator gave it, because annotators disagree (one annotator scores only 0.684 against the other two).
- **What is logit adjustment, and why only one parameter?** It divides the averaged probability by the class prior raised to tau, which boosts rare classes. Nine per-class weights overfit 308 items (E4), while one scalar fitted by nested CV does not.
- **What is nested cross-validation here?** Tau (and lambda for M3) is chosen on 4/5 of dev and scored on the held-out fifth, rotated 5 ways. So no item is scored with a parameter fitted on itself. "10 splits" means we repeat this with 10 different random fold assignments.
- **Why 10 paired seeds?** One model's Subtask 2 varies by about 0.04, as much as most effects. Paired seeds share head initialisation and data order across configurations, so the difference isolates the change.
- **Is the Qwen gain just more parameters or more training?** It is everything a larger pretrained model brings. The learning rate and epochs had to change with the backbone, so "knowledge" is shorthand for that whole package. We cannot rule out that the model saw the public transcripts in pretraining.
- **Why top 4 candidates?** On cross-fitted training rows the gold label is in M1's top 4 for 81% of items, and a small candidate set keeps the LLM prompt and output short (1 output token).
- **Why does deferring every item hurt?** On confident items M1 is already right more often than the LLM. The LLM alone within the top 4 scores 0.603, below the fused system.
- **Is 0.669 better than TeleAI on test?** We cannot say. Test labels are unreleased, so all our numbers are on dev, and the leaderboard is on test.
- **Why is the student weaker than M1 × 10 + M3?** A single model has no seed diversity, and distillation makes students agree with each other, so ensembling students adds little. The student trades about 0.06 of accuracy for 5 times less compute than the accuracy point.
- **How were GPU-seconds measured?** On the lab RTX PRO 6000, alone, timing the forward pass over the 308 dev items in batches of 16 after a warm-up. Model loading is excluded. M3's cost is the mean per-item time from the vLLM dev runs. System costs are computed as seeds × 0.0522 + calls × 0.103.
