# Speaker notes: final presentation (22 slides)

**Slot:** 30 minutes, made up of about 20 minutes of talk and 10 of questions. Aim to finish the talk by **18:36**.

The script is about 2,120 spoken words: about 16 minutes at 135 words a minute,
plus short pauses on the figures and the hand-overs. Read it aloud once with a timer, and use the running clock to see
where you drift.

**Four speakers.** Hand over at the stage boundaries, which match the stage name in the footline of each slide.

| Speaker | Slides | Part | Ends at |
|---|---|---|---|
| A | 1–6 | Opening: the task, three items, the hook, the shared task, the timeline | 4:36 |
| B | 7–12 | Stages 1 and 2: the classifier and the cascade | 10:24 |
| C | 13–18 | Stages 3 and 4: what survives, and what the instrument measures | 15:32 |
| D | 19–22 | Stage 5, limitations, the whole project, thank you | 18:36 |

**The hand-off rule.** Stages 1 and 2 end with a red "What it raised" line. The speaker who reads it hands the
question to the next slide, which answers it. That is what makes four voices sound like one story.

**Three things never to say.** "We beat the winner" (we are on par on S2 and lower on S1). "We are close to human"
(0.684 is a two-annotator score; ours is three-annotator). "Politicians evade strategically" (the role effect is
exploratory, on LLMs, and cannot separate answerability from strategy).

The running clock is in brackets after each heading.

---

## 1. Title [0:22]

> Good afternoon. We are Team Nier_ANLP. Our project started as a SemEval shared task on classifying how clearly
> politicians answer questions, and it ended as a study of what evasion actually is.

## 2. The task [1:23]

> Each item is a sub-question, the journalist's full question, and the President's whole answer. Subtask 2 asks for
> one of nine evasion types; Subtask 1 is the three-way clarity level, and it follows from the type, so we predict the
> type and map it up.
>
> Two facts before any number. The test labels were never released, so every score today is on the 308-item dev set.
> And the metric is macro-F1 where a prediction counts if any annotator gave it, with all nine classes weighted equally.
>
> It is hard because annotators disagree, one against the other two scores 0.684, and because 69% of training answers
> serve more than one sub-question. That second fact is the thread of this talk.

## 3. What are we classifying? Three real dev items [2:14]

> Three real items, one per clarity level, each with all three annotators agreeing. "Do you stand by your figure?" "I
> stand by the figure": Explicit, a clear reply. "Do you see a risk of a wider war?" "I thought you were going to ask me
> about the pig": Dodging. And "are CIA operatives in Pakistan?" "I'm not going to talk about ongoing intelligence
> matters": Declining, a clear non-reply.
>
> Notice the pig answer is attached to all three sub-questions of its turn. So the model has to work out which part of
> an answer responds to which sub-question.

## 4. One answer, two sub-questions, two labels [2:53]

> Here is the extreme case. A journalist asks two things in one breath: is this a make-or-break issue for you, and would
> you campaign for Mike Fitzpatrick? The answer is three words: "I already have."
>
> For "would you campaign for him", all three annotators said Explicit. For "is this make-or-break", all three said
> Dodging. Same words, opposite labels. Hold on to this answer: we come back to it in Stage 4.

## 5. The shared task [3:43]

> The winner, TeleAI, called DeepSeek-V3 in three chain-of-thought stages, about two LLM calls per item. Fine-tuned
> encoders stalled near 0.50. The table is test; TeleAI's dev score, 0.617, is the number we compare with.
>
> That left us two questions. A: why do fine-tuned classifiers stop where they do, and what gets past it? B: once we
> have a good classifier of evasion, what can it tell us about evasion itself? And one rule: every experiment's
> prediction and decision rule was written down before it ran, one change at a time, on ten paired seeds.

## 6. The timeline [4:36]

> This is everything we did, in order. Each column is a stage, and each stage asks one question raised by the one
> before.
>
> Stage 1: on DeBERTa, structural ideas gave no gain; changing only the backbone to Qwen3-8B gained 0.092. Stage 2: that model ranks better than it decides, so a 27B LLM picks among its top four, reaching
> 0.669. Stage 3: we attacked that number and froze what survived, 0.650. Stage 4: we used the frozen system to measure
> evasion. Stage 5: we tried to make it cheaper. Stages 1 to 3 answer question A; 4 and 5 answer question B.

*[Hand over to Speaker B.]*

## 7. Stage 1: one pipeline, two backbones [5:26]

> Stage 1 asks what limits a fine-tuned classifier. The pipeline is the same for both backbones: the three texts go into
> a backbone, a new linear head gives nine probabilities, ten seeds are averaged, and a logit adjustment makes the
> decision.
>
> The logit adjustment divides each probability by the class's training share to a power tau, so rare classes, which
> macro-F1 weights equally, actually get predicted. One scalar took the system from 0.365 to 0.438. Nine free per-class
> weights reached only 0.394; they overfit 308 items. So every system uses the one scalar.

## 8. Structure failed; a stronger backbone did not [6:50]

> Three panes. On the left, the gain of every change we tried on DeBERTa, each against a named control. Longer training
> and the full question improved it. Hierarchies, rebalanced losses, experts, a definition re-ranker and model soups did
> not. And it was not label noise: 53 errors were on items all three annotators agreed on.
>
> The middle and right panes are Subtask 2 and Subtask 1, one line per seed: 16 epochs plus the full question take
> Subtask 2 from 0.315 to 0.384, nine of ten seeds up, and Subtask 1 from 0.576 to 0.614.
>
> Then we changed only the backbone, to Qwen3-8B with LoRA: same rows, input, loss and selection. Plus 0.092 per model,
> on all ten seeds, the dark bar at the bottom left, more than all the encoder changes together. The gain was broad,
> largest on Claims ignorance, not only on the boundary we had predicted.
>
> What it raised: the 8B model ranks the right label far better than it picks it. Can someone else pick for it?

## 9. The classifier ranks better than it decides [7:38]

> M1 is Qwen3-8B at 12 epochs and ten seeds: 0.598. An acceptable label is its first choice 60% of the time, but in its
> top three 93% of the time. And it knows when it is unsure: correctness rises from 40% to 87% with confidence.
>
> Two registered checks on the least sure 30%. An oracle choosing within the top three would reach 0.727. But a random
> choice there drops to 0.587, below M1. So M1's first choice carries information: a second opinion has to be fused
> with it, not replace it.

## 10. The cascade [8:32]

> That gives the cascade. M1 proposes four candidates and an uncertainty. M2, the router, sends only the least confident
> items onward: 30% of the training slice, which is 36% of dev. There, M3, a 27-billion-parameter LLM run locally,
> reads the label definitions and the item and answers with one letter, and that choice is fused with M1. Optionally,
> M4 distils the whole thing into one student model.
>
> Two of our design assumptions were refuted on held-out training rows: retrieved few-shot examples hurt the 27B, and a
> 304-billion-parameter model did worse than our local one. So: a short prompt, a local model, one output token.

## 11. Deferral rate against cost [9:34]

> This plot is score against GPU time per item. The navy line is the ten-seed M1 plus the 27B at growing deferral rates.
> Deferring 30% reaches 0.669, up 0.067 over M1, with 33 items fixed and 9 broken. We only saw this after finding and
> fixing a bug in our own fusion code, and it is in the log. Deferring everything is worse: where M1 is confident, the
> LLM is the weaker reader.
>
> The other two lines are cheaper first stages. One distilled student plus the 27B matches the ten-seed M1 at a quarter
> of its GPU time; ten students add almost nothing.
>
> What it raised: was 0.669 luck? It rests on one ensemble and parameters fitted on 308 items.

## 12. Controls on the cascade [10:24]

> Three controls. On 3,448 cross-fitted training rows the cascade still gains, plus 0.014: smaller than on dev, where its
> parameters were fitted, but real. Left panel: no random routing of the same size comes near confidence routing, 0.669
> against 0.604. And the 27B alone over all nine labels scores 0.591, below M1, so fusion is needed.
>
> Right panel: where M1 is confident, the 27B fixes about as many items as it breaks; it helps only where M1 is unsure.
> A learned router that tried to predict fixes lost to plain confidence at all four budgets.

*[Hand over to Speaker C.]*

## 13. We attacked our own result [11:09]

> Stage 3 goes through every remaining doubt with one registered test each. The first three rows you have just seen.
> Two results changed our claims. A student taught by M1 itself did as well as one taught by the 27B, so distillation is
> a soft-target effect, not the LLM's knowledge. And the 27B prefers option A: 40% in one order, a quarter in others, a
> known bias. Two orders of the same options agree on only 72% of items, so we average four orders.

## 14. The frozen system, against the winner [11:55]

> We froze every setting and re-scored once on one serving setup: 0.650 on Subtask 2 and 0.788 on Subtask 1. Against
> the winner that is higher on Subtask 2 and lower on Subtask 1, with three quarters of their LLM calls and a fifth of
> their input tokens. The distilled version needs a quarter of M1's GPU time.
>
> The task's system papers already used every part we built; none tested whether such systems read the question. So
> from here on, this frozen cascade is our instrument.

## 15. Evasion is a relation [12:43]

> Back to "I already have". Was it a one-off? On the left: two sub-questions that share one answer get different labels
> 72% of the time, nearly as often as two random items, and far more than two annotators disagree on the same item,
> 43%. The label moves because the question moves.
>
> On the right: is evasion just answering off-topic? No. Partial, Implicit, Deflection and General answers fit their own
> question as well as explicit replies. Relevance is not answerhood. So "I already have" is the rule, not the exception.

## 16. Validity check [13:26]

> Before using the instrument on anything new, we checked it actually reads the question. Keep each answer, give it a
> question from another interview. We predicted Clear Replies would at least halve. They went from 25% to 1%, and 62 of
> 78 clear replies became Dodging.
>
> The surprise is the reverse: remove the question and Clear Replies rise to 64%. If nothing was asked, nothing can be
> evaded. And "I already have" under a question about Saddam Hussein: Dodging.

## 17. Question or speaker? [14:19]

> If evasion lives in the pair, is it the question that is hard to answer, or the speaker who chooses not to? We put the
> Presidents' 308 questions to two open LLMs in four roles: N, a neutral analyst with no stake; P, the President in that
> interview; F, the President pressed with "you haven't answered the question"; B, the President with the real answer
> as private notes. The frozen instrument labels all 2,464 answers.
>
> We registered: if the question drives evasion, the neutral analyst evades where the Presidents did. If the stake drives
> it, playing the President raises the evasion rate.

## 18. The role changes where models evade [15:32]

> Neither of those was supported. The neutral analyst's evasions show no clear link to the Presidents': odds ratio 1.18,
> below the 1.5 we predicted. And playing the President does not raise the rate. Pursuit works as predicted: one
> follow-up turns two thirds of evasions into replies.
>
> But look at panel b, the open circles. Where the President answered directly, a neutral analyst still evades 52%; in
> the President's role, only 34%. In the role, the models answer directly what the Presidents answered directly. Odds
> ratio 2.74, in each model separately. We did not register this, so it is exploratory. B is greyed out because it sees
> the real answer, so agreeing with the President is expected.
>
> And what it does not show: anything about real politicians' intent. The instrument was human-checked on LLM text, and
> it is more confident there than on the Presidents' answers.

*[Hand over to Speaker D.]*

## 19. Can the full instrument run cheaper? [16:28]

> Stage 5 asks whether the same instrument can run cheaper. We ran M1 in FP8 and NF4 against 16-bit on the same GPU
> type. FP8 was not faster, 0.83 times the throughput, because activations are quantized in separate, unfused
> operations. NF4 saves a third of the memory but is slower.
>
> Panel a: M1 alone loses 0.045 in FP8, but the full instrument loses only 0.003. Panel b shows why: M1's flipped
> decisions all lose, while the full system's go both ways. So it is not a cost lever with these kernels, but a cheaper
> first stage is safe as long as its errors stay where the adjudicator looks.

## 20. Limitations and the next module [17:24]

> Every limitation comes with what we did and what comes next. The two that matter most: only dev, so the frozen test
> predictions are ready to score when labels are released; and the role finding needs more model families.
>
> The role study also points to the next module, M5. Every module reads the answer, yet the question carries signal.
> When neither P-role model evades, the President evaded 61% of the time; when both do, 88%. As a prior, it moves M1's
> AUROC on its least sure items from 0.722 to 0.746. Small, exploratory, and the next registered test.

## 21. The whole project in one picture [18:19]

> This is the whole project in one picture. In the middle, the cascade that became our instrument: M1 proposes, M2
> routes, M3 decides the uncertain 36%, 0.650 at the bottom. On the left, how it was built: the backbone from Stage 1,
> training and calibration on the slice, nested cross-validation on dev, and the distilled student. On the right, what
> we used it for: labels follow the question, swapping the question destroys Clear Replies, the role study, FP8, and
> M5 as future work.
>
> Our thesis in one line: evasion is a relation between a question and an answer, and the frozen cascade measures it.

## 22. Thank you [18:36]

> Thank you. The code, the pre-registered log and every raw output are at these links. We are happy to take questions.
