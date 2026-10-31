# Speaker notes: final presentation

**Slot:** 30 minutes, made up of about 20 minutes of talk and 10 of questions. Aim to finish the talk at **19:40** so there is slack for speaker changes.

The script below is about 2,300 spoken words, roughly 17 minutes at 135 words a minute, and about 19 with pauses on the figures and the hand-overs. Read it aloud once with a timer, and use the running clock to see where you drift.

**Four speakers.** Hand over at the stage boundaries. These match the progress bar at the top of each slide.

| Speaker | Slides | Part | Target time |
|---|---|---|---|
| A | 1–5 | Opening: the task, examples, shared-task results, overview | 2:55 |
| B | 6–14 | Stage 1 and Stage 2: the encoder, the backbone change and the cascade | 5:55 |
| C | 15–24 | Stage 3 and Stage 4: what survives, and what the instrument measures | 6:55 |
| D | 25–31 | Stage 5 and close | 3:15 |
| | | **Total** | **19:40** |

**Who speaks when** is the team's choice. One rule: each speaker should deliver the last line of their own section, the "What it raised" line. The next speaker then opens their section with that question. This hand-off is what makes the talk sound like one story.

**Running clock:** the time at the end of each slide is shown in brackets after its heading.

---

## 1. Title [0:15]

> Good afternoon. We are Team Nier_ANLP. Our project started as a SemEval shared task on political question evasion. It ended as a study of what evasion actually is.

## 2. The task [0:50]

> Formally, each item is a sub-question, the journalist's full question, and the answer. Subtask 2 assigns one of nine evasion types. Subtask 1 is the three-way clarity level, and it follows directly from the type.
>
> One thing to know before any number. The test labels were never released. So every score in this talk is on the 308-item development set, and it comes with a seed spread or a bootstrap interval.
>
> The metric is macro-F1, and a prediction counts if any annotator gave it. The task is hard for two reasons. Annotators often disagree. And 69% of training answers serve more than one sub-question.

## 3. Three real dev items [1:35]

> Before the system, three real dev items. Each pairs a sub-question with the politician's answer.
>
> "Do you stand by your figure, 30,000?" The answer is "I stand by the figure". That is Explicit, a Clear Reply.
>
> "Do you see a risk of triggering a wider war?" The answer is "I thought you were going to ask me about the pig." That is Dodging: he ignores the question and changes the subject.
>
> Asked about CIA operatives in Pakistan, he says he is not going to talk about ongoing intelligence matters. That is Declining to answer, a Clear Non-Reply.
>
> All three annotators agree on each of these. Note that the "pig" answer is attached to all three sub-questions of its turn, so the model must work out which part of an answer responds to which sub-question.

## 4. Shared-task results [2:10]

> The winning system, TeleAI, called DeepSeek-V3 in three chain-of-thought stages. That is about two LLM calls per item. Fine-tuned encoders stalled near 0.50.
>
> That left us with two questions. A: why do fine-tuned classifiers stop where they do? And B: once we have a good classifier of evasion, what can it tell us about evasion itself?

## 5. Overview [2:55]

> The talk has five stages. Each stage asks one question, and its answer leads to the next.
>
> Stages 1 to 3 answer question A: we build an instrument. Stages 4 and 5 answer question B: we use it.
>
> From experiment 8 on, each plan, prediction and decision rule was written down before the run, and anything we did not register is marked exploratory. So on each result slide you will see what we predicted next to what we found. That includes the many predictions that failed. Those failures are what moved us forward.

*[Hand over to Speaker B.]*

## 6. The classification pipeline [3:25]

> The classifier has the same shape for both backbones. The input is the sub-question, the full question and the answer. A backbone reads it, a new linear head gives nine logits, and a softmax gives one probability per evasion type for each seed.
>
> A system is ten seeds averaged, followed by a logit adjustment. The nine-way type fixes the three-way clarity level through the taxonomy.

## 7. A controlled baseline and seed ensembling [3:55]

> Our first encoder trailed the other teams, and 69% of rows share an answer, so we gave it the full question and trained it longer. That raised a single model from 0.315 to 0.384, with nine of ten seeds up.
>
> The ten models differ only in their random seed. Averaging their softmax outputs reduces the variance between runs, and every claim in this talk compares paired seeds. The ten-seed system scores 0.405 on Subtask 2.
>
> One caution. At 8 epochs the gain was hidden, because the model was still under-trained. So we report fit next to every score.

## 8. The decision rule: logit adjustment [4:35]

> A plain argmax picks the class with the highest averaged probability. But macro-F1 weights all nine classes equally, and the training labels are dominated by Explicit and the Ambivalent types.
>
> Logit adjustment divides each class probability by the training prior raised to a power tau. That lowers the score of frequent classes, so rare ones are predicted more often. With tau zero it is the plain argmax. Tau is the only fitted parameter, and we choose it by nested cross-validation.
>
> We compared it with nine free per-class weights. Logit adjustment took the system from 0.365 to 0.438. Nine weights reached only 0.394, with an in-fold gain that did not transfer. So every system uses one scalar.

## 9. Structural changes and the decision rule [5:15]

> Question A first. Why do classifiers stall? We wrote down four explanations, and each predicts a different fix. We tested them on DeBERTa one change at a time, each against a named control.
>
> - Noisy labels? No. 53 of the errors were on items all three annotators agreed on.
> - The decision rule? It helps a little: a one-parameter logit adjustment adds 0.073. Longer training and the full journalist question help a little too.
> - Label meaning? No. A re-ranker given the label definitions did worse, even though the gold label was in the top three 90% of the time.
>
> On the right is every change we tried. Hierarchies, expert models, rebalanced losses and model soups all failed. Nothing trained on the same 3,448 rows moved the system. So we changed the model itself.

## 10. Where the DeBERTa ensemble fails [5:45]

> This is the confusion matrix of the ten-seed DeBERTa ensemble on dev. Rows are the majority gold label, columns are what the ensemble predicts.
>
> The errors sit inside Ambivalent. Implicit is predicted as Explicit 20 times, General is spread over five other labels, and Partial is never predicted. The encoder fails on how committed an answer is, which is also where annotators disagree with each other.
>
> If a stronger model knows more about these cases, we expect it to improve exactly these boundaries. So we change only the backbone.

## 11. A stronger backbone [6:35]

> One change: DeBERTa becomes Qwen3-8B with LoRA. The data, input, loss and selection rule stay the same, on ten paired seeds.
>
> From the confusion matrix on the previous slide, we expected a stronger model to improve the Explicit–Implicit–General boundaries. The gain was plus 0.092 per model, with all ten seeds up. That is more than all our encoder changes together, which gave 0.069 per model.
>
> But the location was not what we expected. The gain was broad, and largest on the non-reply types, which depend on recognising what kind of statement an answer is.
>
> We say "stronger backbone", not "knowledge". Size, pretraining data and the adapter all changed together.
>
> **What it raised:** this model ranks the right label far better than it picks it.

## 12. Ranking against choosing [7:00]

> An acceptable label is its first choice 60% of the time, but in its top three 93% of the time. And it knows when it is unsure: correctness goes from 40% in the least confident fifth to 87% in the most confident.
>
> So the errors are mostly a choice among a few candidates, made where the model is unsure. Let a larger model choose, but only there.

## 13. System design [7:55]

> That is the cascade.
>
> 1. M1, our classifier, proposes four candidate labels and a confidence.
> 2. M2 sends only the least confident items onward: the threshold is set for 30% on a training slice, which defers 36% of dev.
> 3. On those items, M3, a 27-billion-parameter LLM, picks one of the four candidates, and its choice is fused with M1.
> 4. Optionally, M4 distils all of this into one student model.
>
> Three registered checks set the design, and each surprised us:
>
> - Choosing at random among M1's top three *lowers* the score. So M1's first choice carries information, and the LLM must be fused with M1, not substituted for it.
> - Retrieved examples *hurt* the LLM, and a 304B model did worse than the 27B. So we use a short prompt, a local model and one output token.
> - Candidate sets defined by probability mass did not transfer to new rows; sets defined by rank did. So we use the top four.

## 14. Deferral rate against cost [8:50]

> On this plot, further left is cheaper and higher is better.
>
> - Deferring 30% reaches 0.669, up 0.067 over M1. We only saw this after finding and fixing a bug in our own fusion code. It is in the log.
> - Deferring everything is worse, 0.636. Where M1 is confident, the LLM is the weaker reader.
> - One distilled student, the grey star, matches the ten seeds at a quarter of their GPU time.
>
> **What it raised:** was 0.669 luck? It came from one ensemble, with parameters fitted on 308 items.

*[Hand over to Speaker C.]*

## 15. One answer, two sub-questions, two labels [9:35]

> Before the robustness tests, one more shared answer. A journalist asks the President two things in one breath. Is this a make-or-break issue for you? And, would you campaign for Mike Fitzpatrick?
>
> The answer is three words: "I already have."
>
> Three annotators labelled it once for each sub-question. For "would you campaign for him", all three said Explicit. For "is this make-or-break", all three said Dodging.
>
> Same answer, opposite labels. 69% of training rows share their answer with another sub-question, so this is common. Whether an answer evades depends on the question it is paired with. Stage 4 tests this, and we return to this answer at the end.

## 16. Controls on the cascade [10:25]

> Three controls on the cascade.
>
> On 3,448 cross-fitted training rows the cascade gains a small but clear 0.014. That is less than on dev, where its parameters were fitted.
>
> Left panel: no random choice of which items to defer comes close to confidence routing. The 27B alone over all nine labels scores 0.591, below M1, so fusion is needed too.
>
> Right panel: the 27B fixes about as many items as it breaks where M1 is confident, and helps only where M1 is unsure. A learned fix-or-break router lost to confidence at all four budgets.

## 17. The 27B prioritises option A [11:00]

> One more check on the 27B. We show it four candidate labels as options A to D. In taxonomy order, slot A is usually Explicit, so preferring A and preferring Explicit cannot be told apart. Known work shows that LLMs choosing among lettered options prefer some positions, Zheng and colleagues at ICLR 2024. Ours prioritises option A: it picks A for 40% of items in taxonomy order, against about a quarter in the other orders.
>
> Two orders of the same four candidates give the same pick on only 72% of dev items. So we average four cyclic orders. That costs four calls per deferred item, and it moved the system by 0.011 at one split but only 0.002 over ten.
>
> From here on, the final system averages four orders.

## 18. The frozen system and the winner [11:40]

> With four orders averaged and every setting frozen, the system scores 0.650. We re-scored all records once on one serving setup, which is why this differs from the 0.669 of Stage 2.
>
> Against the winner, that is on par on Subtask 2 and lower on Subtask 1. It uses three quarters of their LLM calls and a fifth of their LLM input tokens.
>
> The task's system papers already used these parts: candidate sets, LLM choosers, routing, distillation. What none of them tested is whether such systems read the question. So we froze every setting and used the cascade as an instrument.
>
> **What it raised:** if a label is not a property of the answer alone, what is it a property of?

## 19. Labels of one answer across its sub-questions [12:30]

> Back to "I already have". Was that a one-off?
>
> Left panel: take two sub-questions that share one answer. Their labels differ 72% of the time. That is almost as often as two random items, at 83%, and far more than two annotators disagreeing on the same item, at 43%.
>
> Right panel: is evasion just answering off-topic? No. Partial, Implicit, Deflection and General answers fit their own question as well as explicit replies do. Relevance is not answerhood.
>
> So "I already have" was the rule, not the exception. Evasion is a relation between a question and an answer.

## 20. Validity check [13:15]

> Before using the instrument on new speakers, we checked that it actually reads the question.
>
> Pair each answer with a question from a different interview. Clear replies drop from 25% to 1%, as registered.
>
> The surprise is the opposite case. Remove the question entirely, and clear replies *rise* to 64%. If nothing was asked, nothing can be evaded.
>
> Here is our answer again, paired with a question about Saddam Hussein and Al Qaida. The instrument says Dodging.

## 21. Design: question or speaker [14:00]

> Now the real question. When an answer evades, is it the question that is hard to answer, or the speaker who chooses not to?
>
> We put the Presidents' 308 questions to two open LLMs, in four roles. N is a neutral analyst with no stake. P is the President, in that interview, on that date. F is the President pressed with "you haven't answered the question". B is the President with his real answer as private notes.
>
> The frozen instrument labels all 2,464 answers. Our registered predictions were these:
>
> - If the question drives evasion, the neutral analyst evades where the Presidents did.
> - If the stake drives it, playing the President raises the evasion rate.

## 22. Evasion rates by role [15:00]

> Both registered predictions failed.
>
> The neutral analyst does not evade where the Presidents did: the odds ratio is 1.18, with an interval spanning 1. And playing the President does not raise the rate. On the left, the neutral and President rates are the same.
>
> But look at the right panel. In the President's role, the same models evade *on the same questions the Presidents evaded*. The odds ratio is 2.74, and its interval is well above 1.
>
> This comparison was not registered, so we label it exploratory.
>
> Pursuit works: one follow-up turns two thirds of evasions into replies.

## 23. What the role result shows and does not show [15:45]

> So the role changes *where* the models evade, not *how often*.
>
> - Why we trust it: N and P are labelled by the same instrument on the same questions, so an instrument bias towards some questions cannot create the gap. And it holds for each model separately.
> - What it does not show: it is two model families, one of them the judge's own. The President prompt adds identity and context, not only stake, and we did not adjust this odds ratio for answer length. The instrument was not human-checked on LLM text. And it says nothing about real politicians' intent.
>
> **What it raised:** measuring this needs the whole answer and the fine-tuned ten-seed ensemble. There is no shortcut, and the details are in backup. Can it run cheaper?

## 24. What the evasion study tells us to act on [16:25]

> Everything so far comes from one pipeline: the classifier, the router and the 27B adjudicator, frozen as an instrument. Its results give three things to act on.
>
> First, labels follow the question, so the full question must always go in with the answer.
>
> Second, the question carries signal on its own. Models playing the President, who never saw the real answer, evade on the questions the Presidents evaded. No module uses that signal yet, and that is the next architectural element, a question-side prior we call M5.
>
> Third, there is no shortcut: the label needs the whole answer and the fine-tuned model. So instead of a smaller model, we try to make the full one cheaper. That is Stage 5.

*[Hand over to Speaker D.]*

## 25. Cost of reduced precision [17:05]

> Quantization is supposed to save memory and time. These are the measured costs of running M1 in 8-bit and 4-bit, each against 16-bit on the same type of GPU.
>
> FP8 ran on the card's native FP8 tensor cores, but it was not faster. Throughput fell from 43 to 36 items per second, 0.83 times, against a predicted 1.3 times or better. Batch-1 latency improved slightly, and peak memory barely changed, 0.96 times. As implemented, activations are quantized in separate operations around each matrix product. Published FP8 speed-ups rely on kernels that fuse these steps, which we did not use.
>
> NF4 saves memory: 0.67 times, from 16.8 to 11.3 gigabytes. That is short of the 0.4 times we predicted. And it is slower, 0.82 times throughput and 1.5 times the batch-1 latency, because the 4-bit weights are stored compactly but computed in bf16. Its speed is not like for like, since the 16-bit reference was split over two cards.
>
> Applied to the accuracy point, FP8's speed would raise the cost from 0.538 to 0.619 GPU-seconds per item, 15% more.

## 26. Lower precision: accuracy lost and flipped decisions [17:45]

> The ten seeds are three quarters of the instrument's GPU time, so we ran them in 8-bit and 4-bit.
>
> As implemented, 8-bit is not faster, and it changes 4% of labels. Both of those predictions failed.
>
> But the third held. 71% of the decisions FP8 flipped fall on items the 27B re-decides anyway, against 36% by chance. With the same frozen parameters the full instrument loses 0.003 S2, while the classifier alone loses 0.045. The flips that change reference-set membership all hurt the classifier: ten of ten. In the system they go both ways, five lose and seven gain. So the cascade turns a one-sided loss into noise.
>
> This is partly by design: flips happen near the margin, which is exactly what the router defers. The measurement tells you how much, and in which direction. A cheaper first stage is safe as long as its errors stay where the adjudicator looks.

## 27. Registered predictions [18:15]

> These are the registered predictions that steered the work. Navy held; red failed.
>
> Most failed, and each failure redirected us. The wrong location of the gain led to the cascade. Our attack on our own headline led to the instrument. The failed question-only hypothesis led to the role finding.
>
> Every registered prediction, held or not, is in the log in our repository.

## 28. Limitations [18:35]

> Each limitation comes with what we already did and what comes next. The most important two:
>
> - Dev only: the frozen test predictions are ready to score when labels are released.
> - The role finding needs a human check of the instrument on LLM text, and more model families, with the effect registered in advance.

## 29. Future work: a question-side evasion prior [19:10]

> One idea for future work. Every module reads the answer, but the role study suggests that the question carries evasion signal of its own.
>
> The proposal is a fifth module, M5, drawn dashed in the system diagram. For the items M2 defers, two open LLMs answer the question as the President, without ever seeing the real answer. The frozen instrument labels those answers, and the share that evade enters the fusion as an extra term.
>
> A first look on dev, which is exploratory: when neither model evades, the President evaded on 61% of questions. When both evade, 88%. Added to M1, though, it helps only a little: AUROC 0.872 to 0.875 overall, and 0.722 to 0.746 on the least confident items.
>
> So we expect a small gain, mainly on the evade-or-not decision. It would need training-row generations, a registered prediction, and a human check of the instrument on LLM text.

## 30. Conclusions [19:40]

> Three things to take away.
>
> 1. A stronger backbone, not more structure, broke the classifier plateau.
> 2. Rank, then adjudicate only where unsure. That gave 0.65, on par with the winner, with fewer LLM calls.
> 3. Evasion is a relation. Labels follow the question, and in two LLMs, which questions get evaded follows the speaker's role.
>
> And our three words: "I already have" is Explicit for one question, Dodging for another, and Dodging for an unrelated one. The instrument gets all three, because it reads the pair.

## 31. Thank you

> Thank you. We are happy to take questions.
