# Final presentation: storyboard

The talk runs 20 minutes, followed by 10 minutes of questions. The target is 17:45. The script is in `SPEAKER_NOTES.md`.

## Structure

- The talk has five stages, each asking one question. A footline label shows the current stage.
- Every stage closes with "What it raised", which is the question the next stage answers.
- Result slides show *Predicted / Found* with a held or failed tag, so surprises and failed predictions are visible.
- Claims are worded no stronger than the evidence. The interval that spans zero is stated, the role effect is marked exploratory, and we say "stronger backbone", not "knowledge".
- From Stage 4 on, one frozen system labels everything.
- Each limitation is paired with what we did and what comes next.
- The example from the mid-evaluation (three real dev items) follows the task slide. The "I already have" answer appears before Stage 3, is tested in Stage 4 and returns in the conclusions.

## Main deck (28 slides)

| # | Stage | Title |
|---|---|---|
| 1 | Title | Title slide |
| 2 | Opening | The task: how clearly does a politician answer each sub-question? |
| 3 | Opening | What are we classifying? Three real dev items |
| 4 | Opening | Shared-task results: LLM pipelines won, encoders stalled near 0.50 |
| 5 | Opening | Overview: five stages, each asking one question |
| 6 | Stage 1: what limits a classifier? | The classification pipeline shared by both backbones |
| 7 | Stage 1: what limits a classifier? | A controlled DeBERTa baseline and seed ensembling |
| 8 | Stage 1: what limits a classifier? | The decision rule: logit adjustment |
| 9 | Stage 1: what limits a classifier? | Structural changes failed, while the decision rule, training length and full question helped |
| 10 | Stage 1: what limits a classifier? | Where the DeBERTa ensemble fails |
| 11 | Stage 1: what limits a classifier? | A stronger pretrained backbone gained more per model than all encoder changes combined |
| 12 | Stage 2: the cascade | The classifier ranks the right label better than it picks it |
| 13 | Stage 2: the cascade | System design: a classifier, a router and an LLM adjudicator |
| 14 | Stage 2: the cascade | Deferral rate against cost on dev |
| 15 | Example: a shared answer | One answer, two sub-questions, two labels |
| 16 | Stage 3: which gains survive? | Controls on the cascade: routing, fusion and where the 27B helps |
| 17 | Stage 3: which gains survive? | Option order changes the 27B's answers |
| 18 | Stage 3: which gains survive? | The frozen system compared with the winning system |
| 19 | Stage 4: what it measures | Labels of one answer across its sub-questions |
| 20 | Stage 4: what it measures | Validity check: removing or swapping the question |
| 21 | Stage 4: what it measures | Does the question or the speaker drive evasion? Design |
| 22 | Stage 4: what it measures | Evasion rates by role |
| 23 | Stage 4: what it measures | What the role result shows and does not show |
| 24 | Stage 5: cheaper? | Lower precision: FP8 and NF4 against 16-bit |
| 25 | Conclusions | Registered predictions that steered the work |
| 26 | Conclusions | Limitations, each with what we did and what comes next |
| 27 | Conclusions | Conclusions |
| 28 | Conclusions | Thank you. Questions? |

## Backup (13 slides)

- Is the comparison with the winning system fair?
- Option order: what we measured, and how the final numbers were produced
- What else changed with the backbone?
- E25b: confounds, controls and per-model results
- Is the instrument valid on LLM-written answers?
- Is the cascade's tolerance of FP8 true by construction?
- Choices made after seeing results, and how we logged them
- The other Stage-3 tests
- Cost on one GPU, and why the frozen system makes 1.43 calls per item
- Shortcuts: answer prefixes and frozen-model probes
- E25: the register of hypotheses
- The metric and the 0.684 reference
- Implementation details
