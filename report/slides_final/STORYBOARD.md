# Final presentation: storyboard

The talk runs 20 minutes, followed by 10 minutes of questions, in 22 slides. The script is in `SPEAKER_NOTES.md`.

## Structure

- **Examples first.** Three real dev items (slide 3) show what is being classified; "I already have" (one answer, two
  unanimous and opposite labels, slide 4) is the hook, tested again in Stage 4 (slides 15–16).
- **A timeline before the detail.** Slide 6 shows every experiment in order, with its outcome in plain words, grouped
  under the two questions, so the audience has the map before the stages.
- **Five stages, one question each.** Stages 1 and 2 end with a red *What it raised* line, which the next slide answers.
- **Outcomes in plain words.** No coloured tags, ticks or crosses: each result says whether the registered prediction was
  supported, and claims are worded no stronger than the evidence (dev only, "stronger backbone", "on par", the role
  effect exploratory).
- **One frozen system** labels everything from Stage 4 on.
- **The whole project in one diagram** (slide 21): the cascade in the middle, how each part was built (left, Stages 1–3)
  and what the frozen instrument was used for (right, Stages 4–5). The talk closes on a thank-you slide with the links.

## Slides

| # | Stage | Title | Figure |
|---|---|---|---|
| 1 | Title | Evasion Is a Relation | — |
| 2 | Opening | The task: how clearly does a politician answer each sub-question? | taxonomy table |
| 3 | Opening | What are we classifying? Three real dev items | example cards |
| 4 | Opening | One answer, two sub-questions, two labels | example cards |
| 5 | Opening | The shared task: LLM pipelines won, fine-tuned encoders stalled near 0.50 | leaderboard |
| 6 | Opening | What we did: each experiment answering the last | **timeline (TikZ)** |
| 7 | Stage 1 | One pipeline, two backbones, one decision rule | `fig_architecture` |
| 8 | Stage 1 | Structural changes failed; a stronger backbone did not | three panes: `s_changes` (gains), `deberta_seeds` (S2, S1) |
| 9 | Stage 2 | The classifier ranks better than it decides | `s_ranking` |
| 10 | Stage 2 | The cascade: a classifier, a router and an LLM adjudicator | cascade (TikZ) |
| 11 | Stage 2 | Deferral rate against cost on dev | `s_frontier` |
| 12 | Stage 2 | Controls on the cascade: routing, fusion and where the 27B helps | `s_robust` |
| 13 | Stage 3 | We attacked our own result, one registered test at a time | test table, `s_optA` |
| 14 | Stage 3 | The frozen system, against the winner | results table |
| 15 | Stage 4 | Evasion is a relation between a question and an answer | `s_relation` |
| 16 | Stage 4 | Validity check: does the instrument read the question? | `s_swap` |
| 17 | Stage 4 | Is evasion driven by the question or by the speaker? | design (TikZ) |
| 18 | Stage 4 | The role changes *where* models evade, not how often | `s_audit` |
| 19 | Stage 5 | Can the full instrument run cheaper? | cost table, `s_precision` |
| 20 | Conclusions | Limitations, and the next module they point to | limitations table, M5 check |
| 21 | Conclusions | The whole project in one picture | **system diagram (TikZ)** |
| 22 | Conclusions | Thank you. Questions? | links |

Material cut from the 31-slide version for time (confusion matrix, shortcut panels, backup slides) stays in
`backup.tex` and the figures folder for questions. `s_cascade` and `s_backbone` are still generated but no longer shown.
