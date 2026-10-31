# Final presentation (Beamer, 16:9)

| File | Contents |
|---|---|
| `main.tex`, `backup.tex` | the deck in the same Madrid Beamer format as the mid-evaluation deck (`../slides_beamer/main.tex`): 28 main slides, then 13 backup slides |
| `figures/s_*.pdf` | data figures, from `code/figures/slide_figures.py` |
| `STORYBOARD.md` | the narrative design, slide by slide |
| `SPEAKER_NOTES.md` | the script, speaker split, running clock, and a question-to-backup map |
| `build.sh` | pdflatex twice; the result is `main.pdf` (published as `../final_slides.pdf`) |

```bash
cd code && python -m figures.slide_figures      # regenerate the figures (CPU, ~10 s)
bash report/slides_final/build.sh                     # -> report/slides_final/main.pdf
```
