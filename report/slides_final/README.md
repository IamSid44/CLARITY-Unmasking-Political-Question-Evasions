# Final presentation (Beamer, 16:9)

| File | Contents |
|---|---|
| `main.tex` | the deck in the same Madrid Beamer format as the mid-evaluation deck (`../slides_beamer/main.tex`): 31 slides. `backup.tex` holds former backup slides, no longer included |
| `figures/s_*.pdf` | data figures, from `code/figures/slide_figures.py` |
| `STORYBOARD.md` | the narrative design, slide by slide |
| `SPEAKER_NOTES.md` | the script, speaker split and running clock |
| `build.sh` | pdflatex twice; the result is `main.pdf` (published as `../final_slides.pdf`) |

```bash
cd code && python -m figures.slide_figures      # regenerate the figures (CPU, ~10 s)
bash report/slides_final/build.sh                     # -> report/slides_final/main.pdf
```
