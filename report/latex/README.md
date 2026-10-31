# Report source (ACL format)

LaTeX source of the mid-evaluation report, [`../mid_evaluation_report.pdf`](../mid_evaluation_report.pdf).
The final report will extend it.

The source uses the official ACL template
([acl-org/acl-style-files](https://github.com/acl-org/acl-style-files), commit `d5adc82`).
`acl.sty` and `acl_natbib.bst` are unmodified copies. The `final` option is set, so the paper is
non-anonymous and has no line numbers.

## Layout

```
main.tex                  preamble, title, authors, section order
figs.tex                  figure macros
sections/                 abstract, 01_introduction ... 07_conclusion, limitations, ethics, appendix
references.bib            cited works only, checked against the ACL Anthology, arXiv or Crossref
figures/fig_architecture.tex, fig_plan.tex           system diagram and planned cascade (TikZ)
figures/fig_deberta_seeds.pdf, fig_qwen_vs_deberta.pdf  data figures (from code/figures/paper_figures.py)
build.sh                  pdflatex -> bibtex -> pdflatex x2, then a log summary
validate_pdf.sh           checks A4 size and embedded fonts, renders pages to qa/ (needs poppler)
```

## Where each number comes from

| Section | Evidence |
|---|---|
| §2 Task, data and evaluation | `docs/raw/taxonomy_counts.txt`, `docs/scorer_geometry.md`, `docs/raw/mideval/mideval_analysis.txt` |
| §4 Encoder track | `docs/experiment_log.md` (E0–E12), `docs/raw/E11_replication.txt` |
| §5 LLM classifier | `code/models/llm_classifier.py`, `docs/raw/E13_*.txt` |
| §6 Analysis | `docs/raw/mideval/`, `docs/raw/E13_final_analysis.txt`, `docs/raw/E13_Q8_topk_confidence.txt` |
| §7 Plan, Appendix A | `docs/experiment_log.md`, `docs/raw/E13_pilots_and_timing.txt` |

[`docs/results_sources.md`](../../docs/results_sources.md) maps every number in the report to the
script and output file behind it.

## Build

```bash
bash report/latex/build.sh                      # -> report/latex/main.pdf (TeX Live or MiKTeX)
POPPLER=<poppler bin dir> bash report/latex/validate_pdf.sh
cd code && python -m figures.paper_figures      # regenerate the data figures (needs runs/ from Hugging Face)
```

On Overleaf, upload this folder as a project and compile `main.tex` with pdfLaTeX. Shell escape is not
needed.
