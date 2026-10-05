# `Report/latex/`: source of the report in ACL format

The mid-evaluation report (`../Report.pdf`) is built from this folder. The final report extends it.
Built on the **official ACL template**
([acl-org/acl-style-files](https://github.com/acl-org/acl-style-files), commit `d5adc82`);
`acl.sty` and `acl_natbib.bst` are unmodified copies. The `final` option is used, so the paper is
non-anonymous and has no line numbers. This source compiles to a PDF byte-identical to the
submitted `Report/Report.pdf` (restored from commit `04b3f44`, `paper/`).

## Layout
```
main.tex                   preamble (template packages + amsmath, amssymb, booktabs, xurl, tikz), title, authors, section order
figs.tex                   the figures, as macros
sections/                  abstract, 01_introduction ... 07_conclusion, limitations, ethics, appendix
references.bib             verified entries only (checked against the papers or their ACL Anthology record)
figures/fig_architecture.tex   Figure 1, the system diagram (TikZ, monochrome)
figures/fig_plan.tex           Figure 5, the planned cascade (TikZ)
figures/fig_deberta_seeds.pdf, fig_qwen_vs_deberta.pdf   Figures 2 and 3 (vector; written here by code/figures/paper_figures.py)
build.sh                   pdflatex -> bibtex -> pdflatex x2 + log summary (undefined refs/citations, overfull boxes, floats)
validate_pdf.sh            A4 page size, fonts embedded, no Type 3 fonts; renders every page to qa/ (needs poppler)
```

## Where each number comes from
| section | evidence |
|---|---|
| §2 Problem Statement, Data and Evaluation | `docs/raw/taxonomy_counts.txt`, `docs/01_scorer_geometry.md`, `docs/raw/mideval/mideval_analysis.txt`, `code/decision_rules/decide.py` |
| §3 Related Work | the cited papers (ACL Anthology records) |
| §4 The Encoder Track | experiment log E0–E12, `docs/raw/E11_replication.txt`, `docs/04_results_sources.md` |
| §5 The 8B LLM classifier | `code/models/llm_classifier.py`, `docs/raw/E13_*.txt` |
| §6 Analysis | `docs/raw/mideval/`, `docs/raw/E13_final_analysis.txt` §D–E, `docs/raw/E13_Q8_topk_confidence.txt`, `docs/raw/paper_example_dev17.txt` |
| §7 Plan, App. A | the experiment log; `docs/raw/E13_pilots_and_timing.txt`; `docs/05_phase2_plan_and_budget.md` |

`docs/04_results_sources.md` maps every number in the report to the script and output file behind it.

## Build
The lab server has no TeX installation; build locally (TeX Live / MiKTeX) or on Overleaf.
```bash
bash Report/latex/build.sh                               # -> Report/latex/main.pdf
POPPLER=<poppler bin dir> bash Report/latex/validate_pdf.sh
cp Report/latex/main.pdf Report/Report.pdf               # publish
cd code && python -m figures.paper_figures               # regenerate figures/fig_*.pdf (needs runs/ from HF; CPU)
cd code && python -m figures.paper_example               # the worked example's numbers
```

**Overleaf:** upload this folder (New Project → Upload Project), compiler **pdfLaTeX**, main document
`main.tex`. No shell-escape is needed.
