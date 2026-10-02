# `paper/` — the mid-evaluation report in ACL format

The project written up as an ACL long paper, built on the **official ACL template**
([acl-org/acl-style-files](https://github.com/acl-org/acl-style-files), commit `d5adc82`).
`acl.sty` and `acl_natbib.bst` are unmodified copies. The paper uses the template's `final`
option, so it is non-anonymous, with no line numbers.

## Layout
```
main.tex                   preamble (template packages + amsmath, amssymb, booktabs, xurl, tikz), title, authors, section order
figs.tex                   the three figures, as macros (\FigureArch, \FigureLLM, \FigureExample)
sections/                  abstract, 01_introduction ... 07_conclusion, limitations, appendix
references.bib             verified entries only (checked against the papers or their ACL Anthology record)
figures/fig_architecture.tex   Figure 1, the system diagram (TikZ, monochrome)
figures/fig_qwen_vs_deberta.pdf   Figure 2 (vector; clarity/paper_figures.py)
build.sh                   pdflatex -> bibtex -> pdflatex x2 + log summary (undefined refs/citations, overfull boxes, floats)
validate_pdf.sh            A4 page size, fonts embedded, no Type 3 fonts; renders every page to qa/ (needs poppler)
```

Table 1 comes from `clarity/taxonomy_counts.py` (`clarity/reports/raw/taxonomy_counts.txt`); the worked example (Figure 3) comes from `clarity/paper_example.py`, which writes
`clarity/reports/raw/paper_example_dev17.txt`.

## Overleaf
Upload `paper-overleaf.zip` (built by `build_mid_submission.sh` in the repository root) with
New Project → Upload Project. Set the
compiler to **pdfLaTeX** and the main document to `main.tex`. No shell-escape is needed.

## Which source documents each section rests on
| section | evidence |
|---|---|
| §2 Problem Statement, Data and Evaluation | `clarity/reports/raw/taxonomy_counts.txt`, `clarity/reports/01_scorer_geometry.md`, `clarity/reports/raw/mideval/mideval_analysis.txt`, `clarity/decide.py` |
| §3 Related Work | the cited papers (ACL Anthology records) |
| §4 The Encoder Track | `clarity/README.md`, the experiment log E0–E12, `clarity/reports/raw/E11_replication.txt`, `clarity/reports/04_results_sources.md` §1–8 |
| §5 The 8B LLM classifier | `clarity/llm_classifier.py`, `clarity/reports/raw/E13_*.txt`, `clarity/reports/04_results_sources.md` §9 |
| §5–6 qualitative analysis, §6 Analysis | `clarity/reports/raw/qualitative_examples.txt`, `clarity/reports/04_results_sources.md` §6–9, `clarity/reports/raw/mideval/`, `raw/E13_final_analysis.txt` §D–E, `raw/E13_Q8_topk_confidence.txt`, `raw/paper_example_dev17.txt` |
| §7 Plan, App. A | the experiment log; `clarity/reports/raw/E13_pilots_and_timing.txt` |

Every number in the paper traces to one of these files. All scores are on the dev set (test labels
were never released).

## Build and check
```bash
bash paper/build.sh
POPPLER=<poppler bin dir> bash paper/validate_pdf.sh
python clarity/paper_figures.py    # regenerate figures/fig_qwen_vs_deberta.pdf (needs clarity/runs/ from the HF repo; CPU)
python clarity/paper_example.py    # the worked example's numbers
```

**Status (2026-10-02):** compiles with no errors, no undefined citations or references and no
overfull boxes; A4, all fonts embedded, 16 references, each checked against its ACL Anthology,
venue or arXiv record. The main text and Limitations fill about 7.6 pages, the references follow,
and the appendix is one page.
