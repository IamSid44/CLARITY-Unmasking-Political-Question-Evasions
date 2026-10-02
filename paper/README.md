# `paper/` — the report in ACL format

The project written up as an ACL long paper, built on the **official ACL template**
([acl-org/acl-style-files](https://github.com/acl-org/acl-style-files), commit `d5adc82`, 2026-06-29).
`acl.sty` and `acl_natbib.bst` are unmodified copies (md5-checked). `acl_latex_template.tex` is the
template as downloaded, kept for reference. The paper uses the template's `final` option, so it is
non-anonymous, with no line numbers.

## Two versions, one text
| file | figures | built by |
|---|---|---|
| `main.tex` → `main.pdf` (**version 1**) | the original PNG renders: `figures/per_seed_progression.png`, `figures/e13_qwen_vs_deberta.png` (from `clarity/mideval_figures.py`, `clarity/e13_figures.py`) | `bash paper/build.sh` |
| `main_v2.tex` → `main_v2.pdf` (**version 2**) | publication figures redrawn by `clarity/paper_figures.py`: vector PDF, serif type, error bars (`figures/v2/`) | `bash paper/build.sh main_v2` |

- **The text is shared.** Both versions input the same `sections/*.tex`.
- **Only the figure blocks differ.** They live in `figs_v1.tex` and `figs_v2.tex`, as the macros
  `\FigureEncSeeds` and `\FigureLLM`.
- **The numbers in both versions' figures are identical:** dev macro-F1 from the saved per-seed
  probabilities, with systems scored by nested CV.

## Layout
```
main.tex, main_v2.tex      preamble (official template packages + amsmath, amssymb, booktabs), title, authors, section order
figs_v1.tex, figs_v2.tex   the two figures of each version
sections/                  one file per section: abstract, 01_introduction ... 07_conclusion, limitations, ethics, appendix_*
references.bib             verified entries only (checked against the PDFs in Materials/ or the venue record)
figures/, figures/v2/      figure files
build.sh                   pdflatex -> bibtex -> pdflatex x2 + log summary (undefined refs/citations, overfull boxes, floats)
check_section.sh           compiles one or more sections in isolation (used while sections were written in parallel)
validate_pdf.sh            A4 page size, fonts embedded, no Type 3 fonts; renders every page to qa/<name>/ (needs poppler)
```

## Which source documents each section rests on
| section | evidence |
|---|---|
| §2 Task, Data and Evaluation | `clarity/reports/01_scorer_geometry.md`, `mideval/analysis/mideval_analysis.txt`, `clarity/decide.py` |
| §3 Related Work | the papers in `Materials/Proposal_Implementation_Material/Papers_References/` |
| §4 The Encoder Track, App. A | `clarity/README.md`, the experiment log E0–E12, `clarity/reports/raw/E11_replication.txt`, `mideval/RESULTS.md` §1–8 |
| §5 The 8B LLM classifier, App. B | `clarity/llm_classifier.py`, `JARVISLABS_PORTING_GUIDE.md`, `clarity/reports/raw/E13_*.txt`, `mideval/RESULTS.md` §9 |
| §6 Analysis | `mideval/RESULTS.md` §6–9, `mideval/analysis/`, `raw/E13_final_analysis.txt` §D–E, `raw/E13_Q8_topk_confidence.txt` |
| Abstract, §1, §7, Limitations, Ethics, App. C | all of the above; `README.md`, `CONTEXT.md` |

Every number in the paper traces to one of these files. All scores are on the dev set (test labels
were never released).

## Build and check
```bash
bash paper/build.sh            # version 1
bash paper/build.sh main_v2    # version 2
POPPLER=<poppler bin dir> bash paper/validate_pdf.sh main_v2
python clarity/paper_figures.py   # regenerate figures/v2 (needs clarity/runs/ from the HF repo; CPU)
```
Tested with MiKTeX 25.12 (pdfTeX 1.40.28) on Windows; any TeX Live 2023+ works.

**Status (2026-10-02):** both versions compile with no errors, no undefined citations or
references, and no overfull boxes. Both are A4 with all fonts embedded and none of Type 3. The
main text ends on page 8, followed by Limitations, Ethical Considerations, Acknowledgments,
References and Appendices A–C.
