# Final report source (ACL format)

LaTeX source of the final report, [`../final_report.pdf`](../final_report.pdf). It extends the mid-evaluation
source in [`../latex/`](../latex/), which is left unchanged.

The official ACL template ([acl-org/acl-style-files](https://github.com/acl-org/acl-style-files), commit `d5adc82`)
is used unmodified, with the `final` option (non-anonymous, no line numbers).

## Page budget (course rules)

| Part | Pages |
|---|---|
| Main text, §1–§9 (including the short ethics paragraph) | 1–8 |
| Limitations, Challenges and Future Work (not counted; only limitations, no results; just before the references) | 8–9 |
| References (15 works) | 9 |
| Appendix A (experiment ledger), B (implementation, frozen instrument) | 10–11 (two pages) |

## Layout

```
main.tex                  preamble, title, authors, section order
figs.tex                  figure macros
sections/                 abstract, 01_introduction ... 09_conclusion, limitations, appendix
figures/fig_path.tex      Figure 1, the research path (TikZ)
figures/fig_system.tex    Figure 2, the whole system: training and fitting, the cascade, its use (TikZ)
figures/fig_e25_*.pdf     from code/figures/report_e25_figures.py (runs/E25/analysis.json), report style
figures/fig_stage1.pdf, fig_frontier.pdf, fig_stage3.pdf, fig_relation.pdf, fig_precision.pdf
                          from code/figures/report_final_figures.py (values from docs/raw/*, costs on one GPU)
references.bib            cited works only
build.sh, validate_pdf.sh build, then check A4 / embedded fonts / no Type 3 fonts
```

## Where each number comes from

| Section | Evidence |
|---|---|
| §4 Stage 1 (E0–E13) | `docs/experiment_log.md`, `docs/raw/mideval/`, `docs/raw/E13_final_analysis.txt` |
| §5 Stage 2 (E14–E18) | `docs/raw/E14_m1_analysis.txt`, `E15_*`, `E16_*`, `E17_m4_analysis.txt`, `E18_final.txt` |
| §6 Stage 3 (E19–E23) | `docs/raw/E19_tier0.txt`, `E20_router.txt`, `E21_E22_analysis.txt`, `E23_systems.txt` |
| §7 Stage 4 (E24–E25) | `docs/raw/E24_prechecks.txt`, `E25_analysis.txt` |
| §8 Stage 5 (E26) | `docs/raw/E26_analysis.txt` |

[`docs/results_sources.md`](../../docs/results_sources.md) maps every number to its script and output file; every GPU-s figure and LLM-call count comes from `docs/raw/cost_same_gpu.txt` (§17 there).

## Build

```bash
bash report/final/build.sh                       # -> report/final/main.pdf (MiKTeX or TeX Live)
cp report/final/main.pdf report/final_report.pdf
POPPLER=<poppler bin dir> bash report/final/validate_pdf.sh
cd code && python -m figures.report_e25_figures && python -m figures.report_final_figures  # regenerate all data figures
```
