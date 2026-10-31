#!/usr/bin/env bash
# Build the slides: pdflatex twice (no bibliography), then a log summary.
#   bash report/slides_final/build.sh   -> report/slides_final/main.pdf (copy to report/final_slides.pdf to publish)
set -u
cd "$(dirname "$0")"
run() { pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex > build_main.log 2>&1; }
{ run && run; } || { echo "PDFLATEX FAILED"; grep -E -A4 '^\./[^:]*:[0-9]+: |^! ' main.log | head -40; exit 1; }
echo "== build OK: $(grep -oE 'Output written on main\.pdf .[0-9]+ pages' main.log | grep -oE '[0-9]+ pages')"
echo "== overfull boxes: $(grep -c 'Overfull' main.log)"; grep -A1 'Overfull' main.log | grep -v '^--' | head -30
echo "== undefined: $(grep -c 'undefined' main.log)"
