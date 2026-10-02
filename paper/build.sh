#!/usr/bin/env bash
# Build one version of the paper (MiKTeX or TeX Live): pdflatex -> bibtex -> pdflatex x2, then a log summary.
#   bash paper/build.sh            -> main.pdf     (version 1: original figures)
#   bash paper/build.sh main_v2    -> main_v2.pdf  (version 2: figures from clarity/paper_figures.py)
set -u
cd "$(dirname "$0")"
J=${1:-main}
rm -f "$J.bbl"
run() { pdflatex -interaction=nonstopmode -halt-on-error -file-line-error "$J.tex" > "build_$J.log" 2>&1; }
run || { echo "PDFLATEX FAILED (pass 1)"; grep -E -A3 '^\./[^:]*:[0-9]+: |^! ' "$J.log" | head -40; exit 1; }
if ! bibtex "$J" > "build_bibtex_$J.log" 2>&1; then
  if grep -q 'found no .citation commands' "$J.blg"; then rm -f "$J.bbl"; else echo "BIBTEX FAILED"; cat "build_bibtex_$J.log"; exit 1; fi
fi
{ run && run; } || { echo "PDFLATEX FAILED"; grep -E -A3 '^\./[^:]*:[0-9]+: |^! ' "$J.log" | head -40; exit 1; }
pages=$(grep -oE "Output written on $J\.pdf .[0-9]+ pages?" "$J.log" | grep -oE '[0-9]+ pages?')
echo "== $J build OK: ${pages}"
echo "== bibtex warnings:";  { grep -i 'warning' "$J.blg" | grep -v 'warning\$ -- 0' || echo "   none"; }
echo "== undefined citations/references:"; { grep -E 'Citation .* undefined|Reference .* undefined|There were undefined' "$J.log" || echo "   none"; }
echo "== overfull boxes: $(grep -c 'Overfull' "$J.log")"; grep -A1 'Overfull' "$J.log" | grep -v '^--' | head -20
echo "== underfull boxes: $(grep -c 'Underfull' "$J.log")"
echo "== multiply defined labels:"; { grep 'multiply defined' "$J.log" || echo "   none"; }
echo "== float warnings:"; { grep -E 'Float too large|float specifier changed' "$J.log" || echo "   none"; }
