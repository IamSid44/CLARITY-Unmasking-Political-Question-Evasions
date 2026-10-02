#!/usr/bin/env bash
# Compile ONE section in isolation (a temp copy of main.tex that inputs only that section),
# so section agents can validate their LaTeX without touching main.pdf or each other.
#   bash paper/check_section.sh 04_encoder [appendix_encoder]
set -u
cd "$(dirname "$0")"
T=$(mktemp -d); cp acl.sty acl_natbib.bst references.bib "$T"/; cp -r figures sections "$T"/
inputs=""; for s in "$@"; do inputs="$inputs\\input{sections/$s}"$'\n'; done
sed -n '1,/\\begin{document}/p' main.tex > "$T/main.tex"
printf '\\maketitle\n%s\n\\bibliography{references}\n\\end{document}\n' "$inputs" >> "$T/main.tex"
cd "$T"
pdflatex -interaction=nonstopmode -file-line-error main.tex >/dev/null 2>&1; bibtex main >/dev/null 2>&1
pdflatex -interaction=nonstopmode -file-line-error main.tex >/dev/null 2>&1
pdflatex -interaction=nonstopmode -file-line-error main.tex >/dev/null 2>&1
pages=$(grep -oE 'Output written on main\.pdf .[0-9]+ pages?' main.log | grep -oE '[0-9]+ pages?')
echo "== $*: ${pages:-NO PDF}"
echo "== LaTeX errors:";        { grep -E '^\./[^:]*:[0-9]+: |^! ' main.log || echo "   none"; } | head -20
echo "== undefined citations (must be none):"; { grep -E 'Citation .* undefined' main.log | sort -u || true; } ; grep -qE 'Citation .* undefined' main.log || echo "   none"
echo "== undefined references (cross-section refs are expected in isolation; check your own labels):"
{ grep -E 'Reference .* undefined' main.log | sort -u | head -20; } ; grep -qE 'Reference .* undefined' main.log || echo "   none"
echo "== overfull boxes (must be none > 1pt):"; { grep -E 'Overfull' main.log || echo "   none"; } | head -10
mkdir -p /tmp/paper_checks && cp main.pdf "/tmp/paper_checks/check_$1.pdf" 2>/dev/null && echo "== pdf: /tmp/paper_checks/check_$1.pdf"
cd / && rm -rf "$T"
