#!/usr/bin/env bash
# ACL compliance checks on a built PDF: page size (A4), fonts embedded, no Type 3 fonts, page count,
# and PNG renders of every page for visual inspection (Report/latex/qa/<name>/page-*.png).
#   bash Report/latex/validate_pdf.sh
# Needs poppler (pdfinfo, pdffonts, pdftoppm) on PATH or in $POPPLER (its bin directory).
set -u
cd "$(dirname "$0")"
J=${1:-main}
P=${POPPLER:-}; t() { if [ -n "$P" ]; then echo "$P/$1"; else echo "$1"; fi; }
info=$("$(t pdfinfo)" "$J.pdf") || { echo "pdfinfo failed"; exit 1; }
echo "== $J.pdf"; echo "$info" | grep -E "Pages|Page size|PDF version|Producer"
size=$(echo "$info" | sed -n 's/^Page size: *\([0-9.]*\) x \([0-9.]*\) pts.*/\1 \2/p')
awk -v s="$size" 'BEGIN{split(s,a," "); w=a[1]; h=a[2]; ok=(w>594.5 && w<596.5 && h>840.5 && h<843); \
  printf "== A4 (595.28 x 841.89 pt): %s (%s x %s)\n", ok?"PASS":"FAIL", w, h; exit !ok}'; a4=$?
fonts=$("$(t pdffonts)" "$J.pdf"); echo "$fonts"
notemb=$(echo "$fonts" | awk 'NR>2 && $(NF-4)!="yes"' | wc -l)
echo "== fonts embedded: $([ "$notemb" -eq 0 ] && echo PASS || echo "FAIL ($notemb not embedded)")"
type3=$(echo "$fonts" | awk 'NR>2 && /Type 3/' | wc -l); echo "== Type 3 fonts: $type3 $([ "$type3" -eq 0 ] && echo PASS || echo FAIL)"
rm -rf "qa/$J" && mkdir -p "qa/$J" && "$(t pdftoppm)" -r 70 -png "$J.pdf" "qa/$J/page" && echo "== renders: $(ls "qa/$J" | wc -l) pages in Report/latex/qa/$J/"
[ "$a4" -eq 0 ] && [ "$notemb" -eq 0 ] && [ "$type3" -eq 0 ]
