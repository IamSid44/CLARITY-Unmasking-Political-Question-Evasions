#!/usr/bin/env bash
# Build the mid-evaluation submission package from the current tree:
#   Nier_ANLP-Mid/        report PDF, README, and clarity/ (code, experiment definitions, records)
#   Nier_ANLP-Mid.zip     the file to submit
#   paper-overleaf.zip    the LaTeX project, for Overleaf (New Project -> Upload Project); not part of the submission
# Nier_ANLP-Mid/README.md is written by hand and kept; everything else in the folder is rebuilt.
# Left out on purpose: the LaTeX source, the GPU-provider launch scripts and guide, the raw VM run logs,
# the data cache (downloaded on first use) and anything gitignored (runs/, logs/, .env).
#   bash build_mid_submission.sh            (run from the repo root, after bash paper/build.sh)
set -euo pipefail
cd "$(dirname "$0")"
OUT=Nier_ANLP-Mid
[ -f paper/main.pdf ] || { echo "build the report first: bash paper/build.sh"; exit 2; }
[ -f "$OUT/README.md" ] || { echo "missing $OUT/README.md"; exit 2; }

# fresh folder, keeping the hand-written README
find "$OUT" -mindepth 1 -maxdepth 1 ! -name README.md -exec rm -rf {} +

# 1. the report
cp paper/main.pdf "$OUT/Nier_ANLP-Mid-Report.pdf"

# 2. code and records: every tracked or new (not ignored) file under clarity/, minus the data cache
#    and the three raw logs written on the rented VMs (the analysis outputs they fed are kept)
git ls-files --cached --others --exclude-standard clarity \
  | grep -v -e '__pycache__' -e '^clarity/data/cache/' \
            -e '^clarity/reports/raw/E13_lane_summaries.txt$' \
            -e '^clarity/reports/raw/E13_pilots_and_timing.txt$' \
            -e '^clarity/reports/raw/E13_Q8_fullq_lora_summary.txt$' \
  | while IFS= read -r f; do [ -f "$f" ] && cp --parents "$f" "$OUT/"; done

# 3. adjust the copies (the repo itself is left unchanged): no machine-specific paths, and no pointers
#    to files that are not in the package
grep -rlZ '/scratch/shlok/Temp/CLARITY-Unmasking-Political-Question-Evasions/' "$OUT" \
  | xargs -0 -r sed -i 's#/scratch/shlok/Temp/CLARITY-Unmasking-Political-Question-Evasions/##g'
python - "$OUT/clarity/README.md" <<'PY'
import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
old = ("(`llm_classifier.py`) was run on rented single-GPU machines with `../jarvis_drive.sh` and\n"
       "`../jarvis_setup.sh`; see `../JARVISLABS_PORTING_GUIDE.md`.")
new = ("(`llm_classifier.py`) was run on rented single-GPU machines; the launch scripts are in the\n"
       "GitHub repository.")
assert old in s, "clarity/README.md launch paragraph changed; update build_mid_submission.sh"
open(p, "w", encoding="utf-8").write(s.replace(old, new))
PY

# 4. zips
rm -f "$OUT.zip" paper-overleaf.zip
python - "$OUT" <<'PY'
import sys, zipfile, pathlib
out = pathlib.Path(sys.argv[1])
with zipfile.ZipFile(f"{out}.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(out.rglob("*")):
        if p.is_file():
            z.write(p, p.as_posix())
paper = pathlib.Path("paper")
files = ["main.tex", "figs.tex", "acl.sty", "acl_natbib.bst", "references.bib"]
files += [f"sections/{p.name}" for p in sorted((paper / "sections").glob("*.tex"))]
files += ["figures/fig_architecture.tex", "figures/fig_qwen_vs_deberta.pdf"]
with zipfile.ZipFile("paper-overleaf.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(paper / f, f)
PY
echo "== $OUT: $(find "$OUT" -type f | wc -l) files"
ls -l "$OUT.zip" paper-overleaf.zip
