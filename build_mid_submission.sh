#!/usr/bin/env bash
# Build the mid-evaluation submission package from the current tree:
#   Nier_ANLP-Mid/              report PDF, the LaTeX source (= the Overleaf project), code, records
#   Nier_ANLP-Mid.zip           the file to submit
#   paper-overleaf.zip          just the LaTeX project, for Overleaf (New Project -> Upload Project)
# Nier_ANLP-Mid/README.md is written by hand and kept; everything else in the folder is rebuilt.
#   bash build_mid_submission.sh            (run from the repo root, after bash paper/build.sh)
set -euo pipefail
cd "$(dirname "$0")"
OUT=Nier_ANLP-Mid
[ -f paper/main.pdf ] || { echo "build the report first: bash paper/build.sh"; exit 2; }
[ -f "$OUT/README.md" ] || { echo "missing $OUT/README.md"; exit 2; }

# fresh folder, keeping the hand-written README
find "$OUT" -mindepth 1 -maxdepth 1 ! -name README.md -exec rm -rf {} +

# 1. the report and its LaTeX source (single version, as compiled)
cp paper/main.pdf "$OUT/Nier_ANLP-Mid-Report.pdf"
mkdir -p "$OUT/paper/sections" "$OUT/paper/figures"
cp paper/main.tex paper/figs.tex paper/acl.sty paper/acl_natbib.bst paper/references.bib "$OUT/paper/"
cp paper/sections/*.tex "$OUT/paper/sections/"
cp paper/figures/fig_architecture.tex paper/figures/fig_plan.tex paper/figures/fig_qwen_vs_deberta.pdf paper/figures/fig_deberta_seeds.pdf "$OUT/paper/figures/"

# 2. code and records: every tracked or new (not ignored) file under clarity/, minus the data cache
git ls-files --cached --others --exclude-standard clarity \
  | grep -v -e '__pycache__' -e '^clarity/data/cache/' \
  | while IFS= read -r f; do [ -f "$f" ] && cp --parents "$f" "$OUT/"; done
cp jarvis_drive.sh jarvis_setup.sh JARVISLABS_PORTING_GUIDE.md "$OUT/"

# 3. scrub machine-specific paths from the copies (the repo itself is left unchanged)
sed -i 's#^PY=/scratch/shlok/Temp/.venv/bin/python#PY=${PY:-python}#' "$OUT/clarity/run_pipeline.sh" "$OUT/clarity/run_queue.sh"
grep -rlZ '/scratch/shlok/Temp/CLARITY-Unmasking-Political-Question-Evasions/' "$OUT" \
  | xargs -0 -r sed -i 's#/scratch/shlok/Temp/CLARITY-Unmasking-Political-Question-Evasions/##g'
sed -i 's#in this OneDrive-synced clone#in a local clone#' "$OUT/jarvis_drive.sh"

# 4. zips (python: available everywhere the scripts run)
rm -f "$OUT.zip" paper-overleaf.zip
python - "$OUT" <<'PY'
import sys, zipfile, pathlib
out = pathlib.Path(sys.argv[1])
with zipfile.ZipFile(f"{out}.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(out.rglob("*")):
        if p.is_file():
            z.write(p, p.as_posix())
with zipfile.ZipFile("paper-overleaf.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted((out / "paper").rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(out / "paper").as_posix())
PY
echo "== $OUT: $(find "$OUT" -type f | wc -l) files"
ls -l "$OUT.zip" paper-overleaf.zip
