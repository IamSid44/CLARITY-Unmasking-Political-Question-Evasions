#!/usr/bin/env bash
# Populate REF/ with read-only reference clones (Phase 0.2).
#
# CLAUDE.md invariant 1: REF/ is reference-only. Never imported, never on the
# Python path, prompt text never copied into src/. This script only fetches.
#
# Repos, in priority order:
#   1. konstantinosftw/Question-Evasion  - QEvasion dataset paper (Thomas et al.,
#      Findings of EMNLP 2024) repo, MIT. Found via footnote 1 of that paper.
#      HIGHEST PRIORITY: ships dataset/Inter-Annotator/ with per-annotator labels
#      and is the most likely home of, or lead to, the official scorer.
#   2. ther7777/semeval-2026-task6-camsr-cot - TeleAI, 1st place both subtasks.
#   3. moswisarut/SemEval2026-Task6-moswisarut - ChulaNLP, 2nd on Subtask 2.
#      (Confirmed to exist, contrary to Implementation_Guide.md's claim.)
#   4-6. Participant repos that commonly vendor the official scorer; used as
#      fallback sources for the scorer if the organizers' copy is not found.
set -uo pipefail

REF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/REF"
mkdir -p "$REF_DIR"

REPOS=(
  "https://github.com/konstantinosftw/Question-Evasion.git"
  "https://github.com/ther7777/semeval-2026-task6-camsr-cot.git"
  "https://github.com/moswisarut/SemEval2026-Task6-moswisarut.git"
  "https://github.com/semeval-2026-kclarity/clarity.git"
  "https://github.com/CLaC-Lab/SemEval-2026-task6-CLARITY.git"
  "https://github.com/syed0093-umn/SemEval2026_Task6_Duluth.git"
)

ok=0; fail=0
for url in "${REPOS[@]}"; do
  name="$(basename "$url" .git)"
  dest="$REF_DIR/$name"
  if [ -d "$dest/.git" ]; then
    echo "[skip]  $name already cloned"
    ok=$((ok+1)); continue
  fi
  echo "[clone] $url"
  if git clone --quiet --depth 50 "$url" "$dest" 2>&1; then
    echo "[ok]    $name @ $(git -C "$dest" rev-parse --short HEAD)"
    ok=$((ok+1))
  else
    echo "[FAIL]  $name — record in reports/01_reference_inventory.md and move on"
    fail=$((fail+1))
  fi
done

echo
echo "cloned/present: $ok   failed: $fail"
echo "REF_DIR: $REF_DIR"
