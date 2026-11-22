#!/usr/bin/env bash
# E32 post-processing (CPU), run once the OLMo pipeline has labelled runs/E32/labels/llm.npz:
# the registered E32 analysis, E31's robustness on the three-model table, and the figures that read them.
set -euo pipefail
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
PY=/scratch/shlok/Temp/.venv-clarity-cpu/bin/python
cd "$ROOT/code"
"$PY" -m preregistered_analyses.e32_analysis
"$PY" -m preregistered_analyses.e31_role_robustness --llm-table "$ROOT/runs/E32/llm_table.parquet"
"$PY" -m figures.paper_figures_v2
"$PY" -c "from figures import slide_figures as s; s.role_robust()"
echo "E32 post done"
