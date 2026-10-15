#!/usr/bin/env bash
# E17: once the student seeds have finished, evaluate them as systems with M3 (GPU 0, then CPU).
#   bash code/launch/e17_post.sh [seeds, default 0,1,2]      (log: logs/E17_post.log)
set -uo pipefail
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
SEEDS=${1:-0,1,2}
VPY=/scratch/shlok/Temp/.venv-vllm/bin/python
PY=/scratch/shlok/Temp/.venv/bin/python
LOG=$ROOT/logs/E17_post.log
note() { echo "$(date '+%F %T') $*" >> "$LOG"; }
cd "$ROOT/code" || exit 1
note "waiting for Q8_distill_a05 seeds $SEEDS"
for s in ${SEEDS//,/ }; do until [ -f "$ROOT/runs/Q8_distill_a05/seed$s/metrics.json" ]; do sleep 120; done; done
while pgrep -f "^bash .*/run_queue\.sh .*/experiments/E17/" >/dev/null && \
      [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)" -gt 20000 ]; do sleep 60; done
note "students done; M3 on their candidates"
"$PY" -m cascade.frontier jobs --run Q8_distill_a05 --ks 1 --seeds "$SEEDS" >> "$LOG" 2>&1
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 "$VPY" -m cascade.adjudicate --backend vllm --model Qwen/Qwen3.8-27B-FP8 \
  --tag qwen38_27b_k0 --k 0 --splits "" --jobs-file ../runs/M3/frontier_jobs_Q8_distill_a05.jsonl \
  --vllm-args '{"limit_mm_per_prompt": {"image": 0, "video": 0}}' >> "$ROOT/logs/E17_m3.log" 2>&1 \
  || { note "FAILED M3 scoring (logs/E17_m3.log)"; exit 1; }
note "frontier analysis"
"$PY" -m cascade.frontier analyse --run Q8_distill_a05 --ks 1 --seeds "$SEEDS" > "$ROOT/docs/raw/E17_frontier_students.txt" 2>&1 \
  || { note "FAILED frontier"; exit 1; }
"$PY" -m preregistered_analyses.e17_m4_analysis > "$ROOT/docs/raw/E17_m4_analysis.txt" 2>&1 \
  || { note "FAILED analysis"; exit 1; }
HF_HUB_DISABLE_XET=1 "$PY" -m cascade.publish hf M3 M4 >> "$LOG" 2>&1 || note "HF upload failed (results are on disk)"
"$PY" -m cascade.publish wandb E17 >> "$LOG" 2>&1 || note "W&B logging failed (results are on disk)"
note "ALL DONE: docs/raw/E17_m4_analysis.txt (+ HF M3/, M4/; W&B E17-results)"
