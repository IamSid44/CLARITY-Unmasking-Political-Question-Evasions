#!/usr/bin/env bash
# E33 after the five folds: CF on the control and dev frames (C33-3, H33-1) -> tau -> checks -> H33-1 -> E35 (dev+train
# when the e35 27B run exists). Runs on our MIG slice; waits for fold 4 to finish.
#   bash code/launch/e33_post.sh
set -uo pipefail
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
VPY=/scratch/shlok/Temp/.venv-vllm/bin/python
CPY=/scratch/shlok/Temp/.venv-clarity-cpu/bin/python
export PYTHONPATH=/scratch/shlok/Temp/.pylib-peft HF_HOME=/scratch/shlok/.cache/huggingface HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${GPU:-MIG-def14cbb-599b-58f4-9cef-4befe9ff8506} CLARITY_E25_DIR=$ROOT/runs/E34
LOG=$ROOT/logs/paper/E33_post.log
until [ -f "$ROOT/runs/E33_cf_url/fold4/metrics.json" ]; do sleep 300; done
cd "$ROOT/code"
echo "$(date '+%F %T') infer_cf controls start" >> "$LOG"
"$VPY" -m audit.infer_cf --batch 16 --frames train_own,dev_llm,dev_llm_rand >> "$LOG" 2>&1 || { echo "infer_cf FAILED" >> "$LOG"; exit 1; }
PYTHONPATH= "$CPY" -m audit.cf_tau >> "$LOG" 2>&1
PYTHONPATH= "$CPY" -m preregistered_analyses.e33_llm_swap >> "$LOG" 2>&1
PYTHONPATH= "$CPY" -m preregistered_analyses.e35_binding >> "$LOG" 2>&1
echo "$(date '+%F %T') E33 post done" >> "$LOG"
