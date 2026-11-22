#!/usr/bin/env bash
# E32: label one E32 frame with the frozen instrument on the lab server (GPU 0, shared):
#   bash code/launch/e32_label.sh <frame-glob> <27B tag prefix>     e.g.  ctrl_mistral e32ctrl   |   llm e32
# Assumes M1's 10 seeds have scored the frame (audit.infer with CLARITY_E25_DIR=runs/E32). Steps: route (jobs) ->
# the 27B on the deferred items in the 4 cyclic option orders -> final labels (runs/E32/labels/<frame>.npz).
set -euo pipefail
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
VPY=/scratch/shlok/Temp/.venv-vllm/bin/python
export CLARITY_E25_DIR=$ROOT/runs/E32 CLARITY_M3_PREFIX=$2 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=${GPU:-0}
export PATH=/scratch/shlok/Temp/.venv-vllm/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
cd "$ROOT/code"
"$VPY" -m audit.system jobs "$1"
for o in taxonomy shift1 shift2 shift3; do
  "$VPY" -m cascade.adjudicate --backend vllm --model Qwen/Qwen3.8-27B-FP8 --k 0 --tag "${2}_$o" --splits "" \
    --jobs-file "$ROOT/runs/E32/jobs_27b.jsonl" --order "$o" --gpu-util "${GPU_UTIL:-0.55}" --max-model-len 8192 \
    --vllm-args '{"limit_mm_per_prompt": {"image": 0, "video": 0}, "max_num_seqs": 128, "enforce_eager": true}'
done
"$VPY" -m audit.system label "$1"
echo "E32 labelled: $1"
