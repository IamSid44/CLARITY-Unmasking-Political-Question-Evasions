#!/usr/bin/env bash
# E34/E35: the 27B zero-shot instrument (ZS) over all nine labels, taxonomy order, k = 0, on one GPU.
#   GPU=<id> bash code/launch/e34_zs.sh llm e34zs_taxonomy      |   GPU=<id> bash code/launch/e34_zs.sh train e35zs_taxonomy
set -uo pipefail
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
VPY=/scratch/shlok/Temp/.venv-vllm/bin/python
export CLARITY_E25_DIR=$ROOT/runs/E34 HF_HOME=/scratch/shlok/.cache/huggingface HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=${GPU:-0}
export PATH=/scratch/shlok/Temp/.venv-vllm/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
cd "$ROOT/code"
"$VPY" -m audit.zs_jobs "$1" "$ROOT/runs/E34/jobs_zs_$1.jsonl"
"$VPY" -m cascade.adjudicate --backend vllm --model Qwen/Qwen3.8-27B-FP8 --k 0 --tag "$2" --splits "" \
  --jobs-file "$ROOT/runs/E34/jobs_zs_$1.jsonl" --order taxonomy --gpu-util "${GPU_UTIL:-0.55}" --max-model-len 8192 \
  --vllm-args '{"limit_mm_per_prompt": {"image": 0, "video": 0}, "max_num_seqs": 128, "enforce_eager": true}' \
  >> "$ROOT/logs/paper/E34_zs_$1.log" 2>&1
echo "$(date '+%F %T') ZS $1 exit $?" >> "$ROOT/logs/paper/E34_zs.log"
