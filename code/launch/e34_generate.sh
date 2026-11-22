#!/usr/bin/env bash
# E34: answers to the train questions under N/S/G/P (audit.generate --split train --protocol e34), one model at a time,
# each in a fresh process, on one GPU (default GPU 0, shared). Waits for a model's download to complete first.
#   bash code/launch/e34_generate.sh "olmo2_7b=allenai/OLMo-2-1124-7B-Instruct qwen3_8b=Qwen/Qwen3-8B ..."
# Env: GPU (CUDA_VISIBLE_DEVICES), GPU_UTIL (default 0.4), QUANT (e.g. fp8), MAXLEN (default 4096).
set -uo pipefail
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
VPY=/scratch/shlok/Temp/.venv-vllm/bin/python
export CLARITY_E25_DIR=$ROOT/runs/E34 HF_HOME=/scratch/shlok/.cache/huggingface CUDA_VISIBLE_DEVICES=${GPU:-0}
export PATH=/scratch/shlok/Temp/.venv-vllm/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
LOG=$ROOT/logs/paper/E34_generate.log
cd "$ROOT/code"
for spec in $1; do
  name=${spec%%=*}; path=${spec#*=}
  dl=$ROOT/logs/model_download_$(echo "$path" | tr / _).log
  if [ -f "$dl" ]; then
    until grep -q COMPLETE "$dl"; do sleep 120; done
  fi
  echo "$(date '+%F %T') $name start" >> "$LOG"
  HF_HUB_OFFLINE=1 "$VPY" -m audit.generate --split train --protocol e34 --model "$spec" \
    --max-model-len "${MAXLEN:-4096}" --gpu-util "${GPU_UTIL:-0.4}" ${QUANT:+--quantization $QUANT} \
    >> "$ROOT/logs/paper/E34_generate_$name.log" 2>&1
  echo "$(date '+%F %T') $name exit $?" >> "$LOG"
done
