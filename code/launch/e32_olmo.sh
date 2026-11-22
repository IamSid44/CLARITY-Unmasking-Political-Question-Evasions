#!/usr/bin/env bash
# E32 (registered 2026-11-20): OLMo-2-7B-Instruct answers the 308 dev sub-questions (N/P/F/B) under protocol A (the E25
# prompts unchanged) and protocol B (no name/date), then the frozen instrument labels them. Lab server, GPU 0 (shared).
#   bash code/launch/e32_olmo.sh       (outputs under runs/E32; logs under logs/paper/)
set -euo pipefail
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
VPY=/scratch/shlok/Temp/.venv-vllm/bin/python
export CLARITY_E25_DIR=$ROOT/runs/E32 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=${GPU:-0}
export PATH=/scratch/shlok/Temp/.venv-vllm/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
M=allenai/OLMo-2-1124-7B-Instruct
cd "$ROOT/code"
# one process per model/protocol (vLLM engines do not always release the GPU)
"$VPY" -m audit.generate --model "olmo2_7b=$M" --max-model-len 4096 --gpu-util 0.3
"$VPY" -m audit.generate --model "olmo2_7b_generic=$M" --max-model-len 4096 --gpu-util 0.3 --generic-persona
"$VPY" -m audit.frames llm
E25_EVAL_BS=32 PYTHONPATH=/scratch/shlok/Temp/.pylib-peft "$VPY" -m audit.infer --frames llm
bash "$ROOT/code/launch/e32_label.sh" llm e32
echo "E32 OLMo pipeline done"
