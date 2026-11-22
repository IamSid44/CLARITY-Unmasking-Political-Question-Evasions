#!/usr/bin/env bash
# E33: the five url-grouped cross-fit folds of M1 (Qwen3-8B-Base LoRA, the M1 recipe, seed 0) on one GPU, in order.
#   GPU=MIG-def14cbb-599b-58f4-9cef-4befe9ff8506 bash code/launch/e33_folds.sh [first_fold] [last_fold]
# Logs: logs/paper/E33_cf_url_fold<k>.log. Resumable (llm_classifier keeps resume.pt; finished folds are skipped).
set -uo pipefail
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
VPY=/scratch/shlok/Temp/.venv-vllm/bin/python
export PYTHONPATH=/scratch/shlok/Temp/.pylib-peft HF_HOME=/scratch/shlok/.cache/huggingface HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${GPU:-MIG-def14cbb-599b-58f4-9cef-4befe9ff8506} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT/code"
for k in $(seq "${1:-0}" "${2:-4}"); do
  echo "$(date '+%F %T') fold $k start" >> "$ROOT/logs/paper/E33_folds.log"
  "$VPY" -m models.llm_classifier --name E33_cf_url --seed 0 --fold "$k" --n-folds 5 --group-by url \
    --epochs 12 --lr 1e-4 --select last --no-save-epoch-adapters --track \
    > "$ROOT/logs/paper/E33_cf_url_fold$k.log" 2>&1
  echo "$(date '+%F %T') fold $k exit $?" >> "$ROOT/logs/paper/E33_folds.log"
done
