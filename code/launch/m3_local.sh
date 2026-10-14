#!/usr/bin/env bash
# E15 (M3): the local Qwen3.8-27B-FP8 arms on the lab server's GPU 0, in tmux session clarity-E15.
#   bash code/launch/m3_local.sh screen        # k = 0, 4, 10 on the 300 screen rows (the example-count measurement)
#   bash code/launch/m3_local.sh full <k>      # the chosen k on slice, dev, test, train50 (screen rows are reused)
#   tail -f logs/E15_local.log
set -uo pipefail
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
PY=/scratch/shlok/Temp/.venv-vllm/bin/python
MODEL=${MODEL:-Qwen/Qwen3.8-27B-FP8}
LOG=$ROOT/logs/E15_local.log
VLLM_ARGS=${VLLM_ARGS:-'{"limit_mm_per_prompt": {"image": 0, "video": 0}}'}

run() {  # tag k splits
  echo "$(date '+%F %T') start $1 (k=$2, $3)" >> "$LOG"
  (cd "$ROOT/code" && CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 "$PY" -m cascade.adjudicate --backend vllm --model "$MODEL" \
      --tag "$1" --k "$2" --splits "$3" --vllm-args "$VLLM_ARGS" >> "$ROOT/logs/E15_$1.log" 2>&1) \
    && echo "$(date '+%F %T') done $1" >> "$LOG" || echo "$(date '+%F %T') FAILED $1 (logs/E15_$1.log)" >> "$LOG"
}

free=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i 0 | awk -F', ' '{print $2-$1}')
[ "$free" -lt 60000 ] && { echo "GPU 0 has only ${free} MiB free; the 27B FP8 model needs ~60 GB with its cache"; exit 1; }

case "${1:-}" in
  screen) for k in 0 4 10; do run "qwen38_27b_k$k" "$k" screen; done ;;
  full)   run "qwen38_27b_k${2:?k}" "$2" slice,dev,test,train50 ;;
  *) echo "usage: $0 screen | full <k>"; exit 2 ;;
esac
