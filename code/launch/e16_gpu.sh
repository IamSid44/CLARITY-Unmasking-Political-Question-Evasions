#!/usr/bin/env bash
# E16 + E17 preparation on the lab server's GPU 0, in order (tmux session clarity-E16), log logs/E16_gpu.log:
#   1. M3 (local Qwen3.8-27B-FP8, k = 0, full prompt) on every train row (E17's targets) and on the frontier's
#      (dev item, top-4) pairs -- one model load, identical pairs reused from E15;
#   2. the prompt-part ablation: no_guide, no_defs, bare on the screen rows and dev;
#   3. the cost harness (M1 seed timing, bf16 and FP8).
set -uo pipefail
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
VPY=/scratch/shlok/Temp/.venv-vllm/bin/python
PY=/scratch/shlok/Temp/.venv/bin/python
LOG=$ROOT/logs/E16_gpu.log
VLLM_ARGS='{"limit_mm_per_prompt": {"image": 0, "video": 0}}'
note() { echo "$(date '+%F %T') $*" >> "$LOG"; }
m3() {  # tag, then extra adjudicate flags
  local tag=$1; shift
  note "start M3 $tag $*"
  (cd "$ROOT/code" && CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 "$VPY" -m cascade.adjudicate --backend vllm \
     --model Qwen/Qwen3.8-27B-FP8 --tag "$tag" --k 0 --vllm-args "$VLLM_ARGS" "$@" >> "$ROOT/logs/E16_$tag.log" 2>&1) \
    && note "done M3 $tag" || note "FAILED M3 $tag (logs/E16_$tag.log)"
}
cd "$ROOT/code" || exit 1
m3 qwen38_27b_k0 --splits trainall --jobs-file ../runs/M3/frontier_jobs.jsonl
for v in no_guide no_defs bare; do m3 "qwen38_27b_k0_$v" --prompt-variant "$v" --splits screen,dev; done
note "start cost harness"
(CUDA_VISIBLE_DEVICES=0 "$PY" -m cascade.cost --fp8 >> "$ROOT/logs/E16_cost.log" 2>&1) && note "done cost harness" \
  || note "FAILED cost harness (logs/E16_cost.log)"
note "ALL DONE"
