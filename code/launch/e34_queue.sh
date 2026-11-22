#!/usr/bin/env bash
# E34 orchestration after the 7-8B generation lane (e34_generate.sh) has finished granite3_8b:
#   GPU 0: frames -> ZS (27B zero-shot) on every answer so far -> OLMo-2-32B answers (fp8, once downloaded) -> frames ->
#          ZS on the new rows (adjudicate skips rows already scored).
#   MIG:   after E33's post-processing and the 32B answers: CF on every answer -> E34 analysis -> E35 -> figures.
# Every step logs to logs/paper/E34_queue.log.
set -uo pipefail
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
VPY=/scratch/shlok/Temp/.venv-vllm/bin/python
CPY=/scratch/shlok/Temp/.venv-clarity-cpu/bin/python
export CLARITY_E25_DIR=$ROOT/runs/E34 HF_HOME=/scratch/shlok/.cache/huggingface
LOG=$ROOT/logs/paper/E34_queue.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
frames() { (cd "$ROOT/code" && "$CPY" -m audit.frames llm >> "$LOG" 2>&1); }
gpu0() {
  until grep -q "granite3_8b exit" "$ROOT/logs/paper/E34_generate.log" 2>/dev/null; do sleep 120; done
  frames; say "frames built (7-8B models)"
  GPU=0 GPU_UTIL=0.5 bash "$ROOT/code/launch/e34_zs.sh" llm e34zs_taxonomy; say "ZS on 7-8B answers done"
  DL=$ROOT/logs/model_download_allenai_OLMo-2-0325-32B-Instruct.log
  until grep -q COMPLETE "$DL" 2>/dev/null; do sleep 300; done
  GPU=0 GPU_UTIL=0.5 QUANT=fp8 MAXLEN=4096 bash "$ROOT/code/launch/e34_generate.sh" "olmo2_32b=allenai/OLMo-2-0325-32B-Instruct"
  say "32B answers done"
  frames; say "frames rebuilt (with 32B)"
  GPU=0 GPU_UTIL=0.5 bash "$ROOT/code/launch/e34_zs.sh" llm e34zs_taxonomy; say "ZS on all answers done"
  touch "$ROOT/runs/E34/.gpu0_done"
}
cf_and_analyse() {  # $1: tag for the log
  say "CF on all answers start ($1)"
  (cd "$ROOT/code" && PYTHONPATH=/scratch/shlok/Temp/.pylib-peft HF_HUB_OFFLINE=1 \
     CUDA_VISIBLE_DEVICES=MIG-def14cbb-599b-58f4-9cef-4befe9ff8506 "$VPY" -m audit.infer_cf --batch 16 --frames llm >> "$LOG" 2>&1) \
     || { say "CF FAILED ($1)"; return 1; }
  say "CF done ($1); running analyses"
  if grep -q FAILED "$ROOT/docs/raw/E33_checks.txt" || [ "$(grep -c PASSED "$ROOT/docs/raw/E33_checks.txt")" != 3 ]; then
    if [ ! -f "$ROOT/runs/E34/primary_instrument.json" ]; then say "C33 checks not all PASSED: E34 analysis NOT run"; return 1; fi
  fi
  (cd "$ROOT/code" && "$CPY" -m preregistered_analyses.e34_analysis train >> "$LOG" 2>&1 && \
     "$CPY" -m preregistered_analyses.e35_binding >> "$LOG" 2>&1 && "$CPY" -m preregistered_analyses.paper_facts >> "$LOG" 2>&1)
  say "analyses done ($1)"
}
mig() {
  until grep -q "E33 post done" "$ROOT/logs/paper/E33_post.log" 2>/dev/null && grep -q "ZS on 7-8B answers done" "$LOG"; do sleep 300; done
  cf_and_analyse "trio + granite"
  until [ -f "$ROOT/runs/E34/.gpu0_done" ]; do sleep 300; done
  cf_and_analyse "all models"
  say "E34 queue done"
}
gpu0 & mig & wait
