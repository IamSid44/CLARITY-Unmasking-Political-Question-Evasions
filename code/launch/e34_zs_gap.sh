#!/usr/bin/env bash
# Use GPU 0's idle gap: after the E35 ZS run, score E34 rows with the 27B until Qwen3-8B's download completes (the
# generation lane then needs GPU 0). adjudicate writes records per chunk and skips scored rows, so stopping is safe.
ROOT=/scratch/shlok/Temp/CLARITY-Unmasking-Political-Question-Evasions
until grep -q "ZS train exit" "$ROOT/logs/paper/E34_zs.log" 2>/dev/null; do sleep 20; done
grep -q COMPLETE "$ROOT/logs/model_download_Qwen_Qwen3-8B.log" && exit 0
GPU=0 GPU_UTIL=0.5 setsid bash "$ROOT/code/launch/e34_zs.sh" llm e34zs_taxonomy &
PG=$!
until grep -q COMPLETE "$ROOT/logs/model_download_Qwen_Qwen3-8B.log"; do
  kill -0 $PG 2>/dev/null || exit 0
  sleep 5
done
kill -TERM -- -$PG 2>/dev/null; sleep 20; kill -KILL -- -$PG 2>/dev/null
echo "$(date '+%F %T') ZS gap run stopped for Qwen generation" >> "$ROOT/logs/paper/E34_zs.log"
