#!/usr/bin/env bash
# Fill GPU 0's idle gap with ZS on the current E34 frame until the given download completes (the generation lane then
# needs GPU 0). Frames are rebuilt first (append-only answers keep earlier row indices); adjudicate skips scored rows.
#   bash code/launch/e34_zs_gap2.sh logs/model_download_<...>.log
ROOT=/scratch/shlok/Temp/CLARITY-Unmasking-Political-Question-Evasions
STOP=$ROOT/$1
grep -q COMPLETE "$STOP" && exit 0
(cd "$ROOT/code" && CLARITY_E25_DIR=$ROOT/runs/E34 /scratch/shlok/Temp/.venv-clarity-cpu/bin/python -m audit.frames llm >> "$ROOT/logs/paper/E34_queue.log" 2>&1)
GPU=0 GPU_UTIL=0.5 setsid bash "$ROOT/code/launch/e34_zs.sh" llm e34zs_taxonomy &
PG=$!
until grep -q COMPLETE "$STOP"; do
  kill -0 $PG 2>/dev/null || exit 0
  sleep 5
done
kill -TERM -- -$PG 2>/dev/null; sleep 20; kill -KILL -- -$PG 2>/dev/null
echo "$(date '+%F %T') ZS gap run stopped ($1 complete)" >> "$ROOT/logs/paper/E34_zs.log"
