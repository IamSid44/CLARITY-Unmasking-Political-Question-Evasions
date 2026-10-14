#!/usr/bin/env bash
# Resumable download of one HF model into the shared HF cache, retried until complete.
#   bash code/launch/download_model.sh Qwen/Qwen3.8-27B-FP8      (log: logs/model_download_<org>_<name>.log)
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
PY=${PY:-/scratch/shlok/Temp/.venv/bin/python}
M=$1; LOG=$ROOT/logs/model_download_$(echo "$M" | tr / _).log
cd "$ROOT" || exit 1
for try in $(seq 1 50); do
  echo "$(date '+%F %T') try $try" >> "$LOG"
  HF_HUB_DISABLE_XET=1 "$PY" - "$M" >> "$LOG" 2>&1 <<'PY' && { echo "$(date '+%F %T') COMPLETE" >> "$LOG"; exit 0; }
import sys
from dotenv import dotenv_values
from huggingface_hub import snapshot_download
p = snapshot_download(sys.argv[1], token=dotenv_values("code/.env").get("HF_TOKEN"), max_workers=8,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"])
print("complete:", p, flush=True)
PY
  sleep 120
done
exit 1
