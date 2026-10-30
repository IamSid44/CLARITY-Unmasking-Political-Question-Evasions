#!/usr/bin/env bash
# E25 (2026-10-30): runs on the laptop, unattended. Waits for the PASS=3 lane to finish, copies every E25 output
# from the Hub into this repo, makes sure the JarvisLabs instance is DESTROYED, then runs the analysis and figures.
#   nohup bash code/launch/e25_collect.sh > logs/jarvis/E25/collect.log 2>&1 &
# Needs: code/.env (HF_TOKEN, HF_REPO_ID), ~/.config/clarity_jarvis/keys.env (JL_API_KEY), JL_PY (jarvislabs SDK).
set -u
REPO=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
MID=${MID:-554906}
JL_PY=${JL_PY:?set JL_PY to a Python with the jarvislabs SDK}
GIVE_UP_IST=${GIVE_UP_IST:-"17:15"}  # after this (IST), collect whatever is on the Hub and destroy anyway
cd "$REPO/code"
set -a; . ~/.config/clarity_jarvis/keys.env; set +a
export PYTHONIOENCODING=utf-8
note() { echo "$(date '+%F %T') $*"; }

hub_log() {  # the lane's own log, as heartbeated to the Hub
  python - <<'EOF' 2>/dev/null
import os
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
load_dotenv(".env")
p = hf_hub_download(os.environ["HF_REPO_ID"], "logs/infer_lane/infer.log", token=os.environ["HF_TOKEN"],
                    force_download=True, local_dir=os.environ.get("TEMP", "/tmp") + "/e25_hub")
print(open(p, encoding="utf-8", errors="replace").read())
EOF
}

instance() { MID=$MID "$JL_PY" -c "
import os
from jarvislabs import Client
ids = {i.machine_id: i.status for i in Client().instances.list()}
print(ids.get(int(os.environ['MID']), 'gone'))" 2>/dev/null | tr -d '\r'; }

note "collect: watching machine $MID (give up at $GIVE_UP_IST IST)"
while true; do
  st=$(instance)
  log=$(hub_log)
  if echo "$log" | grep -qE 'finish: .*(destroying|pausing)'; then note "collect: the lane finished ($(echo "$log" | grep -E 'finish:' | tail -1))"; break; fi
  [ "$st" = gone ] && { note "collect: instance already gone"; break; }
  [ "$(TZ=Asia/Kolkata date +%H:%M)" \> "$GIVE_UP_IST" ] && { note "collect: past $GIVE_UP_IST IST, collecting what is there"; break; }
  note "collect: instance $st; last lane line: $(echo "$log" | tail -1 | cut -c1-140)"
  sleep 120
done
sleep 60  # the lane's last upload and stop call

note "collect: downloading E25 outputs from the Hub"
python - <<'EOF'
import os, shutil
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
load_dotenv(".env")
p = snapshot_download(os.environ["HF_REPO_ID"], local_dir="../runs", token=os.environ["HF_TOKEN"],
                      allow_patterns=["E25/*", "E25/**", "M3/e25_*/*", "logs/infer_lane/*"])
os.makedirs("../logs/jarvis/E25", exist_ok=True)
for f in os.listdir("../runs/logs/infer_lane"):
    shutil.copy(os.path.join("../runs/logs/infer_lane", f), "../logs/jarvis/E25/")
print("downloaded into", p)
EOF

for i in 1 2 3; do
  st=$(instance)
  [ "$st" = gone ] && break
  note "collect: instance is $st -> destroying"
  MID=$MID "$JL_PY" -c "
import os
from jarvislabs import Client
print(Client().instances.destroy(int(os.environ['MID'])))" 2>&1 | tail -1
  sleep 30
done
note "collect: instance now $(instance); balance: $("$JL_PY" -c "from jarvislabs import Client; print(Client().account.balance())" 2>/dev/null | tr -d '\r')"

note "collect: analysis"
python -m preregistered_analyses.e25_analysis > ../docs/raw/E25_analysis.txt 2> ../logs/jarvis/E25/analysis.err \
  && note "collect: wrote docs/raw/E25_analysis.txt" || note "collect: analysis FAILED (logs/jarvis/E25/analysis.err)"
python -m figures.e25_figures 2>> ../logs/jarvis/E25/analysis.err && note "collect: figures written" || note "collect: figures FAILED"
python -m preregistered_analyses.e25_analysis sheet 2>> ../logs/jarvis/E25/analysis.err && note "collect: annotation sheet written" \
  || note "collect: sheet FAILED"
note "collect: done"
