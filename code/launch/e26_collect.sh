#!/usr/bin/env bash
# E26 (2026-10-30): runs on the laptop, unattended. Waits for the PASS=4 lane (FP8 vs 16-bit on an RTX PRO 6000),
# copies its outputs from the Hub, makes sure the instance is DESTROYED, then labels locally and runs the analysis.
#   MID=<machine id> GIVE_UP_IST=HH:MM JL_PY=<python with jarvislabs> nohup bash code/launch/e26_collect.sh \
#       > logs/jarvis/E26/collect.log 2>&1 &
set -u
REPO=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
MID=${MID:?set MID to the machine id}
JL_PY=${JL_PY:?set JL_PY to a Python with the jarvislabs SDK}
GIVE_UP_IST=${GIVE_UP_IST:?set GIVE_UP_IST (HH:MM, IST)}
cd "$REPO/code"
set -a; . ~/.config/clarity_jarvis/keys.env; set +a
export PYTHONIOENCODING=utf-8
note() { echo "$(date '+%F %T') $*"; }

hub_log() {  # the lane's heartbeat log; only the part after THIS lane's start (the file still ends with E25's run)
  python - <<'EOF' 2>/dev/null | sed -n "/${MARK:-PASS=4}/,\$p"  # MARK=PASS=5 for the NF4 lane
import os
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
load_dotenv(".env")
p = hf_hub_download(os.environ["HF_REPO_ID"], "logs/infer_lane/infer.log", token=os.environ["HF_TOKEN"],
                    force_download=True, local_dir=os.environ.get("TEMP", "/tmp") + "/e26_hub")
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
sleep 60

note "collect: downloading E26 outputs from the Hub"
python - <<'EOF'
import os, shutil
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
load_dotenv(".env")
p = snapshot_download(os.environ["HF_REPO_ID"], local_dir="../runs", token=os.environ["HF_TOKEN"],
                      allow_patterns=["E26/*", "E26/**", "M3/e26_*/*", "logs/infer_lane_e26/*"])
os.makedirs("../logs/jarvis/E26", exist_ok=True)
src = "../runs/logs/infer_lane_e26"
for f in (os.listdir(src) if os.path.isdir(src) else []):
    shutil.copy(os.path.join(src, f), "../logs/jarvis/E26/")
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

note "collect: labels and analysis"
python -m audit.quant_system label --site rtxpro > ../logs/jarvis/E26/label_local.log 2>&1 && note "collect: labelled" \
  || note "collect: label FAILED (logs/jarvis/E26/label_local.log)"
python -m preregistered_analyses.e26_analysis > ../docs/raw/E26_analysis.txt 2> ../logs/jarvis/E26/analysis.err \
  && note "collect: wrote docs/raw/E26_analysis.txt" || note "collect: analysis FAILED (logs/jarvis/E26/analysis.err)"
note "collect: done"
