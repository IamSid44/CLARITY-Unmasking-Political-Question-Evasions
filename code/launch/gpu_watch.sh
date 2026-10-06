#!/usr/bin/env bash
# Run an experiment on the shared lab server and use spare GPU capacity politely, from tmux.
#
#   bash code/launch/gpu_watch.sh E14 start     # start the watcher in tmux session clarity-E14-watch
#   bash code/launch/gpu_watch.sh E14 status    # what is running where, and who else is on the slice
#   bash code/launch/gpu_watch.sh E14 release   # give the slice back NOW and keep off it
#   bash code/launch/gpu_watch.sh E14 allow     # let the watcher use the slice again
#   tail -f logs/E14_watch.log
#
# What the watcher does (`run`, inside tmux):
#   1. makes sure the base model is in the HF cache (resumes a partial download);
#   2. starts the experiment's normal lanes once (start.sh -> tmux session clarity-<EXP>);
#   3. every 10 min (every 2 min while it holds the slice) looks at each lane marked "# start: watcher"
#      (a MIG slice: "# device:" + "# gi:"):
#        - starts it when no other user has a process on that slice and >= SLICE_START_FREE_GB are free;
#        - stops it as soon as another user's process appears there (contention), or on `release`.
#      A stopped run loses at most its current epoch: it resumes from resume.pt on whichever lane
#      reaches it next (lanes re-check their lists every 10 min), and keeps its W&B run.
#   4. exits when every run of the experiment has finished.
# The normal lanes are never stopped by the watcher; to free GPU 0, kill its window
# (`tmux kill-window -t =clarity-E14:lane_gpu0_a`) -- its run resumes later from its last epoch.
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
CODE="$(dirname "$DIR")"
ROOT="$(dirname "$CODE")"
EXP="${1:-}"; CMD="${2:-run}"
[ -n "$EXP" ] || { sed -n '2,8p' "$0"; exit 2; }
EXPDIR="$CODE/experiments/$EXP"
[ -d "$EXPDIR" ] || { echo "no experiment folder $EXPDIR"; exit 2; }
PY="${CLARITY_PY:-/scratch/shlok/Temp/.venv/bin/python}"
LOGS="$ROOT/logs"; mkdir -p "$LOGS"
LOG="$LOGS/${EXP}_watch.log"
SESSION="clarity-$EXP"
NO_SLICE="$LOGS/.${EXP}_no_slice"
START_FREE_GB="${SLICE_START_FREE_GB:-30}"
ME="$(id -un)"
RUNS="${CLARITY_RUNS:-$ROOT/runs}"
log() { echo "[$(date '+%F %T')] [watch] $*" | tee -a "$LOG"; }

watcher_lanes() { grep -lE '^#[[:space:]]*start:[[:space:]]*watcher' "$EXPDIR"/lane_gpu*_*.txt 2>/dev/null; }
hdr() { sed -nE "s/^#[[:space:]]*$2:[[:space:]]*([^[:space:]]+).*/\1/p" "$1" | head -1; }
gpu_of() { basename "$1" | sed -nE 's/^lane_gpu([0-9]+)_.*\.txt$/\1/p'; }
lane_running() { pgrep -f "^bash .*/run_queue\.sh $1" >/dev/null; }

slice_free_gb() {  # gpu gi
  nvidia-smi 2>/dev/null | awk -v g="$1" -v gi="$2" '$1=="|" && $2==g && $3==gi && $6=="|" && $7 ~ /MiB$/ {
    u=$7; t=$9; sub("MiB","",u); sub("MiB","",t); printf "%d", (t-u)/1024; exit }'
}
slice_procs() {  # gpu gi -> "pid user" per compute process on that slice
  nvidia-smi 2>/dev/null | awk -v g="$1" -v gi="$2" '$1=="|" && $2==g && $3==gi && $5 ~ /^[0-9]+$/ && $6 ~ /^(C|G|C\+G)$/ {print $5}' |
    while read -r p; do echo "$p $(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')"; done
}
foreign_on() { slice_procs "$1" "$2" | awk -v me="$ME" '$2!=me {printf "%s(pid %s) ", $2, $1}'; }

all_done() {  # every run in every lane file finished (or marked failed)
  local name seed rest d
  while read -r name seed rest; do
    [ "$name" = "@after" ] && continue
    case "$seed" in fold*) d="$RUNS/$name/$seed" ;; *) d="$RUNS/$name/seed$seed" ;; esac
    [ -f "$d/metrics.json" ] || [ -f "$d/.failed" ] || return 1
  done < <(cat "$EXPDIR"/lane_gpu*_*.txt | grep -vE '^\s*(#|$)')
  return 0
}

start_lane() {  # lane file
  local win; win="$(basename "$1" .txt)"
  local v envp="CLARITY_PY=$PY "
  for v in CLARITY_RUNS WANDB_MODE HF_HOME QUEUE_PASS_SLEEP; do [ -n "${!v:-}" ] && envp+="$v=${!v} "; done
  local cmd="env ${envp}bash $DIR/run_queue.sh $1; echo LANE_EXITED; exec bash"
  tmux kill-window -t "=$SESSION:$win" 2>/dev/null
  if tmux has-session -t "=$SESSION" 2>/dev/null; then tmux new-window -d -t "=$SESSION:" -n "$win" "$cmd"
  else tmux new-session -d -s "$SESSION" -n "$win" "$cmd"; fi
}

stop_lane() {  # lane file, reason
  local lane="$1" gpu gi win p
  gpu="$(gpu_of "$lane")"; gi="$(hdr "$lane" gi)"; win="$(basename "$lane" .txt)"
  log "stop  $win: $2"
  tmux kill-window -t "=$SESSION:$win" 2>/dev/null
  pkill -f "^bash .*/run_queue\.sh $lane" 2>/dev/null
  for p in $(slice_procs "$gpu" "$gi" | awk -v me="$ME" '$2==me {print $1}'); do kill -TERM "$p" 2>/dev/null; done
  sleep 20
  for p in $(slice_procs "$gpu" "$gi" | awk -v me="$ME" '$2==me {print $1}'); do kill -KILL "$p" 2>/dev/null; done
  log "stop  $win: done; its run resumes from its last finished epoch on the next free lane"
}

ensure_model() {
  local model; model="$(grep -oE -- '--model[[:space:]]+[^[:space:]]+' "$EXPDIR/common.args" | awk '{print $2}')"
  model="${model:-Qwen/Qwen3-8B-Base}"
  [ "$(grep -c '^@module models.llm_classifier' "$EXPDIR/common.args")" = 0 ] && return 0
  for try in $(seq 1 50); do
    log "model $model: checking the HF cache (resumes a partial download; ~2-8 MB/s here)"
    if (cd "$ROOT" && HF_HUB_DISABLE_XET=1 "$PY" - "$model" >> "$LOGS/model_download.log" 2>&1 <<'EOF'
import sys
from dotenv import dotenv_values
from huggingface_hub import snapshot_download
p = snapshot_download(sys.argv[1], token=dotenv_values("code/.env").get("HF_TOKEN"), max_workers=8,
                      allow_patterns=["*.json", "*.safetensors", "*.txt"])
print("complete:", p, flush=True)
EOF
    ); then log "model $model: complete"; return 0; fi
    log "model download failed (try $try, logs/model_download.log); retrying in 2 min"; sleep 120
  done
  log "model download kept failing -- giving up"; exit 1
}

status() {
  local lane gpu gi
  echo "lanes of $EXP:"; for lane in "$EXPDIR"/lane_gpu*_*.txt; do
    echo "  $(basename "$lane" .txt): $(lane_running "$lane" && echo running || echo stopped)"; done
  for lane in $(watcher_lanes); do
    gpu="$(gpu_of "$lane")"; gi="$(hdr "$lane" gi)"
    echo "slice GPU $gpu GI $gi: $(slice_free_gb "$gpu" "$gi") GB free; other users: $(foreign_on "$gpu" "$gi")"
  done
  echo "GPU 0: $(nvidia-smi -i 0 --query-gpu=memory.free,utilization.gpu --format=csv,noheader)"
  [ -f "$NO_SLICE" ] && echo "slice use is RELEASED (allow with: $0 $EXP allow)"
  echo "finished runs: $(ls "$RUNS"/*/seed*/metrics.json "$RUNS"/*/fold*/metrics.json 2>/dev/null | grep -cE "$(cat "$EXPDIR"/lane_gpu*_*.txt | grep -vE '^\s*(#|$)' | awk '{print $1}' | sort -u | paste -sd'|')")"
}

case "$CMD" in
  start)
    if pgrep -f "^bash $DIR/gpu_watch\.sh $EXP run" >/dev/null; then echo "watcher already running: tmux attach -t $SESSION-watch"; exit 0; fi
    tmux kill-session -t "=$SESSION-watch" 2>/dev/null   # "=": exact session names throughout
    tmux new-session -d -s "$SESSION-watch" "env ${HF_HOME:+HF_HOME=$HF_HOME }bash $DIR/gpu_watch.sh $EXP run; echo WATCH_EXITED; exec bash"
    echo "watcher started: tmux attach -t $SESSION-watch   |   tail -f logs/${EXP}_watch.log"; exit 0 ;;
  status) status; exit 0 ;;
  release)
    touch "$NO_SLICE"
    for lane in $(watcher_lanes); do lane_running "$lane" && stop_lane "$lane" "released by user"; done
    echo "slice released; the watcher keeps off it until: $0 $EXP allow"; exit 0 ;;
  allow) rm -f "$NO_SLICE"; echo "slice use allowed again"; exit 0 ;;
  run) ;;
  *) sed -n '2,8p' "$0"; exit 2 ;;
esac

log "watcher start ($EXP; slice lanes: $(watcher_lanes | xargs -r -n1 basename | tr '\n' ' '))"
ensure_model
if all_done; then log "every run already finished"; exit 0; fi
if ! pgrep -f "^bash .*/run_queue\.sh .*/experiments/$EXP/" >/dev/null; then
  log "start $EXP normal lanes"; bash "$DIR/start.sh" "$EXP" | tee -a "$LOG"
fi
last_hourly=0
while :; do
  holding=0
  for lane in $(watcher_lanes); do
    gpu="$(gpu_of "$lane")"; gi="$(hdr "$lane" gi)"; win="$(basename "$lane" .txt)"
    others="$(foreign_on "$gpu" "$gi")"; free="$(slice_free_gb "$gpu" "$gi")"
    if lane_running "$lane"; then
      if [ -f "$NO_SLICE" ]; then stop_lane "$lane" "released by user"
      elif [ -n "$others" ]; then stop_lane "$lane" "contention: ${others}on GPU $gpu slice $gi"
      else holding=1; fi
    elif [ ! -f "$NO_SLICE" ] && [ -z "$others" ] && [ -n "$free" ] && [ "$free" -ge "$START_FREE_GB" ] && ! all_done; then
      log "start $win: GPU $gpu slice $gi is free (${free} GB, no other users)"; start_lane "$lane"; holding=1
    fi
  done
  if all_done && ! pgrep -f "^bash .*/run_queue\.sh .*/experiments/$EXP/" >/dev/null; then
    log "every run of $EXP finished; watcher exits"; exit 0
  fi
  now=$(date +%s)
  if [ $((now - last_hourly)) -ge 3600 ]; then
    last_hourly=$now
    msg="GPU 0 free/util: $(nvidia-smi -i 0 --query-gpu=memory.free,utilization.gpu --format=csv,noheader)"
    for lane in $(watcher_lanes); do
      gpu="$(gpu_of "$lane")"; gi="$(hdr "$lane" gi)"; others="$(foreign_on "$gpu" "$gi")"
      msg+="; $(basename "$lane" .txt) $(lane_running "$lane" && echo running || echo idle), slice $(slice_free_gb "$gpu" "$gi") GB free, other users: ${others:-none}"
    done
    log "hourly: $msg"
  fi
  sleep $([ $holding = 1 ] && echo "${WATCH_HOLD_INTERVAL:-120}" || echo "${WATCH_INTERVAL:-600}")
done
