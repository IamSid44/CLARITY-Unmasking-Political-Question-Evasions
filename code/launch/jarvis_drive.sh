#!/usr/bin/env bash
set -uo pipefail

REPO=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
CFG=$HOME/.config/clarity_jarvis
KEYS=$CFG/keys.env
PROFILE=${JPROFILE:-}
HOSTCFG=$CFG/host${PROFILE:+_$PROFILE}.env
GPU_TYPE=${GPU_TYPE:-RTX-PRO6000}
SSHKEY=$HOME/.ssh/jarvis_ed25519
NAME=${RUN_NAME:-Q8_fullq_lora}
SEEDS=${SEEDS:-"0 1 2"}
EXTRA_ARGS=${EXTRA_ARGS:-}
MAX_TOTAL_HOURS=${MAX_TOTAL_HOURS:-5}
INTERVAL=${INTERVAL:-300}
RBASE=${RBASE:-/home/clarity_llm}
LOCAL_LOGS=$REPO/logs/jarvis${PROFILE:+/$PROFILE}
WATCH=jarvis-watch${PROFILE:+-$PROFILE}
LOCAL_RUNS=$REPO/runs
UVX=${UVX:-$(command -v uvx || echo "$HOME/.local/bin/uvx")}

mkdir -p "$CFG" "$LOCAL_LOGS"; chmod 700 "$CFG"
need_uv() {
  [ -x "$UVX" ] && return 0
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null && UVX=$HOME/.local/bin/uvx
  [ -x "$UVX" ] || { echo "could not install uv (needed for the JarvisLabs SDK)"; exit 1; }
}
note() { echo "$(date '+%F %T') $*" | tee -a "$LOCAL_LOGS/drive.log"; }
load() { [ -f "$KEYS" ] && set -a && . "$KEYS" && set +a; [ -f "$HOSTCFG" ] && . "$HOSTCFG"; return 0; }
R() { ssh -i "$SSHKEY" -p "$JPORT" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 \
          -o ConnectTimeout=20 -o BatchMode=yes "$JUSER@$JHOST" "$@"; }

need_host() { load; [ -n "${JHOST:-}" ] || { echo "run: bash code/launch/jarvis_drive.sh connect \"<ssh command>\" <machine_id>"; exit 2; }; }

pause_now() {
  load; need_uv
  [ -n "${JL_API_KEY:-}" ] && [ -n "${JL_MACHINE_ID:-}" ] || { note "cannot pause: no key or machine id"; return 1; }
  JL_API_KEY=$JL_API_KEY "$UVX" --quiet --from jarvislabs python -c \
    "from jarvislabs import Client; print('pause:', Client().instances.pause($JL_MACHINE_ID))" 2>&1 | tee -a "$LOCAL_LOGS/drive.log"
}

sync_once() {
  local stage; stage=$(mktemp -d)
  if R "cd $RBASE && mkdir -p runs logs && tar cf - --exclude=resume.pt --exclude=resume.tmp --exclude=adapter_best \
          --exclude='ep*_adapter' --exclude=wandb logs runs \$(ls results_*.tgz 2>/dev/null)" 2>/dev/null \
       | tar xf - -C "$stage" 2>/dev/null; then
    [ -d "$stage/logs" ] || { rm -rf "$stage"; return 1; }
    cp -r "$stage/logs/." "$LOCAL_LOGS/"
    [ -d "$stage/runs" ] && mkdir -p "$LOCAL_RUNS" && cp -r "$stage/runs/." "$LOCAL_RUNS/"
    ls "$stage"/*.tgz >/dev/null 2>&1 && cp "$stage"/*.tgz "$LOCAL_LOGS/"
    rm -rf "$stage"; return 0
  fi
  rm -rf "$stage"; return 1
}

case "${1:-}" in
keygen)
  [ -f "$SSHKEY" ] || ssh-keygen -t ed25519 -N "" -C "clarity-to-jarvis-$(hostname)" -f "$SSHKEY" -q
  echo "Add this public key in JarvisLabs -> SSH keys, BEFORE creating the instance:"; cat "$SSHKEY.pub"
  ;;

keys)
  need_uv
  read -rsp "JarvisLabs API key (jarvislabs.ai/settings/api-keys; input hidden): " k; echo
  umask 077; touch "$KEYS"; grep -v '^JL_API_KEY=' "$KEYS" > "$KEYS.tmp" 2>/dev/null; mv "$KEYS.tmp" "$KEYS"
  echo "JL_API_KEY=$k" >> "$KEYS"; chmod 600 "$KEYS"
  JL_API_KEY=$k "$UVX" --quiet --from jarvislabs python -c \
    "from jarvislabs import Client; print('key OK; instances:', [(i.machine_id, getattr(i,'status','?')) for i in Client().instances.list()])" \
    || echo "key check failed -- re-run: bash code/launch/jarvis_drive.sh keys"
  ;;

connect)
  cmd=${2:-}; mid=${3:-}
  [ -n "$cmd" ] && [ -n "$mid" ] || { echo 'usage: bash code/launch/jarvis_drive.sh connect "ssh -p PORT USER@HOST" MACHINE_ID'; exit 2; }
  port=$(echo "$cmd" | sed -nE 's/.*-p[ ]*([0-9]+).*/\1/p'); port=${port:-22}
  dest=$(echo "$cmd" | grep -oE '[A-Za-z0-9._-]+@[A-Za-z0-9._-]+' | head -1)
  [ -n "$dest" ] || { echo "could not find user@host in: $cmd"; exit 2; }
  printf 'JUSER=%s\nJHOST=%s\nJPORT=%s\nJL_MACHINE_ID=%s\n' "${dest%@*}" "${dest#*@}" "$port" "$mid" > "$HOSTCFG"
  chmod 600 "$HOSTCFG"
  load
  note "connect: ${PROFILE:-default} -> $JUSER@$JHOST port $JPORT, machine $mid"
  ssh-keygen -R "$JHOST" >/dev/null 2>&1
  for i in $(seq 1 18); do R true 2>/dev/null && break; sleep 10; done
  R 'echo "connected to $(hostname)"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; df -h /home | tail -1' \
    || { echo "SSH failed. Is $SSHKEY.pub (bash code/launch/jarvis_drive.sh keygen) added to JarvisLabs, and was the instance created after adding it?"; exit 1; }
  ;;

create|resume)
  load; need_uv
  [ -n "${JL_API_KEY:-}" ] || { echo "run: bash code/launch/jarvis_drive.sh keys"; exit 2; }
  [ "$1" = resume ] && [ -z "${JL_MACHINE_ID:-}" ] && { echo "no machine id for profile ${PROFILE:-default}"; exit 2; }
  out=$(JL_API_KEY=$JL_API_KEY ACTION=$1 MID=${JL_MACHINE_ID:-0} GPU=$GPU_TYPE LABEL="clarity-${PROFILE:-a}" \
    "$UVX" --quiet --from jarvislabs python -c '
import os
from jarvislabs import Client
c = Client()
if os.environ["ACTION"] == "create":
    i = c.instances.create(gpu_type=os.environ["GPU"], template="vm", storage=100, name=os.environ["LABEL"])
else:
    i = c.instances.resume(int(os.environ["MID"]))
print("RESULT", i.machine_id, i.status, "|", i.ssh_command)') || { note "$1 failed: $out"; exit 1; }
  line=$(echo "$out" | grep '^RESULT') || { note "$1: unexpected reply: $out"; exit 1; }
  mid=$(echo "$line" | awk '{print $2}'); sshcmd=${line#*| }
  note "$1: machine $mid ($(echo "$line" | awk '{print $3}')) -- $sshcmd"
  bash "$0" connect "$sshcmd" "$mid"
  ;;

push)
  need_host
  (cd "$REPO" && bash code/launch/jarvis_setup.sh bundle >/dev/null) || { note "bundle failed"; exit 1; }
  R "mkdir -p $RBASE/code 2>/dev/null || { sudo mkdir -p $RBASE && sudo chown \$(id -u):\$(id -g) $RBASE && mkdir -p $RBASE/code; }" \
    && scp -i "$SSHKEY" -P "$JPORT" -o StrictHostKeyChecking=accept-new -q "$REPO/jarvis_bundle.tgz" "$JUSER@$JHOST:$RBASE/jarvis_bundle.tgz" \
    && R "tar xzf $RBASE/jarvis_bundle.tgz -C $RBASE/code && ls $RBASE/code/code" \
    && note "push: bundle unpacked into $RBASE/code" || { note "push failed"; exit 1; }
  if [ -f "$REPO/code/.env" ]; then
    R "umask 077; cat > $RBASE/code/code/.env" < "$REPO/code/.env" && note "push: code/.env copied (mode 600)"
  fi
  ;;

start)
  need_host
  R "umask 077; printf 'JL_API_KEY=%s\nJL_MACHINE_ID=%s\n' '${JL_API_KEY:-}' '${JL_MACHINE_ID:-}' > $RBASE/keys.env"
  R "NAME='$NAME' SEEDS='$SEEDS' EXTRA_ARGS='$EXTRA_ARGS' MAX_TOTAL_HOURS=$MAX_TOTAL_HOURS AUTO_PAUSE=1 PAUSE_DELAY=1800 NO_PROMPT=1 \
     bash $RBASE/code/code/launch/jarvis_setup.sh all" && note "start: pipeline launched on the instance (tmux clarity-llm)"
  ;;

watch)
  need_host
  if [ "${2:-}" != _inner ]; then
    tmux has-session -t "$WATCH" 2>/dev/null && { echo "already watching: tmux attach -t $WATCH"; exit 0; }
    tmux new-session -d -s "$WATCH" "JPROFILE='$PROFILE' RUN_NAME='$NAME' bash '$REPO/code/launch/jarvis_drive.sh' watch _inner; exec bash"
    echo "watching in local tmux session '$WATCH' (tail -f $LOCAL_LOGS/drive.log)"; exit 0
  fi
  note "watch: every $INTERVAL s"
  fails=0
  while true; do
    if sync_once; then
      fails=0
      last=$(tail -n 1 "$LOCAL_LOGS/pipeline.log" 2>/dev/null)
      note "watch: ${last:-no pipeline log yet}"
      case "$last" in
        *"ALL DONE"*|*"pausing instance"*"finished"*)
          sync_once final && note "watch: final results copied to runs/ and $LOCAL_LOGS"
          pause_now; note "watch: finished"; break ;;
        *FAILED*|*"pausing instance"*"failure"*)
          sync_once final; note "watch: the pipeline FAILED -- see $LOCAL_LOGS"; pause_now; break ;;
      esac
    else
      fails=$((fails + 1)); note "watch: instance unreachable ($fails)"
      [ "$fails" -ge 12 ] && { note "watch: unreachable for an hour (paused already?) -- stopping"; break; }
    fi
    sleep "$INTERVAL"
  done
  ;;

all)  bash "$0" push && bash "$0" start && bash "$0" watch ;;
sync) need_host; sync_once final && note "sync: copied" ;;
status) need_host; R "bash $RBASE/code/code/launch/jarvis_setup.sh status" ;;
pause)  pause_now ;;
ssh)    need_host; ssh -i "$SSHKEY" -p "$JPORT" "$JUSER@$JHOST" ;;
*) echo "usage: bash code/launch/jarvis_drive.sh {keygen|keys|connect \"<ssh command>\" <machine_id>|create|resume|push|start|watch|all|sync|status|pause|ssh}"; exit 2 ;;
esac
